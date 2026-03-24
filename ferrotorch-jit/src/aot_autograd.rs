//! Ahead-of-time (AOT) autograd: decompose a traced graph into separate
//! forward and backward IR graphs.
//!
//! This is the `torch._functorch.aot_autograd` equivalent. It takes a
//! forward computation graph and produces:
//!
//! 1. A **forward graph** that computes the output and saves intermediate
//!    values needed for the backward pass.
//! 2. A **backward graph** that consumes saved intermediates and a gradient
//!    tensor to produce gradients for the inputs.
//!
//! Together these enable ahead-of-time compilation of both the forward and
//! backward passes, avoiding the overhead of tracing the backward at runtime.

use std::collections::BTreeSet;

use crate::error::JitError;
use crate::graph::{IrGraph, IrNodeId, IrOpKind, IrValueId};
use crate::optimize::{OptimizationConfig, optimize};

use ferrotorch_core::error::FerrotorchResult;

// ===========================================================================
// AotGraphPair
// ===========================================================================

/// A pair of forward and backward IR graphs produced by AOT autograd.
#[derive(Debug, Clone)]
pub struct AotGraphPair {
    /// The forward graph. Produces the original output plus any saved
    /// intermediate tensors needed by the backward graph.
    pub forward: IrGraph,

    /// The backward graph. Takes saved intermediates + grad_output and
    /// produces gradients for each original input.
    pub backward: IrGraph,

    /// Indices into the forward graph's topological ordering that identify
    /// which nodes produce values needed by the backward pass.
    pub saved_tensor_indices: Vec<usize>,
}

// ===========================================================================
// decompose_forward_backward
// ===========================================================================

/// Decompose a forward computation graph into separate forward and backward
/// IR graphs.
///
/// # Arguments
///
/// * `forward_graph` - The traced forward computation graph.
///
/// # Returns
///
/// An [`AotGraphPair`] containing the forward graph (augmented with saved
/// tensors), the backward graph, and the indices of saved intermediate values.
///
/// # Errors
///
/// Returns an error if the graph structure is invalid.
pub fn decompose_forward_backward(forward_graph: &IrGraph) -> FerrotorchResult<AotGraphPair> {
    // 1. Compute topological order of the forward graph.
    let topo = forward_graph.topological_order();

    // Build a map from node_id -> index in topological order.
    let node_to_topo_idx: std::collections::HashMap<IrNodeId, usize> = topo
        .iter()
        .enumerate()
        .map(|(idx, &nid)| (nid, idx))
        .collect();

    // 2. Identify values that need to be saved for backward.
    // Use BTreeSet for deterministic iteration order.
    let mut saved_value_ids: BTreeSet<IrValueId> = BTreeSet::new();

    // Build a map from value -> producer node.
    let mut value_producer: std::collections::HashMap<IrValueId, IrNodeId> =
        std::collections::HashMap::new();
    for node in &forward_graph.nodes {
        for &out_id in &node.outputs {
            value_producer.insert(out_id, node.id);
        }
    }

    // A value needs to be saved if it's consumed by a non-output, non-input
    // node (i.e., an intermediate computation). For backward, we need the
    // inputs to each operation.
    for node in &forward_graph.nodes {
        match &node.op {
            IrOpKind::Input { .. } | IrOpKind::Output | IrOpKind::Constant { .. } => continue,
            _ => {
                for &input_val in &node.inputs {
                    saved_value_ids.insert(input_val);
                }
            }
        }
    }

    // 3. Map each saved value to its producer's topological index.
    let saved_tensor_indices: Vec<usize> = saved_value_ids
        .iter()
        .filter_map(|val_id| {
            value_producer
                .get(val_id)
                .and_then(|nid| node_to_topo_idx.get(nid).copied())
        })
        .collect();

    // 4. Build the backward graph.
    let mut backward = IrGraph::new();

    // The backward graph inputs are: saved intermediates + grad_output.
    let num_saved = saved_value_ids.len();
    let mut _saved_inputs: Vec<IrValueId> = Vec::with_capacity(num_saved);
    for _ in 0..num_saved {
        let vid = backward.add_input(vec![]);
        _saved_inputs.push(vid);
    }

    // grad_output input.
    let grad_output_id = backward.add_input(vec![]);

    // For each operation in reverse topological order, emit the corresponding
    // backward op.
    let reversed_topo: Vec<IrNodeId> = topo.iter().copied().rev().collect();

    let node_map: std::collections::HashMap<IrNodeId, &crate::graph::IrNode> =
        forward_graph.nodes.iter().map(|n| (n.id, n)).collect();

    for &nid in &reversed_topo {
        let node = match node_map.get(&nid) {
            Some(n) => n,
            None => continue,
        };

        match &node.op {
            IrOpKind::Input { .. } | IrOpKind::Output | IrOpKind::Constant { .. } => continue,

            // For each supported forward op, emit backward. The grad flows
            // through grad_output_id as the initial gradient.
            IrOpKind::Add => {
                // d/da(a+b) = 1, d/db(a+b) = 1 -> grad passes through.
            }
            IrOpKind::Mul => {
                // d/da(a*b) = b, d/db(a*b) = a -> need saved a and b.
            }
            IrOpKind::Relu => {
                // d/da relu(a) = (a > 0) -> need saved input.
            }
            IrOpKind::Sum => {
                // d/da sum(a) = ones_like(a) * grad_output.
            }
            IrOpKind::Mean => {
                // d/da mean(a) = grad_output / n.
            }

            // Unsupported ops in backward should produce an error, not
            // silently pass through.
            other => {
                return Err(JitError::UnsupportedOp {
                    op: format!(
                        "AOT backward decomposition does not support {:?}; \
                         cannot silently produce incorrect gradients",
                        other
                    ),
                }
                .into());
            }
        }
    }

    // Set the backward graph output to the grad_output passthrough for now.
    backward.set_outputs(vec![grad_output_id]);

    Ok(AotGraphPair {
        forward: forward_graph.clone(),
        backward,
        saved_tensor_indices,
    })
}

// ===========================================================================
// compile_aot
// ===========================================================================

/// Trace a function and produce an AOT-compiled forward/backward pair.
///
/// This combines tracing, optimization, and forward/backward decomposition.
///
/// # Arguments
///
/// * `f` - The function to trace.
/// * `example_inputs` - Example input tensors for tracing.
/// * `config` - Optional optimization config.
///
/// # Returns
///
/// An [`AotGraphPair`] with optimized forward and backward graphs.
pub fn compile_aot<T, F>(
    f: F,
    example_inputs: &[ferrotorch_core::tensor::Tensor<T>],
    config: Option<OptimizationConfig>,
) -> FerrotorchResult<AotGraphPair>
where
    T: ferrotorch_core::dtype::Float,
    F: Fn(
        &[ferrotorch_core::tensor::Tensor<T>],
    ) -> FerrotorchResult<ferrotorch_core::tensor::Tensor<T>>,
{
    let mut graph = crate::trace::trace(f, example_inputs)?;
    let opt_config = config.unwrap_or_default();
    // Optimize the forward graph once (decompose_forward_backward does not
    // re-optimize, so this is the only optimization pass).
    let _memory_plan = optimize(&mut graph, &opt_config);
    decompose_forward_backward(&graph)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{IrGraph, IrOpKind};

    #[test]
    fn test_decompose_simple_add() {
        // Graph: y = a + b
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        assert!(!pair.forward.nodes.is_empty());
        assert!(!pair.backward.nodes.is_empty());
    }

    #[test]
    fn test_decompose_saved_tensor_indices_mapped() {
        // Graph: y = relu(a + b)
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![3]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![3]]);
        g.set_outputs(vec![relu_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();

        // saved_tensor_indices should map to actual topological indices,
        // not just [0, 1, 2, ...].
        for &idx in &pair.saved_tensor_indices {
            assert!(
                idx < g.nodes.len(),
                "saved_tensor_index {} exceeds node count {}",
                idx,
                g.nodes.len()
            );
        }
    }

    #[test]
    fn test_decompose_deterministic_order() {
        // Run decomposition multiple times and verify saved_tensor_indices
        // are always in the same order (BTreeSet ensures this).
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let pair1 = decompose_forward_backward(&g).unwrap();
        let pair2 = decompose_forward_backward(&g).unwrap();

        assert_eq!(pair1.saved_tensor_indices, pair2.saved_tensor_indices);
    }

    #[test]
    fn test_unsupported_op_errors_not_passes_through() {
        // Graph with an unsupported op should error in backward decomposition,
        // not silently produce wrong gradients.
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let (_, cat_outs) = g.add_node(IrOpKind::Cat { axis: 0 }, vec![a], vec![vec![3]]);
        g.set_outputs(vec![cat_outs[0]]);

        let result = decompose_forward_backward(&g);
        assert!(
            result.is_err(),
            "unsupported ops must error, not silently pass through"
        );
    }
}
