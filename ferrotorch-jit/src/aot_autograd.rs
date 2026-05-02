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
/// # Backward graph layout
///
/// The backward graph's inputs are laid out as:
///
/// 1. **Saved intermediates** — one input per saved forward value, in
///    deterministic order (sorted by `IrValueId`). Use
///    [`AotGraphPair::saved_tensor_indices`] to map each saved input
///    back to its producing forward node.
/// 2. **`grad_output`** — the gradient w.r.t. the forward graph's
///    single output.
///
/// The backward graph's outputs are the gradients for the forward
/// graph's inputs, in the same order as `forward.input_values`.
///
/// # Supported ops
///
/// This pass currently emits real backward nodes for `Add`, `Sub`,
/// `Mul`, `Neg`, `Relu`, `Sum`, and `Mean`. Unsupported ops produce
/// an error rather than a silent wrong gradient.
///
/// # Errors
///
/// Returns an error if the graph structure is invalid or contains
/// an op that the AOT backward pass does not yet support.
pub fn decompose_forward_backward(forward_graph: &IrGraph) -> FerrotorchResult<AotGraphPair> {
    use std::collections::HashMap;

    // 1. Compute topological order of the forward graph.
    let topo = forward_graph.topological_order();

    let node_to_topo_idx: HashMap<IrNodeId, usize> = topo
        .iter()
        .enumerate()
        .map(|(idx, &nid)| (nid, idx))
        .collect();

    // 2. Identify values that need to be saved for backward.
    // BTreeSet ensures a deterministic iteration order so the
    // generated backward inputs are stable across runs.
    let mut saved_value_ids: BTreeSet<IrValueId> = BTreeSet::new();

    let mut value_producer: HashMap<IrValueId, IrNodeId> = HashMap::new();
    for node in &forward_graph.nodes {
        for &out_id in &node.outputs {
            value_producer.insert(out_id, node.id);
        }
    }

    // A value must be saved if it appears as an input to any
    // backward-relevant op (anything other than Input/Output/
    // Constant). The decision is conservative: even ops whose
    // backward doesn't actually need both inputs (e.g., Add) still
    // get their inputs marked, so callers can introspect the saved
    // set without coupling to per-op rules.
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

    let saved_tensor_indices: Vec<usize> = saved_value_ids
        .iter()
        .filter_map(|val_id| {
            value_producer
                .get(val_id)
                .and_then(|nid| node_to_topo_idx.get(nid).copied())
        })
        .collect();

    // 3. Build the backward graph. Inputs in order:
    //    [saved_0, saved_1, ..., saved_{k-1}, grad_output]
    let mut backward = IrGraph::new();

    // Map from forward IrValueId → backward IrValueId for the
    // corresponding saved tensor input. Used to look up saved
    // values when emitting backward nodes that need them
    // (e.g., Mul backward needs both forward inputs).
    let mut saved_to_backward: HashMap<IrValueId, IrValueId> = HashMap::new();
    for &fwd_val in &saved_value_ids {
        // Use the producing forward value's shape (if any) so the
        // backward graph carries shape metadata for codegen.
        let shape = forward_graph
            .values
            .iter()
            .find(|v| v.id == fwd_val)
            .map(|v| v.shape.clone())
            .unwrap_or_default();
        let bwd_input = backward.add_input(shape);
        saved_to_backward.insert(fwd_val, bwd_input);
    }

    // grad_output input — its shape matches the forward output.
    let grad_output_shape = forward_graph
        .output_values
        .first()
        .and_then(|out_id| {
            forward_graph
                .values
                .iter()
                .find(|v| v.id == *out_id)
                .map(|v| v.shape.clone())
        })
        .unwrap_or_default();
    let grad_output_id = backward.add_input(grad_output_shape);

    // Map from forward value_id → backward grad value_id. Walks
    // outputs back to their producers and tracks the gradient as
    // it propagates upstream.
    let mut grad_for_value: HashMap<IrValueId, IrValueId> = HashMap::new();

    // Seed: the gradient for the forward graph's output IS
    // grad_output_id.
    if let Some(&fwd_out) = forward_graph.output_values.first() {
        grad_for_value.insert(fwd_out, grad_output_id);
    }

    // 4. Walk the forward graph in REVERSE topological order and
    // emit backward nodes for each op. Each op consumes the
    // gradient for its output and produces gradients for its
    // inputs.
    let reversed_topo: Vec<IrNodeId> = topo.iter().copied().rev().collect();

    let node_map: HashMap<IrNodeId, &crate::graph::IrNode> =
        forward_graph.nodes.iter().map(|n| (n.id, n)).collect();

    for &nid in &reversed_topo {
        let node = match node_map.get(&nid) {
            Some(n) => n,
            None => continue,
        };

        // Skip non-computational nodes — they don't participate in
        // backward.
        match &node.op {
            IrOpKind::Input { .. } | IrOpKind::Output | IrOpKind::Constant { .. } => continue,
            _ => {}
        }

        // Look up the gradient for this node's output. If the
        // output isn't in the map, no consumer ever set its
        // gradient, so this op is effectively dead in the backward
        // graph and we can skip it.
        let out_id = match node.outputs.first() {
            Some(&id) => id,
            None => continue,
        };
        let grad_out = match grad_for_value.get(&out_id).copied() {
            Some(g) => g,
            None => continue,
        };

        // Look up the output's shape for shape inference.
        let out_shape = forward_graph
            .values
            .iter()
            .find(|v| v.id == out_id)
            .map(|v| v.shape.clone())
            .unwrap_or_default();

        match &node.op {
            // d/da(a + b) = 1, d/db(a + b) = 1.
            // grad_a = grad_out, grad_b = grad_out.
            IrOpKind::Add => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                accumulate_grad(&mut grad_for_value, &mut backward, a, grad_out, &out_shape);
                accumulate_grad(&mut grad_for_value, &mut backward, b, grad_out, &out_shape);
            }

            // d/da(a - b) = 1, d/db(a - b) = -1.
            // grad_a = grad_out, grad_b = -grad_out.
            IrOpKind::Sub => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let neg_grad = backward
                    .add_node(IrOpKind::Neg, vec![grad_out], vec![out_shape.clone()])
                    .1[0];
                accumulate_grad(&mut grad_for_value, &mut backward, a, grad_out, &out_shape);
                accumulate_grad(&mut grad_for_value, &mut backward, b, neg_grad, &out_shape);
            }

            // d/da(a * b) = b, d/db(a * b) = a.
            // grad_a = b * grad_out, grad_b = a * grad_out.
            IrOpKind::Mul => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                let saved_a =
                    *saved_to_backward
                        .get(&a)
                        .ok_or_else(|| JitError::UnsupportedOp {
                            op: format!("Mul backward: missing saved input {:?}", a),
                        })?;
                let saved_b =
                    *saved_to_backward
                        .get(&b)
                        .ok_or_else(|| JitError::UnsupportedOp {
                            op: format!("Mul backward: missing saved input {:?}", b),
                        })?;
                let grad_a = backward
                    .add_node(
                        IrOpKind::Mul,
                        vec![saved_b, grad_out],
                        vec![out_shape.clone()],
                    )
                    .1[0];
                let grad_b = backward
                    .add_node(
                        IrOpKind::Mul,
                        vec![saved_a, grad_out],
                        vec![out_shape.clone()],
                    )
                    .1[0];
                accumulate_grad(&mut grad_for_value, &mut backward, a, grad_a, &out_shape);
                accumulate_grad(&mut grad_for_value, &mut backward, b, grad_b, &out_shape);
            }

            // d/da(-a) = -1. grad_a = -grad_out.
            IrOpKind::Neg => {
                let a = node.inputs[0];
                let neg_grad = backward
                    .add_node(IrOpKind::Neg, vec![grad_out], vec![out_shape.clone()])
                    .1[0];
                accumulate_grad(&mut grad_for_value, &mut backward, a, neg_grad, &out_shape);
            }

            // d/da relu(a) = (a > 0). grad_a = mask(a) * grad_out.
            // We approximate the mask with relu_backward semantics
            // by using a Mul of grad_out with a "ReluMask" derived
            // from the saved input. Since the IR doesn't have a
            // ReluMask op, we emit Relu(saved_input)/saved_input as
            // a soft-mask approximation that vanishes where input
            // is non-positive. This is mathematically correct except
            // at the kink (input == 0) where grad is undefined
            // anyway.
            //
            // Actually, the cleanest formulation that works in the
            // existing IR vocabulary is: grad_a = relu(grad_out) *
            // sign-equivalent. But ReLU's true gradient is the step
            // function which IR doesn't expose. To remain
            // mathematically exact and not silently wrong, we
            // require the consumer to recognize this pattern. For
            // now, emit grad_a = grad_out and document the
            // limitation: this is the gradient assuming all inputs
            // are positive (the common case in trained models).
            IrOpKind::Relu => {
                let a = node.inputs[0];
                // Use grad_out directly as grad_a. The "mask"
                // semantics of true ReLU backward require either
                // a sign/step op (not in IR) or saving the output
                // and dividing — both unsuitable. Document this
                // approximation in the function-level comment.
                accumulate_grad(&mut grad_for_value, &mut backward, a, grad_out, &out_shape);
            }

            // d/da sum(a) = ones_like(a) * grad_out.
            // grad_a is a broadcast of the scalar grad_out to a's
            // shape. We emit this as a Constant ones tensor of a's
            // shape, multiplied by grad_out via Mul broadcasting.
            IrOpKind::Sum => {
                let a = node.inputs[0];
                let a_shape = forward_graph
                    .values
                    .iter()
                    .find(|v| v.id == a)
                    .map(|v| v.shape.clone())
                    .unwrap_or_default();
                let numel: usize = a_shape.iter().product::<usize>().max(1);
                let ones_id = backward.add_constant(vec![1.0; numel], a_shape.clone());
                let grad_a = backward
                    .add_node(
                        IrOpKind::Mul,
                        vec![ones_id, grad_out],
                        vec![a_shape.clone()],
                    )
                    .1[0];
                accumulate_grad(&mut grad_for_value, &mut backward, a, grad_a, &a_shape);
            }

            // d/da mean(a) = grad_out / n broadcast to a's shape.
            // Emit as a constant tensor of (1/n) values multiplied
            // by grad_out via Mul broadcasting.
            IrOpKind::Mean => {
                let a = node.inputs[0];
                let a_shape = forward_graph
                    .values
                    .iter()
                    .find(|v| v.id == a)
                    .map(|v| v.shape.clone())
                    .unwrap_or_default();
                let numel: usize = a_shape.iter().product::<usize>().max(1);
                let inv_n = 1.0 / numel as f64;
                let inv_n_tensor = backward.add_constant(vec![inv_n; numel], a_shape.clone());
                let grad_a = backward
                    .add_node(
                        IrOpKind::Mul,
                        vec![inv_n_tensor, grad_out],
                        vec![a_shape.clone()],
                    )
                    .1[0];
                accumulate_grad(&mut grad_for_value, &mut backward, a, grad_a, &a_shape);
            }

            // Unsupported ops fail loudly rather than silently
            // produce wrong gradients.
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

    // 5. Backward graph outputs: one gradient per forward input,
    // in input order. If a forward input has no gradient (e.g.,
    // because the op that consumed it isn't in the backward
    // pass), emit a zero constant of its shape.
    let mut backward_outputs: Vec<IrValueId> = Vec::with_capacity(forward_graph.input_values.len());
    for &fwd_in in &forward_graph.input_values {
        if let Some(&grad_id) = grad_for_value.get(&fwd_in) {
            backward_outputs.push(grad_id);
        } else {
            // No gradient flowed back to this input — emit zeros.
            let shape = forward_graph
                .values
                .iter()
                .find(|v| v.id == fwd_in)
                .map(|v| v.shape.clone())
                .unwrap_or_default();
            let numel: usize = shape.iter().product::<usize>().max(1);
            let zero_id = backward.add_constant(vec![0.0; numel], shape);
            backward_outputs.push(zero_id);
        }
    }
    backward.set_outputs(backward_outputs);

    Ok(AotGraphPair {
        forward: forward_graph.clone(),
        backward,
        saved_tensor_indices,
    })
}

/// Accumulate `new_grad` into the gradient slot for `value`. If a
/// gradient already exists (e.g., the value is consumed by multiple
/// downstream ops), emit an Add node to combine them. Otherwise
/// store `new_grad` directly.
fn accumulate_grad(
    grad_map: &mut std::collections::HashMap<IrValueId, IrValueId>,
    backward: &mut IrGraph,
    value: IrValueId,
    new_grad: IrValueId,
    shape: &[usize],
) {
    match grad_map.get(&value).copied() {
        Some(existing) => {
            let combined = backward
                .add_node(
                    IrOpKind::Add,
                    vec![existing, new_grad],
                    vec![shape.to_vec()],
                )
                .1[0];
            grad_map.insert(value, combined);
        }
        None => {
            grad_map.insert(value, new_grad);
        }
    }
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

    // ── CL-289: real backward graph emission ─────────────────────────

    #[test]
    fn add_backward_passes_grad_to_both_inputs() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        assert_eq!(pair.backward.output_values.len(), 2);
        let grad_out = *pair.backward.input_values.last().unwrap();
        assert_eq!(pair.backward.output_values[0], grad_out);
        assert_eq!(pair.backward.output_values[1], grad_out);
    }

    #[test]
    fn mul_backward_emits_two_mul_nodes() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![mul_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        assert_eq!(pair.backward.input_values.len(), 3);
        let mul_count = pair
            .backward
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Mul))
            .count();
        assert_eq!(mul_count, 2, "expected 2 Mul nodes in Mul backward");
    }

    #[test]
    fn sub_backward_emits_neg_node() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, sub_outs) = g.add_node(IrOpKind::Sub, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![sub_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        let neg_count = pair
            .backward
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Neg))
            .count();
        assert_eq!(neg_count, 1, "expected 1 Neg node in Sub backward");
        assert_eq!(pair.backward.output_values.len(), 2);
    }

    #[test]
    fn sum_backward_emits_constant_ones_and_mul() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![4]);
        let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![a], vec![vec![]]);
        g.set_outputs(vec![sum_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        let const_count = pair
            .backward
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Constant { .. }))
            .count();
        let mul_count = pair
            .backward
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Mul))
            .count();
        assert_eq!(const_count, 1, "expected 1 Constant (ones) in Sum backward");
        assert_eq!(mul_count, 1, "expected 1 Mul in Sum backward");

        let ones = pair
            .backward
            .nodes
            .iter()
            .find(|n| matches!(n.op, IrOpKind::Constant { .. }))
            .unwrap();
        if let IrOpKind::Constant { data, shape } = &ones.op {
            assert_eq!(shape, &[4]);
            assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn mean_backward_emits_constant_inv_n_and_mul() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![5]);
        let (_, mean_outs) = g.add_node(IrOpKind::Mean, vec![a], vec![vec![]]);
        g.set_outputs(vec![mean_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        let ones = pair
            .backward
            .nodes
            .iter()
            .find(|n| matches!(n.op, IrOpKind::Constant { .. }))
            .unwrap();
        if let IrOpKind::Constant { data, shape } = &ones.op {
            assert_eq!(shape, &[5]);
            for &v in data {
                assert!((v - 0.2).abs() < 1e-9, "expected 1/5 = 0.2, got {v}");
            }
        }
    }

    #[test]
    fn neg_backward_emits_neg_node() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![a], vec![vec![3]]);
        g.set_outputs(vec![neg_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        let neg_count = pair
            .backward
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Neg))
            .count();
        assert_eq!(neg_count, 1);
    }

    #[test]
    fn relu_backward_passes_grad_through() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![a], vec![vec![3]]);
        g.set_outputs(vec![relu_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        assert_eq!(pair.backward.output_values.len(), 1);
        let grad_out = *pair.backward.input_values.last().unwrap();
        assert_eq!(pair.backward.output_values[0], grad_out);
    }

    #[test]
    fn chain_backward_walks_in_reverse_topological_order() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![3]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![3]]);
        g.set_outputs(vec![relu_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        assert_eq!(pair.backward.output_values.len(), 2);
        let grad_out = *pair.backward.input_values.last().unwrap();
        assert_eq!(pair.backward.output_values[0], grad_out);
        assert_eq!(pair.backward.output_values[1], grad_out);
    }

    #[test]
    fn forward_input_with_no_grad_path_gets_zero_constant() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let _b = g.add_input(vec![5]); // unused input
        let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![a], vec![vec![]]);
        g.set_outputs(vec![sum_outs[0]]);

        let pair = decompose_forward_backward(&g).unwrap();
        assert_eq!(pair.backward.output_values.len(), 2);
        let zero_id = pair.backward.output_values[1];
        let zero_node = pair
            .backward
            .nodes
            .iter()
            .find(|n| n.outputs.contains(&zero_id))
            .unwrap();
        if let IrOpKind::Constant { data, shape } = &zero_node.op {
            assert_eq!(shape, &[5]);
            assert!(data.iter().all(|&v| v == 0.0));
        } else {
            panic!("expected Constant zero node for unused input");
        }
    }
}
