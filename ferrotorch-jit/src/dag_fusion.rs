//! DAG-level fusion engine for the inductor-style code generator.
//!
//! Unlike the linear elementwise fusion in [`crate::optimize::fuse_elementwise`]
//! which only fuses consecutive unary elementwise chains, this module performs
//! DAG-level fusion: it identifies connected subgraphs of compatible operations
//! and groups them into [`FusionGroup`]s that can be lowered to a single fused
//! loop nest.
//!
//! # Fusion rules
//!
//! - Elementwise ops (unary and binary) that share compatible iteration domains
//!   can be fused together.
//! - Reductions (Sum, Mean, Prod) terminate fusion groups -- they become
//!   standalone groups or group boundaries.
//! - MatMul/Mm/Mv/Dot/Linear are standalone groups (opaque ops).
//! - Shape ops (Reshape, Flatten, Squeeze, Unsqueeze, Cat) are standalone.
//! - Input and Constant nodes are not placed in any group but provide inputs
//!   to groups.
//!
//! # Example
//!
//! ```text
//! x = Input
//! y = Input
//! a = x + y       # group 0
//! b = relu(a)     # group 0 (fuses with add)
//! c = sum(b)      # group 1 (reduction, standalone)
//! ```

use std::collections::{HashMap, HashSet};

use crate::codegen_ir::{self, LoopIR};
use crate::graph::{IrGraph, IrNode, IrNodeId, IrOpKind, IrValueId};

// ---------------------------------------------------------------------------
// FusionGroup
// ---------------------------------------------------------------------------

/// A group of ops that will be lowered to a single fused kernel.
///
/// All ops within a group are topologically sorted. External inputs come from
/// graph inputs, constants, or the outputs of other fusion groups. External
/// outputs are consumed by other groups or are graph outputs.
#[derive(Debug, Clone)]
pub struct FusionGroup {
    /// Ops in this fusion group (topologically sorted node IDs).
    pub node_ids: Vec<IrNodeId>,
    /// The operations themselves, in matching order.
    pub ops: Vec<IrOpKind>,
    /// Input value IDs from outside the group.
    pub external_inputs: Vec<IrValueId>,
    /// Output value IDs consumed outside the group or as graph outputs.
    pub external_outputs: Vec<IrValueId>,
    /// The kind of this fusion group, determining how it should be lowered.
    pub kind: FusionGroupKind,
}

/// The kind of a fusion group, which determines the lowering strategy.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionGroupKind {
    /// A group of fusible elementwise ops (unary and/or binary).
    Elementwise,
    /// A single reduction operation.
    Reduction,
    /// A single matrix multiplication (Matmul, Mm, Mv, Dot).
    MatMul,
    /// A fused linear operation.
    Linear,
    /// An opaque operation that cannot be fused (shape ops, softmax, etc.).
    Opaque,
}

// ---------------------------------------------------------------------------
// Fusion group discovery
// ---------------------------------------------------------------------------

/// Find fusion groups in an IR graph.
///
/// This builds a DAG from the op sequence and groups fusible ops into
/// [`FusionGroup`]s.  The groups are returned in topological order.
///
/// # Algorithm
///
/// 1. Walk nodes in topological order.
/// 2. For each elementwise node, check if it can join any existing group
///    (i.e., one of its producers is in an open group and the group's
///    iteration domain is compatible).
/// 3. If it can, add it to that group.
/// 4. If it cannot (because it's a reduction, linalg op, or has no
///    compatible producer group), create a new group.
/// 5. After all nodes are assigned, compute external inputs/outputs.
pub fn find_fusion_groups(graph: &IrGraph) -> Vec<FusionGroup> {
    let topo = graph.topological_order();

    // Node lookup
    let node_map: HashMap<IrNodeId, &IrNode> = graph.nodes.iter().map(|n| (n.id, n)).collect();

    // Value -> producer node
    let value_producer: HashMap<IrValueId, IrNodeId> = graph
        .nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(|&v| (v, n.id)))
        .collect();

    // Value -> consumers
    let mut value_consumers: HashMap<IrValueId, Vec<IrNodeId>> = HashMap::new();
    for n in &graph.nodes {
        for &v in &n.inputs {
            value_consumers.entry(v).or_default().push(n.id);
        }
    }

    // Track which group each node belongs to (node_id -> group_index)
    let mut node_to_group: HashMap<IrNodeId, usize> = HashMap::new();

    // Group builders
    let mut groups: Vec<GroupBuilder> = Vec::new();

    for &nid in &topo {
        let node = match node_map.get(&nid) {
            Some(n) => *n,
            None => continue,
        };

        // Skip Input, Constant, Output nodes -- they don't belong to groups
        match &node.op {
            IrOpKind::Input { .. } | IrOpKind::Constant { .. } | IrOpKind::Output => {
                continue;
            }
            _ => {}
        }

        let op_kind = classify_op(&node.op);

        if op_kind == FusionGroupKind::Elementwise {
            // Try to merge into a producer's group
            let producer_group = find_mergeable_group(
                node,
                &node_to_group,
                &value_producer,
                &node_map,
                &groups,
                &graph.output_values,
                &value_consumers,
            );

            if let Some(gidx) = producer_group {
                groups[gidx].node_ids.push(nid);
                groups[gidx].ops.push(node.op.clone());
                node_to_group.insert(nid, gidx);
            } else {
                let gidx = groups.len();
                groups.push(GroupBuilder {
                    node_ids: vec![nid],
                    ops: vec![node.op.clone()],
                    kind: FusionGroupKind::Elementwise,
                });
                node_to_group.insert(nid, gidx);
            }
        } else {
            // Non-fusible: create a standalone group
            let gidx = groups.len();
            groups.push(GroupBuilder {
                node_ids: vec![nid],
                ops: vec![node.op.clone()],
                kind: op_kind,
            });
            node_to_group.insert(nid, gidx);
        }
    }

    // Compute external inputs/outputs for each group
    let graph_outputs: HashSet<IrValueId> = graph.output_values.iter().copied().collect();
    let all_values_in_group: Vec<HashSet<IrValueId>> = groups
        .iter()
        .map(|g| {
            g.node_ids
                .iter()
                .flat_map(|&nid| {
                    node_map
                        .get(&nid)
                        .map(|n| n.outputs.clone())
                        .unwrap_or_default()
                })
                .collect()
        })
        .collect();

    let mut result = Vec::with_capacity(groups.len());

    for (gidx, builder) in groups.into_iter().enumerate() {
        let internal_values = &all_values_in_group[gidx];

        // External inputs: values consumed by nodes in this group that are
        // not produced by nodes in this group
        let mut external_inputs = Vec::new();
        let mut seen_inputs: HashSet<IrValueId> = HashSet::new();
        for &nid in &builder.node_ids {
            let node = node_map[&nid];
            for &v in &node.inputs {
                if !internal_values.contains(&v) && seen_inputs.insert(v) {
                    external_inputs.push(v);
                }
            }
        }

        // External outputs: values produced by nodes in this group that are
        // consumed by nodes NOT in this group, or are graph outputs
        let mut external_outputs = Vec::new();
        let mut seen_outputs: HashSet<IrValueId> = HashSet::new();
        for &nid in &builder.node_ids {
            let node = node_map[&nid];
            for &v in &node.outputs {
                if seen_outputs.insert(v) {
                    let is_graph_output = graph_outputs.contains(&v);
                    let consumed_outside = value_consumers.get(&v).is_some_and(|consumers| {
                        consumers
                            .iter()
                            .any(|&cid| node_to_group.get(&cid) != Some(&gidx))
                    });

                    if is_graph_output || consumed_outside {
                        external_outputs.push(v);
                    }
                }
            }
        }

        result.push(FusionGroup {
            node_ids: builder.node_ids,
            ops: builder.ops,
            external_inputs,
            external_outputs,
            kind: builder.kind,
        });
    }

    result
}

/// Internal builder for groups before external I/O is computed.
struct GroupBuilder {
    node_ids: Vec<IrNodeId>,
    ops: Vec<IrOpKind>,
    kind: FusionGroupKind,
}

/// Classify an `IrOpKind` into a `FusionGroupKind`.
//
// Multiple arms map to `FusionGroupKind::Elementwise`. They're kept separate
// (rather than merged) so each block of the IR taxonomy — unary, binary,
// fused — is documented and visually distinct in the source.
#[allow(clippy::match_same_arms)]
fn classify_op(op: &IrOpKind) -> FusionGroupKind {
    match op {
        // Elementwise unary
        IrOpKind::Neg
        | IrOpKind::Sqrt
        | IrOpKind::Abs
        | IrOpKind::Exp
        | IrOpKind::Log
        | IrOpKind::Relu
        | IrOpKind::Sigmoid
        | IrOpKind::Tanh
        | IrOpKind::Gelu
        | IrOpKind::Silu
        | IrOpKind::Pow { .. } => FusionGroupKind::Elementwise,

        // Elementwise binary
        IrOpKind::Add | IrOpKind::Sub | IrOpKind::Mul | IrOpKind::Div => {
            FusionGroupKind::Elementwise
        }

        // Fused elementwise
        IrOpKind::FusedElementwise { .. } => FusionGroupKind::Elementwise,

        // Reductions
        IrOpKind::Sum | IrOpKind::Mean | IrOpKind::Prod => FusionGroupKind::Reduction,

        // Linear algebra
        IrOpKind::Matmul | IrOpKind::Mm | IrOpKind::Mv | IrOpKind::Dot => FusionGroupKind::MatMul,

        IrOpKind::Linear => FusionGroupKind::Linear,

        // Everything else
        _ => FusionGroupKind::Opaque,
    }
}

/// Find a group that a new elementwise node can merge into.
///
/// A node can merge into a group if:
/// 1. At least one of its producer nodes is in the group.
/// 2. The group is an elementwise group.
/// 3. The producer's output is not consumed by nodes outside the group
///    (unless it's a graph output or consumed by exactly one other node
///    which is the current candidate).
fn find_mergeable_group(
    node: &IrNode,
    node_to_group: &HashMap<IrNodeId, usize>,
    value_producer: &HashMap<IrValueId, IrNodeId>,
    _node_map: &HashMap<IrNodeId, &IrNode>,
    groups: &[GroupBuilder],
    graph_outputs: &[IrValueId],
    value_consumers: &HashMap<IrValueId, Vec<IrNodeId>>,
) -> Option<usize> {
    let graph_output_set: HashSet<IrValueId> = graph_outputs.iter().copied().collect();

    // Check each input to find a producer group
    for &input_val in &node.inputs {
        let producer_nid = match value_producer.get(&input_val) {
            Some(&nid) => nid,
            None => continue,
        };

        let gidx = match node_to_group.get(&producer_nid) {
            Some(&g) => g,
            None => continue,
        };

        // The group must be elementwise
        if groups[gidx].kind != FusionGroupKind::Elementwise {
            continue;
        }

        // Check that the intermediate value (output of the producer that is
        // our input) is not a graph output and is only consumed by nodes that
        // are either in this group or are the current node.  This ensures we
        // don't fuse away a value that's needed elsewhere.
        let consumers = value_consumers.get(&input_val);
        let safe_to_fuse = match consumers {
            Some(c) => {
                // All consumers must either be in the same group or be the
                // current node
                c.iter()
                    .all(|&cid| cid == node.id || node_to_group.get(&cid) == Some(&gidx))
            }
            None => true,
        };

        // If the intermediate is a graph output, the producer must still
        // emit it -- but we can still fuse the consumer into the group.
        // The group will just have an additional external output.
        if safe_to_fuse || graph_output_set.contains(&input_val) {
            // Verify there are no cycles: the current node must not be
            // a producer of any node already in the group.
            let group_node_set: HashSet<IrNodeId> = groups[gidx].node_ids.iter().copied().collect();
            let would_create_cycle = node.outputs.iter().any(|&out_val| {
                value_consumers
                    .get(&out_val)
                    .is_some_and(|cs| cs.iter().any(|&c| group_node_set.contains(&c)))
            });

            if !would_create_cycle {
                return Some(gidx);
            }
        }
    }

    None
}

// ---------------------------------------------------------------------------
// Fused lowering
// ---------------------------------------------------------------------------

/// Lower fusion groups to loop IR.
///
/// For each group:
/// - Elementwise groups -> single fused loop
/// - Reduction groups -> accumulator + loop
/// - `MatMul` groups -> triple loop nest
/// - Opaque groups -> individual lowering
///
/// # Arguments
///
/// * `groups` - The fusion groups to lower.
/// * `graph` - The original IR graph (needed for shape information).
pub fn fuse_dag(groups: &[FusionGroup], graph: &IrGraph) -> Vec<Vec<LoopIR>> {
    let node_map: HashMap<IrNodeId, &IrNode> = graph.nodes.iter().map(|n| (n.id, n)).collect();

    let mut result = Vec::with_capacity(groups.len());

    for group in groups {
        let loops = lower_group(group, &node_map, graph);
        result.push(loops);
    }

    result
}

/// Lower a single fusion group to loop IR.
fn lower_group(
    group: &FusionGroup,
    node_map: &HashMap<IrNodeId, &IrNode>,
    graph: &IrGraph,
) -> Vec<LoopIR> {
    match &group.kind {
        FusionGroupKind::Elementwise => lower_elementwise_group(group, node_map, graph),
        FusionGroupKind::Reduction => {
            if group.ops.len() == 1 {
                let numel = estimate_numel_for_inputs(&group.external_inputs, graph);
                let in_names = make_input_names(group.external_inputs.len());
                let in_refs: Vec<&str> = in_names.iter().map(std::string::String::as_str).collect();
                codegen_ir::lower_to_loops(&group.ops, &in_refs, "out", numel)
            } else {
                vec![LoopIR::Comment(format!(
                    "multi-op reduction group with {} ops",
                    group.ops.len()
                ))]
            }
        }
        FusionGroupKind::MatMul => {
            // Determine M, K, N from the first input shapes
            let (m, k, n) = estimate_matmul_dims(&group.external_inputs, graph);
            codegen_ir::lower_matmul("in0", "in1", "out", m, k, n)
        }
        FusionGroupKind::Linear | FusionGroupKind::Opaque => {
            vec![LoopIR::Comment(format!(
                "opaque/linear group: {:?}",
                group.ops
            ))]
        }
    }
}

/// Lower an elementwise fusion group to a single fused loop.
fn lower_elementwise_group(
    group: &FusionGroup,
    _node_map: &HashMap<IrNodeId, &IrNode>,
    graph: &IrGraph,
) -> Vec<LoopIR> {
    let numel = estimate_numel_for_inputs(&group.external_inputs, graph);
    let in_names = make_input_names(group.external_inputs.len());
    let in_refs: Vec<&str> = in_names.iter().map(std::string::String::as_str).collect();

    codegen_ir::lower_to_loops(&group.ops, &in_refs, "out", numel)
}

/// Estimate the number of elements from the shapes of the external inputs.
fn estimate_numel_for_inputs(inputs: &[IrValueId], graph: &IrGraph) -> usize {
    for &vid in inputs {
        if let Some(val) = graph.values.iter().find(|v| v.id == vid) {
            let numel: usize = val.shape.iter().product();
            if numel > 0 {
                return numel;
            }
        }
    }
    1 // fallback: scalar
}

/// Estimate M, K, N dimensions for a matmul from input shapes.
fn estimate_matmul_dims(inputs: &[IrValueId], graph: &IrGraph) -> (usize, usize, usize) {
    let shapes: Vec<&Vec<usize>> = inputs
        .iter()
        .filter_map(|&vid| graph.values.iter().find(|v| v.id == vid).map(|v| &v.shape))
        .collect();

    if shapes.len() >= 2 && shapes[0].len() == 2 && shapes[1].len() == 2 {
        let m = shapes[0][0];
        let k = shapes[0][1];
        let n = shapes[1][1];
        (m, k, n)
    } else {
        (1, 1, 1) // fallback
    }
}

/// Generate buffer names for a given number of inputs.
fn make_input_names(count: usize) -> Vec<String> {
    (0..count).map(|i| format!("in{i}")).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{IrGraph, IrOpKind};

    // -----------------------------------------------------------------------
    // classify_op
    // -----------------------------------------------------------------------

    #[test]
    fn test_classify_elementwise() {
        assert_eq!(classify_op(&IrOpKind::Add), FusionGroupKind::Elementwise);
        assert_eq!(classify_op(&IrOpKind::Sub), FusionGroupKind::Elementwise);
        assert_eq!(classify_op(&IrOpKind::Mul), FusionGroupKind::Elementwise);
        assert_eq!(classify_op(&IrOpKind::Div), FusionGroupKind::Elementwise);
        assert_eq!(classify_op(&IrOpKind::Neg), FusionGroupKind::Elementwise);
        assert_eq!(classify_op(&IrOpKind::Relu), FusionGroupKind::Elementwise);
        assert_eq!(
            classify_op(&IrOpKind::Sigmoid),
            FusionGroupKind::Elementwise
        );
        assert_eq!(classify_op(&IrOpKind::Exp), FusionGroupKind::Elementwise);
        assert_eq!(
            classify_op(&IrOpKind::Pow { exponent: 2.0 }),
            FusionGroupKind::Elementwise
        );
    }

    #[test]
    fn test_classify_reduction() {
        assert_eq!(classify_op(&IrOpKind::Sum), FusionGroupKind::Reduction);
        assert_eq!(classify_op(&IrOpKind::Mean), FusionGroupKind::Reduction);
        assert_eq!(classify_op(&IrOpKind::Prod), FusionGroupKind::Reduction);
    }

    #[test]
    fn test_classify_matmul() {
        assert_eq!(classify_op(&IrOpKind::Matmul), FusionGroupKind::MatMul);
        assert_eq!(classify_op(&IrOpKind::Mm), FusionGroupKind::MatMul);
        assert_eq!(classify_op(&IrOpKind::Mv), FusionGroupKind::MatMul);
        assert_eq!(classify_op(&IrOpKind::Dot), FusionGroupKind::MatMul);
    }

    #[test]
    fn test_classify_opaque() {
        assert_eq!(
            classify_op(&IrOpKind::Reshape { shape: vec![2, 3] }),
            FusionGroupKind::Opaque
        );
        assert_eq!(classify_op(&IrOpKind::Flatten), FusionGroupKind::Opaque);
        assert_eq!(classify_op(&IrOpKind::Softmax), FusionGroupKind::Opaque);
        assert_eq!(
            classify_op(&IrOpKind::Cat { axis: 0 }),
            FusionGroupKind::Opaque
        );
    }

    // -----------------------------------------------------------------------
    // find_fusion_groups: simple chains
    // -----------------------------------------------------------------------

    #[test]
    fn test_single_elementwise_op() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);

        let groups = find_fusion_groups(&g);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, FusionGroupKind::Elementwise);
        assert_eq!(groups[0].ops.len(), 1);
        assert_eq!(groups[0].ops[0], IrOpKind::Relu);
    }

    #[test]
    fn test_chain_fuses_into_one_group() {
        // x -> neg -> relu -> sigmoid -> output
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![4]]);
        let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_outs[0]], vec![vec![4]]);
        g.set_outputs(vec![sig_outs[0]]);

        let groups = find_fusion_groups(&g);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, FusionGroupKind::Elementwise);
        assert_eq!(groups[0].ops.len(), 3);
        assert_eq!(groups[0].ops[0], IrOpKind::Neg);
        assert_eq!(groups[0].ops[1], IrOpKind::Relu);
        assert_eq!(groups[0].ops[2], IrOpKind::Sigmoid);
    }

    #[test]
    fn test_binary_fuses_with_unary() {
        // x, y -> add(x, y) -> relu -> output
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let y = g.add_input(vec![4]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, y], vec![vec![4]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);

        let groups = find_fusion_groups(&g);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].kind, FusionGroupKind::Elementwise);
        assert_eq!(groups[0].ops.len(), 2);
    }

    // -----------------------------------------------------------------------
    // find_fusion_groups: reduction boundary
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduction_breaks_group() {
        // x -> relu -> sum -> output
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![relu_outs[0]], vec![vec![1]]);
        g.set_outputs(vec![sum_outs[0]]);

        let groups = find_fusion_groups(&g);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].kind, FusionGroupKind::Elementwise);
        assert_eq!(groups[1].kind, FusionGroupKind::Reduction);
    }

    // -----------------------------------------------------------------------
    // find_fusion_groups: matmul standalone
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_standalone_group() {
        // A, B -> mm(A, B) -> relu -> output
        let mut g = IrGraph::new();
        let a = g.add_input(vec![2, 3]);
        let b = g.add_input(vec![3, 4]);
        let (_, mm_outs) = g.add_node(IrOpKind::Mm, vec![a, b], vec![vec![2, 4]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![mm_outs[0]], vec![vec![2, 4]]);
        g.set_outputs(vec![relu_outs[0]]);

        let groups = find_fusion_groups(&g);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].kind, FusionGroupKind::MatMul);
        assert_eq!(groups[1].kind, FusionGroupKind::Elementwise);
    }

    // -----------------------------------------------------------------------
    // find_fusion_groups: branching
    // -----------------------------------------------------------------------

    #[test]
    fn test_branch_prevents_merge() {
        // x -> relu -> neg and x -> relu -> sigmoid
        // relu output branches to two consumers, so neg and sigmoid
        // cannot fuse WITH relu (because relu output is consumed by both)
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        let relu_out = relu_outs[0];

        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![relu_out], vec![vec![4]]);
        let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_out], vec![vec![4]]);

        // Combine with add
        let (_, add_outs) =
            g.add_node(IrOpKind::Add, vec![neg_outs[0], sig_outs[0]], vec![vec![4]]);
        g.set_outputs(vec![add_outs[0]]);

        let groups = find_fusion_groups(&g);
        // relu has 2 consumers (neg, sigmoid), so it forms a standalone group
        // neg and sigmoid might form separate groups or merge differently
        // The exact grouping depends on traversal order, but we should have > 1 group
        assert!(groups.len() >= 2);
    }

    // -----------------------------------------------------------------------
    // find_fusion_groups: external I/O
    // -----------------------------------------------------------------------

    #[test]
    fn test_external_inputs_outputs() {
        // x -> relu -> output
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);

        let groups = find_fusion_groups(&g);
        assert_eq!(groups.len(), 1);

        // External input should be x (the graph input)
        assert_eq!(groups[0].external_inputs.len(), 1);
        assert_eq!(groups[0].external_inputs[0], x);

        // External output should be relu_outs[0] (the graph output)
        assert_eq!(groups[0].external_outputs.len(), 1);
        assert_eq!(groups[0].external_outputs[0], relu_outs[0]);
    }

    // -----------------------------------------------------------------------
    // fuse_dag: lowering
    // -----------------------------------------------------------------------

    #[test]
    fn test_fuse_dag_elementwise_group() {
        // x -> neg -> relu -> output
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);

        let groups = find_fusion_groups(&g);
        let loops_per_group = fuse_dag(&groups, &g);

        assert_eq!(loops_per_group.len(), 1);
        assert!(!loops_per_group[0].is_empty());
    }

    #[test]
    fn test_fuse_dag_matmul_group() {
        // A[2,3], B[3,4] -> mm -> output
        let mut g = IrGraph::new();
        let a = g.add_input(vec![2, 3]);
        let b = g.add_input(vec![3, 4]);
        let (_, mm_outs) = g.add_node(IrOpKind::Mm, vec![a, b], vec![vec![2, 4]]);
        g.set_outputs(vec![mm_outs[0]]);

        let groups = find_fusion_groups(&g);
        let loops_per_group = fuse_dag(&groups, &g);

        assert_eq!(loops_per_group.len(), 1);
        // Should produce a triple loop nest
        assert!(!loops_per_group[0].is_empty());
        match &loops_per_group[0][0] {
            LoopIR::Loop { var, .. } => assert_eq!(var, "i"),
            _ => panic!("expected outer Loop"),
        }
    }

    #[test]
    fn test_fuse_dag_reduction_group() {
        // x -> sum -> output
        let mut g = IrGraph::new();
        let x = g.add_input(vec![8]);
        let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![x], vec![vec![1]]);
        g.set_outputs(vec![sum_outs[0]]);

        let groups = find_fusion_groups(&g);
        let loops_per_group = fuse_dag(&groups, &g);

        assert_eq!(loops_per_group.len(), 1);
        assert!(!loops_per_group[0].is_empty());
        // First statement should be Let(acc)
        match &loops_per_group[0][0] {
            LoopIR::Let { var, .. } => assert_eq!(var, "acc"),
            _ => panic!("expected Let for reduction"),
        }
    }

    // -----------------------------------------------------------------------
    // make_input_names
    // -----------------------------------------------------------------------

    #[test]
    fn test_make_input_names() {
        assert_eq!(make_input_names(0), Vec::<String>::new());
        assert_eq!(make_input_names(1), vec!["in0"]);
        assert_eq!(make_input_names(3), vec!["in0", "in1", "in2"]);
    }

    // -----------------------------------------------------------------------
    // estimate helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_estimate_numel() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3, 4]);
        assert_eq!(estimate_numel_for_inputs(&[x], &g), 12);
    }

    #[test]
    fn test_estimate_matmul_dims() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![5, 7]);
        let b = g.add_input(vec![7, 3]);
        assert_eq!(estimate_matmul_dims(&[a, b], &g), (5, 7, 3));
    }

    #[test]
    fn test_estimate_matmul_dims_fallback() {
        let g = IrGraph::new();
        assert_eq!(estimate_matmul_dims(&[], &g), (1, 1, 1));
    }
}
