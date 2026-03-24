use std::collections::{HashMap, HashSet};

use crate::graph::{IrGraph, IrNode, IrNodeId, IrOpKind, IrValue, IrValueId};
use crate::memory_plan::{self, MemoryPlan};

/// Configuration controlling which optimization passes are applied.
#[derive(Debug, Clone)]
pub struct OptimizationConfig {
    pub constant_folding: bool,
    pub dead_code_elimination: bool,
    pub operator_fusion: bool,
    pub memory_planning: bool,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            constant_folding: true,
            dead_code_elimination: true,
            operator_fusion: true,
            memory_planning: true,
        }
    }
}

/// Apply the enabled optimization passes to `graph` in order:
/// constant folding, dead code elimination, operator fusion, memory planning.
///
/// Returns `Some(MemoryPlan)` when memory planning is enabled, `None` otherwise.
pub fn optimize(graph: &mut IrGraph, config: &OptimizationConfig) -> Option<MemoryPlan> {
    if config.constant_folding {
        constant_fold(graph);
    }
    if config.dead_code_elimination {
        dead_code_eliminate(graph);
    }
    if config.operator_fusion {
        fuse_elementwise(graph);
    }
    if config.memory_planning {
        Some(memory_plan::plan_memory(graph))
    } else {
        None
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Returns `true` when `op` is a simple elementwise operation that we handle
/// for both constant folding and fusion.
fn is_simple_elementwise(op: &IrOpKind) -> bool {
    matches!(
        op,
        IrOpKind::Add
            | IrOpKind::Sub
            | IrOpKind::Mul
            | IrOpKind::Div
            | IrOpKind::Neg
            | IrOpKind::Relu
            | IrOpKind::Sigmoid
            | IrOpKind::Tanh
            | IrOpKind::Sqrt
            | IrOpKind::Abs
            | IrOpKind::Pow { .. }
            | IrOpKind::Gelu
            | IrOpKind::Silu
            | IrOpKind::Exp
            | IrOpKind::Log
    )
}

/// Returns `true` when `op` is a binary elementwise operation (requires two inputs).
#[allow(dead_code)]
fn is_binary_elementwise(op: &IrOpKind) -> bool {
    matches!(
        op,
        IrOpKind::Add | IrOpKind::Sub | IrOpKind::Mul | IrOpKind::Div
    )
}

/// Look up an `IrNode` by its `IrNodeId`.
fn find_node(graph: &IrGraph, id: IrNodeId) -> Option<&IrNode> {
    graph.nodes.iter().find(|n| n.id == id)
}

/// Find the producer node id for a value.
fn producer_of(graph: &IrGraph, value: IrValueId) -> Option<IrNodeId> {
    graph
        .values
        .iter()
        .find(|v| v.id == value)
        .and_then(|v| v.producer)
}

/// Get the constant data from a `Constant` node that produces `value`.
/// Returns `None` if the value is not produced by a constant node.
fn get_constant_data(graph: &IrGraph, value: IrValueId) -> Option<(Vec<f64>, Vec<usize>)> {
    let producer_id = producer_of(graph, value)?;
    let node = find_node(graph, producer_id)?;
    match &node.op {
        IrOpKind::Constant { data, shape } => Some((data.clone(), shape.clone())),
        _ => None,
    }
}

/// Evaluate a simple elementwise operation on constant data.
fn eval_elementwise(
    op: &IrOpKind,
    inputs: &[(Vec<f64>, Vec<usize>)],
) -> Option<(Vec<f64>, Vec<usize>)> {
    match op {
        IrOpKind::Add => {
            let (a, shape_a) = &inputs[0];
            let (b, _) = &inputs[1];
            if a.len() != b.len() {
                return None;
            }
            let data: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Sub => {
            let (a, shape_a) = &inputs[0];
            let (b, _) = &inputs[1];
            if a.len() != b.len() {
                return None;
            }
            let data: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x - y).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Mul => {
            let (a, shape_a) = &inputs[0];
            let (b, _) = &inputs[1];
            if a.len() != b.len() {
                return None;
            }
            let data: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x * y).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Div => {
            let (a, shape_a) = &inputs[0];
            let (b, _) = &inputs[1];
            if a.len() != b.len() {
                return None;
            }
            let data: Vec<f64> = a.iter().zip(b.iter()).map(|(x, y)| x / y).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Neg => {
            let (a, shape_a) = &inputs[0];
            let data: Vec<f64> = a.iter().map(|x| -x).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Relu => {
            let (a, shape_a) = &inputs[0];
            let data: Vec<f64> = a.iter().map(|x| x.max(0.0)).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Sigmoid => {
            let (a, shape_a) = &inputs[0];
            let data: Vec<f64> = a.iter().map(|x| 1.0 / (1.0 + (-x).exp())).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Tanh => {
            let (a, shape_a) = &inputs[0];
            let data: Vec<f64> = a.iter().map(|x| x.tanh()).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Sqrt => {
            let (a, shape_a) = &inputs[0];
            let data: Vec<f64> = a.iter().map(|x| x.sqrt()).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Abs => {
            let (a, shape_a) = &inputs[0];
            let data: Vec<f64> = a.iter().map(|x| x.abs()).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Pow { exponent } => {
            let (a, shape_a) = &inputs[0];
            let p = *exponent;
            let data: Vec<f64> = a.iter().map(|x| x.powf(p)).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Gelu => {
            let (a, shape_a) = &inputs[0];
            let sqrt_2_over_pi = (2.0_f64 / std::f64::consts::PI).sqrt();
            let data: Vec<f64> = a
                .iter()
                .map(|x| x * 0.5 * (1.0 + (sqrt_2_over_pi * (x + 0.044715 * x.powi(3))).tanh()))
                .collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Silu => {
            let (a, shape_a) = &inputs[0];
            let data: Vec<f64> = a.iter().map(|x| x / (1.0 + (-x).exp())).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Exp => {
            let (a, shape_a) = &inputs[0];
            let data: Vec<f64> = a.iter().map(|x| x.exp()).collect();
            Some((data, shape_a.clone()))
        }
        IrOpKind::Log => {
            let (a, shape_a) = &inputs[0];
            let data: Vec<f64> = a.iter().map(|x| x.ln()).collect();
            Some((data, shape_a.clone()))
        }
        _ => None,
    }
}

// ---------------------------------------------------------------------------
// Constant Folding
// ---------------------------------------------------------------------------

/// Fold nodes whose inputs are all constants into a single `Constant` node.
///
/// Only simple elementwise operations are folded: `Add`, `Sub`, `Mul`, `Div`,
/// `Neg`, `Relu`, `Sigmoid`, `Tanh`.  Iterates to a fixed point so that
/// chained constant expressions (e.g. `Const + Const -> Const * Const`) are
/// fully collapsed.
pub fn constant_fold(graph: &mut IrGraph) {
    loop {
        let mut folded_any = false;

        // Snapshot the current node list so we can mutate the graph.
        let node_ids: Vec<IrNodeId> = graph.nodes.iter().map(|n| n.id).collect();

        for node_id in node_ids {
            let node = match find_node(graph, node_id) {
                Some(n) => n,
                None => continue,
            };

            // Skip non-elementwise ops.
            if !is_simple_elementwise(&node.op) {
                continue;
            }

            // All inputs must be constants.
            let inputs_val_ids = node.inputs.clone();
            let mut constant_inputs: Vec<(Vec<f64>, Vec<usize>)> = Vec::new();
            let mut all_constant = true;
            for &val_id in &inputs_val_ids {
                match get_constant_data(graph, val_id) {
                    Some(pair) => constant_inputs.push(pair),
                    None => {
                        all_constant = false;
                        break;
                    }
                }
            }
            if !all_constant {
                continue;
            }

            // Re-borrow to get the op.
            let op = match find_node(graph, node_id) {
                Some(n) => n.op.clone(),
                None => continue,
            };

            let (result_data, result_shape) = match eval_elementwise(&op, &constant_inputs) {
                Some(pair) => pair,
                None => continue,
            };

            // Retrieve the output value id of this node (elementwise ops produce
            // exactly one output).
            let output_value_id = match find_node(graph, node_id) {
                Some(n) => {
                    if n.outputs.is_empty() {
                        continue;
                    }
                    n.outputs[0]
                }
                None => continue,
            };

            // Check whether this output value is a graph output before we
            // remove the node (remove_node strips it from output_values).
            let is_graph_output = graph.output_values.contains(&output_value_id);

            // Remove the old node (this also removes its output values).
            graph.remove_node(node_id);

            // Re-create the output value with the same IrValueId so that
            // downstream consumers still reference it correctly.
            let const_node_id = graph.alloc_node_id();

            graph.values.push(IrValue {
                id: output_value_id,
                shape: result_shape.clone(),
                producer: Some(const_node_id),
            });

            graph.nodes.push(IrNode {
                id: const_node_id,
                op: IrOpKind::Constant {
                    data: result_data,
                    shape: result_shape,
                },
                inputs: Vec::new(),
                outputs: vec![output_value_id],
            });

            // Restore the value in output_values if remove_node stripped it.
            if is_graph_output {
                graph.output_values.push(output_value_id);
            }

            folded_any = true;

            // Restart scan — the graph has been mutated.
            break;
        }

        if !folded_any {
            break;
        }
    }
}

// ---------------------------------------------------------------------------
// Dead Code Elimination
// ---------------------------------------------------------------------------

/// Remove nodes whose outputs are unused (not consumed by any other node and
/// not graph outputs).  Iterates to a fixed point.
pub fn dead_code_eliminate(graph: &mut IrGraph) {
    loop {
        // Collect every value id that is consumed as an input by some node.
        let mut live_values: HashSet<IrValueId> = HashSet::new();

        // Graph outputs are always live.
        for &v in &graph.output_values {
            live_values.insert(v);
        }

        // Every value consumed as an input to a node is live.
        for node in &graph.nodes {
            for &v in &node.inputs {
                live_values.insert(v);
            }
        }

        // Find dead nodes: every output value is not live.
        let mut dead_nodes: Vec<IrNodeId> = Vec::new();
        for node in &graph.nodes {
            // Nodes with zero outputs are not considered dead.
            if node.outputs.is_empty() {
                continue;
            }
            // Input nodes are never dead (they represent graph entries).
            if matches!(node.op, IrOpKind::Input { .. }) {
                continue;
            }
            let all_dead = node.outputs.iter().all(|v| !live_values.contains(v));
            if all_dead {
                dead_nodes.push(node.id);
            }
        }

        if dead_nodes.is_empty() {
            break;
        }

        for id in dead_nodes {
            graph.remove_node(id);
        }
    }
}

// ---------------------------------------------------------------------------
// Operator Fusion
// ---------------------------------------------------------------------------

/// Fuse groups of elementwise operations into a single
/// `IrOpKind::FusedElementwise { ops }` node.
///
/// Supports both linear chains (`a -> relu -> sigmoid`) and multi-input
/// diamond patterns (`y = relu(x) + sigmoid(x)`) where an intermediate
/// value fans out to multiple consumers, as long as **all** consumers are
/// within the fused group.
pub fn fuse_elementwise(graph: &mut IrGraph) {
    loop {
        if !try_fuse_one_group(graph) {
            break;
        }
    }
}

/// Attempt to find and fuse a single elementwise group of size >= 2.
///
/// Strategy: walk nodes in reverse topological order. For each elementwise
/// node that produces a graph output or feeds a non-elementwise consumer,
/// walk backwards collecting all elementwise ancestors into a fusion group.
/// An intermediate value may fan out to multiple consumers as long as every
/// consumer is in the group.
///
/// Returns `true` if a fusion was performed.
fn try_fuse_one_group(graph: &mut IrGraph) -> bool {
    // Build consumer map: value_id -> set of node ids that consume it.
    let mut value_consumers: HashMap<IrValueId, Vec<IrNodeId>> = HashMap::new();
    for node in &graph.nodes {
        for &v in &node.inputs {
            value_consumers.entry(v).or_default().push(node.id);
        }
    }

    // Node-id -> node index for quick lookup.
    let node_map: HashMap<IrNodeId, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();

    // Producer map: value_id -> node_id that produces it.
    let mut value_producer: HashMap<IrValueId, IrNodeId> = HashMap::new();
    for node in &graph.nodes {
        for &v in &node.outputs {
            value_producer.insert(v, node.id);
        }
    }

    // Set of node IDs that are already FusedElementwise (skip them).
    let already_fused: HashSet<IrNodeId> = graph
        .nodes
        .iter()
        .filter(|n| matches!(n.op, IrOpKind::FusedElementwise { .. }))
        .map(|n| n.id)
        .collect();

    // Walk in reverse topological order to find "sink" elementwise nodes
    // (nodes whose output goes to a non-elementwise consumer or is a
    // graph output).
    let topo = graph.topological_order();

    let graph_output_set: HashSet<IrValueId> = graph.output_values.iter().copied().collect();

    for &sink_id in topo.iter().rev() {
        if already_fused.contains(&sink_id) {
            continue;
        }
        let sink_idx = match node_map.get(&sink_id) {
            Some(&i) => i,
            None => continue,
        };
        let sink_node = &graph.nodes[sink_idx];

        if !is_simple_elementwise(&sink_node.op) {
            continue;
        }

        // This node must produce exactly one output.
        if sink_node.outputs.len() != 1 {
            continue;
        }

        // Check that this is a "sink": its output either is a graph output
        // or is consumed by a non-elementwise node.
        let out_val = sink_node.outputs[0];
        let is_sink = graph_output_set.contains(&out_val)
            || value_consumers.get(&out_val).map_or(true, |consumers| {
                consumers.iter().any(|&cid| {
                    node_map.get(&cid).map_or(true, |&ci| {
                        !is_simple_elementwise(&graph.nodes[ci].op)
                    })
                })
            });

        if !is_sink {
            continue;
        }

        // BFS/DFS backwards from sink_id collecting elementwise ancestors.
        let mut group: HashSet<IrNodeId> = HashSet::new();
        let mut stack: Vec<IrNodeId> = vec![sink_id];

        while let Some(nid) = stack.pop() {
            if !group.insert(nid) {
                continue;
            }
            let idx = match node_map.get(&nid) {
                Some(&i) => i,
                None => continue,
            };
            let node = &graph.nodes[idx];
            for &inp_val in &node.inputs {
                if let Some(&prod_id) = value_producer.get(&inp_val) {
                    let prod_idx = match node_map.get(&prod_id) {
                        Some(&i) => i,
                        None => continue,
                    };
                    let prod_node = &graph.nodes[prod_idx];
                    if is_simple_elementwise(&prod_node.op)
                        && !already_fused.contains(&prod_id)
                        && prod_node.outputs.len() == 1
                    {
                        stack.push(prod_id);
                    }
                }
            }
        }

        // Validate the group: every intermediate value (produced by a group
        // node and consumed by another group node) must have ALL its consumers
        // in the group. If any consumer is outside, remove the producer from
        // the group (and transitively its ancestors that only serve it).
        let group = prune_group(graph, &group, &value_consumers, &node_map, &graph_output_set);

        if group.len() < 2 {
            continue;
        }

        // Fuse the group.
        fuse_group(graph, &group, sink_id, &value_consumers, &value_producer, &node_map, &graph_output_set);
        return true;
    }

    false
}

/// Remove nodes from a candidate fusion group whose output values have
/// consumers outside the group (or are graph outputs that are not the
/// group's final output). This prunes iteratively until stable.
fn prune_group(
    graph: &IrGraph,
    initial: &HashSet<IrNodeId>,
    value_consumers: &HashMap<IrValueId, Vec<IrNodeId>>,
    node_map: &HashMap<IrNodeId, usize>,
    graph_output_set: &HashSet<IrValueId>,
) -> HashSet<IrNodeId> {
    let mut group = initial.clone();

    loop {
        let mut removed_any = false;

        // Find the "sink" of the group (node in topo-last position).
        // We need to identify which node's output is the final output.
        // The sink is the node whose output is NOT consumed by any other
        // group member.
        let group_inputs: HashSet<IrValueId> = group
            .iter()
            .filter_map(|&nid| node_map.get(&nid))
            .flat_map(|&idx| graph.nodes[idx].inputs.iter().copied())
            .collect();

        let mut to_remove: Vec<IrNodeId> = Vec::new();

        for &nid in &group {
            let idx = match node_map.get(&nid) {
                Some(&i) => i,
                None => {
                    to_remove.push(nid);
                    continue;
                }
            };
            let node = &graph.nodes[idx];

            for &out_val in &node.outputs {
                // If this output value is consumed by someone outside the
                // group, this node cannot be fused away.
                let consumers = value_consumers.get(&out_val);
                let has_external_consumer = consumers.map_or(false, |cs| {
                    cs.iter().any(|cid| !group.contains(cid))
                });

                // If this output is a graph output, check if it's the
                // group's final output (consumed by no one in the group).
                let is_intermediate_graph_output =
                    graph_output_set.contains(&out_val) && group_inputs.contains(&out_val);

                if has_external_consumer || is_intermediate_graph_output {
                    to_remove.push(nid);
                    break;
                }
            }
        }

        for nid in to_remove {
            if group.remove(&nid) {
                removed_any = true;
            }
        }

        if !removed_any {
            break;
        }
    }

    group
}

/// Collapse a validated fusion group into a single `FusedElementwise` node.
///
/// The ops are ordered topologically so the fused chain executes correctly.
fn fuse_group(
    graph: &mut IrGraph,
    group: &HashSet<IrNodeId>,
    sink_id: IrNodeId,
    value_consumers: &HashMap<IrValueId, Vec<IrNodeId>>,
    _value_producer: &HashMap<IrValueId, IrNodeId>,
    _node_map: &HashMap<IrNodeId, usize>,
    graph_output_set: &HashSet<IrValueId>,
) {
    // Get topological order and filter to group members.
    let topo = graph.topological_order();
    let ordered: Vec<IrNodeId> = topo.into_iter().filter(|nid| group.contains(nid)).collect();

    // Collect ops in topological order.
    let ops: Vec<IrOpKind> = ordered
        .iter()
        .map(|&nid| find_node(graph, nid).unwrap().op.clone())
        .collect();

    // Intermediate values: produced AND consumed within the group.
    let intermediate_values: HashSet<IrValueId> = ordered
        .iter()
        .filter(|&&nid| nid != sink_id)
        .flat_map(|&nid| find_node(graph, nid).unwrap().outputs.clone())
        .filter(|v| {
            // Only intermediate if all consumers are in the group.
            value_consumers.get(v).map_or(false, |cs| {
                cs.iter().all(|cid| group.contains(cid))
            }) && !graph_output_set.contains(v)
        })
        .collect();

    // External inputs: inputs to group nodes that are NOT intermediate values.
    let mut external_inputs: Vec<IrValueId> = Vec::new();
    let mut seen_inputs: HashSet<IrValueId> = HashSet::new();
    for &nid in &ordered {
        let node = find_node(graph, nid).unwrap();
        for &v in &node.inputs {
            if !intermediate_values.contains(&v) && seen_inputs.insert(v) {
                external_inputs.push(v);
            }
        }
    }

    // The fused node produces the output of the sink node.
    let sink_node = find_node(graph, sink_id).unwrap();
    let fused_output = sink_node.outputs[0];
    let output_shape = graph
        .values
        .iter()
        .find(|v| v.id == fused_output)
        .unwrap()
        .shape
        .clone();

    // Remove all group nodes except the sink (which we mutate in-place).
    for &nid in &ordered {
        if nid != sink_id {
            graph.remove_node(nid);
        }
    }

    // Mutate the sink node into the fused node.
    if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == sink_id) {
        node.op = IrOpKind::FusedElementwise { ops };
        node.inputs = external_inputs;
        node.outputs = vec![fused_output];
    }

    // Ensure the output value metadata is consistent.
    if let Some(val) = graph.values.iter_mut().find(|v| v.id == fused_output) {
        val.shape = output_shape;
        val.producer = Some(sink_id);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{IrGraph, IrOpKind};

    // --- Constant Folding ---------------------------------------------------

    #[test]
    fn test_constant_fold_add() {
        let mut g = IrGraph::new();

        // Constant(2.0) + Constant(3.0) should fold to Constant(5.0).
        let a = g.add_constant(vec![2.0], vec![1]);
        let b = g.add_constant(vec![3.0], vec![1]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![1]]);
        g.set_outputs(vec![add_outs[0]]);

        constant_fold(&mut g);

        let output_val = g.output_values[0];
        let (data, shape) = get_constant_data(&g, output_val).expect("output should be constant");
        assert_eq!(data, vec![5.0]);
        assert_eq!(shape, vec![1]);
    }

    #[test]
    fn test_constant_fold_mul() {
        let mut g = IrGraph::new();

        let a = g.add_constant(vec![4.0, 2.0], vec![2]);
        let b = g.add_constant(vec![3.0, 5.0], vec![2]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![a, b], vec![vec![2]]);
        g.set_outputs(vec![mul_outs[0]]);

        constant_fold(&mut g);

        let output_val = g.output_values[0];
        let (data, _) = get_constant_data(&g, output_val).expect("output should be constant");
        assert_eq!(data, vec![12.0, 10.0]);
    }

    #[test]
    fn test_constant_fold_neg() {
        let mut g = IrGraph::new();

        let a = g.add_constant(vec![7.0, -3.0], vec![2]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![a], vec![vec![2]]);
        g.set_outputs(vec![neg_outs[0]]);

        constant_fold(&mut g);

        let output_val = g.output_values[0];
        let (data, _) = get_constant_data(&g, output_val).expect("output should be constant");
        assert_eq!(data, vec![-7.0, 3.0]);
    }

    #[test]
    fn test_constant_fold_chain() {
        // Constant(2) + Constant(3) = 5, then 5 * Constant(4) = 20.
        let mut g = IrGraph::new();

        let a = g.add_constant(vec![2.0], vec![1]);
        let b = g.add_constant(vec![3.0], vec![1]);
        let c = g.add_constant(vec![4.0], vec![1]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![1]]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![add_outs[0], c], vec![vec![1]]);
        g.set_outputs(vec![mul_outs[0]]);

        constant_fold(&mut g);

        let output_val = g.output_values[0];
        let (data, _) = get_constant_data(&g, output_val).expect("output should be constant");
        assert_eq!(data, vec![20.0]);
    }

    #[test]
    fn test_constant_fold_skips_non_constant_inputs() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![2]);
        let c = g.add_constant(vec![1.0, 2.0], vec![2]);
        let (add_id, add_outs) = g.add_node(IrOpKind::Add, vec![x, c], vec![vec![2]]);
        g.set_outputs(vec![add_outs[0]]);

        let count_before = g.node_count();
        constant_fold(&mut g);
        // Nothing should be folded — the Add has a non-constant input.
        assert_eq!(g.node_count(), count_before);
        assert!(find_node(&g, add_id).is_some());
    }

    // --- Dead Code Elimination -----------------------------------------------

    #[test]
    fn test_dce_removes_unused_branch() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![4]);

        // Used branch: x -> relu -> output.
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);

        // Unused branch: x -> neg (not connected to any output).
        let (_, _neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);

        assert_eq!(g.node_count(), 3); // Input, Relu, Neg

        dead_code_eliminate(&mut g);

        // Neg should be removed; Input + Relu remain.
        assert_eq!(g.node_count(), 2);
        assert_eq!(g.output_values.len(), 1);
    }

    #[test]
    fn test_dce_cascading_removal() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![2]);
        let c = g.add_constant(vec![1.0, 1.0], vec![2]);

        // Unused chain: c -> neg -> add(neg, c) — nothing connected to output.
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![c], vec![vec![2]]);
        let (_, _add_outs) = g.add_node(IrOpKind::Add, vec![neg_outs[0], c], vec![vec![2]]);

        // Used: x -> relu -> output.
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![2]]);
        g.set_outputs(vec![relu_outs[0]]);

        // 5 nodes: Input, Constant, Neg, Add, Relu.
        assert_eq!(g.node_count(), 5);

        dead_code_eliminate(&mut g);

        // After DCE: Input and Relu remain.  Constant, Neg, Add are all dead.
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_dce_preserves_graph_outputs() {
        let mut g = IrGraph::new();

        let c = g.add_constant(vec![42.0], vec![1]);
        g.set_outputs(vec![c]);

        dead_code_eliminate(&mut g);

        // The constant is a graph output — it must not be removed.
        assert_eq!(g.node_count(), 1);
        assert_eq!(g.output_values, vec![c]);
    }

    // --- Operator Fusion -----------------------------------------------------

    #[test]
    fn test_fuse_add_relu_mul() {
        let mut g = IrGraph::new();

        let x = g.add_input(vec![4]);
        let y = g.add_input(vec![4]);
        let c = g.add_constant(vec![2.0; 4], vec![4]);

        // Chain: add(x, y) -> relu -> mul(relu_out, c).
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, y], vec![vec![4]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![4]]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![relu_outs[0], c], vec![vec![4]]);
        g.set_outputs(vec![mul_outs[0]]);

        // Before: Input, Input, Constant, Add, Relu, Mul = 6 nodes.
        assert_eq!(g.node_count(), 6);

        fuse_elementwise(&mut g);

        // After fusion the Add->Relu->Mul chain collapses into one
        // FusedElementwise node.  Total = Input, Input, Constant, Fused = 4.
        assert_eq!(g.node_count(), 4);

        // Find the fused node.
        let fused = g
            .nodes
            .iter()
            .find(|n| matches!(&n.op, IrOpKind::FusedElementwise { .. }))
            .expect("should have a FusedElementwise node");

        if let IrOpKind::FusedElementwise { ops } = &fused.op {
            assert_eq!(ops.len(), 3);
            assert_eq!(ops[0], IrOpKind::Add);
            assert_eq!(ops[1], IrOpKind::Relu);
            assert_eq!(ops[2], IrOpKind::Mul);
        } else {
            panic!("expected FusedElementwise");
        }

        assert_eq!(g.output_values.len(), 1);
    }

    #[test]
    fn test_fuse_does_not_fuse_single_op() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![2]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![2]]);
        g.set_outputs(vec![relu_outs[0]]);

        let count_before = g.node_count();
        fuse_elementwise(&mut g);
        // A single op is not a chain — nothing to fuse.
        assert_eq!(g.node_count(), count_before);
    }

    #[test]
    fn test_fuse_diamond_pattern() {
        // Diamond: relu(x) fans out to neg and sigmoid, which are both
        // consumed by add. All consumers of relu_out are in the group,
        // so the entire diamond should be fused.
        let mut g = IrGraph::new();

        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        let relu_out = relu_outs[0];

        // Two consumers of relu_out, both elementwise.
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![relu_out], vec![vec![4]]);
        let (_, sigmoid_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_out], vec![vec![4]]);

        let (_, add_outs) = g.add_node(
            IrOpKind::Add,
            vec![neg_outs[0], sigmoid_outs[0]],
            vec![vec![4]],
        );
        g.set_outputs(vec![add_outs[0]]);

        // 5 nodes: Input, Relu, Neg, Sigmoid, Add.
        assert_eq!(g.node_count(), 5);

        fuse_elementwise(&mut g);

        // All four elementwise ops (Relu, Neg, Sigmoid, Add) should be
        // fused into one FusedElementwise node.
        let fused = g
            .nodes
            .iter()
            .find(|n| matches!(&n.op, IrOpKind::FusedElementwise { .. }))
            .expect("should have a FusedElementwise node");

        if let IrOpKind::FusedElementwise { ops } = &fused.op {
            assert_eq!(ops.len(), 4, "all 4 ops should be fused");
            // The ops should be in topological order.
            assert_eq!(ops[0], IrOpKind::Relu);
            // Neg and Sigmoid follow (order may vary), then Add last.
            assert_eq!(ops[3], IrOpKind::Add);
        }

        // Remaining: Input + FusedElementwise = 2.
        assert_eq!(g.node_count(), 2);
    }

    #[test]
    fn test_fuse_does_not_fuse_when_branch_escapes() {
        // If an intermediate value is consumed by a node OUTSIDE the
        // elementwise group (or is itself a graph output), the producer
        // must not be fused away.
        let mut g = IrGraph::new();

        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        let relu_out = relu_outs[0];

        // neg consumes relu_out.
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![relu_out], vec![vec![4]]);

        // BOTH relu_out AND neg_out are graph outputs. relu_out escapes,
        // so Relu cannot be fused into the group.
        g.set_outputs(vec![neg_outs[0], relu_out]);

        assert_eq!(g.node_count(), 3); // Input, Relu, Neg

        fuse_elementwise(&mut g);

        // relu_out is a graph output AND an intermediate → Relu should not
        // be fused. Only a single Neg remains, which is too small to fuse.
        // No fusion should happen.
        assert!(
            !g.nodes.iter().any(|n| matches!(&n.op, IrOpKind::FusedElementwise { .. })),
            "should not produce a FusedElementwise when an intermediate escapes"
        );
    }

    // --- Config ---------------------------------------------------------------

    #[test]
    fn test_config_disables_all_passes() {
        let mut g = IrGraph::new();

        let a = g.add_constant(vec![2.0], vec![1]);
        let b = g.add_constant(vec![3.0], vec![1]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![1]]);
        g.set_outputs(vec![add_outs[0]]);

        let config = OptimizationConfig {
            constant_folding: false,
            dead_code_elimination: false,
            operator_fusion: false,
            memory_planning: false,
        };
        let count_before = g.node_count();
        let plan = optimize(&mut g, &config);
        assert_eq!(g.node_count(), count_before);
        assert!(plan.is_none(), "memory planning disabled, should return None");
    }

    #[test]
    fn test_config_default_enables_all() {
        let config = OptimizationConfig::default();
        assert!(config.constant_folding);
        assert!(config.dead_code_elimination);
        assert!(config.operator_fusion);
        assert!(config.memory_planning);
    }

    #[test]
    fn test_full_pipeline() {
        // Graph with a constant-foldable branch, an unused branch, and a
        // fusible elementwise chain.
        let mut g = IrGraph::new();

        let x = g.add_input(vec![2]);

        // Constant branch: Constant(1,1) + Constant(2,2) => Constant(3,3).
        let a = g.add_constant(vec![1.0, 1.0], vec![2]);
        let b = g.add_constant(vec![2.0, 2.0], vec![2]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![2]]);

        // Main path: x * folded_constant -> relu -> output.
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![x, add_outs[0]], vec![vec![2]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![mul_outs[0]], vec![vec![2]]);
        g.set_outputs(vec![relu_outs[0]]);

        // Unused branch hanging off x.
        let (_, _neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![2]]);

        let config = OptimizationConfig::default();
        let plan = optimize(&mut g, &config);

        // After constant folding: Add(Const, Const) -> Const([3,3]).
        // After DCE: original constants + Neg branch removed.
        // After fusion: Mul -> Relu fused.
        // Remaining: Input, Constant([3,3]), FusedElementwise([Mul, Relu]) = 3 nodes.
        assert_eq!(g.node_count(), 3);

        // Memory planning should have run and produced a plan.
        let plan = plan.expect("memory planning enabled by default");
        assert!(plan.num_slots > 0);
        assert!(plan.planned_total <= plan.naive_total);

        let fused = g
            .nodes
            .iter()
            .find(|n| matches!(&n.op, IrOpKind::FusedElementwise { .. }));
        assert!(fused.is_some(), "should have a fused node");
    }

    // --- New ops in is_simple_elementwise ------------------------------------

    #[test]
    fn test_constant_fold_exp() {
        let mut g = IrGraph::new();
        let a = g.add_constant(vec![0.0, 1.0], vec![2]);
        let (_, exp_outs) = g.add_node(IrOpKind::Exp, vec![a], vec![vec![2]]);
        g.set_outputs(vec![exp_outs[0]]);

        constant_fold(&mut g);

        let output_val = g.output_values[0];
        let (data, _) = get_constant_data(&g, output_val).expect("output should be constant");
        assert!((data[0] - 1.0).abs() < 1e-10, "exp(0) = 1");
        assert!((data[1] - std::f64::consts::E).abs() < 1e-10, "exp(1) = e");
    }

    #[test]
    fn test_constant_fold_log() {
        let mut g = IrGraph::new();
        let a = g.add_constant(vec![1.0, std::f64::consts::E], vec![2]);
        let (_, log_outs) = g.add_node(IrOpKind::Log, vec![a], vec![vec![2]]);
        g.set_outputs(vec![log_outs[0]]);

        constant_fold(&mut g);

        let output_val = g.output_values[0];
        let (data, _) = get_constant_data(&g, output_val).expect("output should be constant");
        assert!((data[0]).abs() < 1e-10, "log(1) = 0");
        assert!((data[1] - 1.0).abs() < 1e-10, "log(e) = 1");
    }

    #[test]
    fn test_constant_fold_sqrt() {
        let mut g = IrGraph::new();
        let a = g.add_constant(vec![4.0, 9.0], vec![2]);
        let (_, sqrt_outs) = g.add_node(IrOpKind::Sqrt, vec![a], vec![vec![2]]);
        g.set_outputs(vec![sqrt_outs[0]]);

        constant_fold(&mut g);

        let output_val = g.output_values[0];
        let (data, _) = get_constant_data(&g, output_val).expect("output should be constant");
        assert!((data[0] - 2.0).abs() < 1e-10);
        assert!((data[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_fuse_exp_log_chain() {
        // exp -> log should fuse into a chain of length 2.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, exp_outs) = g.add_node(IrOpKind::Exp, vec![x], vec![vec![3]]);
        let (_, log_outs) = g.add_node(IrOpKind::Log, vec![exp_outs[0]], vec![vec![3]]);
        g.set_outputs(vec![log_outs[0]]);

        assert_eq!(g.node_count(), 3); // Input, Exp, Log

        fuse_elementwise(&mut g);

        assert_eq!(g.node_count(), 2); // Input, FusedElementwise
        let fused = g
            .nodes
            .iter()
            .find(|n| matches!(&n.op, IrOpKind::FusedElementwise { .. }))
            .expect("should have a FusedElementwise node");
        if let IrOpKind::FusedElementwise { ops } = &fused.op {
            assert_eq!(ops.len(), 2);
            assert_eq!(ops[0], IrOpKind::Exp);
            assert_eq!(ops[1], IrOpKind::Log);
        }
    }

    #[test]
    fn test_fuse_empty_graph() {
        let mut g = IrGraph::new();
        // Empty graph — nothing to fuse, should not panic.
        fuse_elementwise(&mut g);
        assert_eq!(g.node_count(), 0);
    }

    #[test]
    fn test_fuse_single_node_graph() {
        // Graph with only an input — nothing to fuse.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        g.set_outputs(vec![x]);

        fuse_elementwise(&mut g);
        assert_eq!(g.node_count(), 1);
    }

    #[test]
    fn test_fuse_non_elementwise_ops_not_fused() {
        // Sum is a reduction, not elementwise — should not be fused.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![relu_outs[0]], vec![vec![1]]);
        g.set_outputs(vec![sum_outs[0]]);

        assert_eq!(g.node_count(), 3);
        fuse_elementwise(&mut g);

        // Relu is a single elementwise op, Sum is a reduction — no chain of >= 2.
        assert!(
            !g.nodes.iter().any(|n| matches!(&n.op, IrOpKind::FusedElementwise { .. })),
            "should not fuse single elementwise op before a reduction"
        );
    }

    #[test]
    fn test_fuse_pow_sqrt_abs_chain() {
        // pow(2) -> sqrt -> abs = chain of 3 new ops.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, pow_outs) = g.add_node(
            IrOpKind::Pow { exponent: 2.0 },
            vec![x],
            vec![vec![3]],
        );
        let (_, sqrt_outs) = g.add_node(IrOpKind::Sqrt, vec![pow_outs[0]], vec![vec![3]]);
        let (_, abs_outs) = g.add_node(IrOpKind::Abs, vec![sqrt_outs[0]], vec![vec![3]]);
        g.set_outputs(vec![abs_outs[0]]);

        assert_eq!(g.node_count(), 4);
        fuse_elementwise(&mut g);
        assert_eq!(g.node_count(), 2); // Input + Fused

        let fused = g
            .nodes
            .iter()
            .find(|n| matches!(&n.op, IrOpKind::FusedElementwise { .. }))
            .expect("should have a FusedElementwise node");
        if let IrOpKind::FusedElementwise { ops } = &fused.op {
            assert_eq!(ops.len(), 3);
            assert_eq!(ops[0], IrOpKind::Pow { exponent: 2.0 });
            assert_eq!(ops[1], IrOpKind::Sqrt);
            assert_eq!(ops[2], IrOpKind::Abs);
        }
    }
}
