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
        pattern_fuse(graph);
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
            | IrOpKind::Exp
            | IrOpKind::Log
    )
}

/// Returns `true` when `op` is a *unary* elementwise operation eligible for
/// fusion.  Binary ops (Add, Sub, Mul, Div) are excluded because the
/// `FusedElementwise` interpreter path applies ops sequentially on a single
/// tensor via `apply_elementwise_op`, which only supports unary ops.  Fusing
/// binary ops would produce an unexecutable node.
fn is_fusable_elementwise(op: &IrOpKind) -> bool {
    matches!(
        op,
        IrOpKind::Neg
            | IrOpKind::Relu
            | IrOpKind::Sigmoid
            | IrOpKind::Tanh
            | IrOpKind::Sqrt
            | IrOpKind::Abs
            | IrOpKind::Gelu
            | IrOpKind::Silu
            | IrOpKind::Exp
            | IrOpKind::Log
            | IrOpKind::Pow { .. }
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

/// Fuse chains of elementwise operations into a single
/// `IrOpKind::FusedElementwise { ops }` node.
///
/// A chain is a sequence of elementwise nodes where each intermediate output
/// value is consumed by exactly one subsequent elementwise node.
pub fn fuse_elementwise(graph: &mut IrGraph) {
    loop {
        if !try_fuse_one_chain(graph) {
            break;
        }
    }
}

/// Attempt to find and fuse a single elementwise chain of length >= 2.
/// Returns `true` if a fusion was performed.
fn try_fuse_one_chain(graph: &mut IrGraph) -> bool {
    // Build a consumer count map: value_id -> number of nodes consuming it.
    let mut consumer_count: HashMap<IrValueId, usize> = HashMap::new();
    for node in &graph.nodes {
        for &v in &node.inputs {
            *consumer_count.entry(v).or_insert(0) += 1;
        }
    }
    // Graph outputs also count as consumers — we must not fuse away an
    // intermediate value that is a graph output.
    for &v in &graph.output_values {
        *consumer_count.entry(v).or_insert(0) += 1;
    }

    // Build a map: value_id -> the unique node consuming it (only when
    // there is exactly one consumer total).
    let mut value_to_consumer: HashMap<IrValueId, IrNodeId> = HashMap::new();
    for node in &graph.nodes {
        for &v in &node.inputs {
            if consumer_count.get(&v).copied().unwrap_or(0) == 1 {
                value_to_consumer.insert(v, node.id);
            }
        }
    }

    // Node-id -> index for quick lookup.
    let node_index: HashMap<IrNodeId, usize> = graph
        .nodes
        .iter()
        .enumerate()
        .map(|(i, n)| (n.id, i))
        .collect();

    // Walk nodes in topological order looking for the start of a fusible chain.
    let topo = graph.topological_order();

    for &start_id in &topo {
        let start_idx = match node_index.get(&start_id) {
            Some(&i) => i,
            None => continue,
        };
        let start_node = &graph.nodes[start_idx];

        // Only fuse unary elementwise ops.  Binary ops (Add, Sub, Mul, Div)
        // cannot appear inside FusedElementwise because the interpreter
        // applies ops sequentially on a single tensor and
        // `apply_elementwise_op` only supports unary ops.
        if !is_fusable_elementwise(&start_node.op) {
            continue;
        }

        // Build the chain forward from this node.
        let mut chain_ids: Vec<IrNodeId> = vec![start_id];
        let mut current_id = start_id;

        while let Some(&cur_idx) = node_index.get(&current_id) {
            let cur_node = &graph.nodes[cur_idx];

            // The node must produce exactly one output.
            if cur_node.outputs.len() != 1 {
                break;
            }
            let out_val = cur_node.outputs[0];

            // That output must have exactly one consumer.
            let next_id = match value_to_consumer.get(&out_val) {
                Some(&nid) => nid,
                None => break,
            };

            let next_idx = match node_index.get(&next_id) {
                Some(&i) => i,
                None => break,
            };
            let next_node = &graph.nodes[next_idx];

            if !is_fusable_elementwise(&next_node.op) {
                break;
            }

            chain_ids.push(next_id);
            current_id = next_id;
        }

        if chain_ids.len() < 2 {
            continue;
        }

        // Found a chain of length >= 2 — fuse it.
        fuse_chain(graph, &chain_ids);
        return true;
    }

    false
}

// ===========================================================================
// Pattern fusion — fuse high-level operation patterns
// ===========================================================================

/// Pattern-level fusion: detects and rewrites known operation patterns.
///
/// Currently recognized patterns:
/// - **Linear + Activation** → `FusedLinearActivation`
/// - **Matmul + Scale + Softmax** (attention) → `FusedAttention`
///
/// Pattern fusion runs before elementwise fusion so that the fused nodes
/// are treated as opaque ops by the elementwise pass.
pub fn pattern_fuse(graph: &mut IrGraph) {
    fuse_linear_activation(graph);
    fuse_attention_pattern(graph);
}

/// Fuse Linear → Activation into FusedLinearActivation.
///
/// Detects chains of `Linear → Relu/Gelu/Silu/Sigmoid/Tanh` and
/// replaces them with a single `FusedLinearActivation` node.
fn fuse_linear_activation(graph: &mut IrGraph) {
    let topo = graph.topological_order();
    let mut fusions: Vec<(IrNodeId, IrNodeId, IrOpKind)> = Vec::new();

    for &nid in &topo {
        let node = match graph.nodes.iter().find(|n| n.id == nid) {
            Some(n) => n,
            None => continue,
        };

        if !matches!(node.op, IrOpKind::Linear) {
            continue;
        }

        // Check if the sole consumer is an activation.
        let linear_output = node.outputs[0];
        let consumers: Vec<IrNodeId> = graph
            .nodes
            .iter()
            .filter(|n| n.inputs.contains(&linear_output))
            .map(|n| n.id)
            .collect();

        if consumers.len() != 1 {
            continue;
        }

        let consumer = match graph.nodes.iter().find(|n| n.id == consumers[0]) {
            Some(n) => n,
            None => continue,
        };

        let is_activation = matches!(
            consumer.op,
            IrOpKind::Relu | IrOpKind::Gelu | IrOpKind::Silu | IrOpKind::Sigmoid | IrOpKind::Tanh
        );

        if is_activation {
            fusions.push((nid, consumers[0], consumer.op.clone()));
        }
    }

    // Apply fusions (replace linear op, remove activation node).
    for (linear_id, act_id, act_op) in fusions {
        // Collect activation outputs before mutating.
        let act_outputs = graph
            .nodes
            .iter()
            .find(|n| n.id == act_id)
            .map(|n| n.outputs.clone())
            .unwrap_or_default();

        // Update the linear node's op to FusedLinearActivation.
        if let Some(linear_node) = graph.nodes.iter_mut().find(|n| n.id == linear_id) {
            linear_node.op = IrOpKind::FusedLinearActivation {
                activation: Box::new(act_op),
            };
            linear_node.outputs = act_outputs;
        }

        // Remove the activation node.
        graph.nodes.retain(|n| n.id != act_id);
    }
}

/// Detect the scaled dot-product attention pattern and replace with FusedAttention.
///
/// Pattern: `Matmul(Q, K^T) → Mul(scale) → Softmax → Matmul(_, V)`
/// This is a heuristic detector — it looks for the sequence in topo order.
fn fuse_attention_pattern(graph: &mut IrGraph) {
    // Simple scan: look for Matmul → Mul/Div → Softmax → Matmul
    let topo = graph.topological_order();
    type FusionTuple = (
        IrNodeId,
        IrNodeId,
        IrNodeId,
        IrNodeId,
        usize,
        crate::graph::IrValueId,
        Vec<crate::graph::IrValueId>,
    );
    let mut fusions: Vec<FusionTuple> = Vec::new();

    for (i, &nid) in topo.iter().enumerate() {
        let node = match graph.nodes.iter().find(|n| n.id == nid) {
            Some(n) if matches!(n.op, IrOpKind::Matmul | IrOpKind::Mm) => n,
            _ => continue,
        };

        // Check: next is Mul or Div (scale), then Softmax, then Matmul.
        if i + 3 >= topo.len() {
            continue;
        }

        let scale_node = graph.nodes.iter().find(|n| n.id == topo[i + 1]);
        let softmax_node = graph.nodes.iter().find(|n| n.id == topo[i + 2]);
        let matmul2_node = graph.nodes.iter().find(|n| n.id == topo[i + 3]);

        let is_attention = matches!(
            (
                scale_node.map(|n| &n.op),
                softmax_node.map(|n| &n.op),
                matmul2_node.map(|n| &n.op),
            ),
            (
                Some(IrOpKind::Mul | IrOpKind::Div),
                Some(IrOpKind::Softmax),
                Some(IrOpKind::Matmul | IrOpKind::Mm),
            )
        );

        if is_attention {
            // Infer head_dim from the first matmul's input shape.
            let head_dim = node
                .inputs
                .first()
                .and_then(|&vid| graph.values.iter().find(|v| v.id == vid))
                .map(|v| *v.shape.last().unwrap_or(&64))
                .unwrap_or(64);

            // Collect data before mutating.
            let v_input = matmul2_node
                .and_then(|n| n.inputs.get(1).copied())
                .unwrap_or(node.inputs[1]);
            let mm2_outputs = matmul2_node.map(|n| n.outputs.clone()).unwrap_or_default();

            fusions.push((
                nid,
                topo[i + 1],
                topo[i + 2],
                topo[i + 3],
                head_dim,
                v_input,
                mm2_outputs,
            ));
        }
    }

    // Apply fusions.
    for (mm1_id, scale_id, softmax_id, matmul2_id, head_dim, v_input, mm2_outputs) in fusions {
        if let Some(mm1) = graph.nodes.iter_mut().find(|n| n.id == mm1_id) {
            mm1.inputs.push(v_input);
            mm1.op = IrOpKind::FusedAttention { head_dim };
            mm1.outputs = mm2_outputs;
        }
        graph
            .nodes
            .retain(|n| n.id != scale_id && n.id != softmax_id && n.id != matmul2_id);
    }
}

/// Collapse `chain` (ordered list of node ids forming a fusible sequence)
/// into a single `FusedElementwise` node.
fn fuse_chain(graph: &mut IrGraph, chain: &[IrNodeId]) {
    assert!(chain.len() >= 2);

    // Validate that the chain is in valid topological order: each node must
    // appear after all of its predecessors in the graph's topological ordering.
    {
        let topo = graph.topological_order();
        let topo_pos: HashMap<IrNodeId, usize> =
            topo.iter().enumerate().map(|(i, &nid)| (nid, i)).collect();
        for window in chain.windows(2) {
            let pos_a = topo_pos.get(&window[0]);
            let pos_b = topo_pos.get(&window[1]);
            assert!(
                pos_a < pos_b,
                "fuse_chain: chain is not in valid topological order: {:?} (pos {:?}) must precede {:?} (pos {:?})",
                window[0],
                pos_a,
                window[1],
                pos_b,
            );
        }
    }

    // Collect ops in chain order.
    let ops: Vec<IrOpKind> = chain
        .iter()
        .map(|&nid| find_node(graph, nid).unwrap().op.clone())
        .collect();

    // Intermediate values are outputs of every chain node except the last.
    let intermediate_values: HashSet<IrValueId> = chain[..chain.len() - 1]
        .iter()
        .flat_map(|&nid| find_node(graph, nid).unwrap().outputs.clone())
        .collect();

    // External inputs: all inputs across the chain that are NOT intermediate.
    let mut external_inputs: Vec<IrValueId> = Vec::new();
    let mut seen_inputs: HashSet<IrValueId> = HashSet::new();
    for &nid in chain {
        let node = find_node(graph, nid).unwrap();
        for &v in &node.inputs {
            if !intermediate_values.contains(&v) && seen_inputs.insert(v) {
                external_inputs.push(v);
            }
        }
    }

    // The fused node produces the output of the last chain node.
    let last_node = find_node(graph, *chain.last().unwrap()).unwrap();
    let fused_output = last_node.outputs[0];
    let output_shape = graph
        .values
        .iter()
        .find(|v| v.id == fused_output)
        .unwrap()
        .shape
        .clone();

    // Remove interior chain nodes (all except the last).  The last node is
    // kept and mutated in-place so that its output value (already referenced
    // by downstream consumers) stays valid.
    for &nid in &chain[..chain.len() - 1] {
        graph.remove_node(nid);
    }

    // Mutate the last node into the fused node.
    let last_id = *chain.last().unwrap();
    if let Some(node) = graph.nodes.iter_mut().find(|n| n.id == last_id) {
        node.op = IrOpKind::FusedElementwise { ops };
        node.inputs = external_inputs;
        node.outputs = vec![fused_output];
    }

    // Ensure the output value metadata is consistent.
    if let Some(val) = graph.values.iter_mut().find(|v| v.id == fused_output) {
        val.shape = output_shape;
        val.producer = Some(last_id);
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
    fn test_fuse_neg_relu_sigmoid() {
        // Fusion only applies to unary elementwise chains.  Binary ops
        // (Add, Sub, Mul, Div) are excluded because the interpreter's
        // `apply_elementwise_op` only supports unary ops.
        let mut g = IrGraph::new();

        let x = g.add_input(vec![4]);

        // Chain: neg(x) -> relu -> sigmoid.
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![4]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![4]]);
        let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![relu_outs[0]], vec![vec![4]]);
        g.set_outputs(vec![sig_outs[0]]);

        // Before: Input, Neg, Relu, Sigmoid = 4 nodes.
        assert_eq!(g.node_count(), 4);

        fuse_elementwise(&mut g);

        // After fusion the Neg->Relu->Sigmoid chain collapses into one
        // FusedElementwise node.  Total = Input, Fused = 2.
        assert_eq!(g.node_count(), 2);

        // Find the fused node.
        let fused = g
            .nodes
            .iter()
            .find(|n| matches!(&n.op, IrOpKind::FusedElementwise { .. }))
            .expect("should have a FusedElementwise node");

        if let IrOpKind::FusedElementwise { ops } = &fused.op {
            assert_eq!(ops.len(), 3);
            assert_eq!(ops[0], IrOpKind::Neg);
            assert_eq!(ops[1], IrOpKind::Relu);
            assert_eq!(ops[2], IrOpKind::Sigmoid);
        } else {
            panic!("expected FusedElementwise");
        }

        assert_eq!(g.output_values.len(), 1);
    }

    #[test]
    fn test_fuse_does_not_fuse_binary_ops() {
        // Binary ops must NOT be fused into FusedElementwise because the
        // interpreter's apply_elementwise_op only handles unary ops.
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

        // Binary ops block fusion: Add and Mul are binary, so the only
        // fusable unary op (Relu) is alone and cannot form a chain of
        // length >= 2.  No fusion occurs.
        assert_eq!(g.node_count(), 6);
        assert!(
            g.nodes
                .iter()
                .all(|n| !matches!(&n.op, IrOpKind::FusedElementwise { .. })),
            "should not have any FusedElementwise node"
        );
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
    fn test_fuse_does_not_fuse_through_branch() {
        // If an intermediate value is consumed by more than one node, the chain
        // must not be fused through that value.
        let mut g = IrGraph::new();

        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        let relu_out = relu_outs[0];

        // Two consumers of relu_out — prevents fusion through relu.
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

        // relu_out has 2 consumers so Input->Relu cannot fuse forward.
        // However Neg->Add (length 2) or Sigmoid->Add could each be fused.
        // At most one such pair is fused per iteration, then we re-scan.
        // Verify we never produce a FusedElementwise that includes Relu
        // (which would be wrong since relu_out fans out).
        for node in &g.nodes {
            if let IrOpKind::FusedElementwise { ops } = &node.op {
                assert!(
                    !ops.contains(&IrOpKind::Relu),
                    "Relu should not be fused when its output branches"
                );
            }
        }
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
        assert!(
            plan.is_none(),
            "memory planning disabled, should return None"
        );
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
        // fusible unary elementwise chain.
        let mut g = IrGraph::new();

        let x = g.add_input(vec![2]);

        // Constant branch: Constant(1,1) + Constant(2,2) => Constant(3,3).
        let a = g.add_constant(vec![1.0, 1.0], vec![2]);
        let b = g.add_constant(vec![2.0, 2.0], vec![2]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![2]]);

        // Main path: x * folded_constant -> neg -> relu -> output.
        // (neg -> relu is a fusible unary chain)
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![x, add_outs[0]], vec![vec![2]]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![mul_outs[0]], vec![vec![2]]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![neg_outs[0]], vec![vec![2]]);
        g.set_outputs(vec![relu_outs[0]]);

        // Unused branch hanging off x.
        let (_, _tanh_outs) = g.add_node(IrOpKind::Tanh, vec![x], vec![vec![2]]);

        let config = OptimizationConfig::default();
        let plan = optimize(&mut g, &config);

        // After constant folding: Add(Const, Const) -> Const([3,3]).
        // After DCE: original constants + Tanh branch removed.
        // After fusion: Neg -> Relu fused.
        // Remaining: Input, Constant([3,3]), Mul, FusedElementwise([Neg, Relu]) = 4 nodes.
        assert_eq!(g.node_count(), 4);

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
}
