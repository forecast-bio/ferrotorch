//! AOT Autograd: ahead-of-time backward graph compilation for cross-op fusion.
//!
//! This module implements the "AOT Autograd" pattern from PyTorch's
//! `torch._functorch.aot_autograd`. Instead of running the backward pass
//! eagerly (dispatching each backward op individually), AOT Autograd traces
//! the backward pass at compile time, producing a joint forward+backward IR
//! graph that enables:
//!
//! - Cross-op fusion in backward (e.g., fusing relu_backward + mul_backward)
//! - Dead code elimination of unused gradient paths
//! - Memory planning across both forward and backward passes
//!
//! # Usage
//!
//! ```ignore
//! use ferrotorch_jit::aot_autograd::{aot_trace, CompiledAotFunction};
//!
//! let aot = aot_trace(
//!     |inputs| {
//!         let a = &inputs[0];
//!         let b = &inputs[1];
//!         vec![a.relu()]
//!     },
//!     &example_inputs,
//! ).unwrap();
//!
//! let compiled = aot.compile(&[true, true]);
//! let (outputs, ctx) = compiled.forward(&inputs);
//! let grad_inputs = compiled.backward(&ctx, &grad_outputs);
//! ```

use std::collections::{HashMap, HashSet};

use crate::error::JitError;
use crate::graph::{IrGraph, IrNodeId, IrOpKind, IrValueId};
use crate::optimize::{self, OptimizationConfig};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// AOT Autograd captures both the forward and backward graphs at trace time.
///
/// The forward graph is the original computation. The backward graph is
/// derived by generating backward ops for each forward op and chaining them
/// in reverse topological order.
#[derive(Debug, Clone)]
pub struct AotAutograd {
    /// The forward graph (traced from the module).
    pub forward_graph: IrGraph,
    /// The backward graph (derived from the forward graph).
    pub backward_graph: IrGraph,
    /// Mapping from forward output value ID to the backward graph input
    /// (grad_output) value ID. Key = forward output value ID index,
    /// Value = backward input value ID.
    pub grad_map: HashMap<usize, IrValueId>,
    /// Indices of forward nodes whose outputs are needed by backward
    /// (the "saved tensors"). These are indices into the forward graph's
    /// topological ordering.
    pub saved_tensor_indices: Vec<usize>,
}

/// Context created during the forward pass, holding saved tensors needed
/// by the backward pass.
#[derive(Debug, Clone)]
pub struct AotContext {
    /// Saved intermediate tensors from the forward pass, indexed by their
    /// position in `saved_tensor_indices`.
    pub saved_tensors: Vec<SavedTensor>,
}

/// A saved tensor is identified by the forward graph value ID and stores
/// metadata about its shape.
#[derive(Debug, Clone)]
pub struct SavedTensor {
    /// The forward graph value ID this tensor corresponds to.
    pub forward_value_id: IrValueId,
    /// The shape of the tensor.
    pub shape: Vec<usize>,
}

/// A compiled AOT function with optimized forward and backward graphs.
#[derive(Debug, Clone)]
pub struct CompiledAotFunction {
    /// The forward IR ops (optimized).
    pub forward_graph: IrGraph,
    /// The backward IR ops (optimized, with fusion applied).
    pub backward_graph: IrGraph,
    /// Which forward value IDs need to be saved for backward.
    pub saved_value_ids: Vec<IrValueId>,
    /// Number of original forward inputs.
    pub num_inputs: usize,
    /// Number of original forward outputs.
    pub num_outputs: usize,
}

// ---------------------------------------------------------------------------
// Saved tensor analysis
// ---------------------------------------------------------------------------

/// Determine which forward values (inputs and outputs) need to be saved for
/// backward computation of a given forward operation.
///
/// # Backward rules (saved tensor requirements)
///
/// - **Add**: no saved tensors needed (grad flows through unchanged).
/// - **Sub**: no saved tensors needed.
/// - **Mul**: save both forward inputs (grad_a = grad * b, grad_b = grad * a).
/// - **Div**: save both forward inputs.
/// - **Neg**: no saved tensors needed.
/// - **ReLU**: save forward input (needed for mask computation).
/// - **Sigmoid**: save forward output (grad = grad * out * (1 - out)).
/// - **Tanh**: save forward output (grad = grad * (1 - out^2)).
/// - **Exp**: save forward output (grad = grad * exp_out).
/// - **Log**: save forward input (grad = grad / input).
/// - **Sum**: no saved tensors needed.
/// - **MatMul/Mm**: save both forward inputs.
/// - **Sqrt**: save forward output (grad = grad / (2 * output)).
/// - **Pow**: save forward input.
/// - **Transpose/Reshape/Flatten**: no saved tensors needed.
fn needed_saved_tensors(
    forward_op: &IrOpKind,
    forward_input_ids: &[IrValueId],
    forward_output_id: IrValueId,
) -> Vec<IrValueId> {
    match forward_op {
        IrOpKind::Add | IrOpKind::Sub | IrOpKind::Neg | IrOpKind::Sum => vec![],

        IrOpKind::Mul | IrOpKind::Div => {
            if forward_input_ids.len() >= 2 {
                vec![forward_input_ids[0], forward_input_ids[1]]
            } else {
                vec![]
            }
        }

        IrOpKind::Relu | IrOpKind::Log | IrOpKind::Abs => {
            if !forward_input_ids.is_empty() {
                vec![forward_input_ids[0]]
            } else {
                vec![]
            }
        }

        IrOpKind::Sigmoid | IrOpKind::Tanh | IrOpKind::Exp | IrOpKind::Sqrt => {
            vec![forward_output_id]
        }

        IrOpKind::Pow { .. } => {
            if !forward_input_ids.is_empty() {
                vec![forward_input_ids[0]]
            } else {
                vec![]
            }
        }

        IrOpKind::Matmul | IrOpKind::Mm => {
            if forward_input_ids.len() >= 2 {
                vec![forward_input_ids[0], forward_input_ids[1]]
            } else {
                vec![]
            }
        }

        IrOpKind::Transpose
        | IrOpKind::Reshape { .. }
        | IrOpKind::Flatten
        | IrOpKind::Squeeze { .. }
        | IrOpKind::Unsqueeze { .. }
        | IrOpKind::Input { .. }
        | IrOpKind::Constant { .. }
        | IrOpKind::Output => vec![],

        _ => vec![],
    }
}

// ---------------------------------------------------------------------------
// AOT Tracing
// ---------------------------------------------------------------------------

/// Trace a forward computation expressed as an IR graph and generate the
/// corresponding backward graph.
///
/// This function takes a forward IR graph and:
/// 1. Walks the forward graph in topological order.
/// 2. For each operation, generates backward ops using `generate_backward_ops`.
/// 3. Builds a backward IR graph by chaining backward ops in reverse topological order.
/// 4. Identifies which forward intermediates are needed by the backward ("saved tensors").
///
/// # Arguments
///
/// * `forward_graph` — A traced forward IR graph (from `crate::trace::trace`).
///
/// # Returns
///
/// An `AotAutograd` containing both the forward and backward graphs, with
/// metadata about which forward tensors need to be saved.
pub fn aot_trace_from_graph(forward_graph: &IrGraph) -> Result<AotAutograd, JitError> {
    let topo_order = forward_graph.topological_order();

    // Collect info about each forward node.
    let node_map: HashMap<IrNodeId, &_> = forward_graph
        .nodes
        .iter()
        .map(|n| (n.id, n))
        .collect();

    // Track which forward values need to be saved for backward.
    let mut all_needed_forward_values: HashSet<IrValueId> = HashSet::new();

    // Build metadata for backward generation.
    struct ForwardNodeInfo {
        op: IrOpKind,
        input_ids: Vec<IrValueId>,
        output_ids: Vec<IrValueId>,
        input_shapes: Vec<Vec<usize>>,
        output_shape: Vec<usize>,
    }

    let mut forward_node_infos: Vec<ForwardNodeInfo> = Vec::new();

    // Walk forward graph in topological order and generate backward info.
    for &node_id in &topo_order {
        let node = match node_map.get(&node_id) {
            Some(n) => *n,
            None => continue,
        };

        // Skip Input and Constant nodes — they don't need backward ops.
        if matches!(
            node.op,
            IrOpKind::Input { .. } | IrOpKind::Constant { .. } | IrOpKind::Output
        ) {
            continue;
        }

        // Gather input shapes.
        let input_shapes: Vec<Vec<usize>> = node
            .inputs
            .iter()
            .map(|&vid| {
                forward_graph
                    .values
                    .iter()
                    .find(|v| v.id == vid)
                    .map(|v| v.shape.clone())
                    .unwrap_or_default()
            })
            .collect();

        // Get output shape (first output).
        let output_shape = node
            .outputs
            .first()
            .and_then(|&vid| {
                forward_graph
                    .values
                    .iter()
                    .find(|v| v.id == vid)
                    .map(|v| v.shape.clone())
            })
            .unwrap_or_default();

        let forward_output_id = node.outputs.first().copied().unwrap_or(IrValueId(0));

        let needed = needed_saved_tensors(
            &node.op,
            &node.inputs,
            forward_output_id,
        );

        // Record which forward values are needed.
        for &fv in &needed {
            all_needed_forward_values.insert(fv);
        }

        forward_node_infos.push(ForwardNodeInfo {
            op: node.op.clone(),
            input_ids: node.inputs.clone(),
            output_ids: node.outputs.clone(),
            input_shapes,
            output_shape,
        });
    }

    // Build the backward IR graph.
    //
    // The backward graph processes operations in REVERSE topological order.
    // For each forward op, the backward ops compute gradients flowing from
    // output to inputs.
    let mut backward_graph = IrGraph::new();

    // Add backward graph inputs: one per forward output (grad_output).
    let mut grad_map: HashMap<usize, IrValueId> = HashMap::new();
    for (i, &output_val) in forward_graph.output_values.iter().enumerate() {
        let output_shape = forward_graph
            .values
            .iter()
            .find(|v| v.id == output_val)
            .map(|v| v.shape.clone())
            .unwrap_or_default();
        let grad_input = backward_graph.add_input(output_shape);
        grad_map.insert(i, grad_input);
    }

    // Add saved tensor inputs to the backward graph.
    // Each saved forward value becomes an input to the backward graph.
    let saved_tensor_indices: Vec<usize> = all_needed_forward_values
        .iter()
        .enumerate()
        .map(|(i, _)| i)
        .collect();

    let saved_values_ordered: Vec<IrValueId> = all_needed_forward_values
        .iter()
        .copied()
        .collect();

    let mut forward_val_to_backward_input: HashMap<IrValueId, IrValueId> = HashMap::new();
    for &fwd_val in &saved_values_ordered {
        let shape = forward_graph
            .values
            .iter()
            .find(|v| v.id == fwd_val)
            .map(|v| v.shape.clone())
            .unwrap_or_default();
        let bwd_input = backward_graph.add_input(shape);
        forward_val_to_backward_input.insert(fwd_val, bwd_input);
    }

    // Track the gradient value for each forward value in the backward graph.
    // Initially, the gradient of each forward output is the corresponding
    // backward graph input.
    let mut grad_values: HashMap<IrValueId, IrValueId> = HashMap::new();
    for (i, &fwd_output) in forward_graph.output_values.iter().enumerate() {
        if let Some(&bwd_input) = grad_map.get(&i) {
            grad_values.insert(fwd_output, bwd_input);
        }
    }

    // Process forward nodes in REVERSE topological order.
    for info in forward_node_infos.iter().rev() {
        // Get the gradient of this forward op's output.
        let output_val = info.output_ids.first().copied().unwrap_or(IrValueId(0));
        let grad_output_val = match grad_values.get(&output_val) {
            Some(&v) => v,
            None => continue, // No gradient flows to this op.
        };

        // Generate backward ops based on the forward op type.
        match &info.op {
            IrOpKind::Add => {
                // grad_a = grad_out, grad_b = grad_out
                // Both inputs get the gradient unchanged.
                if info.input_ids.len() >= 2 {
                    // If both inputs are the same (x + x), we need to accumulate.
                    if info.input_ids[0] == info.input_ids[1] {
                        // grad_x = grad_out + grad_out = 2 * grad_out
                        let (_, add_outs) = backward_graph.add_node(
                            IrOpKind::Add,
                            vec![grad_output_val, grad_output_val],
                            vec![info.input_shapes[0].clone()],
                        );
                        accumulate_grad(
                            &mut grad_values,
                            &mut backward_graph,
                            info.input_ids[0],
                            add_outs[0],
                            &info.input_shapes[0],
                        );
                    } else {
                        accumulate_grad(
                            &mut grad_values,
                            &mut backward_graph,
                            info.input_ids[0],
                            grad_output_val,
                            &info.input_shapes[0],
                        );
                        accumulate_grad(
                            &mut grad_values,
                            &mut backward_graph,
                            info.input_ids[1],
                            grad_output_val,
                            &info.input_shapes[1],
                        );
                    }
                }
            }

            IrOpKind::Sub => {
                // grad_a = grad_out, grad_b = -grad_out
                if info.input_ids.len() >= 2 {
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        info.input_ids[0],
                        grad_output_val,
                        &info.input_shapes[0],
                    );

                    let (_, neg_outs) = backward_graph.add_node(
                        IrOpKind::Neg,
                        vec![grad_output_val],
                        vec![info.input_shapes[1].clone()],
                    );
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        info.input_ids[1],
                        neg_outs[0],
                        &info.input_shapes[1],
                    );
                }
            }

            IrOpKind::Mul => {
                // grad_a = grad_out * b, grad_b = grad_out * a
                if info.input_ids.len() >= 2 {
                    let b_saved = forward_val_to_backward_input
                        .get(&info.input_ids[1])
                        .copied()
                        .unwrap_or(grad_output_val);
                    let a_saved = forward_val_to_backward_input
                        .get(&info.input_ids[0])
                        .copied()
                        .unwrap_or(grad_output_val);

                    // grad_a = grad_out * b
                    let (_, grad_a_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![grad_output_val, b_saved],
                        vec![info.input_shapes[0].clone()],
                    );
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        info.input_ids[0],
                        grad_a_outs[0],
                        &info.input_shapes[0],
                    );

                    // grad_b = grad_out * a
                    let (_, grad_b_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![grad_output_val, a_saved],
                        vec![info.input_shapes[1].clone()],
                    );
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        info.input_ids[1],
                        grad_b_outs[0],
                        &info.input_shapes[1],
                    );
                }
            }

            IrOpKind::Div => {
                // grad_a = grad_out / b
                // grad_b = -grad_out * a / b^2
                if info.input_ids.len() >= 2 {
                    let b_saved = forward_val_to_backward_input
                        .get(&info.input_ids[1])
                        .copied()
                        .unwrap_or(grad_output_val);
                    let a_saved = forward_val_to_backward_input
                        .get(&info.input_ids[0])
                        .copied()
                        .unwrap_or(grad_output_val);

                    // grad_a = grad_out / b
                    let (_, grad_a_outs) = backward_graph.add_node(
                        IrOpKind::Div,
                        vec![grad_output_val, b_saved],
                        vec![info.input_shapes[0].clone()],
                    );
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        info.input_ids[0],
                        grad_a_outs[0],
                        &info.input_shapes[0],
                    );

                    // grad_b = -grad_out * a / b^2
                    // Step 1: neg(grad_out)
                    let (_, neg_outs) = backward_graph.add_node(
                        IrOpKind::Neg,
                        vec![grad_output_val],
                        vec![info.input_shapes[1].clone()],
                    );
                    // Step 2: neg_grad * a
                    let (_, mul_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![neg_outs[0], a_saved],
                        vec![info.input_shapes[1].clone()],
                    );
                    // Step 3: b * b
                    let (_, b_sq_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![b_saved, b_saved],
                        vec![info.input_shapes[1].clone()],
                    );
                    // Step 4: (neg_grad * a) / (b * b)
                    let (_, grad_b_outs) = backward_graph.add_node(
                        IrOpKind::Div,
                        vec![mul_outs[0], b_sq_outs[0]],
                        vec![info.input_shapes[1].clone()],
                    );
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        info.input_ids[1],
                        grad_b_outs[0],
                        &info.input_shapes[1],
                    );
                }
            }

            IrOpKind::Neg => {
                // grad_in = -grad_out
                let (_, neg_outs) = backward_graph.add_node(
                    IrOpKind::Neg,
                    vec![grad_output_val],
                    vec![info
                        .input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| info.output_shape.clone())],
                );
                if let Some(&input_id) = info.input_ids.first() {
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        neg_outs[0],
                        info.input_shapes.first().unwrap_or(&info.output_shape),
                    );
                }
            }

            IrOpKind::Relu => {
                // grad_in = grad_out * relu_mask(saved_input)
                // We generate: relu(input), abs(input), add(abs, eps), div(relu, sum), mul(grad, mask)
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info
                        .input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| info.output_shape.clone());

                    let saved_input = forward_val_to_backward_input
                        .get(&input_id)
                        .copied()
                        .unwrap_or(grad_output_val);

                    // Step 1: relu(saved_input) -> positive parts
                    let (_, relu_outs) = backward_graph.add_node(
                        IrOpKind::Relu,
                        vec![saved_input],
                        vec![input_shape.clone()],
                    );

                    // Step 2: abs(saved_input) -> |x|
                    let (_, abs_outs) = backward_graph.add_node(
                        IrOpKind::Abs,
                        vec![saved_input],
                        vec![input_shape.clone()],
                    );

                    // Step 3: constant epsilon
                    let numel: usize = input_shape.iter().product();
                    let eps_data = vec![1e-20_f64; numel.max(1)];
                    let eps_val =
                        backward_graph.add_constant(eps_data, input_shape.clone());

                    // Step 4: abs + eps
                    let (_, abs_eps_outs) = backward_graph.add_node(
                        IrOpKind::Add,
                        vec![abs_outs[0], eps_val],
                        vec![input_shape.clone()],
                    );

                    // Step 5: relu / (abs + eps) -> mask (0 or ~1)
                    let (_, mask_outs) = backward_graph.add_node(
                        IrOpKind::Div,
                        vec![relu_outs[0], abs_eps_outs[0]],
                        vec![input_shape.clone()],
                    );

                    // Step 6: grad_out * mask -> grad_input
                    let (_, grad_in_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![grad_output_val, mask_outs[0]],
                        vec![input_shape.clone()],
                    );

                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        grad_in_outs[0],
                        &input_shape,
                    );
                }
            }

            IrOpKind::Sigmoid => {
                // grad_in = grad_out * sigmoid_out * (1 - sigmoid_out)
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info
                        .input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| info.output_shape.clone());

                    let saved_output = forward_val_to_backward_input
                        .get(&output_val)
                        .copied()
                        .unwrap_or(grad_output_val);

                    // Step 1: constant ones
                    let numel: usize = info.output_shape.iter().product();
                    let ones_data = vec![1.0_f64; numel.max(1)];
                    let ones_val =
                        backward_graph.add_constant(ones_data, info.output_shape.clone());

                    // Step 2: 1 - sigmoid_out
                    let (_, sub_outs) = backward_graph.add_node(
                        IrOpKind::Sub,
                        vec![ones_val, saved_output],
                        vec![info.output_shape.clone()],
                    );

                    // Step 3: sigmoid_out * (1 - sigmoid_out)
                    let (_, mul1_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![saved_output, sub_outs[0]],
                        vec![info.output_shape.clone()],
                    );

                    // Step 4: grad_out * sigmoid_out * (1 - sigmoid_out)
                    let (_, grad_in_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![grad_output_val, mul1_outs[0]],
                        vec![input_shape.clone()],
                    );

                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        grad_in_outs[0],
                        &input_shape,
                    );
                }
            }

            IrOpKind::Tanh => {
                // grad_in = grad_out * (1 - tanh_out^2)
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info
                        .input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| info.output_shape.clone());

                    let saved_output = forward_val_to_backward_input
                        .get(&output_val)
                        .copied()
                        .unwrap_or(grad_output_val);

                    // Step 1: tanh_out^2
                    let (_, sq_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![saved_output, saved_output],
                        vec![info.output_shape.clone()],
                    );

                    // Step 2: constant ones
                    let numel: usize = info.output_shape.iter().product();
                    let ones_data = vec![1.0_f64; numel.max(1)];
                    let ones_val =
                        backward_graph.add_constant(ones_data, info.output_shape.clone());

                    // Step 3: 1 - tanh_out^2
                    let (_, sub_outs) = backward_graph.add_node(
                        IrOpKind::Sub,
                        vec![ones_val, sq_outs[0]],
                        vec![info.output_shape.clone()],
                    );

                    // Step 4: grad_out * (1 - tanh_out^2)
                    let (_, grad_in_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![grad_output_val, sub_outs[0]],
                        vec![input_shape.clone()],
                    );

                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        grad_in_outs[0],
                        &input_shape,
                    );
                }
            }

            IrOpKind::Exp => {
                // grad_in = grad_out * exp_out
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info
                        .input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| info.output_shape.clone());

                    let saved_output = forward_val_to_backward_input
                        .get(&output_val)
                        .copied()
                        .unwrap_or(grad_output_val);

                    let (_, grad_in_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![grad_output_val, saved_output],
                        vec![input_shape.clone()],
                    );

                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        grad_in_outs[0],
                        &input_shape,
                    );
                }
            }

            IrOpKind::Log => {
                // grad_in = grad_out / input
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info
                        .input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| info.output_shape.clone());

                    let saved_input = forward_val_to_backward_input
                        .get(&input_id)
                        .copied()
                        .unwrap_or(grad_output_val);

                    let (_, grad_in_outs) = backward_graph.add_node(
                        IrOpKind::Div,
                        vec![grad_output_val, saved_input],
                        vec![input_shape.clone()],
                    );

                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        grad_in_outs[0],
                        &input_shape,
                    );
                }
            }

            IrOpKind::Sum => {
                // grad_in = broadcast grad_out to input shape
                // For a full reduction (sum over all elements), the scalar
                // gradient must be expanded to the input shape. We achieve
                // this by creating a ones constant with the input shape and
                // multiplying by the scalar gradient.
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info.input_shapes[0].clone();
                    let numel: usize = input_shape.iter().product();

                    if numel > 0 {
                        // Create ones constant with input shape.
                        let ones_data = vec![1.0_f64; numel];
                        let ones_val =
                            backward_graph.add_constant(ones_data, input_shape.clone());

                        // Mul(ones, grad_out_scalar) — but grad_out is scalar
                        // and we need broadcasting. Since our IR doesn't have
                        // explicit broadcasting, we reshape grad_out to [1,1,..]
                        // then multiply. However, the interpreter's Mul does
                        // support broadcasting via ferrotorch_core::arithmetic::mul.
                        // So we can directly multiply ones * grad_out.
                        let (_, grad_in_outs) = backward_graph.add_node(
                            IrOpKind::Mul,
                            vec![ones_val, grad_output_val],
                            vec![input_shape.clone()],
                        );

                        accumulate_grad(
                            &mut grad_values,
                            &mut backward_graph,
                            input_id,
                            grad_in_outs[0],
                            &input_shape,
                        );
                    }
                }
            }

            IrOpKind::Matmul | IrOpKind::Mm => {
                // grad_A = grad_C @ B^T
                // grad_B = A^T @ grad_C
                if info.input_ids.len() >= 2 {
                    let a_shape = &info.input_shapes[0];
                    let b_shape = &info.input_shapes[1];

                    let a_saved = forward_val_to_backward_input
                        .get(&info.input_ids[0])
                        .copied()
                        .unwrap_or(grad_output_val);
                    let b_saved = forward_val_to_backward_input
                        .get(&info.input_ids[1])
                        .copied()
                        .unwrap_or(grad_output_val);

                    // Transpose B -> B^T
                    let b_t_shape: Vec<usize> = b_shape.iter().rev().cloned().collect();
                    let (_, bt_outs) = backward_graph.add_node(
                        IrOpKind::Transpose,
                        vec![b_saved],
                        vec![b_t_shape],
                    );

                    // grad_A = grad_out @ B^T
                    let (_, grad_a_outs) = backward_graph.add_node(
                        IrOpKind::Mm,
                        vec![grad_output_val, bt_outs[0]],
                        vec![a_shape.clone()],
                    );
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        info.input_ids[0],
                        grad_a_outs[0],
                        a_shape,
                    );

                    // Transpose A -> A^T
                    let a_t_shape: Vec<usize> = a_shape.iter().rev().cloned().collect();
                    let (_, at_outs) = backward_graph.add_node(
                        IrOpKind::Transpose,
                        vec![a_saved],
                        vec![a_t_shape],
                    );

                    // grad_B = A^T @ grad_out
                    let (_, grad_b_outs) = backward_graph.add_node(
                        IrOpKind::Mm,
                        vec![at_outs[0], grad_output_val],
                        vec![b_shape.clone()],
                    );
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        info.input_ids[1],
                        grad_b_outs[0],
                        b_shape,
                    );
                }
            }

            IrOpKind::Sqrt => {
                // grad_in = grad_out * 0.5 / sqrt(input) = grad_out / (2 * output)
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info
                        .input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| info.output_shape.clone());

                    let saved_output = forward_val_to_backward_input
                        .get(&output_val)
                        .copied()
                        .unwrap_or(grad_output_val);

                    // Step 1: 2 * output
                    let (_, two_out_outs) = backward_graph.add_node(
                        IrOpKind::Add,
                        vec![saved_output, saved_output],
                        vec![info.output_shape.clone()],
                    );

                    // Step 2: grad_out / (2 * output)
                    let (_, grad_in_outs) = backward_graph.add_node(
                        IrOpKind::Div,
                        vec![grad_output_val, two_out_outs[0]],
                        vec![input_shape.clone()],
                    );

                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        grad_in_outs[0],
                        &input_shape,
                    );
                }
            }

            IrOpKind::Pow { exponent } => {
                // grad_in = grad_out * exponent * input^(exponent-1)
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info
                        .input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| info.output_shape.clone());

                    let saved_input = forward_val_to_backward_input
                        .get(&input_id)
                        .copied()
                        .unwrap_or(grad_output_val);

                    // Step 1: input^(exponent-1)
                    let (_, pow_outs) = backward_graph.add_node(
                        IrOpKind::Pow {
                            exponent: exponent - 1.0,
                        },
                        vec![saved_input],
                        vec![input_shape.clone()],
                    );

                    // Step 2: exponent * input^(exponent-1)
                    let numel: usize = input_shape.iter().product();
                    let exp_data = vec![*exponent; numel.max(1)];
                    let exp_const =
                        backward_graph.add_constant(exp_data, input_shape.clone());
                    let (_, scaled_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![exp_const, pow_outs[0]],
                        vec![input_shape.clone()],
                    );

                    // Step 3: grad_out * exponent * input^(exponent-1)
                    let (_, grad_in_outs) = backward_graph.add_node(
                        IrOpKind::Mul,
                        vec![grad_output_val, scaled_outs[0]],
                        vec![input_shape.clone()],
                    );

                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        grad_in_outs[0],
                        &input_shape,
                    );
                }
            }

            // Shape ops pass gradients through with appropriate reshaping.
            IrOpKind::Transpose => {
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info.input_shapes[0].clone();
                    let (_, t_outs) = backward_graph.add_node(
                        IrOpKind::Transpose,
                        vec![grad_output_val],
                        vec![input_shape.clone()],
                    );
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        t_outs[0],
                        &input_shape,
                    );
                }
            }

            IrOpKind::Reshape { .. } => {
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info.input_shapes[0].clone();
                    let target_shape: Vec<isize> =
                        input_shape.iter().map(|&d| d as isize).collect();
                    let (_, r_outs) = backward_graph.add_node(
                        IrOpKind::Reshape {
                            shape: target_shape,
                        },
                        vec![grad_output_val],
                        vec![input_shape.clone()],
                    );
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        r_outs[0],
                        &input_shape,
                    );
                }
            }

            // Other ops: gradient passes through unchanged (identity backward).
            _ => {
                if let Some(&input_id) = info.input_ids.first() {
                    let input_shape = info
                        .input_shapes
                        .first()
                        .cloned()
                        .unwrap_or_else(|| info.output_shape.clone());
                    accumulate_grad(
                        &mut grad_values,
                        &mut backward_graph,
                        input_id,
                        grad_output_val,
                        &input_shape,
                    );
                }
            }
        }
    }

    // Set backward graph outputs: gradients for each forward input.
    let mut backward_outputs: Vec<IrValueId> = Vec::new();
    for &fwd_input_val in &forward_graph.input_values {
        if let Some(&grad_val) = grad_values.get(&fwd_input_val) {
            backward_outputs.push(grad_val);
        }
    }
    backward_graph.set_outputs(backward_outputs);

    Ok(AotAutograd {
        forward_graph: forward_graph.clone(),
        backward_graph,
        grad_map,
        saved_tensor_indices,
    })
}

/// Accumulate a gradient value for a forward value ID.
///
/// If there's already a gradient for this forward value, emit an Add node
/// to accumulate. Otherwise, just record the mapping.
fn accumulate_grad(
    grad_values: &mut HashMap<IrValueId, IrValueId>,
    backward_graph: &mut IrGraph,
    forward_val_id: IrValueId,
    new_grad_val: IrValueId,
    shape: &[usize],
) {
    if let Some(&existing_grad) = grad_values.get(&forward_val_id) {
        // Accumulate: add the existing and new gradients.
        let (_, add_outs) = backward_graph.add_node(
            IrOpKind::Add,
            vec![existing_grad, new_grad_val],
            vec![shape.to_vec()],
        );
        grad_values.insert(forward_val_id, add_outs[0]);
    } else {
        grad_values.insert(forward_val_id, new_grad_val);
    }
}

// ---------------------------------------------------------------------------
// Dead Code Elimination for Backward
// ---------------------------------------------------------------------------

/// Eliminate backward ops that don't contribute to any needed gradient.
///
/// If a user only needs gradients for inputs 0 and 2 (not 1), this removes
/// all backward ops that only contribute to input 1's gradient.
///
/// # Algorithm
///
/// 1. Start from the `needed_output_indices` — which backward graph outputs
///    are actually needed.
/// 2. Walk backward through the backward graph, marking all ops that are
///    reachable from the needed outputs.
/// 3. Remove all unreachable ops.
pub fn eliminate_dead_backward_ops(
    backward_graph: &mut IrGraph,
    needed_output_indices: &[usize],
) {
    if backward_graph.output_values.is_empty() {
        return;
    }

    // If no outputs are needed, clear all output values and remove all
    // non-input nodes.
    if needed_output_indices.is_empty() {
        backward_graph.output_values.clear();
        let non_input_nodes: Vec<IrNodeId> = backward_graph
            .nodes
            .iter()
            .filter(|n| !matches!(n.op, IrOpKind::Input { .. } | IrOpKind::Constant { .. }))
            .map(|n| n.id)
            .collect();
        for node_id in non_input_nodes {
            backward_graph.remove_node(node_id);
        }
        return;
    }

    // Determine which output values are needed.
    let needed_output_values: HashSet<IrValueId> = needed_output_indices
        .iter()
        .filter_map(|&i| backward_graph.output_values.get(i).copied())
        .collect();

    // Build producer map: value_id -> node_id.
    let value_producer: HashMap<IrValueId, IrNodeId> = backward_graph
        .nodes
        .iter()
        .flat_map(|n| n.outputs.iter().map(move |&v| (v, n.id)))
        .collect();

    // BFS backward from needed outputs to find all reachable nodes.
    let mut reachable_nodes: HashSet<IrNodeId> = HashSet::new();
    let mut reachable_values: HashSet<IrValueId> = HashSet::new();
    let mut queue: Vec<IrValueId> = needed_output_values.into_iter().collect();

    while let Some(val_id) = queue.pop() {
        if !reachable_values.insert(val_id) {
            continue;
        }

        if let Some(&producer_id) = value_producer.get(&val_id) {
            if reachable_nodes.insert(producer_id) {
                // Add all inputs of this node to the queue.
                if let Some(node) = backward_graph.nodes.iter().find(|n| n.id == producer_id) {
                    for &input_val in &node.inputs {
                        queue.push(input_val);
                    }
                }
            }
        }
    }

    // Input nodes are always reachable (they provide data to the graph).
    for node in &backward_graph.nodes {
        if matches!(node.op, IrOpKind::Input { .. } | IrOpKind::Constant { .. }) {
            reachable_nodes.insert(node.id);
        }
    }

    // Remove unreachable nodes.
    let unreachable: Vec<IrNodeId> = backward_graph
        .nodes
        .iter()
        .filter(|n| !reachable_nodes.contains(&n.id))
        .map(|n| n.id)
        .collect();

    for node_id in unreachable {
        backward_graph.remove_node(node_id);
    }

    // Update output_values to only include needed ones.
    let original_outputs = backward_graph.output_values.clone();
    backward_graph.output_values = needed_output_indices
        .iter()
        .filter_map(|&i| original_outputs.get(i).copied())
        .collect();
}

// ---------------------------------------------------------------------------
// Backward Optimization
// ---------------------------------------------------------------------------

/// Apply optimization passes to the backward graph.
///
/// This runs the existing fusion and optimization passes from
/// `ferrotorch-jit` on the backward ops, enabling cross-op fusion
/// (e.g., fusing consecutive elementwise backward ops like
/// sigmoid_backward followed by mul_backward).
pub fn optimize_backward(backward_graph: &mut IrGraph) {
    let config = OptimizationConfig {
        constant_folding: true,
        dead_code_elimination: true,
        operator_fusion: true,
        memory_planning: false, // Memory planning is done separately for the joint graph.
    };
    optimize::optimize(backward_graph, &config);
}

// ---------------------------------------------------------------------------
// CompiledAotFunction
// ---------------------------------------------------------------------------

impl AotAutograd {
    /// Compile the AOT-traced function with optional dead gradient elimination
    /// and backward fusion.
    ///
    /// # Arguments
    ///
    /// * `needed_grads` — For each forward input, whether we need its gradient.
    ///   If `None`, all input gradients are assumed needed.
    pub fn compile(
        &self,
        needed_grads: Option<&[bool]>,
    ) -> CompiledAotFunction {
        let mut backward_graph = self.backward_graph.clone();

        // Dead code elimination: remove ops for unneeded gradients.
        if let Some(grads) = needed_grads {
            let needed_indices: Vec<usize> = grads
                .iter()
                .enumerate()
                .filter(|&(_, needed)| *needed)
                .map(|(i, _)| i)
                .collect();
            eliminate_dead_backward_ops(&mut backward_graph, &needed_indices);
        }

        // Apply optimization passes to the backward graph.
        optimize_backward(&mut backward_graph);

        // Collect the forward value IDs that need to be saved.
        // These are all forward values referenced as saved tensors in the backward.
        let mut saved_value_ids: Vec<IrValueId> = Vec::new();

        // The saved values are the backward graph inputs after the grad_output inputs.
        // The first N inputs are grad_outputs (one per forward output).
        // The remaining inputs are saved forward tensors.
        let num_grad_inputs = self.forward_graph.output_values.len();
        for (i, &bwd_input) in backward_graph.input_values.iter().enumerate() {
            if i >= num_grad_inputs {
                // This is a saved tensor input.
                saved_value_ids.push(bwd_input);
            }
        }

        CompiledAotFunction {
            forward_graph: self.forward_graph.clone(),
            backward_graph,
            saved_value_ids,
            num_inputs: self.forward_graph.input_values.len(),
            num_outputs: self.forward_graph.output_values.len(),
        }
    }

    /// Access the forward graph.
    pub fn forward_graph(&self) -> &IrGraph {
        &self.forward_graph
    }

    /// Access the backward graph.
    pub fn backward_graph(&self) -> &IrGraph {
        &self.backward_graph
    }

    /// Get the number of forward inputs.
    pub fn num_inputs(&self) -> usize {
        self.forward_graph.input_values.len()
    }

    /// Get the number of forward outputs.
    pub fn num_outputs(&self) -> usize {
        self.forward_graph.output_values.len()
    }

    /// Get the saved tensor indices.
    pub fn saved_tensor_indices(&self) -> &[usize] {
        &self.saved_tensor_indices
    }
}

impl CompiledAotFunction {
    /// Get the forward graph.
    pub fn forward_graph(&self) -> &IrGraph {
        &self.forward_graph
    }

    /// Get the backward graph.
    pub fn backward_graph(&self) -> &IrGraph {
        &self.backward_graph
    }

    /// Number of forward inputs.
    pub fn num_inputs(&self) -> usize {
        self.num_inputs
    }

    /// Number of forward outputs.
    pub fn num_outputs(&self) -> usize {
        self.num_outputs
    }
}

// ---------------------------------------------------------------------------
// Integration with compile
// ---------------------------------------------------------------------------

/// Trace a forward function, generate backward, optimize both, and return
/// a compiled AOT function.
///
/// This is the equivalent of `torch.compile` with AOT autograd enabled.
///
/// # Arguments
///
/// * `forward_graph` — The traced forward IR graph.
/// * `config` — Optimization configuration.
/// * `needed_grads` — Which input gradients are needed (None = all).
///
/// # Returns
///
/// A `CompiledAotFunction` ready for forward+backward execution.
pub fn compile_aot_from_graph(
    forward_graph: &IrGraph,
    config: &OptimizationConfig,
    needed_grads: Option<&[bool]>,
) -> Result<CompiledAotFunction, JitError> {
    // Step 1: Apply optimization passes to the forward graph.
    let mut opt_forward = forward_graph.clone();
    optimize::optimize(&mut opt_forward, config);

    // Step 2: Trace backward from the optimized forward graph.
    let aot = aot_trace_from_graph(&opt_forward)?;

    // Step 3: Compile with dead code elimination and backward fusion.
    Ok(aot.compile(needed_grads))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{IrGraph, IrOpKind};

    // -----------------------------------------------------------------------
    // Helper: build a simple forward graph and verify backward generation
    // -----------------------------------------------------------------------

    /// Build a graph: y = x + x (single input, single output).
    fn build_add_self_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);
        g
    }

    /// Build a graph: y = a + b (two inputs, single output).
    fn build_add_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);
        g
    }

    /// Build a graph: y = a * b (two inputs, single output).
    fn build_mul_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![mul_outs[0]]);
        g
    }

    /// Build a graph: y = relu(x) (single input, single output).
    fn build_relu_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, relu_outs) = g.add_node(IrOpKind::Relu, vec![x], vec![vec![4]]);
        g.set_outputs(vec![relu_outs[0]]);
        g
    }

    /// Build a graph: y = sigmoid(x) (single input, single output).
    fn build_sigmoid_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, sig_outs) = g.add_node(IrOpKind::Sigmoid, vec![x], vec![vec![3]]);
        g.set_outputs(vec![sig_outs[0]]);
        g
    }

    /// Build a graph: y = tanh(x) (single input, single output).
    fn build_tanh_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, tanh_outs) = g.add_node(IrOpKind::Tanh, vec![x], vec![vec![3]]);
        g.set_outputs(vec![tanh_outs[0]]);
        g
    }

    /// Build a graph: y = neg(x) (single input, single output).
    fn build_neg_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, neg_outs) = g.add_node(IrOpKind::Neg, vec![x], vec![vec![3]]);
        g.set_outputs(vec![neg_outs[0]]);
        g
    }

    /// Build a graph: y = exp(x) (single input, single output).
    fn build_exp_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, exp_outs) = g.add_node(IrOpKind::Exp, vec![x], vec![vec![3]]);
        g.set_outputs(vec![exp_outs[0]]);
        g
    }

    /// Build a graph: y = log(x) (single input, single output).
    fn build_log_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, log_outs) = g.add_node(IrOpKind::Log, vec![x], vec![vec![3]]);
        g.set_outputs(vec![log_outs[0]]);
        g
    }

    /// Build a graph: y = sum(x) (single input, scalar output).
    fn build_sum_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![4]);
        let (_, sum_outs) = g.add_node(IrOpKind::Sum, vec![x], vec![vec![]]);
        g.set_outputs(vec![sum_outs[0]]);
        g
    }

    /// Build a graph: y = mm(a, b), a=[2,3], b=[3,2].
    fn build_mm_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![2, 3]);
        let b = g.add_input(vec![3, 2]);
        let (_, mm_outs) = g.add_node(IrOpKind::Mm, vec![a, b], vec![vec![2, 2]]);
        g.set_outputs(vec![mm_outs[0]]);
        g
    }

    /// Build a deeper graph: y = relu(a * b + a)
    fn build_chain_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![a, b], vec![vec![3]]);
        let (_, add_outs) =
            g.add_node(IrOpKind::Add, vec![mul_outs[0], a], vec![vec![3]]);
        let (_, relu_outs) =
            g.add_node(IrOpKind::Relu, vec![add_outs[0]], vec![vec![3]]);
        g.set_outputs(vec![relu_outs[0]]);
        g
    }

    /// Build: y = a - b
    fn build_sub_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, sub_outs) = g.add_node(IrOpKind::Sub, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![sub_outs[0]]);
        g
    }

    /// Build: y = a / b
    fn build_div_graph() -> IrGraph {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, div_outs) = g.add_node(IrOpKind::Div, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![div_outs[0]]);
        g
    }

    // -----------------------------------------------------------------------
    // Test: AOT trace of Add produces a valid backward graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_add_backward_graph_structure() {
        let fwd = build_add_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        // Forward: 2 inputs, 1 Add, 1 output.
        assert_eq!(aot.forward_graph.input_values.len(), 2);
        assert_eq!(aot.forward_graph.output_values.len(), 1);

        // Backward graph should have:
        // - At least 1 input (grad_output)
        // - Outputs for grad_a and grad_b
        assert!(
            aot.backward_graph.input_values.len() >= 1,
            "backward should have at least 1 input (grad_output)"
        );
        assert_eq!(
            aot.backward_graph.output_values.len(),
            2,
            "backward should produce gradients for both inputs"
        );
    }

    #[test]
    fn test_aot_trace_add_self_backward_has_correct_outputs() {
        let fwd = build_add_self_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        // x + x: backward should produce 1 gradient (for the single input x).
        assert_eq!(
            aot.backward_graph.output_values.len(),
            1,
            "x + x should produce 1 input gradient"
        );

        // Backward graph should have an Add node (accumulating 2 gradient contributions).
        let has_add = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Add));
        assert!(has_add, "x + x backward should have an Add to accumulate gradients");
    }

    // -----------------------------------------------------------------------
    // Test: Mul backward saves both inputs and creates Mul ops
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_mul_backward_structure() {
        let fwd = build_mul_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        // Mul backward needs both inputs saved.
        // Backward graph should have inputs for: grad_out + 2 saved tensors = 3.
        assert!(
            aot.backward_graph.input_values.len() >= 3,
            "mul backward needs grad_out + 2 saved inputs, got {}",
            aot.backward_graph.input_values.len()
        );

        // Should produce 2 output gradients.
        assert_eq!(aot.backward_graph.output_values.len(), 2);

        // Backward should contain Mul nodes (grad_a = grad * b, grad_b = grad * a).
        let mul_count = aot
            .backward_graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Mul))
            .count();
        assert!(
            mul_count >= 2,
            "mul backward should have at least 2 Mul nodes, got {mul_count}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Neg backward produces Neg ops
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_neg_backward() {
        let fwd = build_neg_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        // Backward of neg is neg(grad_out).
        assert_eq!(aot.backward_graph.output_values.len(), 1);

        let has_neg = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Neg));
        assert!(has_neg, "neg backward should contain a Neg op");
    }

    // -----------------------------------------------------------------------
    // Test: Exp backward produces Mul with saved output
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_exp_backward() {
        let fwd = build_exp_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // Should save the exp output.
        assert!(
            aot.backward_graph.input_values.len() >= 2,
            "exp backward needs grad_out + saved output"
        );

        // Should have a Mul node (grad_out * exp_out).
        let has_mul = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Mul));
        assert!(has_mul, "exp backward should have a Mul op");
    }

    // -----------------------------------------------------------------------
    // Test: Log backward produces Div with saved input
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_log_backward() {
        let fwd = build_log_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // Should have a Div node (grad_out / input).
        let has_div = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Div));
        assert!(has_div, "log backward should have a Div op");
    }

    // -----------------------------------------------------------------------
    // Test: Sigmoid backward structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_sigmoid_backward_structure() {
        let fwd = build_sigmoid_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // Sigmoid backward: grad * out * (1 - out)
        // Should have Sub, Mul operations.
        let has_sub = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Sub));
        let mul_count = aot
            .backward_graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Mul))
            .count();
        assert!(has_sub, "sigmoid backward should have a Sub op for (1 - out)");
        assert!(
            mul_count >= 2,
            "sigmoid backward should have at least 2 Mul ops, got {mul_count}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Tanh backward structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_tanh_backward_structure() {
        let fwd = build_tanh_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // Tanh backward: grad * (1 - tanh^2)
        // Should have Mul (for squaring), Sub, and Mul (for grad * ...).
        let mul_count = aot
            .backward_graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Mul))
            .count();
        assert!(
            mul_count >= 2,
            "tanh backward should have at least 2 Mul ops (square + grad mul), got {mul_count}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Relu backward structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_relu_backward_structure() {
        let fwd = build_relu_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // Relu backward should save the input and generate mask computation.
        // Ops: Relu, Abs, Add(abs,eps), Div(relu,abs+eps), Mul(grad,mask)
        assert!(
            aot.backward_graph.node_count() >= 5,
            "relu backward should have mask computation ops, got {} nodes",
            aot.backward_graph.node_count()
        );
    }

    // -----------------------------------------------------------------------
    // Test: Sum backward broadcasts gradient
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_sum_backward() {
        let fwd = build_sum_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // Sum backward should have a Mul(ones, grad_out) for broadcasting.
        let has_mul = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Mul));
        assert!(has_mul, "sum backward should have Mul for gradient expansion");

        // Should have a Constant node for the ones tensor.
        let has_const = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Constant { .. }));
        assert!(has_const, "sum backward should have ones constant");
    }

    // -----------------------------------------------------------------------
    // Test: MatMul/Mm backward structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_mm_backward_structure() {
        let fwd = build_mm_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        // Should produce 2 gradients (one per input).
        assert_eq!(aot.backward_graph.output_values.len(), 2);

        // Should have Transpose ops (for A^T and B^T).
        let transpose_count = aot
            .backward_graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Transpose))
            .count();
        assert!(
            transpose_count >= 2,
            "mm backward should have at least 2 Transpose ops, got {transpose_count}"
        );

        // Should have Mm ops (grad @ B^T and A^T @ grad).
        let mm_count = aot
            .backward_graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Mm))
            .count();
        assert!(
            mm_count >= 2,
            "mm backward should have at least 2 Mm ops, got {mm_count}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Sub backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_sub_backward() {
        let fwd = build_sub_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        assert_eq!(aot.backward_graph.output_values.len(), 2);

        // grad_b should be negated.
        let has_neg = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Neg));
        assert!(has_neg, "sub backward should have Neg for grad_b");
    }

    // -----------------------------------------------------------------------
    // Test: Div backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_div_backward() {
        let fwd = build_div_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        assert_eq!(aot.backward_graph.output_values.len(), 2);

        // Should have Div, Neg, Mul ops.
        let has_div = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Div));
        let has_neg = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Neg));
        assert!(has_div, "div backward should have Div op");
        assert!(has_neg, "div backward should have Neg op");
    }

    // -----------------------------------------------------------------------
    // Test: Chain graph backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_chain_backward() {
        let fwd = build_chain_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        // y = relu(a * b + a): 2 inputs, 1 output.
        assert_eq!(aot.backward_graph.output_values.len(), 2);

        // The backward graph should be non-trivial.
        assert!(
            aot.backward_graph.node_count() > 5,
            "chain backward should have multiple nodes, got {}",
            aot.backward_graph.node_count()
        );
    }

    // -----------------------------------------------------------------------
    // Test: Dead code elimination for backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_eliminate_dead_backward_ops_single_needed() {
        let fwd = build_mul_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        let mut backward = aot.backward_graph.clone();
        let original_count = backward.node_count();

        // Only need gradient for input 0, not input 1.
        eliminate_dead_backward_ops(&mut backward, &[0]);

        // Should have fewer nodes (ops for grad_b eliminated).
        assert!(
            backward.node_count() <= original_count,
            "DCE should not increase node count: {} <= {}",
            backward.node_count(),
            original_count
        );
        // Should only have 1 output now.
        assert_eq!(backward.output_values.len(), 1);
    }

    #[test]
    fn test_eliminate_dead_backward_ops_all_needed() {
        let fwd = build_mul_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        let mut backward = aot.backward_graph.clone();
        let node_count_before = backward.node_count();

        // Need both gradients.
        eliminate_dead_backward_ops(&mut backward, &[0, 1]);

        // Should keep all outputs.
        assert_eq!(backward.output_values.len(), 2);
        // Node count should not increase.
        assert!(backward.node_count() <= node_count_before);
    }

    #[test]
    fn test_eliminate_dead_backward_ops_none_needed() {
        let fwd = build_add_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        let mut backward = aot.backward_graph.clone();

        // No gradients needed — should eliminate everything except inputs.
        eliminate_dead_backward_ops(&mut backward, &[]);

        assert_eq!(backward.output_values.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Test: Backward optimization applies fusion
    // -----------------------------------------------------------------------

    #[test]
    fn test_optimize_backward_runs_without_error() {
        let fwd = build_relu_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        let mut backward = aot.backward_graph.clone();
        optimize_backward(&mut backward);

        // Should still have outputs.
        assert_eq!(backward.output_values.len(), 1);
        // Graph should still be valid.
        let topo = backward.topological_order();
        assert!(!topo.is_empty());
    }

    // -----------------------------------------------------------------------
    // Test: Compile AOT function
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_aot_all_grads() {
        let fwd = build_mul_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        let compiled = aot.compile(None);

        assert_eq!(compiled.num_inputs(), 2);
        assert_eq!(compiled.num_outputs(), 1);
        assert!(compiled.backward_graph().node_count() > 0);
    }

    #[test]
    fn test_compile_aot_partial_grads() {
        let fwd = build_mul_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        let compiled = aot.compile(Some(&[true, false]));

        assert_eq!(compiled.num_inputs(), 2);
        assert_eq!(compiled.num_outputs(), 1);

        // Backward should have fewer outputs since we only need grad for input 0.
        assert_eq!(compiled.backward_graph().output_values.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Test: compile_aot_from_graph integration
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_aot_from_graph_integration() {
        let fwd = build_chain_graph();
        let config = OptimizationConfig::default();

        let compiled =
            compile_aot_from_graph(&fwd, &config, None).unwrap();

        assert_eq!(compiled.num_inputs(), 2);
        assert_eq!(compiled.num_outputs(), 1);
        assert!(compiled.forward_graph().node_count() > 0);
        assert!(compiled.backward_graph().node_count() > 0);
    }

    #[test]
    fn test_compile_aot_from_graph_with_partial_grads() {
        let fwd = build_chain_graph();
        let config = OptimizationConfig {
            constant_folding: false,
            dead_code_elimination: false,
            operator_fusion: false,
            memory_planning: false,
        };

        let compiled = compile_aot_from_graph(
            &fwd,
            &config,
            Some(&[true, false]),
        )
        .unwrap();

        assert_eq!(compiled.num_inputs(), 2);
        // Only 1 backward output (grad for input 0).
        assert_eq!(compiled.backward_graph().output_values.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Test: Backward graph topological order is valid
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_graph_topological_order_valid() {
        let fwd = build_chain_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        let order = aot.backward_graph.topological_order();
        assert_eq!(order.len(), aot.backward_graph.node_count());

        // Build position map and verify all dependencies are satisfied.
        let pos = |id: IrNodeId| order.iter().position(|&n| n == id).unwrap();

        for node in &aot.backward_graph.nodes {
            for &input_val in &node.inputs {
                if let Some(producer) = aot
                    .backward_graph
                    .values
                    .iter()
                    .find(|v| v.id == input_val)
                    .and_then(|v| v.producer)
                {
                    assert!(
                        pos(producer) < pos(node.id),
                        "producer {:?} must come before consumer {:?}",
                        producer,
                        node.id
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test: Accessors on AotAutograd
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_autograd_accessors() {
        let fwd = build_add_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        assert_eq!(aot.num_inputs(), 2);
        assert_eq!(aot.num_outputs(), 1);
        assert_eq!(aot.forward_graph().input_values.len(), 2);
        assert_eq!(aot.backward_graph().output_values.len(), 2);
    }

    // -----------------------------------------------------------------------
    // Test: Backward of pure shape ops (transpose)
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_transpose_backward() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![2, 3]);
        let (_, t_outs) = g.add_node(IrOpKind::Transpose, vec![x], vec![vec![3, 2]]);
        g.set_outputs(vec![t_outs[0]]);

        let aot = aot_trace_from_graph(&g).unwrap();
        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // Transpose backward is another transpose.
        let has_transpose = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Transpose));
        assert!(has_transpose, "transpose backward should have Transpose op");
    }

    // -----------------------------------------------------------------------
    // Test: Sqrt backward structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_sqrt_backward() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, sqrt_outs) = g.add_node(IrOpKind::Sqrt, vec![x], vec![vec![3]]);
        g.set_outputs(vec![sqrt_outs[0]]);

        let aot = aot_trace_from_graph(&g).unwrap();
        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // sqrt backward: grad / (2 * output). Should have Add and Div.
        let has_add = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Add));
        let has_div = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Div));
        assert!(has_add, "sqrt backward should have Add for 2*output");
        assert!(has_div, "sqrt backward should have Div for grad/(2*output)");
    }

    // -----------------------------------------------------------------------
    // Test: Pow backward structure
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_pow_backward() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, pow_outs) = g.add_node(
            IrOpKind::Pow { exponent: 3.0 },
            vec![x],
            vec![vec![3]],
        );
        g.set_outputs(vec![pow_outs[0]]);

        let aot = aot_trace_from_graph(&g).unwrap();
        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // pow backward: grad * exp * input^(exp-1). Should have Pow and Mul.
        let has_pow = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Pow { .. }));
        let mul_count = aot
            .backward_graph
            .nodes
            .iter()
            .filter(|n| matches!(n.op, IrOpKind::Mul))
            .count();
        assert!(has_pow, "pow backward should have Pow op for input^(exp-1)");
        assert!(
            mul_count >= 2,
            "pow backward should have Mul ops for scaling, got {mul_count}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: Reshape backward restores original shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_aot_trace_reshape_backward() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![2, 3]);
        let (_, r_outs) = g.add_node(
            IrOpKind::Reshape {
                shape: vec![6],
            },
            vec![x],
            vec![vec![6]],
        );
        g.set_outputs(vec![r_outs[0]]);

        let aot = aot_trace_from_graph(&g).unwrap();
        assert_eq!(aot.backward_graph.output_values.len(), 1);

        // Reshape backward should reshape grad back to [2, 3].
        let has_reshape = aot
            .backward_graph
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Reshape { .. }));
        assert!(has_reshape, "reshape backward should have Reshape op");

        // Verify the reshape target is the original input shape [2, 3].
        let reshape_node = aot
            .backward_graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, IrOpKind::Reshape { .. }))
            .unwrap();
        if let IrOpKind::Reshape { shape } = &reshape_node.op {
            assert_eq!(shape, &[2, 3], "reshape backward should target original shape [2, 3]");
        }
    }

    // -----------------------------------------------------------------------
    // Test: CompiledAotFunction accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_compiled_aot_function_accessors() {
        let fwd = build_add_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();
        let compiled = aot.compile(None);

        assert_eq!(compiled.num_inputs(), 2);
        assert_eq!(compiled.num_outputs(), 1);
        assert!(compiled.forward_graph().node_count() > 0);
        assert!(compiled.backward_graph().node_count() > 0);
    }

    // -----------------------------------------------------------------------
    // Test: Multiple output gradients with DCE
    // -----------------------------------------------------------------------

    #[test]
    fn test_dce_preserves_needed_gradient_computation() {
        // y = a * b: need grad for 'a' only.
        let fwd = build_mul_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        let _original_backward = aot.backward_graph.clone();
        let mut pruned = aot.backward_graph.clone();
        eliminate_dead_backward_ops(&mut pruned, &[0]);

        // The pruned graph should still be valid topologically.
        let topo = pruned.topological_order();
        assert_eq!(topo.len(), pruned.node_count());

        // Should still have Mul ops (needed for grad_a = grad * b).
        let has_mul = pruned
            .nodes
            .iter()
            .any(|n| matches!(n.op, IrOpKind::Mul));
        assert!(
            has_mul,
            "pruned backward should still have Mul for needed gradient"
        );
    }

    // -----------------------------------------------------------------------
    // Test: optimization does not break backward graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_optimization_preserves_validity() {
        let fwd = build_chain_graph();
        let aot = aot_trace_from_graph(&fwd).unwrap();

        let mut backward = aot.backward_graph.clone();
        let output_count_before = backward.output_values.len();

        optimize_backward(&mut backward);

        // Outputs should still be present.
        assert_eq!(backward.output_values.len(), output_count_before);

        // Topological order should still be valid.
        let topo = backward.topological_order();
        assert_eq!(topo.len(), backward.node_count());
    }
}
