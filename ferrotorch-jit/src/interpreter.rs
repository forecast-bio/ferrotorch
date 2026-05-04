//! IR graph interpreter: executes an [`IrGraph`] using ferrotorch-core
//! tensor operations.
//!
//! This is the reference execution backend. It walks the graph in topological
//! order and dispatches each [`IrOpKind`] to the corresponding differentiable
//! (or non-differentiable) operation in ferrotorch-core. The interpreter is
//! useful for correctness testing, debugging, and as a baseline before a
//! compiled backend is available.

use std::collections::HashMap;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::grad_fns::{
    activation, arithmetic,
    linalg::{self as grad_linalg, mm_differentiable},
    reduction, shape, transcendental,
};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::graph::{IrGraph, IrNodeId, IrOpKind, IrValueId};

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

/// Execute an IR graph on the given input tensors, returning the output tensor.
///
/// # Arguments
///
/// * `graph` - The IR graph to interpret.
/// * `inputs` - Concrete tensors, one per graph input, in the same order as
///   `graph.input_values`.
///
/// # Errors
///
/// Returns an error if:
/// - The number of `inputs` does not match `graph.input_values.len()`.
/// - Any operation fails (shape mismatch, unsupported op, etc.).
/// - The graph has no output or more than one output.
///
/// # Example
///
/// ```ignore
/// let mut g = IrGraph::new();
/// let x = g.add_input(vec![3]);
/// let (_, outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
/// g.set_outputs(vec![outs[0]]);
///
/// let input = ferrotorch_core::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]).unwrap();
/// let result = interpret(&g, &[input]).unwrap();
/// // result == [2.0, 4.0, 6.0]
/// ```
pub fn interpret<T: Float>(graph: &IrGraph, inputs: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    // The single-output convenience wrapper requires exactly one graph
    // output. For multi-output graphs use interpret_multi.
    if graph.output_values.len() != 1 {
        if graph.output_values.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "interpret: graph has no outputs".into(),
            });
        }
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "interpret: graph has {} outputs; use interpret_multi for multi-output graphs",
                graph.output_values.len()
            ),
        });
    }
    let mut outs = interpret_multi(graph, inputs)?;
    Ok(outs.remove(0))
}

/// Execute an IR graph on the given input tensors and return all of its
/// output tensors in declaration order.
///
/// # Arguments
///
/// * `graph` - The IR graph to interpret. Must have at least one output
///   declared via [`crate::graph::IrGraph::set_outputs`].
/// * `inputs` - Concrete tensors, one per graph input, in the same order
///   as `graph.input_values`.
///
/// # Returns
///
/// A `Vec<Tensor<T>>` with one entry per `graph.output_values`, in the
/// same order they were registered. CL-368.
///
/// # Errors
///
/// - Input count does not match `graph.input_values.len()`.
/// - Any operation fails (shape mismatch, unsupported op, etc.).
/// - The graph has no output values.
/// - Any declared output value was never produced (graph is malformed).
pub fn interpret_multi<T: Float>(
    graph: &IrGraph,
    inputs: &[Tensor<T>],
) -> FerrotorchResult<Vec<Tensor<T>>> {
    // 1. Validate input count.
    if inputs.len() != graph.input_values.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "interpret: expected {} inputs, got {}",
                graph.input_values.len(),
                inputs.len()
            ),
        });
    }

    if graph.output_values.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "interpret: graph has no outputs".into(),
        });
    }

    // 2. Value storage: maps IrValueId -> Tensor<T>.
    let mut values: HashMap<IrValueId, Tensor<T>> = HashMap::new();

    // 3. Build a node-id -> node lookup.
    let node_map: HashMap<IrNodeId, &_> = graph.nodes.iter().map(|n| (n.id, n)).collect();

    // 4. Walk nodes in topological order.
    let topo = graph.topological_order();

    for node_id in topo {
        let node = node_map[&node_id];

        match &node.op {
            // ----- Inputs / Constants / Outputs -----
            IrOpKind::Input { index } => {
                let tensor = inputs[*index].clone();
                for &out_id in &node.outputs {
                    values.insert(out_id, tensor.clone());
                }
            }

            IrOpKind::Constant {
                data,
                shape: cshape,
            } => {
                let converted: Vec<T> = data
                    .iter()
                    .map(|&v| ferrotorch_core::numeric_cast::cast::<f64, T>(v))
                    .collect::<FerrotorchResult<Vec<T>>>()?;
                let tensor =
                    Tensor::from_storage(TensorStorage::cpu(converted), cshape.clone(), false)?;
                for &out_id in &node.outputs {
                    values.insert(out_id, tensor.clone());
                }
            }

            IrOpKind::Output => {
                // Output node: its single input is the graph output.
                // Just forward the value to the output slot.
                if let Some(&input_id) = node.inputs.first() {
                    let tensor = get_value(&values, input_id)?.clone();
                    set_outputs(&mut values, &node.outputs, tensor);
                }
            }

            // ----- Binary arithmetic -----
            IrOpKind::Add => {
                let (a, b) = get_binary_inputs(&values, &node.inputs)?;
                let result = arithmetic::add(a, b)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Sub => {
                let (a, b) = get_binary_inputs(&values, &node.inputs)?;
                let result = arithmetic::sub(a, b)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Mul => {
                let (a, b) = get_binary_inputs(&values, &node.inputs)?;
                let result = arithmetic::mul(a, b)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Div => {
                let (a, b) = get_binary_inputs(&values, &node.inputs)?;
                let result = arithmetic::div(a, b)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            // ----- Unary arithmetic -----
            IrOpKind::Neg => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = arithmetic::neg(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Sqrt => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = arithmetic::sqrt(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Abs => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = arithmetic::abs(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Pow { exponent } => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = arithmetic::pow(a, *exponent)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Exp => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = transcendental::exp(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Log => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = transcendental::log(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            // ----- Reductions -----
            IrOpKind::Sum => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = reduction::sum(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Mean => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = reduction::mean(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Prod => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = reduction::prod(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            // ----- Activations -----
            IrOpKind::Relu => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = activation::relu(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Sigmoid => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = activation::sigmoid(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Tanh => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = activation::tanh(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Gelu => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = activation::gelu(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Silu => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = activation::silu(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Softmax => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = activation::softmax(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::LogSoftmax => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = activation::log_softmax(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            // ----- Linear algebra -----
            IrOpKind::Mm => {
                let (a, b) = get_binary_inputs(&values, &node.inputs)?;
                let result = mm_differentiable(a, b)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Matmul => {
                let (a, b) = get_binary_inputs(&values, &node.inputs)?;
                let result = grad_linalg::matmul_differentiable(a, b)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Mv => {
                let (a, b) = get_binary_inputs(&values, &node.inputs)?;
                let result = grad_linalg::mv_differentiable(a, b)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Dot => {
                let (a, b) = get_binary_inputs(&values, &node.inputs)?;
                let result = grad_linalg::dot_differentiable(a, b)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Transpose => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = shape::transpose_2d(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Linear => {
                // Inputs: [input, weight] or [input, weight, bias]
                let input = get_value(&values, node.inputs[0])?;
                let weight = get_value(&values, node.inputs[1])?;
                let bias = if node.inputs.len() > 2 {
                    Some(get_value(&values, node.inputs[2])?)
                } else {
                    None
                };
                let result = grad_linalg::linear_fused(input, weight, bias)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            // ----- Shape ops -----
            IrOpKind::Reshape { shape: new_shape } => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = shape::reshape(a, new_shape)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Flatten => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = shape::flatten(a)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Squeeze { axis } => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = shape::squeeze(a, *axis as isize)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Unsqueeze { axis } => {
                let a = get_unary_input(&values, &node.inputs)?;
                let result = shape::unsqueeze(a, *axis as isize)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            IrOpKind::Cat { axis } => {
                // Cat needs all its inputs as a slice.
                let tensors: Vec<Tensor<T>> = node
                    .inputs
                    .iter()
                    .map(|id| get_value(&values, *id).cloned())
                    .collect::<FerrotorchResult<Vec<_>>>()?;
                let result = shape::cat(&tensors, *axis as isize)?;
                set_outputs(&mut values, &node.outputs, result);
            }

            // ----- Higher-order control flow (must be lowered) -----
            IrOpKind::Cond => {
                return Err(FerrotorchError::InvalidArgument {
                    message: "Cond IR nodes must be lowered before interpretation".into(),
                });
            }

            IrOpKind::Scan => {
                return Err(FerrotorchError::InvalidArgument {
                    message: "Scan IR nodes must be lowered before interpretation".into(),
                });
            }

            // ----- Fused elementwise -----
            IrOpKind::FusedElementwise { ops } => {
                // Start with the first input and apply each op sequentially.
                let mut current = get_unary_input(&values, &node.inputs)?.clone();
                for op in ops {
                    current = apply_elementwise_op(&current, op)?;
                }
                set_outputs(&mut values, &node.outputs, current);
            }

            IrOpKind::FusedLinearActivation { .. } | IrOpKind::FusedAttention { .. } => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "interpret: fused pattern op {:?} must be lowered before interpretation",
                        node.op
                    ),
                });
            }
        }
    }

    // 6. Return all graph outputs in declaration order.
    //
    // Use get + clone (not remove) so the same value can be listed as
    // multiple outputs without the second lookup failing. Tensor::clone
    // is an Arc clone so this is cheap.
    let mut results = Vec::with_capacity(graph.output_values.len());
    for &output_id in &graph.output_values {
        let t =
            values
                .get(&output_id)
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "interpret: output value {output_id:?} was not produced during execution"
                    ),
                })?;
        results.push(t);
    }
    Ok(results)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Retrieve a value from the map, returning an error if missing.
fn get_value<T: Float>(
    values: &HashMap<IrValueId, Tensor<T>>,
    id: IrValueId,
) -> FerrotorchResult<&Tensor<T>> {
    values
        .get(&id)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("interpret: value {id:?} not found"),
        })
}

/// Get the single input tensor for a unary operation.
fn get_unary_input<'a, T: Float>(
    values: &'a HashMap<IrValueId, Tensor<T>>,
    inputs: &[IrValueId],
) -> FerrotorchResult<&'a Tensor<T>> {
    if inputs.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "interpret: unary op has no inputs".into(),
        });
    }
    get_value(values, inputs[0])
}

/// Get both input tensors for a binary operation.
fn get_binary_inputs<'a, T: Float>(
    values: &'a HashMap<IrValueId, Tensor<T>>,
    inputs: &[IrValueId],
) -> FerrotorchResult<(&'a Tensor<T>, &'a Tensor<T>)> {
    if inputs.len() < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "interpret: binary op expected 2 inputs, got {}",
                inputs.len()
            ),
        });
    }
    Ok((get_value(values, inputs[0])?, get_value(values, inputs[1])?))
}

/// Store a result tensor in all output value slots of a node.
//
// `tensor` is intentionally taken by value: most callers move a freshly
// computed result here, and the function clones once per output slot.
// Taking `&Tensor` would require every caller to add `&`, churn for no
// behavioural change.
#[allow(clippy::needless_pass_by_value)]
fn set_outputs<T: Float>(
    values: &mut HashMap<IrValueId, Tensor<T>>,
    outputs: &[IrValueId],
    tensor: Tensor<T>,
) {
    for &out_id in outputs {
        values.insert(out_id, tensor.clone());
    }
}

/// Apply a single elementwise operation (used by `FusedElementwise`).
fn apply_elementwise_op<T: Float>(input: &Tensor<T>, op: &IrOpKind) -> FerrotorchResult<Tensor<T>> {
    match op {
        IrOpKind::Neg => arithmetic::neg(input),
        IrOpKind::Sqrt => arithmetic::sqrt(input),
        IrOpKind::Abs => arithmetic::abs(input),
        IrOpKind::Pow { exponent } => arithmetic::pow(input, *exponent),
        IrOpKind::Relu => activation::relu(input),
        IrOpKind::Sigmoid => activation::sigmoid(input),
        IrOpKind::Tanh => activation::tanh(input),
        IrOpKind::Gelu => activation::gelu(input),
        IrOpKind::Silu => activation::silu(input),
        IrOpKind::Exp => transcendental::exp(input),
        IrOpKind::Log => transcendental::log(input),
        _ => Err(FerrotorchError::InvalidArgument {
            message: format!("interpret: unsupported op in FusedElementwise: {op:?}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{IrGraph, IrOpKind};

    /// Helper: create a 1-D f32 tensor (no gradient tracking).
    fn tensor_1d(data: &[f32]) -> Tensor<f32> {
        ferrotorch_core::from_vec(data.to_vec(), &[data.len()]).unwrap()
    }

    /// Helper: create a 2-D f32 tensor (no gradient tracking).
    fn tensor_2d(data: &[f32], rows: usize, cols: usize) -> Tensor<f32> {
        ferrotorch_core::from_vec(data.to_vec(), &[rows, cols]).unwrap()
    }

    /// Assert two f32 slices are elementwise close.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: got {a}, expected {e} (diff {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Test 1: input + input
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpret_add_self() {
        // Graph: y = x + x
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let result = interpret::<f32>(&g, &[input]).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_close(result.data().unwrap(), &[2.0, 4.0, 6.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 2: mm(A, B) and verify against direct mm
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpret_mm() {
        // Graph: C = mm(A, B) where A is [2,3], B is [3,2]
        let mut g = IrGraph::new();
        let a_id = g.add_input(vec![2, 3]);
        let b_id = g.add_input(vec![3, 2]);
        let (_, mm_outs) = g.add_node(IrOpKind::Mm, vec![a_id, b_id], vec![vec![2, 2]]);
        g.set_outputs(vec![mm_outs[0]]);

        // A = [[1, 2, 3], [4, 5, 6]]
        let a = tensor_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        // B = [[7, 8], [9, 10], [11, 12]]
        let b = tensor_2d(&[7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 3, 2);

        let result = interpret::<f32>(&g, &[a.clone(), b.clone()]).unwrap();

        // Direct computation for reference.
        let direct = ferrotorch_core::grad_fns::linalg::mm_differentiable(&a, &b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_close(result.data().unwrap(), direct.data().unwrap(), 1e-5);

        // Expected: [[1*7+2*9+3*11, 1*8+2*10+3*12], [4*7+5*9+6*11, 4*8+5*10+6*12]]
        //         = [[58, 64], [139, 154]]
        assert_close(result.data().unwrap(), &[58.0, 64.0, 139.0, 154.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Test 3: graph with constants
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpret_constant_add() {
        // Graph: y = x + constant([10, 20, 30])
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let c = g.add_constant(vec![10.0, 20.0, 30.0], vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, c], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let result = interpret::<f32>(&g, &[input]).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_close(result.data().unwrap(), &[11.0, 22.0, 33.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 4: fused elementwise chain
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpret_fused_elementwise() {
        // Graph: y = relu(neg(x))
        // For x = [-1, 2, -3]: neg -> [1, -2, 3], relu -> [1, 0, 3]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, fused_outs) = g.add_node(
            IrOpKind::FusedElementwise {
                ops: vec![IrOpKind::Neg, IrOpKind::Relu],
            },
            vec![x],
            vec![vec![3]],
        );
        g.set_outputs(vec![fused_outs[0]]);

        let input = tensor_1d(&[-1.0, 2.0, -3.0]);
        let result = interpret::<f32>(&g, &[input]).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_close(result.data().unwrap(), &[1.0, 0.0, 3.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 5: input count mismatch error
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpret_input_count_mismatch() {
        // Graph expects 2 inputs, but we provide 1.
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let err = interpret::<f32>(&g, &[input]);
        assert!(err.is_err());

        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("expected 2 inputs, got 1"),
            "unexpected error message: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Additional: chain of ops (sub + pow + sqrt)
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpret_chain_sub_pow_sqrt() {
        // Graph: y = sqrt(pow(x - c, 2))  where c = [1, 1, 1]
        // This computes |x - c| for non-negative squared values.
        // x = [4, 1, 5], c = [1, 1, 1]
        // x - c = [3, 0, 4]
        // pow(_, 2) = [9, 0, 16]
        // sqrt = [3, 0, 4]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let c = g.add_constant(vec![1.0, 1.0, 1.0], vec![3]);
        let (_, sub_outs) = g.add_node(IrOpKind::Sub, vec![x, c], vec![vec![3]]);
        let (_, pow_outs) = g.add_node(
            IrOpKind::Pow { exponent: 2.0 },
            vec![sub_outs[0]],
            vec![vec![3]],
        );
        let (_, sqrt_outs) = g.add_node(IrOpKind::Sqrt, vec![pow_outs[0]], vec![vec![3]]);
        g.set_outputs(vec![sqrt_outs[0]]);

        let input = tensor_1d(&[4.0, 1.0, 5.0]);
        let result = interpret::<f32>(&g, &[input]).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_close(result.data().unwrap(), &[3.0, 0.0, 4.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Additional: no outputs error
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpret_no_outputs_error() {
        let mut g = IrGraph::new();
        let _ = g.add_input(vec![3]);
        // No outputs set.

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let err = interpret::<f32>(&g, &[input]);
        assert!(err.is_err());

        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("no outputs"),
            "unexpected error message: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Multi-output graphs (CL-368).
    //
    // The new interpret_multi entry point supports graphs with more than
    // one output. The legacy interpret() entry point still requires
    // exactly one output and returns a clear error pointing users at
    // interpret_multi for multi-output graphs.
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpret_multi_two_outputs_returns_both() {
        // y0 = x + x = 2x
        // y1 = x * x = x^2
        // Outputs: [y0, y1]
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, sum_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![sum_outs[0], mul_outs[0]]);

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let outs = interpret_multi::<f32>(&g, &[input]).unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].data().unwrap(), &[2.0, 4.0, 6.0]);
        assert_eq!(outs[1].data().unwrap(), &[1.0, 4.0, 9.0]);
    }

    #[test]
    fn test_interpret_single_output_via_multi() {
        // interpret_multi must also work for single-output graphs.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let outs = interpret_multi::<f32>(&g, &[input]).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].data().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_interpret_multi_same_value_listed_twice() {
        // A single intermediate value can appear multiple times in the
        // output list. interpret_multi should clone (not consume) so the
        // second lookup succeeds.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![2]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![2]]);
        g.set_outputs(vec![add_outs[0], add_outs[0]]);

        let input = tensor_1d(&[5.0, 7.0]);
        let outs = interpret_multi::<f32>(&g, &[input]).unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].data().unwrap(), &[10.0, 14.0]);
        assert_eq!(outs[1].data().unwrap(), &[10.0, 14.0]);
    }

    #[test]
    fn test_interpret_legacy_errors_on_multi_output_graph() {
        // The single-output `interpret` should refuse a multi-output
        // graph and point users at `interpret_multi`.
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, sum_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![sum_outs[0], mul_outs[0]]);

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let err = interpret::<f32>(&g, &[input]);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(
            msg.contains("interpret_multi"),
            "expected hint at interpret_multi: {msg}"
        );
    }

    #[test]
    fn test_interpret_multi_independent_chains() {
        // Two independent computations sharing the same input.
        // y0 = (x + x) - x   = x
        // y1 = (x * x) + x   = x^2 + x
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        let (_, sub_outs) = g.add_node(IrOpKind::Sub, vec![add_outs[0], x], vec![vec![3]]);
        let (_, mul_outs) = g.add_node(IrOpKind::Mul, vec![x, x], vec![vec![3]]);
        let (_, plus_outs) = g.add_node(IrOpKind::Add, vec![mul_outs[0], x], vec![vec![3]]);
        g.set_outputs(vec![sub_outs[0], plus_outs[0]]);

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let outs = interpret_multi::<f32>(&g, &[input]).unwrap();
        assert_eq!(outs.len(), 2);
        // y0 = x: [1, 2, 3]
        assert_eq!(outs[0].data().unwrap(), &[1.0, 2.0, 3.0]);
        // y1 = x^2 + x: [1+1, 4+2, 9+3] = [2, 6, 12]
        assert_eq!(outs[1].data().unwrap(), &[2.0, 6.0, 12.0]);
    }
}
