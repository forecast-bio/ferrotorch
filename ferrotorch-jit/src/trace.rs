//! Post-hoc tracer: runs a forward function with real inputs, then walks the
//! autograd graph (via `GradFn::inputs()`) to reconstruct an [`IrGraph`].
//!
//! This is the simplest tracing strategy: no proxy tensors, no interpreter.
//! The user-provided function executes normally, building an autograd graph.
//! We then traverse that graph from the output tensor back to the leaf inputs
//! and emit one [`IrNode`] per operation.

use std::collections::{HashMap, VecDeque};

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::{Tensor, TensorId};

use crate::graph::{IrGraph, IrOpKind, IrValueId};

// ---------------------------------------------------------------------------
// Name -> IrOpKind mapping
// ---------------------------------------------------------------------------

/// Map the `GradFn::name()` string to the corresponding IR operation.
///
/// Shape-dependent ops (Reshape) use the tensor's output shape as a fallback
/// since the backward node does not expose the target shape directly.
fn map_name_to_op(name: &str, output_shape: &[usize]) -> FerrotorchResult<IrOpKind> {
    match name {
        // Arithmetic
        "AddBackward" => Ok(IrOpKind::Add),
        "SubBackward" => Ok(IrOpKind::Sub),
        "MulBackward" => Ok(IrOpKind::Mul),
        "DivBackward" => Ok(IrOpKind::Div),
        "NegBackward" => Ok(IrOpKind::Neg),
        "PowBackward" => Ok(IrOpKind::Pow { exponent: 0.0 }),
        "SqrtBackward" => Ok(IrOpKind::Sqrt),
        "AbsBackward" => Ok(IrOpKind::Abs),

        // Reduction
        "SumBackward" => Ok(IrOpKind::Sum),
        "MeanBackward" => Ok(IrOpKind::Mean),
        "ProdBackward" => Ok(IrOpKind::Prod),

        // Linalg
        "MmBackward" => Ok(IrOpKind::Mm),
        "MatmulBackward" => Ok(IrOpKind::Matmul),
        "MvBackward" => Ok(IrOpKind::Mv),
        "DotBackward" => Ok(IrOpKind::Dot),
        "LinearFusedBackward" => Ok(IrOpKind::Linear),

        // Activation
        "ReluBackward" => Ok(IrOpKind::Relu),
        "SigmoidBackward" => Ok(IrOpKind::Sigmoid),
        "TanhBackward" => Ok(IrOpKind::Tanh),
        "GeluBackward" => Ok(IrOpKind::Gelu),
        "SiluBackward" => Ok(IrOpKind::Silu),
        "SoftmaxBackward" => Ok(IrOpKind::Softmax),
        "LogSoftmaxBackward" => Ok(IrOpKind::LogSoftmax),

        // Shape
        "ReshapeBackward" => Ok(IrOpKind::Reshape {
            shape: output_shape.iter().map(|&d| d as isize).collect(),
        }),
        "FlattenBackward" => Ok(IrOpKind::Flatten),
        "TransposeBackward" => Ok(IrOpKind::Transpose),

        other => Err(FerrotorchError::InvalidArgument {
            message: format!("unsupported operation in tracer: {other}"),
        }),
    }
}

// ---------------------------------------------------------------------------
// Internal: operation record captured during BFS
// ---------------------------------------------------------------------------

/// A single autograd operation discovered during the backward BFS.
struct OpRecord {
    /// `TensorId` of the tensor produced by this operation.
    output_id: TensorId,
    /// Shape of the output tensor.
    output_shape: Vec<usize>,
    /// `GradFn::name()` for this operation.
    name: &'static str,
    /// `TensorIds` of the inputs to this operation (from `GradFn::inputs()`).
    input_ids: Vec<TensorId>,
    /// Shapes of the input tensors.
    input_shapes: Vec<Vec<usize>>,
    /// Whether each input is a leaf (no `grad_fn`).
    input_is_leaf: Vec<bool>,
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Trace a function by executing it with real inputs, then walking the
/// resulting autograd graph to build an [`IrGraph`].
///
/// # Arguments
///
/// * `f` - The function to trace. It receives a slice of tensors and must
///   return a single output tensor.
/// * `example_inputs` - Concrete tensors used for a single forward pass.
///   At least one must have `requires_grad = true` so that an autograd
///   graph is constructed.
///
/// # Returns
///
/// An `IrGraph` whose `input_values` correspond to the example inputs (in
/// order) and whose `output_values` contain the single traced output.
///
/// # Errors
///
/// Returns an error if:
/// - The forward function fails.
/// - The output tensor has no `grad_fn` (no autograd graph was built).
/// - An operation name is not recognised by the tracer.
pub fn trace<T, F>(f: F, example_inputs: &[Tensor<T>]) -> FerrotorchResult<IrGraph>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
{
    // Step 1: Execute the forward function to build the autograd graph.
    let output = f(example_inputs)?;

    // Step 2: If output has no grad_fn, there is no graph to trace.
    if output.grad_fn().is_none() {
        return Err(FerrotorchError::InvalidArgument {
            message: "traced function produced a tensor with no grad_fn; \
                      ensure at least one input has requires_grad=true and \
                      gradient tracking is enabled"
                .into(),
        });
    }

    // Step 3: Collect the set of input TensorIds for index lookup.
    let input_ids: HashMap<TensorId, usize> = example_inputs
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id(), i))
        .collect();

    // Step 4: BFS from the output back through grad_fn().inputs().
    //
    // We traverse in reverse (output -> leaves) and record every operation
    // we encounter.  After traversal we build the IR in *forward* order by
    // reversing the collected operations.
    //
    // `tensor_map` keeps Tensor objects alive so we can call grad_fn() on
    // them. Cloning a Tensor is cheap (Arc bump).
    let mut tensor_map: HashMap<TensorId, Tensor<T>> = HashMap::new();
    tensor_map.insert(output.id(), output.clone());
    for t in example_inputs {
        tensor_map.insert(t.id(), t.clone());
    }

    let mut ops: Vec<OpRecord> = Vec::new();
    let mut visited: HashMap<TensorId, ()> = HashMap::new();
    let mut queue: VecDeque<TensorId> = VecDeque::new();

    queue.push_back(output.id());

    while let Some(tid) = queue.pop_front() {
        if visited.contains_key(&tid) {
            continue;
        }
        visited.insert(tid, ());

        // Clone the tensor out of the map so the immutable borrow on
        // `tensor_map` is released before we mutate it below.
        let tensor = match tensor_map.get(&tid) {
            Some(t) => t.clone(),
            None => continue,
        };

        // Leaf tensors (no grad_fn) are graph inputs — skip.
        let grad_fn = match tensor.grad_fn() {
            Some(gf) => gf,
            None => continue,
        };

        let inputs = grad_fn.inputs();

        let mut child_ids = Vec::with_capacity(inputs.len());
        let mut child_shapes = Vec::with_capacity(inputs.len());
        let mut child_is_leaf = Vec::with_capacity(inputs.len());

        for child in &inputs {
            let cid = child.id();
            child_ids.push(cid);
            child_shapes.push(child.shape().to_vec());
            child_is_leaf.push(child.grad_fn().is_none());

            tensor_map.entry(cid).or_insert_with(|| (*child).clone());
            if !visited.contains_key(&cid) {
                queue.push_back(cid);
            }
        }

        ops.push(OpRecord {
            output_id: tid,
            output_shape: tensor.shape().to_vec(),
            name: grad_fn.name(),
            input_ids: child_ids,
            input_shapes: child_shapes,
            input_is_leaf: child_is_leaf,
        });
    }

    // Step 5: Build the IR graph in forward (topological) order.
    //
    // The BFS collected operations from output towards leaves.  Reverse so
    // that producers appear before consumers.
    ops.reverse();

    let mut graph = IrGraph::new();
    let mut tensor_to_ir: HashMap<TensorId, IrValueId> = HashMap::new();

    // Create IR Input nodes for every leaf tensor discovered during BFS.
    // Explicit `example_inputs` get their positional index; closure-captured
    // leaves get indices after the explicit ones.
    let mut next_extra_input = example_inputs.len();

    for op in &ops {
        for (i, &cid) in op.input_ids.iter().enumerate() {
            if op.input_is_leaf[i] && !tensor_to_ir.contains_key(&cid) {
                let index = if let Some(&idx) = input_ids.get(&cid) {
                    idx
                } else {
                    let idx = next_extra_input;
                    next_extra_input += 1;
                    idx
                };

                let value_id = graph.add_input(op.input_shapes[i].clone());

                // `add_input` auto-assigns the index based on insertion
                // order. Override it to match the user-provided order.
                let last_node = graph.nodes.last_mut().unwrap();
                last_node.op = IrOpKind::Input { index };

                tensor_to_ir.insert(cid, value_id);
            }
        }
    }

    // Create IR nodes for each operation (in forward order).
    for op in &ops {
        let ir_inputs: Vec<IrValueId> = op
            .input_ids
            .iter()
            .map(|cid| {
                *tensor_to_ir.get(cid).unwrap_or_else(|| {
                    panic!(
                        "BUG: tensor {cid:?} not found in tensor_to_ir map \
                         during IR construction"
                    )
                })
            })
            .collect();

        let ir_op = map_name_to_op(op.name, &op.output_shape)?;

        let (_, out_ids) = graph.add_node(ir_op, ir_inputs, vec![op.output_shape.clone()]);

        tensor_to_ir.insert(op.output_id, out_ids[0]);
    }

    // Step 6: Mark the graph output.
    let output_ir =
        *tensor_to_ir
            .get(&output.id())
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "traced output tensor not found in IR value map".into(),
            })?;
    graph.set_outputs(vec![output_ir]);

    Ok(graph)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::IrNodeId;
    use ferrotorch_core::error::FerrotorchResult;
    use ferrotorch_core::grad_fns::activation::relu;
    use ferrotorch_core::grad_fns::arithmetic::{add, mul};
    use ferrotorch_core::grad_fns::linalg::mm_differentiable;
    use ferrotorch_core::grad_fns::reduction::sum;
    use ferrotorch_core::storage::TensorStorage;
    use ferrotorch_core::tensor::Tensor;

    /// Helper: create a 1-D f32 tensor with `requires_grad`.
    fn grad_vec(data: Vec<f32>) -> Tensor<f32> {
        let n = data.len();
        Tensor::from_storage(TensorStorage::cpu(data), vec![n], true)
            .unwrap()
            .requires_grad_(true)
    }

    /// Helper: create a 2-D f32 tensor with `requires_grad`.
    fn grad_mat(data: Vec<f32>, rows: usize, cols: usize) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), vec![rows, cols], true)
            .unwrap()
            .requires_grad_(true)
    }

    // -----------------------------------------------------------------------
    // Test: x + x -> 1 Add node, 1 Input node
    // -----------------------------------------------------------------------

    #[test]
    fn trace_add_self() {
        let x = grad_vec(vec![1.0, 2.0, 3.0]);

        let graph = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                add(&inputs[0], &inputs[0])
            },
            &[x],
        )
        .unwrap();

        // 1 Input + 1 Add = 2 nodes.
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.input_values.len(), 1);
        assert_eq!(graph.output_values.len(), 1);

        // The Add node should reference the single input twice.
        let add_node = graph
            .nodes
            .iter()
            .find(|n| matches!(n.op, IrOpKind::Add))
            .expect("should have an Add node");
        assert_eq!(add_node.inputs.len(), 2);
        assert_eq!(add_node.inputs[0], add_node.inputs[1]);
    }

    // -----------------------------------------------------------------------
    // Test: relu(x * y) -> Mul + Relu, 2 inputs
    // -----------------------------------------------------------------------

    #[test]
    fn trace_mul_relu() {
        let x = grad_vec(vec![1.0, 2.0, 3.0]);
        let y = grad_vec(vec![4.0, 5.0, 6.0]);

        let graph = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let product = mul(&inputs[0], &inputs[1])?;
                relu(&product)
            },
            &[x, y],
        )
        .unwrap();

        // 2 Inputs + 1 Mul + 1 Relu = 4 nodes.
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.input_values.len(), 2);
        assert_eq!(graph.output_values.len(), 1);

        // Verify both Mul and Relu nodes exist.
        assert!(graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Mul)));
        assert!(graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Relu)));
    }

    // -----------------------------------------------------------------------
    // Test: sum(mm(A, B)) -> Mm + Sum
    // -----------------------------------------------------------------------

    #[test]
    fn trace_mm_sum() {
        // A: [2, 3], B: [3, 2]
        let a = grad_mat(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);
        let b = grad_mat(vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0], 3, 2);

        let graph = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let product = mm_differentiable(&inputs[0], &inputs[1])?;
                sum(&product)
            },
            &[a, b],
        )
        .unwrap();

        // 2 Inputs + 1 Mm + 1 Sum = 4 nodes.
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.input_values.len(), 2);

        assert!(graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Mm)));
        assert!(graph.nodes.iter().any(|n| matches!(n.op, IrOpKind::Sum)));
    }

    // -----------------------------------------------------------------------
    // Test: no grad_fn (detached inputs) -> error
    // -----------------------------------------------------------------------

    #[test]
    fn trace_no_grad_fn_returns_error() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], false)
            .unwrap();

        let result = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                add(&inputs[0], &inputs[0])
            },
            &[x],
        );

        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("no grad_fn"),
            "error should mention no grad_fn: {msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Test: topological order is valid
    // -----------------------------------------------------------------------

    #[test]
    fn trace_topological_order_valid() {
        let x = grad_vec(vec![1.0, 2.0]);
        let y = grad_vec(vec![3.0, 4.0]);

        let graph = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let s = add(&inputs[0], &inputs[1])?;
                relu(&s)
            },
            &[x, y],
        )
        .unwrap();

        let order = graph.topological_order();
        assert_eq!(order.len(), graph.node_count());

        // Build position map.
        let pos = |id: IrNodeId| order.iter().position(|&n| n == id).unwrap();

        // Every node's inputs must have their producers earlier in the order.
        for node in &graph.nodes {
            for &input_val in &node.inputs {
                if let Some(producer) = graph
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
    // Test: node count for a deeper graph (a + b) * (a + b)
    // -----------------------------------------------------------------------

    #[test]
    fn trace_deeper_graph_node_count() {
        let a = grad_vec(vec![1.0, 2.0]);
        let b = grad_vec(vec![3.0, 4.0]);

        let graph = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let s = add(&inputs[0], &inputs[1])?;
                mul(&s, &s)
            },
            &[a, b],
        )
        .unwrap();

        // 2 Inputs + 1 Add + 1 Mul = 4 nodes.
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.input_values.len(), 2);
        assert_eq!(graph.output_values.len(), 1);
    }
}
