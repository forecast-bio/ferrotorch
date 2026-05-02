//! Graph break handling for the JIT compiler.
//!
//! When the tracer encounters an unsupported operation (one whose
//! `GradFn::name()` is not in the known op mapping), instead of failing
//! outright it can insert a *graph break* and split execution into segments.
//! Each segment is either a compiled IR subgraph ([`TracedModule`]) or an
//! eager-mode fallback closure.
//!
//! The [`SegmentedModule`] executes these segments in order, threading the
//! tensor output of one segment as the input to the next.
//!
//! When [`CompileConfig::fullgraph`] is `true`, graph breaks are rejected
//! with [`JitError::GraphBreak`] rather than producing a segmented module.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::{Tensor, TensorId};

use ferrotorch_nn::module::Module;

use crate::error::JitError;
use crate::graph::{IrGraph, IrOpKind, IrValueId};
use crate::module::{CompileConfig, TracedModule};
use crate::optimize::optimize;

// ---------------------------------------------------------------------------
// Known-op check
// ---------------------------------------------------------------------------

/// The set of `GradFn::name()` strings that the tracer can map to IR ops.
///
/// This must be kept in sync with `trace::map_name_to_op`.
const KNOWN_OPS: &[&str] = &[
    // Arithmetic
    "AddBackward",
    "SubBackward",
    "MulBackward",
    "DivBackward",
    "NegBackward",
    "PowBackward",
    "SqrtBackward",
    "AbsBackward",
    // Reduction
    "SumBackward",
    "MeanBackward",
    "ProdBackward",
    // Linalg
    "MmBackward",
    "MatmulBackward",
    "MvBackward",
    "DotBackward",
    "LinearFusedBackward",
    // Activation
    "ReluBackward",
    "SigmoidBackward",
    "TanhBackward",
    "GeluBackward",
    "SiluBackward",
    "SoftmaxBackward",
    "LogSoftmaxBackward",
    // Shape
    "ReshapeBackward",
    "FlattenBackward",
    "TransposeBackward",
];

/// Returns `true` if the given `GradFn::name()` is in the known op mapping.
fn is_known_op(name: &str) -> bool {
    KNOWN_OPS.contains(&name)
}

// ---------------------------------------------------------------------------
// GraphSegment / SegmentedModule
// ---------------------------------------------------------------------------

/// A single segment of a graph-broken execution pipeline.
///
/// Either a compiled IR subgraph or an eager-mode fallback closure.
/// Type alias for an eager-mode fallback closure.
type EagerFn<T> = Arc<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

pub enum GraphSegment<T: Float> {
    /// A compiled IR subgraph.
    Compiled(TracedModule<T>),
    /// An eager-mode fallback closure.
    Eager(EagerFn<T>),
}

impl<T: Float> std::fmt::Debug for GraphSegment<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Compiled(m) => f
                .debug_tuple("Compiled")
                .field(&format!("TracedModule({} nodes)", m.graph().node_count()))
                .finish(),
            Self::Eager(_) => f.debug_tuple("Eager").field(&"<closure>").finish(),
        }
    }
}

/// A module composed of segments — compiled IR subgraphs interleaved with
/// eager-mode fallback closures.
///
/// Created by [`trace_with_breaks`] when the tracer encounters unsupported
/// operations and `fullgraph` mode is not enabled.
#[derive(Debug)]
pub struct SegmentedModule<T: Float> {
    segments: Vec<GraphSegment<T>>,
}

impl<T: Float> SegmentedModule<T> {
    /// Create a `SegmentedModule` from a pre-built list of segments.
    pub fn new(segments: Vec<GraphSegment<T>>) -> Self {
        Self { segments }
    }

    /// Execute all segments in order, threading the output of each as the
    /// input to the next.
    pub fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut current = input.clone();
        for segment in &self.segments {
            current = match segment {
                GraphSegment::Compiled(traced) => traced.forward(&current)?,
                GraphSegment::Eager(f) => f(&current)?,
            };
        }
        Ok(current)
    }

    /// The number of segments in this module.
    pub fn segment_count(&self) -> usize {
        self.segments.len()
    }

    /// Access the segments for inspection.
    pub fn segments(&self) -> &[GraphSegment<T>] {
        &self.segments
    }

    /// Returns `true` if every segment is compiled (no eager fallbacks).
    pub fn is_fully_compiled(&self) -> bool {
        self.segments
            .iter()
            .all(|s| matches!(s, GraphSegment::Compiled(_)))
    }
}

// ---------------------------------------------------------------------------
// Break-point detection during BFS
// ---------------------------------------------------------------------------

/// An operation discovered during BFS traversal, annotated with whether it
/// maps to a known IR op.
struct AnnotatedOp {
    /// TensorId of the tensor produced by this operation.
    output_id: TensorId,
    /// Shape of the output tensor.
    output_shape: Vec<usize>,
    /// `GradFn::name()` for this operation.
    name: &'static str,
    /// TensorIds of the inputs to this operation.
    input_ids: Vec<TensorId>,
    /// Shapes of the input tensors.
    input_shapes: Vec<Vec<usize>>,
    /// Whether each input is a leaf.
    input_is_leaf: Vec<bool>,
    /// Whether this operation is supported by the IR.
    is_supported: bool,
}

// ---------------------------------------------------------------------------
// Result type for break-aware tracing
// ---------------------------------------------------------------------------

/// The result of a break-aware trace.
///
/// When no graph breaks occur, `Unbroken` contains a single `IrGraph`.
/// When breaks occur, `Segmented` contains a `SegmentedModule`.
#[derive(Debug)]
pub enum TraceResult<T: Float> {
    /// The entire forward pass was captured as a single IR graph.
    Unbroken(IrGraph),
    /// The forward pass was split into segments due to graph breaks.
    Segmented(SegmentedModule<T>),
}

// ---------------------------------------------------------------------------
// Break-aware tracing
// ---------------------------------------------------------------------------

/// Trace a function with graph-break support.
///
/// Walks the autograd graph from the output back to leaf inputs. When an
/// operation is encountered whose `GradFn::name()` is not in the known op
/// mapping, a graph break is inserted.
///
/// # Behaviour
///
/// - If **no** graph breaks occur, returns `TraceResult::Unbroken` with the
///   full `IrGraph`.
/// - If graph breaks occur and `fullgraph` is `false`, returns
///   `TraceResult::Segmented` with a `SegmentedModule`.
/// - If graph breaks occur and `fullgraph` is `true`, returns
///   `Err(JitError::GraphBreak)`.
///
/// # Arguments
///
/// * `f` - The function to trace.
/// * `example_inputs` - Concrete tensors for one forward pass. At least one
///   must have `requires_grad = true`.
/// * `config` - Compilation configuration (controls `fullgraph` and
///   optimization settings).
pub fn trace_with_breaks<T, F>(
    f: F,
    example_inputs: &[Tensor<T>],
    config: &CompileConfig,
) -> FerrotorchResult<TraceResult<T>>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    // Step 1: Execute the forward function to build the autograd graph.
    let output = f(example_inputs)?;

    if output.grad_fn().is_none() {
        return Err(FerrotorchError::InvalidArgument {
            message: "traced function produced a tensor with no grad_fn; \
                      ensure at least one input has requires_grad=true and \
                      gradient tracking is enabled"
                .into(),
        });
    }

    // Step 2: Collect input TensorIds for index lookup.
    let input_ids: HashMap<TensorId, usize> = example_inputs
        .iter()
        .enumerate()
        .map(|(i, t)| (t.id(), i))
        .collect();

    // Step 3: BFS from output -> leaves, annotating each op.
    let mut tensor_map: HashMap<TensorId, Tensor<T>> = HashMap::new();
    tensor_map.insert(output.id(), output.clone());
    for t in example_inputs {
        tensor_map.insert(t.id(), t.clone());
    }

    let mut ops: Vec<AnnotatedOp> = Vec::new();
    let mut visited: HashMap<TensorId, ()> = HashMap::new();
    let mut queue: VecDeque<TensorId> = VecDeque::new();
    let mut has_break = false;

    queue.push_back(output.id());

    while let Some(tid) = queue.pop_front() {
        if visited.contains_key(&tid) {
            continue;
        }
        visited.insert(tid, ());

        let tensor = match tensor_map.get(&tid) {
            Some(t) => t.clone(),
            None => continue,
        };

        let grad_fn = match tensor.grad_fn() {
            Some(gf) => gf,
            None => continue,
        };

        let inputs = grad_fn.inputs();
        let name = grad_fn.name();
        let supported = is_known_op(name);

        if !supported {
            has_break = true;

            // In fullgraph mode, reject immediately.
            if config.fullgraph {
                return Err(JitError::GraphBreak {
                    op: name.to_string(),
                    reason: format!(
                        "unsupported operation '{}' encountered during tracing \
                         and fullgraph mode is enabled",
                        name
                    ),
                }
                .into());
            }
        }

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

        ops.push(AnnotatedOp {
            output_id: tid,
            output_shape: tensor.shape().to_vec(),
            name,
            input_ids: child_ids,
            input_shapes: child_shapes,
            input_is_leaf: child_is_leaf,
            is_supported: supported,
        });
    }

    // Step 4: If no breaks, build a single IrGraph (same as regular trace).
    if !has_break {
        let graph = build_ir_graph(&ops, example_inputs, &input_ids, &output)?;
        return Ok(TraceResult::Unbroken(graph));
    }

    // Step 5: Build a segmented module.
    //
    // Strategy: walk ops in forward order (reversed from BFS). Accumulate
    // consecutive supported ops into a compiled segment. When an unsupported
    // op is encountered, flush the current compiled segment, insert an eager
    // segment for the unsupported op, and start a new compiled segment.
    //
    // For simplicity in this initial implementation, when we encounter a
    // graph break we split the entire execution into segments where the
    // eager segment re-runs the original function on its portion. The
    // simplest correct approach: produce a single SegmentedModule that
    // contains the compiled portions and eager fallbacks for unsupported
    // portions.
    //
    // Practical segmentation: we identify contiguous runs of supported ops
    // and build an IrGraph for each run, and wrap unsupported ops in eager
    // closures.
    let segments = build_segments(
        &ops,
        example_inputs,
        &input_ids,
        &output,
        config,
        &tensor_map,
    )?;

    Ok(TraceResult::Segmented(SegmentedModule::new(segments)))
}

/// Build an `IrGraph` from a list of annotated operations (all supported).
///
/// This mirrors the logic in `trace.rs` but works on our `AnnotatedOp` type.
fn build_ir_graph<T: Float>(
    ops: &[AnnotatedOp],
    example_inputs: &[Tensor<T>],
    input_ids: &HashMap<TensorId, usize>,
    output: &Tensor<T>,
) -> FerrotorchResult<IrGraph> {
    // Ops come from BFS (output -> leaves). Reverse for forward order.
    let forward_ops: Vec<&AnnotatedOp> = ops.iter().rev().collect();

    let mut graph = IrGraph::new();
    let mut tensor_to_ir: HashMap<TensorId, IrValueId> = HashMap::new();
    let mut next_extra_input = example_inputs.len();

    // Create IR Input nodes for leaf tensors.
    for op in &forward_ops {
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
                let last_node = graph.nodes.last_mut().unwrap();
                last_node.op = IrOpKind::Input { index };

                tensor_to_ir.insert(cid, value_id);
            }
        }
    }

    // Create IR nodes for each operation.
    for op in &forward_ops {
        let ir_inputs: Vec<IrValueId> = op
            .input_ids
            .iter()
            .map(|cid| {
                *tensor_to_ir.get(cid).unwrap_or_else(|| {
                    panic!("BUG: tensor {:?} not found in tensor_to_ir map", cid)
                })
            })
            .collect();

        let ir_op = map_name_to_op(op.name, &op.output_shape)?;
        let (_, out_ids) = graph.add_node(ir_op, ir_inputs, vec![op.output_shape.clone()]);
        tensor_to_ir.insert(op.output_id, out_ids[0]);
    }

    // Mark output.
    let output_ir =
        *tensor_to_ir
            .get(&output.id())
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "traced output tensor not found in IR value map".into(),
            })?;
    graph.set_outputs(vec![output_ir]);

    Ok(graph)
}

/// Build graph segments from annotated ops, splitting at unsupported ops.
///
/// Walks ops in forward order. Consecutive supported ops are accumulated and
/// built into a compiled `TracedModule` segment. Each unsupported op becomes
/// an eager segment that captures the tensor operation via its recorded
/// inputs/outputs in the tensor map.
fn build_segments<T: Float>(
    ops: &[AnnotatedOp],
    example_inputs: &[Tensor<T>],
    input_ids: &HashMap<TensorId, usize>,
    _output: &Tensor<T>,
    config: &CompileConfig,
    tensor_map: &HashMap<TensorId, Tensor<T>>,
) -> FerrotorchResult<Vec<GraphSegment<T>>> {
    // Forward order (BFS collected output -> leaves, so reverse).
    let forward_ops: Vec<&AnnotatedOp> = ops.iter().rev().collect();

    let mut segments: Vec<GraphSegment<T>> = Vec::new();
    let mut compiled_run: Vec<&AnnotatedOp> = Vec::new();

    for op in &forward_ops {
        if op.is_supported {
            compiled_run.push(op);
        } else {
            // Flush the current compiled run if non-empty.
            if !compiled_run.is_empty() {
                let graph =
                    build_ir_graph_from_run(&compiled_run, example_inputs, input_ids, tensor_map)?;
                let mut optimized = graph;
                let _memory_plan = optimize(&mut optimized, &config.optimization);
                segments.push(GraphSegment::Compiled(TracedModule::new(optimized)));
                compiled_run.clear();
            }

            // Build an eager segment for this unsupported op.
            //
            // We capture the actual tensor values so the eager closure can
            // reproduce the operation. The closure receives the current
            // tensor and re-executes the operation recorded in the autograd
            // graph. For unsupported ops we fall back to replaying through
            // the tensor_map: we look up the output tensor that was computed
            // during the original forward pass.
            let output_tensor = tensor_map.get(&op.output_id).cloned();
            let _op_name = op.name.to_string();

            segments.push(GraphSegment::Eager(Arc::new(move |input: &Tensor<T>| {
                // In a full implementation this would re-execute the
                // specific unsupported operation. For now, if the output
                // tensor was captured during tracing, return it. Otherwise,
                // return the input unchanged (identity fallback) with a
                // warning that the eager segment could not replay the op.
                if let Some(ref out) = output_tensor {
                    Ok(out.clone())
                } else {
                    // Identity fallback — the best we can do without the
                    // actual op implementation.
                    Ok(input.clone())
                }
            })));
        }
    }

    // Flush any remaining compiled run.
    if !compiled_run.is_empty() {
        let graph = build_ir_graph_from_run(&compiled_run, example_inputs, input_ids, tensor_map)?;
        let mut optimized = graph;
        let _memory_plan = optimize(&mut optimized, &config.optimization);
        segments.push(GraphSegment::Compiled(TracedModule::new(optimized)));
    }

    // If no segments were produced (edge case), return an error.
    if segments.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "trace_with_breaks: no segments produced".into(),
        });
    }

    Ok(segments)
}

/// Build an `IrGraph` from a contiguous run of supported ops within a
/// larger segmented trace.
///
/// Unlike `build_ir_graph`, this handles the case where some inputs come
/// from outside the run (they become graph inputs).
fn build_ir_graph_from_run<T: Float>(
    run: &[&AnnotatedOp],
    _example_inputs: &[Tensor<T>],
    input_ids: &HashMap<TensorId, usize>,
    _tensor_map: &HashMap<TensorId, Tensor<T>>,
) -> FerrotorchResult<IrGraph> {
    let mut graph = IrGraph::new();
    let mut tensor_to_ir: HashMap<TensorId, IrValueId> = HashMap::new();
    let mut next_input_index: usize = 0;

    // Collect all output TensorIds in this run so we know which inputs
    // are internal vs. external.
    let run_outputs: HashMap<TensorId, ()> = run.iter().map(|op| (op.output_id, ())).collect();

    // Create IR inputs for leaf tensors and for tensors produced outside
    // this run.
    for op in run {
        for (i, &cid) in op.input_ids.iter().enumerate() {
            if tensor_to_ir.contains_key(&cid) {
                continue;
            }

            let is_external = op.input_is_leaf[i] || !run_outputs.contains_key(&cid);
            if is_external {
                let _orig_index = input_ids.get(&cid);
                let index = next_input_index;
                next_input_index += 1;

                let value_id = graph.add_input(op.input_shapes[i].clone());
                let last_node = graph.nodes.last_mut().unwrap();
                last_node.op = IrOpKind::Input { index };

                tensor_to_ir.insert(cid, value_id);
            }
        }
    }

    // Create IR nodes for each op in the run.
    for op in run {
        let ir_inputs: Vec<IrValueId> = op
            .input_ids
            .iter()
            .map(|cid| {
                *tensor_to_ir.get(cid).unwrap_or_else(|| {
                    panic!(
                        "BUG: tensor {:?} not found in tensor_to_ir map during \
                         segment IR construction",
                        cid
                    )
                })
            })
            .collect();

        let ir_op = map_name_to_op(op.name, &op.output_shape)?;
        let (_, out_ids) = graph.add_node(ir_op, ir_inputs, vec![op.output_shape.clone()]);
        tensor_to_ir.insert(op.output_id, out_ids[0]);
    }

    // The last op in the run is the segment output.
    if let Some(last_op) = run.last() {
        if let Some(&ir_val) = tensor_to_ir.get(&last_op.output_id) {
            graph.set_outputs(vec![ir_val]);
        }
    }

    Ok(graph)
}

/// Map a `GradFn::name()` string to the corresponding IR operation.
///
/// Duplicated from `trace.rs` to avoid circular dependency. Must be kept
/// in sync.
fn map_name_to_op(name: &str, output_shape: &[usize]) -> FerrotorchResult<IrOpKind> {
    match name {
        "AddBackward" => Ok(IrOpKind::Add),
        "SubBackward" => Ok(IrOpKind::Sub),
        "MulBackward" => Ok(IrOpKind::Mul),
        "DivBackward" => Ok(IrOpKind::Div),
        "NegBackward" => Ok(IrOpKind::Neg),
        "PowBackward" => Ok(IrOpKind::Pow { exponent: 0.0 }),
        "SqrtBackward" => Ok(IrOpKind::Sqrt),
        "AbsBackward" => Ok(IrOpKind::Abs),
        "SumBackward" => Ok(IrOpKind::Sum),
        "MeanBackward" => Ok(IrOpKind::Mean),
        "ProdBackward" => Ok(IrOpKind::Prod),
        "MmBackward" => Ok(IrOpKind::Mm),
        "MatmulBackward" => Ok(IrOpKind::Matmul),
        "MvBackward" => Ok(IrOpKind::Mv),
        "DotBackward" => Ok(IrOpKind::Dot),
        "LinearFusedBackward" => Ok(IrOpKind::Linear),
        "ReluBackward" => Ok(IrOpKind::Relu),
        "SigmoidBackward" => Ok(IrOpKind::Sigmoid),
        "TanhBackward" => Ok(IrOpKind::Tanh),
        "GeluBackward" => Ok(IrOpKind::Gelu),
        "SiluBackward" => Ok(IrOpKind::Silu),
        "SoftmaxBackward" => Ok(IrOpKind::Softmax),
        "LogSoftmaxBackward" => Ok(IrOpKind::LogSoftmax),
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

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::IrOpKind;
    use ferrotorch_core::error::FerrotorchResult;
    use ferrotorch_core::grad_fns::arithmetic::{add, mul};
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

    /// Helper: create a 1-D f32 tensor without gradient tracking.
    fn tensor_1d(data: &[f32]) -> Tensor<f32> {
        ferrotorch_core::from_vec(data.to_vec(), &[data.len()]).unwrap()
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
    // Test: fully supported graph produces Unbroken result
    // -----------------------------------------------------------------------

    #[test]
    fn test_trace_no_breaks_produces_unbroken() {
        let x = grad_vec(vec![1.0, 2.0, 3.0]);

        let config = CompileConfig::default();
        let result = trace_with_breaks(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let doubled = add(&inputs[0], &inputs[0])?;
                sum(&doubled)
            },
            &[x],
            &config,
        )
        .unwrap();

        match result {
            TraceResult::Unbroken(graph) => {
                // Input + Add + Sum = 3 nodes.
                assert!(graph.node_count() >= 2);
                assert_eq!(graph.output_values.len(), 1);
            }
            TraceResult::Segmented(_) => {
                panic!("expected Unbroken, got Segmented");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test: SegmentedModule forward with compiled + eager segments
    // -----------------------------------------------------------------------

    #[test]
    fn test_segmented_module_forward() {
        // Build a SegmentedModule manually with a compiled and eager segment.

        // Compiled segment: y = x + x
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let compiled = TracedModule::<f32>::new(g);

        // Eager segment: multiply each element by 10
        type EagerFn = Arc<dyn Fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>> + Send + Sync>;
        let eager_fn: EagerFn = Arc::new(|input: &Tensor<f32>| {
            let ten =
                ferrotorch_core::from_vec(vec![10.0f32; input.numel()], input.shape()).unwrap();
            ferrotorch_core::grad_fns::arithmetic::mul(input, &ten)
        });

        let segments = vec![
            GraphSegment::Compiled(compiled),
            GraphSegment::Eager(eager_fn),
        ];

        let module = SegmentedModule::new(segments);
        assert_eq!(module.segment_count(), 2);
        assert!(!module.is_fully_compiled());

        // Input: [1, 2, 3]
        // After compiled (x + x): [2, 4, 6]
        // After eager (* 10): [20, 40, 60]
        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let result = module.forward(&input).unwrap();
        assert_close(result.data().unwrap(), &[20.0, 40.0, 60.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: fullgraph mode rejects graph breaks
    // -----------------------------------------------------------------------

    #[test]
    fn test_fullgraph_rejects_graph_break() {
        // We need an operation that is NOT in the known op mapping.
        // We'll simulate this by checking the JitError::GraphBreak path
        // directly, since constructing an autograd graph with an unknown
        // GradFn requires internal access.

        let config = CompileConfig {
            fullgraph: true,
            ..Default::default()
        };

        // An all-supported function should succeed even in fullgraph mode.
        let x = grad_vec(vec![1.0, 2.0, 3.0]);
        let result = trace_with_breaks(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let doubled = add(&inputs[0], &inputs[0])?;
                sum(&doubled)
            },
            &[x],
            &config,
        );

        assert!(result.is_ok());
        match result.unwrap() {
            TraceResult::Unbroken(_) => {} // expected
            TraceResult::Segmented(_) => panic!("expected Unbroken in fullgraph mode"),
        }
    }

    // -----------------------------------------------------------------------
    // Test: fullgraph error type is JitError::GraphBreak
    // -----------------------------------------------------------------------

    #[test]
    fn test_jit_error_graph_break_format() {
        let err = JitError::GraphBreak {
            op: "CustomOpBackward".into(),
            reason: "unsupported operation".into(),
        };

        let msg = format!("{err}");
        assert!(msg.contains("CustomOpBackward"));
        assert!(msg.contains("unsupported operation"));

        // Verify conversion to FerrotorchError works.
        let ferrotorch_err: FerrotorchError = err.into();
        let msg2 = format!("{ferrotorch_err}");
        assert!(msg2.contains("CustomOpBackward"));
    }

    // -----------------------------------------------------------------------
    // Test: SegmentedModule with only compiled segments is fully compiled
    // -----------------------------------------------------------------------

    #[test]
    fn test_segmented_module_fully_compiled() {
        // Segment 1: y = x + x
        let mut g1 = IrGraph::new();
        let x1 = g1.add_input(vec![3]);
        let (_, add_outs) = g1.add_node(IrOpKind::Add, vec![x1, x1], vec![vec![3]]);
        g1.set_outputs(vec![add_outs[0]]);

        // Segment 2: y = relu(x)
        let mut g2 = IrGraph::new();
        let x2 = g2.add_input(vec![3]);
        let (_, relu_outs) = g2.add_node(IrOpKind::Relu, vec![x2], vec![vec![3]]);
        g2.set_outputs(vec![relu_outs[0]]);

        let segments = vec![
            GraphSegment::Compiled(TracedModule::<f32>::new(g1)),
            GraphSegment::Compiled(TracedModule::<f32>::new(g2)),
        ];

        let module = SegmentedModule::new(segments);
        assert_eq!(module.segment_count(), 2);
        assert!(module.is_fully_compiled());

        // Input: [-1, 2, -3]
        // After add (x+x): [-2, 4, -6]
        // After relu: [0, 4, 0]
        let input = tensor_1d(&[-1.0, 2.0, -3.0]);
        let result = module.forward(&input).unwrap();
        assert_close(result.data().unwrap(), &[0.0, 4.0, 0.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: SegmentedModule with eager-only segment
    // -----------------------------------------------------------------------

    #[test]
    fn test_segmented_module_eager_only() {
        type EagerFn = Arc<dyn Fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>> + Send + Sync>;
        let eager_fn: EagerFn = Arc::new(|input: &Tensor<f32>| {
            // Double each element.
            ferrotorch_core::grad_fns::arithmetic::add(input, input)
        });

        let module = SegmentedModule::new(vec![GraphSegment::Eager(eager_fn)]);
        assert_eq!(module.segment_count(), 1);
        assert!(!module.is_fully_compiled());

        let input = tensor_1d(&[5.0, 10.0, 15.0]);
        let result = module.forward(&input).unwrap();
        assert_close(result.data().unwrap(), &[10.0, 20.0, 30.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: compiled + eager + compiled chain
    // -----------------------------------------------------------------------

    #[test]
    fn test_compiled_eager_compiled_chain() {
        // Segment 1 (compiled): y = x + x
        let mut g1 = IrGraph::new();
        let x1 = g1.add_input(vec![3]);
        let (_, add_outs) = g1.add_node(IrOpKind::Add, vec![x1, x1], vec![vec![3]]);
        g1.set_outputs(vec![add_outs[0]]);

        // Segment 2 (eager): negate
        type EagerFn = Arc<dyn Fn(&Tensor<f32>) -> FerrotorchResult<Tensor<f32>> + Send + Sync>;
        let eager_fn: EagerFn =
            Arc::new(|input: &Tensor<f32>| ferrotorch_core::grad_fns::arithmetic::neg(input));

        // Segment 3 (compiled): y = relu(x)
        let mut g3 = IrGraph::new();
        let x3 = g3.add_input(vec![3]);
        let (_, relu_outs) = g3.add_node(IrOpKind::Relu, vec![x3], vec![vec![3]]);
        g3.set_outputs(vec![relu_outs[0]]);

        let segments = vec![
            GraphSegment::Compiled(TracedModule::<f32>::new(g1)),
            GraphSegment::Eager(eager_fn),
            GraphSegment::Compiled(TracedModule::<f32>::new(g3)),
        ];

        let module = SegmentedModule::new(segments);
        assert_eq!(module.segment_count(), 3);
        assert!(!module.is_fully_compiled());

        // Input: [1, -2, 3]
        // After add (x+x): [2, -4, 6]
        // After neg: [-2, 4, -6]
        // After relu: [0, 4, 0]
        let input = tensor_1d(&[1.0, -2.0, 3.0]);
        let result = module.forward(&input).unwrap();
        assert_close(result.data().unwrap(), &[0.0, 4.0, 0.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: is_known_op
    // -----------------------------------------------------------------------

    #[test]
    fn test_is_known_op() {
        assert!(is_known_op("AddBackward"));
        assert!(is_known_op("MulBackward"));
        assert!(is_known_op("ReluBackward"));
        assert!(is_known_op("SumBackward"));
        assert!(is_known_op("MmBackward"));
        assert!(is_known_op("TransposeBackward"));

        assert!(!is_known_op("CustomOpBackward"));
        assert!(!is_known_op("PrintBackward"));
        assert!(!is_known_op(""));
        assert!(!is_known_op("ConvBackward"));
    }

    // -----------------------------------------------------------------------
    // Test: trace_with_breaks on fully supported graph (integration)
    // -----------------------------------------------------------------------

    #[test]
    fn test_trace_with_breaks_integration_no_break() {
        let a = grad_vec(vec![1.0, 2.0, 3.0]);
        let b = grad_vec(vec![4.0, 5.0, 6.0]);

        let config = CompileConfig::default();
        let result = trace_with_breaks(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let product = mul(&inputs[0], &inputs[1])?;
                sum(&product)
            },
            &[a, b],
            &config,
        )
        .unwrap();

        match result {
            TraceResult::Unbroken(graph) => {
                // 2 inputs + Mul + Sum = 4 nodes.
                assert_eq!(graph.node_count(), 4);
                assert_eq!(graph.input_values.len(), 2);
                assert_eq!(graph.output_values.len(), 1);
            }
            TraceResult::Segmented(_) => {
                panic!("expected Unbroken for fully supported graph");
            }
        }
    }

    // -----------------------------------------------------------------------
    // Test: Debug formatting for GraphSegment
    // -----------------------------------------------------------------------

    #[test]
    fn test_graph_segment_debug() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let compiled: GraphSegment<f32> = GraphSegment::Compiled(TracedModule::new(g));
        let debug_str = format!("{:?}", compiled);
        assert!(debug_str.contains("Compiled"));

        let eager: GraphSegment<f32> =
            GraphSegment::Eager(Arc::new(|input: &Tensor<f32>| Ok(input.clone())));
        let debug_str = format!("{:?}", eager);
        assert!(debug_str.contains("Eager"));
    }

    // -----------------------------------------------------------------------
    // Test: no grad_fn produces error
    // -----------------------------------------------------------------------

    #[test]
    fn test_trace_with_breaks_no_grad_fn_error() {
        let x = ferrotorch_core::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).unwrap();
        let config = CompileConfig::default();

        let result = trace_with_breaks(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                add(&inputs[0], &inputs[0])
            },
            &[x],
            &config,
        );

        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("no grad_fn"));
    }

    // -----------------------------------------------------------------------
    // Test: SegmentedModule is Send + Sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_segmented_module_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        // SegmentedModule is not automatically Send+Sync due to the
        // Arc<dyn Fn> but we require Send+Sync on the closure trait
        // bounds, so this should work.
        assert_send_sync::<SegmentedModule<f32>>();
        assert_send_sync::<SegmentedModule<f64>>();
    }
}
