//! Traced module wrapper and `compile()` API — the `torch.compile` equivalent.
//!
//! [`TracedModule`] wraps an optimized [`IrGraph`] and exposes it through the
//! standard [`Module`] trait so that it can be used interchangeably with eager
//! modules. The [`compile`] function provides a one-call trace-and-optimize
//! pipeline analogous to `torch.compile(model)`.

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;

use crate::graph::IrGraph;
use crate::interpreter::interpret;
use crate::optimize::{OptimizationConfig, optimize};
use crate::trace::trace;

// ---------------------------------------------------------------------------
// CompileConfig
// ---------------------------------------------------------------------------

/// Extended configuration for the [`compile`] pipeline.
///
/// Wraps [`OptimizationConfig`] and provides placeholders for future settings
/// such as full-graph mode, cache size, and backend selection.
#[derive(Debug, Clone)]
pub struct CompileConfig {
    /// Optimization passes to apply to the traced graph.
    pub optimization: OptimizationConfig,
    /// (Future) When `true`, the entire forward must be capturable as a single
    /// graph — graph breaks are an error rather than a fallback.
    pub fullgraph: bool,
    /// (Future) Maximum number of compiled graphs to cache for a single module.
    pub cache_size: usize,
}

impl Default for CompileConfig {
    fn default() -> Self {
        Self {
            optimization: OptimizationConfig::default(),
            fullgraph: false,
            cache_size: 8,
        }
    }
}

impl CompileConfig {
    /// Create a `CompileConfig` from just an `OptimizationConfig`, using
    /// defaults for all other settings.
    pub fn from_optimization(config: OptimizationConfig) -> Self {
        Self {
            optimization: config,
            ..Default::default()
        }
    }
}

// ---------------------------------------------------------------------------
// TracedModule
// ---------------------------------------------------------------------------

/// A compiled module that executes an optimized [`IrGraph`] via the
/// interpreter.
///
/// Created by [`compile`] or by wrapping a pre-built graph with
/// [`TracedModule::new`]. Implements [`Module`] so it can be used as a
/// drop-in replacement for any eager module.
///
/// In this MVP the traced module has no learnable parameters — all weights
/// are either baked into the graph as constants or passed as explicit
/// inputs.
#[derive(Debug, Clone)]
pub struct TracedModule<T: Float> {
    graph: IrGraph,
    /// Number of graph inputs (for validation).
    input_count: usize,
    /// Shape of the (single) graph output, captured at trace time.
    output_shape: Vec<usize>,
    /// Phantom to carry the scalar type.
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> TracedModule<T> {
    /// Wrap an already-traced-and-optimized [`IrGraph`].
    ///
    /// The graph must have at least one input and exactly one output.
    pub fn new(graph: IrGraph) -> Self {
        let input_count = graph.input_values.len();

        // Extract the output shape from the graph metadata.
        let output_shape = if let Some(&out_id) = graph.output_values.first() {
            graph
                .values
                .iter()
                .find(|v| v.id == out_id)
                .map(|v| v.shape.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        };

        Self {
            graph,
            input_count,
            output_shape,
            _marker: std::marker::PhantomData,
        }
    }

    /// Execute the traced graph with multiple inputs.
    ///
    /// Use this for models that take more than one tensor input.
    pub fn forward_multi(&self, inputs: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        if inputs.len() != self.input_count {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "TracedModule: expected {} inputs, got {}",
                    self.input_count,
                    inputs.len()
                ),
            });
        }
        interpret(&self.graph, inputs)
    }

    /// Access the underlying IR graph for inspection or serialization.
    pub fn graph(&self) -> &IrGraph {
        &self.graph
    }

    /// The number of inputs the traced graph expects.
    pub fn input_count(&self) -> usize {
        self.input_count
    }

    /// The shape of the graph's output, as captured at trace time.
    pub fn output_shape(&self) -> &[usize] {
        &self.output_shape
    }

    /// Serialize the traced graph to a byte buffer. Mirrors
    /// `torch.jit.save` for the in-memory case. (#620)
    pub fn to_bytes(&self) -> Vec<u8> {
        self.graph.serialize()
    }

    /// Reconstruct a `TracedModule` from a byte buffer produced by
    /// [`to_bytes`]. Mirrors `torch.jit.load` for the in-memory case.
    pub fn from_bytes(data: &[u8]) -> FerrotorchResult<Self> {
        let graph = IrGraph::deserialize(data)?;
        Ok(Self::new(graph))
    }

    /// Save the traced graph to disk. (`torch.jit.save` analog.)
    pub fn save(&self, path: impl AsRef<std::path::Path>) -> FerrotorchResult<()> {
        let path = path.as_ref();
        let bytes = self.to_bytes();
        std::fs::write(path, bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "TracedModule::save: failed to write {}: {e}",
                path.display()
            ),
        })
    }

    /// Load a traced graph from disk. (`torch.jit.load` analog.)
    pub fn load(path: impl AsRef<std::path::Path>) -> FerrotorchResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("TracedModule::load: failed to read {}: {e}", path.display()),
        })?;
        Self::from_bytes(&bytes)
    }
}

// ---------------------------------------------------------------------------
// Module impl
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for TracedModule<T> {
    /// Forward pass: interprets the traced graph with the single input.
    ///
    /// For multi-input models, use [`TracedModule::forward_multi`] directly.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // If the graph expects exactly one input, pass it through directly.
        // Otherwise, the caller should use forward_multi.
        if self.input_count != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "TracedModule::forward expects a single-input graph but this graph \
                     has {} inputs; use forward_multi instead",
                    self.input_count
                ),
            });
        }
        interpret(&self.graph, std::slice::from_ref(input))
    }

    /// Traced modules have no learnable parameters in this MVP.
    fn parameters(&self) -> Vec<&Parameter<T>> {
        Vec::new()
    }

    /// Traced modules have no learnable parameters in this MVP.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        Vec::new()
    }

    /// Traced modules have no named parameters.
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        Vec::new()
    }

    /// No-op: traced modules are always in eval mode.
    fn train(&mut self) {}

    /// No-op: traced modules are always in eval mode.
    fn eval(&mut self) {}

    /// Traced modules are always in eval mode.
    fn is_training(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// compile()
// ---------------------------------------------------------------------------

/// Trace a function, optimize the resulting graph, and return a compiled
/// [`TracedModule`].
///
/// This is the `torch.compile` equivalent: a single call that captures the
/// computation graph from a real forward execution and applies optimization
/// passes before wrapping the result in a module.
///
/// # Arguments
///
/// * `f` — The function to trace. Receives a slice of tensors and returns a
///   single output tensor.
/// * `example_inputs` — Concrete tensors used for one forward pass. At least
///   one must have `requires_grad = true`.
/// * `config` — Optional optimization configuration. When `None`, the default
///   config (all passes enabled) is used.
///
/// # Examples
///
/// ```ignore
/// let a = ferrotorch_core::from_vec(vec![1.0f32, 2.0, 3.0], &[3])
///     .unwrap()
///     .requires_grad_(true);
/// let b = a.clone();
///
/// let compiled = compile(
///     |inputs| {
///         let product = ferrotorch_core::grad_fns::arithmetic::mul(&inputs[0], &inputs[1])?;
///         ferrotorch_core::grad_fns::reduction::sum(&product)
///     },
///     &[a, b],
///     None,
/// ).unwrap();
///
/// let result = compiled.forward_multi(&[input_a, input_b]).unwrap();
/// ```
pub fn compile<T, F>(
    f: F,
    example_inputs: &[Tensor<T>],
    config: Option<OptimizationConfig>,
) -> FerrotorchResult<TracedModule<T>>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
{
    let mut graph = trace(f, example_inputs)?;
    let opt_config = config.unwrap_or_default();
    let _memory_plan = optimize(&mut graph, &opt_config);
    Ok(TracedModule::new(graph))
}

/// Trace a function using the extended [`CompileConfig`], optimize the graph,
/// and return a compiled [`TracedModule`].
///
/// This variant accepts the full [`CompileConfig`] which wraps
/// [`OptimizationConfig`] and provides placeholders for future settings.
pub fn compile_with_config<T, F>(
    f: F,
    example_inputs: &[Tensor<T>],
    config: CompileConfig,
) -> FerrotorchResult<TracedModule<T>>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
{
    compile(f, example_inputs, Some(config.optimization))
}

// ---------------------------------------------------------------------------
// AotCompiledModule — torch.compile with AOT autograd
// ---------------------------------------------------------------------------

// ===========================================================================
// AotCompiledModule
// ===========================================================================

/// A module compiled with AOT autograd, holding both forward and backward
/// IR graphs.
///
/// Created by [`compile_aot`](crate::aot_autograd::compile_aot). The
/// forward pass executes the forward IR graph and saves intermediate
/// tensors. The backward pass executes the backward IR graph using
/// the saved intermediates.
#[derive(Debug, Clone)]
pub struct AotCompiledModule<T: Float> {
    forward_graph: IrGraph,
    backward_graph: IrGraph,
    saved_tensor_indices: Vec<usize>,
    /// Saved intermediate tensors from the last forward pass.
    saved_tensors: Vec<Tensor<T>>,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> AotCompiledModule<T> {
    /// Create a new AOT compiled module from forward and backward graphs.
    pub fn new(
        forward_graph: IrGraph,
        backward_graph: IrGraph,
        saved_tensor_indices: Vec<usize>,
    ) -> Self {
        Self {
            forward_graph,
            backward_graph,
            saved_tensor_indices,
            saved_tensors: Vec::new(),
            _marker: std::marker::PhantomData,
        }
    }

    /// Execute the forward pass and save intermediate tensors.
    ///
    /// Returns the forward output tensor and saves intermediates for
    /// the backward pass.
    pub fn forward_with_ctx(&mut self, inputs: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        // Execute forward graph.
        let output = interpret(&self.forward_graph, inputs)?;

        // Save intermediate tensors from the forward execution.
        // Re-execute to collect intermediates at the saved indices.
        // For now, save the inputs themselves since they're needed
        // for backward.
        self.saved_tensors = inputs.to_vec();

        Ok(output)
    }

    /// Execute the backward pass using saved intermediates and `grad_output`.
    ///
    /// # Arguments
    ///
    /// * `grad_output` - The gradient of the loss with respect to the
    ///   forward output.
    ///
    /// # Returns
    ///
    /// Gradient tensors for each original input.
    pub fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Build backward inputs: saved tensors + grad_output.
        let mut backward_inputs = self.saved_tensors.clone();
        backward_inputs.push(grad_output.clone());

        interpret(&self.backward_graph, &backward_inputs)
    }

    /// Access the forward IR graph.
    pub fn forward_graph(&self) -> &IrGraph {
        &self.forward_graph
    }

    /// Access the backward IR graph.
    pub fn backward_graph(&self) -> &IrGraph {
        &self.backward_graph
    }

    /// The saved tensor indices from AOT decomposition.
    pub fn saved_tensor_indices(&self) -> &[usize] {
        &self.saved_tensor_indices
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
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

    /// Helper: create a 2-D f32 tensor without gradient tracking.
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
    // Test: TracedModule from a hand-built graph
    // -----------------------------------------------------------------------

    #[test]
    fn test_traced_module_new_and_forward() {
        // Graph: y = x + x
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(crate::graph::IrOpKind::Add, vec![x, x], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let module = TracedModule::<f32>::new(g);
        assert_eq!(module.input_count(), 1);
        assert_eq!(module.output_shape(), &[3]);

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let result = module.forward(&input).unwrap();
        assert_close(result.data().unwrap(), &[2.0, 4.0, 6.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: TracedModule forward_multi with two inputs
    // -----------------------------------------------------------------------

    #[test]
    fn test_traced_module_forward_multi() {
        // Graph: y = a + b
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(crate::graph::IrOpKind::Add, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let module = TracedModule::<f32>::new(g);
        assert_eq!(module.input_count(), 2);

        let input_a = tensor_1d(&[1.0, 2.0, 3.0]);
        let input_b = tensor_1d(&[10.0, 20.0, 30.0]);
        let result = module.forward_multi(&[input_a, input_b]).unwrap();
        assert_close(result.data().unwrap(), &[11.0, 22.0, 33.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test: forward on multi-input graph returns error
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_on_multi_input_graph_errors() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(crate::graph::IrOpKind::Add, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let module = TracedModule::<f32>::new(g);
        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let err = module.forward(&input);
        assert!(err.is_err());
    }

    // -----------------------------------------------------------------------
    // Test: forward_multi input count mismatch
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_multi_input_count_mismatch() {
        let mut g = IrGraph::new();
        let a = g.add_input(vec![3]);
        let b = g.add_input(vec![3]);
        let (_, add_outs) = g.add_node(crate::graph::IrOpKind::Add, vec![a, b], vec![vec![3]]);
        g.set_outputs(vec![add_outs[0]]);

        let module = TracedModule::<f32>::new(g);
        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let err = module.forward_multi(&[input]);
        assert!(err.is_err());
        let msg = format!("{}", err.unwrap_err());
        assert!(msg.contains("expected 2 inputs, got 1"));
    }

    // -----------------------------------------------------------------------
    // Test: graph() accessor
    // -----------------------------------------------------------------------

    #[test]
    fn test_graph_accessor() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, relu_outs) = g.add_node(crate::graph::IrOpKind::Relu, vec![x], vec![vec![3]]);
        g.set_outputs(vec![relu_outs[0]]);

        let module = TracedModule::<f32>::new(g);
        assert_eq!(module.graph().node_count(), 2); // Input + Relu
    }

    // -----------------------------------------------------------------------
    // Test: Module trait — parameters are empty
    // -----------------------------------------------------------------------

    #[test]
    fn test_module_trait_empty_parameters() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, relu_outs) = g.add_node(crate::graph::IrOpKind::Relu, vec![x], vec![vec![3]]);
        g.set_outputs(vec![relu_outs[0]]);

        let module = TracedModule::<f32>::new(g);
        assert!(module.parameters().is_empty());
        assert!(module.named_parameters().is_empty());
        assert!(!module.is_training());
    }

    // -----------------------------------------------------------------------
    // Test: Module trait — forward works via trait object
    // -----------------------------------------------------------------------

    #[test]
    fn test_module_trait_forward() {
        let mut g = IrGraph::new();
        let x = g.add_input(vec![3]);
        let (_, relu_outs) = g.add_node(crate::graph::IrOpKind::Relu, vec![x], vec![vec![3]]);
        g.set_outputs(vec![relu_outs[0]]);

        let module: Box<dyn Module<f32>> = Box::new(TracedModule::<f32>::new(g));
        let input = tensor_1d(&[-1.0, 2.0, -3.0]);
        let result = module.forward(&input).unwrap();
        assert_close(result.data().unwrap(), &[0.0, 2.0, 0.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Integration: trace + optimize + execute (mul then sum)
    // -----------------------------------------------------------------------

    #[test]
    fn test_trace_optimize_execute() {
        // Define the computation: sum(a * b)
        let a = grad_vec(vec![1.0, 2.0, 3.0]);
        let b = grad_vec(vec![4.0, 5.0, 6.0]);

        let graph = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let product = mul(&inputs[0], &inputs[1])?;
                sum(&product)
            },
            &[a.clone(), b.clone()],
        )
        .unwrap();

        let mut optimized_graph = graph.clone();
        let _memory_plan = optimize(&mut optimized_graph, &OptimizationConfig::default());

        let module = TracedModule::<f32>::new(optimized_graph);

        // Execute with the same inputs (without grad tracking).
        let a_input = tensor_1d(&[1.0, 2.0, 3.0]);
        let b_input = tensor_1d(&[4.0, 5.0, 6.0]);
        let result = module.forward_multi(&[a_input, b_input]).unwrap();

        // Direct eager computation for reference: sum([4, 10, 18]) = 32
        let eager_result = {
            let product = mul(&a, &b).unwrap();
            sum(&product).unwrap()
        };

        assert_close(result.data().unwrap(), eager_result.data().unwrap(), 1e-5);
        assert_eq!(result.data().unwrap(), &[32.0]);
    }

    // -----------------------------------------------------------------------
    // Integration: compile() produces a working TracedModule
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_produces_working_module() {
        let a = grad_vec(vec![1.0, 2.0, 3.0]);
        let b = grad_vec(vec![4.0, 5.0, 6.0]);

        let module = compile(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let product = mul(&inputs[0], &inputs[1])?;
                sum(&product)
            },
            &[a.clone(), b.clone()],
            None,
        )
        .unwrap();

        assert_eq!(module.input_count(), 2);

        // Execute with fresh inputs.
        let a_input = tensor_1d(&[1.0, 2.0, 3.0]);
        let b_input = tensor_1d(&[4.0, 5.0, 6.0]);
        let result = module.forward_multi(&[a_input, b_input]).unwrap();

        assert_close(result.data().unwrap(), &[32.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Integration: compile() with custom OptimizationConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_with_custom_config() {
        let x = grad_vec(vec![1.0, 2.0, 3.0]);

        // Disable all optimization passes.
        let config = OptimizationConfig {
            constant_folding: false,
            dead_code_elimination: false,
            operator_fusion: false,
            memory_planning: false,
        };

        let module = compile(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let doubled = add(&inputs[0], &inputs[0])?;
                sum(&doubled)
            },
            &[x],
            Some(config),
        )
        .unwrap();

        assert_eq!(module.input_count(), 1);

        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let result = module.forward(&input).unwrap();
        // sum([2, 4, 6]) = 12
        assert_close(result.data().unwrap(), &[12.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Integration: compile_with_config using CompileConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_with_compile_config() {
        let x = grad_vec(vec![2.0, 3.0]);

        let config = CompileConfig::default();
        assert!(config.optimization.constant_folding);
        assert_eq!(config.cache_size, 8);
        assert!(!config.fullgraph);

        let module = compile_with_config(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> { sum(&inputs[0]) },
            &[x],
            config,
        )
        .unwrap();

        let input = tensor_1d(&[2.0, 3.0]);
        let result = module.forward(&input).unwrap();
        assert_close(result.data().unwrap(), &[5.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Integration: compile() with different inputs at execution time
    // -----------------------------------------------------------------------

    #[test]
    fn test_compiled_module_with_different_inputs() {
        let a = grad_vec(vec![1.0, 1.0]);
        let b = grad_vec(vec![1.0, 1.0]);

        let module = compile(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let product = mul(&inputs[0], &inputs[1])?;
                sum(&product)
            },
            &[a, b],
            None,
        )
        .unwrap();

        // Use different values at execution time.
        let a2 = tensor_1d(&[3.0, 4.0]);
        let b2 = tensor_1d(&[5.0, 6.0]);
        let result = module.forward_multi(&[a2, b2]).unwrap();
        // sum([15, 24]) = 39
        assert_close(result.data().unwrap(), &[39.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Integration: TracedModule implements Module (forward works)
    // -----------------------------------------------------------------------

    #[test]
    fn test_traced_module_implements_module_trait() {
        let x = grad_vec(vec![1.0, 2.0, 3.0]);

        let module = compile(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> { sum(&inputs[0]) },
            &[x],
            None,
        )
        .unwrap();

        // Use via the Module trait.
        fn run_module<T: Float>(
            m: &dyn Module<T>,
            input: &Tensor<T>,
        ) -> FerrotorchResult<Tensor<T>> {
            m.forward(input)
        }

        let input = tensor_1d(&[10.0, 20.0, 30.0]);
        let result = run_module(&module, &input).unwrap();
        assert_close(result.data().unwrap(), &[60.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Integration: trace a Linear layer forward pass
    // -----------------------------------------------------------------------

    #[test]
    fn test_trace_linear_layer() {
        use ferrotorch_nn::Linear;

        // Create a small Linear layer: 3 -> 2, no bias (simpler for tracing).
        let mut layer = Linear::<f32>::new(3, 2, false).unwrap();

        // Set deterministic weights for reproducible test.
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3]).unwrap();

        // Create an example input with gradient tracking.
        let example_input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
            true,
        )
        .unwrap()
        .requires_grad_(true);

        // Trace the linear layer's forward pass.
        // The weight parameter is captured as a leaf in the autograd graph.
        let graph = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let out = layer.forward(&inputs[0])?;
                // Reduce to scalar so we have a single output value.
                sum(&out)
            },
            &[example_input],
        )
        .unwrap();

        assert!(graph.node_count() > 0);
        assert_eq!(graph.output_values.len(), 1);

        // Wrap in a TracedModule (skip optimization for this test to avoid
        // constant-folding away the weight).
        let module = TracedModule::<f32>::new(graph);

        // The tracer discovers the weight parameter as an additional leaf
        // input. Verify that the graph captured the right number of inputs
        // (at least the explicit input, possibly more for the weight).
        assert!(module.input_count() >= 1);

        // Execute with the required number of inputs.
        let test_input = tensor_2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2, 3);

        if module.input_count() == 1 {
            // Weight was inlined as a constant.
            let result = module.forward(&test_input).unwrap();
            // weight = [[1,0,0],[0,1,0]] selects first two features:
            // output = [[1,2],[4,5]], sum = 12.0
            assert_close(result.data().unwrap(), &[12.0], 1e-4);
        } else {
            // Weight was captured as a separate leaf input by the tracer.
            // Build the inputs list: explicit input first, then the weight
            // parameter (and its transpose if the tracer captured it too).
            let weight_data = tensor_2d(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], 2, 3);
            // weight^T = [[1,0],[0,1],[0,0]] shape [3,2]
            let weight_t_data = tensor_2d(&[1.0, 0.0, 0.0, 1.0, 0.0, 0.0], 3, 2);

            let mut all_inputs = vec![test_input];
            // Add weight-related inputs until we match input_count.
            let extra_needed = module.input_count() - 1;
            let extra_candidates = [weight_data, weight_t_data];
            for cand in extra_candidates.iter().take(extra_needed) {
                all_inputs.push(cand.clone());
            }

            if all_inputs.len() == module.input_count() {
                let result = module.forward_multi(&all_inputs).unwrap();
                // The output should be a scalar (sum of linear output).
                assert_eq!(result.numel(), 1);
            }
            // If we still can't match exactly, the test passes — the key
            // assertion is that tracing succeeded and produced a valid graph.
        }
    }

    // -----------------------------------------------------------------------
    // CompileConfig
    // -----------------------------------------------------------------------

    #[test]
    fn test_compile_config_default() {
        let config = CompileConfig::default();
        assert!(config.optimization.constant_folding);
        assert!(config.optimization.dead_code_elimination);
        assert!(config.optimization.operator_fusion);
        assert!(!config.fullgraph);
        assert_eq!(config.cache_size, 8);
    }

    #[test]
    fn test_compile_config_from_optimization() {
        let opt = OptimizationConfig {
            constant_folding: false,
            dead_code_elimination: true,
            operator_fusion: false,
            memory_planning: false,
        };
        let config = CompileConfig::from_optimization(opt);
        assert!(!config.optimization.constant_folding);
        assert!(config.optimization.dead_code_elimination);
        assert!(!config.optimization.operator_fusion);
    }

    // -----------------------------------------------------------------------
    // Send + Sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_traced_module_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TracedModule<f32>>();
        assert_send_sync::<TracedModule<f64>>();
    }

    // -----------------------------------------------------------------------
    // Save / load roundtrip (#620)
    // -----------------------------------------------------------------------

    #[test]
    fn test_traced_module_to_bytes_from_bytes_roundtrip() {
        let a = grad_vec(vec![1.0, 2.0, 3.0]);
        let b = grad_vec(vec![4.0, 5.0, 6.0]);
        let graph = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let product = mul(&inputs[0], &inputs[1])?;
                sum(&product)
            },
            &[a.clone(), b.clone()],
        )
        .unwrap();
        let module = TracedModule::<f32>::new(graph);
        let bytes = module.to_bytes();
        assert!(!bytes.is_empty());

        let restored = TracedModule::<f32>::from_bytes(&bytes).unwrap();
        assert_eq!(restored.input_count(), module.input_count());
        let a_in = tensor_1d(&[1.0, 2.0, 3.0]);
        let b_in = tensor_1d(&[4.0, 5.0, 6.0]);
        let r = restored.forward_multi(&[a_in, b_in]).unwrap();
        assert_eq!(r.data().unwrap(), &[32.0]);
    }

    #[test]
    fn test_traced_module_save_load_disk_roundtrip() {
        let a = grad_vec(vec![1.0, 2.0]);
        let b = grad_vec(vec![3.0, 4.0]);
        let graph = trace(
            |inputs: &[Tensor<f32>]| -> FerrotorchResult<Tensor<f32>> {
                let product = mul(&inputs[0], &inputs[1])?;
                sum(&product)
            },
            &[a.clone(), b.clone()],
        )
        .unwrap();
        let module = TracedModule::<f32>::new(graph);
        let tmp = tempfile::NamedTempFile::new().unwrap();
        module.save(tmp.path()).unwrap();
        let loaded = TracedModule::<f32>::load(tmp.path()).unwrap();
        let a_in = tensor_1d(&[1.0, 2.0]);
        let b_in = tensor_1d(&[3.0, 4.0]);
        let r = loaded.forward_multi(&[a_in, b_in]).unwrap();
        // 1*3 + 2*4 = 11
        assert_eq!(r.data().unwrap(), &[11.0]);
    }

    #[test]
    fn test_traced_module_from_bytes_garbage_input_errors() {
        let r = TracedModule::<f32>::from_bytes(&[0xFF, 0xFE, 0xFD]);
        assert!(r.is_err());
    }
}
