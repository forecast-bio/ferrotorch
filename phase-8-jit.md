---
title: "Phase 8 — JIT / Graph Optimization (ferrotorch-jit)"
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-15
updated: 2026-03-15
---


## Design Specification

### Summary

A tracing JIT compiler that captures ferrotorch-core's dynamic computation graphs into a static intermediate representation, applies graph-level optimizations (constant folding, operator fusion, dead code elimination, memory planning, kernel selection), and emits optimized executables for production inference and accelerated training. The crate intercepts the existing `GradFn<T>` autograd graph by replaying a forward function with proxy tensors, producing a `TracedModule` that can be serialized, optimized, and executed without rebuilding the graph each iteration.

### Requirements

- REQ-1: A `trace` function must accept any closure `Fn(&[Tensor<T>]) -> Result<Tensor<T>, FerrotorchError>` and a set of example inputs, execute the closure once while recording every operation into a static IR graph, and return a `TracedModule<T>` that reproduces the same computation without dynamic graph construction overhead. Tracing must capture the complete forward pass — every `GradFn<T>` node created during execution must have a corresponding IR node. Control flow that depends on tensor values (data-dependent branching) must be detected and rejected with `FerrotorchError::TracingError` rather than silently producing an incorrect graph.
- REQ-2: The IR graph must be a directed acyclic graph of typed operation nodes (`IrNode`) connected by value edges (`IrValue`). Each `IrNode` must record: the operation kind (matching ferrotorch-core's op vocabulary — arithmetic, reduction, linalg, activation, shape, indexing), input values, output values, shape/dtype metadata, and an optional device annotation. The IR must support subgraphs for representing fused regions. The graph must be serializable to and deserializable from a binary format for ahead-of-time compilation workflows.
- REQ-3: The optimizer must implement at minimum five graph transformation passes executed in a fixed pipeline order: (1) constant folding — evaluate subgraphs whose inputs are all compile-time constants and replace them with literal tensors, (2) dead code elimination — remove nodes whose outputs are not consumed by any live node or graph output, (3) operator fusion — identify chains of elementwise operations and collapse them into a single fused kernel node, (4) memory planning — analyze the liveness of every IR value and assign buffer slots that maximize memory reuse across non-overlapping lifetimes, (5) kernel selection — annotate each node with the preferred execution strategy (e.g., SIMD width, tiling parameters, or GPU kernel choice) based on the target device and tensor shapes. Each pass must be independently toggleable via an `OptimizationConfig` struct.
- REQ-4: `TracedModule<T>` must implement ferrotorch-nn's `Module<T>` trait so that traced models are drop-in replacements for eager models in inference pipelines. Calling `forward` on a `TracedModule` must execute the optimized IR graph rather than re-tracing. `TracedModule` must also expose `backward` support: when any input `requires_grad`, the traced graph must produce correct gradients by recording the backward pass as a second IR graph (the adjoint graph) during tracing.
- REQ-5: A code generation backend must translate the optimized IR graph into executable form. The initial implementation must target a Rust-native backend (Cranelift) that emits machine code for CPU execution. The `Codegen` trait must be object-safe and extensible so that additional backends (LLVM, CUDA PTX via ferrotorch-gpu) can be added without modifying existing code. Generated code must match eager-mode numerical output within the same floating-point tolerances used by ferrotorch-core's test suite (rtol=1e-4 for f32, rtol=1e-7 for f64).
- REQ-6: All public functions must return `Result<T, FerrotorchError>`. Tracing failures (data-dependent control flow, unsupported operations, shape mismatches in the IR) must produce descriptive error variants. The crate must never panic on invalid input.
- REQ-7: Tracing must be deterministic: given the same function and the same example input shapes, repeated calls to `trace` must produce identical IR graphs. The IR must be independent of the concrete values of the example inputs — only shapes, dtypes, and devices are captured.
- REQ-8: The crate must provide a `GraphProfiler` that instruments a `TracedModule` to collect per-node execution times, memory allocation sizes, and fusion region boundaries. Profiling output must be a structured `ProfileReport` that can be printed as a human-readable summary or serialized for external tooling.

### Acceptance Criteria

- [ ] AC-1: `jit::trace(|inputs| model.forward(&inputs[0]), &[example])` on a 4-layer MLP (Linear-ReLU-Linear-ReLU-Linear-ReLU-Linear) produces a `TracedModule` whose IR contains the expected number of matmul, add (bias), and relu nodes. Re-executing the `TracedModule` with different-valued inputs of the same shape produces numerically identical output to eager-mode execution (within f32 tolerance).
- [ ] AC-2: Tracing a function that contains `if tensor.sum().item() > 0.0` (data-dependent branch) returns `Err(FerrotorchError::TracingError { .. })` with a message identifying the offending operation as data-dependent control flow.
- [ ] AC-3: Constant folding eliminates compile-time constant subgraphs: tracing `|x| x * Tensor::ones([3, 3]) + Tensor::zeros([3, 3])` produces an IR where the `ones` and `zeros` tensors are folded into a single constant node and the dead addition of zero is eliminated.
- [ ] AC-4: Operator fusion merges a chain of 5 sequential elementwise operations (add, mul, relu, sigmoid, neg) into a single fused node in the IR. Executing the fused graph produces the same output as the unfused version within floating-point tolerance.
- [ ] AC-5: Memory planning reduces peak buffer allocation by at least 30% on a 20-layer residual network compared to naive per-node allocation, measured by summing the sizes of all simultaneously live buffers.
- [ ] AC-6: `TracedModule` implements `Module<T>` and can be used as a drop-in replacement in an inference loop: `let traced = jit::trace(|x| model.forward(&x[0]), &[example])?; let out = traced.forward(&input)?;` compiles and produces correct output.
- [ ] AC-7: The Cranelift backend compiles the IR for a simple computation graph (matmul + relu + matmul) into native machine code and executes it, producing output matching eager mode. Compilation time for a 50-node graph is under 100ms on a modern x86-64 CPU.
- [ ] AC-8: `TracedModule::backward` computes correct gradients for all leaf inputs through the traced graph, verified against eager-mode backward for at least 10 representative graphs including matmul chains, activation functions, reductions, and reshape operations.
- [ ] AC-9: IR graphs survive a round-trip through serialization: `let bytes = graph.serialize()?; let restored = IrGraph::deserialize(&bytes)?;` produces a graph that executes identically to the original.
- [ ] AC-10: `cargo test -p ferrotorch-jit` passes with 0 failures. Minimum 150 tests covering tracing, all five optimization passes, codegen, backward through traced graphs, serialization round-trips, error paths, and profiling.

### Architecture

### Crate Layout

```
ferrotorch-jit/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports: trace, TracedModule, IrGraph, OptimizationConfig
│   ├── trace.rs                  # Tracing engine — proxy tensors, op recording, control flow detection
│   ├── graph.rs                  # IrGraph, IrNode, IrValue, IrOpKind — the intermediate representation
│   ├── optimize.rs               # Optimization pipeline — constant fold, DCE, fusion, memory plan, kernel select
│   ├── codegen.rs                # Codegen trait + Cranelift backend (CPU native)
│   ├── module.rs                 # TracedModule — Module<T> impl, forward/backward execution
│   ├── profile.rs                # GraphProfiler, ProfileReport — per-node timing and memory stats
│   ├── serialize.rs              # Binary serialization/deserialization of IrGraph
│   └── error.rs                  # JIT-specific error variants (TracingError, CodegenError, etc.)
└── tests/
    ├── test_trace.rs             # Tracing correctness, data-dependent branch rejection, determinism
    ├── test_graph.rs             # IR construction, node connectivity, shape inference
    ├── test_optimize.rs          # Each optimization pass in isolation + full pipeline
    ├── test_codegen.rs           # Cranelift compilation and execution correctness
    ├── test_backward.rs          # Gradient correctness through traced graphs
    ├── test_serialize.rs         # Round-trip serialization
    └── test_profile.rs           # Profiling instrumentation output
```

### Core Types

**IrGraph** (`graph.rs`):
```rust
/// A static computation graph in SSA form. Every value is defined exactly once.
pub struct IrGraph<T: Element> {
    nodes: Vec<IrNode<T>>,
    inputs: Vec<IrValue>,           // Graph-level input placeholders
    outputs: Vec<IrValue>,          // Graph-level outputs
    constants: HashMap<IrValue, Tensor<T>>,  // Folded constant tensors
    metadata: GraphMetadata,        // Source function name, trace timestamp
}

/// A single operation in the IR.
pub struct IrNode<T: Element> {
    id: NodeId,
    op: IrOpKind,
    inputs: Vec<IrValue>,
    outputs: Vec<IrValue>,
    shape: Vec<usize>,              // Output shape (inferred during tracing)
    dtype: DType,
    device: Device,
    fusion_group: Option<FusionGroupId>,
    _marker: PhantomData<T>,
}

/// Every differentiable operation in ferrotorch-core has a corresponding variant.
pub enum IrOpKind {
    // Arithmetic
    Add, Sub, Mul, Div, Neg, Pow, Sqrt, Abs,
    // Reduction
    Sum { dims: Vec<usize>, keep_dim: bool },
    Mean { dims: Vec<usize>, keep_dim: bool },
    Prod { dims: Vec<usize>, keep_dim: bool },
    // Linalg
    Matmul, Bmm, Mm, Mv, Dot,
    // Activation
    Relu, Sigmoid, Tanh, Gelu, Silu, Softmax { dim: usize }, LogSoftmax { dim: usize },
    // Shape
    Reshape { shape: Vec<usize> }, Transpose { dim0: usize, dim1: usize },
    Permute { dims: Vec<usize> }, Expand { shape: Vec<usize> },
    Cat { dim: usize }, Stack { dim: usize }, Split { dim: usize, sizes: Vec<usize> },
    Squeeze { dim: Option<usize> }, Unsqueeze { dim: usize }, Flatten { start: usize, end: usize },
    // Indexing
    Gather { dim: usize }, ScatterAdd { dim: usize }, IndexSelect { dim: usize }, MaskedFill,
    // Comparison
    Where,
    // Special
    Constant,                       // Literal tensor embedded in the graph
    FusedElementwise { ops: Vec<IrOpKind> },  // Result of operator fusion
}

/// A typed handle to a value produced by a node.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct IrValue(u32);

pub struct NodeId(u32);
pub struct FusionGroupId(u32);
```

**Tracing Engine** (`trace.rs`):
```rust
/// Trace a forward function into a frozen IR graph.
pub fn trace<T, F>(
    f: F,
    example_inputs: &[Tensor<T>],
) -> Result<TracedModule<T>, FerrotorchError>
where
    T: Element,
    F: Fn(&[Tensor<T>]) -> Result<Tensor<T>, FerrotorchError>,
{
    // 1. Create proxy tensors that mirror example_inputs' shapes/dtypes/devices
    // 2. Enable tracing mode (thread-local flag, similar to no_grad)
    // 3. Execute f(&proxies) — every op records an IrNode instead of computing
    // 4. Disable tracing mode
    // 5. Collect recorded nodes into an IrGraph
    // 6. Build adjoint graph for backward support
    // 7. Run optimization pipeline
    // 8. Return TracedModule wrapping the optimized graph
}
```

Tracing intercepts operation dispatch at the `GradFn<T>` boundary. When tracing mode is active (controlled by a `TRACING_ACTIVE` thread-local, mirroring the `GRAD_ENABLED` pattern from ferrotorch-core's `no_grad`), each op that would normally construct a `GradFn<T>` node instead records an `IrNode` into a thread-local `TraceRecorder`. The proxy tensors carry shape metadata but no real data — operations infer output shapes from input shapes using the same broadcasting and shape-propagation rules as eager mode.

Data-dependent control flow detection works by tagging proxy tensors with a `is_proxy: bool` flag. Any attempt to extract a scalar value from a proxy (via `.item()`, comparison to a concrete value, or boolean coercion) raises `FerrotorchError::TracingError` immediately, identifying the operation that would introduce data-dependent branching.

**Adjoint Graph Construction**: During tracing, every `IrNode` has a known `GradFn<T>` counterpart from ferrotorch-core. The adjoint graph is built by symbolically applying each `GradFn`'s backward rule in reverse topological order over the IR, producing a second `IrGraph` that maps output gradients to input gradients. This adjoint graph receives the same optimization passes as the forward graph.

**Optimization Pipeline** (`optimize.rs`):
```rust
pub struct OptimizationConfig {
    pub constant_folding: bool,         // Default: true
    pub dead_code_elimination: bool,    // Default: true
    pub operator_fusion: bool,          // Default: true
    pub memory_planning: bool,          // Default: true
    pub kernel_selection: bool,         // Default: true
}

pub fn optimize<T: Element>(
    graph: &mut IrGraph<T>,
    config: &OptimizationConfig,
) -> Result<(), FerrotorchError> {
    if config.constant_folding { constant_fold(graph)?; }
    if config.dead_code_elimination { dead_code_eliminate(graph)?; }
    if config.operator_fusion { fuse_operators(graph)?; }
    if config.memory_planning { plan_memory(graph)?; }
    if config.kernel_selection { select_kernels(graph)?; }
    Ok(())
}
```

The passes execute in a fixed order because each pass's output feeds the next:
1. **Constant folding** runs first because it materializes constant subgraphs into literal `Constant` nodes, exposing new dead code and fusion opportunities.
2. **Dead code elimination** removes nodes orphaned by folding (and any that were unused in the original graph). Uses a reverse reachability pass from graph outputs — any node not reachable is dead.
3. **Operator fusion** identifies maximal chains of elementwise `IrOpKind` variants (Add, Mul, Relu, Sigmoid, etc.) connected by single-use edges. Each chain is collapsed into a `FusedElementwise` node whose `ops` field records the original sequence. Fusion boundaries are: non-elementwise ops, nodes with multiple consumers (fan-out), and device transitions.
4. **Memory planning** performs liveness analysis on all `IrValue` handles, computes overlapping lifetimes, and assigns buffer slots using a greedy first-fit algorithm. The result is a `MemoryPlan` stored as metadata on the graph — a mapping from `IrValue` to `(slot_id, offset, size)`. The executor allocates one contiguous arena per slot and sub-allocates from it.
5. **Kernel selection** annotates each node with execution hints based on the target `Device` and the node's shapes. For CPU: SIMD width (AVX2/AVX-512 detection at trace time), tiling parameters for matmul, parallelism degree for reductions. For CUDA (when ferrotorch-gpu is present): grid/block dimensions, shared memory usage, and whether to use cuBLAS or a custom kernel.

**Code Generation** (`codegen.rs`):
```rust
/// Backend-agnostic code generation trait.
pub trait Codegen<T: Element>: Send + Sync {
    /// Compile an optimized IR graph into an executable artifact.
    fn compile(&self, graph: &IrGraph<T>) -> Result<CompiledGraph<T>, FerrotorchError>;
}

/// An executable compiled graph.
pub struct CompiledGraph<T: Element> {
    execute_fn: Box<dyn Fn(&[Tensor<T>]) -> Result<Tensor<T>, FerrotorchError> + Send + Sync>,
    memory_plan: MemoryPlan,
    _marker: PhantomData<T>,
}

/// Cranelift-based CPU code generator (default backend).
pub struct CraneliftBackend {
    opt_level: cranelift_codegen::settings::OptLevel,
}

impl<T: Element> Codegen<T> for CraneliftBackend {
    fn compile(&self, graph: &IrGraph<T>) -> Result<CompiledGraph<T>, FerrotorchError> {
        // 1. Create Cranelift IR function
        // 2. For each IrNode, emit Cranelift instructions:
        //    - Elementwise/fused: inline SIMD loop
        //    - Matmul: call into ferray-linalg (extern function reference)
        //    - Reduction: emit reduction loop with accumulator
        // 3. Finalize and compile to native machine code
        // 4. Wrap in CompiledGraph with the memory plan from optimization
    }
}
```

The Cranelift backend handles elementwise and fused ops by emitting tight SIMD loops directly. For complex operations (matmul, convolutions, FFT), it emits calls to the same ferray functions that eager mode uses — the JIT benefit comes from eliminating graph overhead, fusing surrounding elementwise ops, and pre-planning memory, not from replacing BLAS kernels.

**TracedModule** (`module.rs`):
```rust
pub struct TracedModule<T: Element> {
    forward_graph: IrGraph<T>,
    backward_graph: Option<IrGraph<T>>,  // Present when any traced input requires_grad
    compiled: Option<CompiledGraph<T>>,  // Lazily compiled on first forward call
    input_shapes: Vec<Vec<usize>>,       // Expected input shapes from tracing
    config: OptimizationConfig,
}

impl<T: Element> Module<T> for TracedModule<T> {
    fn forward(&self, input: &Tensor<T>) -> Result<Tensor<T>, FerrotorchError> {
        // Validate input shape matches traced shape
        // Execute compiled graph (compile lazily if needed)
        // If input.requires_grad, attach backward_graph as the grad_fn
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        // Return parameters captured as constants during tracing
        Vec::new()  // TracedModule captures weights as constants, not mutable parameters
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> { Vec::new() }
    fn train(&mut self) {}
    fn eval(&mut self) {}
    fn named_parameters(&self) -> Vec<(&str, &Parameter<T>)> { Vec::new() }
    fn load_state_dict(&mut self, _state: &StateDict<T>) -> Result<(), FerrotorchError> {
        Err(FerrotorchError::InvalidArgument {
            message: "TracedModule captures weights at trace time; use trace() again with updated model".into(),
        })
    }
    fn state_dict(&self) -> StateDict<T> { StateDict::new() }
}
```

`TracedModule` compiles lazily: the first call to `forward` invokes `Codegen::compile` and caches the result. Subsequent calls reuse the compiled artifact. This avoids paying compilation cost if the module is serialized before execution.

**Profiling** (`profile.rs`):
```rust
pub struct GraphProfiler;

impl GraphProfiler {
    /// Execute a TracedModule with instrumentation enabled.
    pub fn profile<T: Element>(
        module: &TracedModule<T>,
        inputs: &[Tensor<T>],
    ) -> Result<ProfileReport, FerrotorchError> { ... }
}

pub struct ProfileReport {
    pub node_timings: Vec<NodeTiming>,       // Per-node wall time
    pub memory_usage: Vec<NodeMemory>,       // Per-node allocation size
    pub fusion_regions: Vec<FusionRegion>,   // Fused op boundaries
    pub total_time: std::time::Duration,
    pub peak_memory_bytes: usize,
}
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `GradFn<T>`, `Device`, `FerrotorchError`, autograd graph traversal |
| `ferrotorch-nn` | workspace | `Module<T>` trait (for `TracedModule` impl) |
| `cranelift-codegen` | 0.115 | CPU native code generation (initial backend) |
| `cranelift-module` | 0.115 | JIT module management for Cranelift |
| `cranelift-jit` | 0.115 | In-process JIT compilation |
| `thiserror` | 2.0 | Error derive macros |
| `serde` | 1.0 | IR graph serialization |
| `rmp-serde` | 1.3 | MessagePack binary format for IR serialization |

### Error Variants

`error.rs` extends `FerrotorchError` with JIT-specific variants:

```rust
#[derive(Debug, thiserror::Error)]
pub enum JitError {
    #[error("tracing error: {message}")]
    TracingError { message: String },
    #[error("data-dependent control flow detected at op '{op}': tracing requires static control flow")]
    DataDependentControlFlow { op: String },
    #[error("unsupported operation during tracing: {op}")]
    UnsupportedOp { op: String },
    #[error("shape mismatch: traced with {traced:?}, called with {actual:?}")]
    ShapeMismatch { traced: Vec<usize>, actual: Vec<usize> },
    #[error("codegen error: {message}")]
    CodegenError { message: String },
    #[error("serialization error: {message}")]
    SerializationError { message: String },
}

impl From<JitError> for FerrotorchError {
    fn from(e: JitError) -> Self {
        FerrotorchError::InvalidArgument { message: e.to_string() }
    }
}
```

### Test Strategy

1. **Tracing correctness**: Trace 15+ model architectures (MLP, CNN patterns, attention, residual connections) and verify the IR node count and connectivity match expectations. Re-execute with varied inputs and compare against eager mode.
2. **Optimization passes**: Unit test each pass in isolation — construct a small IR graph, run one pass, assert the expected transformation occurred (node count, fusion groups, eliminated nodes, memory plan slot count).
3. **Codegen numerical parity**: For every `IrOpKind`, compile a single-node graph through Cranelift and compare output against eager execution using the same tolerances as ferrotorch-core (rtol=1e-4 for f32, rtol=1e-7 for f64).
4. **Backward correctness**: Trace forward+backward for 10+ graphs, compute gradients through the traced backward graph, and verify against eager-mode `backward()` output.
5. **Serialization round-trip**: Serialize and deserialize IR graphs, verify the restored graph produces identical output.
6. **Error paths**: Verify that data-dependent control flow, unsupported ops, shape mismatches at execution time, and codegen failures all produce the correct `JitError` variant.
7. **Profiling**: Run the profiler on a traced graph and verify the report contains timing entries for every node and correct fusion group boundaries.
8. **Determinism**: Trace the same function 10 times and assert all IR graphs are byte-identical after serialization.

### Out of Scope

- Scripting (source-level analysis or AST transformation) — ferrotorch-jit is tracing-only, not a Rust-to-IR compiler
- Custom operator registration for the JIT — third-party ops must go through the standard `GradFn<T>` trait and are automatically captured during tracing
- Quantization-aware tracing (int8/int4 inference) — this belongs in a future ferrotorch-quantize crate
- Multi-device graph partitioning (splitting a single traced graph across CPU and GPU) — ferrotorch-distributed handles device placement
- ONNX export from IR — ferrotorch-serialize already handles ONNX; the JIT IR is an internal representation
- Metal and Vulkan codegen backends — these can be added later via the `Codegen` trait but are not in Phase 8's scope
- Automatic differentiation of the compiled code itself — backward support works by tracing the backward pass into a separate IR graph during the original trace, not by differentiating generated machine code

