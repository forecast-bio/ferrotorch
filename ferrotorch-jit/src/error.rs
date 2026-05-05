//! JIT-specific error types.
//!
//! These errors cover tracing failures, graph breaks, and other conditions
//! specific to the JIT compilation pipeline. [`JitError`] converts into
//! [`FerrotorchError`] via the `From` impl so it integrates seamlessly with
//! the rest of the crate's error handling.

use ferrotorch_core::error::FerrotorchError;

/// Errors specific to the JIT tracing and compilation pipeline.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum JitError {
    /// Generic tracing-pipeline failure carrying a free-form diagnostic.
    #[error("tracing error: {message}")]
    TracingError {
        /// Human-readable description of the failure.
        message: String,
    },

    /// Tracing encountered control flow that depends on a runtime tensor
    /// value; only static control flow is supported.
    #[error(
        "data-dependent control flow detected at op '{op}': tracing requires static control flow"
    )]
    DataDependentControlFlow {
        /// Name of the op that triggered the data-dependent branch.
        op: String,
    },

    /// The op encountered during tracing has no JIT lowering.
    #[error("unsupported operation during tracing: {op}")]
    UnsupportedOp {
        /// Name of the unsupported op.
        op: String,
    },

    /// Inputs at call time disagreed with the shapes captured during tracing.
    #[error("shape mismatch: traced with {traced:?}, called with {actual:?}")]
    ShapeMismatch {
        /// Shape recorded when the graph was traced.
        traced: Vec<usize>,
        /// Shape supplied at the failing call site.
        actual: Vec<usize>,
    },

    /// Backend code generation failed.
    #[error("codegen error: {message}")]
    CodegenError {
        /// Human-readable description of the codegen failure.
        message: String,
    },

    /// Serialising or deserialising a compiled artifact failed.
    #[error("serialization error: {message}")]
    SerializationError {
        /// Human-readable description of the serialization failure.
        message: String,
    },

    /// A traced op forced the JIT to fall back to eager execution
    /// (graph break).
    #[error("graph break at op '{op}': {reason}")]
    GraphBreak {
        /// Name of the op that caused the break.
        op: String,
        /// Why the op cannot be captured in the graph.
        reason: String,
    },

    /// An op blocks export mode (which requires `fullgraph` capture).
    #[error(
        "export error at op '{op}': {reason} (export mode requires fullgraph — no graph breaks allowed)"
    )]
    ExportError {
        /// Name of the op that blocked export.
        op: String,
        /// Why the op cannot be exported.
        reason: String,
    },

    /// An invalid argument was supplied to the JIT API.
    #[error("parameter error: {message}")]
    ParameterError {
        /// Human-readable description of the bad parameter.
        message: String,
    },

    /// Dynamic recompilation for a new input shape failed.
    #[error("recompilation failed for shape {shape:?}: {message}")]
    RecompilationError {
        /// Per-input shapes that triggered the recompilation attempt.
        shape: Vec<Vec<usize>>,
        /// Human-readable description of the failure.
        message: String,
    },

    /// The requested GPU backend is not yet wired to a real GPU runtime in
    /// this build of ferrotorch-jit. Analogous to `PyTorch`'s `NotImplementedError`
    /// for (op, device) combinations that have no registered kernel.
    ///
    /// Callers that want opt-in CPU fallback should catch this variant and
    /// re-dispatch to a CPU backend of their choosing. Per `rust-gpu-discipline`
    /// §3, silent fallback is forbidden; opt-in is the only acceptable form.
    #[error(
        "GPU backend unavailable for target '{target}': {reason} \
         (ferrotorch-jit does not yet wire generated {target} source to a GPU runtime; \
         use a CPU InductorTarget or a ferrotorch-gpu backend instead)"
    )]
    GpuBackendUnavailable {
        /// Name of the requested GPU target (e.g. `"cuda"`, `"wgpu"`).
        target: String,
        /// Why the runtime for this target is not wired up in this build.
        reason: String,
    },

    /// The requested (op, dtype) combination has no GPU codegen path.
    ///
    /// This is the structured analog of `PyTorch`'s `NotImplementedError`
    /// for unsupported (op, dtype, device) combinations. As of #729, GPU
    /// codegen dispatches on `Dtype` for arithmetic, load/store, register
    /// declarations, and constant emission. As of #748 (closing the
    /// Phase-2 follow-up to #729), f64 transcendentals — `exp`, `log`,
    /// `sqrt`, `tanh`, `sigmoid`, `gelu`, `silu`, and `pow` — are
    /// supported on the PTX path *with the `cuda` feature enabled* by
    /// routing through NVRTC + libdevice. Without the `cuda` feature, f64
    /// transcendentals still surface this variant since NVRTC isn't
    /// linked at runtime.
    ///
    /// This variant is also the right return for any future (op, dtype)
    /// combination the codegen genuinely cannot lower — keep messages
    /// pointing at the missing feature flag or follow-up issue so callers
    /// can act on them.
    #[error(
        "ferrotorch-jit GPU codegen does not support op '{op}' on dtype `{dtype}'; \
         enable the `cuda` feature on ferrotorch-jit for f64 transcendentals via libdevice"
    )]
    Unsupported {
        /// Name of the unsupported op (e.g. `"exp"`, `"tanh"`).
        op: String,
        /// Name of the offending dtype (e.g. `"f64"`).
        dtype: String,
    },
}

impl From<JitError> for FerrotorchError {
    fn from(e: JitError) -> Self {
        FerrotorchError::InvalidArgument {
            message: e.to_string(),
        }
    }
}
