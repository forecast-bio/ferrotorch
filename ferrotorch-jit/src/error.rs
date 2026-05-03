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
    #[error("tracing error: {message}")]
    TracingError { message: String },

    #[error(
        "data-dependent control flow detected at op '{op}': tracing requires static control flow"
    )]
    DataDependentControlFlow { op: String },

    #[error("unsupported operation during tracing: {op}")]
    UnsupportedOp { op: String },

    #[error("shape mismatch: traced with {traced:?}, called with {actual:?}")]
    ShapeMismatch {
        traced: Vec<usize>,
        actual: Vec<usize>,
    },

    #[error("codegen error: {message}")]
    CodegenError { message: String },

    #[error("serialization error: {message}")]
    SerializationError { message: String },

    #[error("graph break at op '{op}': {reason}")]
    GraphBreak { op: String, reason: String },

    #[error(
        "export error at op '{op}': {reason} (export mode requires fullgraph — no graph breaks allowed)"
    )]
    ExportError { op: String, reason: String },

    #[error("parameter error: {message}")]
    ParameterError { message: String },

    #[error("recompilation failed for shape {shape:?}: {message}")]
    RecompilationError {
        shape: Vec<Vec<usize>>,
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
    GpuBackendUnavailable { target: String, reason: String },
}

impl From<JitError> for FerrotorchError {
    fn from(e: JitError) -> Self {
        FerrotorchError::InvalidArgument {
            message: e.to_string(),
        }
    }
}
