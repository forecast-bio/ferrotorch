use crate::device::Device;

/// Errors produced by ferrotorch operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FerrotorchError {
    #[error("shape mismatch: {message}")]
    ShapeMismatch { message: String },

    #[error("device mismatch: expected {expected}, got {got}")]
    DeviceMismatch { expected: Device, got: Device },

    #[error("backward called on non-scalar tensor with shape {shape:?}")]
    BackwardNonScalar { shape: Vec<usize> },

    #[error("no gradient function on non-leaf tensor")]
    NoGradFn,

    #[error("dtype mismatch: expected {expected}, got {got}")]
    DtypeMismatch { expected: String, got: String },

    #[error("index out of bounds: index {index} on axis {axis} with size {size}")]
    IndexOutOfBounds {
        index: usize,
        axis: usize,
        size: usize,
    },

    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },

    #[error("internal lock poisoned: {message}")]
    LockPoisoned { message: String },

    #[error("internal error: {message}")]
    Internal { message: String },

    #[error("no GPU backend available -- install ferrotorch-gpu and call init()")]
    DeviceUnavailable,

    #[error("cannot access GPU tensor data as CPU slice -- call .cpu() first")]
    GpuTensorNotAccessible,

    #[error("{op} is not supported on CUDA tensors -- call .cpu() first")]
    NotImplementedOnCuda { op: &'static str },

    /// A backend-specific GPU error wrapped for cross-crate propagation.
    ///
    /// `ferrotorch-core` cannot depend on `ferrotorch-gpu` (would create a
    /// workspace dep cycle), so concrete `GpuError` instances are
    /// type-erased through `Box<dyn Error>`. Callers can recover the
    /// original `GpuError` via [`std::error::Error::source`] +
    /// [`downcast_ref`](std::error::Error::downcast_ref):
    ///
    /// ```ignore
    /// use std::error::Error;
    /// use ferrotorch_core::FerrotorchError;
    /// // ferrotorch_gpu::GpuError comes from a separate crate.
    ///
    /// fn handle(e: FerrotorchError) {
    ///     if let FerrotorchError::Gpu { source } = &e {
    ///         if let Some(gpu_err) =
    ///             source.downcast_ref::<ferrotorch_gpu::GpuError>()
    ///         {
    ///             // ... pattern-match on gpu_err's variants ...
    ///         }
    ///     }
    /// }
    /// ```
    ///
    /// The `'static` bound is required so `downcast_ref::<T>` works; the
    /// `Send + Sync` bounds keep `FerrotorchError: Send + Sync` (every
    /// `Result`-returning workspace fn relies on that).
    #[error("gpu error: {source}")]
    Gpu {
        /// The source error (typically a `ferrotorch_gpu::GpuError` or a
        /// `cudarc::driver::DriverError`, but any
        /// `Error + Send + Sync + 'static` is acceptable). Recover the
        /// original type via [`std::error::Error::source`] +
        /// [`downcast_ref`](std::error::Error::downcast_ref).
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    #[error("data loading worker panicked: {message}")]
    WorkerPanic { message: String },

    #[error(transparent)]
    Ferray(#[from] ferray_core::FerrayError),
}

/// Convenience alias for ferrotorch results.
pub type FerrotorchResult<T> = Result<T, FerrotorchError>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;

    #[derive(Debug, thiserror::Error)]
    #[error("test error: {0}")]
    struct TestError(&'static str);

    #[test]
    fn gpu_variant_preserves_source_chain() {
        let inner = TestError("backend kernel failed");
        let outer = FerrotorchError::Gpu {
            source: Box::new(inner),
        };
        let source = outer
            .source()
            .expect("source must be set via #[source]");
        let downcast = source
            .downcast_ref::<TestError>()
            .expect("downcast back to TestError");
        assert_eq!(downcast.0, "backend kernel failed");
    }

    #[test]
    fn gpu_variant_display() {
        let inner = TestError("oom");
        let outer = FerrotorchError::Gpu {
            source: Box::new(inner),
        };
        assert_eq!(outer.to_string(), "gpu error: test error: oom");
    }
}
