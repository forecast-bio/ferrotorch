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

    #[error("no GPU backend available -- install ferrotorch-gpu and call init()")]
    DeviceUnavailable,

    #[error("cannot access GPU tensor data as CPU slice -- call .cpu() first")]
    GpuTensorNotAccessible,

    #[error("data loading worker panicked: {message}")]
    WorkerPanic { message: String },

    #[error(transparent)]
    Ferray(#[from] ferray_core::FerrayError),
}

/// Convenience alias for ferrotorch results.
pub type FerrotorchResult<T> = Result<T, FerrotorchError>;
