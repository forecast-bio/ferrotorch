use core::fmt;

/// Errors produced by GPU operations.
#[derive(Debug)]
#[non_exhaustive]
pub enum GpuError {
    /// CUDA driver error forwarded from cudarc.
    #[cfg(feature = "cuda")]
    Driver(cudarc::driver::DriverError),

    /// Attempted a GPU operation but the `cuda` feature is not enabled.
    #[cfg(not(feature = "cuda"))]
    NoCudaFeature,

    /// Device ordinal is out of range.
    InvalidDevice { ordinal: usize, count: usize },

    /// Tried to operate on buffers from different devices.
    DeviceMismatch { expected: usize, got: usize },

    /// GPU out of memory. Contains the requested size and the free bytes at
    /// the time of the failed allocation.
    OutOfMemory {
        requested_bytes: usize,
        free_bytes: usize,
    },

    /// Allocation rejected because it would exceed the user-configured memory
    /// budget (see [`crate::memory_guard::MemoryGuard::set_budget`]).
    BudgetExceeded {
        requested_bytes: usize,
        budget_bytes: usize,
        used_bytes: usize,
    },

    /// Binary op received buffers with different lengths.
    LengthMismatch { a: usize, b: usize },

    /// Matrix multiplication shape mismatch (inner dimensions differ).
    ShapeMismatch {
        op: &'static str,
        expected: Vec<usize>,
        got: Vec<usize>,
    },

    /// cuBLAS error forwarded from cudarc.
    #[cfg(feature = "cuda")]
    Blas(cudarc::cublas::result::CublasError),

    /// PTX kernel compilation failed (e.g. unsupported GPU architecture).
    PtxCompileFailed { kernel: &'static str },

    /// An operation was attempted in an invalid state (e.g., capture on a
    /// sealed pool, or cuSOLVER reported a negative info value).
    InvalidState { message: String },
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "cuda")]
            GpuError::Driver(e) => write!(f, "CUDA driver error: {e}"),

            #[cfg(not(feature = "cuda"))]
            GpuError::NoCudaFeature => {
                write!(f, "GPU operations require the `cuda` feature")
            }

            GpuError::InvalidDevice { ordinal, count } => {
                write!(
                    f,
                    "invalid device ordinal {ordinal} (only {count} devices available)"
                )
            }

            GpuError::DeviceMismatch { expected, got } => {
                write!(
                    f,
                    "device mismatch: expected cuda:{expected}, got cuda:{got}"
                )
            }

            GpuError::OutOfMemory {
                requested_bytes,
                free_bytes,
            } => {
                write!(
                    f,
                    "GPU out of memory: requested {requested_bytes} bytes but only \
                     {free_bytes} bytes free"
                )
            }

            GpuError::BudgetExceeded {
                requested_bytes,
                budget_bytes,
                used_bytes,
            } => {
                write!(
                    f,
                    "memory budget exceeded: requested {requested_bytes} bytes, \
                     budget is {budget_bytes} bytes with {used_bytes} bytes already used"
                )
            }

            GpuError::LengthMismatch { a, b } => {
                write!(f, "buffer length mismatch: {a} vs {b}")
            }

            GpuError::ShapeMismatch { op, expected, got } => {
                write!(
                    f,
                    "{op}: shape mismatch, expected {expected:?}, got {got:?}"
                )
            }

            #[cfg(feature = "cuda")]
            GpuError::Blas(e) => write!(f, "cuBLAS error: {e}"),

            GpuError::PtxCompileFailed { kernel } => {
                write!(f, "PTX kernel compilation failed: {kernel}")
            }

            GpuError::InvalidState { message } => {
                write!(f, "invalid state: {message}")
            }
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            #[cfg(feature = "cuda")]
            GpuError::Driver(e) => Some(e),
            #[cfg(feature = "cuda")]
            GpuError::Blas(e) => Some(e),
            _ => None,
        }
    }
}

#[cfg(feature = "cuda")]
impl From<cudarc::driver::DriverError> for GpuError {
    fn from(e: cudarc::driver::DriverError) -> Self {
        GpuError::Driver(e)
    }
}

#[cfg(feature = "cuda")]
impl From<cudarc::cublas::result::CublasError> for GpuError {
    fn from(e: cudarc::cublas::result::CublasError) -> Self {
        GpuError::Blas(e)
    }
}

/// Convenience alias for GPU results.
pub type GpuResult<T> = Result<T, GpuError>;
