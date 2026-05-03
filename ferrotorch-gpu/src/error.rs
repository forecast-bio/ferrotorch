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

    /// cuSOLVER error forwarded from cudarc.
    #[cfg(feature = "cuda")]
    Solver(cudarc::cusolver::result::CusolverError),

    /// cuFFT error forwarded from cudarc.
    #[cfg(feature = "cuda")]
    Fft(cudarc::cufft::result::CufftError),

    /// PTX kernel compilation failed (e.g. unsupported GPU architecture, or
    /// invalid PTX rejected by the JIT). Preserves the underlying cudarc error
    /// so callers can diagnose specifically *why* the JIT rejected the kernel.
    #[cfg(feature = "cuda")]
    PtxCompileFailed {
        kernel: &'static str,
        source: cudarc::driver::DriverError,
    },

    /// The op is not supported on this device for this dtype combination.
    /// Mirrors PyTorch's `NotImplementedError: <op> not implemented for 'CUDA'`
    /// for unsupported (op, dtype, device) combinations.
    Unsupported {
        op: &'static str,
        dtype: &'static str,
    },

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

            #[cfg(feature = "cuda")]
            GpuError::Solver(e) => write!(f, "cuSOLVER error: {e}"),

            #[cfg(feature = "cuda")]
            GpuError::Fft(e) => write!(f, "cuFFT error: {e}"),

            #[cfg(feature = "cuda")]
            GpuError::PtxCompileFailed { kernel, source } => {
                write!(f, "PTX kernel compilation failed for `{kernel}`: {source}")
            }

            GpuError::Unsupported { op, dtype } => {
                write!(f, "{op} not implemented for '{dtype}' on CUDA")
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
            #[cfg(feature = "cuda")]
            GpuError::Solver(e) => Some(e),
            #[cfg(feature = "cuda")]
            GpuError::Fft(e) => Some(e),
            #[cfg(feature = "cuda")]
            GpuError::PtxCompileFailed { source, .. } => Some(source),
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

#[cfg(feature = "cuda")]
impl From<cudarc::cusolver::result::CusolverError> for GpuError {
    fn from(e: cudarc::cusolver::result::CusolverError) -> Self {
        GpuError::Solver(e)
    }
}

#[cfg(feature = "cuda")]
impl From<cudarc::cufft::result::CufftError> for GpuError {
    fn from(e: cudarc::cufft::result::CufftError) -> Self {
        GpuError::Fft(e)
    }
}

/// Convenience alias for GPU results.
pub type GpuResult<T> = Result<T, GpuError>;
