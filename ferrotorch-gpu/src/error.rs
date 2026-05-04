//! Error type for the GPU backend.
//!
//! [`GpuError`] is the canonical fallible-result type for everything in this
//! crate; [`GpuResult<T>`] aliases `Result<T, GpuError>`. The variants
//! correspond 1:1 with the failure modes the rest of the crate produces. The
//! cudarc-error wrappers are gated behind the `cuda` feature so the crate can
//! still build (with stubs) when the feature is disabled.

/// Errors produced by GPU operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum GpuError {
    /// CUDA driver error forwarded from cudarc.
    #[cfg(feature = "cuda")]
    #[error("CUDA driver error: {0}")]
    Driver(#[from] cudarc::driver::DriverError),

    /// Attempted a GPU operation but the `cuda` feature is not enabled.
    #[cfg(not(feature = "cuda"))]
    #[error("GPU operations require the `cuda` feature")]
    NoCudaFeature,

    /// Device ordinal is out of range.
    #[error("invalid device ordinal {ordinal} (only {count} devices available)")]
    InvalidDevice {
        /// The requested ordinal.
        ordinal: usize,
        /// The number of devices actually present.
        count: usize,
    },

    /// Tried to operate on buffers from different devices.
    #[error("device mismatch: expected cuda:{expected}, got cuda:{got}")]
    DeviceMismatch {
        /// Ordinal the operation expected.
        expected: usize,
        /// Ordinal the operand was actually on.
        got: usize,
    },

    /// GPU out of memory. Contains the requested size and the free bytes at
    /// the time of the failed allocation.
    #[error(
        "GPU out of memory: requested {requested_bytes} bytes but only \
         {free_bytes} bytes free"
    )]
    OutOfMemory {
        /// Allocation size in bytes that the caller requested.
        requested_bytes: usize,
        /// Free bytes reported by the driver at the time of failure.
        free_bytes: usize,
    },

    /// Allocation rejected because it would exceed the user-configured memory
    /// budget (see [`crate::memory_guard::MemoryGuard::set_budget`]).
    #[error(
        "memory budget exceeded: requested {requested_bytes} bytes, \
         budget is {budget_bytes} bytes with {used_bytes} bytes already used"
    )]
    BudgetExceeded {
        /// Allocation size in bytes that the caller requested.
        requested_bytes: usize,
        /// Configured budget ceiling in bytes.
        budget_bytes: usize,
        /// Bytes already accounted for as live by the guard.
        used_bytes: usize,
    },

    /// Binary op received buffers with different lengths.
    #[error("buffer length mismatch: {a} vs {b}")]
    LengthMismatch {
        /// Length of the first operand.
        a: usize,
        /// Length of the second operand.
        b: usize,
    },

    /// Matrix multiplication shape mismatch (inner dimensions differ).
    #[error("{op}: shape mismatch, expected {expected:?}, got {got:?}")]
    ShapeMismatch {
        /// Operation name (`"matmul"`, `"bmm"`, etc).
        op: &'static str,
        /// Shape the operation expected.
        expected: Vec<usize>,
        /// Shape that was actually supplied.
        got: Vec<usize>,
    },

    /// cuBLAS error forwarded from cudarc.
    #[cfg(feature = "cuda")]
    #[error("cuBLAS error: {0}")]
    Blas(#[from] cudarc::cublas::result::CublasError),

    /// cuSOLVER error forwarded from cudarc.
    #[cfg(feature = "cuda")]
    #[error("cuSOLVER error: {0}")]
    Solver(#[from] cudarc::cusolver::result::CusolverError),

    /// cuFFT error forwarded from cudarc.
    #[cfg(feature = "cuda")]
    #[error("cuFFT error: {0}")]
    Fft(#[from] cudarc::cufft::result::CufftError),

    /// PTX kernel compilation failed (e.g. unsupported GPU architecture, or
    /// invalid PTX rejected by the JIT). Preserves the underlying cudarc error
    /// so callers can diagnose specifically *why* the JIT rejected the kernel.
    #[cfg(feature = "cuda")]
    #[error("PTX kernel compilation failed for `{kernel}`: {source}")]
    PtxCompileFailed {
        /// Name of the kernel whose PTX source the JIT rejected.
        kernel: &'static str,
        /// Underlying cudarc driver error returned by the JIT.
        #[source]
        source: cudarc::driver::DriverError,
    },

    /// The op is not supported on this device for this dtype combination.
    /// Mirrors PyTorch's `NotImplementedError: <op> not implemented for 'CUDA'`
    /// for unsupported (op, dtype, device) combinations.
    #[error("{op} not implemented for '{dtype}' on CUDA")]
    Unsupported {
        /// Operation name.
        op: &'static str,
        /// Dtype that's not supported for this op.
        dtype: &'static str,
    },

    /// An operation was attempted in an invalid state (e.g., capture on a
    /// sealed pool, or cuSOLVER reported a negative info value).
    #[error("invalid state: {message}")]
    InvalidState {
        /// Human-readable description of the invalid state.
        message: String,
    },
}

/// Convenience alias for GPU results.
pub type GpuResult<T> = Result<T, GpuError>;
