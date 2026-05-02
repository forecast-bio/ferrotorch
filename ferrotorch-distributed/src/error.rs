//! Error types for distributed operations.

use ferrotorch_core::FerrotorchError;

/// Errors specific to the distributed training subsystem.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum DistributedError {
    #[error("invalid world size: {world_size} (must be >= 1)")]
    InvalidWorldSize { world_size: usize },

    #[error("invalid rank {rank} for world size {world_size}")]
    InvalidRank { rank: usize, world_size: usize },

    #[error("cannot send to self (rank {rank})")]
    SelfSend { rank: usize },

    #[error("size mismatch: expected {expected} bytes, got {got}")]
    SizeMismatch { expected: usize, got: usize },

    #[error("I/O error: {message}")]
    Io { message: String },

    #[error("lock poisoned: {message}")]
    LockPoisoned { message: String },

    #[error("channel closed: {message}")]
    ChannelClosed { message: String },

    #[error("unsupported reduce operation: {message}")]
    UnsupportedOp { message: String },

    #[error("operation timed out after {seconds}s")]
    Timeout { seconds: u64 },

    #[error("no connection to rank {rank} (star topology: non-zero ranks only connect to rank 0)")]
    NoConnection { rank: usize },

    /// Returned when the user requested a backend whose binding layer
    /// isn't compiled into this build (e.g. `gloo-backend` / `mpi-backend`
    /// / `ucc-backend` feature off, or a CUDA-required backend on a
    /// non-CUDA system). The caller is expected to either enable the
    /// feature, install the underlying C library, or pick a different
    /// backend (`SimulatedBackend` / `TcpBackend` always work). (#459)
    #[error(
        "backend `{backend}` is not available in this build (enable the corresponding cargo feature \
         and ensure the underlying library is installed)"
    )]
    BackendUnavailable { backend: &'static str },
}

impl From<DistributedError> for FerrotorchError {
    fn from(e: DistributedError) -> Self {
        FerrotorchError::InvalidArgument {
            message: e.to_string(),
        }
    }
}
