//! MPI backend skeleton (#459).
//!
//! [MPI](https://www.mpi-forum.org/) (Message Passing Interface) is the
//! HPC-standard collective library, with implementations like Open MPI,
//! MPICH, and MVAPICH. PyTorch's `torch.distributed` ships an MPI
//! backend that's the most common choice on supercomputers and InfiniBand
//! clusters.
//!
//! # Status
//!
//! API-contract-only; `mpi-backend` feature flag is default off. The
//! skeleton lets callers write `MpiBackend::new(...)` paths now; the
//! real bindings via the `mpi` crate (which itself wraps Open MPI / MPICH
//! C ABI) land in a follow-up.
//!
//! # Why a skeleton
//!
//! The `mpi` Rust crate requires:
//! 1. A working MPI install at compile time (`mpicc` on PATH).
//! 2. The C library available at runtime (linker must find `libmpi.so` or
//!    similar).
//! 3. CI infrastructure with multiple MPI processes — substantially
//!    different from typical cargo-test setups.
//!
//! Adding the dep unconditionally would break workspace builds on systems
//! without MPI. Hence the feature gate.

use std::time::Duration;

use ferrotorch_core::FerrotorchResult;

use crate::backend::Backend;
use crate::error::DistributedError;

/// Returns `true` when this build was compiled with the `mpi-backend`
/// feature enabled. Always `false` otherwise.
pub fn is_mpi_available() -> bool {
    cfg!(feature = "mpi-backend")
}

/// Skeleton MPI backend handle. Construction fails with
/// [`DistributedError::BackendUnavailable`] on non-MPI builds.
#[derive(Debug)]
pub struct MpiBackend {
    rank: usize,
    world_size: usize,
}

impl MpiBackend {
    /// Try to initialise an MPI backend. On non-MPI builds, this returns
    /// [`DistributedError::BackendUnavailable`] without attempting any
    /// MPI call. A real implementation would start with `MPI_Init` /
    /// `MPI_Comm_rank` / `MPI_Comm_size`.
    pub fn new(rank: usize, world_size: usize) -> FerrotorchResult<Self> {
        if !is_mpi_available() {
            return Err(DistributedError::BackendUnavailable {
                backend: "mpi",
            }
            .into());
        }
        Ok(Self { rank, world_size })
    }
}

impl Backend for MpiBackend {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, _data: &[u8], _dst_rank: usize) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
    }

    fn recv(&self, _dst: &mut [u8], _src_rank: usize) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
    }

    fn recv_timeout(
        &self,
        _dst: &mut [u8],
        _src_rank: usize,
        _timeout: Duration,
    ) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "mpi" }.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mpi_unavailable_without_feature() {
        if !is_mpi_available() {
            assert!(MpiBackend::new(0, 2).is_err());
        }
    }
}
