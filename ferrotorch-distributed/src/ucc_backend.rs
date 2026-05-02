//! UCC backend skeleton (#459).
//!
//! [UCC](https://github.com/openucx/ucc) (Unified Collective Communication)
//! is NVIDIA / Mellanox's modern unified collectives library, optimised for
//! InfiniBand and NVLink. PyTorch added a UCC backend in 1.13 alongside the
//! existing NCCL/Gloo/MPI options.
//!
//! # Status
//!
//! API-contract-only; `ucc-backend` feature flag is default off. The real
//! bindings need the UCC C library at compile time, which isn't generally
//! available outside HPC clusters and CUDA-enabled CI runners.
//!
//! # Why a skeleton
//!
//! UCC is the newest of the four backends and has the smallest install
//! footprint outside HPC environments. Like the other skeletons here, it
//! exists to nail down the public API so downstream code can write
//! `UccBackend::new(...)` paths now without waiting for the binding crate.

use std::time::Duration;

use ferrotorch_core::FerrotorchResult;

use crate::backend::Backend;
use crate::error::DistributedError;

/// Returns `true` when this build was compiled with the `ucc-backend`
/// feature enabled. Always `false` otherwise.
pub fn is_ucc_available() -> bool {
    cfg!(feature = "ucc-backend")
}

/// Skeleton UCC backend handle. Construction fails with
/// [`DistributedError::BackendUnavailable`] on non-UCC builds.
#[derive(Debug)]
pub struct UccBackend {
    rank: usize,
    world_size: usize,
}

impl UccBackend {
    /// Try to initialise a UCC backend. On non-UCC builds, this returns
    /// [`DistributedError::BackendUnavailable`].
    pub fn new(rank: usize, world_size: usize) -> FerrotorchResult<Self> {
        if !is_ucc_available() {
            return Err(DistributedError::BackendUnavailable {
                backend: "ucc",
            }
            .into());
        }
        Ok(Self { rank, world_size })
    }
}

impl Backend for UccBackend {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, _data: &[u8], _dst_rank: usize) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "ucc" }.into())
    }

    fn recv(&self, _dst: &mut [u8], _src_rank: usize) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "ucc" }.into())
    }

    fn recv_timeout(
        &self,
        _dst: &mut [u8],
        _src_rank: usize,
        _timeout: Duration,
    ) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "ucc" }.into())
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "ucc" }.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ucc_unavailable_without_feature() {
        if !is_ucc_available() {
            assert!(UccBackend::new(0, 2).is_err());
        }
    }
}
