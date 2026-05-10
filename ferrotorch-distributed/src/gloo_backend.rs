//! Gloo backend skeleton (tracked in #1132).
//!
//! [Gloo](https://github.com/facebookincubator/gloo) is the standard
//! CPU-only collective-communication library used by PyTorch's
//! `torch.distributed` when NCCL isn't available (or when ranks include
//! CPU-only nodes). Architecturally it sits parallel to NCCL: a stand-alone
//! C++ library exposing allreduce / allgather / broadcast / scatter / etc.
//! with TCP, IB-verbs, and uv backends.
//!
//! # Status
//!
//! This module ships the **API contract** so callers can already write
//! `Backend::Gloo` paths and get a clear runtime error when the actual
//! Gloo bindings aren't compiled in. Default off via the `gloo-backend`
//! feature flag — non-Gloo builds (the workspace's primary CI target) get
//! the "unavailable" path at runtime without the C++ dep tree.
//!
//! # Why a skeleton
//!
//! The full Gloo binding would need:
//! 1. A `gloo-sys` crate with the C++ FFI (Gloo is C++; needs `bindgen`
//!    + a `cmake`-driven build of libgloo).
//! 2. A Rust wrapper exposing the collective ops we use today (allreduce,
//!    allgather, broadcast, barrier, send/recv).
//! 3. CI coverage on Linux + macOS — Gloo doesn't ship Windows officially.
//!
//! That's a 2000+ LOC effort with a real C++ dep, so this skeleton lets
//! the public API stabilise first. Tracked separately as #1132
//! (replaces closed #459).

use std::time::Duration;

use ferrotorch_core::FerrotorchResult;

use crate::backend::Backend;
use crate::error::DistributedError;

/// Returns `true` when this build was compiled with the `gloo-backend`
/// feature enabled. Always `false` otherwise.
pub fn is_gloo_available() -> bool {
    cfg!(feature = "gloo-backend")
}

/// Skeleton Gloo backend handle. Construction fails with
/// [`DistributedError::BackendUnavailable`] on non-Gloo builds.
#[derive(Debug)]
pub struct GlooBackend {
    rank: usize,
    world_size: usize,
}

impl GlooBackend {
    /// Try to initialise a Gloo backend for the given rank / world size.
    ///
    /// Returns [`DistributedError::BackendUnavailable`] when the
    /// `gloo-backend` feature is off.
    pub fn new(rank: usize, world_size: usize) -> FerrotorchResult<Self> {
        if !is_gloo_available() {
            return Err(DistributedError::BackendUnavailable { backend: "gloo" }.into());
        }
        Ok(Self { rank, world_size })
    }
}

impl Backend for GlooBackend {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, _data: &[u8], _dst_rank: usize) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
    }

    fn recv(&self, _dst: &mut [u8], _src_rank: usize) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
    }

    fn recv_timeout(
        &self,
        _dst: &mut [u8],
        _src_rank: usize,
        _timeout: Duration,
    ) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        Err(DistributedError::BackendUnavailable { backend: "gloo" }.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::FerrotorchError;

    #[test]
    fn gloo_unavailable_without_feature() {
        // Non-vacuous discrimination: when the `gloo-backend` feature is
        // off (the default), construction must fail with a
        // `DistributedError::BackendUnavailable { backend: "gloo" }`,
        // which converts to `FerrotorchError::InvalidArgument { message }`
        // whose `message` carries the backend name.
        if is_gloo_available() {
            // Feature is on; the unavailable contract doesn't apply.
            return;
        }
        let err = GlooBackend::new(0, 2).expect_err("default build must err");
        match err {
            FerrotorchError::InvalidArgument { ref message } => {
                assert!(
                    message.contains("`gloo`"),
                    "expected message to discriminate the gloo backend by name, got: {message}"
                );
                assert!(
                    !message.contains("`mpi`") && !message.contains("`ucc`"),
                    "message must not name a different backend, got: {message}"
                );
            }
            other => panic!(
                "expected FerrotorchError::InvalidArgument from BackendUnavailable, got {other:?}"
            ),
        }
    }

    #[test]
    fn is_gloo_available_default_off() {
        // The default workspace build does not enable `gloo-backend`, so
        // this returns false. A future Gloo build would flip this.
        if !cfg!(feature = "gloo-backend") {
            assert!(!is_gloo_available());
        }
    }
}
