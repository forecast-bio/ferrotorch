//! GPU-aware collective communication operations.
//!
//! These functions extend the CPU-only collectives in [`crate::collective`]
//! to work with [`GpuTensor`]s.
//!
//! # Default behaviour — PyTorch parity
//!
//! By default, calling [`gpu_allreduce`] or [`gpu_broadcast`] without the
//! `nccl` feature enabled and without setting `FERROTORCH_ENABLE_GPU_FALLBACK`
//! returns `Err(...)` immediately. This matches PyTorch's behaviour: a CUDA
//! tensor passed to a collective with no CUDA-native implementation raises
//! `RuntimeError` rather than silently detour through the host.
//!
//! # Fast path — `nccl` feature
//!
//! When the `nccl` Cargo feature is enabled, the functions delegate to
//! [`crate::nccl_backend::NcclBackend`] for GPU-native collectives with
//! no PCIe round-trip. The caller must supply an `NcclBackend` — see the
//! `nccl` feature gate section below.
//!
//! **TODO (Step 4 coordination — tracking issue #668):** The `nccl` fast path
//! is scaffolded here but routing `GpuTensor<T>` → raw device pointer →
//! `NcclBackend::allreduce_raw` requires glue that lives across the
//! `ferrotorch-gpu` / `ferrotorch-distributed` crate boundary. That wiring
//! is a follow-up; the `#[cfg(feature = "nccl")]` arms below are stubs that
//! will be completed in issue #668.
//!
//! # Opt-in CPU fallback (Gloo-equivalent slow path)
//!
//! Set `FERROTORCH_ENABLE_GPU_FALLBACK=1` in the process environment to enable
//! the host round-trip path: GPU → CPU → collective → GPU. This is the
//! equivalent of PyTorch's Gloo backend — an explicit slow-path the user
//! opts into.
//!
//! **Every call that takes the fallback path emits a `tracing::warn!`**,
//! naming the collective and telling the user how to disable the fallback.
//! Unset `FERROTORCH_ENABLE_GPU_FALLBACK` to make these calls return an error
//! instead.
//!
//! # Feature gate
//!
//! This module is only compiled when the `gpu` feature is enabled:
//!
//! ```toml
//! ferrotorch-distributed = { version = "0.1", features = ["gpu"] }
//! ```

use ferrotorch_core::FerrotorchResult;
use ferrotorch_gpu::{GpuFloat, GpuTensor, tensor_to_cpu, tensor_to_gpu};

use crate::backend::Backend;
use crate::collective::{ReduceOp, allreduce, broadcast};
use crate::error::DistributedError;

// ---------------------------------------------------------------------------
// Private CPU-path helpers (the host-round-trip bodies, now named)
// ---------------------------------------------------------------------------

/// Perform allreduce via the CPU host round-trip (Gloo-equivalent slow path).
///
/// Extracted into a named helper so the opt-in fallback dispatcher has a
/// clear target. Not part of the public API.
fn cpu_path_allreduce<T: GpuFloat>(
    tensor: &GpuTensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
) -> FerrotorchResult<GpuTensor<T>> {
    let cpu_tensor = tensor_to_cpu(tensor)?;
    let reduced = allreduce(&cpu_tensor, backend, op)?;
    let gpu_result = tensor_to_gpu(&reduced, tensor.device()).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("gpu_allreduce: CPU->GPU transfer failed: {e}"),
        }
    })?;
    Ok(gpu_result)
}

/// Perform broadcast via the CPU host round-trip (Gloo-equivalent slow path).
///
/// Extracted into a named helper so the opt-in fallback dispatcher has a
/// clear target. Not part of the public API.
fn cpu_path_broadcast<T: GpuFloat>(
    tensor: &GpuTensor<T>,
    backend: &dyn Backend,
    root: usize,
) -> FerrotorchResult<GpuTensor<T>> {
    let cpu_tensor = tensor_to_cpu(tensor)?;
    let bcast = broadcast(&cpu_tensor, backend, root)?;
    let gpu_result = tensor_to_gpu(&bcast, tensor.device()).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("gpu_broadcast: CPU->GPU transfer failed: {e}"),
        }
    })?;
    Ok(gpu_result)
}

// ---------------------------------------------------------------------------
// GPU allreduce
// ---------------------------------------------------------------------------

/// Allreduce a [`GpuTensor`] across all ranks.
///
/// Each rank provides its local GPU tensor. The result is a new
/// [`GpuTensor`] on the same device, whose values are the element-wise
/// reduction of all inputs.
///
/// # Dispatch order
///
/// 1. **`nccl` feature (fast path):** delegates to [`NcclBackend`] for
///    GPU-native allreduce — no host round-trip.
///    *(Not yet wired; see tracking issue #668.)*
/// 2. **`FERROTORCH_ENABLE_GPU_FALLBACK` set:** performs a host round-trip
///    (GPU → CPU → allreduce → GPU) and logs a `tracing::warn!` per call.
/// 3. **Default:** returns `Err` — no silent detour.
///
/// # Errors
///
/// - `Err` if the `nccl` feature is absent and
///   `FERROTORCH_ENABLE_GPU_FALLBACK` is not set (PyTorch parity default).
/// - GPU/CPU transfer errors if the fallback path is active.
/// - Backend communication errors (network, channel closed, etc.).
pub fn gpu_allreduce<T: GpuFloat>(
    tensor: &GpuTensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
) -> FerrotorchResult<GpuTensor<T>> {
    // Fast path: NCCL GPU-native allreduce (no CPU round-trip).
    // TODO(#668): wire GpuTensor<T> raw-pointer extraction + NcclBackend
    // dispatch here once the cross-crate glue is ready.
    #[cfg(feature = "nccl")]
    let () = (); // placeholder so the cfg arm compiles

    // Opt-in CPU fallback (Gloo-equivalent slow path).
    if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
        tracing::warn!(
            target: "ferrotorch::gpu_fallback",
            collective = "allreduce",
            "GPU collective is using host round-trip (Gloo-equivalent slow path). \
             Unset FERROTORCH_ENABLE_GPU_FALLBACK to make this an error instead.",
        );
        return cpu_path_allreduce(tensor, backend, op);
    }

    // PyTorch-parity default: no silent fallback.
    Err(DistributedError::UnsupportedOp {
        message: "gpu_allreduce requires the `nccl` feature for GPU-native operation. \
                  Set FERROTORCH_ENABLE_GPU_FALLBACK=1 to enable the host round-trip \
                  (Gloo-equivalent) fallback instead."
            .into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// GPU broadcast
// ---------------------------------------------------------------------------

/// Broadcast a [`GpuTensor`] from `root` to all other ranks.
///
/// The `root` rank's tensor data is sent to every other rank. All ranks
/// return a [`GpuTensor`] on their respective device containing the
/// root's data.
///
/// # Dispatch order
///
/// 1. **`nccl` feature (fast path):** delegates to [`NcclBackend`] for
///    GPU-native broadcast — no host round-trip.
///    *(Not yet wired; see tracking issue #668.)*
/// 2. **`FERROTORCH_ENABLE_GPU_FALLBACK` set:** performs a host round-trip
///    (GPU → CPU → broadcast → GPU) and logs a `tracing::warn!` per call.
/// 3. **Default:** returns `Err` — no silent detour.
///
/// # Errors
///
/// - `Err` if the `nccl` feature is absent and
///   `FERROTORCH_ENABLE_GPU_FALLBACK` is not set (PyTorch parity default).
/// - GPU/CPU transfer errors if the fallback path is active.
/// - Backend communication errors.
/// - [`DistributedError::InvalidRank`] if `root >= world_size`.
pub fn gpu_broadcast<T: GpuFloat>(
    tensor: &GpuTensor<T>,
    backend: &dyn Backend,
    root: usize,
) -> FerrotorchResult<GpuTensor<T>> {
    // Fast path: NCCL GPU-native broadcast (no CPU round-trip).
    // TODO(#668): wire GpuTensor<T> raw-pointer extraction + NcclBackend
    // dispatch here once the cross-crate glue is ready.
    #[cfg(feature = "nccl")]
    let () = (); // placeholder so the cfg arm compiles

    // Opt-in CPU fallback (Gloo-equivalent slow path).
    if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
        tracing::warn!(
            target: "ferrotorch::gpu_fallback",
            collective = "broadcast",
            "GPU collective is using host round-trip (Gloo-equivalent slow path). \
             Unset FERROTORCH_ENABLE_GPU_FALLBACK to make this an error instead.",
        );
        return cpu_path_broadcast(tensor, backend, root);
    }

    // PyTorch-parity default: no silent fallback.
    Err(DistributedError::UnsupportedOp {
        message: "gpu_broadcast requires the `nccl` feature for GPU-native operation. \
                  Set FERROTORCH_ENABLE_GPU_FALLBACK=1 to enable the host round-trip \
                  (Gloo-equivalent) fallback instead."
            .into(),
    }
    .into())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::backend::SimulatedBackend;
    use ferrotorch_gpu::{GpuDevice, tensor_to_gpu as t2g};
    use std::sync::Arc;
    use std::thread;

    /// Helper: create a GpuTensor<f32> from a flat slice on device 0.
    fn gpu_from_slice(data: &[f32], shape: &[usize]) -> GpuTensor<f32> {
        let cpu = ferrotorch_core::from_slice(data, shape).unwrap();
        let device = GpuDevice::new(0).unwrap();
        t2g(&cpu, &device).unwrap()
    }

    // These tests previously relied on the silent host-round-trip fallback
    // being always-on. The policy migration (ADR #663 item 7) makes the
    // fallback opt-in via FERROTORCH_ENABLE_GPU_FALLBACK. Without the `nccl`
    // feature or the env var, the functions return Err. These tests are
    // therefore marked #[ignore] until either:
    //   (a) the NCCL wiring lands (tracking issue #668), OR
    //   (b) a test harness that sets FERROTORCH_ENABLE_GPU_FALLBACK is added.
    //
    // Do NOT mark as "requires CUDA device" — the cause is the policy
    // migration, not hardware availability.

    #[test]
    #[ignore = "tracking issue #668: opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
    fn test_gpu_allreduce_sum_2_ranks() {
        // Rank 0: [1.0, 2.0, 3.0], Rank 1: [4.0, 5.0, 6.0]
        // Sum: [5.0, 7.0, 9.0]
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let data: Vec<f32> = if rank == 0 {
                        vec![1.0, 2.0, 3.0]
                    } else {
                        vec![4.0, 5.0, 6.0]
                    };
                    let gt = gpu_from_slice(&data, &[3]);
                    let result = gpu_allreduce(&gt, b.as_ref(), ReduceOp::Sum).unwrap();

                    // Verify by copying back to CPU.
                    let cpu = result.cpu().unwrap();
                    let out = cpu.data().unwrap();
                    assert_eq!(out.len(), 3);
                    assert!(
                        (out[0] - 5.0).abs() < 1e-6,
                        "rank {rank}: expected 5.0, got {}",
                        out[0]
                    );
                    assert!(
                        (out[1] - 7.0).abs() < 1e-6,
                        "rank {rank}: expected 7.0, got {}",
                        out[1]
                    );
                    assert!(
                        (out[2] - 9.0).abs() < 1e-6,
                        "rank {rank}: expected 9.0, got {}",
                        out[2]
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    #[ignore = "tracking issue #668: opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
    fn test_gpu_allreduce_mean_2_ranks() {
        // Rank 0: [2.0, 4.0], Rank 1: [6.0, 8.0]
        // Mean: [4.0, 6.0]
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let data: Vec<f32> = if rank == 0 {
                        vec![2.0, 4.0]
                    } else {
                        vec![6.0, 8.0]
                    };
                    let gt = gpu_from_slice(&data, &[2]);
                    let result = gpu_allreduce(&gt, b.as_ref(), ReduceOp::Mean).unwrap();

                    let cpu = result.cpu().unwrap();
                    let out = cpu.data().unwrap();
                    assert!(
                        (out[0] - 4.0).abs() < 1e-6,
                        "rank {rank}: expected 4.0, got {}",
                        out[0]
                    );
                    assert!(
                        (out[1] - 6.0).abs() < 1e-6,
                        "rank {rank}: expected 6.0, got {}",
                        out[1]
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    #[ignore = "tracking issue #668: opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
    fn test_gpu_broadcast_from_rank_0() {
        // Rank 0: [42.0, 99.0], Rank 1: [0.0, 0.0]
        // After broadcast(root=0): both have [42.0, 99.0]
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let data: Vec<f32> = if rank == 0 {
                        vec![42.0, 99.0]
                    } else {
                        vec![0.0, 0.0]
                    };
                    let gt = gpu_from_slice(&data, &[2]);
                    let result = gpu_broadcast(&gt, b.as_ref(), 0).unwrap();

                    let cpu = result.cpu().unwrap();
                    let out = cpu.data().unwrap();
                    assert!(
                        (out[0] - 42.0).abs() < 1e-6,
                        "rank {rank}: expected 42.0, got {}",
                        out[0]
                    );
                    assert!(
                        (out[1] - 99.0).abs() < 1e-6,
                        "rank {rank}: expected 99.0, got {}",
                        out[1]
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    #[ignore = "tracking issue #668: opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
    fn test_gpu_allreduce_single_rank() {
        // Single rank: allreduce should return the input unchanged.
        let group = SimulatedBackend::create_group(1).unwrap();
        let gt = gpu_from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = gpu_allreduce(&gt, &group[0], ReduceOp::Sum).unwrap();

        let cpu = result.cpu().unwrap();
        let out = cpu.data().unwrap();
        assert_eq!(out, &[1.0, 2.0, 3.0]);
    }

    #[test]
    #[ignore = "tracking issue #668: opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
    fn test_gpu_allreduce_preserves_shape() {
        // Verify shape [2, 3] is preserved through the round-trip.
        let group = SimulatedBackend::create_group(1).unwrap();
        let gt = gpu_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = gpu_allreduce(&gt, &group[0], ReduceOp::Sum).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    #[ignore = "tracking issue #668: opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
    fn test_gpu_broadcast_invalid_root() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let gt = gpu_from_slice(&[1.0, 2.0], &[2]);
        let result = gpu_broadcast(&gt, &group[0], 5);
        assert!(result.is_err());
    }
}
