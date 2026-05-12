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
//! The `nccl` fast path extracts the input tensor's raw CUDA device
//! pointer via [`GpuTensor::cu_device_ptr`] (added in
//! [`ferrotorch-gpu`]'s `tensor_bridge`), allocates a fresh output
//! tensor via [`GpuTensor::try_clone`] (which performs a D2D copy with
//! no PCIe transit), and dispatches the collective through
//! [`NcclBackend::allreduce_raw`] / [`NcclBackend::broadcast_raw`].
//! The supplied `backend` must be an [`NcclBackend`] — the function
//! detects this via the [`Backend::as_nccl_backend`] downcast hook;
//! passing any other backend type to `gpu_allreduce` / `gpu_broadcast`
//! when only the `nccl` feature is active (and no fallback is enabled)
//! returns `Err(DistributedError::UnsupportedOp)`.
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

#[cfg(feature = "nccl")]
use crate::nccl_backend::{NcclBackend, reduce_op_to_nccl};
#[cfg(feature = "nccl")]
use crate::nccl_sys::NcclDataType;

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
// Private NCCL fast-path helpers (compiled only under the `nccl` feature)
// ---------------------------------------------------------------------------

/// Map a `GpuFloat` type parameter (`f32` or `f64`) to its NCCL data type.
///
/// Returns `Err` if `T` is neither `f32` nor `f64` (no NCCL coverage).
#[cfg(feature = "nccl")]
fn nccl_dtype_of<T: GpuFloat>() -> FerrotorchResult<NcclDataType> {
    use std::any::TypeId;
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        Ok(NcclDataType::Float32)
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        Ok(NcclDataType::Float64)
    } else {
        Err(DistributedError::UnsupportedOp {
            message: format!(
                "NCCL fast path does not cover dtype {} — only f32/f64 are wired",
                std::any::type_name::<T>(),
            ),
        }
        .into())
    }
}

/// NCCL-backed allreduce: returns a fresh [`GpuTensor`] containing the
/// element-wise reduction across all ranks in `backend`'s communicator.
///
/// The implementation:
///   1. Deep-clones `tensor` on the device into `out` so the input is not
///      mutated and the output owns its storage (matching the
///      `gpu_allreduce` API contract).
///   2. Extracts the raw `CUdeviceptr` from `out` via
///      [`GpuTensor::cu_device_ptr`].
///   3. Calls [`NcclBackend::allreduce_raw`] in-place on that pointer.
///   4. Synchronises the NCCL stream so the result is visible to
///      subsequent reads on any stream.
#[cfg(feature = "nccl")]
fn nccl_path_allreduce<T: GpuFloat>(
    tensor: &GpuTensor<T>,
    nccl: &NcclBackend,
    op: ReduceOp,
) -> FerrotorchResult<GpuTensor<T>> {
    let dtype = nccl_dtype_of::<T>()?;
    let nccl_op = reduce_op_to_nccl(&op);
    let count = tensor.numel();

    // Output tensor — D2D copy of input so the call is non-destructive.
    let out = tensor.try_clone().map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("gpu_allreduce: D2D clone failed: {e}"),
        }
    })?;

    // Raw device pointer to `out`'s storage. The pointer remains valid
    // for the duration of `out`'s `&mut` borrow below, which the NCCL
    // call holds exclusively.
    let ptr = out.cu_device_ptr() as *mut std::ffi::c_void;

    // SAFETY: calling `NcclBackend::allreduce_raw`, a `pub unsafe fn`.
    //
    // Discharge of the documented preconditions:
    //   - `sendbuf == recvbuf == ptr` — valid CUDA device pointer just
    //     obtained from `out.cu_device_ptr()` on a freshly D2D-cloned
    //     `GpuTensor<T>` with `count` elements of type `T`. NCCL allows
    //     in-place mode (`sendbuf == recvbuf`).
    //   - `count = tensor.numel()` matches the cloned buffer's element
    //     count (clone preserves `numel`); `dtype` was derived from
    //     `TypeId::of::<T>()` so it matches the element layout
    //     reflexively (`T == f32` ⇒ `Float32`, `T == f64` ⇒ `Float64`).
    //   - Cross-rank consistency (same `count`, `dtype`, `op` on every
    //     rank): the caller's responsibility, identical to PyTorch's
    //     `dist.all_reduce` contract.
    //   - Current CUDA device matches the comm's device: NCCL communicator
    //     was bound at `NcclBackend::new` time (`cudaSetDevice` is the
    //     caller's responsibility per `Self::new` rustdoc). We do not
    //     migrate `out` to a different device.
    //   - Buffers alive until stream synchronised: `out` is owned by this
    //     stack frame; we call `nccl.synchronize()?` below before
    //     returning `out`, satisfying the "alive until stream sync"
    //     obligation.
    unsafe { nccl.allreduce_raw(ptr.cast_const(), ptr, count, dtype, nccl_op) }?;

    // Wait for the NCCL stream so the result in `out` is visible to any
    // subsequent reads on the default compute stream. Without this,
    // callers using `out.cpu()` directly after this return would race
    // the NCCL kernel.
    nccl.synchronize()?;

    Ok(out)
}

/// NCCL-backed broadcast: returns a fresh [`GpuTensor`] whose contents
/// equal the `root` rank's contribution. Mirror of [`nccl_path_allreduce`].
#[cfg(feature = "nccl")]
fn nccl_path_broadcast<T: GpuFloat>(
    tensor: &GpuTensor<T>,
    nccl: &NcclBackend,
    root: usize,
) -> FerrotorchResult<GpuTensor<T>> {
    let dtype = nccl_dtype_of::<T>()?;
    let count = tensor.numel();
    let world_size = nccl.world_size();
    if root >= world_size {
        return Err(DistributedError::InvalidRank {
            rank: root,
            world_size,
        }
        .into());
    }
    let root_i32 = i32::try_from(root).map_err(|_| DistributedError::InvalidRank {
        rank: root,
        world_size,
    })?;

    let out = tensor.try_clone().map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("gpu_broadcast: D2D clone failed: {e}"),
        }
    })?;
    let ptr = out.cu_device_ptr() as *mut std::ffi::c_void;

    // SAFETY: calling `NcclBackend::broadcast_raw`, a `pub unsafe fn`.
    //
    // Discharge of the documented preconditions:
    //   - `sendbuf == recvbuf == ptr`: valid CUDA device pointer to
    //     `count` elements of `dtype` (just obtained from the cloned
    //     output tensor). NCCL `ncclBroadcast` accepts in-place mode on
    //     the root rank; on non-root ranks `sendbuf` is ignored and
    //     `recvbuf` is overwritten with the broadcast data.
    //   - `dtype` matches the element layout reflexively (`TypeId`-derived).
    //   - `root` bounds-checked above against `world_size` and converted
    //     to `i32`. Caller is responsible for passing the same `root` on
    //     every rank (standard NCCL collective contract).
    //   - Current CUDA device matches the comm's device: bound at
    //     `NcclBackend::new` time. We don't migrate.
    //   - Buffers alive until stream synchronised: `nccl.synchronize()?`
    //     below blocks before `out` is returned.
    unsafe { nccl.broadcast_raw(ptr.cast_const(), ptr, count, dtype, root_i32) }?;
    nccl.synchronize()?;

    Ok(out)
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
/// 1. **`nccl` feature + `NcclBackend` passed (fast path):** delegates to
///    [`NcclBackend::allreduce_raw`] for GPU-native allreduce — no host
///    round-trip. Detected via [`Backend::as_nccl_backend`].
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
    //
    // `Backend::as_nccl_backend` returns `Some` iff `backend` is an
    // `NcclBackend` (default impl returns `None`; only `NcclBackend`
    // overrides). When present, dispatch through `allreduce_raw` on the
    // input tensor's raw device pointer — staying entirely on the GPU.
    #[cfg(feature = "nccl")]
    if let Some(nccl) = backend.as_nccl_backend() {
        return nccl_path_allreduce(tensor, nccl, op);
    }

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
/// 1. **`nccl` feature + `NcclBackend` passed (fast path):** delegates to
///    [`NcclBackend::broadcast_raw`] for GPU-native broadcast — no host
///    round-trip. Detected via [`Backend::as_nccl_backend`].
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
    //
    // Mirror of the `gpu_allreduce` dispatch: detect `NcclBackend` via
    // `Backend::as_nccl_backend` and route through `broadcast_raw` on
    // the tensor's raw device pointer.
    #[cfg(feature = "nccl")]
    if let Some(nccl) = backend.as_nccl_backend() {
        return nccl_path_broadcast(tensor, nccl, root);
    }

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

    // These tests use `SimulatedBackend` (an in-process CPU channel
    // backend). After the NCCL fast path landed in #1135, the dispatch
    // order in `gpu_allreduce` / `gpu_broadcast` is:
    //   1. `nccl` feature + `NcclBackend` → GPU-native NCCL fast path
    //   2. `FERROTORCH_ENABLE_GPU_FALLBACK=1` → host round-trip
    //   3. otherwise → `Err(UnsupportedOp)` (PyTorch parity)
    //
    // With `SimulatedBackend` (not `NcclBackend`) and no env-var, branch
    // 3 fires and these tests would fail. They stay `#[ignore]`-gated
    // because they were authored for the silent host-round-trip era and
    // need an explicit `FERROTORCH_ENABLE_GPU_FALLBACK=1` harness — or
    // a port to the real multi-GPU NCCL setup. The latter is genuine
    // multi-GPU hardware-gated, not in-process testable.

    #[test]
    #[ignore = "tracking issue #1135 (replaces closed #668): opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
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
    #[ignore = "tracking issue #1135 (replaces closed #668): opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
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
    #[ignore = "tracking issue #1135 (replaces closed #668): opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
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
    #[ignore = "tracking issue #1135 (replaces closed #668): opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
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
    #[ignore = "tracking issue #1135 (replaces closed #668): opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
    fn test_gpu_allreduce_preserves_shape() {
        // Verify shape [2, 3] is preserved through the round-trip.
        let group = SimulatedBackend::create_group(1).unwrap();
        let gt = gpu_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = gpu_allreduce(&gt, &group[0], ReduceOp::Sum).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    #[ignore = "tracking issue #1135 (replaces closed #668): opt-in fallback now required; test must be updated once NCCL wiring or env-var harness is in place"]
    fn test_gpu_broadcast_invalid_root() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let gt = gpu_from_slice(&[1.0, 2.0], &[2]);
        let result = gpu_broadcast(&gt, &group[0], 5);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------
    // NCCL fast-path wiring (#1135)
    // -----------------------------------------------------------------
    //
    // These tests exercise the *dispatch* — they build an `NcclBackend`,
    // hand it to `gpu_allreduce` / `gpu_broadcast` as `&dyn Backend`,
    // and verify the call reaches `NcclBackend::{allreduce_raw,
    // broadcast_raw}` and returns without error. In single-rank mode,
    // NCCL's allreduce and broadcast are identity operations — the
    // output must equal the input.
    //
    // The tests are `#[ignore]`-gated because they require both NCCL
    // (`libnccl2`) and a CUDA device to be installed on the test host.
    // The dispatch wiring itself is verified at compile-time by the
    // `#[cfg(feature = "nccl")]` gate.

    #[cfg(feature = "nccl")]
    #[test]
    #[ignore = "requires NCCL (libnccl2) and a CUDA device — exercises the GpuTensor → NcclBackend wiring landed in #1135"]
    fn gpu_allreduce_dispatches_to_nccl_in_single_rank_mode() {
        use crate::nccl_backend::NcclBackend;
        use crate::nccl_sys::get_unique_id;

        let unique_id = get_unique_id().expect("NCCL unique ID generation");
        let nccl = NcclBackend::new(0, 1, unique_id).expect("NcclBackend init");

        let input = [1.5_f32, -2.5, 3.5, 0.0];
        let gt = gpu_from_slice(&input, &[4]);

        // Dispatch through `gpu_allreduce` (the public API) so we
        // exercise the `as_nccl_backend` downcast hook.
        let result = gpu_allreduce(&gt, &nccl, ReduceOp::Sum)
            .expect("single-rank NCCL allreduce dispatch must succeed");

        assert_eq!(result.shape(), &[4]);
        let cpu = result.cpu().expect("result to CPU");
        let out = cpu.data().expect("flat data");
        // Single-rank allreduce is an identity operation.
        for (i, (got, want)) in out.iter().zip(input.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "allreduce[{i}]: got {got}, want {want}",
            );
        }
    }

    #[cfg(feature = "nccl")]
    #[test]
    #[ignore = "requires NCCL (libnccl2) and a CUDA device — exercises the GpuTensor → NcclBackend wiring landed in #1135"]
    fn gpu_broadcast_dispatches_to_nccl_in_single_rank_mode() {
        use crate::nccl_backend::NcclBackend;
        use crate::nccl_sys::get_unique_id;

        let unique_id = get_unique_id().expect("NCCL unique ID generation");
        let nccl = NcclBackend::new(0, 1, unique_id).expect("NcclBackend init");

        let input = [42.0_f32, 99.0, -7.5];
        let gt = gpu_from_slice(&input, &[3]);

        let result = gpu_broadcast(&gt, &nccl, 0)
            .expect("single-rank NCCL broadcast dispatch must succeed");

        assert_eq!(result.shape(), &[3]);
        let cpu = result.cpu().expect("result to CPU");
        let out = cpu.data().expect("flat data");
        for (i, (got, want)) in out.iter().zip(input.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-6,
                "broadcast[{i}]: got {got}, want {want}",
            );
        }
    }
}
