//! GPU-native collective operations via NCCL.
//!
//! Unlike [`crate::gpu_collective`] which transfers to CPU for each
//! collective, these functions operate directly on GPU device memory via
//! NCCL. This eliminates PCIe round-trips and can overlap communication
//! with computation via CUDA streams.
//!
//! # Feature gate
//!
//! Requires the `nccl` feature (which implies `gpu`).

use std::ffi::c_void;

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::gpu_dispatch::{self, GpuBufferHandle};

use crate::collective::ReduceOp;
use crate::nccl_backend::{NcclBackend, reduce_op_to_nccl};
use crate::nccl_sys::NcclDataType;

// ---------------------------------------------------------------------------
// Helper: infer dtype and get raw pointer from GpuBufferHandle
// ---------------------------------------------------------------------------

/// Determine the NCCL data type from the element size stored in the handle.
///
/// The GpuBufferHandle is type-erased; we use the GPU backend's
/// `raw_device_ptr` to verify it's valid and inspect the elem size
/// based on which downcast succeeds.
fn infer_dtype(handle: &GpuBufferHandle) -> FerrotorchResult<NcclDataType> {
    let backend = gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    match backend.buffer_elem_size(handle) {
        4 => Ok(NcclDataType::Float32),
        8 => Ok(NcclDataType::Float64),
        0 => Err(FerrotorchError::InvalidArgument {
            message: "NCCL collective: unrecognized buffer type".into(),
        }),
        other => Err(FerrotorchError::InvalidArgument {
            message: format!("NCCL collective: unsupported element size {other}"),
        }),
    }
}

fn get_ptr(handle: &GpuBufferHandle) -> FerrotorchResult<*const c_void> {
    let backend = gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let ptr = backend.raw_device_ptr(handle);
    if ptr.is_null() {
        return Err(FerrotorchError::InvalidArgument {
            message: "NCCL collective: buffer has no valid device pointer".into(),
        });
    }
    Ok(ptr)
}

fn get_ptr_mut(handle: &mut GpuBufferHandle) -> FerrotorchResult<*mut c_void> {
    let backend = gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let ptr = backend.raw_device_ptr_mut(handle);
    if ptr.is_null() {
        return Err(FerrotorchError::InvalidArgument {
            message: "NCCL collective: buffer has no valid device pointer".into(),
        });
    }
    Ok(ptr)
}

// ---------------------------------------------------------------------------
// High-level GPU collective operations
// ---------------------------------------------------------------------------

/// In-place all-reduce on a GPU buffer via NCCL.
///
/// Automatically detects f32 vs f64 from the buffer type.
/// After this call, `buffer` on every rank contains the element-wise
/// reduction of the original values across all ranks.
pub fn nccl_allreduce(
    buffer: &mut GpuBufferHandle,
    backend: &NcclBackend,
    op: &ReduceOp,
) -> FerrotorchResult<()> {
    let dtype = infer_dtype(buffer)?;
    nccl_allreduce_dtype(buffer, backend, op, dtype)
}

/// In-place all-reduce with explicit NCCL data type.
///
/// Use `NcclDataType::Float64` for f64 tensors.
pub fn nccl_allreduce_dtype(
    buffer: &mut GpuBufferHandle,
    backend: &NcclBackend,
    op: &ReduceOp,
    dtype: NcclDataType,
) -> FerrotorchResult<()> {
    let count = buffer.len();
    let nccl_op = reduce_op_to_nccl(op);
    let ptr = get_ptr_mut(buffer)?;

    // SAFETY: invoking `NcclBackend::allreduce_raw`, a `pub unsafe fn`.
    //
    // The unsafe contract (see `# Safety` on `allreduce_raw`):
    //   1. `sendbuf` / `recvbuf` are valid CUDA device pointers.
    //   2. They reference at least `count` elements of `dtype`.
    //   3. (Implicit, NCCL contract): same `count`, `dtype`, and `op` on
    //      every rank.
    //
    // Discharge:
    //   - `ptr` was produced by `gpu_dispatch::gpu_backend()
    //     .raw_device_ptr_mut(buffer)` and verified non-null in
    //     `get_ptr_mut`. The GPU backend's `raw_device_ptr_mut` returns
    //     a real device-resident pointer (`cudaMalloc`-derived for
    //     cudarc, equivalent for other backends).
    //   - `count = buffer.len()` is the element count the GPU backend
    //     allocated; the buffer is a `GpuBufferHandle` whose layout was
    //     established at construction. Each element is `dtype`-sized;
    //     `infer_dtype`/the explicit dtype must match the buffer's
    //     element type (callers using the `_dtype` overload are
    //     responsible for this match).
    //   - In-place semantics (`sendbuf == recvbuf`) is documented as
    //     allowed by NCCL for `ncclAllReduce`.
    //   - Cross-rank consistency is the caller's responsibility, same
    //     as PyTorch's `dist.all_reduce` — there is no way to verify
    //     this in-process.
    //   - LIFETIME: `buffer: &mut GpuBufferHandle` is borrowed exclusive
    //     for the duration of this call; `ptr` does not outlive
    //     `buffer`. The async NCCL call enqueues on `backend.stream`,
    //     so `buffer` must remain alive until the stream synchronizes;
    //     callers are documented to call `backend.synchronize()` before
    //     reading.
    unsafe { backend.allreduce_raw(ptr as *const c_void, ptr, count, dtype, nccl_op) }
}

/// Broadcast a GPU buffer from `root` to all ranks via NCCL.
///
/// Automatically detects f32 vs f64 from the buffer type.
pub fn nccl_broadcast(
    buffer: &mut GpuBufferHandle,
    backend: &NcclBackend,
    root: usize,
) -> FerrotorchResult<()> {
    let dtype = infer_dtype(buffer)?;
    nccl_broadcast_dtype(buffer, backend, root, dtype)
}

/// Broadcast with explicit NCCL data type.
pub fn nccl_broadcast_dtype(
    buffer: &mut GpuBufferHandle,
    backend: &NcclBackend,
    root: usize,
    dtype: NcclDataType,
) -> FerrotorchResult<()> {
    let count = buffer.len();
    let ptr = get_ptr_mut(buffer)?;

    // SAFETY: invoking `NcclBackend::broadcast_raw`, a `pub unsafe fn`.
    //
    // - DEVICE POINTER: `ptr` was produced by `get_ptr_mut`, which calls
    //   `raw_device_ptr_mut` on the active GPU backend and verifies the
    //   result non-null. It points to `count = buffer.len()` elements
    //   of `dtype`-sized storage on the device.
    // - IN-PLACE: `ncclBroadcast` accepts `sendbuf == recvbuf`; the
    //   root rank reads from `ptr`, non-root ranks overwrite the same
    //   buffer with the broadcast value. Documented behaviour in NCCL.
    // - ROOT BOUNDS: `root as i32` truncates if `root > i32::MAX`,
    //   which would be a logic bug in the caller (NCCL ranks fit in
    //   i32). PyTorch makes the same `int` cast.
    // - CROSS-RANK CONSISTENCY: callers must invoke this on every
    //   rank with the same `root` and `dtype`; this is the standard
    //   NCCL contract and cannot be verified in-process.
    // - LIFETIME: `buffer` is exclusively borrowed for the duration of
    //   this call; the NCCL operation is enqueued on `backend.stream`
    //   and the caller is documented to `synchronize()` before reading.
    unsafe { backend.broadcast_raw(ptr as *const c_void, ptr, count, dtype, root as i32) }
}

/// All-gather GPU buffers via NCCL.
///
/// Automatically detects f32 vs f64 from the send buffer type.
/// Each rank contributes `send_buf` and receives all ranks' data
/// concatenated into `recv_buf`.
pub fn nccl_all_gather(
    send_buf: &GpuBufferHandle,
    recv_buf: &mut GpuBufferHandle,
    backend: &NcclBackend,
) -> FerrotorchResult<()> {
    let dtype = infer_dtype(send_buf)?;
    nccl_all_gather_dtype(send_buf, recv_buf, backend, dtype)
}

/// All-gather with explicit NCCL data type.
pub fn nccl_all_gather_dtype(
    send_buf: &GpuBufferHandle,
    recv_buf: &mut GpuBufferHandle,
    backend: &NcclBackend,
    dtype: NcclDataType,
) -> FerrotorchResult<()> {
    let sendcount = send_buf.len();
    let send_ptr = get_ptr(send_buf)?;
    let recv_ptr = get_ptr_mut(recv_buf)?;

    // SAFETY: invoking `NcclBackend::all_gather_raw`, a `pub unsafe fn`.
    //
    // - DEVICE POINTERS: `send_ptr` and `recv_ptr` came from `get_ptr` /
    //   `get_ptr_mut` (verified non-null). They point to distinct
    //   buffers (`send_buf` is `&GpuBufferHandle`, `recv_buf` is
    //   `&mut GpuBufferHandle`; Rust's borrow rules ensure they are
    //   non-overlapping at the type level).
    // - SIZE CONTRACT: `recv_buf` must have capacity for
    //   `sendcount * world_size` elements of `dtype`. NCCL writes
    //   contiguously into `recv_ptr`; if undersized, NCCL writes past
    //   the buffer and the result is undefined behaviour. The caller
    //   is documented to allocate `recv_buf` accordingly (matching
    //   PyTorch's `dist.all_gather_into_tensor`).
    // - DTYPE: `dtype` must match the element layout of both buffers.
    //   The auto-detect overload `nccl_all_gather` infers from
    //   `send_buf`; the `_dtype` overload trusts the caller.
    // - LIFETIME: both borrows are live for the duration of the call;
    //   the NCCL kernel is async on `backend.stream`. The caller is
    //   documented to `synchronize()` before reading `recv_buf`.
    unsafe { backend.all_gather_raw(send_ptr, recv_ptr, sendcount, dtype) }
}

/// Reduce-scatter GPU buffers via NCCL.
///
/// Automatically detects f32 vs f64 from the recv buffer type.
/// Reduces across all ranks then distributes `recvcount` elements to each.
pub fn nccl_reduce_scatter(
    send_buf: &GpuBufferHandle,
    recv_buf: &mut GpuBufferHandle,
    backend: &NcclBackend,
    op: &ReduceOp,
) -> FerrotorchResult<()> {
    let dtype = infer_dtype(recv_buf)?;
    nccl_reduce_scatter_dtype(send_buf, recv_buf, backend, op, dtype)
}

/// Reduce-scatter with explicit NCCL data type.
pub fn nccl_reduce_scatter_dtype(
    send_buf: &GpuBufferHandle,
    recv_buf: &mut GpuBufferHandle,
    backend: &NcclBackend,
    op: &ReduceOp,
    dtype: NcclDataType,
) -> FerrotorchResult<()> {
    let recvcount = recv_buf.len();
    let nccl_op = reduce_op_to_nccl(op);
    let send_ptr = get_ptr(send_buf)?;
    let recv_ptr = get_ptr_mut(recv_buf)?;

    // SAFETY: invoking `NcclBackend::reduce_scatter_raw`, a `pub unsafe fn`.
    //
    // - DEVICE POINTERS: both pointers came from `get_ptr` / `get_ptr_mut`
    //   (verified non-null). The borrow checker prevents `send_buf` and
    //   `recv_buf` from aliasing because `recv_buf` is `&mut`.
    // - SIZE CONTRACT: `send_buf` must have capacity for
    //   `recvcount * world_size` elements (the per-rank chunk count
    //   times rank count). NCCL reads contiguously and writes only
    //   `recvcount` per rank. The caller is documented to size
    //   `send_buf`/`recv_buf` accordingly (matching PyTorch's
    //   `dist.reduce_scatter_tensor`).
    // - DTYPE/OP: `dtype` and `nccl_op` are converted from the safe
    //   `ReduceOp` enum via `reduce_op_to_nccl`, which only emits
    //   NCCL-supported reductions. `dtype` must match the buffer
    //   element type.
    // - LIFETIME: both borrows are live for the duration of the call;
    //   NCCL kernel runs async on `backend.stream`. The caller is
    //   documented to `synchronize()` before reading `recv_buf`.
    unsafe { backend.reduce_scatter_raw(send_ptr, recv_ptr, recvcount, dtype, nccl_op) }
}
