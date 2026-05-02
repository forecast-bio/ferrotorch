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

    // In-place: sendbuf == recvbuf.
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

    unsafe { backend.reduce_scatter_raw(send_ptr, recv_ptr, recvcount, dtype, nccl_op) }
}
