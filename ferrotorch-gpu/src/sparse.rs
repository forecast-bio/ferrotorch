//! cuSPARSE-backed sparse-dense matrix multiply (SpMM).
//!
//! This module implements the `spmm_csr_f32`/`spmm_csr_f64` GPU paths used
//! by `ferrotorch_core::sparse::SparseTensor::spmm` when the dense operand
//! is a CUDA tensor. PyTorch's `torch.sparse.mm` runs on cuSPARSE in this
//! configuration; ferrotorch mirrors that per `rust-gpu-discipline §3`.
//!
//! # Storage
//!
//! ferrotorch's `SparseTensor` stores indices/values on the host (COO).
//! We coalesce on the host (sort by `(row, col)`, sum duplicates), build
//! a CSR triple `(crow_indices, col_indices, values)` on the host, upload
//! all three to device buffers, then dispatch `cusparseSpMM` with
//! `CUSPARSE_SPMM_ALG_DEFAULT`. The dense operand is already on device.
//!
//! # Handle lifetime
//!
//! cuSPARSE handles are expensive to create. [`CudaBackendImpl`] caches
//! one handle per CUDA device via `OnceLock`, created on first use. The
//! handle is bound to the device's default stream via `cusparseSetStream`
//! before each call so that subsequent `cusparseSpMM` enqueues on the
//! same stream as cuBLAS / kernel launches.

#![cfg(feature = "cuda")]

use cudarc::cusparse::sys as csys;
use cudarc::driver::{DevicePtr, DevicePtrMut};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64, cpu_to_gpu};

// ---------------------------------------------------------------------------
// Handle wrapper
// ---------------------------------------------------------------------------

/// Owning wrapper around a `cusparseHandle_t` that destroys the handle on
/// drop. cuSPARSE handles are not thread-safe in general, but the only
/// caller is [`CudaBackendImpl`] which holds them inside `Mutex`-style
/// `OnceLock` slots and binds the stream per call.
#[derive(Debug)]
pub struct CusparseHandle {
    inner: csys::cusparseHandle_t,
}

// SAFETY:
// `cusparseHandle_t` is a raw pointer to an opaque cuSPARSE context.
// Sending the handle between threads is safe as long as the cuSPARSE
// API is called from one thread at a time, which `CudaBackendImpl`
// guarantees by binding the stream and serialising calls per device.
unsafe impl Send for CusparseHandle {}
// SAFETY:
// Same reasoning as `Send` — the handle is opaque and cuSPARSE calls
// take it via the API which is documented thread-safe when distinct
// streams are used. We don't use it concurrently across threads in
// ferrotorch, so this is at most a sound over-approximation.
unsafe impl Sync for CusparseHandle {}

impl CusparseHandle {
    /// Create a fresh cuSPARSE handle on the current CUDA context.
    pub fn new() -> GpuResult<Self> {
        let inner = cudarc::cusparse::result::create()
            .map_err(|e| GpuError::InvalidState {
                message: format!("cusparseCreate failed: {e:?}"),
            })?;
        Ok(Self { inner })
    }

    /// Raw handle, for FFI calls.
    #[inline]
    pub fn raw(&self) -> csys::cusparseHandle_t {
        self.inner
    }
}

impl Drop for CusparseHandle {
    fn drop(&mut self) {
        // SAFETY: `self.inner` was created via `cusparseCreate` in `new`
        // and has not been destroyed yet. `Drop` runs at most once.
        unsafe {
            let _ = cudarc::cusparse::result::destroy(self.inner);
        }
    }
}

// ---------------------------------------------------------------------------
// Status helper
// ---------------------------------------------------------------------------

fn check(status: csys::cusparseStatus_t, op: &'static str) -> GpuResult<()> {
    if status == csys::cusparseStatus_t::CUSPARSE_STATUS_SUCCESS {
        Ok(())
    } else {
        Err(GpuError::InvalidState {
            message: format!("{op} returned cuSPARSE status {status:?}"),
        })
    }
}

/// Bind the cuSPARSE handle to the given device's stream so subsequent
/// SpMM calls enqueue on the same stream as the rest of the GPU work.
fn set_stream(handle: &CusparseHandle, device: &GpuDevice) -> GpuResult<()> {
    // SAFETY:
    // - `handle` is a valid, non-destroyed `cusparseHandle_t` (CusparseHandle
    //   only exposes a raw handle through `raw()` and destroys on drop).
    // - The stream pointer comes from `device.stream().cu_stream()` which
    //   is a valid `CUstream` for the lifetime of the `Arc<CudaStream>`.
    //   `cudaStream_t` and `CUstream` are both `*mut CUstream_st` (the
    //   driver/runtime ABI is unified at this typedef level), so the
    //   `as *mut _` cast between cudarc's `driver::sys::CUstream_st` and
    //   `cusparse::sys::CUstream_st` opaque structs is a safe re-typing
    //   of the same pointer.
    let stream = device.stream();
    let cu_stream_ptr = stream.cu_stream() as *mut csys::CUstream_st;
    let status = unsafe {
        csys::cusparseSetStream(handle.raw(), cu_stream_ptr as csys::cudaStream_t)
    };
    check(status, "cusparseSetStream")
}

// ---------------------------------------------------------------------------
// SpMM (f32)
// ---------------------------------------------------------------------------

/// Compute `C = A @ B` on the GPU via `cusparseSpMM` for a CSR sparse `A`
/// (`m × k`) and a dense `B` (`k × n`) row-major. Returns a row-major
/// dense `C` (`m × n`) on the same device.
///
/// `crow_indices` (length `m + 1`), `col_indices` (length `nnz`), and
/// `values` (length `nnz`) describe the CSR structure on the host. They
/// are uploaded just-in-time. `dense` is already on device.
#[allow(clippy::too_many_arguments)]
pub fn gpu_spmm_csr_f32(
    handle: &CusparseHandle,
    crow_indices: &[u32],
    col_indices: &[u32],
    values: &[f32],
    dense: &CudaBuffer<f32>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if crow_indices.len() != m + 1 {
        return Err(GpuError::ShapeMismatch {
            op: "spmm_csr_f32",
            expected: vec![m + 1],
            got: vec![crow_indices.len()],
        });
    }
    if col_indices.len() != values.len() {
        return Err(GpuError::ShapeMismatch {
            op: "spmm_csr_f32",
            expected: vec![values.len()],
            got: vec![col_indices.len()],
        });
    }
    if dense.len() != k * n {
        return Err(GpuError::ShapeMismatch {
            op: "spmm_csr_f32",
            expected: vec![k, n],
            got: vec![dense.len()],
        });
    }

    if m == 0 || n == 0 {
        return alloc_zeros_f32(m * n, device);
    }
    // For a zero-row sparse matrix or all-zero K dimension, the result
    // is all zeros. Skip cuSPARSE — it accepts nnz=0 in modern toolkits
    // but allocating + zeroing is cheaper and guaranteed correct.
    let nnz = values.len();
    if nnz == 0 || k == 0 {
        return alloc_zeros_f32(m * n, device);
    }

    set_stream(handle, device)?;

    // Upload CSR to device.
    let mut d_crow = cpu_to_gpu(crow_indices, device)?;
    let mut d_col = cpu_to_gpu(col_indices, device)?;
    let mut d_vals = cpu_to_gpu(values, device)?;

    // Output buffer.
    let mut out = alloc_zeros_f32(m * n, device)?;

    let stream = device.stream();

    let mut sp_mat: csys::cusparseSpMatDescr_t = std::ptr::null_mut();
    let mut dn_b: csys::cusparseDnMatDescr_t = std::ptr::null_mut();
    let mut dn_c: csys::cusparseDnMatDescr_t = std::ptr::null_mut();

    // Convert dimensions to i64 for cuSPARSE descriptors.
    let m_i64 = i64::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "spmm_csr_f32",
        expected: vec![i64::MAX as usize],
        got: vec![m],
    })?;
    let k_i64 = i64::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "spmm_csr_f32",
        expected: vec![i64::MAX as usize],
        got: vec![k],
    })?;
    let n_i64 = i64::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "spmm_csr_f32",
        expected: vec![i64::MAX as usize],
        got: vec![n],
    })?;
    let nnz_i64 = i64::try_from(nnz).map_err(|_| GpuError::ShapeMismatch {
        op: "spmm_csr_f32",
        expected: vec![i64::MAX as usize],
        got: vec![nnz],
    })?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;

    // Build the cuSPARSE descriptors. Use unsafe to drive the raw FFI
    // and make sure each descriptor is destroyed below.
    //
    // SAFETY:
    // - `cusparseCreateCsr` requires three device pointers of the
    //   appropriate types (i32 row offsets, i32 col indices, f32 values)
    //   with exactly `nnz` (or `m+1`) elements. Buffers were uploaded
    //   from `crow_indices`/`col_indices`/`values` which we sized above.
    // - `cusparseCreateDnMat` requires a row-major dense buffer of size
    //   `rows * ld` elements; `dense` is `k*n` and `ld = n` matches.
    //   `out` is `m*n` and we pass `ld = n`. Both buffers stay live for
    //   the lifetime of the descriptors.
    // - `i32` index type matches the `u32` host data: cuSPARSE stores
    //   indices in 32-bit ints; we use unsigned u32 host-side which has
    //   identical layout. cuSPARSE's `CUSPARSE_INDEX_32I` reads them as
    //   signed, but for a non-negative index domain (which CSR is) the
    //   bit pattern is identical.
    let result = (|| -> GpuResult<()> {
        // Acquire device pointers inside the closure so the cudarc
        // `SyncOnDrop` guards drop here, freeing the borrow on `out`
        // before we return it from the outer function.
        let (crow_ptr, _crow_sync) = d_crow.inner_mut().device_ptr_mut(&stream);
        let (col_ptr, _col_sync) = d_col.inner_mut().device_ptr_mut(&stream);
        let (vals_ptr, _vals_sync) = d_vals.inner_mut().device_ptr_mut(&stream);
        let (dense_ptr, _dense_sync) = dense.inner().device_ptr(&stream);
        let (out_ptr, _out_sync) = out.inner_mut().device_ptr_mut(&stream);

        let status = unsafe {
            csys::cusparseCreateCsr(
                &mut sp_mat,
                m_i64,
                k_i64,
                nnz_i64,
                crow_ptr as *mut std::ffi::c_void,
                col_ptr as *mut std::ffi::c_void,
                vals_ptr as *mut std::ffi::c_void,
                csys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                csys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                csys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                csys::cudaDataType_t::CUDA_R_32F,
            )
        };
        check(status, "cusparseCreateCsr")?;

        let status = unsafe {
            csys::cusparseCreateDnMat(
                &mut dn_b,
                k_i64,
                n_i64,
                n_i64,
                dense_ptr as *mut std::ffi::c_void,
                csys::cudaDataType_t::CUDA_R_32F,
                csys::cusparseOrder_t::CUSPARSE_ORDER_ROW,
            )
        };
        check(status, "cusparseCreateDnMat (B)")?;

        let status = unsafe {
            csys::cusparseCreateDnMat(
                &mut dn_c,
                m_i64,
                n_i64,
                n_i64,
                out_ptr as *mut std::ffi::c_void,
                csys::cudaDataType_t::CUDA_R_32F,
                csys::cusparseOrder_t::CUSPARSE_ORDER_ROW,
            )
        };
        check(status, "cusparseCreateDnMat (C)")?;

        // Query workspace size.
        let mut buffer_size: usize = 0;
        let status = unsafe {
            csys::cusparseSpMM_bufferSize(
                handle.raw(),
                csys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                csys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                std::ptr::from_ref::<f32>(&alpha).cast::<std::ffi::c_void>(),
                sp_mat,
                dn_b,
                std::ptr::from_ref::<f32>(&beta).cast::<std::ffi::c_void>(),
                dn_c,
                csys::cudaDataType_t::CUDA_R_32F,
                csys::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                &mut buffer_size,
            )
        };
        check(status, "cusparseSpMM_bufferSize")?;

        // Allocate workspace as a u8 buffer.
        let workspace_bytes = buffer_size;
        let mut workspace_slice = if workspace_bytes > 0 {
            Some(stream.alloc_zeros::<u8>(workspace_bytes)?)
        } else {
            None
        };
        let workspace_ptr = match workspace_slice.as_mut() {
            Some(s) => {
                let (p, _sync) = s.device_ptr_mut(&stream);
                p as *mut std::ffi::c_void
            }
            None => std::ptr::null_mut(),
        };

        let status = unsafe {
            csys::cusparseSpMM(
                handle.raw(),
                csys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                csys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                std::ptr::from_ref::<f32>(&alpha).cast::<std::ffi::c_void>(),
                sp_mat,
                dn_b,
                std::ptr::from_ref::<f32>(&beta).cast::<std::ffi::c_void>(),
                dn_c,
                csys::cudaDataType_t::CUDA_R_32F,
                csys::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                workspace_ptr,
            )
        };
        check(status, "cusparseSpMM")?;

        drop(workspace_slice);
        Ok(())
    })();

    // Always destroy descriptors, in reverse creation order.
    //
    // SAFETY: Each pointer was either left null (creation failed) or set
    // by a successful Create*. cuSPARSE's Destroy* tolerates a null arg
    // by returning SUCCESS for safe destruction. We ignore destruction
    // errors so a primary failure isn't masked.
    unsafe {
        if !dn_c.is_null() {
            let _ = csys::cusparseDestroyDnMat(dn_c);
        }
        if !dn_b.is_null() {
            let _ = csys::cusparseDestroyDnMat(dn_b);
        }
        if !sp_mat.is_null() {
            let _ = csys::cusparseDestroySpMat(sp_mat);
        }
    }

    result?;
    Ok(out)
}

// ---------------------------------------------------------------------------
// SpMM (f64) — mirrors f32 with `CUDA_R_64F` and f64 alpha/beta.
// ---------------------------------------------------------------------------

/// Compute `C = A @ B` on the GPU via `cusparseSpMM` for a CSR sparse `A`
/// (`m × k`) and a dense `B` (`k × n`) row-major (f64). Returns a row-major
/// dense `C` (`m × n`) on the same device. Mirrors [`gpu_spmm_csr_f32`].
#[allow(clippy::too_many_arguments)]
pub fn gpu_spmm_csr_f64(
    handle: &CusparseHandle,
    crow_indices: &[u32],
    col_indices: &[u32],
    values: &[f64],
    dense: &CudaBuffer<f64>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if crow_indices.len() != m + 1 {
        return Err(GpuError::ShapeMismatch {
            op: "spmm_csr_f64",
            expected: vec![m + 1],
            got: vec![crow_indices.len()],
        });
    }
    if col_indices.len() != values.len() {
        return Err(GpuError::ShapeMismatch {
            op: "spmm_csr_f64",
            expected: vec![values.len()],
            got: vec![col_indices.len()],
        });
    }
    if dense.len() != k * n {
        return Err(GpuError::ShapeMismatch {
            op: "spmm_csr_f64",
            expected: vec![k, n],
            got: vec![dense.len()],
        });
    }

    if m == 0 || n == 0 {
        return alloc_zeros_f64(m * n, device);
    }
    let nnz = values.len();
    if nnz == 0 || k == 0 {
        return alloc_zeros_f64(m * n, device);
    }

    set_stream(handle, device)?;

    let mut d_crow = cpu_to_gpu(crow_indices, device)?;
    let mut d_col = cpu_to_gpu(col_indices, device)?;
    let mut d_vals = cpu_to_gpu(values, device)?;
    let mut out = alloc_zeros_f64(m * n, device)?;

    let stream = device.stream();

    let mut sp_mat: csys::cusparseSpMatDescr_t = std::ptr::null_mut();
    let mut dn_b: csys::cusparseDnMatDescr_t = std::ptr::null_mut();
    let mut dn_c: csys::cusparseDnMatDescr_t = std::ptr::null_mut();

    let m_i64 = i64::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "spmm_csr_f64",
        expected: vec![i64::MAX as usize],
        got: vec![m],
    })?;
    let k_i64 = i64::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "spmm_csr_f64",
        expected: vec![i64::MAX as usize],
        got: vec![k],
    })?;
    let n_i64 = i64::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "spmm_csr_f64",
        expected: vec![i64::MAX as usize],
        got: vec![n],
    })?;
    let nnz_i64 = i64::try_from(nnz).map_err(|_| GpuError::ShapeMismatch {
        op: "spmm_csr_f64",
        expected: vec![i64::MAX as usize],
        got: vec![nnz],
    })?;

    let alpha: f64 = 1.0;
    let beta: f64 = 0.0;

    // SAFETY: Same obligations as the f32 path; `CUDA_R_64F` + `f64`
    // alpha/beta substituted for the dtype change. All descriptors are
    // destroyed in the reverse-order block below.
    let result = (|| -> GpuResult<()> {
        // Acquire device pointers inside the closure so cudarc's
        // `SyncOnDrop` guards drop here, freeing the borrow on `out`
        // before the outer function returns it.
        let (crow_ptr, _crow_sync) = d_crow.inner_mut().device_ptr_mut(&stream);
        let (col_ptr, _col_sync) = d_col.inner_mut().device_ptr_mut(&stream);
        let (vals_ptr, _vals_sync) = d_vals.inner_mut().device_ptr_mut(&stream);
        let (dense_ptr, _dense_sync) = dense.inner().device_ptr(&stream);
        let (out_ptr, _out_sync) = out.inner_mut().device_ptr_mut(&stream);

        let status = unsafe {
            csys::cusparseCreateCsr(
                &mut sp_mat,
                m_i64,
                k_i64,
                nnz_i64,
                crow_ptr as *mut std::ffi::c_void,
                col_ptr as *mut std::ffi::c_void,
                vals_ptr as *mut std::ffi::c_void,
                csys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                csys::cusparseIndexType_t::CUSPARSE_INDEX_32I,
                csys::cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                csys::cudaDataType_t::CUDA_R_64F,
            )
        };
        check(status, "cusparseCreateCsr (f64)")?;

        let status = unsafe {
            csys::cusparseCreateDnMat(
                &mut dn_b,
                k_i64,
                n_i64,
                n_i64,
                dense_ptr as *mut std::ffi::c_void,
                csys::cudaDataType_t::CUDA_R_64F,
                csys::cusparseOrder_t::CUSPARSE_ORDER_ROW,
            )
        };
        check(status, "cusparseCreateDnMat B (f64)")?;

        let status = unsafe {
            csys::cusparseCreateDnMat(
                &mut dn_c,
                m_i64,
                n_i64,
                n_i64,
                out_ptr as *mut std::ffi::c_void,
                csys::cudaDataType_t::CUDA_R_64F,
                csys::cusparseOrder_t::CUSPARSE_ORDER_ROW,
            )
        };
        check(status, "cusparseCreateDnMat C (f64)")?;

        let mut buffer_size: usize = 0;
        let status = unsafe {
            csys::cusparseSpMM_bufferSize(
                handle.raw(),
                csys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                csys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                std::ptr::from_ref::<f64>(&alpha).cast::<std::ffi::c_void>(),
                sp_mat,
                dn_b,
                std::ptr::from_ref::<f64>(&beta).cast::<std::ffi::c_void>(),
                dn_c,
                csys::cudaDataType_t::CUDA_R_64F,
                csys::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                &mut buffer_size,
            )
        };
        check(status, "cusparseSpMM_bufferSize (f64)")?;

        let workspace_bytes = buffer_size;
        let mut workspace_slice = if workspace_bytes > 0 {
            Some(stream.alloc_zeros::<u8>(workspace_bytes)?)
        } else {
            None
        };
        let workspace_ptr = match workspace_slice.as_mut() {
            Some(s) => {
                let (p, _sync) = s.device_ptr_mut(&stream);
                p as *mut std::ffi::c_void
            }
            None => std::ptr::null_mut(),
        };

        let status = unsafe {
            csys::cusparseSpMM(
                handle.raw(),
                csys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                csys::cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE,
                std::ptr::from_ref::<f64>(&alpha).cast::<std::ffi::c_void>(),
                sp_mat,
                dn_b,
                std::ptr::from_ref::<f64>(&beta).cast::<std::ffi::c_void>(),
                dn_c,
                csys::cudaDataType_t::CUDA_R_64F,
                csys::cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                workspace_ptr,
            )
        };
        check(status, "cusparseSpMM (f64)")?;

        drop(workspace_slice);
        Ok(())
    })();

    // SAFETY: same as f32 destroy block.
    unsafe {
        if !dn_c.is_null() {
            let _ = csys::cusparseDestroyDnMat(dn_c);
        }
        if !dn_b.is_null() {
            let _ = csys::cusparseDestroyDnMat(dn_b);
        }
        if !sp_mat.is_null() {
            let _ = csys::cusparseDestroySpMat(sp_mat);
        }
    }

    result?;
    Ok(out)
}
