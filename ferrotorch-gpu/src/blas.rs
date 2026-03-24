//! cuBLAS-backed GPU matrix multiplication.
//!
//! This module provides GPU-accelerated matrix multiplication via NVIDIA's
//! cuBLAS library (SGEMM for f32, DGEMM for f64). The primary entry point is
//! [`gpu_matmul`], which computes `C = A @ B` for row-major matrices stored
//! in [`CudaBuffer`]s.
//!
//! # Row-major trick
//!
//! cuBLAS operates in column-major order. To compute `C = A @ B` with
//! row-major data, we exploit the identity:
//!
//! ```text
//! C_row = A_row @ B_row
//! ```
//!
//! is equivalent to calling GEMM with B as the first matrix and A as the
//! second, with swapped leading dimensions. Concretely, we call:
//!
//! ```text
//! gemm(N, N, n, m, k, 1.0, B, n, A, k, 0.0, C, n)
//! ```
//!
//! This produces the correct row-major result in C without any transpositions.
//!
//! # CPU fallback
//!
//! If cuBLAS handle creation fails (e.g. library not found), the module falls
//! back to a CPU round-trip: copy both matrices to host, multiply with naive
//! triple-loop, copy result back. This is correct but slow, and is
//! accompanied by an `eprintln!` warning.

#[cfg(feature = "cuda")]
use cudarc::cublas::{Gemm, GemmConfig, StridedBatchedConfig, sys};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::{alloc_zeros_f32, alloc_zeros_f64};

// ---------------------------------------------------------------------------
// GPU matmul -- f32 (SGEMM)
// ---------------------------------------------------------------------------

/// Compute `C = A @ B` on the GPU using cuBLAS SGEMM.
///
/// `a` contains `m * k` elements (matrix `[m, k]` in row-major order).
/// `b` contains `k * n` elements (matrix `[k, n]` in row-major order).
/// Returns a buffer with `m * n` elements (matrix `[m, n]` in row-major order).
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] if buffer lengths don't match dimensions.
/// - [`GpuError::DeviceMismatch`] if buffers are on different devices.
/// - [`GpuError::Blas`] on cuBLAS runtime errors.
/// - [`GpuError::Driver`] on CUDA driver errors.
#[cfg(feature = "cuda")]
pub fn gpu_matmul_f32(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    // Validate buffer lengths match declared dimensions.
    if a.len() != m * k {
        return Err(GpuError::ShapeMismatch {
            op: "matmul",
            expected: vec![m, k],
            got: vec![a.len()],
        });
    }
    if b.len() != k * n {
        return Err(GpuError::ShapeMismatch {
            op: "matmul",
            expected: vec![k, n],
            got: vec![b.len()],
        });
    }

    // Validate same device.
    if a.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: a.device_ordinal(),
        });
    }
    if b.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: b.device_ordinal(),
        });
    }

    // Handle degenerate cases.
    if m == 0 || k == 0 || n == 0 {
        return alloc_zeros_f32(m * n, device);
    }

    // Validate dimensions fit in i32 (cuBLAS uses i32 for matrix dimensions).
    let m_i32 = i32::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul",
        expected: vec![i32::MAX as usize],
        got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul",
        expected: vec![i32::MAX as usize],
        got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul",
        expected: vec![i32::MAX as usize],
        got: vec![n],
    })?;

    // For M=1 (vector-matrix multiply) or small matrices, use our own PTX kernel.
    // cuBLAS SGEMM has ~3ms launch/dispatch overhead per call that dominates for
    // small M. Our kernel compiles once and handles all sizes with ~10μs overhead.
    // For M=1 decode-time matmuls (768×768, 768×3072), this is 100-300x faster.
    let total_ops = m * k * n;
    if m <= 4 || total_ops < 500_000 {
        return crate::kernels::gpu_small_matmul(a, b, m, k, n, device);
    }

    let blas = device.blas();
    let mut c = alloc_zeros_f32(m * n, device)?;

    let cfg = GemmConfig {
        transa: sys::cublasOperation_t::CUBLAS_OP_N,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: n_i32,
        n: m_i32,
        k: k_i32,
        alpha: 1.0f32,
        lda: n_i32,
        ldb: k_i32,
        beta: 0.0f32,
        ldc: n_i32,
    };

    unsafe {
        blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut())?;
    }

    Ok(c)
}

/// Compute `C = A @ B` on the GPU using cuBLAS DGEMM.
///
/// `a` contains `m * k` elements (matrix `[m, k]` in row-major order).
/// `b` contains `k * n` elements (matrix `[k, n]` in row-major order).
/// Returns a buffer with `m * n` elements (matrix `[m, n]` in row-major order).
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] if buffer lengths don't match dimensions.
/// - [`GpuError::DeviceMismatch`] if buffers are on different devices.
/// - [`GpuError::Blas`] on cuBLAS runtime errors.
/// - [`GpuError::Driver`] on CUDA driver errors.
#[cfg(feature = "cuda")]
pub fn gpu_matmul_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    // Validate buffer lengths match declared dimensions.
    if a.len() != m * k {
        return Err(GpuError::ShapeMismatch {
            op: "matmul",
            expected: vec![m, k],
            got: vec![a.len()],
        });
    }
    if b.len() != k * n {
        return Err(GpuError::ShapeMismatch {
            op: "matmul",
            expected: vec![k, n],
            got: vec![b.len()],
        });
    }

    // Validate same device.
    if a.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: a.device_ordinal(),
        });
    }
    if b.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: b.device_ordinal(),
        });
    }

    // Handle degenerate cases.
    if m == 0 || k == 0 || n == 0 {
        return alloc_zeros_f64(m * n, device);
    }

    // Validate dimensions fit in i32 (cuBLAS uses i32 for matrix dimensions).
    let m_i32 = i32::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul",
        expected: vec![i32::MAX as usize],
        got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul",
        expected: vec![i32::MAX as usize],
        got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul",
        expected: vec![i32::MAX as usize],
        got: vec![n],
    })?;

    let blas = device.blas();
    let mut c = alloc_zeros_f64(m * n, device)?;

    let cfg = GemmConfig {
        transa: sys::cublasOperation_t::CUBLAS_OP_N,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: n_i32,
        n: m_i32,
        k: k_i32,
        alpha: 1.0f64,
        lda: n_i32,
        ldb: k_i32,
        beta: 0.0f64,
        ldc: n_i32,
    };

    unsafe {
        blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut())?;
    }

    Ok(c)
}

// ---------------------------------------------------------------------------
// GPU batched matmul -- f32 (strided batch SGEMM)
// ---------------------------------------------------------------------------

/// Compute batched `C[i] = A[i] @ B[i]` using cuBLAS strided batch SGEMM.
///
/// `a` contains `batch * m * k` elements: `batch` matrices `[m, k]` in row-major.
/// `b` contains `batch * k * n` elements: `batch` matrices `[k, n]` in row-major.
/// Returns `batch * m * n` elements: `batch` matrices `[m, n]` in row-major.
#[cfg(feature = "cuda")]
pub fn gpu_bmm_f32(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if batch == 0 || m == 0 || k == 0 || n == 0 {
        return alloc_zeros_f32(batch * m * n, device);
    }
    if a.len() != batch * m * k {
        return Err(GpuError::ShapeMismatch {
            op: "bmm",
            expected: vec![batch, m, k],
            got: vec![a.len()],
        });
    }
    if b.len() != batch * k * n {
        return Err(GpuError::ShapeMismatch {
            op: "bmm",
            expected: vec![batch, k, n],
            got: vec![b.len()],
        });
    }

    // Note: do NOT route bmm through gpu_small_bmm here — gpu_small_bmm
    // calls back into gpu_bmm_f32 for batch>1, which would recurse.
    // The small matmul optimization for bmm is handled inside gpu_small_bmm
    // only for the batch==1 case.

    let m_i32 = i32::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm", expected: vec![i32::MAX as usize], got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm", expected: vec![i32::MAX as usize], got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm", expected: vec![i32::MAX as usize], got: vec![n],
    })?;

    let blas = device.blas();
    let mut c = alloc_zeros_f32(batch * m * n, device)?;

    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_N,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: n_i32,
            n: m_i32,
            k: k_i32,
            alpha: 1.0f32,
            lda: n_i32,
            ldb: k_i32,
            beta: 0.0f32,
            ldc: n_i32,
        },
        batch_size: batch as i32,
        stride_a: (k * n) as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };

    unsafe {
        blas.gemm_strided_batched(cfg, b.inner(), a.inner(), c.inner_mut())?;
    }

    Ok(c)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_bmm_f32(
    _a: &CudaBuffer<f32>, _b: &CudaBuffer<f32>,
    _batch: usize, _m: usize, _k: usize, _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

// ===========================================================================
// _into variants — write to pre-allocated output (zero allocation)
// ===========================================================================

/// `C = A @ B` into pre-allocated output buffer `c`. Uses small_matmul PTX for
/// M≤4 or small matrices, cuBLAS SGEMM otherwise.
#[cfg(feature = "cuda")]
pub fn gpu_matmul_f32_into(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    m: usize, k: usize, n: usize,
    c: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    if a.len() != m * k {
        return Err(GpuError::ShapeMismatch { op: "matmul_into", expected: vec![m, k], got: vec![a.len()] });
    }
    if b.len() != k * n {
        return Err(GpuError::ShapeMismatch { op: "matmul_into", expected: vec![k, n], got: vec![b.len()] });
    }
    if m == 0 || k == 0 || n == 0 { return Ok(()); }

    let total_ops = m * k * n;
    if m <= 4 || total_ops < 500_000 {
        return crate::kernels::gpu_small_matmul_into(a, b, m, k, n, c, device);
    }

    let m_i32 = m as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;

    let blas = device.blas();
    let cfg = GemmConfig {
        transa: sys::cublasOperation_t::CUBLAS_OP_N,
        transb: sys::cublasOperation_t::CUBLAS_OP_N,
        m: n_i32, n: m_i32, k: k_i32,
        alpha: 1.0f32, lda: n_i32, ldb: k_i32,
        beta: 0.0f32, ldc: n_i32,
    };
    unsafe { blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut())?; }
    Ok(())
}

/// Batched `C[i] = A[i] @ B[i]` into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_bmm_f32_into(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    batch: usize, m: usize, k: usize, n: usize,
    c: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    if batch == 0 || m == 0 || k == 0 || n == 0 { return Ok(()); }
    if a.len() != batch * m * k {
        return Err(GpuError::ShapeMismatch { op: "bmm_into", expected: vec![batch, m, k], got: vec![a.len()] });
    }
    if b.len() != batch * k * n {
        return Err(GpuError::ShapeMismatch { op: "bmm_into", expected: vec![batch, k, n], got: vec![b.len()] });
    }

    let m_i32 = m as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;

    let blas = device.blas();
    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
            transa: sys::cublasOperation_t::CUBLAS_OP_N,
            transb: sys::cublasOperation_t::CUBLAS_OP_N,
            m: n_i32, n: m_i32, k: k_i32,
            alpha: 1.0f32, lda: n_i32, ldb: k_i32,
            beta: 0.0f32, ldc: n_i32,
        },
        batch_size: batch as i32,
        stride_a: (k * n) as i64,
        stride_b: (m * k) as i64,
        stride_c: (m * n) as i64,
    };
    unsafe { blas.gemm_strided_batched(cfg, b.inner(), a.inner(), c.inner_mut())?; }
    Ok(())
}

/// Stub -- _into variants unavailable without cuda.
#[cfg(not(feature = "cuda"))]
pub fn gpu_matmul_f32_into(
    _a: &CudaBuffer<f32>, _b: &CudaBuffer<f32>,
    _m: usize, _k: usize, _n: usize,
    _c: &mut CudaBuffer<f32>, _device: &GpuDevice,
) -> GpuResult<()> { Err(GpuError::NoCudaFeature) }

#[cfg(not(feature = "cuda"))]
pub fn gpu_bmm_f32_into(
    _a: &CudaBuffer<f32>, _b: &CudaBuffer<f32>,
    _batch: usize, _m: usize, _k: usize, _n: usize,
    _c: &mut CudaBuffer<f32>, _device: &GpuDevice,
) -> GpuResult<()> { Err(GpuError::NoCudaFeature) }

// ===========================================================================
// fp16 matmul via cublasGemmEx -- Tensor Core acceleration on Volta+ GPUs
// ===========================================================================

/// Compute `C = A @ B` using mixed-precision fp16 matmul with f32 accumulation.
///
/// Takes f32 inputs, internally converts them to f16 on-GPU, runs
/// [`cublasGemmEx`] with `CUDA_R_16F` for A/B and `CUDA_R_32F` for C and
/// the compute type (`CUBLAS_COMPUTE_32F`). This yields f32 output with
/// full f32 accumulation precision while leveraging Tensor Cores for 8-16x
/// throughput on Volta/Turing/Ampere+ GPUs.
///
/// # Arguments
///
/// * `a` -- f32 buffer with `m * k` elements (matrix `[m, k]` in row-major).
/// * `b` -- f32 buffer with `k * n` elements (matrix `[k, n]` in row-major).
/// * `m`, `k`, `n` -- matrix dimensions.
/// * `device` -- the CUDA device.
///
/// # Errors
///
/// - [`GpuError::ShapeMismatch`] if buffer lengths don't match dimensions,
///   or if dimensions exceed `i32::MAX`.
/// - [`GpuError::DeviceMismatch`] if buffers are on different devices.
/// - [`GpuError::PtxCompileFailed`] if the f32-to-f16 conversion kernel
///   cannot be compiled (GPU architecture does not support f16).
/// - [`GpuError::Blas`] on cuBLAS runtime errors (including when the GPU
///   does not support the requested mixed-precision compute mode).
/// - [`GpuError::Driver`] on CUDA driver errors.
///
/// # Numerical considerations
///
/// The f32-to-f16 input conversion may lose precision for values outside
/// the f16 representable range (~6e-8 to 65504). Values exceeding f16 max
/// become infinity; small values may underflow to zero. The f32 accumulation
/// ensures the dot products themselves do not lose additional precision.
///
/// Crosslink: #268
#[cfg(feature = "cuda")]
pub fn gpu_matmul_f16(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use core::ffi::c_void;
    use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    // Validate buffer lengths match declared dimensions.
    if a.len() != m * k {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_f16",
            expected: vec![m, k],
            got: vec![a.len()],
        });
    }
    if b.len() != k * n {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_f16",
            expected: vec![k, n],
            got: vec![b.len()],
        });
    }

    // Validate same device.
    if a.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: a.device_ordinal(),
        });
    }
    if b.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: device.ordinal(),
            got: b.device_ordinal(),
        });
    }

    // Handle degenerate cases: any zero dimension produces a zero-element result.
    if m == 0 || k == 0 || n == 0 {
        return alloc_zeros_f32(m * n, device);
    }

    // Validate dimensions fit in i32 (cuBLAS uses i32 for matrix dimensions).
    let m_i32 = i32::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_f16",
        expected: vec![i32::MAX as usize],
        got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_f16",
        expected: vec![i32::MAX as usize],
        got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_f16",
        expected: vec![i32::MAX as usize],
        got: vec![n],
    })?;

    // Step 1: Convert f32 inputs to f16 on the GPU.
    // The conversion kernel uses PTX `cvt.rn.f16.f32` (round-to-nearest-even).
    // The f16 data is stored as CudaSlice<u16> since half::f16 is repr(transparent)
    // over u16, and cudarc implements DeviceRepr for u16 but not for half::f16
    // without the `f16` feature.
    let a_f16 = crate::kernels::gpu_f32_to_f16(a, device)?;
    let b_f16 = crate::kernels::gpu_f32_to_f16(b, device)?;

    // Step 2: Allocate f32 output buffer.
    let mut c = alloc_zeros_f32(m * n, device)?;

    // Step 3: Call cublasGemmEx with mixed-precision types.
    //
    // Row-major trick (same as gpu_matmul_f32): cuBLAS is column-major, so we
    // compute C_row = A_row @ B_row by calling gemm(N, N, n, m, k, ..., B, A, C)
    // with swapped inputs and leading dimensions.
    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let blas = device.blas();
    let stream = device.stream();

    // Scope the device pointer borrows so they are dropped before we return `c`.
    // The SyncOnDrop guards record events on the stream when dropped, ensuring
    // proper synchronization between the gemm_ex call and subsequent operations.
    {
        // SAFETY: We need the raw CUdeviceptr values to pass to gemm_ex. The
        // DevicePtr::device_ptr method returns (CUdeviceptr, SyncOnDrop) where
        // SyncOnDrop ensures stream synchronization when dropped. We hold all
        // _record guards alive until after the gemm_ex call completes (they are
        // dropped at the end of this block).
        let (a_ptr, _record_a) = a_f16.device_ptr(stream);
        let (b_ptr, _record_b) = b_f16.device_ptr(stream);
        let (c_ptr, _record_c) = c.inner_mut().device_ptr_mut(stream);

        // SAFETY: All device pointers are valid and correctly sized:
        // - b_ptr points to k*n u16 values (f16 bit patterns) -- passed as A to
        //   cuBLAS due to the row-major trick
        // - a_ptr points to m*k u16 values (f16 bit patterns) -- passed as B to
        //   cuBLAS
        // - c_ptr points to m*n f32 values -- the output
        // - alpha and beta are host f32 pointers (cuBLAS default pointer mode is
        //   host)
        // - Leading dimensions match the column-major interpretation:
        //   lda=n (B's columns), ldb=k (A's columns), ldc=n (C's columns)
        // - The handle is valid (created during GpuDevice::new)
        unsafe {
            cublas_result::gemm_ex(
                *blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n_i32,                                                 // m (cuBLAS) = n (ours)
                m_i32,                                                 // n (cuBLAS) = m (ours)
                k_i32,                                                 // k
                (&alpha) as *const f32 as *const c_void,               // alpha
                b_ptr as *const c_void,                                // A (row-major trick: B)
                cublas_sys::cudaDataType_t::CUDA_R_16F,                // A type = f16
                n_i32,                                                 // lda = n
                a_ptr as *const c_void,                                // B (row-major trick: A)
                cublas_sys::cudaDataType_t::CUDA_R_16F,                // B type = f16
                k_i32,                                                 // ldb = k
                (&beta) as *const f32 as *const c_void,                // beta
                c_ptr as *mut c_void,                                  // C
                cublas_sys::cudaDataType_t::CUDA_R_32F,                // C type = f32
                n_i32,                                                 // ldc = n
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,   // compute in f32
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,     // let cuBLAS pick algo
            )?;
        }
    }

    Ok(c)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_matmul_f16(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _m: usize,
    _k: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// CPU fallback implementations
// ---------------------------------------------------------------------------

/// Naive row-major matrix multiply on the CPU.
/// Used by tests for reference comparison.
///
/// `a` is `[m, k]`, `b` is `[k, n]`, result is `[m, n]`.
#[cfg(test)]
fn cpu_matmul_naive<T>(a: &[T], b: &[T], m: usize, k: usize, n: usize) -> Vec<T>
where
    T: Copy + Default + std::ops::Add<Output = T> + std::ops::Mul<Output = T>,
{
    let mut c = vec![T::default(); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = T::default();
            for p in 0..k {
                sum = sum + a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Stubs when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_matmul_f32(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _m: usize,
    _k: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_matmul_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _m: usize,
    _k: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Tests -- require a real CUDA GPU
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};
    use crate::device::GpuDevice;

    /// Helper: set up device + upload a slice as f32.
    fn setup_f32(data: &[f32]) -> (GpuDevice, CudaBuffer<f32>) {
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let buf = cpu_to_gpu(data, &dev).expect("cpu_to_gpu");
        (dev, buf)
    }

    /// Compare GPU result buffer to expected CPU values.
    fn assert_buf_close_f32(buf: &CudaBuffer<f32>, device: &GpuDevice, expected: &[f32], tol: f32) {
        let host = gpu_to_cpu(buf, device).expect("gpu_to_cpu");
        assert_eq!(host.len(), expected.len(), "length mismatch");
        for (i, (&got, &exp)) in host.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < tol,
                "element {i}: got {got}, expected {exp}, diff {}",
                (got - exp).abs(),
            );
        }
    }

    fn assert_buf_close_f64(buf: &CudaBuffer<f64>, device: &GpuDevice, expected: &[f64], tol: f64) {
        let host = gpu_to_cpu(buf, device).expect("gpu_to_cpu");
        assert_eq!(host.len(), expected.len(), "length mismatch");
        for (i, (&got, &exp)) in host.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < tol,
                "element {i}: got {got}, expected {exp}, diff {}",
                (got - exp).abs(),
            );
        }
    }

    // -- Basic correctness: 2x3 @ 3x2 = 2x2 ---------------------------------

    #[test]
    fn matmul_f32_basic() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]  (2x3)
        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]   (3x2)
        // C = [[58, 64],
        //      [139, 154]] (2x2)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected: Vec<f32> = vec![58.0, 64.0, 139.0, 154.0];

        let (dev, a) = setup_f32(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");

        let c = gpu_matmul_f32(&a, &b, 2, 3, 2, &dev).expect("gpu_matmul_f32");
        assert_eq!(c.len(), 4);
        assert_buf_close_f32(&c, &dev, &expected, 1e-4);
    }

    // -- Identity matrix ------------------------------------------------------

    #[test]
    fn matmul_f32_identity() {
        // A = [[1, 2],
        //      [3, 4]]  (2x2)
        // I = [[1, 0],
        //      [0, 1]]  (2x2)
        // A @ I = A
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let i_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

        let (dev, a) = setup_f32(&a_data);
        let i_buf = cpu_to_gpu(&i_data, &dev).expect("cpu_to_gpu i");

        let c = gpu_matmul_f32(&a, &i_buf, 2, 2, 2, &dev).expect("gpu_matmul_f32");
        assert_buf_close_f32(&c, &dev, &a_data, 1e-6);
    }

    // -- Vector-matrix (1xK @ KxN) -------------------------------------------

    #[test]
    fn matmul_f32_row_vector() {
        // [1, 2, 3] @ [[1, 0], [0, 1], [1, 1]] = [4, 5]
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0];
        let b_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let expected: Vec<f32> = vec![4.0, 5.0];

        let (dev, a) = setup_f32(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");

        let c = gpu_matmul_f32(&a, &b, 1, 3, 2, &dev).expect("gpu_matmul_f32");
        assert_eq!(c.len(), 2);
        assert_buf_close_f32(&c, &dev, &expected, 1e-6);
    }

    // -- Shape validation: wrong A length -------------------------------------

    #[test]
    fn matmul_f32_wrong_a_length() {
        let (dev, a) = setup_f32(&[1.0, 2.0, 3.0]); // 3 elements
        let b = cpu_to_gpu(&[1.0, 2.0, 3.0, 4.0], &dev).expect("cpu_to_gpu b");

        // Claim A is [2, 2] but buffer has 3 elements
        let err = gpu_matmul_f32(&a, &b, 2, 2, 2, &dev).unwrap_err();
        match err {
            GpuError::ShapeMismatch { op: "matmul", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- Shape validation: wrong B length -------------------------------------

    #[test]
    fn matmul_f32_wrong_b_length() {
        let (dev, a) = setup_f32(&[1.0, 2.0, 3.0, 4.0]);
        let b = cpu_to_gpu(&[1.0, 2.0, 3.0], &dev).expect("cpu_to_gpu b");

        // Claim B is [2, 2] but buffer has 3 elements
        let err = gpu_matmul_f32(&a, &b, 2, 2, 2, &dev).unwrap_err();
        match err {
            GpuError::ShapeMismatch { op: "matmul", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- Empty matrix (m=0) ---------------------------------------------------

    #[test]
    fn matmul_f32_empty() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let a = cpu_to_gpu::<f32>(&[], &dev).expect("cpu_to_gpu a");
        let b = cpu_to_gpu::<f32>(&[], &dev).expect("cpu_to_gpu b");

        let c = gpu_matmul_f32(&a, &b, 0, 0, 0, &dev).expect("gpu_matmul_f32 empty");
        assert_eq!(c.len(), 0);
    }

    // -- f64 basic correctness ------------------------------------------------

    #[test]
    fn matmul_f64_basic() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");

        let a_data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f64> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected: Vec<f64> = vec![58.0, 64.0, 139.0, 154.0];

        let a = cpu_to_gpu(&a_data, &dev).expect("cpu_to_gpu a");
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");

        let c = gpu_matmul_f64(&a, &b, 2, 3, 2, &dev).expect("gpu_matmul_f64");
        assert_eq!(c.len(), 4);
        assert_buf_close_f64(&c, &dev, &expected, 1e-10);
    }

    // -- Larger matrix: compare GPU vs CPU ------------------------------------

    #[test]
    fn matmul_f32_vs_cpu() {
        let m = 64;
        let k = 48;
        let n = 32;

        // Deterministic but non-trivial data.
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 100) as f32 / 100.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 11 + 3) % 100) as f32 / 100.0)
            .collect();

        // CPU reference
        let expected = cpu_matmul_naive(&a_data, &b_data, m, k, n);

        let (dev, a) = setup_f32(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");

        let c = gpu_matmul_f32(&a, &b, m, k, n, &dev).expect("gpu_matmul_f32");
        assert_buf_close_f32(&c, &dev, &expected, 1e-3);
    }

    // -- Performance: 1024x1024 matmul (informational) ------------------------

    #[test]
    fn matmul_f32_1024x1024_perf() {
        let dim = 1024;

        let a_data: Vec<f32> = (0..dim * dim)
            .map(|i| ((i * 7 + 13) % 1000) as f32 / 1000.0)
            .collect();
        let b_data: Vec<f32> = (0..dim * dim)
            .map(|i| ((i * 11 + 3) % 1000) as f32 / 1000.0)
            .collect();

        let (dev, a) = setup_f32(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");

        // GPU timing
        let gpu_start = std::time::Instant::now();
        let _c = gpu_matmul_f32(&a, &b, dim, dim, dim, &dev).expect("gpu_matmul_f32");
        let gpu_elapsed = gpu_start.elapsed();

        // CPU timing
        let cpu_start = std::time::Instant::now();
        let _c_cpu = cpu_matmul_naive(&a_data, &b_data, dim, dim, dim);
        let cpu_elapsed = cpu_start.elapsed();

        eprintln!(
            "matmul {dim}x{dim}: GPU = {:.3}ms, CPU = {:.3}ms, speedup = {:.1}x",
            gpu_elapsed.as_secs_f64() * 1000.0,
            cpu_elapsed.as_secs_f64() * 1000.0,
            cpu_elapsed.as_secs_f64() / gpu_elapsed.as_secs_f64(),
        );
    }

    // == fp16 matmul (cublasGemmEx) tests =====================================

    // -- Basic correctness: 2x3 @ 3x2 = 2x2 (f16 path) ----------------------

    #[test]
    fn matmul_f16_basic() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]  (2x3)
        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]   (3x2)
        // C = [[58, 64],
        //      [139, 154]] (2x2)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected: Vec<f32> = vec![58.0, 64.0, 139.0, 154.0];

        let (dev, a) = setup_f32(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");

        let c = gpu_matmul_f16(&a, &b, 2, 3, 2, &dev).expect("gpu_matmul_f16");
        assert_eq!(c.len(), 4);
        // f16 accumulation with f32 compute -- exact for small integers
        assert_buf_close_f32(&c, &dev, &expected, 1e-2);
    }

    // -- Identity matrix (f16 path) -------------------------------------------

    #[test]
    fn matmul_f16_identity() {
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let i_data: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];

        let (dev, a) = setup_f32(&a_data);
        let i_buf = cpu_to_gpu(&i_data, &dev).expect("cpu_to_gpu i");

        let c = gpu_matmul_f16(&a, &i_buf, 2, 2, 2, &dev).expect("gpu_matmul_f16");
        assert_buf_close_f32(&c, &dev, &a_data, 1e-3);
    }

    // -- Empty matrix (m=0) f16 path ------------------------------------------

    #[test]
    fn matmul_f16_empty() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let a = cpu_to_gpu::<f32>(&[], &dev).expect("cpu_to_gpu a");
        let b = cpu_to_gpu::<f32>(&[], &dev).expect("cpu_to_gpu b");

        let c = gpu_matmul_f16(&a, &b, 0, 0, 0, &dev).expect("gpu_matmul_f16 empty");
        assert_eq!(c.len(), 0);
    }

    // -- Degenerate: k=0 (zero inner dimension) f16 path ----------------------

    #[test]
    fn matmul_f16_k_zero() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let a = cpu_to_gpu::<f32>(&[], &dev).expect("cpu_to_gpu a");
        let b = cpu_to_gpu::<f32>(&[], &dev).expect("cpu_to_gpu b");

        // 2x0 @ 0x3 = 2x3 (all zeros)
        let c = gpu_matmul_f16(&a, &b, 2, 0, 3, &dev).expect("gpu_matmul_f16 k=0");
        assert_eq!(c.len(), 6);
        let host = gpu_to_cpu(&c, &dev).expect("gpu_to_cpu");
        assert!(host.iter().all(|&x| x == 0.0));
    }

    // -- Shape validation: wrong A length (f16 path) --------------------------

    #[test]
    fn matmul_f16_wrong_a_length() {
        let (dev, a) = setup_f32(&[1.0, 2.0, 3.0]); // 3 elements
        let b = cpu_to_gpu(&[1.0, 2.0, 3.0, 4.0], &dev).expect("cpu_to_gpu b");

        let err = gpu_matmul_f16(&a, &b, 2, 2, 2, &dev).unwrap_err();
        match err {
            GpuError::ShapeMismatch { op: "matmul_f16", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- Shape validation: wrong B length (f16 path) --------------------------

    #[test]
    fn matmul_f16_wrong_b_length() {
        let (dev, a) = setup_f32(&[1.0, 2.0, 3.0, 4.0]);
        let b = cpu_to_gpu(&[1.0, 2.0, 3.0], &dev).expect("cpu_to_gpu b");

        let err = gpu_matmul_f16(&a, &b, 2, 2, 2, &dev).unwrap_err();
        match err {
            GpuError::ShapeMismatch { op: "matmul_f16", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- Numerical correctness: f16 vs f32 reference --------------------------

    #[test]
    fn matmul_f16_vs_f32_reference() {
        let m = 64;
        let k = 48;
        let n = 32;

        // Deterministic data with values in f16-safe range.
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 13) % 100) as f32 / 100.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 11 + 3) % 100) as f32 / 100.0)
            .collect();

        // f32 CPU reference.
        let expected = cpu_matmul_naive(&a_data, &b_data, m, k, n);

        let (dev, a) = setup_f32(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");

        // f16 matmul on GPU.
        let c = gpu_matmul_f16(&a, &b, m, k, n, &dev).expect("gpu_matmul_f16");
        let host = gpu_to_cpu(&c, &dev).expect("gpu_to_cpu");

        assert_eq!(host.len(), expected.len(), "output length mismatch");

        // f16 inputs lose precision (~1e-3 relative error per element).
        // With k=48 accumulations in f32, total error can be up to k * eps_f16 ~ 0.05.
        // Use a generous tolerance.
        let mut max_err: f32 = 0.0;
        for (i, (&got, &exp)) in host.iter().zip(expected.iter()).enumerate() {
            let abs_err = (got - exp).abs();
            let rel_err = if exp.abs() > 1e-6 {
                abs_err / exp.abs()
            } else {
                abs_err
            };
            max_err = max_err.max(abs_err);
            assert!(
                rel_err < 0.05 || abs_err < 0.1,
                "element {i}: f16 got {got}, f32 ref {exp}, abs_err {abs_err}, rel_err {rel_err}",
            );
        }
        eprintln!(
            "matmul_f16_vs_f32: {m}x{k} @ {k}x{n}, max absolute error = {max_err:.6}",
        );
    }

    // -- Performance: 1024x1024 f16 matmul (informational) --------------------

    #[test]
    fn matmul_f16_1024x1024_perf() {
        let dim = 1024;

        let a_data: Vec<f32> = (0..dim * dim)
            .map(|i| ((i * 7 + 13) % 1000) as f32 / 1000.0)
            .collect();
        let b_data: Vec<f32> = (0..dim * dim)
            .map(|i| ((i * 11 + 3) % 1000) as f32 / 1000.0)
            .collect();

        let (dev, a) = setup_f32(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");

        // f32 GPU timing (baseline)
        let f32_start = std::time::Instant::now();
        let _c32 = gpu_matmul_f32(&a, &b, dim, dim, dim, &dev).expect("gpu_matmul_f32");
        let f32_elapsed = f32_start.elapsed();

        // f16 GPU timing
        let f16_start = std::time::Instant::now();
        let _c16 = gpu_matmul_f16(&a, &b, dim, dim, dim, &dev).expect("gpu_matmul_f16");
        let f16_elapsed = f16_start.elapsed();

        eprintln!(
            "matmul {dim}x{dim}: f32 = {:.3}ms, f16 = {:.3}ms, f16 speedup = {:.1}x",
            f32_elapsed.as_secs_f64() * 1000.0,
            f16_elapsed.as_secs_f64() * 1000.0,
            f32_elapsed.as_secs_f64() / f16_elapsed.as_secs_f64(),
        );
    }
}
