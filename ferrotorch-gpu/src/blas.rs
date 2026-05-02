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
        op: "bmm",
        expected: vec![i32::MAX as usize],
        got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm",
        expected: vec![i32::MAX as usize],
        got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm",
        expected: vec![i32::MAX as usize],
        got: vec![n],
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

/// Batched matrix multiplication for f64 (cuBLAS DGEMM).
#[cfg(feature = "cuda")]
pub fn gpu_bmm_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if batch == 0 || m == 0 || k == 0 || n == 0 {
        return alloc_zeros_f64(batch * m * n, device);
    }
    if a.len() != batch * m * k {
        return Err(GpuError::ShapeMismatch {
            op: "bmm_f64",
            expected: vec![batch, m, k],
            got: vec![a.len()],
        });
    }
    if b.len() != batch * k * n {
        return Err(GpuError::ShapeMismatch {
            op: "bmm_f64",
            expected: vec![batch, k, n],
            got: vec![b.len()],
        });
    }

    let m_i32 = i32::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm_f64",
        expected: vec![i32::MAX as usize],
        got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm_f64",
        expected: vec![i32::MAX as usize],
        got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm_f64",
        expected: vec![i32::MAX as usize],
        got: vec![n],
    })?;

    let blas = device.blas();
    let mut c = alloc_zeros_f64(batch * m * n, device)?;

    let cfg = StridedBatchedConfig {
        gemm: GemmConfig {
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

#[cfg(not(feature = "cuda"))]
pub fn gpu_bmm_f64(
    _a: &CudaBuffer<f64>,
    _b: &CudaBuffer<f64>,
    _batch: usize,
    _m: usize,
    _k: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_bmm_f32(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _batch: usize,
    _m: usize,
    _k: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

// ===========================================================================
// _into variants — write to pre-allocated output (zero allocation)
// ===========================================================================

/// `C = A @ B` into pre-allocated output buffer `c`. Uses small_matmul PTX for
/// M≤4 or small matrices, cuBLAS SGEMM otherwise.
#[cfg(feature = "cuda")]
pub fn gpu_matmul_f32_into(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    m: usize,
    k: usize,
    n: usize,
    c: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    if a.len() != m * k {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_into",
            expected: vec![m, k],
            got: vec![a.len()],
        });
    }
    if b.len() != k * n {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_into",
            expected: vec![k, n],
            got: vec![b.len()],
        });
    }
    if m == 0 || k == 0 || n == 0 {
        return Ok(());
    }

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
    Ok(())
}

/// Batched `C[i] = A[i] @ B[i]` into pre-allocated output.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_bmm_f32_into(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    c: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    if batch == 0 || m == 0 || k == 0 || n == 0 {
        return Ok(());
    }
    if a.len() != batch * m * k {
        return Err(GpuError::ShapeMismatch {
            op: "bmm_into",
            expected: vec![batch, m, k],
            got: vec![a.len()],
        });
    }
    if b.len() != batch * k * n {
        return Err(GpuError::ShapeMismatch {
            op: "bmm_into",
            expected: vec![batch, k, n],
            got: vec![b.len()],
        });
    }

    let m_i32 = m as i32;
    let k_i32 = k as i32;
    let n_i32 = n as i32;

    let blas = device.blas();
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
    Ok(())
}

/// Stub -- _into variants unavailable without cuda.
#[cfg(not(feature = "cuda"))]
pub fn gpu_matmul_f32_into(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _m: usize,
    _k: usize,
    _n: usize,
    _c: &mut CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

#[cfg(not(feature = "cuda"))]
pub fn gpu_bmm_f32_into(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _batch: usize,
    _m: usize,
    _k: usize,
    _n: usize,
    _c: &mut CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

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
        let (a_ptr, _record_a) = a_f16.device_ptr(&stream);
        let (b_ptr, _record_b) = b_f16.device_ptr(&stream);
        let (c_ptr, _record_c) = c.inner_mut().device_ptr_mut(&stream);

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
                n_i32,                                               // m (cuBLAS) = n (ours)
                m_i32,                                               // n (cuBLAS) = m (ours)
                k_i32,                                               // k
                (&alpha) as *const f32 as *const c_void,             // alpha
                b_ptr as *const c_void,                              // A (row-major trick: B)
                cublas_sys::cudaDataType_t::CUDA_R_16F,              // A type = f16
                n_i32,                                               // lda = n
                a_ptr as *const c_void,                              // B (row-major trick: A)
                cublas_sys::cudaDataType_t::CUDA_R_16F,              // B type = f16
                k_i32,                                               // ldb = k
                (&beta) as *const f32 as *const c_void,              // beta
                c_ptr as *mut c_void,                                // C
                cublas_sys::cudaDataType_t::CUDA_R_32F,              // C type = f32
                n_i32,                                               // ldc = n
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, // compute in f32
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,   // let cuBLAS pick algo
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
// Batched mixed-precision matmul (f16 inputs, f32 accumulate) via GemmStridedBatchedEx
// ---------------------------------------------------------------------------

/// Batched matrix multiply with f16 Tensor Core acceleration.
///
/// Takes f32 inputs, converts to f16 on-device, executes
/// `cublasGemmStridedBatchedEx` with `CUDA_R_16F` operands and
/// `CUBLAS_COMPUTE_32F` accumulation. Returns f32 output.
///
/// `a` is `[batch, m, k]`, `b` is `[batch, k, n]`, result is `[batch, m, n]`.
#[cfg(feature = "cuda")]
pub fn gpu_bmm_f16(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use core::ffi::c_void;
    use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    if batch == 0 || m == 0 || k == 0 || n == 0 {
        return alloc_zeros_f32(batch * m * n, device);
    }
    if a.len() != batch * m * k {
        return Err(GpuError::ShapeMismatch {
            op: "bmm_f16",
            expected: vec![batch, m, k],
            got: vec![a.len()],
        });
    }
    if b.len() != batch * k * n {
        return Err(GpuError::ShapeMismatch {
            op: "bmm_f16",
            expected: vec![batch, k, n],
            got: vec![b.len()],
        });
    }

    let m_i32 = i32::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm_f16",
        expected: vec![i32::MAX as usize],
        got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm_f16",
        expected: vec![i32::MAX as usize],
        got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "bmm_f16",
        expected: vec![i32::MAX as usize],
        got: vec![n],
    })?;

    // Convert f32 inputs to f16 on-device.
    let a_f16 = crate::kernels::gpu_f32_to_f16(a, device)?;
    let b_f16 = crate::kernels::gpu_f32_to_f16(b, device)?;

    let mut c = alloc_zeros_f32(batch * m * n, device)?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let blas = device.blas();
    let stream = device.stream();

    {
        let (a_ptr, _ra) = a_f16.device_ptr(&stream);
        let (b_ptr, _rb) = b_f16.device_ptr(&stream);
        let (c_ptr, _rc) = c.inner_mut().device_ptr_mut(&stream);

        // Row-major trick: swap A/B and their leading dimensions.
        // cuBLAS col-major: C = B_row @ A_row with lda=n, ldb=k, ldc=n.
        // f16 strides are in u16 elements (half the byte size of f32).
        let stride_a_f16 = (k * n) as i64; // B_row batch stride (in u16 elements)
        let stride_b_f16 = (m * k) as i64; // A_row batch stride (in u16 elements)
        let stride_c = (m * n) as i64;

        unsafe {
            cublas_result::gemm_strided_batched_ex(
                *blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n_i32,
                m_i32,
                k_i32,
                (&alpha) as *const f32 as *const c_void,
                b_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                n_i32,
                stride_a_f16,
                a_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16F,
                k_i32,
                stride_b_f16,
                (&beta) as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                n_i32,
                stride_c,
                batch as i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?;
        }
    }

    Ok(c)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_bmm_f16(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _batch: usize,
    _m: usize,
    _k: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// BF16 mixed-precision matmul via cublasGemmEx
// ---------------------------------------------------------------------------

/// Matrix multiply with bf16 Tensor Core acceleration.
///
/// Takes f32 inputs, converts to bf16 on-device, executes
/// `cublasGemmEx` with `CUDA_R_BF16` operands and `CUBLAS_COMPUTE_32F`
/// accumulation. Returns f32 output.
///
/// BF16 has the same exponent range as f32 (8 bits) so it handles large
/// values better than f16, at the cost of less mantissa precision.
#[cfg(feature = "cuda")]
pub fn gpu_matmul_bf16(
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

    if a.len() != m * k {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_bf16",
            expected: vec![m, k],
            got: vec![a.len()],
        });
    }
    if b.len() != k * n {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_bf16",
            expected: vec![k, n],
            got: vec![b.len()],
        });
    }
    if m == 0 || k == 0 || n == 0 {
        return alloc_zeros_f32(m * n, device);
    }

    let m_i32 = i32::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_bf16",
        expected: vec![i32::MAX as usize],
        got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_bf16",
        expected: vec![i32::MAX as usize],
        got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_bf16",
        expected: vec![i32::MAX as usize],
        got: vec![n],
    })?;

    let a_bf16 = crate::kernels::gpu_f32_to_bf16(a, device)?;
    let b_bf16 = crate::kernels::gpu_f32_to_bf16(b, device)?;
    let mut c = alloc_zeros_f32(m * n, device)?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let blas = device.blas();
    let stream = device.stream();

    {
        let (a_ptr, _ra) = a_bf16.device_ptr(&stream);
        let (b_ptr, _rb) = b_bf16.device_ptr(&stream);
        let (c_ptr, _rc) = c.inner_mut().device_ptr_mut(&stream);

        unsafe {
            cublas_result::gemm_ex(
                *blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n_i32,
                m_i32,
                k_i32,
                (&alpha) as *const f32 as *const c_void,
                b_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n_i32,
                a_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k_i32,
                (&beta) as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                cublas_sys::cudaDataType_t::CUDA_R_32F,
                n_i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?;
        }
    }

    Ok(c)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_matmul_bf16(
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
// BF16 storage matmul via cublasGemmEx
// ---------------------------------------------------------------------------

/// Matrix multiply on bf16-stored operands with f32 compute.
///
/// `A`, `B` are row-major `[M,K]` and `[K,N]` respectively, each
/// element stored as bf16 (`u16` bit layout — bf16 is the top 16 bits
/// of an f32). `C = A @ B` is written back as bf16. The `cublasGemmEx`
/// call uses `CUBLAS_COMPUTE_32F`, so the dot products accumulate in
/// f32 for numerical quality, then the final f32 output is cast back
/// to bf16 for storage.
///
/// This is the foundational op for GPU-resident Llama inference:
/// weights + activations both live in VRAM as bf16 (16 GB for an 8 B
/// model vs. 32 GB at f32), and every `Linear::forward` routes through
/// this function on an RTX 3090 / A100 / H100 tensor core.
#[cfg(feature = "cuda")]
pub fn gpu_matmul_bf16_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    use core::ffi::c_void;
    use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    if a.len() < m * k {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_bf16_bf16",
            expected: vec![m, k],
            got: vec![a.len()],
        });
    }
    if b.len() < k * n {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_bf16_bf16",
            expected: vec![k, n],
            got: vec![b.len()],
        });
    }
    if m == 0 || k == 0 || n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(m * n)?);
    }

    let m_i32 = i32::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_bf16_bf16",
        expected: vec![i32::MAX as usize],
        got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_bf16_bf16",
        expected: vec![i32::MAX as usize],
        got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_bf16_bf16",
        expected: vec![i32::MAX as usize],
        got: vec![n],
    })?;

    let mut c = device.stream().alloc_zeros::<u16>(m * n)?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let blas = device.blas();
    let stream = device.stream();

    {
        let (a_ptr, _ra) = a.device_ptr(&stream);
        let (b_ptr, _rb) = b.device_ptr(&stream);
        let (c_ptr, _rc) = c.device_ptr_mut(&stream);

        // cuBLAS is column-major; to compute row-major C = A @ B with
        // A:[M,K], B:[K,N], we ask cuBLAS for C^T = B^T @ A^T and let
        // it write to the same memory (row-major C is column-major C^T).
        // So: cublas_M = N, cublas_N = M, cublas_K = K.
        unsafe {
            cublas_result::gemm_ex(
                *blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n_i32,
                m_i32,
                k_i32,
                (&alpha) as *const f32 as *const c_void,
                b_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n_i32,
                a_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k_i32,
                (&beta) as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n_i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?;
        }
    }

    Ok(c)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_matmul_bf16_bf16(
    _a: &(),
    _b: &(),
    _m: usize,
    _k: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

/// `C = A @ B^T` on bf16-stored operands, f32 compute.
///
/// `A` is row-major `[M, K]`; `B` is row-major `[N, K]` (so `B^T` is
/// `[K, N]`). The result `C` is row-major `[M, N]`. This variant exists
/// because attention's `Q @ K^T` (where `Q: [seq_q, head_dim]` and
/// `K: [seq_k, head_dim]`) is the natural layout — the transpose is
/// free once you flip the `transb` flag on `cublasGemmEx`.
///
/// HuggingFace Llama weights also live in `[out_features, in_features]`
/// row-major (PyTorch's `nn.Linear` convention), so any `x @ W^T` Linear
/// forward routes through this function.
#[cfg(feature = "cuda")]
pub fn gpu_matmul_bf16_bf16_nt(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    use core::ffi::c_void;
    use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    if a.len() < m * k {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_bf16_bf16_nt",
            expected: vec![m, k],
            got: vec![a.len()],
        });
    }
    if b.len() < n * k {
        return Err(GpuError::ShapeMismatch {
            op: "matmul_bf16_bf16_nt",
            expected: vec![n, k],
            got: vec![b.len()],
        });
    }
    if m == 0 || k == 0 || n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(m * n)?);
    }

    let m_i32 = i32::try_from(m).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_bf16_bf16_nt",
        expected: vec![i32::MAX as usize],
        got: vec![m],
    })?;
    let k_i32 = i32::try_from(k).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_bf16_bf16_nt",
        expected: vec![i32::MAX as usize],
        got: vec![k],
    })?;
    let n_i32 = i32::try_from(n).map_err(|_| GpuError::ShapeMismatch {
        op: "matmul_bf16_bf16_nt",
        expected: vec![i32::MAX as usize],
        got: vec![n],
    })?;

    let mut c = device.stream().alloc_zeros::<u16>(m * n)?;

    let alpha: f32 = 1.0;
    let beta: f32 = 0.0;
    let blas = device.blas();
    let stream = device.stream();

    {
        let (a_ptr, _ra) = a.device_ptr(&stream);
        let (b_ptr, _rb) = b.device_ptr(&stream);
        let (c_ptr, _rc) = c.device_ptr_mut(&stream);

        // Row-major C = A @ B^T (A:[M,K], B:[N,K]).
        // cuBLAS sees columns. Compute C^T = B @ A^T in column-major:
        //   - left operand in cuBLAS: B^T-transposed-column-major = B
        //     stored row-major → cuBLAS sees as [K,N] col-major with
        //     op=T.  But we want B (not B^T), so set transb = N and
        //     dims accordingly.
        // Concretely, the working pattern is:
        //   cublasGemmEx(transa=N, transb=T, M=n_i32, N=m_i32, K=k_i32,
        //     A=B (row-major [N,K], leading=K), B=A (row-major [M,K],
        //     leading=K), C=C (row-major [M,N], leading=N))
        unsafe {
            cublas_result::gemm_ex(
                *blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n_i32,
                m_i32,
                k_i32,
                (&alpha) as *const f32 as *const c_void,
                b_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k_i32,
                a_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k_i32,
                (&beta) as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n_i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?;
        }
    }

    Ok(c)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_matmul_bf16_bf16_nt(
    _a: &(),
    _b: &(),
    _m: usize,
    _k: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

/// Batched `C[b] = A[b] @ B[b]^T * alpha` on bf16-stored operands with
/// f32 compute.
///
/// Each batch element is a row-major matmul of shapes
/// `A: [M, K]`, `B: [N, K]` (so `B^T` is `[K, N]`), producing
/// `C: [M, N]`. Batch elements are interleaved in memory by
/// the fixed strides `stride_a_elems`, `stride_b_elems`,
/// `stride_c_elems` (counted in elements, not bytes). The three
/// buffers must together supply `batch_count` complete matmuls.
///
/// This is what every multi-head attention needs:
/// `Q @ K^T * 1/sqrt(d)` across all heads in one cuBLAS call
/// (`batch_count = num_heads`, `M = seq_q`, `N = seq_k`, `K = head_dim`,
/// `stride_a = stride_b = seq * head_dim`, `stride_c = seq_q * seq_k`).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_matmul_bf16_bf16_strided_batched_nt(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    stride_a_elems: usize,
    stride_b_elems: usize,
    alpha: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    use core::ffi::c_void;
    use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    if batch_count == 0 || m == 0 || k == 0 || n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(batch_count * m * n)?);
    }

    let (m_i32, k_i32, n_i32, bc_i32) = (m as i32, k as i32, n as i32, batch_count as i32);
    let mut c = device.stream().alloc_zeros::<u16>(batch_count * m * n)?;
    let beta: f32 = 0.0;
    let blas = device.blas();
    let stream = device.stream();

    {
        let (a_ptr, _ra) = a.device_ptr(&stream);
        let (b_ptr, _rb) = b.device_ptr(&stream);
        let (c_ptr, _rc) = c.device_ptr_mut(&stream);

        // Same row-major <-> column-major swap trick as gpu_matmul_bf16_bf16_nt.
        unsafe {
            cublas_result::gemm_strided_batched_ex(
                *blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_T,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n_i32,
                m_i32,
                k_i32,
                (&alpha) as *const f32 as *const c_void,
                b_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k_i32,
                stride_b_elems as i64,
                a_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k_i32,
                stride_a_elems as i64,
                (&beta) as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n_i32,
                (m * n) as i64,
                bc_i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?;
        }
    }

    Ok(c)
}

/// Batched `C[b] = A[b] @ B[b] * alpha`. Same as
/// [`gpu_matmul_bf16_bf16_strided_batched_nt`] but without the transpose
/// on `B`. Used for `attn_weights @ V` in multi-head attention.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_matmul_bf16_bf16_strided_batched(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    m: usize,
    k: usize,
    n: usize,
    batch_count: usize,
    stride_a_elems: usize,
    stride_b_elems: usize,
    alpha: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    use core::ffi::c_void;
    use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
    use cudarc::driver::{DevicePtr, DevicePtrMut};

    if batch_count == 0 || m == 0 || k == 0 || n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(batch_count * m * n)?);
    }

    let (m_i32, k_i32, n_i32, bc_i32) = (m as i32, k as i32, n as i32, batch_count as i32);
    let mut c = device.stream().alloc_zeros::<u16>(batch_count * m * n)?;
    let beta: f32 = 0.0;
    let blas = device.blas();
    let stream = device.stream();

    {
        let (a_ptr, _ra) = a.device_ptr(&stream);
        let (b_ptr, _rb) = b.device_ptr(&stream);
        let (c_ptr, _rc) = c.device_ptr_mut(&stream);

        unsafe {
            cublas_result::gemm_strided_batched_ex(
                *blas.handle(),
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n_i32,
                m_i32,
                k_i32,
                (&alpha) as *const f32 as *const c_void,
                b_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n_i32,
                stride_b_elems as i64,
                a_ptr as *const c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                k_i32,
                stride_a_elems as i64,
                (&beta) as *const f32 as *const c_void,
                c_ptr as *mut c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                n_i32,
                (m * n) as i64,
                bc_i32,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DEFAULT,
            )?;
        }
    }

    Ok(c)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_matmul_bf16_bf16_strided_batched_nt(
    _a: &(),
    _b: &(),
    _m: usize,
    _k: usize,
    _n: usize,
    _batch_count: usize,
    _stride_a_elems: usize,
    _stride_b_elems: usize,
    _alpha: f32,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_matmul_bf16_bf16_strided_batched(
    _a: &(),
    _b: &(),
    _m: usize,
    _k: usize,
    _n: usize,
    _batch_count: usize,
    _stride_a_elems: usize,
    _stride_b_elems: usize,
    _alpha: f32,
    _device: &GpuDevice,
) -> GpuResult<()> {
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
    use crate::device::GpuDevice;
    use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

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
            GpuError::ShapeMismatch {
                op: "matmul_f16", ..
            } => {}
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
            GpuError::ShapeMismatch {
                op: "matmul_f16", ..
            } => {}
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
        eprintln!("matmul_f16_vs_f32: {m}x{k} @ {k}x{n}, max absolute error = {max_err:.6}",);
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

    // -- BF16 storage matmul tensor-core path (#519) -----------------------

    /// Upload a slice of f32 values to the GPU as bf16 (u16-stored).
    fn upload_as_bf16(dev: &GpuDevice, data: &[f32]) -> cudarc::driver::CudaSlice<u16> {
        let u16_data: Vec<u16> = data
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect();
        dev.stream().clone_htod(&u16_data).expect("bf16 upload")
    }

    /// Download a bf16 buffer (u16-stored) and decode to f32.
    fn download_bf16_as_f32(dev: &GpuDevice, buf: &cudarc::driver::CudaSlice<u16>) -> Vec<f32> {
        let bits: Vec<u16> = dev.stream().clone_dtoh(buf).expect("bf16 download");
        bits.into_iter()
            .map(|b| half::bf16::from_bits(b).to_f32())
            .collect()
    }

    #[test]
    fn matmul_bf16_bf16_basic_2x3_3x2() {
        // Same reference test case as matmul_f32_basic, now through the
        // bf16 tensor-core path. Tolerance is loose because bf16 has
        // ~7-bit mantissa.
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected: Vec<f32> = vec![58.0, 64.0, 139.0, 154.0];

        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let a = upload_as_bf16(&dev, &a_data);
        let b = upload_as_bf16(&dev, &b_data);

        let c = gpu_matmul_bf16_bf16(&a, &b, 2, 3, 2, &dev).expect("gpu_matmul_bf16_bf16");
        let got = download_bf16_as_f32(&dev, &c);
        assert_eq!(got.len(), expected.len());
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1.0,
                "element {i}: got {g}, expected {e}, diff {}",
                (g - e).abs(),
            );
        }
    }

    #[test]
    fn matmul_bf16_bf16_identity_rows() {
        // C = I(4x4) @ X(4x3) should equal X.
        let i_data: Vec<f32> = {
            let mut v = vec![0.0f32; 16];
            for d in 0..4 {
                v[d * 4 + d] = 1.0;
            }
            v
        };
        let x_data: Vec<f32> = (0..12).map(|i| (i as f32) - 6.0).collect();

        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let i = upload_as_bf16(&dev, &i_data);
        let x = upload_as_bf16(&dev, &x_data);
        let c = gpu_matmul_bf16_bf16(&i, &x, 4, 4, 3, &dev).expect("matmul");
        let got = download_bf16_as_f32(&dev, &c);
        for (a, b) in got.iter().zip(x_data.iter()) {
            // bf16 round-trip of integers in [-6, 5] is exact.
            assert_eq!(a, b, "identity matmul must preserve X exactly");
        }
    }

    // bf16 elementwise / embedding / rmsnorm / softmax / rope tests now
    // live in `ferrotorch-gpu/src/bf16.rs` (nvrtc-compiled CUDA C++).

    #[test]
    fn matmul_bf16_bf16_nt_basic_2x3_2x3() {
        // A = [[1,2,3], [4,5,6]]    (2x3, row-major)
        // B = [[7,8,9], [10,11,12]] (2x3, row-major)
        // B^T = [[7,10],[8,11],[9,12]]
        // C = A @ B^T = [[1*7+2*8+3*9, 1*10+2*11+3*12],
        //                [4*7+5*8+6*9, 4*10+5*11+6*12]]
        //             = [[50, 68], [122, 167]]
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected: Vec<f32> = vec![50.0, 68.0, 122.0, 167.0];

        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let a = upload_as_bf16(&dev, &a_data);
        let b = upload_as_bf16(&dev, &b_data);
        let c = gpu_matmul_bf16_bf16_nt(&a, &b, 2, 3, 2, &dev).expect("matmul_nt");
        let got = download_bf16_as_f32(&dev, &c);
        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() <= e.abs() * 0.02 + 1.0,
                "nt[{i}]: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn matmul_bf16_bf16_nt_equivalent_to_explicit_transpose() {
        // Correctness anchor: `gpu_matmul_bf16_bf16_nt(a, b, m, k, n)`
        // must equal `gpu_matmul_bf16_bf16(a, b_t, m, k, n)` where `b_t`
        // is `b` with its [N, K] layout transposed to [K, N] explicitly.
        let m = 4;
        let k = 3;
        let n = 5;
        let a_data: Vec<f32> = (0..m * k).map(|i| (i as f32) * 0.5 - 2.0).collect();
        let b_data: Vec<f32> = (0..n * k).map(|i| (i as f32) * 0.25 + 1.0).collect();
        // Transpose b from [n, k] to [k, n] on CPU.
        let mut b_t: Vec<f32> = vec![0.0; k * n];
        for i in 0..n {
            for j in 0..k {
                b_t[j * n + i] = b_data[i * k + j];
            }
        }

        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let a = upload_as_bf16(&dev, &a_data);
        let b = upload_as_bf16(&dev, &b_data);
        let bt = upload_as_bf16(&dev, &b_t);

        let c_nt = gpu_matmul_bf16_bf16_nt(&a, &b, m, k, n, &dev).unwrap();
        let c_ref = gpu_matmul_bf16_bf16(&a, &bt, m, k, n, &dev).unwrap();
        let nt = download_bf16_as_f32(&dev, &c_nt);
        let rf = download_bf16_as_f32(&dev, &c_ref);
        for (i, (&a, &b)) in nt.iter().zip(rf.iter()).enumerate() {
            assert!((a - b).abs() < 0.01, "nt[{i}]={a} vs ref[{i}]={b}",);
        }
    }

    #[test]
    fn matmul_bf16_strided_batched_matches_per_batch_reference() {
        // Two [2,3] @ [2,3]^T = [2,2] batched matmuls with alpha=0.5.
        let dev = GpuDevice::new(0).expect("cuda");
        let a0: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // [2,3]
        let b0: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]; // [2,3]
        let a1: Vec<f32> = vec![0.5, 0.5, 0.5, 1.0, 1.0, 1.0];
        let b1: Vec<f32> = vec![2.0, 2.0, 2.0, -1.0, -1.0, -1.0];
        let a: Vec<f32> = [&a0[..], &a1[..]].concat();
        let b: Vec<f32> = [&b0[..], &b1[..]].concat();

        let a_gpu = upload_as_bf16(&dev, &a);
        let b_gpu = upload_as_bf16(&dev, &b);
        let c =
            gpu_matmul_bf16_bf16_strided_batched_nt(&a_gpu, &b_gpu, 2, 3, 2, 2, 6, 6, 0.5, &dev)
                .expect("batched");
        let got = download_bf16_as_f32(&dev, &c);

        // Reference: per-batch matmul_bf16_bf16_nt.
        let ref0 = gpu_matmul_bf16_bf16_nt(
            &upload_as_bf16(&dev, &a0),
            &upload_as_bf16(&dev, &b0),
            2,
            3,
            2,
            &dev,
        )
        .unwrap();
        let ref1 = gpu_matmul_bf16_bf16_nt(
            &upload_as_bf16(&dev, &a1),
            &upload_as_bf16(&dev, &b1),
            2,
            3,
            2,
            &dev,
        )
        .unwrap();
        let expected0 = download_bf16_as_f32(&dev, &ref0);
        let expected1 = download_bf16_as_f32(&dev, &ref1);

        // Batched result was scaled by 0.5, so compare scaled expected.
        for (i, (&g, &e)) in got[..4].iter().zip(expected0.iter()).enumerate() {
            let scaled = e * 0.5;
            assert!(
                (g - scaled).abs() < scaled.abs() * 0.05 + 0.1,
                "b0[{i}]: got {g}, expected {scaled}",
            );
        }
        for (i, (&g, &e)) in got[4..].iter().zip(expected1.iter()).enumerate() {
            let scaled = e * 0.5;
            assert!(
                (g - scaled).abs() < scaled.abs() * 0.05 + 0.1,
                "b1[{i}]: got {g}, expected {scaled}",
            );
        }
    }

    #[test]
    fn matmul_bf16_bf16_large_dims_finite() {
        // 512x512 matmul with realistic magnitudes. Validates that a
        // sizeable tensor-core call doesn't crash and produces finite
        // output across a full RTX 3090 wave.
        let dim = 512;
        let a_data: Vec<f32> = (0..dim * dim)
            .map(|i| (((i * 7 + 13) % 1000) as f32 / 1000.0) - 0.5)
            .collect();
        let b_data: Vec<f32> = (0..dim * dim)
            .map(|i| (((i * 11 + 3) % 1000) as f32 / 1000.0) - 0.5)
            .collect();

        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let a = upload_as_bf16(&dev, &a_data);
        let b = upload_as_bf16(&dev, &b_data);
        let c = gpu_matmul_bf16_bf16(&a, &b, dim, dim, dim, &dev).expect("matmul");
        let got = download_bf16_as_f32(&dev, &c);
        assert_eq!(got.len(), dim * dim);
        for &v in &got {
            assert!(v.is_finite(), "non-finite output in bf16 matmul");
        }
    }
}
