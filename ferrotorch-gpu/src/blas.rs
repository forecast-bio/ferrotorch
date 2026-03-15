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
use cudarc::cublas::{CudaBlas, Gemm, GemmConfig, sys};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::{alloc_zeros, cpu_to_gpu, gpu_to_cpu};

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
        return alloc_zeros::<f32>(m * n, device);
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

    // Try cuBLAS SGEMM.
    match CudaBlas::new(device.stream().clone()) {
        Ok(blas) => {
            let mut c = alloc_zeros::<f32>(m * n, device)?;

            // Row-major trick: call gemm with B first, A second.
            // C_row = A_row @ B_row  is computed as:
            //   gemm(N, N, n, m, k, 1.0, B, n, A, k, 0.0, C, n)
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

            // SAFETY: All buffers are device-resident, properly sized,
            // and on the same device. The GemmConfig dimensions match the
            // buffer layouts.
            unsafe {
                blas.gemm(cfg, b.inner(), a.inner(), c.inner_mut())?;
            }

            Ok(c)
        }
        Err(_) => {
            // cuBLAS handle creation failed -- fall back to CPU.
            eprintln!(
                "ferrotorch-gpu: cuBLAS handle creation failed, \
                 falling back to CPU matmul for [{m}x{k}] @ [{k}x{n}]"
            );
            cpu_matmul_fallback_f32(a, b, m, k, n, device)
        }
    }
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
        return alloc_zeros::<f64>(m * n, device);
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

    // Try cuBLAS DGEMM.
    match CudaBlas::new(device.stream().clone()) {
        Ok(blas) => {
            let mut c = alloc_zeros::<f64>(m * n, device)?;

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
        Err(_) => {
            eprintln!(
                "ferrotorch-gpu: cuBLAS handle creation failed, \
                 falling back to CPU matmul for [{m}x{k}] @ [{k}x{n}]"
            );
            cpu_matmul_fallback_f64(a, b, m, k, n, device)
        }
    }
}

// ---------------------------------------------------------------------------
// CPU fallback implementations
// ---------------------------------------------------------------------------

/// CPU fallback: copy A and B to host, multiply with triple-loop, copy back.
#[cfg(feature = "cuda")]
fn cpu_matmul_fallback_f32(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let a_host = gpu_to_cpu(a, device)?;
    let b_host = gpu_to_cpu(b, device)?;
    let c_host = cpu_matmul_naive(&a_host, &b_host, m, k, n);
    cpu_to_gpu(&c_host, device)
}

/// CPU fallback for f64.
#[cfg(feature = "cuda")]
fn cpu_matmul_fallback_f64(
    a: &CudaBuffer<f64>,
    b: &CudaBuffer<f64>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    let a_host = gpu_to_cpu(a, device)?;
    let b_host = gpu_to_cpu(b, device)?;
    let c_host = cpu_matmul_naive(&a_host, &b_host, m, k, n);
    cpu_to_gpu(&c_host, device)
}

/// Naive row-major matrix multiply on the CPU.
///
/// `a` is `[m, k]`, `b` is `[k, n]`, result is `[m, n]`.
/// Used only as a fallback when cuBLAS is unavailable.
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
}
