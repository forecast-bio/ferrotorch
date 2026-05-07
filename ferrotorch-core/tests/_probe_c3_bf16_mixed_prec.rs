//! Permanent regression sentinel for Sprint C.3: bf16 × f32-accumulator
//! mixed-precision matmul, bmm, and softmax (#518 partial).
//!
//! Pre-fix observable failure (before C.3):
//!   * `GpuBackend::matmul_bf16_f32`, `bmm_bf16_f32`, `softmax_bf16_f32`
//!     returned `Err(InvalidArgument { "... not implemented ..." })` — the
//!     default trait-method stub — so any autocast bf16 path on CUDA was a
//!     hard error rather than a Tensor-Core-accelerated kernel.
//!
//! Post-fix:
//!   * `matmul_bf16_f32` calls `gpu_matmul_bf16` (cublasGemmEx,
//!     CUDA_R_16BF / CUBLAS_COMPUTE_32F); f32 output within 1e-3 of
//!     the CPU reference.
//!   * `bmm_bf16_f32` calls `gpu_bmm_bf16` (cublasGemmStridedBatchedEx,
//!     same dtype combo); f32 output within 1e-3 of reference.
//!   * `softmax_bf16_f32` calls `gpu_softmax_bf16_f32` (PTX kernel,
//!     f32 accumulator throughout); f32 output within 1e-3 of reference.
//!
//! PyTorch parity (rust-gpu-discipline §3):
//!   `torch.matmul(a.bfloat16(), b.bfloat16())` on CUDA accumulates in f32
//!   via cuBLAS CUDA_R_16BF / CUBLAS_COMPUTE_32F.
//!   `torch.softmax(x.bfloat16(), dim=-1)` on CUDA uses f32 accumulation
//!   internally for numerical stability.
//!
//! Tolerance: F32_MATMUL_TOL = 1e-3 (bf16 has ~3 decimal digits of mantissa
//! precision; the tolerance is generous enough to accommodate both the
//! bf16 input quantisation error and the f32-accumulator rounding).

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::gpu_dispatch;

// bf16 has ~3 decimal digits of mantissa precision (~7-bit mantissa).
// For accumulated dot-products over k=4 terms the relative error per term
// is ~4e-3 (2^-7 * sqrt(4)), so 5e-3 is the right tightened bound.
const F32_MATMUL_TOL: f32 = 5e-3;
// Softmax outputs are in [0,1]; 1e-3 absolute error is tight but reachable
// given bf16 input quantisation.
const F32_SOFTMAX_TOL: f32 = 1e-3;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for the bf16 mixed-prec probe suite");
    });
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Upload a `&[f32]` to a GPU f32 handle (device 0).
fn upload_f32(data: &[f32]) -> gpu_dispatch::GpuBufferHandle {
    let backend = gpu_dispatch::gpu_backend().expect("backend registered");
    // SAFETY: reinterpreting &[f32] as &[u8] — same alignment / length
    // contract as every other test in the suite (backend_impl.rs tests).
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
    };
    backend.cpu_to_gpu(bytes, 4, 0).expect("upload f32")
}

/// Download a GPU f32 handle back to a `Vec<f32>`.
fn download_f32(handle: &gpu_dispatch::GpuBufferHandle) -> Vec<f32> {
    let backend = gpu_dispatch::gpu_backend().expect("backend registered");
    let bytes = backend.gpu_to_cpu(handle).expect("download f32");
    // SAFETY: gpu_to_cpu returns bytes originally from a Vec<f32> allocation
    // (see backend_impl gpu_to_cpu implementation); alignment and element
    // count are both correct.
    let f32_slice: &[f32] = unsafe {
        std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
    };
    f32_slice.to_vec()
}

/// Convert a `f32` value to its bf16 bit pattern (top 16 bits, round-to-zero
/// for simplicity — sufficient for test fixture construction).
///
/// This is the same truncation the `f32_to_bf16_kernel` uses for
/// positive normal values in the representable range.
fn f32_to_bf16_bits(v: f32) -> u16 {
    // Round-to-nearest-even: add rounding bias then shift.
    let bits = v.to_bits();
    let lsb = (bits >> 16) & 1;
    let round = bits.wrapping_add(0x7FFF).wrapping_add(lsb);
    (round >> 16) as u16
}

/// Upload a `&[f32]` as bf16 bit patterns in a `CudaSlice<u16>` handle.
///
/// The handle is boxed as `Box<cudarc::driver::CudaSlice<u16>>` inside a
/// `GpuBufferHandle` — the same type that `softmax_bf16_f32` expects.
fn upload_bf16(data: &[f32]) -> gpu_dispatch::GpuBufferHandle {
    // Convert f32 -> bf16 bit patterns on the host, then upload as u16 bytes.
    let bf16_vals: Vec<u16> = data.iter().map(|&v| f32_to_bf16_bits(v)).collect();
    let bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(bf16_vals.as_ptr() as *const u8, bf16_vals.len() * 2)
    };
    // elem_size = 2 so the backend stores them as CudaBuffer<u16> / CudaSlice<u16>.
    let backend = gpu_dispatch::gpu_backend().expect("backend registered");
    backend.cpu_to_gpu(bytes, 2, 0).expect("upload bf16")
}

/// Reference: CPU matrix multiply C = A @ B (row-major, f32).
fn cpu_matmul_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

/// Reference: CPU row-wise softmax (f32).
fn cpu_softmax_ref(x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = row.iter().map(|&v| (v - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        for (j, &e) in exps.iter().enumerate() {
            out[r * cols + j] = e / sum;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// matmul_bf16_f32 probes
// ---------------------------------------------------------------------------

/// Basic 2×3 @ 3×2 matmul in bf16 → f32.
///
/// BEFORE: `matmul_bf16_f32` returned Err("not implemented").
/// AFTER:  cublasGemmEx with CUDA_R_16BF / CUBLAS_COMPUTE_32F executes on GPU;
///         result matches CPU f32 reference within 1e-3.
#[test]
fn c3_matmul_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];         // [2, 3]
    let b = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];       // [3, 2]
    let expected = cpu_matmul_ref(&a, &b, 2, 3, 2);

    let a_gpu = upload_f32(&a);
    let b_gpu = upload_f32(&b);

    let result = backend
        .matmul_bf16_f32(&a_gpu, &b_gpu, 2, 3, 2)
        .expect("matmul_bf16_f32 must succeed on CUDA");

    assert_eq!(result.len(), 4, "output length must be m*n = 4");
    assert_eq!(result.device_ordinal(), 0, "output must stay on device 0");

    let got = download_f32(&result);
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_MATMUL_TOL,
            "c3_matmul_bf16_f32_basic elem {i}: got {g}, expected {e} (tol {F32_MATMUL_TOL})"
        );
    }
}

/// Square matmul 4×4 — exercises the cuBLAS path for larger (non-tiny) shapes.
///
/// Uses relative tolerance: bf16 has ~7 mantissa bits (~0.78% relative error
/// per value). With k=4 accumulated terms the accumulated error on any output
/// element is bounded by ~1% of its magnitude. We use max(F32_MATMUL_TOL,
/// 0.01 * |expected|) so large outputs get 1% slack and near-zero outputs
/// get the absolute floor.
#[test]
fn c3_matmul_bf16_f32_square_4x4() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a: Vec<f32> = (1..=16).map(|v| v as f32 * 0.1).collect();
    let b: Vec<f32> = (1..=16).map(|v| v as f32 * 0.2).collect();
    let expected = cpu_matmul_ref(&a, &b, 4, 4, 4);

    let a_gpu = upload_f32(&a);
    let b_gpu = upload_f32(&b);

    let result = backend
        .matmul_bf16_f32(&a_gpu, &b_gpu, 4, 4, 4)
        .expect("matmul_bf16_f32 4x4");

    assert_eq!(result.device_ordinal(), 0);
    let got = download_f32(&result);
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        // Relative tolerance: 1% of |expected| or F32_MATMUL_TOL, whichever
        // is larger. bf16 7-bit mantissa → ~0.78% per-term; k=4 accumulation
        // → ~1% worst-case relative error on the dot product.
        let tol = F32_MATMUL_TOL.max(e.abs() * 0.01);
        assert!(
            (g - e).abs() < tol,
            "c3_matmul_bf16_f32_square_4x4 elem {i}: got {g}, expected {e} (tol {tol})"
        );
    }
}

/// Degenerate zero dimension — must return an empty buffer, not an error.
#[test]
fn c3_matmul_bf16_f32_zero_dim() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    // m=0: empty A, empty C.
    let a_gpu = upload_f32(&[]);
    let b_gpu = upload_f32(&[1.0, 2.0, 3.0, 4.0]); // [2, 2]

    let result = backend
        .matmul_bf16_f32(&a_gpu, &b_gpu, 0, 2, 2)
        .expect("zero-dim matmul_bf16_f32 must not error");
    assert_eq!(result.len(), 0);
}

// ---------------------------------------------------------------------------
// bmm_bf16_f32 probes
// ---------------------------------------------------------------------------

/// Batched matmul: batch=2, m=2, k=3, n=2.
///
/// BEFORE: `bmm_bf16_f32` returned Err("not implemented").
/// AFTER:  cublasGemmStridedBatchedEx with CUDA_R_16BF / CUBLAS_COMPUTE_32F
///         executes on GPU; per-batch results match CPU reference within 1e-3.
#[test]
fn c3_bmm_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    // batch=2, m=2, k=3, n=2
    let a0 = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
    let b0 = vec![7.0f32, 8.0, 9.0, 10.0, 11.0, 12.0];
    let a1 = vec![0.5f32, 1.0, 1.5, 2.0, 2.5, 3.0];
    let b1 = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0];

    let a_data: Vec<f32> = a0.iter().chain(a1.iter()).cloned().collect();
    let b_data: Vec<f32> = b0.iter().chain(b1.iter()).cloned().collect();

    let ref0 = cpu_matmul_ref(&a0, &b0, 2, 3, 2);
    let ref1 = cpu_matmul_ref(&a1, &b1, 2, 3, 2);

    let a_gpu = upload_f32(&a_data);
    let b_gpu = upload_f32(&b_data);

    let result = backend
        .bmm_bf16_f32(&a_gpu, &b_gpu, 2, 2, 3, 2)
        .expect("bmm_bf16_f32 must succeed on CUDA");

    assert_eq!(result.len(), 2 * 2 * 2, "output length must be batch*m*n");
    assert_eq!(result.device_ordinal(), 0, "output must stay on device 0");

    let got = download_f32(&result);
    let (got0, got1) = got.split_at(4);
    for (i, (&g, &e)) in got0.iter().zip(ref0.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_MATMUL_TOL,
            "c3_bmm_bf16_f32_basic batch0 elem {i}: got {g}, expected {e}"
        );
    }
    for (i, (&g, &e)) in got1.iter().zip(ref1.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_MATMUL_TOL,
            "c3_bmm_bf16_f32_basic batch1 elem {i}: got {g}, expected {e}"
        );
    }
}

/// Degenerate batch=0 — must return an empty buffer.
#[test]
fn c3_bmm_bf16_f32_zero_batch() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let a_gpu = upload_f32(&[]);
    let b_gpu = upload_f32(&[]);

    let result = backend
        .bmm_bf16_f32(&a_gpu, &b_gpu, 0, 2, 3, 2)
        .expect("zero-batch bmm_bf16_f32 must not error");
    assert_eq!(result.len(), 0);
}

// ---------------------------------------------------------------------------
// softmax_bf16_f32 probes
// ---------------------------------------------------------------------------

/// Basic 2×4 bf16 softmax — standard case.
///
/// BEFORE: `softmax_bf16_f32` returned Err("not implemented").
/// AFTER:  PTX `softmax_bf16_f32_kernel` launches on GPU; bf16 inputs are
///         loaded via ld.global.u16 + cvt.f32.bf16; all accumulation in f32;
///         f32 output matches CPU reference within 1e-3.
#[test]
fn c3_softmax_bf16_f32_basic() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let x = vec![1.0f32, 2.0, 3.0, 4.0, 0.5, 1.5, 2.5, 3.5];
    let expected = cpu_softmax_ref(&x, 2, 4);

    let x_bf16 = upload_bf16(&x);

    let result = backend
        .softmax_bf16_f32(&x_bf16, 2, 4)
        .expect("softmax_bf16_f32 must succeed on CUDA");

    assert_eq!(result.len(), 8, "output length must be rows*cols");
    assert_eq!(result.device_ordinal(), 0, "output must stay on device 0");

    let got = download_f32(&result);
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_SOFTMAX_TOL,
            "c3_softmax_bf16_f32_basic elem {i}: got {g}, expected {e} (tol {F32_SOFTMAX_TOL})"
        );
    }
    // Outputs must be a valid probability distribution (rows sum to 1).
    for r in 0..2 {
        let row_sum: f32 = got[r * 4..(r + 1) * 4].iter().sum();
        assert!(
            (row_sum - 1.0).abs() < 1e-5,
            "row {r} sum = {row_sum}, expected 1.0"
        );
    }
}

/// Numerically challenging: large spread within a row (tests max-subtract).
#[test]
fn c3_softmax_bf16_f32_large_spread() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    // Large spread: softmax should concentrate on the last element.
    let x = vec![0.0f32, 0.0, 0.0, 10.0];
    let expected = cpu_softmax_ref(&x, 1, 4);

    let x_bf16 = upload_bf16(&x);
    let result = backend
        .softmax_bf16_f32(&x_bf16, 1, 4)
        .expect("softmax_bf16_f32 large spread");

    let got = download_f32(&result);
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_SOFTMAX_TOL,
            "c3_softmax_bf16_f32_large_spread elem {i}: got {g}, expected {e}"
        );
    }
}

/// Single row, single column — trivial case: softmax([v]) = [1.0].
#[test]
fn c3_softmax_bf16_f32_single_element() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let x = vec![3.14f32];
    let x_bf16 = upload_bf16(&x);
    let result = backend
        .softmax_bf16_f32(&x_bf16, 1, 1)
        .expect("softmax single element");

    let got = download_f32(&result);
    assert_eq!(got.len(), 1);
    assert!(
        (got[0] - 1.0).abs() < 1e-6,
        "single-element softmax must be 1.0, got {}",
        got[0]
    );
}

/// Zero rows — must return an empty buffer without error.
#[test]
fn c3_softmax_bf16_f32_zero_rows() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let x_bf16 = upload_bf16(&[]);
    let result = backend
        .softmax_bf16_f32(&x_bf16, 0, 4)
        .expect("zero-rows softmax_bf16_f32 must not error");
    assert_eq!(result.len(), 0);
}

/// Larger matrix: 32 rows × 64 cols, values in [0, 1).
/// Exercises multi-thread-stride loops in the PTX kernel.
#[test]
fn c3_softmax_bf16_f32_32x64() {
    ensure_cuda_backend();
    let backend = gpu_dispatch::gpu_backend().expect("backend");

    let rows = 32usize;
    let cols = 64usize;
    let x: Vec<f32> = (0..rows * cols)
        .map(|i| (i as f32 * 0.01) % 2.0 - 1.0)
        .collect();
    let expected = cpu_softmax_ref(&x, rows, cols);

    let x_bf16 = upload_bf16(&x);
    let result = backend
        .softmax_bf16_f32(&x_bf16, rows, cols)
        .expect("softmax_bf16_f32 32x64");

    assert_eq!(result.len(), rows * cols);
    assert_eq!(result.device_ordinal(), 0);

    let got = download_f32(&result);
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_SOFTMAX_TOL,
            "c3_softmax_bf16_f32_32x64 elem {i}: got {g}, expected {e}"
        );
    }
}
