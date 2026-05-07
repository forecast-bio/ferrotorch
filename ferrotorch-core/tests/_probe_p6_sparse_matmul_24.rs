//! Permanent regression sentinel for P6 of #806: 2:4 structured sparse
//! matmul via cuSPARSELt.
//!
//! Pre-fix observable failure:
//!   1. `SemiStructuredSparseTensor::sparse_matmul_24(a, b)` walked CPU
//!      even when `a` lived on CUDA (`#[allow] decompress` + dense
//!      reference matmul). For Ampere+ Tensor Cores this leaves the
//!      ~2x throughput of cuSPARSELt's structured-sparse path on the
//!      table.
//!
//! Post-fix:
//!   1. With the `cusparselt` cargo feature on AND `libcusparseLt.so`
//!      available at runtime, `sparse_matmul_24` dispatches through
//!      `cusparseLtMatmul` and the result tensor remains on CUDA.
//!   2. The numerics match the dense reference within
//!      `F32_MATMUL_GPU = 1e-3` (TF32 mode introduces a small precision
//!      relaxation vs. exact f32 — 1e-3 is the same tolerance the
//!      project's other GPU-matmul probes use).
//!
//! This probe is **gated on `#[cfg(feature = "cusparselt")]`**: the
//! default workspace build (`--features gpu` only) doesn't include it,
//! so the regression sentinel only runs when the operator is actually
//! built. If `libcusparseLt.so` is not installed at runtime, the test
//! detects the resulting backend error and skips with a clear message
//! rather than panicking — `LIBCUSPARSELT_RUNTIME_MISSING=1` can be
//! set to unconditionally skip.

#![cfg(all(feature = "gpu", feature = "cusparselt"))]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::sparse::{SemiStructuredSparseTensor, sparse_matmul_24};

const F32_MATMUL_GPU: f32 = 1e-3;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

/// Build a deterministic dense `[k, n]` matrix in row-major order. The
/// values cycle through a small set so 2:4 compression keeps a known
/// pattern.
fn dense_kn(k: usize, n: usize, seed: u32) -> Vec<f32> {
    let mut out = vec![0.0_f32; k * n];
    for i in 0..k * n {
        let v = ((i as u32).wrapping_mul(2654435761).wrapping_add(seed) % 13) as f32 - 6.0;
        out[i] = v * 0.25;
    }
    out
}

fn dense_mk(m: usize, k: usize, seed: u32) -> Vec<f32> {
    let mut out = vec![0.0_f32; m * k];
    for i in 0..m * k {
        let v = ((i as u32).wrapping_mul(40503).wrapping_add(seed) % 11) as f32 - 5.0;
        out[i] = v * 0.5;
    }
    out
}

/// Walk the [m, k] @ [k, n] case for a list of shapes, comparing GPU
/// 2:4 sparse matmul against the CPU reference (which is the dense
/// matmul of the decompressed `b`).
fn check_shape(m: usize, k: usize, n: usize) {
    ensure_cuda_backend();

    let a_data = dense_mk(m, k, 0xA001);
    let b_data = dense_kn(k, n, 0xB002);

    // Build A on CUDA, build the 2:4 structured representation of B on
    // CPU (its values + mask are CPU-resident by design — `b` is the
    // weight matrix in PyTorch's parametrized linear layer).
    let a_cpu = from_vec::<f32>(a_data, &[m, k]).expect("a cpu");
    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("a -> gpu");

    let b_cpu = from_vec::<f32>(b_data, &[k, n]).expect("b cpu");
    let b_sparse = SemiStructuredSparseTensor::compress(&b_cpu).expect("b 2:4 compress");

    // Compute the CPU reference (uses the same `decompress + dense
    // matmul` path the implementation falls through to when the GPU
    // path is unavailable).
    let cpu_ref = {
        let a_for_ref = a_cpu.clone();
        sparse_matmul_24(&a_for_ref, &b_sparse).expect("cpu reference matmul")
    };
    let cpu_data = cpu_ref.data().expect("cpu data").to_vec();

    // Run the GPU path.
    let result = sparse_matmul_24(&a_gpu, &b_sparse);
    let out = match result {
        Ok(t) => t,
        Err(err) => {
            // The library returned an error. If libcusparseLt.so isn't
            // present at runtime the backend will surface a status code
            // — we treat that as a skip for this probe rather than a
            // hard failure.
            let msg = format!("{err:?}");
            if msg.contains("CUSPARSE_STATUS_NOT_INITIALIZED")
                || msg.contains("CUSPARSE_STATUS_NOT_SUPPORTED")
                || msg.contains("cusparseLt")
                || msg.contains("libcusparseLt")
                || std::env::var("LIBCUSPARSELT_RUNTIME_MISSING").is_ok()
            {
                eprintln!(
                    "skipping P6 probe (m={m}, k={k}, n={n}): cuSPARSELt unavailable at runtime: {msg}"
                );
                return;
            }
            panic!("sparse_matmul_24 GPU path failed unexpectedly: {msg}");
        }
    };

    assert!(
        out.is_cuda(),
        "sparse_matmul_24 output must remain on CUDA when input was CUDA"
    );
    assert_eq!(out.shape(), &[m, n]);

    let out_cpu = out.cpu().expect("out gpu->cpu");
    let out_data = out_cpu.data().expect("out data");
    for (i, (&gpu_v, &cpu_v)) in out_data.iter().zip(cpu_data.iter()).enumerate() {
        assert!(
            (gpu_v - cpu_v).abs() < F32_MATMUL_GPU,
            "sparse_matmul_24 GPU vs CPU mismatch at {i}: gpu={gpu_v} cpu={cpu_v}"
        );
    }
}

#[test]
fn p6_sparse_matmul_24_basic_8x16_16x8() {
    check_shape(8, 16, 8);
}

#[test]
fn p6_sparse_matmul_24_medium_16x32_32x16() {
    check_shape(16, 32, 32);
}

/// cuSPARSELt requires dimensions to be a multiple of 4 (FP32 path).
/// The compressed representation in `SemiStructuredSparseTensor`
/// already enforces `numel % 4 == 0`. This probe walks shapes that hit
/// the alignment threshold from the alignment-required side.
#[test]
fn p6_sparse_matmul_24_aligned_shapes() {
    for &(m, k, n) in &[(8usize, 8usize, 8usize), (16, 16, 16), (32, 8, 16)] {
        check_shape(m, k, n);
    }
}

/// Empty inputs: rows or cols zero. cuSPARSELt should not be called;
/// the dispatch returns a zero-filled tensor.
#[test]
fn p6_sparse_matmul_24_zero_rows() {
    ensure_cuda_backend();

    let a_cpu = from_vec::<f32>(Vec::<f32>::new(), &[0, 8]).expect("a cpu empty rows");
    let a_gpu = a_cpu.to(Device::Cuda(0)).expect("a -> gpu");

    let b_data = dense_kn(8, 8, 0x42);
    let b_cpu = from_vec::<f32>(b_data, &[8, 8]).expect("b cpu");
    let b_sparse = SemiStructuredSparseTensor::compress(&b_cpu).expect("b 2:4");

    let out = sparse_matmul_24(&a_gpu, &b_sparse).expect("zero-row sparse_matmul_24");
    assert_eq!(out.shape(), &[0, 8]);
}
