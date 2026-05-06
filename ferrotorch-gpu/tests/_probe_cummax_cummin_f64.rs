//! Permanent regression sentinel for #787:
//! `cummax_f64_kernel` / `cummin_f64_kernel` PTX returns subnormal
//! denormals (~1e-310) instead of running max/min values.
//!
//! Pre-fix: the f32 PTX template initialises the running max/min via
//! `mov.b32 %acc, 0xFF800000;` (-inf as f32 bit-pattern). The converter
//! `ptx_f32_to_f64` rewrites the opcode `mov.b32 → mov.b64` but does
//! NOT rewrite the literal `0xFF800000` to the f64 -inf bit-pattern
//! `0xFFF0000000000000`. The kernel therefore loads the f32 `-inf`
//! bit pattern as the *low 32 bits* of the f64 register, leaving the
//! high 32 bits zero — that bit pattern interpreted as f64 is a tiny
//! positive denormal (~1e-310), so every input value compares greater
//! than the seed, and the output is a copy of the input bit-pattern in
//! the low half (still tiny denormals after the f64 max).
//!
//! Post-fix: the converter promotes the seed literal as well, the
//! kernel initialises `%acc` to f64 ±inf, and the output values match
//! the host-side reference within `F64_REDUCTION` tolerance (1e-12).

#![cfg(feature = "cuda")]

use ferrotorch_gpu::{GpuDevice, cpu_to_gpu, gpu_to_cpu, init_cuda_backend, kernels};
use std::sync::Once;

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

#[test]
fn cummax_f64_simple_increasing() {
    ensure_cuda();
    let input = [1.0_f64, 2.0, 3.0, 4.0];
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let gpu_in = cpu_to_gpu(&input, &device).expect("upload");
    // shape [outer=1, dim=4, inner=1]
    let (vals, _idxs) =
        kernels::gpu_cummax_f64(&gpu_in, 1, 4, 1, &device).expect("gpu_cummax_f64");
    let out = gpu_to_cpu(&vals, &device).expect("readback");
    let expected = [1.0_f64, 2.0, 3.0, 4.0];
    for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-12,
            "cummax_f64 mismatch at i={i}: got {a:?}, want {e:?} (full output: {out:?})",
        );
    }
}

#[test]
fn cummax_f64_with_negatives() {
    ensure_cuda();
    let input = [-3.0_f64, 1.0, -5.0, 2.0, 0.5];
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let gpu_in = cpu_to_gpu(&input, &device).expect("upload");
    let (vals, _idxs) =
        kernels::gpu_cummax_f64(&gpu_in, 1, 5, 1, &device).expect("gpu_cummax_f64");
    let out = gpu_to_cpu(&vals, &device).expect("readback");
    let expected = [-3.0_f64, 1.0, 1.0, 2.0, 2.0];
    for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-12,
            "cummax_f64 mismatch at i={i}: got {a:?}, want {e:?} (full output: {out:?})",
        );
    }
}

#[test]
fn cummin_f64_simple_decreasing() {
    ensure_cuda();
    let input = [4.0_f64, 3.0, 2.0, 1.0];
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let gpu_in = cpu_to_gpu(&input, &device).expect("upload");
    let (vals, _idxs) =
        kernels::gpu_cummin_f64(&gpu_in, 1, 4, 1, &device).expect("gpu_cummin_f64");
    let out = gpu_to_cpu(&vals, &device).expect("readback");
    let expected = [4.0_f64, 3.0, 2.0, 1.0];
    for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-12,
            "cummin_f64 mismatch at i={i}: got {a:?}, want {e:?} (full output: {out:?})",
        );
    }
}

#[test]
fn cummin_f64_with_positives() {
    ensure_cuda();
    let input = [3.0_f64, -1.0, 5.0, -2.0, 0.5];
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let gpu_in = cpu_to_gpu(&input, &device).expect("upload");
    let (vals, _idxs) =
        kernels::gpu_cummin_f64(&gpu_in, 1, 5, 1, &device).expect("gpu_cummin_f64");
    let out = gpu_to_cpu(&vals, &device).expect("readback");
    let expected = [3.0_f64, -1.0, -1.0, -2.0, -2.0];
    for (i, (a, e)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-12,
            "cummin_f64 mismatch at i={i}: got {a:?}, want {e:?} (full output: {out:?})",
        );
    }
}
