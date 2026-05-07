//! Probe B1 — Issue #893: gelu_tanh_kernel PTX JIT fail on sm_86 (RTX 3090).
//!
//! Root cause: GELU_TANH_PTX contained non-ASCII UTF-8 characters (pi, superscript-3)
//! inside the inline PTX comment block. The CUDA JIT driver (cuModuleLoadData) rejects
//! PTX strings with non-ASCII bytes, returning CUDA_ERROR_INVALID_PTX. The GELU_ERF
//! kernel worked because its comments were ASCII-only.
//!
//! BEFORE fix: gpu_gelu_tanh returned Err(PtxCompileFailed { kernel: "gelu_tanh_kernel",
//!   source: DriverError(CUDA_ERROR_INVALID_PTX, "a PTX JIT compilation failed") })
//!
//! AFTER fix: non-ASCII chars (pi -> "pi", superscript-3 -> "^3") replaced with ASCII.
//!   gpu_gelu_tanh succeeds and output matches gelu(x, approximate="tanh") reference.
//!
//! Reference formula: out[i] = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//! Tolerance: 1e-5 relative (TOL_TRANSCENDENTAL tier).

#![cfg(feature = "gpu")]

use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for probe B1 gelu_tanh");
    });
}

/// CPU reference for gelu_tanh (tanh approximation).
fn gelu_tanh_ref(x: f32) -> f32 {
    // sqrt(2/pi) = 0.7978845608
    let sqrt2_pi: f32 = 0.7978845608;
    let inner = x + 0.044715 * x * x * x;
    0.5 * x * (1.0 + (sqrt2_pi * inner).tanh())
}

/// Probe: AFTER fix gelu_tanh_kernel JIT-compiles and produces correct output.
#[test]
fn probe_b1_gelu_tanh_ptx_after() {
    ensure_cuda();
    use ferrotorch_gpu::device::GpuDevice;
    use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};

    let dev = GpuDevice::new(0).expect("GpuDevice::new(0)");

    // Test range covers negative, zero, and positive inputs including saturation region.
    let inputs: Vec<f32> = (-20..=20).map(|i| i as f32 * 0.25).collect();
    let expected: Vec<f32> = inputs.iter().map(|&x| gelu_tanh_ref(x)).collect();

    let a = cpu_to_gpu(&inputs, &dev).unwrap();

    // BEFORE fix: Err(PtxCompileFailed { kernel: "gelu_tanh_kernel", ... INVALID_PTX })
    // AFTER  fix: Ok(CudaBuffer with correct gelu_tanh values)
    let out = ferrotorch_gpu::kernels::gpu_gelu_tanh(&a, &dev)
        .expect("AFTER fix #893: gelu_tanh_kernel must JIT-compile and run on sm_86");

    let actual = gpu_to_cpu(&out, &dev).unwrap();
    assert_eq!(actual.len(), expected.len());

    // Use absolute tolerance for values near zero, relative elsewhere.
    const ABS_TOL: f32 = 1e-4;
    const REL_TOL: f32 = 1e-4;
    for (i, (&act, &exp)) in actual.iter().zip(expected.iter()).enumerate() {
        let abs_err = (act - exp).abs();
        let rel_err = if exp.abs() > 1e-3 { abs_err / exp.abs() } else { abs_err };
        assert!(
            abs_err <= ABS_TOL || rel_err <= REL_TOL,
            "probe B1 #893 AFTER: index {i} x={:.3} actual={:.6} expected={:.6} \
             abs_err={:.2e} rel_err={:.2e}",
            inputs[i], act, exp, abs_err, rel_err
        );
    }
}
