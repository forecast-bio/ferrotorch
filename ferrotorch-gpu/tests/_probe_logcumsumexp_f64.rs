//! Probe + permanent regression sentinel for #786:
//! `logcumsumexp_f64_kernel` returns ~709.69 (= ln(f64::MAX)) garbage
//! for small/medium inputs.
//!
//! Pre-fix evidence: the conformance fixture for logcumsumexp f64 hits
//! tolerance failures at the gpu reduction tests with characteristic
//! "709.69" outputs — the architect's hint that an `exp(x)` is
//! overflowing inside the inline polynomial sweep. The kernel already
//! includes the #783-cluster polynomial improvements (half-step
//! argument reduction + degree-7 odd Horner + 2-double Cody-Waite
//! ln2). This probe re-tests the small-N, medium-N, and stability
//! cases with `F64_TRANSCENDENTAL = 1e-10` to diagnose where the
//! kernel still drifts past tolerance.
//!
//! Post-fix: every test in this file passes within 1e-10 absolute
//! tolerance of the host-side `f64::ln(f64::exp(...).sum())`
//! reference (computed in the same numerically-stable max-subtract
//! form as the kernel).

#![cfg(feature = "cuda")]

use ferrotorch_gpu::{GpuDevice, cpu_to_gpu, gpu_to_cpu, init_cuda_backend, kernels};
use std::sync::Once;

const TOL: f64 = 1e-10;

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Host-side reference: numerically stable logcumsumexp along a 1-D
/// axis. Mirrors the kernel's max-subtract formulation so the
/// reference exercises the same numerics as the kernel template (any
/// disagreement is therefore the kernel's bug, not host-vs-kernel
/// algorithmic drift).
fn cpu_logcumsumexp(input: &[f64]) -> Vec<f64> {
    let n = input.len();
    let mut out = vec![0.0_f64; n];
    let mut acc = f64::NEG_INFINITY;
    for (i, &x) in input.iter().enumerate() {
        let m = acc.max(x);
        let s = (acc - m).exp() + (x - m).exp();
        acc = m + s.ln();
        out[i] = acc;
    }
    out
}

fn run_logcumsumexp_f64(input: &[f64]) -> Vec<f64> {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let gpu_in = cpu_to_gpu(input, &device).expect("upload");
    // shape [outer=1, dim=N, inner=1]
    let n = input.len();
    let gpu_out = match kernels::gpu_logcumsumexp_f64(&gpu_in, 1, n, 1, &device) {
        Ok(buf) => buf,
        Err(e) => panic!("gpu_logcumsumexp_f64 failed: {e:#?}"),
    };
    gpu_to_cpu(&gpu_out, &device).expect("readback")
}

#[test]
fn logcumsumexp_f64_small_n() {
    let input = vec![1.0_f64, 2.0, 3.0];
    let out = run_logcumsumexp_f64(&input);
    let exp = cpu_logcumsumexp(&input);
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < TOL,
            "small N idx={i}: got {a}, want {e}, diff {} (full: {out:?} vs {exp:?})",
            (a - e).abs(),
        );
    }
}

#[test]
fn logcumsumexp_f64_medium_n() {
    // Architect-specified: [1.0, 2.0, ..., 10.0] — the medium-N test
    // that determines whether Path A reaches `F64_TRANSCENDENTAL = 1e-10`.
    let input: Vec<f64> = (1..=10).map(|i| i as f64).collect();
    let out = run_logcumsumexp_f64(&input);
    let exp = cpu_logcumsumexp(&input);
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < TOL,
            "medium N idx={i}: got {a}, want {e}, diff {} (full: {out:?} vs {exp:?})",
            (a - e).abs(),
        );
    }
}

#[test]
fn logcumsumexp_f64_numerical_stability() {
    // [100.0; 5] — should give 100.0 + ln(k+1) per output index.
    // Demonstrates the max-subtract formulation works (raw exp(100)
    // overflows f64 if not subtracted out).
    let input = vec![100.0_f64; 5];
    let out = run_logcumsumexp_f64(&input);
    let exp = cpu_logcumsumexp(&input);
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < TOL,
            "stability idx={i}: got {a}, want {e}, diff {} (full: {out:?} vs {exp:?})",
            (a - e).abs(),
        );
    }
}

#[test]
fn logcumsumexp_f64_negatives() {
    let input = vec![-3.0_f64, 2.0, -1.0, 0.5];
    let out = run_logcumsumexp_f64(&input);
    let exp = cpu_logcumsumexp(&input);
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < TOL,
            "neg idx={i}: got {a}, want {e}, diff {} (full: {out:?} vs {exp:?})",
            (a - e).abs(),
        );
    }
}
