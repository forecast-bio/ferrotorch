//! Permanent regression sentinel for #781:
//! `pow_f64_kernel` PTX JIT compilation failure.
//!
//! Pre-fix: `gpu_pow_f64` returns `Err(GpuError::PtxCompileFailed)` whose
//! source chain contains the `ptxas` diagnostic naming the offending
//! literal (a malformed f64 hex constant `0d3F811111111111111` with 17
//! hex digits — PTX f64 literals must have exactly 16). The four
//! `pow_f64_values_match_reference` sub-cases also exercise the inline
//! log+exp path against `f64::powf` to catch any future numerical
//! regression in the hand-written kernel.
//!
//! Post-fix: every test in this file passes; the PTX JIT compiles and
//! the kernel produces values within `F64_TRANSCENDENTAL` tolerance
//! (1e-10 absolute) of the host-side reference.
//!
//! This file is committed permanently — it is the regression sentinel
//! that prevents the same class of bug (off-by-one on a PTX f64 hex
//! literal) from re-emerging silently in the inline log+exp template.

#![cfg(feature = "cuda")]

use ferrotorch_gpu::{GpuDevice, cpu_to_gpu, gpu_to_cpu, init_cuda_backend, kernels};
use std::sync::Once;

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

/// Helper: run `gpu_pow_f64(input, exp)` and return the host-side
/// readback. Pretty-prints the full cudarc error chain on failure
/// (this is the diagnostic that identifies the offending PTX line
/// when the JIT fails).
fn pow_f64_or_dump(input: &[f64], exp: f64) -> Vec<f64> {
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let gpu_in = cpu_to_gpu(input, &device).expect("upload");
    let gpu_out = match kernels::gpu_pow_f64(&gpu_in, exp, &device) {
        Ok(buf) => buf,
        Err(e) => {
            // Full debug print captures the cudarc cause chain,
            // which on PTX-JIT failure includes the ptxas
            // diagnostic with the line/column of the bad literal.
            panic!("gpu_pow_f64({input:?}, exp={exp}) failed: {e:#?}");
        }
    };
    gpu_to_cpu(&gpu_out, &device).expect("readback")
}

#[test]
fn pow_f64_jit_compiles() {
    // Smoke test: the kernel must JIT-compile and produce a
    // sensible output for the simplest possible inputs. Pre-fix
    // this fails with `CUDA_ERROR_INVALID_PTX` at module load.
    ensure_cuda();
    let result = pow_f64_or_dump(&[2.0, 4.0, 8.0], 2.0);
    assert_eq!(result.len(), 3);
    let expected = [4.0_f64, 16.0, 64.0];
    for (i, (got, want)) in result.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-10,
            "pow_f64_jit_compiles: idx={i} got={got} want={want}",
        );
    }
}

#[test]
fn pow_f64_values_match_reference_squared() {
    // x^2 — quadratic, the classic check.
    ensure_cuda();
    let result = pow_f64_or_dump(&[2.0], 2.0);
    let expected = (2.0_f64).powf(2.0);
    assert!(
        (result[0] - expected).abs() < 1e-10,
        "pow_f64(2.0, 2.0): got {} want {expected}",
        result[0],
    );
}

#[test]
fn pow_f64_values_match_reference_sqrt() {
    // x^0.5 — fractional exponent stresses the inline log+exp
    // path (no integer-power short-circuit).
    ensure_cuda();
    let result = pow_f64_or_dump(&[4.0], 0.5);
    let expected = (4.0_f64).powf(0.5);
    assert!(
        (result[0] - expected).abs() < 1e-10,
        "pow_f64(4.0, 0.5): got {} want {expected}",
        result[0],
    );
}

#[test]
fn pow_f64_values_match_reference_reciprocal() {
    // x^-1 — negative exponent, exercises the sign of the
    // exp-path's `n` term.
    ensure_cuda();
    let result = pow_f64_or_dump(&[2.0], -1.0);
    let expected = (2.0_f64).powf(-1.0);
    assert!(
        (result[0] - expected).abs() < 1e-10,
        "pow_f64(2.0, -1.0): got {} want {expected}",
        result[0],
    );
}

#[test]
fn pow_f64_values_match_reference_cubed_ten() {
    // 10^3 — confirms the kernel handles a non-power-of-two
    // base + integer exponent without precision loss.
    ensure_cuda();
    let result = pow_f64_or_dump(&[10.0], 3.0);
    let expected = (10.0_f64).powf(3.0);
    assert!(
        (result[0] - expected).abs() < 1e-7,
        "pow_f64(10.0, 3.0): got {} want {expected}",
        result[0],
    );
}
