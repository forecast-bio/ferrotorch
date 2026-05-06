//! Permanent regression sentinel for #785:
//! `ProdBackward` on a CUDA tensor returned `Err(NotImplementedOnCuda)`,
//! making `prod` non-differentiable on GPU even though the forward
//! kernel was already on-device.
//!
//! Pre-fix: `gpu_prod_backward_{f32,f64}` did not exist; the trait
//! method was missing; `ProdBackward::backward` returned `Err`.
//! Post-fix: the kernel implements the prefix-suffix formulation
//! `grad_input[i] = grad_output * prod(input[j != i])`. This file
//! covers the three zero-handling cases architect-mandated:
//! - no zero (`[1, 2, 3, 4]`): grad = [24, 12, 8, 6]
//! - one zero (`[1, 0, 2, 3]`): grad[zero] = 6, others = 0
//! - two zeros (`[1, 0, 0, 3]`): grad = [0, 0, 0, 0]

#![cfg(feature = "cuda")]

use ferrotorch_gpu::{GpuDevice, cpu_to_gpu, gpu_to_cpu, init_cuda_backend, kernels};
use std::sync::Once;

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn run_f32(input: &[f32], grad_out: f32) -> Vec<f32> {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let gpu_in = cpu_to_gpu(input, &device).expect("upload input");
    let gpu_go = cpu_to_gpu(&[grad_out], &device).expect("upload grad_out");
    let gpu_grad =
        kernels::gpu_prod_backward_f32(&gpu_in, &gpu_go, &device).expect("gpu_prod_backward_f32");
    gpu_to_cpu(&gpu_grad, &device).expect("readback")
}

fn run_f64(input: &[f64], grad_out: f64) -> Vec<f64> {
    ensure_cuda();
    let device = GpuDevice::new(0).expect("CUDA device 0");
    let gpu_in = cpu_to_gpu(input, &device).expect("upload input");
    let gpu_go = cpu_to_gpu(&[grad_out], &device).expect("upload grad_out");
    let gpu_grad =
        kernels::gpu_prod_backward_f64(&gpu_in, &gpu_go, &device).expect("gpu_prod_backward_f64");
    gpu_to_cpu(&gpu_grad, &device).expect("readback")
}

#[test]
fn prod_backward_f32_no_zero() {
    // input = [1, 2, 3, 4], total = 24, grad_in[i] = 24 / input[i]
    let out = run_f32(&[1.0, 2.0, 3.0, 4.0], 1.0);
    let exp = [24.0_f32, 12.0, 8.0, 6.0];
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "f32 no-zero: idx {i} got {a}, want {e} (full: {out:?})"
        );
    }
}

#[test]
fn prod_backward_f32_single_zero() {
    // input = [1, 0, 2, 3]; only the zero position has non-zero grad
    // (= prod of remaining = 6). Others have a zero in their suffix
    // or prefix and so get 0.
    let out = run_f32(&[1.0, 0.0, 2.0, 3.0], 1.0);
    let exp = [0.0_f32, 6.0, 0.0, 0.0];
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "f32 single-zero: idx {i} got {a}, want {e} (full: {out:?})"
        );
    }
}

#[test]
fn prod_backward_f32_multi_zero() {
    // input = [1, 0, 0, 3]; both zero positions have a zero in either
    // their prefix or suffix; non-zero positions also see a zero in
    // their suffix/prefix; everything is zero.
    let out = run_f32(&[1.0, 0.0, 0.0, 3.0], 1.0);
    let exp = [0.0_f32; 4];
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "f32 multi-zero: idx {i} got {a}, want {e} (full: {out:?})"
        );
    }
}

#[test]
fn prod_backward_f32_grad_out_propagates() {
    // grad_output != 1 should scale every output linearly.
    let out = run_f32(&[1.0, 2.0, 3.0, 4.0], 0.5);
    let exp = [12.0_f32, 6.0, 4.0, 3.0];
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-5,
            "f32 grad_out: idx {i} got {a}, want {e} (full: {out:?})"
        );
    }
}

#[test]
fn prod_backward_f64_no_zero() {
    let out = run_f64(&[1.0, 2.0, 3.0, 4.0], 1.0);
    let exp = [24.0_f64, 12.0, 8.0, 6.0];
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-12,
            "f64 no-zero: idx {i} got {a}, want {e} (full: {out:?})"
        );
    }
}

#[test]
fn prod_backward_f64_single_zero() {
    let out = run_f64(&[1.0, 0.0, 2.0, 3.0], 1.0);
    let exp = [0.0_f64, 6.0, 0.0, 0.0];
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-12,
            "f64 single-zero: idx {i} got {a}, want {e} (full: {out:?})"
        );
    }
}

#[test]
fn prod_backward_f64_multi_zero() {
    let out = run_f64(&[1.0, 0.0, 0.0, 3.0], 1.0);
    let exp = [0.0_f64; 4];
    for (i, (a, e)) in out.iter().zip(exp.iter()).enumerate() {
        assert!(
            (a - e).abs() < 1e-12,
            "f64 multi-zero: idx {i} got {a}, want {e} (full: {out:?})"
        );
    }
}
