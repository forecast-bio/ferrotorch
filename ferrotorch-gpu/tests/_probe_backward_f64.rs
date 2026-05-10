//! Permanent regression sentinel for #782: f64 GPU backward kernels.
//!
//! Pre-fix: `abs_backward_f64` GPU op returned the trait-default
//! "not yet implemented" error, breaking f64 autograd parity for ops
//! whose backward path is dispatched on-device. The audit listed 13
//! `*_backward_f64` ops as missing under #782; on physical inspection
//! 12 of them already had concrete kernel implementations and only
//! `abs_backward_f64` was genuinely missing — the trait stubs in
//! `gpu_dispatch.rs` are fallbacks that the backend impl supersedes.
//!
//! Post-fix: every `*_backward_f64` op below dispatches to a real
//! GPU kernel and matches the CPU reference within `F64_GPU` (1e-9
//! absolute) for elementwise ops and `F64_REDUCTION` (1e-7 absolute)
//! for the per-row reduction ops (softmax / log_softmax / layernorm
//! / rmsnorm).
//!
//! Each sub-test:
//!   1. Constructs small CPU f64 input + grad_output buffers.
//!   2. Computes a CPU reference via the closed-form gradient.
//!   3. Calls the GPU backward via the corresponding `kernels::*`
//!      entry point.
//!   4. Asserts the readback matches the reference within tolerance.
//!
//! This file is committed permanently — it's the regression sentinel
//! that prevents any of the 13 backward kernels from silently
//! degrading or vanishing.

#![cfg(feature = "cuda")]

use ferrotorch_gpu::{GpuDevice, cpu_to_gpu, gpu_to_cpu, init_cuda_backend, kernels};
use std::sync::Once;

const F64_ELEMENTWISE: f64 = 1e-9;
const F64_REDUCTION: f64 = 1e-7;

fn ensure_cuda() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn dev() -> GpuDevice {
    GpuDevice::new(0).expect("CUDA device 0")
}

fn assert_close(label: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let abs_err = (g - w).abs();
        assert!(
            abs_err < tol,
            "{label}: idx={i} got={g} want={w} abs_err={abs_err:.3e} tol={tol:.0e}",
        );
    }
}

// ---------------------------------------------------------------------------
// 1. relu_backward_f64
// ---------------------------------------------------------------------------

#[test]
fn relu_backward_f64_matches_reference() {
    ensure_cuda();
    let input = [-2.0_f64, -0.5, 0.0, 0.5, 2.0];
    let grad = [1.0_f64, 1.0, 1.0, 1.0, 1.0];
    let want: Vec<f64> = input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| if x > 0.0 { g } else { 0.0 })
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad, &device).expect("upload grad");
    let i_in = cpu_to_gpu(&input, &device).expect("upload input");
    let out = kernels::gpu_relu_backward_f64(&g_in, &i_in, &device).expect("relu_backward_f64");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("relu_backward_f64", &got, &want, F64_ELEMENTWISE);
}

// ---------------------------------------------------------------------------
// 2. abs_backward_f64 — the kernel that was actually missing pre-#782
// ---------------------------------------------------------------------------

#[test]
fn abs_backward_f64_matches_reference() {
    ensure_cuda();
    let input = [-3.0_f64, -1.0, 0.0, 1.0, 3.0];
    let grad = [2.0_f64, 2.0, 2.0, 2.0, 2.0];
    // d/dx |x| = sign(x), with sign(0) = 0 by convention
    let want: Vec<f64> = input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| {
            if x > 0.0 {
                g
            } else if x < 0.0 {
                -g
            } else {
                0.0
            }
        })
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad, &device).expect("upload grad");
    let i_in = cpu_to_gpu(&input, &device).expect("upload input");
    let out = kernels::gpu_abs_backward_f64(&g_in, &i_in, &device).expect("abs_backward_f64");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("abs_backward_f64", &got, &want, F64_ELEMENTWISE);
}

// ---------------------------------------------------------------------------
// 3. sigmoid_backward_f64
// ---------------------------------------------------------------------------

#[test]
fn sigmoid_backward_f64_matches_reference() {
    ensure_cuda();
    // grad * y * (1 - y), where y is the sigmoid output
    let output = [0.1_f64, 0.3, 0.5, 0.7, 0.9];
    let grad = [1.0_f64; 5];
    let want: Vec<f64> = output
        .iter()
        .zip(grad.iter())
        .map(|(&y, &g)| g * y * (1.0 - y))
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad, &device).expect("upload grad");
    let o_in = cpu_to_gpu(&output, &device).expect("upload output");
    let out = kernels::gpu_sigmoid_backward_f64(&g_in, &o_in, &device).expect("sigmoid_bwd");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("sigmoid_backward_f64", &got, &want, F64_ELEMENTWISE);
}

// ---------------------------------------------------------------------------
// 4. tanh_backward_f64
// ---------------------------------------------------------------------------

#[test]
fn tanh_backward_f64_matches_reference() {
    ensure_cuda();
    // grad * (1 - y^2), where y = tanh(x)
    let output = [-0.9_f64, -0.5, 0.0, 0.5, 0.9];
    let grad = [1.0_f64; 5];
    let want: Vec<f64> = output
        .iter()
        .zip(grad.iter())
        .map(|(&y, &g)| g * (1.0 - y * y))
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad, &device).expect("upload grad");
    let o_in = cpu_to_gpu(&output, &device).expect("upload output");
    let out = kernels::gpu_tanh_backward_f64(&g_in, &o_in, &device).expect("tanh_bwd");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("tanh_backward_f64", &got, &want, F64_ELEMENTWISE);
}

// ---------------------------------------------------------------------------
// 5. gelu_backward_f64 (sigmoid approx)
// ---------------------------------------------------------------------------

#[test]
fn gelu_backward_f64_matches_reference() {
    ensure_cuda();
    // GELU sigmoid approx: y = x * sigmoid(k*x), where the kernel uses
    // `k = f64::from_bits(0x3FFB44E400000000)` ~= 1.7043190002441406 (the f32
    // rounding of 1.702 promoted to f64; see GELU_BACKWARD_F64_PTX).
    // dy/dx = sig + k * x * sig * (1 - sig), where sig = sigmoid(k*x)
    const K: f64 = 1.7043190002441406_f64;
    let input = [-2.0_f64, -0.5, 0.0, 0.5, 2.0];
    let grad = [1.0_f64; 5];
    let want: Vec<f64> = input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| {
            let z = K * x;
            let sig = 1.0 / (1.0 + (-z).exp());
            g * (sig + K * x * sig * (1.0 - sig))
        })
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad, &device).expect("upload grad");
    let i_in = cpu_to_gpu(&input, &device).expect("upload input");
    let out = kernels::gpu_gelu_backward_f64(&g_in, &i_in, &device).expect("gelu_bwd");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("gelu_backward_f64", &got, &want, F64_ELEMENTWISE);
}

// ---------------------------------------------------------------------------
// 6. clamp_backward_f64
// ---------------------------------------------------------------------------

#[test]
fn clamp_backward_f64_matches_reference() {
    ensure_cuda();
    let input = [-2.0_f64, -0.5, 0.5, 1.5, 3.0];
    let grad = [1.0_f64; 5];
    let lo = -1.0_f64;
    let hi = 1.0_f64;
    let want: Vec<f64> = input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| if x >= lo && x <= hi { g } else { 0.0 })
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad, &device).expect("upload grad");
    let i_in = cpu_to_gpu(&input, &device).expect("upload input");
    let out = kernels::gpu_clamp_backward_f64(&g_in, &i_in, lo, hi, &device).expect("clamp_bwd");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("clamp_backward_f64", &got, &want, F64_ELEMENTWISE);
}

// ---------------------------------------------------------------------------
// 7. silu_backward_f64
// ---------------------------------------------------------------------------

#[test]
fn silu_backward_f64_matches_reference() {
    ensure_cuda();
    // SiLU: y = x * sigmoid(x); dy/dx = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
    let input = [-2.0_f64, -0.5, 0.0, 0.5, 2.0];
    let grad = [1.0_f64; 5];
    let want: Vec<f64> = input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| {
            let sig = 1.0 / (1.0 + (-x).exp());
            g * sig * (1.0 + x * (1.0 - sig))
        })
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad, &device).expect("upload grad");
    let i_in = cpu_to_gpu(&input, &device).expect("upload input");
    let out = kernels::gpu_silu_backward_f64(&g_in, &i_in, &device).expect("silu_bwd");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("silu_backward_f64", &got, &want, F64_ELEMENTWISE);
}

// ---------------------------------------------------------------------------
// 8. elu_backward_f64
// ---------------------------------------------------------------------------

#[test]
fn elu_backward_f64_matches_reference() {
    ensure_cuda();
    // ELU: y = x if x>0 else alpha*(exp(x)-1)
    // dy/dx = 1 if x>0 else alpha*exp(x)
    let input = [-2.0_f64, -0.5, 0.0, 0.5, 2.0];
    let grad = [1.0_f64; 5];
    let alpha = 1.0_f64;
    let want: Vec<f64> = input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| if x > 0.0 { g } else { g * alpha * x.exp() })
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad, &device).expect("upload grad");
    let i_in = cpu_to_gpu(&input, &device).expect("upload input");
    let out = kernels::gpu_elu_backward_f64(&g_in, &i_in, alpha, &device).expect("elu_bwd");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("elu_backward_f64", &got, &want, F64_ELEMENTWISE);
}

// ---------------------------------------------------------------------------
// 9. mish_backward_f64
// ---------------------------------------------------------------------------

#[test]
fn mish_backward_f64_matches_reference() {
    ensure_cuda();
    // Mish: y = x * tanh(softplus(x)) where softplus(x) = ln(1+exp(x))
    // dy/dx = tanh(sp) + x * sigmoid(x) * (1 - tanh(sp)^2)
    let input = [-1.0_f64, -0.3, 0.0, 0.3, 1.0];
    let grad = [1.0_f64; 5];
    let want: Vec<f64> = input
        .iter()
        .zip(grad.iter())
        .map(|(&x, &g)| {
            let sp = (1.0 + x.exp()).ln();
            let th = sp.tanh();
            let sig = 1.0 / (1.0 + (-x).exp());
            g * (th + x * sig * (1.0 - th * th))
        })
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad, &device).expect("upload grad");
    let i_in = cpu_to_gpu(&input, &device).expect("upload input");
    let out = kernels::gpu_mish_backward_f64(&g_in, &i_in, &device).expect("mish_bwd");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    // Mish backward involves a deep chain of transcendentals; loosen
    // tolerance slightly vs. F64_ELEMENTWISE.
    assert_close("mish_backward_f64", &got, &want, F64_REDUCTION);
}

// ---------------------------------------------------------------------------
// 10. softmax_backward_f64 (per-row reduction)
// ---------------------------------------------------------------------------

#[test]
fn softmax_backward_f64_matches_reference() {
    ensure_cuda();
    // softmax backward: grad_input[i] = output[i] * (grad[i] - sum(output * grad))
    // Single row [1.0, 2.0, 3.0], grad_output = [0.1, 0.2, 0.3]
    // First compute the softmax output: shift by max, exp, normalize.
    let row_in = [1.0_f64, 2.0, 3.0];
    let m = row_in.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = row_in.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = exps.iter().sum();
    let out_row: Vec<f64> = exps.iter().map(|&e| e / s).collect();
    let grad_out = [0.1_f64, 0.2, 0.3];
    let dot: f64 = out_row
        .iter()
        .zip(grad_out.iter())
        .map(|(o, g)| o * g)
        .sum();
    let want: Vec<f64> = out_row
        .iter()
        .zip(grad_out.iter())
        .map(|(&o, &g)| o * (g - dot))
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad_out, &device).expect("upload grad");
    let o_in = cpu_to_gpu(&out_row, &device).expect("upload output");
    let cols = 3usize;
    let out = kernels::gpu_softmax_backward_f64(&g_in, &o_in, cols, &device).expect("softmax_bwd");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("softmax_backward_f64", &got, &want, F64_REDUCTION);
}

// ---------------------------------------------------------------------------
// 11. log_softmax_backward_f64 (per-row reduction)
// ---------------------------------------------------------------------------

#[test]
fn log_softmax_backward_f64_matches_reference() {
    ensure_cuda();
    // log-softmax backward: grad_input[i] = grad[i] - softmax[i] * sum(grad)
    // Per kernel contract (kernels.rs:5400-5404 #820), the host saves
    // softmax probabilities (post-exp) at forward time and passes them
    // here — NOT log probabilities. Re-applying exp() inside the kernel
    // was the original bug the contract was fixed against. This test
    // matches the saved-softmax convention to mirror the autograd flow.
    let row_in = [1.0_f64, 2.0, 3.0];
    let m = row_in.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = row_in.iter().map(|&x| (x - m).exp()).collect();
    let s: f64 = exps.iter().sum();
    let softmax_row: Vec<f64> = exps.iter().map(|&e| e / s).collect();
    let grad_out = [0.1_f64, 0.2, 0.3];
    let grad_sum: f64 = grad_out.iter().sum();
    let want: Vec<f64> = softmax_row
        .iter()
        .zip(grad_out.iter())
        .map(|(&p, &g)| g - p * grad_sum)
        .collect();
    let device = dev();
    let g_in = cpu_to_gpu(&grad_out, &device).expect("upload grad");
    let o_in = cpu_to_gpu(&softmax_row, &device).expect("upload softmax");
    let cols = 3usize;
    let out = kernels::gpu_log_softmax_backward_f64(&g_in, &o_in, cols, &device)
        .expect("log_softmax_bwd");
    let got = gpu_to_cpu(&out, &device).expect("readback");
    assert_close("log_softmax_backward_f64", &got, &want, F64_REDUCTION);
}

// ---------------------------------------------------------------------------
// 12. layernorm_backward_f64 (per-row reduction)
// ---------------------------------------------------------------------------

#[test]
fn layernorm_backward_f64_matches_reference() {
    ensure_cuda();
    // Single-row layernorm: x = [1, 2, 3, 4], w = [0.5, 0.5, 0.5, 0.5], no bias
    let x = [1.0_f64, 2.0, 3.0, 4.0];
    let w = [0.5_f64, 0.5, 0.5, 0.5];
    let go = [0.1_f64, 0.2, 0.3, 0.4];
    let eps = 1e-5_f64;
    let cols = 4usize;
    let rows = 1usize;
    let mean = x.iter().sum::<f64>() / cols as f64;
    let var = x.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / cols as f64;
    let inv_std = 1.0 / (var + eps).sqrt();
    let xhat: Vec<f64> = x.iter().map(|&v| (v - mean) * inv_std).collect();
    // grad_input via standard layernorm backward
    let dy_w: Vec<f64> = go.iter().zip(w.iter()).map(|(&g, &wi)| g * wi).collect();
    let sum_dy_w: f64 = dy_w.iter().sum();
    let sum_dy_w_xhat: f64 = dy_w.iter().zip(xhat.iter()).map(|(d, x)| d * x).sum();
    let n_f = cols as f64;
    let want_gi: Vec<f64> = xhat
        .iter()
        .zip(dy_w.iter())
        .map(|(&xh, &d)| (d - sum_dy_w / n_f - xh * sum_dy_w_xhat / n_f) * inv_std)
        .collect();
    let want_gw: Vec<f64> = go.iter().zip(xhat.iter()).map(|(&g, &xh)| g * xh).collect();
    let want_gb: Vec<f64> = go.to_vec();
    let device = dev();
    let in_buf = cpu_to_gpu(&x, &device).expect("upload input");
    let go_buf = cpu_to_gpu(&go, &device).expect("upload grad_output");
    let w_buf = cpu_to_gpu(&w, &device).expect("upload weight");
    let (gi, gw, gb) =
        kernels::gpu_layernorm_backward_f64(&in_buf, &go_buf, &w_buf, rows, cols, eps, &device)
            .expect("layernorm_bwd");
    let got_gi = gpu_to_cpu(&gi, &device).expect("readback gi");
    let got_gw = gpu_to_cpu(&gw, &device).expect("readback gw");
    let got_gb = gpu_to_cpu(&gb, &device).expect("readback gb");
    assert_close(
        "layernorm_backward_f64 grad_input",
        &got_gi,
        &want_gi,
        F64_REDUCTION,
    );
    assert_close(
        "layernorm_backward_f64 grad_weight",
        &got_gw,
        &want_gw,
        F64_REDUCTION,
    );
    assert_close(
        "layernorm_backward_f64 grad_bias",
        &got_gb,
        &want_gb,
        F64_REDUCTION,
    );
}

// ---------------------------------------------------------------------------
// 13. rmsnorm_backward_f64 (per-row reduction)
// ---------------------------------------------------------------------------

#[test]
fn rmsnorm_backward_f64_matches_reference() {
    ensure_cuda();
    // Single-row RMSNorm: y = x / rms * w, rms = sqrt(mean(x^2) + eps)
    let x = [1.0_f64, 2.0, 3.0, 4.0];
    let w = [1.0_f64, 1.0, 1.0, 1.0];
    let go = [0.1_f64, 0.2, 0.3, 0.4];
    let eps = 1e-5_f64;
    let cols = 4usize;
    let rows = 1usize;
    let n_f = cols as f64;
    let ms = x.iter().map(|v| v * v).sum::<f64>() / n_f;
    let rms = (ms + eps).sqrt();
    let inv_rms = 1.0 / rms;
    let xhat: Vec<f64> = x.iter().map(|&v| v * inv_rms).collect();
    // grad_input for rmsnorm: dy*w * inv_rms - xhat * dot(dy*w, xhat) / (n * rms)
    let dy_w: Vec<f64> = go.iter().zip(w.iter()).map(|(&g, &wi)| g * wi).collect();
    let dot: f64 = dy_w.iter().zip(xhat.iter()).map(|(d, x)| d * x).sum();
    let want_gi: Vec<f64> = xhat
        .iter()
        .zip(dy_w.iter())
        .map(|(&xh, &d)| d * inv_rms - xh * dot * inv_rms / n_f)
        .collect();
    let want_gw: Vec<f64> = go.iter().zip(xhat.iter()).map(|(&g, &xh)| g * xh).collect();
    let device = dev();
    let in_buf = cpu_to_gpu(&x, &device).expect("upload input");
    let go_buf = cpu_to_gpu(&go, &device).expect("upload grad_output");
    let w_buf = cpu_to_gpu(&w, &device).expect("upload weight");
    let (gi, gw) =
        kernels::gpu_rmsnorm_backward_f64(&in_buf, &go_buf, &w_buf, rows, cols, eps, &device)
            .expect("rmsnorm_bwd");
    let got_gi = gpu_to_cpu(&gi, &device).expect("readback gi");
    let got_gw = gpu_to_cpu(&gw, &device).expect("readback gw");
    assert_close(
        "rmsnorm_backward_f64 grad_input",
        &got_gi,
        &want_gi,
        F64_REDUCTION,
    );
    assert_close(
        "rmsnorm_backward_f64 grad_weight",
        &got_gw,
        &want_gw,
        F64_REDUCTION,
    );
}
