//! Probe for Bugfix Batch 8 / Dispatch A2 (#813).
//!
//! flex_attention's f64 GPU path errors with
//! `'GPU handle does not contain a CudaBuffer<f32>'` even on the baseline
//! (no score_mod) chain. This probe walks each step of the composed chain
//! on f64 CUDA and identifies the FIRST step that fails. The architect's
//! "probe-before-fix" pattern is mandatory; do NOT guess.
//!
//! Steps (in dispatch order inside `flex_attention.rs`):
//!
//!   1. f64 CUDA `reshape` — `[B, H, n_q, d] -> [B*H, n_q, d]`
//!   2. f64 CUDA `transpose(1, 2)` — last two dims of a 3-D tensor
//!   3. f64 CUDA `bmm_differentiable` — `(1, 4, 4) @ (1, 4, 4) -> (1, 4, 4)`
//!   4. f64 CUDA scalar `mul` (broadcast `[1]` against `[1, 4, 4]`)
//!   5. f64 CUDA `softmax` along the last dim of a 4-D tensor
//!   6. Full chain on `B=1, H=1, n_q=4, n_k=4, d=4, d_v=4`
//!
//! The first step that errs is the bug.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::creation;
use ferrotorch_core::flex_attention;
use ferrotorch_core::grad_fns::{activation, arithmetic, linalg, shape};
use ferrotorch_core::{Device, FerrotorchResult, Tensor, TensorStorage};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe");
    });
}

fn upload_f64(t: Tensor<f64>) -> Tensor<f64> {
    t.to(Device::Cuda(0)).expect("upload to cuda")
}

fn make_cpu_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).expect("make_cpu_f64")
}

#[test]
fn probe_step1_reshape_f64_cuda() {
    ensure_cuda_backend();
    // B=1, H=1, n_q=4, d=4 -> 16 elems
    let q = upload_f64(make_cpu_f64((0..16).map(|i| i as f64).collect(), vec![1, 1, 4, 4]));
    let r = shape::reshape(&q, &[1, 4, 4]).expect("reshape f64 CUDA failed");
    assert_eq!(r.shape(), &[1, 4, 4]);
    let cpu = r.cpu().expect("D2H readback");
    let v = cpu.data().expect("read CPU data after readback").to_vec();
    assert_eq!(v.len(), 16);
    eprintln!("step1 OK: reshape f64 CUDA");
}

#[test]
fn probe_step2_transpose_f64_cuda() {
    ensure_cuda_backend();
    let k = upload_f64(make_cpu_f64((0..16).map(|i| i as f64).collect(), vec![1, 4, 4]));
    let kt = k.transpose(1, 2).expect("transpose f64 CUDA failed");
    assert_eq!(kt.shape(), &[1, 4, 4]);
    eprintln!("step2 OK: transpose f64 CUDA");
}

#[test]
fn probe_step3_bmm_f64_cuda() {
    ensure_cuda_backend();
    let a = upload_f64(make_cpu_f64((0..16).map(|i| i as f64).collect(), vec![1, 4, 4]));
    let b = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 4, 4]));
    let r = linalg::bmm_differentiable(&a, &b).expect("bmm_differentiable f64 CUDA failed");
    assert_eq!(r.shape(), &[1, 4, 4]);
    let cpu = r.cpu().expect("D2H readback");
    let _v = cpu.data().expect("read CPU data after readback").to_vec();
    eprintln!("step3 OK: bmm_differentiable f64 CUDA");
}

#[test]
fn probe_step3b_bmm_after_transpose_f64_cuda() {
    // The flex_attention chain hands bmm a TRANSPOSED (non-contiguous) view of K.
    // bmm() runs `.contiguous()?` on it before the GPU dispatch — so this step
    // exercises that internal contiguous() path on f64 CUDA.
    ensure_cuda_backend();
    let q = upload_f64(make_cpu_f64((0..16).map(|i| i as f64).collect(), vec![1, 4, 4]));
    let k = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 4, 4]));
    let kt = k.transpose(1, 2).expect("transpose");
    let r = linalg::bmm_differentiable(&q, &kt)
        .expect("bmm_differentiable(q, k.t()) f64 CUDA failed");
    assert_eq!(r.shape(), &[1, 4, 4]);
    let cpu = r.cpu().expect("D2H readback");
    let _v = cpu.data().expect("read CPU data after readback").to_vec();
    eprintln!("step3b OK: bmm_differentiable f64 CUDA after transpose");
}

#[test]
fn probe_step4_scalar_mul_f64_cuda() {
    ensure_cuda_backend();
    // Mirror the flex_attention scale construction: scalar tensor on CUDA
    // multiplied against a [1, 4, 4] CUDA tensor.
    let scale: f64 = 1.0 / (4.0_f64).sqrt();
    let scale_t = creation::scalar(scale)
        .expect("creation::scalar f64 failed")
        .to(Device::Cuda(0))
        .expect("upload scalar to CUDA");
    let scores = upload_f64(make_cpu_f64((0..16).map(|i| i as f64).collect(), vec![1, 4, 4]));
    let r = arithmetic::mul(&scores, &scale_t).expect("scalar mul f64 CUDA failed");
    assert_eq!(r.shape(), &[1, 4, 4]);
    let cpu = r.cpu().expect("D2H readback");
    let _v = cpu.data().expect("read CPU data after readback").to_vec();
    eprintln!("step4 OK: scalar mul f64 CUDA");
}

#[test]
fn probe_step5_softmax_f64_cuda() {
    ensure_cuda_backend();
    let s = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let r = activation::softmax(&s).expect("softmax f64 CUDA failed");
    assert_eq!(r.shape(), &[1, 1, 4, 4]);
    let cpu = r.cpu().expect("D2H readback");
    let _v = cpu.data().expect("read CPU data after readback").to_vec();
    eprintln!("step5 OK: softmax f64 CUDA");
}

#[test]
fn probe_step6_full_chain_f64_cuda() {
    ensure_cuda_backend();
    let q = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let k = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let v = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let out = flex_attention::flex_attention::<
        f64,
        fn(&Tensor<f64>, usize, usize) -> FerrotorchResult<Tensor<f64>>,
    >(&q, &k, &v, None)
    .expect("flex_attention f64 CUDA full chain failed");
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let cpu = out.cpu().expect("D2H readback");
    let _v = cpu.data().expect("read CPU data after readback").to_vec();
    eprintln!("step6 OK: full flex_attention chain f64 CUDA");
}

/// Multi-head (B=1, H=2): two_heads_4x4_baseline shape.
#[test]
fn probe_step7_two_heads_baseline_f64_cuda() {
    ensure_cuda_backend();
    let n = 2 * 4 * 4;
    let q = upload_f64(make_cpu_f64((0..n).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4, 4]));
    let k = upload_f64(make_cpu_f64((0..n).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4, 4]));
    let v = upload_f64(make_cpu_f64((0..n).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4, 4]));
    let out = flex_attention::flex_attention::<
        f64,
        fn(&Tensor<f64>, usize, usize) -> FerrotorchResult<Tensor<f64>>,
    >(&q, &k, &v, None)
    .expect("flex_attention f64 CUDA two_heads baseline failed");
    assert_eq!(out.shape(), &[1, 2, 4, 4]);
    eprintln!("step7 OK: two_heads_baseline f64 CUDA");
}

/// rect_3x5: (B=2, H=1, n_q=3, n_k=5, d=4, d_v=4).
#[test]
fn probe_step8_rect_baseline_f64_cuda() {
    ensure_cuda_backend();
    let q_n = 2 * 3 * 4;
    let k_n = 2 * 5 * 4;
    let v_n = 2 * 5 * 4;
    let q = upload_f64(make_cpu_f64((0..q_n).map(|i| i as f64 * 0.1).collect(), vec![2, 1, 3, 4]));
    let k = upload_f64(make_cpu_f64((0..k_n).map(|i| i as f64 * 0.1).collect(), vec![2, 1, 5, 4]));
    let v = upload_f64(make_cpu_f64((0..v_n).map(|i| i as f64 * 0.1).collect(), vec![2, 1, 5, 4]));
    let out = flex_attention::flex_attention::<
        f64,
        fn(&Tensor<f64>, usize, usize) -> FerrotorchResult<Tensor<f64>>,
    >(&q, &k, &v, None)
    .expect("flex_attention f64 CUDA rect_3x5 baseline failed");
    assert_eq!(out.shape(), &[2, 1, 3, 4]);
    eprintln!("step8 OK: rect_baseline f64 CUDA");
}

/// dv_neq_d_baseline: (B=2, H=2, n_q=4, n_k=4, d=4, d_v=6).
#[test]
fn probe_step9_dv_neq_d_baseline_f64_cuda() {
    ensure_cuda_backend();
    let q_n = 2 * 2 * 4 * 4;
    let k_n = 2 * 2 * 4 * 4;
    let v_n = 2 * 2 * 4 * 6;
    let q = upload_f64(make_cpu_f64((0..q_n).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 4, 4]));
    let k = upload_f64(make_cpu_f64((0..k_n).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 4, 4]));
    let v = upload_f64(make_cpu_f64((0..v_n).map(|i| i as f64 * 0.1).collect(), vec![2, 2, 4, 6]));
    let out = flex_attention::flex_attention::<
        f64,
        fn(&Tensor<f64>, usize, usize) -> FerrotorchResult<Tensor<f64>>,
    >(&q, &k, &v, None)
    .expect("flex_attention f64 CUDA dv_neq_d baseline failed");
    assert_eq!(out.shape(), &[2, 2, 4, 6]);
    eprintln!("step9 OK: dv_neq_d_baseline f64 CUDA");
}

/// Backward (autograd) — the conformance backward test will set requires_grad
/// on Q/K/V and call `loss.backward()`. The forward path is the same, but the
/// backward node chain runs grad ops on f64 CUDA tensors.
#[test]
fn probe_step10_backward_single_head_f64_cuda() {
    ensure_cuda_backend();
    let q = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let k = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let v = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    // Switch on requires_grad post-upload.
    let q = q.requires_grad_(true);
    let k = k.requires_grad_(true);
    let v = v.requires_grad_(true);

    let out = flex_attention::flex_attention::<
        f64,
        fn(&Tensor<f64>, usize, usize) -> FerrotorchResult<Tensor<f64>>,
    >(&q, &k, &v, None)
    .expect("flex_attention f64 CUDA backward forward failed");

    let loss =
        ferrotorch_core::grad_fns::reduction::sum(&out).expect("sum f64");
    loss.backward().expect("backward f64");
    let _gq = q.grad().unwrap().expect("grad_q f64");
    let _gk = k.grad().unwrap().expect("grad_k f64");
    let _gv = v.grad().unwrap().expect("grad_v f64");
    eprintln!("step10 OK: single_head backward f64 CUDA");
}

/// Direct softmax with -inf inputs on f64 CUDA — does the GPU kernel handle
/// `-inf` in a row correctly?
#[test]
fn probe_step5b_softmax_with_neg_inf_f64_cuda() {
    ensure_cuda_backend();
    // Row 0: [0.0, -inf, -inf, -inf]  -> expect [1.0, 0.0, 0.0, 0.0]
    // Row 1: [0.5, 0.5, -inf, -inf]   -> expect [0.5, 0.5, 0.0, 0.0]
    // Row 2: [0.3, 0.5, 0.7, -inf]    -> expect normal softmax over 3
    // Row 3: [0.1, 0.2, 0.3, 0.4]     -> expect normal softmax over 4
    let neg_inf = f64::NEG_INFINITY;
    let data: Vec<f64> = vec![
        0.0, neg_inf, neg_inf, neg_inf,
        0.5, 0.5, neg_inf, neg_inf,
        0.3, 0.5, 0.7, neg_inf,
        0.1, 0.2, 0.3, 0.4,
    ];
    let s = upload_f64(make_cpu_f64(data, vec![1, 1, 4, 4]));
    let r = activation::softmax(&s).expect("softmax with -inf f64 CUDA failed");
    let cpu = r.cpu().expect("D2H readback");
    let v = cpu.data().expect("read CPU data").to_vec();
    eprintln!("softmax_f64 with -inf rows: {:?}", v);
    for (i, x) in v.iter().enumerate() {
        assert!(x.is_finite(), "softmax f64 -inf: index {i} = {x}");
    }
}

/// Same as above but f32 — does the f32 GPU softmax handle -inf?
#[test]
fn probe_step5c_softmax_with_neg_inf_f32_cuda() {
    ensure_cuda_backend();
    let neg_inf = f32::NEG_INFINITY;
    let data: Vec<f32> = vec![
        0.0, neg_inf, neg_inf, neg_inf,
        0.5, 0.5, neg_inf, neg_inf,
        0.3, 0.5, 0.7, neg_inf,
        0.1, 0.2, 0.3, 0.4,
    ];
    let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1, 4, 4], false)
        .expect("make_cpu_f32")
        .to(Device::Cuda(0))
        .expect("upload");
    let r = activation::softmax(&t).expect("softmax with -inf f32 CUDA failed");
    let cpu = r.cpu().expect("D2H readback");
    let v = cpu.data().expect("read CPU data").to_vec();
    eprintln!("softmax_f32 with -inf rows: {:?}", v);
    for (i, x) in v.iter().enumerate() {
        assert!(x.is_finite(), "softmax f32 -inf: index {i} = {x}");
    }
}

/// Single-head + causal mask on f32 GPU — this is what the conformance test
/// tries first. Mirroring conformance_flex_attention.rs::run_forward_for_device.
#[test]
fn probe_step13_single_head_causal_f32_cuda() {
    ensure_cuda_backend();
    let q_data: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
    let q = Tensor::from_storage(TensorStorage::cpu(q_data.clone()), vec![1, 1, 4, 4], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let k = Tensor::from_storage(TensorStorage::cpu(q_data.clone()), vec![1, 1, 4, 4], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let v = Tensor::from_storage(TensorStorage::cpu(q_data), vec![1, 1, 4, 4], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    // causal mask addend
    let mask: Vec<f32> = (0..4)
        .flat_map(|i| (0..4).map(move |j| if j <= i { 0.0 } else { f32::NEG_INFINITY }))
        .collect();
    let mask_t = Tensor::from_storage(TensorStorage::cpu(mask), vec![4, 4], false)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let closure = move |s: &Tensor<f32>, _b: usize, _h: usize| -> FerrotorchResult<Tensor<f32>> {
        ferrotorch_core::grad_fns::arithmetic::add(s, &mask_t)
    };
    let out = flex_attention::flex_attention(&q, &k, &v, Some(closure))
        .expect("flex_attention single_head causal f32 CUDA failed");
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    eprintln!("step13 OK: single_head + causal f32 CUDA");
}

/// Single-head + alibi-style additive bias only (no -inf mask) on f64 GPU.
/// This should work even with a buggy softmax_f64 because there's no -inf.
#[test]
fn probe_step12c_single_head_alibi_only_f64_cuda() {
    ensure_cuda_backend();
    let q = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let k = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let v = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let bias: Vec<f64> = (0..16).map(|i| i as f64 * 0.01).collect();
    let bias_t = upload_f64(make_cpu_f64(bias, vec![4, 4]));
    let closure = move |s: &Tensor<f64>, _b: usize, _h: usize| -> FerrotorchResult<Tensor<f64>> {
        ferrotorch_core::grad_fns::arithmetic::add(s, &bias_t)
    };
    let out = flex_attention::flex_attention(&q, &k, &v, Some(closure))
        .expect("flex_attention single_head alibi f64 CUDA failed");
    let cpu = out.cpu().expect("D2H readback");
    let v_out = cpu.data().expect("read CPU data after readback").to_vec();
    eprintln!("step12c OUT: {:?}", v_out);
    for (i, x) in v_out.iter().enumerate() {
        assert!(x.is_finite(), "single_head alibi f64: index {i} = {x}");
    }
    eprintln!("step12c OK: single_head + alibi f64 CUDA");
}

/// Single-head with causal mask (mask_neg_inf addend): the conformance suite
/// runs this and it produces NaN under f64 GPU. Reproduce:
#[test]
fn probe_step12b_single_head_causal_f64_cuda() {
    ensure_cuda_backend();
    let q = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let k = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    let v = upload_f64(make_cpu_f64((0..16).map(|i| i as f64 * 0.1).collect(), vec![1, 1, 4, 4]));
    // Causal mask addend: lower triangle is 0, upper triangle is -inf.
    let mask: Vec<f64> = (0..4)
        .flat_map(|i| (0..4).map(move |j| if j <= i { 0.0 } else { f64::NEG_INFINITY }))
        .collect();
    let mask_t = upload_f64(make_cpu_f64(mask, vec![4, 4]));
    let closure = move |s: &Tensor<f64>, _b: usize, _h: usize| -> FerrotorchResult<Tensor<f64>> {
        ferrotorch_core::grad_fns::arithmetic::add(s, &mask_t)
    };
    let out = flex_attention::flex_attention(&q, &k, &v, Some(closure))
        .expect("flex_attention single_head causal f64 CUDA failed");
    assert_eq!(out.shape(), &[1, 1, 4, 4]);
    let cpu = out.cpu().expect("D2H readback");
    let v_out = cpu.data().expect("read CPU data after readback").to_vec();
    eprintln!("step12b OUT: {:?}", v_out);
    for (i, x) in v_out.iter().enumerate() {
        assert!(x.is_finite(), "single_head causal f64: index {i} = {x}");
    }
    eprintln!("step12b OK: single_head + causal f64 CUDA");
}

/// Multi-head with score_mod (closure): this is what trips #812 for f32, but
/// we want to know if f64 fares the same — once #813 is removed, will the
/// score_mod multi-bh f64 cases also need #812?
#[test]
fn probe_step12_two_heads_with_score_mod_f64_cuda() {
    ensure_cuda_backend();
    let n = 2 * 4 * 4;
    let q = upload_f64(make_cpu_f64((0..n).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4, 4]));
    let k = upload_f64(make_cpu_f64((0..n).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4, 4]));
    let v = upload_f64(make_cpu_f64((0..n).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4, 4]));
    // Score mod that adds zero (no-op semantically, but exercises the closure path).
    let bias = upload_f64(make_cpu_f64(vec![0.0; 16], vec![4, 4]));
    let closure = move |s: &Tensor<f64>, _b: usize, _h: usize| -> FerrotorchResult<Tensor<f64>> {
        ferrotorch_core::grad_fns::arithmetic::add(s, &bias)
    };
    let result = flex_attention::flex_attention(&q, &k, &v, Some(closure));
    match result {
        Ok(out) => {
            assert_eq!(out.shape(), &[1, 2, 4, 4]);
            eprintln!("step12 OK: two_heads + score_mod f64 CUDA");
        }
        Err(e) => {
            eprintln!("step12 FAIL: two_heads + score_mod f64 CUDA: {e}");
            panic!("step12 FAIL: {e}");
        }
    }
}

/// Backward on multi-head: this is closer to the conformance two_heads_4x4_baseline
/// backward pattern.
#[test]
fn probe_step11_backward_two_heads_f64_cuda() {
    ensure_cuda_backend();
    let n = 2 * 4 * 4;
    let q = upload_f64(make_cpu_f64((0..n).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4, 4]));
    let k = upload_f64(make_cpu_f64((0..n).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4, 4]));
    let v = upload_f64(make_cpu_f64((0..n).map(|i| i as f64 * 0.1).collect(), vec![1, 2, 4, 4]));
    let q = q.requires_grad_(true);
    let k = k.requires_grad_(true);
    let v = v.requires_grad_(true);

    let out = flex_attention::flex_attention::<
        f64,
        fn(&Tensor<f64>, usize, usize) -> FerrotorchResult<Tensor<f64>>,
    >(&q, &k, &v, None)
    .expect("flex_attention f64 CUDA two_heads backward forward failed");
    let loss =
        ferrotorch_core::grad_fns::reduction::sum(&out).expect("sum f64");
    loss.backward().expect("backward f64");
    let _gq = q.grad().unwrap().expect("grad_q f64");
    let _gk = k.grad().unwrap().expect("grad_k f64");
    let _gv = v.grad().unwrap().expect("grad_v f64");
    eprintln!("step11 OK: two_heads backward f64 CUDA");
}
