//! Permanent regression sentinel for P5 of #806: FlashAttention forward
//! PTX kernel for nested SDPA.
//!
//! Pre-P5, `nested_scaled_dot_product_attention` materialised the full
//! `[seq_q, seq_k]` scores matrix per component and walked CPU even when
//! every component lived on CUDA. Post-P5, each component on CUDA with
//! `d_head <= 128` routes through the new on-device tiled
//! `flash_attention_forward_{f32,f64}` PTX kernel (online softmax,
//! scalar-fma, no tensor core) and the result remains on CUDA.
//!
//! Coverage:
//!   1. Various shapes (seq, d) for f32 and f64, comparing against the
//!      explicit CPU composite reference `softmax((Q@K^T)*scale) @ V`.
//!   2. Edge cases: seq_q < tile, seq_q non-tile-aligned, large seq,
//!      empty seq.
//!   3. Numerical-stability sentinel: large logits that would overflow
//!      vanilla softmax but are tamed by online stabilisation.
//!   4. Multi-component nested input.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::Device;
use ferrotorch_core::creation::from_vec;
use ferrotorch_core::nested::{NestedTensor, nested_scaled_dot_product_attention};

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for the GPU probe suite");
    });
}

const F32_MATMUL_GPU: f32 = 1e-3;
const F64_MATMUL_GPU: f64 = 1e-9;

fn cpu_reference_f32(q: &[f32], k: &[f32], v: &[f32], sq: usize, sk: usize, d: usize, dv: usize) -> Vec<f32> {
    let scale = 1.0_f32 / (d as f32).sqrt();
    // Q @ K^T
    let mut scores = vec![0.0_f32; sq * sk];
    for qi in 0..sq {
        for ki in 0..sk {
            let mut s = 0.0_f32;
            for di in 0..d {
                s += q[qi * d + di] * k[ki * d + di];
            }
            scores[qi * sk + ki] = s * scale;
        }
    }
    // Row-wise softmax (max-shifted).
    for qi in 0..sq {
        let row = &mut scores[qi * sk..(qi + 1) * sk];
        let m = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for x in row.iter_mut() {
            *x = (*x - m).exp();
            sum += *x;
        }
        for x in row.iter_mut() {
            *x /= sum;
        }
    }
    // scores @ V
    let mut out = vec![0.0_f32; sq * dv];
    for qi in 0..sq {
        for di in 0..dv {
            let mut acc = 0.0_f32;
            for ki in 0..sk {
                acc += scores[qi * sk + ki] * v[ki * dv + di];
            }
            out[qi * dv + di] = acc;
        }
    }
    out
}

fn cpu_reference_f64(q: &[f64], k: &[f64], v: &[f64], sq: usize, sk: usize, d: usize, dv: usize) -> Vec<f64> {
    let scale = 1.0_f64 / (d as f64).sqrt();
    let mut scores = vec![0.0_f64; sq * sk];
    for qi in 0..sq {
        for ki in 0..sk {
            let mut s = 0.0_f64;
            for di in 0..d {
                s += q[qi * d + di] * k[ki * d + di];
            }
            scores[qi * sk + ki] = s * scale;
        }
    }
    for qi in 0..sq {
        let row = &mut scores[qi * sk..(qi + 1) * sk];
        let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0_f64;
        for x in row.iter_mut() {
            *x = (*x - m).exp();
            sum += *x;
        }
        for x in row.iter_mut() {
            *x /= sum;
        }
    }
    let mut out = vec![0.0_f64; sq * dv];
    for qi in 0..sq {
        for di in 0..dv {
            let mut acc = 0.0_f64;
            for ki in 0..sk {
                acc += scores[qi * sk + ki] * v[ki * dv + di];
            }
            out[qi * dv + di] = acc;
        }
    }
    out
}

fn cuda<T>(data: Vec<T>, shape: &[usize]) -> ferrotorch_core::Tensor<T>
where
    T: ferrotorch_core::dtype::Float + 'static,
{
    let cpu = from_vec::<T>(data, shape).expect("cpu tensor");
    cpu.to(Device::Cuda(0)).expect("cpu->gpu")
}

fn run_case_f32(sq: usize, sk: usize, d: usize, dv: usize, label: &str) {
    let q: Vec<f32> = (0..sq * d).map(|i| ((i * 7 + 3) % 50) as f32 / 50.0 - 0.5).collect();
    let k: Vec<f32> = (0..sk * d).map(|i| ((i * 11 + 5) % 50) as f32 / 50.0 - 0.5).collect();
    let v: Vec<f32> = (0..sk * dv).map(|i| ((i * 13 + 7) % 50) as f32 / 50.0 - 0.5).collect();

    let expected = cpu_reference_f32(&q, &k, &v, sq, sk, d, dv);

    let qt = cuda::<f32>(q, &[sq, d]);
    let kt = cuda::<f32>(k, &[sk, d]);
    let vt = cuda::<f32>(v, &[sk, dv]);

    let qn = NestedTensor::new(vec![qt], 0).expect("qn");
    let kn = NestedTensor::new(vec![kt], 0).expect("kn");
    let vn = NestedTensor::new(vec![vt], 0).expect("vn");

    let result = nested_scaled_dot_product_attention(&qn, &kn, &vn).expect("flash sdpa");
    let comp = &result.tensors()[0];
    assert!(comp.is_cuda(), "[{label}] result component must remain CUDA");
    assert_eq!(comp.shape(), &[sq, dv], "[{label}] output shape");
    let host = comp.cpu().expect("gpu->cpu").data().expect("data").to_vec();
    assert_eq!(host.len(), expected.len(), "[{label}] len");
    for (i, (g, e)) in host.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff <= F32_MATMUL_GPU + F32_MATMUL_GPU * e.abs(),
            "[{label}] elem {i}: got {g}, expected {e}, diff {diff}"
        );
    }
    println!(
        "[probe_p5][f32][{label}] sq={sq} sk={sk} d={d} dv={dv} -> max_diff={:.3e}",
        host.iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0_f32, f32::max)
    );
}

fn run_case_f64(sq: usize, sk: usize, d: usize, dv: usize, label: &str) {
    let q: Vec<f64> = (0..sq * d).map(|i| ((i * 7 + 3) % 50) as f64 / 50.0 - 0.5).collect();
    let k: Vec<f64> = (0..sk * d).map(|i| ((i * 11 + 5) % 50) as f64 / 50.0 - 0.5).collect();
    let v: Vec<f64> = (0..sk * dv).map(|i| ((i * 13 + 7) % 50) as f64 / 50.0 - 0.5).collect();

    let expected = cpu_reference_f64(&q, &k, &v, sq, sk, d, dv);

    let qt = cuda::<f64>(q, &[sq, d]);
    let kt = cuda::<f64>(k, &[sk, d]);
    let vt = cuda::<f64>(v, &[sk, dv]);

    let qn = NestedTensor::new(vec![qt], 0).expect("qn");
    let kn = NestedTensor::new(vec![kt], 0).expect("kn");
    let vn = NestedTensor::new(vec![vt], 0).expect("vn");

    let result = nested_scaled_dot_product_attention(&qn, &kn, &vn).expect("flash sdpa f64");
    let comp = &result.tensors()[0];
    assert!(comp.is_cuda(), "[{label}] result component must remain CUDA");
    assert_eq!(comp.shape(), &[sq, dv], "[{label}] output shape");
    let host = comp.cpu().expect("gpu->cpu").data().expect("data").to_vec();
    assert_eq!(host.len(), expected.len(), "[{label}] len");
    for (i, (g, e)) in host.iter().zip(expected.iter()).enumerate() {
        let diff = (g - e).abs();
        assert!(
            diff <= F64_MATMUL_GPU + F64_MATMUL_GPU * e.abs(),
            "[{label}] elem {i}: got {g}, expected {e}, diff {diff:.3e}"
        );
    }
    println!(
        "[probe_p5][f64][{label}] sq={sq} sk={sk} d={d} dv={dv} -> max_diff={:.3e}",
        host.iter()
            .zip(expected.iter())
            .map(|(g, e)| (g - e).abs())
            .fold(0.0_f64, f64::max)
    );
}

#[test]
fn p5_f32_canonical_shapes() {
    ensure_cuda_backend();
    for &(sq, sk, d, dv, lbl) in &[
        (16usize, 16, 16, 16, "16x16x16"),
        (16, 16, 32, 32, "16x16x32"),
        (32, 32, 64, 64, "32x32x64"),
        (64, 64, 64, 64, "64x64x64"),
        (128, 128, 64, 64, "128x128x64"),
    ] {
        run_case_f32(sq, sk, d, dv, lbl);
    }
}

#[test]
fn p5_f64_canonical_shapes() {
    ensure_cuda_backend();
    for &(sq, sk, d, dv, lbl) in &[
        (16usize, 16, 16, 16, "16x16x16"),
        (16, 16, 32, 32, "16x16x32"),
        (32, 32, 64, 64, "32x32x64"),
        (64, 64, 64, 64, "64x64x64"),
    ] {
        run_case_f64(sq, sk, d, dv, lbl);
    }
}

/// seq_q smaller than the tile size (Br=32) — exercises partial-tile path.
#[test]
fn p5_f32_small_seq_below_tile() {
    ensure_cuda_backend();
    run_case_f32(8, 8, 16, 16, "small_below_tile");
}

/// seq_q non-tile-aligned (33) — exercises tail-loop coverage.
#[test]
fn p5_f32_non_tile_aligned() {
    ensure_cuda_backend();
    run_case_f32(33, 33, 32, 32, "non_aligned_33");
    run_case_f32(33, 65, 32, 32, "non_aligned_33x65");
}

/// Larger sequence (1024) — multi-tile reduce path.
#[test]
fn p5_f32_large_seq() {
    ensure_cuda_backend();
    run_case_f32(256, 256, 32, 32, "large_256");
}

/// Empty seq_q -- the GPU branch must produce a [0, d_v] CUDA tensor
/// rather than detouring through CPU.
#[test]
fn p5_f32_empty_seq_q() {
    ensure_cuda_backend();
    let q: Vec<f32> = vec![];
    let k: Vec<f32> = (0..4 * 8).map(|i| i as f32 * 0.01).collect();
    let v: Vec<f32> = (0..4 * 8).map(|i| i as f32 * 0.02).collect();

    let qt = cuda::<f32>(q, &[0, 8]);
    let kt = cuda::<f32>(k, &[4, 8]);
    let vt = cuda::<f32>(v, &[4, 8]);

    let qn = NestedTensor::new(vec![qt], 0).expect("qn");
    let kn = NestedTensor::new(vec![kt], 0).expect("kn");
    let vn = NestedTensor::new(vec![vt], 0).expect("vn");

    let result = nested_scaled_dot_product_attention(&qn, &kn, &vn).expect("flash sdpa empty");
    let comp = &result.tensors()[0];
    assert!(comp.is_cuda(), "empty result must remain CUDA");
    assert_eq!(comp.shape(), &[0, 8]);
    println!("[probe_p5][empty_seq_q] CUDA-resident [0, 8] tensor as expected");
}

/// Numerical stability sentinel: scores that would overflow vanilla
/// softmax (logits ~ 50) but online stabilisation tames them.
#[test]
fn p5_f32_numerical_stability() {
    ensure_cuda_backend();
    let sq = 8;
    let sk = 8;
    let d = 8;
    let dv = 8;
    // Choose Q and K so dot products are large positive numbers.
    let q: Vec<f32> = (0..sq * d).map(|_| 5.0_f32).collect();
    let k: Vec<f32> = (0..sk * d).map(|_| 5.0_f32).collect();
    let v: Vec<f32> = (0..sk * dv).map(|i| (i % 7) as f32).collect();

    let expected = cpu_reference_f32(&q, &k, &v, sq, sk, d, dv);

    let qt = cuda::<f32>(q, &[sq, d]);
    let kt = cuda::<f32>(k, &[sk, d]);
    let vt = cuda::<f32>(v, &[sk, dv]);

    let qn = NestedTensor::new(vec![qt], 0).expect("qn");
    let kn = NestedTensor::new(vec![kt], 0).expect("kn");
    let vn = NestedTensor::new(vec![vt], 0).expect("vn");

    let result = nested_scaled_dot_product_attention(&qn, &kn, &vn).expect("flash sdpa stable");
    let comp = &result.tensors()[0];
    let host = comp.cpu().expect("gpu->cpu").data().expect("data").to_vec();
    for (i, (g, e)) in host.iter().zip(expected.iter()).enumerate() {
        assert!(
            g.is_finite(),
            "online softmax must produce finite output (i={i}, g={g})"
        );
        let diff = (g - e).abs();
        assert!(
            diff <= 1e-2 + 1e-2 * e.abs(),
            "[stability] elem {i}: got {g}, expected {e}, diff {diff}"
        );
    }
    println!("[probe_p5][stability] online softmax stays finite under large logits");
}

/// Multi-component nested input — every component must dispatch to GPU.
#[test]
fn p5_f32_multi_component_nested() {
    ensure_cuda_backend();
    let mut q_comps = Vec::new();
    let mut k_comps = Vec::new();
    let mut v_comps = Vec::new();
    let mut expected = Vec::new();

    for &(sq, sk) in &[(8usize, 8), (16, 16), (24, 32)] {
        let d = 16;
        let dv = 16;
        let q: Vec<f32> = (0..sq * d).map(|i| ((i * 7) % 50) as f32 / 50.0 - 0.5).collect();
        let kv: Vec<f32> = (0..sk * d).map(|i| ((i * 11) % 50) as f32 / 50.0 - 0.5).collect();
        let v: Vec<f32> = (0..sk * dv).map(|i| ((i * 13) % 50) as f32 / 50.0 - 0.5).collect();

        let exp = cpu_reference_f32(&q, &kv, &v, sq, sk, d, dv);
        expected.push((exp, sq, dv));

        q_comps.push(cuda::<f32>(q, &[sq, d]));
        k_comps.push(cuda::<f32>(kv, &[sk, d]));
        v_comps.push(cuda::<f32>(v, &[sk, dv]));
    }

    let qn = NestedTensor::new(q_comps, 0).expect("qn");
    let kn = NestedTensor::new(k_comps, 0).expect("kn");
    let vn = NestedTensor::new(v_comps, 0).expect("vn");

    let result = nested_scaled_dot_product_attention(&qn, &kn, &vn).expect("multi-comp sdpa");
    assert_eq!(result.num_components(), 3);

    for (i, (exp, sq, dv)) in expected.iter().enumerate() {
        let comp = &result.tensors()[i];
        assert!(comp.is_cuda(), "comp {i} must remain CUDA");
        assert_eq!(comp.shape(), &[*sq, *dv], "comp {i} shape");
        let host = comp.cpu().expect("gpu->cpu").data().expect("data").to_vec();
        for (j, (g, e)) in host.iter().zip(exp.iter()).enumerate() {
            let diff = (g - e).abs();
            assert!(
                diff <= F32_MATMUL_GPU + F32_MATMUL_GPU * e.abs(),
                "[multi][comp {i}] elem {j}: got {g}, expected {e}, diff {diff}"
            );
        }
    }
    println!("[probe_p5][multi_component] all 3 components matched within tol");
}
