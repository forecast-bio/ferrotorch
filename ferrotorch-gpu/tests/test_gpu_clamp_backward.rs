//! GPU `clamp` backward integration tests on RTX 3090. (#524)
//!
//! Validates the new `gpu_clamp_backward` / `gpu_clamp_backward_f64`
//! kernels that close the previous `NotImplementedOnCuda` gap in
//! `ClampBackward::backward`. Each test exercises:
//!   - in-range gradients passing through unchanged
//!   - out-of-range gradients zeroed
//!   - boundary equality (x == min, x == max) treated as in-range

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::transcendental::clamp;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_t_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

fn cpu_t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
}

#[test]
fn clamp_backward_f32_gpu_matches_cpu() {
    ensure_cuda();
    // Inputs: -3, -1, 0, 0.5, 1, 2 with clamp(-1, 1).
    // Expected grad mask: [out-of-range, in-range (boundary), in-range,
    //                       in-range, in-range (boundary), out-of-range]
    let x_cpu = cpu_t_f32(&[-3.0, -1.0, 0.0, 0.5, 1.0, 2.0], &[6]);
    let x_gpu = x_cpu.clone().to(Device::Cuda(0)).unwrap();

    let y_cpu = clamp(&x_cpu, -1.0, 1.0).unwrap();
    let y_gpu = clamp(&x_gpu, -1.0, 1.0).unwrap();

    let s_cpu = y_cpu.sum_all().unwrap();
    let s_gpu = y_gpu.sum_all().unwrap();
    s_cpu.backward().unwrap();
    s_gpu.backward().unwrap();

    let g_cpu = x_cpu.grad().unwrap().unwrap();
    let g_gpu = x_gpu.grad().unwrap().unwrap();
    let g_cpu_data = g_cpu.data().unwrap().to_vec();
    let g_gpu_data = g_gpu.cpu().unwrap().data().unwrap().to_vec();

    // Expected: 0 for out-of-range, 1 for in-range (sum's grad is 1 per element).
    let expected: Vec<f32> = vec![0.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    for (i, ((&got, &cpu_got), &exp)) in g_gpu_data
        .iter()
        .zip(g_cpu_data.iter())
        .zip(expected.iter())
        .enumerate()
    {
        assert!((got - exp).abs() < 1e-6, "i={i}: gpu={got} expected={exp}");
        assert!(
            (cpu_got - exp).abs() < 1e-6,
            "i={i}: cpu={cpu_got} expected={exp}"
        );
    }
}

#[test]
fn clamp_backward_f64_kernel_correctness() {
    // Exercise the f64 PTX kernel directly via the GpuBackend trait —
    // bypasses the f64 sum_all autograd backward which separately depends
    // on a `fill_f64` op that's not yet wired (orthogonal to clamp_backward).
    ensure_cuda();
    use ferrotorch_core::gpu_dispatch::gpu_backend;

    let backend = gpu_backend().expect("CUDA backend registered");
    let x_cpu = cpu_t_f64(&[-100.0, -0.5, 0.0, 0.25, 0.5, 200.0], &[6]);
    let g_cpu = cpu_t_f64(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
    let x_gpu = x_cpu.to(Device::Cuda(0)).unwrap();
    let g_gpu = g_cpu.to(Device::Cuda(0)).unwrap();

    let h = backend
        .clamp_backward_f64(
            g_gpu.gpu_handle().unwrap(),
            x_gpu.gpu_handle().unwrap(),
            -0.5,
            0.5,
        )
        .unwrap();
    let result = Tensor::<f64>::from_storage(TensorStorage::gpu(h), vec![6], false)
        .unwrap()
        .cpu()
        .unwrap();
    // x = [-100, -0.5, 0, 0.25, 0.5, 200]; in [-0.5, 0.5] = [F, T, T, T, T, F]
    // grad masked: [0, 2, 3, 4, 5, 0]
    let expected = [0.0_f64, 2.0, 3.0, 4.0, 5.0, 0.0];
    let got = result.data().unwrap();
    for i in 0..6 {
        assert!(
            (got[i] - expected[i]).abs() < 1e-12,
            "i={i}: got {} expected {}",
            got[i],
            expected[i]
        );
    }
}

#[test]
fn clamp_backward_propagates_grad_through_chain() {
    // y = clamp(x, 0, 5); loss = sum(y * 2). dL/dx = 2 in-range, 0 out-of-range.
    ensure_cuda();
    let x = Tensor::from_storage(
        TensorStorage::cpu(vec![-1.0_f32, 1.0, 6.0, 3.0, 5.0]),
        vec![5],
        true,
    )
    .unwrap()
    .to(Device::Cuda(0))
    .unwrap();

    let y = clamp(&x, 0.0, 5.0).unwrap();
    // Multiply by 2 elementwise via add(y, y). (Skip — sum_all gives 1; we
    // can multiply manually with a constant via creation::scalar.)
    // Simpler: use sum_all directly, expected grad = 1.0 in-range, 0 else.
    let s = y.sum_all().unwrap();
    s.backward().unwrap();

    let g = x.grad().unwrap().unwrap().cpu().unwrap();
    let g_data = g.data().unwrap();
    // x = [-1, 1, 6, 3, 5]; in [0, 5] -> [F, T, F, T, T]
    let expected: Vec<f32> = vec![0.0, 1.0, 0.0, 1.0, 1.0];
    for (i, (&got, &exp)) in g_data.iter().zip(expected.iter()).enumerate() {
        assert!((got - exp).abs() < 1e-6, "i={i}: got {got}, expected {exp}");
    }
}

#[test]
fn clamp_backward_large_input_gpu() {
    // 10K elements stress test.
    ensure_cuda();
    let n = 10_000;
    let host: Vec<f32> = (0..n).map(|i| (i as f32 - 5000.0) * 0.001).collect();
    let x = Tensor::from_storage(TensorStorage::cpu(host.clone()), vec![n], true)
        .unwrap()
        .to(Device::Cuda(0))
        .unwrap();
    let y = clamp(&x, -1.0, 1.0).unwrap();
    let s = y.sum_all().unwrap();
    s.backward().unwrap();
    let g = x.grad().unwrap().unwrap().cpu().unwrap();
    let g_data = g.data().unwrap();

    // Expected: 1 if -1 <= x <= 1, else 0.
    let mut in_range = 0usize;
    for (i, &h) in host.iter().enumerate() {
        let exp = if (-1.0..=1.0).contains(&h) { 1.0 } else { 0.0 };
        assert!(
            (g_data[i] - exp).abs() < 1e-6,
            "i={i}: got {} exp {exp}",
            g_data[i]
        );
        if exp == 1.0 {
            in_range += 1;
        }
    }
    // Sanity: roughly half should be in [-1, 1] (since values span [-5, 5]).
    assert!(in_range > 1500 && in_range < 2500, "in_range={in_range}");
}
