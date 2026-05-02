//! GPU sum_dim backward test on RTX 3090. (#524)
//!
//! Validates the new `gpu_repeat_along_dim` kernel via the `sum_dim`
//! autograd path: forward sum_dim → sum_all → backward, then check that
//! the gradient flowing back to the input is correctly broadcast along
//! the previously-reduced dim.

#![cfg(feature = "cuda")]

use ferrotorch_core::grad_fns::reduction::sum_dim;
use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_gpu::init_cuda_backend;

fn ensure_cuda() {
    use std::sync::Once;
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        init_cuda_backend().expect("CUDA backend init");
    });
}

fn cpu_t_f32(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
    Tensor::from_storage(
        TensorStorage::cpu(data.to_vec()),
        shape.to_vec(),
        requires_grad,
    )
    .unwrap()
}

#[test]
fn sum_dim_backward_gpu_propagates_ones_along_reduced_dim() {
    // Simple 2-D case: input [3, 4] sum over dim 1 → [3]. Then sum to scalar.
    // dL/d(input)[i, j] should be 1.0 for every position (sum's grad
    // flows uniformly).
    ensure_cuda();
    let mut data = Vec::new();
    for i in 0..12 {
        data.push(i as f32 * 0.1);
    }
    let x = cpu_t_f32(&data, &[3, 4], true).to(Device::Cuda(0)).unwrap();
    let s = sum_dim(&x, 1, false).unwrap(); // [3]
    assert_eq!(s.shape(), &[3]);
    let scalar = s.sum_all().unwrap();
    scalar.backward().unwrap();

    let g = x.grad().unwrap().unwrap().cpu().unwrap();
    let g_data = g.data().unwrap();
    for (i, &v) in g_data.iter().enumerate() {
        assert!((v - 1.0).abs() < 1e-6, "i={i}: got {v}");
    }
}

#[test]
fn sum_dim_backward_gpu_keepdim_works() {
    // keepdim=true variant: input [2, 5, 3] sum over dim 1 → [2, 1, 3].
    ensure_cuda();
    let n = 30;
    let data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05).collect();
    let x = cpu_t_f32(&data, &[2, 5, 3], true)
        .to(Device::Cuda(0))
        .unwrap();
    let s = sum_dim(&x, 1, true).unwrap();
    assert_eq!(s.shape(), &[2, 1, 3]);
    let scalar = s.sum_all().unwrap();
    scalar.backward().unwrap();

    let g = x.grad().unwrap().unwrap().cpu().unwrap();
    let g_data = g.data().unwrap();
    // Every position contributed once to the sum → grad is 1.0 everywhere.
    for &v in g_data.iter() {
        assert!((v - 1.0).abs() < 1e-6);
    }
}

#[test]
fn sum_dim_backward_gpu_matches_cpu() {
    // Compare GPU and CPU paths element-wise.
    ensure_cuda();
    let data: Vec<f32> = (0..24).map(|i| ((i as f32) * 0.5).sin()).collect();
    let x_cpu = cpu_t_f32(&data, &[2, 3, 4], true);
    let x_gpu = x_cpu.clone().to(Device::Cuda(0)).unwrap();

    // Reduce along middle dim.
    let s_cpu = sum_dim(&x_cpu, 1, false).unwrap();
    let s_gpu = sum_dim(&x_gpu, 1, false).unwrap();
    let scalar_cpu = s_cpu.sum_all().unwrap();
    let scalar_gpu = s_gpu.sum_all().unwrap();
    scalar_cpu.backward().unwrap();
    scalar_gpu.backward().unwrap();

    let g_cpu = x_cpu.grad().unwrap().unwrap();
    let g_gpu_host = x_gpu.grad().unwrap().unwrap().cpu().unwrap();
    let g_cpu_data = g_cpu.data().unwrap();
    let g_gpu_data = g_gpu_host.data().unwrap();
    for i in 0..g_cpu_data.len() {
        assert!(
            (g_cpu_data[i] - g_gpu_data[i]).abs() < 1e-5,
            "i={i}: cpu={} gpu={}",
            g_cpu_data[i],
            g_gpu_data[i]
        );
    }
}
