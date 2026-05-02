//! GPU prod forward test on RTX 3090. (#524)

#![cfg(feature = "cuda")]

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
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

fn cpu_t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
}

#[test]
fn prod_f32_gpu_matches_cpu() {
    ensure_cuda();
    let a = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
    let cpu_p = a.prod_all().unwrap().item().unwrap();
    let g = a.to(Device::Cuda(0)).unwrap();
    let gpu_p = g.prod_all().unwrap().cpu().unwrap().item().unwrap();
    assert!((cpu_p - 120.0).abs() < 1e-5);
    assert!((gpu_p - 120.0).abs() < 1e-3);
}

#[test]
fn prod_f64_gpu_matches_cpu() {
    ensure_cuda();
    let a = cpu_t_f64(&[2.0, 0.5, 3.0, 0.25, 4.0], &[5]);
    let cpu_p = a.prod_all().unwrap().item().unwrap();
    let g = a.to(Device::Cuda(0)).unwrap();
    let gpu_p = g.prod_all().unwrap().cpu().unwrap().item().unwrap();
    // 2 * 0.5 * 3 * 0.25 * 4 = 3.0
    assert!((cpu_p - 3.0).abs() < 1e-9);
    assert!((gpu_p - 3.0).abs() < 1e-9);
}

#[test]
fn prod_returns_gpu_tensor() {
    ensure_cuda();
    let g = cpu_t_f32(&[1.0, 2.0, 3.0], &[3])
        .to(Device::Cuda(0))
        .unwrap();
    let p = g.prod_all().unwrap();
    assert!(matches!(p.device(), Device::Cuda(0)));
}

#[test]
fn prod_large_input_no_overflow_for_unit_factors() {
    // 10000 elements all = 1.0 -> prod = 1.0 exactly. Tests the multi-pass
    // tree reduction (>256 blocks → recurse).
    ensure_cuda();
    let n = 10_000;
    let g = cpu_t_f32(&vec![1.0_f32; n], &[n])
        .to(Device::Cuda(0))
        .unwrap();
    let p = g.prod_all().unwrap().cpu().unwrap().item().unwrap();
    assert!((p - 1.0).abs() < 1e-3);
}
