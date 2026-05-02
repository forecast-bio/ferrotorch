//! GPU `linalg::matrix_norm` (Frobenius) integration tests. (#576)
//!
//! Composes existing GPU primitives (mul → reduce_sum → sqrt) — verify the
//! result lands on GPU and matches the CPU implementation.
//!
//! `/rust-gpu-discipline` notes:
//! - The full pipeline runs on GPU (no host bounce). Inputs are constructed
//!   on GPU, the result tensor stays on GPU, and we only materialise via
//!   `.cpu()` for the value comparison.

#![cfg(feature = "cuda")]

use ferrotorch_core::linalg::matrix_norm;
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
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor f32")
}

fn cpu_t_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu tensor f64")
}

#[test]
fn matrix_norm_f32_matches_cpu() {
    ensure_cuda();
    // Frobenius norm of [[1, 2], [3, 4]] = sqrt(1+4+9+16) = sqrt(30) ≈ 5.477
    let a_cpu = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).expect("a -> gpu");

    let out_gpu = matrix_norm(&a_gpu).expect("gpu matrix_norm");
    assert!(out_gpu.is_cuda(), "result must remain on GPU");
    assert_eq!(out_gpu.shape(), &[] as &[usize], "norm is 0-dimensional");

    let cpu_val = matrix_norm(&a_cpu)
        .expect("cpu matrix_norm")
        .data()
        .unwrap()[0];
    let gpu_val = out_gpu.cpu().expect(".cpu()").data().unwrap()[0];
    let expected = 30.0_f32.sqrt();
    assert!(
        (gpu_val - cpu_val).abs() < 1e-5,
        "gpu={} cpu={}",
        gpu_val,
        cpu_val
    );
    assert!((gpu_val - expected).abs() < 1e-5, "gpu={}", gpu_val);
}

#[test]
fn matrix_norm_f32_rectangular() {
    ensure_cuda();
    // 3x2 — exercise the m != n path.
    let data: Vec<f32> = vec![1.0, -2.0, 3.0, -4.0, 5.0, -6.0];
    let a_cpu = cpu_t_f32(&data, &[3, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let cpu_val = matrix_norm(&a_cpu).unwrap().data().unwrap()[0];
    let gpu_val = matrix_norm(&a_gpu).unwrap().cpu().unwrap().data().unwrap()[0];
    assert!(
        (gpu_val - cpu_val).abs() < 1e-5,
        "gpu={} cpu={}",
        gpu_val,
        cpu_val
    );
}

#[test]
fn matrix_norm_f64_matches_cpu() {
    ensure_cuda();
    let data: Vec<f64> = vec![0.5, 1.5, 2.5, 3.5, 4.5, 5.5];
    let a_cpu = cpu_t_f64(&data, &[2, 3]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let out_gpu = matrix_norm(&a_gpu).expect("gpu matrix_norm f64");
    assert!(out_gpu.is_cuda());

    let cpu_val = matrix_norm(&a_cpu).unwrap().data().unwrap()[0];
    let gpu_val = out_gpu.cpu().unwrap().data().unwrap()[0];
    assert!(
        (gpu_val - cpu_val).abs() < 1e-12,
        "gpu={} cpu={}",
        gpu_val,
        cpu_val
    );
}

#[test]
fn matrix_norm_zero_matrix() {
    ensure_cuda();
    let a_cpu = cpu_t_f32(&[0.0; 9], &[3, 3]);
    let a_gpu = a_cpu.to(Device::Cuda(0)).unwrap();
    let v = matrix_norm(&a_gpu).unwrap().cpu().unwrap().data().unwrap()[0];
    assert_eq!(v, 0.0);
}

#[test]
fn matrix_norm_rejects_non_2d() {
    ensure_cuda();
    let a_cpu = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let a_gpu = a_cpu.to(Device::Cuda(0)).unwrap();
    let err = matrix_norm(&a_gpu).unwrap_err();
    matches!(
        err,
        ferrotorch_core::error::FerrotorchError::InvalidArgument { .. }
    );
}
