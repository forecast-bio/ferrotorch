//! GPU solve device-resident path on RTX 3090. (#632)
//!
//! Validates the new `gpu_solve_f32_dev` / `gpu_solve_f64_dev` cuSOLVER
//! getrf+getrs path against closed-form references. The bounce-drop
//! refactor changed `Backend::solve_f32` / `solve_f64` to call these
//! directly instead of the `gpu_to_cpu → cusolver_on_host_buffers →
//! cpu_to_gpu` sandwich the previous impl used. Behavior must match.

#![cfg(feature = "cuda")]

use ferrotorch_core::gpu_dispatch::gpu_backend;
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

fn solve_via_backend_f32(a: &Tensor<f32>, b: &Tensor<f32>, n: usize, nrhs: usize) -> Vec<f32> {
    let backend = gpu_backend().expect("CUDA backend registered");
    let h = backend
        .solve_f32(a.gpu_handle().unwrap(), b.gpu_handle().unwrap(), n, nrhs)
        .unwrap();
    let out_shape = if nrhs == 1 { vec![n] } else { vec![n, nrhs] };
    Tensor::<f32>::from_storage(TensorStorage::gpu(h), out_shape, false)
        .unwrap()
        .cpu()
        .unwrap()
        .data()
        .unwrap()
        .to_vec()
}

fn solve_via_backend_f64(a: &Tensor<f64>, b: &Tensor<f64>, n: usize, nrhs: usize) -> Vec<f64> {
    let backend = gpu_backend().expect("CUDA backend registered");
    let h = backend
        .solve_f64(a.gpu_handle().unwrap(), b.gpu_handle().unwrap(), n, nrhs)
        .unwrap();
    let out_shape = if nrhs == 1 { vec![n] } else { vec![n, nrhs] };
    Tensor::<f64>::from_storage(TensorStorage::gpu(h), out_shape, false)
        .unwrap()
        .cpu()
        .unwrap()
        .data()
        .unwrap()
        .to_vec()
}

#[test]
fn solve_dev_2x2_recovers_x() {
    // [1 2; 3 4] [x1; x2] = [5; 11] -> [1; 2]
    ensure_cuda();
    let a = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = cpu_t_f32(&[5.0, 11.0], &[2, 1])
        .to(Device::Cuda(0))
        .unwrap();
    let x = solve_via_backend_f32(&a, &b, 2, 1);
    assert!((x[0] - 1.0).abs() < 1e-4, "got {}", x[0]);
    assert!((x[1] - 2.0).abs() < 1e-4, "got {}", x[1]);
}

#[test]
fn solve_dev_diagonal_system() {
    // diag(2, 3, 5) * X = [4, 6, 10] -> X = [2, 2, 2]
    ensure_cuda();
    let a = cpu_t_f32(&[2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 5.0], &[3, 3])
        .to(Device::Cuda(0))
        .unwrap();
    let b = cpu_t_f32(&[4.0, 6.0, 10.0], &[3, 1])
        .to(Device::Cuda(0))
        .unwrap();
    let x = solve_via_backend_f32(&a, &b, 3, 1);
    for &v in &x {
        assert!((v - 2.0).abs() < 1e-4, "got {v}");
    }
}

#[test]
fn solve_dev_f64_4x4() {
    // diag(1.5, 2.5, 3.5, 4.5) * X = [3, 5, 7, 9] -> [2, 2, 2, 2]
    ensure_cuda();
    let mut a_data = vec![0.0_f64; 16];
    for i in 0..4 {
        a_data[i * 4 + i] = 1.5 + i as f64;
    }
    let a = cpu_t_f64(&a_data, &[4, 4]).to(Device::Cuda(0)).unwrap();
    let b = cpu_t_f64(&[3.0, 5.0, 7.0, 9.0], &[4, 1])
        .to(Device::Cuda(0))
        .unwrap();
    let x = solve_via_backend_f64(&a, &b, 4, 1);
    for &v in &x {
        assert!((v - 2.0).abs() < 1e-9, "got {v}");
    }
}

#[test]
fn solve_dev_multi_column_b() {
    // Identity * B = B
    ensure_cuda();
    let a = cpu_t_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = cpu_t_f32(&[10.0, 20.0, 30.0, 40.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let x = solve_via_backend_f32(&a, &b, 2, 2);
    let expected = [10.0, 20.0, 30.0, 40.0];
    for (i, &v) in x.iter().enumerate() {
        assert!((v - expected[i]).abs() < 1e-4, "i={i}: got {v}");
    }
}

#[test]
fn solve_dev_singular_returns_error() {
    // Singular matrix [[1,1],[1,1]] should make getrf fail.
    ensure_cuda();
    let a = cpu_t_f32(&[1.0, 1.0, 1.0, 1.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = cpu_t_f32(&[1.0, 1.0], &[2, 1]).to(Device::Cuda(0)).unwrap();
    let backend = gpu_backend().expect("CUDA backend registered");
    let result = backend.solve_f32(a.gpu_handle().unwrap(), b.gpu_handle().unwrap(), 2, 1);
    assert!(result.is_err(), "expected error for singular matrix");
}
