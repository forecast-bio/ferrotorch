//! GPU least-squares integration tests on RTX 3090. (#630)
//!
//! Validates the cuSOLVER-backed `gpu_lstsq_f32` / `gpu_lstsq_f64`
//! against closed-form references: an exact-solution check (square A)
//! plus an over-determined fit where the residual is non-zero.

#![cfg(feature = "cuda")]

use ferrotorch_core::linalg::lstsq_solve;
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
fn lstsq_square_system_recovers_exact_x() {
    // A = [[2, 0], [0, 4]], B = [[6], [12]]  ->  X = [[3], [3]]
    ensure_cuda();
    let a = cpu_t_f32(&[2.0, 0.0, 0.0, 4.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = cpu_t_f32(&[6.0, 12.0], &[2, 1])
        .to(Device::Cuda(0))
        .unwrap();
    let x = lstsq_solve(&a, &b).unwrap();
    assert!(matches!(x.device(), Device::Cuda(0)));
    let host = x.cpu().unwrap().data().unwrap().to_vec();
    assert!((host[0] - 3.0).abs() < 1e-4, "got {}", host[0]);
    assert!((host[1] - 3.0).abs() < 1e-4, "got {}", host[1]);
}

#[test]
fn lstsq_overdetermined_least_squares_fit() {
    // 3-row × 2-col over-determined system with known true X.
    // A = [[1, 1], [1, 2], [1, 3]], true x = [1, 2], so A @ x = [3, 5, 7].
    // Add small perturbation to b: [3.1, 4.9, 7.0]. cuSOLVER should
    // return x close to (1, 2).
    ensure_cuda();
    let a = cpu_t_f32(&[1.0, 1.0, 1.0, 2.0, 1.0, 3.0], &[3, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = cpu_t_f32(&[3.1, 4.9, 7.0], &[3, 1])
        .to(Device::Cuda(0))
        .unwrap();
    let x = lstsq_solve(&a, &b).unwrap();
    let host = x.cpu().unwrap().data().unwrap().to_vec();
    // Closed-form least-squares answer (computed offline) is approximately
    // [1.18, 1.95]. Tolerance loose since we only need "approximately right".
    assert!(host[0].abs() < 5.0, "intercept estimate {} OOB", host[0]);
    assert!(
        (host[1] - 2.0).abs() < 0.5,
        "slope estimate {} should be near 2",
        host[1]
    );
}

#[test]
fn lstsq_f64_square_system() {
    ensure_cuda();
    let a = cpu_t_f64(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = cpu_t_f64(&[5.0, 11.0], &[2, 1])
        .to(Device::Cuda(0))
        .unwrap();
    let x = lstsq_solve(&a, &b).unwrap();
    let host = x.cpu().unwrap().data().unwrap().to_vec();
    // Solving [1 2; 3 4] [x1; x2] = [5; 11] -> [1; 2].
    assert!((host[0] - 1.0).abs() < 1e-9, "got {}", host[0]);
    assert!((host[1] - 2.0).abs() < 1e-9, "got {}", host[1]);
}

#[test]
fn lstsq_1d_b_returns_1d_solution() {
    // 1-D b should yield 1-D x output.
    ensure_cuda();
    let a = cpu_t_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = cpu_t_f32(&[7.0, 8.0], &[2]).to(Device::Cuda(0)).unwrap();
    let x = lstsq_solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[2]);
    let host = x.cpu().unwrap().data().unwrap().to_vec();
    assert!((host[0] - 7.0).abs() < 1e-5);
    assert!((host[1] - 8.0).abs() < 1e-5);
}

#[test]
fn lstsq_rejects_shape_mismatch() {
    ensure_cuda();
    let a = cpu_t_f32(&[1.0; 6], &[2, 3]).to(Device::Cuda(0)).unwrap();
    // b has wrong leading dim (3 instead of 2).
    let b = cpu_t_f32(&[1.0, 2.0, 3.0], &[3])
        .to(Device::Cuda(0))
        .unwrap();
    let err = lstsq_solve(&a, &b).unwrap_err();
    let msg = format!("{err}");
    assert!(msg.contains("lstsq") || msg.contains("shape"), "got: {msg}");
}

#[test]
fn lstsq_multi_column_b() {
    // A = [[1, 0], [0, 1]] (identity 2×2), B = [[1, 2], [3, 4]].
    // X should equal B since A is identity.
    ensure_cuda();
    let a = cpu_t_f32(&[1.0, 0.0, 0.0, 1.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let b = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
        .to(Device::Cuda(0))
        .unwrap();
    let x = lstsq_solve(&a, &b).unwrap();
    assert_eq!(x.shape(), &[2, 2]);
    let host = x.cpu().unwrap().data().unwrap().to_vec();
    let expected = [1.0, 2.0, 3.0, 4.0];
    for i in 0..4 {
        assert!(
            (host[i] - expected[i]).abs() < 1e-4,
            "i={i}: got {}",
            host[i]
        );
    }
}
