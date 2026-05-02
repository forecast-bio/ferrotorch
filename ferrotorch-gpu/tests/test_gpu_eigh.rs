//! GPU `linalg::eigh` / `linalg::eigvalsh` integration tests. (#575)
//!
//! Exercises the full `linalg::eigh` and `linalg::eigvalsh` code paths on a
//! real CUDA device, verifying that the f32 and f64 GPU dispatch produces
//! eigenvalues that match the CPU implementation and eigenvectors that
//! reconstruct the input symmetric matrix.
//!
//! `/rust-gpu-discipline` notes:
//! - Inputs are constructed on GPU explicitly (`Tensor::to(Device::Cuda(0))`)
//!   and we assert the result tensors are also on GPU before reading them
//!   back. The GPU dispatch path stays GPU-resident: `gpu_eigh_*` takes
//!   `&CudaBuffer<T>`, clones via `memcpy_dtod`, and returns `CudaBuffer<T>`
//!   without bouncing through host.
//! - The result is materialised on host only via `.cpu()` at the end of each
//!   test for the value comparison — the explicit device boundary.

#![cfg(feature = "cuda")]

use ferrotorch_core::linalg::{eigh, eigvalsh};
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
fn eigh_f32_diagonal_matches_cpu() {
    ensure_cuda();
    // Diagonal: eigenvalues are the diagonal entries (sorted ascending).
    let a_cpu = cpu_t_f32(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], &[3, 3]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).expect("a -> gpu");

    let (w_gpu, q_gpu) = eigh(&a_gpu).expect("gpu eigh");
    assert!(w_gpu.is_cuda(), "eigenvalues must remain on GPU");
    assert!(q_gpu.is_cuda(), "eigenvectors must remain on GPU");
    assert_eq!(w_gpu.shape(), &[3]);
    assert_eq!(q_gpu.shape(), &[3, 3]);

    let w_host = w_gpu.cpu().expect(".cpu()").data().unwrap().to_vec();
    // Ascending: 1.0, 2.0, 3.0
    assert!((w_host[0] - 1.0).abs() < 1e-5, "w[0]={}", w_host[0]);
    assert!((w_host[1] - 2.0).abs() < 1e-5, "w[1]={}", w_host[1]);
    assert!((w_host[2] - 3.0).abs() < 1e-5, "w[2]={}", w_host[2]);
}

#[test]
fn eigh_f32_reconstructs_symmetric() {
    ensure_cuda();
    // A = [[4, 1], [1, 3]] symmetric. Verify A == Q diag(w) Q^T.
    let a_cpu = cpu_t_f32(&[4.0, 1.0, 1.0, 3.0], &[2, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let (w_gpu, q_gpu) = eigh(&a_gpu).unwrap();
    let w = w_gpu.cpu().unwrap().data().unwrap().to_vec();
    let q = q_gpu.cpu().unwrap().data().unwrap().to_vec();
    let n = 2;

    let mut recon = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += q[i * n + k] * w[k] * q[j * n + k];
            }
            recon[i * n + j] = acc;
        }
    }
    let a_data = a_cpu.data().unwrap();
    for i in 0..n * n {
        assert!(
            (recon[i] - a_data[i]).abs() < 1e-4,
            "recon[{i}]={} vs a={}",
            recon[i],
            a_data[i]
        );
    }
}

#[test]
fn eigh_f64_reconstructs_symmetric() {
    ensure_cuda();
    let a_cpu = cpu_t_f64(&[4.0, 1.0, 0.5, 1.0, 3.0, 0.25, 0.5, 0.25, 2.0], &[3, 3]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let (w_gpu, q_gpu) = eigh(&a_gpu).unwrap();
    assert!(w_gpu.is_cuda());
    assert!(q_gpu.is_cuda());

    let w = w_gpu.cpu().unwrap().data().unwrap().to_vec();
    let q = q_gpu.cpu().unwrap().data().unwrap().to_vec();
    let n = 3;

    let mut recon = vec![0.0f64; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut acc = 0.0;
            for k in 0..n {
                acc += q[i * n + k] * w[k] * q[j * n + k];
            }
            recon[i * n + j] = acc;
        }
    }
    let a_data = a_cpu.data().unwrap();
    for i in 0..n * n {
        assert!(
            (recon[i] - a_data[i]).abs() < 1e-10,
            "recon[{i}]={} vs a={}",
            recon[i],
            a_data[i]
        );
    }
}

#[test]
fn eigh_f32_matches_cpu_eigenvalues() {
    ensure_cuda();
    // Run eigh on both CPU and GPU, compare eigenvalue spectra.
    let data: Vec<f32> = vec![5.0, 1.0, 2.0, 1.0, 4.0, 0.5, 2.0, 0.5, 6.0];
    let a_cpu = cpu_t_f32(&data, &[3, 3]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let (w_cpu, _) = eigh(&a_cpu).unwrap();
    let (w_gpu, _) = eigh(&a_gpu).unwrap();
    assert!(w_gpu.is_cuda());

    let cpu_vals = w_cpu.data().unwrap().to_vec();
    let gpu_vals = w_gpu.cpu().unwrap().data().unwrap().to_vec();
    for i in 0..3 {
        assert!(
            (cpu_vals[i] - gpu_vals[i]).abs() < 1e-4,
            "w[{i}] cpu={} gpu={}",
            cpu_vals[i],
            gpu_vals[i]
        );
    }
}

#[test]
fn eigvalsh_f32_matches_eigh_first_return() {
    ensure_cuda();
    let a_cpu = cpu_t_f32(&[2.0, 1.0, 1.0, 2.0], &[2, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let w_only = eigvalsh(&a_gpu).expect("gpu eigvalsh");
    assert!(w_only.is_cuda());
    assert_eq!(w_only.shape(), &[2]);

    let (w_full, _q) = eigh(&a_gpu).unwrap();

    let only = w_only.cpu().unwrap().data().unwrap().to_vec();
    let full = w_full.cpu().unwrap().data().unwrap().to_vec();
    for i in 0..2 {
        assert!(
            (only[i] - full[i]).abs() < 1e-5,
            "eigvalsh[{i}]={} vs eigh.0[{i}]={}",
            only[i],
            full[i]
        );
    }
}

#[test]
fn eigvalsh_f64_matches_cpu() {
    ensure_cuda();
    let a_cpu = cpu_t_f64(&[7.0, 2.0, 2.0, 5.0], &[2, 2]);
    let a_gpu = a_cpu.clone().to(Device::Cuda(0)).unwrap();

    let w_cpu = eigvalsh(&a_cpu).unwrap();
    let w_gpu = eigvalsh(&a_gpu).unwrap();
    assert!(w_gpu.is_cuda());

    let cpu_vals = w_cpu.data().unwrap().to_vec();
    let gpu_vals = w_gpu.cpu().unwrap().data().unwrap().to_vec();
    for i in 0..2 {
        assert!(
            (cpu_vals[i] - gpu_vals[i]).abs() < 1e-10,
            "w[{i}] cpu={} gpu={}",
            cpu_vals[i],
            gpu_vals[i]
        );
    }
}

#[test]
fn eigh_rejects_non_square() {
    ensure_cuda();
    let a_cpu = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let a_gpu = a_cpu.to(Device::Cuda(0)).unwrap();
    let err = eigh(&a_gpu).unwrap_err();
    matches!(
        err,
        ferrotorch_core::error::FerrotorchError::InvalidArgument { .. }
    );
}
