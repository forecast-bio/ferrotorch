//! GPU 2-D FFT tests on RTX 3090. (#634)
//!
//! Validates the new `gpu_fft2_c2c_f32` / `gpu_fft2_c2c_f64` cufftPlan2d
//! wrappers via `fft2` / `ifft2` round-trips against the CPU reference.

#![cfg(feature = "cuda")]

use ferrotorch_core::fft::{fft2, ifft2};
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
fn fft2_4x4_matches_cpu() {
    ensure_cuda();
    let mut data = Vec::new();
    for i in 0..16 {
        data.push(i as f32);
        data.push((i as f32) * 0.5);
    }
    let cpu = cpu_t_f32(&data, &[4, 4, 2]);
    let gpu = cpu.clone().to(Device::Cuda(0)).unwrap();

    let cpu_out = fft2(&cpu).unwrap();
    let gpu_out = fft2(&gpu).unwrap();
    assert!(gpu_out.is_cuda(), "fft2 should keep output on GPU");
    assert_eq!(gpu_out.shape(), &[4, 4, 2]);

    let cpu_data = cpu_out.data().unwrap();
    let gpu_data = gpu_out.cpu().unwrap().data().unwrap().to_vec();
    for i in 0..gpu_data.len() {
        assert!(
            (gpu_data[i] - cpu_data[i]).abs() < 1e-3,
            "i={i}: gpu={} cpu={}",
            gpu_data[i],
            cpu_data[i]
        );
    }
}

#[test]
fn fft2_ifft2_round_trip_f32() {
    ensure_cuda();
    let mut data = Vec::new();
    for i in 0..16 {
        data.push((i as f32) * 0.25);
        data.push(((i as f32) * 0.1).sin());
    }
    let original = cpu_t_f32(&data, &[4, 4, 2]).to(Device::Cuda(0)).unwrap();

    let f = fft2(&original).unwrap();
    let back = ifft2(&f).unwrap();
    assert!(back.is_cuda());

    let original_host = original.cpu().unwrap().data().unwrap().to_vec();
    let back_host = back.cpu().unwrap().data().unwrap().to_vec();
    for i in 0..back_host.len() {
        assert!(
            (back_host[i] - original_host[i]).abs() < 1e-3,
            "i={i}: back={} original={}",
            back_host[i],
            original_host[i]
        );
    }
}

#[test]
fn fft2_f64_round_trip() {
    ensure_cuda();
    let mut data = Vec::new();
    for i in 0..16 {
        data.push((i as f64) * 0.5);
        data.push(0.0);
    }
    let original = cpu_t_f64(&data, &[4, 4, 2]).to(Device::Cuda(0)).unwrap();
    let f = fft2(&original).unwrap();
    let back = ifft2(&f).unwrap();
    let original_host = original.cpu().unwrap().data().unwrap().to_vec();
    let back_host = back.cpu().unwrap().data().unwrap().to_vec();
    for i in 0..back_host.len() {
        assert!(
            (back_host[i] - original_host[i]).abs() < 1e-9,
            "i={i}: back={} original={}",
            back_host[i],
            original_host[i]
        );
    }
}

#[test]
fn fft2_8x8_matches_cpu() {
    ensure_cuda();
    let mut data = Vec::new();
    for i in 0..64 {
        data.push(((i as f32) * 0.1).cos());
        data.push(((i as f32) * 0.1).sin());
    }
    let cpu = cpu_t_f32(&data, &[8, 8, 2]);
    let gpu = cpu.clone().to(Device::Cuda(0)).unwrap();
    let cpu_out = fft2(&cpu).unwrap();
    let gpu_out = fft2(&gpu).unwrap().cpu().unwrap();
    let cd = cpu_out.data().unwrap();
    let gd = gpu_out.data().unwrap();
    for i in 0..cd.len() {
        assert!(
            (gd[i] - cd[i]).abs() < 1e-3,
            "i={i}: gpu={} cpu={}",
            gd[i],
            cd[i]
        );
    }
}

#[test]
fn fft2_returns_gpu_tensor() {
    ensure_cuda();
    let data = vec![0.0_f32; 16 * 2];
    let gpu = cpu_t_f32(&data, &[4, 4, 2]).to(Device::Cuda(0)).unwrap();
    let out = fft2(&gpu).unwrap();
    assert!(matches!(out.device(), Device::Cuda(0)));
}
