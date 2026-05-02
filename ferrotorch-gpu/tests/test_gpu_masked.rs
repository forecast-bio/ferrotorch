//! GPU `masked_sum` / `masked_mean` integration tests via cuBLAS+reduce_sum. (#597)
//!
//! Validates the lowering: `masked_sum` = `sum(data * mask_as_float)`,
//! `masked_mean` = `masked_sum / count_valid` (with NaN on all-masked).

#![cfg(feature = "cuda")]

use ferrotorch_core::masked::{MaskedTensor, masked_max, masked_mean, masked_min, masked_sum};
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
fn masked_sum_f32_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let mask = vec![true, false, true, true, false, true];

    let cpu_sum = masked_sum(&MaskedTensor::new(data.clone(), mask.clone()).unwrap())
        .unwrap()
        .item()
        .unwrap();
    // 1 + 3 + 4 + 6 = 14
    assert!((cpu_sum - 14.0).abs() < 1e-5);

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let gpu_mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let gpu_sum = masked_sum(&gpu_mt).unwrap().cpu().unwrap().item().unwrap();
    assert!(
        (gpu_sum - cpu_sum).abs() < 1e-4,
        "gpu_sum={gpu_sum} cpu_sum={cpu_sum}"
    );
}

#[test]
fn masked_sum_f64_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f64(&[10.5, 20.5, 30.5, 40.5], &[4]);
    let mask = vec![true, true, false, true];
    let expected = 10.5 + 20.5 + 40.5;

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let result = masked_sum(&mt).unwrap().cpu().unwrap().item().unwrap();
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn masked_sum_all_masked_returns_zero() {
    ensure_cuda();
    let data = cpu_t_f32(&[1.0, 2.0, 3.0], &[3]);
    let mask = vec![false, false, false];
    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let result = masked_sum(&mt).unwrap().cpu().unwrap().item().unwrap();
    assert!(result.abs() < 1e-7);
}

#[test]
fn masked_mean_f32_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f32(&[1.0, 2.0, 3.0, 4.0], &[4]);
    let mask = vec![true, true, false, true];
    let cpu_mean = (1.0 + 2.0 + 4.0) / 3.0;

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let gpu_mean = masked_mean(&mt).unwrap().item().unwrap();
    assert!(
        (gpu_mean - cpu_mean).abs() < 1e-5,
        "gpu={gpu_mean} cpu={cpu_mean}"
    );
}

#[test]
fn masked_mean_all_masked_is_nan() {
    ensure_cuda();
    let data = cpu_t_f32(&[1.0, 2.0, 3.0], &[3]);
    let mask = vec![false, false, false];
    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let v = masked_mean(&mt).unwrap().item().unwrap();
    assert!(v.is_nan());
}

// ===========================================================================
// GPU masked_min / masked_max via inf-fill + reduce_min/max kernels (#627)
// ===========================================================================

#[test]
fn masked_min_f32_gpu_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f32(&[5.0, -3.0, 7.0, 1.0, 9.0, -8.0], &[2, 3]);
    // Mask out positions 1 and 5 (values -3.0 and -8.0). Visible: {5, 7, 1, 9}.
    let mask = vec![true, false, true, true, true, false];

    let cpu_min = masked_min(&MaskedTensor::new(data.clone(), mask.clone()).unwrap())
        .unwrap()
        .item()
        .unwrap();
    assert!((cpu_min - 1.0).abs() < 1e-7);

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let gpu_mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let gpu_min = masked_min(&gpu_mt).unwrap().cpu().unwrap().item().unwrap();
    assert!(
        (gpu_min - cpu_min).abs() < 1e-5,
        "gpu_min={gpu_min} cpu_min={cpu_min}"
    );
}

#[test]
fn masked_max_f32_gpu_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f32(&[5.0, 100.0, 7.0, 1.0, 9.0, 200.0], &[2, 3]);
    // Mask out positions 1 and 5 (values 100, 200). Visible: {5, 7, 1, 9}.
    let mask = vec![true, false, true, true, true, false];

    let cpu_max = masked_max(&MaskedTensor::new(data.clone(), mask.clone()).unwrap())
        .unwrap()
        .item()
        .unwrap();
    assert!((cpu_max - 9.0).abs() < 1e-7);

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let gpu_mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let gpu_max = masked_max(&gpu_mt).unwrap().cpu().unwrap().item().unwrap();
    assert!(
        (gpu_max - cpu_max).abs() < 1e-5,
        "gpu_max={gpu_max} cpu_max={cpu_max}"
    );
}

#[test]
fn masked_min_f64_gpu_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f64(&[10.5, -100.0, 30.5, 40.5, -50.0], &[5]);
    let mask = vec![true, false, true, true, false];
    let expected = 10.5_f64;

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let result = masked_min(&mt).unwrap().cpu().unwrap().item().unwrap();
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn masked_max_f64_gpu_matches_cpu() {
    ensure_cuda();
    let data = cpu_t_f64(&[10.5, 1000.0, 30.5, 40.5, 999.0], &[5]);
    let mask = vec![true, false, true, true, false];
    let expected = 40.5_f64;

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let result = masked_max(&mt).unwrap().cpu().unwrap().item().unwrap();
    assert!((result - expected).abs() < 1e-10);
}

#[test]
fn masked_min_max_all_masked_returns_nan_on_gpu() {
    ensure_cuda();
    let data = cpu_t_f32(&[1.0, 2.0, 3.0], &[3]);
    let gpu_data = data.to(Device::Cuda(0)).unwrap();

    let all_masked = MaskedTensor::new(gpu_data.clone(), vec![false; 3]).unwrap();
    assert!(masked_min(&all_masked).unwrap().item().unwrap().is_nan());
    assert!(masked_max(&all_masked).unwrap().item().unwrap().is_nan());
}

#[test]
fn masked_min_max_large_input_matches_cpu() {
    // Stress test: 4096 elements, half masked, GPU path should match CPU.
    ensure_cuda();
    let n = 4096;
    let host: Vec<f32> = (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.5).collect();
    // Mask every other element.
    let mask: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();

    let data = cpu_t_f32(&host, &[n]);
    let cpu_min = masked_min(&MaskedTensor::new(data.clone(), mask.clone()).unwrap())
        .unwrap()
        .item()
        .unwrap();
    let cpu_max = masked_max(&MaskedTensor::new(data.clone(), mask.clone()).unwrap())
        .unwrap()
        .item()
        .unwrap();

    let gpu_data = data.to(Device::Cuda(0)).unwrap();
    let mt = MaskedTensor::new(gpu_data, mask).unwrap();
    let gpu_min = masked_min(&mt).unwrap().cpu().unwrap().item().unwrap();
    let gpu_max = masked_max(&mt).unwrap().cpu().unwrap().item().unwrap();

    assert!(
        (gpu_min - cpu_min).abs() < 1e-3,
        "min: gpu={gpu_min} cpu={cpu_min}"
    );
    assert!(
        (gpu_max - cpu_max).abs() < 1e-3,
        "max: gpu={gpu_max} cpu={cpu_max}"
    );
}

// ===========================================================================
// Tensor::amin / Tensor::amax — public global-min/max surface (#627)
// ===========================================================================

#[test]
fn tensor_amin_amax_f32_gpu_matches_cpu() {
    ensure_cuda();
    let host: Vec<f32> = vec![3.5, -2.0, 7.5, 1.0, -10.0, 100.0];
    let cpu = cpu_t_f32(&host, &[6]);
    let cpu_min = cpu.amin().unwrap().item().unwrap();
    let cpu_max = cpu.amax().unwrap().item().unwrap();
    assert!((cpu_min - (-10.0)).abs() < 1e-7);
    assert!((cpu_max - 100.0).abs() < 1e-7);

    let gpu = cpu.to(Device::Cuda(0)).unwrap();
    let gpu_min = gpu.amin().unwrap().cpu().unwrap().item().unwrap();
    let gpu_max = gpu.amax().unwrap().cpu().unwrap().item().unwrap();
    assert!((gpu_min - cpu_min).abs() < 1e-5, "gpu_min={gpu_min}");
    assert!((gpu_max - cpu_max).abs() < 1e-5, "gpu_max={gpu_max}");
}

#[test]
fn tensor_amin_amax_f64_gpu_matches_cpu() {
    ensure_cuda();
    let host: Vec<f64> = (0..1000).map(|i| ((i as f64) - 500.0) * 0.001).collect();
    let cpu = cpu_t_f64(&host, &[1000]);
    let gpu = cpu.clone().to(Device::Cuda(0)).unwrap();

    let gpu_min = gpu.amin().unwrap().cpu().unwrap().item().unwrap();
    let gpu_max = gpu.amax().unwrap().cpu().unwrap().item().unwrap();
    let cpu_min = cpu.amin().unwrap().item().unwrap();
    let cpu_max = cpu.amax().unwrap().item().unwrap();
    assert!((gpu_min - cpu_min).abs() < 1e-10);
    assert!((gpu_max - cpu_max).abs() < 1e-10);
}

#[test]
fn tensor_amin_returns_gpu_tensor() {
    // Result must stay on the input device (matches sum / mean contract).
    ensure_cuda();
    let cpu = cpu_t_f32(&[1.0, 2.0, 3.0], &[3]);
    let gpu = cpu.to(Device::Cuda(0)).unwrap();
    let result = gpu.amin().unwrap();
    assert!(matches!(result.device(), Device::Cuda(0)));
}

#[test]
fn tensor_amin_amax_large_input_gpu() {
    // 100 000 elements — exercises both kernel passes (>256 blocks).
    ensure_cuda();
    let n = 100_000;
    let host: Vec<f32> = (0..n).map(|i| ((i as i64 - 50_000) as f32) * 0.5).collect();
    let cpu = cpu_t_f32(&host, &[n]);
    let gpu = cpu.clone().to(Device::Cuda(0)).unwrap();
    let gpu_min = gpu.amin().unwrap().cpu().unwrap().item().unwrap();
    let gpu_max = gpu.amax().unwrap().cpu().unwrap().item().unwrap();
    let expected_min = -25_000.0_f32;
    let expected_max = 24_999.5_f32;
    assert!((gpu_min - expected_min).abs() < 1e-3, "gpu_min={gpu_min}");
    assert!((gpu_max - expected_max).abs() < 1e-3, "gpu_max={gpu_max}");
}
