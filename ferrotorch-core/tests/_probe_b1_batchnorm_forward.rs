//! Probe B1 — Issue #892: gpu_batchnorm_forward stub removal.
//!
//! BEFORE fix: gpu_batchnorm_forward returned Err(ShapeMismatch { expected: [0], got: [1] })
//!   unconditionally. The PTX kernel existed but the Rust wrapper was a stub.
//!
//! AFTER fix: real two-pass PTX kernel (pass-1 reduce mean+var, pass-2 normalize).
//!   Input (4, 3, 8, 8), gamma=1, beta=0, training=true, momentum=0.1, eps=1e-5.
//!   Per-channel mean of output must be ~0 and std ~1 (gamma=1, beta=0, training mode).
//!
//! This probe is a permanent regression sentinel: if the stub returns or the kernel
//! regresses to NaN/wrong output, this probe detects it.

#![cfg(feature = "gpu")]

use std::sync::Once;

static GPU_INIT: Once = Once::new();

fn ensure_cuda() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialize for probe B1 batchnorm");
    });
}

/// Probe: AFTER fix the kernel succeeds and output has ~zero mean / ~unit std per channel.
#[test]
fn probe_b1_batchnorm_forward_after() {
    ensure_cuda();
    use ferrotorch_gpu::device::GpuDevice;
    use ferrotorch_gpu::transfer::{cpu_to_gpu, gpu_to_cpu};

    let dev = GpuDevice::new(0).expect("GpuDevice::new(0)");

    let n = 4usize;
    let c = 3usize;
    let h = 8usize;
    let w = 8usize;
    let spatial = h * w;
    let total_per_ch = n * spatial;

    // Input: values spread so per-channel mean != 0, std != 1 before normalisation.
    let input_data: Vec<f32> = (0..n * c * h * w)
        .map(|i| (i as f32 * 0.07) - 3.0)
        .collect();
    let weight = vec![1.0f32; c]; // gamma = 1
    let bias = vec![0.0f32; c];   // beta  = 0
    let running_mean = vec![0.0f32; c];
    let running_var = vec![1.0f32; c];

    let inp = cpu_to_gpu(&input_data, &dev).unwrap();
    let wt = cpu_to_gpu(&weight, &dev).unwrap();
    let bi = cpu_to_gpu(&bias, &dev).unwrap();
    let mut rm = cpu_to_gpu(&running_mean, &dev).unwrap();
    let mut rv = cpu_to_gpu(&running_var, &dev).unwrap();

    // BEFORE fix: Err(ShapeMismatch { expected: [0], got: [1] })
    // AFTER  fix: Ok((output, save_mean, save_invstd))
    let (out, _save_mean, _save_invstd) = ferrotorch_gpu::kernels::gpu_batchnorm_forward(
        &inp, &wt, &bi, &mut rm, &mut rv, c, spatial, 1e-5, 0.1, true, &dev,
    )
    .expect("AFTER fix #892: gpu_batchnorm_forward must return Ok");

    let actual = gpu_to_cpu(&out, &dev).unwrap();
    assert_eq!(actual.len(), n * c * h * w);

    // Per-channel mean must be ~0 and std ~1 (gamma=1, beta=0, training mode).
    for ch in 0..c {
        let mut ch_vals: Vec<f32> = Vec::with_capacity(total_per_ch);
        for bi in 0..n {
            for hi in 0..h {
                for wi in 0..w {
                    let flat = bi * c * h * w + ch * h * w + hi * w + wi;
                    ch_vals.push(actual[flat]);
                }
            }
        }
        let mean: f32 = ch_vals.iter().sum::<f32>() / ch_vals.len() as f32;
        let var: f32 = ch_vals.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / ch_vals.len() as f32;
        let std = var.sqrt();
        assert!(
            mean.abs() < 1e-3,
            "probe B1 #892 AFTER: channel {ch} mean={mean:.6}, expected ~0"
        );
        assert!(
            (std - 1.0).abs() < 1e-2,
            "probe B1 #892 AFTER: channel {ch} std={std:.6}, expected ~1"
        );
    }
}
