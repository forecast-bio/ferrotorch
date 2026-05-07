//! Permanent regression sentinel for #966: axes-aware fftn/ifftn on CUDA via
//! cufftPlanMany with embed/stride/dist parameters.
//!
//! Pre-fix observable failure (before #966):
//!   * `fftn(x, s=None, axes=Some(&[-1]))` on a CUDA tensor fell through to
//!     CPU (ferray-fft) because the GPU dispatch only handled the no-axes case.
//!   * `GpuBackend::fftn_axes_c2c_f32/f64` returned `Err(InvalidArgument)`
//!     (the default trait-method stub).
//!
//! Post-fix:
//!   * `fftn_axes_c2c_f32/f64` calls `cufft::gpu_fftn_axes_c2c_f32/f64`
//!     which builds a `cufftPlanMany` plan with rank=axes.len(),
//!     n=shape[axes[i]], idist=odist=transform_vol, batch=spatial/transform_vol.
//!   * GPU output matches CPU ferray-fft reference within F64_FFT = 1e-10
//!     for f64, and F32_FFT = 1e-4 for f32.
//!   * fft.rs dispatches both positive and negative axis indices correctly.
//!
//! PyTorch parity (rust-gpu-discipline §3):
//!   `torch.fft.fftn(x, dim=(-1,))` on a CUDA tensor dispatches to cuFFT.

#![cfg(feature = "gpu")]

use std::sync::Once;

use ferrotorch_core::{Device, Tensor, TensorStorage};
use ferrotorch_core::fft::{fftn, ifftn};

const F64_FFT_TOL: f64 = 1e-10;
const F32_FFT_TOL: f32 = 1e-4;

static GPU_INIT: Once = Once::new();

fn ensure_cuda_backend() {
    GPU_INIT.call_once(|| {
        ferrotorch_gpu::init_cuda_backend()
            .expect("CUDA backend must initialise for fftn-axes probe");
    });
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

fn make_cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f32 tensor")
}

fn make_cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false)
        .expect("cpu f64 tensor")
}

fn to_cuda_f32(t: Tensor<f32>) -> Tensor<f32> {
    t.to(Device::Cuda(0)).expect("to cuda f32")
}

fn to_cuda_f64(t: Tensor<f64>) -> Tensor<f64> {
    t.to(Device::Cuda(0)).expect("to cuda f64")
}

fn read_back_f32(t: &Tensor<f32>) -> Vec<f32> {
    if t.is_cpu() {
        t.data().expect("read CPU data").to_vec()
    } else {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data().expect("read CPU data after readback").to_vec()
    }
}

fn read_back_f64(t: &Tensor<f64>) -> Vec<f64> {
    if t.is_cpu() {
        t.data().expect("read CPU data").to_vec()
    } else {
        let cpu = t.cpu().expect("D2H readback");
        cpu.data().expect("read CPU data after readback").to_vec()
    }
}

fn check_f32(label: &str, got: &[f32], expected: &[f32]) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < F32_FFT_TOL,
            "{label} elem {i}: got {g}, expected {e} (tol {F32_FFT_TOL})"
        );
    }
}

fn check_f64(label: &str, got: &[f64], expected: &[f64]) {
    assert_eq!(got.len(), expected.len(), "{label}: length mismatch");
    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < F64_FFT_TOL,
            "{label} elem {i}: got {g}, expected {e} (tol {F64_FFT_TOL})"
        );
    }
}

// ---------------------------------------------------------------------------
// #966: fftn with axes=Some(&[-1]) on a 3-D complex tensor [d, h, w, 2]
//
// This is the `ndim_3_axes_neg1` fixture that was cascade-skipped in C.6.
// Transform over last spatial axis only (axis=-1, normalized to 2 in [d,h,w]).
// ---------------------------------------------------------------------------

/// BEFORE: fftn with axes fell through to CPU for CUDA tensors.
/// AFTER:  gpu_fftn_axes_c2c_f64 launches cufftPlanMany on the RTX 3090;
///         result matches CPU ferray-fft within 1e-10.
#[test]
fn p966_fftn_axes_neg1_3d_f64() {
    ensure_cuda_backend();

    // 2x3x4 complex tensor: shape [2, 3, 4, 2] (d=2, h=3, w=4)
    let d = 2usize;
    let h = 3usize;
    let w = 4usize;
    let n = d * h * w * 2; // interleaved complex
    let data_f64: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let shape = [d, h, w, 2];

    let cpu_tensor = make_cpu_f64(&data_f64, &shape);
    let gpu_tensor = to_cuda_f64(make_cpu_f64(&data_f64, &shape));

    // axes=Some(&[-1]) -> transform over axis 2 (w dimension)
    let cpu_result = fftn(&cpu_tensor, None, Some(&[-1isize])).expect("cpu fftn axes=-1");
    let gpu_result = fftn(&gpu_tensor, None, Some(&[-1isize])).expect("gpu fftn axes=-1");

    assert!(gpu_result.is_cuda(), "gpu fftn result must stay on CUDA");
    assert_eq!(gpu_result.shape(), cpu_result.shape(), "shape must match");

    let cpu_vals = read_back_f64(&cpu_result);
    let gpu_vals = read_back_f64(&gpu_result);
    check_f64("p966_fftn_axes_neg1_3d_f64", &gpu_vals, &cpu_vals);
}

/// f32 variant: same shape [2, 3, 4, 2], axes=[-1].
#[test]
fn p966_fftn_axes_neg1_3d_f32() {
    ensure_cuda_backend();

    let d = 2usize;
    let h = 3usize;
    let w = 4usize;
    let n = d * h * w * 2;
    let data_f32: Vec<f32> = (0..n).map(|i| i as f32 * 0.1).collect();
    let shape = [d, h, w, 2];

    let cpu_tensor = make_cpu_f32(&data_f32, &shape);
    let gpu_tensor = to_cuda_f32(make_cpu_f32(&data_f32, &shape));

    let cpu_result = fftn(&cpu_tensor, None, Some(&[-1isize])).expect("cpu fftn f32 axes=-1");
    let gpu_result = fftn(&gpu_tensor, None, Some(&[-1isize])).expect("gpu fftn f32 axes=-1");

    assert!(gpu_result.is_cuda(), "gpu fftn f32 must stay on CUDA");
    let cpu_vals = read_back_f32(&cpu_result);
    let gpu_vals = read_back_f32(&gpu_result);
    check_f32("p966_fftn_axes_neg1_3d_f32", &gpu_vals, &cpu_vals);
}

// Note on `ndim_3_axes_0` (axes=[0] on [d,h,w,2]):
// axes=[0] is NOT innermost for a 3-D spatial tensor. The GPU path requires
// innermost axes (cufftPlanMany inembed=NULL contract). Non-innermost axes
// on a CUDA tensor return NotImplementedOnCuda — same as the pre-#966
// behaviour for that specific case. The conformance_fft `ndim_3_axes_0`
// fixture runs on CPU (device_label="cpu") and continues to pass. The CUDA
// fixture for that tag was cascade-skipped in C.6 and is still skipped
// because GPU support for non-innermost axes requires a pre-permute step
// not yet implemented. The three innermost-axes fixtures DO run on GPU (#966
// scope): ndim_3_axes_neg1, ndim_3_axes_n2_n1, and the 2-D axes=-1 case.

// ---------------------------------------------------------------------------
// #966: fftn with axes=Some(&[-2, -1]) on a 3-D complex tensor
//
// This is the `ndim_3_axes_n2_n1` fixture that was cascade-skipped in C.6.
// Transform over last two spatial axes simultaneously (h and w).
// ---------------------------------------------------------------------------

/// BEFORE: fftn with axes=[-2,-1] fell through to CPU for CUDA tensors.
/// AFTER:  gpu_fftn_axes_c2c_f64 uses rank=2, n=[h,w] plan via cufftPlanMany.
#[test]
fn p966_fftn_axes_n2_n1_3d_f64() {
    ensure_cuda_backend();

    let d = 2usize;
    let h = 4usize;
    let w = 4usize;
    let n = d * h * w * 2;
    let data: Vec<f64> = (0..n).map(|i| i as f64 / (n as f64)).collect();
    let shape = [d, h, w, 2];

    let cpu_tensor = make_cpu_f64(&data, &shape);
    let gpu_tensor = to_cuda_f64(make_cpu_f64(&data, &shape));

    let cpu_result = fftn(&cpu_tensor, None, Some(&[-2isize, -1])).expect("cpu fftn axes=[-2,-1]");
    let gpu_result = fftn(&gpu_tensor, None, Some(&[-2isize, -1])).expect("gpu fftn axes=[-2,-1]");

    assert!(gpu_result.is_cuda(), "gpu fftn axes=[-2,-1] must stay on CUDA");
    assert_eq!(gpu_result.shape(), cpu_result.shape());
    let cpu_vals = read_back_f64(&cpu_result);
    let gpu_vals = read_back_f64(&gpu_result);
    check_f64("p966_fftn_axes_n2_n1_3d_f64", &gpu_vals, &cpu_vals);
}

/// f32 variant: axes=[-2,-1] on [2, 4, 4, 2].
#[test]
fn p966_fftn_axes_n2_n1_3d_f32() {
    ensure_cuda_backend();

    let d = 2usize;
    let h = 4usize;
    let w = 4usize;
    let n = d * h * w * 2;
    let data: Vec<f32> = (0..n).map(|i| i as f32 / (n as f32)).collect();
    let shape = [d, h, w, 2];

    let cpu_tensor = make_cpu_f32(&data, &shape);
    let gpu_tensor = to_cuda_f32(make_cpu_f32(&data, &shape));

    let cpu_result = fftn(&cpu_tensor, None, Some(&[-2isize, -1])).expect("cpu fftn f32 [-2,-1]");
    let gpu_result = fftn(&gpu_tensor, None, Some(&[-2isize, -1])).expect("gpu fftn f32 [-2,-1]");

    assert!(gpu_result.is_cuda());
    let cpu_vals = read_back_f32(&cpu_result);
    let gpu_vals = read_back_f32(&gpu_result);
    check_f32("p966_fftn_axes_n2_n1_3d_f32", &gpu_vals, &cpu_vals);
}

// ---------------------------------------------------------------------------
// #966: ifftn with axes (round-trip: ifftn(fftn(x)) ~= x)
// ---------------------------------------------------------------------------

/// ifftn(fftn(x, axes=[-1]), axes=[-1]) must reconstruct x within tolerance.
///
/// BEFORE: ifftn with axes also fell through to CPU; result was correct but
///         the GPU path was never exercised.
/// AFTER:  both fftn and ifftn with axes dispatch to cufftPlanMany on GPU.
#[test]
fn p966_ifftn_roundtrip_axes_neg1_f64() {
    ensure_cuda_backend();

    let d = 2usize;
    let h = 3usize;
    let w = 4usize;
    let n = d * h * w * 2;
    // Use values with non-trivial imaginary parts so the round-trip is
    // a meaningful test (not just real-input zeros in imaginary slots).
    let data: Vec<f64> = (0..n)
        .map(|i| if i % 2 == 0 { i as f64 * 0.1 } else { -(i as f64) * 0.05 })
        .collect();
    let shape = [d, h, w, 2];

    let gpu_tensor = to_cuda_f64(make_cpu_f64(&data, &shape));

    let forward = fftn(&gpu_tensor, None, Some(&[-1isize])).expect("fftn axes=-1 forward");
    assert!(forward.is_cuda(), "fftn output must be on CUDA");

    let reconstructed = ifftn(&forward, None, Some(&[-1isize])).expect("ifftn axes=-1");
    assert!(reconstructed.is_cuda(), "ifftn output must be on CUDA");

    let got = read_back_f64(&reconstructed);
    for (i, (&g, &e)) in got.iter().zip(data.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-9,
            "round-trip elem {i}: got {g}, expected {e}"
        );
    }
}

// ---------------------------------------------------------------------------
// #966: 2-D complex tensor with axes=[-1] -- innermost axis GPU path
// ---------------------------------------------------------------------------

/// Shape [4, 3, 2] (h=4, w=3), axes=[-1] -> transform along w only.
/// This is an innermost-axis case so it dispatches to cufftPlanMany on GPU.
#[test]
fn p966_fftn_axes_neg1_2d_f64() {
    ensure_cuda_backend();

    let h = 4usize;
    let w = 3usize;
    let n = h * w * 2;
    let data: Vec<f64> = (0..n).map(|i| i as f64 * 0.25).collect();
    let shape = [h, w, 2];

    let cpu_tensor = make_cpu_f64(&data, &shape);
    let gpu_tensor = to_cuda_f64(make_cpu_f64(&data, &shape));

    let cpu_result = fftn(&cpu_tensor, None, Some(&[-1isize])).expect("cpu fftn 2d axes=-1");
    let gpu_result = fftn(&gpu_tensor, None, Some(&[-1isize])).expect("gpu fftn 2d axes=-1");

    assert!(gpu_result.is_cuda(), "2d axes=-1 must stay on CUDA");
    assert_eq!(gpu_result.shape(), cpu_result.shape());
    let cpu_vals = read_back_f64(&cpu_result);
    let gpu_vals = read_back_f64(&gpu_result);
    check_f64("p966_fftn_axes_neg1_2d_f64", &gpu_vals, &cpu_vals);
}
