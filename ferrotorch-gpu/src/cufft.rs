//! cuFFT-backed GPU 1-D FFT primitives. (#579)
//!
//! Each function is GPU-resident: takes `&CudaBuffer<T>`, returns
//! `CudaBuffer<T>`. The complex layout is the workspace convention —
//! interleaved `[re, im, re, im, ...]` in a real f32/f64 buffer — which
//! happens to match `cufftComplex` / `cufftDoubleComplex` byte-for-byte, so
//! we can hand cuFFT the same allocation without any reformat step.
//!
//! Plans are recreated per call; cuFFT's plan creation is light enough that
//! we don't bother caching today, and the lifetime story for cached plans
//! across multi-stream / multi-device usage is fiddly. Performance-critical
//! paths can lift planning out themselves.
//!
//! Inverse transforms apply `1/n` normalization in a follow-up kernel
//! launch, matching torch / numpy convention. cuFFT itself does not
//! normalize.

#![cfg(feature = "cuda")]

use std::ffi::c_int;

use cudarc::cufft::result as cufft_result;
use cudarc::cufft::sys as cufft_sys;
use cudarc::cufft::{CudaFft, FftDirection};
use cudarc::driver::DevicePtrMut;

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};

fn check_n_batch(n: usize, batch: usize, op: &'static str) -> GpuResult<()> {
    if n == 0 {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![1],
            got: vec![0],
        });
    }
    if batch == 0 {
        return Err(GpuError::ShapeMismatch {
            op,
            expected: vec![1],
            got: vec![0],
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Complex-to-complex (f32)
// ---------------------------------------------------------------------------

/// Forward (`inverse=false`) or inverse (`inverse=true`) 1-D complex-to-complex
/// FFT, batched. Layout: `[batch * n * 2]` interleaved `(re, im)` f32.
///
/// On `inverse=true`, the output is divided by `n` so that
/// `ifft(fft(x)) ≈ x` matches torch / numpy.
pub fn gpu_fft_c2c_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    n: usize,
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    check_n_batch(n, batch, "gpu_fft_c2c_f32")?;
    let total =
        batch
            .checked_mul(n)
            .and_then(|v| v.checked_mul(2))
            .ok_or(GpuError::ShapeMismatch {
                op: "gpu_fft_c2c_f32",
                expected: vec![batch, n, 2],
                got: vec![input.len()],
            })?;
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fft_c2c_f32",
            expected: vec![batch, n, 2],
            got: vec![input.len()],
        });
    }

    let stream = device.stream();
    let plan = CudaFft::plan_1d(
        n as i32,
        cufft_sys::cufftType::CUFFT_C2C,
        batch as i32,
        stream.clone(),
    )?;
    let mut out = crate::transfer::alloc_zeros_f32(total, device)?;

    // Clone input (cuFFT C2C is documented as writeable on `idata` —
    // pre-allocated copy keeps the input buffer pristine for the caller).
    let mut tmp = crate::transfer::alloc_zeros_f32(total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;

    let direction = if inverse {
        FftDirection::Inverse
    } else {
        FftDirection::Forward
    };
    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_c2c(
            plan.handle(),
            idata as *mut cufft_sys::cufftComplex,
            odata as *mut cufft_sys::cufftComplex,
            direction as c_int,
        )?;
    }

    if inverse {
        let scale = 1.0_f32 / n as f32;
        out = crate::kernels::gpu_scale(&out, scale, device)?;
    }
    Ok(out)
}

/// f64 variant of [`gpu_fft_c2c_f32`].
pub fn gpu_fft_c2c_f64(
    input: &CudaBuffer<f64>,
    batch: usize,
    n: usize,
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    check_n_batch(n, batch, "gpu_fft_c2c_f64")?;
    let total =
        batch
            .checked_mul(n)
            .and_then(|v| v.checked_mul(2))
            .ok_or(GpuError::ShapeMismatch {
                op: "gpu_fft_c2c_f64",
                expected: vec![batch, n, 2],
                got: vec![input.len()],
            })?;
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fft_c2c_f64",
            expected: vec![batch, n, 2],
            got: vec![input.len()],
        });
    }

    let stream = device.stream();
    let plan = CudaFft::plan_1d(
        n as i32,
        cufft_sys::cufftType::CUFFT_Z2Z,
        batch as i32,
        stream.clone(),
    )?;
    let mut out = crate::transfer::alloc_zeros_f64(total, device)?;
    let mut tmp = crate::transfer::alloc_zeros_f64(total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;

    let direction = if inverse {
        FftDirection::Inverse
    } else {
        FftDirection::Forward
    };
    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_z2z(
            plan.handle(),
            idata as *mut cufft_sys::cufftDoubleComplex,
            odata as *mut cufft_sys::cufftDoubleComplex,
            direction as c_int,
        )?;
    }

    if inverse {
        let scale = 1.0_f64 / n as f64;
        out = crate::kernels::gpu_scale_f64(&out, scale, device)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// 2-D complex-to-complex FFT (#634)
// ---------------------------------------------------------------------------
//
// Wraps `cufftPlan2d` for the unbatched 2-D case. Input/output layout:
// interleaved `[h, w, 2]` (re/im pairs), same convention as the existing
// 1-D ops. `inverse=true` divides by `h*w` to match torch / numpy.

/// 2-D forward (`inverse=false`) or inverse C2C FFT for f32.
/// Input layout: `[h, w, 2]` interleaved complex; output same shape.
pub fn gpu_fft2_c2c_f32(
    input: &CudaBuffer<f32>,
    h: usize,
    w: usize,
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if h == 0 || w == 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fft2_c2c_f32",
            expected: vec![1, 1],
            got: vec![h, w],
        });
    }
    let total = h * w * 2;
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fft2_c2c_f32",
            expected: vec![h, w, 2],
            got: vec![input.len()],
        });
    }

    let stream = device.stream();
    let plan = CudaFft::plan_2d(
        h as i32,
        w as i32,
        cufft_sys::cufftType::CUFFT_C2C,
        stream.clone(),
    )?;

    let mut tmp = crate::transfer::alloc_zeros_f32(total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;
    let mut out = crate::transfer::alloc_zeros_f32(total, device)?;

    let direction = if inverse {
        FftDirection::Inverse
    } else {
        FftDirection::Forward
    };
    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_c2c(
            plan.handle(),
            idata as *mut cufft_sys::cufftComplex,
            odata as *mut cufft_sys::cufftComplex,
            direction as c_int,
        )?;
    }

    if inverse {
        let scale = 1.0_f32 / (h as f32 * w as f32);
        out = crate::kernels::gpu_scale(&out, scale, device)?;
    }
    Ok(out)
}

/// f64 variant of [`gpu_fft2_c2c_f32`].
pub fn gpu_fft2_c2c_f64(
    input: &CudaBuffer<f64>,
    h: usize,
    w: usize,
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if h == 0 || w == 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fft2_c2c_f64",
            expected: vec![1, 1],
            got: vec![h, w],
        });
    }
    let total = h * w * 2;
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fft2_c2c_f64",
            expected: vec![h, w, 2],
            got: vec![input.len()],
        });
    }

    let stream = device.stream();
    let plan = CudaFft::plan_2d(
        h as i32,
        w as i32,
        cufft_sys::cufftType::CUFFT_Z2Z,
        stream.clone(),
    )?;

    let mut tmp = crate::transfer::alloc_zeros_f64(total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;
    let mut out = crate::transfer::alloc_zeros_f64(total, device)?;

    let direction = if inverse {
        FftDirection::Inverse
    } else {
        FftDirection::Forward
    };
    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_z2z(
            plan.handle(),
            idata as *mut cufft_sys::cufftDoubleComplex,
            odata as *mut cufft_sys::cufftDoubleComplex,
            direction as c_int,
        )?;
    }

    if inverse {
        let scale = 1.0_f64 / (h as f64 * w as f64);
        out = crate::kernels::gpu_scale_f64(&out, scale, device)?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Real-to-complex (f32)
// ---------------------------------------------------------------------------

/// Forward 1-D real-to-complex FFT, batched. Reads `[batch * n]` real f32,
/// writes `[batch * (n/2 + 1) * 2]` interleaved complex f32.
pub fn gpu_rfft_r2c_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    check_n_batch(n, batch, "gpu_rfft_r2c_f32")?;
    let in_total = batch.checked_mul(n).ok_or(GpuError::ShapeMismatch {
        op: "gpu_rfft_r2c_f32",
        expected: vec![batch, n],
        got: vec![input.len()],
    })?;
    if input.len() != in_total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_rfft_r2c_f32",
            expected: vec![batch, n],
            got: vec![input.len()],
        });
    }
    let half = n / 2 + 1;
    let out_total = batch * half * 2;

    let stream = device.stream();
    let plan = CudaFft::plan_1d(
        n as i32,
        cufft_sys::cufftType::CUFFT_R2C,
        batch as i32,
        stream.clone(),
    )?;
    let mut tmp = crate::transfer::alloc_zeros_f32(in_total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;
    let mut out = crate::transfer::alloc_zeros_f32(out_total, device)?;

    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_r2c(
            plan.handle(),
            idata as *mut cufft_sys::cufftReal,
            odata as *mut cufft_sys::cufftComplex,
        )?;
    }
    Ok(out)
}

/// f64 variant of [`gpu_rfft_r2c_f32`].
pub fn gpu_rfft_r2c_f64(
    input: &CudaBuffer<f64>,
    batch: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    check_n_batch(n, batch, "gpu_rfft_r2c_f64")?;
    let in_total = batch.checked_mul(n).ok_or(GpuError::ShapeMismatch {
        op: "gpu_rfft_r2c_f64",
        expected: vec![batch, n],
        got: vec![input.len()],
    })?;
    if input.len() != in_total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_rfft_r2c_f64",
            expected: vec![batch, n],
            got: vec![input.len()],
        });
    }
    let half = n / 2 + 1;
    let out_total = batch * half * 2;

    let stream = device.stream();
    let plan = CudaFft::plan_1d(
        n as i32,
        cufft_sys::cufftType::CUFFT_D2Z,
        batch as i32,
        stream.clone(),
    )?;
    let mut tmp = crate::transfer::alloc_zeros_f64(in_total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;
    let mut out = crate::transfer::alloc_zeros_f64(out_total, device)?;

    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_d2z(
            plan.handle(),
            idata as *mut cufft_sys::cufftDoubleReal,
            odata as *mut cufft_sys::cufftDoubleComplex,
        )?;
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Complex-to-real (f32)
// ---------------------------------------------------------------------------

/// Inverse 1-D real FFT, batched. Reads `[batch * (n_out/2 + 1) * 2]`
/// interleaved complex f32, writes `[batch * n_out]` real f32, normalized
/// by `1/n_out` (matches torch convention).
pub fn gpu_irfft_c2r_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    n_out: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    check_n_batch(n_out, batch, "gpu_irfft_c2r_f32")?;
    let half = n_out / 2 + 1;
    let in_total = batch
        .checked_mul(half)
        .and_then(|v| v.checked_mul(2))
        .ok_or(GpuError::ShapeMismatch {
            op: "gpu_irfft_c2r_f32",
            expected: vec![batch, half, 2],
            got: vec![input.len()],
        })?;
    if input.len() != in_total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_irfft_c2r_f32",
            expected: vec![batch, half, 2],
            got: vec![input.len()],
        });
    }
    let out_total = batch * n_out;

    let stream = device.stream();
    let plan = CudaFft::plan_1d(
        n_out as i32,
        cufft_sys::cufftType::CUFFT_C2R,
        batch as i32,
        stream.clone(),
    )?;
    // C2R is documented as overwriting input on some architectures — clone.
    let mut tmp = crate::transfer::alloc_zeros_f32(in_total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;
    let mut out = crate::transfer::alloc_zeros_f32(out_total, device)?;

    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_c2r(
            plan.handle(),
            idata as *mut cufft_sys::cufftComplex,
            odata as *mut cufft_sys::cufftReal,
        )?;
    }
    let scale = 1.0_f32 / n_out as f32;
    let normed = crate::kernels::gpu_scale(&out, scale, device)?;
    Ok(normed)
}

/// f64 variant of [`gpu_irfft_c2r_f32`].
pub fn gpu_irfft_c2r_f64(
    input: &CudaBuffer<f64>,
    batch: usize,
    n_out: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    check_n_batch(n_out, batch, "gpu_irfft_c2r_f64")?;
    let half = n_out / 2 + 1;
    let in_total = batch
        .checked_mul(half)
        .and_then(|v| v.checked_mul(2))
        .ok_or(GpuError::ShapeMismatch {
            op: "gpu_irfft_c2r_f64",
            expected: vec![batch, half, 2],
            got: vec![input.len()],
        })?;
    if input.len() != in_total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_irfft_c2r_f64",
            expected: vec![batch, half, 2],
            got: vec![input.len()],
        });
    }
    let out_total = batch * n_out;

    let stream = device.stream();
    let plan = CudaFft::plan_1d(
        n_out as i32,
        cufft_sys::cufftType::CUFFT_Z2D,
        batch as i32,
        stream.clone(),
    )?;
    let mut tmp = crate::transfer::alloc_zeros_f64(in_total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;
    let mut out = crate::transfer::alloc_zeros_f64(out_total, device)?;

    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_z2d(
            plan.handle(),
            idata as *mut cufft_sys::cufftDoubleComplex,
            odata as *mut cufft_sys::cufftDoubleReal,
        )?;
    }
    let scale = 1.0_f64 / n_out as f64;
    let normed = crate::kernels::gpu_scale_f64(&out, scale, device)?;
    Ok(normed)
}
