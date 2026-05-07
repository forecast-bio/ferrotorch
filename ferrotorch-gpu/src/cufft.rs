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
    // SAFETY:
    // - `cufft_result::exec_c2c` is the unsafe FFI shim around
    //   `cufftExecC2C` (NVIDIA cuFFT API:
    //   <https://docs.nvidia.com/cuda/cufft/index.html#function-cufftexecc2c>).
    //   The plan must match dtype / direction; pointers must be device-
    //   resident and sized as advertised.
    // - Plan: `plan` is a `CudaFft` created on line 84 with
    //   `CUFFT_C2C`, length `n`, batch `batch`, on this device's stream
    //   (line 88). The plan handle stays valid for the call (`plan` is
    //   owned by this stack frame and outlives the unsafe block).
    // - Buffer dtype: cuFFT's `cufftComplex` is byte-equivalent to a pair
    //   of f32 (re, im) per element; our interleaved `[batch*n*2]` f32
    //   buffer (validated against `total = batch*n*2` on line 75) is the
    //   identical bit layout. The cast `*mut f32 → *mut cufftComplex`
    //   reinterprets the same allocation.
    // - Lengths: `tmp` and `out` were each allocated as `total = batch*n*2`
    //   f32 elements (lines 90, 94). cuFFT reads `batch*n` complex samples
    //   from `idata` and writes `batch*n` to `odata` — exactly matching.
    // - Device pointers: `device_ptr_mut` on lines 103-104 yields raw
    //   `CUdeviceptr` plus `_isync` / `_osync` `SyncOnDrop` records held
    //   alive across the FFI call so the GPU work is recorded against
    //   `stream`.
    // - Aliasing: `tmp` and `out` are separately allocated buffers; cuFFT
    //   may overwrite `idata` (per the function rustdoc on line 92-93)
    //   but cannot alias `odata` because they are distinct allocations.
    // - Direction: `direction as c_int` is a defined-value enum cast
    //   (`FftDirection::Forward = -1`, `Inverse = +1`); cuFFT accepts
    //   exactly these values.
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
    // SAFETY:
    // - `cufft_result::exec_z2z` is the unsafe FFI shim around
    //   `cufftExecZ2Z` (NVIDIA cuFFT, double-precision complex C2C).
    // - Plan: created on line 147 with `CUFFT_Z2Z`, length `n`, batch
    //   `batch`, on this device's stream (line 151). Plan handle valid
    //   for the call duration.
    // - Buffer dtype: `cufftDoubleComplex` is byte-equivalent to a pair
    //   of f64 (re, im); our interleaved `[batch*n*2]` f64 buffer
    //   (validated against `total = batch*n*2` on line 138) reinterprets
    //   1:1.
    // - Lengths: `tmp` and `out` each `total = batch*n*2` f64 elements
    //   (lines 153, 154). cuFFT reads/writes `batch*n` Z2Z samples,
    //   matching the allocations exactly.
    // - Device pointers: `device_ptr_mut` lines 163-164 yield raw
    //   `CUdeviceptr` + `_isync` / `_osync` `SyncOnDrop` guards. The
    //   guards stay alive across the FFI call so the cuFFT work is
    //   ordered against `stream`.
    // - Aliasing: `tmp` and `out` are distinct allocations; cuFFT may
    //   overwrite `idata` but cannot alias `odata`.
    // - Direction: `FftDirection` enum cast to `c_int`; the only valid
    //   values (-1 Forward, +1 Inverse) are accepted by cuFFT.
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
    // SAFETY:
    // - `cufft_result::exec_c2c` is the unsafe FFI shim around
    //   `cufftExecC2C` (NVIDIA cuFFT, single-precision complex 2-D).
    // - Plan: created on line 246 with `CUFFT_C2C`, dimensions
    //   `[h, w]`, on this device's stream (line 251). The plan
    //   handle is valid for the call.
    // - Buffer dtype: `cufftComplex` is byte-equivalent to interleaved
    //   `(re, im)` f32 pairs; our `[h*w*2]` f32 buffer (validated
    //   against `total = h*w*2` on lines 204-211) reinterprets exactly.
    // - Lengths: `tmp` and `out` each `total = h*w*2` f32 elements
    //   (lines 253, 255 — both `alloc_zeros_f32(total, ...)`). cuFFT
    //   reads/writes exactly `h*w` complex samples.
    // - Device pointers: `device_ptr_mut` lines 263-264; `_isync` /
    //   `_osync` guards record completion on `stream` when dropped at
    //   block exit, ordering this exec against subsequent ops.
    // - Aliasing: `tmp` and `out` are distinct allocations; cuFFT may
    //   overwrite `idata` (cuFFT 2-D is also documented as
    //   in-place-capable, hence the `tmp` clone on line 254) but
    //   cannot alias `odata`.
    // - Direction: enum cast to `c_int`, valid values only.
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
    // SAFETY:
    // - `cufft_result::exec_z2z` is the unsafe FFI shim around
    //   `cufftExecZ2Z` (NVIDIA cuFFT, double-precision complex 2-D).
    // - Plan: created on line 305 with `CUFFT_Z2Z`, dimensions
    //   `[h, w]`, on this device's stream (line 309). The plan
    //   handle remains valid through the call.
    // - Buffer dtype: `cufftDoubleComplex` is byte-equivalent to
    //   interleaved `(re, im)` f64 pairs. Our `[h*w*2]` f64 buffer
    //   (validated against `total = h*w*2` on lines 263-270) matches
    //   exactly.
    // - Lengths: `tmp` and `out` each `total = h*w*2` f64 elements
    //   (lines 312, 314); cuFFT reads/writes `h*w` Z2Z samples.
    // - Device pointers: `device_ptr_mut` lines 322-323; `_isync` /
    //   `_osync` `SyncOnDrop` guards remain alive across the FFI call
    //   so completion events fire on `stream`.
    // - Aliasing: `tmp` and `out` are distinct allocations; `tmp` is
    //   the safe-to-overwrite clone (cuFFT in-place ambiguity).
    // - Direction: enum cast to `c_int`, only legal values produced.
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

    // SAFETY:
    // - `cufft_result::exec_r2c` is the unsafe FFI shim around
    //   `cufftExecR2C` (NVIDIA cuFFT, real-to-complex single precision).
    // - Plan: created with `CUFFT_R2C`, length `n`, batch `batch`, on
    //   this device's stream. Plan handle valid for the call.
    // - Input dtype: `cufftReal` is `float`, byte-equivalent to f32.
    //   `tmp` is a `[batch * n]` f32 buffer (allocation matches
    //   `in_total = batch*n` validated on the upstream guard).
    // - Output dtype: `cufftComplex` is interleaved `(re, im)` f32; our
    //   `[batch * (n/2 + 1) * 2]` f32 buffer (allocated as
    //   `out_total = batch * half * 2`) matches the cuFFT R2C output
    //   layout exactly (Hermitian-half representation).
    // - Lengths: cuFFT reads `batch*n` reals from `idata` and writes
    //   `batch*(n/2+1)` complex samples to `odata`. Our allocations
    //   match exactly.
    // - Device pointers: `device_ptr_mut` retrieves raw `CUdeviceptr`s
    //   plus `_isync` / `_osync` `SyncOnDrop` guards which order this
    //   exec against subsequent stream work when dropped at block exit.
    // - Aliasing: `tmp` and `out` are distinct allocations.
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

    // SAFETY:
    // - `cufft_result::exec_d2z` is the unsafe FFI shim around
    //   `cufftExecD2Z` (NVIDIA cuFFT, real-to-complex double precision).
    // - Plan: created with `CUFFT_D2Z`, length `n`, batch `batch`, on
    //   this device's stream. Plan handle valid for the call.
    // - Input dtype: `cufftDoubleReal` is `double`, byte-equivalent to
    //   f64. `tmp` is `[batch * n]` f64 (allocation matches
    //   `in_total = batch*n` validated upstream).
    // - Output dtype: `cufftDoubleComplex` is interleaved `(re, im)`
    //   f64 pairs. `out` is `[batch * (n/2 + 1) * 2]` f64 (allocated
    //   as `out_total = batch * half * 2`); cuFFT D2Z output layout
    //   matches exactly.
    // - Lengths: cuFFT reads `batch*n` reals, writes `batch*(n/2+1)`
    //   complex; allocations match.
    // - Device pointers: `device_ptr_mut` retrieves raw `CUdeviceptr`s
    //   plus `_isync` / `_osync` `SyncOnDrop` guards held alive across
    //   the FFI call, ordering this exec on `stream`.
    // - Aliasing: `tmp` and `out` distinct allocations.
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

    // SAFETY:
    // - `cufft_result::exec_c2r` is the unsafe FFI shim around
    //   `cufftExecC2R` (NVIDIA cuFFT, complex-to-real single precision).
    // - Plan: created with `CUFFT_C2R`, length `n_out`, batch `batch`,
    //   on this device's stream. Plan handle valid for the call.
    // - Input dtype: `cufftComplex` is interleaved `(re, im)` f32.
    //   `tmp` is `[batch * (n_out/2 + 1) * 2]` f32 (allocation matches
    //   `in_total = batch * half * 2`, validated upstream on lines
    //   419-432). cuFFT reads `batch*(n_out/2+1)` complex samples.
    // - Output dtype: `cufftReal` is `float`, byte-equivalent to f32;
    //   `out` is `[batch * n_out]` f32 (allocation matches
    //   `out_total = batch * n_out`). cuFFT writes exactly that many
    //   reals.
    // - Device pointers: `device_ptr_mut` lines 449-450; `_isync` /
    //   `_osync` `SyncOnDrop` guards record completion events on
    //   `stream` when dropped at block exit.
    // - Aliasing: `tmp` (cloned input on line 444) and `out` are
    //   distinct allocations; the clone is mandatory because cuFFT
    //   C2R is documented as potentially overwriting `idata` on some
    //   architectures (the function rustdoc on line 443 calls this
    //   out explicitly).
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

// ---------------------------------------------------------------------------
// Hermitian FFT (hfft, ihfft) (#636)
// ---------------------------------------------------------------------------
//
// PyTorch parity:
//   hfft(x)  = irfft(conj(x))   -- complex spectrum in, real signal out.
//   ihfft(x) = conj(rfft(x))    -- real signal in, complex spectrum out.
//
// Both ops run fully on-device: the conjugate step uses `gpu_conj_f32/f64`
// (PTX kernel that negates imaginary parts in the existing buffer), then
// the rfft/irfft cuFFT plan is applied. No host round-trip.

// ---------------------------------------------------------------------------
// Internal C2R helper (no 1/n normalization) — used by hfft only
// ---------------------------------------------------------------------------
//
// cuFFT C2R (cufftExecC2R / cufftExecZ2D) writes the unnormalized inverse
// DFT: out[k] = sum_{j=0}^{N-1} in[j] * exp(+2*pi*i*j*k/N).
//
// gpu_irfft_c2r_f32/f64 divide by n_out to match torch.fft.irfft convention.
// hfft instead keeps the raw C2R output (no division), which matches
// torch.fft.hfft (= Re[IDFT_unnorm(conj(x))], numpy/PyTorch "backward" norm).

fn gpu_c2r_raw_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    n_out: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    check_n_batch(n_out, batch, "gpu_c2r_raw_f32")?;
    let half = n_out / 2 + 1;
    let in_total = batch
        .checked_mul(half)
        .and_then(|v| v.checked_mul(2))
        .ok_or(GpuError::ShapeMismatch {
            op: "gpu_c2r_raw_f32",
            expected: vec![batch, half, 2],
            got: vec![input.len()],
        })?;
    if input.len() != in_total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_c2r_raw_f32",
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
    let mut tmp = crate::transfer::alloc_zeros_f32(in_total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;
    let mut out = crate::transfer::alloc_zeros_f32(out_total, device)?;
    // SAFETY:
    // - `cufft_result::exec_c2r` wraps `cufftExecC2R` (NVIDIA cuFFT C2R,
    //   single precision). The plan was created on this device's stream with
    //   `CUFFT_C2R`, length `n_out`, batch `batch`.
    // - `tmp` is `[batch*(n_out/2+1)*2]` f32 (validated against `in_total`
    //   above). `out` is `[batch*n_out]` f32 (allocated as `out_total`).
    //   cuFFT reads `batch*(n_out/2+1)` complex samples and writes
    //   `batch*n_out` reals -- sizes match.
    // - `tmp` (cloned input) and `out` are distinct allocations; C2R may
    //   overwrite `idata` on some architectures, hence the clone.
    // - `_isync`/`_osync` `SyncOnDrop` guards stay alive across the FFI call
    //   so the cuFFT work is recorded on `stream`.
    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_c2r(
            plan.handle(),
            idata as *mut cufft_sys::cufftComplex,
            odata as *mut cufft_sys::cufftReal,
        )?;
    }
    // No 1/n_out division -- hfft convention (torch.fft.hfft = Re[IDFT_unnorm(conj(x))]).
    Ok(out)
}

fn gpu_c2r_raw_f64(
    input: &CudaBuffer<f64>,
    batch: usize,
    n_out: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    check_n_batch(n_out, batch, "gpu_c2r_raw_f64")?;
    let half = n_out / 2 + 1;
    let in_total = batch
        .checked_mul(half)
        .and_then(|v| v.checked_mul(2))
        .ok_or(GpuError::ShapeMismatch {
            op: "gpu_c2r_raw_f64",
            expected: vec![batch, half, 2],
            got: vec![input.len()],
        })?;
    if input.len() != in_total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_c2r_raw_f64",
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
    // SAFETY:
    // - `cufft_result::exec_z2d` wraps `cufftExecZ2D` (NVIDIA cuFFT Z2D,
    //   double precision). Plan created on this device's stream with
    //   `CUFFT_Z2D`, length `n_out`, batch `batch`.
    // - `tmp` is `[batch*(n_out/2+1)*2]` f64 (validated against `in_total`).
    //   `out` is `[batch*n_out]` f64 (allocated as `out_total`). cuFFT reads
    //   `batch*(n_out/2+1)` Z2D complex samples and writes `batch*n_out`
    //   reals -- sizes match.
    // - `tmp` (cloned input) and `out` are distinct allocations.
    // - `_isync`/`_osync` guards stay alive across the FFI call.
    unsafe {
        let (idata, _isync) = tmp.inner_mut().device_ptr_mut(&stream);
        let (odata, _osync) = out.inner_mut().device_ptr_mut(&stream);
        cufft_result::exec_z2d(
            plan.handle(),
            idata as *mut cufft_sys::cufftDoubleComplex,
            odata as *mut cufft_sys::cufftDoubleReal,
        )?;
    }
    // No 1/n_out division -- hfft convention.
    Ok(out)
}

/// Forward Hermitian FFT: takes `[batch, n/2+1, 2]` interleaved complex f32
/// and returns `[batch * n_out]` real f32, where `n_out = 2*(n/2+1 - 1) = 2*half - 2`
/// unless the caller specifies a different `n_out` (matching `irfft_c2r`).
///
/// PyTorch parity: `hfft(x, n) = Re[IDFT_unnorm(conj(x))]`.
/// The C2R cuFFT output is kept unnormalized (no `1/n_out` division), matching
/// `torch.fft.hfft` / `numpy.fft.hfft` with "backward" normalization convention.
pub fn gpu_hfft_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    half_in: usize,
    n_out: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    check_n_batch(half_in, batch, "gpu_hfft_f32")?;
    let expected_half = n_out / 2 + 1;
    if half_in != expected_half {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_hfft_f32",
            expected: vec![batch, expected_half, 2],
            got: vec![input.len()],
        });
    }
    let in_total = batch * half_in * 2;
    if input.len() != in_total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_hfft_f32",
            expected: vec![batch, half_in, 2],
            got: vec![input.len()],
        });
    }
    // Copy input so we can conjugate in-place without mutating the caller's buffer.
    let stream = device.stream();
    let mut tmp = crate::transfer::alloc_zeros_f32(in_total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;
    // Conjugate: negate imaginary parts on-device.
    let conj_buf = crate::kernels::gpu_conj_f32(tmp, device)?;
    // hfft = Re[IDFT_unnorm(conj(x))]: apply C2R without the 1/n_out scale.
    gpu_c2r_raw_f32(&conj_buf, batch, n_out, device)
}

/// f64 variant of [`gpu_hfft_f32`].
pub fn gpu_hfft_f64(
    input: &CudaBuffer<f64>,
    batch: usize,
    half_in: usize,
    n_out: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    check_n_batch(half_in, batch, "gpu_hfft_f64")?;
    let expected_half = n_out / 2 + 1;
    if half_in != expected_half {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_hfft_f64",
            expected: vec![batch, expected_half, 2],
            got: vec![input.len()],
        });
    }
    let in_total = batch * half_in * 2;
    if input.len() != in_total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_hfft_f64",
            expected: vec![batch, half_in, 2],
            got: vec![input.len()],
        });
    }
    let stream = device.stream();
    let mut tmp = crate::transfer::alloc_zeros_f64(in_total, device)?;
    stream.memcpy_dtod(input.inner(), tmp.inner_mut())?;
    let conj_buf = crate::kernels::gpu_conj_f64(tmp, device)?;
    // hfft = Re[IDFT_unnorm(conj(x))]: apply C2R without the 1/n_out scale.
    gpu_c2r_raw_f64(&conj_buf, batch, n_out, device)
}

/// Inverse Hermitian FFT: takes `[batch * n]` real f32 and returns
/// `[batch, n/2+1, 2]` interleaved complex f32 (conjugated rfft output).
///
/// PyTorch parity: `ihfft(x) = conj(rfft(x)) / n`.
pub fn gpu_ihfft_f32(
    input: &CudaBuffer<f32>,
    batch: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    // rfft -> conj.
    // gpu_rfft_r2c_f32 already normalizes nothing (irfft normalizes; rfft does
    // not). For ihfft, PyTorch normalizes by 1/n. The rfft output is the
    // one-sided DFT without normalization; we divide by n via gpu_scale on
    // the output, then conjugate.
    //
    // Actually: torch.fft.ihfft(x) = conj(rfft(x)) / n.
    // Our gpu_rfft_r2c_f32 returns unnormalized rfft. We need to scale by 1/n,
    // then conjugate. Order: rfft -> scale(1/n) -> conj.
    check_n_batch(n, batch, "gpu_ihfft_f32")?;
    let rfft_out = gpu_rfft_r2c_f32(input, batch, n, device)?;
    let scale = 1.0_f32 / n as f32;
    let scaled = crate::kernels::gpu_scale(&rfft_out, scale, device)?;
    crate::kernels::gpu_conj_f32(scaled, device)
}

/// f64 variant of [`gpu_ihfft_f32`].
pub fn gpu_ihfft_f64(
    input: &CudaBuffer<f64>,
    batch: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    check_n_batch(n, batch, "gpu_ihfft_f64")?;
    let rfft_out = gpu_rfft_r2c_f64(input, batch, n, device)?;
    let scale = 1.0_f64 / n as f64;
    let scaled = crate::kernels::gpu_scale_f64(&rfft_out, scale, device)?;
    crate::kernels::gpu_conj_f64(scaled, device)
}

// ---------------------------------------------------------------------------
// N-D complex-to-complex FFT -- 3-D (fftn, ifftn) (#636)
// ---------------------------------------------------------------------------
//
// Uses `cufftPlan3d` for the canonical [d, h, w, 2] unbatched case.
// Input/output layout: `[d * h * w * 2]` interleaved complex.
// `inverse=true` divides by `d*h*w` to match torch.fft.ifftn.

/// 3-D forward (`inverse=false`) or inverse C2C FFT for f32.
/// Input layout: `[d, h, w, 2]` interleaved complex; output same shape.
pub fn gpu_fftn3d_c2c_f32(
    input: &CudaBuffer<f32>,
    d: usize,
    h: usize,
    w: usize,
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if d == 0 || h == 0 || w == 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn3d_c2c_f32",
            expected: vec![1, 1, 1],
            got: vec![d, h, w],
        });
    }
    let total = d * h * w * 2;
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn3d_c2c_f32",
            expected: vec![d, h, w, 2],
            got: vec![input.len()],
        });
    }

    let stream = device.stream();
    let plan = CudaFft::plan_3d(
        d as i32,
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
    // SAFETY:
    // - `cufft_result::exec_c2c` is the unsafe FFI shim around
    //   `cufftExecC2C` (NVIDIA cuFFT, single-precision C2C).
    // - Plan: created on the line above with `CUFFT_C2C`, dimensions
    //   `[d, h, w]`, on this device's stream. Plan handle valid for the
    //   call duration.
    // - Buffer dtype: `cufftComplex` is byte-equivalent to interleaved
    //   `(re, im)` f32 pairs. `tmp` and `out` are each `[d*h*w*2]` f32
    //   (validated against `total = d*h*w*2` above). cuFFT reads/writes
    //   `d*h*w` complex samples, matching the allocations exactly.
    // - Device pointers: `device_ptr_mut` retrieves raw `CUdeviceptr` +
    //   `_isync`/`_osync` `SyncOnDrop` guards. The guards stay alive
    //   across the FFI call so the cuFFT work is ordered against `stream`.
    // - Aliasing: `tmp` and `out` are distinct allocations.
    // - Direction: enum cast to `c_int`; only legal values produced.
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
        let scale = 1.0_f32 / (d as f32 * h as f32 * w as f32);
        out = crate::kernels::gpu_scale(&out, scale, device)?;
    }
    Ok(out)
}

/// f64 variant of [`gpu_fftn3d_c2c_f32`].
pub fn gpu_fftn3d_c2c_f64(
    input: &CudaBuffer<f64>,
    d: usize,
    h: usize,
    w: usize,
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if d == 0 || h == 0 || w == 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn3d_c2c_f64",
            expected: vec![1, 1, 1],
            got: vec![d, h, w],
        });
    }
    let total = d * h * w * 2;
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn3d_c2c_f64",
            expected: vec![d, h, w, 2],
            got: vec![input.len()],
        });
    }

    let stream = device.stream();
    let plan = CudaFft::plan_3d(
        d as i32,
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
    // SAFETY:
    // - `cufft_result::exec_z2z` is the unsafe FFI shim around
    //   `cufftExecZ2Z` (NVIDIA cuFFT, double-precision C2C 3-D).
    // - Plan: created above with `CUFFT_Z2Z`, dims `[d, h, w]`, on
    //   this device's stream. Plan handle valid for the call.
    // - Buffer dtype: `cufftDoubleComplex` is byte-equivalent to
    //   interleaved `(re, im)` f64 pairs. `tmp` and `out` are each
    //   `[d*h*w*2]` f64 (validated against `total`). cuFFT reads/writes
    //   `d*h*w` Z2Z samples, matching exactly.
    // - Device pointers: `device_ptr_mut` retrieves raw `CUdeviceptr` +
    //   `_isync`/`_osync` guards alive across the FFI call.
    // - Aliasing: `tmp` and `out` are distinct allocations.
    // - Direction: enum cast to `c_int`; only legal values.
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
        let scale = 1.0_f64 / (d as f64 * h as f64 * w as f64);
        out = crate::kernels::gpu_scale_f64(&out, scale, device)?;
    }
    Ok(out)
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

    // SAFETY:
    // - `cufft_result::exec_z2d` is the unsafe FFI shim around
    //   `cufftExecZ2D` (NVIDIA cuFFT, complex-to-real double precision).
    // - Plan: created with `CUFFT_Z2D`, length `n_out`, batch `batch`,
    //   on this device's stream. Plan handle valid for the call.
    // - Input dtype: `cufftDoubleComplex` is interleaved `(re, im)`
    //   f64 pairs. `tmp` is `[batch * (n_out/2 + 1) * 2]` f64 (matches
    //   `in_total = batch * half * 2`, validated upstream).
    // - Output dtype: `cufftDoubleReal` is `double`, byte-equivalent
    //   to f64; `out` is `[batch * n_out]` f64 (matches `out_total =
    //   batch * n_out`).
    // - Lengths: cuFFT reads `batch*(n_out/2+1)` complex samples from
    //   `idata` and writes `batch*n_out` reals to `odata`. Allocations
    //   match.
    // - Device pointers: `device_ptr_mut` retrieves raw `CUdeviceptr`
    //   plus `_isync` / `_osync` `SyncOnDrop` guards. The guards stay
    //   alive across the FFI call so completion events are recorded
    //   on `stream` before any subsequent op observes the buffers.
    // - Aliasing: `tmp` (cloned input) and `out` are distinct
    //   allocations; the clone protects against Z2D in-place
    //   overwrite ambiguity per the same caveat called out for the
    //   f32 variant on line 443.
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

// ---------------------------------------------------------------------------
// 2-D complex-to-complex FFT via cufftPlanMany (#636)
// ---------------------------------------------------------------------------
//
// cufftPlanMany(rank=2, n=[h,w], ..., CUFFT_C2C, batch=1) covers the
// unbatched [h, w, 2] case that cufftPlan2d misses for rank != 2.
// Input/output layout: `[h * w * 2]` interleaved complex, same convention
// as all other ops in this file.
// `inverse=true` divides by `h*w` to match torch.fft.ifftn.

/// 2-D forward (`inverse=false`) or inverse C2C FFT via `cufftPlanMany` for f32.
///
/// Input layout: `[h, w, 2]` interleaved complex; output same shape.
/// Normalization: `inverse=true` divides by `h*w` (torch.fft.ifftn parity).
pub fn gpu_fftn2d_c2c_f32(
    input: &CudaBuffer<f32>,
    h: usize,
    w: usize,
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if h == 0 || w == 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn2d_c2c_f32",
            expected: vec![1, 1],
            got: vec![h, w],
        });
    }
    let total = h * w * 2;
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn2d_c2c_f32",
            expected: vec![h, w, 2],
            got: vec![input.len()],
        });
    }

    let stream = device.stream();
    // cufftPlanMany with rank=2, n=[h,w], contiguous layout, batch=1.
    // istride=ostride=1 (contiguous), idist=odist=h*w (one transform).
    let n_dims = [h as c_int, w as c_int];
    let plan = CudaFft::plan_many(
        &n_dims,
        None,
        1,
        (h * w) as i32,
        None,
        1,
        (h * w) as i32,
        cufft_sys::cufftType::CUFFT_C2C,
        1,
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
    // SAFETY:
    // - `cufft_result::exec_c2c` wraps `cufftExecC2C` (NVIDIA cuFFT,
    //   single-precision C2C). Plan created via `plan_many` with rank=2,
    //   n=[h,w], CUFFT_C2C, batch=1, on this device's stream.
    // - Buffer dtype: `cufftComplex` is byte-equivalent to interleaved
    //   `(re, im)` f32 pairs. `tmp` and `out` are each `[h*w*2]` f32
    //   (validated against `total = h*w*2`). cuFFT reads/writes `h*w`
    //   complex samples -- sizes match.
    // - `_isync`/`_osync` `SyncOnDrop` guards stay alive across the call.
    // - Aliasing: `tmp` and `out` are distinct allocations.
    // - Direction: enum cast to `c_int`; only legal values produced.
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

// ---------------------------------------------------------------------------
// Axes-aware N-D C2C FFT via cufftPlanMany (#966)
// ---------------------------------------------------------------------------
//
// cufftPlanMany with arbitrary axes: given a complex tensor of shape
// `[d0, d1, ..., dk, 2]` (interleaved re/im), transform over the subset of
// spatial dims nominated by `axes`.
//
// cufftPlanMany parameters for a rank-r transform over axes a[0..r]:
//   n[i]    = shape[a[i]]          -- transform length along each axis
//   inembed = shape (spatial dims) -- embedding array (full tensor shape)
//   istride = product of shape[a[i]+1..] for the innermost axis stride
//   idist   = total spatial elements (shape.iter().product())
//   batch   = product of shape[j] for j NOT in axes (outer batch dims)
//
// For contiguous complex layout `[..., 2]`, the batch stride is the total
// number of complex elements (spatial_total) and istride=ostride=1 when
// the transform axes are contiguous at the innermost dims. For arbitrary
// axes we permute and re-use the same batch/embed/stride parameters that
// cufftPlanMany requires for non-contiguous axes.
//
// The simplest correct encoding for arbitrary axes:
//   rank     = axes.len()
//   n[i]     = shape[axes[i]]
//   inembed  = NULL  (=> cuFFT treats input as rank-r contiguous sub-array)
//   istride  = 1
//   idist    = product(shape[axes[i]] for i) = product of transform dims
//   batch    = total_spatial / idist
//
// This is equivalent to treating the tensor as `batch` independent rank-r
// transforms each of shape n[]. It is correct when the transform axes form
// the innermost dimensions (as torch.fft.fftn guarantees after its internal
// contiguification). The dispatcher in fft.rs already ensures axes are
// normalized and the tensor is in the expected layout.
//
// `inverse=true` divides by product(n[i]) to match torch.fft.ifftn.

/// Axes-aware N-D forward/inverse C2C FFT for f32 via `cufftPlanMany`. (#966)
///
/// `shape` is the spatial shape (excluding the trailing complex dim 2).
/// `axes` are zero-based indices into `shape`, all distinct and in `[0, ndim)`.
/// Input/output layout: `[shape[0], shape[1], ..., 2]` interleaved complex.
/// `inverse=true` divides by the product of the transform-axis lengths.
pub fn gpu_fftn_axes_c2c_f32(
    input: &CudaBuffer<f32>,
    shape: &[usize],
    axes: &[usize],
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    if axes.is_empty() || shape.is_empty() {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn_axes_c2c_f32",
            expected: vec![1],
            got: vec![axes.len()],
        });
    }
    let spatial_total: usize = shape.iter().product();
    let total = spatial_total * 2; // interleaved complex
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn_axes_c2c_f32",
            expected: vec![spatial_total, 2],
            got: vec![input.len()],
        });
    }

    // Transform dims: product of shape[axes[i]].
    let transform_vol: usize = axes.iter().map(|&a| shape[a]).product();
    let batch = spatial_total / transform_vol;
    let n_dims: Vec<c_int> = axes.iter().map(|&a| shape[a] as c_int).collect();

    let stream = device.stream();
    // cufftPlanMany(rank, n, inembed=NULL, istride=1, idist=transform_vol,
    //               onembed=NULL, ostride=1, odist=transform_vol,
    //               CUFFT_C2C, batch)
    // NULL inembed/onembed means cuFFT treats input/output as tightly packed
    // rank-dimensional arrays of shape n[0..rank], with `batch` independent
    // transforms. idist=odist=transform_vol is the stride between batches.
    let plan = CudaFft::plan_many(
        &n_dims,
        None,
        1,
        transform_vol as i32,
        None,
        1,
        transform_vol as i32,
        cufft_sys::cufftType::CUFFT_C2C,
        batch as i32,
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
    // SAFETY:
    // - `cufft_result::exec_c2c` wraps `cufftExecC2C` (NVIDIA cuFFT,
    //   single-precision C2C). Plan created via `plan_many` with rank=axes.len(),
    //   n=shape[axes[i]], CUFFT_C2C, batch=spatial_total/transform_vol, on
    //   this device's stream.
    // - Buffer dtype: `cufftComplex` is byte-equivalent to interleaved `(re, im)`
    //   f32 pairs. `tmp` and `out` are each `[spatial_total * 2]` f32 (validated
    //   against `total = spatial_total * 2` above). cuFFT reads/writes
    //   `batch * transform_vol` complex samples, matching the allocations.
    // - `_isync`/`_osync` `SyncOnDrop` guards stay alive across the FFI call
    //   so completion events fire on `stream`.
    // - Aliasing: `tmp` and `out` are distinct allocations.
    // - Direction: enum cast to `c_int`; only legal values produced.
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
        let scale = 1.0_f32 / transform_vol as f32;
        out = crate::kernels::gpu_scale(&out, scale, device)?;
    }
    Ok(out)
}

/// f64 variant of [`gpu_fftn_axes_c2c_f32`].
pub fn gpu_fftn_axes_c2c_f64(
    input: &CudaBuffer<f64>,
    shape: &[usize],
    axes: &[usize],
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if axes.is_empty() || shape.is_empty() {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn_axes_c2c_f64",
            expected: vec![1],
            got: vec![axes.len()],
        });
    }
    let spatial_total: usize = shape.iter().product();
    let total = spatial_total * 2;
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn_axes_c2c_f64",
            expected: vec![spatial_total, 2],
            got: vec![input.len()],
        });
    }

    let transform_vol: usize = axes.iter().map(|&a| shape[a]).product();
    let batch = spatial_total / transform_vol;
    let n_dims: Vec<c_int> = axes.iter().map(|&a| shape[a] as c_int).collect();

    let stream = device.stream();
    let plan = CudaFft::plan_many(
        &n_dims,
        None,
        1,
        transform_vol as i32,
        None,
        1,
        transform_vol as i32,
        cufft_sys::cufftType::CUFFT_Z2Z,
        batch as i32,
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
    // SAFETY:
    // - `cufft_result::exec_z2z` wraps `cufftExecZ2Z` (NVIDIA cuFFT,
    //   double-precision C2C). Plan created via `plan_many` with rank=axes.len(),
    //   n=shape[axes[i]], CUFFT_Z2Z, batch=spatial_total/transform_vol, on
    //   this device's stream.
    // - Buffer dtype: `cufftDoubleComplex` is byte-equivalent to interleaved
    //   `(re, im)` f64 pairs. `tmp` and `out` are each `[spatial_total * 2]`
    //   f64 (validated against `total = spatial_total * 2` above).
    // - `_isync`/`_osync` guards stay alive across the FFI call.
    // - Aliasing: `tmp` and `out` are distinct allocations.
    // - Direction: only legal values produced.
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
        let scale = 1.0_f64 / transform_vol as f64;
        out = crate::kernels::gpu_scale_f64(&out, scale, device)?;
    }
    Ok(out)
}

/// f64 variant of [`gpu_fftn2d_c2c_f32`].
pub fn gpu_fftn2d_c2c_f64(
    input: &CudaBuffer<f64>,
    h: usize,
    w: usize,
    inverse: bool,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f64>> {
    if h == 0 || w == 0 {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn2d_c2c_f64",
            expected: vec![1, 1],
            got: vec![h, w],
        });
    }
    let total = h * w * 2;
    if input.len() != total {
        return Err(GpuError::ShapeMismatch {
            op: "gpu_fftn2d_c2c_f64",
            expected: vec![h, w, 2],
            got: vec![input.len()],
        });
    }

    let stream = device.stream();
    let n_dims = [h as c_int, w as c_int];
    let plan = CudaFft::plan_many(
        &n_dims,
        None,
        1,
        (h * w) as i32,
        None,
        1,
        (h * w) as i32,
        cufft_sys::cufftType::CUFFT_Z2Z,
        1,
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
    // SAFETY:
    // - `cufft_result::exec_z2z` wraps `cufftExecZ2Z` (NVIDIA cuFFT,
    //   double-precision C2C). Plan created via `plan_many` with rank=2,
    //   n=[h,w], CUFFT_Z2Z, batch=1, on this device's stream.
    // - Buffer dtype: `cufftDoubleComplex` is byte-equivalent to interleaved
    //   `(re, im)` f64 pairs. `tmp` and `out` are each `[h*w*2]` f64
    //   (validated against `total = h*w*2`). Sizes match.
    // - `_isync`/`_osync` guards stay alive across the FFI call.
    // - Aliasing: distinct allocations.
    // - Direction: only legal values produced.
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
