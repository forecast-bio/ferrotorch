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
