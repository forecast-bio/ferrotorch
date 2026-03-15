//! Custom PTX CUDA kernels for elementwise GPU operations.
//!
//! Each operation has two code paths:
//!
//! 1. **PTX kernel** -- a hand-written PTX string that is loaded into the CUDA
//!    driver at runtime via [`cudarc`]. This is the fast path and runs entirely
//!    on the GPU.
//! 2. **CPU fallback** -- copies data to the host, performs the operation with
//!    standard Rust iterators, and copies the result back. Correct but slow;
//!    used when the PTX module cannot be loaded (e.g. architecture mismatch).
//!
//! # Supported operations
//!
//! | Function | Formula |
//! |----------|---------|
//! | [`gpu_add`] | `out[i] = a[i] + b[i]` |
//! | [`gpu_sub`] | `out[i] = a[i] - b[i]` |
//! | [`gpu_mul`] | `out[i] = a[i] * b[i]` |
//! | [`gpu_neg`] | `out[i] = -a[i]` |
//! | [`gpu_relu`] | `out[i] = max(a[i], 0.0)` |

#[cfg(feature = "cuda")]
use cudarc::driver::LaunchConfig;

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::{alloc_zeros, cpu_to_gpu, gpu_to_cpu};

// ---------------------------------------------------------------------------
// PTX kernel source strings
// ---------------------------------------------------------------------------

/// PTX source for `add_kernel`: `out[i] = a[i] + b[i]`.
#[cfg(feature = "cuda")]
pub(crate) const ADD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry add_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %tid, %bid, %bdim, %tid;

    setp.ge.u32 %p, %tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    ld.global.f32 %vb, [%b];
    add.f32 %vr, %va, %vb;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `sub_kernel`: `out[i] = a[i] - b[i]`.
#[cfg(feature = "cuda")]
pub(crate) const SUB_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry sub_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %tid, %bid, %bdim, %tid;

    setp.ge.u32 %p, %tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    ld.global.f32 %vb, [%b];
    sub.f32 %vr, %va, %vb;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `mul_kernel`: `out[i] = a[i] * b[i]`.
#[cfg(feature = "cuda")]
pub(crate) const MUL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mul_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %tid, %bid, %bdim, %tid;

    setp.ge.u32 %p, %tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    ld.global.f32 %vb, [%b];
    mul.f32 %vr, %va, %vb;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `neg_kernel`: `out[i] = -a[i]`.
#[cfg(feature = "cuda")]
pub(crate) const NEG_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry neg_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %tid, %bid, %bdim, %tid;

    setp.ge.u32 %p, %tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    neg.f32 %vr, %va;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `relu_kernel`: `out[i] = max(a[i], 0.0)`.
#[cfg(feature = "cuda")]
pub(crate) const RELU_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry relu_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr, %zero;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %tid, %bid, %bdim, %tid;

    setp.ge.u32 %p, %tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    mov.f32 %zero, 0f00000000;
    max.f32 %vr, %va, %zero;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Launch configuration helper
// ---------------------------------------------------------------------------

/// Standard 1-D launch config for `n` elements.
///
/// Uses 256 threads per block, which is a good default for elementwise ops
/// on all modern NVIDIA architectures.
///
/// # Errors
///
/// Returns [`GpuError::ShapeMismatch`] if `n` exceeds `u32::MAX`, which
/// would silently truncate the grid dimension.
#[cfg(feature = "cuda")]
fn launch_cfg(n: usize) -> GpuResult<LaunchConfig> {
    if n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "kernel_launch",
            expected: vec![u32::MAX as usize],
            got: vec![n],
        });
    }
    const BLOCK: u32 = 256;
    let grid = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    Ok(LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that two buffers are on the same device and have the same length.
#[cfg(feature = "cuda")]
fn validate_binary(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    if a.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: a.device_ordinal(),
            got: device.ordinal(),
        });
    }
    if b.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: b.device_ordinal(),
            got: device.ordinal(),
        });
    }
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    Ok(())
}

/// Validate that a unary buffer is on the correct device.
#[cfg(feature = "cuda")]
fn validate_unary(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<()> {
    if a.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: a.device_ordinal(),
            got: device.ordinal(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX kernel launch helpers
// ---------------------------------------------------------------------------

/// Try to launch a binary PTX kernel. Returns `Ok(Some(buf))` on success,
/// `Ok(None)` if the PTX module failed to load (caller should fall back to
/// CPU), or `Err` on a real CUDA error after a successful launch.
#[cfg(feature = "cuda")]
fn try_launch_binary(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<Option<CudaBuffer<f32>>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    // Attempt to load the kernel (cached after first compilation).
    // If it fails (e.g. unsupported arch), return None so the caller
    // can use the CPU fallback.
    let f = match crate::module_cache::get_or_compile(ctx, ptx_src, kernel_name, device.ordinal() as u32) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    let mut out = alloc_zeros::<f32>(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY: The kernel reads `n` f32 values from `a` and `b`, writes `n`
    // f32 values to `out`. All three buffers are device-resident and at
    // least `n` elements long. The grid covers exactly `n` threads.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(Some(out))
}

/// Try to launch a unary PTX kernel. Returns `Ok(Some(buf))` on success,
/// `Ok(None)` if the PTX module failed to load.
#[cfg(feature = "cuda")]
fn try_launch_unary(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<Option<CudaBuffer<f32>>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    // Attempt to load the kernel (cached after first compilation).
    let f = match crate::module_cache::get_or_compile(ctx, ptx_src, kernel_name, device.ordinal() as u32) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    let mut out = alloc_zeros::<f32>(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY: The kernel reads `n` f32 values from `a` and writes `n` f32
    // values to `out`. Both buffers are device-resident with length >= n.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// CPU fallback helpers
// ---------------------------------------------------------------------------

/// CPU fallback for binary ops: copy both inputs to host, apply `op`, copy
/// the result back.
#[cfg(feature = "cuda")]
fn cpu_fallback_binary(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
    op: fn(f32, f32) -> f32,
) -> GpuResult<CudaBuffer<f32>> {
    let a_host = gpu_to_cpu(a, device)?;
    let b_host = gpu_to_cpu(b, device)?;
    let result: Vec<f32> = a_host
        .iter()
        .zip(b_host.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();
    cpu_to_gpu(&result, device)
}

/// CPU fallback for unary ops.
#[cfg(feature = "cuda")]
fn cpu_fallback_unary(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
    op: fn(f32) -> f32,
) -> GpuResult<CudaBuffer<f32>> {
    let a_host = gpu_to_cpu(a, device)?;
    let result: Vec<f32> = a_host.iter().map(|&x| op(x)).collect();
    cpu_to_gpu(&result, device)
}

// ---------------------------------------------------------------------------
// Public API -- binary ops
// ---------------------------------------------------------------------------

/// Elementwise addition: `out[i] = a[i] + b[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a`, `b`, or `device` refer to
///   different CUDA devices.
/// - [`GpuError::LengthMismatch`] if `a` and `b` have different lengths.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_add(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(a, b, device)?;

    if let Some(out) = try_launch_binary(a, b, device, ADD_PTX, "add_kernel")? {
        return Ok(out);
    }

    // CPU fallback (correct but slow).
    cpu_fallback_binary(a, b, device, |x, y| x + y)
}

/// Elementwise subtraction: `out[i] = a[i] - b[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a`, `b`, or `device` refer to
///   different CUDA devices.
/// - [`GpuError::LengthMismatch`] if `a` and `b` have different lengths.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_sub(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(a, b, device)?;

    if let Some(out) = try_launch_binary(a, b, device, SUB_PTX, "sub_kernel")? {
        return Ok(out);
    }

    cpu_fallback_binary(a, b, device, |x, y| x - y)
}

/// Elementwise multiplication: `out[i] = a[i] * b[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a`, `b`, or `device` refer to
///   different CUDA devices.
/// - [`GpuError::LengthMismatch`] if `a` and `b` have different lengths.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_mul(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(a, b, device)?;

    if let Some(out) = try_launch_binary(a, b, device, MUL_PTX, "mul_kernel")? {
        return Ok(out);
    }

    cpu_fallback_binary(a, b, device, |x, y| x * y)
}

// ---------------------------------------------------------------------------
// Public API -- unary ops
// ---------------------------------------------------------------------------

/// Elementwise negation: `out[i] = -a[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a` and `device` refer to different
///   CUDA devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_neg(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;

    if let Some(out) = try_launch_unary(a, device, NEG_PTX, "neg_kernel")? {
        return Ok(out);
    }

    cpu_fallback_unary(a, device, |x| -x)
}

/// Elementwise ReLU: `out[i] = max(a[i], 0.0)`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a` and `device` refer to different
///   CUDA devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_relu(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;

    if let Some(out) = try_launch_unary(a, device, RELU_PTX, "relu_kernel")? {
        return Ok(out);
    }

    cpu_fallback_unary(a, device, |x| x.max(0.0))
}

// ---------------------------------------------------------------------------
// Stubs when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_add(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_sub(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_mul(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_neg(
    _a: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_relu(
    _a: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Tests -- require a real CUDA GPU
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    /// Helper: set up device + upload a slice.
    fn setup(data: &[f32]) -> (GpuDevice, CudaBuffer<f32>) {
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let buf = cpu_to_gpu(data, &dev).expect("cpu_to_gpu");
        (dev, buf)
    }

    /// Round-trip helper: download a GPU buffer and compare against expected
    /// CPU output element-wise.
    fn assert_buf_eq(buf: &CudaBuffer<f32>, device: &GpuDevice, expected: &[f32]) {
        let host = gpu_to_cpu(buf, device).expect("gpu_to_cpu");
        assert_eq!(host.len(), expected.len(), "length mismatch");
        for (i, (&got, &exp)) in host.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "element {i}: got {got}, expected {exp}",
            );
        }
    }

    // -- gpu_add -------------------------------------------------------------

    #[test]
    fn add_basic() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_add(&a, &b, &dev).expect("gpu_add");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn add_empty() {
        let (dev, a) = setup(&[]);
        let b = cpu_to_gpu::<f32>(&[], &dev).expect("cpu_to_gpu b");
        let out = gpu_add(&a, &b, &dev).expect("gpu_add empty");
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn add_large() {
        let n = 100_000;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_add(&a, &b, &dev).expect("gpu_add large");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn add_length_mismatch() {
        let (dev, a) = setup(&[1.0, 2.0, 3.0]);
        let b = cpu_to_gpu(&[1.0, 2.0], &dev).expect("cpu_to_gpu b");
        let err = gpu_add(&a, &b, &dev).unwrap_err();
        match err {
            GpuError::LengthMismatch { a: 3, b: 2 } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- gpu_sub -------------------------------------------------------------

    #[test]
    fn sub_basic() {
        let a_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let b_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x - y).collect();

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_sub(&a, &b, &dev).expect("gpu_sub");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn sub_negative_result() {
        let a_data = vec![1.0f32, 2.0];
        let b_data = vec![5.0f32, 10.0];
        let expected: Vec<f32> = vec![-4.0, -8.0];

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_sub(&a, &b, &dev).expect("gpu_sub");
        assert_buf_eq(&out, &dev, &expected);
    }

    // -- gpu_mul -------------------------------------------------------------

    #[test]
    fn mul_basic() {
        let a_data = vec![2.0f32, 3.0, 4.0, 5.0];
        let b_data = vec![10.0f32, 10.0, 10.0, 10.0];
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x * y).collect();

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_mul(&a, &b, &dev).expect("gpu_mul");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn mul_by_zero() {
        let a_data = vec![1.0f32, 2.0, 3.0];
        let b_data = vec![0.0f32, 0.0, 0.0];
        let expected = vec![0.0f32, 0.0, 0.0];

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_mul(&a, &b, &dev).expect("gpu_mul");
        assert_buf_eq(&out, &dev, &expected);
    }

    // -- gpu_neg -------------------------------------------------------------

    #[test]
    fn neg_basic() {
        let a_data = vec![1.0f32, -2.0, 3.0, 0.0, -5.5];
        let expected: Vec<f32> = a_data.iter().map(|x| -x).collect();

        let (dev, a) = setup(&a_data);
        let out = gpu_neg(&a, &dev).expect("gpu_neg");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn neg_double_negation() {
        let a_data = vec![1.0f32, -2.0, 3.0];
        let (dev, a) = setup(&a_data);
        let neg1 = gpu_neg(&a, &dev).expect("gpu_neg 1");
        let neg2 = gpu_neg(&neg1, &dev).expect("gpu_neg 2");
        assert_buf_eq(&neg2, &dev, &a_data);
    }

    // -- gpu_relu ------------------------------------------------------------

    #[test]
    fn relu_basic() {
        let a_data = vec![-3.0f32, -1.0, 0.0, 1.0, 3.0];
        let expected = vec![0.0f32, 0.0, 0.0, 1.0, 3.0];

        let (dev, a) = setup(&a_data);
        let out = gpu_relu(&a, &dev).expect("gpu_relu");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn relu_all_negative() {
        let a_data = vec![-5.0f32, -0.1, -100.0];
        let expected = vec![0.0f32, 0.0, 0.0];

        let (dev, a) = setup(&a_data);
        let out = gpu_relu(&a, &dev).expect("gpu_relu");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn relu_all_positive() {
        let a_data = vec![0.1f32, 1.0, 100.0];

        let (dev, a) = setup(&a_data);
        let out = gpu_relu(&a, &dev).expect("gpu_relu");
        assert_buf_eq(&out, &dev, &a_data);
    }

    #[test]
    fn relu_empty() {
        let (dev, a) = setup(&[]);
        let out = gpu_relu(&a, &dev).expect("gpu_relu empty");
        assert_eq!(out.len(), 0);
    }
}
