//! CUDA implementation of the [`GpuBackend`] trait from ferrotorch-core.
//!
//! This module bridges the existing GPU operations (`gpu_add`, `gpu_matmul_f32`,
//! etc.) to the type-erased [`GpuBackend`] dispatch interface, enabling
//! ferrotorch-core to call GPU operations without depending on this crate
//! directly.
//!
//! # Initialization
//!
//! Call [`init_cuda_backend`] once at startup (typically via `ferrotorch::init()`).
//! This creates a [`CudaBackendImpl`], initializes CUDA device 0, and registers
//! it with [`ferrotorch_core::gpu_dispatch::register_gpu_backend`].

use std::sync::Arc;

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::gpu_dispatch::{GpuBackend, GpuBufferHandle, GpuRngState};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;

// ---------------------------------------------------------------------------
// CudaBackendImpl
// ---------------------------------------------------------------------------

/// CUDA implementation of the [`GpuBackend`] trait.
///
/// Holds one or more [`GpuDevice`] handles (currently device 0 only) and
/// delegates every trait method to the corresponding function in
/// [`crate::kernels`], [`crate::blas`], or [`crate::transfer`].
pub struct CudaBackendImpl {
    devices: Vec<Arc<GpuDevice>>,
}

impl CudaBackendImpl {
    /// Create a new CUDA backend, initializing device 0.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if CUDA initialization fails
    /// (e.g. no GPU available, driver not loaded).
    pub fn new() -> FerrotorchResult<Self> {
        let device = Arc::new(
            GpuDevice::new(0).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("CUDA init failed: {e}"),
            })?,
        );
        Ok(Self {
            devices: vec![device],
        })
    }

    /// Get the device for ordinal 0 (the default device).
    pub fn default_device(&self) -> FerrotorchResult<&Arc<GpuDevice>> {
        self.device(0)
    }

    /// Look up a device by ordinal.
    fn device(&self, ordinal: usize) -> FerrotorchResult<&Arc<GpuDevice>> {
        self.devices
            .get(ordinal)
            .ok_or(FerrotorchError::InvalidArgument {
                message: format!("CUDA device {ordinal} not available"),
            })
    }

    /// Wrap a `CudaBuffer<f32>` into a type-erased [`GpuBufferHandle`].
    fn wrap_buffer(buf: CudaBuffer<f32>, ordinal: usize) -> GpuBufferHandle {
        let len = buf.len();
        GpuBufferHandle::new(Box::new(buf), ordinal, len)
    }

    /// Wrap a `CudaBuffer<f64>` into a type-erased [`GpuBufferHandle`].
    fn wrap_buffer_f64(buf: CudaBuffer<f64>, ordinal: usize) -> GpuBufferHandle {
        let len = buf.len();
        GpuBufferHandle::new(Box::new(buf), ordinal, len)
    }

    /// Extract a `&CudaBuffer<f32>` from a [`GpuBufferHandle`].
    fn unwrap_buffer(handle: &GpuBufferHandle) -> FerrotorchResult<&CudaBuffer<f32>> {
        handle
            .downcast_ref::<CudaBuffer<f32>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<f32>".into(),
            })
    }

    /// Extract a `&mut CudaBuffer<f32>` from a [`GpuBufferHandle`].
    fn unwrap_buffer_mut(handle: &mut GpuBufferHandle) -> FerrotorchResult<&mut CudaBuffer<f32>> {
        handle
            .downcast_mut::<CudaBuffer<f32>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<f32>".into(),
            })
    }

    /// Extract a `&mut CudaBuffer<f64>` from a [`GpuBufferHandle`].
    fn unwrap_buffer_f64_mut(
        handle: &mut GpuBufferHandle,
    ) -> FerrotorchResult<&mut CudaBuffer<f64>> {
        handle
            .downcast_mut::<CudaBuffer<f64>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<f64>".into(),
            })
    }

    /// Extract a `&CudaBuffer<f64>` from a [`GpuBufferHandle`].
    fn unwrap_buffer_f64(handle: &GpuBufferHandle) -> FerrotorchResult<&CudaBuffer<f64>> {
        handle
            .downcast_ref::<CudaBuffer<f64>>()
            .ok_or(FerrotorchError::InvalidArgument {
                message: "GPU handle does not contain a CudaBuffer<f64>".into(),
            })
    }

    /// Convert a [`crate::error::GpuError`] into a [`FerrotorchError`].
    fn map_gpu_err(e: crate::error::GpuError) -> FerrotorchError {
        FerrotorchError::InvalidArgument {
            message: format!("{e}"),
        }
    }
}

// ---------------------------------------------------------------------------
// GpuBackend implementation
// ---------------------------------------------------------------------------

impl GpuBackend for CudaBackendImpl {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn raw_device_ptr(&self, handle: &GpuBufferHandle) -> *const std::ffi::c_void {
        use cudarc::driver::DevicePtr;
        let dev = match self.device(handle.device_ordinal()) {
            Ok(d) => d,
            Err(_) => return std::ptr::null(),
        };
        let stream = dev.stream();
        if let Ok(buf) = Self::unwrap_buffer(handle) {
            let (ptr, _sync) = buf.inner().device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else if let Ok(buf) = Self::unwrap_buffer_f64(handle) {
            let (ptr, _sync) = buf.inner().device_ptr(&stream);
            ptr as *const std::ffi::c_void
        } else {
            std::ptr::null()
        }
    }

    fn raw_device_ptr_mut(&self, handle: &mut GpuBufferHandle) -> *mut std::ffi::c_void {
        use cudarc::driver::DevicePtrMut;
        let ordinal = handle.device_ordinal();
        let dev = match self.device(ordinal) {
            Ok(d) => d,
            Err(_) => return std::ptr::null_mut(),
        };
        let stream = dev.stream();
        if let Some(buf) = handle.downcast_mut::<CudaBuffer<f32>>() {
            let (ptr, _sync) = buf.inner_mut().device_ptr_mut(&stream);
            ptr as *mut std::ffi::c_void
        } else if let Some(buf) = handle.downcast_mut::<CudaBuffer<f64>>() {
            let (ptr, _sync) = buf.inner_mut().device_ptr_mut(&stream);
            ptr as *mut std::ffi::c_void
        } else {
            std::ptr::null_mut()
        }
    }

    fn buffer_elem_size(&self, handle: &GpuBufferHandle) -> usize {
        if Self::unwrap_buffer(handle).is_ok() {
            4 // f32
        } else if Self::unwrap_buffer_f64(handle).is_ok() {
            8 // f64
        } else {
            0
        }
    }

    fn cpu_to_gpu(
        &self,
        data: &[u8],
        elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device)?;
        match elem_size {
            4 => {
                let count = data.len() / 4;
                // SAFETY:
                // - The caller (ferrotorch-core) guarantees that `data` is the
                //   byte serialisation of a contiguous `&[f32]`; this is the
                //   contract of the `cpu_to_gpu` trait method whose signature
                //   accepts `&[u8]` + `elem_size` as a type-erased façade
                //   across f32/f64 (see trait definition in
                //   ferrotorch-core/src/gpu_dispatch.rs:146-151).
                // - The `elem_size == 4` arm is only entered when the upstream
                //   caller asserted f32 layout. The canonical caller pattern
                //   (ferrotorch-core/src/storage.rs:117-133, `on_device`)
                //   originates a `Vec<T>` and forwards `size_of::<T>()` as
                //   `elem_size`; reaching this arm means `T == f32` upstream.
                // - Alignment: f32 has size 4 and align 4 (Rust reference,
                //   primitive data layout). The source `Vec<f32>` allocation
                //   is 4-byte-aligned; that alignment propagates through
                //   `data.as_ptr()` because the `&[u8]` view of an f32 slice
                //   is always at least 4-byte-aligned at its start.
                // - Length: `count = data.len() / 4` is exact when the caller
                //   honours the contract (`data.len() == count * size_of::<f32>()`);
                //   any remainder is a caller-contract violation, not a bug
                //   we must defend against here.
                // - Provenance: `slice::from_raw_parts` reads `count * 4` bytes,
                //   which equals `data.len()` exactly, so the new slice spans
                //   no memory beyond the input.
                // - Lifetime: the reinterpreted `&[f32]` is bound by `data`
                //   (a `&[u8]` parameter borrowed for the duration of this
                //   call). The slice is consumed by `cpu_to_gpu` on the next
                //   line before this stack frame returns; no dangling
                //   reference can escape.
                // - No `&mut` aliases: `data: &[u8]` is a shared borrow, so
                //   no concurrent `&mut [f32]` to the same allocation can
                //   exist while this call is active.
                let f32_data: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, count) };
                let buf = crate::transfer::cpu_to_gpu(f32_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer(buf, device))
            }
            8 => {
                let count = data.len() / 8;
                // SAFETY:
                // - The caller (ferrotorch-core) guarantees that `data` is the
                //   byte serialisation of a contiguous `&[f64]`; this is the
                //   `elem_size == 8` arm's precondition documented at the
                //   trait method (ferrotorch-core/src/gpu_dispatch.rs:146-151).
                //   Canonical caller path: ferrotorch-core/src/storage.rs:117-133
                //   forwards `size_of::<T>()`; reaching this arm means
                //   `T == f64` upstream.
                // - Alignment: f64 has size 8 and align 8 (Rust reference,
                //   primitive data layout). The source `Vec<f64>` allocation
                //   guarantees 8-byte alignment which propagates to
                //   `data.as_ptr()` for the byte view of an f64 slice.
                // - Length: `count = data.len() / 8` is exact when the caller
                //   honours the contract; remainder bytes would be a
                //   caller-contract violation upstream of this site.
                // - Provenance: `slice::from_raw_parts` reads `count * 8`
                //   bytes which equals `data.len()` exactly, so the new
                //   slice covers exactly the byte range of `data` with no
                //   overrun.
                // - Lifetime: the reinterpreted `&[f64]` is bounded by `data`
                //   and consumed by `cpu_to_gpu` on the next line, never
                //   escaping this stack frame.
                // - No `&mut` aliases: shared `&[u8]` input rules out any
                //   concurrent `&mut [f64]` aliasing of the same allocation.
                let f64_data: &[f64] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, count) };
                let buf = crate::transfer::cpu_to_gpu(f64_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_f64(buf, device))
            }
            other => Err(FerrotorchError::InvalidArgument {
                message: format!("cpu_to_gpu: unsupported elem_size {other} (expected 4 or 8)"),
            }),
        }
    }

    fn cpu_to_gpu_pinned(
        &self,
        data: &[u8],
        elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device)?;
        match elem_size {
            4 => {
                let count = data.len() / 4;
                // SAFETY:
                // - The caller (ferrotorch-core) guarantees that `data` is the
                //   byte serialisation of a contiguous `&[f32]`; this is the
                //   contract of the `cpu_to_gpu_pinned` trait method whose
                //   signature accepts `&[u8]` + `elem_size` as a type-erased
                //   façade across f32/f64 (see trait definition in
                //   ferrotorch-core gpu_dispatch.rs).
                // - The `elem_size == 4` arm is only entered when the upstream
                //   caller asserted f32 layout. f32 has size 4 and align 4
                //   (Rust reference: <https://doc.rust-lang.org/reference/type-layout.html#primitive-data-layout>),
                //   so `count = data.len() / 4` is exact and `data.as_ptr()`
                //   inherits 4-byte alignment from any f32 source Vec.
                // - Lifetime: the reinterpreted `&[f32]` is bounded by `data`
                //   (a `&[u8]` parameter borrowed for the duration of this
                //   call). The slice is consumed by `cpu_to_gpu_pinned` on
                //   line 226 before this stack frame returns; no dangling
                //   reference can escape.
                // - Provenance: `slice::from_raw_parts` requires the entire
                //   `count * 4` byte range to be readable; that is exactly
                //   `data.len()` bytes (since `count = data.len() / 4`), so
                //   the new slice spans no memory beyond the input.
                // - No `&mut` aliases: `data` is a shared `&[u8]`, so no
                //   concurrent `&mut [f32]` to the same allocation can exist.
                let f32_data: &[f32] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, count) };
                let buf =
                    crate::transfer::cpu_to_gpu_pinned(f32_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer(buf, device))
            }
            8 => {
                let count = data.len() / 8;
                // SAFETY:
                // - The caller (ferrotorch-core) guarantees that `data` is the
                //   byte serialisation of a contiguous `&[f64]`; this is the
                //   `elem_size == 8` arm's precondition documented at the
                //   trait method (see gpu_dispatch.rs).
                // - f64 has size 8 and align 8 (Rust reference: primitive data
                //   layout). `count = data.len() / 8` is exact; the source
                //   `Vec<f64>` allocation guarantees 8-byte alignment which
                //   propagates to `data.as_ptr()`.
                // - Lifetime: the reinterpreted `&[f64]` is bounded by `data`
                //   and consumed by `cpu_to_gpu_pinned` on line 234, never
                //   escaping this stack frame.
                // - Provenance: `count * 8 == data.len()` so the new slice
                //   covers exactly the byte range of `data` with no overrun.
                // - No `&mut` aliases: shared `&[u8]` input rules out any
                //   concurrent `&mut [f64]` aliasing of the same allocation.
                let f64_data: &[f64] =
                    unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f64, count) };
                let buf =
                    crate::transfer::cpu_to_gpu_pinned(f64_data, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_f64(buf, device))
            }
            other => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "cpu_to_gpu_pinned: unsupported elem_size {other} (expected 4 or 8)"
                ),
            }),
        }
    }

    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>> {
        let dev = self.device(handle.device_ordinal())?;

        // Try f32 first, then f64.
        if let Ok(buf) = Self::unwrap_buffer(handle) {
            let f32_data = crate::transfer::gpu_to_cpu(buf, dev).map_err(Self::map_gpu_err)?;

            // Reinterpret Vec<f32> as Vec<u8> without copying.
            // SAFETY: f32 has alignment 4 and size 4. We adjust len and capacity
            // accordingly. The original Vec is consumed via ManuallyDrop so its
            // destructor won't free the allocation.
            let bytes = unsafe {
                let mut v = std::mem::ManuallyDrop::new(f32_data);
                let ptr = v.as_mut_ptr() as *mut u8;
                let len = v.len() * 4;
                let cap = v.capacity() * 4;
                Vec::from_raw_parts(ptr, len, cap)
            };
            Ok(bytes)
        } else if let Ok(buf) = Self::unwrap_buffer_f64(handle) {
            let f64_data = crate::transfer::gpu_to_cpu(buf, dev).map_err(Self::map_gpu_err)?;

            // Reinterpret Vec<f64> as Vec<u8> without copying.
            // SAFETY: f64 has alignment 8 and size 8. We adjust len and capacity
            // accordingly. The original Vec is consumed via ManuallyDrop so its
            // destructor won't free the allocation.
            let bytes = unsafe {
                let mut v = std::mem::ManuallyDrop::new(f64_data);
                let ptr = v.as_mut_ptr() as *mut u8;
                let len = v.len() * 8;
                let cap = v.capacity() * 8;
                Vec::from_raw_parts(ptr, len, cap)
            };
            Ok(bytes)
        } else {
            Err(FerrotorchError::InvalidArgument {
                message: "gpu_to_cpu: handle is neither CudaBuffer<f32> nor CudaBuffer<f64>".into(),
            })
        }
    }

    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        // Clone via GPU -> CPU -> GPU round-trip.
        // Correct but not optimal; a device-to-device memcpy would be better.
        let bytes = self.gpu_to_cpu(handle)?;
        // Determine elem_size from the concrete buffer type.
        let elem_size = if handle.downcast_ref::<CudaBuffer<f64>>().is_some() {
            8
        } else {
            4
        };
        self.cpu_to_gpu(&bytes, elem_size, handle.device_ordinal())
    }

    fn has_inf_nan_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<bool> {
        // TODO #687: replace with a real GPU reduction kernel.
        // The trait used to provide this body as a default impl, which made
        // the synchronous device-to-host readback invisible at every call
        // site. Moving the body here keeps behaviour identical but makes the
        // host-readback explicit at the backend impl, where a real GPU
        // kernel can be slotted in.
        let bytes = self.gpu_to_cpu(a)?;
        // SAFETY: gpu_to_cpu returns a byte buffer whose layout is exactly
        // `bytes.len() / 4` consecutive `f32` values (4-byte little-endian
        // IEEE 754). The pointer comes from a `Vec<u8>` which is
        // 4-byte-aligned for f32 on every platform we support, and the
        // resulting slice's lifetime is bounded by `bytes`.
        let floats: &[f32] =
            unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<f32>(), bytes.len() / 4) };
        Ok(floats.iter().any(|v| !v.is_finite()))
    }

    fn alloc_zeros(
        &self,
        len: usize,
        elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device)?;
        match elem_size {
            4 => {
                let buf = crate::transfer::alloc_zeros_f32(len, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer(buf, device))
            }
            8 => {
                let buf = crate::transfer::alloc_zeros_f64(len, dev).map_err(Self::map_gpu_err)?;
                Ok(Self::wrap_buffer_f64(buf, device))
            }
            other => Err(FerrotorchError::InvalidArgument {
                message: format!("alloc_zeros: unsupported elem_size {other} (expected 4 or 8)"),
            }),
        }
    }

    // -- Elementwise f32 ------------------------------------------------------

    fn add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_add(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sub(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_mul(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn neg_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_neg(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn relu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_relu(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn div_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_div(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn exp_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_exp(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn log_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_log(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn sqrt_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sqrt(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn pow_f32(&self, a: &GpuBufferHandle, exponent: f32) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_pow(a_buf, exponent, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn abs_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_abs(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn sigmoid_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sigmoid(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn tanh_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_tanh(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    // -----------------------------------------------------------------------
    // f64 elementwise ops
    // -----------------------------------------------------------------------

    fn add_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_add_f64(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn sub_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sub_f64(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn mul_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_mul_f64(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn div_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_div_f64(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn neg_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_neg_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn relu_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_relu_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn scale_f64(&self, a: &GpuBufferHandle, scalar: f64) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_scale_f64(a_buf, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn exp_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_exp_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn log_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_log_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn sqrt_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sqrt_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn pow_f64(&self, a: &GpuBufferHandle, exponent: f64) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_pow_f64(a_buf, exponent, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn abs_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_abs_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn sigmoid_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_sigmoid_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn tanh_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_tanh_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // f64 backward ops
    fn relu_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_relu_backward_f64(g_buf, i_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn sigmoid_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let o_buf = Self::unwrap_buffer_f64(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_sigmoid_backward_f64(g_buf, o_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn tanh_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let o_buf = Self::unwrap_buffer_f64(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_tanh_backward_f64(g_buf, o_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    // f64 activation forward ops

    fn gelu_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn gelu_tanh_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_tanh_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn gelu_erf_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_erf_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn silu_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_silu_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn elu_f64(&self, a: &GpuBufferHandle, alpha: f64) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_elu_f64(a_buf, alpha, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn mish_f64(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_mish_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn clamp_f64(
        &self,
        a: &GpuBufferHandle,
        min_val: f64,
        max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_clamp_f64(a_buf, min_val, max_val, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn clamp_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
        min_val: f64,
        max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_clamp_backward_f64(g_buf, i_buf, min_val, max_val, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    // f64 activation backward ops

    fn gelu_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_gelu_backward_f64(g_buf, i_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn gelu_backward_tanh_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward_tanh_f64(g_buf, i_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn gelu_backward_erf_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward_erf_f64(g_buf, i_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn silu_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_silu_backward_f64(g_buf, i_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn elu_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
        alpha: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_elu_backward_f64(g_buf, i_buf, alpha, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn mish_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer_f64(grad)?;
        let i_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_mish_backward_f64(g_buf, i_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    // f64 cumulative ops
    fn cumsum_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_cumsum_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn cumprod_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_cumprod_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn cummax_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_cummax_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(vals, ord),
            Self::wrap_buffer_f64(idxs, ord),
        ))
    }

    fn cummin_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_cummin_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(vals, ord),
            Self::wrap_buffer_f64(idxs, ord),
        ))
    }

    fn logcumsumexp_f64(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_logcumsumexp_f64(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // f64 shape ops
    fn transpose_2d_f64(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_transpose_2d_f64(a_buf, m, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn permute_0213_f64(
        &self,
        a: &GpuBufferHandle,
        d0: usize,
        d1: usize,
        d2: usize,
        d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_permute_0213_f64(a_buf, d0, d1, d2, d3, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // f64 broadcast ops
    fn broadcast_add_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_add_f64(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn broadcast_sub_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_sub_f64(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn broadcast_mul_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_mul_f64(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn broadcast_div_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_div_f64(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // f64 reduction ops
    fn sum_f64(&self, a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_sum_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn prod_f64(&self, a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_prod_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn min_f64(&self, a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_min_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn max_f64(&self, a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_max_f64(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn masked_min_f64(
        &self,
        data: &GpuBufferHandle,
        mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let d_buf = Self::unwrap_buffer_f64(data)?;
        let m_buf = Self::unwrap_buffer_f64(mask_f)?;
        let dev = self.device(data.device_ordinal())?;
        let result = crate::kernels::gpu_masked_reduce_min_f64(d_buf, m_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, data.device_ordinal()))
    }

    fn masked_max_f64(
        &self,
        data: &GpuBufferHandle,
        mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let d_buf = Self::unwrap_buffer_f64(data)?;
        let m_buf = Self::unwrap_buffer_f64(mask_f)?;
        let dev = self.device(data.device_ordinal())?;
        let result = crate::kernels::gpu_masked_reduce_max_f64(d_buf, m_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, data.device_ordinal()))
    }

    fn sum_axis_f64(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product();
        let result = crate::kernels::gpu_sum_axis_f64(a_buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // f64 softmax / log-softmax / layernorm / rmsnorm

    fn softmax_f64(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_softmax_f64(a_buf, rows, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn softmax_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer_f64(grad)?;
        let output_buf = Self::unwrap_buffer_f64(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_softmax_backward_f64(grad_buf, output_buf, cols, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn log_softmax_f64(
        &self,
        a: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_log_softmax_f64(a_buf, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn log_softmax_backward_f64(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer_f64(grad)?;
        let output_buf = Self::unwrap_buffer_f64(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_log_softmax_backward_f64(grad_buf, output_buf, cols, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    fn layernorm_f64(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let b_buf = Self::unwrap_buffer_f64(bias)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_layernorm_f64(in_buf, w_buf, b_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn layernorm_backward_f64(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let go_buf = Self::unwrap_buffer_f64(grad_output)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let (gi, gw, gb) =
            crate::kernels::gpu_layernorm_backward_f64(in_buf, go_buf, w_buf, rows, cols, eps, dev)
                .map_err(Self::map_gpu_err)?;
        let ordinal = input.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(gi, ordinal),
            Self::wrap_buffer_f64(gw, ordinal),
            Self::wrap_buffer_f64(gb, ordinal),
        ))
    }

    fn rmsnorm_f64(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_rmsnorm_f64(in_buf, w_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn rmsnorm_backward_f64(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let go_buf = Self::unwrap_buffer_f64(grad_output)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let (gi, gw) =
            crate::kernels::gpu_rmsnorm_backward_f64(in_buf, go_buf, w_buf, rows, cols, eps, dev)
                .map_err(Self::map_gpu_err)?;
        let ordinal = input.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(gi, ordinal),
            Self::wrap_buffer_f64(gw, ordinal),
        ))
    }

    // f64 embedding / scatter / indexing

    fn embed_lookup_f64(
        &self,
        idx: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(idx)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(idx.device_ordinal())?;
        let result = crate::kernels::gpu_embed_lookup_f64(idx_buf, w_buf, d, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, idx.device_ordinal()))
    }

    fn embed_lookup_batch_f64(
        &self,
        indices: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        n: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(indices)?;
        let w_buf = Self::unwrap_buffer_f64(weight)?;
        let dev = self.device(indices.device_ordinal())?;
        let result = crate::kernels::gpu_embed_lookup_batch_f64(idx_buf, w_buf, n, d, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, indices.device_ordinal()))
    }

    fn scatter_add_rows_f64(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        num_embeddings: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let go_buf = Self::unwrap_buffer_f64(grad_output)?;
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(grad_output.device_ordinal())?;
        let result =
            crate::kernels::gpu_scatter_add_rows_f64(go_buf, idx_buf, num_embeddings, d, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad_output.device_ordinal()))
    }

    // f64 masked fill / masked zero
    //
    // The f64 kernels expect CudaBuffer<u8> for the mask, but the trait
    // provides a GpuBufferHandle containing CudaBuffer<f32> (1.0/0.0 encoding).
    // We convert f32 mask -> u8 mask via a CPU roundtrip.

    fn masked_fill_f64(
        &self,
        input: &GpuBufferHandle,
        mask: &GpuBufferHandle,
        value: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        let mask_f32 = Self::unwrap_buffer(mask)?;
        let dev = self.device(input.device_ordinal())?;
        // Convert f32 mask to u8 mask on GPU via CPU roundtrip
        let mask_host = crate::transfer::gpu_to_cpu(mask_f32, dev).map_err(Self::map_gpu_err)?;
        let mask_u8: Vec<u8> = mask_host
            .iter()
            .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
            .collect();
        let mask_gpu = crate::transfer::cpu_to_gpu(&mask_u8, dev).map_err(Self::map_gpu_err)?;
        let result = crate::kernels::gpu_masked_fill_f64(input_buf, &mask_gpu, value, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn masked_zero_f64(
        &self,
        grad: &GpuBufferHandle,
        mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer_f64(grad)?;
        let mask_f32 = Self::unwrap_buffer(mask)?;
        let dev = self.device(grad.device_ordinal())?;
        // Convert f32 mask to u8 mask on GPU via CPU roundtrip
        let mask_host = crate::transfer::gpu_to_cpu(mask_f32, dev).map_err(Self::map_gpu_err)?;
        let mask_u8: Vec<u8> = mask_host
            .iter()
            .map(|&v| if v != 0.0 { 1u8 } else { 0u8 })
            .collect();
        let mask_gpu = crate::transfer::cpu_to_gpu(&mask_u8, dev).map_err(Self::map_gpu_err)?;
        let result = crate::kernels::gpu_masked_zero_f64(grad_buf, &mask_gpu, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad.device_ordinal()))
    }

    // f64 slice ops

    fn slice_write_f64(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        n_batch: usize,
        d: usize,
        max_len: usize,
        pos: usize,
    ) -> FerrotorchResult<()> {
        let src_buf = Self::unwrap_buffer_f64(src)?;
        let dst_buf = Self::unwrap_buffer_f64_mut(dst)?;
        let dev = self.device(src.device_ordinal())?;
        crate::kernels::gpu_slice_write_f64(src_buf, dst_buf, n_batch, d, max_len, pos, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(())
    }

    fn slice_read_f64(
        &self,
        src: &GpuBufferHandle,
        n_batch: usize,
        d: usize,
        len: usize,
        max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let src_buf = Self::unwrap_buffer_f64(src)?;
        let dev = self.device(src.device_ordinal())?;
        let result = crate::kernels::gpu_slice_read_f64(src_buf, n_batch, d, len, max_len, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, src.device_ordinal()))
    }

    // f64 strided split / cat

    fn strided_split_f64(
        &self,
        input: &GpuBufferHandle,
        total_along_axis: usize,
        split_offset: usize,
        split_size: usize,
        inner_size: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_strided_split_f64(
            in_buf,
            total_along_axis,
            split_offset,
            split_size,
            inner_size,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn strided_cat_f64(
        &self,
        input: &GpuBufferHandle,
        output: &mut GpuBufferHandle,
        total_along_axis: usize,
        cat_offset: usize,
        part_size: usize,
        inner_size: usize,
        n: usize,
    ) -> FerrotorchResult<()> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let out_buf = Self::unwrap_buffer_f64_mut(output)?;
        crate::kernels::gpu_strided_cat_f64(
            in_buf,
            out_buf,
            total_along_axis,
            cat_offset,
            part_size,
            inner_size,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(())
    }

    // f64 indexing ops

    fn index_select_1d_f64(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer_f64(input)?;
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_index_select_1d_f64(input_buf, idx_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn scatter_add_1d_f64(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let go_buf = Self::unwrap_buffer_f64(grad_output)?;
        // indices are always f32-encoded
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(grad_output.device_ordinal())?;
        let result = crate::kernels::gpu_scatter_add_1d_f64(go_buf, idx_buf, input_len, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, grad_output.device_ordinal()))
    }

    fn bmm_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_bmm_f64(a_buf, b_buf, batch, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    #[allow(clippy::too_many_arguments)]
    fn fused_adam_f32(
        &self,
        param: &mut GpuBufferHandle,
        grad: &GpuBufferHandle,
        exp_avg: &mut GpuBufferHandle,
        exp_avg_sq: &mut GpuBufferHandle,
        beta1: f32,
        beta2: f32,
        lr: f32,
        eps: f32,
        bc1: f32,
        bc2: f32,
        weight_decay: f32,
    ) -> FerrotorchResult<()> {
        let ordinal = param.device_ordinal();
        let dev = self.device(ordinal)?;
        let p_buf = Self::unwrap_buffer_mut(param)?;
        let g_buf = Self::unwrap_buffer(grad)?;
        let m_buf = Self::unwrap_buffer_mut(exp_avg)?;
        let v_buf = Self::unwrap_buffer_mut(exp_avg_sq)?;
        crate::kernels::gpu_fused_adam(
            p_buf,
            g_buf,
            m_buf,
            v_buf,
            beta1,
            beta2,
            lr,
            eps,
            bc1,
            bc2,
            weight_decay,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn maxpool2d_f32(
        &self,
        input: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let (out, shape) = crate::kernels::gpu_maxpool2d(
            buf, batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok((Self::wrap_buffer(out, input.device_ordinal()), shape))
    }

    #[allow(clippy::too_many_arguments)]
    fn avgpool2d_f32(
        &self,
        input: &GpuBufferHandle,
        batch: usize,
        channels: usize,
        h_in: usize,
        w_in: usize,
        kh: usize,
        kw: usize,
        sh: usize,
        sw: usize,
        ph: usize,
        pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let (out, shape) = crate::kernels::gpu_avgpool2d(
            buf, batch, channels, h_in, w_in, kh, kw, sh, sw, ph, pw, dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok((Self::wrap_buffer(out, input.device_ordinal()), shape))
    }

    #[allow(clippy::too_many_arguments)]
    fn conv2d_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: Option<&GpuBufferHandle>,
        input_shape: [usize; 4],
        weight_shape: [usize; 4],
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        let input_buf = Self::unwrap_buffer(input)?;
        let weight_buf = Self::unwrap_buffer(weight)?;
        let bias_buf = match bias {
            Some(b) => Some(Self::unwrap_buffer(b)?),
            None => None,
        };
        let dev = self.device(input.device_ordinal())?;
        let (out_buf, out_shape) = crate::conv::gpu_conv2d_f32(
            input_buf,
            weight_buf,
            bias_buf,
            input_shape,
            weight_shape,
            stride,
            padding,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok((
            Self::wrap_buffer(out_buf, input.device_ordinal()),
            out_shape,
        ))
    }

    fn fused_gru_cell_f32(
        &self,
        input_gates: &GpuBufferHandle,
        hidden_gates: &GpuBufferHandle,
        bias_ih: &GpuBufferHandle,
        bias_hh: &GpuBufferHandle,
        hx: &GpuBufferHandle,
        hidden_size: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let ig = Self::unwrap_buffer(input_gates)?;
        let hg = Self::unwrap_buffer(hidden_gates)?;
        let bih = Self::unwrap_buffer(bias_ih)?;
        let bhh = Self::unwrap_buffer(bias_hh)?;
        let hx_buf = Self::unwrap_buffer(hx)?;
        let dev = self.device(input_gates.device_ordinal())?;
        let (hy, ws) =
            crate::kernels::gpu_fused_gru_forward(ig, hg, bih, bhh, hx_buf, hidden_size, dev)
                .map_err(Self::map_gpu_err)?;
        let ord = input_gates.device_ordinal();
        Ok((Self::wrap_buffer(hy, ord), Self::wrap_buffer(ws, ord)))
    }

    fn synchronize(&self, device: usize) -> FerrotorchResult<()> {
        let dev = self.device(device)?;
        dev.stream()
            .synchronize()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("CUDA synchronize failed: {e}"),
            })?;
        Ok(())
    }

    fn stream_count(&self, device: usize) -> usize {
        crate::stream::StreamPool::pool_size(device)
    }

    // -- Linalg f32 -----------------------------------------------------------

    fn matmul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::blas::gpu_matmul_f32(a_buf, b_buf, m, k, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    // -- Reduction f32 --------------------------------------------------------

    fn sum_f32(&self, a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_sum(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn prod_f32(&self, a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_prod(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn min_f32(&self, a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_min(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn max_f32(&self, a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_reduce_max(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn masked_min_f32(
        &self,
        data: &GpuBufferHandle,
        mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let d_buf = Self::unwrap_buffer(data)?;
        let m_buf = Self::unwrap_buffer(mask_f)?;
        let dev = self.device(data.device_ordinal())?;
        let result =
            crate::kernels::gpu_masked_reduce_min(d_buf, m_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, data.device_ordinal()))
    }

    fn masked_max_f32(
        &self,
        data: &GpuBufferHandle,
        mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let d_buf = Self::unwrap_buffer(data)?;
        let m_buf = Self::unwrap_buffer(mask_f)?;
        let dev = self.device(data.device_ordinal())?;
        let result =
            crate::kernels::gpu_masked_reduce_max(d_buf, m_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, data.device_ordinal()))
    }

    // -- Linalg f64 (cuBLAS DGEMM) --------------------------------------------

    fn matmul_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::blas::gpu_matmul_f64(a_buf, b_buf, m, k, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    // -- Broadcast binary f32 -------------------------------------------------

    fn broadcast_add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_add(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn broadcast_sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_sub(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn broadcast_mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_mul(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn broadcast_div_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_broadcast_div(a_buf, b_buf, a_shape, b_shape, out_shape, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn softmax_f32(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_softmax(a_buf, rows, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn dropout_f32(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f32,
        seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_dropout(a_buf, threshold, scale, seed, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn dropout_philox_f32(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuRngState)> {
        let device_ordinal = a.device_ordinal();
        let n = a.len();

        // Snapshot the current RNG state and advance it.
        let rng_state = {
            let mut mgr = crate::rng::cuda_rng_manager().lock().map_err(|_| {
                FerrotorchError::InvalidArgument {
                    message: "failed to lock CUDA RNG manager".into(),
                }
            })?;
            let philox_gen = mgr.generator(device_ordinal);
            let state = philox_gen.get_state();
            // Advance by ceil(n/4) counters (each counter produces 4 u32 values)
            let counters_needed = n.div_ceil(4);
            philox_gen.advance(counters_needed as u64);
            state
        };

        // Use the Philox state as the seed for the dropout kernel.
        // We encode the Philox counter+seed into a u32 seed that the existing
        // dropout kernel can use. For full correctness on GPU, we should use
        // the Philox uniform kernel to generate the mask, then apply it.
        // However, for consistency between GPU forward and CPU backward mask
        // regeneration, we use the Philox state to deterministically derive a
        // seed for the existing kernel.
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(device_ordinal)?;

        // Use the Philox counter XOR seed as the dropout kernel's seed.
        // This gives us deterministic behavior tied to the Philox state.
        let derived_seed = (rng_state.counter ^ rng_state.seed) as u32;
        let result = crate::kernels::gpu_dropout(a_buf, threshold, scale, derived_seed, dev)
            .map_err(Self::map_gpu_err)?;

        let gpu_rng_state = GpuRngState::new(
            rng_state.counter,
            rng_state.seed,
            rng_state.offset,
            device_ordinal,
        );

        Ok((Self::wrap_buffer(result, device_ordinal), gpu_rng_state))
    }

    fn dropout_f64(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f64,
        seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_dropout_f64(a_buf, threshold, scale, seed, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }

    fn dropout_philox_f64(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuRngState)> {
        let device_ordinal = a.device_ordinal();
        let n = a.len();

        let rng_state = {
            let mut mgr = crate::rng::cuda_rng_manager().lock().map_err(|_| {
                FerrotorchError::InvalidArgument {
                    message: "failed to lock CUDA RNG manager".into(),
                }
            })?;
            let philox_gen = mgr.generator(device_ordinal);
            let state = philox_gen.get_state();
            let counters_needed = n.div_ceil(4);
            philox_gen.advance(counters_needed as u64);
            state
        };

        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(device_ordinal)?;
        let derived_seed = (rng_state.counter ^ rng_state.seed) as u32;
        let result = crate::kernels::gpu_dropout_f64(a_buf, threshold, scale, derived_seed, dev)
            .map_err(Self::map_gpu_err)?;

        let gpu_rng_state = GpuRngState::new(
            rng_state.counter,
            rng_state.seed,
            rng_state.offset,
            device_ordinal,
        );

        Ok((Self::wrap_buffer_f64(result, device_ordinal), gpu_rng_state))
    }

    fn transpose_2d_f32(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_transpose_2d(a_buf, m, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn permute_0213_f32(
        &self,
        a: &GpuBufferHandle,
        d0: usize,
        d1: usize,
        d2: usize,
        d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_permute_0213(a_buf, d0, d1, d2, d3, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn bmm_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_bmm_f32(a_buf, b_buf, batch, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn bmm_f16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::blas::gpu_bmm_f16(a_buf, b_buf, batch, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn gelu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn gelu_tanh_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_tanh(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn gelu_erf_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_erf(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn layernorm_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let b_buf = Self::unwrap_buffer(bias)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_layernorm(in_buf, w_buf, b_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn rmsnorm_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_rmsnorm(in_buf, w_buf, rows, cols, eps, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn rmsnorm_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer(input)?;
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let (gi, gw) =
            crate::kernels::gpu_rmsnorm_backward(in_buf, go_buf, w_buf, rows, cols, eps, dev)
                .map_err(Self::map_gpu_err)?;
        let ordinal = input.device_ordinal();
        Ok((
            Self::wrap_buffer(gi, ordinal),
            Self::wrap_buffer(gw, ordinal),
        ))
    }

    fn slice_write_f32(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        n_batch: usize,
        d: usize,
        max_len: usize,
        pos: usize,
    ) -> FerrotorchResult<()> {
        let src_buf = Self::unwrap_buffer(src)?;
        let dst_buf =
            dst.downcast_mut::<CudaBuffer<f32>>()
                .ok_or(FerrotorchError::InvalidArgument {
                    message: "slice_write_f32: dst is not CudaBuffer<f32>".into(),
                })?;
        let dev = self.device(src.device_ordinal())?;
        crate::kernels::gpu_slice_write(src_buf, dst_buf, n_batch, d, max_len, pos, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(())
    }

    fn slice_read_f32(
        &self,
        src: &GpuBufferHandle,
        n_batch: usize,
        d: usize,
        len: usize,
        max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let src_buf = Self::unwrap_buffer(src)?;
        let dev = self.device(src.device_ordinal())?;
        let result = crate::kernels::gpu_slice_read(src_buf, n_batch, d, len, max_len, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, src.device_ordinal()))
    }

    fn embed_lookup_f32(
        &self,
        idx: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let idx_buf = Self::unwrap_buffer(idx)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(idx.device_ordinal())?;
        let result =
            crate::kernels::gpu_embed_lookup(idx_buf, w_buf, d, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, idx.device_ordinal()))
    }

    fn embed_lookup_batch_f32(
        &self,
        indices: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        n: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let idx_buf = Self::unwrap_buffer(indices)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(indices.device_ordinal())?;
        let result = crate::kernels::gpu_embed_lookup_batch(idx_buf, w_buf, n, d, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, indices.device_ordinal()))
    }

    fn scatter_add_rows_f32(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        num_embeddings: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(grad_output.device_ordinal())?;
        let result = crate::kernels::gpu_scatter_add_rows(go_buf, idx_buf, num_embeddings, d, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad_output.device_ordinal()))
    }

    fn scale_f32(&self, a: &GpuBufferHandle, scalar: f32) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_scale(a_buf, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn relu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_relu_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn abs_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_abs_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn fill_f32(&self, n: usize, scalar: f32, ordinal: usize) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(ordinal)?;
        let result = crate::kernels::gpu_fill_f32(n, scalar, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, ordinal))
    }

    fn gelu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn gelu_backward_tanh_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward_tanh(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn gelu_backward_erf_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_gelu_backward_erf(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn cumsum_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_cumsum(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn cumprod_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_cumprod(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn cummax_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_cummax(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer(vals, ord), Self::wrap_buffer(idxs, ord)))
    }

    fn cummin_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (vals, idxs) = crate::kernels::gpu_cummin(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer(vals, ord), Self::wrap_buffer(idxs, ord)))
    }

    fn logcumsumexp_f32(
        &self,
        a: &GpuBufferHandle,
        outer: usize,
        dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_logcumsumexp(a_buf, outer, dim_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn clamp_f32(
        &self,
        a: &GpuBufferHandle,
        min_val: f32,
        max_val: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_clamp(a_buf, min_val, max_val, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn clamp_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
        min_val: f32,
        max_val: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let g_buf = Self::unwrap_buffer(grad)?;
        let i_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_clamp_backward(g_buf, i_buf, min_val, max_val, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn silu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_silu(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn silu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_silu_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn elu_f32(&self, a: &GpuBufferHandle, alpha: f32) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_elu(a_buf, alpha, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn elu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
        alpha: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_elu_backward(grad_buf, input_buf, alpha, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn mish_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result = crate::kernels::gpu_mish(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn mish_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let input_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_mish_backward(grad_buf, input_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn log_softmax_f32(
        &self,
        a: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_log_softmax(a_buf, cols, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn log_softmax_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let output_buf = Self::unwrap_buffer(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_log_softmax_backward(grad_buf, output_buf, cols, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn index_select_1d_f32(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_index_select_1d(input_buf, idx_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn scatter_add_1d_f32(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let idx_buf = Self::unwrap_buffer(indices)?;
        let dev = self.device(grad_output.device_ordinal())?;
        let result = crate::kernels::gpu_scatter_add_1d(go_buf, idx_buf, input_len, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad_output.device_ordinal()))
    }

    fn masked_fill_f32(
        &self,
        input: &GpuBufferHandle,
        mask: &GpuBufferHandle,
        value: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let input_buf = Self::unwrap_buffer(input)?;
        let mask_buf = Self::unwrap_buffer(mask)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_masked_fill(input_buf, mask_buf, value, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn masked_zero_f32(
        &self,
        grad: &GpuBufferHandle,
        mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let mask_buf = Self::unwrap_buffer(mask)?;
        let dev = self.device(grad.device_ordinal())?;
        let result =
            crate::kernels::gpu_masked_zero(grad_buf, mask_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn sigmoid_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let output_buf = Self::unwrap_buffer(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_sigmoid_backward(grad_buf, output_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn tanh_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let output_buf = Self::unwrap_buffer(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_tanh_backward(grad_buf, output_buf, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn softmax_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        output: &GpuBufferHandle,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let grad_buf = Self::unwrap_buffer(grad)?;
        let output_buf = Self::unwrap_buffer(output)?;
        let dev = self.device(grad.device_ordinal())?;
        let result = crate::kernels::gpu_softmax_backward(grad_buf, output_buf, cols, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, grad.device_ordinal()))
    }

    fn layernorm_backward_f32(
        &self,
        input: &GpuBufferHandle,
        grad_output: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        let in_buf = Self::unwrap_buffer(input)?;
        let go_buf = Self::unwrap_buffer(grad_output)?;
        let w_buf = Self::unwrap_buffer(weight)?;
        let dev = self.device(input.device_ordinal())?;
        let (gi, gw, gb) =
            crate::kernels::gpu_layernorm_backward(in_buf, go_buf, w_buf, rows, cols, eps, dev)
                .map_err(Self::map_gpu_err)?;
        let ordinal = input.device_ordinal();
        Ok((
            Self::wrap_buffer(gi, ordinal),
            Self::wrap_buffer(gw, ordinal),
            Self::wrap_buffer(gb, ordinal),
        ))
    }

    fn sum_axis_f32(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let outer: usize = shape[..axis].iter().product();
        let axis_size = shape[axis];
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let result = crate::kernels::gpu_sum_axis(a_buf, outer, axis_size, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn matmul_f16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::blas::gpu_matmul_f16(a_buf, b_buf, m, k, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn save_rng_state(&self, device: usize) -> FerrotorchResult<GpuRngState> {
        let mut mgr = crate::rng::cuda_rng_manager().lock().map_err(|_| {
            FerrotorchError::InvalidArgument {
                message: "failed to lock CUDA RNG manager".into(),
            }
        })?;
        let state = mgr.get_rng_state(device);
        Ok(GpuRngState::new(
            state.counter,
            state.seed,
            state.offset,
            device,
        ))
    }

    fn restore_rng_state(&self, state: GpuRngState) -> FerrotorchResult<()> {
        let mut mgr = crate::rng::cuda_rng_manager().lock().map_err(|_| {
            FerrotorchError::InvalidArgument {
                message: "failed to lock CUDA RNG manager".into(),
            }
        })?;
        let philox =
            crate::rng::PhiloxState::from_parts(state.counter(), state.seed(), state.offset())
                .map_err(Self::map_gpu_err)?;
        mgr.set_rng_state(state.device(), philox);
        Ok(())
    }

    fn strided_split_f32(
        &self,
        input: &GpuBufferHandle,
        total_along_axis: usize,
        split_offset: usize,
        split_size: usize,
        inner_size: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result = crate::kernels::gpu_strided_split(
            in_buf,
            total_along_axis,
            split_offset,
            split_size,
            inner_size,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn strided_copy_f32(
        &self,
        input: &GpuBufferHandle,
        out_shape: &[usize],
        src_strides: &[isize],
        src_offset: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result =
            crate::kernels::gpu_strided_copy(in_buf, out_shape, src_strides, src_offset, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, input.device_ordinal()))
    }

    fn strided_copy_f64(
        &self,
        input: &GpuBufferHandle,
        out_shape: &[usize],
        src_strides: &[isize],
        src_offset: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let result =
            crate::kernels::gpu_strided_copy_f64(in_buf, out_shape, src_strides, src_offset, dev)
                .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, input.device_ordinal()))
    }

    fn strided_scatter_f32(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        view_shape: &[usize],
        dst_strides: &[isize],
        dst_offset: usize,
    ) -> FerrotorchResult<()> {
        let ord = src.device_ordinal();
        if dst.device_ordinal() != ord {
            return Err(FerrotorchError::DeviceMismatch {
                expected: ferrotorch_core::Device::Cuda(ord),
                got: ferrotorch_core::Device::Cuda(dst.device_ordinal()),
            });
        }
        let src_buf_ptr = Self::unwrap_buffer(src)? as *const CudaBuffer<f32>;
        let dst_buf = Self::unwrap_buffer_mut(dst)?;
        let dev = self.device(ord)?;
        // SAFETY: `src` and `dst` are distinct GpuBufferHandles supplied
        // by the caller; the borrow checker forbids overlapping &/&mut
        // through the same handle, and CudaBuffer<f32> doesn't share
        // mutable state with anything reachable from the &CudaBuffer
        // pointer. The `*const CudaBuffer<f32>` is reborrowed as `&` for
        // the kernel call only.
        let src_ref = unsafe { &*src_buf_ptr };
        crate::kernels::gpu_strided_scatter(
            src_ref,
            dst_buf,
            view_shape,
            dst_strides,
            dst_offset,
            dev,
        )
        .map_err(Self::map_gpu_err)
    }

    fn strided_scatter_f64(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        view_shape: &[usize],
        dst_strides: &[isize],
        dst_offset: usize,
    ) -> FerrotorchResult<()> {
        let ord = src.device_ordinal();
        if dst.device_ordinal() != ord {
            return Err(FerrotorchError::DeviceMismatch {
                expected: ferrotorch_core::Device::Cuda(ord),
                got: ferrotorch_core::Device::Cuda(dst.device_ordinal()),
            });
        }
        let src_buf_ptr = Self::unwrap_buffer_f64(src)? as *const CudaBuffer<f64>;
        let dst_buf = Self::unwrap_buffer_f64_mut(dst)?;
        let dev = self.device(ord)?;
        // SAFETY:
        // - `src` and `dst` are distinct `GpuBufferHandle` parameters
        //   supplied by the caller. The borrow checker forbids the same
        //   handle being passed as both `&` and `&mut` simultaneously, so
        //   `src` and `dst` necessarily refer to non-aliasing handles for
        //   the duration of this call.
        // - `unwrap_buffer_f64` (line 109) produces a `&CudaBuffer<f64>`
        //   borrowed from `src`; we cast to `*const CudaBuffer<f64>` only
        //   to release the borrow on `src` so we can subsequently call
        //   `unwrap_buffer_f64_mut(dst)` without the borrow checker
        //   flagging `dst`'s mutable borrow as conflicting with `src`'s
        //   shared borrow on `self`. The two handles' inner storage is
        //   disjoint by the previous bullet.
        // - `CudaBuffer<f64>` has no interior mutability that would let
        //   `dst_buf`'s mutation affect what `src_ref` reads: it owns a
        //   `cudarc::driver::CudaSlice<f64>` (a device pointer + length)
        //   plus pool-ticket metadata. Mutating one buffer's contents on
        //   the device cannot alias another buffer's allocation because
        //   each `CudaBuffer` owns its own `CudaSlice` allocation.
        // - The `*const CudaBuffer<f64>` is reborrowed as `&CudaBuffer<f64>`
        //   for the duration of the kernel call only; the resulting
        //   reference does not outlive this stack frame.
        // - Pointer is non-null and aligned: it originates from the
        //   `&CudaBuffer<f64>` returned by `unwrap_buffer_f64` (a Rust
        //   reference, by definition non-null and aligned).
        // - Reads through `src_ref` see a fully-initialised
        //   `CudaBuffer<f64>` because the source reference was obtained
        //   from a live `Box<CudaBuffer<f64>>` inside the handle.
        // (Same shape as the f32 sibling at `strided_scatter_f32`; the
        // only differences are the buffer dtype and the kernel name.)
        let src_ref = unsafe { &*src_buf_ptr };
        crate::kernels::gpu_strided_scatter_f64(
            src_ref,
            dst_buf,
            view_shape,
            dst_strides,
            dst_offset,
            dev,
        )
        .map_err(Self::map_gpu_err)
    }

    fn strided_cat_f32(
        &self,
        input: &GpuBufferHandle,
        output: &mut GpuBufferHandle,
        total_along_axis: usize,
        cat_offset: usize,
        part_size: usize,
        inner_size: usize,
        n: usize,
    ) -> FerrotorchResult<()> {
        let in_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let out_buf =
            output
                .downcast_mut::<CudaBuffer<f32>>()
                .ok_or(FerrotorchError::InvalidArgument {
                    message: "strided_cat_f32: output is not CudaBuffer<f32>".into(),
                })?;
        crate::kernels::gpu_strided_cat(
            in_buf,
            out_buf,
            total_along_axis,
            cat_offset,
            part_size,
            inner_size,
            n,
            dev,
        )
        .map_err(Self::map_gpu_err)?;
        Ok(())
    }

    // -- cuSOLVER linear algebra -------------------------------------------------

    fn svd_f32(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let a_host = crate::transfer::gpu_to_cpu(a_buf, dev).map_err(Self::map_gpu_err)?;
        let (u, s, vt) =
            crate::cusolver::gpu_svd_f32(&a_host, m, n, dev).map_err(Self::map_gpu_err)?;
        let u_buf = crate::transfer::cpu_to_gpu(&u, dev).map_err(Self::map_gpu_err)?;
        let s_buf = crate::transfer::cpu_to_gpu(&s, dev).map_err(Self::map_gpu_err)?;
        let vt_buf = crate::transfer::cpu_to_gpu(&vt, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer(u_buf, ord),
            Self::wrap_buffer(s_buf, ord),
            Self::wrap_buffer(vt_buf, ord),
        ))
    }

    fn svd_f64(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let a_host = crate::transfer::gpu_to_cpu(a_buf, dev).map_err(Self::map_gpu_err)?;
        let (u, s, vt) =
            crate::cusolver::gpu_svd_f64(&a_host, m, n, dev).map_err(Self::map_gpu_err)?;
        let u_buf = crate::transfer::cpu_to_gpu(&u, dev).map_err(Self::map_gpu_err)?;
        let s_buf = crate::transfer::cpu_to_gpu(&s, dev).map_err(Self::map_gpu_err)?;
        let vt_buf = crate::transfer::cpu_to_gpu(&vt, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(u_buf, ord),
            Self::wrap_buffer_f64(s_buf, ord),
            Self::wrap_buffer_f64(vt_buf, ord),
        ))
    }

    fn cholesky_f32(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        // (#632) Device-resident Cholesky: cuSOLVER potrf operates on a
        // memcpy_dtod clone of A, then a small host-side mask of the
        // upper triangle.
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let l_buf =
            crate::cusolver::gpu_cholesky_f32_dev(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(l_buf, a.device_ordinal()))
    }

    fn cholesky_f64(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        // (#632) Device-resident Cholesky.
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let l_buf =
            crate::cusolver::gpu_cholesky_f64_dev(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(l_buf, a.device_ordinal()))
    }

    fn solve_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
        nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // (#632) Device-resident path: no host bounce; on-device transposes
        // + cuSOLVER getrf/getrs working on column-major copies.
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let x = crate::cusolver::gpu_solve_f32_dev(a_buf, b_buf, n, nrhs, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(x, a.device_ordinal()))
    }

    fn solve_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        n: usize,
        nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // (#632) Device-resident path: no host bounce.
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let x = crate::cusolver::gpu_solve_f64_dev(a_buf, b_buf, n, nrhs, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(x, a.device_ordinal()))
    }

    fn qr_f32(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let a_host = crate::transfer::gpu_to_cpu(a_buf, dev).map_err(Self::map_gpu_err)?;
        let (q, r) = crate::cusolver::gpu_qr_f32(&a_host, m, n, dev).map_err(Self::map_gpu_err)?;
        let q_buf = crate::transfer::cpu_to_gpu(&q, dev).map_err(Self::map_gpu_err)?;
        let r_buf = crate::transfer::cpu_to_gpu(&r, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer(q_buf, ord), Self::wrap_buffer(r_buf, ord)))
    }

    fn qr_f64(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let a_host = crate::transfer::gpu_to_cpu(a_buf, dev).map_err(Self::map_gpu_err)?;
        let (q, r) = crate::cusolver::gpu_qr_f64(&a_host, m, n, dev).map_err(Self::map_gpu_err)?;
        let q_buf = crate::transfer::cpu_to_gpu(&q, dev).map_err(Self::map_gpu_err)?;
        let r_buf = crate::transfer::cpu_to_gpu(&r, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((
            Self::wrap_buffer_f64(q_buf, ord),
            Self::wrap_buffer_f64(r_buf, ord),
        ))
    }

    // GPU-resident LU factorization (no host bounces). Returns (LU_packed, pivots).
    fn lu_factor_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<i32>)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (lu, ipiv) =
            crate::cusolver::gpu_lu_factor_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        // Pivots are O(n) ints — download to host. The LU matrix (O(n²))
        // stays on device.
        let ipiv_host = crate::transfer::gpu_to_cpu(&ipiv, dev).map_err(Self::map_gpu_err)?;
        Ok((Self::wrap_buffer(lu, a.device_ordinal()), ipiv_host))
    }

    fn lu_factor_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<i32>)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (lu, ipiv) =
            crate::cusolver::gpu_lu_factor_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ipiv_host = crate::transfer::gpu_to_cpu(&ipiv, dev).map_err(Self::map_gpu_err)?;
        Ok((Self::wrap_buffer_f64(lu, a.device_ordinal()), ipiv_host))
    }

    fn lstsq_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        n: usize,
        nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let b_buf = Self::unwrap_buffer(b)?;
        let dev = self.device(a.device_ordinal())?;
        let x = crate::cusolver::gpu_lstsq_f32(a_buf, b_buf, m, n, nrhs, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(x, a.device_ordinal()))
    }

    fn lstsq_f64(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        n: usize,
        nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let b_buf = Self::unwrap_buffer_f64(b)?;
        let dev = self.device(a.device_ordinal())?;
        let x = crate::cusolver::gpu_lstsq_f64(a_buf, b_buf, m, n, nrhs, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(x, a.device_ordinal()))
    }

    fn eig_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (w, v) = crate::cusolver::gpu_eig_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer(w, ord), Self::wrap_buffer(v, ord)))
    }

    fn eig_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (w, v) = crate::cusolver::gpu_eig_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer_f64(w, ord), Self::wrap_buffer_f64(v, ord)))
    }

    // GPU-resident eigh / eigvalsh (no host bounces — see cusolver::gpu_eigh_*).
    fn eigh_f32(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (w, v) = crate::cusolver::gpu_eigh_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer(w, ord), Self::wrap_buffer(v, ord)))
    }

    fn eigh_f64(
        &self,
        a: &GpuBufferHandle,
        n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let (w, v) = crate::cusolver::gpu_eigh_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        let ord = a.device_ordinal();
        Ok((Self::wrap_buffer_f64(w, ord), Self::wrap_buffer_f64(v, ord)))
    }

    fn eigvalsh_f32(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let w = crate::cusolver::gpu_eigvalsh_f32(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(w, a.device_ordinal()))
    }

    fn eigvalsh_f64(&self, a: &GpuBufferHandle, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let w = crate::cusolver::gpu_eigvalsh_f64(a_buf, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(w, a.device_ordinal()))
    }

    // GPU 1-D FFT via cuFFT (#579). All paths are GPU-resident — see
    // `crate::cufft` for layout / normalization details.
    fn fft_c2c_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fft_c2c_f32(a_buf, batch, n, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn fft_c2c_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out = crate::cufft::gpu_fft_c2c_f64(a_buf, batch, n, inverse, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    fn pad_truncate_complex_f32(
        &self,
        src: &GpuBufferHandle,
        batch: usize,
        src_n: usize,
        dst_n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let src_buf = Self::unwrap_buffer(src)?;
        let dev = self.device(src.device_ordinal())?;
        let out = crate::kernels::gpu_pad_truncate_complex_f32(src_buf, batch, src_n, dst_n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, src.device_ordinal()))
    }

    fn pad_truncate_complex_f64(
        &self,
        src: &GpuBufferHandle,
        batch: usize,
        src_n: usize,
        dst_n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let src_buf = Self::unwrap_buffer_f64(src)?;
        let dev = self.device(src.device_ordinal())?;
        let out = crate::kernels::gpu_pad_truncate_complex_f64(src_buf, batch, src_n, dst_n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, src.device_ordinal()))
    }

    fn fft2_c2c_f32(
        &self,
        a: &GpuBufferHandle,
        h: usize,
        w: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_fft2_c2c_f32(a_buf, h, w, inverse, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn fft2_c2c_f64(
        &self,
        a: &GpuBufferHandle,
        h: usize,
        w: usize,
        inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_fft2_c2c_f64(a_buf, h, w, inverse, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    fn repeat_along_dim_f32(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        repeat_count: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer(input)?;
        let dev = self.device(input.device_ordinal())?;
        let out = crate::kernels::gpu_repeat_along_dim(in_buf, outer, repeat_count, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, input.device_ordinal()))
    }

    fn repeat_along_dim_f64(
        &self,
        input: &GpuBufferHandle,
        outer: usize,
        repeat_count: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let in_buf = Self::unwrap_buffer_f64(input)?;
        let dev = self.device(input.device_ordinal())?;
        let out = crate::kernels::gpu_repeat_along_dim_f64(in_buf, outer, repeat_count, inner, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, input.device_ordinal()))
    }

    fn rfft_r2c_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_rfft_r2c_f32(a_buf, batch, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn rfft_r2c_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_rfft_r2c_f64(a_buf, batch, n, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }

    fn irfft_c2r_f32(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_irfft_c2r_f32(a_buf, batch, n_out, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(out, a.device_ordinal()))
    }

    fn irfft_c2r_f64(
        &self,
        a: &GpuBufferHandle,
        batch: usize,
        n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer_f64(a)?;
        let dev = self.device(a.device_ordinal())?;
        let out =
            crate::cufft::gpu_irfft_c2r_f64(a_buf, batch, n_out, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(out, a.device_ordinal()))
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

/// Get the `GpuDevice` from the registered CUDA backend.
///
/// This retrieves the device that was created during [`init_cuda_backend`],
/// ensuring all kernel modules and cuBLAS handles are shared. Creating a
/// second `GpuDevice` via `GpuDevice::new(0)` would create a separate
/// CUDA context with its own module cache, which is not interoperable.
pub fn get_cuda_device() -> FerrotorchResult<Arc<GpuDevice>> {
    let backend =
        ferrotorch_core::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    // The global backend is a &dyn GpuBackend. We know it's CudaBackendImpl
    // because init_cuda_backend registered it. Downcast via Any.
    let cuda_backend = backend.as_any().downcast_ref::<CudaBackendImpl>().ok_or(
        FerrotorchError::InvalidArgument {
            message: "registered GPU backend is not CudaBackendImpl".into(),
        },
    )?;
    Ok(Arc::clone(cuda_backend.default_device()?))
}

/// Initialize the CUDA backend and register it with ferrotorch-core.
///
/// This must be called before any GPU tensor operations. It creates a
/// [`CudaBackendImpl`] (initializing CUDA device 0) and registers it via
/// [`ferrotorch_core::gpu_dispatch::register_gpu_backend`].
///
/// Calling this a second time returns an error (the backend is already
/// registered).
///
/// # Errors
///
/// - [`FerrotorchError::InvalidArgument`] if CUDA initialization fails.
/// - [`FerrotorchError::InvalidArgument`] if a GPU backend is already registered.
pub fn init_cuda_backend() -> FerrotorchResult<()> {
    // Idempotent: if already registered, return Ok silently.
    if ferrotorch_core::gpu_dispatch::has_gpu_backend() {
        return Ok(());
    }
    let backend = CudaBackendImpl::new()?;
    // OnceLock::set can still race if two threads call init concurrently —
    // if that happens, the second set() fails but the backend is registered
    // by the first. We treat that as success.
    let _ = ferrotorch_core::gpu_dispatch::register_gpu_backend(Box::new(backend));
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use ferrotorch_core::gpu_dispatch;

    // Note: Because `register_gpu_backend` uses a `OnceLock`, only the first
    // test to call `init_cuda_backend()` will succeed at registration. The
    // others will see the backend as already registered. We handle this by
    // checking `has_gpu_backend()` before calling init.

    /// Ensure the backend can be initialized (or was already initialized).
    fn ensure_init() {
        if !gpu_dispatch::has_gpu_backend() {
            init_cuda_backend().expect("init_cuda_backend");
        }
    }

    #[test]
    fn test_init_cuda_backend() {
        // First call succeeds (or backend was already registered by another test).
        ensure_init();
        assert!(gpu_dispatch::has_gpu_backend());
    }

    #[test]
    fn test_gpu_backend_returns_some() {
        ensure_init();
        assert!(gpu_dispatch::gpu_backend().is_some());
    }

    #[test]
    fn test_roundtrip_cpu_gpu_cpu() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");

        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        // SAFETY:
        // - `host` is a `Vec<f32>` whose backing allocation is at least
        //   4-byte-aligned (Vec follows the alignment of T = f32, per
        //   `Vec::as_ptr` documentation). Reinterpreting the pointer as
        //   `*const u8` is always sound: u8 has alignment 1, so any pointer
        //   alignment is sufficient for u8 access.
        // - The byte length `host.len() * size_of::<f32>()` exactly matches
        //   the byte extent of the f32 allocation (5 elements × 4 bytes =
        //   20 bytes); the resulting slice never overruns the allocation.
        // - Every f32 bit pattern is a valid u8 sequence (u8 has no invalid
        //   bit patterns), so the byte read is well-defined for all elements
        //   including IEEE 754 NaNs.
        // - Lifetime: the resulting `&[u8]` is bound by `host` (used on the
        //   very next line as `cpu_to_gpu(bytes, ...)` and not escaping the
        //   test scope); `host` is not dropped before line 3176.
        // - No `&mut` aliases: `host` is read via `host.as_ptr()` (shared);
        //   no `&mut [f32]` to the same allocation exists.
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                host.as_ptr() as *const u8,
                host.len() * std::mem::size_of::<f32>(),
            )
        };

        let handle = backend.cpu_to_gpu(bytes, 4, 0).expect("cpu_to_gpu");
        assert_eq!(handle.len(), 5);
        assert_eq!(handle.device_ordinal(), 0);

        let back_bytes = backend.gpu_to_cpu(&handle).expect("gpu_to_cpu");
        // SAFETY:
        // - `back_bytes` is a `Vec<u8>` returned by `gpu_to_cpu` (line 245)
        //   which constructs it via `Vec::from_raw_parts` from a
        //   `Vec<f32>::ManuallyDrop` (see line 256-262 in this file). The
        //   resulting `Vec<u8>` therefore inherits f32's 4-byte alignment
        //   on the original allocation pointer.
        // - `back_bytes.len() / 4` is the original f32 element count (the
        //   byte length is `host.len() * 4 == 20`, so divided by 4 yields
        //   `5`, the original `host.len()`).
        // - Every u32 bit pattern is a valid f32 (including all NaN
        //   payloads) — the round-trip CUDA memcpy preserves bytes exactly,
        //   so reinterpreting the bytes as f32 yields the original values.
        // - Lifetime: `&[f32]` is bound to `&back_bytes`; both are dropped
        //   at end of scope after the `assert_eq!` on line 3184.
        // - No `&mut` aliases: `back_bytes` is read via `as_ptr()` only.
        let back: &[f32] = unsafe {
            std::slice::from_raw_parts(back_bytes.as_ptr() as *const f32, back_bytes.len() / 4)
        };
        assert_eq!(back, &host[..]);
    }

    #[test]
    fn test_add_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");

        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let expected: Vec<f32> = vec![11.0, 22.0, 33.0, 44.0];

        // SAFETY:
        // - `a_data` is a `Vec<f32>` (4 elements, 4-byte-aligned allocation).
        //   Reinterpret as `*const u8` always sound (u8 alignment is 1; f32
        //   alignment 4 ≥ 1).
        // - Length `a_data.len() * 4 == 16` matches the f32 byte extent
        //   exactly; no overrun.
        // - Every f32 bit pattern is a valid u8 byte sequence; bytes are
        //   well-defined for all stored values (1.0..4.0 here).
        // - Lifetime: `&[u8]` borrows `a_data`'s allocation; `a_data` lives
        //   to the end of this test function (line 3216), and the slice is
        //   consumed by `cpu_to_gpu` on line 3201 before reuse.
        // - No `&mut` aliases: `a_data` is accessed via shared `as_ptr()`.
        let a_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4) };
        // SAFETY:
        // - `b_data` is a `Vec<f32>` (4 elements, 4-byte-aligned). All
        //   alignment, length, validity, lifetime, and aliasing properties
        //   are identical to `a_bytes` above. The byte length
        //   `b_data.len() * 4 == 16` matches the f32 byte extent exactly.
        // - The `&[u8]` is consumed by `cpu_to_gpu` on line 3202 before
        //   `b_data` is read again.
        let b_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4) };

        let a_handle = backend.cpu_to_gpu(a_bytes, 4, 0).expect("cpu_to_gpu a");
        let b_handle = backend.cpu_to_gpu(b_bytes, 4, 0).expect("cpu_to_gpu b");

        let result = backend.add_f32(&a_handle, &b_handle).expect("add_f32");
        assert_eq!(result.len(), 4);

        let result_bytes = backend.gpu_to_cpu(&result).expect("gpu_to_cpu");
        // SAFETY:
        // - `result_bytes` is a `Vec<u8>` constructed by `gpu_to_cpu` (line
        //   245) via `Vec::from_raw_parts` from a `Vec<f32>::ManuallyDrop`
        //   so its allocation pointer inherits 4-byte alignment from the
        //   original f32 allocation.
        // - `result_bytes.len() / 4` is the original f32 element count
        //   (`add_f32` produces 4 f32s → 16 bytes / 4 = 4 elements), so the
        //   reinterpreted slice covers exactly the f32 region.
        // - Every u32 bit pattern is a valid f32 (the bytes that
        //   `gpu_to_cpu` returned are exactly those produced by the GPU
        //   `gpu_add` kernel, so each 4-byte word is a valid IEEE 754 f32).
        // - Lifetime: `&[f32]` is bound by `&result_bytes`; the iterator on
        //   line 3212 consumes the slice within the same scope.
        // - No `&mut` aliases: `result_bytes` is accessed only via shared
        //   `as_ptr()`.
        let result_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(result_bytes.as_ptr() as *const f32, result_bytes.len() / 4)
        };

        for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "element {i}: got {got}, expected {exp}",
            );
        }
    }

    #[test]
    fn test_matmul_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend registered");

        // A = [[1, 2, 3],
        //      [4, 5, 6]]  (2x3)
        // B = [[7, 8],
        //      [9, 10],
        //      [11, 12]]   (3x2)
        // C = [[58, 64],
        //      [139, 154]] (2x2)
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0];
        let expected: Vec<f32> = vec![58.0, 64.0, 139.0, 154.0];

        // SAFETY:
        // - `a_data` is a `Vec<f32>` of 6 elements (24 bytes total). f32
        //   alignment 4 ≥ u8 alignment 1, so the pointer cast is sound.
        // - `a_data.len() * 4 == 24` exactly matches the f32 byte extent;
        //   no overrun, no misalignment.
        // - All values (1.0..=6.0) are finite; every f32 byte sequence is a
        //   valid u8 sequence regardless.
        // - Lifetime: `&[u8]` is bound by `a_data` through `cpu_to_gpu` on
        //   line 3242 (within the same test scope). `a_data` is not dropped
        //   before line 3252.
        // - No `&mut` aliases: shared `as_ptr()` access only.
        let a_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4) };
        // SAFETY:
        // - `b_data` is a `Vec<f32>` of 6 elements; same alignment,
        //   length-exactness, validity, lifetime, and aliasing properties
        //   as `a_bytes` above. Byte length `b_data.len() * 4 == 24`
        //   matches the f32 byte extent exactly.
        // - Slice is consumed by `cpu_to_gpu` on line 3243.
        let b_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4) };

        let a_handle = backend.cpu_to_gpu(a_bytes, 4, 0).expect("cpu_to_gpu a");
        let b_handle = backend.cpu_to_gpu(b_bytes, 4, 0).expect("cpu_to_gpu b");

        let result = backend
            .matmul_f32(&a_handle, &b_handle, 2, 3, 2)
            .expect("matmul_f32");
        assert_eq!(result.len(), 4);

        let result_bytes = backend.gpu_to_cpu(&result).expect("gpu_to_cpu");
        // SAFETY:
        // - `result_bytes` is a `Vec<u8>` returned by `gpu_to_cpu` (line
        //   245), constructed via `Vec::from_raw_parts` from a
        //   `ManuallyDrop<Vec<f32>>` (lines 256-262), so its underlying
        //   allocation has f32's 4-byte alignment.
        // - `result_bytes.len() / 4` is the original f32 element count
        //   (matmul produces a 2×2 matrix → 4 f32 → 16 bytes; 16/4 = 4).
        // - Every u32 bit pattern is a valid f32; the bytes were produced
        //   by the cuBLAS GEMM kernel (so each 4-byte word is a finite
        //   IEEE 754 f32 modulo NaN, all of which are valid f32 values).
        // - Lifetime: `&[f32]` is bound by `&result_bytes`; the iterator on
        //   line 3284 consumes the slice in the same scope and `result_bytes`
        //   outlives it.
        // - No `&mut` aliases: shared `as_ptr()` only.
        let result_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(result_bytes.as_ptr() as *const f32, result_bytes.len() / 4)
        };

        for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "element {i}: got {got}, expected {exp}",
            );
        }
    }
}
