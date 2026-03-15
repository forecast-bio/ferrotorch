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
use ferrotorch_core::gpu_dispatch::{GpuBackend, GpuBufferHandle};

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
        let device =
            Arc::new(GpuDevice::new(0).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("CUDA init failed: {e}"),
            })?);
        Ok(Self {
            devices: vec![device],
        })
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
    fn cpu_to_gpu(
        &self,
        data: &[u8],
        elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device)?;
        // Reinterpret raw bytes as an f32 slice.
        // SAFETY: The caller (ferrotorch-core) guarantees that `data` was
        // originally an f32 slice serialised to bytes, and `elem_size == 4`.
        let f32_count = data.len() / elem_size;
        let f32_data: &[f32] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const f32, f32_count)
        };
        let buf = crate::transfer::cpu_to_gpu(f32_data, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(buf, device))
    }

    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>> {
        let buf = Self::unwrap_buffer(handle)?;
        let dev = self.device(handle.device_ordinal())?;
        let f32_data =
            crate::transfer::gpu_to_cpu(buf, dev).map_err(Self::map_gpu_err)?;

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
    }

    fn clone_buffer(
        &self,
        handle: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // Clone via GPU -> CPU -> GPU round-trip.
        // Correct but not optimal; a device-to-device memcpy would be better.
        let bytes = self.gpu_to_cpu(handle)?;
        self.cpu_to_gpu(&bytes, 4, handle.device_ordinal())
    }

    fn alloc_zeros(
        &self,
        len: usize,
        _elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let dev = self.device(device)?;
        let buf =
            crate::transfer::alloc_zeros::<f32>(len, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(buf, device))
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
        let result =
            crate::kernels::gpu_add(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
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
        let result =
            crate::kernels::gpu_sub(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
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
        let result =
            crate::kernels::gpu_mul(a_buf, b_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn neg_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_neg(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    fn relu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let result =
            crate::kernels::gpu_relu(a_buf, dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
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
        let result = crate::blas::gpu_matmul_f32(a_buf, b_buf, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result, a.device_ordinal()))
    }

    // -- Reduction f32 --------------------------------------------------------

    fn sum_f32(
        &self,
        a: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // No dedicated GPU sum kernel yet. Fall back to GPU -> CPU -> sum -> GPU.
        let a_buf = Self::unwrap_buffer(a)?;
        let dev = self.device(a.device_ordinal())?;
        let host_data =
            crate::transfer::gpu_to_cpu(a_buf, dev).map_err(Self::map_gpu_err)?;

        let total: f32 = host_data.iter().sum();

        let result_buf =
            crate::transfer::cpu_to_gpu(&[total], dev).map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer(result_buf, a.device_ordinal()))
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
        let result = crate::blas::gpu_matmul_f64(a_buf, b_buf, m, k, n, dev)
            .map_err(Self::map_gpu_err)?;
        Ok(Self::wrap_buffer_f64(result, a.device_ordinal()))
    }
}

// ---------------------------------------------------------------------------
// Registration
// ---------------------------------------------------------------------------

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
        let back: &[f32] = unsafe {
            std::slice::from_raw_parts(
                back_bytes.as_ptr() as *const f32,
                back_bytes.len() / 4,
            )
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

        let a_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                a_data.as_ptr() as *const u8,
                a_data.len() * 4,
            )
        };
        let b_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                b_data.as_ptr() as *const u8,
                b_data.len() * 4,
            )
        };

        let a_handle = backend.cpu_to_gpu(a_bytes, 4, 0).expect("cpu_to_gpu a");
        let b_handle = backend.cpu_to_gpu(b_bytes, 4, 0).expect("cpu_to_gpu b");

        let result = backend.add_f32(&a_handle, &b_handle).expect("add_f32");
        assert_eq!(result.len(), 4);

        let result_bytes = backend.gpu_to_cpu(&result).expect("gpu_to_cpu");
        let result_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(
                result_bytes.as_ptr() as *const f32,
                result_bytes.len() / 4,
            )
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

        let a_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(a_data.as_ptr() as *const u8, a_data.len() * 4)
        };
        let b_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(b_data.as_ptr() as *const u8, b_data.len() * 4)
        };

        let a_handle = backend.cpu_to_gpu(a_bytes, 4, 0).expect("cpu_to_gpu a");
        let b_handle = backend.cpu_to_gpu(b_bytes, 4, 0).expect("cpu_to_gpu b");

        let result = backend
            .matmul_f32(&a_handle, &b_handle, 2, 3, 2)
            .expect("matmul_f32");
        assert_eq!(result.len(), 4);

        let result_bytes = backend.gpu_to_cpu(&result).expect("gpu_to_cpu");
        let result_f32: &[f32] = unsafe {
            std::slice::from_raw_parts(
                result_bytes.as_ptr() as *const f32,
                result_bytes.len() / 4,
            )
        };

        for (i, (&got, &exp)) in result_f32.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "element {i}: got {got}, expected {exp}",
            );
        }
    }
}
