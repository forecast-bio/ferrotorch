//! GPU backend dispatch layer.
//!
//! ferrotorch-core defines the [`GpuBackend`] trait and [`GpuBufferHandle`].
//! ferrotorch-gpu (or any other GPU crate) implements and registers a backend.
//! This avoids circular dependencies: core doesn't depend on gpu.

use std::any::Any;
use std::sync::OnceLock;

use crate::error::{FerrotorchError, FerrotorchResult};

/// Opaque handle to GPU memory.
///
/// ferrotorch-core doesn't know what's inside -- the GPU backend provides
/// the concrete type (e.g., `CudaBuffer<f32>`). We store it as
/// `Box<dyn Any + Send + Sync>` for type erasure.
pub struct GpuBufferHandle {
    pub(crate) inner: Box<dyn Any + Send + Sync>,
    pub(crate) device_ordinal: usize,
    pub(crate) len: usize,
}

impl GpuBufferHandle {
    pub fn new(inner: Box<dyn Any + Send + Sync>, device_ordinal: usize, len: usize) -> Self {
        Self { inner, device_ordinal, len }
    }

    #[inline]
    pub fn device_ordinal(&self) -> usize { self.device_ordinal }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }
}

impl std::fmt::Debug for GpuBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBufferHandle")
            .field("device", &self.device_ordinal)
            .field("len", &self.len)
            .finish()
    }
}

/// Trait that GPU backends implement to handle tensor operations.
///
/// ferrotorch-core calls these methods; ferrotorch-gpu provides the implementation.
pub trait GpuBackend: Send + Sync {
    fn cpu_to_gpu(&self, data: &[u8], elem_size: usize, device: usize) -> FerrotorchResult<GpuBufferHandle>;
    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>>;
    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn alloc_zeros(&self, len: usize, elem_size: usize, device: usize) -> FerrotorchResult<GpuBufferHandle>;

    // Elementwise f32
    fn add_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn sub_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn mul_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn neg_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn relu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;

    // Linalg f32
    fn matmul_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, m: usize, k: usize, n: usize) -> FerrotorchResult<GpuBufferHandle>;

    // Reduction f32
    fn sum_f32(&self, a: &GpuBufferHandle, len: usize) -> FerrotorchResult<GpuBufferHandle>;

    // Elementwise f64 (default impls return "not yet implemented" errors)
    fn add_f64(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }
    fn sub_f64(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }
    fn mul_f64(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }
    fn neg_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }
    fn relu_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }

    // Linalg f64
    fn matmul_f64(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle, _m: usize, _k: usize, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }

    // Reduction f64
    fn sum_f64(&self, _a: &GpuBufferHandle, _numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }
}

static GPU_BACKEND: OnceLock<Box<dyn GpuBackend>> = OnceLock::new();

/// Register a GPU backend. Called once by the GPU crate on init.
pub fn register_gpu_backend(backend: Box<dyn GpuBackend>) -> Result<(), Box<dyn GpuBackend>> {
    GPU_BACKEND.set(backend)
}

/// Get the registered GPU backend, if any.
pub fn gpu_backend() -> Option<&'static dyn GpuBackend> {
    GPU_BACKEND.get().map(|b| b.as_ref())
}

/// Returns `true` if a GPU backend has been registered.
pub fn has_gpu_backend() -> bool {
    GPU_BACKEND.get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer_handle() {
        let handle = GpuBufferHandle::new(Box::new(42u64), 0, 100);
        assert_eq!(handle.device_ordinal(), 0);
        assert_eq!(handle.len(), 100);
        assert!(!handle.is_empty());
        assert_eq!(handle.downcast_ref::<u64>(), Some(&42));
    }

    #[test]
    fn test_gpu_buffer_handle_debug() {
        let handle = GpuBufferHandle::new(Box::new(()), 1, 50);
        let s = format!("{handle:?}");
        assert!(s.contains("device: 1"));
    }
}
