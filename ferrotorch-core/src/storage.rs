use crate::device::Device;
use crate::dtype::Element;
use crate::gpu_dispatch::GpuBufferHandle;

/// The underlying data buffer for a tensor, tagged with its device.
///
/// Owns the data directly (`Vec<T>` for CPU, `GpuBufferHandle` for GPU).
/// The GPU handle is type-erased -- ferrotorch-gpu provides the concrete
/// implementation via the `GpuBackend` trait.
#[derive(Debug)]
pub struct TensorStorage<T: Element> {
    pub(crate) data: StorageBuffer<T>,
    pub(crate) device: Device,
}

/// Device-specific data buffer.
pub enum StorageBuffer<T: Element> {
    /// CPU heap-allocated data.
    Cpu(Vec<T>),
    /// GPU device memory, accessed via the registered `GpuBackend`.
    Gpu(GpuBufferHandle),
}

impl<T: Element> TensorStorage<T> {
    /// Create a new CPU storage from a `Vec<T>`.
    pub fn cpu(data: Vec<T>) -> Self {
        Self {
            data: StorageBuffer::Cpu(data),
            device: Device::Cpu,
        }
    }

    /// Create storage on `target_device` from CPU data.
    ///
    /// If `target_device` is CPU, wraps the `Vec` directly (zero-copy).
    /// If `target_device` is GPU, uploads the data and returns GPU storage.
    ///
    /// Use this instead of `TensorStorage::cpu(data).to(device)` to avoid
    /// injecting a `ToDeviceBackward` node into the autograd graph.
    pub fn on_device(data: Vec<T>, target_device: Device) -> crate::error::FerrotorchResult<Self> {
        match target_device {
            Device::Cpu => Ok(Self::cpu(data)),
            Device::Cuda(ordinal) => {
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
                let bytes: &[u8] = unsafe {
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<T>(),
                    )
                };
                let handle = backend.cpu_to_gpu(bytes, std::mem::size_of::<T>(), ordinal)?;
                Ok(Self::gpu(handle))
            }
        }
    }

    /// Create a new GPU storage from a handle.
    pub fn gpu(handle: GpuBufferHandle) -> Self {
        let device = Device::Cuda(handle.device_ordinal());
        Self {
            data: StorageBuffer::Gpu(handle),
            device,
        }
    }

    /// The device this storage resides on.
    #[inline]
    pub fn device(&self) -> Device {
        self.device
    }

    /// Total number of elements in the buffer.
    pub fn len(&self) -> usize {
        match &self.data {
            StorageBuffer::Cpu(v) => v.len(),
            StorageBuffer::Gpu(h) => h.len(),
        }
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Borrow the data as a slice. Only available for CPU storage.
    ///
    /// # Panics
    /// Panics if the tensor is on a GPU device. Call `.cpu()` first.
    pub fn as_slice(&self) -> &[T] {
        match &self.data {
            StorageBuffer::Cpu(v) => v.as_slice(),
            StorageBuffer::Gpu(_) => {
                panic!("cannot access GPU tensor as CPU slice -- call .cpu() first")
            }
        }
    }

    /// Borrow the data as a mutable slice. Only available for CPU storage.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.data {
            StorageBuffer::Cpu(v) => v.as_mut_slice(),
            StorageBuffer::Gpu(_) => {
                panic!("cannot mutate GPU tensor as CPU slice -- call .cpu() first")
            }
        }
    }

    /// Returns `true` if this storage is on CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(&self.data, StorageBuffer::Cpu(_))
    }

    /// Returns `true` if this storage is on a GPU.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(&self.data, StorageBuffer::Gpu(_))
    }

    /// Get the GPU buffer handle. Returns `None` for CPU storage.
    pub fn gpu_handle(&self) -> Option<&GpuBufferHandle> {
        match &self.data {
            StorageBuffer::Gpu(h) => Some(h),
            StorageBuffer::Cpu(_) => None,
        }
    }

    /// Fallible clone — same as `Clone::clone` but returns `Result` instead
    /// of panicking when the GPU backend is missing or a CUDA call fails.
    pub fn try_clone(&self) -> crate::error::FerrotorchResult<Self> {
        match &self.data {
            StorageBuffer::Cpu(v) => Ok(Self {
                data: StorageBuffer::Cpu(v.clone()),
                device: self.device,
            }),
            StorageBuffer::Gpu(h) => {
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
                let cloned = backend.clone_buffer(h)?;
                Ok(Self {
                    data: StorageBuffer::Gpu(cloned),
                    device: self.device,
                })
            }
        }
    }

    /// Clone a contiguous sub-region `[offset..offset+numel]` of this storage.
    ///
    /// For CPU, slices the `Vec` directly. For GPU, round-trips through the
    /// host to extract the sub-region (correct, not yet optimized with D2D
    /// memcpy). Returns an error instead of panicking on GPU failures.
    pub fn try_clone_subregion(
        &self,
        offset: usize,
        numel: usize,
    ) -> crate::error::FerrotorchResult<Self> {
        if offset == 0 && numel == self.len() {
            return self.try_clone();
        }
        match &self.data {
            StorageBuffer::Cpu(v) => {
                let slice = &v[offset..offset + numel];
                Ok(Self {
                    data: StorageBuffer::Cpu(slice.to_vec()),
                    device: self.device,
                })
            }
            StorageBuffer::Gpu(h) => {
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
                let bytes = backend.gpu_to_cpu(h)?;
                let elem_size = std::mem::size_of::<T>();
                let start = offset * elem_size;
                let end = (offset + numel) * elem_size;
                let handle =
                    backend.cpu_to_gpu(&bytes[start..end], elem_size, h.device_ordinal())?;
                Ok(Self {
                    data: StorageBuffer::Gpu(handle),
                    device: self.device,
                })
            }
        }
    }
}

impl<T: Element> Clone for TensorStorage<T> {
    fn clone(&self) -> Self {
        match &self.data {
            StorageBuffer::Cpu(v) => Self {
                data: StorageBuffer::Cpu(v.clone()),
                device: self.device,
            },
            StorageBuffer::Gpu(h) => {
                // Clone GPU buffer via the registered backend
                if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                    match backend.clone_buffer(h) {
                        Ok(cloned) => Self {
                            data: StorageBuffer::Gpu(cloned),
                            device: self.device,
                        },
                        Err(_) => panic!("failed to clone GPU buffer"),
                    }
                } else {
                    panic!("no GPU backend registered -- cannot clone GPU tensor")
                }
            }
        }
    }
}

impl<T: Element> Drop for TensorStorage<T> {
    fn drop(&mut self) {
        // Return CPU buffers to the pool for reuse.
        if let StorageBuffer::Cpu(ref mut v) = self.data {
            if !v.is_empty() {
                // Take the Vec out, replacing with an empty one (no alloc).
                let buf = std::mem::take(v);
                crate::cpu_pool::pool_return_cpu(buf);
            }
        }
        // GPU buffers are dropped normally (returned to GPU pool by CudaBuffer's Drop).
    }
}

impl<T: Element> std::fmt::Debug for StorageBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageBuffer::Cpu(v) => write!(f, "Cpu({} elements)", v.len()),
            StorageBuffer::Gpu(h) => write!(f, "Gpu({h:?})"),
        }
    }
}
