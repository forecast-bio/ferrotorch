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
    /// Meta storage — no backing memory, only the element count is
    /// recorded. Tensors built on this variant carry shape and dtype
    /// info but cannot be read or written. Used for shape inference
    /// and dry-run model construction. CL-395.
    Meta {
        numel: usize,
        _phantom: std::marker::PhantomData<T>,
    },
}

impl<T: Element> TensorStorage<T> {
    /// Create a new CPU storage from a `Vec<T>`.
    pub fn cpu(data: Vec<T>) -> Self {
        Self {
            data: StorageBuffer::Cpu(data),
            device: Device::Cpu,
        }
    }

    /// Create a meta storage with the given element count. No memory is
    /// allocated for the elements; only the size is recorded. Reading
    /// the data of a meta tensor returns an error.
    pub fn meta(numel: usize) -> Self {
        Self {
            data: StorageBuffer::Meta {
                numel,
                _phantom: std::marker::PhantomData,
            },
            device: Device::Meta,
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
            Device::Meta => {
                // Discard the data; only the element count matters.
                Ok(Self::meta(data.len()))
            }
        }
    }

    /// Create storage on `target_device` from CPU data, using pinned host
    /// memory for the CPU→GPU transfer (~2x faster for large tensors).
    ///
    /// Falls back to regular transfer if no GPU backend or if target is CPU.
    pub fn on_device_pinned(
        data: Vec<T>,
        target_device: Device,
    ) -> crate::error::FerrotorchResult<Self> {
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
                let handle =
                    backend.cpu_to_gpu_pinned(bytes, std::mem::size_of::<T>(), ordinal)?;
                Ok(Self::gpu(handle))
            }
            Device::Meta => Ok(Self::meta(data.len())),
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
            StorageBuffer::Meta { numel, .. } => *numel,
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
    /// Panics if the tensor is a meta tensor.
    pub fn as_slice(&self) -> &[T] {
        match &self.data {
            StorageBuffer::Cpu(v) => v.as_slice(),
            StorageBuffer::Gpu(_) => {
                panic!("cannot access GPU tensor as CPU slice -- call .cpu() first")
            }
            StorageBuffer::Meta { .. } => {
                panic!("cannot access meta tensor as a slice -- meta tensors carry no data")
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
            StorageBuffer::Meta { .. } => {
                panic!("cannot mutate meta tensor as a slice -- meta tensors carry no data")
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

    /// Returns `true` if this storage is a meta (no-data) tensor.
    #[inline]
    pub fn is_meta(&self) -> bool {
        matches!(&self.data, StorageBuffer::Meta { .. })
    }

    /// Get the GPU buffer handle. Returns `None` for CPU and Meta storage.
    pub fn gpu_handle(&self) -> Option<&GpuBufferHandle> {
        match &self.data {
            StorageBuffer::Gpu(h) => Some(h),
            StorageBuffer::Cpu(_) | StorageBuffer::Meta { .. } => None,
        }
    }

    /// Get a mutable GPU buffer handle. Returns `None` for CPU and Meta storage.
    ///
    /// # Safety note
    ///
    /// Callers must ensure exclusive access to the storage (e.g. via the
    /// same unsafe contract as `update_data`).
    pub fn gpu_handle_mut(&mut self) -> Option<&mut GpuBufferHandle> {
        match &mut self.data {
            StorageBuffer::Gpu(h) => Some(h),
            StorageBuffer::Cpu(_) | StorageBuffer::Meta { .. } => None,
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
            StorageBuffer::Meta { numel, .. } => Ok(Self::meta(*numel)),
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
            StorageBuffer::Meta { .. } => Ok(Self::meta(numel)),
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
            StorageBuffer::Meta { numel, .. } => Self::meta(*numel),
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
            StorageBuffer::Meta { numel, .. } => write!(f, "Meta({numel} elements)"),
        }
    }
}
