use crate::device::Device;
use crate::dtype::Element;
use crate::gpu_dispatch::GpuBufferHandle;

// ---------------------------------------------------------------------------
// CubeStorageHandle — trait-erased CubeCL device handle
// ---------------------------------------------------------------------------

/// Trait-erased handle to a CubeCL device-resident buffer.
///
/// `ferrotorch-cubecl` provides the concrete implementation; `ferrotorch-core`
/// defines only this interface so there is no circular dependency. The concrete
/// type wraps a `cubecl::server::Handle` plus an `Arc<CubeRuntime>` so the
/// runtime remains alive as long as any handle exists.
///
/// This mirrors the `GpuBufferHandle` / `GpuBackend` pattern used for CUDA.
/// Issue #673.
pub trait CubeStorageHandle: std::fmt::Debug + Send + Sync {
    /// Upcast to `&dyn Any` for concrete-type downcasting.
    ///
    /// Implementors must return `self` via `self as &dyn std::any::Any`.
    /// This mirrors the `GpuBackend::as_any` pattern.
    fn as_any(&self) -> &dyn std::any::Any;

    /// Number of `f32` elements in the buffer.
    fn len(&self) -> usize;

    /// Whether the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Device ordinal this buffer resides on.
    fn ordinal(&self) -> usize;

    /// Read the buffer contents back to the host as `Vec<f32>`.
    ///
    /// Performs a device-to-host transfer (D2H). Call sparingly — this is the
    /// explicit readback that the PyTorch-parity design requires users to opt
    /// into via `.cpu()`.
    fn read_to_host(&self) -> crate::error::FerrotorchResult<Vec<f32>>;

    /// Clone the handle (allocate a new device buffer with the same contents).
    fn clone_handle(&self) -> Box<dyn CubeStorageHandle>;
}

// ---------------------------------------------------------------------------
// TensorStorage / StorageBuffer
// ---------------------------------------------------------------------------

/// The underlying data buffer for a tensor, tagged with its device.
///
/// Owns the data directly (`Vec<T>` for CPU, `GpuBufferHandle` for CUDA,
/// `Box<dyn CubeStorageHandle>` for CubeCL/XPU). GPU handles are type-erased;
/// the backend crates provide concrete implementations.
#[derive(Debug)]
pub struct TensorStorage<T: Element> {
    pub(crate) data: StorageBuffer<T>,
    pub(crate) device: Device,
}

/// Device-specific data buffer.
pub enum StorageBuffer<T: Element> {
    /// CPU heap-allocated data.
    Cpu(Vec<T>),
    /// CUDA device memory, accessed via the registered `GpuBackend`.
    Gpu(GpuBufferHandle),
    /// CubeCL device-resident buffer (XPU / portable GPU via wgpu/CUDA/ROCm).
    ///
    /// The concrete handle type is provided by `ferrotorch-cubecl`; core sees
    /// only the `CubeStorageHandle` trait object. Issue #673.
    Cubecl(Box<dyn CubeStorageHandle>),
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
    /// If `target_device` is CUDA, uploads the data and returns GPU storage.
    ///
    /// Use this instead of `TensorStorage::cpu(data).to(device)` to avoid
    /// injecting a `ToDeviceBackward` node into the autograd graph.
    ///
    /// Note: `Device::Xpu` is not supported here because an H2D upload for XPU
    /// requires a `CubeRuntime`, which core does not own. Use
    /// `Tensor::to(Device::Xpu(n))` instead, which routes through
    /// `ferrotorch-xpu`'s `XpuDevice`.
    pub fn on_device(data: Vec<T>, target_device: Device) -> crate::error::FerrotorchResult<Self> {
        match target_device {
            Device::Cpu => Ok(Self::cpu(data)),
            Device::Cuda(ordinal) => {
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(crate::error::FerrotorchError::DeviceUnavailable)?;
                let bytes: &[u8] = unsafe {
                    // SAFETY: `data` is a valid, aligned `Vec<T>` on the heap.
                    // Reinterpreting as `&[u8]` is safe because we only use
                    // the bytes to copy to the GPU; the vec is not dropped
                    // until after `cpu_to_gpu` returns.
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<T>(),
                    )
                };
                let handle = backend.cpu_to_gpu(bytes, std::mem::size_of::<T>(), ordinal)?;
                Ok(Self::gpu(handle))
            }
            Device::Xpu(_) => Err(crate::error::FerrotorchError::InvalidArgument {
                message: "XPU storage requires a CubeRuntime; use Tensor::to(Device::Xpu(n)) \
                          via ferrotorch-xpu instead of TensorStorage::on_device. Issue #673."
                    .into(),
            }),
            Device::Mps(_) => Err(crate::error::FerrotorchError::InvalidArgument {
                message: "MPS storage requires the ferrotorch-mps backend; not yet wired into TensorStorage".into(),
            }),
            Device::Meta => {
                // Discard the data; only the element count matters.
                Ok(Self::meta(data.len()))
            }
        }
    }

    /// Create storage on `target_device` from CPU data, using pinned host
    /// memory for the CPU→CUDA transfer (~2x faster for large tensors).
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
                    // SAFETY: same invariant as in `on_device`.
                    std::slice::from_raw_parts(
                        data.as_ptr() as *const u8,
                        data.len() * std::mem::size_of::<T>(),
                    )
                };
                let handle =
                    backend.cpu_to_gpu_pinned(bytes, std::mem::size_of::<T>(), ordinal)?;
                Ok(Self::gpu(handle))
            }
            Device::Xpu(_) => Err(crate::error::FerrotorchError::InvalidArgument {
                message: "XPU storage requires a CubeRuntime; use Tensor::to(Device::Xpu(n)) \
                          via ferrotorch-xpu instead of TensorStorage::on_device_pinned. Issue #673."
                    .into(),
            }),
            Device::Mps(_) => Err(crate::error::FerrotorchError::InvalidArgument {
                message: "MPS storage requires the ferrotorch-mps backend; not yet wired into TensorStorage".into(),
            }),
            Device::Meta => Ok(Self::meta(data.len())),
        }
    }

    /// Create XPU (CubeCL device-resident) storage from a trait-erased handle.
    ///
    /// The handle wraps a `cubecl::server::Handle` and holds an `Arc<CubeRuntime>`
    /// so the device stays alive. This is the correct post-#673 constructor:
    /// XPU storage is truly device-resident, not a CPU `Vec<T>`.
    ///
    /// Called by `ferrotorch-xpu` (and `ferrotorch-cubecl`) after uploading data
    /// to the device.
    pub fn xpu_from_handle(handle: Box<dyn CubeStorageHandle>, ordinal: usize) -> Self {
        Self {
            data: StorageBuffer::Cubecl(handle),
            device: Device::Xpu(ordinal),
        }
    }

    /// Create a new CUDA storage from a handle.
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
            StorageBuffer::Cubecl(h) => h.len(),
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
    /// Panics if the tensor is on a GPU or XPU device. Call `.cpu()` first.
    /// Panics if the tensor is a meta tensor.
    #[deprecated(
        since = "0.4.5",
        note = "use try_as_slice() instead; this version panics on non-CPU storage"
    )]
    pub fn as_slice(&self) -> &[T] {
        match &self.data {
            StorageBuffer::Cpu(v) => v.as_slice(),
            StorageBuffer::Gpu(_) => {
                panic!("cannot access GPU tensor as CPU slice -- call .cpu() first")
            }
            StorageBuffer::Cubecl(_) => {
                panic!("cannot access XPU tensor as CPU slice -- call .cpu() first")
            }
            StorageBuffer::Meta { .. } => {
                panic!("cannot access meta tensor as a slice -- meta tensors carry no data")
            }
        }
    }

    /// Borrow the data as a mutable slice. Only available for CPU storage.
    ///
    /// # Panics
    /// Panics if the tensor is on a GPU or XPU device. Call `.cpu()` first.
    /// Panics if the tensor is a meta tensor.
    #[deprecated(
        since = "0.4.5",
        note = "use try_as_mut_slice() instead; this version panics on non-CPU storage"
    )]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        match &mut self.data {
            StorageBuffer::Cpu(v) => v.as_mut_slice(),
            StorageBuffer::Gpu(_) => {
                panic!("cannot mutate GPU tensor as CPU slice -- call .cpu() first")
            }
            StorageBuffer::Cubecl(_) => {
                panic!("cannot mutate XPU tensor as CPU slice -- call .cpu() first")
            }
            StorageBuffer::Meta { .. } => {
                panic!("cannot mutate meta tensor as a slice -- meta tensors carry no data")
            }
        }
    }

    /// Fallible borrow of the data as a slice — same as `as_slice` but returns
    /// `Err(GpuTensorNotAccessible)` instead of panicking when the tensor is
    /// not CPU-resident (GPU, XPU, or meta).
    ///
    /// # Errors
    /// Returns [`FerrotorchError::GpuTensorNotAccessible`] when the storage is
    /// on a GPU or XPU device, or is a meta tensor.
    pub fn try_as_slice(&self) -> crate::error::FerrotorchResult<&[T]> {
        match &self.data {
            StorageBuffer::Cpu(v) => Ok(v.as_slice()),
            StorageBuffer::Gpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => {
                Err(crate::error::FerrotorchError::GpuTensorNotAccessible)
            }
        }
    }

    /// Fallible mutable borrow of the data as a slice — same as `as_mut_slice`
    /// but returns `Err(GpuTensorNotAccessible)` instead of panicking when the
    /// tensor is not CPU-resident (GPU, XPU, or meta).
    ///
    /// # Errors
    /// Returns [`FerrotorchError::GpuTensorNotAccessible`] when the storage is
    /// on a GPU or XPU device, or is a meta tensor.
    pub fn try_as_mut_slice(&mut self) -> crate::error::FerrotorchResult<&mut [T]> {
        match &mut self.data {
            StorageBuffer::Cpu(v) => Ok(v.as_mut_slice()),
            StorageBuffer::Gpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => {
                Err(crate::error::FerrotorchError::GpuTensorNotAccessible)
            }
        }
    }

    /// Returns `true` if this storage is on CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        matches!(&self.data, StorageBuffer::Cpu(_))
    }

    /// Returns `true` if this storage is a CUDA device buffer.
    #[inline]
    pub fn is_gpu(&self) -> bool {
        matches!(&self.data, StorageBuffer::Gpu(_))
    }

    /// Returns `true` if this storage is a CubeCL device-resident buffer (XPU).
    #[inline]
    pub fn is_cubecl(&self) -> bool {
        matches!(&self.data, StorageBuffer::Cubecl(_))
    }

    /// Returns `true` if this storage is a meta (no-data) tensor.
    #[inline]
    pub fn is_meta(&self) -> bool {
        matches!(&self.data, StorageBuffer::Meta { .. })
    }

    /// Get the CUDA buffer handle. Returns `None` for CPU, XPU, and Meta storage.
    pub fn gpu_handle(&self) -> Option<&GpuBufferHandle> {
        match &self.data {
            StorageBuffer::Gpu(h) => Some(h),
            StorageBuffer::Cpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => None,
        }
    }

    /// Get a mutable CUDA buffer handle. Returns `None` for CPU, XPU, and Meta storage.
    ///
    /// # Safety note
    ///
    /// Callers must ensure exclusive access to the storage (e.g. via the
    /// same unsafe contract as `update_data`).
    pub fn gpu_handle_mut(&mut self) -> Option<&mut GpuBufferHandle> {
        match &mut self.data {
            StorageBuffer::Gpu(h) => Some(h),
            StorageBuffer::Cpu(_) | StorageBuffer::Cubecl(_) | StorageBuffer::Meta { .. } => None,
        }
    }

    /// Get the CubeCL storage handle. Returns `None` for non-Cubecl storage.
    pub fn cubecl_handle(&self) -> Option<&dyn CubeStorageHandle> {
        match &self.data {
            StorageBuffer::Cubecl(h) => Some(h.as_ref()),
            _ => None,
        }
    }

    /// Fallible clone — same as `Clone::clone` but returns `Result` instead
    /// of panicking when a backend call fails.
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
            StorageBuffer::Cubecl(h) => {
                let cloned = h.clone_handle();
                Ok(Self {
                    data: StorageBuffer::Cubecl(cloned),
                    device: self.device,
                })
            }
            StorageBuffer::Meta { numel, .. } => Ok(Self::meta(*numel)),
        }
    }

    /// Clone a contiguous sub-region `[offset..offset+numel]` of this storage.
    ///
    /// For CPU, slices the `Vec` directly. For CUDA/XPU, round-trips through the
    /// host to extract the sub-region. Returns an error instead of panicking
    /// on backend failures.
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
            StorageBuffer::Cubecl(h) => {
                // D2H readback, slice, then re-upload via a new handle.
                // The new handle reuses the same runtime (held by the original
                // handle's Arc<CubeRuntime>).
                let all = h.read_to_host()?;
                let slice = all[offset..offset + numel].to_vec();
                // Re-upload: the concrete impl's `clone_handle` clones the full
                // buffer; for sub-regions we go through host for now (correct,
                // can be optimised later with a device-side copy).
                // We need a new handle wrapping just `slice` — but
                // `CubeStorageHandle` doesn't expose an upload method (that
                // lives in ferrotorch-cubecl). Return an error directing the
                // caller to use `.cpu()` for sub-region reads instead.
                //
                // This path is only hit for non-contiguous XPU tensors, which
                // are rare in practice. If this becomes a bottleneck, add an
                // `upload_slice` method to `CubeStorageHandle`. Issue #673.
                let _ = slice;
                Err(crate::error::FerrotorchError::InvalidArgument {
                    message: format!(
                        "try_clone_subregion on XPU storage is not yet supported \
                         (offset={offset}, numel={numel}); call .cpu() first. Issue #673."
                    ),
                })
            }
            StorageBuffer::Meta { .. } => Ok(Self::meta(numel)),
        }
    }
}

impl<T: Element> Clone for TensorStorage<T> {
    /// Clone the storage. Delegates to [`Self::try_clone`] so the GPU/CubeCL
    /// branches share one fallible-clone implementation.
    ///
    /// # Panics
    /// Panics with a structured message naming the underlying [`crate::error::FerrotorchError`]
    /// (most commonly [`crate::error::FerrotorchError::DeviceUnavailable`] when no GPU backend
    /// is registered, or a backend `clone_buffer` failure). Use
    /// [`Self::try_clone`] when you need to handle the failure explicitly
    /// instead of panicking.
    fn clone(&self) -> Self {
        match self.try_clone() {
            Ok(cloned) => cloned,
            Err(e) => panic!(
                "TensorStorage::clone failed: {e}. \
                 Use TensorStorage::try_clone() to handle this case explicitly."
            ),
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
        // GPU/CubeCL buffers are dropped normally (runtime handles cleanup).
    }
}

impl<T: Element> std::fmt::Debug for StorageBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StorageBuffer::Cpu(v) => write!(f, "Cpu({} elements)", v.len()),
            StorageBuffer::Gpu(h) => write!(f, "Gpu({h:?})"),
            StorageBuffer::Cubecl(h) => {
                write!(f, "Cubecl(ordinal={}, len={})", h.ordinal(), h.len())
            }
            StorageBuffer::Meta { numel, .. } => write!(f, "Meta({numel} elements)"),
        }
    }
}
