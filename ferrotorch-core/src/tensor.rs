use std::fmt;
use std::sync::{Arc, Mutex};

use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::shape::c_contiguous_strides;
use crate::storage::TensorStorage;

/// Unique identifier for a tensor, used for gradient accumulation.
static NEXT_TENSOR_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A unique, monotonically increasing tensor identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TensorId(u64);

impl TensorId {
    fn next() -> Self {
        Self(NEXT_TENSOR_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

/// The backward function trait for reverse-mode automatic differentiation.
///
/// Every differentiable operation implements this trait. The autograd engine
/// calls `backward()` during the reverse pass, passing the upstream gradient
/// and receiving gradients for each input.
pub trait GradFn<T: Float>: Send + Sync + fmt::Debug {
    /// Compute gradients of inputs given gradient of output.
    ///
    /// Returns one `Option<Tensor<T>>` per input: `None` for inputs that
    /// don't require gradients.
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>>;

    /// References to input tensors for graph traversal.
    fn inputs(&self) -> Vec<&Tensor<T>>;

    /// Name of this operation (e.g., "AddBackward", "MatmulBackward").
    fn name(&self) -> &'static str;
}

/// Inner storage for a tensor, shared via `Arc`.
///
/// `Tensor<T>` is a thin `Arc` wrapper around this struct. Cloning a tensor
/// clones the `Arc`, so all copies share the same identity, data, and grad
/// storage. This is essential for autograd: the backward engine writes
/// gradients to the same `TensorInner` that the user holds.
struct TensorInner<T: Float> {
    id: TensorId,
    storage: Arc<TensorStorage<T>>,
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,
    grad: Mutex<Option<Box<Tensor<T>>>>,
    grad_fn: Option<Arc<dyn GradFn<T>>>,
    requires_grad: bool,
    is_leaf: bool,
}

/// The central type. A dynamically-shaped tensor with gradient tracking
/// and device placement.
///
/// Internally an `Arc<TensorInner>` — cloning a tensor is cheap and
/// preserves identity. Two clones of the same tensor share the same
/// data, grad, and TensorId.
///
/// # Type parameter
///
/// `T` must implement [`Float`] — currently `f32` or `f64`. This bound
/// ensures the tensor can participate in gradient computation.
pub struct Tensor<T: Float = f32> {
    inner: Arc<TensorInner<T>>,
}

// --- Construction ---

impl<T: Float> Tensor<T> {
    /// Create a new leaf tensor from raw components.
    pub fn from_storage(
        storage: TensorStorage<T>,
        shape: Vec<usize>,
        requires_grad: bool,
    ) -> FerrotorchResult<Self> {
        let numel: usize = shape.iter().product();

        if numel > storage.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "shape {:?} requires {} elements but storage has {}",
                    shape,
                    numel,
                    storage.len()
                ),
            });
        }

        let strides = c_contiguous_strides(&shape);

        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::new(storage),
                shape,
                strides,
                offset: 0,
                grad: Mutex::new(None),
                grad_fn: None,
                requires_grad,
                is_leaf: true,
            }),
        })
    }

    /// Create a view of this tensor with a different shape, sharing the
    /// same underlying storage. Zero-copy — no data movement.
    ///
    /// The new shape must have the same total number of elements.
    /// Non-contiguous tensors are materialized first (requires a copy).
    pub fn view_reshape(&self, new_shape: Vec<usize>) -> FerrotorchResult<Self> {
        // Non-contiguous tensors must be materialized first — a view over
        // non-contiguous storage with new strides would read wrong elements.
        if !self.is_contiguous() {
            let data = self.data_vec()?;
            let storage = TensorStorage::cpu(data);
            let t = Tensor::from_storage(storage, self.shape().to_vec(), false)?;
            return t.view_reshape(new_shape);
        }

        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "view_reshape: new shape {:?} ({} elements) vs old {:?} ({} elements)",
                    new_shape, new_numel, self.shape(), self.numel()
                ),
            });
        }
        let strides = c_contiguous_strides(&new_shape);
        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&self.inner.storage),
                shape: new_shape,
                strides,
                offset: self.inner.offset,
                grad: Mutex::new(None),
                grad_fn: None,
                requires_grad: false,
                is_leaf: true,
            }),
        })
    }

    /// Create a zero-copy view with a grad_fn attached. Used for shape ops
    /// (squeeze, unsqueeze, reshape, etc.) that don't change data layout.
    /// Shares the underlying storage with the source tensor.
    ///
    /// Non-contiguous tensors are materialized first (requires a copy).
    pub fn view_operation(
        &self,
        new_shape: Vec<usize>,
        grad_fn: Arc<dyn GradFn<T>>,
    ) -> FerrotorchResult<Self> {
        // Non-contiguous tensors must be materialized first — a view over
        // non-contiguous storage with new strides would read wrong elements.
        if !self.is_contiguous() {
            let data = self.data_vec()?;
            let storage = TensorStorage::cpu(data);
            let contiguous =
                Tensor::from_storage(storage, self.shape().to_vec(), self.requires_grad())?;
            return contiguous.view_operation(new_shape, grad_fn);
        }

        let new_numel: usize = new_shape.iter().product();
        if new_numel != self.numel() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "view_operation: new shape {:?} ({} elements) vs {:?} ({} elements)",
                    new_shape, new_numel, self.shape(), self.numel()
                ),
            });
        }
        let strides = c_contiguous_strides(&new_shape);
        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&self.inner.storage),
                shape: new_shape,
                strides,
                offset: self.inner.offset,
                grad: Mutex::new(None),
                grad_fn: Some(grad_fn),
                requires_grad: true,
                is_leaf: false,
            }),
        })
    }

    /// Create a tensor that is the result of an operation (non-leaf).
    ///
    /// The resulting tensor has `requires_grad = true`, `is_leaf = false`,
    /// and the given `grad_fn` attached for reverse-mode autodiff.
    pub fn from_operation(
        storage: TensorStorage<T>,
        shape: Vec<usize>,
        grad_fn: Arc<dyn GradFn<T>>,
    ) -> FerrotorchResult<Self> {
        let numel: usize = shape.iter().product();

        if numel > storage.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "shape {:?} requires {} elements but storage has {}",
                    shape,
                    numel,
                    storage.len()
                ),
            });
        }

        let strides = c_contiguous_strides(&shape);

        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::new(storage),
                shape,
                strides,
                offset: 0,
                grad: Mutex::new(None),
                grad_fn: Some(grad_fn),
                requires_grad: true,
                is_leaf: false,
            }),
        })
    }
}

// --- ToDeviceBackward ---

/// Backward for `Tensor::to(device)`.
///
/// Copies the gradient back to the source tensor's device so that
/// gradients flow through device transfers.
#[derive(Debug)]
struct ToDeviceBackward<T: Float> {
    source: Tensor<T>,
}

impl<T: Float> GradFn<T> for ToDeviceBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let target_device = self.source.device();
        if grad_output.device() == target_device {
            Ok(vec![Some(grad_output.clone())])
        } else {
            Ok(vec![Some(grad_output.to(target_device)?)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.source]
    }

    fn name(&self) -> &'static str {
        "ToDeviceBackward"
    }
}

// --- Accessors ---

impl<T: Float> Tensor<T> {
    #[inline]
    pub fn id(&self) -> TensorId {
        self.inner.id
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.inner.shape
    }

    #[inline]
    pub fn ndim(&self) -> usize {
        self.inner.shape.len()
    }

    #[inline]
    pub fn numel(&self) -> usize {
        self.inner.shape.iter().product()
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.inner.strides
    }

    #[inline]
    pub fn device(&self) -> Device {
        self.inner.storage.device()
    }

    #[inline]
    pub fn requires_grad(&self) -> bool {
        self.inner.requires_grad
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.inner.is_leaf
    }

    #[inline]
    pub fn grad_fn(&self) -> Option<&Arc<dyn GradFn<T>>> {
        self.inner.grad_fn.as_ref()
    }

    /// Read the accumulated gradient. Returns `None` if no gradient has
    /// been computed yet.
    pub fn grad(&self) -> FerrotorchResult<Option<Tensor<T>>> {
        let guard = self
            .inner
            .grad
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("grad mutex: {e}"),
            })?;
        Ok(guard.as_ref().map(|b| (**b).clone()))
    }

    /// Set or replace the accumulated gradient.
    pub fn set_grad(&self, grad: Option<Tensor<T>>) -> FerrotorchResult<()> {
        let mut guard =
            self.inner
                .grad
                .lock()
                .map_err(|e| FerrotorchError::LockPoisoned {
                    message: format!("grad mutex: {e}"),
                })?;
        *guard = grad.map(Box::new);
        Ok(())
    }

    /// Zero out the gradient of this tensor.
    ///
    /// Equivalent to `self.set_grad(None)`. Typically called before each
    /// training iteration to prevent gradient accumulation across steps.
    pub fn zero_grad(&self) -> FerrotorchResult<()> {
        self.set_grad(None)
    }

    /// Returns the `Arc` strong reference count for the tensor's inner storage.
    ///
    /// Used by the backward engine to determine whether in-place gradient
    /// accumulation is safe (strong_count == 1 means we hold the only ref).
    #[inline]
    pub(crate) fn inner_refcount(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Accumulate a gradient additively (used by the backward engine).
    /// Handles GPU tensors by transferring to CPU for accumulation.
    pub(crate) fn accumulate_grad(&self, incoming: &Tensor<T>) -> FerrotorchResult<()> {
        let mut guard =
            self.inner
                .grad
                .lock()
                .map_err(|e| FerrotorchError::LockPoisoned {
                    message: format!("grad mutex: {e}"),
                })?;
        match guard.as_mut() {
            None => {
                // First gradient: clone the incoming gradient (on CPU).
                let cpu_incoming = if incoming.is_cuda() { incoming.cpu()? } else { incoming.clone() };
                let data = cpu_incoming.data()?.to_vec();
                let storage = TensorStorage::cpu(data);
                let tensor = Tensor::from_storage(storage, incoming.shape().to_vec(), false)?;
                *guard = Some(Box::new(tensor));
            }
            Some(existing) => {
                // Accumulate: existing_grad += incoming_grad.
                // Allocate new tensor instead of data_mut() to avoid UB
                // from mutating potentially shared Arc<TensorStorage>.
                let cpu_incoming = if incoming.is_cuda() { incoming.cpu()? } else { incoming.clone() };
                let incoming_data = cpu_incoming.data()?;
                let mut buf = existing.data_vec()?;
                if buf.len() != incoming_data.len() {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "gradient accumulation shape mismatch: {:?} vs {:?}",
                            existing.shape(),
                            incoming.shape()
                        ),
                    });
                }
                for (e, &n) in buf.iter_mut().zip(incoming_data.iter()) {
                    *e = *e + n;
                }
                let combined = Tensor::from_storage(
                    TensorStorage::cpu(buf),
                    existing.shape().to_vec(),
                    false,
                )?;
                *guard = Some(Box::new(combined));
            }
        }
        Ok(())
    }

    /// Borrow the underlying data as a flat slice.
    ///
    /// Returns `Err(GpuTensorNotAccessible)` if the tensor is on a GPU.
    /// Call `.cpu()` first to transfer it.
    ///
    /// Returns `Err` if the tensor is not contiguous — the raw storage
    /// slice would not correspond to the logical element order. Use
    /// [`data_vec()`](Self::data_vec) or call `.contiguous()` first.
    pub fn data(&self) -> FerrotorchResult<&[T]> {
        if self.inner.storage.is_gpu() {
            return Err(FerrotorchError::GpuTensorNotAccessible);
        }
        if !self.is_contiguous() {
            return Err(FerrotorchError::InvalidArgument {
                message: "tensor is not contiguous; call .contiguous() or use .data_vec()".into(),
            });
        }
        let slice = self.inner.storage.as_slice();
        let end = self.inner.offset + self.numel();
        if end > slice.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: "tensor view extends beyond storage".into(),
            });
        }
        Ok(&slice[self.inner.offset..end])
    }

    /// Borrow the underlying data as a flat slice (CPU-only alias for `data()`).
    ///
    /// Identical to [`data()`](Self::data) — returns a zero-copy `&[T]` reference
    /// to the tensor's storage. Returns `Err(GpuTensorNotAccessible)` if the
    /// tensor lives on a GPU; call `.cpu()` first to transfer.
    ///
    /// This alias exists for call-site clarity: use `data_ref()` when you want
    /// to emphasise that no copy is made, vs `data_vec()` which always copies.
    #[inline]
    pub fn data_ref(&self) -> FerrotorchResult<&[T]> {
        self.data()
    }

    /// Get tensor data as an owned `Vec<T>`, transparently transferring from
    /// GPU if needed and correctly handling non-contiguous tensors.
    ///
    /// For contiguous CPU tensors this copies the slice. For non-contiguous
    /// CPU tensors it gathers elements in logical (C-order) sequence. For
    /// GPU tensors it performs a device-to-host transfer.
    pub fn data_vec(&self) -> FerrotorchResult<Vec<T>> {
        if self.is_cuda() {
            let cpu_tensor = self.cpu()?;
            Ok(cpu_tensor.data()?.to_vec())
        } else if self.is_contiguous() {
            Ok(self.data()?.to_vec())
        } else {
            // Non-contiguous: gather elements by walking strides.
            let slice = self.inner.storage.as_slice();
            let shape = &self.inner.shape;
            let strides = &self.inner.strides;
            let offset = self.inner.offset;
            let numel = self.numel();
            let ndim = shape.len();

            let mut result = Vec::with_capacity(numel);
            let mut indices = vec![0usize; ndim];
            for _ in 0..numel {
                let mut flat = offset as isize;
                for d in 0..ndim {
                    flat += indices[d] as isize * strides[d];
                }
                result.push(slice[flat as usize]);
                // Increment multi-index (rightmost first).
                for d in (0..ndim).rev() {
                    indices[d] += 1;
                    if indices[d] < shape[d] {
                        break;
                    }
                    indices[d] = 0;
                }
            }
            Ok(result)
        }
    }

    /// Consume this tensor and return its storage and shape.
    ///
    /// If this is the only reference to the underlying data, the storage Vec
    /// is extracted without copying. Otherwise falls back to cloning.
    /// Used internally to avoid double-copies when rewrapping op results.
    pub fn into_storage_and_shape(self) -> FerrotorchResult<(TensorStorage<T>, Vec<usize>)> {
        // Non-contiguous tensors must be materialized — the raw storage
        // does not match the logical element order.
        if !self.is_contiguous() {
            let data = self.data_vec()?;
            let shape = self.shape().to_vec();
            return Ok((TensorStorage::cpu(data), shape));
        }

        let shape = self.inner.shape.clone();
        // Try to unwrap the inner Arc to get ownership of TensorInner.
        match Arc::try_unwrap(self.inner) {
            Ok(inner) => {
                // We own the inner. Try to unwrap the storage Arc.
                match Arc::try_unwrap(inner.storage) {
                    Ok(storage) => Ok((storage, shape)),
                    Err(arc_storage) => {
                        // Storage is shared — must clone.
                        Ok(((*arc_storage).clone(), shape))
                    }
                }
            }
            Err(arc_inner) => {
                // Inner is shared — must clone data.
                if arc_inner.storage.is_gpu() {
                    // GPU storage cannot be sliced on the host; clone the
                    // entire buffer via the backend's clone_buffer().
                    Ok(((*arc_inner.storage).clone(), shape))
                } else {
                    let data = arc_inner.storage.as_slice();
                    let end = arc_inner.offset + shape.iter().product::<usize>();
                    Ok((TensorStorage::cpu(data[arc_inner.offset..end].to_vec()), shape))
                }
            }
        }
    }

    /// Move this tensor to a device, returning a new tensor.
    ///
    /// If the tensor is already on the target device, returns a cheap clone
    /// (shared Arc storage).
    pub fn to(&self, device: Device) -> FerrotorchResult<Tensor<T>> {
        if self.device() == device {
            return Ok(self.clone());
        }

        let needs_grad_fn = self.requires_grad()
            && !self.is_leaf()
            && crate::autograd::no_grad::is_grad_enabled();

        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(ordinal)) => {
                // Non-contiguous tensors must be materialized before GPU upload.
                let contiguous_self = if !self.is_contiguous() {
                    crate::methods::contiguous_t(self)?
                } else {
                    self.clone()
                };
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(FerrotorchError::DeviceUnavailable)?;
                let cpu_data = contiguous_self.data()?;
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        cpu_data.as_ptr() as *const u8,
                        cpu_data.len() * std::mem::size_of::<T>(),
                    )
                };
                let handle = backend.cpu_to_gpu(bytes, std::mem::size_of::<T>(), ordinal)?;
                let storage = TensorStorage::gpu(handle);
                if needs_grad_fn {
                    let grad_fn = Arc::new(ToDeviceBackward {
                        source: self.clone(),
                    });
                    Tensor::from_operation(storage, self.shape().to_vec(), grad_fn)
                } else {
                    Tensor::from_storage(storage, self.shape().to_vec(), self.requires_grad())
                }
            }
            (Device::Cuda(_), Device::Cpu) => {
                let backend = crate::gpu_dispatch::gpu_backend()
                    .ok_or(FerrotorchError::DeviceUnavailable)?;
                let handle = self.gpu_handle()?;
                let bytes = backend.gpu_to_cpu(handle)?;
                let data: Vec<T> = unsafe {
                    let mut bytes = std::mem::ManuallyDrop::new(bytes);
                    let len = bytes.len() / std::mem::size_of::<T>();
                    let cap = bytes.capacity() / std::mem::size_of::<T>();
                    Vec::from_raw_parts(bytes.as_mut_ptr() as *mut T, len, cap)
                };
                let storage = TensorStorage::cpu(data);
                if needs_grad_fn {
                    let grad_fn = Arc::new(ToDeviceBackward {
                        source: self.clone(),
                    });
                    Tensor::from_operation(storage, self.shape().to_vec(), grad_fn)
                } else {
                    Tensor::from_storage(storage, self.shape().to_vec(), self.requires_grad())
                }
            }
            (Device::Cuda(a), Device::Cuda(b)) if a != b => {
                // Cross-GPU: go through CPU for now
                let cpu = self.to(Device::Cpu)?;
                cpu.to(Device::Cuda(b))
            }
            _ => Ok(self.clone()),
        }
    }

    /// Move to CUDA device 0.
    pub fn cuda(&self) -> FerrotorchResult<Tensor<T>> {
        self.to(Device::Cuda(0))
    }

    /// Move to CPU.
    pub fn cpu(&self) -> FerrotorchResult<Tensor<T>> {
        self.to(Device::Cpu)
    }

    /// Returns `true` if this tensor is on CPU.
    #[inline]
    pub fn is_cpu(&self) -> bool {
        self.device().is_cpu()
    }

    /// Returns `true` if this tensor is on a CUDA GPU.
    #[inline]
    pub fn is_cuda(&self) -> bool {
        self.device().is_cuda()
    }

    /// Get the GPU buffer handle. Returns `Err` for CPU tensors.
    pub fn gpu_handle(&self) -> FerrotorchResult<&crate::gpu_dispatch::GpuBufferHandle> {
        self.inner.storage.gpu_handle().ok_or(FerrotorchError::InvalidArgument {
            message: "tensor is on CPU, not GPU".into(),
        })
    }

    /// Borrow the underlying data as a mutable flat slice.
    ///
    /// # Safety
    ///
    /// The caller must ensure exclusive access to this tensor's storage.
    /// No other references to this tensor's data may exist concurrently.
    /// Optimizer `step()` methods satisfy this requirement: they run inside
    /// `no_grad()` (no graph is being built) and hold `&mut self` (exclusive
    /// access to the optimizer's parameter copies).
    pub unsafe fn data_mut(&self) -> FerrotorchResult<&mut [T]> {
        if !self.is_contiguous() {
            return Err(FerrotorchError::InvalidArgument {
                message: "data_mut requires a contiguous tensor".into(),
            });
        }
        let storage_ptr = Arc::as_ptr(&self.inner.storage) as *mut TensorStorage<T>;
        // SAFETY: Caller guarantees exclusive access (optimizer step inside no_grad).
        let storage = unsafe { &mut *storage_ptr };
        let slice = storage.as_mut_slice();
        let end = self.inner.offset + self.numel();
        if end > slice.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: "tensor view extends beyond storage".into(),
            });
        }
        Ok(&mut slice[self.inner.offset..end])
    }

    /// Write `new_data` into this tensor's storage, preserving tensor identity.
    ///
    /// - **CPU**: copies data into the existing storage Vec.
    /// - **GPU**: uploads data to GPU and replaces the storage buffer.
    ///
    /// This is the device-transparent alternative to `data_mut()` for
    /// optimizer step implementations.
    ///
    /// # Safety
    ///
    /// Same requirements as `data_mut()` — caller must ensure exclusive
    /// access. No concurrent reads or writes to this tensor's storage may
    /// exist. Optimizer `step()` methods satisfy this by running inside
    /// `no_grad()` with `&mut self`.
    pub unsafe fn update_data(&self, new_data: &[T]) -> FerrotorchResult<()> {
        let numel = self.numel();
        if new_data.len() != numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "update_data: new data has {} elements but tensor has {}",
                    new_data.len(),
                    numel,
                ),
            });
        }

        let storage_ptr = Arc::as_ptr(&self.inner.storage) as *mut TensorStorage<T>;
        // SAFETY: Caller guarantees exclusive access (optimizer step inside no_grad).
        let storage = unsafe { &mut *storage_ptr };

        if storage.is_gpu() {
            let backend = crate::gpu_dispatch::gpu_backend()
                .ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match storage.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    new_data.as_ptr() as *const u8,
                    new_data.len() * std::mem::size_of::<T>(),
                )
            };
            let new_handle = backend.cpu_to_gpu(bytes, std::mem::size_of::<T>(), ordinal)?;
            storage.data = crate::storage::StorageBuffer::Gpu(new_handle);
        } else {
            let slice = storage.as_mut_slice();
            let offset = self.inner.offset;
            slice[offset..offset + numel].copy_from_slice(new_data);
        }

        Ok(())
    }

    /// Detach this tensor from the computation graph, returning a new
    /// tensor that shares storage but has no grad_fn.
    pub fn detach(&self) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&self.inner.storage),
                shape: self.inner.shape.clone(),
                strides: self.inner.strides.clone(),
                offset: self.inner.offset,
                grad: Mutex::new(None),
                grad_fn: None,
                requires_grad: false,
                is_leaf: true,
            }),
        }
    }

    /// Return a new tensor with `requires_grad` set.
    pub fn requires_grad_(self, requires_grad: bool) -> Self {
        // Must create a new inner since Arc<TensorInner> is immutable.
        Self {
            inner: Arc::new(TensorInner {
                id: self.inner.id,
                storage: Arc::clone(&self.inner.storage),
                shape: self.inner.shape.clone(),
                strides: self.inner.strides.clone(),
                offset: self.inner.offset,
                grad: Mutex::new(None),
                grad_fn: self.inner.grad_fn.clone(),
                requires_grad,
                is_leaf: self.inner.is_leaf,
            }),
        }
    }

    /// Whether this tensor is contiguous in memory (C-order).
    ///
    /// Dimensions with size 1 can have any stride without affecting
    /// contiguity, since they contribute no index offset.
    pub fn is_contiguous(&self) -> bool {
        if self.inner.shape.is_empty() {
            return true;
        }
        let mut expected_stride: isize = 1;
        for d in (0..self.ndim()).rev() {
            if self.inner.shape[d] == 0 {
                return true;
            }
            if self.inner.shape[d] != 1 && self.inner.strides[d] != expected_stride {
                return false;
            }
            if self.inner.shape[d] != 1 {
                expected_stride *= self.inner.shape[d] as isize;
            }
        }
        true
    }

    /// Returns `true` if this is a scalar (0-dimensional) tensor.
    #[inline]
    pub fn is_scalar(&self) -> bool {
        self.inner.shape.is_empty()
    }

    /// For a scalar tensor, extract the single value.
    pub fn item(&self) -> FerrotorchResult<T> {
        if !self.is_scalar() && self.numel() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "item() requires a scalar or single-element tensor, got shape {:?}",
                    self.shape()
                ),
            });
        }
        let data = self.data()?;
        Ok(data[0])
    }

    /// Returns true if two tensors are the same object (same Arc).
    pub fn is_same(&self, other: &Self) -> bool {
        self.inner.id == other.inner.id
    }

    /// Returns true if two tensors share the same underlying storage allocation.
    ///
    /// Used by tests to verify that view operations (squeeze, unsqueeze, flatten)
    /// are zero-copy.
    #[cfg(test)]
    pub(crate) fn shares_storage(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner.storage, &other.inner.storage)
    }
}

// --- Trait impls ---

impl<T: Float> Clone for Tensor<T> {
    /// Clone is cheap — it increments the Arc refcount. Both copies
    /// share the same data, grad, and identity.
    fn clone(&self) -> Self {
        Self {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl<T: Float> fmt::Debug for Tensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("id", &self.inner.id)
            .field("shape", &self.inner.shape)
            .field("device", &self.device())
            .field("requires_grad", &self.inner.requires_grad)
            .field("is_leaf", &self.inner.is_leaf)
            .field(
                "grad_fn",
                &self.inner.grad_fn.as_ref().map(|gf| gf.name()),
            )
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    #[test]
    fn test_tensor_from_storage() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = Tensor::from_storage(storage, vec![2, 3], false).unwrap();

        assert_eq!(t.shape(), &[2, 3]);
        assert_eq!(t.strides(), &[3, 1]);
        assert_eq!(t.ndim(), 2);
        assert_eq!(t.numel(), 6);
        assert!(t.is_contiguous());
        assert!(t.is_leaf());
        assert!(!t.requires_grad());
        assert_eq!(t.device(), Device::Cpu);
    }

    #[test]
    fn test_tensor_shape_mismatch() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]);
        let result = Tensor::from_storage(storage, vec![2, 3], false);
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_data_access() {
        let storage = TensorStorage::cpu(vec![1.0f64, 2.0, 3.0]);
        let t = Tensor::from_storage(storage, vec![3], false).unwrap();
        assert_eq!(t.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_tensor_scalar() {
        let storage = TensorStorage::cpu(vec![42.0f32]);
        let t = Tensor::from_storage(storage, vec![], false).unwrap();
        assert!(t.is_scalar());
        assert_eq!(t.item().unwrap(), 42.0);
    }

    #[test]
    fn test_tensor_detach() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0]);
        let t = Tensor::from_storage(storage, vec![2], true).unwrap();
        assert!(t.requires_grad());

        let d = t.detach();
        assert!(!d.requires_grad());
        assert!(d.is_leaf());
        assert!(d.grad_fn().is_none());
    }

    #[test]
    fn test_tensor_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Tensor<f32>>();
        assert_send_sync::<Tensor<f64>>();
    }

    #[test]
    fn test_clone_shares_identity() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0]);
        let t = Tensor::from_storage(storage, vec![2], true).unwrap();
        let t2 = t.clone();

        assert!(t.is_same(&t2));
        assert_eq!(t.id(), t2.id());
    }

    #[test]
    fn test_view_operation_shares_storage() {
        use crate::grad_fns::shape::FlattenBackward;
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = Tensor::from_storage(storage, vec![2, 3], true).unwrap();
        let grad_fn = Arc::new(FlattenBackward::new(t.clone(), t.shape().to_vec()));
        let view = t.view_operation(vec![6], grad_fn).unwrap();
        assert!(t.shares_storage(&view), "view_operation must share storage");
        assert!(!t.is_same(&view), "view_operation creates new tensor identity");
    }

    #[test]
    fn test_clone_shares_grad() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]);
        let t = Tensor::from_storage(storage, vec![3], true).unwrap();
        let t2 = t.clone();

        // Accumulate grad via one clone.
        let g = Tensor::from_storage(TensorStorage::cpu(vec![0.1, 0.2, 0.3]), vec![3], false)
            .unwrap();
        t.accumulate_grad(&g).unwrap();

        // Visible from the other clone.
        let grad = t2.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 0.1).abs() < 1e-7);
    }

    #[test]
    fn test_tensor_grad_accumulation() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]);
        let t = Tensor::from_storage(storage, vec![3], true).unwrap();

        assert!(t.grad().unwrap().is_none());

        let g1 = Tensor::from_storage(TensorStorage::cpu(vec![0.1, 0.2, 0.3]), vec![3], false)
            .unwrap();
        t.accumulate_grad(&g1).unwrap();

        let grad = t.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 0.1).abs() < 1e-7);

        let g2 = Tensor::from_storage(TensorStorage::cpu(vec![1.0, 1.0, 1.0]), vec![3], false)
            .unwrap();
        t.accumulate_grad(&g2).unwrap();

        let grad = t.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 1.1).abs() < 1e-6);
        assert!((data[1] - 1.2).abs() < 1e-6);
        assert!((data[2] - 1.3).abs() < 1e-6);
    }
}
