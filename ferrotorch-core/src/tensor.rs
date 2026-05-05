use std::fmt;
use std::sync::{Arc, Mutex};

use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::shape::{c_contiguous_strides, channels_last_3d_strides, channels_last_strides};
use crate::storage::TensorStorage;

/// Describes the physical memory layout of a tensor.
///
/// The *shape* (logical dimension order) never changes — only the strides are
/// rearranged so that the underlying data is stored in a different order.
///
/// [CL-309] WU-05: channels-last memory format support
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MemoryFormat {
    /// Standard C-contiguous / row-major layout (NCHW for 4D tensors).
    Contiguous,
    /// Channels-last layout for 4D tensors: physical order is NHWC.
    /// The shape remains `[N, C, H, W]`, but strides are `[H*W*C, 1, W*C, C]`.
    ChannelsLast,
    /// Channels-last layout for 5D tensors: physical order is NDHWC.
    /// The shape remains `[N, C, D, H, W]`, but strides are `[D*H*W*C, 1, H*W*C, W*C, C]`.
    ChannelsLast3d,
}

/// Unique identifier for a tensor, used for gradient accumulation.
static NEXT_TENSOR_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// A unique, monotonically increasing tensor identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
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
    /// Hook storage for gradient hooks and post-accumulate-grad hooks.
    hooks: Mutex<crate::autograd::hooks::HookStorage<T>>,
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
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
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
                    new_shape,
                    new_numel,
                    self.shape(),
                    self.numel()
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
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
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
                    new_shape,
                    new_numel,
                    self.shape(),
                    self.numel()
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
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
        })
    }

    /// Create a zero-copy view with explicit shape, strides, and offset.
    ///
    /// This is the lowest-level view constructor — used by permute, transpose,
    /// narrow, and other operations that change the logical layout without
    /// copying data. The caller is responsible for ensuring that the given
    /// shape + strides + offset are valid for the underlying storage.
    pub fn stride_view(
        &self,
        new_shape: Vec<usize>,
        new_strides: Vec<isize>,
        new_offset: usize,
    ) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&self.inner.storage),
                shape: new_shape,
                strides: new_strides,
                offset: new_offset,
                grad: Mutex::new(None),
                grad_fn: None,
                requires_grad: false,
                is_leaf: true,
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
        }
    }

    /// Create a zero-copy view with explicit shape, strides, and offset,
    /// with an attached gradient function for autograd.
    pub fn stride_view_operation(
        &self,
        new_shape: Vec<usize>,
        new_strides: Vec<isize>,
        new_offset: usize,
        grad_fn: Arc<dyn GradFn<T>>,
    ) -> Self {
        Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::clone(&self.inner.storage),
                shape: new_shape,
                strides: new_strides,
                offset: new_offset,
                grad: Mutex::new(None),
                grad_fn: Some(grad_fn),
                requires_grad: true,
                is_leaf: false,
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
        }
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
        // In inference mode, skip all autograd bookkeeping — create a plain
        // tensor without grad_fn. This avoids allocating autograd metadata
        // and makes operations faster for pure inference.
        if crate::autograd::no_grad::is_inference_mode() {
            return Self::from_storage(storage, shape, false);
        }

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
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
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

    /// Offset (in number of elements) into the underlying storage.
    ///
    /// Non-zero for views created by narrow, select, or other subregion ops.
    #[inline]
    pub fn storage_offset(&self) -> usize {
        self.inner.offset
    }

    /// Number of elements in the underlying storage buffer.
    ///
    /// May be larger than [`numel()`](Self::numel) for views (transpose,
    /// narrow, as_strided, etc.) that address only a subset of the
    /// storage. Used by stride-manipulation ops (`as_strided`,
    /// `as_strided_copy`) for bounds validation.
    #[inline]
    pub fn storage_len(&self) -> usize {
        self.inner.storage.len()
    }

    /// Borrow the underlying [`TensorStorage`]. Used by ops that need
    /// access to the GPU buffer handle or to share storage Arc-wise.
    #[inline]
    pub fn storage(&self) -> &TensorStorage<T> {
        &self.inner.storage
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

    /// Access the hook storage for this tensor.
    pub(crate) fn hooks(&self) -> &Mutex<crate::autograd::hooks::HookStorage<T>> {
        &self.inner.hooks
    }

    /// Register a gradient hook on this tensor.
    ///
    /// The hook is called during backward whenever a gradient is computed for
    /// this tensor. It receives the gradient and may return `Some(new_grad)` to
    /// replace it, or `None` to keep the original.
    ///
    /// Returns a [`HookHandle`](crate::autograd::hooks::HookHandle) that can
    /// be used to remove the hook later via [`remove_hook`](Self::remove_hook).
    pub fn register_hook<F>(&self, func: F) -> FerrotorchResult<crate::autograd::hooks::HookHandle>
    where
        F: Fn(&Tensor<T>) -> Option<Tensor<T>> + Send + Sync + 'static,
    {
        let mut guard = self
            .inner
            .hooks
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("hook storage mutex: {e}"),
            })?;
        Ok(guard.add_grad_hook(func))
    }

    /// Register a post-accumulate-grad hook on this tensor.
    ///
    /// The hook is called after gradient accumulation completes on a leaf
    /// tensor. It receives a reference to the tensor itself (so the hook can
    /// read `.grad()`). Cannot modify the gradient — use
    /// [`register_hook`](Self::register_hook) for that.
    pub fn register_post_accumulate_grad_hook<F>(
        &self,
        func: F,
    ) -> FerrotorchResult<crate::autograd::hooks::HookHandle>
    where
        F: Fn(&Tensor<T>) + Send + Sync + 'static,
    {
        let mut guard = self
            .inner
            .hooks
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("hook storage mutex: {e}"),
            })?;
        Ok(guard.add_post_accumulate_hook(func))
    }

    /// Remove a previously registered hook by its handle.
    ///
    /// Returns `true` if the hook was found and removed.
    pub fn remove_hook(
        &self,
        handle: crate::autograd::hooks::HookHandle,
    ) -> FerrotorchResult<bool> {
        let mut guard = self
            .inner
            .hooks
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("hook storage mutex: {e}"),
            })?;
        Ok(guard.remove(handle))
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
        let mut guard = self
            .inner
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

    /// Accumulate a gradient additively (used by the backward engine).
    ///
    /// Keeps gradients on their original device to avoid GPU↔CPU round-trips.
    /// When both the existing gradient and the incoming gradient are on GPU
    /// (and the element type is f32), accumulation uses `backend.add_f32()`
    /// entirely on-device.
    pub(crate) fn accumulate_grad(&self, incoming: &Tensor<T>) -> FerrotorchResult<()> {
        use std::any::TypeId;

        let mut guard = self
            .inner
            .grad
            .lock()
            .map_err(|e| FerrotorchError::LockPoisoned {
                message: format!("grad mutex: {e}"),
            })?;
        match guard.as_mut() {
            None => {
                // First gradient: store a detached copy on the same device.
                let (storage, shape) = incoming.clone().into_storage_and_shape()?;
                let tensor = Tensor::from_storage(storage, shape, false)?;
                *guard = Some(Box::new(tensor));
            }
            Some(existing) => {
                // Accumulate: existing_grad += incoming_grad.
                let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();

                // GPU-native path: both on GPU and f32.
                if is_f32 && existing.is_cuda() && incoming.is_cuda() {
                    let backend = crate::gpu_dispatch::gpu_backend()
                        .ok_or(FerrotorchError::DeviceUnavailable)?;
                    if existing.numel() != incoming.numel() {
                        return Err(FerrotorchError::ShapeMismatch {
                            message: format!(
                                "gradient accumulation shape mismatch: {:?} vs {:?}",
                                existing.shape(),
                                incoming.shape()
                            ),
                        });
                    }
                    let sum_handle =
                        backend.add_f32(existing.gpu_handle()?, incoming.gpu_handle()?)?;
                    let storage = TensorStorage::gpu(sum_handle);
                    let combined = Tensor::from_storage(storage, existing.shape().to_vec(), false)?;
                    *guard = Some(Box::new(combined));
                } else {
                    // CPU path (or mixed-device): download if needed and
                    // accumulate on the host.
                    let incoming_data = incoming.data_vec()?;
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
                        *e += n;
                    }
                    // Store on the parameter's device.
                    let device = existing.device();
                    let combined = Tensor::from_storage(
                        TensorStorage::on_device(buf, device)?,
                        existing.shape().to_vec(),
                        false,
                    )?;
                    *guard = Some(Box::new(combined));
                }
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
        if self.inner.storage.is_cubecl() {
            return Err(FerrotorchError::GpuTensorNotAccessible);
        }
        if self.inner.storage.is_meta() {
            return Err(FerrotorchError::InvalidArgument {
                message: "cannot read data from a meta tensor; meta tensors carry shape only. \
                     Call .to(Device::Cpu) to materialize, or use .shape() / .numel() / .device() \
                     for metadata access."
                    .into(),
            });
        }
        if !self.is_contiguous() {
            return Err(FerrotorchError::InvalidArgument {
                message: "tensor is not contiguous; call .contiguous() or use .data_vec()".into(),
            });
        }
        let slice = self.inner.storage.try_as_slice()?;
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
        if self.inner.storage.is_meta() {
            return Err(FerrotorchError::InvalidArgument {
                message: "cannot read data from a meta tensor; meta tensors carry shape only. \
                     Call .to(Device::Cpu) to materialize, or use .shape() / .numel() / .device() \
                     for metadata access."
                    .into(),
            });
        }
        if self.is_cuda() || self.inner.storage.is_cubecl() {
            let cpu_tensor = self.cpu()?;
            Ok(cpu_tensor.data()?.to_vec())
        } else if self.is_contiguous() {
            Ok(self.data()?.to_vec())
        } else {
            // Non-contiguous: gather elements by walking strides.
            // is_cuda/is_cubecl branches above already routed GPU storage,
            // so try_as_slice on this CPU/Meta arm only Errs on Meta — which
            // the data_vec entry guard at the top of the function rejects.
            let slice = self.inner.storage.try_as_slice()?;
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
            let device = self.device();
            return Ok((TensorStorage::on_device(data, device)?, shape));
        }

        let shape = self.inner.shape.clone();
        let offset = self.inner.offset;
        let numel: usize = shape.iter().product();

        // Try to unwrap the inner Arc to get ownership of TensorInner.
        match Arc::try_unwrap(self.inner) {
            Ok(inner) => {
                // We own the inner. Try to unwrap the storage Arc.
                match Arc::try_unwrap(inner.storage) {
                    Ok(storage) if offset == 0 && storage.len() == numel => {
                        // Fast path: sole owner, no offset — zero-copy return.
                        Ok((storage, shape))
                    }
                    Ok(storage) => {
                        // Sole owner but offset or extra elements — extract
                        // the subregion. For CPU we can slice the owned Vec
                        // directly; for GPU we round-trip through the host.
                        let sub = storage.try_clone_subregion(offset, numel)?;
                        Ok((sub, shape))
                    }
                    Err(arc_storage) => {
                        // Storage is shared — clone the relevant subregion.
                        let sub = arc_storage.try_clone_subregion(offset, numel)?;
                        Ok((sub, shape))
                    }
                }
            }
            Err(arc_inner) => {
                // Inner is shared — clone the relevant subregion.
                let sub = arc_inner
                    .storage
                    .try_clone_subregion(arc_inner.offset, numel)?;
                Ok((sub, shape))
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

        let needs_grad_fn =
            self.requires_grad() && !self.is_leaf() && crate::autograd::no_grad::is_grad_enabled();

        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(ordinal)) => {
                // Non-contiguous tensors must be materialized before GPU upload.
                let contiguous_self = if !self.is_contiguous() {
                    crate::methods::contiguous_t(self)?
                } else {
                    self.clone()
                };
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let cpu_data = contiguous_self.data()?;
                // SAFETY: `cpu_data` is a borrow into the contiguous CPU
                // storage of contiguous_self, guaranteed to live for the
                // duration of this scope. Reinterpreting as `&[u8]` is sound
                // because every value of T is fully initialized (it's a
                // numeric Float type with no padding bytes); the byte length
                // is computed from `size_of_val(cpu_data) = cpu_data.len() *
                // size_of::<T>()`, which matches the underlying allocation.
                let bytes = unsafe {
                    std::slice::from_raw_parts(
                        cpu_data.as_ptr() as *const u8,
                        std::mem::size_of_val(cpu_data),
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
                let backend =
                    crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let handle = self.gpu_handle()?;
                let bytes = backend.gpu_to_cpu(handle)?;
                // SAFETY: `bytes` is a `Vec<u8>` returned by gpu_to_cpu
                // containing the raw bytes of a Tensor<T>'s GPU storage —
                // those bytes were originally written by an upload of T-sized
                // values (cpu_to_gpu uploads bytes of T). Wrapping `bytes` in
                // ManuallyDrop prevents its destructor from running, then we
                // reconstruct a Vec<T> from the same allocation: ptr/len/cap
                // are from the now-orphaned Vec<u8> with len/cap divided by
                // size_of::<T>(). This requires bytes.len() and bytes.capacity()
                // to be multiples of size_of::<T>() and the allocator layout to
                // be compatible. The ferrotorch-gpu backend guarantees both
                // (it allocates Vec<u8> with size_of::<T>()-aligned, T-multiple
                // length and capacity for D2H readbacks).
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
            // CPU → XPU: requires a CubeRuntime owned by ferrotorch-xpu.
            // Core cannot perform the H2D upload here. Use
            // `ferrotorch_xpu::XpuDevice::upload(tensor)` or
            // `tensor.to(Device::Xpu(n))` via the xpu crate's integration.
            // Issue #673.
            (Device::Cpu, Device::Xpu(_)) => Err(FerrotorchError::InvalidArgument {
                message: "CPU→XPU transfer requires a CubeRuntime. \
                              Use ferrotorch_xpu::make_xpu_tensor or \
                              ferrotorch_xpu::XpuDevice::upload instead. Issue #673."
                    .into(),
            }),
            // XPU → CPU: real D2H readback via the CubeStorageHandle.
            // This is the explicit transfer path — mirrors PyTorch's
            // `.cpu()` which requires a synchronous D2H copy. Issue #673.
            (Device::Xpu(_), Device::Cpu) => {
                let handle = self.inner.storage.cubecl_handle().ok_or_else(|| {
                    FerrotorchError::InvalidArgument {
                        message: "XPU→CPU transfer: storage does not contain a CubeCL handle. \
                                  This tensor may have been created before issue #673 was applied."
                            .into(),
                    }
                })?;
                // f32 readback only for now — T=f32 is the only supported XPU dtype.
                let host_f32 = handle.read_to_host()?;
                // SAFETY: `T` on XPU is always `f32` (the only dtype the cubecl
                // kernels support). `host_f32` contains exactly `handle.len()`
                // f32 values. We transmute via bytecast rather than assuming
                // T=f32 at the type level, so this is gated at runtime.
                let data: Vec<T> = {
                    if std::mem::size_of::<T>() != std::mem::size_of::<f32>() {
                        return Err(FerrotorchError::InvalidArgument {
                            message: format!(
                                "XPU→CPU: expected f32 storage (size 4), got size {}; \
                                 only f32 XPU tensors are supported. Issue #673.",
                                std::mem::size_of::<T>()
                            ),
                        });
                    }
                    // SAFETY: T and f32 have the same size (checked above).
                    // Both are plain-data numeric types with no padding.
                    // The bytes came from a cubecl kernel that wrote f32 values.
                    unsafe {
                        let mut md = std::mem::ManuallyDrop::new(host_f32);
                        Vec::from_raw_parts(md.as_mut_ptr() as *mut T, md.len(), md.capacity())
                    }
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
            // XPU → XPU on a different ordinal: route through CPU for
            // now. CL-452.
            (Device::Xpu(a), Device::Xpu(b)) if a != b => {
                let cpu = self.to(Device::Cpu)?;
                cpu.to(Device::Xpu(b))
            }
            // CUDA ↔ XPU: round-trip via CPU. CL-452.
            (Device::Cuda(_), Device::Xpu(_)) => {
                let cpu = self.to(Device::Cpu)?;
                cpu.to(device)
            }
            (Device::Xpu(_), Device::Cuda(_)) => {
                let cpu = self.to(Device::Cpu)?;
                cpu.to(device)
            }
            // Move TO the meta device: drop the data, keep shape only.
            // Works from any source device.
            (_, Device::Meta) => {
                let storage = TensorStorage::meta(self.numel());
                Tensor::from_storage(storage, self.shape().to_vec(), self.requires_grad())
            }
            // Move FROM the meta device: cannot materialize random data,
            // so this errors. Users should construct fresh tensors with
            // creation::zeros / randn / etc. on the target device instead.
            (Device::Meta, _) => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "cannot move a meta tensor to {device} -- meta tensors carry no data. \
                     Construct a real tensor on {device} via creation::zeros/randn/etc."
                ),
            }),
            _ => Ok(self.clone()),
        }
    }

    /// Like [`to`](Self::to), but uses pinned (page-locked) host memory for
    /// the CPU→CUDA transfer when applicable.
    ///
    /// On CPU→CUDA, allocates a temporary pinned host buffer, copies the
    /// tensor data into it, and uses DMA to transfer to the device. This is
    /// roughly 2x faster than the regular `to()` path for large buffers
    /// because it avoids one extra page-locked staging copy inside the CUDA
    /// driver. For small buffers (< ~64KB) the pinning overhead may
    /// outweigh the gain — measure before defaulting to this path.
    ///
    /// Behaves identically to [`to`](Self::to) for CPU→CPU, CUDA→CPU, and
    /// cross-GPU paths (which all bypass pinned memory).
    ///
    /// Used by `ferrotorch_data::DataLoader` when `pin_memory(true)` is set
    /// alongside a target device.
    pub fn to_pinned(&self, device: Device) -> FerrotorchResult<Tensor<T>> {
        if self.device() == device {
            return Ok(self.clone());
        }

        // Only the CPU→CUDA case benefits from pinned memory; for everything
        // else fall through to the regular `to()` path.
        match (self.device(), device) {
            (Device::Cpu, Device::Cuda(_)) => {
                let needs_grad_fn = self.requires_grad()
                    && !self.is_leaf()
                    && crate::autograd::no_grad::is_grad_enabled();

                // Materialize non-contiguous tensors before upload.
                let contiguous_self = if !self.is_contiguous() {
                    crate::methods::contiguous_t(self)?
                } else {
                    self.clone()
                };
                let cpu_data = contiguous_self.data()?;
                // on_device_pinned takes ownership of the Vec, so we copy.
                let owned: Vec<T> = cpu_data.to_vec();
                let storage = TensorStorage::on_device_pinned(owned, device)?;
                if needs_grad_fn {
                    let grad_fn = Arc::new(ToDeviceBackward {
                        source: self.clone(),
                    });
                    Tensor::from_operation(storage, self.shape().to_vec(), grad_fn)
                } else {
                    Tensor::from_storage(storage, self.shape().to_vec(), self.requires_grad())
                }
            }
            _ => self.to(device),
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

    /// Returns `true` if this tensor is on the meta device (no backing data).
    #[inline]
    pub fn is_meta(&self) -> bool {
        self.device().is_meta()
    }

    /// Returns `true` if this tensor is on a CUDA GPU.
    #[inline]
    pub fn is_cuda(&self) -> bool {
        self.device().is_cuda()
    }

    /// Returns `true` if this tensor is on an XPU (CubeCL / Intel GPU) device.
    #[inline]
    pub fn is_xpu(&self) -> bool {
        matches!(self.device(), crate::device::Device::Xpu(_))
    }

    /// Get the GPU buffer handle. Returns `Err` for CPU tensors.
    pub fn gpu_handle(&self) -> FerrotorchResult<&crate::gpu_dispatch::GpuBufferHandle> {
        self.inner
            .storage
            .gpu_handle()
            .ok_or(FerrotorchError::InvalidArgument {
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
    #[allow(clippy::mut_from_ref)]
    pub unsafe fn data_mut(&self) -> FerrotorchResult<&mut [T]> {
        if !self.is_contiguous() {
            return Err(FerrotorchError::InvalidArgument {
                message: "data_mut requires a contiguous tensor".into(),
            });
        }
        let storage_ptr = Arc::as_ptr(&self.inner.storage) as *mut TensorStorage<T>;
        // SAFETY: Caller guarantees exclusive access (optimizer step inside no_grad).
        let storage = unsafe { &mut *storage_ptr };
        // Returns Err(GpuTensorNotAccessible) for GPU/Cubecl/Meta storage —
        // the deprecated `as_mut_slice` panicked here. Optimizer step
        // implementations now get a clean error path for misuse against
        // GPU-resident parameters.
        let slice = storage.try_as_mut_slice()?;
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
            let backend =
                crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match storage.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            // SAFETY: `new_data: &[T]` is borrowed for the duration of this
            // function and is fully initialized (it's a slice of T values
            // with no padding bytes — T is a numeric Float type). Reading
            // its bytes via reinterpretation is sound; the requested length
            // `size_of_val(new_data) = new_data.len() * size_of::<T>()`
            // matches the actual byte size of the underlying allocation.
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    new_data.as_ptr() as *const u8,
                    std::mem::size_of_val(new_data),
                )
            };
            let new_handle = backend.cpu_to_gpu(bytes, std::mem::size_of::<T>(), ordinal)?;
            storage.data = crate::storage::StorageBuffer::Gpu(new_handle);
        } else {
            // is_gpu() branch above already routed Gpu storage; try_as_mut_slice
            // here Errs only on Cubecl/Meta — neither of which an update_data
            // caller in the optimizer hot path is expected to hit, so the
            // explicit Err is a strict improvement over the deprecated panic.
            let slice = storage.try_as_mut_slice()?;
            let offset = self.inner.offset;
            slice[offset..offset + numel].copy_from_slice(new_data);
        }

        Ok(())
    }

    /// Replace this tensor's storage with a new `TensorStorage` in-place.
    ///
    /// Used by GPU-native optimizer steps that compute the updated parameter
    /// entirely on-device and need to swap the underlying buffer without a
    /// CPU round-trip.
    ///
    /// # Safety
    ///
    /// Same as [`update_data`]: caller must ensure exclusive access. The new
    /// storage must have the same number of elements as the tensor and reside
    /// on the same device.
    pub unsafe fn update_storage(&self, new_storage: TensorStorage<T>) -> FerrotorchResult<()> {
        let numel = self.numel();
        if new_storage.len() != numel {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "update_storage: new storage has {} elements but tensor has {}",
                    new_storage.len(),
                    numel,
                ),
            });
        }

        let storage_ptr = Arc::as_ptr(&self.inner.storage) as *mut TensorStorage<T>;
        // SAFETY: Caller guarantees exclusive access (optimizer step inside
        // no_grad).
        //
        // Critical: `ptr::replace` returns the OLD value so its destructor
        // runs. The previous `ptr::write` here treated the target as
        // uninitialized memory and dropped nothing, leaking the old
        // `TensorStorage` — and in the GPU case, every leaked storage also
        // leaked its `GpuBufferHandle` -> `CudaBuffer` -> pooled `CudaSlice`.
        // Running optimizer steps in a long training loop (e.g. AdamW
        // foreach, which calls `update_storage` once per parameter per
        // step) would climb toward VRAM exhaustion at a rate proportional
        // to parameter size × step count. Replacing with `ptr::replace` +
        // explicit `drop` restores correct single-ownership semantics.
        let old = unsafe { std::ptr::replace(storage_ptr, new_storage) };
        drop(old);

        Ok(())
    }

    /// Run `f` with mutable access to this tensor's underlying
    /// [`GpuBufferHandle`], in-place.
    ///
    /// This is a safe wrapper for the optimizer fast-path that fuses the
    /// parameter update directly into a GPU kernel: the kernel needs an
    /// `&mut GpuBufferHandle` aliased into the param tensor's storage, but
    /// `Tensor` is `Arc`-shared so a naïve `&self -> &mut Storage` route
    /// requires `unsafe` at every call site. By centralizing the
    /// `Arc::as_ptr -> *mut TensorStorage<T>` cast inside this single
    /// method and returning `Err(FerrotorchError::DeviceUnavailable)` for
    /// non-GPU storage, callers do not need to write any `unsafe` of their
    /// own.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::DeviceUnavailable`] when this tensor's
    /// storage is not GPU-resident.
    ///
    /// # Safety contract (encapsulated)
    ///
    /// This method is **safe** because the caller cannot violate any
    /// invariant exposed through it: the closure receives a fresh
    /// `&mut GpuBufferHandle` whose lifetime is bounded by the body of
    /// this method, and no other reference to the storage can be created
    /// concurrently from within the closure (the closure is `FnOnce`).
    /// The only remaining hazard is concurrent access to the same `Arc`
    /// from another thread — `Tensor` is not `Sync` for storage mutation
    /// purposes, and the optimizer step that drives this method holds
    /// `&mut self` on the outer `Optimizer` for the whole step, so no
    /// other handle can be observing or mutating this storage during the
    /// call. This is the same exclusive-access guarantee that
    /// [`update_data`] and [`update_storage`] depend on; this method
    /// simply lets the optimizer mutate the GPU handle in place rather
    /// than swap the entire `TensorStorage`.
    pub fn with_gpu_handle_mut<R>(
        &self,
        f: impl FnOnce(&mut crate::gpu_dispatch::GpuBufferHandle) -> FerrotorchResult<R>,
    ) -> FerrotorchResult<R> {
        let storage_ptr = Arc::as_ptr(&self.inner.storage) as *mut TensorStorage<T>;
        // SAFETY: We materialize a single `&mut TensorStorage<T>` for the
        // duration of the call. The storage lives behind `Arc`, but the
        // optimizer-step caller holds `&mut self` on the outer optimizer
        // for the whole step, so no other handle into this storage can be
        // observed or mutated concurrently — neither this thread (the
        // closure is `FnOnce` and consumes the `&mut` borrow chain) nor
        // any other thread (the storage is not exposed to a sharing
        // primitive that would let another thread reach it during the
        // optimizer step). The resulting `&mut GpuBufferHandle` reborrows
        // out of that exclusive `&mut TensorStorage<T>` and shares its
        // lifetime, so the borrow checker enforces non-aliasing within
        // the closure body. No reference produced here outlives the call.
        let storage: &mut TensorStorage<T> = unsafe { &mut *storage_ptr };
        let handle = storage
            .gpu_handle_mut()
            .ok_or(FerrotorchError::DeviceUnavailable)?;
        f(handle)
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
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
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
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
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

    /// Check whether this tensor is contiguous in a specific memory format.
    ///
    /// - `MemoryFormat::Contiguous` — standard C-order (NCHW for 4D).
    /// - `MemoryFormat::ChannelsLast` — NHWC stride pattern for 4D tensors.
    /// - `MemoryFormat::ChannelsLast3d` — NDHWC stride pattern for 5D tensors.
    ///
    /// Dimensions of size 1 are treated as matching any stride, consistent
    /// with PyTorch behaviour.
    ///
    /// [CL-309] WU-05: channels-last memory format support
    pub fn is_contiguous_for(&self, format: MemoryFormat) -> bool {
        match format {
            MemoryFormat::Contiguous => self.is_contiguous(),
            MemoryFormat::ChannelsLast => {
                if self.ndim() != 4 {
                    return false;
                }
                let expected = channels_last_strides(&self.inner.shape);
                strides_match_with_size1(&self.inner.shape, &self.inner.strides, &expected)
            }
            MemoryFormat::ChannelsLast3d => {
                if self.ndim() != 5 {
                    return false;
                }
                let expected = channels_last_3d_strides(&self.inner.shape);
                strides_match_with_size1(&self.inner.shape, &self.inner.strides, &expected)
            }
        }
    }

    /// Rearrange this tensor to the target memory format.
    ///
    /// If the tensor is already contiguous in the target format, returns a
    /// cheap clone (shared storage). Otherwise, physically rearranges the
    /// data and returns a new tensor with the correct strides.
    ///
    /// The *shape* is never changed — only the strides (and possibly the
    /// underlying data order) are altered.
    ///
    /// [CL-309] WU-05: channels-last memory format support
    pub fn to_memory_format(&self, format: MemoryFormat) -> FerrotorchResult<Self> {
        if self.is_contiguous_for(format) {
            return Ok(self.clone());
        }
        self.materialize_format(format)
    }

    /// Return a tensor that is contiguous in the given memory format,
    /// materializing (copying) the data if necessary.
    ///
    /// Equivalent to `.to_memory_format(format)` — both names are provided
    /// for API familiarity: `contiguous()` is the PyTorch-style entry point
    /// while `to_memory_format()` is the explicit variant.
    ///
    /// [CL-309] WU-05: channels-last memory format support
    pub fn contiguous_in(&self, format: MemoryFormat) -> FerrotorchResult<Self> {
        self.to_memory_format(format)
    }

    /// Physically rearrange data into the target memory format.
    ///
    /// Called when the tensor is NOT already contiguous in `format`.
    /// Gathers elements in the physical order dictated by the target strides
    /// and writes them into a fresh, contiguous-in-format buffer.
    ///
    /// [CL-309] WU-05: channels-last memory format support
    fn materialize_format(&self, format: MemoryFormat) -> FerrotorchResult<Self> {
        let shape = &self.inner.shape;
        let ndim = shape.len();

        match format {
            MemoryFormat::ChannelsLast if ndim != 4 => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("ChannelsLast requires a 4D tensor, got {ndim}D"),
                });
            }
            MemoryFormat::ChannelsLast3d if ndim != 5 => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("ChannelsLast3d requires a 5D tensor, got {ndim}D"),
                });
            }
            _ => {}
        }

        let target_strides = match format {
            MemoryFormat::Contiguous => c_contiguous_strides(shape),
            MemoryFormat::ChannelsLast => channels_last_strides(shape),
            MemoryFormat::ChannelsLast3d => channels_last_3d_strides(shape),
        };

        // GPU fast path (CL-455): for non-meta CUDA tensors of rank
        // <= 8, gather the source view directly into a fresh contiguous
        // device buffer using gpu_strided_copy. The trick is that
        // gpu_strided_copy iterates its output linearly using
        // c-contiguous output strides — so we pass a *permuted* shape
        // and src_stride pair such that the linear iteration order
        // matches the target memory layout (e.g., NHWC for
        // ChannelsLast). The gathered buffer is then naturally
        // contiguous in the target format.
        if self.is_cuda() && ndim <= 8 {
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                use std::any::TypeId;
                let perm = format_permutation(format, ndim);
                let permuted_shape: Vec<usize> = perm.iter().map(|&d| shape[d]).collect();
                let permuted_src_strides: Vec<isize> =
                    perm.iter().map(|&d| self.inner.strides[d]).collect();
                let in_handle = self.gpu_handle()?;
                let src_offset = self.inner.offset;

                let out_handle = if TypeId::of::<T>() == TypeId::of::<f32>() {
                    backend.strided_copy_f32(
                        in_handle,
                        &permuted_shape,
                        &permuted_src_strides,
                        src_offset,
                    )
                } else if TypeId::of::<T>() == TypeId::of::<f64>() {
                    backend.strided_copy_f64(
                        in_handle,
                        &permuted_shape,
                        &permuted_src_strides,
                        src_offset,
                    )
                } else {
                    // Unsupported dtype — fall through to CPU path.
                    return self.materialize_format_cpu(format, target_strides);
                };

                if let Ok(handle) = out_handle {
                    let storage = TensorStorage::gpu(handle);
                    return Ok(Self {
                        inner: Arc::new(TensorInner {
                            id: TensorId::next(),
                            storage: Arc::new(storage),
                            shape: shape.to_vec(),
                            strides: target_strides,
                            offset: 0,
                            grad: Mutex::new(None),
                            grad_fn: None,
                            requires_grad: self.inner.requires_grad,
                            is_leaf: true,
                            hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
                        }),
                    });
                }
                // Kernel failure — fall through to CPU.
            }
        }

        self.materialize_format_cpu(format, target_strides)
    }

    /// CPU path for [`materialize_format`]. Always valid for any
    /// layout; used as a fallback when the GPU fast path declines or
    /// errors. CL-455.
    fn materialize_format_cpu(
        &self,
        _format: MemoryFormat,
        target_strides: Vec<isize>,
    ) -> FerrotorchResult<Self> {
        let shape = &self.inner.shape;
        let ndim = shape.len();
        let numel = self.numel();
        let src_strides = &self.inner.strides;
        let offset = self.inner.offset;

        let device = self.device();
        let src_owned: Vec<T>;
        let src_ref: &[T] = if self.is_cuda() {
            src_owned = self.data_vec()?;
            &src_owned
        } else {
            // is_cuda branch above routes Gpu storage; try_as_slice
            // here Errs only on Cubecl/Meta — both signal a misuse of
            // materialize_format_cpu and propagate cleanly.
            self.inner.storage.try_as_slice()?
        };

        let mut dst = vec![<T as num_traits::Zero>::zero(); numel];

        let mut indices = vec![0usize; ndim];
        for _ in 0..numel {
            let mut src_flat = offset as isize;
            let mut dst_flat: isize = 0;
            for d in 0..ndim {
                src_flat += indices[d] as isize * src_strides[d];
                dst_flat += indices[d] as isize * target_strides[d];
            }
            dst[dst_flat as usize] = src_ref[src_flat as usize];

            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < shape[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        let storage = TensorStorage::on_device(dst, device)?;
        Ok(Self {
            inner: Arc::new(TensorInner {
                id: TensorId::next(),
                storage: Arc::new(storage),
                shape: shape.to_vec(),
                strides: target_strides,
                offset: 0,
                grad: Mutex::new(None),
                grad_fn: None,
                requires_grad: self.inner.requires_grad,
                is_leaf: true,
                hooks: Mutex::new(crate::autograd::hooks::HookStorage::new()),
            }),
        })
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

    /// Number of strong references to the outer `Arc<TensorInner>`.
    ///
    /// Used by the backward engine to decide whether in-place gradient
    /// accumulation is safe (refcount == 1 means exclusive ownership).
    #[inline]
    pub(crate) fn inner_refcount(&self) -> usize {
        Arc::strong_count(&self.inner)
    }

    /// Number of strong references to the inner `Arc<TensorStorage>`.
    ///
    /// Even when `inner_refcount() == 1`, the storage may be shared
    /// (e.g. via `view_reshape` or `detach`). Both must be 1 for
    /// in-place mutation to be safe.
    #[inline]
    pub(crate) fn storage_refcount(&self) -> usize {
        Arc::strong_count(&self.inner.storage)
    }

    /// Get a reference to the inner storage `Arc`.
    ///
    /// Exposed for optimizer kernels that need to modify the param's GPU
    /// buffer in-place via `unsafe` pointer cast (same pattern as
    /// `update_data`).
    #[inline]
    pub fn inner_storage_arc(&self) -> &Arc<TensorStorage<T>> {
        &self.inner.storage
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

/// Compare actual strides against expected strides, treating dimensions
/// of size 1 as wildcards (any stride is acceptable for size-1 dims).
///
/// This matches PyTorch's contiguity semantics where a size-1 dimension
/// does not constrain the stride because it only ever indexes at 0.
///
/// Permutation that maps *output position d* (in linear iteration
/// order over a buffer that's contiguous in `format`) to the
/// corresponding *input dim*. Used by [`materialize_format`] to
/// reuse [`gpu_strided_copy`](crate::gpu_dispatch::GpuBackend::strided_copy_f32)
/// for non-c-contiguous targets like channels-last.
///
/// For an `ndim`-dimensional input shape `[d0, d1, ..., d_{n-1}]`:
///
/// - `Contiguous` → identity `[0, 1, 2, ..., n-1]`. Linear iteration
///   over the c-contiguous output is the same as iterating dim 0
///   slowest.
/// - `ChannelsLast` (4D, NCHW input → NHWC output): `[0, 2, 3, 1]`.
///   Iterating output[n*HWC + h*WC + w*C + c] linearly produces the
///   coordinate sequence (n, h, w, c), so we want input dim mapping
///   {output 0 → input N=0, output 1 → input H=2, output 2 → input
///   W=3, output 3 → input C=1}.
/// - `ChannelsLast3d` (5D, NCDHW → NDHWC): `[0, 2, 3, 4, 1]`. Same
///   reasoning extended to volumetric tensors.
///
/// `ndim` is taken as a parameter so callers can pass it without
/// re-computing from the input. The function only returns
/// non-identity permutations for the formats whose rank constraints
/// are satisfied; everything else returns the identity.
///
/// CL-455.
fn format_permutation(format: MemoryFormat, ndim: usize) -> Vec<usize> {
    match format {
        MemoryFormat::ChannelsLast if ndim == 4 => vec![0, 2, 3, 1],
        MemoryFormat::ChannelsLast3d if ndim == 5 => vec![0, 2, 3, 4, 1],
        // Contiguous (or any rank-mismatched format that the caller
        // already validated): identity permutation.
        _ => (0..ndim).collect(),
    }
}

/// [CL-309] WU-05: channels-last memory format support
fn strides_match_with_size1(shape: &[usize], actual: &[isize], expected: &[isize]) -> bool {
    if actual.len() != expected.len() {
        return false;
    }
    for i in 0..shape.len() {
        if shape[i] != 1 && actual[i] != expected[i] {
            return false;
        }
    }
    true
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
            .field("grad_fn", &self.inner.grad_fn.as_ref().map(|gf| gf.name()))
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
    // reason: round-trip bit-equality — the storage contains the exact bit
    // pattern of 42.0 and item() reads it without arithmetic, so equality is
    // the correct check.
    #[allow(clippy::float_cmp)]
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
        assert!(
            !t.is_same(&view),
            "view_operation creates new tensor identity"
        );
    }

    #[test]
    fn test_clone_shares_grad() {
        let storage = TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]);
        let t = Tensor::from_storage(storage, vec![3], true).unwrap();
        let t2 = t.clone();

        // Accumulate grad via one clone.
        let g =
            Tensor::from_storage(TensorStorage::cpu(vec![0.1, 0.2, 0.3]), vec![3], false).unwrap();
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

        let g1 =
            Tensor::from_storage(TensorStorage::cpu(vec![0.1, 0.2, 0.3]), vec![3], false).unwrap();
        t.accumulate_grad(&g1).unwrap();

        let grad = t.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 0.1).abs() < 1e-7);

        let g2 =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 1.0, 1.0]), vec![3], false).unwrap();
        t.accumulate_grad(&g2).unwrap();

        let grad = t.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 1.1).abs() < 1e-6);
        assert!((data[1] - 1.2).abs() < 1e-6);
        assert!((data[2] - 1.3).abs() < 1e-6);
    }
}
