//! GradFn backward implementations for shape-manipulation operations.
//!
//! Shape ops (reshape, squeeze, unsqueeze, flatten, transpose, expand, cat)
//! are essentially bookkeeping — the data moves around but is never scaled.
//! Their VJPs either reinterpret the gradient buffer under the original
//! shape, transpose it, or split/sum along axes.

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Returns `true` if `T` is `f32`.
#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

/// Returns `true` if `T` is `f64`.
#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

// ---------------------------------------------------------------------------
// GPU-aware helper
// ---------------------------------------------------------------------------

/// Verify a tensor is on CPU, returning it with its device tag.
///
/// Shape ops don't have native GPU kernels yet. Instead of silently
/// downloading from GPU (which hides a costly roundtrip), we error
/// immediately so the caller can move data explicitly.
#[inline]
fn ensure_cpu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Device)> {
    let device = input.device();
    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda {
            op: "shape backward",
        });
    }
    Ok((input.clone(), device))
}

/// Move a tensor to the given device if it isn't already there.
#[inline]
fn restore_device<T: Float>(tensor: Tensor<T>, device: Device) -> FerrotorchResult<Tensor<T>> {
    if device.is_cuda() {
        tensor.to(device)
    } else {
        Ok(tensor)
    }
}

// ---------------------------------------------------------------------------
// ReshapeBackward
// ---------------------------------------------------------------------------

/// Backward for `reshape(x, new_shape)`.
///
/// VJP: `grad_input = reshape(grad_output, original_shape)`.
/// The data is identical — we just reinterpret the flat buffer.
#[derive(Debug)]
pub struct ReshapeBackward<T: Float> {
    input: Tensor<T>,
    /// The shape of `input` before the reshape.
    input_shape: Vec<usize>,
}

impl<T: Float> ReshapeBackward<T> {
    pub fn new(input: Tensor<T>, input_shape: Vec<usize>) -> Self {
        Self { input, input_shape }
    }
}

impl<T: Float> GradFn<T> for ReshapeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Reshape is a pure metadata change — zero-copy view on any device.
        let grad_input = grad_output.view_reshape(self.input_shape.clone())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ReshapeBackward"
    }
}

/// Reshape a tensor to `new_shape`, preserving the computation graph.
///
/// The product of `new_shape` must equal `input.numel()`. Exactly one
/// dimension may be `-1`, in which case it is inferred.
pub fn reshape<T: Float>(input: &Tensor<T>, new_shape: &[isize]) -> FerrotorchResult<Tensor<T>> {
    let numel = input.numel();
    let resolved = resolve_shape(new_shape, numel)?;

    // No-grad fast path: zero-copy view reshape (works on any device).
    if !is_grad_enabled() || !input.requires_grad() {
        return input.view_reshape(resolved);
    }

    // Grad path: zero-copy view with grad_fn attached (works on any device).
    let grad_fn = Arc::new(ReshapeBackward::new(input.clone(), input.shape().to_vec()));
    input.view_operation(resolved, grad_fn)
}

// ---------------------------------------------------------------------------
// FlattenBackward
// ---------------------------------------------------------------------------

/// Backward for `flatten(x)`.
///
/// VJP: `grad_input = reshape(grad_output, original_shape)`.
#[derive(Debug)]
pub struct FlattenBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
}

impl<T: Float> FlattenBackward<T> {
    pub fn new(input: Tensor<T>, input_shape: Vec<usize>) -> Self {
        Self { input, input_shape }
    }
}

impl<T: Float> GradFn<T> for FlattenBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Unflatten is a pure metadata change — zero-copy view on any device.
        let grad_input = grad_output.view_reshape(self.input_shape.clone())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FlattenBackward"
    }
}

/// Flatten a tensor to 1-D.
pub fn flatten<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let numel = input.numel();

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(FlattenBackward::new(input.clone(), input.shape().to_vec()));
        input.view_operation(vec![numel], grad_fn)
    } else {
        input.view_reshape(vec![numel])
    }
}

// ---------------------------------------------------------------------------
// SqueezeBackward
// ---------------------------------------------------------------------------

/// Backward for `squeeze(x, axis)`.
///
/// VJP: `grad_input = unsqueeze(grad_output, axis)` — insert the removed
/// size-1 dimension back.
#[derive(Debug)]
pub struct SqueezeBackward<T: Float> {
    input: Tensor<T>,
    /// The axis that was squeezed (after normalization).
    axis: usize,
}

impl<T: Float> SqueezeBackward<T> {
    pub fn new(input: Tensor<T>, axis: usize) -> Self {
        Self { input, axis }
    }
}

impl<T: Float> GradFn<T> for SqueezeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Re-insert the size-1 dimension at `self.axis` — pure metadata change.
        let mut new_shape = grad_output.shape().to_vec();
        new_shape.insert(self.axis, 1);
        let grad_input = grad_output.view_reshape(new_shape)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SqueezeBackward"
    }
}

/// Remove a size-1 dimension at `axis`.
pub fn squeeze<T: Float>(input: &Tensor<T>, axis: isize) -> FerrotorchResult<Tensor<T>> {
    let norm_axis = crate::shape::normalize_axis(axis, input.ndim())?;

    if input.shape()[norm_axis] != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "squeeze: dimension {} has size {}, expected 1",
                norm_axis,
                input.shape()[norm_axis]
            ),
        });
    }

    let mut new_shape = input.shape().to_vec();
    new_shape.remove(norm_axis);

    // No-grad fast path: zero-copy view reshape.
    if !is_grad_enabled() || !input.requires_grad() {
        return input.view_reshape(new_shape);
    }

    let grad_fn = Arc::new(SqueezeBackward::new(input.clone(), norm_axis));
    input.view_operation(new_shape, grad_fn)
}

// ---------------------------------------------------------------------------
// UnsqueezeBackward
// ---------------------------------------------------------------------------

/// Backward for `unsqueeze(x, axis)`.
///
/// VJP: `grad_input = squeeze(grad_output, axis)` — remove the inserted
/// size-1 dimension.
#[derive(Debug)]
pub struct UnsqueezeBackward<T: Float> {
    input: Tensor<T>,
    /// The axis that was unsqueezed.
    axis: usize,
}

impl<T: Float> UnsqueezeBackward<T> {
    pub fn new(input: Tensor<T>, axis: usize) -> Self {
        Self { input, axis }
    }
}

impl<T: Float> GradFn<T> for UnsqueezeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Remove the size-1 dimension at `self.axis` — pure metadata change.
        let mut new_shape = grad_output.shape().to_vec();
        new_shape.remove(self.axis);
        let grad_input = grad_output.view_reshape(new_shape)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "UnsqueezeBackward"
    }
}

/// Insert a size-1 dimension at `axis`.
///
/// `axis` may be in the range `[-(ndim+1), ndim]` (inclusive on both ends),
/// following PyTorch semantics where a new dimension is inserted *before*
/// the given position.
pub fn unsqueeze<T: Float>(input: &Tensor<T>, axis: isize) -> FerrotorchResult<Tensor<T>> {
    // For unsqueeze, the valid range is [-(ndim+1), ndim].
    let ndim = input.ndim();
    let new_ndim = ndim + 1;
    let ndim_i = new_ndim as isize;

    if axis >= ndim_i || axis < -ndim_i {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "unsqueeze: axis {} is out of bounds for tensor with {} dimensions (new ndim = {})",
                axis, ndim, new_ndim
            ),
        });
    }

    let norm_axis = if axis < 0 {
        (ndim_i + axis) as usize
    } else {
        axis as usize
    };

    let mut new_shape = input.shape().to_vec();
    new_shape.insert(norm_axis, 1);

    // No-grad fast path: zero-copy view reshape.
    if !is_grad_enabled() || !input.requires_grad() {
        return input.view_reshape(new_shape);
    }

    let grad_fn = Arc::new(UnsqueezeBackward::new(input.clone(), norm_axis));
    input.view_operation(new_shape, grad_fn)
}

// ---------------------------------------------------------------------------
// TransposeBackward
// ---------------------------------------------------------------------------

/// Backward for `transpose_2d(x)` (2-D transpose).
///
/// VJP: `grad_input = transpose(grad_output)`.
#[derive(Debug)]
pub struct TransposeBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> TransposeBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for TransposeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Zero-copy stride swap: transpose is its own inverse.
        let grad_input = crate::methods::permute_t(grad_output, &[1, 0])?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "TransposeBackward"
    }
}

/// Transpose a 2-D tensor, preserving the computation graph.
///
/// CPU path: zero-copy O(1) stride swap — shares storage with the input.
/// GPU path: runs a transpose kernel (data copy on device).
pub fn transpose_2d<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("transpose_2d requires 2-D tensor, got {:?}", input.shape()),
        });
    }

    // Zero-copy stride swap — works on both CPU and GPU.
    crate::methods::permute_t(input, &[1, 0])
}

// ---------------------------------------------------------------------------
// ExpandBackward
// ---------------------------------------------------------------------------

/// Backward for `expand(x, new_shape)`.
///
/// VJP: sum along every axis where the input dimension was 1 (and the
/// output dimension was > 1), or where the input had fewer dimensions
/// (implicit leading 1s).
#[derive(Debug)]
pub struct ExpandBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
}

impl<T: Float> ExpandBackward<T> {
    pub fn new(input: Tensor<T>, input_shape: Vec<usize>) -> Self {
        Self { input, input_shape }
    }
}

impl<T: Float> GradFn<T> for ExpandBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        // reduce_grad_to_shape sums over broadcast dimensions (leading dims
        // not in target + size-1 dims) and works natively on GPU via
        // sum_axis_f32/f64 — no CPU roundtrip.
        let grad_input = super::arithmetic::reduce_grad_to_shape(grad_output, &self.input_shape)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ExpandBackward"
    }
}

/// Broadcast (expand) a tensor to `new_shape`.
///
/// Only size-1 dimensions can be expanded. This follows PyTorch's
/// `Tensor.expand()` semantics.
pub fn expand<T: Float>(input: &Tensor<T>, new_shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
    let in_shape = input.shape();
    let out_ndim = new_shape.len();
    let in_ndim = in_shape.len();

    if out_ndim < in_ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "expand: target shape {:?} has fewer dimensions than input {:?}",
                new_shape, in_shape
            ),
        });
    }

    // Validate that non-1 dimensions match.
    for i in 0..in_ndim {
        let in_dim = in_shape[in_ndim - 1 - i];
        let out_dim = new_shape[out_ndim - 1 - i];
        if in_dim != 1 && in_dim != out_dim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "expand: cannot expand dimension {} from {} to {}",
                    in_ndim - 1 - i,
                    in_dim,
                    out_dim
                ),
            });
        }
    }

    // GPU fast path for f32/f64: broadcast-add with a zeros scalar to produce
    // the expanded tensor entirely on device (no CPU roundtrip).
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let device_ord = input.gpu_handle()?.device_ordinal();
            let elem_size = if is_f64::<T>() { 8 } else { 4 };
            let zeros = backend.alloc_zeros(1, elem_size, device_ord)?;
            let expanded = if is_f64::<T>() {
                backend.broadcast_add_f64(input.gpu_handle()?, &zeros, in_shape, &[1], new_shape)?
            } else {
                backend.broadcast_add_f32(input.gpu_handle()?, &zeros, in_shape, &[1], new_shape)?
            };
            let storage = TensorStorage::gpu(expanded);
            return if is_grad_enabled() && input.requires_grad() {
                let grad_fn = Arc::new(ExpandBackward::new(input.clone(), in_shape.to_vec()));
                Tensor::from_operation(storage, new_shape.to_vec(), grad_fn)
            } else {
                Tensor::from_storage(storage, new_shape.to_vec(), false)
            };
        }
    }

    // CPU path — error if GPU tensor without supported dtype.
    if input.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "expand" });
    }

    // Build expanded data via broadcast indexing.
    let in_data = input.data()?;
    let out_numel: usize = new_shape.iter().product();
    let mut out_data = Vec::with_capacity(out_numel);

    for flat in 0..out_numel {
        let idx = broadcast_flat_index(flat, new_shape, in_shape);
        out_data.push(in_data[idx]);
    }

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(ExpandBackward::new(input.clone(), in_shape.to_vec()));
        Tensor::from_operation(TensorStorage::cpu(out_data), new_shape.to_vec(), grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(out_data), new_shape.to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// CatBackward
// ---------------------------------------------------------------------------

/// Backward for `cat(tensors, axis)`.
///
/// VJP: split `grad_output` along `axis` at the original sizes, yielding
/// one gradient per input tensor.
#[derive(Debug)]
pub struct CatBackward<T: Float> {
    inputs: Vec<Tensor<T>>,
    axis: usize,
    /// The size of each input along `axis`.
    split_sizes: Vec<usize>,
}

impl<T: Float> CatBackward<T> {
    pub fn new(inputs: Vec<Tensor<T>>, axis: usize, split_sizes: Vec<usize>) -> Self {
        Self {
            inputs,
            axis,
            split_sizes,
        }
    }
}

impl<T: Float> GradFn<T> for CatBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // GPU fast path: use strided_split to extract each chunk directly on GPU.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let go_shape = grad_output.shape();
                let ndim = go_shape.len();
                let axis = self.axis;
                let total_along_axis = go_shape[axis];
                let inner: usize = if axis + 1 < ndim {
                    go_shape[axis + 1..].iter().product()
                } else {
                    1
                };
                let go_handle = grad_output.gpu_handle()?;
                let f64_path = is_f64::<T>();

                let mut result = Vec::with_capacity(self.inputs.len());
                let mut offset = 0usize;

                for (i, &split_size) in self.split_sizes.iter().enumerate() {
                    if !self.inputs[i].requires_grad() {
                        result.push(None);
                        offset += split_size;
                        continue;
                    }

                    let chunk_numel = self.inputs[i].numel();
                    let chunk_handle = if f64_path {
                        backend.strided_split_f64(
                            go_handle,
                            total_along_axis,
                            offset,
                            split_size,
                            inner,
                            chunk_numel,
                        )?
                    } else {
                        backend.strided_split_f32(
                            go_handle,
                            total_along_axis,
                            offset,
                            split_size,
                            inner,
                            chunk_numel,
                        )?
                    };

                    let grad_tensor = Tensor::from_storage(
                        TensorStorage::gpu(chunk_handle),
                        self.inputs[i].shape().to_vec(),
                        false,
                    )?;
                    result.push(Some(grad_tensor));
                    offset += split_size;
                }

                return Ok(result);
            }
        }

        // CPU path (also serves as fallback for non-f32 or missing backend).
        let (cpu_go, device) = ensure_cpu(grad_output)?;
        let grad_data = cpu_go.data()?;
        let out_shape = cpu_go.shape();
        let ndim = out_shape.len();
        let axis = self.axis;

        // Compute the product of dimensions before and after the cat axis.
        let outer: usize = out_shape[..axis].iter().product();
        let inner: usize = if axis + 1 < ndim {
            out_shape[axis + 1..].iter().product()
        } else {
            1
        };

        let mut result = Vec::with_capacity(self.inputs.len());
        let mut offset = 0usize;

        for (i, split_size) in self.split_sizes.iter().enumerate() {
            if !self.inputs[i].requires_grad() {
                result.push(None);
                offset += split_size * inner;
                continue;
            }

            let chunk_numel: usize = self.inputs[i].numel();
            let mut grad_chunk = vec![<T as num_traits::Zero>::zero(); chunk_numel];

            // Copy the appropriate slice from grad_output for each "outer" row.
            for o in 0..outer {
                let src_row_start = o * out_shape[axis] * inner + offset;
                let dst_row_start = o * split_size * inner;
                let row_len = split_size * inner;
                grad_chunk[dst_row_start..dst_row_start + row_len]
                    .copy_from_slice(&grad_data[src_row_start..src_row_start + row_len]);
            }

            let grad_tensor = Tensor::from_storage(
                TensorStorage::cpu(grad_chunk),
                self.inputs[i].shape().to_vec(),
                false,
            )?;
            result.push(Some(restore_device(grad_tensor, device)?));
            offset += split_size * inner;
        }

        Ok(result)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        self.inputs.iter().collect()
    }

    fn name(&self) -> &'static str {
        "CatBackward"
    }
}

// ---------------------------------------------------------------------------
// SplitBackward — backward for split/chunk
// ---------------------------------------------------------------------------

/// Backward for a single chunk produced by `split`.
///
/// Each chunk gets its own `SplitBackward`. The VJP zero-pads the incoming
/// gradient into the original tensor's shape at the correct offset along
/// the split dimension.
#[derive(Debug)]
pub struct SplitBackward<T: Float> {
    /// The original unsplit tensor (needed for shape and requires_grad).
    input: Tensor<T>,
    /// The split dimension.
    dim: usize,
    /// Offset of this chunk along `dim` in the original tensor.
    offset: usize,
    /// Size of this chunk along `dim`.
    chunk_size: usize,
}

impl<T: Float> SplitBackward<T> {
    pub fn new(input: Tensor<T>, dim: usize, offset: usize, chunk_size: usize) -> Self {
        Self {
            input,
            dim,
            offset,
            chunk_size,
        }
    }
}

impl<T: Float> GradFn<T> for SplitBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        // GPU fast path: allocate zeros on GPU, strided_cat the gradient into it.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let orig_shape = self.input.shape();
                let ndim = orig_shape.len();
                let inner: usize = if self.dim + 1 < ndim {
                    orig_shape[self.dim + 1..].iter().product()
                } else {
                    1
                };
                let total_along_dim = orig_shape[self.dim];
                let orig_numel: usize = orig_shape.iter().product();
                let device_ord = grad_output.gpu_handle()?.device_ordinal();
                let f64_path = is_f64::<T>();
                let elem_size = if f64_path { 8 } else { 4 };

                let mut zeros_handle = backend.alloc_zeros(orig_numel, elem_size, device_ord)?;

                let go_handle = grad_output.gpu_handle()?;
                let chunk_numel = grad_output.numel();
                if f64_path {
                    backend.strided_cat_f64(
                        go_handle,
                        &mut zeros_handle,
                        total_along_dim,
                        self.offset,
                        self.chunk_size,
                        inner,
                        chunk_numel,
                    )?;
                } else {
                    backend.strided_cat_f32(
                        go_handle,
                        &mut zeros_handle,
                        total_along_dim,
                        self.offset,
                        self.chunk_size,
                        inner,
                        chunk_numel,
                    )?;
                }

                let grad_tensor = Tensor::from_storage(
                    TensorStorage::gpu(zeros_handle),
                    orig_shape.to_vec(),
                    false,
                )?;
                return Ok(vec![Some(grad_tensor)]);
            }
        }

        // CPU path (also serves as fallback for non-f32 or missing backend).
        let (cpu_go, device) = ensure_cpu(grad_output)?;
        let grad_data = cpu_go.data()?;
        let orig_shape = self.input.shape();
        let ndim = orig_shape.len();

        let outer: usize = orig_shape[..self.dim].iter().product();
        let inner: usize = if self.dim + 1 < ndim {
            orig_shape[self.dim + 1..].iter().product()
        } else {
            1
        };
        let total_along_dim = orig_shape[self.dim];

        // Build a zero tensor with the original shape, then copy the gradient
        // into the correct slice.
        let orig_numel: usize = orig_shape.iter().product();
        let mut result = vec![<T as num_traits::Zero>::zero(); orig_numel];

        for o in 0..outer {
            let dst_start = o * total_along_dim * inner + self.offset * inner;
            let src_start = o * self.chunk_size * inner;
            let row_len = self.chunk_size * inner;
            result[dst_start..dst_start + row_len]
                .copy_from_slice(&grad_data[src_start..src_start + row_len]);
        }

        let grad_tensor =
            Tensor::from_storage(TensorStorage::cpu(result), orig_shape.to_vec(), false)?;
        Ok(vec![Some(restore_device(grad_tensor, device)?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SplitBackward"
    }
}

/// Concatenate tensors along an axis.
///
/// All tensors must have the same shape except along `axis`.
pub fn cat<T: Float>(tensors: &[Tensor<T>], axis: isize) -> FerrotorchResult<Tensor<T>> {
    if tensors.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "cat: empty tensor list".into(),
        });
    }

    let ndim = tensors[0].ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "cat: cannot concatenate scalar (0-D) tensors".into(),
        });
    }

    let norm_axis = crate::shape::normalize_axis(axis, ndim)?;

    // Validate shapes: all dims except `axis` must match.
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.ndim() != ndim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("cat: tensor {} has {} dims, expected {}", i, t.ndim(), ndim),
            });
        }
        for d in 0..ndim {
            if d != norm_axis && t.shape()[d] != tensors[0].shape()[d] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "cat: tensor {} has shape {:?}, incompatible with {:?} on axis {}",
                        i,
                        t.shape(),
                        tensors[0].shape(),
                        d
                    ),
                });
            }
        }
    }

    // Build output shape.
    let mut out_shape = tensors[0].shape().to_vec();
    let split_sizes: Vec<usize> = tensors.iter().map(|t| t.shape()[norm_axis]).collect();
    let total_along_axis: usize = split_sizes.iter().sum();
    out_shape[norm_axis] = total_along_axis;

    let device = tensors[0].device();

    // GPU fast path: allocate output on GPU, then strided_cat each input — no CPU
    // download needed.
    if device.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let inner: usize = if norm_axis + 1 < ndim {
                out_shape[norm_axis + 1..].iter().product()
            } else {
                1
            };
            let out_numel: usize = out_shape.iter().product();
            let device_ord = tensors[0].gpu_handle()?.device_ordinal();
            let f64_path = is_f64::<T>();
            let elem_size = if f64_path { 8 } else { 4 };

            let mut out_handle = backend.alloc_zeros(out_numel, elem_size, device_ord)?;

            let mut offset = 0usize;
            for t in tensors {
                let t_axis_size = t.shape()[norm_axis];
                let t_numel = t.numel();
                let t_handle = t.gpu_handle()?;
                if f64_path {
                    backend.strided_cat_f64(
                        t_handle,
                        &mut out_handle,
                        total_along_axis,
                        offset,
                        t_axis_size,
                        inner,
                        t_numel,
                    )?;
                } else {
                    backend.strided_cat_f32(
                        t_handle,
                        &mut out_handle,
                        total_along_axis,
                        offset,
                        t_axis_size,
                        inner,
                        t_numel,
                    )?;
                }
                offset += t_axis_size;
            }

            let any_requires_grad = tensors.iter().any(|t| t.requires_grad());
            let storage = TensorStorage::gpu(out_handle);

            return if is_grad_enabled() && any_requires_grad {
                let grad_fn = Arc::new(CatBackward::new(tensors.to_vec(), norm_axis, split_sizes));
                Tensor::from_operation(storage, out_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, out_shape, false)
            };
        }
    }

    // CPU path — non-f32 GPU tensors have no GPU kernel, error out.
    if device.is_cuda() {
        return Err(crate::error::FerrotorchError::NotImplementedOnCuda { op: "cat" });
    }
    let cpu_tensors: Vec<Tensor<T>> = tensors.to_vec();

    // Compute strides for the interleaved copy.
    let outer: usize = out_shape[..norm_axis].iter().product();
    let inner: usize = if norm_axis + 1 < ndim {
        out_shape[norm_axis + 1..].iter().product()
    } else {
        1
    };

    let out_numel: usize = out_shape.iter().product();
    let mut out_data = vec![<T as num_traits::Zero>::zero(); out_numel];

    let mut offset = 0usize;
    for t in &cpu_tensors {
        let t_data = t.data()?;
        let t_axis_size = t.shape()[norm_axis];
        for o in 0..outer {
            let src_start = o * t_axis_size * inner;
            let dst_start = o * total_along_axis * inner + offset;
            let row_len = t_axis_size * inner;
            out_data[dst_start..dst_start + row_len]
                .copy_from_slice(&t_data[src_start..src_start + row_len]);
        }
        offset += t_axis_size * inner;
    }

    let any_requires_grad = tensors.iter().any(|t| t.requires_grad());

    if is_grad_enabled() && any_requires_grad {
        let storage = if device.is_cuda() {
            let tmp = Tensor::from_storage(TensorStorage::cpu(out_data), out_shape.clone(), false)?;
            let gpu_tmp = tmp.to(device)?;
            gpu_tmp.into_storage_and_shape()?.0
        } else {
            TensorStorage::cpu(out_data)
        };
        let grad_fn = Arc::new(CatBackward::new(tensors.to_vec(), norm_axis, split_sizes));
        Tensor::from_operation(storage, out_shape, grad_fn)
    } else {
        let result = Tensor::from_storage(TensorStorage::cpu(out_data), out_shape, false)?;
        restore_device(result, device)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Resolve a shape specification that may contain exactly one `-1`.
///
/// Returns the fully-resolved `Vec<usize>`.
fn resolve_shape(shape: &[isize], numel: usize) -> FerrotorchResult<Vec<usize>> {
    let mut inferred_idx: Option<usize> = None;
    let mut product: usize = 1;

    for (i, &dim) in shape.iter().enumerate() {
        if dim == -1 {
            if inferred_idx.is_some() {
                return Err(FerrotorchError::InvalidArgument {
                    message: "reshape: only one dimension can be -1".into(),
                });
            }
            inferred_idx = Some(i);
        } else if dim < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("reshape: invalid dimension {dim}"),
            });
        } else {
            product *= dim as usize;
        }
    }

    let mut result: Vec<usize> = shape.iter().map(|&d| d as usize).collect();

    if let Some(idx) = inferred_idx {
        if product == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "reshape: cannot infer dimension with zero-size dimensions".into(),
            });
        }
        if numel % product != 0 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "reshape: cannot reshape tensor of {numel} elements into shape {shape:?}"
                ),
            });
        }
        result[idx] = numel / product;
    } else if product != numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "reshape: cannot reshape tensor of {numel} elements into shape {shape:?}"
            ),
        });
    }

    Ok(result)
}

/// Map a flat index in `out_shape` to a flat index in `in_shape` with
/// broadcasting (size-1 dims map to 0).
fn broadcast_flat_index(flat: usize, out_shape: &[usize], in_shape: &[usize]) -> usize {
    let out_ndim = out_shape.len();
    let in_ndim = in_shape.len();

    let mut in_flat = 0usize;
    let mut in_stride = 1usize;
    let mut out_stride = 1usize;

    for i in 0..in_ndim {
        let out_axis = out_ndim - 1 - i;
        let in_axis = in_ndim - 1 - i;

        let out_dim = out_shape[out_axis];
        let in_dim = in_shape[in_axis];

        let coord = (flat / out_stride) % out_dim;
        let in_coord = if in_dim == 1 { 0 } else { coord };

        in_flat += in_coord * in_stride;
        in_stride *= in_dim;
        out_stride *= out_dim;
    }

    in_flat
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;

    /// Helper: create a leaf tensor.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    /// A trivial SumBackward for testing: broadcasts ones back to input shape.
    #[derive(Debug)]
    struct SumBackward<T: Float> {
        input: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for SumBackward<T> {
        fn backward(&self, _grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let n = self.input.numel();
            let ones = vec![<T as num_traits::One>::one(); n];
            let g =
                Tensor::from_storage(TensorStorage::cpu(ones), self.input.shape().to_vec(), false)?;
            Ok(vec![Some(g)])
        }

        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.input]
        }

        fn name(&self) -> &'static str {
            "SumBackward"
        }
    }

    /// Helper: wrap a tensor in sum-to-scalar so backward() can be called.
    fn sum_to_scalar(t: &Tensor<f32>) -> Tensor<f32> {
        let data = t.data().unwrap();
        let total: f32 = data.iter().sum();
        Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward { input: t.clone() }),
        )
        .unwrap()
    }

    // -- reshape --

    #[test]
    fn test_reshape_forward() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = reshape(&x, &[3, 2]).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        assert_eq!(y.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_reshape_infer_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6], false);
        let y = reshape(&x, &[2, -1]).unwrap();
        assert_eq!(y.shape(), &[2, 3]);
    }

    #[test]
    fn test_reshape_backward() {
        // x: [2,3] -> reshape to [3,2] -> sum -> scalar -> backward
        // grad_output at reshape is ones([3,2]), backward produces ones([2,3]).
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let y = reshape(&x, &[3, 2]).unwrap();
        let loss = sum_to_scalar(&y);

        backward(&loss).unwrap();

        let grad = x.grad().unwrap().expect("x should have a gradient");
        assert_eq!(grad.shape(), &[2, 3]);
        for &v in grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6, "expected 1.0, got {v}");
        }
    }

    #[test]
    fn test_reshape_shape_mismatch() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        assert!(reshape(&x, &[2, 2]).is_err());
    }

    // -- flatten --

    #[test]
    fn test_flatten_forward() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let y = flatten(&x).unwrap();
        assert_eq!(y.shape(), &[4]);
        assert_eq!(y.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_flatten_backward() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let y = flatten(&x).unwrap();
        let loss = sum_to_scalar(&y);

        backward(&loss).unwrap();

        let grad = x.grad().unwrap().expect("x should have a gradient");
        assert_eq!(grad.shape(), &[2, 3]);
    }

    // -- squeeze / unsqueeze --

    #[test]
    fn test_squeeze_forward() {
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
        let y = squeeze(&x, 0).unwrap();
        assert_eq!(y.shape(), &[3]);
    }

    #[test]
    fn test_squeeze_non_one_error() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        assert!(squeeze(&x, 0).is_err());
    }

    #[test]
    fn test_unsqueeze_forward() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let y = unsqueeze(&x, 0).unwrap();
        assert_eq!(y.shape(), &[1, 3]);

        let z = unsqueeze(&x, -1).unwrap();
        assert_eq!(z.shape(), &[3, 1]);
    }

    #[test]
    fn test_squeeze_unsqueeze_roundtrip() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let y = unsqueeze(&x, 1).unwrap();
        assert_eq!(y.shape(), &[3, 1]);
        let z = squeeze(&y, 1).unwrap();
        assert_eq!(z.shape(), &[3]);
        assert_eq!(z.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    // -- transpose --

    #[test]
    fn test_transpose_2d_forward() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let y = transpose_2d(&x).unwrap();
        assert_eq!(y.shape(), &[3, 2]);
        // Transpose is now a zero-copy stride swap; use data_vec for logical order.
        assert_eq!(y.data_vec().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // -- cat --

    #[test]
    fn test_cat_forward_axis0() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let b = leaf(&[5.0, 6.0], &[1, 2], false);
        let c = cat(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[3, 2]);
        assert_eq!(c.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_cat_forward_axis1() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let b = leaf(&[5.0, 6.0], &[2, 1], false);
        let c = cat(&[a, b], 1).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        assert_eq!(c.data().unwrap(), &[1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn test_cat_backward_axis0() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = leaf(&[5.0, 6.0], &[1, 2], true);
        let c = cat(&[a.clone(), b.clone()], 0).unwrap();
        let loss = sum_to_scalar(&c);

        backward(&loss).unwrap();

        let a_grad = a.grad().unwrap().expect("a should have gradient");
        assert_eq!(a_grad.shape(), &[2, 2]);
        for &v in a_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }

        let b_grad = b.grad().unwrap().expect("b should have gradient");
        assert_eq!(b_grad.shape(), &[1, 2]);
        for &v in b_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cat_backward_axis1() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let b = leaf(&[5.0, 6.0], &[2, 1], true);
        let c = cat(&[a.clone(), b.clone()], 1).unwrap();
        let loss = sum_to_scalar(&c);

        backward(&loss).unwrap();

        let a_grad = a.grad().unwrap().expect("a should have gradient");
        assert_eq!(a_grad.shape(), &[2, 2]);
        for &v in a_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }

        let b_grad = b.grad().unwrap().expect("b should have gradient");
        assert_eq!(b_grad.shape(), &[2, 1]);
        for &v in b_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_cat_backward_mixed_requires_grad() {
        let a = leaf(&[1.0, 2.0], &[2], true);
        let b = leaf(&[3.0, 4.0], &[2], false);
        let c = cat(&[a.clone(), b.clone()], 0).unwrap();
        let loss = sum_to_scalar(&c);

        backward(&loss).unwrap();

        let a_grad = a.grad().unwrap().expect("a should have gradient");
        assert_eq!(a_grad.shape(), &[2]);
        for &v in a_grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6);
        }

        assert!(b.grad().unwrap().is_none());
    }

    #[test]
    fn test_cat_empty_error() {
        let result: FerrotorchResult<Tensor<f32>> = cat(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_cat_1d() {
        let a = leaf(&[1.0, 2.0], &[2], false);
        let b = leaf(&[3.0, 4.0, 5.0], &[3], false);
        let c = cat(&[a, b], 0).unwrap();
        assert_eq!(c.shape(), &[5]);
        assert_eq!(c.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    // -- no_grad --

    #[test]
    fn test_reshape_no_grad() {
        crate::autograd::no_grad(|| {
            let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);
            let y = reshape(&x, &[2, 2]).unwrap();
            assert!(y.grad_fn().is_none());
        });
    }

    // -- resolve_shape helper --

    #[test]
    fn test_resolve_shape_basic() {
        assert_eq!(resolve_shape(&[2, 3], 6).unwrap(), vec![2, 3]);
    }

    #[test]
    fn test_resolve_shape_infer() {
        assert_eq!(resolve_shape(&[2, -1], 6).unwrap(), vec![2, 3]);
        assert_eq!(resolve_shape(&[-1, 2], 6).unwrap(), vec![3, 2]);
        assert_eq!(resolve_shape(&[-1], 6).unwrap(), vec![6]);
    }

    #[test]
    fn test_resolve_shape_multiple_infer_error() {
        assert!(resolve_shape(&[-1, -1], 6).is_err());
    }

    #[test]
    fn test_resolve_shape_mismatch() {
        assert!(resolve_shape(&[2, 2], 6).is_err());
    }

    // -- graph preservation through shape ops --
    //
    // Shape ops must produce non-leaf tensors with grad_fn when the input
    // requires_grad. This is critical on GPU where `Tensor::to()` creates
    // detached leaf tensors — a `restore_device(from_operation(...))` pattern
    // would sever the graph.

    #[test]
    fn test_squeeze_preserves_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], true);
        let y = squeeze(&x, 0).unwrap();
        assert!(y.grad_fn().is_some(), "squeeze must attach a grad_fn");
        assert!(!y.is_leaf(), "squeeze output must be non-leaf");
        assert!(y.requires_grad(), "squeeze output must require grad");
    }

    #[test]
    fn test_unsqueeze_preserves_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let y = unsqueeze(&x, 0).unwrap();
        assert!(y.grad_fn().is_some(), "unsqueeze must attach a grad_fn");
        assert!(!y.is_leaf(), "unsqueeze output must be non-leaf");
        assert!(y.requires_grad(), "unsqueeze output must require grad");
    }

    #[test]
    fn test_flatten_preserves_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let y = flatten(&x).unwrap();
        assert!(y.grad_fn().is_some(), "flatten must attach a grad_fn");
        assert!(!y.is_leaf(), "flatten output must be non-leaf");
        assert!(y.requires_grad(), "flatten output must require grad");
    }

    #[test]
    fn test_squeeze_backward_reaches_leaf() {
        // Simulates the goodness_from_output pattern: x -> pow -> mm -> squeeze -> loss.
        // The squeeze backward must propagate gradients back to x.
        let x = leaf(&[1.0, 2.0, 3.0], &[3, 1], true);
        let squeezed = squeeze(&x, 1).unwrap();
        let loss = sum_to_scalar(&squeezed);

        backward(&loss).unwrap();

        let grad = x
            .grad()
            .unwrap()
            .expect("squeeze must propagate gradients to leaf input");
        assert_eq!(grad.shape(), &[3, 1]);
        for &v in grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6, "expected gradient 1.0, got {v}");
        }
    }

    #[test]
    fn test_unsqueeze_backward_reaches_leaf() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let unsqueezed = unsqueeze(&x, 1).unwrap();
        let loss = sum_to_scalar(&unsqueezed);

        backward(&loss).unwrap();

        let grad = x
            .grad()
            .unwrap()
            .expect("unsqueeze must propagate gradients to leaf input");
        assert_eq!(grad.shape(), &[3]);
        for &v in grad.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-6, "expected gradient 1.0, got {v}");
        }
    }

    #[test]
    fn test_squeeze_in_longer_chain() {
        // Mirrors the FF loss computation graph: leaf -> op -> squeeze -> op -> scalar.
        // Backward must reach the original leaf through the squeeze node.
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], true);

        // Multiply x by a constant (creates an intermediate with grad_fn).
        let two = leaf(&[2.0; 6], &[3, 2], false);
        let scaled = crate::grad_fns::arithmetic::mul(&x, &two).unwrap();

        // Sum columns via matmul with ones to get [3, 1].
        let ones = leaf(&[1.0, 1.0], &[2, 1], false);
        let row_sums = crate::grad_fns::linalg::mm_differentiable(&scaled, &ones).unwrap();

        // Squeeze to [3] — this is the operation that previously severed the graph on GPU.
        let squeezed = squeeze(&row_sums, 1).unwrap();
        assert!(squeezed.grad_fn().is_some(), "squeeze must preserve graph");

        let loss = sum_to_scalar(&squeezed);
        backward(&loss).unwrap();

        let grad = x
            .grad()
            .unwrap()
            .expect("backward through squeeze in a longer chain must reach leaf parameters");
        assert_eq!(grad.shape(), &[3, 2]);
        // d(loss)/d(x) = 2.0 * ones (from the scaling and sum).
        for &v in grad.data().unwrap() {
            assert!((v - 2.0).abs() < 1e-6, "expected gradient 2.0, got {v}");
        }
    }

    #[test]
    fn test_shape_ops_share_storage_with_input() {
        // view_operation must share storage — no data copy.
        // This catches the old ensure_cpu/restore_device pattern which
        // allocated new storage even on CPU.
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let flat = flatten(&x).unwrap();

        assert_eq!(flat.data().unwrap(), x.data().unwrap());
        assert_eq!(flat.shape(), &[6]);
        // Pointer equality: must be the same Arc, not a copy.
        assert!(
            flat.shares_storage(&x),
            "flatten should share storage with input (zero-copy)"
        );

        let orig = leaf(&[1.0, 2.0, 3.0], &[1, 3], true);
        let sq2 = squeeze(&orig, 0).unwrap();
        assert!(
            sq2.shares_storage(&orig),
            "squeeze should share storage with input (zero-copy)"
        );

        let orig3 = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let us = unsqueeze(&orig3, 0).unwrap();
        assert!(
            us.shares_storage(&orig3),
            "unsqueeze should share storage with input (zero-copy)"
        );
    }

    #[test]
    fn test_squeeze_no_grad_is_view() {
        // Without requires_grad, squeeze should be a cheap view_reshape.
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
        let y = squeeze(&x, 0).unwrap();
        assert!(y.grad_fn().is_none());
        assert_eq!(y.shape(), &[3]);
        assert_eq!(y.data().unwrap(), &[1.0, 2.0, 3.0]);
    }
}
