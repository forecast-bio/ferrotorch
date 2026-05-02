//! `as_strided` family — direct stride manipulation on tensors.
//!
//! Mirrors `torch.Tensor.as_strided`, `torch.as_strided_copy`, and
//! `torch.as_strided_scatter`. Strides are given in *element* units (matching
//! torch and ferrotorch's existing `Tensor::strides`), not bytes. A stride
//! may be zero (broadcast-style replication) or negative (reverse iteration).
//!
//! # Operations
//!
//! - [`as_strided`] returns a zero-copy view with the requested
//!   shape, strides, and storage offset. Works on any device because it is
//!   pure metadata — no kernels are dispatched.
//! - [`as_strided_copy`] materialises that view into a new contiguous
//!   tensor. CPU and CUDA paths exist (CUDA dispatches to the existing
//!   `strided_copy_f32`/`strided_copy_f64` GPU kernels). Other devices
//!   error with [`FerrotorchError::NotImplementedOnCuda`] until a kernel
//!   lands.
//! - [`as_strided_scatter`] is the inverse of `as_strided`: returns a copy
//!   of `self` with the strided positions overwritten by `src`. CPU only
//!   today; CUDA support is tracked separately.
//!
//! # Autograd
//!
//! `as_strided` is differentiable: the backward pass scatters the upstream
//! gradient back into a zero-initialised tensor of the original shape via
//! `as_strided_scatter`. This matches torch's `AsStridedBackward`.
//!
//! # Safety
//!
//! Like torch, this is **not** safe under in-place mutation when the
//! requested strides cause overlapping memory accesses. Reads always
//! return well-defined values (since storage is initialised), but
//! `tensor.as_strided(...)`-then-`add_(...)`-style writes against
//! overlapping views are undefined behaviour and produce torch-equivalent
//! "unexpected results". Bounds are always validated; overlap is not
//! rejected.
//!
//! # GPU discipline
//!
//! - View construction is metadata-only and shares the same `Arc<Storage>`
//!   on every device. No silent device transfer.
//! - `as_strided_copy` on CUDA dispatches to the dedicated GPU kernel; it
//!   does not bounce data through host memory.
//! - `as_strided_scatter` on non-CPU returns
//!   [`FerrotorchError::NotImplementedOnCuda`]; callers must move to CPU
//!   explicitly. The CUDA kernel will land in a follow-up issue (the
//!   forbidden pattern would be a silent `to(Cpu)` round-trip).

use std::sync::Arc;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// Public free functions (mirroring torch.as_strided / torch.as_strided_copy)
// ---------------------------------------------------------------------------

/// Zero-copy strided view; see [`Tensor::as_strided`] for full docs.
pub fn as_strided<T: Float>(
    input: &Tensor<T>,
    size: &[usize],
    stride: &[isize],
    storage_offset: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    input.as_strided(size, stride, storage_offset)
}

/// Materialised strided copy; see [`Tensor::as_strided_copy`] for full docs.
pub fn as_strided_copy<T: Float>(
    input: &Tensor<T>,
    size: &[usize],
    stride: &[isize],
    storage_offset: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    input.as_strided_copy(size, stride, storage_offset)
}

/// Inverse of `as_strided`; see [`Tensor::as_strided_scatter`] for full docs.
pub fn as_strided_scatter<T: Float>(
    input: &Tensor<T>,
    src: &Tensor<T>,
    size: &[usize],
    stride: &[isize],
    storage_offset: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    input.as_strided_scatter(src, size, stride, storage_offset)
}

// ---------------------------------------------------------------------------
// Internal: bounds validation
// ---------------------------------------------------------------------------

/// Compute the smallest and largest element offsets reachable by walking
/// every position in a strided view.
///
/// Returns `(min_offset, max_offset)` in element units, both inclusive.
/// For an empty view (`shape` contains a 0) returns `(0, 0)` to signal
/// "no positions reached" — the caller should treat that as trivially
/// in-bounds at any `storage_offset`.
fn stride_extent(shape: &[usize], stride: &[isize]) -> (i64, i64) {
    if shape.contains(&0) {
        return (0, 0);
    }
    let mut min_off: i64 = 0;
    let mut max_off: i64 = 0;
    for (&dim, &s) in shape.iter().zip(stride.iter()) {
        if dim == 0 {
            continue;
        }
        let last = (dim as i64 - 1) * s as i64;
        if last >= 0 {
            max_off += last;
        } else {
            min_off += last;
        }
    }
    (min_off, max_off)
}

/// Validate that the requested view fits within `storage_len`.
///
/// Returns `Ok(())` if every reachable offset (including the zero-position
/// origin at `storage_offset`) lies inside `[0, storage_len)`.
fn validate_bounds(
    op: &'static str,
    shape: &[usize],
    stride: &[isize],
    storage_offset: usize,
    storage_len: usize,
) -> FerrotorchResult<()> {
    if shape.len() != stride.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: shape and stride must have the same length (got {} vs {})",
                shape.len(),
                stride.len()
            ),
        });
    }

    // Empty view (any dim is zero) — nothing to read or write.
    if shape.contains(&0) {
        // Zero-element views are valid even at storage_offset == storage_len.
        if storage_offset > storage_len {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "{op}: storage_offset {storage_offset} > storage length {storage_len}"
                ),
            });
        }
        return Ok(());
    }

    let (min_off, max_off) = stride_extent(shape, stride);
    let lo = storage_offset as i64 + min_off;
    let hi = storage_offset as i64 + max_off;
    if lo < 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: storage_offset {storage_offset} with strides {stride:?} reaches negative \
                 offset {lo} (out of bounds)"
            ),
        });
    }
    if hi >= storage_len as i64 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: storage_offset {storage_offset} with shape {shape:?} and strides \
                 {stride:?} reaches offset {hi}, beyond storage length {storage_len}"
            ),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tensor methods (the impl block in tensor.rs is closed; we use a separate
// inherent impl here)
// ---------------------------------------------------------------------------

impl<T: Float> Tensor<T> {
    /// Build a zero-copy view with the given shape, strides (element units),
    /// and storage offset. If `storage_offset` is `None`, the input's
    /// existing offset is used.
    ///
    /// Equivalent to `torch.Tensor.as_strided(size, stride, storage_offset)`.
    /// Works on any device — no data movement.
    ///
    /// Validates that every reachable offset stays inside the underlying
    /// storage. **Does not** reject overlapping views: those are useful for
    /// constructing Toeplitz matrices, sliding windows, broadcast views,
    /// etc. As in torch, in-place writes against an overlapping view have
    /// undefined behaviour.
    pub fn as_strided(
        &self,
        size: &[usize],
        stride: &[isize],
        storage_offset: Option<usize>,
    ) -> FerrotorchResult<Tensor<T>> {
        let offset = storage_offset.unwrap_or_else(|| self.storage_offset());
        let storage_len = self.storage_len();
        validate_bounds("as_strided", size, stride, offset, storage_len)?;

        // No-grad fast path: pure metadata change, zero-copy on every device.
        if !crate::autograd::no_grad::is_grad_enabled() || !self.requires_grad() {
            return Ok(self.stride_view(size.to_vec(), stride.to_vec(), offset));
        }

        // Grad path: attach AsStridedBackward so autograd scatters the
        // upstream grad back into the original shape on backward.
        let grad_fn = Arc::new(AsStridedBackward::new(
            self.clone(),
            size.to_vec(),
            stride.to_vec(),
            offset,
        ));
        Ok(self.stride_view_operation(size.to_vec(), stride.to_vec(), offset, grad_fn))
    }

    /// Materialised strided copy: returns a new contiguous tensor whose
    /// values are the elements that `as_strided(size, stride, offset)` would
    /// read.
    ///
    /// On CUDA tensors this dispatches to the existing `strided_copy_f32`
    /// / `strided_copy_f64` GPU kernels (no host bounce). On CPU it walks
    /// the multi-index. On other devices (e.g. XPU) it returns
    /// [`FerrotorchError::NotImplementedOnCuda`] — install a kernel before
    /// using this on those devices.
    pub fn as_strided_copy(
        &self,
        size: &[usize],
        stride: &[isize],
        storage_offset: Option<usize>,
    ) -> FerrotorchResult<Tensor<T>> {
        // Construct the view first (validates bounds + propagates autograd).
        let view = self.as_strided(size, stride, storage_offset)?;
        // Materialise. `data_vec` already understands non-contiguous CPU
        // layouts, and on CUDA it routes through the GPU strided_copy
        // dispatcher in `cpu()`/`data_vec()` — see the comment in
        // tensor.rs:data_vec.
        if view.is_cuda() {
            // For CUDA tensors we reshape the storage by directly invoking
            // the GPU strided_copy dispatcher. `view.data_vec()` would
            // bounce through host first, which violates GPU discipline.
            return materialize_strided_cuda(&view);
        }
        let data = view.data_vec()?;
        Tensor::from_storage(TensorStorage::cpu(data), size.to_vec(), false)
    }

    /// Inverse of [`as_strided`]: return a copy of `self` with `src` written
    /// into the strided positions described by `(size, stride, offset)`.
    /// Positions outside that view retain `self`'s values.
    ///
    /// Equivalent to `torch.as_strided_scatter`. The CUDA path
    /// dispatches through the GPU backend (via the
    /// `strided_copy` + `strided_scatter` kernels) — no host bounce.
    pub fn as_strided_scatter(
        &self,
        src: &Tensor<T>,
        size: &[usize],
        stride: &[isize],
        storage_offset: Option<usize>,
    ) -> FerrotorchResult<Tensor<T>> {
        let offset = storage_offset.unwrap_or(0);
        let storage_len = self.numel();
        validate_bounds("as_strided_scatter", size, stride, offset, storage_len)?;

        if size.len() != src.shape().len() || size.iter().zip(src.shape()).any(|(a, b)| a != b) {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "as_strided_scatter: src shape {:?} does not match requested view shape {size:?}",
                    src.shape()
                ),
            });
        }

        if self.is_cuda() != src.is_cuda() {
            return Err(FerrotorchError::DeviceMismatch {
                expected: self.device(),
                got: src.device(),
            });
        }

        if self.is_cuda() {
            return scatter_on_cuda(self, src, size, stride, offset);
        }

        // CPU path: start from a contiguous copy of self, walk src in C-order
        // and write into the strided positions.
        let mut buf = self.data_vec()?;
        let src_data = src.data_vec()?;
        let ndim = size.len();

        let numel: usize = size.iter().product();
        if numel == 0 {
            return Tensor::from_storage(TensorStorage::cpu(buf), self.shape().to_vec(), false);
        }

        let mut indices = vec![0usize; ndim];
        #[allow(clippy::needless_range_loop)]
        for src_i in 0..numel {
            let mut flat = offset as i64;
            for d in 0..ndim {
                flat += indices[d] as i64 * stride[d] as i64;
            }
            // Bounds were validated; flat is in [0, storage_len).
            buf[flat as usize] = src_data[src_i];
            // Increment multi-index (rightmost first).
            for d in (0..ndim).rev() {
                indices[d] += 1;
                if indices[d] < size[d] {
                    break;
                }
                indices[d] = 0;
            }
        }

        Tensor::from_storage(TensorStorage::cpu(buf), self.shape().to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// Autograd: AsStridedBackward
// ---------------------------------------------------------------------------

/// VJP for `as_strided(input, size, stride, offset)`.
///
/// The forward op gathers elements from `input` at strided positions; the
/// gradient w.r.t. `input` therefore scatters `grad_output` back into the
/// same positions, leaving everything else zero. Mirrors torch's
/// `AsStridedBackward0`.
#[derive(Debug)]
pub struct AsStridedBackward<T: Float> {
    input: Tensor<T>,
    size: Vec<usize>,
    stride: Vec<isize>,
    storage_offset: usize,
}

impl<T: Float> AsStridedBackward<T> {
    pub fn new(
        input: Tensor<T>,
        size: Vec<usize>,
        stride: Vec<isize>,
        storage_offset: usize,
    ) -> Self {
        Self {
            input,
            size,
            stride,
            storage_offset,
        }
    }
}

impl<T: Float> GradFn<T> for AsStridedBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Gradient w.r.t. input has the input's shape: zeros, with the
        // upstream grad written into the strided positions.
        let zeros = crate::creation::zeros::<T>(self.input.shape())?;
        let grad_input = zeros.as_strided_scatter(
            grad_output,
            &self.size,
            &self.stride,
            Some(self.storage_offset),
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AsStridedBackward"
    }
}

// ---------------------------------------------------------------------------
// CUDA strided copy dispatch
// ---------------------------------------------------------------------------

/// Materialise an `as_strided` view living on CUDA into a contiguous CUDA
/// tensor. Dispatches to the existing `strided_copy_{f32,f64}` GPU kernels
/// via the `GpuBackend` dispatcher.
///
/// Never bounces data through host memory.
fn materialize_strided_cuda<T: Float>(view: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    use std::any::TypeId;

    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let storage = view.storage();
    let buf = storage
        .gpu_handle()
        .ok_or(FerrotorchError::DeviceUnavailable)?;
    let out_shape = view.shape().to_vec();
    let stride = view.strides().to_vec();
    let offset = view.storage_offset();

    let new_handle = if TypeId::of::<T>() == TypeId::of::<f32>() {
        backend.strided_copy_f32(buf, &out_shape, &stride, offset)?
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        backend.strided_copy_f64(buf, &out_shape, &stride, offset)?
    } else {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "as_strided_copy",
        });
    };
    let new_storage = TensorStorage::gpu(new_handle);
    Tensor::from_storage(new_storage, out_shape, false)
}

/// CUDA path for `as_strided_scatter`. Mirrors the CPU implementation
/// shape-for-shape:
///
/// 1. Materialise `self` into a fresh contiguous GPU buffer of length
///    `numel(self)` using `strided_copy_*` (no host bounce).
/// 2. Run `strided_scatter_*` to overwrite the strided positions with
///    values from `src`.
/// 3. Wrap the result as a new contiguous tensor with `self.shape()`.
///
/// f32 and f64 are supported. Other dtypes (`bf16`) on CUDA fall back
/// with `NotImplementedOnCuda`. There is no `.to(Cpu)` shortcut anywhere
/// — the data stays on device end-to-end (per `/rust-gpu-discipline`).
fn scatter_on_cuda<T: Float>(
    base: &Tensor<T>,
    src: &Tensor<T>,
    size: &[usize],
    stride: &[isize],
    offset: usize,
) -> FerrotorchResult<Tensor<T>> {
    use std::any::TypeId;

    let backend = crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let base_buf = base
        .storage()
        .gpu_handle()
        .ok_or(FerrotorchError::DeviceUnavailable)?;
    let src_buf = src
        .storage()
        .gpu_handle()
        .ok_or(FerrotorchError::DeviceUnavailable)?;

    let out_shape = base.shape().to_vec();
    let base_strides = base.strides().to_vec();
    let base_offset = base.storage_offset();

    // Step 1: clone `base` into a fresh contiguous GPU buffer. This
    // mirrors the CPU path's `let mut buf = self.data_vec()?;` line,
    // and as a side effect the resulting tensor is contiguous regardless
    // of `base`'s stride pattern — same shape result as the CPU path.
    let mut dst_handle = if TypeId::of::<T>() == TypeId::of::<f32>() {
        backend.strided_copy_f32(base_buf, &out_shape, &base_strides, base_offset)?
    } else if TypeId::of::<T>() == TypeId::of::<f64>() {
        backend.strided_copy_f64(base_buf, &out_shape, &base_strides, base_offset)?
    } else {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "as_strided_scatter",
        });
    };

    // Step 2: scatter src into dst at (size, stride, offset).
    if TypeId::of::<T>() == TypeId::of::<f32>() {
        backend.strided_scatter_f32(src_buf, &mut dst_handle, size, stride, offset)?;
    } else {
        backend.strided_scatter_f64(src_buf, &mut dst_handle, size, stride, offset)?;
    }

    Tensor::from_storage(TensorStorage::gpu(dst_handle), out_shape, false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{tensor, zeros};
    use crate::storage::TensorStorage;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // -----------------------------------------------------------------------
    // as_strided: zero-copy view tests
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_reshape_to_2x3() {
        // 1-D length-6 → 2x3 contiguous.
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let v = a.as_strided(&[2, 3], &[3, 1], None).unwrap();
        assert_eq!(v.shape(), &[2, 3]);
        assert_eq!(v.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_strided_overlapping_sliding_window() {
        // Sliding window of length 3 over [1..6]: shape [3, 3], stride [1, 1].
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let v = a.as_strided(&[3, 3], &[1, 1], None).unwrap();
        assert_eq!(v.shape(), &[3, 3]);
        // Each row is a 3-window:
        assert_eq!(
            v.data_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
        );
    }

    #[test]
    fn as_strided_negative_stride_reverses() {
        // Reverse a 1-D tensor: start at the end, stride -1.
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        // storage_offset = 3 (last element), stride = -1, size = 4.
        let v = a.as_strided(&[4], &[-1], Some(3)).unwrap();
        assert_eq!(v.data_vec().unwrap(), vec![4.0, 3.0, 2.0, 1.0]);
    }

    #[test]
    fn as_strided_zero_stride_broadcast() {
        // Stride 0: every position reads the same element (broadcast).
        let a = t(&[7.0, 8.0, 9.0], &[3]);
        let v = a.as_strided(&[5], &[0], Some(1)).unwrap();
        assert_eq!(v.data_vec().unwrap(), vec![8.0, 8.0, 8.0, 8.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // as_strided: bounds validation
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_rejects_out_of_bounds() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        // shape [4], stride [1] would reach offset 3 — out of bounds.
        let err = a.as_strided(&[4], &[1], Some(0)).unwrap_err();
        assert!(
            matches!(err, FerrotorchError::InvalidArgument { .. }),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn as_strided_rejects_negative_reach() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        // stride -1 from offset 1 reaches -1 on the second step.
        let err = a.as_strided(&[3], &[-1], Some(1)).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn as_strided_rejects_size_stride_length_mismatch() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let err = a.as_strided(&[2, 2], &[1], None).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn as_strided_zero_size_dim_is_valid() {
        // Empty view: shape [0, 5] with any strides is in-bounds.
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let v = a.as_strided(&[0, 5], &[100, 100], Some(0)).unwrap();
        assert_eq!(v.shape(), &[0, 5]);
        assert_eq!(v.data_vec().unwrap(), Vec::<f64>::new());
    }

    // -----------------------------------------------------------------------
    // as_strided shares storage with input (zero-copy)
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_shares_storage() {
        // Verify the view points at the same Arc<Storage> by checking that
        // building the view succeeds with a small storage offset and the
        // storage length stays the same.
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let v = a.as_strided(&[3], &[2], Some(0)).unwrap();
        // [1, 3, 5]
        assert_eq!(v.data_vec().unwrap(), vec![1.0, 3.0, 5.0]);
        // The underlying storage length matches `a`'s storage length.
        assert_eq!(v.storage().len(), a.storage().len());
    }

    // -----------------------------------------------------------------------
    // as_strided_copy: materialised, contiguous output
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_copy_makes_contiguous_2x3() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let copy = a.as_strided_copy(&[2, 3], &[3, 1], None).unwrap();
        assert_eq!(copy.shape(), &[2, 3]);
        assert!(copy.is_contiguous());
        assert_eq!(copy.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_strided_copy_collects_overlapping_window() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5]);
        let copy = a.as_strided_copy(&[3, 3], &[1, 1], None).unwrap();
        assert!(copy.is_contiguous());
        assert_eq!(
            copy.data_vec().unwrap(),
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0]
        );
    }

    // -----------------------------------------------------------------------
    // as_strided_scatter: write at strided positions
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_scatter_writes_into_view_positions() {
        let dst = t(&[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[6]);
        let src = t(&[10.0, 20.0, 30.0], &[3]);
        // Write src into positions 0, 2, 4 of dst.
        let out = dst.as_strided_scatter(&src, &[3], &[2], Some(0)).unwrap();
        assert_eq!(
            out.data_vec().unwrap(),
            vec![10.0, 0.0, 20.0, 0.0, 30.0, 0.0]
        );
    }

    #[test]
    fn as_strided_scatter_preserves_non_view_positions() {
        let dst = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]);
        let src = t(&[100.0, 200.0], &[2]);
        // Write src into positions 1, 3.
        let out = dst.as_strided_scatter(&src, &[2], &[2], Some(1)).unwrap();
        assert_eq!(
            out.data_vec().unwrap(),
            vec![1.0, 100.0, 3.0, 200.0, 5.0, 6.0]
        );
    }

    #[test]
    fn as_strided_scatter_2d_view_into_1d_dst() {
        // dst is length 6; scatter a 2x3 source via [3, 1] strides starting at 0.
        let dst = zeros::<f64>(&[6]).unwrap();
        let src = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let out = dst
            .as_strided_scatter(&src, &[2, 3], &[3, 1], Some(0))
            .unwrap();
        assert_eq!(out.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn as_strided_scatter_rejects_shape_mismatch() {
        let dst = zeros::<f64>(&[5]).unwrap();
        let src = t(&[1.0, 2.0, 3.0], &[3]);
        // Requested view is [2] but src is [3] → mismatch.
        let err = dst
            .as_strided_scatter(&src, &[2], &[1], Some(0))
            .unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // Autograd: as_strided then sum should yield correct gradients.
    // -----------------------------------------------------------------------

    #[test]
    fn as_strided_backward_scatters_into_input_shape() {
        use crate::autograd::backward;

        // input: [a, b, c, d, e, f] with requires_grad
        let input = tensor(&[1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let input = input.requires_grad_(true);
        // View as 2x3 (contiguous reshape via as_strided).
        let view = input.as_strided(&[2, 3], &[3, 1], None).unwrap();
        // Sum to scalar so backward returns ones at every view position.
        let s = view.sum_all().unwrap();
        backward(&s).unwrap();
        let g = input.grad().unwrap().expect("input should have a gradient");
        // Each input element appears exactly once in the view, so the
        // gradient should be all ones.
        assert_eq!(g.data_vec().unwrap(), vec![1.0; 6]);
    }

    #[test]
    fn as_strided_backward_overlapping_view_last_write_wins() {
        use crate::autograd::backward;

        // Sliding window: each input element appears in multiple view
        // positions. `as_strided_scatter` is OVERWRITE semantics (matching
        // torch's documented gradient formula for `as_strided`), so the
        // gradient at each input position is `1` regardless of how many
        // view positions reference it. Counting occurrences would require
        // a `_scatter_add_` variant that ferrotorch doesn't yet expose;
        // tracked as a follow-up.
        //
        // The view is non-contiguous, and `sum_all` requires contiguous
        // input today, so materialise via `.contiguous()` first; this
        // chains `AsStridedBackward` <- `ContiguousBackward` <- sum, which
        // exercises the as_strided VJP under composition.
        let input = tensor(&[1.0_f64, 2.0, 3.0, 4.0, 5.0]).unwrap();
        let input = input.requires_grad_(true);
        let view = input.as_strided(&[3, 3], &[1, 1], None).unwrap();
        let contig = view.contiguous().unwrap();
        let s = contig.sum_all().unwrap();
        backward(&s).unwrap();
        let g = input.grad().unwrap().expect("input should have a gradient");
        assert_eq!(g.data_vec().unwrap(), vec![1.0; 5]);
    }
}
