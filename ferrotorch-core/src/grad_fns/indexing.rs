//! Backward functions for indexing operations.
//!
//! Implements `GradFn` for:
//! - `index_select` (1D) — selects elements along an axis by integer indices
//! - `masked_fill` — fills elements where a boolean mask is true
//! - `gather` — gathers elements along an axis (N-D)
//! - `scatter` — scatters src values into input along an axis
//! - `scatter_add` — scatter with addition
//! - `where_cond` — ternary selection

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::device::Device;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::gpu_backend;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

use crate::bool_tensor::BoolTensor;
use crate::int_tensor::{IntElement, IntTensor};

/// Upload a CPU `&[f32]` slice to a GPU buffer on the given device ordinal.
fn upload_f32_to_gpu(
    data: &[f32],
    ordinal: usize,
) -> FerrotorchResult<crate::gpu_dispatch::GpuBufferHandle> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    // SAFETY: `data: &[f32]` is borrowed for the duration of this function
    // and is fully initialized (f32 has no padding, no niches). Reading its
    // bytes as &[u8] of length `data.len() * 4` (== `data.len() *
    // size_of::<f32>()`) is sound and matches the actual byte size of the
    // underlying allocation; the resulting slice does not outlive `data`.
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr().cast::<u8>(), data.len() * 4) };
    backend.cpu_to_gpu(bytes, 4, ordinal)
}

/// For `ScatterBackward` grad_input: build a flat boolean mask (1.0 at positions
/// overwritten by scatter, 0.0 elsewhere) in the input's flat space.
fn scatter_write_mask(
    index: &[usize],
    index_shape: &[usize],
    input_shape: &[usize],
    dim: usize,
) -> Vec<f32> {
    let input_numel: usize = input_shape.iter().product();
    let index_numel: usize = index_shape.iter().product();
    let mut mask = vec![0.0f32; input_numel];
    let ndim = input_shape.len();
    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        let dst_flat = flat_index(&dst_coords, input_shape);
        mask[dst_flat] = 1.0;
        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }
    mask
}

/// For `GatherBackward`: compute flat destination indices (into input space)
/// for each element of the index tensor — i.e. the same flat positions that
/// `gather` read from, so scatter-add routes gradients back there.
fn gather_dst_flat_indices(
    index: &[usize],
    index_shape: &[usize],
    input_shape: &[usize],
    dim: usize,
) -> Vec<f32> {
    let ndim = input_shape.len();
    let index_numel: usize = index_shape.iter().product();
    let mut result = Vec::with_capacity(index_numel);
    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        // The destination in input space: same coords as the index position
        // but with `dim` replaced by idx_val.
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        result.push(flat_index(&dst_coords, input_shape) as f32);
        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }
    result
}

/// For scatter/scatter_add backward grad_src: the source gradient comes from
/// gathering grad_output at the index-mapped positions in input space — the
/// inverse of what scatter wrote. Returns flat indices into grad_output space.
fn scatter_src_flat_indices(
    index: &[usize],
    index_shape: &[usize],
    input_shape: &[usize],
    dim: usize,
) -> Vec<f32> {
    // Same computation as gather_dst_flat_indices: for each position in the
    // index tensor, the source flat index in grad_output (= input) is the same
    // flat location that was overwritten during scatter.
    gather_dst_flat_indices(index, index_shape, input_shape, dim)
}

// ---------------------------------------------------------------------------
// Helpers for N-D backward (shared by gather/scatter/scatter_add)
// ---------------------------------------------------------------------------

/// Compute the flat index into a C-contiguous buffer from per-axis coordinates.
#[inline]
fn flat_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for d in (0..shape.len()).rev() {
        idx += coords[d] * stride;
        stride *= shape[d];
    }
    idx
}

/// Increment a multi-dimensional coordinate vector in C-order (last axis
/// fastest). Returns `false` when the coordinate wraps past the last element.
#[inline]
fn increment_coords(coords: &mut [usize], shape: &[usize]) -> bool {
    for d in (0..shape.len()).rev() {
        coords[d] += 1;
        if coords[d] < shape[d] {
            return true;
        }
        coords[d] = 0;
    }
    false
}

// ---------------------------------------------------------------------------
// index_select (1D)
// ---------------------------------------------------------------------------

/// Backward function for `index_select` on a 1-D input tensor.
///
/// Forward: `output[i] = input[indices[i]]`
///
/// VJP: `grad_input = zeros(input.len()); for (i, idx) in indices: grad_input[idx] += grad_output[i]`
///
/// This is equivalent to a scatter-add of `grad_output` back into the input shape.
#[derive(Debug)]
pub struct IndexSelectBackward<T: Float> {
    /// The original input tensor (saved for shape information).
    pub input: Tensor<T>,
    /// The index vector used during the forward pass.
    pub indices: Vec<usize>,
}

impl<T: Float> GradFn<T> for IndexSelectBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        let input_len = self.input.numel();

        if grad_output.is_cuda() {
            // GPU path: scatter-add via GPU kernel.
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let indices_f32: Vec<f32> = self.indices.iter().map(|&i| i as f32).collect();
            let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;
            let result_handle =
                backend.scatter_add_1d_f32(grad_output.gpu_handle()?, &idx_handle, input_len)?;
            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(result_handle),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(grad_tensor)])
        } else {
            // CPU path: direct scatter-add.
            let go_data = grad_output.data()?;
            let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_len];
            for (i, &idx) in self.indices.iter().enumerate() {
                grad_input[idx] += go_data[i];
            }
            let grad_tensor = Tensor::from_storage(
                TensorStorage::cpu(grad_input),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(grad_tensor)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IndexSelectBackward"
    }
}

/// Perform 1-D `index_select`: gather elements from `input` at `indices`.
///
/// Returns a new tensor of the same dtype with shape `[indices.len()]`.
/// If `input.requires_grad()` and grad is enabled, the result tensor
/// carries an `IndexSelectBackward` grad_fn.
pub fn index_select_1d<T: Float>(
    input: &Tensor<T>,
    indices: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    // Validate: input must be 1-D.
    if input.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "index_select_1d requires a 1-D input, got shape {:?}",
                input.shape()
            ),
        });
    }

    let input_len = input.shape()[0];

    // Validate all indices are in bounds (shape is CPU metadata).
    for &idx in indices {
        if idx >= input_len {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: idx,
                axis: 0,
                size: input_len,
            });
        }
    }

    let output_shape = vec![indices.len()];

    if input.is_cuda() {
        // GPU path: gather via GPU kernel.
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let ordinal = match input.device() {
            Device::Cuda(o) => o,
            _ => unreachable!(),
        };
        let indices_f32: Vec<f32> = indices.iter().map(|&i| i as f32).collect();
        let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;
        let result_handle = backend.index_select_1d_f32(input.gpu_handle()?, &idx_handle)?;
        let storage = TensorStorage::gpu(result_handle);

        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(IndexSelectBackward {
                input: input.clone(),
                indices: indices.to_vec(),
            });
            Tensor::from_operation(storage, output_shape, grad_fn)
        } else {
            Tensor::from_storage(storage, output_shape, false)
        }
    } else {
        // CPU path: direct gather.
        let input_data = input.data()?;
        let output_data: Vec<T> = indices.iter().map(|&idx| input_data[idx]).collect();

        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(IndexSelectBackward {
                input: input.clone(),
                indices: indices.to_vec(),
            });
            Tensor::from_operation(TensorStorage::cpu(output_data), output_shape, grad_fn)
        } else {
            Tensor::from_storage(TensorStorage::cpu(output_data), output_shape, false)
        }
    }
}

// ---------------------------------------------------------------------------
// masked_fill
// ---------------------------------------------------------------------------

/// Backward function for `masked_fill`.
///
/// Forward: `output[i] = if mask[i] { value } else { input[i] }`
///
/// VJP: `grad_input[i] = if mask[i] { 0 } else { grad_output[i] }`
///
/// The gradient is zeroed at every position where the mask was true, because
/// those positions were replaced by a constant and no longer depend on the input.
///
/// The mask is stored as a flat `Vec<bool>` for GPU reconstruction.
#[derive(Debug)]
pub struct MaskedFillBackward<T: Float> {
    /// The original input tensor (saved for shape).
    pub input: Tensor<T>,
    /// The full boolean mask from the forward pass.
    pub mask: Vec<bool>,
}

impl<T: Float> GradFn<T> for MaskedFillBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        if grad_output.is_cuda() {
            // GPU path: masked-zero via GPU kernel.
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let mask_f32: Vec<f32> = self
                .mask
                .iter()
                .map(|&m| if m { 1.0 } else { 0.0 })
                .collect();
            let mask_handle = upload_f32_to_gpu(&mask_f32, ordinal)?;
            let result_handle = backend.masked_zero_f32(grad_output.gpu_handle()?, &mask_handle)?;
            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(result_handle),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(grad_tensor)])
        } else {
            // CPU path: direct mask zeroing.
            let go_data = grad_output.data()?;
            let mut grad_input: Vec<T> = go_data.to_vec();
            for (i, &m) in self.mask.iter().enumerate() {
                if m {
                    grad_input[i] = <T as num_traits::Zero>::zero();
                }
            }
            let grad_tensor = Tensor::from_storage(
                TensorStorage::cpu(grad_input),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(grad_tensor)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MaskedFillBackward"
    }
}

/// Fill elements of `input` with `value` where `mask` is `true`.
///
/// `mask` is a boolean slice with the same number of elements as `input`
/// (flat layout). Returns a new tensor; the original is not mutated.
///
/// If `input.requires_grad()` and grad is enabled, the result carries a
/// `MaskedFillBackward` grad_fn.
pub fn masked_fill<T: Float>(
    input: &Tensor<T>,
    mask: &[bool],
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    let input_len = input.numel();

    if mask.len() != input_len {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "masked_fill: mask length {} does not match input length {}",
                mask.len(),
                input_len
            ),
        });
    }

    let output_shape = input.shape().to_vec();

    if input.is_cuda() {
        // GPU path: masked-fill via GPU kernel.
        let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let ordinal = match input.device() {
            Device::Cuda(o) => o,
            _ => unreachable!(),
        };
        let mask_f32: Vec<f32> = mask.iter().map(|&m| if m { 1.0 } else { 0.0 }).collect();
        let mask_handle = upload_f32_to_gpu(&mask_f32, ordinal)?;
        // value must be f32 for the GPU kernel.
        let value_f32: f32 = num_traits::ToPrimitive::to_f32(&value).unwrap_or(0.0);
        let result_handle =
            backend.masked_fill_f32(input.gpu_handle()?, &mask_handle, value_f32)?;
        let storage = TensorStorage::gpu(result_handle);

        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(MaskedFillBackward {
                input: input.clone(),
                mask: mask.to_vec(),
            });
            Tensor::from_operation(storage, output_shape, grad_fn)
        } else {
            Tensor::from_storage(storage, output_shape, false)
        }
    } else {
        // CPU path: direct masked fill.
        let input_data = input.data()?;
        let output_data: Vec<T> = input_data
            .iter()
            .zip(mask.iter())
            .map(|(&x, &m)| if m { value } else { x })
            .collect();

        if input.requires_grad() && is_grad_enabled() {
            let grad_fn = Arc::new(MaskedFillBackward {
                input: input.clone(),
                mask: mask.to_vec(),
            });
            Tensor::from_operation(TensorStorage::cpu(output_data), output_shape, grad_fn)
        } else {
            Tensor::from_storage(TensorStorage::cpu(output_data), output_shape, false)
        }
    }
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

/// Backward function for N-D `gather`.
///
/// Forward: `output[coords] = input[coords with dim replaced by index[coords]]`
///
/// VJP: scatter-add `grad_output` back into zeros of input shape along `dim`
/// using the same `index`.
#[derive(Debug)]
pub struct GatherBackward<T: Float> {
    /// The original input tensor (saved for shape).
    pub input: Tensor<T>,
    /// The dimension along which gathering was performed.
    pub dim: usize,
    /// The flat index data used during the forward pass.
    pub index: Vec<usize>,
    /// The shape of the index tensor.
    pub index_shape: Vec<usize>,
}

impl<T: Float> GradFn<T> for GatherBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }

        let input_shape = self.input.shape();
        let input_numel: usize = input_shape.iter().product();

        // §3 GPU-native path: flatten grad_output, compute flat dst indices CPU-side
        // (the index tensor is always CPU-resident), scatter-add via existing 1-D kernel.
        if grad_output.is_cuda() {
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let dst_indices =
                gather_dst_flat_indices(&self.index, &self.index_shape, input_shape, self.dim);
            let idx_handle = upload_f32_to_gpu(&dst_indices, ordinal)?;
            // scatter_add_1d_f32 treats grad_output as a flat 1-D buffer and
            // accumulates each element at its flat destination index.
            let result_handle =
                backend.scatter_add_1d_f32(grad_output.gpu_handle()?, &idx_handle, input_numel)?;
            let grad_tensor = Tensor::from_storage(
                TensorStorage::gpu(result_handle),
                input_shape.to_vec(),
                false,
            )?;
            return Ok(vec![Some(grad_tensor)]);
        }

        let go_data = grad_output.data_vec()?;
        let ndim = input_shape.len();
        let index_numel: usize = self.index_shape.iter().product();

        let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_numel];

        // Scatter-add grad_output into grad_input using the saved index and dim.
        let mut coords = vec![0usize; ndim];
        for (i, &go_val) in go_data.iter().enumerate().take(index_numel) {
            let idx_val = self.index[i];
            let mut dst_coords = coords.clone();
            dst_coords[self.dim] = idx_val;
            let dst_flat = flat_index(&dst_coords, input_shape);
            grad_input[dst_flat] += go_val;

            if i + 1 < index_numel {
                increment_coords(&mut coords, &self.index_shape);
            }
        }

        let grad_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_input), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "GatherBackward"
    }
}

// ---------------------------------------------------------------------------
// scatter
// ---------------------------------------------------------------------------

/// Backward function for N-D `scatter`.
///
/// Forward: `output = input.clone(); output[index-mapped coords] = src[coords]`
///
/// VJP for input: `grad_input = grad_output` with scattered positions zeroed out
/// (those positions came from src, not input).
///
/// VJP for src: `grad_src[coords] = grad_output[index-mapped coords]` (gather).
#[derive(Debug)]
pub struct ScatterBackward<T: Float> {
    /// The original input tensor.
    pub input: Tensor<T>,
    /// The source tensor scattered into input.
    pub src: Tensor<T>,
    /// The dimension along which scattering was performed.
    pub dim: usize,
    /// The flat index data.
    pub index: Vec<usize>,
    /// The shape of the index tensor.
    pub index_shape: Vec<usize>,
}

impl<T: Float> GradFn<T> for ScatterBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }

        let input_shape = self.input.shape();
        let index_numel: usize = self.index_shape.iter().product();

        // §3 GPU-native path:
        //   grad_input = masked_zero_f32(grad_output, write_mask)
        //     — zeros at every position scatter wrote to (those positions came from src).
        //   grad_src   = index_select_1d_f32(flat(grad_output), scatter_src_indices)
        //     — gathers from the flat positions that scatter wrote into.
        if grad_output.is_cuda() {
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

            let grad_input = if self.input.requires_grad() {
                // Build a 1.0/0.0 mask for the written positions, upload, zero them out.
                let mask_f32 =
                    scatter_write_mask(&self.index, &self.index_shape, input_shape, self.dim);
                let mask_handle = upload_f32_to_gpu(&mask_f32, ordinal)?;
                let result_h = backend.masked_zero_f32(grad_output.gpu_handle()?, &mask_handle)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    input_shape.to_vec(),
                    false,
                )?)
            } else {
                None
            };

            let grad_src = if self.src.requires_grad() {
                // Gather grad_output at the flat positions that scatter wrote into.
                let src_indices =
                    scatter_src_flat_indices(&self.index, &self.index_shape, input_shape, self.dim);
                let idx_handle = upload_f32_to_gpu(&src_indices, ordinal)?;
                let result_h =
                    backend.index_select_1d_f32(grad_output.gpu_handle()?, &idx_handle)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.index_shape.clone(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_input, grad_src]);
        }

        let ndim = input_shape.len();
        let go_data = grad_output.data_vec()?;

        // grad for input: grad_output with scattered positions zeroed.
        let grad_input = if self.input.requires_grad() {
            let mut gi = go_data.clone();
            let mut coords = vec![0usize; ndim];
            for i in 0..index_numel {
                let idx_val = self.index[i];
                let mut dst_coords = coords.clone();
                dst_coords[self.dim] = idx_val;
                let dst_flat = flat_index(&dst_coords, input_shape);
                gi[dst_flat] = <T as num_traits::Zero>::zero();

                if i + 1 < index_numel {
                    increment_coords(&mut coords, &self.index_shape);
                }
            }
            let t = Tensor::from_storage(TensorStorage::cpu(gi), input_shape.to_vec(), false)?;
            Some(t)
        } else {
            None
        };

        // grad for src: gather from grad_output at index positions.
        let grad_src = if self.src.requires_grad() {
            let mut gs = vec![<T as num_traits::Zero>::zero(); index_numel];
            let mut coords = vec![0usize; ndim];
            for (i, gs_elem) in gs.iter_mut().enumerate() {
                let idx_val = self.index[i];
                let mut src_coords = coords.clone();
                src_coords[self.dim] = idx_val;
                let src_flat = flat_index(&src_coords, input_shape);
                *gs_elem = go_data[src_flat];

                if i + 1 < index_numel {
                    increment_coords(&mut coords, &self.index_shape);
                }
            }
            let t = Tensor::from_storage(TensorStorage::cpu(gs), self.index_shape.clone(), false)?;
            Some(t)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.src]
    }

    fn name(&self) -> &'static str {
        "ScatterBackward"
    }
}

// ---------------------------------------------------------------------------
// scatter_add
// ---------------------------------------------------------------------------

/// Backward function for N-D `scatter_add`.
///
/// Forward: `output = input.clone(); output[index-mapped coords] += src[coords]`
///
/// VJP for input: `grad_input = grad_output` (identity — addition passes
/// gradient through unchanged).
///
/// VJP for src: `grad_src[coords] = grad_output[index-mapped coords]` (gather).
#[derive(Debug)]
pub struct ScatterAddBackward<T: Float> {
    /// The original input tensor.
    pub input: Tensor<T>,
    /// The source tensor that was scatter-added.
    pub src: Tensor<T>,
    /// The dimension along which scatter_add was performed.
    pub dim: usize,
    /// The flat index data.
    pub index: Vec<usize>,
    /// The shape of the index tensor.
    pub index_shape: Vec<usize>,
}

impl<T: Float> GradFn<T> for ScatterAddBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }

        let input_shape = self.input.shape();
        let index_numel: usize = self.index_shape.iter().product();

        // §3 GPU-native path:
        //   grad_input = grad_output  (identity — addition passes grad through unchanged).
        //   grad_src   = index_select_1d_f32(flat(grad_output), scatter_src_indices)
        //     — gathers the positions that scatter_add accumulated into.
        if grad_output.is_cuda() {
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

            let grad_input = if self.input.requires_grad() {
                // Identity: grad_input is an on-device copy of grad_output.
                let cloned_h = backend.clone_buffer(grad_output.gpu_handle()?)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(cloned_h),
                    input_shape.to_vec(),
                    false,
                )?)
            } else {
                None
            };

            let grad_src = if self.src.requires_grad() {
                let src_indices =
                    scatter_src_flat_indices(&self.index, &self.index_shape, input_shape, self.dim);
                let idx_handle = upload_f32_to_gpu(&src_indices, ordinal)?;
                let result_h =
                    backend.index_select_1d_f32(grad_output.gpu_handle()?, &idx_handle)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.index_shape.clone(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_input, grad_src]);
        }

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "scatter_add backward",
            });
        }

        let ndim = input_shape.len();
        let go_data = grad_output.data_vec()?;

        // grad for input: identity (pass grad_output through).
        let grad_input = if self.input.requires_grad() {
            let t = Tensor::from_storage(
                TensorStorage::cpu(go_data.clone()),
                input_shape.to_vec(),
                false,
            )?;
            Some(t)
        } else {
            None
        };

        // grad for src: gather from grad_output at index positions.
        let grad_src = if self.src.requires_grad() {
            let mut gs = vec![<T as num_traits::Zero>::zero(); index_numel];
            let mut coords = vec![0usize; ndim];
            for (i, gs_elem) in gs.iter_mut().enumerate() {
                let idx_val = self.index[i];
                let mut src_coords = coords.clone();
                src_coords[self.dim] = idx_val;
                let src_flat = flat_index(&src_coords, input_shape);
                *gs_elem = go_data[src_flat];

                if i + 1 < index_numel {
                    increment_coords(&mut coords, &self.index_shape);
                }
            }
            let t = Tensor::from_storage(TensorStorage::cpu(gs), self.index_shape.clone(), false)?;
            Some(t)
        } else {
            None
        };

        Ok(vec![grad_input, grad_src])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.src]
    }

    fn name(&self) -> &'static str {
        "ScatterAddBackward"
    }
}

// ---------------------------------------------------------------------------
// where_cond
// ---------------------------------------------------------------------------

/// Backward function for `where_cond`.
///
/// Forward: `output[i] = condition[i] ? x[i] : y[i]`
///
/// VJP for x: `grad_x[i] = condition[i] ? grad_output[i] : 0`
/// VJP for y: `grad_y[i] = condition[i] ? 0 : grad_output[i]`
#[derive(Debug)]
pub struct WhereCondBackward<T: Float> {
    /// The x tensor from the forward pass.
    pub x: Tensor<T>,
    /// The y tensor from the forward pass.
    pub y: Tensor<T>,
    /// The condition mask from the forward pass.
    pub condition: Vec<bool>,
}

impl<T: Float> GradFn<T> for WhereCondBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None, None]);
        }

        // §3 GPU-native path: upload the bool condition as a f32 mask (1.0=true, 0.0=false)
        // and use masked_zero_f32 on-device:
        //   grad_x[i] = condition[i] ? grad[i] : 0  → zero where condition=false (NOT-mask)
        //   grad_y[i] = condition[i] ? 0 : grad[i]  → zero where condition=true  (mask)
        if grad_output.is_cuda() {
            let ordinal = match grad_output.device() {
                Device::Cuda(o) => o,
                _ => unreachable!(),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

            // condition_mask: 1.0 where condition=true, 0.0 where false.
            let condition_mask: Vec<f32> = self
                .condition
                .iter()
                .map(|&c| if c { 1.0f32 } else { 0.0 })
                .collect();
            // not_mask: 1.0 where condition=false (used to zero grad_x at those positions).
            let not_mask: Vec<f32> = self
                .condition
                .iter()
                .map(|&c| if c { 0.0f32 } else { 1.0 })
                .collect();

            let grad_x = if self.x.requires_grad() {
                let not_mask_h = upload_f32_to_gpu(&not_mask, ordinal)?;
                let result_h = backend.masked_zero_f32(grad_output.gpu_handle()?, &not_mask_h)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.x.shape().to_vec(),
                    false,
                )?)
            } else {
                None
            };

            let grad_y = if self.y.requires_grad() {
                let cond_mask_h = upload_f32_to_gpu(&condition_mask, ordinal)?;
                let result_h = backend.masked_zero_f32(grad_output.gpu_handle()?, &cond_mask_h)?;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    self.y.shape().to_vec(),
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_x, grad_y]);
        }

        let go_data = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        let grad_x = if self.x.requires_grad() {
            let gx: Vec<T> = self
                .condition
                .iter()
                .zip(go_data.iter())
                .map(|(&c, &g)| if c { g } else { zero })
                .collect();
            let t = Tensor::from_storage(TensorStorage::cpu(gx), self.x.shape().to_vec(), false)?;
            Some(t)
        } else {
            None
        };

        let grad_y = if self.y.requires_grad() {
            let gy: Vec<T> = self
                .condition
                .iter()
                .zip(go_data.iter())
                .map(|(&c, &g)| if c { zero } else { g })
                .collect();
            let t = Tensor::from_storage(TensorStorage::cpu(gy), self.y.shape().to_vec(), false)?;
            Some(t)
        } else {
            None
        };

        Ok(vec![grad_x, grad_y])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.x, &self.y]
    }

    fn name(&self) -> &'static str {
        "WhereCondBackward"
    }
}

// ---------------------------------------------------------------------------
// First-class IntTensor / BoolTensor wrappers (#615)
// ---------------------------------------------------------------------------

/// `masked_fill` taking a [`BoolTensor`] mask. Shape and numel must
/// match `input`. Returns a new tensor; original unchanged. Mirrors
/// torch's `tensor.masked_fill(mask, value)` with mask convention
/// "true → fill" (same as the existing `&[bool]` variant).
pub fn masked_fill_bt<T: Float>(
    input: &Tensor<T>,
    mask: &BoolTensor,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    if mask.numel() != input.numel() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "masked_fill_bt: mask numel={} != input numel={}",
                mask.numel(),
                input.numel()
            ),
        });
    }
    masked_fill(input, mask.data(), value)
}

/// `index_select_1d` taking an [`IntTensor`] of indices. The index tensor
/// must be 1-D and contain non-negative values within range.
pub fn index_select_1d_it<T: Float, I: IntElement>(
    input: &Tensor<T>,
    indices: &IntTensor<I>,
) -> FerrotorchResult<Tensor<T>> {
    if indices.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "index_select_1d_it: indices must be 1-D, got shape {:?}",
                indices.shape()
            ),
        });
    }
    let mut idx_usize: Vec<usize> = Vec::with_capacity(indices.numel());
    for v in indices.data() {
        let i = v.to_i64();
        if i < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("index_select_1d_it: negative index {i} not allowed"),
            });
        }
        idx_usize.push(i as usize);
    }
    index_select_1d(input, &idx_usize)
}

// ---------------------------------------------------------------------------
// index_select_dim — N-D, gather along arbitrary axis with 1-D indices (#1014)
// ---------------------------------------------------------------------------

/// Backward function for [`index_select_dim`].
///
/// Forward (for `dim = D`): `output[..., i, ...] = input[..., indices[i], ...]`,
/// i.e. each "slice" along `dim` of `output` at position `i` is a copy of the
/// `input` slice at position `indices[i]`.
///
/// VJP: scatter-add `grad_output` back along `dim` at positions `indices`,
/// accumulating duplicates. This is the N-D analogue of the 1-D
/// `IndexSelectBackward` above.
#[derive(Debug)]
pub struct IndexSelectDimBackward<T: Float> {
    /// Saved input handle (for shape and `requires_grad` propagation).
    pub input: Tensor<T>,
    /// The dimension along which selection was performed.
    pub dim: usize,
    /// The 1-D index vector used during the forward pass.
    pub indices: Vec<usize>,
}

impl<T: Float> GradFn<T> for IndexSelectDimBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !is_grad_enabled() {
            return Ok(vec![None]);
        }
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let input_shape = self.input.shape();
        let input_numel: usize = input_shape.iter().product();
        let dim = self.dim;
        let outer: usize = input_shape[..dim].iter().product();
        let inner: usize = input_shape[dim + 1..].iter().product();
        let in_dim_size = input_shape[dim];
        let out_dim_size = self.indices.len();

        // GPU path: scatter-add via the existing 1-D kernel. We compute the
        // flat destination index in input-space for every element of
        // grad_output (which is dense, in C-order, with shape replacing
        // `dim` by `out_dim_size`), upload, and reuse
        // `scatter_add_1d_{f32,f64}`. f64 inputs now reach this path
        // via #1098 (CUDA forward for `index_select_dim`); fall back to
        // CPU only for non-{f32,f64} floats so we never silently demote
        // an in-graph CUDA buffer.
        if grad_output.is_cuda() {
            use std::any::TypeId;
            let is_t_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
            let is_t_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
            if is_t_f32 || is_t_f64 {
                let ordinal = match grad_output.device() {
                    Device::Cuda(o) => o,
                    _ => unreachable!(),
                };
                let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

                // Build flat destination indices, one per grad_output element.
                //
                // For grad_output with C-contiguous layout
                //   [outer, out_dim_size, inner]
                // and target buffer (= input space) with layout
                //   [outer, in_dim_size, inner]
                // a grad_output element at flat position
                //   o * out_dim_size * inner + i * inner + k
                // maps to flat dst position
                //   o * in_dim_size * inner + indices[i] * inner + k
                let go_numel = outer * out_dim_size * inner;
                let mut dst_indices: Vec<f32> = Vec::with_capacity(go_numel);
                for o in 0..outer {
                    for i in 0..out_dim_size {
                        let dst_i = self.indices[i];
                        let base = o * in_dim_size * inner + dst_i * inner;
                        for k in 0..inner {
                            dst_indices.push((base + k) as f32);
                        }
                    }
                }

                let idx_handle = upload_f32_to_gpu(&dst_indices, ordinal)?;
                let result_handle = if is_t_f32 {
                    backend.scatter_add_1d_f32(
                        grad_output.gpu_handle()?,
                        &idx_handle,
                        input_numel,
                    )?
                } else {
                    backend.scatter_add_1d_f64(
                        grad_output.gpu_handle()?,
                        &idx_handle,
                        input_numel,
                    )?
                };
                let grad_tensor = Tensor::from_storage(
                    TensorStorage::gpu(result_handle),
                    input_shape.to_vec(),
                    false,
                )?;
                return Ok(vec![Some(grad_tensor)]);
            }
            // Unsupported float dtype on CUDA: surface explicitly.
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "IndexSelectDimBackward",
            });
        }

        // CPU path: scatter-add directly.
        let go_data = grad_output.data_vec()?;
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_numel];
        for o in 0..outer {
            for i in 0..out_dim_size {
                let dst_i = self.indices[i];
                let go_base = o * out_dim_size * inner + i * inner;
                let gi_base = o * in_dim_size * inner + dst_i * inner;
                for k in 0..inner {
                    grad_input[gi_base + k] += go_data[go_base + k];
                }
            }
        }

        let grad_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_input), input_shape.to_vec(), false)?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IndexSelectDimBackward"
    }
}

/// Differentiable N-D `index_select`: gathers slices along `dim` of `input`
/// using a 1-D vector of indices.
///
/// Mirrors `torch.index_select(input, dim, indices)`:
///
/// ```text
/// output[..., i, ...] = input[..., indices[i], ...]   (slice at axis `dim`)
/// ```
///
/// The output has the same shape as `input` except `output.shape()[dim] ==
/// indices.len()`. Indices may repeat — duplicates accumulate in backward.
///
/// If `input.requires_grad()` and grad is enabled, the result carries an
/// [`IndexSelectDimBackward`] grad_fn whose VJP scatter-adds `grad_output`
/// along `dim` back at the saved `indices` positions.
pub fn index_select_dim<T: Float, I: IntElement>(
    input: &Tensor<T>,
    dim: usize,
    indices: &IntTensor<I>,
) -> FerrotorchResult<Tensor<T>> {
    let input_shape = input.shape();
    let ndim = input_shape.len();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "index_select_dim: input must have at least 1 dimension".into(),
        });
    }
    if dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "index_select_dim: dim {dim} out of range for shape {input_shape:?}"
            ),
        });
    }
    if indices.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "index_select_dim: indices must be 1-D, got shape {:?}",
                indices.shape()
            ),
        });
    }

    let in_dim_size = input_shape[dim];
    // Validate + widen indices.
    let mut idx_usize: Vec<usize> = Vec::with_capacity(indices.numel());
    for v in indices.data() {
        let i = v.to_i64();
        if i < 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("index_select_dim: negative index {i} not allowed"),
            });
        }
        let iu = i as usize;
        if iu >= in_dim_size {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: iu,
                axis: dim,
                size: in_dim_size,
            });
        }
        idx_usize.push(iu);
    }

    // Compute output: same shape but axis `dim` replaced by indices.len().
    let mut output_shape = input_shape.to_vec();
    output_shape[dim] = idx_usize.len();

    let outer: usize = input_shape[..dim].iter().product();
    let inner: usize = input_shape[dim + 1..].iter().product();
    let out_dim_size = idx_usize.len();

    // GPU path: route via TypeId to the f32/f64 device-resident gather
    // kernel. The output buffer is allocated on-device; no host
    // round-trip. Indices are f32-encoded (backend convention shared
    // with `index_select_1d_f32`, `scatter_add_1d_f32`, etc.).
    if input.is_cuda() {
        use std::any::TypeId;
        let is_t_f32 = TypeId::of::<T>() == TypeId::of::<f32>();
        let is_t_f64 = TypeId::of::<T>() == TypeId::of::<f64>();
        if is_t_f32 || is_t_f64 {
            let ordinal = match input.device() {
                Device::Cuda(o) => o,
                _ => unreachable!("input.is_cuda() but device() not Cuda"),
            };
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            // Upload indices as f32 (the established encoding for
            // index buffers across the GPU dispatch surface).
            let indices_f32: Vec<f32> = idx_usize.iter().map(|&u| u as f32).collect();
            let idx_handle = upload_f32_to_gpu(&indices_f32, ordinal)?;

            let result_handle = if is_t_f32 {
                backend.index_select_dim_f32(
                    input.gpu_handle()?,
                    &idx_handle,
                    outer,
                    in_dim_size,
                    out_dim_size,
                    inner,
                )?
            } else {
                backend.index_select_dim_f64(
                    input.gpu_handle()?,
                    &idx_handle,
                    outer,
                    in_dim_size,
                    out_dim_size,
                    inner,
                )?
            };

            let storage = TensorStorage::gpu(result_handle);
            return if input.requires_grad() && is_grad_enabled() {
                let grad_fn = Arc::new(IndexSelectDimBackward {
                    input: input.clone(),
                    dim,
                    indices: idx_usize,
                });
                Tensor::from_operation(storage, output_shape, grad_fn)
            } else {
                Tensor::from_storage(storage, output_shape, false)
            };
        }
        // Non-f32/f64 floats (e.g., bf16) still surface explicit
        // NotImplementedOnCuda — preserves the "no silent fallback"
        // contract for unsupported dtypes.
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "index_select_dim",
        });
    }

    // CPU path: dense memcpy along axis.
    let out_numel: usize = output_shape.iter().product();
    let in_data = input.data_vec()?;
    let mut out = vec![<T as num_traits::Zero>::zero(); out_numel];
    for o in 0..outer {
        for i in 0..out_dim_size {
            let src_i = idx_usize[i];
            let in_base = o * in_dim_size * inner + src_i * inner;
            let out_base = o * out_dim_size * inner + i * inner;
            out[out_base..out_base + inner].copy_from_slice(&in_data[in_base..in_base + inner]);
        }
    }

    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = Arc::new(IndexSelectDimBackward {
            input: input.clone(),
            dim,
            indices: idx_usize,
        });
        Tensor::from_operation(TensorStorage::cpu(out), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(out), output_shape, false)
    }
}

#[cfg(test)]
mod first_class_wrappers_tests {
    use super::*;

    #[test]
    fn masked_fill_bt_replaces_true_positions() {
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
            vec![4],
            false,
        )
        .unwrap();
        let mask = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();
        let out = masked_fill_bt(&t, &mask, -1.0).unwrap();
        assert_eq!(out.data().unwrap(), &[-1.0, 2.0, -1.0, 4.0]);
    }

    #[test]
    fn masked_fill_bt_rejects_shape_mismatch() {
        let t =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32, 2.0]), vec![2], false).unwrap();
        let mask = BoolTensor::from_vec(vec![true, false, true], vec![3]).unwrap();
        let err = masked_fill_bt(&t, &mask, 0.0).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn index_select_1d_it_picks_at_indices() {
        let t = Tensor::from_storage(
            TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0, 40.0]),
            vec![4],
            false,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![3, 0, 2], vec![3]).unwrap();
        let out = index_select_1d_it(&t, &idx).unwrap();
        assert_eq!(out.data().unwrap(), &[40.0, 10.0, 30.0]);
    }

    #[test]
    fn index_select_1d_it_rejects_2d_indices() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 4]), vec![4], false).unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 1, 2, 3], vec![2, 2]).unwrap();
        let err = index_select_1d_it(&t, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn index_select_1d_it_rejects_negative() {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 4]), vec![4], false).unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, -1, 2], vec![3]).unwrap();
        let err = index_select_1d_it(&t, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::graph::backward;
    use crate::autograd::no_grad;
    use crate::storage::TensorStorage;

    /// Helper: create a 1-D leaf tensor with `requires_grad`.
    fn leaf_1d(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            requires_grad,
        )
        .unwrap()
    }

    // --- index_select_1d forward ---

    #[test]
    fn test_index_select_1d_forward() {
        let input = leaf_1d(&[10.0, 20.0, 30.0, 40.0, 50.0], false);
        let result = index_select_1d(&input, &[0, 2, 4]).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data().unwrap(), &[10.0, 30.0, 50.0]);
    }

    #[test]
    fn test_index_select_1d_duplicate_indices() {
        let input = leaf_1d(&[10.0, 20.0, 30.0], false);
        let result = index_select_1d(&input, &[1, 1, 2, 0, 1]).unwrap();

        assert_eq!(result.shape(), &[5]);
        assert_eq!(result.data().unwrap(), &[20.0, 20.0, 30.0, 10.0, 20.0]);
    }

    #[test]
    fn test_index_select_1d_out_of_bounds() {
        let input = leaf_1d(&[10.0, 20.0, 30.0], false);
        let result = index_select_1d(&input, &[0, 5]);
        assert!(result.is_err());
    }

    #[test]
    fn test_index_select_1d_non_1d_input() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let result = index_select_1d(&input, &[0]);
        assert!(result.is_err());
    }

    // --- index_select_1d backward ---

    #[test]
    fn test_index_select_1d_backward_simple() {
        // input = [10, 20, 30, 40], select indices [1, 3]
        // output = [20, 40]
        // sum(output) = 60   (scalar for backward)
        //
        // grad_output for sum = [1, 1]
        // grad_input = [0, 1, 0, 1]  (scatter_add of [1,1] at [1,3])
        let input = leaf_1d(&[10.0, 20.0, 30.0, 40.0], true);
        let selected = index_select_1d(&input, &[1, 3]).unwrap();

        assert!(selected.requires_grad());
        assert!(!selected.is_leaf());
        assert_eq!(selected.grad_fn().unwrap().name(), "IndexSelectBackward");

        // Sum the selected tensor to get a scalar.
        let data = selected.data().unwrap();
        let total: f32 = data.iter().sum();
        let sum_storage = TensorStorage::cpu(vec![total]);

        // SumBackward: broadcasts the scalar grad_output to the shape of the input.
        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let go_val = grad_output.data()?[0];
                let grad = vec![go_val; self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(grad),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            sum_storage,
            vec![],
            Arc::new(SumBackward {
                input: selected.clone(),
            }),
        )
        .unwrap();

        backward(&loss).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        assert_eq!(grad_data.len(), 4);
        assert!((grad_data[0] - 0.0).abs() < 1e-6, "grad[0] should be 0");
        assert!((grad_data[1] - 1.0).abs() < 1e-6, "grad[1] should be 1");
        assert!((grad_data[2] - 0.0).abs() < 1e-6, "grad[2] should be 0");
        assert!((grad_data[3] - 1.0).abs() < 1e-6, "grad[3] should be 1");
    }

    #[test]
    fn test_index_select_1d_backward_duplicate_indices() {
        // input = [10, 20, 30], select indices [0, 1, 1, 2, 1]
        // output = [10, 20, 20, 30, 20]
        // sum(output) = 100
        //
        // grad_output for sum = [1, 1, 1, 1, 1]
        // grad_input:
        //   idx 0 appears 1 time -> grad_input[0] = 1
        //   idx 1 appears 3 times -> grad_input[1] = 3
        //   idx 2 appears 1 time -> grad_input[2] = 1
        let input = leaf_1d(&[10.0, 20.0, 30.0], true);
        let selected = index_select_1d(&input, &[0, 1, 1, 2, 1]).unwrap();

        // Manually invoke the backward of IndexSelectBackward with a
        // uniform grad_output of ones.
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0; 5]), vec![5], false).unwrap();

        let grad_fn = selected.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_input = grads[0].as_ref().unwrap();
        let gd = grad_input.data().unwrap();

        assert_eq!(gd.len(), 3);
        assert!(
            (gd[0] - 1.0).abs() < 1e-6,
            "grad[0] = {}, expected 1",
            gd[0]
        );
        assert!(
            (gd[1] - 3.0).abs() < 1e-6,
            "grad[1] = {}, expected 3",
            gd[1]
        );
        assert!(
            (gd[2] - 1.0).abs() < 1e-6,
            "grad[2] = {}, expected 1",
            gd[2]
        );
    }

    #[test]
    fn test_index_select_1d_backward_weighted_grad() {
        // input = [100, 200, 300], select indices [2, 0]
        // output = [300, 100]
        // grad_output = [0.5, 2.0]
        //
        // grad_input[0] += 2.0  (from output[1])
        // grad_input[2] += 0.5  (from output[0])
        // grad_input[1] = 0
        let input = leaf_1d(&[100.0, 200.0, 300.0], true);
        let selected = index_select_1d(&input, &[2, 0]).unwrap();

        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![0.5, 2.0]), vec![2], false).unwrap();

        let grad_fn = selected.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_input = grads[0].as_ref().unwrap();
        let gd = grad_input.data().unwrap();

        assert!(
            (gd[0] - 2.0).abs() < 1e-6,
            "grad[0] = {}, expected 2.0",
            gd[0]
        );
        assert!(
            (gd[1] - 0.0).abs() < 1e-6,
            "grad[1] = {}, expected 0.0",
            gd[1]
        );
        assert!(
            (gd[2] - 0.5).abs() < 1e-6,
            "grad[2] = {}, expected 0.5",
            gd[2]
        );
    }

    // --- index_select_1d: no grad when grad disabled ---

    #[test]
    fn test_index_select_1d_no_grad_context() {
        let input = leaf_1d(&[10.0, 20.0, 30.0], true);

        let result = no_grad(|| index_select_1d(&input, &[0, 2])).unwrap();

        // Under no_grad, the result should be a leaf with no grad_fn.
        assert!(!result.requires_grad());
        assert!(result.grad_fn().is_none());
    }

    // --- masked_fill forward ---

    #[test]
    fn test_masked_fill_forward() {
        let input = leaf_1d(&[1.0, 2.0, 3.0, 4.0], false);
        let mask = [false, true, false, true];
        let result = masked_fill(&input, &mask, -999.0).unwrap();

        assert_eq!(result.data().unwrap(), &[1.0, -999.0, 3.0, -999.0]);
    }

    // --- masked_fill backward ---

    #[test]
    fn test_masked_fill_backward() {
        let input = leaf_1d(&[1.0, 2.0, 3.0, 4.0], true);
        let mask = [false, true, false, true];
        let filled = masked_fill(&input, &mask, 0.0).unwrap();

        // grad_output = [1, 1, 1, 1]
        // grad_input  = [1, 0, 1, 0]  (zeroed where mask is true)
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0; 4]), vec![4], false).unwrap();

        let grad_fn = filled.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let grad_input = grads[0].as_ref().unwrap();
        let gd = grad_input.data().unwrap();

        assert!((gd[0] - 1.0).abs() < 1e-6);
        assert!((gd[1] - 0.0).abs() < 1e-6);
        assert!((gd[2] - 1.0).abs() < 1e-6);
        assert!((gd[3] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_masked_fill_shape_mismatch() {
        let input = leaf_1d(&[1.0, 2.0, 3.0], false);
        let mask = [true, false]; // wrong length
        let result = masked_fill(&input, &mask, 0.0);
        assert!(result.is_err());
    }

    // --- gather backward ---

    #[test]
    fn test_gather_backward_stub() {
        let input = leaf_1d(&[1.0, 2.0], true);
        let gf = GatherBackward {
            input,
            dim: 0,
            index: vec![0, 1],
            index_shape: vec![2],
        };
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 1.0]), vec![2], false).unwrap();
        // Should now succeed rather than error.
        let result = gf.backward(&grad_output);
        assert!(result.is_ok());
    }

    #[test]
    fn test_scatter_add_backward_stub() {
        let input = leaf_1d(&[1.0, 2.0], true);
        let src = leaf_1d(&[3.0], false);
        let gf = ScatterAddBackward {
            input,
            src,
            dim: 0,
            index: vec![0],
            index_shape: vec![1],
        };
        let grad_output =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 1.0]), vec![2], false).unwrap();
        let result = gf.backward(&grad_output);
        assert!(result.is_ok());
    }

    // -- index_select_dim (#1014) --

    #[test]
    fn test_index_select_dim_2d_dim0_forward() {
        // input: shape [4, 3]
        //   row 0: [10, 11, 12]
        //   row 1: [20, 21, 22]
        //   row 2: [30, 31, 32]
        //   row 3: [40, 41, 42]
        // indices: [3, 0, 2]
        // output: shape [3, 3]
        //   row 0 = input row 3 = [40, 41, 42]
        //   row 1 = input row 0 = [10, 11, 12]
        //   row 2 = input row 2 = [30, 31, 32]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![
                10.0_f32, 11.0, 12.0, 20.0, 21.0, 22.0, 30.0, 31.0, 32.0, 40.0, 41.0, 42.0,
            ]),
            vec![4, 3],
            false,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![3, 0, 2], vec![3]).unwrap();
        let out = index_select_dim(&input, 0, &idx).unwrap();
        assert_eq!(out.shape(), &[3, 3]);
        assert_eq!(
            out.data().unwrap(),
            &[40.0, 41.0, 42.0, 10.0, 11.0, 12.0, 30.0, 31.0, 32.0]
        );
    }

    #[test]
    fn test_index_select_dim_2d_dim1_forward() {
        // input: shape [2, 4]
        //   [[1, 2, 3, 4],
        //    [5, 6, 7, 8]]
        // dim=1, indices=[1, 3, 0]
        // output: shape [2, 3]
        //   [[2, 4, 1],
        //    [6, 8, 5]]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            vec![2, 4],
            false,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![1, 3, 0], vec![3]).unwrap();
        let out = index_select_dim(&input, 1, &idx).unwrap();
        assert_eq!(out.shape(), &[2, 3]);
        assert_eq!(out.data().unwrap(), &[2.0, 4.0, 1.0, 6.0, 8.0, 5.0]);
    }

    #[test]
    fn test_index_select_dim_registers_grad_fn() {
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![3, 2],
            true,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 2], vec![2]).unwrap();
        let out = index_select_dim(&input, 0, &idx).unwrap();
        assert!(out.requires_grad());
        assert!(!out.is_leaf());
        assert_eq!(out.grad_fn().unwrap().name(), "IndexSelectDimBackward");
    }

    #[test]
    fn test_index_select_dim_backward_simple_2d() {
        // input: [4, 2], indices [2, 0, 2] along dim=0 → output [3, 2]
        // grad_output =
        //   [[1, 10],
        //    [100, 1000],
        //    [10000, 100000]]
        // expected grad_input (scatter-add along dim 0, accumulating dups):
        //   row 0: from grad_output row 1            -> [100, 1000]
        //   row 1: untouched                         -> [0, 0]
        //   row 2: from grad_output rows 0 + 2       -> [1+10000, 10+100000] = [10001, 100010]
        //   row 3: untouched                         -> [0, 0]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![
                1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, // arbitrary
            ]),
            vec![4, 2],
            true,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![2, 0, 2], vec![3]).unwrap();
        let out = index_select_dim(&input, 0, &idx).unwrap();

        let grad_output = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 10.0, 100.0, 1000.0, 10000.0, 100000.0]),
            vec![3, 2],
            false,
        )
        .unwrap();

        let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();
        let g = grads[0].as_ref().unwrap();
        assert_eq!(g.shape(), &[4, 2]);
        let gd = g.data().unwrap();
        let expected = [100.0_f32, 1000.0, 0.0, 0.0, 10001.0, 100010.0, 0.0, 0.0];
        for (i, (&got, &exp)) in gd.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "grad[{i}] = {got}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_index_select_dim_backward_dim1() {
        // input: [2, 4], indices [3, 1] along dim=1 → output [2, 2]
        // grad_output =
        //   [[1, 10], [100, 1000]]
        // expected grad_input (per-row scatter into 4 columns at cols 3 and 1):
        //   row 0: [0, 10, 0, 1]
        //   row 1: [0, 1000, 0, 100]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
            vec![2, 4],
            true,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![3, 1], vec![2]).unwrap();
        let out = index_select_dim(&input, 1, &idx).unwrap();

        let grad_output = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 10.0, 100.0, 1000.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let grads = out.grad_fn().unwrap().backward(&grad_output).unwrap();
        let g = grads[0].as_ref().unwrap();
        assert_eq!(g.shape(), &[2, 4]);
        let gd = g.data().unwrap();
        let expected = [0.0_f32, 10.0, 0.0, 1.0, 0.0, 1000.0, 0.0, 100.0];
        for (i, (&got, &exp)) in gd.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "grad[{i}] = {got}, expected {exp}");
        }
    }

    #[test]
    fn test_index_select_dim_e2e_via_autograd() {
        // End-to-end: drive the gradient through the autograd graph (rather
        // than calling backward() directly on the grad_fn) and verify the
        // input.grad() lands on the bias-table parameter equivalent.
        // input: [3, 2] = [[1,2],[3,4],[5,6]], indices [0, 2, 0] on dim=0
        // out: [3, 2] = [[1,2],[5,6],[1,2]]
        // sum(out) = 1+2+5+6+1+2 = 17
        // grad_out (from sum) = ones([3, 2])
        // grad_input (scatter-add along dim 0):
        //   row 0: from out rows 0 and 2 -> [2, 2]
        //   row 1: untouched              -> [0, 0]
        //   row 2: from out row 1         -> [1, 1]
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![3, 2],
            true,
        )
        .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 2, 0], vec![3]).unwrap();
        let out = index_select_dim(&x, 0, &idx).unwrap();
        let total: f32 = out.data().unwrap().iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new({
                #[derive(Debug)]
                struct SumBackward<T: Float> {
                    input: Tensor<T>,
                }
                impl<T: Float> GradFn<T> for SumBackward<T> {
                    fn backward(
                        &self,
                        _go: &Tensor<T>,
                    ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                        let n = self.input.numel();
                        let ones = vec![<T as num_traits::One>::one(); n];
                        let g = Tensor::from_storage(
                            TensorStorage::cpu(ones),
                            self.input.shape().to_vec(),
                            false,
                        )?;
                        Ok(vec![Some(g)])
                    }
                    fn inputs(&self) -> Vec<&Tensor<T>> {
                        vec![&self.input]
                    }
                    fn name(&self) -> &'static str {
                        "SumBackward"
                    }
                }
                SumBackward {
                    input: out.clone(),
                }
            }),
        )
        .unwrap();

        crate::autograd::graph::backward(&loss).unwrap();

        let grad = x.grad().unwrap().expect("x.grad() should be Some");
        assert_eq!(grad.shape(), &[3, 2]);
        let gd = grad.data().unwrap();
        let expected = [2.0_f32, 2.0, 0.0, 0.0, 1.0, 1.0];
        for (i, (&got, &exp)) in gd.iter().zip(expected.iter()).enumerate() {
            assert!((got - exp).abs() < 1e-6, "grad[{i}] = {got}, expected {exp}");
        }
    }

    #[test]
    fn test_index_select_dim_rejects_2d_indices() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 6]), vec![3, 2], false)
            .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 1, 0, 1], vec![2, 2]).unwrap();
        let err = index_select_dim(&x, 0, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn test_index_select_dim_rejects_oob() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 6]), vec![3, 2], false)
            .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, 7], vec![2]).unwrap();
        let err = index_select_dim(&x, 0, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::IndexOutOfBounds { .. }));
    }

    #[test]
    fn test_index_select_dim_rejects_negative() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 6]), vec![3, 2], false)
            .unwrap();
        let idx: IntTensor<i64> = IntTensor::from_vec(vec![0, -1], vec![2]).unwrap();
        let err = index_select_dim(&x, 0, &idx).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }
}
