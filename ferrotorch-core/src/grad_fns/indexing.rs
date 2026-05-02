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
    let bytes: &[u8] =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    backend.cpu_to_gpu(bytes, 4, ordinal)
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

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "gather backward",
            });
        }

        let input_shape = self.input.shape();
        let input_numel: usize = input_shape.iter().product();
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

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "scatter backward",
            });
        }

        let input_shape = self.input.shape();
        let ndim = input_shape.len();
        let index_numel: usize = self.index_shape.iter().product();
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

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "scatter_add backward",
            });
        }

        let input_shape = self.input.shape();
        let ndim = input_shape.len();
        let index_numel: usize = self.index_shape.iter().product();
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

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "where_cond backward",
            });
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
}
