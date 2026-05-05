//! Cumulative (scan) tensor operations along a specified dimension.
//!
//! Provides `cumsum`, `cumprod`, `cummax`, `cummin`, and `logcumsumexp` as
//! pure forward-pass kernels. The differentiable wrappers that attach autograd
//! nodes live in `grad_fns::cumulative`.
//!
//! [CL-306]

use std::any::TypeId;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::shape::normalize_axis;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

// ---------------------------------------------------------------------------
// Stride helpers
// ---------------------------------------------------------------------------

/// Compute (outer_size, dim_size, inner_size) for iterating along `dim`.
///
/// For a shape `[d0, d1, ..., dn]` and a given `dim`, this factorises the
/// flat index space into:
///   - `outer_size` = product of dims before `dim`
///   - `dim_size`   = shape[dim]
///   - `inner_size` = product of dims after `dim`
///
/// The flat index of element `(outer, i, inner)` is:
///   `outer * dim_size * inner_size + i * inner_size + inner`
fn dim_strides(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, dim_size, inner)
}

/// Normalise and validate `dim` for cumulative ops.
fn validate_dim(ndim: usize, dim: i64, op_name: &str) -> FerrotorchResult<usize> {
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("{op_name}: cannot operate on a scalar (0-D) tensor"),
        });
    }
    normalize_axis(dim as isize, ndim)
}

// ---------------------------------------------------------------------------
// cumsum
// ---------------------------------------------------------------------------

/// Cumulative sum along `dim`.
///
/// Output shape is identical to input shape. Element `[..., i, ...]` along
/// `dim` equals `sum(input[..., 0..=i, ...])`.
pub fn cumsum_forward<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "cumsum")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let handle = if is_f32::<T>() {
                backend.cumsum_f32(input.gpu_handle()?, outer, dim_size, inner)?
            } else {
                backend.cumsum_f64(input.gpu_handle()?, outer, dim_size, inner)?
            };
            return Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false);
        }
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cumsum" });
    }

    let in_data = input.data()?;

    let mut out = vec![<T as num_traits::Zero>::zero(); in_data.len()];

    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut acc = <T as num_traits::Zero>::zero();
            for i in 0..dim_size {
                let idx = base + i * inner;
                acc += in_data[idx];
                out[idx] = acc;
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}

/// Reverse cumulative sum along `dim` (used for cumsum backward).
///
/// `reverse_cumsum[..., i, ...] = sum(input[..., i..dim_size, ...])`
pub fn reverse_cumsum<T: Float>(data: &[T], shape: &[usize], dim: usize) -> Vec<T> {
    let (outer, dim_size, inner) = dim_strides(shape, dim);
    let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];

    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut acc = <T as num_traits::Zero>::zero();
            for i in (0..dim_size).rev() {
                let idx = base + i * inner;
                acc += data[idx];
                out[idx] = acc;
            }
        }
    }
    out
}

// ---------------------------------------------------------------------------
// cumprod
// ---------------------------------------------------------------------------

/// Cumulative product along `dim`.
///
/// Output shape is identical to input shape. Element `[..., i, ...]` along
/// `dim` equals `prod(input[..., 0..=i, ...])`.
pub fn cumprod_forward<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "cumprod")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let handle = if is_f32::<T>() {
                backend.cumprod_f32(input.gpu_handle()?, outer, dim_size, inner)?
            } else {
                backend.cumprod_f64(input.gpu_handle()?, outer, dim_size, inner)?
            };
            return Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false);
        }
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cumprod" });
    }

    let in_data = input.data()?;

    let mut out = vec![<T as num_traits::Zero>::zero(); in_data.len()];

    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut acc = <T as num_traits::One>::one();
            for i in 0..dim_size {
                let idx = base + i * inner;
                acc = acc * in_data[idx];
                out[idx] = acc;
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}

// ---------------------------------------------------------------------------
// cummax
// ---------------------------------------------------------------------------

/// Result of `cummax` / `cummin`: values tensor and indices tensor.
#[derive(Debug)]
pub struct CumExtremeResult<T: Float> {
    pub values: Tensor<T>,
    pub indices: Vec<usize>,
}

/// Cumulative maximum along `dim`.
///
/// Returns `(values, indices)` where `values[..., i, ...]` is the running
/// maximum of `input[..., 0..=i, ...]` and `indices` holds the flat-along-dim
/// index at which each running maximum was attained.
pub fn cummax_forward<T: Float>(
    input: &Tensor<T>,
    dim: i64,
) -> FerrotorchResult<CumExtremeResult<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "cummax")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64 — kernel returns both values and indices
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let (vals_h, idxs_h) = if is_f32::<T>() {
                backend.cummax_f32(input.gpu_handle()?, outer, dim_size, inner)?
            } else {
                backend.cummax_f64(input.gpu_handle()?, outer, dim_size, inner)?
            };
            let values = Tensor::from_storage(TensorStorage::gpu(vals_h), shape.to_vec(), false)?;
            // Indices are stored as f32 on GPU — download and convert to usize
            let idxs_tensor =
                Tensor::<f32>::from_storage(TensorStorage::gpu(idxs_h), shape.to_vec(), false)?;
            let idxs_cpu = idxs_tensor.cpu()?;
            let idxs_data = idxs_cpu.data()?;
            let indices: Vec<usize> = idxs_data.iter().map(|&v| v as usize).collect();
            return Ok(CumExtremeResult { values, indices });
        }
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cummax" });
    }

    let in_data = input.data()?;
    let numel = in_data.len();
    let mut out_vals = vec![<T as num_traits::Zero>::zero(); numel];
    let mut out_idxs = vec![0usize; numel];

    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut cur_max = <T as num_traits::Float>::neg_infinity();
            let mut cur_idx = 0usize;
            for i in 0..dim_size {
                let idx = base + i * inner;
                if in_data[idx] > cur_max {
                    cur_max = in_data[idx];
                    cur_idx = i;
                }
                out_vals[idx] = cur_max;
                out_idxs[idx] = cur_idx;
            }
        }
    }

    let values = Tensor::from_storage(TensorStorage::cpu(out_vals), shape.to_vec(), false)?;
    Ok(CumExtremeResult {
        values,
        indices: out_idxs,
    })
}

// ---------------------------------------------------------------------------
// cummin
// ---------------------------------------------------------------------------

/// Cumulative minimum along `dim`.
///
/// Returns `(values, indices)` analogous to [`cummax_forward`] but tracking
/// the running minimum.
pub fn cummin_forward<T: Float>(
    input: &Tensor<T>,
    dim: i64,
) -> FerrotorchResult<CumExtremeResult<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "cummin")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64 — kernel returns both values and indices
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let (vals_h, idxs_h) = if is_f32::<T>() {
                backend.cummin_f32(input.gpu_handle()?, outer, dim_size, inner)?
            } else {
                backend.cummin_f64(input.gpu_handle()?, outer, dim_size, inner)?
            };
            let values = Tensor::from_storage(TensorStorage::gpu(vals_h), shape.to_vec(), false)?;
            let idxs_tensor =
                Tensor::<f32>::from_storage(TensorStorage::gpu(idxs_h), shape.to_vec(), false)?;
            let idxs_cpu = idxs_tensor.cpu()?;
            let idxs_data = idxs_cpu.data()?;
            let indices: Vec<usize> = idxs_data.iter().map(|&v| v as usize).collect();
            return Ok(CumExtremeResult { values, indices });
        }
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cummin" });
    }

    let in_data = input.data()?;
    let numel = in_data.len();
    let mut out_vals = vec![<T as num_traits::Zero>::zero(); numel];
    let mut out_idxs = vec![0usize; numel];

    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;
            let mut cur_min = <T as num_traits::Float>::infinity();
            let mut cur_idx = 0usize;
            for i in 0..dim_size {
                let idx = base + i * inner;
                if in_data[idx] < cur_min {
                    cur_min = in_data[idx];
                    cur_idx = i;
                }
                out_vals[idx] = cur_min;
                out_idxs[idx] = cur_idx;
            }
        }
    }

    let values = Tensor::from_storage(TensorStorage::cpu(out_vals), shape.to_vec(), false)?;
    Ok(CumExtremeResult {
        values,
        indices: out_idxs,
    })
}

// ---------------------------------------------------------------------------
// logcumsumexp
// ---------------------------------------------------------------------------

/// Log-cumulative-sum-exp along `dim`.
///
/// `output[..., i, ...] = log(sum(exp(input[..., 0..=i, ...])))`
///
/// Numerically stable: uses the running-max trick to avoid overflow.
pub fn logcumsumexp_forward<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let norm_dim = validate_dim(input.ndim(), dim, "logcumsumexp")?;
    let shape = input.shape();
    let (outer, dim_size, inner) = dim_strides(shape, norm_dim);

    // GPU fast path for f32/f64
    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let handle = if is_f32::<T>() {
                backend.logcumsumexp_f32(input.gpu_handle()?, outer, dim_size, inner)?
            } else {
                backend.logcumsumexp_f64(input.gpu_handle()?, outer, dim_size, inner)?
            };
            return Tensor::from_storage(TensorStorage::gpu(handle), shape.to_vec(), false);
        }
    }

    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "logcumsumexp" });
    }

    let in_data = input.data()?;

    let mut out = vec![<T as num_traits::Zero>::zero(); in_data.len()];
    let neg_inf = <T as num_traits::Float>::neg_infinity();

    for o in 0..outer {
        for k in 0..inner {
            let base = o * dim_size * inner + k;

            // First pass: compute running max for numerical stability.
            let mut running_max = neg_inf;
            let mut maxes = Vec::with_capacity(dim_size);
            for i in 0..dim_size {
                let idx = base + i * inner;
                if in_data[idx] > running_max {
                    running_max = in_data[idx];
                }
                maxes.push(running_max);
            }

            // Second pass: accumulate exp(x_i - running_max) and take log.
            let mut acc = <T as num_traits::Zero>::zero();
            let mut prev_max = neg_inf;
            for (i, &m) in maxes.iter().enumerate().take(dim_size) {
                let idx = base + i * inner;

                // When the running max changes, rescale the accumulator.
                if m > prev_max && prev_max != neg_inf {
                    #[allow(clippy::assign_op_pattern)]
                    {
                        acc = acc * (prev_max - m).exp();
                    }
                }
                acc += (in_data[idx] - m).exp();
                out[idx] = m + acc.ln();
                prev_max = m;
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}
