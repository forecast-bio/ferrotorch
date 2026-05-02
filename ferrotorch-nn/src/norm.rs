//! Normalization layers: LayerNorm, GroupNorm, RMSNorm, BatchNorm1d/2d/3d,
//! InstanceNorm1d/2d/3d, LocalResponseNorm.
//!
//! Each layer normalizes its input along specified dimensions and optionally
//! applies a learnable affine transform (weight/bias). Backward functions
//! implement `GradFn<T>` to propagate gradients through the normalization.

use std::any::TypeId;
use std::sync::{Arc, Mutex};

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::gpu_dispatch::gpu_backend;
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::module::Module;
use crate::parameter::Parameter;

#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Shorthand for the unambiguous zero.
#[inline]
fn zero<T: Float>() -> T {
    <T as num_traits::Zero>::zero()
}

// ===========================================================================
// LayerNorm
// ===========================================================================

/// Layer normalization over the last dimension.
///
/// Applies the transform:
///
/// ```text
/// y = (x - mean) / sqrt(var + eps) * weight + bias
/// ```
///
/// where `mean` and `var` are computed over the last dimension of the input.
/// This simplified implementation supports 1-D `normalized_shape` (a single
/// integer), which is the most common use case (transformer hidden dim).
///
/// Matches `torch.nn.LayerNorm` with a single-element `normalized_shape`.
#[derive(Debug)]
pub struct LayerNorm<T: Float> {
    /// The size of the normalized dimension.
    pub normalized_shape: Vec<usize>,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Whether to apply learnable affine parameters.
    pub elementwise_affine: bool,
    /// Learnable scale (gamma), shape = `normalized_shape`.
    pub weight: Parameter<T>,
    /// Learnable shift (beta), shape = `normalized_shape`.
    pub bias: Parameter<T>,
    training: bool,
}

impl<T: Float> LayerNorm<T> {
    /// Create a new `LayerNorm` layer.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - The shape of the dimensions to normalize over.
    ///   For the simplified implementation, this should be a single-element
    ///   slice `[hidden_dim]`.
    /// * `eps` - Small constant for numerical stability (default: 1e-5).
    /// * `elementwise_affine` - Whether to include learnable weight and bias.
    pub fn new(
        normalized_shape: Vec<usize>,
        eps: f64,
        elementwise_affine: bool,
    ) -> FerrotorchResult<Self> {
        if normalized_shape.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "normalized_shape must not be empty".into(),
            });
        }

        let weight = Parameter::ones(&normalized_shape)?;
        let bias = Parameter::zeros(&normalized_shape)?;

        Ok(Self {
            normalized_shape,
            eps,
            elementwise_affine,
            weight,
            bias,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for LayerNorm<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        let ndim = shape.len();
        let norm_ndim = self.normalized_shape.len();

        if ndim < norm_ndim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LayerNorm: input has {} dims but normalized_shape has {} dims",
                    ndim, norm_ndim
                ),
            });
        }

        // Verify that the last N dims of input match normalized_shape.
        let last_dims = &shape[ndim - norm_ndim..];
        if last_dims != self.normalized_shape.as_slice() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LayerNorm: input last dims {:?} don't match normalized_shape {:?}",
                    last_dims, self.normalized_shape
                ),
            });
        }

        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size = input.numel() / norm_size;

        // GPU fast path: native LayerNorm kernel.
        if input.is_cuda() && self.elementwise_affine {
            if let Some(backend) = ferrotorch_core::gpu_dispatch::gpu_backend() {
                let eps_f32 = self.eps as f32;
                let handle = backend.layernorm_f32(
                    input.gpu_handle()?,
                    self.weight.tensor().gpu_handle()?,
                    self.bias.tensor().gpu_handle()?,
                    batch_size,
                    norm_size,
                    eps_f32,
                )?;
                return if is_grad_enabled() && input.requires_grad() {
                    let grad_fn = Arc::new(LayerNormBackward {
                        input: input.clone(),
                        weight: self.weight.tensor().clone(),
                        bias: self.bias.tensor().clone(),
                        normalized_shape: self.normalized_shape.clone(),
                        eps: self.eps,
                        elementwise_affine: self.elementwise_affine,
                    });
                    Tensor::from_operation(TensorStorage::gpu(handle), shape, grad_fn)
                } else {
                    Tensor::from_storage(TensorStorage::gpu(handle), shape, false)
                };
            }
        }

        // CPU path — CUDA inputs without a GPU backend are rejected above.
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "LayerNorm::forward",
            });
        }
        let input_data = input.data()?;
        let eps_t = T::from(self.eps).unwrap();
        let n_t = T::from(norm_size).unwrap();

        let weight_data = self.weight.tensor().data()?;
        let bias_data = self.bias.tensor().data()?;

        let mut output = Vec::with_capacity(input.numel());

        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let slice = &input_data[start..end];

            let mean = slice.iter().copied().fold(zero::<T>(), |a, x| a + x) / n_t;
            let var = slice.iter().copied().fold(zero::<T>(), |a, x| {
                let d = x - mean;
                a + d * d
            }) / n_t;
            let inv_std = (var + eps_t).sqrt().recip();

            for (i, &x) in slice.iter().enumerate() {
                let normed = (x - mean) * inv_std;
                if self.elementwise_affine {
                    output.push(normed * weight_data[i] + bias_data[i]);
                } else {
                    output.push(normed);
                }
            }
        }

        let storage = TensorStorage::cpu(output);

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(LayerNormBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.tensor().clone(),
                normalized_shape: self.normalized_shape.clone(),
                eps: self.eps,
                elementwise_affine: self.elementwise_affine,
            });
            Tensor::from_operation(storage, shape.to_vec(), grad_fn)
        } else {
            Tensor::from_storage(storage, shape.to_vec(), false)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        if self.elementwise_affine {
            vec![&self.weight, &self.bias]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        if self.elementwise_affine {
            vec![&mut self.weight, &mut self.bias]
        } else {
            vec![]
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        if self.elementwise_affine {
            vec![
                ("weight".to_string(), &self.weight),
                ("bias".to_string(), &self.bias),
            ]
        } else {
            vec![]
        }
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// LayerNormBackward
// ---------------------------------------------------------------------------

/// Backward node for LayerNorm.
///
/// Given forward: `y = (x - mean) / std * weight + bias`
///
/// The gradients are:
/// - `d_bias = sum(grad_output, over batch dims)`
/// - `d_weight = sum(grad_output * x_hat, over batch dims)`
/// - `d_input`: standard layer norm VJP
///
/// Inputs stored: `[input, weight, bias]`.
/// Returns: `[grad_input, grad_weight, grad_bias]`.
#[derive(Debug)]
struct LayerNormBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    normalized_shape: Vec<usize>,
    eps: f64,
    elementwise_affine: bool,
}

impl<T: Float> GradFn<T> for LayerNormBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size = self.input.numel() / norm_size;

        // GPU-native fast path for f32/f64 with elementwise affine
        if self.input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) && self.elementwise_affine {
            if let Some(backend) = gpu_backend() {
                let (gi_h, gw_h, gb_h) = if is_f64::<T>() {
                    backend.layernorm_backward_f64(
                        self.input.gpu_handle()?,
                        grad_output.gpu_handle()?,
                        self.weight.gpu_handle()?,
                        batch_size,
                        norm_size,
                        self.eps,
                    )?
                } else {
                    backend.layernorm_backward_f32(
                        self.input.gpu_handle()?,
                        grad_output.gpu_handle()?,
                        self.weight.gpu_handle()?,
                        batch_size,
                        norm_size,
                        self.eps as f32,
                    )?
                };

                let grad_input_tensor = Tensor::from_storage(
                    TensorStorage::gpu(gi_h),
                    self.input.shape().to_vec(),
                    false,
                )?;

                let grad_weight_out = if self.weight.requires_grad() {
                    Some(Tensor::from_storage(
                        TensorStorage::gpu(gw_h),
                        self.normalized_shape.clone(),
                        false,
                    )?)
                } else {
                    None
                };

                let grad_bias_out = if self.bias.requires_grad() {
                    Some(Tensor::from_storage(
                        TensorStorage::gpu(gb_h),
                        self.normalized_shape.clone(),
                        false,
                    )?)
                } else {
                    None
                };

                return Ok(vec![
                    Some(grad_input_tensor),
                    grad_weight_out,
                    grad_bias_out,
                ]);
            }
        }

        // CPU-only path — CUDA inputs without a GPU backend are rejected above.
        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "LayerNormBackward",
            });
        }
        let n_t = T::from(norm_size).unwrap();
        let eps_t = T::from(self.eps).unwrap();

        let input_data = self.input.data()?;
        let go_data = grad_output.data()?;
        let weight_data = self.weight.data()?;

        let mut grad_input = vec![zero::<T>(); self.input.numel()];
        let mut grad_weight = vec![zero::<T>(); norm_size];
        let mut grad_bias = vec![zero::<T>(); norm_size];

        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let x_slice = &input_data[start..end];
            let go_slice = &go_data[start..end];

            // Recompute mean and inv_std.
            let mean = x_slice.iter().copied().fold(zero::<T>(), |a, x| a + x) / n_t;
            let var = x_slice.iter().copied().fold(zero::<T>(), |a, x| {
                let d = x - mean;
                a + d * d
            }) / n_t;
            let inv_std = (var + eps_t).sqrt().recip();

            // dl/dx_hat = go * weight (if affine) or go (if not).
            // Accumulate sums needed for the VJP.
            let mut dl_dx_hat_sum = zero::<T>();
            let mut dl_dx_hat_x_hat_sum = zero::<T>();

            for i in 0..norm_size {
                let x_hat_i = (x_slice[i] - mean) * inv_std;
                let dl_dx_hat_i = if self.elementwise_affine {
                    go_slice[i] * weight_data[i]
                } else {
                    go_slice[i]
                };

                dl_dx_hat_sum += dl_dx_hat_i;
                dl_dx_hat_x_hat_sum += dl_dx_hat_i * x_hat_i;

                if self.elementwise_affine {
                    grad_weight[i] += go_slice[i] * x_hat_i;
                    grad_bias[i] += go_slice[i];
                }
            }

            // Compute grad_input for this batch element.
            let dl_dx_hat_mean = dl_dx_hat_sum / n_t;
            let dl_dx_hat_x_hat_mean = dl_dx_hat_x_hat_sum / n_t;

            for i in 0..norm_size {
                let x_hat_i = (x_slice[i] - mean) * inv_std;
                let dl_dx_hat_i = if self.elementwise_affine {
                    go_slice[i] * weight_data[i]
                } else {
                    go_slice[i]
                };

                grad_input[start + i] =
                    inv_std * (dl_dx_hat_i - dl_dx_hat_mean - x_hat_i * dl_dx_hat_x_hat_mean);
            }
        }

        let grad_input_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;

        let grad_weight_out = if self.elementwise_affine && self.weight.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_weight),
                self.normalized_shape.clone(),
                false,
            )?)
        } else {
            None
        };

        let grad_bias_out = if self.elementwise_affine && self.bias.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_bias),
                self.normalized_shape.clone(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![
            Some(grad_input_tensor),
            grad_weight_out,
            grad_bias_out,
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.weight, &self.bias]
    }

    fn name(&self) -> &'static str {
        "LayerNormBackward"
    }
}

// ===========================================================================
// GroupNorm
// ===========================================================================

/// Group normalization.
///
/// Divides channels into groups and normalizes within each group.
/// For input of shape `[B, C, ...]`, divides `C` channels into `num_groups`
/// groups of `C / num_groups` channels each, and normalizes the values
/// within each group (over channels and spatial dimensions).
///
/// Matches `torch.nn.GroupNorm`.
#[derive(Debug)]
pub struct GroupNorm<T: Float> {
    /// Number of groups to divide channels into.
    pub num_groups: usize,
    /// Number of channels (expected C dimension).
    pub num_channels: usize,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Whether to apply learnable affine parameters.
    pub affine: bool,
    /// Learnable scale (gamma), shape = `[num_channels]`.
    pub weight: Parameter<T>,
    /// Learnable shift (beta), shape = `[num_channels]`.
    pub bias: Parameter<T>,
    training: bool,
}

impl<T: Float> GroupNorm<T> {
    /// Create a new `GroupNorm` layer.
    ///
    /// # Arguments
    ///
    /// * `num_groups` - Number of groups to divide channels into.
    /// * `num_channels` - Number of channels. Must be divisible by `num_groups`.
    /// * `eps` - Small constant for numerical stability (default: 1e-5).
    /// * `affine` - Whether to include learnable weight and bias.
    pub fn new(
        num_groups: usize,
        num_channels: usize,
        eps: f64,
        affine: bool,
    ) -> FerrotorchResult<Self> {
        if num_groups == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "num_groups must be positive".into(),
            });
        }
        if num_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "num_channels must be positive".into(),
            });
        }
        if num_channels % num_groups != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "num_channels ({num_channels}) must be divisible by num_groups ({num_groups})"
                ),
            });
        }

        let weight = Parameter::ones(&[num_channels])?;
        let bias = Parameter::zeros(&[num_channels])?;

        Ok(Self {
            num_groups,
            num_channels,
            eps,
            affine,
            weight,
            bias,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for GroupNorm<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() < 2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GroupNorm: input must have at least 2 dims [B, C, ...], got {:?}",
                    shape
                ),
            });
        }

        let batch_size = shape[0];
        let channels = shape[1];

        if channels != self.num_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GroupNorm: expected {} channels, got {}",
                    self.num_channels, channels
                ),
            });
        }

        let channels_per_group = channels / self.num_groups;
        // spatial_size = product of dims after C.
        let spatial_size: usize = shape[2..].iter().product();
        let spatial = spatial_size.max(1);
        let group_size = channels_per_group * spatial;

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "GroupNorm::forward",
            });
        }
        let input_data = input.data()?;
        let weight_data = self.weight.tensor().data()?;
        let bias_data = self.bias.tensor().data()?;
        let eps_t = T::from(self.eps).unwrap();
        let group_n = T::from(group_size).unwrap();

        let mut output = vec![zero::<T>(); input.numel()];

        for b in 0..batch_size {
            for g in 0..self.num_groups {
                let c_start = g * channels_per_group;
                let c_end = c_start + channels_per_group;

                // Compute mean over the group.
                let mut sum = zero::<T>();
                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        sum += input_data[idx];
                    }
                }
                let mean = sum / group_n;

                // Compute variance over the group.
                let mut var_sum = zero::<T>();
                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let d = input_data[idx] - mean;
                        var_sum += d * d;
                    }
                }
                let var = var_sum / group_n;
                let inv_std = (var + eps_t).sqrt().recip();

                // Normalize and apply per-channel affine.
                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let normed = (input_data[idx] - mean) * inv_std;
                        if self.affine {
                            output[idx] = normed * weight_data[c] + bias_data[c];
                        } else {
                            output[idx] = normed;
                        }
                    }
                }
            }
        }

        let result = Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(GroupNormBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.tensor().clone(),
                num_groups: self.num_groups,
                num_channels: self.num_channels,
                eps: self.eps,
                affine: self.affine,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        if self.affine {
            vec![&self.weight, &self.bias]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        if self.affine {
            vec![&mut self.weight, &mut self.bias]
        } else {
            vec![]
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        if self.affine {
            vec![
                ("weight".to_string(), &self.weight),
                ("bias".to_string(), &self.bias),
            ]
        } else {
            vec![]
        }
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// GroupNormBackward
// ---------------------------------------------------------------------------

/// Backward node for GroupNorm.
///
/// Inputs stored: `[input, weight, bias]`.
/// Returns: `[grad_input, grad_weight, grad_bias]`.
#[derive(Debug)]
struct GroupNormBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    num_groups: usize,
    num_channels: usize,
    eps: f64,
    affine: bool,
}

impl<T: Float> GradFn<T> for GroupNormBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.input.shape();
        let batch_size = shape[0];
        let channels = shape[1];
        let channels_per_group = channels / self.num_groups;
        let spatial_size: usize = shape[2..].iter().product();
        let spatial = spatial_size.max(1);
        let group_size = channels_per_group * spatial;
        let group_n = T::from(group_size).unwrap();
        let eps_t = T::from(self.eps).unwrap();

        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "GroupNormBackward",
            });
        }
        let input_data = self.input.data()?;
        let go_data = grad_output.data()?;
        let weight_data = self.weight.data()?;

        let mut grad_input = vec![zero::<T>(); self.input.numel()];
        let mut grad_weight = vec![zero::<T>(); self.num_channels];
        let mut grad_bias = vec![zero::<T>(); self.num_channels];

        for b in 0..batch_size {
            for g in 0..self.num_groups {
                let c_start = g * channels_per_group;
                let c_end = c_start + channels_per_group;

                // Recompute mean and inv_std for this group.
                let mut sum = zero::<T>();
                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        sum += input_data[idx];
                    }
                }
                let mean = sum / group_n;

                let mut var_sum = zero::<T>();
                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let d = input_data[idx] - mean;
                        var_sum += d * d;
                    }
                }
                let var = var_sum / group_n;
                let inv_std = (var + eps_t).sqrt().recip();

                // Compute sums for the VJP.
                let mut dl_dx_hat_sum = zero::<T>();
                let mut dl_dx_hat_x_hat_sum = zero::<T>();

                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let x_hat = (input_data[idx] - mean) * inv_std;
                        let dl_dx_hat = if self.affine {
                            go_data[idx] * weight_data[c]
                        } else {
                            go_data[idx]
                        };
                        dl_dx_hat_sum += dl_dx_hat;
                        dl_dx_hat_x_hat_sum += dl_dx_hat * x_hat;

                        if self.affine {
                            grad_weight[c] += go_data[idx] * x_hat;
                            grad_bias[c] += go_data[idx];
                        }
                    }
                }

                let dl_dx_hat_mean = dl_dx_hat_sum / group_n;
                let dl_dx_hat_x_hat_mean = dl_dx_hat_x_hat_sum / group_n;

                for (ci, &wd) in weight_data[c_start..c_end].iter().enumerate() {
                    let c = c_start + ci;
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let x_hat = (input_data[idx] - mean) * inv_std;
                        let dl_dx_hat = if self.affine {
                            go_data[idx] * wd
                        } else {
                            go_data[idx]
                        };
                        grad_input[idx] =
                            inv_std * (dl_dx_hat - dl_dx_hat_mean - x_hat * dl_dx_hat_x_hat_mean);
                    }
                }
            }
        }

        let grad_input_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;

        let grad_weight_out = if self.affine && self.weight.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_weight),
                vec![self.num_channels],
                false,
            )?)
        } else {
            None
        };

        let grad_bias_out = if self.affine && self.bias.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_bias),
                vec![self.num_channels],
                false,
            )?)
        } else {
            None
        };

        Ok(vec![
            Some(grad_input_tensor),
            grad_weight_out,
            grad_bias_out,
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.weight, &self.bias]
    }

    fn name(&self) -> &'static str {
        "GroupNormBackward"
    }
}

// ===========================================================================
// RMSNorm
// ===========================================================================

/// Root Mean Square Layer Normalization.
///
/// Applies the transform:
///
/// ```text
/// y = x / sqrt(mean(x^2) + eps) * weight
/// ```
///
/// Unlike LayerNorm, RMSNorm does not center the input (no mean subtraction)
/// and has no bias parameter. This makes it slightly faster and is used in
/// many modern transformer architectures (LLaMA, Gemma, etc.).
///
/// Matches the RMSNorm formulation from "Root Mean Square Layer Normalization"
/// (Zhang & Sennrich, 2019).
#[derive(Debug)]
pub struct RMSNorm<T: Float> {
    /// The size of the normalized dimension.
    pub normalized_shape: Vec<usize>,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Learnable scale (gamma), shape = `normalized_shape`.
    pub weight: Parameter<T>,
    training: bool,
}

impl<T: Float> RMSNorm<T> {
    /// Create a new `RMSNorm` layer.
    ///
    /// # Arguments
    ///
    /// * `normalized_shape` - The shape of the dimensions to normalize over.
    /// * `eps` - Small constant for numerical stability (default: 1e-5).
    pub fn new(normalized_shape: Vec<usize>, eps: f64) -> FerrotorchResult<Self> {
        if normalized_shape.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "normalized_shape must not be empty".into(),
            });
        }

        let weight = Parameter::ones(&normalized_shape)?;

        Ok(Self {
            normalized_shape,
            eps,
            weight,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for RMSNorm<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        let ndim = shape.len();
        let norm_ndim = self.normalized_shape.len();

        if ndim < norm_ndim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "RMSNorm: input has {} dims but normalized_shape has {} dims",
                    ndim, norm_ndim
                ),
            });
        }

        let last_dims = &shape[ndim - norm_ndim..];
        if last_dims != self.normalized_shape.as_slice() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "RMSNorm: input last dims {:?} don't match normalized_shape {:?}",
                    last_dims, self.normalized_shape
                ),
            });
        }

        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size = input.numel() / norm_size;

        // GPU fast path: native RMSNorm kernel.
        if input.is_cuda() {
            if let Some(backend) = ferrotorch_core::gpu_dispatch::gpu_backend() {
                let eps_f32 = self.eps as f32;
                let handle = backend.rmsnorm_f32(
                    input.gpu_handle()?,
                    self.weight.tensor().gpu_handle()?,
                    batch_size,
                    norm_size,
                    eps_f32,
                )?;
                return if is_grad_enabled() && input.requires_grad() {
                    let grad_fn = Arc::new(RMSNormBackward {
                        input: input.clone(),
                        weight: self.weight.tensor().clone(),
                        normalized_shape: self.normalized_shape.clone(),
                        eps: self.eps,
                    });
                    Tensor::from_operation(TensorStorage::gpu(handle), shape, grad_fn)
                } else {
                    Tensor::from_storage(TensorStorage::gpu(handle), shape, false)
                };
            }
        }

        // CPU path — CUDA inputs without a GPU backend are rejected above.
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "RMSNorm::forward",
            });
        }
        let input_data = input.data()?;
        let weight_data = self.weight.tensor().data()?;
        let eps_t = T::from(self.eps).unwrap();
        let n_t = T::from(norm_size).unwrap();

        // bf16 has a 7-bit mantissa; a mean-of-squares over hundreds of
        // elements saturates the accumulator and collapses into near-
        // constant outputs. Detect bf16 and promote the accumulator
        // (and the eps / normalization) to f32.
        let is_bf16 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>();
        let mut output = Vec::with_capacity(input.numel());

        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let slice = &input_data[start..end];

            if is_bf16 {
                // f32 accumulator path for bf16.
                let eps_f32 = self.eps as f32;
                let n_f32 = norm_size as f32;
                let mut sum_sq = 0.0f32;
                for &x in slice {
                    let xf = x.to_f32().unwrap();
                    sum_sq += xf * xf;
                }
                let inv_rms_f32 = 1.0f32 / ((sum_sq / n_f32) + eps_f32).sqrt();
                let inv_rms = T::from(inv_rms_f32).unwrap();
                for (i, &x) in slice.iter().enumerate() {
                    output.push(x * inv_rms * weight_data[i]);
                }
            } else {
                // rms = sqrt(mean(x^2) + eps)
                let mean_sq = slice.iter().copied().fold(zero::<T>(), |a, x| a + x * x) / n_t;
                let rms = (mean_sq + eps_t).sqrt();
                let inv_rms = rms.recip();

                for (i, &x) in slice.iter().enumerate() {
                    output.push(x * inv_rms * weight_data[i]);
                }
            }
        }

        let result = Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(RMSNormBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                normalized_shape: self.normalized_shape.clone(),
                eps: self.eps,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.weight]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.weight]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![("weight".to_string(), &self.weight)]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// RMSNormBackward
// ---------------------------------------------------------------------------

/// Backward node for RMSNorm.
///
/// Forward: `y = x / rms * weight` where `rms = sqrt(mean(x^2) + eps)`.
///
/// Let `s = 1/rms`. Then `y_i = x_i * s * w_i`.
///
/// `ds/dx_j = -x_j / (n * rms^3)`
///
/// `dy_i/dx_j = delta_ij * s * w_i + x_i * w_i * ds/dx_j`
///            = `delta_ij * s * w_i - x_i * w_i * x_j / (n * rms^3)`
///
/// `grad_x_j = sum_i go_i * dy_i/dx_j`
///           = `go_j * s * w_j - (1/(n * rms^3)) * x_j * sum_i(go_i * x_i * w_i)`
///           = `s * (go_j * w_j - x_j * s^2 * mean(go * x * w))`
///
/// Inputs stored: `[input, weight]`.
/// Returns: `[grad_input, grad_weight]`.
#[derive(Debug)]
struct RMSNormBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    normalized_shape: Vec<usize>,
    eps: f64,
}

impl<T: Float> GradFn<T> for RMSNormBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let norm_size: usize = self.normalized_shape.iter().product();
        let batch_size = self.input.numel() / norm_size;

        // GPU-native fast path for f32/f64
        if self.input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            if let Some(backend) = gpu_backend() {
                let (gi_h, gw_h) = if is_f64::<T>() {
                    backend.rmsnorm_backward_f64(
                        self.input.gpu_handle()?,
                        grad_output.gpu_handle()?,
                        self.weight.gpu_handle()?,
                        batch_size,
                        norm_size,
                        self.eps,
                    )?
                } else {
                    backend.rmsnorm_backward_f32(
                        self.input.gpu_handle()?,
                        grad_output.gpu_handle()?,
                        self.weight.gpu_handle()?,
                        batch_size,
                        norm_size,
                        self.eps as f32,
                    )?
                };

                let grad_input_tensor = Tensor::from_storage(
                    TensorStorage::gpu(gi_h),
                    self.input.shape().to_vec(),
                    false,
                )?;

                let grad_weight_out = if self.weight.requires_grad() {
                    Some(Tensor::from_storage(
                        TensorStorage::gpu(gw_h),
                        self.normalized_shape.clone(),
                        false,
                    )?)
                } else {
                    None
                };

                return Ok(vec![Some(grad_input_tensor), grad_weight_out]);
            }
        }

        // CPU-only path — CUDA inputs without a GPU backend are rejected above.
        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "RMSNormBackward",
            });
        }
        let n_t = T::from(norm_size).unwrap();
        let eps_t = T::from(self.eps).unwrap();

        let input_data = self.input.data()?;
        let go_data = grad_output.data()?;
        let weight_data = self.weight.data()?;

        let mut grad_input = vec![zero::<T>(); self.input.numel()];
        let mut grad_weight = vec![zero::<T>(); norm_size];

        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let x_slice = &input_data[start..end];
            let go_slice = &go_data[start..end];

            // Recompute rms.
            let mean_sq = x_slice.iter().copied().fold(zero::<T>(), |a, x| a + x * x) / n_t;
            let rms = (mean_sq + eps_t).sqrt();
            let inv_rms = rms.recip();
            let inv_rms_sq = inv_rms * inv_rms;

            // sum_i(go_i * x_i * w_i) / n
            let go_x_w_mean = x_slice
                .iter()
                .zip(go_slice.iter())
                .zip(weight_data.iter())
                .fold(zero::<T>(), |a, ((&x, &go), &w)| a + go * x * w)
                / n_t;

            for i in 0..norm_size {
                // grad_x_j = inv_rms * (go_j * w_j - x_j * inv_rms^2 * go_x_w_mean)
                grad_input[start + i] = inv_rms
                    * (go_slice[i] * weight_data[i] - x_slice[i] * inv_rms_sq * go_x_w_mean);

                // grad_weight_i += go_i * x_i * inv_rms
                grad_weight[i] += go_slice[i] * x_slice[i] * inv_rms;
            }
        }

        let grad_input_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;

        let grad_weight_out = if self.weight.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_weight),
                self.normalized_shape.clone(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![Some(grad_input_tensor), grad_weight_out])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.weight]
    }

    fn name(&self) -> &'static str {
        "RMSNormBackward"
    }
}

// ===========================================================================
// BatchNorm2d
// ===========================================================================

/// Batch normalization over 4D inputs (a mini-batch of 2D inputs with an
/// additional channel dimension).
///
/// Applies the transform per channel:
///
/// ```text
/// y = (x - mean) / sqrt(var + eps) * weight + bias
/// ```
///
/// During **training**, `mean` and `var` are computed from the current
/// mini-batch over the `(B, H, W)` dimensions, and exponential moving
/// averages of these statistics are maintained in `running_mean` and
/// `running_var`.
///
/// During **evaluation**, the accumulated `running_mean` and `running_var`
/// are used instead of batch statistics.
///
/// Matches `torch.nn.BatchNorm2d`.
pub struct BatchNorm2d<T: Float> {
    /// Number of channels (features) `C`.
    pub num_features: usize,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Momentum for the running mean / variance update
    /// (`running = (1 - momentum) * running + momentum * batch`).
    pub momentum: f64,
    /// Whether to apply a learnable affine transform.
    pub affine: bool,
    /// Learnable scale (gamma), shape `[C]`. `None` when `affine == false`.
    pub weight: Option<Parameter<T>>,
    /// Learnable shift (beta), shape `[C]`. `None` when `affine == false`.
    pub bias: Option<Parameter<T>>,
    /// Exponential moving average of per-channel means.
    /// Uses `Mutex` for interior mutability because `Module::forward` takes `&self`
    /// and `Module` requires `Send + Sync`.
    running_mean: Mutex<Vec<f64>>,
    /// Exponential moving average of per-channel variances.
    running_var: Mutex<Vec<f64>>,
    /// Number of forward calls in training mode (for tracking).
    num_batches_tracked: Mutex<usize>,
    /// Whether the layer is in training mode.
    training: Mutex<bool>,
}

// Manual Debug because Mutex doesn't derive Debug in all contexts nicely.
impl<T: Float> std::fmt::Debug for BatchNorm2d<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchNorm2d")
            .field("num_features", &self.num_features)
            .field("eps", &self.eps)
            .field("momentum", &self.momentum)
            .field("affine", &self.affine)
            .field("weight", &self.weight)
            .field("bias", &self.bias)
            .field("training", &self.training)
            .finish()
    }
}

impl<T: Float> BatchNorm2d<T> {
    /// Create a new `BatchNorm2d` layer.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of channels `C`.
    /// * `eps` - Numerical stability constant (default: `1e-5`).
    /// * `momentum` - Running-statistics momentum (default: `0.1`).
    /// * `affine` - Whether to include learnable weight and bias.
    pub fn new(
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
    ) -> FerrotorchResult<Self> {
        if num_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "num_features must be positive".into(),
            });
        }

        let weight = if affine {
            Some(Parameter::ones(&[num_features])?)
        } else {
            None
        };

        let bias = if affine {
            Some(Parameter::zeros(&[num_features])?)
        } else {
            None
        };

        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            weight,
            bias,
            running_mean: Mutex::new(vec![0.0; num_features]),
            running_var: Mutex::new(vec![1.0; num_features]),
            num_batches_tracked: Mutex::new(0),
            training: Mutex::new(true),
        })
    }

    /// Access the current running mean (snapshot copy).
    pub fn running_mean(&self) -> Vec<f64> {
        self.running_mean.lock().unwrap().clone()
    }

    /// Access the current running variance (snapshot copy).
    pub fn running_var(&self) -> Vec<f64> {
        self.running_var.lock().unwrap().clone()
    }

    /// Number of training batches tracked so far.
    pub fn num_batches_tracked(&self) -> usize {
        *self.num_batches_tracked.lock().unwrap()
    }
}

impl<T: Float> Module<T> for BatchNorm2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 4 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BatchNorm2d: expected 4D input [B, C, H, W], got {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let spatial = height * width;

        if channels != self.num_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BatchNorm2d: expected {} channels, got {}",
                    self.num_features, channels
                ),
            });
        }

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "BatchNorm2d::forward",
            });
        }
        let input_data = input.data()?;
        let eps_t = T::from(self.eps).unwrap();

        let weight_data = self.weight.as_ref().map(|w| w.tensor().data().unwrap());
        let bias_data = self.bias.as_ref().map(|b| b.tensor().data().unwrap());

        let is_training = *self.training.lock().unwrap();

        // Per-channel mean and variance (as T for computation).
        let mut chan_mean = vec![zero::<T>(); channels];
        let mut chan_var = vec![zero::<T>(); channels];

        if is_training {
            // Compute batch statistics over (B, H, W).
            let count = batch * spatial;
            let count_t = T::from(count).unwrap();

            for c in 0..channels {
                let mut sum = zero::<T>();
                for b in 0..batch {
                    let base = b * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        sum += input_data[base + s];
                    }
                }
                chan_mean[c] = sum / count_t;

                let mut var_sum = zero::<T>();
                for b in 0..batch {
                    let base = b * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        let d = input_data[base + s] - chan_mean[c];
                        var_sum += d * d;
                    }
                }
                // Biased variance (like PyTorch).
                chan_var[c] = var_sum / count_t;
            }

            // Update running statistics.
            {
                let mut rm = self.running_mean.lock().unwrap();
                let mut rv = self.running_var.lock().unwrap();
                let mut nbt = self.num_batches_tracked.lock().unwrap();
                *nbt += 1;

                let mom = self.momentum;
                // For running_var, PyTorch uses unbiased (Bessel-corrected)
                // variance in the running update.
                let bessel = if count > 1 {
                    count as f64 / (count as f64 - 1.0)
                } else {
                    1.0
                };

                for c in 0..channels {
                    let batch_mean_f64 = chan_mean[c].to_f64().unwrap();
                    let batch_var_f64 = chan_var[c].to_f64().unwrap();

                    rm[c] = (1.0 - mom) * rm[c] + mom * batch_mean_f64;
                    rv[c] = (1.0 - mom) * rv[c] + mom * batch_var_f64 * bessel;
                }
            }
        } else {
            // Eval mode: use running statistics.
            let rm = self.running_mean.lock().unwrap();
            let rv = self.running_var.lock().unwrap();

            for c in 0..channels {
                chan_mean[c] = T::from(rm[c]).unwrap();
                chan_var[c] = T::from(rv[c]).unwrap();
            }
        }

        // Normalize and optionally scale/shift.
        let mut output = vec![zero::<T>(); input.numel()];

        // Pre-compute inv_std per channel.
        let mut inv_std = vec![zero::<T>(); channels];
        // Also store x_hat for the backward pass if needed.
        let mut x_hat_data = if is_grad_enabled() && input.requires_grad() {
            Vec::with_capacity(input.numel())
        } else {
            Vec::new()
        };
        let need_x_hat = is_grad_enabled() && input.requires_grad();

        for c in 0..channels {
            inv_std[c] = (chan_var[c] + eps_t).sqrt().recip();
        }

        for b in 0..batch {
            for c in 0..channels {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let normed = (input_data[idx] - chan_mean[c]) * inv_std[c];

                    if need_x_hat {
                        x_hat_data.push(normed);
                    }

                    if self.affine {
                        let w = weight_data.as_ref().unwrap();
                        let bi = bias_data.as_ref().unwrap();
                        output[idx] = normed * w[c] + bi[c];
                    } else {
                        output[idx] = normed;
                    }
                }
            }
        }

        let result = Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?;

        if is_grad_enabled() && input.requires_grad() {
            let weight_tensor = self.weight.as_ref().map(|w| w.tensor().clone());
            let bias_tensor = self.bias.as_ref().map(|b| b.tensor().clone());

            let grad_fn = Arc::new(BatchNorm2dBackward {
                input: input.clone(),
                x_hat: Tensor::from_storage(TensorStorage::cpu(x_hat_data), shape.to_vec(), false)?,
                weight: weight_tensor,
                bias: bias_tensor,
                chan_var: chan_var.iter().map(|v| v.to_f64().unwrap()).collect(),
                eps: self.eps,
                affine: self.affine,
            });

            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => vec![w, b],
            _ => vec![],
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        match (&mut self.weight, &mut self.bias) {
            (Some(w), Some(b)) => vec![w, b],
            _ => vec![],
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => vec![("weight".to_string(), w), ("bias".to_string(), b)],
            _ => vec![],
        }
    }

    fn train(&mut self) {
        *self.training.lock().unwrap() = true;
    }

    fn eval(&mut self) {
        *self.training.lock().unwrap() = false;
    }

    fn is_training(&self) -> bool {
        *self.training.lock().unwrap()
    }
}

// ---------------------------------------------------------------------------
// BatchNorm2dBackward
// ---------------------------------------------------------------------------

/// Backward node for `BatchNorm2d`.
///
/// Given the forward:
///
/// ```text
/// x_hat = (x - mean) / sqrt(var + eps)
/// y = weight * x_hat + bias          (if affine)
/// ```
///
/// The gradients are:
///
/// - `grad_bias[c]  = sum(grad_output[:, c, :, :])` over `(B, H, W)`
/// - `grad_weight[c] = sum(grad_output[:, c, :, :] * x_hat[:, c, :, :])` over `(B, H, W)`
/// - `grad_input`:
///   ```text
///   dl_dx_hat = grad_output * weight              (if affine, else grad_output)
///   grad_input = (1 / sqrt(var + eps)) *
///       (dl_dx_hat - mean(dl_dx_hat) - x_hat * mean(dl_dx_hat * x_hat))
///   ```
///   where the means are taken over `(B, H, W)`.
///
/// Inputs stored: `[input, weight?, bias?]`.
/// Returns: `[grad_input, grad_weight?, grad_bias?]`.
#[derive(Debug)]
struct BatchNorm2dBackward<T: Float> {
    input: Tensor<T>,
    /// Pre-computed normalized values `(x - mean) / sqrt(var + eps)`.
    x_hat: Tensor<T>,
    weight: Option<Tensor<T>>,
    bias: Option<Tensor<T>>,
    /// Per-channel batch variances (biased).
    chan_var: Vec<f64>,
    eps: f64,
    affine: bool,
}

impl<T: Float> GradFn<T> for BatchNorm2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let height = shape[2];
        let width = shape[3];
        let spatial = height * width;
        let count = batch * spatial;
        let count_t = T::from(count).unwrap();

        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "BatchNorm2dBackward",
            });
        }
        let go_data = grad_output.data()?;
        let x_hat_data = self.x_hat.data()?;

        let weight_data = self.weight.as_ref().map(|w| w.data().unwrap().to_vec());

        let mut grad_input = vec![zero::<T>(); self.input.numel()];
        let mut grad_weight = vec![zero::<T>(); channels];
        let mut grad_bias = vec![zero::<T>(); channels];

        for c in 0..channels {
            let var_f64 = self.chan_var[c];
            let inv_std = T::from(1.0 / (var_f64 + self.eps).sqrt()).unwrap();

            // First pass: accumulate sums for the VJP.
            let mut dl_dx_hat_sum = zero::<T>();
            let mut dl_dx_hat_x_hat_sum = zero::<T>();

            for b in 0..batch {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let x_h = x_hat_data[idx];
                    let go = go_data[idx];

                    let dl_dx_hat = if self.affine {
                        go * weight_data.as_ref().unwrap()[c]
                    } else {
                        go
                    };

                    dl_dx_hat_sum += dl_dx_hat;
                    dl_dx_hat_x_hat_sum += dl_dx_hat * x_h;

                    if self.affine {
                        grad_weight[c] += go * x_h;
                        grad_bias[c] += go;
                    }
                }
            }

            let dl_dx_hat_mean = dl_dx_hat_sum / count_t;
            let dl_dx_hat_x_hat_mean = dl_dx_hat_x_hat_sum / count_t;

            // Second pass: compute grad_input.
            for b in 0..batch {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let x_h = x_hat_data[idx];
                    let go = go_data[idx];

                    let dl_dx_hat = if self.affine {
                        go * weight_data.as_ref().unwrap()[c]
                    } else {
                        go
                    };

                    grad_input[idx] =
                        inv_std * (dl_dx_hat - dl_dx_hat_mean - x_h * dl_dx_hat_x_hat_mean);
                }
            }
        }

        let grad_input_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;

        let grad_weight_out = if self.affine {
            if let Some(ref w) = self.weight {
                if w.requires_grad() {
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(grad_weight),
                        vec![channels],
                        false,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let grad_bias_out = if self.affine {
            if let Some(ref b) = self.bias {
                if b.requires_grad() {
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(grad_bias),
                        vec![channels],
                        false,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(vec![
            Some(grad_input_tensor),
            grad_weight_out,
            grad_bias_out,
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v: Vec<&Tensor<T>> = vec![&self.input];
        if let Some(ref w) = self.weight {
            v.push(w);
        }
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "BatchNorm2dBackward"
    }
}

// ===========================================================================
// BatchNorm1d
// ===========================================================================

/// Batch normalization for 2D input `[N, C]` or 3D input `[N, C, L]`.
///
/// Applies per-channel normalization:
///
/// ```text
/// y = (x - mean) / sqrt(var + eps) * weight + bias
/// ```
///
/// During **training**, `mean` and `var` are computed from the current
/// mini-batch over the `(N,)` or `(N, L)` dimensions, and exponential
/// moving averages are maintained in `running_mean` and `running_var`.
///
/// During **evaluation**, the accumulated `running_mean` and `running_var`
/// are used instead of batch statistics.
///
/// Matches `torch.nn.BatchNorm1d`.
pub struct BatchNorm1d<T: Float> {
    /// Number of channels (features) `C`.
    pub num_features: usize,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Momentum for the running mean / variance update
    /// (`running = (1 - momentum) * running + momentum * batch`).
    pub momentum: f64,
    /// Whether to apply a learnable affine transform.
    pub affine: bool,
    /// Learnable scale (gamma), shape `[C]`. `None` when `affine == false`.
    pub weight: Option<Parameter<T>>,
    /// Learnable shift (beta), shape `[C]`. `None` when `affine == false`.
    pub bias: Option<Parameter<T>>,
    /// Exponential moving average of per-channel means.
    running_mean: Mutex<Vec<f64>>,
    /// Exponential moving average of per-channel variances.
    running_var: Mutex<Vec<f64>>,
    /// Number of forward calls in training mode.
    num_batches_tracked: Mutex<usize>,
    /// Whether the layer is in training mode.
    training: Mutex<bool>,
}

impl<T: Float> std::fmt::Debug for BatchNorm1d<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchNorm1d")
            .field("num_features", &self.num_features)
            .field("eps", &self.eps)
            .field("momentum", &self.momentum)
            .field("affine", &self.affine)
            .field("weight", &self.weight)
            .field("bias", &self.bias)
            .field("training", &self.training)
            .finish()
    }
}

impl<T: Float> BatchNorm1d<T> {
    /// Create a new `BatchNorm1d` layer.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of channels `C`.
    /// * `eps` - Numerical stability constant (default: `1e-5`).
    /// * `momentum` - Running-statistics momentum (default: `0.1`).
    /// * `affine` - Whether to include learnable weight and bias.
    pub fn new(
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
    ) -> FerrotorchResult<Self> {
        if num_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "BatchNorm1d: num_features must be positive".into(),
            });
        }

        let weight = if affine {
            Some(Parameter::ones(&[num_features])?)
        } else {
            None
        };

        let bias = if affine {
            Some(Parameter::zeros(&[num_features])?)
        } else {
            None
        };

        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            weight,
            bias,
            running_mean: Mutex::new(vec![0.0; num_features]),
            running_var: Mutex::new(vec![1.0; num_features]),
            num_batches_tracked: Mutex::new(0),
            training: Mutex::new(true),
        })
    }

    /// Access the current running mean (snapshot copy).
    pub fn running_mean(&self) -> Vec<f64> {
        self.running_mean.lock().unwrap().clone()
    }

    /// Access the current running variance (snapshot copy).
    pub fn running_var(&self) -> Vec<f64> {
        self.running_var.lock().unwrap().clone()
    }

    /// Number of training batches tracked so far.
    pub fn num_batches_tracked(&self) -> usize {
        *self.num_batches_tracked.lock().unwrap()
    }
}

impl<T: Float> Module<T> for BatchNorm1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        let ndim = shape.len();

        // Accept 2D [N, C] or 3D [N, C, L].
        if ndim != 2 && ndim != 3 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BatchNorm1d: expected 2D [N, C] or 3D [N, C, L] input, got {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let length = if ndim == 3 { shape[2] } else { 1 };

        if channels != self.num_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BatchNorm1d: expected {} channels, got {}",
                    self.num_features, channels
                ),
            });
        }

        // Edge case: batch size 0.
        if batch == 0 {
            return Ok(input.clone());
        }

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "BatchNorm1d::forward",
            });
        }
        let input_data = input.data()?;
        let eps_t = T::from(self.eps).unwrap();

        let weight_data = self.weight.as_ref().map(|w| w.tensor().data().unwrap());
        let bias_data = self.bias.as_ref().map(|b| b.tensor().data().unwrap());

        let is_training = *self.training.lock().unwrap();

        let mut chan_mean = vec![zero::<T>(); channels];
        let mut chan_var = vec![zero::<T>(); channels];

        if is_training {
            let count = batch * length;
            let count_t = T::from(count).unwrap();

            for c in 0..channels {
                let mut s = zero::<T>();
                for b in 0..batch {
                    let base = b * channels * length + c * length;
                    for l in 0..length {
                        s += input_data[base + l];
                    }
                }
                chan_mean[c] = s / count_t;

                let mut var_sum = zero::<T>();
                for b in 0..batch {
                    let base = b * channels * length + c * length;
                    for l in 0..length {
                        let d = input_data[base + l] - chan_mean[c];
                        var_sum += d * d;
                    }
                }
                chan_var[c] = var_sum / count_t;
            }

            // Update running statistics.
            {
                let mut rm = self.running_mean.lock().unwrap();
                let mut rv = self.running_var.lock().unwrap();
                let mut nbt = self.num_batches_tracked.lock().unwrap();
                *nbt += 1;

                let mom = self.momentum;
                let bessel = if count > 1 {
                    count as f64 / (count as f64 - 1.0)
                } else {
                    1.0
                };

                for c in 0..channels {
                    let batch_mean_f64 = chan_mean[c].to_f64().unwrap();
                    let batch_var_f64 = chan_var[c].to_f64().unwrap();

                    rm[c] = (1.0 - mom) * rm[c] + mom * batch_mean_f64;
                    rv[c] = (1.0 - mom) * rv[c] + mom * batch_var_f64 * bessel;
                }
            }
        } else {
            let rm = self.running_mean.lock().unwrap();
            let rv = self.running_var.lock().unwrap();

            for c in 0..channels {
                chan_mean[c] = T::from(rm[c]).unwrap();
                chan_var[c] = T::from(rv[c]).unwrap();
            }
        }

        let mut output = vec![zero::<T>(); input.numel()];

        let mut inv_std = vec![zero::<T>(); channels];
        let need_x_hat = is_grad_enabled() && input.requires_grad();
        let mut x_hat_data = if need_x_hat {
            Vec::with_capacity(input.numel())
        } else {
            Vec::new()
        };

        for c in 0..channels {
            inv_std[c] = (chan_var[c] + eps_t).sqrt().recip();
        }

        for b in 0..batch {
            for c in 0..channels {
                let base = b * channels * length + c * length;
                for l in 0..length {
                    let idx = base + l;
                    let normed = (input_data[idx] - chan_mean[c]) * inv_std[c];

                    if need_x_hat {
                        x_hat_data.push(normed);
                    }

                    if self.affine {
                        let w = weight_data.as_ref().unwrap();
                        let bi = bias_data.as_ref().unwrap();
                        output[idx] = normed * w[c] + bi[c];
                    } else {
                        output[idx] = normed;
                    }
                }
            }
        }

        let result = Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?;

        if is_grad_enabled() && input.requires_grad() {
            let weight_tensor = self.weight.as_ref().map(|w| w.tensor().clone());
            let bias_tensor = self.bias.as_ref().map(|b| b.tensor().clone());

            let grad_fn = Arc::new(BatchNorm1dBackward {
                input: input.clone(),
                x_hat: Tensor::from_storage(TensorStorage::cpu(x_hat_data), shape.to_vec(), false)?,
                weight: weight_tensor,
                bias: bias_tensor,
                chan_var: chan_var.iter().map(|v| v.to_f64().unwrap()).collect(),
                eps: self.eps,
                affine: self.affine,
            });

            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => vec![w, b],
            _ => vec![],
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        match (&mut self.weight, &mut self.bias) {
            (Some(w), Some(b)) => vec![w, b],
            _ => vec![],
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => vec![("weight".to_string(), w), ("bias".to_string(), b)],
            _ => vec![],
        }
    }

    fn train(&mut self) {
        *self.training.lock().unwrap() = true;
    }

    fn eval(&mut self) {
        *self.training.lock().unwrap() = false;
    }

    fn is_training(&self) -> bool {
        *self.training.lock().unwrap()
    }
}

// ---------------------------------------------------------------------------
// BatchNorm1dBackward
// ---------------------------------------------------------------------------

/// Backward node for `BatchNorm1d`.
///
/// Same math as `BatchNorm2dBackward` but over `(N,)` or `(N, L)` spatial dims
/// instead of `(N, H, W)`.
#[derive(Debug)]
struct BatchNorm1dBackward<T: Float> {
    input: Tensor<T>,
    x_hat: Tensor<T>,
    weight: Option<Tensor<T>>,
    bias: Option<Tensor<T>>,
    chan_var: Vec<f64>,
    eps: f64,
    affine: bool,
}

impl<T: Float> GradFn<T> for BatchNorm1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.input.shape();
        let ndim = shape.len();
        let batch = shape[0];
        let channels = shape[1];
        let length = if ndim == 3 { shape[2] } else { 1 };
        let count = batch * length;
        let count_t = T::from(count).unwrap();

        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "BatchNorm1dBackward",
            });
        }
        let go_data = grad_output.data()?;
        let x_hat_data = self.x_hat.data()?;

        let weight_data = self.weight.as_ref().map(|w| w.data().unwrap().to_vec());

        let mut grad_input = vec![zero::<T>(); self.input.numel()];
        let mut grad_weight = vec![zero::<T>(); channels];
        let mut grad_bias = vec![zero::<T>(); channels];

        for c in 0..channels {
            let var_f64 = self.chan_var[c];
            let inv_std = T::from(1.0 / (var_f64 + self.eps).sqrt()).unwrap();

            let mut dl_dx_hat_sum = zero::<T>();
            let mut dl_dx_hat_x_hat_sum = zero::<T>();

            for b in 0..batch {
                let base = b * channels * length + c * length;
                for l in 0..length {
                    let idx = base + l;
                    let x_h = x_hat_data[idx];
                    let go = go_data[idx];

                    let dl_dx_hat = if self.affine {
                        go * weight_data.as_ref().unwrap()[c]
                    } else {
                        go
                    };

                    dl_dx_hat_sum += dl_dx_hat;
                    dl_dx_hat_x_hat_sum += dl_dx_hat * x_h;

                    if self.affine {
                        grad_weight[c] += go * x_h;
                        grad_bias[c] += go;
                    }
                }
            }

            let dl_dx_hat_mean = dl_dx_hat_sum / count_t;
            let dl_dx_hat_x_hat_mean = dl_dx_hat_x_hat_sum / count_t;

            for b in 0..batch {
                let base = b * channels * length + c * length;
                for l in 0..length {
                    let idx = base + l;
                    let x_h = x_hat_data[idx];
                    let go = go_data[idx];

                    let dl_dx_hat = if self.affine {
                        go * weight_data.as_ref().unwrap()[c]
                    } else {
                        go
                    };

                    grad_input[idx] =
                        inv_std * (dl_dx_hat - dl_dx_hat_mean - x_h * dl_dx_hat_x_hat_mean);
                }
            }
        }

        let grad_input_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;

        let grad_weight_out = if self.affine {
            if let Some(ref w) = self.weight {
                if w.requires_grad() {
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(grad_weight),
                        vec![channels],
                        false,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let grad_bias_out = if self.affine {
            if let Some(ref b) = self.bias {
                if b.requires_grad() {
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(grad_bias),
                        vec![channels],
                        false,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(vec![
            Some(grad_input_tensor),
            grad_weight_out,
            grad_bias_out,
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v: Vec<&Tensor<T>> = vec![&self.input];
        if let Some(ref w) = self.weight {
            v.push(w);
        }
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "BatchNorm1dBackward"
    }
}

// ===========================================================================
// BatchNorm3d — CL-434
// ===========================================================================

/// Batch normalization over 5D inputs (a mini-batch of 3D inputs with an
/// additional channel dimension).
///
/// Applies the transform per channel:
///
/// ```text
/// y = (x - mean) / sqrt(var + eps) * weight + bias
/// ```
///
/// During **training**, `mean` and `var` are computed from the current
/// mini-batch over the `(B, D, H, W)` dimensions, and exponential moving
/// averages of these statistics are maintained in `running_mean` and
/// `running_var`.
///
/// During **evaluation**, the accumulated `running_mean` and `running_var`
/// are used instead of batch statistics.
///
/// Matches `torch.nn.BatchNorm3d`.
pub struct BatchNorm3d<T: Float> {
    /// Number of channels (features) `C`.
    pub num_features: usize,
    /// Small constant for numerical stability.
    pub eps: f64,
    /// Momentum for the running mean / variance update
    /// (`running = (1 - momentum) * running + momentum * batch`).
    pub momentum: f64,
    /// Whether to apply a learnable affine transform.
    pub affine: bool,
    /// Learnable scale (gamma), shape `[C]`. `None` when `affine == false`.
    pub weight: Option<Parameter<T>>,
    /// Learnable shift (beta), shape `[C]`. `None` when `affine == false`.
    pub bias: Option<Parameter<T>>,
    /// Exponential moving average of per-channel means.
    running_mean: Mutex<Vec<f64>>,
    /// Exponential moving average of per-channel variances.
    running_var: Mutex<Vec<f64>>,
    /// Number of forward calls in training mode.
    num_batches_tracked: Mutex<usize>,
    /// Whether the layer is in training mode.
    training: Mutex<bool>,
}

impl<T: Float> std::fmt::Debug for BatchNorm3d<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchNorm3d")
            .field("num_features", &self.num_features)
            .field("eps", &self.eps)
            .field("momentum", &self.momentum)
            .field("affine", &self.affine)
            .field("weight", &self.weight)
            .field("bias", &self.bias)
            .field("training", &self.training)
            .finish()
    }
}

impl<T: Float> BatchNorm3d<T> {
    /// Create a new `BatchNorm3d` layer.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of channels `C`.
    /// * `eps` - Numerical stability constant (default: `1e-5`).
    /// * `momentum` - Running-statistics momentum (default: `0.1`).
    /// * `affine` - Whether to include learnable weight and bias.
    pub fn new(
        num_features: usize,
        eps: f64,
        momentum: f64,
        affine: bool,
    ) -> FerrotorchResult<Self> {
        if num_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "BatchNorm3d: num_features must be positive".into(),
            });
        }

        let weight = if affine {
            Some(Parameter::ones(&[num_features])?)
        } else {
            None
        };

        let bias = if affine {
            Some(Parameter::zeros(&[num_features])?)
        } else {
            None
        };

        Ok(Self {
            num_features,
            eps,
            momentum,
            affine,
            weight,
            bias,
            running_mean: Mutex::new(vec![0.0; num_features]),
            running_var: Mutex::new(vec![1.0; num_features]),
            num_batches_tracked: Mutex::new(0),
            training: Mutex::new(true),
        })
    }

    /// Access the current running mean (snapshot copy).
    pub fn running_mean(&self) -> Vec<f64> {
        self.running_mean.lock().unwrap().clone()
    }

    /// Access the current running variance (snapshot copy).
    pub fn running_var(&self) -> Vec<f64> {
        self.running_var.lock().unwrap().clone()
    }

    /// Number of training batches tracked so far.
    pub fn num_batches_tracked(&self) -> usize {
        *self.num_batches_tracked.lock().unwrap()
    }
}

impl<T: Float> Module<T> for BatchNorm3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 5 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BatchNorm3d: expected 5D input [B, C, D, H, W], got {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let depth = shape[2];
        let height = shape[3];
        let width = shape[4];
        let spatial = depth * height * width;

        if channels != self.num_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BatchNorm3d: expected {} channels, got {}",
                    self.num_features, channels
                ),
            });
        }

        if batch == 0 {
            return Ok(input.clone());
        }

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "BatchNorm3d::forward",
            });
        }
        let input_data = input.data()?;
        let eps_t = T::from(self.eps).unwrap();

        let weight_data = self.weight.as_ref().map(|w| w.tensor().data().unwrap());
        let bias_data = self.bias.as_ref().map(|b| b.tensor().data().unwrap());

        let is_training = *self.training.lock().unwrap();

        let mut chan_mean = vec![zero::<T>(); channels];
        let mut chan_var = vec![zero::<T>(); channels];

        if is_training {
            let count = batch * spatial;
            let count_t = T::from(count).unwrap();

            for c in 0..channels {
                let mut sum = zero::<T>();
                for b in 0..batch {
                    let base = b * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        sum += input_data[base + s];
                    }
                }
                chan_mean[c] = sum / count_t;

                let mut var_sum = zero::<T>();
                for b in 0..batch {
                    let base = b * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        let d = input_data[base + s] - chan_mean[c];
                        var_sum += d * d;
                    }
                }
                chan_var[c] = var_sum / count_t;
            }

            // Update running statistics.
            {
                let mut rm = self.running_mean.lock().unwrap();
                let mut rv = self.running_var.lock().unwrap();
                let mut nbt = self.num_batches_tracked.lock().unwrap();
                *nbt += 1;

                let mom = self.momentum;
                let bessel = if count > 1 {
                    count as f64 / (count as f64 - 1.0)
                } else {
                    1.0
                };

                for c in 0..channels {
                    let batch_mean_f64 = chan_mean[c].to_f64().unwrap();
                    let batch_var_f64 = chan_var[c].to_f64().unwrap();

                    rm[c] = (1.0 - mom) * rm[c] + mom * batch_mean_f64;
                    rv[c] = (1.0 - mom) * rv[c] + mom * batch_var_f64 * bessel;
                }
            }
        } else {
            let rm = self.running_mean.lock().unwrap();
            let rv = self.running_var.lock().unwrap();

            for c in 0..channels {
                chan_mean[c] = T::from(rm[c]).unwrap();
                chan_var[c] = T::from(rv[c]).unwrap();
            }
        }

        let mut output = vec![zero::<T>(); input.numel()];

        let mut inv_std = vec![zero::<T>(); channels];
        let need_x_hat = is_grad_enabled() && input.requires_grad();
        let mut x_hat_data = if need_x_hat {
            Vec::with_capacity(input.numel())
        } else {
            Vec::new()
        };

        for c in 0..channels {
            inv_std[c] = (chan_var[c] + eps_t).sqrt().recip();
        }

        for b in 0..batch {
            for c in 0..channels {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let normed = (input_data[idx] - chan_mean[c]) * inv_std[c];

                    if need_x_hat {
                        x_hat_data.push(normed);
                    }

                    if self.affine {
                        let w = weight_data.as_ref().unwrap();
                        let bi = bias_data.as_ref().unwrap();
                        output[idx] = normed * w[c] + bi[c];
                    } else {
                        output[idx] = normed;
                    }
                }
            }
        }

        let result = Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?;

        if is_grad_enabled() && input.requires_grad() {
            let weight_tensor = self.weight.as_ref().map(|w| w.tensor().clone());
            let bias_tensor = self.bias.as_ref().map(|b| b.tensor().clone());

            let grad_fn = Arc::new(BatchNorm3dBackward {
                input: input.clone(),
                x_hat: Tensor::from_storage(TensorStorage::cpu(x_hat_data), shape.to_vec(), false)?,
                weight: weight_tensor,
                bias: bias_tensor,
                chan_var: chan_var.iter().map(|v| v.to_f64().unwrap()).collect(),
                eps: self.eps,
                affine: self.affine,
            });

            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => vec![w, b],
            _ => vec![],
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        match (&mut self.weight, &mut self.bias) {
            (Some(w), Some(b)) => vec![w, b],
            _ => vec![],
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        match (&self.weight, &self.bias) {
            (Some(w), Some(b)) => vec![("weight".to_string(), w), ("bias".to_string(), b)],
            _ => vec![],
        }
    }

    fn train(&mut self) {
        *self.training.lock().unwrap() = true;
    }

    fn eval(&mut self) {
        *self.training.lock().unwrap() = false;
    }

    fn is_training(&self) -> bool {
        *self.training.lock().unwrap()
    }
}

// ---------------------------------------------------------------------------
// BatchNorm3dBackward
// ---------------------------------------------------------------------------

/// Backward node for `BatchNorm3d`.
///
/// Same math as `BatchNorm2dBackward` but over `(B, D, H, W)` spatial dims.
#[derive(Debug)]
struct BatchNorm3dBackward<T: Float> {
    input: Tensor<T>,
    x_hat: Tensor<T>,
    weight: Option<Tensor<T>>,
    bias: Option<Tensor<T>>,
    chan_var: Vec<f64>,
    eps: f64,
    affine: bool,
}

impl<T: Float> GradFn<T> for BatchNorm3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let spatial: usize = shape[2..].iter().product();
        let count = batch * spatial;
        let count_t = T::from(count).unwrap();

        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "BatchNorm3dBackward",
            });
        }
        let go_data = grad_output.data()?;
        let x_hat_data = self.x_hat.data()?;

        let weight_data = self.weight.as_ref().map(|w| w.data().unwrap().to_vec());

        let mut grad_input = vec![zero::<T>(); self.input.numel()];
        let mut grad_weight = vec![zero::<T>(); channels];
        let mut grad_bias = vec![zero::<T>(); channels];

        for c in 0..channels {
            let var_f64 = self.chan_var[c];
            let inv_std = T::from(1.0 / (var_f64 + self.eps).sqrt()).unwrap();

            let mut dl_dx_hat_sum = zero::<T>();
            let mut dl_dx_hat_x_hat_sum = zero::<T>();

            for b in 0..batch {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let x_h = x_hat_data[idx];
                    let go = go_data[idx];

                    let dl_dx_hat = if self.affine {
                        go * weight_data.as_ref().unwrap()[c]
                    } else {
                        go
                    };

                    dl_dx_hat_sum += dl_dx_hat;
                    dl_dx_hat_x_hat_sum += dl_dx_hat * x_h;

                    if self.affine {
                        grad_weight[c] += go * x_h;
                        grad_bias[c] += go;
                    }
                }
            }

            let dl_dx_hat_mean = dl_dx_hat_sum / count_t;
            let dl_dx_hat_x_hat_mean = dl_dx_hat_x_hat_sum / count_t;

            for b in 0..batch {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let idx = base + s;
                    let x_h = x_hat_data[idx];
                    let go = go_data[idx];

                    let dl_dx_hat = if self.affine {
                        go * weight_data.as_ref().unwrap()[c]
                    } else {
                        go
                    };

                    grad_input[idx] =
                        inv_std * (dl_dx_hat - dl_dx_hat_mean - x_h * dl_dx_hat_x_hat_mean);
                }
            }
        }

        let grad_input_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;

        let grad_weight_out = if self.affine {
            if let Some(ref w) = self.weight {
                if w.requires_grad() {
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(grad_weight),
                        vec![channels],
                        false,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        let grad_bias_out = if self.affine {
            if let Some(ref b) = self.bias {
                if b.requires_grad() {
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(grad_bias),
                        vec![channels],
                        false,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        Ok(vec![
            Some(grad_input_tensor),
            grad_weight_out,
            grad_bias_out,
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v: Vec<&Tensor<T>> = vec![&self.input];
        if let Some(ref w) = self.weight {
            v.push(w);
        }
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "BatchNorm3dBackward"
    }
}

// ===========================================================================
// LocalResponseNorm — CL-435
// ===========================================================================

/// Local Response Normalization (cross-channel normalization).
///
/// Applies the transform:
///
/// ```text
/// output[c] = input[c] / (k + alpha/size * sum(input[j]^2 for j in [c-size/2, c+size/2]))^beta
/// ```
///
/// where the sum is over `size` neighbouring channels (clamped at boundaries).
///
/// Parameters:
/// - `size`: number of neighbouring channels to normalize over.
/// - `alpha`: multiplicative factor (default: `1e-4`).
/// - `beta`: exponent (default: `0.75`).
/// - `k`: additive constant (default: `1.0`).
///
/// This layer has no learnable parameters.
///
/// Matches `torch.nn.LocalResponseNorm`.
#[derive(Debug, Clone)]
pub struct LocalResponseNorm {
    pub size: usize,
    pub alpha: f64,
    pub beta: f64,
    pub k: f64,
    /// Training-mode flag. Carried for Module-trait consistency; the
    /// layer itself is stateless and produces the same output in both
    /// modes (matches PyTorch's `LocalResponseNorm` behaviour).
    training: bool,
}

impl LocalResponseNorm {
    /// Create a new `LocalResponseNorm` layer.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of neighbouring channels used for normalization.
    /// * `alpha` - Multiplicative factor (default: `1e-4`).
    /// * `beta` - Exponent (default: `0.75`).
    /// * `k` - Additive constant (default: `1.0`).
    pub fn new(size: usize, alpha: f64, beta: f64, k: f64) -> FerrotorchResult<Self> {
        if size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LocalResponseNorm: size must be positive".into(),
            });
        }
        Ok(Self {
            size,
            alpha,
            beta,
            k,
            training: true,
        })
    }

    /// Create with default alpha=1e-4, beta=0.75, k=1.0.
    pub fn default_params(size: usize) -> FerrotorchResult<Self> {
        Self::new(size, 1e-4, 0.75, 1.0)
    }
}

impl<T: Float> Module<T> for LocalResponseNorm {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() < 3 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LocalResponseNorm: expected at least 3D input [B, C, ...], got {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let spatial: usize = shape[2..].iter().product();

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "LocalResponseNorm::forward",
            });
        }
        let input_data = input.data()?;
        let alpha_t = T::from(self.alpha).unwrap();
        let beta_t = T::from(self.beta).unwrap();
        let k_t = T::from(self.k).unwrap();
        let size_t = T::from(self.size).unwrap();
        let half = self.size / 2;

        let mut output = vec![zero::<T>(); input.numel()];

        // Pre-compute squared values per channel per spatial position.
        // Also store the denominator for backward.
        let mut denom = vec![zero::<T>(); input.numel()];

        for b in 0..batch {
            for c in 0..channels {
                let c_start = c.saturating_sub(half);
                let c_end = (c + half + 1).min(channels);

                for s in 0..spatial {
                    let mut sq_sum = zero::<T>();
                    for j in c_start..c_end {
                        let jidx = b * channels * spatial + j * spatial + s;
                        sq_sum += input_data[jidx] * input_data[jidx];
                    }

                    let idx = b * channels * spatial + c * spatial + s;
                    let d = k_t + alpha_t / size_t * sq_sum;
                    denom[idx] = d;
                    output[idx] = input_data[idx] * d.powf(-beta_t);
                }
            }
        }

        let storage = TensorStorage::cpu(output);

        if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(LocalResponseNormBackward {
                    input: input.clone(),
                    denom,
                    size: self.size,
                    alpha: self.alpha,
                    beta: self.beta,
                }),
            )
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// LocalResponseNormBackward
// ---------------------------------------------------------------------------

/// Backward node for `LocalResponseNorm`.
///
/// Using:
/// ```text
/// y_c = x_c * D_c^(-beta)
/// ```
/// where `D_c = k + (alpha/size) * sum_{j in window} x_j^2`
///
/// The gradient is:
/// ```text
/// dy/dx_i = D_i^(-beta) - 2*beta*alpha/size * x_i * sum_{c in window_of(i)} (x_c * D_c^(-beta-1))
/// ```
/// combined with the chain rule from upstream.
#[derive(Debug)]
struct LocalResponseNormBackward<T: Float> {
    input: Tensor<T>,
    /// Pre-computed denominator `D_c` per element.
    denom: Vec<T>,
    size: usize,
    alpha: f64,
    beta: f64,
}

impl<T: Float> GradFn<T> for LocalResponseNormBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "LocalResponseNormBackward",
            });
        }

        let shape = self.input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let spatial: usize = shape[2..].iter().product();

        let input_data = self.input.data()?;
        let go_data = grad_output.data()?;
        let alpha_t = T::from(self.alpha).unwrap();
        let beta_t = T::from(self.beta).unwrap();
        let size_t = T::from(self.size).unwrap();
        let two = T::from(2.0).unwrap();
        let half = self.size / 2;

        let mut grad_input = vec![zero::<T>(); self.input.numel()];

        for b in 0..batch {
            for i_c in 0..channels {
                for s in 0..spatial {
                    let i_idx = b * channels * spatial + i_c * spatial + s;

                    // Term 1: D_i^(-beta) * grad_output
                    let term1 = self.denom[i_idx].powf(-beta_t) * go_data[i_idx];

                    // Term 2: cross-channel interaction
                    // For each channel c whose window includes i_c:
                    // contribution = -2*beta*alpha/size * x_i * x_c * D_c^(-beta-1) * go_c
                    let c_start = i_c.saturating_sub(half);
                    let c_end = (i_c + half + 1).min(channels);

                    let mut cross_sum = zero::<T>();
                    for c in c_start..c_end {
                        let c_idx = b * channels * spatial + c * spatial + s;
                        cross_sum += go_data[c_idx]
                            * input_data[c_idx]
                            * self.denom[c_idx].powf(-beta_t - T::from(1.0).unwrap());
                    }

                    grad_input[i_idx] =
                        term1 - two * beta_t * alpha_t / size_t * input_data[i_idx] * cross_sum;
                }
            }
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LocalResponseNormBackward"
    }
}

// ===========================================================================
// InstanceNorm — CL-315
// ===========================================================================

/// Instance normalization: normalizes each **(batch, channel)** slice
/// independently, i.e. statistics are computed over the spatial dimensions
/// only — never across the batch or across channels.
///
/// This is equivalent to `GroupNorm` with `num_groups == num_channels`, but
/// semantically emphasised as a per-instance, per-channel operation.
///
/// Unlike `BatchNorm`, `InstanceNorm` does **not** maintain running
/// statistics, so its behaviour is identical in train and eval modes.
///
/// The generic `InstanceNorm<T>` is the shared engine; the public type
/// aliases `InstanceNorm1d`, `InstanceNorm2d`, `InstanceNorm3d` simply
/// validate that the input tensor has the expected number of dimensions.
/// Internal engine shared by `InstanceNorm1d/2d/3d`.
#[derive(Debug)]
struct InstanceNormInner<T: Float> {
    /// Number of channels (features) `C`.
    num_features: usize,
    /// Small constant for numerical stability.
    eps: f64,
    /// Whether to apply learnable affine parameters.
    affine: bool,
    /// Learnable scale (gamma), shape `[C]`.
    weight: Parameter<T>,
    /// Learnable shift (beta), shape `[C]`.
    bias: Parameter<T>,
    training: bool,
}

impl<T: Float> InstanceNormInner<T> {
    fn new(num_features: usize, eps: f64, affine: bool) -> FerrotorchResult<Self> {
        if num_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "InstanceNorm: num_features must be positive".into(),
            });
        }

        let weight = Parameter::ones(&[num_features])?;
        let bias = Parameter::zeros(&[num_features])?;

        Ok(Self {
            num_features,
            eps,
            affine,
            weight,
            bias,
            training: true,
        })
    }

    /// Forward for input of shape `[B, C, *spatial]`.
    /// `expected_ndim` is used only for error messages (3 = 1d, 4 = 2d, 5 = 3d).
    fn forward_impl(&self, input: &Tensor<T>, expected_ndim: usize) -> FerrotorchResult<Tensor<T>> {
        let label = match expected_ndim {
            3 => "InstanceNorm1d",
            4 => "InstanceNorm2d",
            _ => "InstanceNorm3d",
        };
        let shape = input.shape().to_vec();

        if shape.len() != expected_ndim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("{label}: expected {expected_ndim}D input, got {:?}", shape),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        if channels != self.num_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "{label}: expected {} channels, got {}",
                    self.num_features, channels
                ),
            });
        }

        let spatial: usize = shape[2..].iter().product();
        if spatial == 0 {
            return Ok(input.clone());
        }

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "InstanceNorm::forward",
            });
        }
        let input_data = input.data()?;
        let eps_t = T::from(self.eps).unwrap();
        let n_t = T::from(spatial).unwrap();

        let weight_data = self.weight.tensor().data()?;
        let bias_data = self.bias.tensor().data()?;

        let mut output = vec![zero::<T>(); input.numel()];

        for b in 0..batch {
            for c in 0..channels {
                let base = b * channels * spatial + c * spatial;
                let slice = &input_data[base..base + spatial];

                // Compute mean and variance over spatial dims for this (b, c).
                let mean = slice.iter().copied().fold(zero::<T>(), |a, x| a + x) / n_t;
                let var = slice.iter().copied().fold(zero::<T>(), |a, x| {
                    let d = x - mean;
                    a + d * d
                }) / n_t;
                let inv_std = (var + eps_t).sqrt().recip();

                for s in 0..spatial {
                    let idx = base + s;
                    let normed = (input_data[idx] - mean) * inv_std;
                    if self.affine {
                        output[idx] = normed * weight_data[c] + bias_data[c];
                    } else {
                        output[idx] = normed;
                    }
                }
            }
        }

        let result = Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(InstanceNormBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.tensor().clone(),
                num_features: self.num_features,
                eps: self.eps,
                affine: self.affine,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
// InstanceNormBackward
// ---------------------------------------------------------------------------

/// Backward node for InstanceNorm.
///
/// Same VJP as GroupNorm / LayerNorm, but the normalization group is
/// a single **(batch, channel)** slice over spatial dims.
#[derive(Debug)]
struct InstanceNormBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Tensor<T>,
    num_features: usize,
    eps: f64,
    affine: bool,
}

impl<T: Float> GradFn<T> for InstanceNormBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.input.shape();
        let batch = shape[0];
        let channels = shape[1];
        let spatial: usize = shape[2..].iter().product();
        let n_t = T::from(spatial).unwrap();
        let eps_t = T::from(self.eps).unwrap();

        if self.input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "InstanceNormBackward",
            });
        }
        let input_data = self.input.data()?;
        let go_data = grad_output.data()?;
        let weight_data = self.weight.data()?;

        let mut grad_input = vec![zero::<T>(); self.input.numel()];
        let mut grad_weight = vec![zero::<T>(); self.num_features];
        let mut grad_bias = vec![zero::<T>(); self.num_features];

        for b in 0..batch {
            for c in 0..channels {
                let base = b * channels * spatial + c * spatial;
                let x_slice = &input_data[base..base + spatial];
                let go_slice = &go_data[base..base + spatial];

                // Recompute mean and inv_std for this (b, c).
                let mean = x_slice.iter().copied().fold(zero::<T>(), |a, x| a + x) / n_t;
                let var = x_slice.iter().copied().fold(zero::<T>(), |a, x| {
                    let d = x - mean;
                    a + d * d
                }) / n_t;
                let inv_std = (var + eps_t).sqrt().recip();

                // Accumulate sums for the VJP.
                let mut dl_dx_hat_sum = zero::<T>();
                let mut dl_dx_hat_x_hat_sum = zero::<T>();

                for s in 0..spatial {
                    let x_hat = (x_slice[s] - mean) * inv_std;
                    let dl_dx_hat = if self.affine {
                        go_slice[s] * weight_data[c]
                    } else {
                        go_slice[s]
                    };
                    dl_dx_hat_sum += dl_dx_hat;
                    dl_dx_hat_x_hat_sum += dl_dx_hat * x_hat;

                    if self.affine {
                        grad_weight[c] += go_slice[s] * x_hat;
                        grad_bias[c] += go_slice[s];
                    }
                }

                let dl_dx_hat_mean = dl_dx_hat_sum / n_t;
                let dl_dx_hat_x_hat_mean = dl_dx_hat_x_hat_sum / n_t;

                for s in 0..spatial {
                    let x_hat = (x_slice[s] - mean) * inv_std;
                    let dl_dx_hat = if self.affine {
                        go_slice[s] * weight_data[c]
                    } else {
                        go_slice[s]
                    };
                    grad_input[base + s] =
                        inv_std * (dl_dx_hat - dl_dx_hat_mean - x_hat * dl_dx_hat_x_hat_mean);
                }
            }
        }

        let grad_input_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;

        let grad_weight_out = if self.affine && self.weight.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_weight),
                vec![self.num_features],
                false,
            )?)
        } else {
            None
        };

        let grad_bias_out = if self.affine && self.bias.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_bias),
                vec![self.num_features],
                false,
            )?)
        } else {
            None
        };

        Ok(vec![
            Some(grad_input_tensor),
            grad_weight_out,
            grad_bias_out,
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.weight, &self.bias]
    }

    fn name(&self) -> &'static str {
        "InstanceNormBackward"
    }
}

// ---------------------------------------------------------------------------
// InstanceNorm1d — CL-315
// ---------------------------------------------------------------------------

/// Instance normalization for 3D input `[N, C, L]`.
///
/// Normalizes each `(n, c)` slice independently over the `L` dimension.
/// No running statistics are maintained.
///
/// Matches `torch.nn.InstanceNorm1d`.
#[derive(Debug)]
pub struct InstanceNorm1d<T: Float> {
    inner: InstanceNormInner<T>,
}

impl<T: Float> InstanceNorm1d<T> {
    /// Create a new `InstanceNorm1d` layer.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of channels `C`.
    /// * `eps` - Numerical stability constant (default: `1e-5`).
    /// * `affine` - Whether to include learnable weight and bias.
    pub fn new(num_features: usize, eps: f64, affine: bool) -> FerrotorchResult<Self> {
        Ok(Self {
            inner: InstanceNormInner::new(num_features, eps, affine)?,
        })
    }
}

impl<T: Float> Module<T> for InstanceNorm1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.inner.forward_impl(input, 3)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        if self.inner.affine {
            vec![&self.inner.weight, &self.inner.bias]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        if self.inner.affine {
            vec![&mut self.inner.weight, &mut self.inner.bias]
        } else {
            vec![]
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        if self.inner.affine {
            vec![
                ("weight".to_string(), &self.inner.weight),
                ("bias".to_string(), &self.inner.bias),
            ]
        } else {
            vec![]
        }
    }

    fn train(&mut self) {
        self.inner.training = true;
    }

    fn eval(&mut self) {
        self.inner.training = false;
    }

    fn is_training(&self) -> bool {
        self.inner.training
    }
}

// ---------------------------------------------------------------------------
// InstanceNorm2d — CL-315
// ---------------------------------------------------------------------------

/// Instance normalization for 4D input `[N, C, H, W]`.
///
/// Normalizes each `(n, c)` slice independently over the `(H, W)` dimensions.
/// No running statistics are maintained.
///
/// Matches `torch.nn.InstanceNorm2d`.
#[derive(Debug)]
pub struct InstanceNorm2d<T: Float> {
    inner: InstanceNormInner<T>,
}

impl<T: Float> InstanceNorm2d<T> {
    /// Create a new `InstanceNorm2d` layer.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of channels `C`.
    /// * `eps` - Numerical stability constant (default: `1e-5`).
    /// * `affine` - Whether to include learnable weight and bias.
    pub fn new(num_features: usize, eps: f64, affine: bool) -> FerrotorchResult<Self> {
        Ok(Self {
            inner: InstanceNormInner::new(num_features, eps, affine)?,
        })
    }
}

impl<T: Float> Module<T> for InstanceNorm2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.inner.forward_impl(input, 4)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        if self.inner.affine {
            vec![&self.inner.weight, &self.inner.bias]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        if self.inner.affine {
            vec![&mut self.inner.weight, &mut self.inner.bias]
        } else {
            vec![]
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        if self.inner.affine {
            vec![
                ("weight".to_string(), &self.inner.weight),
                ("bias".to_string(), &self.inner.bias),
            ]
        } else {
            vec![]
        }
    }

    fn train(&mut self) {
        self.inner.training = true;
    }

    fn eval(&mut self) {
        self.inner.training = false;
    }

    fn is_training(&self) -> bool {
        self.inner.training
    }
}

// ---------------------------------------------------------------------------
// InstanceNorm3d — CL-315
// ---------------------------------------------------------------------------

/// Instance normalization for 5D input `[N, C, D, H, W]`.
///
/// Normalizes each `(n, c)` slice independently over the `(D, H, W)` dims.
/// No running statistics are maintained.
///
/// Matches `torch.nn.InstanceNorm3d`.
#[derive(Debug)]
pub struct InstanceNorm3d<T: Float> {
    inner: InstanceNormInner<T>,
}

impl<T: Float> InstanceNorm3d<T> {
    /// Create a new `InstanceNorm3d` layer.
    ///
    /// # Arguments
    ///
    /// * `num_features` - Number of channels `C`.
    /// * `eps` - Numerical stability constant (default: `1e-5`).
    /// * `affine` - Whether to include learnable weight and bias.
    pub fn new(num_features: usize, eps: f64, affine: bool) -> FerrotorchResult<Self> {
        Ok(Self {
            inner: InstanceNormInner::new(num_features, eps, affine)?,
        })
    }
}

impl<T: Float> Module<T> for InstanceNorm3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.inner.forward_impl(input, 5)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        if self.inner.affine {
            vec![&self.inner.weight, &self.inner.bias]
        } else {
            vec![]
        }
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        if self.inner.affine {
            vec![&mut self.inner.weight, &mut self.inner.bias]
        } else {
            vec![]
        }
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        if self.inner.affine {
            vec![
                ("weight".to_string(), &self.inner.weight),
                ("bias".to_string(), &self.inner.bias),
            ]
        } else {
            vec![]
        }
    }

    fn train(&mut self) {
        self.inner.training = true;
    }

    fn eval(&mut self) {
        self.inner.training = false;
    }

    fn is_training(&self) -> bool {
        self.inner.training
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::autograd::no_grad::no_grad;

    /// Helper: create a leaf tensor with given data, shape, and requires_grad.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // LayerNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_layer_norm_parameter_shapes() {
        let ln = LayerNorm::<f32>::new(vec![8], 1e-5, true).unwrap();
        let params = ln.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[8]); // weight
        assert_eq!(params[1].shape(), &[8]); // bias
    }

    #[test]
    fn test_layer_norm_no_affine_no_params() {
        let ln = LayerNorm::<f32>::new(vec![8], 1e-5, false).unwrap();
        assert_eq!(ln.parameters().len(), 0);
    }

    #[test]
    fn test_layer_norm_forward_zero_mean_unit_var() {
        // After LayerNorm (with default weight=1, bias=0), each row should
        // have approximately zero mean and unit variance.
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, // row 0
            -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, // row 1
        ];
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 8], false).unwrap();

        let ln = LayerNorm::<f32>::new(vec![8], 1e-5, true).unwrap();
        let output = ln.forward(&input).unwrap();
        let out_data = output.data().unwrap();

        for row in 0..2 {
            let start = row * 8;
            let end = start + 8;
            let row_data = &out_data[start..end];

            let mean: f32 = row_data.iter().sum::<f32>() / 8.0;
            let var: f32 = row_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 8.0;

            assert!(mean.abs() < 1e-5, "row {row} mean = {mean}, expected ~0");
            assert!(
                (var - 1.0).abs() < 0.05,
                "row {row} var = {var}, expected ~1"
            );
        }
    }

    #[test]
    fn test_layer_norm_forward_shape_preserved() {
        let input =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0; 24]), vec![2, 3, 4], false)
                .unwrap();

        let ln = LayerNorm::<f32>::new(vec![4], 1e-5, true).unwrap();
        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_layer_norm_shape_mismatch() {
        let input =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0; 12]), vec![3, 4], false)
                .unwrap();

        let ln = LayerNorm::<f32>::new(vec![5], 1e-5, true).unwrap();
        assert!(ln.forward(&input).is_err());
    }

    #[test]
    fn test_layer_norm_empty_normalized_shape() {
        assert!(LayerNorm::<f32>::new(vec![], 1e-5, true).is_err());
    }

    #[test]
    fn test_layer_norm_has_grad_fn_when_input_requires_grad() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
            vec![1, 4],
            true,
        )
        .unwrap();

        let ln = LayerNorm::<f32>::new(vec![4], 1e-5, true).unwrap();
        let output = ln.forward(&input).unwrap();
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "LayerNormBackward");
    }

    #[test]
    fn test_layer_norm_no_grad_fn_in_no_grad_context() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
            vec![1, 4],
            true,
        )
        .unwrap();

        let ln = LayerNorm::<f32>::new(vec![4], 1e-5, true).unwrap();
        let output = no_grad(|| ln.forward(&input)).unwrap();
        assert!(output.grad_fn().is_none());
    }

    #[test]
    fn test_layer_norm_backward_gradient_check() -> FerrotorchResult<()> {
        // Numerical gradient check for LayerNorm on a small input.
        // Use f64 for better precision.
        let h = 1e-7;
        let hidden = 4;
        let input_data = vec![1.0f64, -0.5, 2.0, 0.3];

        let ln = LayerNorm::<f64>::new(vec![hidden], 1e-5, true)?;

        // Forward and backward.
        let input = leaf(&input_data, &[1, hidden], true);
        let output = ln.forward(&input)?;
        let out_data = output.data()?.to_vec();
        let total: f64 = out_data.iter().sum();

        // Build sum backward manually.
        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf)?;
        loss.backward()?;

        let analytic_grad = input.grad().unwrap().unwrap();
        let analytic = analytic_grad.data()?.to_vec();

        // Numerical gradient.
        for i in 0..hidden {
            let mut data_plus = input_data.clone();
            data_plus[i] += h;
            let inp_plus = leaf(&data_plus, &[1, hidden], false);
            let out_plus = no_grad(|| ln.forward(&inp_plus)).unwrap();
            let sum_plus: f64 = out_plus.data().unwrap().iter().sum();

            let mut data_minus = input_data.clone();
            data_minus[i] -= h;
            let inp_minus = leaf(&data_minus, &[1, hidden], false);
            let out_minus = no_grad(|| ln.forward(&inp_minus)).unwrap();
            let sum_minus: f64 = out_minus.data().unwrap().iter().sum();

            let numerical = (sum_plus - sum_minus) / (2.0 * h);
            assert!(
                (numerical - analytic[i]).abs() < 1e-4,
                "LayerNorm grad[{i}]: numerical={numerical}, analytic={}",
                analytic[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_layer_norm_named_parameters() {
        let ln = LayerNorm::<f32>::new(vec![16], 1e-5, true).unwrap();
        let named = ln.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_layer_norm_train_eval() {
        let mut ln = LayerNorm::<f32>::new(vec![8], 1e-5, true).unwrap();
        assert!(ln.is_training());
        ln.eval();
        assert!(!ln.is_training());
        ln.train();
        assert!(ln.is_training());
    }

    // -----------------------------------------------------------------------
    // GroupNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_group_norm_parameter_shapes() {
        let gn = GroupNorm::<f32>::new(4, 8, 1e-5, true).unwrap();
        let params = gn.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[8]); // weight
        assert_eq!(params[1].shape(), &[8]); // bias
    }

    #[test]
    fn test_group_norm_no_affine_no_params() {
        let gn = GroupNorm::<f32>::new(2, 4, 1e-5, false).unwrap();
        assert_eq!(gn.parameters().len(), 0);
    }

    #[test]
    fn test_group_norm_invalid_groups() {
        assert!(GroupNorm::<f32>::new(0, 8, 1e-5, true).is_err());
        assert!(GroupNorm::<f32>::new(3, 8, 1e-5, true).is_err()); // 8 not divisible by 3
    }

    #[test]
    fn test_group_norm_forward_zero_mean_unit_var() {
        // With groups=2, channels=4: groups are [0,1] and [2,3].
        // Each group should be normalized to ~zero mean, ~unit var.
        let data: Vec<f32> = vec![
            // batch=0, channel 0: 1, 2; channel 1: 3, 4; channel 2: 5, 6; channel 3: 7, 8
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
        ];
        // Shape: [1, 4, 2] (B=1, C=4, spatial=2)
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 4, 2], false).unwrap();

        let gn = GroupNorm::<f32>::new(2, 4, 1e-5, true).unwrap();
        let output = gn.forward(&input).unwrap();
        let out_data = output.data().unwrap();

        // Group 0: channels 0,1 -> indices [0,1,2,3] -> values were [1,2,3,4]
        let group0: Vec<f32> = out_data[0..4].to_vec();
        let mean0: f32 = group0.iter().sum::<f32>() / 4.0;
        let var0: f32 = group0.iter().map(|&x| (x - mean0).powi(2)).sum::<f32>() / 4.0;
        assert!(mean0.abs() < 1e-5, "group0 mean = {mean0}");
        assert!((var0 - 1.0).abs() < 0.05, "group0 var = {var0}");

        // Group 1: channels 2,3 -> indices [4,5,6,7] -> values were [5,6,7,8]
        let group1: Vec<f32> = out_data[4..8].to_vec();
        let mean1: f32 = group1.iter().sum::<f32>() / 4.0;
        let var1: f32 = group1.iter().map(|&x| (x - mean1).powi(2)).sum::<f32>() / 4.0;
        assert!(mean1.abs() < 1e-5, "group1 mean = {mean1}");
        assert!((var1 - 1.0).abs() < 0.05, "group1 var = {var1}");
    }

    #[test]
    fn test_group_norm_forward_shape_preserved() {
        let input =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0; 48]), vec![2, 4, 6], false)
                .unwrap();

        let gn = GroupNorm::<f32>::new(2, 4, 1e-5, true).unwrap();
        let output = gn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 4, 6]);
    }

    #[test]
    fn test_group_norm_channel_mismatch() {
        let input =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0; 24]), vec![2, 3, 4], false)
                .unwrap();

        let gn = GroupNorm::<f32>::new(2, 4, 1e-5, true).unwrap();
        assert!(gn.forward(&input).is_err());
    }

    #[test]
    fn test_group_norm_has_grad_fn() {
        let input =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0; 8]), vec![1, 4, 2], true)
                .unwrap();

        let gn = GroupNorm::<f32>::new(2, 4, 1e-5, true).unwrap();
        let output = gn.forward(&input).unwrap();
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "GroupNormBackward");
    }

    #[test]
    fn test_group_norm_backward_gradient_check() -> FerrotorchResult<()> {
        let h = 1e-7;
        // Shape: [1, 4, 2], groups=2
        let input_data = vec![1.0f64, -0.5, 2.0, 0.3, -1.0, 0.7, 1.5, -0.2];
        let gn = GroupNorm::<f64>::new(2, 4, 1e-5, true)?;

        let input = leaf(&input_data, &[1, 4, 2], true);
        let output = gn.forward(&input)?;
        let out_data = output.data()?.to_vec();
        let total: f64 = out_data.iter().sum();

        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf)?;
        loss.backward()?;

        let analytic_grad = input.grad().unwrap().unwrap();
        let analytic = analytic_grad.data()?.to_vec();

        for i in 0..8 {
            let mut data_plus = input_data.clone();
            data_plus[i] += h;
            let inp_plus = leaf(&data_plus, &[1, 4, 2], false);
            let out_plus = no_grad(|| gn.forward(&inp_plus)).unwrap();
            let sum_plus: f64 = out_plus.data().unwrap().iter().sum();

            let mut data_minus = input_data.clone();
            data_minus[i] -= h;
            let inp_minus = leaf(&data_minus, &[1, 4, 2], false);
            let out_minus = no_grad(|| gn.forward(&inp_minus)).unwrap();
            let sum_minus: f64 = out_minus.data().unwrap().iter().sum();

            let numerical = (sum_plus - sum_minus) / (2.0 * h);
            assert!(
                (numerical - analytic[i]).abs() < 1e-4,
                "GroupNorm grad[{i}]: numerical={numerical}, analytic={}",
                analytic[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_group_norm_named_parameters() {
        let gn = GroupNorm::<f32>::new(2, 8, 1e-5, true).unwrap();
        let named = gn.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    // -----------------------------------------------------------------------
    // RMSNorm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rms_norm_parameter_shapes() {
        let rn = RMSNorm::<f32>::new(vec![8], 1e-5).unwrap();
        let params = rn.parameters();
        assert_eq!(params.len(), 1);
        assert_eq!(params[0].shape(), &[8]); // weight only
    }

    #[test]
    fn test_rms_norm_forward_scale() {
        // After RMSNorm (with weight=1), the RMS of each row should be ~1.
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // row 0
            -1.0, 0.5, 2.0, -3.0, // row 1
        ];
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 4], false).unwrap();

        let rn = RMSNorm::<f32>::new(vec![4], 1e-5).unwrap();
        let output = rn.forward(&input).unwrap();
        let out_data = output.data().unwrap();

        for row in 0..2 {
            let start = row * 4;
            let end = start + 4;
            let row_data = &out_data[start..end];

            let mean_sq: f32 = row_data.iter().map(|x| x * x).sum::<f32>() / 4.0;
            let rms = mean_sq.sqrt();

            assert!(
                (rms - 1.0).abs() < 0.05,
                "row {row} RMS = {rms}, expected ~1"
            );
        }
    }

    #[test]
    fn test_rms_norm_forward_shape_preserved() {
        let input =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0; 24]), vec![2, 3, 4], false)
                .unwrap();

        let rn = RMSNorm::<f32>::new(vec![4], 1e-5).unwrap();
        let output = rn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_rms_norm_empty_normalized_shape() {
        assert!(RMSNorm::<f32>::new(vec![], 1e-5).is_err());
    }

    #[test]
    fn test_rms_norm_has_grad_fn() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
            vec![1, 4],
            true,
        )
        .unwrap();

        let rn = RMSNorm::<f32>::new(vec![4], 1e-5).unwrap();
        let output = rn.forward(&input).unwrap();
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "RMSNormBackward");
    }

    #[test]
    fn test_rms_norm_no_grad_fn_in_no_grad_context() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
            vec![1, 4],
            true,
        )
        .unwrap();

        let rn = RMSNorm::<f32>::new(vec![4], 1e-5).unwrap();
        let output = no_grad(|| rn.forward(&input)).unwrap();
        assert!(output.grad_fn().is_none());
    }

    #[test]
    fn test_rms_norm_backward_gradient_check() -> FerrotorchResult<()> {
        let h = 1e-7;
        let hidden = 4;
        let input_data = vec![1.0f64, -0.5, 2.0, 0.3];

        let rn = RMSNorm::<f64>::new(vec![hidden], 1e-5)?;

        let input = leaf(&input_data, &[1, hidden], true);
        let output = rn.forward(&input)?;
        let out_data = output.data()?.to_vec();
        let total: f64 = out_data.iter().sum();

        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf)?;
        loss.backward()?;

        let analytic_grad = input.grad().unwrap().unwrap();
        let analytic = analytic_grad.data()?.to_vec();

        for i in 0..hidden {
            let mut data_plus = input_data.clone();
            data_plus[i] += h;
            let inp_plus = leaf(&data_plus, &[1, hidden], false);
            let out_plus = no_grad(|| rn.forward(&inp_plus)).unwrap();
            let sum_plus: f64 = out_plus.data().unwrap().iter().sum();

            let mut data_minus = input_data.clone();
            data_minus[i] -= h;
            let inp_minus = leaf(&data_minus, &[1, hidden], false);
            let out_minus = no_grad(|| rn.forward(&inp_minus)).unwrap();
            let sum_minus: f64 = out_minus.data().unwrap().iter().sum();

            let numerical = (sum_plus - sum_minus) / (2.0 * h);
            assert!(
                (numerical - analytic[i]).abs() < 1e-4,
                "RMSNorm grad[{i}]: numerical={numerical}, analytic={}",
                analytic[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_rms_norm_named_parameters() {
        let rn = RMSNorm::<f32>::new(vec![16], 1e-5).unwrap();
        let named = rn.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    #[test]
    fn test_rms_norm_train_eval() {
        let mut rn = RMSNorm::<f32>::new(vec![8], 1e-5).unwrap();
        assert!(rn.is_training());
        rn.eval();
        assert!(!rn.is_training());
        rn.train();
        assert!(rn.is_training());
    }

    // -----------------------------------------------------------------------
    // BatchNorm2d tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batch_norm_2d_output_shape() {
        let bn = BatchNorm2d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        // [B=2, C=3, H=4, W=4]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 2 * 3 * 4 * 4]),
            vec![2, 3, 4, 4],
            false,
        )
        .unwrap();

        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4, 4]);
    }

    #[test]
    fn test_batch_norm_2d_rejects_non_4d() {
        let bn = BatchNorm2d::<f32>::new(4, 1e-5, 0.1, true).unwrap();
        // 3D input should fail.
        let input =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 24]), vec![2, 4, 3], false)
                .unwrap();
        assert!(bn.forward(&input).is_err());
    }

    #[test]
    fn test_batch_norm_2d_channel_mismatch() {
        let bn = BatchNorm2d::<f32>::new(4, 1e-5, 0.1, true).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 2 * 3 * 2 * 2]),
            vec![2, 3, 2, 2],
            false,
        )
        .unwrap();
        assert!(bn.forward(&input).is_err());
    }

    #[test]
    fn test_batch_norm_2d_zero_features() {
        assert!(BatchNorm2d::<f32>::new(0, 1e-5, 0.1, true).is_err());
    }

    #[test]
    fn test_batch_norm_2d_training_normalizes() {
        // After training-mode BatchNorm2d (weight=1, bias=0), each channel
        // should have approximately zero mean and unit variance over (B, H, W).
        let channels = 2;
        let b = 2;
        let h = 3;
        let w = 3;
        let spatial = h * w;
        // Build data: channel 0 has values 1..18, channel 1 has 101..118
        let mut data = Vec::new();
        for bi in 0..b {
            for c in 0..channels {
                let offset = c as f32 * 100.0;
                for s in 0..spatial {
                    data.push(offset + (bi * spatial + s) as f32 + 1.0);
                }
            }
        }
        let input =
            Tensor::from_storage(TensorStorage::cpu(data), vec![b, channels, h, w], false).unwrap();

        let bn = BatchNorm2d::<f32>::new(channels, 1e-5, 0.1, true).unwrap();
        let output = bn.forward(&input).unwrap();
        let out_data = output.data().unwrap();

        for c in 0..channels {
            // Gather all values for this channel across (B, H, W).
            let mut vals = Vec::new();
            for bi in 0..b {
                let base = bi * channels * spatial + c * spatial;
                for s in 0..spatial {
                    vals.push(out_data[base + s]);
                }
            }
            let n = vals.len() as f32;
            let mean: f32 = vals.iter().sum::<f32>() / n;
            let var: f32 = vals.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

            assert!(mean.abs() < 1e-4, "channel {c}: mean = {mean}, expected ~0");
            assert!(
                (var - 1.0).abs() < 0.1,
                "channel {c}: var = {var}, expected ~1"
            );
        }
    }

    #[test]
    fn test_batch_norm_2d_eval_uses_running_stats() {
        let channels = 2;
        let b = 4;
        let h = 2;
        let w = 2;
        let spatial = h * w;

        // Create layer and run a few training batches to build up running stats.
        let bn = BatchNorm2d::<f64>::new(channels, 1e-5, 0.1, true).unwrap();

        // Training batch with known data.
        let mut data = vec![0.0f64; b * channels * spatial];
        for bi in 0..b {
            for c in 0..channels {
                let base = bi * channels * spatial + c * spatial;
                for s in 0..spatial {
                    data[base + s] = (c as f64) * 10.0 + (bi * spatial + s) as f64;
                }
            }
        }
        let input = Tensor::from_storage(
            TensorStorage::cpu(data.clone()),
            vec![b, channels, h, w],
            false,
        )
        .unwrap();

        // Training forward to update running stats.
        let _ = bn.forward(&input).unwrap();
        let rm_after_train = bn.running_mean();
        let rv_after_train = bn.running_var();

        // Running stats should no longer be the initial [0,0] and [1,1].
        assert!(
            rm_after_train[0].abs() > 1e-6 || rm_after_train[1].abs() > 1e-6,
            "running_mean should have been updated"
        );

        // Switch to eval mode and forward again.
        // Use a sneaky mut reference via the training mutex.
        *bn.training.lock().unwrap() = false;

        let output_eval = bn.forward(&input).unwrap();
        let eval_data = output_eval.data().unwrap();

        // In eval mode, the output should use running_mean/running_var,
        // so per-channel values won't necessarily have zero mean.
        // Verify that the output is deterministic and matches
        // manual computation using running stats.
        for c in 0..channels {
            let expected_mean = rm_after_train[c];
            let expected_var = rv_after_train[c];
            let inv_std = 1.0 / (expected_var + 1e-5).sqrt();

            for bi in 0..b {
                let base = bi * channels * spatial + c * spatial;
                for s in 0..spatial {
                    let x = (c as f64) * 10.0 + (bi * spatial + s) as f64;
                    let expected = (x - expected_mean) * inv_std;
                    // weight=1, bias=0 by default.
                    let actual = eval_data[base + s];
                    assert!(
                        (actual - expected).abs() < 1e-6,
                        "eval output mismatch at b={bi}, c={c}, s={s}: actual={actual}, expected={expected}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_batch_norm_2d_running_stats_update() {
        let channels = 2;
        let bn = BatchNorm2d::<f32>::new(channels, 1e-5, 0.1, true).unwrap();

        // Initial state.
        assert_eq!(bn.running_mean(), vec![0.0, 0.0]);
        assert_eq!(bn.running_var(), vec![1.0, 1.0]);
        assert_eq!(bn.num_batches_tracked(), 0);

        // Forward pass 1.
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 2 * 2 * 2 * 2]),
            vec![2, 2, 2, 2],
            false,
        )
        .unwrap();
        let _ = bn.forward(&input).unwrap();
        assert_eq!(bn.num_batches_tracked(), 1);

        let rm = bn.running_mean();
        let rv = bn.running_var();
        // running_mean = (1-0.1)*0 + 0.1*batch_mean = 0.1*1.0 = 0.1
        assert!(
            (rm[0] - 0.1).abs() < 1e-5,
            "running_mean[0] = {}, expected 0.1",
            rm[0]
        );
        // batch_var = 0 (all values are 1.0), bessel-corrected var = 0
        // running_var = (1-0.1)*1.0 + 0.1*0.0 = 0.9
        assert!(
            (rv[0] - 0.9).abs() < 1e-5,
            "running_var[0] = {}, expected 0.9",
            rv[0]
        );

        // Forward pass 2.
        let _ = bn.forward(&input).unwrap();
        assert_eq!(bn.num_batches_tracked(), 2);
    }

    #[test]
    fn test_batch_norm_2d_affine_parameters() {
        let bn = BatchNorm2d::<f32>::new(8, 1e-5, 0.1, true).unwrap();
        let params = bn.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[8]); // weight
        assert_eq!(params[1].shape(), &[8]); // bias

        let named = bn.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");

        // Weight should be ones, bias should be zeros.
        let weight_data = params[0].data().unwrap();
        let bias_data = params[1].data().unwrap();
        assert!(weight_data.iter().all(|&x| (x - 1.0).abs() < 1e-7));
        assert!(bias_data.iter().all(|&x| x.abs() < 1e-7));
    }

    #[test]
    fn test_batch_norm_2d_no_affine_no_params() {
        let bn = BatchNorm2d::<f32>::new(4, 1e-5, 0.1, false).unwrap();
        assert_eq!(bn.parameters().len(), 0);
        assert_eq!(bn.named_parameters().len(), 0);
        assert!(bn.weight.is_none());
        assert!(bn.bias.is_none());
    }

    #[test]
    fn test_batch_norm_2d_train_eval_toggle() {
        let mut bn = BatchNorm2d::<f32>::new(4, 1e-5, 0.1, true).unwrap();
        assert!(bn.is_training());
        bn.eval();
        assert!(!bn.is_training());
        bn.train();
        assert!(bn.is_training());
    }

    #[test]
    fn test_batch_norm_2d_has_grad_fn() {
        let bn = BatchNorm2d::<f32>::new(2, 1e-5, 0.1, true).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 2 * 2 * 3 * 3]),
            vec![2, 2, 3, 3],
            true,
        )
        .unwrap();

        let output = bn.forward(&input).unwrap();
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "BatchNorm2dBackward");
    }

    #[test]
    fn test_batch_norm_2d_no_grad_fn_in_no_grad_context() {
        let bn = BatchNorm2d::<f32>::new(2, 1e-5, 0.1, true).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 2 * 2 * 3 * 3]),
            vec![2, 2, 3, 3],
            true,
        )
        .unwrap();

        let output = no_grad(|| bn.forward(&input)).unwrap();
        assert!(output.grad_fn().is_none());
    }

    #[test]
    fn test_batch_norm_2d_backward_gradient_check() -> FerrotorchResult<()> {
        let h_eps = 1e-7;
        let channels = 2;
        let b = 2;
        let height = 2;
        let width = 2;
        let spatial = height * width;
        let numel = b * channels * spatial;

        // Build non-trivial input data.
        let input_data: Vec<f64> = (0..numel).map(|i| (i as f64) * 0.3 - 1.0).collect();

        let bn = BatchNorm2d::<f64>::new(channels, 1e-5, 0.1, true)?;

        let input = leaf(&input_data, &[b, channels, height, width], true);
        let output = bn.forward(&input)?;
        let out_data = output.data()?.to_vec();
        let total: f64 = out_data.iter().sum();

        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf)?;
        loss.backward()?;

        let analytic_grad = input.grad().unwrap().unwrap();
        let analytic = analytic_grad.data()?.to_vec();

        // Numerical gradient with fresh BatchNorm2d instances to avoid
        // running-stats side effects. We use eval mode with the same
        // batch statistics to keep the function pure.
        for i in 0..numel {
            // f(x + h)
            let mut data_plus = input_data.clone();
            data_plus[i] += h_eps;
            let inp_plus = leaf(&data_plus, &[b, channels, height, width], false);
            let bn_plus = BatchNorm2d::<f64>::new(channels, 1e-5, 0.1, true)?;
            let out_plus = no_grad(|| bn_plus.forward(&inp_plus)).unwrap();
            let sum_plus: f64 = out_plus.data().unwrap().iter().sum();

            // f(x - h)
            let mut data_minus = input_data.clone();
            data_minus[i] -= h_eps;
            let inp_minus = leaf(&data_minus, &[b, channels, height, width], false);
            let bn_minus = BatchNorm2d::<f64>::new(channels, 1e-5, 0.1, true)?;
            let out_minus = no_grad(|| bn_minus.forward(&inp_minus)).unwrap();
            let sum_minus: f64 = out_minus.data().unwrap().iter().sum();

            let numerical = (sum_plus - sum_minus) / (2.0 * h_eps);
            assert!(
                (numerical - analytic[i]).abs() < 1e-4,
                "BatchNorm2d grad[{i}]: numerical={numerical}, analytic={}",
                analytic[i]
            );
        }

        Ok(())
    }

    #[test]
    fn test_batch_norm_2d_no_affine_forward() {
        // Verify that non-affine mode still normalizes correctly.
        let channels = 2;
        let b = 2;
        let h = 2;
        let w = 2;
        let spatial = h * w;

        let mut data = Vec::new();
        for bi in 0..b {
            for c in 0..channels {
                for s in 0..spatial {
                    data.push((c as f32) * 5.0 + (bi * spatial + s) as f32);
                }
            }
        }

        let input =
            Tensor::from_storage(TensorStorage::cpu(data), vec![b, channels, h, w], false).unwrap();

        let bn = BatchNorm2d::<f32>::new(channels, 1e-5, 0.1, false).unwrap();
        let output = bn.forward(&input).unwrap();
        let out_data = output.data().unwrap();

        for c in 0..channels {
            let mut vals = Vec::new();
            for bi in 0..b {
                let base = bi * channels * spatial + c * spatial;
                for s in 0..spatial {
                    vals.push(out_data[base + s]);
                }
            }
            let n = vals.len() as f32;
            let mean: f32 = vals.iter().sum::<f32>() / n;
            let var: f32 = vals.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / n;

            assert!(mean.abs() < 1e-4, "no-affine channel {c}: mean = {mean}");
            assert!(
                (var - 1.0).abs() < 0.1,
                "no-affine channel {c}: var = {var}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Send + Sync tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_layer_norm_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LayerNorm<f32>>();
    }

    #[test]
    fn test_group_norm_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<GroupNorm<f32>>();
    }

    #[test]
    fn test_rms_norm_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<RMSNorm<f32>>();
    }

    #[test]
    fn test_batch_norm_2d_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BatchNorm2d<f32>>();
    }

    // -----------------------------------------------------------------------
    // Helper backward node for tests
    // -----------------------------------------------------------------------

    /// Shorthand for the unambiguous one (test-only).
    fn one<T: Float>() -> T {
        <T as num_traits::One>::one()
    }

    /// Sum reduction backward for test use: loss = sum(input).
    #[derive(Debug)]
    struct SumBackwardHelper<T: Float> {
        input: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for SumBackwardHelper<T> {
        fn backward(&self, _grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let ones_data = vec![one::<T>(); self.input.numel()];
            let ones = Tensor::from_storage(
                TensorStorage::cpu(ones_data),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(ones)])
        }

        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.input]
        }

        fn name(&self) -> &'static str {
            "SumBackwardHelper"
        }
    }

    // -----------------------------------------------------------------------
    // BatchNorm1d tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_batchnorm1d_parameter_shapes() {
        let bn = BatchNorm1d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let params = bn.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[3]); // weight
        assert_eq!(params[1].shape(), &[3]); // bias
    }

    #[test]
    fn test_batchnorm1d_no_affine() {
        let bn = BatchNorm1d::<f32>::new(4, 1e-5, 0.1, false).unwrap();
        assert!(bn.parameters().is_empty());
    }

    #[test]
    fn test_batchnorm1d_2d_input() {
        // Input: [N=4, C=2]
        let bn = BatchNorm1d::<f32>::new(2, 1e-5, 0.1, true).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]),
            vec![4, 2],
            false,
        )
        .unwrap();
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 2]);
    }

    #[test]
    fn test_batchnorm1d_3d_input() {
        // Input: [N=2, C=3, L=4]
        let bn = BatchNorm1d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 4], false).unwrap();
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_batchnorm1d_wrong_dims() {
        // 1D, 4D, 5D should fail.
        let bn = BatchNorm1d::<f32>::new(4, 1e-5, 0.1, true).unwrap();

        let input_1d = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
            vec![4],
            false,
        )
        .unwrap();
        assert!(bn.forward(&input_1d).is_err());

        let input_4d = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32; 32]),
            vec![2, 4, 2, 2],
            false,
        )
        .unwrap();
        assert!(bn.forward(&input_4d).is_err());
    }

    #[test]
    fn test_batchnorm1d_zero_features() {
        assert!(BatchNorm1d::<f32>::new(0, 1e-5, 0.1, true).is_err());
    }

    #[test]
    fn test_batchnorm1d_channel_mismatch() {
        let bn = BatchNorm1d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let input =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 8]), vec![2, 4], false).unwrap();
        assert!(bn.forward(&input).is_err());
    }

    #[test]
    fn test_batchnorm1d_training_normalizes() {
        // After training-mode BatchNorm1d (weight=1, bias=0), each channel
        // should have approximately zero mean and unit variance.
        let channels = 2;
        let bn = BatchNorm1d::<f64>::new(channels, 1e-5, 0.1, true).unwrap();

        // Input [4, 2]: channel 0 = [1,3,5,7], channel 1 = [2,4,6,8]
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2], false);
        let output = bn.forward(&input).unwrap();
        let data = output.data().unwrap();

        // Check channel 0 mean ~ 0
        let ch0: Vec<f64> = (0..4).map(|b| data[b * 2]).collect();
        let ch0_mean: f64 = ch0.iter().sum::<f64>() / 4.0;
        assert!(
            ch0_mean.abs() < 1e-5,
            "BatchNorm1d channel 0 mean should be ~0, got {}",
            ch0_mean
        );

        // Check channel 0 variance ~ 1
        let ch0_var: f64 = ch0.iter().map(|&x| (x - ch0_mean).powi(2)).sum::<f64>() / 4.0;
        assert!(
            (ch0_var - 1.0).abs() < 0.1,
            "BatchNorm1d channel 0 var should be ~1, got {}",
            ch0_var
        );
    }

    #[test]
    fn test_batchnorm1d_running_stats_update() {
        let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1, true).unwrap();
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2], false);
        let _ = bn.forward(&input).unwrap();

        assert_eq!(bn.num_batches_tracked(), 1);
        let rm = bn.running_mean();
        let rv = bn.running_var();
        // Channel 0 mean = (1+3+5+7)/4 = 4.0
        // Channel 1 mean = (2+4+6+8)/4 = 5.0
        // running_mean = 0.9 * 0 + 0.1 * batch_mean
        assert!(
            (rm[0] - 0.1 * 4.0).abs() < 1e-7,
            "running_mean[0]: expected {}, got {}",
            0.1 * 4.0,
            rm[0]
        );
        assert!(
            (rm[1] - 0.1 * 5.0).abs() < 1e-7,
            "running_mean[1]: expected {}, got {}",
            0.1 * 5.0,
            rm[1]
        );

        // running_var uses Bessel-corrected variance
        assert!(rv[0] > 0.0);
        assert!(rv[1] > 0.0);
    }

    #[test]
    fn test_batchnorm1d_eval_mode() {
        let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1, true).unwrap();

        // Run training forward to populate running stats.
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2], false);
        let _ = bn.forward(&input).unwrap();

        // Switch to eval mode.
        // We need a mutable reference for eval, so use a workaround.
        *bn.training.lock().unwrap() = false;

        let eval_out = bn.forward(&input).unwrap();
        assert_eq!(eval_out.shape(), &[4, 2]);
    }

    #[test]
    fn test_batchnorm1d_no_affine_normalizes() {
        let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1, false).unwrap();
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2], false);
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 2]);
    }

    #[test]
    fn test_batchnorm1d_3d_normalizes() {
        // [N=2, C=2, L=3]
        let channels = 2;
        let bn = BatchNorm1d::<f64>::new(channels, 1e-5, 0.1, true).unwrap();
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let input = leaf(&data, &[2, 2, 3], false);
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 2, 3]);

        // Each channel normalized over (N, L) = 6 elements.
        let out_data = output.data().unwrap();

        // Channel 0 indices in [N, C, L] layout:
        // [0, 0, :] = indices 0,1,2; [1, 0, :] = indices 6,7,8
        let ch0: Vec<f64> = vec![
            out_data[0],
            out_data[1],
            out_data[2],
            out_data[6],
            out_data[7],
            out_data[8],
        ];
        let ch0_mean: f64 = ch0.iter().sum::<f64>() / 6.0;
        assert!(
            ch0_mean.abs() < 1e-5,
            "BatchNorm1d 3D channel 0 mean should be ~0, got {}",
            ch0_mean
        );
    }

    #[test]
    fn test_batchnorm1d_train_eval_toggle() {
        let bn = BatchNorm1d::<f32>::new(4, 1e-5, 0.1, true).unwrap();
        assert!(bn.is_training());
        *bn.training.lock().unwrap() = false;
        assert!(!bn.is_training());
        *bn.training.lock().unwrap() = true;
        assert!(bn.is_training());
    }

    #[test]
    fn test_batchnorm1d_grad_fn_name() {
        let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1, true).unwrap();
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2], true);
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.grad_fn().unwrap().name(), "BatchNorm1dBackward");
    }

    #[test]
    fn test_batchnorm1d_backward_grad_shapes() {
        use ferrotorch_core::autograd::graph::backward;

        let bn = BatchNorm1d::<f64>::new(2, 1e-5, 0.1, true).unwrap();
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[4, 2], true);
        let output = bn.forward(&input).unwrap();

        // Create a differentiable sum via SumBackwardHelper.
        let out_data = output.data().unwrap().to_vec();
        let total: f64 = out_data.iter().sum();
        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf).unwrap();
        backward(&loss).unwrap();

        let grad = input.grad().unwrap().unwrap();
        assert_eq!(grad.shape(), &[4, 2]);
    }

    #[test]
    fn test_batchnorm1d_backward_numerical() {
        use ferrotorch_core::autograd::graph::backward;

        // Numerical gradient check for BatchNorm1d.
        let channels = 2;
        let eps_val = 1e-5;
        let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let shape = [4usize, 2];

        // Analytic gradient.
        let bn = BatchNorm1d::<f64>::new(channels, eps_val, 0.1, true).unwrap();
        let input = leaf(&input_data, &shape, true);
        let output = bn.forward(&input).unwrap();
        let out_data = output.data().unwrap().to_vec();
        let total: f64 = out_data.iter().sum();
        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf).unwrap();
        backward(&loss).unwrap();
        let analytic_grad = input.grad().unwrap().unwrap().data_vec().unwrap();

        // Numerical gradient.
        let h = 1e-5;
        let mut numerical_grad = vec![0.0f64; input_data.len()];
        for i in 0..input_data.len() {
            let mut data_plus = input_data.clone();
            data_plus[i] += h;
            let bn_plus = BatchNorm1d::<f64>::new(channels, eps_val, 0.1, true).unwrap();
            let input_plus = leaf(&data_plus, &shape, false);
            let out_plus = no_grad(|| bn_plus.forward(&input_plus)).unwrap();
            let sum_plus: f64 = out_plus.data().unwrap().iter().sum();

            let mut data_minus = input_data.clone();
            data_minus[i] -= h;
            let bn_minus = BatchNorm1d::<f64>::new(channels, eps_val, 0.1, true).unwrap();
            let input_minus = leaf(&data_minus, &shape, false);
            let out_minus = no_grad(|| bn_minus.forward(&input_minus)).unwrap();
            let sum_minus: f64 = out_minus.data().unwrap().iter().sum();

            numerical_grad[i] = (sum_plus - sum_minus) / (2.0 * h);
        }

        for i in 0..input_data.len() {
            assert!(
                (analytic_grad[i] - numerical_grad[i]).abs() < 1e-3,
                "BatchNorm1d grad[{}]: numerical={}, analytic={}",
                i,
                numerical_grad[i],
                analytic_grad[i]
            );
        }
    }

    #[test]
    fn test_batchnorm1d_empty_batch() {
        let bn = BatchNorm1d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let input = Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0, 3], false).unwrap();
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[0, 3]);
        assert_eq!(output.numel(), 0);
    }

    #[test]
    fn test_batchnorm1d_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BatchNorm1d<f32>>();
    }

    // -----------------------------------------------------------------------
    // BatchNorm3d tests — CL-434
    // -----------------------------------------------------------------------

    #[test]
    fn test_batchnorm3d_output_shape() {
        let bn = BatchNorm3d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 2 * 3 * 2 * 2 * 2]),
            vec![2, 3, 2, 2, 2],
            false,
        )
        .unwrap();
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 2, 2, 2]);
    }

    #[test]
    fn test_batchnorm3d_rejects_non_5d() {
        let bn = BatchNorm3d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let input =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 24]), vec![2, 3, 4], false)
                .unwrap();
        assert!(bn.forward(&input).is_err());
    }

    #[test]
    fn test_batchnorm3d_channel_mismatch() {
        let bn = BatchNorm3d::<f32>::new(3, 1e-5, 0.1, true).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 2 * 4 * 2 * 2 * 2]),
            vec![2, 4, 2, 2, 2],
            false,
        )
        .unwrap();
        assert!(bn.forward(&input).is_err());
    }

    #[test]
    fn test_batchnorm3d_zero_features_rejected() {
        assert!(BatchNorm3d::<f32>::new(0, 1e-5, 0.1, true).is_err());
    }

    #[test]
    fn test_batchnorm3d_training_normalizes() {
        // After BatchNorm3d in training mode (weight=1, bias=0),
        // each channel should have approximately zero mean.
        let channels = 2;
        let bn = BatchNorm3d::<f64>::new(channels, 1e-5, 0.1, true).unwrap();
        let mut data = Vec::with_capacity(2 * 2 * 2 * 2 * 2);
        for i in 0..(2 * 2 * 2 * 2 * 2) {
            data.push(i as f64);
        }
        let input = leaf(&data, &[2, 2, 2, 2, 2], false);
        let output = bn.forward(&input).unwrap();
        let out_data = output.data().unwrap();

        let spatial = 2 * 2 * 2;
        let batch = 2;
        for c in 0..channels {
            let mut sum = 0.0;
            for b in 0..batch {
                let base = b * channels * spatial + c * spatial;
                for s in 0..spatial {
                    sum += out_data[base + s];
                }
            }
            let mean = sum / (batch * spatial) as f64;
            assert!(mean.abs() < 1e-5, "channel {c} mean = {mean}, expected ~0");
        }
    }

    #[test]
    fn test_batchnorm3d_running_stats_updated() {
        let bn = BatchNorm3d::<f64>::new(2, 1e-5, 0.1, true).unwrap();
        let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let input = leaf(&data, &[2, 2, 2, 2, 2], false);
        let _ = bn.forward(&input).unwrap();

        assert_eq!(bn.num_batches_tracked(), 1);
        let rm = bn.running_mean();
        assert!(
            rm[0] != 0.0 || rm[1] != 0.0,
            "running mean should be updated"
        );
    }

    #[test]
    fn test_batchnorm3d_eval_uses_running_stats() {
        let mut bn = BatchNorm3d::<f64>::new(2, 1e-5, 0.1, true).unwrap();
        // Train on one batch to populate running stats.
        let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let input = leaf(&data, &[2, 2, 2, 2, 2], false);
        let _ = bn.forward(&input).unwrap();

        bn.eval();
        // Eval forward should use running stats, not batch stats.
        let output = bn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_batchnorm3d_parameters() {
        let bn = BatchNorm3d::<f32>::new(4, 1e-5, 0.1, true).unwrap();
        let params = bn.parameters();
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[4]); // weight
        assert_eq!(params[1].shape(), &[4]); // bias
    }

    #[test]
    fn test_batchnorm3d_no_affine_no_params() {
        let bn = BatchNorm3d::<f32>::new(4, 1e-5, 0.1, false).unwrap();
        assert!(bn.parameters().is_empty());
    }

    #[test]
    fn test_batchnorm3d_backward_grad_shapes() {
        use ferrotorch_core::autograd::graph::backward;

        let bn = BatchNorm3d::<f64>::new(2, 1e-5, 0.1, true).unwrap();
        let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let input = leaf(&data, &[2, 2, 2, 2, 2], true);
        let output = bn.forward(&input).unwrap();

        let out_data = output.data().unwrap().to_vec();
        let total: f64 = out_data.iter().sum();
        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf).unwrap();
        backward(&loss).unwrap();

        let grad = input.grad().unwrap().unwrap();
        assert_eq!(grad.shape(), &[2, 2, 2, 2, 2]);
    }

    #[test]
    fn test_batchnorm3d_backward_numerical() {
        use ferrotorch_core::autograd::graph::backward;

        let channels = 2;
        let eps_val = 1e-5;
        let data: Vec<f64> = (0..32).map(|i| i as f64 * 0.1).collect();
        let shape = [2usize, 2, 2, 2, 2];

        let bn = BatchNorm3d::<f64>::new(channels, eps_val, 0.1, true).unwrap();
        let input = leaf(&data, &shape, true);
        let output = bn.forward(&input).unwrap();
        let out_data = output.data().unwrap().to_vec();
        let total: f64 = out_data.iter().sum();
        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf).unwrap();
        backward(&loss).unwrap();
        let analytic_grad = input.grad().unwrap().unwrap().data_vec().unwrap();

        let h = 1e-5;
        for i in 0..data.len() {
            let mut data_plus = data.clone();
            data_plus[i] += h;
            let bn_plus = BatchNorm3d::<f64>::new(channels, eps_val, 0.1, true).unwrap();
            let input_plus = leaf(&data_plus, &shape, false);
            let out_plus = no_grad(|| bn_plus.forward(&input_plus)).unwrap();
            let sum_plus: f64 = out_plus.data().unwrap().iter().sum();

            let mut data_minus = data.clone();
            data_minus[i] -= h;
            let bn_minus = BatchNorm3d::<f64>::new(channels, eps_val, 0.1, true).unwrap();
            let input_minus = leaf(&data_minus, &shape, false);
            let out_minus = no_grad(|| bn_minus.forward(&input_minus)).unwrap();
            let sum_minus: f64 = out_minus.data().unwrap().iter().sum();

            let numerical = (sum_plus - sum_minus) / (2.0 * h);
            assert!(
                (analytic_grad[i] - numerical).abs() < 1e-3,
                "BatchNorm3d grad[{}]: numerical={}, analytic={}",
                i,
                numerical,
                analytic_grad[i]
            );
        }
    }

    #[test]
    fn test_batchnorm3d_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<BatchNorm3d<f32>>();
    }

    // -----------------------------------------------------------------------
    // LocalResponseNorm tests — CL-435
    // -----------------------------------------------------------------------

    #[test]
    fn test_lrn_output_shape() {
        let lrn = LocalResponseNorm::new(5, 1e-4, 0.75, 1.0).unwrap();
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0f32; 2 * 4 * 3 * 3]),
            vec![2, 4, 3, 3],
            false,
        )
        .unwrap();
        let output = Module::<f32>::forward(&lrn, &input).unwrap();
        assert_eq!(output.shape(), &[2, 4, 3, 3]);
    }

    #[test]
    fn test_lrn_3d_input() {
        let lrn = LocalResponseNorm::new(3, 1e-4, 0.75, 1.0).unwrap();
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0f32; 2 * 4 * 8]),
            vec![2, 4, 8],
            false,
        )
        .unwrap();
        let output = Module::<f32>::forward(&lrn, &input).unwrap();
        assert_eq!(output.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_lrn_rejects_2d() {
        let lrn = LocalResponseNorm::new(3, 1e-4, 0.75, 1.0).unwrap();
        let input =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0f32; 8]), vec![2, 4], false)
                .unwrap();
        assert!(Module::<f32>::forward(&lrn, &input).is_err());
    }

    #[test]
    fn test_lrn_zero_size_rejected() {
        assert!(LocalResponseNorm::new(0, 1e-4, 0.75, 1.0).is_err());
    }

    #[test]
    fn test_lrn_default_params() {
        let lrn = LocalResponseNorm::default_params(5).unwrap();
        assert_eq!(lrn.size, 5);
        assert!((lrn.alpha - 1e-4).abs() < 1e-10);
        assert!((lrn.beta - 0.75).abs() < 1e-10);
        assert!((lrn.k - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_lrn_no_parameters() {
        let lrn = LocalResponseNorm::new(5, 1e-4, 0.75, 1.0).unwrap();
        assert!(Module::<f32>::parameters(&lrn).is_empty());
    }

    #[test]
    fn test_lrn_divides_by_norm() {
        // With large alpha and k=0 (edge case), output should be significantly
        // attenuated compared to input.
        let lrn = LocalResponseNorm::new(3, 10.0, 1.0, 1.0).unwrap();
        let data: Vec<f32> = vec![1.0; 3 * 2];
        let input =
            Tensor::<f32>::from_storage(TensorStorage::cpu(data), vec![1, 3, 2], false).unwrap();
        let output = Module::<f32>::forward(&lrn, &input).unwrap();
        let out_data = output.data().unwrap();

        // All outputs should be smaller than 1.0 since normalization divides.
        for &v in out_data.iter() {
            assert!(
                v < 1.0 && v > 0.0,
                "LRN output {v} should be attenuated (0 < v < 1)"
            );
        }
    }

    #[test]
    fn test_lrn_backward_numerical() {
        let lrn = LocalResponseNorm::new(3, 1e-4, 0.75, 1.0).unwrap();
        let input_data: Vec<f64> = vec![
            1.0, -0.5, 2.0, 0.3, 0.7, -1.2, 0.4, 1.5, -0.3, 0.8, 1.1, -0.7,
        ];
        let shape = vec![1usize, 3, 4];

        let input = leaf(&input_data, &shape, true);
        let output = Module::<f64>::forward(&lrn, &input).unwrap();
        let out_data = output.data().unwrap().to_vec();
        let total: f64 = out_data.iter().sum();

        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf).unwrap();
        loss.backward().unwrap();

        let analytic_grad = input.grad().unwrap().unwrap();
        let analytic = analytic_grad.data().unwrap().to_vec();

        let h = 1e-6;
        for i in 0..input_data.len() {
            let mut data_plus = input_data.clone();
            data_plus[i] += h;
            let inp_plus = leaf(&data_plus, &shape, false);
            let out_plus = no_grad(|| Module::<f64>::forward(&lrn, &inp_plus)).unwrap();
            let sum_plus: f64 = out_plus.data().unwrap().iter().sum();

            let mut data_minus = input_data.clone();
            data_minus[i] -= h;
            let inp_minus = leaf(&data_minus, &shape, false);
            let out_minus = no_grad(|| Module::<f64>::forward(&lrn, &inp_minus)).unwrap();
            let sum_minus: f64 = out_minus.data().unwrap().iter().sum();

            let numerical = (sum_plus - sum_minus) / (2.0 * h);
            assert!(
                (numerical - analytic[i]).abs() < 1e-4,
                "LRN grad[{i}]: numerical={numerical}, analytic={}",
                analytic[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // InstanceNorm tests — CL-315
    // -----------------------------------------------------------------------

    #[test]
    fn test_instancenorm1d_output_shape() {
        let norm = InstanceNorm1d::<f32>::new(3, 1e-5, true).unwrap();
        // Input [B=2, C=3, L=8]
        let input =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 48]), vec![2, 3, 8], false)
                .unwrap();
        let out = norm.forward(&input).unwrap();
        assert_eq!(out.shape(), &[2, 3, 8]);
    }

    #[test]
    fn test_instancenorm1d_rejects_wrong_ndim() {
        let norm = InstanceNorm1d::<f32>::new(3, 1e-5, true).unwrap();
        // 4D input should fail.
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 48]),
            vec![2, 3, 4, 2],
            false,
        )
        .unwrap();
        assert!(norm.forward(&input).is_err());
    }

    #[test]
    fn test_instancenorm2d_normalizes_per_instance_channel() {
        // Each (b, c) spatial plane should have ~zero mean, ~unit var after norm.
        let norm = InstanceNorm2d::<f32>::new(2, 1e-5, true).unwrap();
        // [B=1, C=2, H=2, W=2]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0
            5.0, 6.0, 7.0, 8.0, // channel 1
        ];
        let input =
            Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2, 2, 2], false).unwrap();
        let out = norm.forward(&input).unwrap();
        let d = out.data().unwrap();

        // Check each channel independently.
        for c in 0..2 {
            let start = c * 4;
            let end = start + 4;
            let slice = &d[start..end];
            let mean: f32 = slice.iter().sum::<f32>() / 4.0;
            let var: f32 = slice.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 4.0;
            assert!(mean.abs() < 1e-5, "channel {c} mean = {mean}, expected ~0");
            assert!(
                (var - 1.0).abs() < 0.1,
                "channel {c} var = {var}, expected ~1"
            );
        }
    }

    #[test]
    fn test_instancenorm2d_rejects_wrong_ndim() {
        let norm = InstanceNorm2d::<f32>::new(3, 1e-5, true).unwrap();
        // 3D input should fail.
        let input =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 24]), vec![2, 3, 4], false)
                .unwrap();
        assert!(norm.forward(&input).is_err());
    }

    #[test]
    fn test_instancenorm3d_output_shape() {
        let norm = InstanceNorm3d::<f32>::new(2, 1e-5, false).unwrap();
        // [B=1, C=2, D=2, H=2, W=2]
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 16]),
            vec![1, 2, 2, 2, 2],
            false,
        )
        .unwrap();
        let out = norm.forward(&input).unwrap();
        assert_eq!(out.shape(), &[1, 2, 2, 2, 2]);
    }

    #[test]
    fn test_instancenorm2d_no_affine_no_params() {
        let norm = InstanceNorm2d::<f32>::new(4, 1e-5, false).unwrap();
        assert!(Module::<f32>::parameters(&norm).is_empty());
    }

    #[test]
    fn test_instancenorm2d_has_affine_params() {
        let norm = InstanceNorm2d::<f32>::new(4, 1e-5, true).unwrap();
        let params = Module::<f32>::parameters(&norm);
        assert_eq!(params.len(), 2);
        assert_eq!(params[0].shape(), &[4]); // weight
        assert_eq!(params[1].shape(), &[4]); // bias
    }

    #[test]
    fn test_instancenorm2d_backward_gradient_check() {
        let h = 1e-7;
        let num_features = 2;
        // Input [1, 2, 2, 2]
        let input_data: Vec<f64> = vec![1.0, -0.5, 2.0, 0.3, 0.7, -1.2, 0.4, 1.5];
        let shape = vec![1usize, 2, 2, 2];

        let norm = InstanceNorm2d::<f64>::new(num_features, 1e-5, true).unwrap();

        // Forward + backward.
        let input = leaf(&input_data, &shape, true);
        let output = norm.forward(&input).unwrap();
        let out_data = output.data().unwrap().to_vec();
        let total: f64 = out_data.iter().sum();

        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf).unwrap();
        loss.backward().unwrap();

        let analytic_grad = input.grad().unwrap().unwrap();
        let analytic = analytic_grad.data().unwrap().to_vec();

        // Numerical gradient.
        for i in 0..input_data.len() {
            let mut data_plus = input_data.clone();
            data_plus[i] += h;
            let inp_plus = leaf(&data_plus, &shape, false);
            let out_plus = no_grad(|| norm.forward(&inp_plus)).unwrap();
            let sum_plus: f64 = out_plus.data().unwrap().iter().sum();

            let mut data_minus = input_data.clone();
            data_minus[i] -= h;
            let inp_minus = leaf(&data_minus, &shape, false);
            let out_minus = no_grad(|| norm.forward(&inp_minus)).unwrap();
            let sum_minus: f64 = out_minus.data().unwrap().iter().sum();

            let numerical = (sum_plus - sum_minus) / (2.0 * h);
            assert!(
                (numerical - analytic[i]).abs() < 1e-4,
                "InstanceNorm2d grad[{i}]: numerical={numerical}, analytic={}",
                analytic[i]
            );
        }
    }

    #[test]
    fn test_instancenorm_zero_features_rejected() {
        assert!(InstanceNorm1d::<f32>::new(0, 1e-5, true).is_err());
        assert!(InstanceNorm2d::<f32>::new(0, 1e-5, true).is_err());
        assert!(InstanceNorm3d::<f32>::new(0, 1e-5, true).is_err());
    }
}
