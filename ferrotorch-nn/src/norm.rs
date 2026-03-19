//! Normalization layers: LayerNorm, GroupNorm, RMSNorm, BatchNorm2d.
//!
//! Each layer normalizes its input along specified dimensions and optionally
//! applies a learnable affine transform (weight/bias). Backward functions
//! implement `GradFn<T>` to propagate gradients through the normalization.

use std::sync::{Arc, Mutex};

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::module::Module;
use crate::parameter::Parameter;

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
        let device = input.device();
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
                    Tensor::from_operation(
                        TensorStorage::gpu(handle),
                        shape,
                        grad_fn,
                    )
                } else {
                    Tensor::from_storage(TensorStorage::gpu(handle), shape, false)
                };
            }
        }

        // CPU path.
        let cpu_input = if input.is_cuda() { input.cpu()? } else { input.clone() };
        let input_data = cpu_input.data()?;
        let eps_t = T::from(self.eps).unwrap();
        let n_t = T::from(norm_size).unwrap();

        let cpu_weight = if self.weight.tensor().is_cuda() { self.weight.tensor().cpu()? } else { self.weight.tensor().clone() };
        let cpu_bias = if self.bias.tensor().is_cuda() { self.bias.tensor().cpu()? } else { self.bias.tensor().clone() };
        let weight_data = cpu_weight.data()?;
        let bias_data = cpu_bias.data()?;

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
            let out = Tensor::from_operation(
                storage,
                shape.to_vec(),
                grad_fn,
            )?;
            if device.is_cuda() { out.to(device) } else { Ok(out) }
        } else {
            let result = Tensor::from_storage(storage, shape.to_vec(), false)?;
            if device.is_cuda() { result.to(device) } else { Ok(result) }
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
        let n_t = T::from(norm_size).unwrap();
        let eps_t = T::from(self.eps).unwrap();

        let cpu_input = if self.input.is_cuda() { self.input.cpu()? } else { self.input.clone() };
        let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
        let cpu_weight = if self.weight.is_cuda() { self.weight.cpu()? } else { self.weight.clone() };
        let input_data = cpu_input.data()?;
        let go_data = cpu_go.data()?;
        let weight_data = cpu_weight.data()?;

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
            let var = x_slice
                .iter()
                .copied()
                .fold(zero::<T>(), |a, x| {
                    let d = x - mean;
                    a + d * d
                })
                / n_t;
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

                dl_dx_hat_sum = dl_dx_hat_sum + dl_dx_hat_i;
                dl_dx_hat_x_hat_sum = dl_dx_hat_x_hat_sum + dl_dx_hat_i * x_hat_i;

                if self.elementwise_affine {
                    grad_weight[i] = grad_weight[i] + go_slice[i] * x_hat_i;
                    grad_bias[i] = grad_bias[i] + go_slice[i];
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

        let device = self.input.device();
        let grad_input_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        let grad_input_tensor = if device.is_cuda() { grad_input_tensor.to(device)? } else { grad_input_tensor };

        let grad_weight_out = if self.elementwise_affine && self.weight.requires_grad() {
            let gw = Tensor::from_storage(
                TensorStorage::cpu(grad_weight),
                self.normalized_shape.clone(),
                false,
            )?;
            Some(if device.is_cuda() { gw.to(device)? } else { gw })
        } else {
            None
        };

        let grad_bias_out = if self.elementwise_affine && self.bias.requires_grad() {
            let gb = Tensor::from_storage(
                TensorStorage::cpu(grad_bias),
                self.normalized_shape.clone(),
                false,
            )?;
            Some(if device.is_cuda() { gb.to(device)? } else { gb })
        } else {
            None
        };

        Ok(vec![Some(grad_input_tensor), grad_weight_out, grad_bias_out])
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
        let device = input.device();
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

        // Transfer to CPU for computation if on GPU.
        let cpu_input = if input.is_cuda() { input.cpu()? } else { input.clone() };
        let input_data = cpu_input.data()?;
        let cpu_weight = if self.weight.tensor().is_cuda() { self.weight.tensor().cpu()? } else { self.weight.tensor().clone() };
        let cpu_bias = if self.bias.tensor().is_cuda() { self.bias.tensor().cpu()? } else { self.bias.tensor().clone() };
        let weight_data = cpu_weight.data()?;
        let bias_data = cpu_bias.data()?;
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
                        sum = sum + input_data[idx];
                    }
                }
                let mean = sum / group_n;

                // Compute variance over the group.
                let mut var_sum = zero::<T>();
                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let d = input_data[idx] - mean;
                        var_sum = var_sum + d * d;
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

        let result =
            Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?;

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
            let out = Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?;
            if device.is_cuda() { out.to(device) } else { Ok(out) }
        } else if device.is_cuda() {
            result.to(device)
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

        let cpu_input = if self.input.is_cuda() { self.input.cpu()? } else { self.input.clone() };
        let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
        let cpu_weight = if self.weight.is_cuda() { self.weight.cpu()? } else { self.weight.clone() };
        let input_data = cpu_input.data()?;
        let go_data = cpu_go.data()?;
        let weight_data = cpu_weight.data()?;

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
                        sum = sum + input_data[idx];
                    }
                }
                let mean = sum / group_n;

                let mut var_sum = zero::<T>();
                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let d = input_data[idx] - mean;
                        var_sum = var_sum + d * d;
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
                        dl_dx_hat_sum = dl_dx_hat_sum + dl_dx_hat;
                        dl_dx_hat_x_hat_sum = dl_dx_hat_x_hat_sum + dl_dx_hat * x_hat;

                        if self.affine {
                            grad_weight[c] = grad_weight[c] + go_data[idx] * x_hat;
                            grad_bias[c] = grad_bias[c] + go_data[idx];
                        }
                    }
                }

                let dl_dx_hat_mean = dl_dx_hat_sum / group_n;
                let dl_dx_hat_x_hat_mean = dl_dx_hat_x_hat_sum / group_n;

                for c in c_start..c_end {
                    for s in 0..spatial {
                        let idx = b * channels * spatial + c * spatial + s;
                        let x_hat = (input_data[idx] - mean) * inv_std;
                        let dl_dx_hat = if self.affine {
                            go_data[idx] * weight_data[c]
                        } else {
                            go_data[idx]
                        };
                        grad_input[idx] = inv_std
                            * (dl_dx_hat - dl_dx_hat_mean - x_hat * dl_dx_hat_x_hat_mean);
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

        Ok(vec![Some(grad_input_tensor), grad_weight_out, grad_bias_out])
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
        let device = input.device();
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

        // Transfer to CPU for computation if on GPU.
        let cpu_input = if input.is_cuda() { input.cpu()? } else { input.clone() };
        let input_data = cpu_input.data()?;
        let cpu_weight = if self.weight.tensor().is_cuda() { self.weight.tensor().cpu()? } else { self.weight.tensor().clone() };
        let weight_data = cpu_weight.data()?;
        let eps_t = T::from(self.eps).unwrap();
        let n_t = T::from(norm_size).unwrap();

        let mut output = Vec::with_capacity(input.numel());

        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let slice = &input_data[start..end];

            // rms = sqrt(mean(x^2) + eps)
            let mean_sq =
                slice.iter().copied().fold(zero::<T>(), |a, x| a + x * x) / n_t;
            let rms = (mean_sq + eps_t).sqrt();
            let inv_rms = rms.recip();

            for (i, &x) in slice.iter().enumerate() {
                output.push(x * inv_rms * weight_data[i]);
            }
        }

        let result =
            Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(RMSNormBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                normalized_shape: self.normalized_shape.clone(),
                eps: self.eps,
            });
            let out = Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?;
            if device.is_cuda() { out.to(device) } else { Ok(out) }
        } else if device.is_cuda() {
            result.to(device)
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
        let n_t = T::from(norm_size).unwrap();
        let eps_t = T::from(self.eps).unwrap();

        let cpu_input = if self.input.is_cuda() { self.input.cpu()? } else { self.input.clone() };
        let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
        let cpu_weight = if self.weight.is_cuda() { self.weight.cpu()? } else { self.weight.clone() };
        let input_data = cpu_input.data()?;
        let go_data = cpu_go.data()?;
        let weight_data = cpu_weight.data()?;

        let mut grad_input = vec![zero::<T>(); self.input.numel()];
        let mut grad_weight = vec![zero::<T>(); norm_size];

        for b in 0..batch_size {
            let start = b * norm_size;
            let end = start + norm_size;
            let x_slice = &input_data[start..end];
            let go_slice = &go_data[start..end];

            // Recompute rms.
            let mean_sq =
                x_slice.iter().copied().fold(zero::<T>(), |a, x| a + x * x) / n_t;
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
                    * (go_slice[i] * weight_data[i]
                        - x_slice[i] * inv_rms_sq * go_x_w_mean);

                // grad_weight_i += go_i * x_i * inv_rms
                grad_weight[i] = grad_weight[i] + go_slice[i] * x_slice[i] * inv_rms;
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
        let device = input.device();
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

        // Transfer to CPU for computation if on GPU.
        let cpu_input = if input.is_cuda() { input.cpu()? } else { input.clone() };
        let input_data = cpu_input.data()?;
        let eps_t = T::from(self.eps).unwrap();

        let cpu_weight = self.weight.as_ref().map(|w| {
            if w.tensor().is_cuda() { w.tensor().cpu().unwrap() } else { w.tensor().clone() }
        });
        let cpu_bias = self.bias.as_ref().map(|b| {
            if b.tensor().is_cuda() { b.tensor().cpu().unwrap() } else { b.tensor().clone() }
        });
        let weight_data = cpu_weight.as_ref().map(|w| w.data().unwrap());
        let bias_data = cpu_bias.as_ref().map(|b| b.data().unwrap());

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
                        sum = sum + input_data[base + s];
                    }
                }
                chan_mean[c] = sum / count_t;

                let mut var_sum = zero::<T>();
                for b in 0..batch {
                    let base = b * channels * spatial + c * spatial;
                    for s in 0..spatial {
                        let d = input_data[base + s] - chan_mean[c];
                        var_sum = var_sum + d * d;
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

        let result =
            Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?;

        if is_grad_enabled() && input.requires_grad() {
            let weight_tensor = self.weight.as_ref().map(|w| w.tensor().clone());
            let bias_tensor = self.bias.as_ref().map(|b| b.tensor().clone());

            let grad_fn = Arc::new(BatchNorm2dBackward {
                input: input.clone(),
                x_hat: Tensor::from_storage(
                    TensorStorage::cpu(x_hat_data),
                    shape.to_vec(),
                    false,
                )?,
                weight: weight_tensor,
                bias: bias_tensor,
                chan_var: chan_var.iter().map(|v| v.to_f64().unwrap()).collect(),
                eps: self.eps,
                affine: self.affine,
            });

            let out = Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?;
            if device.is_cuda() { out.to(device) } else { Ok(out) }
        } else if device.is_cuda() {
            result.to(device)
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
            (Some(w), Some(b)) => vec![
                ("weight".to_string(), w),
                ("bias".to_string(), b),
            ],
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

        let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
        let cpu_x_hat = if self.x_hat.is_cuda() { self.x_hat.cpu()? } else { self.x_hat.clone() };
        let go_data = cpu_go.data()?;
        let x_hat_data = cpu_x_hat.data()?;

        let weight_data = self.weight.as_ref().map(|w| {
            let cpu_w = if w.is_cuda() { w.cpu().unwrap() } else { w.clone() };
            cpu_w.data().unwrap().to_vec()
        });

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

                    dl_dx_hat_sum = dl_dx_hat_sum + dl_dx_hat;
                    dl_dx_hat_x_hat_sum = dl_dx_hat_x_hat_sum + dl_dx_hat * x_h;

                    if self.affine {
                        grad_weight[c] = grad_weight[c] + go * x_h;
                        grad_bias[c] = grad_bias[c] + go;
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

        Ok(vec![Some(grad_input_tensor), grad_weight_out, grad_bias_out])
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
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::autograd::no_grad::no_grad;

    /// Helper: create a leaf tensor with given data, shape, and requires_grad.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), requires_grad)
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
        let input =
            Tensor::from_storage(TensorStorage::cpu(data), vec![2, 8], false).unwrap();

        let ln = LayerNorm::<f32>::new(vec![8], 1e-5, true).unwrap();
        let output = ln.forward(&input).unwrap();
        let out_data = output.data().unwrap();

        for row in 0..2 {
            let start = row * 8;
            let end = start + 8;
            let row_data = &out_data[start..end];

            let mean: f32 = row_data.iter().sum::<f32>() / 8.0;
            let var: f32 =
                row_data.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / 8.0;

            assert!(mean.abs() < 1e-5, "row {row} mean = {mean}, expected ~0");
            assert!(
                (var - 1.0).abs() < 0.05,
                "row {row} var = {var}, expected ~1"
            );
        }
    }

    #[test]
    fn test_layer_norm_forward_shape_preserved() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0; 24]),
            vec![2, 3, 4],
            false,
        )
        .unwrap();

        let ln = LayerNorm::<f32>::new(vec![4], 1e-5, true).unwrap();
        let output = ln.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_layer_norm_shape_mismatch() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0; 12]),
            vec![3, 4],
            false,
        )
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
        let loss =
            Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf)?;
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
        let input =
            Tensor::from_storage(TensorStorage::cpu(data), vec![1, 4, 2], false).unwrap();

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
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0; 48]),
            vec![2, 4, 6],
            false,
        )
        .unwrap();

        let gn = GroupNorm::<f32>::new(2, 4, 1e-5, true).unwrap();
        let output = gn.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 4, 6]);
    }

    #[test]
    fn test_group_norm_channel_mismatch() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0; 24]),
            vec![2, 3, 4],
            false,
        )
        .unwrap();

        let gn = GroupNorm::<f32>::new(2, 4, 1e-5, true).unwrap();
        assert!(gn.forward(&input).is_err());
    }

    #[test]
    fn test_group_norm_has_grad_fn() {
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0; 8]),
            vec![1, 4, 2],
            true,
        )
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
        let loss =
            Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf)?;
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
        let input =
            Tensor::from_storage(TensorStorage::cpu(data), vec![2, 4], false).unwrap();

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
        let input = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0; 24]),
            vec![2, 3, 4],
            false,
        )
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
        let loss =
            Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf)?;
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
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 24]),
            vec![2, 4, 3],
            false,
        )
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
        let input = Tensor::from_storage(
            TensorStorage::cpu(data),
            vec![b, channels, h, w],
            false,
        )
        .unwrap();

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

            assert!(
                mean.abs() < 1e-4,
                "channel {c}: mean = {mean}, expected ~0"
            );
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
        let input_data: Vec<f64> = (0..numel)
            .map(|i| (i as f64) * 0.3 - 1.0)
            .collect();

        let bn = BatchNorm2d::<f64>::new(channels, 1e-5, 0.1, true)?;

        let input = leaf(&input_data, &[b, channels, height, width], true);
        let output = bn.forward(&input)?;
        let out_data = output.data()?.to_vec();
        let total: f64 = out_data.iter().sum();

        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss =
            Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf)?;
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

        let input = Tensor::from_storage(
            TensorStorage::cpu(data),
            vec![b, channels, h, w],
            false,
        )
        .unwrap();

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

            assert!(
                mean.abs() < 1e-4,
                "no-affine channel {c}: mean = {mean}"
            );
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
        fn backward(
            &self,
            _grad_output: &Tensor<T>,
        ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
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
}
