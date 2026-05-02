//! Stateless functional API for common neural network operations.
//!
//! This module mirrors PyTorch's `torch.nn.functional` — every function takes
//! explicit weight/bias tensors (where applicable) instead of encapsulating
//! them in a `Module`. This is useful when you want full control over
//! parameters or need to share weights between layers.
//!
//! # Autograd
//!
//! All operations delegate to the differentiable primitives in
//! `ferrotorch_core::grad_fns`, so the backward graph is built automatically
//! when gradient tracking is enabled.

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::grad_fns::activation as act;
use ferrotorch_core::grad_fns::arithmetic;
use ferrotorch_core::grad_fns::linalg::mm_differentiable;
use ferrotorch_core::grad_fns::reduction as red;
use ferrotorch_core::grad_fns::shape::transpose_2d;
use ferrotorch_core::grad_fns::transcendental as trans;
use ferrotorch_core::ops::elementwise::{binary_map, mean as elem_mean};
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

// ===========================================================================
// Linear
// ===========================================================================

/// Applies a linear transformation: `output = input @ weight^T + bias`.
///
/// This is the stateless equivalent of [`crate::Linear`]. The weight and bias
/// are passed explicitly rather than owned by a module.
///
/// # Shapes
///
/// - `input`: `[B, in_features]`
/// - `weight`: `[out_features, in_features]`
/// - `bias` (optional): `[out_features]`
/// - **returns**: `[B, out_features]`
///
/// # Examples
///
/// ```ignore
/// let output = functional::linear(&input, &weight, Some(&bias))?;
/// ```
pub fn linear<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
) -> FerrotorchResult<Tensor<T>> {
    // Validate input shape.
    if input.ndim() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "functional::linear expects 2D input [B, in_features], got shape {:?}",
                input.shape()
            ),
        });
    }

    // Validate weight shape.
    if weight.ndim() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "functional::linear expects 2D weight [out, in], got shape {:?}",
                weight.shape()
            ),
        });
    }

    let in_features = input.shape()[1];
    let weight_in = weight.shape()[1];
    let out_features = weight.shape()[0];

    if in_features != weight_in {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "functional::linear: input has {} features but weight expects {}",
                in_features, weight_in,
            ),
        });
    }

    // weight^T: [in_features, out_features]
    let weight_t = transpose_2d(weight)?;

    // input @ weight^T: [B, out_features]
    let output = mm_differentiable(input, &weight_t)?;

    // Add bias if present.
    match bias {
        Some(b) => {
            if b.ndim() != 1 || b.shape()[0] != out_features {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "functional::linear: bias shape {:?} does not match out_features {}",
                        b.shape(),
                        out_features,
                    ),
                });
            }
            // Reshape bias to [1, out_features] for broadcast addition.
            let bias_data = b.data()?;
            let bias_2d = Tensor::from_storage(
                TensorStorage::cpu(bias_data.to_vec()),
                vec![1, out_features],
                b.requires_grad(),
            )?;
            arithmetic::add(&output, &bias_2d)
        }
        None => Ok(output),
    }
}

// ===========================================================================
// Activations
// ===========================================================================

/// Applies the rectified linear unit function elementwise: `relu(x) = max(0, x)`.
///
/// Differentiable — attaches `ReluBackward` when gradient tracking is enabled.
#[inline]
pub fn relu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::relu(input)
}

/// Applies the logistic sigmoid function elementwise: `sigmoid(x) = 1 / (1 + exp(-x))`.
///
/// Differentiable — attaches `SigmoidBackward` when gradient tracking is enabled.
#[inline]
pub fn sigmoid<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::sigmoid(input)
}

/// Applies the hyperbolic tangent function elementwise.
///
/// Differentiable — attaches `TanhBackward` when gradient tracking is enabled.
#[inline]
pub fn tanh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::tanh(input)
}

/// Applies the Gaussian Error Linear Unit with the default exact (erf) mode.
///
/// For other approximation modes, use [`gelu_with`].
/// Differentiable — attaches `GeluBackward` when gradient tracking is enabled.
#[inline]
pub fn gelu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::gelu(input)
}

/// Applies GELU with a configurable approximation mode.
///
/// See [`act::GeluApproximate`] for available modes.
#[inline]
pub fn gelu_with<T: Float>(
    input: &Tensor<T>,
    approximate: act::GeluApproximate,
) -> FerrotorchResult<Tensor<T>> {
    act::gelu_with(input, approximate)
}

/// Applies the Sigmoid Linear Unit (Swish) function:
/// `silu(x) = x * sigmoid(x)`.
///
/// Differentiable — attaches `SiluBackward` when gradient tracking is enabled.
#[inline]
pub fn silu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::silu(input)
}

/// Applies softmax along the last dimension.
///
/// Differentiable — attaches `SoftmaxBackward` when gradient tracking is enabled.
#[inline]
pub fn softmax<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::softmax(input)
}

/// Applies log-softmax along the last dimension.
///
/// More numerically stable than computing `log(softmax(x))` separately.
/// Differentiable — attaches `LogSoftmaxBackward` when gradient tracking is enabled.
#[inline]
pub fn log_softmax<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::log_softmax(input)
}

/// Applies the leaky rectified linear unit function:
/// `leaky_relu(x) = max(0, x) + negative_slope * min(0, x)`.
///
/// Implemented as `(1 - negative_slope) * relu(x) + negative_slope * x`
/// using composable differentiable primitives so autograd works automatically.
///
/// # Arguments
///
/// - `input` — Input tensor of any shape.
/// - `negative_slope` — Slope for negative inputs (default in PyTorch: 0.01).
pub fn leaky_relu<T: Float>(input: &Tensor<T>, negative_slope: f64) -> FerrotorchResult<Tensor<T>> {
    if (negative_slope - 0.0).abs() < f64::EPSILON {
        // Degenerate case: standard ReLU.
        return act::relu(input);
    }
    if (negative_slope - 1.0).abs() < f64::EPSILON {
        // Degenerate case: identity.
        return Ok(input.clone());
    }

    // relu_x = relu(input)
    let relu_x = act::relu(input)?;

    let scale = T::from(1.0 - negative_slope).unwrap();
    let slope = T::from(negative_slope).unwrap();

    let scale_tensor = ferrotorch_core::scalar(scale)?;
    let slope_tensor = ferrotorch_core::scalar(slope)?;

    // result = (1 - negative_slope) * relu(x) + negative_slope * x
    let scaled_relu = arithmetic::mul(&relu_x, &scale_tensor)?;
    let scaled_x = arithmetic::mul(input, &slope_tensor)?;
    arithmetic::add(&scaled_relu, &scaled_x)
}

// ===========================================================================
// Reduction
// ===========================================================================

/// Reduces the tensor to a scalar by summing all elements.
///
/// Differentiable — attaches `SumBackward` when gradient tracking is enabled.
#[inline]
pub fn sum<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    red::sum(input)
}

/// Reduces the tensor to a scalar by computing the mean of all elements.
///
/// Differentiable — attaches `MeanBackward` when gradient tracking is enabled.
#[inline]
pub fn mean<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    red::mean(input)
}

// ===========================================================================
// Dropout
// ===========================================================================

// Internal xorshift PRNG — mirrors the implementation in `crate::dropout`.
fn xorshift_seed() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let mut state = hasher.finish();
    if state == 0 {
        state = 0xdeadbeefcafe;
    }
    state
}

#[inline]
fn xorshift_next(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

/// Backward node for functional dropout.
#[derive(Debug)]
struct DropoutBackward<T: Float> {
    input: Tensor<T>,
    scaled_mask: Vec<T>,
}

impl<T: Float> GradFn<T> for DropoutBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go_data = grad_output.data_vec()?;
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(self.scaled_mask.iter())
                .map(|(&g, &m)| g * m)
                .collect();
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.input.shape().to_vec(),
                false,
            )?;
            Some(if self.input.is_cuda() {
                g.to(self.input.device())?
            } else {
                g
            })
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "DropoutBackward"
    }
}

/// Randomly zeroes elements with probability `p` during training.
///
/// When `training` is `false`, returns the input unchanged. When `training`
/// is `true`, each element is independently zeroed with probability `p` and
/// surviving elements are scaled by `1/(1-p)` (inverted dropout).
///
/// This is the stateless equivalent of [`crate::Dropout`].
///
/// # Arguments
///
/// - `input` — Input tensor of any shape.
/// - `p` — Probability of an element being zeroed. Must be in `[0, 1)`.
/// - `training` — If `false`, the input is returned unchanged.
///
/// # Errors
///
/// Returns an error if `p` is outside `[0, 1)`.
pub fn dropout<T: Float>(input: &Tensor<T>, p: f64, training: bool) -> FerrotorchResult<Tensor<T>> {
    if !(0.0..1.0).contains(&p) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("dropout probability must be in [0, 1), got {p}"),
        });
    }

    // Eval mode or p == 0: identity.
    if !training || p == 0.0 {
        return Ok(input.clone());
    }

    let numel = input.numel();
    let scale = T::from(1.0 / (1.0 - p)).unwrap();
    let zero = <T as num_traits::Zero>::zero();

    let mut state = xorshift_seed();
    let scaled_mask: Vec<T> = (0..numel)
        .map(|_| {
            if xorshift_next(&mut state) < p {
                zero
            } else {
                scale
            }
        })
        .collect();

    // Forward: element-wise multiply input by scaled mask.
    let device = input.device();
    let input_data = input.data_vec()?;
    let output_data: Vec<T> = input_data
        .iter()
        .zip(scaled_mask.iter())
        .map(|(&x, &m)| x * m)
        .collect();

    let result = if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(output_data),
            input.shape().to_vec(),
            Arc::new(DropoutBackward {
                input: input.clone(),
                scaled_mask,
            }),
        )?
    } else {
        Tensor::from_storage(
            TensorStorage::cpu(output_data),
            input.shape().to_vec(),
            false,
        )?
    };
    if device.is_cuda() {
        result.to(device)
    } else {
        Ok(result)
    }
}

// ===========================================================================
// Loss functions
// ===========================================================================

/// Mean Squared Error loss: `mean((pred - target)^2)`.
///
/// This is a convenience wrapper that always uses mean reduction, matching
/// the most common usage. For configurable reduction, use [`crate::MSELoss`].
///
/// # Shapes
///
/// - `pred` and `target` must have the same shape.
/// - **returns**: scalar tensor.
pub fn mse_loss<T: Float>(pred: &Tensor<T>, target: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if pred.shape() != target.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mse_loss: pred shape {:?} != target shape {:?}",
                pred.shape(),
                target.shape(),
            ),
        });
    }

    // diff = pred - target
    let diff = binary_map(pred, target, |p, t| p - t)?;
    // sq = diff^2
    let sq = ferrotorch_core::ops::elementwise::unary_map(&diff, |x| x * x)?;
    // loss = mean(sq)
    let reduced = elem_mean(&sq)?;

    if is_grad_enabled() && pred.requires_grad() {
        let grad_fn = Arc::new(MSEBackward {
            pred: pred.clone(),
            target: target.clone(),
        });
        Tensor::from_operation(
            TensorStorage::cpu(reduced.data_vec()?),
            reduced.shape().to_vec(),
            grad_fn,
        )
    } else {
        Ok(reduced)
    }
}

/// Backward for functional `mse_loss` (mean reduction).
///
/// `grad_pred = 2 * (pred - target) * grad_output / n`
#[derive(Debug)]
struct MSEBackward<T: Float> {
    pred: Tensor<T>,
    target: Tensor<T>,
}

impl<T: Float> GradFn<T> for MSEBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let pred_data = self.pred.data_vec()?;
        let target_data = self.target.data_vec()?;
        let grad_data = grad_output.data_vec()?;
        let two = T::from(2.0).unwrap();
        let n = T::from(pred_data.len()).unwrap();
        let go = grad_data[0];

        let result: Vec<T> = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(&p, &t)| two * (p - t) * go / n)
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.pred.shape().to_vec(),
            false,
        )?;
        let grad_input = grad_input.to(self.pred.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.pred]
    }

    fn name(&self) -> &'static str {
        "MSEBackward"
    }
}

/// Cross-entropy loss combining log-softmax and negative log-likelihood.
///
/// Expects logits `[B, C]` and integer class targets `[B]` (stored as
/// floats, e.g. `0.0`, `1.0`, `2.0`). Always uses mean reduction.
///
/// For configurable reduction and label smoothing, use [`crate::CrossEntropyLoss`].
///
/// # Shapes
///
/// - `logits`: `[B, C]` (raw scores, **not** probabilities).
/// - `targets`: `[B]` (class indices as floats).
/// - **returns**: scalar tensor.
pub fn cross_entropy<T: Float>(
    logits: &Tensor<T>,
    targets: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let shape = logits.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "cross_entropy: expected 2D logits [B, C], got shape {:?}",
                shape,
            ),
        });
    }
    let batch = shape[0];
    let classes = shape[1];

    if targets.shape() != [batch] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "cross_entropy: target shape {:?} does not match batch size {}",
                targets.shape(),
                batch,
            ),
        });
    }

    let logits_data = logits.data_vec()?;
    let targets_data = targets.data_vec()?;

    // Compute log_softmax along dim=-1 and the softmax values (for backward).
    let mut log_probs = vec![<T as num_traits::Zero>::zero(); batch * classes];
    let mut softmax_out = vec![<T as num_traits::Zero>::zero(); batch * classes];

    for b in 0..batch {
        let base = b * classes;
        // Numerical stability: subtract max.
        let mut max_val = logits_data[base];
        for c in 1..classes {
            if logits_data[base + c] > max_val {
                max_val = logits_data[base + c];
            }
        }
        let mut sum_exp = <T as num_traits::Zero>::zero();
        for c in 0..classes {
            let e = (logits_data[base + c] - max_val).exp();
            softmax_out[base + c] = e;
            sum_exp += e;
        }
        let log_sum = sum_exp.ln();
        for c in 0..classes {
            softmax_out[base + c] = softmax_out[base + c] / sum_exp;
            log_probs[base + c] = logits_data[base + c] - max_val - log_sum;
        }
    }

    // Compute per-sample NLL and reduce with mean.
    let mut total_loss = <T as num_traits::Zero>::zero();
    for (b, &target) in targets_data.iter().enumerate() {
        let base = b * classes;
        let target_class = target.to_usize().unwrap_or(0);
        total_loss = total_loss - log_probs[base + target_class];
    }
    let loss_val = total_loss / T::from(batch).unwrap();

    let reduced = Tensor::from_storage(TensorStorage::cpu(vec![loss_val]), vec![], false)?;

    if is_grad_enabled() && logits.requires_grad() {
        let softmax_tensor =
            Tensor::from_storage(TensorStorage::cpu(softmax_out), vec![batch, classes], false)?;
        let grad_fn = Arc::new(CrossEntropyBackward {
            logits: logits.clone(),
            targets: targets.clone(),
            softmax: softmax_tensor,
        });
        Tensor::from_operation(
            TensorStorage::cpu(reduced.data_vec()?),
            reduced.shape().to_vec(),
            grad_fn,
        )
    } else {
        Ok(reduced)
    }
}

/// Backward for functional `cross_entropy` (mean reduction, no label smoothing).
///
/// `grad_logits[b, c] = (softmax[b, c] - one_hot[b, c]) / B`
#[derive(Debug)]
struct CrossEntropyBackward<T: Float> {
    logits: Tensor<T>,
    targets: Tensor<T>,
    softmax: Tensor<T>,
}

impl<T: Float> GradFn<T> for CrossEntropyBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.logits.shape();
        let batch = shape[0];
        let classes = shape[1];
        let sm_data = self.softmax.data_vec()?;
        let targets_data = self.targets.data_vec()?;
        let grad_data = grad_output.data_vec()?;
        let go = grad_data[0];

        let mut result = vec![<T as num_traits::Zero>::zero(); batch * classes];
        let inv_batch = T::from(1.0).unwrap() / T::from(batch).unwrap();

        for (b, &target) in targets_data.iter().enumerate() {
            let base = b * classes;
            let target_class = target.to_usize().unwrap_or(0);
            for c in 0..classes {
                let one_hot = if c == target_class {
                    <T as num_traits::One>::one()
                } else {
                    <T as num_traits::Zero>::zero()
                };
                result[base + c] = (sm_data[base + c] - one_hot) * inv_batch * go;
            }
        }

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.logits.shape().to_vec(),
            false,
        )?;
        let grad_input = grad_input.to(self.logits.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.logits]
    }

    fn name(&self) -> &'static str {
        "CrossEntropyBackward"
    }
}

// ===========================================================================
// Interpolation / Vision ops (CL-317)
// ===========================================================================

// Re-export types for convenience.
pub use crate::upsample::{GridSampleMode, GridSamplePaddingMode, InterpolateMode};

/// Spatially resize a `[B, C, H, W]` tensor using the specified interpolation mode.
///
/// Exactly one of `size` or `scale_factor` must be provided.
///
/// See [`crate::upsample::interpolate`] for full documentation.
///
/// CL-317
pub fn interpolate<T: Float>(
    input: &Tensor<T>,
    size: Option<[usize; 2]>,
    scale_factor: Option<[f64; 2]>,
    mode: InterpolateMode,
    align_corners: bool,
) -> FerrotorchResult<Tensor<T>> {
    crate::upsample::interpolate(input, size, scale_factor, mode, align_corners)
}

/// Sample `input` at spatial locations specified by `grid` (spatial transformer).
///
/// See [`crate::upsample::grid_sample`] for full documentation.
///
/// CL-317
pub fn grid_sample<T: Float>(
    input: &Tensor<T>,
    grid: &Tensor<T>,
    mode: GridSampleMode,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> FerrotorchResult<Tensor<T>> {
    crate::upsample::grid_sample(input, grid, mode, padding_mode, align_corners)
}

/// Generate a 2D affine sampling grid for use with [`grid_sample`].
///
/// See [`crate::upsample::affine_grid`] for full documentation.
///
/// CL-317
pub fn affine_grid<T: Float>(
    theta: &Tensor<T>,
    size: [usize; 4],
    align_corners: bool,
) -> FerrotorchResult<Tensor<T>> {
    crate::upsample::affine_grid(theta, size, align_corners)
}

/// Pixel shuffle: `[B, C*r*r, H, W]` -> `[B, C, H*r, W*r]`.
///
/// See [`crate::upsample::pixel_shuffle`] for full documentation.
///
/// CL-317
pub fn pixel_shuffle<T: Float>(
    input: &Tensor<T>,
    upscale_factor: usize,
) -> FerrotorchResult<Tensor<T>> {
    crate::upsample::pixel_shuffle(input, upscale_factor)
}

/// Pixel unshuffle: `[B, C, H*r, W*r]` -> `[B, C*r*r, H, W]`.
///
/// See [`crate::upsample::pixel_unshuffle`] for full documentation.
///
/// CL-317
pub fn pixel_unshuffle<T: Float>(
    input: &Tensor<T>,
    downscale_factor: usize,
) -> FerrotorchResult<Tensor<T>> {
    crate::upsample::pixel_unshuffle(input, downscale_factor)
}

/// Unfold (im2col): `[B, C, H, W]` -> `[B, C*kH*kW, L]`.
///
/// See [`crate::upsample::unfold`] for full documentation.
///
/// CL-317
pub fn unfold<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 2],
    dilation: [usize; 2],
    padding: [usize; 2],
    stride: [usize; 2],
) -> FerrotorchResult<Tensor<T>> {
    crate::upsample::unfold(input, kernel_size, dilation, padding, stride)
}

/// Fold (col2im): `[B, C*kH*kW, L]` -> `[B, C, H, W]`.
///
/// See [`crate::upsample::fold`] for full documentation.
///
/// CL-317
pub fn fold<T: Float>(
    input: &Tensor<T>,
    output_size: [usize; 2],
    kernel_size: [usize; 2],
    dilation: [usize; 2],
    padding: [usize; 2],
    stride: [usize; 2],
) -> FerrotorchResult<Tensor<T>> {
    crate::upsample::fold(input, output_size, kernel_size, dilation, padding, stride)
}

// ===========================================================================
// Activation parity expansion (#584)
// ===========================================================================

/// Saturating hyperbolic tangent: `hardtanh(x) = clamp(x, min_val, max_val)`.
///
/// Default range matches `torch.nn.functional.hardtanh`: `[-1, 1]`. Pass
/// custom bounds via [`hardtanh_with`] when you need a different range.
#[inline]
pub fn hardtanh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    hardtanh_with(input, T::from(-1.0).unwrap(), T::from(1.0).unwrap())
}

/// `hardtanh` with explicit bounds. Differentiable via [`trans::clamp`];
/// gradient is zero outside `[min_val, max_val]`.
pub fn hardtanh_with<T: Float>(
    input: &Tensor<T>,
    min_val: T,
    max_val: T,
) -> FerrotorchResult<Tensor<T>> {
    trans::clamp(input, min_val, max_val)
}

/// `relu6(x) = clamp(x, 0, 6)`. Used by MobileNet et al.
#[inline]
pub fn relu6<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    trans::clamp(input, T::from(0.0).unwrap(), T::from(6.0).unwrap())
}

/// `hardsigmoid(x) = clamp((x + 3) / 6, 0, 1)`. The piecewise-linear
/// sigmoid approximation used in MobileNetV3.
pub fn hardsigmoid<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let three = ferrotorch_core::scalar(T::from(3.0).unwrap())?;
    let inv_six = ferrotorch_core::scalar(T::from(1.0 / 6.0).unwrap())?;
    let shifted = arithmetic::add(input, &three)?;
    let scaled = arithmetic::mul(&shifted, &inv_six)?;
    trans::clamp(&scaled, T::from(0.0).unwrap(), T::from(1.0).unwrap())
}

/// `hardswish(x) = x * hardsigmoid(x)`. Smooth-ish approximation to SiLU
/// used in mobile architectures.
pub fn hardswish<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let hs = hardsigmoid(input)?;
    arithmetic::mul(input, &hs)
}

/// `log_sigmoid(x) = log(sigmoid(x))`, computed numerically-stably as
/// `-softplus(-x)`.
pub fn log_sigmoid<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let neg = arithmetic::neg(input)?;
    // Use beta=1, threshold=20 (matches torch defaults).
    let sp = act::softplus(&neg, 1.0, 20.0)?;
    arithmetic::neg(&sp)
}

/// `softmin(x) = softmax(-x)`. Operates along the last dimension.
pub fn softmin<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let neg = arithmetic::neg(input)?;
    act::softmax(&neg)
}

/// `softsign(x) = x / (1 + |x|)`. A smooth alternative to `tanh` with
/// rational asymptotes.
pub fn softsign<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let abs_x = arithmetic::abs(input)?;
    let one = ferrotorch_core::scalar(T::from(1.0).unwrap())?;
    let denom = arithmetic::add(&abs_x, &one)?;
    arithmetic::div(input, &denom)
}

/// `tanhshrink(x) = x - tanh(x)`.
pub fn tanhshrink<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let t = act::tanh(input)?;
    arithmetic::sub(input, &t)
}

/// SELU: `selu(x) = scale * elu(x, alpha)` with the canonical
/// self-normalizing constants `alpha ≈ 1.6732632...`,
/// `scale ≈ 1.0507009...` (Klambauer et al. 2017).
pub fn selu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    const ALPHA: f64 = 1.6732632423543772;
    const SCALE: f64 = 1.0507009873554805;
    let e = act::elu(input, ALPHA)?;
    let scale = ferrotorch_core::scalar(T::from(SCALE).unwrap())?;
    arithmetic::mul(&e, &scale)
}

/// `softplus(x) = log(1 + exp(x))` — numerically stable smooth-ReLU
/// approximation. Defaults to `beta=1, threshold=20` matching torch.
#[inline]
pub fn softplus<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::softplus(input, 1.0, 20.0)
}

/// `softplus(x; beta, threshold)` with explicit parameters.
#[inline]
pub fn softplus_with<T: Float>(
    input: &Tensor<T>,
    beta: f64,
    threshold: f64,
) -> FerrotorchResult<Tensor<T>> {
    act::softplus(input, beta, threshold)
}

/// `elu(x; alpha) = x where x > 0 else alpha * (exp(x) - 1)`. Default
/// `alpha = 1.0` matches torch.
#[inline]
pub fn elu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::elu(input, 1.0)
}

/// `elu` with explicit alpha.
#[inline]
pub fn elu_with<T: Float>(input: &Tensor<T>, alpha: f64) -> FerrotorchResult<Tensor<T>> {
    act::elu(input, alpha)
}

/// `mish(x) = x * tanh(softplus(x))`. Differentiable via the existing
/// `MishBackward` in `ferrotorch_core::grad_fns::activation`.
#[inline]
pub fn mish<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::mish(input)
}

/// Gated Linear Unit: split the input along `dim` into halves `(a, b)`,
/// return `a * sigmoid(b)`. The size of `dim` must be even.
pub fn glu<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    act::glu(input, dim)
}

/// Functional Parametric ReLU. `alpha` must be a scalar (numel==1) tensor.
/// Mirrors `torch.nn.functional.prelu`. Single fused forward + backward node.
pub fn prelu<T: Float>(input: &Tensor<T>, alpha: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    act::prelu(input, alpha)
}

// ===========================================================================
// Loss parity expansion (#584)
// ===========================================================================

/// Apply the requested reduction to a per-element loss tensor.
fn apply_reduction<T: Float>(
    loss: Tensor<T>,
    reduction: crate::module::Reduction,
) -> FerrotorchResult<Tensor<T>> {
    match reduction {
        crate::module::Reduction::None => Ok(loss),
        crate::module::Reduction::Sum => red::sum(&loss),
        crate::module::Reduction::Mean => red::mean(&loss),
    }
}

/// L1 (mean absolute error) loss: `mean(|pred - target|)`.
///
/// `reduction` controls aggregation. With `Reduction::None` returns the
/// per-element absolute differences.
pub fn l1_loss<T: Float>(
    pred: &Tensor<T>,
    target: &Tensor<T>,
    reduction: crate::module::Reduction,
) -> FerrotorchResult<Tensor<T>> {
    let diff = arithmetic::sub(pred, target)?;
    let abs_diff = arithmetic::abs(&diff)?;
    apply_reduction(abs_diff, reduction)
}

/// Binary cross-entropy (probabilities). Expects `pred` ∈ (0, 1).
///
/// `bce(p, y) = -[y * log(p) + (1 - y) * log(1 - p)]`. For inputs that are
/// raw logits use [`binary_cross_entropy_with_logits`] which is numerically
/// more stable (it folds in a sigmoid).
pub fn binary_cross_entropy<T: Float>(
    pred: &Tensor<T>,
    target: &Tensor<T>,
    reduction: crate::module::Reduction,
) -> FerrotorchResult<Tensor<T>> {
    let one = ferrotorch_core::scalar(T::from(1.0).unwrap())?;
    let log_p = trans::log(pred)?;
    let one_minus_p = arithmetic::sub(&one, pred)?;
    let log_1mp = trans::log(&one_minus_p)?;
    let one_minus_y = arithmetic::sub(&one, target)?;
    let term1 = arithmetic::mul(target, &log_p)?;
    let term2 = arithmetic::mul(&one_minus_y, &log_1mp)?;
    let sum = arithmetic::add(&term1, &term2)?;
    let neg = arithmetic::neg(&sum)?;
    apply_reduction(neg, reduction)
}

/// Numerically-stable binary cross-entropy that takes raw logits.
///
/// `bce_with_logits(z, y) = -y * z + softplus(z) = -y * z + log(1 + exp(z))`.
/// This avoids the sigmoid → log composition's catastrophic cancellation
/// for large |z|.
pub fn binary_cross_entropy_with_logits<T: Float>(
    logits: &Tensor<T>,
    target: &Tensor<T>,
    reduction: crate::module::Reduction,
) -> FerrotorchResult<Tensor<T>> {
    let neg_y = arithmetic::neg(target)?;
    let term1 = arithmetic::mul(&neg_y, logits)?;
    let sp = act::softplus(logits, 1.0, 20.0)?;
    let total = arithmetic::add(&term1, &sp)?;
    apply_reduction(total, reduction)
}

/// KL divergence: `kl(target || exp(pred)) = sum(target * (log(target) - pred))`.
///
/// `pred` is expected to be log-probabilities (matches torch's
/// `F.kl_div(input, target, ...)` where `input` is in log-space). `target`
/// is in probability space.
pub fn kl_div<T: Float>(
    pred: &Tensor<T>,
    target: &Tensor<T>,
    reduction: crate::module::Reduction,
) -> FerrotorchResult<Tensor<T>> {
    // target * (log(target) - pred), with the convention 0 * log(0) := 0.
    // Add a tiny epsilon to keep log finite at zero, then mask: since
    // target=0 would multiply the (log eps - pred) term against 0 anyway,
    // the eps doesn't bias the result.
    let eps = ferrotorch_core::scalar(T::from(1e-12_f64).unwrap())?;
    let target_eps = arithmetic::add(target, &eps)?;
    let log_target = trans::log(&target_eps)?;
    let diff = arithmetic::sub(&log_target, pred)?;
    let elemwise = arithmetic::mul(target, &diff)?;
    apply_reduction(elemwise, reduction)
}

// ===========================================================================
// Distance / normalization (#584)
// ===========================================================================

/// L_p normalization along `dim`: `x / max(||x||_p, eps)`.
///
/// `p` is the order of the norm (typically 2). The denominator is clamped
/// from below by `eps` for numerical stability.
pub fn normalize<T: Float>(
    input: &Tensor<T>,
    p: f64,
    dim: i64,
    eps: f64,
) -> FerrotorchResult<Tensor<T>> {
    // |x|^p
    let abs_x = arithmetic::abs(input)?;
    let abs_p = arithmetic::pow(&abs_x, p)?;
    // Sum along dim, keep dim for broadcasting.
    let summed = red::sum_dim(&abs_p, dim, /* keepdim */ true)?;
    // Norm = sum^(1/p)
    let norm = arithmetic::pow(&summed, 1.0 / p)?;
    // Clamp from below by eps.
    let eps_t = T::from(eps).unwrap();
    let clamped = trans::clamp(&norm, eps_t, T::from(f64::INFINITY).unwrap())?;
    arithmetic::div(input, &clamped)
}

/// Cosine similarity along `dim`: `<x, y> / (||x||_2 ||y||_2)`, with
/// denominator clamped from below by `eps` for stability.
pub fn cosine_similarity<T: Float>(
    x: &Tensor<T>,
    y: &Tensor<T>,
    dim: i64,
    eps: f64,
) -> FerrotorchResult<Tensor<T>> {
    let xy = arithmetic::mul(x, y)?;
    let dot = red::sum_dim(&xy, dim, /* keepdim */ false)?;

    let xx = arithmetic::mul(x, x)?;
    let nx_sq = red::sum_dim(&xx, dim, false)?;
    let nx = arithmetic::sqrt(&nx_sq)?;

    let yy = arithmetic::mul(y, y)?;
    let ny_sq = red::sum_dim(&yy, dim, false)?;
    let ny = arithmetic::sqrt(&ny_sq)?;

    let prod = arithmetic::mul(&nx, &ny)?;
    let eps_t = T::from(eps).unwrap();
    let denom = trans::clamp(&prod, eps_t, T::from(f64::INFINITY).unwrap())?;
    arithmetic::div(&dot, &denom)
}

/// Pairwise distance: `||x - y||_p` along the last dim, with optional
/// `eps` smoothing inside the L_p calculation. Defaults `keepdim = false`.
pub fn pairwise_distance<T: Float>(
    x: &Tensor<T>,
    y: &Tensor<T>,
    p: f64,
    eps: f64,
) -> FerrotorchResult<Tensor<T>> {
    let diff = arithmetic::sub(x, y)?;
    let abs_diff = arithmetic::abs(&diff)?;
    // Add eps before raising to p to keep gradients smooth at zero.
    let eps_t = ferrotorch_core::scalar(T::from(eps).unwrap())?;
    let shifted = arithmetic::add(&abs_diff, &eps_t)?;
    let pwr = arithmetic::pow(&shifted, p)?;
    let summed = red::sum_dim(&pwr, /* dim */ -1, /* keepdim */ false)?;
    arithmetic::pow(&summed, 1.0 / p)
}

// ===========================================================================
// Misc utility (#584)
// ===========================================================================

/// One-hot encoding: convert integer-valued indices in `input` to a tensor
/// of shape `[*input.shape, num_classes]` with `1.0` at each index.
///
/// `input` is treated as a tensor of `T` whose values round to integers
/// in `[0, num_classes)`. Out-of-range indices return an error. This
/// operation is **not differentiable** — the output has no grad_fn.
pub fn one_hot<T: Float>(input: &Tensor<T>, num_classes: usize) -> FerrotorchResult<Tensor<T>> {
    if num_classes == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "one_hot: num_classes must be > 0".into(),
        });
    }
    let in_data = input.data_vec()?;
    let in_shape = input.shape().to_vec();

    let mut out_shape = in_shape.clone();
    out_shape.push(num_classes);
    let total: usize = in_data.len() * num_classes;
    let mut out = vec![T::from(0.0).unwrap(); total];

    let one = T::from(1.0).unwrap();
    for (i, val) in in_data.iter().enumerate() {
        // Round-to-nearest, then bounds-check.
        let f = val.to_f64().unwrap_or(-1.0);
        if !f.is_finite() || f < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "one_hot: index at flat position {i} is {f}, must be in [0, {num_classes})"
                ),
            });
        }
        let idx = f.round() as usize;
        if idx >= num_classes {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("one_hot: index {idx} out of range (num_classes = {num_classes})"),
            });
        }
        out[i * num_classes + idx] = one;
    }
    Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
}

// ===========================================================================
// Conv / ConvTranspose forwarders (#607)
// ===========================================================================
//
// These are thin wrappers over the existing `Conv*::from_parts(..)` constructor
// + `Module::forward(..)` path. They take user-supplied weight/bias tensors,
// build a transient module, and run the standard im2col + matmul forward (with
// the existing GPU fast path and CPU backward graph). Since the weight/bias
// tensors are cloned into the transient module, autograd attribution flows
// back to the caller's leaf tensors unchanged.

use crate::conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d};
use crate::module::Module;

/// Functional 1-D convolution. `weight` is `[out, in, k]`, `bias` is
/// optional `[out]`. Mirrors `torch.nn.functional.conv1d`.
pub fn conv1d<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: usize,
) -> FerrotorchResult<Tensor<T>> {
    let layer = Conv1d::from_parts(weight.clone(), bias.cloned(), stride, padding)?;
    layer.forward(input)
}

/// Functional 2-D convolution. `weight` is `[out, in, kH, kW]`, `bias` is
/// optional `[out]`. Mirrors `torch.nn.functional.conv2d`.
pub fn conv2d<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: (usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    let layer = Conv2d::from_parts(weight.clone(), bias.cloned(), stride, padding)?;
    layer.forward(input)
}

/// Functional 3-D convolution. `weight` is `[out, in, kD, kH, kW]`, `bias` is
/// optional `[out]`. Mirrors `torch.nn.functional.conv3d`.
pub fn conv3d<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    let layer = Conv3d::from_parts(weight.clone(), bias.cloned(), stride, padding)?;
    layer.forward(input)
}

/// Functional 1-D transposed convolution. `weight` is `[in, out, k]`.
/// Mirrors `torch.nn.functional.conv_transpose1d`.
pub fn conv_transpose1d<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: usize,
    padding: usize,
    output_padding: usize,
) -> FerrotorchResult<Tensor<T>> {
    let layer = ConvTranspose1d::from_parts(
        weight.clone(),
        bias.cloned(),
        stride,
        padding,
        output_padding,
    )?;
    layer.forward(input)
}

/// Functional 2-D transposed convolution. `weight` is `[in, out, kH, kW]`.
/// Mirrors `torch.nn.functional.conv_transpose2d`.
pub fn conv_transpose2d<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    let layer = ConvTranspose2d::from_parts(
        weight.clone(),
        bias.cloned(),
        stride,
        padding,
        output_padding,
    )?;
    layer.forward(input)
}

/// Functional 3-D transposed convolution. `weight` is
/// `[in, out, kD, kH, kW]`. Mirrors `torch.nn.functional.conv_transpose3d`.
pub fn conv_transpose3d<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    let layer = ConvTranspose3d::from_parts(
        weight.clone(),
        bias.cloned(),
        stride,
        padding,
        output_padding,
    )?;
    layer.forward(input)
}

// ===========================================================================
// Pooling re-exports (#607)
// ===========================================================================
//
// PyTorch exposes pool fns under `torch.nn.functional`; we keep the existing
// implementations in `pooling.rs` and re-export them here so callers can write
// `nn::functional::max_pool2d(..)` instead of `nn::pooling::max_pool2d(..)`.

pub use crate::pooling::{
    adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool1d,
    adaptive_max_pool2d, adaptive_max_pool3d, avg_pool1d, avg_pool2d, avg_pool3d, lp_pool1d,
    lp_pool2d, max_pool1d, max_pool2d, max_pool3d,
};

// ===========================================================================
// Padding re-exports (#607)
// ===========================================================================

pub use crate::padding::{
    PaddingMode, functional_pad_1d as pad1d, functional_pad_2d as pad2d, functional_pad_3d as pad3d,
};

// ===========================================================================
// Embedding (#607)
// ===========================================================================

/// Functional embedding lookup. `weight` is the lookup table
/// `[num_embeddings, embedding_dim]`; `input` is a 1-D tensor of indices
/// stored as floats (cast to `usize` internally). Mirrors
/// `torch.nn.functional.embedding`.
pub fn embedding<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    padding_idx: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let layer = crate::embedding::Embedding::from_pretrained(weight.clone(), padding_idx)?;
    layer.forward(input)
}

// ===========================================================================
// Scaled-dot-product attention (#607)
// ===========================================================================

/// Stateless scaled-dot-product attention. Mirrors
/// `torch.nn.functional.scaled_dot_product_attention`.
///
/// Inputs:
///   - `query`: `[B, N_q, d]`
///   - `key`:   `[B, N_k, d]`
///   - `value`: `[B, N_k, d_v]`
///   - `is_causal`: apply causal mask (requires `N_q == N_k`).
///
/// Returns `[B, N_q, d_v]`. Block size defaults to 64 to match the standard
/// `flash_attention` tile width; tweak via [`crate::flash_attention`] directly
/// when more control is needed.
pub fn scaled_dot_product_attention<T: Float>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    is_causal: bool,
) -> FerrotorchResult<Tensor<T>> {
    crate::flash_attention::flash_attention(query, key, value, is_causal, 64)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    /// Create a leaf tensor with given data and shape, optionally with grad.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    /// Assert two float slices are element-wise close.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len(),
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: actual={a} expected={e} diff={}",
                (a - e).abs(),
            );
        }
    }

    // -----------------------------------------------------------------------
    // functional::linear
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_no_bias() {
        // weight = [[1, 0, 0], [0, 1, 0]] (2x3) — selects first two features
        let weight = leaf(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], &[2, 3], false);
        // input = [[1, 2, 3], [4, 5, 6]] (2x3)
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);

        let output = linear(&input, &weight, None).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
        assert_close(output.data().unwrap(), &[1.0, 2.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn test_linear_with_bias() {
        // weight = identity 2x2
        let weight = leaf(&[1.0, 0.0, 0.0, 1.0], &[2, 2], false);
        let bias = leaf(&[10.0, 20.0], &[2], false);
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);

        let output = linear(&input, &weight, Some(&bias)).unwrap();
        assert_eq!(output.shape(), &[2, 2]);
        assert_close(output.data().unwrap(), &[11.0, 22.0, 13.0, 24.0], 1e-6);
    }

    #[test]
    fn test_linear_matches_module() {
        // Build a Linear module and a functional call with the same weights.
        use crate::linear::Linear;
        use crate::module::Module;
        use crate::parameter::Parameter;

        let mut layer = Linear::<f32>::new(3, 2, true).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        *layer.bias.as_mut().unwrap() = Parameter::from_slice(&[0.1, 0.2], &[2]).unwrap();

        let input = leaf(&[1.0, 0.0, -1.0, 2.0, 1.0, 0.0], &[2, 3], false);

        let module_out = layer.forward(&input).unwrap();
        let func_out = linear(
            &input,
            layer.weight.tensor(),
            Some(layer.bias.as_ref().unwrap().tensor()),
        )
        .unwrap();

        assert_eq!(module_out.shape(), func_out.shape());
        assert_close(module_out.data().unwrap(), func_out.data().unwrap(), 1e-5);
    }

    #[test]
    fn test_linear_wrong_input_dims() {
        let weight = leaf(&[1.0; 6], &[2, 3], false);
        let input_1d = leaf(&[1.0, 2.0, 3.0], &[3], false);
        assert!(linear(&input_1d, &weight, None).is_err());
    }

    #[test]
    fn test_linear_wrong_weight_dims() {
        let weight = leaf(&[1.0; 6], &[6], false);
        let input = leaf(&[1.0; 6], &[2, 3], false);
        assert!(linear(&input, &weight, None).is_err());
    }

    #[test]
    fn test_linear_feature_mismatch() {
        let weight = leaf(&[1.0; 8], &[2, 4], false);
        let input = leaf(&[1.0; 6], &[2, 3], false);
        assert!(linear(&input, &weight, None).is_err());
    }

    #[test]
    fn test_linear_bias_shape_mismatch() {
        let weight = leaf(&[1.0; 6], &[2, 3], false);
        let bias = leaf(&[1.0; 3], &[3], false); // should be [2]
        let input = leaf(&[1.0; 6], &[2, 3], false);
        assert!(linear(&input, &weight, Some(&bias)).is_err());
    }

    // -----------------------------------------------------------------------
    // functional::relu matches core relu
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu_matches_core() {
        let input = leaf(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], false);

        let func_out = relu(&input).unwrap();
        let core_out = act::relu(&input).unwrap();

        assert_close(func_out.data().unwrap(), core_out.data().unwrap(), 1e-7);
    }

    #[test]
    fn test_relu_values() {
        let input = leaf(&[-3.0, -1.0, 0.0, 0.5, 2.0], &[5], false);
        let output = relu(&input).unwrap();
        assert_close(output.data().unwrap(), &[0.0, 0.0, 0.0, 0.5, 2.0], 1e-7);
    }

    // -----------------------------------------------------------------------
    // Other activations
    // -----------------------------------------------------------------------

    #[test]
    fn test_sigmoid_values() {
        let input = leaf(&[0.0], &[1], false);
        let output = sigmoid(&input).unwrap();
        assert!((output.data().unwrap()[0] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_tanh_values() {
        let input = leaf(&[0.0], &[1], false);
        let output = tanh(&input).unwrap();
        assert!(output.data().unwrap()[0].abs() < 1e-6);
    }

    #[test]
    fn test_gelu_positive() {
        let input = leaf(&[1.0, 2.0], &[2], false);
        let output = gelu(&input).unwrap();
        let d = output.data().unwrap();
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);
    }

    #[test]
    fn test_silu_zero() {
        let input = leaf(&[0.0], &[1], false);
        let output = silu(&input).unwrap();
        assert!(output.data().unwrap()[0].abs() < 1e-6);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let output = softmax(&input).unwrap();
        let d = output.data().unwrap();
        let total: f32 = d.iter().sum();
        assert!((total - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_log_softmax_negative() {
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let output = log_softmax(&input).unwrap();
        let d = output.data().unwrap();
        assert!(d.iter().all(|&v| v <= 0.0));
    }

    // -----------------------------------------------------------------------
    // LeakyReLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_leaky_relu_values() {
        let input = leaf(&[-2.0, -1.0, 0.0, 1.0, 2.0], &[5], false);
        let output = leaky_relu(&input, 0.01).unwrap();
        let d = output.data().unwrap();
        assert!((d[0] - (-0.02)).abs() < 1e-5);
        assert!((d[1] - (-0.01)).abs() < 1e-5);
        assert!((d[2] - 0.0).abs() < 1e-5);
        assert!((d[3] - 1.0).abs() < 1e-5);
        assert!((d[4] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_leaky_relu_zero_slope_is_relu() {
        let input = leaf(&[-2.0, 0.0, 3.0], &[3], false);
        let lrelu_out = leaky_relu(&input, 0.0).unwrap();
        let relu_out = relu(&input).unwrap();
        assert_close(lrelu_out.data().unwrap(), relu_out.data().unwrap(), 1e-7);
    }

    #[test]
    fn test_leaky_relu_one_slope_is_identity() {
        let input = leaf(&[-2.0, 0.0, 3.0], &[3], false);
        let output = leaky_relu(&input, 1.0).unwrap();
        assert_close(output.data().unwrap(), &[-2.0, 0.0, 3.0], 1e-7);
    }

    // -----------------------------------------------------------------------
    // Reduction
    // -----------------------------------------------------------------------

    #[test]
    fn test_sum_values() {
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let output = sum(&input).unwrap();
        assert!((output.item().unwrap() - 10.0).abs() < 1e-6);
    }

    #[test]
    fn test_mean_values() {
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let output = mean(&input).unwrap();
        assert!((output.item().unwrap() - 2.5).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Dropout
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout_eval_is_identity() {
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], false);
        let output = dropout(&input, 0.5, false).unwrap();
        // In eval mode (training=false), the output should be the same tensor.
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout_zero_prob_is_identity() {
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let output = dropout(&input, 0.0, true).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout_invalid_p() {
        let input = leaf(&[1.0], &[1], false);
        assert!(dropout(&input, 1.0, true).is_err());
        assert!(dropout(&input, -0.1, true).is_err());
        assert!(dropout(&input, 1.5, true).is_err());
    }

    #[test]
    fn test_dropout_rate_approximately_correct() {
        let input = ferrotorch_core::ones::<f32>(&[100_000]).unwrap();
        let output = dropout(&input, 0.5, true).unwrap();
        let data = output.data().unwrap();

        let zeros = data.iter().filter(|&&x| x == 0.0).count();
        let rate = zeros as f64 / data.len() as f64;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "dropout rate = {rate}, expected ~0.5"
        );

        // Surviving elements should be scaled by 1/(1-0.5) = 2.0.
        let non_zero: Vec<f32> = data.iter().copied().filter(|&x| x != 0.0).collect();
        assert!(!non_zero.is_empty());
        for &v in &non_zero {
            assert!(
                (v - 2.0).abs() < 1e-6,
                "surviving element = {v}, expected 2.0"
            );
        }
    }

    #[test]
    fn test_dropout_training_flag() {
        // With training=false, even high p should not drop anything.
        let input = ferrotorch_core::ones::<f32>(&[1000]).unwrap();
        let output = dropout(&input, 0.99, false).unwrap();
        assert!(output.is_same(&input));
    }

    // -----------------------------------------------------------------------
    // MSE loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_mse_loss_zero() {
        let pred = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let target = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let loss = mse_loss(&pred, &target).unwrap();
        assert!(loss.item().unwrap().abs() < 1e-7);
    }

    #[test]
    fn test_mse_loss_known_value() {
        // pred = [1, 2], target = [3, 4]
        // diff = [-2, -2], sq = [4, 4], mean = 4
        let pred = leaf(&[1.0, 2.0], &[2], false);
        let target = leaf(&[3.0, 4.0], &[2], false);
        let loss = mse_loss(&pred, &target).unwrap();
        assert!((loss.item().unwrap() - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_mse_loss_shape_mismatch() {
        let pred = leaf(&[1.0, 2.0], &[2], false);
        let target = leaf(&[1.0, 2.0, 3.0], &[3], false);
        assert!(mse_loss(&pred, &target).is_err());
    }

    // -----------------------------------------------------------------------
    // Cross-entropy loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_entropy_basic() {
        // logits = [[10, 0], [0, 10]] (2 samples, 2 classes)
        // targets = [0, 1] (correct classes)
        // With very confident predictions, loss should be close to 0.
        let logits = leaf(&[10.0, 0.0, 0.0, 10.0], &[2, 2], false);
        let targets = leaf(&[0.0, 1.0], &[2], false);
        let loss = cross_entropy(&logits, &targets).unwrap();
        assert!(loss.item().unwrap() < 0.01);
    }

    #[test]
    fn test_cross_entropy_wrong_logits_shape() {
        let logits = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let targets = leaf(&[0.0], &[1], false);
        assert!(cross_entropy(&logits, &targets).is_err());
    }

    #[test]
    fn test_cross_entropy_target_batch_mismatch() {
        let logits = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let targets = leaf(&[0.0, 1.0, 0.0], &[3], false);
        assert!(cross_entropy(&logits, &targets).is_err());
    }

    #[test]
    fn test_cross_entropy_uniform_logits() {
        // Uniform logits: softmax = [0.5, 0.5], log_softmax = [-ln(2), -ln(2)]
        // NLL for class 0 = ln(2), NLL for class 1 = ln(2)
        // Mean loss = ln(2)
        let logits = leaf(&[0.0, 0.0, 0.0, 0.0], &[2, 2], false);
        let targets = leaf(&[0.0, 1.0], &[2], false);
        let loss = cross_entropy(&logits, &targets).unwrap();
        let expected = (2.0_f32).ln();
        assert!(
            (loss.item().unwrap() - expected).abs() < 1e-5,
            "loss = {}, expected ln(2) = {}",
            loss.item().unwrap(),
            expected,
        );
    }

    // -----------------------------------------------------------------------
    // Activation parity expansion (#584)
    // -----------------------------------------------------------------------

    fn close(a: f32, b: f32, tol: f32) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn test_hardtanh_default_clamps_to_minus_one_one() {
        let x = leaf(&[-3.0, -1.0, 0.5, 1.5], &[4], false);
        let out = hardtanh(&x).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d, &[-1.0, -1.0, 0.5, 1.0]);
    }

    #[test]
    fn test_hardtanh_with_custom_bounds() {
        let x = leaf(&[-3.0, -0.5, 1.0, 5.0], &[4], false);
        let out = hardtanh_with(&x, -2.0, 3.0).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d, &[-2.0, -0.5, 1.0, 3.0]);
    }

    #[test]
    fn test_relu6_clamps_top_at_6() {
        let x = leaf(&[-1.0, 0.0, 3.0, 7.0], &[4], false);
        let out = relu6(&x).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d, &[0.0, 0.0, 3.0, 6.0]);
    }

    #[test]
    fn test_hardsigmoid_endpoints() {
        // x = -3 → 0, x = 3 → 1, x = 0 → 0.5
        let x = leaf(&[-5.0, -3.0, 0.0, 3.0, 5.0], &[5], false);
        let out = hardsigmoid(&x).unwrap();
        let d = out.data().unwrap();
        assert!(close(d[0], 0.0, 1e-6));
        assert!(close(d[1], 0.0, 1e-6));
        assert!(close(d[2], 0.5, 1e-6));
        assert!(close(d[3], 1.0, 1e-6));
        assert!(close(d[4], 1.0, 1e-6));
    }

    #[test]
    fn test_hardswish_zero_at_minus_three_and_below() {
        let x = leaf(&[-5.0, -3.0, 0.0, 1.0, 5.0], &[5], false);
        let out = hardswish(&x).unwrap();
        let d = out.data().unwrap();
        // x = -5 → -5 * 0 = 0; x = -3 → -3 * 0 = 0
        assert!(close(d[0], 0.0, 1e-6));
        assert!(close(d[1], 0.0, 1e-6));
        // x = 0 → 0 * 0.5 = 0
        assert!(close(d[2], 0.0, 1e-6));
        // x = 5 → 5 * 1 = 5 (saturated)
        assert!(close(d[4], 5.0, 1e-6));
    }

    #[test]
    fn test_log_sigmoid_matches_log_of_sigmoid() {
        let x = leaf(&[-2.0, -0.5, 0.0, 0.5, 2.0], &[5], false);
        let out = log_sigmoid(&x).unwrap();
        let d = out.data().unwrap();
        for (i, &xi) in [-2.0, -0.5, 0.0, 0.5, 2.0].iter().enumerate() {
            let ref_val = (1.0_f32 / (1.0 + (-xi as f32).exp())).ln();
            assert!(
                close(d[i], ref_val, 1e-5),
                "log_sigmoid({xi}) = {} vs {ref_val}",
                d[i]
            );
        }
    }

    #[test]
    fn test_softmin_inverts_softmax() {
        // softmin(x) = softmax(-x)
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let out = softmin(&x).unwrap();
        let neg_x = leaf(&[-1.0, -2.0, -3.0], &[3], false);
        let ref_out = softmax(&neg_x).unwrap();
        let d = out.data().unwrap();
        let r = ref_out.data().unwrap();
        for i in 0..3 {
            assert!(close(d[i], r[i], 1e-6));
        }
    }

    #[test]
    fn test_softsign_bounded() {
        // softsign(0) = 0; softsign(x) → ±1 as x → ±∞.
        let x = leaf(&[-1000.0, -1.0, 0.0, 1.0, 1000.0], &[5], false);
        let out = softsign(&x).unwrap();
        let d = out.data().unwrap();
        assert!(close(d[0], -1.0, 1e-2));
        assert!(close(d[1], -0.5, 1e-6));
        assert!(close(d[2], 0.0, 1e-6));
        assert!(close(d[3], 0.5, 1e-6));
        assert!(close(d[4], 1.0, 1e-2));
    }

    #[test]
    fn test_tanhshrink() {
        let x = leaf(&[0.0, 1.0, 2.0], &[3], false);
        let out = tanhshrink(&x).unwrap();
        let d = out.data().unwrap();
        assert!(close(d[0], 0.0, 1e-6));
        assert!(close(d[1], 1.0 - 1.0_f32.tanh(), 1e-6));
        assert!(close(d[2], 2.0 - 2.0_f32.tanh(), 1e-6));
    }

    #[test]
    fn test_selu_scale_constants() {
        // selu(0) = 0, selu(1) = scale * 1 ≈ 1.0507
        let x = leaf(&[0.0, 1.0], &[2], false);
        let out = selu(&x).unwrap();
        let d = out.data().unwrap();
        assert!(close(d[0], 0.0, 1e-6));
        assert!(close(d[1], 1.050_700_9, 1e-5));
    }

    #[test]
    fn test_softplus_default_matches_explicit() {
        let x = leaf(&[-1.0, 0.0, 1.0, 2.0], &[4], false);
        let out_default = softplus(&x).unwrap();
        let out_explicit = softplus_with(&x, 1.0, 20.0).unwrap();
        let a = out_default.data().unwrap();
        let b = out_explicit.data().unwrap();
        for i in 0..4 {
            assert!(close(a[i], b[i], 1e-6));
        }
    }

    #[test]
    fn test_elu_zero_at_origin() {
        let x = leaf(&[-1.0, 0.0, 1.0], &[3], false);
        let out = elu(&x).unwrap();
        let d = out.data().unwrap();
        assert!(close(d[0], (-1.0_f32).exp() - 1.0, 1e-5));
        assert!(close(d[1], 0.0, 1e-6));
        assert!(close(d[2], 1.0, 1e-6));
    }

    #[test]
    fn test_glu_halves_input() {
        // [[1, 2, 3, 4]] split along last dim → a=[[1, 2]], b=[[3, 4]].
        // out = a * sigmoid(b).
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 4], false);
        let out = glu(&x, -1).unwrap();
        assert_eq!(out.shape(), &[1, 2]);
        let d = out.data().unwrap();
        let s3 = 1.0 / (1.0 + (-3.0_f32).exp());
        let s4 = 1.0 / (1.0 + (-4.0_f32).exp());
        assert!(close(d[0], 1.0 * s3, 1e-5));
        assert!(close(d[1], 2.0 * s4, 1e-5));
    }

    #[test]
    fn test_glu_rejects_odd_dim() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let err = glu(&x, 0).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    // -----------------------------------------------------------------------
    // Loss parity expansion (#584)
    // -----------------------------------------------------------------------

    use crate::module::Reduction;

    #[test]
    fn test_l1_loss_mean() {
        // pred = [1, 2, 3], target = [0, 0, 0] → |diff| = [1, 2, 3], mean = 2
        let p = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let t = leaf(&[0.0, 0.0, 0.0], &[3], false);
        let loss = l1_loss(&p, &t, Reduction::Mean).unwrap();
        assert!(close(loss.item().unwrap(), 2.0, 1e-5));
    }

    #[test]
    fn test_l1_loss_sum() {
        let p = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let t = leaf(&[0.0, 0.0, 0.0], &[3], false);
        let loss = l1_loss(&p, &t, Reduction::Sum).unwrap();
        assert!(close(loss.item().unwrap(), 6.0, 1e-5));
    }

    #[test]
    fn test_l1_loss_none_returns_per_element() {
        let p = leaf(&[1.0, -2.0, 3.0], &[3], false);
        let t = leaf(&[0.0, 0.0, 0.0], &[3], false);
        let loss = l1_loss(&p, &t, Reduction::None).unwrap();
        assert_eq!(loss.shape(), &[3]);
        let d = loss.data().unwrap();
        assert_eq!(d, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_binary_cross_entropy_log2_loss_for_uniform() {
        // pred = 0.5, target = 1 → -log(0.5) = ln(2)
        let p = leaf(&[0.5], &[1], false);
        let t = leaf(&[1.0], &[1], false);
        let loss = binary_cross_entropy(&p, &t, Reduction::Mean).unwrap();
        assert!(close(loss.item().unwrap(), 2.0_f32.ln(), 1e-5));
    }

    #[test]
    fn test_binary_cross_entropy_with_logits_matches_bce_for_sigmoid_input() {
        // bce_with_logits(z, y) should equal bce(sigmoid(z), y) for finite z.
        let z = leaf(&[-1.0, 0.5, 2.0], &[3], false);
        let y = leaf(&[0.0, 1.0, 1.0], &[3], false);
        let s = sigmoid(&z).unwrap();
        let lhs = binary_cross_entropy_with_logits(&z, &y, Reduction::Mean)
            .unwrap()
            .item()
            .unwrap();
        let rhs = binary_cross_entropy(&s, &y, Reduction::Mean)
            .unwrap()
            .item()
            .unwrap();
        assert!(close(lhs, rhs, 1e-4), "logit-bce={lhs} vs bce(sig)={rhs}");
    }

    #[test]
    fn test_kl_div_zero_for_matched_distributions() {
        // target = [0.5, 0.5], pred (in log-space) = log([0.5, 0.5]).
        // KL = sum(target * (log(target) - pred)) = 0.
        let target = leaf(&[0.5, 0.5], &[2], false);
        let pred = leaf(&[2.0_f32.recip().ln(), 2.0_f32.recip().ln()], &[2], false);
        let loss = kl_div(&pred, &target, Reduction::Sum).unwrap();
        assert!(close(loss.item().unwrap(), 0.0, 1e-5));
    }

    // -----------------------------------------------------------------------
    // Distance / normalization (#584)
    // -----------------------------------------------------------------------

    #[test]
    fn test_normalize_l2_unit_norm() {
        // [3, 4] / sqrt(9 + 16) = [3/5, 4/5]
        let x = leaf(&[3.0, 4.0], &[2], false);
        let out = normalize(&x, 2.0, 0, 1e-12).unwrap();
        let d = out.data().unwrap();
        assert!(close(d[0], 0.6, 1e-5));
        assert!(close(d[1], 0.8, 1e-5));
    }

    #[test]
    fn test_normalize_l2_2d_per_row() {
        // [[3, 4], [6, 8]] normalized along dim=1 → [[0.6, 0.8], [0.6, 0.8]]
        let x = leaf(&[3.0, 4.0, 6.0, 8.0], &[2, 2], false);
        let out = normalize(&x, 2.0, 1, 1e-12).unwrap();
        let d = out.data().unwrap();
        assert!(close(d[0], 0.6, 1e-5));
        assert!(close(d[1], 0.8, 1e-5));
        assert!(close(d[2], 0.6, 1e-5));
        assert!(close(d[3], 0.8, 1e-5));
    }

    #[test]
    fn test_cosine_similarity_aligned_vectors() {
        // x and 2*x should have cosine similarity 1.
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let y = leaf(&[2.0, 4.0, 6.0], &[3], false);
        let sim = cosine_similarity(&x, &y, 0, 1e-8).unwrap();
        assert!(close(sim.item().unwrap(), 1.0, 1e-4));
    }

    #[test]
    fn test_cosine_similarity_orthogonal_vectors() {
        let x = leaf(&[1.0, 0.0], &[2], false);
        let y = leaf(&[0.0, 1.0], &[2], false);
        let sim = cosine_similarity(&x, &y, 0, 1e-8).unwrap();
        assert!(close(sim.item().unwrap(), 0.0, 1e-5));
    }

    #[test]
    fn test_pairwise_distance_zero_for_equal_vectors() {
        let x = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
        let y = leaf(&[1.0, 2.0, 3.0], &[1, 3], false);
        let d = pairwise_distance(&x, &y, 2.0, 0.0).unwrap();
        // Distance should be 0 (no eps floor).
        assert!(close(d.item().unwrap(), 0.0, 1e-5));
    }

    #[test]
    fn test_pairwise_distance_l2_simple() {
        // ||[3, 4] - [0, 0]||_2 = 5 (rows of length 2 give per-row distance)
        let x = leaf(&[3.0, 4.0], &[1, 2], false);
        let y = leaf(&[0.0, 0.0], &[1, 2], false);
        let d = pairwise_distance(&x, &y, 2.0, 0.0).unwrap();
        assert!(close(d.item().unwrap(), 5.0, 1e-4));
    }

    // -----------------------------------------------------------------------
    // Misc utility (#584)
    // -----------------------------------------------------------------------

    #[test]
    fn test_one_hot_basic() {
        // indices = [0, 2, 1] with num_classes = 3.
        let idx = leaf(&[0.0, 2.0, 1.0], &[3], false);
        let oh = one_hot(&idx, 3).unwrap();
        assert_eq!(oh.shape(), &[3, 3]);
        let d = oh.data().unwrap();
        assert_eq!(
            d,
            &[
                1.0, 0.0, 0.0, // 0 → [1, 0, 0]
                0.0, 0.0, 1.0, // 2 → [0, 0, 1]
                0.0, 1.0, 0.0, // 1 → [0, 1, 0]
            ]
        );
    }

    #[test]
    fn test_one_hot_2d_input() {
        // [[0, 1], [2, 0]] with num_classes=3 → [[ [1,0,0], [0,1,0] ], [ [0,0,1], [1,0,0] ]]
        let idx = leaf(&[0.0, 1.0, 2.0, 0.0], &[2, 2], false);
        let oh = one_hot(&idx, 3).unwrap();
        assert_eq!(oh.shape(), &[2, 2, 3]);
    }

    #[test]
    fn test_one_hot_rejects_out_of_range() {
        let idx = leaf(&[0.0, 5.0], &[2], false);
        let err = one_hot(&idx, 3).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn test_one_hot_rejects_negative() {
        let idx = leaf(&[-1.0, 0.0], &[2], false);
        let err = one_hot(&idx, 3).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    // -------------------------------------------------------------------
    // #607 forwarders
    // -------------------------------------------------------------------

    #[test]
    fn conv2d_forwarder_matches_module() {
        // Compare nn::functional::conv2d output against an equivalent
        // Conv2d module driven via the same weight/bias.
        let input = leaf(&[1.0; 16], &[1, 1, 4, 4], false);
        let weight = leaf(&[1.0; 9], &[1, 1, 3, 3], false);
        let bias = leaf(&[0.0], &[1], false);

        let f_out = super::conv2d(&input, &weight, Some(&bias), (1, 1), (0, 0)).unwrap();
        // Output spatial size: (4 - 3 + 0) / 1 + 1 = 2
        assert_eq!(f_out.shape(), &[1, 1, 2, 2]);
        // All ones in a 3x3 window summed = 9.0 per output cell.
        assert_close(f_out.data().unwrap(), &[9.0, 9.0, 9.0, 9.0], 1e-5);
    }

    #[test]
    fn conv2d_rejects_bad_weight_shape() {
        let input = leaf(&[0.0; 16], &[1, 1, 4, 4], false);
        let weight = leaf(&[1.0; 4], &[2, 2], false);
        let err = super::conv2d(&input, &weight, None, (1, 1), (0, 0)).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn conv1d_forwarder_smoke() {
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4], false);
        let weight = leaf(&[1.0, 1.0], &[1, 1, 2], false);
        let out = super::conv1d(&input, &weight, None, 1, 0).unwrap();
        // Sum-of-pairs: (1+2)=3, (2+3)=5, (3+4)=7
        assert_eq!(out.shape(), &[1, 1, 3]);
        assert_close(out.data().unwrap(), &[3.0, 5.0, 7.0], 1e-5);
    }

    #[test]
    fn embedding_forwarder_matches_module() {
        let weight = leaf(&[0.0, 0.1, 1.0, 1.1, 2.0, 2.1], &[3, 2], false);
        let idx = leaf(&[0.0, 2.0, 1.0], &[3], false);
        let out = super::embedding(&idx, &weight, None).unwrap();
        assert_eq!(out.shape(), &[3, 2]);
        assert_close(out.data().unwrap(), &[0.0, 0.1, 2.0, 2.1, 1.0, 1.1], 1e-5);
    }

    #[test]
    fn pad_re_export_2d() {
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2], false);
        let out = super::pad2d(&input, 1, 1, 1, 1, super::PaddingMode::Zeros, 0.0).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn sdpa_forwarder_smoke() {
        // Single token attention: q=k=v of shape [1,1,2] -> output [1,1,2].
        let q = leaf(&[1.0, 0.0], &[1, 1, 2], false);
        let k = leaf(&[1.0, 0.0], &[1, 1, 2], false);
        let v = leaf(&[3.0, 4.0], &[1, 1, 2], false);
        let out = super::scaled_dot_product_attention(&q, &k, &v, false).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2]);
        // softmax over single key returns 1.0; output should equal v.
        assert_close(out.data().unwrap(), &[3.0, 4.0], 1e-5);
    }
}
