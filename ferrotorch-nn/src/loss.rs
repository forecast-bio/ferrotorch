//! Loss functions for training neural networks.
//!
//! Unlike layers, loss functions are **not** `Module<T>`. They are callable
//! structs with a `forward(&self, pred, target) -> FerrotorchResult<Tensor<T>>`
//! method. Each loss attaches a backward node to the returned tensor when
//! gradient tracking is enabled.

use std::sync::Arc;

use ferrotorch_core::Float;
use ferrotorch_core::autograd::autocast_ops::autocast_guard;
use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::ops::elementwise::{binary_map, mean, sum, unary_map};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};
use num_traits::{One, Zero};

use crate::module::Reduction;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Apply the chosen reduction to an unreduced loss tensor.
fn apply_reduction<T: Float>(
    unreduced: &Tensor<T>,
    reduction: Reduction,
) -> FerrotorchResult<Tensor<T>> {
    match reduction {
        Reduction::None => Ok(unreduced.clone()),
        Reduction::Mean => mean(unreduced),
        Reduction::Sum => sum(unreduced),
    }
}

// ===========================================================================
// MSELoss
// ===========================================================================

/// Mean Squared Error loss.
///
/// `loss_i = (pred_i - target_i)^2`, then the chosen reduction is applied.
#[derive(Debug, Clone)]
pub struct MSELoss {
    pub reduction: Reduction,
}

impl MSELoss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute MSE loss.
    ///
    /// Participates in autocast: classified as `FullPrecision` (`"mse_loss"`).
    pub fn forward<T: Float>(
        &self,
        pred: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        autocast_guard("mse_loss");

        if pred.shape() != target.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MSELoss: pred shape {:?} != target shape {:?}",
                    pred.shape(),
                    target.shape()
                ),
            });
        }

        let diff = binary_map(pred, target, |p, t| p - t)?;
        let sq = unary_map(&diff, |x| x * x)?;
        let reduced = apply_reduction(&sq, self.reduction)?;

        if is_grad_enabled() && pred.requires_grad() {
            let grad_fn = Arc::new(MSEBackward {
                pred: pred.clone(),
                target: target.clone(),
                reduction: self.reduction,
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
}

impl Default for MSELoss {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

/// Backward for `MSELoss`.
///
/// `grad_pred = 2 * (pred - target) * grad_output / n` (mean reduction)
/// `grad_pred = 2 * (pred - target) * grad_output`     (sum reduction)
/// `grad_pred = 2 * (pred - target) * grad_output`     (no reduction, elementwise)
#[derive(Debug)]
struct MSEBackward<T: Float> {
    pred: Tensor<T>,
    target: Tensor<T>,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for MSEBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        use ferrotorch_core::autograd::no_grad::no_grad;
        use ferrotorch_core::grad_fns::arithmetic::{mul, sub};

        // grad = 2 * (pred - target) * grad_output [/ n for mean]
        let grad_input = no_grad(|| {
            let diff = sub(&self.pred, &self.target)?;
            let two =
                ferrotorch_core::creation::scalar(T::from(2.0).unwrap())?.to(self.pred.device())?;
            let scaled = mul(&diff, &two)?;
            let result = mul(&scaled, grad_output)?;
            match self.reduction {
                Reduction::Mean => {
                    let n = ferrotorch_core::creation::scalar(
                        T::from(self.pred.shape().iter().product::<usize>()).unwrap(),
                    )?
                    .to(self.pred.device())?;
                    ferrotorch_core::grad_fns::arithmetic::div(&result, &n)
                }
                _ => Ok(result),
            }
        })?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.pred]
    }

    fn name(&self) -> &'static str {
        "MSEBackward"
    }
}

// ===========================================================================
// CrossEntropyLoss
// ===========================================================================

/// Cross-entropy loss combining log-softmax and NLL.
///
/// Expects logits `[B, C]` and integer class targets `[B]` (stored as floats,
/// e.g. `0.0`, `1.0`, `2.0`).
///
/// With label smoothing `ls`:
/// ```text
/// loss = (1 - ls) * nll + ls * (-log_probs.mean(dim=-1))
/// ```
#[derive(Debug, Clone)]
pub struct CrossEntropyLoss {
    pub reduction: Reduction,
    pub label_smoothing: f64,
}

impl CrossEntropyLoss {
    pub fn new(reduction: Reduction, label_smoothing: f64) -> Self {
        Self {
            reduction,
            label_smoothing,
        }
    }

    /// Compute cross-entropy loss.
    ///
    /// Participates in autocast: classified as `FullPrecision` (`"cross_entropy"`).
    pub fn forward<T: Float>(
        &self,
        logits: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        autocast_guard("cross_entropy");

        let shape = logits.shape();
        if shape.len() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CrossEntropyLoss: expected 2D logits [B, C], got shape {:?}",
                    shape
                ),
            });
        }
        let batch = shape[0];
        let classes = shape[1];

        if targets.shape() != [batch] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "CrossEntropyLoss: target shape {:?} does not match batch size {}",
                    targets.shape(),
                    batch,
                ),
            });
        }

        let logits_data = logits.data_vec()?;
        let targets_data = targets.data_vec()?;
        let ls = T::from(self.label_smoothing).unwrap();
        let one = <T as One>::one();

        // Compute log_softmax along dim=-1 (the class dimension).
        let mut log_probs = vec![<T as Zero>::zero(); batch * classes];
        let mut softmax_out = vec![<T as Zero>::zero(); batch * classes];

        for b in 0..batch {
            let base = b * classes;
            // Numerical stability: subtract max.
            let mut max_val = logits_data[base];
            for c in 1..classes {
                if logits_data[base + c] > max_val {
                    max_val = logits_data[base + c];
                }
            }
            let mut sum_exp = <T as Zero>::zero();
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

        // Compute per-sample loss.
        let mut losses = vec![<T as Zero>::zero(); batch];
        for b in 0..batch {
            let base = b * classes;
            let target_class = targets_data[b].to_usize().unwrap_or(0);

            // NLL component: -log_probs[target_class]
            let nll = -log_probs[base + target_class];

            if self.label_smoothing > 0.0 {
                // Smooth component: -mean(log_probs along class dim)
                let mut sum_lp = <T as Zero>::zero();
                for c in 0..classes {
                    sum_lp += log_probs[base + c];
                }
                let smooth = -sum_lp / T::from(classes).unwrap();
                losses[b] = (one - ls) * nll + ls * smooth;
            } else {
                losses[b] = nll;
            }
        }

        let unreduced = Tensor::from_storage(TensorStorage::cpu(losses), vec![batch], false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && logits.requires_grad() {
            let softmax_tensor =
                Tensor::from_storage(TensorStorage::cpu(softmax_out), vec![batch, classes], false)?;
            let grad_fn = Arc::new(CrossEntropyBackward {
                logits: logits.clone(),
                targets: targets.clone(),
                softmax: softmax_tensor,
                label_smoothing: self.label_smoothing,
                reduction: self.reduction,
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
}

impl Default for CrossEntropyLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, 0.0)
    }
}

/// Backward for `CrossEntropyLoss`.
///
/// Gradient through log_softmax + NLL:
/// `grad_logits[b, c] = softmax[b, c] - one_hot[b, c]` (for mean: divided by B)
///
/// With label smoothing `ls`:
/// `grad_logits[b, c] = (1 - ls) * (softmax[b, c] - one_hot[b, c])
///                     + ls * (softmax[b, c] - 1/C)`
/// Simplifies to: `softmax[b, c] - ((1 - ls) * one_hot[b, c] + ls / C)`
#[derive(Debug)]
struct CrossEntropyBackward<T: Float> {
    logits: Tensor<T>,
    targets: Tensor<T>,
    softmax: Tensor<T>,
    label_smoothing: f64,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for CrossEntropyBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.logits.shape();
        let batch = shape[0];
        let classes = shape[1];
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CrossEntropy backward",
            });
        }
        let sm_data = self.softmax.data()?;
        let targets_data = self.targets.data()?;
        let grad_data = grad_output.data()?;
        let ls = T::from(self.label_smoothing).unwrap();
        let one = <T as One>::one();
        let inv_c = T::from(1.0).unwrap() / T::from(classes).unwrap();

        let mut result = vec![<T as Zero>::zero(); batch * classes];

        for b in 0..batch {
            let base = b * classes;
            let target_class = targets_data[b].to_usize().unwrap_or(0);

            let scale = match self.reduction {
                Reduction::Mean => grad_data[0] / T::from(batch).unwrap(),
                Reduction::Sum => grad_data[0],
                Reduction::None => grad_data[b],
            };

            for c in 0..classes {
                let sm = sm_data[base + c];
                let one_hot = if c == target_class {
                    one
                } else {
                    <T as Zero>::zero()
                };
                // grad = softmax - ((1 - ls) * one_hot + ls / C)
                let target_dist = (one - ls) * one_hot + ls * inv_c;
                result[base + c] = (sm - target_dist) * scale;
            }
        }

        let grad_input = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
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
// BCEWithLogitsLoss
// ===========================================================================

/// Binary cross-entropy loss with logits (numerically stable).
///
/// ```text
/// loss = max(x, 0) - x*y + log(1 + exp(-|x|))
/// ```
///
/// Backward: `grad = sigmoid(x) - y`
#[derive(Debug, Clone)]
pub struct BCEWithLogitsLoss {
    pub reduction: Reduction,
}

impl BCEWithLogitsLoss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute BCE with logits loss.
    ///
    /// Participates in autocast: classified as `FullPrecision` (`"bce_with_logits"`).
    pub fn forward<T: Float>(
        &self,
        logits: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        autocast_guard("bce_with_logits");

        if logits.shape() != targets.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BCEWithLogitsLoss: logits shape {:?} != targets shape {:?}",
                    logits.shape(),
                    targets.shape()
                ),
            });
        }

        let logits_data = logits.data_vec()?;
        let targets_data = targets.data_vec()?;
        let zero = <T as Zero>::zero();
        let one = <T as One>::one();

        // loss = max(x, 0) - x*y + log(1 + exp(-|x|))
        let loss_data: Vec<T> = logits_data
            .iter()
            .zip(targets_data.iter())
            .map(|(&x, &y)| {
                let relu_x = if x > zero { x } else { zero };
                let abs_x = if x > zero { x } else { -x };
                relu_x - x * y + (one + (-abs_x).exp()).ln()
            })
            .collect();

        let unreduced = Tensor::from_storage(
            TensorStorage::cpu(loss_data),
            logits.shape().to_vec(),
            false,
        )?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && logits.requires_grad() {
            let grad_fn = Arc::new(BCEWithLogitsBackward {
                logits: logits.clone(),
                targets: targets.clone(),
                reduction: self.reduction,
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
}

impl Default for BCEWithLogitsLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

/// Backward for `BCEWithLogitsLoss`.
///
/// `grad = (sigmoid(x) - y) * grad_output`
#[derive(Debug)]
struct BCEWithLogitsBackward<T: Float> {
    logits: Tensor<T>,
    targets: Tensor<T>,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for BCEWithLogitsBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        use ferrotorch_core::autograd::no_grad::no_grad;
        use ferrotorch_core::grad_fns::activation::sigmoid;
        use ferrotorch_core::grad_fns::arithmetic::{div, mul, sub};

        // grad = (sigmoid(logits) - targets) * grad_output [/ n for mean]
        let grad_input = no_grad(|| {
            let sig = sigmoid(&self.logits)?;
            let diff = sub(&sig, &self.targets)?;
            let result = mul(&diff, grad_output)?;
            match self.reduction {
                Reduction::Mean => {
                    let n = ferrotorch_core::creation::scalar(
                        T::from(self.logits.shape().iter().product::<usize>()).unwrap(),
                    )?
                    .to(self.logits.device())?;
                    div(&result, &n)
                }
                _ => Ok(result),
            }
        })?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.logits]
    }

    fn name(&self) -> &'static str {
        "BCEWithLogitsBackward"
    }
}

// ===========================================================================
// HuberLoss
// ===========================================================================

/// Huber loss (smooth L1).
///
/// ```text
/// if |error| < delta:  0.5 * error^2
/// else:                delta * (|error| - 0.5 * delta)
/// ```
#[derive(Debug, Clone)]
pub struct HuberLoss {
    pub reduction: Reduction,
    pub delta: f64,
}

impl HuberLoss {
    pub fn new(reduction: Reduction, delta: f64) -> Self {
        Self { reduction, delta }
    }

    pub fn forward<T: Float>(
        &self,
        pred: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if pred.shape() != target.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "HuberLoss: pred shape {:?} != target shape {:?}",
                    pred.shape(),
                    target.shape()
                ),
            });
        }

        let pred_data = pred.data_vec()?;
        let target_data = target.data_vec()?;
        let delta = T::from(self.delta).unwrap();
        let half = T::from(0.5).unwrap();

        let loss_data: Vec<T> = pred_data
            .iter()
            .zip(target_data.iter())
            .map(|(&p, &t)| {
                let error = p - t;
                let abs_error = error.abs();
                if abs_error < delta {
                    half * error * error
                } else {
                    delta * (abs_error - half * delta)
                }
            })
            .collect();

        let unreduced =
            Tensor::from_storage(TensorStorage::cpu(loss_data), pred.shape().to_vec(), false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && pred.requires_grad() {
            let grad_fn = Arc::new(HuberBackward {
                pred: pred.clone(),
                target: target.clone(),
                delta: self.delta,
                reduction: self.reduction,
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
}

impl Default for HuberLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, 1.0)
    }
}

/// Backward for `HuberLoss`.
///
/// ```text
/// if |error| < delta:  grad * error
/// else:                grad * delta * sign(error)
/// ```
#[derive(Debug)]
struct HuberBackward<T: Float> {
    pred: Tensor<T>,
    target: Tensor<T>,
    delta: f64,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for HuberBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        use ferrotorch_core::autograd::no_grad::no_grad;
        use ferrotorch_core::grad_fns::arithmetic::{div, mul, sub};
        use ferrotorch_core::grad_fns::transcendental::clamp;

        // Huber gradient: clamp(pred - target, -delta, delta) * grad_output [/ n]
        let delta_t = T::from(self.delta).unwrap();
        let grad_input = no_grad(|| {
            let error = sub(&self.pred, &self.target)?;
            let clamped = clamp(&error, -delta_t, delta_t)?;
            let result = mul(&clamped, grad_output)?;
            match self.reduction {
                Reduction::Mean => {
                    let n = ferrotorch_core::creation::scalar(
                        T::from(self.pred.shape().iter().product::<usize>()).unwrap(),
                    )?
                    .to(self.pred.device())?;
                    div(&result, &n)
                }
                _ => Ok(result),
            }
        })?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.pred]
    }

    fn name(&self) -> &'static str {
        "HuberBackward"
    }
}

// ===========================================================================
// KLDivLoss
// ===========================================================================

/// Kullback-Leibler divergence loss.
///
/// Expects **log-probabilities** as `input` and **probabilities** as `target`:
///
/// ```text
/// loss_i = target_i * (log(target_i) - input_i)
/// ```
///
/// This matches PyTorch's `KLDivLoss` with `log_target=False`. The caller
/// is responsible for passing log-probabilities (e.g., from `LogSoftmax`).
///
/// Note: entries where `target_i == 0` contribute zero loss (0 * log(0) is
/// treated as 0 following the convention).
#[derive(Debug, Clone)]
pub struct KLDivLoss {
    pub reduction: Reduction,
}

impl KLDivLoss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }

    pub fn forward<T: Float>(
        &self,
        input: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if input.shape() != target.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "KLDivLoss: input shape {:?} != target shape {:?}",
                    input.shape(),
                    target.shape()
                ),
            });
        }

        let input_data = input.data_vec()?;
        let target_data = target.data_vec()?;
        let zero = <T as Zero>::zero();

        // KL(target || input) = sum(target * (log(target) - input))
        // where log(0) * 0 = 0 by convention.
        let loss_data: Vec<T> = input_data
            .iter()
            .zip(target_data.iter())
            .map(|(&inp, &tgt)| {
                if tgt > zero {
                    tgt * (tgt.ln() - inp)
                } else {
                    zero
                }
            })
            .collect();

        let unreduced =
            Tensor::from_storage(TensorStorage::cpu(loss_data), input.shape().to_vec(), false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(KLDivBackward {
                input: input.clone(),
                target: target.clone(),
                reduction: self.reduction,
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
}

impl Default for KLDivLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

/// Backward for `KLDivLoss`.
///
/// `grad_input = -target * grad_output` (since d/d(input) of target*(log(target) - input) = -target)
#[derive(Debug)]
struct KLDivBackward<T: Float> {
    input: Tensor<T>,
    target: Tensor<T>,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for KLDivBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        use ferrotorch_core::autograd::no_grad::no_grad;
        use ferrotorch_core::grad_fns::arithmetic::{div, mul, neg};

        // grad = -target * grad_output [/ n for mean]
        let grad_input = no_grad(|| {
            let neg_target = neg(&self.target)?;
            let result = mul(&neg_target, grad_output)?;
            match self.reduction {
                Reduction::Mean => {
                    let n = ferrotorch_core::creation::scalar(
                        T::from(self.input.shape().iter().product::<usize>()).unwrap(),
                    )?
                    .to(self.input.device())?;
                    div(&result, &n)
                }
                _ => Ok(result),
            }
        })?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "KLDivBackward"
    }
}

// ===========================================================================
// CosineEmbeddingLoss
// ===========================================================================

/// Cosine embedding loss for measuring similarity between pairs.
///
/// For positive pairs (y = 1):
/// ```text
/// loss = 1 - cos(x1, x2)
/// ```
///
/// For negative pairs (y = -1):
/// ```text
/// loss = max(0, cos(x1, x2) - margin)
/// ```
///
/// `x1` and `x2` must have the same shape. `y` must be a 1-D tensor of
/// `1.0` or `-1.0` values with length equal to the batch size (first dim).
#[derive(Debug, Clone)]
pub struct CosineEmbeddingLoss {
    pub reduction: Reduction,
    pub margin: f64,
}

impl CosineEmbeddingLoss {
    pub fn new(reduction: Reduction, margin: f64) -> Self {
        Self { reduction, margin }
    }

    /// Forward pass.
    ///
    /// - `x1`: tensor of shape `[B, D]` or `[D]`.
    /// - `x2`: tensor of shape `[B, D]` or `[D]`.
    /// - `y`: tensor of shape `[B]` or `[1]` with values `1.0` or `-1.0`.
    pub fn forward_pair<T: Float>(
        &self,
        x1: &Tensor<T>,
        x2: &Tensor<T>,
        y: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if x1.shape() != x2.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "CosineEmbeddingLoss: x1 shape {:?} != x2 shape {:?}",
                    x1.shape(),
                    x2.shape()
                ),
            });
        }

        let x1_data = x1.data_vec()?;
        let x2_data = x2.data_vec()?;
        let y_data = y.data_vec()?;
        let zero = <T as Zero>::zero();
        let one = <T as One>::one();
        let margin_t = T::from(self.margin).unwrap();

        let shape = x1.shape();
        let (batch, feat) = if shape.len() == 1 {
            (1, shape[0])
        } else if shape.len() == 2 {
            (shape[0], shape[1])
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CosineEmbeddingLoss: expected 1D or 2D input, got shape {:?}",
                    shape
                ),
            });
        };

        if y_data.len() != batch {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "CosineEmbeddingLoss: y length {} != batch size {}",
                    y_data.len(),
                    batch
                ),
            });
        }

        let mut losses = vec![zero; batch];
        for b in 0..batch {
            let base = b * feat;
            // Compute cosine similarity.
            let mut dot = zero;
            let mut norm1_sq = zero;
            let mut norm2_sq = zero;
            for f in 0..feat {
                let a = x1_data[base + f];
                let bv = x2_data[base + f];
                dot += a * bv;
                norm1_sq += a * a;
                norm2_sq += bv * bv;
            }
            let denom = norm1_sq.sqrt() * norm2_sq.sqrt();
            let cos_sim = if denom > zero { dot / denom } else { zero };

            if y_data[b] > zero {
                // Positive pair: loss = 1 - cos_sim.
                losses[b] = one - cos_sim;
            } else {
                // Negative pair: loss = max(0, cos_sim - margin).
                let v = cos_sim - margin_t;
                losses[b] = if v > zero { v } else { zero };
            }
        }

        let unreduced = Tensor::from_storage(TensorStorage::cpu(losses), vec![batch], false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && (x1.requires_grad() || x2.requires_grad()) {
            let grad_fn = Arc::new(CosineEmbeddingBackward {
                x1: x1.clone(),
                x2: x2.clone(),
                y: y.clone(),
                margin: self.margin,
                reduction: self.reduction,
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
}

impl Default for CosineEmbeddingLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, 0.0)
    }
}

/// Backward for `CosineEmbeddingLoss`.
///
/// For positive pairs (y = 1):
/// ```text
/// d(loss)/d(x1_f) = -(x2_f / (||x1|| * ||x2||) - cos_sim * x1_f / ||x1||^2)
/// d(loss)/d(x2_f) = -(x1_f / (||x1|| * ||x2||) - cos_sim * x2_f / ||x2||^2)
/// ```
///
/// For negative pairs (y = -1) where `cos(x1, x2) - margin > 0`:
/// ```text
/// d(loss)/d(x1_f) = x2_f / (||x1|| * ||x2||) - cos_sim * x1_f / ||x1||^2
/// d(loss)/d(x2_f) = x1_f / (||x1|| * ||x2||) - cos_sim * x2_f / ||x2||^2
/// ```
#[derive(Debug)]
struct CosineEmbeddingBackward<T: Float> {
    x1: Tensor<T>,
    x2: Tensor<T>,
    y: Tensor<T>,
    margin: f64,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for CosineEmbeddingBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.x1.shape();
        let (batch, feat) = if shape.len() == 1 {
            (1, shape[0])
        } else {
            (shape[0], shape[1])
        };

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CosineEmbedding backward",
            });
        }
        let x1_data = self.x1.data()?;
        let x2_data = self.x2.data()?;
        let y_data = self.y.data()?;
        let grad_data = grad_output.data()?;
        let zero = <T as Zero>::zero();
        let margin_t = T::from(self.margin).unwrap();

        let mut grad_x1 = vec![zero; batch * feat];
        let mut grad_x2 = vec![zero; batch * feat];

        for b in 0..batch {
            let base = b * feat;

            let scale = match self.reduction {
                Reduction::Mean => grad_data[0] / T::from(batch).unwrap(),
                Reduction::Sum => grad_data[0],
                Reduction::None => grad_data[b],
            };

            // Compute cosine similarity for this sample.
            let mut dot = zero;
            let mut norm1_sq = zero;
            let mut norm2_sq = zero;
            for f in 0..feat {
                let a = x1_data[base + f];
                let bv = x2_data[base + f];
                dot += a * bv;
                norm1_sq += a * a;
                norm2_sq += bv * bv;
            }
            let norm1 = norm1_sq.sqrt();
            let norm2 = norm2_sq.sqrt();
            let denom = norm1 * norm2;

            if denom <= zero {
                continue;
            }

            let cos_sim = dot / denom;

            let is_positive = y_data[b] > zero;
            let is_active = if is_positive {
                true
            } else {
                cos_sim - margin_t > zero
            };

            if !is_active {
                continue;
            }

            // sign: -1 for positive pairs, +1 for negative pairs
            let sign = if is_positive {
                -<T as One>::one()
            } else {
                <T as One>::one()
            };

            for f in 0..feat {
                let a = x1_data[base + f];
                let bv = x2_data[base + f];
                // d(cos)/d(x1_f) = x2_f / (||x1|| * ||x2||) - cos * x1_f / ||x1||^2
                let d_cos_x1 = bv / denom - cos_sim * a / norm1_sq;
                let d_cos_x2 = a / denom - cos_sim * bv / norm2_sq;
                grad_x1[base + f] = sign * d_cos_x1 * scale;
                grad_x2[base + f] = sign * d_cos_x2 * scale;
            }
        }

        let grad_x1_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_x1), shape.to_vec(), false)?;
        let grad_x2_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_x2), shape.to_vec(), false)?;
        Ok(vec![Some(grad_x1_tensor), Some(grad_x2_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.x1, &self.x2]
    }

    fn name(&self) -> &'static str {
        "CosineEmbeddingBackward"
    }
}

// ===========================================================================
// L1Loss
// ===========================================================================

/// L1 (Mean Absolute Error) loss.
///
/// ```text
/// loss_i = |pred_i - target_i|
/// ```
///
/// Then the chosen reduction is applied.
///
/// Matches `torch.nn.L1Loss`.
#[derive(Debug, Clone)]
pub struct L1Loss {
    pub reduction: Reduction,
}

impl L1Loss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute L1 loss.
    ///
    /// Participates in autocast: classified as `FullPrecision` (`"l1_loss"`).
    pub fn forward<T: Float>(
        &self,
        pred: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        autocast_guard("l1_loss");

        if pred.shape() != target.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "L1Loss: pred shape {:?} != target shape {:?}",
                    pred.shape(),
                    target.shape()
                ),
            });
        }

        let diff = binary_map(pred, target, |p, t| p - t)?;
        let abs_diff = unary_map(&diff, |x| x.abs())?;
        let reduced = apply_reduction(&abs_diff, self.reduction)?;

        if is_grad_enabled() && pred.requires_grad() {
            let grad_fn = Arc::new(L1Backward {
                pred: pred.clone(),
                target: target.clone(),
                reduction: self.reduction,
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
}

impl Default for L1Loss {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

/// Backward for `L1Loss`.
///
/// `grad_pred = sign(pred - target) * grad_output / n` (mean reduction)
/// `grad_pred = sign(pred - target) * grad_output`     (sum reduction)
/// `grad_pred = sign(pred - target) * grad_output`     (no reduction, elementwise)
///
/// `sign(0)` is defined as `0` to match PyTorch behavior.
#[derive(Debug)]
struct L1Backward<T: Float> {
    pred: Tensor<T>,
    target: Tensor<T>,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for L1Backward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "L1 backward" });
        }
        let pred_data = self.pred.data()?;
        let target_data = self.target.data()?;
        let grad_data = grad_output.data()?;
        let n = T::from(pred_data.len()).unwrap();

        let sign = |x: T| -> T {
            let zero = <T as Zero>::zero();
            if x > zero {
                <T as One>::one()
            } else if x < zero {
                -<T as One>::one()
            } else {
                zero
            }
        };

        let result: Vec<T> = match self.reduction {
            Reduction::Mean => {
                let go = grad_data[0];
                pred_data
                    .iter()
                    .zip(target_data.iter())
                    .map(|(&p, &t)| sign(p - t) * go / n)
                    .collect()
            }
            Reduction::Sum => {
                let go = grad_data[0];
                pred_data
                    .iter()
                    .zip(target_data.iter())
                    .map(|(&p, &t)| sign(p - t) * go)
                    .collect()
            }
            Reduction::None => pred_data
                .iter()
                .zip(target_data.iter())
                .zip(grad_data.iter())
                .map(|((&p, &t), &g)| sign(p - t) * g)
                .collect(),
        };

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.pred.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.pred]
    }

    fn name(&self) -> &'static str {
        "L1Backward"
    }
}

// ===========================================================================
// NLLLoss
// ===========================================================================

/// Negative log-likelihood loss.
///
/// Takes **log-probabilities** of shape `[B, C]` and integer class targets
/// `[B]` (stored as floats, e.g. `0.0`, `1.0`, `2.0`).
///
/// ```text
/// loss_b = -log_probs[b, target[b]]
/// ```
///
/// Supports an optional `ignore_index`: samples whose target equals this
/// value are excluded from the loss computation. When using `Reduction::Mean`,
/// the denominator is the count of non-ignored samples.
///
/// Matches `torch.nn.NLLLoss`.
#[derive(Debug, Clone)]
pub struct NLLLoss {
    pub reduction: Reduction,
    /// If set, class indices equal to this value are ignored.
    pub ignore_index: Option<isize>,
}

impl NLLLoss {
    pub fn new(reduction: Reduction, ignore_index: Option<isize>) -> Self {
        Self {
            reduction,
            ignore_index,
        }
    }

    /// Compute NLL loss.
    ///
    /// # Arguments
    ///
    /// * `log_probs` - Log-probabilities of shape `[B, C]`.
    /// * `targets` - Class indices of shape `[B]`, stored as floats.
    ///
    /// Participates in autocast: classified as `FullPrecision` (`"nll_loss"`).
    pub fn forward<T: Float>(
        &self,
        log_probs: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        autocast_guard("nll_loss");

        let shape = log_probs.shape();
        if shape.len() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "NLLLoss: expected 2D log_probs [B, C], got shape {:?}",
                    shape
                ),
            });
        }
        let batch = shape[0];
        let classes = shape[1];

        if targets.shape() != [batch] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "NLLLoss: target shape {:?} does not match batch size {}",
                    targets.shape(),
                    batch,
                ),
            });
        }

        if batch == 0 {
            // Empty batch: return scalar zero for Mean/Sum, empty [0] for None.
            return match self.reduction {
                Reduction::None => Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0], false),
                _ => Tensor::from_storage(
                    TensorStorage::cpu(vec![<T as Zero>::zero()]),
                    vec![],
                    false,
                ),
            };
        }

        let lp_data = log_probs.data_vec()?;
        let targets_data = targets.data_vec()?;

        let mut losses = vec![<T as Zero>::zero(); batch];
        let mut valid_count: usize = 0;

        for b in 0..batch {
            let target_idx = targets_data[b].to_isize().unwrap_or(0);

            // Check ignore_index.
            if let Some(ignore) = self.ignore_index {
                if target_idx == ignore {
                    // This sample is ignored (loss = 0, not counted).
                    continue;
                }
            }

            let target_class = target_idx as usize;
            if target_class >= classes {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "NLLLoss: target index {} is out of range for {} classes at batch element {}",
                        target_class, classes, b
                    ),
                });
            }

            losses[b] = -lp_data[b * classes + target_class];
            valid_count += 1;
        }

        let unreduced = Tensor::from_storage(TensorStorage::cpu(losses), vec![batch], false)?;

        // Apply reduction, but for Mean we need to use valid_count instead of batch.
        let reduced = match self.reduction {
            Reduction::None => unreduced.clone(),
            Reduction::Sum => sum(&unreduced)?,
            Reduction::Mean => {
                if valid_count == 0 {
                    // All samples ignored: return 0.
                    Tensor::from_storage(
                        TensorStorage::cpu(vec![<T as Zero>::zero()]),
                        vec![],
                        false,
                    )?
                } else {
                    let s = sum(&unreduced)?;
                    let s_data = s.data_vec()?;
                    let mean_val = s_data[0] / T::from(valid_count).unwrap();
                    Tensor::from_storage(TensorStorage::cpu(vec![mean_val]), vec![], false)?
                }
            }
        };

        if is_grad_enabled() && log_probs.requires_grad() {
            let grad_fn = Arc::new(NLLBackward {
                log_probs: log_probs.clone(),
                targets: targets.clone(),
                reduction: self.reduction,
                ignore_index: self.ignore_index,
                valid_count,
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
}

impl Default for NLLLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, None)
    }
}

/// Backward for `NLLLoss`.
///
/// `grad_log_probs[b, c] = 0` for `c != target[b]`
/// `grad_log_probs[b, target[b]] = -1 * scale`
///
/// where `scale = grad_output / valid_count` (mean) or `grad_output` (sum).
#[derive(Debug)]
struct NLLBackward<T: Float> {
    log_probs: Tensor<T>,
    targets: Tensor<T>,
    reduction: Reduction,
    ignore_index: Option<isize>,
    valid_count: usize,
}

impl<T: Float> GradFn<T> for NLLBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.log_probs.shape();
        let batch = shape[0];
        let classes = shape[1];

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "NLL backward" });
        }
        let targets_data = self.targets.data()?;
        let grad_data = grad_output.data()?;

        let mut result = vec![<T as Zero>::zero(); batch * classes];

        for b in 0..batch {
            let target_idx = targets_data[b].to_isize().unwrap_or(0);

            if let Some(ignore) = self.ignore_index {
                if target_idx == ignore {
                    continue;
                }
            }

            let target_class = target_idx as usize;

            let scale = match self.reduction {
                Reduction::Mean => {
                    if self.valid_count > 0 {
                        grad_data[0] / T::from(self.valid_count).unwrap()
                    } else {
                        <T as Zero>::zero()
                    }
                }
                Reduction::Sum => grad_data[0],
                Reduction::None => grad_data[b],
            };

            result[b * classes + target_class] = -scale;
        }

        let grad_input = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.log_probs]
    }

    fn name(&self) -> &'static str {
        "NLLBackward"
    }
}

// ===========================================================================
// SmoothL1Loss
// ===========================================================================

/// Smooth L1 loss, an alias for [`HuberLoss`] with `delta = 1.0`.
///
/// ```text
/// if |error| < 1:  0.5 * error^2
/// else:            |error| - 0.5
/// ```
///
/// This is the same as PyTorch's `SmoothL1Loss` with `beta=1.0`.
#[derive(Debug, Clone)]
pub struct SmoothL1Loss {
    pub reduction: Reduction,
}

impl SmoothL1Loss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }

    pub fn forward<T: Float>(
        &self,
        pred: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        // Delegate to HuberLoss with delta = 1.0.
        let huber = HuberLoss::new(self.reduction, 1.0);
        huber.forward(pred, target)
    }
}

impl Default for SmoothL1Loss {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

// ===========================================================================
// BCELoss
// ===========================================================================

/// Binary cross-entropy loss.
///
/// Expects **probabilities** (after sigmoid) as `input` and binary targets in `{0, 1}`.
///
/// ```text
/// loss_i = -(target_i * log(input_i) + (1 - target_i) * log(1 - input_i))
/// ```
///
/// **Important**: unlike [`BCEWithLogitsLoss`], this does **not** apply sigmoid
/// internally. Inputs must be in `[0, 1]`. For numerical stability, values are
/// clamped to `[eps, 1 - eps]` where `eps = 1e-12`.
///
/// Matches `torch.nn.BCELoss`.
#[derive(Debug, Clone)]
pub struct BCELoss {
    pub reduction: Reduction,
}

impl BCELoss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute BCE loss.
    ///
    /// Participates in autocast: classified as `FullPrecision` (`"bce_loss"`).
    pub fn forward<T: Float>(
        &self,
        input: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        autocast_guard("bce_loss");

        if input.shape() != target.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BCELoss: input shape {:?} != target shape {:?}",
                    input.shape(),
                    target.shape()
                ),
            });
        }

        let input_data = input.data_vec()?;
        let target_data = target.data_vec()?;
        let one = <T as One>::one();
        let eps = T::from(1e-12).unwrap();
        let one_m_eps = one - eps;

        let loss_data: Vec<T> = input_data
            .iter()
            .zip(target_data.iter())
            .map(|(&x, &y)| {
                // Clamp for numerical stability.
                let xc = if x < eps {
                    eps
                } else if x > one_m_eps {
                    one_m_eps
                } else {
                    x
                };
                -(y * xc.ln() + (one - y) * (one - xc).ln())
            })
            .collect();

        let unreduced =
            Tensor::from_storage(TensorStorage::cpu(loss_data), input.shape().to_vec(), false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(BCEBackward {
                input: input.clone(),
                target: target.clone(),
                reduction: self.reduction,
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
}

impl Default for BCELoss {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

/// Backward for `BCELoss`.
///
/// `grad = (-target / input + (1 - target) / (1 - input)) * grad_output`
#[derive(Debug)]
struct BCEBackward<T: Float> {
    input: Tensor<T>,
    target: Tensor<T>,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for BCEBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "BCE backward" });
        }
        let input_data = self.input.data()?;
        let target_data = self.target.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as One>::one();
        let eps = T::from(1e-12).unwrap();
        let one_m_eps = one - eps;
        let n = T::from(input_data.len()).unwrap();

        let result: Vec<T> = match self.reduction {
            Reduction::Mean => {
                let go = grad_data[0];
                input_data
                    .iter()
                    .zip(target_data.iter())
                    .map(|(&x, &y)| {
                        let xc = if x < eps {
                            eps
                        } else if x > one_m_eps {
                            one_m_eps
                        } else {
                            x
                        };
                        (-y / xc + (one - y) / (one - xc)) * go / n
                    })
                    .collect()
            }
            Reduction::Sum => {
                let go = grad_data[0];
                input_data
                    .iter()
                    .zip(target_data.iter())
                    .map(|(&x, &y)| {
                        let xc = if x < eps {
                            eps
                        } else if x > one_m_eps {
                            one_m_eps
                        } else {
                            x
                        };
                        (-y / xc + (one - y) / (one - xc)) * go
                    })
                    .collect()
            }
            Reduction::None => input_data
                .iter()
                .zip(target_data.iter())
                .zip(grad_data.iter())
                .map(|((&x, &y), &g)| {
                    let xc = if x < eps {
                        eps
                    } else if x > one_m_eps {
                        one_m_eps
                    } else {
                        x
                    };
                    (-y / xc + (one - y) / (one - xc)) * g
                })
                .collect(),
        };

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "BCEBackward"
    }
}

// ===========================================================================
// TripletMarginLoss
// ===========================================================================

/// Triplet margin loss for metric learning.
///
/// ```text
/// loss = max(0, d(anchor, positive) - d(anchor, negative) + margin)
/// ```
///
/// where `d(x, y) = ||x - y||_p` is the Lp distance. Default `p = 2`.
///
/// Matches `torch.nn.TripletMarginLoss`.
#[derive(Debug, Clone)]
pub struct TripletMarginLoss {
    pub reduction: Reduction,
    pub margin: f64,
    pub p: f64,
}

impl TripletMarginLoss {
    pub fn new(reduction: Reduction, margin: f64, p: f64) -> Self {
        Self {
            reduction,
            margin,
            p,
        }
    }

    /// Compute triplet margin loss.
    ///
    /// - `anchor`: tensor of shape `[B, D]`.
    /// - `positive`: tensor of shape `[B, D]`.
    /// - `negative`: tensor of shape `[B, D]`.
    pub fn forward<T: Float>(
        &self,
        anchor: &Tensor<T>,
        positive: &Tensor<T>,
        negative: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if anchor.shape() != positive.shape() || anchor.shape() != negative.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "TripletMarginLoss: shape mismatch: anchor {:?}, positive {:?}, negative {:?}",
                    anchor.shape(),
                    positive.shape(),
                    negative.shape()
                ),
            });
        }

        let shape = anchor.shape();
        if shape.len() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "TripletMarginLoss: expected 2D input [B, D], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let feat = shape[1];
        let anchor_data = anchor.data_vec()?;
        let positive_data = positive.data_vec()?;
        let negative_data = negative.data_vec()?;
        let zero = <T as Zero>::zero();
        let margin_t = T::from(self.margin).unwrap();
        let p_val = T::from(self.p).unwrap();
        let inv_p = T::from(1.0 / self.p).unwrap();

        let mut losses = vec![zero; batch];

        for (b, loss) in losses.iter_mut().enumerate() {
            let base = b * feat;
            let mut dist_pos = zero;
            let mut dist_neg = zero;
            for f in 0..feat {
                let dp = (anchor_data[base + f] - positive_data[base + f]).abs();
                let dn = (anchor_data[base + f] - negative_data[base + f]).abs();
                dist_pos += dp.powf(p_val);
                dist_neg += dn.powf(p_val);
            }
            dist_pos = dist_pos.powf(inv_p);
            dist_neg = dist_neg.powf(inv_p);

            let val = dist_pos - dist_neg + margin_t;
            *loss = if val > zero { val } else { zero };
        }

        let unreduced = Tensor::from_storage(TensorStorage::cpu(losses), vec![batch], false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && anchor.requires_grad() {
            let grad_fn = Arc::new(TripletMarginBackward {
                anchor: anchor.clone(),
                positive: positive.clone(),
                negative: negative.clone(),
                margin: self.margin,
                p: self.p,
                reduction: self.reduction,
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
}

impl Default for TripletMarginLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, 1.0, 2.0)
    }
}

/// Backward for `TripletMarginLoss`.
///
/// Only produces gradient for the anchor input. The gradient for the positive
/// and negative inputs is symmetric but we only track the anchor.
///
/// When `d_pos - d_neg + margin > 0`:
/// ```text
/// grad_anchor = (d(anchor - positive)/||a-p||_p - d(anchor - negative)/||a-n||_p) * scale
/// ```
#[derive(Debug)]
struct TripletMarginBackward<T: Float> {
    anchor: Tensor<T>,
    positive: Tensor<T>,
    negative: Tensor<T>,
    margin: f64,
    p: f64,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for TripletMarginBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.anchor.shape();
        let batch = shape[0];
        let feat = shape[1];
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "TripletMargin backward",
            });
        }
        let anchor_data = self.anchor.data()?;
        let positive_data = self.positive.data()?;
        let negative_data = self.negative.data()?;
        let grad_data = grad_output.data()?;

        let zero = <T as Zero>::zero();
        let p_val = T::from(self.p).unwrap();
        let margin_t = T::from(self.margin).unwrap();
        let inv_p = T::from(1.0 / self.p).unwrap();
        let p_m1 = p_val - <T as One>::one();

        let mut result = vec![zero; batch * feat];

        for b in 0..batch {
            let base = b * feat;

            // Compute distances.
            let mut dist_pos = zero;
            let mut dist_neg = zero;
            for f in 0..feat {
                let dp = (anchor_data[base + f] - positive_data[base + f]).abs();
                let dn = (anchor_data[base + f] - negative_data[base + f]).abs();
                dist_pos += dp.powf(p_val);
                dist_neg += dn.powf(p_val);
            }
            dist_pos = dist_pos.powf(inv_p);
            dist_neg = dist_neg.powf(inv_p);

            let triplet_val = dist_pos - dist_neg + margin_t;
            if triplet_val <= zero {
                // Hinge is zero — no gradient.
                continue;
            }

            let scale = match self.reduction {
                Reduction::Mean => grad_data[0] / T::from(batch).unwrap(),
                Reduction::Sum => grad_data[0],
                Reduction::None => grad_data[b],
            };

            let eps = T::from(1e-12).unwrap();

            for f in 0..feat {
                let diff_pos = anchor_data[base + f] - positive_data[base + f];
                let diff_neg = anchor_data[base + f] - negative_data[base + f];

                // d(||x||_p)/d(x_i) = sign(x_i) * |x_i|^(p-1) / ||x||_p^(p-1)
                let grad_pos = if dist_pos > eps {
                    diff_pos.signum() * diff_pos.abs().powf(p_m1) / dist_pos.powf(p_m1)
                } else {
                    zero
                };
                let grad_neg = if dist_neg > eps {
                    diff_neg.signum() * diff_neg.abs().powf(p_m1) / dist_neg.powf(p_m1)
                } else {
                    zero
                };

                result[base + f] = (grad_pos - grad_neg) * scale;
            }
        }

        let grad_input = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.anchor]
    }

    fn name(&self) -> &'static str {
        "TripletMarginBackward"
    }
}

// ===========================================================================
// MarginRankingLoss
// ===========================================================================

/// Margin ranking loss.
///
/// Given inputs `x1`, `x2` and label `y` (1 or -1):
///
/// ```text
/// loss = max(0, -y * (x1 - x2) + margin)
/// ```
///
/// Matches `torch.nn.MarginRankingLoss`.
#[derive(Debug, Clone)]
pub struct MarginRankingLoss {
    pub reduction: Reduction,
    pub margin: f64,
}

impl MarginRankingLoss {
    pub fn new(reduction: Reduction, margin: f64) -> Self {
        Self { reduction, margin }
    }

    /// Compute margin ranking loss.
    ///
    /// - `x1`: 1-D tensor of shape `[N]`.
    /// - `x2`: 1-D tensor of shape `[N]`.
    /// - `y`: 1-D tensor of shape `[N]` with values `1.0` or `-1.0`.
    pub fn forward<T: Float>(
        &self,
        x1: &Tensor<T>,
        x2: &Tensor<T>,
        y: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if x1.shape() != x2.shape() || x1.shape() != y.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MarginRankingLoss: shape mismatch: x1 {:?}, x2 {:?}, y {:?}",
                    x1.shape(),
                    x2.shape(),
                    y.shape()
                ),
            });
        }

        let x1_data = x1.data_vec()?;
        let x2_data = x2.data_vec()?;
        let y_data = y.data_vec()?;
        let zero = <T as Zero>::zero();
        let margin_t = T::from(self.margin).unwrap();

        let loss_data: Vec<T> = x1_data
            .iter()
            .zip(x2_data.iter())
            .zip(y_data.iter())
            .map(|((&a, &b), &yi)| {
                let val = -yi * (a - b) + margin_t;
                if val > zero { val } else { zero }
            })
            .collect();

        let unreduced =
            Tensor::from_storage(TensorStorage::cpu(loss_data), x1.shape().to_vec(), false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && x1.requires_grad() {
            let grad_fn = Arc::new(MarginRankingBackward {
                x1: x1.clone(),
                x2: x2.clone(),
                y: y.clone(),
                margin: self.margin,
                reduction: self.reduction,
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
}

impl Default for MarginRankingLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, 0.0)
    }
}

/// Backward for `MarginRankingLoss`.
///
/// When `-y * (x1 - x2) + margin > 0`:
/// `grad_x1 = -y * grad_output`
#[derive(Debug)]
struct MarginRankingBackward<T: Float> {
    x1: Tensor<T>,
    x2: Tensor<T>,
    y: Tensor<T>,
    margin: f64,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for MarginRankingBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "MarginRanking backward",
            });
        }
        let x1_data = self.x1.data()?;
        let x2_data = self.x2.data()?;
        let y_data = self.y.data()?;
        let grad_data = grad_output.data()?;
        let zero = <T as Zero>::zero();
        let margin_t = T::from(self.margin).unwrap();
        let n = T::from(x1_data.len()).unwrap();

        let result: Vec<T> = match self.reduction {
            Reduction::Mean => {
                let go = grad_data[0];
                x1_data
                    .iter()
                    .zip(x2_data.iter())
                    .zip(y_data.iter())
                    .map(|((&a, &b), &yi)| {
                        let val = -yi * (a - b) + margin_t;
                        if val > zero { -yi * go / n } else { zero }
                    })
                    .collect()
            }
            Reduction::Sum => {
                let go = grad_data[0];
                x1_data
                    .iter()
                    .zip(x2_data.iter())
                    .zip(y_data.iter())
                    .map(|((&a, &b), &yi)| {
                        let val = -yi * (a - b) + margin_t;
                        if val > zero { -yi * go } else { zero }
                    })
                    .collect()
            }
            Reduction::None => x1_data
                .iter()
                .zip(x2_data.iter())
                .zip(y_data.iter())
                .zip(grad_data.iter())
                .map(|(((&a, &b), &yi), &g)| {
                    let val = -yi * (a - b) + margin_t;
                    if val > zero { -yi * g } else { zero }
                })
                .collect(),
        };

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(result), self.x1.shape().to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.x1]
    }

    fn name(&self) -> &'static str {
        "MarginRankingBackward"
    }
}

// ===========================================================================
// CTCLoss
// ===========================================================================

/// Connectionist Temporal Classification loss.
///
/// Computes the CTC loss between a continuous (unsegmented) time series and a
/// target sequence. Used in speech recognition and OCR where alignment between
/// input and output is unknown.
///
/// - `log_probs`: Log-probabilities of shape `[T, B, C]` (time, batch, classes).
/// - `targets`: 1-D tensor of concatenated target sequences.
/// - `input_lengths`: Length of each input sequence in the batch.
/// - `target_lengths`: Length of each target sequence in the batch.
/// - `blank`: Index of the blank label (default 0).
///
/// Matches `torch.nn.CTCLoss`.
#[derive(Debug, Clone)]
pub struct CTCLoss {
    pub reduction: Reduction,
    pub blank: usize,
    pub zero_infinity: bool,
}

impl CTCLoss {
    pub fn new(reduction: Reduction, blank: usize, zero_infinity: bool) -> Self {
        Self {
            reduction,
            blank,
            zero_infinity,
        }
    }

    /// Compute CTC loss using the forward-backward algorithm.
    ///
    /// # Arguments
    ///
    /// * `log_probs` - Log-probabilities `[T, B, C]`.
    /// * `targets` - Concatenated target labels (1-D).
    /// * `input_lengths` - Input sequence lengths `[B]`.
    /// * `target_lengths` - Target sequence lengths `[B]`.
    pub fn forward<T: Float>(
        &self,
        log_probs: &Tensor<T>,
        targets: &Tensor<T>,
        input_lengths: &[usize],
        target_lengths: &[usize],
    ) -> FerrotorchResult<Tensor<T>> {
        let shape = log_probs.shape();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CTCLoss: expected 3D log_probs [T, B, C], got shape {:?}",
                    shape
                ),
            });
        }

        let max_t = shape[0];
        let batch = shape[1];
        let num_classes = shape[2];

        if input_lengths.len() != batch || target_lengths.len() != batch {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CTCLoss: batch size {} but input_lengths.len()={}, target_lengths.len()={}",
                    batch,
                    input_lengths.len(),
                    target_lengths.len()
                ),
            });
        }

        let lp_data = log_probs.data_vec()?;
        let targets_data = targets.data_vec()?;
        let neg_inf = T::from(-1e30).unwrap();
        let zero = <T as Zero>::zero();

        let mut losses = vec![zero; batch];
        let mut target_offset = 0usize;

        for b in 0..batch {
            let t_len = input_lengths[b].min(max_t);
            let s_len = target_lengths[b];

            // Extract this sample's targets.
            let tgt: Vec<usize> = (0..s_len)
                .map(|i| targets_data[target_offset + i].to_usize().unwrap_or(0))
                .collect();
            target_offset += s_len;

            if s_len == 0 {
                // Empty target: loss = -log_prob(blank at every timestep).
                let mut log_prob_blank = zero;
                for t in 0..t_len {
                    let idx = t * batch * num_classes + b * num_classes + self.blank;
                    log_prob_blank += lp_data[idx];
                }
                losses[b] = -log_prob_blank;
                continue;
            }

            // Build extended label sequence with blanks: [blank, s0, blank, s1, blank, ...]
            let ext_len = 2 * s_len + 1;
            let mut ext_labels = vec![self.blank; ext_len];
            for i in 0..s_len {
                ext_labels[2 * i + 1] = tgt[i];
            }

            // Forward pass: alpha[t][s] = log-prob of emitting ext_labels[0..=s] in time 0..=t.
            let mut alpha = vec![vec![neg_inf; ext_len]; t_len];

            // t = 0
            let lp_blank_0 = lp_data[b * num_classes + ext_labels[0]];
            alpha[0][0] = lp_blank_0;
            if ext_len > 1 {
                let lp_first = lp_data[b * num_classes + ext_labels[1]];
                alpha[0][1] = lp_first;
            }

            for t in 1..t_len {
                for s in 0..ext_len {
                    let lp_val = lp_data[t * batch * num_classes + b * num_classes + ext_labels[s]];

                    let mut log_sum = alpha[t - 1][s];
                    if s >= 1 {
                        log_sum = log_add_exp(log_sum, alpha[t - 1][s - 1]);
                    }
                    if s >= 2 && ext_labels[s] != self.blank && ext_labels[s] != ext_labels[s - 2] {
                        log_sum = log_add_exp(log_sum, alpha[t - 1][s - 2]);
                    }

                    alpha[t][s] = log_sum + lp_val;
                }
            }

            // Total log-probability: log_add_exp of the last two states.
            let log_prob =
                log_add_exp(alpha[t_len - 1][ext_len - 1], alpha[t_len - 1][ext_len - 2]);

            let loss_val = -log_prob;

            if self.zero_infinity && (loss_val == T::infinity() || loss_val.is_nan()) {
                losses[b] = zero;
            } else {
                losses[b] = loss_val;
            }
        }

        let unreduced = Tensor::from_storage(TensorStorage::cpu(losses), vec![batch], false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && log_probs.requires_grad() {
            let grad_fn = Arc::new(CTCBackward {
                log_probs: log_probs.clone(),
                targets: targets.clone(),
                input_lengths: input_lengths.to_vec(),
                target_lengths: target_lengths.to_vec(),
                blank: self.blank,
                zero_infinity: self.zero_infinity,
                reduction: self.reduction,
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
}

impl Default for CTCLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, 0, false)
    }
}

/// Log-add-exp: `log(exp(a) + exp(b))` in a numerically stable way.
fn log_add_exp<T: Float>(a: T, b: T) -> T {
    let max = if a > b { a } else { b };
    let min = if a > b { b } else { a };
    // If max is -inf, both are -inf.
    let threshold = T::from(-1e29).unwrap();
    if max < threshold {
        max
    } else {
        max + (min - max).exp().ln_1p()
    }
}

/// Backward for `CTCLoss`.
///
/// Uses the full forward-backward algorithm to compute gradients w.r.t.
/// `log_probs`. For each `(t, b, c)`:
///
/// ```text
/// grad[t, b, c] = exp(log_probs[t,b,c]) - (1/P) * sum_{s: ext[s]==c} exp(alpha[t][s] + beta[t][s] - log_probs[t,b,c])
/// ```
///
/// where `P = exp(log_prob_total)` is the total path probability.
#[derive(Debug)]
struct CTCBackward<T: Float> {
    log_probs: Tensor<T>,
    targets: Tensor<T>,
    input_lengths: Vec<usize>,
    target_lengths: Vec<usize>,
    blank: usize,
    zero_infinity: bool,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for CTCBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "CTC backward" });
        }

        let shape = self.log_probs.shape();
        let max_t = shape[0];
        let batch = shape[1];
        let num_classes = shape[2];
        let total_size = max_t * batch * num_classes;

        let lp_data = self.log_probs.data()?;
        let targets_data = self.targets.data()?;
        let grad_data = grad_output.data()?;
        let neg_inf = T::from(-1e30).unwrap();
        let zero = <T as Zero>::zero();

        let mut result = vec![zero; total_size];
        let mut target_offset = 0usize;

        for b in 0..batch {
            let t_len = self.input_lengths[b].min(max_t);
            let s_len = self.target_lengths[b];

            let tgt: Vec<usize> = (0..s_len)
                .map(|i| targets_data[target_offset + i].to_usize().unwrap_or(0))
                .collect();
            target_offset += s_len;

            let go_scale = match self.reduction {
                Reduction::Mean => grad_data[0] / T::from(batch).unwrap(),
                Reduction::Sum => grad_data[0],
                Reduction::None => grad_data[b],
            };

            if s_len == 0 {
                // Empty target: grad = -1 at blank for each timestep.
                for t in 0..t_len {
                    let idx = t * batch * num_classes + b * num_classes + self.blank;
                    result[idx] = -go_scale;
                }
                continue;
            }

            let ext_len = 2 * s_len + 1;
            let mut ext_labels = vec![self.blank; ext_len];
            for i in 0..s_len {
                ext_labels[2 * i + 1] = tgt[i];
            }

            // Forward pass (alpha).
            let mut alpha = vec![vec![neg_inf; ext_len]; t_len];
            alpha[0][0] = lp_data[b * num_classes + ext_labels[0]];
            if ext_len > 1 {
                alpha[0][1] = lp_data[b * num_classes + ext_labels[1]];
            }
            for t in 1..t_len {
                for s in 0..ext_len {
                    let lp_val = lp_data[t * batch * num_classes + b * num_classes + ext_labels[s]];
                    let mut log_sum = alpha[t - 1][s];
                    if s >= 1 {
                        log_sum = log_add_exp(log_sum, alpha[t - 1][s - 1]);
                    }
                    if s >= 2 && ext_labels[s] != self.blank && ext_labels[s] != ext_labels[s - 2] {
                        log_sum = log_add_exp(log_sum, alpha[t - 1][s - 2]);
                    }
                    alpha[t][s] = log_sum + lp_val;
                }
            }

            let log_prob =
                log_add_exp(alpha[t_len - 1][ext_len - 1], alpha[t_len - 1][ext_len - 2]);

            if self.zero_infinity && ((-log_prob) == T::infinity() || (-log_prob).is_nan()) {
                continue;
            }

            // Backward pass (beta).
            // beta[t][s] = log P(observing labels l'[s..] from time t onward),
            // where alpha INCLUDES the emission at time t but beta does NOT.
            // This avoids double-counting: alpha[t][s] + beta[t][s] =
            // log P(path passes through state s at time t).
            //
            // Initialization: beta[T-1][s] = 0 for valid ending states.
            // Recurrence: beta[t][s] = log_add_exp over successors s' of
            //   (y_{t+1}^{l'[s']} + beta[t+1][s']).
            let mut beta = vec![vec![neg_inf; ext_len]; t_len];
            beta[t_len - 1][ext_len - 1] = zero;
            if ext_len > 1 {
                beta[t_len - 1][ext_len - 2] = zero;
            }
            for t in (0..t_len.saturating_sub(1)).rev() {
                for s in (0..ext_len).rev() {
                    // Successor s (same state): y_{t+1}^{l'[s]} + beta[t+1][s]
                    let lp_s =
                        lp_data[(t + 1) * batch * num_classes + b * num_classes + ext_labels[s]];
                    let mut log_sum = lp_s + beta[t + 1][s];
                    if s + 1 < ext_len {
                        let lp_s1 = lp_data
                            [(t + 1) * batch * num_classes + b * num_classes + ext_labels[s + 1]];
                        log_sum = log_add_exp(log_sum, lp_s1 + beta[t + 1][s + 1]);
                    }
                    if s + 2 < ext_len
                        && ext_labels[s] != self.blank
                        && ext_labels[s] != ext_labels[s + 2]
                    {
                        let lp_s2 = lp_data
                            [(t + 1) * batch * num_classes + b * num_classes + ext_labels[s + 2]];
                        log_sum = log_add_exp(log_sum, lp_s2 + beta[t + 1][s + 2]);
                    }
                    beta[t][s] = log_sum;
                }
            }

            // Accumulate gradients.
            // alpha[t][s] includes emission, beta[t][s] does not, so
            // alpha[t][s] + beta[t][s] = log P(all paths through state s at time t).
            //
            // d(-log P)/d(log_probs[t,b,c]) = -(1/P) * dP/d(log_probs[t,b,c])
            // Since y_t^c = exp(log_probs[t,b,c]):
            //   dP/d(log_probs[t,b,c]) = sum_{s:l'[s]=c} exp(alpha[t][s] + beta[t][s])
            //
            // So: grad[t,b,c] = -exp(log_ab_per_class[c] - log_prob)
            for t in 0..t_len {
                let mut log_ab_per_class = vec![neg_inf; num_classes];
                for s in 0..ext_len {
                    let c = ext_labels[s];
                    let ab = alpha[t][s] + beta[t][s];
                    log_ab_per_class[c] = log_add_exp(log_ab_per_class[c], ab);
                }

                let threshold = T::from(-1e29).unwrap();
                let base_idx = t * batch * num_classes + b * num_classes;
                for (c, &log_ab) in log_ab_per_class.iter().enumerate() {
                    let occupation = if log_ab > threshold {
                        (log_ab - log_prob).exp()
                    } else {
                        zero
                    };
                    result[base_idx + c] = -occupation * go_scale;
                }
            }
        }

        let grad_input = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.log_probs]
    }

    fn name(&self) -> &'static str {
        "CTCBackward"
    }
}

// ===========================================================================
// PoissonNLLLoss
// ===========================================================================

/// Negative log-likelihood loss with Poisson distribution.
///
/// Target is expected to be a count (non-negative integer or float).
///
/// ```text
/// loss = exp(input) - target * input      (if log_input=true, the default)
/// loss = input - target * log(input+eps)  (if log_input=false)
/// ```
///
/// Matches `torch.nn.PoissonNLLLoss`.
#[derive(Debug, Clone)]
pub struct PoissonNLLLoss {
    pub reduction: Reduction,
    pub log_input: bool,
    pub eps: f64,
}

impl PoissonNLLLoss {
    pub fn new(reduction: Reduction, log_input: bool, eps: f64) -> Self {
        Self {
            reduction,
            log_input,
            eps,
        }
    }

    /// Compute Poisson NLL loss.
    pub fn forward<T: Float>(
        &self,
        input: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if input.shape() != target.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "PoissonNLLLoss: input shape {:?} != target shape {:?}",
                    input.shape(),
                    target.shape()
                ),
            });
        }

        let input_data = input.data_vec()?;
        let target_data = target.data_vec()?;
        let eps_t = T::from(self.eps).unwrap();

        let loss_data: Vec<T> = input_data
            .iter()
            .zip(target_data.iter())
            .map(|(&x, &y)| {
                if self.log_input {
                    // loss = exp(x) - y * x
                    x.exp() - y * x
                } else {
                    // loss = x - y * log(x + eps)
                    x - y * (x + eps_t).ln()
                }
            })
            .collect();

        let unreduced =
            Tensor::from_storage(TensorStorage::cpu(loss_data), input.shape().to_vec(), false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(PoissonNLLBackward {
                input: input.clone(),
                target: target.clone(),
                log_input: self.log_input,
                eps: self.eps,
                reduction: self.reduction,
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
}

impl Default for PoissonNLLLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, true, 1e-8)
    }
}

/// Backward for `PoissonNLLLoss`.
///
/// ```text
/// log_input=true:  grad = (exp(input) - target) * grad_output
/// log_input=false: grad = (1 - target / (input + eps)) * grad_output
/// ```
#[derive(Debug)]
struct PoissonNLLBackward<T: Float> {
    input: Tensor<T>,
    target: Tensor<T>,
    log_input: bool,
    eps: f64,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for PoissonNLLBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        use ferrotorch_core::autograd::no_grad::no_grad;
        use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, sub};
        use ferrotorch_core::grad_fns::transcendental::exp;

        let device = self.input.device();
        let grad_input = no_grad(|| {
            // local = exp(input) - target  OR  1 - target / (input + eps)
            let local = if self.log_input {
                let exp_input = exp(&self.input)?;
                sub(&exp_input, &self.target)?
            } else {
                let eps =
                    ferrotorch_core::creation::scalar(T::from(self.eps).unwrap())?.to(device)?;
                let one = ferrotorch_core::creation::scalar(<T as One>::one())?.to(device)?;
                let denom = add(&self.input, &eps)?;
                let ratio = div(&self.target, &denom)?;
                sub(&one, &ratio)?
            };
            let result = mul(&local, grad_output)?;
            match self.reduction {
                Reduction::Mean => {
                    let n = ferrotorch_core::creation::scalar(
                        T::from(self.input.shape().iter().product::<usize>()).unwrap(),
                    )?
                    .to(device)?;
                    div(&result, &n)
                }
                _ => Ok(result),
            }
        })?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "PoissonNLLBackward"
    }
}

// ===========================================================================
// MultiMarginLoss
// ===========================================================================

/// Multi-class margin loss (hinge loss for classification).
///
/// For each sample with true class `y`:
///
/// ```text
/// loss = (1/C) * sum_{j != y} max(0, margin - x[y] + x[j])^p
/// ```
///
/// where `p` is 1 or 2 (default 1).
///
/// Matches `torch.nn.MultiMarginLoss`.
#[derive(Debug, Clone)]
pub struct MultiMarginLoss {
    pub reduction: Reduction,
    pub p: usize,
    pub margin: f64,
}

impl MultiMarginLoss {
    pub fn new(reduction: Reduction, p: usize, margin: f64) -> Self {
        Self {
            reduction,
            p,
            margin,
        }
    }

    /// Compute multi-margin loss.
    ///
    /// - `input`: `[B, C]` (scores for each class).
    /// - `target`: `[B]` (class indices, stored as floats).
    pub fn forward<T: Float>(
        &self,
        input: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MultiMarginLoss: expected 2D input [B, C], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let classes = shape[1];

        if target.shape() != [batch] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MultiMarginLoss: target shape {:?} does not match batch size {}",
                    target.shape(),
                    batch,
                ),
            });
        }

        let input_data = input.data_vec()?;
        let target_data = target.data_vec()?;
        let zero = <T as Zero>::zero();
        let margin_t = T::from(self.margin).unwrap();
        let inv_c = T::from(1.0).unwrap() / T::from(classes).unwrap();

        let mut losses = vec![zero; batch];

        for b in 0..batch {
            let base = b * classes;
            let y = target_data[b].to_usize().unwrap_or(0);
            let x_y = input_data[base + y];
            let mut sample_loss = zero;

            for j in 0..classes {
                if j == y {
                    continue;
                }
                let val = margin_t - x_y + input_data[base + j];
                if val > zero {
                    sample_loss += if self.p == 2 { val * val } else { val };
                }
            }

            losses[b] = sample_loss * inv_c;
        }

        let unreduced = Tensor::from_storage(TensorStorage::cpu(losses), vec![batch], false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(MultiMarginBackward {
                input: input.clone(),
                target: target.clone(),
                p: self.p,
                margin: self.margin,
                reduction: self.reduction,
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
}

impl Default for MultiMarginLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, 1, 1.0)
    }
}

/// Backward for `MultiMarginLoss`.
#[derive(Debug)]
struct MultiMarginBackward<T: Float> {
    input: Tensor<T>,
    target: Tensor<T>,
    p: usize,
    margin: f64,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for MultiMarginBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.input.shape();
        let batch = shape[0];
        let classes = shape[1];

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "MultiMargin backward",
            });
        }
        let input_data = self.input.data()?;
        let target_data = self.target.data()?;
        let grad_data = grad_output.data()?;

        let zero = <T as Zero>::zero();
        let one = <T as One>::one();
        let two = T::from(2.0).unwrap();
        let margin_t = T::from(self.margin).unwrap();
        let inv_c = one / T::from(classes).unwrap();

        let mut result = vec![zero; batch * classes];

        for b in 0..batch {
            let base = b * classes;
            let y = target_data[b].to_usize().unwrap_or(0);
            let x_y = input_data[base + y];

            let scale = match self.reduction {
                Reduction::Mean => grad_data[0] / T::from(batch).unwrap(),
                Reduction::Sum => grad_data[0],
                Reduction::None => grad_data[b],
            };

            let mut grad_y = zero;
            for j in 0..classes {
                if j == y {
                    continue;
                }
                let val = margin_t - x_y + input_data[base + j];
                if val > zero {
                    // d/d(x_j) of max(0, margin - x_y + x_j)^p
                    let g_j = if self.p == 2 { two * val } else { one };
                    result[base + j] = g_j * inv_c * scale;
                    // d/d(x_y) accumulates -g_j
                    grad_y = grad_y - g_j * inv_c * scale;
                }
            }
            result[base + y] += grad_y;
        }

        let grad_input = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MultiMarginBackward"
    }
}

// ===========================================================================
// MultiLabelSoftMarginLoss
// ===========================================================================

/// Multi-label one-versus-all loss based on max-entropy.
///
/// For each element:
///
/// ```text
/// loss = -(target * log(sigma(input)) + (1 - target) * log(1 - sigma(input)))
/// ```
///
/// This is equivalent to `BCEWithLogitsLoss` applied independently per label,
/// then summed over the class dimension and reduced over the batch.
///
/// Matches `torch.nn.MultiLabelSoftMarginLoss`.
#[derive(Debug, Clone)]
pub struct MultiLabelSoftMarginLoss {
    pub reduction: Reduction,
}

impl MultiLabelSoftMarginLoss {
    pub fn new(reduction: Reduction) -> Self {
        Self { reduction }
    }

    /// Compute multi-label soft margin loss.
    ///
    /// - `input`: `[B, C]` (raw logits).
    /// - `target`: `[B, C]` (binary labels, 0 or 1).
    pub fn forward<T: Float>(
        &self,
        input: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        if shape.len() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MultiLabelSoftMarginLoss: expected 2D input [B, C], got shape {:?}",
                    shape
                ),
            });
        }

        if input.shape() != target.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MultiLabelSoftMarginLoss: input shape {:?} != target shape {:?}",
                    input.shape(),
                    target.shape()
                ),
            });
        }

        let batch = shape[0];
        let classes = shape[1];
        let input_data = input.data_vec()?;
        let target_data = target.data_vec()?;
        let zero = <T as Zero>::zero();
        let one = <T as One>::one();
        let inv_c = one / T::from(classes).unwrap();

        // Per-sample loss: mean over classes of BCE-with-logits.
        let mut losses = vec![zero; batch];

        for (b, loss) in losses.iter_mut().enumerate() {
            let base = b * classes;
            let mut sample_loss = zero;
            for c in 0..classes {
                let x = input_data[base + c];
                let y = target_data[base + c];
                // BCE with logits: max(x,0) - x*y + log(1 + exp(-|x|))
                let relu_x = if x > zero { x } else { zero };
                let abs_x = if x > zero { x } else { -x };
                sample_loss += relu_x - x * y + (one + (-abs_x).exp()).ln();
            }
            *loss = sample_loss * inv_c;
        }

        let unreduced = Tensor::from_storage(TensorStorage::cpu(losses), vec![batch], false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(MultiLabelSoftMarginBackward {
                input: input.clone(),
                target: target.clone(),
                reduction: self.reduction,
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
}

impl Default for MultiLabelSoftMarginLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean)
    }
}

/// Backward for `MultiLabelSoftMarginLoss`.
///
/// `grad = (sigmoid(input) - target) / C * scale`
#[derive(Debug)]
struct MultiLabelSoftMarginBackward<T: Float> {
    input: Tensor<T>,
    target: Tensor<T>,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for MultiLabelSoftMarginBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let shape = self.input.shape();
        let batch = shape[0];
        let classes = shape[1];

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "MultiLabelSoftMargin backward",
            });
        }
        let input_data = self.input.data()?;
        let target_data = self.target.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as One>::one();
        let inv_c = one / T::from(classes).unwrap();

        let mut result = vec![<T as Zero>::zero(); batch * classes];

        for b in 0..batch {
            let base = b * classes;

            let scale = match self.reduction {
                Reduction::Mean => grad_data[0] / T::from(batch).unwrap(),
                Reduction::Sum => grad_data[0],
                Reduction::None => grad_data[b],
            };

            for c in 0..classes {
                let x = input_data[base + c];
                let y = target_data[base + c];
                let sig = one / (one + (-x).exp());
                result[base + c] = (sig - y) * inv_c * scale;
            }
        }

        let grad_input = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MultiLabelSoftMarginBackward"
    }
}

// ===========================================================================
// HingeEmbeddingLoss
// ===========================================================================

/// Hinge embedding loss for learning non-linear embeddings or semi-supervised
/// learning.
///
/// ```text
/// loss = x           if y == 1
/// loss = max(0, margin - x)  if y == -1
/// ```
///
/// Matches `torch.nn.HingeEmbeddingLoss`.
#[derive(Debug, Clone)]
pub struct HingeEmbeddingLoss {
    pub reduction: Reduction,
    pub margin: f64,
}

impl HingeEmbeddingLoss {
    pub fn new(reduction: Reduction, margin: f64) -> Self {
        Self { reduction, margin }
    }

    /// Compute hinge embedding loss.
    ///
    /// - `input`: distance or similarity values.
    /// - `y`: labels, `1.0` for positive, `-1.0` for negative.
    pub fn forward<T: Float>(
        &self,
        input: &Tensor<T>,
        y: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if input.shape() != y.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "HingeEmbeddingLoss: input shape {:?} != y shape {:?}",
                    input.shape(),
                    y.shape()
                ),
            });
        }

        let input_data = input.data_vec()?;
        let y_data = y.data_vec()?;
        let zero = <T as Zero>::zero();
        let margin_t = T::from(self.margin).unwrap();

        let loss_data: Vec<T> = input_data
            .iter()
            .zip(y_data.iter())
            .map(|(&x, &yi)| {
                if yi > zero {
                    // Positive: loss = x
                    x
                } else {
                    // Negative: loss = max(0, margin - x)
                    let val = margin_t - x;
                    if val > zero { val } else { zero }
                }
            })
            .collect();

        let unreduced =
            Tensor::from_storage(TensorStorage::cpu(loss_data), input.shape().to_vec(), false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(HingeEmbeddingBackward {
                input: input.clone(),
                y: y.clone(),
                margin: self.margin,
                reduction: self.reduction,
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
}

impl Default for HingeEmbeddingLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, 1.0)
    }
}

/// Backward for `HingeEmbeddingLoss`.
///
/// ```text
/// y == 1:   grad = 1 * grad_output
/// y == -1:  grad = -1 * grad_output  if margin - x > 0, else 0
/// ```
#[derive(Debug)]
struct HingeEmbeddingBackward<T: Float> {
    input: Tensor<T>,
    y: Tensor<T>,
    margin: f64,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for HingeEmbeddingBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "HingeEmbedding backward",
            });
        }
        let input_data = self.input.data()?;
        let y_data = self.y.data()?;
        let grad_data = grad_output.data()?;
        let zero = <T as Zero>::zero();
        let one = <T as One>::one();
        let margin_t = T::from(self.margin).unwrap();
        let n = T::from(input_data.len()).unwrap();

        let result: Vec<T> = match self.reduction {
            Reduction::Mean => {
                let go = grad_data[0];
                input_data
                    .iter()
                    .zip(y_data.iter())
                    .map(|(&x, &yi)| {
                        if yi > zero {
                            one * go / n
                        } else {
                            let val = margin_t - x;
                            if val > zero { -one * go / n } else { zero }
                        }
                    })
                    .collect()
            }
            Reduction::Sum => {
                let go = grad_data[0];
                input_data
                    .iter()
                    .zip(y_data.iter())
                    .map(|(&x, &yi)| {
                        if yi > zero {
                            one * go
                        } else {
                            let val = margin_t - x;
                            if val > zero { -one * go } else { zero }
                        }
                    })
                    .collect()
            }
            Reduction::None => input_data
                .iter()
                .zip(y_data.iter())
                .zip(grad_data.iter())
                .map(|((&x, &yi), &g)| {
                    if yi > zero {
                        one * g
                    } else {
                        let val = margin_t - x;
                        if val > zero { -one * g } else { zero }
                    }
                })
                .collect(),
        };

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "HingeEmbeddingBackward"
    }
}

// ===========================================================================
// GaussianNLLLoss
// ===========================================================================

/// Gaussian negative log-likelihood loss.
///
/// Models the target as drawn from a Gaussian with predicted mean and variance:
///
/// ```text
/// loss = 0.5 * (log(var) + (input - target)^2 / var + log(2*pi))
/// ```
///
/// The `log(2*pi)` constant is included when `full` is `true` (default `false`),
/// matching `torch.nn.GaussianNLLLoss`.
///
/// - `input`: predicted mean, any shape.
/// - `target`: observed values, same shape as `input`.
/// - `var`: predicted variance, same shape as `input` (must be positive).
///
/// The `eps` parameter clamps variance from below for numerical stability.
#[derive(Debug, Clone)]
pub struct GaussianNLLLoss {
    pub reduction: Reduction,
    pub full: bool,
    pub eps: f64,
}

impl GaussianNLLLoss {
    pub fn new(reduction: Reduction, full: bool, eps: f64) -> Self {
        Self {
            reduction,
            full,
            eps,
        }
    }

    /// Compute Gaussian NLL loss.
    ///
    /// Participates in autocast: classified as `FullPrecision` (`"gaussian_nll_loss"`).
    pub fn forward<T: Float>(
        &self,
        input: &Tensor<T>,
        target: &Tensor<T>,
        var: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        autocast_guard("gaussian_nll_loss");

        if input.shape() != target.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GaussianNLLLoss: input shape {:?} != target shape {:?}",
                    input.shape(),
                    target.shape()
                ),
            });
        }

        if input.shape() != var.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GaussianNLLLoss: input shape {:?} != var shape {:?}",
                    input.shape(),
                    var.shape()
                ),
            });
        }

        let input_data = input.data_vec()?;
        let target_data = target.data_vec()?;
        let var_data = var.data_vec()?;
        let half = T::from(0.5).unwrap();
        let eps_t = T::from(self.eps).unwrap();
        let log_2pi = T::from((2.0 * std::f64::consts::PI).ln()).unwrap();

        let loss_data: Vec<T> = input_data
            .iter()
            .zip(target_data.iter())
            .zip(var_data.iter())
            .map(|((&inp, &tgt), &v)| {
                let v_clamped = if v < eps_t { eps_t } else { v };
                let diff = inp - tgt;
                let mut l = half * (v_clamped.ln() + diff * diff / v_clamped);
                if self.full {
                    l += half * log_2pi;
                }
                l
            })
            .collect();

        let unreduced =
            Tensor::from_storage(TensorStorage::cpu(loss_data), input.shape().to_vec(), false)?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && (input.requires_grad() || var.requires_grad()) {
            let grad_fn = Arc::new(GaussianNLLBackward {
                input: input.clone(),
                target: target.clone(),
                var: var.clone(),
                eps: self.eps,
                reduction: self.reduction,
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
}

impl Default for GaussianNLLLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, false, 1e-6)
    }
}

/// Backward for `GaussianNLLLoss`.
///
/// ```text
/// d(loss)/d(input) = (input - target) / var
/// d(loss)/d(var)   = 0.5 * (1/var - (input - target)^2 / var^2)
/// ```
#[derive(Debug)]
struct GaussianNLLBackward<T: Float> {
    input: Tensor<T>,
    target: Tensor<T>,
    var: Tensor<T>,
    eps: f64,
    reduction: Reduction,
}

impl<T: Float> GradFn<T> for GaussianNLLBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "GaussianNLL backward",
            });
        }

        let input_data = self.input.data()?;
        let target_data = self.target.data()?;
        let var_data = self.var.data()?;
        let grad_data = grad_output.data()?;
        let n = input_data.len();
        let half = T::from(0.5).unwrap();
        let eps_t = T::from(self.eps).unwrap();
        let zero = <T as Zero>::zero();

        let mut grad_input = vec![zero; n];
        let mut grad_var = vec![zero; n];

        for i in 0..n {
            let scale = match self.reduction {
                Reduction::Mean => grad_data[0] / T::from(n).unwrap(),
                Reduction::Sum => grad_data[0],
                Reduction::None => grad_data[i],
            };

            let v = if var_data[i] < eps_t {
                eps_t
            } else {
                var_data[i]
            };
            let diff = input_data[i] - target_data[i];

            // d(loss)/d(input) = (input - target) / var
            grad_input[i] = diff / v * scale;

            // d(loss)/d(var) = 0.5 * (1/var - diff^2 / var^2)
            grad_var[i] = half * (<T as One>::one() / v - diff * diff / (v * v)) * scale;
        }

        let shape = self.input.shape().to_vec();
        let grad_input_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_input), shape.clone(), false)?;
        let grad_var_tensor = Tensor::from_storage(TensorStorage::cpu(grad_var), shape, false)?;
        Ok(vec![Some(grad_input_tensor), Some(grad_var_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.var]
    }

    fn name(&self) -> &'static str {
        "GaussianNLLBackward"
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
#[allow(clippy::needless_range_loop)]
mod tests {
    use super::*;
    use ferrotorch_core::autograd::graph::backward;
    use ferrotorch_core::storage::TensorStorage;

    /// Helper: 1-D leaf tensor with requires_grad.
    fn leaf_vec(vals: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vals.to_vec()), vec![vals.len()], true).unwrap()
    }

    /// Helper: 1-D tensor without grad (for targets).
    fn target_vec(vals: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vals.to_vec()), vec![vals.len()], false).unwrap()
    }

    /// Helper: 2-D leaf tensor with requires_grad.
    fn leaf_2d(vals: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vals.to_vec()), shape.to_vec(), true).unwrap()
    }

    // -----------------------------------------------------------------------
    // MSELoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_mse_forward_mean() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);
        let loss = MSELoss::new(Reduction::Mean);
        let out = loss.forward(&pred, &target).unwrap();
        // Each diff is 0.5, squared is 0.25, mean is 0.25.
        assert!(out.is_scalar());
        assert!(
            (out.item().unwrap() - 0.25).abs() < 1e-7,
            "MSE mean: expected 0.25, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_mse_forward_sum() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);
        let loss = MSELoss::new(Reduction::Sum);
        let out = loss.forward(&pred, &target).unwrap();
        // sum of 0.25 * 3 = 0.75
        assert!(
            (out.item().unwrap() - 0.75).abs() < 1e-7,
            "MSE sum: expected 0.75, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_mse_forward_none() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);
        let loss = MSELoss::new(Reduction::None);
        let out = loss.forward(&pred, &target).unwrap();
        assert_eq!(out.shape(), &[3]);
        let d = out.data().unwrap();
        for i in 0..3 {
            assert!(
                (d[i] - 0.25).abs() < 1e-7,
                "MSE none[{}]: expected 0.25, got {}",
                i,
                d[i]
            );
        }
    }

    #[test]
    fn test_mse_backward_mean() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);
        let loss = MSELoss::new(Reduction::Mean);
        let out = loss.forward(&pred, &target).unwrap();
        backward(&out).unwrap();

        let grad = pred.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad = 2 * (pred - target) / n = 2 * (-0.5) / 3 = -1/3
        let expected = -1.0 / 3.0;
        for i in 0..3 {
            assert!(
                (g[i] - expected).abs() < 1e-7,
                "MSE grad[{}]: expected {}, got {}",
                i,
                expected,
                g[i]
            );
        }
    }

    #[test]
    fn test_mse_backward_sum() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);
        let loss = MSELoss::new(Reduction::Sum);
        let out = loss.forward(&pred, &target).unwrap();
        backward(&out).unwrap();

        let grad = pred.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad = 2 * (-0.5) = -1.0
        for i in 0..3 {
            assert!(
                (g[i] - (-1.0)).abs() < 1e-7,
                "MSE sum grad[{}]: expected -1.0, got {}",
                i,
                g[i]
            );
        }
    }

    #[test]
    fn test_mse_zero_loss() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.0, 2.0, 3.0]);
        let loss = MSELoss::default();
        let out = loss.forward(&pred, &target).unwrap();
        assert!(out.item().unwrap().abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // CrossEntropyLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_cross_entropy_forward_mean() {
        // 2 samples, 3 classes
        // logits: [[1, 2, 3], [1, 2, 3]], targets: [2, 0]
        let logits = leaf_2d(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let targets = target_vec(&[2.0, 0.0]);
        let loss = CrossEntropyLoss::default();
        let out = loss.forward(&logits, &targets).unwrap();

        // log_softmax uses max subtraction: max=3, shifted = [-2, -1, 0]
        // sum_exp = e^{-2} + e^{-1} + e^0
        let sum_exp = (-2.0_f64).exp() + (-1.0_f64).exp() + 1.0;
        let log_sum = sum_exp.ln();
        // log_softmax[c] = logits[c] - max - log_sum
        let lsm = [
            1.0 - 3.0 - log_sum,
            2.0 - 3.0 - log_sum,
            3.0 - 3.0 - log_sum,
        ];
        // nll for sample 0 (target=2): -lsm[2]
        // nll for sample 1 (target=0): -lsm[0]
        let expected = (-lsm[2] + (-lsm[0])) / 2.0;

        assert!(
            (out.item().unwrap() - expected).abs() < 1e-6,
            "CE mean: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_cross_entropy_forward_sum() {
        let logits = leaf_2d(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let targets = target_vec(&[2.0, 0.0]);
        let loss = CrossEntropyLoss::new(Reduction::Sum, 0.0);
        let out = loss.forward(&logits, &targets).unwrap();

        let sum_exp = (-2.0_f64).exp() + (-1.0_f64).exp() + 1.0;
        let log_sum = sum_exp.ln();
        let lsm = [
            1.0 - 3.0 - log_sum,
            2.0 - 3.0 - log_sum,
            3.0 - 3.0 - log_sum,
        ];
        let expected = -lsm[2] + (-lsm[0]);

        assert!(
            (out.item().unwrap() - expected).abs() < 1e-6,
            "CE sum: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_cross_entropy_forward_none() {
        let logits = leaf_2d(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let targets = target_vec(&[2.0, 0.0]);
        let loss = CrossEntropyLoss::new(Reduction::None, 0.0);
        let out = loss.forward(&logits, &targets).unwrap();

        assert_eq!(out.shape(), &[2]);
        let d = out.data().unwrap();

        let sum_exp = (-2.0_f64).exp() + (-1.0_f64).exp() + 1.0;
        let log_sum = sum_exp.ln();
        let lsm = [
            1.0 - 3.0 - log_sum,
            2.0 - 3.0 - log_sum,
            3.0 - 3.0 - log_sum,
        ];

        assert!((d[0] - (-lsm[2])).abs() < 1e-6);
        assert!((d[1] - (-lsm[0])).abs() < 1e-6);
    }

    #[test]
    fn test_cross_entropy_backward_mean() {
        // Single sample for simpler gradient check.
        let logits = leaf_2d(&[1.0, 2.0, 3.0], &[1, 3]);
        let targets = target_vec(&[1.0]);
        let loss = CrossEntropyLoss::default();
        let out = loss.forward(&logits, &targets).unwrap();
        backward(&out).unwrap();

        let grad = logits.grad().unwrap().unwrap();
        let g = grad.data().unwrap();

        // softmax([1,2,3])
        let sum_exp = 1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp();
        let sm = [
            1.0_f64.exp() / sum_exp,
            2.0_f64.exp() / sum_exp,
            3.0_f64.exp() / sum_exp,
        ];
        // grad = (softmax - one_hot) / batch_size, target=1
        // batch_size = 1
        let expected = [sm[0] - 0.0, sm[1] - 1.0, sm[2] - 0.0];

        for i in 0..3 {
            assert!(
                (g[i] - expected[i]).abs() < 1e-6,
                "CE grad[{}]: expected {}, got {}",
                i,
                expected[i],
                g[i]
            );
        }
    }

    #[test]
    fn test_cross_entropy_label_smoothing() {
        let logits = leaf_2d(&[1.0, 2.0, 3.0], &[1, 3]);
        let targets = target_vec(&[2.0]);
        let ls = 0.1;
        let loss = CrossEntropyLoss::new(Reduction::Mean, ls);
        let out = loss.forward(&logits, &targets).unwrap();

        // Compute expected: max=3, shifted=[-2,-1,0]
        let max_val = 3.0_f64;
        let sum_exp = (-2.0_f64).exp() + (-1.0_f64).exp() + 1.0;
        let log_sum = sum_exp.ln();
        let lsm = [
            1.0 - max_val - log_sum,
            2.0 - max_val - log_sum,
            3.0 - max_val - log_sum,
        ];
        let nll = -lsm[2];
        let smooth = -(lsm[0] + lsm[1] + lsm[2]) / 3.0;
        let expected = (1.0 - ls) * nll + ls * smooth;

        assert!(
            (out.item().unwrap() - expected).abs() < 1e-6,
            "CE label smoothing: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_cross_entropy_large_logits_stability() {
        // Large logits should not produce NaN or Inf.
        let logits = leaf_2d(&[1000.0, 1001.0, 999.0], &[1, 3]);
        let targets = target_vec(&[1.0]);
        let loss = CrossEntropyLoss::default();
        let out = loss.forward(&logits, &targets).unwrap();
        let val = out.item().unwrap();
        assert!(
            val.is_finite(),
            "CE with large logits produced non-finite: {}",
            val
        );

        // The correct answer: softmax([1000,1001,999]) ~= [e^(-1), e^0, e^(-2)] / Z
        // log_softmax([1000,1001,999]) = [1000-1001-log(Z), 0-log(Z), 999-1001-log(Z)]
        // where Z = e^(-1) + 1 + e^(-2)
        let z = (-1.0_f64).exp() + 1.0 + (-2.0_f64).exp();
        let expected = -(1001.0 - 1001.0 - z.ln()); // nll for target 1
        assert!(
            (val - expected).abs() < 1e-5,
            "CE large logits: expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_cross_entropy_negative_logits_stability() {
        let logits = leaf_2d(&[-1000.0, -999.0, -1001.0], &[1, 3]);
        let targets = target_vec(&[1.0]);
        let loss = CrossEntropyLoss::default();
        let out = loss.forward(&logits, &targets).unwrap();
        let val = out.item().unwrap();
        assert!(
            val.is_finite(),
            "CE with large negative logits produced non-finite: {}",
            val
        );
    }

    // -----------------------------------------------------------------------
    // BCEWithLogitsLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_bce_forward_mean() {
        // x = [0, 0], y = [1, 0]
        // loss(0, 1) = max(0,0) - 0*1 + log(1+exp(0)) = 0 - 0 + log(2) = log(2)
        // loss(0, 0) = max(0,0) - 0*0 + log(1+exp(0)) = 0 - 0 + log(2) = log(2)
        // mean = log(2)
        let logits = leaf_vec(&[0.0, 0.0]);
        let targets = target_vec(&[1.0, 0.0]);
        let loss = BCEWithLogitsLoss::new(Reduction::Mean);
        let out = loss.forward(&logits, &targets).unwrap();
        let expected = 2.0_f64.ln();
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "BCE mean: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_bce_forward_sum() {
        let logits = leaf_vec(&[0.0, 0.0]);
        let targets = target_vec(&[1.0, 0.0]);
        let loss = BCEWithLogitsLoss::new(Reduction::Sum);
        let out = loss.forward(&logits, &targets).unwrap();
        let expected = 2.0 * 2.0_f64.ln();
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "BCE sum: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_bce_forward_none() {
        let logits = leaf_vec(&[0.0, 0.0]);
        let targets = target_vec(&[1.0, 0.0]);
        let loss = BCEWithLogitsLoss::new(Reduction::None);
        let out = loss.forward(&logits, &targets).unwrap();
        assert_eq!(out.shape(), &[2]);
        let d = out.data().unwrap();
        let ln2 = 2.0_f64.ln();
        assert!((d[0] - ln2).abs() < 1e-7);
        assert!((d[1] - ln2).abs() < 1e-7);
    }

    #[test]
    fn test_bce_backward_mean() {
        let logits = leaf_vec(&[0.0, 0.0]);
        let targets = target_vec(&[1.0, 0.0]);
        let loss = BCEWithLogitsLoss::new(Reduction::Mean);
        let out = loss.forward(&logits, &targets).unwrap();
        backward(&out).unwrap();

        let grad = logits.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad = (sigmoid(x) - y) / n
        // sigmoid(0) = 0.5
        // g[0] = (0.5 - 1.0) / 2 = -0.25
        // g[1] = (0.5 - 0.0) / 2 = 0.25
        assert!(
            (g[0] - (-0.25)).abs() < 1e-7,
            "BCE grad[0]: expected -0.25, got {}",
            g[0]
        );
        assert!(
            (g[1] - 0.25).abs() < 1e-7,
            "BCE grad[1]: expected 0.25, got {}",
            g[1]
        );
    }

    #[test]
    fn test_bce_numerical_stability_large_positive() {
        // Large positive logits should not overflow.
        let logits = leaf_vec(&[100.0]);
        let targets = target_vec(&[1.0]);
        let loss = BCEWithLogitsLoss::new(Reduction::Mean);
        let out = loss.forward(&logits, &targets).unwrap();
        let val = out.item().unwrap();
        assert!(
            val.is_finite(),
            "BCE large positive logit: non-finite {}",
            val
        );
        // loss = max(100,0) - 100*1 + log(1+exp(-100)) ~ 0 + ~0 = ~0
        assert!(
            val < 1e-10,
            "BCE large positive logit: expected ~0, got {}",
            val
        );
    }

    #[test]
    fn test_bce_numerical_stability_large_negative() {
        let logits = leaf_vec(&[-100.0]);
        let targets = target_vec(&[0.0]);
        let loss = BCEWithLogitsLoss::new(Reduction::Mean);
        let out = loss.forward(&logits, &targets).unwrap();
        let val = out.item().unwrap();
        assert!(
            val.is_finite(),
            "BCE large negative logit: non-finite {}",
            val
        );
        // loss = max(-100,0) - (-100)*0 + log(1+exp(-100)) ~ 0 + 0 + ~0
        assert!(
            val < 1e-10,
            "BCE large negative logit: expected ~0, got {}",
            val
        );
    }

    // -----------------------------------------------------------------------
    // HuberLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_huber_forward_quadratic_region() {
        // error = 0.3 (< delta=1.0), loss = 0.5 * 0.3^2 = 0.045
        let pred = leaf_vec(&[1.3]);
        let target = target_vec(&[1.0]);
        let loss = HuberLoss::default(); // delta = 1.0
        let out = loss.forward(&pred, &target).unwrap();
        assert!(
            (out.item().unwrap() - 0.045).abs() < 1e-7,
            "Huber quadratic: expected 0.045, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_huber_forward_linear_region() {
        // error = 2.0 (>= delta=1.0), loss = 1.0 * (2.0 - 0.5) = 1.5
        let pred = leaf_vec(&[3.0]);
        let target = target_vec(&[1.0]);
        let loss = HuberLoss::default();
        let out = loss.forward(&pred, &target).unwrap();
        assert!(
            (out.item().unwrap() - 1.5).abs() < 1e-7,
            "Huber linear: expected 1.5, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_huber_forward_sum() {
        let pred = leaf_vec(&[1.3, 3.0]);
        let target = target_vec(&[1.0, 1.0]);
        let loss = HuberLoss::new(Reduction::Sum, 1.0);
        let out = loss.forward(&pred, &target).unwrap();
        let expected = 0.045 + 1.5;
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "Huber sum: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_huber_forward_none() {
        let pred = leaf_vec(&[1.3, 3.0]);
        let target = target_vec(&[1.0, 1.0]);
        let loss = HuberLoss::new(Reduction::None, 1.0);
        let out = loss.forward(&pred, &target).unwrap();
        assert_eq!(out.shape(), &[2]);
        let d = out.data().unwrap();
        assert!((d[0] - 0.045).abs() < 1e-7);
        assert!((d[1] - 1.5).abs() < 1e-7);
    }

    #[test]
    fn test_huber_backward_quadratic() {
        // error = 0.3 (< delta=1.0), grad = error / n = 0.3 / 1 = 0.3
        let pred = leaf_vec(&[1.3]);
        let target = target_vec(&[1.0]);
        let loss = HuberLoss::default();
        let out = loss.forward(&pred, &target).unwrap();
        backward(&out).unwrap();

        let grad = pred.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // mean reduction with n=1: grad = error / n = 0.3
        assert!(
            (g[0] - 0.3).abs() < 1e-7,
            "Huber quadratic grad: expected 0.3, got {}",
            g[0]
        );
    }

    #[test]
    fn test_huber_backward_linear() {
        // error = 2.0 (>= delta=1.0), grad = delta * sign(error) / n = 1.0 / 1 = 1.0
        let pred = leaf_vec(&[3.0]);
        let target = target_vec(&[1.0]);
        let loss = HuberLoss::default();
        let out = loss.forward(&pred, &target).unwrap();
        backward(&out).unwrap();

        let grad = pred.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        assert!(
            (g[0] - 1.0).abs() < 1e-7,
            "Huber linear grad: expected 1.0, got {}",
            g[0]
        );
    }

    #[test]
    fn test_huber_backward_negative_error() {
        // error = -2.0, grad = delta * sign(-2.0) / n = -1.0
        let pred = leaf_vec(&[-1.0]);
        let target = target_vec(&[1.0]);
        let loss = HuberLoss::default();
        let out = loss.forward(&pred, &target).unwrap();
        backward(&out).unwrap();

        let grad = pred.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        assert!(
            (g[0] - (-1.0)).abs() < 1e-7,
            "Huber negative error grad: expected -1.0, got {}",
            g[0]
        );
    }

    #[test]
    fn test_huber_custom_delta() {
        // delta=0.5, error=0.3 (< 0.5): quadratic, loss = 0.5 * 0.09 = 0.045
        let pred = leaf_vec(&[1.3]);
        let target = target_vec(&[1.0]);
        let loss = HuberLoss::new(Reduction::Mean, 0.5);
        let out = loss.forward(&pred, &target).unwrap();
        assert!(
            (out.item().unwrap() - 0.045).abs() < 1e-7,
            "Huber custom delta quadratic: expected 0.045, got {}",
            out.item().unwrap()
        );

        // delta=0.5, error=1.0 (>= 0.5): linear, loss = 0.5 * (1.0 - 0.25) = 0.375
        let pred2 = leaf_vec(&[2.0]);
        let target2 = target_vec(&[1.0]);
        let out2 = loss.forward(&pred2, &target2).unwrap();
        assert!(
            (out2.item().unwrap() - 0.375).abs() < 1e-7,
            "Huber custom delta linear: expected 0.375, got {}",
            out2.item().unwrap()
        );
    }

    #[test]
    fn test_huber_zero_loss() {
        let pred = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.0, 2.0]);
        let loss = HuberLoss::default();
        let out = loss.forward(&pred, &target).unwrap();
        assert!(out.item().unwrap().abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // no_grad disables backward nodes
    // -----------------------------------------------------------------------

    #[test]
    fn test_mse_no_grad() {
        ferrotorch_core::no_grad(|| {
            let pred = leaf_vec(&[1.0, 2.0]);
            let target = target_vec(&[1.5, 2.5]);
            let loss = MSELoss::default();
            let out = loss.forward(&pred, &target).unwrap();
            assert!(
                out.grad_fn().is_none(),
                "MSELoss inside no_grad should not attach grad_fn"
            );
        });
    }

    #[test]
    fn test_ce_no_grad() {
        ferrotorch_core::no_grad(|| {
            let logits = leaf_2d(&[1.0, 2.0, 3.0], &[1, 3]);
            let targets = target_vec(&[0.0]);
            let loss = CrossEntropyLoss::default();
            let out = loss.forward(&logits, &targets).unwrap();
            assert!(
                out.grad_fn().is_none(),
                "CrossEntropyLoss inside no_grad should not attach grad_fn"
            );
        });
    }

    #[test]
    fn test_bce_no_grad() {
        ferrotorch_core::no_grad(|| {
            let logits = leaf_vec(&[0.0, 1.0]);
            let targets = target_vec(&[1.0, 0.0]);
            let loss = BCEWithLogitsLoss::default();
            let out = loss.forward(&logits, &targets).unwrap();
            assert!(
                out.grad_fn().is_none(),
                "BCEWithLogitsLoss inside no_grad should not attach grad_fn"
            );
        });
    }

    #[test]
    fn test_huber_no_grad() {
        ferrotorch_core::no_grad(|| {
            let pred = leaf_vec(&[1.0]);
            let target = target_vec(&[2.0]);
            let loss = HuberLoss::default();
            let out = loss.forward(&pred, &target).unwrap();
            assert!(
                out.grad_fn().is_none(),
                "HuberLoss inside no_grad should not attach grad_fn"
            );
        });
    }

    // -----------------------------------------------------------------------
    // Shape mismatch errors
    // -----------------------------------------------------------------------

    #[test]
    fn test_mse_shape_mismatch() {
        let pred = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.0, 2.0, 3.0]);
        let loss = MSELoss::default();
        assert!(loss.forward(&pred, &target).is_err());
    }

    #[test]
    fn test_bce_shape_mismatch() {
        let logits = leaf_vec(&[0.0]);
        let targets = target_vec(&[1.0, 0.0]);
        let loss = BCEWithLogitsLoss::default();
        assert!(loss.forward(&logits, &targets).is_err());
    }

    #[test]
    fn test_huber_shape_mismatch() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.0]);
        let loss = HuberLoss::default();
        assert!(loss.forward(&pred, &target).is_err());
    }

    #[test]
    fn test_ce_logits_wrong_dims() {
        // 1D logits should fail (needs 2D).
        let logits = leaf_vec(&[1.0, 2.0, 3.0]);
        let targets = target_vec(&[1.0]);
        let loss = CrossEntropyLoss::default();
        assert!(loss.forward(&logits, &targets).is_err());
    }

    #[test]
    fn test_ce_target_batch_mismatch() {
        let logits = leaf_2d(&[1.0, 2.0, 3.0], &[1, 3]);
        let targets = target_vec(&[0.0, 1.0]); // batch size 2, logits batch 1
        let loss = CrossEntropyLoss::default();
        assert!(loss.forward(&logits, &targets).is_err());
    }

    // -----------------------------------------------------------------------
    // KLDivLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_kl_div_forward_mean() {
        // input = log-probabilities, target = probabilities.
        // target = [0.25, 0.75], input = [ln(0.5), ln(0.5)]
        // KL = 0.25 * (ln(0.25) - ln(0.5)) + 0.75 * (ln(0.75) - ln(0.5))
        //    = 0.25 * ln(0.5) + 0.75 * ln(1.5)
        let input = leaf_vec(&[0.5_f64.ln(), 0.5_f64.ln()]);
        let target = target_vec(&[0.25, 0.75]);
        let loss = KLDivLoss::new(Reduction::Sum);
        let out = loss.forward(&input, &target).unwrap();

        let expected =
            0.25 * (0.25_f64.ln() - 0.5_f64.ln()) + 0.75 * (0.75_f64.ln() - 0.5_f64.ln());
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "KL sum: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_kl_div_zero_target_contributes_zero() {
        // target = [0, 1], input = [ln(0.5), ln(0.5)]
        // KL = 0 + 1 * (ln(1) - ln(0.5)) = 0 + ln(2)
        let input = leaf_vec(&[0.5_f64.ln(), 0.5_f64.ln()]);
        let target = target_vec(&[0.0, 1.0]);
        let loss = KLDivLoss::new(Reduction::Sum);
        let out = loss.forward(&input, &target).unwrap();
        let expected = 2.0_f64.ln();
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "KL zero target: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_kl_div_identical_distributions() {
        // When input = log(target), KL should be 0.
        let target = target_vec(&[0.3, 0.7]);
        let input = leaf_vec(&[0.3_f64.ln(), 0.7_f64.ln()]);
        let loss = KLDivLoss::new(Reduction::Sum);
        let out = loss.forward(&input, &target).unwrap();
        assert!(
            out.item().unwrap().abs() < 1e-10,
            "KL same dist: expected ~0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_kl_div_backward() {
        let input = leaf_vec(&[0.5_f64.ln(), 0.5_f64.ln()]);
        let target = target_vec(&[0.25, 0.75]);
        let loss = KLDivLoss::new(Reduction::Sum);
        let out = loss.forward(&input, &target).unwrap();
        backward(&out).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad = -target (for sum reduction)
        assert!(
            (g[0] - (-0.25)).abs() < 1e-7,
            "KL grad[0]: expected -0.25, got {}",
            g[0]
        );
        assert!(
            (g[1] - (-0.75)).abs() < 1e-7,
            "KL grad[1]: expected -0.75, got {}",
            g[1]
        );
    }

    #[test]
    fn test_kl_div_shape_mismatch() {
        let input = leaf_vec(&[0.0, 0.0]);
        let target = target_vec(&[0.5]);
        let loss = KLDivLoss::default();
        assert!(loss.forward(&input, &target).is_err());
    }

    // -----------------------------------------------------------------------
    // CosineEmbeddingLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_embedding_positive_pair() {
        // x1 = [1, 0], x2 = [0, 1], y = 1 (positive)
        // cos(x1, x2) = 0, loss = 1 - 0 = 1
        let x1 = leaf_2d(&[1.0, 0.0], &[1, 2]);
        let x2 = leaf_2d(&[0.0, 1.0], &[1, 2]);
        let y = target_vec(&[1.0]);
        let loss = CosineEmbeddingLoss::default();
        let out = loss.forward_pair(&x1, &x2, &y).unwrap();
        assert!(
            (out.item().unwrap() - 1.0).abs() < 1e-7,
            "cosine positive orthogonal: expected 1.0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_cosine_embedding_positive_identical() {
        // x1 = x2 = [1, 1], y = 1, cos = 1, loss = 0
        let x1 = leaf_2d(&[1.0, 1.0], &[1, 2]);
        let x2 = leaf_2d(&[1.0, 1.0], &[1, 2]);
        let y = target_vec(&[1.0]);
        let loss = CosineEmbeddingLoss::default();
        let out = loss.forward_pair(&x1, &x2, &y).unwrap();
        assert!(
            out.item().unwrap().abs() < 1e-7,
            "cosine positive identical: expected 0.0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_cosine_embedding_negative_pair() {
        // x1 = [1, 0], x2 = [1, 0], y = -1 (negative), margin = 0.5
        // cos = 1.0, loss = max(0, 1.0 - 0.5) = 0.5
        let x1 = leaf_2d(&[1.0, 0.0], &[1, 2]);
        let x2 = leaf_2d(&[1.0, 0.0], &[1, 2]);
        let y = target_vec(&[-1.0]);
        let loss = CosineEmbeddingLoss::new(Reduction::Mean, 0.5);
        let out = loss.forward_pair(&x1, &x2, &y).unwrap();
        assert!(
            (out.item().unwrap() - 0.5).abs() < 1e-7,
            "cosine negative same: expected 0.5, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_cosine_embedding_negative_orthogonal() {
        // x1 = [1, 0], x2 = [0, 1], y = -1, margin = 0.0
        // cos = 0.0, loss = max(0, 0.0 - 0.0) = 0
        let x1 = leaf_2d(&[1.0, 0.0], &[1, 2]);
        let x2 = leaf_2d(&[0.0, 1.0], &[1, 2]);
        let y = target_vec(&[-1.0]);
        let loss = CosineEmbeddingLoss::new(Reduction::Mean, 0.0);
        let out = loss.forward_pair(&x1, &x2, &y).unwrap();
        assert!(
            out.item().unwrap().abs() < 1e-7,
            "cosine negative orthogonal: expected 0.0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_cosine_embedding_shape_mismatch() {
        let x1 = leaf_2d(&[1.0, 0.0], &[1, 2]);
        let x2 = leaf_2d(&[1.0, 0.0, 0.0], &[1, 3]);
        let y = target_vec(&[1.0]);
        let loss = CosineEmbeddingLoss::default();
        assert!(loss.forward_pair(&x1, &x2, &y).is_err());
    }

    // -----------------------------------------------------------------------
    // SmoothL1Loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_smooth_l1_forward_quadratic() {
        // error = 0.3 (< 1.0), loss = 0.5 * 0.09 = 0.045
        let pred = leaf_vec(&[1.3]);
        let target = target_vec(&[1.0]);
        let loss = SmoothL1Loss::default();
        let out = loss.forward(&pred, &target).unwrap();
        assert!(
            (out.item().unwrap() - 0.045).abs() < 1e-7,
            "SmoothL1 quadratic: expected 0.045, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_smooth_l1_forward_linear() {
        // error = 2.0 (>= 1.0), loss = 1.0 * (2.0 - 0.5) = 1.5
        let pred = leaf_vec(&[3.0]);
        let target = target_vec(&[1.0]);
        let loss = SmoothL1Loss::default();
        let out = loss.forward(&pred, &target).unwrap();
        assert!(
            (out.item().unwrap() - 1.5).abs() < 1e-7,
            "SmoothL1 linear: expected 1.5, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_smooth_l1_matches_huber() {
        // SmoothL1Loss should produce identical results to HuberLoss(delta=1.0).
        let pred = leaf_vec(&[0.5, 2.0, -1.0]);
        let target = target_vec(&[1.0, 0.0, 0.5]);

        let smooth = SmoothL1Loss::new(Reduction::Sum);
        let huber = HuberLoss::new(Reduction::Sum, 1.0);

        let s_out = smooth.forward(&pred, &target).unwrap();
        let h_out = huber.forward(&pred, &target).unwrap();

        assert!(
            (s_out.item().unwrap() - h_out.item().unwrap()).abs() < 1e-10,
            "SmoothL1 and Huber(1.0) diverge: {} vs {}",
            s_out.item().unwrap(),
            h_out.item().unwrap()
        );
    }

    #[test]
    fn test_smooth_l1_zero_loss() {
        let pred = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.0, 2.0]);
        let loss = SmoothL1Loss::default();
        let out = loss.forward(&pred, &target).unwrap();
        assert!(out.item().unwrap().abs() < 1e-10);
    }

    // -------------------------------------------------------------------
    // autocast_guard integration: loss forwards fire the guard
    // -------------------------------------------------------------------

    #[test]
    fn test_mse_loss_fires_autocast_guard_when_enabled() {
        use ferrotorch_core::autograd::autocast::{AutocastDtype, autocast, set_autocast_debug};
        use ferrotorch_core::autograd::autocast_ops::{AutocastCategory, drain_autocast_events};
        set_autocast_debug(true);

        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);

        // Outside autocast: no events.
        drain_autocast_events();
        let _ = MSELoss::new(Reduction::Mean)
            .forward(&pred, &target)
            .unwrap();
        assert!(drain_autocast_events().is_empty());

        // Inside autocast: records "mse_loss" as FullPrecision.
        autocast(AutocastDtype::F16, || {
            drain_autocast_events();
            let _ = MSELoss::new(Reduction::Mean)
                .forward(&pred, &target)
                .unwrap();
            let events = drain_autocast_events();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].op, "mse_loss");
            assert_eq!(events[0].category, AutocastCategory::FullPrecision);
        });
    }

    #[test]
    fn test_cross_entropy_fires_autocast_guard_when_enabled() {
        use ferrotorch_core::autograd::autocast::{AutocastDtype, autocast, set_autocast_debug};
        use ferrotorch_core::autograd::autocast_ops::{AutocastCategory, drain_autocast_events};
        set_autocast_debug(true);

        // 2 samples, 3 classes.
        let logits = leaf_2d(&[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3]);
        let targets = target_vec(&[2.0, 0.0]); // class indices

        // Outside autocast: no events.
        drain_autocast_events();
        let _ = CrossEntropyLoss::new(Reduction::Mean, 0.0)
            .forward(&logits, &targets)
            .unwrap();
        assert!(drain_autocast_events().is_empty());

        // Inside autocast: records "cross_entropy" as FullPrecision.
        autocast(AutocastDtype::BF16, || {
            drain_autocast_events();
            let _ = CrossEntropyLoss::new(Reduction::Mean, 0.0)
                .forward(&logits, &targets)
                .unwrap();
            let events = drain_autocast_events();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].op, "cross_entropy");
            assert_eq!(events[0].category, AutocastCategory::FullPrecision);
        });
    }

    #[test]
    fn test_bce_with_logits_fires_autocast_guard_when_enabled() {
        use ferrotorch_core::autograd::autocast::{AutocastDtype, autocast, set_autocast_debug};
        use ferrotorch_core::autograd::autocast_ops::{AutocastCategory, drain_autocast_events};
        set_autocast_debug(true);

        let logits = leaf_vec(&[0.5, -0.5, 1.0]);
        let targets = target_vec(&[1.0, 0.0, 1.0]);

        // Outside autocast: no events.
        drain_autocast_events();
        let _ = BCEWithLogitsLoss::new(Reduction::Mean)
            .forward(&logits, &targets)
            .unwrap();
        assert!(drain_autocast_events().is_empty());

        // Inside autocast: records "bce_with_logits" as FullPrecision.
        autocast(AutocastDtype::F16, || {
            drain_autocast_events();
            let _ = BCEWithLogitsLoss::new(Reduction::Mean)
                .forward(&logits, &targets)
                .unwrap();
            let events = drain_autocast_events();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].op, "bce_with_logits");
            assert_eq!(events[0].category, AutocastCategory::FullPrecision);
        });
    }

    // -----------------------------------------------------------------------
    // L1Loss
    // -----------------------------------------------------------------------

    #[test]
    fn test_l1_forward_mean() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);
        let loss = L1Loss::new(Reduction::Mean);
        let out = loss.forward(&pred, &target).unwrap();
        // Each |diff| is 0.5, mean is 0.5.
        assert!(out.is_scalar());
        assert!(
            (out.item().unwrap() - 0.5).abs() < 1e-7,
            "L1 mean: expected 0.5, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_l1_forward_sum() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);
        let loss = L1Loss::new(Reduction::Sum);
        let out = loss.forward(&pred, &target).unwrap();
        // sum of 0.5 * 3 = 1.5
        assert!(
            (out.item().unwrap() - 1.5).abs() < 1e-7,
            "L1 sum: expected 1.5, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_l1_forward_none() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);
        let loss = L1Loss::new(Reduction::None);
        let out = loss.forward(&pred, &target).unwrap();
        assert_eq!(out.shape(), &[3]);
        let d = out.data().unwrap();
        for i in 0..3 {
            assert!(
                (d[i] - 0.5).abs() < 1e-7,
                "L1 none[{}]: expected 0.5, got {}",
                i,
                d[i]
            );
        }
    }

    #[test]
    fn test_l1_backward_mean() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.5, 2.5, 3.5]);
        let loss = L1Loss::new(Reduction::Mean);
        let out = loss.forward(&pred, &target).unwrap();
        backward(&out).unwrap();

        let grad = pred.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad = sign(pred - target) / n = sign(-0.5) / 3 = -1/3
        let expected = -1.0 / 3.0;
        for i in 0..3 {
            assert!(
                (g[i] - expected).abs() < 1e-7,
                "L1 backward mean[{}]: expected {}, got {}",
                i,
                expected,
                g[i]
            );
        }
    }

    #[test]
    fn test_l1_backward_sum() {
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[0.5, 1.5, 2.5]);
        let loss = L1Loss::new(Reduction::Sum);
        let out = loss.forward(&pred, &target).unwrap();
        backward(&out).unwrap();

        let grad = pred.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad = sign(pred - target) * 1 = sign(0.5) = 1
        for i in 0..3 {
            assert!(
                (g[i] - 1.0).abs() < 1e-7,
                "L1 backward sum[{}]: expected 1.0, got {}",
                i,
                g[i]
            );
        }
    }

    #[test]
    fn test_l1_shape_mismatch() {
        let pred = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.0, 2.0, 3.0]);
        let loss = L1Loss::default();
        assert!(loss.forward(&pred, &target).is_err());
    }

    #[test]
    fn test_l1_zero_diff() {
        // When pred == target, loss = 0 and sign(0) = 0.
        let pred = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.0, 2.0, 3.0]);
        let loss = L1Loss::new(Reduction::Mean);
        let out = loss.forward(&pred, &target).unwrap();
        assert!((out.item().unwrap()).abs() < 1e-10);

        backward(&out).unwrap();
        let grad = pred.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        for i in 0..3 {
            assert!(
                g[i].abs() < 1e-10,
                "L1 zero diff grad[{}] should be 0, got {}",
                i,
                g[i]
            );
        }
    }

    #[test]
    fn test_l1_mixed_signs() {
        // Pred above and below target.
        let pred = leaf_vec(&[3.0, 1.0]);
        let target = target_vec(&[1.0, 3.0]);
        let loss = L1Loss::new(Reduction::Mean);
        let out = loss.forward(&pred, &target).unwrap();
        // |3-1| = 2, |1-3| = 2, mean = 2
        assert!(
            (out.item().unwrap() - 2.0).abs() < 1e-7,
            "L1 mixed: expected 2.0, got {}",
            out.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // NLLLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_nll_forward_mean() {
        // log_probs: [2, 3], targets: [2]
        // Manually compute: sample 0 target=1, sample 1 target=0
        let log_probs = leaf_2d(&[-1.5, -0.5, -2.0, -0.8, -1.2, -1.0], &[2, 3]);
        let targets = target_vec(&[1.0, 0.0]);
        let loss = NLLLoss::default();
        let out = loss.forward(&log_probs, &targets).unwrap();
        // loss = -(-0.5 + -0.8) / 2 = (0.5 + 0.8) / 2 = 0.65
        assert!(out.is_scalar());
        assert!(
            (out.item().unwrap() - 0.65).abs() < 1e-7,
            "NLL mean: expected 0.65, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_nll_forward_sum() {
        let log_probs = leaf_2d(&[-1.5, -0.5, -2.0, -0.8, -1.2, -1.0], &[2, 3]);
        let targets = target_vec(&[1.0, 0.0]);
        let loss = NLLLoss::new(Reduction::Sum, None);
        let out = loss.forward(&log_probs, &targets).unwrap();
        // loss = 0.5 + 0.8 = 1.3
        assert!(
            (out.item().unwrap() - 1.3).abs() < 1e-7,
            "NLL sum: expected 1.3, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_nll_forward_none() {
        let log_probs = leaf_2d(&[-1.5, -0.5, -2.0, -0.8, -1.2, -1.0], &[2, 3]);
        let targets = target_vec(&[1.0, 0.0]);
        let loss = NLLLoss::new(Reduction::None, None);
        let out = loss.forward(&log_probs, &targets).unwrap();
        assert_eq!(out.shape(), &[2]);
        let d = out.data().unwrap();
        assert!(
            (d[0] - 0.5).abs() < 1e-7,
            "NLL none[0]: expected 0.5, got {}",
            d[0]
        );
        assert!(
            (d[1] - 0.8).abs() < 1e-7,
            "NLL none[1]: expected 0.8, got {}",
            d[1]
        );
    }

    #[test]
    fn test_nll_backward_mean() {
        let log_probs = leaf_2d(&[-1.5, -0.5, -2.0, -0.8, -1.2, -1.0], &[2, 3]);
        let targets = target_vec(&[1.0, 0.0]);
        let loss = NLLLoss::default();
        let out = loss.forward(&log_probs, &targets).unwrap();
        backward(&out).unwrap();

        let grad = log_probs.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad[0, 1] = -1/2, grad[1, 0] = -1/2, rest = 0
        let expected = [0.0, -0.5, 0.0, -0.5, 0.0, 0.0];
        for i in 0..6 {
            assert!(
                (g[i] - expected[i]).abs() < 1e-7,
                "NLL backward mean[{}]: expected {}, got {}",
                i,
                expected[i],
                g[i]
            );
        }
    }

    #[test]
    fn test_nll_ignore_index() {
        let log_probs = leaf_2d(&[-1.5, -0.5, -2.0, -0.8, -1.2, -1.0], &[2, 3]);
        let targets = target_vec(&[1.0, 0.0]);
        // Ignore class 0 — only sample 0 (target=1) contributes.
        let loss = NLLLoss::new(Reduction::Mean, Some(0));
        let out = loss.forward(&log_probs, &targets).unwrap();
        // loss = -(-0.5) / 1 = 0.5
        assert!(
            (out.item().unwrap() - 0.5).abs() < 1e-7,
            "NLL ignore_index mean: expected 0.5, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_nll_ignore_index_all_ignored() {
        let log_probs = leaf_2d(&[-1.5, -0.5, -0.8, -1.2], &[2, 2]);
        let targets = target_vec(&[0.0, 0.0]);
        let loss = NLLLoss::new(Reduction::Mean, Some(0));
        let out = loss.forward(&log_probs, &targets).unwrap();
        // All ignored => 0.
        assert!(
            (out.item().unwrap()).abs() < 1e-10,
            "NLL all ignored: expected 0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_nll_wrong_log_probs_shape() {
        // 1-D input should error.
        let log_probs = leaf_vec(&[-0.5, -1.0, -1.5]);
        let targets = target_vec(&[1.0]);
        let loss = NLLLoss::default();
        assert!(loss.forward(&log_probs, &targets).is_err());
    }

    #[test]
    fn test_nll_target_shape_mismatch() {
        let log_probs = leaf_2d(&[-0.5, -1.0, -1.5, -0.8, -1.2, -1.0], &[2, 3]);
        let targets = target_vec(&[1.0, 0.0, 2.0]);
        let loss = NLLLoss::default();
        assert!(loss.forward(&log_probs, &targets).is_err());
    }

    #[test]
    fn test_nll_target_out_of_range() {
        let log_probs = leaf_2d(&[-0.5, -1.0], &[1, 2]);
        let targets = target_vec(&[5.0]); // Only 2 classes.
        let loss = NLLLoss::default();
        assert!(loss.forward(&log_probs, &targets).is_err());
    }

    #[test]
    fn test_nll_empty_batch() {
        let log_probs = leaf_2d(&[], &[0, 3]);
        let targets = target_vec(&[]);
        let loss = NLLLoss::default();
        let out = loss.forward(&log_probs, &targets).unwrap();
        assert!((out.item().unwrap()).abs() < 1e-10);
    }

    #[test]
    fn test_nll_backward_sum() {
        let log_probs = leaf_2d(&[-1.0, -2.0, -3.0, -4.0], &[2, 2]);
        let targets = target_vec(&[0.0, 1.0]);
        let loss = NLLLoss::new(Reduction::Sum, None);
        let out = loss.forward(&log_probs, &targets).unwrap();
        backward(&out).unwrap();

        let grad = log_probs.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad[0, 0] = -1, grad[1, 1] = -1, rest = 0
        let expected = [-1.0, 0.0, 0.0, -1.0];
        for i in 0..4 {
            assert!(
                (g[i] - expected[i]).abs() < 1e-7,
                "NLL backward sum[{}]: expected {}, got {}",
                i,
                expected[i],
                g[i]
            );
        }
    }

    #[test]
    fn test_nll_backward_with_ignore() {
        let log_probs = leaf_2d(&[-1.0, -2.0, -3.0, -4.0], &[2, 2]);
        let targets = target_vec(&[0.0, 1.0]);
        // Ignore target=0, so only sample 1 (target=1) contributes.
        let loss = NLLLoss::new(Reduction::Mean, Some(0));
        let out = loss.forward(&log_probs, &targets).unwrap();
        backward(&out).unwrap();

        let grad = log_probs.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // Only sample 1, target=1: grad[1,1] = -1/1 = -1, rest = 0.
        let expected = [0.0, 0.0, 0.0, -1.0];
        for i in 0..4 {
            assert!(
                (g[i] - expected[i]).abs() < 1e-7,
                "NLL backward ignore[{}]: expected {}, got {}",
                i,
                expected[i],
                g[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // BCELoss (probability input, not logits)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bce_loss_forward_mean() {
        // input = sigmoid output (probabilities), target = binary labels
        let input = leaf_vec(&[0.8, 0.4, 0.6]);
        let target = target_vec(&[1.0, 0.0, 1.0]);
        let loss = BCELoss::new(Reduction::Mean);
        let out = loss.forward(&input, &target).unwrap();
        // Per-element:
        //   [0]: -(1*ln(0.8) + 0*ln(0.2)) = -ln(0.8)
        //   [1]: -(0*ln(0.4) + 1*ln(0.6)) = -ln(0.6)
        //   [2]: -(1*ln(0.6) + 0*ln(0.4)) = -ln(0.6)
        let expected = (-0.8_f64.ln() + -0.6_f64.ln() + -0.6_f64.ln()) / 3.0;
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "BCELoss mean: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_bce_loss_forward_sum() {
        let input = leaf_vec(&[0.8, 0.4]);
        let target = target_vec(&[1.0, 0.0]);
        let loss = BCELoss::new(Reduction::Sum);
        let out = loss.forward(&input, &target).unwrap();
        let expected = -0.8_f64.ln() + -(0.6_f64).ln();
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "BCELoss sum: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_bce_loss_forward_none() {
        let input = leaf_vec(&[0.8, 0.4]);
        let target = target_vec(&[1.0, 0.0]);
        let loss = BCELoss::new(Reduction::None);
        let out = loss.forward(&input, &target).unwrap();
        assert_eq!(out.shape(), &[2]);
        let d = out.data().unwrap();
        assert!(
            (d[0] - (-0.8_f64.ln())).abs() < 1e-7,
            "BCELoss none[0]: expected {}, got {}",
            -0.8_f64.ln(),
            d[0]
        );
        assert!(
            (d[1] - (-0.6_f64.ln())).abs() < 1e-7,
            "BCELoss none[1]: expected {}, got {}",
            -0.6_f64.ln(),
            d[1]
        );
    }

    #[test]
    fn test_bce_loss_backward_mean() {
        let input = leaf_vec(&[0.8, 0.4]);
        let target = target_vec(&[1.0, 0.0]);
        let loss = BCELoss::new(Reduction::Mean);
        let out = loss.forward(&input, &target).unwrap();
        backward(&out).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad[0] = (-1/0.8 + 0/0.2) / 2 = -1.25 / 2 = -0.625
        // grad[1] = (0/0.4 + 1/0.6) / 2 = 1.6667 / 2 = 0.8333
        assert!(
            (g[0] - (-0.625)).abs() < 1e-5,
            "BCELoss backward[0]: expected -0.625, got {}",
            g[0]
        );
        let exp1 = 1.0 / 0.6 / 2.0;
        assert!(
            (g[1] - exp1).abs() < 1e-5,
            "BCELoss backward[1]: expected {}, got {}",
            exp1,
            g[1]
        );
    }

    #[test]
    fn test_bce_loss_shape_mismatch() {
        let input = leaf_vec(&[0.5, 0.5]);
        let target = target_vec(&[1.0]);
        let loss = BCELoss::default();
        assert!(loss.forward(&input, &target).is_err());
    }

    // -----------------------------------------------------------------------
    // TripletMarginLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_triplet_margin_forward_mean() {
        // anchor=[0, 0], positive=[1, 0], negative=[3, 0]
        // d_pos = 1.0, d_neg = 3.0, margin=1.0
        // loss = max(0, 1 - 3 + 1) = max(0, -1) = 0
        let anchor = leaf_2d(&[0.0, 0.0], &[1, 2]);
        let positive =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0]), vec![1, 2], false).unwrap();
        let negative =
            Tensor::from_storage(TensorStorage::cpu(vec![3.0, 0.0]), vec![1, 2], false).unwrap();
        let loss = TripletMarginLoss::default();
        let out = loss.forward(&anchor, &positive, &negative).unwrap();
        assert!(
            out.item().unwrap().abs() < 1e-7,
            "Triplet loss should be 0 when negative is far, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_triplet_margin_forward_active() {
        // anchor=[0, 0], positive=[2, 0], negative=[1, 0]
        // d_pos = 2.0, d_neg = 1.0, margin=1.0
        // loss = max(0, 2 - 1 + 1) = 2.0
        let anchor = leaf_2d(&[0.0, 0.0], &[1, 2]);
        let positive =
            Tensor::from_storage(TensorStorage::cpu(vec![2.0, 0.0]), vec![1, 2], false).unwrap();
        let negative =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0]), vec![1, 2], false).unwrap();
        let loss = TripletMarginLoss::default();
        let out = loss.forward(&anchor, &positive, &negative).unwrap();
        assert!(
            (out.item().unwrap() - 2.0).abs() < 1e-7,
            "Triplet loss: expected 2.0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_triplet_margin_batch() {
        // batch of 2: first triplet active, second not
        let anchor = leaf_2d(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
        let positive = Tensor::from_storage(
            TensorStorage::cpu(vec![2.0, 0.0, 1.0, 0.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let negative = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 0.0, 5.0, 0.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let loss = TripletMarginLoss::new(Reduction::Mean, 1.0, 2.0);
        let out = loss.forward(&anchor, &positive, &negative).unwrap();
        // Sample 0: d_pos=2, d_neg=1, loss=max(0, 2-1+1)=2
        // Sample 1: d_pos=1, d_neg=5, loss=max(0, 1-5+1)=0
        // Mean = 1.0
        assert!(
            (out.item().unwrap() - 1.0).abs() < 1e-7,
            "Triplet batch mean: expected 1.0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_triplet_margin_shape_mismatch() {
        let anchor = leaf_2d(&[0.0, 0.0], &[1, 2]);
        let positive =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0, 0.0]), vec![1, 3], false)
                .unwrap();
        let negative =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0]), vec![1, 2], false).unwrap();
        let loss = TripletMarginLoss::default();
        assert!(loss.forward(&anchor, &positive, &negative).is_err());
    }

    // -----------------------------------------------------------------------
    // MarginRankingLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_margin_ranking_forward_mean() {
        // x1=[1.0, -1.0], x2=[0.0, 0.0], y=[1.0, -1.0], margin=0.0
        // sample 0: max(0, -1*(1-0) + 0) = max(0, -1) = 0
        // sample 1: max(0, 1*(-1-0) + 0) = max(0, -1) = 0
        // BUT with margin=1:
        // sample 0: max(0, -1*(1-0) + 1) = 0
        // sample 1: max(0, 1*(-1-0) + 1) = 0
        let x1 = leaf_vec(&[2.0, 0.5]);
        let x2 = target_vec(&[1.0, 1.0]);
        let y = target_vec(&[1.0, -1.0]);
        let loss = MarginRankingLoss::new(Reduction::Mean, 0.0);
        let out = loss.forward(&x1, &x2, &y).unwrap();
        // sample 0: max(0, -1*(2-1)+0) = max(0, -1) = 0
        // sample 1: max(0, 1*(0.5-1)+0) = max(0, -0.5) = 0
        assert!(
            out.item().unwrap().abs() < 1e-7,
            "MarginRanking: expected 0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_margin_ranking_forward_with_margin() {
        let x1 = leaf_vec(&[1.0, 0.5]);
        let x2 = target_vec(&[0.5, 1.0]);
        let y = target_vec(&[1.0, 1.0]);
        let loss = MarginRankingLoss::new(Reduction::Mean, 1.0);
        let out = loss.forward(&x1, &x2, &y).unwrap();
        // sample 0: max(0, -1*(1-0.5)+1) = max(0, 0.5) = 0.5
        // sample 1: max(0, -1*(0.5-1)+1) = max(0, 1.5) = 1.5
        // mean = 1.0
        assert!(
            (out.item().unwrap() - 1.0).abs() < 1e-7,
            "MarginRanking: expected 1.0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_margin_ranking_backward() {
        let x1 = leaf_vec(&[0.5]);
        let x2 = target_vec(&[1.0]);
        let y = target_vec(&[1.0]);
        let loss = MarginRankingLoss::new(Reduction::Mean, 1.0);
        // max(0, -1*(0.5-1)+1) = max(0, 1.5) = 1.5
        let out = loss.forward(&x1, &x2, &y).unwrap();
        backward(&out).unwrap();

        let grad = x1.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // Hinge is active, grad = -y / n = -1.0 / 1 = -1.0
        assert!(
            (g[0] - (-1.0)).abs() < 1e-7,
            "MarginRanking backward: expected -1, got {}",
            g[0]
        );
    }

    #[test]
    fn test_margin_ranking_shape_mismatch() {
        let x1 = leaf_vec(&[1.0, 2.0]);
        let x2 = target_vec(&[1.0]);
        let y = target_vec(&[1.0, -1.0]);
        let loss = MarginRankingLoss::default();
        assert!(loss.forward(&x1, &x2, &y).is_err());
    }

    // -----------------------------------------------------------------------
    // CTCLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_ctc_simple() {
        // T=3, B=1, C=3 (blank=0, labels 1,2)
        // Perfect alignment: [blank, 1, 2] for target [1, 2]
        // log_probs that heavily favor the correct alignment
        let mut lp = vec![-10.0_f64; 3 * 3]; // [T=3, B=1, C=3]
        // Shape [T, B=1, C=3]; row-major stride = (3, 3, 1).
        let idx = |t: usize, c: usize| t * 3 + c;
        // t=0: blank (class 0) is likely
        lp[idx(0, 0)] = -0.1;
        lp[idx(0, 1)] = -10.0;
        lp[idx(0, 2)] = -10.0;
        // t=1: class 1 is likely
        lp[idx(1, 0)] = -10.0;
        lp[idx(1, 1)] = -0.1;
        lp[idx(1, 2)] = -10.0;
        // t=2: class 2 is likely
        lp[idx(2, 0)] = -10.0;
        lp[idx(2, 1)] = -10.0;
        lp[idx(2, 2)] = -0.1;

        let log_probs = Tensor::from_storage(TensorStorage::cpu(lp), vec![3, 1, 3], false).unwrap();
        let targets = target_vec(&[1.0, 2.0]);
        let loss = CTCLoss::default();
        let out = loss.forward(&log_probs, &targets, &[3], &[2]).unwrap();
        // Loss should be close to 0.3 (sum of -(-0.1)*3 paths, dominated by direct path)
        assert!(
            out.item().unwrap() < 1.0,
            "CTC: expected low loss for aligned input, got {}",
            out.item().unwrap()
        );
        assert!(
            out.item().unwrap() >= 0.0,
            "CTC: loss should be non-negative, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_ctc_empty_target() {
        // Empty target: loss = -sum(log_prob_blank over time)
        let lp = vec![-0.5_f64, -10.0, -10.0, -0.3, -10.0, -10.0]; // [T=2, B=1, C=3]
        let log_probs = Tensor::from_storage(TensorStorage::cpu(lp), vec![2, 1, 3], false).unwrap();
        let targets = target_vec(&[]);
        let loss = CTCLoss::new(Reduction::Mean, 0, false);
        let out = loss.forward(&log_probs, &targets, &[2], &[0]).unwrap();
        // loss = -(-0.5 + -0.3) = 0.8
        assert!(
            (out.item().unwrap() - 0.8).abs() < 1e-7,
            "CTC empty target: expected 0.8, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_ctc_wrong_shape() {
        let log_probs = leaf_vec(&[0.0, 0.0, 0.0]); // 1-D, not 3-D
        let targets = target_vec(&[1.0]);
        let loss = CTCLoss::default();
        assert!(loss.forward(&log_probs, &targets, &[3], &[1]).is_err());
    }

    // -----------------------------------------------------------------------
    // PoissonNLLLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_poisson_nll_forward_log_input() {
        let input = leaf_vec(&[0.0, 1.0, 2.0]);
        let target = target_vec(&[1.0, 2.0, 3.0]);
        let loss = PoissonNLLLoss::default(); // log_input=true
        let out = loss.forward(&input, &target).unwrap();
        // loss[i] = exp(x) - y * x
        let e0 = 0.0_f64.exp() - 1.0 * 0.0; // 1 - 0 = 1.0
        let e1 = 1.0_f64.exp() - 2.0 * 1.0; // e - 2 ≈ 0.718
        let e2 = 2.0_f64.exp() - 3.0 * 2.0; // e^2 - 6 ≈ 1.389
        let expected = (e0 + e1 + e2) / 3.0;
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-6,
            "Poisson NLL: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_poisson_nll_forward_no_log_input() {
        let input = leaf_vec(&[1.0, 2.0, 3.0]);
        let target = target_vec(&[1.0, 2.0, 3.0]);
        let eps = 1e-8;
        let loss = PoissonNLLLoss::new(Reduction::Mean, false, eps);
        let out = loss.forward(&input, &target).unwrap();
        // loss[i] = x - y * log(x + eps)
        let e0 = 1.0 - 1.0 * (1.0 + eps).ln();
        let e1 = 2.0 - 2.0 * (2.0 + eps).ln();
        let e2 = 3.0 - 3.0 * (3.0 + eps).ln();
        let expected = (e0 + e1 + e2) / 3.0;
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-6,
            "Poisson NLL no_log: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_poisson_nll_backward() {
        let input = leaf_vec(&[1.0]);
        let target = target_vec(&[2.0]);
        let loss = PoissonNLLLoss::default();
        let out = loss.forward(&input, &target).unwrap();
        backward(&out).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // d/dx(exp(x) - 2x) = exp(x) - 2, at x=1: e - 2
        let expected = 1.0_f64.exp() - 2.0;
        assert!(
            (g[0] - expected).abs() < 1e-6,
            "Poisson backward: expected {}, got {}",
            expected,
            g[0]
        );
    }

    #[test]
    fn test_poisson_nll_shape_mismatch() {
        let input = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.0]);
        let loss = PoissonNLLLoss::default();
        assert!(loss.forward(&input, &target).is_err());
    }

    // -----------------------------------------------------------------------
    // MultiMarginLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_margin_forward_mean() {
        // B=1, C=3, target=1
        // input = [1, 3, 2], target = 1
        // loss = (1/3) * sum_{j!=1} max(0, 1 - x[1] + x[j])
        //      = (1/3) * (max(0, 1-3+1) + max(0, 1-3+2))
        //      = (1/3) * (0 + 0) = 0 (correct class has highest margin)
        let input = leaf_2d(&[1.0, 3.0, 2.0], &[1, 3]);
        let target = target_vec(&[1.0]);
        let loss = MultiMarginLoss::default();
        let out = loss.forward(&input, &target).unwrap();
        assert!(
            out.item().unwrap().abs() < 1e-7,
            "MultiMargin: expected 0, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_multi_margin_forward_active() {
        // input = [2, 1, 3], target = 1
        // loss = (1/3) * (max(0, 1-1+2) + max(0, 1-1+3))
        //      = (1/3) * (2 + 3) = 5/3
        let input = leaf_2d(&[2.0, 1.0, 3.0], &[1, 3]);
        let target = target_vec(&[1.0]);
        let loss = MultiMarginLoss::default();
        let out = loss.forward(&input, &target).unwrap();
        let expected = 5.0 / 3.0;
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "MultiMargin active: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_multi_margin_p2() {
        // input = [2, 1, 3], target = 1, p=2
        // loss = (1/3) * (max(0, 1-1+2)^2 + max(0, 1-1+3)^2)
        //      = (1/3) * (4 + 9) = 13/3
        let input = leaf_2d(&[2.0, 1.0, 3.0], &[1, 3]);
        let target = target_vec(&[1.0]);
        let loss = MultiMarginLoss::new(Reduction::Mean, 2, 1.0);
        let out = loss.forward(&input, &target).unwrap();
        let expected = 13.0 / 3.0;
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "MultiMargin p=2: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_multi_margin_backward() {
        let input = leaf_2d(&[2.0, 1.0, 3.0], &[1, 3]);
        let target = target_vec(&[1.0]);
        let loss = MultiMarginLoss::new(Reduction::Sum, 1, 1.0);
        let out = loss.forward(&input, &target).unwrap();
        backward(&out).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // p=1: d/dx[j] for j!=y where hinge is active: 1/C
        // d/dx[y] = -sum of active / C
        // j=0: hinge = max(0, 1-1+2)=2 > 0, active => grad[0] = 1/3
        // j=2: hinge = max(0, 1-1+3)=3 > 0, active => grad[2] = 1/3
        // grad[1] (target) = -(1/3 + 1/3) = -2/3
        assert!(
            (g[0] - 1.0 / 3.0).abs() < 1e-7,
            "MultiMargin grad[0]: expected 1/3, got {}",
            g[0]
        );
        assert!(
            (g[1] - (-2.0 / 3.0)).abs() < 1e-7,
            "MultiMargin grad[1]: expected -2/3, got {}",
            g[1]
        );
        assert!(
            (g[2] - 1.0 / 3.0).abs() < 1e-7,
            "MultiMargin grad[2]: expected 1/3, got {}",
            g[2]
        );
    }

    #[test]
    fn test_multi_margin_wrong_shape() {
        let input = leaf_vec(&[1.0, 2.0, 3.0]); // 1-D, not 2-D
        let target = target_vec(&[1.0]);
        let loss = MultiMarginLoss::default();
        assert!(loss.forward(&input, &target).is_err());
    }

    // -----------------------------------------------------------------------
    // MultiLabelSoftMarginLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_label_soft_margin_forward() {
        // B=1, C=2, input = [2.0, -1.0], target = [1.0, 0.0]
        // Per-class BCE-with-logits:
        //   c=0: max(2,0) - 2*1 + log(1+exp(-2)) = 2 - 2 + log(1+exp(-2)) = log(1+exp(-2))
        //   c=1: max(0,0) - (-1)*0 + log(1+exp(-1)) = log(1+exp(-1))
        // loss = (bce0 + bce1) / C = (bce0 + bce1) / 2
        let input = leaf_2d(&[2.0, -1.0], &[1, 2]);
        let target =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0]), vec![1, 2], false).unwrap();
        let loss = MultiLabelSoftMarginLoss::default();
        let out = loss.forward(&input, &target).unwrap();
        let bce0 = (1.0 + (-2.0_f64).exp()).ln();
        let bce1 = (1.0 + (-1.0_f64).exp()).ln();
        let expected = (bce0 + bce1) / 2.0;
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-6,
            "MultiLabelSoftMargin: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_multi_label_soft_margin_backward() {
        let input = leaf_2d(&[0.0, 0.0], &[1, 2]);
        let target =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0]), vec![1, 2], false).unwrap();
        let loss = MultiLabelSoftMarginLoss::new(Reduction::Sum);
        let out = loss.forward(&input, &target).unwrap();
        backward(&out).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // sigmoid(0) = 0.5
        // grad[0] = (0.5 - 1) / 2 = -0.25
        // grad[1] = (0.5 - 0) / 2 = 0.25
        assert!(
            (g[0] - (-0.25)).abs() < 1e-6,
            "MLSM backward[0]: expected -0.25, got {}",
            g[0]
        );
        assert!(
            (g[1] - 0.25).abs() < 1e-6,
            "MLSM backward[1]: expected 0.25, got {}",
            g[1]
        );
    }

    #[test]
    fn test_multi_label_soft_margin_shape_mismatch() {
        let input = leaf_2d(&[1.0, 2.0], &[1, 2]);
        let target =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0, 0.0]), vec![1, 3], false)
                .unwrap();
        let loss = MultiLabelSoftMarginLoss::default();
        assert!(loss.forward(&input, &target).is_err());
    }

    // -----------------------------------------------------------------------
    // HingeEmbeddingLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_hinge_embedding_forward_mean() {
        // input = [0.5, 2.0], y = [1.0, -1.0], margin=1.0
        // loss[0] = 0.5 (positive)
        // loss[1] = max(0, 1.0 - 2.0) = 0 (negative, large input)
        // mean = 0.25
        let input = leaf_vec(&[0.5, 2.0]);
        let y = target_vec(&[1.0, -1.0]);
        let loss = HingeEmbeddingLoss::default();
        let out = loss.forward(&input, &y).unwrap();
        assert!(
            (out.item().unwrap() - 0.25).abs() < 1e-7,
            "HingeEmbedding mean: expected 0.25, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_hinge_embedding_negative_active() {
        // input = [0.3], y = [-1.0], margin=1.0
        // loss = max(0, 1.0 - 0.3) = 0.7
        let input = leaf_vec(&[0.3]);
        let y = target_vec(&[-1.0]);
        let loss = HingeEmbeddingLoss::new(Reduction::Mean, 1.0);
        let out = loss.forward(&input, &y).unwrap();
        assert!(
            (out.item().unwrap() - 0.7).abs() < 1e-7,
            "HingeEmbedding active: expected 0.7, got {}",
            out.item().unwrap()
        );
    }

    #[test]
    fn test_hinge_embedding_backward() {
        let input = leaf_vec(&[0.5, 0.3]);
        let y = target_vec(&[1.0, -1.0]);
        let loss = HingeEmbeddingLoss::new(Reduction::Mean, 1.0);
        let out = loss.forward(&input, &y).unwrap();
        backward(&out).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // grad[0]: y=1, grad = 1 / 2 = 0.5
        // grad[1]: y=-1, margin-x = 0.7 > 0, grad = -1 / 2 = -0.5
        assert!(
            (g[0] - 0.5).abs() < 1e-7,
            "HingeEmb backward[0]: expected 0.5, got {}",
            g[0]
        );
        assert!(
            (g[1] - (-0.5)).abs() < 1e-7,
            "HingeEmb backward[1]: expected -0.5, got {}",
            g[1]
        );
    }

    #[test]
    fn test_hinge_embedding_shape_mismatch() {
        let input = leaf_vec(&[0.5, 0.3]);
        let y = target_vec(&[1.0]);
        let loss = HingeEmbeddingLoss::default();
        assert!(loss.forward(&input, &y).is_err());
    }

    #[test]
    fn test_hinge_embedding_all_reductions() {
        let input = leaf_vec(&[0.5, 0.3]);
        let y = target_vec(&[1.0, -1.0]);
        // margin = 1.0
        // loss = [0.5, 0.7]

        let sum_loss = HingeEmbeddingLoss::new(Reduction::Sum, 1.0)
            .forward(&input, &y)
            .unwrap();
        assert!(
            (sum_loss.item().unwrap() - 1.2).abs() < 1e-7,
            "HingeEmb sum: expected 1.2, got {}",
            sum_loss.item().unwrap()
        );

        let none_loss = HingeEmbeddingLoss::new(Reduction::None, 1.0)
            .forward(&input, &y)
            .unwrap();
        assert_eq!(none_loss.shape(), &[2]);
        let d = none_loss.data().unwrap();
        assert!((d[0] - 0.5).abs() < 1e-7);
        assert!((d[1] - 0.7).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // GaussianNLLLoss
    // -----------------------------------------------------------------------

    #[test]
    fn test_gaussian_nll_forward_mean() {
        // input = [1.0, 2.0], target = [1.5, 2.5], var = [1.0, 2.0]
        // loss[0] = 0.5 * (ln(1.0) + (1.0-1.5)^2 / 1.0) = 0.5 * (0 + 0.25) = 0.125
        // loss[1] = 0.5 * (ln(2.0) + (2.0-2.5)^2 / 2.0) = 0.5 * (ln(2) + 0.125)
        // mean = (loss[0] + loss[1]) / 2
        let input = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.5, 2.5]);
        let var = target_vec(&[1.0, 2.0]);
        let loss = GaussianNLLLoss::default();
        let out = loss.forward(&input, &target, &var).unwrap();
        let e0 = 0.5 * (0.0 + 0.25);
        let e1 = 0.5 * (2.0_f64.ln() + 0.125);
        let expected = (e0 + e1) / 2.0;
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "GaussianNLL mean: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_gaussian_nll_forward_sum() {
        let input = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.5, 2.5]);
        let var = target_vec(&[1.0, 2.0]);
        let loss = GaussianNLLLoss::new(Reduction::Sum, false, 1e-6);
        let out = loss.forward(&input, &target, &var).unwrap();
        let e0 = 0.5 * (0.0 + 0.25);
        let e1 = 0.5 * (2.0_f64.ln() + 0.125);
        let expected = e0 + e1;
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "GaussianNLL sum: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_gaussian_nll_forward_none() {
        let input = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.5, 2.5]);
        let var = target_vec(&[1.0, 2.0]);
        let loss = GaussianNLLLoss::new(Reduction::None, false, 1e-6);
        let out = loss.forward(&input, &target, &var).unwrap();
        assert_eq!(out.shape(), &[2]);
        let d = out.data().unwrap();
        let e0 = 0.5 * (0.0 + 0.25);
        let e1 = 0.5 * (2.0_f64.ln() + 0.125);
        assert!(
            (d[0] - e0).abs() < 1e-7,
            "GaussianNLL none[0]: expected {}, got {}",
            e0,
            d[0]
        );
        assert!(
            (d[1] - e1).abs() < 1e-7,
            "GaussianNLL none[1]: expected {}, got {}",
            e1,
            d[1]
        );
    }

    #[test]
    fn test_gaussian_nll_full_mode() {
        // With full=true, adds 0.5 * log(2*pi) per element.
        let input = leaf_vec(&[0.0]);
        let target = target_vec(&[0.0]);
        let var = target_vec(&[1.0]);
        let loss = GaussianNLLLoss::new(Reduction::Mean, true, 1e-6);
        let out = loss.forward(&input, &target, &var).unwrap();
        // loss = 0.5 * (ln(1) + 0 + ln(2*pi)) = 0.5 * ln(2*pi)
        let expected = 0.5 * (2.0 * std::f64::consts::PI).ln();
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-7,
            "GaussianNLL full: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_gaussian_nll_backward_input() {
        // d(loss)/d(input) = (input - target) / var
        // input=2.0, target=1.0, var=4.0 => grad = 1.0/4.0 = 0.25 (mean, n=1)
        let input = leaf_vec(&[2.0]);
        let target = target_vec(&[1.0]);
        let var = target_vec(&[4.0]);
        let loss = GaussianNLLLoss::new(Reduction::Mean, false, 1e-6);
        let out = loss.forward(&input, &target, &var).unwrap();
        backward(&out).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        let expected = (2.0 - 1.0) / 4.0;
        assert!(
            (g[0] - expected).abs() < 1e-7,
            "GaussianNLL backward input: expected {}, got {}",
            expected,
            g[0]
        );
    }

    #[test]
    fn test_gaussian_nll_backward_var() {
        // d(loss)/d(var) = 0.5 * (1/var - diff^2/var^2)
        // input=2.0, target=1.0, var=4.0
        // => 0.5 * (1/4 - 1/16) = 0.5 * (0.25 - 0.0625) = 0.09375
        let input = leaf_vec(&[2.0]);
        let target = target_vec(&[1.0]);
        let var_tensor =
            Tensor::from_storage(TensorStorage::cpu(vec![4.0]), vec![1], true).unwrap();
        let loss = GaussianNLLLoss::new(Reduction::Mean, false, 1e-6);
        let out = loss.forward(&input, &target, &var_tensor).unwrap();
        backward(&out).unwrap();

        let grad = var_tensor.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        let expected = 0.5 * (1.0 / 4.0 - 1.0 / 16.0);
        assert!(
            (g[0] - expected).abs() < 1e-7,
            "GaussianNLL backward var: expected {}, got {}",
            expected,
            g[0]
        );
    }

    #[test]
    fn test_gaussian_nll_eps_clamp() {
        // Very small variance should be clamped to eps.
        let input = leaf_vec(&[1.0]);
        let target = target_vec(&[1.0]);
        let var = target_vec(&[0.0]); // zero variance
        let eps = 1e-6;
        let loss = GaussianNLLLoss::new(Reduction::Mean, false, eps);
        let out = loss.forward(&input, &target, &var).unwrap();
        // diff = 0, so loss = 0.5 * ln(eps)
        let expected = 0.5 * eps.ln();
        assert!(
            (out.item().unwrap() - expected).abs() < 1e-5,
            "GaussianNLL eps clamp: expected {}, got {}",
            expected,
            out.item().unwrap()
        );
    }

    #[test]
    fn test_gaussian_nll_shape_mismatch() {
        let input = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.0]);
        let var = target_vec(&[1.0, 1.0]);
        let loss = GaussianNLLLoss::default();
        assert!(loss.forward(&input, &target, &var).is_err());
    }

    #[test]
    fn test_gaussian_nll_var_shape_mismatch() {
        let input = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.0, 2.0]);
        let var = target_vec(&[1.0]);
        let loss = GaussianNLLLoss::default();
        assert!(loss.forward(&input, &target, &var).is_err());
    }

    #[test]
    fn test_gaussian_nll_zero_loss() {
        // When input == target and var == 1, loss = 0.5 * (0 + 0) = 0
        let input = leaf_vec(&[1.0, 2.0]);
        let target = target_vec(&[1.0, 2.0]);
        let var = target_vec(&[1.0, 1.0]);
        let loss = GaussianNLLLoss::default();
        let out = loss.forward(&input, &target, &var).unwrap();
        assert!(
            out.item().unwrap().abs() < 1e-10,
            "GaussianNLL zero loss: expected ~0, got {}",
            out.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // CosineEmbeddingLoss backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_embedding_backward_positive() {
        // x1 = [3, 4], x2 = [4, 3], y = 1 (positive)
        // ||x1|| = 5, ||x2|| = 5, dot = 24, cos = 24/25 = 0.96
        // loss = 1 - 0.96 = 0.04
        let x1 = leaf_2d(&[3.0, 4.0], &[1, 2]);
        let x2 =
            Tensor::from_storage(TensorStorage::cpu(vec![4.0, 3.0]), vec![1, 2], true).unwrap();
        let y = target_vec(&[1.0]);
        let loss = CosineEmbeddingLoss::new(Reduction::Sum, 0.0);
        let out = loss.forward_pair(&x1, &x2, &y).unwrap();

        let cos_val = 24.0 / 25.0;
        assert!(
            (out.item().unwrap() - (1.0 - cos_val)).abs() < 1e-7,
            "CosEmb positive: expected {}, got {}",
            1.0 - cos_val,
            out.item().unwrap()
        );

        backward(&out).unwrap();

        let grad_x1 = x1.grad().unwrap().unwrap();
        let g1 = grad_x1.data().unwrap();
        // d(loss)/d(x1_f) = -(x2_f/(||x1||*||x2||) - cos*x1_f/||x1||^2)
        // f=0: -(4/25 - 0.96*3/25) = -(0.16 - 0.1152) = -0.0448
        // f=1: -(3/25 - 0.96*4/25) = -(0.12 - 0.1536) = 0.0336
        let expected_g1_0 = -(4.0 / 25.0 - cos_val * 3.0 / 25.0);
        let expected_g1_1 = -(3.0 / 25.0 - cos_val * 4.0 / 25.0);
        assert!(
            (g1[0] - expected_g1_0).abs() < 1e-7,
            "CosEmb backward x1[0]: expected {}, got {}",
            expected_g1_0,
            g1[0]
        );
        assert!(
            (g1[1] - expected_g1_1).abs() < 1e-7,
            "CosEmb backward x1[1]: expected {}, got {}",
            expected_g1_1,
            g1[1]
        );

        let grad_x2 = x2.grad().unwrap().unwrap();
        let g2 = grad_x2.data().unwrap();
        let expected_g2_0 = -(3.0 / 25.0 - cos_val * 4.0 / 25.0);
        let expected_g2_1 = -(4.0 / 25.0 - cos_val * 3.0 / 25.0);
        assert!(
            (g2[0] - expected_g2_0).abs() < 1e-7,
            "CosEmb backward x2[0]: expected {}, got {}",
            expected_g2_0,
            g2[0]
        );
        assert!(
            (g2[1] - expected_g2_1).abs() < 1e-7,
            "CosEmb backward x2[1]: expected {}, got {}",
            expected_g2_1,
            g2[1]
        );
    }

    #[test]
    fn test_cosine_embedding_backward_negative_active() {
        // x1 = [1, 0], x2 = [1, 0], y = -1, margin = 0.5
        // cos = 1.0, loss = max(0, 1.0 - 0.5) = 0.5
        // Gradients should be opposite sign of positive case.
        let x1 = leaf_2d(&[1.0, 0.0], &[1, 2]);
        let x2 =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 0.0]), vec![1, 2], true).unwrap();
        let y = target_vec(&[-1.0]);
        let loss = CosineEmbeddingLoss::new(Reduction::Sum, 0.5);
        let out = loss.forward_pair(&x1, &x2, &y).unwrap();
        assert!(
            (out.item().unwrap() - 0.5).abs() < 1e-7,
            "CosEmb negative active: expected 0.5, got {}",
            out.item().unwrap()
        );
        backward(&out).unwrap();

        let grad_x1 = x1.grad().unwrap().unwrap();
        let g1 = grad_x1.data().unwrap();
        // d(cos)/d(x1_0) = x2_0/(||x1||*||x2||) - cos*x1_0/||x1||^2 = 1/1 - 1*1/1 = 0
        // d(cos)/d(x1_1) = x2_1/(||x1||*||x2||) - cos*x1_1/||x1||^2 = 0/1 - 1*0/1 = 0
        // For negative: grad = +d(cos)/d(x1)
        assert!(
            g1[0].abs() < 1e-7,
            "CosEmb neg backward x1[0]: expected 0, got {}",
            g1[0]
        );
        assert!(
            g1[1].abs() < 1e-7,
            "CosEmb neg backward x1[1]: expected 0, got {}",
            g1[1]
        );
    }

    #[test]
    fn test_cosine_embedding_backward_negative_inactive() {
        // x1 = [1, 0], x2 = [0, 1], y = -1, margin = 0.0
        // cos = 0, loss = max(0, 0 - 0) = 0, hinge inactive => grad = 0
        let x1 = leaf_2d(&[1.0, 0.0], &[1, 2]);
        let x2 =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0, 1.0]), vec![1, 2], true).unwrap();
        let y = target_vec(&[-1.0]);
        let loss = CosineEmbeddingLoss::new(Reduction::Sum, 0.0);
        let out = loss.forward_pair(&x1, &x2, &y).unwrap();
        backward(&out).unwrap();

        let grad_x1 = x1.grad().unwrap().unwrap();
        let g1 = grad_x1.data().unwrap();
        assert!(
            g1[0].abs() < 1e-7 && g1[1].abs() < 1e-7,
            "CosEmb inactive neg: expected zero grad, got {:?}",
            g1
        );
    }

    // -----------------------------------------------------------------------
    // CTCLoss backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_ctc_backward_gradients_sum_to_zero() {
        // For a valid probability distribution, gradients w.r.t. log_probs
        // should approximately sum to zero over classes for each timestep
        // (since probabilities sum to 1).
        let mut lp = vec![-10.0_f64; 3 * 3];
        // t=0: blank likely
        lp[0] = -0.1;
        lp[1] = -5.0;
        lp[2] = -5.0;
        // t=1: class 1 likely
        lp[3] = -5.0;
        lp[4] = -0.1;
        lp[5] = -5.0;
        // t=2: class 2 likely
        lp[6] = -5.0;
        lp[7] = -5.0;
        lp[8] = -0.1;

        let log_probs = Tensor::from_storage(TensorStorage::cpu(lp), vec![3, 1, 3], true).unwrap();
        let targets = target_vec(&[1.0, 2.0]);
        let loss = CTCLoss::new(Reduction::Sum, 0, false);
        let out = loss.forward(&log_probs, &targets, &[3], &[2]).unwrap();
        backward(&out).unwrap();

        let grad = log_probs.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // Verify gradients are finite.
        for i in 0..9 {
            assert!(g[i].is_finite(), "CTC grad[{}] is not finite: {}", i, g[i]);
        }
    }

    #[test]
    fn test_ctc_backward_empty_target() {
        // Empty target: grad should be -1 at blank for each timestep.
        let lp = vec![-0.5_f64, -10.0, -10.0, -0.3, -10.0, -10.0];
        let log_probs = Tensor::from_storage(TensorStorage::cpu(lp), vec![2, 1, 3], true).unwrap();
        let targets = target_vec(&[]);
        let loss = CTCLoss::new(Reduction::Sum, 0, false);
        let out = loss.forward(&log_probs, &targets, &[2], &[0]).unwrap();
        backward(&out).unwrap();

        let grad = log_probs.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // blank is class 0; grad at blank positions should be -1.
        assert!(
            (g[0] - (-1.0)).abs() < 1e-7,
            "CTC empty target grad[t=0,blank]: expected -1, got {}",
            g[0]
        );
        assert!(
            (g[3] - (-1.0)).abs() < 1e-7,
            "CTC empty target grad[t=1,blank]: expected -1, got {}",
            g[3]
        );
        // Non-blank positions should be 0.
        assert!(
            g[1].abs() < 1e-7,
            "CTC empty target grad[t=0,c=1]: expected 0, got {}",
            g[1]
        );
    }

    #[test]
    fn test_ctc_backward_no_grad() {
        // Inside no_grad, CTC should not attach grad_fn.
        ferrotorch_core::no_grad(|| {
            let lp = vec![-0.5_f64; 3 * 2];
            let log_probs =
                Tensor::from_storage(TensorStorage::cpu(lp), vec![3, 1, 2], true).unwrap();
            let targets = target_vec(&[1.0]);
            let loss = CTCLoss::default();
            let out = loss.forward(&log_probs, &targets, &[3], &[1]).unwrap();
            assert!(
                out.grad_fn().is_none(),
                "CTCLoss inside no_grad should not attach grad_fn"
            );
        });
    }

    #[test]
    fn test_ctc_backward_numerical_gradient() {
        // Numerical gradient check with central differences.
        let base_lp = vec![
            -0.5_f64, -1.0, -2.0, // t=0
            -1.0, -0.5, -2.0, // t=1
            -2.0, -1.0, -0.5, // t=2
        ];
        let eps = 1e-5;

        // Compute analytical gradient.
        let log_probs =
            Tensor::from_storage(TensorStorage::cpu(base_lp.clone()), vec![3, 1, 3], true).unwrap();
        let targets = target_vec(&[1.0, 2.0]);
        let loss = CTCLoss::new(Reduction::Sum, 0, false);
        let out = loss.forward(&log_probs, &targets, &[3], &[2]).unwrap();
        backward(&out).unwrap();

        let grad = log_probs.grad().unwrap().unwrap();
        let analytical = grad.data_vec().unwrap();

        // Numerical gradient via central differences.
        for idx in 0..9 {
            let mut lp_plus = base_lp.clone();
            lp_plus[idx] += eps;
            let lp_p =
                Tensor::from_storage(TensorStorage::cpu(lp_plus), vec![3, 1, 3], false).unwrap();
            let t_p = target_vec(&[1.0, 2.0]);
            let out_p = CTCLoss::new(Reduction::Sum, 0, false)
                .forward(&lp_p, &t_p, &[3], &[2])
                .unwrap();

            let mut lp_minus = base_lp.clone();
            lp_minus[idx] -= eps;
            let lp_m =
                Tensor::from_storage(TensorStorage::cpu(lp_minus), vec![3, 1, 3], false).unwrap();
            let t_m = target_vec(&[1.0, 2.0]);
            let out_m = CTCLoss::new(Reduction::Sum, 0, false)
                .forward(&lp_m, &t_m, &[3], &[2])
                .unwrap();

            let numerical = (out_p.item().unwrap() - out_m.item().unwrap()) / (2.0 * eps);
            assert!(
                (analytical[idx] - numerical).abs() < 1e-4,
                "CTC grad[{}]: analytical={}, numerical={}",
                idx,
                analytical[idx],
                numerical,
            );
        }
    }
}
