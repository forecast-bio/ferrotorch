//! Loss functions for training neural networks.
//!
//! Unlike layers, loss functions are **not** `Module<T>`. They are callable
//! structs with a `forward(&self, pred, target) -> FerrotorchResult<Tensor<T>>`
//! method. Each loss attaches a backward node to the returned tensor when
//! gradient tracking is enabled.

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::ops::elementwise::{binary_map, mean, sum, unary_map};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};
use ferrotorch_core::Float;
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

    pub fn forward<T: Float>(
        &self,
        pred: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
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
                TensorStorage::cpu(reduced.data()?.to_vec()),
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
        let cpu_pred = if self.pred.is_cuda() { self.pred.cpu()? } else { self.pred.clone() };
        let cpu_target = if self.target.is_cuda() { self.target.cpu()? } else { self.target.clone() };
        let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
        let pred_data = cpu_pred.data()?;
        let target_data = cpu_target.data()?;
        let grad_data = cpu_go.data()?;
        let two = T::from(2.0).unwrap();
        let n = T::from(pred_data.len()).unwrap();

        let result: Vec<T> = match self.reduction {
            Reduction::Mean => {
                // grad_output is scalar
                let go = grad_data[0];
                pred_data
                    .iter()
                    .zip(target_data.iter())
                    .map(|(&p, &t)| two * (p - t) * go / n)
                    .collect()
            }
            Reduction::Sum => {
                // grad_output is scalar
                let go = grad_data[0];
                pred_data
                    .iter()
                    .zip(target_data.iter())
                    .map(|(&p, &t)| two * (p - t) * go)
                    .collect()
            }
            Reduction::None => {
                // grad_output has same shape as pred
                pred_data
                    .iter()
                    .zip(target_data.iter())
                    .zip(grad_data.iter())
                    .map(|((&p, &t), &g)| two * (p - t) * g)
                    .collect()
            }
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

    pub fn forward<T: Float>(
        &self,
        logits: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
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
                sum_exp = sum_exp + e;
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
                    sum_lp = sum_lp + log_probs[base + c];
                }
                let smooth = -sum_lp / T::from(classes).unwrap();
                losses[b] = (one - ls) * nll + ls * smooth;
            } else {
                losses[b] = nll;
            }
        }

        let unreduced = Tensor::from_storage(
            TensorStorage::cpu(losses),
            vec![batch],
            false,
        )?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && logits.requires_grad() {
            let softmax_tensor = Tensor::from_storage(
                TensorStorage::cpu(softmax_out),
                vec![batch, classes],
                false,
            )?;
            let grad_fn = Arc::new(CrossEntropyBackward {
                logits: logits.clone(),
                targets: targets.clone(),
                softmax: softmax_tensor,
                label_smoothing: self.label_smoothing,
                reduction: self.reduction,
            });
            Tensor::from_operation(
                TensorStorage::cpu(reduced.data()?.to_vec()),
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
        let cpu_sm = if self.softmax.is_cuda() { self.softmax.cpu()? } else { self.softmax.clone() };
        let cpu_targets = if self.targets.is_cuda() { self.targets.cpu()? } else { self.targets.clone() };
        let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
        let sm_data = cpu_sm.data()?;
        let targets_data = cpu_targets.data()?;
        let grad_data = cpu_go.data()?;
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
                let one_hot = if c == target_class { one } else { <T as Zero>::zero() };
                // grad = softmax - ((1 - ls) * one_hot + ls / C)
                let target_dist = (one - ls) * one_hot + ls * inv_c;
                result[base + c] = (sm - target_dist) * scale;
            }
        }

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            shape.to_vec(),
            false,
        )?;
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

    pub fn forward<T: Float>(
        &self,
        logits: &Tensor<T>,
        targets: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
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
                TensorStorage::cpu(reduced.data()?.to_vec()),
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
        let cpu_logits = if self.logits.is_cuda() { self.logits.cpu()? } else { self.logits.clone() };
        let cpu_targets = if self.targets.is_cuda() { self.targets.cpu()? } else { self.targets.clone() };
        let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
        let logits_data = cpu_logits.data()?;
        let targets_data = cpu_targets.data()?;
        let grad_data = cpu_go.data()?;
        let one = <T as One>::one();
        let n = T::from(logits_data.len()).unwrap();

        let result: Vec<T> = match self.reduction {
            Reduction::Mean => {
                let go = grad_data[0];
                logits_data
                    .iter()
                    .zip(targets_data.iter())
                    .map(|(&x, &y)| {
                        let sig = one / (one + (-x).exp());
                        (sig - y) * go / n
                    })
                    .collect()
            }
            Reduction::Sum => {
                let go = grad_data[0];
                logits_data
                    .iter()
                    .zip(targets_data.iter())
                    .map(|(&x, &y)| {
                        let sig = one / (one + (-x).exp());
                        (sig - y) * go
                    })
                    .collect()
            }
            Reduction::None => logits_data
                .iter()
                .zip(targets_data.iter())
                .zip(grad_data.iter())
                .map(|((&x, &y), &g)| {
                    let sig = one / (one + (-x).exp());
                    (sig - y) * g
                })
                .collect(),
        };

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.logits.shape().to_vec(),
            false,
        )?;
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

        let unreduced = Tensor::from_storage(
            TensorStorage::cpu(loss_data),
            pred.shape().to_vec(),
            false,
        )?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && pred.requires_grad() {
            let grad_fn = Arc::new(HuberBackward {
                pred: pred.clone(),
                target: target.clone(),
                delta: self.delta,
                reduction: self.reduction,
            });
            Tensor::from_operation(
                TensorStorage::cpu(reduced.data()?.to_vec()),
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
        let cpu_pred = if self.pred.is_cuda() { self.pred.cpu()? } else { self.pred.clone() };
        let cpu_target = if self.target.is_cuda() { self.target.cpu()? } else { self.target.clone() };
        let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
        let pred_data = cpu_pred.data()?;
        let target_data = cpu_target.data()?;
        let grad_data = cpu_go.data()?;
        let delta = T::from(self.delta).unwrap();
        let n = T::from(pred_data.len()).unwrap();

        let result: Vec<T> = match self.reduction {
            Reduction::Mean => {
                let go = grad_data[0];
                pred_data
                    .iter()
                    .zip(target_data.iter())
                    .map(|(&p, &t)| {
                        let error = p - t;
                        let abs_error = error.abs();
                        let local_grad = if abs_error < delta {
                            error
                        } else {
                            delta * error.signum()
                        };
                        local_grad * go / n
                    })
                    .collect()
            }
            Reduction::Sum => {
                let go = grad_data[0];
                pred_data
                    .iter()
                    .zip(target_data.iter())
                    .map(|(&p, &t)| {
                        let error = p - t;
                        let abs_error = error.abs();
                        let local_grad = if abs_error < delta {
                            error
                        } else {
                            delta * error.signum()
                        };
                        local_grad * go
                    })
                    .collect()
            }
            Reduction::None => pred_data
                .iter()
                .zip(target_data.iter())
                .zip(grad_data.iter())
                .map(|((&p, &t), &g)| {
                    let error = p - t;
                    let abs_error = error.abs();
                    let local_grad = if abs_error < delta {
                        error
                    } else {
                        delta * error.signum()
                    };
                    local_grad * g
                })
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

        let unreduced = Tensor::from_storage(
            TensorStorage::cpu(loss_data),
            input.shape().to_vec(),
            false,
        )?;
        let reduced = apply_reduction(&unreduced, self.reduction)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(KLDivBackward {
                input: input.clone(),
                target: target.clone(),
                reduction: self.reduction,
            });
            Tensor::from_operation(
                TensorStorage::cpu(reduced.data()?.to_vec()),
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
        let cpu_target = if self.target.is_cuda() { self.target.cpu()? } else { self.target.clone() };
        let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
        let target_data = cpu_target.data()?;
        let grad_data = cpu_go.data()?;
        let n = T::from(target_data.len()).unwrap();

        let result: Vec<T> = match self.reduction {
            Reduction::Mean => {
                let go = grad_data[0];
                target_data.iter().map(|&t| -t * go / n).collect()
            }
            Reduction::Sum => {
                let go = grad_data[0];
                target_data.iter().map(|&t| -t * go).collect()
            }
            Reduction::None => target_data
                .iter()
                .zip(grad_data.iter())
                .map(|(&t, &g)| -t * g)
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
                dot = dot + a * bv;
                norm1_sq = norm1_sq + a * a;
                norm2_sq = norm2_sq + bv * bv;
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

        let unreduced = Tensor::from_storage(
            TensorStorage::cpu(losses),
            vec![batch],
            false,
        )?;
        apply_reduction(&unreduced, self.reduction)
    }
}

impl Default for CosineEmbeddingLoss {
    fn default() -> Self {
        Self::new(Reduction::Mean, 0.0)
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
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::autograd::graph::backward;
    use ferrotorch_core::storage::TensorStorage;

    /// Helper: 1-D leaf tensor with requires_grad.
    fn leaf_vec(vals: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(vals.to_vec()),
            vec![vals.len()],
            true,
        )
        .unwrap()
    }

    /// Helper: 1-D tensor without grad (for targets).
    fn target_vec(vals: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(vals.to_vec()),
            vec![vals.len()],
            false,
        )
        .unwrap()
    }

    /// Helper: 2-D leaf tensor with requires_grad.
    fn leaf_2d(vals: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(vals.to_vec()),
            shape.to_vec(),
            true,
        )
        .unwrap()
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
        let lsm = [1.0 - 3.0 - log_sum, 2.0 - 3.0 - log_sum, 3.0 - 3.0 - log_sum];
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
        let lsm = [1.0 - 3.0 - log_sum, 2.0 - 3.0 - log_sum, 3.0 - 3.0 - log_sum];
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
        let lsm = [1.0 - 3.0 - log_sum, 2.0 - 3.0 - log_sum, 3.0 - 3.0 - log_sum];

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
        let sm = [1.0_f64.exp() / sum_exp, 2.0_f64.exp() / sum_exp, 3.0_f64.exp() / sum_exp];
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
        assert!(val.is_finite(), "CE with large logits produced non-finite: {}", val);

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
        assert!(val.is_finite(), "CE with large negative logits produced non-finite: {}", val);
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
        assert!(val.is_finite(), "BCE large positive logit: non-finite {}", val);
        // loss = max(100,0) - 100*1 + log(1+exp(-100)) ~ 0 + ~0 = ~0
        assert!(val < 1e-10, "BCE large positive logit: expected ~0, got {}", val);
    }

    #[test]
    fn test_bce_numerical_stability_large_negative() {
        let logits = leaf_vec(&[-100.0]);
        let targets = target_vec(&[0.0]);
        let loss = BCEWithLogitsLoss::new(Reduction::Mean);
        let out = loss.forward(&logits, &targets).unwrap();
        let val = out.item().unwrap();
        assert!(val.is_finite(), "BCE large negative logit: non-finite {}", val);
        // loss = max(-100,0) - (-100)*0 + log(1+exp(-100)) ~ 0 + 0 + ~0
        assert!(val < 1e-10, "BCE large negative logit: expected ~0, got {}", val);
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

        let expected = 0.25 * (0.25_f64.ln() - 0.5_f64.ln())
            + 0.75 * (0.75_f64.ln() - 0.5_f64.ln());
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
}
