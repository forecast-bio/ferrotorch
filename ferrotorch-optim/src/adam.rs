//! Adam optimizer (Adaptive Moment Estimation).
//!
//! Implements the algorithm from Kingma & Ba, "Adam: A Method for Stochastic
//! Optimization" (ICLR 2015), including the AMSGrad variant from Reddi et al.
//!
//! All parameter updates execute inside `no_grad()` so the optimizer step is
//! never recorded in the autograd graph.

use std::collections::HashMap;

use ferrotorch_core::{no_grad, Float, FerrotorchError, FerrotorchResult};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// AdamConfig
// ---------------------------------------------------------------------------

/// Hyperparameters for the Adam optimizer.
#[derive(Debug, Clone, Copy)]
pub struct AdamConfig {
    /// Learning rate (default: 0.001).
    pub lr: f64,
    /// Exponential decay rates for the first and second moment estimates
    /// (default: (0.9, 0.999)).
    pub betas: (f64, f64),
    /// Term added to the denominator for numerical stability (default: 1e-8).
    pub eps: f64,
    /// Weight decay coefficient for L2 regularization (default: 0.0).
    pub weight_decay: f64,
    /// Whether to use the AMSGrad variant (default: false).
    pub amsgrad: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-parameter state
// ---------------------------------------------------------------------------

/// Mutable state tracked for each parameter across steps.
#[derive(Debug)]
struct AdamParamState {
    /// Number of optimizer steps taken for this parameter.
    step_count: u64,
    /// First moment estimate (exponential moving average of gradients).
    exp_avg: Vec<f64>,
    /// Second moment estimate (exponential moving average of squared gradients).
    exp_avg_sq: Vec<f64>,
    /// Maximum of bias-corrected second moment estimates (AMSGrad only).
    max_exp_avg_sq: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

/// The Adam optimizer.
///
/// Maintains per-parameter exponential moving averages of the gradient (first
/// moment) and the squared gradient (second moment). With `amsgrad = true`, it
/// additionally tracks the running maximum of the bias-corrected second moment.
#[derive(Debug)]
pub struct Adam<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdamConfig,
    /// Per-parameter state, keyed by `"g{group_idx}_p{param_idx}"`.
    state: HashMap<String, AdamParamState>,
}

impl<T: Float> Adam<T> {
    /// Create a new Adam optimizer for the given parameters.
    pub fn new(params: Vec<Parameter<T>>, config: AdamConfig) -> Self {
        let mut group = ParamGroup::new(params, config.lr);
        group.weight_decay = config.weight_decay;

        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
        }
    }

    /// Generate the state key for a parameter.
    #[inline]
    fn param_key(group_idx: usize, param_idx: usize) -> String {
        format!("g{group_idx}_p{param_idx}")
    }
}

impl<T: Float> Optimizer<T> for Adam<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        let config = self.config;
        let (beta1, beta2) = config.betas;

        for gi in 0..self.param_groups.len() {
            let group_lr = self.param_groups[gi].lr;
            let group_wd = self.param_groups[gi].weight_decay;

            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];
                let tensor = param.tensor();

                // Skip parameters without gradients.
                let grad_tensor = match tensor.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                let key = Self::param_key(gi, pi);

                // Read parameter data and gradient data into f64 workspace.
                let param_data: Vec<f64> = tensor
                    .data()?
                    .iter()
                    .map(|&v| v.to_f64().unwrap())
                    .collect();
                let mut grad_data: Vec<f64> = grad_tensor
                    .data()?
                    .iter()
                    .map(|&v| v.to_f64().unwrap())
                    .collect();

                // L2 weight decay: grad = grad + weight_decay * param.
                if group_wd > 0.0 {
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g += group_wd * p;
                    }
                }

                // Lazy-init state.
                let numel = param_data.len();
                let state = self.state.entry(key).or_insert_with(|| AdamParamState {
                    step_count: 0,
                    exp_avg: vec![0.0; numel],
                    exp_avg_sq: vec![0.0; numel],
                    max_exp_avg_sq: if config.amsgrad {
                        Some(vec![0.0; numel])
                    } else {
                        None
                    },
                });

                state.step_count += 1;
                let step = state.step_count;

                // Update first moment: exp_avg = beta1 * exp_avg + (1 - beta1) * grad.
                for (m, &g) in state.exp_avg.iter_mut().zip(grad_data.iter()) {
                    *m = beta1 * *m + (1.0 - beta1) * g;
                }

                // Update second moment: exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2.
                for (v, &g) in state.exp_avg_sq.iter_mut().zip(grad_data.iter()) {
                    *v = beta2 * *v + (1.0 - beta2) * g * g;
                }

                // Bias correction.
                let bc1 = 1.0 - beta1.powi(step as i32);
                let bc2 = 1.0 - beta2.powi(step as i32);

                // Apply update to parameter data in-place via data_mut.
                no_grad(|| {
                    // SAFETY: Optimizer step runs inside no_grad() with exclusive
                    // access to parameters, so no aliasing references exist.
                    let param_slice = unsafe { param.tensor().data_mut()? };
                    for i in 0..numel {
                        let corrected_avg = state.exp_avg[i] / bc1;
                        let corrected_sq = state.exp_avg_sq[i] / bc2;

                        let denom = if config.amsgrad {
                            let max_sq = state.max_exp_avg_sq.as_mut().unwrap();
                            if state.exp_avg_sq[i] > max_sq[i] {
                                max_sq[i] = state.exp_avg_sq[i];
                            }
                            (max_sq[i] / bc2).sqrt() + config.eps
                        } else {
                            corrected_sq.sqrt() + config.eps
                        };

                        let updated = param_data[i] - group_lr * corrected_avg / denom;
                        param_slice[i] = T::from(updated).unwrap();
                    }
                    Ok::<(), FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> FerrotorchResult<()> {
        for group in &self.param_groups {
            for param in &group.params {
                param.tensor().set_grad(None)?;
            }
        }
        Ok(())
    }

    fn lr(&self) -> f64 {
        self.param_groups
            .first()
            .map(|g| g.lr)
            .unwrap_or(self.config.lr)
    }

    fn set_lr(&mut self, lr: f64) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn param_groups(&self) -> &[ParamGroup<T>] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup<T>] {
        &mut self.param_groups
    }

    fn add_param_group(&mut self, group: ParamGroup<T>) {
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> OptimizerState {
        let mut out = OptimizerState::new();
        for (key, pstate) in &self.state {
            let mut entry = HashMap::new();
            entry.insert("step_count".to_string(), vec![pstate.step_count as f64]);
            entry.insert("exp_avg".to_string(), pstate.exp_avg.clone());
            entry.insert("exp_avg_sq".to_string(), pstate.exp_avg_sq.clone());
            if let Some(ref max_sq) = pstate.max_exp_avg_sq {
                entry.insert("max_exp_avg_sq".to_string(), max_sq.clone());
            }
            out.insert(key.clone(), entry);
        }
        out
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        for (key, entry) in state {
            let step_count = entry
                .get("step_count")
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(0.0) as u64;

            let exp_avg = entry
                .get("exp_avg")
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing exp_avg in state for key {key}"),
                })?;

            let exp_avg_sq = entry
                .get("exp_avg_sq")
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing exp_avg_sq in state for key {key}"),
                })?;

            let max_exp_avg_sq = entry.get("max_exp_avg_sq").cloned();

            self.state.insert(
                key.clone(),
                AdamParamState {
                    step_count,
                    exp_avg,
                    exp_avg_sq,
                    max_exp_avg_sq,
                },
            );
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::grad_fns::arithmetic::{add, mul, pow, sub};
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Create a scalar parameter from a single f64 value.
    fn scalar_param(val: f64) -> Parameter<f64> {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap();
        Parameter::new(t)
    }

    /// Read a scalar parameter's current value.
    fn param_val(opt: &Adam<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx]
            .tensor()
            .data()
            .unwrap()[0]
    }

    // -----------------------------------------------------------------------
    // Rosenbrock convergence test
    // -----------------------------------------------------------------------

    /// Rosenbrock function: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
    /// Global minimum at (1, 1) with f(1,1) = 0.
    #[test]
    fn test_adam_rosenbrock_convergence() {
        let px = scalar_param(-1.0);
        let py = scalar_param(1.0);

        let mut opt = Adam::new(
            vec![px, py],
            AdamConfig {
                lr: 0.01,
                ..Default::default()
            },
        );

        for _ in 0..5000 {
            opt.zero_grad().unwrap();

            // Read current parameter tensors from the optimizer (they get
            // replaced on each step).
            let x_tensor = opt.param_groups[0].params[0].tensor().clone();
            let y_tensor = opt.param_groups[0].params[1].tensor().clone();

            // f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
            let one = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false)
                .unwrap();
            let hundred =
                Tensor::from_storage(TensorStorage::cpu(vec![100.0_f64]), vec![], false).unwrap();

            // term1 = (1 - x)^2
            let diff1 = sub(&one, &x_tensor).unwrap();
            let term1 = pow(&diff1, 2.0).unwrap();

            // term2 = 100 * (y - x^2)^2
            let x_sq = pow(&x_tensor, 2.0).unwrap();
            let diff2 = sub(&y_tensor, &x_sq).unwrap();
            let diff2_sq = pow(&diff2, 2.0).unwrap();
            let term2 = mul(&hundred, &diff2_sq).unwrap();

            let loss = add(&term1, &term2).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        let final_x = param_val(&opt, 0, 0);
        let final_y = param_val(&opt, 0, 1);

        assert!(
            (final_x - 1.0).abs() < 0.05,
            "expected x near 1.0, got {final_x}"
        );
        assert!(
            (final_y - 1.0).abs() < 0.05,
            "expected y near 1.0, got {final_y}"
        );
    }

    // -----------------------------------------------------------------------
    // zero_grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_zero_grad() {
        let p = scalar_param(3.0);
        let mut opt = Adam::new(vec![p], AdamConfig::default());

        // Manually set a gradient.
        let grad = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        opt.param_groups[0].params[0]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();

        // Verify gradient exists.
        assert!(opt.param_groups[0].params[0]
            .tensor()
            .grad()
            .unwrap()
            .is_some());

        // Zero it out.
        opt.zero_grad().unwrap();

        assert!(opt.param_groups[0].params[0]
            .tensor()
            .grad()
            .unwrap()
            .is_none());
    }

    // -----------------------------------------------------------------------
    // state_dict / load_state_dict
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_state_dict_roundtrip() {
        let p = scalar_param(2.0);
        let mut opt = Adam::new(vec![p], AdamConfig::default());

        // Run a few steps so the state is populated.
        for _ in 0..3 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        // Save state.
        let saved = opt.state_dict();
        assert!(!saved.is_empty(), "state dict should be non-empty after steps");

        // Verify the state contains expected keys.
        let key = Adam::<f64>::param_key(0, 0);
        assert!(saved.contains_key(&key), "expected key {key} in state dict");

        let entry = &saved[&key];
        assert!(entry.contains_key("step_count"));
        assert!(entry.contains_key("exp_avg"));
        assert!(entry.contains_key("exp_avg_sq"));

        let step_count = entry["step_count"][0] as u64;
        assert_eq!(step_count, 3);

        // Load state into a fresh optimizer.
        let p2 = scalar_param(2.0);
        let mut opt2 = Adam::new(vec![p2], AdamConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2.state_dict();
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["exp_avg"], saved[&key]["exp_avg"]);
        assert_eq!(loaded[&key]["exp_avg_sq"], saved[&key]["exp_avg_sq"]);
    }

    // -----------------------------------------------------------------------
    // AMSGrad variant
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_amsgrad() {
        let p = scalar_param(5.0);
        let mut opt = Adam::new(
            vec![p],
            AdamConfig {
                amsgrad: true,
                ..Default::default()
            },
        );

        // A few steps to exercise the amsgrad path.
        for _ in 0..10 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        // Verify state contains max_exp_avg_sq.
        let saved = opt.state_dict();
        let key = Adam::<f64>::param_key(0, 0);
        assert!(saved[&key].contains_key("max_exp_avg_sq"));

        // The parameter should have moved towards zero (minimizing x^2).
        let val = param_val(&opt, 0, 0);
        assert!(
            val.abs() < 5.0,
            "parameter should have decreased from 5.0, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Weight decay
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_weight_decay() {
        let p = scalar_param(5.0);
        let config = AdamConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..Default::default()
        };
        let mut opt = Adam::new(vec![p], config);

        for _ in 0..50 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        // With weight decay, the parameter should still converge towards zero.
        let val = param_val(&opt, 0, 0);
        assert!(
            val.abs() < 1.0,
            "parameter should have moved towards 0 with weight decay, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Learning rate accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_lr_accessors() {
        let p = scalar_param(1.0);
        let mut opt = Adam::new(
            vec![p],
            AdamConfig {
                lr: 0.05,
                ..Default::default()
            },
        );

        assert!((opt.lr() - 0.05).abs() < 1e-12);

        opt.set_lr(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-12);
        assert!((opt.param_groups()[0].lr - 0.01).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Multiple parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_multiple_params() {
        let p1 = scalar_param(3.0);
        let p2 = scalar_param(-2.0);

        let mut opt = Adam::new(
            vec![p1, p2],
            AdamConfig {
                lr: 0.01,
                ..Default::default()
            },
        );

        for _ in 0..1000 {
            // Minimize f(a, b) = a^2 + b^2.
            opt.zero_grad().unwrap();
            let a = opt.param_groups[0].params[0].tensor().clone();
            let b = opt.param_groups[0].params[1].tensor().clone();

            let a_sq = pow(&a, 2.0).unwrap();
            let b_sq = pow(&b, 2.0).unwrap();
            let loss = add(&a_sq, &b_sq).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        let v1 = param_val(&opt, 0, 0);
        let v2 = param_val(&opt, 0, 1);
        assert!(
            v1.abs() < 0.5,
            "expected p1 near 0, got {v1}"
        );
        assert!(
            v2.abs() < 0.5,
            "expected p2 near 0, got {v2}"
        );
    }
}
