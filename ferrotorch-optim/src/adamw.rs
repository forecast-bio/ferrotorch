//! AdamW optimizer — Adam with decoupled weight decay.
//!
//! Unlike standard Adam (which adds L2 regularization to the gradient before
//! moment estimation), AdamW applies weight decay directly to the parameters
//! *before* the gradient update. This decoupling produces better
//! generalization in practice and is the default optimizer for most modern
//! deep learning workloads.
//!
//! Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
//! (ICLR 2019).

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Hyperparameters for the [`AdamW`] optimizer.
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    /// Learning rate (default: 0.001).
    pub lr: f64,
    /// Exponential decay rates for the first and second moment estimates
    /// (default: `(0.9, 0.999)`).
    pub betas: (f64, f64),
    /// Term added to the denominator for numerical stability (default: 1e-8).
    pub eps: f64,
    /// Decoupled weight decay coefficient (default: 0.01).
    ///
    /// Note: this default is higher than Adam's typical `0.0` because
    /// decoupled weight decay is the whole point of AdamW.
    pub weight_decay: f64,
    /// When `true`, maximize the objective by negating the gradient (default:
    /// false). CL-321
    pub maximize: bool,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
            maximize: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-parameter state
// ---------------------------------------------------------------------------

/// Internal state for a single parameter tracked by AdamW.
#[derive(Debug)]
struct AdamWParamState {
    /// Number of steps taken for this parameter.
    step_count: u64,
    /// First moment estimate (exponential moving average of gradients).
    exp_avg: Vec<f64>,
    /// Second moment estimate (exponential moving average of squared gradients).
    exp_avg_sq: Vec<f64>,
}

// ---------------------------------------------------------------------------
// AdamW
// ---------------------------------------------------------------------------

/// AdamW optimizer — Adam with **decoupled** weight decay.
///
/// # Algorithm
///
/// For each parameter `p` with gradient `g`:
///
/// 1. **Decoupled weight decay**: `p = p * (1 - lr * weight_decay)`
/// 2. **First moment update**: `m = beta1 * m + (1 - beta1) * g`
/// 3. **Second moment update**: `v = beta2 * v + (1 - beta2) * g^2`
/// 4. **Bias-corrected estimates**:
///    - `m_hat = m / (1 - beta1^t)`
///    - `v_hat = v / (1 - beta2^t)`
/// 5. **Parameter update**: `p = p - lr * m_hat / (sqrt(v_hat) + eps)`
///
/// The key difference from [`Adam`] is step 1: weight decay is applied
/// directly to the parameter, *not* added to the gradient. This means the
/// regularization strength is independent of the adaptive learning rate.
#[derive(Debug)]
pub struct AdamW<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdamWConfig,
    /// Per-parameter optimizer state, keyed by `"g{group_idx}_p{param_idx}"`.
    state: HashMap<String, AdamWParamState>,
}

impl<T: Float> AdamW<T> {
    /// Create a new AdamW optimizer with the given parameters and config.
    pub fn new(params: Vec<Parameter<T>>, config: AdamWConfig) -> Self {
        let mut group = ParamGroup::new(params, config.lr);
        group.weight_decay = config.weight_decay;
        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
        }
    }

    /// Create a new AdamW optimizer with pre-configured parameter groups.
    pub fn new_with_groups(groups: Vec<ParamGroup<T>>, config: AdamWConfig) -> Self {
        Self {
            param_groups: groups,
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

impl<T: Float> Optimizer<T> for AdamW<T> {
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
                // data_vec() handles GPU→CPU transfer transparently.
                let param_data: Vec<f64> = tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| v.to_f64().unwrap())
                    .collect();
                let mut grad_data: Vec<f64> = grad_tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| v.to_f64().unwrap())
                    .collect();

                // Maximize: negate gradient. CL-321
                if config.maximize {
                    for g in grad_data.iter_mut() {
                        *g = -*g;
                    }
                }

                let numel = param_data.len();

                // ----------------------------------------------------------
                // 1. Decoupled weight decay: p = p * (1 - lr * wd)
                //
                // This is the defining characteristic of AdamW. Unlike Adam
                // which adds wd*param to the gradient (L2 regularization),
                // AdamW decays the parameter values directly. The gradient
                // is NOT modified.
                // ----------------------------------------------------------
                let decay_factor = 1.0 - group_lr * group_wd;

                // ----------------------------------------------------------
                // 2-3. Moment updates (standard Adam, no L2 in gradient)
                //
                // Compute into temporaries so that on partial failure
                // (e.g. GPU upload error in update_data) the optimizer
                // state remains consistent and the step can be retried.
                // ----------------------------------------------------------
                let state = self.state.entry(key).or_insert_with(|| AdamWParamState {
                    step_count: 0,
                    exp_avg: vec![0.0; numel],
                    exp_avg_sq: vec![0.0; numel],
                });

                let next_step = state.step_count + 1;

                // exp_avg_new  = beta1 * exp_avg  + (1 - beta1) * grad
                // exp_avg_sq_new = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                let mut exp_avg_new = Vec::with_capacity(numel);
                let mut exp_avg_sq_new = Vec::with_capacity(numel);
                for i in 0..numel {
                    let g = grad_data[i];
                    exp_avg_new.push(beta1 * state.exp_avg[i] + (1.0 - beta1) * g);
                    exp_avg_sq_new.push(beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * g * g);
                }

                // ----------------------------------------------------------
                // 4-5. Bias correction and parameter update computation
                // ----------------------------------------------------------
                let bc1 = 1.0 - beta1.powi(next_step as i32);
                let bc2 = 1.0 - beta2.powi(next_step as i32);

                let new_values: Vec<T> = (0..numel)
                    .map(|i| {
                        let m_hat = exp_avg_new[i] / bc1;
                        let v_hat = exp_avg_sq_new[i] / bc2;
                        let decayed = param_data[i] * decay_factor;
                        let updated = decayed - group_lr * m_hat / (v_hat.sqrt() + config.eps);
                        T::from(updated).unwrap()
                    })
                    .collect();

                // ----------------------------------------------------------
                // Write the parameter update. Only commit state AFTER this
                // succeeds, so a failure leaves state unchanged.
                // ----------------------------------------------------------
                no_grad(|| {
                    // SAFETY: update_data() casts through Arc::as_ptr() to mutate
                    // shared storage. This is sound here because:
                    // 1. We are inside no_grad(), so no autograd graph is being
                    //    built and no grad_fn holds a reference to this storage.
                    // 2. The optimizer owns the Parameter, and step() borrows
                    //    &mut self, so no concurrent reads exist.
                    // 3. All data was already read (param_data, grad_data) before
                    //    this mutation, so we do not alias live references.
                    //
                    // Invariant: callers must NOT hold references into
                    // param.tensor().data() across this call.
                    unsafe { param.tensor().update_data(&new_values) }
                })?;

                // Commit state only after successful parameter write.
                state.step_count = next_step;
                state.exp_avg = exp_avg_new;
                state.exp_avg_sq = exp_avg_sq_new;
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
        for (key, ps) in &self.state {
            let mut entry = HashMap::new();
            entry.insert("step_count".to_string(), vec![ps.step_count as f64]);
            entry.insert("exp_avg".to_string(), ps.exp_avg.clone());
            entry.insert("exp_avg_sq".to_string(), ps.exp_avg_sq.clone());
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

            let exp_avg =
                entry
                    .get("exp_avg")
                    .cloned()
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!("missing exp_avg in state for key {key}"),
                    })?;

            let exp_avg_sq = entry.get("exp_avg_sq").cloned().ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("missing exp_avg_sq in state for key {key}"),
                }
            })?;

            self.state.insert(
                key.clone(),
                AdamWParamState {
                    step_count,
                    exp_avg,
                    exp_avg_sq,
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
    use ferrotorch_core::grad_fns::arithmetic::{add, pow};
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Create a scalar parameter from a single f64 value.
    fn scalar_param(val: f64) -> Parameter<f64> {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap();
        Parameter::new(t)
    }

    /// Create a 1-D parameter from a slice.
    fn vec_param(data: &[f64]) -> Parameter<f64> {
        let t = Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], true)
            .unwrap();
        Parameter::new(t)
    }

    /// Read a scalar parameter's current value from the optimizer.
    fn param_val(opt: &AdamW<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    /// Read a vector parameter's data from the optimizer.
    fn param_vec(opt: &AdamW<f64>, group: usize, idx: usize) -> Vec<f64> {
        opt.param_groups[group].params[idx]
            .tensor()
            .data()
            .unwrap()
            .to_vec()
    }

    /// Set gradient on a parameter inside the optimizer.
    fn set_grad_scalar(opt: &AdamW<f64>, group: usize, idx: usize, val: f64) {
        let grad = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false).unwrap();
        opt.param_groups[group].params[idx]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();
    }

    /// Set gradient on a vector parameter inside the optimizer.
    fn set_grad_vec(opt: &AdamW<f64>, group: usize, idx: usize, data: &[f64]) {
        let grad = Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false)
            .unwrap();
        opt.param_groups[group].params[idx]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();
    }

    /// L2 norm of a slice.
    fn l2_norm(data: &[f64]) -> f64 {
        data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    // -------------------------------------------------------------------
    // Default config values
    // -------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let config = AdamWConfig::default();
        assert_eq!(config.lr, 1e-3);
        assert_eq!(config.betas, (0.9, 0.999));
        assert_eq!(config.eps, 1e-8);
        assert_eq!(config.weight_decay, 0.01);
    }

    // -------------------------------------------------------------------
    // Basic step test
    // -------------------------------------------------------------------

    #[test]
    fn test_adamw_single_step() {
        let p = vec_param(&[1.0, 2.0, 3.0]);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.01,
                weight_decay: 0.1,
                ..Default::default()
            },
        );

        set_grad_vec(&opt, 0, 0, &[0.1, 0.2, 0.3]);
        opt.step().unwrap();

        let data = param_vec(&opt, 0, 0);
        // After one step, each element should have decreased
        // (weight decay pulls toward zero, gradient update also pulls down).
        assert!(data[0] < 1.0, "param[0] should decrease, got {}", data[0]);
        assert!(data[1] < 2.0, "param[1] should decrease, got {}", data[1]);
        assert!(data[2] < 3.0, "param[2] should decrease, got {}", data[2]);
    }

    // -------------------------------------------------------------------
    // Decoupled weight decay verification
    // -------------------------------------------------------------------

    #[test]
    fn test_weight_decay_with_zero_gradient() {
        // With zero gradient, only decoupled weight decay should shrink params.
        // With zero grad, AdamW still decays: p *= (1 - lr * wd).
        // The Adam moment update with zero grad contributes nothing to the update.
        let p = vec_param(&[5.0, -3.0, 10.0]);
        let lr = 0.01;
        let wd = 0.1;
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr,
                weight_decay: wd,
                ..Default::default()
            },
        );

        let initial = param_vec(&opt, 0, 0);
        set_grad_vec(&opt, 0, 0, &[0.0, 0.0, 0.0]);
        opt.step().unwrap();

        let after = param_vec(&opt, 0, 0);
        let decay_factor = 1.0 - lr * wd;

        for i in 0..3 {
            let expected = initial[i] * decay_factor;
            assert!(
                (after[i] - expected).abs() < 1e-10,
                "param[{i}]: expected {expected}, got {}",
                after[i]
            );
        }
    }

    #[test]
    fn test_param_norm_decreases_with_zero_gradient() {
        // Even with zero gradients, param norm should decrease due to
        // decoupled weight decay. This is the hallmark of AdamW.
        let p = vec_param(&[3.0, 4.0]);
        let lr = 0.1;
        let wd = 0.5;
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr,
                weight_decay: wd,
                ..Default::default()
            },
        );

        let norm_before = l2_norm(&param_vec(&opt, 0, 0));

        for _ in 0..10 {
            set_grad_vec(&opt, 0, 0, &[0.0, 0.0]);
            opt.step().unwrap();
        }

        let norm_after = l2_norm(&param_vec(&opt, 0, 0));
        assert!(
            norm_after < norm_before,
            "norm should decrease with zero grad due to decoupled wd: {norm_before} -> {norm_after}"
        );

        // Verify the decay is multiplicative: (1 - 0.1 * 0.5)^10 = 0.95^10
        let expected_ratio = (1.0 - lr * wd).powi(10);
        let actual_ratio = norm_after / norm_before;
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-10,
            "expected ratio {expected_ratio}, got {actual_ratio}"
        );
    }

    // -------------------------------------------------------------------
    // Convergence test — minimize f(x) = x^2 starting from x = 5.0
    // -------------------------------------------------------------------

    #[test]
    fn test_convergence_quadratic() {
        let p = scalar_param(5.0);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.1,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        for _ in 0..1000 {
            opt.zero_grad().unwrap();
            let val = param_val(&opt, 0, 0);
            // grad of x^2 = 2x
            set_grad_scalar(&opt, 0, 0, 2.0 * val);
            opt.step().unwrap();
        }

        let final_val = param_val(&opt, 0, 0);
        assert!(
            final_val.abs() < 0.01,
            "should converge near zero, got {final_val}"
        );
    }

    // -------------------------------------------------------------------
    // Convergence on Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // -------------------------------------------------------------------

    #[test]
    fn test_convergence_rosenbrock() {
        let p = vec_param(&[-1.0, 1.0]);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.001,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        for _ in 0..10000 {
            opt.zero_grad().unwrap();
            let vals = param_vec(&opt, 0, 0);
            let x = vals[0];
            let y = vals[1];

            // grad_x = 2(x-1) - 400x(y-x^2)
            // grad_y = 200(y-x^2)
            let gx = 2.0 * (x - 1.0) - 400.0 * x * (y - x * x);
            let gy = 200.0 * (y - x * x);

            set_grad_vec(&opt, 0, 0, &[gx, gy]);
            opt.step().unwrap();
        }

        let vals = param_vec(&opt, 0, 0);
        let dist = ((vals[0] - 1.0).powi(2) + (vals[1] - 1.0).powi(2)).sqrt();
        assert!(
            dist < 0.1,
            "should converge near (1,1), got ({}, {}), dist={dist}",
            vals[0],
            vals[1]
        );
    }

    // -------------------------------------------------------------------
    // Bias correction test
    // -------------------------------------------------------------------

    #[test]
    fn test_bias_correction_early_steps() {
        // After one step with grad=1.0, param starts at 0.0:
        //   m = 0.1 * 1.0 = 0.1, v = 0.001 * 1.0 = 0.001
        //   bc1 = 0.1, bc2 = 0.001
        //   m_hat = 0.1 / 0.1 = 1.0, v_hat = 0.001 / 0.001 = 1.0
        //   update = 0.001 * 1.0 / (1.0 + 1e-8) ~= 0.001
        //   param = 0.0 - 0.001 = -0.001
        let p = scalar_param(0.0);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.001,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        let val = param_val(&opt, 0, 0);
        assert!((val - (-0.001)).abs() < 1e-8, "expected ~-0.001, got {val}");
    }

    // -------------------------------------------------------------------
    // Zero grad test
    // -------------------------------------------------------------------

    #[test]
    fn test_zero_grad() {
        let p = vec_param(&[1.0, 2.0]);
        let mut opt = AdamW::new(vec![p], AdamWConfig::default());

        // Set a gradient.
        set_grad_vec(&opt, 0, 0, &[0.5, 0.5]);
        assert!(
            opt.param_groups[0].params[0]
                .tensor()
                .grad()
                .unwrap()
                .is_some()
        );

        opt.zero_grad().unwrap();
        assert!(
            opt.param_groups[0].params[0]
                .tensor()
                .grad()
                .unwrap()
                .is_none()
        );
    }

    // -------------------------------------------------------------------
    // LR get/set test
    // -------------------------------------------------------------------

    #[test]
    fn test_lr_get_set() {
        let p = scalar_param(1.0);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.01,
                ..Default::default()
            },
        );

        assert!((opt.lr() - 0.01).abs() < 1e-12);
        opt.set_lr(0.001);
        assert!((opt.lr() - 0.001).abs() < 1e-12);
        assert!((opt.param_groups()[0].lr - 0.001).abs() < 1e-12);
    }

    // -------------------------------------------------------------------
    // State dict round-trip
    // -------------------------------------------------------------------

    #[test]
    fn test_state_dict_round_trip() {
        let p = vec_param(&[1.0, 2.0, 3.0]);
        let mut opt = AdamW::new(vec![p], AdamWConfig::default());

        // Take a few steps to build state.
        for _ in 0..3 {
            set_grad_vec(&opt, 0, 0, &[0.1, 0.2, 0.3]);
            opt.step().unwrap();
        }

        let saved = opt.state_dict();
        assert!(!saved.is_empty(), "state dict should be non-empty");

        // Verify the saved state has expected keys.
        let key = AdamW::<f64>::param_key(0, 0);
        assert!(saved.contains_key(&key), "expected key {key} in state dict");

        let entry = &saved[&key];
        assert!(entry.contains_key("step_count"));
        assert!(entry.contains_key("exp_avg"));
        assert!(entry.contains_key("exp_avg_sq"));

        let step_count = entry["step_count"][0] as u64;
        assert_eq!(step_count, 3);

        // Load state into a fresh optimizer.
        let p2 = vec_param(&[1.0, 2.0, 3.0]);
        let mut opt2 = AdamW::new(vec![p2], AdamWConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2.state_dict();
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["exp_avg"], saved[&key]["exp_avg"]);
        assert_eq!(loaded[&key]["exp_avg_sq"], saved[&key]["exp_avg_sq"]);
    }

    // -------------------------------------------------------------------
    // Multiple parameter groups
    // -------------------------------------------------------------------

    #[test]
    fn test_multiple_param_groups() {
        let p1 = scalar_param(5.0);
        let p2 = scalar_param(5.0);

        let group1 = ParamGroup::new(vec![p1], 0.01).with_weight_decay(0.0);
        let group2 = ParamGroup::new(vec![p2], 0.01).with_weight_decay(1.0);

        let mut opt = AdamW::new_with_groups(
            vec![group1, group2],
            AdamWConfig {
                lr: 0.01,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        // With zero gradient, only group2 (wd=1.0) should see decay.
        set_grad_scalar(&opt, 0, 0, 0.0);
        set_grad_scalar(&opt, 1, 0, 0.0);
        opt.step().unwrap();

        let v1 = param_val(&opt, 0, 0);
        let v2 = param_val(&opt, 1, 0);

        // p1 should be unchanged (no wd, no grad update with zero grad).
        assert!((v1 - 5.0).abs() < 1e-10, "p1 should stay at 5.0, got {v1}");

        // p2 should have decayed: 5.0 * (1 - 0.01 * 1.0) = 5.0 * 0.99 = 4.95
        assert!(
            (v2 - 4.95).abs() < 1e-10,
            "p2 should decay to ~4.95, got {v2}"
        );
    }

    // -------------------------------------------------------------------
    // Skip params with no gradient
    // -------------------------------------------------------------------

    #[test]
    fn test_skip_params_without_grad() {
        let p1 = scalar_param(1.0);
        let p2 = scalar_param(2.0);

        let mut opt = AdamW::new(
            vec![p1, p2],
            AdamWConfig {
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        // Only set grad on p1.
        set_grad_scalar(&opt, 0, 0, 0.5);
        // p2 has no gradient.

        opt.step().unwrap();

        // p1 should have been updated.
        let v1 = param_val(&opt, 0, 0);
        assert!(v1 != 1.0, "p1 should have been updated, got {v1}");

        // p2 should be unchanged (no grad = skipped).
        let v2 = param_val(&opt, 0, 1);
        assert_eq!(v2, 2.0, "p2 should be unchanged (no grad)");
    }

    // -------------------------------------------------------------------
    // Multiple steps produce monotonic loss decrease on quadratic
    // -------------------------------------------------------------------

    #[test]
    fn test_monotonic_loss_decrease() {
        let p = scalar_param(10.0);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.1,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        let mut prev_loss = f64::MAX;
        for _ in 0..300 {
            opt.zero_grad().unwrap();
            let x = param_val(&opt, 0, 0);
            let loss = x * x;
            assert!(
                loss <= prev_loss + 1e-10,
                "loss should decrease: {prev_loss} -> {loss}"
            );
            prev_loss = loss;
            set_grad_scalar(&opt, 0, 0, 2.0 * x);
            opt.step().unwrap();
        }
        assert!(
            prev_loss < 0.01,
            "should converge, final loss = {prev_loss}"
        );
    }

    // -------------------------------------------------------------------
    // Add param group mid-training
    // -------------------------------------------------------------------

    #[test]
    fn test_add_param_group() {
        let p1 = scalar_param(5.0);
        let mut opt = AdamW::new(
            vec![p1],
            AdamWConfig {
                lr: 0.01,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();
        assert!(param_val(&opt, 0, 0) < 5.0);

        // Add a second group.
        let p2 = scalar_param(10.0);
        let group2 = ParamGroup::new(vec![p2], 0.01);
        opt.add_param_group(group2);

        set_grad_scalar(&opt, 0, 0, 1.0);
        set_grad_scalar(&opt, 1, 0, 1.0);
        opt.step().unwrap();

        let v2 = param_val(&opt, 1, 0);
        assert!(
            v2 < 10.0,
            "p2 should have been updated after add_param_group, got {v2}"
        );
    }

    // -------------------------------------------------------------------
    // Convergence with autograd backward (end-to-end)
    // -------------------------------------------------------------------

    #[test]
    fn test_convergence_with_autograd() {
        // Minimize f(x,y) = x^2 + y^2 using autograd backward.
        let px = scalar_param(3.0);
        let py = scalar_param(-2.0);

        let mut opt = AdamW::new(
            vec![px, py],
            AdamWConfig {
                lr: 0.01,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        for _ in 0..2000 {
            opt.zero_grad().unwrap();

            let x = opt.param_groups[0].params[0].tensor().clone();
            let y = opt.param_groups[0].params[1].tensor().clone();

            let x_sq = pow(&x, 2.0).unwrap();
            let y_sq = pow(&y, 2.0).unwrap();
            let loss = add(&x_sq, &y_sq).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        let vx = param_val(&opt, 0, 0);
        let vy = param_val(&opt, 0, 1);
        assert!(vx.abs() < 0.1, "expected x near 0, got {vx}");
        assert!(vy.abs() < 0.1, "expected y near 0, got {vy}");
    }
}
