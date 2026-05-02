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

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, no_grad};
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
    /// When `true`, the optimizer uses an on-device tensor-op update path
    /// that keeps moments on the parameter's native device and avoids the
    /// per-step CPU↔GPU round-trip the legacy `f64` workspace path
    /// requires (default: false). On CUDA params this is the dominant
    /// performance win for transformer training. CL-388
    pub foreach: bool,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
            maximize: false,
            foreach: false,
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

/// On-device state used by the foreach (tensor-op) update path. Moments
/// live on the parameter's device and are updated via GPU-aware tensor ops.
#[derive(Debug)]
struct AdamWForeachState<T: Float> {
    step_count: u64,
    exp_avg: Tensor<T>,
    exp_avg_sq: Tensor<T>,
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
    /// Legacy CPU-side per-parameter state, keyed by
    /// `"g{group_idx}_p{param_idx}"`. Used when `config.foreach == false`.
    state: HashMap<String, AdamWParamState>,
    /// Foreach (on-device) per-parameter state. Used when
    /// `config.foreach == true`.
    foreach_state: HashMap<String, AdamWForeachState<T>>,
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
            foreach_state: HashMap::new(),
        }
    }

    /// Create a new AdamW optimizer with pre-configured parameter groups.
    pub fn new_with_groups(groups: Vec<ParamGroup<T>>, config: AdamWConfig) -> Self {
        Self {
            param_groups: groups,
            config,
            state: HashMap::new(),
            foreach_state: HashMap::new(),
        }
    }

    /// Generate the state key for a parameter.
    #[inline]
    fn param_key(group_idx: usize, param_idx: usize) -> String {
        format!("g{group_idx}_p{param_idx}")
    }

    /// Foreach (on-device, tensor-op) update path used when
    /// `config.foreach == true`. Mirrors the legacy CPU path
    /// numerically (within f32/f64 precision).
    fn step_foreach(&mut self) -> FerrotorchResult<()> {
        use ferrotorch_core::creation::{scalar, zeros};
        use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, neg, sqrt, sub};

        let config = self.config;
        let (beta1, beta2) = config.betas;

        for gi in 0..self.param_groups.len() {
            let group_lr = self.param_groups[gi].lr;
            let group_wd = self.param_groups[gi].weight_decay;

            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];
                let param_t = param.tensor();

                let grad_tensor = match param.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                let device = param_t.device();
                let key = Self::param_key(gi, pi);

                no_grad(|| {
                    // grad (possibly negated for maximize). AdamW does NOT
                    // L2-add wd*param to the gradient — that's the difference
                    // from Adam. The decay is applied directly to the param.
                    let grad: Tensor<T> = if config.maximize {
                        neg(&grad_tensor)?
                    } else {
                        grad_tensor.clone()
                    };

                    // Initialize state if first step.
                    let next_step = {
                        let st = self.foreach_state.entry(key.clone()).or_insert_with(|| {
                            AdamWForeachState {
                                step_count: 0,
                                exp_avg: zeros::<T>(param_t.shape())
                                    .expect("zeros allocation")
                                    .to(device)
                                    .expect("zeros to device"),
                                exp_avg_sq: zeros::<T>(param_t.shape())
                                    .expect("zeros allocation")
                                    .to(device)
                                    .expect("zeros to device"),
                            }
                        });
                        st.step_count + 1
                    };

                    // Read current moment tensors.
                    let exp_avg_old = self.foreach_state[&key].exp_avg.clone();
                    let exp_avg_sq_old = self.foreach_state[&key].exp_avg_sq.clone();

                    // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                    let beta1_t = scalar(T::from(beta1).unwrap())?.to(device)?;
                    let one_minus_beta1 = scalar(T::from(1.0 - beta1).unwrap())?.to(device)?;
                    let exp_avg_new = add(
                        &mul(&exp_avg_old, &beta1_t)?,
                        &mul(&grad, &one_minus_beta1)?,
                    )?;

                    // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                    let beta2_t = scalar(T::from(beta2).unwrap())?.to(device)?;
                    let one_minus_beta2 = scalar(T::from(1.0 - beta2).unwrap())?.to(device)?;
                    let grad_sq = mul(&grad, &grad)?;
                    let exp_avg_sq_new = add(
                        &mul(&exp_avg_sq_old, &beta2_t)?,
                        &mul(&grad_sq, &one_minus_beta2)?,
                    )?;

                    // Bias-corrected moments. m_hat = exp_avg / (1 - beta1^t)
                    let bc1 = 1.0 - beta1.powi(next_step as i32);
                    let bc2 = 1.0 - beta2.powi(next_step as i32);
                    let inv_bc1 = scalar(T::from(1.0 / bc1).unwrap())?.to(device)?;
                    let inv_bc2 = scalar(T::from(1.0 / bc2).unwrap())?.to(device)?;
                    let m_hat = mul(&exp_avg_new, &inv_bc1)?;
                    let v_hat = mul(&exp_avg_sq_new, &inv_bc2)?;

                    // sqrt(v_hat) + eps
                    let sqrt_v = sqrt(&v_hat)?;
                    let eps_t = scalar(T::from(config.eps).unwrap())?.to(device)?;
                    let denom = add(&sqrt_v, &eps_t)?;
                    let update = div(&m_hat, &denom)?;

                    // Decoupled weight decay: param *= (1 - lr * wd)
                    // Then param -= lr * update.
                    let decay_factor = T::from(1.0 - group_lr * group_wd).unwrap();
                    let decay_t = scalar(decay_factor)?.to(device)?;
                    let lr_t = scalar(T::from(group_lr).unwrap())?.to(device)?;

                    let decayed = mul(param_t, &decay_t)?;
                    let scaled_update = mul(&update, &lr_t)?;
                    let new_param = sub(&decayed, &scaled_update)?;

                    // Commit param update.
                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: optimizer step inside no_grad, exclusive access.
                    unsafe { param_t.update_storage(storage)? };

                    // Commit state ONLY after the parameter update succeeded,
                    // matching the legacy path's failure semantics.
                    let st = self.foreach_state.get_mut(&key).unwrap();
                    st.step_count = next_step;
                    st.exp_avg = exp_avg_new;
                    st.exp_avg_sq = exp_avg_sq_new;

                    Ok::<(), ferrotorch_core::FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }
}

impl<T: Float> Optimizer<T> for AdamW<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        // When foreach mode is on, dispatch to the GPU-aware tensor-op
        // path. The legacy CPU f64 path below remains the default.
        if self.config.foreach {
            return self.step_foreach();
        }

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
                for ((&g, &ea), &eas) in grad_data
                    .iter()
                    .zip(state.exp_avg.iter())
                    .zip(state.exp_avg_sq.iter())
                {
                    exp_avg_new.push(beta1 * ea + (1.0 - beta1) * g);
                    exp_avg_sq_new.push(beta2 * eas + (1.0 - beta2) * g * g);
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

    // -----------------------------------------------------------------------
    // Foreach mode parity tests. CL-388
    //
    // The foreach path uses tensor ops on the parameter's native device
    // instead of downloading to a CPU f64 workspace. For correctness we
    // verify the on-device path matches the legacy CPU path step-for-step
    // within float precision.
    //
    // Note: AdamW's CPU path computes in f64 internally, so we allow a
    // small tolerance for the f32 foreach path. The convergence behavior
    // (steady-state error after many steps) should still match.
    // -----------------------------------------------------------------------

    fn paired(data: &[f32]) -> (Parameter<f32>, Parameter<f32>) {
        let p1 = Parameter::from_slice(data, &[data.len()]).unwrap();
        let p2 = Parameter::from_slice(data, &[data.len()]).unwrap();
        (p1, p2)
    }

    fn leaf_grad_f32(data: &[f32]) -> ferrotorch_core::Tensor<f32> {
        ferrotorch_core::Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            false,
        )
        .unwrap()
    }

    #[test]
    fn test_adamw_foreach_basic_parity_no_decay() {
        // Single step, no weight decay — pure Adam moment update.
        let (p_legacy, p_foreach) = paired(&[1.0, 2.0, 3.0, 4.0]);

        let mut opt_legacy = AdamW::new(
            vec![p_legacy.clone()],
            AdamWConfig {
                lr: 1e-2,
                weight_decay: 0.0,
                ..Default::default()
            },
        );
        let mut opt_foreach = AdamW::new(
            vec![p_foreach.clone()],
            AdamWConfig {
                lr: 1e-2,
                weight_decay: 0.0,
                foreach: true,
                ..Default::default()
            },
        );

        let g = leaf_grad_f32(&[0.5, -0.5, 0.25, -0.25]);
        p_legacy.set_grad(Some(g.clone())).unwrap();
        p_foreach.set_grad(Some(g)).unwrap();
        opt_legacy.step().unwrap();
        opt_foreach.step().unwrap();

        let l = opt_legacy.param_groups()[0].params[0]
            .data()
            .unwrap()
            .to_vec();
        let f = opt_foreach.param_groups()[0].params[0]
            .data()
            .unwrap()
            .to_vec();
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "foreach output {} differs from legacy {} by more than 1e-5",
                b,
                a
            );
        }
    }

    #[test]
    fn test_adamw_foreach_parity_with_weight_decay() {
        let (p_legacy, p_foreach) = paired(&[5.0, -3.0, 2.0]);
        let mut opt_legacy = AdamW::new(
            vec![p_legacy.clone()],
            AdamWConfig {
                lr: 5e-2,
                weight_decay: 0.05,
                ..Default::default()
            },
        );
        let mut opt_foreach = AdamW::new(
            vec![p_foreach.clone()],
            AdamWConfig {
                lr: 5e-2,
                weight_decay: 0.05,
                foreach: true,
                ..Default::default()
            },
        );

        for _ in 0..3 {
            let g = leaf_grad_f32(&[0.1, -0.2, 0.3]);
            p_legacy.set_grad(Some(g.clone())).unwrap();
            p_foreach.set_grad(Some(g)).unwrap();
            opt_legacy.step().unwrap();
            opt_foreach.step().unwrap();
        }

        let l = opt_legacy.param_groups()[0].params[0]
            .data()
            .unwrap()
            .to_vec();
        let f = opt_foreach.param_groups()[0].params[0]
            .data()
            .unwrap()
            .to_vec();
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "weight decay parity: legacy={}, foreach={}",
                a,
                b
            );
        }
    }

    #[test]
    fn test_adamw_foreach_multiple_steps_bias_correction() {
        // Bias correction matters most in early steps. Run 5 steps and
        // ensure parity throughout.
        let (p_legacy, p_foreach) = paired(&[0.0, 0.0]);
        let mut opt_legacy = AdamW::new(
            vec![p_legacy.clone()],
            AdamWConfig {
                lr: 1e-2,
                weight_decay: 0.0,
                ..Default::default()
            },
        );
        let mut opt_foreach = AdamW::new(
            vec![p_foreach.clone()],
            AdamWConfig {
                lr: 1e-2,
                weight_decay: 0.0,
                foreach: true,
                ..Default::default()
            },
        );

        for step in 1..=5 {
            let g = leaf_grad_f32(&[1.0, -1.0]);
            p_legacy.set_grad(Some(g.clone())).unwrap();
            p_foreach.set_grad(Some(g)).unwrap();
            opt_legacy.step().unwrap();
            opt_foreach.step().unwrap();

            let l = opt_legacy.param_groups()[0].params[0]
                .data()
                .unwrap()
                .to_vec();
            let f = opt_foreach.param_groups()[0].params[0]
                .data()
                .unwrap()
                .to_vec();
            for (a, b) in l.iter().zip(f.iter()) {
                assert!(
                    (a - b).abs() < 1e-4,
                    "step {} parity: legacy={}, foreach={}",
                    step,
                    a,
                    b
                );
            }
        }
    }

    #[test]
    fn test_adamw_foreach_skips_params_without_grad() {
        let p1 = Parameter::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let p2 = Parameter::from_slice(&[3.0f32, 4.0], &[2]).unwrap();
        let g = leaf_grad_f32(&[0.5, 0.5]);
        p1.set_grad(Some(g)).unwrap();
        // p2 has no grad.

        let mut opt = AdamW::new(
            vec![p1, p2],
            AdamWConfig {
                lr: 1e-1,
                foreach: true,
                ..Default::default()
            },
        );
        opt.step().unwrap();

        // p2 should be unchanged since it had no grad.
        let p2_data = opt.param_groups()[0].params[1].data().unwrap();
        assert!((p2_data[0] - 3.0).abs() < 1e-6);
        assert!((p2_data[1] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_adamw_foreach_convergence_quadratic() {
        // The foreach path should converge to the minimum of x^2 + y^2 just
        // like the legacy path. This is the test that was previously
        // commented out due to the reduce_grad_to_shape underflow bug
        // (now fixed in CL-498). It exercises the full backward chain
        // through pow + add on [1]-shaped parameters.
        use ferrotorch_core::grad_fns::{arithmetic::add as t_add, arithmetic::pow as t_pow};

        let x = Parameter::from_slice(&[3.0f32], &[1]).unwrap();
        let y = Parameter::from_slice(&[-4.0f32], &[1]).unwrap();
        let mut opt = AdamW::new(
            vec![x.clone(), y.clone()],
            AdamWConfig {
                lr: 1e-1,
                weight_decay: 0.0,
                foreach: true,
                ..Default::default()
            },
        );

        for _ in 0..400 {
            opt.zero_grad().unwrap();
            let xt = opt.param_groups[0].params[0].tensor().clone();
            let yt = opt.param_groups[0].params[1].tensor().clone();
            let loss = t_add(&t_pow(&xt, 2.0).unwrap(), &t_pow(&yt, 2.0).unwrap()).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        let xv = opt.param_groups()[0].params[0].data().unwrap()[0];
        let yv = opt.param_groups()[0].params[1].data().unwrap()[0];
        assert!(xv.abs() < 0.1, "expected x near 0, got {xv}");
        assert!(yv.abs() < 0.1, "expected y near 0, got {yv}");
    }

    #[test]
    fn test_adamw_foreach_long_run_drives_to_zero_with_zero_grad() {
        // Trivial sanity check: with zero gradients and weight decay only,
        // the foreach path's decoupled-weight-decay update should drive
        // the parameter monotonically toward zero. This catches gross
        // wiring bugs in the on-device update path without depending on
        // autograd through pow.
        let p = Parameter::from_slice(&[1.0f32, -1.0, 0.5], &[3]).unwrap();
        let mut opt = AdamW::new(
            vec![p.clone()],
            AdamWConfig {
                lr: 1e-1,
                weight_decay: 0.5,
                foreach: true,
                ..Default::default()
            },
        );

        let zero_grad = leaf_grad_f32(&[0.0, 0.0, 0.0]);
        let initial_norm = {
            let d = p.data().unwrap();
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        for _ in 0..50 {
            p.set_grad(Some(zero_grad.clone())).unwrap();
            opt.step().unwrap();
        }
        let final_norm = {
            let d = opt.param_groups()[0].params[0].data().unwrap();
            (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
        };
        assert!(
            final_norm < initial_norm * 0.1,
            "weight decay should shrink params: initial={initial_norm}, final={final_norm}"
        );
    }
}
