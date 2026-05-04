//! RAdam optimizer — Rectified Adam.
//!
//! Implements the algorithm from Liu et al., "On the Variance of the Adaptive
//! Learning Rate and Beyond" (ICLR 2020). RAdam automatically switches between
//! an adaptive learning rate (like Adam) and a non-adaptive one (like SGD with
//! momentum), based on the variance of the adaptive learning rate.
//!
//! When the approximated SMA length rho_t > 5, the variance is low enough to
//! use the adaptive update with a rectification term. Otherwise, it falls back
//! to a simple bias-corrected first-moment update.
//!
//! CL-319

use std::collections::HashMap;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, no_grad};
use ferrotorch_nn::Parameter;

use crate::foreach_utils::f64_scalar_on;
use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Hyperparameters for the [`RAdam`] optimizer.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RAdamConfig {
    /// Learning rate (default: 0.001).
    pub lr: f64,
    /// Exponential decay rates for the first and second moment estimates
    /// (default: `(0.9, 0.999)`).
    pub betas: (f64, f64),
    /// Term added to the denominator for numerical stability (default: 1e-8).
    pub eps: f64,
    /// Weight decay coefficient (default: 0.0).
    pub weight_decay: f64,
    /// Whether to apply decoupled weight decay (AdamW-style) instead of L2
    /// regularization (default: false).
    pub decoupled_weight_decay: bool,
    /// When `true`, use the on-device tensor-op update path. CL-497
    pub foreach: bool,
}

impl Default for RAdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            decoupled_weight_decay: false,
            foreach: false,
        }
    }
}

impl RAdamConfig {
    /// Set the learning rate.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the exponential decay rates for the first and second moment estimates.
    #[must_use]
    pub fn with_betas(mut self, betas: (f64, f64)) -> Self {
        self.betas = betas;
        self
    }

    /// Set the term added to the denominator for numerical stability.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the weight decay coefficient.
    #[must_use]
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable or disable decoupled weight decay (AdamW-style instead of L2).
    #[must_use]
    pub fn with_decoupled_weight_decay(mut self, decoupled_weight_decay: bool) -> Self {
        self.decoupled_weight_decay = decoupled_weight_decay;
        self
    }

    /// Enable or disable the on-device tensor-op (foreach) update path.
    #[must_use]
    pub fn with_foreach(mut self, foreach: bool) -> Self {
        self.foreach = foreach;
        self
    }
}

// ---------------------------------------------------------------------------
// Per-parameter state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RAdamParamState {
    step_count: u64,
    exp_avg: Vec<f64>,
    exp_avg_sq: Vec<f64>,
}

/// On-device foreach state for RAdam.
#[derive(Debug)]
struct RAdamForeachState<T: Float> {
    step_count: u64,
    exp_avg: Tensor<T>,
    exp_avg_sq: Tensor<T>,
}

// ---------------------------------------------------------------------------
// RAdam
// ---------------------------------------------------------------------------

/// Rectified Adam optimizer.
///
/// Automatically applies variance rectification when the approximated SMA
/// length rho_t exceeds 5, falling back to a simple momentum update otherwise.
#[derive(Debug)]
pub struct RAdam<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: RAdamConfig,
    state: HashMap<String, RAdamParamState>,
    /// Foreach (on-device) state. Used when `config.foreach == true`.
    foreach_state: HashMap<String, RAdamForeachState<T>>,
}

impl<T: Float> RAdam<T> {
    /// Create a new RAdam optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: RAdamConfig) -> Self {
        let mut group = ParamGroup::new(params, config.lr);
        group.weight_decay = config.weight_decay;
        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
            foreach_state: HashMap::new(),
        }
    }

    #[inline]
    fn param_key(group_idx: usize, param_idx: usize) -> String {
        format!("g{group_idx}_p{param_idx}")
    }

    /// Foreach (on-device, tensor-op) update path. CL-497
    fn step_foreach(&mut self) -> FerrotorchResult<()> {
        use ferrotorch_core::creation::zeros;
        use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, sqrt, sub};

        let config = self.config;
        let (beta1, beta2) = config.betas;
        let rho_inf = 2.0 / (1.0 - beta2) - 1.0;

        for gi in 0..self.param_groups.len() {
            let group_lr = self.param_groups[gi].lr;
            let group_wd = self.param_groups[gi].weight_decay;

            for pi in 0..self.param_groups[gi].params.len() {
                let grad_opt = self.param_groups[gi].params[pi].grad()?;
                let grad_tensor = match grad_opt {
                    Some(g) => g,
                    None => continue,
                };

                let param_t = self.param_groups[gi].params[pi].tensor().clone();
                let device = param_t.device();
                let key = Self::param_key(gi, pi);

                if !self.foreach_state.contains_key(&key) {
                    self.foreach_state.insert(
                        key.clone(),
                        RAdamForeachState {
                            step_count: 0,
                            exp_avg: zeros::<T>(param_t.shape())?.to(device)?,
                            exp_avg_sq: zeros::<T>(param_t.shape())?.to(device)?,
                        },
                    );
                }

                no_grad(|| {
                    // grad with L2 decay if not decoupled.
                    let mut grad: Tensor<T> = grad_tensor.clone();
                    if !config.decoupled_weight_decay && group_wd > 0.0 {
                        let wd_t = f64_scalar_on::<T>(group_wd, device)?;
                        let weighted = mul(&param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    let next_step = self.foreach_state[&key].step_count + 1;
                    let step_i32 = next_step as i32;
                    let bc1 = 1.0 - beta1.powi(step_i32);
                    let bc2 = 1.0 - beta2.powi(step_i32);
                    let rho_t = rho_inf - 2.0 * (next_step as f64) * beta2.powi(step_i32) / bc2;

                    let beta1_t = f64_scalar_on::<T>(beta1, device)?;
                    let one_minus_beta1 = f64_scalar_on::<T>(1.0 - beta1, device)?;
                    let beta2_t = f64_scalar_on::<T>(beta2, device)?;
                    let one_minus_beta2 = f64_scalar_on::<T>(1.0 - beta2, device)?;

                    // exp_avg and exp_avg_sq updates.
                    let exp_avg_old = self.foreach_state[&key].exp_avg.clone();
                    let exp_avg_new = add(
                        &mul(&exp_avg_old, &beta1_t)?,
                        &mul(&grad, &one_minus_beta1)?,
                    )?;

                    let exp_avg_sq_old = self.foreach_state[&key].exp_avg_sq.clone();
                    let grad_sq = mul(&grad, &grad)?;
                    let exp_avg_sq_new = add(
                        &mul(&exp_avg_sq_old, &beta2_t)?,
                        &mul(&grad_sq, &one_minus_beta2)?,
                    )?;

                    // Bias-corrected first moment: m_hat = exp_avg / bc1
                    let inv_bc1 = f64_scalar_on::<T>(1.0 / bc1, device)?;
                    let m_hat = mul(&exp_avg_new, &inv_bc1)?;

                    // Decoupled weight decay.
                    let decay_factor = if config.decoupled_weight_decay && group_wd > 0.0 {
                        1.0 - group_lr * group_wd
                    } else {
                        1.0
                    };
                    let decayed = if decay_factor != 1.0 {
                        let decay_t = f64_scalar_on::<T>(decay_factor, device)?;
                        mul(&param_t, &decay_t)?
                    } else {
                        param_t.clone()
                    };

                    let new_param = if rho_t > 5.0 {
                        // Adaptive rectified update.
                        let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                            / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                            .sqrt();
                        let sqrt_bc2 = bc2.sqrt();
                        let eps_t = f64_scalar_on::<T>(config.eps, device)?;
                        let denom = add(&sqrt(&exp_avg_sq_new)?, &eps_t)?;
                        // adaptive_lr = sqrt(bc2) / denom
                        let sqrt_bc2_t = f64_scalar_on::<T>(sqrt_bc2, device)?;
                        let adaptive_lr_num = mul(&m_hat, &sqrt_bc2_t)?;
                        let adaptive = div(&adaptive_lr_num, &denom)?;
                        let factor = f64_scalar_on::<T>(group_lr * rect, device)?;
                        let scaled = mul(&adaptive, &factor)?;
                        sub(&decayed, &scaled)?
                    } else {
                        // Fall back to un-adapted step: theta = theta - lr * m_hat
                        let lr_t = f64_scalar_on::<T>(group_lr, device)?;
                        let scaled = mul(&m_hat, &lr_t)?;
                        sub(&decayed, &scaled)?
                    };

                    // Commit.
                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: `update_storage` swaps the storage Arc; sole
                    // writer guarantees:
                    //  1. `step_foreach` runs under
                    //     `Optimizer::step(&mut self)` for this `Radam<T>`,
                    //     so no other thread/task is iterating these params.
                    //  2. The foreach loop body sits inside `no_grad`, so no
                    //     autograd `grad_fn` will retain a clone of
                    //     `param_t`'s storage Arc during this swap.
                    //  3. `param_t` is a per-iteration clone of the
                    //     parameter's `Tensor`; tensor temporaries
                    //     (`m_hat`, `decayed`, `scaled`, `new_param`) own
                    //     independent storage and `new_param` was just
                    //     consumed by `into_storage_and_shape`. No live
                    //     borrow into the parameter's existing storage
                    //     remains.
                    //  4. New `storage` was produced by ops dispatched on
                    //     `param_t`'s device, so device + numel match.
                    unsafe { param_t.update_storage(storage)? };

                    let state = self.foreach_state.get_mut(&key).unwrap();
                    state.step_count = next_step;
                    state.exp_avg = exp_avg_new;
                    state.exp_avg_sq = exp_avg_sq_new;

                    Ok::<(), ferrotorch_core::FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }
}

impl<T: Float> Optimizer<T> for RAdam<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        if self.config.foreach {
            return self.step_foreach();
        }

        let config = self.config;
        let (beta1, beta2) = config.betas;

        // Maximum length of the approximated SMA.
        let rho_inf = 2.0 / (1.0 - beta2) - 1.0;

        for gi in 0..self.param_groups.len() {
            let group_lr = self.param_groups[gi].lr;
            let group_wd = self.param_groups[gi].weight_decay;

            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];
                let tensor = param.tensor();

                let grad_tensor = match tensor.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                let key = Self::param_key(gi, pi);

                let param_data: Vec<f64> = tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                let mut grad_data: Vec<f64> = grad_tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;

                let numel = param_data.len();

                // Decoupled weight decay: applied directly to parameters.
                let decay_factor = if config.decoupled_weight_decay && group_wd > 0.0 {
                    1.0 - group_lr * group_wd
                } else {
                    1.0
                };

                // L2 weight decay: added to gradient.
                if !config.decoupled_weight_decay && group_wd > 0.0 {
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g += group_wd * p;
                    }
                }

                let state = self.state.entry(key).or_insert_with(|| RAdamParamState {
                    step_count: 0,
                    exp_avg: vec![0.0; numel],
                    exp_avg_sq: vec![0.0; numel],
                });

                state.step_count += 1;
                let step = state.step_count;

                // Update first moment: m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                for (m, &g) in state.exp_avg.iter_mut().zip(grad_data.iter()) {
                    *m = beta1 * *m + (1.0 - beta1) * g;
                }

                // Update second moment: v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                for (v, &g) in state.exp_avg_sq.iter_mut().zip(grad_data.iter()) {
                    *v = beta2 * *v + (1.0 - beta2) * g * g;
                }

                // Bias correction for first moment.
                let bc1 = 1.0 - beta1.powi(step as i32);
                let bc2 = 1.0 - beta2.powi(step as i32);

                // Bias-corrected first moment.
                // m_hat = m_t / (1 - beta1^t)

                // Compute approximated SMA length.
                // rho_t = rho_inf - 2 * t * beta2^t / (1 - beta2^t)
                let rho_t = rho_inf - 2.0 * (step as f64) * beta2.powi(step as i32) / bc2;

                let new_values: Vec<T> = if rho_t > 5.0 {
                    // Variance is tractable: use adaptive learning rate with
                    // rectification term.
                    //
                    // r_t = sqrt( (rho_t - 4)(rho_t - 2) * rho_inf /
                    //             ((rho_inf - 4)(rho_inf - 2) * rho_t) )
                    // l_t = sqrt(1 - beta2^t) / (sqrt(v_t) + eps)
                    // theta_t = theta_t - lr * m_hat * r_t * l_t
                    let rect = ((rho_t - 4.0) * (rho_t - 2.0) * rho_inf
                        / ((rho_inf - 4.0) * (rho_inf - 2.0) * rho_t))
                        .sqrt();

                    (0..numel)
                        .map(|i| {
                            let m_hat = state.exp_avg[i] / bc1;
                            let adaptive_lr =
                                bc2.sqrt() / (state.exp_avg_sq[i].sqrt() + config.eps);
                            let decayed = param_data[i] * decay_factor;
                            let updated = decayed - group_lr * m_hat * rect * adaptive_lr;
                            cast::<f64, T>(updated)
                        })
                        .collect::<FerrotorchResult<Vec<T>>>()?
                } else {
                    // Variance is not tractable: fall back to un-adapted step.
                    // theta_t = theta_t - lr * m_hat
                    (0..numel)
                        .map(|i| {
                            let m_hat = state.exp_avg[i] / bc1;
                            let decayed = param_data[i] * decay_factor;
                            let updated = decayed - group_lr * m_hat;
                            cast::<f64, T>(updated)
                        })
                        .collect::<FerrotorchResult<Vec<T>>>()?
                };

                // SAFETY: `update_data` writes through `Arc::as_ptr` and
                // requires sole-writer access to the parameter's storage.
                //  1. `Radam::step(&mut self)` is the unique mutable handle
                //     to this optimiser; the per-(gi, pi) loop is sequential.
                //  2. The `no_grad` closure prevents `grad_fn` recording, so
                //     no autograd node will clone the storage Arc.
                //  3. The earlier `tensor.data_vec()` and
                //     `grad_tensor.data_vec()` reads returned owned
                //     `Vec<f64>` values that have been consumed. `new_values`
                //     is a fresh owned `Vec<T>`.
                no_grad(|| unsafe { param.tensor().update_data(&new_values) })?;
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

    fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
        let mut out = OptimizerState::new();
        for (key, ps) in &self.state {
            let mut entry = HashMap::new();
            entry.insert("step_count".to_string(), vec![ps.step_count as f64]);
            entry.insert("exp_avg".to_string(), ps.exp_avg.clone());
            entry.insert("exp_avg_sq".to_string(), ps.exp_avg_sq.clone());
            out.insert(key.clone(), entry);
        }
        Ok(out)
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
                RAdamParamState {
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

    fn scalar_param(val: f64) -> Parameter<f64> {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap();
        Parameter::new(t)
    }

    fn param_val(opt: &RAdam<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    #[test]
    fn test_radam_convergence_quadratic() {
        let p = scalar_param(5.0);
        let mut opt = RAdam::new(
            vec![p],
            RAdamConfig {
                lr: 0.01,
                ..Default::default()
            },
        );

        for _ in 0..3000 {
            opt.zero_grad().unwrap();
            let t = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&t, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        let val = param_val(&opt, 0, 0);
        assert!(val.abs() < 0.1, "expected near 0, got {val}");
    }

    #[test]
    fn test_radam_convergence_two_params() {
        let px = scalar_param(3.0);
        let py = scalar_param(-2.0);

        let mut opt = RAdam::new(
            vec![px, py],
            RAdamConfig {
                lr: 0.01,
                ..Default::default()
            },
        );

        for _ in 0..2000 {
            opt.zero_grad().unwrap();
            let x = opt.param_groups[0].params[0].tensor().clone();
            let y = opt.param_groups[0].params[1].tensor().clone();
            let loss = add(&pow(&x, 2.0).unwrap(), &pow(&y, 2.0).unwrap()).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        let vx = param_val(&opt, 0, 0);
        let vy = param_val(&opt, 0, 1);
        assert!(vx.abs() < 0.1, "expected x near 0, got {vx}");
        assert!(vy.abs() < 0.1, "expected y near 0, got {vy}");
    }

    #[test]
    fn test_radam_zero_grad() {
        let p = scalar_param(1.0);
        let mut opt = RAdam::new(vec![p], RAdamConfig::default());

        let grad = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        opt.param_groups[0].params[0]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();
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

    #[test]
    fn test_radam_state_dict_roundtrip() {
        let p = scalar_param(2.0);
        let mut opt = RAdam::new(vec![p], RAdamConfig::default());

        for _ in 0..3 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        let saved = opt
            .state_dict()
            .expect("radam state_dict must succeed in test");
        assert!(!saved.is_empty());

        let key = RAdam::<f64>::param_key(0, 0);
        assert_eq!(saved[&key]["step_count"][0] as u64, 3);

        let p2 = scalar_param(2.0);
        let mut opt2 = RAdam::new(vec![p2], RAdamConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2
            .state_dict()
            .expect("radam state_dict round-trip must succeed in test");
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["exp_avg"], saved[&key]["exp_avg"]);
    }

    #[test]
    fn test_radam_lr_accessors() {
        let p = scalar_param(1.0);
        let mut opt = RAdam::new(
            vec![p],
            RAdamConfig {
                lr: 0.05,
                ..Default::default()
            },
        );
        assert!((opt.lr() - 0.05).abs() < 1e-12);
        opt.set_lr(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_radam_rectification_kicks_in() {
        // After enough steps, rho_t should exceed 5 and the adaptive path
        // should be used. We verify by checking the parameter actually moves.
        let p = scalar_param(10.0);
        let mut opt = RAdam::new(
            vec![p],
            RAdamConfig {
                lr: 0.01,
                ..Default::default()
            },
        );

        let grad = Tensor::from_storage(TensorStorage::cpu(vec![2.0_f64]), vec![], false).unwrap();
        opt.param_groups[0].params[0]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();

        // Step several times to build up enough history for rho_t > 5.
        for _ in 0..10 {
            opt.step().unwrap();
            // Re-set gradient for next step.
            let grad =
                Tensor::from_storage(TensorStorage::cpu(vec![2.0_f64]), vec![], false).unwrap();
            opt.param_groups[0].params[0]
                .tensor()
                .set_grad(Some(grad))
                .unwrap();
        }

        let val = param_val(&opt, 0, 0);
        assert!(val < 10.0, "param should have decreased, got {val}");
    }

    #[test]
    fn test_radam_weight_decay() {
        let p = scalar_param(5.0);
        let mut opt = RAdam::new(
            vec![p],
            RAdamConfig {
                lr: 0.01,
                weight_decay: 0.1,
                ..Default::default()
            },
        );

        for _ in 0..100 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        let val = param_val(&opt, 0, 0);
        assert!(val.abs() < 5.0, "should have moved from 5.0, got {val}");
    }

    // -----------------------------------------------------------------------
    // Foreach mode parity tests. CL-497
    // -----------------------------------------------------------------------

    fn paired_radam(data: &[f32]) -> (Parameter<f32>, Parameter<f32>) {
        (
            Parameter::from_slice(data, &[data.len()]).unwrap(),
            Parameter::from_slice(data, &[data.len()]).unwrap(),
        )
    }

    fn radam_run_pair(
        cfg: RAdamConfig,
        init: &[f32],
        steps: usize,
        grad: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let (p_legacy, p_foreach) = paired_radam(init);
        let mut legacy = RAdam::new(vec![p_legacy.clone()], cfg);
        let mut foreach = RAdam::new(
            vec![p_foreach.clone()],
            RAdamConfig {
                foreach: true,
                ..cfg
            },
        );

        for _ in 0..steps {
            let g =
                Tensor::from_storage(TensorStorage::cpu(grad.to_vec()), vec![init.len()], false)
                    .unwrap();
            p_legacy.set_grad(Some(g.clone())).unwrap();
            p_foreach.set_grad(Some(g)).unwrap();
            legacy.step().unwrap();
            foreach.step().unwrap();
        }

        (
            legacy.param_groups()[0].params[0].data().unwrap().to_vec(),
            foreach.param_groups()[0].params[0].data().unwrap().to_vec(),
        )
    }

    #[test]
    fn test_radam_foreach_basic_parity() {
        // Run enough steps to cross the rho_t > 5 boundary and exercise the
        // rectified path (rho_inf ≈ 1999 for beta2=0.999).
        let (l, f) = radam_run_pair(
            RAdamConfig::default(),
            &[1.0, 2.0, 3.0, 4.0],
            10,
            &[0.1, 0.2, -0.3, 0.4],
        );
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "radam foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_radam_foreach_parity_with_decoupled_wd() {
        let cfg = RAdamConfig {
            lr: 0.01,
            weight_decay: 0.05,
            decoupled_weight_decay: true,
            ..Default::default()
        };
        let (l, f) = radam_run_pair(cfg, &[5.0, -3.0, 2.0], 8, &[0.5, -0.5, 1.0]);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "radam decoupled wd parity: legacy={a}, foreach={b}"
            );
        }
    }
}
