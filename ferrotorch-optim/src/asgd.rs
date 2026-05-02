//! ASGD optimizer — Averaged Stochastic Gradient Descent.
//!
//! Implements the algorithm from Polyak & Juditsky, "Acceleration of
//! Stochastic Approximation by Averaging" (1992). ASGD maintains a running
//! average of parameters that often generalizes better than the last iterate.
//!
//! The effective learning rate decays as eta_t = lr / (1 + lambd * lr * t)^alpha,
//! and parameter averaging begins after step t0.
//!
//! CL-319

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, no_grad};
use ferrotorch_nn::Parameter;

use crate::foreach_utils::f64_scalar_on;
use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Hyperparameters for the [`Asgd`] optimizer.
#[derive(Debug, Clone, Copy)]
pub struct AsgdConfig {
    /// Learning rate (default: 0.01).
    pub lr: f64,
    /// Decay term (default: 1e-4).
    pub lambd: f64,
    /// Power for eta update (default: 0.75).
    pub alpha: f64,
    /// Point at which to start averaging (default: 1e6).
    pub t0: f64,
    /// Weight decay coefficient (default: 0.0).
    pub weight_decay: f64,
    /// When `true`, use the on-device tensor-op update path. CL-497
    pub foreach: bool,
}

impl Default for AsgdConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            lambd: 1e-4,
            alpha: 0.75,
            t0: 1e6,
            weight_decay: 0.0,
            foreach: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-parameter state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct AsgdParamState {
    step_count: u64,
    /// Effective learning rate: eta_t = lr / (1 + lambd * lr * t)^alpha.
    eta: f64,
    /// Averaging coefficient: mu_t = 1 / max(1, t - t0).
    mu: f64,
    /// Running average of parameters.
    ax: Vec<f64>,
}

/// On-device foreach state for ASGD.
#[derive(Debug)]
struct AsgdForeachState<T: Float> {
    step_count: u64,
    eta: f64,
    mu: f64,
    /// Running average of parameters as a device-resident tensor.
    ax: Tensor<T>,
}

// ---------------------------------------------------------------------------
// Asgd
// ---------------------------------------------------------------------------

/// Averaged Stochastic Gradient Descent optimizer.
///
/// # Algorithm
///
/// For each parameter `p` with gradient `g`:
///
/// 1. `g = g + weight_decay * p` (if weight_decay != 0)
/// 2. `p = p * (1 - lambd * eta) - eta * g`
/// 3. Averaging: `ax = ax + mu * (p - ax)`
/// 4. Update eta: `eta = lr / (1 + lambd * lr * t)^alpha`
/// 5. Update mu: `mu = 1 / max(1, t - t0)`
///
/// After training, use [`Asgd::averaged_params`] to get the averaged
/// parameter values which typically generalize better.
#[derive(Debug)]
pub struct Asgd<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AsgdConfig,
    state: HashMap<String, AsgdParamState>,
    /// Foreach (on-device) state. Used when `config.foreach == true`.
    foreach_state: HashMap<String, AsgdForeachState<T>>,
}

impl<T: Float> Asgd<T> {
    /// Create a new ASGD optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: AsgdConfig) -> Self {
        let mut group = ParamGroup::new(params, config.lr);
        group.weight_decay = config.weight_decay;
        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
            foreach_state: HashMap::new(),
        }
    }

    /// Get the averaged parameter value for a given group/param index.
    ///
    /// Returns `None` if the parameter has no state yet (no steps taken).
    pub fn averaged_param(&self, group_idx: usize, param_idx: usize) -> Option<&[f64]> {
        let key = Self::param_key(group_idx, param_idx);
        self.state.get(&key).map(|s| s.ax.as_slice())
    }

    #[inline]
    fn param_key(group_idx: usize, param_idx: usize) -> String {
        format!("g{group_idx}_p{param_idx}")
    }

    /// Foreach (on-device, tensor-op) update path. CL-497
    fn step_foreach(&mut self) -> FerrotorchResult<()> {
        use ferrotorch_core::grad_fns::arithmetic::{add, mul, sub};

        let config = self.config;

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

                // Lazy-init state. ax starts as a copy of the parameter.
                if !self.foreach_state.contains_key(&key) {
                    self.foreach_state.insert(
                        key.clone(),
                        AsgdForeachState {
                            step_count: 0,
                            eta: group_lr,
                            mu: 1.0,
                            ax: param_t.clone(),
                        },
                    );
                }

                no_grad(|| {
                    // grad with L2 decay.
                    let mut grad: Tensor<T> = grad_tensor.clone();
                    if group_wd > 0.0 {
                        let wd_t = f64_scalar_on::<T>(group_wd, device)?;
                        let weighted = mul(&param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    let (eta, mu) = {
                        let s = &self.foreach_state[&key];
                        (s.eta, s.mu)
                    };

                    // p_new = p * (1 - lambd * eta) - eta * g
                    let shrink_factor = f64_scalar_on::<T>(1.0 - config.lambd * eta, device)?;
                    let eta_t = f64_scalar_on::<T>(eta, device)?;
                    let shrunk = mul(&param_t, &shrink_factor)?;
                    let scaled_grad = mul(&grad, &eta_t)?;
                    let new_param = sub(&shrunk, &scaled_grad)?;

                    // Update ax = ax + mu * (p_new - ax) if mu != 1, else ax = p_new.
                    let ax_new = if mu != 1.0 {
                        let ax_old = self.foreach_state[&key].ax.clone();
                        let diff = sub(&new_param, &ax_old)?;
                        let mu_t = f64_scalar_on::<T>(mu, device)?;
                        let scaled_diff = mul(&diff, &mu_t)?;
                        add(&ax_old, &scaled_diff)?
                    } else {
                        new_param.clone()
                    };

                    // Commit param.
                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: optimizer step inside no_grad, exclusive access.
                    unsafe { param_t.update_storage(storage)? };

                    // Commit state + schedule updates.
                    let next_step = self.foreach_state[&key].step_count + 1;
                    let step = next_step as f64;
                    let new_eta =
                        group_lr / (1.0 + config.lambd * group_lr * step).powf(config.alpha);
                    let new_mu = 1.0 / f64::max(1.0, step - config.t0);

                    let state = self.foreach_state.get_mut(&key).unwrap();
                    state.step_count = next_step;
                    state.ax = ax_new;
                    state.eta = new_eta;
                    state.mu = new_mu;

                    Ok::<(), ferrotorch_core::FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }
}

impl<T: Float> Optimizer<T> for Asgd<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        if self.config.foreach {
            return self.step_foreach();
        }

        let config = self.config;

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
                    .map(|&v| v.to_f64().unwrap())
                    .collect();
                let mut grad_data: Vec<f64> = grad_tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| v.to_f64().unwrap())
                    .collect();

                let numel = param_data.len();

                // L2 weight decay.
                if group_wd > 0.0 {
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g += group_wd * p;
                    }
                }

                let state = self.state.entry(key).or_insert_with(|| AsgdParamState {
                    step_count: 0,
                    eta: group_lr,
                    mu: 1.0,
                    ax: param_data.clone(),
                });

                state.step_count += 1;
                let step = state.step_count as f64;
                let eta = state.eta;
                let mu = state.mu;

                // Update parameters.
                // p = p * (1 - lambd * eta) - eta * g
                let new_param_data: Vec<f64> = (0..numel)
                    .map(|i| param_data[i] * (1.0 - config.lambd * eta) - eta * grad_data[i])
                    .collect();

                // Update running average.
                // ax = ax + mu * (p_new - ax)
                if mu != 1.0 {
                    for (ax, &np) in state.ax.iter_mut().zip(new_param_data.iter()) {
                        *ax += mu * (np - *ax);
                    }
                } else {
                    state.ax = new_param_data.clone();
                }

                // Update eta: eta = lr / (1 + lambd * lr * t)^alpha
                state.eta = group_lr / (1.0 + config.lambd * group_lr * step).powf(config.alpha);

                // Update mu: mu = 1 / max(1, t - t0)
                state.mu = 1.0 / f64::max(1.0, step - config.t0);

                let new_values: Vec<T> = new_param_data
                    .iter()
                    .map(|&v| T::from(v).unwrap())
                    .collect();

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

    fn state_dict(&self) -> OptimizerState {
        let mut out = OptimizerState::new();
        for (key, ps) in &self.state {
            let mut entry = HashMap::new();
            entry.insert("step_count".to_string(), vec![ps.step_count as f64]);
            entry.insert("eta".to_string(), vec![ps.eta]);
            entry.insert("mu".to_string(), vec![ps.mu]);
            entry.insert("ax".to_string(), ps.ax.clone());
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

            let eta = entry
                .get("eta")
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(self.config.lr);

            let mu = entry
                .get("mu")
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(1.0);

            let ax = entry
                .get("ax")
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing ax in state for key {key}"),
                })?;

            self.state.insert(
                key.clone(),
                AsgdParamState {
                    step_count,
                    eta,
                    mu,
                    ax,
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

    fn param_val(opt: &Asgd<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    fn set_grad_scalar(opt: &Asgd<f64>, group: usize, idx: usize, val: f64) {
        let grad = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false).unwrap();
        opt.param_groups[group].params[idx]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();
    }

    #[test]
    fn test_asgd_convergence_quadratic() {
        let p = scalar_param(5.0);
        let mut opt = Asgd::new(
            vec![p],
            AsgdConfig {
                lr: 0.1,
                t0: 0.0,
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
        assert!(val.abs() < 0.5, "expected near 0, got {val}");
    }

    #[test]
    fn test_asgd_convergence_two_params() {
        let px = scalar_param(3.0);
        let py = scalar_param(-2.0);

        let mut opt = Asgd::new(
            vec![px, py],
            AsgdConfig {
                lr: 0.1,
                t0: 0.0,
                ..Default::default()
            },
        );

        for _ in 0..3000 {
            opt.zero_grad().unwrap();
            let x = opt.param_groups[0].params[0].tensor().clone();
            let y = opt.param_groups[0].params[1].tensor().clone();
            let loss = add(&pow(&x, 2.0).unwrap(), &pow(&y, 2.0).unwrap()).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        let vx = param_val(&opt, 0, 0);
        let vy = param_val(&opt, 0, 1);
        assert!(vx.abs() < 0.5, "expected x near 0, got {vx}");
        assert!(vy.abs() < 0.5, "expected y near 0, got {vy}");
    }

    #[test]
    fn test_asgd_eta_decay() {
        // Verify that the effective learning rate decreases over steps.
        let p = scalar_param(5.0);
        let mut opt = Asgd::new(vec![p], AsgdConfig::default());

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        let key = Asgd::<f64>::param_key(0, 0);
        let eta_1 = opt.state[&key].eta;

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        let eta_2 = opt.state[&key].eta;
        assert!(eta_2 < eta_1, "eta should decrease: {eta_1} -> {eta_2}");
    }

    #[test]
    fn test_asgd_averaging_starts_after_t0() {
        // With t0=2, averaging should start from step 3.
        let p = scalar_param(5.0);
        let mut opt = Asgd::new(
            vec![p],
            AsgdConfig {
                lr: 0.1,
                t0: 2.0,
                lambd: 0.0,
                alpha: 0.75,
                weight_decay: 0.0,
                foreach: false,
            },
        );

        // Steps 1 and 2: mu = 1/(max(1, t - 2)) = 1.0, so ax = param.
        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        let key = Asgd::<f64>::param_key(0, 0);

        // Step 3: t=3, t-t0=1, mu=1/max(1,1)=1.0 still
        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        // Step 4: t=4, t-t0=2, mu=0.5 -- averaging is blending
        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        let mu = opt.state[&key].mu;
        assert!(mu < 1.0, "mu should be less than 1 after t0, got {mu}");
    }

    #[test]
    fn test_asgd_averaged_params() {
        let p = scalar_param(5.0);
        let mut opt = Asgd::new(
            vec![p],
            AsgdConfig {
                lr: 0.1,
                t0: 0.0,
                ..Default::default()
            },
        );

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        let averaged = opt.averaged_param(0, 0);
        assert!(
            averaged.is_some(),
            "averaged params should exist after step"
        );
    }

    #[test]
    fn test_asgd_zero_grad() {
        let p = scalar_param(1.0);
        let mut opt = Asgd::new(vec![p], AsgdConfig::default());

        let grad = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        opt.param_groups[0].params[0]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();

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
    fn test_asgd_state_dict_roundtrip() {
        let p = scalar_param(2.0);
        let mut opt = Asgd::new(vec![p], AsgdConfig::default());

        for _ in 0..3 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        let saved = opt.state_dict();
        let key = Asgd::<f64>::param_key(0, 0);
        assert_eq!(saved[&key]["step_count"][0] as u64, 3);
        assert!(saved[&key].contains_key("eta"));
        assert!(saved[&key].contains_key("mu"));
        assert!(saved[&key].contains_key("ax"));

        let p2 = scalar_param(2.0);
        let mut opt2 = Asgd::new(vec![p2], AsgdConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2.state_dict();
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["eta"], saved[&key]["eta"]);
        assert_eq!(loaded[&key]["mu"], saved[&key]["mu"]);
        assert_eq!(loaded[&key]["ax"], saved[&key]["ax"]);
    }

    #[test]
    fn test_asgd_lr_accessors() {
        let p = scalar_param(1.0);
        let mut opt = Asgd::new(
            vec![p],
            AsgdConfig {
                lr: 0.05,
                ..Default::default()
            },
        );
        assert!((opt.lr() - 0.05).abs() < 1e-12);
        opt.set_lr(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_asgd_default_config() {
        let config = AsgdConfig::default();
        assert_eq!(config.lr, 1e-2);
        assert_eq!(config.lambd, 1e-4);
        assert_eq!(config.alpha, 0.75);
        assert_eq!(config.t0, 1e6);
        assert_eq!(config.weight_decay, 0.0);
    }

    #[test]
    fn test_asgd_weight_decay() {
        let p = scalar_param(5.0);
        let mut opt = Asgd::new(
            vec![p],
            AsgdConfig {
                lr: 0.1,
                weight_decay: 0.01,
                t0: 0.0,
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

    #[test]
    fn test_asgd_foreach_basic_parity() {
        let p_legacy = Parameter::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let p_foreach = Parameter::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

        let mut legacy = Asgd::new(vec![p_legacy.clone()], AsgdConfig::default());
        let mut foreach = Asgd::new(
            vec![p_foreach.clone()],
            AsgdConfig {
                foreach: true,
                ..Default::default()
            },
        );

        for _ in 0..5 {
            let g = Tensor::from_storage(
                TensorStorage::cpu(vec![0.1f32, 0.2, -0.3, 0.4]),
                vec![4],
                false,
            )
            .unwrap();
            p_legacy.set_grad(Some(g.clone())).unwrap();
            p_foreach.set_grad(Some(g)).unwrap();
            legacy.step().unwrap();
            foreach.step().unwrap();
        }

        let l = legacy.param_groups()[0].params[0].data().unwrap().to_vec();
        let f = foreach.param_groups()[0].params[0].data().unwrap().to_vec();
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "asgd foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_asgd_foreach_parity_with_weight_decay_and_averaging() {
        // t0 = 0 so averaging kicks in immediately (mu < 1).
        let p_legacy = Parameter::from_slice(&[2.0f32, -1.0, 0.5], &[3]).unwrap();
        let p_foreach = Parameter::from_slice(&[2.0f32, -1.0, 0.5], &[3]).unwrap();

        let cfg = AsgdConfig {
            lr: 0.05,
            weight_decay: 0.02,
            t0: 0.0,
            ..Default::default()
        };
        let mut legacy = Asgd::new(vec![p_legacy.clone()], cfg);
        let mut foreach = Asgd::new(
            vec![p_foreach.clone()],
            AsgdConfig {
                foreach: true,
                ..cfg
            },
        );

        for _ in 0..6 {
            let g =
                Tensor::from_storage(TensorStorage::cpu(vec![0.3f32, -0.2, 0.1]), vec![3], false)
                    .unwrap();
            p_legacy.set_grad(Some(g.clone())).unwrap();
            p_foreach.set_grad(Some(g)).unwrap();
            legacy.step().unwrap();
            foreach.step().unwrap();
        }

        let l = legacy.param_groups()[0].params[0].data().unwrap().to_vec();
        let f = foreach.param_groups()[0].params[0].data().unwrap().to_vec();
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "asgd wd+avg parity: legacy={a}, foreach={b}"
            );
        }
    }
}
