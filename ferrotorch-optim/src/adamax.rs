//! Adamax optimizer — Adam variant using L-infinity norm.
//!
//! Implements the Adamax algorithm from Kingma & Ba, "Adam: A Method for
//! Stochastic Optimization" (ICLR 2015), Section 7. Instead of the L2 norm
//! used by Adam's second moment, Adamax uses the L-infinity norm, which
//! makes it more robust to sparse gradients and gradient outliers.
//!
//! CL-319

use std::collections::HashMap;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, no_grad};
use ferrotorch_nn::Parameter;

use crate::foreach_utils::{elemwise_max, f64_scalar_on};
use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Hyperparameters for the [`Adamax`] optimizer.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AdamaxConfig {
    /// Learning rate (default: 0.002).
    pub lr: f64,
    /// Exponential decay rates for the first moment and infinity norm
    /// (default: `(0.9, 0.999)`).
    pub betas: (f64, f64),
    /// Term added to the denominator for numerical stability (default: 1e-8).
    pub eps: f64,
    /// Weight decay coefficient (default: 0.0).
    pub weight_decay: f64,
    /// When `true`, use the on-device tensor-op update path. CL-497
    pub foreach: bool,
}

impl Default for AdamaxConfig {
    fn default() -> Self {
        Self {
            lr: 2e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            foreach: false,
        }
    }
}

impl AdamaxConfig {
    /// Set the learning rate.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the exponential decay rates for the first moment and infinity norm.
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
struct AdamaxParamState {
    step_count: u64,
    /// First moment estimate (exponential moving average of gradients).
    exp_avg: Vec<f64>,
    /// Exponentially weighted infinity norm.
    exp_inf: Vec<f64>,
}

/// On-device foreach state for Adamax.
#[derive(Debug)]
struct AdamaxForeachState<T: Float> {
    step_count: u64,
    exp_avg: Tensor<T>,
    exp_inf: Tensor<T>,
}

// ---------------------------------------------------------------------------
// Adamax
// ---------------------------------------------------------------------------

/// Adamax optimizer — Adam variant using the L-infinity norm.
///
/// # Algorithm
///
/// For each parameter `p` with gradient `g`:
///
/// 1. `m_t = beta1 * m_{t-1} + (1 - beta1) * g_t`
/// 2. `u_t = max(beta2 * u_{t-1}, |g_t| + eps)`
/// 3. `p_t = p_{t-1} - lr / (1 - beta1^t) * m_t / u_t`
///
/// The infinity norm `u_t` replaces Adam's second-moment estimate, eliminating
/// the need for bias correction on the second moment.
#[derive(Debug)]
pub struct Adamax<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdamaxConfig,
    state: HashMap<String, AdamaxParamState>,
    /// Foreach (on-device) state. Used when `config.foreach == true`.
    foreach_state: HashMap<String, AdamaxForeachState<T>>,
}

impl<T: Float> Adamax<T> {
    /// Create a new Adamax optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: AdamaxConfig) -> Self {
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
        use ferrotorch_core::grad_fns::arithmetic::{abs, add, div, mul, sub};

        let config = self.config;
        let (beta1, beta2) = config.betas;

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

                // Lazy-init state.
                if !self.foreach_state.contains_key(&key) {
                    self.foreach_state.insert(
                        key.clone(),
                        AdamaxForeachState {
                            step_count: 0,
                            exp_avg: zeros::<T>(param_t.shape())?.to(device)?,
                            exp_inf: zeros::<T>(param_t.shape())?.to(device)?,
                        },
                    );
                }

                no_grad(|| {
                    // grad (with L2 weight decay if enabled).
                    let mut grad: Tensor<T> = grad_tensor.clone();
                    if group_wd > 0.0 {
                        let wd_t = f64_scalar_on::<T>(group_wd, device)?;
                        let weighted = mul(&param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    let beta1_t = f64_scalar_on::<T>(beta1, device)?;
                    let one_minus_beta1 = f64_scalar_on::<T>(1.0 - beta1, device)?;
                    let beta2_t = f64_scalar_on::<T>(beta2, device)?;
                    let eps_t = f64_scalar_on::<T>(config.eps, device)?;

                    // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                    let exp_avg_old = self.foreach_state[&key].exp_avg.clone();
                    let exp_avg_new = add(
                        &mul(&exp_avg_old, &beta1_t)?,
                        &mul(&grad, &one_minus_beta1)?,
                    )?;

                    // exp_inf = max(beta2 * exp_inf, |grad| + eps)
                    let exp_inf_old = self.foreach_state[&key].exp_inf.clone();
                    let beta2_scaled_inf = mul(&exp_inf_old, &beta2_t)?;
                    let abs_grad = abs(&grad)?;
                    let abs_grad_plus_eps = add(&abs_grad, &eps_t)?;
                    let exp_inf_new = elemwise_max(&beta2_scaled_inf, &abs_grad_plus_eps, device)?;

                    // Bias correction for the first moment only.
                    let next_step = self.foreach_state[&key].step_count + 1;
                    let bc1 = 1.0 - beta1.powi(next_step as i32);
                    let clr = group_lr / bc1;
                    let clr_t = f64_scalar_on::<T>(clr, device)?;

                    // param = param - clr * exp_avg / exp_inf
                    let update = div(&exp_avg_new, &exp_inf_new)?;
                    let scaled = mul(&update, &clr_t)?;
                    let new_param = sub(&param_t, &scaled)?;

                    // Commit.
                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: `update_storage` requires sole-writer
                    // semantics on the parameter's storage Arc.
                    //  1. `step_foreach` is invoked from `Optimizer::step`
                    //     under `&mut self`, ruling out concurrent steps.
                    //  2. The whole foreach body is wrapped in `no_grad`, so
                    //     no `grad_fn` retains a clone of `param_t`'s
                    //     storage Arc during this swap.
                    //  3. `param_t` is a per-iteration clone of the
                    //     parameter's `Tensor`; the temporaries derived
                    //     from it (`update`, `scaled`, `new_param`) own
                    //     independent storage, and `new_param` was just
                    //     consumed by `into_storage_and_shape`. No live
                    //     borrow into the parameter's existing storage
                    //     remains.
                    //  4. The new `storage` was produced by ops on
                    //     `param_t`'s device with shape preserved — same
                    //     device and numel as the existing tensor view.
                    unsafe { param_t.update_storage(storage)? };

                    let state = self.foreach_state.get_mut(&key).unwrap();
                    state.step_count = next_step;
                    state.exp_avg = exp_avg_new;
                    state.exp_inf = exp_inf_new;

                    Ok::<(), ferrotorch_core::FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }
}

impl<T: Float> Optimizer<T> for Adamax<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
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

                // L2 weight decay.
                if group_wd > 0.0 {
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g += group_wd * p;
                    }
                }

                let state = self.state.entry(key).or_insert_with(|| AdamaxParamState {
                    step_count: 0,
                    exp_avg: vec![0.0; numel],
                    exp_inf: vec![0.0; numel],
                });

                state.step_count += 1;
                let step = state.step_count;

                // Update biased first moment estimate.
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                for (m, &g) in state.exp_avg.iter_mut().zip(grad_data.iter()) {
                    *m = beta1 * *m + (1.0 - beta1) * g;
                }

                // Update the exponentially weighted infinity norm.
                // u_t = max(beta2 * u_{t-1}, |g_t| + eps)
                for (u, &g) in state.exp_inf.iter_mut().zip(grad_data.iter()) {
                    *u = f64::max(beta2 * *u, g.abs() + config.eps);
                }

                // Bias correction only for the first moment.
                let bc1 = 1.0 - beta1.powi(step as i32);
                let clr = group_lr / bc1;

                // Update parameters.
                // p_t = p_{t-1} - clr * m_t / u_t
                let new_values: Vec<T> = (0..numel)
                    .map(|i| {
                        let updated = param_data[i] - clr * state.exp_avg[i] / state.exp_inf[i];
                        cast::<f64, T>(updated)
                    })
                    .collect::<FerrotorchResult<Vec<T>>>()?;

                // SAFETY: `update_data` mutates the parameter's storage via
                // `Arc::as_ptr`; safe here because:
                //  1. `Adamax::step(&mut self)` holds the only live mutable
                //     borrow on this optimiser; the per-(gi, pi) loop is
                //     sequential, so no two iterations alias the same
                //     parameter storage.
                //  2. The `no_grad` closure suppresses `grad_fn`
                //     construction during the write — no autograd node will
                //     clone the storage Arc as part of this update.
                //  3. The earlier reads from this parameter
                //     (`tensor.data_vec()` line 232 and
                //     `grad_tensor.data_vec()` line 237) returned owned
                //     `Vec<f64>` values that have already been consumed by
                //     the moment-update loops. `new_values` is a fresh
                //     `Vec<T>` independent of the parameter's storage.
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
            entry.insert("exp_inf".to_string(), ps.exp_inf.clone());
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

            let exp_inf =
                entry
                    .get("exp_inf")
                    .cloned()
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!("missing exp_inf in state for key {key}"),
                    })?;

            self.state.insert(
                key.clone(),
                AdamaxParamState {
                    step_count,
                    exp_avg,
                    exp_inf,
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

    fn param_val(opt: &Adamax<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    #[test]
    fn test_adamax_convergence_quadratic() {
        let p = scalar_param(5.0);
        let mut opt = Adamax::new(
            vec![p],
            AdamaxConfig {
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
    fn test_adamax_convergence_two_params() {
        let px = scalar_param(3.0);
        let py = scalar_param(-2.0);

        let mut opt = Adamax::new(
            vec![px, py],
            AdamaxConfig {
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
    fn test_adamax_zero_grad() {
        let p = scalar_param(1.0);
        let mut opt = Adamax::new(vec![p], AdamaxConfig::default());

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
    fn test_adamax_state_dict_roundtrip() {
        let p = scalar_param(2.0);
        let mut opt = Adamax::new(vec![p], AdamaxConfig::default());

        for _ in 0..3 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        let saved = opt
            .state_dict()
            .expect("adamax state_dict must succeed in test");
        let key = Adamax::<f64>::param_key(0, 0);
        assert_eq!(saved[&key]["step_count"][0] as u64, 3);
        assert!(saved[&key].contains_key("exp_inf"));

        let p2 = scalar_param(2.0);
        let mut opt2 = Adamax::new(vec![p2], AdamaxConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2
            .state_dict()
            .expect("adamax state_dict round-trip must succeed in test");
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["exp_inf"], saved[&key]["exp_inf"]);
    }

    #[test]
    fn test_adamax_lr_accessors() {
        let p = scalar_param(1.0);
        let mut opt = Adamax::new(
            vec![p],
            AdamaxConfig {
                lr: 0.05,
                ..Default::default()
            },
        );
        assert!((opt.lr() - 0.05).abs() < 1e-12);
        opt.set_lr(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_adamax_default_config() {
        let config = AdamaxConfig::default();
        assert_eq!(config.lr, 2e-3);
        assert_eq!(config.betas, (0.9, 0.999));
        assert_eq!(config.eps, 1e-8);
        assert_eq!(config.weight_decay, 0.0);
    }

    #[test]
    fn test_adamax_weight_decay() {
        let p = scalar_param(5.0);
        let mut opt = Adamax::new(
            vec![p],
            AdamaxConfig {
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

    fn paired_adamax(data: &[f32]) -> (Parameter<f32>, Parameter<f32>) {
        (
            Parameter::from_slice(data, &[data.len()]).unwrap(),
            Parameter::from_slice(data, &[data.len()]).unwrap(),
        )
    }

    fn adamax_run_pair(
        cfg: AdamaxConfig,
        init: &[f32],
        steps: usize,
        grad: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let (p_legacy, p_foreach) = paired_adamax(init);
        let mut legacy = Adamax::new(vec![p_legacy.clone()], cfg);
        let mut foreach = Adamax::new(
            vec![p_foreach.clone()],
            AdamaxConfig {
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
    fn test_adamax_foreach_basic_parity() {
        let (l, f) = adamax_run_pair(
            AdamaxConfig::default(),
            &[1.0, 2.0, 3.0, 4.0],
            5,
            &[0.1, 0.2, -0.3, 0.4],
        );
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "adamax foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_adamax_foreach_parity_with_weight_decay() {
        let cfg = AdamaxConfig {
            lr: 0.002,
            weight_decay: 0.05,
            ..Default::default()
        };
        let (l, f) = adamax_run_pair(cfg, &[5.0, -3.0, 2.0], 6, &[0.5, -0.5, 1.0]);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "adamax wd parity: legacy={a}, foreach={b}"
            );
        }
    }
}
