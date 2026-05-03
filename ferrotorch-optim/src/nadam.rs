//! NAdam optimizer — Nesterov-accelerated Adam.
//!
//! Implements the algorithm from Dozat, "Incorporating Nesterov Momentum into
//! Adam" (ICLR 2016 Workshop). NAdam modifies Adam to use Nesterov momentum
//! in the first-moment estimate, yielding faster convergence on some problems.
//!
//! The key difference from Adam: the parameter update uses a lookahead
//! first-moment estimate that combines the current gradient with the
//! next-step momentum, weighted by a schedule-aware mu_t.
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

/// Hyperparameters for the [`NAdam`] optimizer.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct NAdamConfig {
    /// Learning rate (default: 0.002).
    pub lr: f64,
    /// Exponential decay rates for the first and second moment estimates
    /// (default: `(0.9, 0.999)`).
    pub betas: (f64, f64),
    /// Term added to the denominator for numerical stability (default: 1e-8).
    pub eps: f64,
    /// Weight decay coefficient (default: 0.0).
    pub weight_decay: f64,
    /// Momentum decay for the mu schedule (default: 4e-3).
    pub momentum_decay: f64,
    /// Whether to apply decoupled weight decay (AdamW-style) instead of L2
    /// regularization (default: false).
    pub decoupled_weight_decay: bool,
    /// When `true`, use the on-device tensor-op update path. CL-497
    pub foreach: bool,
}

impl Default for NAdamConfig {
    fn default() -> Self {
        Self {
            lr: 2e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            momentum_decay: 4e-3,
            decoupled_weight_decay: false,
            foreach: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-parameter state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct NAdamParamState {
    step_count: u64,
    /// Cumulative product of mu schedule values.
    mu_product: f64,
    exp_avg: Vec<f64>,
    exp_avg_sq: Vec<f64>,
}

/// On-device foreach state for NAdam.
#[derive(Debug)]
struct NAdamForeachState<T: Float> {
    step_count: u64,
    mu_product: f64,
    exp_avg: Tensor<T>,
    exp_avg_sq: Tensor<T>,
}

// ---------------------------------------------------------------------------
// NAdam
// ---------------------------------------------------------------------------

/// Nesterov-accelerated Adam optimizer.
///
/// Combines Adam's adaptive learning rate with Nesterov's lookahead momentum
/// trick, using a per-step mu schedule: mu_t = beta1 * (1 - 0.5 * 0.96^(t * psi)).
#[derive(Debug)]
pub struct NAdam<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: NAdamConfig,
    state: HashMap<String, NAdamParamState>,
    /// Foreach (on-device) state. Used when `config.foreach == true`.
    foreach_state: HashMap<String, NAdamForeachState<T>>,
}

impl<T: Float> NAdam<T> {
    /// Create a new NAdam optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: NAdamConfig) -> Self {
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
        let psi = config.momentum_decay;

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
                        NAdamForeachState {
                            step_count: 0,
                            mu_product: 1.0,
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
                    let step = next_step as f64;

                    // mu schedule.
                    let mu = beta1 * (1.0 - 0.5 * 0.96_f64.powf(step * psi));
                    let mu_next = beta1 * (1.0 - 0.5 * 0.96_f64.powf((step + 1.0) * psi));
                    let mu_product = self.foreach_state[&key].mu_product * mu;
                    let mu_product_next = mu_product * mu_next;
                    let bc2 = 1.0 - beta2.powi(next_step as i32);

                    let beta1_t = f64_scalar_on::<T>(beta1, device)?;
                    let one_minus_beta1 = f64_scalar_on::<T>(1.0 - beta1, device)?;
                    let beta2_t = f64_scalar_on::<T>(beta2, device)?;
                    let one_minus_beta2 = f64_scalar_on::<T>(1.0 - beta2, device)?;

                    // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                    let exp_avg_old = self.foreach_state[&key].exp_avg.clone();
                    let exp_avg_new = add(
                        &mul(&exp_avg_old, &beta1_t)?,
                        &mul(&grad, &one_minus_beta1)?,
                    )?;

                    // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                    let exp_avg_sq_old = self.foreach_state[&key].exp_avg_sq.clone();
                    let grad_sq = mul(&grad, &grad)?;
                    let exp_avg_sq_new = add(
                        &mul(&exp_avg_sq_old, &beta2_t)?,
                        &mul(&grad_sq, &one_minus_beta2)?,
                    )?;

                    // m_hat = (1-mu)*grad/(1-mu_prod) + mu_next*exp_avg/(1-mu_prod_next)
                    let grad_coef = f64_scalar_on::<T>((1.0 - mu) / (1.0 - mu_product), device)?;
                    let mom_coef = f64_scalar_on::<T>(mu_next / (1.0 - mu_product_next), device)?;
                    let grad_component = mul(&grad, &grad_coef)?;
                    let mom_component = mul(&exp_avg_new, &mom_coef)?;
                    let m_hat = add(&grad_component, &mom_component)?;

                    // v_hat = exp_avg_sq / bc2
                    let inv_bc2 = f64_scalar_on::<T>(1.0 / bc2, device)?;
                    let v_hat = mul(&exp_avg_sq_new, &inv_bc2)?;

                    // sqrt(v_hat) + eps
                    let eps_t = f64_scalar_on::<T>(config.eps, device)?;
                    let denom = add(&sqrt(&v_hat)?, &eps_t)?;
                    let update = div(&m_hat, &denom)?;

                    // Decoupled weight decay: param *= (1 - lr * wd)
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

                    let lr_t = f64_scalar_on::<T>(group_lr, device)?;
                    let scaled_update = mul(&update, &lr_t)?;
                    let new_param = sub(&decayed, &scaled_update)?;

                    // Commit.
                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: `update_storage` swaps the storage Arc; sole
                    // writer guarantees:
                    //  1. `step_foreach` runs under
                    //     `Optimizer::step(&mut self)` for this `Nadam<T>`,
                    //     so no other thread/task is iterating these params.
                    //  2. The foreach loop is wrapped in `no_grad`, so no
                    //     autograd `grad_fn` will retain a clone of
                    //     `param_t`'s storage Arc.
                    //  3. `param_t` is the per-iteration clone of the
                    //     parameter's `Tensor`; tensor temporaries
                    //     (`decayed`, `scaled_update`, `new_param`,
                    //     `update`) hold independent storage and `new_param`
                    //     was just consumed by `into_storage_and_shape`.
                    //     No live borrow into the existing storage remains.
                    //  4. New `storage` was produced by ops dispatched on
                    //     `param_t`'s device, so device + numel match.
                    unsafe { param_t.update_storage(storage)? };

                    let state = self.foreach_state.get_mut(&key).unwrap();
                    state.step_count = next_step;
                    state.mu_product = mu_product;
                    state.exp_avg = exp_avg_new;
                    state.exp_avg_sq = exp_avg_sq_new;

                    Ok::<(), ferrotorch_core::FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }
}

impl<T: Float> Optimizer<T> for NAdam<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        if self.config.foreach {
            return self.step_foreach();
        }

        let config = self.config;
        let (beta1, beta2) = config.betas;
        let psi = config.momentum_decay;

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

                // Decoupled weight decay.
                let decay_factor = if config.decoupled_weight_decay && group_wd > 0.0 {
                    1.0 - group_lr * group_wd
                } else {
                    1.0
                };

                // L2 weight decay.
                if !config.decoupled_weight_decay && group_wd > 0.0 {
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g += group_wd * p;
                    }
                }

                let state = self.state.entry(key).or_insert_with(|| NAdamParamState {
                    step_count: 0,
                    mu_product: 1.0,
                    exp_avg: vec![0.0; numel],
                    exp_avg_sq: vec![0.0; numel],
                });

                state.step_count += 1;
                let step = state.step_count as f64;

                // Compute mu schedule values.
                // mu_t = beta1 * (1 - 0.5 * 0.96^(t * psi))
                let mu = beta1 * (1.0 - 0.5 * 0.96_f64.powf(step * psi));
                let mu_next = beta1 * (1.0 - 0.5 * 0.96_f64.powf((step + 1.0) * psi));

                // Update cumulative mu product.
                state.mu_product *= mu;

                let bias_correction2 = 1.0 - beta2.powi(state.step_count as i32);

                // Update moments.
                // m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                for (m, &g) in state.exp_avg.iter_mut().zip(grad_data.iter()) {
                    *m = beta1 * *m + (1.0 - beta1) * g;
                }

                // v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                for (v, &g) in state.exp_avg_sq.iter_mut().zip(grad_data.iter()) {
                    *v = beta2 * *v + (1.0 - beta2) * g * g;
                }

                // NAdam update with Nesterov lookahead:
                //
                // m_hat = mu_{t+1} * m_t / (1 - prod_{i=1}^{t+1} mu_i)
                //       + (1 - mu_t) * g_t / (1 - prod_{i=1}^{t} mu_i)
                //
                // v_hat = v_t / (1 - beta2^t)
                //
                // theta_t = theta_t - lr * m_hat / (sqrt(v_hat) + eps)
                let mu_product_next = state.mu_product * mu_next;

                let new_values: Vec<T> = (0..numel)
                    .map(|i| {
                        let grad_component = (1.0 - mu) * grad_data[i] / (1.0 - state.mu_product);
                        let momentum_component =
                            mu_next * state.exp_avg[i] / (1.0 - mu_product_next);
                        let m_hat = grad_component + momentum_component;
                        let v_hat = state.exp_avg_sq[i] / bias_correction2;
                        let denom = v_hat.sqrt() + config.eps;
                        let decayed = param_data[i] * decay_factor;
                        let updated = decayed - group_lr * m_hat / denom;
                        cast::<f64, T>(updated)
                    })
                    .collect::<FerrotorchResult<Vec<T>>>()?;

                // SAFETY: `update_data` writes through `Arc::as_ptr` and
                // requires the parameter's storage to have no live aliases.
                //  1. `Nadam::step(&mut self)` is the unique mutable handle
                //     to this optimiser; the per-(gi, pi) loop is sequential.
                //  2. The `no_grad` closure prevents `grad_fn` recording, so
                //     no autograd node will clone the storage Arc.
                //  3. The earlier `tensor.data_vec()` and
                //     `grad_tensor.data_vec()` calls returned owned
                //     `Vec<f64>` values now consumed by the moment-update
                //     loops. `new_values` is a fresh `Vec<T>` independent
                //     of the parameter's storage.
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
            entry.insert("mu_product".to_string(), vec![ps.mu_product]);
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

            let mu_product = entry
                .get("mu_product")
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(1.0);

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
                NAdamParamState {
                    step_count,
                    mu_product,
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

    fn param_val(opt: &NAdam<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    #[test]
    fn test_nadam_convergence_quadratic() {
        let p = scalar_param(5.0);
        let mut opt = NAdam::new(
            vec![p],
            NAdamConfig {
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
    fn test_nadam_convergence_two_params() {
        let px = scalar_param(3.0);
        let py = scalar_param(-2.0);

        let mut opt = NAdam::new(
            vec![px, py],
            NAdamConfig {
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
    fn test_nadam_zero_grad() {
        let p = scalar_param(1.0);
        let mut opt = NAdam::new(vec![p], NAdamConfig::default());

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
    fn test_nadam_state_dict_roundtrip() {
        let p = scalar_param(2.0);
        let mut opt = NAdam::new(vec![p], NAdamConfig::default());

        for _ in 0..3 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        let saved = opt.state_dict();
        let key = NAdam::<f64>::param_key(0, 0);
        assert_eq!(saved[&key]["step_count"][0] as u64, 3);
        assert!(saved[&key].contains_key("mu_product"));

        let p2 = scalar_param(2.0);
        let mut opt2 = NAdam::new(vec![p2], NAdamConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2.state_dict();
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["mu_product"], saved[&key]["mu_product"]);
    }

    #[test]
    fn test_nadam_lr_accessors() {
        let p = scalar_param(1.0);
        let mut opt = NAdam::new(
            vec![p],
            NAdamConfig {
                lr: 0.05,
                ..Default::default()
            },
        );
        assert!((opt.lr() - 0.05).abs() < 1e-12);
        opt.set_lr(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_nadam_default_config() {
        let config = NAdamConfig::default();
        assert_eq!(config.lr, 2e-3);
        assert_eq!(config.betas, (0.9, 0.999));
        assert_eq!(config.eps, 1e-8);
        assert_eq!(config.weight_decay, 0.0);
        assert_eq!(config.momentum_decay, 4e-3);
    }

    // -----------------------------------------------------------------------
    // Foreach mode parity tests. CL-497
    // -----------------------------------------------------------------------

    fn paired_nadam(data: &[f32]) -> (Parameter<f32>, Parameter<f32>) {
        (
            Parameter::from_slice(data, &[data.len()]).unwrap(),
            Parameter::from_slice(data, &[data.len()]).unwrap(),
        )
    }

    fn nadam_run_pair(
        cfg: NAdamConfig,
        init: &[f32],
        steps: usize,
        grad: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let (p_legacy, p_foreach) = paired_nadam(init);
        let mut legacy = NAdam::new(vec![p_legacy.clone()], cfg);
        let mut foreach = NAdam::new(
            vec![p_foreach.clone()],
            NAdamConfig {
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
    fn test_nadam_foreach_basic_parity() {
        let (l, f) = nadam_run_pair(
            NAdamConfig::default(),
            &[1.0, 2.0, 3.0, 4.0],
            5,
            &[0.1, 0.2, -0.3, 0.4],
        );
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "nadam foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_nadam_foreach_parity_with_decoupled_wd() {
        let cfg = NAdamConfig {
            lr: 0.002,
            weight_decay: 0.05,
            decoupled_weight_decay: true,
            ..Default::default()
        };
        let (l, f) = nadam_run_pair(cfg, &[5.0, -3.0, 2.0], 6, &[0.5, -0.5, 1.0]);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "nadam decoupled wd parity: legacy={a}, foreach={b}"
            );
        }
    }
}
