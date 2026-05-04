//! Adadelta optimizer — adaptive learning rate per-parameter.
//!
//! Implements the algorithm from Zeiler, "ADADELTA: An Adaptive Learning Rate
//! Method" (2012). Adadelta adapts learning rates based on a moving window of
//! gradient updates, eliminating the need to set an initial learning rate.
//! Unlike Adagrad, Adadelta does not monotonically decrease the learning rate.
//!
//! The key innovation: instead of accumulating all past squared gradients,
//! Adadelta restricts the window of accumulated past gradients to a fixed size
//! via an exponential moving average, and uses the ratio of RMS(accumulated
//! deltas) to RMS(accumulated gradients) as the effective learning rate.
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

/// Hyperparameters for the [`Adadelta`] optimizer.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AdadeltaConfig {
    /// Coefficient that scales the delta before it is applied to parameters
    /// (default: 1.0). Note: unlike Adam-family optimizers, the default is 1.0.
    pub lr: f64,
    /// Coefficient for computing a running average of squared gradients
    /// (default: 0.9).
    pub rho: f64,
    /// Term added to the denominator for numerical stability (default: 1e-6).
    pub eps: f64,
    /// Weight decay coefficient (default: 0.0).
    pub weight_decay: f64,
    /// When `true`, use the on-device tensor-op update path. CL-497
    pub foreach: bool,
}

impl Default for AdadeltaConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            rho: 0.9,
            eps: 1e-6,
            weight_decay: 0.0,
            foreach: false,
        }
    }
}

impl AdadeltaConfig {
    /// Set the learning rate (scales the delta before it is applied to parameters).
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the coefficient for computing the running average of squared gradients.
    #[must_use]
    pub fn with_rho(mut self, rho: f64) -> Self {
        self.rho = rho;
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
struct AdadeltaParamState {
    step_count: u64,
    /// Running average of squared gradients.
    square_avg: Vec<f64>,
    /// Running average of squared parameter deltas.
    acc_delta: Vec<f64>,
}

/// On-device foreach state for Adadelta.
#[derive(Debug)]
struct AdadeltaForeachState<T: Float> {
    step_count: u64,
    square_avg: Tensor<T>,
    acc_delta: Tensor<T>,
}

// ---------------------------------------------------------------------------
// Adadelta
// ---------------------------------------------------------------------------

/// Adadelta optimizer — adaptive learning rate without manual tuning.
///
/// # Algorithm
///
/// For each parameter `p` with gradient `g`:
///
/// 1. `v_t = rho * v_{t-1} + (1 - rho) * g_t^2`
/// 2. `delta_t = sqrt(u_{t-1} + eps) / sqrt(v_t + eps) * g_t`
/// 3. `u_t = rho * u_{t-1} + (1 - rho) * delta_t^2`
/// 4. `p_t = p_{t-1} - lr * delta_t`
///
/// Where `v` is the squared gradient average and `u` is the squared delta
/// average.
#[derive(Debug)]
pub struct Adadelta<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdadeltaConfig,
    state: HashMap<String, AdadeltaParamState>,
    /// Foreach (on-device) state. Used when `config.foreach == true`.
    foreach_state: HashMap<String, AdadeltaForeachState<T>>,
}

impl<T: Float> Adadelta<T> {
    /// Create a new Adadelta optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: AdadeltaConfig) -> Self {
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
        let rho = config.rho;

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
                        AdadeltaForeachState {
                            step_count: 0,
                            square_avg: zeros::<T>(param_t.shape())?.to(device)?,
                            acc_delta: zeros::<T>(param_t.shape())?.to(device)?,
                        },
                    );
                }

                no_grad(|| {
                    // grad with L2 decay if enabled.
                    let mut grad: Tensor<T> = grad_tensor.clone();
                    if group_wd > 0.0 {
                        let wd_t = f64_scalar_on::<T>(group_wd, device)?;
                        let weighted = mul(&param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    let rho_t = f64_scalar_on::<T>(rho, device)?;
                    let one_minus_rho = f64_scalar_on::<T>(1.0 - rho, device)?;
                    let eps_t = f64_scalar_on::<T>(config.eps, device)?;

                    // square_avg = rho * square_avg + (1 - rho) * g^2
                    let sq_old = self.foreach_state[&key].square_avg.clone();
                    let g_sq = mul(&grad, &grad)?;
                    let square_avg_new = add(&mul(&sq_old, &rho_t)?, &mul(&g_sq, &one_minus_rho)?)?;

                    // std = sqrt(square_avg + eps)
                    let std_sq = add(&square_avg_new, &eps_t)?;
                    let std = sqrt(&std_sq)?;

                    // delta = sqrt(acc_delta + eps) / std * grad
                    let acc_delta_old = self.foreach_state[&key].acc_delta.clone();
                    let ad_plus_eps = add(&acc_delta_old, &eps_t)?;
                    let sqrt_ad = sqrt(&ad_plus_eps)?;
                    let ratio = div(&sqrt_ad, &std)?;
                    let delta = mul(&ratio, &grad)?;

                    // acc_delta = rho * acc_delta + (1 - rho) * delta^2
                    let delta_sq = mul(&delta, &delta)?;
                    let acc_delta_new = add(
                        &mul(&acc_delta_old, &rho_t)?,
                        &mul(&delta_sq, &one_minus_rho)?,
                    )?;

                    // param = param - lr * delta
                    let lr_t = f64_scalar_on::<T>(group_lr, device)?;
                    let scaled = mul(&delta, &lr_t)?;
                    let new_param = sub(&param_t, &scaled)?;

                    // Commit.
                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: `update_storage` swaps the storage Arc; sole
                    // writer guarantees:
                    //  1. `step_foreach` is the body of
                    //     `Optimizer::step(&mut self)` for this `Adadelta<T>`,
                    //     so no other thread/task is iterating these params.
                    //  2. The foreach loop is wrapped in `no_grad`, so no
                    //     autograd `grad_fn` will retain a clone of
                    //     `param_t`'s storage Arc.
                    //  3. `param_t` is the per-iteration clone of the
                    //     parameter's `Tensor`; tensor temporaries
                    //     (`scaled`, `new_param`) hold independent storage
                    //     and `new_param` was just consumed by
                    //     `into_storage_and_shape`. No live borrow into
                    //     the existing storage remains.
                    //  4. New `storage` was produced by ops dispatched on
                    //     `param_t`'s device, so device + numel match.
                    unsafe { param_t.update_storage(storage)? };

                    let state = self.foreach_state.get_mut(&key).unwrap();
                    state.step_count += 1;
                    state.square_avg = square_avg_new;
                    state.acc_delta = acc_delta_new;

                    Ok::<(), ferrotorch_core::FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }
}

impl<T: Float> Optimizer<T> for Adadelta<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        if self.config.foreach {
            return self.step_foreach();
        }

        let config = self.config;
        let rho = config.rho;

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

                let state = self.state.entry(key).or_insert_with(|| AdadeltaParamState {
                    step_count: 0,
                    square_avg: vec![0.0; numel],
                    acc_delta: vec![0.0; numel],
                });

                state.step_count += 1;

                // Compute parameter update.
                let new_values: Vec<T> = (0..numel)
                    .map(|i| {
                        let g = grad_data[i];

                        // Update running average of squared gradients.
                        // v_t = rho * v_{t-1} + (1 - rho) * g_t^2
                        state.square_avg[i] = rho * state.square_avg[i] + (1.0 - rho) * g * g;

                        // Compute delta.
                        // std = sqrt(v_t + eps)
                        let std = (state.square_avg[i] + config.eps).sqrt();
                        // delta = sqrt(u_{t-1} + eps) / std * g
                        let delta = (state.acc_delta[i] + config.eps).sqrt() / std * g;

                        // Update running average of squared deltas.
                        // u_t = rho * u_{t-1} + (1 - rho) * delta^2
                        state.acc_delta[i] = rho * state.acc_delta[i] + (1.0 - rho) * delta * delta;

                        // p_t = p_{t-1} - lr * delta
                        let updated = param_data[i] - group_lr * delta;
                        cast::<f64, T>(updated)
                    })
                    .collect::<FerrotorchResult<Vec<T>>>()?;

                // SAFETY: `update_data` writes through `Arc::as_ptr` and
                // requires sole-writer access to the parameter's storage.
                //  1. `Adadelta::step(&mut self)` holds the only mutable
                //     borrow on this optimiser; the per-(gi, pi) loop is
                //     sequential.
                //  2. The `no_grad` closure suppresses `grad_fn` recording
                //     for this write — no autograd node will keep a clone
                //     of the storage Arc.
                //  3. The earlier reads (`tensor.data_vec()` at line 244,
                //     `grad_tensor.data_vec()` at line 248) returned owned
                //     `Vec<f64>` values that have already been consumed.
                //     `new_values` is a fresh owned `Vec<T>`.
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
            entry.insert("square_avg".to_string(), ps.square_avg.clone());
            entry.insert("acc_delta".to_string(), ps.acc_delta.clone());
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

            let square_avg = entry.get("square_avg").cloned().ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("missing square_avg in state for key {key}"),
                }
            })?;

            let acc_delta = entry.get("acc_delta").cloned().ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("missing acc_delta in state for key {key}"),
                }
            })?;

            self.state.insert(
                key.clone(),
                AdadeltaParamState {
                    step_count,
                    square_avg,
                    acc_delta,
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

    fn param_val(opt: &Adadelta<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    #[test]
    fn test_adadelta_convergence_quadratic() {
        let p = scalar_param(5.0);
        let mut opt = Adadelta::new(vec![p], AdadeltaConfig::default());

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
    fn test_adadelta_convergence_two_params() {
        let px = scalar_param(3.0);
        let py = scalar_param(-2.0);

        let mut opt = Adadelta::new(vec![px, py], AdadeltaConfig::default());

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
    fn test_adadelta_zero_grad() {
        let p = scalar_param(1.0);
        let mut opt = Adadelta::new(vec![p], AdadeltaConfig::default());

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
    fn test_adadelta_state_dict_roundtrip() {
        let p = scalar_param(2.0);
        let mut opt = Adadelta::new(vec![p], AdadeltaConfig::default());

        for _ in 0..3 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        let saved = opt
            .state_dict()
            .expect("adadelta state_dict must succeed in test");
        let key = Adadelta::<f64>::param_key(0, 0);
        assert_eq!(saved[&key]["step_count"][0] as u64, 3);
        assert!(saved[&key].contains_key("square_avg"));
        assert!(saved[&key].contains_key("acc_delta"));

        let p2 = scalar_param(2.0);
        let mut opt2 = Adadelta::new(vec![p2], AdadeltaConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2
            .state_dict()
            .expect("adadelta state_dict round-trip must succeed in test");
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["square_avg"], saved[&key]["square_avg"]);
        assert_eq!(loaded[&key]["acc_delta"], saved[&key]["acc_delta"]);
    }

    #[test]
    fn test_adadelta_lr_accessors() {
        let p = scalar_param(1.0);
        let mut opt = Adadelta::new(
            vec![p],
            AdadeltaConfig {
                lr: 0.5,
                ..Default::default()
            },
        );
        assert!((opt.lr() - 0.5).abs() < 1e-12);
        opt.set_lr(0.1);
        assert!((opt.lr() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_adadelta_default_config() {
        let config = AdadeltaConfig::default();
        assert_eq!(config.lr, 1.0);
        assert_eq!(config.rho, 0.9);
        assert_eq!(config.eps, 1e-6);
        assert_eq!(config.weight_decay, 0.0);
    }

    #[test]
    fn test_adadelta_single_step_direction() {
        // After one step with positive gradient, parameter should decrease.
        let p = scalar_param(5.0);
        let mut opt = Adadelta::new(vec![p], AdadeltaConfig::default());

        let grad = Tensor::from_storage(TensorStorage::cpu(vec![2.0_f64]), vec![], false).unwrap();
        opt.param_groups[0].params[0]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();
        opt.step().unwrap();

        let val = param_val(&opt, 0, 0);
        assert!(
            val < 5.0,
            "param should decrease with positive gradient, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Foreach mode parity tests. CL-497
    // -----------------------------------------------------------------------

    #[test]
    fn test_adadelta_foreach_basic_parity() {
        let p_legacy = Parameter::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let p_foreach = Parameter::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

        let mut legacy = Adadelta::new(vec![p_legacy.clone()], AdadeltaConfig::default());
        let mut foreach = Adadelta::new(
            vec![p_foreach.clone()],
            AdadeltaConfig {
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
                "adadelta foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_adadelta_foreach_parity_with_weight_decay() {
        let p_legacy = Parameter::from_slice(&[5.0f32, -3.0, 2.0], &[3]).unwrap();
        let p_foreach = Parameter::from_slice(&[5.0f32, -3.0, 2.0], &[3]).unwrap();

        let cfg = AdadeltaConfig {
            weight_decay: 0.02,
            ..Default::default()
        };
        let mut legacy = Adadelta::new(vec![p_legacy.clone()], cfg);
        let mut foreach = Adadelta::new(
            vec![p_foreach.clone()],
            AdadeltaConfig {
                foreach: true,
                ..cfg
            },
        );

        for _ in 0..6 {
            let g =
                Tensor::from_storage(TensorStorage::cpu(vec![0.5f32, -0.5, 1.0]), vec![3], false)
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
                "adadelta wd parity: legacy={a}, foreach={b}"
            );
        }
    }
}
