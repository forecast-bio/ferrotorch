use std::collections::HashMap;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{
    Optimizer, OptimizerState, ParamGroup, f64_vec_to_tensor, tensor_to_f64_vec,
};

/// Configuration for the Adagrad optimizer.
///
/// Adagrad adapts the learning rate per-parameter, scaling inversely with
/// the square root of the sum of all historical squared gradients. This
/// makes it well-suited for sparse features where some parameters are
/// updated infrequently.
///
/// Reference: Duchi, Hazan & Singer, "Adaptive Subgradient Methods for
/// Online Learning and Stochastic Optimization", JMLR 2011.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct AdagradConfig {
    /// Base learning rate. Default: `0.01`.
    pub lr: f64,
    /// Learning rate decay applied per step. Default: `0.0`.
    pub lr_decay: f64,
    /// L2 penalty (weight decay). Default: `0.0`.
    pub weight_decay: f64,
    /// Initial value for the accumulator. Default: `0.0`.
    pub initial_accumulator_value: f64,
    /// Small constant for numerical stability. Default: `1e-10`.
    pub eps: f64,
    /// When `true`, maximize the objective by negating the gradient (default:
    /// false). CL-321
    pub maximize: bool,
    /// When `true`, use the on-device tensor-op update path. Avoids per-step
    /// CPU↔GPU round-trips at the cost of computing in `T` precision
    /// instead of f64. Default: false. CL-497
    pub foreach: bool,
}

impl Default for AdagradConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            lr_decay: 0.0,
            weight_decay: 0.0,
            initial_accumulator_value: 0.0,
            eps: 1e-10,
            maximize: false,
            foreach: false,
        }
    }
}

/// Per-parameter state maintained by Adagrad.
#[derive(Debug)]
struct AdagradParamState<T: Float> {
    /// Running sum of squared gradients.
    sum: Tensor<T>,
    /// Number of optimizer steps taken for this parameter.
    step_count: u64,
}

/// The Adagrad optimizer.
///
/// Adapts the learning rate for each parameter individually, dividing by
/// the square root of the accumulated sum of squared gradients. Parameters
/// that receive large gradients will have their effective learning rate
/// shrink quickly, while parameters with small gradients retain a larger
/// effective learning rate.
///
/// # Example
///
/// ```ignore
/// use ferrotorch_optim::{Adagrad, AdagradConfig, Optimizer, ParamGroup};
///
/// let params = model.parameters();
/// let config = AdagradConfig { lr: 0.01, ..Default::default() };
/// let mut optimizer = Adagrad::new(vec![ParamGroup::new(params, config.lr)], config);
///
/// // Training loop:
/// optimizer.zero_grad()?;
/// let loss = model.forward(&input)?;
/// loss.backward()?;
/// optimizer.step()?;
/// ```
#[derive(Debug)]
pub struct Adagrad<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdagradConfig,
    /// Per-parameter state, keyed by `(group_index, param_index)`.
    state: HashMap<(usize, usize), AdagradParamState<T>>,
}

impl<T: Float> Adagrad<T> {
    /// Create a new Adagrad optimizer.
    pub fn new(param_groups: Vec<ParamGroup<T>>, config: AdagradConfig) -> Self {
        Self {
            param_groups,
            config,
            state: HashMap::new(),
        }
    }

    /// Create an Adagrad optimizer from a flat list of parameters with default config.
    pub fn from_params(params: Vec<Parameter<T>>, lr: f64) -> Self {
        let config = AdagradConfig {
            lr,
            ..Default::default()
        };
        let group = ParamGroup::new(params, lr);
        Self::new(vec![group], config)
    }

    /// Initialize per-parameter state on first access.
    fn ensure_state(
        &mut self,
        group_idx: usize,
        param_idx: usize,
        shape: &[usize],
    ) -> FerrotorchResult<()> {
        let key = (group_idx, param_idx);
        if !self.state.contains_key(&key) {
            let numel: usize = shape.iter().product();
            let init_val = cast::<f64, T>(self.config.initial_accumulator_value)?;
            let sum_data = vec![init_val; numel];
            let sum = Tensor::from_storage(TensorStorage::cpu(sum_data), shape.to_vec(), false)?;
            self.state
                .insert(key, AdagradParamState { sum, step_count: 0 });
        }
        Ok(())
    }

    /// Foreach (on-device, tensor-op) update path used when
    /// `config.foreach == true`. CL-497
    fn step_foreach(&mut self) -> FerrotorchResult<()> {
        use ferrotorch_core::creation::scalar;
        use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, neg, sqrt, sub};

        let lr = self.config.lr;
        let lr_decay = self.config.lr_decay;
        let weight_decay = self.config.weight_decay;
        let eps = self.config.eps;

        for group_idx in 0..self.param_groups.len() {
            for param_idx in 0..self.param_groups[group_idx].params.len() {
                let shape = self.param_groups[group_idx].params[param_idx]
                    .shape()
                    .to_vec();

                // Clone once: grabs fresh Arc, avoids aliasing borrow.
                let grad_tensor_opt = self.param_groups[group_idx].params[param_idx].grad()?;
                let grad_tensor = match grad_tensor_opt {
                    Some(g) => g,
                    None => continue,
                };

                let param_t = self.param_groups[group_idx].params[param_idx]
                    .tensor()
                    .clone();
                let device = param_t.device();

                // Ensure the state exists. The sum accumulator is allocated
                // on CPU initially; move it to the parameter's device on
                // first use so all subsequent ops stay on-device.
                self.ensure_state(group_idx, param_idx, &shape)?;
                {
                    let state = self.state.get_mut(&(group_idx, param_idx)).unwrap();
                    if state.sum.device() != device {
                        state.sum = state.sum.clone().to(device)?;
                    }
                }

                let state_key = (group_idx, param_idx);

                no_grad(|| {
                    // grad (possibly negated, possibly with L2 weight decay).
                    let mut grad: Tensor<T> = if self.config.maximize {
                        neg(&grad_tensor)?
                    } else {
                        grad_tensor.clone()
                    };
                    if weight_decay > 0.0 {
                        let wd_t = scalar(cast::<f64, T>(weight_decay)?)?.to(device)?;
                        let weighted = mul(&param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    // sum += grad^2
                    let sum_old = self.state[&state_key].sum.clone();
                    let grad_sq = mul(&grad, &grad)?;
                    let sum_new = add(&sum_old, &grad_sq)?;

                    // Effective learning rate with decay.
                    let next_step = self.state[&state_key].step_count + 1;
                    let clr = lr / (1.0 + (next_step as f64 - 1.0) * lr_decay);

                    // param = param - clr * grad / (sqrt(sum) + eps)
                    let clr_t = scalar(cast::<f64, T>(clr)?)?.to(device)?;
                    let eps_t = scalar(cast::<f64, T>(eps)?)?.to(device)?;
                    let denom = add(&sqrt(&sum_new)?, &eps_t)?;
                    let scaled_grad = mul(&grad, &clr_t)?;
                    let update = div(&scaled_grad, &denom)?;
                    let new_param = sub(&param_t, &update)?;

                    // Commit parameter and state.
                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: `update_storage` requires the caller to hold
                    // exclusive access to the parameter's storage Arc.
                    // Conditions here:
                    //  1. We are inside `Optimizer::step` (`&mut self`), so no
                    //     other clone of `Adagrad<T>` can be running.
                    //  2. The enclosing closure is wrapped in `no_grad`, so no
                    //     autograd graph is being constructed and no `grad_fn`
                    //     holds a clone of the parameter tensor.
                    //  3. `param_t` is a fresh clone of the param's `Tensor`
                    //     held only in this iteration of the param loop; all
                    //     prior reads (`param_t.clone()`, the gradient
                    //     materialisation above) borrowed by value into
                    //     freshly-allocated tensors that are about to drop —
                    //     no live borrow into this storage exists.
                    //  4. `new_param.into_storage_and_shape()` consumed
                    //     `new_param`, so the only remaining handle to
                    //     `storage` is local.
                    // The new storage is on the same device (it was produced
                    // by ops dispatched on `device`) and has matching numel
                    // (verified internally by `update_storage`).
                    unsafe { param_t.update_storage(storage)? };

                    let state = self.state.get_mut(&state_key).unwrap();
                    state.sum = sum_new;
                    state.step_count = next_step;

                    Ok::<(), ferrotorch_core::FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }
}

impl<T: Float> Optimizer<T> for Adagrad<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        // Foreach path: stay on-device throughout, no CPU roundtrip.
        if self.config.foreach {
            return self.step_foreach();
        }

        let lr = self.config.lr;
        let lr_decay = self.config.lr_decay;
        let weight_decay = self.config.weight_decay;
        let eps = self.config.eps;

        for group_idx in 0..self.param_groups.len() {
            for param_idx in 0..self.param_groups[group_idx].params.len() {
                // Extract shape and gradient data before borrowing self mutably.
                let shape = self.param_groups[group_idx].params[param_idx]
                    .shape()
                    .to_vec();
                let grad_opt = self.param_groups[group_idx].params[param_idx].grad()?;
                let grad_tensor = match grad_opt {
                    Some(g) => g,
                    None => continue,
                };
                let mut grad_vec: Vec<T> = grad_tensor.data_vec()?;
                let param_vec: Vec<T> =
                    self.param_groups[group_idx].params[param_idx].data_vec()?;

                // Maximize: negate gradient. CL-321
                if self.config.maximize {
                    let neg = cast::<f64, T>(-1.0)?;
                    for g in grad_vec.iter_mut() {
                        *g = *g * neg;
                    }
                }

                self.ensure_state(group_idx, param_idx, &shape)?;

                let state = self.state.get_mut(&(group_idx, param_idx)).unwrap();
                state.step_count += 1;

                // Effective learning rate with decay.
                let clr = lr / (1.0 + (state.step_count as f64 - 1.0) * lr_decay);

                let sum_vec: Vec<T> = state.sum.data_vec()?;
                let numel = param_vec.len();

                // Compute updated values inside no_grad to prevent graph tracking.
                let (new_param_data, new_sum_data) =
                    no_grad(|| -> FerrotorchResult<(Vec<T>, Vec<T>)> {
                        let mut grad_buf = grad_vec;

                        // L2 regularization: grad = grad + weight_decay * param.
                        if weight_decay > 0.0 {
                            let wd = cast::<f64, T>(weight_decay)?;
                            for i in 0..numel {
                                grad_buf[i] += wd * param_vec[i];
                            }
                        }

                        // Accumulate squared gradients: sum += grad^2.
                        let mut new_sum = sum_vec;
                        for i in 0..numel {
                            new_sum[i] += grad_buf[i] * grad_buf[i];
                        }

                        // Update parameters: param = param - clr * grad / (sqrt(sum) + eps).
                        let clr_t = cast::<f64, T>(clr)?;
                        let eps_t = cast::<f64, T>(eps)?;
                        let mut new_param: Vec<T> = Vec::with_capacity(numel);
                        for i in 0..numel {
                            let denom = new_sum[i].sqrt() + eps_t;
                            new_param.push(param_vec[i] - clr_t * grad_buf[i] / denom);
                        }

                        Ok((new_param, new_sum))
                    })?;

                // Write back the updated sum accumulator.
                // SAFETY: `state.sum` is the per-parameter `Tensor<T>` held in
                // `self.state`, which is reachable only through `&mut self`.
                // No autograd graph references it: the optimiser constructs
                // it via `Tensor::from_storage(.., requires_grad=false)`, and
                // the new_sum_data Vec was just produced inside the `no_grad`
                // closure above, so no `grad_fn` could have cloned its Arc.
                // No live `&[T]` / `&mut [T]` borrow into `state.sum`'s
                // storage exists at this point — `state.sum.data_vec()` (line
                // 286) returned an owned `Vec<T>` that is now consumed.
                unsafe { state.sum.update_data(&new_sum_data)? };

                // Write back the updated parameter tensor (works on CPU and GPU).
                // SAFETY: Same exclusivity argument as the `state.sum` write
                // above, applied to the parameter tensor:
                //  1. `&mut self` on `Optimizer::step` precludes concurrent
                //     mutation through any other handle on this optimizer.
                //  2. The parameter's `Tensor<T>` is held by `Parameter<T>`,
                //     whose only reachable clones are inside `param_groups`;
                //     iteration is sequential by `(group_idx, param_idx)` so
                //     no other iteration of this loop holds a clone.
                //  3. The previous `data_vec()` (line 268) and the closure's
                //     output (line 290) produced owned `Vec<T>` values; no
                //     live borrow into the param's storage remains.
                //  4. The legacy CPU path runs no autograd ops in the
                //     enclosing scope, so even outside `no_grad` no `grad_fn`
                //     could clone the storage Arc between the closure exit
                //     and this write.
                unsafe {
                    self.param_groups[group_idx].params[param_idx]
                        .tensor()
                        .update_data(&new_param_data)?;
                };
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> FerrotorchResult<()> {
        for group in &self.param_groups {
            for param in &group.params {
                param.set_grad(None)?;
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
        self.config.lr = lr;
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
        let mut out = HashMap::new();
        for (&(gi, pi), s) in &self.state {
            let key = format!("group{gi}_param{pi}");
            let mut param_state = HashMap::new();

            // Serialize sum accumulator.
            if let Ok(sum_vec) = tensor_to_f64_vec(&s.sum) {
                param_state.insert("sum".to_string(), sum_vec);
            }

            // Serialize step count.
            param_state.insert("step_count".to_string(), vec![s.step_count as f64]);

            out.insert(key, param_state);
        }
        out
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        for (key, param_state) in state {
            // Parse "group{gi}_param{pi}".
            let parts: Vec<&str> = key.split('_').collect();
            if parts.len() != 2 {
                continue;
            }
            let gi: usize = parts[0]
                .strip_prefix("group")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);
            let pi: usize = parts[1]
                .strip_prefix("param")
                .and_then(|s| s.parse().ok())
                .unwrap_or(0);

            if gi >= self.param_groups.len() || pi >= self.param_groups[gi].params.len() {
                continue;
            }

            let shape = self.param_groups[gi].params[pi].shape().to_vec();
            let step_count = param_state
                .get("step_count")
                .and_then(|v| v.first())
                .map(|&v| v as u64)
                .unwrap_or(0);

            let sum = if let Some(sum_data) = param_state.get("sum") {
                f64_vec_to_tensor::<T>(sum_data, &shape)?
            } else {
                let numel: usize = shape.iter().product();
                let init = cast::<f64, T>(self.config.initial_accumulator_value)?;
                Tensor::from_storage(TensorStorage::cpu(vec![init; numel]), shape.clone(), false)?
            };

            self.state
                .insert((gi, pi), AdagradParamState { sum, step_count });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a parameter with known data and set its gradient.
    fn make_param_with_grad(data: &[f32], grad: &[f32]) -> FerrotorchResult<Parameter<f32>> {
        let shape = vec![data.len()];
        let param = Parameter::from_slice(data, &shape)?;
        let grad_tensor = Tensor::from_storage(TensorStorage::cpu(grad.to_vec()), shape, false)?;
        param.set_grad(Some(grad_tensor))?;
        Ok(param)
    }

    #[test]
    fn test_adagrad_convergence() {
        // Minimize f(x) = x^2, gradient = 2x, starting at x = 5.0.
        // Adagrad should reduce |x| over many steps.
        let param = Parameter::from_slice(&[5.0f32], &[1]).unwrap();
        let config = AdagradConfig {
            lr: 0.5,
            ..Default::default()
        };
        let mut opt = Adagrad::new(
            vec![ParamGroup::new(vec![param.clone()], config.lr)],
            config,
        );

        let mut current_val = 5.0f32;

        for _ in 0..100 {
            // Compute gradient: d/dx(x^2) = 2x.
            let grad_val = 2.0 * current_val;
            let grad_tensor =
                Tensor::from_storage(TensorStorage::cpu(vec![grad_val]), vec![1], false).unwrap();
            opt.param_groups_mut()[0].params[0]
                .set_grad(Some(grad_tensor))
                .unwrap();

            opt.step().unwrap();

            current_val = opt.param_groups()[0].params[0].data().unwrap()[0];
        }

        // After 100 steps, x should be much closer to 0 than the initial 5.0.
        assert!(
            current_val.abs() < 1.0,
            "expected convergence toward 0, got {current_val}"
        );
    }

    #[test]
    fn test_accumulator_grows_monotonically() {
        let param = make_param_with_grad(&[1.0, 2.0, 3.0], &[0.5, -1.0, 0.3]).unwrap();
        let config = AdagradConfig::default();
        let mut opt = Adagrad::new(vec![ParamGroup::new(vec![param], config.lr)], config);

        // Step once to initialize state.
        opt.step().unwrap();

        let sum_after_1 = opt.state[&(0, 0)].sum.data().unwrap().to_vec();

        // All accumulators should be positive (grad^2 > 0).
        for &v in &sum_after_1 {
            assert!(v > 0.0, "accumulator should be positive, got {v}");
        }

        // Set new gradients and step again.
        let grad2 =
            Tensor::from_storage(TensorStorage::cpu(vec![0.2f32, 0.8, -0.1]), vec![3], false)
                .unwrap();
        opt.param_groups_mut()[0].params[0]
            .set_grad(Some(grad2))
            .unwrap();
        opt.step().unwrap();

        let sum_after_2 = opt.state[&(0, 0)].sum.data().unwrap().to_vec();

        // Each accumulator element should have grown (sum += grad^2 >= 0).
        for (i, (&s1, &s2)) in sum_after_1.iter().zip(sum_after_2.iter()).enumerate() {
            assert!(
                s2 >= s1,
                "accumulator[{i}] should be non-decreasing: {s1} -> {s2}"
            );
        }
    }

    #[test]
    fn test_lr_decay_reduces_effective_lr() {
        // With lr_decay > 0, the effective lr should decrease each step.
        // We verify by checking that identical gradients produce smaller
        // parameter updates on later steps.
        let data = [10.0f32];
        let grad = [1.0f32];

        // Without decay.
        let p1 = make_param_with_grad(&data, &grad).unwrap();
        let config_no_decay = AdagradConfig {
            lr: 1.0,
            lr_decay: 0.0,
            ..Default::default()
        };
        let mut opt_no_decay = Adagrad::new(
            vec![ParamGroup::new(vec![p1], config_no_decay.lr)],
            config_no_decay,
        );

        // With decay.
        let p2 = make_param_with_grad(&data, &grad).unwrap();
        let config_decay = AdagradConfig {
            lr: 1.0,
            lr_decay: 0.1,
            ..Default::default()
        };
        let mut opt_decay = Adagrad::new(
            vec![ParamGroup::new(vec![p2], config_decay.lr)],
            config_decay,
        );

        // Step both optimizers with the same gradient.
        opt_no_decay.step().unwrap();
        opt_decay.step().unwrap();

        // First step: step_count = 1, clr = lr / (1 + 0 * decay) = lr.
        // Both should have the same result after step 1.
        let val_no_decay_1 = opt_no_decay.param_groups()[0].params[0].data().unwrap()[0];
        let val_decay_1 = opt_decay.param_groups()[0].params[0].data().unwrap()[0];
        assert!(
            (val_no_decay_1 - val_decay_1).abs() < 1e-6,
            "first step should be identical: {val_no_decay_1} vs {val_decay_1}"
        );

        // Set the same gradient again for step 2.
        let g1 = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![1], false).unwrap();
        let g2 = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![1], false).unwrap();
        opt_no_decay.param_groups_mut()[0].params[0]
            .set_grad(Some(g1))
            .unwrap();
        opt_decay.param_groups_mut()[0].params[0]
            .set_grad(Some(g2))
            .unwrap();

        opt_no_decay.step().unwrap();
        opt_decay.step().unwrap();

        let val_no_decay_2 = opt_no_decay.param_groups()[0].params[0].data().unwrap()[0];
        let val_decay_2 = opt_decay.param_groups()[0].params[0].data().unwrap()[0];

        // The decayed optimizer should have moved less (closer to initial).
        let update_no_decay = (val_no_decay_1 - val_no_decay_2).abs();
        let update_decay = (val_decay_1 - val_decay_2).abs();
        assert!(
            update_decay < update_no_decay,
            "lr_decay should reduce step size: no_decay update={update_no_decay}, decay update={update_decay}"
        );
    }

    #[test]
    fn test_weight_decay() {
        // With weight_decay, the effective gradient becomes `grad + wd * param`,
        // which is larger in magnitude than the bare gradient. This means the
        // accumulator (sum of squared effective gradients) grows faster.
        let data = [5.0f32];
        let grad = [1.0f32];

        // Without weight decay.
        let p_no_wd = make_param_with_grad(&data, &grad).unwrap();
        let config_no_wd = AdagradConfig {
            lr: 0.1,
            ..Default::default()
        };
        let mut opt_no_wd = Adagrad::new(
            vec![ParamGroup::new(vec![p_no_wd], config_no_wd.lr)],
            config_no_wd,
        );

        // With weight decay.
        let p_wd = make_param_with_grad(&data, &grad).unwrap();
        let config_wd = AdagradConfig {
            lr: 0.1,
            weight_decay: 0.5,
            ..Default::default()
        };
        let mut opt_wd = Adagrad::new(vec![ParamGroup::new(vec![p_wd], config_wd.lr)], config_wd);

        opt_no_wd.step().unwrap();
        opt_wd.step().unwrap();

        // The accumulator with weight decay should be larger because the
        // effective gradient (grad + wd * param) has larger magnitude.
        let sum_no_wd = opt_no_wd.state[&(0, 0)].sum.data().unwrap()[0];
        let sum_wd = opt_wd.state[&(0, 0)].sum.data().unwrap()[0];

        // Without wd: sum = 1.0^2 = 1.0
        // With wd:    sum = (1.0 + 0.5 * 5.0)^2 = 3.5^2 = 12.25
        assert!(
            sum_wd > sum_no_wd,
            "weight_decay should increase accumulator: no_wd={sum_no_wd}, wd={sum_wd}"
        );
        assert!(
            (sum_no_wd - 1.0).abs() < 1e-6,
            "no-wd accumulator should be grad^2 = 1.0, got {sum_no_wd}"
        );
        assert!(
            (sum_wd - 12.25).abs() < 1e-4,
            "wd accumulator should be (1 + 0.5*5)^2 = 12.25, got {sum_wd}"
        );

        // On the first step with zero initial accumulator, Adagrad normalizes
        // the update to clr * sign(grad), so parameter values match. The effect
        // of weight decay becomes visible on subsequent steps where the larger
        // accumulator (from the inflated effective gradient) causes smaller updates.
        // Set identical gradients for step 2 and verify divergence.
        let g1 = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![1], false).unwrap();
        let g2 = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![1], false).unwrap();
        opt_no_wd.param_groups_mut()[0].params[0]
            .set_grad(Some(g1))
            .unwrap();
        opt_wd.param_groups_mut()[0].params[0]
            .set_grad(Some(g2))
            .unwrap();

        opt_no_wd.step().unwrap();
        opt_wd.step().unwrap();

        let val_no_wd = opt_no_wd.param_groups()[0].params[0].data().unwrap()[0];
        let val_wd = opt_wd.param_groups()[0].params[0].data().unwrap()[0];
        assert!(
            (val_no_wd - val_wd).abs() > 1e-4,
            "weight_decay should cause parameter values to diverge after step 2: \
             no_wd={val_no_wd}, wd={val_wd}"
        );
    }

    #[test]
    fn test_zero_grad() {
        let param = make_param_with_grad(&[1.0, 2.0], &[0.5, 0.5]).unwrap();
        let config = AdagradConfig::default();
        let mut opt = Adagrad::new(vec![ParamGroup::new(vec![param], config.lr)], config);

        // Gradients should exist before zero_grad.
        assert!(opt.param_groups()[0].params[0].grad().unwrap().is_some());

        opt.zero_grad().unwrap();

        // Gradients should be cleared.
        assert!(opt.param_groups()[0].params[0].grad().unwrap().is_none());
    }

    #[test]
    fn test_state_dict_roundtrip() {
        let param = make_param_with_grad(&[1.0, 2.0, 3.0], &[0.1, 0.2, 0.3]).unwrap();
        let config = AdagradConfig {
            lr: 0.05,
            ..Default::default()
        };
        let mut opt = Adagrad::new(vec![ParamGroup::new(vec![param], config.lr)], config);

        // Take a step to populate state.
        opt.step().unwrap();
        let saved = opt.state_dict();

        // Create a fresh optimizer and load state.
        let param2 = Parameter::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let config2 = AdagradConfig {
            lr: 0.05,
            ..Default::default()
        };
        let mut opt2 = Adagrad::new(vec![ParamGroup::new(vec![param2], config2.lr)], config2);
        opt2.load_state_dict(&saved).unwrap();

        // Verify step count was restored.
        assert_eq!(opt2.state[&(0, 0)].step_count, 1);

        // Verify sum accumulator was restored.
        let original_sum = opt.state[&(0, 0)].sum.data().unwrap();
        let loaded_sum = opt2.state[&(0, 0)].sum.data().unwrap();
        for (&a, &b) in original_sum.iter().zip(loaded_sum.iter()) {
            assert!((a - b).abs() < 1e-6, "sum mismatch: {a:?} vs {b:?}");
        }
    }

    #[test]
    fn test_default_config() {
        let config = AdagradConfig::default();
        assert_eq!(config.lr, 0.01);
        assert_eq!(config.lr_decay, 0.0);
        assert_eq!(config.weight_decay, 0.0);
        assert_eq!(config.initial_accumulator_value, 0.0);
        assert_eq!(config.eps, 1e-10);
    }

    #[test]
    fn test_skip_params_without_grad() {
        // A parameter without a gradient should be untouched.
        let p_with = make_param_with_grad(&[5.0], &[1.0]).unwrap();
        let p_without = Parameter::from_slice(&[3.0f32], &[1]).unwrap();
        // p_without has no grad set.

        let config = AdagradConfig {
            lr: 0.1,
            ..Default::default()
        };
        let mut opt = Adagrad::new(
            vec![ParamGroup::new(vec![p_with, p_without], config.lr)],
            config,
        );

        opt.step().unwrap();

        // p_without should still be 3.0.
        let val = opt.param_groups()[0].params[1].data().unwrap()[0];
        assert!(
            (val - 3.0).abs() < 1e-7,
            "param without grad should be unchanged, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Foreach mode parity tests. CL-497
    // -----------------------------------------------------------------------

    fn paired_adagrad(data: &[f32]) -> (Parameter<f32>, Parameter<f32>) {
        (
            Parameter::from_slice(data, &[data.len()]).unwrap(),
            Parameter::from_slice(data, &[data.len()]).unwrap(),
        )
    }

    fn adagrad_run_pair(
        cfg: AdagradConfig,
        init: &[f32],
        grads: &[&[f32]],
    ) -> (Vec<f32>, Vec<f32>) {
        let (p_legacy, p_foreach) = paired_adagrad(init);
        let mut legacy = Adagrad::new(
            vec![ParamGroup::new(vec![p_legacy.clone()], cfg.lr)],
            cfg.clone(),
        );
        let mut foreach = Adagrad::new(
            vec![ParamGroup::new(vec![p_foreach.clone()], cfg.lr)],
            AdagradConfig {
                foreach: true,
                ..cfg
            },
        );

        for g in grads {
            let gt = Tensor::from_storage(TensorStorage::cpu(g.to_vec()), vec![init.len()], false)
                .unwrap();
            p_legacy.set_grad(Some(gt.clone())).unwrap();
            p_foreach.set_grad(Some(gt)).unwrap();
            legacy.step().unwrap();
            foreach.step().unwrap();
        }

        let l = legacy.param_groups()[0].params[0].data().unwrap().to_vec();
        let f = foreach.param_groups()[0].params[0].data().unwrap().to_vec();
        (l, f)
    }

    #[test]
    fn test_adagrad_foreach_basic_parity() {
        let g: &[f32] = &[0.1, 0.2, -0.3, 0.4];
        let grads: Vec<&[f32]> = vec![g; 5];
        let (l, f) = adagrad_run_pair(AdagradConfig::default(), &[1.0, 2.0, 3.0, 4.0], &grads);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "adagrad foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_adagrad_foreach_parity_with_weight_decay() {
        let cfg = AdagradConfig {
            lr: 0.1,
            weight_decay: 0.02,
            ..Default::default()
        };
        let g: &[f32] = &[0.5, -0.5, 1.0];
        let grads: Vec<&[f32]> = vec![g; 4];
        let (l, f) = adagrad_run_pair(cfg, &[5.0, -3.0, 2.0], &grads);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "adagrad weight decay parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_adagrad_foreach_parity_with_lr_decay() {
        let cfg = AdagradConfig {
            lr: 0.1,
            lr_decay: 0.01,
            ..Default::default()
        };
        let g: &[f32] = &[0.3, -0.2, 0.1];
        let grads: Vec<&[f32]> = vec![g; 6];
        let (l, f) = adagrad_run_pair(cfg, &[1.0, -1.0, 0.5], &grads);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "adagrad lr_decay parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_adagrad_foreach_parity_with_maximize() {
        let cfg = AdagradConfig {
            lr: 0.05,
            maximize: true,
            ..Default::default()
        };
        let g: &[f32] = &[0.3, 0.4];
        let grads: Vec<&[f32]> = vec![g; 3];
        let (l, f) = adagrad_run_pair(cfg, &[0.5, 1.0], &grads);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "adagrad maximize parity: legacy={a}, foreach={b}"
            );
        }
    }
}
