//! Rprop optimizer — Resilient Backpropagation.
//!
//! Implements the RPROP algorithm from Riedmiller & Braun, "A Direct Adaptive
//! Method for Faster Backpropagation Learning: The RPROP Algorithm" (1993).
//!
//! Rprop adapts per-parameter step sizes based on the sign of consecutive
//! gradients. When the gradient sign is consistent (same direction), the step
//! size increases; when it flips, the step size decreases. The actual gradient
//! magnitude is ignored; only the sign matters.
//!
//! This makes Rprop particularly effective for batch (non-stochastic) training
//! where gradient noise is minimal.
//!
//! CL-319

use std::collections::HashMap;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Hyperparameters for the [`Rprop`] optimizer.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct RpropConfig {
    /// Initial learning rate / step size (default: 0.01).
    pub lr: f64,
    /// Multiplicative decrease and increase factors (default: (0.5, 1.2)).
    /// eta_minus must be in (0, 1) and eta_plus must be > 1.
    pub etas: (f64, f64),
    /// Minimum and maximum allowed step sizes (default: (1e-6, 50.0)).
    pub step_sizes: (f64, f64),
}

impl Default for RpropConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            etas: (0.5, 1.2),
            step_sizes: (1e-6, 50.0),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-parameter state
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RpropParamState {
    step_count: u64,
    /// Previous gradient (used for sign comparison).
    prev_grad: Vec<f64>,
    /// Per-element step sizes.
    step_size: Vec<f64>,
}

// ---------------------------------------------------------------------------
// Rprop
// ---------------------------------------------------------------------------

/// Resilient Backpropagation optimizer.
///
/// # Algorithm
///
/// For each parameter element `i`:
///
/// 1. Compute `sign = sign(g_prev[i] * g_t[i])`
/// 2. If `sign > 0`: `eta[i] = min(eta[i] * eta_plus, step_max)`
/// 3. Else if `sign < 0`: `eta[i] = max(eta[i] * eta_minus, step_min)`, `g_t[i] = 0`
/// 4. Else: `eta[i]` unchanged
/// 5. `p[i] = p[i] - sign(g_t[i]) * eta[i]`
/// 6. `g_prev[i] = g_t[i]`
#[derive(Debug)]
pub struct Rprop<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: RpropConfig,
    state: HashMap<String, RpropParamState>,
}

impl<T: Float> Rprop<T> {
    /// Create a new Rprop optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: RpropConfig) -> Self {
        let group = ParamGroup::new(params, config.lr);
        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
        }
    }

    #[inline]
    fn param_key(group_idx: usize, param_idx: usize) -> String {
        format!("g{group_idx}_p{param_idx}")
    }
}

impl<T: Float> Optimizer<T> for Rprop<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        let (eta_minus, eta_plus) = self.config.etas;
        let (step_min, step_max) = self.config.step_sizes;
        let init_lr = self.config.lr;

        for gi in 0..self.param_groups.len() {
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
                let grad_data: Vec<f64> = grad_tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;

                let numel = param_data.len();

                let state = self.state.entry(key).or_insert_with(|| RpropParamState {
                    step_count: 0,
                    prev_grad: vec![0.0; numel],
                    step_size: vec![init_lr; numel],
                });

                state.step_count += 1;

                let new_values: Vec<T> = (0..numel)
                    .map(|i| {
                        let g = grad_data[i];
                        let prev = state.prev_grad[i];

                        // Compute sign of the product of current and previous gradient.
                        let sign_product = if (g * prev) > 0.0 {
                            1
                        } else if (g * prev) < 0.0 {
                            -1
                        } else {
                            0
                        };

                        // Adapt step size.
                        if sign_product > 0 {
                            state.step_size[i] = (state.step_size[i] * eta_plus).min(step_max);
                        } else if sign_product < 0 {
                            state.step_size[i] = (state.step_size[i] * eta_minus).max(step_min);
                        }
                        // sign_product == 0: step size unchanged.

                        // For sign reversal, zero out the gradient so the
                        // parameter is not updated in the wrong direction,
                        // and save zero as the previous gradient.
                        let effective_grad = if sign_product < 0 { 0.0 } else { g };

                        // Update parameter: p = p - sign(g) * step_size.
                        let sign_g = if effective_grad > 0.0 {
                            1.0
                        } else if effective_grad < 0.0 {
                            -1.0
                        } else {
                            0.0
                        };

                        // Save current gradient (zeroed if reversed) for next step.
                        state.prev_grad[i] = effective_grad;

                        let updated = param_data[i] - sign_g * state.step_size[i];
                        cast::<f64, T>(updated)
                    })
                    .collect::<FerrotorchResult<Vec<T>>>()?;

                // SAFETY: `update_data` writes through `Arc::as_ptr` and
                // requires sole-writer access to the parameter's storage.
                //  1. `Rprop::step(&mut self)` is the unique mutable handle
                //     to this optimiser; the per-(gi, pi) loop is sequential
                //     so no two iterations alias the same parameter.
                //  2. The `no_grad` closure suppresses `grad_fn` recording,
                //     so no autograd node will clone the storage Arc as
                //     part of this write.
                //  3. The earlier `param.data_vec()` and `grad.data_vec()`
                //     reads returned owned `Vec<T>` / `Vec<f64>` values now
                //     consumed by the per-element update loop. `new_values`
                //     is a fresh owned `Vec<T>` independent of the
                //     parameter's storage.
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
            entry.insert("prev_grad".to_string(), ps.prev_grad.clone());
            entry.insert("step_size".to_string(), ps.step_size.clone());
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

            let prev_grad = entry.get("prev_grad").cloned().ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("missing prev_grad in state for key {key}"),
                }
            })?;

            let step_size = entry.get("step_size").cloned().ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("missing step_size in state for key {key}"),
                }
            })?;

            self.state.insert(
                key.clone(),
                RpropParamState {
                    step_count,
                    prev_grad,
                    step_size,
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

    fn param_val(opt: &Rprop<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    fn set_grad_scalar(opt: &Rprop<f64>, group: usize, idx: usize, val: f64) {
        let grad = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false).unwrap();
        opt.param_groups[group].params[idx]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();
    }

    #[test]
    fn test_rprop_convergence_quadratic() {
        let p = scalar_param(5.0);
        let mut opt = Rprop::new(vec![p], RpropConfig::default());

        for _ in 0..1000 {
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
    fn test_rprop_convergence_two_params() {
        let px = scalar_param(3.0);
        let py = scalar_param(-2.0);

        let mut opt = Rprop::new(vec![px, py], RpropConfig::default());

        for _ in 0..1000 {
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
    fn test_rprop_step_size_increase() {
        // Two consecutive steps with the same gradient sign should increase
        // the step size.
        let p = scalar_param(5.0);
        let mut opt = Rprop::new(vec![p], RpropConfig::default());

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        let key = Rprop::<f64>::param_key(0, 0);
        let step_size = opt.state[&key].step_size[0];
        // After init (0.01) and one consistent step: 0.01 * 1.2 = 0.012
        assert!(
            (step_size - 0.012).abs() < 1e-10,
            "step size should increase to 0.012, got {step_size}"
        );
    }

    #[test]
    fn test_rprop_step_size_decrease() {
        // Two steps with flipping gradient sign should decrease the step size.
        let p = scalar_param(5.0);
        let mut opt = Rprop::new(vec![p], RpropConfig::default());

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        set_grad_scalar(&opt, 0, 0, -1.0);
        opt.step().unwrap();

        let key = Rprop::<f64>::param_key(0, 0);
        let step_size = opt.state[&key].step_size[0];
        // After sign flip: 0.01 * 0.5 = 0.005
        assert!(
            (step_size - 0.005).abs() < 1e-10,
            "step size should decrease to 0.005, got {step_size}"
        );
    }

    #[test]
    fn test_rprop_zero_grad() {
        let p = scalar_param(1.0);
        let mut opt = Rprop::new(vec![p], RpropConfig::default());

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
    fn test_rprop_state_dict_roundtrip() {
        let p = scalar_param(2.0);
        let mut opt = Rprop::new(vec![p], RpropConfig::default());

        for _ in 0..3 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        let saved = opt.state_dict();
        let key = Rprop::<f64>::param_key(0, 0);
        assert_eq!(saved[&key]["step_count"][0] as u64, 3);
        assert!(saved[&key].contains_key("prev_grad"));
        assert!(saved[&key].contains_key("step_size"));

        let p2 = scalar_param(2.0);
        let mut opt2 = Rprop::new(vec![p2], RpropConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2.state_dict();
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["prev_grad"], saved[&key]["prev_grad"]);
        assert_eq!(loaded[&key]["step_size"], saved[&key]["step_size"]);
    }

    #[test]
    fn test_rprop_lr_accessors() {
        let p = scalar_param(1.0);
        let mut opt = Rprop::new(
            vec![p],
            RpropConfig {
                lr: 0.05,
                ..Default::default()
            },
        );
        assert!((opt.lr() - 0.05).abs() < 1e-12);
        opt.set_lr(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_rprop_default_config() {
        let config = RpropConfig::default();
        assert_eq!(config.lr, 1e-2);
        assert_eq!(config.etas, (0.5, 1.2));
        assert_eq!(config.step_sizes, (1e-6, 50.0));
    }

    #[test]
    fn test_rprop_step_size_clamping() {
        // Many steps with consistent gradient sign should not exceed step_max.
        let p = scalar_param(100.0);
        let mut opt = Rprop::new(
            vec![p],
            RpropConfig {
                lr: 1.0,
                step_sizes: (1e-6, 5.0),
                ..Default::default()
            },
        );

        for _ in 0..100 {
            set_grad_scalar(&opt, 0, 0, 1.0);
            opt.step().unwrap();
        }

        let key = Rprop::<f64>::param_key(0, 0);
        let step_size = opt.state[&key].step_size[0];
        assert!(
            step_size <= 5.0 + 1e-10,
            "step size should be clamped to 5.0, got {step_size}"
        );
    }
}
