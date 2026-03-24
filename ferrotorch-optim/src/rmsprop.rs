//! RMSprop optimizer.
//!
//! Implements the RMSprop algorithm as described by Hinton in his Coursera
//! lecture 6e. Supports optional momentum, centered gradient normalization,
//! and L2 weight decay.

use std::collections::HashMap;

use ferrotorch_core::{no_grad, Float, FerrotorchResult};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Hyperparameters for [`Rmsprop`].
#[derive(Debug, Clone)]
pub struct RmspropConfig {
    /// Learning rate (default: 0.01).
    pub lr: f64,
    /// Smoothing constant / decay rate for the running average of squared
    /// gradients (default: 0.99).
    pub alpha: f64,
    /// Term added to the denominator for numerical stability (default: 1e-8).
    pub eps: f64,
    /// L2 penalty coefficient (default: 0.0).
    pub weight_decay: f64,
    /// Momentum factor (default: 0.0 — disabled).
    pub momentum: f64,
    /// If `true`, compute the centered RMSprop, normalizing the gradient by
    /// an estimate of its variance (default: false).
    pub centered: bool,
    /// When `true`, maximize the objective by negating the gradient (default:
    /// false). CL-321
    pub maximize: bool,
}

impl Default for RmspropConfig {
    fn default() -> Self {
        Self {
            lr: 0.01,
            alpha: 0.99,
            eps: 1e-8,
            weight_decay: 0.0,
            momentum: 0.0,
            centered: false,
            maximize: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-parameter state
// ---------------------------------------------------------------------------

/// Internal per-parameter running averages.
#[derive(Debug, Clone)]
struct ParamState<T: Float> {
    /// Running average of squared gradients.
    square_avg: Vec<T>,
    /// Running average of gradients (only when `centered = true`).
    grad_avg: Option<Vec<T>>,
    /// Momentum buffer (only when `momentum > 0`).
    momentum_buf: Option<Vec<T>>,
}

/// Composite key for per-parameter state: `(group_index, param_index)`.
///
/// Stable as long as parameter groups are not reordered, which is the
/// standard optimizer contract.
type ParamKey = (usize, usize);

// ---------------------------------------------------------------------------
// Optimizer struct
// ---------------------------------------------------------------------------

/// RMSprop optimizer.
///
/// Maintains a per-parameter exponential moving average of squared gradients
/// to adapt the learning rate element-wise.
///
/// # Algorithm
///
/// For each parameter with gradient `g`:
///
/// 1. If `weight_decay > 0`: `g = g + weight_decay * param`
/// 2. `square_avg = alpha * square_avg + (1 - alpha) * g^2`
/// 3. If `centered`: `grad_avg = alpha * grad_avg + (1 - alpha) * g`
///    then `avg = sqrt(square_avg - grad_avg^2 + eps)`
///    else `avg = sqrt(square_avg + eps)`
/// 4. If `momentum > 0`: `buf = momentum * buf + g / avg`; `param -= lr * buf`
///    else `param -= lr * g / avg`
#[derive(Debug)]
pub struct Rmsprop<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: RmspropConfig,
    /// Per-parameter state keyed by `(group_index, param_index)`.
    state: HashMap<ParamKey, ParamState<T>>,
}

impl<T: Float> Rmsprop<T> {
    /// Create an RMSprop optimizer for the given parameters.
    pub fn new(params: Vec<Parameter<T>>, config: RmspropConfig) -> Self {
        let lr = config.lr;
        let wd = config.weight_decay;
        let group = ParamGroup::new(params, lr).with_weight_decay(wd);
        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
        }
    }

    /// Create an RMSprop optimizer with pre-built parameter groups.
    pub fn new_with_groups(groups: Vec<ParamGroup<T>>, config: RmspropConfig) -> Self {
        Self {
            param_groups: groups,
            config,
            state: HashMap::new(),
        }
    }
}

// ---------------------------------------------------------------------------
// Optimizer trait impl
// ---------------------------------------------------------------------------

impl<T: Float> Optimizer<T> for Rmsprop<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        let alpha = self.config.alpha;
        let eps = self.config.eps;
        let momentum = self.config.momentum;
        let centered = self.config.centered;

        let alpha_t = T::from(alpha).unwrap();
        let one_minus_alpha = T::from(1.0 - alpha).unwrap();
        let eps_t = T::from(eps).unwrap();
        let momentum_t = T::from(momentum).unwrap();

        no_grad(|| {
            for (gi, group) in self.param_groups.iter().enumerate() {
                let lr_t = T::from(group.lr).unwrap();
                let wd_t = T::from(group.weight_decay).unwrap();
                let zero = T::from(0.0).unwrap();

                for (pi, param) in group.params.iter().enumerate() {
                    let grad_opt = param.grad()?;
                    let grad_tensor = match grad_opt {
                        Some(g) => g,
                        None => continue,
                    };

                    let param_data = param.data_vec()?;
                    let mut grad_data = grad_tensor.data_vec()?;
                    let n = param_data.len();

                    // Maximize: negate gradient. CL-321
                    let maximize = self.config.maximize;
                    if maximize {
                        let neg = T::from(-1.0).unwrap();
                        for g in grad_data.iter_mut() {
                            *g = *g * neg;
                        }
                    }

                    // Materialise gradient (applying weight decay if needed).
                    let grad: Vec<T> = if wd_t > zero {
                        grad_data
                            .iter()
                            .zip(param_data.iter())
                            .map(|(&g, &p)| g + wd_t * p)
                            .collect()
                    } else {
                        grad_data.to_vec()
                    };

                    // Initialise or retrieve per-parameter state.
                    let key = (gi, pi);
                    let state = self.state.entry(key).or_insert_with(|| ParamState {
                        square_avg: vec![zero; n],
                        grad_avg: if centered { Some(vec![zero; n]) } else { None },
                        momentum_buf: if momentum > 0.0 {
                            Some(vec![zero; n])
                        } else {
                            None
                        },
                    });

                    // Update square_avg: square_avg = alpha * square_avg + (1 - alpha) * g^2
                    for (sq, &g) in state.square_avg.iter_mut().zip(grad.iter()) {
                        *sq = alpha_t * *sq + one_minus_alpha * g * g;
                    }

                    // Compute denominator `avg`.
                    let avg: Vec<T> = if centered {
                        // Update grad_avg.
                        let ga = state.grad_avg.as_mut().unwrap();
                        for (ga_i, &g) in ga.iter_mut().zip(grad.iter()) {
                            *ga_i = alpha_t * *ga_i + one_minus_alpha * g;
                        }
                        // avg = sqrt(square_avg - grad_avg^2 + eps)
                        state
                            .square_avg
                            .iter()
                            .zip(ga.iter())
                            .map(|(&sq, &ga_i)| (sq - ga_i * ga_i + eps_t).sqrt())
                            .collect()
                    } else {
                        // avg = sqrt(square_avg + eps)
                        state
                            .square_avg
                            .iter()
                            .map(|&sq| (sq + eps_t).sqrt())
                            .collect()
                    };

                    // Compute updated parameter values.
                    let new_values: Vec<T> = if momentum > 0.0 {
                        let buf = state.momentum_buf.as_mut().unwrap();
                        (0..n)
                            .map(|i| {
                                buf[i] = momentum_t * buf[i] + grad[i] / avg[i];
                                param_data[i] - lr_t * buf[i]
                            })
                            .collect()
                    } else {
                        (0..n)
                            .map(|i| param_data[i] - lr_t * grad[i] / avg[i])
                            .collect()
                    };

                    // SAFETY: Optimizer step runs inside no_grad() with exclusive
                    // access to parameters, so no aliasing references exist.
                    unsafe { param.tensor().update_data(&new_values)? };
                }
            }

            Ok(())
        })
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
        for (&(gi, pi), pstate) in &self.state {
            let key = format!("group{gi}_param{pi}");
            let mut entry = HashMap::new();
            entry.insert(
                "square_avg".to_string(),
                pstate
                    .square_avg
                    .iter()
                    .map(|&v| v.to_f64().unwrap())
                    .collect(),
            );
            if let Some(ref ga) = pstate.grad_avg {
                entry.insert(
                    "grad_avg".to_string(),
                    ga.iter().map(|&v| v.to_f64().unwrap()).collect(),
                );
            }
            if let Some(ref mb) = pstate.momentum_buf {
                entry.insert(
                    "momentum_buf".to_string(),
                    mb.iter().map(|&v| v.to_f64().unwrap()).collect(),
                );
            }
            out.insert(key, entry);
        }
        out
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        for (key, entry) in state {
            // Parse "groupG_paramP".
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

            let sq = entry
                .get("square_avg")
                .map(|v| v.iter().map(|&x| T::from(x).unwrap()).collect())
                .unwrap_or_default();

            let ga = entry
                .get("grad_avg")
                .map(|v| v.iter().map(|&x| T::from(x).unwrap()).collect());

            let mb = entry
                .get("momentum_buf")
                .map(|v| v.iter().map(|&x| T::from(x).unwrap()).collect());

            self.state.insert(
                (gi, pi),
                ParamState {
                    square_avg: sq,
                    grad_avg: ga,
                    momentum_buf: mb,
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
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Create a parameter from the given data.
    fn make_param(data: &[f32]) -> Parameter<f32> {
        let t = Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            true,
        )
        .unwrap();
        Parameter::new(t)
    }

    /// Manually set a gradient on a parameter.
    fn set_grad(param: &Parameter<f32>, grad_data: &[f32]) {
        let g = Tensor::from_storage(
            TensorStorage::cpu(grad_data.to_vec()),
            vec![grad_data.len()],
            false,
        )
        .unwrap();
        param.set_grad(Some(g)).unwrap();
    }

    /// Read parameter data as a Vec.
    fn read_param(param: &Parameter<f32>) -> Vec<f32> {
        param.data().unwrap().to_vec()
    }

    // -----------------------------------------------------------------------
    // Convergence on a simple quadratic: f(x) = sum(x^2)
    // grad = 2*x, minimum at x = 0.
    // -----------------------------------------------------------------------
    #[test]
    fn test_rmsprop_quadratic_convergence() {
        let p = make_param(&[3.0, -4.0, 5.0]);
        let mut opt = Rmsprop::new(
            vec![p.clone()],
            RmspropConfig {
                lr: 0.01,
                alpha: 0.99,
                eps: 1e-8,
                ..Default::default()
            },
        );

        for _ in 0..2000 {
            opt.zero_grad().unwrap();
            let vals = read_param(&p);
            let grad: Vec<f32> = vals.iter().map(|&x| 2.0 * x).collect();
            set_grad(&p, &grad);
            opt.step().unwrap();
        }

        let final_vals = read_param(&p);
        for &v in &final_vals {
            assert!(
                v.abs() < 0.1,
                "expected convergence near zero, got {v}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Centered RMSprop converges on the same quadratic.
    // -----------------------------------------------------------------------
    #[test]
    fn test_rmsprop_centered_convergence() {
        let p = make_param(&[3.0, -4.0, 5.0]);
        let mut opt = Rmsprop::new(
            vec![p.clone()],
            RmspropConfig {
                lr: 0.01,
                alpha: 0.99,
                eps: 1e-8,
                centered: true,
                ..Default::default()
            },
        );

        for _ in 0..2000 {
            opt.zero_grad().unwrap();
            let vals = read_param(&p);
            let grad: Vec<f32> = vals.iter().map(|&x| 2.0 * x).collect();
            set_grad(&p, &grad);
            opt.step().unwrap();
        }

        let final_vals = read_param(&p);
        for &v in &final_vals {
            assert!(
                v.abs() < 0.1,
                "expected convergence near zero with centered, got {v}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Momentum variant converges.
    // -----------------------------------------------------------------------
    #[test]
    fn test_rmsprop_momentum_convergence() {
        let p = make_param(&[3.0, -4.0, 5.0]);
        let mut opt = Rmsprop::new(
            vec![p.clone()],
            RmspropConfig {
                lr: 0.01,
                alpha: 0.99,
                eps: 1e-8,
                momentum: 0.9,
                ..Default::default()
            },
        );

        for _ in 0..2000 {
            opt.zero_grad().unwrap();
            let vals = read_param(&p);
            let grad: Vec<f32> = vals.iter().map(|&x| 2.0 * x).collect();
            set_grad(&p, &grad);
            opt.step().unwrap();
        }

        let final_vals = read_param(&p);
        for &v in &final_vals {
            assert!(
                v.abs() < 0.1,
                "expected convergence with momentum, got {v}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Weight decay adds L2 penalty.
    // -----------------------------------------------------------------------
    #[test]
    fn test_rmsprop_weight_decay() {
        let p = make_param(&[10.0]);
        let mut opt = Rmsprop::new(
            vec![p.clone()],
            RmspropConfig {
                lr: 0.05,
                weight_decay: 0.5,
                ..Default::default()
            },
        );

        // Supply zero gradient -- weight decay alone should shrink the param.
        for _ in 0..500 {
            opt.zero_grad().unwrap();
            set_grad(&p, &[0.0]);
            opt.step().unwrap();
        }

        let final_val = read_param(&p)[0];
        assert!(
            final_val < 5.0,
            "weight decay should shrink param, got {final_val}"
        );
    }

    // -----------------------------------------------------------------------
    // Single step numerical check (no momentum, not centered).
    // -----------------------------------------------------------------------
    #[test]
    fn test_rmsprop_single_step_numerics() {
        let p = make_param(&[2.0]);
        let mut opt = Rmsprop::new(
            vec![p.clone()],
            RmspropConfig {
                lr: 0.1,
                alpha: 0.9,
                eps: 1e-8,
                ..Default::default()
            },
        );

        opt.zero_grad().unwrap();
        set_grad(&p, &[4.0]);
        opt.step().unwrap();

        // square_avg = 0.9 * 0 + 0.1 * 16 = 1.6
        // avg = sqrt(1.6 + 1e-8)
        // param = 2.0 - 0.1 * 4.0 / avg
        let val = read_param(&p)[0];
        let expected = 2.0 - 0.1 * 4.0 / (1.6_f32 + 1e-8).sqrt();
        assert!(
            (val - expected).abs() < 1e-5,
            "expected {expected}, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // state_dict / load_state_dict round-trip.
    // -----------------------------------------------------------------------
    #[test]
    fn test_rmsprop_state_dict_roundtrip() {
        let p = make_param(&[1.0, 2.0]);
        let mut opt = Rmsprop::new(
            vec![p.clone()],
            RmspropConfig {
                lr: 0.01,
                centered: true,
                momentum: 0.9,
                ..Default::default()
            },
        );

        // Run a few steps to populate state.
        for _ in 0..5 {
            opt.zero_grad().unwrap();
            let vals = read_param(&p);
            let grad: Vec<f32> = vals.iter().map(|&x| 2.0 * x).collect();
            set_grad(&p, &grad);
            opt.step().unwrap();
        }

        let sd = opt.state_dict();
        assert!(!sd.is_empty(), "state_dict should not be empty after steps");

        // Create a fresh optimizer and load the state.
        let p2 = make_param(&[1.0, 2.0]);
        let mut opt2 = Rmsprop::new(
            vec![p2.clone()],
            RmspropConfig {
                lr: 0.01,
                centered: true,
                momentum: 0.9,
                ..Default::default()
            },
        );
        opt2.load_state_dict(&sd).unwrap();

        // Verify internal state was restored.
        let sd2 = opt2.state_dict();
        assert_eq!(sd.len(), sd2.len());
        for (key, entry) in &sd {
            let entry2 = sd2.get(key).expect("key should exist after load");
            for (field, vals) in entry {
                let vals2 = entry2.get(field).expect("field should exist");
                assert_eq!(vals.len(), vals2.len());
                for (a, b) in vals.iter().zip(vals2.iter()) {
                    assert!(
                        (a - b).abs() < 1e-10,
                        "state mismatch for {key}/{field}"
                    );
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // zero_grad clears gradients.
    // -----------------------------------------------------------------------
    #[test]
    fn test_rmsprop_zero_grad() {
        let p = make_param(&[1.0, 2.0, 3.0]);
        set_grad(&p, &[0.5, 0.5, 0.5]);
        assert!(p.grad().unwrap().is_some());

        let mut opt = Rmsprop::new(vec![p.clone()], RmspropConfig::default());
        opt.zero_grad().unwrap();
        assert!(p.grad().unwrap().is_none());
    }

    // -----------------------------------------------------------------------
    // lr / set_lr.
    // -----------------------------------------------------------------------
    #[test]
    fn test_rmsprop_lr_accessors() {
        let p = make_param(&[1.0]);
        let mut opt = Rmsprop::new(
            vec![p],
            RmspropConfig {
                lr: 0.05,
                ..Default::default()
            },
        );
        assert!((opt.lr() - 0.05).abs() < 1e-12);
        opt.set_lr(0.001);
        assert!((opt.lr() - 0.001).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Skips parameters without gradients.
    // -----------------------------------------------------------------------
    #[test]
    fn test_rmsprop_skips_none_grad() {
        let p = make_param(&[5.0]);
        let mut opt = Rmsprop::new(vec![p.clone()], RmspropConfig::default());

        // Don't set any gradient -- step should be a no-op.
        opt.step().unwrap();
        let val = read_param(&p)[0];
        assert!(
            (val - 5.0).abs() < 1e-12,
            "param should be unchanged without grad, got {val}"
        );
    }
}
