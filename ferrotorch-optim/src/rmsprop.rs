//! RMSprop optimizer.
//!
//! Implements the RMSprop algorithm as described by Hinton in his Coursera
//! lecture 6e. Supports optional momentum, centered gradient normalization,
//! and L2 weight decay.

use std::collections::HashMap;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Hyperparameters for [`Rmsprop`].
#[derive(Debug, Clone)]
#[non_exhaustive]
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
    /// When `true`, use the on-device tensor-op update path. Avoids per-step
    /// CPU↔GPU round-trips. Default: false. CL-497
    pub foreach: bool,
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
            foreach: false,
        }
    }
}

impl RmspropConfig {
    /// Set the learning rate.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the smoothing constant / decay rate for the running average of squared gradients.
    #[must_use]
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the term added to the denominator for numerical stability.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the L2 penalty coefficient.
    #[must_use]
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set the momentum factor.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable or disable centered RMSprop (normalize by an estimate of the gradient variance).
    #[must_use]
    pub fn with_centered(mut self, centered: bool) -> Self {
        self.centered = centered;
        self
    }

    /// Set the maximize flag (when `true`, negate the gradient to maximize).
    #[must_use]
    pub fn with_maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
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

/// On-device foreach state: all buffers live as `Tensor<T>` on the
/// parameter's device.
#[derive(Debug)]
struct ForeachState<T: Float> {
    square_avg: Tensor<T>,
    grad_avg: Option<Tensor<T>>,
    momentum_buf: Option<Tensor<T>>,
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
    /// Foreach (on-device) state. Used when `config.foreach == true`.
    foreach_state: HashMap<ParamKey, ForeachState<T>>,
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
            foreach_state: HashMap::new(),
        }
    }

    /// Create an RMSprop optimizer with pre-built parameter groups.
    pub fn new_with_groups(groups: Vec<ParamGroup<T>>, config: RmspropConfig) -> Self {
        Self {
            param_groups: groups,
            config,
            state: HashMap::new(),
            foreach_state: HashMap::new(),
        }
    }

    /// Foreach (on-device, tensor-op) update path. CL-497
    fn step_foreach(&mut self) -> FerrotorchResult<()> {
        use ferrotorch_core::creation::{scalar, zeros};
        use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, neg, sqrt, sub};

        let config = &self.config;
        let alpha = config.alpha;
        let eps = config.eps;
        let momentum = config.momentum;
        let centered = config.centered;

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
                let key = (gi, pi);

                // Lazy-init state.
                if let std::collections::hash_map::Entry::Vacant(e) = self.foreach_state.entry(key)
                {
                    e.insert(ForeachState {
                        square_avg: zeros::<T>(param_t.shape())?.to(device)?,
                        grad_avg: if centered {
                            Some(zeros::<T>(param_t.shape())?.to(device)?)
                        } else {
                            None
                        },
                        momentum_buf: if momentum > 0.0 {
                            Some(zeros::<T>(param_t.shape())?.to(device)?)
                        } else {
                            None
                        },
                    });
                }

                no_grad(|| {
                    // grad (possibly negated, possibly L2-augmented).
                    let mut grad: Tensor<T> = if config.maximize {
                        neg(&grad_tensor)?
                    } else {
                        grad_tensor.clone()
                    };
                    if group_wd > 0.0 {
                        let wd_t = scalar(cast::<f64, T>(group_wd)?)?.to(device)?;
                        let weighted = mul(&param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    let alpha_t = scalar(cast::<f64, T>(alpha)?)?.to(device)?;
                    let one_minus_alpha_t = scalar(cast::<f64, T>(1.0 - alpha)?)?.to(device)?;
                    let eps_t = scalar(cast::<f64, T>(eps)?)?.to(device)?;

                    // square_avg = alpha * square_avg + (1 - alpha) * g^2
                    let sq_old = self.foreach_state[&key].square_avg.clone();
                    let g_sq = mul(&grad, &grad)?;
                    let square_avg_new =
                        add(&mul(&sq_old, &alpha_t)?, &mul(&g_sq, &one_minus_alpha_t)?)?;

                    // avg denominator.
                    let (avg, grad_avg_new) = if centered {
                        let ga_old = self.foreach_state[&key]
                            .grad_avg
                            .as_ref()
                            .expect("centered grad_avg")
                            .clone();
                        let grad_avg_new_t =
                            add(&mul(&ga_old, &alpha_t)?, &mul(&grad, &one_minus_alpha_t)?)?;
                        // avg = sqrt(square_avg - grad_avg^2 + eps)
                        let ga_sq = mul(&grad_avg_new_t, &grad_avg_new_t)?;
                        let inner = sub(&square_avg_new, &ga_sq)?;
                        let inner_eps = add(&inner, &eps_t)?;
                        (sqrt(&inner_eps)?, Some(grad_avg_new_t))
                    } else {
                        // avg = sqrt(square_avg + eps)
                        let inner = add(&square_avg_new, &eps_t)?;
                        (sqrt(&inner)?, None)
                    };

                    // Compute update and momentum buffer.
                    let lr_t = scalar(cast::<f64, T>(group_lr)?)?.to(device)?;

                    let (new_param, momentum_buf_new) = if momentum > 0.0 {
                        let momentum_t = scalar(cast::<f64, T>(momentum)?)?.to(device)?;
                        let buf_old = self.foreach_state[&key]
                            .momentum_buf
                            .as_ref()
                            .expect("momentum_buf")
                            .clone();
                        // buf = momentum * buf + grad / avg
                        let grad_over_avg = div(&grad, &avg)?;
                        let new_buf = add(&mul(&buf_old, &momentum_t)?, &grad_over_avg)?;
                        let scaled = mul(&new_buf, &lr_t)?;
                        let np = sub(&param_t, &scaled)?;
                        (np, Some(new_buf))
                    } else {
                        // param = param - lr * grad / avg
                        let grad_over_avg = div(&grad, &avg)?;
                        let scaled = mul(&grad_over_avg, &lr_t)?;
                        let np = sub(&param_t, &scaled)?;
                        (np, None)
                    };

                    // Commit param update and state.
                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: `update_storage` swaps the storage Arc; this
                    // is sound when no other live handle observes the old
                    // storage during the swap.
                    //  1. Reachable only from `Rmsprop::step_foreach`, which
                    //     is called from `Optimizer::step(&mut self)` — sole
                    //     mutator of this optimiser.
                    //  2. The enclosing block runs inside the `no_grad`
                    //     closure that wraps the foreach loop, so no
                    //     `grad_fn` records a clone of `param_t`'s storage.
                    //  3. `param_t` is a per-iteration clone of the
                    //     parameter's `Tensor`; the temporaries built from
                    //     it (`scaled`, `grad_over_avg`, `new_param`,
                    //     etc.) hold their own storage, and `new_param` was
                    //     just consumed by `into_storage_and_shape`. No live
                    //     borrow into the original storage remains.
                    //  4. New `storage` was produced by tensor ops on
                    //     `param_t`'s device, so device + numel match (numel
                    //     re-checked inside `update_storage`).
                    unsafe { param_t.update_storage(storage)? };

                    let state = self.foreach_state.get_mut(&key).unwrap();
                    state.square_avg = square_avg_new;
                    if let Some(ga) = grad_avg_new {
                        state.grad_avg = Some(ga);
                    }
                    if let Some(buf) = momentum_buf_new {
                        state.momentum_buf = Some(buf);
                    }

                    Ok::<(), ferrotorch_core::FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Optimizer trait impl
// ---------------------------------------------------------------------------

impl<T: Float> Optimizer<T> for Rmsprop<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        if self.config.foreach {
            return self.step_foreach();
        }

        let alpha = self.config.alpha;
        let eps = self.config.eps;
        let momentum = self.config.momentum;
        let centered = self.config.centered;

        let alpha_t = cast::<f64, T>(alpha)?;
        let one_minus_alpha = cast::<f64, T>(1.0 - alpha)?;
        let eps_t = cast::<f64, T>(eps)?;
        let momentum_t = cast::<f64, T>(momentum)?;

        no_grad(|| {
            for (gi, group) in self.param_groups.iter().enumerate() {
                let lr_t = cast::<f64, T>(group.lr)?;
                let wd_t = cast::<f64, T>(group.weight_decay)?;
                let zero = cast::<f64, T>(0.0)?;

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
                        let neg = cast::<f64, T>(-1.0)?;
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

                    // SAFETY: We are inside the `no_grad` closure that wraps
                    // the legacy step body (started at the top of `step`),
                    // and inside `Optimizer::step(&mut self)`, so:
                    //  1. No `grad_fn` is being constructed that could hold
                    //     a clone of `param.tensor()`'s storage Arc.
                    //  2. No other handle to this `Rmsprop<T>` exists during
                    //     the step.
                    //  3. The per-(gi, pi) loop is sequential; the only
                    //     borrows we previously took into the parameter were
                    //     `param.data_vec()` and `grad_tensor.data_vec()`,
                    //     both of which returned owned `Vec<T>` values that
                    //     have already been consumed (or shadowed) by this
                    //     point. `new_values` is a fresh owned `Vec<T>`.
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

    fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
        let mut out = OptimizerState::new();
        for (&(gi, pi), pstate) in &self.state {
            let key = format!("group{gi}_param{pi}");
            let mut entry = HashMap::new();
            let square_avg: Vec<f64> = pstate
                .square_avg
                .iter()
                .map(|&v| cast::<T, f64>(v))
                .collect::<FerrotorchResult<Vec<f64>>>()?;
            entry.insert("square_avg".to_string(), square_avg);
            if let Some(ref ga) = pstate.grad_avg {
                let grad_avg: Vec<f64> = ga
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                entry.insert("grad_avg".to_string(), grad_avg);
            }
            if let Some(ref mb) = pstate.momentum_buf {
                let momentum_buf: Vec<f64> = mb
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                entry.insert("momentum_buf".to_string(), momentum_buf);
            }
            out.insert(key, entry);
        }
        Ok(out)
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

            let sq: Vec<T> = match entry.get("square_avg") {
                Some(v) => v
                    .iter()
                    .map(|&x| cast::<f64, T>(x))
                    .collect::<FerrotorchResult<Vec<T>>>()?,
                None => Vec::new(),
            };

            let ga: Option<Vec<T>> = match entry.get("grad_avg") {
                Some(v) => Some(
                    v.iter()
                        .map(|&x| cast::<f64, T>(x))
                        .collect::<FerrotorchResult<Vec<T>>>()?,
                ),
                None => None,
            };

            let mb: Option<Vec<T>> = match entry.get("momentum_buf") {
                Some(v) => Some(
                    v.iter()
                        .map(|&x| cast::<f64, T>(x))
                        .collect::<FerrotorchResult<Vec<T>>>()?,
                ),
                None => None,
            };

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
        let t = Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], true)
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
            assert!(v.abs() < 0.1, "expected convergence near zero, got {v}");
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
            assert!(v.abs() < 0.1, "expected convergence with momentum, got {v}");
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

        let sd = opt
            .state_dict()
            .expect("rmsprop state_dict must succeed in test");
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
        let sd2 = opt2
            .state_dict()
            .expect("rmsprop state_dict round-trip must succeed in test");
        assert_eq!(sd.len(), sd2.len());
        for (key, entry) in &sd {
            let entry2 = sd2.get(key).expect("key should exist after load");
            for (field, vals) in entry {
                let vals2 = entry2.get(field).expect("field should exist");
                assert_eq!(vals.len(), vals2.len());
                for (a, b) in vals.iter().zip(vals2.iter()) {
                    assert!((a - b).abs() < 1e-10, "state mismatch for {key}/{field}");
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

    // -----------------------------------------------------------------------
    // Foreach mode parity tests. CL-497
    // -----------------------------------------------------------------------

    fn paired_rmsprop(data: &[f32]) -> (Parameter<f32>, Parameter<f32>) {
        (make_param(data), make_param(data))
    }

    fn rmsprop_run_pair(
        cfg: RmspropConfig,
        init: &[f32],
        steps: usize,
        grad: &[f32],
    ) -> (Vec<f32>, Vec<f32>) {
        let (p_legacy, p_foreach) = paired_rmsprop(init);
        let mut legacy = Rmsprop::new(vec![p_legacy.clone()], cfg.clone());
        let mut foreach = Rmsprop::new(
            vec![p_foreach.clone()],
            RmspropConfig {
                foreach: true,
                ..cfg
            },
        );

        for _ in 0..steps {
            set_grad(&p_legacy, grad);
            set_grad(&p_foreach, grad);
            legacy.step().unwrap();
            foreach.step().unwrap();
        }

        (read_param(&p_legacy), read_param(&p_foreach))
    }

    #[test]
    fn test_rmsprop_foreach_basic_parity() {
        let (l, f) = rmsprop_run_pair(
            RmspropConfig::default(),
            &[1.0, 2.0, 3.0, 4.0],
            5,
            &[0.1, 0.2, -0.3, 0.4],
        );
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "rmsprop foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_rmsprop_foreach_parity_with_momentum() {
        let cfg = RmspropConfig {
            lr: 0.01,
            momentum: 0.9,
            ..Default::default()
        };
        let (l, f) = rmsprop_run_pair(cfg, &[5.0, -3.0, 2.0], 5, &[0.5, -0.5, 1.0]);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "rmsprop momentum parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_rmsprop_foreach_parity_with_centered() {
        let cfg = RmspropConfig {
            lr: 0.01,
            centered: true,
            ..Default::default()
        };
        let (l, f) = rmsprop_run_pair(cfg, &[1.0, -1.0, 0.5], 6, &[0.3, -0.2, 0.1]);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "rmsprop centered parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_rmsprop_foreach_parity_with_weight_decay() {
        let cfg = RmspropConfig {
            lr: 0.01,
            weight_decay: 0.02,
            ..Default::default()
        };
        let (l, f) = rmsprop_run_pair(cfg, &[2.0, -1.0, 0.5, 0.0], 4, &[0.2, 0.1, -0.05, 0.3]);
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "rmsprop wd parity: legacy={a}, foreach={b}"
            );
        }
    }
}
