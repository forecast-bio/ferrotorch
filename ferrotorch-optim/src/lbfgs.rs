//! L-BFGS optimizer (Limited-memory Broyden-Fletcher-Goldfarb-Shanno).
//!
//! A quasi-Newton optimizer that approximates the inverse Hessian using the
//! last `m` gradient updates. Best suited for small models where second-order
//! optimization is beneficial (e.g. full-batch training, physics-informed
//! neural networks, style transfer).
//!
//! The core algorithm is the two-loop recursion that computes a search
//! direction from the curvature pairs `(s, y)` accumulated over previous
//! iterations.
//!
//! All parameter updates execute inside `no_grad()` so the optimizer step is
//! never recorded in the autograd graph.

use std::collections::HashMap;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// LineSearchFn (placeholder enum for future expansion)
// ---------------------------------------------------------------------------

/// Line search strategy. Currently only `StrongWolfe` is defined as a
/// variant name; the actual implementation is deferred. Passing `None` in
/// the config means the optimizer uses a fixed step size equal to `lr`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LineSearchFn {
    /// Strong Wolfe line search (not yet implemented).
    StrongWolfe,
}

// ---------------------------------------------------------------------------
// LbfgsConfig
// ---------------------------------------------------------------------------

/// Hyperparameters for the L-BFGS optimizer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct LbfgsConfig {
    /// Learning rate / step size (default: 1.0).
    pub lr: f64,
    /// Maximum number of iterations per `step()` call (default: 20).
    pub max_iter: usize,
    /// Maximum number of function evaluations per `step()` call.
    /// Defaults to `max_iter * 5 / 4` when `None`.
    pub max_eval: Option<usize>,
    /// Termination tolerance on the gradient infinity norm (default: 1e-7).
    pub tolerance_grad: f64,
    /// Termination tolerance on the function value change (default: 1e-9).
    pub tolerance_change: f64,
    /// Number of curvature pairs to keep (default: 10).
    pub history_size: usize,
    /// Line search function (default: `None` -- fixed step with `lr`).
    pub line_search_fn: Option<LineSearchFn>,
    /// When `true`, maximize the objective by negating the gradient (default:
    /// false). CL-321
    pub maximize: bool,
}

impl Default for LbfgsConfig {
    fn default() -> Self {
        Self {
            lr: 1.0,
            max_iter: 20,
            max_eval: None,
            tolerance_grad: 1e-7,
            tolerance_change: 1e-9,
            history_size: 10,
            line_search_fn: None,
            maximize: false,
        }
    }
}

impl LbfgsConfig {
    /// Effective maximum number of function evaluations.
    fn effective_max_eval(&self) -> usize {
        self.max_eval.unwrap_or(self.max_iter * 5 / 4)
    }
}

// ---------------------------------------------------------------------------
// LbfgsState
// ---------------------------------------------------------------------------

/// Mutable state maintained across L-BFGS iterations.
#[derive(Debug)]
struct LbfgsState {
    /// Parameter differences: s_k = x_{k+1} - x_k.
    s_history: Vec<Vec<f64>>,
    /// Gradient differences: y_k = g_{k+1} - g_k.
    y_history: Vec<Vec<f64>>,
    /// Cached 1 / (y_k . s_k) for each curvature pair.
    rho_history: Vec<f64>,
    /// Previous flat parameter vector (needed to compute s_k).
    prev_flat_params: Option<Vec<f64>>,
    /// Previous flat gradient vector (needed to compute y_k).
    prev_flat_grad: Option<Vec<f64>>,
    /// Number of optimizer steps completed.
    n_iter: u64,
}

impl LbfgsState {
    fn new() -> Self {
        Self {
            s_history: Vec::new(),
            y_history: Vec::new(),
            rho_history: Vec::new(),
            prev_flat_params: None,
            prev_flat_grad: None,
            n_iter: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Lbfgs
// ---------------------------------------------------------------------------

/// The L-BFGS optimizer.
///
/// Flattens all parameter tensors into a single vector, computes the L-BFGS
/// search direction via the two-loop recursion, and writes updated values
/// back into the parameter tensors.
#[derive(Debug)]
pub struct Lbfgs<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: LbfgsConfig,
    state: LbfgsState,
}

impl<T: Float> Lbfgs<T> {
    /// Create a new L-BFGS optimizer for the given parameters.
    pub fn new(params: Vec<Parameter<T>>, config: LbfgsConfig) -> Self {
        let group = ParamGroup::new(params, config.lr);
        Self {
            param_groups: vec![group],
            config,
            state: LbfgsState::new(),
        }
    }

    // -- helpers ----------------------------------------------------------

    /// Flatten all parameter data into a single `f64` vector, collecting
    /// shapes along the way so we can scatter the results back.
    fn gather_params(&self) -> FerrotorchResult<(Vec<f64>, Vec<Vec<usize>>)> {
        let mut flat = Vec::new();
        let mut shapes = Vec::new();
        for group in &self.param_groups {
            for param in &group.params {
                let tensor = param.tensor();
                let data = tensor.data_vec()?;
                shapes.push(tensor.shape().to_vec());
                let cast_data: Vec<f64> = data
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                flat.extend(cast_data);
            }
        }
        Ok((flat, shapes))
    }

    /// Flatten all parameter gradients into a single `f64` vector.
    ///
    /// When `config.maximize` is set, the gradient is negated. CL-321
    fn gather_grads(&self) -> FerrotorchResult<Vec<f64>> {
        let negate = self.config.maximize;
        let mut flat = Vec::new();
        for group in &self.param_groups {
            for param in &group.params {
                let tensor = param.tensor();
                match tensor.grad()? {
                    Some(g) => {
                        let g_data = g.data_vec()?;
                        let cast_g: Vec<f64> = g_data
                            .iter()
                            .map(|&v| cast::<T, f64>(v))
                            .collect::<FerrotorchResult<Vec<f64>>>()?;
                        if negate {
                            flat.extend(cast_g.iter().map(|&v| -v));
                        } else {
                            flat.extend(cast_g);
                        }
                    }
                    None => {
                        // No gradient: treat as zero.
                        let numel = tensor.numel();
                        flat.extend(std::iter::repeat_n(0.0, numel));
                    }
                }
            }
        }
        Ok(flat)
    }

    /// Scatter a flat `f64` vector back into the parameter tensors (inside
    /// `no_grad`).
    fn scatter_params(&mut self, flat: &[f64], shapes: &[Vec<usize>]) -> FerrotorchResult<()> {
        let mut offset = 0usize;
        let mut shape_idx = 0usize;

        for gi in 0..self.param_groups.len() {
            for pi in 0..self.param_groups[gi].params.len() {
                let shape = &shapes[shape_idx];
                let numel: usize = if shape.is_empty() {
                    1
                } else {
                    shape.iter().product()
                };

                let slice = &flat[offset..offset + numel];
                let new_data: Vec<T> = slice
                    .iter()
                    .map(|&v| cast::<f64, T>(v))
                    .collect::<FerrotorchResult<Vec<T>>>()?;

                no_grad(|| {
                    // SAFETY: `update_data` writes through `Arc::as_ptr` to
                    // the parameter's storage; sole-writer required.
                    //  1. `scatter_params(&mut self, ..)` is called from
                    //     L-BFGS internals (`take_step`, `step`,
                    //     `step_with_closure`) which all hold `&mut self`
                    //     on the Lbfgs optimizer, so no other handle into
                    //     this optimizer can be running.
                    //  2. The `no_grad` closure suppresses `grad_fn`
                    //     recording for the write — no autograd node will
                    //     retain a clone of the storage Arc.
                    //  3. The flat `flat: &[f64]` argument is borrowed
                    //     immutably; we materialised `new_data: Vec<T>` from
                    //     it as an owned vector before this call, and the
                    //     parameter's storage was previously read only via
                    //     `gather_flat_grad` / `gather_flat` (returning
                    //     owned `Vec<f64>`s), so no live `&[T]` / `&mut [T]`
                    //     slice into the parameter's storage exists at this
                    //     point.
                    //  4. `(gi, pi)` indexing iterates each parameter
                    //     exactly once per scatter call.
                    unsafe {
                        self.param_groups[gi].params[pi]
                            .tensor()
                            .update_data(&new_data)
                    }
                })?;

                offset += numel;
                shape_idx += 1;
            }
        }

        Ok(())
    }

    /// L-BFGS two-loop recursion.
    ///
    /// Given the current gradient `q` and the curvature history, returns the
    /// search direction `d = -H_k * g` where `H_k` is the L-BFGS
    /// approximation to the inverse Hessian.
    fn two_loop_recursion(&self, grad: &[f64]) -> Vec<f64> {
        let m = self.state.s_history.len();
        let n = grad.len();

        // If we have no history yet, fall back to steepest descent.
        if m == 0 {
            return grad.iter().map(|&g| -g).collect();
        }

        let mut q = grad.to_vec();
        let mut alpha = vec![0.0; m];

        // ---- first loop (backward through history) ----
        for i in (0..m).rev() {
            let s = &self.state.s_history[i];
            let rho = self.state.rho_history[i];
            let a = rho * dot(s, &q);
            alpha[i] = a;
            // q = q - alpha_i * y_i
            let y = &self.state.y_history[i];
            for j in 0..n {
                q[j] -= a * y[j];
            }
        }

        // ---- initial Hessian approximation H_0 = gamma * I ----
        let s_last = &self.state.s_history[m - 1];
        let y_last = &self.state.y_history[m - 1];
        let y_dot_y = dot(y_last, y_last);
        let gamma = if y_dot_y.abs() > 1e-30 {
            dot(s_last, y_last) / y_dot_y
        } else {
            1.0
        };

        // r = H_0 * q = gamma * q
        let mut r: Vec<f64> = q.iter().map(|&v| gamma * v).collect();

        // ---- second loop (forward through history) ----
        for (i, (y, (&rho, s))) in self
            .state
            .y_history
            .iter()
            .zip(
                self.state
                    .rho_history
                    .iter()
                    .zip(self.state.s_history.iter()),
            )
            .take(m)
            .enumerate()
        {
            let beta = rho * dot(y, &r);
            for (rj, &sj) in r.iter_mut().zip(s.iter()) {
                *rj += sj * (alpha[i] - beta);
            }
        }

        // Search direction = -r (descent direction).
        for v in &mut r {
            *v = -*v;
        }

        r
    }

    /// Update the curvature history with a new (s, y) pair.
    fn update_history(&mut self, s: Vec<f64>, y: Vec<f64>) {
        let ys = dot(&s, &y);

        // Skip the update if curvature condition is not satisfied.
        if ys <= 1e-30 {
            return;
        }

        let rho = 1.0 / ys;

        // If we've reached the history limit, evict the oldest entry.
        if self.state.s_history.len() >= self.config.history_size {
            self.state.s_history.remove(0);
            self.state.y_history.remove(0);
            self.state.rho_history.remove(0);
        }

        self.state.s_history.push(s);
        self.state.y_history.push(y);
        self.state.rho_history.push(rho);
    }
}

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

/// Infinity norm (max absolute value).
#[inline]
fn inf_norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x.abs()).fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// Strong Wolfe line search (Nocedal & Wright, Algorithms 3.5 + 3.6)
// ---------------------------------------------------------------------------

/// Sufficient decrease constant (Armijo condition).
const WOLFE_C1: f64 = 1e-4;
/// Curvature condition constant (suitable for quasi-Newton methods).
const WOLFE_C2: f64 = 0.9;
/// Maximum bracket expansion factor.
const WOLFE_ALPHA_MAX: f64 = 50.0;

/// Strong Wolfe line search.
///
/// Finds a step size `alpha` along the search direction such that:
///
/// 1. **Armijo** (sufficient decrease):
///    `f(x + alpha*d) <= f(x) + c1 * alpha * g0^T * d`
///
/// 2. **Strong curvature**:
///    `|g(x + alpha*d)^T * d| <= c2 * |g0^T * d|`
///
/// The `eval_fn` is called with a candidate `alpha`, sets parameters to
/// `x0 + alpha * direction`, runs forward + backward, and returns
/// `(loss, directional_derivative)`.
///
/// Reference: Nocedal & Wright, "Numerical Optimization", 2nd ed.,
/// Algorithm 3.5 (line search) and Algorithm 3.6 (zoom).
fn strong_wolfe_search(
    f0: f64,
    g0_dot_d: f64,
    max_evals: usize,
    mut eval_fn: impl FnMut(f64) -> FerrotorchResult<(f64, f64)>,
) -> FerrotorchResult<f64> {
    let mut alpha_prev = 0.0;
    let mut f_prev = f0;
    let mut alpha = 1.0;
    let mut evals = 0;

    for i in 0..max_evals {
        let (fi, gi_dot_d) = eval_fn(alpha)?;
        evals += 1;

        // Armijo violation or non-monotone: bracket found.
        if fi > f0 + WOLFE_C1 * alpha * g0_dot_d || (i > 0 && fi >= f_prev) {
            return zoom(
                alpha_prev,
                alpha,
                f0,
                g0_dot_d,
                f_prev,
                fi,
                max_evals.saturating_sub(evals),
                &mut eval_fn,
            );
        }

        // Strong curvature condition satisfied.
        if gi_dot_d.abs() <= WOLFE_C2 * g0_dot_d.abs() {
            return Ok(alpha);
        }

        // Positive slope: bracket found in the opposite direction.
        if gi_dot_d >= 0.0 {
            return zoom(
                alpha,
                alpha_prev,
                f0,
                g0_dot_d,
                fi,
                f_prev,
                max_evals.saturating_sub(evals),
                &mut eval_fn,
            );
        }

        f_prev = fi;
        alpha_prev = alpha;
        alpha = (2.0 * alpha).min(WOLFE_ALPHA_MAX);
    }

    Ok(alpha)
}

/// Zoom phase of the Strong Wolfe line search (Algorithm 3.6).
///
/// Bisects the interval `[alpha_lo, alpha_hi]` to find a step size
/// satisfying both Wolfe conditions.
#[allow(clippy::too_many_arguments)]
fn zoom(
    mut alpha_lo: f64,
    mut alpha_hi: f64,
    f0: f64,
    g0_dot_d: f64,
    mut f_lo: f64,
    _f_hi: f64,
    max_evals: usize,
    eval_fn: &mut impl FnMut(f64) -> FerrotorchResult<(f64, f64)>,
) -> FerrotorchResult<f64> {
    for _ in 0..max_evals {
        let alpha_j = 0.5 * (alpha_lo + alpha_hi);

        let (fj, gj_dot_d) = eval_fn(alpha_j)?;

        if fj > f0 + WOLFE_C1 * alpha_j * g0_dot_d || fj >= f_lo {
            alpha_hi = alpha_j;
        } else {
            if gj_dot_d.abs() <= WOLFE_C2 * g0_dot_d.abs() {
                return Ok(alpha_j);
            }

            if gj_dot_d * (alpha_hi - alpha_lo) >= 0.0 {
                alpha_hi = alpha_lo;
            }

            alpha_lo = alpha_j;
            f_lo = fj;
        }

        if (alpha_hi - alpha_lo).abs() < 1e-12 {
            return Ok(alpha_lo);
        }
    }

    Ok(alpha_lo)
}

// ---------------------------------------------------------------------------
// step_with_closure (Strong Wolfe line search entry point)
// ---------------------------------------------------------------------------

impl<T: Float> Lbfgs<T> {
    /// One optimization step with a re-evaluatable closure.
    ///
    /// Required when `line_search_fn` is `Some(StrongWolfe)`. The closure
    /// must compute the forward and backward passes on the current
    /// parameters and return the scalar loss:
    ///
    /// ```ignore
    /// let loss = optimizer.step_with_closure(|| {
    ///     // zero_grad is called internally by the optimizer
    ///     let output = model.forward(&input)?;
    ///     let loss = loss_fn.forward(&output, &target)?;
    ///     loss.backward()?;
    ///     loss.item().map(|v| v as f64)
    /// })?;
    /// ```
    pub fn step_with_closure(
        &mut self,
        mut closure: impl FnMut() -> FerrotorchResult<f64>,
    ) -> FerrotorchResult<f64> {
        let lr = self
            .param_groups
            .first()
            .map(|g| g.lr)
            .unwrap_or(self.config.lr);

        // Evaluate closure at current point to get initial loss & gradient.
        self.zero_grad()?;
        let loss0 = closure()?;

        let (flat_params, shapes) = self.gather_params()?;
        let flat_grad = self.gather_grads()?;

        if inf_norm(&flat_grad) <= self.config.tolerance_grad {
            return Ok(loss0);
        }

        // Update curvature history from previous step.
        if let (Some(prev_params), Some(prev_grad)) = (
            self.state.prev_flat_params.take(),
            self.state.prev_flat_grad.take(),
        ) {
            let n = flat_params.len();
            let mut s = vec![0.0; n];
            let mut y = vec![0.0; n];
            for i in 0..n {
                s[i] = flat_params[i] - prev_params[i];
                y[i] = flat_grad[i] - prev_grad[i];
            }
            self.update_history(s, y);
        }

        let direction = self.two_loop_recursion(&flat_grad);
        let g0_dot_d = dot(&flat_grad, &direction);

        // Choose step size via line search or fixed lr.
        let alpha = if self.config.line_search_fn == Some(LineSearchFn::StrongWolfe) {
            let max_evals = self.config.effective_max_eval();
            let shapes_ref = &shapes;
            let params_ref = &flat_params;
            let dir_ref = &direction;

            strong_wolfe_search(loss0, g0_dot_d, max_evals, |alpha| {
                let n = params_ref.len();
                let mut candidate = vec![0.0; n];
                for i in 0..n {
                    candidate[i] = params_ref[i] + alpha * dir_ref[i];
                }
                self.scatter_params(&candidate, shapes_ref)?;

                self.zero_grad()?;
                let fi = closure()?;
                let gi = self.gather_grads()?;
                let gi_dot_d = dot(&gi, dir_ref);
                Ok((fi, gi_dot_d))
            })?
        } else {
            lr
        };

        // Apply the chosen step size.
        let n = flat_params.len();
        let mut new_params = vec![0.0; n];
        for i in 0..n {
            new_params[i] = flat_params[i] + alpha * direction[i];
        }

        self.state.prev_flat_params = Some(flat_params);
        self.state.prev_flat_grad = Some(flat_grad);
        self.state.n_iter += 1;

        self.scatter_params(&new_params, &shapes)?;

        // Re-evaluate at the final point.
        self.zero_grad()?;
        closure()
    }
}

// ---------------------------------------------------------------------------
// Optimizer trait implementation
// ---------------------------------------------------------------------------

impl<T: Float> Optimizer<T> for Lbfgs<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        if self.config.line_search_fn.is_some() {
            return Err(FerrotorchError::InvalidArgument {
                message: "L-BFGS with line search requires a closure; \
                          use optimizer.step_with_closure(|| { ... }) instead"
                    .to_string(),
            });
        }

        let lr = self
            .param_groups
            .first()
            .map(|g| g.lr)
            .unwrap_or(self.config.lr);

        let (flat_params, shapes) = self.gather_params()?;
        let flat_grad = self.gather_grads()?;

        if inf_norm(&flat_grad) <= self.config.tolerance_grad {
            return Ok(());
        }

        if let (Some(prev_params), Some(prev_grad)) = (
            self.state.prev_flat_params.take(),
            self.state.prev_flat_grad.take(),
        ) {
            let n = flat_params.len();
            let mut s = vec![0.0; n];
            let mut y = vec![0.0; n];
            for i in 0..n {
                s[i] = flat_params[i] - prev_params[i];
                y[i] = flat_grad[i] - prev_grad[i];
            }
            self.update_history(s, y);
        }

        let direction = self.two_loop_recursion(&flat_grad);

        let n = flat_params.len();
        let mut new_params = vec![0.0; n];
        for i in 0..n {
            new_params[i] = flat_params[i] + lr * direction[i];
        }

        self.state.prev_flat_params = Some(flat_params);
        self.state.prev_flat_grad = Some(flat_grad);
        self.state.n_iter += 1;

        self.scatter_params(&new_params, &shapes)
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

        // Serialize each curvature pair under its index key.
        let mut meta = HashMap::new();
        meta.insert("n_iter".to_string(), vec![self.state.n_iter as f64]);
        meta.insert(
            "history_len".to_string(),
            vec![self.state.s_history.len() as f64],
        );
        out.insert("meta".to_string(), meta);

        for (i, ((s, y), &rho)) in self
            .state
            .s_history
            .iter()
            .zip(self.state.y_history.iter())
            .zip(self.state.rho_history.iter())
            .enumerate()
        {
            let mut entry = HashMap::new();
            entry.insert("s".to_string(), s.clone());
            entry.insert("y".to_string(), y.clone());
            entry.insert("rho".to_string(), vec![rho]);
            out.insert(format!("curvature_{i}"), entry);
        }

        if let Some(ref prev_p) = self.state.prev_flat_params {
            let mut entry = HashMap::new();
            entry.insert("prev_flat_params".to_string(), prev_p.clone());
            out.insert("prev_params".to_string(), entry);
        }

        if let Some(ref prev_g) = self.state.prev_flat_grad {
            let mut entry = HashMap::new();
            entry.insert("prev_flat_grad".to_string(), prev_g.clone());
            out.insert("prev_grad".to_string(), entry);
        }

        out
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        // Load metadata.
        let meta = state
            .get("meta")
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "missing 'meta' in L-BFGS state dict".to_string(),
            })?;

        self.state.n_iter = meta
            .get("n_iter")
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0.0) as u64;

        let history_len = meta
            .get("history_len")
            .and_then(|v| v.first())
            .copied()
            .unwrap_or(0.0) as usize;

        // Load curvature pairs.
        self.state.s_history.clear();
        self.state.y_history.clear();
        self.state.rho_history.clear();

        for i in 0..history_len {
            let key = format!("curvature_{i}");
            let entry = state
                .get(&key)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing '{key}' in L-BFGS state dict"),
                })?;

            let s = entry
                .get("s")
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing 's' in {key}"),
                })?;
            let y = entry
                .get("y")
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing 'y' in {key}"),
                })?;
            let rho = entry
                .get("rho")
                .and_then(|v| v.first())
                .copied()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing 'rho' in {key}"),
                })?;

            self.state.s_history.push(s);
            self.state.y_history.push(y);
            self.state.rho_history.push(rho);
        }

        // Load previous params/grad if present.
        self.state.prev_flat_params = state
            .get("prev_params")
            .and_then(|e| e.get("prev_flat_params").cloned());
        self.state.prev_flat_grad = state
            .get("prev_grad")
            .and_then(|e| e.get("prev_flat_grad").cloned());

        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::grad_fns::arithmetic::{add, mul, pow, sub};
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Create a scalar parameter from a single f64 value.
    fn scalar_param(val: f64) -> Parameter<f64> {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap();
        Parameter::new(t)
    }

    /// Read a scalar parameter's current value.
    fn param_val(opt: &Lbfgs<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    // -----------------------------------------------------------------------
    // Simple quadratic convergence: f(x) = x^2, min at x=0
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbfgs_quadratic_convergence() {
        let p = scalar_param(5.0);
        let mut opt = Lbfgs::new(
            vec![p],
            LbfgsConfig {
                lr: 0.5,
                ..Default::default()
            },
        );

        for _ in 0..100 {
            opt.zero_grad().unwrap();

            let x = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&x, 2.0).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        let val = param_val(&opt, 0, 0);
        assert!(val.abs() < 1e-3, "expected x near 0.0, got {val}");
    }

    // -----------------------------------------------------------------------
    // Multi-dimensional quadratic: f(a, b) = a^2 + b^2, min at (0, 0)
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbfgs_multidim_quadratic() {
        let pa = scalar_param(3.0);
        let pb = scalar_param(-4.0);
        let mut opt = Lbfgs::new(
            vec![pa, pb],
            LbfgsConfig {
                lr: 0.5,
                ..Default::default()
            },
        );

        for _ in 0..100 {
            opt.zero_grad().unwrap();

            let a = opt.param_groups[0].params[0].tensor().clone();
            let b = opt.param_groups[0].params[1].tensor().clone();

            let a_sq = pow(&a, 2.0).unwrap();
            let b_sq = pow(&b, 2.0).unwrap();
            let loss = add(&a_sq, &b_sq).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        let va = param_val(&opt, 0, 0);
        let vb = param_val(&opt, 0, 1);
        assert!(va.abs() < 1e-3, "expected a near 0.0, got {va}");
        assert!(vb.abs() < 1e-3, "expected b near 0.0, got {vb}");
    }

    // -----------------------------------------------------------------------
    // Rosenbrock convergence: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
    // min at (1, 1)
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbfgs_rosenbrock_convergence() {
        let px = scalar_param(-1.0);
        let py = scalar_param(1.0);

        let mut opt = Lbfgs::new(
            vec![px, py],
            LbfgsConfig {
                lr: 0.001,
                history_size: 10,
                ..Default::default()
            },
        );

        for _ in 0..8000 {
            opt.zero_grad().unwrap();

            let x = opt.param_groups[0].params[0].tensor().clone();
            let y = opt.param_groups[0].params[1].tensor().clone();

            let one =
                Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
            let hundred =
                Tensor::from_storage(TensorStorage::cpu(vec![100.0_f64]), vec![], false).unwrap();

            // term1 = (1 - x)^2
            let diff1 = sub(&one, &x).unwrap();
            let term1 = pow(&diff1, 2.0).unwrap();

            // term2 = 100 * (y - x^2)^2
            let x_sq = pow(&x, 2.0).unwrap();
            let diff2 = sub(&y, &x_sq).unwrap();
            let diff2_sq = pow(&diff2, 2.0).unwrap();
            let term2 = mul(&hundred, &diff2_sq).unwrap();

            let loss = add(&term1, &term2).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        let final_x = param_val(&opt, 0, 0);
        let final_y = param_val(&opt, 0, 1);

        assert!(
            (final_x - 1.0).abs() < 0.1,
            "expected x near 1.0, got {final_x}"
        );
        assert!(
            (final_y - 1.0).abs() < 0.1,
            "expected y near 1.0, got {final_y}"
        );
    }

    // -----------------------------------------------------------------------
    // zero_grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbfgs_zero_grad() {
        let p = scalar_param(3.0);
        let mut opt = Lbfgs::new(vec![p], LbfgsConfig::default());

        // Manually set a gradient.
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

    // -----------------------------------------------------------------------
    // state_dict / load_state_dict roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbfgs_state_dict_roundtrip() {
        // Use Rosenbrock so the optimizer accumulates multiple distinct
        // curvature pairs (a pure quadratic converges too fast for this).
        let px = scalar_param(-1.0);
        let py = scalar_param(1.0);
        let mut opt = Lbfgs::new(
            vec![px, py],
            LbfgsConfig {
                lr: 0.001,
                history_size: 10,
                ..Default::default()
            },
        );

        let num_steps = 20;
        for _ in 0..num_steps {
            opt.zero_grad().unwrap();

            let x = opt.param_groups[0].params[0].tensor().clone();
            let y = opt.param_groups[0].params[1].tensor().clone();

            let one =
                Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
            let hundred =
                Tensor::from_storage(TensorStorage::cpu(vec![100.0_f64]), vec![], false).unwrap();

            let diff1 = sub(&one, &x).unwrap();
            let term1 = pow(&diff1, 2.0).unwrap();
            let x_sq = pow(&x, 2.0).unwrap();
            let diff2 = sub(&y, &x_sq).unwrap();
            let diff2_sq = pow(&diff2, 2.0).unwrap();
            let term2 = mul(&hundred, &diff2_sq).unwrap();
            let loss = add(&term1, &term2).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        // Save state.
        let saved = opt.state_dict();
        assert!(
            !saved.is_empty(),
            "state dict should be non-empty after steps"
        );
        assert!(saved.contains_key("meta"));

        let meta = &saved["meta"];
        let n_iter = meta["n_iter"][0] as u64;
        assert_eq!(n_iter, num_steps as u64);

        let history_len = meta["history_len"][0] as usize;
        assert!(
            history_len > 0,
            "should have accumulated at least one curvature pair"
        );

        // Load into a fresh optimizer.
        let p2x = scalar_param(-1.0);
        let p2y = scalar_param(1.0);
        let mut opt2 = Lbfgs::new(vec![p2x, p2y], LbfgsConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2.state_dict();
        assert_eq!(loaded["meta"]["n_iter"], saved["meta"]["n_iter"]);
        assert_eq!(loaded["meta"]["history_len"], saved["meta"]["history_len"]);

        // Verify every curvature pair round-trips correctly.
        for i in 0..history_len {
            let key = format!("curvature_{i}");
            assert_eq!(loaded[&key]["s"], saved[&key]["s"]);
            assert_eq!(loaded[&key]["y"], saved[&key]["y"]);
            assert_eq!(loaded[&key]["rho"], saved[&key]["rho"]);
        }
    }

    // -----------------------------------------------------------------------
    // LR accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbfgs_lr_accessors() {
        let p = scalar_param(1.0);
        let mut opt = Lbfgs::new(
            vec![p],
            LbfgsConfig {
                lr: 0.5,
                ..Default::default()
            },
        );

        assert!((opt.lr() - 0.5).abs() < 1e-12);

        opt.set_lr(0.1);
        assert!((opt.lr() - 0.1).abs() < 1e-12);
        assert!((opt.param_groups()[0].lr - 0.1).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Default config
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbfgs_default_config() {
        let config = LbfgsConfig::default();
        assert!((config.lr - 1.0).abs() < 1e-12);
        assert_eq!(config.max_iter, 20);
        assert!(config.max_eval.is_none());
        assert!((config.tolerance_grad - 1e-7).abs() < 1e-15);
        assert!((config.tolerance_change - 1e-9).abs() < 1e-15);
        assert_eq!(config.history_size, 10);
        assert!(config.line_search_fn.is_none());
        assert_eq!(config.effective_max_eval(), 25); // 20 * 5 / 4
    }

    // -----------------------------------------------------------------------
    // History eviction
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbfgs_history_eviction() {
        let p = scalar_param(10.0);
        let mut opt = Lbfgs::new(
            vec![p],
            LbfgsConfig {
                lr: 0.1,
                history_size: 3,
                ..Default::default()
            },
        );

        // Run more steps than the history size.
        for _ in 0..10 {
            opt.zero_grad().unwrap();
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
        }

        // History should be capped at 3.
        assert!(
            opt.state.s_history.len() <= 3,
            "history should be capped at 3, got {}",
            opt.state.s_history.len()
        );
    }

    // -----------------------------------------------------------------------
    // step() with line_search_fn errors (must use step_with_closure)
    // -----------------------------------------------------------------------

    #[test]
    fn test_lbfgs_step_requires_closure_for_line_search() {
        let p = scalar_param(1.0);
        let mut opt = Lbfgs::new(
            vec![p],
            LbfgsConfig {
                line_search_fn: Some(LineSearchFn::StrongWolfe),
                ..Default::default()
            },
        );

        let grad = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        opt.param_groups[0].params[0]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();

        let result = opt.step();
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Strong Wolfe line search: convergence on quadratic
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Helper: run step_with_closure using external param access to avoid
    // borrow conflict (closure captures params by clone, not by ref to opt).
    // -----------------------------------------------------------------------

    /// Run L-BFGS with step_with_closure on f(x) = x^2.
    /// Params are accessed through the optimizer's scatter/gather internally;
    /// the closure reads them by cloning from the param group.
    fn run_quadratic_with_closure(line_search: Option<LineSearchFn>, lr: f64, steps: usize) -> f64 {
        let p = scalar_param(5.0);
        let mut opt = Lbfgs::new(
            vec![p],
            LbfgsConfig {
                lr,
                line_search_fn: line_search,
                ..Default::default()
            },
        );

        // We need a shared reference to the parameter that the closure can
        // access independently of &mut opt. Clone the tensor each iteration
        // by reading through a raw pointer — safe because step_with_closure
        // only mutates params between closure calls, not during.
        //
        // The idiomatic way: extract the param pointer before the loop.
        // step_with_closure sets params internally, so we read from the
        // param_groups after each scatter.
        for _ in 0..steps {
            // Read current param value BEFORE the mutable borrow.
            let param_ptr = &opt.param_groups[0].params[0] as *const Parameter<f64>;

            opt.step_with_closure(|| {
                // SAFETY: We dereference the raw pointer `param_ptr` to read
                // the parameter's `Tensor<f64>` via `.tensor().clone()`.
                // The pointee is `opt.param_groups[0].params[0]: Parameter<f64>`.
                //  - Liveness: `opt` is on the enclosing function's stack and
                //    is not moved/dropped between the pointer take and the
                //    deref; the closure runs synchronously inside
                //    `step_with_closure`, so `opt` outlives this read.
                //  - Validity of the parameter slot: `step_with_closure`
                //    scatters new flat values into the existing `Parameter`
                //    handles before invoking the closure (it does not
                //    replace the `Parameter` themselves), so the address at
                //    `params[0]` remains valid for reads and the underlying
                //    `Tensor`'s storage Arc is intact.
                //  - Aliasing: this is a test-only escape hatch. A shared
                //    `&Parameter<f64>` is reborrowed concurrently with
                //    `step_with_closure`'s `&mut self` on `opt`. The
                //    optimizer's `&mut self` does not actually mutate the
                //    `Parameter` slot during the closure (only the
                //    underlying tensor data via `update_data`), so under
                //    Tree Borrows this read is sound, but it is on the
                //    boundary of stacked borrows. FOLLOW-UP: replace this
                //    test scaffolding with a Parameter clone passed by
                //    value into the closure (Parameter wraps `Arc<Inner>`,
                //    so the clone shares the same storage cheaply).
                let x = unsafe { &*param_ptr }.tensor().clone();
                let loss = pow(&x, 2.0).unwrap();
                loss.backward().unwrap();
                loss.item()
            })
            .unwrap();
        }

        param_val(&opt, 0, 0)
    }

    #[test]
    fn test_strong_wolfe_quadratic() {
        let val = run_quadratic_with_closure(Some(LineSearchFn::StrongWolfe), 1.0, 50);
        assert!(
            val.abs() < 1e-3,
            "Strong Wolfe: expected x near 0, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Strong Wolfe line search: convergence on Rosenbrock
    // -----------------------------------------------------------------------

    #[test]
    fn test_strong_wolfe_rosenbrock() {
        let px = scalar_param(-1.0);
        let py = scalar_param(1.0);

        let mut opt = Lbfgs::new(
            vec![px, py],
            LbfgsConfig {
                lr: 1.0,
                line_search_fn: Some(LineSearchFn::StrongWolfe),
                history_size: 10,
                ..Default::default()
            },
        );

        for _ in 0..5000 {
            let px_ptr = &opt.param_groups[0].params[0] as *const Parameter<f64>;
            let py_ptr = &opt.param_groups[0].params[1] as *const Parameter<f64>;

            opt.step_with_closure(|| {
                // SAFETY: same justification as the quadratic test above —
                // `px_ptr` and `py_ptr` are raw pointers into
                // `opt.param_groups[0].params[{0,1}]`, both of which:
                //  - point into `opt`, which lives on this stack frame and
                //    is not dropped/moved before the closure returns;
                //  - reference distinct slots (`params[0]` vs. `params[1]`),
                //    so deref-of-`px_ptr` and deref-of-`py_ptr` do not
                //    alias each other;
                //  - point at `Parameter` slots whose addresses are stable
                //    across `step_with_closure` (it scatters new flat values
                //    into the existing parameters; it does not relocate
                //    them).
                // Aliasing caveat (same as quadratic): each shared reborrow
                // is concurrent with `step_with_closure`'s `&mut self` on
                // `opt`. The optimizer does not mutate the `Parameter` slot
                // through that `&mut`, only the underlying tensor data. See
                // the FOLLOW-UP note above (replace with cloned Parameter
                // captured by value).
                let x = unsafe { &*px_ptr }.tensor().clone();
                let y = unsafe { &*py_ptr }.tensor().clone();

                let one =
                    Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
                let hundred =
                    Tensor::from_storage(TensorStorage::cpu(vec![100.0_f64]), vec![], false)
                        .unwrap();

                let diff1 = sub(&one, &x).unwrap();
                let term1 = pow(&diff1, 2.0).unwrap();
                let x_sq = pow(&x, 2.0).unwrap();
                let diff2 = sub(&y, &x_sq).unwrap();
                let diff2_sq = pow(&diff2, 2.0).unwrap();
                let term2 = mul(&hundred, &diff2_sq).unwrap();
                let loss = add(&term1, &term2).unwrap();
                loss.backward().unwrap();
                loss.item()
            })
            .unwrap();
        }

        let final_x = param_val(&opt, 0, 0);
        let final_y = param_val(&opt, 0, 1);

        assert!(
            (final_x - 1.0).abs() < 0.1,
            "Strong Wolfe Rosenbrock: expected x near 1.0, got {final_x}"
        );
        assert!(
            (final_y - 1.0).abs() < 0.1,
            "Strong Wolfe Rosenbrock: expected y near 1.0, got {final_y}"
        );
    }

    // -----------------------------------------------------------------------
    // step_with_closure works without line search too (fixed step)
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_with_closure_fixed_step() {
        let val = run_quadratic_with_closure(None, 0.5, 100);
        assert!(
            val.abs() < 1e-3,
            "closure fixed-step: expected x near 0, got {val}"
        );
    }
}
