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

use std::collections::{HashMap, VecDeque};

use ferrotorch_core::creation::scalar;
use ferrotorch_core::grad_fns::arithmetic::{add, mul, neg, sub};
use ferrotorch_core::grad_fns::reduction::sum as tensor_sum;
use ferrotorch_core::grad_fns::shape::{cat as tensor_cat, flatten as tensor_flatten};
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage, no_grad};
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

    /// Set the learning rate / step size.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the maximum number of iterations per `step()` call.
    #[must_use]
    pub fn with_max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the maximum number of function evaluations per `step()` call.
    #[must_use]
    pub fn with_max_eval(mut self, max_eval: Option<usize>) -> Self {
        self.max_eval = max_eval;
        self
    }

    /// Set the termination tolerance on the gradient infinity norm.
    #[must_use]
    pub fn with_tolerance_grad(mut self, tolerance_grad: f64) -> Self {
        self.tolerance_grad = tolerance_grad;
        self
    }

    /// Set the termination tolerance on the function value change.
    #[must_use]
    pub fn with_tolerance_change(mut self, tolerance_change: f64) -> Self {
        self.tolerance_change = tolerance_change;
        self
    }

    /// Set the number of curvature pairs to keep.
    #[must_use]
    pub fn with_history_size(mut self, history_size: usize) -> Self {
        self.history_size = history_size;
        self
    }

    /// Set the line search function.
    #[must_use]
    pub fn with_line_search_fn(mut self, line_search_fn: Option<LineSearchFn>) -> Self {
        self.line_search_fn = line_search_fn;
        self
    }

    /// Set the maximize flag (when `true`, negate the gradient to maximize).
    #[must_use]
    pub fn with_maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }
}

// ---------------------------------------------------------------------------
// LbfgsState
// ---------------------------------------------------------------------------

/// Mutable state maintained across L-BFGS iterations.
///
/// CL-1105 Pattern B: history slots and prev-step caches are 1-D
/// device-resident [`Tensor<T>`] instead of flat `Vec<f64>` so the two-loop
/// recursion runs entirely on the parameter's device.
///
/// CL-1125: the history buffers are [`VecDeque`] (not `Vec`) so eviction of
/// the oldest entry when `history_size` is reached is O(1) (`pop_front`)
/// instead of `Vec::remove(0)`'s O(history_size) memmove of every later
/// element. Random indexed reads (`[i]`) and `iter()` work the same way
/// on `VecDeque`, so callers are unaffected.
#[derive(Debug)]
struct LbfgsState<T: Float> {
    /// Parameter differences: s_k = x_{k+1} - x_k. 1-D tensors, on device.
    s_history: VecDeque<Tensor<T>>,
    /// Gradient differences: y_k = g_{k+1} - g_k. 1-D tensors, on device.
    y_history: VecDeque<Tensor<T>>,
    /// Cached 1 / (y_k . s_k) for each curvature pair. Stored as f64 so the
    /// two-loop recursion's gamma/alpha/beta scalars retain precision when
    /// T = f32.
    rho_history: VecDeque<f64>,
    /// Previous flat parameter vector (needed to compute s_k).
    prev_flat_params: Option<Tensor<T>>,
    /// Previous flat gradient vector (needed to compute y_k).
    prev_flat_grad: Option<Tensor<T>>,
    /// Number of optimizer steps completed.
    n_iter: u64,
}

impl<T: Float> LbfgsState<T> {
    fn new() -> Self {
        Self {
            s_history: VecDeque::new(),
            y_history: VecDeque::new(),
            rho_history: VecDeque::new(),
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
    state: LbfgsState<T>,
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

    /// Flatten all parameter data into a single 1-D [`Tensor<T>`] on the
    /// parameter's device. Returns the concatenated tensor plus the per-
    /// parameter shapes (used by `scatter_params` to undo the flatten).
    ///
    /// CL-1105 Pattern B: composes `flatten` (zero-copy view) and `cat`
    /// (GPU fast-path on CUDA) — no `data_vec()` round-trip.
    fn gather_params(&self) -> FerrotorchResult<(Tensor<T>, Vec<Vec<usize>>)> {
        let mut flats: Vec<Tensor<T>> = Vec::new();
        let mut shapes = Vec::new();
        for group in &self.param_groups {
            for param in &group.params {
                let tensor = param.tensor();
                shapes.push(tensor.shape().to_vec());
                // `flatten` is a zero-copy view that works on any device.
                // `contiguous()` ensures the storage layout matches the
                // logical element order so the cat fast-path stays
                // device-resident.
                let contig = tensor.contiguous()?;
                let flat_view = no_grad(|| tensor_flatten(&contig))?;
                flats.push(flat_view);
            }
        }
        let combined = no_grad(|| tensor_cat(&flats, 0))?;
        Ok((combined, shapes))
    }

    /// Flatten all parameter gradients into a single 1-D device-resident
    /// [`Tensor<T>`].
    ///
    /// When `config.maximize` is set, the gradient is negated. CL-321
    fn gather_grads(&self) -> FerrotorchResult<Tensor<T>> {
        let negate = self.config.maximize;
        let mut flats: Vec<Tensor<T>> = Vec::new();
        for group in &self.param_groups {
            for param in &group.params {
                let tensor = param.tensor();
                match tensor.grad()? {
                    Some(g) => {
                        let g_contig = g.contiguous()?;
                        let g_flat = no_grad(|| tensor_flatten(&g_contig))?;
                        if negate {
                            flats.push(no_grad(|| neg(&g_flat))?);
                        } else {
                            flats.push(g_flat);
                        }
                    }
                    None => {
                        // No gradient: treat as zero on the parameter's
                        // device.
                        let numel = tensor.numel();
                        let device = tensor.device();
                        let zeros =
                            ferrotorch_core::creation::zeros::<T>(&[numel])?.to(device)?;
                        flats.push(zeros);
                    }
                }
            }
        }
        let combined = no_grad(|| tensor_cat(&flats, 0))?;
        Ok(combined)
    }

    /// Scatter a flat 1-D [`Tensor<T>`] back into the parameter tensors
    /// (inside `no_grad`).
    fn scatter_params(
        &mut self,
        flat: &Tensor<T>,
        shapes: &[Vec<usize>],
    ) -> FerrotorchResult<()> {
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

                no_grad(|| -> FerrotorchResult<()> {
                    // narrow gives us a zero-copy 1-D view over [offset,
                    // offset+numel); reshape it back to the original
                    // shape (also zero-copy when contiguous).
                    let chunk = flat.narrow(0, offset, numel)?;
                    let chunk_contig = chunk.contiguous()?;
                    let reshaped: Vec<isize> =
                        shape.iter().map(|&d| d as isize).collect();
                    let chunk_reshaped = if shape.is_empty() {
                        // 0-D scalar: leave as 1-element 1-D tensor and
                        // reshape via `view_reshape` (handles empty shape).
                        chunk_contig.view_reshape(vec![])?
                    } else {
                        chunk_contig.reshape_t(&reshaped)?
                    };

                    let (storage, _) = chunk_reshaped.into_storage_and_shape()?;
                    // SAFETY: same as Muon::step.
                    //  1. `scatter_params(&mut self, ..)` holds `&mut
                    //     self`; no other handle can call into this
                    //     optimizer concurrently.
                    //  2. We are inside `no_grad`, so no autograd node
                    //     retains a clone of the parameter's storage Arc.
                    //  3. The narrow + reshape produced a fresh tensor
                    //     whose storage Arc we just consumed via
                    //     `into_storage_and_shape`, so the only handle to
                    //     `storage` is local.
                    //  4. The (gi, pi) loop iterates each parameter
                    //     exactly once per call.
                    unsafe {
                        self.param_groups[gi].params[pi]
                            .tensor()
                            .update_storage(storage)?;
                    }
                    Ok(())
                })?;

                offset += numel;
                shape_idx += 1;
            }
        }

        Ok(())
    }

    /// L-BFGS two-loop recursion (device-resident).
    ///
    /// Given the current gradient `q` (1-D tensor) and the curvature
    /// history, returns the search direction `d = -H_k * g` where `H_k` is
    /// the L-BFGS approximation to the inverse Hessian.
    ///
    /// CL-1105 Pattern B: dot products are `sum(s * q)`, scalar updates use
    /// `mul`/`sub`/`add` over scalar tensors broadcast against the 1-D
    /// flat tensor. All ops dispatch on the gradient's device.
    fn two_loop_recursion(&self, grad: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let m = self.state.s_history.len();
        let device = grad.device();

        // If we have no history yet, fall back to steepest descent.
        if m == 0 {
            return no_grad(|| neg(grad));
        }

        let mut q = grad.clone();
        let mut alpha: Vec<f64> = vec![0.0; m];

        // ---- first loop (backward through history) ----
        for i in (0..m).rev() {
            let s = &self.state.s_history[i];
            let rho = self.state.rho_history[i];
            let s_dot_q = dot_tensor(s, &q)?;
            let a = rho * s_dot_q;
            alpha[i] = a;

            // q = q - a * y_i
            let y = &self.state.y_history[i];
            let a_t = scalar(cast::<f64, T>(a)?)?.to(device)?;
            let scaled = no_grad(|| mul(y, &a_t))?;
            q = no_grad(|| sub(&q, &scaled))?;
        }

        // ---- initial Hessian approximation H_0 = gamma * I ----
        let s_last = &self.state.s_history[m - 1];
        let y_last = &self.state.y_history[m - 1];
        let y_dot_y = dot_tensor(y_last, y_last)?;
        let gamma = if y_dot_y.abs() > 1e-30 {
            dot_tensor(s_last, y_last)? / y_dot_y
        } else {
            1.0
        };

        // r = H_0 * q = gamma * q
        let gamma_t = scalar(cast::<f64, T>(gamma)?)?.to(device)?;
        let mut r = no_grad(|| mul(&q, &gamma_t))?;

        // ---- second loop (forward through history) ----
        // `alpha[i]` is read-only here; `enumerate().take(m)` keeps the index
        // available for the parallel `s_history[i]` / `y_history[i]` /
        // `rho_history[i]` accesses while satisfying clippy::needless_range_loop.
        for (i, &alpha_i) in alpha.iter().enumerate().take(m) {
            let y = &self.state.y_history[i];
            let s = &self.state.s_history[i];
            let rho = self.state.rho_history[i];
            let beta = rho * dot_tensor(y, &r)?;
            // r = r + (alpha[i] - beta) * s
            let coeff = alpha_i - beta;
            let coeff_t = scalar(cast::<f64, T>(coeff)?)?.to(device)?;
            let scaled = no_grad(|| mul(s, &coeff_t))?;
            r = no_grad(|| add(&r, &scaled))?;
        }

        // Search direction = -r (descent direction).
        no_grad(|| neg(&r))
    }

    /// Update the curvature history with a new (s, y) pair (1-D tensors).
    fn update_history(&mut self, s: Tensor<T>, y: Tensor<T>) -> FerrotorchResult<()> {
        let ys = dot_tensor(&s, &y)?;

        // Skip the update if curvature condition is not satisfied.
        if ys <= 1e-30 {
            return Ok(());
        }

        let rho = 1.0 / ys;

        // If we've reached the history limit, evict the oldest entry.
        //
        // CL-1125: `VecDeque::pop_front` is O(1); the previous
        // `Vec::remove(0)` was O(history_size) per call because every later
        // element had to shift down. With `history_size = 10` (the typical
        // L-BFGS setting) that is up to 10 memmove copies per step that
        // are now eliminated.
        if self.state.s_history.len() >= self.config.history_size {
            self.state.s_history.pop_front();
            self.state.y_history.pop_front();
            self.state.rho_history.pop_front();
        }

        self.state.s_history.push_back(s);
        self.state.y_history.push_back(y);
        self.state.rho_history.push_back(rho);
        Ok(())
    }
}

/// Dot product of two 1-D tensors via device-resident `sum(a * b)`.
///
/// Scalar result is downloaded once at the end so the recursion can use
/// it as a regular f64; the heavy element-wise mul stays on device.
fn dot_tensor<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<f64> {
    let prod = no_grad(|| mul(a, b))?;
    let s = no_grad(|| tensor_sum(&prod))?;
    let s_cpu = if s.is_cuda() { s.cpu()? } else { s };
    let v = s_cpu.data()?[0];
    cast::<T, f64>(v)
}

/// Infinity norm (max absolute value) of a 1-D tensor.
///
/// Currently dispatches to `data_vec()` for the final reduction because no
/// device-resident `max(abs())` primitive exists yet; the per-step cost is
/// O(n) and runs once per step. Lift this when a reduction-max GPU kernel
/// lands.
fn inf_norm<T: Float>(v: &Tensor<T>) -> FerrotorchResult<f64> {
    let v_cpu = if v.is_cuda() { v.cpu()? } else { v.clone() };
    let data = v_cpu.data_vec()?;
    let mut m = 0.0_f64;
    for x in data {
        let xf = cast::<T, f64>(x)?;
        if xf.abs() > m {
            m = xf.abs();
        }
    }
    Ok(m)
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
        let device = flat_params.device();

        if inf_norm(&flat_grad)? <= self.config.tolerance_grad {
            return Ok(loss0);
        }

        // Update curvature history from previous step.
        if let (Some(prev_params), Some(prev_grad)) = (
            self.state.prev_flat_params.take(),
            self.state.prev_flat_grad.take(),
        ) {
            let s = no_grad(|| sub(&flat_params, &prev_params))?;
            let y = no_grad(|| sub(&flat_grad, &prev_grad))?;
            self.update_history(s, y)?;
        }

        let direction = self.two_loop_recursion(&flat_grad)?;
        let g0_dot_d = dot_tensor(&flat_grad, &direction)?;

        // Choose step size via line search or fixed lr.
        let alpha = if self.config.line_search_fn == Some(LineSearchFn::StrongWolfe) {
            let max_evals = self.config.effective_max_eval();
            let shapes_ref = &shapes;
            let params_ref = &flat_params;
            let dir_ref = &direction;

            strong_wolfe_search(loss0, g0_dot_d, max_evals, |alpha| {
                let alpha_t = scalar(cast::<f64, T>(alpha)?)?.to(device)?;
                let scaled = no_grad(|| mul(dir_ref, &alpha_t))?;
                let candidate = no_grad(|| add(params_ref, &scaled))?;
                self.scatter_params(&candidate, shapes_ref)?;

                self.zero_grad()?;
                let fi = closure()?;
                let gi = self.gather_grads()?;
                let gi_dot_d = dot_tensor(&gi, dir_ref)?;
                Ok((fi, gi_dot_d))
            })?
        } else {
            lr
        };

        // Apply the chosen step size: new_params = flat_params + alpha * direction
        let alpha_t = scalar(cast::<f64, T>(alpha)?)?.to(device)?;
        let scaled_dir = no_grad(|| mul(&direction, &alpha_t))?;
        let new_params = no_grad(|| add(&flat_params, &scaled_dir))?;

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
    /// Run one optimizer step.
    ///
    /// CL-1105: device-resident Pattern B. `gather_params`/`gather_grads`
    /// produce a single 1-D tensor on the parameter's device via cat;
    /// the two-loop recursion runs entirely on-device; the parameter
    /// update commits via `scatter_params` -> `update_storage`.
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
        let device = flat_params.device();

        if inf_norm(&flat_grad)? <= self.config.tolerance_grad {
            return Ok(());
        }

        if let (Some(prev_params), Some(prev_grad)) = (
            self.state.prev_flat_params.take(),
            self.state.prev_flat_grad.take(),
        ) {
            let s = no_grad(|| sub(&flat_params, &prev_params))?;
            let y = no_grad(|| sub(&flat_grad, &prev_grad))?;
            self.update_history(s, y)?;
        }

        let direction = self.two_loop_recursion(&flat_grad)?;

        // new_params = flat_params + lr * direction
        let lr_t = scalar(cast::<f64, T>(lr)?)?.to(device)?;
        let scaled_dir = no_grad(|| mul(&direction, &lr_t))?;
        let new_params = no_grad(|| add(&flat_params, &scaled_dir))?;

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

    fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
        let mut out = OptimizerState::new();

        // Serialize each curvature pair under its index key. Tensors are
        // downloaded once at checkpoint time and cast to f64 for a
        // dtype-agnostic on-disk format.
        let mut meta = HashMap::new();
        meta.insert("n_iter".to_string(), vec![self.state.n_iter as f64]);
        meta.insert(
            "history_len".to_string(),
            vec![self.state.s_history.len() as f64],
        );
        out.insert("meta".to_string(), meta);

        let tensor_to_f64 = |t: &Tensor<T>| -> FerrotorchResult<Vec<f64>> {
            let cpu = if t.is_cuda() { t.cpu()? } else { t.clone() };
            cpu.data_vec()?
                .iter()
                .map(|&v| cast::<T, f64>(v))
                .collect::<FerrotorchResult<Vec<f64>>>()
        };

        for (i, ((s, y), &rho)) in self
            .state
            .s_history
            .iter()
            .zip(self.state.y_history.iter())
            .zip(self.state.rho_history.iter())
            .enumerate()
        {
            let mut entry = HashMap::new();
            entry.insert("s".to_string(), tensor_to_f64(s)?);
            entry.insert("y".to_string(), tensor_to_f64(y)?);
            entry.insert("rho".to_string(), vec![rho]);
            out.insert(format!("curvature_{i}"), entry);
        }

        if let Some(ref prev_p) = self.state.prev_flat_params {
            let mut entry = HashMap::new();
            entry.insert("prev_flat_params".to_string(), tensor_to_f64(prev_p)?);
            out.insert("prev_params".to_string(), entry);
        }

        if let Some(ref prev_g) = self.state.prev_flat_grad {
            let mut entry = HashMap::new();
            entry.insert("prev_flat_grad".to_string(), tensor_to_f64(prev_g)?);
            out.insert("prev_grad".to_string(), entry);
        }

        Ok(out)
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

            let s_f64 = entry
                .get("s")
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing 's' in {key}"),
                })?;
            let y_f64 = entry
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

            let s_t: Vec<T> = s_f64
                .iter()
                .map(|&v| cast::<f64, T>(v))
                .collect::<FerrotorchResult<Vec<T>>>()?;
            let s_len = s_t.len();
            let s_tensor =
                Tensor::from_storage(TensorStorage::cpu(s_t), vec![s_len], false)?;
            let y_t: Vec<T> = y_f64
                .iter()
                .map(|&v| cast::<f64, T>(v))
                .collect::<FerrotorchResult<Vec<T>>>()?;
            let y_len = y_t.len();
            let y_tensor =
                Tensor::from_storage(TensorStorage::cpu(y_t), vec![y_len], false)?;

            self.state.s_history.push_back(s_tensor);
            self.state.y_history.push_back(y_tensor);
            self.state.rho_history.push_back(rho);
        }

        // Load previous params/grad if present.
        self.state.prev_flat_params = match state
            .get("prev_params")
            .and_then(|e| e.get("prev_flat_params"))
        {
            Some(v) => {
                let t: Vec<T> = v
                    .iter()
                    .map(|&x| cast::<f64, T>(x))
                    .collect::<FerrotorchResult<Vec<T>>>()?;
                let n = t.len();
                Some(Tensor::from_storage(TensorStorage::cpu(t), vec![n], false)?)
            }
            None => None,
        };
        self.state.prev_flat_grad = match state
            .get("prev_grad")
            .and_then(|e| e.get("prev_flat_grad"))
        {
            Some(v) => {
                let t: Vec<T> = v
                    .iter()
                    .map(|&x| cast::<f64, T>(x))
                    .collect::<FerrotorchResult<Vec<T>>>()?;
                let n = t.len();
                Some(Tensor::from_storage(TensorStorage::cpu(t), vec![n], false)?)
            }
            None => None,
        };

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
        let saved = opt
            .state_dict()
            .expect("lbfgs state_dict must succeed in test");
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

        let loaded = opt2
            .state_dict()
            .expect("lbfgs state_dict round-trip must succeed in test");
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

        // The closure needs to read the live parameter value but cannot hold
        // `&opt` because `step_with_closure` takes `&mut self`. We clone the
        // `Parameter` once per iteration and move the clone into the closure.
        // `Parameter` wraps `Arc<Tensor<Arc<TensorInner>>>`, so the clone
        // shares the same storage Arc as `opt.param_groups[0].params[0]`;
        // when `step_with_closure` scatters new flat values into the
        // parameter via `update_data`/`update_storage` (which mutate through
        // `Arc::as_ptr`), the cloned `Parameter` observes those mutations
        // because it points at the same `Arc<TensorInner>`. This is the
        // safe equivalent of the previous raw-pointer read.
        for _ in 0..steps {
            let param = opt.param_groups[0].params[0].clone();
            opt.step_with_closure(move || {
                let x = param.tensor().clone();
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
            // Same construction as `run_quadratic_with_closure`: clone each
            // `Parameter` once per iteration and move the clones into the
            // closure. The clones share the same `Arc<TensorInner>` as
            // `opt.param_groups[0].params[{0,1}]`, so writes performed by
            // `step_with_closure` through `update_data`/`update_storage`
            // are observed by these reads.
            let px_clone = opt.param_groups[0].params[0].clone();
            let py_clone = opt.param_groups[0].params[1].clone();

            opt.step_with_closure(move || {
                let x = px_clone.tensor().clone();
                let y = py_clone.tensor().clone();

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

    // -----------------------------------------------------------------------
    // CL-1105 Pattern B — CUDA device-resident step tests.
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn try_init_cuda() -> bool {
        match ferrotorch_gpu::init_cuda_backend() {
            Ok(_) => true,
            Err(e) => {
                eprintln!("[cascade_skip] no CUDA device: {e}");
                false
            }
        }
    }

    /// Construct a CUDA-resident parameter from a CPU scalar value.
    #[cfg(feature = "cuda")]
    fn cuda_scalar_param(val: f64) -> Parameter<f64> {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap();
        let t_gpu = t.cuda().unwrap();
        Parameter::new(t_gpu)
    }

    /// L-BFGS step must keep CUDA-resident parameters on CUDA. The
    /// two-loop recursion and `scatter_params` must commit on-device.
    #[cfg(feature = "cuda")]
    #[test]
    fn lbfgs_step_preserves_device_for_cuda_input() {
        if !try_init_cuda() {
            return;
        }
        let p = cuda_scalar_param(5.0);
        let mut opt = Lbfgs::new(
            vec![p],
            LbfgsConfig {
                lr: 0.5,
                ..Default::default()
            },
        );

        // One forward/backward/step cycle on CUDA.
        opt.zero_grad().unwrap();
        let x = opt.param_groups[0].params[0].tensor().clone();
        let loss = pow(&x, 2.0).unwrap();
        loss.backward().unwrap();
        opt.step().unwrap();

        let after = &opt.param_groups[0].params[0];
        assert!(
            after.tensor().is_cuda(),
            "Lbfgs::step must preserve CUDA residence; got device {:?}",
            after.tensor().device()
        );
        assert_eq!(after.tensor().device(), ferrotorch_core::Device::Cuda(0));
    }

    /// L-BFGS CUDA run must match CPU reference within tolerance.
    #[cfg(feature = "cuda")]
    #[test]
    fn lbfgs_step_matches_cpu_within_tolerance() {
        if !try_init_cuda() {
            return;
        }
        // CPU reference: minimize x^2 from x=5 for 20 steps.
        let p_cpu = scalar_param(5.0);
        let mut opt_cpu = Lbfgs::new(
            vec![p_cpu],
            LbfgsConfig {
                lr: 0.5,
                ..Default::default()
            },
        );
        for _ in 0..20 {
            opt_cpu.zero_grad().unwrap();
            let x = opt_cpu.param_groups[0].params[0].tensor().clone();
            let loss = pow(&x, 2.0).unwrap();
            loss.backward().unwrap();
            opt_cpu.step().unwrap();
        }
        let cpu_val = param_val(&opt_cpu, 0, 0);

        // CUDA run with the same initial value and config.
        let p_gpu = cuda_scalar_param(5.0);
        let mut opt_gpu = Lbfgs::new(
            vec![p_gpu],
            LbfgsConfig {
                lr: 0.5,
                ..Default::default()
            },
        );
        for _ in 0..20 {
            opt_gpu.zero_grad().unwrap();
            let x = opt_gpu.param_groups[0].params[0].tensor().clone();
            let loss = pow(&x, 2.0).unwrap();
            loss.backward().unwrap();
            opt_gpu.step().unwrap();
        }
        let gpu_t = opt_gpu.param_groups[0].params[0]
            .tensor()
            .cpu()
            .unwrap();
        let gpu_val = gpu_t.data().unwrap()[0];

        assert!(
            (cpu_val - gpu_val).abs() < 1e-6,
            "Lbfgs CPU/GPU mismatch: cpu={cpu_val}, gpu={gpu_val}"
        );
    }
}
