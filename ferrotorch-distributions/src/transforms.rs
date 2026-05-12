//! Bijective transforms for distribution reparameterization.
//!
//! Transforms map between spaces (e.g., real line to positive reals) and
//! compute the log-absolute-determinant of the Jacobian needed for the
//! change-of-variables formula in [`TransformedDistribution`].
//!
//! This mirrors PyTorch's `torch.distributions.transforms` module.
//!
//! CL-330

use ferrotorch_core::autograd::no_grad;
use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::grad_fns::activation::{sigmoid as sigmoid_op, softplus as softplus_op, tanh as tanh_op};
use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, neg, sub};
use ferrotorch_core::grad_fns::transcendental::{exp as exp_op, log as log_op};
use ferrotorch_core::tensor::Tensor;

// ---------------------------------------------------------------------------
// Transform trait
// ---------------------------------------------------------------------------

/// A differentiable, invertible transformation with computable log-det-Jacobian.
///
/// Transforms are used by [`TransformedDistribution`] to map samples from a
/// base distribution through a bijection, accumulating the
/// log-absolute-determinant of the Jacobian for correct density computation.
///
/// # Required methods
///
/// - [`forward`](Transform::forward) — compute `y = f(x)`
/// - [`inverse`](Transform::inverse) — compute `x = f^{-1}(y)`
/// - [`log_abs_det_jacobian`](Transform::log_abs_det_jacobian) — compute
///   `log |det df/dx|` given `(x, y)` where `y = f(x)`
pub trait Transform<T: Float>: Send + Sync {
    /// Apply the forward transformation: `y = f(x)`.
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Apply the inverse transformation: `x = f^{-1}(y)`.
    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Compute the log absolute determinant of the Jacobian.
    ///
    /// Given `(x, y)` where `y = f(x)`, returns `log |det df/dx|`.
    /// For element-wise transforms this is a tensor with the same shape as `x`.
    fn log_abs_det_jacobian(&self, x: &Tensor<T>, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Human-readable name.
    fn name(&self) -> &'static str;

    /// Closed-form `E_X[log|det J_f(X)|]` when it is independent of `X`.
    ///
    /// Used by [`TransformedDistribution::entropy`] for the standard
    /// change-of-variables entropy identity
    /// `H(Y) = H(X) + E_X[log|det J_f(X)|]`. Returning `Some(c)` advertises
    /// that the contribution is the constant `c` regardless of `X` — e.g.
    /// affine transforms whose Jacobian is `log|scale|`. Returning `None`
    /// (the default) means the contribution depends on `X` (or on the
    /// dispatcher applying a special case such as [`ExpTransform`] paired
    /// with the base mean).
    fn constant_entropy_contribution(&self) -> Option<T> {
        None
    }

    /// Whether this transform is an [`ExpTransform`].
    ///
    /// `ExpTransform` is the only `x`-dependent transform with a closed-form
    /// entropy contribution: `log|det dy/dx| = x` so the contribution is
    /// `E_X[X] = base.mean()`. The dispatcher in
    /// [`TransformedDistribution::entropy`] uses this flag to apply that
    /// special case instead of returning the generic "intractable" error.
    fn is_exp_transform(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// ExpTransform: y = exp(x)
// ---------------------------------------------------------------------------

/// Transform via `y = exp(x)`.
///
/// Maps the real line to the positive reals. The log-det-Jacobian is simply `x`
/// since `d(exp(x))/dx = exp(x)` and `log|exp(x)| = x`.
#[derive(Debug, Clone, Copy)]
pub struct ExpTransform;

impl<T: Float> Transform<T> for ExpTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident: dispatches to GPU exp_inner when x is_cuda.
        no_grad(|| exp_op(x))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident: dispatches to GPU log_inner when y is_cuda.
        no_grad(|| log_op(y))
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log|d(exp(x))/dx| = log(exp(x)) = x. We must return a tensor with
        // the same shape and device as `x`, but as a fresh leaf (no grad) per
        // the prior contract; cloning preserves storage Arc and device.
        no_grad(|| Ok(x.clone()))
    }

    fn name(&self) -> &'static str {
        "ExpTransform"
    }

    fn is_exp_transform(&self) -> bool {
        true
    }
}

// ---------------------------------------------------------------------------
// AffineTransform: y = loc + scale * x
// ---------------------------------------------------------------------------

/// Pointwise affine transform: `y = loc + scale * x`.
///
/// The log-det-Jacobian is `log|scale|` broadcast to the input shape.
#[derive(Debug, Clone)]
pub struct AffineTransform<T: Float> {
    /// Location (shift) parameter.
    pub loc: T,
    /// Scale (multiplication) parameter.
    pub scale: T,
}

impl<T: Float> AffineTransform<T> {
    /// Create a new affine transform with the given `loc` and `scale`.
    pub fn new(loc: T, scale: T) -> Self {
        Self { loc, scale }
    }
}

impl<T: Float> AffineTransform<T> {
    /// Materialize a 0-D scalar tensor on `device` filled with `value`.
    ///
    /// All Affine ops take broadcasted scalar `loc`/`scale` tensors; building
    /// them at apply time avoids caching state and re-uploading state for
    /// every device.
    fn scalar_on(value: T, device: ferrotorch_core::device::Device) -> FerrotorchResult<Tensor<T>> {
        let s = creation::scalar(value)?;
        s.to(device)
    }
}

impl<T: Float> Transform<T> for AffineTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // y = loc + scale * x, fully on x.device().
        no_grad(|| {
            let device = x.device();
            let loc_t = Self::scalar_on(self.loc, device)?;
            let scale_t = Self::scalar_on(self.scale, device)?;
            let scaled = mul(x, &scale_t)?;
            add(&loc_t, &scaled)
        })
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // x = (y - loc) / scale, fully on y.device().
        no_grad(|| {
            let device = y.device();
            let loc_t = Self::scalar_on(self.loc, device)?;
            let scale_t = Self::scalar_on(self.scale, device)?;
            let centered = sub(y, &loc_t)?;
            div(&centered, &scale_t)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log|d(loc + scale*x)/dx| = log|scale|, broadcast to x.shape().
        no_grad(|| {
            let log_abs_scale = if self.scale > T::from(0.0).unwrap() {
                self.scale.ln()
            } else {
                (T::from(0.0).unwrap() - self.scale).ln()
            };
            let cpu = creation::full(x.shape(), log_abs_scale)?;
            cpu.to(x.device())
        })
    }

    fn name(&self) -> &'static str {
        "AffineTransform"
    }

    fn constant_entropy_contribution(&self) -> Option<T> {
        // log|d(loc + scale*x)/dx| = log|scale|, a scalar that does not
        // depend on x. `scale == 0` would not be a valid bijection, so we
        // treat the abs-then-ln branch as the only physical case.
        let zero = T::from(0.0).unwrap();
        let abs_scale = if self.scale >= zero {
            self.scale
        } else {
            zero - self.scale
        };
        Some(abs_scale.ln())
    }
}

// ---------------------------------------------------------------------------
// SigmoidTransform: y = 1 / (1 + exp(-x))
// ---------------------------------------------------------------------------

/// Transform via the sigmoid function: `y = sigma(x) = 1 / (1 + exp(-x))`.
///
/// Maps the real line to the unit interval `(0, 1)`.
/// The log-det-Jacobian is `-softplus(-x) - softplus(x)`.
#[derive(Debug, Clone, Copy)]
pub struct SigmoidTransform;

impl<T: Float> Transform<T> for SigmoidTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident sigmoid (already numerically stable in core).
        no_grad(|| sigmoid_op(x))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // logit(y) = log(y) - log(1 - y), clamped into (eps, 1-eps) to match
        // the prior CPU body's domain-safety contract. All ops device-resident.
        no_grad(|| {
            let one = T::from(1.0).unwrap();
            let eps = T::from(1e-7).unwrap();
            let clamped =
                ferrotorch_core::grad_fns::transcendental::clamp(y, eps, one - eps)?;
            let device = y.device();
            let one_t = creation::scalar(one)?.to(device)?;
            let one_minus = sub(&one_t, &clamped)?;
            let log_y = log_op(&clamped)?;
            let log_one_minus = log_op(&one_minus)?;
            sub(&log_y, &log_one_minus)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log|d(sigma(x))/dx| = -softplus(-x) - softplus(x). Implement via
        // device-resident neg/softplus; matches the prior scalar formula.
        no_grad(|| {
            let neg_x = neg(x)?;
            let sp_neg = softplus_op(&neg_x, 1.0, 20.0)?;
            let sp_pos = softplus_op(x, 1.0, 20.0)?;
            let neg_sp_neg = neg(&sp_neg)?;
            let neg_sp_pos = neg(&sp_pos)?;
            add(&neg_sp_neg, &neg_sp_pos)
        })
    }

    fn name(&self) -> &'static str {
        "SigmoidTransform"
    }
}

// ---------------------------------------------------------------------------
// TanhTransform: y = tanh(x)
// ---------------------------------------------------------------------------

/// Transform via `y = tanh(x)`.
///
/// Maps the real line to `(-1, 1)`. Uses the numerically stable formula
/// `log_abs_det_jacobian = 2 * (log(2) - x - softplus(-2x))`.
#[derive(Debug, Clone, Copy)]
pub struct TanhTransform;

impl<T: Float> Transform<T> for TanhTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident tanh.
        no_grad(|| tanh_op(x))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // atanh(y) = 0.5 * log((1+y)/(1-y)) — device-resident algebra.
        no_grad(|| {
            let device = y.device();
            let one = T::from(1.0).unwrap();
            let half = T::from(0.5).unwrap();
            let one_t = creation::scalar(one)?.to(device)?;
            let half_t = creation::scalar(half)?.to(device)?;
            let one_plus = add(&one_t, y)?;
            let one_minus = sub(&one_t, y)?;
            let ratio = div(&one_plus, &one_minus)?;
            let log_ratio = log_op(&ratio)?;
            mul(&half_t, &log_ratio)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Numerically stable formula (TensorFlow Probability):
        //   log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2*x))
        // All ops device-resident.
        no_grad(|| {
            let device = x.device();
            let two = T::from(2.0).unwrap();
            let ln2 = T::from(2.0f64.ln()).unwrap();
            let two_t = creation::scalar(two)?.to(device)?;
            let ln2_t = creation::full(x.shape(), ln2)?.to(device)?;
            // -2*x
            let neg_two_x = neg(&mul(&two_t, x)?)?;
            // softplus(-2x)
            let sp = softplus_op(&neg_two_x, 1.0, 20.0)?;
            // ln2 - x - softplus(-2x)
            let inner = sub(&sub(&ln2_t, x)?, &sp)?;
            // 2 * inner
            mul(&two_t, &inner)
        })
    }

    fn name(&self) -> &'static str {
        "TanhTransform"
    }
}

// ---------------------------------------------------------------------------
// SoftplusTransform: y = log(1 + exp(x))
// ---------------------------------------------------------------------------

/// Transform via `y = softplus(x) = log(1 + exp(x))`.
///
/// Maps the real line to the positive reals. Reverts to the identity for
/// large `x` (> 20) for numerical stability.
#[derive(Debug, Clone, Copy)]
pub struct SoftplusTransform;

impl<T: Float> Transform<T> for SoftplusTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Device-resident softplus(x) with beta=1, threshold=20 to match the
        // prior scalar contract.
        no_grad(|| softplus_op(x, 1.0, 20.0))
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // softplus^{-1}(y) = log(exp(y) - 1). Device-resident algebra.
        // Note: the prior CPU body short-circuited y>20 to y itself for
        // overflow safety; the device-resident form is mathematically exact
        // and well-defined in the same range exp's GPU kernel handles. For
        // f32 inputs, exp(20) ~ 4.85e8 is far from the f32 max (3.4e38), so
        // the chain remains numerically safe in the previously tested range.
        no_grad(|| {
            let device = y.device();
            let one_t = creation::scalar(T::from(1.0).unwrap())?.to(device)?;
            let exp_y = exp_op(y)?;
            let exp_minus_one = sub(&exp_y, &one_t)?;
            log_op(&exp_minus_one)
        })
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // d(softplus(x))/dx = sigmoid(x); log|sigmoid(x)| = -softplus(-x).
        no_grad(|| {
            let neg_x = neg(x)?;
            let sp = softplus_op(&neg_x, 1.0, 20.0)?;
            neg(&sp)
        })
    }

    fn name(&self) -> &'static str {
        "SoftplusTransform"
    }
}

// ---------------------------------------------------------------------------
// ComposeTransform: chain multiple transforms
// ---------------------------------------------------------------------------

/// Compose multiple transforms into a single transform.
///
/// Given transforms `[f1, f2, ..., fn]`, the composed forward pass computes
/// `fn(... f2(f1(x)) ...)` and the log-det-Jacobian is the sum of the
/// individual log-det-Jacobians along the chain.
pub struct ComposeTransform<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
}

impl<T: Float> ComposeTransform<T> {
    /// Create a composed transform from an ordered list of transforms.
    ///
    /// Transforms are applied left-to-right in the forward direction.
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> Self {
        Self { transforms }
    }

    /// The number of transforms in the chain.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Whether the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl<T: Float> Transform<T> for ComposeTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut val = x.clone();
        for t in &self.transforms {
            val = t.forward(&val)?;
        }
        Ok(val)
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut val = y.clone();
        for t in self.transforms.iter().rev() {
            val = t.inverse(&val)?;
        }
        Ok(val)
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Empty chain → identity → zero log-det-Jacobian on x.device().
        no_grad(|| {
            if self.transforms.is_empty() {
                let zeros = creation::full(x.shape(), T::from(0.0).unwrap())?;
                return zeros.to(x.device());
            }

            // Compute intermediates xs[0..=n] where xs[i+1] = transforms[i].forward(xs[i]).
            // Each child forward preserves device when its body is device-resident.
            let mut xs = Vec::with_capacity(self.transforms.len() + 1);
            xs.push(x.clone());
            for t in &self.transforms {
                let next = t.forward(xs.last().unwrap())?;
                xs.push(next);
            }

            // Sum the per-link log-det-Jacobians element-wise via device-resident add.
            let mut total = self
                .transforms
                .first()
                .unwrap()
                .log_abs_det_jacobian(&xs[0], &xs[1])?;
            for (i, t) in self.transforms.iter().enumerate().skip(1) {
                let ldj = t.log_abs_det_jacobian(&xs[i], &xs[i + 1])?;
                total = add(&total, &ldj)?;
            }
            Ok(total)
        })
    }

    fn name(&self) -> &'static str {
        "ComposeTransform"
    }

    fn constant_entropy_contribution(&self) -> Option<T> {
        // A composed chain is constant-Jacobian iff every link is. In that
        // case the contributions sum: log|det J_compose| = sum_i log|det J_i|.
        let mut acc = T::from(0.0).unwrap();
        for t in &self.transforms {
            let c = t.constant_entropy_contribution()?;
            acc += c;
        }
        Some(acc)
    }
}

// ---------------------------------------------------------------------------
// TransformedDistribution
// ---------------------------------------------------------------------------

use crate::Distribution;

/// A distribution formed by applying a sequence of transforms to a base
/// distribution.
///
/// Given a base distribution `p(x)` and a bijective transform `f` with
/// `y = f(x)`, the density of `y` is:
///
/// ```text
/// log p(y) = log p(f^{-1}(y)) + log |det df^{-1}/dy|
///          = log p(f^{-1}(y)) - log |det df/dx|_{x = f^{-1}(y)}
/// ```
///
/// This is the change-of-variables formula.
///
/// # Examples
///
/// ```ignore
/// // LogNormal = Normal pushed through exp
/// let base = Normal::new(loc, scale)?;
/// let transforms: Vec<Box<dyn Transform<f32>>> = vec![Box::new(ExpTransform)];
/// let log_normal = TransformedDistribution::new(base, transforms);
/// ```
pub struct TransformedDistribution<T: Float> {
    base: Box<dyn Distribution<T>>,
    transforms: Vec<Box<dyn Transform<T>>>,
}

impl<T: Float> TransformedDistribution<T> {
    /// Create a transformed distribution.
    ///
    /// `base` is the base distribution. `transforms` are applied left-to-right
    /// in the forward (sampling) direction.
    pub fn new(base: Box<dyn Distribution<T>>, transforms: Vec<Box<dyn Transform<T>>>) -> Self {
        Self { base, transforms }
    }
}

impl<T: Float> Distribution<T> for TransformedDistribution<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let mut x = self.base.sample(shape)?;
        for t in &self.transforms {
            x = t.forward(&x)?;
        }
        Ok(x)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let mut x = self.base.rsample(shape)?;
        for t in &self.transforms {
            x = t.forward(&x)?;
        }
        Ok(x)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Walk transforms in reverse, inverting back to the base sample and
        // accumulating sum-of-log-det-jacobians device-resident. Final result
        // is base_log_prob(inverted) - sum_log_dets.
        no_grad(|| {
            let mut y = value.clone();
            // Initialize accumulator as zeros on value.device() with value's shape.
            let mut sum_ldj: Tensor<T> =
                creation::full(value.shape(), T::from(0.0).unwrap())?.to(value.device())?;

            for t in self.transforms.iter().rev() {
                let x = t.inverse(&y)?;
                let ldj = t.log_abs_det_jacobian(&x, &y)?;
                sum_ldj = add(&sum_ldj, &ldj)?;
                y = x;
            }

            let base_lp = self.base.log_prob(&y)?;
            // Result = base_lp - sum_ldj. base_lp is on value.device() if the
            // base distribution preserves device; sub will fail-fast on a
            // mismatch, which is the correct PyTorch-faithful behaviour.
            sub(&base_lp, &sum_ldj)
        })
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // Change-of-variables identity:
        //   H(Y) = H(X) + E_X[log|det J_f(X)|]
        // We accept three closed-form dispatches:
        //
        //   1. Empty transform list                       — H(Y) = H(X).
        //   2. Every transform reports
        //      `constant_entropy_contribution()`           — contributions sum
        //      to a scalar `c` independent of X, and
        //      H(Y) = H(X) + c (broadcast onto X's entropy shape).
        //   3. A single [`ExpTransform`] with a base that
        //      implements `mean()`                         — H(Y) = H(X)
        //      + E[X] = H(X) + base.mean().
        //
        // Anything else (sigmoid, tanh, softplus, multi-Exp chains, Exp
        // composed with non-trivial transforms, etc.) is not currently
        // tractable in closed form and we surface a precise, structured
        // error naming the offending transform(s) so the caller can either
        // resort to Monte-Carlo or extend this dispatch.
        let base_entropy = self.base.entropy()?;
        if self.transforms.is_empty() {
            return Ok(base_entropy);
        }

        // Path 2 — all-constant Jacobian contributions sum to a scalar.
        let all_constant: Option<T> = {
            let mut acc = T::from(0.0).unwrap();
            let mut ok = true;
            for t in &self.transforms {
                match t.constant_entropy_contribution() {
                    Some(c) => acc += c,
                    None => {
                        ok = false;
                        break;
                    }
                }
            }
            if ok { Some(acc) } else { None }
        };
        if let Some(c) = all_constant {
            // Broadcast the scalar onto base_entropy's shape and add.
            let device = base_entropy.device();
            let c_tensor = creation::full(base_entropy.shape(), c)?.to(device)?;
            return add(&base_entropy, &c_tensor);
        }

        // Path 3 — a single Exp transform: contribution is E[X] = base.mean().
        if self.transforms.len() == 1 && self.transforms[0].is_exp_transform() {
            let mean = self.base.mean()?;
            // base.mean() is shape-compatible with base.entropy() for the
            // distributions we support (Normal et al. parameterised by
            // loc/scale). If the base distribution does not implement mean
            // it surfaces its own InvalidArgument here, which we propagate.
            //
            // mean may live on a different device than entropy (e.g. Normal
            // returns `self.loc.clone()` while entropy materialises on
            // scale.device()). Move mean onto entropy's device before adding.
            let device = base_entropy.device();
            let mean_on_device = if mean.device() == device {
                mean
            } else {
                mean.to(device)?
            };
            return add(&base_entropy, &mean_on_device);
        }

        // Fall-through: enumerate the problematic transforms by name so the
        // caller knows which link blocks the closed-form path.
        let problematic: Vec<&'static str> = self
            .transforms
            .iter()
            .filter(|t| t.constant_entropy_contribution().is_none() && !t.is_exp_transform())
            .map(|t| t.name())
            .collect();
        let summary = if problematic.is_empty() {
            // The list contains an Exp transform but is not exactly
            // [Exp] — e.g. [Exp, Exp] or [Exp, Affine]. Multi-Exp /
            // mixed-Exp chains require E[exp(...)] which is not constant
            // in closed form for arbitrary bases.
            "TransformedDistribution::entropy: chain contains ExpTransform but is not a \
             single Exp — the contribution would require evaluating E_X[exp(...)] which \
             has no closed form for the general base"
                .to_string()
        } else {
            format!(
                "TransformedDistribution::entropy: closed-form contribution is \
                 intractable for transform(s) {problematic:?}. Supported transforms \
                 are AffineTransform (and compositions thereof) plus a single \
                 ExpTransform; everything else (SigmoidTransform, TanhTransform, \
                 SoftplusTransform, ...) has no general analytic form. \
                 Use Monte-Carlo estimation instead.",
            )
        };
        Err(ferrotorch_core::error::FerrotorchError::InvalidArgument { message: summary })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    // -- ExpTransform --------------------------------------------------------

    #[test]
    fn test_exp_forward() {
        let x = from_slice(&[0.0f32, 1.0, -1.0], &[3]).unwrap();
        let t = ExpTransform;
        let y = t.forward(&x).unwrap();
        let data = y.data().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 1.0f32.exp()).abs() < 1e-5);
        assert!((data[2] - (-1.0f32).exp()).abs() < 1e-5);
    }

    #[test]
    fn test_exp_inverse() {
        let y = from_slice(
            &[1.0f32, std::f32::consts::E, 1.0 / std::f32::consts::E],
            &[3],
        )
        .unwrap();
        let t = ExpTransform;
        let x = t.inverse(&y).unwrap();
        let data = x.data().unwrap();
        assert!(data[0].abs() < 1e-5); // ln(1) = 0
        assert!((data[1] - 1.0).abs() < 1e-5); // ln(e) = 1
    }

    #[test]
    fn test_exp_roundtrip() {
        let x = from_slice(&[-2.0f32, 0.0, 3.0], &[3]).unwrap();
        let t = ExpTransform;
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-5, "roundtrip failed: {a} vs {b}");
        }
    }

    #[test]
    fn test_exp_log_det_jacobian() {
        // For exp transform, log|det J| = x
        let x = from_slice(&[-1.0f32, 0.0, 2.0], &[3]).unwrap();
        let y = ExpTransform.forward(&x).unwrap();
        let ldj = ExpTransform.log_abs_det_jacobian(&x, &y).unwrap();
        let ldj_data = ldj.data().unwrap();
        let x_data = x.data().unwrap();
        for (ld, xv) in ldj_data.iter().zip(x_data.iter()) {
            assert!((ld - xv).abs() < 1e-6);
        }
    }

    // -- AffineTransform -----------------------------------------------------

    #[test]
    fn test_affine_forward() {
        let x = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let t = AffineTransform::new(10.0f32, 2.0);
        let y = t.forward(&x).unwrap();
        let data = y.data().unwrap();
        assert!((data[0] - 12.0).abs() < 1e-6);
        assert!((data[1] - 14.0).abs() < 1e-6);
        assert!((data[2] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_affine_inverse() {
        let y = from_slice(&[12.0f32, 14.0, 16.0], &[3]).unwrap();
        let t = AffineTransform::new(10.0f32, 2.0);
        let x = t.inverse(&y).unwrap();
        let data = x.data().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_affine_roundtrip() {
        let x = from_slice(&[-5.0f32, 0.0, 7.0], &[3]).unwrap();
        let t = AffineTransform::new(3.0f32, -0.5);
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_affine_log_det_jacobian() {
        let x = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let t = AffineTransform::new(0.0f32, 3.0);
        let y = t.forward(&x).unwrap();
        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        let data = ldj.data().unwrap();
        let expected = 3.0f32.ln();
        for &v in data {
            assert!((v - expected).abs() < 1e-6);
        }
    }

    #[test]
    fn test_affine_negative_scale_log_det() {
        let x = from_slice(&[1.0f32], &[1]).unwrap();
        let t = AffineTransform::new(0.0f32, -2.0);
        let y = t.forward(&x).unwrap();
        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        let expected = 2.0f32.ln(); // log|scale|
        assert!((ldj.item().unwrap() - expected).abs() < 1e-6);
    }

    // -- SigmoidTransform ----------------------------------------------------

    #[test]
    fn test_sigmoid_forward() {
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = SigmoidTransform.forward(&x).unwrap();
        assert!((y.item().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_sigmoid_roundtrip() {
        let x = from_slice(&[-3.0f32, 0.0, 3.0], &[3]).unwrap();
        let t = SigmoidTransform;
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-4, "sigmoid roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_sigmoid_log_det_jacobian() {
        // At x=0: sigmoid(0) = 0.5, sigmoid'(0) = 0.25
        // log|0.25| = log(0.25) ~ -1.3863
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = SigmoidTransform.forward(&x).unwrap();
        let ldj = SigmoidTransform.log_abs_det_jacobian(&x, &y).unwrap();
        let expected = 0.25f32.ln();
        assert!(
            (ldj.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            ldj.item().unwrap()
        );
    }

    // -- TanhTransform -------------------------------------------------------

    #[test]
    fn test_tanh_forward() {
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = TanhTransform.forward(&x).unwrap();
        assert!(y.item().unwrap().abs() < 1e-6);
    }

    #[test]
    fn test_tanh_roundtrip() {
        let x = from_slice(&[-2.0f32, 0.0, 2.0], &[3]).unwrap();
        let t = TanhTransform;
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-4, "tanh roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_tanh_log_det_jacobian() {
        // At x=0: tanh(0) = 0, tanh'(0) = 1 - 0^2 = 1
        // log|1| = 0
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = TanhTransform.forward(&x).unwrap();
        let ldj = TanhTransform.log_abs_det_jacobian(&x, &y).unwrap();
        assert!(
            ldj.item().unwrap().abs() < 1e-5,
            "expected ~0, got {}",
            ldj.item().unwrap()
        );
    }

    // -- SoftplusTransform ---------------------------------------------------

    #[test]
    fn test_softplus_forward() {
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = SoftplusTransform.forward(&x).unwrap();
        // softplus(0) = ln(2)
        assert!(
            (y.item().unwrap() - 2.0f32.ln()).abs() < 1e-6,
            "expected ln(2), got {}",
            y.item().unwrap()
        );
    }

    #[test]
    fn test_softplus_roundtrip() {
        let x = from_slice(&[-2.0f32, 0.0, 5.0], &[3]).unwrap();
        let t = SoftplusTransform;
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-4, "softplus roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_softplus_log_det_jacobian() {
        // softplus'(x) = sigmoid(x), at x=0: sigmoid(0)=0.5
        // log|0.5| = -ln(2)
        let x = from_slice(&[0.0f32], &[1]).unwrap();
        let y = SoftplusTransform.forward(&x).unwrap();
        let ldj = SoftplusTransform.log_abs_det_jacobian(&x, &y).unwrap();
        let expected = -(2.0f32.ln());
        assert!(
            (ldj.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            ldj.item().unwrap()
        );
    }

    // -- ComposeTransform ----------------------------------------------------

    #[test]
    fn test_compose_empty_is_identity() {
        let x = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![]);
        let y = t.forward(&x).unwrap();
        let orig = x.data().unwrap();
        let fwd = y.data().unwrap();
        for (a, b) in orig.iter().zip(fwd.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_compose_exp_then_affine() {
        // y = 2 * exp(x) + 1
        let x = from_slice(&[0.0f32, 1.0], &[2]).unwrap();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![
            Box::new(ExpTransform),
            Box::new(AffineTransform::new(1.0, 2.0)),
        ]);
        let y = t.forward(&x).unwrap();
        let data = y.data().unwrap();
        // exp(0)=1, 2*1+1=3
        assert!((data[0] - 3.0).abs() < 1e-5);
        // exp(1)~2.718, 2*2.718+1~6.436
        assert!((data[1] - (2.0 * 1.0f32.exp() + 1.0)).abs() < 1e-4);
    }

    #[test]
    fn test_compose_roundtrip() {
        let x = from_slice(&[0.5f32, 1.5], &[2]).unwrap();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![
            Box::new(AffineTransform::new(0.0, 2.0)),
            Box::new(ExpTransform),
        ]);
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-4, "compose roundtrip: {a} vs {b}");
        }
    }

    #[test]
    fn test_compose_log_det_jacobian() {
        // Compose: affine(0, 2) then exp => y = exp(2x)
        // dy/dx = 2 * exp(2x), log|dy/dx| = ln(2) + 2x
        let x = from_slice(&[0.0f32, 1.0], &[2]).unwrap();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![
            Box::new(AffineTransform::new(0.0, 2.0)),
            Box::new(ExpTransform),
        ]);
        let y = t.forward(&x).unwrap();
        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        let data = ldj.data().unwrap();
        // At x=0: ln(2) + 0 = ln(2)
        assert!(
            (data[0] - 2.0f32.ln()).abs() < 1e-5,
            "expected ln(2), got {}",
            data[0]
        );
        // At x=1: ln(2) + 2
        assert!(
            (data[1] - (2.0f32.ln() + 2.0)).abs() < 1e-5,
            "expected {}, got {}",
            2.0f32.ln() + 2.0,
            data[1]
        );
    }

    // -- TransformedDistribution ---------------------------------------------

    #[test]
    fn test_transformed_distribution_sample_shape() {
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);
        let samples = td.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        // All samples should be positive (exp maps R -> R+)
        let data = samples.data().unwrap();
        for &v in data {
            assert!(v > 0.0, "expected positive, got {v}");
        }
    }

    #[test]
    fn test_transformed_distribution_log_prob() {
        // LogNormal: base = Normal(0, 1), transform = exp
        // log_prob(y) = log_prob_normal(ln(y)) - ln(y)
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);

        let y = scalar(1.0f32).unwrap(); // ln(1) = 0
        let lp = td.log_prob(&y).unwrap();
        // At y=1: log_prob_normal(0) - 0 = -0.5*ln(2*pi)
        let expected = -0.5 * (2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_transformed_distribution_log_prob_general() {
        // LogNormal(0,1) at y=e: log_prob_normal(1) - 1
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);

        let e = std::f32::consts::E;
        let y = scalar(e).unwrap();
        let lp = td.log_prob(&y).unwrap();
        // log_prob_normal(1) = -0.5*(1)^2 - 0.5*ln(2*pi)
        // log_prob_lognormal(e) = log_prob_normal(1) - ln(e) = log_prob_normal(1) - 1
        let log_prob_normal_1 = -0.5 - 0.5 * (2.0f32 * std::f32::consts::PI).ln();
        let expected = log_prob_normal_1 - 1.0;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_empty_chain_matches_base() {
        // Empty chain → identity; entropy must equal the base distribution's
        // entropy exactly.
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc.clone(), scale.clone()).unwrap();
        let td: TransformedDistribution<f32> =
            TransformedDistribution::new(Box::new(base), vec![]);
        let base2 = Normal::new(loc, scale).unwrap();
        let ent = td.entropy().unwrap().item().unwrap();
        let base_ent = base2.entropy().unwrap().item().unwrap();
        assert!(
            (ent - base_ent).abs() < 1e-6,
            "empty-chain entropy: td={ent} base={base_ent}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_affine() {
        // entropy(Normal(0,1) → affine(loc=2, scale=3)) =
        //   entropy(Normal(0,1)) + log|3|
        //   = entropy(Normal(2, 3))
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let affine = AffineTransform::new(2.0f32, 3.0f32);
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(affine)]);
        let ent_td = td.entropy().unwrap().item().unwrap();

        let loc2 = scalar(2.0f32).unwrap();
        let scale2 = scalar(3.0f32).unwrap();
        let direct = Normal::new(loc2, scale2).unwrap();
        let ent_direct = direct.entropy().unwrap().item().unwrap();
        assert!(
            (ent_td - ent_direct).abs() < 1e-5,
            "affine entropy: td={ent_td} direct={ent_direct}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_affine_negative_scale() {
        // Negative scale: the contribution is log|scale|, which equals
        // log|scale| of the absolute value.
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let affine = AffineTransform::new(0.0f32, -2.5f32);
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(affine)]);
        let ent_td = td.entropy().unwrap().item().unwrap();

        // entropy(Normal(0,1)) = 0.5 + 0.5*ln(2*pi); + ln(2.5).
        let half = 0.5f32;
        let expected = half + half * (2.0f32 * std::f32::consts::PI).ln() + 2.5f32.ln();
        assert!(
            (ent_td - expected).abs() < 1e-5,
            "affine-neg entropy: td={ent_td} expected={expected}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_exp_matches_lognormal() {
        // entropy(Normal(loc, scale) → exp) = entropy(LogNormal(loc, scale)) =
        //   loc + 0.5 + ln(scale) + 0.5*ln(2*pi)
        // (identical to the Normal entropy + base.mean() = entropy + loc).
        use crate::{LogNormal, Normal};
        let loc_v = 1.3f32;
        let scale_v = 0.7f32;
        let loc = scalar(loc_v).unwrap();
        let scale = scalar(scale_v).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);
        let ent_td = td.entropy().unwrap().item().unwrap();

        let loc2 = scalar(loc_v).unwrap();
        let scale2 = scalar(scale_v).unwrap();
        let direct = LogNormal::new(loc2, scale2).unwrap();
        let ent_direct = direct.entropy().unwrap().item().unwrap();
        assert!(
            (ent_td - ent_direct).abs() < 1e-5,
            "exp entropy: td={ent_td} lognormal={ent_direct}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_compose_affine_chain() {
        // Affine(0, 2) ∘ Affine(1, 3) is constant-Jacobian: log(2)+log(3).
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(
            Box::new(base),
            vec![
                Box::new(AffineTransform::new(0.0f32, 2.0)),
                Box::new(AffineTransform::new(1.0f32, 3.0)),
            ],
        );
        let ent_td = td.entropy().unwrap().item().unwrap();
        let half = 0.5f32;
        let expected = half
            + half * (2.0f32 * std::f32::consts::PI).ln()
            + 2.0f32.ln()
            + 3.0f32.ln();
        assert!(
            (ent_td - expected).abs() < 1e-5,
            "affine-chain entropy: td={ent_td} expected={expected}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_sigmoid_errors() {
        // SigmoidTransform has no closed-form analytic contribution; the
        // dispatcher must return an InvalidArgument naming Sigmoid.
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(SigmoidTransform)]);
        let err = td.entropy().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("SigmoidTransform"),
            "expected SigmoidTransform in error, got: {msg}",
        );
    }

    #[test]
    fn test_transformed_distribution_entropy_exp_then_affine_errors() {
        // A chain with Exp AND another non-Affine link is *not* in our
        // currently supported set (mixed chains with Exp require the chain
        // to be exactly [Exp]). The dispatcher should surface a precise
        // error referencing Exp's intractability in mixed chains.
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(
            Box::new(base),
            vec![Box::new(ExpTransform), Box::new(AffineTransform::new(0.0f32, 1.0))],
        );
        let err = td.entropy().unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("Exp"),
            "expected Exp mention in mixed-Exp error, got: {msg}",
        );
    }

    // -- f64 tests -----------------------------------------------------------

    #[test]
    fn test_transforms_f64() {
        let x = from_slice(&[0.0f64, 1.0, -1.0], &[3]).unwrap();

        // Exp
        let y = ExpTransform.forward(&x).unwrap();
        let x2 = ExpTransform.inverse(&y).unwrap();
        let orig = x.data().unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        // Affine
        let t = AffineTransform::new(1.0f64, 3.0);
        let y = t.forward(&x).unwrap();
        let x2 = t.inverse(&y).unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-12);
        }

        // Sigmoid
        let y = SigmoidTransform.forward(&x).unwrap();
        let x2 = SigmoidTransform.inverse(&y).unwrap();
        let recov = x2.data().unwrap();
        for (a, b) in orig.iter().zip(recov.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // -- Pass 5.B.1 discriminating tests (#1103) -----------------------------
    //
    // These tests exercise non-degenerate, multi-element shapes that go
    // through the device-resident migration. On Linux/CPU, `result.device()
    // == input.device()` is a tautology — the discriminating signal is
    // *numerical correctness*: each transform must compute exactly the same
    // values it did before the migration. A regression in any of the
    // device-resident op chains (drop a softplus, swap an add for a sub, etc.)
    // surfaces here as a concrete numerical drift, which is what the
    // sabotage probe in the report exercises.

    #[test]
    fn exp_transform_preserves_device_and_value() {
        // Shape [2, 3]; check exp/inverse-log/log_det numerically and verify
        // device preservation through each leg.
        let x = from_slice(
            &[-1.0f32, 0.0, 1.0, 2.0, -2.0, 0.5],
            &[2, 3],
        )
        .unwrap();
        let t = ExpTransform;
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device, "ExpTransform::forward changed device");
        assert_eq!(y.shape(), &[2, 3]);
        let y_data = y.data().unwrap();
        let x_data = x.data().unwrap();
        for (yv, xv) in y_data.iter().zip(x_data.iter()) {
            assert!(
                (yv - xv.exp()).abs() < 1e-5,
                "exp forward: got {yv} expected {} for x={xv}",
                xv.exp()
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device, "ExpTransform::inverse changed device");
        let xr_data = xr.data().unwrap();
        for (xv0, xrv) in x_data.iter().zip(xr_data.iter()) {
            assert!(
                (xv0 - xrv).abs() < 1e-5,
                "exp roundtrip: got {xrv} expected {xv0}",
            );
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device, "ExpTransform::ldj changed device");
        let ldj_data = ldj.data().unwrap();
        for (ld, xv) in ldj_data.iter().zip(x_data.iter()) {
            assert!((ld - xv).abs() < 1e-6, "exp ldj: got {ld} expected {xv}");
        }
    }

    #[test]
    fn affine_transform_preserves_device_and_value() {
        // y = 2.5 + (-1.5) * x; non-trivial, negative-scale path exercises
        // the abs-value branch in log_abs_det_jacobian.
        let x = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let t = AffineTransform::new(2.5f32, -1.5f32);
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        let y_data = y.data().unwrap();
        let x_data = x.data().unwrap();
        for (yv, xv) in y_data.iter().zip(x_data.iter()) {
            let expected = 2.5f32 + (-1.5f32) * xv;
            assert!(
                (yv - expected).abs() < 1e-5,
                "affine forward: got {yv} expected {expected}",
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x_data.iter().zip(xr.data().unwrap().iter()) {
            assert!((xv0 - xrv).abs() < 1e-5, "affine roundtrip: {xrv} vs {xv0}");
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        let expected_ldj = 1.5f32.ln(); // log|-1.5| = ln(1.5)
        for v in ldj.data().unwrap().iter() {
            assert!(
                (v - expected_ldj).abs() < 1e-5,
                "affine ldj: got {v} expected {expected_ldj}",
            );
        }
    }

    #[test]
    fn sigmoid_transform_preserves_device_and_value() {
        let x = from_slice(&[-2.0f32, -0.5, 0.0, 0.5, 2.0, 3.0], &[2, 3]).unwrap();
        let t = SigmoidTransform;
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        // sigmoid(0) = 0.5
        let y_data = y.data().unwrap();
        let x_data = x.data().unwrap();
        for (yv, xv) in y_data.iter().zip(x_data.iter()) {
            // Reference: 1 / (1 + exp(-x))
            let expected = 1.0f32 / (1.0 + (-xv).exp());
            assert!(
                (yv - expected).abs() < 1e-5,
                "sigmoid forward: got {yv} expected {expected} for x={xv}",
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x_data.iter().zip(xr.data().unwrap().iter()) {
            assert!(
                (xv0 - xrv).abs() < 1e-4,
                "sigmoid roundtrip: {xrv} vs {xv0}",
            );
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        // log|sigma'(x)| = -softplus(-x) - softplus(x). At x=0 → -2*ln(2) + ln(2) ... actually
        //   sigma'(0) = 0.25, log(0.25) = -2*ln(2) = -ln(4)
        let ldj_data = ldj.data().unwrap();
        let expected_at_zero = 0.25f32.ln();
        // index of x=0 in flat shape [2,3] -> position 2
        assert!(
            (ldj_data[2] - expected_at_zero).abs() < 1e-5,
            "sigmoid ldj at x=0: got {} expected {expected_at_zero}",
            ldj_data[2],
        );
    }

    #[test]
    fn tanh_transform_preserves_device_and_value() {
        let x = from_slice(&[-1.5f32, -0.5, 0.0, 0.5, 1.5, 2.0], &[2, 3]).unwrap();
        let t = TanhTransform;
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        for (yv, xv) in y.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            assert!(
                (yv - xv.tanh()).abs() < 1e-5,
                "tanh forward: got {yv} expected {} for x={xv}",
                xv.tanh()
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x.data().unwrap().iter().zip(xr.data().unwrap().iter()) {
            assert!((xv0 - xrv).abs() < 1e-4, "tanh roundtrip: {xrv} vs {xv0}");
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        // log(1 - tanh(x)^2) at x=0 is log(1) = 0; at x=0.5, tanh(0.5)^2≈0.21, log(0.79)≈-0.236
        let ldj_data = ldj.data().unwrap();
        // index of x=0 in flat -> position 2
        assert!(
            ldj_data[2].abs() < 1e-5,
            "tanh ldj at x=0: got {}",
            ldj_data[2],
        );
        // index of x=0.5 -> position 3
        let expected = (1.0f32 - 0.5f32.tanh().powi(2)).ln();
        assert!(
            (ldj_data[3] - expected).abs() < 1e-4,
            "tanh ldj at x=0.5: got {} expected {expected}",
            ldj_data[3],
        );
    }

    #[test]
    fn softplus_transform_preserves_device_and_value() {
        let x = from_slice(&[-2.0f32, -0.5, 0.0, 1.0, 2.5, 4.0], &[2, 3]).unwrap();
        let t = SoftplusTransform;
        let device = x.device();

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        for (yv, xv) in y.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            // softplus(x) = log(1 + exp(x)); for x>20 it's x, but our test
            // points are all ≤ 4 so the elementary form is exact within tol.
            let expected = (1.0f32 + xv.exp()).ln();
            assert!(
                (yv - expected).abs() < 1e-5,
                "softplus forward: got {yv} expected {expected} for x={xv}",
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x.data().unwrap().iter().zip(xr.data().unwrap().iter()) {
            assert!(
                (xv0 - xrv).abs() < 1e-3,
                "softplus roundtrip: {xrv} vs {xv0}",
            );
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        // log|sigmoid(x)| at x=0 is log(0.5) = -ln(2)
        let ldj_data = ldj.data().unwrap();
        // x=0 is at index 2 in the [2,3] flat layout.
        let expected = -(2.0f32.ln());
        assert!(
            (ldj_data[2] - expected).abs() < 1e-5,
            "softplus ldj at x=0: got {} expected {expected}",
            ldj_data[2],
        );
    }

    #[test]
    fn compose_transform_chain_preserves_device() {
        // Chain: Affine(loc=1, scale=2) then Exp; check forward/inverse/ldj
        // numerical correctness AND device preservation end-to-end.
        let x = from_slice(&[-1.0f32, 0.0, 1.0], &[3]).unwrap();
        let device = x.device();
        let t: ComposeTransform<f32> = ComposeTransform::new(vec![
            Box::new(AffineTransform::new(1.0f32, 2.0f32)),
            Box::new(ExpTransform),
        ]);

        let y = t.forward(&x).unwrap();
        assert_eq!(y.device(), device);
        // y = exp(1 + 2*x)
        for (yv, xv) in y.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            let expected = (1.0f32 + 2.0 * xv).exp();
            assert!(
                (yv - expected).abs() < 1e-4,
                "compose forward: got {yv} expected {expected}",
            );
        }

        let xr = t.inverse(&y).unwrap();
        assert_eq!(xr.device(), device);
        for (xv0, xrv) in x.data().unwrap().iter().zip(xr.data().unwrap().iter()) {
            assert!(
                (xv0 - xrv).abs() < 1e-4,
                "compose roundtrip: {xrv} vs {xv0}",
            );
        }

        let ldj = t.log_abs_det_jacobian(&x, &y).unwrap();
        assert_eq!(ldj.device(), device);
        // ldj = ln(2) + (1 + 2*x): affine contributes ln|2|, exp contributes
        // its argument (which is 1 + 2*x).
        for (lv, xv) in ldj.data().unwrap().iter().zip(x.data().unwrap().iter()) {
            let expected = 2.0f32.ln() + (1.0 + 2.0 * xv);
            assert!(
                (lv - expected).abs() < 1e-4,
                "compose ldj: got {lv} expected {expected}",
            );
        }
    }

    #[test]
    fn transformed_distribution_log_prob_preserves_device() {
        // LogNormal(0,1) at value=e: log_prob_normal(1) - 1.
        // Verifies the device-resident log_prob path numerically and asserts
        // device preservation.
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);

        // Use a multi-element input to exercise broadcasting/sum_ldj paths.
        let value = from_slice(
            &[1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)],
            &[3],
        )
        .unwrap();
        let device = value.device();

        let lp = td.log_prob(&value).unwrap();
        assert_eq!(lp.device(), device, "log_prob changed device");
        assert_eq!(lp.shape(), &[3]);

        // Reference: log_prob_lognormal(y; mu=0, sigma=1) =
        //   -0.5*ln(2*pi) - 0.5*(ln y)^2 - ln y.
        let two_pi_ln = (2.0f32 * std::f32::consts::PI).ln();
        let lp_data = lp.data().unwrap();
        for (lv, yv) in lp_data
            .iter()
            .zip([1.0f32, std::f32::consts::E, std::f32::consts::E.powi(2)].iter())
        {
            let ln_y = yv.ln();
            let expected = -0.5 * two_pi_ln - 0.5 * ln_y * ln_y - ln_y;
            assert!(
                (lv - expected).abs() < 1e-4,
                "td log_prob at y={yv}: got {lv} expected {expected}",
            );
        }
    }
}
