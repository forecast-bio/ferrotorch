//! Bijective transforms for distribution reparameterization.
//!
//! Transforms map between spaces (e.g., real line to positive reals) and
//! compute the log-absolute-determinant of the Jacobian needed for the
//! change-of-variables formula in [`TransformedDistribution`].
//!
//! This mirrors PyTorch's `torch.distributions.transforms` module.
//!
//! CL-330

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::storage::TensorStorage;
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
        crate::fallback::check_gpu_fallback_opt_in(&[x], "ExpTransform::forward")?;
        let data = x.data_vec()?;
        let result: Vec<T> = data.iter().map(|&v| v.exp()).collect();
        Tensor::from_storage(TensorStorage::cpu(result), x.shape().to_vec(), false)
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[y], "ExpTransform::inverse")?;
        let data = y.data_vec()?;
        let result: Vec<T> = data.iter().map(|&v| v.ln()).collect();
        Tensor::from_storage(TensorStorage::cpu(result), y.shape().to_vec(), false)
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[x], "ExpTransform::log_abs_det_jacobian")?;
        // log|d(exp(x))/dx| = log(exp(x)) = x
        let data = x.data_vec()?;
        Tensor::from_storage(TensorStorage::cpu(data), x.shape().to_vec(), false)
    }

    fn name(&self) -> &'static str {
        "ExpTransform"
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

impl<T: Float> Transform<T> for AffineTransform<T> {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[x], "AffineTransform::forward")?;
        let data = x.data_vec()?;
        let result: Vec<T> = data.iter().map(|&v| self.loc + self.scale * v).collect();
        Tensor::from_storage(TensorStorage::cpu(result), x.shape().to_vec(), false)
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[y], "AffineTransform::inverse")?;
        let data = y.data_vec()?;
        let result: Vec<T> = data.iter().map(|&v| (v - self.loc) / self.scale).collect();
        Tensor::from_storage(TensorStorage::cpu(result), y.shape().to_vec(), false)
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[x], "AffineTransform::log_abs_det_jacobian")?;
        // log|d(loc + scale*x)/dx| = log|scale| (broadcast to input shape)
        let log_abs_scale = if self.scale > T::from(0.0).unwrap() {
            self.scale.ln()
        } else {
            (T::from(0.0).unwrap() - self.scale).ln()
        };
        let numel = x.data_vec()?.len();
        let result = vec![log_abs_scale; numel];
        Tensor::from_storage(TensorStorage::cpu(result), x.shape().to_vec(), false)
    }

    fn name(&self) -> &'static str {
        "AffineTransform"
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

/// Numerically stable softplus: `log(1 + exp(x))`.
fn softplus<T: Float>(x: T) -> T {
    let threshold = T::from(20.0).unwrap();
    if x > threshold {
        x
    } else {
        (T::from(1.0).unwrap() + x.exp()).ln()
    }
}

/// Numerically stable sigmoid.
fn sigmoid<T: Float>(x: T) -> T {
    let one = T::from(1.0).unwrap();
    let zero = T::from(0.0).unwrap();
    if x >= zero {
        let ez = (-x).exp();
        one / (one + ez)
    } else {
        let ez = x.exp();
        ez / (one + ez)
    }
}

impl<T: Float> Transform<T> for SigmoidTransform {
    fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[x], "SigmoidTransform::forward")?;
        let data = x.data_vec()?;
        let result: Vec<T> = data.iter().map(|&v| sigmoid(v)).collect();
        Tensor::from_storage(TensorStorage::cpu(result), x.shape().to_vec(), false)
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[y], "SigmoidTransform::inverse")?;
        // logit(y) = log(y) - log(1 - y)
        let data = y.data_vec()?;
        let eps = T::from(1e-7).unwrap();
        let one = T::from(1.0).unwrap();
        let result: Vec<T> = data
            .iter()
            .map(|&v| {
                let clamped = v.max(eps).min(one - eps);
                clamped.ln() - (one - clamped).ln()
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), y.shape().to_vec(), false)
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[x], "SigmoidTransform::log_abs_det_jacobian")?;
        // log|d(sigma(x))/dx| = -softplus(-x) - softplus(x)
        let data = x.data_vec()?;
        let result: Vec<T> = data
            .iter()
            .map(|&v| {
                let zero = T::from(0.0).unwrap();
                -softplus(zero - v) - softplus(v)
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), x.shape().to_vec(), false)
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
        crate::fallback::check_gpu_fallback_opt_in(&[x], "TanhTransform::forward")?;
        let data = x.data_vec()?;
        let result: Vec<T> = data.iter().map(|&v| v.tanh()).collect();
        Tensor::from_storage(TensorStorage::cpu(result), x.shape().to_vec(), false)
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[y], "TanhTransform::inverse")?;
        // atanh(y) = 0.5 * log((1+y)/(1-y))
        let data = y.data_vec()?;
        let half = T::from(0.5).unwrap();
        let one = T::from(1.0).unwrap();
        let result: Vec<T> = data
            .iter()
            .map(|&v| half * ((one + v) / (one - v)).ln())
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), y.shape().to_vec(), false)
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[x], "TanhTransform::log_abs_det_jacobian")?;
        // Numerically stable formula from TensorFlow Probability:
        // log(1 - tanh(x)^2) = 2 * (log(2) - x - softplus(-2*x))
        let data = x.data_vec()?;
        let two = T::from(2.0).unwrap();
        let ln2 = T::from(2.0f64.ln()).unwrap();
        let result: Vec<T> = data
            .iter()
            .map(|&v| two * (ln2 - v - softplus(T::from(0.0).unwrap() - two * v)))
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), x.shape().to_vec(), false)
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
        crate::fallback::check_gpu_fallback_opt_in(&[x], "SoftplusTransform::forward")?;
        let data = x.data_vec()?;
        let result: Vec<T> = data.iter().map(|&v| softplus(v)).collect();
        Tensor::from_storage(TensorStorage::cpu(result), x.shape().to_vec(), false)
    }

    fn inverse(&self, y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[y], "SoftplusTransform::inverse")?;
        // inverse of softplus: log(exp(y) - 1) = log(-expm1(-y)) + y
        let data = y.data_vec()?;
        let one = T::from(1.0).unwrap();
        let result: Vec<T> = data
            .iter()
            .map(|&v| {
                // For large y, inverse is approximately y itself.
                let threshold = T::from(20.0).unwrap();
                if v > threshold {
                    v
                } else {
                    (v.exp() - one).ln()
                }
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), y.shape().to_vec(), false)
    }

    fn log_abs_det_jacobian(&self, x: &Tensor<T>, _y: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[x],
            "SoftplusTransform::log_abs_det_jacobian",
        )?;
        // d(softplus(x))/dx = sigmoid(x)
        // log|sigmoid(x)| = -softplus(-x)
        let data = x.data_vec()?;
        let zero = T::from(0.0).unwrap();
        let result: Vec<T> = data.iter().map(|&v| -softplus(zero - v)).collect();
        Tensor::from_storage(TensorStorage::cpu(result), x.shape().to_vec(), false)
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
        crate::fallback::check_gpu_fallback_opt_in(&[x], "ComposeTransform::log_abs_det_jacobian")?;
        if self.transforms.is_empty() {
            // Identity: zero log-det-Jacobian.
            let data = x.data_vec()?;
            let zeros = vec![T::from(0.0).unwrap(); data.len()];
            return Tensor::from_storage(TensorStorage::cpu(zeros), x.shape().to_vec(), false);
        }

        // Compute intermediates and accumulate log-det-Jacobian terms.
        let mut xs = vec![x.clone()];
        for t in &self.transforms {
            let next = t.forward(xs.last().unwrap())?;
            xs.push(next);
        }

        // Sum the log-det-Jacobians element-wise.
        let numel = x.data_vec()?.len();
        let mut total = vec![T::from(0.0).unwrap(); numel];

        for (i, t) in self.transforms.iter().enumerate() {
            let ldj = t.log_abs_det_jacobian(&xs[i], &xs[i + 1])?;
            let ldj_data = ldj.data_vec()?;
            for (j, &v) in ldj_data.iter().enumerate() {
                if j < total.len() {
                    total[j] += v;
                }
            }
        }

        Tensor::from_storage(TensorStorage::cpu(total), x.shape().to_vec(), false)
    }

    fn name(&self) -> &'static str {
        "ComposeTransform"
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
        crate::fallback::check_gpu_fallback_opt_in(&[value], "TransformedDistribution::log_prob")?;
        // Invert transforms to get back to the base space, accumulating
        // log-det-Jacobian corrections along the way.
        let mut y = value.clone();
        let numel = value.data_vec()?.len();
        let zero = T::from(0.0).unwrap();
        let mut log_det_correction = vec![zero; numel];

        for t in self.transforms.iter().rev() {
            let x = t.inverse(&y)?;
            let ldj = t.log_abs_det_jacobian(&x, &y)?;
            let ldj_data = ldj.data_vec()?;
            for (j, &v) in ldj_data.iter().enumerate() {
                if j < log_det_correction.len() {
                    log_det_correction[j] = log_det_correction[j] - v;
                }
            }
            y = x;
        }

        // Compute base log_prob and add the correction.
        let base_lp = self.base.log_prob(&y)?;
        let base_data = base_lp.data_vec()?;

        let result: Vec<T> = base_data
            .iter()
            .zip(log_det_correction.iter().cycle())
            .map(|(&lp, &corr)| lp + corr)
            .collect();

        Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // Entropy of a transformed distribution is generally not available
        // in closed form. We could estimate it via sampling but that's not
        // what PyTorch does either — it raises NotImplementedError for the
        // general case. We return the base entropy as an approximation only
        // when there are no transforms; otherwise error.
        if self.transforms.is_empty() {
            return self.base.entropy();
        }
        Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
            message: "entropy() is not implemented for TransformedDistribution \
                      in the general case. Use sampling-based estimates instead."
                .into(),
        })
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
    fn test_transformed_distribution_entropy_errors() {
        use crate::Normal;
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let base = Normal::new(loc, scale).unwrap();
        let td = TransformedDistribution::new(Box::new(base), vec![Box::new(ExpTransform)]);
        assert!(td.entropy().is_err());
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
}
