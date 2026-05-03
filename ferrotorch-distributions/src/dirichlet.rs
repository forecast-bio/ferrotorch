//! Dirichlet distribution.
//!
//! `Dirichlet(concentration)` defines a distribution over the probability simplex.
//! Samples are K-dimensional vectors whose elements are positive and sum to 1.
//!
//! Sampling uses the Gamma-based reparameterization: draw independent
//! `Gamma(alpha_k, 1)` samples and normalize.
//!
//! [CL-331] ferrotorch#331 — multivariate distributions

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Dirichlet distribution parameterized by `concentration` (alpha).
///
/// `concentration` is a 1-D tensor of length `K` whose elements must be positive.
/// Samples lie on the `(K-1)`-dimensional probability simplex.
///
/// # Reparameterization
///
/// `rsample` uses the implicit reparameterization through Gamma samples.
/// Gradients flow through the concentration parameters.
pub struct Dirichlet<T: Float> {
    concentration: Tensor<T>,
    k: usize,
}

impl<T: Float> Dirichlet<T> {
    /// Create a new Dirichlet distribution.
    ///
    /// `concentration` must be a 1-D tensor with positive elements.
    pub fn new(concentration: Tensor<T>) -> FerrotorchResult<Self> {
        if concentration.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dirichlet: concentration must be 1-D, got shape {:?}",
                    concentration.shape()
                ),
            });
        }
        let k = concentration.shape()[0];
        if k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Dirichlet: concentration must have at least one element".into(),
            });
        }
        Ok(Self { concentration, k })
    }

    /// The concentration (alpha) parameter.
    pub fn concentration(&self) -> &Tensor<T> {
        &self.concentration
    }

    /// Number of categories (K).
    pub fn num_categories(&self) -> usize {
        self.k
    }
}

/// Sample a Gamma(alpha, 1) variable using Marsaglia & Tsang's method.
///
/// This handles alpha >= 1 directly. For alpha < 1 we use the Ahrens-Dieter
/// boost: Gamma(alpha, 1) = Gamma(alpha+1, 1) * U^(1/alpha).
fn sample_gamma<T: Float>(alpha: T) -> T {
    let one = <T as num_traits::One>::one();
    let zero = <T as num_traits::Zero>::zero();
    let third = T::from(1.0 / 3.0).unwrap();

    if alpha < one {
        // Boost: Gamma(a) = Gamma(a+1) * U^(1/a)
        let g = sample_gamma(alpha + one);
        let u = sample_uniform_01::<T>();
        return g * u.powf(one / alpha);
    }

    // Marsaglia & Tsang for alpha >= 1
    let d = alpha - third;
    let c = third / d.sqrt();

    loop {
        let x = sample_standard_normal::<T>();
        let v_base = one + c * x;
        if v_base <= zero {
            continue;
        }
        let v = v_base * v_base * v_base;
        let u = sample_uniform_01::<T>();

        let half = T::from(0.5).unwrap();
        let threshold = T::from(0.0331).unwrap();

        if u < one - threshold * x * x * x * x {
            return d * v;
        }
        if u.ln() < half * x * x + d * (one - v + v.ln()) {
            return d * v;
        }
    }
}

/// Draw U ~ Uniform(0, 1) using the same RNG approach as creation::rand.
fn sample_uniform_01<T: Float>() -> T {
    // Use the creation module's rand for a single element
    let t = creation::rand::<T>(&[1]).unwrap();
    t.data_vec().unwrap()[0]
}

/// Draw Z ~ N(0, 1) using the same RNG approach as creation::randn.
fn sample_standard_normal<T: Float>() -> T {
    let t = creation::randn::<T>(&[1]).unwrap();
    t.data_vec().unwrap()[0]
}

impl<T: Float> Distribution<T> for Dirichlet<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.concentration], "Dirichlet::sample")?;
        let device = self.concentration.device();
        let n: usize = shape.iter().product();
        let k = self.k;
        let alpha = self.concentration.data_vec()?;

        let mut result = Vec::with_capacity(n * k);
        for _ in 0..n {
            // Draw Gamma(alpha_j, 1) for each category and normalize
            let mut gammas = Vec::with_capacity(k);
            let mut total = <T as num_traits::Zero>::zero();
            for &a in &alpha {
                let g = sample_gamma(a);
                gammas.push(g);
                total += g;
            }
            for g in gammas {
                result.push(g / total);
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape.push(k);
        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.concentration], "Dirichlet::rsample")?;
        let device = self.concentration.device();
        let n: usize = shape.iter().product();
        let k = self.k;
        let alpha = self.concentration.data_vec()?;

        // Sample Gamma and normalize (same as sample, but track gradients)
        let mut gamma_vals = Vec::with_capacity(n * k);
        let mut result = Vec::with_capacity(n * k);

        for s in 0..n {
            let mut total = <T as num_traits::Zero>::zero();
            for &a in &alpha {
                let g = sample_gamma(a);
                gamma_vals.push(g);
                total += g;
            }
            for j in 0..k {
                result.push(gamma_vals[s * k + j] / total);
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape.push(k);
        let storage = TensorStorage::cpu(result.clone());

        let out = if self.concentration.requires_grad() && ferrotorch_core::is_grad_enabled() {
            let sample_tensor =
                Tensor::from_storage(TensorStorage::cpu(result), out_shape.clone(), false)?;
            let grad_fn = Arc::new(DirichletRsampleBackward {
                concentration: self.concentration.clone(),
                samples: sample_tensor,
                n,
                k,
            });
            Tensor::from_operation(storage, out_shape, grad_fn)?
        } else {
            Tensor::from_storage(storage, out_shape, false)?
        };
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration, value],
            "Dirichlet::log_prob",
        )?;
        // log_prob = sum((alpha_k - 1) * log(x_k)) + lgamma(sum(alpha)) - sum(lgamma(alpha_k))
        let device = self.concentration.device();
        let k = self.k;
        let alpha = self.concentration.data_vec()?;
        let val_data = value.data_vec()?;

        let n = val_data.len() / k;
        if val_data.len() != n * k {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Dirichlet log_prob: value has {} elements, not divisible by k={}",
                    val_data.len(),
                    k
                ),
            });
        }

        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        // Precompute: lgamma(sum(alpha)) - sum(lgamma(alpha_k))
        let alpha_sum: T = alpha.iter().copied().fold(zero, |a, b| a + b);
        let lgamma_alpha_sum = lgamma_t(alpha_sum);
        let sum_lgamma_alpha: T = alpha.iter().map(|&a| lgamma_t(a)).fold(zero, |a, b| a + b);
        let normalizer = lgamma_alpha_sum - sum_lgamma_alpha;

        let mut result = Vec::with_capacity(n);
        for s in 0..n {
            let mut log_prob = normalizer;
            for j in 0..k {
                let x_j = val_data[s * k + j];
                // xlogy: (alpha - 1) * log(x), handle x=0 when alpha=1
                let a_minus_1 = alpha[j] - one;
                if a_minus_1 != zero {
                    log_prob += a_minus_1 * x_j.max(T::from(1e-30).unwrap()).ln();
                }
            }
            result.push(log_prob);
        }

        let val_shape = value.shape();
        let out_shape = if val_shape.len() > 1 {
            val_shape[..val_shape.len() - 1].to_vec()
        } else {
            vec![]
        };

        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.concentration], "Dirichlet::entropy")?;
        // H = sum(lgamma(alpha_k)) - lgamma(sum(alpha))
        //     - (K - sum(alpha)) * digamma(sum(alpha))
        //     - sum((alpha_k - 1) * digamma(alpha_k))
        let device = self.concentration.device();
        let k = self.k;
        let alpha = self.concentration.data_vec()?;

        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();

        let alpha_sum: T = alpha.iter().copied().fold(zero, |a, b| a + b);
        let k_t = T::from(k).unwrap();

        let sum_lgamma: T = alpha.iter().map(|&a| lgamma_t(a)).fold(zero, |a, b| a + b);
        let lgamma_sum = lgamma_t(alpha_sum);
        let digamma_sum = digamma_t(alpha_sum);

        let mut sum_digamma_term = zero;
        for &a in &alpha {
            sum_digamma_term += (a - one) * digamma_t(a);
        }

        let h = sum_lgamma - lgamma_sum - (k_t - alpha_sum) * digamma_sum - sum_digamma_term;

        let out = Tensor::from_storage(TensorStorage::cpu(vec![h]), vec![], false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar special functions
// ---------------------------------------------------------------------------

/// Scalar lgamma using the Float trait's built-in method.
fn lgamma_t<T: Float>(x: T) -> T {
    // num_traits::Float doesn't have lgamma, use the f64 path.
    let x64 = x.to_f64().unwrap();
    T::from(lgamma_f64(x64)).unwrap()
}

/// lgamma for f64.
fn lgamma_f64(x: f64) -> f64 {
    // Use Stirling-Lanczos approximation (same as C lgamma)
    // Coefficients from "Numerical Recipes"
    if x <= 0.0 && x == x.floor() {
        return f64::INFINITY;
    }

    let coeffs: [f64; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -5.395_239_384_953_e-6,
    ];

    let x_adj = if x < 0.5 {
        // Reflection formula
        let sin_pi_x = (std::f64::consts::PI * x).sin();
        if sin_pi_x.abs() < 1e-30 {
            return f64::INFINITY;
        }
        return std::f64::consts::PI.ln() - sin_pi_x.abs().ln() - lgamma_f64(1.0 - x);
    } else {
        x - 1.0
    };

    let mut ser = 1.000_000_000_190_015;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (x_adj + 1.0 + i as f64);
    }

    let tmp = x_adj + 5.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + (tmp).ln() * (x_adj + 0.5) - tmp + ser.ln()
}

/// Scalar digamma (psi function).
fn digamma_t<T: Float>(x: T) -> T {
    let x64 = x.to_f64().unwrap();
    T::from(digamma_f64(x64)).unwrap()
}

/// Digamma for f64 via asymptotic expansion with shift.
fn digamma_f64(mut x: f64) -> f64 {
    let mut result = 0.0;

    // Shift argument up until x >= 6 for asymptotic accuracy.
    while x < 6.0 {
        result -= 1.0 / x;
        x += 1.0;
    }

    // Asymptotic expansion: psi(x) ~ ln(x) - 1/(2x) - 1/(12x^2) + ...
    let x2 = x * x;
    result += x.ln() - 0.5 / x - 1.0 / (12.0 * x2) + 1.0 / (120.0 * x2 * x2)
        - 1.0 / (252.0 * x2 * x2 * x2);

    result
}

// ---------------------------------------------------------------------------
// Backward node
// ---------------------------------------------------------------------------

/// Backward for Dirichlet rsample.
///
/// Uses the implicit reparameterization gradient through the Gamma-based
/// sampling. This is an approximation using the relationship:
/// d(x_k)/d(alpha_k) ≈ x_k * (digamma(alpha_k) - digamma(sum(alpha)))
/// corrected by the Jacobian of the simplex projection.
#[derive(Debug)]
struct DirichletRsampleBackward<T: Float> {
    concentration: Tensor<T>,
    samples: Tensor<T>,
    n: usize,
    k: usize,
}

impl<T: Float> GradFn<T> for DirichletRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let x_data = self.samples.data_vec()?;
        let alpha = self.concentration.data_vec()?;
        let n = self.n;
        let k = self.k;
        let zero = <T as num_traits::Zero>::zero();

        let alpha_sum: T = alpha.iter().copied().fold(zero, |a, b| a + b);
        let dig_sum = digamma_t(alpha_sum);

        // Accumulate gradient for concentration
        let mut grad_alpha = vec![zero; k];
        for s in 0..n {
            // Compute the correction factor
            let mut xg_sum = zero;
            for j in 0..k {
                xg_sum += x_data[s * k + j] * go[s * k + j];
            }

            for j in 0..k {
                let dig_alpha_j = digamma_t(alpha[j]);
                let grad_j = x_data[s * k + j] * (dig_alpha_j - dig_sum);
                // Correct with the simplex Jacobian
                grad_alpha[j] += (go[s * k + j] - xg_sum) * grad_j;
            }
        }

        let grad_alpha_t = Tensor::from_storage(
            TensorStorage::cpu(grad_alpha),
            self.concentration.shape().to_vec(),
            false,
        )?;
        let grad_alpha_t = if device.is_cuda() {
            grad_alpha_t.to(device)?
        } else {
            grad_alpha_t
        };

        Ok(vec![if self.concentration.requires_grad() {
            Some(grad_alpha_t)
        } else {
            None
        }])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.concentration]
    }

    fn name(&self) -> &'static str {
        "DirichletRsampleBackward"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, tensor};

    #[test]
    fn test_dirichlet_sample_shape() {
        let alpha = tensor(&[1.0f32, 1.0, 1.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100, 3]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_dirichlet_sample_2d_shape() {
        let alpha = tensor(&[2.0f32, 3.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.sample(&[5, 10]).unwrap();
        assert_eq!(samples.shape(), &[5, 10, 2]);
    }

    #[test]
    fn test_dirichlet_sample_on_simplex() {
        let alpha = tensor(&[0.5f32, 0.5, 0.5]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        let data = samples.data().unwrap();

        for s in 0..50 {
            let mut sum = 0.0f32;
            for j in 0..3 {
                let val = data[s * 3 + j];
                assert!(
                    val > 0.0,
                    "Dirichlet sample elements must be positive, got {val}"
                );
                sum += val;
            }
            assert!(
                (sum - 1.0).abs() < 1e-5,
                "Dirichlet sample must sum to 1, got {sum}"
            );
        }
    }

    #[test]
    fn test_dirichlet_rsample_has_grad() {
        let alpha = tensor(&[2.0f32, 3.0, 4.0]).unwrap().requires_grad_(true);
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert_eq!(samples.shape(), &[5, 3]);
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_dirichlet_rsample_no_grad_when_detached() {
        let alpha = tensor(&[2.0f32, 3.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_dirichlet_log_prob_uniform() {
        // Dirichlet([1, 1, 1]) is uniform on the simplex.
        // log_prob = lgamma(3) - 3*lgamma(1) = ln(2!) = ln(2)
        let alpha = tensor(&[1.0f32, 1.0, 1.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        // Any point on the simplex should have same log_prob
        let x = tensor(&[0.25f32, 0.25, 0.5]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 2.0f32.ln(); // lgamma(3) - 3*lgamma(1)
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_dirichlet_log_prob_batch() {
        let alpha = tensor(&[2.0f32, 2.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let x = from_slice(&[0.5f32, 0.5, 0.9, 0.1], &[2, 2]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[2]);

        let data = lp.data().unwrap();
        // For Dirichlet([2,2]), the mode is at [0.5, 0.5]
        assert!(data[0] > data[1], "log_prob at mode should be highest");
    }

    #[test]
    fn test_dirichlet_entropy_uniform() {
        // For Dirichlet([1,1,...,1]) with K categories:
        // H = sum(lgamma(1)) - lgamma(K) - (K - K)*digamma(K) - sum(0 * digamma(1))
        //   = -lgamma(K) = -ln((K-1)!)
        let alpha = tensor(&[1.0f32, 1.0, 1.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let h = dist.entropy().unwrap();
        // H = -lgamma(3) = -ln(2) ≈ -0.6931
        let expected = -(2.0f32.ln());
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-3,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_dirichlet_not_1d_errors() {
        let alpha = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(Dirichlet::new(alpha).is_err());
    }

    #[test]
    fn test_dirichlet_empty_errors() {
        let alpha = from_slice::<f32>(&[], &[0]).unwrap();
        assert!(Dirichlet::new(alpha).is_err());
    }

    #[test]
    fn test_dirichlet_num_categories() {
        let alpha = tensor(&[1.0f32, 2.0, 3.0, 4.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();
        assert_eq!(dist.num_categories(), 4);
    }

    #[test]
    fn test_dirichlet_f64() {
        let alpha = tensor(&[2.0f64, 3.0, 4.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50, 3]);

        let data = samples.data().unwrap();
        for s in 0..50 {
            let sum: f64 = (0..3).map(|j| data[s * 3 + j]).sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dirichlet_concentrated() {
        // High concentration => samples cluster near the uniform mean.
        //
        // For Dir(α=100, 100, 100) the per-component std is
        //   sqrt(α_i (α_0 - α_i) / (α_0² (α_0 + 1)))
        //   = sqrt(100·200 / (300²·301)) ≈ 0.0272
        // and the mean is 1/3 by symmetry. The test originally checked
        // each of 60 samples (20 batches × 3 components) against a
        // ±0.1 (~3.7σ) bound, which fails ~0.4% of the time across the
        // 60 draws and made the test flaky under workspace-parallel
        // runs.
        //
        // Switching to an empirical-mean check tightens the bound by
        // sqrt(N_SAMPLES) via CLT: with N_SAMPLES=200 the mean's std is
        // ≈ 0.0272 / sqrt(200) ≈ 0.00193, so a 0.05 tolerance is ~26σ —
        // genuinely never fails for a correct sampler.
        let alpha = tensor(&[100.0f32, 100.0, 100.0]).unwrap();
        let dist = Dirichlet::new(alpha).unwrap();

        const N_SAMPLES: usize = 200;
        let samples = dist.sample(&[N_SAMPLES]).unwrap();
        let data = samples.data().unwrap();
        let third = 1.0f32 / 3.0;

        // Empirical mean per component.
        let mut means = [0.0f32; 3];
        for s in 0..N_SAMPLES {
            for (j, m) in means.iter_mut().enumerate() {
                *m += data[s * 3 + j];
            }
        }
        for m in means.iter_mut() {
            *m /= N_SAMPLES as f32;
        }

        for (j, &m) in means.iter().enumerate() {
            assert!(
                (m - third).abs() < 0.05,
                "concentrated Dirichlet empirical mean for component {j} \
                 should be near 1/3 across {N_SAMPLES} samples, got {m}"
            );
        }

        // Sanity: every individual sample lies inside the simplex
        // [0, 1] (no per-element tolerance — that bound is racy).
        for s in 0..N_SAMPLES {
            for j in 0..3 {
                let v = data[s * 3 + j];
                assert!(
                    (0.0..=1.0).contains(&v),
                    "Dirichlet sample [s={s}, j={j}] = {v} not in [0, 1]"
                );
            }
        }
    }
}
