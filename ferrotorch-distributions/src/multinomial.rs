//! Multinomial distribution.
//!
//! `Multinomial(total_count, probs)` defines a distribution over count vectors
//! representing the number of times each of `K` categories was drawn in
//! `total_count` independent trials, where each trial draws a category
//! according to `probs`.
//!
//! This is a discrete distribution and does not support reparameterized sampling.
//!
//! [CL-331] ferrotorch#331 — multivariate distributions

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Multinomial distribution parameterized by `total_count` (number of trials)
/// and `probs` (category probabilities).
///
/// Samples are `K`-dimensional vectors of non-negative integers (as floats)
/// that sum to `total_count`.
///
/// # Discrete
///
/// This is a discrete distribution. `rsample` returns an error.
pub struct Multinomial<T: Float> {
    total_count: usize,
    /// Normalized probabilities.
    probs: Tensor<T>,
    /// Precomputed CDF for inverse-CDF sampling.
    cdf: Vec<T>,
    /// Precomputed log-probs for log_prob computation.
    log_probs: Vec<T>,
    num_categories: usize,
}

impl<T: Float> Multinomial<T> {
    /// Create a new Multinomial distribution.
    ///
    /// `probs` must be a 1-D tensor whose elements are non-negative and sum to
    /// a positive value. Probabilities are normalized internally.
    ///
    /// # Errors
    ///
    /// Returns an error if `probs` is not 1-D, empty, or has zero total probability.
    pub fn new(total_count: usize, probs: Tensor<T>) -> FerrotorchResult<Self> {
        if probs.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Multinomial: probs must be 1-D, got shape {:?}",
                    probs.shape()
                ),
            });
        }

        let k = probs.shape()[0];
        if k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Multinomial: probs must have at least one category".into(),
            });
        }

        let probs_data = probs.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let total: T = probs_data.iter().copied().fold(zero, |a, b| a + b);

        if total <= zero {
            return Err(FerrotorchError::InvalidArgument {
                message: "Multinomial: probs must sum to a positive value".into(),
            });
        }

        // Build CDF for sampling
        let mut cdf = Vec::with_capacity(k);
        let mut cumsum = zero;
        for &p in probs_data.iter() {
            cumsum += p / total;
            cdf.push(cumsum);
        }
        if let Some(last) = cdf.last_mut() {
            *last = one;
        }

        // Precompute log-probs (log-softmax)
        let eps = T::from(1e-7).unwrap();
        let log_probs: Vec<T> = probs_data
            .iter()
            .map(|&p| (p / total).max(eps).ln())
            .collect();

        Ok(Self {
            total_count,
            probs,
            cdf,
            log_probs,
            num_categories: k,
        })
    }

    /// The total number of trials.
    pub fn total_count(&self) -> usize {
        self.total_count
    }

    /// The probability parameters.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// The number of categories.
    pub fn num_categories(&self) -> usize {
        self.num_categories
    }
}

impl<T: Float> Distribution<T> for Multinomial<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Multinomial::sample")?;
        let device = self.probs.device();
        let n: usize = shape.iter().product();
        let k = self.num_categories;

        // For each sample in the batch, draw total_count categorical samples
        // and count occurrences.
        let u_flat = creation::rand::<T>(&[n * self.total_count])?;
        let u_data = u_flat.data_vec()?;

        let mut result = Vec::with_capacity(n * k);
        for s in 0..n {
            let mut counts = vec![<T as num_traits::Zero>::zero(); k];
            let one = <T as num_traits::One>::one();

            for t in 0..self.total_count {
                let u_val = u_data[s * self.total_count + t];

                // Binary search through CDF
                let mut lo = 0usize;
                let mut hi = k;
                while lo < hi {
                    let mid = lo + (hi - lo) / 2;
                    if self.cdf[mid] <= u_val {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                let idx = lo.min(k - 1);
                counts[idx] += one;
            }

            result.extend_from_slice(&counts);
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

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Multinomial distribution does not support reparameterized sampling. \
                      Use sample() with REINFORCE or relaxation methods instead."
                .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs, value], "Multinomial::log_prob")?;
        // log_prob = lgamma(n+1) - sum(lgamma(x_k+1)) + sum(x_k * log(p_k))
        let device = self.probs.device();
        let k = self.num_categories;
        let val_data = value.data_vec()?;

        let n = val_data.len() / k;
        if val_data.len() != n * k {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Multinomial log_prob: value has {} elements, not divisible by k={}",
                    val_data.len(),
                    k
                ),
            });
        }

        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let mut result = Vec::with_capacity(n);
        for s in 0..n {
            // n_total = sum of counts for this sample
            let n_total: T = (0..k).map(|j| val_data[s * k + j]).fold(zero, |a, b| a + b);

            // lgamma(n+1) - sum(lgamma(x_k+1)) + sum(x_k * log(p_k))
            let log_factorial_n = lgamma_t(n_total + one);
            let mut log_factorial_xs = zero;
            let mut log_powers = zero;

            for j in 0..k {
                let x_j = val_data[s * k + j];
                log_factorial_xs += lgamma_t(x_j + one);
                if x_j > zero {
                    log_powers += x_j * self.log_probs[j];
                }
            }

            result.push(log_factorial_n - log_factorial_xs + log_powers);
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Multinomial::entropy")?;
        // Use the exact formula:
        // H = lgamma(n+1) + n * H_cat - sum_x P(x) * sum_k lgamma(x_k + 1)
        // For large n this is expensive, so we use the approximation:
        // H ≈ 0.5 * (K-1) * ln(2*pi*e*n) + 0.5 * sum(ln(p_k)) for large n
        //
        // For exact computation with small total_count, we enumerate.
        // For simplicity and correctness, use the Stirling-based approximation.
        let device = self.probs.device();
        let n_t = T::from(self.total_count).unwrap();
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let probs_data = self.probs.data_vec()?;
        let total: T = probs_data.iter().copied().fold(zero, |a, b| a + b);
        let eps = T::from(1e-7).unwrap();

        // Categorical entropy
        let mut cat_entropy = zero;
        for &pj in &probs_data {
            let p = (pj / total).max(eps);
            cat_entropy = cat_entropy - p * p.ln();
        }

        // Approximate multinomial entropy via:
        // H = n * H_cat + 0.5 * (K-1) * ln(2*pi*n/e) + ... (normal approx)
        // Simpler: H ≈ n * H_cat (good for moderate-large n)
        let h = n_t * cat_entropy - lgamma_t(n_t + one);

        // Add the expected sum of lgamma(x_k + 1) term (approximation)
        // For each category: E[lgamma(X_k + 1)] ≈ lgamma(n*p_k + 1) for large n
        let mut correction = zero;
        for &pj in &probs_data {
            let p = (pj / total).max(eps);
            correction += lgamma_t(n_t * p + one);
        }
        let h = h + correction;

        let out = Tensor::from_storage(TensorStorage::cpu(vec![h]), vec![], false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Scalar special functions (shared with dirichlet)
// ---------------------------------------------------------------------------

fn lgamma_t<T: Float>(x: T) -> T {
    let x64 = x.to_f64().unwrap();
    T::from(lgamma_f64(x64)).unwrap()
}

fn lgamma_f64(x: f64) -> f64 {
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

    if x < 0.5 {
        let sin_pi_x = (std::f64::consts::PI * x).sin();
        if sin_pi_x.abs() < 1e-30 {
            return f64::INFINITY;
        }
        return std::f64::consts::PI.ln() - sin_pi_x.abs().ln() - lgamma_f64(1.0 - x);
    }

    let x_adj = x - 1.0;
    let mut ser = 1.000_000_000_190_015;
    for (i, &c) in coeffs.iter().enumerate() {
        ser += c / (x_adj + 1.0 + i as f64);
    }

    let tmp = x_adj + 5.5;
    (2.0 * std::f64::consts::PI).sqrt().ln() + tmp.ln() * (x_adj + 0.5) - tmp + ser.ln()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, tensor};

    #[test]
    fn test_multinomial_sample_shape() {
        let probs = tensor(&[0.2f32, 0.3, 0.5]).unwrap();
        let dist = Multinomial::new(10, probs).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100, 3]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_multinomial_sample_2d_shape() {
        let probs = tensor(&[0.5f32, 0.5]).unwrap();
        let dist = Multinomial::new(5, probs).unwrap();

        let samples = dist.sample(&[4, 6]).unwrap();
        assert_eq!(samples.shape(), &[4, 6, 2]);
    }

    #[test]
    fn test_multinomial_sample_sums_to_total_count() {
        let probs = tensor(&[0.1f32, 0.2, 0.3, 0.4]).unwrap();
        let dist = Multinomial::new(20, probs).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        let data = samples.data().unwrap();

        for s in 0..50 {
            let sum: f32 = (0..4).map(|j| data[s * 4 + j]).sum();
            assert!(
                (sum - 20.0).abs() < 1e-5,
                "Multinomial samples must sum to total_count={}, got {sum}",
                20
            );
        }
    }

    #[test]
    fn test_multinomial_sample_nonnegative() {
        let probs = tensor(&[0.5f32, 0.5]).unwrap();
        let dist = Multinomial::new(10, probs).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        let data = samples.data().unwrap();

        for &x in data {
            assert!(x >= 0.0, "Multinomial counts must be non-negative, got {x}");
        }
    }

    #[test]
    fn test_multinomial_sample_deterministic() {
        // probs = [0, 0, 1] => all counts go to category 2
        let probs = tensor(&[0.0f32, 0.0, 1.0]).unwrap();
        let dist = Multinomial::new(15, probs).unwrap();

        let samples = dist.sample(&[20]).unwrap();
        let data = samples.data().unwrap();

        for s in 0..20 {
            assert_eq!(data[s * 3], 0.0);
            assert_eq!(data[s * 3 + 1], 0.0);
            assert_eq!(data[s * 3 + 2], 15.0);
        }
    }

    #[test]
    fn test_multinomial_rsample_errors() {
        let probs = tensor(&[0.5f32, 0.5]).unwrap();
        let dist = Multinomial::new(5, probs).unwrap();
        assert!(dist.rsample(&[5]).is_err());
    }

    #[test]
    fn test_multinomial_log_prob() {
        // For Multinomial(10, [0.5, 0.5]), value = [5, 5]:
        // log_prob = lgamma(11) - 2*lgamma(6) + 5*log(0.5) + 5*log(0.5)
        //          = log(10!) - 2*log(5!) + 10*log(0.5)
        //          = log(252) + 10*log(0.5)
        let probs = tensor(&[0.5f32, 0.5]).unwrap();
        let dist = Multinomial::new(10, probs).unwrap();

        let x = tensor(&[5.0f32, 5.0]).unwrap();
        let lp = dist.log_prob(&x).unwrap();

        let expected = 252.0f32.ln() + 10.0 * 0.5f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-3,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_multinomial_log_prob_batch() {
        let probs = tensor(&[0.5f32, 0.5]).unwrap();
        let dist = Multinomial::new(10, probs).unwrap();

        let x = from_slice(&[5.0f32, 5.0, 10.0, 0.0], &[2, 2]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[2]);

        let data = lp.data().unwrap();
        // [5, 5] should have higher log_prob than [10, 0] for uniform probs
        assert!(data[0] > data[1]);
    }

    #[test]
    fn test_multinomial_not_1d_errors() {
        let probs = from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2]).unwrap();
        assert!(Multinomial::new(5, probs).is_err());
    }

    #[test]
    fn test_multinomial_empty_errors() {
        let probs = from_slice::<f32>(&[], &[0]).unwrap();
        assert!(Multinomial::new(5, probs).is_err());
    }

    #[test]
    fn test_multinomial_num_categories() {
        let probs = tensor(&[0.1f32, 0.2, 0.3, 0.4]).unwrap();
        let dist = Multinomial::new(10, probs).unwrap();
        assert_eq!(dist.num_categories(), 4);
        assert_eq!(dist.total_count(), 10);
    }

    #[test]
    fn test_multinomial_f64() {
        let probs = tensor(&[0.3f64, 0.7]).unwrap();
        let dist = Multinomial::new(20, probs).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50, 2]);

        let data = samples.data().unwrap();
        for s in 0..50 {
            let sum: f64 = data[s * 2] + data[s * 2 + 1];
            assert!((sum - 20.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_multinomial_single_trial() {
        // total_count = 1 is equivalent to one-hot Categorical
        let probs = tensor(&[0.2f32, 0.3, 0.5]).unwrap();
        let dist = Multinomial::new(1, probs).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        let data = samples.data().unwrap();

        for s in 0..100 {
            let sum: f32 = (0..3).map(|j| data[s * 3 + j]).sum();
            assert!((sum - 1.0).abs() < 1e-5);

            // Exactly one category should have count=1
            let mut one_count = 0;
            for j in 0..3 {
                let v = data[s * 3 + j];
                assert!(
                    v == 0.0 || v == 1.0,
                    "single trial counts should be 0 or 1, got {v}"
                );
                if v == 1.0 {
                    one_count += 1;
                }
            }
            assert_eq!(one_count, 1);
        }
    }
}
