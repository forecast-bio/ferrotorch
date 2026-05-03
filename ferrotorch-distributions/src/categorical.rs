//! Categorical distribution.
//!
//! `Categorical(probs)` defines a distribution over `{0, 1, ..., K-1}` where
//! `K` is the number of categories. This is a discrete distribution and does
//! not support reparameterized sampling.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Categorical distribution over `K` classes, parameterized by `probs`.
///
/// `probs` is a 1-D tensor of length `K` whose elements sum to 1. Each
/// element `probs[k]` gives the probability of drawing class `k`.
///
/// Samples are returned as float tensors containing integer class indices
/// (e.g., `0.0`, `1.0`, `2.0`, ...).
///
/// # Discrete
///
/// This is a discrete distribution. `rsample` returns an error. Use
/// score-function estimators (REINFORCE) or the Gumbel-Softmax trick
/// for gradient-based optimization with categorical variables.
pub struct Categorical<T: Float> {
    probs: Tensor<T>,
    /// Cumulative distribution function, precomputed for efficient sampling.
    cdf: Vec<T>,
    num_categories: usize,
}

impl<T: Float> Categorical<T> {
    /// Create a new Categorical distribution.
    ///
    /// `probs` must be a 1-D tensor whose elements are non-negative and sum
    /// to a positive value. The probabilities are normalized internally.
    ///
    /// # Errors
    ///
    /// Returns an error if `probs` is not 1-D or has zero total probability.
    pub fn new(probs: Tensor<T>) -> FerrotorchResult<Self> {
        if probs.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Categorical: probs must be 1-D, got shape {:?}",
                    probs.shape()
                ),
            });
        }

        let k = probs.shape()[0];
        if k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Categorical: probs must have at least one category".into(),
            });
        }

        let probs_data = probs.data_vec()?;

        // Normalize probabilities.
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let total: T = probs_data.iter().copied().fold(zero, |a, b| a + b);
        if total <= zero {
            return Err(FerrotorchError::InvalidArgument {
                message: "Categorical: probs must sum to a positive value".into(),
            });
        }

        // Build CDF for inverse-CDF sampling.
        let mut cdf = Vec::with_capacity(k);
        let mut cumsum = zero;
        for &p in probs_data.iter() {
            cumsum += p / total;
            cdf.push(cumsum);
        }
        // Ensure the last CDF entry is exactly 1 to avoid floating-point edge cases.
        if let Some(last) = cdf.last_mut() {
            *last = one;
        }

        Ok(Self {
            probs,
            cdf,
            num_categories: k,
        })
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

impl<T: Float> Distribution<T> for Categorical<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Categorical::sample")?;
        let device = self.probs.device();
        let numel: usize = shape.iter().product();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;

        let mut result = Vec::with_capacity(numel);
        for &u_val in u_data.iter() {
            // Binary search through CDF for inverse-CDF sampling.
            let mut lo = 0usize;
            let mut hi = self.num_categories;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.cdf[mid] <= u_val {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            // Clamp to valid range.
            let idx = lo.min(self.num_categories - 1);
            result.push(T::from(idx).unwrap());
        }

        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Categorical distribution does not support reparameterized sampling. \
                      Use sample() with REINFORCE or the Gumbel-Softmax trick instead."
                .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs, value], "Categorical::log_prob")?;
        // log_prob = log(probs[index])
        let device = self.probs.device();
        let probs_data = self.probs.data_vec()?;
        let val_data = value.data_vec()?;

        // Normalize probs for log computation.
        let zero = <T as num_traits::Zero>::zero();
        let total: T = probs_data.iter().copied().fold(zero, |a, b| a + b);
        let eps = T::from(1e-7).unwrap();

        let result: Vec<T> = val_data
            .iter()
            .map(|&x| {
                let idx = x.to_usize().unwrap_or(usize::MAX);
                if idx < self.num_categories {
                    let p = probs_data[idx] / total;
                    p.max(eps).ln()
                } else {
                    T::neg_infinity()
                }
            })
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Categorical::entropy")?;
        // entropy = -sum(p * log(p))
        let device = self.probs.device();
        let probs_data = self.probs.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let total: T = probs_data.iter().copied().fold(zero, |a, b| a + b);
        let eps = T::from(1e-7).unwrap();

        let h: T = probs_data.iter().fold(zero, |acc, &p| {
            let p_norm = (p / total).max(eps);
            acc - p_norm * p_norm.ln()
        });

        let out = Tensor::from_storage(TensorStorage::cpu(vec![h]), vec![], false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar, tensor};

    #[test]
    fn test_categorical_sample_shape() {
        let probs = tensor(&[0.2f32, 0.3, 0.5]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_categorical_sample_2d_shape() {
        let probs = tensor(&[0.5f32, 0.5]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let samples = dist.sample(&[10, 20]).unwrap();
        assert_eq!(samples.shape(), &[10, 20]);
    }

    #[test]
    fn test_categorical_sample_valid_range() {
        let probs = tensor(&[0.1f32, 0.2, 0.3, 0.4]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            let idx = x as usize;
            assert!(idx < 4, "Categorical sample should be in [0, 3], got {idx}");
        }
    }

    #[test]
    fn test_categorical_sample_deterministic() {
        // probs = [0, 0, 1] => all samples should be 2
        let probs = tensor(&[0.0f32, 0.0, 1.0]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        let data = samples.data().unwrap();
        assert!(
            data.iter().all(|&x| x == 2.0),
            "all samples should be 2.0 when probs=[0,0,1]"
        );
    }

    #[test]
    fn test_categorical_rsample_errors() {
        let probs = tensor(&[0.5f32, 0.5]).unwrap();
        let dist = Categorical::new(probs).unwrap();
        assert!(dist.rsample(&[5]).is_err());
    }

    #[test]
    fn test_categorical_log_prob() {
        // probs = [0.2, 0.3, 0.5], log_prob(2) = log(0.5)
        let probs = tensor(&[0.2f32, 0.3, 0.5]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let x = scalar(2.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_categorical_log_prob_first_class() {
        let probs = tensor(&[0.2f32, 0.3, 0.5]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.2f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_categorical_log_prob_batch() {
        let probs = tensor(&[0.25f32, 0.25, 0.25, 0.25]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let x = from_slice(&[0.0f32, 1.0, 2.0, 3.0], &[4]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[4]);

        let data = lp.data().unwrap();
        let expected = 0.25f32.ln();
        for &val in data {
            assert!(
                (val - expected).abs() < 1e-5,
                "expected {expected}, got {val}"
            );
        }
    }

    #[test]
    fn test_categorical_log_prob_out_of_range() {
        let probs = tensor(&[0.5f32, 0.5]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let x = scalar(5.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!(lp.item().unwrap().is_infinite() && lp.item().unwrap() < 0.0);
    }

    #[test]
    fn test_categorical_entropy_uniform() {
        // entropy of uniform(K) = log(K)
        let probs = tensor(&[0.25f32, 0.25, 0.25, 0.25]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 4.0f32.ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_categorical_entropy_deterministic() {
        // entropy approaches 0 for deterministic distribution
        let probs = tensor(&[0.0f32, 0.0, 1.0]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            h.item().unwrap() < 0.01,
            "expected near-zero entropy for deterministic dist, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_categorical_entropy_binary() {
        // entropy of Categorical([0.5, 0.5]) = log(2)
        let probs = tensor(&[0.5f32, 0.5]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 2.0f32.ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_categorical_not_1d_errors() {
        let probs = from_slice(&[0.5f32, 0.5, 0.5, 0.5], &[2, 2]).unwrap();
        assert!(Categorical::new(probs).is_err());
    }

    #[test]
    fn test_categorical_empty_errors() {
        let probs = from_slice::<f32>(&[], &[0]).unwrap();
        assert!(Categorical::new(probs).is_err());
    }

    #[test]
    fn test_categorical_unnormalized_probs() {
        // Probs that don't sum to 1 should still work (normalized internally)
        let probs = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        // log_prob(2) = log(3/6) = log(0.5)
        let x = scalar(2.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_categorical_num_categories() {
        let probs = tensor(&[0.1f32, 0.2, 0.3, 0.4]).unwrap();
        let dist = Categorical::new(probs).unwrap();
        assert_eq!(dist.num_categories(), 4);
    }

    #[test]
    fn test_categorical_f64() {
        let probs = tensor(&[0.3f64, 0.7]).unwrap();
        let dist = Categorical::new(probs).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(1.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.7f64.ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-10);
    }
}
