//! Bernoulli distribution.
//!
//! `Bernoulli(probs)` defines a distribution over `{0, 1}` where the
//! probability of drawing `1` is `probs`. This is a discrete distribution
//! and does not support reparameterized sampling.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Bernoulli distribution parameterized by `probs` (probability of 1).
///
/// # Discrete
///
/// This is a discrete distribution. `rsample` returns an error because
/// there is no continuous reparameterization for Bernoulli. Use `sample`
/// and score-function estimators (REINFORCE) for gradient-based optimization.
pub struct Bernoulli<T: Float> {
    probs: Tensor<T>,
}

impl<T: Float> Bernoulli<T> {
    /// Create a new Bernoulli distribution.
    ///
    /// Each element of `probs` is the probability of drawing 1 at that
    /// position. Values should be in `[0, 1]`.
    pub fn new(probs: Tensor<T>) -> FerrotorchResult<Self> {
        Ok(Self { probs })
    }

    /// The probability parameters.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }
}

impl<T: Float> Distribution<T> for Bernoulli<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Bernoulli::sample")?;
        // sample = (rand < probs) as float
        let device = self.probs.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let probs_data = self.probs.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let result: Vec<T> = u_data
            .iter()
            .zip(probs_data.iter().cycle())
            .map(|(&u_val, &p)| if u_val < p { one } else { zero })
            .collect();
        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Bernoulli distribution does not support reparameterized sampling. \
                      Use sample() with score-function estimators (REINFORCE) instead."
                .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs, value], "Bernoulli::log_prob")?;
        // log_prob = x * log(p) + (1 - x) * log(1 - p)
        let device = self.probs.device();
        let probs_data = self.probs.data_vec()?;
        let val_data = value.data_vec()?;
        let one = <T as num_traits::One>::one();

        // Clamp probs to avoid log(0)
        let eps = T::from(1e-7).unwrap();

        let result: Vec<T> = val_data
            .iter()
            .zip(probs_data.iter().cycle())
            .map(|(&x, &p)| {
                let p_clamped = p.max(eps).min(one - eps);
                x * p_clamped.ln() + (one - x) * (one - p_clamped).ln()
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Bernoulli::entropy")?;
        // entropy = -p * log(p) - (1-p) * log(1-p)
        let device = self.probs.device();
        let probs_data = self.probs.data_vec()?;
        let one = <T as num_traits::One>::one();
        let eps = T::from(1e-7).unwrap();

        let result: Vec<T> = probs_data
            .iter()
            .map(|&p| {
                let p_clamped = p.max(eps).min(one - eps);
                -(p_clamped * p_clamped.ln()) - (one - p_clamped) * (one - p_clamped).ln()
            })
            .collect();

        let out = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.probs.shape().to_vec(),
            false,
        )?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs, value], "Bernoulli::cdf")?;
        // For x < 0: 0; for 0 <= x < 1: 1 - p; for x >= 1: 1.
        let val = value.data_vec()?;
        let probs_data = self.probs.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = val
            .iter()
            .zip(probs_data.iter().cycle())
            .map(|(&x, &p)| {
                if x < zero {
                    zero
                } else if x < one {
                    one - p
                } else {
                    one
                }
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        Ok(self.probs.clone())
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Bernoulli::mode")?;
        // Mode is 1 if p > 0.5 else 0 (NaN for p == 0.5 is the strict
        // convention; we use 0 to keep a valid finite answer).
        let probs_data = self.probs.data_vec()?;
        let half = T::from(0.5).unwrap();
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = probs_data
            .iter()
            .map(|&p| if p > half { one } else { zero })
            .collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.probs.shape().to_vec(),
            false,
        )
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs], "Bernoulli::variance")?;
        // p * (1 - p)
        let probs_data = self.probs.data_vec()?;
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = probs_data.iter().map(|&p| p * (one - p)).collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.probs.shape().to_vec(),
            false,
        )
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.probs, q], "Bernoulli::icdf")?;
        // Generalized inverse of the step CDF: F^{-1}(p) = 1 if p > 1-prob,
        // else 0 (matches torch's piecewise definition for discrete dists).
        // Equivalently: 1 if p > 1 - prob.
        let q_data = q.data_vec()?;
        let probs_data = self.probs.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = q_data
            .iter()
            .enumerate()
            .map(|(i, &p)| {
                let pi = if probs_data.len() == 1 {
                    0
                } else {
                    i % probs_data.len()
                };
                if p > one - probs_data[pi] { one } else { zero }
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), q.shape().to_vec(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_bernoulli_sample_shape() {
        let probs = scalar(0.5f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_bernoulli_sample_values_binary() {
        let probs = scalar(0.5f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(
                x == 0.0 || x == 1.0,
                "Bernoulli sample should be 0 or 1, got {x}"
            );
        }
    }

    #[test]
    fn test_bernoulli_sample_prob_1() {
        // probs = 1.0 => all samples should be 1
        let probs = scalar(1.0f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        let data = samples.data().unwrap();
        assert!(data.iter().all(|&x| x == 1.0));
    }

    #[test]
    fn test_bernoulli_sample_prob_0() {
        // probs = 0.0 => all samples should be 0
        let probs = scalar(0.0f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        let data = samples.data().unwrap();
        assert!(data.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_bernoulli_rsample_errors() {
        let probs = scalar(0.5f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();
        assert!(dist.rsample(&[5]).is_err());
    }

    #[test]
    fn test_bernoulli_log_prob_one() {
        // log_prob(1) for Bernoulli(0.7) = log(0.7)
        let probs = scalar(0.7f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.7f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_bernoulli_log_prob_zero() {
        // log_prob(0) for Bernoulli(0.7) = log(0.3)
        let probs = scalar(0.7f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.3f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_bernoulli_log_prob_batch() {
        let probs = scalar(0.5f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let x = from_slice(&[0.0, 1.0, 0.0, 1.0], &[4]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[4]);

        let data = lp.data().unwrap();
        let expected = 0.5f32.ln();
        // For p=0.5, log_prob(0) == log_prob(1)
        for &val in data {
            assert!((val - expected).abs() < 1e-5);
        }
    }

    #[test]
    fn test_bernoulli_entropy_fair() {
        // entropy of Bernoulli(0.5) = log(2)
        let probs = scalar(0.5f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 2.0f32.ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_bernoulli_entropy_deterministic() {
        // entropy approaches 0 as p -> 0 or p -> 1
        let probs = scalar(0.999f32).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            h.item().unwrap() < 0.1,
            "expected near-zero entropy, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_bernoulli_entropy_batch() {
        let probs = from_slice(&[0.2f32, 0.5, 0.8], &[3]).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let h = dist.entropy().unwrap();
        assert_eq!(h.shape(), &[3]);

        let data = h.data().unwrap();
        // Entropy is maximized at p=0.5
        assert!(data[1] > data[0]);
        assert!(data[1] > data[2]);
        // Symmetry: entropy(0.2) == entropy(0.8)
        assert!((data[0] - data[2]).abs() < 1e-5);
    }

    #[test]
    fn test_bernoulli_f64() {
        let probs = scalar(0.3f64).unwrap();
        let dist = Bernoulli::new(probs).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(1.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.3f64.ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // CDF / mean / mode / variance (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bernoulli_mean_variance() {
        let dist = Bernoulli::new(scalar(0.7f64).unwrap()).unwrap();
        assert!((dist.mean().unwrap().item().unwrap() - 0.7).abs() < 1e-12);
        assert!((dist.variance().unwrap().item().unwrap() - 0.21).abs() < 1e-12);
    }

    #[test]
    fn test_bernoulli_mode_high_p() {
        let dist = Bernoulli::new(scalar(0.8f64).unwrap()).unwrap();
        assert!((dist.mode().unwrap().item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_bernoulli_mode_low_p() {
        let dist = Bernoulli::new(scalar(0.2f64).unwrap()).unwrap();
        assert!(dist.mode().unwrap().item().unwrap().abs() < 1e-12);
    }

    #[test]
    fn test_bernoulli_cdf() {
        let dist = Bernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        let x = ferrotorch_core::creation::from_slice::<f64>(&[-1.0, 0.0, 0.5, 1.0, 2.0], &[5])
            .unwrap();
        let c = dist.cdf(&x).unwrap();
        let d = c.data().unwrap();
        // CDF(<0)=0; CDF([0,1))=1-p=0.7; CDF(>=1)=1
        assert!((d[0] - 0.0).abs() < 1e-12);
        assert!((d[1] - 0.7).abs() < 1e-12);
        assert!((d[2] - 0.7).abs() < 1e-12);
        assert!((d[3] - 1.0).abs() < 1e-12);
        assert!((d[4] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_bernoulli_icdf_step_at_one_minus_p() {
        // p=0.3: F(x)=1 if p_q > 0.7, else 0.
        let dist = Bernoulli::new(scalar(0.3f64).unwrap()).unwrap();
        let q =
            ferrotorch_core::creation::from_slice::<f64>(&[0.5, 0.7, 0.71, 0.99], &[4]).unwrap();
        let x = dist.icdf(&q).unwrap();
        let d = x.data().unwrap();
        // 0.5 -> 0; 0.7 -> 0 (boundary, strict gt); 0.71 -> 1; 0.99 -> 1.
        assert_eq!(d, &[0.0, 0.0, 1.0, 1.0]);
    }
}
