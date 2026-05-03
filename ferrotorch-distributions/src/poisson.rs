//! Poisson distribution.
//!
//! `Poisson(rate)` defines a Poisson distribution with rate parameter `rate`
//! (lambda). This is a discrete distribution and does not support
//! reparameterized sampling.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;
use crate::special_fns::lgamma_scalar;

/// Poisson distribution parameterized by `rate` (lambda).
///
/// # Discrete
///
/// This is a discrete distribution. `rsample` returns an error because there
/// is no continuous reparameterization for Poisson. Use `sample` and
/// score-function estimators (REINFORCE) for gradient-based optimization.
pub struct Poisson<T: Float> {
    rate: Tensor<T>,
}

impl<T: Float> Poisson<T> {
    /// Create a new Poisson distribution.
    ///
    /// Each element of `rate` is the rate parameter (lambda) for that position.
    /// Values must be positive.
    pub fn new(rate: Tensor<T>) -> FerrotorchResult<Self> {
        Ok(Self { rate })
    }

    /// The rate (lambda) parameter.
    pub fn rate(&self) -> &Tensor<T> {
        &self.rate
    }

    /// The mean of the distribution: E[X] = lambda.
    pub fn mean(&self) -> &Tensor<T> {
        &self.rate
    }

    /// The variance of the distribution: Var[X] = lambda.
    pub fn variance(&self) -> &Tensor<T> {
        &self.rate
    }
}

impl<T: Float> Distribution<T> for Poisson<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Poisson::sample")?;
        // Knuth's algorithm for Poisson sampling.
        // For each sample, draw U ~ Uniform(0,1) repeatedly until product < exp(-lambda).
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let n: usize = shape.iter().product();

        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        // Pre-draw a generous batch of uniform samples
        let batch = (n * 30).max(1024);
        let mut unif_buf: Vec<T> = creation::rand::<T>(&[batch])?.data_vec()?;
        let mut ui = 0usize;

        let next_uniform = |ui: &mut usize, unif_buf: &mut Vec<T>| -> FerrotorchResult<T> {
            if *ui >= unif_buf.len() {
                *unif_buf = creation::rand::<T>(&[batch])?.data_vec()?;
                *ui = 0;
            }
            let val = unif_buf[*ui];
            *ui += 1;
            Ok(val)
        };

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let lambda = rate_data[i % rate_data.len()];
            let l = (-lambda).exp();
            let mut k = zero;
            let mut p = one;

            loop {
                let u = next_uniform(&mut ui, &mut unif_buf)?;
                p = p * u;
                if p <= l {
                    break;
                }
                k += one;
            }
            result.push(k);
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
            message: "Poisson distribution does not support reparameterized sampling. \
                      Use sample() with score-function estimators (REINFORCE) instead."
                .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate, value], "Poisson::log_prob")?;
        // log_prob = k * ln(lambda) - lambda - lgamma(k + 1)
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let val_data = value.data_vec()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = val_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&k, &lambda)| k * lambda.ln() - lambda - lgamma_scalar(k + one))
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Poisson::entropy")?;
        // No simple closed-form for Poisson entropy. Use the approximation:
        // H ~ 0.5 * ln(2 * pi * e * lambda) - 1/(12*lambda) - 1/(24*lambda^2)
        // This is accurate for lambda >= 1. For small lambda, we compute exactly.
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let half = T::from(0.5).unwrap();
        let two_pi_e = T::from(2.0 * std::f64::consts::PI * std::f64::consts::E).unwrap();

        let result: Vec<T> = rate_data
            .iter()
            .map(|&lambda| {
                if lambda < T::from(1.0).unwrap() {
                    // Exact computation for small lambda: sum -p(k)*log(p(k))
                    // Truncate when p(k) is negligible
                    let mut entropy = zero;
                    let mut log_p = -lambda; // log(p(0)) = -lambda
                    let mut k = zero;
                    for _i in 0..100 {
                        let p = log_p.exp();
                        if p > T::from(1e-15).unwrap() {
                            entropy = entropy - p * log_p;
                        }
                        k += one;
                        log_p = log_p + lambda.ln() - k.ln();
                        if log_p < T::from(-40.0).unwrap() {
                            break;
                        }
                    }
                    entropy
                } else {
                    // Stirling-series approximation
                    let inv_lambda = one / lambda;
                    half * (two_pi_e * lambda).ln()
                        - T::from(1.0 / 12.0).unwrap() * inv_lambda
                        - T::from(1.0 / 24.0).unwrap() * inv_lambda * inv_lambda
                }
            })
            .collect();

        let out = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.rate.shape().to_vec(),
            false,
        )?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        Ok(self.rate.clone())
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.rate], "Poisson::mode")?;
        // Mode = floor(rate); for integer rate, both rate-1 and rate are
        // modes — torch returns floor(rate).
        let rate_data = self.rate.data_vec()?;
        let result: Vec<T> = rate_data
            .iter()
            .map(|&r| T::from(r.to_f64().unwrap_or(0.0).floor()).unwrap())
            .collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.rate.shape().to_vec(),
            false,
        )
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        Ok(self.rate.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_poisson_sample_shape() {
        let rate = scalar(5.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_poisson_sample_nonnegative_integers() {
        let rate = scalar(3.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let samples = dist.sample(&[500]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(x >= 0.0, "Poisson sample should be non-negative, got {x}");
            assert!(
                (x - x.round()).abs() < 1e-6,
                "Poisson sample should be an integer, got {x}"
            );
        }
    }

    #[test]
    fn test_poisson_sample_mean() {
        // E[X] = lambda = 4.0
        let rate = scalar(4.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 4.0).abs() < 0.3, "expected mean ~4.0, got {mean}");
    }

    #[test]
    fn test_poisson_rsample_errors() {
        let rate = scalar(5.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();
        assert!(dist.rsample(&[5]).is_err());
    }

    #[test]
    fn test_poisson_log_prob() {
        // Poisson(lambda=1): P(k=0) = e^(-1), log_prob = -1
        let rate = scalar(1.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -1.0f32; // 0*ln(1) - 1 - lgamma(1) = -1
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_poisson_log_prob_k1() {
        // Poisson(lambda=2): P(k=1) = 2*e^(-2)
        // log_prob = 1*ln(2) - 2 - lgamma(2) = ln(2) - 2 - 0 = ln(2) - 2
        let rate = scalar(2.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 2.0f32.ln() - 2.0;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_poisson_log_prob_batch() {
        let rate = scalar(3.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let x = from_slice(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[5]);

        // Mode of Poisson(3) is at k=2 or k=3 (floor(lambda)), log_prob should peak there
        let data = lp.data().unwrap();
        assert!(data[2] > data[0]); // lp(2) > lp(0)
        assert!(data[3] > data[0]); // lp(3) > lp(0)
    }

    #[test]
    fn test_poisson_entropy_positive() {
        let rate = scalar(5.0f32).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            h.item().unwrap() > 0.0,
            "Poisson entropy should be positive, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_poisson_f64() {
        let rate = scalar(1.0f64).unwrap();
        let dist = Poisson::new(rate).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(0.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!((lp.item().unwrap() - (-1.0f64)).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // mean / mode / variance (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_poisson_mean_eq_variance_eq_rate() {
        let dist = Poisson::new(scalar(4.7f64).unwrap()).unwrap();
        // Poisson has an inherent `mean()` returning &Tensor; use FQ syntax
        // to invoke the trait methods which return Tensor by value.
        assert!((Distribution::mean(&dist).unwrap().item().unwrap() - 4.7).abs() < 1e-12);
        assert!((Distribution::variance(&dist).unwrap().item().unwrap() - 4.7).abs() < 1e-12);
        // mode = floor(4.7) = 4
        assert!((dist.mode().unwrap().item().unwrap() - 4.0).abs() < 1e-12);
    }
}
