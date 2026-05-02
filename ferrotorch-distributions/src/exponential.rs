//! Exponential distribution.
//!
//! `Exponential(rate)` defines an exponential distribution with rate parameter
//! `rate` (lambda). Supports reparameterized sampling via inverse CDF.
//!
//! [CL-329]

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Exponential distribution parameterized by `rate` (lambda).
///
/// # Reparameterization
///
/// `rsample` uses the inverse CDF (quantile) transform:
/// ```text
/// u ~ Uniform(0, 1)
/// z = -log(u) / rate
/// ```
/// Gradients flow through `rate` via the autograd graph.
pub struct Exponential<T: Float> {
    rate: Tensor<T>,
}

impl<T: Float> Exponential<T> {
    /// Create a new Exponential distribution.
    pub fn new(rate: Tensor<T>) -> FerrotorchResult<Self> {
        Ok(Self { rate })
    }

    /// The rate parameter.
    pub fn rate(&self) -> &Tensor<T> {
        &self.rate
    }
}

impl<T: Float> Distribution<T> for Exponential<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let device = self.rate.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let rate_data = self.rate.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&u_val, &r)| {
                // Clamp u away from 0 for numerical stability
                let u_safe = u_val.max(T::from(1e-30).unwrap());
                -u_safe.ln() / r
            })
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let device = self.rate.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let rate_data = self.rate.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&u_val, &r)| {
                let u_safe = u_val.max(T::from(1e-30).unwrap());
                -u_safe.ln() / r
            })
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if self.rate.requires_grad() && ferrotorch_core::is_grad_enabled() {
            let grad_fn = Arc::new(ExponentialRsampleBackward {
                rate: self.rate.clone(),
                u: u.clone(),
            });
            Tensor::from_operation(storage, shape.to_vec(), grad_fn)?
        } else {
            Tensor::from_storage(storage, shape.to_vec(), false)?
        };
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log_prob = log(rate) - rate * x
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let val_data = value.data_vec()?;

        let result: Vec<T> = val_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&x, &r)| r.ln() - r * x)
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // entropy = 1 - log(rate)
        let device = self.rate.device();
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = rate_data.iter().map(|&r| one - r.ln()).collect();

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

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // cdf(x) = 1 - exp(-rate * x) for x >= 0; 0 for x < 0.
        let val = value.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = val
            .iter()
            .zip(rate_data.iter().cycle())
            .map(
                |(&x, &r)| {
                    if x < zero { zero } else { one - (-r * x).exp() }
                },
            )
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // icdf(p) = -ln(1 - p) / rate, for p in [0, 1).
        let q_data = q.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = q_data
            .iter()
            .zip(rate_data.iter().cycle())
            .map(|(&p, &r)| -((one - p).ln()) / r)
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // 1 / rate
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = rate_data.iter().map(|&r| one / r).collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.rate.shape().to_vec(),
            false,
        )
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // Mode of exponential is 0.
        let zero = <T as num_traits::Zero>::zero();
        let n: usize = self.rate.shape().iter().product();
        Tensor::from_storage(
            TensorStorage::cpu(vec![zero; n.max(1)]),
            self.rate.shape().to_vec(),
            false,
        )
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // 1 / rate^2
        let rate_data = self.rate.data_vec()?;
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = rate_data.iter().map(|&r| one / (r * r)).collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.rate.shape().to_vec(),
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for Exponential rsample: z = -log(u) / rate.
///
/// d(z)/d(rate) = log(u) / rate^2 = -z / rate
#[derive(Debug)]
struct ExponentialRsampleBackward<T: Float> {
    rate: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for ExponentialRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let rate_data = self.rate.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // d(z)/d(rate) = log(u) / rate^2
        let grad_rate_val: T = go
            .iter()
            .zip(u_data.iter())
            .zip(rate_data.iter().cycle())
            .fold(zero, |acc, ((&g, &u_val), &r)| {
                let u_safe = u_val.max(T::from(1e-30).unwrap());
                acc + g * u_safe.ln() / (r * r)
            });

        let grad_rate = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_rate_val]),
            self.rate.shape().to_vec(),
            false,
        )?;
        let grad_rate = if device.is_cuda() {
            grad_rate.to(device)?
        } else {
            grad_rate
        };

        Ok(vec![if self.rate.requires_grad() {
            Some(grad_rate)
        } else {
            None
        }])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.rate]
    }

    fn name(&self) -> &'static str {
        "ExponentialRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_exponential_sample_shape() {
        let rate = scalar(1.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_exponential_sample_positive() {
        let rate = scalar(2.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(x > 0.0, "Exponential sample should be positive, got {x}");
        }
    }

    #[test]
    fn test_exponential_sample_mean() {
        // E[X] = 1/rate = 0.5
        let rate = scalar(2.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 0.5).abs() < 0.05, "expected mean ~0.5, got {mean}");
    }

    #[test]
    fn test_exponential_rsample_has_grad() {
        let rate = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_exponential_log_prob() {
        // Exp(1): log_prob(x) = -x
        let rate = scalar(1.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let x = scalar(2.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -2.0f32; // log(1) - 1*2 = -2
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_log_prob_rate2() {
        // Exp(2): log_prob(1) = log(2) - 2
        let rate = scalar(2.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 2.0f32.ln() - 2.0;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_entropy() {
        // entropy = 1 - log(rate)
        let rate = scalar(2.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 1.0 - 2.0f32.ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_entropy_rate1() {
        // Exp(1): entropy = 1
        let rate = scalar(1.0f32).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            (h.item().unwrap() - 1.0).abs() < 1e-5,
            "expected 1.0, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_rsample_backward() {
        let rate = scalar(2.0f32).unwrap().requires_grad_(true);
        let dist = Exponential::new(rate.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        let rate_grad = rate.grad().unwrap().unwrap();
        assert!(rate_grad.item().unwrap().is_finite());
        // Gradient should be negative (increasing rate decreases samples)
        assert!(
            rate_grad.item().unwrap() < 0.0,
            "expected negative grad, got {}",
            rate_grad.item().unwrap()
        );
    }

    #[test]
    fn test_exponential_f64() {
        let rate = scalar(1.0f64).unwrap();
        let dist = Exponential::new(rate).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(1.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!((lp.item().unwrap() - (-1.0f64)).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // CDF / ICDF / mean / mode / variance / stddev (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_exponential_mean_mode_variance() {
        // rate=2 → mean=0.5, mode=0, var=0.25
        let dist = Exponential::new(scalar(2.0f64).unwrap()).unwrap();
        assert!((dist.mean().unwrap().item().unwrap() - 0.5).abs() < 1e-10);
        assert!(dist.mode().unwrap().item().unwrap().abs() < 1e-12);
        assert!((dist.variance().unwrap().item().unwrap() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_cdf() {
        let dist = Exponential::new(scalar(1.0f64).unwrap()).unwrap();
        // cdf(0) = 0; cdf(1) = 1 - 1/e
        let x = from_slice::<f64>(&[-1.0, 0.0, 1.0], &[3]).unwrap();
        let c = dist.cdf(&x).unwrap();
        let d = c.data().unwrap();
        assert!(d[0].abs() < 1e-12);
        assert!(d[1].abs() < 1e-12);
        assert!((d[2] - (1.0 - (-1.0_f64).exp())).abs() < 1e-10);
    }

    #[test]
    fn test_exponential_icdf_roundtrip() {
        let dist = Exponential::new(scalar(2.5f64).unwrap()).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = scalar(p).unwrap();
            let x = dist.icdf(&q).unwrap();
            let p2 = dist.cdf(&x).unwrap();
            assert!((p2.item().unwrap() - p).abs() < 1e-10);
        }
    }
}
