//! Log-Normal distribution.
//!
//! `LogNormal(loc, scale)` defines a log-normal distribution whose logarithm
//! is normally distributed with mean `loc` and standard deviation `scale`.
//! Supports reparameterized sampling via the underlying Normal distribution.

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Log-Normal distribution parameterized by `loc` (mu) and `scale` (sigma)
/// of the underlying normal distribution.
///
/// If `X ~ LogNormal(mu, sigma)`, then `ln(X) ~ Normal(mu, sigma)`.
///
/// # Reparameterization
///
/// `rsample` uses the reparameterization trick through the Normal distribution:
/// ```text
/// eps ~ N(0, 1)
/// z = exp(loc + scale * eps)
/// ```
/// Gradients flow through `loc` and `scale` via the autograd graph.
pub struct LogNormal<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
}

impl<T: Float> LogNormal<T> {
    /// Create a new Log-Normal distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `loc` and `scale` have incompatible shapes.
    pub fn new(loc: Tensor<T>, scale: Tensor<T>) -> FerrotorchResult<Self> {
        if loc.shape() != scale.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LogNormal: loc shape {:?} != scale shape {:?}",
                    loc.shape(),
                    scale.shape()
                ),
            });
        }
        Ok(Self { loc, scale })
    }

    /// The mean of the underlying normal distribution (mu).
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The standard deviation of the underlying normal distribution (sigma).
    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }

    /// The mean of the log-normal distribution: E[X] = exp(mu + sigma^2 / 2).
    pub fn mean_value(&self) -> FerrotorchResult<Vec<T>> {
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        Ok(loc_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&mu, &sigma)| (mu + half * sigma * sigma).exp())
            .collect())
    }

    /// The variance of the log-normal distribution:
    /// Var[X] = (exp(sigma^2) - 1) * exp(2*mu + sigma^2).
    pub fn variance_value(&self) -> FerrotorchResult<Vec<T>> {
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        Ok(loc_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&mu, &sigma)| {
                let s2 = sigma * sigma;
                ((s2).exp() - one) * (two * mu + s2).exp()
            })
            .collect())
    }
}

impl<T: Float> Distribution<T> for LogNormal<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "LogNormal::sample")?;
        // sample = exp(Normal(loc, scale).sample())
        let device = self.loc.device();
        let eps = creation::randn::<T>(shape)?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let eps_data = eps.data_vec()?;

        let result: Vec<T> = eps_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&e, &l), &s)| (l + s * e).exp())
            .collect();
        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.scale],
            "LogNormal::rsample",
        )?;
        let device = self.loc.device();
        let eps = creation::randn::<T>(shape)?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let eps_data = eps.data_vec()?;

        let result: Vec<T> = eps_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&e, &l), &s)| (l + s * e).exp())
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if (self.loc.requires_grad() || self.scale.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(LogNormalRsampleBackward {
                loc: self.loc.clone(),
                scale: self.scale.clone(),
                eps: eps.clone(),
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
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.scale, value],
            "LogNormal::log_prob",
        )?;
        // log_prob = Normal(loc, scale).log_prob(ln(value)) - ln(value)
        // = -0.5 * ((ln(x) - loc) / scale)^2 - ln(scale) - 0.5*ln(2*pi) - ln(x)
        let device = self.loc.device();
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let val_data = value.data_vec()?;

        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let half = T::from(0.5).unwrap();
        let log_2pi = two_pi.ln();

        let result: Vec<T> = val_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&x, &loc), &scale)| {
                let ln_x = x.ln();
                let z = (ln_x - loc) / scale;
                -(half * z * z) - scale.ln() - half * log_2pi - ln_x
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
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.scale],
            "LogNormal::entropy",
        )?;
        // entropy = loc + 0.5 + ln(scale * sqrt(2*pi))
        //         = loc + 0.5 + ln(scale) + 0.5 * ln(2*pi)
        let device = self.scale.device();
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let log_2pi = two_pi.ln();

        let result: Vec<T> = loc_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&mu, &sigma)| mu + half + sigma.ln() + half * log_2pi)
            .collect();

        let out = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.scale.shape().to_vec(),
            false,
        )?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "LogNormal::mean")?;
        let data = self.mean_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.loc.shape().to_vec(), false)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.scale],
            "LogNormal::variance",
        )?;
        let data = self.variance_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.loc.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "LogNormal::mode")?;
        // Mode = exp(loc - scale^2)
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let result: Vec<T> = loc_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&mu, &sigma)| (mu - sigma * sigma).exp())
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), self.loc.shape().to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for `z = exp(loc + scale * eps)`.
///
/// - d(z)/d(loc)   = z           (sum over sample dims)
/// - d(z)/d(scale) = z * eps     (sum over sample dims)
#[derive(Debug)]
struct LogNormalRsampleBackward<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
    eps: Tensor<T>,
}

impl<T: Float> GradFn<T> for LogNormalRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let eps_data = self.eps.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // z = exp(loc + scale * eps)
        // d(z)/d(loc) = z, summed over sample dims
        let grad_loc_val: T = go
            .iter()
            .zip(eps_data.iter())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .fold(zero, |acc, (((&g, &e), &l), &s)| {
                let z = (l + s * e).exp();
                acc + g * z
            });
        let grad_loc = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_loc_val]),
            self.loc.shape().to_vec(),
            false,
        )?;
        let grad_loc = if device.is_cuda() {
            grad_loc.to(device)?
        } else {
            grad_loc
        };

        // d(z)/d(scale) = z * eps, summed over sample dims
        let grad_scale_val: T = go
            .iter()
            .zip(eps_data.iter())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .fold(zero, |acc, (((&g, &e), &l), &s)| {
                let z = (l + s * e).exp();
                acc + g * z * e
            });
        let grad_scale = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_scale_val]),
            self.scale.shape().to_vec(),
            false,
        )?;
        let grad_scale = if device.is_cuda() {
            grad_scale.to(device)?
        } else {
            grad_scale
        };

        Ok(vec![
            if self.loc.requires_grad() {
                Some(grad_loc)
            } else {
                None
            },
            if self.scale.requires_grad() {
                Some(grad_scale)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.loc, &self.scale]
    }

    fn name(&self) -> &'static str {
        "LogNormalRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_lognormal_sample_shape() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = LogNormal::new(loc, scale).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_lognormal_sample_positive() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = LogNormal::new(loc, scale).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(x > 0.0, "LogNormal sample should be positive, got {x}");
        }
    }

    #[test]
    fn test_lognormal_rsample_has_grad() {
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = LogNormal::new(loc, scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_lognormal_log_prob() {
        // LogNormal(0, 1) at x=1: ln(1) = 0, Normal(0,1).log_prob(0) - ln(1)
        // = -0.5*ln(2*pi) - 0 = -0.5*ln(2*pi)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = LogNormal::new(loc, scale).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -0.5 * (2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_lognormal_log_prob_at_e() {
        // LogNormal(0, 1) at x=e: ln(e) = 1
        // log_prob = -0.5*(1/1)^2 - ln(1) - 0.5*ln(2*pi) - 1
        // = -0.5 - 0.5*ln(2*pi) - 1 = -1.5 - 0.5*ln(2*pi)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = LogNormal::new(loc, scale).unwrap();

        let x = scalar(std::f32::consts::E).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -1.5 - 0.5 * (2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_lognormal_entropy() {
        // entropy = mu + 0.5 + ln(sigma) + 0.5 * ln(2*pi)
        // For mu=0, sigma=1: entropy = 0.5 + 0 + 0.5*ln(2*pi) = 0.5 + 0.5*ln(2*pi)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = LogNormal::new(loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 0.5 + 0.5 * (2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_lognormal_shape_mismatch() {
        let loc = scalar(0.0f32).unwrap();
        let scale = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(LogNormal::new(loc, scale).is_err());
    }

    #[test]
    fn test_lognormal_rsample_backward() {
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = LogNormal::new(loc.clone(), scale.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        let loc_grad = loc.grad().unwrap().unwrap();
        assert!(loc_grad.item().unwrap().is_finite());
        // d(sum(exp(loc + scale*eps)))/d(loc) = sum(exp(loc + scale*eps)) > 0
        assert!(
            loc_grad.item().unwrap() > 0.0,
            "expected positive loc_grad, got {}",
            loc_grad.item().unwrap()
        );

        let scale_grad = scale.grad().unwrap().unwrap();
        assert!(scale_grad.item().unwrap().is_finite());
    }

    #[test]
    fn test_lognormal_f64() {
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = LogNormal::new(loc, scale).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(1.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -0.5 * (2.0f64 * std::f64::consts::PI).ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // mean / mode / variance (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_lognormal_mean_mode_variance() {
        let dist = LogNormal::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        // mean = exp(0 + 0.5) = e^0.5
        assert!((dist.mean().unwrap().item().unwrap() - 0.5_f64.exp()).abs() < 1e-10);
        // mode = exp(0 - 1) = e^-1
        assert!((dist.mode().unwrap().item().unwrap() - (-1.0_f64).exp()).abs() < 1e-10);
        // var = (e - 1) * e
        let v = dist.variance().unwrap().item().unwrap();
        let expected = (1.0_f64.exp() - 1.0) * 1.0_f64.exp();
        assert!((v - expected).abs() < 1e-10);
    }
}
