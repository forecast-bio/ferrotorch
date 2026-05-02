//! Half-Normal distribution.
//!
//! `HalfNormal(scale)` defines a half-normal distribution — the absolute value
//! of a `Normal(0, scale)` random variable. Supported on `[0, inf)`.
//! Supports reparameterized sampling.

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Half-Normal distribution parameterized by `scale`.
///
/// If `X ~ Normal(0, scale)`, then `|X| ~ HalfNormal(scale)`.
///
/// # Reparameterization
///
/// `rsample` uses the reparameterization trick:
/// ```text
/// eps ~ N(0, 1)
/// z = scale * |eps|
/// ```
/// Gradients flow through `scale` via the autograd graph.
pub struct HalfNormal<T: Float> {
    scale: Tensor<T>,
}

impl<T: Float> HalfNormal<T> {
    /// Create a new Half-Normal distribution.
    pub fn new(scale: Tensor<T>) -> FerrotorchResult<Self> {
        Ok(Self { scale })
    }

    /// The scale parameter.
    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }

    /// The mean of the distribution: E[X] = scale * sqrt(2/pi).
    pub fn mean_value(&self) -> FerrotorchResult<Vec<T>> {
        let scale_data = self.scale.data_vec()?;
        let sqrt_2_over_pi = T::from((2.0 / std::f64::consts::PI).sqrt()).unwrap();
        Ok(scale_data.iter().map(|&s| s * sqrt_2_over_pi).collect())
    }

    /// The variance of the distribution: Var[X] = scale^2 * (1 - 2/pi).
    pub fn variance_value(&self) -> FerrotorchResult<Vec<T>> {
        let scale_data = self.scale.data_vec()?;
        let one = <T as num_traits::One>::one();
        let two_over_pi = T::from(2.0 / std::f64::consts::PI).unwrap();
        Ok(scale_data
            .iter()
            .map(|&s| s * s * (one - two_over_pi))
            .collect())
    }
}

impl<T: Float> Distribution<T> for HalfNormal<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let device = self.scale.device();
        let eps = creation::randn::<T>(shape)?;
        let eps_data = eps.data_vec()?;
        let scale_data = self.scale.data_vec()?;

        let result: Vec<T> = eps_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&e, &s)| s * e.abs())
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let device = self.scale.device();
        let eps = creation::randn::<T>(shape)?;
        let eps_data = eps.data_vec()?;
        let scale_data = self.scale.data_vec()?;

        let result: Vec<T> = eps_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&e, &s)| s * e.abs())
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if self.scale.requires_grad() && ferrotorch_core::is_grad_enabled() {
            let grad_fn = Arc::new(HalfNormalRsampleBackward {
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
        // PDF = sqrt(2 / (pi * scale^2)) * exp(-x^2 / (2 * scale^2))  for x >= 0
        // log_prob = 0.5 * ln(2/pi) - ln(scale) - x^2 / (2 * scale^2)
        //
        // For x < 0, return -inf (log(0)).
        let device = self.scale.device();
        let scale_data = self.scale.data_vec()?;
        let val_data = value.data_vec()?;
        let half = T::from(0.5).unwrap();
        let two_over_pi = T::from(2.0 / std::f64::consts::PI).unwrap();
        let half_ln_2_over_pi = half * two_over_pi.ln();
        let zero = <T as num_traits::Zero>::zero();

        let result: Vec<T> = val_data
            .iter()
            .zip(scale_data.iter().cycle())
            .map(|(&x, &scale)| {
                if x < zero {
                    T::neg_infinity()
                } else {
                    half_ln_2_over_pi - scale.ln() - x * x / (T::from(2.0).unwrap() * scale * scale)
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
        // entropy = 0.5 * ln(pi * scale^2 / 2) + 0.5
        //         = 0.5 * ln(pi/2) + ln(scale) + 0.5
        let device = self.scale.device();
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        let pi_over_2 = T::from(std::f64::consts::PI / 2.0).unwrap();
        let half_ln_pi_over_2 = half * pi_over_2.ln();

        let result: Vec<T> = scale_data
            .iter()
            .map(|&scale| half_ln_pi_over_2 + scale.ln() + half)
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
        let data = self.mean_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.scale.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // Mode of HalfNormal is 0.
        let zero = <T as num_traits::Zero>::zero();
        let n: usize = self.scale.shape().iter().product();
        Tensor::from_storage(
            TensorStorage::cpu(vec![zero; n.max(1)]),
            self.scale.shape().to_vec(),
            false,
        )
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        let data = self.variance_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.scale.shape().to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for `z = scale * |eps|`.
///
/// d(z)/d(scale) = |eps| (sum over sample dims)
#[derive(Debug)]
struct HalfNormalRsampleBackward<T: Float> {
    scale: Tensor<T>,
    eps: Tensor<T>,
}

impl<T: Float> GradFn<T> for HalfNormalRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let eps_data = self.eps.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // grad_scale = sum(grad_output * |eps|)
        let grad_scale_val: T = go
            .iter()
            .zip(eps_data.iter())
            .fold(zero, |acc, (&g, &e)| acc + g * e.abs());
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

        Ok(vec![if self.scale.requires_grad() {
            Some(grad_scale)
        } else {
            None
        }])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.scale]
    }

    fn name(&self) -> &'static str {
        "HalfNormalRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::scalar;

    #[test]
    fn test_half_normal_sample_shape() {
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_half_normal_sample_nonnegative() {
        let scale = scalar(2.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(
                x >= 0.0,
                "HalfNormal sample should be non-negative, got {x}"
            );
        }
    }

    #[test]
    fn test_half_normal_sample_mean() {
        // E[X] = scale * sqrt(2/pi) ~ 1.0 * 0.7979 for scale=1
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let expected = (2.0f32 / std::f32::consts::PI).sqrt();
        assert!(
            (mean - expected).abs() < 0.05,
            "expected mean ~{expected}, got {mean}"
        );
    }

    #[test]
    fn test_half_normal_rsample_has_grad() {
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_half_normal_log_prob_at_zero() {
        // HalfNormal(1) at x=0: log_prob = 0.5*ln(2/pi) - ln(1) - 0 = 0.5*ln(2/pi)
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5 * (2.0f32 / std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_log_prob_at_one() {
        // HalfNormal(1) at x=1: log_prob = 0.5*ln(2/pi) - 0 - 0.5 = 0.5*ln(2/pi) - 0.5
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5 * (2.0f32 / std::f32::consts::PI).ln() - 0.5;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_log_prob_negative_is_neginf() {
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let x = scalar(-1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!(
            lp.item().unwrap().is_infinite() && lp.item().unwrap() < 0.0,
            "log_prob of negative value should be -inf, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_log_prob_scale2() {
        // HalfNormal(2) at x=0: log_prob = 0.5*ln(2/pi) - ln(2) - 0
        let scale = scalar(2.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5 * (2.0f32 / std::f32::consts::PI).ln() - 2.0f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_entropy() {
        // entropy = 0.5*ln(pi/2) + ln(scale) + 0.5
        // For scale=1: 0.5*ln(pi/2) + 0.5
        let scale = scalar(1.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 0.5 * (std::f32::consts::PI / 2.0).ln() + 0.5;
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_entropy_scale2() {
        let scale = scalar(2.0f32).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 0.5 * (std::f32::consts::PI / 2.0).ln() + 2.0f32.ln() + 0.5;
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_rsample_backward() {
        let scale = scalar(2.0f32).unwrap().requires_grad_(true);
        let dist = HalfNormal::new(scale.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        let scale_grad = scale.grad().unwrap().unwrap();
        assert!(scale_grad.item().unwrap().is_finite());
        // d(sum(scale*|eps|))/d(scale) = sum(|eps|) > 0
        assert!(
            scale_grad.item().unwrap() > 0.0,
            "expected positive scale_grad, got {}",
            scale_grad.item().unwrap()
        );
    }

    #[test]
    fn test_half_normal_f64() {
        let scale = scalar(1.0f64).unwrap();
        let dist = HalfNormal::new(scale).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(0.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 0.5 * (2.0f64 / std::f64::consts::PI).ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // mean / mode / variance (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_half_normal_mean_mode_variance() {
        let dist = HalfNormal::new(scalar(1.0f64).unwrap()).unwrap();
        // mean = sqrt(2/pi)
        assert!(
            (dist.mean().unwrap().item().unwrap() - (2.0_f64 / std::f64::consts::PI).sqrt()).abs()
                < 1e-10
        );
        // mode = 0
        assert!(dist.mode().unwrap().item().unwrap().abs() < 1e-12);
        // var = 1 - 2/pi
        assert!(
            (dist.variance().unwrap().item().unwrap() - (1.0 - 2.0 / std::f64::consts::PI)).abs()
                < 1e-10
        );
    }
}
