//! Gumbel distribution.
//!
//! `Gumbel(loc, scale)` defines a Gumbel (Type-I extreme value) distribution
//! with location `loc` and scale `scale`. Used in the Gumbel-Softmax trick.
//! Supports reparameterized sampling via inverse CDF.

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Gumbel (Type-I extreme value) distribution parameterized by `loc` and
/// `scale`.
///
/// # Reparameterization
///
/// `rsample` uses the inverse CDF:
/// ```text
/// u ~ Uniform(0, 1)
/// z = loc - scale * ln(-ln(u))
/// ```
/// Gradients flow through `loc` and `scale` via the autograd graph.
pub struct Gumbel<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
}

impl<T: Float> Gumbel<T> {
    /// Create a new Gumbel distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `loc` and `scale` have incompatible shapes.
    pub fn new(loc: Tensor<T>, scale: Tensor<T>) -> FerrotorchResult<Self> {
        if loc.shape() != scale.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Gumbel: loc shape {:?} != scale shape {:?}",
                    loc.shape(),
                    scale.shape()
                ),
            });
        }
        Ok(Self { loc, scale })
    }

    /// The location parameter.
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The scale parameter.
    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }

    /// The mean of the distribution: E[X] = loc + scale * euler_gamma.
    pub fn mean_value(&self) -> FerrotorchResult<Vec<T>> {
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let euler_gamma = T::from(0.577_215_664_901_532_9_f64).unwrap();
        Ok(loc_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&loc, &scale)| loc + scale * euler_gamma)
            .collect())
    }

    /// The variance of the distribution: Var[X] = (pi * scale)^2 / 6.
    pub fn variance_value(&self) -> FerrotorchResult<Vec<T>> {
        let scale_data = self.scale.data_vec()?;
        let pi = T::from(std::f64::consts::PI).unwrap();
        let six = T::from(6.0).unwrap();
        Ok(scale_data
            .iter()
            .map(|&scale| (pi * scale) * (pi * scale) / six)
            .collect())
    }
}

/// Gumbel inverse CDF sample: loc - scale * ln(-ln(u)).
fn gumbel_icdf<T: Float>(u: T, loc: T, scale: T) -> T {
    let eps = T::from(1e-20).unwrap();
    let one = <T as num_traits::One>::one();
    // Clamp u to (eps, 1-eps) to avoid log(0)
    let u_safe = u.max(eps).min(one - eps);
    loc - scale * (-u_safe.ln()).ln()
}

impl<T: Float> Distribution<T> for Gumbel<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "Gumbel::sample")?;
        let device = self.loc.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&u_val, &loc), &scale)| gumbel_icdf(u_val, loc, scale))
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "Gumbel::rsample")?;
        let device = self.loc.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&u_val, &loc), &scale)| gumbel_icdf(u_val, loc, scale))
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if (self.loc.requires_grad() || self.scale.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(GumbelRsampleBackward {
                loc: self.loc.clone(),
                scale: self.scale.clone(),
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
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.scale, value],
            "Gumbel::log_prob",
        )?;
        // log_prob = -(z + exp(-z)) - ln(scale)
        // where z = (x - loc) / scale
        let device = self.loc.device();
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let val_data = value.data_vec()?;

        let result: Vec<T> = val_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&x, &loc), &scale)| {
                let z = (x - loc) / scale;
                -(z + (-z).exp()) - scale.ln()
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "Gumbel::entropy")?;
        // entropy = 1 + ln(scale) + euler_gamma
        let device = self.scale.device();
        let scale_data = self.scale.data_vec()?;
        let one = <T as num_traits::One>::one();
        let euler_gamma = T::from(0.577_215_664_901_532_9_f64).unwrap();

        let result: Vec<T> = scale_data
            .iter()
            .map(|&scale| one + scale.ln() + euler_gamma)
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

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.scale, value],
            "Gumbel::cdf",
        )?;
        // cdf(x) = exp(-exp(-(x - loc) / scale))
        let val = value.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let result: Vec<T> = val
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&x, &l), &s)| (-(-(x - l) / s).exp()).exp())
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale, q], "Gumbel::icdf")?;
        // icdf(p) = loc - scale * ln(-ln(p))
        let q_data = q.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let result: Vec<T> = q_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&p, &l), &s)| l - s * (-p.ln()).ln())
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "Gumbel::mean")?;
        let data = self.mean_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.loc.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        Ok(self.loc.clone())
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "Gumbel::variance")?;
        let data = self.variance_value()?;
        Tensor::from_storage(TensorStorage::cpu(data), self.loc.shape().to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for Gumbel rsample: z = loc - scale * ln(-ln(u)).
///
/// - d(z)/d(loc)   = 1
/// - d(z)/d(scale) = -ln(-ln(u)) = (z - loc) / scale
#[derive(Debug)]
struct GumbelRsampleBackward<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for GumbelRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let eps = T::from(1e-20).unwrap();
        let one = <T as num_traits::One>::one();

        // grad_loc = sum(grad_output)
        let grad_loc_val: T = go.iter().copied().fold(zero, |acc, g| acc + g);
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

        // grad_scale = sum(grad_output * (-ln(-ln(u))))
        let grad_scale_val: T = go
            .iter()
            .zip(u_data.iter())
            .fold(zero, |acc, (&g, &u_val)| {
                let u_safe = u_val.max(eps).min(one - eps);
                acc + g * (-(-u_safe.ln()).ln())
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
        "GumbelRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_gumbel_sample_shape() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Gumbel::new(loc, scale).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_gumbel_rsample_has_grad() {
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Gumbel::new(loc, scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_gumbel_log_prob_at_loc() {
        // Gumbel(0, 1) at x=0: z=0, log_prob = -(0 + exp(0)) - ln(1) = -1
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Gumbel::new(loc, scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -1.0f32;
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_gumbel_log_prob_nonzero() {
        // Gumbel(0, 1) at x=1: z=1, log_prob = -(1 + exp(-1))
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Gumbel::new(loc, scale).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(1.0 + (-1.0f32).exp());
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_gumbel_log_prob_with_scale() {
        // Gumbel(0, 2) at x=0: z=0, log_prob = -(0 + 1) - ln(2)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(2.0f32).unwrap();
        let dist = Gumbel::new(loc, scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -1.0 - 2.0f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_gumbel_entropy() {
        // entropy = 1 + ln(scale) + euler_gamma
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Gumbel::new(loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        let euler_gamma = 0.577_215_7_f32;
        let expected = 1.0 + 0.0 + euler_gamma; // ln(1) = 0
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_gumbel_entropy_scale2() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(2.0f32).unwrap();
        let dist = Gumbel::new(loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        let euler_gamma = 0.577_215_7_f32;
        let expected = 1.0 + 2.0f32.ln() + euler_gamma;
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_gumbel_shape_mismatch() {
        let loc = scalar(0.0f32).unwrap();
        let scale = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(Gumbel::new(loc, scale).is_err());
    }

    #[test]
    fn test_gumbel_rsample_backward() {
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Gumbel::new(loc.clone(), scale.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        let loc_grad = loc.grad().unwrap().unwrap();
        assert!(
            (loc_grad.item().unwrap() - 10.0).abs() < 1e-4,
            "expected loc_grad=10.0, got {}",
            loc_grad.item().unwrap()
        );

        let scale_grad = scale.grad().unwrap().unwrap();
        assert!(scale_grad.item().unwrap().is_finite());
    }

    #[test]
    fn test_gumbel_f64() {
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = Gumbel::new(loc, scale).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(0.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -1.0f64;
        assert!((lp.item().unwrap() - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // CDF / ICDF / mean / mode / variance (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_gumbel_cdf_at_loc_is_one_over_e() {
        // cdf(loc) = exp(-1) ≈ 0.3679
        let dist = Gumbel::new(scalar(2.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let x = scalar(2.0f64).unwrap();
        let c = dist.cdf(&x).unwrap();
        assert!((c.item().unwrap() - (-1.0_f64).exp()).abs() < 1e-10);
    }

    #[test]
    fn test_gumbel_icdf_roundtrip() {
        let dist = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = scalar(p).unwrap();
            let x = dist.icdf(&q).unwrap();
            let p2 = dist.cdf(&x).unwrap();
            assert!((p2.item().unwrap() - p).abs() < 1e-10);
        }
    }

    #[test]
    fn test_gumbel_mean_mode_variance() {
        let dist = Gumbel::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let euler_gamma = 0.5772156649015329_f64;
        assert!((dist.mean().unwrap().item().unwrap() - euler_gamma).abs() < 1e-10);
        assert!(dist.mode().unwrap().item().unwrap().abs() < 1e-12);
        let pi2_over_6 = std::f64::consts::PI * std::f64::consts::PI / 6.0;
        assert!((dist.variance().unwrap().item().unwrap() - pi2_over_6).abs() < 1e-10);
    }
}
