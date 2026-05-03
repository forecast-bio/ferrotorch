//! Cauchy distribution.
//!
//! `Cauchy(loc, scale)` defines a Cauchy (Lorentz) distribution with location
//! `loc` and scale `scale`. Supports reparameterized sampling via inverse CDF.

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Cauchy distribution parameterized by `loc` (location / median) and `scale`
/// (half-width at half-maximum).
///
/// The Cauchy distribution has no defined mean or variance (heavy tails).
///
/// # Reparameterization
///
/// `rsample` uses the inverse CDF:
/// ```text
/// u ~ Uniform(0, 1)
/// z = loc + scale * tan(pi * (u - 0.5))
/// ```
/// Gradients flow through `loc` and `scale` via the autograd graph.
pub struct Cauchy<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
}

impl<T: Float> Cauchy<T> {
    /// Create a new Cauchy distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `loc` and `scale` have incompatible shapes.
    pub fn new(loc: Tensor<T>, scale: Tensor<T>) -> FerrotorchResult<Self> {
        if loc.shape() != scale.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Cauchy: loc shape {:?} != scale shape {:?}",
                    loc.shape(),
                    scale.shape()
                ),
            });
        }
        Ok(Self { loc, scale })
    }

    /// The location parameter (median).
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The scale parameter (half-width at half-maximum).
    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }

    /// The median of the distribution (equals loc).
    pub fn median(&self) -> &Tensor<T> {
        &self.loc
    }
}

impl<T: Float> Distribution<T> for Cauchy<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "Cauchy::sample")?;
        let device = self.loc.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let pi = T::from(std::f64::consts::PI).unwrap();
        let half = T::from(0.5).unwrap();
        let eps = T::from(1e-7).unwrap();

        let result: Vec<T> = u_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&u_val, &loc), &scale)| {
                // Clamp u away from 0 and 1 to avoid tan(+-pi/2) = +-inf
                let u_clamped = u_val.max(eps).min(<T as num_traits::One>::one() - eps);
                loc + scale * (pi * (u_clamped - half)).tan()
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "Cauchy::rsample")?;
        let device = self.loc.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let pi = T::from(std::f64::consts::PI).unwrap();
        let half = T::from(0.5).unwrap();
        let eps = T::from(1e-7).unwrap();

        let result: Vec<T> = u_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&u_val, &loc), &scale)| {
                let u_clamped = u_val.max(eps).min(<T as num_traits::One>::one() - eps);
                loc + scale * (pi * (u_clamped - half)).tan()
            })
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if (self.loc.requires_grad() || self.scale.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(CauchyRsampleBackward {
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
            "Cauchy::log_prob",
        )?;
        // log_prob = -ln(pi) - ln(scale) - ln(1 + ((x - loc) / scale)^2)
        let device = self.loc.device();
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let val_data = value.data_vec()?;
        let pi = T::from(std::f64::consts::PI).unwrap();
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = val_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&x, &loc), &scale)| {
                let z = (x - loc) / scale;
                -pi.ln() - scale.ln() - (one + z * z).ln()
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "Cauchy::entropy")?;
        // entropy = ln(4 * pi * scale)
        let device = self.scale.device();
        let scale_data = self.scale.data_vec()?;
        let four_pi = T::from(4.0 * std::f64::consts::PI).unwrap();

        let result: Vec<T> = scale_data
            .iter()
            .map(|&scale| (four_pi * scale).ln())
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
            "Cauchy::cdf",
        )?;
        // cdf(x) = 1/2 + atan((x - loc) / scale) / pi
        let val = value.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        let inv_pi = T::from(1.0 / std::f64::consts::PI).unwrap();
        let result: Vec<T> = val
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&x, &l), &s)| half + inv_pi * ((x - l) / s).atan())
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale, q], "Cauchy::icdf")?;
        // icdf(p) = loc + scale * tan(pi * (p - 1/2))
        let q_data = q.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        let pi = T::from(std::f64::consts::PI).unwrap();
        let result: Vec<T> = q_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&p, &l), &s)| l + s * (pi * (p - half)).tan())
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc], "Cauchy::mean")?;
        // Mean is undefined; return NaN to match torch.
        let n: usize = self.loc.shape().iter().product();
        let nan = T::from(f64::NAN).unwrap();
        Tensor::from_storage(
            TensorStorage::cpu(vec![nan; n.max(1)]),
            self.loc.shape().to_vec(),
            false,
        )
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        Ok(self.loc.clone())
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc], "Cauchy::variance")?;
        // Variance is undefined; return +∞ to match torch.
        let n: usize = self.loc.shape().iter().product();
        let inf = T::from(f64::INFINITY).unwrap();
        Tensor::from_storage(
            TensorStorage::cpu(vec![inf; n.max(1)]),
            self.loc.shape().to_vec(),
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for Cauchy rsample: z = loc + scale * tan(pi * (u - 0.5)).
///
/// - d(z)/d(loc)   = 1
/// - d(z)/d(scale) = tan(pi * (u - 0.5)) = (z - loc) / scale
#[derive(Debug)]
struct CauchyRsampleBackward<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for CauchyRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let pi = T::from(std::f64::consts::PI).unwrap();
        let half = T::from(0.5).unwrap();
        let eps = T::from(1e-7).unwrap();
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

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

        // grad_scale = sum(grad_output * tan(pi * (u - 0.5)))
        let grad_scale_val: T = go
            .iter()
            .zip(u_data.iter())
            .fold(zero, |acc, (&g, &u_val)| {
                let u_clamped = u_val.max(eps).min(one - eps);
                let tan_val = (pi * (u_clamped - half)).tan();
                acc + g * tan_val
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
        "CauchyRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_cauchy_sample_shape() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Cauchy::new(loc, scale).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_cauchy_rsample_has_grad() {
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Cauchy::new(loc, scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_cauchy_log_prob_at_loc() {
        // Cauchy(0, 1) at x=0: log_prob = -ln(pi) - ln(1) - ln(1 + 0) = -ln(pi)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Cauchy::new(loc, scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_cauchy_log_prob_at_scale() {
        // Cauchy(0, 1) at x=1: log_prob = -ln(pi) - ln(1) - ln(1 + 1) = -ln(pi) - ln(2)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Cauchy::new(loc, scale).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(std::f32::consts::PI).ln() - 2.0f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_cauchy_log_prob_symmetry() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Cauchy::new(loc, scale).unwrap();

        let x = from_slice(&[-3.0, 3.0], &[2]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let data = lp.data().unwrap();
        assert!(
            (data[0] - data[1]).abs() < 1e-5,
            "Cauchy log_prob should be symmetric around loc"
        );
    }

    #[test]
    fn test_cauchy_log_prob_nonunit_scale() {
        // Cauchy(0, 2) at x=0: log_prob = -ln(pi) - ln(2) - ln(1)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(2.0f32).unwrap();
        let dist = Cauchy::new(loc, scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(std::f32::consts::PI).ln() - 2.0f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_cauchy_entropy() {
        // entropy = ln(4 * pi * scale)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Cauchy::new(loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = (4.0f32 * std::f32::consts::PI).ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_cauchy_entropy_scale2() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(2.0f32).unwrap();
        let dist = Cauchy::new(loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = (8.0f32 * std::f32::consts::PI).ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_cauchy_shape_mismatch() {
        let loc = scalar(0.0f32).unwrap();
        let scale = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(Cauchy::new(loc, scale).is_err());
    }

    #[test]
    fn test_cauchy_rsample_backward() {
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Cauchy::new(loc.clone(), scale.clone()).unwrap();

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
    fn test_cauchy_f64() {
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = Cauchy::new(loc, scale).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(0.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(std::f64::consts::PI).ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // CDF / ICDF / mean (NaN) / mode / variance (∞) (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_cauchy_mean_is_nan_variance_is_inf() {
        let dist = Cauchy::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        assert!(dist.mean().unwrap().item().unwrap().is_nan());
        assert!(dist.variance().unwrap().item().unwrap().is_infinite());
        assert!((dist.mode().unwrap().item().unwrap() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_cauchy_cdf_at_loc_is_half() {
        let dist = Cauchy::new(scalar(2.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        let x = scalar(2.0f64).unwrap();
        let c = dist.cdf(&x).unwrap();
        assert!((c.item().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_cauchy_icdf_roundtrip() {
        let dist = Cauchy::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = scalar(p).unwrap();
            let x = dist.icdf(&q).unwrap();
            let p2 = dist.cdf(&x).unwrap();
            assert!((p2.item().unwrap() - p).abs() < 1e-9);
        }
    }
}
