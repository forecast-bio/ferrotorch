//! Laplace distribution.
//!
//! `Laplace(loc, scale)` defines a Laplace (double exponential) distribution
//! parameterized by location `loc` and scale `scale`.
//! Supports reparameterized sampling via inverse CDF.
//!
//! [CL-329]

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Laplace distribution parameterized by `loc` (mean) and `scale`.
///
/// # Reparameterization
///
/// `rsample` uses the inverse CDF:
/// ```text
/// u ~ Uniform(-1, 1)
/// z = loc - scale * sign(u) * log(1 - |u|)
/// ```
/// Gradients flow through `loc` and `scale` via the autograd graph.
pub struct Laplace<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
}

impl<T: Float> Laplace<T> {
    /// Create a new Laplace distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `loc` and `scale` have incompatible shapes.
    pub fn new(loc: Tensor<T>, scale: Tensor<T>) -> FerrotorchResult<Self> {
        if loc.shape() != scale.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Laplace: loc shape {:?} != scale shape {:?}",
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
}

/// Convert Uniform(0,1) to Uniform(-1, 1) range suitable for Laplace sampling,
/// then compute the inverse CDF sample.
fn laplace_icdf_sample<T: Float>(u01: T, loc: T, scale: T) -> T {
    let one = <T as num_traits::One>::one();
    let two = T::from(2.0).unwrap();
    let eps = T::from(1e-7).unwrap();

    // Map [0, 1) to (-1, 1): u = 2*u01 - 1
    let u = two * u01 - one;
    let u_abs = u.abs().min(one - eps); // clamp to avoid log(0)

    // sign(u) * (-log(1 - |u|))
    let sign = if u >= <T as num_traits::Zero>::zero() {
        one
    } else {
        -one
    };
    loc - scale * sign * (one - u_abs).ln()
}

impl<T: Float> Distribution<T> for Laplace<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "Laplace::sample")?;
        let device = self.loc.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&u_val, &l), &s)| laplace_icdf_sample(u_val, l, s))
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale], "Laplace::rsample")?;
        let device = self.loc.device();
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&u_val, &l), &s)| laplace_icdf_sample(u_val, l, s))
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if (self.loc.requires_grad() || self.scale.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(LaplaceRsampleBackward {
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
            "Laplace::log_prob",
        )?;
        // log_prob = -log(2 * scale) - |x - loc| / scale
        let device = self.loc.device();
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let val_data = value.data_vec()?;
        let two = T::from(2.0).unwrap();

        let result: Vec<T> = val_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&x, &l), &s)| -(two * s).ln() - (x - l).abs() / s)
            .collect();

        let out = Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "Laplace::entropy")?;
        // entropy = 1 + log(2 * scale)
        let device = self.scale.device();
        let scale_data = self.scale.data_vec()?;
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();

        let result: Vec<T> = scale_data.iter().map(|&s| one + (two * s).ln()).collect();

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
            "Laplace::cdf",
        )?;
        // cdf(x) = 0.5 + 0.5 * sign(x - loc) * (1 - exp(-|x - loc| / scale))
        let val = value.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let result: Vec<T> = val
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&x, &l), &s)| {
                let z = x - l;
                let sign = if z > zero {
                    one
                } else if z < zero {
                    -one
                } else {
                    zero
                };
                half + half * sign * (one - (-z.abs() / s).exp())
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.loc, &self.scale, q], "Laplace::icdf")?;
        // icdf(p) = loc - scale * sign(p - 0.5) * ln(1 - 2|p - 0.5|)
        let q_data = q.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        let zero = <T as num_traits::Zero>::zero();
        let result: Vec<T> = q_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&p, &l), &s)| {
                let d = p - half;
                let sign = if d > zero {
                    one
                } else if d < zero {
                    -one
                } else {
                    zero
                };
                l - s * sign * (one - two * d.abs()).ln()
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        Ok(self.loc.clone())
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        Ok(self.loc.clone())
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(&[&self.scale], "Laplace::variance")?;
        // 2 * scale^2
        let scale_data = self.scale.data_vec()?;
        let two = T::from(2.0).unwrap();
        let result: Vec<T> = scale_data.iter().map(|&s| two * s * s).collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.scale.shape().to_vec(),
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for Laplace rsample.
///
/// z = loc - scale * sign(u) * log(1 - |u|)
/// where u = 2*u01 - 1
///
/// d(z)/d(loc) = 1
/// d(z)/d(scale) = -sign(u) * log(1 - |u|) = (z - loc) / scale
#[derive(Debug)]
struct LaplaceRsampleBackward<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for LaplaceRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        let eps = T::from(1e-7).unwrap();

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

        // grad_scale = sum(grad_output * (-sign(u) * log(1 - |u|)))
        let grad_scale_val: T = go.iter().zip(u_data.iter()).fold(zero, |acc, (&g, &u01)| {
            let u = two * u01 - one;
            let sign = if u >= zero { one } else { -one };
            let u_abs = u.abs().min(one - eps);
            acc + g * (-sign) * (one - u_abs).ln()
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
        "LaplaceRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_laplace_sample_shape() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Laplace::new(loc, scale).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_laplace_sample_mean() {
        // E[X] = loc = 3.0
        let loc = scalar(3.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Laplace::new(loc, scale).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 3.0).abs() < 0.15, "expected mean ~3.0, got {mean}");
    }

    #[test]
    fn test_laplace_rsample_has_grad() {
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Laplace::new(loc, scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_laplace_log_prob_at_loc() {
        // log_prob(loc) = -log(2 * scale)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Laplace::new(loc, scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(2.0f32).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_laplace_log_prob_symmetry() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Laplace::new(loc, scale).unwrap();

        let x = from_slice(&[-1.0, 1.0], &[2]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let data = lp.data().unwrap();
        assert!(
            (data[0] - data[1]).abs() < 1e-5,
            "Laplace log_prob should be symmetric"
        );
    }

    #[test]
    fn test_laplace_entropy() {
        // entropy = 1 + log(2 * scale)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(2.0f32).unwrap();
        let dist = Laplace::new(loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 1.0 + (4.0f32).ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_laplace_entropy_unit() {
        // Laplace(0, 1): entropy = 1 + log(2)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Laplace::new(loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 1.0 + 2.0f32.ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_laplace_shape_mismatch() {
        let loc = scalar(0.0f32).unwrap();
        let scale = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(Laplace::new(loc, scale).is_err());
    }

    #[test]
    fn test_laplace_rsample_backward() {
        let loc = scalar(1.0f32).unwrap().requires_grad_(true);
        let scale = scalar(2.0f32).unwrap().requires_grad_(true);
        let dist = Laplace::new(loc.clone(), scale.clone()).unwrap();

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
    fn test_laplace_f64() {
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = Laplace::new(loc, scale).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(0.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(2.0f64).ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // CDF / ICDF / mean / mode / variance (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_laplace_mean_mode_variance() {
        let dist = Laplace::new(scalar(3.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        assert!((dist.mean().unwrap().item().unwrap() - 3.0).abs() < 1e-10);
        assert!((dist.mode().unwrap().item().unwrap() - 3.0).abs() < 1e-10);
        // var = 2 * scale^2 = 8
        assert!((dist.variance().unwrap().item().unwrap() - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_cdf_at_loc_is_half() {
        let dist = Laplace::new(scalar(1.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let x = scalar(1.0f64).unwrap();
        let c = dist.cdf(&x).unwrap();
        assert!((c.item().unwrap() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_laplace_icdf_roundtrip() {
        let dist = Laplace::new(scalar(0.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        for p in [0.1, 0.3, 0.5, 0.7, 0.9] {
            let q = scalar(p).unwrap();
            let x = dist.icdf(&q).unwrap();
            let p2 = dist.cdf(&x).unwrap();
            assert!((p2.item().unwrap() - p).abs() < 1e-10);
        }
    }
}
