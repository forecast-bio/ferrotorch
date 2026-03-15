//! Normal (Gaussian) distribution.
//!
//! `Normal(loc, scale)` defines a univariate normal distribution with mean
//! `loc` and standard deviation `scale`. Supports reparameterized sampling.

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Normal (Gaussian) distribution parameterized by `loc` (mean) and `scale`
/// (standard deviation).
///
/// # Reparameterization
///
/// `rsample` uses the reparameterization trick:
/// ```text
/// z = loc + scale * eps,   eps ~ N(0, 1)
/// ```
/// Gradients flow through `loc` and `scale` via the autograd graph.
pub struct Normal<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
}

impl<T: Float> Normal<T> {
    /// Create a new Normal distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `loc` and `scale` have incompatible shapes.
    pub fn new(loc: Tensor<T>, scale: Tensor<T>) -> FerrotorchResult<Self> {
        if loc.shape() != scale.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Normal: loc shape {:?} != scale shape {:?}",
                    loc.shape(),
                    scale.shape()
                ),
            });
        }
        Ok(Self { loc, scale })
    }

    /// The mean parameter.
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The standard deviation parameter.
    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }
}

impl<T: Float> Distribution<T> for Normal<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let eps = creation::randn::<T>(shape)?;
        let loc_data = self.loc.data()?;
        let scale_data = self.scale.data()?;
        let eps_data = eps.data()?;

        let result: Vec<T> = eps_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&e, &l), &s)| l + s * e)
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let eps = creation::randn::<T>(shape)?;
        let loc_data = self.loc.data()?;
        let scale_data = self.scale.data()?;
        let eps_data = eps.data()?;

        let result: Vec<T> = eps_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&e, &l), &s)| l + s * e)
            .collect();
        let storage = TensorStorage::cpu(result);

        if (self.loc.requires_grad() || self.scale.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(NormalRsampleBackward {
                loc: self.loc.clone(),
                scale: self.scale.clone(),
                eps: eps.clone(),
            });
            Tensor::from_operation(storage, shape.to_vec(), grad_fn)
        } else {
            Tensor::from_storage(storage, shape.to_vec(), false)
        }
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log_prob = -0.5 * ((x - loc) / scale)^2 - log(scale) - 0.5 * log(2*pi)
        let loc_data = self.loc.data()?;
        let scale_data = self.scale.data()?;
        let val_data = value.data()?;

        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let half = T::from(0.5).unwrap();
        let log_2pi = two_pi.ln();

        let result: Vec<T> = val_data
            .iter()
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((&x, &loc), &scale)| {
                let z = (x - loc) / scale;
                -(half * z * z) - scale.ln() - half * log_2pi
            })
            .collect();

        if (self.loc.requires_grad() || self.scale.requires_grad() || value.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(NormalLogProbBackward {
                loc: self.loc.clone(),
                scale: self.scale.clone(),
                value: value.clone(),
            });
            Tensor::from_operation(
                TensorStorage::cpu(result),
                value.shape().to_vec(),
                grad_fn,
            )
        } else {
            Tensor::from_storage(
                TensorStorage::cpu(result),
                value.shape().to_vec(),
                false,
            )
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // entropy = 0.5 + 0.5 * log(2*pi) + log(scale)
        let scale_data = self.scale.data()?;
        let half = T::from(0.5).unwrap();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let log_2pi = two_pi.ln();

        let result: Vec<T> = scale_data
            .iter()
            .map(|&s| half + half * log_2pi + s.ln())
            .collect();

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

/// Backward for `z = loc + scale * eps`.
///
/// - d(z)/d(loc)   = 1         (sum over sample dims)
/// - d(z)/d(scale) = eps       (sum over sample dims)
#[derive(Debug)]
struct NormalRsampleBackward<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
    eps: Tensor<T>,
}

impl<T: Float> GradFn<T> for NormalRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = grad_output.data()?;
        let eps_data = self.eps.data()?;

        // grad_loc = sum(grad_output) (scalar loc broadcast to all samples)
        let grad_loc_val: T = go
            .iter()
            .copied()
            .fold(<T as num_traits::Zero>::zero(), |acc, g| acc + g);
        let grad_loc = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_loc_val]),
            self.loc.shape().to_vec(),
            false,
        )?;

        // grad_scale = sum(grad_output * eps)
        let grad_scale_val: T = go.iter().zip(eps_data.iter()).fold(
            <T as num_traits::Zero>::zero(),
            |acc, (&g, &e)| acc + g * e,
        );
        let grad_scale = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_scale_val]),
            self.scale.shape().to_vec(),
            false,
        )?;

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
        "NormalRsampleBackward"
    }
}

/// Backward for Normal log_prob.
///
/// log_prob = -0.5 * ((x - loc) / scale)^2 - log(scale) - 0.5 * log(2*pi)
///
/// - d(lp)/d(x)     = -(x - loc) / scale^2
/// - d(lp)/d(loc)   = (x - loc) / scale^2
/// - d(lp)/d(scale) = ((x - loc)^2 / scale^3) - 1/scale
#[derive(Debug)]
struct NormalLogProbBackward<T: Float> {
    loc: Tensor<T>,
    scale: Tensor<T>,
    value: Tensor<T>,
}

impl<T: Float> GradFn<T> for NormalLogProbBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = grad_output.data()?;
        let loc_data = self.loc.data()?;
        let scale_data = self.scale.data()?;
        let val_data = self.value.data()?;
        let one = <T as num_traits::One>::one();

        // d(lp)/d(loc) = (x - loc) / scale^2, then sum over sample dims
        let grad_loc_val: T = go
            .iter()
            .zip(val_data.iter())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .fold(
                <T as num_traits::Zero>::zero(),
                |acc, (((&g, &x), &l), &s)| acc + g * (x - l) / (s * s),
            );
        let grad_loc = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_loc_val]),
            self.loc.shape().to_vec(),
            false,
        )?;

        // d(lp)/d(scale) = ((x - loc)^2 / scale^3) - 1/scale, summed
        let grad_scale_val: T = go
            .iter()
            .zip(val_data.iter())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .fold(
                <T as num_traits::Zero>::zero(),
                |acc, (((&g, &x), &l), &s)| {
                    let diff = x - l;
                    acc + g * (diff * diff / (s * s * s) - one / s)
                },
            );
        let grad_scale = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_scale_val]),
            self.scale.shape().to_vec(),
            false,
        )?;

        // d(lp)/d(value) = -(x - loc) / scale^2, per-element
        let grad_value: Vec<T> = go
            .iter()
            .zip(val_data.iter())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|(((&g, &x), &l), &s)| g * (-(x - l) / (s * s)))
            .collect();
        let grad_val_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_value),
            self.value.shape().to_vec(),
            false,
        )?;

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
            if self.value.requires_grad() {
                Some(grad_val_tensor)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.loc, &self.scale, &self.value]
    }

    fn name(&self) -> &'static str {
        "NormalLogProbBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_normal_sample_shape() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Normal::new(loc, scale).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_normal_sample_2d_shape() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Normal::new(loc, scale).unwrap();

        let samples = dist.sample(&[10, 20]).unwrap();
        assert_eq!(samples.shape(), &[10, 20]);
    }

    #[test]
    fn test_normal_rsample_has_grad() {
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Normal::new(loc, scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert_eq!(samples.shape(), &[5]);
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_normal_rsample_no_grad_when_params_detached() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Normal::new(loc, scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_normal_log_prob_standard() {
        // log_prob of x=0 under N(0,1) = -0.5 * log(2*pi) ~ -0.9189
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Normal::new(loc, scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -0.5 * (2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_normal_log_prob_nonzero_mean() {
        // log_prob of x=2 under N(2, 1) = -0.5 * log(2*pi)
        let loc = scalar(2.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Normal::new(loc, scale).unwrap();

        let x = scalar(2.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -0.5 * (2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_normal_log_prob_batch() {
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Normal::new(loc, scale).unwrap();

        let x = from_slice(&[-1.0f32, 0.0, 1.0], &[3]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[3]);

        let data = lp.data().unwrap();
        // lp at x=0 should be the maximum
        assert!(data[1] > data[0]);
        assert!(data[1] > data[2]);
        // Symmetry: lp(-1) == lp(1)
        assert!((data[0] - data[2]).abs() < 1e-5);
    }

    #[test]
    fn test_normal_entropy() {
        // entropy of N(0, sigma) = 0.5 * ln(2*pi*e*sigma^2)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(2.0f32).unwrap();
        let dist = Normal::new(loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 0.5 * (2.0f32 * std::f32::consts::PI * std::f32::consts::E * 4.0).ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_normal_entropy_unit_variance() {
        // entropy of N(0, 1) = 0.5 * ln(2*pi*e)
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = Normal::new(loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 0.5 * (2.0f32 * std::f32::consts::PI * std::f32::consts::E).ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_normal_shape_mismatch() {
        let loc = scalar(0.0f32).unwrap();
        let scale = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(Normal::new(loc, scale).is_err());
    }

    #[test]
    fn test_normal_rsample_backward() {
        let loc = scalar(1.0f32).unwrap().requires_grad_(true);
        let scale = scalar(2.0f32).unwrap().requires_grad_(true);
        let dist = Normal::new(loc.clone(), scale.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        // Sum the samples to get a scalar for backward.
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        // d(sum(loc + scale*eps))/d(loc) = n = 10
        let loc_grad = loc.grad().unwrap().unwrap();
        assert!(
            (loc_grad.item().unwrap() - 10.0).abs() < 1e-4,
            "expected loc_grad=10.0, got {}",
            loc_grad.item().unwrap()
        );

        // d(sum(loc + scale*eps))/d(scale) = sum(eps)
        let scale_grad = scale.grad().unwrap().unwrap();
        assert!(scale_grad.item().unwrap().is_finite());
    }

    #[test]
    fn test_normal_f64() {
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = Normal::new(loc, scale).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(0.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -0.5 * (2.0f64 * std::f64::consts::PI).ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-10);
    }
}
