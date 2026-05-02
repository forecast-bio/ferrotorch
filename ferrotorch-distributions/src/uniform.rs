//! Uniform distribution.
//!
//! `Uniform(low, high)` defines a continuous uniform distribution on the
//! interval `[low, high)`. Supports reparameterized sampling.

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Continuous uniform distribution on `[low, high)`.
///
/// # Reparameterization
///
/// `rsample` uses the reparameterization trick:
/// ```text
/// z = low + (high - low) * u,   u ~ Uniform(0, 1)
/// ```
/// Gradients flow through `low` and `high` via the autograd graph.
pub struct Uniform<T: Float> {
    low: Tensor<T>,
    high: Tensor<T>,
}

impl<T: Float> Uniform<T> {
    /// Create a new Uniform distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `low` and `high` have incompatible shapes.
    pub fn new(low: Tensor<T>, high: Tensor<T>) -> FerrotorchResult<Self> {
        if low.shape() != high.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Uniform: low shape {:?} != high shape {:?}",
                    low.shape(),
                    high.shape()
                ),
            });
        }
        Ok(Self { low, high })
    }

    /// The lower bound.
    pub fn low(&self) -> &Tensor<T> {
        &self.low
    }

    /// The upper bound.
    pub fn high(&self) -> &Tensor<T> {
        &self.high
    }
}

impl<T: Float> Distribution<T> for Uniform<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let device = self.low.device();
        let u = creation::rand::<T>(shape)?;
        let low_data = self.low.data_vec()?;
        let high_data = self.high.data_vec()?;
        let u_data = u.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(low_data.iter().cycle())
            .zip(high_data.iter().cycle())
            .map(|((&u_val, &lo), &hi)| lo + (hi - lo) * u_val)
            .collect();
        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let device = self.low.device();
        let u = creation::rand::<T>(shape)?;
        let low_data = self.low.data_vec()?;
        let high_data = self.high.data_vec()?;
        let u_data = u.data_vec()?;

        let result: Vec<T> = u_data
            .iter()
            .zip(low_data.iter().cycle())
            .zip(high_data.iter().cycle())
            .map(|((&u_val, &lo), &hi)| lo + (hi - lo) * u_val)
            .collect();
        let storage = TensorStorage::cpu(result);

        let out = if (self.low.requires_grad() || self.high.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(UniformRsampleBackward {
                low: self.low.clone(),
                high: self.high.clone(),
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
        // log_prob = -log(high - low) if low <= x < high, else -inf
        let device = self.low.device();
        let low_data = self.low.data_vec()?;
        let high_data = self.high.data_vec()?;
        let val_data = value.data_vec()?;

        let neg_inf = T::neg_infinity();

        let result: Vec<T> = val_data
            .iter()
            .zip(low_data.iter().cycle())
            .zip(high_data.iter().cycle())
            .map(|((&x, &lo), &hi)| {
                if x >= lo && x < hi {
                    -((hi - lo).ln())
                } else {
                    neg_inf
                }
            })
            .collect();

        let out =
            if (self.low.requires_grad() || self.high.requires_grad() || value.requires_grad())
                && ferrotorch_core::is_grad_enabled()
            {
                let grad_fn = Arc::new(UniformLogProbBackward {
                    low: self.low.clone(),
                    high: self.high.clone(),
                    value: value.clone(),
                });
                Tensor::from_operation(TensorStorage::cpu(result), value.shape().to_vec(), grad_fn)?
            } else {
                Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?
            };
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // entropy = log(high - low)
        let device = self.low.device();
        let low_data = self.low.data_vec()?;
        let high_data = self.high.data_vec()?;

        let result: Vec<T> = low_data
            .iter()
            .zip(high_data.iter())
            .map(|(&lo, &hi)| (hi - lo).ln())
            .collect();

        let out =
            Tensor::from_storage(TensorStorage::cpu(result), self.low.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // cdf(x) = 0 if x < low; (x - low) / (high - low) if low <= x < high; 1 if x >= high.
        let val = value.data_vec()?;
        let lo = self.low.data_vec()?;
        let hi = self.high.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = val
            .iter()
            .zip(lo.iter().cycle())
            .zip(hi.iter().cycle())
            .map(|((&x, &low), &high)| {
                if x < low {
                    zero
                } else if x >= high {
                    one
                } else {
                    (x - low) / (high - low)
                }
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // icdf(p) = low + (high - low) * p (assumes p in [0, 1])
        let q_data = q.data_vec()?;
        let lo = self.low.data_vec()?;
        let hi = self.high.data_vec()?;
        let result: Vec<T> = q_data
            .iter()
            .zip(lo.iter().cycle())
            .zip(hi.iter().cycle())
            .map(|((&p, &low), &high)| low + (high - low) * p)
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        // (low + high) / 2
        let lo = self.low.data_vec()?;
        let hi = self.high.data_vec()?;
        let half = T::from(0.5).unwrap();
        let result: Vec<T> = lo
            .iter()
            .zip(hi.iter())
            .map(|(&l, &h)| half * (l + h))
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), self.low.shape().to_vec(), false)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        // (high - low)^2 / 12
        let lo = self.low.data_vec()?;
        let hi = self.high.data_vec()?;
        let twelfth = T::from(1.0 / 12.0).unwrap();
        let result: Vec<T> = lo
            .iter()
            .zip(hi.iter())
            .map(|(&l, &h)| {
                let d = h - l;
                d * d * twelfth
            })
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), self.low.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        // The Uniform PDF is constant — every point in [low, high] is a
        // mode. We return the midpoint as a representative value (matches
        // the convention in `torch.distributions.Uniform.mode`).
        self.mean()
    }

    fn stddev(&self) -> FerrotorchResult<Tensor<T>> {
        // sqrt(Var) = (high - low) / sqrt(12)
        let lo = self.low.data_vec()?;
        let hi = self.high.data_vec()?;
        let inv_sqrt_12 = T::from(1.0 / (12.0_f64).sqrt()).unwrap();
        let result: Vec<T> = lo
            .iter()
            .zip(hi.iter())
            .map(|(&l, &h)| (h - l) * inv_sqrt_12)
            .collect();
        Tensor::from_storage(TensorStorage::cpu(result), self.low.shape().to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for `z = low + (high - low) * u`.
///
/// - d(z)/d(low)  = 1 - u      (sum over sample dims)
/// - d(z)/d(high) = u          (sum over sample dims)
#[derive(Debug)]
struct UniformRsampleBackward<T: Float> {
    low: Tensor<T>,
    high: Tensor<T>,
    u: Tensor<T>,
}

impl<T: Float> GradFn<T> for UniformRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let u_data = self.u.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        // grad_low = sum(grad_output * (1 - u))
        let grad_low_val: T = go
            .iter()
            .zip(u_data.iter())
            .fold(zero, |acc, (&g, &u_val)| acc + g * (one - u_val));
        let grad_low = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_low_val]),
            self.low.shape().to_vec(),
            false,
        )?;
        let grad_low = if device.is_cuda() {
            grad_low.to(device)?
        } else {
            grad_low
        };

        // grad_high = sum(grad_output * u)
        let grad_high_val: T = go
            .iter()
            .zip(u_data.iter())
            .fold(zero, |acc, (&g, &u_val)| acc + g * u_val);
        let grad_high = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_high_val]),
            self.high.shape().to_vec(),
            false,
        )?;
        let grad_high = if device.is_cuda() {
            grad_high.to(device)?
        } else {
            grad_high
        };

        Ok(vec![
            if self.low.requires_grad() {
                Some(grad_low)
            } else {
                None
            },
            if self.high.requires_grad() {
                Some(grad_high)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.low, &self.high]
    }

    fn name(&self) -> &'static str {
        "UniformRsampleBackward"
    }
}

/// Backward for Uniform log_prob.
///
/// log_prob = -log(high - low) inside bounds, -inf outside.
///
/// - d(lp)/d(low)  = 1 / (high - low)  (summed over sample dims)
/// - d(lp)/d(high) = -1 / (high - low) (summed over sample dims)
/// - d(lp)/d(value) = 0  (flat density)
#[derive(Debug)]
struct UniformLogProbBackward<T: Float> {
    low: Tensor<T>,
    high: Tensor<T>,
    value: Tensor<T>,
}

impl<T: Float> GradFn<T> for UniformLogProbBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let low_data = self.low.data_vec()?;
        let high_data = self.high.data_vec()?;
        let val_data = self.value.data_vec()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();

        let mut grad_low_val = zero;
        let mut grad_high_val = zero;

        for (((&g, &x), &lo), &hi) in go
            .iter()
            .zip(val_data.iter())
            .zip(low_data.iter().cycle())
            .zip(high_data.iter().cycle())
        {
            if x >= lo && x < hi {
                let range = hi - lo;
                grad_low_val += g * one / range;
                grad_high_val += g * (-one / range);
            }
        }

        let grad_low = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_low_val]),
            self.low.shape().to_vec(),
            false,
        )?;
        let grad_low = if device.is_cuda() {
            grad_low.to(device)?
        } else {
            grad_low
        };
        let grad_high = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_high_val]),
            self.high.shape().to_vec(),
            false,
        )?;
        let grad_high = if device.is_cuda() {
            grad_high.to(device)?
        } else {
            grad_high
        };

        // d(lp)/d(value) = 0 inside bounds
        let grad_value = Tensor::from_storage(
            TensorStorage::cpu(vec![zero; val_data.len()]),
            self.value.shape().to_vec(),
            false,
        )?;
        let grad_value = if device.is_cuda() {
            grad_value.to(device)?
        } else {
            grad_value
        };

        Ok(vec![
            if self.low.requires_grad() {
                Some(grad_low)
            } else {
                None
            },
            if self.high.requires_grad() {
                Some(grad_high)
            } else {
                None
            },
            if self.value.requires_grad() {
                Some(grad_value)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.low, &self.high, &self.value]
    }

    fn name(&self) -> &'static str {
        "UniformLogProbBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_uniform_sample_shape() {
        let low = scalar(0.0f32).unwrap();
        let high = scalar(1.0f32).unwrap();
        let dist = Uniform::new(low, high).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_uniform_sample_in_range() {
        let low = scalar(2.0f32).unwrap();
        let high = scalar(5.0f32).unwrap();
        let dist = Uniform::new(low, high).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!((2.0..5.0).contains(&x), "sample {x} out of range [2, 5)");
        }
    }

    #[test]
    fn test_uniform_rsample_has_grad() {
        let low = scalar(0.0f32).unwrap().requires_grad_(true);
        let high = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = Uniform::new(low, high).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_uniform_rsample_no_grad_when_detached() {
        let low = scalar(0.0f32).unwrap();
        let high = scalar(1.0f32).unwrap();
        let dist = Uniform::new(low, high).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_uniform_log_prob_in_range() {
        // Uniform(0, 2): log_prob = -log(2) for x in [0, 2)
        let low = scalar(0.0f32).unwrap();
        let high = scalar(2.0f32).unwrap();
        let dist = Uniform::new(low, high).unwrap();

        let x = scalar(1.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(2.0f32.ln());
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_uniform_log_prob_out_of_range() {
        let low = scalar(0.0f32).unwrap();
        let high = scalar(1.0f32).unwrap();
        let dist = Uniform::new(low, high).unwrap();

        let x = scalar(2.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!(lp.item().unwrap().is_infinite() && lp.item().unwrap() < 0.0);
    }

    #[test]
    fn test_uniform_log_prob_batch() {
        let low = scalar(0.0f32).unwrap();
        let high = scalar(1.0f32).unwrap();
        let dist = Uniform::new(low, high).unwrap();

        let x = from_slice(&[-0.5, 0.5, 1.5], &[3]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[3]);

        let data = lp.data().unwrap();
        assert!(data[0].is_infinite()); // -0.5 out of range
        assert!((data[1] - 0.0).abs() < 1e-6); // log(1) = 0 for Uniform(0,1)
        assert!(data[2].is_infinite()); // 1.5 out of range
    }

    #[test]
    fn test_uniform_entropy() {
        // entropy of Uniform(a, b) = log(b - a)
        let low = scalar(1.0f32).unwrap();
        let high = scalar(4.0f32).unwrap();
        let dist = Uniform::new(low, high).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 3.0f32.ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-6,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_uniform_entropy_unit() {
        // entropy of Uniform(0, 1) = 0
        let low = scalar(0.0f32).unwrap();
        let high = scalar(1.0f32).unwrap();
        let dist = Uniform::new(low, high).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            h.item().unwrap().abs() < 1e-6,
            "expected 0.0, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_uniform_shape_mismatch() {
        let low = scalar(0.0f32).unwrap();
        let high = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(Uniform::new(low, high).is_err());
    }

    #[test]
    fn test_uniform_rsample_backward() {
        let low = scalar(1.0f32).unwrap().requires_grad_(true);
        let high = scalar(3.0f32).unwrap().requires_grad_(true);
        let dist = Uniform::new(low.clone(), high.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        let low_grad = low.grad().unwrap().unwrap();
        let high_grad = high.grad().unwrap().unwrap();

        // Gradients should be finite
        assert!(low_grad.item().unwrap().is_finite());
        assert!(high_grad.item().unwrap().is_finite());

        // grad_low + grad_high should sum to n (since d(low + (high-low)*u)/d(low) + d(...)/d(high) = 1)
        let total = low_grad.item().unwrap() + high_grad.item().unwrap();
        assert!(
            (total - 10.0).abs() < 1e-4,
            "expected grad sum = 10.0, got {total}"
        );
    }

    #[test]
    fn test_uniform_f64() {
        let low = scalar(0.0f64).unwrap();
        let high = scalar(1.0f64).unwrap();
        let dist = Uniform::new(low, high).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);

        let x = scalar(0.5f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!((lp.item().unwrap() - 0.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // CDF / ICDF / mean / variance / stddev (#585)
    // -----------------------------------------------------------------------

    #[test]
    fn test_uniform_mean_variance_stddev() {
        // Uniform(0, 4) → mean=2, var=16/12, stddev=sqrt(16/12)
        let dist = Uniform::new(scalar(0.0f64).unwrap(), scalar(4.0f64).unwrap()).unwrap();
        assert!((dist.mean().unwrap().item().unwrap() - 2.0).abs() < 1e-10);
        assert!((dist.variance().unwrap().item().unwrap() - 16.0 / 12.0).abs() < 1e-10);
        let std = dist.stddev().unwrap().item().unwrap();
        assert!((std - (16.0_f64 / 12.0).sqrt()).abs() < 1e-10);
    }

    #[test]
    fn test_uniform_cdf_endpoints() {
        let dist = Uniform::new(scalar(0.0f64).unwrap(), scalar(2.0f64).unwrap()).unwrap();
        let x = from_slice::<f64>(&[-1.0, 0.0, 1.0, 2.0, 3.0], &[5]).unwrap();
        let c = dist.cdf(&x).unwrap();
        let d = c.data().unwrap();
        assert_eq!(d, &[0.0, 0.0, 0.5, 1.0, 1.0]);
    }

    #[test]
    fn test_uniform_icdf_linear() {
        let dist = Uniform::new(scalar(10.0f64).unwrap(), scalar(20.0f64).unwrap()).unwrap();
        let q = from_slice::<f64>(&[0.0, 0.25, 0.5, 1.0], &[4]).unwrap();
        let x = dist.icdf(&q).unwrap();
        let d = x.data().unwrap();
        assert_eq!(d, &[10.0, 12.5, 15.0, 20.0]);
    }

    // ---- properties (#608) ----

    #[test]
    fn test_uniform_mode_is_midpoint() {
        let dist = Uniform::new(scalar(0.0f64).unwrap(), scalar(4.0f64).unwrap()).unwrap();
        let m = dist.mode().unwrap();
        assert!((m.item().unwrap() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_uniform_stddev_is_range_over_sqrt12() {
        let dist = Uniform::new(scalar(0.0f64).unwrap(), scalar(12.0f64).unwrap()).unwrap();
        let s = dist.stddev().unwrap();
        assert!((s.item().unwrap() - 12.0 / (12.0_f64).sqrt()).abs() < 1e-12);
    }
}
