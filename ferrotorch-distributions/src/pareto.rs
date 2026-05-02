//! Pareto (Type I) distribution.
//!
//! `Pareto(scale, alpha)` — a heavy-tailed power-law distribution.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Pareto Type I distribution parameterized by `scale` (x_m, minimum value)
/// and `alpha` (shape/tail index).
///
/// PDF: `f(x) = alpha * scale^alpha / x^(alpha+1)` for `x >= scale`.
///
/// Sampling: `x = scale / u^(1/alpha)` where `u ~ Uniform(0,1)`.
pub struct Pareto<T: Float> {
    scale: Tensor<T>,
    alpha: Tensor<T>,
}

impl<T: Float> Pareto<T> {
    pub fn new(scale: Tensor<T>, alpha: Tensor<T>) -> FerrotorchResult<Self> {
        if scale.shape() != alpha.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Pareto: scale shape {:?} != alpha shape {:?}",
                    scale.shape(),
                    alpha.shape()
                ),
            });
        }
        Ok(Self { scale, alpha })
    }

    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }
    pub fn alpha(&self) -> &Tensor<T> {
        &self.alpha
    }
}

impl<T: Float> Distribution<T> for Pareto<T> {
    #[allow(clippy::needless_range_loop)]
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data()?;
        let s_data = self.scale.data()?;
        let a_data = self.alpha.data()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s_data.len() == 1 {
                0
            } else {
                i % s_data.len()
            };
            let ai = if a_data.len() == 1 {
                0
            } else {
                i % a_data.len()
            };
            // x = scale / u^(1/alpha)
            let val = s_data[si]
                / u_data[i]
                    .max(T::from(1e-30).unwrap())
                    .powf(one / a_data[ai]);
            out.push(val);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Pareto: rsample not yet implemented".into(),
        })
    }

    #[allow(clippy::needless_range_loop)]
    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let v = value.data()?;
        let s = self.scale.data()?;
        let a = self.alpha.data()?;
        let numel = v.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ai = if a.len() == 1 { 0 } else { i % a.len() };
            if v[i] < s[si] {
                out.push(T::neg_infinity());
            } else {
                // log_prob = log(alpha) + alpha*log(scale) - (alpha+1)*log(x)
                let lp = a[ai].ln() + a[ai] * s[si].ln() - (a[ai] + one) * v[i].ln();
                out.push(lp);
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // H = log(scale/alpha) + 1 + 1/alpha
        let s = self.scale.data()?;
        let a = self.alpha.data()?;
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            out.push((s[i] / a[i]).ln() + one + one / a[i]);
        }

        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_pareto_samples_above_scale() {
        let d = Pareto::new(scalar(2.0), scalar(3.0)).unwrap();
        let s = d.sample(&[200]).unwrap();
        for &v in s.data().unwrap() {
            assert!(v >= 2.0, "Pareto sample should be >= scale, got {v}");
        }
    }

    #[test]
    fn test_pareto_log_prob_below_scale() {
        let d = Pareto::new(scalar(5.0), scalar(1.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![3.0]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap();
        assert!(lp.data().unwrap()[0].is_infinite() && lp.data().unwrap()[0] < 0.0);
    }

    #[test]
    fn test_pareto_log_prob_at_scale() {
        let d = Pareto::new(scalar(1.0), scalar(2.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![1.0]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap();
        // log_prob(1) = log(2) + 2*log(1) - 3*log(1) = log(2) ≈ 0.693
        assert!((lp.data().unwrap()[0] - 2.0f64.ln()).abs() < 1e-6);
    }
}
