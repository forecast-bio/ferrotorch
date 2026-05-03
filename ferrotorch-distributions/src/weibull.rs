//! Weibull distribution.
//!
//! `Weibull(scale, concentration)` — a two-parameter continuous distribution
//! commonly used in reliability engineering and survival analysis.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Weibull distribution parameterized by `scale` (lambda) and
/// `concentration` (k, also called shape parameter).
///
/// PDF: `f(x) = (k/lambda) * (x/lambda)^(k-1) * exp(-(x/lambda)^k)` for x >= 0.
///
/// Sampling uses inverse CDF: `x = scale * (-log(1 - u))^(1/concentration)`.
pub struct Weibull<T: Float> {
    scale: Tensor<T>,
    concentration: Tensor<T>,
}

impl<T: Float> Weibull<T> {
    pub fn new(scale: Tensor<T>, concentration: Tensor<T>) -> FerrotorchResult<Self> {
        if scale.shape() != concentration.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Weibull: scale shape {:?} != concentration shape {:?}",
                    scale.shape(),
                    concentration.shape()
                ),
            });
        }
        Ok(Self {
            scale,
            concentration,
        })
    }

    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }
    pub fn concentration(&self) -> &Tensor<T> {
        &self.concentration
    }
}

impl<T: Float> Distribution<T> for Weibull<T> {
    #[allow(clippy::needless_range_loop)]
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::sample",
        )?;
        let u = creation::rand::<T>(shape)?;
        let u_data = u.data()?;
        let s_data = self.scale.data()?;
        let k_data = self.concentration.data()?;
        let numel = u_data.len();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s_data.len() == 1 {
                0
            } else {
                i % s_data.len()
            };
            let ki = if k_data.len() == 1 {
                0
            } else {
                i % k_data.len()
            };
            // x = scale * (-log(1-u))^(1/k)
            let log_term = (one - u_data[i]).max(T::from(1e-30).unwrap()).ln();
            let val =
                s_data[si] * (<T as num_traits::Zero>::zero() - log_term).powf(one / k_data[ki]);
            out.push(val);
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Weibull: rsample not yet implemented (requires inverse CDF backward)".into(),
        })
    }

    #[allow(clippy::needless_range_loop)]
    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration, value],
            "Weibull::log_prob",
        )?;
        let v = value.data()?;
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let numel = v.len();
        let zero = <T as num_traits::Zero>::zero();

        let mut out = Vec::with_capacity(numel);
        for i in 0..numel {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ki = if k.len() == 1 { 0 } else { i % k.len() };
            if v[i] < zero {
                out.push(T::neg_infinity());
            } else {
                // log_prob = log(k/lambda) + (k-1)*log(x/lambda) - (x/lambda)^k
                let x_over_l = v[i] / s[si];
                let lp = (k[ki] / s[si]).ln()
                    + (k[ki] - <T as num_traits::One>::one()) * x_over_l.ln()
                    - x_over_l.powf(k[ki]);
                out.push(lp);
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::entropy",
        )?;
        // H = euler_gamma * (1 - 1/k) + log(lambda/k) + 1
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let euler = T::from(0.5772156649015329).unwrap();
        let one = <T as num_traits::One>::one();

        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            let h = euler * (one - one / k[i]) + (s[i] / k[i]).ln() + one;
            out.push(h);
        }

        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    fn cdf(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration, value],
            "Weibull::cdf",
        )?;
        // F(x; lambda, k) = 1 - exp(-(x/lambda)^k) for x >= 0
        let v = value.data()?;
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(v.len());
        for (i, &vi) in v.iter().enumerate() {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ki = if k.len() == 1 { 0 } else { i % k.len() };
            if vi < zero {
                out.push(zero);
            } else {
                let x_over_l = vi / s[si];
                out.push(one - (zero - x_over_l.powf(k[ki])).exp());
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), value.shape().to_vec(), false)
    }

    fn icdf(&self, q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration, q],
            "Weibull::icdf",
        )?;
        // F^{-1}(p) = lambda * (-log(1 - p))^(1/k)
        let p = q.data()?;
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(p.len());
        for (i, &pi) in p.iter().enumerate() {
            let si = if s.len() == 1 { 0 } else { i % s.len() };
            let ki = if k.len() == 1 { 0 } else { i % k.len() };
            let log_term = (one - pi).max(T::from(1e-30).unwrap()).ln();
            out.push(s[si] * (zero - log_term).powf(one / k[ki]));
        }
        Tensor::from_storage(TensorStorage::cpu(out), q.shape().to_vec(), false)
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::mean",
        )?;
        // E[X] = lambda * Gamma(1 + 1/k)
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let one = <T as num_traits::One>::one();
        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            // Gamma(1 + 1/k) via exp(lgamma(...)) — supports fractional k.
            let lg = lgamma_scalar(one + one / k[i]);
            out.push(s[i] * lg.exp());
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::mode",
        )?;
        // mode = lambda * ((k-1)/k)^(1/k) for k > 1, else 0.
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            if k[i] > one {
                out.push(s[i] * ((k[i] - one) / k[i]).powf(one / k[i]));
            } else {
                out.push(zero);
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale, &self.concentration],
            "Weibull::variance",
        )?;
        // Var[X] = lambda^2 * (Gamma(1 + 2/k) - Gamma(1 + 1/k)^2)
        let s = self.scale.data()?;
        let k = self.concentration.data()?;
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        let mut out = Vec::with_capacity(s.len());
        for i in 0..s.len() {
            let g1 = lgamma_scalar(one + one / k[i]).exp();
            let g2 = lgamma_scalar(one + two / k[i]).exp();
            out.push(s[i] * s[i] * (g2 - g1 * g1));
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.scale.shape().to_vec(), false)
    }
}

/// Scalar lgamma — Lanczos approximation. Mirrors the impl in
/// `ferrotorch_core::special::lgamma_scalar` but kept inline here so this
/// crate doesn't need an extra dependency hop just for property closures.
fn lgamma_scalar<T: Float>(x: T) -> T {
    // Use ferrotorch_core's special-fn surface via a tiny Tensor wrapper.
    // This is the simplest path that stays correct and avoids reimplementing
    // Lanczos here.
    let t = Tensor::from_storage(TensorStorage::cpu(vec![x]), vec![1], false)
        .expect("lgamma_scalar: scalar tensor build");
    let r = ferrotorch_core::special::lgamma(&t).expect("lgamma_scalar: lgamma op");
    r.data().expect("lgamma_scalar: lgamma data")[0]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scalar(v: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![v]), vec![1], false).unwrap()
    }

    #[test]
    fn test_weibull_sample_shape() {
        let d = Weibull::new(scalar(1.0), scalar(1.5)).unwrap();
        let s = d.sample(&[100]).unwrap();
        assert_eq!(s.shape(), &[100]);
        // All samples should be non-negative.
        for &v in s.data().unwrap() {
            assert!(v >= 0.0, "Weibull sample should be >= 0, got {v}");
        }
    }

    #[test]
    fn test_weibull_log_prob_negative() {
        let d = Weibull::new(scalar(1.0), scalar(2.0)).unwrap();
        let v = Tensor::from_storage(TensorStorage::cpu(vec![-1.0]), vec![1], false).unwrap();
        let lp = d.log_prob(&v).unwrap();
        assert!(lp.data().unwrap()[0].is_infinite() && lp.data().unwrap()[0] < 0.0);
    }

    #[test]
    fn test_weibull_entropy() {
        let d = Weibull::new(scalar(1.0), scalar(1.0)).unwrap();
        let h = d.entropy().unwrap();
        // For k=1, lambda=1: H = euler*(1-1/1) + ln(1/1) + 1 = 0 + 0 + 1 = 1.0
        assert!((h.data().unwrap()[0] - 1.0).abs() < 0.01);
    }

    // ---- properties (#608) ----

    #[test]
    fn test_weibull_cdf_at_scale_is_one_minus_e_inv() {
        // F(lambda; lambda, k) = 1 - exp(-1) for any k.
        let d = Weibull::new(scalar(2.0), scalar(3.0)).unwrap();
        let v = scalar(2.0);
        let c = d.cdf(&v).unwrap();
        let expected = 1.0 - (-1.0_f64).exp();
        assert!((c.data().unwrap()[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn test_weibull_cdf_icdf_roundtrip() {
        let d = Weibull::new(scalar(1.5), scalar(2.5)).unwrap();
        for p in [0.1, 0.3, 0.7, 0.9] {
            let q = scalar(p);
            let x = d.icdf(&q).unwrap();
            let p2 = d.cdf(&x).unwrap();
            assert!((p2.data().unwrap()[0] - p).abs() < 1e-6, "p={p}");
        }
    }

    #[test]
    fn test_weibull_mean_k_one_equals_lambda() {
        // For k=1, mean = lambda * Gamma(2) = lambda * 1 = lambda.
        let d = Weibull::new(scalar(3.5), scalar(1.0)).unwrap();
        let m = d.mean().unwrap();
        assert!((m.data().unwrap()[0] - 3.5).abs() < 1e-9);
    }

    #[test]
    fn test_weibull_mode_k_below_one_is_zero() {
        // For k <= 1, mode = 0.
        let d = Weibull::new(scalar(2.0), scalar(0.7)).unwrap();
        let m = d.mode().unwrap();
        assert!(m.data().unwrap()[0].abs() < 1e-12);
    }

    #[test]
    fn test_weibull_variance_k_one_equals_lambda_sq() {
        // For k=1, Var = lambda^2 * (Gamma(3) - Gamma(2)^2) = lambda^2 * (2 - 1) = lambda^2.
        let d = Weibull::new(scalar(2.0), scalar(1.0)).unwrap();
        let v = d.variance().unwrap();
        assert!((v.data().unwrap()[0] - 4.0).abs() < 1e-9);
    }
}
