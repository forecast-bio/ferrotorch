//! Beta distribution.
//!
//! `Beta(concentration1, concentration0)` defines a Beta distribution on [0, 1]
//! parameterized by two positive concentration parameters (alpha, beta).
//! Supports reparameterized sampling via the Gamma ratio trick:
//! Beta(a, b) = Gamma(a, 1) / (Gamma(a, 1) + Gamma(b, 1)).
//!
//! [CL-329]

use std::sync::Arc;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;
use crate::special_fns::{digamma_scalar, lgamma_scalar};

/// Beta distribution parameterized by `concentration1` (alpha) and
/// `concentration0` (beta).
///
/// # Reparameterization
///
/// `rsample` draws two independent Gamma samples and computes their ratio:
/// ```text
/// x ~ Gamma(alpha, 1)
/// y ~ Gamma(beta, 1)
/// sample = x / (x + y)
/// ```
pub struct Beta<T: Float> {
    concentration1: Tensor<T>,
    concentration0: Tensor<T>,
}

impl<T: Float> Beta<T> {
    /// Create a new Beta distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `concentration1` and `concentration0` have incompatible shapes.
    pub fn new(concentration1: Tensor<T>, concentration0: Tensor<T>) -> FerrotorchResult<Self> {
        if concentration1.shape() != concentration0.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Beta: concentration1 shape {:?} != concentration0 shape {:?}",
                    concentration1.shape(),
                    concentration0.shape()
                ),
            });
        }
        Ok(Self {
            concentration1,
            concentration0,
        })
    }

    /// The first concentration parameter (alpha).
    pub fn concentration1(&self) -> &Tensor<T> {
        &self.concentration1
    }

    /// The second concentration parameter (beta).
    pub fn concentration0(&self) -> &Tensor<T> {
        &self.concentration0
    }
}

impl<T: Float> Distribution<T> for Beta<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration1, &self.concentration0],
            "Beta::sample",
        )?;
        let device = self.concentration1.device();
        let ones = ferrotorch_core::creation::scalar(<T as num_traits::One>::one())?;
        let gamma_a = crate::Gamma::new(self.concentration1.clone(), ones.clone())?;
        let gamma_b = crate::Gamma::new(self.concentration0.clone(), ones)?;

        let xa = gamma_a.sample(shape)?;
        let xb = gamma_b.sample(shape)?;

        let a_data = xa.data_vec()?;
        let b_data = xb.data_vec()?;

        let result: Vec<T> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&a, &b)| {
                let sum = a + b;
                if sum == <T as num_traits::Zero>::zero() {
                    T::from(0.5).unwrap()
                } else {
                    a / sum
                }
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
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration1, &self.concentration0],
            "Beta::rsample",
        )?;
        let device = self.concentration1.device();
        let ones = ferrotorch_core::creation::scalar(<T as num_traits::One>::one())?;
        let gamma_a = crate::Gamma::new(self.concentration1.clone(), ones.clone())?;
        let gamma_b = crate::Gamma::new(self.concentration0.clone(), ones)?;

        // Use rsample from Gamma to get gradient flow
        let xa = gamma_a.rsample(shape)?;
        let xb = gamma_b.rsample(shape)?;

        let a_data = xa.data_vec()?;
        let b_data = xb.data_vec()?;

        let tiny = T::from(1e-30).unwrap();
        let result: Vec<T> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&a, &b)| {
                let sum = a + b;
                let clamped = if sum < tiny { tiny } else { sum };
                a / clamped
            })
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if (self.concentration1.requires_grad() || self.concentration0.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(BetaRsampleBackward {
                concentration1: self.concentration1.clone(),
                concentration0: self.concentration0.clone(),
                gamma_a: xa,
                gamma_b: xb,
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
            &[&self.concentration1, &self.concentration0, value],
            "Beta::log_prob",
        )?;
        // log_prob = (a-1)*log(x) + (b-1)*log(1-x) - lbeta(a, b)
        // where lbeta(a, b) = lgamma(a) + lgamma(b) - lgamma(a+b)
        let device = self.concentration1.device();
        let a_data = self.concentration1.data_vec()?;
        let b_data = self.concentration0.data_vec()?;
        let val_data = value.data_vec()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = val_data
            .iter()
            .zip(a_data.iter().cycle())
            .zip(b_data.iter().cycle())
            .map(|((&x, &a), &b)| {
                let lbeta = lgamma_scalar(a) + lgamma_scalar(b) - lgamma_scalar(a + b);
                (a - one) * x.ln() + (b - one) * (one - x).ln() - lbeta
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
            &[&self.concentration1, &self.concentration0],
            "Beta::entropy",
        )?;
        // entropy = lbeta(a, b) - (a-1)*digamma(a) - (b-1)*digamma(b)
        //         + (a+b-2)*digamma(a+b)
        let device = self.concentration1.device();
        let a_data = self.concentration1.data_vec()?;
        let b_data = self.concentration0.data_vec()?;
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();

        let result: Vec<T> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&a, &b)| {
                let lbeta = lgamma_scalar(a) + lgamma_scalar(b) - lgamma_scalar(a + b);
                lbeta - (a - one) * digamma_scalar(a) - (b - one) * digamma_scalar(b)
                    + (a + b - two) * digamma_scalar(a + b)
            })
            .collect();

        let out = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.concentration1.shape().to_vec(),
            false,
        )?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration1, &self.concentration0],
            "Beta::mean",
        )?;
        // mean = c1 / (c1 + c0)
        let a = self.concentration1.data_vec()?;
        let b = self.concentration0.data_vec()?;
        let result: Vec<T> = a.iter().zip(b.iter()).map(|(&x, &y)| x / (x + y)).collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.concentration1.shape().to_vec(),
            false,
        )
    }

    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration1, &self.concentration0],
            "Beta::mode",
        )?;
        // Mode = (c1 - 1) / (c1 + c0 - 2) when c1, c0 > 1; NaN otherwise.
        let a = self.concentration1.data_vec()?;
        let b = self.concentration0.data_vec()?;
        let one = <T as num_traits::One>::one();
        let two = T::from(2.0).unwrap();
        let nan = T::from(f64::NAN).unwrap();
        let result: Vec<T> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                if x > one && y > one {
                    (x - one) / (x + y - two)
                } else {
                    nan
                }
            })
            .collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.concentration1.shape().to_vec(),
            false,
        )
    }

    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.concentration1, &self.concentration0],
            "Beta::variance",
        )?;
        // var = c1 * c0 / ((c1 + c0)^2 * (c1 + c0 + 1))
        let a = self.concentration1.data_vec()?;
        let b = self.concentration0.data_vec()?;
        let one = <T as num_traits::One>::one();
        let result: Vec<T> = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let s = x + y;
                (x * y) / (s * s * (s + one))
            })
            .collect();
        Tensor::from_storage(
            TensorStorage::cpu(result),
            self.concentration1.shape().to_vec(),
            false,
        )
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for Beta rsample via Gamma ratio.
///
/// output = gamma_a / (gamma_a + gamma_b)
/// Gradients propagate through the gamma samples.
#[derive(Debug)]
struct BetaRsampleBackward<T: Float> {
    concentration1: Tensor<T>,
    concentration0: Tensor<T>,
    gamma_a: Tensor<T>,
    gamma_b: Tensor<T>,
}

impl<T: Float> GradFn<T> for BetaRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let ga_data = self.gamma_a.data_vec()?;
        let gb_data = self.gamma_b.data_vec()?;
        let conc1_data = self.concentration1.data_vec()?;
        let conc0_data = self.concentration0.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let tiny = T::from(1e-30).unwrap();

        // output = ga / (ga + gb)
        // d(output)/d(ga) = gb / (ga + gb)^2
        // d(output)/d(gb) = -ga / (ga + gb)^2
        // d(ga)/d(alpha) ~= ga * (log(ga) - digamma(alpha))  (implicit reparam)
        // d(gb)/d(beta) ~= gb * (log(gb) - digamma(beta))

        let mut grad_conc1 = zero;
        let mut grad_conc0 = zero;

        for i in 0..go.len() {
            let g = go[i];
            let ga = ga_data[i].max(tiny);
            let gb = gb_data[i].max(tiny);
            let alpha = conc1_data[i % conc1_data.len()];
            let beta_p = conc0_data[i % conc0_data.len()];
            let sum = ga + gb;
            let sum2 = sum * sum;

            let dout_dga = gb / sum2;
            let dout_dgb = -ga / sum2;

            let dga_dalpha = ga * (ga.ln() - digamma_scalar(alpha));
            let dgb_dbeta = gb * (gb.ln() - digamma_scalar(beta_p));

            grad_conc1 += g * dout_dga * dga_dalpha;
            grad_conc0 += g * dout_dgb * dgb_dbeta;
        }

        let grad_c1 = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_conc1]),
            self.concentration1.shape().to_vec(),
            false,
        )?;
        let grad_c1 = if device.is_cuda() {
            grad_c1.to(device)?
        } else {
            grad_c1
        };

        let grad_c0 = Tensor::from_storage(
            TensorStorage::cpu(vec![grad_conc0]),
            self.concentration0.shape().to_vec(),
            false,
        )?;
        let grad_c0 = if device.is_cuda() {
            grad_c0.to(device)?
        } else {
            grad_c0
        };

        Ok(vec![
            if self.concentration1.requires_grad() {
                Some(grad_c1)
            } else {
                None
            },
            if self.concentration0.requires_grad() {
                Some(grad_c0)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.concentration1, &self.concentration0]
    }

    fn name(&self) -> &'static str {
        "BetaRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::scalar;

    #[test]
    fn test_beta_sample_shape() {
        let a = scalar(2.0f32).unwrap();
        let b = scalar(5.0f32).unwrap();
        let dist = Beta::new(a, b).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_beta_sample_in_unit_interval() {
        let a = scalar(2.0f32).unwrap();
        let b = scalar(5.0f32).unwrap();
        let dist = Beta::new(a, b).unwrap();

        let samples = dist.sample(&[1000]).unwrap();
        let data = samples.data().unwrap();
        for &x in data {
            assert!(
                x > 0.0 && x < 1.0,
                "Beta sample should be in (0, 1), got {x}"
            );
        }
    }

    #[test]
    fn test_beta_sample_mean() {
        // E[X] = a / (a + b) = 2 / 7 ~ 0.2857
        let a = scalar(2.0f32).unwrap();
        let b = scalar(5.0f32).unwrap();
        let dist = Beta::new(a, b).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        let expected = 2.0 / 7.0;
        assert!(
            (mean - expected).abs() < 0.05,
            "expected mean ~{expected}, got {mean}"
        );
    }

    #[test]
    fn test_beta_rsample_has_grad() {
        let a = scalar(2.0f32).unwrap().requires_grad_(true);
        let b = scalar(5.0f32).unwrap().requires_grad_(true);
        let dist = Beta::new(a, b).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_beta_log_prob_symmetric() {
        // Beta(1, 1) = Uniform(0, 1): log_prob = 0 for all x in (0, 1)
        let a = scalar(1.0f32).unwrap();
        let b = scalar(1.0f32).unwrap();
        let dist = Beta::new(a, b).unwrap();

        let x = scalar(0.5f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!(
            lp.item().unwrap().abs() < 1e-4,
            "Beta(1,1) log_prob(0.5) should be ~0, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_beta_log_prob_known() {
        // Beta(2, 3): pdf(x) = 12 * x * (1-x)^2
        // log_prob(0.5) = log(12 * 0.5 * 0.25) = log(1.5)
        let a = scalar(2.0f32).unwrap();
        let b = scalar(3.0f32).unwrap();
        let dist = Beta::new(a, b).unwrap();

        let x = scalar(0.5f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = 1.5f32.ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_beta_entropy() {
        // Beta(1, 1) entropy = lbeta(1,1) = lgamma(1)+lgamma(1)-lgamma(2) = 0+0-0 = 0
        let a = scalar(1.0f32).unwrap();
        let b = scalar(1.0f32).unwrap();
        let dist = Beta::new(a, b).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            h.item().unwrap().abs() < 1e-4,
            "expected 0.0, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_beta_shape_mismatch() {
        let a = scalar(1.0f32).unwrap();
        let b = ferrotorch_core::creation::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(Beta::new(a, b).is_err());
    }

    #[test]
    fn test_beta_f64() {
        let a = scalar(2.0f64).unwrap();
        let b = scalar(3.0f64).unwrap();
        let dist = Beta::new(a, b).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);
    }

    // -----------------------------------------------------------------------
    // mean / mode / variance (#585) — no closed-form CDF
    // -----------------------------------------------------------------------

    #[test]
    fn test_beta_mean_variance_mode() {
        // Beta(2, 5): mean = 2/7, var = 10/(49*8), mode = 1/5
        let dist = Beta::new(scalar(2.0f64).unwrap(), scalar(5.0f64).unwrap()).unwrap();
        assert!((dist.mean().unwrap().item().unwrap() - 2.0 / 7.0).abs() < 1e-10);
        assert!((dist.variance().unwrap().item().unwrap() - 10.0 / (49.0 * 8.0)).abs() < 1e-10);
        assert!((dist.mode().unwrap().item().unwrap() - 1.0 / 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_beta_mode_undefined_for_alpha_le_one() {
        // alpha=1, beta=1 → mode is undefined → NaN per torch.
        let dist = Beta::new(scalar(1.0f64).unwrap(), scalar(1.0f64).unwrap()).unwrap();
        assert!(dist.mode().unwrap().item().unwrap().is_nan());
    }
}
