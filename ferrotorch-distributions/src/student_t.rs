//! Student's t-distribution.
//!
//! `StudentT(df, loc, scale)` defines a Student's t-distribution with `df`
//! degrees of freedom, location `loc`, and scale `scale`.
//! Supports reparameterized sampling.

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;
use crate::special_fns::{digamma_scalar, lgamma_scalar};

/// Student's t-distribution parameterized by `df` (degrees of freedom),
/// `loc` (location), and `scale`.
///
/// # Reparameterization
///
/// `rsample` uses the representation:
/// ```text
/// z ~ Normal(0, 1)
/// chi2 ~ Chi2(df)  (= Gamma(df/2, 1/2))
/// t = loc + scale * z * sqrt(df / chi2)
/// ```
/// Gradients flow through `loc` and `scale` via the autograd graph.
pub struct StudentT<T: Float> {
    df: Tensor<T>,
    loc: Tensor<T>,
    scale: Tensor<T>,
}

impl<T: Float> StudentT<T> {
    /// Create a new Student's t-distribution.
    ///
    /// # Errors
    ///
    /// Returns an error if `df`, `loc`, and `scale` have incompatible shapes.
    pub fn new(df: Tensor<T>, loc: Tensor<T>, scale: Tensor<T>) -> FerrotorchResult<Self> {
        if df.shape() != loc.shape() || df.shape() != scale.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "StudentT: df shape {:?}, loc shape {:?}, scale shape {:?} must all match",
                    df.shape(),
                    loc.shape(),
                    scale.shape()
                ),
            });
        }
        Ok(Self { df, loc, scale })
    }

    /// The degrees of freedom parameter.
    pub fn df(&self) -> &Tensor<T> {
        &self.df
    }

    /// The location parameter.
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The scale parameter.
    pub fn scale(&self) -> &Tensor<T> {
        &self.scale
    }

    /// The mean of the distribution (defined for df > 1, equals loc).
    pub fn mean_value(&self) -> FerrotorchResult<Vec<T>> {
        let df_data = self.df.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let one = <T as num_traits::One>::one();
        Ok(df_data
            .iter()
            .zip(loc_data.iter())
            .map(|(&df, &loc)| if df > one { loc } else { T::nan() })
            .collect())
    }

    /// The variance of the distribution (defined for df > 2).
    /// Var[X] = scale^2 * df / (df - 2).
    pub fn variance_value(&self) -> FerrotorchResult<Vec<T>> {
        let df_data = self.df.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let two = T::from(2.0).unwrap();
        Ok(df_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&df, &scale)| {
                if df > two {
                    scale * scale * df / (df - two)
                } else {
                    T::infinity()
                }
            })
            .collect())
    }
}

/// Sample from Chi-squared(df) = Gamma(df/2, 1/2) using Marsaglia & Tsang.
/// Returns `n` samples.
fn sample_chi2<T: Float>(df_values: &[T], n: usize) -> FerrotorchResult<Vec<T>> {
    let one = <T as num_traits::One>::one();
    let zero = <T as num_traits::Zero>::zero();
    let half = T::from(0.5).unwrap();
    let third = T::from(1.0 / 3.0).unwrap();

    let batch = n.max(256);
    let mut norm_buf: Vec<T> = creation::randn::<T>(&[batch])?.data_vec()?;
    let mut unif_buf: Vec<T> = creation::rand::<T>(&[batch])?.data_vec()?;
    let mut ni = 0usize;
    let mut ui = 0usize;

    let next_normal = |ni: &mut usize, norm_buf: &mut Vec<T>| -> FerrotorchResult<T> {
        if *ni >= norm_buf.len() {
            *norm_buf = creation::randn::<T>(&[batch])?.data_vec()?;
            *ni = 0;
        }
        let val = norm_buf[*ni];
        *ni += 1;
        Ok(val)
    };

    let next_uniform = |ui: &mut usize, unif_buf: &mut Vec<T>| -> FerrotorchResult<T> {
        if *ui >= unif_buf.len() {
            *unif_buf = creation::rand::<T>(&[batch])?.data_vec()?;
            *ui = 0;
        }
        let val = unif_buf[*ui];
        *ui += 1;
        Ok(val)
    };

    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let df = df_values[i % df_values.len()];
        // Chi2(df) = Gamma(df/2, rate=1/2), scale = 2
        // So sample Gamma(df/2, 1) and multiply by 2
        let alpha = df * half;

        let (effective_alpha, needs_boost) = if alpha < one {
            (alpha + one, true)
        } else {
            (alpha, false)
        };

        let d = effective_alpha - third;
        let c = third / d.sqrt();

        let gamma_sample = loop {
            let x = next_normal(&mut ni, &mut norm_buf)?;
            let v_base = one + c * x;
            if v_base <= zero {
                continue;
            }
            let v = v_base * v_base * v_base;
            let u = next_uniform(&mut ui, &mut unif_buf)?;

            let x2 = x * x;
            if u < one - T::from(0.0331).unwrap() * x2 * x2 {
                break d * v;
            }
            if u.ln() < half * x2 + d * (one - v + v.ln()) {
                break d * v;
            }
        };

        let gamma_final = if needs_boost {
            let u = next_uniform(&mut ui, &mut unif_buf)?;
            let u_safe = u.max(T::from(1e-30).unwrap());
            gamma_sample * u_safe.powf(one / alpha)
        } else {
            gamma_sample
        };

        // Chi2 = Gamma(df/2, rate=1/2) = Gamma(df/2, 1) * 2
        let chi2_sample = gamma_final * T::from(2.0).unwrap();
        result.push(chi2_sample);
    }

    Ok(result)
}

impl<T: Float> Distribution<T> for StudentT<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.df, &self.loc, &self.scale],
            "StudentT::sample",
        )?;
        let device = self.loc.device();
        let df_data = self.df.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let n: usize = shape.iter().product();

        let z = creation::randn::<T>(shape)?;
        let z_data = z.data_vec()?;
        let chi2_samples = sample_chi2(&df_data, n)?;

        let result: Vec<T> = z_data
            .iter()
            .zip(chi2_samples.iter())
            .zip(df_data.iter().cycle())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((((&z_val, &chi2), &df), &loc), &scale)| {
                loc + scale * z_val * (df / chi2).sqrt()
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
            &[&self.df, &self.loc, &self.scale],
            "StudentT::rsample",
        )?;
        let device = self.loc.device();
        let df_data = self.df.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let n: usize = shape.iter().product();

        let z = creation::randn::<T>(shape)?;
        let z_data = z.data_vec()?;
        let chi2_samples = sample_chi2(&df_data, n)?;

        let result: Vec<T> = z_data
            .iter()
            .zip(chi2_samples.iter())
            .zip(df_data.iter().cycle())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|((((&z_val, &chi2), &df), &loc), &scale)| {
                loc + scale * z_val * (df / chi2).sqrt()
            })
            .collect();

        let storage = TensorStorage::cpu(result);

        let out = if (self.loc.requires_grad() || self.scale.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let z_tensor = z.clone();
            let chi2_tensor =
                Tensor::from_storage(TensorStorage::cpu(chi2_samples), shape.to_vec(), false)?;
            let grad_fn = Arc::new(StudentTRsampleBackward {
                df: self.df.clone(),
                loc: self.loc.clone(),
                scale: self.scale.clone(),
                z: z_tensor,
                chi2: chi2_tensor,
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
            &[&self.df, &self.loc, &self.scale, value],
            "StudentT::log_prob",
        )?;
        // log_prob = lgamma((df+1)/2) - lgamma(df/2)
        //          - 0.5 * ln(df * pi) - ln(scale)
        //          - (df+1)/2 * ln(1 + ((x - loc)/scale)^2 / df)
        let device = self.loc.device();
        let df_data = self.df.data_vec()?;
        let loc_data = self.loc.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let val_data = value.data_vec()?;
        let half = T::from(0.5).unwrap();
        let one = <T as num_traits::One>::one();
        let pi = T::from(std::f64::consts::PI).unwrap();

        let result: Vec<T> = val_data
            .iter()
            .zip(df_data.iter().cycle())
            .zip(loc_data.iter().cycle())
            .zip(scale_data.iter().cycle())
            .map(|(((&x, &df), &loc), &scale)| {
                let y = (x - loc) / scale;
                lgamma_scalar((df + one) * half)
                    - lgamma_scalar(df * half)
                    - half * (df * pi).ln()
                    - scale.ln()
                    - (df + one) * half * (one + y * y / df).ln()
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
        crate::fallback::check_gpu_fallback_opt_in(&[&self.df, &self.scale], "StudentT::entropy")?;
        // entropy = (df + 1)/2 * (digamma((df+1)/2) - digamma(df/2))
        //         + ln(sqrt(df) * B(df/2, 1/2))
        //         + ln(scale)
        // where B(a, b) = Gamma(a)*Gamma(b) / Gamma(a+b)
        // Simplifying: ln(sqrt(df) * B(df/2, 1/2))
        //   = 0.5*ln(df) + lgamma(df/2) + 0.5*ln(pi) - lgamma((df+1)/2)
        //   = 0.5*ln(df) + lgamma(df/2) + 0.5*ln(pi) - lgamma((df+1)/2)
        let device = self.df.device();
        let df_data = self.df.data_vec()?;
        let scale_data = self.scale.data_vec()?;
        let half = T::from(0.5).unwrap();
        let one = <T as num_traits::One>::one();
        let pi = T::from(std::f64::consts::PI).unwrap();

        let result: Vec<T> = df_data
            .iter()
            .zip(scale_data.iter())
            .map(|(&df, &scale)| {
                let df_plus_1_half = (df + one) * half;
                let df_half = df * half;
                df_plus_1_half * (digamma_scalar(df_plus_1_half) - digamma_scalar(df_half))
                    + half * df.ln()
                    + lgamma_scalar(df_half)
                    + half * pi.ln()
                    - lgamma_scalar(df_plus_1_half)
                    + scale.ln()
            })
            .collect();

        let out =
            Tensor::from_storage(TensorStorage::cpu(result), self.df.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for StudentT rsample.
///
/// output = loc + scale * z * sqrt(df / chi2)
///
/// - d(out)/d(loc) = 1
/// - d(out)/d(scale) = z * sqrt(df / chi2) = (out - loc) / scale
#[derive(Debug)]
struct StudentTRsampleBackward<T: Float> {
    df: Tensor<T>,
    loc: Tensor<T>,
    scale: Tensor<T>,
    z: Tensor<T>,
    chi2: Tensor<T>,
}

impl<T: Float> GradFn<T> for StudentTRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let z_data = self.z.data_vec()?;
        let chi2_data = self.chi2.data_vec()?;
        let df_data = self.df.data_vec()?;
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

        // grad_scale = sum(grad_output * z * sqrt(df / chi2))
        let grad_scale_val: T = go
            .iter()
            .zip(z_data.iter())
            .zip(chi2_data.iter())
            .zip(df_data.iter().cycle())
            .fold(zero, |acc, (((&g, &z), &chi2), &df)| {
                acc + g * z * (df / chi2).sqrt()
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
            None, // df gradient not supported (discrete-like parameter)
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
        vec![&self.df, &self.loc, &self.scale]
    }

    fn name(&self) -> &'static str {
        "StudentTRsampleBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar};

    #[test]
    fn test_student_t_sample_shape() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_student_t_sample_mean() {
        // E[X] = loc = 2.0 for df > 1
        let df = scalar(10.0f32).unwrap();
        let loc = scalar(2.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let samples = dist.sample(&[10000]).unwrap();
        let data = samples.data().unwrap();
        let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
        assert!((mean - 2.0).abs() < 0.2, "expected mean ~2.0, got {mean}");
    }

    #[test]
    fn test_student_t_rsample_has_grad() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap().requires_grad_(true);
        let scale = scalar(1.0f32).unwrap().requires_grad_(true);
        let dist = StudentT::new(df, loc, scale).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_student_t_log_prob_at_loc() {
        // StudentT(df=1, loc=0, scale=1) is the standard Cauchy distribution.
        // At x=0: log_prob = lgamma(1) - lgamma(0.5) - 0.5*ln(pi) - ln(1) - 1*ln(1)
        // = 0 - lgamma(0.5) - 0.5*ln(pi)
        // lgamma(0.5) = 0.5*ln(pi), so log_prob = -ln(pi)
        let df = scalar(1.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let x = scalar(0.0f32).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_student_t_log_prob_symmetry() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let x = from_slice(&[-2.0, 2.0], &[2]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let data = lp.data().unwrap();
        assert!(
            (data[0] - data[1]).abs() < 1e-5,
            "StudentT log_prob should be symmetric around loc"
        );
    }

    #[test]
    fn test_student_t_log_prob_high_df_approaches_normal() {
        // As df -> inf, StudentT -> Normal
        let df = scalar(10000.0f64).unwrap();
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let x = scalar(1.0f64).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        // Normal(0,1).log_prob(1) = -0.5 - 0.5*ln(2*pi)
        let expected = -0.5 - 0.5 * (2.0f64 * std::f64::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 0.01,
            "expected ~{expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_student_t_entropy_positive() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = scalar(1.0f32).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let h = dist.entropy().unwrap();
        assert!(
            h.item().unwrap() > 0.0,
            "StudentT entropy should be positive, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_student_t_shape_mismatch() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(0.0f32).unwrap();
        let scale = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        assert!(StudentT::new(df, loc, scale).is_err());
    }

    #[test]
    fn test_student_t_rsample_backward() {
        let df = scalar(5.0f32).unwrap();
        let loc = scalar(1.0f32).unwrap().requires_grad_(true);
        let scale = scalar(2.0f32).unwrap().requires_grad_(true);
        let dist = StudentT::new(df, loc.clone(), scale.clone()).unwrap();

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
    fn test_student_t_f64() {
        let df = scalar(5.0f64).unwrap();
        let loc = scalar(0.0f64).unwrap();
        let scale = scalar(1.0f64).unwrap();
        let dist = StudentT::new(df, loc, scale).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50]);
    }
}
