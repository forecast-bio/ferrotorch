//! Relaxed one-hot categorical (Concrete) distribution.
//!
//! `RelaxedOneHotCategorical(temperature, probs)` is a continuous relaxation
//! of [`OneHotCategorical`](crate::OneHotCategorical) over the open
//! probability simplex via the Gumbel-softmax trick (Maddison et al. 2017,
//! Jang et al. 2017). Samples are points in the open `K-1` simplex, not
//! discrete one-hot vectors.
//!
//! As `temperature → 0`, samples concentrate on the corners of the simplex
//! and recover the discrete OneHotCategorical. As `temperature → ∞`,
//! samples approach the uniform distribution on the simplex.
//!
//! # Reparameterization
//!
//! Sampling is reparameterizable via Gumbel noise:
//! ```text
//! g_i ~ Gumbel(0, 1)            (i.e. g_i = -log(-log(U_i)), U_i ~ Uniform)
//! z_i = exp((log(probs_i) + g_i) / temperature)
//! z = z / sum_j z_j             (softmax over the K dimensions)
//! ```
//! Mirrors `torch.distributions.RelaxedOneHotCategorical`.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Continuous relaxation of a categorical distribution.
pub struct RelaxedOneHotCategorical<T: Float> {
    temperature: T,
    probs: Tensor<T>,
    /// Cached normalized probabilities.
    normalized: Vec<T>,
    num_categories: usize,
}

impl<T: Float> RelaxedOneHotCategorical<T> {
    /// Construct a RelaxedOneHotCategorical with the given temperature
    /// and unnormalized class probabilities.
    ///
    /// `probs` must be 1-D shape `[K]`, all entries strictly positive.
    pub fn new(temperature: T, probs: Tensor<T>) -> FerrotorchResult<Self> {
        let zero = <T as num_traits::Zero>::zero();
        if temperature <= zero {
            return Err(FerrotorchError::InvalidArgument {
                message: "RelaxedOneHotCategorical: temperature must be > 0".into(),
            });
        }
        if probs.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RelaxedOneHotCategorical: probs must be 1-D, got shape {:?}",
                    probs.shape()
                ),
            });
        }
        let k = probs.shape()[0];
        if k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "RelaxedOneHotCategorical: probs must have at least one category".into(),
            });
        }

        let probs_data = probs.data_vec()?;
        for (i, &p) in probs_data.iter().enumerate() {
            if p <= zero {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "RelaxedOneHotCategorical: probs[{i}] = {} must be > 0",
                        p.to_f64().unwrap_or(f64::NAN)
                    ),
                });
            }
        }
        let total: T = probs_data.iter().copied().fold(zero, |a, b| a + b);
        let normalized: Vec<T> = probs_data.iter().map(|&p| p / total).collect();

        Ok(Self {
            temperature,
            probs,
            normalized,
            num_categories: k,
        })
    }

    /// Temperature parameter.
    pub fn temperature(&self) -> T {
        self.temperature
    }

    /// (Unnormalized) probabilities.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// Number of categories.
    pub fn num_categories(&self) -> usize {
        self.num_categories
    }
}

impl<T: Float> Distribution<T> for RelaxedOneHotCategorical<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs],
            "RelaxedOneHotCategorical::sample",
        )?;
        relaxed_one_hot_sample(
            self.temperature,
            &self.normalized,
            self.num_categories,
            &self.probs,
            shape,
            false,
        )
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs],
            "RelaxedOneHotCategorical::rsample",
        )?;
        // See RelaxedBernoulli rsample doc for the gradient-flow caveat.
        relaxed_one_hot_sample(
            self.temperature,
            &self.normalized,
            self.num_categories,
            &self.probs,
            shape,
            true,
        )
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.probs, value],
            "RelaxedOneHotCategorical::log_prob",
        )?;
        // Concrete log density on the simplex (Maddison et al. 2017, eqn 26):
        //
        //   log p(z; alpha, lambda) = log((K-1)!) + (K-1) * log(lambda)
        //       + sum_k ( log(alpha_k) - (lambda + 1) * log(z_k) )
        //       - K * log( sum_k alpha_k * z_k^(-lambda) )
        //
        // where alpha_k = probs_k (normalized) and lambda = temperature.
        //
        // We use logs throughout for numerical stability.
        let v_shape = value.shape().to_vec();
        if v_shape.is_empty() || *v_shape.last().unwrap() != self.num_categories {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "RelaxedOneHotCategorical: log_prob value last dim must be K={}, got shape {:?}",
                    self.num_categories, v_shape
                ),
            });
        }

        let lambda = self.temperature;
        let one = <T as num_traits::One>::one();
        let zero = <T as num_traits::Zero>::zero();
        let neg_lambda = zero - lambda;
        let k = self.num_categories;
        let kt = T::from(k).unwrap();

        // log((K-1)!)
        let mut log_fact_km1 = zero;
        for i in 1..k {
            log_fact_km1 += T::from(i).unwrap().ln();
        }
        let km1 = T::from((k as i64 - 1).max(0)).unwrap();
        let log_lambda = lambda.ln();
        let log_alpha: Vec<T> = self.normalized.iter().map(|&p| p.ln()).collect();
        let constant = log_fact_km1 + km1 * log_lambda;

        let v_data = value.data_vec()?;
        let n = v_data.len() / k;
        let mut result = Vec::with_capacity(n);
        let eps = T::from(1e-20).unwrap();

        for i in 0..n {
            let base = i * k;
            // sum_k log(alpha_k) - (lambda + 1) * log(z_k)
            let mut linear = zero;
            // logsumexp over k of log(alpha_k) - lambda * log(z_k)
            // (for the - K * log(sum) term)
            let mut max_lse = T::neg_infinity();
            let mut tmp = vec![zero; k];
            for j in 0..k {
                let z = v_data[base + j].max(eps);
                let log_z = z.ln();
                linear += log_alpha[j] - (lambda + one) * log_z;
                let t = log_alpha[j] + neg_lambda * log_z;
                tmp[j] = t;
                if t > max_lse {
                    max_lse = t;
                }
            }
            let mut sum_exp = zero;
            for &t in &tmp {
                sum_exp += (t - max_lse).exp();
            }
            let lse = max_lse + sum_exp.ln();
            result.push(constant + linear - kt * lse);
        }

        let mut out_shape = v_shape;
        out_shape.pop();
        let device = self.probs.device();
        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "RelaxedOneHotCategorical: entropy has no closed form".into(),
        })
    }
}

/// Concrete forward sampling for RelaxedOneHotCategorical (shared by sample
/// and rsample). Result lies on the open K-simplex.
#[allow(clippy::needless_range_loop)]
fn relaxed_one_hot_sample<T: Float>(
    temperature: T,
    normalized: &[T],
    k: usize,
    probs: &Tensor<T>,
    shape: &[usize],
    _reparam: bool,
) -> FerrotorchResult<Tensor<T>> {
    let device = probs.device();
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let n: usize = shape.iter().product();

    // Draw N*K Uniform samples for the Gumbel noise.
    let u = creation::rand::<T>(&[n * k])?;
    let u_data = u.data_vec()?;
    let eps = T::from(1e-20).unwrap();

    let log_alpha: Vec<T> = normalized.iter().map(|&p| (p + eps).ln()).collect();

    let mut result = Vec::with_capacity(n * k);
    for i in 0..n {
        // Compute z_j = exp((log_alpha[j] + g_j) / temperature) and softmax.
        let mut logits = vec![zero; k];
        let mut max_l = T::neg_infinity();
        for j in 0..k {
            let u_val = u_data[i * k + j].max(eps).min(one - eps);
            // Gumbel(0,1) = -log(-log(U))
            let g = zero - (zero - u_val.ln()).ln();
            let l = (log_alpha[j] + g) / temperature;
            logits[j] = l;
            if l > max_l {
                max_l = l;
            }
        }
        let mut sum_exp = zero;
        for j in 0..k {
            logits[j] = (logits[j] - max_l).exp();
            sum_exp += logits[j];
        }
        for j in 0..k {
            result.push(logits[j] / sum_exp);
        }
    }

    let mut out_shape = shape.to_vec();
    out_shape.push(k);
    let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
    if device.is_cuda() {
        out.to(device)
    } else {
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_relaxed_one_hot_invalid_temperature() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        assert!(RelaxedOneHotCategorical::new(0.0_f32, probs).is_err());
    }

    #[test]
    fn test_relaxed_one_hot_invalid_probs() {
        let probs = cpu_tensor(&[0.0, 0.5], &[2]);
        assert!(RelaxedOneHotCategorical::new(1.0_f32, probs).is_err());
    }

    #[test]
    fn test_relaxed_one_hot_sample_shape_and_simplex() {
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let s = d.sample(&[100]).unwrap();
        assert_eq!(s.shape(), &[100, 3]);
        let data = s.data().unwrap();
        for row in 0..100 {
            let row_sum: f32 = (0..3).map(|c| data[row * 3 + c]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-4,
                "row {row} not on simplex: sum={row_sum}"
            );
            for c in 0..3 {
                // Closed simplex [0, 1]: at finite precision a softmax
                // output can underflow to exactly 0.0 or saturate to 1.0
                // for low-prob / high-prob entries at temperature 0.5,
                // so the strict (0, 1) bound was occasionally flaky.
                let v = data[row * 3 + c];
                assert!(
                    (0.0..=1.0).contains(&v),
                    "row {row} col {c}: {v} not in [0, 1]"
                );
            }
        }
    }

    #[test]
    fn test_relaxed_one_hot_low_temperature_concentrates() {
        // At very low temperature, the largest probability should
        // dominate -- mode-collapse toward category 2 in this example.
        let probs = cpu_tensor(&[0.1, 0.1, 0.8], &[3]);
        let d = RelaxedOneHotCategorical::new(0.05_f32, probs).unwrap();
        let s = d.sample(&[200]).unwrap();
        let data = s.data().unwrap();
        let mut category_2_dominant = 0;
        for row in 0..200 {
            let r0 = data[row * 3];
            let r1 = data[row * 3 + 1];
            let r2 = data[row * 3 + 2];
            if r2 > r0 && r2 > r1 {
                category_2_dominant += 1;
            }
        }
        // With probs 0.8 on category 2 and a tiny temperature, we expect
        // category 2 to dominate in most draws (≥ 70%).
        assert!(
            category_2_dominant >= 140,
            "expected category 2 to dominate, only {category_2_dominant}/200"
        );
    }

    #[test]
    fn test_relaxed_one_hot_log_prob_finite() {
        let probs = cpu_tensor(&[0.3, 0.3, 0.4], &[3]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let value = cpu_tensor(&[0.2, 0.3, 0.5], &[3]);
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), [] as [usize; 0]);
        let v = lp.item().unwrap();
        assert!(v.is_finite(), "log_prob should be finite, got {v}");
    }

    #[test]
    fn test_relaxed_one_hot_log_prob_batch() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let value = cpu_tensor(&[0.3, 0.7, 0.5, 0.5], &[2, 2]);
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), &[2]);
        let data = lp.data().unwrap();
        for &v in data {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_relaxed_one_hot_entropy_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        assert!(d.entropy().is_err());
    }

    #[test]
    fn test_relaxed_one_hot_log_prob_wrong_shape_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = RelaxedOneHotCategorical::new(0.5_f32, probs).unwrap();
        let bad = cpu_tensor(&[0.3, 0.3, 0.4], &[3]);
        assert!(d.log_prob(&bad).is_err());
    }
}
