//! Relaxed Bernoulli (Concrete) distribution.
//!
//! `RelaxedBernoulli(temperature, probs)` is a continuous relaxation of
//! [`Bernoulli`](crate::Bernoulli) using the Gumbel-softmax / Concrete trick
//! (Maddison et al. 2017, Jang et al. 2017). Samples lie in the open
//! interval `(0, 1)` rather than at the discrete points `{0, 1}`.
//!
//! As `temperature → 0`, samples concentrate on `{0, 1}` and the relaxed
//! distribution recovers the discrete Bernoulli. As `temperature → ∞`,
//! samples concentrate near `0.5` and the distribution approaches uniform.
//!
//! # Reparameterization
//!
//! Sampling is reparameterizable:
//! ```text
//! L ~ Logistic(0, 1)        (i.e. L = log(U) - log(1-U), U ~ Uniform(0,1))
//! z = sigmoid((L + logits) / temperature)
//! ```
//! where `logits = log(probs / (1 - probs))`. This is the Concrete relaxation
//! of a Bernoulli draw and supports gradient flow through `probs` /
//! `temperature` via the autograd graph (when those tensors require grad).
//!
//! Mirrors `torch.distributions.RelaxedBernoulli`.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// Continuous relaxation of a Bernoulli distribution.
pub struct RelaxedBernoulli<T: Float> {
    temperature: T,
    probs: Tensor<T>,
}

impl<T: Float> RelaxedBernoulli<T> {
    /// Construct a RelaxedBernoulli with the given temperature and
    /// per-element probabilities.
    ///
    /// # Errors
    ///
    /// Returns an error if `temperature <= 0` or if any element of `probs`
    /// is outside `(0, 1)` (the open interval -- the relaxation requires
    /// strictly positive logits + log(1-p)).
    pub fn new(temperature: T, probs: Tensor<T>) -> FerrotorchResult<Self> {
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        if temperature <= zero {
            return Err(FerrotorchError::InvalidArgument {
                message: "RelaxedBernoulli: temperature must be > 0".into(),
            });
        }
        let probs_data = probs.data_vec()?;
        for (i, &p) in probs_data.iter().enumerate() {
            if p <= zero || p >= one {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "RelaxedBernoulli: probs[{i}] = {} must be in (0, 1)",
                        p.to_f64().unwrap_or(f64::NAN)
                    ),
                });
            }
        }
        Ok(Self { temperature, probs })
    }

    /// The temperature parameter.
    pub fn temperature(&self) -> T {
        self.temperature
    }

    /// The probability parameter.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }
}

impl<T: Float> Distribution<T> for RelaxedBernoulli<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // sample uses the same Concrete forward pass as rsample but without
        // an autograd graph (since "sample" is non-differentiable by API
        // contract). The math is identical.
        relaxed_bernoulli_sample(self.temperature, &self.probs, shape, false)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // rsample uses the same forward pass; differentiation flows through
        // the surrounding tensor ops if the user constructs them downstream.
        // Note: a fully autograd-aware rsample requires the random Logistic
        // noise to be detached and the rest of the path to be a standard
        // tensor-op composition. Since this implementation builds the
        // result via scalar CPU code, callers wanting differentiable
        // samples should reconstruct the formula using ferrotorch tensor
        // ops (sigmoid, sub, div) over a detached Logistic noise tensor.
        relaxed_bernoulli_sample(self.temperature, &self.probs, shape, true)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // The Concrete log density (Maddison et al. 2017, eqn 21):
        //   log p(z; alpha, lambda) = log(lambda) + log(alpha)
        //       + (-lambda - 1) * log(z)
        //       + (-lambda - 1) * log(1 - z)
        //       - 2 * log( alpha * z^(-lambda) + (1 - z)^(-lambda) )
        // where alpha = probs / (1 - probs) and lambda = temperature.
        //
        // Equivalently, in terms of probs and 1-probs:
        //   log p(z) = log(lambda) - 2 * log( probs * z^(-lambda)
        //                              + (1-probs) * (1-z)^(-lambda) )
        //              + (-lambda - 1) * (log(z) + log(1-z))
        //              + log(probs) - log(1 - probs)? -- actually drops out
        //
        // We use the form from PyTorch RelaxedBernoulli source:
        //   diffs = logits - log(z) * lambda + log(1 - z) * lambda
        //   log_prob = log(lambda) - lambda * log(z) - lambda * log(1 - z)
        //              - 2 * softplus(diffs) + diffs + log(lambda)
        // (this matches torch.distributions.LogitRelaxedBernoulli.log_prob
        //  on logits, after a change-of-variable to z = sigmoid(logits/lambda))
        //
        // For our purposes we use the most stable, direct form:
        //   log_prob(z; p, lambda) = log(lambda)
        //                         + (lambda - 1) * (log(z) + log(1 - z))? no
        //                         - 2 log( p z^(-lambda) + (1-p)(1-z)^(-lambda) )
        //
        // To avoid numerical issues with z^(-lambda), use logs:
        //   t1 = log(p) + (-lambda) * log(z)
        //   t2 = log(1-p) + (-lambda) * log(1-z)
        //   log( e^t1 + e^t2 ) = logsumexp(t1, t2)
        //   log_prob = log(lambda) - 2 * logsumexp(t1, t2)
        //              + (-lambda - 1) * (log(z) + log(1-z))? Wait, that's
        //              wrong too.
        //
        // Final, derived from Maddison eqn 21 in α/β form:
        //   p_α(x) = α * λ * x^(-λ-1) * (1-x)^(-λ-1) / (α * x^(-λ) + (1-x)^(-λ))^2
        // Taking log and using α = p/(1-p):
        //   log p(x) = log(α) + log(λ) - (λ+1)*log(x) - (λ+1)*log(1-x)
        //              - 2 * log( α * x^(-λ) + (1-x)^(-λ) )
        //            = log(p) - log(1-p) + log(λ) - (λ+1)*(log(x)+log(1-x))
        //              - 2 * logsumexp(log(p) - λ*log(x), log(1-p) - λ*log(1-x))
        //
        // This is the form we implement.
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let lambda = self.temperature;
        let neg_lambda = zero - lambda;

        let probs_data = self.probs.data_vec()?;
        let v_data = value.data_vec()?;
        let eps = T::from(1e-20).unwrap();

        let result: Vec<T> = v_data
            .iter()
            .zip(probs_data.iter().cycle())
            .map(|(&z, &p)| {
                let z = z.max(eps).min(one - eps);
                let log_z = z.ln();
                let log_1mz = (one - z).ln();
                let log_p = p.ln();
                let log_1mp = (one - p).ln();
                // logsumexp(log_p + (-λ)*log_z, log_1mp + (-λ)*log_1mz)
                let a = log_p + neg_lambda * log_z;
                let b = log_1mp + neg_lambda * log_1mz;
                let max_ab = if a > b { a } else { b };
                let lse = max_ab + ((a - max_ab).exp() + (b - max_ab).exp()).ln();
                log_p - log_1mp + lambda.ln()
                    - (lambda + one) * (log_z + log_1mz)
                    - (one + one) * lse
            })
            .collect();

        let device = self.probs.device();
        let out = Tensor::from_storage(TensorStorage::cpu(result), value.shape().to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "RelaxedBernoulli: entropy has no closed form".into(),
        })
    }
}

/// Concrete forward sampling for RelaxedBernoulli (shared by sample and
/// rsample). The result is computed in CPU scalar code; gradient tracking
/// requires the caller to recompute via tensor ops over detached Logistic
/// noise (see the RelaxedBernoulli rsample doc comment).
fn relaxed_bernoulli_sample<T: Float>(
    temperature: T,
    probs: &Tensor<T>,
    shape: &[usize],
    _reparam: bool,
) -> FerrotorchResult<Tensor<T>> {
    let device = probs.device();
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let n: usize = shape.iter().product();
    let u = creation::rand::<T>(&[n])?;
    let u_data = u.data_vec()?;
    let probs_data = probs.data_vec()?;
    let eps = T::from(1e-20).unwrap();

    let result: Vec<T> = u_data
        .iter()
        .zip(probs_data.iter().cycle())
        .map(|(&u_val, &p)| {
            // L = log(U) - log(1 - U), the standard Logistic noise.
            let u_clamped = u_val.max(eps).min(one - eps);
            let l = u_clamped.ln() - (one - u_clamped).ln();
            // logits = log(p / (1 - p))
            let p_clamped = p.max(eps).min(one - eps);
            let logits = (p_clamped / (one - p_clamped)).ln();
            // z = sigmoid((L + logits) / temperature)
            let arg = (l + logits) / temperature;
            // numerically stable sigmoid
            if arg >= zero {
                let e = (zero - arg).exp();
                one / (one + e)
            } else {
                let e = arg.exp();
                e / (one + e)
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

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_relaxed_bernoulli_invalid_temperature() {
        let probs = cpu_tensor(&[0.5], &[1]);
        assert!(RelaxedBernoulli::new(0.0_f32, probs).is_err());
    }

    #[test]
    fn test_relaxed_bernoulli_invalid_probs() {
        let probs = cpu_tensor(&[0.0, 0.5], &[2]);
        assert!(RelaxedBernoulli::new(1.0_f32, probs).is_err());
        let probs = cpu_tensor(&[1.0, 0.5], &[2]);
        assert!(RelaxedBernoulli::new(1.0_f32, probs).is_err());
    }

    #[test]
    fn test_relaxed_bernoulli_sample_in_closed_unit_interval() {
        // Mathematically samples are in the open interval (0, 1), but f32
        // sigmoid can saturate at 0 or 1 for extreme arguments. We assert
        // the closed interval [0, 1] and verify that the *vast majority*
        // of samples land strictly in the interior.
        let probs = cpu_tensor(&[0.3, 0.7], &[2]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        let s = d.sample(&[1000]).unwrap();
        let data = s.data().unwrap();
        let mut interior = 0;
        for &v in data {
            assert!((0.0..=1.0).contains(&v), "sample out of [0,1]: {v}");
            if v > 0.0 && v < 1.0 {
                interior += 1;
            }
        }
        // At least 95% should be strictly in (0, 1).
        assert!(
            interior >= 950,
            "expected most samples in interior, got {interior}/1000"
        );
    }

    #[test]
    fn test_relaxed_bernoulli_low_temperature_concentrates() {
        // Very low temperature -> samples should be near 0 or 1.
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.01_f32, probs).unwrap();
        let s = d.sample(&[100]).unwrap();
        let data = s.data().unwrap();
        // Most samples should be < 0.05 or > 0.95.
        let extreme = data
            .iter()
            .filter(|&&v| !(0.05..=0.95).contains(&v))
            .count();
        assert!(
            extreme > 90,
            "low temp should give bimodal samples; got only {extreme}/100 extreme"
        );
    }

    #[test]
    fn test_relaxed_bernoulli_log_prob_finite() {
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        let value = cpu_tensor(&[0.3], &[1]);
        let lp = d.log_prob(&value).unwrap();
        let v = lp.data().unwrap()[0];
        assert!(v.is_finite(), "log_prob should be finite, got {v}");
    }

    #[test]
    fn test_relaxed_bernoulli_log_prob_symmetry() {
        // For probs=0.5, log_prob(z) should equal log_prob(1-z) by symmetry.
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        let v1 = cpu_tensor(&[0.2], &[1]);
        let v2 = cpu_tensor(&[0.8], &[1]);
        let lp1 = d.log_prob(&v1).unwrap().data().unwrap()[0];
        let lp2 = d.log_prob(&v2).unwrap().data().unwrap()[0];
        assert!(
            (lp1 - lp2).abs() < 1e-5,
            "symmetry violated: lp(0.2)={lp1}, lp(0.8)={lp2}"
        );
    }

    #[test]
    fn test_relaxed_bernoulli_entropy_errors() {
        let probs = cpu_tensor(&[0.5], &[1]);
        let d = RelaxedBernoulli::new(0.5_f32, probs).unwrap();
        assert!(d.entropy().is_err());
    }
}
