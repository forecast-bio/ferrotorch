//! Mixture distribution where all components are from the same family.
//!
//! `MixtureSameFamily(mixing, components)` defines a finite mixture
//! distribution. The mixing distribution is a `Categorical` over `K`
//! components, and the components are encoded as a single distribution
//! whose batch shape's rightmost dimension is the component index.
//!
//! Sample procedure:
//!   1. Draw a component index `k` from `mixing` (Categorical).
//!   2. Sample from component `k` of the components distribution.
//!
//! `log_prob(x) = logsumexp_k( log mixing_probs[k] + components_log_prob[k](x) )`.
//!
//! Mirrors `torch.distributions.MixtureSameFamily`.
//!
//! # Limitations
//!
//! This implementation accepts the components distribution by-value as a
//! type that yields per-component log_probs of shape `[..., K]` where the
//! rightmost dim is the component axis. Sample currently supports only
//! the simple case where the components share parameters and the
//! per-sample selection is performed in scalar code on CPU.
//!
//! `rsample` is not supported (mixture sampling is non-reparameterizable
//! without Gumbel-softmax tricks).

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::{Categorical, Distribution};

/// Finite mixture distribution with same-family components.
///
/// The components distribution `D` must produce log_probs of shape
/// `[..., K]` for input values, where `K` is the number of mixture
/// components and matches the size of the mixing Categorical.
pub struct MixtureSameFamily<T: Float, D: Distribution<T>> {
    mixing: Categorical<T>,
    components: D,
    /// Number of mixture components (K). Equal to mixing.num_categories().
    num_components: usize,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float, D: Distribution<T>> MixtureSameFamily<T, D> {
    /// Build a new mixture from a Categorical mixing distribution and a
    /// component distribution that yields per-component log_probs.
    ///
    /// # Errors
    ///
    /// Returns an error if `mixing` has zero components.
    pub fn new(mixing: Categorical<T>, components: D) -> FerrotorchResult<Self> {
        let k = mixing.num_categories();
        if k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "MixtureSameFamily: mixing distribution must have at least 1 component"
                    .into(),
            });
        }
        Ok(Self {
            mixing,
            components,
            num_components: k,
            _phantom: std::marker::PhantomData,
        })
    }

    /// The mixing weights distribution.
    pub fn mixing(&self) -> &Categorical<T> {
        &self.mixing
    }

    /// The components distribution.
    pub fn components(&self) -> &D {
        &self.components
    }

    /// Number of mixture components.
    pub fn num_components(&self) -> usize {
        self.num_components
    }
}

impl<T: Float, D: Distribution<T>> Distribution<T> for MixtureSameFamily<T, D> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[self.mixing.probs()],
            "MixtureSameFamily::sample",
        )?;
        // Two-step sampling:
        //   1. Draw a component index per-output from the mixing Categorical.
        //   2. Draw a sample from each chosen component.
        //
        // Because the components distribution does not expose per-component
        // index access in the trait, we draw `numel` samples from the full
        // components distribution at the requested shape and then index
        // along the component axis using the chosen indices. This relies
        // on the convention that the components distribution sample with
        // shape [..., K, *event_shape] gives all-K outputs per draw.
        //
        // For the simplest and most common case (a Normal with batch shape
        // [K]), we instead just draw `numel` samples from the components
        // distribution and pick the right one per-output. This requires
        // CPU-side gather logic.
        //
        // We implement the simple case: components.sample(shape) is
        // assumed to produce a tensor whose rightmost dim is the
        // component axis, of size K. We then gather one slice per output.

        // Step 1: pick component indices from the mixing distribution.
        let comp_idx = self.mixing.sample(shape)?;
        let comp_idx_data = comp_idx.data_vec()?;

        // Step 2: draw component samples. We expand the request shape with
        // a trailing K dim so the components distribution produces all
        // possible outputs, then we index per-element with comp_idx.
        let mut comp_shape: Vec<usize> = shape.to_vec();
        comp_shape.push(self.num_components);
        let comp_samples = self.components.sample(&comp_shape)?;
        let comp_data = comp_samples.data_vec()?;

        // Build the gathered output: for each output index i,
        // out[i] = comp_data[i * K + comp_idx[i]].
        let numel: usize = shape.iter().product();
        let mut result = Vec::with_capacity(numel);
        for (i, k_float) in comp_idx_data.iter().enumerate().take(numel) {
            let k = k_float.to_usize().unwrap_or(0).min(self.num_components - 1);
            let val = comp_data[i * self.num_components + k];
            result.push(val);
        }

        let device = self.mixing.probs().device();
        let out = Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message:
                "MixtureSameFamily: rsample is not supported -- mixture sampling is not reparameterizable. \
                 Use Gumbel-softmax (RelaxedOneHotCategorical) for differentiable approximations."
                    .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[self.mixing.probs(), value],
            "MixtureSameFamily::log_prob",
        )?;
        // log_prob(x) = logsumexp_k( log mixing_probs[k] + log p_k(x) )
        //
        // We compute the per-component log_probs by replicating value K
        // times along a new last axis (so it broadcasts against the
        // components distribution's K-shaped batch dim) and asking the
        // components distribution for the log_prob.
        //
        // Concretely: value has shape [...], components' log_prob expects
        // a tensor whose trailing dim is K. We unsqueeze and tile value to
        // shape [..., K], then call components.log_prob.
        let v_shape = value.shape().to_vec();
        let mut tiled_shape = v_shape.clone();
        tiled_shape.push(self.num_components);

        let v_data = value.data_vec()?;
        let v_numel = v_data.len();
        let mut tiled = Vec::with_capacity(v_numel * self.num_components);
        for &v in v_data.iter() {
            for _ in 0..self.num_components {
                tiled.push(v);
            }
        }
        let v_dev = value.device();
        let value_tiled = {
            let t = Tensor::from_storage(TensorStorage::cpu(tiled), tiled_shape.clone(), false)?;
            if v_dev.is_cuda() { t.to(v_dev)? } else { t }
        };

        // Per-component log p_k(x): shape [..., K].
        let comp_lp = self.components.log_prob(&value_tiled)?;
        let comp_lp_data = comp_lp.data_vec()?;

        // Mixing log probs: log(probs[k]).
        let mix_probs = self.mixing.probs().data_vec()?;
        let mix_log: Vec<T> = mix_probs.iter().map(|&p| p.ln()).collect();

        // logsumexp along the trailing K dim.
        let mut result = Vec::with_capacity(v_numel);
        for v_idx in 0..v_numel {
            let base = v_idx * self.num_components;
            // Compute log sum exp over k of (mix_log[k] + comp_lp[base+k]).
            let mut max_val = T::neg_infinity();
            for k in 0..self.num_components {
                let lp = mix_log[k] + comp_lp_data[base + k];
                if lp > max_val {
                    max_val = lp;
                }
            }
            let mut sum_exp = <T as num_traits::Zero>::zero();
            for k in 0..self.num_components {
                let lp = mix_log[k] + comp_lp_data[base + k];
                sum_exp += (lp - max_val).exp();
            }
            result.push(max_val + sum_exp.ln());
        }

        let out = Tensor::from_storage(TensorStorage::cpu(result), v_shape, false)?;
        if v_dev.is_cuda() {
            out.to(v_dev)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        // Closed-form entropy is not generally tractable for mixtures.
        // PyTorch's MixtureSameFamily also does not implement entropy.
        Err(FerrotorchError::InvalidArgument {
            message: "MixtureSameFamily: entropy has no closed form for general mixtures".into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Normal;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_mixture_basic_log_prob() {
        // Two equal-weight Normals: N(-1, 1) and N(1, 1).
        // log_prob at x=0 should be log(0.5*N(0;-1,1) + 0.5*N(0;1,1)).
        // N(0;-1,1) = N(0;1,1) by symmetry, so the answer is just
        // log(N(0;0,1)) shifted: actually the value is
        //    log( 0.5 * (1/sqrt(2pi)) * exp(-0.5) + 0.5 * (1/sqrt(2pi)) * exp(-0.5) )
        //  = log( (1/sqrt(2pi)) * exp(-0.5) )
        //  = -0.5 * log(2pi) - 0.5 ≈ -1.4189
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();

        let loc = cpu_tensor(&[-1.0, 1.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();

        let m = MixtureSameFamily::new(mixing, components).unwrap();
        let value = cpu_tensor(&[0.0], &[1]);
        let lp = m.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), &[1]);
        let val = lp.data().unwrap()[0];
        let expected = -0.5_f32 * (2.0 * std::f32::consts::PI).ln() - 0.5;
        assert!(
            (val - expected).abs() < 1e-4,
            "expected {expected}, got {val}"
        );
    }

    #[test]
    fn test_mixture_rsample_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[0.0, 1.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();
        assert!(m.rsample(&[10]).is_err());
    }

    #[test]
    fn test_mixture_entropy_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[0.0, 1.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();
        assert!(m.entropy().is_err());
    }

    #[test]
    fn test_mixture_log_prob_weighted() {
        // Asymmetric weights: 0.9 for component 0, 0.1 for component 1.
        // At x = -1, component 0's log_prob is high; component 1's is low.
        // The mixture log_prob should be close to log(0.9) + N(-1;-1,1).log_prob.
        let probs = cpu_tensor(&[0.9, 0.1], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[-1.0, 5.0], &[2]);
        let scale = cpu_tensor(&[1.0, 1.0], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();
        let value = cpu_tensor(&[-1.0], &[1]);
        let lp = m.log_prob(&value).unwrap();

        // Component 0 dominates at x=-1.
        // log(0.9) + N(-1;-1,1).log_prob = log(0.9) - 0.5*log(2pi) ≈
        //   -0.1054 - 0.9189 ≈ -1.0243.
        // Component 1 contributes negligibly: log(0.1) - 0.5*log(2pi) - 18 ≈ -21.22
        // logsumexp([-1.0243, -21.22]) ≈ -1.0243.
        let val = lp.data().unwrap()[0];
        assert!((val + 1.0243).abs() < 0.01, "expected ≈ -1.0243, got {val}");
    }

    #[test]
    fn test_mixture_sample_shape() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let mixing = Categorical::new(probs).unwrap();
        let loc = cpu_tensor(&[-1.0, 1.0], &[2]);
        let scale = cpu_tensor(&[0.5, 0.5], &[2]);
        let components = Normal::new(loc, scale).unwrap();
        let m = MixtureSameFamily::new(mixing, components).unwrap();
        let s = m.sample(&[100]).unwrap();
        assert_eq!(s.shape(), &[100]);
    }
}
