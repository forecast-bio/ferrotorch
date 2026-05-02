//! Probability distributions for ferrotorch.
//!
//! This crate provides differentiable probability distributions following the
//! PyTorch `torch.distributions` API. Each distribution supports:
//!
//! - **`sample`** — draw samples (no gradient)
//! - **`rsample`** — reparameterized sampling (gradient flows through samples)
//! - **`log_prob`** — compute log-probability of a value
//! - **`entropy`** — compute the distribution's entropy
//!
//! # Distributions
//!
//! | Distribution | Parameters | Reparameterized |
//! |-------------|-----------|-----------------|
//! | [`Normal`] | `loc`, `scale` | Yes |
//! | [`Uniform`] | `low`, `high` | Yes |
//! | [`Bernoulli`] | `probs` | No (discrete) |
//! | [`Categorical`] | `probs` | No (discrete) |
//! | [`Beta`] | `concentration1`, `concentration0` | Yes |
//! | [`Gamma`] | `concentration`, `rate` | Yes |
//! | [`Exponential`] | `rate` | Yes |
//! | [`Laplace`] | `loc`, `scale` | Yes |
//! | [`Cauchy`] | `loc`, `scale` | Yes |
//! | [`Gumbel`] | `loc`, `scale` | Yes |
//! | [`HalfNormal`] | `scale` | Yes |
//! | [`LogNormal`] | `loc`, `scale` | Yes |
//! | [`Poisson`] | `rate` | No (discrete) |
//! | [`StudentT`] | `df`, `loc`, `scale` | Yes |
//! | [`MultivariateNormal`] | `loc`, `scale_tril` | Yes |
//! | [`LowRankMultivariateNormal`] | `loc`, `cov_factor`, `cov_diag` | Yes |
//! | [`Dirichlet`] | `concentration` | Yes |
//! | [`Multinomial`] | `total_count`, `probs` | No (discrete) |
//! | [`Independent`] | base distribution + `reinterpreted_batch_ndims` | inherits |
//! | [`MixtureSameFamily`] | mixing `Categorical` + components | No |
//! | [`OneHotCategorical`] | `probs` | No (discrete) |
//! | [`RelaxedBernoulli`] | `temperature`, `probs` | Yes (Concrete relaxation) |
//! | [`RelaxedOneHotCategorical`] | `temperature`, `probs` | Yes (Concrete relaxation) |
//!
//! # Infrastructure
//!
//! - [`constraints`] — constraint objects for parameter and support validation
//! - [`transforms`] — bijective transforms with log-det-Jacobian computation
//! - [`kl`] — analytical KL divergence for same-family distribution pairs
//! - [`TransformedDistribution`](transforms::TransformedDistribution) — apply
//!   bijective transforms to a base distribution

mod bernoulli;
mod beta;
mod categorical;
mod cauchy;
pub mod constraints;
mod dirichlet;
mod exponential;
mod gamma;
mod gumbel;
mod half_normal;
mod independent;
pub mod kl;
mod kumaraswamy;
mod laplace;
mod lognormal;
mod low_rank_multivariate_normal;
mod mixture_same_family;
mod multinomial;
mod multivariate_normal;
mod normal;
mod one_hot_categorical;
mod pareto;
mod poisson;
mod relaxed_bernoulli;
mod relaxed_one_hot_categorical;
pub(crate) mod special_fns;
mod student_t;
pub mod transforms;
mod uniform;
mod von_mises;
mod weibull;

pub use bernoulli::Bernoulli;
pub use beta::Beta;
pub use categorical::Categorical;
pub use cauchy::Cauchy;
pub use dirichlet::Dirichlet;
pub use exponential::Exponential;
pub use gamma::Gamma;
pub use gumbel::Gumbel;
pub use half_normal::HalfNormal;
pub use independent::Independent;
pub use kumaraswamy::Kumaraswamy;
pub use laplace::Laplace;
pub use lognormal::LogNormal;
pub use low_rank_multivariate_normal::LowRankMultivariateNormal;
pub use mixture_same_family::MixtureSameFamily;
pub use multinomial::Multinomial;
pub use multivariate_normal::MultivariateNormal;
pub use normal::Normal;
pub use one_hot_categorical::OneHotCategorical;
pub use pareto::Pareto;
pub use poisson::Poisson;
pub use relaxed_bernoulli::RelaxedBernoulli;
pub use relaxed_one_hot_categorical::RelaxedOneHotCategorical;
pub use student_t::StudentT;
pub use transforms::TransformedDistribution;
pub use uniform::Uniform;
pub use von_mises::VonMises;
pub use weibull::Weibull;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::tensor::Tensor;

/// A probability distribution over tensors.
///
/// This trait mirrors PyTorch's `torch.distributions.Distribution` base class.
/// Implementations define how to sample, compute log-probabilities, and
/// measure entropy.
///
/// # Type parameter
///
/// `T` must implement [`Float`] — currently `f32` or `f64`.
///
/// # `sample` vs `rsample`
///
/// - [`sample`](Distribution::sample) draws samples with no gradient. Use for
///   discrete distributions or when gradients through sampling are not needed.
/// - [`rsample`](Distribution::rsample) draws reparameterized samples. The
///   result has `requires_grad = true` and gradients flow back through the
///   sampling operation via the reparameterization trick. This is essential
///   for variational inference (VAE, etc.).
///
/// Distributions that cannot be reparameterized (e.g., [`Bernoulli`],
/// [`Categorical`]) return an error from `rsample`.
pub trait Distribution<T: Float>: Send + Sync {
    /// Draw samples from the distribution.
    ///
    /// The returned tensor has the given `shape` and `requires_grad = false`.
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>>;

    /// Draw reparameterized samples from the distribution.
    ///
    /// The returned tensor has `requires_grad = true` and gradients flow
    /// through the sampling operation back to the distribution parameters.
    ///
    /// Returns an error for distributions that cannot be reparameterized.
    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>>;

    /// Compute the log-probability of `value` under the distribution.
    ///
    /// Returns a tensor with the same shape as `value`.
    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Compute the entropy of the distribution.
    ///
    /// Returns a scalar tensor (or a tensor matching the batch shape of the
    /// distribution parameters).
    fn entropy(&self) -> FerrotorchResult<Tensor<T>>;

    // -----------------------------------------------------------------------
    // Distribution properties (#585) — default implementations return
    // NotImplementedOnCuda-style errors. Concrete distributions override
    // what they can express in closed form.
    // -----------------------------------------------------------------------

    /// Cumulative distribution function: `P(X <= value)`. Default returns an
    /// `InvalidArgument` error for distributions without a closed-form CDF.
    fn cdf(&self, _value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "cdf not implemented for this distribution".into(),
        })
    }

    /// Inverse CDF (quantile function): the value `x` such that
    /// `P(X <= x) = q`. Default returns an `InvalidArgument` error.
    fn icdf(&self, _q: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "icdf not implemented for this distribution".into(),
        })
    }

    /// Distribution mean. Default returns an `InvalidArgument` error.
    fn mean(&self) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "mean not implemented for this distribution".into(),
        })
    }

    /// Distribution mode. Default returns an `InvalidArgument` error.
    fn mode(&self) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "mode not implemented for this distribution".into(),
        })
    }

    /// Distribution variance. Default returns an `InvalidArgument` error.
    fn variance(&self) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "variance not implemented for this distribution".into(),
        })
    }

    /// Distribution standard deviation. Default: `sqrt(variance)`.
    fn stddev(&self) -> FerrotorchResult<Tensor<T>> {
        let v = self.variance()?;
        let data = v.data_vec()?;
        let out: Vec<T> = data.iter().map(|x| x.sqrt()).collect();
        Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(out),
            v.shape().to_vec(),
            false,
        )
    }
}
