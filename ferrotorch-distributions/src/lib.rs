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
//! | [`MultivariateNormal`] | `loc`, `scale_tril` | Yes |
//! | [`Dirichlet`] | `concentration` | Yes |
//! | [`Multinomial`] | `total_count`, `probs` | No (discrete) |

mod bernoulli;
mod categorical;
mod dirichlet;
mod multinomial;
mod multivariate_normal;
mod normal;
mod uniform;

pub use bernoulli::Bernoulli;
pub use categorical::Categorical;
pub use dirichlet::Dirichlet;
pub use multinomial::Multinomial;
pub use multivariate_normal::MultivariateNormal;
pub use normal::Normal;
pub use uniform::Uniform;

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::FerrotorchResult;
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
}
