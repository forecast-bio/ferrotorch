//! Low-rank multivariate normal distribution.
//!
//! `LowRankMultivariateNormal(loc, cov_factor, cov_diag)` defines a Gaussian
//! whose covariance is `Σ = W W^T + diag(D)` where:
//!
//! - `loc` is the mean, shape `[d]`
//! - `cov_factor` (W) is `[d, r]` for some rank `r ≤ d`
//! - `cov_diag` (D) is `[d]` and elementwise positive
//!
//! This is the standard low-rank-plus-diagonal parameterization used in
//! probabilistic PCA, factor analysis, and many variational inference
//! settings. When `r ≪ d`, evaluating Σ⁻¹ and log det Σ via the matrix
//! determinant lemma + Woodbury identity is `O(d r²)` instead of `O(d³)`.
//!
//! # Implementation note
//!
//! The current implementation builds the dense `[d, d]` covariance and
//! delegates to [`MultivariateNormal::from_covariance`] for sampling and
//! log_prob. This is correct but `O(d²)` in memory and `O(d³)` in
//! per-call cost — acceptable for moderate `d` (a few hundred). The
//! O(d r²) Woodbury fast paths can be added incrementally as a follow-up
//! once we need them; correctness lands first.
//!
//! Mirrors `torch.distributions.LowRankMultivariateNormal`.

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::{Distribution, MultivariateNormal};

/// Multivariate normal with low-rank-plus-diagonal covariance.
pub struct LowRankMultivariateNormal<T: Float> {
    loc: Tensor<T>,
    cov_factor: Tensor<T>,
    cov_diag: Tensor<T>,
    /// Inner [`MultivariateNormal`] built from the dense `Σ = W W^T + diag(D)`.
    /// All sample/log_prob/entropy calls delegate here.
    inner: MultivariateNormal<T>,
    d: usize,
    r: usize,
}

impl<T: Float> LowRankMultivariateNormal<T> {
    /// Construct from a mean vector, low-rank covariance factor, and
    /// diagonal correction.
    ///
    /// # Errors
    ///
    /// - `loc` must be 1-D shape `[d]`.
    /// - `cov_factor` must be 2-D shape `[d, r]`.
    /// - `cov_diag` must be 1-D shape `[d]` and contain only positive values.
    pub fn new(
        loc: Tensor<T>,
        cov_factor: Tensor<T>,
        cov_diag: Tensor<T>,
    ) -> FerrotorchResult<Self> {
        let loc_shape = loc.shape().to_vec();
        let factor_shape = cov_factor.shape().to_vec();
        let diag_shape = cov_diag.shape().to_vec();

        if loc_shape.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LowRankMultivariateNormal: loc must be 1-D, got shape {loc_shape:?}"
                ),
            });
        }
        let d = loc_shape[0];

        if factor_shape.len() != 2 || factor_shape[0] != d {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LowRankMultivariateNormal: cov_factor must be [{d}, r], got {factor_shape:?}"
                ),
            });
        }
        let r = factor_shape[1];

        if diag_shape != [d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LowRankMultivariateNormal: cov_diag must be [{d}], got {diag_shape:?}"
                ),
            });
        }

        // Validate that cov_diag is positive.
        let diag_data = cov_diag.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        for (i, &v) in diag_data.iter().enumerate() {
            if v <= zero {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "LowRankMultivariateNormal: cov_diag[{i}] = {} must be > 0",
                        v.to_f64().unwrap_or(f64::NAN)
                    ),
                });
            }
        }

        // Build the dense covariance Σ = W W^T + diag(D) on CPU. We
        // walk the factor data directly rather than calling matmul to
        // keep this self-contained and to avoid pulling in autograd
        // dependencies for what is a one-shot construction step.
        let factor_data = cov_factor.data_vec()?;
        let mut cov = vec![zero; d * d];
        for i in 0..d {
            for j in 0..d {
                let mut acc = zero;
                for k in 0..r {
                    acc += factor_data[i * r + k] * factor_data[j * r + k];
                }
                if i == j {
                    acc += diag_data[i];
                }
                cov[i * d + j] = acc;
            }
        }
        let device = loc.device();
        let cov_t = {
            let t = Tensor::from_storage(TensorStorage::cpu(cov), vec![d, d], false)?;
            if device.is_cuda() { t.to(device)? } else { t }
        };

        let inner = MultivariateNormal::from_covariance(loc.clone(), cov_t)?;

        Ok(Self {
            loc,
            cov_factor,
            cov_diag,
            inner,
            d,
            r,
        })
    }

    /// The mean vector.
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The low-rank covariance factor `W` of shape `[d, r]`.
    pub fn cov_factor(&self) -> &Tensor<T> {
        &self.cov_factor
    }

    /// The diagonal correction `D` of shape `[d]`.
    pub fn cov_diag(&self) -> &Tensor<T> {
        &self.cov_diag
    }

    /// Dimensionality `d`.
    pub fn dim(&self) -> usize {
        self.d
    }

    /// Rank `r` of the low-rank factor.
    pub fn rank(&self) -> usize {
        self.r
    }
}

impl<T: Float> Distribution<T> for LowRankMultivariateNormal<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        self.inner.sample(shape)
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        self.inner.rsample(shape)
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.inner.log_prob(value)
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        self.inner.entropy()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_low_rank_basic_construction() {
        // d=3, r=1: factor=[3,1], diag=[3].
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let factor = cpu_tensor(&[1.0, 0.5, -0.5], &[3, 1]);
        let diag = cpu_tensor(&[1.0, 1.0, 1.0], &[3]);
        let mvn = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
        assert_eq!(mvn.dim(), 3);
        assert_eq!(mvn.rank(), 1);
    }

    #[test]
    fn test_low_rank_negative_diag_errors() {
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let factor = cpu_tensor(&[1.0, 0.0], &[2, 1]);
        let diag = cpu_tensor(&[1.0, -0.5], &[2]);
        assert!(LowRankMultivariateNormal::new(loc, factor, diag).is_err());
    }

    #[test]
    fn test_low_rank_wrong_factor_shape_errors() {
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        // factor [3, 1] doesn't match d=2
        let factor = cpu_tensor(&[1.0, 0.0, 0.5], &[3, 1]);
        let diag = cpu_tensor(&[1.0, 1.0], &[2]);
        assert!(LowRankMultivariateNormal::new(loc, factor, diag).is_err());
    }

    #[test]
    fn test_low_rank_wrong_diag_shape_errors() {
        let loc = cpu_tensor(&[0.0, 0.0], &[2]);
        let factor = cpu_tensor(&[1.0, 0.0], &[2, 1]);
        let diag = cpu_tensor(&[1.0, 1.0, 1.0], &[3]);
        assert!(LowRankMultivariateNormal::new(loc, factor, diag).is_err());
    }

    #[test]
    fn test_low_rank_log_prob_at_mean_diagonal_only() {
        // With cov_factor = 0 (rank 1, all zeros) and cov_diag = ones,
        // the distribution is N(0, I). log_prob at mean = -d/2 log(2pi).
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let factor = cpu_tensor(&[0.0, 0.0, 0.0], &[3, 1]);
        let diag = cpu_tensor(&[1.0, 1.0, 1.0], &[3]);
        let mvn = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
        let value = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let lp = mvn.log_prob(&value).unwrap();
        let val = lp.item().unwrap();
        let expected = -1.5_f32 * (2.0 * std::f32::consts::PI).ln();
        assert!(
            (val - expected).abs() < 1e-4,
            "expected ≈ {expected}, got {val}"
        );
    }

    #[test]
    fn test_low_rank_sample_shape() {
        // Inner MultivariateNormal samples with shape `[batch..., event_dim]`
        // where event_dim == d. Passing batch_shape=[10] yields output [10, 3].
        let loc = cpu_tensor(&[0.0, 0.0, 0.0], &[3]);
        let factor = cpu_tensor(&[0.5, 0.5, 0.5], &[3, 1]);
        let diag = cpu_tensor(&[0.1, 0.1, 0.1], &[3]);
        let mvn = LowRankMultivariateNormal::new(loc, factor, diag).unwrap();
        let s = mvn.sample(&[10]).unwrap();
        assert_eq!(s.shape(), &[10, 3]);
    }
}
