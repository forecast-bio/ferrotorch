//! One-hot categorical distribution.
//!
//! `OneHotCategorical(probs)` is the same as [`Categorical`](crate::Categorical)
//! except samples are returned as one-hot vectors of shape `[..., K]` instead
//! of integer indices.
//!
//! Samples are still discrete: each draw produces exactly one `1.0` and the
//! rest `0.0`. `log_prob` accepts a one-hot value (or any value that picks a
//! single category by argmax / by single non-zero entry) and returns the
//! corresponding `log probs[k]`.
//!
//! Mirrors `torch.distributions.OneHotCategorical`.

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

use crate::Distribution;

/// One-hot categorical distribution over `K` classes.
pub struct OneHotCategorical<T: Float> {
    probs: Tensor<T>,
    /// Cached normalized probability vector.
    normalized: Vec<T>,
    /// Cached cumulative distribution function for inverse-CDF sampling.
    cdf: Vec<T>,
    num_categories: usize,
}

impl<T: Float> OneHotCategorical<T> {
    /// Create a new `OneHotCategorical` over `K = probs.len()` classes.
    ///
    /// `probs` must be a 1-D tensor with non-negative entries summing to a
    /// positive value. They are normalized internally.
    pub fn new(probs: Tensor<T>) -> FerrotorchResult<Self> {
        if probs.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "OneHotCategorical: probs must be 1-D, got shape {:?}",
                    probs.shape()
                ),
            });
        }
        let k = probs.shape()[0];
        if k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "OneHotCategorical: probs must have at least one category".into(),
            });
        }

        let probs_data = probs.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let total: T = probs_data.iter().copied().fold(zero, |a, b| a + b);
        if total <= zero {
            return Err(FerrotorchError::InvalidArgument {
                message: "OneHotCategorical: probs must sum to a positive value".into(),
            });
        }

        // Normalize and build CDF.
        let normalized: Vec<T> = probs_data.iter().map(|&p| p / total).collect();
        let mut cdf = Vec::with_capacity(k);
        let mut cumsum = zero;
        for &p in &normalized {
            cumsum += p;
            cdf.push(cumsum);
        }
        if let Some(last) = cdf.last_mut() {
            *last = one;
        }

        Ok(Self {
            probs,
            normalized,
            cdf,
            num_categories: k,
        })
    }

    /// The (normalized) probability tensor as originally provided.
    pub fn probs(&self) -> &Tensor<T> {
        &self.probs
    }

    /// Number of categories.
    pub fn num_categories(&self) -> usize {
        self.num_categories
    }
}

impl<T: Float> Distribution<T> for OneHotCategorical<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        // Output shape: [shape..., K], one-hot along the last dim.
        let device = self.probs.device();
        let n: usize = shape.iter().product();
        let k = self.num_categories;

        let u = creation::rand::<T>(&[n])?;
        let u_data = u.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();

        let mut result = vec![zero; n * k];
        for (i, &uv) in u_data.iter().enumerate().take(n) {
            // Inverse-CDF sample.
            let mut lo = 0usize;
            let mut hi = k;
            while lo < hi {
                let mid = lo + (hi - lo) / 2;
                if self.cdf[mid] <= uv {
                    lo = mid + 1;
                } else {
                    hi = mid;
                }
            }
            let cat = lo.min(k - 1);
            result[i * k + cat] = one;
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

    fn rsample(&self, _shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "OneHotCategorical: rsample is not supported -- discrete distribution. \
                 Use RelaxedOneHotCategorical for a differentiable continuous relaxation."
                .into(),
        })
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // value: [..., K] where each row is a one-hot (or arbitrary
        // non-negative weights — we compute sum_k value[k] * log(probs[k])).
        // Returns shape [...] with the K dim removed.
        let v_shape = value.shape().to_vec();
        if v_shape.is_empty() || *v_shape.last().unwrap() != self.num_categories {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "OneHotCategorical: log_prob value must have last dim K={}, got shape {:?}",
                    self.num_categories, v_shape
                ),
            });
        }

        let v_data = value.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let eps = T::from(1e-30).unwrap();
        // Precompute log(normalized).
        let log_p: Vec<T> = self.normalized.iter().map(|&p| (p + eps).ln()).collect();

        let n = v_data.len() / self.num_categories;
        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let base = i * self.num_categories;
            let mut sum = zero;
            for k in 0..self.num_categories {
                sum += v_data[base + k] * log_p[k];
            }
            result.push(sum);
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
        // H = -sum_k p_k * log(p_k). Same as Categorical.
        let zero = <T as num_traits::Zero>::zero();
        let eps = T::from(1e-30).unwrap();
        let mut h = zero;
        for &p in &self.normalized {
            let lp = (p + eps).ln();
            h += -p * lp;
        }
        let device = self.probs.device();
        let out = Tensor::from_storage(TensorStorage::cpu(vec![h]), vec![], false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_one_hot_categorical_sample_shape() {
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]);
        let d = OneHotCategorical::new(probs).unwrap();
        let s = d.sample(&[10]).unwrap();
        assert_eq!(s.shape(), &[10, 3]);
        // Each row should be a one-hot.
        let data = s.data().unwrap();
        for row in 0..10 {
            let row_sum: f32 = (0..3).map(|c| data[row * 3 + c]).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-6,
                "row {row} not one-hot: sum={row_sum}"
            );
        }
    }

    #[test]
    fn test_one_hot_categorical_log_prob_pure_one_hot() {
        let probs = cpu_tensor(&[0.2, 0.3, 0.5], &[3]);
        let d = OneHotCategorical::new(probs).unwrap();
        let value = cpu_tensor(&[0.0, 1.0, 0.0], &[3]); // pick category 1
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), [] as [usize; 0]);
        let val = lp.item().unwrap();
        let expected = 0.3_f32.ln();
        assert!(
            (val - expected).abs() < 1e-5,
            "expected {expected}, got {val}"
        );
    }

    #[test]
    fn test_one_hot_categorical_log_prob_batch() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = OneHotCategorical::new(probs).unwrap();
        // Two one-hots, both pick category 0.
        let value = cpu_tensor(&[1.0, 0.0, 1.0, 0.0], &[2, 2]);
        let lp = d.log_prob(&value).unwrap();
        assert_eq!(lp.shape(), &[2]);
        let data = lp.data().unwrap();
        let expected = 0.5_f32.ln();
        assert!((data[0] - expected).abs() < 1e-5);
        assert!((data[1] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_one_hot_categorical_entropy() {
        // Uniform [0.5, 0.5] -> entropy = log(2) ≈ 0.693.
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = OneHotCategorical::new(probs).unwrap();
        let h = d.entropy().unwrap();
        let val = h.item().unwrap();
        assert!((val - 2f32.ln()).abs() < 1e-5);
    }

    #[test]
    fn test_one_hot_categorical_rsample_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = OneHotCategorical::new(probs).unwrap();
        assert!(d.rsample(&[5]).is_err());
    }

    #[test]
    fn test_one_hot_categorical_wrong_shape_errors() {
        let probs = cpu_tensor(&[0.5, 0.5], &[2]);
        let d = OneHotCategorical::new(probs).unwrap();
        // value last dim is 3, but K=2.
        let bad = cpu_tensor(&[0.0, 0.0, 1.0], &[3]);
        assert!(d.log_prob(&bad).is_err());
    }
}
