//! Multivariate normal (Gaussian) distribution.
//!
//! `MultivariateNormal(loc, scale_tril)` defines a multivariate Gaussian with
//! mean vector `loc` and covariance `L L^T` where `L = scale_tril` is a
//! lower-triangular Cholesky factor.
//!
//! Supports three parameterizations — `scale_tril`, `covariance_matrix`, or
//! `precision_matrix` — but all are converted to `scale_tril` internally.
//!
//! [CL-331] ferrotorch#331 — multivariate distributions

use std::sync::Arc;

use ferrotorch_core::creation;
use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};

use crate::Distribution;

/// Multivariate normal distribution parameterized by a mean vector and a
/// lower-triangular scale matrix (Cholesky factor of the covariance).
///
/// # Construction
///
/// Use one of the three named constructors:
/// - [`MultivariateNormal::from_scale_tril`] — most efficient, no decomposition needed
/// - [`MultivariateNormal::from_covariance`] — computes Cholesky of the covariance
/// - [`MultivariateNormal::from_precision`] — inverts the precision via Cholesky
///
/// # Reparameterization
///
/// `rsample` uses the reparameterization trick:
/// ```text
/// z = loc + scale_tril @ eps,   eps ~ N(0, I)
/// ```
/// Gradients flow through `loc` and `scale_tril` via the autograd graph.
pub struct MultivariateNormal<T: Float> {
    loc: Tensor<T>,
    /// Lower-triangular Cholesky factor (d x d).
    scale_tril: Tensor<T>,
    /// Dimensionality of the distribution.
    d: usize,
}

impl<T: Float> MultivariateNormal<T> {
    /// Create from a lower-triangular Cholesky factor `L` such that
    /// `Sigma = L L^T`.
    ///
    /// `loc` must be 1-D with length `d`. `scale_tril` must be `[d, d]`.
    pub fn from_scale_tril(loc: Tensor<T>, scale_tril: Tensor<T>) -> FerrotorchResult<Self> {
        let loc_shape = loc.shape().to_vec();
        let tril_shape = scale_tril.shape().to_vec();

        if loc_shape.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("MultivariateNormal: loc must be 1-D, got shape {loc_shape:?}"),
            });
        }
        let d = loc_shape[0];
        if tril_shape != [d, d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MultivariateNormal: scale_tril must be [{d}, {d}], got {tril_shape:?}"
                ),
            });
        }

        Ok(Self { loc, scale_tril, d })
    }

    /// Create from a positive-definite covariance matrix.
    ///
    /// Internally computes the Cholesky decomposition `Sigma = L L^T`.
    pub fn from_covariance(loc: Tensor<T>, covariance_matrix: Tensor<T>) -> FerrotorchResult<Self> {
        let loc_shape = loc.shape().to_vec();
        let cov_shape = covariance_matrix.shape().to_vec();

        if loc_shape.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("MultivariateNormal: loc must be 1-D, got shape {loc_shape:?}"),
            });
        }
        let d = loc_shape[0];
        if cov_shape != [d, d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MultivariateNormal: covariance_matrix must be [{d}, {d}], got {cov_shape:?}"
                ),
            });
        }

        let scale_tril = cholesky_lower(&covariance_matrix, d)?;
        Ok(Self { loc, scale_tril, d })
    }

    /// Create from a positive-definite precision matrix `P = Sigma^{-1}`.
    ///
    /// Internally converts to `scale_tril` via Cholesky inversion.
    pub fn from_precision(loc: Tensor<T>, precision_matrix: Tensor<T>) -> FerrotorchResult<Self> {
        let loc_shape = loc.shape().to_vec();
        let prec_shape = precision_matrix.shape().to_vec();

        if loc_shape.len() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("MultivariateNormal: loc must be 1-D, got shape {loc_shape:?}"),
            });
        }
        let d = loc_shape[0];
        if prec_shape != [d, d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MultivariateNormal: precision_matrix must be [{d}, {d}], got {prec_shape:?}"
                ),
            });
        }

        let scale_tril = precision_to_scale_tril(&precision_matrix, d)?;
        Ok(Self { loc, scale_tril, d })
    }

    /// The mean vector.
    pub fn loc(&self) -> &Tensor<T> {
        &self.loc
    }

    /// The lower-triangular Cholesky factor.
    pub fn scale_tril(&self) -> &Tensor<T> {
        &self.scale_tril
    }

    /// Dimensionality of the distribution.
    pub fn dim(&self) -> usize {
        self.d
    }
}

impl<T: Float> Distribution<T> for MultivariateNormal<T> {
    fn sample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.scale_tril],
            "MultivariateNormal::sample",
        )?;
        let device = self.loc.device();
        let n: usize = shape.iter().product();
        let d = self.d;

        // eps ~ N(0, I), shape [n, d]
        let eps = creation::randn::<T>(&[n * d])?;
        let eps_data = eps.data_vec()?;

        let loc_data = self.loc.data_vec()?;
        let l_data = self.scale_tril.data_vec()?;

        // result[i] = loc + L @ eps[i]
        let mut result = Vec::with_capacity(n * d);
        for s in 0..n {
            for i in 0..d {
                let mut val = loc_data[i];
                // L is lower-triangular so L[i, j] = 0 for j > i
                for j in 0..=i {
                    val += l_data[i * d + j] * eps_data[s * d + j];
                }
                result.push(val);
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape.push(d);
        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn rsample(&self, shape: &[usize]) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.scale_tril],
            "MultivariateNormal::rsample",
        )?;
        let device = self.loc.device();
        let n: usize = shape.iter().product();
        let d = self.d;

        let eps = creation::randn::<T>(&[n * d])?;
        let eps_data = eps.data_vec()?;

        let loc_data = self.loc.data_vec()?;
        let l_data = self.scale_tril.data_vec()?;

        let mut result = Vec::with_capacity(n * d);
        for s in 0..n {
            for i in 0..d {
                let mut val = loc_data[i];
                for j in 0..=i {
                    val += l_data[i * d + j] * eps_data[s * d + j];
                }
                result.push(val);
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape.push(d);
        let storage = TensorStorage::cpu(result);

        let out = if (self.loc.requires_grad() || self.scale_tril.requires_grad())
            && ferrotorch_core::is_grad_enabled()
        {
            let grad_fn = Arc::new(MvnRsampleBackward {
                loc: self.loc.clone(),
                scale_tril: self.scale_tril.clone(),
                eps: eps.clone(),
                n,
                d,
            });
            Tensor::from_operation(storage, out_shape, grad_fn)?
        } else {
            Tensor::from_storage(storage, out_shape, false)?
        };
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn log_prob(&self, value: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.loc, &self.scale_tril, value],
            "MultivariateNormal::log_prob",
        )?;
        // log_prob = -0.5 * (d*log(2*pi) + mahal^2) - sum(log(diag(L)))
        //
        // mahal^2 = (x - mu)^T Sigma^{-1} (x - mu)
        //         = ||L^{-1} (x - mu)||^2
        let device = self.loc.device();
        let d = self.d;
        let loc_data = self.loc.data_vec()?;
        let l_data = self.scale_tril.data_vec()?;
        let val_data = value.data_vec()?;

        let n = val_data.len() / d;
        if val_data.len() != n * d {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "MultivariateNormal log_prob: value has {} elements, not divisible by d={}",
                    val_data.len(),
                    d
                ),
            });
        }

        let half = T::from(0.5).unwrap();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let d_log_2pi = T::from(d).unwrap() * two_pi.ln();

        // half_log_det = sum(log(diag(L)))
        let mut half_log_det = <T as num_traits::Zero>::zero();
        for i in 0..d {
            half_log_det += l_data[i * d + i].ln();
        }

        let mut result = Vec::with_capacity(n);
        for s in 0..n {
            // diff = x - mu
            let diff: Vec<T> = (0..d).map(|i| val_data[s * d + i] - loc_data[i]).collect();

            // Solve L y = diff for y (forward substitution)
            let mut y = vec![<T as num_traits::Zero>::zero(); d];
            for i in 0..d {
                let mut sum = diff[i];
                for j in 0..i {
                    sum = sum - l_data[i * d + j] * y[j];
                }
                y[i] = sum / l_data[i * d + i];
            }

            // mahal^2 = ||y||^2
            let mahal_sq: T = y
                .iter()
                .fold(<T as num_traits::Zero>::zero(), |acc, &v| acc + v * v);

            result.push(-half * (d_log_2pi + mahal_sq) - half_log_det);
        }

        // Output shape: all dims except the last event dim
        let val_shape = value.shape();
        let out_shape = if val_shape.len() > 1 {
            val_shape[..val_shape.len() - 1].to_vec()
        } else {
            vec![]
        };

        let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }

    fn entropy(&self) -> FerrotorchResult<Tensor<T>> {
        crate::fallback::check_gpu_fallback_opt_in(
            &[&self.scale_tril],
            "MultivariateNormal::entropy",
        )?;
        // H = 0.5 * d * (1 + log(2*pi)) + sum(log(diag(L)))
        let device = self.loc.device();
        let d = self.d;
        let l_data = self.scale_tril.data_vec()?;

        let half = T::from(0.5).unwrap();
        let one = <T as num_traits::One>::one();
        let two_pi = T::from(2.0 * std::f64::consts::PI).unwrap();
        let d_t = T::from(d).unwrap();

        let mut half_log_det = <T as num_traits::Zero>::zero();
        for i in 0..d {
            half_log_det += l_data[i * d + i].ln();
        }

        let h = half * d_t * (one + two_pi.ln()) + half_log_det;

        let out = Tensor::from_storage(TensorStorage::cpu(vec![h]), vec![], false)?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition (CPU, for small matrices in distribution params)
// ---------------------------------------------------------------------------

/// Compute the lower-triangular Cholesky factor of a `d x d` symmetric
/// positive-definite matrix stored row-major.
fn cholesky_lower<T: Float>(matrix: &Tensor<T>, d: usize) -> FerrotorchResult<Tensor<T>> {
    let a = matrix.data_vec()?;
    let mut l = vec![<T as num_traits::Zero>::zero(); d * d];

    for i in 0..d {
        for j in 0..=i {
            let mut sum = <T as num_traits::Zero>::zero();
            for k in 0..j {
                sum += l[i * d + k] * l[j * d + k];
            }
            if i == j {
                let diag = a[i * d + i] - sum;
                if diag <= <T as num_traits::Zero>::zero() {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "MultivariateNormal: covariance matrix is not positive definite"
                            .into(),
                    });
                }
                l[i * d + j] = diag.sqrt();
            } else {
                l[i * d + j] = (a[i * d + j] - sum) / l[j * d + j];
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(l), vec![d, d], false)
}

/// Convert a precision matrix to a lower-triangular Cholesky factor of the
/// covariance.
///
/// Uses the algorithm: L = solve_triangular(chol(flip(P)), I).
/// For small d this is simple direct inversion.
fn precision_to_scale_tril<T: Float>(
    precision: &Tensor<T>,
    d: usize,
) -> FerrotorchResult<Tensor<T>> {
    let p = precision.data_vec()?;

    // Invert precision to get covariance, then Cholesky.
    // For distribution-sized matrices this is fine.
    let cov = invert_symmetric_pd(&p, d)?;
    let cov_tensor = Tensor::from_storage(TensorStorage::cpu(cov), vec![d, d], false)?;
    cholesky_lower(&cov_tensor, d)
}

/// Invert a symmetric positive-definite matrix via Cholesky: A^{-1} = (L^{-1})^T (L^{-1}).
fn invert_symmetric_pd<T: Float>(a: &[T], d: usize) -> FerrotorchResult<Vec<T>> {
    let zero = <T as num_traits::Zero>::zero();

    // Cholesky: A = L L^T
    let mut l = vec![zero; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut sum = zero;
            for k in 0..j {
                sum += l[i * d + k] * l[j * d + k];
            }
            if i == j {
                let diag = a[i * d + i] - sum;
                if diag <= zero {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "precision matrix is not positive definite".into(),
                    });
                }
                l[i * d + j] = diag.sqrt();
            } else {
                l[i * d + j] = (a[i * d + j] - sum) / l[j * d + j];
            }
        }
    }

    // Forward-solve L Y = I for Y = L^{-1}
    let mut l_inv = vec![zero; d * d];
    let one = <T as num_traits::One>::one();
    for col in 0..d {
        for row in col..d {
            let mut val = if row == col { one } else { zero };
            for k in col..row {
                val = val - l[row * d + k] * l_inv[k * d + col];
            }
            l_inv[row * d + col] = val / l[row * d + row];
        }
    }

    // A^{-1} = L^{-T} L^{-1}
    let mut result = vec![zero; d * d];
    for i in 0..d {
        for j in 0..=i {
            let mut val = zero;
            for k in i.max(j)..d {
                val += l_inv[k * d + i] * l_inv[k * d + j];
            }
            result[i * d + j] = val;
            result[j * d + i] = val;
        }
    }

    Ok(result)
}

// ---------------------------------------------------------------------------
// Backward nodes
// ---------------------------------------------------------------------------

/// Backward for MVN rsample: `z = loc + L @ eps`.
///
/// - d(z)/d(loc)       = I  (summed over samples)
/// - d(z)/d(scale_tril) = outer products of grad with eps
#[derive(Debug)]
struct MvnRsampleBackward<T: Float> {
    loc: Tensor<T>,
    scale_tril: Tensor<T>,
    eps: Tensor<T>,
    n: usize,
    d: usize,
}

impl<T: Float> GradFn<T> for MvnRsampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let eps_data = self.eps.data_vec()?;
        let d = self.d;
        let n = self.n;
        let zero = <T as num_traits::Zero>::zero();

        // grad_loc[i] = sum over samples of grad_output[s, i]
        let mut grad_loc = vec![zero; d];
        for s in 0..n {
            for i in 0..d {
                grad_loc[i] += go[s * d + i];
            }
        }
        let grad_loc_t = Tensor::from_storage(
            TensorStorage::cpu(grad_loc),
            self.loc.shape().to_vec(),
            false,
        )?;
        let grad_loc_t = if device.is_cuda() {
            grad_loc_t.to(device)?
        } else {
            grad_loc_t
        };

        // grad_scale_tril[i, j] = sum_s grad_output[s, i] * eps[s, j]  (lower-tri only)
        let mut grad_l = vec![zero; d * d];
        for s in 0..n {
            for i in 0..d {
                for j in 0..=i {
                    grad_l[i * d + j] += go[s * d + i] * eps_data[s * d + j];
                }
            }
        }
        let grad_l_t = Tensor::from_storage(
            TensorStorage::cpu(grad_l),
            self.scale_tril.shape().to_vec(),
            false,
        )?;
        let grad_l_t = if device.is_cuda() {
            grad_l_t.to(device)?
        } else {
            grad_l_t
        };

        Ok(vec![
            if self.loc.requires_grad() {
                Some(grad_loc_t)
            } else {
                None
            },
            if self.scale_tril.requires_grad() {
                Some(grad_l_t)
            } else {
                None
            },
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.loc, &self.scale_tril]
    }

    fn name(&self) -> &'static str {
        "MvnRsampleBackward"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::{from_slice, scalar, tensor};

    fn eye_2x2() -> Tensor<f32> {
        from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap()
    }

    #[test]
    fn test_mvn_sample_shape() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let samples = dist.sample(&[100]).unwrap();
        assert_eq!(samples.shape(), &[100, 2]);
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_mvn_sample_2d_shape() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let samples = dist.sample(&[5, 10]).unwrap();
        assert_eq!(samples.shape(), &[5, 10, 2]);
    }

    #[test]
    fn test_mvn_rsample_has_grad() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap().requires_grad_(true);
        let l = from_slice(&[1.0f32, 0.0, 0.5, 1.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert_eq!(samples.shape(), &[5, 2]);
        assert!(samples.requires_grad());
        assert!(samples.grad_fn().is_some());
    }

    #[test]
    fn test_mvn_rsample_no_grad_when_detached() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let samples = dist.rsample(&[5]).unwrap();
        assert!(!samples.requires_grad());
    }

    #[test]
    fn test_mvn_log_prob_standard_at_mean() {
        // log_prob at mean for N(0, I) in d=2:
        // = -0.5 * (2 * log(2*pi) + 0) - 0 = -log(2*pi)
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let x = tensor(&[0.0f32, 0.0]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_mvn_log_prob_batch() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        // Two points: [0,0] and [1,0]
        let x = from_slice(&[0.0f32, 0.0, 1.0, 0.0], &[2, 2]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert_eq!(lp.shape(), &[2]);

        let data = lp.data().unwrap();
        // lp at mean should be greater than lp away from mean
        assert!(data[0] > data[1]);
    }

    #[test]
    fn test_mvn_from_covariance() {
        let loc = tensor(&[1.0f32, 2.0]).unwrap();
        let cov = from_slice(&[4.0f32, 1.0, 1.0, 2.0], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_covariance(loc, cov).unwrap();
        assert_eq!(dist.dim(), 2);

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50, 2]);
    }

    #[test]
    fn test_mvn_from_precision() {
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        // precision = identity => covariance = identity
        let prec = from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_precision(loc, prec).unwrap();

        let x = tensor(&[0.0f32, 0.0]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(2.0f32 * std::f32::consts::PI).ln();
        assert!(
            (lp.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            lp.item().unwrap()
        );
    }

    #[test]
    fn test_mvn_entropy_standard() {
        // entropy of N(0, I) in d=2: 0.5 * d * (1 + log(2*pi)) + 0
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, eye_2x2()).unwrap();

        let h = dist.entropy().unwrap();
        let expected = 0.5 * 2.0 * (1.0 + (2.0f32 * std::f32::consts::PI).ln());
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_mvn_entropy_scaled() {
        // scale_tril = [[2, 0], [0, 3]] => det = 6, log_det/2 = ln(2) + ln(3)
        let loc = tensor(&[0.0f32, 0.0]).unwrap();
        let l = from_slice(&[2.0f32, 0.0, 0.0, 3.0], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();

        let h = dist.entropy().unwrap();
        let expected =
            0.5 * 2.0 * (1.0 + (2.0f32 * std::f32::consts::PI).ln()) + 2.0f32.ln() + 3.0f32.ln();
        assert!(
            (h.item().unwrap() - expected).abs() < 1e-4,
            "expected {expected}, got {}",
            h.item().unwrap()
        );
    }

    #[test]
    fn test_mvn_rsample_backward() {
        let loc = tensor(&[1.0f32, 2.0]).unwrap().requires_grad_(true);
        let l = from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[2, 2])
            .unwrap()
            .requires_grad_(true);
        let dist = MultivariateNormal::from_scale_tril(loc.clone(), l.clone()).unwrap();

        let z = dist.rsample(&[10]).unwrap();
        let loss = z.sum_all().unwrap();
        loss.backward().unwrap();

        // d(sum(loc + I @ eps))/d(loc) = [10, 10] (each component summed 10 times)
        let loc_grad = loc.grad().unwrap().unwrap();
        let grad_data = loc_grad.data().unwrap();
        assert!(
            (grad_data[0] - 10.0).abs() < 1e-3,
            "expected loc_grad[0]=10.0, got {}",
            grad_data[0]
        );
        assert!(
            (grad_data[1] - 10.0).abs() < 1e-3,
            "expected loc_grad[1]=10.0, got {}",
            grad_data[1]
        );

        let l_grad = l.grad().unwrap().unwrap();
        assert!(l_grad.data().unwrap().iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_mvn_shape_mismatch_loc() {
        let loc = scalar(0.0f32).unwrap(); // 0-D
        let l = eye_2x2();
        assert!(MultivariateNormal::from_scale_tril(loc, l).is_err());
    }

    #[test]
    fn test_mvn_shape_mismatch_tril() {
        let loc = tensor(&[0.0f32, 0.0, 0.0]).unwrap(); // d=3
        let l = eye_2x2(); // 2x2
        assert!(MultivariateNormal::from_scale_tril(loc, l).is_err());
    }

    #[test]
    fn test_mvn_f64() {
        let loc = tensor(&[0.0f64, 0.0]).unwrap();
        let l = from_slice(&[1.0f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();

        let samples = dist.sample(&[50]).unwrap();
        assert_eq!(samples.shape(), &[50, 2]);

        let x = tensor(&[0.0f64, 0.0]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        let expected = -(2.0f64 * std::f64::consts::PI).ln();
        assert!((lp.item().unwrap() - expected).abs() < 1e-10);
    }

    #[test]
    fn test_mvn_3d() {
        // Test with d=3
        let loc = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        let l = from_slice(&[2.0f32, 0.0, 0.0, 0.5, 1.5, 0.0, 0.3, 0.2, 1.0], &[3, 3]).unwrap();
        let dist = MultivariateNormal::from_scale_tril(loc, l).unwrap();
        assert_eq!(dist.dim(), 3);

        let samples = dist.sample(&[20]).unwrap();
        assert_eq!(samples.shape(), &[20, 3]);

        let x = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        let lp = dist.log_prob(&x).unwrap();
        assert!(lp.item().unwrap().is_finite());
    }
}
