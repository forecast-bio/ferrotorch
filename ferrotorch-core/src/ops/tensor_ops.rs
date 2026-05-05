//! Common tensor manipulation operations.
//!
//! - [`triu`] / [`tril`] — upper/lower triangular masks
//! - [`diag`] / [`diagflat`] — diagonal extraction/construction
//! - [`roll`] — circular shift along a dimension
//! - [`cdist`] — pairwise distance matrix

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Upper triangular part of a 2-D tensor.
///
/// Elements below the `diagonal`-th diagonal are set to zero.
/// `diagonal=0` is the main diagonal, `diagonal>0` is above, `diagonal<0` is below.
///
/// Matches PyTorch's `torch.triu`.
pub fn triu<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("triu: expected 2-D tensor, got shape {:?}", input.shape()),
        });
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "triu" });
    }

    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let data = input.data()?;
    let zero = <T as num_traits::Zero>::zero();

    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            if (c as i64) >= (r as i64) + diagonal {
                out.push(data[r * cols + c]);
            } else {
                out.push(zero);
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![rows, cols], false)
}

/// Lower triangular part of a 2-D tensor.
///
/// Elements above the `diagonal`-th diagonal are set to zero.
///
/// Matches PyTorch's `torch.tril`.
pub fn tril<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("tril: expected 2-D tensor, got shape {:?}", input.shape()),
        });
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "tril" });
    }

    let rows = input.shape()[0];
    let cols = input.shape()[1];
    let data = input.data()?;
    let zero = <T as num_traits::Zero>::zero();

    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        for c in 0..cols {
            if (c as i64) <= (r as i64) + diagonal {
                out.push(data[r * cols + c]);
            } else {
                out.push(zero);
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![rows, cols], false)
}

/// Extract the diagonal of a 2-D tensor, or construct a 2-D diagonal matrix
/// from a 1-D tensor.
///
/// - If `input` is 2-D: returns the `diagonal`-th diagonal as a 1-D tensor.
/// - If `input` is 1-D: returns a 2-D tensor with `input` on the `diagonal`-th diagonal.
///
/// Matches PyTorch's `torch.diag`.
pub fn diag<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "diag" });
    }

    match input.ndim() {
        1 => {
            // 1-D → 2-D diagonal matrix
            let data = input.data()?;
            let n = data.len();
            let offset = diagonal.unsigned_abs() as usize;
            let size = n + offset;
            let zero = <T as num_traits::Zero>::zero();
            let mut out = vec![zero; size * size];

            for (i, &val) in data[..n].iter().enumerate() {
                let (r, c) = if diagonal >= 0 {
                    (i, i + offset)
                } else {
                    (i + offset, i)
                };
                out[r * size + c] = val;
            }

            Tensor::from_storage(TensorStorage::cpu(out), vec![size, size], false)
        }
        2 => {
            // 2-D → extract diagonal
            let rows = input.shape()[0];
            let cols = input.shape()[1];
            let data = input.data()?;

            let (start_r, start_c) = if diagonal >= 0 {
                (0, diagonal as usize)
            } else {
                ((-diagonal) as usize, 0)
            };

            let diag_len = (rows - start_r).min(cols - start_c);
            let mut out = Vec::with_capacity(diag_len);
            for i in 0..diag_len {
                out.push(data[(start_r + i) * cols + (start_c + i)]);
            }

            Tensor::from_storage(TensorStorage::cpu(out), vec![diag_len], false)
        }
        _ => Err(FerrotorchError::InvalidArgument {
            message: format!("diag: expected 1-D or 2-D tensor, got {:?}", input.shape()),
        }),
    }
}

/// Construct a diagonal matrix from a 1-D tensor (flattened if needed).
///
/// Like `diag` with a 1-D input, but first flattens multi-dimensional input.
///
/// Matches PyTorch's `torch.diagflat`.
pub fn diagflat<T: Float>(input: &Tensor<T>, diagonal: i64) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "diagflat" });
    }

    let flat = if input.ndim() == 1 {
        input.clone()
    } else {
        let data = input.data_vec()?;
        let n = data.len();
        Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)?
    };

    diag(&flat, diagonal)
}

/// Roll (circular shift) a tensor along a dimension.
///
/// Elements shifted past the last position wrap to the beginning.
///
/// Matches PyTorch's `torch.roll`.
pub fn roll<T: Float>(input: &Tensor<T>, shifts: i64, dim: usize) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "roll" });
    }
    let shape = input.shape();
    if dim >= shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("roll: dim {dim} out of range for shape {:?}", shape),
        });
    }

    let data = input.data_vec()?;
    let numel = data.len();
    let dim_size = shape[dim] as i64;
    let shift = ((shifts % dim_size) + dim_size) % dim_size; // normalize to positive

    if shift == 0 {
        return Ok(input.clone());
    }

    let inner: usize = shape[dim + 1..].iter().product();
    let outer: usize = numel / (shape[dim] * inner);
    let mut out = vec![<T as num_traits::Zero>::zero(); numel];

    for o in 0..outer {
        for d in 0..shape[dim] {
            let new_d = ((d as i64 + shift) % dim_size) as usize;
            for i in 0..inner {
                let src = o * shape[dim] * inner + d * inner + i;
                let dst = o * shape[dim] * inner + new_d * inner + i;
                out[dst] = data[src];
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), shape.to_vec(), false)
}

/// Pairwise distance matrix between two sets of vectors.
///
/// `x1` has shape `[B, P, M]`, `x2` has shape `[B, R, M]`.
/// Returns shape `[B, P, R]` with Lp distances.
///
/// If `x1` is 2-D `[P, M]` and `x2` is 2-D `[R, M]`, returns `[P, R]`.
///
/// Matches PyTorch's `torch.cdist`.
pub fn cdist<T: Float>(x1: &Tensor<T>, x2: &Tensor<T>, p: f64) -> FerrotorchResult<Tensor<T>> {
    if x1.is_cuda() || x2.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "cdist" });
    }

    let (batched, b, p_dim, r_dim, m) = match (x1.ndim(), x2.ndim()) {
        (2, 2) => {
            let p_dim = x1.shape()[0];
            let m1 = x1.shape()[1];
            let r_dim = x2.shape()[0];
            let m2 = x2.shape()[1];
            if m1 != m2 {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!("cdist: feature dims mismatch: {} vs {}", m1, m2),
                });
            }
            (false, 1, p_dim, r_dim, m1)
        }
        (3, 3) => {
            if x1.shape()[0] != x2.shape()[0] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "cdist: batch dims mismatch: {} vs {}",
                        x1.shape()[0],
                        x2.shape()[0]
                    ),
                });
            }
            if x1.shape()[2] != x2.shape()[2] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "cdist: feature dims mismatch: {} vs {}",
                        x1.shape()[2],
                        x2.shape()[2]
                    ),
                });
            }
            (
                true,
                x1.shape()[0],
                x1.shape()[1],
                x2.shape()[1],
                x1.shape()[2],
            )
        }
        _ => {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "cdist: expected 2-D or 3-D inputs, got {:?} and {:?}",
                    x1.shape(),
                    x2.shape()
                ),
            });
        }
    };

    let d1 = x1.data()?;
    let d2 = x2.data()?;
    let p_val = T::from(p).unwrap();
    let inv_p = T::from(1.0 / p).unwrap();
    let mut out = Vec::with_capacity(b * p_dim * r_dim);

    for batch in 0..b {
        let off1 = batch * p_dim * m;
        let off2 = batch * r_dim * m;
        for i in 0..p_dim {
            for j in 0..r_dim {
                let mut dist = <T as num_traits::Zero>::zero();
                for k in 0..m {
                    let diff = d1[off1 + i * m + k] - d2[off2 + j * m + k];
                    let abs_diff = if diff < <T as num_traits::Zero>::zero() {
                        <T as num_traits::Zero>::zero() - diff
                    } else {
                        diff
                    };
                    dist += abs_diff.powf(p_val);
                }
                out.push(dist.powf(inv_p));
            }
        }
    }

    let out_shape = if batched {
        vec![b, p_dim, r_dim]
    } else {
        vec![p_dim, r_dim]
    };

    Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t2d(data: &[f32], rows: usize, cols: usize) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![rows, cols], false).unwrap()
    }

    fn t1d(data: &[f32]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
    }

    #[test]
    fn test_triu_main_diagonal() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let result = triu(&input, 0).unwrap();
        assert_eq!(
            result.data().unwrap(),
            &[1.0, 2.0, 3.0, 0.0, 5.0, 6.0, 0.0, 0.0, 9.0]
        );
    }

    #[test]
    fn test_tril_main_diagonal() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let result = tril(&input, 0).unwrap();
        assert_eq!(
            result.data().unwrap(),
            &[1.0, 0.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 9.0]
        );
    }

    #[test]
    fn test_triu_positive_diagonal() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let result = triu(&input, 1).unwrap();
        assert_eq!(
            result.data().unwrap(),
            &[0.0, 2.0, 3.0, 0.0, 0.0, 6.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_diag_extract() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], 3, 3);
        let result = diag(&input, 0).unwrap();
        assert_eq!(result.data().unwrap(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_diag_construct() {
        let input = t1d(&[1.0, 2.0, 3.0]);
        let result = diag(&input, 0).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(
            result.data().unwrap(),
            &[1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0]
        );
    }

    #[test]
    fn test_diag_off_diagonal() {
        let input = t1d(&[1.0, 2.0]);
        let result = diag(&input, 1).unwrap();
        assert_eq!(result.shape(), &[3, 3]);
        assert_eq!(
            result.data().unwrap(),
            &[0.0, 1.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_roll_basic() {
        let input = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = roll(&input, 2, 0).unwrap();
        assert_eq!(result.data().unwrap(), &[4.0, 5.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_roll_negative() {
        let input = t1d(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let result = roll(&input, -1, 0).unwrap();
        assert_eq!(result.data().unwrap(), &[2.0, 3.0, 4.0, 5.0, 1.0]);
    }

    #[test]
    fn test_cdist_l2() {
        let x1 = t2d(&[0.0, 0.0, 1.0, 0.0, 0.0, 1.0], 3, 2);
        let x2 = t2d(&[1.0, 1.0], 1, 2);
        let result = cdist(&x1, &x2, 2.0).unwrap();
        assert_eq!(result.shape(), &[3, 1]);
        let d = result.data().unwrap();
        assert!((d[0] - 2.0f32.sqrt()).abs() < 1e-5); // dist([0,0],[1,1]) = sqrt(2)
        assert!((d[1] - 1.0).abs() < 1e-5); // dist([1,0],[1,1]) = 1
        assert!((d[2] - 1.0).abs() < 1e-5); // dist([0,1],[1,1]) = 1
    }

    #[test]
    // reason: diagflat is pure indexing — each input element is copied to
    // a diagonal slot without arithmetic, so the bit pattern is preserved
    // and equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_diagflat() {
        let input = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let result = diagflat(&input, 0).unwrap();
        assert_eq!(result.shape(), &[4, 4]);
        let d = result.data().unwrap();
        assert_eq!(d[0], 1.0);
        assert_eq!(d[5], 2.0);
        assert_eq!(d[10], 3.0);
        assert_eq!(d[15], 4.0);
    }
}
