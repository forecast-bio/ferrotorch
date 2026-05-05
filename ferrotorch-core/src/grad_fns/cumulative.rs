//! Backward functions and differentiable wrappers for cumulative (scan) ops.
//!
//! - `cumsum`       — backward is reverse cumsum of the gradient
//! - `cumprod`      — backward uses the forward output and prefix/suffix products
//! - `cummax`       — non-differentiable (returns indices); no backward
//! - `cummin`       — non-differentiable (returns indices); no backward
//! - `logcumsumexp` — backward via softmax-weighted reverse cumsum
//!
//! [CL-306]

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::ops::cumulative::{
    CumExtremeResult, cummax_forward, cummin_forward, cumprod_forward, cumsum_forward,
    logcumsumexp_forward, reverse_cumsum,
};
use crate::shape::normalize_axis;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// CumsumBackward
// ---------------------------------------------------------------------------

/// Backward node for `cumsum(input, dim)`.
///
/// VJP: `grad_input = reverse_cumsum(grad_output, dim)`.
///
/// Intuition: cumsum is a lower-triangular matrix multiply along dim.
/// Its transpose is the upper-triangular (reverse cumsum).
#[derive(Debug)]
pub struct CumsumBackward<T: Float> {
    input: Tensor<T>,
    dim: usize,
}

impl<T: Float> GradFn<T> for CumsumBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(crate::error::FerrotorchError::NotImplementedOnCuda {
                op: "CumsumBackward",
            });
        }
        let go_data = grad_output.data()?;
        let shape = grad_output.shape();

        let grad_data = reverse_cumsum(go_data, shape, self.dim);

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(grad_data), shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CumsumBackward"
    }
}

/// Differentiable cumulative sum along `dim`.
///
/// When gradient tracking is enabled, the returned tensor carries a
/// [`CumsumBackward`] node.
///
/// `dim` supports negative indexing.
pub fn cumsum<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let norm_dim = normalize_axis(dim as isize, input.ndim())?;
    let result = cumsum_forward(input, dim)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(CumsumBackward {
            input: input.clone(),
            dim: norm_dim,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// CumprodBackward
// ---------------------------------------------------------------------------

/// Backward node for `cumprod(input, dim)`.
///
/// VJP for cumprod is:
///   `grad_input[i] = sum_{j >= i} grad_output[j] * cumprod_output[j] / input[i]`
///
/// To handle zeros safely, we use the identity:
///   `grad_input[i] = (1/input[i]) * reverse_cumsum(grad_output * cumprod_output, dim)[i]`
///
/// When `input[i] == 0`, we recompute using prefix/suffix products along
/// the scan direction to avoid division by zero.
#[derive(Debug)]
pub struct CumprodBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
    dim: usize,
}

impl<T: Float> GradFn<T> for CumprodBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.input.is_cuda() || self.output.is_cuda() {
            return Err(crate::error::FerrotorchError::NotImplementedOnCuda {
                op: "CumprodBackward",
            });
        }

        let go_data = grad_output.data()?;
        let in_data = self.input.data()?;
        let out_data = self.output.data()?;
        let shape = self.input.shape();

        let (outer, dim_size, inner) = dim_strides(shape, self.dim);
        let numel = in_data.len();
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); numel];

        for o in 0..outer {
            for k in 0..inner {
                let base = o * dim_size * inner + k;

                // Check if any element along this scan line is zero.
                let has_zero = (0..dim_size)
                    .any(|i| in_data[base + i * inner] == <T as num_traits::Zero>::zero());

                if has_zero {
                    // Slow path: zeros present. Use prefix/suffix product
                    // approach to avoid division by zero.
                    //
                    // For each position i:
                    //   grad_input[i] = sum_{j >= i} go[j] * prod_{k in [i..=j], k != i} input[k]
                    //                 = sum_{j >= i} go[j] * (cumprod[j] / cumprod[i-1]) / input[i]
                    //
                    // But with zeros this is fragile, so we just brute-force
                    // the partial products for correctness.
                    for i in 0..dim_size {
                        let mut acc = <T as num_traits::Zero>::zero();
                        for j in i..dim_size {
                            // Compute product of input[k] for k in [0..=j], excluding k=i.
                            let mut partial = <T as num_traits::One>::one();
                            for kk in 0..=j {
                                if kk != i {
                                    #[allow(clippy::assign_op_pattern)]
                                    {
                                        partial = partial * in_data[base + kk * inner];
                                    }
                                }
                            }
                            acc += go_data[base + j * inner] * partial;
                        }
                        grad_input[base + i * inner] = acc;
                    }
                } else {
                    // Fast path: no zeros, safe to use output / input.
                    // grad_input[i] = reverse_cumsum(grad_output * output)[i] / input[i]
                    //
                    // We compute `product = go * out` then reverse-cumsum it,
                    // then divide each element by input[i].
                    let mut product = vec![<T as num_traits::Zero>::zero(); dim_size];
                    for (i, prod_elem) in product.iter_mut().enumerate().take(dim_size) {
                        let idx = base + i * inner;
                        *prod_elem = go_data[idx] * out_data[idx];
                    }
                    // Reverse cumsum of product.
                    let mut rev_acc = <T as num_traits::Zero>::zero();
                    for i in (0..dim_size).rev() {
                        let idx = base + i * inner;
                        rev_acc += product[i];
                        grad_input[idx] = rev_acc / in_data[idx];
                    }
                }
            }
        }

        let result = Tensor::from_storage(TensorStorage::cpu(grad_input), shape.to_vec(), false)?;
        Ok(vec![Some(result)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CumprodBackward"
    }
}

/// Differentiable cumulative product along `dim`.
///
/// When gradient tracking is enabled, the returned tensor carries a
/// [`CumprodBackward`] node.
///
/// `dim` supports negative indexing.
pub fn cumprod<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let norm_dim = normalize_axis(dim as isize, input.ndim())?;
    let result = cumprod_forward(input, dim)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(CumprodBackward {
            input: input.clone(),
            output: result.clone(),
            dim: norm_dim,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// cummax / cummin (non-differentiable)
// ---------------------------------------------------------------------------

/// Cumulative maximum along `dim`.
///
/// Returns `(values, indices)` where `values` has the running maximum and
/// `indices` has the position (along `dim`) where each maximum was attained.
///
/// This operation is **not differentiable** — the returned values tensor does
/// not carry a gradient function.
pub fn cummax<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<CumExtremeResult<T>> {
    cummax_forward(input, dim)
}

/// Cumulative minimum along `dim`.
///
/// Returns `(values, indices)` analogous to [`cummax`] but tracking the
/// running minimum.
///
/// This operation is **not differentiable**.
pub fn cummin<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<CumExtremeResult<T>> {
    cummin_forward(input, dim)
}

// ---------------------------------------------------------------------------
// LogcumsumexpBackward
// ---------------------------------------------------------------------------

/// Backward node for `logcumsumexp(input, dim)`.
///
/// VJP: `grad_input[i] = sum_{j >= i} grad_output[j] * softmax_weight(i, j)`
///
/// where `softmax_weight(i, j) = exp(input[i] - logcumsumexp_output[j])`.
///
/// This is equivalent to:
///   `grad_input = reverse_cumsum(grad_output * exp(input - output), dim)`
///   Wait — that's not quite right because the output at position j uses input
///   positions 0..=j. The correct form is:
///   `grad_input[i] = exp(input[i]) * reverse_cumsum(grad_output * exp(-output), dim)[i]`
///
/// Which factors as: `grad_input = exp(input) * reverse_cumsum(grad_output / exp(output))`.
/// Since `exp(output)` = `cumsumexp(input)`, this becomes:
///   `grad_input = exp(input) * reverse_cumsum(grad_output * exp(-output))`
#[derive(Debug)]
pub struct LogcumsumexpBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
    dim: usize,
}

impl<T: Float> GradFn<T> for LogcumsumexpBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.input.is_cuda() || self.output.is_cuda() {
            return Err(crate::error::FerrotorchError::NotImplementedOnCuda {
                op: "LogcumsumexpBackward",
            });
        }

        let go_data = grad_output.data()?;
        let in_data = self.input.data()?;
        let out_data = self.output.data()?;
        let shape = self.input.shape();

        // Compute: product[i] = grad_output[i] * exp(-output[i])
        let product: Vec<T> = go_data
            .iter()
            .zip(out_data.iter())
            .map(|(&g, &o)| g * (-o).exp())
            .collect();

        // Reverse cumsum of product along dim.
        let rev = reverse_cumsum(&product, shape, self.dim);

        // grad_input[i] = exp(input[i]) * rev[i]
        let grad_data: Vec<T> = in_data
            .iter()
            .zip(rev.iter())
            .map(|(&x, &r)| x.exp() * r)
            .collect();

        let grad_input =
            Tensor::from_storage(TensorStorage::cpu(grad_data), shape.to_vec(), false)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LogcumsumexpBackward"
    }
}

/// Differentiable log-cumulative-sum-exp along `dim`.
///
/// `output[..., i, ...] = log(sum(exp(input[..., 0..=i, ...])))`
///
/// When gradient tracking is enabled, the returned tensor carries a
/// [`LogcumsumexpBackward`] node.
///
/// `dim` supports negative indexing.
pub fn logcumsumexp<T: Float>(input: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    let norm_dim = normalize_axis(dim as isize, input.ndim())?;
    let result = logcumsumexp_forward(input, dim)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(LogcumsumexpBackward {
            input: input.clone(),
            output: result.clone(),
            dim: norm_dim,
        });
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Helpers (re-used from ops::cumulative internally)
// ---------------------------------------------------------------------------

/// Compute (outer_size, dim_size, inner_size) — mirrors the one in `ops::cumulative`.
fn dim_strides(shape: &[usize], dim: usize) -> (usize, usize, usize) {
    let outer: usize = shape[..dim].iter().product();
    let dim_size = shape[dim];
    let inner: usize = shape[dim + 1..].iter().product();
    (outer, dim_size, inner)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad::no_grad;
    use crate::grad_fns::reduction::sum;
    use crate::storage::TensorStorage;

    /// Helper: create a leaf tensor.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // =======================================================================
    // cumsum forward
    // =======================================================================

    #[test]
    fn test_cumsum_1d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let cs = cumsum(&x, 0).unwrap();
        assert_eq!(cs.shape(), &[4]);
        let d = cs.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
        assert!((d[3] - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum_2d_dim0() {
        // [[1, 2, 3], [4, 5, 6]] cumsum along dim 0
        // -> [[1, 2, 3], [5, 7, 9]]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cs = cumsum(&x, 0).unwrap();
        assert_eq!(cs.shape(), &[2, 3]);
        let d = cs.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        assert!((d[3] - 5.0).abs() < 1e-12);
        assert!((d[4] - 7.0).abs() < 1e-12);
        assert!((d[5] - 9.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum_2d_dim1() {
        // [[1, 2, 3], [4, 5, 6]] cumsum along dim 1
        // -> [[1, 3, 6], [4, 9, 15]]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cs = cumsum(&x, 1).unwrap();
        let d = cs.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
        assert!((d[3] - 4.0).abs() < 1e-12);
        assert!((d[4] - 9.0).abs() < 1e-12);
        assert!((d[5] - 15.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum_negative_dim() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cs = cumsum(&x, -1).unwrap();
        let d = cs.data().unwrap();
        // Same as dim=1.
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumsum_3d() {
        // shape [2, 2, 3], cumsum along dim=1
        let data: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let x = leaf(&data, &[2, 2, 3], false);
        let cs = cumsum(&x, 1).unwrap();
        assert_eq!(cs.shape(), &[2, 2, 3]);
        let d = cs.data().unwrap();
        // First slice: [[1,2,3],[4,5,6]] -> [[1,2,3],[5,7,9]]
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[3] - 5.0).abs() < 1e-12);
        assert!((d[4] - 7.0).abs() < 1e-12);
        assert!((d[5] - 9.0).abs() < 1e-12);
    }

    // =======================================================================
    // cumsum backward
    // =======================================================================

    #[test]
    fn test_cumsum_backward_1d() {
        // cumsum([a, b, c]) = [a, a+b, a+b+c]
        // d(sum of cumsum)/da = 3, /db = 2, /dc = 1
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cs = cumsum(&x, 0).unwrap();
        let loss = sum(&cs).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 3.0).abs() < 1e-12, "got {}", gd[0]);
        assert!((gd[1] - 2.0).abs() < 1e-12, "got {}", gd[1]);
        assert!((gd[2] - 1.0).abs() < 1e-12, "got {}", gd[2]);
    }

    #[test]
    fn test_cumsum_backward_2d_dim0() {
        // x: [[1, 2], [3, 4]], cumsum(dim=0) -> [[1, 2], [4, 6]]
        // loss = sum = 1+2+4+6 = 13
        // d/dx[0,0] = d(1)/dx[0,0] + d(4)/dx[0,0] = 1 + 1 = 2
        // d/dx[0,1] = d(2)/dx[0,1] + d(6)/dx[0,1] = 1 + 1 = 2
        // d/dx[1,0] = d(4)/dx[1,0] = 1
        // d/dx[1,1] = d(6)/dx[1,1] = 1
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], true);
        let cs = cumsum(&x, 0).unwrap();
        let loss = sum(&cs).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 2.0).abs() < 1e-12, "got {}", gd[0]);
        assert!((gd[1] - 2.0).abs() < 1e-12, "got {}", gd[1]);
        assert!((gd[2] - 1.0).abs() < 1e-12, "got {}", gd[2]);
        assert!((gd[3] - 1.0).abs() < 1e-12, "got {}", gd[3]);
    }

    #[test]
    fn test_cumsum_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cs = cumsum(&x, 0).unwrap();
        assert!(cs.grad_fn().is_some());
        assert_eq!(cs.grad_fn().unwrap().name(), "CumsumBackward");
    }

    #[test]
    fn test_cumsum_no_grad_fn_when_not_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let cs = cumsum(&x, 0).unwrap();
        assert!(cs.grad_fn().is_none());
    }

    #[test]
    fn test_cumsum_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cs = no_grad(|| cumsum(&x, 0)).unwrap();
        assert!(cs.grad_fn().is_none());
    }

    // =======================================================================
    // cumprod forward
    // =======================================================================

    #[test]
    fn test_cumprod_1d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let cp = cumprod(&x, 0).unwrap();
        let d = cp.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
        assert!((d[3] - 24.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumprod_2d_dim0() {
        // [[1, 2, 3], [4, 5, 6]] cumprod dim 0
        // -> [[1, 2, 3], [4, 10, 18]]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cp = cumprod(&x, 0).unwrap();
        let d = cp.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        assert!((d[3] - 4.0).abs() < 1e-12);
        assert!((d[4] - 10.0).abs() < 1e-12);
        assert!((d[5] - 18.0).abs() < 1e-12);
    }

    #[test]
    fn test_cumprod_2d_dim1() {
        // [[1, 2, 3], [4, 5, 6]] cumprod dim 1
        // -> [[1, 2, 6], [4, 20, 120]]
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let cp = cumprod(&x, 1).unwrap();
        let d = cp.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 6.0).abs() < 1e-12);
        assert!((d[3] - 4.0).abs() < 1e-12);
        assert!((d[4] - 20.0).abs() < 1e-12);
        assert!((d[5] - 120.0).abs() < 1e-12);
    }

    // =======================================================================
    // cumprod backward
    // =======================================================================

    #[test]
    fn test_cumprod_backward_1d() {
        // cumprod([a, b, c]) = [a, ab, abc]
        // loss = sum = a + ab + abc
        // d/da = 1 + b + bc = 1 + 2 + 6 = 9
        // d/db = 0 + a + ac = 0 + 1 + 3 = 4
        // d/dc = 0 + 0 + ab = 0 + 0 + 2 = 2
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cp = cumprod(&x, 0).unwrap();
        let loss = sum(&cp).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 9.0).abs() < 1e-10, "got {}", gd[0]);
        assert!((gd[1] - 4.0).abs() < 1e-10, "got {}", gd[1]);
        assert!((gd[2] - 2.0).abs() < 1e-10, "got {}", gd[2]);
    }

    #[test]
    fn test_cumprod_backward_with_zero() {
        // cumprod([2, 0, 3]) = [2, 0, 0]
        // loss = sum = 2 + 0 + 0 = 2
        // d/dx[0] = 1 + 0 + 0 = 1 (only first element of cumprod depends on x[0] non-zero)
        // Actually: cumprod = [x0, x0*x1, x0*x1*x2]
        // cumprod([2, 0, 3]) = [2, 0, 0]
        // d(cumprod[j])/d(input[i]) = prod_{k in 0..=j, k!=i} input[k]
        // d(loss)/dx0 = prod(empty) + prod(x1) + prod(x1,x2) = 1 + 0 + 0 = 1
        // d(loss)/dx1 = prod(x0) + prod(x0,x2) = 2 + 6 = 8
        // d(loss)/dx2 = prod(x0,x1) = 0
        let x = leaf(&[2.0, 0.0, 3.0], &[3], true);
        let cp = cumprod(&x, 0).unwrap();
        let loss = sum(&cp).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 1.0).abs() < 1e-10, "d/dx[0]: got {}", gd[0]);
        assert!((gd[1] - 8.0).abs() < 1e-10, "d/dx[1]: got {}", gd[1]);
        assert!((gd[2] - 0.0).abs() < 1e-10, "d/dx[2]: got {}", gd[2]);
    }

    #[test]
    fn test_cumprod_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cp = cumprod(&x, 0).unwrap();
        assert!(cp.grad_fn().is_some());
        assert_eq!(cp.grad_fn().unwrap().name(), "CumprodBackward");
    }

    #[test]
    fn test_cumprod_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let cp = no_grad(|| cumprod(&x, 0)).unwrap();
        assert!(cp.grad_fn().is_none());
    }

    // =======================================================================
    // cummax forward
    // =======================================================================

    #[test]
    fn test_cummax_1d() {
        let x = leaf(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], false);
        let r = cummax(&x, 0).unwrap();
        let d = r.values.data().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 4.0).abs() < 1e-12);
        assert!((d[3] - 4.0).abs() < 1e-12);
        assert!((d[4] - 5.0).abs() < 1e-12);
        assert_eq!(r.indices, vec![0, 0, 2, 2, 4]);
    }

    #[test]
    fn test_cummax_2d_dim1() {
        // [[1, 3, 2], [5, 4, 6]] cummax along dim 1
        // -> values: [[1, 3, 3], [5, 5, 6]]
        // -> indices: [[0, 1, 1], [0, 0, 2]]
        let x = leaf(&[1.0, 3.0, 2.0, 5.0, 4.0, 6.0], &[2, 3], false);
        let r = cummax(&x, 1).unwrap();
        let d = r.values.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1] - 3.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        assert!((d[3] - 5.0).abs() < 1e-12);
        assert!((d[4] - 5.0).abs() < 1e-12);
        assert!((d[5] - 6.0).abs() < 1e-12);
        assert_eq!(r.indices, vec![0, 1, 1, 0, 0, 2]);
    }

    // =======================================================================
    // cummin forward
    // =======================================================================

    #[test]
    fn test_cummin_1d() {
        let x = leaf(&[3.0, 1.0, 4.0, 1.0, 5.0], &[5], false);
        let r = cummin(&x, 0).unwrap();
        let d = r.values.data().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-12);
        assert!((d[1] - 1.0).abs() < 1e-12);
        assert!((d[2] - 1.0).abs() < 1e-12);
        assert!((d[3] - 1.0).abs() < 1e-12);
        assert!((d[4] - 1.0).abs() < 1e-12);
        assert_eq!(r.indices, vec![0, 1, 1, 1, 1]);
    }

    #[test]
    fn test_cummin_2d_dim0() {
        // [[5, 2], [3, 4]] cummin along dim 0
        // -> values: [[5, 2], [3, 2]]
        // -> indices: [[0, 0], [1, 0]]
        let x = leaf(&[5.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let r = cummin(&x, 0).unwrap();
        let d = r.values.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-12);
        assert!((d[1] - 2.0).abs() < 1e-12);
        assert!((d[2] - 3.0).abs() < 1e-12);
        assert!((d[3] - 2.0).abs() < 1e-12);
        assert_eq!(r.indices, vec![0, 0, 1, 0]);
    }

    // =======================================================================
    // logcumsumexp forward
    // =======================================================================

    #[test]
    fn test_logcumsumexp_1d() {
        // logcumsumexp([a, b, c]) = [a, log(exp(a)+exp(b)), log(exp(a)+exp(b)+exp(c))]
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let lcs = logcumsumexp(&x, 0).unwrap();
        let d = lcs.data().unwrap();

        let expected_0 = 1.0_f64;
        let expected_1 = (1.0_f64.exp() + 2.0_f64.exp()).ln();
        let expected_2 = (1.0_f64.exp() + 2.0_f64.exp() + 3.0_f64.exp()).ln();

        assert!((d[0] - expected_0).abs() < 1e-10, "got {}", d[0]);
        assert!((d[1] - expected_1).abs() < 1e-10, "got {}", d[1]);
        assert!((d[2] - expected_2).abs() < 1e-10, "got {}", d[2]);
    }

    #[test]
    fn test_logcumsumexp_2d_dim1() {
        // [[0, 1], [2, 3]] logcumsumexp along dim 1
        let x = leaf(&[0.0, 1.0, 2.0, 3.0], &[2, 2], false);
        let lcs = logcumsumexp(&x, 1).unwrap();
        let d = lcs.data().unwrap();

        let e0 = 0.0_f64;
        let e1 = (0.0_f64.exp() + 1.0_f64.exp()).ln();
        let e2 = 2.0_f64;
        let e3 = (2.0_f64.exp() + 3.0_f64.exp()).ln();

        assert!((d[0] - e0).abs() < 1e-10, "got {}", d[0]);
        assert!((d[1] - e1).abs() < 1e-10, "got {}", d[1]);
        assert!((d[2] - e2).abs() < 1e-10, "got {}", d[2]);
        assert!((d[3] - e3).abs() < 1e-10, "got {}", d[3]);
    }

    #[test]
    fn test_logcumsumexp_numerical_stability() {
        // Large values that would overflow naive exp.
        let x = leaf(&[1000.0, 1001.0, 1002.0], &[3], false);
        let lcs = logcumsumexp(&x, 0).unwrap();
        let d = lcs.data().unwrap();

        // All results should be finite.
        for &v in d {
            assert!(v.is_finite(), "got non-finite: {v}");
        }

        // First element should be 1000.
        assert!((d[0] - 1000.0).abs() < 1e-10);

        // Second: log(exp(1000) + exp(1001)) = 1001 + log(exp(-1) + 1)
        let expected_1 = 1001.0 + ((-1.0_f64).exp() + 1.0).ln();
        assert!((d[1] - expected_1).abs() < 1e-8, "got {}", d[1]);
    }

    // =======================================================================
    // logcumsumexp backward
    // =======================================================================

    #[test]
    fn test_logcumsumexp_backward_1d() {
        // Numerical gradient check.
        let x_vals = [1.0_f64, 2.0, 3.0];
        let eps = 1e-6;

        let x = leaf(&x_vals, &[3], true);
        let lcs = logcumsumexp(&x, 0).unwrap();
        let loss = sum(&lcs).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();

        // Check against finite differences.
        for idx in 0..3 {
            let mut x_plus = x_vals.to_vec();
            let mut x_minus = x_vals.to_vec();
            x_plus[idx] += eps;
            x_minus[idx] -= eps;

            let tp = leaf(&x_plus, &[3], false);
            let lp = logcumsumexp(&tp, 0).unwrap();
            let sp = sum(&lp).unwrap().item().unwrap();

            let tm = leaf(&x_minus, &[3], false);
            let lm = logcumsumexp(&tm, 0).unwrap();
            let sm = sum(&lm).unwrap().item().unwrap();

            let numerical = (sp - sm) / (2.0 * eps);
            assert!(
                (gd[idx] - numerical).abs() < 1e-4,
                "index {idx}: analytic={}, numerical={}",
                gd[idx],
                numerical,
            );
        }
    }

    #[test]
    fn test_logcumsumexp_has_grad_fn() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let lcs = logcumsumexp(&x, 0).unwrap();
        assert!(lcs.grad_fn().is_some());
        assert_eq!(lcs.grad_fn().unwrap().name(), "LogcumsumexpBackward");
    }

    #[test]
    fn test_logcumsumexp_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let lcs = no_grad(|| logcumsumexp(&x, 0)).unwrap();
        assert!(lcs.grad_fn().is_none());
    }

    // =======================================================================
    // Error cases
    // =======================================================================

    #[test]
    fn test_cumsum_scalar_error() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        assert!(cumsum(&x, 0).is_err());
    }

    #[test]
    fn test_cumprod_scalar_error() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        assert!(cumprod(&x, 0).is_err());
    }

    #[test]
    fn test_cummax_scalar_error() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        assert!(cummax(&x, 0).is_err());
    }

    #[test]
    fn test_cummin_scalar_error() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        assert!(cummin(&x, 0).is_err());
    }

    #[test]
    fn test_logcumsumexp_scalar_error() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        assert!(logcumsumexp(&x, 0).is_err());
    }

    #[test]
    fn test_cumsum_dim_out_of_bounds() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        assert!(cumsum(&x, 1).is_err());
        assert!(cumsum(&x, -2).is_err());
    }

    // =======================================================================
    // cumprod backward numerical gradient check
    // =======================================================================

    #[test]
    fn test_cumprod_backward_numerical() {
        let x_vals = [2.0_f64, 3.0, 0.5];
        let eps = 1e-6;

        let x = leaf(&x_vals, &[3], true);
        let cp = cumprod(&x, 0).unwrap();
        let loss = sum(&cp).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();

        for idx in 0..3 {
            let mut x_plus = x_vals.to_vec();
            let mut x_minus = x_vals.to_vec();
            x_plus[idx] += eps;
            x_minus[idx] -= eps;

            let tp = leaf(&x_plus, &[3], false);
            let fp = sum(&cumprod(&tp, 0).unwrap()).unwrap().item().unwrap();

            let tm = leaf(&x_minus, &[3], false);
            let fm = sum(&cumprod(&tm, 0).unwrap()).unwrap().item().unwrap();

            let numerical = (fp - fm) / (2.0 * eps);
            assert!(
                (gd[idx] - numerical).abs() < 1e-4,
                "index {idx}: analytic={}, numerical={}",
                gd[idx],
                numerical,
            );
        }
    }

    // =======================================================================
    // cumsum backward numerical gradient check
    // =======================================================================

    #[test]
    fn test_cumsum_backward_numerical() {
        let x_vals = [1.0_f64, -2.0, 3.5, 0.7];
        let eps = 1e-6;

        let x = leaf(&x_vals, &[4], true);
        let cs = cumsum(&x, 0).unwrap();
        let loss = sum(&cs).unwrap();
        loss.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();

        for idx in 0..4 {
            let mut x_plus = x_vals.to_vec();
            let mut x_minus = x_vals.to_vec();
            x_plus[idx] += eps;
            x_minus[idx] -= eps;

            let tp = leaf(&x_plus, &[4], false);
            let fp = sum(&cumsum(&tp, 0).unwrap()).unwrap().item().unwrap();

            let tm = leaf(&x_minus, &[4], false);
            let fm = sum(&cumsum(&tm, 0).unwrap()).unwrap().item().unwrap();

            let numerical = (fp - fm) / (2.0 * eps);
            assert!(
                (gd[idx] - numerical).abs() < 1e-4,
                "index {idx}: analytic={}, numerical={}",
                gd[idx],
                numerical,
            );
        }
    }
}
