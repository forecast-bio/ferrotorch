//! Forward-pass implementations for N-D indexing operations.
//!
//! - `gather(input, dim, index)` — gather elements along an axis
//! - `scatter(input, dim, index, src)` — scatter src values into input
//! - `scatter_add(input, dim, index, src)` — scatter with addition
//! - `where_cond(condition, x, y)` — ternary selection
//!
//! All operations are CPU-only; GPU tensors are transferred transparently.
//! Backward (gradient) functions live in `grad_fns::indexing`.

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::shape::normalize_axis;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether at least one of two tensors requires grad (and grad is enabled).
#[inline]
fn needs_grad<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    is_grad_enabled() && (a.requires_grad() || b.requires_grad())
}

/// Compute the flat index into a C-contiguous buffer from per-axis coordinates.
#[inline]
fn flat_index(coords: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0;
    let mut stride = 1;
    for d in (0..shape.len()).rev() {
        idx += coords[d] * stride;
        stride *= shape[d];
    }
    idx
}

/// Increment a multi-dimensional coordinate vector in C-order (last axis
/// fastest). Returns `false` when the coordinate wraps past the last element.
#[inline]
fn increment_coords(coords: &mut [usize], shape: &[usize]) -> bool {
    for d in (0..shape.len()).rev() {
        coords[d] += 1;
        if coords[d] < shape[d] {
            return true;
        }
        coords[d] = 0;
    }
    false
}

/// Validate that `index` shape matches `input` shape on all dimensions
/// except `dim`, and that every index value is in-bounds for `input.shape()[dim]`.
///
/// This mirrors PyTorch's gather/scatter shape requirements:
///   - `index.ndim() == input.ndim()`
///   - For all d != dim: `index.shape()[d] <= input.shape()[d]`  (gather)
///     or `index.shape()[d] <= src.shape()[d]` (scatter)
///
/// We enforce the simpler check that ndim matches.
fn validate_gather_shapes(
    input_shape: &[usize],
    dim: usize,
    index_shape: &[usize],
    index_data: &[usize],
    axis_size: usize,
) -> FerrotorchResult<()> {
    if input_shape.len() != index_shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "gather/scatter: input ndim ({}) must equal index ndim ({})",
                input_shape.len(),
                index_shape.len()
            ),
        });
    }
    // Validate index values are in-bounds along `dim`.
    for &v in index_data {
        if v >= axis_size {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: v,
                axis: dim,
                size: axis_size,
            });
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// gather
// ---------------------------------------------------------------------------

/// Gather values from `input` along `dim` using `index`.
///
/// PyTorch semantics:
/// ```text
/// output[i][j][k] = input[index[i][j][k]][j][k]  # if dim == 0
/// output[i][j][k] = input[i][index[i][j][k]][k]  # if dim == 1
/// output[i][j][k] = input[i][j][index[i][j][k]]  # if dim == 2
/// ```
///
/// The output has the same shape as `index`.
///
/// `index` is passed as a flat `&[usize]` slice with shape `index_shape`.
/// If `input.requires_grad()`, attaches a `GatherBackward` grad_fn.
pub fn gather<T: Float>(
    input: &Tensor<T>,
    dim: isize,
    index: &[usize],
    index_shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "gather: input must have at least 1 dimension".into(),
        });
    }
    let dim = normalize_axis(dim, ndim)?;
    let input_shape = input.shape();

    validate_gather_shapes(input_shape, dim, index_shape, index, input_shape[dim])?;

    let input_data = input.data_vec()?;
    let out_numel: usize = index_shape.iter().product();
    let mut output = vec![<T as num_traits::Zero>::zero(); out_numel];

    let mut coords = vec![0usize; ndim];
    for out_flat in 0..out_numel {
        // Build source coordinates: same as output coords, but replace dim
        // with the index value.
        let idx_val = index[out_flat];
        let mut src_coords = coords.clone();
        src_coords[dim] = idx_val;
        let src_flat = flat_index(&src_coords, input_shape);
        output[out_flat] = input_data[src_flat];

        if out_flat + 1 < out_numel {
            increment_coords(&mut coords, index_shape);
        }
    }

    let output_shape = index_shape.to_vec();

    if input.requires_grad() && is_grad_enabled() {
        let grad_fn = Arc::new(crate::grad_fns::indexing::GatherBackward {
            input: input.clone(),
            dim,
            index: index.to_vec(),
            index_shape: index_shape.to_vec(),
        });
        Tensor::from_operation(TensorStorage::cpu(output), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// scatter
// ---------------------------------------------------------------------------

/// Scatter `src` values into a clone of `input` along `dim` using `index`.
///
/// PyTorch semantics:
/// ```text
/// output = input.clone()
/// output[index[i][j][k]][j][k] = src[i][j][k]  # if dim == 0
/// ```
///
/// The output has the same shape as `input`.
///
/// `index` and `src` are flat slices with shape `index_shape`.
/// If either `input` or `src` requires grad, attaches a `ScatterBackward`.
pub fn scatter<T: Float>(
    input: &Tensor<T>,
    dim: isize,
    index: &[usize],
    index_shape: &[usize],
    src: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "scatter: input must have at least 1 dimension".into(),
        });
    }
    let dim = normalize_axis(dim, ndim)?;
    let input_shape = input.shape();

    validate_gather_shapes(input_shape, dim, index_shape, index, input_shape[dim])?;

    let index_numel: usize = index_shape.iter().product();
    if src.numel() < index_numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter: src has {} elements but index has {}",
                src.numel(),
                index_numel
            ),
        });
    }

    let mut output = input.data_vec()?;
    let src_data = src.data_vec()?;

    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        let dst_flat = flat_index(&dst_coords, input_shape);
        output[dst_flat] = src_data[i];

        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }

    let output_shape = input_shape.to_vec();

    if needs_grad(input, src) {
        let grad_fn = Arc::new(crate::grad_fns::indexing::ScatterBackward {
            input: input.clone(),
            src: src.clone(),
            dim,
            index: index.to_vec(),
            index_shape: index_shape.to_vec(),
        });
        Tensor::from_operation(TensorStorage::cpu(output), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// scatter_add
// ---------------------------------------------------------------------------

/// Scatter-add `src` values into a clone of `input` along `dim`.
///
/// Like `scatter`, but uses addition instead of assignment:
/// ```text
/// output = input.clone()
/// output[index[i][j][k]][j][k] += src[i][j][k]  # if dim == 0
/// ```
pub fn scatter_add<T: Float>(
    input: &Tensor<T>,
    dim: isize,
    index: &[usize],
    index_shape: &[usize],
    src: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "scatter_add: input must have at least 1 dimension".into(),
        });
    }
    let dim = normalize_axis(dim, ndim)?;
    let input_shape = input.shape();

    validate_gather_shapes(input_shape, dim, index_shape, index, input_shape[dim])?;

    let index_numel: usize = index_shape.iter().product();
    if src.numel() < index_numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "scatter_add: src has {} elements but index has {}",
                src.numel(),
                index_numel
            ),
        });
    }

    let mut output = input.data_vec()?;
    let src_data = src.data_vec()?;

    let mut coords = vec![0usize; ndim];
    for i in 0..index_numel {
        let idx_val = index[i];
        let mut dst_coords = coords.clone();
        dst_coords[dim] = idx_val;
        let dst_flat = flat_index(&dst_coords, input_shape);
        output[dst_flat] += src_data[i];

        if i + 1 < index_numel {
            increment_coords(&mut coords, index_shape);
        }
    }

    let output_shape = input_shape.to_vec();

    if needs_grad(input, src) {
        let grad_fn = Arc::new(crate::grad_fns::indexing::ScatterAddBackward {
            input: input.clone(),
            src: src.clone(),
            dim,
            index: index.to_vec(),
            index_shape: index_shape.to_vec(),
        });
        Tensor::from_operation(TensorStorage::cpu(output), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// where_cond
// ---------------------------------------------------------------------------

/// Ternary selection: `output[i] = condition[i] ? x[i] : y[i]`.
///
/// All three tensors must have the same shape (no broadcasting yet).
/// `condition` is a flat `&[bool]` slice.
///
/// If either `x` or `y` requires grad, attaches a `WhereCondBackward`.
pub fn where_cond<T: Float>(
    condition: &[bool],
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if x.shape() != y.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "where_cond: x shape {:?} != y shape {:?}",
                x.shape(),
                y.shape()
            ),
        });
    }
    let numel = x.numel();
    if condition.len() != numel {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "where_cond: condition length {} != tensor numel {}",
                condition.len(),
                numel
            ),
        });
    }

    let x_data = x.data_vec()?;
    let y_data = y.data_vec()?;

    let output: Vec<T> = condition
        .iter()
        .zip(x_data.iter().zip(y_data.iter()))
        .map(|(&c, (&xv, &yv))| if c { xv } else { yv })
        .collect();

    let output_shape = x.shape().to_vec();

    if needs_grad(x, y) {
        let grad_fn = Arc::new(crate::grad_fns::indexing::WhereCondBackward {
            x: x.clone(),
            y: y.clone(),
            condition: condition.to_vec(),
        });
        Tensor::from_operation(TensorStorage::cpu(output), output_shape, grad_fn)
    } else {
        Tensor::from_storage(TensorStorage::cpu(output), output_shape, false)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::graph::backward;
    use crate::autograd::no_grad;
    use crate::storage::TensorStorage;
    use crate::tensor::GradFn;

    /// Create a leaf tensor from a flat slice and shape.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // gather forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_1d() {
        // input = [10, 20, 30, 40], gather along dim 0 with index [3, 0, 2]
        let input = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], false);
        let index = &[3, 0, 2];
        let result = gather(&input, 0, index, &[3]).unwrap();
        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data().unwrap(), &[40.0, 10.0, 30.0]);
    }

    #[test]
    fn test_gather_2d_dim0() {
        // input = [[1, 2], [3, 4], [5, 6]]  shape [3, 2]
        // index = [[2, 0], [1, 1]]           shape [2, 2]
        // output[i][j] = input[index[i][j]][j]
        // output = [[5, 2], [3, 4]]
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
        let index = &[2, 0, 1, 1];
        let result = gather(&input, 0, index, &[2, 2]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data().unwrap(), &[5.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_gather_2d_dim1() {
        // input = [[1, 2, 3], [4, 5, 6]]  shape [2, 3]
        // index = [[0, 2], [1, 0]]        shape [2, 2]
        // output[i][j] = input[i][index[i][j]]
        // output = [[1, 3], [5, 4]]
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let index = &[0, 2, 1, 0];
        let result = gather(&input, 1, index, &[2, 2]).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data().unwrap(), &[1.0, 3.0, 5.0, 4.0]);
    }

    #[test]
    fn test_gather_out_of_bounds() {
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let result = gather(&input, 0, &[5], &[1]);
        assert!(result.is_err());
    }

    #[test]
    fn test_gather_ndim_mismatch() {
        // input is 2D, index is 1D
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let result = gather(&input, 0, &[0, 1], &[2]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // scatter forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_1d() {
        // input = [0, 0, 0, 0, 0], scatter src=[10, 20, 30] at index=[1, 3, 0]
        let input = leaf(&[0.0; 5], &[5], false);
        let src = leaf(&[10.0, 20.0, 30.0], &[3], false);
        let result = scatter(&input, 0, &[1, 3, 0], &[3], &src).unwrap();
        assert_eq!(result.data().unwrap(), &[30.0, 10.0, 0.0, 20.0, 0.0]);
    }

    #[test]
    fn test_scatter_2d_dim0() {
        // input = [[0,0],[0,0],[0,0]]  shape [3, 2]
        // src   = [[1,2]]              shape [1, 2]
        // index = [[2,0]]              shape [1, 2]
        // output[index[i][j]][j] = src[i][j]
        // output = [[0,2],[0,0],[1,0]]
        let input = leaf(&[0.0; 6], &[3, 2], false);
        let src = leaf(&[1.0, 2.0], &[1, 2], false);
        let result = scatter(&input, 0, &[2, 0], &[1, 2], &src).unwrap();
        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(result.data().unwrap(), &[0.0, 2.0, 0.0, 0.0, 1.0, 0.0]);
    }

    #[test]
    fn test_scatter_2d_dim1() {
        // input = [[0,0,0],[0,0,0]]  shape [2, 3]
        // src   = [[5],[6]]          shape [2, 1]
        // index = [[2],[0]]          shape [2, 1]
        // output[i][index[i][j]] = src[i][j]
        // output = [[0,0,5],[6,0,0]]
        let input = leaf(&[0.0; 6], &[2, 3], false);
        let src = leaf(&[5.0, 6.0], &[2, 1], false);
        let result = scatter(&input, 1, &[2, 0], &[2, 1], &src).unwrap();
        assert_eq!(result.data().unwrap(), &[0.0, 0.0, 5.0, 6.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // scatter_add forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_add_1d() {
        // input = [1, 2, 3], scatter_add src=[10, 20, 30] at index=[0, 2, 0]
        // output = [1+10+30, 2, 3+20] = [41, 2, 23]
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let src = leaf(&[10.0, 20.0, 30.0], &[3], false);
        let result = scatter_add(&input, 0, &[0, 2, 0], &[3], &src).unwrap();
        assert_eq!(result.data().unwrap(), &[41.0, 2.0, 23.0]);
    }

    #[test]
    fn test_scatter_add_2d_dim0() {
        // input = [[0,0],[0,0]]  shape [2, 2]
        // src   = [[1,2],[3,4],[5,6]]  shape [3, 2]
        // index = [[0,1],[1,0],[0,0]]  shape [3, 2]
        //
        // output[index[i][j]][j] += src[i][j]
        // (0,0) += 1, (1,0) += 2
        // (1,0) += 3, (0,1) += 4
        // (0,0) += 5, (0,1) += 6
        // output = [[0+1+5, 0+4+6], [0+3, 0+2]] = [[6, 10], [3, 2]]
        let input = leaf(&[0.0; 4], &[2, 2], false);
        let src = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2], false);
        let result = scatter_add(&input, 0, &[0, 1, 1, 0, 0, 0], &[3, 2], &src).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data().unwrap(), &[6.0, 10.0, 3.0, 2.0]);
    }

    // -----------------------------------------------------------------------
    // where_cond forward
    // -----------------------------------------------------------------------

    #[test]
    fn test_where_cond_basic() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], false);
        let cond = [true, false, true, false];
        let result = where_cond(&cond, &x, &y).unwrap();
        assert_eq!(result.data().unwrap(), &[1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn test_where_cond_all_true() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        let y = leaf(&[10.0, 20.0], &[2], false);
        let result = where_cond(&[true, true], &x, &y).unwrap();
        assert_eq!(result.data().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_where_cond_all_false() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        let y = leaf(&[10.0, 20.0], &[2], false);
        let result = where_cond(&[false, false], &x, &y).unwrap();
        assert_eq!(result.data().unwrap(), &[10.0, 20.0]);
    }

    #[test]
    fn test_where_cond_shape_mismatch() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        let y = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let result = where_cond(&[true, false], &x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_where_cond_cond_length_mismatch() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        let y = leaf(&[10.0, 20.0], &[2], false);
        let result = where_cond(&[true], &x, &y);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // gather backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_backward_1d() {
        // input = [10, 20, 30], gather at [2, 0, 0] -> output = [30, 10, 10]
        // grad_output = [1, 1, 1]
        // grad_input: scatter_add of [1,1,1] at [2,0,0] into zeros(3)
        //   = [2, 0, 1]
        let input = leaf(&[10.0, 20.0, 30.0], &[3], true);
        let result = gather(&input, 0, &[2, 0, 0], &[3]).unwrap();

        assert!(result.requires_grad());

        let grad_output = leaf(&[1.0, 1.0, 1.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();
        let gi = grads[0].as_ref().unwrap();
        let gd = gi.data().unwrap();
        assert!((gd[0] - 2.0).abs() < 1e-6, "grad[0]={}, expected 2", gd[0]);
        assert!((gd[1] - 0.0).abs() < 1e-6, "grad[1]={}, expected 0", gd[1]);
        assert!((gd[2] - 1.0).abs() < 1e-6, "grad[2]={}, expected 1", gd[2]);
    }

    #[test]
    fn test_gather_backward_2d() {
        // input shape [2, 3], gather dim=1, index shape [2, 2]
        // input = [[1,2,3],[4,5,6]]
        // index = [[0, 2], [1, 0]]
        // output = [[1,3],[5,4]]
        //
        // grad_output = [[1,1],[1,1]]
        // grad_input: scatter_add along dim=1
        //   row 0: idx [0,2] -> [1, 0, 1]
        //   row 1: idx [1,0] -> [1, 1, 0]
        //   grad_input = [[1,0,1],[1,1,0]]
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let result = gather(&input, 1, &[0, 2, 1, 0], &[2, 2]).unwrap();

        let grad_output = leaf(&[1.0, 1.0, 1.0, 1.0], &[2, 2], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();
        let gi = grads[0].as_ref().unwrap();
        let gd = gi.data().unwrap();
        assert_eq!(gi.shape(), &[2, 3]);
        // row 0: [1, 0, 1]
        assert!((gd[0] - 1.0).abs() < 1e-6);
        assert!((gd[1] - 0.0).abs() < 1e-6);
        assert!((gd[2] - 1.0).abs() < 1e-6);
        // row 1: [1, 1, 0]
        assert!((gd[3] - 1.0).abs() < 1e-6);
        assert!((gd[4] - 1.0).abs() < 1e-6);
        assert!((gd[5] - 0.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // scatter backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_backward_input() {
        // scatter zeros out the positions that were overwritten.
        // input = [1, 2, 3, 4, 5], scatter src at [1, 3]
        // grad wrt input: ones everywhere except positions 1 and 3
        // -> [1, 0, 1, 0, 1]
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0], &[5], true);
        let src = leaf(&[10.0, 20.0], &[2], false);
        let result = scatter(&input, 0, &[1, 3], &[2], &src).unwrap();

        let grad_output = leaf(&[1.0; 5], &[5], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();
        let gi = grads[0].as_ref().unwrap();
        let gd = gi.data().unwrap();
        assert_eq!(gd, &[1.0, 0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_scatter_backward_src() {
        // scatter grad wrt src is gather from grad_output at index positions.
        // input = [0, 0, 0], scatter src at [2, 0]
        // grad_output = [10, 20, 30]
        // grad_src = [grad_output[2], grad_output[0]] = [30, 10]
        let input = leaf(&[0.0; 3], &[3], false);
        let src = leaf(&[1.0, 2.0], &[2], true);
        let result = scatter(&input, 0, &[2, 0], &[2], &src).unwrap();

        let grad_output = leaf(&[10.0, 20.0, 30.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        // grads[0] is for input (not requiring grad -> None)
        assert!(grads[0].is_none());
        // grads[1] is for src
        let gs = grads[1].as_ref().unwrap();
        let gd = gs.data().unwrap();
        assert_eq!(gd, &[30.0, 10.0]);
    }

    // -----------------------------------------------------------------------
    // scatter_add backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_scatter_add_backward_input() {
        // scatter_add backward for input is just grad_output (identity).
        let input = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let src = leaf(&[10.0, 20.0], &[2], false);
        let result = scatter_add(&input, 0, &[0, 2], &[2], &src).unwrap();

        let grad_output = leaf(&[5.0, 6.0, 7.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();
        let gi = grads[0].as_ref().unwrap();
        assert_eq!(gi.data().unwrap(), &[5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_scatter_add_backward_src() {
        // scatter_add backward for src is gather from grad_output.
        // index = [2, 0], grad_output = [5, 6, 7]
        // grad_src = [grad_output[2], grad_output[0]] = [7, 5]
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let src = leaf(&[10.0, 20.0], &[2], true);
        let result = scatter_add(&input, 0, &[2, 0], &[2], &src).unwrap();

        let grad_output = leaf(&[5.0, 6.0, 7.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        assert!(grads[0].is_none());
        let gs = grads[1].as_ref().unwrap();
        assert_eq!(gs.data().unwrap(), &[7.0, 5.0]);
    }

    // -----------------------------------------------------------------------
    // where_cond backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_where_cond_backward_x() {
        // where_cond grad for x: grad_output where condition is true, 0 otherwise.
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], false);
        let cond = [true, false, true, false];
        let result = where_cond(&cond, &x, &y).unwrap();

        let grad_output = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let gx = grads[0].as_ref().unwrap();
        assert_eq!(gx.data().unwrap(), &[1.0, 0.0, 3.0, 0.0]);
        assert!(grads[1].is_none());
    }

    #[test]
    fn test_where_cond_backward_y() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], true);
        let cond = [true, false, true, false];
        let result = where_cond(&cond, &x, &y).unwrap();

        let grad_output = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        assert!(grads[0].is_none());
        let gy = grads[1].as_ref().unwrap();
        assert_eq!(gy.data().unwrap(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_where_cond_backward_both() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let y = leaf(&[10.0, 20.0, 30.0], &[3], true);
        let cond = [false, true, false];
        let result = where_cond(&cond, &x, &y).unwrap();

        let grad_output = leaf(&[5.0, 6.0, 7.0], &[3], false);
        let grad_fn = result.grad_fn().unwrap();
        let grads = grad_fn.backward(&grad_output).unwrap();

        let gx = grads[0].as_ref().unwrap();
        assert_eq!(gx.data().unwrap(), &[0.0, 6.0, 0.0]);
        let gy = grads[1].as_ref().unwrap();
        assert_eq!(gy.data().unwrap(), &[5.0, 0.0, 7.0]);
    }

    // -----------------------------------------------------------------------
    // no_grad context
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_no_grad() {
        let input = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let result = no_grad(|| gather(&input, 0, &[2, 0], &[2])).unwrap();
        assert!(!result.requires_grad());
        assert!(result.grad_fn().is_none());
    }

    #[test]
    fn test_where_cond_no_grad() {
        let x = leaf(&[1.0, 2.0], &[2], true);
        let y = leaf(&[3.0, 4.0], &[2], true);
        let result = no_grad(|| where_cond(&[true, false], &x, &y)).unwrap();
        assert!(!result.requires_grad());
    }

    // -----------------------------------------------------------------------
    // End-to-end backward through autograd
    // -----------------------------------------------------------------------

    #[test]
    fn test_gather_end_to_end_backward() {
        let input = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], true);
        let gathered = gather(&input, 0, &[1, 3], &[2]).unwrap();

        // Sum to scalar via inline SumBackward.
        let data = gathered.data().unwrap();
        let total: f32 = data.iter().sum();

        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let go_val = grad_output.data()?[0];
                let grad = vec![go_val; self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(grad),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: gathered.clone(),
            }),
        )
        .unwrap();

        backward(&loss).unwrap();

        let grad = input.grad().unwrap().unwrap();
        let gd = grad.data().unwrap();
        // indices [1, 3]: grad = [0, 1, 0, 1]
        assert!((gd[0] - 0.0).abs() < 1e-6);
        assert!((gd[1] - 1.0).abs() < 1e-6);
        assert!((gd[2] - 0.0).abs() < 1e-6);
        assert!((gd[3] - 1.0).abs() < 1e-6);
    }
}
