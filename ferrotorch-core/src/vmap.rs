//! Vectorized map (vmap) — apply a function over a batch dimension.
//!
//! This module provides [`vmap`] and [`vmap2`], which take a per-element
//! function and vectorize it over a batch dimension. The MVP implementation
//! is loop-based (correct but not fused); a future version may trace the
//! function to produce a batched kernel.
//!
//! Helper utilities [`select`] and [`stack`] are also provided for
//! extracting slices and reassembling tensors along arbitrary dimensions.

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// select — extract a single slice along a dimension, removing it
// ---------------------------------------------------------------------------

/// Extract a single slice along `dim` at position `index`, removing the
/// dimension.
///
/// For a tensor of shape `[B, M, N]`, `select(t, 0, i)` returns the
/// `[M, N]` slice at batch index `i`.
pub fn select<T: Float>(
    input: &Tensor<T>,
    dim: usize,
    index: usize,
) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "select: dim {dim} is out of bounds for tensor with {ndim} dimensions"
            ),
        });
    }

    if index >= shape[dim] {
        return Err(FerrotorchError::IndexOutOfBounds {
            index,
            axis: dim,
            size: shape[dim],
        });
    }

    // Build output shape: the input shape with `dim` removed.
    let mut out_shape: Vec<usize> = shape.to_vec();
    out_shape.remove(dim);

    let data = input.data()?;

    // Dimensions factored as: outer * shape[dim] * inner.
    let outer: usize = shape[..dim].iter().product();
    let inner: usize = if dim + 1 < ndim {
        shape[dim + 1..].iter().product()
    } else {
        1
    };
    let dim_size = shape[dim];

    let out_numel: usize = outer * inner;
    let mut out_data = Vec::with_capacity(out_numel);

    for o in 0..outer {
        let src_base = o * dim_size * inner + index * inner;
        for j in 0..inner {
            out_data.push(data[src_base + j]);
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out_data), out_shape, false)
}

// ---------------------------------------------------------------------------
// stack — stack tensors along a new dimension
// ---------------------------------------------------------------------------

/// Stack a slice of tensors along a new dimension `dim`.
///
/// All tensors must have the same shape. Given `N` tensors of shape
/// `[M, K]`, `stack(ts, 0)` returns a tensor of shape `[N, M, K]`.
pub fn stack<T: Float>(tensors: &[Tensor<T>], dim: usize) -> FerrotorchResult<Tensor<T>> {
    if tensors.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "stack: empty tensor list".into(),
        });
    }

    let base_shape = tensors[0].shape();
    let base_ndim = base_shape.len();

    // `dim` can be at most `base_ndim` (insert at the end).
    if dim > base_ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "stack: dim {dim} is out of bounds for tensors with {base_ndim} dimensions (max = {base_ndim})"
            ),
        });
    }

    // Validate all tensors share the same shape.
    for (i, t) in tensors.iter().enumerate().skip(1) {
        if t.shape() != base_shape {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "stack: tensor {} has shape {:?}, expected {:?}",
                    i,
                    t.shape(),
                    base_shape
                ),
            });
        }
    }

    let n = tensors.len();

    // Output shape: insert `n` at position `dim`.
    let mut out_shape = base_shape.to_vec();
    out_shape.insert(dim, n);

    // Dimensions: outer * n * inner, where outer = product of dims before `dim`,
    // inner = product of dims from `dim` onward (in the original shape).
    let outer: usize = base_shape[..dim].iter().product();
    let inner: usize = if dim < base_ndim {
        base_shape[dim..].iter().product()
    } else {
        1
    };

    let out_numel: usize = out_shape.iter().product();
    let mut out_data = vec![<T as num_traits::Zero>::zero(); out_numel];

    for (t_idx, t) in tensors.iter().enumerate() {
        let t_data = t.data()?;
        for o in 0..outer {
            let dst_base = o * n * inner + t_idx * inner;
            let src_base = o * inner;
            out_data[dst_base..dst_base + inner]
                .copy_from_slice(&t_data[src_base..src_base + inner]);
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out_data), out_shape, false)
}

// ---------------------------------------------------------------------------
// vmap — vectorize a function over one batch dimension
// ---------------------------------------------------------------------------

/// Vectorize a function over a batch dimension.
///
/// Applies `f` independently to each slice along `in_dim` of the input
/// tensor, stacking the results along `out_dim` of the output.
///
/// # Example
///
/// ```ignore
/// // Apply matmul per-batch-element without writing a loop
/// let result = vmap(|x| x.matmul(&weights), 0, 0)(&batched_input)?;
/// ```
///
/// # Current implementation
///
/// This is a loop-based MVP: each batch element is processed sequentially.
/// A future version may trace `f` to produce a fused batched kernel.
pub fn vmap<T, F>(
    f: F,
    in_dim: usize,
    out_dim: usize,
) -> impl Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>>,
{
    move |input: &Tensor<T>| {
        let shape = input.shape();

        if in_dim >= shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "vmap: in_dim {} is out of bounds for tensor with {} dimensions",
                    in_dim,
                    shape.len()
                ),
            });
        }

        let batch_size = shape[in_dim];
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let slice = select(input, in_dim, i)?;
            let output = f(&slice)?;
            results.push(output);
        }

        stack(&results, out_dim)
    }
}

// ---------------------------------------------------------------------------
// vmap2 — vectorize a function over two inputs
// ---------------------------------------------------------------------------

/// Vectorize a two-argument function over batch dimensions.
///
/// Both inputs must have the same size along their respective batch
/// dimensions. The function `f` is called once per batch element with
/// the corresponding slices, and the results are stacked along `out_dim`.
pub fn vmap2<T, F>(
    f: F,
    in_dim_a: usize,
    in_dim_b: usize,
    out_dim: usize,
) -> impl Fn(&Tensor<T>, &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>, &Tensor<T>) -> FerrotorchResult<Tensor<T>>,
{
    move |a: &Tensor<T>, b: &Tensor<T>| {
        let a_shape = a.shape();
        let b_shape = b.shape();

        if in_dim_a >= a_shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "vmap2: in_dim_a {} is out of bounds for tensor a with {} dimensions",
                    in_dim_a,
                    a_shape.len()
                ),
            });
        }

        if in_dim_b >= b_shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "vmap2: in_dim_b {} is out of bounds for tensor b with {} dimensions",
                    in_dim_b,
                    b_shape.len()
                ),
            });
        }

        let batch_a = a_shape[in_dim_a];
        let batch_b = b_shape[in_dim_b];

        if batch_a != batch_b {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "vmap2: batch size mismatch: a has {batch_a} along dim {in_dim_a}, b has {batch_b} along dim {in_dim_b}"
                ),
            });
        }

        let batch_size = batch_a;
        let mut results = Vec::with_capacity(batch_size);

        for i in 0..batch_size {
            let slice_a = select(a, in_dim_a, i)?;
            let slice_b = select(b, in_dim_b, i)?;
            let output = f(&slice_a, &slice_b)?;
            results.push(output);
        }

        stack(&results, out_dim)
    }
}

// ---------------------------------------------------------------------------
// vmap3 — vectorize a three-argument function (CL-312, CL-362)
// ---------------------------------------------------------------------------

/// Vectorize a three-argument function over batch dimensions.
///
/// All three inputs must have the same size along their respective batch
/// dimensions. The function `f` is called once per batch element with the
/// corresponding slices, and the results are stacked along `out_dim`.
///
/// Useful for ops that take three tensors (e.g. `where(cond, a, b)`,
/// custom triple-input losses).
pub fn vmap3<T, F>(
    f: F,
    in_dim_a: usize,
    in_dim_b: usize,
    in_dim_c: usize,
    out_dim: usize,
) -> impl Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>) -> FerrotorchResult<Tensor<T>>,
{
    move |a: &Tensor<T>, b: &Tensor<T>, c: &Tensor<T>| {
        let a_shape = a.shape();
        let b_shape = b.shape();
        let c_shape = c.shape();

        if in_dim_a >= a_shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "vmap3: in_dim_a {} is out of bounds for tensor a with {} dimensions",
                    in_dim_a,
                    a_shape.len()
                ),
            });
        }
        if in_dim_b >= b_shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "vmap3: in_dim_b {} is out of bounds for tensor b with {} dimensions",
                    in_dim_b,
                    b_shape.len()
                ),
            });
        }
        if in_dim_c >= c_shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "vmap3: in_dim_c {} is out of bounds for tensor c with {} dimensions",
                    in_dim_c,
                    c_shape.len()
                ),
            });
        }

        let batch_a = a_shape[in_dim_a];
        let batch_b = b_shape[in_dim_b];
        let batch_c = c_shape[in_dim_c];
        if batch_a != batch_b || batch_a != batch_c {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "vmap3: batch size mismatch: a={batch_a} dim {in_dim_a}, b={batch_b} dim {in_dim_b}, c={batch_c} dim {in_dim_c}"
                ),
            });
        }

        let mut results = Vec::with_capacity(batch_a);
        for i in 0..batch_a {
            let sa = select(a, in_dim_a, i)?;
            let sb = select(b, in_dim_b, i)?;
            let sc = select(c, in_dim_c, i)?;
            results.push(f(&sa, &sb, &sc)?);
        }
        stack(&results, out_dim)
    }
}

// ---------------------------------------------------------------------------
// vmap_many — vectorize an N-argument function (CL-312, CL-362)
// ---------------------------------------------------------------------------

/// Vectorize an N-argument function over a shared batch dimension.
///
/// Each input is sliced along its corresponding `in_dims[i]`, the
/// resulting per-batch slices are passed as a slice to `f`, and the
/// outputs are stacked along `out_dim`. Use this when you have more
/// than three inputs (or when the input count is dynamic).
///
/// All inputs must have the same size along their respective batch dim.
pub fn vmap_many<T, F>(
    f: F,
    in_dims: Vec<usize>,
    out_dim: usize,
) -> impl Fn(&[&Tensor<T>]) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
{
    move |inputs: &[&Tensor<T>]| {
        if inputs.len() != in_dims.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "vmap_many: got {} inputs but {} in_dims",
                    inputs.len(),
                    in_dims.len()
                ),
            });
        }
        if inputs.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "vmap_many: at least one input required".into(),
            });
        }

        // Validate dims and check batch size consistency.
        let mut batch_size: Option<usize> = None;
        for (i, (input, &dim)) in inputs.iter().zip(in_dims.iter()).enumerate() {
            if dim >= input.ndim() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "vmap_many: in_dims[{}] = {} is out of bounds for input with {} dims",
                        i,
                        dim,
                        input.ndim()
                    ),
                });
            }
            let bs = input.shape()[dim];
            match batch_size {
                None => batch_size = Some(bs),
                Some(b) if b != bs => {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "vmap_many: batch size mismatch: input[{i}] has {bs} along dim {dim}, others have {b}"
                        ),
                    });
                }
                Some(_) => {}
            }
        }
        let batch_size = batch_size.unwrap();

        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let mut slices: Vec<Tensor<T>> = Vec::with_capacity(inputs.len());
            for (input, &dim) in inputs.iter().zip(in_dims.iter()) {
                slices.push(select(input, dim, i)?);
            }
            results.push(f(&slices)?);
        }
        stack(&results, out_dim)
    }
}

// ---------------------------------------------------------------------------
// vmap_multi_output — vectorize a function returning multiple tensors
// ---------------------------------------------------------------------------

/// Vectorize a function that returns multiple output tensors per call.
///
/// The closure must return a `Vec<Tensor<T>>` of fixed length per call
/// (the length is detected from the first invocation). Each output's
/// per-batch slices are stacked along `out_dim` independently, yielding
/// a `Vec<Tensor<T>>` of stacked results.
///
/// Useful for ops that produce multiple results like `(values, indices)`
/// from a topk, or `(mean, var)` from a stat block.
pub fn vmap_multi_output<T, F>(
    f: F,
    in_dim: usize,
    out_dim: usize,
) -> impl Fn(&Tensor<T>) -> FerrotorchResult<Vec<Tensor<T>>>
where
    T: Float,
    F: Fn(&Tensor<T>) -> FerrotorchResult<Vec<Tensor<T>>>,
{
    move |input: &Tensor<T>| {
        if in_dim >= input.ndim() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "vmap_multi_output: in_dim {} is out of bounds for tensor with {} dims",
                    in_dim,
                    input.ndim()
                ),
            });
        }
        let batch_size = input.shape()[in_dim];
        // per_call[batch_idx][output_idx] = Tensor
        let mut per_call: Vec<Vec<Tensor<T>>> = Vec::with_capacity(batch_size);
        let mut num_outputs: Option<usize> = None;
        for i in 0..batch_size {
            let slice = select(input, in_dim, i)?;
            let outs = f(&slice)?;
            match num_outputs {
                None => num_outputs = Some(outs.len()),
                Some(n) if n != outs.len() => {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "vmap_multi_output: closure returned {} outputs at batch index {} \
                             but {} at the first call -- output count must be fixed",
                            outs.len(),
                            i,
                            n
                        ),
                    });
                }
                Some(_) => {}
            }
            per_call.push(outs);
        }
        let num_outputs = num_outputs.unwrap_or(0);

        // Transpose: stacked[output_idx] = stack(per_call[*][output_idx], out_dim)
        let mut stacked: Vec<Tensor<T>> = Vec::with_capacity(num_outputs);
        for out_idx in 0..num_outputs {
            let mut col: Vec<Tensor<T>> = Vec::with_capacity(batch_size);
            for batch_outputs in &per_call {
                col.push(batch_outputs[out_idx].clone());
            }
            stacked.push(stack(&col, out_dim)?);
        }
        Ok(stacked)
    }
}

// ---------------------------------------------------------------------------
// per_sample_grad — vmap of grad
// ---------------------------------------------------------------------------

/// Compute per-sample gradients of a scalar loss function with respect to
/// a parameter tensor, by running the forward + backward once per sample.
///
/// This is the canonical "vmap of grad" operation requested by CL-312:
/// instead of getting a single accumulated gradient over the whole batch,
/// you get a stack of `[batch_size, *param.shape()]` per-sample gradients.
///
/// `loss_fn` takes one input slice and one parameter tensor, returns a
/// scalar loss tensor. The parameter tensor is cloned per-call with
/// `requires_grad = true` so each call accumulates an isolated gradient.
///
/// # Note on cost
///
/// This is the loop-based reference implementation: O(batch_size) forward
/// and backward calls. A future fused per-sample-grad path may use the
/// expand-trick (replicate the parameter, vmap the loss, batched backward)
/// but this version is correct and useful immediately.
pub fn per_sample_grad<T, F>(
    loss_fn: F,
    inputs: &Tensor<T>,
    param: &Tensor<T>,
    in_dim: usize,
) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>, &Tensor<T>) -> FerrotorchResult<Tensor<T>>,
{
    if in_dim >= inputs.ndim() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "per_sample_grad: in_dim {} out of bounds for input with {} dims",
                in_dim,
                inputs.ndim()
            ),
        });
    }
    let batch_size = inputs.shape()[in_dim];
    let mut grads: Vec<Tensor<T>> = Vec::with_capacity(batch_size);

    for i in 0..batch_size {
        let slice = select(inputs, in_dim, i)?;
        // Build a fresh leaf parameter that requires grad. We clone the
        // underlying storage so each iteration gets an isolated gradient
        // slot, then call the user's loss_fn with our leaf.
        let p_data = param.data_vec()?;
        let p_leaf = crate::tensor::Tensor::from_storage(
            crate::storage::TensorStorage::cpu(p_data),
            param.shape().to_vec(),
            true,
        )?;
        let loss = loss_fn(&slice, &p_leaf)?;
        if !loss.is_scalar() && loss.numel() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "per_sample_grad: loss_fn must return a scalar tensor, got shape {:?}",
                    loss.shape()
                ),
            });
        }
        loss.backward()?;
        let g = p_leaf
            .grad()?
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "per_sample_grad: parameter received no gradient -- check that loss_fn \
                      uses the parameter argument and produces a differentiable result"
                    .into(),
            })?;
        grads.push(g);
    }
    // Stack along a new leading axis so the shape is [batch, *param.shape()].
    stack(&grads, 0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::from_slice;

    /// Helper: create a tensor from data and shape.
    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        from_slice(data, shape).unwrap()
    }

    // -- select --

    #[test]
    fn test_select_axis0() {
        // [3, 4] tensor, select along axis 0
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = t(&data, &[3, 4]);

        let s0 = select(&input, 0, 0).unwrap();
        assert_eq!(s0.shape(), &[4]);
        assert_eq!(s0.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);

        let s1 = select(&input, 0, 1).unwrap();
        assert_eq!(s1.shape(), &[4]);
        assert_eq!(s1.data().unwrap(), &[5.0, 6.0, 7.0, 8.0]);

        let s2 = select(&input, 0, 2).unwrap();
        assert_eq!(s2.shape(), &[4]);
        assert_eq!(s2.data().unwrap(), &[9.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn test_select_axis1() {
        // [2, 3] tensor, select along axis 1
        let input = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);

        let s0 = select(&input, 1, 0).unwrap();
        assert_eq!(s0.shape(), &[2]);
        assert_eq!(s0.data().unwrap(), &[1.0, 4.0]);

        let s1 = select(&input, 1, 1).unwrap();
        assert_eq!(s1.data().unwrap(), &[2.0, 5.0]);

        let s2 = select(&input, 1, 2).unwrap();
        assert_eq!(s2.data().unwrap(), &[3.0, 6.0]);
    }

    #[test]
    fn test_select_3d() {
        // [2, 3, 4] tensor, select along axis 0 -> [3, 4]
        let data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let input = t(&data, &[2, 3, 4]);

        let s0 = select(&input, 0, 0).unwrap();
        assert_eq!(s0.shape(), &[3, 4]);
        let expected: Vec<f32> = (0..12).map(|x| x as f32).collect();
        assert_eq!(s0.data().unwrap(), &expected);

        let s1 = select(&input, 0, 1).unwrap();
        assert_eq!(s1.shape(), &[3, 4]);
        let expected: Vec<f32> = (12..24).map(|x| x as f32).collect();
        assert_eq!(s1.data().unwrap(), &expected);
    }

    #[test]
    // reason: select() is pure indexing (no arithmetic); 20.0 is read back
    // bit-for-bit from storage, so equality is the correct check.
    #[allow(clippy::float_cmp)]
    fn test_select_from_1d() {
        // Select from a 1-D tensor yields a scalar.
        let input = t(&[10.0, 20.0, 30.0], &[3]);
        let s = select(&input, 0, 1).unwrap();
        assert!(s.is_scalar());
        assert_eq!(s.item().unwrap(), 20.0);
    }

    #[test]
    fn test_select_invalid_dim() {
        let input = t(&[1.0, 2.0], &[2]);
        assert!(select(&input, 1, 0).is_err());
    }

    #[test]
    fn test_select_invalid_index() {
        let input = t(&[1.0, 2.0, 3.0], &[3]);
        assert!(select(&input, 0, 3).is_err());
    }

    // -- stack --

    #[test]
    fn test_stack_axis0() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        let c = t(&[5.0, 6.0], &[2]);
        let s = stack(&[a, b, c], 0).unwrap();
        assert_eq!(s.shape(), &[3, 2]);
        assert_eq!(s.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_stack_axis1() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        let s = stack(&[a, b], 1).unwrap();
        assert_eq!(s.shape(), &[2, 2]);
        // a=[1,2], b=[3,4], stacked along dim 1:
        // [[1, 3], [2, 4]]
        assert_eq!(s.data().unwrap(), &[1.0, 3.0, 2.0, 4.0]);
    }

    #[test]
    fn test_stack_2d_axis0() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let s = stack(&[a, b], 0).unwrap();
        assert_eq!(s.shape(), &[2, 2, 2]);
        assert_eq!(s.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_stack_empty_error() {
        let result: FerrotorchResult<Tensor<f32>> = stack(&[], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_stack_shape_mismatch() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0, 5.0], &[3]);
        assert!(stack(&[a, b], 0).is_err());
    }

    #[test]
    fn test_stack_invalid_dim() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        // dim=2 is out of bounds for 1-D tensors (max is 1).
        assert!(stack(&[a, b], 2).is_err());
    }

    // -- select/stack round-trip --

    #[test]
    fn test_select_stack_roundtrip() {
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = t(&data, &[3, 4]);

        // Select each row, then stack them back.
        let rows: Vec<Tensor<f32>> = (0..3).map(|i| select(&input, 0, i).unwrap()).collect();
        let reconstructed = stack(&rows, 0).unwrap();

        assert_eq!(reconstructed.shape(), input.shape());
        assert_eq!(reconstructed.data().unwrap(), input.data().unwrap());
    }

    // -- vmap --

    #[test]
    fn test_vmap_double() {
        // vmap(|x| x * 2, 0, 0) on [3, 4] tensor -> [3, 4]
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = t(&data, &[3, 4]);

        let doubled = vmap(
            |x| {
                let two = from_slice(&vec![2.0f32; x.numel()], x.shape())?;
                x * &two
            },
            0,
            0,
        )(&input)
        .unwrap();

        assert_eq!(doubled.shape(), &[3, 4]);
        let expected: Vec<f32> = data.iter().map(|x| x * 2.0).collect();
        assert_eq!(doubled.data().unwrap(), &expected);
    }

    #[test]
    fn test_vmap_sum_per_row() {
        // vmap(|x| x.sum_all(), 0, 0) on [3, 4] -> [3]
        let data: Vec<f32> = (1..=12).map(|x| x as f32).collect();
        let input = t(&data, &[3, 4]);

        let sums = vmap(|x| x.sum_all(), 0, 0)(&input).unwrap();

        assert_eq!(sums.shape(), &[3]);
        let sums_data = sums.data().unwrap();
        // Row 0: 1+2+3+4 = 10
        assert!((sums_data[0] - 10.0).abs() < 1e-6);
        // Row 1: 5+6+7+8 = 26
        assert!((sums_data[1] - 26.0).abs() < 1e-6);
        // Row 2: 9+10+11+12 = 42
        assert!((sums_data[2] - 42.0).abs() < 1e-6);
    }

    #[test]
    fn test_vmap_invalid_in_dim() {
        let input = t(&[1.0, 2.0], &[2]);
        let result = vmap(|x: &Tensor<f32>| Ok(x.clone()), 1, 0)(&input);
        assert!(result.is_err());
    }

    // -- vmap with matmul matches bmm --

    #[test]
    fn test_vmap_matmul_matches_bmm() {
        // Two batched matrices: A=[2, 3, 4], B=[2, 4, 2]
        // bmm(A, B) should equal vmap of per-element matmul.
        let a_data: Vec<f32> = (0..24).map(|x| x as f32).collect();
        let b_data: Vec<f32> = (0..16).map(|x| (x as f32) * 0.1).collect();

        let a = t(&a_data, &[2, 3, 4]);
        let b = t(&b_data, &[2, 4, 2]);

        // bmm result
        let bmm_result = a.bmm(&b).unwrap();

        // vmap2 result
        let vmap_result = vmap2(|x, y| x.matmul(y), 0, 0, 0)(&a, &b).unwrap();

        assert_eq!(bmm_result.shape(), vmap_result.shape());
        let bmm_data = bmm_result.data().unwrap();
        let vmap_data = vmap_result.data().unwrap();
        for (bv, vv) in bmm_data.iter().zip(vmap_data.iter()) {
            assert!((bv - vv).abs() < 1e-4, "bmm={bv}, vmap={vv}");
        }
    }

    // -- vmap2 --

    #[test]
    fn test_vmap2_elementwise_add() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let b = t(&[10.0, 20.0, 30.0, 40.0, 50.0, 60.0], &[3, 2]);

        let result = vmap2(|x, y| x + y, 0, 0, 0)(&a, &b).unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        assert_eq!(
            result.data().unwrap(),
            &[11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
        );
    }

    #[test]
    fn test_vmap2_batch_mismatch() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[1.0, 2.0], &[2]);

        let result = vmap2(|x, y| x + y, 0, 0, 0)(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_vmap2_invalid_dim_a() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        let result = vmap2(|x: &Tensor<f32>, y: &Tensor<f32>| x + y, 2, 0, 0)(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_vmap2_invalid_dim_b() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[3.0, 4.0], &[2]);
        let result = vmap2(|x: &Tensor<f32>, y: &Tensor<f32>| x + y, 0, 2, 0)(&a, &b);
        assert!(result.is_err());
    }

    // -- stack scalars --

    #[test]
    fn test_stack_scalars() {
        // Stacking scalars along dim 0 yields a 1-D tensor.
        use crate::creation::scalar;
        let a = scalar(1.0f32).unwrap();
        let b = scalar(2.0f32).unwrap();
        let c = scalar(3.0f32).unwrap();
        let s = stack(&[a, b, c], 0).unwrap();
        assert_eq!(s.shape(), &[3]);
        assert_eq!(s.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    // -----------------------------------------------------------------------
    // CL-312: vmap3, vmap_many, vmap_multi_output, per_sample_grad,
    // and nested vmap composability
    // -----------------------------------------------------------------------

    #[test]
    fn test_vmap3_three_way_add() {
        // f(a, b, c) = a + b + c, vmapped over dim 0.
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let c = t(&[100.0, 200.0, 300.0, 400.0], &[2, 2]);
        let result = vmap3(
            |x, y, z| {
                use crate::grad_fns::arithmetic::add;
                let xy = add(x, y)?;
                add(&xy, z)
            },
            0,
            0,
            0,
            0,
        )(&a, &b, &c)
        .unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data().unwrap(), &[111.0, 222.0, 333.0, 444.0]);
    }

    #[test]
    fn test_vmap3_batch_size_mismatch() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[1.0, 2.0, 3.0], &[3]);
        let c = t(&[1.0, 2.0], &[2]);
        let result = vmap3(|x, _y, _z| Ok(x.clone()), 0, 0, 0, 0)(&a, &b, &c);
        assert!(result.is_err());
    }

    #[test]
    fn test_vmap3_invalid_dim() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[1.0, 2.0], &[2]);
        let c = t(&[1.0, 2.0], &[2]);
        let result = vmap3(|x, _y, _z| Ok(x.clone()), 5, 0, 0, 0)(&a, &b, &c);
        assert!(result.is_err());
    }

    #[test]
    fn test_vmap_many_four_inputs() {
        // sum of four batched vectors
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[10.0, 20.0, 30.0, 40.0], &[2, 2]);
        let c = t(&[100.0, 200.0, 300.0, 400.0], &[2, 2]);
        let d = t(&[1000.0, 2000.0, 3000.0, 4000.0], &[2, 2]);
        let result = vmap_many(
            |slices: &[Tensor<f32>]| {
                use crate::grad_fns::arithmetic::add;
                let mut acc = slices[0].clone();
                for s in &slices[1..] {
                    acc = add(&acc, s)?;
                }
                Ok(acc)
            },
            vec![0, 0, 0, 0],
            0,
        )(&[&a, &b, &c, &d])
        .unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data().unwrap(), &[1111.0, 2222.0, 3333.0, 4444.0]);
    }

    #[test]
    fn test_vmap_many_input_count_mismatch() {
        let a = t(&[1.0, 2.0], &[2]);
        let b = t(&[1.0, 2.0], &[2]);
        let result = vmap_many(
            |slices: &[Tensor<f32>]| Ok(slices[0].clone()),
            vec![0, 0, 0],
            0,
        )(&[&a, &b]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vmap_many_empty_inputs_errors() {
        let result = vmap_many(|slices: &[Tensor<f32>]| Ok(slices[0].clone()), vec![], 0)(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_vmap_multi_output_two_outputs() {
        // f(x) = (x, x*x) — duplicates input and squares it.
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let outs = vmap_multi_output(
            |slice| {
                use crate::grad_fns::arithmetic::mul;
                let sq = mul(slice, slice)?;
                Ok(vec![slice.clone(), sq])
            },
            0,
            0,
        )(&x)
        .unwrap();
        assert_eq!(outs.len(), 2);
        assert_eq!(outs[0].shape(), &[2, 2]);
        assert_eq!(outs[0].data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(outs[1].shape(), &[2, 2]);
        assert_eq!(outs[1].data().unwrap(), &[1.0, 4.0, 9.0, 16.0]);
    }

    #[test]
    fn test_vmap_multi_output_single_output_acts_like_vmap() {
        let x = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let outs = vmap_multi_output(|s| Ok(vec![s.clone()]), 0, 0)(&x).unwrap();
        assert_eq!(outs.len(), 1);
        assert_eq!(outs[0].shape(), &[2, 2]);
        assert_eq!(outs[0].data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_vmap_nested_composability() {
        // vmap(vmap(f)) over a 3-D tensor.
        // Inner vmap: identity per-row.
        // Outer vmap: identity per-batch.
        // Net: identity over a [2, 3, 4] tensor.
        let x_data: Vec<f32> = (0..24).map(|v| v as f32).collect();
        let x = t(&x_data, &[2, 3, 4]);

        // Outer vmap takes each [3, 4] slice and applies an inner vmap
        // that takes each [4] row and returns it unchanged.
        let result = vmap(
            |outer_slice| vmap(|inner_slice| Ok(inner_slice.clone()), 0, 0)(outer_slice),
            0,
            0,
        )(&x)
        .unwrap();
        assert_eq!(result.shape(), &[2, 3, 4]);
        // Element-by-element identity check.
        let r = result.data().unwrap();
        for (i, &v) in r.iter().enumerate() {
            assert!(
                (v - x_data[i]).abs() < 1e-6,
                "mismatch at {i}: {v} vs {}",
                x_data[i]
            );
        }
    }

    #[test]
    fn test_vmap_nested_double_negation() {
        // vmap(vmap(neg)) over a [2, 3] tensor — outer vmaps along dim 0,
        // inner vmaps along dim 0 of each [3] slice. Result should be
        // -original elementwise.
        let x = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = vmap(
            |outer_slice| vmap(crate::grad_fns::arithmetic::neg, 0, 0)(outer_slice),
            0,
            0,
        )(&x)
        .unwrap();
        assert_eq!(result.shape(), &[2, 3]);
        assert_eq!(
            result.data().unwrap(),
            &[-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]
        );
    }

    #[test]
    fn test_per_sample_grad_simple_quadratic() {
        // loss(x, p) = sum((x * p)^2)
        // d/dp = 2 * sum(x * p) * x ... actually d(sum((xp)^2))/dp_i
        //       = 2 * x_i^2 * p_i (per-element)
        // For x in [1, 2, 3] and p = [0.5, 0.5, 0.5]:
        //   d/dp_i = 2 * x_i^2 * 0.5 = x_i^2
        // Per-sample with batch [[1,2,3], [4,5,6]]:
        //   sample 0: [1, 4, 9]
        //   sample 1: [16, 25, 36]
        let inputs = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let param = t(&[0.5, 0.5, 0.5], &[3]);
        let grads = per_sample_grad(
            |x: &Tensor<f32>, p: &Tensor<f32>| {
                use crate::grad_fns::arithmetic::mul;
                use crate::grad_fns::reduction::sum;
                let xp = mul(x, p)?;
                let sq = mul(&xp, &xp)?;
                sum(&sq)
            },
            &inputs,
            &param,
            0,
        )
        .unwrap();
        assert_eq!(grads.shape(), &[2, 3]);
        let g = grads.data().unwrap();
        // sample 0
        assert!((g[0] - 1.0).abs() < 1e-4);
        assert!((g[1] - 4.0).abs() < 1e-4);
        assert!((g[2] - 9.0).abs() < 1e-4);
        // sample 1
        assert!((g[3] - 16.0).abs() < 1e-4);
        assert!((g[4] - 25.0).abs() < 1e-4);
        assert!((g[5] - 36.0).abs() < 1e-4);
    }

    #[test]
    fn test_per_sample_grad_invalid_dim() {
        let x = t(&[1.0, 2.0], &[2]);
        let p = t(&[0.5], &[1]);
        let result = per_sample_grad(|_x, p| Ok(p.clone()), &x, &p, 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_per_sample_grad_non_scalar_loss_errors() {
        // loss_fn returns a non-scalar -- should error.
        let x = t(&[1.0, 2.0], &[2]);
        let p = t(&[0.5], &[1]);
        let result = per_sample_grad(|x: &Tensor<f32>, _p: &Tensor<f32>| Ok(x.clone()), &x, &p, 0);
        assert!(result.is_err());
    }
}
