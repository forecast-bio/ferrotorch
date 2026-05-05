//! Helpers for propagating the meta device through tensor operations.
//!
//! When all inputs to an operation are on `Device::Meta`, the op produces
//! a meta tensor with the correct output shape and skips the data
//! computation entirely. When inputs are mixed (some meta, some real),
//! the op returns an error since meta tensors carry no data.
//!
//! Each helper returns:
//! - `Ok(Some(t))` — the inputs were all meta, here is the meta result
//! - `Ok(None)` — no inputs were meta, the caller should run the normal
//!   computation path
//! - `Err(e)` — inputs were mixed, or the requested op is invalid for the
//!   given shapes
//!
//! Op authors call these at the top of their implementations:
//!
//! ```ignore
//! pub fn add<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
//!     if let Some(out) = meta_propagate::binary_broadcast(a, b)? {
//!         return Ok(out);
//!     }
//!     // ... normal path ...
//! }
//! ```
//!
//! CL-500 builds on the meta device foundation in CL-395.

use crate::creation;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::shape::broadcast_shapes;
use crate::tensor::Tensor;

/// Meta-device fast path for unary ops that produce an output of the same
/// shape as the input (most elementwise activations, neg, abs, sqrt, etc.).
pub fn unary_same_shape<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Option<Tensor<T>>> {
    if input.is_meta() {
        Ok(Some(creation::zeros_meta(input.shape())?))
    } else {
        Ok(None)
    }
}

/// Meta-device fast path for binary broadcast ops (add, sub, mul, div, etc.).
///
/// Returns the broadcast meta output when both inputs are meta. Errors
/// when only one input is meta — there is no defined behavior for mixing
/// real and meta tensors in an op, since the real side has data the
/// meta side does not.
pub fn binary_broadcast<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Option<Tensor<T>>> {
    match (a.is_meta(), b.is_meta()) {
        (true, true) => {
            let out_shape = broadcast_shapes(a.shape(), b.shape())?;
            Ok(Some(creation::zeros_meta(&out_shape)?))
        }
        (false, false) => Ok(None),
        _ => Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        }),
    }
}

/// Meta-device fast path for reductions over a single dimension.
///
/// Returns a meta tensor whose shape is the input shape with the given
/// dim removed (when `keepdim == false`) or reduced to size 1 (when
/// `keepdim == true`). Mirrors `sum_dim`/`mean_dim`/`max_dim` shape rules.
pub fn reduce_dim<T: Float>(
    input: &Tensor<T>,
    dim: i64,
    keepdim: bool,
) -> FerrotorchResult<Option<Tensor<T>>> {
    if !input.is_meta() {
        return Ok(None);
    }
    let ndim = input.ndim();
    if ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "meta_propagate::reduce_dim: cannot reduce a scalar tensor".into(),
        });
    }
    let norm_dim = if dim < 0 {
        (ndim as i64 + dim) as usize
    } else {
        dim as usize
    };
    if norm_dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "meta_propagate::reduce_dim: dim {dim} out of bounds for tensor with {ndim} dimensions"
            ),
        });
    }
    let mut out_shape: Vec<usize> = input.shape().to_vec();
    if keepdim {
        out_shape[norm_dim] = 1;
    } else {
        out_shape.remove(norm_dim);
    }
    Ok(Some(creation::zeros_meta(&out_shape)?))
}

/// Meta-device fast path for full reductions (sum, mean, prod) that
/// collapse the entire tensor to a scalar.
pub fn reduce_all<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Option<Tensor<T>>> {
    if input.is_meta() {
        // Scalar (0-D) shape.
        Ok(Some(creation::zeros_meta(&[])?))
    } else {
        Ok(None)
    }
}

/// Meta-device fast path for matmul-style ops following PyTorch's
/// shape rules:
///
/// - 1-D × 1-D → scalar (dot product)
/// - 2-D × 1-D → 1-D vector
/// - 1-D × 2-D → 1-D vector
/// - 2-D × 2-D → 2-D matrix
/// - N-D × M-D → broadcast batch dims, contract last-two/first-two of the
///   trailing pair
pub fn matmul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Option<Tensor<T>>> {
    match (a.is_meta(), b.is_meta()) {
        (false, false) => return Ok(None),
        (true, true) => {}
        _ => {
            return Err(FerrotorchError::DeviceMismatch {
                expected: a.device(),
                got: b.device(),
            });
        }
    }

    let a_shape = a.shape();
    let b_shape = b.shape();
    let a_ndim = a_shape.len();
    let b_ndim = b_shape.len();

    if a_ndim == 0 || b_ndim == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "meta_propagate::matmul: scalar operands not supported, got {a_shape:?} and {b_shape:?}"
            ),
        });
    }

    let out_shape: Vec<usize> = match (a_ndim, b_ndim) {
        (1, 1) => {
            // dot product → scalar
            if a_shape[0] != b_shape[0] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "meta_propagate::matmul: 1D dot dimension mismatch {} vs {}",
                        a_shape[0], b_shape[0]
                    ),
                });
            }
            vec![]
        }
        (2, 1) => {
            // [m, k] @ [k] → [m]
            if a_shape[1] != b_shape[0] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "meta_propagate::matmul: mv dim mismatch {} vs {}",
                        a_shape[1], b_shape[0]
                    ),
                });
            }
            vec![a_shape[0]]
        }
        (1, 2) => {
            // [k] @ [k, n] → [n]
            if a_shape[0] != b_shape[0] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "meta_propagate::matmul: vm dim mismatch {} vs {}",
                        a_shape[0], b_shape[0]
                    ),
                });
            }
            vec![b_shape[1]]
        }
        (2, 2) => {
            // [m, k] @ [k, n] → [m, n]
            if a_shape[1] != b_shape[0] {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "meta_propagate::matmul: mm inner dim mismatch {} vs {}",
                        a_shape[1], b_shape[0]
                    ),
                });
            }
            vec![a_shape[0], b_shape[1]]
        }
        _ => {
            // Batched: broadcast leading dims, contract last two of the trailing pair.
            // a: [..., m, k], b: [..., k, n]
            let a_inner_k = a_shape[a_ndim - 1];
            let b_inner_k = b_shape[b_ndim - 2];
            if a_inner_k != b_inner_k {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "meta_propagate::matmul: batched inner dim mismatch {a_inner_k} vs {b_inner_k}"
                    ),
                });
            }
            let m = a_shape[a_ndim - 2];
            let n = b_shape[b_ndim - 1];
            // Broadcast batch dims (everything except the last 2 axes).
            let a_batch = &a_shape[..a_ndim - 2];
            let b_batch = &b_shape[..b_ndim - 2];
            let mut batch = broadcast_shapes(a_batch, b_batch)?;
            batch.push(m);
            batch.push(n);
            batch
        }
    };

    Ok(Some(creation::zeros_meta(&out_shape)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tensor;

    fn meta<T: Float>(shape: &[usize]) -> Tensor<T> {
        creation::zeros_meta(shape).unwrap()
    }

    fn cpu(shape: &[usize]) -> Tensor<f32> {
        creation::zeros(shape).unwrap()
    }

    // -----------------------------------------------------------------------
    // unary_same_shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_unary_same_shape_meta_returns_meta() {
        let m: Tensor<f32> = meta(&[3, 4]);
        let out = unary_same_shape(&m).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_unary_same_shape_cpu_returns_none() {
        let t: Tensor<f32> = cpu(&[3, 4]);
        let out = unary_same_shape(&t).unwrap();
        assert!(out.is_none());
    }

    // -----------------------------------------------------------------------
    // binary_broadcast
    // -----------------------------------------------------------------------

    #[test]
    fn test_binary_broadcast_both_meta_returns_broadcasted() {
        let a: Tensor<f32> = meta(&[3, 1]);
        let b: Tensor<f32> = meta(&[1, 4]);
        let out = binary_broadcast(&a, &b).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_binary_broadcast_neither_meta_returns_none() {
        let a: Tensor<f32> = cpu(&[2, 3]);
        let b: Tensor<f32> = cpu(&[2, 3]);
        let out = binary_broadcast(&a, &b).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_binary_broadcast_mixed_errors() {
        let a: Tensor<f32> = meta(&[2, 3]);
        let b: Tensor<f32> = cpu(&[2, 3]);
        let result = binary_broadcast(&a, &b);
        assert!(result.is_err());
    }

    #[test]
    fn test_binary_broadcast_meta_shape_mismatch_errors() {
        let a: Tensor<f32> = meta(&[3, 4]);
        let b: Tensor<f32> = meta(&[5, 6]);
        let result = binary_broadcast(&a, &b);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // reduce_dim
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_dim_meta_removes_axis() {
        let m: Tensor<f32> = meta(&[2, 3, 4]);
        let out = reduce_dim(&m, 1, false).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[2, 4]);
    }

    #[test]
    fn test_reduce_dim_meta_keepdim_keeps_size_one() {
        let m: Tensor<f32> = meta(&[2, 3, 4]);
        let out = reduce_dim(&m, 1, true).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[2, 1, 4]);
    }

    #[test]
    fn test_reduce_dim_negative_axis() {
        let m: Tensor<f32> = meta(&[2, 3, 4]);
        let out = reduce_dim(&m, -1, false).unwrap().unwrap();
        assert_eq!(out.shape(), &[2, 3]);
    }

    #[test]
    fn test_reduce_dim_cpu_returns_none() {
        let t: Tensor<f32> = cpu(&[2, 3]);
        let out = reduce_dim(&t, 0, false).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_reduce_dim_out_of_bounds_errors() {
        let m: Tensor<f32> = meta(&[2, 3]);
        assert!(reduce_dim(&m, 5, false).is_err());
    }

    // -----------------------------------------------------------------------
    // reduce_all
    // -----------------------------------------------------------------------

    #[test]
    fn test_reduce_all_meta_returns_scalar() {
        let m: Tensor<f32> = meta(&[2, 3, 4]);
        let out = reduce_all(&m).unwrap().unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), [] as [usize; 0]);
    }

    #[test]
    fn test_reduce_all_cpu_returns_none() {
        let t: Tensor<f32> = cpu(&[2, 3]);
        let out = reduce_all(&t).unwrap();
        assert!(out.is_none());
    }

    // -----------------------------------------------------------------------
    // matmul shape rules
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_1d_1d_dot() {
        let a: Tensor<f32> = meta(&[5]);
        let b: Tensor<f32> = meta(&[5]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), [] as [usize; 0]);
    }

    #[test]
    fn test_matmul_2d_1d_mv() {
        let a: Tensor<f32> = meta(&[3, 5]);
        let b: Tensor<f32> = meta(&[5]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), &[3]);
    }

    #[test]
    fn test_matmul_1d_2d_vm() {
        let a: Tensor<f32> = meta(&[5]);
        let b: Tensor<f32> = meta(&[5, 4]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), &[4]);
    }

    #[test]
    fn test_matmul_2d_2d_mm() {
        let a: Tensor<f32> = meta(&[3, 5]);
        let b: Tensor<f32> = meta(&[5, 4]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), &[3, 4]);
    }

    #[test]
    fn test_matmul_batched_3d() {
        let a: Tensor<f32> = meta(&[2, 3, 5]);
        let b: Tensor<f32> = meta(&[2, 5, 4]);
        let out = matmul(&a, &b).unwrap().unwrap();
        assert_eq!(out.shape(), &[2, 3, 4]);
    }

    #[test]
    fn test_matmul_batched_with_broadcast() {
        let a: Tensor<f32> = meta(&[1, 3, 5]);
        let b: Tensor<f32> = meta(&[4, 5, 7]);
        let out = matmul(&a, &b).unwrap().unwrap();
        // Batch broadcast (1, 4) → 4, then 3×5 @ 5×7 → 3×7
        assert_eq!(out.shape(), &[4, 3, 7]);
    }

    #[test]
    fn test_matmul_inner_dim_mismatch_errors() {
        let a: Tensor<f32> = meta(&[3, 5]);
        let b: Tensor<f32> = meta(&[6, 4]);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_matmul_cpu_returns_none() {
        let a: Tensor<f32> = cpu(&[3, 5]);
        let b: Tensor<f32> = cpu(&[5, 4]);
        let out = matmul(&a, &b).unwrap();
        assert!(out.is_none());
    }

    #[test]
    fn test_matmul_mixed_meta_errors() {
        let a: Tensor<f32> = meta(&[3, 5]);
        let b: Tensor<f32> = cpu(&[5, 4]);
        assert!(matmul(&a, &b).is_err());
    }

    // -----------------------------------------------------------------------
    // End-to-end pipeline tests through the actual instrumented ops.
    //
    // These verify that meta tensors propagate correctly through the public
    // arithmetic, reduction, linalg, and activation entry points without
    // ever touching data.
    // -----------------------------------------------------------------------

    #[test]
    fn test_e2e_meta_arithmetic_chain() {
        use crate::grad_fns::arithmetic::{add, mul, neg, sqrt};

        let a: Tensor<f32> = meta(&[3, 4]);
        let b: Tensor<f32> = meta(&[3, 4]);
        let c = add(&a, &b).unwrap();
        let d = mul(&c, &a).unwrap();
        let e = neg(&d).unwrap();
        let f = sqrt(&e).unwrap();
        assert!(f.is_meta());
        assert_eq!(f.shape(), &[3, 4]);
    }

    #[test]
    fn test_e2e_meta_arithmetic_with_broadcast() {
        use crate::grad_fns::arithmetic::add;

        let a: Tensor<f32> = meta(&[5, 1, 7]);
        let b: Tensor<f32> = meta(&[3, 1]);
        let out = add(&a, &b).unwrap();
        assert!(out.is_meta());
        // Broadcast: [5, 1, 7] x [3, 1] -> [5, 3, 7]
        assert_eq!(out.shape(), &[5, 3, 7]);
    }

    #[test]
    fn test_e2e_meta_reductions() {
        use crate::grad_fns::reduction::{mean_dim, sum, sum_dim};

        let x: Tensor<f32> = meta(&[2, 3, 4]);
        let s = sum(&x).unwrap();
        assert!(s.is_meta());
        assert_eq!(s.shape(), [] as [usize; 0]);

        let s2 = sum_dim(&x, 1, false).unwrap();
        assert!(s2.is_meta());
        assert_eq!(s2.shape(), &[2, 4]);

        let m = mean_dim(&x, 2, true).unwrap();
        assert!(m.is_meta());
        assert_eq!(m.shape(), &[2, 3, 1]);
    }

    #[test]
    fn test_e2e_meta_matmul() {
        use crate::ops::linalg::matmul as op_matmul;

        let a: Tensor<f32> = meta(&[8, 16]);
        let b: Tensor<f32> = meta(&[16, 32]);
        let out = op_matmul(&a, &b).unwrap();
        assert!(out.is_meta());
        assert_eq!(out.shape(), &[8, 32]);
    }

    #[test]
    fn test_e2e_meta_activations() {
        use crate::grad_fns::activation::{gelu, relu, sigmoid, silu, softmax, tanh};

        let x: Tensor<f32> = meta(&[2, 5]);
        for op_out in [
            relu(&x).unwrap(),
            sigmoid(&x).unwrap(),
            tanh(&x).unwrap(),
            gelu(&x).unwrap(),
            silu(&x).unwrap(),
            softmax(&x).unwrap(),
        ] {
            assert!(op_out.is_meta());
            assert_eq!(op_out.shape(), &[2, 5]);
        }
    }

    #[test]
    fn test_e2e_meta_mlp_dry_run() {
        // Simulate building a tiny MLP and running its forward on meta
        // inputs to determine output shape. No real allocation happens
        // for the activations, only for the parameter tensors (which we
        // also keep on meta). This is the canonical use case for the
        // meta device.
        use crate::grad_fns::activation::relu;
        use crate::grad_fns::arithmetic::add;
        use crate::ops::linalg::matmul as op_matmul;

        // Layer 1: 64 -> 32
        let x: Tensor<f32> = meta(&[16, 64]); // batch=16
        let w1: Tensor<f32> = meta(&[64, 32]);
        let b1: Tensor<f32> = meta(&[32]);
        let h1 = add(&op_matmul(&x, &w1).unwrap(), &b1).unwrap();
        let h1_relu = relu(&h1).unwrap();
        assert!(h1_relu.is_meta());
        assert_eq!(h1_relu.shape(), &[16, 32]);

        // Layer 2: 32 -> 10
        let w2: Tensor<f32> = meta(&[32, 10]);
        let b2: Tensor<f32> = meta(&[10]);
        let logits = add(&op_matmul(&h1_relu, &w2).unwrap(), &b2).unwrap();
        assert!(logits.is_meta());
        assert_eq!(logits.shape(), &[16, 10]);
    }
}
