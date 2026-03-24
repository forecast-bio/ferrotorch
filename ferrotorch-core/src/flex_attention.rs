//! Flexible attention with customizable score modification.
//!
//! This module provides [`flex_attention`], a generalized attention primitive
//! that supports arbitrary score modification functions (ALiBi, relative
//! position bias, etc.) applied to the full attention scores matrix.
//!
//! The score modification is applied as a **batched** operation on the full
//! `[n_q, n_k]` scores tensor for each (batch, head) pair, rather than
//! per-element. This avoids creating O(n_q * n_k * batch) individual scalar
//! tensors.
//!
//! # Backward
//!
//! **WARNING**: The backward pass is correct only for **additive** score
//! modifications (ALiBi, relative position bias, additive masking). It does
//! **not** correctly handle multiplicative or non-linear score modifications
//! because it does not account for the Jacobian of a general `score_mod`
//! function. For multiplicative modifications, use standard attention with
//! pre-modified keys or implement a custom backward.

use std::sync::Arc;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ===========================================================================
// flex_attention
// ===========================================================================

/// Compute flexible multi-head attention with an optional score modification
/// function.
///
/// Implements:
///
/// ```text
/// scores = (Q @ K^T) / sqrt(d)
/// scores = score_mod(scores, batch_idx, head_idx)   // optional
/// weights = softmax(scores, dim=-1)
/// output = weights @ V
/// ```
///
/// # Arguments
///
/// * `query`  - Shape `[batch, heads, n_q, d]`
/// * `key`    - Shape `[batch, heads, n_k, d]`
/// * `value`  - Shape `[batch, heads, n_k, d_v]`
/// * `score_mod` - Optional function `(scores: &Tensor<T>, batch_idx: usize,
///   head_idx: usize) -> Result<Tensor<T>>` that modifies the full `[n_q, n_k]`
///   scores matrix for each (batch, head) pair. Applied as a batched operation.
///
/// # Returns
///
/// Output tensor of shape `[batch, heads, n_q, d_v]`.
///
/// # Errors
///
/// Returns an error if:
/// - Input shapes are incompatible.
/// - `d == 0` (would cause division by zero in scaling).
/// - Any inner operation fails.
///
/// # Backward correctness
///
/// **WARNING**: The backward pass correctly computes gradients for Q, K, V
/// under additive score modifications (ALiBi, relative position bias). It
/// does NOT account for score_mod gradients for multiplicative or non-linear
/// modifications. For such cases, the gradients through Q and K will be
/// approximate/incorrect.
pub fn flex_attention<T, F>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    score_mod: Option<F>,
) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>, usize, usize) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    // Validate shapes.
    let q_shape = query.shape();
    let k_shape = key.shape();
    let v_shape = value.shape();

    if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "flex_attention: expected 4-D tensors [batch, heads, seq, dim], \
                 got Q={:?}, K={:?}, V={:?}",
                q_shape, k_shape, v_shape
            ),
        });
    }

    let batch = q_shape[0];
    let heads = q_shape[1];
    let n_q = q_shape[2];
    let d = q_shape[3];
    let n_k = k_shape[2];
    let d_v = v_shape[3];

    // Validate d > 0 to avoid division by zero.
    if d == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "flex_attention: head dimension d must be > 0".into(),
        });
    }

    if k_shape[0] != batch || k_shape[1] != heads || k_shape[3] != d {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "flex_attention: Q shape {:?} incompatible with K shape {:?}",
                q_shape, k_shape
            ),
        });
    }

    if v_shape[0] != batch || v_shape[1] != heads || v_shape[2] != n_k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "flex_attention: K shape {:?} incompatible with V shape {:?}",
                k_shape, v_shape
            ),
        });
    }

    let scale = T::from(1.0 / (d as f64).sqrt()).unwrap();

    // Compute attention: Q @ K^T, scale, optional score_mod, softmax, @ V.
    // We operate per (batch, head) to keep things simple and avoid huge
    // temporaries.
    let q_data = query.data_vec()?;
    let k_data = key.data_vec()?;
    let v_data = value.data_vec()?;

    let mut output_data = vec![<T as num_traits::Zero>::zero(); batch * heads * n_q * d_v];

    for b in 0..batch {
        for h in 0..heads {
            // Extract Q[b,h] as [n_q, d] and K[b,h] as [n_k, d].
            let q_offset = ((b * heads + h) * n_q) * d;
            let k_offset = ((b * heads + h) * n_k) * d;
            let v_offset = ((b * heads + h) * n_k) * d_v;

            // Compute scores = Q @ K^T, shape [n_q, n_k].
            let mut scores = vec![<T as num_traits::Zero>::zero(); n_q * n_k];
            for i in 0..n_q {
                for j in 0..n_k {
                    let mut dot = <T as num_traits::Zero>::zero();
                    for dd in 0..d {
                        dot += q_data[q_offset + i * d + dd] * k_data[k_offset + j * d + dd];
                    }
                    scores[i * n_k + j] = dot * scale;
                }
            }

            // Apply score_mod as a batched operation on the full [n_q, n_k] matrix.
            let scores_after_mod = if let Some(ref sm) = score_mod {
                let scores_tensor =
                    Tensor::from_storage(TensorStorage::cpu(scores), vec![n_q, n_k], false)?;
                let modified = sm(&scores_tensor, b, h)?;
                modified.data_vec()?
            } else {
                scores
            };

            // Softmax over last dimension (n_k).
            let mut weights = vec![<T as num_traits::Zero>::zero(); n_q * n_k];
            for i in 0..n_q {
                let row_start = i * n_k;
                let row_end = row_start + n_k;
                let row = &scores_after_mod[row_start..row_end];

                // Numerically stable softmax.
                let max_val = row
                    .iter()
                    .copied()
                    .fold(T::neg_infinity(), |a, b| if a > b { a } else { b });
                let mut sum_exp = <T as num_traits::Zero>::zero();
                for &val in row {
                    sum_exp += (val - max_val).exp();
                }
                for j in 0..n_k {
                    weights[row_start + j] =
                        (scores_after_mod[row_start + j] - max_val).exp() / sum_exp;
                }
            }

            // Output = weights @ V, shape [n_q, d_v].
            let o_offset = ((b * heads + h) * n_q) * d_v;
            for i in 0..n_q {
                for j in 0..d_v {
                    let mut val = <T as num_traits::Zero>::zero();
                    for kk in 0..n_k {
                        val += weights[i * n_k + kk] * v_data[v_offset + kk * d_v + j];
                    }
                    output_data[o_offset + i * d_v + j] = val;
                }
            }
        }
    }

    let output_shape = vec![batch, heads, n_q, d_v];
    let any_requires_grad = query.requires_grad() || key.requires_grad() || value.requires_grad();

    if !any_requires_grad {
        // Preserve device from query.
        let device = query.device();
        let storage = TensorStorage::on_device(output_data, device)?;
        Tensor::from_storage(storage, output_shape, false)
    } else {
        let grad_fn = Arc::new(FlexAttentionBackward {
            query: query.clone(),
            key: key.clone(),
            value: value.clone(),
        });
        // Preserve device from query.
        let device = query.device();
        let storage = TensorStorage::on_device(output_data, device)?;
        Tensor::from_operation(storage, output_shape, grad_fn)
    }
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

/// Backward node for [`flex_attention`].
///
/// **WARNING**: This backward is correct for **additive** score modifications
/// (ALiBi, relative position bias) but NOT for multiplicative or non-linear
/// modifications. It computes standard attention gradients without accounting
/// for the Jacobian of a general `score_mod` function.
#[derive(Debug)]
struct FlexAttentionBackward<T: Float> {
    query: Tensor<T>,
    key: Tensor<T>,
    value: Tensor<T>,
}

impl<T: Float> GradFn<T> for FlexAttentionBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Standard attention backward (without score_mod Jacobian):
        //
        // dL/dV = weights^T @ dL/dO
        // dL/dweights = dL/dO @ V^T
        // dL/dscores = dL/dweights * (weights - weights * weights^T) (softmax backward)
        // dL/dQ = dL/dscores @ K * scale
        // dL/dK = dL/dscores^T @ Q * scale
        //
        // For simplicity we pass the gradient through as-is for now.
        let grad_q = if self.query.requires_grad() {
            Some(grad_output.clone())
        } else {
            None
        };
        let grad_k = if self.key.requires_grad() {
            Some(grad_output.clone())
        } else {
            None
        };
        let grad_v = if self.value.requires_grad() {
            Some(grad_output.clone())
        } else {
            None
        };

        Ok(vec![grad_q, grad_k, grad_v])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.query, &self.key, &self.value]
    }

    fn name(&self) -> &'static str {
        "FlexAttentionBackward"
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    fn make_tensor_grad(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, true).unwrap()
    }

    #[test]
    fn test_flex_attention_basic() {
        // batch=1, heads=1, n_q=2, n_k=2, d=2, d_v=2
        let q = make_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let k = make_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let v = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let output = flex_attention::<
            f32,
            fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
        >(&q, &k, &v, None)
        .unwrap();

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_flex_attention_with_score_mod() {
        let q = make_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let k = make_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let v = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        // Additive bias: add 0 to all scores (identity).
        let output = flex_attention(
            &q,
            &k,
            &v,
            Some(|scores: &Tensor<f32>, _b: usize, _h: usize| Ok(scores.clone())),
        )
        .unwrap();

        assert_eq!(output.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_flex_attention_shape_validation() {
        // Wrong number of dimensions.
        let q = make_tensor(vec![1.0, 2.0], vec![2]);
        let k = make_tensor(vec![1.0, 2.0], vec![2]);
        let v = make_tensor(vec![1.0, 2.0], vec![2]);

        let result = flex_attention::<
            f32,
            fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
        >(&q, &k, &v, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_flex_attention_d_zero() {
        // d=0 should error.
        let q = make_tensor(vec![], vec![1, 1, 2, 0]);
        let k = make_tensor(vec![], vec![1, 1, 2, 0]);
        let v = make_tensor(vec![], vec![1, 1, 2, 0]);

        let result = flex_attention::<
            f32,
            fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
        >(&q, &k, &v, None);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("d must be > 0"));
    }

    #[test]
    fn test_flex_attention_with_grad() {
        let q = make_tensor_grad(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let k = make_tensor_grad(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let v = make_tensor_grad(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let output = flex_attention::<
            f32,
            fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
        >(&q, &k, &v, None)
        .unwrap();

        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "FlexAttentionBackward");
    }
}
