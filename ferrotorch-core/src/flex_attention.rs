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
//! Gradients flow through the standard autograd chain: bmm → mul → softmax →
//! bmm. The backward correctly handles Q, K, V gradients for additive,
//! multiplicative, and arbitrary differentiable score modifications, because
//! the score_mod callback's own ops record their backward in the graph as
//! they execute.
//!
//! If `score_mod` returns a non-differentiable result (e.g. it constructs
//! a fresh tensor without requires_grad), gradients will not flow back into
//! its inputs — but Q/K/V gradients will still be correct relative to the
//! modified scores.

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::Tensor;

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
/// Gradients flow through the standard autograd chain (bmm → mul → softmax →
/// bmm). Q, K, V gradients are correct for any score modification whose ops
/// are themselves differentiable, including multiplicative and non-linear
/// modifications. There is no custom backward node — autograd composes
/// the per-op backwards automatically.
///
/// # Device behavior
///
/// Runs end-to-end on the input device (CPU or CUDA). On CUDA, the per-op
/// dispatch goes through cuBLAS for `bmm`, the native softmax kernel, and
/// the strided `cat` kernel for score_mod reassembly. The previous
/// implementation downloaded Q, K, V to CPU and ran nested loops; the
/// current path stays on-device.
#[allow(clippy::needless_pass_by_value)] // reason: pub API stability — Option<F> by value is the natural ergonomic shape; re-exported via ferrotorch-nn
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
                 got Q={q_shape:?}, K={k_shape:?}, V={v_shape:?}"
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
                "flex_attention: Q shape {q_shape:?} incompatible with K shape {k_shape:?}"
            ),
        });
    }

    if v_shape[0] != batch || v_shape[1] != heads || v_shape[2] != n_k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "flex_attention: K shape {k_shape:?} incompatible with V shape {v_shape:?}"
            ),
        });
    }

    if query.device() != key.device() || query.device() != value.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: query.device(),
            got: key.device(),
        });
    }

    let scale = T::from(1.0 / (d as f64).sqrt()).unwrap();
    let device = query.device();
    let bh = batch * heads;

    // ----------------------------------------------------------------------
    // GPU-aware path: compose the attention computation from existing
    // device-aware tensor ops. Each step (view_reshape, transpose, bmm,
    // mul, softmax, cat) dispatches to its native CUDA kernel when the
    // input is on GPU. The previous implementation downloaded Q, K, V to
    // CPU and ran nested loops; this version stays on-device end-to-end
    // (modulo the score_mod callback dispatch, which still runs per
    // (batch, head) to preserve the existing API contract).
    //
    // Bonus: the previous FlexAttentionBackward was a passthrough and did
    // not compute correct gradients. By composing differentiable ops
    // (bmm_differentiable + softmax + mul), the autograd chain now produces
    // mathematically correct gradients for Q, K, V automatically — no
    // custom backward node required.
    // ----------------------------------------------------------------------

    // Reshape [B, H, N, D] -> [B*H, N, D] for bmm. We use the differentiable
    // `reshape` (grad_fns::shape::reshape) so grad flows back to Q/K/V — the
    // bare `view_reshape` strips grad_fn and would sever the autograd chain.
    // When no grad is required, reshape internally falls back to view_reshape
    // (zero-copy on any device).
    let q3 = crate::grad_fns::shape::reshape(query, &[bh as isize, n_q as isize, d as isize])?;
    let k3 = crate::grad_fns::shape::reshape(key, &[bh as isize, n_k as isize, d as isize])?;
    let v3 = crate::grad_fns::shape::reshape(value, &[bh as isize, n_k as isize, d_v as isize])?;

    // K^T: swap last two dims of [B*H, n_k, d] -> [B*H, d, n_k]. transpose
    // is a zero-copy stride view; bmm() materializes via contiguous() if
    // needed (which on CUDA still goes through CPU for now — see #496 —
    // but on contiguous inputs the rest of the path is fully on-device).
    let k3_t = k3.transpose(1, 2)?;

    // Q @ K^T: [B*H, n_q, d] @ [B*H, d, n_k] -> [B*H, n_q, n_k]. bmm
    // dispatches to cuBLAS SgemmStridedBatched on CUDA.
    let scores3 = crate::grad_fns::linalg::bmm_differentiable(&q3, &k3_t)?;

    // Multiply by 1/sqrt(d). scale is a scalar tensor on the same device
    // as the inputs; mul broadcasts.
    let scale_t = crate::creation::scalar(scale)?.to(device)?;
    let scores3_scaled = crate::grad_fns::arithmetic::mul(&scores3, &scale_t)?;

    // Reshape to [B, H, n_q, n_k] for score_mod and softmax. Differentiable
    // reshape so grad continues to flow.
    let scores4 = crate::grad_fns::shape::reshape(
        &scores3_scaled,
        &[batch as isize, heads as isize, n_q as isize, n_k as isize],
    )?;

    // Apply score_mod per (b, h). When score_mod is None this is a no-op
    // and we keep the on-device tensor. When score_mod is provided, we
    // extract per-(b,h) [n_q, n_k] views, call the user callback, and
    // reassemble via cat — all on the input's device, since narrow and
    // cat are device-aware.
    let scores_after_mod = if let Some(ref sm) = score_mod {
        // Extract per-(b,h) sub-tensors as [n_q, n_k] views, run user code,
        // and reassemble. The narrow + squeeze are zero-copy stride views;
        // cat is GPU-aware. The only potential CPU round-trip is in the
        // user's score_mod callback itself (out of our control).
        let mut per_bh: Vec<Tensor<T>> = Vec::with_capacity(bh);
        for b in 0..batch {
            for h in 0..heads {
                // [B, H, n_q, n_k] -> narrow b -> [1, H, n_q, n_k]
                //                  -> narrow h -> [1, 1, n_q, n_k]
                //                  -> squeeze 0, squeeze 0 -> [n_q, n_k]
                let bh_view = scores4
                    .narrow(0, b, 1)?
                    .narrow(1, h, 1)?
                    .squeeze_t(0)?
                    .squeeze_t(0)?;
                let modified = sm(&bh_view, b, h)?;
                if modified.shape() != [n_q, n_k] {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "flex_attention: score_mod returned shape {:?}, expected [{}, {}]",
                            modified.shape(),
                            n_q,
                            n_k
                        ),
                    });
                }
                // Lift back to [1, 1, n_q, n_k] so we can cat along dims 0/1.
                let lifted = modified.unsqueeze_t(0)?.unsqueeze_t(0)?;
                per_bh.push(lifted);
            }
        }
        // Concatenate first along the heads axis (dim 1) for each batch,
        // then along the batch axis (dim 0).
        let mut head_groups: Vec<Tensor<T>> = Vec::with_capacity(batch);
        for b in 0..batch {
            let group: Vec<Tensor<T>> = per_bh[b * heads..(b + 1) * heads].to_vec();
            let cat_h = crate::grad_fns::shape::cat(&group, 1)?;
            head_groups.push(cat_h);
        }
        crate::grad_fns::shape::cat(&head_groups, 0)?
    } else {
        scores4
    };

    // Softmax along the last (n_k) dimension. softmax is GPU-aware and
    // operates on the last dim regardless of rank.
    let weights4 = crate::grad_fns::activation::softmax(&scores_after_mod)?;

    // Reshape weights to [B*H, n_q, n_k] for the second bmm. Differentiable
    // reshape preserves the autograd chain.
    let weights3 =
        crate::grad_fns::shape::reshape(&weights4, &[bh as isize, n_q as isize, n_k as isize])?;

    // weights @ V: [B*H, n_q, n_k] @ [B*H, n_k, d_v] -> [B*H, n_q, d_v]
    let output3 = crate::grad_fns::linalg::bmm_differentiable(&weights3, &v3)?;

    // Reshape to the canonical [B, H, n_q, d_v] output shape.
    crate::grad_fns::shape::reshape(
        &output3,
        &[batch as isize, heads as isize, n_q as isize, d_v as isize],
    )
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
        // The new implementation composes flex_attention from differentiable
        // tensor ops (bmm, mul, softmax, reshape), so the output's grad_fn
        // is the LAST op in the chain (ReshapeBackward) rather than a custom
        // FlexAttentionBackward. What matters for autograd correctness is
        // that the chain is connected end-to-end — verified by checking that
        // a grad_fn exists on the output.
        let q = make_tensor_grad(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let k = make_tensor_grad(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let v = make_tensor_grad(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let output = flex_attention::<
            f32,
            fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
        >(&q, &k, &v, None)
        .unwrap();

        assert!(
            output.grad_fn().is_some(),
            "expected output to have a grad_fn so backward propagates to Q/K/V"
        );
    }

    #[test]
    fn test_flex_attention_numerical_value() {
        // Hand-computed reference: B=1, H=1, n_q=2, n_k=2, d=2, d_v=2.
        // Q = [[1,0],[0,1]],  K = [[1,0],[0,1]],  V = [[1,2],[3,4]]
        // scale = 1/sqrt(2) ≈ 0.7071
        // scores = Q @ K^T = [[1,0],[0,1]],  scaled = [[0.7071,0],[0,0.7071]]
        // softmax row 0 over [0.7071, 0]: e^0.7071 = 2.0281, e^0 = 1
        //   sum = 3.0281, weights = [0.6699, 0.3301]
        // softmax row 1 over [0, 0.7071]: weights = [0.3301, 0.6699]
        // out[0] = 0.6699*[1,2] + 0.3301*[3,4] = [0.6699+0.9903, 1.3398+1.3204]
        //        ≈ [1.6603, 2.6602]
        // out[1] = 0.3301*[1,2] + 0.6699*[3,4] = [0.3301+2.0098, 0.6602+2.6797]
        //        ≈ [2.3399, 3.3399]
        let q = make_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let k = make_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let v = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);
        let out = flex_attention::<
            f32,
            fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
        >(&q, &k, &v, None)
        .unwrap();
        let data = out.data().unwrap();
        let expected = [1.6603, 2.6602, 2.3399, 3.3399];
        for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-3,
                "out[{i}]: expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn test_flex_attention_score_mod_additive_bias() {
        // Verify that an additive score modification (e.g. ALiBi-style)
        // actually changes the output. We add +1.0 to all scores via the
        // score_mod callback, which biases the softmax distribution slightly.
        let q = make_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let k = make_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![1, 1, 2, 2]);
        let v = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        // No score_mod baseline.
        let baseline = flex_attention::<
            f32,
            fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
        >(&q, &k, &v, None)
        .unwrap();

        // With additive bias: adding a constant to all scores in a row is
        // softmax-invariant, so the output should equal the baseline. This
        // is a strong correctness check on the score_mod plumbing — if the
        // narrow/cat reassembly is wrong, this WON'T match.
        let with_const_bias = flex_attention(
            &q,
            &k,
            &v,
            Some(|s: &Tensor<f32>, _b: usize, _h: usize| {
                let one = crate::creation::scalar(1.0f32).unwrap();
                crate::grad_fns::arithmetic::add(s, &one)
            }),
        )
        .unwrap();

        let base_data = baseline.data().unwrap();
        let mod_data = with_const_bias.data().unwrap();
        for (i, (&b, &m)) in base_data.iter().zip(mod_data.iter()).enumerate() {
            assert!(
                (b - m).abs() < 1e-5,
                "softmax-invariant additive bias should not change output[{i}]: base={b}, mod={m}"
            );
        }
    }

    #[test]
    fn test_flex_attention_grad_propagates_to_qkv() {
        // Verify that calling backward on the output actually fills gradients
        // on Q, K, V — this is the bug the previous FlexAttentionBackward
        // pass-through couldn't catch (it returned grad_output as-is for all
        // three, which was shape-wrong and value-wrong).
        let q = make_tensor_grad(vec![1.0, 0.0, 0.5, 1.0], vec![1, 1, 2, 2]);
        let k = make_tensor_grad(vec![0.5, 1.0, 1.0, 0.0], vec![1, 1, 2, 2]);
        let v = make_tensor_grad(vec![1.0, 2.0, 3.0, 4.0], vec![1, 1, 2, 2]);

        let output = flex_attention::<
            f32,
            fn(&Tensor<f32>, usize, usize) -> FerrotorchResult<Tensor<f32>>,
        >(&q, &k, &v, None)
        .unwrap();

        // Sum to a scalar and run backward.
        let loss = crate::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        // All three inputs should now carry gradients of their original
        // shape. grad() returns Result<Option<Tensor>>, so unwrap both
        // layers.
        let gq = q
            .grad()
            .unwrap()
            .expect("query should have a gradient after backward");
        let gk = k
            .grad()
            .unwrap()
            .expect("key should have a gradient after backward");
        let gv = v
            .grad()
            .unwrap()
            .expect("value should have a gradient after backward");
        assert_eq!(gq.shape(), &[1, 1, 2, 2]);
        assert_eq!(gk.shape(), &[1, 1, 2, 2]);
        assert_eq!(gv.shape(), &[1, 1, 2, 2]);

        // The existence of non-zero gradients on V is the key correctness
        // signal vs. the old broken pass-through, which returned the output
        // gradient as-is for all three (shape-wrong garbage).
        let gv_data = gv.data().unwrap();
        assert!(
            gv_data.iter().any(|&x| x != 0.0),
            "expected non-zero gradient on V"
        );
    }
}
