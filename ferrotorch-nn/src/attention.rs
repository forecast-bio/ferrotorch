//! Multi-head attention layer.
//!
//! Implements scaled dot-product attention with multiple heads, following the
//! "Attention Is All You Need" paper (Vaswani et al., 2017). All operations
//! use differentiable primitives from `ferrotorch_core`, so autograd handles
//! the backward pass automatically.

use ferrotorch_core::grad_fns::activation::softmax;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::linalg::{bmm_differentiable, mm_differentiable};
use ferrotorch_core::grad_fns::shape::transpose_2d;
use ferrotorch_core::{permute_t, Float, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

use crate::init::{xavier_uniform, zeros};
use crate::module::Module;
use crate::parameter::Parameter;

/// Multi-head attention mechanism.
///
/// Computes scaled dot-product attention across `num_heads` parallel heads,
/// projecting queries, keys, and values through learned linear transformations.
///
/// # Shape contract
///
/// - Input: `[batch, seq_len, embed_dim]`
/// - Output: `[batch, seq_len, embed_dim]`
///
/// # Example
///
/// ```ignore
/// let mha = MultiheadAttention::<f32>::new(64, 8, true)?;
/// let input = ferrotorch_core::randn::<f32>(&[2, 10, 64])?;
/// let output = mha.forward(&input)?;
/// assert_eq!(output.shape(), &[2, 10, 64]);
/// ```
#[derive(Debug)]
pub struct MultiheadAttention<T: Float> {
    pub embed_dim: usize,
    pub num_heads: usize,
    pub head_dim: usize,

    /// Query projection weight: [embed_dim, embed_dim].
    pub q_proj: Parameter<T>,
    /// Key projection weight: [embed_dim, embed_dim].
    pub k_proj: Parameter<T>,
    /// Value projection weight: [embed_dim, embed_dim].
    pub v_proj: Parameter<T>,
    /// Output projection weight: [embed_dim, embed_dim].
    pub out_proj: Parameter<T>,

    /// Optional biases: [embed_dim].
    pub q_bias: Option<Parameter<T>>,
    pub k_bias: Option<Parameter<T>>,
    pub v_bias: Option<Parameter<T>>,
    pub out_bias: Option<Parameter<T>>,

    pub training: bool,
}

impl<T: Float> MultiheadAttention<T> {
    /// Create a new multi-head attention layer.
    ///
    /// # Arguments
    ///
    /// - `embed_dim` - Total embedding dimension (must be divisible by `num_heads`).
    /// - `num_heads` - Number of parallel attention heads.
    /// - `bias` - Whether to include additive bias in projections.
    ///
    /// # Errors
    ///
    /// Returns `FerrotorchError::InvalidArgument` if `embed_dim % num_heads != 0`.
    pub fn new(embed_dim: usize, num_heads: usize, bias: bool) -> FerrotorchResult<Self> {
        if embed_dim == 0 || num_heads == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "embed_dim and num_heads must be positive".into(),
            });
        }
        if embed_dim % num_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                ),
            });
        }

        let head_dim = embed_dim / num_heads;

        let mut q_proj = Parameter::zeros(&[embed_dim, embed_dim])?;
        let mut k_proj = Parameter::zeros(&[embed_dim, embed_dim])?;
        let mut v_proj = Parameter::zeros(&[embed_dim, embed_dim])?;
        let mut out_proj = Parameter::zeros(&[embed_dim, embed_dim])?;

        xavier_uniform(&mut q_proj)?;
        xavier_uniform(&mut k_proj)?;
        xavier_uniform(&mut v_proj)?;
        xavier_uniform(&mut out_proj)?;

        let (q_bias, k_bias, v_bias, out_bias) = if bias {
            let mut qb = Parameter::zeros(&[embed_dim])?;
            let mut kb = Parameter::zeros(&[embed_dim])?;
            let mut vb = Parameter::zeros(&[embed_dim])?;
            let mut ob = Parameter::zeros(&[embed_dim])?;
            zeros(&mut qb)?;
            zeros(&mut kb)?;
            zeros(&mut vb)?;
            zeros(&mut ob)?;
            (Some(qb), Some(kb), Some(vb), Some(ob))
        } else {
            (None, None, None, None)
        };

        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            q_bias,
            k_bias,
            v_bias,
            out_bias,
            training: true,
        })
    }

    /// Forward pass with separate query, key, and value tensors (cross-attention).
    ///
    /// # Arguments
    ///
    /// - `query` - `[batch, seq_q, embed_dim]`
    /// - `key` - `[batch, seq_k, embed_dim]`
    /// - `value` - `[batch, seq_k, embed_dim]`
    /// - `causal_mask` - If `true`, apply a causal (lower-triangular) mask so that
    ///   position `i` cannot attend to positions `j > i`. Only valid when
    ///   `seq_q == seq_k`.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, seq_q, embed_dim]`.
    pub fn forward_qkv(
        &self,
        query: &Tensor<T>,
        key: &Tensor<T>,
        value: &Tensor<T>,
        causal_mask: bool,
    ) -> FerrotorchResult<Tensor<T>> {
        // --- Validate input shapes ---
        if query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MultiheadAttention expects 3-D inputs [batch, seq, embed_dim], \
                     got query {:?}, key {:?}, value {:?}",
                    query.shape(),
                    key.shape(),
                    value.shape()
                ),
            });
        }

        let batch = query.shape()[0];
        let seq_q = query.shape()[1];
        let seq_k = key.shape()[0 + 1]; // key.shape()[1]

        if query.shape()[2] != self.embed_dim
            || key.shape()[2] != self.embed_dim
            || value.shape()[2] != self.embed_dim
        {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "embed_dim mismatch: expected {}, got query={}, key={}, value={}",
                    self.embed_dim,
                    query.shape()[2],
                    key.shape()[2],
                    value.shape()[2]
                ),
            });
        }

        if key.shape()[0] != batch || value.shape()[0] != batch {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "batch size mismatch: query batch={}, key batch={}, value batch={}",
                    batch,
                    key.shape()[0],
                    value.shape()[0]
                ),
            });
        }

        if key.shape()[1] != value.shape()[1] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "key and value seq_len must match: key={}, value={}",
                    key.shape()[1],
                    value.shape()[1]
                ),
            });
        }

        if causal_mask && seq_q != seq_k {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "causal mask requires seq_q == seq_k, got seq_q={seq_q}, seq_k={seq_k}"
                ),
            });
        }

        // ─── Fast path: self-attention with seq_len=1 ───────────────
        // When seq_len=1, attention scores are [1,1] -> softmax = 1.0,
        // so context = V identically. The whole MHA reduces to:
        //   output = V_proj(input) @ W_O^T + bias
        // We skip Q/K projections (saves 2 matmuls) and the per-head
        // attention loop (saves reshape, transpose, softmax per head).
        //
        // We also batch across the batch dimension: instead of looping
        // over batch elements doing [1,D]@[D,D], we squeeze to [B,D]
        // and do a single [B,D]@[D,D] matmul.
        if seq_q == 1 && seq_k == 1 && !causal_mask {
            use ferrotorch_core::grad_fns::linalg::linear_fused;

            // Squeeze [batch, 1, embed_dim] -> [batch, embed_dim]
            let v_2d = value.reshape_t(&[batch as isize, self.embed_dim as isize])?;

            // V_proj = v_2d @ W_V^T + v_bias -> [batch, embed_dim]
            let v_proj = linear_fused(
                &v_2d,
                self.v_proj.tensor(),
                self.v_bias.as_ref().map(|b| b.tensor()),
            )?;

            // O_proj = v_proj @ W_O^T + o_bias -> [batch, embed_dim]
            let output = linear_fused(
                &v_proj,
                self.out_proj.tensor(),
                self.out_bias.as_ref().map(|b| b.tensor()),
            )?;

            // Unsqueeze [batch, embed_dim] -> [batch, 1, embed_dim]
            return output.reshape_t(&[batch as isize, 1, self.embed_dim as isize]);
        }

        // ─── General path: batched multi-head attention ──────────────
        //
        // Instead of serial loops over batch and head dimensions, we flatten
        // [batch, seq, embed_dim] to [batch*seq, embed_dim], project once,
        // then reshape to [batch*num_heads, seq, head_dim] and use
        // bmm_differentiable for the two attention matmuls.

        // Edge case: seq_len=0 produces an empty tensor immediately.
        if seq_q == 0 {
            return Tensor::from_storage(
                TensorStorage::cpu(vec![]),
                vec![batch, 0, self.embed_dim],
                false,
            );
        }

        // Transpose projection weights once: W_Q.T, W_K.T, W_V.T, W_O.T
        let wq_t = transpose_2d(self.q_proj.tensor())?;
        let wk_t = transpose_2d(self.k_proj.tensor())?;
        let wv_t = transpose_2d(self.v_proj.tensor())?;
        let wo_t = transpose_2d(self.out_proj.tensor())?;

        let bh = batch * self.num_heads;
        let embed = self.embed_dim;
        let hd = self.head_dim;
        let nh = self.num_heads;

        // 1. Flatten [batch, seq, embed_dim] -> [batch*seq, embed_dim] and project.
        let q_flat = query.reshape_t(&[(batch * seq_q) as isize, embed as isize])?;
        let k_flat = key.reshape_t(&[(batch * seq_k) as isize, embed as isize])?;
        let v_flat = value.reshape_t(&[(batch * seq_k) as isize, embed as isize])?;

        let mut q_proj = mm_differentiable(&q_flat, &wq_t)?; // [B*Sq, E]
        let mut k_proj = mm_differentiable(&k_flat, &wk_t)?; // [B*Sk, E]
        let mut v_proj = mm_differentiable(&v_flat, &wv_t)?; // [B*Sk, E]

        // Add biases (broadcast over the rows dimension).
        if let Some(ref qb) = self.q_bias {
            let bias_exp = expand_bias_to_2d(qb.tensor(), batch * seq_q)?;
            q_proj = add(&q_proj, &bias_exp)?;
        }
        if let Some(ref kb) = self.k_bias {
            let bias_exp = expand_bias_to_2d(kb.tensor(), batch * seq_k)?;
            k_proj = add(&k_proj, &bias_exp)?;
        }
        if let Some(ref vb) = self.v_bias {
            let bias_exp = expand_bias_to_2d(vb.tensor(), batch * seq_k)?;
            v_proj = add(&v_proj, &bias_exp)?;
        }

        // 2. Reshape projected Q/K/V to [batch, seq, num_heads, head_dim],
        //    permute to [batch, num_heads, seq, head_dim],
        //    then flatten to [batch*num_heads, seq, head_dim].
        let q_4d = q_proj.reshape_t(&[
            batch as isize, seq_q as isize, nh as isize, hd as isize,
        ])?;
        let q_perm = permute_t(&q_4d, &[0, 2, 1, 3])?; // [B, H, Sq, D]
        let q_3d = q_perm.reshape_t(&[bh as isize, seq_q as isize, hd as isize])?;

        let k_4d = k_proj.reshape_t(&[
            batch as isize, seq_k as isize, nh as isize, hd as isize,
        ])?;
        let k_perm = permute_t(&k_4d, &[0, 2, 1, 3])?; // [B, H, Sk, D]
        let k_3d = k_perm.reshape_t(&[bh as isize, seq_k as isize, hd as isize])?;

        let v_4d = v_proj.reshape_t(&[
            batch as isize, seq_k as isize, nh as isize, hd as isize,
        ])?;
        let v_perm = permute_t(&v_4d, &[0, 2, 1, 3])?; // [B, H, Sk, D]
        let v_3d = v_perm.reshape_t(&[bh as isize, seq_k as isize, hd as isize])?;

        // 3. Attention scores: Q @ K^T -> [BH, Sq, Sk]
        let k_3d_t = permute_t(&k_3d, &[0, 2, 1])?; // [BH, D, Sk]
        let scores = bmm_differentiable(&q_3d, &k_3d_t)?;

        // 4. Scale by 1/sqrt(head_dim).
        let scale_val = T::from(1.0 / (hd as f64).sqrt()).unwrap();
        let scale = expand_scalar_to_3d::<T>(scale_val, bh, seq_q, seq_k)?;
        let scaled_scores = mul(&scores, &scale)?;

        // 5. Apply causal mask if requested.
        let masked_scores = if causal_mask {
            apply_causal_mask_batched(&scaled_scores, bh, seq_q)?
        } else {
            scaled_scores
        };

        // 6. Softmax along last dim (works on any shape — operates per row).
        let weights = softmax(&masked_scores)?;

        // 7. Attention output: attn @ V -> [BH, Sq, D]
        let context_3d = bmm_differentiable(&weights, &v_3d)?;

        // 8. Reshape back: [BH, Sq, D] -> [B, H, Sq, D] -> [B, Sq, H, D] -> [B, Sq, E]
        let ctx_4d = context_3d.reshape_t(&[
            batch as isize, nh as isize, seq_q as isize, hd as isize,
        ])?;
        let ctx_perm = permute_t(&ctx_4d, &[0, 2, 1, 3])?; // [B, Sq, H, D]
        let ctx_2d = ctx_perm.reshape_t(&[
            (batch * seq_q) as isize,
            embed as isize,
        ])?;

        // 9. Output projection: [B*Sq, E] @ W_O.T -> [B*Sq, E]
        let mut output = mm_differentiable(&ctx_2d, &wo_t)?;

        if let Some(ref ob) = self.out_bias {
            let bias_exp = expand_bias_to_2d(ob.tensor(), batch * seq_q)?;
            output = add(&output, &bias_exp)?;
        }

        // 10. Reshape to [B, Sq, E]
        output.reshape_t(&[batch as isize, seq_q as isize, embed as isize])
    }

    /// The embedding dimension.
    #[inline]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// The number of attention heads.
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// The dimension of each attention head.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Fast 2D self-attention for seq_len=1: [batch, embed_dim] -> [batch, embed_dim].
    /// Avoids unsqueeze/squeeze overhead. For seq_len=1, attention is identity on V,
    /// so this is just V_proj + O_proj (two fused linear ops).
    pub fn forward_2d(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        use ferrotorch_core::grad_fns::linalg::linear_fused;

        let v_proj = linear_fused(
            input,
            self.v_proj.tensor(),
            self.v_bias.as_ref().map(|b| b.tensor()),
        )?;
        linear_fused(
            &v_proj,
            self.out_proj.tensor(),
            self.out_bias.as_ref().map(|b| b.tensor()),
        )
    }
}

impl<T: Float> Module<T> for MultiheadAttention<T> {
    /// Self-attention forward: query = key = value = input.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_qkv(input, input, input, false)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.q_proj, &self.k_proj, &self.v_proj, &self.out_proj];
        if let Some(ref b) = self.q_bias {
            params.push(b);
        }
        if let Some(ref b) = self.k_bias {
            params.push(b);
        }
        if let Some(ref b) = self.v_bias {
            params.push(b);
        }
        if let Some(ref b) = self.out_bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params: Vec<&mut Parameter<T>> = vec![
            &mut self.q_proj,
            &mut self.k_proj,
            &mut self.v_proj,
            &mut self.out_proj,
        ];
        if let Some(ref mut b) = self.q_bias {
            params.push(b);
        }
        if let Some(ref mut b) = self.k_bias {
            params.push(b);
        }
        if let Some(ref mut b) = self.v_bias {
            params.push(b);
        }
        if let Some(ref mut b) = self.out_bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![
            ("q_proj.weight".to_string(), &self.q_proj),
            ("k_proj.weight".to_string(), &self.k_proj),
            ("v_proj.weight".to_string(), &self.v_proj),
            ("out_proj.weight".to_string(), &self.out_proj),
        ];
        if let Some(ref b) = self.q_bias {
            params.push(("q_proj.bias".to_string(), b));
        }
        if let Some(ref b) = self.k_bias {
            params.push(("k_proj.bias".to_string(), b));
        }
        if let Some(ref b) = self.v_bias {
            params.push(("v_proj.bias".to_string(), b));
        }
        if let Some(ref b) = self.out_bias {
            params.push(("out_proj.bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Extract a 2-D slice `[seq, dim]` from a 3-D tensor at batch index `b`.
///
/// This creates a new tensor (copies data) since we don't have strided views.
#[cfg(test)]
fn extract_batch_slice<T: Float>(
    tensor: &Tensor<T>,
    b: usize,
) -> FerrotorchResult<Tensor<T>> {
    let shape = tensor.shape();
    let dim1 = shape[1];
    let dim2 = shape[2];
    let slice_size = dim1 * dim2;
    let data = tensor.data()?;
    let start = b * slice_size;
    let end = start + slice_size;
    let slice_data = data[start..end].to_vec();
    Tensor::from_storage(
        TensorStorage::cpu(slice_data),
        vec![dim1, dim2],
        tensor.requires_grad(),
    )
}

/// Expand a 1-D bias `[dim]` to `[rows, dim]` by repeating it along rows.
fn expand_bias_to_2d<T: Float>(
    bias: &Tensor<T>,
    rows: usize,
) -> FerrotorchResult<Tensor<T>> {
    let bias_vec = bias.data()?;
    let dim = bias_vec.len();
    let mut expanded = Vec::with_capacity(rows * dim);
    for _ in 0..rows {
        expanded.extend_from_slice(&bias_vec);
    }
    Tensor::from_storage(
        TensorStorage::cpu(expanded),
        vec![rows, dim],
        bias.requires_grad(),
    )
}

/// Reshape `[seq, embed_dim]` to `[num_heads, seq, head_dim]`.
///
/// Conceptually: `[seq, num_heads * head_dim]` -> `[seq, num_heads, head_dim]`
/// -> transpose(0,1) -> `[num_heads, seq, head_dim]`.
///
/// Since we lack a general N-D transpose, we do this with explicit data shuffling.
#[cfg(test)]
fn reshape_to_heads<T: Float>(
    tensor: &Tensor<T>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    let data = tensor.data()?;
    // data layout: [seq_len, embed_dim] where embed_dim = num_heads * head_dim
    // Interpret as [seq_len, num_heads, head_dim], then transpose to [num_heads, seq_len, head_dim]
    let mut result = vec![<T as num_traits::Zero>::zero(); num_heads * seq_len * head_dim];

    for s in 0..seq_len {
        for h in 0..num_heads {
            for d in 0..head_dim {
                let src_idx = s * (num_heads * head_dim) + h * head_dim + d;
                let dst_idx = h * (seq_len * head_dim) + s * head_dim + d;
                result[dst_idx] = data[src_idx];
            }
        }
    }

    Tensor::from_storage(
        TensorStorage::cpu(result),
        vec![num_heads, seq_len, head_dim],
        tensor.requires_grad(),
    )
}

/// Expand a scalar-ish tensor `[1]` to `[rows, cols]` for elementwise multiply.
#[cfg(test)]
fn expand_scalar_to_2d<T: Float>(
    scalar: &Tensor<T>,
    rows: usize,
    cols: usize,
) -> FerrotorchResult<Tensor<T>> {
    let val = scalar.data()?[0];
    let data = vec![val; rows * cols];
    Tensor::from_storage(
        TensorStorage::cpu(data),
        vec![rows, cols],
        false,
    )
}

/// Expand a scalar value to a 3-D tensor `[batch, rows, cols]` for elementwise multiply.
fn expand_scalar_to_3d<T: Float>(
    val: T,
    batch: usize,
    rows: usize,
    cols: usize,
) -> FerrotorchResult<Tensor<T>> {
    let data = vec![val; batch * rows * cols];
    Tensor::from_storage(
        TensorStorage::cpu(data),
        vec![batch, rows, cols],
        false,
    )
}

/// Apply a causal (lower-triangular) mask to 2-D attention scores `[seq, seq]`.
///
/// Sets positions where `col > row` to a very large negative value (-1e9)
/// so that softmax drives them to zero.
#[cfg(test)]
fn apply_causal_mask<T: Float>(
    scores: &Tensor<T>,
    seq_len: usize,
) -> FerrotorchResult<Tensor<T>> {
    let neg_inf = T::from(-1e9).unwrap();
    let mut masked = scores.data()?.to_vec();

    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            masked[i * seq_len + j] = neg_inf;
        }
    }

    Tensor::from_storage(
        TensorStorage::cpu(masked),
        scores.shape().to_vec(),
        scores.requires_grad(),
    )
}

/// Apply a causal mask to batched 3-D attention scores `[batch, seq, seq]`.
///
/// Same semantics as `apply_causal_mask` but applied identically to every
/// batch slice, avoiding a per-batch loop.
fn apply_causal_mask_batched<T: Float>(
    scores: &Tensor<T>,
    batch: usize,
    seq_len: usize,
) -> FerrotorchResult<Tensor<T>> {
    let neg_inf = T::from(-1e9).unwrap();
    let mut masked = scores.data()?.to_vec();
    let sq = seq_len * seq_len;

    for b in 0..batch {
        let base = b * sq;
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                masked[base + i * seq_len + j] = neg_inf;
            }
        }
    }

    Tensor::from_storage(
        TensorStorage::cpu(masked),
        scores.shape().to_vec(),
        scores.requires_grad(),
    )
}

/// Concatenate per-head outputs `[seq, head_dim]` back to `[seq, embed_dim]`.
///
/// Inverse of `reshape_to_heads`: gathers head outputs into
/// `[seq, num_heads * head_dim]` = `[seq, embed_dim]`.
#[cfg(test)]
fn concat_heads<T: Float>(
    heads: &[Tensor<T>],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    let embed_dim = num_heads * head_dim;
    let mut result = vec![<T as num_traits::Zero>::zero(); seq_len * embed_dim];

    for (h, head) in heads.iter().enumerate() {
        let head_data = head.data()?;
        for s in 0..seq_len {
            for d in 0..head_dim {
                let src_idx = s * head_dim + d;
                let dst_idx = s * embed_dim + h * head_dim + d;
                result[dst_idx] = head_data[src_idx];
            }
        }
    }

    Tensor::from_storage(
        TensorStorage::cpu(result),
        vec![seq_len, embed_dim],
        false,
    )
}

// ---------------------------------------------------------------------------
// Serial reference implementation (used by tests for numerical comparison)
// ---------------------------------------------------------------------------

/// Serial per-batch, per-head attention — the original O(B*H) loop approach.
/// Kept for numerical equivalence tests against the batched implementation.
#[cfg(test)]
fn forward_qkv_serial<T: Float>(
    mha: &MultiheadAttention<T>,
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    causal_mask: bool,
) -> FerrotorchResult<Tensor<T>> {
    let batch = query.shape()[0];
    let seq_q = query.shape()[1];
    let seq_k = key.shape()[1];

    if seq_q == 0 {
        return Tensor::from_storage(
            TensorStorage::cpu(vec![]),
            vec![batch, 0, mha.embed_dim],
            false,
        );
    }

    let wq_t = transpose_2d(mha.q_proj.tensor())?;
    let wk_t = transpose_2d(mha.k_proj.tensor())?;
    let wv_t = transpose_2d(mha.v_proj.tensor())?;
    let wo_t = transpose_2d(mha.out_proj.tensor())?;

    let scale_val = T::from(1.0 / (mha.head_dim as f64).sqrt()).unwrap();
    let scale = Tensor::from_storage(
        TensorStorage::cpu(vec![scale_val]),
        vec![1],
        false,
    )?;

    let total_elements = batch * seq_q * mha.embed_dim;
    let mut result_data: Vec<T> = Vec::with_capacity(total_elements);

    for b in 0..batch {
        let q_slice = extract_batch_slice(query, b)?;
        let k_slice = extract_batch_slice(key, b)?;
        let v_slice = extract_batch_slice(value, b)?;

        let mut q_proj = mm_differentiable(&q_slice, &wq_t)?;
        let mut k_proj = mm_differentiable(&k_slice, &wk_t)?;
        let mut v_proj = mm_differentiable(&v_slice, &wv_t)?;

        if let Some(ref qb) = mha.q_bias {
            let bias_expanded = expand_bias_to_2d(qb.tensor(), seq_q)?;
            q_proj = add(&q_proj, &bias_expanded)?;
        }
        if let Some(ref kb) = mha.k_bias {
            let bias_expanded = expand_bias_to_2d(kb.tensor(), seq_k)?;
            k_proj = add(&k_proj, &bias_expanded)?;
        }
        if let Some(ref vb) = mha.v_bias {
            let bias_expanded = expand_bias_to_2d(vb.tensor(), seq_k)?;
            v_proj = add(&v_proj, &bias_expanded)?;
        }

        let q_heads = reshape_to_heads(&q_proj, mha.num_heads, seq_q, mha.head_dim)?;
        let k_heads = reshape_to_heads(&k_proj, mha.num_heads, seq_k, mha.head_dim)?;
        let v_heads = reshape_to_heads(&v_proj, mha.num_heads, seq_k, mha.head_dim)?;

        let mut head_outputs: Vec<Tensor<T>> = Vec::with_capacity(mha.num_heads);

        for h in 0..mha.num_heads {
            let q_h = extract_batch_slice(&q_heads, h)?;
            let k_h = extract_batch_slice(&k_heads, h)?;
            let v_h = extract_batch_slice(&v_heads, h)?;

            let k_h_t = transpose_2d(&k_h)?;
            let scores = mm_differentiable(&q_h, &k_h_t)?;

            let scale_expanded = expand_scalar_to_2d(&scale, seq_q, seq_k)?;
            let scaled_scores = mul(&scores, &scale_expanded)?;

            let masked_scores = if causal_mask {
                apply_causal_mask(&scaled_scores, seq_q)?
            } else {
                scaled_scores
            };

            let weights = softmax(&masked_scores)?;
            let context_h = mm_differentiable(&weights, &v_h)?;
            head_outputs.push(context_h);
        }

        let context = concat_heads(&head_outputs, seq_q, mha.num_heads, mha.head_dim)?;

        let mut output = mm_differentiable(&context, &wo_t)?;
        if let Some(ref ob) = mha.out_bias {
            let bias_expanded = expand_bias_to_2d(ob.tensor(), seq_q)?;
            output = add(&output, &bias_expanded)?;
        }

        let out_data = output.data()?;
        result_data.extend_from_slice(&out_data);
    }

    Tensor::from_storage(
        TensorStorage::cpu(result_data),
        vec![batch, seq_q, mha.embed_dim],
        false,
    )
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_valid() {
        let mha = MultiheadAttention::<f32>::new(64, 8, true);
        assert!(mha.is_ok());
        let mha = mha.unwrap();
        assert_eq!(mha.embed_dim(), 64);
        assert_eq!(mha.num_heads(), 8);
        assert_eq!(mha.head_dim(), 8);
    }

    #[test]
    fn test_new_invalid_divisibility() {
        let result = MultiheadAttention::<f32>::new(65, 8, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_zero_dims() {
        assert!(MultiheadAttention::<f32>::new(0, 4, false).is_err());
        assert!(MultiheadAttention::<f32>::new(64, 0, false).is_err());
    }

    #[test]
    fn test_parameter_count_with_bias() {
        let mha = MultiheadAttention::<f32>::new(16, 4, true).unwrap();
        let params = mha.parameters();
        // 4 weight matrices: 4 * 16 * 16 = 1024
        // 4 bias vectors: 4 * 16 = 64
        // Total params: 1088
        let total: usize = params.iter().map(|p| p.numel()).sum();
        let embed_dim = 16usize;
        let expected = 4 * embed_dim * embed_dim + 4 * embed_dim;
        assert_eq!(total, expected);
        assert_eq!(params.len(), 8); // 4 weights + 4 biases
    }

    #[test]
    fn test_parameter_count_without_bias() {
        let mha = MultiheadAttention::<f32>::new(16, 4, false).unwrap();
        let params = mha.parameters();
        let total: usize = params.iter().map(|p| p.numel()).sum();
        let embed_dim = 16usize;
        let expected = 4 * embed_dim * embed_dim;
        assert_eq!(total, expected);
        assert_eq!(params.len(), 4); // 4 weights only
    }

    #[test]
    fn test_named_parameters() {
        let mha = MultiheadAttention::<f32>::new(8, 2, true).unwrap();
        let named = mha.named_parameters();
        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"q_proj.weight"));
        assert!(names.contains(&"k_proj.weight"));
        assert!(names.contains(&"v_proj.weight"));
        assert!(names.contains(&"out_proj.weight"));
        assert!(names.contains(&"q_proj.bias"));
        assert!(names.contains(&"k_proj.bias"));
        assert!(names.contains(&"v_proj.bias"));
        assert!(names.contains(&"out_proj.bias"));
    }

    #[test]
    fn test_output_shape() {
        let mha = MultiheadAttention::<f32>::new(16, 4, true).unwrap();
        // Input: [batch=2, seq_len=5, embed_dim=16]
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 16]).unwrap();
        let output = mha.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_output_shape_no_bias() {
        let mha = MultiheadAttention::<f32>::new(8, 2, false).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 3, 8]).unwrap();
        let output = mha.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_self_attention_basic_forward() {
        // Use a small model to verify forward pass produces finite values.
        let mha = MultiheadAttention::<f64>::new(4, 2, true).unwrap();
        let input = ferrotorch_core::ones::<f64>(&[1, 2, 4]).unwrap();
        let output = mha.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 2, 4]);
        let data = output.data().unwrap();
        // All values should be finite (not NaN, not Inf).
        for &v in data {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    #[test]
    fn test_cross_attention_shape() {
        let mha = MultiheadAttention::<f32>::new(8, 2, true).unwrap();
        // query: [1, 3, 8], key/value: [1, 5, 8]
        let query = ferrotorch_core::zeros::<f32>(&[1, 3, 8]).unwrap();
        let kv = ferrotorch_core::zeros::<f32>(&[1, 5, 8]).unwrap();
        let output = mha.forward_qkv(&query, &kv, &kv, false).unwrap();
        assert_eq!(output.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_causal_mask_different_seq_lens_error() {
        let mha = MultiheadAttention::<f32>::new(8, 2, false).unwrap();
        let query = ferrotorch_core::zeros::<f32>(&[1, 3, 8]).unwrap();
        let kv = ferrotorch_core::zeros::<f32>(&[1, 5, 8]).unwrap();
        // Causal mask requires seq_q == seq_k.
        let result = mha.forward_qkv(&query, &kv, &kv, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_train_eval_toggle() {
        let mut mha = MultiheadAttention::<f32>::new(8, 2, true).unwrap();
        assert!(mha.is_training());
        mha.eval();
        assert!(!mha.is_training());
        mha.train();
        assert!(mha.is_training());
    }

    #[test]
    fn test_wrong_embed_dim_input() {
        let mha = MultiheadAttention::<f32>::new(8, 2, true).unwrap();
        // Wrong embed_dim: 4 instead of 8.
        let input = ferrotorch_core::zeros::<f32>(&[1, 3, 4]).unwrap();
        let result = mha.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_2d_input_rejected() {
        let mha = MultiheadAttention::<f32>::new(8, 2, true).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[3, 8]).unwrap();
        let result = mha.forward(&input);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MultiheadAttention<f32>>();
        assert_send_sync::<MultiheadAttention<f64>>();
    }

    // -----------------------------------------------------------------------
    // Batched vs serial numerical equivalence tests
    // -----------------------------------------------------------------------

    /// Assert two tensors are elementwise equal within the given tolerance.
    fn assert_tensors_close(
        a: &Tensor<f64>,
        b: &Tensor<f64>,
        tol: f64,
        label: &str,
    ) {
        assert_eq!(a.shape(), b.shape(), "{label}: shape mismatch");
        let a_data = a.data().unwrap();
        let b_data = b.data().unwrap();
        for (i, (&va, &vb)) in a_data.iter().zip(b_data.iter()).enumerate() {
            let diff = (va - vb).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: batched={va}, serial={vb}, diff={diff} > tol={tol}"
            );
        }
    }

    fn assert_tensors_close_f32(
        a: &Tensor<f32>,
        b: &Tensor<f32>,
        tol: f32,
        label: &str,
    ) {
        assert_eq!(a.shape(), b.shape(), "{label}: shape mismatch");
        let a_data = a.data().unwrap();
        let b_data = b.data().unwrap();
        for (i, (&va, &vb)) in a_data.iter().zip(b_data.iter()).enumerate() {
            let diff = (va - vb).abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: batched={va}, serial={vb}, diff={diff} > tol={tol}"
            );
        }
    }

    #[test]
    fn test_batched_vs_serial_self_attn_no_bias() {
        let mha = MultiheadAttention::<f64>::new(8, 2, false).unwrap();
        let input = ferrotorch_core::randn::<f64>(&[2, 4, 8]).unwrap();

        let batched = mha.forward_qkv(&input, &input, &input, false).unwrap();
        let serial = forward_qkv_serial(&mha, &input, &input, &input, false).unwrap();

        assert_tensors_close(&batched, &serial, 1e-10, "self_attn_no_bias");
    }

    #[test]
    fn test_batched_vs_serial_self_attn_with_bias() {
        let mha = MultiheadAttention::<f64>::new(8, 2, true).unwrap();
        let input = ferrotorch_core::randn::<f64>(&[2, 4, 8]).unwrap();

        let batched = mha.forward_qkv(&input, &input, &input, false).unwrap();
        let serial = forward_qkv_serial(&mha, &input, &input, &input, false).unwrap();

        assert_tensors_close(&batched, &serial, 1e-10, "self_attn_with_bias");
    }

    #[test]
    fn test_batched_vs_serial_cross_attn() {
        let mha = MultiheadAttention::<f64>::new(8, 2, true).unwrap();
        let query = ferrotorch_core::randn::<f64>(&[2, 3, 8]).unwrap();
        let kv = ferrotorch_core::randn::<f64>(&[2, 5, 8]).unwrap();

        let batched = mha.forward_qkv(&query, &kv, &kv, false).unwrap();
        let serial = forward_qkv_serial(&mha, &query, &kv, &kv, false).unwrap();

        assert_tensors_close(&batched, &serial, 1e-10, "cross_attn");
    }

    #[test]
    fn test_batched_vs_serial_causal_mask() {
        let mha = MultiheadAttention::<f64>::new(8, 2, true).unwrap();
        let input = ferrotorch_core::randn::<f64>(&[2, 4, 8]).unwrap();

        let batched = mha.forward_qkv(&input, &input, &input, true).unwrap();
        let serial = forward_qkv_serial(&mha, &input, &input, &input, true).unwrap();

        assert_tensors_close(&batched, &serial, 1e-10, "causal_mask");
    }

    #[test]
    fn test_batched_vs_serial_batch_1_head_1() {
        let mha = MultiheadAttention::<f64>::new(4, 1, true).unwrap();
        let input = ferrotorch_core::randn::<f64>(&[1, 3, 4]).unwrap();

        let batched = mha.forward_qkv(&input, &input, &input, false).unwrap();
        let serial = forward_qkv_serial(&mha, &input, &input, &input, false).unwrap();

        assert_tensors_close(&batched, &serial, 1e-10, "batch1_head1");
    }

    #[test]
    fn test_batched_vs_serial_batch_1_head_1_causal() {
        let mha = MultiheadAttention::<f64>::new(4, 1, true).unwrap();
        let input = ferrotorch_core::randn::<f64>(&[1, 5, 4]).unwrap();

        let batched = mha.forward_qkv(&input, &input, &input, true).unwrap();
        let serial = forward_qkv_serial(&mha, &input, &input, &input, true).unwrap();

        assert_tensors_close(&batched, &serial, 1e-10, "batch1_head1_causal");
    }

    #[test]
    fn test_batched_vs_serial_larger_model() {
        let mha = MultiheadAttention::<f64>::new(16, 4, true).unwrap();
        let input = ferrotorch_core::randn::<f64>(&[3, 6, 16]).unwrap();

        let batched = mha.forward_qkv(&input, &input, &input, false).unwrap();
        let serial = forward_qkv_serial(&mha, &input, &input, &input, false).unwrap();

        assert_tensors_close(&batched, &serial, 1e-9, "larger_model");
    }

    #[test]
    fn test_batched_vs_serial_f32() {
        let mha = MultiheadAttention::<f32>::new(8, 2, true).unwrap();
        let input = ferrotorch_core::randn::<f32>(&[2, 4, 8]).unwrap();

        let batched = mha.forward_qkv(&input, &input, &input, false).unwrap();
        let serial = forward_qkv_serial(&mha, &input, &input, &input, false).unwrap();

        assert_tensors_close_f32(&batched, &serial, 1e-5, "f32_self_attn");
    }

    #[test]
    fn test_batched_seq_len_0() {
        let mha = MultiheadAttention::<f32>::new(8, 2, true).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 0, 8]).unwrap();
        let output = mha.forward_qkv(&input, &input, &input, false).unwrap();
        assert_eq!(output.shape(), &[2, 0, 8]);
        assert_eq!(output.numel(), 0);
    }

    #[test]
    fn test_batched_vs_serial_no_bias_causal() {
        let mha = MultiheadAttention::<f64>::new(8, 4, false).unwrap();
        let input = ferrotorch_core::randn::<f64>(&[2, 4, 8]).unwrap();

        let batched = mha.forward_qkv(&input, &input, &input, true).unwrap();
        let serial = forward_qkv_serial(&mha, &input, &input, &input, true).unwrap();

        assert_tensors_close(&batched, &serial, 1e-10, "no_bias_causal");
    }

    #[test]
    fn test_batched_vs_serial_many_heads() {
        // 8 heads with head_dim=1 — edge case for very small per-head dimension.
        let mha = MultiheadAttention::<f64>::new(8, 8, true).unwrap();
        let input = ferrotorch_core::randn::<f64>(&[2, 3, 8]).unwrap();

        let batched = mha.forward_qkv(&input, &input, &input, false).unwrap();
        let serial = forward_qkv_serial(&mha, &input, &input, &input, false).unwrap();

        assert_tensors_close(&batched, &serial, 1e-10, "many_heads");
    }

    #[test]
    fn test_batched_output_finite() {
        let mha = MultiheadAttention::<f64>::new(8, 2, true).unwrap();
        let input = ferrotorch_core::randn::<f64>(&[2, 4, 8]).unwrap();
        let output = mha.forward(&input).unwrap();

        for &v in output.data().unwrap().iter() {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }
}
