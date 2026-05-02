//! Multi-head attention layer.
//!
//! Implements scaled dot-product attention with multiple heads, following the
//! "Attention Is All You Need" paper (Vaswani et al., 2017). All operations
//! use differentiable primitives from `ferrotorch_core`, so autograd handles
//! the backward pass automatically.
//!
//! # Grouped-Query Attention (GQA)
//!
//! When `num_kv_heads < num_heads`, keys and values share a smaller head
//! count than queries (Llama 3 uses 32 Q heads : 8 KV heads). The K and V
//! projections are sized `[num_kv_heads * head_dim, embed_dim]` and each
//! KV head serves `group_size = num_heads / num_kv_heads` consecutive
//! Q heads via `repeat_kv` before the attention matmul. Construct a GQA
//! attention with [`MultiheadAttention::with_gqa`]; the default [`MultiheadAttention::new`]
//! preserves classical MHA (`num_kv_heads = num_heads`).

use ferrotorch_core::grad_fns::activation::softmax;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::linalg::{bmm_differentiable, mm_differentiable};
use ferrotorch_core::grad_fns::shape::{expand, transpose_2d};
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

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
    /// Number of key/value heads. Equals `num_heads` for classical MHA;
    /// less than `num_heads` (and a divisor thereof) for grouped-query
    /// attention.
    pub num_kv_heads: usize,
    pub head_dim: usize,

    /// Query projection weight: `[embed_dim, embed_dim]`.
    pub q_proj: Parameter<T>,
    /// Key projection weight: `[num_kv_heads * head_dim, embed_dim]`.
    pub k_proj: Parameter<T>,
    /// Value projection weight: `[num_kv_heads * head_dim, embed_dim]`.
    pub v_proj: Parameter<T>,
    /// Output projection weight: `[embed_dim, embed_dim]`.
    pub out_proj: Parameter<T>,

    /// Optional Q/O biases: `[embed_dim]`. K/V biases: `[num_kv_heads * head_dim]`.
    pub q_bias: Option<Parameter<T>>,
    pub k_bias: Option<Parameter<T>>,
    pub v_bias: Option<Parameter<T>>,
    pub out_bias: Option<Parameter<T>>,

    pub training: bool,
}

impl<T: Float> MultiheadAttention<T> {
    /// Create a new classical multi-head attention layer
    /// (`num_kv_heads == num_heads`).
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
        Self::with_gqa(embed_dim, num_heads, num_heads, bias)
    }

    /// Create a grouped-query (or classical) attention layer.
    ///
    /// When `num_kv_heads < num_heads`, each KV head serves
    /// `group_size = num_heads / num_kv_heads` consecutive query heads.
    /// `num_kv_heads == num_heads` reproduces classical MHA.
    ///
    /// # Arguments
    ///
    /// - `embed_dim` - Total embedding dimension (must be divisible by `num_heads`).
    /// - `num_heads` - Number of parallel query heads.
    /// - `num_kv_heads` - Number of key/value heads. Must divide `num_heads` evenly.
    /// - `bias` - Whether to include additive bias in projections.
    ///
    /// # Errors
    ///
    /// - `embed_dim == 0` or either head count is zero.
    /// - `embed_dim % num_heads != 0`.
    /// - `num_heads % num_kv_heads != 0`.
    pub fn with_gqa(
        embed_dim: usize,
        num_heads: usize,
        num_kv_heads: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if embed_dim == 0 || num_heads == 0 || num_kv_heads == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "embed_dim, num_heads, num_kv_heads must be positive".into(),
            });
        }
        if embed_dim % num_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
                ),
            });
        }
        if num_heads % num_kv_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
                ),
            });
        }

        let head_dim = embed_dim / num_heads;
        let kv_dim = num_kv_heads * head_dim;

        let mut q_proj = Parameter::zeros(&[embed_dim, embed_dim])?;
        let mut k_proj = Parameter::zeros(&[kv_dim, embed_dim])?;
        let mut v_proj = Parameter::zeros(&[kv_dim, embed_dim])?;
        let mut out_proj = Parameter::zeros(&[embed_dim, embed_dim])?;

        xavier_uniform(&mut q_proj)?;
        xavier_uniform(&mut k_proj)?;
        xavier_uniform(&mut v_proj)?;
        xavier_uniform(&mut out_proj)?;

        let (q_bias, k_bias, v_bias, out_bias) = if bias {
            let mut qb = Parameter::zeros(&[embed_dim])?;
            let mut kb = Parameter::zeros(&[kv_dim])?;
            let mut vb = Parameter::zeros(&[kv_dim])?;
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
            num_kv_heads,
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
        let seq_k = key.shape()[1];

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
        // Gated on `num_kv_heads == num_heads` — the GQA case needs a
        // repeat_kv step between V_proj and O_proj (V_proj outputs kv_dim,
        // O_proj expects embed_dim), so we fall through to the general path.
        if seq_q == 1 && seq_k == 1 && !causal_mask && self.num_kv_heads == self.num_heads {
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

        // ─── General path: batched multi-head attention ────────────
        //
        // Fully differentiable, GPU-compatible. Uses reshape/permute
        // (zero-copy metadata ops) instead of data shuffling, and
        // bmm_differentiable for the full batch at once.

        let nh = self.num_heads;
        let nkv = self.num_kv_heads;
        let hd = self.head_dim;
        let group_size = nh / nkv;

        // 1. Project Q/K/V. Flatten to 2-D for the matmul.
        let wq_t = transpose_2d(self.q_proj.tensor())?;
        let wk_t = transpose_2d(self.k_proj.tensor())?;
        let wv_t = transpose_2d(self.v_proj.tensor())?;
        let wo_t = transpose_2d(self.out_proj.tensor())?;

        let flat_q = query.reshape_t(&[-1, self.embed_dim as isize])?;
        let flat_k = key.reshape_t(&[-1, self.embed_dim as isize])?;
        let flat_v = value.reshape_t(&[-1, self.embed_dim as isize])?;

        let mut q_proj = mm_differentiable(&flat_q, &wq_t)?;
        let mut k_proj = mm_differentiable(&flat_k, &wk_t)?;
        let mut v_proj = mm_differentiable(&flat_v, &wv_t)?;

        if let Some(ref qb) = self.q_bias {
            let b = expand_bias_to_2d(qb.tensor(), batch * seq_q)?;
            q_proj = add(&q_proj, &b)?;
        }
        if let Some(ref kb) = self.k_bias {
            let b = expand_bias_to_2d(kb.tensor(), batch * seq_k)?;
            k_proj = add(&k_proj, &b)?;
        }
        if let Some(ref vb) = self.v_bias {
            let b = expand_bias_to_2d(vb.tensor(), batch * seq_k)?;
            v_proj = add(&v_proj, &b)?;
        }

        // 2. Reshape to per-head layout via permute (zero-copy + contiguous).
        //    Q: [B*Sq, D] → [B, Sq, H, Hd] → [B, H, Sq, Hd] → [B*H, Sq, Hd]
        //    K/V: same but with Hkv instead of H.
        let q = q_proj
            .reshape_t(&[batch as isize, seq_q as isize, nh as isize, hd as isize])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape_t(&[(batch * nh) as isize, seq_q as isize, hd as isize])?;

        let mut k = k_proj
            .reshape_t(&[batch as isize, seq_k as isize, nkv as isize, hd as isize])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;
        let mut v = v_proj
            .reshape_t(&[batch as isize, seq_k as isize, nkv as isize, hd as isize])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?;

        // 3. GQA repeat: expand each KV head to serve `group_size` Q heads.
        if group_size > 1 {
            // [B, Hkv, S, Hd] → [B, Hkv, 1, S, Hd] → expand [B, Hkv, G, S, Hd]
            // → reshape [B, H, S, Hd]
            k = k.reshape_t(&[batch as isize, nkv as isize, 1, seq_k as isize, hd as isize])?;
            k = expand(&k, &[batch, nkv, group_size, seq_k, hd])?;
            k = k.reshape_t(&[batch as isize, nh as isize, seq_k as isize, hd as isize])?;

            v = v.reshape_t(&[batch as isize, nkv as isize, 1, seq_k as isize, hd as isize])?;
            v = expand(&v, &[batch, nkv, group_size, seq_k, hd])?;
            v = v.reshape_t(&[batch as isize, nh as isize, seq_k as isize, hd as isize])?;
        }

        let k = k.reshape_t(&[(batch * nh) as isize, seq_k as isize, hd as isize])?;
        let v = v.reshape_t(&[(batch * nh) as isize, seq_k as isize, hd as isize])?;

        // 4. Scaled dot-product attention.
        //    scores = Q @ K^T → [B*H, Sq, Sk]
        let k_t = k.permute(&[0, 2, 1])?.contiguous()?;
        let scores = bmm_differentiable(&q, &k_t)?;

        let scale_val = T::from(1.0 / (hd as f64).sqrt()).unwrap();
        let scale_tensor = Tensor::from_storage(
            TensorStorage::on_device(vec![scale_val], scores.device())?,
            vec![1],
            false,
        )?;
        let scaled = mul(&scores, &scale_tensor)?;

        // 5. Causal mask: additive -1e9 for future positions.
        let masked = if causal_mask {
            let neg_inf = T::from(-1e9).unwrap();
            let zero = <T as num_traits::Zero>::zero();
            let mut mask_data = vec![zero; seq_q * seq_k];
            for i in 0..seq_q {
                for j in (i + 1)..seq_k {
                    mask_data[i * seq_k + j] = neg_inf;
                }
            }
            let mask =
                Tensor::from_storage(TensorStorage::cpu(mask_data), vec![1, seq_q, seq_k], false)?;
            let mask = if scaled.is_cuda() {
                mask.to(scaled.device())?
            } else {
                mask
            };
            add(&scaled, &mask)?
        } else {
            scaled
        };

        // 6. Softmax + context.
        let weights = softmax(&masked)?;
        let context = bmm_differentiable(&weights, &v)?;

        // 7. Reshape back: [B*H, Sq, Hd] → [B, H, Sq, Hd] → [B, Sq, H, Hd] → [B*Sq, D]
        let context = context
            .reshape_t(&[batch as isize, nh as isize, seq_q as isize, hd as isize])?
            .permute(&[0, 2, 1, 3])?
            .contiguous()?
            .reshape_t(&[(batch * seq_q) as isize, self.embed_dim as isize])?;

        // 8. Output projection.
        let mut output = mm_differentiable(&context, &wo_t)?;
        if let Some(ref ob) = self.out_bias {
            let b = expand_bias_to_2d(ob.tensor(), batch * seq_q)?;
            output = add(&output, &b)?;
        }

        output.reshape_t(&[batch as isize, seq_q as isize, self.embed_dim as isize])
    }

    /// The embedding dimension.
    #[inline]
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }

    /// The number of query attention heads.
    #[inline]
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// The number of key/value heads (equal to `num_heads` for classical MHA,
    /// less for grouped-query attention).
    #[inline]
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// The dimension of each attention head.
    #[inline]
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Whether this layer is configured for grouped-query attention
    /// (`num_kv_heads < num_heads`).
    #[inline]
    pub fn is_gqa(&self) -> bool {
        self.num_kv_heads != self.num_heads
    }

    /// Fast 2D self-attention for seq_len=1: [batch, embed_dim] -> [batch, embed_dim].
    /// Avoids unsqueeze/squeeze overhead. For seq_len=1, attention is identity on V,
    /// so this is just V_proj + O_proj (two fused linear ops).
    ///
    /// # Errors
    ///
    /// Returns `InvalidArgument` when called on a GQA layer
    /// (`num_kv_heads != num_heads`): the V/O shapes no longer match and a
    /// `repeat_kv` step is required. Use [`forward_qkv`] for GQA.
    pub fn forward_2d(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        use ferrotorch_core::grad_fns::linalg::linear_fused;

        if self.is_gqa() {
            return Err(FerrotorchError::InvalidArgument {
                message:
                    "forward_2d is MHA-only; use forward_qkv for GQA (num_kv_heads != num_heads)"
                        .into(),
            });
        }

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

/// Expand a 1-D bias `[dim]` to `[rows, dim]` by repeating it along rows.
///
/// Uses the differentiable `expand` primitive so that gradients flow back
/// to the original bias parameter through `ExpandBackward`.
fn expand_bias_to_2d<T: Float>(bias: &Tensor<T>, rows: usize) -> FerrotorchResult<Tensor<T>> {
    let dim = bias.shape()[0];
    // Reshape [dim] -> [1, dim], then expand to [rows, dim].
    let bias_2d = bias.reshape_t(&[1, dim as isize])?;
    expand(&bias_2d, &[rows, dim])
}

/// Reshape `[seq, embed_dim]` to `[num_heads, seq, head_dim]`.
///
/// Conceptually: `[seq, num_heads * head_dim]` -> `[seq, num_heads, head_dim]`
/// -> transpose(0,1) -> `[num_heads, seq, head_dim]`.
///
/// Since we lack a general N-D transpose, we do this with explicit data shuffling.
pub fn reshape_to_heads<T: Float>(
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

/// Transpose [num_heads, seq, head_dim] → [seq, num_heads * head_dim] = [seq, embed_dim].
///
/// Inverse of `reshape_to_heads` for the batched attention output.
pub fn transpose_heads_to_2d<T: Float>(
    tensor: &Tensor<T>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
) -> FerrotorchResult<Tensor<T>> {
    let embed_dim = num_heads * head_dim;
    let data = tensor.data_vec()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); seq_len * embed_dim];

    for h in 0..num_heads {
        for s in 0..seq_len {
            for d in 0..head_dim {
                let src_idx = h * (seq_len * head_dim) + s * head_dim + d;
                let dst_idx = s * embed_dim + h * head_dim + d;
                result[dst_idx] = data[src_idx];
            }
        }
    }

    let device = tensor.device();
    Tensor::from_storage(
        TensorStorage::on_device(result, device)?,
        vec![seq_len, embed_dim],
        false,
    )
}

/// Repeat each KV head `group_size` times along the head axis to match
/// the query-head count for grouped-query attention.
///
/// Input:  `[num_kv_heads, seq, head_dim]`
/// Output: `[num_kv_heads * group_size, seq, head_dim]`
///
/// For output head `h`, the slice is copied from input head `h / group_size`.
/// This matches the standard GQA convention where each KV head serves
/// `group_size` consecutive query heads.
///
/// `group_size == 1` is a fast no-op clone (classical MHA path pays
/// nothing).
///
/// Note: like the other reshape helpers in this module, this breaks the
/// autograd graph — it is correct for inference but training-broken. A
/// fully-differentiable variant would require a `RepeatKvBackward` op
/// that sums gradients across replicated groups.
pub fn repeat_kv<T: Float>(kv: &Tensor<T>, group_size: usize) -> FerrotorchResult<Tensor<T>> {
    if group_size == 1 {
        return Ok(kv.clone());
    }
    let shape = kv.shape();
    if shape.len() != 3 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "repeat_kv expects 3-D [num_kv_heads, seq, head_dim], got {:?}",
                shape
            ),
        });
    }
    let num_kv_heads = shape[0];
    let seq = shape[1];
    let head_dim = shape[2];
    let num_q_heads = num_kv_heads * group_size;
    let data = kv.data_vec()?;
    let head_stride = seq * head_dim;
    let mut out = vec![<T as num_traits::Zero>::zero(); num_q_heads * head_stride];
    for h in 0..num_q_heads {
        let kv_h = h / group_size;
        let src_start = kv_h * head_stride;
        let dst_start = h * head_stride;
        out[dst_start..dst_start + head_stride]
            .copy_from_slice(&data[src_start..src_start + head_stride]);
    }
    let device = kv.device();
    Tensor::from_storage(
        TensorStorage::on_device(out, device)?,
        vec![num_q_heads, seq, head_dim],
        kv.requires_grad(),
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

    // -- Grouped-Query Attention tests (#505) ---------------------------

    #[test]
    fn test_with_gqa_valid_construction() {
        // Llama 3 8B layout: 32 query heads, 8 KV heads, head_dim=128.
        let mha = MultiheadAttention::<f32>::with_gqa(4096, 32, 8, false).unwrap();
        assert_eq!(mha.embed_dim(), 4096);
        assert_eq!(mha.num_heads(), 32);
        assert_eq!(mha.num_kv_heads(), 8);
        assert_eq!(mha.head_dim(), 128);
        assert!(mha.is_gqa());
    }

    #[test]
    fn test_with_gqa_kv_proj_shapes() {
        // K/V projections must output `num_kv_heads * head_dim`, not `embed_dim`.
        let mha = MultiheadAttention::<f32>::with_gqa(64, 8, 2, true).unwrap();
        let kv_dim = 2 * (64 / 8); // num_kv_heads * head_dim = 16
        assert_eq!(mha.q_proj.shape(), &[64, 64]);
        assert_eq!(mha.k_proj.shape(), &[kv_dim, 64]);
        assert_eq!(mha.v_proj.shape(), &[kv_dim, 64]);
        assert_eq!(mha.out_proj.shape(), &[64, 64]);
        // Biases follow the same split.
        assert_eq!(mha.q_bias.as_ref().unwrap().shape(), &[64]);
        assert_eq!(mha.k_bias.as_ref().unwrap().shape(), &[kv_dim]);
        assert_eq!(mha.v_bias.as_ref().unwrap().shape(), &[kv_dim]);
        assert_eq!(mha.out_bias.as_ref().unwrap().shape(), &[64]);
    }

    #[test]
    fn test_with_gqa_rejects_non_divisible_kv_heads() {
        // num_heads=8, num_kv_heads=3 → 8 % 3 != 0.
        let result = MultiheadAttention::<f32>::with_gqa(64, 8, 3, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_gqa_rejects_zero_kv_heads() {
        let result = MultiheadAttention::<f32>::with_gqa(64, 8, 0, false);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_gqa_equivalent_to_new_when_kv_equals_q() {
        // Passing num_kv_heads == num_heads must reproduce classical MHA shapes.
        let gqa = MultiheadAttention::<f32>::with_gqa(32, 4, 4, true).unwrap();
        let mha = MultiheadAttention::<f32>::new(32, 4, true).unwrap();
        assert_eq!(gqa.num_kv_heads(), mha.num_kv_heads());
        assert_eq!(gqa.k_proj.shape(), mha.k_proj.shape());
        assert_eq!(gqa.v_proj.shape(), mha.v_proj.shape());
        assert!(!gqa.is_gqa());
    }

    #[test]
    fn test_repeat_kv_noop_on_group_size_1() {
        // group_size=1 must be a cheap clone (the MHA hot path).
        let kv = ferrotorch_core::from_slice::<f32>(
            &(0..24).map(|i| i as f32).collect::<Vec<_>>(),
            &[2, 3, 4], // [num_kv_heads=2, seq=3, head_dim=4]
        )
        .unwrap();
        let out = repeat_kv(&kv, 1).unwrap();
        assert_eq!(out.shape(), kv.shape());
        assert_eq!(out.data_vec().unwrap(), kv.data_vec().unwrap());
    }

    #[test]
    fn test_repeat_kv_copies_correct_heads() {
        // Input: 2 KV heads, each a 1x3 row with distinct values per head.
        // group_size=3 → 6 output heads. Heads 0,1,2 should equal input head 0;
        // heads 3,4,5 should equal input head 1.
        let data: Vec<f32> = vec![
            10.0, 11.0, 12.0, // head 0, seq 0
            13.0, 14.0, 15.0, // head 0, seq 1
            20.0, 21.0, 22.0, // head 1, seq 0
            23.0, 24.0, 25.0, // head 1, seq 1
        ];
        let kv = ferrotorch_core::from_slice::<f32>(&data, &[2, 2, 3]).unwrap();
        let out = repeat_kv(&kv, 3).unwrap();
        assert_eq!(out.shape(), &[6, 2, 3]);
        let out_data = out.data_vec().unwrap();
        let head_stride = 2 * 3; // seq * head_dim
        // Heads 0, 1, 2 come from input head 0.
        for h in 0..3 {
            let start = h * head_stride;
            assert_eq!(&out_data[start..start + head_stride], &data[0..head_stride]);
        }
        // Heads 3, 4, 5 come from input head 1.
        for h in 3..6 {
            let start = h * head_stride;
            assert_eq!(
                &out_data[start..start + head_stride],
                &data[head_stride..2 * head_stride]
            );
        }
    }

    #[test]
    fn test_repeat_kv_rejects_wrong_rank() {
        let kv = ferrotorch_core::zeros::<f32>(&[4, 8]).unwrap(); // 2-D
        assert!(repeat_kv(&kv, 2).is_err());
    }

    #[test]
    fn test_gqa_forward_output_shape_preserved() {
        // GQA must return the same [batch, seq, embed_dim] shape as MHA.
        let mha = MultiheadAttention::<f32>::with_gqa(16, 4, 2, true).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 16]).unwrap();
        let out = mha.forward(&input).unwrap();
        assert_eq!(out.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_gqa_forward_produces_finite_values() {
        let mha = MultiheadAttention::<f64>::with_gqa(8, 4, 2, true).unwrap();
        let input = ferrotorch_core::ones::<f64>(&[1, 3, 8]).unwrap();
        let out = mha.forward(&input).unwrap();
        let data = out.data().unwrap();
        for &v in data {
            assert!(v.is_finite(), "GQA output non-finite: {v}");
        }
    }

    #[test]
    fn test_gqa_forward_decoder_style_single_token() {
        // Single-token forward (seq_q == seq_k == 1) must stay numerically
        // stable on the GQA path — this is the autoregressive generation
        // hot case for Llama inference.
        let mha = MultiheadAttention::<f32>::with_gqa(32, 8, 2, false).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[1, 1, 32]).unwrap();
        let out = mha.forward(&input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 32]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_gqa_forward_with_causal_mask() {
        // Causal masking must still work when K/V have fewer heads than Q.
        let mha = MultiheadAttention::<f32>::with_gqa(16, 4, 2, false).unwrap();
        let x = ferrotorch_core::ones::<f32>(&[1, 4, 16]).unwrap();
        let out = mha.forward_qkv(&x, &x, &x, true).unwrap();
        assert_eq!(out.shape(), &[1, 4, 16]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite());
        }
    }
}
