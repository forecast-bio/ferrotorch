//! Llama attention layer.
//!
//! Composes `Linear` projections for Q / K / V / O with
//! `RotaryPositionEmbedding` and a GQA-aware attention kernel built on
//! the [`ferrotorch_nn::standard_attention`] primitive.
//!
//! # Shape contract
//!
//! Current scope is single-batch inference: the input tensor must be
//! `[1, seq_len, hidden_size]`. Multi-batch (`B > 1`) support needs a
//! 4-D `reshape_to_heads` helper and will follow.
//!
//! # RoPE
//!
//! RoPE is applied to `Q` and `K` *after* projection but *before*
//! attention, using the [`RoPEConvention::HalfRotation`] pairing to
//! match the HuggingFace / Meta Llama implementation.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{
    Linear, RoPEConvention, RoPEScaling, RotaryPositionEmbedding, repeat_kv, reshape_to_heads,
    standard_attention, transpose_heads_to_2d,
};

use crate::config::LlamaConfig;
use crate::kv_cache::LayerKvCache;

/// Llama multi-head / grouped-query attention block.
#[derive(Debug)]
pub struct LlamaAttention<T: Float> {
    /// Query projection. Maps `[B, S, hidden]` to `[B, S, hidden]`.
    pub q_proj: Linear<T>,
    /// Key projection. Maps `[B, S, hidden]` to `[B, S, num_kv_heads * head_dim]`.
    pub k_proj: Linear<T>,
    /// Value projection. Same shape contract as `k_proj`.
    pub v_proj: Linear<T>,
    /// Output projection. Maps `[B, S, hidden]` back to `[B, S, hidden]`.
    pub o_proj: Linear<T>,
    /// Half-rotation rotary positional embedding applied to Q and K
    /// before the attention scores matmul.
    pub rope: RotaryPositionEmbedding<T>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    training: bool,
}

impl<T: Float> LlamaAttention<T> {
    /// Build a randomly-initialized attention block for the given config.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] if config validation
    /// fails or any of the four `Linear` projections / RoPE table fails
    /// to construct (typically a `ShapeMismatch` on a degenerate
    /// `hidden_size` / `num_heads` / `head_dim` combination).
    pub fn new(cfg: &LlamaConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let head_dim = cfg.head_dim();
        let kv_dim = cfg.num_key_value_heads * head_dim;

        Ok(Self {
            q_proj: Linear::new(cfg.hidden_size, cfg.hidden_size, false)?,
            k_proj: Linear::new(cfg.hidden_size, kv_dim, false)?,
            v_proj: Linear::new(cfg.hidden_size, kv_dim, false)?,
            o_proj: Linear::new(cfg.hidden_size, cfg.hidden_size, false)?,
            rope: RotaryPositionEmbedding::with_scaling(
                head_dim,
                cfg.max_position_embeddings,
                cfg.rope_theta,
                RoPEConvention::HalfRotation,
                RoPEScaling::None,
            )?,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            training: false,
        })
    }

    /// Number of query heads in this attention block.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
    /// Number of key/value heads (`<= num_heads` for grouped-query attention).
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
    /// Per-head feature dimension; equals `hidden_size / num_heads`.
    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Incremental forward pass for a single new token, attending
    /// against a persistent K/V cache (#1129).
    ///
    /// # Input
    ///
    /// - `input`: `[1, 1, hidden_size]` — the embedded representation
    ///   of the single new token at sequence position `seq_offset`.
    /// - `cache`: optional existing [`LayerKvCache`] holding the
    ///   previously-attended `[Hkv, S, d]` K and V slabs. `None` on
    ///   the very first token.
    /// - `seq_offset`: absolute position of the new token (i.e.
    ///   `cache.seq_len()` when `cache` is `Some`, else `0`). Drives
    ///   the RoPE angle.
    ///
    /// # Output
    ///
    /// `(attention_output[1, 1, hidden_size], new_layer_kv_cache)`.
    /// The returned `LayerKvCache` has `seq_len = seq_offset + 1`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when `input` is
    /// not `[1, 1, hidden_size]`, or [`FerrotorchError::ShapeMismatch`]
    /// when an existing `cache`'s K/V tensors disagree with
    /// `num_kv_heads` / `head_dim`.
    pub fn forward_with_cache(
        &self,
        input: &Tensor<T>,
        cache: Option<&LayerKvCache<T>>,
        seq_offset: usize,
    ) -> FerrotorchResult<(Tensor<T>, LayerKvCache<T>)> {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[1] != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LlamaAttention::forward_with_cache expects [1, 1, hidden] input, got {shape:?}"
                ),
            });
        }
        let hidden = self.num_heads * self.head_dim;
        if shape[2] != hidden {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LlamaAttention::forward_with_cache: input last dim {} != hidden {}",
                    shape[2], hidden
                ),
            });
        }

        // Project Q/K/V for the single new token: q [1,1,H*d], k/v [1,1,Hkv*d].
        let q = self.q_proj.forward(input)?;
        let k_new = self.k_proj.forward(input)?;
        let v_new = self.v_proj.forward(input)?;
        let kv_dim = self.num_kv_heads * self.head_dim;
        // [1, 1, H*d] -> [1, H*d]
        let q2 = reshape_2d(&q, 1, hidden)?;
        let k2 = reshape_2d(&k_new, 1, kv_dim)?;
        let v2 = reshape_2d(&v_new, 1, kv_dim)?;
        // [1, H*d] -> [H, 1, d]
        let q_h = reshape_to_heads(&q2, self.num_heads, 1, self.head_dim)?;
        let k_h = reshape_to_heads(&k2, self.num_kv_heads, 1, self.head_dim)?;
        let v_h = reshape_to_heads(&v2, self.num_kv_heads, 1, self.head_dim)?;

        // Apply RoPE at absolute position `seq_offset` (only Q and K
        // get rotated; V is positional-invariant).
        let q_rot = self.rope.apply(&q_h, seq_offset)?;
        let k_rot = self.rope.apply(&k_h, seq_offset)?;

        // Append the new K/V row to the existing cache (or seed the
        // cache if this is the first token). The cache stores
        // [Hkv, seq, d] post-RoPE keys and raw values.
        let new_cache = if let Some(prev) = cache {
            prev.append(&k_rot, &v_h)?
        } else {
            if seq_offset != 0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "LlamaAttention::forward_with_cache: cache=None but \
                         seq_offset={seq_offset} (expected 0 for first token)"
                    ),
                });
            }
            LayerKvCache::from_single_token(k_rot, v_h)?
        };

        // GQA broadcast: K/V heads expand to match Q head count.
        let group_size = self.num_heads / self.num_kv_heads;
        let k_full = repeat_kv(&new_cache.k, group_size)?;
        let v_full = repeat_kv(&new_cache.v, group_size)?;

        // Attention: N_q=1 against N_k=seq_offset+1. Use causal=false
        // since the standard_attention causal mask is "j > i" which
        // would mask everything but position 0 when N_q=1 — wrong for
        // incremental decoding where the single query position is the
        // *last* row of the full sequence and can attend to all
        // history. With N_q=1 there is no future to mask.
        let ctx = standard_attention(&q_rot, &k_full, &v_full, false)?;

        // [H, 1, d] -> [1, H*d]
        let ctx2 = transpose_heads_to_2d(&ctx, self.num_heads, 1, self.head_dim)?;
        // [1, H*d] -> [1, 1, hidden]
        let ctx3 = reshape_3d(&ctx2, 1, 1, hidden)?;
        let out = self.o_proj.forward(&ctx3)?;
        Ok((out, new_cache))
    }
}

impl<T: Float> Module<T> for LlamaAttention<T> {
    /// Forward pass.
    ///
    /// # Input
    ///
    /// `[batch=1, seq_len, hidden_size]`.
    ///
    /// # Output
    ///
    /// `[1, seq_len, hidden_size]`.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LlamaAttention expects 3-D [B, S, D] input, got {}-D {:?}",
                    shape.len(),
                    shape
                ),
            });
        }
        let (batch, seq_len, embed_dim) = (shape[0], shape[1], shape[2]);
        if batch != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LlamaAttention currently supports batch=1 only, got batch={batch}"
                ),
            });
        }
        let hidden = self.num_heads * self.head_dim;
        if embed_dim != hidden {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LlamaAttention input last dim {embed_dim} does not match \
                     num_heads * head_dim = {hidden}"
                ),
            });
        }

        // Project QKV. Linear handles any rank; output is [1, S, H*d] for Q
        // and [1, S, Hkv*d] for K/V.
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Squeeze the leading batch=1 dim; reshape_to_heads is 2-D in → 3-D out.
        let q2 = reshape_2d(&q, seq_len, hidden)?;
        let kv_dim = self.num_kv_heads * self.head_dim;
        let k2 = reshape_2d(&k, seq_len, kv_dim)?;
        let v2 = reshape_2d(&v, seq_len, kv_dim)?;

        // Split into heads. [S, H*d] → [H, S, d], [S, Hkv*d] → [Hkv, S, d].
        let q_h = reshape_to_heads(&q2, self.num_heads, seq_len, self.head_dim)?;
        let k_h = reshape_to_heads(&k2, self.num_kv_heads, seq_len, self.head_dim)?;
        let v_h = reshape_to_heads(&v2, self.num_kv_heads, seq_len, self.head_dim)?;

        // Apply RoPE to Q and K (RoPE applies over the last two axes,
        // here (seq, head_dim) per head).
        let q_h = self.rope.apply(&q_h, 0)?;
        let k_h = self.rope.apply(&k_h, 0)?;

        // GQA broadcast: K/V heads expand to match Q head count.
        let group_size = self.num_heads / self.num_kv_heads;
        let k_h = repeat_kv(&k_h, group_size)?;
        let v_h = repeat_kv(&v_h, group_size)?;

        // Per-head scaled dot-product attention. standard_attention
        // treats the first axis as batch, which is what we want: one
        // attention computation per head.
        let ctx = standard_attention(&q_h, &k_h, &v_h, true)?;

        // Re-merge heads. [H, S, d] → [S, H*d].
        let ctx2 = transpose_heads_to_2d(&ctx, self.num_heads, seq_len, self.head_dim)?;

        // Restore the leading batch=1 dim and project out.
        let ctx3 = reshape_3d(&ctx2, 1, seq_len, hidden)?;
        self.o_proj.forward(&ctx3)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.q_proj.parameters());
        out.extend(self.k_proj.parameters());
        out.extend(self.v_proj.parameters());
        out.extend(self.o_proj.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.q_proj.parameters_mut());
        out.extend(self.k_proj.parameters_mut());
        out.extend(self.v_proj.parameters_mut());
        out.extend(self.o_proj.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.q_proj.named_parameters() {
            out.push((format!("q_proj.{n}"), p));
        }
        for (n, p) in self.k_proj.named_parameters() {
            out.push((format!("k_proj.{n}"), p));
        }
        for (n, p) in self.v_proj.named_parameters() {
            out.push((format!("v_proj.{n}"), p));
        }
        for (n, p) in self.o_proj.named_parameters() {
            out.push((format!("o_proj.{n}"), p));
        }
        out
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

    fn state_dict(&self) -> StateDict<T> {
        self.named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.tensor().clone()))
            .collect()
    }

    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let expected = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| {
                    k.strip_prefix(&expected)
                        .map(|rest| (rest.to_string(), v.clone()))
                })
                .collect()
        };

        if strict {
            let prefixes = ["q_proj", "k_proj", "v_proj", "o_proj"];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in LlamaAttention state_dict: \"{key}\""),
                    });
                }
            }
        }

        self.q_proj.load_state_dict(&extract("q_proj"), strict)?;
        self.k_proj.load_state_dict(&extract("k_proj"), strict)?;
        self.v_proj.load_state_dict(&extract("v_proj"), strict)?;
        self.o_proj.load_state_dict(&extract("o_proj"), strict)?;
        Ok(())
    }
}

/// Reshape a 3-D `[1, a, b]` tensor to 2-D `[a, b]` (squeeze batch).
fn reshape_2d<T: Float>(t: &Tensor<T>, a: usize, b: usize) -> FerrotorchResult<Tensor<T>> {
    let data = t.data_vec()?;
    use ferrotorch_core::TensorStorage;
    Tensor::from_storage(TensorStorage::cpu(data), vec![a, b], t.requires_grad())
}

/// Inverse of [`reshape_2d`]: `[a, b]` → `[c, a, b]` (add leading dim).
fn reshape_3d<T: Float>(
    t: &Tensor<T>,
    c: usize,
    a: usize,
    b: usize,
) -> FerrotorchResult<Tensor<T>> {
    let data = t.data_vec()?;
    use ferrotorch_core::TensorStorage;
    Tensor::from_storage(TensorStorage::cpu(data), vec![c, a, b], t.requires_grad())
}
