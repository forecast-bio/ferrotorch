//! LLM-critical transformer building blocks.
//!
//! This module provides the components needed to build modern large language
//! models:
//!
//! - [`RotaryPositionEmbedding`] — Rotary Position Embeddings (RoPE) as
//!   described in Su et al. (2021). Precomputes sin/cos tables and applies
//!   pairwise rotation to queries and keys.
//!
//! - [`SwiGLU`] — The gated linear unit with SiLU activation used in
//!   LLaMA, Mistral, and other modern architectures: `w3(silu(w1(x)) * w2(x))`.
//!
//! - [`KVCache`] — Key-value cache for efficient autoregressive inference,
//!   concatenating new key/value pairs with previously cached ones.
//!
//! - [`TransformerEncoderLayer`] — A pre-norm encoder block:
//!   `norm -> self-attn -> residual -> norm -> ffn -> residual`.
//!
//! - [`TransformerDecoderLayer`] — A pre-norm decoder block with
//!   self-attention, cross-attention, and feedforward sub-layers.

use ferrotorch_core::grad_fns::activation::silu;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::{Float, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

use crate::attention::MultiheadAttention;
use crate::dropout::Dropout;
use crate::linear::Linear;
use crate::module::Module;
use crate::norm::LayerNorm;
use crate::parameter::Parameter;

// ===========================================================================
// RotaryPositionEmbedding (RoPE)
// ===========================================================================

/// Selects how RoPE pairs elements for rotation.
///
/// - **`Interleaved`** (default) — pairs `(x[2i], x[2i+1])`.
///   Used by the original RoFormer paper.
/// - **`HalfRotation`** — pairs `(x[i], x[i + d/2])`.
///   Used by LLaMA, GPT-NeoX, and Pythia.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RoPEConvention {
    /// Pairs consecutive elements: `(x[2i], x[2i+1])`. Original RoFormer.
    #[default]
    Interleaved,
    /// Pairs first-half with second-half: `(x[i], x[i+d/2])`. LLaMA/GPT-NeoX.
    HalfRotation,
}

impl std::fmt::Display for RoPEConvention {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RoPEConvention::Interleaved => write!(f, "interleaved"),
            RoPEConvention::HalfRotation => write!(f, "half_rotation"),
        }
    }
}

/// Rotary Position Embeddings (RoPE).
///
/// Precomputes sin/cos frequency tables up to `max_seq_len` and applies
/// pairwise rotation to an input tensor, encoding absolute position
/// information that degrades gracefully with relative distance.
///
/// Two element-pairing conventions are supported (see [`RoPEConvention`]):
///
/// - **Interleaved** (default): pairs `(x[2i], x[2i+1])` — RoFormer.
/// - **HalfRotation**: pairs `(x[i], x[i+d/2])` — LLaMA, GPT-NeoX, Pythia.
///
/// RoPE is **not** a [`Module`] — it is a stateless utility applied inside
/// attention layers before the dot product.
///
/// # Shape contract
///
/// - Input: `[..., seq_len, dim]` where `dim` is even.
/// - Output: same shape as input.
///
/// # Reference
///
/// Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021).
#[derive(Debug)]
pub struct RotaryPositionEmbedding<T: Float> {
    dim: usize,
    max_seq_len: usize,
    base: f64,
    convention: RoPEConvention,
    /// Precomputed cosines: `[max_seq_len, dim/2]`.
    cos_cache: Tensor<T>,
    /// Precomputed sines: `[max_seq_len, dim/2]`.
    sin_cache: Tensor<T>,
}

impl<T: Float> RotaryPositionEmbedding<T> {
    /// Create a new RoPE instance with the default interleaved convention.
    ///
    /// # Arguments
    ///
    /// - `dim` - The embedding dimension (must be even).
    /// - `max_seq_len` - Maximum sequence length to precompute.
    /// - `base` - Base for the frequency computation (default: 10 000.0).
    ///
    /// # Errors
    ///
    /// Returns an error if `dim` is odd or zero, or if `max_seq_len` is zero.
    pub fn new(dim: usize, max_seq_len: usize, base: f64) -> FerrotorchResult<Self> {
        Self::with_convention(dim, max_seq_len, base, RoPEConvention::default())
    }

    /// Create a new RoPE instance with a specified pairing convention.
    ///
    /// Use [`RoPEConvention::HalfRotation`] for LLaMA, GPT-NeoX, and Pythia
    /// compatibility.
    pub fn with_convention(
        dim: usize,
        max_seq_len: usize,
        base: f64,
        convention: RoPEConvention,
    ) -> FerrotorchResult<Self> {
        if dim == 0 || dim % 2 != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("RoPE dim must be even and positive, got {dim}"),
            });
        }
        if max_seq_len == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "RoPE max_seq_len must be positive".into(),
            });
        }

        let half_dim = dim / 2;

        // theta_i = 1 / base^(2i / dim) for i in 0..dim/2
        let thetas: Vec<f64> = (0..half_dim)
            .map(|i| 1.0 / base.powf(2.0 * i as f64 / dim as f64))
            .collect();

        // cos_cache[pos, i] = cos(pos * theta_i)
        // sin_cache[pos, i] = sin(pos * theta_i)
        let total = max_seq_len * half_dim;
        let mut cos_data = Vec::with_capacity(total);
        let mut sin_data = Vec::with_capacity(total);

        for pos in 0..max_seq_len {
            for i in 0..half_dim {
                let angle = pos as f64 * thetas[i];
                cos_data.push(T::from(angle.cos()).unwrap());
                sin_data.push(T::from(angle.sin()).unwrap());
            }
        }

        let cos_cache = Tensor::from_storage(
            TensorStorage::cpu(cos_data),
            vec![max_seq_len, half_dim],
            false,
        )?;
        let sin_cache = Tensor::from_storage(
            TensorStorage::cpu(sin_data),
            vec![max_seq_len, half_dim],
            false,
        )?;

        Ok(Self {
            dim,
            max_seq_len,
            base,
            convention,
            cos_cache,
            sin_cache,
        })
    }

    /// Apply rotary embeddings to `x` starting at position `seq_offset`.
    ///
    /// # Shape
    ///
    /// - `x`: any shape where the last dimension equals `dim` and the
    ///   second-to-last dimension is the sequence length.
    /// - Returns a tensor of the same shape.
    ///
    /// For a typical attention head input of shape `[batch, num_heads, seq_len, head_dim]`,
    /// the rotation is applied over `(seq_len, head_dim)`.
    ///
    /// # Errors
    ///
    /// Returns an error if `seq_offset + seq_len > max_seq_len` or if the
    /// last dimension of `x` does not equal `dim`.
    pub fn apply(&self, x: &Tensor<T>, seq_offset: usize) -> FerrotorchResult<Tensor<T>> {
        let shape = x.shape();
        let ndim = shape.len();
        if ndim < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RoPE input must be at least 2-D, got {ndim}-D with shape {shape:?}"
                ),
            });
        }

        let last_dim = shape[ndim - 1];
        if last_dim != self.dim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "RoPE: last dim of input ({last_dim}) != dim ({})",
                    self.dim
                ),
            });
        }

        let seq_len = shape[ndim - 2];
        if seq_offset + seq_len > self.max_seq_len {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RoPE: seq_offset ({seq_offset}) + seq_len ({seq_len}) > max_seq_len ({})",
                    self.max_seq_len
                ),
            });
        }

        let half_dim = self.dim / 2;
        let cos_data = self.cos_cache.data()?;
        let sin_data = self.sin_cache.data()?;
        let x_data = x.data()?;

        // Number of independent "instances" before the (seq, dim) axes.
        let batch_dims: usize = shape[..ndim - 2].iter().product();

        let total = x.numel();
        let mut output = Vec::with_capacity(total);

        match self.convention {
            RoPEConvention::Interleaved => {
                // Pair (x[2i], x[2i+1]) — original RoFormer convention.
                for b in 0..batch_dims {
                    for s in 0..seq_len {
                        let pos = seq_offset + s;
                        let cache_start = pos * half_dim;
                        let x_start = b * seq_len * self.dim + s * self.dim;

                        for i in 0..half_dim {
                            let x_even = x_data[x_start + 2 * i];
                            let x_odd = x_data[x_start + 2 * i + 1];
                            let cos_val = cos_data[cache_start + i];
                            let sin_val = sin_data[cache_start + i];

                            output.push(x_even * cos_val - x_odd * sin_val);
                            output.push(x_even * sin_val + x_odd * cos_val);
                        }
                    }
                }
            }
            RoPEConvention::HalfRotation => {
                // Pair (x[i], x[i + d/2]) — LLaMA/GPT-NeoX convention.
                // Output layout: first half then second half (same as input).
                for b in 0..batch_dims {
                    for s in 0..seq_len {
                        let pos = seq_offset + s;
                        let cache_start = pos * half_dim;
                        let x_start = b * seq_len * self.dim + s * self.dim;

                        // First half: x_rot[i] = x[i] * cos - x[i + d/2] * sin
                        for i in 0..half_dim {
                            let x_first = x_data[x_start + i];
                            let x_second = x_data[x_start + half_dim + i];
                            let cos_val = cos_data[cache_start + i];
                            let sin_val = sin_data[cache_start + i];

                            output.push(x_first * cos_val - x_second * sin_val);
                        }
                        // Second half: x_rot[i + d/2] = x[i] * sin + x[i + d/2] * cos
                        for i in 0..half_dim {
                            let x_first = x_data[x_start + i];
                            let x_second = x_data[x_start + half_dim + i];
                            let cos_val = cos_data[cache_start + i];
                            let sin_val = sin_data[cache_start + i];

                            output.push(x_first * sin_val + x_second * cos_val);
                        }
                    }
                }
            }
        }

        Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)
    }

    /// The embedding dimension.
    #[inline]
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The maximum sequence length the cache supports.
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// The frequency base.
    #[inline]
    pub fn base(&self) -> f64 {
        self.base
    }

    /// The pairing convention.
    #[inline]
    pub fn convention(&self) -> RoPEConvention {
        self.convention
    }
}

// ===========================================================================
// SwiGLU
// ===========================================================================

/// Gated Linear Unit with SiLU activation (SwiGLU).
///
/// Applies the feedforward network used in LLaMA, Mistral, and other
/// modern transformer architectures:
///
/// ```text
/// SwiGLU(x) = w3(silu(w1(x)) * w2(x))
/// ```
///
/// where `w1` is the gate projection, `w2` is the up projection, and
/// `w3` is the down projection.
///
/// # Shape contract
///
/// - Input: `[batch, seq_len, in_features]` (3-D) or `[batch, in_features]` (2-D).
/// - Output: same shape as input.
///
/// Internally, `w1` and `w2` project from `in_features` to `hidden_features`,
/// and `w3` projects back from `hidden_features` to `in_features`.
#[derive(Debug)]
pub struct SwiGLU<T: Float> {
    /// Gate projection: `[in_features] -> [hidden_features]`.
    w1: Linear<T>,
    /// Up projection: `[in_features] -> [hidden_features]`.
    w2: Linear<T>,
    /// Down projection: `[hidden_features] -> [in_features]`.
    w3: Linear<T>,
    training: bool,
}

impl<T: Float> SwiGLU<T> {
    /// Create a new SwiGLU layer.
    ///
    /// # Arguments
    ///
    /// - `in_features` - Input (and output) dimension.
    /// - `hidden_features` - Hidden dimension of the gate/up projections.
    ///   A common choice is `(8/3) * in_features` rounded to a multiple of 256.
    /// - `bias` - Whether to include bias in the linear layers.
    pub fn new(
        in_features: usize,
        hidden_features: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        let w1 = Linear::new(in_features, hidden_features, bias)?;
        let w2 = Linear::new(in_features, hidden_features, bias)?;
        let w3 = Linear::new(hidden_features, in_features, bias)?;

        Ok(Self {
            w1,
            w2,
            w3,
            training: true,
        })
    }

    /// Forward pass for 3-D input `[batch, seq_len, in_features]`.
    ///
    /// Internally reshapes to 2-D, applies the SwiGLU computation, then
    /// reshapes back.
    fn forward_3d(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        let batch = shape[0];
        let seq_len = shape[1];
        let features = shape[2];

        // Flatten to [batch * seq_len, features].
        let flat = Tensor::from_storage(
            TensorStorage::cpu(input.data()?.to_vec()),
            vec![batch * seq_len, features],
            input.requires_grad(),
        )?;

        let output_flat = self.forward_2d(&flat)?;

        // Reshape back to [batch, seq_len, out_features].
        let out_features = output_flat.shape()[1];
        Tensor::from_storage(
            TensorStorage::cpu(output_flat.data()?.to_vec()),
            vec![batch, seq_len, out_features],
            false,
        )
    }

    /// Forward pass for 2-D input `[batch, in_features]`.
    fn forward_2d(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // gate = silu(w1(x))
        let w1_out = self.w1.forward(input)?;
        let gate = silu(&w1_out)?;

        // up = w2(x)
        let up = self.w2.forward(input)?;

        // gated = gate * up (elementwise)
        let gated = mul(&gate, &up)?;

        // down = w3(gated)
        self.w3.forward(&gated)
    }
}

impl<T: Float> Module<T> for SwiGLU<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        match input.ndim() {
            2 => self.forward_2d(input),
            3 => self.forward_3d(input),
            _ => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "SwiGLU expects 2-D or 3-D input, got {}-D with shape {:?}",
                    input.ndim(),
                    input.shape()
                ),
            }),
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = self.w1.parameters();
        params.extend(self.w2.parameters());
        params.extend(self.w3.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = self.w1.parameters_mut();
        params.extend(self.w2.parameters_mut());
        params.extend(self.w3.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, param) in self.w1.named_parameters() {
            params.push((format!("w1.{name}"), param));
        }
        for (name, param) in self.w2.named_parameters() {
            params.push((format!("w2.{name}"), param));
        }
        for (name, param) in self.w3.named_parameters() {
            params.push((format!("w3.{name}"), param));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.w1.train();
        self.w2.train();
        self.w3.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.w1.eval();
        self.w2.eval();
        self.w3.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// KVCache
// ===========================================================================

/// Key-value cache for efficient autoregressive (incremental) inference.
///
/// During generation, previously computed keys and values are cached so that
/// each new token only requires computing its own K and V, then concatenating
/// with the cache before attention.
///
/// # Shape convention
///
/// Keys and values are expected as `[batch, num_heads, seq_len, head_dim]`.
/// The cache grows along the `seq_len` axis (dimension 2).
#[derive(Debug)]
pub struct KVCache<T: Float> {
    /// Cached keys: `[B, num_heads, cached_seq, head_dim]`, or `None` if empty.
    key_cache: Option<Tensor<T>>,
    /// Cached values: same shape as `key_cache`.
    value_cache: Option<Tensor<T>>,
    /// Maximum sequence length the cache will hold.
    max_seq_len: usize,
}

impl<T: Float> KVCache<T> {
    /// Create a new empty cache.
    ///
    /// `max_seq_len` is the upper bound on the total cached sequence length.
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            key_cache: None,
            value_cache: None,
            max_seq_len,
        }
    }

    /// Append new keys and values to the cache.
    ///
    /// Returns the **full** (concatenated) keys and values for use in the
    /// current attention step.
    ///
    /// # Arguments
    ///
    /// - `key` - New key tensor: `[B, num_heads, new_seq, head_dim]`.
    /// - `value` - New value tensor: same shape as `key`.
    ///
    /// # Returns
    ///
    /// `(full_key, full_value)` with shape `[B, num_heads, cached_seq + new_seq, head_dim]`.
    pub fn update(
        &mut self,
        key: Tensor<T>,
        value: Tensor<T>,
    ) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        if key.ndim() != 4 || value.ndim() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "KVCache expects 4-D [B, heads, seq, dim] tensors, \
                     got key {:?}, value {:?}",
                    key.shape(),
                    value.shape()
                ),
            });
        }

        if key.shape() != value.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "KVCache: key shape {:?} != value shape {:?}",
                    key.shape(),
                    value.shape()
                ),
            });
        }

        let (full_key, full_value) = match (&self.key_cache, &self.value_cache) {
            (Some(ck), Some(cv)) => {
                let fk = concat_along_dim2(ck, &key)?;
                let fv = concat_along_dim2(cv, &value)?;
                (fk, fv)
            }
            _ => (key.clone(), value.clone()),
        };

        // Check that the total cached sequence length does not exceed the limit.
        let total_seq = full_key.shape()[2];
        if total_seq > self.max_seq_len {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "KVCache: total sequence length ({total_seq}) exceeds max_seq_len ({})",
                    self.max_seq_len
                ),
            });
        }

        self.key_cache = Some(full_key.clone());
        self.value_cache = Some(full_value.clone());

        Ok((full_key, full_value))
    }

    /// Reset the cache, discarding all stored keys and values.
    pub fn reset(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
    }

    /// The current cached sequence length (0 if empty).
    pub fn seq_len(&self) -> usize {
        self.key_cache
            .as_ref()
            .map(|k| k.shape()[2])
            .unwrap_or(0)
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.key_cache.is_none()
    }

    /// Maximum sequence length.
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }
}

/// Concatenate two 4-D tensors along dimension 2 (the sequence axis).
///
/// Shapes must match on dims 0, 1, 3.
fn concat_along_dim2<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let sa = a.shape();
    let sb = b.shape();

    if sa[0] != sb[0] || sa[1] != sb[1] || sa[3] != sb[3] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "concat_along_dim2: shapes {:?} and {:?} must match on dims 0, 1, 3",
                sa, sb
            ),
        });
    }

    let (batch, heads, seq_a, dim) = (sa[0], sa[1], sa[2], sa[3]);
    let seq_b = sb[2];
    let seq_out = seq_a + seq_b;

    let a_data = a.data()?;
    let b_data = b.data()?;

    let mut output = Vec::with_capacity(batch * heads * seq_out * dim);

    for ba in 0..batch {
        for h in 0..heads {
            // Copy rows from a.
            let a_start = (ba * heads + h) * seq_a * dim;
            output.extend_from_slice(&a_data[a_start..a_start + seq_a * dim]);
            // Copy rows from b.
            let b_start = (ba * heads + h) * seq_b * dim;
            output.extend_from_slice(&b_data[b_start..b_start + seq_b * dim]);
        }
    }

    Tensor::from_storage(
        TensorStorage::cpu(output),
        vec![batch, heads, seq_out, dim],
        false,
    )
}

// ===========================================================================
// TransformerEncoderLayer
// ===========================================================================

/// A single pre-norm transformer encoder layer.
///
/// Applies the following computation:
///
/// ```text
/// x = x + dropout(self_attn(norm1(x)))
/// x = x + dropout(ffn(norm2(x)))
/// ```
///
/// This matches the pre-norm (Pre-LN) style used in GPT-2, LLaMA, and most
/// modern LLMs, which trains more stably than the original post-norm design.
///
/// # Shape contract
///
/// - Input: `[batch, seq_len, d_model]`
/// - Output: `[batch, seq_len, d_model]`
#[derive(Debug)]
pub struct TransformerEncoderLayer<T: Float> {
    self_attn: MultiheadAttention<T>,
    ffn: SwiGLU<T>,
    norm1: LayerNorm<T>,
    norm2: LayerNorm<T>,
    dropout: Dropout<T>,
    training: bool,
}

impl<T: Float> TransformerEncoderLayer<T> {
    /// Create a new transformer encoder layer.
    ///
    /// # Arguments
    ///
    /// - `d_model` - The model dimension (embedding size).
    /// - `num_heads` - Number of attention heads.
    /// - `d_ff` - Hidden dimension of the SwiGLU feedforward network.
    /// - `dropout_p` - Dropout probability (applied after attention and FFN).
    /// - `layer_norm_eps` - Epsilon for layer normalization.
    /// - `bias` - Whether to use bias in attention and FFN projections.
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_p: f64,
        layer_norm_eps: f64,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        let self_attn = MultiheadAttention::new(d_model, num_heads, bias)?;
        let ffn = SwiGLU::new(d_model, d_ff, bias)?;
        let norm1 = LayerNorm::new(vec![d_model], layer_norm_eps, true)?;
        let norm2 = LayerNorm::new(vec![d_model], layer_norm_eps, true)?;
        let dropout = Dropout::new(dropout_p)?;

        Ok(Self {
            self_attn,
            ffn,
            norm1,
            norm2,
            dropout,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for TransformerEncoderLayer<T> {
    /// Forward pass with pre-norm residual connections.
    ///
    /// Input shape: `[batch, seq_len, d_model]`.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "TransformerEncoderLayer expects 3-D [batch, seq, d_model], got {:?}",
                    input.shape()
                ),
            });
        }

        // Pre-norm self-attention block.
        let normed1 = self.norm1.forward(input)?;
        let attn_out = self.self_attn.forward(&normed1)?;
        let attn_out = self.dropout.forward(&attn_out)?;
        let residual1 = add(input, &attn_out)?;

        // Pre-norm feedforward block.
        let normed2 = self.norm2.forward(&residual1)?;
        let ffn_out = self.ffn.forward(&normed2)?;
        let ffn_out = self.dropout.forward(&ffn_out)?;
        let residual2 = add(&residual1, &ffn_out)?;

        Ok(residual2)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.ffn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        // Dropout has no parameters.
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.ffn.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, param) in self.self_attn.named_parameters() {
            params.push((format!("self_attn.{name}"), param));
        }
        for (name, param) in self.ffn.named_parameters() {
            params.push((format!("ffn.{name}"), param));
        }
        for (name, param) in self.norm1.named_parameters() {
            params.push((format!("norm1.{name}"), param));
        }
        for (name, param) in self.norm2.named_parameters() {
            params.push((format!("norm2.{name}"), param));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.self_attn.train();
        self.ffn.train();
        self.norm1.train();
        self.norm2.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.self_attn.eval();
        self.ffn.eval();
        self.norm1.eval();
        self.norm2.eval();
        self.dropout.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// TransformerDecoderLayer
// ===========================================================================

/// A single pre-norm transformer decoder layer with cross-attention.
///
/// Applies the following computation:
///
/// ```text
/// x = x + dropout(self_attn(norm1(x)))
/// x = x + dropout(cross_attn(norm2(x), memory, memory))
/// x = x + dropout(ffn(norm3(x)))
/// ```
///
/// The self-attention sub-layer uses causal masking. The cross-attention
/// sub-layer attends over encoder output (`memory`).
///
/// # Shape contract
///
/// - `input` (decoder): `[batch, tgt_seq, d_model]`
/// - `memory` (encoder output): `[batch, src_seq, d_model]`
/// - Output: `[batch, tgt_seq, d_model]`
#[derive(Debug)]
pub struct TransformerDecoderLayer<T: Float> {
    self_attn: MultiheadAttention<T>,
    cross_attn: MultiheadAttention<T>,
    ffn: SwiGLU<T>,
    norm1: LayerNorm<T>,
    norm2: LayerNorm<T>,
    norm3: LayerNorm<T>,
    dropout: Dropout<T>,
    training: bool,
}

impl<T: Float> TransformerDecoderLayer<T> {
    /// Create a new transformer decoder layer.
    ///
    /// # Arguments
    ///
    /// - `d_model` - The model dimension (embedding size).
    /// - `num_heads` - Number of attention heads.
    /// - `d_ff` - Hidden dimension of the SwiGLU feedforward network.
    /// - `dropout_p` - Dropout probability.
    /// - `layer_norm_eps` - Epsilon for layer normalization.
    /// - `bias` - Whether to use bias in attention and FFN projections.
    pub fn new(
        d_model: usize,
        num_heads: usize,
        d_ff: usize,
        dropout_p: f64,
        layer_norm_eps: f64,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        let self_attn = MultiheadAttention::new(d_model, num_heads, bias)?;
        let cross_attn = MultiheadAttention::new(d_model, num_heads, bias)?;
        let ffn = SwiGLU::new(d_model, d_ff, bias)?;
        let norm1 = LayerNorm::new(vec![d_model], layer_norm_eps, true)?;
        let norm2 = LayerNorm::new(vec![d_model], layer_norm_eps, true)?;
        let norm3 = LayerNorm::new(vec![d_model], layer_norm_eps, true)?;
        let dropout = Dropout::new(dropout_p)?;

        Ok(Self {
            self_attn,
            cross_attn,
            ffn,
            norm1,
            norm2,
            norm3,
            dropout,
            training: true,
        })
    }

    /// Forward pass.
    ///
    /// # Arguments
    ///
    /// - `input` - Decoder input: `[batch, tgt_seq, d_model]`.
    /// - `memory` - Encoder output: `[batch, src_seq, d_model]`.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, tgt_seq, d_model]`.
    pub fn forward_with_memory(
        &self,
        input: &Tensor<T>,
        memory: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 3 || memory.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "TransformerDecoderLayer expects 3-D inputs, \
                     got input {:?}, memory {:?}",
                    input.shape(),
                    memory.shape()
                ),
            });
        }

        // Pre-norm causal self-attention.
        let normed1 = self.norm1.forward(input)?;
        let self_attn_out =
            self.self_attn
                .forward_qkv(&normed1, &normed1, &normed1, true)?;
        let self_attn_out = self.dropout.forward(&self_attn_out)?;
        let residual1 = add(input, &self_attn_out)?;

        // Pre-norm cross-attention.
        let normed2 = self.norm2.forward(&residual1)?;
        let cross_attn_out =
            self.cross_attn
                .forward_qkv(&normed2, memory, memory, false)?;
        let cross_attn_out = self.dropout.forward(&cross_attn_out)?;
        let residual2 = add(&residual1, &cross_attn_out)?;

        // Pre-norm feedforward.
        let normed3 = self.norm3.forward(&residual2)?;
        let ffn_out = self.ffn.forward(&normed3)?;
        let ffn_out = self.dropout.forward(&ffn_out)?;
        let residual3 = add(&residual2, &ffn_out)?;

        Ok(residual3)
    }
}

impl<T: Float> Module<T> for TransformerDecoderLayer<T> {
    /// Forward pass using `input` as both decoder input and memory.
    ///
    /// For the typical decoder use case with separate encoder output, call
    /// [`forward_with_memory`](Self::forward_with_memory) directly.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_with_memory(input, input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters());
        params.extend(self.cross_attn.parameters());
        params.extend(self.ffn.parameters());
        params.extend(self.norm1.parameters());
        params.extend(self.norm2.parameters());
        params.extend(self.norm3.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.self_attn.parameters_mut());
        params.extend(self.cross_attn.parameters_mut());
        params.extend(self.ffn.parameters_mut());
        params.extend(self.norm1.parameters_mut());
        params.extend(self.norm2.parameters_mut());
        params.extend(self.norm3.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, param) in self.self_attn.named_parameters() {
            params.push((format!("self_attn.{name}"), param));
        }
        for (name, param) in self.cross_attn.named_parameters() {
            params.push((format!("cross_attn.{name}"), param));
        }
        for (name, param) in self.ffn.named_parameters() {
            params.push((format!("ffn.{name}"), param));
        }
        for (name, param) in self.norm1.named_parameters() {
            params.push((format!("norm1.{name}"), param));
        }
        for (name, param) in self.norm2.named_parameters() {
            params.push((format!("norm2.{name}"), param));
        }
        for (name, param) in self.norm3.named_parameters() {
            params.push((format!("norm3.{name}"), param));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.self_attn.train();
        self.cross_attn.train();
        self.ffn.train();
        self.norm1.train();
        self.norm2.train();
        self.norm3.train();
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.self_attn.eval();
        self.cross_attn.eval();
        self.ffn.eval();
        self.norm1.eval();
        self.norm2.eval();
        self.norm3.eval();
        self.dropout.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // RoPE
    // -----------------------------------------------------------------------

    #[test]
    fn test_rope_construction() {
        let rope = RotaryPositionEmbedding::<f32>::new(64, 512, 10000.0);
        assert!(rope.is_ok());
        let rope = rope.unwrap();
        assert_eq!(rope.dim(), 64);
        assert_eq!(rope.max_seq_len(), 512);
        assert_eq!(rope.base(), 10000.0);
    }

    #[test]
    fn test_rope_odd_dim_rejected() {
        assert!(RotaryPositionEmbedding::<f32>::new(63, 512, 10000.0).is_err());
    }

    #[test]
    fn test_rope_zero_dim_rejected() {
        assert!(RotaryPositionEmbedding::<f32>::new(0, 512, 10000.0).is_err());
    }

    #[test]
    fn test_rope_zero_seq_rejected() {
        assert!(RotaryPositionEmbedding::<f32>::new(64, 0, 10000.0).is_err());
    }

    #[test]
    fn test_rope_output_shape_2d() {
        let rope = RotaryPositionEmbedding::<f32>::new(8, 128, 10000.0).unwrap();
        // Input: [seq_len=4, dim=8]
        let x = ferrotorch_core::zeros::<f32>(&[4, 8]).unwrap();
        let y = rope.apply(&x, 0).unwrap();
        assert_eq!(y.shape(), &[4, 8]);
    }

    #[test]
    fn test_rope_output_shape_3d() {
        let rope = RotaryPositionEmbedding::<f32>::new(16, 256, 10000.0).unwrap();
        // Input: [batch=2, seq_len=10, dim=16]
        let x = ferrotorch_core::zeros::<f32>(&[2, 10, 16]).unwrap();
        let y = rope.apply(&x, 0).unwrap();
        assert_eq!(y.shape(), &[2, 10, 16]);
    }

    #[test]
    fn test_rope_output_shape_4d() {
        let rope = RotaryPositionEmbedding::<f32>::new(8, 128, 10000.0).unwrap();
        // Input: [batch=2, heads=4, seq=6, head_dim=8]
        let x = ferrotorch_core::zeros::<f32>(&[2, 4, 6, 8]).unwrap();
        let y = rope.apply(&x, 0).unwrap();
        assert_eq!(y.shape(), &[2, 4, 6, 8]);
    }

    #[test]
    fn test_rope_with_offset() {
        let rope = RotaryPositionEmbedding::<f32>::new(8, 128, 10000.0).unwrap();
        let x = ferrotorch_core::ones::<f32>(&[4, 8]).unwrap();
        // Offset 10, seq_len 4 -> positions 10..14, fine since 14 <= 128.
        let y = rope.apply(&x, 10).unwrap();
        assert_eq!(y.shape(), &[4, 8]);
    }

    #[test]
    fn test_rope_offset_overflow_rejected() {
        let rope = RotaryPositionEmbedding::<f32>::new(8, 16, 10000.0).unwrap();
        // seq_len=10, offset=10 -> 10+10=20 > 16 -> error.
        let x = ferrotorch_core::zeros::<f32>(&[10, 8]).unwrap();
        assert!(rope.apply(&x, 10).is_err());
    }

    #[test]
    fn test_rope_position_zero_is_identity() {
        // At position 0, cos(0) = 1, sin(0) = 0, so rotation is identity.
        let rope = RotaryPositionEmbedding::<f64>::new(4, 64, 10000.0).unwrap();
        let x = ferrotorch_core::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let y = rope.apply(&x, 0).unwrap();
        let y_data = y.data().unwrap();
        let x_data = x.data().unwrap();
        for (i, (&xv, &yv)) in x_data.iter().zip(y_data.iter()).enumerate() {
            assert!(
                (xv - yv).abs() < 1e-10,
                "position 0 should be identity, index {i}: x={xv}, y={yv}"
            );
        }
    }

    #[test]
    fn test_rope_values_are_finite() {
        let rope = RotaryPositionEmbedding::<f32>::new(16, 512, 10000.0).unwrap();
        let x = ferrotorch_core::ones::<f32>(&[2, 4, 10, 16]).unwrap();
        let y = rope.apply(&x, 0).unwrap();
        for &v in y.data().unwrap() {
            assert!(v.is_finite(), "RoPE produced non-finite value: {v}");
        }
    }

    #[test]
    fn test_rope_wrong_dim_rejected() {
        let rope = RotaryPositionEmbedding::<f32>::new(8, 128, 10000.0).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[4, 10]).unwrap(); // dim=10 != 8
        assert!(rope.apply(&x, 0).is_err());
    }

    // -----------------------------------------------------------------------
    // RoPE — HalfRotation convention
    // -----------------------------------------------------------------------

    #[test]
    fn test_rope_half_rotation_construction() {
        let rope = RotaryPositionEmbedding::<f32>::with_convention(
            8, 128, 10000.0, RoPEConvention::HalfRotation,
        ).unwrap();
        assert_eq!(rope.convention(), RoPEConvention::HalfRotation);
    }

    #[test]
    fn test_rope_half_rotation_output_shape() {
        let rope = RotaryPositionEmbedding::<f32>::with_convention(
            8, 128, 10000.0, RoPEConvention::HalfRotation,
        ).unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[2, 4, 8]).unwrap();
        let y = rope.apply(&x, 0).unwrap();
        assert_eq!(y.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_rope_half_rotation_position_zero_is_identity() {
        // At position 0, cos(0)=1, sin(0)=0 → identity regardless of convention.
        let rope = RotaryPositionEmbedding::<f64>::with_convention(
            4, 64, 10000.0, RoPEConvention::HalfRotation,
        ).unwrap();
        let x = ferrotorch_core::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let y = rope.apply(&x, 0).unwrap();
        let x_data = x.data().unwrap();
        let y_data = y.data().unwrap();
        for (i, (&xv, &yv)) in x_data.iter().zip(y_data.iter()).enumerate() {
            assert!(
                (xv - yv).abs() < 1e-10,
                "half-rot pos 0 should be identity, index {i}: x={xv}, y={yv}"
            );
        }
    }

    #[test]
    fn test_rope_half_rotation_correctness() {
        // dim=4, so half_dim=2. For half-rotation:
        //   x_rot[0] = x[0]*cos0 - x[2]*sin0
        //   x_rot[1] = x[1]*cos1 - x[3]*sin1
        //   x_rot[2] = x[0]*sin0 + x[2]*cos0
        //   x_rot[3] = x[1]*sin1 + x[3]*cos1
        let rope = RotaryPositionEmbedding::<f64>::with_convention(
            4, 64, 10000.0, RoPEConvention::HalfRotation,
        ).unwrap();

        // Use position 1 so sin != 0.
        let x = ferrotorch_core::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let y = rope.apply(&x, 1).unwrap();

        // Get the cached cos/sin at position 1.
        let cos_data = rope.cos_cache.data().unwrap();
        let sin_data = rope.sin_cache.data().unwrap();
        // Position 1 → row offset = 1 * half_dim = 2
        let c0 = cos_data[2]; let c1 = cos_data[3];
        let s0 = sin_data[2]; let s1 = sin_data[3];

        let expected = [
            1.0 * c0 - 3.0 * s0,
            2.0 * c1 - 4.0 * s1,
            1.0 * s0 + 3.0 * c0,
            2.0 * s1 + 4.0 * c1,
        ];

        let y_data = y.data().unwrap();
        for (i, (&actual, &exp)) in y_data.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-10,
                "half-rot index {i}: actual={actual}, expected={exp}"
            );
        }
    }

    #[test]
    fn test_rope_interleaved_vs_half_rotation_differ() {
        // Same input at position > 0 should produce different outputs.
        let rope_il = RotaryPositionEmbedding::<f64>::with_convention(
            4, 64, 10000.0, RoPEConvention::Interleaved,
        ).unwrap();
        let rope_hr = RotaryPositionEmbedding::<f64>::with_convention(
            4, 64, 10000.0, RoPEConvention::HalfRotation,
        ).unwrap();

        let x = ferrotorch_core::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let y_il = rope_il.apply(&x, 1).unwrap();
        let y_hr = rope_hr.apply(&x, 1).unwrap();

        // They should differ (different pairing).
        let il_data = y_il.data().unwrap();
        let hr_data = y_hr.data().unwrap();
        let any_differ = il_data.iter().zip(hr_data.iter()).any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(any_differ, "interleaved and half-rotation should produce different outputs at pos > 0");
    }

    #[test]
    fn test_rope_default_convention_is_interleaved() {
        let rope = RotaryPositionEmbedding::<f32>::new(8, 128, 10000.0).unwrap();
        assert_eq!(rope.convention(), RoPEConvention::Interleaved);
    }

    // -----------------------------------------------------------------------
    // SwiGLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_swiglu_construction() {
        let swiglu = SwiGLU::<f32>::new(64, 128, true);
        assert!(swiglu.is_ok());
    }

    #[test]
    fn test_swiglu_forward_shape_2d() {
        let swiglu = SwiGLU::<f32>::new(16, 32, true).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[4, 16]).unwrap();
        let output = swiglu.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 16]);
    }

    #[test]
    fn test_swiglu_forward_shape_3d() {
        let swiglu = SwiGLU::<f32>::new(16, 32, false).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 16]).unwrap();
        let output = swiglu.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_swiglu_forward_values_finite() {
        let swiglu = SwiGLU::<f32>::new(8, 16, true).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[2, 3, 8]).unwrap();
        let output = swiglu.forward(&input).unwrap();
        for &v in output.data().unwrap() {
            assert!(v.is_finite(), "SwiGLU produced non-finite value: {v}");
        }
    }

    #[test]
    fn test_swiglu_1d_rejected() {
        let swiglu = SwiGLU::<f32>::new(8, 16, false).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[8]).unwrap();
        assert!(swiglu.forward(&input).is_err());
    }

    #[test]
    fn test_swiglu_parameters() {
        let swiglu = SwiGLU::<f32>::new(8, 16, true).unwrap();
        let params = swiglu.parameters();
        // w1: weight + bias, w2: weight + bias, w3: weight + bias = 6
        assert_eq!(params.len(), 6);

        let named = swiglu.named_parameters();
        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"w1.weight"));
        assert!(names.contains(&"w1.bias"));
        assert!(names.contains(&"w2.weight"));
        assert!(names.contains(&"w2.bias"));
        assert!(names.contains(&"w3.weight"));
        assert!(names.contains(&"w3.bias"));
    }

    #[test]
    fn test_swiglu_parameters_no_bias() {
        let swiglu = SwiGLU::<f32>::new(8, 16, false).unwrap();
        let params = swiglu.parameters();
        // w1: weight, w2: weight, w3: weight = 3
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn test_swiglu_train_eval() {
        let mut swiglu = SwiGLU::<f32>::new(8, 16, false).unwrap();
        assert!(swiglu.is_training());
        swiglu.eval();
        assert!(!swiglu.is_training());
        swiglu.train();
        assert!(swiglu.is_training());
    }

    // -----------------------------------------------------------------------
    // KVCache
    // -----------------------------------------------------------------------

    #[test]
    fn test_kv_cache_new_empty() {
        let cache = KVCache::<f32>::new(1024);
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
        assert_eq!(cache.max_seq_len(), 1024);
    }

    #[test]
    fn test_kv_cache_single_update() {
        let mut cache = KVCache::<f32>::new(128);
        // [B=1, heads=2, seq=3, dim=4]
        let k = ferrotorch_core::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        let v = ferrotorch_core::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        let (fk, fv) = cache.update(k, v).unwrap();
        assert_eq!(fk.shape(), &[1, 2, 3, 4]);
        assert_eq!(fv.shape(), &[1, 2, 3, 4]);
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_kv_cache_append() {
        let mut cache = KVCache::<f32>::new(128);
        // First: [1, 2, 3, 4]
        let k1 = ferrotorch_core::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        let v1 = ferrotorch_core::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        cache.update(k1, v1).unwrap();
        assert_eq!(cache.seq_len(), 3);

        // Append: [1, 2, 2, 4]
        let k2 = ferrotorch_core::ones::<f32>(&[1, 2, 2, 4]).unwrap();
        let v2 = ferrotorch_core::ones::<f32>(&[1, 2, 2, 4]).unwrap();
        let (fk, fv) = cache.update(k2, v2).unwrap();
        assert_eq!(fk.shape(), &[1, 2, 5, 4]); // 3 + 2 = 5
        assert_eq!(fv.shape(), &[1, 2, 5, 4]);
        assert_eq!(cache.seq_len(), 5);
    }

    #[test]
    fn test_kv_cache_reset() {
        let mut cache = KVCache::<f32>::new(128);
        let k = ferrotorch_core::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        let v = ferrotorch_core::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        cache.update(k, v).unwrap();
        assert_eq!(cache.seq_len(), 3);

        cache.reset();
        assert!(cache.is_empty());
        assert_eq!(cache.seq_len(), 0);
    }

    #[test]
    fn test_kv_cache_overflow_rejected() {
        let mut cache = KVCache::<f32>::new(4);
        let k = ferrotorch_core::ones::<f32>(&[1, 1, 5, 2]).unwrap();
        let v = ferrotorch_core::ones::<f32>(&[1, 1, 5, 2]).unwrap();
        // seq=5 > max_seq_len=4 -> error.
        assert!(cache.update(k, v).is_err());
    }

    #[test]
    fn test_kv_cache_shape_mismatch_rejected() {
        let mut cache = KVCache::<f32>::new(128);
        let k = ferrotorch_core::ones::<f32>(&[1, 2, 3, 4]).unwrap();
        let v = ferrotorch_core::ones::<f32>(&[1, 2, 3, 8]).unwrap(); // dim mismatch
        assert!(cache.update(k, v).is_err());
    }

    #[test]
    fn test_kv_cache_values_preserved() {
        let mut cache = KVCache::<f64>::new(128);
        // First update: all 1s.
        let k1 = ferrotorch_core::ones::<f64>(&[1, 1, 2, 3]).unwrap();
        let v1 = ferrotorch_core::ones::<f64>(&[1, 1, 2, 3]).unwrap();
        cache.update(k1, v1).unwrap();

        // Second update: all 2s.
        let k2_data = vec![2.0f64; 1 * 1 * 1 * 3];
        let k2 = ferrotorch_core::from_slice(&k2_data, &[1, 1, 1, 3]).unwrap();
        let v2 = ferrotorch_core::from_slice(&k2_data, &[1, 1, 1, 3]).unwrap();
        let (fk, _fv) = cache.update(k2, v2).unwrap();

        assert_eq!(fk.shape(), &[1, 1, 3, 3]); // 2 + 1 = 3
        let fk_data = fk.data().unwrap();
        // First 2 rows should be 1.0, last row should be 2.0.
        for &v in &fk_data[..6] {
            assert!((v - 1.0).abs() < 1e-10, "expected 1.0, got {v}");
        }
        for &v in &fk_data[6..9] {
            assert!((v - 2.0).abs() < 1e-10, "expected 2.0, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // TransformerEncoderLayer
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoder_layer_construction() {
        let layer = TransformerEncoderLayer::<f32>::new(16, 4, 32, 0.0, 1e-5, true);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_encoder_layer_forward_shape() {
        let layer = TransformerEncoderLayer::<f32>::new(16, 4, 32, 0.0, 1e-5, false).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 16]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_encoder_layer_forward_values_finite() {
        let layer = TransformerEncoderLayer::<f32>::new(8, 2, 16, 0.0, 1e-5, true).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[1, 3, 8]).unwrap();
        let output = layer.forward(&input).unwrap();
        for &v in output.data().unwrap() {
            assert!(
                v.is_finite(),
                "TransformerEncoderLayer produced non-finite value: {v}"
            );
        }
    }

    #[test]
    fn test_encoder_layer_2d_rejected() {
        let layer = TransformerEncoderLayer::<f32>::new(8, 2, 16, 0.0, 1e-5, false).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[4, 8]).unwrap();
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_encoder_layer_parameters_count() {
        let layer = TransformerEncoderLayer::<f32>::new(8, 2, 16, 0.0, 1e-5, true).unwrap();
        let params = layer.parameters();
        // self_attn: 4 weights + 4 biases = 8
        // ffn (SwiGLU): 3 weights + 3 biases = 6
        // norm1: weight + bias = 2
        // norm2: weight + bias = 2
        // Total: 18
        assert_eq!(params.len(), 18);
    }

    #[test]
    fn test_encoder_layer_train_eval() {
        let mut layer =
            TransformerEncoderLayer::<f32>::new(8, 2, 16, 0.1, 1e-5, false).unwrap();
        assert!(layer.is_training());
        layer.eval();
        assert!(!layer.is_training());
        layer.train();
        assert!(layer.is_training());
    }

    #[test]
    fn test_encoder_layer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TransformerEncoderLayer<f32>>();
        assert_send_sync::<TransformerEncoderLayer<f64>>();
    }

    // -----------------------------------------------------------------------
    // TransformerDecoderLayer
    // -----------------------------------------------------------------------

    #[test]
    fn test_decoder_layer_construction() {
        let layer = TransformerDecoderLayer::<f32>::new(16, 4, 32, 0.0, 1e-5, true);
        assert!(layer.is_ok());
    }

    #[test]
    fn test_decoder_layer_forward_shape() {
        let layer = TransformerDecoderLayer::<f32>::new(16, 4, 32, 0.0, 1e-5, false).unwrap();
        // decoder input: [2, 4, 16], encoder memory: [2, 6, 16]
        let tgt = ferrotorch_core::zeros::<f32>(&[2, 4, 16]).unwrap();
        let memory = ferrotorch_core::zeros::<f32>(&[2, 6, 16]).unwrap();
        let output = layer.forward_with_memory(&tgt, &memory).unwrap();
        assert_eq!(output.shape(), &[2, 4, 16]);
    }

    #[test]
    fn test_decoder_layer_self_forward_shape() {
        // Module::forward uses input as both decoder input and memory.
        let layer = TransformerDecoderLayer::<f32>::new(8, 2, 16, 0.0, 1e-5, true).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 3, 8]).unwrap();
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_decoder_layer_forward_values_finite() {
        let layer = TransformerDecoderLayer::<f32>::new(8, 2, 16, 0.0, 1e-5, true).unwrap();
        let tgt = ferrotorch_core::ones::<f32>(&[1, 3, 8]).unwrap();
        let mem = ferrotorch_core::ones::<f32>(&[1, 5, 8]).unwrap();
        let output = layer.forward_with_memory(&tgt, &mem).unwrap();
        for &v in output.data().unwrap() {
            assert!(
                v.is_finite(),
                "TransformerDecoderLayer produced non-finite value: {v}"
            );
        }
    }

    #[test]
    fn test_decoder_layer_2d_rejected() {
        let layer = TransformerDecoderLayer::<f32>::new(8, 2, 16, 0.0, 1e-5, false).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[4, 8]).unwrap();
        let memory = ferrotorch_core::zeros::<f32>(&[4, 8]).unwrap();
        assert!(layer.forward_with_memory(&input, &memory).is_err());
    }

    #[test]
    fn test_decoder_layer_parameters_count() {
        let layer = TransformerDecoderLayer::<f32>::new(8, 2, 16, 0.0, 1e-5, true).unwrap();
        let params = layer.parameters();
        // self_attn: 4 weights + 4 biases = 8
        // cross_attn: 4 weights + 4 biases = 8
        // ffn (SwiGLU): 3 weights + 3 biases = 6
        // norm1: weight + bias = 2
        // norm2: weight + bias = 2
        // norm3: weight + bias = 2
        // Total: 28
        assert_eq!(params.len(), 28);
    }

    #[test]
    fn test_decoder_layer_train_eval() {
        let mut layer =
            TransformerDecoderLayer::<f32>::new(8, 2, 16, 0.1, 1e-5, false).unwrap();
        assert!(layer.is_training());
        layer.eval();
        assert!(!layer.is_training());
        layer.train();
        assert!(layer.is_training());
    }

    #[test]
    fn test_decoder_layer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TransformerDecoderLayer<f32>>();
        assert_send_sync::<TransformerDecoderLayer<f64>>();
    }
}
