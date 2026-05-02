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

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::grad_fns::activation::silu;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

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

// ---------------------------------------------------------------------------
// RoPEBackward
// ---------------------------------------------------------------------------

/// Backward node for RoPE.
///
/// RoPE applies a linear rotation per position, so the backward pass
/// applies the *inverse* rotation (transpose of the rotation matrix,
/// which is just cos / -sin swap).
#[derive(Debug)]
struct RoPEBackward<T: Float> {
    input: Tensor<T>,
    cos_flat: Vec<T>,
    sin_flat: Vec<T>,
    half_dim: usize,
    seq_len: usize,
    batch_dims: usize,
    dim: usize,
    seq_offset: usize,
    convention: RoPEConvention,
}

impl<T: Float> GradFn<T> for RoPEBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let go_data = grad_output.data_vec()?;
            let total = go_data.len();
            let mut grad_input = Vec::with_capacity(total);

            match self.convention {
                RoPEConvention::Interleaved => {
                    for b in 0..self.batch_dims {
                        for s in 0..self.seq_len {
                            let cache_start = (self.seq_offset + s) * self.half_dim;
                            let go_start = b * self.seq_len * self.dim + s * self.dim;

                            for i in 0..self.half_dim {
                                let go_even = go_data[go_start + 2 * i];
                                let go_odd = go_data[go_start + 2 * i + 1];
                                let cos_val = self.cos_flat[cache_start + i];
                                let sin_val = self.sin_flat[cache_start + i];

                                // Inverse rotation: R^T * grad
                                grad_input.push(go_even * cos_val + go_odd * sin_val);
                                grad_input.push(-go_even * sin_val + go_odd * cos_val);
                            }
                        }
                    }
                }
                RoPEConvention::HalfRotation => {
                    for b in 0..self.batch_dims {
                        for s in 0..self.seq_len {
                            let cache_start = (self.seq_offset + s) * self.half_dim;
                            let go_start = b * self.seq_len * self.dim + s * self.dim;

                            // First half of grad_input: dx[i] = go_first[i]*cos + go_second[i]*sin
                            for i in 0..self.half_dim {
                                let go_first = go_data[go_start + i];
                                let go_second = go_data[go_start + self.half_dim + i];
                                let cos_val = self.cos_flat[cache_start + i];
                                let sin_val = self.sin_flat[cache_start + i];

                                grad_input.push(go_first * cos_val + go_second * sin_val);
                            }
                            // Second half: dx[i+d/2] = -go_first[i]*sin + go_second[i]*cos
                            for i in 0..self.half_dim {
                                let go_first = go_data[go_start + i];
                                let go_second = go_data[go_start + self.half_dim + i];
                                let cos_val = self.cos_flat[cache_start + i];
                                let sin_val = self.sin_flat[cache_start + i];

                                grad_input.push(-go_first * sin_val + go_second * cos_val);
                            }
                        }
                    }
                }
            }

            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_input),
                self.input.shape().to_vec(),
                false,
            )?;
            Some(if self.input.is_cuda() {
                g.to(self.input.device())?
            } else {
                g
            })
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "RoPEBackward"
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
    scaling: RoPEScaling,
    /// Precomputed cosines: `[max_seq_len, dim/2]`.
    cos_cache: Tensor<T>,
    /// Precomputed sines: `[max_seq_len, dim/2]`.
    sin_cache: Tensor<T>,
}

/// Frequency-scaling strategy for [`RotaryPositionEmbedding`].
///
/// Determines how `inv_freq[i] = 1 / base^(2i / dim)` is modified to
/// support context lengths beyond the model's training distribution.
/// Default is [`RoPEScaling::None`] (classical RoPE with no modification).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub enum RoPEScaling {
    /// No scaling. `inv_freq[i] = 1 / base^(2i / dim)`. Matches the
    /// original RoFormer formulation and the Llama 3 8B base model
    /// (8k context, no stretching).
    #[default]
    None,

    /// Linear / positional interpolation (Chen et al. 2023,
    /// kaiokendev's "linear RoPE scaling"). Divides the frequencies
    /// uniformly by `factor`, so a position of `p` tokens rotates at
    /// the same angle as position `p / factor` under the unscaled
    /// schedule. Extends context by `factor` at the cost of short-text
    /// quality.
    Linear {
        /// Target-context / original-context ratio. >= 1.
        factor: f64,
    },

    /// NTK-aware scaling (bloc97's "NTK-Aware Scaled RoPE"). Scales the
    /// base itself via `base_new = base * factor^(dim / (dim - 2))`.
    /// Preserves high-frequency components (short-range token ordering)
    /// while stretching low-frequency components (long-range position).
    /// Default choice for short-extension long-context Llama variants
    /// that don't want short-text degradation.
    NtkAware {
        /// Target-context / original-context ratio. >= 1.
        factor: f64,
        /// Original maximum context length the model was trained on.
        /// Reserved for future use — current NTK formulation only
        /// depends on `factor` and `dim`. Held on the struct so the
        /// config the model ships with is fully captured.
        original_max_pos_embeddings: usize,
    },

    /// YARN scaling (Peng et al. 2023, "YaRN: Efficient Context Window
    /// Extension of Large Language Models"). Per-dimension piecewise
    /// mix between PI (linear interpolation) for low-frequency dims
    /// and extrapolation (no scaling) for high-frequency dims, with a
    /// linear ramp in between. Generally the best-quality long-context
    /// scaling; used by Mistral 7B v0.2, CodeLlama, and Yi.
    Yarn {
        /// Target-context / original-context ratio. >= 1.
        factor: f64,
        /// Original maximum context length the model was trained on.
        original_max_pos_embeddings: usize,
        /// Number of rotations above which a dimension extrapolates
        /// (no interpolation). Default in the paper is 32.
        beta_fast: f64,
        /// Number of rotations below which a dimension fully
        /// interpolates (linear PI). Default in the paper is 1.
        beta_slow: f64,
    },
}

impl RoPEScaling {
    /// Convenience: YARN with the paper's default beta_fast=32, beta_slow=1.
    pub const fn yarn_default(factor: f64, original_max_pos_embeddings: usize) -> Self {
        RoPEScaling::Yarn {
            factor,
            original_max_pos_embeddings,
            beta_fast: 32.0,
            beta_slow: 1.0,
        }
    }
}

/// YARN helper: for a given `num_rotations` target, solve for the
/// dimension index at which a base-schedule RoPE produces that many
/// rotations across `original_max_pos_embeddings` positions.
fn yarn_find_correction_dim(
    num_rotations: f64,
    dim: usize,
    base: f64,
    original_max_pos_embeddings: usize,
) -> f64 {
    // dim * ln(L / (num_rotations * 2pi)) / (2 ln(base))
    (dim as f64
        * (original_max_pos_embeddings as f64 / (num_rotations * 2.0 * std::f64::consts::PI)).ln())
        / (2.0 * base.ln())
}

/// YARN helper: bracket the correction range to valid indices.
fn yarn_find_correction_range(
    low_rot: f64,
    high_rot: f64,
    dim: usize,
    base: f64,
    original_max_pos_embeddings: usize,
) -> (f64, f64) {
    let low = yarn_find_correction_dim(low_rot, dim, base, original_max_pos_embeddings).floor();
    let high = yarn_find_correction_dim(high_rot, dim, base, original_max_pos_embeddings).ceil();
    (low.max(0.0), high.min((dim - 1) as f64))
}

/// Compute inv_freq[i] = 1 / base^(2i / dim) for i in 0..dim/2.
fn compute_base_inv_freq(dim: usize, base: f64) -> Vec<f64> {
    let half = dim / 2;
    (0..half)
        .map(|i| 1.0 / base.powf(2.0 * i as f64 / dim as f64))
        .collect()
}

/// Compute the per-dim inv_freq vector under the given scaling policy.
///
/// Exposed at `pub(crate)` so unit tests can verify the math directly
/// without round-tripping through the precomputed cos/sin caches.
pub(crate) fn compute_scaled_inv_freq(dim: usize, base: f64, scaling: RoPEScaling) -> Vec<f64> {
    match scaling {
        RoPEScaling::None => compute_base_inv_freq(dim, base),

        RoPEScaling::Linear { factor } => {
            let mut iv = compute_base_inv_freq(dim, base);
            for v in iv.iter_mut() {
                *v /= factor;
            }
            iv
        }

        RoPEScaling::NtkAware { factor, .. } => {
            // NTK-Aware: base' = base * factor^(dim / (dim - 2)).
            // Exponent is chosen so the *highest* frequency (i = 0) is
            // unchanged while the *lowest* frequency (i = dim/2 - 1)
            // is scaled by ~1/factor, matching linear PI at the long end.
            let exp = dim as f64 / (dim as f64 - 2.0);
            let base_scaled = base * factor.powf(exp);
            compute_base_inv_freq(dim, base_scaled)
        }

        RoPEScaling::Yarn {
            factor,
            original_max_pos_embeddings,
            beta_fast,
            beta_slow,
        } => {
            let half = dim / 2;
            let pos_freqs: Vec<f64> = (0..half)
                .map(|i| base.powf(2.0 * i as f64 / dim as f64))
                .collect();
            let extrapolation: Vec<f64> = pos_freqs.iter().map(|p| 1.0 / p).collect();
            let interpolation: Vec<f64> = pos_freqs.iter().map(|p| 1.0 / (factor * p)).collect();

            let (low, high) = yarn_find_correction_range(
                beta_fast,
                beta_slow,
                dim,
                base,
                original_max_pos_embeddings,
            );
            // Map the full-dim correction range onto the half-dim inv_freq
            // index space.
            let (low, high) = (low / 2.0, high / 2.0);

            // ramp_mask[i] is 1.0 at i <= low (full extrapolation),
            // 0.0 at i >= high (full interpolation), linear ramp
            // between. Paper uses "1 - linear_ramp(low, high, dim/2)"
            // to get this shape.
            let denom = if high == low { 0.001 } else { high - low };
            (0..half)
                .map(|i| {
                    let t = ((i as f64 - low) / denom).clamp(0.0, 1.0);
                    // Invert: high-freq (small i) keeps extrapolation.
                    let mask = 1.0 - t;
                    interpolation[i] * (1.0 - mask) + extrapolation[i] * mask
                })
                .collect()
        }
    }
}

impl<T: Float> RotaryPositionEmbedding<T> {
    /// Create a new RoPE instance with the default interleaved convention
    /// and no frequency scaling.
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
        Self::with_scaling(
            dim,
            max_seq_len,
            base,
            RoPEConvention::default(),
            RoPEScaling::None,
        )
    }

    /// Create a new RoPE instance with a specified pairing convention
    /// and no frequency scaling.
    ///
    /// Use [`RoPEConvention::HalfRotation`] for LLaMA, GPT-NeoX, and Pythia
    /// compatibility.
    pub fn with_convention(
        dim: usize,
        max_seq_len: usize,
        base: f64,
        convention: RoPEConvention,
    ) -> FerrotorchResult<Self> {
        Self::with_scaling(dim, max_seq_len, base, convention, RoPEScaling::None)
    }

    /// Create a new RoPE instance with explicit convention and scaling.
    ///
    /// `scaling` selects between no scaling, linear PI, NTK-aware, and
    /// YARN extension strategies. See [`RoPEScaling`] for the per-variant
    /// semantics; [`RoPEScaling::None`] reproduces classical RoPE.
    pub fn with_scaling(
        dim: usize,
        max_seq_len: usize,
        base: f64,
        convention: RoPEConvention,
        scaling: RoPEScaling,
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
        if let RoPEScaling::Linear { factor }
        | RoPEScaling::NtkAware { factor, .. }
        | RoPEScaling::Yarn { factor, .. } = scaling
        {
            if !(factor.is_finite() && factor > 0.0) {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("RoPE scaling factor must be finite and > 0, got {factor}"),
                });
            }
        }

        let half_dim = dim / 2;
        let thetas = compute_scaled_inv_freq(dim, base, scaling);

        // cos_cache[pos, i] = cos(pos * theta_i)
        // sin_cache[pos, i] = sin(pos * theta_i)
        let total = max_seq_len * half_dim;
        let mut cos_data = Vec::with_capacity(total);
        let mut sin_data = Vec::with_capacity(total);

        for pos in 0..max_seq_len {
            for &theta in &thetas {
                let angle = pos as f64 * theta;
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
            scaling,
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
                message: format!("RoPE: last dim of input ({last_dim}) != dim ({})", self.dim),
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

        let device = x.device();
        let half_dim = self.dim / 2;
        let cos_data = self.cos_cache.data_vec()?;
        let sin_data = self.sin_cache.data_vec()?;
        let x_data = x.data_vec()?;

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

        let result = if is_grad_enabled() && x.requires_grad() {
            Tensor::from_operation(
                TensorStorage::cpu(output),
                shape.to_vec(),
                Arc::new(RoPEBackward {
                    input: x.clone(),
                    cos_flat: cos_data,
                    sin_flat: sin_data,
                    half_dim,
                    seq_len,
                    batch_dims,
                    dim: self.dim,
                    seq_offset,
                    convention: self.convention,
                }),
            )?
        } else {
            Tensor::from_storage(TensorStorage::cpu(output), shape.to_vec(), false)?
        };
        if device.is_cuda() {
            result.to(device)
        } else {
            Ok(result)
        }
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

    /// The frequency-scaling strategy this RoPE instance was built with.
    #[inline]
    pub fn scaling(&self) -> RoPEScaling {
        self.scaling
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
    pub fn new(in_features: usize, hidden_features: usize, bias: bool) -> FerrotorchResult<Self> {
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
    /// reshapes back. Uses differentiable `reshape` to preserve autograd.
    fn forward_3d(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        let batch = shape[0];
        let seq_len = shape[1];

        // Flatten to [batch * seq_len, features] — differentiable reshape.
        let flat = reshape(input, &[(batch * seq_len) as isize, -1])?;

        let output_flat = self.forward_2d(&flat)?;

        // Reshape back to [batch, seq_len, out_features] — differentiable.
        let out_features = output_flat.shape()[1];
        reshape(
            &output_flat,
            &[batch as isize, seq_len as isize, out_features as isize],
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

/// Dimensions a [`KVCache`] is pinned to after its first update (or when
/// pre-declared via [`KVCache::with_dims`]). Every subsequent update must
/// match these exactly on `batch`, `num_kv_heads`, and `head_dim`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CacheDims {
    batch: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

/// Key-value cache for efficient autoregressive (incremental) inference.
///
/// During generation, previously computed keys and values are cached so that
/// each new token only requires computing its own K and V, then concatenating
/// with the cache before attention.
///
/// # Shape convention
///
/// Keys and values are stored as `[batch, num_kv_heads, seq_len, head_dim]`.
/// The cache grows along the `seq_len` axis (dimension 2).
///
/// ## Grouped-Query Attention
///
/// For models with grouped-query attention (e.g. Llama 3 8B: 32 Q heads,
/// 8 KV heads) the cache stores at KV-head granularity — `dim 1 = num_kv_heads`,
/// not `num_q_heads`. This keeps the cache ~1/4 the size versus storing
/// at Q-head granularity. `repeat_kv` happens at *read* time, inside the
/// attention computation, not at cache time.
///
/// Pre-declare the expected shape with [`KVCache::with_dims`] to get
/// strict validation from the first update, or use [`KVCache::new`] to
/// let the dims be inferred from the first push (matching the pre-GQA
/// behaviour).
#[derive(Debug)]
pub struct KVCache<T: Float> {
    /// Cached keys: `[B, num_kv_heads, cached_seq, head_dim]`, or `None` if empty.
    key_cache: Option<Tensor<T>>,
    /// Cached values: same shape as `key_cache`.
    value_cache: Option<Tensor<T>>,
    /// Maximum sequence length the cache will hold.
    max_seq_len: usize,
    /// Pinned dimensions (batch, num_kv_heads, head_dim). `None` means the
    /// cache hasn't been populated or pre-declared yet.
    dims: Option<CacheDims>,
}

impl<T: Float> KVCache<T> {
    /// Create an empty cache with dims inferred from the first update.
    ///
    /// `max_seq_len` is the upper bound on the total cached sequence length.
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            key_cache: None,
            value_cache: None,
            max_seq_len,
            dims: None,
        }
    }

    /// Create an empty cache with pre-declared dimensions.
    ///
    /// Every subsequent [`update`](Self::update) must supply tensors with
    /// shape `[batch, num_kv_heads, _, head_dim]`. Mismatches fail on the
    /// very first update rather than silently poisoning the cache.
    ///
    /// For Llama 3 8B: `with_dims(max_seq_len, 1, 8, 128)`.
    pub fn with_dims(
        max_seq_len: usize,
        batch: usize,
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Self {
        Self {
            key_cache: None,
            value_cache: None,
            max_seq_len,
            dims: Some(CacheDims {
                batch,
                num_kv_heads,
                head_dim,
            }),
        }
    }

    /// Append new keys and values to the cache.
    ///
    /// Returns the **full** (concatenated) keys and values for use in the
    /// current attention step.
    ///
    /// # Arguments
    ///
    /// - `key` - New key tensor: `[B, num_kv_heads, new_seq, head_dim]`.
    /// - `value` - New value tensor: same shape as `key`.
    ///
    /// # Returns
    ///
    /// `(full_key, full_value)` with shape
    /// `[B, num_kv_heads, cached_seq + new_seq, head_dim]`.
    pub fn update(
        &mut self,
        key: Tensor<T>,
        value: Tensor<T>,
    ) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        if key.ndim() != 4 || value.ndim() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "KVCache expects 4-D [B, kv_heads, seq, dim] tensors, \
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

        let ks = key.shape();
        let incoming = CacheDims {
            batch: ks[0],
            num_kv_heads: ks[1],
            head_dim: ks[3],
        };

        match &self.dims {
            Some(expected) if expected != &incoming => {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "KVCache: update shape [B={}, kv_heads={}, _, dim={}] does not \
                         match pinned dims [B={}, kv_heads={}, _, dim={}]",
                        incoming.batch,
                        incoming.num_kv_heads,
                        incoming.head_dim,
                        expected.batch,
                        expected.num_kv_heads,
                        expected.head_dim,
                    ),
                });
            }
            None => self.dims = Some(incoming),
            _ => {}
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

    /// Reset the cache, discarding all stored keys and values. The pinned
    /// dimensions (if any) are preserved — a cache created via
    /// [`with_dims`](Self::with_dims) still validates the next update
    /// against the original declaration.
    pub fn reset(&mut self) {
        self.key_cache = None;
        self.value_cache = None;
    }

    /// The current cached sequence length (0 if empty).
    pub fn seq_len(&self) -> usize {
        self.key_cache.as_ref().map(|k| k.shape()[2]).unwrap_or(0)
    }

    /// Whether the cache holds any keys/values.
    pub fn is_empty(&self) -> bool {
        self.key_cache.is_none()
    }

    /// Maximum sequence length.
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Number of KV heads the cache is pinned to, once an update has
    /// happened (or pre-declared via [`with_dims`](Self::with_dims)).
    /// Returns `None` for a fresh [`new`](Self::new) cache that has not
    /// yet been updated.
    pub fn num_kv_heads(&self) -> Option<usize> {
        self.dims.map(|d| d.num_kv_heads)
    }

    /// Head dimension (`head_dim`), once pinned. See [`num_kv_heads`].
    pub fn head_dim(&self) -> Option<usize> {
        self.dims.map(|d| d.head_dim)
    }

    /// Batch size, once pinned. See [`num_kv_heads`].
    pub fn batch_size(&self) -> Option<usize> {
        self.dims.map(|d| d.batch)
    }
}

/// Concatenate two 4-D tensors along dimension 2 (the sequence axis).
///
/// Shapes must match on dims 0, 1, 3.
fn concat_along_dim2<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
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

    let device = a.device();
    let (batch, heads, seq_a, dim) = (sa[0], sa[1], sa[2], sa[3]);
    let seq_b = sb[2];
    let seq_out = seq_a + seq_b;

    let a_data = a.data_vec()?;
    let b_data = b.data_vec()?;

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

    let result = Tensor::from_storage(
        TensorStorage::cpu(output),
        vec![batch, heads, seq_out, dim],
        false,
    )?;
    if device.is_cuda() {
        result.to(device)
    } else {
        Ok(result)
    }
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
        let self_attn_out = self
            .self_attn
            .forward_qkv(&normed1, &normed1, &normed1, true)?;
        let self_attn_out = self.dropout.forward(&self_attn_out)?;
        let residual1 = add(input, &self_attn_out)?;

        // Pre-norm cross-attention.
        let normed2 = self.norm2.forward(&residual1)?;
        let cross_attn_out = self
            .cross_attn
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
// TransformerEncoder
// ===========================================================================

/// A stack of N [`TransformerEncoderLayer`] modules with an optional final
/// layer normalization.
///
/// This mirrors `torch.nn.TransformerEncoder`: it iterates the input through
/// `num_layers` identical (but independently parameterized) encoder layers,
/// then optionally applies a final `LayerNorm`.
///
/// # Shape contract
///
/// - Input: `[batch, seq_len, d_model]`
/// - Output: `[batch, seq_len, d_model]`
#[derive(Debug)]
pub struct TransformerEncoder<T: Float> {
    layers: Vec<TransformerEncoderLayer<T>>,
    norm: Option<LayerNorm<T>>,
    training: bool,
}

impl<T: Float> TransformerEncoder<T> {
    /// Create a new transformer encoder.
    ///
    /// Each layer is constructed fresh with the same hyperparameters (not
    /// cloned), so they have independent initial weights.
    ///
    /// # Arguments
    ///
    /// - `d_model` - The model dimension (embedding size).
    /// - `num_heads` - Number of attention heads.
    /// - `num_layers` - Number of encoder layers to stack.
    /// - `d_ff` - Hidden dimension of the SwiGLU feedforward network.
    /// - `dropout_p` - Dropout probability.
    /// - `layer_norm_eps` - Epsilon for layer normalization.
    /// - `bias` - Whether to use bias in attention and FFN projections.
    /// - `final_norm` - Whether to add a final `LayerNorm` after the last layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        d_ff: usize,
        dropout_p: f64,
        layer_norm_eps: f64,
        bias: bool,
        final_norm: bool,
    ) -> FerrotorchResult<Self> {
        if num_layers == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "TransformerEncoder: num_layers must be > 0".into(),
            });
        }

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerEncoderLayer::new(
                d_model,
                num_heads,
                d_ff,
                dropout_p,
                layer_norm_eps,
                bias,
            )?);
        }

        let norm = if final_norm {
            Some(LayerNorm::new(vec![d_model], layer_norm_eps, true)?)
        } else {
            None
        };

        Ok(Self {
            layers,
            norm,
            training: true,
        })
    }

    /// The number of stacked encoder layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<T: Float> Module<T> for TransformerEncoder<T> {
    /// Forward pass: iterate through all encoder layers, then apply final norm.
    ///
    /// Input shape: `[batch, seq_len, d_model]`.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward(&output)?;
        }
        if let Some(ref norm) = self.norm {
            output = norm.forward(&output)?;
        }
        Ok(output)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            let _ = i; // layer index not needed for unnamed params
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        if let Some(ref mut norm) = self.norm {
            params.extend(norm.parameters_mut());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.push((format!("layers.{i}.{name}"), param));
            }
        }
        if let Some(ref norm) = self.norm {
            for (name, param) in norm.named_parameters() {
                params.push((format!("norm.{name}"), param));
            }
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
        if let Some(ref mut norm) = self.norm {
            norm.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
        if let Some(ref mut norm) = self.norm {
            norm.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// TransformerDecoder
// ===========================================================================

/// A stack of N [`TransformerDecoderLayer`] modules with an optional final
/// layer normalization.
///
/// This mirrors `torch.nn.TransformerDecoder`: it iterates the target input
/// through `num_layers` identical (but independently parameterized) decoder
/// layers, each attending to the encoder `memory`, then optionally applies a
/// final `LayerNorm`.
///
/// # Shape contract
///
/// - `input` (decoder): `[batch, tgt_seq, d_model]`
/// - `memory` (encoder output): `[batch, src_seq, d_model]`
/// - Output: `[batch, tgt_seq, d_model]`
#[derive(Debug)]
pub struct TransformerDecoder<T: Float> {
    layers: Vec<TransformerDecoderLayer<T>>,
    norm: Option<LayerNorm<T>>,
    training: bool,
}

impl<T: Float> TransformerDecoder<T> {
    /// Create a new transformer decoder.
    ///
    /// Each layer is constructed fresh with the same hyperparameters (not
    /// cloned), so they have independent initial weights.
    ///
    /// # Arguments
    ///
    /// - `d_model` - The model dimension (embedding size).
    /// - `num_heads` - Number of attention heads.
    /// - `num_layers` - Number of decoder layers to stack.
    /// - `d_ff` - Hidden dimension of the SwiGLU feedforward network.
    /// - `dropout_p` - Dropout probability.
    /// - `layer_norm_eps` - Epsilon for layer normalization.
    /// - `bias` - Whether to use bias in attention and FFN projections.
    /// - `final_norm` - Whether to add a final `LayerNorm` after the last layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_layers: usize,
        d_ff: usize,
        dropout_p: f64,
        layer_norm_eps: f64,
        bias: bool,
        final_norm: bool,
    ) -> FerrotorchResult<Self> {
        if num_layers == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "TransformerDecoder: num_layers must be > 0".into(),
            });
        }

        let mut layers = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            layers.push(TransformerDecoderLayer::new(
                d_model,
                num_heads,
                d_ff,
                dropout_p,
                layer_norm_eps,
                bias,
            )?);
        }

        let norm = if final_norm {
            Some(LayerNorm::new(vec![d_model], layer_norm_eps, true)?)
        } else {
            None
        };

        Ok(Self {
            layers,
            norm,
            training: true,
        })
    }

    /// Forward pass with encoder memory.
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
        let mut output = input.clone();
        for layer in &self.layers {
            output = layer.forward_with_memory(&output, memory)?;
        }
        if let Some(ref norm) = self.norm {
            output = norm.forward(&output)?;
        }
        Ok(output)
    }

    /// The number of stacked decoder layers.
    #[inline]
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

impl<T: Float> Module<T> for TransformerDecoder<T> {
    /// Forward pass using `input` as both decoder input and memory.
    ///
    /// For the typical decoder use case with separate encoder output, call
    /// [`forward_with_memory`](Self::forward_with_memory) directly.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_with_memory(input, input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        for layer in &self.layers {
            params.extend(layer.parameters());
        }
        if let Some(ref norm) = self.norm {
            params.extend(norm.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        for layer in &mut self.layers {
            params.extend(layer.parameters_mut());
        }
        if let Some(ref mut norm) = self.norm {
            params.extend(norm.parameters_mut());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            for (name, param) in layer.named_parameters() {
                params.push((format!("layers.{i}.{name}"), param));
            }
        }
        if let Some(ref norm) = self.norm {
            for (name, param) in norm.named_parameters() {
                params.push((format!("norm.{name}"), param));
            }
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.layers {
            layer.train();
        }
        if let Some(ref mut norm) = self.norm {
            norm.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.layers {
            layer.eval();
        }
        if let Some(ref mut norm) = self.norm {
            norm.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Transformer
// ===========================================================================

/// Full encoder-decoder transformer model.
///
/// Combines a [`TransformerEncoder`] and [`TransformerDecoder`] into a single
/// module, matching `torch.nn.Transformer`.
///
/// # Shape contract
///
/// - `src` (encoder input): `[batch, src_seq, d_model]`
/// - `tgt` (decoder input): `[batch, tgt_seq, d_model]`
/// - Output: `[batch, tgt_seq, d_model]`
///
/// # Example
///
/// ```ignore
/// let transformer = Transformer::<f32>::new(64, 4, 3, 3, 128, 0.1, 1e-5, true)?;
/// let src = ferrotorch_core::randn::<f32>(&[2, 10, 64])?;
/// let tgt = ferrotorch_core::randn::<f32>(&[2, 5, 64])?;
/// let output = transformer.forward_transformer(&src, &tgt)?;
/// assert_eq!(output.shape(), &[2, 5, 64]);
/// ```
#[derive(Debug)]
pub struct Transformer<T: Float> {
    encoder: TransformerEncoder<T>,
    decoder: TransformerDecoder<T>,
    training: bool,
}

impl<T: Float> Transformer<T> {
    /// Create a new full encoder-decoder transformer.
    ///
    /// # Arguments
    ///
    /// - `d_model` - The model dimension (embedding size).
    /// - `num_heads` - Number of attention heads.
    /// - `num_encoder_layers` - Number of encoder layers (default: 6).
    /// - `num_decoder_layers` - Number of decoder layers (default: 6).
    /// - `d_ff` - Hidden dimension of the SwiGLU feedforward network (default: 2048).
    /// - `dropout_p` - Dropout probability (default: 0.1).
    /// - `layer_norm_eps` - Epsilon for layer normalization.
    /// - `bias` - Whether to use bias in attention and FFN projections.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        d_model: usize,
        num_heads: usize,
        num_encoder_layers: usize,
        num_decoder_layers: usize,
        d_ff: usize,
        dropout_p: f64,
        layer_norm_eps: f64,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        let encoder = TransformerEncoder::new(
            d_model,
            num_heads,
            num_encoder_layers,
            d_ff,
            dropout_p,
            layer_norm_eps,
            bias,
            true, // final norm on encoder
        )?;
        let decoder = TransformerDecoder::new(
            d_model,
            num_heads,
            num_decoder_layers,
            d_ff,
            dropout_p,
            layer_norm_eps,
            bias,
            true, // final norm on decoder
        )?;

        Ok(Self {
            encoder,
            decoder,
            training: true,
        })
    }

    /// Forward pass: encode `src`, then decode `tgt` using the encoded memory.
    ///
    /// # Arguments
    ///
    /// - `src` - Encoder input: `[batch, src_seq, d_model]`.
    /// - `tgt` - Decoder input: `[batch, tgt_seq, d_model]`.
    ///
    /// # Returns
    ///
    /// Output tensor of shape `[batch, tgt_seq, d_model]`.
    pub fn forward_transformer(
        &self,
        src: &Tensor<T>,
        tgt: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        let memory = self.encoder.forward(src)?;
        self.decoder.forward_with_memory(tgt, &memory)
    }

    /// The number of encoder layers.
    #[inline]
    pub fn num_encoder_layers(&self) -> usize {
        self.encoder.num_layers()
    }

    /// The number of decoder layers.
    #[inline]
    pub fn num_decoder_layers(&self) -> usize {
        self.decoder.num_layers()
    }
}

impl<T: Float> Module<T> for Transformer<T> {
    /// Forward pass using `input` as both source and target.
    ///
    /// For the typical encoder-decoder use case with separate src/tgt, call
    /// [`forward_transformer`](Self::forward_transformer) directly.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_transformer(input, input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = self.encoder.parameters();
        params.extend(self.decoder.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = self.encoder.parameters_mut();
        params.extend(self.decoder.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, param) in self.encoder.named_parameters() {
            params.push((format!("encoder.{name}"), param));
        }
        for (name, param) in self.decoder.named_parameters() {
            params.push((format!("decoder.{name}"), param));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        self.encoder.train();
        self.decoder.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.encoder.eval();
        self.decoder.eval();
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

    // -- RoPE scaling tests (#515) -----------------------------------------

    #[test]
    fn test_rope_scaling_default_is_none() {
        let rope = RotaryPositionEmbedding::<f32>::new(16, 128, 10000.0).unwrap();
        assert_eq!(rope.scaling(), RoPEScaling::None);
    }

    #[test]
    fn test_rope_scaling_none_matches_classical() {
        // with_scaling(RoPEScaling::None) must produce the same caches as new().
        let a = RotaryPositionEmbedding::<f64>::new(16, 32, 10000.0).unwrap();
        let b = RotaryPositionEmbedding::<f64>::with_scaling(
            16,
            32,
            10000.0,
            RoPEConvention::default(),
            RoPEScaling::None,
        )
        .unwrap();
        let x = ferrotorch_core::from_slice(
            &(0..16).map(|i| i as f64 * 0.1).collect::<Vec<_>>(),
            &[1, 16],
        )
        .unwrap();
        let ya = a.apply(&x, 7).unwrap();
        let yb = b.apply(&x, 7).unwrap();
        for (va, vb) in ya.data().unwrap().iter().zip(yb.data().unwrap().iter()) {
            assert!((va - vb).abs() < 1e-12);
        }
    }

    #[test]
    fn test_rope_scaling_linear_halves_angles() {
        // Linear factor=2: all angles at position p under the scaled
        // schedule equal angles at p/2 under the unscaled schedule.
        let scaled = RotaryPositionEmbedding::<f64>::with_scaling(
            8,
            64,
            10000.0,
            RoPEConvention::default(),
            RoPEScaling::Linear { factor: 2.0 },
        )
        .unwrap();
        let plain = RotaryPositionEmbedding::<f64>::new(8, 64, 10000.0).unwrap();

        // All-ones probe; applying at pos=8 on scaled should equal
        // applying at pos=4 on plain.
        let x = ferrotorch_core::ones::<f64>(&[1, 8]).unwrap();
        let y_scaled = scaled.apply(&x, 8).unwrap();
        let y_plain = plain.apply(&x, 4).unwrap();
        for (a, b) in y_scaled
            .data()
            .unwrap()
            .iter()
            .zip(y_plain.data().unwrap().iter())
        {
            assert!(
                (a - b).abs() < 1e-6,
                "scaled(pos=8) should match plain(pos=4): {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_rope_scaling_ntk_inv_freq() {
        // NTK-aware scaling: base' = base * factor^(dim / (dim - 2)).
        // For i=0, inv_freq[0] = 1 / base'^0 = 1 exactly, matching the
        // unscaled schedule. For i = dim/2 - 1, NTK stretches by
        // approximately 1/factor (matches linear PI at the long end).
        use super::compute_scaled_inv_freq;

        let dim = 64;
        let base = 10000.0;
        let factor = 4.0;
        let ntk = compute_scaled_inv_freq(
            dim,
            base,
            RoPEScaling::NtkAware {
                factor,
                original_max_pos_embeddings: 2048,
            },
        );
        let plain = compute_scaled_inv_freq(dim, base, RoPEScaling::None);
        assert_eq!(ntk.len(), 32);
        assert_eq!(plain.len(), 32);

        // High-frequency dim (i=0) must round-trip bit-identically.
        assert!(
            (ntk[0] - plain[0]).abs() < 1e-15,
            "NTK inv_freq[0] should equal plain inv_freq[0]: ntk={}, plain={}",
            ntk[0],
            plain[0]
        );

        // Lowest-frequency dim (i = dim/2 - 1 = 31) should approach the
        // linear-PI scaling of 1/factor of the plain frequency.
        let ratio = ntk[31] / plain[31];
        let expected = 1.0 / factor;
        assert!(
            (ratio - expected).abs() < 0.05,
            "NTK inv_freq[31]/plain ratio should be ~{expected}: got {ratio}"
        );
    }

    #[test]
    fn test_rope_scaling_linear_inv_freq_halved() {
        use super::compute_scaled_inv_freq;
        let lin = compute_scaled_inv_freq(8, 10000.0, RoPEScaling::Linear { factor: 2.0 });
        let plain = compute_scaled_inv_freq(8, 10000.0, RoPEScaling::None);
        for (a, b) in lin.iter().zip(plain.iter()) {
            assert!(
                (a - b / 2.0).abs() < 1e-15,
                "linear should halve: {a} vs {b}/2"
            );
        }
    }

    #[test]
    fn test_rope_scaling_yarn_inv_freq_piecewise() {
        // YARN mixes extrapolation (no scale) at the highest frequencies
        // with interpolation (1/factor) at the lowest frequencies.
        use super::compute_scaled_inv_freq;
        let dim = 64;
        let base = 10000.0;
        let factor = 4.0;
        let yarn = compute_scaled_inv_freq(dim, base, RoPEScaling::yarn_default(factor, 2048));
        let plain = compute_scaled_inv_freq(dim, base, RoPEScaling::None);

        // Highest-frequency dim: extrapolation regime (value matches plain).
        assert!(
            (yarn[0] - plain[0]).abs() < 1e-12,
            "YARN[0] (extrapolation) should equal plain[0]: {} vs {}",
            yarn[0],
            plain[0]
        );
        // Lowest-frequency dim: interpolation regime (value matches plain/factor).
        let expected_low = plain[dim / 2 - 1] / factor;
        let ratio = yarn[dim / 2 - 1] / expected_low;
        assert!(
            (ratio - 1.0).abs() < 0.1,
            "YARN[dim/2-1] (interpolation) should approx equal plain/factor: {} vs {}",
            yarn[dim / 2 - 1],
            expected_low
        );
    }

    #[test]
    fn test_rope_scaling_yarn_constructs() {
        let rope = RotaryPositionEmbedding::<f32>::with_scaling(
            64,
            256,
            10000.0,
            RoPEConvention::default(),
            RoPEScaling::yarn_default(2.0, 2048),
        )
        .unwrap();
        assert!(matches!(rope.scaling(), RoPEScaling::Yarn { .. }));
        let x = ferrotorch_core::ones::<f32>(&[1, 64]).unwrap();
        for &v in rope.apply(&x, 0).unwrap().data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_rope_scaling_rejects_zero_factor() {
        let r = RotaryPositionEmbedding::<f32>::with_scaling(
            8,
            16,
            10000.0,
            RoPEConvention::default(),
            RoPEScaling::Linear { factor: 0.0 },
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_rope_scaling_rejects_negative_factor() {
        let r = RotaryPositionEmbedding::<f32>::with_scaling(
            8,
            16,
            10000.0,
            RoPEConvention::default(),
            RoPEScaling::NtkAware {
                factor: -2.0,
                original_max_pos_embeddings: 2048,
            },
        );
        assert!(r.is_err());
    }

    #[test]
    fn test_rope_scaling_accessor() {
        let rope = RotaryPositionEmbedding::<f32>::with_scaling(
            16,
            64,
            10000.0,
            RoPEConvention::default(),
            RoPEScaling::Linear { factor: 4.0 },
        )
        .unwrap();
        assert_eq!(rope.scaling(), RoPEScaling::Linear { factor: 4.0 });
    }

    // -----------------------------------------------------------------------
    // RoPE — HalfRotation convention
    // -----------------------------------------------------------------------

    #[test]
    fn test_rope_half_rotation_construction() {
        let rope = RotaryPositionEmbedding::<f32>::with_convention(
            8,
            128,
            10000.0,
            RoPEConvention::HalfRotation,
        )
        .unwrap();
        assert_eq!(rope.convention(), RoPEConvention::HalfRotation);
    }

    #[test]
    fn test_rope_half_rotation_output_shape() {
        let rope = RotaryPositionEmbedding::<f32>::with_convention(
            8,
            128,
            10000.0,
            RoPEConvention::HalfRotation,
        )
        .unwrap();
        let x = ferrotorch_core::zeros::<f32>(&[2, 4, 8]).unwrap();
        let y = rope.apply(&x, 0).unwrap();
        assert_eq!(y.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_rope_half_rotation_position_zero_is_identity() {
        // At position 0, cos(0)=1, sin(0)=0 → identity regardless of convention.
        let rope = RotaryPositionEmbedding::<f64>::with_convention(
            4,
            64,
            10000.0,
            RoPEConvention::HalfRotation,
        )
        .unwrap();
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
            4,
            64,
            10000.0,
            RoPEConvention::HalfRotation,
        )
        .unwrap();

        // Use position 1 so sin != 0.
        let x = ferrotorch_core::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let y = rope.apply(&x, 1).unwrap();

        // Get the cached cos/sin at position 1.
        let cos_data = rope.cos_cache.data().unwrap();
        let sin_data = rope.sin_cache.data().unwrap();
        // Position 1 → row offset = 1 * half_dim = 2
        let c0 = cos_data[2];
        let c1 = cos_data[3];
        let s0 = sin_data[2];
        let s1 = sin_data[3];

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
            4,
            64,
            10000.0,
            RoPEConvention::Interleaved,
        )
        .unwrap();
        let rope_hr = RotaryPositionEmbedding::<f64>::with_convention(
            4,
            64,
            10000.0,
            RoPEConvention::HalfRotation,
        )
        .unwrap();

        let x = ferrotorch_core::from_slice(&[1.0, 2.0, 3.0, 4.0], &[1, 4]).unwrap();
        let y_il = rope_il.apply(&x, 1).unwrap();
        let y_hr = rope_hr.apply(&x, 1).unwrap();

        // They should differ (different pairing).
        let il_data = y_il.data().unwrap();
        let hr_data = y_hr.data().unwrap();
        let any_differ = il_data
            .iter()
            .zip(hr_data.iter())
            .any(|(&a, &b)| (a - b).abs() > 1e-10);
        assert!(
            any_differ,
            "interleaved and half-rotation should produce different outputs at pos > 0"
        );
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
        let k2_data = vec![2.0f64; 3];
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

    // -- GQA KVCache tests (#506) -------------------------------------------

    #[test]
    fn test_kv_cache_gqa_stores_at_kv_head_granularity() {
        // Llama 3 8B: 8 KV heads, not 32. Cache dim 1 must be num_kv_heads.
        let mut cache = KVCache::<f32>::new(8192);
        let k = ferrotorch_core::zeros::<f32>(&[1, 8, 3, 128]).unwrap();
        let v = ferrotorch_core::zeros::<f32>(&[1, 8, 3, 128]).unwrap();
        let (fk, _) = cache.update(k, v).unwrap();
        assert_eq!(fk.shape(), &[1, 8, 3, 128]);
        assert_eq!(cache.num_kv_heads(), Some(8));
        assert_eq!(cache.head_dim(), Some(128));
        assert_eq!(cache.batch_size(), Some(1));
    }

    #[test]
    fn test_kv_cache_with_dims_pre_declares_shape() {
        let cache = KVCache::<f32>::with_dims(8192, 1, 8, 128);
        assert_eq!(cache.num_kv_heads(), Some(8));
        assert_eq!(cache.head_dim(), Some(128));
        assert_eq!(cache.batch_size(), Some(1));
        assert!(cache.is_empty());
    }

    #[test]
    fn test_kv_cache_with_dims_rejects_first_update_mismatch() {
        // Pre-declare num_kv_heads=8, then try to push num_kv_heads=4.
        let mut cache = KVCache::<f32>::with_dims(128, 1, 8, 16);
        let k = ferrotorch_core::zeros::<f32>(&[1, 4, 2, 16]).unwrap();
        let v = ferrotorch_core::zeros::<f32>(&[1, 4, 2, 16]).unwrap();
        assert!(cache.update(k, v).is_err());
    }

    #[test]
    fn test_kv_cache_with_dims_rejects_head_dim_mismatch() {
        let mut cache = KVCache::<f32>::with_dims(128, 1, 8, 16);
        let k = ferrotorch_core::zeros::<f32>(&[1, 8, 2, 32]).unwrap(); // dim=32 != 16
        let v = ferrotorch_core::zeros::<f32>(&[1, 8, 2, 32]).unwrap();
        assert!(cache.update(k, v).is_err());
    }

    #[test]
    fn test_kv_cache_with_dims_rejects_batch_mismatch() {
        let mut cache = KVCache::<f32>::with_dims(128, 2, 4, 8);
        let k = ferrotorch_core::zeros::<f32>(&[1, 4, 2, 8]).unwrap(); // B=1 != 2
        let v = ferrotorch_core::zeros::<f32>(&[1, 4, 2, 8]).unwrap();
        assert!(cache.update(k, v).is_err());
    }

    #[test]
    fn test_kv_cache_with_dims_accepts_matching_update() {
        let mut cache = KVCache::<f32>::with_dims(128, 1, 8, 16);
        let k = ferrotorch_core::ones::<f32>(&[1, 8, 3, 16]).unwrap();
        let v = ferrotorch_core::ones::<f32>(&[1, 8, 3, 16]).unwrap();
        assert!(cache.update(k, v).is_ok());
        assert_eq!(cache.seq_len(), 3);
    }

    #[test]
    fn test_kv_cache_inferred_dims_reject_subsequent_mismatch() {
        // First push defines dims; second push with different num_kv_heads must fail.
        let mut cache = KVCache::<f32>::new(128);
        let k1 = ferrotorch_core::zeros::<f32>(&[1, 8, 2, 16]).unwrap();
        let v1 = ferrotorch_core::zeros::<f32>(&[1, 8, 2, 16]).unwrap();
        cache.update(k1, v1).unwrap();
        assert_eq!(cache.num_kv_heads(), Some(8));

        let k2 = ferrotorch_core::zeros::<f32>(&[1, 4, 1, 16]).unwrap(); // 4 != 8
        let v2 = ferrotorch_core::zeros::<f32>(&[1, 4, 1, 16]).unwrap();
        assert!(cache.update(k2, v2).is_err());
    }

    #[test]
    fn test_kv_cache_dims_not_yet_pinned_on_fresh_new() {
        let cache = KVCache::<f32>::new(128);
        assert_eq!(cache.num_kv_heads(), None);
        assert_eq!(cache.head_dim(), None);
        assert_eq!(cache.batch_size(), None);
    }

    #[test]
    fn test_kv_cache_reset_preserves_pinned_dims() {
        let mut cache = KVCache::<f32>::with_dims(128, 1, 8, 16);
        let k = ferrotorch_core::ones::<f32>(&[1, 8, 2, 16]).unwrap();
        let v = ferrotorch_core::ones::<f32>(&[1, 8, 2, 16]).unwrap();
        cache.update(k, v).unwrap();
        cache.reset();
        assert!(cache.is_empty());
        // Dims are retained so the cache still validates the next push.
        assert_eq!(cache.num_kv_heads(), Some(8));
        let bad = ferrotorch_core::zeros::<f32>(&[1, 4, 1, 16]).unwrap();
        assert!(cache.update(bad.clone(), bad).is_err());
    }

    #[test]
    fn test_kv_cache_gqa_prefill_then_decode_preserves_all_positions() {
        // Acceptance: "Decoder step using this cache produces outputs
        // matching un-cached GQA attention on the same inputs." We prove
        // the cache round-trips data faithfully by:
        //   (1) prefilling 4 tokens, then pushing 1 decode token
        //   (2) verifying every (batch, head, seq, dim) position in the
        //       returned full tensor matches the source tensors at the
        //       corresponding index.
        let build = |seed: u64, shape: &[usize]| {
            let numel: usize = shape.iter().product();
            let data: Vec<f32> = (0..numel)
                .map(|i| ((i as u64).wrapping_mul(seed) % 997) as f32 * 0.001)
                .collect();
            ferrotorch_core::from_slice(&data, shape).unwrap()
        };

        // Llama-8B-ish: 1 batch, 8 KV heads, head_dim=16 (scaled down).
        let (b, h, s_prefill, s_decode, d) = (1usize, 8usize, 4usize, 1usize, 16usize);
        let s_full = s_prefill + s_decode;

        let k_prefill = build(7, &[b, h, s_prefill, d]);
        let v_prefill = build(11, &[b, h, s_prefill, d]);
        let k_decode = build(13, &[b, h, s_decode, d]);
        let v_decode = build(17, &[b, h, s_decode, d]);

        let mut cache = KVCache::<f32>::with_dims(16, b, h, d);
        cache.update(k_prefill.clone(), v_prefill.clone()).unwrap();
        let (fk, fv) = cache.update(k_decode.clone(), v_decode.clone()).unwrap();
        assert_eq!(fk.shape(), &[b, h, s_full, d]);
        assert_eq!(fv.shape(), &[b, h, s_full, d]);

        let fk_data = fk.data_vec().unwrap();
        let fv_data = fv.data_vec().unwrap();
        let kp = k_prefill.data_vec().unwrap();
        let vp = v_prefill.data_vec().unwrap();
        let kd = k_decode.data_vec().unwrap();
        let vd = v_decode.data_vec().unwrap();

        // Row-major [B, H, S, D] stride.
        let full_idx = |bi, hi, si, di| ((bi * h + hi) * s_full + si) * d + di;
        let src_idx = |bi, hi, si, di, s_len| ((bi * h + hi) * s_len + si) * d + di;

        for bi in 0..b {
            for hi in 0..h {
                for si in 0..s_full {
                    for di in 0..d {
                        let out = full_idx(bi, hi, si, di);
                        let (exp_k, exp_v) = if si < s_prefill {
                            let src = src_idx(bi, hi, si, di, s_prefill);
                            (kp[src], vp[src])
                        } else {
                            let src = src_idx(bi, hi, si - s_prefill, di, s_decode);
                            (kd[src], vd[src])
                        };
                        assert!(
                            (fk_data[out] - exp_k).abs() < 1e-6,
                            "k mismatch at [b={bi}, h={hi}, s={si}, d={di}]: got {}, want {exp_k}",
                            fk_data[out]
                        );
                        assert!(
                            (fv_data[out] - exp_v).abs() < 1e-6,
                            "v mismatch at [b={bi}, h={hi}, s={si}, d={di}]: got {}, want {exp_v}",
                            fv_data[out]
                        );
                    }
                }
            }
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
        let mut layer = TransformerEncoderLayer::<f32>::new(8, 2, 16, 0.1, 1e-5, false).unwrap();
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
        let mut layer = TransformerDecoderLayer::<f32>::new(8, 2, 16, 0.1, 1e-5, false).unwrap();
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

    // -----------------------------------------------------------------------
    // TransformerEncoder
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoder_construction() {
        let enc = TransformerEncoder::<f32>::new(16, 4, 3, 32, 0.0, 1e-5, true, true);
        assert!(enc.is_ok());
        assert_eq!(enc.unwrap().num_layers(), 3);
    }

    #[test]
    fn test_encoder_zero_layers_rejected() {
        assert!(TransformerEncoder::<f32>::new(16, 4, 0, 32, 0.0, 1e-5, true, true).is_err());
    }

    #[test]
    fn test_encoder_forward_shape() {
        let enc = TransformerEncoder::<f32>::new(16, 4, 2, 32, 0.0, 1e-5, false, true).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[2, 5, 16]).unwrap();
        let output = enc.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_encoder_forward_no_final_norm() {
        let enc = TransformerEncoder::<f32>::new(8, 2, 2, 16, 0.0, 1e-5, false, false).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 3, 8]).unwrap();
        let output = enc.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_encoder_forward_values_finite() {
        let enc = TransformerEncoder::<f32>::new(8, 2, 2, 16, 0.0, 1e-5, true, true).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[1, 3, 8]).unwrap();
        let output = enc.forward(&input).unwrap();
        for &v in output.data().unwrap() {
            assert!(
                v.is_finite(),
                "TransformerEncoder produced non-finite value: {v}"
            );
        }
    }

    #[test]
    fn test_encoder_parameters_with_final_norm() {
        let enc = TransformerEncoder::<f32>::new(8, 2, 2, 16, 0.0, 1e-5, true, true).unwrap();
        // Each encoder layer: 18 params (see test_encoder_layer_parameters_count)
        // Final norm: 2 params (weight + bias)
        // Total: 2 * 18 + 2 = 38
        assert_eq!(enc.parameters().len(), 38);
    }

    #[test]
    fn test_encoder_named_parameters_have_layer_prefix() {
        let enc = TransformerEncoder::<f32>::new(8, 2, 2, 16, 0.0, 1e-5, true, true).unwrap();
        let named = enc.named_parameters();
        // Verify layer indexing in names.
        let has_layer_0 = named.iter().any(|(n, _)| n.starts_with("layers.0."));
        let has_layer_1 = named.iter().any(|(n, _)| n.starts_with("layers.1."));
        let has_norm = named.iter().any(|(n, _)| n.starts_with("norm."));
        assert!(has_layer_0, "missing layers.0.* in named_parameters");
        assert!(has_layer_1, "missing layers.1.* in named_parameters");
        assert!(has_norm, "missing norm.* in named_parameters");
    }

    #[test]
    fn test_encoder_train_eval() {
        let mut enc = TransformerEncoder::<f32>::new(8, 2, 2, 16, 0.1, 1e-5, false, false).unwrap();
        assert!(enc.is_training());
        enc.eval();
        assert!(!enc.is_training());
        enc.train();
        assert!(enc.is_training());
    }

    #[test]
    fn test_encoder_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TransformerEncoder<f32>>();
        assert_send_sync::<TransformerEncoder<f64>>();
    }

    // -----------------------------------------------------------------------
    // TransformerDecoder
    // -----------------------------------------------------------------------

    #[test]
    fn test_decoder_construction() {
        let dec = TransformerDecoder::<f32>::new(16, 4, 3, 32, 0.0, 1e-5, true, true);
        assert!(dec.is_ok());
        assert_eq!(dec.unwrap().num_layers(), 3);
    }

    #[test]
    fn test_decoder_zero_layers_rejected() {
        assert!(TransformerDecoder::<f32>::new(16, 4, 0, 32, 0.0, 1e-5, true, true).is_err());
    }

    #[test]
    fn test_decoder_forward_with_memory_shape() {
        let dec = TransformerDecoder::<f32>::new(16, 4, 2, 32, 0.0, 1e-5, false, true).unwrap();
        let tgt = ferrotorch_core::zeros::<f32>(&[2, 4, 16]).unwrap();
        let memory = ferrotorch_core::zeros::<f32>(&[2, 6, 16]).unwrap();
        let output = dec.forward_with_memory(&tgt, &memory).unwrap();
        assert_eq!(output.shape(), &[2, 4, 16]);
    }

    #[test]
    fn test_decoder_forward_values_finite() {
        let dec = TransformerDecoder::<f32>::new(8, 2, 2, 16, 0.0, 1e-5, true, true).unwrap();
        let tgt = ferrotorch_core::ones::<f32>(&[1, 3, 8]).unwrap();
        let mem = ferrotorch_core::ones::<f32>(&[1, 5, 8]).unwrap();
        let output = dec.forward_with_memory(&tgt, &mem).unwrap();
        for &v in output.data().unwrap() {
            assert!(
                v.is_finite(),
                "TransformerDecoder produced non-finite value: {v}"
            );
        }
    }

    #[test]
    fn test_decoder_parameters_with_final_norm() {
        let dec = TransformerDecoder::<f32>::new(8, 2, 2, 16, 0.0, 1e-5, true, true).unwrap();
        // Each decoder layer: 28 params (see test_decoder_layer_parameters_count)
        // Final norm: 2 params (weight + bias)
        // Total: 2 * 28 + 2 = 58
        assert_eq!(dec.parameters().len(), 58);
    }

    #[test]
    fn test_decoder_named_parameters_have_layer_prefix() {
        let dec = TransformerDecoder::<f32>::new(8, 2, 2, 16, 0.0, 1e-5, true, true).unwrap();
        let named = dec.named_parameters();
        let has_layer_0 = named.iter().any(|(n, _)| n.starts_with("layers.0."));
        let has_layer_1 = named.iter().any(|(n, _)| n.starts_with("layers.1."));
        let has_norm = named.iter().any(|(n, _)| n.starts_with("norm."));
        assert!(has_layer_0, "missing layers.0.* in named_parameters");
        assert!(has_layer_1, "missing layers.1.* in named_parameters");
        assert!(has_norm, "missing norm.* in named_parameters");
    }

    #[test]
    fn test_decoder_train_eval() {
        let mut dec = TransformerDecoder::<f32>::new(8, 2, 2, 16, 0.1, 1e-5, false, false).unwrap();
        assert!(dec.is_training());
        dec.eval();
        assert!(!dec.is_training());
        dec.train();
        assert!(dec.is_training());
    }

    #[test]
    fn test_decoder_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<TransformerDecoder<f32>>();
        assert_send_sync::<TransformerDecoder<f64>>();
    }

    // -----------------------------------------------------------------------
    // Transformer (full encoder-decoder)
    // -----------------------------------------------------------------------

    #[test]
    fn test_transformer_construction() {
        let t = Transformer::<f32>::new(16, 4, 2, 2, 32, 0.0, 1e-5, true);
        assert!(t.is_ok());
        let t = t.unwrap();
        assert_eq!(t.num_encoder_layers(), 2);
        assert_eq!(t.num_decoder_layers(), 2);
    }

    #[test]
    fn test_transformer_forward_shape() {
        let t = Transformer::<f32>::new(16, 4, 2, 2, 32, 0.0, 1e-5, false).unwrap();
        let src = ferrotorch_core::zeros::<f32>(&[2, 10, 16]).unwrap();
        let tgt = ferrotorch_core::zeros::<f32>(&[2, 5, 16]).unwrap();
        let output = t.forward_transformer(&src, &tgt).unwrap();
        assert_eq!(output.shape(), &[2, 5, 16]);
    }

    #[test]
    fn test_transformer_self_forward_shape() {
        // Module::forward uses input as both src and tgt.
        let t = Transformer::<f32>::new(8, 2, 1, 1, 16, 0.0, 1e-5, false).unwrap();
        let input = ferrotorch_core::zeros::<f32>(&[1, 3, 8]).unwrap();
        let output = t.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 3, 8]);
    }

    #[test]
    fn test_transformer_forward_values_finite() {
        let t = Transformer::<f32>::new(8, 2, 2, 2, 16, 0.0, 1e-5, true).unwrap();
        let src = ferrotorch_core::ones::<f32>(&[1, 4, 8]).unwrap();
        let tgt = ferrotorch_core::ones::<f32>(&[1, 3, 8]).unwrap();
        let output = t.forward_transformer(&src, &tgt).unwrap();
        for &v in output.data().unwrap() {
            assert!(v.is_finite(), "Transformer produced non-finite value: {v}");
        }
    }

    #[test]
    fn test_transformer_parameters_count() {
        let t = Transformer::<f32>::new(8, 2, 2, 2, 16, 0.0, 1e-5, true).unwrap();
        // Encoder: 2 layers * 18 params + 2 (final norm) = 38
        // Decoder: 2 layers * 28 params + 2 (final norm) = 58
        // Total: 96
        assert_eq!(t.parameters().len(), 96);
    }

    #[test]
    fn test_transformer_named_parameters_prefixed() {
        let t = Transformer::<f32>::new(8, 2, 1, 1, 16, 0.0, 1e-5, true).unwrap();
        let named = t.named_parameters();
        let has_encoder = named.iter().any(|(n, _)| n.starts_with("encoder."));
        let has_decoder = named.iter().any(|(n, _)| n.starts_with("decoder."));
        assert!(has_encoder, "missing encoder.* in named_parameters");
        assert!(has_decoder, "missing decoder.* in named_parameters");
    }

    #[test]
    fn test_transformer_train_eval() {
        let mut t = Transformer::<f32>::new(8, 2, 1, 1, 16, 0.1, 1e-5, false).unwrap();
        assert!(t.is_training());
        t.eval();
        assert!(!t.is_training());
        t.train();
        assert!(t.is_training());
    }

    #[test]
    fn test_transformer_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Transformer<f32>>();
        assert_send_sync::<Transformer<f64>>();
    }
}
