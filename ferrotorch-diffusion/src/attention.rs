//! Multi-head attention + the `Transformer2DModel` wrapper used by the
//! SD UNet's CrossAttn blocks.
//!
//! Diffusers' layout 1:1:
//!
//! ```text
//! Attention(query_dim, cross_attention_dim?, heads, dim_head)
//!   to_q.{weight,bias}    [inner, query_dim]   (inner = heads * dim_head)
//!   to_k.{weight,bias}    [inner, kv_dim]       (kv_dim = cross_attention_dim
//!                                                or query_dim for self-attn)
//!   to_v.{weight,bias}    [inner, kv_dim]
//!   to_out.0.{weight,bias}[query_dim, inner]
//!   to_out.1              Dropout (no params)
//! ```
//!
//! For SD-1.5 UNet, biases on `to_q/to_k/to_v` are disabled
//! (`bias=False`); the output projection `to_out.0` keeps its bias.
//!
//! `BasicTransformerBlock`:
//!
//! ```text
//! h0 = LayerNorm1(x)
//! h1 = Attention1(h0, h0, h0)             # self-attn
//! x  = x + h1
//! h0 = LayerNorm2(x)
//! h2 = Attention2(h0, encoder_hidden, …)  # cross-attn
//! x  = x + h2
//! h0 = LayerNorm3(x)
//! h3 = FeedForward(h0)                    # GEGLU + Linear
//! x  = x + h3
//! ```
//!
//! `Transformer2DModel`:
//!
//! ```text
//! GroupNorm(32) -> proj_in (Conv2d k=1) -> flatten [B, HW, C]
//!   -> N × BasicTransformerBlock -> reshape back -> proj_out + residual
//! ```

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv2d, GELU, GroupNorm, LayerNorm, Linear};

// ---------------------------------------------------------------------------
// Attention (multi-head, optional cross-attention)
// ---------------------------------------------------------------------------

/// Multi-head attention block. Supports self-attention (when `key` and
/// `value` are derived from the same tensor as `query`) and
/// cross-attention (when they come from `encoder_hidden_states`).
///
/// This is the `Attention` class in
/// `diffusers.models.attention_processor` configured as it appears in
/// SD-1.5's UNet — `bias = False` on q/k/v and `out_bias = True` on
/// `to_out.0`, no `group_norm`, no `spatial_norm`, no
/// `added_kv_proj_dim`.
#[derive(Debug)]
pub struct Attention<T: Float> {
    /// Per-head dimension.
    pub dim_head: usize,
    /// Number of heads.
    pub heads: usize,
    /// `inner_dim = heads * dim_head`.
    pub inner_dim: usize,
    /// Query projection: `[inner_dim, query_dim]`.
    pub to_q: Linear<T>,
    /// Key projection: `[inner_dim, kv_dim]`.
    pub to_k: Linear<T>,
    /// Value projection: `[inner_dim, kv_dim]`.
    pub to_v: Linear<T>,
    /// Output projection: `[query_dim, inner_dim]` (with bias).
    pub to_out_0: Linear<T>,
    query_dim: usize,
    kv_dim: usize,
    scale: f64,
    training: bool,
}

impl<T: Float> Attention<T> {
    /// Build a randomly-initialized `Attention`.
    ///
    /// `cross_attention_dim = None` means self-attention
    /// (`kv_dim = query_dim`); a `Some(_)` value enables cross-attention.
    ///
    /// `bias` controls `to_q/to_k/to_v` bias (SD-1.5 sets this to false).
    /// `to_out.0` always has bias (matches diffusers default
    /// `out_bias=True`).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] when any `Linear` size
    /// is invalid.
    pub fn new(
        query_dim: usize,
        cross_attention_dim: Option<usize>,
        heads: usize,
        dim_head: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        let inner_dim = heads * dim_head;
        let kv_dim = cross_attention_dim.unwrap_or(query_dim);
        let to_q = Linear::<T>::new(query_dim, inner_dim, bias)?;
        let to_k = Linear::<T>::new(kv_dim, inner_dim, bias)?;
        let to_v = Linear::<T>::new(kv_dim, inner_dim, bias)?;
        let to_out_0 = Linear::<T>::new(inner_dim, query_dim, true)?;
        let scale = (dim_head as f64).sqrt().recip();
        Ok(Self {
            dim_head,
            heads,
            inner_dim,
            to_q,
            to_k,
            to_v,
            to_out_0,
            query_dim,
            kv_dim,
            scale,
            training: false,
        })
    }

    /// Forward with optional encoder hidden states.
    ///
    /// `hidden_states` has shape `[B, N, query_dim]`. When
    /// `encoder_hidden_states` is `None` this is self-attention; when
    /// `Some([B, S, kv_dim])` it's cross-attention.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] on rank / dim
    /// disagreement.
    pub fn forward_xattn(
        &self,
        hidden_states: &Tensor<T>,
        encoder_hidden_states: Option<&Tensor<T>>,
    ) -> FerrotorchResult<Tensor<T>> {
        if hidden_states.ndim() != 3 || hidden_states.shape()[2] != self.query_dim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Attention::forward_xattn: expected hidden_states [B, N, {}], got {:?}",
                    self.query_dim,
                    hidden_states.shape()
                ),
            });
        }
        let b = hidden_states.shape()[0];
        let n = hidden_states.shape()[1];
        // kv source.
        let kv = encoder_hidden_states.unwrap_or(hidden_states);
        if kv.ndim() != 3 || kv.shape()[0] != b || kv.shape()[2] != self.kv_dim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Attention::forward_xattn: expected kv [B={b}, S, {}], got {:?}",
                    self.kv_dim,
                    kv.shape()
                ),
            });
        }
        let s = kv.shape()[1];

        // -- Linear projections. q: [B, N, inner], k/v: [B, S, inner].
        let q = self.to_q.forward(hidden_states)?;
        let k = self.to_k.forward(kv)?;
        let v = self.to_v.forward(kv)?;

        // -- Reshape to per-head: [B, N, H, D] then transpose to
        //    [B, H, N, D], collapse to [B*H, N, D] for the BMM kernel.
        //    Same trick for k, v over S.
        let h = self.heads;
        let d = self.dim_head;
        let q = q
            .reshape_t(&[b as isize, n as isize, h as isize, d as isize])?
            .transpose(1, 2)? // [B, H, N, D]
            .contiguous()?
            .reshape_t(&[(b * h) as isize, n as isize, d as isize])?;
        let k = k
            .reshape_t(&[b as isize, s as isize, h as isize, d as isize])?
            .transpose(1, 2)? // [B, H, S, D]
            .contiguous()?
            .reshape_t(&[(b * h) as isize, s as isize, d as isize])?;
        let v = v
            .reshape_t(&[b as isize, s as isize, h as isize, d as isize])?
            .transpose(1, 2)? // [B, H, S, D]
            .contiguous()?
            .reshape_t(&[(b * h) as isize, s as isize, d as isize])?;

        // -- scores = (q @ k^T) * scale.
        let k_t = k.transpose(1, 2)?.contiguous()?; // [B*H, D, S]
        let scores = q.bmm(&k_t)?; // [B*H, N, S]
        let scale_t = T::from(self.scale).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "Attention::forward_xattn: failed to cast attention scale into Float".into(),
        })?;
        let scale_tensor = ferrotorch_core::scalar::<T>(scale_t)?;
        let scores_scaled = ferrotorch_core::grad_fns::arithmetic::mul(&scores, &scale_tensor)?;
        let probs = scores_scaled.softmax()?; // [B*H, N, S]
        let attended = probs.bmm(&v)?; // [B*H, N, D]

        // -- Merge heads: [B*H, N, D] -> [B, H, N, D] -> [B, N, H, D]
        //    -> [B, N, inner_dim].
        let attended = attended
            .reshape_t(&[b as isize, h as isize, n as isize, d as isize])?
            .transpose(1, 2)? // [B, N, H, D]
            .contiguous()?
            .reshape_t(&[b as isize, n as isize, self.inner_dim as isize])?;

        // -- Output projection.
        self.to_out_0.forward(&attended)
    }
}

impl<T: Float> Module<T> for Attention<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Self-attention path.
        self.forward_xattn(input, None)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.to_q.parameters());
        o.extend(self.to_k.parameters());
        o.extend(self.to_v.parameters());
        o.extend(self.to_out_0.parameters());
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.to_q.parameters_mut());
        o.extend(self.to_k.parameters_mut());
        o.extend(self.to_v.parameters_mut());
        o.extend(self.to_out_0.parameters_mut());
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (n, p) in self.to_q.named_parameters() {
            o.push((format!("to_q.{n}"), p));
        }
        for (n, p) in self.to_k.named_parameters() {
            o.push((format!("to_k.{n}"), p));
        }
        for (n, p) in self.to_v.named_parameters() {
            o.push((format!("to_v.{n}"), p));
        }
        for (n, p) in self.to_out_0.named_parameters() {
            o.push((format!("to_out.0.{n}"), p));
        }
        o
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
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                let ok = k.starts_with("to_q.")
                    || k.starts_with("to_k.")
                    || k.starts_with("to_v.")
                    || k.starts_with("to_out.0.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in Attention state_dict: \"{k}\""),
                    });
                }
            }
        }
        self.to_q.load_state_dict(&extract("to_q"), strict)?;
        self.to_k.load_state_dict(&extract("to_k"), strict)?;
        self.to_v.load_state_dict(&extract("to_v"), strict)?;
        self.to_out_0
            .load_state_dict(&extract("to_out.0"), strict)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// FeedForward (GEGLU + Linear)
// ---------------------------------------------------------------------------

/// GEGLU-style feed-forward (matches the SD UNet's
/// `BasicTransformerBlock.ff` exactly):
///
/// ```text
/// net.0  = GEGLU(dim, dim * mult)
///          = Linear(dim, 2 * dim * mult)         # proj
///          x, gate = chunk(2, dim=-1)
///          return x * gelu(gate)
/// net.1  = Dropout (no params)
/// net.2  = Linear(dim * mult, dim)
/// ```
///
/// Diffusers' `FeedForward` defaults: `mult = 4`,
/// `activation_fn = "geglu"`. SD-1.5 UNet uses both defaults.
///
/// State-dict layout:
///
/// ```text
/// net.0.proj.{weight,bias}    [2 * dim_ff, dim], [2 * dim_ff]
/// net.2.{weight,bias}         [dim, dim_ff],     [dim]
/// ```
#[derive(Debug)]
pub struct FeedForward<T: Float> {
    /// GEGLU's expansion projection (`dim -> 2 * dim_ff`).
    pub net_0_proj: Linear<T>,
    /// Output projection (`dim_ff -> dim`).
    pub net_2: Linear<T>,
    activation: GELU,
    dim_ff: usize,
    training: bool,
}

impl<T: Float> FeedForward<T> {
    /// Build a GEGLU `FeedForward` (`dim_ff = dim * mult`).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] for invalid dims.
    pub fn new(dim: usize, mult: usize) -> FerrotorchResult<Self> {
        let dim_ff = dim * mult;
        let net_0_proj = Linear::<T>::new(dim, 2 * dim_ff, true)?;
        let net_2 = Linear::<T>::new(dim_ff, dim, true)?;
        Ok(Self {
            net_0_proj,
            net_2,
            activation: GELU::new(),
            dim_ff,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for FeedForward<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // proj -> chunk(2, -1) -> x * gelu(gate)
        let proj = self.net_0_proj.forward(input)?;
        // `chunk` operates on a positive dim — last axis here.
        let last = proj.ndim() - 1;
        let parts = proj.chunk(2, last)?;
        if parts.len() != 2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "FeedForward: chunk(2) returned {} parts (expected 2)",
                    parts.len()
                ),
            });
        }
        let x = parts[0].contiguous()?;
        let gate = parts[1].contiguous()?;
        let gated = self.activation.forward(&gate)?;
        let activated = ferrotorch_core::grad_fns::arithmetic::mul(&x, &gated)?;
        self.net_2.forward(&activated)
    }
    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.net_0_proj.parameters());
        o.extend(self.net_2.parameters());
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.net_0_proj.parameters_mut());
        o.extend(self.net_2.parameters_mut());
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (n, p) in self.net_0_proj.named_parameters() {
            o.push((format!("net.0.proj.{n}"), p));
        }
        for (n, p) in self.net_2.named_parameters() {
            o.push((format!("net.2.{n}"), p));
        }
        o
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
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                let ok = k.starts_with("net.0.proj.") || k.starts_with("net.2.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in FeedForward state_dict: \"{k}\""),
                    });
                }
            }
        }
        self.net_0_proj
            .load_state_dict(&extract("net.0.proj"), strict)?;
        self.net_2.load_state_dict(&extract("net.2"), strict)?;
        let _ = self.dim_ff;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// BasicTransformerBlock
// ---------------------------------------------------------------------------

/// Diffusers' `BasicTransformerBlock` configured the way SD-1.5's UNet
/// uses it: pre-LayerNorm on every sub-layer, self-attn followed by
/// cross-attn followed by GEGLU FeedForward, all with residuals.
///
/// State-dict layout:
///
/// ```text
/// norm1.{weight,bias}   [dim], [dim]
/// attn1.<keys>          # self-attn (Attention with cross_attention_dim=None)
/// norm2.{weight,bias}
/// attn2.<keys>          # cross-attn
/// norm3.{weight,bias}
/// ff.<keys>             # FeedForward (GEGLU)
/// ```
#[derive(Debug)]
pub struct BasicTransformerBlock<T: Float> {
    /// LayerNorm before self-attn.
    pub norm1: LayerNorm<T>,
    /// Self-attention.
    pub attn1: Attention<T>,
    /// LayerNorm before cross-attn.
    pub norm2: LayerNorm<T>,
    /// Cross-attention.
    pub attn2: Attention<T>,
    /// LayerNorm before FF.
    pub norm3: LayerNorm<T>,
    /// GEGLU FeedForward.
    pub ff: FeedForward<T>,
    dim: usize,
    training: bool,
}

impl<T: Float> BasicTransformerBlock<T> {
    /// Build a randomly-initialized `BasicTransformerBlock`.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] for invalid dims.
    pub fn new(
        dim: usize,
        heads: usize,
        dim_head: usize,
        cross_attention_dim: usize,
    ) -> FerrotorchResult<Self> {
        // SD-1.5 sets `attention_bias = False` for the UNet (no bias on
        // q/k/v); the output projection (`to_out.0`) keeps its bias
        // unconditionally.
        let norm1 = LayerNorm::<T>::new(vec![dim], 1e-5, true)?;
        let attn1 = Attention::<T>::new(dim, None, heads, dim_head, false)?;
        let norm2 = LayerNorm::<T>::new(vec![dim], 1e-5, true)?;
        let attn2 = Attention::<T>::new(dim, Some(cross_attention_dim), heads, dim_head, false)?;
        let norm3 = LayerNorm::<T>::new(vec![dim], 1e-5, true)?;
        let ff = FeedForward::<T>::new(dim, 4)?;
        Ok(Self {
            norm1,
            attn1,
            norm2,
            attn2,
            norm3,
            ff,
            dim,
            training: false,
        })
    }

    /// Forward with optional encoder hidden states. Self-attn ignores
    /// `encoder_hidden_states`, cross-attn uses it.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] on rank disagreement,
    /// underlying op errors otherwise.
    pub fn forward_xattn(
        &self,
        x: &Tensor<T>,
        encoder_hidden_states: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if x.ndim() != 3 || x.shape()[2] != self.dim {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "BasicTransformerBlock::forward: expected x [B, N, {}], got {:?}",
                    self.dim,
                    x.shape()
                ),
            });
        }
        // Sub-block 1: self-attn.
        let h1 = self.norm1.forward(x)?;
        let h1 = self.attn1.forward_xattn(&h1, None)?;
        let x = ferrotorch_core::grad_fns::arithmetic::add(&h1, x)?;
        // Sub-block 2: cross-attn.
        let h2 = self.norm2.forward(&x)?;
        let h2 = self.attn2.forward_xattn(&h2, Some(encoder_hidden_states))?;
        let x = ferrotorch_core::grad_fns::arithmetic::add(&h2, &x)?;
        // Sub-block 3: FF.
        let h3 = self.norm3.forward(&x)?;
        let h3 = self.ff.forward(&h3)?;
        ferrotorch_core::grad_fns::arithmetic::add(&h3, &x)
    }
}

impl<T: Float> Module<T> for BasicTransformerBlock<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "BasicTransformerBlock::forward: cross-attn requires \
                      encoder_hidden_states — call forward_xattn instead"
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.norm1.parameters());
        o.extend(self.attn1.parameters());
        o.extend(self.norm2.parameters());
        o.extend(self.attn2.parameters());
        o.extend(self.norm3.parameters());
        o.extend(self.ff.parameters());
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.norm1.parameters_mut());
        o.extend(self.attn1.parameters_mut());
        o.extend(self.norm2.parameters_mut());
        o.extend(self.attn2.parameters_mut());
        o.extend(self.norm3.parameters_mut());
        o.extend(self.ff.parameters_mut());
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (n, p) in self.norm1.named_parameters() {
            o.push((format!("norm1.{n}"), p));
        }
        for (n, p) in self.attn1.named_parameters() {
            o.push((format!("attn1.{n}"), p));
        }
        for (n, p) in self.norm2.named_parameters() {
            o.push((format!("norm2.{n}"), p));
        }
        for (n, p) in self.attn2.named_parameters() {
            o.push((format!("attn2.{n}"), p));
        }
        for (n, p) in self.norm3.named_parameters() {
            o.push((format!("norm3.{n}"), p));
        }
        for (n, p) in self.ff.named_parameters() {
            o.push((format!("ff.{n}"), p));
        }
        o
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
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                let ok = k.starts_with("norm1.")
                    || k.starts_with("attn1.")
                    || k.starts_with("norm2.")
                    || k.starts_with("attn2.")
                    || k.starts_with("norm3.")
                    || k.starts_with("ff.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in BasicTransformerBlock state_dict: \"{k}\""
                        ),
                    });
                }
            }
        }
        self.norm1.load_state_dict(&extract("norm1"), strict)?;
        self.attn1.load_state_dict(&extract("attn1"), strict)?;
        self.norm2.load_state_dict(&extract("norm2"), strict)?;
        self.attn2.load_state_dict(&extract("attn2"), strict)?;
        self.norm3.load_state_dict(&extract("norm3"), strict)?;
        self.ff.load_state_dict(&extract("ff"), strict)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Transformer2DModel
// ---------------------------------------------------------------------------

/// Diffusers' `Transformer2DModel` configured the way SD-1.5's UNet
/// uses it:
///
/// ```text
/// h = norm(x)                              # GroupNorm(32, in_channels)
/// h = proj_in(h)                           # Conv2d(C, inner, k=1) [use_linear_projection=False]
/// h = h.permute(0, 2, 3, 1).reshape(B, H*W, inner)
/// for block in transformer_blocks:
///     h = block(h, encoder_hidden_states)
/// h = h.reshape(B, H, W, inner).permute(0, 3, 1, 2)
/// h = proj_out(h)                          # Conv2d(inner, C, k=1)
/// return h + residual
/// ```
///
/// SD-1.5 v1 uses `Conv2d` (not Linear) for `proj_in`/`proj_out`
/// (`use_linear_projection=False`). `transformer_layers_per_block=1`
/// (the diffusers default and the SD-1.5 v1 setting).
#[derive(Debug)]
pub struct Transformer2DModel<T: Float> {
    /// GroupNorm before `proj_in`.
    pub norm: GroupNorm<T>,
    /// `proj_in`: Conv2d(C, inner, k=1).
    pub proj_in: Conv2d<T>,
    /// `N × BasicTransformerBlock`.
    pub transformer_blocks: Vec<BasicTransformerBlock<T>>,
    /// `proj_out`: Conv2d(inner, C, k=1).
    pub proj_out: Conv2d<T>,
    channels: usize,
    inner_dim: usize,
    training: bool,
}

impl<T: Float> Transformer2DModel<T> {
    /// Build a randomly-initialized `Transformer2DModel`.
    ///
    /// `inner_dim = heads * dim_head` for the SD UNet (proj_in expands
    /// only when these disagree; for SD it's always equal to
    /// `in_channels`).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] for invalid dims.
    pub fn new(
        in_channels: usize,
        heads: usize,
        dim_head: usize,
        num_layers: usize,
        cross_attention_dim: usize,
        norm_num_groups: usize,
    ) -> FerrotorchResult<Self> {
        let inner_dim = heads * dim_head;
        let norm = GroupNorm::<T>::new(norm_num_groups, in_channels, 1e-6, true)?;
        let proj_in = Conv2d::<T>::new(in_channels, inner_dim, (1, 1), (1, 1), (0, 0), true)?;
        let proj_out = Conv2d::<T>::new(inner_dim, in_channels, (1, 1), (1, 1), (0, 0), true)?;
        let mut transformer_blocks = Vec::with_capacity(num_layers);
        for _ in 0..num_layers {
            transformer_blocks.push(BasicTransformerBlock::<T>::new(
                inner_dim,
                heads,
                dim_head,
                cross_attention_dim,
            )?);
        }
        Ok(Self {
            norm,
            proj_in,
            transformer_blocks,
            proj_out,
            channels: in_channels,
            inner_dim,
            training: false,
        })
    }

    /// Forward with encoder hidden states for cross-attn.
    ///
    /// `x` has shape `[B, C, H, W]`. The result has the same shape.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] when the input is not
    /// `[B, channels, H, W]`.
    pub fn forward_xattn(
        &self,
        x: &Tensor<T>,
        encoder_hidden_states: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if x.ndim() != 4 || x.shape()[1] != self.channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Transformer2DModel::forward: expected [B, {}, H, W], got {:?}",
                    self.channels,
                    x.shape()
                ),
            });
        }
        let b = x.shape()[0];
        let c = x.shape()[1];
        let h = x.shape()[2];
        let w = x.shape()[3];
        let hw = h * w;

        let residual = x.clone();
        // norm + proj_in (Conv2d k=1) keeps the [B, C', H, W] layout.
        let mut hidden = self.norm.forward(x)?;
        hidden = self.proj_in.forward(&hidden)?;
        // [B, inner, H, W] -> [B, inner, HW] -> [B, HW, inner]
        let mut hidden_seq = hidden
            .reshape_t(&[b as isize, self.inner_dim as isize, hw as isize])?
            .transpose(1, 2)?
            .contiguous()?;
        // Run the transformer blocks.
        for block in &self.transformer_blocks {
            hidden_seq = block.forward_xattn(&hidden_seq, encoder_hidden_states)?;
        }
        // Back to spatial: [B, HW, inner] -> [B, inner, HW] -> [B, inner, H, W]
        let hidden_back = hidden_seq
            .transpose(1, 2)?
            .reshape_t(&[b as isize, self.inner_dim as isize, h as isize, w as isize])?
            .contiguous()?;
        // proj_out (Conv2d k=1) + residual.
        let out = self.proj_out.forward(&hidden_back)?;
        let _ = c;
        ferrotorch_core::grad_fns::arithmetic::add(&out, &residual)
    }
}

impl<T: Float> Module<T> for Transformer2DModel<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "Transformer2DModel::forward: cross-attn requires \
                      encoder_hidden_states — call forward_xattn instead"
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.norm.parameters());
        o.extend(self.proj_in.parameters());
        for b in &self.transformer_blocks {
            o.extend(b.parameters());
        }
        o.extend(self.proj_out.parameters());
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.norm.parameters_mut());
        o.extend(self.proj_in.parameters_mut());
        for b in &mut self.transformer_blocks {
            o.extend(b.parameters_mut());
        }
        o.extend(self.proj_out.parameters_mut());
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (n, p) in self.norm.named_parameters() {
            o.push((format!("norm.{n}"), p));
        }
        for (n, p) in self.proj_in.named_parameters() {
            o.push((format!("proj_in.{n}"), p));
        }
        for (i, b) in self.transformer_blocks.iter().enumerate() {
            for (n, p) in b.named_parameters() {
                o.push((format!("transformer_blocks.{i}.{n}"), p));
            }
        }
        for (n, p) in self.proj_out.named_parameters() {
            o.push((format!("proj_out.{n}"), p));
        }
        o
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
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let p = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| k.strip_prefix(&p).map(|r| (r.to_string(), v.clone())))
                .collect()
        };
        if strict {
            for k in state.keys() {
                let ok = k.starts_with("norm.")
                    || k.starts_with("proj_in.")
                    || k.starts_with("transformer_blocks.")
                    || k.starts_with("proj_out.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in Transformer2DModel state_dict: \"{k}\""
                        ),
                    });
                }
            }
        }
        self.norm.load_state_dict(&extract("norm"), strict)?;
        self.proj_in.load_state_dict(&extract("proj_in"), strict)?;
        for (i, b) in self.transformer_blocks.iter_mut().enumerate() {
            b.load_state_dict(&extract(&format!("transformer_blocks.{i}")), strict)?;
        }
        self.proj_out
            .load_state_dict(&extract("proj_out"), strict)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    #[test]
    fn attention_self_shape() {
        let a = Attention::<f32>::new(16, None, 4, 4, false).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 5 * 16]),
            vec![1, 5, 16],
            false,
        )
        .unwrap();
        let y = a.forward_xattn(&x, None).unwrap();
        assert_eq!(y.shape(), &[1, 5, 16]);
    }

    #[test]
    fn attention_cross_shape() {
        let a = Attention::<f32>::new(16, Some(24), 4, 4, false).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 5 * 16]),
            vec![1, 5, 16],
            false,
        )
        .unwrap();
        let ehs = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 7 * 24]),
            vec![1, 7, 24],
            false,
        )
        .unwrap();
        let y = a.forward_xattn(&x, Some(&ehs)).unwrap();
        assert_eq!(y.shape(), &[1, 5, 16]);
    }

    #[test]
    fn feedforward_shape_and_keys() {
        let ff = FeedForward::<f32>::new(16, 2).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 5 * 16]),
            vec![1, 5, 16],
            false,
        )
        .unwrap();
        let y = ff.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 5, 16]);
        let names: Vec<String> = ff.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in ["net.0.proj.weight", "net.0.proj.bias", "net.2.weight", "net.2.bias"] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }

    #[test]
    fn basic_transformer_block_shape() {
        let blk = BasicTransformerBlock::<f32>::new(16, 4, 4, 24).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 5 * 16]),
            vec![1, 5, 16],
            false,
        )
        .unwrap();
        let ehs = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 7 * 24]),
            vec![1, 7, 24],
            false,
        )
        .unwrap();
        let y = blk.forward_xattn(&x, &ehs).unwrap();
        assert_eq!(y.shape(), &[1, 5, 16]);
    }

    #[test]
    fn transformer_2d_shape() {
        let t = Transformer2DModel::<f32>::new(16, 4, 4, 1, 24, 4).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 16 * 3 * 3]),
            vec![1, 16, 3, 3],
            false,
        )
        .unwrap();
        let ehs = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 5 * 24]),
            vec![1, 5, 24],
            false,
        )
        .unwrap();
        let y = t.forward_xattn(&x, &ehs).unwrap();
        assert_eq!(y.shape(), &[1, 16, 3, 3]);
    }

    #[test]
    fn transformer_2d_named_parameters() {
        let t = Transformer2DModel::<f32>::new(16, 4, 4, 1, 24, 4).unwrap();
        let names: Vec<String> = t.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "norm.weight",
            "proj_in.weight",
            "proj_in.bias",
            "transformer_blocks.0.norm1.weight",
            "transformer_blocks.0.attn1.to_q.weight",
            "transformer_blocks.0.attn2.to_k.weight",
            "transformer_blocks.0.ff.net.0.proj.weight",
            "transformer_blocks.0.ff.net.2.weight",
            "proj_out.weight",
        ] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }
}
