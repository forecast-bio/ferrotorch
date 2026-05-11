//! Building blocks of the Stable-Diffusion VAE decoder.
//!
//! All blocks here match the diffusers `models/{resnet,upsampling,
//! attention_processor}.py` / `models/unets/unet_2d_blocks.py` reference
//! layout 1:1 in parameter naming and forward semantics so the upstream
//! state dict (`runwayml/stable-diffusion-v1-5/vae/diffusion_pytorch_model.safetensors`)
//! loads byte-for-byte.

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{
    Conv2d, GroupNorm, InterpolateMode, Linear, SiLU, Upsample,
};

// ---------------------------------------------------------------------------
// ResnetBlock2D
// ---------------------------------------------------------------------------

/// `ResnetBlock2D` — the building block of every UNet/VAE up/down/mid stack.
///
/// The VAE flavour has no time embedding (`temb_channels=None` in
/// diffusers), so this implementation does not carry a
/// `time_emb_proj`. The forward pass is:
///
/// ```text
/// h = norm1(x); h = silu(h); h = conv1(h)
/// h = norm2(h); h = silu(h); h = conv2(h)
/// out = h + (x if in==out else conv_shortcut(x))
/// ```
///
/// The `output_scale_factor` from the diffusers reference is always 1.0
/// in the SD VAE so we hard-code it; if a future config requires
/// rescaling, add an explicit field.
#[derive(Debug)]
pub struct ResnetBlock2D<T: Float> {
    /// First GroupNorm (over `in_channels`).
    pub norm1: GroupNorm<T>,
    /// First Conv2d (`in_channels -> out_channels`, k=3, pad=1).
    pub conv1: Conv2d<T>,
    /// Second GroupNorm (over `out_channels`).
    pub norm2: GroupNorm<T>,
    /// Second Conv2d (`out_channels -> out_channels`, k=3, pad=1).
    pub conv2: Conv2d<T>,
    /// Optional 1x1 shortcut conv (present iff `in_channels !=
    /// out_channels`).
    pub conv_shortcut: Option<Conv2d<T>>,
    activation: SiLU,
    in_channels: usize,
    #[allow(dead_code)]
    out_channels: usize,
    training: bool,
}

impl<T: Float> ResnetBlock2D<T> {
    /// Build a randomly-initialized `ResnetBlock2D`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] on bad channel /
    /// group config.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        norm_num_groups: usize,
        eps: f64,
    ) -> FerrotorchResult<Self> {
        let norm1 = GroupNorm::<T>::new(norm_num_groups, in_channels, eps, true)?;
        let conv1 = Conv2d::<T>::new(in_channels, out_channels, (3, 3), (1, 1), (1, 1), true)?;
        let norm2 = GroupNorm::<T>::new(norm_num_groups, out_channels, eps, true)?;
        let conv2 = Conv2d::<T>::new(out_channels, out_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv_shortcut = if in_channels == out_channels {
            None
        } else {
            Some(Conv2d::<T>::new(
                in_channels,
                out_channels,
                (1, 1),
                (1, 1),
                (0, 0),
                true,
            )?)
        };
        Ok(Self {
            norm1,
            conv1,
            norm2,
            conv2,
            conv_shortcut,
            activation: SiLU::new(),
            in_channels,
            out_channels,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for ResnetBlock2D<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 4 || input.shape()[1] != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ResnetBlock2D::forward: expected [B, {}, H, W], got {:?}",
                    self.in_channels,
                    input.shape()
                ),
            });
        }
        // h = norm1(x); silu; conv1.
        let mut h = self.norm1.forward(input)?;
        h = self.activation.forward(&h)?;
        h = self.conv1.forward(&h)?;
        // h = norm2(h); silu; conv2 (no dropout at eval / inference time).
        h = self.norm2.forward(&h)?;
        h = self.activation.forward(&h)?;
        h = self.conv2.forward(&h)?;
        // Residual (optionally projected via 1x1 conv).
        let res = if let Some(sc) = &self.conv_shortcut {
            sc.forward(input)?
        } else {
            input.clone()
        };
        // out = h + res. `output_scale_factor=1.0` in the SD VAE so no
        // division on the way out.
        ferrotorch_core::grad_fns::arithmetic::add(&h, &res)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.norm1.parameters());
        out.extend(self.conv1.parameters());
        out.extend(self.norm2.parameters());
        out.extend(self.conv2.parameters());
        if let Some(sc) = &self.conv_shortcut {
            out.extend(sc.parameters());
        }
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.norm1.parameters_mut());
        out.extend(self.conv1.parameters_mut());
        out.extend(self.norm2.parameters_mut());
        out.extend(self.conv2.parameters_mut());
        if let Some(sc) = &mut self.conv_shortcut {
            out.extend(sc.parameters_mut());
        }
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.norm1.named_parameters() {
            out.push((format!("norm1.{n}"), p));
        }
        for (n, p) in self.conv1.named_parameters() {
            out.push((format!("conv1.{n}"), p));
        }
        for (n, p) in self.norm2.named_parameters() {
            out.push((format!("norm2.{n}"), p));
        }
        for (n, p) in self.conv2.named_parameters() {
            out.push((format!("conv2.{n}"), p));
        }
        if let Some(sc) = &self.conv_shortcut {
            for (n, p) in sc.named_parameters() {
                out.push((format!("conv_shortcut.{n}"), p));
            }
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
                    || k.starts_with("conv1.")
                    || k.starts_with("norm2.")
                    || k.starts_with("conv2.")
                    || k.starts_with("conv_shortcut.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in ResnetBlock2D state_dict: \"{k}\""),
                    });
                }
            }
        }
        self.norm1.load_state_dict(&extract("norm1"), strict)?;
        self.conv1.load_state_dict(&extract("conv1"), strict)?;
        self.norm2.load_state_dict(&extract("norm2"), strict)?;
        self.conv2.load_state_dict(&extract("conv2"), strict)?;
        if let Some(sc) = self.conv_shortcut.as_mut() {
            sc.load_state_dict(&extract("conv_shortcut"), strict)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// AttnBlock2D
// ---------------------------------------------------------------------------

/// Single-head spatial self-attention with residual + GroupNorm — the
/// VAE mid-block attention.
///
/// Matches `diffusers.models.attention_processor.Attention` configured
/// with:
///
///   - `heads = in_channels / attention_head_dim = 512 / 512 = 1`
///   - `norm_num_groups = 32` (so `group_norm` is enabled)
///   - `residual_connection = True`
///   - `bias = True`, `out_bias = True`
///   - `eps = 1e-6`
///
/// State-dict layout (HF diffusers):
///
/// ```text
/// group_norm.{weight,bias}    [C], [C]
/// to_q.{weight,bias}          [C, C], [C]
/// to_k.{weight,bias}          [C, C], [C]
/// to_v.{weight,bias}          [C, C], [C]
/// to_out.0.{weight,bias}      [C, C], [C]      // to_out[1] is Dropout (no params)
/// ```
#[derive(Debug)]
pub struct AttnBlock2D<T: Float> {
    /// GroupNorm over the channel axis.
    pub group_norm: GroupNorm<T>,
    /// Query projection `Linear(C -> C, bias)`.
    pub to_q: Linear<T>,
    /// Key projection `Linear(C -> C, bias)`.
    pub to_k: Linear<T>,
    /// Value projection `Linear(C -> C, bias)`.
    pub to_v: Linear<T>,
    /// Output projection `to_out[0] = Linear(C -> C, bias)`.
    pub to_out_0: Linear<T>,
    channels: usize,
    training: bool,
}

impl<T: Float> AttnBlock2D<T> {
    /// Build a randomly-initialized `AttnBlock2D`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when `channels`
    /// is not divisible by `norm_num_groups` (the GroupNorm constructor
    /// surfaces this).
    pub fn new(channels: usize, norm_num_groups: usize, eps: f64) -> FerrotorchResult<Self> {
        let group_norm = GroupNorm::<T>::new(norm_num_groups, channels, eps, true)?;
        let to_q = Linear::<T>::new(channels, channels, true)?;
        let to_k = Linear::<T>::new(channels, channels, true)?;
        let to_v = Linear::<T>::new(channels, channels, true)?;
        let to_out_0 = Linear::<T>::new(channels, channels, true)?;
        Ok(Self {
            group_norm,
            to_q,
            to_k,
            to_v,
            to_out_0,
            channels,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for AttnBlock2D<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 4 || input.shape()[1] != self.channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "AttnBlock2D::forward: expected [B, {}, H, W], got {:?}",
                    self.channels,
                    input.shape()
                ),
            });
        }
        let b = input.shape()[0];
        let c = input.shape()[1];
        let h = input.shape()[2];
        let w = input.shape()[3];
        let hw = h * w;

        // Residual (kept in the spatial layout).
        let residual = input.clone();

        // -- 1. Reshape to [B, HW, C]:
        //       diffusers does `view(B, C, HW).transpose(1, 2)`.
        //       Then it group-norms on the channel axis by re-transposing
        //       back to [B, C, HW], norming, and re-transposing.
        let hidden = input
            .reshape_t(&[b as isize, c as isize, hw as isize])?
            .transpose(1, 2)?
            .contiguous()?;
        // GroupNorm: input [B, C, HW] -> [B, HW, C] after the final transpose.
        let normed_hwc = self
            .group_norm
            .forward(&hidden.transpose(1, 2)?.contiguous()?)?
            .transpose(1, 2)?
            .contiguous()?;

        // -- 2. q / k / v projections (Linear over the trailing C dim).
        let q = self.to_q.forward(&normed_hwc)?; // [B, HW, C]
        let k = self.to_k.forward(&normed_hwc)?; // [B, HW, C]
        let v = self.to_v.forward(&normed_hwc)?; // [B, HW, C]

        // -- 3. Single-head attention:
        //          scores = q @ k^T * (1/sqrt(C))
        //          probs  = softmax(scores, dim=-1)
        //          out    = probs @ v
        //       For single-head the head-split is a no-op, so this is
        //       just a per-batch BMM. `bmm` handles the [B, HW, HW]
        //       intermediate without materialising a head axis.
        let scale = (c as f64).sqrt().recip();
        let scale_t = T::from(scale).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "AttnBlock2D::forward: failed to cast attention scale into Float".into(),
        })?;
        let scale_tensor = ferrotorch_core::scalar::<T>(scale_t)?;
        let k_t = k.transpose(1, 2)?.contiguous()?; // [B, C, HW]
        let scores = q.bmm(&k_t)?; // [B, HW, HW]
        let scores_scaled =
            ferrotorch_core::grad_fns::arithmetic::mul(&scores, &scale_tensor)?;
        let probs = scores_scaled.softmax()?; // dim = -1 by default
        let attended = probs.bmm(&v)?; // [B, HW, C]

        // -- 4. Output projection.
        let projected = self.to_out_0.forward(&attended)?; // [B, HW, C]

        // -- 5. Back to [B, C, H, W] and add residual.
        let back = projected
            .transpose(1, 2)? // [B, C, HW]
            .reshape_t(&[b as isize, c as isize, h as isize, w as isize])?
            .contiguous()?;
        ferrotorch_core::grad_fns::arithmetic::add(&back, &residual)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.group_norm.parameters());
        out.extend(self.to_q.parameters());
        out.extend(self.to_k.parameters());
        out.extend(self.to_v.parameters());
        out.extend(self.to_out_0.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.group_norm.parameters_mut());
        out.extend(self.to_q.parameters_mut());
        out.extend(self.to_k.parameters_mut());
        out.extend(self.to_v.parameters_mut());
        out.extend(self.to_out_0.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.group_norm.named_parameters() {
            out.push((format!("group_norm.{n}"), p));
        }
        for (n, p) in self.to_q.named_parameters() {
            out.push((format!("to_q.{n}"), p));
        }
        for (n, p) in self.to_k.named_parameters() {
            out.push((format!("to_k.{n}"), p));
        }
        for (n, p) in self.to_v.named_parameters() {
            out.push((format!("to_v.{n}"), p));
        }
        for (n, p) in self.to_out_0.named_parameters() {
            out.push((format!("to_out.0.{n}"), p));
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
                let ok = k.starts_with("group_norm.")
                    || k.starts_with("to_q.")
                    || k.starts_with("to_k.")
                    || k.starts_with("to_v.")
                    || k.starts_with("to_out.0.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in AttnBlock2D state_dict: \"{k}\""),
                    });
                }
            }
        }
        self.group_norm
            .load_state_dict(&extract("group_norm"), strict)?;
        self.to_q.load_state_dict(&extract("to_q"), strict)?;
        self.to_k.load_state_dict(&extract("to_k"), strict)?;
        self.to_v.load_state_dict(&extract("to_v"), strict)?;
        self.to_out_0
            .load_state_dict(&extract("to_out.0"), strict)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Upsample2D
// ---------------------------------------------------------------------------

/// Diffusers-style `Upsample2D` — nearest-neighbor 2x interpolation
/// followed by a `Conv2d(C, C, k=3, pad=1, bias=True)`.
///
/// State-dict key: `conv.{weight,bias}` (matching `name="conv"` in the
/// diffusers reference).
#[derive(Debug)]
pub struct Upsample2D<T: Float> {
    /// Output conv (also serves as the "post-upsample smoothing" filter).
    pub conv: Conv2d<T>,
    channels: usize,
    training: bool,
}

impl<T: Float> Upsample2D<T> {
    /// Build a randomly-initialized `Upsample2D` (`C -> C`, k=3, pad=1).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad conv config.
    pub fn new(channels: usize) -> FerrotorchResult<Self> {
        let conv = Conv2d::<T>::new(channels, channels, (3, 3), (1, 1), (1, 1), true)?;
        Ok(Self {
            conv,
            channels,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for Upsample2D<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 4 || input.shape()[1] != self.channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Upsample2D::forward: expected [B, {}, H, W], got {:?}",
                    self.channels,
                    input.shape()
                ),
            });
        }
        let h = input.shape()[2];
        let w = input.shape()[3];
        let up = Upsample::new([h * 2, w * 2], InterpolateMode::Nearest);
        let upsampled = up.forward(input)?;
        self.conv.forward(&upsampled)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.conv.parameters()
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.conv.parameters_mut()
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.conv
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (format!("conv.{n}"), p))
            .collect()
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
        let conv_sd: StateDict<T> = state
            .iter()
            .filter_map(|(k, v)| k.strip_prefix("conv.").map(|r| (r.to_string(), v.clone())))
            .collect();
        if strict {
            for k in state.keys() {
                if !k.starts_with("conv.") {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in Upsample2D state_dict: \"{k}\""),
                    });
                }
            }
        }
        self.conv.load_state_dict(&conv_sd, strict)
    }
}

// ---------------------------------------------------------------------------
// Downsample2D
// ---------------------------------------------------------------------------

/// Diffusers-style `Downsample2D` — a single `Conv2d(C, C, k=3, stride=2,
/// pad=1, bias=True)`.
///
/// SD-1.5 uses `use_conv=True` and `padding=1`, so there is no separate
/// pre-conv padding step. State-dict key: `conv.{weight,bias}`.
#[derive(Debug)]
pub struct Downsample2D<T: Float> {
    /// Output conv (k=3, stride=2, pad=1).
    pub conv: Conv2d<T>,
    channels: usize,
    training: bool,
}

impl<T: Float> Downsample2D<T> {
    /// Build a randomly-initialized `Downsample2D` (`C -> C`).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad conv config.
    pub fn new(channels: usize) -> FerrotorchResult<Self> {
        let conv = Conv2d::<T>::new(channels, channels, (3, 3), (2, 2), (1, 1), true)?;
        Ok(Self {
            conv,
            channels,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for Downsample2D<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 4 || input.shape()[1] != self.channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Downsample2D::forward: expected [B, {}, H, W], got {:?}",
                    self.channels,
                    input.shape()
                ),
            });
        }
        self.conv.forward(input)
    }
    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.conv.parameters()
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.conv.parameters_mut()
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.conv
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (format!("conv.{n}"), p))
            .collect()
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
        let conv_sd: StateDict<T> = state
            .iter()
            .filter_map(|(k, v)| k.strip_prefix("conv.").map(|r| (r.to_string(), v.clone())))
            .collect();
        if strict {
            for k in state.keys() {
                if !k.starts_with("conv.") {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in Downsample2D state_dict: \"{k}\""),
                    });
                }
            }
        }
        self.conv.load_state_dict(&conv_sd, strict)
    }
}

// ---------------------------------------------------------------------------
// UpDecoderBlock2D
// ---------------------------------------------------------------------------

/// `UpDecoderBlock2D` — a stack of `layers_per_block + 1` resnets at
/// `out_channels`, optionally followed by an `Upsample2D`.
///
/// State-dict key layout:
///
/// ```text
/// resnets.{i}.<resnet keys>    i in 0..(layers_per_block + 1)
/// upsamplers.0.<upsample keys>  (present iff add_upsample)
/// ```
#[derive(Debug)]
pub struct UpDecoderBlock2D<T: Float> {
    /// `layers_per_block + 1` resnets at `out_channels`.
    pub resnets: Vec<ResnetBlock2D<T>>,
    /// Optional upsample (absent on the last decoder block).
    pub upsamplers_0: Option<Upsample2D<T>>,
    training: bool,
}

impl<T: Float> UpDecoderBlock2D<T> {
    /// Build a randomly-initialized `UpDecoderBlock2D`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] on bad channel /
    /// group config.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        num_resnets: usize,
        norm_num_groups: usize,
        eps: f64,
        add_upsample: bool,
    ) -> FerrotorchResult<Self> {
        if num_resnets == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "UpDecoderBlock2D: num_resnets must be > 0".into(),
            });
        }
        let mut resnets = Vec::with_capacity(num_resnets);
        for i in 0..num_resnets {
            let in_c = if i == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2D::<T>::new(
                in_c,
                out_channels,
                norm_num_groups,
                eps,
            )?);
        }
        let upsamplers_0 = if add_upsample {
            Some(Upsample2D::<T>::new(out_channels)?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            upsamplers_0,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for UpDecoderBlock2D<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut h = input.clone();
        for r in &self.resnets {
            h = r.forward(&h)?;
        }
        if let Some(u) = &self.upsamplers_0 {
            h = u.forward(&h)?;
        }
        Ok(h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        for r in &self.resnets {
            out.extend(r.parameters());
        }
        if let Some(u) = &self.upsamplers_0 {
            out.extend(u.parameters());
        }
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        for r in &mut self.resnets {
            out.extend(r.parameters_mut());
        }
        if let Some(u) = self.upsamplers_0.as_mut() {
            out.extend(u.parameters_mut());
        }
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (i, r) in self.resnets.iter().enumerate() {
            for (n, p) in r.named_parameters() {
                out.push((format!("resnets.{i}.{n}"), p));
            }
        }
        if let Some(u) = &self.upsamplers_0 {
            for (n, p) in u.named_parameters() {
                out.push((format!("upsamplers.0.{n}"), p));
            }
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        for r in &mut self.resnets {
            r.train();
        }
        if let Some(u) = self.upsamplers_0.as_mut() {
            u.train();
        }
    }
    fn eval(&mut self) {
        self.training = false;
        for r in &mut self.resnets {
            r.eval();
        }
        if let Some(u) = self.upsamplers_0.as_mut() {
            u.eval();
        }
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
                let ok =
                    k.starts_with("resnets.") || k.starts_with("upsamplers.0.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in UpDecoderBlock2D state_dict: \"{k}\""
                        ),
                    });
                }
            }
        }
        for (i, r) in self.resnets.iter_mut().enumerate() {
            r.load_state_dict(&extract(&format!("resnets.{i}")), strict)?;
        }
        if let Some(u) = self.upsamplers_0.as_mut() {
            u.load_state_dict(&extract("upsamplers.0"), strict)?;
        } else if strict {
            // Verify no upsamplers keys leaked through if we have no upsample.
            for k in state.keys() {
                if k.starts_with("upsamplers.") {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "UpDecoderBlock2D has no upsampler but state_dict contains \"{k}\""
                        ),
                    });
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// UNetMidBlock2D (VAE flavour)
// ---------------------------------------------------------------------------

/// `UNetMidBlock2D` configured the way the SD VAE uses it:
///
/// ```text
/// resnets[0] -> attentions[0] -> resnets[1]
/// ```
///
/// 2 resnets, 1 attention; all at `in_channels = out_channels = 512` for
/// SD 1.5. Diffusers always has `len(resnets) == num_layers + 1` and
/// `len(attentions) == num_layers`; in the VAE `num_layers = 1`, so two
/// resnets and one attention.
#[derive(Debug)]
pub struct UNetMidBlock2D<T: Float> {
    /// Pre-attention + post-attention resnets (size 2 for SD VAE).
    pub resnets: Vec<ResnetBlock2D<T>>,
    /// Single attention block.
    pub attentions: Vec<AttnBlock2D<T>>,
    channels: usize,
    training: bool,
}

impl<T: Float> UNetMidBlock2D<T> {
    /// Build a randomly-initialized VAE-flavoured `UNetMidBlock2D`.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config.
    pub fn new(channels: usize, norm_num_groups: usize, eps: f64) -> FerrotorchResult<Self> {
        // SD VAE: num_layers=1 => 2 resnets + 1 attention.
        let resnets = vec![
            ResnetBlock2D::<T>::new(channels, channels, norm_num_groups, eps)?,
            ResnetBlock2D::<T>::new(channels, channels, norm_num_groups, eps)?,
        ];
        let attentions = vec![AttnBlock2D::<T>::new(channels, norm_num_groups, eps)?];
        Ok(Self {
            resnets,
            attentions,
            channels,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for UNetMidBlock2D<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // diffusers UNetMidBlock2D.forward (no temb):
        //   h = resnets[0](x)
        //   for (attn, resnet) in zip(attentions, resnets[1:]):
        //       h = attn(h); h = resnet(h)
        let mut h = self.resnets[0].forward(input)?;
        for (attn, resnet) in self.attentions.iter().zip(self.resnets.iter().skip(1)) {
            h = attn.forward(&h)?;
            h = resnet.forward(&h)?;
        }
        Ok(h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        // The HF diffusers state-dict key order is `attentions` before
        // `resnets`. We match the ordering used for state-dict
        // round-trip — see `named_parameters()` for the canonical
        // sequence.
        for a in &self.attentions {
            out.extend(a.parameters());
        }
        for r in &self.resnets {
            out.extend(r.parameters());
        }
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        for a in &mut self.attentions {
            out.extend(a.parameters_mut());
        }
        for r in &mut self.resnets {
            out.extend(r.parameters_mut());
        }
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        // Match the order used by `safetensors` listings of the HF VAE:
        // `attentions.*` first, then `resnets.*`.
        for (i, a) in self.attentions.iter().enumerate() {
            for (n, p) in a.named_parameters() {
                out.push((format!("attentions.{i}.{n}"), p));
            }
        }
        for (i, r) in self.resnets.iter().enumerate() {
            for (n, p) in r.named_parameters() {
                out.push((format!("resnets.{i}.{n}"), p));
            }
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        for r in &mut self.resnets {
            r.train();
        }
        for a in &mut self.attentions {
            a.train();
        }
    }
    fn eval(&mut self) {
        self.training = false;
        for r in &mut self.resnets {
            r.eval();
        }
        for a in &mut self.attentions {
            a.eval();
        }
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
                let ok = k.starts_with("attentions.") || k.starts_with("resnets.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in UNetMidBlock2D state_dict: \"{k}\""
                        ),
                    });
                }
            }
        }
        for (i, a) in self.attentions.iter_mut().enumerate() {
            a.load_state_dict(&extract(&format!("attentions.{i}")), strict)?;
        }
        for (i, r) in self.resnets.iter_mut().enumerate() {
            r.load_state_dict(&extract(&format!("resnets.{i}")), strict)?;
        }
        let _ = self.channels; // silence dead_code warning under some lint configs
        let _: HashMap<String, Tensor<T>> = HashMap::new(); // pull HashMap into scope
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    #[test]
    fn resnet_same_channels_no_shortcut() {
        let r = ResnetBlock2D::<f32>::new(32, 32, 32, 1e-6).unwrap();
        assert!(r.conv_shortcut.is_none());
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 32 * 4 * 4]),
            vec![1, 32, 4, 4],
            false,
        )
        .unwrap();
        let y = r.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 32, 4, 4]);
        for &v in y.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn resnet_different_channels_has_shortcut() {
        let r = ResnetBlock2D::<f32>::new(32, 64, 32, 1e-6).unwrap();
        assert!(r.conv_shortcut.is_some());
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 32 * 4 * 4]),
            vec![1, 32, 4, 4],
            false,
        )
        .unwrap();
        let y = r.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 64, 4, 4]);
    }

    #[test]
    fn resnet_named_parameters_layout() {
        let r = ResnetBlock2D::<f32>::new(32, 64, 32, 1e-6).unwrap();
        let names: Vec<String> = r.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "norm1.weight",
            "norm1.bias",
            "conv1.weight",
            "conv1.bias",
            "norm2.weight",
            "norm2.bias",
            "conv2.weight",
            "conv2.bias",
            "conv_shortcut.weight",
            "conv_shortcut.bias",
        ] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }

    #[test]
    fn attn_shape_and_residual() {
        // 32 channels, 4 groups => 8 channels per group; small spatial.
        let a = AttnBlock2D::<f32>::new(32, 4, 1e-6).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 32 * 4 * 4]),
            vec![1, 32, 4, 4],
            false,
        )
        .unwrap();
        let y = a.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 32, 4, 4]);
        for &v in y.data().unwrap() {
            assert!(v.is_finite(), "attn output non-finite: {v}");
        }
    }

    #[test]
    fn attn_named_parameters_layout() {
        let a = AttnBlock2D::<f32>::new(32, 4, 1e-6).unwrap();
        let names: Vec<String> = a.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "group_norm.weight",
            "group_norm.bias",
            "to_q.weight",
            "to_q.bias",
            "to_k.weight",
            "to_k.bias",
            "to_v.weight",
            "to_v.bias",
            "to_out.0.weight",
            "to_out.0.bias",
        ] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }

    #[test]
    fn upsample2d_doubles_spatial() {
        let u = Upsample2D::<f32>::new(8).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.05f32; 8 * 3 * 3]),
            vec![1, 8, 3, 3],
            false,
        )
        .unwrap();
        let y = u.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 6, 6]);
    }

    #[test]
    fn up_decoder_block_shape_with_upsample() {
        let b = UpDecoderBlock2D::<f32>::new(16, 8, 2, 4, 1e-6, true).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.05f32; 16 * 3 * 3]),
            vec![1, 16, 3, 3],
            false,
        )
        .unwrap();
        let y = b.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 6, 6]);
    }

    #[test]
    fn up_decoder_block_shape_no_upsample() {
        let b = UpDecoderBlock2D::<f32>::new(8, 8, 2, 4, 1e-6, false).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.05f32; 8 * 3 * 3]),
            vec![1, 8, 3, 3],
            false,
        )
        .unwrap();
        let y = b.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 8, 3, 3]);
    }

    #[test]
    fn mid_block_shape() {
        let m = UNetMidBlock2D::<f32>::new(16, 4, 1e-6).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.05f32; 16 * 3 * 3]),
            vec![1, 16, 3, 3],
            false,
        )
        .unwrap();
        let y = m.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 16, 3, 3]);
    }

    #[test]
    fn mid_block_named_parameters_layout() {
        let m = UNetMidBlock2D::<f32>::new(16, 4, 1e-6).unwrap();
        let names: Vec<String> = m.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "attentions.0.group_norm.weight",
            "attentions.0.to_q.weight",
            "resnets.0.norm1.weight",
            "resnets.0.conv1.weight",
            "resnets.1.conv2.weight",
        ] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }
}
