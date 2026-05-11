//! Stable-Diffusion UNet2DConditionModel forward pass.
//!
//! Matches `diffusers.models.unets.UNet2DConditionModel.forward(sample,
//! timestep, encoder_hidden_states).sample` for
//! `runwayml/stable-diffusion-v1-5` (the
//! `down_block_types=[CrossAttn × 3, DownBlock2D]` / `up_block_types=
//! [UpBlock2D, CrossAttn × 3]` / `mid_block=UNetMidBlock2DCrossAttn`
//! topology).
//!
//! ```text
//! t  = time_proj(timesteps)                # [B, 320]
//! t  = time_embedding(t)                   # [B, 1280]
//! h0 = conv_in(sample)                     # [B, 320, H, W]
//! skips = [h0]
//! for db in down_blocks:                   # 4 down blocks
//!     h, skip = db(h, t, ehs)
//!     skips.extend(skip)
//! h = mid_block(h, t, ehs)
//! for ub in up_blocks:                     # 4 up blocks
//!     pop the trailing N skip tensors and concat at each resnet stage
//!     h = ub(h, t, ehs, skips_popped)
//! h = silu(conv_norm_out(h))
//! h = conv_out(h)                          # [B, 4, H, W]
//! ```
//!
//! State-dict key layout (matches diffusers byte-for-byte):
//!
//! ```text
//! time_embedding.linear_{1,2}.{weight,bias}
//! conv_in.{weight,bias}
//! down_blocks.{i}.{resnets,attentions,downsamplers}.<…>
//! mid_block.{resnets,attentions}.<…>
//! up_blocks.{i}.{resnets,attentions,upsamplers}.<…>
//! conv_norm_out.{weight,bias}
//! conv_out.{weight,bias}
//! ```
//!
//! Note: the time embedding has NO sinusoidal parameters (it's the
//! arithmetic-only `Timesteps` module followed by the trainable
//! `TimestepEmbedding` MLP).

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv2d, GroupNorm, SiLU};

use crate::attention::Transformer2DModel;
use crate::blocks::{Downsample2D, Upsample2D};
use crate::resnet_block_time::ResnetBlock2DTime;
use crate::time_embedding::{TimestepEmbedding, Timesteps};
use crate::unet_config::UNet2DConditionConfig;

// ---------------------------------------------------------------------------
// CrossAttnDownBlock2D
// ---------------------------------------------------------------------------

/// `CrossAttnDownBlock2D` — `layers_per_block` × (ResnetBlock2DTime +
/// Transformer2DModel) + optional Downsample2D.
///
/// State-dict layout:
///
/// ```text
/// resnets.{j}.<keys>          j in 0..layers_per_block
/// attentions.{j}.<keys>       j in 0..layers_per_block
/// downsamplers.0.<keys>        present iff add_downsample
/// ```
#[derive(Debug)]
pub struct CrossAttnDownBlock2D<T: Float> {
    /// `layers_per_block` time-conditioned resnets.
    pub resnets: Vec<ResnetBlock2DTime<T>>,
    /// `layers_per_block` transformer blocks (cross-attn).
    pub attentions: Vec<Transformer2DModel<T>>,
    /// Optional downsampler.
    pub downsamplers_0: Option<Downsample2D<T>>,
    training: bool,
}

impl<T: Float> CrossAttnDownBlock2D<T> {
    /// Build a randomly-initialized cross-attention down-block.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        num_layers: usize,
        attention_head_dim: usize,
        cross_attention_dim: usize,
        norm_num_groups: usize,
        add_downsample: bool,
        transformer_layers_per_block: usize,
    ) -> FerrotorchResult<Self> {
        // Diffusers footgun: for SD-1.5 the `attention_head_dim` config
        // entry is actually the *number of heads*, not the per-head
        // dimension. `num_attention_heads` was added later and defaults
        // to `attention_head_dim` when unset (see
        // diffusers/models/unets/unet_2d_condition.py: `num_attention_heads
        // = num_attention_heads or attention_head_dim`). The per-head
        // dimension is then `out_channels // num_heads`.
        let heads = attention_head_dim;
        let dim_head = out_channels / heads;
        let mut resnets = Vec::with_capacity(num_layers);
        let mut attentions = Vec::with_capacity(num_layers);
        for j in 0..num_layers {
            let in_c = if j == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2DTime::<T>::new(
                in_c,
                out_channels,
                temb_channels,
                norm_num_groups,
                1e-5,
            )?);
            attentions.push(Transformer2DModel::<T>::new(
                out_channels,
                heads,
                dim_head,
                transformer_layers_per_block,
                cross_attention_dim,
                norm_num_groups,
            )?);
        }
        let downsamplers_0 = if add_downsample {
            Some(Downsample2D::<T>::new(out_channels)?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            downsamplers_0,
            training: false,
        })
    }

    /// Forward; returns `(output, [skip_after_each_resnet+optional_downsample])`.
    ///
    /// Skips are pushed in diffusers order — every resnet+attn pair
    /// emits one skip, and the downsampler (if present) emits the post-
    /// downsample tensor as the trailing skip.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    pub fn forward_t(
        &self,
        x: &Tensor<T>,
        temb: &Tensor<T>,
        encoder_hidden_states: &Tensor<T>,
    ) -> FerrotorchResult<(Tensor<T>, Vec<Tensor<T>>)> {
        let mut h = x.clone();
        let mut skips = Vec::with_capacity(self.resnets.len() + 1);
        for (r, a) in self.resnets.iter().zip(self.attentions.iter()) {
            h = r.forward_t(&h, temb)?;
            h = a.forward_xattn(&h, encoder_hidden_states)?;
            skips.push(h.clone());
        }
        if let Some(d) = &self.downsamplers_0 {
            h = d.forward(&h)?;
            skips.push(h.clone());
        }
        Ok((h, skips))
    }

    fn named_params_internal(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (i, r) in self.resnets.iter().enumerate() {
            for (n, p) in r.named_parameters() {
                o.push((format!("resnets.{i}.{n}"), p));
            }
        }
        for (i, a) in self.attentions.iter().enumerate() {
            for (n, p) in a.named_parameters() {
                o.push((format!("attentions.{i}.{n}"), p));
            }
        }
        if let Some(d) = &self.downsamplers_0 {
            for (n, p) in d.named_parameters() {
                o.push((format!("downsamplers.0.{n}"), p));
            }
        }
        o
    }
}

impl<T: Float> Module<T> for CrossAttnDownBlock2D<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "CrossAttnDownBlock2D::forward requires (x, temb, ehs) — call \
                      forward_t instead"
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        for r in &self.resnets {
            o.extend(r.parameters());
        }
        for a in &self.attentions {
            o.extend(a.parameters());
        }
        if let Some(d) = &self.downsamplers_0 {
            o.extend(d.parameters());
        }
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        for r in &mut self.resnets {
            o.extend(r.parameters_mut());
        }
        for a in &mut self.attentions {
            o.extend(a.parameters_mut());
        }
        if let Some(d) = self.downsamplers_0.as_mut() {
            o.extend(d.parameters_mut());
        }
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.named_params_internal()
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
                let ok = k.starts_with("resnets.")
                    || k.starts_with("attentions.")
                    || k.starts_with("downsamplers.0.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in CrossAttnDownBlock2D state_dict: \"{k}\""
                        ),
                    });
                }
            }
        }
        for (i, r) in self.resnets.iter_mut().enumerate() {
            r.load_state_dict(&extract(&format!("resnets.{i}")), strict)?;
        }
        for (i, a) in self.attentions.iter_mut().enumerate() {
            a.load_state_dict(&extract(&format!("attentions.{i}")), strict)?;
        }
        if let Some(d) = self.downsamplers_0.as_mut() {
            d.load_state_dict(&extract("downsamplers.0"), strict)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// DownBlock2D (no cross-attn — the final SD down-block)
// ---------------------------------------------------------------------------

/// `DownBlock2D` — `layers_per_block` ResnetBlock2DTime + optional
/// Downsample2D. No transformer attention.
#[derive(Debug)]
pub struct DownBlock2D<T: Float> {
    /// `layers_per_block` time-conditioned resnets.
    pub resnets: Vec<ResnetBlock2DTime<T>>,
    /// Optional downsampler.
    pub downsamplers_0: Option<Downsample2D<T>>,
    training: bool,
}

impl<T: Float> DownBlock2D<T> {
    /// Build a randomly-initialized non-cross-attn down-block.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        temb_channels: usize,
        num_layers: usize,
        norm_num_groups: usize,
        add_downsample: bool,
    ) -> FerrotorchResult<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for j in 0..num_layers {
            let in_c = if j == 0 { in_channels } else { out_channels };
            resnets.push(ResnetBlock2DTime::<T>::new(
                in_c,
                out_channels,
                temb_channels,
                norm_num_groups,
                1e-5,
            )?);
        }
        let downsamplers_0 = if add_downsample {
            Some(Downsample2D::<T>::new(out_channels)?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            downsamplers_0,
            training: false,
        })
    }

    /// Forward; returns `(output, skips)`.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    pub fn forward_t(
        &self,
        x: &Tensor<T>,
        temb: &Tensor<T>,
    ) -> FerrotorchResult<(Tensor<T>, Vec<Tensor<T>>)> {
        let mut h = x.clone();
        let mut skips = Vec::with_capacity(self.resnets.len() + 1);
        for r in &self.resnets {
            h = r.forward_t(&h, temb)?;
            skips.push(h.clone());
        }
        if let Some(d) = &self.downsamplers_0 {
            h = d.forward(&h)?;
            skips.push(h.clone());
        }
        Ok((h, skips))
    }
}

impl<T: Float> Module<T> for DownBlock2D<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "DownBlock2D::forward requires (x, temb) — call forward_t instead".into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        for r in &self.resnets {
            o.extend(r.parameters());
        }
        if let Some(d) = &self.downsamplers_0 {
            o.extend(d.parameters());
        }
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        for r in &mut self.resnets {
            o.extend(r.parameters_mut());
        }
        if let Some(d) = self.downsamplers_0.as_mut() {
            o.extend(d.parameters_mut());
        }
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (i, r) in self.resnets.iter().enumerate() {
            for (n, p) in r.named_parameters() {
                o.push((format!("resnets.{i}.{n}"), p));
            }
        }
        if let Some(d) = &self.downsamplers_0 {
            for (n, p) in d.named_parameters() {
                o.push((format!("downsamplers.0.{n}"), p));
            }
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
                let ok = k.starts_with("resnets.") || k.starts_with("downsamplers.0.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in DownBlock2D state_dict: \"{k}\""),
                    });
                }
            }
        }
        for (i, r) in self.resnets.iter_mut().enumerate() {
            r.load_state_dict(&extract(&format!("resnets.{i}")), strict)?;
        }
        if let Some(d) = self.downsamplers_0.as_mut() {
            d.load_state_dict(&extract("downsamplers.0"), strict)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// UNetMidBlock2DCrossAttn
// ---------------------------------------------------------------------------

/// `UNetMidBlock2DCrossAttn` — the mid block of the SD UNet:
///
/// ```text
/// resnets[0] -> attentions[0] -> resnets[1]
/// ```
///
/// State-dict layout:
///
/// ```text
/// attentions.0.<keys>
/// resnets.0.<keys>
/// resnets.1.<keys>
/// ```
#[derive(Debug)]
pub struct UNetMidBlock2DCrossAttn<T: Float> {
    /// 2 time-conditioned resnets.
    pub resnets: Vec<ResnetBlock2DTime<T>>,
    /// 1 cross-attn transformer.
    pub attentions: Vec<Transformer2DModel<T>>,
    training: bool,
}

impl<T: Float> UNetMidBlock2DCrossAttn<T> {
    /// Build a randomly-initialized mid block (`channels -> channels`).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    pub fn new(
        channels: usize,
        temb_channels: usize,
        attention_head_dim: usize,
        cross_attention_dim: usize,
        norm_num_groups: usize,
        transformer_layers_per_block: usize,
    ) -> FerrotorchResult<Self> {
        // See CrossAttnDownBlock2D::new for the heads/dim_head footgun.
        let heads = attention_head_dim;
        let dim_head = channels / heads;
        let r0 = ResnetBlock2DTime::<T>::new(
            channels,
            channels,
            temb_channels,
            norm_num_groups,
            1e-5,
        )?;
        let attn = Transformer2DModel::<T>::new(
            channels,
            heads,
            dim_head,
            transformer_layers_per_block,
            cross_attention_dim,
            norm_num_groups,
        )?;
        let r1 = ResnetBlock2DTime::<T>::new(
            channels,
            channels,
            temb_channels,
            norm_num_groups,
            1e-5,
        )?;
        Ok(Self {
            resnets: vec![r0, r1],
            attentions: vec![attn],
            training: false,
        })
    }

    /// Forward with time embedding and encoder hidden states.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    pub fn forward_t(
        &self,
        x: &Tensor<T>,
        temb: &Tensor<T>,
        encoder_hidden_states: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        let mut h = self.resnets[0].forward_t(x, temb)?;
        for (a, r) in self.attentions.iter().zip(self.resnets.iter().skip(1)) {
            h = a.forward_xattn(&h, encoder_hidden_states)?;
            h = r.forward_t(&h, temb)?;
        }
        Ok(h)
    }
}

impl<T: Float> Module<T> for UNetMidBlock2DCrossAttn<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "UNetMidBlock2DCrossAttn::forward requires (x, temb, ehs) — call \
                      forward_t instead"
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        // Match diffusers: attentions before resnets at the state-dict
        // level. The forward path uses resnets[0] first, but the named
        // params are listed `attentions.*` then `resnets.*`.
        for a in &self.attentions {
            o.extend(a.parameters());
        }
        for r in &self.resnets {
            o.extend(r.parameters());
        }
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        for a in &mut self.attentions {
            o.extend(a.parameters_mut());
        }
        for r in &mut self.resnets {
            o.extend(r.parameters_mut());
        }
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (i, a) in self.attentions.iter().enumerate() {
            for (n, p) in a.named_parameters() {
                o.push((format!("attentions.{i}.{n}"), p));
            }
        }
        for (i, r) in self.resnets.iter().enumerate() {
            for (n, p) in r.named_parameters() {
                o.push((format!("resnets.{i}.{n}"), p));
            }
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
                let ok = k.starts_with("resnets.") || k.starts_with("attentions.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in UNetMidBlock2DCrossAttn state_dict: \"{k}\""
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
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CrossAttnUpBlock2D
// ---------------------------------------------------------------------------

/// `CrossAttnUpBlock2D` — `(layers_per_block + 1)` × (ResnetBlock2DTime
/// + Transformer2DModel) + optional Upsample2D. Each resnet's input
///   is `cat([h, skip], dim=channel)` where `skip` is popped from the
///   down-side skip stack.
#[derive(Debug)]
pub struct CrossAttnUpBlock2D<T: Float> {
    /// `(layers_per_block + 1)` time-conditioned resnets.
    pub resnets: Vec<ResnetBlock2DTime<T>>,
    /// `(layers_per_block + 1)` transformer blocks.
    pub attentions: Vec<Transformer2DModel<T>>,
    /// Optional upsample.
    pub upsamplers_0: Option<Upsample2D<T>>,
    training: bool,
}

impl<T: Float> CrossAttnUpBlock2D<T> {
    /// Build a randomly-initialized cross-attention up-block.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        prev_output_channels: usize,
        temb_channels: usize,
        num_layers: usize,
        attention_head_dim: usize,
        cross_attention_dim: usize,
        norm_num_groups: usize,
        add_upsample: bool,
        transformer_layers_per_block: usize,
    ) -> FerrotorchResult<Self> {
        // See CrossAttnDownBlock2D::new for the heads/dim_head footgun.
        let heads = attention_head_dim;
        let dim_head = out_channels / heads;
        let mut resnets = Vec::with_capacity(num_layers);
        let mut attentions = Vec::with_capacity(num_layers);
        // Diffusers' resnet input channel widths for an up-block of
        // length L:
        //   resnet[0]:  in = prev_output + skip_at_same_resolution
        //   resnet[i in 1..L-1]: in = out + skip_at_same_resolution
        //   resnet[L-1]: in = out + in_channels (the skip at the
        //                deepest resolution feeds back to the *previous*
        //                resolution's res output)
        // For the canonical SD shape:
        //   skip for j = 0:  res_skip = out
        //   skip for j = 1:  res_skip = out
        //   skip for j = L-1: res_skip = in_channels (the channel count
        //                that came from the down-side pre-downsample)
        for j in 0..num_layers {
            let res_skip = if j == num_layers - 1 {
                in_channels
            } else {
                out_channels
            };
            let resnet_in = if j == 0 {
                prev_output_channels + res_skip
            } else {
                out_channels + res_skip
            };
            resnets.push(ResnetBlock2DTime::<T>::new(
                resnet_in,
                out_channels,
                temb_channels,
                norm_num_groups,
                1e-5,
            )?);
            attentions.push(Transformer2DModel::<T>::new(
                out_channels,
                heads,
                dim_head,
                transformer_layers_per_block,
                cross_attention_dim,
                norm_num_groups,
            )?);
        }
        let upsamplers_0 = if add_upsample {
            Some(Upsample2D::<T>::new(out_channels)?)
        } else {
            None
        };
        Ok(Self {
            resnets,
            attentions,
            upsamplers_0,
            training: false,
        })
    }

    /// Forward, consuming `skips.len()` skip tensors (one per resnet).
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    pub fn forward_t(
        &self,
        x: &Tensor<T>,
        skips: &[Tensor<T>],
        temb: &Tensor<T>,
        encoder_hidden_states: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if skips.len() != self.resnets.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CrossAttnUpBlock2D::forward_t: got {} skips for {} resnets",
                    skips.len(),
                    self.resnets.len()
                ),
            });
        }
        let mut h = x.clone();
        for ((r, a), skip) in self
            .resnets
            .iter()
            .zip(self.attentions.iter())
            .zip(skips.iter())
        {
            h = ferrotorch_core::grad_fns::shape::cat(&[h.clone(), skip.clone()], 1)?;
            h = r.forward_t(&h, temb)?;
            h = a.forward_xattn(&h, encoder_hidden_states)?;
        }
        if let Some(u) = &self.upsamplers_0 {
            h = u.forward(&h)?;
        }
        Ok(h)
    }
}

impl<T: Float> Module<T> for CrossAttnUpBlock2D<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "CrossAttnUpBlock2D::forward requires (x, skips, temb, ehs) — call \
                      forward_t instead"
                .into(),
        })
    }
    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        for r in &self.resnets {
            o.extend(r.parameters());
        }
        for a in &self.attentions {
            o.extend(a.parameters());
        }
        if let Some(u) = &self.upsamplers_0 {
            o.extend(u.parameters());
        }
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        for r in &mut self.resnets {
            o.extend(r.parameters_mut());
        }
        for a in &mut self.attentions {
            o.extend(a.parameters_mut());
        }
        if let Some(u) = self.upsamplers_0.as_mut() {
            o.extend(u.parameters_mut());
        }
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (i, r) in self.resnets.iter().enumerate() {
            for (n, p) in r.named_parameters() {
                o.push((format!("resnets.{i}.{n}"), p));
            }
        }
        for (i, a) in self.attentions.iter().enumerate() {
            for (n, p) in a.named_parameters() {
                o.push((format!("attentions.{i}.{n}"), p));
            }
        }
        if let Some(u) = &self.upsamplers_0 {
            for (n, p) in u.named_parameters() {
                o.push((format!("upsamplers.0.{n}"), p));
            }
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
                let ok = k.starts_with("resnets.")
                    || k.starts_with("attentions.")
                    || k.starts_with("upsamplers.0.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in CrossAttnUpBlock2D state_dict: \"{k}\""
                        ),
                    });
                }
            }
        }
        for (i, r) in self.resnets.iter_mut().enumerate() {
            r.load_state_dict(&extract(&format!("resnets.{i}")), strict)?;
        }
        for (i, a) in self.attentions.iter_mut().enumerate() {
            a.load_state_dict(&extract(&format!("attentions.{i}")), strict)?;
        }
        if let Some(u) = self.upsamplers_0.as_mut() {
            u.load_state_dict(&extract("upsamplers.0"), strict)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// UpBlock2D (no cross-attn — the first SD up-block)
// ---------------------------------------------------------------------------

/// `UpBlock2D` — `(layers_per_block + 1)` ResnetBlock2DTime + optional
/// Upsample2D. No transformer attention.
#[derive(Debug)]
pub struct UpBlock2D<T: Float> {
    /// `(layers_per_block + 1)` time-conditioned resnets.
    pub resnets: Vec<ResnetBlock2DTime<T>>,
    /// Optional upsample.
    pub upsamplers_0: Option<Upsample2D<T>>,
    training: bool,
}

impl<T: Float> UpBlock2D<T> {
    /// Build a randomly-initialized non-cross-attn up-block.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        prev_output_channels: usize,
        temb_channels: usize,
        num_layers: usize,
        norm_num_groups: usize,
        add_upsample: bool,
    ) -> FerrotorchResult<Self> {
        let mut resnets = Vec::with_capacity(num_layers);
        for j in 0..num_layers {
            let res_skip = if j == num_layers - 1 {
                in_channels
            } else {
                out_channels
            };
            let resnet_in = if j == 0 {
                prev_output_channels + res_skip
            } else {
                out_channels + res_skip
            };
            resnets.push(ResnetBlock2DTime::<T>::new(
                resnet_in,
                out_channels,
                temb_channels,
                norm_num_groups,
                1e-5,
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

    /// Forward.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`].
    pub fn forward_t(
        &self,
        x: &Tensor<T>,
        skips: &[Tensor<T>],
        temb: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        if skips.len() != self.resnets.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "UpBlock2D::forward_t: got {} skips for {} resnets",
                    skips.len(),
                    self.resnets.len()
                ),
            });
        }
        let mut h = x.clone();
        for (r, skip) in self.resnets.iter().zip(skips.iter()) {
            h = ferrotorch_core::grad_fns::shape::cat(&[h.clone(), skip.clone()], 1)?;
            h = r.forward_t(&h, temb)?;
        }
        if let Some(u) = &self.upsamplers_0 {
            h = u.forward(&h)?;
        }
        Ok(h)
    }
}

impl<T: Float> Module<T> for UpBlock2D<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "UpBlock2D::forward requires (x, skips, temb) — call forward_t instead"
                .into(),
        })
    }
    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        for r in &self.resnets {
            o.extend(r.parameters());
        }
        if let Some(u) = &self.upsamplers_0 {
            o.extend(u.parameters());
        }
        o
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        for r in &mut self.resnets {
            o.extend(r.parameters_mut());
        }
        if let Some(u) = self.upsamplers_0.as_mut() {
            o.extend(u.parameters_mut());
        }
        o
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (i, r) in self.resnets.iter().enumerate() {
            for (n, p) in r.named_parameters() {
                o.push((format!("resnets.{i}.{n}"), p));
            }
        }
        if let Some(u) = &self.upsamplers_0 {
            for (n, p) in u.named_parameters() {
                o.push((format!("upsamplers.0.{n}"), p));
            }
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
                let ok = k.starts_with("resnets.") || k.starts_with("upsamplers.0.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in UpBlock2D state_dict: \"{k}\""),
                    });
                }
            }
        }
        for (i, r) in self.resnets.iter_mut().enumerate() {
            r.load_state_dict(&extract(&format!("resnets.{i}")), strict)?;
        }
        if let Some(u) = self.upsamplers_0.as_mut() {
            u.load_state_dict(&extract("upsamplers.0"), strict)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Discriminated down/up block enum
// ---------------------------------------------------------------------------

/// Discriminated wrapper over the two down-block variants.
#[derive(Debug)]
pub enum AnyDownBlock<T: Float> {
    /// CrossAttn variant.
    CrossAttn(CrossAttnDownBlock2D<T>),
    /// Plain variant.
    Plain(DownBlock2D<T>),
}

impl<T: Float> AnyDownBlock<T> {
    fn forward_t(
        &self,
        x: &Tensor<T>,
        temb: &Tensor<T>,
        encoder_hidden_states: &Tensor<T>,
    ) -> FerrotorchResult<(Tensor<T>, Vec<Tensor<T>>)> {
        match self {
            AnyDownBlock::CrossAttn(b) => b.forward_t(x, temb, encoder_hidden_states),
            AnyDownBlock::Plain(b) => b.forward_t(x, temb),
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        match self {
            AnyDownBlock::CrossAttn(b) => b.parameters(),
            AnyDownBlock::Plain(b) => b.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        match self {
            AnyDownBlock::CrossAttn(b) => b.parameters_mut(),
            AnyDownBlock::Plain(b) => b.parameters_mut(),
        }
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        match self {
            AnyDownBlock::CrossAttn(b) => b.named_parameters(),
            AnyDownBlock::Plain(b) => b.named_parameters(),
        }
    }
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        match self {
            AnyDownBlock::CrossAttn(b) => b.load_state_dict(state, strict),
            AnyDownBlock::Plain(b) => b.load_state_dict(state, strict),
        }
    }
}

/// Discriminated wrapper over the two up-block variants.
#[derive(Debug)]
pub enum AnyUpBlock<T: Float> {
    /// CrossAttn variant.
    CrossAttn(CrossAttnUpBlock2D<T>),
    /// Plain variant.
    Plain(UpBlock2D<T>),
}

impl<T: Float> AnyUpBlock<T> {
    fn forward_t(
        &self,
        x: &Tensor<T>,
        skips: &[Tensor<T>],
        temb: &Tensor<T>,
        encoder_hidden_states: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        match self {
            AnyUpBlock::CrossAttn(b) => b.forward_t(x, skips, temb, encoder_hidden_states),
            AnyUpBlock::Plain(b) => b.forward_t(x, skips, temb),
        }
    }
    fn num_resnets(&self) -> usize {
        match self {
            AnyUpBlock::CrossAttn(b) => b.resnets.len(),
            AnyUpBlock::Plain(b) => b.resnets.len(),
        }
    }
    fn parameters(&self) -> Vec<&Parameter<T>> {
        match self {
            AnyUpBlock::CrossAttn(b) => b.parameters(),
            AnyUpBlock::Plain(b) => b.parameters(),
        }
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        match self {
            AnyUpBlock::CrossAttn(b) => b.parameters_mut(),
            AnyUpBlock::Plain(b) => b.parameters_mut(),
        }
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        match self {
            AnyUpBlock::CrossAttn(b) => b.named_parameters(),
            AnyUpBlock::Plain(b) => b.named_parameters(),
        }
    }
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        match self {
            AnyUpBlock::CrossAttn(b) => b.load_state_dict(state, strict),
            AnyUpBlock::Plain(b) => b.load_state_dict(state, strict),
        }
    }
}

// ---------------------------------------------------------------------------
// UNet2DConditionModel
// ---------------------------------------------------------------------------

/// Diffusers-style `UNet2DConditionModel` — the SD-1.5 UNet.
#[derive(Debug)]
pub struct UNet2DConditionModel<T: Float> {
    /// Parameter-free sinusoidal encoding.
    pub time_proj: Timesteps,
    /// Trainable MLP after the encoding.
    pub time_embedding: TimestepEmbedding<T>,
    /// `conv_in`: Conv2d(in_channels, block_out_channels[0], k=3, pad=1).
    pub conv_in: Conv2d<T>,
    /// 4 down blocks.
    pub down_blocks: Vec<AnyDownBlock<T>>,
    /// 1 mid block.
    pub mid_block: UNetMidBlock2DCrossAttn<T>,
    /// 4 up blocks (in diffusers' order — index 0 is the deepest /
    /// lowest resolution / no-cross-attn block).
    pub up_blocks: Vec<AnyUpBlock<T>>,
    /// `conv_norm_out`: GroupNorm(32, block_out_channels[0]).
    pub conv_norm_out: GroupNorm<T>,
    /// SiLU activation.
    pub conv_act: SiLU,
    /// `conv_out`: Conv2d(block_out_channels[0], out_channels, k=3, pad=1).
    pub conv_out: Conv2d<T>,
    /// Frozen config copy.
    pub config: UNet2DConditionConfig,
    training: bool,
}

impl<T: Float> UNet2DConditionModel<T> {
    /// Build a randomly-initialized `UNet2DConditionModel`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for invalid config or
    /// underlying [`FerrotorchError`] from the conv constructor.
    pub fn new(cfg: UNet2DConditionConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let temb_channels = cfg.time_embed_dim();
        let bocs = &cfg.block_out_channels;
        let groups = cfg.norm_num_groups;
        let num_blocks = bocs.len();

        // time projection: sinusoidal encoding into block_out_channels[0],
        // then Linear(block_out_channels[0] -> time_embed_dim) + SiLU +
        // Linear(time_embed_dim -> time_embed_dim).
        let time_proj = Timesteps::new(bocs[0], cfg.flip_sin_to_cos, cfg.freq_shift)?;
        let time_embedding = TimestepEmbedding::<T>::new(bocs[0], temb_channels)?;

        // conv_in.
        let conv_in =
            Conv2d::<T>::new(cfg.in_channels, bocs[0], (3, 3), (1, 1), (1, 1), true)?;

        // Down blocks.
        let mut down_blocks: Vec<AnyDownBlock<T>> = Vec::with_capacity(num_blocks);
        let mut prev = bocs[0];
        for i in 0..num_blocks {
            let out_c = bocs[i];
            let is_final = i == num_blocks - 1;
            let add_downsample = !is_final;
            let block: AnyDownBlock<T> = if cfg.down_block_has_attn[i] {
                AnyDownBlock::CrossAttn(CrossAttnDownBlock2D::<T>::new(
                    prev,
                    out_c,
                    temb_channels,
                    cfg.layers_per_block,
                    cfg.attention_head_dim,
                    cfg.cross_attention_dim,
                    groups,
                    add_downsample,
                    cfg.transformer_layers_per_block,
                )?)
            } else {
                AnyDownBlock::Plain(DownBlock2D::<T>::new(
                    prev,
                    out_c,
                    temb_channels,
                    cfg.layers_per_block,
                    groups,
                    add_downsample,
                )?)
            };
            down_blocks.push(block);
            prev = out_c;
        }

        // Mid block (at the deepest channels = bocs[-1]).
        let mid_channels = bocs[num_blocks - 1];
        let mid_block = UNetMidBlock2DCrossAttn::<T>::new(
            mid_channels,
            temb_channels,
            cfg.attention_head_dim,
            cfg.cross_attention_dim,
            groups,
            cfg.transformer_layers_per_block,
        )?;

        // Up blocks — iterate through reversed_block_out_channels.
        //   reversed = bocs.reverse() = e.g. [1280, 1280, 640, 320]
        //   For each up-block i in 0..num_blocks:
        //     in_channels = reversed[min(i+1, num_blocks-1)] (channel
        //                    count of the corresponding down-block's
        //                    pre-downsample resolution)
        //     out_channels = reversed[i]
        //     prev_output  = reversed[i-1] if i>0 else mid_channels
        //                    (== reversed[0])
        //   Diffusers gates `add_upsample` on `i < num_blocks - 1`.
        let mut up_blocks: Vec<AnyUpBlock<T>> = Vec::with_capacity(num_blocks);
        let reversed: Vec<usize> = bocs.iter().rev().copied().collect();
        let mut prev_up = mid_channels;
        for i in 0..num_blocks {
            let out_c = reversed[i];
            let in_c = reversed[(i + 1).min(num_blocks - 1)];
            let is_final = i == num_blocks - 1;
            let add_upsample = !is_final;
            let block: AnyUpBlock<T> = if cfg.up_block_has_attn[i] {
                AnyUpBlock::CrossAttn(CrossAttnUpBlock2D::<T>::new(
                    in_c,
                    out_c,
                    prev_up,
                    temb_channels,
                    cfg.layers_per_block + 1,
                    cfg.attention_head_dim,
                    cfg.cross_attention_dim,
                    groups,
                    add_upsample,
                    cfg.transformer_layers_per_block,
                )?)
            } else {
                AnyUpBlock::Plain(UpBlock2D::<T>::new(
                    in_c,
                    out_c,
                    prev_up,
                    temb_channels,
                    cfg.layers_per_block + 1,
                    groups,
                    add_upsample,
                )?)
            };
            up_blocks.push(block);
            prev_up = out_c;
        }

        let conv_norm_out = GroupNorm::<T>::new(groups, bocs[0], 1e-5, true)?;
        let conv_out =
            Conv2d::<T>::new(bocs[0], cfg.out_channels, (3, 3), (1, 1), (1, 1), true)?;

        Ok(Self {
            time_proj,
            time_embedding,
            conv_in,
            down_blocks,
            mid_block,
            up_blocks,
            conv_norm_out,
            conv_act: SiLU::new(),
            conv_out,
            config: cfg,
            training: false,
        })
    }

    /// Run UNet forward.
    ///
    /// * `sample`: `[B, in_channels, H, W]` — noisy latent.
    /// * `timesteps`: `[B]` — diffusion timestep per batch entry.
    /// * `encoder_hidden_states`: `[B, S, cross_attention_dim]` — the
    ///   text-encoder conditioning.
    ///
    /// Returns the predicted noise `[B, out_channels, H, W]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] for bad input ranks.
    pub fn forward_t(
        &self,
        sample: &Tensor<T>,
        timesteps: &Tensor<T>,
        encoder_hidden_states: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        let cfg = &self.config;
        if sample.ndim() != 4 || sample.shape()[1] != cfg.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "UNet2DConditionModel: expected sample [B, {}, H, W], got {:?}",
                    cfg.in_channels,
                    sample.shape()
                ),
            });
        }
        if timesteps.ndim() != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "UNet2DConditionModel: expected timesteps [B], got {:?}",
                    timesteps.shape()
                ),
            });
        }
        if encoder_hidden_states.ndim() != 3
            || encoder_hidden_states.shape()[2] != cfg.cross_attention_dim
        {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "UNet2DConditionModel: expected encoder_hidden_states [B, S, {}], got {:?}",
                    cfg.cross_attention_dim,
                    encoder_hidden_states.shape()
                ),
            });
        }

        // 1. Time embedding: [B] -> [B, bocs[0]] -> [B, time_embed_dim].
        let t_enc = self.time_proj.forward_t(timesteps)?;
        let temb = self.time_embedding.forward(&t_enc)?;

        // 2. conv_in.
        let mut h = self.conv_in.forward(sample)?;
        let mut skips: Vec<Tensor<T>> = Vec::new();
        // The initial conv_in output is the first skip (the down-block
        // loop pushes its own; diffusers prepends `sample` so this skip
        // is consumed by the *last* up-block's last resnet).
        skips.push(h.clone());

        // 3. Down blocks.
        for db in &self.down_blocks {
            let (out, mut block_skips) = db.forward_t(&h, &temb, encoder_hidden_states)?;
            h = out;
            skips.append(&mut block_skips);
        }

        // 4. Mid block.
        h = self.mid_block.forward_t(&h, &temb, encoder_hidden_states)?;

        // 5. Up blocks. Each consumes the trailing N skips (where N =
        //    block.num_resnets()).
        for ub in &self.up_blocks {
            let n = ub.num_resnets();
            if skips.len() < n {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "UNet2DConditionModel: up-block needs {n} skips, only {} left",
                        skips.len()
                    ),
                });
            }
            let split_at = skips.len() - n;
            let popped = skips.split_off(split_at);
            // Diffusers iterates skips in the order they were popped
            // (most-recent-first per resnet inside the block). The
            // standard idiom is `res_samples = skips[-N:]` then the
            // block consumes them in reverse order (one per resnet).
            // We reverse `popped` so resnet[0] gets the most recently
            // pushed skip — the same as diffusers'
            // `down_block_res_samples[-1]` for `res_hidden_states_tuple
            // [-1]`.
            let popped_rev: Vec<Tensor<T>> = popped.into_iter().rev().collect();
            h = ub.forward_t(&h, &popped_rev, &temb, encoder_hidden_states)?;
        }

        // 6. Output head.
        h = self.conv_norm_out.forward(&h)?;
        h = self.conv_act.forward(&h)?;
        self.conv_out.forward(&h)
    }
}

impl<T: Float> Module<T> for UNet2DConditionModel<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Err(FerrotorchError::InvalidArgument {
            message: "UNet2DConditionModel::forward requires (sample, timesteps, ehs) — \
                      call forward_t instead"
                .into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.time_embedding.parameters());
        o.extend(self.conv_in.parameters());
        for db in &self.down_blocks {
            o.extend(db.parameters());
        }
        o.extend(self.mid_block.parameters());
        for ub in &self.up_blocks {
            o.extend(ub.parameters());
        }
        o.extend(self.conv_norm_out.parameters());
        o.extend(self.conv_out.parameters());
        o
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut o = Vec::new();
        o.extend(self.time_embedding.parameters_mut());
        o.extend(self.conv_in.parameters_mut());
        for db in &mut self.down_blocks {
            o.extend(db.parameters_mut());
        }
        o.extend(self.mid_block.parameters_mut());
        for ub in &mut self.up_blocks {
            o.extend(ub.parameters_mut());
        }
        o.extend(self.conv_norm_out.parameters_mut());
        o.extend(self.conv_out.parameters_mut());
        o
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut o = Vec::new();
        for (n, p) in self.time_embedding.named_parameters() {
            o.push((format!("time_embedding.{n}"), p));
        }
        for (n, p) in self.conv_in.named_parameters() {
            o.push((format!("conv_in.{n}"), p));
        }
        for (i, db) in self.down_blocks.iter().enumerate() {
            for (n, p) in db.named_parameters() {
                o.push((format!("down_blocks.{i}.{n}"), p));
            }
        }
        for (n, p) in self.mid_block.named_parameters() {
            o.push((format!("mid_block.{n}"), p));
        }
        for (i, ub) in self.up_blocks.iter().enumerate() {
            for (n, p) in ub.named_parameters() {
                o.push((format!("up_blocks.{i}.{n}"), p));
            }
        }
        for (n, p) in self.conv_norm_out.named_parameters() {
            o.push((format!("conv_norm_out.{n}"), p));
        }
        for (n, p) in self.conv_out.named_parameters() {
            o.push((format!("conv_out.{n}"), p));
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
                let ok = k.starts_with("time_embedding.")
                    || k.starts_with("conv_in.")
                    || k.starts_with("down_blocks.")
                    || k.starts_with("mid_block.")
                    || k.starts_with("up_blocks.")
                    || k.starts_with("conv_norm_out.")
                    || k.starts_with("conv_out.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in UNet2DConditionModel state_dict: \"{k}\""
                        ),
                    });
                }
            }
        }
        self.time_embedding
            .load_state_dict(&extract("time_embedding"), strict)?;
        self.conv_in.load_state_dict(&extract("conv_in"), strict)?;
        for (i, db) in self.down_blocks.iter_mut().enumerate() {
            db.load_state_dict(&extract(&format!("down_blocks.{i}")), strict)?;
        }
        self.mid_block
            .load_state_dict(&extract("mid_block"), strict)?;
        for (i, ub) in self.up_blocks.iter_mut().enumerate() {
            ub.load_state_dict(&extract(&format!("up_blocks.{i}")), strict)?;
        }
        self.conv_norm_out
            .load_state_dict(&extract("conv_norm_out"), strict)?;
        self.conv_out
            .load_state_dict(&extract("conv_out"), strict)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    fn tiny_cfg() -> UNet2DConditionConfig {
        UNet2DConditionConfig {
            in_channels: 4,
            out_channels: 4,
            block_out_channels: vec![16, 32, 64, 64],
            layers_per_block: 1,
            attention_head_dim: 8,
            cross_attention_dim: 24,
            norm_num_groups: 4,
            sample_size: 8,
            flip_sin_to_cos: true,
            freq_shift: 0.0,
            transformer_layers_per_block: 1,
            down_block_has_attn: vec![true, true, true, false],
            up_block_has_attn: vec![false, true, true, true],
        }
    }

    #[test]
    fn unet_forward_shape() {
        let cfg = tiny_cfg();
        let unet = UNet2DConditionModel::<f32>::new(cfg.clone()).unwrap();
        let sample = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 4 * 8 * 8]),
            vec![1, 4, 8, 8],
            false,
        )
        .unwrap();
        let timesteps = Tensor::from_storage(
            TensorStorage::cpu(vec![5.0f32]),
            vec![1],
            false,
        )
        .unwrap();
        let ehs = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 7 * 24]),
            vec![1, 7, 24],
            false,
        )
        .unwrap();
        let y = unet.forward_t(&sample, &timesteps, &ehs).unwrap();
        assert_eq!(y.shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn unet_named_parameters_includes_canonical_keys() {
        let cfg = tiny_cfg();
        let unet = UNet2DConditionModel::<f32>::new(cfg).unwrap();
        let names: Vec<String> = unet.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "time_embedding.linear_1.weight",
            "time_embedding.linear_2.bias",
            "conv_in.weight",
            "down_blocks.0.resnets.0.norm1.weight",
            "down_blocks.0.attentions.0.transformer_blocks.0.attn1.to_q.weight",
            "mid_block.attentions.0.transformer_blocks.0.attn2.to_v.weight",
            "mid_block.resnets.1.conv2.weight",
            "up_blocks.0.resnets.0.norm1.weight",
            "up_blocks.1.attentions.0.transformer_blocks.0.ff.net.0.proj.weight",
            "conv_norm_out.weight",
            "conv_out.bias",
        ] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }
}
