//! Stable-Diffusion VAE encoder composition.
//!
//! Mirrors `diffusers.AutoencoderKL.encode(image).latent_dist` for
//! `runwayml/stable-diffusion-v1-5`:
//!
//! ```text
//! image (pixel-space, [B, 3, H, W])
//!   -> Encoder.conv_in
//!   -> Encoder.down_blocks[0..N]   (last block has no Downsample2D)
//!   -> Encoder.mid_block
//!   -> Encoder.conv_norm_out -> SiLU -> Encoder.conv_out
//!     (output: [B, 2 * latent_channels, H/8, W/8] — mean/logvar concat)
//!   -> quant_conv                  ([2*L -> 2*L], 1x1)
//!   -> DiagonalGaussianDistribution::from_parameters
//! ```
//!
//! The encoder-side mirror of [`crate::vae::VaeDecoder`]. The
//! `encode_with_scaling` helper composes
//! `latent_dist.sample(seed) * scaling_factor`, matching
//! `AutoencoderKL.encode(x).latent_dist.sample() * vae.config.scaling_factor`.
//! The bare [`Module::forward`] returns the raw `[B, 2*L, h, w]` parameters
//! tensor (no split, no sample, no scaling) so callers can swap in their
//! own sampling strategy (e.g. `.mode()` for deterministic decoding).

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv2d, GroupNorm, SiLU};

use crate::blocks::{DownEncoderBlock2D, UNetMidBlock2D};
use crate::config::VaeDecoderConfig;

/// Type alias — the SD VAE encoder and decoder share their config shape
/// (mirrors `diffusers.AutoencoderKL.config`, which spans both halves).
///
/// Using a single config type also lets a downstream caller load
/// `vae/config.json` once via [`VaeDecoderConfig::from_file`] and pass it
/// to both `VaeEncoder::new` and `VaeDecoder::new`.
pub type VaeEncoderConfig = VaeDecoderConfig;

/// Diffusers logvar clamp range (`DiagonalGaussianDistribution.__init__`).
///
/// Matches `torch.clamp(self.logvar, -30.0, 20.0)` in
/// `diffusers.models.autoencoders.vae.DiagonalGaussianDistribution`.
const LOGVAR_CLAMP_MIN: f64 = -30.0;
const LOGVAR_CLAMP_MAX: f64 = 20.0;

/// The bare `Encoder` half — matches `diffusers.models.autoencoders.vae.Encoder`.
#[derive(Debug)]
pub struct Encoder<T: Float> {
    /// First conv: `out_channels -> block_out_channels[0]` (k=3, pad=1).
    ///
    /// `out_channels` in [`VaeDecoderConfig`] is the image-side channel
    /// count (the decoder *output* and the encoder *input* — they're the
    /// same value, the field name is decoder-centric).
    pub conv_in: Conv2d<T>,
    /// Down-blocks in *encoder order* — block 0 operates at the lowest
    /// channel count and highest spatial resolution. The deepest block
    /// (`down_blocks[N-1]`) has no downsample (it preserves spatial
    /// resolution into the mid-block).
    pub down_blocks: Vec<DownEncoderBlock2D<T>>,
    /// VAE mid-block at `block_out_channels[-1]` channels (same module
    /// as in the decoder).
    pub mid_block: UNetMidBlock2D<T>,
    /// Final GroupNorm before the output conv (operates on
    /// `block_out_channels[-1]` channels).
    pub conv_norm_out: GroupNorm<T>,
    /// Output activation (SiLU).
    pub conv_act: SiLU,
    /// Output conv: `block_out_channels[-1] -> 2 * latent_channels`
    /// (k=3, pad=1). The factor of 2 holds the concatenated mean/logvar
    /// produced by the diagonal Gaussian head.
    pub conv_out: Conv2d<T>,
    /// Frozen copy of the config.
    pub config: VaeEncoderConfig,
    training: bool,
}

impl<T: Float> Encoder<T> {
    /// Build a randomly-initialized `Encoder`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] for any invalid
    /// config field (forwarded from [`VaeDecoderConfig::validate`]). In
    /// particular `block_out_channels` must be non-empty — the index
    /// into `[0]` / `[N-1]` below is preceded by `cfg.validate()?`
    /// which checks exactly that.
    pub fn new(cfg: VaeEncoderConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let groups = cfg.norm_num_groups;
        let resnet_eps = 1e-6_f64;
        let bottom_channels = cfg.block_out_channels[0];
        let top_channels =
            *cfg.block_out_channels
                .last()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: "Encoder::new: block_out_channels is empty (should be \
                              unreachable after validate)"
                        .into(),
                })?;

        // conv_in: image (out_channels) -> bottom_channels
        let conv_in = Conv2d::<T>::new(
            cfg.out_channels,
            bottom_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
        )?;

        // Down-blocks. Encoder uses `layers_per_block` resnets per block
        // (vs `layers_per_block + 1` on the decoder side).
        let num_blocks = cfg.block_out_channels.len();
        let resnets = cfg.layers_per_block;
        let mut down_blocks = Vec::with_capacity(num_blocks);
        let mut prev_out = bottom_channels;
        for (i, &c) in cfg.block_out_channels.iter().enumerate() {
            let is_final = i == num_blocks - 1;
            down_blocks.push(DownEncoderBlock2D::<T>::new(
                prev_out,
                c,
                resnets,
                groups,
                resnet_eps,
                !is_final,
            )?);
            prev_out = c;
        }

        // Mid-block at the deepest channel count.
        let mid_block = UNetMidBlock2D::<T>::new(top_channels, groups, resnet_eps)?;

        let conv_norm_out =
            GroupNorm::<T>::new(groups, top_channels, resnet_eps, true)?;
        // The `2 *` factor produces the concatenated mean/logvar tensor.
        let conv_out = Conv2d::<T>::new(
            top_channels,
            2 * cfg.latent_channels,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
        )?;

        Ok(Self {
            conv_in,
            down_blocks,
            mid_block,
            conv_norm_out,
            conv_act: SiLU::new(),
            conv_out,
            config: cfg,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for Encoder<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Sanity check: [B, out_channels, H, W] (out_channels is the
        // image channel count — naming is decoder-centric).
        let cfg = &self.config;
        if input.ndim() != 4 || input.shape()[1] != cfg.out_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Encoder::forward: expected [B, {}, H, W], got {:?}",
                    cfg.out_channels,
                    input.shape()
                ),
            });
        }
        let mut h = self.conv_in.forward(input)?;
        for d in &self.down_blocks {
            h = d.forward(&h)?;
        }
        h = self.mid_block.forward(&h)?;
        h = self.conv_norm_out.forward(&h)?;
        h = self.conv_act.forward(&h)?;
        self.conv_out.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.conv_in.parameters());
        for b in &self.down_blocks {
            out.extend(b.parameters());
        }
        out.extend(self.mid_block.parameters());
        out.extend(self.conv_norm_out.parameters());
        out.extend(self.conv_out.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.conv_in.parameters_mut());
        for b in &mut self.down_blocks {
            out.extend(b.parameters_mut());
        }
        out.extend(self.mid_block.parameters_mut());
        out.extend(self.conv_norm_out.parameters_mut());
        out.extend(self.conv_out.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.conv_in.named_parameters() {
            out.push((format!("conv_in.{n}"), p));
        }
        for (i, b) in self.down_blocks.iter().enumerate() {
            for (n, p) in b.named_parameters() {
                out.push((format!("down_blocks.{i}.{n}"), p));
            }
        }
        for (n, p) in self.mid_block.named_parameters() {
            out.push((format!("mid_block.{n}"), p));
        }
        for (n, p) in self.conv_norm_out.named_parameters() {
            out.push((format!("conv_norm_out.{n}"), p));
        }
        for (n, p) in self.conv_out.named_parameters() {
            out.push((format!("conv_out.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        for b in &mut self.down_blocks {
            b.train();
        }
        self.mid_block.train();
    }
    fn eval(&mut self) {
        self.training = false;
        for b in &mut self.down_blocks {
            b.eval();
        }
        self.mid_block.eval();
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
                let ok = k.starts_with("conv_in.")
                    || k.starts_with("down_blocks.")
                    || k.starts_with("mid_block.")
                    || k.starts_with("conv_norm_out.")
                    || k.starts_with("conv_out.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in Encoder state_dict: \"{k}\""),
                    });
                }
            }
        }

        self.conv_in.load_state_dict(&extract("conv_in"), strict)?;
        for (i, b) in self.down_blocks.iter_mut().enumerate() {
            b.load_state_dict(&extract(&format!("down_blocks.{i}")), strict)?;
        }
        self.mid_block
            .load_state_dict(&extract("mid_block"), strict)?;
        self.conv_norm_out
            .load_state_dict(&extract("conv_norm_out"), strict)?;
        self.conv_out
            .load_state_dict(&extract("conv_out"), strict)?;
        Ok(())
    }
}

/// `AutoencoderKL`-style VAE encoder = [`Encoder`] + `quant_conv`.
///
/// The encoder produces `[B, 2*latent_channels, H/8, W/8]` after the
/// `Encoder` stack; the 1x1 `quant_conv` then projects this through a
/// learned linear map. The split into (mean, logvar) and any sampling
/// happens in [`DiagonalGaussianDistribution`].
#[derive(Debug)]
pub struct VaeEncoder<T: Float> {
    /// The actual `Encoder` stack.
    pub encoder: Encoder<T>,
    /// 1x1 quant projection over `2 * latent_channels` channels.
    pub quant_conv: Conv2d<T>,
    /// Frozen config copy.
    pub config: VaeEncoderConfig,
    training: bool,
}

impl<T: Float> VaeEncoder<T> {
    /// Build a randomly-initialized `VaeEncoder`.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: VaeEncoderConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let encoder = Encoder::<T>::new(cfg.clone())?;
        let two_l = 2 * cfg.latent_channels;
        let quant_conv = Conv2d::<T>::new(two_l, two_l, (1, 1), (1, 1), (0, 0), true)?;
        Ok(Self {
            encoder,
            quant_conv,
            config: cfg,
            training: false,
        })
    }

    /// Encode an image into a diagonal Gaussian distribution over
    /// latent space. Matches `AutoencoderKL.encode(image).latent_dist`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] when the input is not
    /// `[B, out_channels, H, W]`. Propagates downstream op errors.
    pub fn encode(&self, image: &Tensor<T>) -> FerrotorchResult<DiagonalGaussianDistribution<T>> {
        let params = self.forward(image)?;
        DiagonalGaussianDistribution::from_parameters(&params, self.config.latent_channels)
    }

    /// Encode an image, sample from the latent distribution with a
    /// deterministic seed, then multiply by `scaling_factor`. This
    /// matches the canonical SD pipeline call:
    ///
    /// ```text
    /// latent = vae.encode(image).latent_dist.sample() * vae.config.scaling_factor
    /// ```
    ///
    /// The Box-Muller / xorshift noise generator runs on CPU and is
    /// fully deterministic for a given `seed` — different from
    /// PyTorch's CUDA RNG, so the produced latent will NOT be
    /// numerically identical to a Python reference run; only
    /// statistically equivalent.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] for shape or
    /// arithmetic failures.
    pub fn encode_with_scaling(
        &self,
        image: &Tensor<T>,
        seed: u64,
    ) -> FerrotorchResult<Tensor<T>> {
        let dist = self.encode(image)?;
        let sample = dist.sample_with_seed(seed)?;
        let sf = T::from(self.config.scaling_factor)
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!(
                    "VaeEncoder::encode_with_scaling: cannot cast scaling_factor={} into Float",
                    self.config.scaling_factor
                ),
            })?;
        let sf_t = ferrotorch_core::scalar::<T>(sf)?;
        ferrotorch_core::grad_fns::arithmetic::mul(&sample, &sf_t)
    }
}

impl<T: Float> Module<T> for VaeEncoder<T> {
    /// Forward returns the raw `[B, 2*latent_channels, h, w]` parameters
    /// tensor (concatenated mean/logvar, post `quant_conv`). Callers
    /// who want a latent should use [`Self::encode`] or
    /// [`Self::encode_with_scaling`].
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let cfg = &self.config;
        if input.ndim() != 4 || input.shape()[1] != cfg.out_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "VaeEncoder::forward: expected [B, {}, H, W], got {:?}",
                    cfg.out_channels,
                    input.shape()
                ),
            });
        }
        let h = self.encoder.forward(input)?;
        self.quant_conv.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.encoder.parameters());
        out.extend(self.quant_conv.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.encoder.parameters_mut());
        out.extend(self.quant_conv.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.encoder.named_parameters() {
            out.push((format!("encoder.{n}"), p));
        }
        for (n, p) in self.quant_conv.named_parameters() {
            out.push((format!("quant_conv.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.encoder.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.encoder.eval();
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
                let ok = k.starts_with("encoder.") || k.starts_with("quant_conv.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in VaeEncoder state_dict: \"{k}\""),
                    });
                }
            }
        }
        self.encoder.load_state_dict(&extract("encoder"), strict)?;
        self.quant_conv
            .load_state_dict(&extract("quant_conv"), strict)?;
        let _: HashMap<String, Tensor<T>> = HashMap::new(); // keep HashMap import alive
        Ok(())
    }
}

/// Diagonal Gaussian over latent space — the same parameterization
/// `diffusers.models.autoencoders.vae.DiagonalGaussianDistribution`
/// uses. Holds `mean` and `logvar` tensors (both `[B, L, h, w]`) split
/// from the encoder's concatenated parameters output.
///
/// `logvar` is clamped to `[-30, 20]` on construction, matching the
/// diffusers reference exactly.
#[derive(Debug)]
pub struct DiagonalGaussianDistribution<T: Float> {
    /// Mean of the diagonal Gaussian, `[B, L, h, w]`.
    pub mean: Tensor<T>,
    /// Log-variance of the diagonal Gaussian (clamped), `[B, L, h, w]`.
    pub logvar: Tensor<T>,
}

impl<T: Float> DiagonalGaussianDistribution<T> {
    /// Split the encoder's `[B, 2*L, h, w]` parameters tensor into
    /// mean / logvar halves along the channel axis.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] when the channel
    /// dimension is not `2 * latent_channels`. Propagates downstream
    /// errors from the chunk / clamp ops.
    pub fn from_parameters(
        params: &Tensor<T>,
        latent_channels: usize,
    ) -> FerrotorchResult<Self> {
        if params.ndim() != 4 || params.shape()[1] != 2 * latent_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "DiagonalGaussianDistribution::from_parameters: expected [B, {}, H, W], \
                     got {:?}",
                    2 * latent_channels,
                    params.shape()
                ),
            });
        }
        // chunk(2, dim=1): two [B, L, h, w] tensors — [mean, logvar].
        let parts = params.chunk(2, 1)?;
        let mean = parts[0].clone();
        let logvar_raw = parts[1].clone();
        // Match diffusers' `torch.clamp(logvar, -30.0, 20.0)`.
        let lo = T::from(LOGVAR_CLAMP_MIN).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!(
                "DiagonalGaussianDistribution: cannot cast logvar clamp min {LOGVAR_CLAMP_MIN} \
                 into Float"
            ),
        })?;
        let hi = T::from(LOGVAR_CLAMP_MAX).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!(
                "DiagonalGaussianDistribution: cannot cast logvar clamp max {LOGVAR_CLAMP_MAX} \
                 into Float"
            ),
        })?;
        let logvar = logvar_raw.clamp_t(lo, hi)?;
        Ok(Self { mean, logvar })
    }

    /// Return the distribution mode — the deterministic latent if the
    /// caller does not want to sample. Matches
    /// `DiagonalGaussianDistribution.mode()` in diffusers.
    pub fn mode(&self) -> &Tensor<T> {
        &self.mean
    }

    /// Sample from the distribution with a deterministic seed.
    ///
    /// Computes `mean + std * eps` where `std = exp(0.5 * logvar)`
    /// and `eps` is a fresh `N(0, 1)` tensor produced by Box-Muller
    /// over a seeded xorshift64 PRNG (mirroring the CPU branch of
    /// [`ferrotorch_core::randn`] but using `seed` as the state).
    ///
    /// This is fully deterministic for a given `seed` on a given host
    /// — but the bitwise output will NOT match a CUDA PyTorch reference
    /// run, because PyTorch's CUDA RNG uses Philox, not xorshift +
    /// Box-Muller. Tests should compare statistical properties (mean,
    /// std) or use `.mode()` for bit-exact reference matches.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] for shape or
    /// arithmetic failures.
    pub fn sample_with_seed(&self, seed: u64) -> FerrotorchResult<Tensor<T>> {
        let eps = randn_with_seed::<T>(self.mean.shape(), seed)?;
        // std = exp(0.5 * logvar)
        let half = T::from(0.5f64).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "DiagonalGaussianDistribution::sample_with_seed: cannot cast 0.5 into Float"
                .into(),
        })?;
        let half_t = ferrotorch_core::scalar::<T>(half)?;
        let scaled_logvar =
            ferrotorch_core::grad_fns::arithmetic::mul(&self.logvar, &half_t)?;
        let std = ferrotorch_core::grad_fns::transcendental::exp(&scaled_logvar)?;
        let noise = ferrotorch_core::grad_fns::arithmetic::mul(&std, &eps)?;
        ferrotorch_core::grad_fns::arithmetic::add(&self.mean, &noise)
    }
}

/// Box-Muller + xorshift64 N(0, 1) tensor generator seeded by `seed`.
///
/// Local to this module so the encoder doesn't pull in a fresh `rand`
/// dependency. The algorithm mirrors the CPU branch of
/// [`ferrotorch_core::randn`] exactly, but takes the seed as an input
/// rather than deriving it from system time + thread id.
fn randn_with_seed<T: Float>(shape: &[usize], seed: u64) -> FerrotorchResult<Tensor<T>> {
    let numel: usize = shape.iter().product();
    let mut data = Vec::with_capacity(numel);
    let mut state = if seed == 0 { 0x0000_dead_beef_cafe } else { seed };

    let mut next_uniform = || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        ((state as f64) / (u64::MAX as f64)).max(1e-300)
    };

    let mut i = 0;
    while i < numel {
        let u1 = next_uniform();
        let u2 = next_uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        data.push(T::from(r * theta.cos()).ok_or_else(|| FerrotorchError::InvalidArgument {
            message: "randn_with_seed: cannot cast Box-Muller output into Float".into(),
        })?);
        if i + 1 < numel {
            data.push(T::from(r * theta.sin()).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: "randn_with_seed: cannot cast Box-Muller output into Float".into(),
                }
            })?);
        }
        i += 2;
    }

    data.truncate(numel);
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Tiny config that exercises every architectural feature (mid-block
    /// attn, 4 down-blocks, channel-changing resnet shortcut) without
    /// making tests slow. Same shape as the decoder's `tiny_cfg`.
    fn tiny_cfg() -> VaeEncoderConfig {
        VaeDecoderConfig {
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![4, 8, 16, 16],
            layers_per_block: 1,
            norm_num_groups: 4,
            sample_size: 8,
            scaling_factor: 0.18215,
        }
    }

    #[test]
    fn encoder_forward_shape() {
        let cfg = tiny_cfg();
        let e = Encoder::<f32>::new(cfg.clone()).unwrap();
        // image: [1, 3, 8, 8] -> after 3 downsamples => [1, 2*4=8, 1, 1].
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 3 * 8 * 8]),
            vec![1, 3, 8, 8],
            false,
        )
        .unwrap();
        let y = e.forward(&x).unwrap();
        // 8 -> 4 -> 2 -> 1 (3 downsamples, last block has no downsample).
        assert_eq!(y.shape(), &[1, 2 * cfg.latent_channels, 1, 1]);
        for &v in y.data().unwrap() {
            assert!(v.is_finite(), "encoder output non-finite: {v}");
        }
    }

    #[test]
    fn vae_encoder_forward_shape() {
        let cfg = tiny_cfg();
        let v = VaeEncoder::<f32>::new(cfg.clone()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 3 * 8 * 8]),
            vec![1, 3, 8, 8],
            false,
        )
        .unwrap();
        let y = v.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2 * cfg.latent_channels, 1, 1]);
    }

    #[test]
    fn vae_encoder_named_parameters_include_quant_conv() {
        let cfg = tiny_cfg();
        let v = VaeEncoder::<f32>::new(cfg).unwrap();
        let names: Vec<String> = v.named_parameters().into_iter().map(|(n, _)| n).collect();
        for k in [
            "quant_conv.weight",
            "quant_conv.bias",
            "encoder.conv_in.weight",
            "encoder.down_blocks.0.resnets.0.norm1.weight",
            "encoder.mid_block.attentions.0.to_q.weight",
            "encoder.conv_norm_out.weight",
            "encoder.conv_out.bias",
        ] {
            assert!(names.iter().any(|n| n == k), "missing {k} in {names:?}");
        }
    }

    #[test]
    fn diag_gauss_split_and_mode_shapes() {
        let cfg = tiny_cfg();
        let v = VaeEncoder::<f32>::new(cfg.clone()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 3 * 8 * 8]),
            vec![1, 3, 8, 8],
            false,
        )
        .unwrap();
        let dist = v.encode(&x).unwrap();
        assert_eq!(dist.mean.shape(), &[1, cfg.latent_channels, 1, 1]);
        assert_eq!(dist.logvar.shape(), &[1, cfg.latent_channels, 1, 1]);
        // .mode() must hand back exactly the mean tensor.
        let mode = dist.mode();
        assert_eq!(mode.shape(), dist.mean.shape());
        for (a, b) in mode
            .data()
            .unwrap()
            .iter()
            .zip(dist.mean.data().unwrap().iter())
        {
            assert!((a - b).abs() < 1e-7);
        }
    }

    #[test]
    fn diag_gauss_logvar_is_clamped() {
        // Build a synthetic params tensor where the logvar half is
        // wildly out of range [-30, 20] and verify the clamp lands.
        let l = 2;
        let mut data = vec![0.5f32; l * 2]; // mean half: 0.5 everywhere
        for d in data.iter_mut().skip(l) {
            *d = 1e6; // logvar half: way above the +20 ceiling
        }
        let params =
            Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2 * l, 1, 1], false).unwrap();
        let dist = DiagonalGaussianDistribution::<f32>::from_parameters(&params, l).unwrap();
        for &v in dist.logvar.data().unwrap() {
            assert!(
                v <= LOGVAR_CLAMP_MAX as f32 + 1e-4,
                "logvar not clamped to <= {LOGVAR_CLAMP_MAX}: got {v}"
            );
        }
    }

    #[test]
    fn diag_gauss_sample_with_seed_is_deterministic() {
        // Synthetic, controlled distribution: mean=0, logvar=0 ⇒ std=1
        // ⇒ sample == eps, which is deterministic in our PRNG given a
        // fixed seed.
        let l = 4;
        let zeros = vec![0.0f32; 2 * l * 3 * 3];
        let params = Tensor::from_storage(
            TensorStorage::cpu(zeros),
            vec![1, 2 * l, 3, 3],
            false,
        )
        .unwrap();
        let dist = DiagonalGaussianDistribution::<f32>::from_parameters(&params, l).unwrap();
        let s1 = dist.sample_with_seed(42).unwrap();
        let s2 = dist.sample_with_seed(42).unwrap();
        let s3 = dist.sample_with_seed(43).unwrap();
        assert_eq!(s1.shape(), &[1, l, 3, 3]);
        for (a, b) in s1.data().unwrap().iter().zip(s2.data().unwrap().iter()) {
            assert!(
                (a - b).abs() < 1e-7,
                "sample_with_seed(42) not deterministic: {a} vs {b}"
            );
        }
        // Different seed should produce different output (with extreme
        // probability — Box-Muller from a different xorshift state).
        let mut differ = false;
        for (a, c) in s1.data().unwrap().iter().zip(s3.data().unwrap().iter()) {
            if (a - c).abs() > 1e-6 {
                differ = true;
                break;
            }
        }
        assert!(differ, "sample_with_seed(42) and sample_with_seed(43) produced identical output");
    }

    #[test]
    fn vae_encoder_round_trip_state_dict() {
        let cfg = tiny_cfg();
        let src = VaeEncoder::<f32>::new(cfg.clone()).unwrap();
        let sd = src.state_dict();
        let mut dst = VaeEncoder::<f32>::new(cfg.clone()).unwrap();
        dst.load_state_dict(&sd, true).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 3 * 8 * 8]),
            vec![1, 3, 8, 8],
            false,
        )
        .unwrap();
        let a = src.forward(&x).unwrap();
        let b = dst.forward(&x).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-5, "round-trip differs: {x} vs {y}");
        }
    }

    #[test]
    fn encode_with_scaling_applies_scaling_factor() {
        let cfg = tiny_cfg();
        let v = VaeEncoder::<f32>::new(cfg.clone()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.05f32; 3 * 8 * 8]),
            vec![1, 3, 8, 8],
            false,
        )
        .unwrap();
        // Sample via the high-level API, then independently compute
        // sample * scaling_factor and compare.
        let scaled = v.encode_with_scaling(&x, 99).unwrap();
        let dist = v.encode(&x).unwrap();
        let raw = dist.sample_with_seed(99).unwrap();
        for (s, r) in scaled
            .data()
            .unwrap()
            .iter()
            .zip(raw.data().unwrap().iter())
        {
            let expected = r * cfg.scaling_factor as f32;
            assert!(
                (s - expected).abs() < 1e-5,
                "encode_with_scaling didn't apply scaling_factor: got {s}, expected {expected}"
            );
        }
    }
}
