// Crate-level lint baseline. Mirrors the ferrotorch-whisper / ferrotorch-bert
// posture: deny correctness / idiom / Debug / docs problems; warn pedantic
// stylistic issues. Specific pedantic lints are allowed crate-wide where
// the lint is consistently wrong for ML/numeric kernel code.

#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// Casts: dimension math (`as usize`, `as f32`, `as u32`) is intrinsic
// to tensor indexing — every kernel call would otherwise need a
// per-call allow.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
// Builder-style accessors don't all need `#[must_use]`.
#![allow(clippy::must_use_candidate)]
// Identifiers like `bf16`, `f32`, `VAE`, `SD`, `SiLU` are flagged as
// missing backticks even when they appear in code-fenced text.
#![allow(clippy::doc_markdown)]
// `needless_pass_by_value` would force `&VaeDecoderConfig` signatures
// throughout, hiding intent in the API.
#![allow(clippy::needless_pass_by_value)]
// `unnecessary_wraps` flags `Result`-returning helpers that today
// always succeed but are part of an extensible API surface.
#![allow(clippy::unnecessary_wraps)]
// `uninlined_format_args` flags `format!("x={}", x)` vs
// `format!("x={x}")`. Both are equally clear; the fixup churn is high.
#![allow(clippy::uninlined_format_args)]
// `many_single_char_names` flags conventional ML kernel locals
// (`q`, `k`, `v`, `h`).
#![allow(clippy::many_single_char_names)]
// `similar_names` flags variable pairs that are intentionally similar
// (e.g. `q2` / `q_h`).
#![allow(clippy::similar_names)]
// `module_name_repetitions`: every type starts with `Vae` / `UNet`
// (matching the HF / diffusers naming) — the lint would force renames
// that lose the upstream-1:1 mapping.
#![allow(clippy::module_name_repetitions)]
// `too_many_lines`: the decoder / UNet forward is one cohesive sequence
// of ops mirroring the diffusers reference; splitting it hurts
// cross-reading.
#![allow(clippy::too_many_lines)]
// UNet builders take a handful of (in_c, out_c, temb, layers, heads,
// dim_head, cross_dim, groups, …) parameters — the explicit list is
// shorter than the struct-of-args alternative for an internal builder.
#![allow(clippy::too_many_arguments)]
// `items_after_statements` flags the in-test helper layout used widely.
#![allow(clippy::items_after_statements)]
// `redundant_else` flags `if x { return …; } else { … }`; the
// alternative (`if x { return …; } …`) loses the structural shape.
#![allow(clippy::redundant_else)]
// Tensor ops naturally use `for i in 0..n { … }` over `.iter()` when
// the index itself is used; clippy's preferred form hurts readability.
#![allow(clippy::needless_range_loop)]

//! Stable-Diffusion model composition for ferrotorch.
//!
//! Phase B.3 of real-artifact-driven development. This crate implements
//! the **VAE decoder** (Phase B.3a) and the **UNet2DConditionModel**
//! (Phase B.3b) of `runwayml/stable-diffusion-v1-5`. The encoder, the
//! CLIP text encoder, and the scheduler are out of scope and tracked
//! under follow-up dispatches.
//!
//! ## VAE decoder
//!
//! Mirrors `vae/config.json` — `VaeDecoder` inverts a latent
//! `[B, 4, 64, 64]` into an image `[B, 3, 512, 512]`. See [`vae`].
//!
//! ## UNet2DConditionModel
//!
//! Mirrors `unet/config.json` — `UNet2DConditionModel` consumes
//! `(noisy_latent [B, 4, 64, 64], timestep [B], text_embed [B, S, 768])`
//! and returns predicted noise `[B, 4, 64, 64]`. See [`unet`].
//!
//! ResnetBlock2DTime (UNet flavour with time bias):
//!
//! ```text
//! h = silu(norm1(x)); h = conv1(h)
//! t = silu(temb); h = h + Linear(t).view(B, out, 1, 1)
//! h = silu(norm2(h)); h = conv2(h)
//! out = h + (x if in==out else conv_shortcut(x))
//! ```
//!
//! Transformer2DModel (SD UNet flavour):
//!
//! ```text
//! h = GroupNorm(x); h = proj_in (Conv2d k=1, [B, inner, H, W])
//! h = flatten to [B, HW, inner]; for block in blocks: h = block(h, ehs)
//! h = reshape back; h = proj_out (Conv2d k=1); out = h + residual
//! ```
//!
//! Each `BasicTransformerBlock` is the canonical pre-LN
//! (self-attn → cross-attn → GEGLU FF) stack.

pub mod attention;
pub mod blocks;
pub mod config;
pub mod resnet_block_time;
pub mod safetensors_loader;
pub mod time_embedding;
pub mod unet;
pub mod unet_config;
pub mod vae;

pub use attention::{Attention, BasicTransformerBlock, FeedForward, Transformer2DModel};
pub use blocks::{AttnBlock2D, Downsample2D, ResnetBlock2D, UNetMidBlock2D, UpDecoderBlock2D, Upsample2D};
pub use config::VaeDecoderConfig;
pub use resnet_block_time::ResnetBlock2DTime;
pub use safetensors_loader::{load_unet, load_vae_decoder, DropReport};
pub use time_embedding::{TimestepEmbedding, Timesteps};
pub use unet::{
    AnyDownBlock, AnyUpBlock, CrossAttnDownBlock2D, CrossAttnUpBlock2D, DownBlock2D, UNet2DConditionModel,
    UNetMidBlock2DCrossAttn, UpBlock2D,
};
pub use unet_config::UNet2DConditionConfig;
pub use vae::{Decoder, VaeDecoder};
