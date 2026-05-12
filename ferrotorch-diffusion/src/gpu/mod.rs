#![cfg(feature = "cuda")]

//! GPU forward paths for SD-1.5 sub-models (gated on `feature = "cuda"`).
//!
//! This module is the GPU twin of the CPU [`crate::vae`] /
//! [`crate::unet`] / [`crate::clip_text_encoder`] families. Sub-models
//! are added one at a time as their kernels land in
//! `ferrotorch-gpu`. The current surface:
//!
//! - [`vae::GpuVaeDecoder`] — VAE decoder forward path, mirroring
//!   [`crate::vae::VaeDecoder`] op-for-op on CUDA.
//! - [`clip::GpuClipTextEncoder`] — SD-1.5 CLIP text-encoder forward
//!   path, mirroring [`crate::clip_text_encoder::ClipTextEncoder`]
//!   op-for-op on CUDA.
//! - [`unet::GpuUNet2DConditional`] — SD-1.5 UNet2DConditionModel
//!   forward path, mirroring [`crate::unet::UNet2DConditionModel`]
//!   op-for-op on CUDA.

pub mod clip;
pub mod unet;
pub mod vae;

pub use clip::GpuClipTextEncoder;
pub use unet::GpuUNet2DConditional;
pub use vae::GpuVaeDecoder;
