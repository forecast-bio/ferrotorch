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
//! - [`vae_encoder::GpuVaeEncoder`] — VAE encoder forward path,
//!   mirroring [`crate::vae_encoder::VaeEncoder`] op-for-op on CUDA
//!   (#1177). Composes the existing ferrotorch-gpu element kernels
//!   plus `gpu_philox_normal` for the diagonal-Gaussian sample step.
//! - [`clip::GpuClipTextEncoder`] — SD-1.5 CLIP text-encoder forward
//!   path, mirroring [`crate::clip_text_encoder::ClipTextEncoder`]
//!   op-for-op on CUDA.
//! - [`unet::GpuUNet2DConditional`] — SD-1.5 UNet2DConditionModel
//!   forward path, mirroring [`crate::unet::UNet2DConditionModel`]
//!   op-for-op on CUDA.
//! - [`pipeline::GpuStableDiffusionPipeline`] — end-to-end SD-1.5
//!   text-to-image generation pipeline composing the three GPU
//!   sub-models above with the host-side
//!   [`crate::scheduler::DDIMScheduler`]. Mirrors
//!   [`crate::pipeline::StableDiffusionPipeline`] op-for-op on CUDA.

pub mod clip;
pub mod pipeline;
pub mod unet;
pub mod vae;
pub mod vae_encoder;

pub use clip::GpuClipTextEncoder;
pub use pipeline::GpuStableDiffusionPipeline;
pub use unet::GpuUNet2DConditional;
pub use vae::GpuVaeDecoder;
pub use vae_encoder::GpuVaeEncoder;
