# ferrotorch-diffusion

Stable Diffusion family model composition for ferrotorch — CLIP text
encoder + UNet (cross-attention) + VAE decoder + DDIM scheduler + the
glue `StableDiffusionPipeline` that wires them into the canonical
classifier-free-guidance text-to-image loop.

## What it provides

### Models

- **`ClipTextEncoder` / `ClipTextConfig`** — the CLIP-ViT-L/14 text
  encoder (1 × `Embedding` + N × pre-norm transformer encoder layer +
  final LayerNorm + pooled-output gather). Loaded via
  `safetensors_loader::load_clip_text_encoder`.
- **`UNet2DConditionModel` / `UNet2DConditionConfig`** — the SD-1.5
  UNet (`conv_in` + 4 down-blocks + mid-block + 4 up-blocks + final
  `GroupNorm` + SiLU + `conv_out`) with cross-attention to the text
  embeddings, time-embedding MLP, and `ResnetBlock2DTime` residual
  blocks. Loaded via `safetensors_loader::load_unet`.
- **`VaeDecoder` / `Decoder` / `VaeDecoderConfig`** — the
  `diffusers.AutoencoderKL` decoder half (post-`scaling_factor`
  latent → image). Loaded via `safetensors_loader::load_vae_decoder`.

### Building blocks

`Attention`, `BasicTransformerBlock`, `FeedForward`,
`Transformer2DModel` (attention.rs); `AttnBlock2D`, `Downsample2D`,
`ResnetBlock2D`, `UNetMidBlock2D`, `UpDecoderBlock2D`, `Upsample2D`
(blocks.rs); `ResnetBlock2DTime` (resnet_block_time.rs);
`TimestepEmbedding`, `Timesteps` (time_embedding.rs).

### Scheduler

- **`DDIMScheduler` / `DDIMConfig`** with `BetaSchedule`,
  `PredictionType`, `TimestepSpacing` — the DDIM sampling step
  (`set_timesteps`, `scale_model_input`, `step`).

### Pipeline

- **`StableDiffusionPipeline`** — `encode_prompt(input_ids)` →
  `generate(cond_embeds, uncond_embeds, init_latent, steps,
  guidance_scale)` → `(image [1, 3, 512, 512], Vec<PipelineStepDump>)`.
  `PipelineStepDump` captures `(noise_uncond, noise_cond, guided_noise,
  latent_next)` per step for diagnostic parity.

### GPU paths (`cuda` feature)

`gpu::GpuClipTextEncoder`, `gpu::GpuUNet2DConditional`,
`gpu::GpuVaeDecoder`, `gpu::GpuStableDiffusionPipeline` — VRAM-resident
versions that exercise the CUDA forward path through
[`ferrotorch-gpu`](../ferrotorch-gpu) (#1164, #1165, #1166).

## Quick start

```rust
use ferrotorch_diffusion::{
    StableDiffusionPipeline, DDIMScheduler, DDIMConfig,
    safetensors_loader::{load_clip_text_encoder, load_unet, load_vae_decoder},
};

let text   = load_clip_text_encoder::<f32>("/path/to/sd-v1-5/text_encoder")?;
let unet   = load_unet::<f32>("/path/to/sd-v1-5/unet")?;
let vae    = load_vae_decoder::<f32>("/path/to/sd-v1-5/vae")?;
let sched  = DDIMScheduler::new(DDIMConfig::sd_v1_5())?;
let mut pipe = StableDiffusionPipeline::new(text, unet, vae, sched)?;

let cond   = pipe.encode_prompt(&prompt_ids)?;
let uncond = pipe.encode_prompt(&empty_prompt_ids)?;
let init_latent = /* [1, 4, 64, 64] standard-normal noise */;

let (image, dumps) = pipe.generate(&cond, &uncond, &init_latent, 50, 7.5)?;
// image: [1, 3, 512, 512] in [-1, 1] (apply (x+1)/2 + clamp to write PNG)
```

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `cuda`  | no      | Pulls in `ferrotorch-gpu` + `cudarc` and exposes the `gpu::*` types for VRAM-resident inference |

## Real-artifact parity

`scripts/verify_sd_pipeline_inference.py` runs this crate's
`StableDiffusionPipeline` against a frozen `diffusers==0.38.0`
`StableDiffusionPipeline.__call__` reference on a deterministic latent
(`torch.manual_seed(42); torch.randn(1, 4, 64, 64)`) with the pinned
[`runwayml/stable-diffusion-v1-5`](https://huggingface.co/runwayml/stable-diffusion-v1-5)
checkpoint, mirrored to `ferrotorch/sd-v1-5` and registered in
[`ferrotorch-hub`](../ferrotorch-hub).

PASS floor: `cosine_sim >= 0.999, max_abs <= 0.5` — same baseline as
the other Phase-B real-artifact harnesses (#1163).

Per-component dumps (`vae_decode_dump`, `unet_predict_dump`,
`clip_text_encode_dump`, `sd_pipeline_dump`) live under `examples/`.

## Status timeline

| Phase | What landed | Issue |
|-------|-------------|-------|
| B.3a  | VAE decoder + reference latent harness   | #1150 |
| B.3b  | UNet (cross-attention + time embedding)  | #1151 |
| B.3c  | CLIP text encoder                        | #1152 |
| B.3d  | DDIM scheduler                           | #1153 |
| B.3e  | `StableDiffusionPipeline` end-to-end     | #1163 |
| B.3f  | CUDA `gpu::GpuVaeDecoder`                | #1164 |
| B.3g  | CUDA `gpu::GpuClipTextEncoder`           | #1165 |
| B.3h  | CUDA `gpu::GpuUNet2DConditional`         | #1166 |

## Part of ferrotorch

This crate is one component of the
[ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
