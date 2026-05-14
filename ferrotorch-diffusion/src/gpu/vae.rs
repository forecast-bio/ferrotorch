#![cfg(feature = "cuda")]
//! GPU VAE-decoder forward path for SD-1.5.
//!
//! [`GpuVaeDecoder`] mirrors [`crate::vae::VaeDecoder`] op-for-op,
//! resident in VRAM. Every conv / group-norm / attention / upsample /
//! residual call is dispatched through the matching `ferrotorch-gpu`
//! kernel; the only host-side traffic is the one-shot weight upload at
//! construction and the final image download at [`Self::decode`].
//!
//! The decoder hierarchy matches the diffusers `Decoder` layout used by
//! the CPU path:
//!
//! ```text
//! post_quant_conv (1x1)
//!   -> conv_in (3x3, latent_channels -> 512)
//!   -> mid_block (resnet0 -> attn0 -> resnet1)
//!   -> up_blocks (4 blocks, channels 512,512,256,128)
//!        each: 3x ResnetBlock2D, optional Upsample2D
//!   -> conv_norm_out (GroupNorm) -> SiLU -> conv_out (3x3 -> 3)
//! ```
//!
//! [`GpuVaeDecoder::decode`] applies the `1/scaling_factor` divide on
//! the input latent (matching `AutoencoderKL.decode(z).sample`) and
//! returns the decoded image as a ferrotorch CPU `Tensor`
//! `[B, 3, 512, 512]`.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_gpu::{
    CudaBuffer, GpuDevice, GpuError, gpu_bmm_f32, gpu_conv2d_f32, gpu_group_norm_f32,
    gpu_matmul_f32, gpu_nearest_upsample2x_f32, gpu_softmax,
    kernels::{gpu_add, gpu_broadcast_add, gpu_scale, gpu_silu},
    transfer::{cpu_to_gpu, gpu_to_cpu},
};
use ferrotorch_nn::module::{Module, StateDict};

use crate::config::VaeDecoderConfig;
use crate::safetensors_loader::DropReport;
use crate::vae::VaeDecoder;

// Helper PTX exists implicitly: we use gpu_mul_scalar_f32 from kernels.

// ---------------------------------------------------------------------------
// Per-layer buffer bundles
// ---------------------------------------------------------------------------

/// Parameters for a single `Conv2d(in, out, kernel, stride, padding,
/// bias=true)` resident on the GPU.
///
/// Weight shape: `[out, in, kH, kW]` (PyTorch convention).
#[derive(Debug)]
pub(super) struct GpuConv2d {
    pub(super) weight: CudaBuffer<f32>,
    pub(super) bias: CudaBuffer<f32>,
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
    pub(super) kernel: (usize, usize),
    pub(super) stride: (usize, usize),
    pub(super) padding: (usize, usize),
}

/// Parameters for a `GroupNorm(num_groups, num_channels, eps,
/// affine=true)` resident on the GPU.
#[derive(Debug)]
pub(super) struct GpuGroupNorm {
    pub(super) weight: CudaBuffer<f32>,
    pub(super) bias: CudaBuffer<f32>,
    pub(super) num_groups: usize,
    pub(super) num_channels: usize,
    pub(super) eps: f32,
}

/// Parameters for a single `Linear(in, out, bias=true)` resident on
/// the GPU. Weight shape `[out, in]` (PyTorch convention).
#[derive(Debug)]
pub(super) struct GpuLinear {
    pub(super) weight: CudaBuffer<f32>,
    pub(super) bias: CudaBuffer<f32>,
    pub(super) in_features: usize,
    pub(super) out_features: usize,
}

/// One VAE `ResnetBlock2D`: GN -> SiLU -> Conv -> GN -> SiLU -> Conv
/// + residual (optional 1x1 shortcut conv when `in != out`).
#[derive(Debug)]
pub(super) struct GpuResnet {
    pub(super) norm1: GpuGroupNorm,
    pub(super) conv1: GpuConv2d,
    pub(super) norm2: GpuGroupNorm,
    pub(super) conv2: GpuConv2d,
    /// Present iff `in_channels != out_channels`.
    pub(super) conv_shortcut: Option<GpuConv2d>,
    pub(super) in_channels: usize,
    pub(super) out_channels: usize,
}

/// Mid-block self-attention (single-head, GroupNorm + Linear q/k/v
/// + Linear to_out.0).
#[derive(Debug)]
pub(super) struct GpuAttn {
    pub(super) group_norm: GpuGroupNorm,
    pub(super) to_q: GpuLinear,
    pub(super) to_k: GpuLinear,
    pub(super) to_v: GpuLinear,
    pub(super) to_out_0: GpuLinear,
    pub(super) channels: usize,
}

/// `Upsample2D`: nearest-2x then `Conv2d(C->C, 3x3, pad=1)`.
#[derive(Debug)]
struct GpuUpsample {
    conv: GpuConv2d,
    channels: usize,
}

/// `UpDecoderBlock2D`: stack of resnets + optional upsample.
#[derive(Debug)]
struct GpuUpDecoderBlock {
    resnets: Vec<GpuResnet>,
    upsample: Option<GpuUpsample>,
}

/// `UNetMidBlock2D` (VAE flavour): resnet0 -> attn -> resnet1.
#[derive(Debug)]
pub(super) struct GpuMidBlock {
    pub(super) resnets: Vec<GpuResnet>,
    pub(super) attentions: Vec<GpuAttn>,
}

// ---------------------------------------------------------------------------
// GpuVaeDecoder
// ---------------------------------------------------------------------------

/// VAE-decoder forward path resident on a single CUDA device.
///
/// Constructed from a [`VaeDecoderConfig`] and a host-side
/// [`StateDict<f32>`] (the standard `decoder.*` / `post_quant_conv.*`
/// key layout produced by [`crate::load_vae_decoder`]). Every parameter
/// tensor is uploaded once into GPU memory; the host copy is dropped
/// after construction.
///
/// # Example
///
/// ```ignore
/// let device = GpuDevice::new(0)?;
/// let (cpu_dec, _drop) = load_vae_decoder::<f32>(weights, cfg.clone(), false)?;
/// let gpu = GpuVaeDecoder::from_module(&cpu_dec, &device)?;
/// let img = gpu.decode(&latent)?; // [1, 3, 512, 512]
/// ```
#[derive(Debug)]
pub struct GpuVaeDecoder {
    post_quant_conv: GpuConv2d,
    conv_in: GpuConv2d,
    mid_block: GpuMidBlock,
    up_blocks: Vec<GpuUpDecoderBlock>,
    conv_norm_out: GpuGroupNorm,
    conv_out: GpuConv2d,
    config: VaeDecoderConfig,
    device: GpuDevice,
}

impl GpuVaeDecoder {
    /// Build the GPU decoder from a config + state-dict.
    ///
    /// The state-dict is expected in the same shape as the CPU
    /// `VaeDecoder` produces: `post_quant_conv.*` + `decoder.*` keys.
    /// Tensors are checked for length (the GPU code re-uses the CPU
    /// shape contract) and uploaded once to VRAM.
    ///
    /// Unmapped keys are returned via [`DropReport`]; the caller may
    /// assert the drop set is empty (strict load) or use it as audit.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`FerrotorchError::InvalidArgument`] for an empty state-dict
    ///   or a key whose tensor data is unavailable.
    /// - [`FerrotorchError::ShapeMismatch`] when a tensor's element
    ///   count does not match the architectural shape implied by
    ///   `config`.
    /// - Any GPU error surfaced by `cpu_to_gpu` during upload (wrapped
    ///   in `FerrotorchError::InvalidArgument`).
    pub fn new(
        config: VaeDecoderConfig,
        mut state: StateDict<f32>,
        device: GpuDevice,
    ) -> FerrotorchResult<(Self, DropReport)> {
        config.validate()?;
        let eps = 1e-6_f32;
        let groups = config.norm_num_groups;
        let latent_c = config.latent_channels;
        let top_c =
            *config
                .block_out_channels
                .last()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: "GpuVaeDecoder: block_out_channels empty".into(),
                })?;
        let bottom_c = config.block_out_channels[0];
        let resnets_per_block = config.resnets_per_up_block();

        // post_quant_conv: 1x1, latent->latent, pad 0.
        let post_quant_conv = pop_conv(
            &mut state,
            "post_quant_conv",
            latent_c,
            latent_c,
            (1, 1),
            (1, 1),
            (0, 0),
            &device,
        )?;

        // conv_in: 3x3, latent->top, pad 1.
        let conv_in = pop_conv(
            &mut state,
            "decoder.conv_in",
            latent_c,
            top_c,
            (3, 3),
            (1, 1),
            (1, 1),
            &device,
        )?;

        // Mid block: resnets[0], attn[0], resnets[1] (all at top_c).
        let mid_resnet0 =
            pop_resnet(&mut state, "decoder.mid_block.resnets.0", top_c, top_c, groups, eps, &device)?;
        let mid_attn0 =
            pop_attn(&mut state, "decoder.mid_block.attentions.0", top_c, groups, eps, &device)?;
        let mid_resnet1 =
            pop_resnet(&mut state, "decoder.mid_block.resnets.1", top_c, top_c, groups, eps, &device)?;
        let mid_block = GpuMidBlock {
            resnets: vec![mid_resnet0, mid_resnet1],
            attentions: vec![mid_attn0],
        };

        // Up blocks (reversed block_out_channels).
        let reversed: Vec<usize> = config.block_out_channels.iter().rev().copied().collect();
        let mut up_blocks: Vec<GpuUpDecoderBlock> = Vec::with_capacity(reversed.len());
        let mut prev_out = reversed[0];
        let num_blocks = reversed.len();
        for (bi, &out_c) in reversed.iter().enumerate() {
            let is_final = bi == num_blocks - 1;
            let mut resnets = Vec::with_capacity(resnets_per_block);
            for ri in 0..resnets_per_block {
                let in_c = if ri == 0 { prev_out } else { out_c };
                let prefix = format!("decoder.up_blocks.{bi}.resnets.{ri}");
                resnets.push(pop_resnet(
                    &mut state, &prefix, in_c, out_c, groups, eps, &device,
                )?);
            }
            let upsample = if is_final {
                None
            } else {
                let prefix = format!("decoder.up_blocks.{bi}.upsamplers.0.conv");
                let conv = pop_conv(
                    &mut state, &prefix, out_c, out_c, (3, 3), (1, 1), (1, 1), &device,
                )?;
                Some(GpuUpsample { conv, channels: out_c })
            };
            up_blocks.push(GpuUpDecoderBlock { resnets, upsample });
            prev_out = out_c;
        }

        // conv_norm_out + conv_out
        let conv_norm_out = pop_groupnorm(
            &mut state, "decoder.conv_norm_out", groups, bottom_c, eps, &device,
        )?;
        let conv_out = pop_conv(
            &mut state, "decoder.conv_out", bottom_c, config.out_channels,
            (3, 3), (1, 1), (1, 1), &device,
        )?;

        // Whatever is left in `state` is unmapped — surface as DropReport.
        let mut dropped: Vec<String> = state.keys().cloned().collect();
        dropped.sort();
        let report = DropReport { dropped };

        Ok((
            Self {
                post_quant_conv,
                conv_in,
                mid_block,
                up_blocks,
                conv_norm_out,
                conv_out,
                config,
                device,
            },
            report,
        ))
    }

    /// Convenience constructor: build a [`GpuVaeDecoder`] from an
    /// already-loaded CPU [`VaeDecoder`].
    ///
    /// Equivalent to extracting `cpu.state_dict()` and calling
    /// [`Self::new`].
    ///
    /// # Errors
    ///
    /// Forwards every error from [`Self::new`].
    pub fn from_module(
        cpu: &VaeDecoder<f32>,
        device: &GpuDevice,
    ) -> FerrotorchResult<(Self, DropReport)> {
        let state: StateDict<f32> = cpu.state_dict();
        let device_clone = device.clone();
        Self::new(cpu.config.clone(), state, device_clone)
    }

    /// Decode a latent into an image.
    ///
    /// Mirrors [`VaeDecoder::decode_with_scaling`]: the input latent is
    /// divided by `scaling_factor` (matching
    /// `AutoencoderKL.decode(z).sample`) before flowing through
    /// `post_quant_conv -> Decoder`.
    ///
    /// # Errors
    ///
    /// Returns:
    /// - [`FerrotorchError::ShapeMismatch`] if the latent is not
    ///   `[B, latent_channels, H, W]`.
    /// - Errors propagated from any GPU op on the forward path
    ///   (wrapped in `FerrotorchError::InvalidArgument`).
    pub fn decode(&self, latent: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
        let shape = latent.shape();
        if shape.len() != 4 || shape[1] != self.config.latent_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "GpuVaeDecoder::decode: expected [B, {}, H, W], got {:?}",
                    self.config.latent_channels, shape
                ),
            });
        }
        let b = shape[0];
        let h = shape[2];
        let w = shape[3];
        let inv = 1.0_f64 / self.config.scaling_factor;
        let inv_f32 = inv as f32;

        // Upload latent.
        let data = latent.data()?;
        let mut x = cpu_to_gpu(data, &self.device).map_err(gpu_err)?;
        // Apply scaling factor divide.
        x = gpu_scale(&x, inv_f32, &self.device).map_err(gpu_err)?;

        // post_quant_conv (1x1)
        let (mut hbuf, mut hshape) = conv_forward(
            &self.post_quant_conv, &x, [b, self.config.latent_channels, h, w], &self.device,
        )?;

        // conv_in (3x3 pad 1, latent -> top_c)
        (hbuf, hshape) = conv_forward(&self.conv_in, &hbuf, hshape, &self.device)?;

        // Mid block: resnet0 -> attn -> resnet1
        (hbuf, hshape) = resnet_forward(&self.mid_block.resnets[0], &hbuf, hshape, &self.device)?;
        (hbuf, hshape) = attn_forward(&self.mid_block.attentions[0], &hbuf, hshape, &self.device)?;
        (hbuf, hshape) = resnet_forward(&self.mid_block.resnets[1], &hbuf, hshape, &self.device)?;

        // Up blocks.
        for up in &self.up_blocks {
            for r in &up.resnets {
                (hbuf, hshape) = resnet_forward(r, &hbuf, hshape, &self.device)?;
            }
            if let Some(ups) = &up.upsample {
                (hbuf, hshape) = upsample_forward(ups, &hbuf, hshape, &self.device)?;
            }
        }

        // conv_norm_out -> SiLU -> conv_out
        hbuf = group_norm_forward(&self.conv_norm_out, &hbuf, hshape, &self.device)?;
        hbuf = gpu_silu(&hbuf, &self.device).map_err(gpu_err)?;
        let (out_buf, out_shape) =
            conv_forward(&self.conv_out, &hbuf, hshape, &self.device)?;

        // Download.
        let out_data = gpu_to_cpu(&out_buf, &self.device).map_err(gpu_err)?;
        Tensor::from_storage(TensorStorage::cpu(out_data), out_shape.to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers (state-dict popping, forward ops)
// ---------------------------------------------------------------------------

pub(super) fn gpu_err(e: GpuError) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("GpuVae GPU op failed: {e}"),
    }
}

/// Remove a key from the state-dict and upload it as a CUDA buffer.
pub(super) fn pop_tensor(
    state: &mut StateDict<f32>,
    key: &str,
    expected_len: usize,
    device: &GpuDevice,
) -> FerrotorchResult<CudaBuffer<f32>> {
    let t = state
        .remove(key)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("GpuVaeDecoder: missing tensor {key:?}"),
        })?;
    let data = t.data()?;
    if data.len() != expected_len {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "GpuVaeDecoder: tensor {key:?} length {} != expected {expected_len}",
                data.len()
            ),
        });
    }
    cpu_to_gpu(data, device).map_err(gpu_err)
}

pub(super) fn pop_conv(
    state: &mut StateDict<f32>,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    kernel: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    device: &GpuDevice,
) -> FerrotorchResult<GpuConv2d> {
    let weight_len = out_c * in_c * kernel.0 * kernel.1;
    let weight = pop_tensor(state, &format!("{prefix}.weight"), weight_len, device)?;
    let bias = pop_tensor(state, &format!("{prefix}.bias"), out_c, device)?;
    Ok(GpuConv2d {
        weight,
        bias,
        in_channels: in_c,
        out_channels: out_c,
        kernel,
        stride,
        padding,
    })
}

pub(super) fn pop_linear(
    state: &mut StateDict<f32>,
    prefix: &str,
    in_f: usize,
    out_f: usize,
    device: &GpuDevice,
) -> FerrotorchResult<GpuLinear> {
    let weight = pop_tensor(state, &format!("{prefix}.weight"), out_f * in_f, device)?;
    let bias = pop_tensor(state, &format!("{prefix}.bias"), out_f, device)?;
    Ok(GpuLinear {
        weight,
        bias,
        in_features: in_f,
        out_features: out_f,
    })
}

pub(super) fn pop_groupnorm(
    state: &mut StateDict<f32>,
    prefix: &str,
    groups: usize,
    channels: usize,
    eps: f32,
    device: &GpuDevice,
) -> FerrotorchResult<GpuGroupNorm> {
    let weight = pop_tensor(state, &format!("{prefix}.weight"), channels, device)?;
    let bias = pop_tensor(state, &format!("{prefix}.bias"), channels, device)?;
    Ok(GpuGroupNorm {
        weight,
        bias,
        num_groups: groups,
        num_channels: channels,
        eps,
    })
}

pub(super) fn pop_resnet(
    state: &mut StateDict<f32>,
    prefix: &str,
    in_c: usize,
    out_c: usize,
    groups: usize,
    eps: f32,
    device: &GpuDevice,
) -> FerrotorchResult<GpuResnet> {
    let norm1 = pop_groupnorm(state, &format!("{prefix}.norm1"), groups, in_c, eps, device)?;
    let conv1 = pop_conv(
        state, &format!("{prefix}.conv1"), in_c, out_c, (3, 3), (1, 1), (1, 1), device,
    )?;
    let norm2 = pop_groupnorm(state, &format!("{prefix}.norm2"), groups, out_c, eps, device)?;
    let conv2 = pop_conv(
        state, &format!("{prefix}.conv2"), out_c, out_c, (3, 3), (1, 1), (1, 1), device,
    )?;
    let conv_shortcut = if in_c == out_c {
        None
    } else {
        Some(pop_conv(
            state,
            &format!("{prefix}.conv_shortcut"),
            in_c,
            out_c,
            (1, 1),
            (1, 1),
            (0, 0),
            device,
        )?)
    };
    Ok(GpuResnet {
        norm1,
        conv1,
        norm2,
        conv2,
        conv_shortcut,
        in_channels: in_c,
        out_channels: out_c,
    })
}

pub(super) fn pop_attn(
    state: &mut StateDict<f32>,
    prefix: &str,
    channels: usize,
    groups: usize,
    eps: f32,
    device: &GpuDevice,
) -> FerrotorchResult<GpuAttn> {
    let group_norm = pop_groupnorm(
        state, &format!("{prefix}.group_norm"), groups, channels, eps, device,
    )?;
    let to_q = pop_linear(state, &format!("{prefix}.to_q"), channels, channels, device)?;
    let to_k = pop_linear(state, &format!("{prefix}.to_k"), channels, channels, device)?;
    let to_v = pop_linear(state, &format!("{prefix}.to_v"), channels, channels, device)?;
    let to_out_0 = pop_linear(
        state, &format!("{prefix}.to_out.0"), channels, channels, device,
    )?;
    Ok(GpuAttn {
        group_norm,
        to_q,
        to_k,
        to_v,
        to_out_0,
        channels,
    })
}

// ---- Forward op dispatchers --------------------------------------------------

pub(super) fn conv_forward(
    c: &GpuConv2d,
    x: &CudaBuffer<f32>,
    shape: [usize; 4],
    device: &GpuDevice,
) -> FerrotorchResult<(CudaBuffer<f32>, [usize; 4])> {
    let (out, out_shape) = gpu_conv2d_f32(
        x,
        &c.weight,
        Some(&c.bias),
        shape,
        [c.out_channels, c.in_channels, c.kernel.0, c.kernel.1],
        c.stride,
        c.padding,
        (1, 1),
        1,
        device,
    )
    .map_err(gpu_err)?;
    Ok((out, out_shape))
}

pub(super) fn group_norm_forward(
    g: &GpuGroupNorm,
    x: &CudaBuffer<f32>,
    shape: [usize; 4],
    device: &GpuDevice,
) -> FerrotorchResult<CudaBuffer<f32>> {
    let [b, c, h, w] = shape;
    if c != g.num_channels {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "GpuVaeDecoder::group_norm: expected C={}, got {}",
                g.num_channels, c
            ),
        });
    }
    gpu_group_norm_f32(x, &g.weight, &g.bias, b, c, g.num_groups, h * w, g.eps, device)
        .map_err(gpu_err)
}

/// Apply `y = x @ W^T + b` for an `[m, in_f]` x against an `[out_f,
/// in_f]` row-major W. The CPU `Linear` calls
/// `linear_fused(x, W, b)`; this is the same op in GPU land.
///
/// Implementation: we materialise `W^T` (`[in_f, out_f]`) via a host
/// bounce per call (~MB per attn-Linear at SD VAE scales — five
/// invocations per decode at C=512). A per-call transpose kernel is
/// the obvious perf follow-up; correctness first.
fn linear_xwt_plus_b(
    weight: &CudaBuffer<f32>,
    weight_shape: (usize, usize), // (out_f, in_f)
    bias: &CudaBuffer<f32>,
    x: &CudaBuffer<f32>,
    m: usize,
    device: &GpuDevice,
) -> FerrotorchResult<CudaBuffer<f32>> {
    let (out_f, in_f) = weight_shape;
    let w_host = gpu_to_cpu(weight, device).map_err(gpu_err)?;
    let mut w_t = vec![0.0_f32; in_f * out_f];
    for o in 0..out_f {
        for i in 0..in_f {
            w_t[i * out_f + o] = w_host[o * in_f + i];
        }
    }
    let w_t_gpu = cpu_to_gpu(&w_t, device).map_err(gpu_err)?;
    let y = gpu_matmul_f32(x, &w_t_gpu, m, in_f, out_f, device).map_err(gpu_err)?;
    let y_shape: [usize; 2] = [m, out_f];
    let b_shape: [usize; 2] = [1, out_f];
    gpu_broadcast_add(&y, bias, &y_shape, &b_shape, &y_shape, device).map_err(gpu_err)
}

pub(super) fn resnet_forward(
    r: &GpuResnet,
    x: &CudaBuffer<f32>,
    shape: [usize; 4],
    device: &GpuDevice,
) -> FerrotorchResult<(CudaBuffer<f32>, [usize; 4])> {
    // h = norm1(x); silu; conv1
    let mut h = group_norm_forward(&r.norm1, x, shape, device)?;
    h = gpu_silu(&h, device).map_err(gpu_err)?;
    let mut h_shape;
    (h, h_shape) = conv_forward(&r.conv1, &h, shape, device)?;
    // h = norm2(h); silu; conv2
    h = group_norm_forward(&r.norm2, &h, h_shape, device)?;
    h = gpu_silu(&h, device).map_err(gpu_err)?;
    (h, h_shape) = conv_forward(&r.conv2, &h, h_shape, device)?;
    // Residual: x or shortcut(x)
    let res = if let Some(sc) = &r.conv_shortcut {
        let (s, _) = conv_forward(sc, x, shape, device)?;
        s
    } else {
        // Same channels: need to clone x. We have only &CudaBuffer<f32>;
        // re-upload via host bounce would defeat the purpose. Instead:
        // perform the add as gpu_add(h, x) — the buffer is shared and
        // not mutated.
        let _ = r.in_channels;
        let _ = r.out_channels;
        // Allocate a new buffer that aliases x by copying through GPU.
        // The kernel API expects an owned `CudaBuffer<f32>`, so we
        // perform a device-to-device copy via a host bounce. To avoid
        // allocations, route through `gpu_add` directly below.
        return gpu_add(&h, x, device).map(|sum| (sum, h_shape)).map_err(gpu_err);
    };
    let sum = gpu_add(&h, &res, device).map_err(gpu_err)?;
    Ok((sum, h_shape))
}

pub(super) fn attn_forward(
    a: &GpuAttn,
    x: &CudaBuffer<f32>,
    shape: [usize; 4],
    device: &GpuDevice,
) -> FerrotorchResult<(CudaBuffer<f32>, [usize; 4])> {
    let [b, c, h, w] = shape;
    if c != a.channels {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "GpuVaeDecoder::attn: expected C={}, got {}",
                a.channels, c
            ),
        });
    }
    let hw = h * w;

    // Group-norm over [B, C, H, W] directly (operates on channel axis
    // with HW per group).
    let normed_bchw = group_norm_forward(&a.group_norm, x, shape, device)?;
    // Reshape to [B, HW, C] for Linear. The buffer is logically
    // [B, C, HW]; we need to transpose the last two axes. Implement
    // via host bounce since ferrotorch-gpu lacks a public 3-D
    // transpose helper at this surface. For SD VAE only one mid-block
    // attn fires per decode at C=512, HW=64x64=4096 → 4096*512*4 =
    // ~8 MB per direction × ~2 transposes = 32 MB host bounce. OK
    // for correctness; perf is a follow-up.
    let mut normed = vec![0.0_f32; b * c * hw];
    {
        let host_bchw = gpu_to_cpu(&normed_bchw, device).map_err(gpu_err)?;
        for bi in 0..b {
            for hwi in 0..hw {
                for ci in 0..c {
                    let src = ((bi * c + ci) * hw) + hwi;
                    let dst = ((bi * hw + hwi) * c) + ci;
                    normed[dst] = host_bchw[src];
                }
            }
        }
    }
    let normed_hwc = cpu_to_gpu(&normed, device).map_err(gpu_err)?;

    // q, k, v projections: each is Linear(C, C) applied to a
    // [B*HW, C] flat buffer (Linear is row-wise).
    let m = b * hw;
    let q = linear_xwt_plus_b(&a.to_q.weight, (a.to_q.out_features, a.to_q.in_features),
                              &a.to_q.bias, &normed_hwc, m, device)?;
    let k = linear_xwt_plus_b(&a.to_k.weight, (a.to_k.out_features, a.to_k.in_features),
                              &a.to_k.bias, &normed_hwc, m, device)?;
    let v = linear_xwt_plus_b(&a.to_v.weight, (a.to_v.out_features, a.to_v.in_features),
                              &a.to_v.bias, &normed_hwc, m, device)?;

    // Single-head attention:
    //   scores = (q @ k^T) * (1/sqrt(C))  -> [B, HW, HW]
    //   probs  = softmax(scores, dim=-1)
    //   out    = probs @ v               -> [B, HW, C]
    let scale = (c as f64).sqrt().recip() as f32;
    // q: [B, HW, C], k: [B, HW, C]. We want scores[b, i, j] = sum_c
    // q[b,i,c] * k[b,j,c]. Use gpu_bmm with k transposed: bmm(q, k^T)
    // where k^T has shape [B, C, HW]. Transpose via host bounce.
    let k_host = gpu_to_cpu(&k, device).map_err(gpu_err)?;
    let mut k_t = vec![0.0_f32; b * c * hw];
    for bi in 0..b {
        for hwi in 0..hw {
            for ci in 0..c {
                let src = ((bi * hw + hwi) * c) + ci;
                let dst = ((bi * c + ci) * hw) + hwi;
                k_t[dst] = k_host[src];
            }
        }
    }
    let k_t_gpu = cpu_to_gpu(&k_t, device).map_err(gpu_err)?;
    let scores = gpu_bmm_f32(&q, &k_t_gpu, b, hw, c, hw, device).map_err(gpu_err)?;
    let scaled = gpu_scale(&scores, scale, device).map_err(gpu_err)?;
    // Softmax over last dim: rows = B*HW, cols = HW.
    let probs = gpu_softmax(&scaled, b * hw, hw, device).map_err(gpu_err)?;
    // attended = probs @ v : [B, HW, HW] @ [B, HW, C] -> [B, HW, C]
    let attended = gpu_bmm_f32(&probs, &v, b, hw, hw, c, device).map_err(gpu_err)?;

    // to_out.0 projection: still [B*HW, C] -> [B*HW, C]
    let projected = linear_xwt_plus_b(
        &a.to_out_0.weight,
        (a.to_out_0.out_features, a.to_out_0.in_features),
        &a.to_out_0.bias,
        &attended,
        m,
        device,
    )?;

    // Reshape back from [B, HW, C] to [B, C, H, W]. Inverse of the
    // earlier transpose.
    let proj_host = gpu_to_cpu(&projected, device).map_err(gpu_err)?;
    let mut back = vec![0.0_f32; b * c * hw];
    for bi in 0..b {
        for hwi in 0..hw {
            for ci in 0..c {
                let src = ((bi * hw + hwi) * c) + ci;
                let dst = ((bi * c + ci) * hw) + hwi;
                back[dst] = proj_host[src];
            }
        }
    }
    let back_gpu = cpu_to_gpu(&back, device).map_err(gpu_err)?;

    // Add residual.
    let sum = gpu_add(&back_gpu, x, device).map_err(gpu_err)?;
    Ok((sum, shape))
}

fn upsample_forward(
    u: &GpuUpsample,
    x: &CudaBuffer<f32>,
    shape: [usize; 4],
    device: &GpuDevice,
) -> FerrotorchResult<(CudaBuffer<f32>, [usize; 4])> {
    let [b, c, h, w] = shape;
    if c != u.channels {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "GpuVaeDecoder::upsample: expected C={}, got {}",
                u.channels, c
            ),
        });
    }
    let upsampled = gpu_nearest_upsample2x_f32(x, b, c, h, w, device).map_err(gpu_err)?;
    let new_shape = [b, c, h * 2, w * 2];
    conv_forward(&u.conv, &upsampled, new_shape, device)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    /// Tiny config that hits every op shape (mid attn, resnet shortcut,
    /// 3 upsamples) but stays fast.
    fn tiny_cfg() -> VaeDecoderConfig {
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
    fn gpu_vae_matches_cpu_tiny() {
        let Ok(device) = GpuDevice::new(0) else {
            return;
        };
        let cfg = tiny_cfg();
        let cpu = VaeDecoder::<f32>::new(cfg.clone()).unwrap();
        let (gpu, report) = GpuVaeDecoder::from_module(&cpu, &device).unwrap();
        assert!(
            report.dropped.is_empty(),
            "unexpected dropped keys: {:?}",
            report.dropped
        );

        // Build a deterministic latent.
        let n = cfg.latent_channels;
        let data: Vec<f32> = (0..n).map(|i| ((i % 5) as f32) * 0.07 - 0.1).collect();
        let latent = Tensor::from_storage(
            TensorStorage::cpu(data.clone()),
            vec![1, cfg.latent_channels, 1, 1],
            false,
        )
        .unwrap();

        let cpu_out = cpu.decode_with_scaling(&latent).unwrap();
        let gpu_out = gpu.decode(&latent).unwrap();
        assert_eq!(cpu_out.shape(), gpu_out.shape());
        let cpu_data = cpu_out.data().unwrap();
        let gpu_data = gpu_out.data().unwrap();
        let mut max_abs = 0.0_f32;
        for (a, b) in cpu_data.iter().zip(gpu_data.iter()) {
            let d = (a - b).abs();
            if d > max_abs {
                max_abs = d;
            }
        }
        assert!(max_abs < 1e-3, "gpu vs cpu tiny VAE max_abs = {max_abs}");
    }
}
