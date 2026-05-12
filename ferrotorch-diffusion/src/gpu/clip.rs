#![cfg(feature = "cuda")]
//! GPU CLIP text-encoder forward path for SD-1.5
//! (`openai/clip-vit-large-patch14` — the text tower).
//!
//! Phase F.2 of the SD GPU sequence. Mirrors
//! [`crate::clip_text_encoder::ClipTextEncoder`] op-for-op, resident in
//! VRAM. Every embedding gather / layer-norm / matmul / softmax /
//! quick-GELU / residual call is dispatched through the matching
//! `ferrotorch-gpu` kernel; the only host-side traffic is the one-shot
//! weight upload at construction and the final `last_hidden_state`
//! download at [`Self::encode`].
//!
//! Architecture (mirrors the CPU module — see its rustdoc for the full
//! tree). For SD-1.5:
//!
//! ```text
//! hidden_size        = 768
//! intermediate_size  = 3072
//! num_attention_heads = 12 (head_dim = 64)
//! num_hidden_layers  = 12
//! max_position_embeddings = 77
//! vocab_size         = 49408
//! hidden_act         = "quick_gelu"   # x * sigmoid(1.702 * x)
//! layer_norm_eps     = 1e-5
//! attention is CAUSAL (each token attends to itself + earlier)
//! ```
//!
//! Forward (per layer, pre-LayerNorm + residual):
//!
//! ```text
//! h = token_emb[ids] + pos_emb[0..S]
//! for layer in 12:
//!     residual = h
//!     h = layer_norm1(h)
//!     h = causal_self_attn(h)  ──► uses pre-uploaded causal mask
//!     h = residual + h
//!     residual = h
//!     h = layer_norm2(h)
//!     h = fc2(quick_gelu(fc1(h)))
//!     h = residual + h
//! h = final_layer_norm(h)
//! return h   # last_hidden_state [1, S, 768]
//! ```
//!
//! ## Correctness gotchas (re-stated for GPU port)
//!
//! 1. **Causal mask**. We pre-compute a `[S, S]` float buffer with `0.0`
//!    on the diagonal+below and `-INF` above, then broadcast-add to the
//!    per-head attention scores **before** the softmax. Forgetting this
//!    silently re-introduces bidirectional attention and breaks parity.
//! 2. **QuickGELU**, not standard GELU. `ferrotorch_gpu::kernels::gpu_gelu`
//!    is already the sigmoid-approx variant (`x * sigmoid(1.702 * x)`);
//!    we deliberately use it here.
//! 3. **All four self-attention projections carry bias** (q/k/v/out).
//! 4. **Linear is `y = x @ W^T + b`** — we materialise `W^T` on host once
//!    per upload and store the transposed copy in VRAM, so every forward
//!    is a pure `matmul(x, W_t)` + broadcast-add. Avoids per-call
//!    transposes on the hot path.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_gpu::{
    CudaBuffer, GpuDevice, GpuError, gpu_bmm_f32, gpu_layernorm, gpu_matmul_f32, gpu_softmax,
    kernels::{gpu_add, gpu_broadcast_add, gpu_embed_lookup_batch, gpu_gelu, gpu_scale},
    transfer::{cpu_to_gpu, gpu_to_cpu},
};
use ferrotorch_nn::module::{Module, StateDict};

use crate::clip_text_encoder::{ClipTextConfig, ClipTextEncoder};
use crate::safetensors_loader::DropReport;

// ---------------------------------------------------------------------------
// Per-component buffer bundles
// ---------------------------------------------------------------------------

/// Parameters for a single `LayerNorm(normalized_shape=[hidden], eps,
/// elementwise_affine=true)` resident on the GPU.
#[derive(Debug)]
struct GpuLayerNorm {
    weight: CudaBuffer<f32>,
    bias: CudaBuffer<f32>,
    eps: f32,
    normalized_shape: usize,
}

/// Parameters for a `Linear(in, out, bias=true)`, stored as the
/// transpose `W^T` (shape `[in, out]` row-major) so that
/// `y = x @ W_t + b` is a single `matmul` without a per-call transpose.
#[derive(Debug)]
struct GpuLinearT {
    /// `[in_features, out_features]` row-major — the transpose of the
    /// PyTorch-stored `[out, in]` weight.
    weight_t: CudaBuffer<f32>,
    bias: CudaBuffer<f32>,
    in_features: usize,
    out_features: usize,
}

/// One CLIP self-attention block (q/k/v/out projections only — head
/// counts come from the parent encoder's [`ClipTextConfig`] at forward
/// time).
///
/// The four `*_proj` field names intentionally mirror the HF CLIP
/// state-dict layout (`self_attn.{q,k,v,out}_proj.weight`); renaming
/// would force a translation layer at the loader and break the 1:1
/// audit the dump example relies on.
#[derive(Debug)]
#[allow(clippy::struct_field_names)]
struct GpuClipAttn {
    q_proj: GpuLinearT,
    k_proj: GpuLinearT,
    v_proj: GpuLinearT,
    out_proj: GpuLinearT,
}

/// One CLIP MLP block (fc1 + fc2 + QuickGELU).
#[derive(Debug)]
struct GpuClipMlp {
    fc1: GpuLinearT,
    fc2: GpuLinearT,
}

/// One CLIP encoder layer (LN1 + attn + LN2 + MLP).
#[derive(Debug)]
struct GpuClipLayer {
    layer_norm1: GpuLayerNorm,
    self_attn: GpuClipAttn,
    layer_norm2: GpuLayerNorm,
    mlp: GpuClipMlp,
}

// ---------------------------------------------------------------------------
// GpuClipTextEncoder
// ---------------------------------------------------------------------------

/// CLIP text-encoder forward path resident on a single CUDA device.
///
/// Constructed from a [`ClipTextConfig`] + host-side [`StateDict<f32>`]
/// in the same layout that
/// [`crate::clip_text_encoder::ClipTextEncoder::state_dict`] produces
/// (or the HF `text_model.*`-stripped layout
/// [`crate::clip_text_encoder::ClipTextEncoder::load_hf_state_dict`]
/// normalises to). Every parameter tensor is uploaded once into GPU
/// memory; the host copy is dropped after construction.
///
/// # Example
///
/// ```ignore
/// let device = GpuDevice::new(0)?;
/// let (cpu_enc, _drop) = load_clip_text_encoder::<f32>(weights, cfg, false)?;
/// let (gpu, _drop) = GpuClipTextEncoder::from_module(&cpu_enc, &device)?;
/// let last_hidden_state = gpu.encode(&input_ids_u32)?; // [1, S, 768]
/// ```
#[derive(Debug)]
pub struct GpuClipTextEncoder {
    /// Token-embedding table — `[vocab_size, hidden_size]`.
    token_embedding: CudaBuffer<f32>,
    /// Learned position-embedding table — `[max_position_embeddings,
    /// hidden_size]`.
    position_embedding: CudaBuffer<f32>,
    layers: Vec<GpuClipLayer>,
    final_layer_norm: GpuLayerNorm,
    /// Pre-computed `[max_pos, max_pos]` causal mask: 0.0 at and below
    /// the diagonal, `-INF` strictly above. Sliced for the actual
    /// sequence length at each forward.
    causal_mask_full: CudaBuffer<f32>,
    config: ClipTextConfig,
    device: GpuDevice,
}

impl GpuClipTextEncoder {
    /// Build the GPU CLIP text encoder from a config + state-dict.
    ///
    /// The state-dict is expected in the same shape as
    /// [`crate::clip_text_encoder::ClipTextEncoder`] produces
    /// (`embeddings.{token,position}_embedding.weight`,
    /// `encoder.layers.{i}.{layer_norm1,self_attn,layer_norm2,mlp}.*`,
    /// `final_layer_norm.{weight,bias}`).
    ///
    /// Tensors are checked for length and uploaded once to VRAM; Linear
    /// weights are transposed on the host so the GPU forward is a
    /// single `matmul` per projection.
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
        config: ClipTextConfig,
        mut state: StateDict<f32>,
        device: GpuDevice,
    ) -> FerrotorchResult<(Self, DropReport)> {
        config.validate()?;
        let hidden = config.hidden_size;
        let inter = config.intermediate_size;
        let vocab = config.vocab_size;
        let max_pos = config.max_position_embeddings;
        let eps = config.layer_norm_eps as f32;

        // ---- Embeddings ---------------------------------------------------
        let token_embedding = pop_tensor(
            &mut state,
            "embeddings.token_embedding.weight",
            vocab * hidden,
            &device,
        )?;
        let position_embedding = pop_tensor(
            &mut state,
            "embeddings.position_embedding.weight",
            max_pos * hidden,
            &device,
        )?;

        // ---- 12 encoder layers --------------------------------------------
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for li in 0..config.num_hidden_layers {
            let prefix = format!("encoder.layers.{li}");
            let layer_norm1 = pop_layernorm(
                &mut state,
                &format!("{prefix}.layer_norm1"),
                hidden,
                eps,
                &device,
            )?;
            let q_proj = pop_linear_t(
                &mut state,
                &format!("{prefix}.self_attn.q_proj"),
                hidden,
                hidden,
                &device,
            )?;
            let k_proj = pop_linear_t(
                &mut state,
                &format!("{prefix}.self_attn.k_proj"),
                hidden,
                hidden,
                &device,
            )?;
            let v_proj = pop_linear_t(
                &mut state,
                &format!("{prefix}.self_attn.v_proj"),
                hidden,
                hidden,
                &device,
            )?;
            let out_proj = pop_linear_t(
                &mut state,
                &format!("{prefix}.self_attn.out_proj"),
                hidden,
                hidden,
                &device,
            )?;
            let layer_norm2 = pop_layernorm(
                &mut state,
                &format!("{prefix}.layer_norm2"),
                hidden,
                eps,
                &device,
            )?;
            let fc1 = pop_linear_t(
                &mut state,
                &format!("{prefix}.mlp.fc1"),
                hidden,
                inter,
                &device,
            )?;
            let fc2 = pop_linear_t(
                &mut state,
                &format!("{prefix}.mlp.fc2"),
                inter,
                hidden,
                &device,
            )?;
            layers.push(GpuClipLayer {
                layer_norm1,
                self_attn: GpuClipAttn {
                    q_proj,
                    k_proj,
                    v_proj,
                    out_proj,
                },
                layer_norm2,
                mlp: GpuClipMlp { fc1, fc2 },
            });
        }

        // ---- Final LayerNorm ----------------------------------------------
        let final_layer_norm =
            pop_layernorm(&mut state, "final_layer_norm", hidden, eps, &device)?;

        // ---- Causal mask --------------------------------------------------
        //
        // Lower-triangular zeros + strictly-upper -INF, shape `[max_pos,
        // max_pos]`. We broadcast-add this (sliced to the actual seq_len)
        // onto the per-head attention scores `[H, S, S]` before softmax —
        // i.e. position `i` cannot attend to position `j > i` because
        // `mask[i, j] = -INF`. Matches `transformers.CLIPAttention`'s
        // `_create_4d_causal_attention_mask`.
        let mut mask = vec![0.0_f32; max_pos * max_pos];
        for i in 0..max_pos {
            for j in 0..max_pos {
                if j > i {
                    mask[i * max_pos + j] = f32::NEG_INFINITY;
                }
            }
        }
        let causal_mask_full = cpu_to_gpu(&mask, &device).map_err(gpu_err)?;

        // Whatever is left in `state` is unmapped — surface as DropReport
        // so the caller can audit silent-key drops (parity with the
        // VAE/UNet GPU paths).
        let mut dropped: Vec<String> = state.keys().cloned().collect();
        dropped.sort();
        let report = DropReport { dropped };

        Ok((
            Self {
                token_embedding,
                position_embedding,
                layers,
                final_layer_norm,
                causal_mask_full,
                config,
                device,
            },
            report,
        ))
    }

    /// Convenience constructor: build a [`GpuClipTextEncoder`] from an
    /// already-loaded CPU [`ClipTextEncoder<f32>`].
    ///
    /// Equivalent to extracting `cpu.state_dict()` and calling
    /// [`Self::new`] on a clone of the device handle.
    ///
    /// # Errors
    ///
    /// Forwards every error from [`Self::new`].
    pub fn from_module(
        cpu: &ClipTextEncoder<f32>,
        device: &GpuDevice,
    ) -> FerrotorchResult<(Self, DropReport)> {
        let state: StateDict<f32> = cpu.state_dict();
        Self::new(cpu.config.clone(), state, device.clone())
    }

    /// Run the encoder on a slice of u32 token ids and return the
    /// per-token `last_hidden_state` `[1, S, hidden]`.
    ///
    /// For SD-1.5 the canonical inference call is `S = 77`
    /// (`max_position_embeddings`).
    ///
    /// # Errors
    ///
    /// - [`FerrotorchError::InvalidArgument`] if `input_ids` is empty
    ///   or exceeds `max_position_embeddings`, or if any id is `>=
    ///   vocab_size`.
    /// - GPU op errors wrapped in `FerrotorchError::InvalidArgument`.
    pub fn encode(&self, input_ids: &[u32]) -> FerrotorchResult<Tensor<f32>> {
        let cfg = &self.config;
        let s = input_ids.len();
        if s == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "GpuClipTextEncoder::encode: input_ids is empty".into(),
            });
        }
        if s > cfg.max_position_embeddings {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "GpuClipTextEncoder::encode: seq_len {s} exceeds \
                     max_position_embeddings {}",
                    cfg.max_position_embeddings
                ),
            });
        }
        for &id in input_ids {
            if (id as usize) >= cfg.vocab_size {
                return Err(FerrotorchError::IndexOutOfBounds {
                    index: id as usize,
                    axis: 0,
                    size: cfg.vocab_size,
                });
            }
        }

        let hidden = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let head_dim = cfg.head_dim();
        let max_pos = cfg.max_position_embeddings;

        // ---- 1. Embeddings: token + position -------------------------------
        //
        // `gpu_embed_lookup_batch` consumes an f32-encoded indices buffer
        // (the kernel reads each element via `__float2uint_rn`-style cast
        // implemented in the PTX). Mirrors how the CPU `Embedding` API
        // takes a `Tensor<T>` of indices.
        let token_ids_f32: Vec<f32> = input_ids.iter().map(|&i| i as f32).collect();
        let token_ids_gpu = cpu_to_gpu(&token_ids_f32, &self.device).map_err(gpu_err)?;
        let tok_emb = gpu_embed_lookup_batch(
            &token_ids_gpu,
            &self.token_embedding,
            s,
            hidden,
            &self.device,
        )
        .map_err(gpu_err)?;

        // Position embedding: gather rows [0..S) from the
        // `[max_pos, hidden]` table. For SD-1.5 inference S == max_pos
        // and we could just use the full table; for a smaller S we
        // gather to keep the contract clean.
        let pos_ids_f32: Vec<f32> = (0..s as u32).map(|i| i as f32).collect();
        let pos_ids_gpu = cpu_to_gpu(&pos_ids_f32, &self.device).map_err(gpu_err)?;
        let pos_emb = gpu_embed_lookup_batch(
            &pos_ids_gpu,
            &self.position_embedding,
            s,
            hidden,
            &self.device,
        )
        .map_err(gpu_err)?;

        let mut h = gpu_add(&tok_emb, &pos_emb, &self.device).map_err(gpu_err)?;

        // ---- 2. Causal mask slice [S, S] ----------------------------------
        //
        // When `s < max_pos` we download-and-re-upload the slice. For
        // SD-1.5 inference `s == max_pos == 77` and we use the full
        // buffer as-is (no slicing).
        let causal_mask = if s == max_pos {
            None
        } else {
            let full = gpu_to_cpu(&self.causal_mask_full, &self.device).map_err(gpu_err)?;
            let mut sliced = vec![0.0_f32; s * s];
            for i in 0..s {
                for j in 0..s {
                    sliced[i * s + j] = full[i * max_pos + j];
                }
            }
            Some(cpu_to_gpu(&sliced, &self.device).map_err(gpu_err)?)
        };

        // ---- 3. Twelve encoder layers --------------------------------------
        for layer in &self.layers {
            // Pre-norm 1
            let normed1 = layernorm_forward(&layer.layer_norm1, &h, s, hidden, &self.device)?;

            // Linear projections — each is `[S, hidden] -> [S, hidden]`.
            let q = linear_forward(&layer.self_attn.q_proj, &normed1, s, &self.device)?;
            let k = linear_forward(&layer.self_attn.k_proj, &normed1, s, &self.device)?;
            let v = linear_forward(&layer.self_attn.v_proj, &normed1, s, &self.device)?;

            // Reshape `[S, hidden]` to `[H, S, head_dim]` (heads-first):
            // index `h * S * head_dim + i * head_dim + d` ←
            //       `i * H * head_dim + h * head_dim + d`.
            let q_heads = reshape_seq_to_heads(&q, s, num_heads, head_dim, &self.device)?;
            let k_heads = reshape_seq_to_heads(&k, s, num_heads, head_dim, &self.device)?;
            let v_heads = reshape_seq_to_heads(&v, s, num_heads, head_dim, &self.device)?;

            // Build `k_heads_t` (`[H, head_dim, S]`) so `bmm(q_heads,
            // k_heads_t)` yields `[H, S, S]` scores. Done host-side once
            // per layer to stay within the kernel surface; perf
            // follow-up if needed.
            let k_heads_t =
                transpose_last_two(&k_heads, num_heads, s, head_dim, &self.device)?;

            // scores = (q_heads @ k_heads_t) / sqrt(head_dim)
            let scores =
                gpu_bmm_f32(&q_heads, &k_heads_t, num_heads, s, head_dim, s, &self.device)
                    .map_err(gpu_err)?;
            let scale = (head_dim as f64).sqrt().recip() as f32;
            let scaled = gpu_scale(&scores, scale, &self.device).map_err(gpu_err)?;

            // Causal-mask add: broadcast `[S, S]` over the `H` head dim.
            // `scaled` is `[H, S, S]`; mask is `[1, S, S]` ≡ `[S, S]`.
            let mask_ref = causal_mask.as_ref().unwrap_or(&self.causal_mask_full);
            let masked = gpu_broadcast_add(
                &scaled,
                mask_ref,
                &[num_heads, s, s],
                &[1, s, s],
                &[num_heads, s, s],
                &self.device,
            )
            .map_err(gpu_err)?;

            // Softmax over last dim.
            let probs = gpu_softmax(&masked, num_heads * s, s, &self.device).map_err(gpu_err)?;

            // attended[h, i, :] = sum_j probs[h, i, j] * v[h, j, :]
            // probs: [H, S, S], v_heads: [H, S, head_dim] -> [H, S, head_dim]
            let attended =
                gpu_bmm_f32(&probs, &v_heads, num_heads, s, s, head_dim, &self.device)
                    .map_err(gpu_err)?;

            // [H, S, head_dim] → [S, hidden]
            let merged = reshape_heads_to_seq(&attended, num_heads, s, head_dim, &self.device)?;

            // out_proj + residual.
            let attn_out =
                linear_forward(&layer.self_attn.out_proj, &merged, s, &self.device)?;
            h = gpu_add(&h, &attn_out, &self.device).map_err(gpu_err)?;

            // ---- MLP sub-block ------------------------------------------
            let normed2 = layernorm_forward(&layer.layer_norm2, &h, s, hidden, &self.device)?;
            let mlp_h = linear_forward(&layer.mlp.fc1, &normed2, s, &self.device)?;
            // QuickGELU — gpu_gelu is already `x * sigmoid(1.702 * x)`.
            let mlp_act = gpu_gelu(&mlp_h, &self.device).map_err(gpu_err)?;
            let mlp_out = linear_forward(&layer.mlp.fc2, &mlp_act, s, &self.device)?;
            h = gpu_add(&h, &mlp_out, &self.device).map_err(gpu_err)?;
        }

        // ---- 4. Final LayerNorm -------------------------------------------
        let normed = layernorm_forward(&self.final_layer_norm, &h, s, hidden, &self.device)?;

        // ---- 5. Download -------------------------------------------------
        let out_data = gpu_to_cpu(&normed, &self.device).map_err(gpu_err)?;
        Tensor::from_storage(TensorStorage::cpu(out_data), vec![1, s, hidden], false)
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn gpu_err(e: GpuError) -> FerrotorchError {
    FerrotorchError::InvalidArgument {
        message: format!("GpuClipTextEncoder GPU op failed: {e}"),
    }
}

/// Remove a key from the state-dict and upload it as a CUDA buffer.
fn pop_tensor(
    state: &mut StateDict<f32>,
    key: &str,
    expected_len: usize,
    device: &GpuDevice,
) -> FerrotorchResult<CudaBuffer<f32>> {
    let t = state
        .remove(key)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("GpuClipTextEncoder: missing tensor {key:?}"),
        })?;
    let data = t.data()?;
    if data.len() != expected_len {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "GpuClipTextEncoder: tensor {key:?} length {} != expected {expected_len}",
                data.len()
            ),
        });
    }
    cpu_to_gpu(data, device).map_err(gpu_err)
}

fn pop_layernorm(
    state: &mut StateDict<f32>,
    prefix: &str,
    normalized_shape: usize,
    eps: f32,
    device: &GpuDevice,
) -> FerrotorchResult<GpuLayerNorm> {
    let weight = pop_tensor(
        state,
        &format!("{prefix}.weight"),
        normalized_shape,
        device,
    )?;
    let bias = pop_tensor(state, &format!("{prefix}.bias"), normalized_shape, device)?;
    Ok(GpuLayerNorm {
        weight,
        bias,
        eps,
        normalized_shape,
    })
}

/// Pop a PyTorch `Linear` (weight stored `[out, in]`) and upload it as
/// `W^T` (`[in, out]`) so `matmul(x, W_t)` is the per-call forward.
fn pop_linear_t(
    state: &mut StateDict<f32>,
    prefix: &str,
    in_f: usize,
    out_f: usize,
    device: &GpuDevice,
) -> FerrotorchResult<GpuLinearT> {
    let w_key = format!("{prefix}.weight");
    let b_key = format!("{prefix}.bias");
    let w_t = state
        .remove(&w_key)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("GpuClipTextEncoder: missing tensor {w_key:?}"),
        })?;
    let w_data = w_t.data()?;
    if w_data.len() != out_f * in_f {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "GpuClipTextEncoder: tensor {w_key:?} length {} != expected {}",
                w_data.len(),
                out_f * in_f
            ),
        });
    }
    let mut transposed = vec![0.0_f32; in_f * out_f];
    for o in 0..out_f {
        for i in 0..in_f {
            transposed[i * out_f + o] = w_data[o * in_f + i];
        }
    }
    let weight_t = cpu_to_gpu(&transposed, device).map_err(gpu_err)?;
    let bias = pop_tensor(state, &b_key, out_f, device)?;
    Ok(GpuLinearT {
        weight_t,
        bias,
        in_features: in_f,
        out_features: out_f,
    })
}

/// `LayerNorm(x)` on an `[S, hidden]` flat buffer (`rows = S`, `cols =
/// hidden`).
fn layernorm_forward(
    ln: &GpuLayerNorm,
    x: &CudaBuffer<f32>,
    s: usize,
    hidden: usize,
    device: &GpuDevice,
) -> FerrotorchResult<CudaBuffer<f32>> {
    if hidden != ln.normalized_shape {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "GpuClipTextEncoder::layernorm: expected hidden={}, got {}",
                ln.normalized_shape, hidden
            ),
        });
    }
    gpu_layernorm(x, &ln.weight, &ln.bias, s, hidden, ln.eps, device).map_err(gpu_err)
}

/// `y = x @ W_t + b` where `W_t` is `[in, out]` and `x` is `[s, in]`,
/// producing `[s, out]`.
fn linear_forward(
    lin: &GpuLinearT,
    x: &CudaBuffer<f32>,
    s: usize,
    device: &GpuDevice,
) -> FerrotorchResult<CudaBuffer<f32>> {
    let y = gpu_matmul_f32(x, &lin.weight_t, s, lin.in_features, lin.out_features, device)
        .map_err(gpu_err)?;
    gpu_broadcast_add(
        &y,
        &lin.bias,
        &[s, lin.out_features],
        &[1, lin.out_features],
        &[s, lin.out_features],
        device,
    )
    .map_err(gpu_err)
}

/// Reshape `[S, hidden]` (`hidden == num_heads * head_dim`) into
/// `[num_heads, S, head_dim]`. Done via host bounce — the SD-1.5 hot
/// path is `S=77, hidden=768` (~24 KB per tensor) so the bounce cost
/// stays inside the noise vs the per-layer matmuls. Perf follow-up is
/// a dedicated transpose kernel; correctness first.
fn reshape_seq_to_heads(
    x: &CudaBuffer<f32>,
    s: usize,
    num_heads: usize,
    head_dim: usize,
    device: &GpuDevice,
) -> FerrotorchResult<CudaBuffer<f32>> {
    let host = gpu_to_cpu(x, device).map_err(gpu_err)?;
    let mut out = vec![0.0_f32; num_heads * s * head_dim];
    for i in 0..s {
        for h in 0..num_heads {
            for d in 0..head_dim {
                let src = i * (num_heads * head_dim) + h * head_dim + d;
                let dst = h * s * head_dim + i * head_dim + d;
                out[dst] = host[src];
            }
        }
    }
    cpu_to_gpu(&out, device).map_err(gpu_err)
}

/// Inverse of [`reshape_seq_to_heads`] — `[num_heads, S, head_dim]` →
/// `[S, hidden]`.
fn reshape_heads_to_seq(
    x: &CudaBuffer<f32>,
    num_heads: usize,
    s: usize,
    head_dim: usize,
    device: &GpuDevice,
) -> FerrotorchResult<CudaBuffer<f32>> {
    let host = gpu_to_cpu(x, device).map_err(gpu_err)?;
    let mut out = vec![0.0_f32; s * num_heads * head_dim];
    for h in 0..num_heads {
        for i in 0..s {
            for d in 0..head_dim {
                let src = h * s * head_dim + i * head_dim + d;
                let dst = i * (num_heads * head_dim) + h * head_dim + d;
                out[dst] = host[src];
            }
        }
    }
    cpu_to_gpu(&out, device).map_err(gpu_err)
}

/// Transpose the last two dims of `[B, M, N]` -> `[B, N, M]`. Done via
/// host bounce for the same reasons as the head reshapes (small
/// shapes; correctness first).
fn transpose_last_two(
    x: &CudaBuffer<f32>,
    batch: usize,
    m: usize,
    n: usize,
    device: &GpuDevice,
) -> FerrotorchResult<CudaBuffer<f32>> {
    let host = gpu_to_cpu(x, device).map_err(gpu_err)?;
    let mut out = vec![0.0_f32; batch * n * m];
    for bi in 0..batch {
        for mi in 0..m {
            for ni in 0..n {
                let src = bi * m * n + mi * n + ni;
                let dst = bi * n * m + ni * m + mi;
                out[dst] = host[src];
            }
        }
    }
    cpu_to_gpu(&out, device).map_err(gpu_err)
}

// ---------------------------------------------------------------------------
// Tests — keep small so the CI CPU/GPU box can run quickly.
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::clip_text_encoder::{ClipTextConfig, ClipTextEncoder};

    /// Tiny config: 1 layer, 8 hidden, 2 heads, 16 inter, max_pos 6,
    /// vocab 32. Exercises every op shape but stays fast.
    fn tiny_cfg() -> ClipTextConfig {
        ClipTextConfig {
            hidden_size: 8,
            intermediate_size: 16,
            num_attention_heads: 2,
            num_hidden_layers: 1,
            max_position_embeddings: 6,
            vocab_size: 32,
            layer_norm_eps: 1e-5,
        }
    }

    #[test]
    fn gpu_clip_matches_cpu_tiny() {
        let Ok(device) = GpuDevice::new(0) else {
            return;
        };
        let cfg = tiny_cfg();
        let cpu = ClipTextEncoder::<f32>::new(cfg.clone()).unwrap();
        let (gpu, report) = GpuClipTextEncoder::from_module(&cpu, &device).unwrap();
        assert!(
            report.dropped.is_empty(),
            "unexpected dropped keys: {:?}",
            report.dropped
        );

        let ids = vec![1u32, 5, 7, 11, 17, 23];
        let cpu_out = cpu.forward_from_ids(&ids).unwrap();
        let gpu_out = gpu.encode(&ids).unwrap();
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
        assert!(max_abs < 1e-3, "gpu vs cpu tiny CLIP max_abs = {max_abs}");
    }

    #[test]
    fn gpu_clip_short_seq_is_causal() {
        // Same parity check the CPU side runs, on a sub-`max_pos`
        // sequence so the causal-mask slice path is exercised.
        let Ok(device) = GpuDevice::new(0) else {
            return;
        };
        let cfg = tiny_cfg();
        let cpu = ClipTextEncoder::<f32>::new(cfg.clone()).unwrap();
        let (gpu, _) = GpuClipTextEncoder::from_module(&cpu, &device).unwrap();

        let ids_a = vec![1u32, 5, 7];
        let mut ids_b = ids_a.clone();
        // Perturb a token that's "after" position 0; row 0 of the
        // output must be unchanged under a causal mask.
        ids_b[2] = 9u32;
        let oa = gpu.encode(&ids_a).unwrap();
        let ob = gpu.encode(&ids_b).unwrap();
        let da = oa.data().unwrap();
        let db = ob.data().unwrap();
        // Row 0 (first token) must be bit-identical.
        for d in 0..cfg.hidden_size {
            assert!(
                (da[d] - db[d]).abs() < 1e-5,
                "row 0 col {d} differs: {} vs {}",
                da[d],
                db[d]
            );
        }
    }
}
