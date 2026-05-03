//! GPU-resident Llama 3 inference.
//!
//! [`LlamaGpuInferencer`] owns the full model weights as
//! `CudaSlice<u16>` (bf16 bit layout) in VRAM and runs the forward
//! pass directly against the `ferrotorch-gpu` hand-written PTX bf16
//! kernels. This is the llama.cpp-style architecture: no generic
//! `Tensor<T>` dispatch, no CPU round-trips between layers, no
//! external toolchain. Every op in the forward pass is a CUDA
//! kernel launch against `CudaSlice<u16>` buffers.
//!
//! Enabled by the `cuda` feature on `ferrotorch-llama`, which pulls
//! in `ferrotorch-gpu` and `cudarc`.

#![cfg(feature = "cuda")]

use cudarc::driver::CudaSlice;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor};
use ferrotorch_gpu::{
    GpuDevice, gpu_add_bf16, gpu_block_reduce_max_abs_bf16, gpu_causal_mask_bf16,
    gpu_embedding_gather_bf16, gpu_fatrelu_bf16, gpu_matmul_bf16_bf16_nt,
    gpu_matmul_bf16_bf16_strided_batched, gpu_matmul_bf16_bf16_strided_batched_nt, gpu_mul_bf16,
    gpu_relu_bf16, gpu_repeat_kv_bf16, gpu_rmsnorm_bf16, gpu_rope_half_bf16, gpu_silu_bf16,
    gpu_softmax_bf16, gpu_transpose_from_heads_bf16, gpu_transpose_to_heads_bf16,
};
use ferrotorch_nn::StateDict;
use half::bf16;

use crate::config::{LlamaActivation, LlamaConfig};

/// Per-layer activation taps collected during a profiled forward pass.
///
/// Shape summary:
/// * each `attn_f32` entry: `[n_heads * seq]` — max|attn_out[h, t, :]|
/// * each `mlp_f32` entry: `[seq, n_mlp_blocks]` — max|gated[t, block]|
/// * `bootstrap_hidden` (Some iff `bootstrap_k` was set): a `[seq,
///   hidden_size]` bf16 bits snapshot of the hidden state after the
///   bootstrap_k-th layer, intended for downstream routing
///   predictors that condition on the model's own intermediate
///   activations rather than raw token ids.
struct ForwardTaps {
    mlp_block_size: usize,
    n_mlp_blocks: usize,
    bootstrap_k: Option<usize>,
    /// When true, the MLP magnitude tap uses a local plainact
    /// surrogate `max_i in block_b |gated[i] * (down_proj^T @ y_full)[i]|`
    /// instead of `max_i |gated[i]|`. Local plainact aligns the
    /// per-block training target with the block's contribution to the
    /// layer output (multiplied through down_proj), which is closer
    /// to the ShadowLLM paper's `plainact = grad × activation`
    /// criterion than raw activation magnitude. Costs one extra
    /// matmul per layer.
    plainact_local: bool,
    attn_f32: Vec<CudaSlice<f32>>,
    mlp_f32: Vec<CudaSlice<f32>>,
    bootstrap_hidden: Option<CudaSlice<u16>>,
}

/// Result of a profiled forward: per-token, per-layer, per-block
/// magnitudes downloaded to host. Returned by
/// [`LlamaGpuInferencer::forward_from_ids_profiled`].
#[derive(Debug, Clone)]
pub struct ProfiledForwardResult {
    /// Sequence length the forward pass was run on.
    pub seq_len: usize,
    /// `[n_layers, seq, n_heads]` f32.
    pub attn_magnitudes: Vec<f32>,
    /// `[n_layers, seq, n_mlp_blocks]` f32.
    pub mlp_magnitudes: Vec<f32>,
    /// When `bootstrap_k` was `Some(k)` on the request, this is the
    /// bf16 bit pattern of the hidden state after layer `k-1` (i.e.
    /// the input to layer `k`). Shape `[seq, hidden_size]`.
    pub bootstrap_hidden: Option<Vec<u16>>,
    /// Hidden size carried for downstream consumers — avoids their
    /// having to re-derive it from the model config.
    pub bootstrap_k: Option<usize>,
    /// Hidden size of the model, copied from `config.hidden_size`. Used
    /// by downstream profiler consumers as the row stride for
    /// `bootstrap_hidden`.
    pub hidden_size: usize,
}

/// All the weights that make up one Llama decoder layer, uploaded to GPU.
pub struct LlamaGpuLayer {
    /// `[hidden]` — pre-attention RMSNorm weight.
    pub input_norm: CudaSlice<u16>,
    /// `[hidden, hidden]` — query projection weight.
    pub q_proj: CudaSlice<u16>,
    /// `[n_kv_heads * head_dim, hidden]` — key projection weight.
    pub k_proj: CudaSlice<u16>,
    /// `[n_kv_heads * head_dim, hidden]` — value projection weight.
    pub v_proj: CudaSlice<u16>,
    /// `[hidden, hidden]` — output projection weight.
    pub o_proj: CudaSlice<u16>,
    /// `[hidden]` — post-attention (pre-MLP) RMSNorm weight.
    pub post_attn_norm: CudaSlice<u16>,
    /// `[intermediate, hidden]` — MLP gate projection weight.
    pub gate_proj: CudaSlice<u16>,
    /// `[intermediate, hidden]` — MLP up projection weight.
    pub up_proj: CudaSlice<u16>,
    /// `[hidden, intermediate]` — MLP down projection weight.
    pub down_proj: CudaSlice<u16>,
}

// `cudarc::CudaSlice<u16>` does not implement `Debug`, so we can't
// `#[derive(Debug)]`. Manual impl prints field names and the bf16
// element count for each on-device buffer; the actual VRAM bytes are
// not reachable on the host without a download. This keeps debug
// output structurally informative without a CUDA round-trip.
impl std::fmt::Debug for LlamaGpuLayer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaGpuLayer")
            .field(
                "input_norm",
                &format_args!("<CudaSlice<u16> {} elems>", self.input_norm.len()),
            )
            .field(
                "q_proj",
                &format_args!("<CudaSlice<u16> {} elems>", self.q_proj.len()),
            )
            .field(
                "k_proj",
                &format_args!("<CudaSlice<u16> {} elems>", self.k_proj.len()),
            )
            .field(
                "v_proj",
                &format_args!("<CudaSlice<u16> {} elems>", self.v_proj.len()),
            )
            .field(
                "o_proj",
                &format_args!("<CudaSlice<u16> {} elems>", self.o_proj.len()),
            )
            .field(
                "post_attn_norm",
                &format_args!("<CudaSlice<u16> {} elems>", self.post_attn_norm.len()),
            )
            .field(
                "gate_proj",
                &format_args!("<CudaSlice<u16> {} elems>", self.gate_proj.len()),
            )
            .field(
                "up_proj",
                &format_args!("<CudaSlice<u16> {} elems>", self.up_proj.len()),
            )
            .field(
                "down_proj",
                &format_args!("<CudaSlice<u16> {} elems>", self.down_proj.len()),
            )
            .finish()
    }
}

/// A GPU-resident Llama model ready for inference.
pub struct LlamaGpuInferencer {
    /// Frozen copy of the model configuration the weights were uploaded for.
    pub config: LlamaConfig,
    /// Owning handle to the CUDA device + stream the buffers live on.
    pub device: GpuDevice,
    /// `[vocab_size, hidden]` — token embedding table.
    pub embed_tokens: CudaSlice<u16>,
    /// One per `num_hidden_layers` in `config`.
    pub layers: Vec<LlamaGpuLayer>,
    /// `[hidden]` — final RMSNorm weight (post-stack, pre-`lm_head`).
    pub norm: CudaSlice<u16>,
    /// `[vocab_size, hidden]` — vocabulary projection.
    pub lm_head: CudaSlice<u16>,
    /// `[max_seq, head_dim / 2]` — precomputed RoPE cosines.
    pub cos_cache: CudaSlice<u16>,
    /// `[max_seq, head_dim / 2]` — precomputed RoPE sines.
    pub sin_cache: CudaSlice<u16>,
}

// Manual Debug for the same reason as `LlamaGpuLayer`. `GpuDevice`
// itself implements Debug; only the `CudaSlice<u16>` fields need an
// elem-count placeholder.
impl std::fmt::Debug for LlamaGpuInferencer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlamaGpuInferencer")
            .field("config", &self.config)
            .field("device", &self.device)
            .field(
                "embed_tokens",
                &format_args!("<CudaSlice<u16> {} elems>", self.embed_tokens.len()),
            )
            .field("layers", &self.layers)
            .field(
                "norm",
                &format_args!("<CudaSlice<u16> {} elems>", self.norm.len()),
            )
            .field(
                "lm_head",
                &format_args!("<CudaSlice<u16> {} elems>", self.lm_head.len()),
            )
            .field(
                "cos_cache",
                &format_args!("<CudaSlice<u16> {} elems>", self.cos_cache.len()),
            )
            .field(
                "sin_cache",
                &format_args!("<CudaSlice<u16> {} elems>", self.sin_cache.len()),
            )
            .finish()
    }
}

impl LlamaGpuInferencer {
    /// Upload an HF-style `StateDict<bf16>` into GPU memory and build
    /// an inferencer ready to run. The `state` map is drained as each
    /// tensor is uploaded so the host copy can be freed incrementally.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when the config
    /// fails validation or a required tensor (`model.embed_tokens.weight`,
    /// per-layer projections, `model.norm.weight`, `lm_head.weight`) is
    /// missing from `state`. Returns [`FerrotorchError::ShapeMismatch`]
    /// if a tensor's declared shape does not match the config-derived
    /// expected shape. Returns [`FerrotorchError::Internal`] for any
    /// CUDA driver error during the host-to-device copies (via
    /// [`map_driver_err`]).
    pub fn new(
        config: LlamaConfig,
        mut state: StateDict<bf16>,
        device: GpuDevice,
    ) -> FerrotorchResult<Self> {
        config.validate()?;

        let embed_tokens = upload_bf16_tensor(
            &mut state,
            "model.embed_tokens.weight",
            &[config.vocab_size, config.hidden_size],
            &device,
        )?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            layers.push(upload_layer(&config, i, &mut state, &device)?);
        }

        let norm = upload_bf16_tensor(
            &mut state,
            "model.norm.weight",
            &[config.hidden_size],
            &device,
        )?;
        let lm_head = upload_bf16_tensor(
            &mut state,
            "lm_head.weight",
            &[config.vocab_size, config.hidden_size],
            &device,
        )?;

        let (cos_cache, sin_cache) = build_rope_caches(&config, &device)?;

        Ok(Self {
            config,
            device,
            embed_tokens,
            layers,
            norm,
            lm_head,
            cos_cache,
            sin_cache,
        })
    }

    /// Forward `ids` through the network and return the last-token
    /// logits as `Vec<f32>` of length `vocab_size`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when `ids` is
    /// empty or longer than `config.max_position_embeddings`.
    /// Otherwise returns [`FerrotorchError::Internal`] /
    /// [`FerrotorchError::ShapeMismatch`] /
    /// [`FerrotorchError::NotImplementedOnCuda`] for any GPU error
    /// surfaced by [`map_gpu_err`] / [`map_driver_err`] during kernel
    /// dispatch or the device → host logits copy.
    pub fn forward_from_ids(&self, ids: &[u32]) -> FerrotorchResult<Vec<f32>> {
        let seq = ids.len();
        let cfg = &self.config;
        let dev = &self.device;
        let hidden = cfg.hidden_size;
        let vocab = cfg.vocab_size;

        let final_norm = self.forward_core(ids, None)?;

        let logits = gpu_matmul_bf16_bf16_nt(&final_norm, &self.lm_head, seq, hidden, vocab, dev)
            .map_err(map_gpu_err)?;

        // Download just the last-token row of logits and convert bf16 -> f32.
        let logits_host: Vec<u16> = dev.stream().clone_dtoh(&logits).map_err(map_driver_err)?;
        let last_offset = (seq - 1) * vocab;
        let last_row = &logits_host[last_offset..last_offset + vocab];
        Ok(last_row
            .iter()
            .map(|&b| bf16::from_bits(b).to_f32())
            .collect())
    }

    /// Forward `ids` through the network with activation taps on every
    /// layer, returning per-(layer, token, block) magnitudes for the
    /// paged-weight sparsity profiler.
    ///
    /// `mlp_block_size * n_mlp_blocks` must equal `intermediate_size`.
    /// The lm_head is skipped — profiling doesn't need logits and the
    /// 128k-row projection is the single biggest kernel in the pass.
    ///
    /// # Errors
    ///
    /// Forwards every error from
    /// [`Self::forward_from_ids_profiled_with_bootstrap`].
    pub fn forward_from_ids_profiled(
        &self,
        ids: &[u32],
        mlp_block_size: usize,
        n_mlp_blocks: usize,
    ) -> FerrotorchResult<ProfiledForwardResult> {
        self.forward_from_ids_profiled_with_bootstrap(
            ids,
            mlp_block_size,
            n_mlp_blocks,
            None,
            false,
        )
    }

    /// Profiled forward with an optional bootstrap hidden-state tap.
    /// When `bootstrap_k` is `Some(k)`, the hidden state after layer
    /// `k-1` (i.e. the input to layer `k`) is cloned on-device and
    /// downloaded to the returned [`ProfiledForwardResult`]. This is
    /// the input signal the Phase-3 predictor consumes — the 8B's
    /// routing decisions at layer `k` are driven by this tensor, not
    /// by raw token ids.
    ///
    /// Valid bootstrap_k range: `1..=num_hidden_layers`. `bootstrap_k
    /// = num_hidden_layers` captures the final pre-norm state.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when
    /// `mlp_block_size == 0`, `n_mlp_blocks == 0`, the product does
    /// not equal `intermediate_size`, or `bootstrap_k` is outside the
    /// valid range. Otherwise returns whatever the per-layer kernel
    /// dispatch returns via [`map_gpu_err`] / [`map_driver_err`].
    pub fn forward_from_ids_profiled_with_bootstrap(
        &self,
        ids: &[u32],
        mlp_block_size: usize,
        n_mlp_blocks: usize,
        bootstrap_k: Option<usize>,
        plainact_local: bool,
    ) -> FerrotorchResult<ProfiledForwardResult> {
        let cfg = &self.config;
        let seq = ids.len();
        let n_heads = cfg.num_attention_heads;
        let n_layers = cfg.num_hidden_layers;
        let hidden_size = cfg.hidden_size;

        if mlp_block_size == 0 || n_mlp_blocks == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "forward_from_ids_profiled: block sizing must be positive".into(),
            });
        }
        if mlp_block_size * n_mlp_blocks != cfg.intermediate_size {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "forward_from_ids_profiled: mlp_block_size ({mlp_block_size}) * \
                     n_mlp_blocks ({n_mlp_blocks}) must equal intermediate_size ({})",
                    cfg.intermediate_size
                ),
            });
        }
        if let Some(k) = bootstrap_k {
            if k == 0 || k > n_layers {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("bootstrap_k must be in 1..={n_layers}, got {k}"),
                });
            }
        }

        let mut taps = ForwardTaps {
            mlp_block_size,
            n_mlp_blocks,
            bootstrap_k,
            plainact_local,
            attn_f32: Vec::with_capacity(n_layers),
            mlp_f32: Vec::with_capacity(n_layers),
            bootstrap_hidden: None,
        };
        // Run the forward; the helper fills `taps` with per-layer f32 buffers.
        let _final_norm = self.forward_core(ids, Some(&mut taps))?;

        // Download all tap buffers, then assemble in **token-major**
        // order (`[t, l, h]` for attn, `[t, l, b]` for mlp) — matches
        // the trace's documented `[total_tokens, n_layers, ...]`
        // layout so consumers like `attn_at(tok, layer, head)` and
        // `from_trace` indexing resolve correctly when seq > 1. The
        // earlier loop nesting was layer-major and silently scrambled
        // the data for any seq > 1 capture window.
        let stream = self.device.stream();
        let mut attn_per_layer: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        let mut mlp_per_layer: Vec<Vec<f32>> = Vec::with_capacity(n_layers);
        for l in 0..n_layers {
            attn_per_layer.push(
                stream
                    .clone_dtoh(&taps.attn_f32[l])
                    .map_err(map_driver_err)?,
            );
            mlp_per_layer.push(
                stream
                    .clone_dtoh(&taps.mlp_f32[l])
                    .map_err(map_driver_err)?,
            );
        }
        let mut attn_magnitudes = Vec::with_capacity(seq * n_layers * n_heads);
        let mut mlp_magnitudes = Vec::with_capacity(seq * n_layers * n_mlp_blocks);
        for t in 0..seq {
            for l in 0..n_layers {
                let alayer = &attn_per_layer[l];
                for h in 0..n_heads {
                    attn_magnitudes.push(alayer[h * seq + t]);
                }
                let mlayer = &mlp_per_layer[l];
                let off = t * n_mlp_blocks;
                mlp_magnitudes.extend_from_slice(&mlayer[off..off + n_mlp_blocks]);
            }
        }

        let bootstrap_hidden = if let Some(cuda_hidden) = taps.bootstrap_hidden.as_ref() {
            Some(
                self.device
                    .stream()
                    .clone_dtoh(cuda_hidden)
                    .map_err(map_driver_err)?,
            )
        } else {
            None
        };

        Ok(ProfiledForwardResult {
            seq_len: seq,
            attn_magnitudes,
            mlp_magnitudes,
            bootstrap_hidden,
            bootstrap_k,
            hidden_size,
        })
    }

    /// Core forward pass shared between `forward_from_ids` and
    /// `forward_from_ids_profiled`. Returns the final rmsnorm output;
    /// the caller is responsible for optional lm_head + download.
    ///
    /// If `taps` is `Some`, each layer pushes one f32 CudaSlice of
    /// per-(head, token) attn magnitudes and one of per-(token, block)
    /// MLP magnitudes into the struct. No CPU round-trip happens here.
    fn forward_core(
        &self,
        ids: &[u32],
        mut taps: Option<&mut ForwardTaps>,
    ) -> FerrotorchResult<CudaSlice<u16>> {
        if ids.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "forward_core: empty id list".into(),
            });
        }

        let cfg = &self.config;
        let dev = &self.device;
        let seq = ids.len();
        let hidden = cfg.hidden_size;
        let n_heads = cfg.num_attention_heads;
        let n_kv = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        let group = cfg.kv_group_size();
        let ffn = cfg.intermediate_size;
        let scale = 1.0f32 / (head_dim as f32).sqrt();

        if seq > cfg.max_position_embeddings {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "seq_len {seq} exceeds max_position_embeddings {}",
                    cfg.max_position_embeddings
                ),
            });
        }

        let ids_gpu = dev.stream().clone_htod(ids).map_err(map_driver_err)?;

        let mut hidden_buf = gpu_embedding_gather_bf16(&self.embed_tokens, &ids_gpu, hidden, dev)
            .map_err(map_gpu_err)?;

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let h_norm = gpu_rmsnorm_bf16(
                &hidden_buf,
                &layer.input_norm,
                seq,
                hidden,
                cfg.rms_norm_eps as f32,
                dev,
            )
            .map_err(map_gpu_err)?;

            let q = gpu_matmul_bf16_bf16_nt(&h_norm, &layer.q_proj, seq, hidden, hidden, dev)
                .map_err(map_gpu_err)?;
            let k =
                gpu_matmul_bf16_bf16_nt(&h_norm, &layer.k_proj, seq, hidden, n_kv * head_dim, dev)
                    .map_err(map_gpu_err)?;
            let v =
                gpu_matmul_bf16_bf16_nt(&h_norm, &layer.v_proj, seq, hidden, n_kv * head_dim, dev)
                    .map_err(map_gpu_err)?;

            let q_heads = gpu_transpose_to_heads_bf16(&q, n_heads, seq, head_dim, dev)
                .map_err(map_gpu_err)?;
            let k_heads =
                gpu_transpose_to_heads_bf16(&k, n_kv, seq, head_dim, dev).map_err(map_gpu_err)?;
            let v_heads =
                gpu_transpose_to_heads_bf16(&v, n_kv, seq, head_dim, dev).map_err(map_gpu_err)?;

            let q_rot = gpu_rope_half_bf16(
                &q_heads,
                &self.cos_cache,
                &self.sin_cache,
                n_heads,
                seq,
                head_dim,
                0,
                dev,
            )
            .map_err(map_gpu_err)?;
            let k_rot = gpu_rope_half_bf16(
                &k_heads,
                &self.cos_cache,
                &self.sin_cache,
                n_kv,
                seq,
                head_dim,
                0,
                dev,
            )
            .map_err(map_gpu_err)?;

            let k_full =
                gpu_repeat_kv_bf16(&k_rot, n_kv, group, seq, head_dim, dev).map_err(map_gpu_err)?;
            let v_full = gpu_repeat_kv_bf16(&v_heads, n_kv, group, seq, head_dim, dev)
                .map_err(map_gpu_err)?;

            let mut scores = gpu_matmul_bf16_bf16_strided_batched_nt(
                &q_rot,
                &k_full,
                seq,
                head_dim,
                seq,
                n_heads,
                seq * head_dim,
                seq * head_dim,
                scale,
                dev,
            )
            .map_err(map_gpu_err)?;

            gpu_causal_mask_bf16(&mut scores, n_heads, seq, seq, dev).map_err(map_gpu_err)?;

            let attn_weights =
                gpu_softmax_bf16(&scores, n_heads * seq, seq, dev).map_err(map_gpu_err)?;

            let attn_out = gpu_matmul_bf16_bf16_strided_batched(
                &attn_weights,
                &v_full,
                seq,
                seq,
                head_dim,
                n_heads,
                seq * seq,
                seq * head_dim,
                1.0,
                dev,
            )
            .map_err(map_gpu_err)?;

            // Attention tap — per-head L-inf magnitude before o_proj.
            if let Some(t) = taps.as_deref_mut() {
                let a = gpu_block_reduce_max_abs_bf16(&attn_out, n_heads * seq, 1, head_dim, dev)
                    .map_err(map_gpu_err)?;
                t.attn_f32.push(a);
            }

            let attn_flat = gpu_transpose_from_heads_bf16(&attn_out, n_heads, seq, head_dim, dev)
                .map_err(map_gpu_err)?;

            let attn_proj =
                gpu_matmul_bf16_bf16_nt(&attn_flat, &layer.o_proj, seq, hidden, hidden, dev)
                    .map_err(map_gpu_err)?;

            hidden_buf = gpu_add_bf16(&hidden_buf, &attn_proj, dev).map_err(map_gpu_err)?;

            let h_norm2 = gpu_rmsnorm_bf16(
                &hidden_buf,
                &layer.post_attn_norm,
                seq,
                hidden,
                cfg.rms_norm_eps as f32,
                dev,
            )
            .map_err(map_gpu_err)?;

            let gate = gpu_matmul_bf16_bf16_nt(&h_norm2, &layer.gate_proj, seq, hidden, ffn, dev)
                .map_err(map_gpu_err)?;
            let up = gpu_matmul_bf16_bf16_nt(&h_norm2, &layer.up_proj, seq, hidden, ffn, dev)
                .map_err(map_gpu_err)?;

            let activated_gate = match cfg.hidden_act {
                LlamaActivation::Silu => gpu_silu_bf16(&gate, dev).map_err(map_gpu_err)?,
                LlamaActivation::Relu => gpu_relu_bf16(&gate, dev).map_err(map_gpu_err)?,
                LlamaActivation::FatRelu(threshold) => {
                    gpu_fatrelu_bf16(&gate, threshold as f32, dev).map_err(map_gpu_err)?
                }
            };
            let gated = gpu_mul_bf16(&activated_gate, &up, dev).map_err(map_gpu_err)?;

            // MLP tap (default) — per-block L-inf magnitude of the gated
            // activation (what multiplies each row of down_proj). When
            // `plainact_local` is set, we delay the tap until after the
            // residual addition so we can compute the local plainact
            // surrogate `|gated * (down_proj^T @ y_full)|` per neuron.
            if let Some(t) = taps.as_deref_mut() {
                if !t.plainact_local {
                    let m = gpu_block_reduce_max_abs_bf16(
                        &gated,
                        seq,
                        t.n_mlp_blocks,
                        t.mlp_block_size,
                        dev,
                    )
                    .map_err(map_gpu_err)?;
                    t.mlp_f32.push(m);
                }
            }

            let down = gpu_matmul_bf16_bf16_nt(&gated, &layer.down_proj, seq, ffn, hidden, dev)
                .map_err(map_gpu_err)?;

            hidden_buf = gpu_add_bf16(&hidden_buf, &down, dev).map_err(map_gpu_err)?;

            // Plainact-local tap. Computed AFTER the residual add so
            // we have y_full = residual + mlp_out. The signal is
            // `|gated[i] * (y_full @ down_proj)[i]|` per neuron i,
            // reduced to per-block max-abs. This approximates the
            // ShadowLLM paper's plainact criterion (gradient of CE
            // loss × activation) using the gradient of `||y_full||²`
            // wrt each neuron — which is closed-form `2 * y_full @
            // down_proj`. No backward pass needed; ~10ms extra per
            // forward at seq=256 (one extra matmul per layer).
            if let Some(t) = taps.as_deref_mut() {
                if t.plainact_local {
                    use ferrotorch_gpu::gpu_matmul_bf16_bf16;
                    // z = y_full @ down_proj  → [seq, ffn]
                    let z =
                        gpu_matmul_bf16_bf16(&hidden_buf, &layer.down_proj, seq, hidden, ffn, dev)
                            .map_err(map_gpu_err)?;
                    let act_grad = gpu_mul_bf16(&gated, &z, dev).map_err(map_gpu_err)?;
                    let m = gpu_block_reduce_max_abs_bf16(
                        &act_grad,
                        seq,
                        t.n_mlp_blocks,
                        t.mlp_block_size,
                        dev,
                    )
                    .map_err(map_gpu_err)?;
                    t.mlp_f32.push(m);
                }
            }

            // Bootstrap tap — after finishing layer `layer_idx`, if
            // this matches the requested `bootstrap_k` we clone the
            // hidden state on-device so the subsequent layers can
            // keep mutating `hidden_buf` without disturbing the tap.
            if let Some(t) = taps.as_deref_mut() {
                if let Some(k) = t.bootstrap_k {
                    if layer_idx + 1 == k && t.bootstrap_hidden.is_none() {
                        let mut snapshot = dev
                            .stream()
                            .alloc_zeros::<u16>(seq * hidden)
                            .map_err(map_driver_err)?;
                        dev.stream()
                            .memcpy_dtod(&hidden_buf, &mut snapshot)
                            .map_err(map_driver_err)?;
                        t.bootstrap_hidden = Some(snapshot);
                    }
                }
            }
        }

        let final_norm = gpu_rmsnorm_bf16(
            &hidden_buf,
            &self.norm,
            seq,
            hidden,
            cfg.rms_norm_eps as f32,
            dev,
        )
        .map_err(map_gpu_err)?;
        Ok(final_norm)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn upload_layer(
    cfg: &LlamaConfig,
    i: usize,
    state: &mut StateDict<bf16>,
    device: &GpuDevice,
) -> FerrotorchResult<LlamaGpuLayer> {
    let kv_dim = cfg.num_key_value_heads * cfg.head_dim();
    Ok(LlamaGpuLayer {
        input_norm: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.input_layernorm.weight"),
            &[cfg.hidden_size],
            device,
        )?,
        q_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.self_attn.q_proj.weight"),
            &[cfg.hidden_size, cfg.hidden_size],
            device,
        )?,
        k_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.self_attn.k_proj.weight"),
            &[kv_dim, cfg.hidden_size],
            device,
        )?,
        v_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.self_attn.v_proj.weight"),
            &[kv_dim, cfg.hidden_size],
            device,
        )?,
        o_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.self_attn.o_proj.weight"),
            &[cfg.hidden_size, cfg.hidden_size],
            device,
        )?,
        post_attn_norm: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.post_attention_layernorm.weight"),
            &[cfg.hidden_size],
            device,
        )?,
        gate_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.mlp.gate_proj.weight"),
            &[cfg.intermediate_size, cfg.hidden_size],
            device,
        )?,
        up_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.mlp.up_proj.weight"),
            &[cfg.intermediate_size, cfg.hidden_size],
            device,
        )?,
        down_proj: upload_bf16_tensor(
            state,
            &format!("model.layers.{i}.mlp.down_proj.weight"),
            &[cfg.hidden_size, cfg.intermediate_size],
            device,
        )?,
    })
}

/// Pop a tensor out of the `StateDict`, validate its shape, convert
/// its `Vec<bf16>` data to `Vec<u16>` bits, and upload to the device.
/// The tensor is removed from `state` so the host copy can be freed.
fn upload_bf16_tensor(
    state: &mut StateDict<bf16>,
    name: &str,
    expected_shape: &[usize],
    device: &GpuDevice,
) -> FerrotorchResult<CudaSlice<u16>> {
    let tensor = state
        .remove(name)
        .ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("missing tensor in StateDict: \"{name}\""),
        })?;
    if tensor.shape() != expected_shape {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{name}: expected shape {expected_shape:?}, got {:?}",
                tensor.shape()
            ),
        });
    }
    let bits = bf16_tensor_as_bits(&tensor)?;
    device.stream().clone_htod(&bits).map_err(map_driver_err)
}

/// View a `Tensor<bf16>`'s underlying data as `Vec<u16>` bits.
/// Allocates a fresh Vec — correct and simple. Callers immediately
/// upload to GPU and drop this Vec, so the duplication is bounded
/// to one tensor at a time. `bf16` is `repr(transparent)` over `u16`,
/// so the conversion is a single bulk memcpy via `bytemuck` rather
/// than an element-by-element iterator (the iterator form was the
/// dominant bottleneck on large weight loads — Llama-3-70B paged
/// load was spending most of its time here).
fn bf16_tensor_as_bits(t: &Tensor<bf16>) -> FerrotorchResult<Vec<u16>> {
    let data = t.data()?;
    Ok(bytemuck::cast_slice::<bf16, u16>(data).to_vec())
}

/// Precompute cos/sin caches for Llama's half-rotation RoPE, upload
/// as bf16. Each cache is `[max_seq, head_dim / 2]` — the caller
/// slices into `[pos * half_dim + d]` per the kernel's indexing.
fn build_rope_caches(
    cfg: &LlamaConfig,
    device: &GpuDevice,
) -> FerrotorchResult<(CudaSlice<u16>, CudaSlice<u16>)> {
    let head_dim = cfg.head_dim();
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "RoPE requires even head_dim".into(),
        });
    }
    let half = head_dim / 2;
    let max_seq = cfg.max_position_embeddings;
    let base = cfg.rope_theta;

    let inv_freq: Vec<f64> = (0..half)
        .map(|i| 1.0 / base.powf(2.0 * i as f64 / head_dim as f64))
        .collect();

    let mut cos_bits = Vec::with_capacity(max_seq * half);
    let mut sin_bits = Vec::with_capacity(max_seq * half);
    for pos in 0..max_seq {
        for d in 0..half {
            let angle = pos as f64 * inv_freq[d];
            cos_bits.push(bf16::from_f32(angle.cos() as f32).to_bits());
            sin_bits.push(bf16::from_f32(angle.sin() as f32).to_bits());
        }
    }

    let cos_cache = device
        .stream()
        .clone_htod(&cos_bits)
        .map_err(map_driver_err)?;
    let sin_cache = device
        .stream()
        .clone_htod(&sin_bits)
        .map_err(map_driver_err)?;
    Ok((cos_cache, sin_cache))
}

/// Categorically map a [`ferrotorch_gpu::GpuError`] into the existing
/// [`FerrotorchError`] taxonomy.
///
/// This is a best-effort categorical mapping into existing variants —
/// the audit's full fix (a new `FerrotorchError::Gpu(source)` variant
/// that owns the source-chained error) is a workspace-coordination
/// event and is tracked separately (see #699). Until that lands, every
/// arm of this match preserves the original `GpuError` Debug output in
/// its `message` so the underlying cause is visible to anyone reading
/// the error string. Each arm picks the closest semantic variant; we
/// avoid collapsing all GPU errors into `InvalidArgument` (which the
/// previous implementation did) because OOM and PTX-compile failures
/// are not a *parameter* problem and were misleading callers' bug
/// triage.
fn map_gpu_err(e: ferrotorch_gpu::GpuError) -> FerrotorchError {
    use ferrotorch_gpu::GpuError as G;
    match e {
        // Shape and length problems are genuine input shape mismatches.
        G::ShapeMismatch { op, expected, got } => FerrotorchError::ShapeMismatch {
            message: format!("{op}: expected {expected:?}, got {got:?}"),
        },
        G::LengthMismatch { a, b } => FerrotorchError::ShapeMismatch {
            message: format!("buffer length mismatch: {a} vs {b}"),
        },
        // Device-selection problems map to the existing device taxonomy.
        // `InvalidDevice` and `DeviceMismatch` carry usize ordinals; we
        // surface the structured variant via Debug so the message is rich.
        G::InvalidDevice { .. } | G::DeviceMismatch { .. } => FerrotorchError::Internal {
            message: format!("gpu device error: {e:?}"),
        },
        // Unsupported (op, dtype) is the closest match to the existing
        // NotImplementedOnCuda variant; preserve the structured op.
        G::Unsupported { op, dtype } => {
            // op is &'static str — we can carry it through; dtype goes
            // into the message so the diagnostic isn't lost.
            let _ = dtype; // dtype kept in Debug fallback below if needed
            FerrotorchError::NotImplementedOnCuda { op }
        }
        // Memory exhaustion is a runtime/resource condition, not a
        // parameter problem.
        G::OutOfMemory { .. } | G::BudgetExceeded { .. } => FerrotorchError::Internal {
            message: format!("gpu memory error: {e:?}"),
        },
        // PTX compile failures are environment/toolchain issues.
        G::PtxCompileFailed { .. } => FerrotorchError::Internal {
            message: format!("gpu kernel compile error: {e:?}"),
        },
        // Driver / cuBLAS / cuSOLVER / cuFFT errors are runtime CUDA
        // failures; preserve via Debug formatting.
        G::Driver(_) | G::Blas(_) | G::Solver(_) | G::Fft(_) => FerrotorchError::Internal {
            message: format!("cuda runtime error: {e:?}"),
        },
        // Invalid-state is a logic/runtime invariant, not a parameter
        // problem; route through Internal with the message preserved.
        G::InvalidState { ref message } => FerrotorchError::Internal {
            message: format!("gpu invalid state: {message}"),
        },
        // GpuError is `#[non_exhaustive]`; future variants fall through
        // to Internal with full Debug output. Once the source-chain
        // variant lands (#699) this becomes lossless.
        other => FerrotorchError::Internal {
            message: format!("gpu error: {other:?}"),
        },
    }
}

/// Map a [`cudarc::driver::DriverError`] into [`FerrotorchError`].
///
/// Every CUDA driver error is a runtime/resource problem (allocation
/// failure, kernel launch failure, context lost, etc.) — never a
/// parameter problem. Routes through [`FerrotorchError::Internal`] with
/// the Debug formatting of the upstream error so the exact CUresult
/// code and any cudarc context is preserved in the message.
fn map_driver_err(e: cudarc::driver::DriverError) -> FerrotorchError {
    FerrotorchError::Internal {
        message: format!("cuda driver error: {e:?}"),
    }
}
