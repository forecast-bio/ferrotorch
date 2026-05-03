//! GPU smoke test for [`LlamaGpuInferencer`].
//!
//! The audit (line 1192 of `audits/2026-05-02-rust-quality-and-gpu-discipline.md`)
//! called out that the GPU forward path was tested only via `examples/`,
//! never via `cargo test`. Closing finding #10 means a test that actually
//! constructs a tiny synthetic model on the device, runs
//! `forward_from_ids`, and checks the output has teeth.
//!
//! Test placement: integration test (`tests/gpu_smoke.rs`). Conventional
//! Rust idiom for end-to-end smoke tests, exercises only the public API
//! (`LlamaConfig`, `LlamaGpuInferencer`), and runs without compromising
//! `--lib` test coverage.
//!
//! Run with `cargo test -p ferrotorch-llama --features cuda`.

#![cfg(feature = "cuda")]

use std::collections::HashMap;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_gpu::GpuDevice;
use ferrotorch_llama::{LlamaConfig, LlamaGpuInferencer};
use half::bf16;

/// Build a `Tensor<bf16>` of the given shape, populated with a small
/// deterministic non-zero pattern so RMSNorm doesn't divide by zero
/// and matmuls don't trivially produce zero outputs. The pattern
/// `0.01 * (i + 1)` keeps every element finite and well under bf16's
/// representable range.
fn fill_tensor(shape: Vec<usize>) -> Tensor<bf16> {
    let numel: usize = shape.iter().product();
    let data: Vec<bf16> = (0..numel)
        .map(|i| bf16::from_f32(0.01 * ((i % 13) as f32 + 1.0)))
        .collect();
    Tensor::from_storage(TensorStorage::cpu(data), shape, false)
        .expect("synthetic tensor construction must succeed for valid shapes")
}

#[test]
fn gpu_forward_from_ids_smoke() {
    // Tiny config — exercises every kernel in the forward stack
    // (embed, RMSNorm, q/k/v/o matmul, RoPE, repeat_kv, attention
    // strided-batched matmul, causal mask, softmax, MLP gate/up/down,
    // final RMSNorm, lm_head matmul) without spending real VRAM.
    //
    //   vocab_size:         32  (small, but big enough to exercise lm_head shape)
    //   hidden_size:        16
    //   n_layers:            1
    //   n_attention_heads:   2  → head_dim = 8 (even, RoPE-compatible)
    //   n_kv_heads:          2  (MHA, kv_group_size = 1)
    //   intermediate_size:  32
    //   max_position_embeddings: 16
    //   rope_theta:        10000.0
    let cfg = LlamaConfig {
        vocab_size: 32,
        hidden_size: 16,
        intermediate_size: 32,
        num_hidden_layers: 1,
        num_attention_heads: 2,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 16,
        tie_word_embeddings: false,
        hidden_act: ferrotorch_llama::LlamaActivation::Silu,
    };
    cfg.validate().expect("smoke cfg must validate");
    let head_dim = cfg.head_dim();
    let kv_dim = cfg.num_key_value_heads * head_dim;

    // Build a synthetic state dict matching the HF naming convention
    // that `LlamaGpuInferencer::new` consumes via `upload_layer` /
    // `upload_bf16_tensor`.
    let mut state: HashMap<String, Tensor<bf16>> = HashMap::new();
    state.insert(
        "model.embed_tokens.weight".to_string(),
        fill_tensor(vec![cfg.vocab_size, cfg.hidden_size]),
    );
    state.insert(
        "model.norm.weight".to_string(),
        fill_tensor(vec![cfg.hidden_size]),
    );
    state.insert(
        "lm_head.weight".to_string(),
        fill_tensor(vec![cfg.vocab_size, cfg.hidden_size]),
    );
    for i in 0..cfg.num_hidden_layers {
        state.insert(
            format!("model.layers.{i}.input_layernorm.weight"),
            fill_tensor(vec![cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.post_attention_layernorm.weight"),
            fill_tensor(vec![cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.self_attn.q_proj.weight"),
            fill_tensor(vec![cfg.hidden_size, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.self_attn.k_proj.weight"),
            fill_tensor(vec![kv_dim, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.self_attn.v_proj.weight"),
            fill_tensor(vec![kv_dim, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.self_attn.o_proj.weight"),
            fill_tensor(vec![cfg.hidden_size, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.mlp.gate_proj.weight"),
            fill_tensor(vec![cfg.intermediate_size, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.mlp.up_proj.weight"),
            fill_tensor(vec![cfg.intermediate_size, cfg.hidden_size]),
        );
        state.insert(
            format!("model.layers.{i}.mlp.down_proj.weight"),
            fill_tensor(vec![cfg.hidden_size, cfg.intermediate_size]),
        );
    }

    // Construct the device. Using GpuDevice::new(0) — the canonical
    // construction shape used by every example in this crate
    // (see examples/llama3_8b_gpu.rs:96).
    let device = GpuDevice::new(0).expect("CUDA device 0 must be available for the GPU smoke test");

    // Upload the synthetic weights and build the inferencer.
    let inferencer = LlamaGpuInferencer::new(cfg, state, device)
        .expect("LlamaGpuInferencer::new must succeed with a synthetic-shaped StateDict");

    // Run a 3-token forward pass. This exercises the full kernel
    // pipeline (embedding gather → 1 decoder layer with attention &
    // SwiGLU MLP → final norm → lm_head) and downloads the
    // last-token logits as Vec<f32>.
    let result: Vec<f32> = inferencer
        .forward_from_ids(&[1u32, 2u32, 3u32])
        .expect("forward_from_ids must succeed against a valid synthetic config");

    // ---- Assertions with teeth ----

    // (1) Shape: result is exactly vocab_size logits for the last token.
    assert_eq!(
        result.len(),
        32,
        "expected vocab_size (32) logits, got {}",
        result.len()
    );

    // (2) Finiteness: every logit is a finite f32 (catches NaN/Inf
    //     fallout from broken kernels — softmax overflow, RMSNorm
    //     div-by-zero, RoPE phase out of range, etc.).
    for (i, &v) in result.iter().enumerate() {
        assert!(
            v.is_finite(),
            "non-finite logit at index {i}: {v} (full result: {result:?})",
        );
    }

    // (3) Non-trivial output: the kernel actually computed something
    //     instead of returning zeros (catches the case where every
    //     kernel was no-op'd or the buffer was never written). We
    //     check both "not all zeros" and "has variance" — the latter
    //     guards against the kernel writing a constant value.
    let all_zero = result.iter().all(|&v| v == 0.0);
    assert!(
        !all_zero,
        "result is all zeros — GPU kernels likely produced no output: {result:?}",
    );
    let mean = result.iter().sum::<f32>() / result.len() as f32;
    let var: f32 = result.iter().map(|&v| (v - mean).powi(2)).sum::<f32>() / result.len() as f32;
    assert!(
        var > 0.0,
        "result has zero variance (constant value) — GPU kernels likely degenerate: \
         mean={mean}, var={var}, result={result:?}",
    );
}
