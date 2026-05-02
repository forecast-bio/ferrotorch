//! Meta-Llama-3.3-70B-Instruct GPU end-to-end smoke (bf16).
//!
//! Same shape as `llama3_8b_gpu` but pointed at the 70B-Instruct snapshot.
//! Requires a multi-GPU box: 70B in bf16 occupies ~140 GB before activations
//! and KV cache, so an 8 × H100 (80 GB each) or 8 × L40S (48 GB each) node is
//! the realistic minimum target.
//!
//! ```sh
//! LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 \
//!   cargo run --release --features cuda -p ferrotorch-llama \
//!     --example llama3_70b_gpu -- "The meaning of life is"
//! ```
//!
//! Pipeline (identical to the 8B variant — just bigger):
//! 1. Parse `config.json` → `LlamaConfig` (or use [`LlamaConfig::llama3_3_70b_instruct`]).
//! 2. Load `tokenizer.json`.
//! 3. Load all safetensors shards into a `StateDict<bf16>` on CPU.
//!    The 70B checkpoint ships as ~30 shards totalling ~140 GB.
//! 4. `LlamaGpuInferencer::new` uploads tensors to GPU as `CudaSlice<u16>`
//!    and precomputes the RoPE cos/sin caches.
//! 5. `forward_from_ids` runs the 80-layer decoder against the bf16 CUDA
//!    kernels. Last-token logits come back as `Vec<f32>`; we argmax + decode
//!    for the continuation.
//!
//! For a single-host fit (i.e. one 80 GB GPU), the bf16 path here will not
//! work — 70B at bf16 is ~140 GB. The single-host fit requires GPU-side
//! dequantization of GGUF-quantized weights via `ferrotorch_cubecl::quant` and
//! the streaming loader in `ferrotorch_llama::gguf_streaming` (see those
//! modules for the path that keeps the full f32 state dict off the host).

use std::path::PathBuf;
use std::time::Instant;

use ferrotorch_gpu::GpuDevice;
use ferrotorch_hub::HfTransformerConfig;
use ferrotorch_llama::{LlamaConfig, LlamaGpuInferencer};
use ferrotorch_serialize::safetensors_io::load_safetensors_sharded;
use ferrotorch_tokenize::{decode, encode, load_tokenizer};

fn home() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .expect("$HOME not set")
}

fn hf_snapshot(repo_slug: &str) -> PathBuf {
    let base = home()
        .join(".cache/huggingface/hub")
        .join(format!("models--{}", repo_slug.replace('/', "--")))
        .join("snapshots");
    std::fs::read_dir(&base)
        .unwrap_or_else(|e| panic!("HF cache {} missing: {e}", base.display()))
        .next()
        .unwrap_or_else(|| panic!("no snapshot in {}", base.display()))
        .unwrap()
        .path()
}

fn main() {
    let prompt = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "The meaning of life is".to_string());
    let max_new_tokens: usize = std::env::var("LLAMA_MAX_NEW_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let snapshot = hf_snapshot("meta-llama/Llama-3.3-70B-Instruct");
    println!("[llama3_70b_gpu] snapshot = {}", snapshot.display());

    // Step 1: config — prefer the on-disk config.json, fall back to the
    // canonical constructor if the snapshot is incomplete.
    let cfg_path = snapshot.join("config.json");
    let cfg = if cfg_path.exists() {
        let hf_cfg =
            HfTransformerConfig::from_file(&cfg_path).expect("failed to parse config.json");
        LlamaConfig::from_hf(&hf_cfg).expect("invalid LlamaConfig")
    } else {
        LlamaConfig::llama3_3_70b_instruct()
    };
    println!(
        "[llama3_70b_gpu] config: hidden={} layers={} heads={} kv={} head_dim={} vocab={} rope_theta={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.head_dim(),
        cfg.vocab_size,
        cfg.rope_theta,
    );

    // Step 2: tokenizer
    let tok_path = snapshot.join("tokenizer.json");
    let tok = load_tokenizer(&tok_path).expect("failed to load tokenizer.json");

    // Step 3: load weights
    let idx_path = snapshot.join("model.safetensors.index.json");
    println!(
        "[llama3_70b_gpu] loading CPU bf16 weights from {} shards...",
        "many"
    );
    let t_load = Instant::now();
    let state = load_safetensors_sharded::<half::bf16>(&idx_path)
        .expect("failed to load sharded safetensors");
    println!(
        "[llama3_70b_gpu] loaded {} tensors in {:.1}s",
        state.len(),
        t_load.elapsed().as_secs_f64()
    );

    // Step 4: upload to GPU
    println!("[llama3_70b_gpu] initialising CUDA device...");
    let device = GpuDevice::new(0).expect("CUDA device 0 unavailable");
    println!("[llama3_70b_gpu] uploading weights to VRAM (this is ~140 GB; may take minutes)...");
    let t_up = Instant::now();
    let inferencer = LlamaGpuInferencer::new(cfg.clone(), state, device)
        .expect("failed to build LlamaGpuInferencer");
    println!(
        "[llama3_70b_gpu] weights uploaded in {:.1}s",
        t_up.elapsed().as_secs_f64()
    );

    // Step 5: encode prompt
    let prompt_ids =
        encode(&tok, &prompt, /* add_special_tokens = */ true).expect("failed to encode prompt");
    println!(
        "[llama3_70b_gpu] prompt tokens ({}): {:?}",
        prompt_ids.len(),
        prompt_ids
    );

    // Step 6: greedy-decode loop
    let mut tokens: Vec<u32> = prompt_ids.clone();
    let mut generation_secs = 0.0_f64;
    for step in 0..max_new_tokens {
        let t_step = Instant::now();
        let logits = inferencer
            .forward_from_ids(&tokens)
            .expect("forward_from_ids failed");
        let step_s = t_step.elapsed().as_secs_f64();
        generation_secs += step_s;

        assert_eq!(logits.len(), cfg.vocab_size);
        let mut best_id = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_id = i as u32;
            }
        }
        let piece = decode(&tok, &[best_id], /* skip_special_tokens = */ false)
            .expect("failed to decode token");
        println!(
            "[llama3_70b_gpu] step {}/{}: id={} text={:?} ({:.2}s, ctx={})",
            step + 1,
            max_new_tokens,
            best_id,
            piece,
            step_s,
            tokens.len()
        );
        tokens.push(best_id);
    }

    let full = decode(&tok, &tokens, /* skip_special_tokens = */ true)
        .expect("failed to decode full sequence");
    let avg = generation_secs / max_new_tokens as f64;
    println!(
        "[llama3_70b_gpu] generated {} tokens in {:.2}s (avg {:.2}s/token, {:.3} tok/s)",
        max_new_tokens,
        generation_secs,
        avg,
        1.0 / avg,
    );
    println!("[llama3_70b_gpu] full continuation: {full}");
}
