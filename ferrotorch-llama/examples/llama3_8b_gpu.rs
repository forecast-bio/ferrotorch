//! Meta-Llama-3-8B GPU end-to-end smoke.
//!
//! Runs the full decoder forward pass on an RTX 3090 (or any CUDA
//! device) with weights stored in VRAM as bf16.  Requires
//! `--features cuda` on `ferrotorch-llama`:
//!
//! ```sh
//! LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 \
//!   cargo run --release --features cuda -p ferrotorch-llama \
//!     --example llama3_8b_gpu -- "The meaning of life is"
//! ```
//!
//! Pipeline (all on-device past step 4):
//! 1. Parse `config.json` → `LlamaConfig`.
//! 2. Load `tokenizer.json`.
//! 3. Load all 4 safetensors shards into a `StateDict<bf16>` on CPU.
//! 4. `LlamaGpuInferencer::new` uploads every tensor to GPU as
//!    `CudaSlice<u16>` and precomputes the RoPE cos/sin caches.
//! 5. `forward_from_ids` runs the 32-layer decoder directly against
//!    the hand-written bf16 CUDA kernels.  Last-token logits come
//!    back as `Vec<f32>`; we argmax + decode for the continuation.

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

    let snapshot = hf_snapshot("meta-llama/Meta-Llama-3-8B");
    println!("[llama3_8b_gpu] snapshot = {}", snapshot.display());

    // Step 1: config
    let cfg_path = snapshot.join("config.json");
    let hf_cfg = HfTransformerConfig::from_file(&cfg_path).expect("failed to parse config.json");
    let cfg = LlamaConfig::from_hf(&hf_cfg).expect("invalid LlamaConfig");
    println!(
        "[llama3_8b_gpu] config: hidden={} layers={} heads={} kv={} head_dim={} vocab={} rope_theta={}",
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
    println!("[llama3_8b_gpu] loading CPU bf16 weights...");
    let t_load = Instant::now();
    let state = load_safetensors_sharded::<half::bf16>(&idx_path)
        .expect("failed to load sharded safetensors");
    println!(
        "[llama3_8b_gpu] loaded {} tensors in {:.1}s",
        state.len(),
        t_load.elapsed().as_secs_f64()
    );

    // Step 4: upload to GPU
    println!("[llama3_8b_gpu] initialising CUDA device...");
    let device = GpuDevice::new(0).expect("CUDA device 0 unavailable");
    println!("[llama3_8b_gpu] uploading weights to VRAM...");
    let t_up = Instant::now();
    let inferencer = LlamaGpuInferencer::new(cfg.clone(), state, device)
        .expect("failed to build LlamaGpuInferencer");
    println!(
        "[llama3_8b_gpu] weights uploaded in {:.1}s",
        t_up.elapsed().as_secs_f64()
    );

    // Step 5: encode prompt
    let prompt_ids =
        encode(&tok, &prompt, /* add_special_tokens = */ true).expect("failed to encode prompt");
    println!(
        "[llama3_8b_gpu] prompt tokens ({}): {:?}",
        prompt_ids.len(),
        prompt_ids
    );

    // Step 6: greedy-decode loop
    let mut tokens: Vec<u32> = prompt_ids.clone();
    let mut generation_secs = 0.0f64;
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
            "[llama3_8b_gpu] step {}/{}: id={} text={:?} ({:.2}s, ctx={})",
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
        "[llama3_8b_gpu] generated {} tokens in {:.2}s (avg {:.2}s/token, {:.3} tok/s)",
        max_new_tokens,
        generation_secs,
        avg,
        1.0 / avg,
    );
    println!("[llama3_8b_gpu] full continuation: {}", full);
}
