//! ProSparse Llama 2 7B — FATReLU sparsity verification on GPU.
//!
//! Loads the SparseLLM/prosparse-llama-2-7b checkpoint, runs a forward
//! pass with per-neuron activation taps, and reports the percentage of
//! zero activations per layer. Expected: ~89% MLP sparsity.
//!
//! ```sh
//! LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64 \
//!   cargo run --release --features cuda -p ferrotorch-llama \
//!     --example prosparse_7b_gpu
//! ```

use std::path::PathBuf;
use std::time::Instant;

use ferrotorch_gpu::GpuDevice;
use ferrotorch_hub::HfTransformerConfig;
use ferrotorch_llama::{LlamaConfig, LlamaGpuInferencer};
use ferrotorch_serialize::safetensors_io::load_safetensors_sharded;

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
    let snapshot = hf_snapshot("SparseLLM/prosparse-llama-2-7b");
    println!("[prosparse] snapshot = {}", snapshot.display());

    // Config
    let cfg_path = snapshot.join("config.json");
    let hf_cfg = HfTransformerConfig::from_file(&cfg_path).expect("failed to parse config.json");
    let cfg = LlamaConfig::from_hf(&hf_cfg).expect("invalid LlamaConfig");
    println!(
        "[prosparse] config: hidden={} intermediate={} layers={} heads={} act={:?}",
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.hidden_act,
    );

    // Load weights (F32 on disk, auto-downcast to bf16)
    let idx_path = snapshot.join("model.safetensors.index.json");
    println!("[prosparse] loading CPU weights (F32 -> bf16 downcast)...");
    let t_load = Instant::now();
    let state = load_safetensors_sharded::<half::bf16>(&idx_path)
        .expect("failed to load sharded safetensors");
    println!(
        "[prosparse] loaded {} tensors in {:.1}s",
        state.len(),
        t_load.elapsed().as_secs_f64()
    );

    // Upload to GPU
    println!("[prosparse] initialising CUDA device...");
    let device = GpuDevice::new(0).expect("CUDA device 0 unavailable");
    println!("[prosparse] uploading weights to VRAM...");
    let t_up = Instant::now();
    let inferencer =
        LlamaGpuInferencer::new(cfg, state, device).expect("failed to build LlamaGpuInferencer");
    println!(
        "[prosparse] weights uploaded in {:.1}s",
        t_up.elapsed().as_secs_f64()
    );

    // Profiled forward with per-neuron taps to measure FATReLU sparsity.
    // BOS=1, then a handful of arbitrary valid token IDs.
    let ids: Vec<u32> = vec![1, 450, 6593, 310, 2834, 338];
    let seq = ids.len();
    let ffn = cfg.intermediate_size;
    let n_layers = cfg.num_hidden_layers;

    println!(
        "[prosparse] running profiled forward (seq={}, per-neuron taps)...",
        seq
    );
    let t_fwd = Instant::now();
    let result = inferencer
        .forward_from_ids_profiled_with_bootstrap(&ids, 1, ffn, None, false)
        .expect("profiled forward failed");
    println!(
        "[prosparse] forward done in {:.2}s",
        t_fwd.elapsed().as_secs_f64()
    );

    // Measure sparsity: count zero magnitudes per layer.
    assert_eq!(
        result.mlp_magnitudes.len(),
        n_layers * seq * ffn,
        "unexpected mlp_magnitudes length"
    );

    let mut total_zeros = 0usize;
    let mut total_neurons = 0usize;
    println!("\n  Layer  |  Zero neurons  |  Sparsity");
    println!("  -------|----------------|----------");
    for layer in 0..n_layers {
        let offset = layer * seq * ffn;
        let layer_slice = &result.mlp_magnitudes[offset..offset + seq * ffn];
        let zeros = layer_slice.iter().filter(|&&v| v == 0.0).count();
        let total = layer_slice.len();
        let sparsity = zeros as f64 / total as f64 * 100.0;
        println!(
            "  {:>5}  |  {:>7}/{:<7}|  {:.1}%",
            layer, zeros, total, sparsity
        );
        total_zeros += zeros;
        total_neurons += total;
    }
    let overall = total_zeros as f64 / total_neurons as f64 * 100.0;
    println!("  -------|----------------|----------");
    println!(
        "  total  |  {:>7}/{:<7}|  {:.1}%",
        total_zeros, total_neurons, overall
    );
    println!(
        "\n[prosparse] overall MLP sparsity: {:.1}% (target: ~89%)",
        overall
    );

    // Quick greedy decode to verify forward pass produces valid logits.
    println!("\n[prosparse] greedy decode sanity check...");
    let mut tokens = ids.clone();
    for step in 0..5 {
        let logits = inferencer
            .forward_from_ids(&tokens)
            .expect("forward failed");
        let best_id = logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i as u32)
            .unwrap();
        println!(
            "[prosparse] step {}: argmax token_id={} logit={:.2}",
            step + 1,
            best_id,
            logits[best_id as usize]
        );
        tokens.push(best_id);
    }
    println!("[prosparse] final token ids: {:?}", tokens);
}
