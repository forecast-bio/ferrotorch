//! Causal-LM inference-dump binary for crosslink #1147 / #1154 verification.
//!
//! Companion to `scripts/verify_causal_lm_inference.py` (CPU f32 parity vs
//! transformers) and `scripts/verify_causal_lm_gpu_inference.py` (GPU bf16
//! parity vs the same CPU f32 reference). Loads one of the pinned
//! causal-LM mirrors from `ferrotorch/<name>` on the HuggingFace Hub,
//! runs a single prefill pass on a fixed prompt, and dumps the resulting
//! logits to disk in the same `[u32 ndim][u32 × ndim shape][f32 data]`
//! little-endian format the vision-side `inference_dump.rs` example uses.
//!
//! Usage (network required to first-touch the HF mirror; subsequent runs
//! hit the local hub cache):
//! ```text
//! # CPU f32 path
//! cargo run -p ferrotorch-llama --release --example llm_inference_dump -- \
//!     --model smollm-135m \
//!     --output /tmp/ferrotorch_llm_dump.bin
//!
//! # GPU bf16 path (requires `--features cuda`)
//! cargo run -p ferrotorch-llama --release --features cuda \
//!     --example llm_inference_dump -- \
//!     --model smollm-135m \
//!     --device gpu \
//!     --output /tmp/ferrotorch_llm_dump_gpu.bin
//! ```
//!
//! The example deliberately performs the full
//! `ferrotorch_hub::hf_download_model("ferrotorch/<name>", ...)` →
//! `ferrotorch_hub::load_pretrained(<name>)` →
//! `ferrotorch_llama::LlamaForCausalLM::load_hf_state_dict(...)` (CPU) or
//! `ferrotorch_llama::LlamaGpuInferencer::new(...)` (GPU) →
//! `forward_from_ids` pipeline so the harness exercises every public
//! contract the registry promises.
//!
//! Output:
//!   * `--output <path>`: logits tensor `[1, seq_len, vocab_size]` in
//!     the dump format above. GPU path: bf16 logits round-tripped to f32
//!     on the host, full `[seq, vocab]` tensor (not just the last row) so
//!     the harness can score top-1 argmax agreement at every position.
//!   * stdout: one line of JSON
//!     `{"shape":[1,S,V],"argmax_last":<token_id>,"prompt":...,"token_ids":[...]}`
//!     so the Python harness can parse the prefill verdict without
//!     re-reading the bin.
//!
//! Honest scope notes:
//!   * CPU path: f32 only. `LlamaForCausalLM::<f32>` because the parity
//!     probe was generated with `torch_dtype=float32`; matching dtype is
//!     the whole point of the CPU vs. transformers comparison.
//!   * GPU path: bf16 only. `LlamaGpuInferencer` stores weights as
//!     `CudaSlice<u16>` (bf16 bits) and runs every kernel in bf16 — that
//!     IS the path under test. The dump is the f32 round-trip of the
//!     bf16 logits, compared by the harness to the CPU f32 reference.
//!   * Single prefill, no KV cache. Matches transformers's
//!     `model(input_ids=ids, use_cache=False)` semantics.
//!   * Reads the prompt from the local-cache `_value_parity_input.txt`
//!     (or from `--prompt`) — falling back to the harness's frozen prompt
//!     so the Rust and Python sides always agree on the input.

use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use ferrotorch_core::FerrotorchResult;
use ferrotorch_hub::{HfTransformerConfig, HubCache, hf_download_model};
use ferrotorch_llama::{LlamaConfig, LlamaForCausalLM};
use ferrotorch_serialize::load_safetensors;
use ferrotorch_tokenize::{encode, load_tokenizer};

/// Which forward path the example exercises.
///
/// The CPU variant is the historical f32 reference path
/// (LlamaForCausalLM::f32 + load_safetensors::f32). The GPU variant
/// selects the bf16 path (LlamaGpuInferencer + load_safetensors::bf16);
/// it is only reachable when the binary is compiled with
/// `--features cuda` and produces a parse error at argument-time
/// otherwise so the user sees the build-flag mistake clearly instead
/// of a silent CPU fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    Gpu,
}

#[derive(Debug)]
struct Args {
    model: String,
    output: PathBuf,
    prompt: Option<String>,
    device: Device,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut prompt: Option<String> = None;
    let mut device: Device = Device::Cpu;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(
                    argv.get(i + 1)
                        .ok_or("--model needs a value")?
                        .clone(),
                );
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output needs a value")?,
                ));
                i += 2;
            }
            "--prompt" => {
                prompt = Some(
                    argv.get(i + 1)
                        .ok_or("--prompt needs a value")?
                        .clone(),
                );
                i += 2;
            }
            "--device" => {
                let v = argv.get(i + 1).ok_or("--device needs a value")?;
                device = match v.as_str() {
                    "cpu" => Device::Cpu,
                    "gpu" => {
                        // Categorical refusal: if the binary was built
                        // without the cuda feature, asking for --device
                        // gpu is a build-flag mistake, not a runtime
                        // configuration the example can satisfy.
                        if !cfg!(feature = "cuda") {
                            return Err(
                                "--device gpu requires building with `--features cuda`"
                                    .to_string(),
                            );
                        }
                        Device::Gpu
                    }
                    other => return Err(format!(
                        "--device must be 'cpu' or 'gpu', got {other:?}"
                    )),
                };
                i += 2;
            }
            other => {
                return Err(format!("unknown argument {other:?}"));
            }
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required (e.g. --model smollm-135m)")?,
        output: output.ok_or("--output is required (path to logits .bin)")?,
        prompt,
        device,
    })
}

fn write_dump_f32(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let expected: usize = shape.iter().product();
    assert_eq!(
        data.len(),
        expected,
        "data length {} disagrees with shape product {}",
        data.len(),
        expected
    );
    let mut f = File::create(path)?;
    f.write_all(&(shape.len() as u32).to_le_bytes())?;
    for &d in shape {
        f.write_all(&(d as u32).to_le_bytes())?;
    }
    let mut buf = Vec::with_capacity(data.len() * 4);
    for &v in data {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    f.write_all(&buf)
}

fn read_token_ids_json(path: &Path) -> std::io::Result<Vec<u32>> {
    let bytes = std::fs::read(path)?;
    let s = std::str::from_utf8(&bytes)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?
        .trim();
    // Tiny inline parser: expect "[<u32>, <u32>, ...]"
    let inner = s
        .strip_prefix('[')
        .and_then(|s| s.strip_suffix(']'))
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("token_ids file is not a JSON array: {s:?}"),
            )
        })?;
    let mut ids = Vec::new();
    for chunk in inner.split(',') {
        let t = chunk.trim();
        if t.is_empty() {
            continue;
        }
        let v: u32 = t.parse().map_err(|e| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("token_ids parse error on {t:?}: {e}"),
            )
        })?;
        ids.push(v);
    }
    Ok(ids)
}

/// CPU f32 forward — historical path, unchanged behaviour. Loads
/// `model.safetensors` as `Tensor<f32>`, constructs `LlamaForCausalLM`,
/// runs prefill, and returns `(data, shape)` where `shape = [1, S, V]`.
fn run_cpu_forward(
    repo_dir: &Path,
    cfg: LlamaConfig,
    ids: &[u32],
) -> FerrotorchResult<(Vec<f32>, Vec<usize>)> {
    let weights_path = repo_dir.join("model.safetensors");
    let state = load_safetensors::<f32>(&weights_path)?;
    eprintln!(
        "[llm_inference_dump] loaded state dict (f32): {} tensors",
        state.len()
    );
    let mut model = LlamaForCausalLM::<f32>::new(cfg)?;
    model.load_hf_state_dict(&state, /* strict = */ true)?;
    // Free the staging copy; the model owns its own parameter tensors now.
    drop(state);

    let logits = model.forward_from_ids(ids)?;
    let shape = logits.shape().to_vec();
    let data = logits.data()?.to_vec();
    Ok((data, shape))
}

/// GPU bf16 forward — guarded by `feature = "cuda"`. Loads
/// `model.safetensors` as `Tensor<bf16>`, uploads to VRAM via
/// `LlamaGpuInferencer::new`, runs prefill, and downloads the full
/// `[seq, vocab]` logits tensor as `Vec<f32>` (round-trip from bf16
/// bits). The synthetic batch axis is re-introduced as a leading `1` in
/// the shape so the harness sees a uniform `[1, S, V]` tensor on either
/// path.
#[cfg(feature = "cuda")]
fn run_gpu_forward(
    repo_dir: &Path,
    cfg: LlamaConfig,
    ids: &[u32],
) -> FerrotorchResult<(Vec<f32>, Vec<usize>)> {
    use ferrotorch_gpu::GpuDevice;
    use ferrotorch_llama::LlamaGpuInferencer;
    use half::bf16;

    let weights_path = repo_dir.join("model.safetensors");
    let state = load_safetensors::<bf16>(&weights_path)?;
    eprintln!(
        "[llm_inference_dump] loaded state dict (bf16): {} tensors",
        state.len()
    );

    let device =
        GpuDevice::new(0).map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("CUDA device 0 unavailable: {e}"),
        })?;
    let vocab = cfg.vocab_size;
    let inferencer = LlamaGpuInferencer::new(cfg, state, device)?;
    eprintln!("[llm_inference_dump] LlamaGpuInferencer uploaded; running forward...");

    let flat = inferencer.forward_logits_from_ids_all(ids)?;
    let seq = ids.len();
    if flat.len() != seq * vocab {
        return Err(ferrotorch_core::FerrotorchError::ShapeMismatch {
            message: format!(
                "GPU forward returned {} logits, expected {} (= seq {} * vocab {})",
                flat.len(),
                seq * vocab,
                seq,
                vocab,
            ),
        });
    }
    Ok((flat, vec![1, seq, vocab]))
}

/// Non-cuda stub. Reachable only if `parse_args` is bypassed — the
/// `--device gpu` flag is rejected at argument parsing on a build
/// without `feature = "cuda"`. Surfaces an explicit error rather than
/// any "succeed on CPU under the hood" silent demotion.
#[cfg(not(feature = "cuda"))]
fn run_gpu_forward(
    _repo_dir: &Path,
    _cfg: LlamaConfig,
    _ids: &[u32],
) -> FerrotorchResult<(Vec<f32>, Vec<usize>)> {
    Err(ferrotorch_core::FerrotorchError::InvalidArgument {
        message: "GPU path unreachable: this binary was built without `--features cuda`".into(),
    })
}

fn run() -> FerrotorchResult<()> {
    let args = parse_args().map_err(|m| ferrotorch_core::FerrotorchError::InvalidArgument {
        message: m,
    })?;

    let repo = format!("ferrotorch/{}", args.model);
    eprintln!(
        "[llm_inference_dump] repo = {repo}, device = {:?}",
        args.device
    );

    // -- 1. Download the full bundle into the hub cache. -----------------
    let cache = HubCache::with_default_dir();
    let repo_dir = hf_download_model(&repo, "main", &cache)?;
    eprintln!(
        "[llm_inference_dump] cached at {} ({} files)",
        repo_dir.display(),
        std::fs::read_dir(&repo_dir)
            .map(|r| r.count())
            .unwrap_or(0)
    );

    // -- 2. Parse config + tokenizer. ------------------------------------
    let cfg_path = repo_dir.join("config.json");
    let hf_cfg = HfTransformerConfig::from_file(&cfg_path)?;
    let cfg = LlamaConfig::from_hf(&hf_cfg)?;
    eprintln!(
        "[llm_inference_dump] cfg: hidden={} layers={} heads={} kv={} vocab={} tie={}",
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.num_attention_heads,
        cfg.num_key_value_heads,
        cfg.vocab_size,
        cfg.tie_word_embeddings,
    );

    let tok = load_tokenizer(repo_dir.join("tokenizer.json"))?;

    // -- 3. Resolve the prompt. ------------------------------------------
    let prompt_str = if let Some(p) = args.prompt.clone() {
        p
    } else {
        // Read the parity-probe prompt the pin script froze into the
        // mirror. Strip trailing newline so the tokenization round-trips.
        let parity = repo_dir.join("_value_parity_input.txt");
        let raw = std::fs::read_to_string(&parity).map_err(|e| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!(
                    "missing parity-probe prompt {}: {e}",
                    parity.display()
                ),
            }
        })?;
        raw.trim_end_matches('\n').to_string()
    };
    eprintln!("[llm_inference_dump] prompt = {prompt_str:?}");

    // Re-encode locally so the Rust path exercises ferrotorch-tokenize.
    let local_ids = encode(&tok, &prompt_str, /* add_special_tokens = */ true)?;
    eprintln!(
        "[llm_inference_dump] local encode: {} tokens: {:?}",
        local_ids.len(),
        local_ids,
    );

    // Cross-check against the frozen token-ids JSON so we catch tokenizer
    // drift loudly. The harness will assert these match; we surface it
    // here as well for direct debugging.
    let frozen_path = repo_dir.join("_value_parity_token_ids.json");
    if frozen_path.exists() {
        let frozen = read_token_ids_json(&frozen_path).map_err(|e| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!(
                    "failed reading {}: {e}",
                    frozen_path.display()
                ),
            }
        })?;
        if frozen != local_ids {
            return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!(
                    "tokenizer mismatch: local={local_ids:?} vs frozen={frozen:?}"
                ),
            });
        }
        eprintln!("[llm_inference_dump] local encode matches frozen token_ids");
    }

    // -- 4. Load weights + forward (dispatched on --device). -------------
    // Owned Vec<f32> of length `seq_len * vocab` laid out as [seq, vocab].
    let (data, shape): (Vec<f32>, Vec<usize>) = match args.device {
        Device::Cpu => run_cpu_forward(&repo_dir, cfg, &local_ids)?,
        Device::Gpu => run_gpu_forward(&repo_dir, cfg, &local_ids)?,
    };
    assert_eq!(shape.len(), 3, "logits must be [1, S, V], got {shape:?}");
    let seq_len = shape[1];
    let vocab = shape[2];

    write_dump_f32(&args.output, &shape, &data).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed writing logits to {}: {e}",
                args.output.display()
            ),
        }
    })?;
    eprintln!(
        "[llm_inference_dump] wrote {} ({} bytes, shape={shape:?})",
        args.output.display(),
        std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0),
    );

    // -- 6. JSON verdict line on stdout (harness consumes this). ---------
    let last_offset = (seq_len - 1) * vocab;
    let mut argmax_last_id: u32 = 0;
    let mut argmax_last_val = f32::NEG_INFINITY;
    for (i, &v) in data[last_offset..last_offset + vocab].iter().enumerate() {
        if v > argmax_last_val {
            argmax_last_val = v;
            argmax_last_id = i as u32;
        }
    }
    // Compute argmax_per_position for fast top-1 comparison.
    let mut argmax_per_pos = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let off = s * vocab;
        let mut bid: u32 = 0;
        let mut bval = f32::NEG_INFINITY;
        for (i, &v) in data[off..off + vocab].iter().enumerate() {
            if v > bval {
                bval = v;
                bid = i as u32;
            }
        }
        argmax_per_pos.push(bid);
    }

    // Hand-rolled JSON (no serde_json runtime dep needed in this example).
    let device_tag = match args.device {
        Device::Cpu => "cpu",
        Device::Gpu => "gpu",
    };
    let mut out = String::new();
    out.push('{');
    out.push_str(&format!("\"device\":\"{device_tag}\","));
    out.push_str(&format!(
        "\"shape\":[{},{},{}],",
        shape[0], seq_len, vocab
    ));
    out.push_str(&format!("\"seq_len\":{seq_len},"));
    out.push_str(&format!("\"vocab\":{vocab},"));
    out.push_str(&format!("\"argmax_last\":{argmax_last_id},"));
    out.push_str("\"argmax_per_pos\":[");
    for (i, id) in argmax_per_pos.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&id.to_string());
    }
    out.push_str("],");
    out.push_str("\"token_ids\":[");
    for (i, id) in local_ids.iter().enumerate() {
        if i > 0 {
            out.push(',');
        }
        out.push_str(&id.to_string());
    }
    out.push(']');
    out.push('}');
    println!("{out}");

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[llm_inference_dump] error: {e}");
        std::process::exit(1);
    }
}
