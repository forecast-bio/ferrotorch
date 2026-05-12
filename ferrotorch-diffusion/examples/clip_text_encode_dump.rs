//! CLIP text encoder inference-dump binary for the SD-1.5 real-
//! artifact harness.
//!
//! Companion to `scripts/verify_diffusion_inference.py`. Loads the
//! pinned `ferrotorch/sd-v1-5-clip-text-encoder` mirror from HF, runs
//! the CLIP text encoder forward pass on the frozen parity probe
//! (`_value_parity_input_ids.bin` — pre-tokenized CLIP-BPE ids for the
//! fixed prompt), and dumps the resulting `last_hidden_state` tensor
//! `[1, 77, 768]` to disk in the standard `[u32 ndim][u32 × ndim shape]
//! [f32 data]` little-endian format used across every other ferrotorch
//! dump.
//!
//! Usage (network required for first-touch; subsequent runs use the
//! local hub cache):
//! ```text
//! cargo run -p ferrotorch-diffusion --release --example clip_text_encode_dump -- \
//!     --model sd-v1-5-clip-text-encoder \
//!     --output /tmp/rust_last_hidden_state.bin
//! ```
//!
//! Optional override: `--input-ids` points at a caller-supplied
//! `.bin` dump (in the standard format) of int32 / float32 token ids;
//! absent it falls back to the frozen parity probe shipped by the
//! mirror.
//!
//! Tokenization happens in the pin script (`scripts/pin_pretrained_
//! diffusion_weights.py`) via `transformers.CLIPTokenizer`; the
//! ferrotorch side just consumes the pre-tokenized ids. This keeps the
//! Rust binary tokenizer-free (#1152's scope is the encoder, not the
//! tokenizer).
//!
//! Output:
//!   * `--output <path>`: last_hidden_state tensor `[1, 77, 768]` in
//!     the standard dump format.
//!   * stdout: one JSON line
//!     `{"shape":[1,77,768],"seq_len":77,"vocab_size":49408,
//!       "dropped_keys":N}` so the Python harness can parse the verdict.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_diffusion::{load_clip_text_encoder, ClipTextConfig};
use ferrotorch_hub::{hf_download_model, HubCache};

/// Target device for the forward pass.
///
/// `--device gpu` requires the `cuda` cargo feature; without it the
/// example errors out at arg-parse time so a missing-feature build
/// can't silently fall back to CPU.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Device {
    Cpu,
    Gpu,
}

#[derive(Debug)]
struct Args {
    model: String,
    output: PathBuf,
    input_ids: Option<PathBuf>,
    device: Device,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut input_ids: Option<PathBuf> = None;
    let mut device = Device::Cpu;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(argv.get(i + 1).ok_or("--model needs a value")?.clone());
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output needs a value")?,
                ));
                i += 2;
            }
            "--input-ids" => {
                input_ids = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--input-ids needs a value")?,
                ));
                i += 2;
            }
            "--device" => {
                let v = argv.get(i + 1).ok_or("--device needs a value (cpu|gpu)")?;
                device = match v.as_str() {
                    "cpu" => Device::Cpu,
                    "gpu" => Device::Gpu,
                    other => return Err(format!("--device must be cpu|gpu, got {other:?}")),
                };
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required (e.g. --model sd-v1-5-clip-text-encoder)")?,
        output: output.ok_or("--output is required (path to last_hidden_state .bin)")?,
        input_ids,
        device,
    })
}

fn read_dump_f32(path: &Path) -> Result<(Vec<usize>, Vec<f32>), String> {
    let mut f = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut header4 = [0u8; 4];
    f.read_exact(&mut header4)
        .map_err(|e| format!("read header from {}: {e}", path.display()))?;
    let ndim = u32::from_le_bytes(header4) as usize;
    let mut shape = vec![0usize; ndim];
    for entry in &mut shape {
        f.read_exact(&mut header4)
            .map_err(|e| format!("read shape entry from {}: {e}", path.display()))?;
        *entry = u32::from_le_bytes(header4) as usize;
    }
    let count: usize = shape.iter().product();
    let mut buf = vec![0u8; count * 4];
    f.read_exact(&mut buf)
        .map_err(|e| format!("read data from {}: {e}", path.display()))?;
    let mut data = Vec::with_capacity(count);
    for chunk in buf.chunks_exact(4) {
        data.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Ok((shape, data))
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

/// Resolve the input-ids file: honour the caller's explicit
/// `--input-ids <path>` override if present, else fall back to the
/// frozen `<repo_dir>/_value_parity_input_ids.bin` shipped by the
/// mirror.
fn resolve_input_ids(user_override: Option<&Path>, repo_dir: &Path) -> FerrotorchResult<PathBuf> {
    if let Some(p) = user_override {
        return Ok(p.to_path_buf());
    }
    let parity = repo_dir.join("_value_parity_input_ids.bin");
    if !parity.is_file() {
        return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "neither --input-ids passed nor parity-probe input found at {}",
                parity.display(),
            ),
        });
    }
    Ok(parity)
}

fn run() -> FerrotorchResult<()> {
    let args = parse_args().map_err(|m| ferrotorch_core::FerrotorchError::InvalidArgument {
        message: m,
    })?;

    let repo = format!("ferrotorch/{}", args.model);
    eprintln!("[clip_text_encode_dump] repo = {repo}");

    // -- 1. Download the bundle into the hub cache. ---------------------
    let cache = HubCache::with_default_dir();
    let repo_dir = hf_download_model(&repo, "main", &cache)?;
    eprintln!(
        "[clip_text_encode_dump] cached at {} ({} files)",
        repo_dir.display(),
        std::fs::read_dir(&repo_dir).map(|r| r.count()).unwrap_or(0)
    );

    // -- 2. Parse config. -----------------------------------------------
    let cfg_path = repo_dir.join("config.json");
    let cfg = ClipTextConfig::from_file(&cfg_path)?;
    eprintln!(
        "[clip_text_encode_dump] cfg: hidden_size={} intermediate_size={} \
         num_heads={} num_layers={} max_pos={} vocab_size={} eps={}",
        cfg.hidden_size,
        cfg.intermediate_size,
        cfg.num_attention_heads,
        cfg.num_hidden_layers,
        cfg.max_position_embeddings,
        cfg.vocab_size,
        cfg.layer_norm_eps,
    );

    // -- 3. Resolve and read the input-ids parity probe. ----------------
    let ids_path = resolve_input_ids(args.input_ids.as_deref(), &repo_dir)?;
    let (ids_shape, ids_data) = read_dump_f32(&ids_path).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("failed to read input_ids from {}: {e}", ids_path.display()),
        }
    })?;
    eprintln!(
        "[clip_text_encode_dump] input_ids: shape={ids_shape:?} from {}",
        ids_path.display(),
    );

    // The parity probe is stored as [1, S] but the encoder consumes a
    // 1-D id vector. Verify shape is [1, S] (or [S]) and flatten.
    if !(ids_shape.len() == 1 || (ids_shape.len() == 2 && ids_shape[0] == 1)) {
        return Err(ferrotorch_core::FerrotorchError::ShapeMismatch {
            message: format!(
                "expected input_ids shape [S] or [1, S], got {ids_shape:?}",
            ),
        });
    }
    let seq_len = if ids_shape.len() == 1 {
        ids_shape[0]
    } else {
        ids_shape[1]
    };
    if seq_len != cfg.max_position_embeddings {
        return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "input_ids seq_len {seq_len} != cfg.max_position_embeddings {}",
                cfg.max_position_embeddings
            ),
        });
    }

    // Decode the f32 ids into u32.
    let mut u32_ids: Vec<u32> = Vec::with_capacity(seq_len);
    for (i, &v) in ids_data.iter().enumerate() {
        if !v.is_finite() || v < 0.0 || v.fract() != 0.0 || v > u32::MAX as f32 {
            return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!(
                    "input_ids entry {i} ({v}) is not a non-negative integer"
                ),
            });
        }
        let idv = v as u32;
        if (idv as usize) >= cfg.vocab_size {
            return Err(ferrotorch_core::FerrotorchError::IndexOutOfBounds {
                index: idv as usize,
                axis: 0,
                size: cfg.vocab_size,
            });
        }
        u32_ids.push(idv);
    }

    // -- 4. Load weights and build encoder. -----------------------------
    let weights_path = locate_weights(&repo_dir)?;
    eprintln!(
        "[clip_text_encode_dump] weights file: {}",
        weights_path.display()
    );
    let (encoder, drop_report) =
        load_clip_text_encoder::<f32>(&weights_path, cfg.clone(), /* strict = */ false)?;
    eprintln!(
        "[clip_text_encode_dump] loaded weights: dropped_keys={}",
        drop_report.dropped.len(),
    );

    // -- 5. Forward + dump. --------------------------------------------
    let out = match args.device {
        Device::Cpu => {
            eprintln!("[clip_text_encode_dump] device = cpu");
            encoder.forward_from_ids(&u32_ids)?
        }
        Device::Gpu => run_gpu(&encoder, &u32_ids)?,
    };
    let out_shape = out.shape();
    let out_data = out.data()?;
    assert_eq!(
        out_shape.len(),
        3,
        "CLIP text encoder output must be [B, S, hidden], got {out_shape:?}",
    );
    assert_eq!(out_shape[0], 1, "expected batch=1, got {out_shape:?}");
    assert_eq!(
        out_shape[1], seq_len,
        "encoder shrank seq_len: {} -> {}",
        seq_len, out_shape[1]
    );
    assert_eq!(
        out_shape[2], cfg.hidden_size,
        "hidden dim mismatch: {} vs cfg.hidden_size {}",
        out_shape[2], cfg.hidden_size
    );

    write_dump_f32(&args.output, out_shape, out_data).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed writing last_hidden_state to {}: {e}",
                args.output.display()
            ),
        }
    })?;
    eprintln!(
        "[clip_text_encode_dump] wrote {} ({} bytes, shape={out_shape:?})",
        args.output.display(),
        std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0)
    );

    // -- 6. JSON verdict line. -----------------------------------------
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!(
        "\"shape\":[{},{},{}],",
        out_shape[0], out_shape[1], out_shape[2]
    ));
    s.push_str(&format!("\"seq_len\":{seq_len},"));
    s.push_str(&format!("\"vocab_size\":{},", cfg.vocab_size));
    s.push_str(&format!("\"dropped_keys\":{}", drop_report.dropped.len()));
    s.push('}');
    println!("{s}");

    // Keep the Tensor / TensorStorage symbols used; the local `Tensor::from_storage`
    // dance below is reserved for future input-tensor wiring (e.g. passing the
    // id tensor straight through `forward_from_id_tensor`).
    let _ = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0f32; 1]), vec![1], false);

    Ok(())
}

/// Locate the weights file inside the mirror directory. The pin script
/// uploads as `model.safetensors`; some upstreams use
/// `pytorch_model.bin` (we only accept safetensors).
fn locate_weights(dir: &Path) -> FerrotorchResult<PathBuf> {
    let p = dir.join("model.safetensors");
    if p.is_file() {
        return Ok(p);
    }
    Err(ferrotorch_core::FerrotorchError::InvalidArgument {
        message: format!(
            "model.safetensors not found in {} (CLIP text encoder mirror layout requires it)",
            dir.display()
        ),
    })
}

/// GPU forward path. Builds the [`GpuClipTextEncoder`] from the already-
/// loaded CPU encoder's state-dict and runs `encode` on device 0.
///
/// Without the `cuda` cargo feature this is a hard error — the example
/// must refuse to silently fall back to CPU when the harness asked for
/// GPU.
#[cfg(feature = "cuda")]
fn run_gpu(
    encoder: &ferrotorch_diffusion::ClipTextEncoder<f32>,
    input_ids: &[u32],
) -> FerrotorchResult<Tensor<f32>> {
    use ferrotorch_diffusion::gpu::GpuClipTextEncoder;
    use ferrotorch_gpu::GpuDevice;

    eprintln!("[clip_text_encode_dump] device = gpu");
    let device =
        GpuDevice::new(0).map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("GpuDevice::new(0) failed: {e}"),
        })?;
    let (gpu, report) = GpuClipTextEncoder::from_module(encoder, &device)?;
    eprintln!(
        "[clip_text_encode_dump] gpu state-dict load: dropped_keys={}",
        report.dropped.len(),
    );
    gpu.encode(input_ids)
}

#[cfg(not(feature = "cuda"))]
fn run_gpu(
    _encoder: &ferrotorch_diffusion::ClipTextEncoder<f32>,
    _input_ids: &[u32],
) -> FerrotorchResult<Tensor<f32>> {
    Err(ferrotorch_core::FerrotorchError::InvalidArgument {
        message: "--device gpu requires the `cuda` cargo feature \
                  (build with `--features=cuda`)"
            .into(),
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[clip_text_encode_dump] error: {e}");
        std::process::exit(1);
    }
}
