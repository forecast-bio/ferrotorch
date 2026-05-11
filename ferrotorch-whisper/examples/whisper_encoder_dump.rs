//! Whisper-encoder inference-dump binary for the audio real-artifact harness.
//!
//! Companion to `scripts/verify_audio_encoder_inference.py`. Loads the
//! pinned encoder-only mirror from `ferrotorch/<name>` on the HuggingFace
//! Hub, runs the encoder forward pass on the parity-probe mel
//! spectrogram (or a caller-supplied `.bin` mel file), and dumps the
//! resulting hidden states `[1, 1500, 384]` to disk in the standard
//! `[u32 ndim][u32 × ndim shape][f32 data]` little-endian format used
//! across vision / causal-LM / text-embedding dumps.
//!
//! Usage (network required for first-touch; subsequent runs use the
//! local hub cache):
//! ```text
//! cargo run -p ferrotorch-whisper --release --example whisper_encoder_dump -- \
//!     --model whisper-tiny-encoder \
//!     --mel /tmp/parity_mel.bin \
//!     --output /tmp/rust_enc.bin
//! ```
//!
//! `--mel` is required and points at a `[u32 ndim][u32 shape][f32]`
//! little-endian dump of the reference log-mel spectrogram (shape
//! `[1, 80, 3000]`). The Python harness writes this with the same
//! format the upstream WhisperFeatureExtractor produced. The mirror
//! also ships `_value_parity_mel.bin` so the harness can fall back to
//! it if `--mel` is omitted.
//!
//! Output:
//!   * `--output <path>`: encoder output tensor `[1, 1500, 384]` in
//!     the standard dump format.
//!   * stdout: one JSON line
//!     `{"shape":[1,1500,384],"mel_shape":[1,80,3000],"dropped_keys":N}`
//!     so the Python harness can parse the verdict.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_hub::{hf_download_model, HubCache};
use ferrotorch_whisper::{
    HfWhisperConfig, WhisperConfig, load_whisper_encoder,
};

#[derive(Debug)]
struct Args {
    model: String,
    output: PathBuf,
    mel: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut mel: Option<PathBuf> = None;
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
            "--mel" => {
                mel = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--mel needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required (e.g. --model whisper-tiny-encoder)")?,
        output: output.ok_or("--output is required (path to encoder hidden-state .bin)")?,
        mel,
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

fn run() -> FerrotorchResult<()> {
    let args = parse_args().map_err(|m| ferrotorch_core::FerrotorchError::InvalidArgument {
        message: m,
    })?;

    let repo = format!("ferrotorch/{}", args.model);
    eprintln!("[whisper_encoder_dump] repo = {repo}");

    // -- 1. Download the full bundle into the hub cache. -----------------
    let cache = HubCache::with_default_dir();
    let repo_dir = hf_download_model(&repo, "main", &cache)?;
    eprintln!(
        "[whisper_encoder_dump] cached at {} ({} files)",
        repo_dir.display(),
        std::fs::read_dir(&repo_dir).map(|r| r.count()).unwrap_or(0)
    );

    // -- 2. Parse config. -----------------------------------------------
    let cfg_path = repo_dir.join("config.json");
    let hf_cfg = HfWhisperConfig::from_file(&cfg_path)?;
    let cfg = WhisperConfig::from_hf(&hf_cfg)?;
    eprintln!(
        "[whisper_encoder_dump] cfg: d_model={} enc_layers={} heads={} mel_bins={} max_src_pos={}",
        cfg.d_model,
        cfg.encoder_layers,
        cfg.encoder_attention_heads,
        cfg.num_mel_bins,
        cfg.max_source_positions,
    );

    // -- 3. Resolve mel input. The harness can supply --mel; otherwise
    //       fall back to the frozen `_value_parity_mel.bin` shipped by
    //       the mirror. ---------------------------------------------------
    let mel_path = if let Some(p) = args.mel.clone() {
        p
    } else {
        let parity = repo_dir.join("_value_parity_mel.bin");
        if !parity.is_file() {
            return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!(
                    "neither --mel passed nor parity-probe mel found at {}",
                    parity.display(),
                ),
            });
        }
        parity
    };
    let (mel_shape, mel_data) =
        read_dump_f32(&mel_path).map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("failed to read mel input from {}: {e}", mel_path.display()),
        })?;
    eprintln!(
        "[whisper_encoder_dump] mel: shape={mel_shape:?} from {}",
        mel_path.display(),
    );
    let mel = Tensor::from_storage(TensorStorage::cpu(mel_data), mel_shape.clone(), false)?;

    // -- 4. Load weights and build encoder. -----------------------------
    let weights_path = repo_dir.join("model.safetensors");
    let (encoder, drop_report) =
        load_whisper_encoder::<f32>(&weights_path, cfg, /* strict = */ false)?;
    eprintln!(
        "[whisper_encoder_dump] loaded weights: dropped_keys={}",
        drop_report.dropped.len(),
    );

    // -- 5. Forward + dump. --------------------------------------------
    let out = encoder.forward_from_mel(&mel)?;
    let out_shape = out.shape();
    let out_data = out.data()?;
    assert_eq!(
        out_shape.len(),
        3,
        "encoder output must be [1, seq, hidden], got {out_shape:?}",
    );

    write_dump_f32(&args.output, out_shape, out_data).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed writing encoder output to {}: {e}",
                args.output.display()
            ),
        }
    })?;
    eprintln!(
        "[whisper_encoder_dump] wrote {} ({} bytes, shape={out_shape:?})",
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
    s.push_str(&format!(
        "\"mel_shape\":[{},{},{}],",
        mel_shape[0], mel_shape[1], mel_shape[2]
    ));
    s.push_str(&format!("\"dropped_keys\":{}", drop_report.dropped.len()));
    s.push('}');
    println!("{s}");

    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[whisper_encoder_dump] error: {e}");
        std::process::exit(1);
    }
}
