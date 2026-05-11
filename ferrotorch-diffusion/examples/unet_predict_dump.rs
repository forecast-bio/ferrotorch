//! UNet2DConditionModel inference-dump binary for the SD-1.5 real-
//! artifact harness.
//!
//! Companion to `scripts/verify_diffusion_inference.py`. Loads the
//! pinned `ferrotorch/sd-v1-5-unet` mirror from HF, runs the UNet
//! forward pass on the frozen parity probe
//! (`_value_parity_noisy_latent.bin`, `_value_parity_timestep.bin`,
//! `_value_parity_text_embedding.bin` — all shipped by the mirror), and
//! dumps the predicted noise tensor `[1, 4, 64, 64]` to disk in the
//! standard `[u32 ndim][u32 × ndim shape][f32 data]` little-endian
//! format used across vision / causal-LM / text-embedding / audio
//! dumps.
//!
//! Usage (network required for first-touch; subsequent runs use the
//! local hub cache):
//! ```text
//! cargo run -p ferrotorch-diffusion --release --example unet_predict_dump -- \
//!     --model sd-v1-5-unet \
//!     --output /tmp/rust_predicted_noise.bin
//! ```
//!
//! Optional overrides: `--latent`, `--timestep`, `--text-embedding`
//! point at caller-supplied `.bin` dumps in the same format; absent
//! flags fall back to the frozen parity probe shipped by the mirror.
//!
//! The CLIP text encoder is not yet pinned (Phase B.3c), so the
//! `_value_parity_text_embedding.bin` shipped by the mirror is a
//! deterministic `randn(1, 77, 768)` stand-in for the encoder output.
//! The harness exercises the UNet forward pass without depending on a
//! real text encoder.
//!
//! Output:
//!   * `--output <path>`: predicted-noise tensor `[1, 4, 64, 64]` in
//!     the standard dump format.
//!   * stdout: one JSON line
//!     `{"shape":[1,4,64,64],"latent_shape":[1,4,64,64],
//!       "text_shape":[1,77,768],"timestep":500.0,"dropped_keys":N}`
//!     so the Python harness can parse the verdict.

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_diffusion::{load_unet, UNet2DConditionConfig};
use ferrotorch_hub::{hf_download_model, HubCache};

#[derive(Debug)]
struct Args {
    model: String,
    output: PathBuf,
    latent: Option<PathBuf>,
    timestep: Option<PathBuf>,
    text_embedding: Option<PathBuf>,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut output: Option<PathBuf> = None;
    let mut latent: Option<PathBuf> = None;
    let mut timestep: Option<PathBuf> = None;
    let mut text_embedding: Option<PathBuf> = None;
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
            "--latent" => {
                latent = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--latent needs a value")?,
                ));
                i += 2;
            }
            "--timestep" => {
                timestep = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--timestep needs a value")?,
                ));
                i += 2;
            }
            "--text-embedding" => {
                text_embedding = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--text-embedding needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required (e.g. --model sd-v1-5-unet)")?,
        output: output.ok_or("--output is required (path to predicted-noise .bin)")?,
        latent,
        timestep,
        text_embedding,
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

/// Resolve one of the parity-probe inputs: honour the caller's
/// explicit `--flag <path>` override if present, else fall back to the
/// frozen `<repo_dir>/<fallback>` shipped by the mirror.
fn resolve_input(
    user_override: Option<&Path>,
    repo_dir: &Path,
    fallback: &str,
    label: &str,
) -> FerrotorchResult<PathBuf> {
    if let Some(p) = user_override {
        return Ok(p.to_path_buf());
    }
    let parity = repo_dir.join(fallback);
    if !parity.is_file() {
        return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "neither --{label} passed nor parity-probe input found at {}",
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
    eprintln!("[unet_predict_dump] repo = {repo}");

    // -- 1. Download the bundle into the hub cache. ---------------------
    let cache = HubCache::with_default_dir();
    let repo_dir = hf_download_model(&repo, "main", &cache)?;
    eprintln!(
        "[unet_predict_dump] cached at {} ({} files)",
        repo_dir.display(),
        std::fs::read_dir(&repo_dir).map(|r| r.count()).unwrap_or(0)
    );

    // -- 2. Parse config. -----------------------------------------------
    let cfg_path = repo_dir.join("config.json");
    let cfg = UNet2DConditionConfig::from_file(&cfg_path)?;
    eprintln!(
        "[unet_predict_dump] cfg: block_out_channels={:?} layers_per_block={} \
         attention_head_dim={} cross_attention_dim={} sample_size={} \
         in_channels={} out_channels={}",
        cfg.block_out_channels,
        cfg.layers_per_block,
        cfg.attention_head_dim,
        cfg.cross_attention_dim,
        cfg.sample_size,
        cfg.in_channels,
        cfg.out_channels,
    );

    // -- 3. Resolve the three parity-probe inputs. -----------------------
    let latent_path = resolve_input(
        args.latent.as_deref(),
        &repo_dir,
        "_value_parity_noisy_latent.bin",
        "latent",
    )?;
    let timestep_path = resolve_input(
        args.timestep.as_deref(),
        &repo_dir,
        "_value_parity_timestep.bin",
        "timestep",
    )?;
    let text_path = resolve_input(
        args.text_embedding.as_deref(),
        &repo_dir,
        "_value_parity_text_embedding.bin",
        "text-embedding",
    )?;

    let (lat_shape, lat_data) =
        read_dump_f32(&latent_path).map_err(|e| ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed to read noisy latent from {}: {e}",
                latent_path.display()
            ),
        })?;
    let (ts_shape, ts_data) = read_dump_f32(&timestep_path).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed to read timestep from {}: {e}",
                timestep_path.display()
            ),
        }
    })?;
    let (text_shape, text_data) = read_dump_f32(&text_path).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed to read text embedding from {}: {e}",
                text_path.display()
            ),
        }
    })?;
    eprintln!(
        "[unet_predict_dump] inputs: latent shape={lat_shape:?} from {}; \
         timestep shape={ts_shape:?} ({:?}); text shape={text_shape:?}",
        latent_path.display(),
        ts_data.first().copied().unwrap_or(f32::NAN),
    );

    let latent = Tensor::from_storage(TensorStorage::cpu(lat_data), lat_shape.clone(), false)?;
    let timestep = Tensor::from_storage(TensorStorage::cpu(ts_data.clone()), ts_shape, false)?;
    let text_embedding =
        Tensor::from_storage(TensorStorage::cpu(text_data), text_shape.clone(), false)?;

    // -- 4. Load weights and build UNet. --------------------------------
    let weights_path = locate_weights(&repo_dir)?;
    eprintln!(
        "[unet_predict_dump] weights file: {}",
        weights_path.display()
    );
    let (unet, drop_report) =
        load_unet::<f32>(&weights_path, cfg, /* strict = */ false)?;
    eprintln!(
        "[unet_predict_dump] loaded weights: dropped_keys={}",
        drop_report.dropped.len(),
    );

    // -- 5. Forward + dump. --------------------------------------------
    let out = unet.forward_t(&latent, &timestep, &text_embedding)?;
    let out_shape = out.shape();
    let out_data = out.data()?;
    assert_eq!(
        out_shape.len(),
        4,
        "UNet output must be [B, C, H, W], got {out_shape:?}",
    );

    write_dump_f32(&args.output, out_shape, out_data).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!(
                "failed writing predicted noise to {}: {e}",
                args.output.display()
            ),
        }
    })?;
    eprintln!(
        "[unet_predict_dump] wrote {} ({} bytes, shape={out_shape:?})",
        args.output.display(),
        std::fs::metadata(&args.output)
            .map(|m| m.len())
            .unwrap_or(0)
    );

    // -- 6. JSON verdict line. -----------------------------------------
    let ts_scalar = ts_data.first().copied().unwrap_or(f32::NAN);
    let mut s = String::new();
    s.push('{');
    s.push_str(&format!(
        "\"shape\":[{},{},{},{}],",
        out_shape[0], out_shape[1], out_shape[2], out_shape[3]
    ));
    s.push_str(&format!(
        "\"latent_shape\":[{},{},{},{}],",
        lat_shape[0], lat_shape[1], lat_shape[2], lat_shape[3]
    ));
    s.push_str(&format!(
        "\"text_shape\":[{},{},{}],",
        text_shape[0], text_shape[1], text_shape[2]
    ));
    s.push_str(&format!("\"timestep\":{ts_scalar},"));
    s.push_str(&format!("\"dropped_keys\":{}", drop_report.dropped.len()));
    s.push('}');
    println!("{s}");

    Ok(())
}

/// Locate the weights file inside the mirror directory. The pin script
/// uploads as `model.safetensors`; some upstreams use
/// `diffusion_pytorch_model.safetensors`.
fn locate_weights(dir: &Path) -> FerrotorchResult<PathBuf> {
    for name in ["model.safetensors", "diffusion_pytorch_model.safetensors"] {
        let p = dir.join(name);
        if p.is_file() {
            return Ok(p);
        }
    }
    Err(ferrotorch_core::FerrotorchError::InvalidArgument {
        message: format!(
            "neither model.safetensors nor diffusion_pytorch_model.safetensors found in {}",
            dir.display()
        ),
    })
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[unet_predict_dump] error: {e}");
        std::process::exit(1);
    }
}
