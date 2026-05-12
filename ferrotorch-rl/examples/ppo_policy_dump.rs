//! PPO policy forward-dump binary for the RL real-artifact harness.
//!
//! Companion to `scripts/verify_rl_inference.py`. The harness pulls
//! the pinned policy (`ferrotorch/<model>`) plus a fixed observation
//! (shipped on the HF mirror as `_value_parity_obs.bin`) and invokes
//! this binary to:
//!
//!   1. Pull `model.safetensors` for `ferrotorch/<model>` into the
//!      ferrotorch hub cache.
//!   2. Read the fixed obs `[1, obs_dim]` from
//!      `<repo_dir>/_value_parity_obs.bin` (or from `--obs-bin`
//!      override if provided).
//!   3. Build the `MlpPolicy` with the right dims and load the pinned
//!      weights via `load_ppo_policy`.
//!   4. Forward → dump `action_logits` `[B, n_actions]` and `value`
//!      `[B, 1]` to `<output_prefix>_action_logits.bin` and
//!      `<output_prefix>_value.bin`.
//!   5. Print one JSON verdict line to stdout for the harness.
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-rl --release --example ppo_policy_dump -- \
//!     --model ppo-cartpole-v1 \
//!     --obs-dim 4 --hidden 64 --n-actions 2 \
//!     --output-prefix /tmp/rust_ppo
//! ```

use std::fs::File;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_hub::{HubCache, hf_download_model};
use ferrotorch_rl::{MlpPolicyConfig, load_ppo_policy};

#[derive(Debug)]
struct Args {
    model: String,
    obs_bin: Option<PathBuf>,
    output_prefix: PathBuf,
    obs_dim: usize,
    hidden: usize,
    n_actions: usize,
}

fn parse_args() -> Result<Args, String> {
    let mut model: Option<String> = None;
    let mut obs_bin: Option<PathBuf> = None;
    let mut output_prefix: Option<PathBuf> = None;
    let mut obs_dim: Option<usize> = None;
    let mut hidden: Option<usize> = None;
    let mut n_actions: Option<usize> = None;
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1usize;
    while i < argv.len() {
        match argv[i].as_str() {
            "--model" => {
                model = Some(argv.get(i + 1).ok_or("--model needs a value")?.clone());
                i += 2;
            }
            "--obs-bin" => {
                obs_bin = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--obs-bin needs a value")?,
                ));
                i += 2;
            }
            "--output-prefix" => {
                output_prefix = Some(PathBuf::from(
                    argv.get(i + 1).ok_or("--output-prefix needs a value")?,
                ));
                i += 2;
            }
            "--obs-dim" => {
                obs_dim = Some(
                    argv.get(i + 1)
                        .ok_or("--obs-dim needs a value")?
                        .parse()
                        .map_err(|e| format!("--obs-dim parse: {e}"))?,
                );
                i += 2;
            }
            "--hidden" => {
                hidden = Some(
                    argv.get(i + 1)
                        .ok_or("--hidden needs a value")?
                        .parse()
                        .map_err(|e| format!("--hidden parse: {e}"))?,
                );
                i += 2;
            }
            "--n-actions" => {
                n_actions = Some(
                    argv.get(i + 1)
                        .ok_or("--n-actions needs a value")?
                        .parse()
                        .map_err(|e| format!("--n-actions parse: {e}"))?,
                );
                i += 2;
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
    }
    Ok(Args {
        model: model.ok_or("--model is required")?,
        obs_bin,
        output_prefix: output_prefix.ok_or("--output-prefix is required")?,
        obs_dim: obs_dim.ok_or("--obs-dim is required")?,
        hidden: hidden.ok_or("--hidden is required")?,
        n_actions: n_actions.ok_or("--n-actions is required")?,
    })
}

/// Read a `[u32 ndim][u32 × ndim shape][f32 × prod(shape)]` LE blob.
fn read_dump_f32(path: &Path) -> std::io::Result<(Vec<usize>, Vec<f32>)> {
    let mut f = File::open(path)?;
    let mut buf4 = [0u8; 4];
    f.read_exact(&mut buf4)?;
    let ndim = u32::from_le_bytes(buf4) as usize;
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        f.read_exact(&mut buf4)?;
        shape.push(u32::from_le_bytes(buf4) as usize);
    }
    let n: usize = shape.iter().product();
    let mut data = vec![0.0_f32; n];
    let mut bytes = vec![0u8; n * 4];
    f.read_exact(&mut bytes)?;
    for (i, slot) in data.iter_mut().enumerate() {
        let b = &bytes[i * 4..(i + 1) * 4];
        *slot = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
    }
    Ok((shape, data))
}

fn write_dump_f32(path: &Path, shape: &[usize], data: &[f32]) -> std::io::Result<()> {
    let expected: usize = shape.iter().product();
    assert_eq!(
        data.len(),
        expected,
        "data len {} != shape product {}",
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
    let args = parse_args().map_err(|m| FerrotorchError::InvalidArgument { message: m })?;

    // -- 1. Hub-cache the safetensors + the parity-probe fixtures. ----------
    let repo = format!("ferrotorch/{}", args.model);
    eprintln!("[ppo_policy_dump] repo = {repo}");
    let cache = HubCache::with_default_dir();
    let repo_dir = hf_download_model(&repo, "main", &cache)?;
    eprintln!(
        "[ppo_policy_dump] cached at {} ({} files)",
        repo_dir.display(),
        std::fs::read_dir(&repo_dir)
            .map(|r| r.count())
            .unwrap_or(0)
    );

    // -- 2. Resolve the obs bin (CLI override > mirror's parity probe). -----
    let obs_path = if let Some(p) = args.obs_bin.clone() {
        p
    } else {
        repo_dir.join("_value_parity_obs.bin")
    };
    let (obs_shape, obs_data) = read_dump_f32(&obs_path).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("failed reading {}: {e}", obs_path.display()),
        }
    })?;
    if obs_shape.len() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("--obs-bin must be 2-D [B, obs_dim], got {obs_shape:?}"),
        });
    }
    let batch = obs_shape[0];
    let f_in = obs_shape[1];
    if f_in != args.obs_dim {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "obs.shape()[-1] = {f_in} != --obs-dim {}",
                args.obs_dim
            ),
        });
    }
    let obs = Tensor::<f32>::from_storage(TensorStorage::cpu(obs_data), obs_shape.clone(), false)?;
    eprintln!("[ppo_policy_dump] obs shape = {obs_shape:?}");

    // -- 3. Build MlpPolicy and load pinned weights. ------------------------
    let cfg = MlpPolicyConfig {
        obs_dim: args.obs_dim,
        hidden: args.hidden,
        n_actions: args.n_actions,
    };
    let weights_path = repo_dir.join("model.safetensors");
    let (policy, report) = load_ppo_policy(&weights_path, cfg, /* strict = */ true)?;
    eprintln!(
        "[ppo_policy_dump] loaded weights: unmapped={:?}",
        report.unmapped
    );

    // -- 4. Forward → (action_logits, value). -------------------------------
    let out = policy.forward(&obs)?;
    let logits_shape = out.action_logits.shape().to_vec();
    let value_shape = out.value.shape().to_vec();
    if logits_shape != vec![batch, args.n_actions] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "action_logits shape {logits_shape:?} != [B={batch}, n_actions={}]",
                args.n_actions
            ),
        });
    }
    if value_shape != vec![batch, 1] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("value shape {value_shape:?} != [B={batch}, 1]"),
        });
    }
    let logits_data = out.action_logits.data_vec()?;
    let value_data = out.value.data_vec()?;

    // -- 5. Dump + verdict line. -------------------------------------------
    let prefix_str = args.output_prefix.to_string_lossy();
    let logits_path: PathBuf = format!("{prefix_str}_action_logits.bin").into();
    let value_path: PathBuf = format!("{prefix_str}_value.bin").into();
    write_dump_f32(&logits_path, &logits_shape, &logits_data).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("failed writing {}: {e}", logits_path.display()),
        }
    })?;
    write_dump_f32(&value_path, &value_shape, &value_data).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("failed writing {}: {e}", value_path.display()),
        }
    })?;
    eprintln!(
        "[ppo_policy_dump] wrote {} + {}",
        logits_path.display(),
        value_path.display()
    );

    let mut s = String::new();
    s.push('{');
    s.push_str(&format!(
        "\"logits_shape\":[{},{}],",
        logits_shape[0], logits_shape[1]
    ));
    s.push_str(&format!(
        "\"value_shape\":[{},{}],",
        value_shape[0], value_shape[1]
    ));
    s.push_str(&format!("\"obs_dim\":{},", args.obs_dim));
    s.push_str(&format!("\"hidden\":{},", args.hidden));
    s.push_str(&format!("\"n_actions\":{},", args.n_actions));
    s.push_str(&format!("\"unmapped\":{}", report.unmapped.len()));
    s.push('}');
    println!("{s}");
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("[ppo_policy_dump] error: {e}");
        std::process::exit(1);
    }
}
