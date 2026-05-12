//! Conformance Phase G.3 — `ferrotorch-serialize` format-parity gate
//! against the canonical references pinned at
//! `ferrotorch/serialize-parity-v1`.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/1169>.
//!
//! The mirror ships four self-contained sub-targets, one per loader/
//! exporter under test:
//!
//!   * `resnet18-pth/` — official torchvision `.pth`
//!     (`resnet18-f37072fd.pth`, ZIP-pickle).
//!   * `safetensors-rt/` — same resnet18 state_dict re-saved via
//!     `safetensors.torch.save_file`.
//!   * `gguf/` — `unsloth/SmolLM2-135M-Instruct-GGUF/SmolLM2-135M-Instruct-Q8_0.gguf`
//!     (all Q8_0 + F32, both supported by ferrotorch-serialize's
//!     GGUF reader today).
//!   * `onnx-mlp/` — fixed-seed `Linear(4 -> 8) + ReLU +
//!     Linear(8 -> 2)` MLP weights + 3 fixed inputs + 3 torch
//!     reference outputs.
//!
//! Per-target hard tolerances (loosening forbidden by dispatch):
//!
//!   * `pth_load` — byte-exact (`max_abs == 0`)
//!   * `safetensors_round_trip` — byte-exact (`max_abs == 0`)
//!   * `gguf_load` — `max_abs <= 1e-4` (block-quant noise floor)
//!   * `onnx_export` — `max_abs <= 1e-5` AND `cosine_sim >= 0.9999`
//!     between (rust-emitted ONNX run via onnxruntime) and
//!     (ferrotorch's own forward).
//!
//! All four tests are `#[ignore]`-gated because they shell out to the
//! python verifier (which downloads from HuggingFace and invokes
//! `onnxruntime` for the ONNX target). Run with:
//!
//! ```text
//! cargo test -p ferrotorch-serialize --test conformance_format_parity -- --ignored
//! ```
//!
//! The python harness (`scripts/verify_serialize_inference.py`) is
//! the single source of truth for the tolerance contract; this
//! cargo-side gate shells out to it so a `cargo test --workspace`
//! run can never accidentally skip parity verification.

use std::path::Path;
use std::process::Command;

const VERIFIER: &str = "scripts/verify_serialize_inference.py";

fn repo_root() -> &'static Path {
    // The crate sits one level under the workspace root.
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("workspace root above CARGO_MANIFEST_DIR")
}

/// Shell out to the python verifier for a single target.
///
/// On success the verifier prints `[PASS] <label>: ...` and exits 0.
/// On failure it exits non-zero and we attach the full stdout/stderr
/// to the panic so the regression is obvious in CI logs.
fn run_verifier(target: &str) {
    let root = repo_root();
    let script = root.join(VERIFIER);
    assert!(
        script.is_file(),
        "missing verifier {} (run from workspace root or fix CARGO_MANIFEST_DIR)",
        script.display(),
    );
    let out = Command::new("python3")
        .arg(&script)
        .arg("--targets")
        .arg(target)
        .current_dir(root)
        .output()
        .unwrap_or_else(|e| panic!("spawn python3 {}: {e}", script.display()));

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    if !out.status.success() {
        panic!(
            "{} --targets {target} failed (exit {:?}):\n--- STDOUT ---\n{}\n--- STDERR ---\n{}",
            script.display(),
            out.status.code(),
            stdout,
            stderr,
        );
    }
    // Defensive: ensure the verifier really printed PASS, not a quiet
    // no-op. The python harness always emits a "<label>: PASS — ..."
    // line in the SUMMARY block; check for it.
    let pass_marker = ": PASS — ";
    assert!(
        stdout.contains(pass_marker),
        "verifier exited 0 but did not print '{pass_marker}' for {target} — stdout:\n{stdout}",
    );
}

#[test]
#[ignore = "downloads ferrotorch/serialize-parity-v1 from HuggingFace and shells out to scripts/verify_serialize_inference.py"]
fn format_parity_pth_load() {
    run_verifier("pth");
}

#[test]
#[ignore = "downloads ferrotorch/serialize-parity-v1 from HuggingFace and shells out to scripts/verify_serialize_inference.py"]
fn format_parity_safetensors_round_trip() {
    run_verifier("safetensors");
}

#[test]
#[ignore = "downloads ferrotorch/serialize-parity-v1 from HuggingFace and shells out to scripts/verify_serialize_inference.py"]
fn format_parity_gguf_load() {
    run_verifier("gguf");
}

#[test]
#[ignore = "downloads ferrotorch/serialize-parity-v1 from HuggingFace and shells out to scripts/verify_serialize_inference.py"]
fn format_parity_onnx_export() {
    run_verifier("onnx");
}
