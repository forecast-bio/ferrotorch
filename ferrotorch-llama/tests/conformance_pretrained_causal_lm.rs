//! Real-artifact pretrained causal-LM parity test gated on network.
//!
//! Runs `scripts/verify_causal_lm_inference.py` end-to-end against the
//! pinned `ferrotorch/smollm-135m` HF mirror. Marked `#[ignore]` since it
//! requires network access (to first-touch the HF mirror) and a Python
//! environment with `transformers`, `torch`, `huggingface_hub`, `numpy`
//! installed.
//!
//! Enable via:
//!
//! ```text
//! cargo test -p ferrotorch-llama --test conformance_pretrained_causal_lm \
//!     -- --ignored
//! ```
//!
//! Mirrors the vision-side `conformance_pretrained_inference` cargo test.

use std::path::PathBuf;
use std::process::Command;

/// Resolve the workspace root from this crate's `CARGO_MANIFEST_DIR`.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("ferrotorch-llama manifest must have a parent (the workspace root)")
        .to_path_buf()
}

#[test]
#[ignore = "Requires network access — enable with --ignored"]
fn pretrained_causal_lm_parity_smoke() {
    let root = workspace_root();
    let harness = root.join("scripts").join("verify_causal_lm_inference.py");
    assert!(
        harness.is_file(),
        "harness missing at {}",
        harness.display()
    );

    let output = Command::new("python3")
        .arg(&harness)
        .args(["--models", "smollm-135m", "--quiet"])
        .current_dir(&root)
        .output()
        .expect("failed to launch verify_causal_lm_inference.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Exit code 0 = all PASS; non-zero = at least one FAIL.
    assert!(
        output.status.success(),
        "harness exited non-zero ({:?}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status,
    );

    // Belt-and-braces: also verify the verdict line says PASS, not FAIL.
    assert!(
        stdout.contains("smollm-135m: PASS"),
        "expected 'smollm-135m: PASS' in stdout but got:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stdout.contains(" FAIL"),
        "stdout contains a FAIL verdict line:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
