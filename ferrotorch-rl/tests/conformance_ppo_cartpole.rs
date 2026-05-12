//! Real-artifact PPO-CartPole parity test, gated on network.
//!
//! Runs `scripts/verify_rl_inference.py` end-to-end against the pinned
//! `ferrotorch/ppo-cartpole-v1` HuggingFace mirror. Marked `#[ignore]`
//! because it requires network access (to first-touch the HF mirror)
//! and a Python environment with `torch`, `stable-baselines3`,
//! `huggingface_hub`, `numpy`, `safetensors` installed.
//!
//! Enable via:
//!
//! ```text
//! cargo test -p ferrotorch-rl --test conformance_ppo_cartpole -- --ignored
//! ```
//!
//! Mirrors the graph-side `conformance_gcn_cora` and BERT-side
//! `conformance_pretrained_text_embedding` cargo tests.

use std::path::PathBuf;
use std::process::Command;

/// Resolve the workspace root from this crate's `CARGO_MANIFEST_DIR`.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("ferrotorch-rl manifest must have a parent (the workspace root)")
        .to_path_buf()
}

#[test]
#[ignore = "Requires network access — enable with --ignored"]
fn pretrained_ppo_cartpole_parity_smoke() {
    let root = workspace_root();
    let harness = root.join("scripts").join("verify_rl_inference.py");
    assert!(
        harness.is_file(),
        "harness missing at {}",
        harness.display()
    );

    let output = Command::new("python3")
        .arg(&harness)
        .args(["--models", "ppo-cartpole-v1", "--quiet"])
        .current_dir(&root)
        .output()
        .expect("failed to launch verify_rl_inference.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "harness exited non-zero ({:?}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status,
    );
    assert!(
        stdout.contains("ppo-cartpole-v1: PASS"),
        "expected 'ppo-cartpole-v1: PASS' in stdout but got:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
    assert!(
        !stdout.contains(" FAIL"),
        "stdout contains a FAIL verdict line:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
