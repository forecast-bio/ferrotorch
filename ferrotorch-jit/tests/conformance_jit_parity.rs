//! Real-artifact conformance test for ferrotorch-jit tracing + AoT
//! compilation parity vs torch (Phase G.4, #1170).
//!
//! Calls `scripts/verify_jit_inference.py` which:
//!
//!   1. Resolves the `_value_parity_{input,output}.bin` reference
//!      fixtures (either from the pin script's WORK_DIR, or via
//!      `huggingface_hub.hf_hub_download` from
//!      `ferrotorch/jit-trace-parity-v1`).
//!   2. Drives the `jit_trace_dump` example which runs eager / traced
//!      / compiled forwards on the pinned MLP.
//!   3. Verifies the three stages against torch (eager) and against
//!      ferrotorch eager (traced + compiled) with the frozen
//!      tolerances:
//!
//!     eager  vs torch    : max_abs <= 1e-4, cosine_sim >= 0.99999
//!     traced vs eager    : max_abs <= 1e-5, cosine_sim >= 0.99999
//!     compiled vs eager  : max_abs <= 1e-4, cosine_sim >= 0.9999
//!
//! Gated behind `#[ignore]` so the test is network-aware and only
//! runs on operator request (`cargo test --test conformance_jit_parity
//! -- --ignored`).

#![allow(clippy::missing_panics_doc)]

use std::path::PathBuf;
use std::process::Command;

/// Repository root resolved from `CARGO_MANIFEST_DIR`. The manifest
/// lives at `ferrotorch-jit/Cargo.toml`, so the parent is the
/// workspace root.
fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("ferrotorch-jit/Cargo.toml must have a parent (workspace root)")
        .to_path_buf()
}

#[test]
#[ignore = "network-aware real-artifact harness; run with --ignored"]
fn jit_trace_parity_via_python_harness() {
    let script = repo_root().join("scripts/verify_jit_inference.py");
    assert!(
        script.exists(),
        "verify_jit_inference.py not found at {}",
        script.display()
    );

    let output = Command::new("python3")
        .arg(&script)
        .current_dir(repo_root())
        .output()
        .expect("failed to spawn python3");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    eprintln!("---- verify_jit_inference.py stdout ----\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("---- verify_jit_inference.py stderr ----\n{stderr}");
    }

    assert!(
        output.status.success(),
        "verify_jit_inference.py exited with {:?}; PASS line not produced",
        output.status
    );
    // Belt-and-braces: the script's last line should be `OVERALL: PASS`.
    assert!(
        stdout.contains("OVERALL: PASS"),
        "verify_jit_inference.py did not report OVERALL: PASS"
    );
}
