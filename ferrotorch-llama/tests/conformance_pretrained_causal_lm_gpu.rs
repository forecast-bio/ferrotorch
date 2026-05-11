//! Real-artifact pretrained causal-LM GPU bf16 parity test — gated on
//! network access AND a working CUDA device. Crosslink #1154.
//!
//! Runs `scripts/verify_causal_lm_gpu_inference.py` end-to-end against
//! the pinned `ferrotorch/smollm-135m` HF mirror, exercising the
//! `LlamaGpuInferencer` (bf16) forward path and asserting numerical
//! parity against the CPU f32 transformers reference (#1147).
//!
//! Marked `#[ignore]` since it requires both:
//!   * network access (to first-touch the HF mirror)
//!   * a working CUDA device + Python `transformers`, `torch`,
//!     `huggingface_hub`, `numpy` installed
//!
//! Enable via:
//!
//! ```text
//! cargo test -p ferrotorch-llama --features cuda \
//!     --test conformance_pretrained_causal_lm_gpu -- --ignored
//! ```
//!
//! Honest behaviour when CUDA is missing: the test does NOT panic — it
//! prints a clear "GPU not available, skipping" line and returns OK.
//! This is deliberate: a runtime panic would fail CPU-only CI runs
//! that happen to ignore-list this test by mistake. The CUDA-absence
//! branch is also documented in #1154's task spec.
//!
//! Mirrors the CPU-side `conformance_pretrained_causal_lm` cargo test
//! at the harness level and the GPU smoke test at the runtime
//! CUDA-availability check.

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

/// Runtime CUDA-availability probe. Returns `true` iff
/// [`ferrotorch_gpu::GpuDevice::new(0)`] succeeds. Compiled out unless
/// `feature = "cuda"` is on — without that feature this whole test is
/// already conditionally compiled away (see the `#[cfg]` on the test
/// fn) so we never reach a "would-be GPU" branch from CPU-only code.
#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    ferrotorch_gpu::GpuDevice::new(0).is_ok()
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "Requires network access + CUDA device — enable with --ignored"]
fn pretrained_causal_lm_gpu_parity_smoke() {
    // 1. Detect CUDA at runtime. If absent (CPU-only CI, missing
    //    driver, etc.) we skip with a clear message rather than
    //    panicking. The task spec explicitly forbids a panic on
    //    CUDA-unavailable so this branch DOES NOT use `assert!`.
    if !cuda_available() {
        eprintln!(
            "[conformance_pretrained_causal_lm_gpu] CUDA device 0 unavailable; \
             skipping GPU bf16 parity harness. Build was compiled with \
             `--features cuda` but no usable CUDA device was found at runtime."
        );
        return;
    }

    let root = workspace_root();
    let harness = root
        .join("scripts")
        .join("verify_causal_lm_gpu_inference.py");
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
        .expect("failed to launch verify_causal_lm_gpu_inference.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Exit code 0 = all PASS; non-zero = at least one FAIL.
    assert!(
        output.status.success(),
        "GPU harness exited non-zero ({:?}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
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

/// When built without `--features cuda` this test is conditionally
/// compiled away. We retain a placeholder fn here only as a doc anchor
/// — `cargo test --test conformance_pretrained_causal_lm_gpu` (no
/// feature) prints the standard "running 0 tests" line and exits OK,
/// which is the contract a CPU-only CI run needs.
#[cfg(not(feature = "cuda"))]
#[test]
#[ignore = "feature 'cuda' not enabled — rebuild with --features cuda"]
fn pretrained_causal_lm_gpu_parity_smoke_requires_cuda_feature() {
    eprintln!(
        "[conformance_pretrained_causal_lm_gpu] built without `--features cuda`; \
         GPU bf16 parity test is unreachable on this build."
    );
}
