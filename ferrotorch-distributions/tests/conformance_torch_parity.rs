//! Real-artifact torch.distributions parity tests, gated on network.
//!
//! Runs `scripts/verify_distributions_inference.py` end-to-end against
//! the pinned `ferrotorch/distributions-parity-v1` HF mirror (#1167).
//! Marked `#[ignore]` since it requires network access (to first-touch
//! the HF mirror) and a Python environment with `huggingface_hub`,
//! `numpy`, `torch` installed.
//!
//! Enable via:
//!
//! ```text
//! cargo test -p ferrotorch-distributions --test conformance_torch_parity \
//!     -- --ignored
//! ```
//!
//! Mirrors the optimizer / dataloader / GNN / SD-pipeline real-artifact
//! conformance test wrappers in shape: shell out to the Python harness,
//! assert on its PASS verdict lines.

use std::path::PathBuf;
use std::process::Command;

/// Resolve the workspace root from this crate's `CARGO_MANIFEST_DIR`.
fn workspace_root() -> PathBuf {
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    manifest
        .parent()
        .expect("ferrotorch-distributions manifest must have a parent (the workspace root)")
        .to_path_buf()
}

/// Every (config) name the harness verifies. Asserting on `<name>: PASS`
/// rather than `PASS` alone catches a future regression that silently
/// skips a config and reports the (now smaller) remaining set as
/// all-PASS. The naming matches the configs in
/// `scripts/pin_pretrained_distributions_fixtures.py`.
const EXPECTED_PASS_LINES: &[&str] = &[
    // 17 distributions
    "normal_standard: PASS",
    "normal_shifted: PASS",
    "beta_25: PASS",
    "gamma_21: PASS",
    "cauchy_standard: PASS",
    "exponential_1p5: PASS",
    "uniform_neg2_3: PASS",
    "lognormal_0_p5: PASS",
    "laplace_0_1: PASS",
    "halfnormal_1: PASS",
    "studentt_df5: PASS",
    "bernoulli_p3: PASS",
    "poisson_3: PASS",
    "categorical_k4: PASS",
    "dirichlet_k4: PASS",
    "mvn_3d: PASS",
    "multinomial_k3_n20: PASS",
    "transformed_normal_affine: PASS",
    // 8 KL pairs
    "kl_normal_normal: PASS",
    "kl_bernoulli_bernoulli: PASS",
    "kl_uniform_uniform: PASS",
    "kl_categorical_categorical: PASS",
    "kl_laplace_laplace: PASS",
    "kl_exponential_exponential: PASS",
    "kl_gamma_gamma: PASS",
    "kl_poisson_poisson: PASS",
];

#[test]
#[ignore = "Requires network access — enable with --ignored"]
fn pretrained_distributions_parity_smoke() {
    let root = workspace_root();
    let harness = root.join("scripts").join("verify_distributions_inference.py");
    assert!(
        harness.is_file(),
        "harness missing at {}",
        harness.display()
    );

    let output = Command::new("python3")
        .arg(&harness)
        .arg("--quiet")
        .current_dir(&root)
        .output()
        .expect("failed to launch verify_distributions_inference.py");

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert!(
        output.status.success(),
        "harness exited non-zero ({:?}).\nstdout:\n{stdout}\nstderr:\n{stderr}",
        output.status,
    );

    for expected in EXPECTED_PASS_LINES {
        assert!(
            stdout.contains(expected),
            "expected '{expected}' in stdout but got:\nstdout:\n{stdout}\nstderr:\n{stderr}"
        );
    }
    assert!(
        !stdout.contains(" FAIL"),
        "stdout contains a FAIL verdict line:\nstdout:\n{stdout}\nstderr:\n{stderr}"
    );
}
