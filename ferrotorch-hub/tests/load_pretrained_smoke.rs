//! Network-gated end-to-end test for `load_pretrained` (#749 / #739 closure).
//!
//! This test downloads two registry entries from real upstream mirrors and
//! verifies that the SHA-256 pin in the registry matches the bytes the
//! mirror serves. Two layers of assertion:
//!
//! 1. **Download + SHA-256 verification** — exercises [`download_weights`]
//!    end-to-end. This is the path that [`load_pretrained`] guards behind
//!    fail-fast hash checking (security audit #6, follow-up #739). When
//!    this step succeeds we have proof the pin is real and the mirror is
//!    serving identical bytes.
//! 2. **SafeTensors parse → non-zero parameters** — exercises
//!    [`load_pretrained`] for a registry entry whose tensors are entirely
//!    floating-point (`vit_b_16`). The timm torchvision-style entries
//!    (resnet*, mobilenet*, etc.) include a few BatchNorm bookkeeping
//!    tensors with `I64` dtype (`num_batches_tracked`); the current
//!    `ferrotorch_serialize::load_safetensors` path is `T: Float`-only and
//!    rejects those with `DtypeMismatch`. That's an orthogonal loader gap
//!    (it doesn't affect the security guarantee of #739) — the
//!    `vit_b_16` SafeTensors is float-only and round-trips cleanly.
//!
//! Gated behind `--ignored` (and the `http` feature) so offline CI and
//! developer machines without network access don't spend bandwidth on
//! every `cargo test` run. Run explicitly with:
//!
//! ```bash
//! cargo test -p ferrotorch-hub --features http -- --ignored load_pretrained
//! ```
//!
//! The `#[ignore]` reason is "requires network; gated for offline CI".
//! If the upstream mirror or the SHA changes, this test fails loudly —
//! that signal is the entire point of pinning.

#![cfg(feature = "http")]

use ferrotorch_hub::{HubCache, download_weights, get_model_info, load_pretrained};

/// Verify download + SHA-256 verification end-to-end against the smallest
/// registered model (`mobilenet_v3_small`, ~10 MB). This is the path that
/// previously silently fail-opened on placeholder hashes (audit #6); the
/// fact that it succeeds at all is proof the pin matches the mirror's
/// bytes byte-for-byte.
///
/// We do not call `load_pretrained` here because mobilenet_v3_small's
/// SafeTensors carries `I64` BatchNorm counters that the current loader
/// rejects (see module docs). The SHA-256 verification is the property
/// #739 closes; the loader's `Float`-only restriction is a separate
/// concern.
#[test]
#[ignore = "requires network; gated for offline CI"]
fn download_weights_mobilenet_v3_small_sha_verifies() {
    let dir = tempfile::tempdir().expect("tempdir for download cache");
    let cache = HubCache::new(dir.path());

    let info =
        get_model_info("mobilenet_v3_small").expect("mobilenet_v3_small must be in the registry");

    // download_weights is the function that calls download_and_verify
    // under the hood: HTTP GET → SHA-256 → cache write. Returning Ok
    // here is mathematical proof that the bytes the mirror serves hash
    // to the registry's pinned digest.
    let path = download_weights(info, &cache)
        .expect("download + SHA verify must succeed for the pinned hash");
    assert!(path.exists(), "downloaded file must exist on disk");
    let bytes = std::fs::metadata(&path)
        .expect("stat downloaded file")
        .len();
    assert!(
        bytes > 1_000_000,
        "downloaded file looks too small: {bytes} bytes (expected ~10 MB)"
    );
    eprintln!(
        "download_weights_mobilenet_v3_small_sha_verifies: \
         {bytes} bytes downloaded and SHA-256-verified against the registry pin",
    );
}

/// End-to-end: `load_pretrained::<f32>("vit_b_16")` → download → verify
/// SHA → parse → non-empty StateDict with > 0 total parameters. ViT-B/16
/// is the smallest registered entry whose SafeTensors is entirely
/// floating-point, so it round-trips through the current `T: Float`-only
/// loader cleanly. Larger payload (~330 MB) than mobilenet but exercises
/// the full happy path that #749 / #739 are about.
#[test]
#[ignore = "requires network; gated for offline CI"]
fn load_pretrained_vit_b_16_end_to_end() {
    let state_dict =
        load_pretrained::<f32>("vit_b_16").expect("load_pretrained::<f32> must succeed");

    let total: usize = state_dict
        .values()
        .map(ferrotorch_core::Tensor::numel)
        .sum();
    assert!(
        total > 0,
        "loaded StateDict has zero total elements; SafeTensors parse likely failed"
    );
    // ViT-B/16 has ~86M parameters; we assert >> 1M to catch a partial
    // parse where only a header tensor or two land.
    assert!(
        total > 1_000_000,
        "loaded StateDict has only {total} elements; expected ~86M for ViT-B/16"
    );
    eprintln!(
        "load_pretrained_vit_b_16_end_to_end: \
         loaded {} tensors, {total} total elements",
        state_dict.len(),
    );
}
