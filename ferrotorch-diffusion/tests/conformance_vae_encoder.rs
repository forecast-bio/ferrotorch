//! Real-artifact VAE encoder conformance tests, gated on network and
//! the (forthcoming) `ferrotorch/sd-v1-5-vae-encoder` mirror.
//!
//! Two test surfaces:
//!
//!   1. `vae_encoder_sd_scale_shape_sanity` — constructs a full
//!      SD-1.5-sized [`VaeEncoder`] with *random* weights and runs it
//!      on a `[1, 3, 512, 512]` image. Asserts the output is
//!      `[1, 8, 64, 64]` and finite. Catches scale-up bugs (wrong
//!      stride, broken downsample) at production dimensions without
//!      needing real weights. Marked `#[ignore]` because it allocates
//!      ~600 MB and runs ~10 s on CPU.
//!
//!   2. `vae_encoder_diffusers_parity_smoke` — placeholder that, once
//!      a `ferrotorch/sd-v1-5-vae-encoder` HF mirror exists in
//!      `ferrotorch-hub/src/registry.rs`, will download the mirror,
//!      load it via [`load_vae_encoder`], run on the frozen reference
//!      input produced by `scripts/verify_vae_encoder.py`, and compare
//!      numerically against the diffusers reference. Today it asserts
//!      the mirror is missing and skips (so `cargo test --ignored`
//!      surfaces the gap without failing CI).
//!
//! Run with:
//!
//! ```text
//! cargo test -p ferrotorch-diffusion \
//!     --test conformance_vae_encoder -- --ignored
//! ```

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_diffusion::{VaeDecoderConfig, VaeEncoder, VaeEncoderConfig};
use ferrotorch_nn::module::Module;

/// SD-1.5 VAE config (canonical defaults from
/// `diffusers.AutoencoderKL.config` for `runwayml/stable-diffusion-v1-5`).
fn sd_v1_5_config() -> VaeEncoderConfig {
    VaeDecoderConfig::sd_v1_5()
}

/// Build a deterministic constant-pattern image — RGB stripes along H,
/// in `[-1, 1]` (the value range the SD VAE encoder expects). Avoids
/// constant-zero inputs that would silently mask bugs in the
/// post-norm path.
fn striped_image(b: usize, h: usize, w: usize) -> Tensor<f32> {
    let mut data = Vec::with_capacity(b * 3 * h * w);
    for _ in 0..b {
        for c in 0..3 {
            for y in 0..h {
                for _ in 0..w {
                    // Different gradient per channel — encoder will produce
                    // a different mean/logvar pattern in each latent channel.
                    let base = (y as f32 / h as f32) * 2.0 - 1.0;
                    let v = base + (c as f32) * 0.05;
                    data.push(v.clamp(-1.0, 1.0));
                }
            }
        }
    }
    Tensor::from_storage(TensorStorage::cpu(data), vec![b, 3, h, w], false)
        .expect("striped_image: tensor construction must succeed")
}

#[test]
#[ignore = "Allocates ~600 MB / 10 s CPU — enable with --ignored"]
fn vae_encoder_sd_scale_shape_sanity() {
    let cfg = sd_v1_5_config();
    let enc = VaeEncoder::<f32>::new(cfg.clone())
        .expect("VaeEncoder::new must succeed for the canonical SD-1.5 config");

    // Single 512x512 RGB image — the canonical SD input.
    let x = striped_image(1, 512, 512);
    let params = enc
        .forward(&x)
        .expect("VaeEncoder forward must succeed at SD-1.5 scale");

    // SD-1.5: latent grid is 512/8 = 64 per axis; channels = 2 * 4 = 8
    // (mean + logvar concatenated).
    assert_eq!(
        params.shape(),
        &[1, 2 * cfg.latent_channels, 64, 64],
        "SD-1.5 VAE encoder must produce [1, 8, 64, 64], got {:?}",
        params.shape()
    );

    // Every value must be finite — non-finites would indicate a numerical
    // blow-up in one of the down-blocks, the mid-block, or the output conv.
    let mut count_nonfinite = 0usize;
    for &v in params.data().expect("params must have data") {
        if !v.is_finite() {
            count_nonfinite += 1;
        }
    }
    assert_eq!(
        count_nonfinite, 0,
        "SD-1.5 VAE encoder produced {count_nonfinite} non-finite values"
    );

    // The diagonal-Gaussian split must produce mean/logvar tensors of
    // shape [1, 4, 64, 64] each, with the logvar clamped to [-30, 20].
    let dist = enc
        .encode(&x)
        .expect("VaeEncoder::encode must succeed at SD-1.5 scale");
    assert_eq!(dist.mean.shape(), &[1, 4, 64, 64]);
    assert_eq!(dist.logvar.shape(), &[1, 4, 64, 64]);
    for &v in dist.logvar.data().expect("logvar must have data") {
        assert!(
            v.is_finite() && (-30.0..=20.0).contains(&v),
            "logvar value {v} outside the [-30, 20] clamp range"
        );
    }
}

#[test]
#[ignore = "Awaits the `ferrotorch/sd-v1-5-vae-encoder` HF mirror — enable with --ignored"]
fn vae_encoder_diffusers_parity_smoke() {
    // When the mirror lands in `ferrotorch-hub/src/registry.rs` (entry
    // `sd-v1-5-vae-encoder`), replace this block with the standard
    // `hf_download_model` + `load_vae_encoder` + reference-input
    // comparison pattern used by `conformance_pretrained_diffusion.rs`.
    //
    // Until then this test surfaces the missing mirror without failing
    // CI: under `--ignored` it asserts the gap and prints a pointer to
    // the python reference script. The script
    // (`scripts/verify_vae_encoder.py`) is already in place and can be
    // used to produce the reference numerics once a mirror exists.
    let mirror_present = ferrotorch_hub::registry::get_model_info("sd-v1-5-vae-encoder").is_some();
    if !mirror_present {
        eprintln!(
            "vae_encoder_diffusers_parity_smoke: skipping — \
             `ferrotorch/sd-v1-5-vae-encoder` mirror is not yet registered. \
             To enable, (1) run `scripts/verify_vae_encoder.py --pin` to \
             produce the encoder safetensors + `_value_parity_*.bin` \
             fixtures, (2) push the mirror to HuggingFace, (3) add a \
             `ModelInfo` entry to ferrotorch-hub/src/registry.rs."
        );
        return;
    }

    // The mirror exists — exercise the full parity path. This branch is
    // unreachable today but is left in place so the test transitions
    // from skipping to running automatically once the mirror lands.
    panic!(
        "vae_encoder_diffusers_parity_smoke: mirror present but parity harness \
         not implemented. Wire up via the same pattern as \
         conformance_pretrained_diffusion.rs."
    );
}
