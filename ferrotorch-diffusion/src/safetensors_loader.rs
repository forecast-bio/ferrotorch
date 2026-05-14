//! Helpers that turn a path-to-safetensors into a loaded
//! [`VaeDecoder`].
//!
//! The pinned SD-1.5 VAE mirror carries the full VAE state-dict
//! (encoder + post_quant_conv + decoder + quant_conv). Inference needs
//! only the decoder slice (`post_quant_conv.*` + `decoder.*`). This
//! loader drops everything else and returns a [`DropReport`] so the pin
//! script can audit the drop set.

use std::collections::HashMap;
use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_serialize::load_safetensors;

use crate::clip_text_encoder::{ClipTextConfig, ClipTextEncoder};
use crate::config::VaeDecoderConfig;
use crate::unet::UNet2DConditionModel;
use crate::unet_config::UNet2DConditionConfig;
use crate::vae::VaeDecoder;
use crate::vae_encoder::{VaeEncoder, VaeEncoderConfig};

/// Audit trail returned by [`load_vae_decoder`] / [`VaeDecoder::load_hf_state_dict`].
///
/// Records HF keys that were dropped because they do not belong to the
/// decoder (typically the encoder + `quant_conv` weights of a full
/// `AutoencoderKL` checkpoint). The pin script asserts the dropped set
/// equals the documented encoder / quant_conv key surface so a silent
/// parameter drop cannot recur.
#[derive(Debug, Default, Clone)]
pub struct DropReport {
    /// Keys present in the upstream state dict that did not belong to
    /// the VAE decoder. Sorted for deterministic equality.
    pub dropped: Vec<String>,
}

impl<T: Float> VaeDecoder<T> {
    /// Load a HuggingFace AutoencoderKL state dict into this module.
    ///
    /// Accepts both:
    ///   - `post_quant_conv.*` / `decoder.*` (bare-VAE layout, the
    ///     normalised form the pin script produces)
    ///   - `vae.post_quant_conv.*` / `vae.decoder.*` (when bundled
    ///     inside a full SD pipeline checkpoint)
    ///
    /// Any other key (encoder, `quant_conv`, etc.) is recorded in the
    /// returned [`DropReport`] (or, in strict mode, surfaces as
    /// [`FerrotorchError::InvalidArgument`]).
    ///
    /// # Errors
    ///
    /// Forwards whatever each sub-module's `load_state_dict` returns
    /// (`ShapeMismatch` on a wrong-shape tensor, `InvalidArgument` in
    /// strict mode when a required tensor is missing). Strict mode will
    /// surface `encoder.*` / `quant_conv.*` / etc. as errors; callers
    /// with a full VAE checkpoint must pass `strict=false`.
    pub fn load_hf_state_dict(
        &mut self,
        hf_state: &StateDict<T>,
        strict: bool,
    ) -> FerrotorchResult<DropReport> {
        let mut remapped: StateDict<T> = HashMap::with_capacity(hf_state.len());
        let mut dropped: Vec<String> = Vec::new();

        for (k, v) in hf_state {
            // Try (a) bare-VAE prefix → as-is; (b) full-pipeline
            // `vae.<rest>` prefix → strip the `vae.` and accept.
            let after_vae = k.strip_prefix("vae.").map_or_else(|| k.clone(), str::to_owned);
            if after_vae.starts_with("post_quant_conv.") || after_vae.starts_with("decoder.") {
                remapped.insert(after_vae, v.clone());
                continue;
            }
            if strict {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "VaeDecoder::load_hf_state_dict: key {k:?} is not under \
                         `post_quant_conv.*` / `decoder.*` (with optional `vae.` prefix) \
                         and strict mode is on. Pass strict=false to drop encoder / \
                         quant_conv keys."
                    ),
                });
            }
            dropped.push(k.clone());
        }
        dropped.sort();
        self.load_state_dict(&remapped, strict)?;
        Ok(DropReport { dropped })
    }
}

// ---------------------------------------------------------------------------
// UNet2DConditionModel loader
// ---------------------------------------------------------------------------

impl<T: Float> UNet2DConditionModel<T> {
    /// Load a HuggingFace UNet state dict into this module.
    ///
    /// Accepts both:
    ///   - bare-UNet layout (the pin script normalises to this form)
    ///   - `unet.<rest>` prefix (full SD pipeline checkpoint)
    ///
    /// Any unrecognised key is recorded in the returned [`DropReport`]
    /// (or surfaces as [`FerrotorchError::InvalidArgument`] in strict
    /// mode).
    ///
    /// # Errors
    ///
    /// Forwards whatever each sub-module's `load_state_dict` returns
    /// (shape mismatch / strict-mode missing key).
    pub fn load_hf_state_dict(
        &mut self,
        hf_state: &StateDict<T>,
        strict: bool,
    ) -> FerrotorchResult<DropReport> {
        let mut remapped: StateDict<T> = HashMap::with_capacity(hf_state.len());
        let mut dropped: Vec<String> = Vec::new();
        for (k, v) in hf_state {
            let after_unet = k.strip_prefix("unet.").map_or_else(|| k.clone(), str::to_owned);
            let is_unet_key = after_unet.starts_with("time_embedding.")
                || after_unet.starts_with("conv_in.")
                || after_unet.starts_with("down_blocks.")
                || after_unet.starts_with("mid_block.")
                || after_unet.starts_with("up_blocks.")
                || after_unet.starts_with("conv_norm_out.")
                || after_unet.starts_with("conv_out.");
            if is_unet_key {
                remapped.insert(after_unet, v.clone());
                continue;
            }
            if strict {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "UNet2DConditionModel::load_hf_state_dict: key {k:?} is not under \
                         a UNet prefix (with optional `unet.`) and strict mode is on."
                    ),
                });
            }
            dropped.push(k.clone());
        }
        dropped.sort();
        self.load_state_dict(&remapped, strict)?;
        Ok(DropReport { dropped })
    }
}

/// Load a [`UNet2DConditionModel`] from a UNet
/// `diffusion_pytorch_model.safetensors` file plus a parsed config.
///
/// `strict=false` is required when loading a full SD pipeline
/// checkpoint (which carries `vae.*` / `text_encoder.*` keys); for a
/// bare-UNet mirror (the form `pin_pretrained_diffusion_weights.py`
/// uploads) `strict=true` is fine.
///
/// # Errors
///
/// Propagates safetensors parse errors, [`UNet2DConditionModel`]
/// construction errors, and any per-key shape / strict-mode mismatch.
pub fn load_unet<T: Float>(
    weights_path: &Path,
    cfg: UNet2DConditionConfig,
    strict: bool,
) -> FerrotorchResult<(UNet2DConditionModel<T>, DropReport)> {
    let state =
        load_safetensors::<T>(weights_path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "load_unet: failed to decode safetensors {}: {e}",
                weights_path.display()
            ),
        })?;
    let mut unet = UNet2DConditionModel::<T>::new(cfg)?;
    let report = unet.load_hf_state_dict(&state, strict)?;
    Ok((unet, report))
}

// ---------------------------------------------------------------------------
// ClipTextEncoder loader
// ---------------------------------------------------------------------------

/// Read a single non-sharded safetensors file into a typed `StateDict`,
/// dropping any int64 `position_ids` buffer BEFORE the generic-`T`
/// decode. The CLIP-text checkpoint ships an
/// `embeddings.position_ids` (or `text_model.embeddings.position_ids`)
/// `[1, 77]` int64 buffer that would poison a `load_safetensors::<f32>`
/// pass because i64 is not representable as f32 in the underlying
/// dispatch. Mirrors the trick `ferrotorch-bert`'s loader uses.
fn load_safetensors_clip_filtered<T: Float>(
    weights_path: &Path,
) -> FerrotorchResult<(StateDict<T>, bool)> {
    use safetensors::SafeTensors;

    let bytes =
        std::fs::read(weights_path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "load_safetensors_clip_filtered: failed to read {}: {e}",
                weights_path.display()
            ),
        })?;
    let st = SafeTensors::deserialize(&bytes).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!(
            "load_safetensors_clip_filtered: failed to parse {}: {e}",
            weights_path.display()
        ),
    })?;
    let mut keep: Vec<String> = Vec::new();
    let mut had_position_ids = false;
    for k in st.names() {
        let s: &str = k.as_str();
        // The position_ids buffer is the only int64 surface in
        // CLIPTextModel and it has no parameter slot on our side.
        if s == "embeddings.position_ids" || s == "text_model.embeddings.position_ids" {
            had_position_ids = true;
            continue;
        }
        keep.push(String::from(s));
    }

    // Re-serialize only the kept tensors into an in-memory safetensors
    // blob and feed that to `load_safetensors::<T>`. Reuses the audited
    // generic decoder instead of re-implementing dtype dispatch here.
    let mut subset: Vec<(String, safetensors::tensor::TensorView<'_>)> =
        Vec::with_capacity(keep.len());
    for k in &keep {
        let v = st.tensor(k).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "load_safetensors_clip_filtered: missing tensor {k:?} after filter: {e}"
            ),
        })?;
        subset.push((k.clone(), v));
    }
    let serialized = safetensors::serialize(subset, &None).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("load_safetensors_clip_filtered: re-serialize failed: {e}"),
        }
    })?;
    let tmp = tempfile::NamedTempFile::new().map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("load_safetensors_clip_filtered: tempfile: {e}"),
    })?;
    std::fs::write(tmp.path(), &serialized).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("load_safetensors_clip_filtered: tempfile write: {e}"),
    })?;
    let state = load_safetensors::<T>(tmp.path())?;
    Ok((state, had_position_ids))
}

/// Load a [`ClipTextEncoder`] from a CLIP text-tower
/// `model.safetensors` file plus a parsed [`ClipTextConfig`].
///
/// Accepts both upstream layouts:
///   - bare `embeddings.* / encoder.* / final_layer_norm.*` (what the
///     pin script normalises to).
///   - `text_model.<rest>` prefix (what the upstream HF checkpoint
///     ships).
///
/// The int64 `embeddings.position_ids` buffer (a `[1, max_pos]`
/// `arange(max_pos)` constant regenerated each forward pass) is
/// dropped at decode time and surfaced via the returned
/// [`DropReport`].
///
/// `strict=false` is required when the upstream checkpoint carries the
/// position_ids buffer (the default for `runwayml/stable-diffusion-v1-5`'s
/// `text_encoder/model.safetensors`).
///
/// # Errors
///
/// Propagates safetensors parse errors, [`ClipTextEncoder`] construction
/// errors, and any per-key shape / strict-mode mismatch.
pub fn load_clip_text_encoder<T: Float>(
    weights_path: &Path,
    cfg: ClipTextConfig,
    strict: bool,
) -> FerrotorchResult<(ClipTextEncoder<T>, DropReport)> {
    let (mut state, had_position_ids) =
        load_safetensors_clip_filtered::<T>(weights_path).map_err(|e| {
            FerrotorchError::InvalidArgument {
                message: format!(
                    "load_clip_text_encoder: failed to decode safetensors {}: {e}",
                    weights_path.display()
                ),
            }
        })?;

    // Re-insert a placeholder entry for the position_ids buffer (with
    // the upstream key it actually used) so the model's DropReport
    // captures it as an intentionally-dropped upstream key. The
    // placeholder tensor is never consumed — `load_hf_state_dict`
    // drops the entry before any parameter slot sees it.
    if had_position_ids {
        let key = if state
            .keys()
            .any(|k| k.starts_with("text_model."))
        {
            "text_model.embeddings.position_ids".to_string()
        } else {
            "embeddings.position_ids".to_string()
        };
        state.insert(key, ferrotorch_core::zeros::<T>(&[1])?);
    }

    let mut enc = ClipTextEncoder::<T>::new(cfg)?;
    let report = enc.load_hf_state_dict(&state, strict)?;
    Ok((enc, report))
}

/// Load a [`VaeDecoder`] from a VAE `diffusion_pytorch_model.safetensors`
/// file plus a parsed config.
///
/// `strict=false` is required for a full `AutoencoderKL` checkpoint
/// (which ships encoder + quant_conv weights this decoder-only loader
/// has no slot for). The returned [`DropReport`] captures every
/// dropped key so the pin script can confirm the drop set is exactly
/// the documented encoder/quant_conv surface.
///
/// # Errors
///
/// Propagates safetensors parse errors, [`VaeDecoder`] construction
/// errors, and any per-key shape / strict-mode mismatch from the
/// underlying load.
pub fn load_vae_decoder<T: Float>(
    weights_path: &Path,
    cfg: VaeDecoderConfig,
    strict: bool,
) -> FerrotorchResult<(VaeDecoder<T>, DropReport)> {
    let state =
        load_safetensors::<T>(weights_path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "load_vae_decoder: failed to decode safetensors {}: {e}",
                weights_path.display()
            ),
        })?;
    let mut decoder = VaeDecoder::<T>::new(cfg)?;
    let report = decoder.load_hf_state_dict(&state, strict)?;
    Ok((decoder, report))
}

// ---------------------------------------------------------------------------
// VaeEncoder loader
// ---------------------------------------------------------------------------

impl<T: Float> VaeEncoder<T> {
    /// Load a HuggingFace AutoencoderKL state dict into this module.
    ///
    /// Accepts both:
    ///   - `encoder.*` / `quant_conv.*` (bare-VAE layout, the normalised
    ///     form produced by the pin script for an encoder-only artifact)
    ///   - `vae.encoder.*` / `vae.quant_conv.*` (full SD pipeline
    ///     checkpoint, e.g. an upstream `runwayml/stable-diffusion-v1-5`
    ///     `model.safetensors`)
    ///
    /// Any other key (decoder, `post_quant_conv`, UNet, text encoder, …)
    /// is recorded in the returned [`DropReport`] (or, in strict mode,
    /// surfaces as [`FerrotorchError::InvalidArgument`]). This mirrors
    /// [`VaeDecoder::load_hf_state_dict`] so the same full-checkpoint
    /// file can be loaded twice — once for the encoder and once for the
    /// decoder — each time dropping the other half.
    ///
    /// # Errors
    ///
    /// Forwards whatever each sub-module's `load_state_dict` returns
    /// (`ShapeMismatch` on a wrong-shape tensor, `InvalidArgument` in
    /// strict mode when a required tensor is missing). Strict mode will
    /// surface `decoder.*` / `post_quant_conv.*` / etc. as errors;
    /// callers with a full VAE checkpoint must pass `strict=false`.
    pub fn load_hf_state_dict(
        &mut self,
        hf_state: &StateDict<T>,
        strict: bool,
    ) -> FerrotorchResult<DropReport> {
        let mut remapped: StateDict<T> = HashMap::with_capacity(hf_state.len());
        let mut dropped: Vec<String> = Vec::new();

        for (k, v) in hf_state {
            // (a) bare-VAE prefix → as-is; (b) full-pipeline `vae.<rest>`
            // prefix → strip the `vae.` and accept.
            let after_vae = k.strip_prefix("vae.").map_or_else(|| k.clone(), str::to_owned);
            if after_vae.starts_with("encoder.") || after_vae.starts_with("quant_conv.") {
                remapped.insert(after_vae, v.clone());
                continue;
            }
            if strict {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "VaeEncoder::load_hf_state_dict: key {k:?} is not under \
                         `encoder.*` / `quant_conv.*` (with optional `vae.` prefix) \
                         and strict mode is on. Pass strict=false to drop decoder / \
                         post_quant_conv keys."
                    ),
                });
            }
            dropped.push(k.clone());
        }
        dropped.sort();
        self.load_state_dict(&remapped, strict)?;
        Ok(DropReport { dropped })
    }
}

/// Load a [`VaeEncoder`] from a VAE `diffusion_pytorch_model.safetensors`
/// file plus a parsed config.
///
/// Symmetric counterpart of [`load_vae_decoder`]. `strict=false` is
/// required for a full `AutoencoderKL` checkpoint (which ships
/// `decoder` + `post_quant_conv` weights this encoder-only loader has
/// no slot for). The returned [`DropReport`] captures every dropped
/// key so a pin script can audit the drop set.
///
/// # Errors
///
/// Propagates safetensors parse errors, [`VaeEncoder`] construction
/// errors, and any per-key shape / strict-mode mismatch from the
/// underlying load.
pub fn load_vae_encoder<T: Float>(
    weights_path: &Path,
    cfg: VaeEncoderConfig,
    strict: bool,
) -> FerrotorchResult<(VaeEncoder<T>, DropReport)> {
    let state =
        load_safetensors::<T>(weights_path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "load_vae_encoder: failed to decode safetensors {}: {e}",
                weights_path.display()
            ),
        })?;
    let mut encoder = VaeEncoder::<T>::new(cfg)?;
    let report = encoder.load_hf_state_dict(&state, strict)?;
    Ok((encoder, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};
    use ferrotorch_serialize::save_safetensors;
    use std::path::PathBuf;

    fn tiny_cfg() -> VaeDecoderConfig {
        VaeDecoderConfig {
            out_channels: 3,
            latent_channels: 4,
            block_out_channels: vec![4, 8, 16, 16],
            layers_per_block: 1,
            norm_num_groups: 4,
            sample_size: 8,
            scaling_factor: 0.18215,
        }
    }

    fn tmp_safetensors_from(v: &VaeDecoder<f32>) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        // The on-disk file uses the bare-VAE prefix (no `vae.`); the
        // loader's strip-vae path is exercised by the dedicated test
        // below.
        let sd = v.state_dict();
        save_safetensors(&sd, &path).unwrap();
        (dir, path)
    }

    #[test]
    fn round_trip_safetensors_into_decoder() {
        let cfg = tiny_cfg();
        let src = VaeDecoder::<f32>::new(cfg.clone()).unwrap();
        let (_d, p) = tmp_safetensors_from(&src);
        let (dst, report) = load_vae_decoder::<f32>(&p, cfg.clone(), false).unwrap();
        assert!(
            report.dropped.is_empty(),
            "round-trip should have empty drop list, got {:?}",
            report.dropped
        );
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 4]),
            vec![1, 4, 1, 1],
            false,
        )
        .unwrap();
        let a = src.forward(&x).unwrap();
        let b = dst.forward(&x).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn load_hf_drops_encoder_keys_nonstrict() {
        let cfg = tiny_cfg();
        let mut v = VaeDecoder::<f32>::new(cfg).unwrap();
        let mut hf_sd: StateDict<f32> = v.state_dict();
        // Add an encoder key — this should be dropped.
        hf_sd.insert(
            "encoder.conv_in.weight".into(),
            ferrotorch_core::zeros::<f32>(&[4, 4]).unwrap(),
        );
        // Add a quant_conv key — also dropped.
        hf_sd.insert(
            "quant_conv.weight".into(),
            ferrotorch_core::zeros::<f32>(&[4, 4]).unwrap(),
        );
        let rep = v.load_hf_state_dict(&hf_sd, false).unwrap();
        assert_eq!(
            rep.dropped,
            vec![
                "encoder.conv_in.weight".to_string(),
                "quant_conv.weight".to_string(),
            ]
        );
    }

    #[test]
    fn load_hf_strict_rejects_encoder_keys() {
        let cfg = tiny_cfg();
        let mut v = VaeDecoder::<f32>::new(cfg).unwrap();
        let mut hf_sd: StateDict<f32> = HashMap::new();
        hf_sd.insert(
            "encoder.conv_in.weight".into(),
            ferrotorch_core::zeros::<f32>(&[4, 4]).unwrap(),
        );
        assert!(v.load_hf_state_dict(&hf_sd, true).is_err());
    }

    #[test]
    fn load_hf_strips_vae_prefix() {
        let cfg = tiny_cfg();
        let src = VaeDecoder::<f32>::new(cfg.clone()).unwrap();
        let bare = src.state_dict();
        // Re-prefix with `vae.` (the layout SD pipeline checkpoints use).
        let mut prefixed: StateDict<f32> = HashMap::new();
        for (k, v) in bare {
            prefixed.insert(format!("vae.{k}"), v);
        }
        let mut dst = VaeDecoder::<f32>::new(cfg).unwrap();
        let rep = dst.load_hf_state_dict(&prefixed, false).unwrap();
        assert!(rep.dropped.is_empty(), "got {:?}", rep.dropped);
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 4]),
            vec![1, 4, 1, 1],
            false,
        )
        .unwrap();
        let a = src.forward(&x).unwrap();
        let b = dst.forward(&x).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    fn tmp_encoder_safetensors_from(v: &VaeEncoder<f32>) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        let sd = v.state_dict();
        save_safetensors(&sd, &path).unwrap();
        (dir, path)
    }

    #[test]
    fn round_trip_safetensors_into_encoder() {
        let cfg = tiny_cfg();
        let src = VaeEncoder::<f32>::new(cfg.clone()).unwrap();
        let (_d, p) = tmp_encoder_safetensors_from(&src);
        let (dst, report) = load_vae_encoder::<f32>(&p, cfg.clone(), false).unwrap();
        assert!(
            report.dropped.is_empty(),
            "encoder round-trip should have empty drop list, got {:?}",
            report.dropped
        );
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 3 * 8 * 8]),
            vec![1, 3, 8, 8],
            false,
        )
        .unwrap();
        let a = src.forward(&x).unwrap();
        let b = dst.forward(&x).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn encoder_load_hf_drops_decoder_keys_nonstrict() {
        let cfg = tiny_cfg();
        let mut v = VaeEncoder::<f32>::new(cfg).unwrap();
        let mut hf_sd: StateDict<f32> = v.state_dict();
        // Add decoder keys that the encoder should drop.
        hf_sd.insert(
            "decoder.conv_in.weight".into(),
            ferrotorch_core::zeros::<f32>(&[4, 4]).unwrap(),
        );
        hf_sd.insert(
            "post_quant_conv.weight".into(),
            ferrotorch_core::zeros::<f32>(&[4, 4]).unwrap(),
        );
        let rep = v.load_hf_state_dict(&hf_sd, false).unwrap();
        assert_eq!(
            rep.dropped,
            vec![
                "decoder.conv_in.weight".to_string(),
                "post_quant_conv.weight".to_string(),
            ]
        );
    }

    #[test]
    fn encoder_load_hf_strict_rejects_decoder_keys() {
        let cfg = tiny_cfg();
        let mut v = VaeEncoder::<f32>::new(cfg).unwrap();
        let mut hf_sd: StateDict<f32> = HashMap::new();
        hf_sd.insert(
            "decoder.conv_in.weight".into(),
            ferrotorch_core::zeros::<f32>(&[4, 4]).unwrap(),
        );
        assert!(v.load_hf_state_dict(&hf_sd, true).is_err());
    }

    #[test]
    fn encoder_load_hf_strips_vae_prefix() {
        let cfg = tiny_cfg();
        let src = VaeEncoder::<f32>::new(cfg.clone()).unwrap();
        let bare = src.state_dict();
        let mut prefixed: StateDict<f32> = HashMap::new();
        for (k, v) in bare {
            prefixed.insert(format!("vae.{k}"), v);
        }
        let mut dst = VaeEncoder::<f32>::new(cfg).unwrap();
        let rep = dst.load_hf_state_dict(&prefixed, false).unwrap();
        assert!(rep.dropped.is_empty(), "got {:?}", rep.dropped);
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.01f32; 3 * 8 * 8]),
            vec![1, 3, 8, 8],
            false,
        )
        .unwrap();
        let a = src.forward(&x).unwrap();
        let b = dst.forward(&x).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-5);
        }
    }

    #[test]
    fn full_vae_checkpoint_loadable_by_both_halves() {
        // The interesting property: a single full-VAE state dict
        // (encoder + post_quant_conv + decoder + quant_conv) is loadable
        // by both VaeDecoder and VaeEncoder, each cleanly dropping the
        // half it doesn't own.
        let cfg = tiny_cfg();
        let dec_src = VaeDecoder::<f32>::new(cfg.clone()).unwrap();
        let enc_src = VaeEncoder::<f32>::new(cfg.clone()).unwrap();

        let mut combined: StateDict<f32> = HashMap::new();
        for (k, v) in dec_src.state_dict() {
            combined.insert(k, v);
        }
        for (k, v) in enc_src.state_dict() {
            combined.insert(k, v);
        }

        let mut dec_dst = VaeDecoder::<f32>::new(cfg.clone()).unwrap();
        let dec_rep = dec_dst.load_hf_state_dict(&combined, false).unwrap();
        let mut enc_dst = VaeEncoder::<f32>::new(cfg).unwrap();
        let enc_rep = enc_dst.load_hf_state_dict(&combined, false).unwrap();

        // Decoder should have dropped exactly the encoder + quant_conv keys.
        for k in &dec_rep.dropped {
            assert!(
                k.starts_with("encoder.") || k.starts_with("quant_conv."),
                "decoder dropped unexpected key: {k}"
            );
        }
        // Encoder should have dropped exactly the decoder + post_quant_conv keys.
        for k in &enc_rep.dropped {
            assert!(
                k.starts_with("decoder.") || k.starts_with("post_quant_conv."),
                "encoder dropped unexpected key: {k}"
            );
        }
        assert!(!dec_rep.dropped.is_empty(), "decoder should have dropped some keys");
        assert!(!enc_rep.dropped.is_empty(), "encoder should have dropped some keys");
    }
}
