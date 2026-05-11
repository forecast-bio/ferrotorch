//! Helpers that turn a path-to-safetensors into a loaded
//! [`WhisperEncoder`].
//!
//! The HF Whisper safetensors carries the full encoder+decoder model;
//! [`load_whisper_encoder`] filters in only the `encoder.*` keys (the
//! drop report records the rest so the pin script can audit no encoder
//! key was silently lost).

use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_serialize::load_safetensors;

use crate::config::WhisperConfig;
use crate::encoder::{DropReport, WhisperEncoder};

/// Load a [`WhisperEncoder`] from a `model.safetensors` file plus a
/// parsed config.
///
/// `strict=false` is required for full Whisper checkpoints (they ship
/// decoder + `proj_out` weights this encoder-only loader has no slot
/// for). The returned [`DropReport`] captures every dropped key so the
/// pin script can confirm the drop set is exactly the documented
/// decoder/proj_out surface.
///
/// # Errors
///
/// Propagates safetensors parse errors, [`WhisperEncoder`] construction
/// errors, and any per-key shape / strict-mode mismatch from the
/// underlying load.
pub fn load_whisper_encoder<T: Float>(
    weights_path: &Path,
    cfg: WhisperConfig,
    strict: bool,
) -> FerrotorchResult<(WhisperEncoder<T>, DropReport)> {
    let state = load_safetensors::<T>(weights_path).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!(
                "load_whisper_encoder: failed to decode safetensors {}: {e}",
                weights_path.display()
            ),
        }
    })?;
    let mut encoder = WhisperEncoder::<T>::new(cfg)?;
    let report = encoder.load_hf_state_dict(&state, strict)?;
    Ok((encoder, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_nn::module::Module;
    use ferrotorch_serialize::save_safetensors;
    use std::path::PathBuf;

    fn tiny_cfg() -> WhisperConfig {
        WhisperConfig {
            vocab_size: 32,
            num_mel_bins: 80,
            d_model: 8,
            encoder_layers: 1,
            encoder_attention_heads: 2,
            encoder_ffn_dim: 16,
            decoder_layers: 1,
            decoder_attention_heads: 2,
            decoder_ffn_dim: 16,
            max_source_positions: 4,
            max_target_positions: 4,
        }
    }

    fn tmp_safetensors_from(enc: &WhisperEncoder<f32>) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.safetensors");
        // The on-disk file uses the HF prefix `encoder.<rest>`; mimic
        // that here so the loader's prefix-strip path is exercised.
        let mut hf_sd = std::collections::HashMap::new();
        for (k, v) in enc.state_dict() {
            hf_sd.insert(format!("encoder.{k}"), v);
        }
        save_safetensors(&hf_sd, &path).unwrap();
        (dir, path)
    }

    #[test]
    fn round_trip_safetensors_into_encoder() {
        let src = WhisperEncoder::<f32>::new(tiny_cfg()).unwrap();
        let (_d, p) = tmp_safetensors_from(&src);
        let (dst, report) = load_whisper_encoder::<f32>(&p, tiny_cfg(), false).unwrap();
        assert!(report.dropped.is_empty());
        let x = ferrotorch_core::Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![0.05f32; 80 * 8]),
            vec![1, 80, 8],
            false,
        )
        .unwrap();
        let a = src.forward_from_mel(&x).unwrap();
        let b = dst.forward_from_mel(&x).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-6);
        }
    }
}
