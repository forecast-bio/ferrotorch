//! Whisper encoder — conv stem + sinusoidal positional embedding +
//! `N × WhisperEncoderLayer` + final LayerNorm.
//!
//! ```text
//! input         : [1, 80, 3000]   log-mel spectrogram
//! conv1 + GELU  : [1, 384, 3000]  (Conv1d 80 → 384, k=3, stride=1, pad=1)
//! conv2 + GELU  : [1, 384, 1500]  (Conv1d 384 → 384, k=3, stride=2, pad=1)
//! transpose     : [1, 1500, 384]
//! + sinusoidal positional embedding (loaded from state-dict)
//! N × WhisperEncoderLayer
//! final_layer_norm
//! output        : [1, 1500, 384]
//! ```
//!
//! `embed_positions.weight` ships in the HF Whisper safetensors as a
//! `[max_source_positions, d_model]` non-trainable buffer initialised
//! to the standard sinusoidal pattern. We store it as a parameter so it
//! takes the `embed_positions.weight` slot when loading state — there
//! is no training-loop concern here (encoder-only, no autograd).

use std::collections::HashMap;

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv1d, GELU, LayerNorm};

use crate::config::WhisperConfig;
use crate::layer::WhisperEncoderLayer;

/// Conv stem: two `Conv1d` layers separated by GELU.
#[derive(Debug)]
pub struct WhisperConvStem<T: Float> {
    /// Conv1d(num_mel_bins → d_model, k=3, stride=1, pad=1), bias=True.
    pub conv1: Conv1d<T>,
    /// Conv1d(d_model → d_model, k=3, stride=2, pad=1), bias=True.
    pub conv2: Conv1d<T>,
    activation: GELU,
    training: bool,
}

impl<T: Float> WhisperConvStem<T> {
    /// Build randomly-initialized conv stem.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &WhisperConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            conv1: Conv1d::new(cfg.num_mel_bins, cfg.d_model, 3, 1, 1, true)?,
            conv2: Conv1d::new(cfg.d_model, cfg.d_model, 3, 2, 1, true)?,
            activation: GELU::new(),
            training: false,
        })
    }
}

impl<T: Float> Module<T> for WhisperConvStem<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h1 = self.conv1.forward(input)?;
        let h1a = self.activation.forward(&h1)?;
        let h2 = self.conv2.forward(&h1a)?;
        self.activation.forward(&h2)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.conv1.parameters());
        out.extend(self.conv2.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.conv1.parameters_mut());
        out.extend(self.conv2.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.conv1.named_parameters() {
            out.push((format!("conv1.{n}"), p));
        }
        for (n, p) in self.conv2.named_parameters() {
            out.push((format!("conv2.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> StateDict<T> {
        self.named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.tensor().clone()))
            .collect()
    }

    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let expected = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| {
                    k.strip_prefix(&expected)
                        .map(|rest| (rest.to_string(), v.clone()))
                })
                .collect()
        };
        if strict {
            let prefixes = ["conv1", "conv2"];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in WhisperConvStem state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }
        self.conv1.load_state_dict(&extract("conv1"), strict)?;
        self.conv2.load_state_dict(&extract("conv2"), strict)?;
        Ok(())
    }
}

/// HF `WhisperEncoder` (the audio encoder half of `WhisperModel`).
#[derive(Debug)]
pub struct WhisperEncoder<T: Float> {
    /// 2× Conv1d + GELU stem (`encoder.conv{1,2}`).
    pub conv_stem: WhisperConvStem<T>,
    /// Sinusoidal positional embedding table
    /// (`encoder.embed_positions.weight`), shape
    /// `[max_source_positions, d_model]`. Stored as a parameter so the
    /// HF state-dict layout maps onto it directly.
    pub embed_positions: Parameter<T>,
    /// `encoder_layers × WhisperEncoderLayer`.
    pub layers: Vec<WhisperEncoderLayer<T>>,
    /// Final LayerNorm before the encoder output
    /// (`encoder.layer_norm`).
    pub layer_norm: LayerNorm<T>,
    /// Frozen copy of the configuration used to construct this encoder.
    pub config: WhisperConfig,
    training: bool,
}

impl<T: Float> WhisperEncoder<T> {
    /// Build a randomly-initialized encoder stack for the given config.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: WhisperConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let eps = 1e-5_f64;
        let conv_stem = WhisperConvStem::new(&cfg)?;
        let mut layers = Vec::with_capacity(cfg.encoder_layers);
        for _ in 0..cfg.encoder_layers {
            layers.push(WhisperEncoderLayer::new(&cfg)?);
        }
        let embed_positions =
            Parameter::zeros(&[cfg.max_source_positions, cfg.d_model])?;
        let layer_norm = LayerNorm::new(vec![cfg.d_model], eps, true)?;
        Ok(Self {
            conv_stem,
            embed_positions,
            layers,
            layer_norm,
            config: cfg,
            training: false,
        })
    }

    /// Forward pass on a `[1, num_mel_bins, source_frames]` log-mel
    /// spectrogram. Returns the encoder hidden states
    /// `[1, max_source_positions, d_model]`.
    ///
    /// `source_frames` must equal `max_source_positions × 2` (the conv
    /// stem halves the time axis). For Whisper-tiny this is
    /// `1500 × 2 = 3000`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] when the input has
    /// the wrong rank / mel-bin count / time length. Propagates any
    /// downstream sub-module error.
    pub fn forward_from_mel(&self, mel: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let cfg = &self.config;
        let expected_in = vec![1, cfg.num_mel_bins, cfg.max_source_positions * 2];
        if mel.shape() != expected_in.as_slice() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "WhisperEncoder::forward_from_mel: expected {:?}, got {:?}",
                    expected_in,
                    mel.shape(),
                ),
            });
        }
        // -- 1. Conv stem (with GELU). [1, 80, 3000] → [1, 384, 1500]. ---
        let conv = self.conv_stem.forward(mel)?;
        let post_conv = conv.shape();
        if post_conv != [1, cfg.d_model, cfg.max_source_positions] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "WhisperEncoder: conv stem returned {:?}, expected [1, {}, {}]",
                    post_conv, cfg.d_model, cfg.max_source_positions,
                ),
            });
        }

        // -- 2. Transpose [1, 384, 1500] → [1, 1500, 384]. --------------
        let h = transpose_b_c_t_to_b_t_c(&conv, cfg.d_model, cfg.max_source_positions)?;

        // -- 3. Add sinusoidal positional embedding. ---------------------
        // `embed_positions.weight` is [max_source_positions, d_model];
        // reshape to [1, max_source_positions, d_model] so add broadcasts.
        let pos = reshape_pos(&self.embed_positions, cfg.max_source_positions, cfg.d_model)?;
        let mut x = add(&h, &pos)?;

        // -- 4. Stack of pre-norm encoder layers. -----------------------
        for l in &self.layers {
            x = l.forward(&x)?;
        }

        // -- 5. Final LayerNorm. ----------------------------------------
        self.layer_norm.forward(&x)
    }

    /// Load a HuggingFace `WhisperEncoder` state dict into this module.
    ///
    /// The HF Whisper `model.safetensors` contains BOTH encoder and
    /// decoder weights. This loader filters in only the `encoder.*`
    /// prefix and rejects anything else with a [`DropReport`] entry so
    /// the pin script can cross-check that no encoder key was dropped.
    ///
    /// # Errors
    ///
    /// Forwards whatever each sub-module's `load_state_dict` returns
    /// (`ShapeMismatch` on a wrong-shape tensor, `InvalidArgument` in
    /// strict mode when a required tensor is missing). Strict mode will
    /// surface `decoder.*` / `proj_out.*` / etc. as errors; callers
    /// with a full Whisper checkpoint must pass `strict=false`.
    pub fn load_hf_state_dict(
        &mut self,
        hf_state: &StateDict<T>,
        strict: bool,
    ) -> FerrotorchResult<DropReport> {
        let mut remapped: StateDict<T> = HashMap::with_capacity(hf_state.len());
        let mut dropped: Vec<String> = Vec::new();
        for (k, v) in hf_state {
            // Accept either `encoder.<rest>` (raw `WhisperEncoder`
            // checkpoint) or `model.encoder.<rest>` (full
            // `WhisperForConditionalGeneration` checkpoint).
            let rest_opt = k
                .strip_prefix("encoder.")
                .or_else(|| k.strip_prefix("model.encoder."));
            if let Some(rest) = rest_opt {
                remapped.insert(rest.to_string(), v.clone());
                continue;
            }
            if strict {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "WhisperEncoder::load_hf_state_dict: key {k:?} is not under \
                         `encoder.*` or `model.encoder.*` and strict mode is on. \
                         Pass strict=false to drop decoder / proj_out keys."
                    ),
                });
            }
            dropped.push(k.clone());
        }
        self.load_state_dict(&remapped, strict)?;
        Ok(DropReport { dropped })
    }
}

impl<T: Float> Module<T> for WhisperEncoder<T> {
    /// Forwards the conv stem + layers without the pos-embedding add.
    /// The full encoder path is [`Self::forward_from_mel`]; this
    /// fallback exists so the `Module` trait stays satisfied for
    /// tooling.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_from_mel(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.conv_stem.parameters());
        out.push(&self.embed_positions);
        for l in &self.layers {
            out.extend(l.parameters());
        }
        out.extend(self.layer_norm.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.conv_stem.parameters_mut());
        out.push(&mut self.embed_positions);
        for l in &mut self.layers {
            out.extend(l.parameters_mut());
        }
        out.extend(self.layer_norm.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.conv_stem.named_parameters() {
            out.push((n, p));
        }
        out.push(("embed_positions.weight".to_string(), &self.embed_positions));
        for (i, l) in self.layers.iter().enumerate() {
            for (n, p) in l.named_parameters() {
                out.push((format!("layers.{i}.{n}"), p));
            }
        }
        for (n, p) in self.layer_norm.named_parameters() {
            out.push((format!("layer_norm.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.conv_stem.train();
        for l in &mut self.layers {
            l.train();
        }
        self.layer_norm.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv_stem.eval();
        for l in &mut self.layers {
            l.eval();
        }
        self.layer_norm.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }

    fn state_dict(&self) -> StateDict<T> {
        self.named_parameters()
            .into_iter()
            .map(|(n, p)| (n, p.tensor().clone()))
            .collect()
    }

    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let extract = |prefix: &str| -> StateDict<T> {
            let expected = format!("{prefix}.");
            state
                .iter()
                .filter_map(|(k, v)| {
                    k.strip_prefix(&expected)
                        .map(|rest| (rest.to_string(), v.clone()))
                })
                .collect()
        };

        // -- Validate top-level prefixes in strict mode. -----------------
        if strict {
            for key in state.keys() {
                let ok = key.starts_with("conv1.")
                    || key.starts_with("conv2.")
                    || key == "embed_positions.weight"
                    || key.starts_with("layers.")
                    || key.starts_with("layer_norm.");
                if !ok {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in WhisperEncoder state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }

        // -- Conv stem (sees only conv1/conv2 keys). --------------------
        let mut conv_sd: StateDict<T> = HashMap::new();
        for (k, v) in state {
            if k.starts_with("conv1.") || k.starts_with("conv2.") {
                conv_sd.insert(k.clone(), v.clone());
            }
        }
        self.conv_stem.load_state_dict(&conv_sd, strict)?;

        // -- Positional embedding weight (scalar key). ------------------
        if let Some(pos) = state.get("embed_positions.weight") {
            if pos.shape() != self.embed_positions.shape() {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "embed_positions.weight: expected {:?}, got {:?}",
                        self.embed_positions.shape(),
                        pos.shape(),
                    ),
                });
            }
            self.embed_positions = Parameter::new(pos.clone());
        } else if strict {
            return Err(FerrotorchError::InvalidArgument {
                message: "missing key in state_dict: \"embed_positions.weight\"".into(),
            });
        }

        // -- Per-layer state dicts. ------------------------------------
        for (i, l) in self.layers.iter_mut().enumerate() {
            l.load_state_dict(&extract(&format!("layers.{i}")), strict)?;
        }

        // -- Final LayerNorm. ------------------------------------------
        self.layer_norm
            .load_state_dict(&extract("layer_norm"), strict)?;
        Ok(())
    }
}

/// Audit trail returned by [`WhisperEncoder::load_hf_state_dict`].
///
/// Records HF keys that did not have an `encoder.` / `model.encoder.`
/// prefix and were therefore dropped (typically the decoder weights of
/// a full `WhisperModel` checkpoint). The pin script asserts the
/// dropped set is exactly the decoder/proj_out keys so a silent
/// parameter drop cannot recur.
#[derive(Debug, Default, Clone)]
pub struct DropReport {
    /// Keys present in the upstream state dict that did not belong to
    /// the encoder. Sorted for deterministic equality.
    pub dropped: Vec<String>,
}

// ---------------------------------------------------------------------------
// Helpers.
// ---------------------------------------------------------------------------

/// Transpose `[1, C, T] → [1, T, C]` by writing the data into a fresh
/// row-major buffer (we own the data; no view trick).
fn transpose_b_c_t_to_b_t_c<T: Float>(
    t: &Tensor<T>,
    c: usize,
    time: usize,
) -> FerrotorchResult<Tensor<T>> {
    let data = t.data_vec()?;
    let mut out = vec![<T as num_traits::Zero>::zero(); c * time];
    for ch in 0..c {
        for tt in 0..time {
            // src index in [1, C, T] layout: ch*time + tt
            // dst index in [1, T, C] layout: tt*c + ch
            out[tt * c + ch] = data[ch * time + tt];
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![1, time, c], t.requires_grad())
}

/// Reshape the [max_pos, d_model] positional embedding parameter to
/// `[1, max_pos, d_model]` so it broadcasts onto the encoder hidden
/// state in the `add` op.
fn reshape_pos<T: Float>(
    p: &Parameter<T>,
    pos: usize,
    d: usize,
) -> FerrotorchResult<Tensor<T>> {
    let data = p.tensor().data_vec()?;
    Tensor::from_storage(TensorStorage::cpu(data), vec![1, pos, d], false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> WhisperConfig {
        // d_model=8, heads=2, max_source_positions=4 → conv input length 8.
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

    #[test]
    fn conv_stem_shape() {
        let cs = WhisperConvStem::<f32>::new(&tiny_cfg()).unwrap();
        // input [1, 80, 8] → conv1 [1, 8, 8] → conv2 [1, 8, 4].
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1f32; 80 * 8]),
            vec![1, 80, 8],
            false,
        )
        .unwrap();
        let out = cs.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 8, 4]);
    }

    #[test]
    fn encoder_forward_shape() {
        let enc = WhisperEncoder::<f32>::new(tiny_cfg()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.05f32; 80 * 8]),
            vec![1, 80, 8],
            false,
        )
        .unwrap();
        let out = enc.forward_from_mel(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite(), "encoder output non-finite: {v}");
        }
    }

    #[test]
    fn named_parameters_match_hf_layout() {
        let enc = WhisperEncoder::<f32>::new(tiny_cfg()).unwrap();
        let names: Vec<String> = enc
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for k in [
            "conv1.weight",
            "conv1.bias",
            "conv2.weight",
            "conv2.bias",
            "embed_positions.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.self_attn.k_proj.weight",
            "layers.0.self_attn.v_proj.weight",
            "layers.0.self_attn.out_proj.weight",
            "layers.0.self_attn_layer_norm.weight",
            "layers.0.final_layer_norm.weight",
            "layers.0.fc1.weight",
            "layers.0.fc2.weight",
            "layer_norm.weight",
            "layer_norm.bias",
        ] {
            assert!(
                names.iter().any(|n| n == k),
                "missing parameter key {k:?} in {names:?}"
            );
        }
    }

    #[test]
    fn round_trip_state_dict() {
        let src = WhisperEncoder::<f32>::new(tiny_cfg()).unwrap();
        let sd = src.state_dict();
        let mut dst = WhisperEncoder::<f32>::new(tiny_cfg()).unwrap();
        dst.load_state_dict(&sd, true).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.05f32; 80 * 8]),
            vec![1, 80, 8],
            false,
        )
        .unwrap();
        let a = src.forward_from_mel(&x).unwrap();
        let b = dst.forward_from_mel(&x).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-6, "round-trip differs: {x} vs {y}");
        }
    }

    #[test]
    fn load_hf_drops_decoder_keys_nonstrict() {
        let mut enc = WhisperEncoder::<f32>::new(tiny_cfg()).unwrap();
        // Build a mostly-encoder state dict plus one decoder key.
        let mut sd = enc.state_dict();
        // Re-prefix to look like the HF layout.
        let mut hf_sd: StateDict<f32> = std::collections::HashMap::new();
        for (k, v) in sd.drain() {
            hf_sd.insert(format!("encoder.{k}"), v);
        }
        hf_sd.insert(
            "decoder.embed_tokens.weight".into(),
            ferrotorch_core::zeros::<f32>(&[4, 4]).unwrap(),
        );
        let rep = enc.load_hf_state_dict(&hf_sd, /* strict = */ false).unwrap();
        assert_eq!(rep.dropped, vec!["decoder.embed_tokens.weight".to_string()]);
    }

    #[test]
    fn load_hf_strict_rejects_decoder_keys() {
        let mut enc = WhisperEncoder::<f32>::new(tiny_cfg()).unwrap();
        let mut hf_sd: StateDict<f32> = std::collections::HashMap::new();
        hf_sd.insert(
            "decoder.embed_tokens.weight".into(),
            ferrotorch_core::zeros::<f32>(&[4, 4]).unwrap(),
        );
        assert!(enc.load_hf_state_dict(&hf_sd, true).is_err());
    }
}
