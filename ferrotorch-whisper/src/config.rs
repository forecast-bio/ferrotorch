//! Typed Whisper encoder configuration.
//!
//! [`WhisperConfig`] is the flat, self-contained struct the encoder
//! layer works against. Construct it directly for synthetic tests, or
//! via [`WhisperConfig::from_hf`] from a HuggingFace Whisper
//! `config.json`.
//!
//! The encoder-only scope means the `decoder_*` fields are parsed and
//! preserved (so [`WhisperConfig`] round-trips a real upstream config)
//! but the decoder forward pass is intentionally not implemented in
//! this crate (Phase B.2 of real-artifact-driven development).

use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};
use serde::Deserialize;

/// Whisper model hyperparameters.
///
/// Mirrors the encoder-relevant union of fields a HuggingFace
/// `WhisperConfig` exposes. Decoder fields are preserved for forward
/// compatibility but the decoder is not implemented in this crate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WhisperConfig {
    /// Vocabulary size (tokenizer vocabulary). Not consumed by the
    /// encoder but kept so the config round-trips.
    pub vocab_size: usize,
    /// Number of mel bins on the encoder input. Whisper uses 80.
    pub num_mel_bins: usize,
    /// Encoder/decoder hidden width (`d_model`). Whisper-tiny: 384.
    pub d_model: usize,
    /// Number of encoder transformer layers. Whisper-tiny: 4.
    pub encoder_layers: usize,
    /// Number of encoder attention heads. Whisper-tiny: 6.
    pub encoder_attention_heads: usize,
    /// Encoder FFN inner dim (`d_ff`). Whisper-tiny: 1536.
    pub encoder_ffn_dim: usize,
    /// Number of decoder layers (parsed, not consumed by the encoder).
    pub decoder_layers: usize,
    /// Number of decoder attention heads (parsed, not consumed).
    pub decoder_attention_heads: usize,
    /// Decoder FFN inner dim (parsed, not consumed).
    pub decoder_ffn_dim: usize,
    /// Number of encoder positional slots. Whisper-tiny: 1500.
    pub max_source_positions: usize,
    /// Number of decoder positional slots (parsed, not consumed).
    pub max_target_positions: usize,
}

impl WhisperConfig {
    /// The published `openai/whisper-tiny` config.
    pub fn whisper_tiny() -> Self {
        Self {
            vocab_size: 51_865,
            num_mel_bins: 80,
            d_model: 384,
            encoder_layers: 4,
            encoder_attention_heads: 6,
            encoder_ffn_dim: 1_536,
            decoder_layers: 4,
            decoder_attention_heads: 6,
            decoder_ffn_dim: 1_536,
            max_source_positions: 1_500,
            max_target_positions: 448,
        }
    }

    /// Build a [`WhisperConfig`] from a parsed HuggingFace
    /// `whisper/config.json`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if validation fails
    /// or the activation function is not `gelu` (the only activation
    /// the ferrotorch-whisper forward pass implements).
    pub fn from_hf(hf: &HfWhisperConfig) -> FerrotorchResult<Self> {
        hf.validate()?;
        Ok(Self {
            vocab_size: hf.vocab_size,
            num_mel_bins: hf.num_mel_bins,
            d_model: hf.d_model,
            encoder_layers: hf.encoder_layers,
            encoder_attention_heads: hf.encoder_attention_heads,
            encoder_ffn_dim: hf.encoder_ffn_dim,
            decoder_layers: hf.decoder_layers,
            decoder_attention_heads: hf.decoder_attention_heads,
            decoder_ffn_dim: hf.decoder_ffn_dim,
            max_source_positions: hf.max_source_positions,
            max_target_positions: hf.max_target_positions,
        })
    }

    /// Per-head feature dimension: `d_model / encoder_attention_heads`.
    pub fn encoder_head_dim(&self) -> usize {
        self.d_model / self.encoder_attention_heads
    }

    /// Enforce invariants the encoder relies on.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when any size is
    /// zero or when `d_model % encoder_attention_heads != 0`.
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.num_mel_bins == 0
            || self.d_model == 0
            || self.encoder_layers == 0
            || self.encoder_attention_heads == 0
            || self.encoder_ffn_dim == 0
            || self.max_source_positions == 0
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "WhisperConfig: num_mel_bins / d_model / encoder_layers / \
                          encoder_attention_heads / encoder_ffn_dim / \
                          max_source_positions must all be positive"
                    .into(),
            });
        }
        if self.d_model % self.encoder_attention_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "WhisperConfig: d_model ({}) must be divisible by \
                     encoder_attention_heads ({})",
                    self.d_model, self.encoder_attention_heads
                ),
            });
        }
        Ok(())
    }
}

/// Parsed contents of a HuggingFace Whisper `config.json`.
///
/// `#[non_exhaustive]` so the schema can grow without breaking callers.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct HfWhisperConfig {
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Number of mel bins.
    pub num_mel_bins: usize,
    /// Hidden dimension.
    pub d_model: usize,
    /// Encoder layer count.
    pub encoder_layers: usize,
    /// Encoder attention head count.
    pub encoder_attention_heads: usize,
    /// Encoder FFN inner dim.
    pub encoder_ffn_dim: usize,
    /// Decoder layer count.
    pub decoder_layers: usize,
    /// Decoder attention head count.
    pub decoder_attention_heads: usize,
    /// Decoder FFN inner dim.
    pub decoder_ffn_dim: usize,
    /// Encoder positional capacity.
    pub max_source_positions: usize,
    /// Decoder positional capacity.
    pub max_target_positions: usize,
    /// Activation function. Whisper uses `"gelu"`.
    #[serde(default = "default_activation_function")]
    pub activation_function: String,
    /// Architectures (e.g. `["WhisperForConditionalGeneration"]`).
    #[serde(default)]
    pub architectures: Vec<String>,
    /// Model type tag (e.g. `"whisper"`).
    #[serde(default)]
    pub model_type: Option<String>,
}

fn default_activation_function() -> String {
    "gelu".to_string()
}

impl HfWhisperConfig {
    /// Parse from a JSON string.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the JSON is not
    /// a valid HF Whisper config (missing required fields, malformed).
    pub fn from_json_str(s: &str) -> FerrotorchResult<Self> {
        serde_json::from_str(s).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to parse Whisper config JSON: {e}"),
        })
    }

    /// Parse from a `config.json` file on disk.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the file cannot
    /// be read or its contents are not a valid HF Whisper config.
    pub fn from_file(path: impl AsRef<Path>) -> FerrotorchResult<Self> {
        let path = path.as_ref();
        let bytes = std::fs::read(path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to read config file {}: {e}", path.display()),
        })?;
        let s = std::str::from_utf8(&bytes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("config file {} is not valid UTF-8: {e}", path.display()),
        })?;
        Self::from_json_str(s)
    }

    /// Validate invariants downstream code relies on.
    ///
    /// Checks:
    /// - all counts are positive
    /// - `d_model % encoder_attention_heads == 0`
    /// - `activation_function == "gelu"`
    /// - `num_mel_bins == 80` (the only mel-bin count this crate ships
    ///   a filter bank for)
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] on any failed check.
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.d_model == 0
            || self.encoder_attention_heads == 0
            || self.encoder_layers == 0
            || self.encoder_ffn_dim == 0
            || self.vocab_size == 0
            || self.num_mel_bins == 0
            || self.max_source_positions == 0
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "HfWhisperConfig: d_model / encoder_attention_heads / \
                          encoder_layers / encoder_ffn_dim / vocab_size / \
                          num_mel_bins / max_source_positions must all be positive"
                    .into(),
            });
        }
        if self.d_model % self.encoder_attention_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "HfWhisperConfig: d_model ({}) must be divisible by \
                     encoder_attention_heads ({})",
                    self.d_model, self.encoder_attention_heads
                ),
            });
        }
        if self.activation_function != "gelu" {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "HfWhisperConfig: unsupported activation_function {:?} \
                     (ferrotorch-whisper implements only \"gelu\")",
                    self.activation_function
                ),
            });
        }
        if self.num_mel_bins != 80 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "HfWhisperConfig: unsupported num_mel_bins {} \
                     (ferrotorch-whisper ships only the 80-bin filter bank)",
                    self.num_mel_bins
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TINY_CONFIG: &str = r#"{
        "architectures": ["WhisperForConditionalGeneration"],
        "model_type": "whisper",
        "vocab_size": 51865,
        "num_mel_bins": 80,
        "d_model": 384,
        "encoder_layers": 4,
        "encoder_attention_heads": 6,
        "encoder_ffn_dim": 1536,
        "decoder_layers": 4,
        "decoder_attention_heads": 6,
        "decoder_ffn_dim": 1536,
        "max_source_positions": 1500,
        "max_target_positions": 448,
        "activation_function": "gelu"
    }"#;

    #[test]
    fn parses_whisper_tiny_config() {
        let cfg = HfWhisperConfig::from_json_str(TINY_CONFIG).unwrap();
        assert_eq!(cfg.d_model, 384);
        assert_eq!(cfg.encoder_layers, 4);
        assert_eq!(cfg.encoder_attention_heads, 6);
        assert_eq!(cfg.encoder_ffn_dim, 1_536);
        assert_eq!(cfg.max_source_positions, 1_500);
        assert_eq!(cfg.activation_function, "gelu");
    }

    #[test]
    fn from_hf_round_trips_tiny() {
        let hf = HfWhisperConfig::from_json_str(TINY_CONFIG).unwrap();
        let cfg = WhisperConfig::from_hf(&hf).unwrap();
        assert_eq!(cfg, WhisperConfig::whisper_tiny());
    }

    #[test]
    fn whisper_tiny_is_valid() {
        let cfg = WhisperConfig::whisper_tiny();
        cfg.validate().unwrap();
        assert_eq!(cfg.encoder_head_dim(), 64);
    }

    #[test]
    fn validate_rejects_zero_fields() {
        let mut cfg = WhisperConfig::whisper_tiny();
        cfg.d_model = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_divisible_heads() {
        let mut cfg = WhisperConfig::whisper_tiny();
        cfg.encoder_attention_heads = 5; // 384 % 5 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_unsupported_activation() {
        let json = r#"{
            "vocab_size": 51865, "num_mel_bins": 80, "d_model": 384,
            "encoder_layers": 4, "encoder_attention_heads": 6, "encoder_ffn_dim": 1536,
            "decoder_layers": 4, "decoder_attention_heads": 6, "decoder_ffn_dim": 1536,
            "max_source_positions": 1500, "max_target_positions": 448,
            "activation_function": "relu"
        }"#;
        let hf = HfWhisperConfig::from_json_str(json).unwrap();
        assert!(hf.validate().is_err());
    }

    #[test]
    fn validate_rejects_unsupported_mel_bins() {
        let json = r#"{
            "vocab_size": 51865, "num_mel_bins": 128, "d_model": 384,
            "encoder_layers": 4, "encoder_attention_heads": 6, "encoder_ffn_dim": 1536,
            "decoder_layers": 4, "decoder_attention_heads": 6, "decoder_ffn_dim": 1536,
            "max_source_positions": 1500, "max_target_positions": 448
        }"#;
        let hf = HfWhisperConfig::from_json_str(json).unwrap();
        assert!(hf.validate().is_err());
    }

    #[test]
    fn unknown_fields_ignored() {
        let json = r#"{
            "vocab_size": 51865, "num_mel_bins": 80, "d_model": 384,
            "encoder_layers": 4, "encoder_attention_heads": 6, "encoder_ffn_dim": 1536,
            "decoder_layers": 4, "decoder_attention_heads": 6, "decoder_ffn_dim": 1536,
            "max_source_positions": 1500, "max_target_positions": 448,
            "some_brand_new_field": "ignored"
        }"#;
        let cfg = HfWhisperConfig::from_json_str(json).unwrap();
        cfg.validate().unwrap();
    }
}
