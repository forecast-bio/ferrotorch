//! HuggingFace transformer `config.json` parser.
//!
//! HuggingFace models publish their architecture hyperparameters in a
//! `config.json` file alongside the weight shards (Llama 3:
//! `meta-llama/Meta-Llama-3-8B/config.json`). This module deserializes
//! that file into a flat [`HfTransformerConfig`] — the union of fields
//! needed by decoder-only transformers (Llama, Mistral, Gemma, Qwen,
//! Falcon, etc.).
//!
//! Downstream code maps this flat struct to model-specific configs
//! (e.g. `LlamaConfig` in `ferrotorch-nn`) rather than building a full
//! HuggingFace `AutoConfig`-style dispatch layer.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrotorch_hub::HfTransformerConfig;
//!
//! let cfg = HfTransformerConfig::from_file(
//!     "~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/.../config.json",
//! ).unwrap();
//! assert_eq!(cfg.num_attention_heads, 32);
//! assert_eq!(cfg.num_key_value_heads(), 8); // GQA
//! ```

use std::path::Path;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};
use serde::Deserialize;

/// Parsed contents of a HuggingFace transformer `config.json`.
///
/// This is the union of fields needed by contemporary decoder-only
/// transformers. Model-specific configs (`LlamaConfig`, `MistralConfig`,
/// …) should construct themselves from an [`HfTransformerConfig`] and
/// enforce any additional invariants they require.
///
/// Unknown fields in the JSON are silently ignored so a config emitted
/// by a newer HF version still parses. Missing optional fields fall
/// back to the defaults documented on each accessor.
///
/// Marked `#[non_exhaustive]` so HuggingFace's evolving config schema
/// can pick up new fields (rope-scaling parameter shapes, attention
/// implementation tags, etc.) in a minor version without breaking
/// external code. External callers should construct via
/// [`Self::from_json_str`] / [`Self::from_file`] rather than struct
/// literal. A workspace-level grep for `HfTransformerConfig {` outside
/// `ferrotorch-hub/` returned zero hits at audit time.
#[derive(Debug, Clone, Deserialize)]
#[non_exhaustive]
pub struct HfTransformerConfig {
    /// Embedding / hidden dimension (`d_model`).
    pub hidden_size: usize,

    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,

    /// Number of attention (query) heads per layer.
    pub num_attention_heads: usize,

    /// Number of key/value heads (GQA). Missing means classical MHA —
    /// use [`num_key_value_heads`](Self::num_key_value_heads) for the
    /// resolved value.
    #[serde(default)]
    pub num_key_value_heads: Option<usize>,

    /// FFN inner dimension (also known as `d_ff`).
    pub intermediate_size: usize,

    /// Epsilon for RMSNorm / LayerNorm. HF default for Llama is `1e-6`;
    /// Llama 3 publishes `1e-5`. Default to `1e-6` when the field is
    /// absent (matches `transformers` library behaviour).
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f32,

    /// RoPE base frequency θ. Original RoPE uses `10000.0`; Llama 3
    /// uses `500000.0` for the extended context window.
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f32,

    /// Maximum positional context length the model was trained with.
    pub max_position_embeddings: usize,

    /// Vocabulary size (size of the token embedding table).
    pub vocab_size: usize,

    /// Whether the input embedding and the LM head share weights.
    /// Llama 3 sets this to `false`.
    #[serde(default)]
    pub tie_word_embeddings: bool,

    /// Activation function name inside the FFN. Llama family uses
    /// `"silu"` which composes with SwiGLU. ProSparse uses `"fatrelu"`.
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,

    /// Activation parameter for parameterised activations (e.g. FATReLU
    /// threshold). `None` for standard activations.
    #[serde(default)]
    pub hidden_act_param: Option<f32>,

    /// The dtype the weights were saved in. Llama 3 8B: `"bfloat16"`.
    /// May be absent on older configs.
    #[serde(default)]
    pub torch_dtype: Option<String>,

    /// Model architectures declared in the config (e.g.
    /// `["LlamaForCausalLM"]`). The loader uses this to pick the
    /// correct model-specific config.
    #[serde(default)]
    pub architectures: Vec<String>,

    /// Optional RoPE scaling config (NTK / linear / dynamic). Opaque
    /// JSON — model-specific wrappers parse the structure they need.
    #[serde(default)]
    pub rope_scaling: Option<serde_json::Value>,

    /// Model type tag (e.g. `"llama"`). Some loaders use this before
    /// `architectures` is populated.
    #[serde(default)]
    pub model_type: Option<String>,
}

fn default_rms_norm_eps() -> f32 {
    1e-6
}

fn default_rope_theta() -> f32 {
    10_000.0
}

fn default_hidden_act() -> String {
    "silu".to_string()
}

impl HfTransformerConfig {
    /// Parse from a JSON string.
    pub fn from_json_str(s: &str) -> FerrotorchResult<Self> {
        serde_json::from_str(s).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to parse HF config JSON: {e}"),
        })
    }

    /// Parse from a `config.json` file on disk.
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

    /// Number of key/value heads, with the MHA default
    /// (`num_key_value_heads = num_attention_heads`) applied when the
    /// JSON did not specify one.
    pub fn num_key_value_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Per-head embedding dimension: `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Whether the config describes a GQA model (i.e. the number of KV
    /// heads is strictly less than the number of Q heads).
    pub fn is_gqa(&self) -> bool {
        self.num_key_value_heads() < self.num_attention_heads
    }

    /// Validate invariants that downstream code relies on. Called
    /// before constructing model-specific configs.
    ///
    /// Checks:
    /// - `hidden_size % num_attention_heads == 0`
    /// - `num_attention_heads % num_key_value_heads == 0`
    /// - all counts are positive
    /// - `hidden_act` matches one of the supported activations
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.hidden_size == 0
            || self.num_attention_heads == 0
            || self.num_hidden_layers == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "HfTransformerConfig: hidden_size / num_attention_heads / \
                          num_hidden_layers / intermediate_size / vocab_size must all be positive"
                    .into(),
            });
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "HfTransformerConfig: hidden_size ({}) must be divisible by \
                     num_attention_heads ({})",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        let kv = self.num_key_value_heads();
        if kv == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "HfTransformerConfig: num_key_value_heads must be positive".into(),
            });
        }
        if self.num_attention_heads % kv != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "HfTransformerConfig: num_attention_heads ({}) must be divisible \
                     by num_key_value_heads ({})",
                    self.num_attention_heads, kv
                ),
            });
        }
        match self.hidden_act.as_str() {
            "silu" | "swish" | "gelu" | "relu" | "gelu_new" | "fatrelu" => Ok(()),
            other => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "HfTransformerConfig: unsupported hidden_act {other:?} \
                     (supported: silu, swish, gelu, gelu_new, relu, fatrelu)"
                ),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The exact `config.json` published for `meta-llama/Meta-Llama-3-8B`.
    const LLAMA_3_8B_CONFIG: &str = r#"{
        "architectures": ["LlamaForCausalLM"],
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 128000,
        "eos_token_id": 128001,
        "hidden_act": "silu",
        "hidden_size": 4096,
        "initializer_range": 0.02,
        "intermediate_size": 14336,
        "max_position_embeddings": 8192,
        "model_type": "llama",
        "num_attention_heads": 32,
        "num_hidden_layers": 32,
        "num_key_value_heads": 8,
        "pretraining_tp": 1,
        "rms_norm_eps": 1e-05,
        "rope_scaling": null,
        "rope_theta": 500000.0,
        "tie_word_embeddings": false,
        "torch_dtype": "bfloat16",
        "transformers_version": "4.40.0.dev0",
        "use_cache": true,
        "vocab_size": 128256
    }"#;

    #[test]
    fn parses_llama_3_8b_config() {
        let cfg = HfTransformerConfig::from_json_str(LLAMA_3_8B_CONFIG).unwrap();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads(), 8);
        assert_eq!(cfg.intermediate_size, 14336);
        assert!((cfg.rms_norm_eps - 1e-5).abs() < 1e-9);
        assert!((cfg.rope_theta - 500_000.0).abs() < 1e-3);
        assert_eq!(cfg.max_position_embeddings, 8192);
        assert_eq!(cfg.vocab_size, 128_256);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.hidden_act, "silu");
        assert_eq!(cfg.torch_dtype.as_deref(), Some("bfloat16"));
        assert_eq!(cfg.architectures, vec!["LlamaForCausalLM".to_string()]);
        assert_eq!(cfg.model_type.as_deref(), Some("llama"));
    }

    #[test]
    fn derived_values_for_llama_3_8b() {
        let cfg = HfTransformerConfig::from_json_str(LLAMA_3_8B_CONFIG).unwrap();
        assert_eq!(cfg.head_dim(), 128); // 4096 / 32
        assert!(cfg.is_gqa()); // 8 < 32
    }

    #[test]
    fn validate_passes_on_llama_3_8b() {
        HfTransformerConfig::from_json_str(LLAMA_3_8B_CONFIG)
            .unwrap()
            .validate()
            .unwrap();
    }

    #[test]
    fn num_kv_heads_defaults_to_num_heads_when_absent() {
        // Classical MHA config — no num_key_value_heads.
        let json = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "vocab_size": 1000
        }"#;
        let cfg = HfTransformerConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.num_key_value_heads(), 8); // falls back to num_attention_heads
        assert!(!cfg.is_gqa());
    }

    #[test]
    fn defaults_applied_when_fields_absent() {
        let json = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "vocab_size": 1000
        }"#;
        let cfg = HfTransformerConfig::from_json_str(json).unwrap();
        assert!((cfg.rms_norm_eps - 1e-6).abs() < 1e-9);
        assert!((cfg.rope_theta - 10_000.0).abs() < 1e-3);
        assert_eq!(cfg.hidden_act, "silu");
        assert!(!cfg.tie_word_embeddings);
        assert!(cfg.architectures.is_empty());
        assert!(cfg.torch_dtype.is_none());
        assert!(cfg.rope_scaling.is_none());
    }

    #[test]
    fn unknown_fields_are_ignored() {
        // HF regularly adds new fields; older code must still parse.
        let json = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "vocab_size": 1000,
            "some_new_field_from_future_hf": "ignored",
            "another_one": {"nested": true}
        }"#;
        let cfg = HfTransformerConfig::from_json_str(json).unwrap();
        assert_eq!(cfg.hidden_size, 64);
    }

    #[test]
    fn validate_rejects_non_divisible_heads() {
        // hidden_size not divisible by num_attention_heads.
        let json = r#"{
            "hidden_size": 65,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "vocab_size": 1000
        }"#;
        let cfg = HfTransformerConfig::from_json_str(json).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_divisible_kv_heads() {
        let json = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "num_key_value_heads": 3,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "vocab_size": 1000
        }"#;
        let cfg = HfTransformerConfig::from_json_str(json).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_zero_counts() {
        let json = r#"{
            "hidden_size": 0,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "vocab_size": 1000
        }"#;
        let cfg = HfTransformerConfig::from_json_str(json).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_unsupported_activation() {
        let json = r#"{
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "vocab_size": 1000,
            "hidden_act": "tanh_xyz"
        }"#;
        let cfg = HfTransformerConfig::from_json_str(json).unwrap();
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn from_file_reads_config_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("config.json");
        std::fs::write(&path, LLAMA_3_8B_CONFIG).unwrap();
        let cfg = HfTransformerConfig::from_file(&path).unwrap();
        assert_eq!(cfg.num_attention_heads, 32);
    }

    #[test]
    fn from_file_reports_missing_file() {
        let r = HfTransformerConfig::from_file("/nonexistent/config.json");
        assert!(r.is_err());
    }

    #[test]
    fn from_json_str_reports_bad_json() {
        let r = HfTransformerConfig::from_json_str("{ not valid json");
        assert!(r.is_err());
    }

    #[test]
    fn from_json_str_reports_missing_required_field() {
        // `hidden_size` is required.
        let json = r#"{
            "num_hidden_layers": 2,
            "num_attention_heads": 8,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "vocab_size": 1000
        }"#;
        assert!(HfTransformerConfig::from_json_str(json).is_err());
    }
}
