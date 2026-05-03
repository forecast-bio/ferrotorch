//! Typed Llama configuration.
//!
//! [`LlamaConfig`] is the flat, self-contained struct the model layer
//! works against. Construct it directly for synthetic tests, or via
//! [`LlamaConfig::from_hf`] from a parsed HuggingFace `config.json`.

use ferrotorch_core::{FerrotorchError, FerrotorchResult};
use ferrotorch_hub::HfTransformerConfig;

/// FFN activation function.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LlamaActivation {
    /// SiLU / SwiGLU (standard Llama 2 / Llama 3).
    Silu,
    /// FATReLU: `x if x >= threshold else 0` (ProSparse).
    FatRelu(f64),
    /// Plain ReLU (ReluLLaMA).
    Relu,
}

/// Llama model hyperparameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LlamaConfig {
    /// Vocabulary size (token embedding rows).
    pub vocab_size: usize,
    /// Hidden / model dimension (often called `d_model`).
    pub hidden_size: usize,
    /// Feedforward inner dimension.
    pub intermediate_size: usize,
    /// Number of transformer decoder layers.
    pub num_hidden_layers: usize,
    /// Number of attention query heads.
    pub num_attention_heads: usize,
    /// Number of key / value heads (GQA). Must divide `num_attention_heads`.
    pub num_key_value_heads: usize,
    /// Epsilon for RMSNorm.
    pub rms_norm_eps: f64,
    /// RoPE base frequency θ. Llama 3: 500_000.0; Llama 2: 10_000.0.
    pub rope_theta: f64,
    /// Maximum positional context length.
    pub max_position_embeddings: usize,
    /// Whether the input embedding and LM head share weights.
    pub tie_word_embeddings: bool,
    /// FFN activation function. Default: SiLU (SwiGLU).
    pub hidden_act: LlamaActivation,
}

impl LlamaConfig {
    /// The canonical Meta-Llama-3-8B configuration.
    pub fn llama3_8b() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_size: 4096,
            intermediate_size: 14_336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            max_position_embeddings: 8192,
            tie_word_embeddings: false,
            hidden_act: LlamaActivation::Silu,
        }
    }

    /// The canonical Meta-Llama-2-7B configuration.
    pub fn llama2_7b() -> Self {
        Self {
            vocab_size: 32_000,
            hidden_size: 4096,
            intermediate_size: 11_008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 4096,
            tie_word_embeddings: false,
            hidden_act: LlamaActivation::Silu,
        }
    }

    /// SparseLLM/prosparse-llama-2-7b — FATReLU with 89% MLP sparsity.
    pub fn prosparse_7b() -> Self {
        Self {
            hidden_act: LlamaActivation::FatRelu(0.01),
            ..Self::llama2_7b()
        }
    }

    /// The canonical Meta-Llama-3.3-70B-Instruct configuration.
    ///
    /// Architecture parameters match `meta-llama/Llama-3.3-70B-Instruct`'s
    /// HuggingFace `config.json`: 80 hidden layers, 64 attention heads, 8 KV
    /// heads (GQA), hidden size 8192, intermediate size 28 672, RoPE θ
    /// 500 000, vocab 128 256, context length 131 072. Same `tie_word_embeddings = false`
    /// as the 8B variant.
    pub fn llama3_3_70b_instruct() -> Self {
        Self {
            vocab_size: 128_256,
            hidden_size: 8192,
            intermediate_size: 28_672,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            max_position_embeddings: 131_072,
            tie_word_embeddings: false,
            hidden_act: LlamaActivation::Silu,
        }
    }

    /// Build a [`LlamaConfig`] from a parsed HuggingFace `config.json`.
    ///
    /// Validates the config first; downstream construction expects all
    /// invariants (head divisibility, positive sizes) to already hold.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when:
    /// - The HF config fails its own validation (negative or zero
    ///   sizes, head-count divisibility violations).
    /// - `hidden_act` is not one of `"silu"` / `"swish"` / `"relu"` /
    ///   `"fatrelu"`.
    pub fn from_hf(hf: &HfTransformerConfig) -> FerrotorchResult<Self> {
        hf.validate()?;
        let hidden_act = match hf.hidden_act.as_str() {
            "silu" | "swish" => LlamaActivation::Silu,
            "relu" => LlamaActivation::Relu,
            "fatrelu" => {
                let threshold = hf.hidden_act_param.unwrap_or(0.01) as f64;
                LlamaActivation::FatRelu(threshold)
            }
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("LlamaConfig: unsupported hidden_act {other:?}"),
                });
            }
        };
        Ok(Self {
            vocab_size: hf.vocab_size,
            hidden_size: hf.hidden_size,
            intermediate_size: hf.intermediate_size,
            num_hidden_layers: hf.num_hidden_layers,
            num_attention_heads: hf.num_attention_heads,
            num_key_value_heads: hf.num_key_value_heads(),
            rms_norm_eps: hf.rms_norm_eps as f64,
            rope_theta: hf.rope_theta as f64,
            max_position_embeddings: hf.max_position_embeddings,
            tie_word_embeddings: hf.tie_word_embeddings,
            hidden_act,
        })
    }

    /// Per-head embedding dimension: `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Number of query heads served by each KV head (the GQA group size).
    pub fn kv_group_size(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Enforce invariants the model layer relies on. Called by every
    /// `LlamaConfig`-consuming constructor.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when any size is
    /// zero, when `hidden_size % num_attention_heads != 0`, or when
    /// `num_attention_heads % num_key_value_heads != 0`.
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.vocab_size == 0
            || self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.max_position_embeddings == 0
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "LlamaConfig: vocab_size / hidden_size / intermediate_size / \
                          num_hidden_layers / num_attention_heads / num_key_value_heads / \
                          max_position_embeddings must all be positive"
                    .into(),
            });
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LlamaConfig: hidden_size ({}) must be divisible by num_attention_heads ({})",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LlamaConfig: num_attention_heads ({}) must be divisible by \
                     num_key_value_heads ({})",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn llama3_8b_is_valid() {
        let cfg = LlamaConfig::llama3_8b();
        cfg.validate().unwrap();
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.kv_group_size(), 4);
        assert_eq!(cfg.hidden_act, LlamaActivation::Silu);
    }

    #[test]
    fn llama2_7b_is_valid() {
        let cfg = LlamaConfig::llama2_7b();
        cfg.validate().unwrap();
        assert_eq!(cfg.head_dim(), 128);
        assert_eq!(cfg.kv_group_size(), 1); // MHA, not GQA
        assert_eq!(cfg.vocab_size, 32_000);
        assert_eq!(cfg.hidden_act, LlamaActivation::Silu);
    }

    #[test]
    fn prosparse_7b_is_valid() {
        let cfg = LlamaConfig::prosparse_7b();
        cfg.validate().unwrap();
        assert_eq!(cfg.hidden_act, LlamaActivation::FatRelu(0.01));
        assert_eq!(cfg.vocab_size, 32_000);
    }

    #[test]
    fn llama3_3_70b_instruct_is_valid() {
        let cfg = LlamaConfig::llama3_3_70b_instruct();
        cfg.validate().unwrap();
        assert_eq!(cfg.vocab_size, 128_256);
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.intermediate_size, 28_672);
        assert_eq!(cfg.num_hidden_layers, 80);
        assert_eq!(cfg.num_attention_heads, 64);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim(), 128); // 8192 / 64
        assert_eq!(cfg.kv_group_size(), 8); // 64 / 8
        assert_eq!(cfg.rope_theta, 500_000.0);
        assert_eq!(cfg.max_position_embeddings, 131_072);
        assert!(!cfg.tie_word_embeddings);
        assert_eq!(cfg.hidden_act, LlamaActivation::Silu);
    }

    #[test]
    fn from_hf_round_trips_70b_config() {
        // Real Meta-Llama-3.3-70B-Instruct config.json values.
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_act": "silu",
            "hidden_size": 8192,
            "intermediate_size": 28672,
            "max_position_embeddings": 131072,
            "num_attention_heads": 64,
            "num_hidden_layers": 80,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 500000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "vocab_size": 128256
        }"#;
        let hf = HfTransformerConfig::from_json_str(json).unwrap();
        let cfg = LlamaConfig::from_hf(&hf).unwrap();
        let expected = LlamaConfig::llama3_3_70b_instruct();
        assert_eq!(cfg.vocab_size, expected.vocab_size);
        assert_eq!(cfg.hidden_size, expected.hidden_size);
        assert_eq!(cfg.intermediate_size, expected.intermediate_size);
        assert_eq!(cfg.num_hidden_layers, expected.num_hidden_layers);
        assert_eq!(cfg.num_attention_heads, expected.num_attention_heads);
        assert_eq!(cfg.num_key_value_heads, expected.num_key_value_heads);
        assert_eq!(
            cfg.max_position_embeddings,
            expected.max_position_embeddings
        );
        assert_eq!(cfg.tie_word_embeddings, expected.tie_word_embeddings);
        assert!((cfg.rope_theta - expected.rope_theta).abs() < 1e-2);
    }

    #[test]
    fn from_hf_round_trips_llama3_config() {
        let json = r#"{
            "architectures": ["LlamaForCausalLM"],
            "hidden_act": "silu",
            "hidden_size": 4096,
            "intermediate_size": 14336,
            "max_position_embeddings": 8192,
            "num_attention_heads": 32,
            "num_hidden_layers": 32,
            "num_key_value_heads": 8,
            "rms_norm_eps": 1e-05,
            "rope_theta": 500000.0,
            "tie_word_embeddings": false,
            "torch_dtype": "bfloat16",
            "vocab_size": 128256
        }"#;
        let hf = HfTransformerConfig::from_json_str(json).unwrap();
        let cfg = LlamaConfig::from_hf(&hf).unwrap();
        let expected = LlamaConfig::llama3_8b();
        assert_eq!(cfg.vocab_size, expected.vocab_size);
        assert_eq!(cfg.hidden_size, expected.hidden_size);
        assert_eq!(cfg.intermediate_size, expected.intermediate_size);
        assert_eq!(cfg.num_hidden_layers, expected.num_hidden_layers);
        assert_eq!(cfg.num_attention_heads, expected.num_attention_heads);
        assert_eq!(cfg.num_key_value_heads, expected.num_key_value_heads);
        assert_eq!(
            cfg.max_position_embeddings,
            expected.max_position_embeddings
        );
        assert_eq!(cfg.tie_word_embeddings, expected.tie_word_embeddings);
        // HF config stores eps/theta as f32, so expect f32 rounding after
        // widening to f64.
        assert!((cfg.rms_norm_eps - expected.rms_norm_eps).abs() < 1e-7);
        assert!((cfg.rope_theta - expected.rope_theta).abs() < 1e-2);
    }

    #[test]
    fn validate_rejects_zero_fields() {
        let mut cfg = LlamaConfig::llama3_8b();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_divisible_heads() {
        let mut cfg = LlamaConfig::llama3_8b();
        cfg.num_attention_heads = 7; // 4096 % 7 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_non_divisible_kv_heads() {
        let mut cfg = LlamaConfig::llama3_8b();
        cfg.num_key_value_heads = 5; // 32 % 5 != 0
        assert!(cfg.validate().is_err());
    }
}
