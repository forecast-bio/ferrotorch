//! Top-level Llama model + causal-LM head.
//!
//! [`LlamaModel`] is the decoder stack (embedding + N layers + final
//! RMSNorm). [`LlamaForCausalLM`] wraps it with an LM projection head
//! and knows how to ingest HuggingFace-format state dicts.

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Embedding, Linear, RMSNorm};

use crate::config::LlamaConfig;
use crate::layer::LlamaDecoderLayer;

/// Decoder stack: `Embedding` → `N × LlamaDecoderLayer` → final `RMSNorm`.
#[derive(Debug)]
pub struct LlamaModel<T: Float> {
    /// Token embedding layer (vocab → hidden).
    pub embed_tokens: Embedding<T>,
    /// One [`LlamaDecoderLayer`] per `num_hidden_layers` in the config.
    pub layers: Vec<LlamaDecoderLayer<T>>,
    /// Final RMSNorm applied before the LM head.
    pub norm: RMSNorm<T>,
    training: bool,
}

impl<T: Float> LlamaModel<T> {
    /// Build a randomly-initialized decoder stack for the given config.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] if any sub-module
    /// fails to construct (typically a `ShapeMismatch` on bad config
    /// dimensions).
    pub fn new(cfg: &LlamaConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let embed_tokens = Embedding::new(cfg.vocab_size, cfg.hidden_size, None)?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for _ in 0..cfg.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new(cfg)?);
        }
        let norm = RMSNorm::new(vec![cfg.hidden_size], cfg.rms_norm_eps)?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for LlamaModel<T> {
    /// Forward pass.
    ///
    /// The input should be the token-embedding-ready hidden state
    /// (`[1, seq_len, hidden_size]`). Callers that start from token ids
    /// should use [`LlamaForCausalLM::forward_from_ids`] instead.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut h = input.clone();
        for layer in &self.layers {
            h = layer.forward(&h)?;
        }
        self.norm.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.embed_tokens.parameters());
        for l in &self.layers {
            out.extend(l.parameters());
        }
        out.extend(self.norm.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.embed_tokens.parameters_mut());
        for l in &mut self.layers {
            out.extend(l.parameters_mut());
        }
        out.extend(self.norm.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.embed_tokens.named_parameters() {
            out.push((format!("embed_tokens.{n}"), p));
        }
        for (i, l) in self.layers.iter().enumerate() {
            for (n, p) in l.named_parameters() {
                out.push((format!("layers.{i}.{n}"), p));
            }
        }
        for (n, p) in self.norm.named_parameters() {
            out.push((format!("norm.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.embed_tokens.train();
        for l in &mut self.layers {
            l.train();
        }
        self.norm.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.embed_tokens.eval();
        for l in &mut self.layers {
            l.eval();
        }
        self.norm.eval();
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
            for key in state.keys() {
                let recognized = key.starts_with("embed_tokens.")
                    || key.starts_with("norm.")
                    || key.starts_with("layers.");
                if !recognized {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in LlamaModel state_dict: \"{key}\""),
                    });
                }
            }
        }

        self.embed_tokens
            .load_state_dict(&extract("embed_tokens"), strict)?;
        for (i, layer) in self.layers.iter_mut().enumerate() {
            layer.load_state_dict(&extract(&format!("layers.{i}")), strict)?;
        }
        self.norm.load_state_dict(&extract("norm"), strict)?;
        Ok(())
    }
}

/// Llama language-model head: [`LlamaModel`] + `lm_head: Linear`.
#[derive(Debug)]
pub struct LlamaForCausalLM<T: Float> {
    /// Underlying decoder stack (embeddings + layers + final norm).
    pub model: LlamaModel<T>,
    /// Vocabulary projection: `[hidden] → [vocab_size]`. No bias for
    /// any Llama variant.
    pub lm_head: Linear<T>,
    /// Frozen copy of the configuration used to construct this model.
    pub config: LlamaConfig,
    training: bool,
}

impl<T: Float> LlamaForCausalLM<T> {
    /// Build a randomly-initialized Llama model for the given config.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] if config validation
    /// fails or any sub-module ([`LlamaModel`] / `Linear` `lm_head`)
    /// fails to construct.
    pub fn new(cfg: LlamaConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        let model = LlamaModel::new(&cfg)?;
        // The LM head has bias=false for every Llama variant.
        let lm_head = Linear::new(cfg.hidden_size, cfg.vocab_size, false)?;
        Ok(Self {
            model,
            lm_head,
            config: cfg,
            training: false,
        })
    }

    /// Forward pass from token ids `[1, seq_len]` (u32) to logits
    /// `[1, seq_len, vocab_size]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when `ids` is
    /// empty, or when a token id is not representable as the model's
    /// element type `T` (only happens for absurdly large vocabularies
    /// against narrow float types — e.g. a vocab_size > 65 504 with
    /// `T = bf16`). Otherwise propagates whatever the embedding /
    /// decoder-stack / lm-head forward passes return (e.g.
    /// `ShapeMismatch` if the upstream tensor construction rejects
    /// the seq-length-derived shape).
    ///
    /// # Panics
    ///
    /// Does not panic. Out-of-range token ids and other
    /// non-representable conversions are returned as
    /// [`FerrotorchError::InvalidArgument`] via
    /// [`ferrotorch_core::numeric_cast::cast`].
    pub fn forward_from_ids(&self, ids: &[u32]) -> FerrotorchResult<Tensor<T>> {
        if ids.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "LlamaForCausalLM::forward_from_ids needs at least one token".into(),
            });
        }
        // Embedding.forward takes a 1-D tensor of float-encoded indices.
        // Direct u32 -> T cast: skips the historical f64 round-trip; the
        // helper returns Err for any token id not representable in T.
        let idx_data: Vec<T> = ids
            .iter()
            .map(|&i| ferrotorch_core::numeric_cast::cast::<u32, T>(i))
            .collect::<FerrotorchResult<Vec<T>>>()?;
        let seq_len = ids.len();
        let hidden = self.config.hidden_size;
        let idx_tensor = ferrotorch_core::Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(idx_data),
            vec![seq_len],
            false,
        )?;
        let embeds_2d = self.model.embed_tokens.forward(&idx_tensor)?;
        // embeds_2d is [S, hidden]; promote to [1, S, hidden].
        let data = embeds_2d.data_vec()?;
        let hidden_3d = ferrotorch_core::Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(data),
            vec![1, seq_len, hidden],
            false,
        )?;
        let h = self.model.forward(&hidden_3d)?;
        self.lm_head.forward(&h)
    }

    /// Load a HuggingFace-format `StateDict` by rewriting keys to match
    /// our parameter paths, then delegating to [`load_state_dict`].
    ///
    /// HF → ferrotorch key mapping:
    ///
    /// - `model.embed_tokens.weight`            → `model.embed_tokens.weight`
    /// - `model.layers.{i}.input_layernorm.weight`            → `model.layers.{i}.input_layernorm.weight`
    /// - `model.layers.{i}.post_attention_layernorm.weight`   → `model.layers.{i}.post_attention_layernorm.weight`
    /// - `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`   → `model.layers.{i}.self_attn.{q,k,v,o}_proj.weight`
    /// - `model.layers.{i}.mlp.{gate,up,down}_proj.weight`    → `model.layers.{i}.mlp.{gate,up,down}_proj.weight`
    /// - `model.norm.weight`                    → `model.norm.weight`
    /// - `lm_head.weight`                       → `lm_head.weight`
    ///
    /// The HF layout maps onto ours byte-for-byte; the wrapper exists
    /// so future renames (e.g. dropping the `model.` prefix) stay
    /// localised. When `tie_word_embeddings` is set in the config,
    /// `lm_head.weight` is copied from `model.embed_tokens.weight` if
    /// absent from the state dict.
    ///
    /// # Errors
    ///
    /// Forwards whatever each sub-module's `load_state_dict` returns
    /// — typically [`FerrotorchError::ShapeMismatch`] if a checkpoint
    /// tensor has the wrong shape, or [`FerrotorchError::InvalidArgument`]
    /// in `strict` mode when a required tensor is missing.
    pub fn load_hf_state_dict(
        &mut self,
        hf_state: &StateDict<T>,
        strict: bool,
    ) -> FerrotorchResult<()> {
        let mut remapped: StateDict<T> = HashMap::with_capacity(hf_state.len());
        for (k, v) in hf_state {
            remapped.insert(k.clone(), v.clone());
        }
        if self.config.tie_word_embeddings && !remapped.contains_key("lm_head.weight") {
            if let Some(embed) = remapped.get("model.embed_tokens.weight").cloned() {
                remapped.insert("lm_head.weight".to_string(), embed);
            }
        }
        self.load_state_dict(&remapped, strict)
    }
}

impl<T: Float> Module<T> for LlamaForCausalLM<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h = self.model.forward(input)?;
        self.lm_head.forward(&h)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.model.parameters());
        out.extend(self.lm_head.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.model.parameters_mut());
        out.extend(self.lm_head.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.model.named_parameters() {
            out.push((format!("model.{n}"), p));
        }
        for (n, p) in self.lm_head.named_parameters() {
            out.push((format!("lm_head.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.model.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.model.eval();
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
            for key in state.keys() {
                if !(key.starts_with("model.") || key.starts_with("lm_head.")) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in LlamaForCausalLM state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }

        self.model.load_state_dict(&extract("model"), strict)?;
        self.lm_head.load_state_dict(&extract("lm_head"), strict)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny Llama config suitable for round-trip unit tests.
    fn tiny_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            tie_word_embeddings: false,
            hidden_act: crate::config::LlamaActivation::Silu,
        }
    }

    #[test]
    fn tiny_model_constructs_and_parameter_count_sane() {
        let model = LlamaForCausalLM::<f32>::new(tiny_config()).unwrap();
        let params = model.parameters();
        // embed (32*16) + 2*(attn + mlp + 2 norms) + norm(16) + lm_head(32*16)
        //   attn params per layer:
        //     q_proj: 16*16
        //     k_proj: 8*16
        //     v_proj: 8*16
        //     o_proj: 16*16
        //   mlp params per layer:
        //     gate_proj: 32*16
        //     up_proj:   32*16
        //     down_proj: 16*32
        let total_params: usize = params.iter().map(|p| p.numel()).sum();
        let embed = 32 * 16;
        let attn_per = 16 * 16 + 8 * 16 + 8 * 16 + 16 * 16;
        let mlp_per = 32 * 16 * 3;
        let norms_per_layer = 16 * 2;
        let final_norm = 16;
        let lm_head = 32 * 16;
        let expected = embed + 2 * (attn_per + mlp_per + norms_per_layer) + final_norm + lm_head;
        assert_eq!(total_params, expected, "param count mismatch");
    }

    #[test]
    fn tiny_model_named_parameters_use_hf_layout() {
        let model = LlamaForCausalLM::<f32>::new(tiny_config()).unwrap();
        let named: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(named.contains(&"model.embed_tokens.weight".to_string()));
        assert!(named.contains(&"model.norm.weight".to_string()));
        assert!(named.contains(&"lm_head.weight".to_string()));
        assert!(named.contains(&"model.layers.0.input_layernorm.weight".to_string()));
        assert!(named.contains(&"model.layers.0.post_attention_layernorm.weight".to_string()));
        assert!(named.contains(&"model.layers.0.self_attn.q_proj.weight".to_string()));
        assert!(named.contains(&"model.layers.0.self_attn.k_proj.weight".to_string()));
        assert!(named.contains(&"model.layers.0.self_attn.v_proj.weight".to_string()));
        assert!(named.contains(&"model.layers.0.self_attn.o_proj.weight".to_string()));
        assert!(named.contains(&"model.layers.0.mlp.gate_proj.weight".to_string()));
        assert!(named.contains(&"model.layers.0.mlp.up_proj.weight".to_string()));
        assert!(named.contains(&"model.layers.0.mlp.down_proj.weight".to_string()));
        assert!(named.contains(&"model.layers.1.self_attn.q_proj.weight".to_string()));
    }

    #[test]
    fn tiny_model_forward_from_ids_produces_correct_shape() {
        let model = LlamaForCausalLM::<f32>::new(tiny_config()).unwrap();
        let ids = vec![1u32, 5, 7, 9];
        let logits = model.forward_from_ids(&ids).unwrap();
        // [1, seq_len, vocab_size]
        assert_eq!(logits.shape(), &[1, 4, 32]);
        // All logits must be finite.
        for &v in logits.data().unwrap() {
            assert!(v.is_finite(), "logit non-finite: {v}");
        }
    }

    #[test]
    fn load_state_dict_round_trip_tiny() {
        let src = LlamaForCausalLM::<f32>::new(tiny_config()).unwrap();
        let sd = src.state_dict();
        let mut dst = LlamaForCausalLM::<f32>::new(tiny_config()).unwrap();
        dst.load_state_dict(&sd, true).unwrap();
        // Loaded model should produce identical logits on the same input.
        let ids = vec![2u32, 4, 6];
        let a = src.forward_from_ids(&ids).unwrap();
        let b = dst.forward_from_ids(&ids).unwrap();
        for (x, y) in a.data().unwrap().iter().zip(b.data().unwrap().iter()) {
            assert!((x - y).abs() < 1e-6, "round-trip logits differ: {x} vs {y}");
        }
    }

    #[test]
    fn load_state_dict_strict_rejects_unknown_key() {
        let mut model = LlamaForCausalLM::<f32>::new(tiny_config()).unwrap();
        let mut sd = model.state_dict();
        sd.insert(
            "mystery.key".to_string(),
            ferrotorch_core::zeros::<f32>(&[1]).unwrap(),
        );
        assert!(model.load_state_dict(&sd, true).is_err());
    }

    #[test]
    fn load_hf_state_dict_with_tied_embeddings_copies_lm_head() {
        let mut cfg = tiny_config();
        cfg.tie_word_embeddings = true;
        let src = LlamaForCausalLM::<f32>::new(cfg).unwrap();
        let mut sd = src.state_dict();
        // Simulate an HF tied-embedding export: remove lm_head.weight.
        sd.remove("lm_head.weight");
        let mut dst = LlamaForCausalLM::<f32>::new(cfg).unwrap();
        dst.load_hf_state_dict(&sd, true).unwrap();
        // lm_head should now equal embed_tokens.
        let lm_head = dst.lm_head.parameters()[0]
            .tensor()
            .data()
            .unwrap()
            .to_vec();
        let embed = dst.model.embed_tokens.parameters()[0]
            .tensor()
            .data()
            .unwrap()
            .to_vec();
        assert_eq!(lm_head, embed);
    }
}
