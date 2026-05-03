//! Translate GGUF tensor names into the HuggingFace transformers naming
//! convention used by [`crate::LlamaForCausalLM::load_hf_state_dict`].
//!
//! GGUF (the llama.cpp format) uses short identifiers like `blk.{i}.attn_q.weight`
//! for the same tensors HuggingFace calls `model.layers.{i}.self_attn.q_proj.weight`.
//! Once a GGUF file is dequantized via
//! `ferrotorch_serialize::gguf::load_gguf_state_dict`, the result is a
//! `StateDict<f32>` keyed in the GGUF convention. Rerouting it through
//! [`gguf_to_hf_state_dict`] produces an HF-keyed `StateDict<f32>` ready for
//! the standard model loader, unlocking GGUF-quantized weights as a
//! single-host fit path for 70B-class checkpoints.

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_nn::StateDict;

/// Translate one GGUF tensor name to the equivalent HuggingFace name for a
/// Llama-architecture model.
///
/// Returns `None` if the name does not match any known Llama pattern (caller
/// can decide whether to drop the tensor or pass it through unchanged).
pub fn gguf_key_to_hf(gguf_name: &str) -> Option<String> {
    match gguf_name {
        "token_embd.weight" => Some("model.embed_tokens.weight".to_string()),
        "output_norm.weight" => Some("model.norm.weight".to_string()),
        "output.weight" => Some("lm_head.weight".to_string()),
        other => translate_layer_key(other),
    }
}

fn translate_layer_key(name: &str) -> Option<String> {
    // Names of the form `blk.{i}.<suffix>`; parse `i` and look up the suffix.
    let rest = name.strip_prefix("blk.")?;
    let dot = rest.find('.')?;
    let layer_idx_str = &rest[..dot];
    let suffix = &rest[dot + 1..];
    let layer_idx: usize = layer_idx_str.parse().ok()?;
    let mapped_suffix = match suffix {
        "attn_norm.weight" => "input_layernorm.weight",
        "attn_q.weight" => "self_attn.q_proj.weight",
        "attn_k.weight" => "self_attn.k_proj.weight",
        "attn_v.weight" => "self_attn.v_proj.weight",
        "attn_output.weight" => "self_attn.o_proj.weight",
        "ffn_norm.weight" => "post_attention_layernorm.weight",
        "ffn_gate.weight" => "mlp.gate_proj.weight",
        "ffn_up.weight" => "mlp.up_proj.weight",
        "ffn_down.weight" => "mlp.down_proj.weight",
        _ => return None,
    };
    Some(format!("model.layers.{layer_idx}.{mapped_suffix}"))
}

/// Translate every GGUF-keyed tensor in `state` into HF-keyed equivalents.
///
/// Tensors whose names are not recognised as Llama parameters are dropped,
/// matching `load_hf_state_dict`'s semantics (HF state dicts can carry
/// quantization scales / unrelated metadata that the model does not consume).
/// If `strict` is true, any unrecognised key produces an error instead.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] when `strict` is true
/// and the input contains a key that does not map to a known Llama
/// parameter via [`gguf_key_to_hf`].
pub fn gguf_to_hf_state_dict<T: Float + Clone>(
    state: &StateDict<T>,
    strict: bool,
) -> FerrotorchResult<StateDict<T>> {
    let mut out: StateDict<T> = HashMap::with_capacity(state.len());
    for (gguf_key, tensor) in state {
        match gguf_key_to_hf(gguf_key) {
            Some(hf_key) => {
                out.insert(hf_key, tensor.clone());
            }
            None if strict => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "gguf_to_hf_state_dict: unrecognised tensor name {gguf_key:?}"
                    ),
                });
            }
            None => continue,
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};

    #[test]
    fn embedding_and_norms() {
        assert_eq!(
            gguf_key_to_hf("token_embd.weight").as_deref(),
            Some("model.embed_tokens.weight")
        );
        assert_eq!(
            gguf_key_to_hf("output_norm.weight").as_deref(),
            Some("model.norm.weight")
        );
        assert_eq!(
            gguf_key_to_hf("output.weight").as_deref(),
            Some("lm_head.weight")
        );
    }

    #[test]
    fn per_layer_attention_projections() {
        assert_eq!(
            gguf_key_to_hf("blk.0.attn_q.weight").as_deref(),
            Some("model.layers.0.self_attn.q_proj.weight")
        );
        assert_eq!(
            gguf_key_to_hf("blk.79.attn_k.weight").as_deref(),
            Some("model.layers.79.self_attn.k_proj.weight")
        );
        assert_eq!(
            gguf_key_to_hf("blk.42.attn_v.weight").as_deref(),
            Some("model.layers.42.self_attn.v_proj.weight")
        );
        assert_eq!(
            gguf_key_to_hf("blk.7.attn_output.weight").as_deref(),
            Some("model.layers.7.self_attn.o_proj.weight")
        );
    }

    #[test]
    fn per_layer_norms_and_mlp() {
        assert_eq!(
            gguf_key_to_hf("blk.5.attn_norm.weight").as_deref(),
            Some("model.layers.5.input_layernorm.weight")
        );
        assert_eq!(
            gguf_key_to_hf("blk.5.ffn_norm.weight").as_deref(),
            Some("model.layers.5.post_attention_layernorm.weight")
        );
        assert_eq!(
            gguf_key_to_hf("blk.5.ffn_gate.weight").as_deref(),
            Some("model.layers.5.mlp.gate_proj.weight")
        );
        assert_eq!(
            gguf_key_to_hf("blk.5.ffn_up.weight").as_deref(),
            Some("model.layers.5.mlp.up_proj.weight")
        );
        assert_eq!(
            gguf_key_to_hf("blk.5.ffn_down.weight").as_deref(),
            Some("model.layers.5.mlp.down_proj.weight")
        );
    }

    #[test]
    fn unknown_keys_return_none() {
        assert_eq!(gguf_key_to_hf("rope_freqs.weight"), None);
        assert_eq!(gguf_key_to_hf("blk.0.unknown.weight"), None);
        assert_eq!(gguf_key_to_hf("not_a_recognised_name"), None);
    }

    #[test]
    fn malformed_layer_indices_return_none() {
        assert_eq!(gguf_key_to_hf("blk.x.attn_q.weight"), None);
        assert_eq!(gguf_key_to_hf("blk..attn_q.weight"), None);
    }

    #[test]
    fn state_dict_translation_drops_unknown_by_default() {
        let mut state: StateDict<f32> = HashMap::new();
        let dummy = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0]),
            vec![1],
            /* requires_grad = */ false,
        )
        .unwrap();
        state.insert("token_embd.weight".to_string(), dummy.clone());
        state.insert("blk.0.attn_q.weight".to_string(), dummy.clone());
        state.insert("rope_freqs.weight".to_string(), dummy.clone()); // unknown

        let out = gguf_to_hf_state_dict(&state, /*strict=*/ false).unwrap();
        assert_eq!(out.len(), 2);
        assert!(out.contains_key("model.embed_tokens.weight"));
        assert!(out.contains_key("model.layers.0.self_attn.q_proj.weight"));
        assert!(!out.keys().any(|k| k.contains("rope_freqs")));
    }

    #[test]
    fn strict_mode_rejects_unknown_keys() {
        let mut state: StateDict<f32> = HashMap::new();
        let dummy = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0]),
            vec![1],
            /* requires_grad = */ false,
        )
        .unwrap();
        state.insert("rope_freqs.weight".to_string(), dummy);
        let err = gguf_to_hf_state_dict(&state, /*strict=*/ true).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("rope_freqs"), "{msg}");
    }

    #[test]
    fn full_70b_layer_set_translates_completely() {
        // Build the exact key set a GGUF Llama-3.3-70B file produces (80 layers).
        let mut state: StateDict<f32> = HashMap::new();
        let dummy = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![0.0]),
            vec![1],
            /* requires_grad = */ false,
        )
        .unwrap();
        state.insert("token_embd.weight".into(), dummy.clone());
        state.insert("output_norm.weight".into(), dummy.clone());
        state.insert("output.weight".into(), dummy.clone());
        for i in 0..80 {
            for suffix in [
                "attn_norm.weight",
                "attn_q.weight",
                "attn_k.weight",
                "attn_v.weight",
                "attn_output.weight",
                "ffn_norm.weight",
                "ffn_gate.weight",
                "ffn_up.weight",
                "ffn_down.weight",
            ] {
                state.insert(format!("blk.{i}.{suffix}"), dummy.clone());
            }
        }
        // 3 top-level + 80 × 9 = 723 tensors total.
        assert_eq!(state.len(), 3 + 80 * 9);
        let out = gguf_to_hf_state_dict(&state, /*strict=*/ true).unwrap();
        assert_eq!(out.len(), state.len());
        // Spot-check the 70B-specific layer indices.
        assert!(out.contains_key("model.layers.79.self_attn.q_proj.weight"));
        assert!(out.contains_key("model.layers.79.mlp.down_proj.weight"));
    }
}
