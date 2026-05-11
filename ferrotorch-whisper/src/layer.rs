//! Single Whisper encoder layer (pre-norm).
//!
//! ```text
//! attn_out  = input + self_attn(self_attn_layer_norm(input))   // PRE-NORM residual
//! layer_out = attn_out + fc2(GELU(fc1(final_layer_norm(attn_out))))
//! ```

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{GELU, LayerNorm, Linear};

use crate::attention::WhisperEncoderSelfAttention;
use crate::config::WhisperConfig;

/// HF `WhisperEncoderLayer` — one full encoder block, pre-norm residual.
#[derive(Debug)]
pub struct WhisperEncoderLayer<T: Float> {
    /// LayerNorm applied to the residual input BEFORE self-attention.
    pub self_attn_layer_norm: LayerNorm<T>,
    /// Self-attention sub-block (Q, K (no bias), V, out_proj).
    pub self_attn: WhisperEncoderSelfAttention<T>,
    /// LayerNorm applied to the post-attention residual BEFORE the FFN.
    pub final_layer_norm: LayerNorm<T>,
    /// FFN expansion projection: `d_model -> encoder_ffn_dim`, bias=True.
    pub fc1: Linear<T>,
    /// FFN reduction projection: `encoder_ffn_dim -> d_model`, bias=True.
    pub fc2: Linear<T>,
    activation: GELU,
    training: bool,
}

impl<T: Float> WhisperEncoderLayer<T> {
    /// Build a randomly-initialized encoder layer.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &WhisperConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        // HF Whisper uses `layer_norm_eps = 1e-5` (the PyTorch default,
        // not BERT's 1e-12). Hard-coded here because Whisper's
        // `config.json` does NOT carry the field.
        let eps = 1e-5_f64;
        Ok(Self {
            self_attn_layer_norm: LayerNorm::new(vec![cfg.d_model], eps, true)?,
            self_attn: WhisperEncoderSelfAttention::new(cfg)?,
            final_layer_norm: LayerNorm::new(vec![cfg.d_model], eps, true)?,
            fc1: Linear::new(cfg.d_model, cfg.encoder_ffn_dim, true)?,
            fc2: Linear::new(cfg.encoder_ffn_dim, cfg.d_model, true)?,
            activation: GELU::new(),
            training: false,
        })
    }
}

impl<T: Float> Module<T> for WhisperEncoderLayer<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // -- Self-attention residual (pre-norm). -----------------------------
        let normed = self.self_attn_layer_norm.forward(input)?;
        let attn = self.self_attn.forward(&normed)?;
        let attn_residual = add(input, &attn)?;

        // -- FFN residual (pre-norm). ---------------------------------------
        let normed_ffn = self.final_layer_norm.forward(&attn_residual)?;
        let fc1_out = self.fc1.forward(&normed_ffn)?;
        let activated = self.activation.forward(&fc1_out)?;
        let fc2_out = self.fc2.forward(&activated)?;
        add(&attn_residual, &fc2_out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.self_attn_layer_norm.parameters());
        out.extend(self.self_attn.parameters());
        out.extend(self.final_layer_norm.parameters());
        out.extend(self.fc1.parameters());
        out.extend(self.fc2.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.self_attn_layer_norm.parameters_mut());
        out.extend(self.self_attn.parameters_mut());
        out.extend(self.final_layer_norm.parameters_mut());
        out.extend(self.fc1.parameters_mut());
        out.extend(self.fc2.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.self_attn_layer_norm.named_parameters() {
            out.push((format!("self_attn_layer_norm.{n}"), p));
        }
        for (n, p) in self.self_attn.named_parameters() {
            out.push((format!("self_attn.{n}"), p));
        }
        for (n, p) in self.final_layer_norm.named_parameters() {
            out.push((format!("final_layer_norm.{n}"), p));
        }
        for (n, p) in self.fc1.named_parameters() {
            out.push((format!("fc1.{n}"), p));
        }
        for (n, p) in self.fc2.named_parameters() {
            out.push((format!("fc2.{n}"), p));
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
            let prefixes = [
                "self_attn_layer_norm",
                "self_attn",
                "final_layer_norm",
                "fc1",
                "fc2",
            ];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in WhisperEncoderLayer state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }
        self.self_attn_layer_norm
            .load_state_dict(&extract("self_attn_layer_norm"), strict)?;
        self.self_attn
            .load_state_dict(&extract("self_attn"), strict)?;
        self.final_layer_norm
            .load_state_dict(&extract("final_layer_norm"), strict)?;
        self.fc1.load_state_dict(&extract("fc1"), strict)?;
        self.fc2.load_state_dict(&extract("fc2"), strict)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    fn tiny_cfg() -> WhisperConfig {
        WhisperConfig {
            vocab_size: 32,
            num_mel_bins: 80,
            d_model: 16,
            encoder_layers: 1,
            encoder_attention_heads: 2,
            encoder_ffn_dim: 32,
            decoder_layers: 1,
            decoder_attention_heads: 2,
            decoder_ffn_dim: 32,
            max_source_positions: 8,
            max_target_positions: 8,
        }
    }

    #[test]
    fn layer_forward_shape() {
        let layer = WhisperEncoderLayer::<f32>::new(&tiny_cfg()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1f32; 5 * 16]),
            vec![1, 5, 16],
            false,
        )
        .unwrap();
        let out = layer.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 5, 16]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite(), "layer output non-finite: {v}");
        }
    }

    #[test]
    fn named_parameters_match_hf_layout() {
        let layer = WhisperEncoderLayer::<f32>::new(&tiny_cfg()).unwrap();
        let names: Vec<String> = layer
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for k in [
            "self_attn_layer_norm.weight",
            "self_attn_layer_norm.bias",
            "self_attn.q_proj.weight",
            "self_attn.q_proj.bias",
            "self_attn.k_proj.weight",
            "self_attn.v_proj.weight",
            "self_attn.v_proj.bias",
            "self_attn.out_proj.weight",
            "self_attn.out_proj.bias",
            "final_layer_norm.weight",
            "final_layer_norm.bias",
            "fc1.weight",
            "fc1.bias",
            "fc2.weight",
            "fc2.bias",
        ] {
            assert!(
                names.iter().any(|n| n == k),
                "missing parameter key {k:?} in {names:?}"
            );
        }
        // k_proj has NO bias — HF Whisper convention.
        assert!(!names.iter().any(|n| n == "self_attn.k_proj.bias"));
    }
}
