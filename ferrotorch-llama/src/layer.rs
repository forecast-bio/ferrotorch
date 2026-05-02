//! Single Llama decoder layer.
//!
//! Pre-norm residual block matching the HuggingFace reference:
//!
//! ```text
//! x = x + self_attn(input_layernorm(x))
//! x = x + mlp(post_attention_layernorm(x))
//! ```

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::RMSNorm;
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;

use crate::attention::LlamaAttention;
use crate::config::LlamaConfig;
use crate::mlp::LlamaMLP;

/// One decoder layer: RMSNorm → attention → residual → RMSNorm → MLP → residual.
pub struct LlamaDecoderLayer<T: Float> {
    pub input_layernorm: RMSNorm<T>,
    pub self_attn: LlamaAttention<T>,
    pub post_attention_layernorm: RMSNorm<T>,
    pub mlp: LlamaMLP<T>,
    training: bool,
}

impl<T: Float> LlamaDecoderLayer<T> {
    pub fn new(cfg: &LlamaConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            input_layernorm: RMSNorm::new(vec![cfg.hidden_size], cfg.rms_norm_eps)?,
            self_attn: LlamaAttention::new(cfg)?,
            post_attention_layernorm: RMSNorm::new(vec![cfg.hidden_size], cfg.rms_norm_eps)?,
            mlp: LlamaMLP::new(cfg)?,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for LlamaDecoderLayer<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Attention sub-block.
        let h = self.input_layernorm.forward(input)?;
        let attn_out = self.self_attn.forward(&h)?;
        let x = add(input, &attn_out)?;

        // MLP sub-block.
        let h2 = self.post_attention_layernorm.forward(&x)?;
        let mlp_out = self.mlp.forward(&h2)?;
        add(&x, &mlp_out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.input_layernorm.parameters());
        out.extend(self.self_attn.parameters());
        out.extend(self.post_attention_layernorm.parameters());
        out.extend(self.mlp.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.input_layernorm.parameters_mut());
        out.extend(self.self_attn.parameters_mut());
        out.extend(self.post_attention_layernorm.parameters_mut());
        out.extend(self.mlp.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.input_layernorm.named_parameters() {
            out.push((format!("input_layernorm.{n}"), p));
        }
        for (n, p) in self.self_attn.named_parameters() {
            out.push((format!("self_attn.{n}"), p));
        }
        for (n, p) in self.post_attention_layernorm.named_parameters() {
            out.push((format!("post_attention_layernorm.{n}"), p));
        }
        for (n, p) in self.mlp.named_parameters() {
            out.push((format!("mlp.{n}"), p));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.input_layernorm.train();
        self.self_attn.train();
        self.post_attention_layernorm.train();
        self.mlp.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.input_layernorm.eval();
        self.self_attn.eval();
        self.post_attention_layernorm.eval();
        self.mlp.eval();
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
                "input_layernorm",
                "self_attn",
                "post_attention_layernorm",
                "mlp",
            ];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in LlamaDecoderLayer state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }

        self.input_layernorm
            .load_state_dict(&extract("input_layernorm"), strict)?;
        self.self_attn
            .load_state_dict(&extract("self_attn"), strict)?;
        self.post_attention_layernorm
            .load_state_dict(&extract("post_attention_layernorm"), strict)?;
        self.mlp.load_state_dict(&extract("mlp"), strict)?;
        Ok(())
    }
}
