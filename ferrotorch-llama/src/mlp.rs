//! Llama feedforward block.
//!
//! Equivalent to `ferrotorch_nn::SwiGLU` but exposes the three
//! projections as separate named fields (`gate_proj`, `up_proj`,
//! `down_proj`) so HuggingFace weight names can be mapped directly
//! onto them.

use ferrotorch_core::grad_fns::activation::silu;
use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Linear;
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;

use crate::config::LlamaConfig;

/// Llama SwiGLU feedforward: `down(silu(gate(x)) * up(x))`.
pub struct LlamaMLP<T: Float> {
    pub gate_proj: Linear<T>,
    pub up_proj: Linear<T>,
    pub down_proj: Linear<T>,
    training: bool,
}

impl<T: Float> LlamaMLP<T> {
    pub fn new(cfg: &LlamaConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            gate_proj: Linear::new(cfg.hidden_size, cfg.intermediate_size, false)?,
            up_proj: Linear::new(cfg.hidden_size, cfg.intermediate_size, false)?,
            down_proj: Linear::new(cfg.intermediate_size, cfg.hidden_size, false)?,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for LlamaMLP<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let gate = self.gate_proj.forward(input)?;
        let up = self.up_proj.forward(input)?;
        let activated = silu(&gate)?;
        let gated = mul(&activated, &up)?;
        self.down_proj.forward(&gated)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.gate_proj.parameters());
        out.extend(self.up_proj.parameters());
        out.extend(self.down_proj.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.gate_proj.parameters_mut());
        out.extend(self.up_proj.parameters_mut());
        out.extend(self.down_proj.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.gate_proj.named_parameters() {
            out.push((format!("gate_proj.{n}"), p));
        }
        for (n, p) in self.up_proj.named_parameters() {
            out.push((format!("up_proj.{n}"), p));
        }
        for (n, p) in self.down_proj.named_parameters() {
            out.push((format!("down_proj.{n}"), p));
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

    fn load_state_dict(
        &mut self,
        state: &StateDict<T>,
        strict: bool,
    ) -> FerrotorchResult<()> {
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
            let prefixes = ["gate_proj", "up_proj", "down_proj"];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in LlamaMLP state_dict: \"{key}\""),
                    });
                }
            }
        }

        self.gate_proj
            .load_state_dict(&extract("gate_proj"), strict)?;
        self.up_proj.load_state_dict(&extract("up_proj"), strict)?;
        self.down_proj
            .load_state_dict(&extract("down_proj"), strict)?;
        Ok(())
    }
}
