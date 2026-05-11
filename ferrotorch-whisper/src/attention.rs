//! Whisper encoder self-attention block.
//!
//! Bidirectional (non-causal) multi-head attention. Differs from the
//! BERT block in three structural ways:
//!
//! * **No K bias.** Q, V, and out_proj have additive biases; K does not.
//!   The upstream HF `WhisperAttention` follows
//!   `Linear(bias=False)` for `k_proj` and `Linear(bias=True)` for the
//!   other three — and `WhisperEncoderLayer` wraps the attention in a
//!   PRE-norm residual block (vs BERT's post-norm).
//! * **Pre-norm only.** Self-attention is invoked on the
//!   `self_attn_layer_norm(input)` output; the residual `input +
//!   self_attn(...)` is added downstream in the layer module.
//! * **GELU FFN** (already shared with BERT).
//!
//! ```text
//! q = Linear(input, bias=True)
//! k = Linear(input, bias=False)
//! v = Linear(input, bias=True)
//! ctx = softmax(Q K^T / √d) V   // 6 heads, no causal mask
//! out = Linear(ctx, bias=True)
//! ```
//!
//! The returned tensor is the raw attention output `[1, S, hidden]` —
//! the residual add is the caller's responsibility (matches HF's
//! `WhisperEncoderLayer.forward`).

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::{Module, StateDict};
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Linear, reshape_to_heads, standard_attention, transpose_heads_to_2d};

use crate::config::WhisperConfig;

/// HF `WhisperAttention` (encoder, self-attention variant).
#[derive(Debug)]
pub struct WhisperEncoderSelfAttention<T: Float> {
    /// Query projection — `[d_model -> d_model]`, with bias.
    pub q_proj: Linear<T>,
    /// Key projection — `[d_model -> d_model]`, NO bias.
    pub k_proj: Linear<T>,
    /// Value projection — `[d_model -> d_model]`, with bias.
    pub v_proj: Linear<T>,
    /// Output projection — `[d_model -> d_model]`, with bias.
    pub out_proj: Linear<T>,
    num_heads: usize,
    head_dim: usize,
    hidden: usize,
    training: bool,
}

impl<T: Float> WhisperEncoderSelfAttention<T> {
    /// Build randomly-initialized encoder self-attention projections.
    ///
    /// # Errors
    ///
    /// Returns the underlying [`FerrotorchError`] on bad config dims.
    pub fn new(cfg: &WhisperConfig) -> FerrotorchResult<Self> {
        cfg.validate()?;
        Ok(Self {
            q_proj: Linear::new(cfg.d_model, cfg.d_model, /* bias = */ true)?,
            k_proj: Linear::new(cfg.d_model, cfg.d_model, /* bias = */ false)?,
            v_proj: Linear::new(cfg.d_model, cfg.d_model, /* bias = */ true)?,
            out_proj: Linear::new(cfg.d_model, cfg.d_model, /* bias = */ true)?,
            num_heads: cfg.encoder_attention_heads,
            head_dim: cfg.encoder_head_dim(),
            hidden: cfg.d_model,
            training: false,
        })
    }
}

impl<T: Float> Module<T> for WhisperEncoderSelfAttention<T> {
    /// Forward pass.
    ///
    /// `input` is the *layer-normed* `[1, seq_len, d_model]` tensor.
    /// Output is `[1, seq_len, d_model]` ready to add to the residual.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        if shape.len() != 3 || shape[0] != 1 || shape[2] != self.hidden {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "WhisperEncoderSelfAttention expects [1, S, {}], got {:?}",
                    self.hidden, shape,
                ),
            });
        }
        let seq_len = shape[1];

        // Q / K / V projections.
        let q = self.q_proj.forward(input)?;
        let k = self.k_proj.forward(input)?;
        let v = self.v_proj.forward(input)?;

        // Drop the batch=1 lead and split into heads.
        let q2 = reshape_2d(&q, seq_len, self.hidden)?;
        let k2 = reshape_2d(&k, seq_len, self.hidden)?;
        let v2 = reshape_2d(&v, seq_len, self.hidden)?;
        let q_h = reshape_to_heads(&q2, self.num_heads, seq_len, self.head_dim)?;
        let k_h = reshape_to_heads(&k2, self.num_heads, seq_len, self.head_dim)?;
        let v_h = reshape_to_heads(&v2, self.num_heads, seq_len, self.head_dim)?;

        // Non-causal scaled dot-product attention.
        let ctx = standard_attention(&q_h, &k_h, &v_h, /* causal = */ false)?;
        let ctx2 = transpose_heads_to_2d(&ctx, self.num_heads, seq_len, self.head_dim)?;
        let ctx3 = reshape_3d(&ctx2, 1, seq_len, self.hidden)?;

        // Final output projection.
        self.out_proj.forward(&ctx3)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.q_proj.parameters());
        out.extend(self.k_proj.parameters());
        out.extend(self.v_proj.parameters());
        out.extend(self.out_proj.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut out = Vec::new();
        out.extend(self.q_proj.parameters_mut());
        out.extend(self.k_proj.parameters_mut());
        out.extend(self.v_proj.parameters_mut());
        out.extend(self.out_proj.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.q_proj.named_parameters() {
            out.push((format!("q_proj.{n}"), p));
        }
        for (n, p) in self.k_proj.named_parameters() {
            out.push((format!("k_proj.{n}"), p));
        }
        for (n, p) in self.v_proj.named_parameters() {
            out.push((format!("v_proj.{n}"), p));
        }
        for (n, p) in self.out_proj.named_parameters() {
            out.push((format!("out_proj.{n}"), p));
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
            let prefixes = ["q_proj", "k_proj", "v_proj", "out_proj"];
            for key in state.keys() {
                if !prefixes.iter().any(|p| key.starts_with(&format!("{p}."))) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!(
                            "unexpected key in WhisperEncoderSelfAttention state_dict: \"{key}\""
                        ),
                    });
                }
            }
        }
        self.q_proj.load_state_dict(&extract("q_proj"), strict)?;
        self.k_proj.load_state_dict(&extract("k_proj"), strict)?;
        self.v_proj.load_state_dict(&extract("v_proj"), strict)?;
        self.out_proj
            .load_state_dict(&extract("out_proj"), strict)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// 2-D / 3-D reshape helpers (own the data; no view trick).
// ---------------------------------------------------------------------------

fn reshape_2d<T: Float>(t: &Tensor<T>, a: usize, b: usize) -> FerrotorchResult<Tensor<T>> {
    let data = t.data_vec()?;
    Tensor::from_storage(TensorStorage::cpu(data), vec![a, b], t.requires_grad())
}

fn reshape_3d<T: Float>(
    t: &Tensor<T>,
    a: usize,
    b: usize,
    c: usize,
) -> FerrotorchResult<Tensor<T>> {
    let data = t.data_vec()?;
    Tensor::from_storage(TensorStorage::cpu(data), vec![a, b, c], t.requires_grad())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_cfg() -> WhisperConfig {
        WhisperConfig {
            vocab_size: 32,
            num_mel_bins: 80,
            d_model: 16,
            encoder_layers: 2,
            encoder_attention_heads: 2,
            encoder_ffn_dim: 32,
            decoder_layers: 2,
            decoder_attention_heads: 2,
            decoder_ffn_dim: 32,
            max_source_positions: 8,
            max_target_positions: 8,
        }
    }

    #[test]
    fn attention_shape() {
        let attn = WhisperEncoderSelfAttention::<f32>::new(&tiny_cfg()).unwrap();
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![0.5f32; 4 * 16]),
            vec![1, 4, 16],
            false,
        )
        .unwrap();
        let out = attn.forward(&x).unwrap();
        assert_eq!(out.shape(), &[1, 4, 16]);
        for &v in out.data().unwrap() {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn named_parameters_match_hf_layout_no_k_bias() {
        let attn = WhisperEncoderSelfAttention::<f32>::new(&tiny_cfg()).unwrap();
        let names: Vec<String> = attn
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        // HF Whisper has q/v/out_proj bias but NO k_proj bias.
        assert!(names.contains(&"q_proj.weight".to_string()));
        assert!(names.contains(&"q_proj.bias".to_string()));
        assert!(names.contains(&"k_proj.weight".to_string()));
        assert!(!names.iter().any(|n| n == "k_proj.bias"));
        assert!(names.contains(&"v_proj.weight".to_string()));
        assert!(names.contains(&"v_proj.bias".to_string()));
        assert!(names.contains(&"out_proj.weight".to_string()));
        assert!(names.contains(&"out_proj.bias".to_string()));
    }
}
