//! Llama 3 (Meta LLaMA) model composition for ferrotorch.
//!
//! Assembles the standard Llama decoder stack from ferrotorch primitives:
//!
//! ```text
//! LlamaForCausalLM
//! ├── LlamaModel
//! │   ├── Embedding                      (token embeddings)
//! │   ├── LlamaDecoderLayer × N
//! │   │   ├── RMSNorm                    (pre-attn)
//! │   │   ├── LlamaAttention             (GQA + RoPE)
//! │   │   ├── residual
//! │   │   ├── RMSNorm                    (pre-MLP)
//! │   │   ├── SwiGLU                     (gate/up/down projections)
//! │   │   └── residual
//! │   └── RMSNorm                        (final)
//! └── Linear lm_head                     (projection to vocab)
//! ```
//!
//! # Loading real weights
//!
//! [`LlamaForCausalLM::load_hf_state_dict`] accepts a `StateDict` whose
//! keys use the HuggingFace transformers naming convention and rewrites
//! them to match the ferrotorch parameter paths before delegating to
//! [`Module::load_state_dict`]. Combined with
//! `ferrotorch_serialize::load_safetensors_sharded` this gives a direct
//! path from a downloaded Meta-Llama-3-8B checkpoint to a loaded model.

pub mod attention;
pub mod config;
pub mod generation;
pub mod gguf_remap;
#[cfg(feature = "cuda")]
pub mod gpu;
pub mod grammar;
pub mod layer;
pub mod mlp;
pub mod model;
pub mod quant_loaders;

pub use attention::LlamaAttention;
pub use config::{LlamaActivation, LlamaConfig};
pub use generation::{
    GenerationConfig, apply_repetition_penalty, apply_temperature, argmax, generate,
    generate_with_streamer, sample_softmax, top_k_filter, top_p_filter,
};
pub use gguf_remap::{gguf_key_to_hf, gguf_to_hf_state_dict};
pub use layer::LlamaDecoderLayer;
pub use mlp::LlamaMLP;
pub use model::{LlamaForCausalLM, LlamaModel};
pub use quant_loaders::{AwqQ4, GptqQ4, dequantize_awq_q4, dequantize_gptq_q4};

#[cfg(feature = "cuda")]
pub use gpu::{LlamaGpuInferencer, LlamaGpuLayer, ProfiledForwardResult};
