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

pub mod config;
pub mod attention;
pub mod mlp;
pub mod layer;
pub mod model;
#[cfg(feature = "cuda")]
pub mod gpu;

pub use attention::LlamaAttention;
pub use config::{LlamaActivation, LlamaConfig};
pub use layer::LlamaDecoderLayer;
pub use mlp::LlamaMLP;
pub use model::{LlamaForCausalLM, LlamaModel};

#[cfg(feature = "cuda")]
pub use gpu::{LlamaGpuInferencer, LlamaGpuLayer, ProfiledForwardResult};
