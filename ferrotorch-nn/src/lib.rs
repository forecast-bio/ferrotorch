// Allow the proc macro's generated code (`::ferrotorch_nn::Module`, etc.)
// to resolve when used from *within* this crate (e.g., integration tests
// compiled as part of ferrotorch-nn itself).
extern crate self as ferrotorch_nn;

pub mod activation;
pub mod attention;
pub mod container;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod functional;
pub mod hooks;
pub mod init;
pub mod linear;
pub mod loss;
pub mod module;
pub mod norm;
pub mod parameter;
pub mod pooling;
pub mod rnn;
pub mod transformer;
pub mod utils;

pub use activation::{
    CELU, ELU, GELU, GLU, HardSigmoid, HardSwish, LeakyReLU, LogSoftmax, Mish, PReLU, ReLU, SELU,
    SiLU, Sigmoid, Softmax, Softplus, Tanh,
};
pub use attention::MultiheadAttention;
pub use container::{ModuleDict, ModuleList, Sequential};
pub use conv::{Conv1d, Conv2d, ConvTranspose2d};
pub use dropout::{Dropout, Dropout2d};
pub use embedding::Embedding;
pub use init::NonLinearity;
pub use linear::Linear;
pub use loss::{
    BCEWithLogitsLoss, CosineEmbeddingLoss, CrossEntropyLoss, HuberLoss, KLDivLoss, MSELoss,
    SmoothL1Loss,
};
pub use hooks::{BackwardHook, ForwardHook, ForwardPreHook, HookHandle, HookedModule};
pub use module::{Module, Reduction, StateDict};
// Re-export the derive macro. The derive macro and the trait share the name
// `Module` but live in different namespaces (macro vs type), so both are
// usable simultaneously: `use ferrotorch_nn::{Module, ...}` gives the trait,
// and `#[derive(Module)]` resolves to the derive macro.
pub use ferrotorch_nn_derive::Module;
pub use norm::{BatchNorm2d, GroupNorm, LayerNorm, RMSNorm};
pub use parameter::Parameter;
pub use pooling::{
    adaptive_avg_pool2d, avg_pool2d, max_pool2d, AdaptiveAvgPool2d, AvgPool2d, MaxPool2d,
};
pub use rnn::LSTM;
pub use transformer::{
    KVCache, RotaryPositionEmbedding, SwiGLU, TransformerDecoderLayer, TransformerEncoderLayer,
};
pub use utils::{clip_grad_norm_, clip_grad_value_};
