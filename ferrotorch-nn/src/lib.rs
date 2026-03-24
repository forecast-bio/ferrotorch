// Allow the proc macro's generated code (`::ferrotorch_nn::Module`, etc.)
// to resolve when used from *within* this crate (e.g., integration tests
// compiled as part of ferrotorch-nn itself).
extern crate self as ferrotorch_nn;

pub mod activation;
pub mod attention;
pub mod flash_attention;
pub mod flex_attention;
pub mod container;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod functional;
pub mod hooks;
pub mod identity;
pub mod init;
pub mod linear;
pub mod lora;
pub mod loss;
pub mod module;
pub mod norm;
pub mod padding;
pub mod paged_attention;
pub mod parameter;
pub mod pooling;
pub mod qat;
pub mod rnn;
pub mod rnn_utils;
pub mod transformer;
pub mod upsample;
pub mod utils;

pub use activation::{
    CELU, ELU, GELU, GLU, GeluApproximate, HardSigmoid, HardSwish, LeakyReLU, LogSoftmax, Mish,
    PReLU, ReLU, SELU, SiLU, Sigmoid, Softmax, Softplus, Tanh,
};
pub use attention::MultiheadAttention;
pub use flash_attention::{flash_attention, standard_attention};
pub use flex_attention::{
    flex_attention, BlockMask, alibi_score_mod, causal_score_mod, relative_position_bias_score_mod,
};
pub use container::{ModuleDict, ModuleList, Sequential};
pub use conv::{Conv1d, Conv2d, ConvTranspose2d};
pub use dropout::{Dropout, Dropout2d};
pub use embedding::Embedding;
pub use identity::{Flatten, Identity};
pub use init::NonLinearity;
pub use linear::Linear;
pub use lora::LoRALinear;
pub use loss::{
    BCELoss, BCEWithLogitsLoss, CTCLoss, CosineEmbeddingLoss, CrossEntropyLoss,
    HingeEmbeddingLoss, HuberLoss, KLDivLoss, L1Loss, MSELoss, MarginRankingLoss,
    MultiLabelSoftMarginLoss, MultiMarginLoss, NLLLoss, PoissonNLLLoss, SmoothL1Loss,
    TripletMarginLoss,
};
pub use hooks::{BackwardHook, ForwardHook, ForwardPreHook, HookHandle, HookedModule};
pub use module::{Module, Reduction, StateDict};
// Re-export the derive macro. The derive macro and the trait share the name
// `Module` but live in different namespaces (macro vs type), so both are
// usable simultaneously: `use ferrotorch_nn::{Module, ...}` gives the trait,
// and `#[derive(Module)]` resolves to the derive macro.
pub use ferrotorch_nn_derive::Module;
pub use norm::{
    BatchNorm1d, BatchNorm2d, GroupNorm, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d, LayerNorm,
    RMSNorm,
};
pub use padding::{
    ConstantPad1d, ConstantPad2d, ConstantPad3d, PaddingMode, ReflectionPad1d, ReflectionPad2d,
    ReflectionPad3d, ReplicationPad1d, ReplicationPad2d, ReplicationPad3d, ZeroPad1d, ZeroPad2d,
    ZeroPad3d,
};
pub use paged_attention::{KVPage, PagePool, PagedAttentionManager, PagedKVCache};
pub use parameter::Parameter;
pub use pooling::{
    adaptive_avg_pool1d, adaptive_avg_pool2d, adaptive_avg_pool3d, adaptive_max_pool2d,
    avg_pool1d, avg_pool2d, avg_pool3d, max_pool1d, max_pool2d, max_pool3d, max_unpool2d,
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool2d, AvgPool1d,
    AvgPool2d, AvgPool3d, MaxPool1d, MaxPool2d, MaxPool3d, MaxUnpool2d,
};
pub use rnn::{GRU, LSTM};
pub use rnn_utils::{PackedSequence, pack_padded_sequence, pad_packed_sequence};
pub use transformer::{
    KVCache, RoPEConvention, RotaryPositionEmbedding, SwiGLU, TransformerDecoderLayer,
    TransformerEncoderLayer,
};
pub use qat::{prepare_qat, ObserverType, QatConfig, QatModel, QuantizedModel};
pub use upsample::{
    affine_grid, fold, grid_sample, interpolate, pixel_shuffle, pixel_unshuffle, unfold, Fold,
    GridSampleMode, GridSamplePaddingMode, InterpolateMode, PixelShuffle, PixelUnshuffle, Unfold,
    Upsample,
};
pub use utils::{clip_grad_norm_, clip_grad_value_};
