// Allow the proc macro's generated code (`::ferrotorch_nn::Module`, etc.)
// to resolve when used from *within* this crate (e.g., integration tests
// compiled as part of ferrotorch-nn itself).
extern crate self as ferrotorch_nn;

pub mod activation;
pub mod attention;
pub mod buffer;
pub mod container;
pub mod conv;
pub mod dropout;
pub mod embedding;
pub mod flash_attention;
pub mod flex_attention;
pub mod functional;
pub mod hooks;
pub mod identity;
pub mod init;
pub mod lazy_conv;
pub mod lazy_conv_transpose;
pub mod lazy_linear;
pub mod lazy_norm;
pub mod linear;
pub mod lora;
pub mod loss;
pub mod module;
pub mod norm;
pub mod padding;
pub mod paged_attention;
pub mod parameter;
pub mod parameter_container;
pub mod pooling;
pub mod qat;
pub mod rnn;
pub mod rnn_utils;
pub mod transformer;
pub mod upsample;
pub mod utils;

pub use activation::{
    CELU, ELU, GELU, GLU, GeluApproximate, HardSigmoid, HardSwish, Hardshrink, Hardtanh, LeakyReLU,
    LogSigmoid, LogSoftmax, Mish, PReLU, RReLU, ReLU, ReLU6, SELU, SiLU, Sigmoid, Softmax,
    Softmax2d, Softmin, Softplus, Softshrink, Softsign, Tanh, Tanhshrink, Threshold,
};
pub use attention::{MultiheadAttention, repeat_kv, reshape_to_heads, transpose_heads_to_2d};
pub use container::{ModuleDict, ModuleList, Sequential};
pub use conv::{Conv1d, Conv2d, Conv3d, ConvTranspose1d, ConvTranspose2d, ConvTranspose3d};
pub use dropout::{AlphaDropout, Dropout, Dropout1d, Dropout2d, Dropout3d};
pub use embedding::{Embedding, EmbeddingBag, EmbeddingBagMode};
pub use flash_attention::{flash_attention, standard_attention};
pub use flex_attention::{
    BlockMask, alibi_score_mod, causal_score_mod, flex_attention, relative_position_bias_score_mod,
};
pub use hooks::{BackwardHook, ForwardHook, ForwardPreHook, HookHandle, HookedModule};
pub use identity::{
    ChannelShuffle, CosineSimilarity, Flatten, Identity, PairwiseDistance, Unflatten,
};
pub use init::NonLinearity;
pub use lazy_conv::{LazyConv1d, LazyConv2d, LazyConv3d};
pub use lazy_linear::LazyLinear;
pub use linear::Linear;
pub use lora::LoRALinear;
pub use loss::{
    BCELoss, BCEWithLogitsLoss, CTCLoss, CosineEmbeddingLoss, CrossEntropyLoss, GaussianNLLLoss,
    HingeEmbeddingLoss, HuberLoss, KLDivLoss, L1Loss, MSELoss, MarginRankingLoss,
    MultiLabelSoftMarginLoss, MultiMarginLoss, NLLLoss, PoissonNLLLoss, SmoothL1Loss,
    TripletMarginLoss,
};
pub use module::{Module, Reduction, StateDict};
// Re-export the derive macro. The derive macro and the trait share the name
// `Module` but live in different namespaces (macro vs type), so both are
// usable simultaneously: `use ferrotorch_nn::{Module, ...}` gives the trait,
// and `#[derive(Module)]` resolves to the derive macro.
pub use buffer::Buffer;
pub use ferrotorch_nn_derive::Module;
pub use norm::{
    BatchNorm1d, BatchNorm2d, BatchNorm3d, GroupNorm, InstanceNorm1d, InstanceNorm2d,
    InstanceNorm3d, LayerNorm, LocalResponseNorm, RMSNorm,
};
pub use padding::{
    CircularPad1d, CircularPad2d, CircularPad3d, ConstantPad1d, ConstantPad2d, ConstantPad3d,
    PaddingMode, ReflectionPad1d, ReflectionPad2d, ReflectionPad3d, ReplicationPad1d,
    ReplicationPad2d, ReplicationPad3d, ZeroPad1d, ZeroPad2d, ZeroPad3d,
};
pub use paged_attention::{KVPage, PagePool, PagedAttentionManager, PagedKVCache};
pub use parameter::Parameter;
pub use parameter_container::{ParameterDict, ParameterList};
pub use pooling::{
    AdaptiveAvgPool1d, AdaptiveAvgPool2d, AdaptiveAvgPool3d, AdaptiveMaxPool1d, AdaptiveMaxPool2d,
    AdaptiveMaxPool3d, AvgPool1d, AvgPool2d, AvgPool3d, FractionalMaxPool2d, LPPool1d, LPPool2d,
    MaxPool1d, MaxPool2d, MaxPool3d, MaxUnpool2d, adaptive_avg_pool1d, adaptive_avg_pool2d,
    adaptive_avg_pool3d, adaptive_max_pool1d, adaptive_max_pool2d, adaptive_max_pool3d, avg_pool1d,
    avg_pool2d, avg_pool3d, lp_pool1d, lp_pool2d, max_pool1d, max_pool2d, max_pool3d, max_unpool2d,
};
pub use qat::{ObserverType, QatConfig, QatModel, QuantizedModel, prepare_qat};
pub use rnn::{GRU, GRUCell, LSTM, LSTMCell, RNN, RNNCell, RNNNonlinearity};
pub use rnn_utils::{PackedSequence, pack_padded_sequence, pad_packed_sequence};
pub use transformer::{
    KVCache, RoPEConvention, RoPEScaling, RotaryPositionEmbedding, SwiGLU, Transformer,
    TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer,
};
pub use upsample::{
    Fold, GridSampleMode, GridSamplePaddingMode, InterpolateMode, PixelShuffle, PixelUnshuffle,
    Unfold, Upsample, affine_grid, fold, grid_sample, interpolate, pixel_shuffle, pixel_unshuffle,
    unfold,
};
pub use utils::{clip_grad_norm_, clip_grad_value_};
