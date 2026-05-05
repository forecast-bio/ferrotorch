// Lint baseline mirrors the workspace-standard pattern from
// `ferrotorch-core`/`-distributed`/`-jit`/`-cubecl`/`-xpu` lib.rs.
#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms)]
// `missing_docs` and `missing_debug_implementations` are held at `allow`
// while the workspace-wide rustdoc / `Debug` pass is tracked separately
// (matches the existing `ferrotorch-core`/`-gpu`/`-distributed` precedent —
// diverging unilaterally from a leaf crate would be Step 4 architectural
// unilateralism). Several modules expose `Box<dyn Module<T>>` trait
// objects whose `Debug` impls require careful hand-rolling.
#![allow(missing_docs, missing_debug_implementations)]
// Pedantic lints we explicitly accept across this crate. Each allow names
// a concrete reason — the alternative would be churn-for-zero-benefit or
// a worse API. Mirrors the ferrotorch-core / ferrotorch-distributed
// baseline; add to this list only with a one-line justification.
#![allow(
    // The crate is laid out so submodule names (`module::Module`,
    // `parameter::Parameter`, `loss::MSELoss`) match the public type they
    // export; renaming would force ergonomic breakage.
    clippy::module_name_repetitions,
    // # Errors / # Panics sections are added as part of focused passes
    // (this audit's Finding #5 covers the high-leverage NotImplementedOnCuda
    // sites in loss.rs); a blanket sweep is tracked separately.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // NN code casts pervasively between `usize` (shape, indices) and
    // floating-point (norms, scales) and between `f32`/`f64` (mixed
    // precision). The explicit cast is more readable than a `cast()` call
    // through num-traits in arithmetic-heavy kernels.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    // `#[must_use]` on every getter is churn for marginal value; callers
    // in this codebase already use the returned values.
    clippy::must_use_candidate,
    // Builder-style methods returning `Self` document their pattern in
    // the type signature; `#[must_use]` is noise.
    clippy::return_self_not_must_use,
    // Doc comments follow the standard rustdoc layout; pedantic
    // doc-markdown rules are too aggressive for the technical prose
    // (PyTorch op names, math notation).
    clippy::doc_markdown,
    // Test/helper modules define small fns after `let`-bindings; the
    // hoisting requirement is style-only.
    clippy::items_after_statements,
    // Long `match`-on-reduction / op blocks mirror PyTorch's taxonomy 1:1;
    // splitting reduces legibility.
    clippy::too_many_lines,
    // Most NN ops take large structs / tensors by reference already; the
    // pedantic threshold flags `Reduction` (Copy, single byte) when passed
    // by value.
    clippy::needless_pass_by_value,
    // `if let Some(x) = y { ... } else { ... }` is the idiomatic Rust
    // shape; the lint's preferred ladders are less readable.
    clippy::option_if_let_else,
    // Trivial getters return references to small `Copy` fields; the
    // suggested change to direct field exposure would conflict with the
    // pub-field-via-non_exhaustive policy used by loss configuration types.
    clippy::trivially_copy_pass_by_ref,
    // Long literals appear in test reference values (PyTorch parity
    // fixtures); separating with underscores reduces fidelity to the
    // original numeric source.
    clippy::unreadable_literal,
    // `match` on small enums (Reduction, GeluApproximate, …) is more
    // legible than chained `if let`.
    clippy::single_match_else,
    // Pred / target / grad / loss share short identifiers in math-heavy
    // kernels; `similar_names` flags them but renaming hurts readability
    // for readers familiar with the PyTorch reference.
    clippy::similar_names,
    // Math kernels naturally use single-character names (m, k, n for matmul
    // dims; i, j for indices); requiring longer names hurts readability.
    clippy::many_single_char_names,
    // `let ... else { return }` rewrites of `match { Some(x) => x, None => ... }`
    // are often less readable when the match arm is the natural pattern.
    clippy::manual_let_else,
    // GPU host-side kernels often have many `&[T]` parameters mirroring a
    // kernel's input signature; refactoring each into a struct adds churn
    // without benefit. Per-site comments justify retained allows.
    clippy::too_many_arguments,
    // `.collect::<Vec<_>>()` after mapping is the idiomatic shape; rewriting
    // to extend(map(..)) is lossier and clippy's preference is contested.
    clippy::redundant_closure_for_method_calls,
    // Tensor ops naturally use `for i in 0..n { ... }` over `for x in arr.iter()`
    // when the index itself is needed (multi-dim addressing); the pedantic
    // preference for `iter()` is contested in this codebase.
    clippy::needless_range_loop,
    // Many NN kernels compare floats exactly to zero / one as a fast-path
    // guard (e.g., `if scale == 0.0 { return zeros }`). The math is exact
    // for these literal cases; bit-level comparison is intentional.
    clippy::float_cmp,
    // Format-string inlining (`format!("{x}")` over `format!("{}", x)`) is
    // a churn-only sweep; the workspace tracks this in a separate cleanup
    // pass per the existing precedent in ferrotorch-core / -distributed.
    clippy::uninlined_format_args,
    // `.to_vec()` via Deref is the natural shape for tensor slice → owned;
    // the lint's preferred form (`.iter().copied().collect()`) is uglier.
    clippy::implicit_clone,
    // `(a + b) / 2` idiom appears in interpolation; the `midpoint`
    // intrinsic that clippy suggests has subtle precision differences.
    clippy::manual_midpoint,
    // `match` arms wrapping `(a, b)` against `Some` / `None` with explicit
    // arms are clearer than the `option_map_unit_fn` rewrite.
    clippy::option_map_unit_fn,
    // `map(...).unwrap_or(...)` is a perfectly clear idiom; the suggested
    // `map_or` rewrite is one shorter token but loses the structural shape.
    clippy::map_unwrap_or,
    // `as` casts between raw pointers (e.g. `*const T as *const u8`) are
    // load-bearing in GPU buffer reinterpretation paths; the suggested
    // `.cast::<U>()` shape obscures the byte-reinterpretation intent.
    clippy::ptr_as_ptr,
    // Bind-to-_ patterns appear in tests where the compiler would otherwise
    // complain about an unused mutability or move; the `_` is intentional.
    clippy::no_effect_underscore_binding,
    // `IntoIterator for &Foo` is a legitimate API decision but `&Foo`
    // here is the right type even without explicit `iter()` blanket impls.
    clippy::iter_without_into_iter,
    // Manual `Debug` impls in this crate intentionally elide internal
    // fields (e.g., heavy tensors) for log readability; clippy wants every
    // field included.
    clippy::missing_fields_in_debug,
    // Shape arrays are naturally `[usize; N]`; the lint's preferred
    // `Box<[usize]>` would force allocation in a hot path.
    clippy::single_match,
    // `if let Some(x) = ... { ... } else { ... }` is the idiomatic Rust
    // shape for biased Option matching; clippy's `else` rewrites are noisier.
    clippy::if_not_else,
    // `unsafe_derive_deserialize` doesn't apply (we don't serde unsafe types
    // through this crate), but the lint mis-fires on `#[derive(Debug)]` of
    // structs containing `Box<dyn Module>`. Held until a workspace-wide
    // serde audit lands.
    clippy::unsafe_derive_deserialize,
    // `for x in arr.iter() { ... }` is the explicit form some readers
    // prefer; the rewrite to `for x in arr` is a style-only refactor.
    clippy::explicit_iter_loop,
    // `if cond { return ...; } else { ... }` with the else block holding
    // the natural fall-through is a deliberate style choice when the
    // early-return path is logically the exception; clippy wants to
    // collapse it.
    clippy::redundant_else,
)]

// Allow the proc macro's generated code (`::ferrotorch_nn::Module`, etc.)
// to resolve when used from *within* this crate (e.g., integration tests
// compiled as part of ferrotorch-nn itself). This `extern crate` is
// load-bearing for the derive macro's hygienic path even though it appears
// unused to the compiler — `rust_2018_idioms` flags it without seeing the
// macro expansion.
#[allow(unused_extern_crates)]
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

/// Glob-import-friendly re-exports of the most commonly used items.
///
/// Pulls in the core building blocks needed to write a model: the `Module`
/// trait + derive macro, `Parameter`, `StateDict`, the standard layers
/// (`Linear`, `Conv2d`, `LayerNorm`, `GELU`, …), the canonical losses
/// (`MSELoss`, `CrossEntropyLoss`), and the gradient-clipping helpers.
///
/// Mirrors PyTorch's `from torch import nn` ergonomics.
///
/// ```ignore
/// use ferrotorch_nn::prelude::*;
/// ```
pub mod prelude {
    // Core abstractions: trait, parameter, state-dict, derive macro.
    pub use crate::buffer::Buffer;
    pub use crate::module::{Module, Reduction, StateDict};
    pub use crate::parameter::Parameter;
    pub use ferrotorch_nn_derive::Module as DeriveModule;

    // Standard layers most models use.
    pub use crate::activation::{GELU, ReLU, Sigmoid, Softmax, Tanh};
    pub use crate::container::{ModuleDict, ModuleList, Sequential};
    pub use crate::conv::{Conv1d, Conv2d, Conv3d};
    pub use crate::dropout::Dropout;
    pub use crate::embedding::Embedding;
    pub use crate::linear::Linear;
    pub use crate::norm::{BatchNorm1d, BatchNorm2d, GroupNorm, LayerNorm, RMSNorm};
    pub use crate::pooling::{AdaptiveAvgPool2d, MaxPool2d};

    // Canonical losses.
    pub use crate::loss::{BCELoss, BCEWithLogitsLoss, CrossEntropyLoss, L1Loss, MSELoss, NLLLoss};

    // Gradient clipping (utils).
    pub use crate::utils::{clip_grad_norm_, clip_grad_value_};
}
