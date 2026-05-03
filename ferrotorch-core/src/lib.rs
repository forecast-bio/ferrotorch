// Lint baseline mirrors `ferrotorch-jit/src/lib.rs`. `missing_docs` is held at
// `warn` while the crate-wide rustdoc pass is tracked as a follow-up issue.
// Tighten to `deny` once that pass lands.
#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms)]
// `missing_debug_implementations` is held at warn while the rustdoc / Debug
// pass is tracked as a follow-up issue alongside the missing_docs sweep.
#![warn(missing_debug_implementations)]
// Pedantic lints we explicitly accept across this crate. Each allow names a
// concrete reason — the alternative would be churn-for-zero-benefit or a
// worse API. Add to this list only with a one-line justification.
#![allow(
    // The IR is laid out so helper structs inherit their parents' naming;
    // unifying would break call-site ergonomics.
    clippy::module_name_repetitions,
    // # Errors / # Panics sections will be added during the rustdoc pass
    // tracked as a follow-up issue, not gated on this lint baseline.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Op-dispatch `match` blocks mirror the operator taxonomy 1:1; splitting
    // reduces legibility.
    clippy::too_many_lines,
    // Trivial casts are pervasive (offsets, indices, byte reinterpretation
    // around GPU buffers); the explicit cast is more readable than alternatives.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    // `#[must_use]` on every getter is churn for marginal value; callers in
    // this codebase already use the returned values.
    clippy::must_use_candidate,
    // `let ... else { return }` rewrites of `match { Some(x) => x, None => return }`
    // are often less readable when the match arm is the natural pattern.
    clippy::manual_let_else,
    // Test/helper modules define small private fns after `let`-bindings; the
    // hoisting requirement is style-only.
    clippy::items_after_statements,
    // GPU trait methods often have many `&GpuBufferHandle` parameters mirroring
    // a kernel's input signature; refactoring each into a struct is tracked as
    // a separate (low-priority) follow-up issue.
    clippy::too_many_arguments,
    // Hex-encoded constants in indexing/codegen don't gain readability from
    // the underscore separators clippy prefers.
    clippy::unreadable_literal,
    // Builder-style methods on configs that return `Self` already document
    // their consume-and-return pattern; `#[must_use]` is noise.
    clippy::return_self_not_must_use,
    // Math kernels naturally use single-character names (m, k, n for matmul
    // dims; i, j for indices); requiring longer names hurts readability.
    clippy::many_single_char_names,
    clippy::similar_names,
    // Doc comments that begin with `///` follow the standard rustdoc layout;
    // pedantic doc-markdown rules are too aggressive for technical prose.
    clippy::doc_markdown,
    // Shape arrays and indices commonly use `as` casts between usize and
    // smaller integer types; alternatives (try_into + unwrap, num_traits)
    // either still panic or substantially harm readability.
    clippy::cast_lossless,
    // `.collect::<Vec<_>>()` after mapping is the idiomatic shape; rewriting
    // to extend(map(..)) is lossier and clippy's preference is contested.
    clippy::redundant_closure_for_method_calls,
    // Function parameter names `_a`, `_b` mirror tensor naming conventions
    // (left/right operand) and are not single-letter cargo-cult.
    clippy::single_match_else,
    // Tensor ops naturally use `.iter().map(...).sum()` chains; the explicit
    // chain is more readable than alternative fold patterns.
    clippy::needless_range_loop,
    // Match arms that wrap a single variant and re-export are intentional
    // when the variant set is documented and the wrapper is part of the API.
    clippy::match_wildcard_for_single_variants,
)]
// `missing_docs` is held at warn while the crate-wide rustdoc pass is tracked
// in a follow-up issue (see issue list); flip to `deny` once that pass lands.
#![allow(missing_docs)]
// `unsafe_code` is permitted: ferrotorch-core wraps GPU buffers, raw byte
// transmutes for SIMD f32/f64 fast paths, and `Arc`-shared storage that the
// optimizer mutates under documented exclusive-access invariants. Each block
// has a per-site `// SAFETY:` justification.

pub mod autograd;
pub mod bool_tensor;
pub mod complex_tensor;
pub mod cpu_pool;
pub mod creation;
pub mod device;
pub mod dispatch;
mod display;
pub mod dtype;
pub mod einops;
pub mod einsum;
pub mod error;
pub mod fft;
pub mod flex_attention;
pub mod gpu_dispatch;
pub mod grad_fns;
mod inplace;
pub mod int_tensor;
pub mod linalg;
pub mod masked;
pub mod meta_propagate;
mod methods;
pub mod named_tensor;
pub mod nested;
pub mod numeric_cast;
pub mod ops;
mod ops_trait;
pub mod profiler_hook;
pub mod pruning;
pub mod quantize;
pub mod shape;
pub mod signal;
pub mod sparse;
pub mod special;
pub mod storage;
pub mod stride_tricks;
pub mod tensor;
pub mod vmap;

// Public re-exports for ergonomic use.
pub use autograd::anomaly::{
    AnomalyMode, ForwardBacktrace, check_gradient_anomaly, detect_anomaly,
};
pub use autograd::hooks::HookHandle;
pub use autograd::{
    AutocastCategory, AutocastDtype, autocast, autocast_dtype, autocast_guard, backward,
    backward_with_grad, cond, enable_grad, fixed_point, grad, grad_norm, gradient_penalty, hessian,
    is_autocast_debug, is_autocast_enabled, is_grad_enabled, jacobian, jvp, no_grad, scan,
    set_autocast_debug, set_grad_enabled, validate_cond_branches, vjp,
};
pub use autograd::{
    DualTensor, dual_add, dual_cos, dual_div, dual_exp, dual_log, dual_matmul, dual_mul, dual_neg,
    dual_relu, dual_sigmoid, dual_sin, dual_sub, dual_tanh, jacfwd, jvp_exact,
};
pub use bool_tensor::BoolTensor;
pub use complex_tensor::ComplexTensor;
pub use creation::{
    arange, eye, from_slice, from_vec, full, full_like, linspace, ones, ones_like, rand, rand_like,
    randn, randn_like, scalar, tensor, zeros, zeros_like,
};
pub use device::Device;
pub use dtype::{DType, Element, Float};
pub use einops::{EinopsReduction, rearrange, rearrange_with, reduce, repeat};
pub use einsum::{einsum, einsum_differentiable};
pub use error::{FerrotorchError, FerrotorchResult};
pub use int_tensor::{IntElement, IntTensor};
pub use named_tensor::NamedTensor;
// Linalg ops are accessed via the `linalg` module namespace
// (e.g. `ferrotorch_core::linalg::svd`) to mirror `torch.linalg.*` and avoid
// shadowing top-level identifiers (autograd::cond, etc.). The whole module is
// already declared `pub mod linalg;` above.
pub use dispatch::{DispatchKey, DispatchKeySet, Dispatcher, Kernel};
pub use fft::{
    fft, fft2, fftfreq, fftn, fftshift, hfft, ifft, ifft2, ifftn, ifftshift, ihfft, irfft, irfftn,
    rfft, rfftfreq, rfftn,
};
pub use flex_attention::flex_attention;
pub use grad_fns::activation::{GeluApproximate, gelu, gelu_with, sigmoid, tanh};
pub use grad_fns::cumulative::{cummax, cummin, cumprod, cumsum, logcumsumexp};
pub use grad_fns::fft::{
    fft_differentiable, ifft_differentiable, irfft_differentiable, rfft_differentiable,
};
pub use grad_fns::quantize_grad::fake_quantize_differentiable;
pub use grad_fns::reduction::{mean_dim, sum_dim};
pub use grad_fns::shape::{cat, expand};
pub use grad_fns::transcendental::{clamp, cos, exp, log, sin};
pub use masked::{
    MaskedTensor, masked_count, masked_equal, masked_invalid, masked_max, masked_mean, masked_min,
    masked_sum, masked_where,
};
pub use methods::{chunk_t, contiguous_t, permute_t, split_t, view_t};
pub use nested::{NestedTensor, PackedNestedTensor, nested_scaled_dot_product_attention};
pub use ops::cumulative::CumExtremeResult;
pub use ops::indexing::{gather, scatter, scatter_add, where_cond};
pub use ops::search::{bucketize, histc, meshgrid, searchsorted, topk, unique, unique_consecutive};
pub use ops::tensor_ops::{cdist, diag, diagflat, roll, tril, triu};
pub use pruning::{apply_2_4_mask, magnitude_prune, sparsity_ratio};
pub use quantize::{
    FakeQuantize, HistogramObserver, MinMaxObserver, Observer, PerChannelMinMaxObserver, QParams,
    QatLayer, QatModel, QuantDtype, QuantScheme, QuantizedTensor, cuda_rng, dequantize,
    prepare_qat, quantize, quantize_named_tensors, quantized_matmul,
};
pub use shape::{broadcast_shapes, normalize_axis};
pub use sparse::{CooTensor, CscTensor, CsrTensor, SparseGrad, SparseTensor};
pub use sparse::{SemiStructuredSparseTensor, sparse_matmul_24};
pub use special::{digamma, erf, erfc, erfinv, expm1, lgamma, log1p, sinc, xlogy};
pub use storage::{StorageBuffer, TensorStorage};
pub use stride_tricks::{AsStridedBackward, as_strided, as_strided_copy, as_strided_scatter};
pub use tensor::{GradFn, MemoryFormat, Tensor, TensorId};
pub use vmap::{select, stack, vmap, vmap2};
