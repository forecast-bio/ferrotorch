// Crate-level lint baseline. Mirrors the workspace-wide rust-quality
// posture: deny correctness/idiom/Debug/docs problems; warn pedantic
// stylistic issues. Specific pedantic lints are allowed crate-wide
// where the lint is consistently wrong for ML/numeric kernel code —
// each allow names the reason it's noise here rather than signal.

// Correctness / hygiene — these are real bugs if they fire.
#![deny(unsafe_code)]
#![deny(rust_2018_idioms)]
#![deny(missing_debug_implementations)]
#![deny(missing_docs)]
// Style baseline.
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
// Pedantic lints allowed crate-wide. Each of these has a reason that
// repeats in this crate at scale; allow at the lint level (not at
// `clippy::pedantic` group level) so we still catch the rest.
//
// Casts: dimension math (`as usize`, `as f32`, `as u32`) is intrinsic
// to tensor indexing and bf16 ↔ f32 conversion — every kernel call
// would otherwise need a per-call allow. Lint fires ~150x in the crate
// with no actionable signal.
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_lossless)]
// `must_use_candidate` would flag every getter on every public struct.
// We use `Result` returns where misuse is observable; the warning is
// noise for infallible accessors.
#![allow(clippy::must_use_candidate)]
// Long kernel-dispatch routines (gpu.rs forward_core) are a single
// linear pipeline; splitting them yields no clarity. Fires ~6 times.
#![allow(clippy::too_many_lines)]
// Identifiers like `bf16`, `f32`, `RoPE`, `cuBLAS` are flagged as
// missing backticks even when they appear in code-fenced text. The
// lint's heuristic produces false positives in this crate's docs.
#![allow(clippy::doc_markdown)]
// Triggers on `if !x { ... } else { ... }` patterns that are clearer
// in context (e.g. error-path-first matches the audit-finding shape).
#![allow(clippy::if_not_else)]
// `explicit_iter_loop` flags `for x in v.iter()` where `for x in &v`
// is supposedly clearer; the explicit form is more discoverable here.
#![allow(clippy::explicit_iter_loop)]
// `items_after_statements` flags the kernel-dispatch helper structs
// declared inside `forward_core`-shaped functions; co-locating them
// with the only caller is the more readable choice.
#![allow(clippy::items_after_statements)]
// `match_same_arms` collapses arms whose bodies happen to match but
// whose meanings differ (different error categories with the same
// message format).
#![allow(clippy::match_same_arms)]
// `match_wildcard_for_single_variants` requires enumerating
// `#[non_exhaustive]` enums explicitly; with `FerrotorchError` and
// `GpuError` both `#[non_exhaustive]` upstream the wildcard is the
// future-proof match.
#![allow(clippy::match_wildcard_for_single_variants)]
// `return_self_not_must_use` flags every builder-style method; we
// use `must_use` selectively where it actually matters.
#![allow(clippy::return_self_not_must_use)]
// `redundant_closure_for_method_calls` would fail on every
// `.map(|e| e.to_string())` style call; rewriting to method ref
// is purely cosmetic.
#![allow(clippy::redundant_closure_for_method_calls)]
// `manual_let_else` flags `match … { Some(x) => x, None => return … }`
// patterns that pre-date `let-else`; the explicit form is fine here.
#![allow(clippy::manual_let_else)]
// `needless_pass_by_value` would force `&Config` and `&Tensor<T>`
// signatures throughout, hiding ownership transfer in the API.
#![allow(clippy::needless_pass_by_value)]
// `map_unwrap_or` and `option_if_let_else` favour combinators that
// hurt readability when the branches are non-trivial.
#![allow(clippy::map_unwrap_or)]
// `float_cmp` fires on epsilon-free comparisons that are correct here
// (e.g. `temperature == 0.0` switches to greedy decoding — exact
// match against the sentinel is the contract).
#![allow(clippy::float_cmp)]
// `implicit_clone` fires on `String::from(s)` style conversions
// inside hot paths where the explicit form is clearer than `.clone()`.
#![allow(clippy::implicit_clone)]
// `ptr_as_ptr` flags numeric casts in bytemuck-backed reinterprets;
// the casts are correct and bytemuck verifies layout.
#![allow(clippy::ptr_as_ptr)]
// `unnecessary_wraps` flags `Result`-returning helpers that today
// always succeed but are part of an extensible API surface.
#![allow(clippy::unnecessary_wraps)]
// `uninlined_format_args` flags `format!("x={}", x)` vs `format!("x={x}")`.
// Both forms are equally clear and the fixup churn is high.
#![allow(clippy::uninlined_format_args)]
// `unnested_or_patterns` flags `(A, X) | (B, X)` vs `(A | B, X)`. The
// nested form mirrors the pattern shape across the rest of the
// match in grammar/state.rs (every other arm is a 2-tuple of refs);
// rewriting collapses readability.
#![allow(clippy::unnested_or_patterns)]
// `needless_continue` flags a redundant `continue` at the tail of a
// loop branch; some sites use it for symmetry with parallel branches.
#![allow(clippy::needless_continue)]
// `needless_range_loop` flags index-driven loops where the body needs
// the index for parallel arrays (`cos_bits`, `sin_bits` etc.). The
// `enumerate()` rewrite obscures the intent.
#![allow(clippy::needless_range_loop)]
// `similar_names` flags variable pairs like `walk_l_state` and
// `walk_u_state` (lower / upper bounds in the DFA construction) — the
// similarity is the *point* and the name is shorter than any
// disambiguator the lint would prefer.
#![allow(clippy::similar_names)]
// `many_single_char_names` flags conventional ML kernel locals
// (`q`, `k`, `v` for query/key/value, `t` for time/token, `h` for
// hidden) where the convention is the documentation.
#![allow(clippy::many_single_char_names)]
// `doc_link_with_quotes` flags identifiers in single-quoted
// terminator examples (`',', ']'`) as missing intra-doc-link
// backticks. Char literals are not intra-doc links.
#![allow(clippy::doc_link_with_quotes)]

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
