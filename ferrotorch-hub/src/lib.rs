//! Pretrained model registry, download, and caching for ferrotorch.
//!
//! This crate provides a central hub for discovering, downloading, and caching
//! pretrained model weights. It mirrors the workflow of `torch.hub` and
//! `torchvision.models` with pretrained weight support.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use ferrotorch_core::FerrotorchError;
//! use ferrotorch_hub::{list_models, load_pretrained};
//!
//! fn main() -> Result<(), FerrotorchError> {
//!     // Browse available models.
//!     for model in list_models() {
//!         println!("{}: {} ({} params)", model.name, model.description, model.num_parameters);
//!     }
//!
//!     // Load pretrained weights (requires cached weights on disk).
//!     let _state_dict = load_pretrained::<f32>("resnet50")?;
//!     Ok(())
//! }
//! ```

// Lint baseline mirrors the workspace pattern (see ferrotorch-core,
// ferrotorch-jit, ferrotorch-ml). `missing_docs` and
// `missing_debug_implementations` are held at `allow` while the
// workspace-wide rustdoc / Debug pass is tracked separately — diverging
// unilaterally from a leaf crate would be Step 4 architectural
// unilateralism. `unsafe_code` is intentionally NOT denied: the test
// modules need `unsafe` for env::set_var (Rust 2024) and per-block
// SAFETY substantiation lives at each `unsafe { ... }` site.
#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms)]
#![allow(missing_docs, missing_debug_implementations)]
// Pedantic lints we explicitly accept across this crate. Each allow
// names a concrete reason — the alternative would be churn-for-zero-
// benefit or a worse API. Mirrors the ferrotorch-core / -jit baselines;
// add to this list only with a one-line justification.
#![allow(
    // Hub types (HubCache, HfModelInfo, etc.) intentionally repeat the
    // crate name; renaming them would weaken the API.
    clippy::module_name_repetitions,
    // # Errors / # Panics rustdoc sections are added per-fn where they
    // matter; a crate-wide deny would force noisy boilerplate on every
    // tiny helper.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // `#[must_use]` on every getter is churn for marginal value.
    clippy::must_use_candidate,
    // Builder-style methods on SearchQuery already document their
    // consume-and-return pattern; `#[must_use]` is noise.
    clippy::return_self_not_must_use,
    // Doc comments mention paths like `~/.cache/huggingface/token` and
    // bare HF URLs that don't gain readability from backticks; the
    // `doc_markdown` pedantic rule is too aggressive for technical prose.
    clippy::doc_markdown,
    // `format!(..)` appended to existing `String` is the natural
    // shape for the discovery query-string builder; rewriting to
    // `write!(s, ...).unwrap()` introduces a fallible path that
    // genuinely can't fail and obscures the build pattern.
    clippy::format_push_string,
    // Test/helper closures define small fns after `let`-bindings inside
    // `hf_download_model`; hoisting them would split the helper from
    // its only caller.
    clippy::items_after_statements,
    // Tests use `format!("{:?}", path)` to surface PathBuf state for
    // diagnostics; uninlining harms readability of test failure output.
    clippy::uninlined_format_args,
    // PathBuf doesn't implement Display; `{:?}` is the conventional
    // formatting for path errors and matches the rest of the workspace.
    clippy::unnecessary_debug_formatting,
    // Test fixtures and example HF download counts (e.g. 1234567)
    // intentionally use round numbers without underscore separators;
    // they're test data, not code that needs to be visually parsed.
    clippy::unreadable_literal,
    // `hf_download_model` mirrors the HF API call sequence (config →
    // index → shards → tokenizer); splitting reduces traceability against
    // the spec at <https://huggingface.co/docs/hub/api>.
    clippy::too_many_lines,
)]

#[cfg(feature = "http")]
pub mod auth;
pub mod cache;
#[cfg(feature = "http")]
pub mod discovery;
pub mod download;
pub mod hf_config;
pub mod registry;

#[cfg(feature = "http")]
pub use auth::{hf_token, with_auth};
pub use cache::{HubCache, default_cache_dir};
#[cfg(feature = "http")]
pub use discovery::{
    HfModelInfo, HfModelSummary, HfRepoFile, SearchQuery, get_model, search_models,
};
#[cfg(feature = "http")]
pub use download::hf_download_model;
pub use download::{download_weights, load_pretrained};
pub use hf_config::HfTransformerConfig;
pub use registry::{ModelInfo, WeightsFormat, get_model_info, list_models};
