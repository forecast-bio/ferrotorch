//! Training loop, metrics, and callbacks for ferrotorch.
//!
//! This crate provides the [`Learner`] abstraction that ties together a model,
//! optimizer, loss function, metrics, and callbacks into a complete training
//! pipeline.
//!
//! # Overview
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`Learner`] | High-level training loop: `fit()`, `evaluate()` |
//! | [`Metric`] | Accumulate batch-level values into epoch-level summaries |
//! | [`Callback`] | Hook into epoch/batch boundaries (early stopping, logging, EMA) |
//! | [`TrainingHistory`] | Record of per-epoch results |
//! | [`checkpoint`] | Gradient checkpointing: recompute forward in backward |
//! | [`amp`] | Automatic mixed precision: autocast + GradScaler |
//! | [`clip_grad_norm_`] | Clip total gradient norm across parameters |
//! | [`clip_grad_value_`] | Clamp individual gradient elements |
//!
//! # Quick start
//!
//! Constructing common training plumbing without running an actual `fit()`
//! (which needs a model + dataset) — these examples are runnable as
//! doctests:
//!
//! ```
//! use ferrotorch_train::{
//!     Callback, EarlyStopping, EmaCallback, LossMetric, Metric, ProgressLogger,
//!     TrainingHistory,
//! };
//!
//! // Per-epoch metric (accumulates loss values, reports a mean).
//! let mut metric = LossMetric::new();
//! metric.update(&0.5);
//! metric.update(&0.3);
//! assert!((metric.compute() - 0.4).abs() < 1e-9);
//!
//! // Early-stopping callback: triggers after `patience` no-improvement epochs.
//! let es = EarlyStopping::new(5, 0.001);
//! assert!(!Callback::<f32>::should_stop(&es));
//!
//! // Progress logger emits via `tracing::info!` — install a subscriber to see output.
//! let _logger = ProgressLogger::new(100);
//!
//! // EMA callback for tracking shadow parameters.
//! let _ema = EmaCallback::new(0.999);
//!
//! // History accumulates `EpochResult`s pushed by `Learner::fit`.
//! let history = TrainingHistory::new();
//! assert!(history.is_empty());
//! ```
//!
//! A complete `Learner::new(...).fit(...)` example requires a model,
//! optimizer, loss function, and data iterator and is therefore omitted
//! from the doctest harness; see `learner.rs`'s `# Examples` for the full
//! shape (marked `ignore` because the test environment has no model).
//!
//! [CL-334] Add gradient checkpointing, autocast context, gradient clipping, and EMA callback

// Lint baseline mirrors the workspace-standard pattern from
// `ferrotorch-core`/`-distributed`/`-jit`/`-cubecl`/`-xpu`/`-nn` lib.rs.
#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms)]
// `missing_docs` and `missing_debug_implementations` are held at `allow`
// while the workspace-wide rustdoc / `Debug` pass is tracked separately
// (matches the existing precedent in sibling crates — diverging
// unilaterally from a leaf crate would be Step 4 architectural
// unilateralism). Several public types in this crate hold trait objects
// (`Box<dyn Optimizer<T>>`, `Box<dyn LrScheduler<T>>`, `Box<dyn Callback<T>>`)
// whose `Debug` impls require careful hand-rolling.
#![allow(missing_docs, missing_debug_implementations)]
// Pedantic lints we explicitly accept across this crate. Each allow names
// a concrete reason; mirrors the ferrotorch-nn / ferrotorch-core baseline
// almost verbatim — diverging on the allow list from a leaf crate would
// be Step 4 architectural unilateralism. Add to this list only with a
// one-line justification.
#![allow(
    // Submodule names match the type they export (`callback::Callback`,
    // `metric::Metric`, `history::TrainingHistory`); renaming would force
    // ergonomic breakage.
    clippy::module_name_repetitions,
    // # Errors / # Panics sections are added as part of focused passes;
    // a blanket sweep is tracked separately. The high-leverage public
    // sites carry them already (Learner::fit / evaluate / load_checkpoint,
    // TensorBoardWriter::* and TensorBoardCallback::new).
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Training-loop code casts pervasively between integer-typed counters
    // (`usize` epoch / step / batch) and floating-point summaries (`f64`
    // loss, lr, duration). Routing every cast through `num_traits::cast`
    // would obscure rather than illuminate the arithmetic.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    // `#[must_use]` on every getter / builder-setter is churn for marginal
    // value; callers in this codebase already use the returned values.
    clippy::must_use_candidate,
    // Builder-style methods returning `Self` document their pattern in
    // the type signature; `#[must_use]` is noise.
    clippy::return_self_not_must_use,
    // Doc comments follow the standard rustdoc layout; pedantic
    // doc-markdown rules are too aggressive for the technical prose
    // (PyTorch op names, math notation, file_version / wall_time).
    clippy::doc_markdown,
    // Test/helper modules define small fns after `let`-bindings; the
    // hoisting requirement is style-only.
    clippy::items_after_statements,
    // The training loop in `Learner::fit` is naturally a single function
    // mirroring PyTorch's `Trainer.fit`; splitting reduces legibility.
    clippy::too_many_lines,
    // The `Float` generic appears throughout this crate and propagates
    // to most public functions; suppressing the bound noise keeps
    // signatures readable.
    clippy::needless_pass_by_value,
    // `if let Some(x) = y { ... } else { ... }` is the idiomatic Rust
    // shape; the lint's preferred ladders are less readable.
    clippy::option_if_let_else,
    // Trivial getters return references to small `Copy` fields; the
    // suggested change to direct field exposure would conflict with the
    // pub-field-via-non_exhaustive policy used by `EpochResult` etc.
    clippy::trivially_copy_pass_by_ref,
    // `match` on small enums is more legible than chained `if let`.
    clippy::single_match_else,
    // `let ... else { return }` rewrites of `match { Some(x) => x, None => ... }`
    // are often less readable when the match arm is the natural pattern.
    clippy::manual_let_else,
    // Format-string inlining sweep is tracked separately per the
    // workspace precedent in ferrotorch-core / -distributed / -nn.
    clippy::uninlined_format_args,
    // `.collect::<Vec<_>>()` after mapping is the idiomatic shape;
    // rewriting to `extend(map(..))` is lossier.
    clippy::redundant_closure_for_method_calls,
    // `for i in 0..n { ... }` is natural when the index itself is needed.
    clippy::needless_range_loop,
    // `(a + b) / 2` idiom appears in interpolation; the `midpoint`
    // intrinsic clippy suggests has subtle precision differences.
    clippy::manual_midpoint,
    // `map(...).unwrap_or(...)` is a perfectly clear idiom.
    clippy::map_unwrap_or,
    // Bind-to-_ patterns appear in tests where the `_` is intentional.
    clippy::no_effect_underscore_binding,
    // `if let Some(x) = ... { ... } else { ... }` is idiomatic for
    // biased Option matching; clippy's `else` rewrites are noisier.
    clippy::if_not_else,
    // `for x in arr.iter() { ... }` is the explicit form some readers
    // prefer; the rewrite to `for x in arr` is a style-only refactor.
    clippy::explicit_iter_loop,
    // `if cond { return ...; } else { ... }` with the else block holding
    // the natural fall-through is a deliberate style choice.
    clippy::redundant_else,
    // `Default::default()` on a workspace type-alias (`OptimizerState`)
    // is the natural shape; the suggested `HashMap::default()` rewrite
    // leaks the alias's underlying type.
    clippy::default_trait_access,
    // `&str` returned from accessors on owned-string fields is the
    // idiomatic API; the lint's preferred `.as_str()` ladder adds noise.
    clippy::elidable_lifetime_names,
    // `match` over `&str` arms is more readable than chained `if-else`.
    clippy::single_match,
    // Builder-style chained setters consume `self` and return it; the
    // trailing comma at the end of a multi-line `tracing::info!` /
    // `tracing::warn!` arg list is part of the style guide.
    clippy::trailing_empty_array,
)]

pub mod amp;
pub mod callback;
pub mod checkpoint;
pub mod grad_utils;
pub mod history;
pub mod learner;
pub mod metric;
pub mod tensorboard;

pub use callback::{Callback, EarlyStopping, EmaCallback, ProgressLogger};
pub use checkpoint::{checkpoint, checkpoint_sequential};
pub use grad_utils::{clip_grad_norm_, clip_grad_value_};
pub use history::{EpochResult, EvalResult, TrainingHistory};
pub use learner::{Learner, LossFn};
pub use metric::{AccuracyMetric, LossMetric, Metric, RunningAverage, TopKAccuracy};
pub use tensorboard::{TensorBoardCallback, TensorBoardWriter};
