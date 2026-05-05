// Lint baseline mirrors `ferrotorch-gpu/src/lib.rs` and the wider workspace:
// `clippy::all + pedantic` are warned, `rust_2018_idioms` and
// `missing_debug_implementations` are denied, `missing_docs` is held at
// `allow` while the workspace-wide rustdoc pass is tracked separately
// (#731 follow-up). Diverging unilaterally from a leaf crate would be
// Step 4 architectural unilateralism.
#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms, missing_debug_implementations)]
#![allow(missing_docs)]
// tracked workspace-wide rustdoc pass
// Pedantic lints we explicitly accept across this crate. Each allow names a
// concrete reason — the alternative would be churn-for-zero-benefit or a
// worse API. Mirrors the ferrotorch-gpu / ferrotorch-jit baselines; add to
// this list only with a one-line justification.
#![allow(
    // Wrapper fns intentionally re-export the ferrolearn metric / dataset
    // names so the API maps 1:1 onto sklearn's surface; renaming would
    // make the bridge harder to use.
    clippy::module_name_repetitions,
    // # Errors / # Panics sections will be added during the workspace-wide
    // rustdoc pass tracked as a follow-up issue (#731), not gated on this
    // lint baseline. The four adapter fns named in the audit are filled in
    // explicitly below.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // `#[must_use]` on every wrapper is churn for marginal value; callers
    // already use the returned values (the fns return `Result<Tensor<T>>`
    // and discarding it makes no sense).
    clippy::must_use_candidate,
    // ML code casts pervasively between integer/float widths around label
    // encodings (`usize` ↔ `f64`) and ranking indices; the explicit cast is
    // more readable than try_into / num-traits indirection. Mirrors the
    // ferrotorch-gpu / ferrotorch-jit precedent.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    // Adapter fns take ownership of `Array1<usize>` etc. by value because
    // they pass the data to ferrolearn / Tensor::from_storage which
    // benefits from owned storage; switching to `&Array1<usize>` would
    // force an extra clone and is the wrong shape for the API.
    clippy::needless_pass_by_value,
    // Doc-markdown's autocorrect insists on backtick-wrapping every
    // sklearn / ferrolearn type name (PCA, KFold, OneHotEncoder, …);
    // the prose reads better unmarked and matches the upstream rustdoc.
    clippy::doc_markdown,
    // Tests assert exact label values (0.0 / 1.0 / 2.0 from integer
    // class encodings) where strict equality is intentional and a
    // tolerance check would obscure the property under test.
    clippy::float_cmp,
    // `as f64` on small literal integers is the idiomatic shape; the
    // `From`-based migration adds noise to test setup without value.
    clippy::cast_lossless,
)]

//! Sklearn-compatible adapter for ferrotorch.
//!
//! Bridges [`ferrotorch_core::Tensor`] to `ndarray::{Array1, Array2}` so
//! ferrolearn (a scikit-learn equivalent in Rust) can be driven from a
//! tensor-shaped pipeline.
//!
//! See the crate-level [`README`](https://crates.io/crates/ferrotorch-ml)
//! for the full design rationale. The short version:
//!
//! - [`adapter`] — `Tensor ↔ ndarray` round-trip primitives.
//! - [`metrics`] — sklearn classification + regression metrics shaped
//!   for `&Tensor<T>` inputs.
//! - [`datasets`] — built-in toy + synthetic generators returning tensor
//!   pairs.
//! - [`preprocess`], [`decomposition`], [`model_selection`] — direct
//!   re-exports from ferrolearn so the full sklearn surface is one
//!   import away.
//!
//! # CPU-only by design, GPU input transparently materialised
//!
//! ferrolearn is a CPU-only library (built on `ndarray` + `faer`).
//! `ferrotorch-ml` accepts tensors on **any** device — if the input
//! lives on CUDA / XPU, the data is moved to host memory before
//! conversion. This matches the torch idiom of `loss.cpu().item()`
//! where a single materialisation step crosses the device boundary.
//!
//! Compute crates (`ferrotorch-core`, `-nn`, `-gpu`) continue to enforce
//! the strict `/rust-gpu-discipline` no-silent-fallback rule. The
//! relaxation here applies **only** to this dedicated bridge crate —
//! the function names (`tensor_to_array2`, etc.) make the device
//! crossing self-evident at the call site.
//!
//! If you want fail-fast strictness in a hot path, add an explicit
//! `assert!(t.device().is_cpu())` before calling the adapter.

pub mod adapter;
pub mod datasets;
pub mod metrics;

// Direct re-exports of ferrolearn modules. These are CPU-only and
// operate on `ndarray::{Array1, Array2}` — pair them with `adapter::*`
// to round-trip from `Tensor`.
pub mod preprocess {
    //! Re-export of [`ferrolearn_preprocess`] — sklearn-style fit/transform
    //! preprocessors (`StandardScaler`, `MinMaxScaler`, `OneHotEncoder`,
    //! `PolynomialFeatures`, `KBinsDiscretizer`, `SimpleImputer`, etc.).
    //! Drive these from `Tensor` inputs by routing through
    //! [`super::adapter::tensor_to_array2`] and back via
    //! [`super::adapter::array2_to_tensor`].
    pub use ferrolearn_preprocess::*;
}

pub mod decomposition {
    //! Re-export of [`ferrolearn_decomp`] — dimensionality reduction
    //! (PCA, IncrementalPCA, FastICA, NMF, KernelPCA, t-SNE, UMAP,
    //! Isomap, LLE, MDS, FactorAnalysis, TruncatedSVD, SparsePCA).
    pub use ferrolearn_decomp::*;
}

pub mod model_selection {
    //! Re-export of [`ferrolearn_model_sel`] — cross-validation splitters
    //! (`KFold`, `StratifiedKFold`, `GroupKFold`, `ShuffleSplit`,
    //! `train_test_split`), pipelines (`make_pipeline`, `Pipeline`),
    //! grid search (`GridSearchCV`, `RandomizedSearchCV`), and the
    //! dummy / multiclass / multioutput meta-estimators.
    pub use ferrolearn_model_sel::*;
}
