//! Vision models, datasets, and image transforms for ferrotorch.
//!
//! Mirrors the surface of `torchvision`: image-classification model
//! constructors and a global registry, image / synthetic-image datasets
//! (`MNIST`, `CIFAR-10/100`, `ImageFolder`), and per-tensor augmentation
//! transforms. Float-typed tensors flow through every public API.

#![warn(clippy::all, clippy::pedantic)]
#![deny(unsafe_code, rust_2018_idioms)]
// `missing_debug_implementations` is allowed at the crate level while the
// crate-wide Debug-derive pass is tracked as a follow-up (mirrors jit #677
// for missing_docs). The pub sample types `RawImage`, `MnistSample`,
// `CifarSample`, `ImageSample` and the dataset wrappers `Cifar10`,
// `Cifar100`, `Mnist` already derive Debug — the remaining ~44 sites are
// internal model blocks (encoder/decoder structs etc.) and would be
// churn-only mechanical changes. Flip to deny once that sweep lands.
#![allow(missing_debug_implementations)]
// Pedantic lints we explicitly accept across this crate. Each allow names a
// concrete reason — the alternative would be churn-for-zero-benefit or a
// worse API. Add to this list only with a one-line justification.
#![allow(
    // Vision types deliberately echo their parent module name (e.g.
    // `RandomGaussianBlur` in `random_gaussian_blur`) to mirror torchvision.
    clippy::module_name_repetitions,
    // # Errors / # Panics rustdoc sections will be added during the next
    // docs pass; not gated on this lint baseline.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Image kernels need explicit casts between usize/i64/f64 for index
    // arithmetic and pixel scaling; rewriting hides the intent.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    // Long `match` blocks in transforms (e.g. TrivialAugmentWide ops) and
    // model definitions mirror the underlying taxonomy 1:1.
    clippy::too_many_lines,
    // `#[must_use]` on every getter is churn; callers in this codebase
    // already use the returned values.
    clippy::must_use_candidate,
    // `let ... else { return }` is sometimes less readable than the
    // `match { Some(x) => x, _ => return }` form transforms use.
    clippy::manual_let_else,
    // Helper fns appear after `let`-bindings inside transforms; hoisting is
    // style-only.
    clippy::items_after_statements,
    // Dataset / augmentation tests use small literal probability and
    // pixel-range constants (e.g. 0.5, 1e-6) where the underscored form
    // gains nothing.
    clippy::unreadable_literal,
    // Builder-style `Self`-returning configs already document their
    // consume-and-return pattern; `#[must_use]` is noise.
    clippy::return_self_not_must_use,
    // `iter().map(...).collect::<Result<_, _>>()?` is idiomatic in cast
    // pipelines; rewriting to imperative loops obscures intent.
    clippy::needless_collect,
    // The rustdoc-cleanliness pass is tracked as a follow-up; many existing
    // doc comments contain unbacktick-quoted identifiers (`Tensor` etc.).
    clippy::doc_markdown,
    // Existing format strings use the positional `{}` form pervasively;
    // sweep to inline-args is its own follow-up.
    clippy::uninlined_format_args,
    // Vision tests use small loop counters cast to f64; the explicit `as`
    // mirrors PyTorch test idioms. Sweep to `f64::from(i)` is follow-up.
    clippy::cast_lossless,
    // Loop variables `i, j, k, c, h, w, b` are the canonical kernel-index
    // names matching tensor-shape conventions in this codebase.
    clippy::similar_names,
    clippy::many_single_char_names,
    // `iter().map(...).unwrap_or(...)` is more readable than `map_or` in
    // some sites and equivalent semantically.
    clippy::map_unwrap_or,
    // Explicit `for x in collection.iter()` is sometimes clearer than the
    // implicit `for x in &collection`; both are idiomatic.
    clippy::explicit_iter_loop,
    // Float-equality comparisons in tests check for literal sentinels
    // (e.g. `x == 0.0` for "input is exactly zero"); they're intentional.
    clippy::float_cmp,
    // `use ferrotorch_data::*` in tests is intentional for ergonomics.
    clippy::enum_glob_use,
    // Builder fns may take a value argument and return it; the lint flags
    // some of those false-positively in transform constructors.
    clippy::needless_pass_by_value,
    // Trailing commas after `format!` args are stylistic and don't change
    // semantics; the rustfmt-rules of this workspace already manage them.
    clippy::trailing_empty_array,
)]
// `missing_docs` is held at warn while the crate-wide rustdoc pass is
// tracked as a follow-up; flip to deny once that lands (mirrors jit #677).
#![warn(missing_docs)]
#![allow(missing_docs)]

pub mod datasets;
pub mod io;
pub mod models;
pub mod ops;
pub mod transforms;

pub use datasets::{Cifar10, Cifar100, CifarSample, Mnist, MnistSample, Split};
pub use io::{
    RawImage, raw_image_to_tensor, read_image, read_image_as_tensor, read_image_rgba,
    tensor_to_raw_image, write_image, write_tensor_as_image,
};
pub use models::{
    FeatureExtractor, ModelConstructor, ModelRegistry, create_feature_extractor, get_model,
    list_models, register_model,
};
pub use transforms::{
    CenterCrop, ColorJitter, IMAGENET_MEAN, IMAGENET_STD, RandomApply, RandomChoice,
    RandomGaussianBlur, RandomResizedCrop, RandomRotation, RandomVerticalFlip, Resize,
    VisionNormalize, VisionToTensor, vision_manual_seed,
};
