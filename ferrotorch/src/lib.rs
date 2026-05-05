//! ferrotorch — PyTorch-shaped deep learning framework in Rust.
//!
//! This crate is the umbrella re-export crate. Sub-crates own the actual
//! implementation; this crate exists so users can `use ferrotorch::*;` (or
//! `use ferrotorch::prelude::*;`) and pick up the canonical public surface
//! in one import.
//!
//! # Examples
//!
//! ```rust,no_run
//! use ferrotorch::{FerrotorchResult, zeros};
//!
//! fn main() -> FerrotorchResult<()> {
//!     let t = zeros::<f32>(&[2, 3])?;
//!     assert_eq!(t.shape(), &[2, 3]);
//!     Ok(())
//! }
//! ```
//!
//! See the `prelude` module for the items most users want, and the per-feature
//! modules (`nn`, `optim`, `data`, `vision`, `train`, `serialize`, `jit`,
//! `distributions`, `profiler`, `hub`, `tokenize`, `gpu`, `cubecl`, `mps`,
//! `xpu`, `distributed`, `llama`, `ml`) for sub-crate access.
//!
//! Lint baseline mirrors the per-crate convention used across the workspace
//! (`ferrotorch-core`, `ferrotorch-jit`, `ferrotorch-cubecl`, etc.). Workspace
//! `[lints]` is intentionally not used — every crate carries its own
//! `#![warn/deny(...)]` so the policy lives next to the code it governs.

#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms, missing_debug_implementations)]
// `missing_docs` is held off here because the umbrella crate is exclusively
// re-exports from already-documented sub-crates; a workspace-wide rustdoc
// pass is tracked separately and will lift this allow once it lands.
#![allow(missing_docs)]

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

pub use ferrotorch_core::*;

/// Prelude module — import everything commonly needed.
pub mod prelude {
    pub use ferrotorch_core::*;
    pub use ferrotorch_nn::{BatchNorm2d, Dropout, LayerNorm};
    pub use ferrotorch_nn::{Conv2d, Linear, Module, Parameter, Sequential};
    pub use ferrotorch_nn::{CrossEntropyLoss, MSELoss};
    pub use ferrotorch_nn::{GELU, ReLU, SiLU, Sigmoid, Softmax, Tanh};
    pub use ferrotorch_nn::{GRU, LSTM};
    pub use ferrotorch_optim::{Adam, AdamW, Optimizer, Sgd};
}

/// Neural network modules and layers.
pub mod nn {
    pub use ferrotorch_nn::*;
}

/// Optimizers and learning rate schedulers.
pub mod optim {
    pub use ferrotorch_optim::*;
}

/// Data loading, datasets, samplers, and transforms.
pub mod data {
    pub use ferrotorch_data::*;
}

/// Computer vision models, datasets, and transforms.
pub mod vision {
    pub use ferrotorch_vision::*;
}

/// Training loop, Learner, callbacks, and metrics.
#[cfg(feature = "train")]
pub mod train {
    pub use ferrotorch_train::*;
}

/// Model serialization: ONNX export, `PyTorch` import, safetensors, GGUF.
#[cfg(feature = "serialize")]
pub mod serialize {
    pub use ferrotorch_serialize::*;
}

/// JIT tracing, IR graph, optimization passes, and code generation.
#[cfg(feature = "jit")]
pub mod jit {
    pub use ferrotorch_jit::*;
}

/// `#[script]` proc macro for source-based graph capture.
#[cfg(feature = "jit-script")]
pub mod jit_script {
    pub use ferrotorch_jit_script::*;
}

/// Probability distributions for sampling and variational inference.
#[cfg(feature = "distributions")]
pub mod distributions {
    pub use ferrotorch_distributions::*;
}

/// Performance profiling and Chrome trace export.
#[cfg(feature = "profiler")]
pub mod profiler {
    pub use ferrotorch_profiler::*;
}

/// Model hub for downloading and caching pretrained models.
#[cfg(feature = "hub")]
pub mod hub {
    pub use ferrotorch_hub::*;
}

/// `HuggingFace` tokenizer wrapper (BPE, `WordPiece`, Unigram).
#[cfg(feature = "tokenize")]
pub mod tokenize {
    pub use ferrotorch_tokenize::*;
}

/// CUDA GPU backend with PTX kernels and cuBLAS.
#[cfg(feature = "gpu")]
pub mod gpu {
    pub use ferrotorch_gpu::*;
}

/// Portable GPU compute via `CubeCL` (CUDA + WGPU + `ROCm`).
#[cfg(feature = "cubecl")]
pub mod cubecl {
    pub use ferrotorch_cubecl::*;
}

/// Apple Silicon Metal Performance Shaders backend.
#[cfg(feature = "mps")]
pub mod mps {
    pub use ferrotorch_mps::*;
}

/// Intel Arc / Data Center GPU Max via `CubeCL` wgpu.
#[cfg(feature = "xpu")]
pub mod xpu {
    pub use ferrotorch_xpu::*;
}

/// Distributed training: DDP, collective ops, TCP backend.
#[cfg(feature = "distributed")]
pub mod distributed {
    pub use ferrotorch_distributed::*;
}

/// Llama 3 model composition and (with `llama-cuda`) GPU bf16 inference.
#[cfg(feature = "llama")]
pub mod llama {
    pub use ferrotorch_llama::*;
}

/// Sklearn-compatible adapter and classic-ML datasets.
#[cfg(feature = "ml")]
pub mod ml {
    pub use ferrotorch_ml::*;
}
