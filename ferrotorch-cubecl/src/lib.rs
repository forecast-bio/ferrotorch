//! Portable GPU backend for ferrotorch via CubeCL.
//!
//! CubeCL compiles a single kernel definition to CUDA PTX, AMD HIP/ROCm, and
//! WGPU (Vulkan/Metal/DX12). This crate wraps CubeCL's runtime and dispatches
//! real `#[cube]` kernels to the active backend — no CPU fallbacks.
//!
//! # Feature flags
//!
//! | Feature | Backend              | GPU vendors            |
//! |---------|----------------------|------------------------|
//! | `cuda`  | NVIDIA CUDA via PTX  | NVIDIA                 |
//! | `wgpu`  | WGPU (Vulkan/Metal)  | AMD, Intel, Apple, ... |
//! | `rocm`  | AMD HIP (native)     | AMD                    |
//!
//! Enable at least one backend feature to use GPU acceleration. Without any
//! backend feature [`CubeRuntime::new`] returns
//! `FerrotorchError::DeviceUnavailable` and [`CubeRuntime::auto`] returns
//! `None`.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrotorch_cubecl::{CubeDevice, CubeRuntime};
//!
//! // Auto-detect the best available backend
//! if let Some(rt) = CubeRuntime::auto() {
//!     println!("Using device: {:?}", rt.device());
//! }
//! ```

#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms, missing_debug_implementations)]
// Rustdoc coverage is being swept workspace-wide in a separate dispatch
// (tracked workspace-wide rustdoc pass); matches the gpu / jit precedent
// until that lands.
#![allow(missing_docs)]
// Pedantic lints we explicitly accept across this crate. Each allow names a
// concrete reason — the alternative would be churn-for-zero-benefit, a
// worse API, or scope-creep into frozen files (storage.rs, runtime.rs).
// Mirrors the ferrotorch-gpu / ferrotorch-jit baseline; add only with a
// one-line justification.
#![allow(
    // Doc prose includes `CubeCL`, `GGUF`, `Q4_0`, etc. — surrounding every
    // such word in backticks would hurt readability for technical prose.
    clippy::doc_markdown,
    // # Errors / # Panics sections will be added in the workspace-wide
    // rustdoc pass; not gated on this lint baseline.
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Numeric ML code casts pervasively between `usize` and `u32` for buffer
    // sizes, dimensions, and CubeCL launch arithmetic; explicit `as` is more
    // readable than `try_into().unwrap()` cluttering hot paths.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    // `#[must_use]` on every getter is churn for marginal value; existing
    // callers already use the returned values.
    clippy::must_use_candidate,
    // Math kernels naturally use single-character names (m, k, n for matmul
    // dims; a, b for binary operands); requiring longer names hurts
    // readability.
    clippy::many_single_char_names,
    // Pre-existing pedantic warnings in this crate's frozen files
    // (storage.rs, runtime.rs) and the dataplane-heavy quant.rs / kernels.rs
    // bodies are tracked for the cubecl-B SAFETY substantiation pass and a
    // workspace-wide rustdoc / format-args sweep. Allowing them keeps
    // `-D warnings` viable now without scope-creeping into frozen files.
    clippy::ptr_as_ptr,
    clippy::uninlined_format_args,
)]

pub mod grammar;
pub mod kernels;
pub mod ops;
pub mod quant;
pub mod runtime;
pub mod storage;

// ---------------------------------------------------------------------------
// Crate-internal helpers shared across modules
// ---------------------------------------------------------------------------

/// Choose a 1-D cube count and cube dim that cover `n` elements when each
/// unit processes exactly one element.
///
/// 256 units per cube is a safe default across all backends (wgpu, cuda,
/// rocm). Returned as a tuple ready to feed into `kernel::launch_unchecked`.
///
/// Callers: [`kernels`], [`quant`], [`grammar`]. Previously duplicated
/// verbatim across all three modules; consolidated here so launch geometry
/// stays consistent.
pub(crate) fn elementwise_launch_dims(
    n: u32,
) -> (cubecl::prelude::CubeCount, cubecl::prelude::CubeDim) {
    let units_per_cube: u32 = 256;
    let num_cubes = n.div_ceil(units_per_cube).max(1);
    (
        cubecl::prelude::CubeCount::Static(num_cubes, 1, 1),
        cubecl::prelude::CubeDim::new_1d(units_per_cube),
    )
}

/// Debug-build runtime check that a cubecl `Handle` has at least
/// `n * size_of::<T>()` bytes capacity. Release builds elide via
/// `debug_assert!`. Use before `ArrayArg::from_raw_parts(handle, n)`
/// for caller-provided handles where the cubecl-side `unsafe` API
/// requires the byte capacity to match.
///
/// `T` carries the kernel-side element type (e.g. `f32`) so that the
/// byte stride is computed from `size_of::<T>()`. Closes #717 / #718.
pub(crate) fn debug_assert_handle_capacity<T>(handle: &cubecl::server::Handle, n: usize) {
    debug_assert!(
        handle.size() as usize >= n.saturating_mul(std::mem::size_of::<T>()),
        "cubecl handle capacity {} bytes < required {} bytes ({} elements x {} byte stride)",
        handle.size(),
        n.saturating_mul(std::mem::size_of::<T>()),
        n,
        std::mem::size_of::<T>(),
    );
}

// Re-export runtime types.
pub use runtime::{CubeClient, CubeDevice, CubeRuntime};

// Re-export storage handle types and upload/wrapping helpers.
pub use storage::{CubeclStorageHandle, cubecl_handle_of, upload_f32, wrap_kernel_output};

// Re-export quantized-weight dequantization API.
pub use quant::{
    GgufBlockKind, dequantize_q4_0_to_gpu, dequantize_q4_1_to_gpu, dequantize_q5_0_to_gpu,
    dequantize_q5_1_to_gpu, dequantize_q8_0_to_gpu, dequantize_q8_1_to_gpu, split_q4_0_blocks,
    split_q4_1_blocks, split_q5_0_blocks, split_q5_1_blocks, split_q8_0_blocks, split_q8_1_blocks,
};

// Re-export GPU constrained-decoding token-mask compute API.
pub use grammar::{DfaMaskInputs, compute_token_mask_dfa_to_gpu, kernel_compute_token_mask_dfa};
