// Lint baseline mirrors `ferrotorch-core/src/lib.rs`. `missing_docs` and
// `missing_debug_implementations` are held at `warn` while the workspace-wide
// rustdoc / `Debug` pass is tracked as a follow-up issue (matches the existing
// `ferrotorch-core` precedent — diverging unilaterally from a leaf crate would
// be Step 4 architectural unilateralism). `unsafe_code` is intentionally NOT
// denied: this crate is fundamentally unsafe-using (PTX launches, raw pointer
// slices, FFI to cudarc); per-block SAFETY substantiation is tracked in the
// gpu-B..gpu-F dispatches.
#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms)]
// `missing_debug_implementations` is held at `allow` while the workspace-wide
// `Debug` follow-up is tracked separately. `missing_docs` flipped from `allow`
// to `deny` as part of the workspace-wide rustdoc pass (#703).
#![allow(missing_debug_implementations)]
#![deny(missing_docs)]
// Pedantic lints we explicitly accept across this crate. Each allow names a
// concrete reason — the alternative would be churn-for-zero-benefit or a
// worse API. Mirrors the ferrotorch-core baseline; add to this list only with
// a one-line justification.
#![allow(
    // `MpsDevice`/`GpuDevice`/`GpuTensor`/`GpuError`-style names intentionally
    // repeat the crate name — that's the API shape consumers expect.
    clippy::module_name_repetitions,
    // # Errors / # Panics sections will be added in the workspace-wide
    // rustdoc pass (tracked separately, not gated on this lint baseline).
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Long match-on-op blocks mirror the kernel taxonomy 1:1; splitting
    // reduces legibility.
    clippy::too_many_lines,
    // Numeric ML code casts pervasively between integer/float widths around
    // GPU buffer sizes, dimensions, and indexing; the explicit cast is more
    // readable than try_into/unwrap or num-traits indirection.
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_lossless,
    // `#[must_use]` on every getter is churn for marginal value; callers in
    // this codebase already use the returned values.
    clippy::must_use_candidate,
    // Builder-style methods returning `Self` document their pattern in the
    // type signature; `#[must_use]` is noise.
    clippy::return_self_not_must_use,
    // Math kernels naturally use single-character names (m, k, n for matmul
    // dims; i, j for indices); requiring longer names hurts readability.
    clippy::many_single_char_names,
    clippy::similar_names,
    // Doc comments follow the standard rustdoc layout; pedantic doc-markdown
    // rules are too aggressive for technical prose with PTX assembly.
    clippy::doc_markdown,
    // Hex-encoded constants in PTX templates and Philox round constants don't
    // gain readability from the underscore separators clippy prefers.
    clippy::unreadable_literal,
    // Test/helper modules define small fns after `let`-bindings; the
    // hoisting requirement is style-only.
    clippy::items_after_statements,
    // GPU trait methods often take many `&GpuBufferHandle` parameters that
    // mirror a kernel's input signature; each refactor is its own follow-up.
    clippy::too_many_arguments,
    // `let ... else { return }` rewrites of `match { Some(x) => x, None => return }`
    // are often less readable when the match arm is the natural pattern.
    clippy::manual_let_else,
    // `.collect::<Vec<_>>()` after mapping is the idiomatic shape; rewriting
    // to `extend(map(..))` is lossier and clippy's preference is contested.
    clippy::redundant_closure_for_method_calls,
    // Match arms that wrap a single variant of an enum and re-export are
    // intentional when the variant set is documented and the wrapper is part
    // of the API.
    clippy::match_wildcard_for_single_variants,
    // Parameter names `_a`, `_b` mirror tensor naming conventions
    // (left/right operand) and are not single-letter cargo-cult.
    clippy::single_match_else,
    // `for i in 0..n { ... }` over indices is the natural shape for kernel
    // launch math; .iter().enumerate() is needlessly indirect.
    clippy::needless_range_loop,
    // Manual `Debug` impls intentionally omit non-Debug fields like
    // `Box<dyn Fn>` callbacks, `cudarc` opaque handles, and `Mutex<...>`
    // contents to keep the formatted output useful and free of lock probes.
    clippy::missing_fields_in_debug,
    // Methods that take `&self` for a uniform interface (e.g., guard accessors
    // that are conceptually about the guard but don't read state) are part of
    // the public API shape and not refactor candidates from gpu-A.
    clippy::unused_self,
    // `.map(...).unwrap_or(...)` is the documented PyTorch-style fallback
    // shape used in the OOM recovery path; rewriting to `match` is lossier.
    clippy::map_unwrap_or,
    // PTX template strings, blas/solver test code, and Box<dyn Any>-erased
    // capture pools predate gpu-A's hygiene baseline; remaining pedantic
    // warnings (raw-pointer cast styles, `cloned` vs `copied`, trailing
    // commas in `assert!(.., "msg",)`, `if x { 1 } else { 0 }` patterns,
    // wildcard enum imports, `!=` simplifications, `format!` string interp,
    // strict-float-eq in identity-matmul tests) are tracked for the
    // gpu-B..gpu-F dispatches. Keeping `-D warnings` viable now while the
    // SAFETY substantiation work decides how to phrase those sites.
    clippy::ptr_as_ptr,
    clippy::ref_as_ptr,
    clippy::borrow_as_ptr,
    clippy::cast_ptr_alignment,
    clippy::bool_to_int_with_if,
    clippy::float_cmp,
    clippy::cloned_instead_of_copied,
    clippy::single_char_pattern,
    clippy::uninlined_format_args,
    clippy::wildcard_imports,
    clippy::enum_glob_use,
    clippy::if_not_else,
    clippy::needless_pass_by_value,
    clippy::assigning_clones,
    clippy::semicolon_if_nothing_returned,
    clippy::redundant_else,
    clippy::unnecessary_trailing_comma,
)]

//! CUDA GPU backend for ferrotorch.
//!
//! This crate provides device management, memory allocation, and host/device
//! data transfers built on [`cudarc`]. It is the bridge between ferrotorch's
//! CPU tensor world and NVIDIA GPUs.
//!
//! # Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `cuda`  | **yes** | Links against the CUDA driver API via cudarc. Disable to compile on machines without a GPU. |
//!
//! # Quick start
//!
//! ```rust,no_run
//! use ferrotorch_gpu::{GpuDevice, GpuError, cpu_to_gpu, gpu_to_cpu};
//!
//! fn main() -> Result<(), GpuError> {
//!     let device = GpuDevice::new(0)?;
//!     let host_data = vec![1.0_f32, 2.0, 3.0];
//!     let gpu_buf = cpu_to_gpu(&host_data, &device)?;
//!     let back = gpu_to_cpu(&gpu_buf, &device)?;
//!     assert_eq!(back, host_data);
//!     Ok(())
//! }
//! ```

pub mod allocator;
pub mod backend_impl;
#[cfg(feature = "cuda")]
pub mod bf16;
pub mod blas;
pub mod buffer;
pub mod conv;
#[cfg(feature = "cuda")]
pub mod cufft;
pub mod cusolver;
pub mod device;
pub mod error;
pub mod flash_attention;
pub mod graph;
pub mod kernels;
pub mod memory_guard;
pub mod module_cache;
pub mod pool;
pub mod rng;
pub mod stream;
pub mod tensor_bridge;
pub mod transfer;

// Re-exports for ergonomic use.
pub use allocator::CudaAllocator;
pub use backend_impl::{CudaBackendImpl, get_cuda_device, init_cuda_backend};
#[cfg(feature = "cuda")]
pub use bf16::{
    gpu_add_bf16, gpu_block_reduce_max_abs_bf16, gpu_causal_mask_bf16, gpu_embedding_gather_bf16,
    gpu_embedding_gather_bf16_to_f32, gpu_fatrelu_bf16, gpu_mul_bf16, gpu_relu_bf16,
    gpu_repeat_kv_bf16, gpu_rmsnorm_bf16, gpu_rope_half_bf16, gpu_scale_bf16, gpu_silu_bf16,
    gpu_softmax_bf16, gpu_transpose_from_heads_bf16, gpu_transpose_to_heads_bf16,
};
pub use blas::gpu_bmm_f32;
pub use blas::{gpu_bmm_f32_into, gpu_matmul_f32_into};
#[cfg(feature = "cuda")]
pub use blas::{
    gpu_matmul_bf16_bf16, gpu_matmul_bf16_bf16_nt, gpu_matmul_bf16_bf16_strided_batched,
    gpu_matmul_bf16_bf16_strided_batched_nt,
};
pub use blas::{gpu_matmul_f32, gpu_matmul_f64};
pub use buffer::CudaBuffer;
pub use conv::gpu_conv2d_f32;
pub use device::GpuDevice;
pub use error::{GpuError, GpuResult};
pub use flash_attention::gpu_flash_attention_f32;
pub use graph::{
    CaptureMode, CapturePool, CaptureStatus, CapturedGraph, GraphPoolHandle, begin_capture,
    capture_pool_for_handle, end_capture, end_capture_with_pool, graph_pool_handle,
    make_graphed_callable, release_graph_pool_handle,
};
#[cfg(feature = "cuda")]
pub use graph::{
    GraphCaptureGuard, begin_capture_with_mode, begin_capture_with_pool, capture_status,
    is_stream_capturing,
};
pub use kernels::{gpu_add, gpu_mul, gpu_neg, gpu_relu, gpu_sub};
pub use kernels::{
    gpu_add_into, gpu_embed_lookup_into, gpu_gelu_into, gpu_layernorm_into, gpu_mul_into,
    gpu_permute_0213_into, gpu_scale_into, gpu_slice_read_into, gpu_small_matmul_into,
    gpu_softmax_into, gpu_transpose_2d_into,
};
pub use kernels::{gpu_broadcast_add, gpu_broadcast_mul, gpu_broadcast_sub};
pub use kernels::{gpu_causal_mask_indirect, gpu_slice_write_indirect};
pub use kernels::{
    gpu_dropout, gpu_embed_lookup, gpu_gelu, gpu_layernorm, gpu_permute_0213, gpu_slice_read,
    gpu_slice_write, gpu_small_bmm, gpu_small_matmul, gpu_softmax, gpu_transpose_2d,
};
pub use memory_guard::{
    MemoryGuard, MemoryGuardBuilder, MemoryGuardedDevice, MemoryHook, MemoryPressureListener,
    MemoryReservation, MemoryStats, MemoryWatchdog, OomPolicy, PressureLevel,
};
pub use pool::{cached_bytes, empty_cache, empty_cache_all, round_len};
pub use rng::{CudaRngManager, PhiloxGenerator, PhiloxState, cuda_rng_manager, fork_rng, join_rng};
pub use tensor_bridge::{GpuFloat, GpuTensor, cuda, cuda_default, tensor_to_cpu, tensor_to_gpu};
pub use transfer::{alloc_zeros, alloc_zeros_f32, alloc_zeros_f64, cpu_to_gpu, gpu_to_cpu};
