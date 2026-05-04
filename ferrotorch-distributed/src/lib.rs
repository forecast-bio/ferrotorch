// Lint baseline mirrors the workspace-standard pattern from
// `ferrotorch-gpu`/`-jit`/`-cubecl`/`-xpu` lib.rs. `unsafe_code` is NOT
// denied: this crate calls into NCCL via raw FFI (`nccl_sys`), uses
// `dlopen`/`dlsym`/`std::mem::transmute` to load CUDA stream symbols
// without a compile-time CUDA dependency (`nccl_backend`), and performs
// byte-reinterpret tensor I/O (`checkpoint`, `pipeline`). Per-block
// SAFETY substantiation lives at each `unsafe { ... }` site.
#![warn(clippy::all, clippy::pedantic)]
#![deny(rust_2018_idioms)]
// `missing_docs` and `missing_debug_implementations` are held at `allow`
// while the workspace-wide rustdoc / `Debug` pass is tracked separately
// (matches the existing `ferrotorch-gpu`/`-core` precedent — diverging
// unilaterally from a leaf crate would be Step 4 architectural
// unilateralism). Several distributed types own `Mutex<NcclComm>` raw
// FFI pointers, `Arc<dyn Backend>` trait objects, or `Box<dyn Fn>` RPC
// handlers whose `Debug` impls require careful hand-rolling.
#![allow(missing_docs, missing_debug_implementations)]
// Pedantic lints we explicitly accept across this crate. Each allow names
// a concrete reason — the alternative would be churn-for-zero-benefit or
// a worse API. Mirrors the ferrotorch-gpu baseline; add to this list only
// with a one-line justification.
#![allow(
    // # Errors / # Panics sections will be added during the workspace-wide
    // rustdoc pass (tracked separately, not gated on this lint baseline).
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    // Distributed code casts pervasively between `usize` rank/world_size
    // and `i32` NCCL/MPI peer indices, and between `u64` byte counters
    // and `usize` buffer lengths; the explicit cast is more readable
    // than try_into/unwrap or num-traits indirection.
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
    // doc-markdown rules are too aggressive for technical prose with
    // NCCL/MPI/RPC terminology.
    clippy::doc_markdown,
    // Test/helper modules define small fns after `let`-bindings; the
    // hoisting requirement is style-only.
    clippy::items_after_statements,
    // Long match-on-strategy/op blocks mirror the NCCL/PyTorch
    // taxonomy 1:1; splitting reduces legibility.
    clippy::too_many_lines,
    // Manual `Debug` impls intentionally omit non-Debug fields like
    // `Mutex<NcclComm>` (raw FFI pointers) and `Arc<dyn Backend>` to
    // keep formatted output free of lock probes / opaque handles.
    clippy::missing_fields_in_debug,
    // `match { Some(x) => x, None => return }` is the natural shape
    // when the `else` branch is non-trivial.
    clippy::single_match_else,
    // Methods that take `&self` for a uniform interface (e.g.,
    // `world_size()` on backends with a single rank) are part of the
    // public API shape and not refactor candidates.
    clippy::unused_self,
    // `.map(...).unwrap_or(...)` is the documented PyTorch-style
    // fallback shape used in option-bearing collectives; rewriting
    // to `match` is lossier.
    clippy::map_unwrap_or,
    // Match arms that each call out a specific reduction/op variant
    // are intentional when the variant set is documented and the
    // "wildcard branch" would hide future additions.
    clippy::match_same_arms,
    // `.collect::<Vec<_>>()` after mapping is the idiomatic shape;
    // rewriting to `extend(map(..))` is lossier and clippy's preference
    // is contested.
    clippy::redundant_closure_for_method_calls,
    // `for elem in vec.into_iter()` on owned `Vec`s mirrors the consumed
    // semantics in iteration; clippy's `for elem in vec` rewrite hides
    // that the value is consumed.
    clippy::explicit_into_iter_loop,
    // FFI raw-pointer casts (`*const c_void` <-> `*const T`) and
    // `&T as *const T` are the natural shape in NCCL bindings; clippy's
    // preferred `.cast()` / `std::ptr::from_ref` rewrites do not
    // improve legibility in this context.
    clippy::ptr_as_ptr,
    clippy::ref_as_ptr,
    // `format!("{x}")` already uses inline captures where `Display` is
    // direct; some sites use `.to_string()` or pass `&str` for
    // readability with structured prefixes.
    clippy::uninlined_format_args,
    // `HashMap<String, Tensor<T>>` parameters in checkpoint helpers
    // mirror PyTorch's `state_dict` shape; generalising over the
    // hasher would leak `S: BuildHasher` to every caller.
    clippy::implicit_hasher,
)]

//! Distributed training for ferrotorch.
//!
//! This crate provides the building blocks for multi-rank training:
//!
//! - **Backends** ([`backend`]) — Transport-agnostic communication.
//!   [`TcpBackend`](backend::TcpBackend) for real multi-process training,
//!   [`SimulatedBackend`](backend::SimulatedBackend) for in-process testing.
//!
//! - **Collectives** ([`collective`]) — [`allreduce`](collective::allreduce),
//!   [`all_gather`](collective::all_gather),
//!   [`reduce_scatter`](collective::reduce_scatter),
//!   [`broadcast`](collective::broadcast), and [`barrier`](collective::barrier).
//!
//! - **Async collectives** ([`async_collective`]) —
//!   [`async_all_gather`](async_collective::async_all_gather) and
//!   [`async_reduce_scatter`](async_collective::async_reduce_scatter)
//!   return a [`PendingCollective`](async_collective::PendingCollective)
//!   handle that can be `wait()`ed on after local compute, enabling FSDP
//!   backward prefetch.
//!
//! - **DDP** ([`ddp`]) — [`DDP`](ddp::DDP) wraps a `Module` and
//!   synchronizes gradients across ranks after each backward pass.
//!
//! - **FSDP** ([`fsdp`]) — [`FSDP`](fsdp::FSDP) wraps a `Module` and
//!   shards parameters across ranks, all-gathering during forward and
//!   reduce-scattering gradients during backward.
//!
//! - **RPC** ([`rpc`]) — Remote Procedure Call framework with
//!   [`RpcContext`](rpc::RpcContext) for invoking functions on remote ranks,
//!   and [`RRef`](rpc::RRef) for holding references to remote data.
//!
//! - **Pipeline parallelism** ([`pipeline`]) —
//!   [`Pipeline`](pipeline::Pipeline) splits a model into sequential stages
//!   and processes microbatches through them. Supports
//!   [`GPipe`](pipeline::PipelineSchedule::GPipe) and
//!   [`Interleaved1F1B`](pipeline::PipelineSchedule::Interleaved1F1B) schedules.
//!
//! - **GPU collectives** ([`gpu_collective`], requires `gpu` feature) —
//!   [`gpu_allreduce`](gpu_collective::gpu_allreduce) and
//!   [`gpu_broadcast`](gpu_collective::gpu_broadcast) route through NCCL
//!   when the `nccl` feature is enabled, or through an opt-in host
//!   round-trip when `FERROTORCH_ENABLE_GPU_FALLBACK=1` is set. Without
//!   either, they return `Err` (PyTorch parity). See [`gpu_collective`]
//!   for details.
//!
//! # Quick start
//!
//! ```ignore
//! use ferrotorch_distributed::backend::SimulatedBackend;
//! use ferrotorch_distributed::collective::{allreduce, ReduceOp};
//! use ferrotorch_distributed::ddp::DDP;
//! use ferrotorch_distributed::fsdp::FSDP;
//! use ferrotorch_distributed::rpc::{RpcContext, SimulatedRpcBackend};
//! use ferrotorch_distributed::pipeline::{Pipeline, PipelineStage, PipelineSchedule};
//! ```

pub mod async_collective;
pub mod backend;
pub mod checkpoint;
pub mod collective;
pub mod ddp;
pub mod device_mesh;
pub mod dtensor;
pub mod error;
pub mod fsdp;
pub mod gloo_backend;
pub mod mpi_backend;
pub mod p2p;
pub mod pipeline;
pub mod rpc;
pub mod sync_batch_norm;
pub mod ucc_backend;

#[cfg(feature = "gpu")]
pub mod gpu_collective;

#[cfg(feature = "nccl")]
pub mod hybrid_backend;
#[cfg(feature = "nccl")]
pub mod nccl_backend;
#[cfg(feature = "nccl")]
pub mod nccl_collective;
#[cfg(feature = "nccl")]
pub mod nccl_sys;

// Re-export key types at crate root for convenience.
pub use async_collective::{PendingCollective, async_all_gather, async_reduce_scatter};
pub use backend::{Backend, SimulatedBackend, SubBackend, TcpBackend};
pub use checkpoint::{
    AsyncCheckpointer, CheckpointFuture, DistCheckpointError, DistributedCheckpoint, ShardMetadata,
    TensorShardSpec, flat_shard_metadata, load_distributed, reshard, save_distributed,
};
pub use collective::{
    DEFAULT_COLLECTIVE_TIMEOUT, ReduceOp, all_gather, all_gather_with_timeout, all_to_all,
    all_to_all_single_uneven, all_to_all_with_timeout, allreduce, allreduce_with_timeout, barrier,
    broadcast, reduce_scatter, reduce_scatter_tensor, reduce_scatter_with_timeout,
};
pub use ddp::DDP;
pub use device_mesh::DeviceMesh;
pub use dtensor::{DTensor, Placement};
pub use error::DistributedError;
pub use fsdp::FSDP;
pub use gloo_backend::{GlooBackend, is_gloo_available};
pub use mpi_backend::{MpiBackend, is_mpi_available};
pub use p2p::{recv, recv_into, recv_into_with_timeout, recv_with_timeout, send, sendrecv};
pub use pipeline::{Pipeline, PipelineSchedule};
pub use rpc::{RpcAgent, RpcError, TcpRpcBackend};
pub use sync_batch_norm::SyncBatchNorm2d;
pub use ucc_backend::{UccBackend, is_ucc_available};

#[cfg(feature = "gpu")]
pub use gpu_collective::{gpu_allreduce, gpu_broadcast};

#[cfg(feature = "nccl")]
pub use hybrid_backend::HybridBackend;
#[cfg(feature = "nccl")]
pub use nccl_backend::{NcclBackend, is_nccl_available};
#[cfg(feature = "nccl")]
pub use nccl_collective::{nccl_all_gather, nccl_allreduce, nccl_broadcast, nccl_reduce_scatter};
#[cfg(feature = "nccl")]
pub use nccl_sys::NcclUniqueId;
