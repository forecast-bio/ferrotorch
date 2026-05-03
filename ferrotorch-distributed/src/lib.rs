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
