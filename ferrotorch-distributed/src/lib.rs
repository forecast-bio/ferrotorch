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
//!   [`gpu_broadcast`](gpu_collective::gpu_broadcast) transfer GPU tensors
//!   to CPU, run the standard TCP collective, and copy back. Portable
//!   alternative to NCCL.
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

pub mod backend;
pub mod checkpoint;
pub mod collective;
pub mod ddp;
pub mod error;
pub mod fsdp;
pub mod pipeline;
pub mod rpc;

#[cfg(feature = "gpu")]
pub mod gpu_collective;

// Re-export key types at crate root for convenience.
pub use backend::{Backend, SimulatedBackend, TcpBackend};
pub use checkpoint::{
    AsyncCheckpointer, CheckpointFuture, DistCheckpointError, DistributedCheckpoint,
    ShardMetadata, TensorShardSpec, flat_shard_metadata, load_distributed, reshard,
    save_distributed,
};
pub use collective::{
    all_gather, all_gather_with_timeout, allreduce, allreduce_with_timeout, barrier, broadcast,
    reduce_scatter, reduce_scatter_with_timeout, ReduceOp, DEFAULT_COLLECTIVE_TIMEOUT,
};
pub use ddp::DDP;
pub use error::DistributedError;
pub use fsdp::FSDP;
pub use pipeline::{Pipeline, PipelineActivations, PipelineSchedule, PipelineStage};
pub use rpc::{RRef, RpcBackend, RpcContext, RpcError, RpcFuture, SimulatedRpcBackend, TcpRpcBackend};

#[cfg(feature = "gpu")]
pub use gpu_collective::{gpu_allreduce, gpu_broadcast};
