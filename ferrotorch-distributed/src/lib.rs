//! Distributed training for ferrotorch.
//!
//! This crate provides the building blocks for multi-rank training:
//!
//! - **Backends** ([`backend`]) — Transport-agnostic communication.
//!   [`TcpBackend`](backend::TcpBackend) for real multi-process training,
//!   [`SimulatedBackend`](backend::SimulatedBackend) for in-process testing.
//!
//! - **Collectives** ([`collective`]) — [`allreduce`](collective::allreduce),
//!   [`broadcast`](collective::broadcast), and [`barrier`](collective::barrier).
//!
//! - **DDP** ([`ddp`]) — [`DDP`](ddp::DDP) wraps a `Module` and
//!   synchronizes gradients across ranks after each backward pass.
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
//! ```

pub mod backend;
pub mod collective;
pub mod ddp;
pub mod error;

#[cfg(feature = "gpu")]
pub mod gpu_collective;

// Re-export key types at crate root for convenience.
pub use backend::{Backend, SimulatedBackend, TcpBackend};
pub use collective::{allreduce, allreduce_with_timeout, barrier, broadcast, ReduceOp, DEFAULT_COLLECTIVE_TIMEOUT};
pub use ddp::DDP;
pub use error::DistributedError;

#[cfg(feature = "gpu")]
pub use gpu_collective::{gpu_allreduce, gpu_broadcast};
