#![warn(clippy::all, clippy::pedantic)]
#![warn(missing_debug_implementations, rust_2018_idioms)]
#![deny(unsafe_code)]
#![allow(clippy::module_name_repetitions)] // ProfileConfig, ProfileEvent, etc. are intentional
//! Operation profiling for ferrotorch.
//!
//! Provides [`Profiler`] for recording operation timings, memory events, and
//! input shapes during a forward/backward pass.  The resulting [`ProfileReport`]
//! can be rendered as a human-readable table or exported to Chrome trace JSON
//! (`chrome://tracing`).
//!
//! # Quick start
//!
//! ```rust
//! use ferrotorch_profiler::{with_profiler, ProfileConfig};
//!
//! let config = ProfileConfig::default();
//! let (result, report) = with_profiler(config, |profiler| {
//!     profiler.record("matmul", "tensor_op", &[&[32, 784], &[784, 256]]);
//!     profiler.record("relu", "tensor_op", &[&[32, 256]]);
//!     42
//! });
//!
//! println!("{}", report.table(10));
//! ```

#[cfg(feature = "cuda")]
pub mod cuda_timing;
mod event;
pub mod flops;
mod profiler;
mod report;
pub mod schedule;

// `CudaKernelScope` is the public API for users who want to time a GPU kernel
// region. `PendingCudaScope` is an internal queue type used by `Profiler`;
// it has no meaningful public contract and is not re-exported.
#[cfg(feature = "cuda")]
pub use cuda_timing::CudaKernelScope;
pub use event::{MemoryCategory, ProfileEvent};
pub use profiler::{ProfileConfig, Profiler, with_profiler};
pub use report::{OpSummary, ProfileReport};
pub use schedule::{ProfileSchedule, SchedulePhase};
