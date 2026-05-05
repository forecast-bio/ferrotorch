//! CUDA event-based GPU kernel timing for the profiler. CL-380.
//!
//! The base [`Profiler`](crate::Profiler) records GPU events using
//! CPU wall-clock fallback (`Instant::now`) which only measures the
//! latency between the CPU dispatching the kernel and the next CPU
//! instruction. That is wildly inaccurate for asynchronously-launched
//! CUDA kernels, where the actual GPU work may continue long after
//! the launch returns.
//!
//! This module replaces the CPU-side timing with **CUDA events**:
//!
//! 1. Before the kernel runs, record a "start" event on the kernel's
//!    stream via `cuEventRecord`.
//! 2. Run the kernel(s).
//! 3. Record an "end" event on the same stream.
//! 4. Push the (start, end, name, category) tuple onto a pending
//!    queue without synchronizing — the events are async and the
//!    actual GPU times will not be ready yet.
//! 5. At report-export time (or via an explicit
//!    [`Profiler::flush_cuda_kernels`](crate::Profiler::flush_cuda_kernels)
//!    call), iterate the queue, synchronize each end event, and
//!    convert `cuEventElapsedTime` to a real `ProfileEvent` with
//!    `device_type = DeviceType::Cuda` and the GPU-measured duration
//!    in microseconds.
//!
//! # Why not CUPTI?
//!
//! CUPTI (the CUDA Profiling Tools Interface) gives automatic
//! callbacks for every kernel launch, but requires linking against
//! `libcupti.so` which is not always available, has versioning
//! issues with the CUDA driver, and adds significant overhead for
//! per-event tracing. CUDA events are first-class CUDA driver
//! primitives that have been part of `cudarc` since 0.10, work on
//! every CUDA-capable GPU including consumer cards (RTX 3090 etc.),
//! and add only ~1 µs of overhead per record. They give the same
//! per-kernel GPU timing accuracy that CUPTI's activity API does,
//! at the cost of having to wrap each timed region explicitly.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaEvent, CudaStream};

use crate::event::{DeviceType, ProfileEvent};
use crate::profiler::Profiler;

/// A pair of CUDA events bracketing a kernel region whose GPU-side
/// duration we want to measure. CL-380.
///
/// Created by [`CudaKernelScope::new`] which records the start
/// event on the given stream. Call [`CudaKernelScope::stop`] after
/// launching the kernel(s) to record the end event and queue the
/// pair on the profiler. The actual `cuEventElapsedTime` query is
/// deferred to [`Profiler::flush_cuda_kernels`] so the start/stop
/// path stays asynchronous and adds only the cost of two
/// `cuEventRecord` calls.
#[derive(Debug)]
pub struct CudaKernelScope {
    name: String,
    category: String,
    start: Arc<CudaEvent>,
    stream: Arc<CudaStream>,
}

impl CudaKernelScope {
    /// Create a new scope and record the start event on `stream`.
    ///
    /// Both the start and end events are timing-enabled (created
    /// without `CU_EVENT_DISABLE_TIMING`) so `cuEventElapsedTime`
    /// works on the pair.
    ///
    /// # Errors
    ///
    /// Returns the cudarc `DriverError` if event creation or the
    /// initial `cuEventRecord` fails.
    pub fn new(
        ctx: &Arc<CudaContext>,
        stream: &Arc<CudaStream>,
        name: impl Into<String>,
        category: impl Into<String>,
    ) -> Result<Self, cudarc::driver::DriverError> {
        let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let start = Arc::new(ctx.new_event(Some(flags))?);
        start.record(stream)?;
        Ok(Self {
            name: name.into(),
            category: category.into(),
            start,
            stream: Arc::clone(stream),
        })
    }

    /// Record the end event on the same stream the start was
    /// recorded on, then queue the (start, end, name, category)
    /// tuple onto `profiler`'s pending CUDA scope list.
    ///
    /// Does **not** synchronize — the actual elapsed-time query is
    /// deferred to [`Profiler::flush_cuda_kernels`]. This keeps
    /// `stop()` cheap so it can be called inline in hot kernel
    /// dispatch loops without serializing the device queue.
    ///
    /// # Errors
    ///
    /// Returns the cudarc `DriverError` if event creation or the
    /// final `cuEventRecord` fails.
    pub fn stop(self, profiler: &Profiler) -> Result<(), cudarc::driver::DriverError> {
        let ctx = self.stream.context();
        let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let end = Arc::new(ctx.new_event(Some(flags))?);
        end.record(&self.stream)?;
        profiler.push_pending_cuda_scope(PendingCudaScope {
            name: self.name,
            category: self.category,
            start: self.start,
            end,
        });
        Ok(())
    }
}

/// One queued (start, end) event pair waiting for `flush_cuda_kernels`
/// to synchronize and convert into a [`ProfileEvent`]. Internal type
/// used by [`Profiler`]; exposed only via the
/// [`Profiler::push_pending_cuda_scope`] entry point.
///
/// Visibility is `pub(crate)` because no caller outside this crate has
/// any way to construct one — the only producer is
/// [`CudaKernelScope::stop`] and the only consumer is `Profiler`'s
/// pending-queue. Verified zero external uses workspace-wide.
#[derive(Debug)]
pub(crate) struct PendingCudaScope {
    pub(crate) name: String,
    pub(crate) category: String,
    pub(crate) start: Arc<CudaEvent>,
    pub(crate) end: Arc<CudaEvent>,
}

impl PendingCudaScope {
    /// Synchronize the end event (blocks the calling CPU thread
    /// until the GPU reaches it) and convert the (start, end) pair
    /// into a real [`ProfileEvent`] with the GPU-measured duration.
    ///
    /// `profiler_epoch_us` is "now" minus the profiler's start time
    /// in microseconds — used as the synthetic `start_us` for the
    /// event since CUDA events don't carry an absolute timestamp,
    /// only relative durations.
    pub(crate) fn finalize(self, profiler_epoch_us: u64) -> ProfileEvent {
        // Block until the end event has been reached on the GPU.
        // Errors are structured: a failed synchronize or elapsed_ms
        // query appends " [timing_error]" to the event name so callers
        // can distinguish a genuine <1 µs kernel from a failed query.
        // Per rust-gpu-discipline §3, silent zero is forbidden — this
        // makes the failure visible in the trace without losing the op.
        let sync_ok = self.end.synchronize().is_ok();
        let (duration_us, timing_failed) = if sync_ok {
            match self.start.elapsed_ms(&self.end) {
                Ok(ms) if ms > 0.0 => {
                    // `ms` is finite-positive and bounded by the longest
                    // realistic kernel (well under 2^53 µs ≈ 285 years),
                    // so the f32 → u64 conversion is exact in practice.
                    // `as u64` saturates on +∞/NaN to 0, which is
                    // benign here (just reports a 0 µs kernel).
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    let us = (ms * 1000.0).round() as u64;
                    (us, false)
                }
                Ok(_) => (0, false), // genuine sub-µs kernel
                Err(_) => (0, true), // elapsed_ms query failed
            }
        } else {
            (0, true) // synchronize failed
        };
        let name = if timing_failed {
            format!("{} [timing_error]", self.name)
        } else {
            self.name
        };
        ProfileEvent {
            name,
            category: self.category,
            // start_us is anchored to "when stop() was called"; the
            // exact value isn't meaningful for ordering analysis
            // since CUDA events are async, but it places the event
            // in the same relative time window as CPU events for
            // chrome trace alignment.
            start_us: profiler_epoch_us.saturating_sub(duration_us),
            duration_us,
            input_shapes: Vec::new(),
            memory_bytes: None,
            memory_category: None,
            thread_id: 0,
            device_type: DeviceType::Cuda,
            flops: None,
            stack_trace: None,
        }
    }
}
