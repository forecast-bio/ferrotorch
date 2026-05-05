use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::event::{DeviceType, GpuTimingPair, MemoryCategory, ProfileEvent};
use crate::flops;
use crate::report::ProfileReport;
use ferrotorch_core::profiler_hook;

/// Controls what the profiler records.
#[derive(Debug, Clone)]
pub struct ProfileConfig {
    /// Capture input tensor shapes (slight overhead per event).
    pub record_shapes: bool,
    /// Capture memory allocation/free events.
    pub record_memory: bool,
    /// Capture call stacks (not yet implemented).
    pub with_stack: bool,
}

impl Default for ProfileConfig {
    fn default() -> Self {
        Self {
            record_shapes: true,
            record_memory: false,
            with_stack: false,
        }
    }
}

/// Thread-safe operation profiler.
///
/// Create one via [`with_profiler`] and pass it into the closure that
/// contains the work you want to measure.
///
/// # Lock ordering
///
/// The profiler contains a single `Mutex<Vec<ProfileEvent>>` (`events`).
/// There is no other mutex in this struct, so there is no lock-ordering
/// concern *within* the profiler. However, callers must not hold external
/// locks that could be acquired by the profiler's methods (currently none)
/// while calling `record*` or `push_gpu_event`, to avoid deadlocks.
///
/// If a future version adds additional mutexes (e.g., for GPU timing
/// pairs), the ordering must be: `gpu_timings` before `events`.
pub struct Profiler {
    events: Mutex<Vec<ProfileEvent>>,
    start_time: Instant,
    config: ProfileConfig,
    active: AtomicBool,
    /// Queued CUDA event pairs waiting for `flush_cuda_kernels` to
    /// synchronize and convert into [`ProfileEvent`]s. CL-380.
    #[cfg(feature = "cuda")]
    pending_cuda: Mutex<Vec<crate::cuda_timing::PendingCudaScope>>,
}

impl std::fmt::Debug for Profiler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let event_count = self.events.lock().map_or(0, |g| g.len());
        let mut dbg = f.debug_struct("Profiler");
        dbg.field("config", &self.config)
            .field("active", &self.active)
            .field("start_time", &self.start_time)
            .field("event_count", &event_count);
        #[cfg(feature = "cuda")]
        {
            let pending_count = self.pending_cuda.lock().map_or(0, |g| g.len());
            dbg.field("pending_cuda_count", &pending_count);
        }
        dbg.finish()
    }
}

impl Profiler {
    /// Create a new profiler with the given configuration.
    fn new(config: ProfileConfig) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            start_time: Instant::now(),
            config,
            active: AtomicBool::new(true),
            #[cfg(feature = "cuda")]
            pending_cuda: Mutex::new(Vec::new()),
        }
    }

    /// Record a completed CPU operation.
    ///
    /// The event's `start_us` is set to "now" and `duration_us` to zero.
    /// Use [`record_with_duration`](Self::record_with_duration) when you
    /// already know the elapsed time.
    pub fn record(&self, name: &str, category: &str, shapes: &[&[usize]]) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        let start_us = self.elapsed_us();
        let input_shapes_vec: Vec<Vec<usize>> = if self.config.record_shapes {
            shapes.iter().map(|s| s.to_vec()).collect()
        } else {
            Vec::new()
        };
        // FLOPS estimate from shapes (only when record_shapes is on,
        // otherwise we have nothing to estimate from).
        let flops = if self.config.record_shapes {
            flops::estimate(name, &input_shapes_vec)
        } else {
            None
        };
        let event = ProfileEvent {
            name: name.to_owned(),
            category: category.to_owned(),
            start_us,
            duration_us: 0,
            input_shapes: input_shapes_vec,
            memory_bytes: None,
            memory_category: None,
            thread_id: current_thread_id(),
            device_type: DeviceType::Cpu,
            flops,
            stack_trace: self.maybe_capture_stack(),
        };
        self.push_event(event);
    }

    /// Record an operation whose duration is already known.
    pub fn record_with_duration(&self, name: &str, category: &str, duration_us: u64) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        let start_us = self.elapsed_us().saturating_sub(duration_us);
        let event = ProfileEvent {
            name: name.to_owned(),
            category: category.to_owned(),
            start_us,
            duration_us,
            input_shapes: Vec::new(),
            memory_bytes: None,
            memory_category: None,
            thread_id: current_thread_id(),
            device_type: DeviceType::Cpu,
            flops: None,
            stack_trace: self.maybe_capture_stack(),
        };
        self.push_event(event);
    }

    /// Record a memory allocation or free event with the default
    /// "Other" category. Use [`record_memory_categorized`] to attach
    /// a specific category.
    pub fn record_memory(&self, name: &str, bytes: i64) {
        self.record_memory_categorized(name, bytes, MemoryCategory::Other);
    }

    /// Record a memory allocation or free event with a specific
    /// category (Activations, Parameters, `OptimizerState`, Gradients,
    /// Other). CL-333.
    pub fn record_memory_categorized(&self, name: &str, bytes: i64, category: MemoryCategory) {
        if !self.active.load(Ordering::Relaxed) || !self.config.record_memory {
            return;
        }
        let event = ProfileEvent {
            name: name.to_owned(),
            category: "memory".to_owned(),
            start_us: self.elapsed_us(),
            duration_us: 0,
            input_shapes: Vec::new(),
            memory_bytes: Some(bytes),
            memory_category: Some(category),
            thread_id: current_thread_id(),
            device_type: DeviceType::Cpu,
            flops: None,
            stack_trace: self.maybe_capture_stack(),
        };
        self.push_event(event);
    }

    /// Record a GPU operation event and return its index in the event list.
    ///
    /// Returns `Some(index)` on success, or `None` if the events mutex is
    /// poisoned. When `None` is returned, the caller should skip storing
    /// any associated `GpuTimingPair` — the event was not recorded, so
    /// there is nothing to correlate.
    ///
    /// The `device_type` is always set to [`DeviceType::Cuda`], even when
    /// the timing was measured by CPU-side `Instant` as a fallback (e.g.,
    /// when CUDA event timing is unavailable). This correctly reflects
    /// that the *operation* ran on the GPU, regardless of how its duration
    /// was measured.
    pub fn push_gpu_event(
        &self,
        name: &str,
        category: &str,
        timing: GpuTimingPair,
    ) -> Option<usize> {
        if !self.active.load(Ordering::Relaxed) {
            return None;
        }
        let event = ProfileEvent {
            name: name.to_owned(),
            category: category.to_owned(),
            start_us: timing.start_us,
            duration_us: timing.end_us.saturating_sub(timing.start_us),
            input_shapes: Vec::new(),
            memory_bytes: None,
            memory_category: None,
            thread_id: current_thread_id(),
            // BUG-17 fix: always Cuda for GPU ops, even when timed by CPU fallback.
            device_type: DeviceType::Cuda,
            flops: None,
            stack_trace: self.maybe_capture_stack(),
        };
        self.push_event_returning_index(event)
    }

    /// Capture a stack trace if `with_stack` is enabled in the config.
    /// Uses [`std::backtrace::Backtrace::capture`] which is a no-op
    /// (returns a Disabled backtrace) unless the `RUST_BACKTRACE` env
    /// var is set, so the cost when stack capture is requested but the
    /// env var is unset is just a few atomic loads.
    fn maybe_capture_stack(&self) -> Option<String> {
        if !self.config.with_stack {
            return None;
        }
        let bt = std::backtrace::Backtrace::capture();
        Some(format!("{bt}"))
    }

    /// Whether the profiler is currently active.
    pub fn is_active(&self) -> bool {
        self.active.load(Ordering::Relaxed)
    }

    /// Stop the profiler (subsequent `record*` calls become no-ops).
    pub fn stop(&self) {
        self.active.store(false, Ordering::Relaxed);
    }

    /// Microseconds elapsed since the profiler was created.
    fn elapsed_us(&self) -> u64 {
        // as_micros() returns u128; saturate to u64::MAX rather than truncate.
        // A profiling session lasting >584,542 years would saturate — acceptable.
        u64::try_from(self.start_time.elapsed().as_micros()).unwrap_or(u64::MAX)
    }

    /// Queue a CUDA event scope for later finalization. Called by
    /// [`crate::cuda_timing::CudaKernelScope::stop`]. CL-380.
    ///
    /// `pub(crate)` because the parameter type is `pub(crate)` — this
    /// is an internal hand-off between `CudaKernelScope::stop` and the
    /// profiler's pending-flush queue, not part of the public API.
    #[cfg(feature = "cuda")]
    pub(crate) fn push_pending_cuda_scope(&self, scope: crate::cuda_timing::PendingCudaScope) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        match self.pending_cuda.lock() {
            Ok(mut q) => q.push(scope),
            Err(_) => Self::warn_poisoned(),
        }
    }

    /// Synchronize every queued CUDA event scope and convert them
    /// into [`ProfileEvent`]s with the GPU-measured duration.
    ///
    /// Call this once before exporting the report. Idempotent: a
    /// second call after the queue is drained is a no-op. CL-380.
    ///
    /// Each pending scope is finalized in registration order. The
    /// elapsed time comes from `cuEventElapsedTime` so the recorded
    /// `duration_us` reflects actual GPU kernel time, not the CPU
    /// dispatch latency that the wall-clock fallback measures.
    #[cfg(feature = "cuda")]
    pub fn flush_cuda_kernels(&self) {
        let pending: Vec<crate::cuda_timing::PendingCudaScope> =
            if let Ok(mut q) = self.pending_cuda.lock() {
                std::mem::take(&mut *q)
            } else {
                Self::warn_poisoned();
                return;
            };
        if pending.is_empty() {
            return;
        }
        let epoch_us = self.elapsed_us();
        for scope in pending {
            let event = scope.finalize(epoch_us);
            self.push_event(event);
        }
    }

    /// No-op without the `cuda` feature. Provided so callers compile
    /// against the same API on both feature configurations.
    #[cfg(not(feature = "cuda"))]
    pub fn flush_cuda_kernels(&self) {
        // Without the cuda feature there are no CUDA events to flush.
    }

    /// Number of CUDA scopes currently queued waiting for flush.
    /// Returns 0 without the cuda feature. CL-380.
    pub fn pending_cuda_count(&self) -> usize {
        #[cfg(feature = "cuda")]
        {
            self.pending_cuda.lock().map_or(0, |q| q.len())
        }
        #[cfg(not(feature = "cuda"))]
        {
            0
        }
    }

    /// Push an event into the event list, logging a warning on the first
    /// poisoned mutex encounter.
    fn push_event(&self, event: ProfileEvent) {
        match self.events.lock() {
            Ok(mut events) => {
                events.push(event);
            }
            Err(_) => {
                Self::warn_poisoned();
            }
        }
    }

    /// Push an event and return its index, or `None` on lock failure.
    fn push_event_returning_index(&self, event: ProfileEvent) -> Option<usize> {
        if let Ok(mut events) = self.events.lock() {
            let idx = events.len();
            events.push(event);
            Some(idx)
        } else {
            // Mutex poisoned: silently drop — see `push_event` doc.
            None
        }
    }

    /// Called when a mutex lock returns a `PoisonError`.
    ///
    /// The workspace has no `tracing`/`log` dependency, and this library
    /// has no `Result`-returning path to propagate the condition through.
    /// Silently dropping events is the least-bad option: the profiler is
    /// a diagnostic tool, so losing events on a poisoned mutex (which
    /// only happens when another thread has already panicked) is
    /// acceptable.  Callers that need to detect this should inspect the
    /// returned `Option::None` from `push_event_returning_index` or the
    /// empty event list in the final `ProfileReport`.
    fn warn_poisoned() {
        // Intentional no-op: see doc comment above.
    }

    /// Drain collected events into a [`ProfileReport`].
    fn into_report(self) -> ProfileReport {
        let events = self.events.into_inner().unwrap_or_default();
        ProfileReport::new(events)
    }
}

// ---------------------------------------------------------------------------
// OpProfiler impl: lets ferrotorch-core's tensor ops auto-record themselves
// when this profiler is the active one for the thread. CL-379.
// ---------------------------------------------------------------------------

impl profiler_hook::OpProfiler for Profiler {
    fn record_op(&self, name: &str, category: &str, shapes: &[&[usize]], duration_us: u64) {
        if !self.active.load(Ordering::Relaxed) {
            return;
        }
        let start_us = self.elapsed_us().saturating_sub(duration_us);
        let input_shapes_vec: Vec<Vec<usize>> = if self.config.record_shapes {
            shapes.iter().map(|s| s.to_vec()).collect()
        } else {
            Vec::new()
        };
        // Estimate FLOPS from shapes when shape recording is on. CL-333.
        let flops = if self.config.record_shapes {
            flops::estimate(name, &input_shapes_vec)
        } else {
            None
        };
        let event = ProfileEvent {
            name: name.to_owned(),
            category: category.to_owned(),
            start_us,
            duration_us,
            input_shapes: input_shapes_vec,
            memory_bytes: None,
            memory_category: None,
            thread_id: current_thread_id(),
            device_type: DeviceType::Cpu,
            flops,
            stack_trace: self.maybe_capture_stack(),
        };
        self.push_event(event);
    }
}

/// RAII guard that clears the thread-local profiler hook on drop.
/// Used by `with_profiler` so even on panic the hook is cleared and
/// subsequent code in the same thread doesn't see a stale profiler.
struct ProfilerHookGuard;

impl Drop for ProfilerHookGuard {
    fn drop(&mut self) {
        profiler_hook::set_current(None);
    }
}

/// Profile a closure.
///
/// Returns the closure's return value together with a [`ProfileReport`]
/// containing every event recorded during execution.
///
/// While the closure runs, this profiler is also installed as the
/// thread-local hook in `ferrotorch_core::profiler_hook`, so any tensor
/// op invoked from inside the closure that uses
/// `profile_op_scope` will automatically record itself here. The hook
/// is cleared on closure exit (including panic unwind) via an RAII
/// guard. CL-379.
///
/// # Panics
///
/// Reaches an [`unreachable!`] if an internal invariant is violated:
/// the `Arc<Profiler>` strong count is not 1 after the closure and
/// the RAII hook guard have both been dropped. This is unreachable
/// given the current API — the profiler `Arc` is created inside
/// this function and `f` only ever receives a `&Profiler` borrow,
/// so no caller can hold a second clone. The only other strong
/// reference is the one stashed in the thread-local profiler hook,
/// and dropping `ProfilerHookGuard` clears it before
/// [`Arc::try_unwrap`] is called.
pub fn with_profiler<F, R>(config: ProfileConfig, f: F) -> (R, ProfileReport)
where
    F: FnOnce(&Profiler) -> R,
{
    let profiler = Arc::new(Profiler::new(config));
    // Install the hook for the duration of f().
    let hook: Arc<dyn profiler_hook::OpProfiler> = profiler.clone();
    profiler_hook::set_current(Some(hook));
    let guard = ProfilerHookGuard;
    let result = f(&profiler);
    profiler.stop();
    // Drop the RAII guard to clear the thread-local hook *before*
    // try_unwrap, so the Arc count returns to 1.
    drop(guard);
    // Invariant: the only extra Arc clone was `hook` above; dropping
    // `guard` removed it from the thread-local, so strong count == 1.
    // `unreachable!` (not `panic!`) documents that this branch cannot
    // be exercised by any caller given the current API surface — the
    // `Arc<Profiler>` never escapes this function.
    let profiler = Arc::try_unwrap(profiler).unwrap_or_else(|_| {
        unreachable!(
            "Arc<Profiler> strong count != 1 after hook cleared; \
             unreachable given the current with_profiler API shape"
        )
    });
    let report = profiler.into_report();
    (result, report)
}

/// Best-effort thread id (not guaranteed to be unique across the
/// lifetime of the process, but good enough for trace display).
fn current_thread_id() -> u64 {
    // `ThreadId::as_u64` (feature `thread_id_value`) is not yet stable
    // as of rustc 1.95 (tracked in rust-lang/rust#67939).  Until it
    // stabilises, parse the inner number from the Debug output
    // "ThreadId(N)".  The format is a private implementation detail but
    // has been stable in practice since Rust 1.0 and the `unwrap_or(0)`
    // fallback means we degrade gracefully if it ever changes.
    let id = std::thread::current().id();
    let dbg = format!("{id:?}");
    dbg.trim_start_matches("ThreadId(")
        .trim_end_matches(')')
        .parse::<u64>()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_push_gpu_event_returns_index() {
        let profiler = Profiler::new(ProfileConfig::default());
        let timing = GpuTimingPair {
            start_us: 100,
            end_us: 200,
        };

        let idx = profiler.push_gpu_event("matmul", "cuda_kernel", timing);
        assert_eq!(idx, Some(0));

        let idx = profiler.push_gpu_event("relu", "cuda_kernel", timing);
        assert_eq!(idx, Some(1));
    }

    #[test]
    fn test_push_gpu_event_sets_cuda_device_type() {
        let ((), report) = with_profiler(ProfileConfig::default(), |p| {
            let timing = GpuTimingPair {
                start_us: 0,
                end_us: 50,
            };
            p.push_gpu_event("conv2d", "cuda_kernel", timing);
        });
        let events = report.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].device_type, DeviceType::Cuda);
        assert_eq!(events[0].duration_us, 50);
    }

    #[test]
    fn test_push_gpu_event_inactive_returns_none() {
        let profiler = Profiler::new(ProfileConfig::default());
        profiler.stop();
        let timing = GpuTimingPair {
            start_us: 0,
            end_us: 100,
        };
        assert_eq!(profiler.push_gpu_event("mm", "cuda_kernel", timing), None);
    }

    #[test]
    fn test_record_sets_cpu_device_type() {
        let ((), report) = with_profiler(ProfileConfig::default(), |p| {
            p.record("add", "tensor_op", &[&[3, 4]]);
        });
        let events = report.events();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].device_type, DeviceType::Cpu);
    }
}
