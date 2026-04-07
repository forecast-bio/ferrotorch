use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Instant;

use crate::event::{DeviceType, GpuTimingPair, ProfileEvent};
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
    /// Set to `true` after the first poisoned-lock warning is logged.
    /// Prevents flooding stderr with repeated warnings.
    poison_warned: AtomicBool,
}

impl Profiler {
    /// Create a new profiler with the given configuration.
    fn new(config: ProfileConfig) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            start_time: Instant::now(),
            config,
            active: AtomicBool::new(true),
            poison_warned: AtomicBool::new(false),
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
        let input_shapes = if self.config.record_shapes {
            shapes.iter().map(|s| s.to_vec()).collect()
        } else {
            Vec::new()
        };
        let event = ProfileEvent {
            name: name.to_owned(),
            category: category.to_owned(),
            start_us,
            duration_us: 0,
            input_shapes,
            memory_bytes: None,
            thread_id: current_thread_id(),
            device_type: DeviceType::Cpu,
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
            thread_id: current_thread_id(),
            device_type: DeviceType::Cpu,
        };
        self.push_event(event);
    }

    /// Record a memory allocation or free event.
    pub fn record_memory(&self, name: &str, bytes: i64) {
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
            thread_id: current_thread_id(),
            device_type: DeviceType::Cpu,
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
            thread_id: current_thread_id(),
            // BUG-17 fix: always Cuda for GPU ops, even when timed by CPU fallback.
            device_type: DeviceType::Cuda,
        };
        self.push_event_returning_index(event)
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
        self.start_time.elapsed().as_micros() as u64
    }

    /// Push an event into the event list, logging a warning on the first
    /// poisoned mutex encounter.
    fn push_event(&self, event: ProfileEvent) {
        match self.events.lock() {
            Ok(mut events) => {
                events.push(event);
            }
            Err(_) => {
                self.warn_poisoned();
            }
        }
    }

    /// Push an event and return its index, or `None` on lock failure.
    fn push_event_returning_index(&self, event: ProfileEvent) -> Option<usize> {
        match self.events.lock() {
            Ok(mut events) => {
                let idx = events.len();
                events.push(event);
                Some(idx)
            }
            Err(_) => {
                self.warn_poisoned();
                None
            }
        }
    }

    /// Log a warning the first time a poisoned mutex is encountered.
    ///
    /// Subsequent calls are no-ops to avoid flooding stderr.
    fn warn_poisoned(&self) {
        if !self.poison_warned.swap(true, Ordering::Relaxed) {
            eprintln!(
                "ferrotorch-profiler: events mutex is poisoned — \
                 profiling events are being silently dropped. \
                 This typically means a thread panicked while holding the lock."
            );
        }
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
        let input_shapes = if self.config.record_shapes {
            shapes.iter().map(|s| s.to_vec()).collect()
        } else {
            Vec::new()
        };
        let event = ProfileEvent {
            name: name.to_owned(),
            category: category.to_owned(),
            start_us,
            duration_us,
            input_shapes,
            memory_bytes: None,
            thread_id: current_thread_id(),
            device_type: DeviceType::Cpu,
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
pub fn with_profiler<F, R>(config: ProfileConfig, f: F) -> (R, ProfileReport)
where
    F: FnOnce(&Profiler) -> R,
{
    let profiler = Arc::new(Profiler::new(config));
    // Install the hook for the duration of f().
    let hook: Arc<dyn profiler_hook::OpProfiler> = profiler.clone();
    profiler_hook::set_current(Some(hook));
    let _guard = ProfilerHookGuard;
    let result = f(&profiler);
    profiler.stop();
    // Take the profiler back out of the Arc. There may be a lingering
    // reference inside the thread-local at this point, but it'll be
    // cleared by the guard's Drop after we move into_report. We need
    // to drop the local hook BEFORE into_report so the Arc count goes
    // back to 1 and try_unwrap succeeds.
    drop(_guard);
    // Now Arc count is back to 1 (only `profiler` holds it), so we
    // can extract the inner Profiler.
    let profiler = Arc::try_unwrap(profiler)
        .unwrap_or_else(|_| panic!("profiler still has dangling references after closure exit"));
    let report = profiler.into_report();
    (result, report)
}

/// Best-effort thread id (not guaranteed to be unique across the
/// lifetime of the process, but good enough for trace display).
fn current_thread_id() -> u64 {
    // Use the raw pthread/thread id truncated to u64.
    let id = std::thread::current().id();
    // Debug format is "ThreadId(N)" — parse the number out.
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
