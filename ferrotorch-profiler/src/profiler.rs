use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::event::{DeviceType, ProfileEvent};
use crate::report::ProfileReport;

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

// ---------------------------------------------------------------------------
// GpuTimingPair — deferred CUDA event pair for lazy elapsed-time resolution
// ---------------------------------------------------------------------------

/// A pair of CUDA events recorded before and after a GPU operation.
///
/// GPU timing is asynchronous: the events are *recorded* into the stream
/// immediately, but the elapsed time between them can only be queried after
/// both events have completed on the GPU. We store these pairs and resolve
/// them lazily when the profiling report is generated, at which point we
/// synchronize and call `CudaEvent::elapsed_ms`.
#[cfg(feature = "cuda")]
struct GpuTimingPair {
    start_event: cudarc::driver::CudaEvent,
    end_event: cudarc::driver::CudaEvent,
    /// Index into `Profiler::events` where the corresponding `ProfileEvent`
    /// lives. We patch `duration_us` on that event during resolution.
    event_index: usize,
}

/// Thread-safe operation profiler.
///
/// Create one via [`with_profiler`] and pass it into the closure that
/// contains the work you want to measure.
pub struct Profiler {
    events: Mutex<Vec<ProfileEvent>>,
    start_time: Instant,
    config: ProfileConfig,
    active: AtomicBool,
    /// Deferred GPU event pairs awaiting elapsed-time resolution.
    #[cfg(feature = "cuda")]
    gpu_pairs: Mutex<Vec<GpuTimingPair>>,
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
            gpu_pairs: Mutex::new(Vec::new()),
        }
    }

    /// Record a completed operation.
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
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }

    /// Record an operation whose duration is already known.
    pub fn record_with_duration(
        &self,
        name: &str,
        category: &str,
        duration_us: u64,
    ) {
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
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
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
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
    }

    /// Record a GPU operation, timing it with CUDA events.
    ///
    /// CUDA events are recorded into `stream` before and after calling `f()`.
    /// The elapsed time is resolved lazily when the profiling report is
    /// generated (via [`into_report`](Self::into_report)), because the GPU
    /// operation may still be in-flight when `record_gpu` returns.
    ///
    /// # Arguments
    ///
    /// * `name` — Operation name (e.g. `"matmul"`, `"conv2d"`).
    /// * `category` — Category tag (e.g. `"cuda_kernel"`).
    /// * `stream` — The CUDA stream the operation will be submitted to.
    /// * `ctx` — The CUDA context (needed to create timing events).
    /// * `f` — The closure that performs the GPU work.
    ///
    /// # Errors
    ///
    /// If event creation or recording fails, the closure is still executed
    /// and the event is recorded as a CPU-timed fallback with a warning
    /// category suffix. This ensures profiling never silently drops operations.
    #[cfg(feature = "cuda")]
    pub fn record_gpu<F, R>(
        &self,
        name: &str,
        category: &str,
        stream: &std::sync::Arc<cudarc::driver::CudaStream>,
        ctx: &std::sync::Arc<cudarc::driver::CudaContext>,
        f: F,
    ) -> R
    where
        F: FnOnce() -> R,
    {
        if !self.active.load(Ordering::Relaxed) {
            return f();
        }

        let start_us = self.elapsed_us();

        // Create timing-enabled events. If this fails, fall back to CPU timing.
        let timing_flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let start_event = match ctx.new_event(Some(timing_flags)) {
            Ok(e) => e,
            Err(_) => return self.record_gpu_cpu_fallback(name, category, f),
        };
        let end_event = match ctx.new_event(Some(timing_flags)) {
            Ok(e) => e,
            Err(_) => return self.record_gpu_cpu_fallback(name, category, f),
        };

        // Record start event into the stream.
        if start_event.record(stream).is_err() {
            return self.record_gpu_cpu_fallback(name, category, f);
        }

        let result = f();

        // Record end event into the stream.
        if end_event.record(stream).is_err() {
            // The operation ran but we can't time it. Record a zero-duration
            // GPU event so it still appears in the report.
            self.push_gpu_event(name, category, start_us, 0);
            return result;
        }

        // Push a placeholder event (duration_us = 0, will be patched later).
        let event_index = self.push_gpu_event(name, category, start_us, 0);

        // Store the event pair for deferred resolution.
        if let Ok(mut pairs) = self.gpu_pairs.lock() {
            pairs.push(GpuTimingPair {
                start_event,
                end_event,
                event_index,
            });
        }

        result
    }

    /// Fallback: time a GPU operation with CPU wall clock when CUDA event
    /// creation or recording fails. The category is suffixed with
    /// `"[cpu_fallback]"` so the user can see that GPU timing was not available.
    #[cfg(feature = "cuda")]
    fn record_gpu_cpu_fallback<F, R>(&self, name: &str, category: &str, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        let cpu_start = Instant::now();
        let result = f();
        let duration_us = cpu_start.elapsed().as_micros() as u64;
        let start_us = self.elapsed_us().saturating_sub(duration_us);
        let fallback_category = format!("{category}[cpu_fallback]");
        let event = ProfileEvent {
            name: name.to_owned(),
            category: fallback_category,
            start_us,
            duration_us,
            input_shapes: Vec::new(),
            memory_bytes: None,
            thread_id: current_thread_id(),
            device_type: DeviceType::Cpu,
        };
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
        }
        result
    }

    /// Push a GPU event and return its index in the events vec.
    #[cfg(feature = "cuda")]
    fn push_gpu_event(
        &self,
        name: &str,
        category: &str,
        start_us: u64,
        duration_us: u64,
    ) -> usize {
        let event = ProfileEvent {
            name: name.to_owned(),
            category: category.to_owned(),
            start_us,
            duration_us,
            input_shapes: Vec::new(),
            memory_bytes: None,
            thread_id: current_thread_id(),
            device_type: DeviceType::Cuda,
        };
        if let Ok(mut events) = self.events.lock() {
            let idx = events.len();
            events.push(event);
            idx
        } else {
            0
        }
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

    /// Resolve all deferred GPU event pairs by synchronizing and querying
    /// `CudaEvent::elapsed_ms`. Patches `duration_us` on the corresponding
    /// `ProfileEvent` entries.
    #[cfg(feature = "cuda")]
    fn resolve_gpu_timings(&self) {
        let pairs = match self.gpu_pairs.lock() {
            Ok(mut guard) => std::mem::take(&mut *guard),
            Err(_) => return,
        };

        if pairs.is_empty() {
            return;
        }

        let mut events = match self.events.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };

        for pair in &pairs {
            // elapsed_ms synchronizes both events internally, then queries
            // the GPU-side elapsed time. Returns milliseconds as f32.
            match pair.start_event.elapsed_ms(&pair.end_event) {
                Ok(ms) => {
                    // Convert milliseconds to microseconds (our canonical unit).
                    let us = (ms * 1000.0) as u64;
                    if pair.event_index < events.len() {
                        events[pair.event_index].duration_us = us;
                    }
                }
                Err(_) => {
                    // GPU timing resolution failed. Leave duration_us as 0
                    // and mark the event so the user knows.
                    if pair.event_index < events.len() {
                        events[pair.event_index].category.push_str("[timing_failed]");
                    }
                }
            }
        }
    }

    /// Drain collected events into a [`ProfileReport`].
    ///
    /// On the `cuda` feature, this first resolves all deferred GPU event
    /// pairs (synchronizing the GPU and querying elapsed times).
    fn into_report(self) -> ProfileReport {
        #[cfg(feature = "cuda")]
        self.resolve_gpu_timings();

        let events = self
            .events
            .into_inner()
            .unwrap_or_default();
        ProfileReport::new(events)
    }
}

/// Profile a closure.
///
/// Returns the closure's return value together with a [`ProfileReport`]
/// containing every event recorded during execution.
pub fn with_profiler<F, R>(config: ProfileConfig, f: F) -> (R, ProfileReport)
where
    F: FnOnce(&Profiler) -> R,
{
    let profiler = Profiler::new(config);
    let result = f(&profiler);
    profiler.stop();
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
