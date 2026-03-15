use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use crate::event::ProfileEvent;
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

/// Thread-safe operation profiler.
///
/// Create one via [`with_profiler`] and pass it into the closure that
/// contains the work you want to measure.
pub struct Profiler {
    events: Mutex<Vec<ProfileEvent>>,
    start_time: Instant,
    config: ProfileConfig,
    active: AtomicBool,
}

impl Profiler {
    /// Create a new profiler with the given configuration.
    fn new(config: ProfileConfig) -> Self {
        Self {
            events: Mutex::new(Vec::new()),
            start_time: Instant::now(),
            config,
            active: AtomicBool::new(true),
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
        };
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
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

    /// Drain collected events into a [`ProfileReport`].
    fn into_report(self) -> ProfileReport {
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
