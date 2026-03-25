//! CUDA stream pool with thread-local current stream and event wrappers.
//!
//! Provides multi-stream concurrency for overlapping compute and data transfers:
//!
//! - [`CudaEventWrapper`] — safe wrapper around cudarc's `CudaEvent` with record/sync/query.
//! - [`StreamPool`] — per-device pool of CUDA streams, created lazily, round-robin dispatch.
//! - [`get_current_stream`] / [`set_current_stream`] — thread-local "active" stream per device.
//! - [`StreamGuard`] — RAII guard that sets the current stream and restores the previous on drop.
//!
//! # Design
//!
//! Each device gets [`STREAMS_PER_DEVICE`] non-blocking streams created via
//! [`CudaContext::new_stream`]. The pool is initialized lazily on first access
//! using [`OnceLock`]. Streams are distributed round-robin via an atomic counter.
//!
//! The thread-local current stream allows callers to override which stream a
//! device operation targets without threading a stream parameter through every
//! function. [`StreamGuard`] makes this ergonomic and exception-safe.

#[cfg(feature = "cuda")]
use std::cell::RefCell;
#[cfg(feature = "cuda")]
use std::collections::HashMap;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicUsize, Ordering};
#[cfg(feature = "cuda")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaEvent, CudaStream};

use crate::error::{GpuError, GpuResult};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Number of streams created per device in the pool.
#[cfg(feature = "cuda")]
const STREAMS_PER_DEVICE: usize = 8;

/// Maximum supported device ordinal. Guards against unbounded allocation
/// if a caller passes a bogus ordinal.
#[cfg(feature = "cuda")]
const MAX_DEVICES: usize = 64;

// ---------------------------------------------------------------------------
// CudaEventWrapper — safe wrapper around cudarc's CudaEvent
// ---------------------------------------------------------------------------

/// Safe wrapper around a cudarc [`CudaEvent`].
///
/// Records a point in a stream's execution timeline and allows the host or
/// other streams to wait until that point is reached.
///
/// All methods return [`GpuResult`] rather than panicking on CUDA errors.
#[cfg(feature = "cuda")]
pub struct CudaEventWrapper {
    inner: CudaEvent,
}

#[cfg(feature = "cuda")]
impl CudaEventWrapper {
    /// Create a new event associated with the given device's context.
    ///
    /// The event is created with `CU_EVENT_DISABLE_TIMING` (the cudarc default
    /// when `None` is passed for flags). Use [`new_with_timing`] if you need
    /// elapsed-time queries.
    pub fn new(ctx: &Arc<CudaContext>) -> GpuResult<Self> {
        let inner = ctx.new_event(None)?;
        Ok(Self { inner })
    }

    /// Create a new event with timing enabled.
    ///
    /// Required if you want to call [`elapsed_ms`](CudaEvent::elapsed_ms).
    /// Timing events are slightly more expensive than non-timing events.
    pub fn new_with_timing(ctx: &Arc<CudaContext>) -> GpuResult<Self> {
        let flags = cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT;
        let inner = ctx.new_event(Some(flags))?;
        Ok(Self { inner })
    }

    /// Record the current point in `stream`'s execution into this event.
    ///
    /// After recording, [`synchronize`](Self::synchronize) will block until all
    /// work submitted to `stream` before this call has completed.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the stream belongs to a different CUDA context than
    /// the event, or if the CUDA driver reports an error.
    pub fn record(&self, stream: &CudaStream) -> GpuResult<()> {
        self.inner.record(stream)?;
        Ok(())
    }

    /// Block the calling CPU thread until all work recorded in this event
    /// has completed on the GPU.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the CUDA driver reports an error (e.g., a previous
    /// async kernel launch failed).
    pub fn synchronize(&self) -> GpuResult<()> {
        self.inner.synchronize()?;
        Ok(())
    }

    /// Query whether all work recorded in this event has completed.
    ///
    /// Returns `Ok(true)` if complete, `Ok(false)` if still in progress.
    /// This is a non-blocking check.
    pub fn query(&self) -> GpuResult<bool> {
        Ok(self.inner.is_complete())
    }

    /// Make `stream` wait for all work recorded in this event to complete
    /// before executing any subsequent operations.
    ///
    /// This is a GPU-side wait — it does not block the CPU.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the stream and event belong to different CUDA contexts.
    pub fn wait_on(&self, stream: &CudaStream) -> GpuResult<()> {
        stream.wait(&self.inner)?;
        Ok(())
    }

    /// Borrow the underlying cudarc [`CudaEvent`].
    #[inline]
    pub fn inner(&self) -> &CudaEvent {
        &self.inner
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaEventWrapper {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaEventWrapper").finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// StreamPool — per-device pool of CUDA streams
// ---------------------------------------------------------------------------

/// Per-device pool of CUDA streams for concurrent kernel execution.
///
/// Streams are created lazily on first access for a given device ordinal.
/// [`get_stream`](StreamPool::get_stream) distributes streams round-robin
/// across the pool, ensuring balanced utilization.
///
/// The pool holds [`STREAMS_PER_DEVICE`] streams per device (currently 8).
#[cfg(feature = "cuda")]
struct DeviceStreams {
    streams: Vec<Arc<CudaStream>>,
    counter: AtomicUsize,
}

/// Global stream pool. Each entry is lazily initialized via `OnceLock`.
///
/// We use a fixed-size array of `OnceLock` rather than a `HashMap` to avoid
/// locking on the hot path. The index is the device ordinal.
#[cfg(feature = "cuda")]
static STREAM_POOL: OnceLock<Vec<OnceLock<DeviceStreams>>> = OnceLock::new();

/// Initialize the pool structure (array of `OnceLock` slots). Called once.
#[cfg(feature = "cuda")]
fn pool_slots() -> &'static Vec<OnceLock<DeviceStreams>> {
    STREAM_POOL.get_or_init(|| {
        (0..MAX_DEVICES).map(|_| OnceLock::new()).collect()
    })
}

/// Public interface for the CUDA stream pool.
pub struct StreamPool;

#[cfg(feature = "cuda")]
impl StreamPool {
    /// Get a stream for the given device, round-robin across the pool.
    ///
    /// On first call for a device ordinal, lazily creates [`STREAMS_PER_DEVICE`]
    /// non-blocking streams from the device's CUDA context.
    ///
    /// # Arguments
    ///
    /// * `ctx` — The CUDA context for the target device. Must match the
    ///   ordinal (callers are responsible for passing the correct context).
    /// * `device_ordinal` — The GPU device index (0-based).
    ///
    /// # Errors
    ///
    /// - Returns [`GpuError::InvalidDevice`] if `device_ordinal >= MAX_DEVICES`.
    /// - Returns a CUDA driver error if stream creation fails.
    pub fn get_stream(
        ctx: &Arc<CudaContext>,
        device_ordinal: usize,
    ) -> GpuResult<Arc<CudaStream>> {
        if device_ordinal >= MAX_DEVICES {
            return Err(GpuError::InvalidDevice {
                ordinal: device_ordinal,
                count: MAX_DEVICES,
            });
        }

        let slots = pool_slots();
        let device_streams = slots[device_ordinal].get_or_init(|| {
            // We create the streams eagerly within this device's OnceLock init.
            // If any stream creation fails, we store what we got (at least 1).
            let mut streams = Vec::with_capacity(STREAMS_PER_DEVICE);
            for _ in 0..STREAMS_PER_DEVICE {
                match ctx.new_stream() {
                    Ok(s) => streams.push(s),
                    Err(_) => break,
                }
            }
            // If we got zero streams, push a fallback: fork from default stream.
            if streams.is_empty() {
                if let Ok(s) = ctx.default_stream().fork() {
                    streams.push(s);
                }
            }
            DeviceStreams {
                streams,
                counter: AtomicUsize::new(0),
            }
        });

        if device_streams.streams.is_empty() {
            return Err(GpuError::Driver(cudarc::driver::DriverError(
                cudarc::driver::sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY,
            )));
        }

        let idx = device_streams.counter.fetch_add(1, Ordering::Relaxed)
            % device_streams.streams.len();
        Ok(Arc::clone(&device_streams.streams[idx]))
    }

    /// Return the number of streams currently in the pool for a device.
    /// Returns 0 if the device has not been initialized yet.
    pub fn pool_size(device_ordinal: usize) -> usize {
        if device_ordinal >= MAX_DEVICES {
            return 0;
        }
        let slots = pool_slots();
        slots[device_ordinal]
            .get()
            .map(|ds| ds.streams.len())
            .unwrap_or(0)
    }
}

// ---------------------------------------------------------------------------
// Thread-local current stream
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
thread_local! {
    /// Per-thread map from device ordinal to the "current" stream for that device.
    /// When set, GPU operations on that device should use this stream instead of
    /// the device's default stream.
    static CURRENT_STREAMS: RefCell<HashMap<usize, Arc<CudaStream>>> =
        RefCell::new(HashMap::new());
}

/// Get the current thread-local stream for the given device.
///
/// Returns `None` if no stream has been set for this device on the current
/// thread. In that case, callers should fall back to the device's default stream.
#[cfg(feature = "cuda")]
pub fn get_current_stream(device: usize) -> Option<Arc<CudaStream>> {
    CURRENT_STREAMS.with(|map| map.borrow().get(&device).cloned())
}

/// Set the current thread-local stream for the given device.
///
/// After this call, [`get_current_stream`] will return `Some(stream)` for
/// this device on the current thread until it is changed or cleared.
#[cfg(feature = "cuda")]
pub fn set_current_stream(device: usize, stream: Arc<CudaStream>) {
    CURRENT_STREAMS.with(|map| {
        map.borrow_mut().insert(device, stream);
    });
}

/// Clear the current thread-local stream for the given device, reverting
/// to the device's default stream.
#[cfg(feature = "cuda")]
pub fn clear_current_stream(device: usize) {
    CURRENT_STREAMS.with(|map| {
        map.borrow_mut().remove(&device);
    });
}

/// Get the current stream for a device, falling back to the device's default
/// stream if none has been set on this thread.
///
/// This is the primary entry point for operations that need "the stream to use."
#[cfg(feature = "cuda")]
pub fn current_stream_or_default(device: &crate::device::GpuDevice) -> Arc<CudaStream> {
    get_current_stream(device.ordinal())
        .unwrap_or_else(|| Arc::clone(device.default_stream()))
}

// ---------------------------------------------------------------------------
// StreamGuard — RAII guard for thread-local current stream
// ---------------------------------------------------------------------------

/// RAII guard that sets the thread-local current stream on construction and
/// restores the previous stream (or clears it) on drop.
///
/// # Example
///
/// ```ignore
/// use ferrotorch_gpu::stream::{StreamGuard, StreamPool};
///
/// let stream = StreamPool::get_stream(&ctx, 0)?;
/// {
///     let _guard = StreamGuard::new(0, stream);
///     // All operations on device 0 in this scope use `stream`.
///     // ...
/// }
/// // Previous stream (or default) is restored here.
/// ```
#[cfg(feature = "cuda")]
pub struct StreamGuard {
    device: usize,
    previous: Option<Arc<CudaStream>>,
}

#[cfg(feature = "cuda")]
impl StreamGuard {
    /// Set `stream` as the current stream for `device` on this thread.
    ///
    /// The previous current stream (if any) is saved and will be restored
    /// when this guard is dropped.
    pub fn new(device: usize, stream: Arc<CudaStream>) -> Self {
        let previous = get_current_stream(device);
        set_current_stream(device, stream);
        Self { device, previous }
    }
}

#[cfg(feature = "cuda")]
impl Drop for StreamGuard {
    fn drop(&mut self) {
        match self.previous.take() {
            Some(prev) => set_current_stream(self.device, prev),
            None => clear_current_stream(self.device),
        }
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for StreamGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("StreamGuard")
            .field("device", &self.device)
            .field("has_previous", &self.previous.is_some())
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Stubs when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub `CudaEventWrapper` when the `cuda` feature is not enabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct CudaEventWrapper;

#[cfg(not(feature = "cuda"))]
impl StreamPool {
    /// Always returns an error — compile with `features = ["cuda"]`.
    pub fn get_stream(_device_ordinal: usize) -> GpuResult<()> {
        Err(GpuError::NoCudaFeature)
    }

    /// Returns 0 — no streams without CUDA.
    pub fn pool_size(_device_ordinal: usize) -> usize {
        0
    }
}

/// Stub `StreamGuard` when the `cuda` feature is not enabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct StreamGuard;

/// Stub — returns `None` without CUDA.
#[cfg(not(feature = "cuda"))]
pub fn get_current_stream(_device: usize) -> Option<()> {
    None
}

/// Stub — no-op without CUDA.
#[cfg(not(feature = "cuda"))]
pub fn set_current_stream(_device: usize, _stream: ()) {}

/// Stub — no-op without CUDA.
#[cfg(not(feature = "cuda"))]
pub fn clear_current_stream(_device: usize) {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use cudarc::driver::CudaContext;

    /// Helper: create a context for device 0. Skips the test if no GPU.
    fn test_ctx() -> Option<Arc<CudaContext>> {
        CudaContext::new(0).ok()
    }

    #[test]
    fn event_record_sync() {
        let Some(ctx) = test_ctx() else { return };
        let stream = ctx.default_stream();

        let event = CudaEventWrapper::new(&ctx)
            .expect("event creation should succeed");

        // Record on the default stream (which has no pending work).
        event.record(&stream).expect("record should succeed");

        // Synchronize should complete immediately (no work queued).
        event.synchronize().expect("synchronize should succeed");

        // Query should return true — all work is done.
        assert!(
            event.query().expect("query should succeed"),
            "event should be complete after synchronize"
        );
    }

    #[test]
    fn event_query_before_record() {
        let Some(ctx) = test_ctx() else { return };

        let event = CudaEventWrapper::new(&ctx)
            .expect("event creation should succeed");

        // A freshly created event with no work recorded. Per CUDA semantics,
        // cuEventQuery on an event that has never been recorded returns
        // CUDA_SUCCESS (it is considered "complete"). cudarc's is_complete()
        // wraps this.
        let complete = event.query().expect("query should not error");
        // The event has no recorded work, so it reports complete.
        assert!(complete, "unrecorded event should report complete");
    }

    #[test]
    fn stream_pool_round_robin() {
        let Some(ctx) = test_ctx() else { return };
        // Use a high ordinal unlikely to collide with other tests.
        let dev = 0;

        let s1 = StreamPool::get_stream(&ctx, dev)
            .expect("first get_stream should succeed");
        let s2 = StreamPool::get_stream(&ctx, dev)
            .expect("second get_stream should succeed");

        // After STREAMS_PER_DEVICE calls, we should wrap around.
        let pool_size = StreamPool::pool_size(dev);
        assert!(pool_size > 0, "pool should have streams");
        assert!(pool_size <= STREAMS_PER_DEVICE, "pool should not exceed configured size");

        // Collect all streams from a full cycle.
        let mut streams = vec![s1, s2];
        for _ in 2..pool_size {
            streams.push(
                StreamPool::get_stream(&ctx, dev).expect("get_stream should succeed"),
            );
        }

        // The next stream should wrap around to the same as the first.
        let wrap = StreamPool::get_stream(&ctx, dev)
            .expect("wrapped get_stream should succeed");

        // Because round-robin, `wrap` should be the same Arc as `streams[0]`.
        // We compare the underlying cu_stream pointers.
        assert_eq!(
            Arc::as_ptr(&wrap),
            Arc::as_ptr(&streams[0]),
            "round-robin should wrap back to the first stream"
        );
    }

    #[test]
    fn stream_pool_invalid_device() {
        let Some(ctx) = test_ctx() else { return };
        let result = StreamPool::get_stream(&ctx, MAX_DEVICES + 1);
        assert!(result.is_err(), "should reject ordinal >= MAX_DEVICES");
    }

    #[test]
    fn stream_guard_restores_previous() {
        let Some(ctx) = test_ctx() else { return };
        let dev = 0;

        // Initially, no current stream.
        assert!(
            get_current_stream(dev).is_none(),
            "should start with no current stream"
        );

        let s1 = ctx.new_stream().expect("new_stream should succeed");
        let s2 = ctx.new_stream().expect("new_stream should succeed");

        let s1_ptr = Arc::as_ptr(&s1);
        let s2_ptr = Arc::as_ptr(&s2);

        // Set s1 as current.
        set_current_stream(dev, Arc::clone(&s1));
        assert_eq!(
            Arc::as_ptr(&get_current_stream(dev).unwrap()),
            s1_ptr,
            "current stream should be s1"
        );

        // Create a guard that sets s2.
        {
            let _guard = StreamGuard::new(dev, Arc::clone(&s2));
            assert_eq!(
                Arc::as_ptr(&get_current_stream(dev).unwrap()),
                s2_ptr,
                "current stream should be s2 inside guard"
            );
        }

        // After guard drop, s1 should be restored.
        assert_eq!(
            Arc::as_ptr(&get_current_stream(dev).unwrap()),
            s1_ptr,
            "current stream should be restored to s1 after guard drop"
        );

        // Clean up.
        clear_current_stream(dev);
        assert!(
            get_current_stream(dev).is_none(),
            "should be cleared after explicit clear"
        );
    }

    #[test]
    fn stream_guard_clears_when_no_previous() {
        let Some(ctx) = test_ctx() else { return };
        let dev = 0;

        // Ensure no current stream.
        clear_current_stream(dev);
        assert!(get_current_stream(dev).is_none());

        let s1 = ctx.new_stream().expect("new_stream should succeed");

        {
            let _guard = StreamGuard::new(dev, Arc::clone(&s1));
            assert!(
                get_current_stream(dev).is_some(),
                "guard should set current stream"
            );
        }

        // Guard had no previous — should clear.
        assert!(
            get_current_stream(dev).is_none(),
            "guard with no previous should clear current stream on drop"
        );
    }

    #[test]
    fn current_stream_or_default_fallback() {
        // We can't easily construct a GpuDevice in tests without a real GPU
        // context, but we can test the thread-local logic in isolation.
        let Some(ctx) = test_ctx() else { return };
        let dev_ordinal = 0;

        // Clear any leftover state.
        clear_current_stream(dev_ordinal);

        let device = crate::device::GpuDevice::new(dev_ordinal)
            .expect("GpuDevice::new should succeed");
        let default_ptr = Arc::as_ptr(device.default_stream());

        // No current stream set — should fall back to device default.
        let stream = current_stream_or_default(&device);
        assert_eq!(
            Arc::as_ptr(&stream),
            default_ptr,
            "should fall back to device default stream"
        );

        // Set a custom stream — should use it instead.
        let custom = ctx.new_stream().expect("new_stream should succeed");
        let custom_ptr = Arc::as_ptr(&custom);
        set_current_stream(dev_ordinal, custom);

        let stream = current_stream_or_default(&device);
        assert_eq!(
            Arc::as_ptr(&stream),
            custom_ptr,
            "should use thread-local current stream"
        );

        // Clean up.
        clear_current_stream(dev_ordinal);
    }

    #[test]
    fn event_wait_on_stream() {
        let Some(ctx) = test_ctx() else { return };
        let stream1 = ctx.default_stream();
        let stream2 = ctx.new_stream().expect("new_stream should succeed");

        let event = CudaEventWrapper::new(&ctx)
            .expect("event creation should succeed");

        // Record on stream1.
        event.record(&stream1).expect("record should succeed");

        // Make stream2 wait on the event (GPU-side sync).
        event.wait_on(&stream2).expect("wait_on should succeed");

        // Synchronize stream2 — this implicitly waits for stream1's work too.
        stream2.synchronize().expect("synchronize should succeed");
    }
}
