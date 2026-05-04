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
//! Each device gets `STREAMS_PER_DEVICE` (8) non-blocking streams created via
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

/// Number of streams per priority level in the priority pool.
#[cfg(feature = "cuda")]
const STREAMS_PER_PRIORITY: usize = 4;

/// Maximum supported device ordinal. Guards against unbounded allocation
/// if a caller passes a bogus ordinal.
#[cfg(feature = "cuda")]
const MAX_DEVICES: usize = 64;

// ---------------------------------------------------------------------------
// Stream priority — CL-322
// ---------------------------------------------------------------------------

/// Coarse-grained CUDA stream priority level.
///
/// Maps onto the device's reported priority range from
/// `cuCtxGetStreamPriorityRange`. CUDA's convention is that **lower**
/// integer values = **higher** priority, so [`StreamPriority::High`]
/// resolves to the device's reported "greatest priority" (numerically
/// smallest), and [`StreamPriority::Low`] resolves to the reported
/// "least priority" (numerically largest).
///
/// Devices that don't support stream priorities (range collapses to
/// `(0, 0)`) accept all three variants but produce streams of the same
/// effective priority.
///
/// CL-322.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StreamPriority {
    /// Highest priority — resolves to the device's greatest priority
    /// (numerically smallest int).
    High,
    /// Default priority — typically 0 on most devices.
    Normal,
    /// Lowest priority — resolves to the device's least priority
    /// (numerically largest int).
    Low,
}

impl StreamPriority {
    /// Resolve this enum to a concrete CUDA priority integer given a
    /// `(least_priority, greatest_priority)` range from
    /// [`get_stream_priority_range`]. CUDA's convention: lower int =
    /// higher priority.
    pub fn to_cuda_priority(self, range: (i32, i32)) -> i32 {
        let (least, greatest) = range;
        match self {
            StreamPriority::High => greatest,
            // Normal sits in the middle but clamped to [greatest, least]
            // so it's well-defined even if the range is collapsed or
            // inverted on weird drivers.
            StreamPriority::Normal => {
                if least == greatest {
                    least
                } else {
                    // Midpoint; the +/-1 handling for odd-length ranges
                    // doesn't matter for stream priority.
                    (least + greatest) / 2
                }
            }
            StreamPriority::Low => least,
        }
    }
}

/// Query the device's supported stream priority range.
///
/// Returns `(least_priority, greatest_priority)` where, by CUDA
/// convention, **lower** integer = **higher** scheduling priority.
/// On devices without priority support both values are 0. CL-322.
#[cfg(feature = "cuda")]
pub fn get_stream_priority_range(ctx: &Arc<CudaContext>) -> GpuResult<(i32, i32)> {
    use cudarc::driver::sys;
    ctx.bind_to_thread()?;
    let mut least: std::ffi::c_int = 0;
    let mut greatest: std::ffi::c_int = 0;
    // SAFETY: cuCtxGetStreamPriorityRange writes two ints; pointers
    // are valid stack locations. The current context is bound by the
    // call above.
    unsafe {
        sys::cuCtxGetStreamPriorityRange(&mut least as *mut _, &mut greatest as *mut _).result()?;
    }
    Ok((least, greatest))
}

// --- Layout mirror for cudarc::driver::CudaStream ---
//
// cudarc 0.19 declares `CudaStream` as a 2-field struct:
//
//     pub struct CudaStream {
//         pub(crate) cu_stream: sys::CUstream,
//         pub(crate) ctx: Arc<CudaContext>,
//     }
//
// The fields are not public, so we cannot construct a `CudaStream`
// from a raw `CUstream` produced by `cuStreamCreateWithPriority` via
// the safe API. Until cudarc upstream exposes a priority-aware
// constructor, we mirror the layout exactly here and use
// `mem::transmute` to convert. The const assertion below catches any
// future cudarc layout change at compile time so the conversion
// fails loudly at build time rather than silently producing UB.
//
// Workspace pins `cudarc = "0.19"` so the layout is stable for the
// lifetime of this minor version.
//
// CL-322.
#[cfg(feature = "cuda")]
struct CudaStreamMirror {
    _cu_stream: cudarc::driver::sys::CUstream,
    _ctx: Arc<CudaContext>,
}

#[cfg(feature = "cuda")]
const _CUDA_STREAM_LAYOUT_GUARD: () = {
    // The two-field struct mirrors cudarc::driver::CudaStream's
    // layout exactly on every supported target as long as the field
    // types and order match.
    assert!(
        std::mem::size_of::<CudaStreamMirror>() == std::mem::size_of::<CudaStream>(),
        "cudarc::driver::CudaStream layout has changed; update CudaStreamMirror"
    );
    assert!(
        std::mem::align_of::<CudaStreamMirror>() == std::mem::align_of::<CudaStream>(),
        "cudarc::driver::CudaStream alignment has changed; update CudaStreamMirror"
    );
};

/// Create a new CUDA stream with a specific priority.
///
/// Uses `cuStreamCreateWithPriority` under the hood. The `priority`
/// parameter must be within `get_stream_priority_range(ctx)` and is
/// silently clamped to the range otherwise. The returned stream is
/// always non-blocking with respect to the legacy default stream.
///
/// # Returns
///
/// `Arc<CudaStream>` interoperable with all the rest of the cudarc
/// API: kernel launches, event recording, synchronization, etc.
///
/// # Safety contract
///
/// This function uses `mem::transmute` to wrap the raw `CUstream`
/// into cudarc's `CudaStream`, because cudarc 0.19 does not expose a
/// public constructor for `CudaStream` from a raw pointer. The
/// transmute is bounded by a const layout-guard assertion against the
/// pinned cudarc version. See the `CudaStreamMirror` helper above for the
/// rationale and update procedure.
///
/// CL-322.
#[cfg(feature = "cuda")]
pub fn new_stream_with_priority(
    ctx: &Arc<CudaContext>,
    priority: StreamPriority,
) -> GpuResult<Arc<CudaStream>> {
    use cudarc::driver::sys;
    use std::mem::MaybeUninit;

    ctx.bind_to_thread()?;
    let range = get_stream_priority_range(ctx)?;
    let cuda_prio = priority.to_cuda_priority(range);

    // Create the raw stream with cuStreamCreateWithPriority.
    let mut raw_stream: MaybeUninit<sys::CUstream> = MaybeUninit::uninit();
    // SAFETY: out-pointer is a valid local; flags are CU_STREAM_NON_BLOCKING;
    // priority is clamped to the device's reported range.
    let res = unsafe {
        sys::cuStreamCreateWithPriority(
            raw_stream.as_mut_ptr(),
            sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
            cuda_prio,
        )
    };
    res.result()?;
    // SAFETY: cuStreamCreateWithPriority succeeded and wrote a valid
    // stream handle into raw_stream.
    let raw_stream = unsafe { raw_stream.assume_init() };

    // Build a layout-mirror, then transmute to the real type. See
    // CudaStreamMirror's docs for the safety rationale and the
    // const layout assertion.
    let mirror = CudaStreamMirror {
        _cu_stream: raw_stream,
        _ctx: ctx.clone(),
    };
    // SAFETY: CudaStreamMirror has the same field types in the same
    // order as cudarc 0.19's CudaStream, and the const assertion
    // _CUDA_STREAM_LAYOUT_GUARD verifies size and alignment match at
    // compile time. The cudarc dependency is pinned to "0.19" in
    // the workspace Cargo.toml so the layout is stable for the
    // lifetime of this build.
    let cuda_stream: CudaStream = unsafe { std::mem::transmute(mirror) };
    Ok(Arc::new(cuda_stream))
}

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
    /// when `None` is passed for flags). Use [`Self::new_with_timing`] if you need
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

    /// Compute the GPU-side elapsed time between this event (start)
    /// and `end` (end), in milliseconds.
    ///
    /// Both events must have been created with timing enabled
    /// ([`new_with_timing`](Self::new_with_timing)) and both must
    /// have been recorded before this call. The end event must also
    /// be reached on the GPU — call `end.synchronize()` first if you
    /// need a blocking measurement.
    ///
    /// Wraps `cuEventElapsedTime` (cudarc's `CudaEvent::elapsed_ms`).
    /// CL-380.
    pub fn elapsed_ms(&self, end: &Self) -> GpuResult<f32> {
        Ok(self.inner.elapsed_ms(&end.inner)?)
    }

    /// Compute the GPU-side elapsed time between this event and
    /// `end`, in **microseconds**, as an integer. Convenience for
    /// callers (like the profiler) that store durations in u64
    /// microseconds.
    ///
    /// Returns 0 for negative or sub-microsecond elapsed times.
    /// CL-380.
    pub fn elapsed_us(&self, end: &Self) -> GpuResult<u64> {
        let ms = self.elapsed_ms(end)?;
        if ms <= 0.0 {
            return Ok(0);
        }
        Ok((ms * 1000.0).round() as u64)
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
    STREAM_POOL.get_or_init(|| (0..MAX_DEVICES).map(|_| OnceLock::new()).collect())
}

/// Public interface for the CUDA stream pool.
pub struct StreamPool;

#[cfg(feature = "cuda")]
impl StreamPool {
    /// Get a stream for the given device, round-robin across the pool.
    ///
    /// On first call for a device ordinal, lazily creates `STREAMS_PER_DEVICE` (8)
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
    pub fn get_stream(ctx: &Arc<CudaContext>, device_ordinal: usize) -> GpuResult<Arc<CudaStream>> {
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

        let idx =
            device_streams.counter.fetch_add(1, Ordering::Relaxed) % device_streams.streams.len();
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

    /// Get a stream of the requested [`StreamPriority`] for the given
    /// device, round-robin across the priority pool. CL-322.
    ///
    /// On first call for a `(device, priority)` pair, lazily creates
    /// `STREAMS_PER_PRIORITY` streams via
    /// [`new_stream_with_priority`]. Subsequent calls reuse them
    /// round-robin.
    ///
    /// # Arguments
    ///
    /// * `ctx` — The CUDA context for the target device. Must match
    ///   `device_ordinal`.
    /// * `device_ordinal` — The GPU device index (0-based).
    /// * `priority` — The desired priority bucket.
    ///
    /// # Errors
    ///
    /// - Returns [`GpuError::InvalidDevice`] if
    ///   `device_ordinal >= MAX_DEVICES`.
    /// - Returns a CUDA driver error if stream creation fails.
    pub fn get_priority_stream(
        ctx: &Arc<CudaContext>,
        device_ordinal: usize,
        priority: StreamPriority,
    ) -> GpuResult<Arc<CudaStream>> {
        if device_ordinal >= MAX_DEVICES {
            return Err(GpuError::InvalidDevice {
                ordinal: device_ordinal,
                count: MAX_DEVICES,
            });
        }

        let slots = priority_pool_slots();
        let key = (device_ordinal, priority);
        let priority_streams = slots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .entry(key)
            .or_default()
            .clone();

        // Lazy: if the pool for this (device, priority) is empty,
        // populate it. Re-acquire the lock for the insert.
        if priority_streams.is_empty() {
            let mut new_streams = Vec::with_capacity(STREAMS_PER_PRIORITY);
            for _ in 0..STREAMS_PER_PRIORITY {
                match new_stream_with_priority(ctx, priority) {
                    Ok(s) => new_streams.push(s),
                    Err(_) => break,
                }
            }
            if new_streams.is_empty() {
                return Err(GpuError::Driver(cudarc::driver::DriverError(
                    cudarc::driver::sys::cudaError_enum::CUDA_ERROR_OUT_OF_MEMORY,
                )));
            }
            let mut guard = slots.lock().unwrap_or_else(|p| p.into_inner());
            // Race-safe: another thread may have populated meanwhile;
            // check before overwriting.
            let entry = guard.entry(key).or_default();
            if entry.is_empty() {
                *entry = new_streams.clone();
            }
            // Read back the (possibly other thread's) populated pool
            // so we serve a stable cloned snapshot below.
            let snapshot = entry.clone();
            drop(guard);
            // Round-robin pick from the snapshot.
            let idx = priority_pool_counter(key).fetch_add(1, Ordering::Relaxed) % snapshot.len();
            return Ok(Arc::clone(&snapshot[idx]));
        }

        let idx =
            priority_pool_counter(key).fetch_add(1, Ordering::Relaxed) % priority_streams.len();
        Ok(Arc::clone(&priority_streams[idx]))
    }

    /// Return the size of the priority pool for the given
    /// `(device_ordinal, priority)` pair, or 0 if not yet
    /// initialized. CL-322.
    pub fn priority_pool_size(device_ordinal: usize, priority: StreamPriority) -> usize {
        if device_ordinal >= MAX_DEVICES {
            return 0;
        }
        let slots = priority_pool_slots();
        slots
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .get(&(device_ordinal, priority))
            .map(|v| v.len())
            .unwrap_or(0)
    }
}

// Priority pool: a Mutex-protected map from (device_ordinal,
// StreamPriority) to a Vec of cached streams. We use a Mutex rather
// than per-key OnceLock because the key set is dynamic (3 priorities
// × MAX_DEVICES rather than a flat array indexed by ordinal). The
// critical section is short — a hash lookup + clone — so contention
// is negligible. CL-322.
#[cfg(feature = "cuda")]
type PriorityPoolMap = std::sync::Mutex<HashMap<(usize, StreamPriority), Vec<Arc<CudaStream>>>>;
#[cfg(feature = "cuda")]
type PriorityCounterMap = std::sync::Mutex<HashMap<(usize, StreamPriority), Arc<AtomicUsize>>>;

#[cfg(feature = "cuda")]
static PRIORITY_POOL: OnceLock<PriorityPoolMap> = OnceLock::new();

#[cfg(feature = "cuda")]
fn priority_pool_slots() -> &'static PriorityPoolMap {
    PRIORITY_POOL.get_or_init(|| std::sync::Mutex::new(HashMap::new()))
}

// Per-key round-robin counter for the priority pool. Stored as a
// `Mutex<HashMap>` to allow lazy creation; the get path itself uses
// `fetch_add` on the contained AtomicUsize so the lock is only held
// briefly during the (rare) first lookup per key.
#[cfg(feature = "cuda")]
static PRIORITY_POOL_COUNTERS: OnceLock<PriorityCounterMap> = OnceLock::new();

#[cfg(feature = "cuda")]
fn priority_pool_counter(key: (usize, StreamPriority)) -> Arc<AtomicUsize> {
    let map = PRIORITY_POOL_COUNTERS.get_or_init(|| std::sync::Mutex::new(HashMap::new()));
    let mut guard = map.lock().unwrap_or_else(|p| p.into_inner());
    Arc::clone(
        guard
            .entry(key)
            .or_insert_with(|| Arc::new(AtomicUsize::new(0))),
    )
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
    get_current_stream(device.ordinal()).unwrap_or_else(|| Arc::clone(device.default_stream()))
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

/// No-op without CUDA: there is no thread-local stream to set.
/// Provided so callers compile against the same API on both feature
/// configurations.
#[cfg(not(feature = "cuda"))]
pub fn set_current_stream(_device: usize, _stream: ()) {
    // Without the cuda feature there are no real streams to track.
    // The argument types are unit so there is nothing to store.
}

/// No-op without CUDA: there is no thread-local stream to clear.
/// Provided so callers compile against the same API on both feature
/// configurations.
#[cfg(not(feature = "cuda"))]
pub fn clear_current_stream(_device: usize) {
    // Without the cuda feature there are no real streams to track.
    // The thread-local map only exists in the cuda-enabled module.
}

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

        let event = CudaEventWrapper::new(&ctx).expect("event creation should succeed");

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

        let event = CudaEventWrapper::new(&ctx).expect("event creation should succeed");

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

        let s1 = StreamPool::get_stream(&ctx, dev).expect("first get_stream should succeed");
        let s2 = StreamPool::get_stream(&ctx, dev).expect("second get_stream should succeed");

        // After STREAMS_PER_DEVICE calls, we should wrap around.
        let pool_size = StreamPool::pool_size(dev);
        assert!(pool_size > 0, "pool should have streams");
        assert!(
            pool_size <= STREAMS_PER_DEVICE,
            "pool should not exceed configured size"
        );

        // Collect all streams from a full cycle.
        let mut streams = vec![s1, s2];
        for _ in 2..pool_size {
            streams.push(StreamPool::get_stream(&ctx, dev).expect("get_stream should succeed"));
        }

        // The next stream should wrap around to the same as the first.
        let wrap = StreamPool::get_stream(&ctx, dev).expect("wrapped get_stream should succeed");

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

        let device =
            crate::device::GpuDevice::new(dev_ordinal).expect("GpuDevice::new should succeed");
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

        let event = CudaEventWrapper::new(&ctx).expect("event creation should succeed");

        // Record on stream1.
        event.record(&stream1).expect("record should succeed");

        // Make stream2 wait on the event (GPU-side sync).
        event.wait_on(&stream2).expect("wait_on should succeed");

        // Synchronize stream2 — this implicitly waits for stream1's work too.
        stream2.synchronize().expect("synchronize should succeed");
    }

    // ── CL-322: stream priority ───────────────────────────────────────

    #[test]
    fn priority_range_returns_sane_values() {
        let Some(ctx) = test_ctx() else { return };
        let (least, greatest) =
            get_stream_priority_range(&ctx).expect("priority range should query successfully");
        // CUDA convention: lower int = higher priority, so
        // greatest <= least always holds. On devices that don't
        // support priority both are 0.
        assert!(
            greatest <= least,
            "priority range invariant violated: greatest={greatest} > least={least}"
        );
    }

    #[test]
    fn stream_priority_resolves_within_range() {
        // Synthetic range with three distinct levels.
        let range = (5, -5);
        assert_eq!(StreamPriority::High.to_cuda_priority(range), -5);
        assert_eq!(StreamPriority::Low.to_cuda_priority(range), 5);
        // Normal sits at the midpoint of the integer range. Both
        // greatest and least bracket it.
        let normal = StreamPriority::Normal.to_cuda_priority(range);
        assert!((-5..=5).contains(&normal));
    }

    #[test]
    fn stream_priority_collapsed_range_resolves_to_zero() {
        let range = (0, 0);
        assert_eq!(StreamPriority::High.to_cuda_priority(range), 0);
        assert_eq!(StreamPriority::Normal.to_cuda_priority(range), 0);
        assert_eq!(StreamPriority::Low.to_cuda_priority(range), 0);
    }

    #[test]
    fn new_stream_with_priority_succeeds_for_all_three_levels() {
        let Some(ctx) = test_ctx() else { return };
        let high = new_stream_with_priority(&ctx, StreamPriority::High)
            .expect("high-priority stream creation should succeed");
        let normal = new_stream_with_priority(&ctx, StreamPriority::Normal)
            .expect("normal-priority stream creation should succeed");
        let low = new_stream_with_priority(&ctx, StreamPriority::Low)
            .expect("low-priority stream creation should succeed");

        // The three streams must be distinct (sanity check that we
        // didn't accidentally return the same Arc).
        assert_ne!(Arc::as_ptr(&high), Arc::as_ptr(&normal));
        assert_ne!(Arc::as_ptr(&normal), Arc::as_ptr(&low));
        assert_ne!(Arc::as_ptr(&high), Arc::as_ptr(&low));
    }

    #[test]
    fn new_stream_with_priority_actually_runs_kernels() {
        // Create a high-priority stream and synchronize it. If our
        // transmute conversion is layout-incorrect, cudarc would
        // segfault here on the synchronize() call.
        let Some(ctx) = test_ctx() else { return };
        let stream = new_stream_with_priority(&ctx, StreamPriority::High)
            .expect("high-priority stream creation should succeed");
        stream.synchronize().expect("synchronize should succeed");
    }

    #[test]
    fn priority_pool_caches_streams_per_device_and_priority() {
        let Some(ctx) = test_ctx() else { return };
        let dev = 0;

        // Get a few streams of each priority. The pool should serve
        // up to STREAMS_PER_PRIORITY distinct streams per priority,
        // then round-robin.
        let _h1 = StreamPool::get_priority_stream(&ctx, dev, StreamPriority::High)
            .expect("get_priority_stream High should succeed");
        let _l1 = StreamPool::get_priority_stream(&ctx, dev, StreamPriority::Low)
            .expect("get_priority_stream Low should succeed");

        // Pool sizes should be > 0 after first access.
        let high_size = StreamPool::priority_pool_size(dev, StreamPriority::High);
        let low_size = StreamPool::priority_pool_size(dev, StreamPriority::Low);
        assert!(high_size > 0, "high-priority pool should have streams");
        assert!(low_size > 0, "low-priority pool should have streams");
        assert!(high_size <= STREAMS_PER_PRIORITY);
        assert!(low_size <= STREAMS_PER_PRIORITY);
    }

    #[test]
    fn priority_pool_invalid_device() {
        let Some(ctx) = test_ctx() else { return };
        let result = StreamPool::get_priority_stream(&ctx, 9999, StreamPriority::High);
        assert!(matches!(result, Err(GpuError::InvalidDevice { .. })));
    }
}
