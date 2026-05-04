//! CUDA graph capture and replay infrastructure.
//!
//! A CUDA graph records a sequence of GPU operations (kernel launches, memcpys)
//! and replays them as a single driver submission. This eliminates per-kernel
//! launch overhead (~70μs on WSL2, ~5μs on native Linux per call) by collapsing
//! hundreds of launches into one.
//!
//! # Usage
//!
//! ```ignore
//! use ferrotorch_gpu::graph::{DeviceScalar, begin_capture, end_capture};
//!
//! // Pre-allocate all buffers BEFORE capture
//! let mut out = alloc_zeros_f32(768, &device)?;
//!
//! // Parameters that change between replays go in DeviceScalar
//! let mut pos = DeviceScalar::new(device.stream(), 0u32)?;
//!
//! // Capture
//! begin_capture(device.stream())?;
//! gpu_add_into(&a, &b, &mut out, &device)?;  // recorded, not executed
//! let graph = end_capture(device.stream())?;
//!
//! // Replay loop
//! for i in 0..100 {
//!     pos.update(i as u32)?;  // memcpy before replay
//!     graph.launch()?;         // replay all captured ops
//! }
//! ```

#[cfg(feature = "cuda")]
use std::sync::Arc;
#[cfg(feature = "cuda")]
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(feature = "cuda")]
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::error::{GpuError, GpuResult};

// ---------------------------------------------------------------------------
// CaptureMode — typed wrapper over cudarc's CUstreamCaptureMode
// ---------------------------------------------------------------------------

/// Selects how CUDA graph capture serializes interactions with other
/// threads. Mirrors `cudaStreamCaptureMode`.
///
/// - `Global` — any CUDA API call from any thread that touches the
///   capturing stream (or any thread that is also capturing) will
///   invalidate capture. Safest for debugging; matches PyTorch's
///   default.
/// - `ThreadLocal` — only calls from the capturing thread can
///   invalidate capture. Other threads may freely use unrelated
///   streams. This is what ferrotorch-gpu has always used.
/// - `Relaxed` — the driver does not track cross-thread interactions
///   at all. Fastest, but the caller is fully responsible for making
///   sure no other thread interferes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum CaptureMode {
    /// Global serialization (`CU_STREAM_CAPTURE_MODE_GLOBAL`).
    Global,
    /// Thread-local serialization (`CU_STREAM_CAPTURE_MODE_THREAD_LOCAL`).
    /// This is the default in PyTorch's `cuda.graph` context.
    #[default]
    ThreadLocal,
    /// Relaxed — no cross-thread serialization
    /// (`CU_STREAM_CAPTURE_MODE_RELAXED`).
    Relaxed,
}

#[cfg(feature = "cuda")]
impl CaptureMode {
    /// Convert to the raw cudarc enum.
    #[inline]
    pub fn to_cuda(self) -> cudarc::driver::sys::CUstreamCaptureMode {
        use cudarc::driver::sys::CUstreamCaptureMode::*;
        match self {
            Self::Global => CU_STREAM_CAPTURE_MODE_GLOBAL,
            Self::ThreadLocal => CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
            Self::Relaxed => CU_STREAM_CAPTURE_MODE_RELAXED,
        }
    }
}

// ---------------------------------------------------------------------------
// CaptureStatus — typed wrapper over cudarc's CUstreamCaptureStatus
// ---------------------------------------------------------------------------

/// The capture state of a CUDA stream. Matches `cudaStreamCaptureStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CaptureStatus {
    /// The stream is not currently capturing any graph.
    None,
    /// The stream is actively capturing a graph.
    Active,
    /// Capture was invalidated (e.g., by a forbidden API call or a
    /// cross-stream dependency). The caller must call `end_capture`
    /// to discard the broken graph before doing anything else on the
    /// stream.
    Invalidated,
}

#[cfg(feature = "cuda")]
impl CaptureStatus {
    fn from_cuda(raw: cudarc::driver::sys::CUstreamCaptureStatus) -> Self {
        use cudarc::driver::sys::CUstreamCaptureStatus::*;
        match raw {
            CU_STREAM_CAPTURE_STATUS_NONE => Self::None,
            CU_STREAM_CAPTURE_STATUS_ACTIVE => Self::Active,
            CU_STREAM_CAPTURE_STATUS_INVALIDATED => Self::Invalidated,
        }
    }
}

impl CaptureStatus {
    /// Returns `true` if this stream is actively capturing a graph.
    #[inline]
    pub fn is_capturing(&self) -> bool {
        matches!(self, Self::Active)
    }

    /// Returns `true` if capture was invalidated and must be ended.
    #[inline]
    pub fn is_invalidated(&self) -> bool {
        matches!(self, Self::Invalidated)
    }
}

// ---------------------------------------------------------------------------
// DeviceScalar — a single value in GPU memory, updatable before graph replay
// ---------------------------------------------------------------------------

/// A single scalar value stored in GPU device memory.
///
/// Used for CUDA graph capture: the graph records the device pointer (fixed
/// address), and the caller updates the value via [`update`](DeviceScalar::update)
/// before each [`CapturedGraph::launch`]. The update is a 4-or-8 byte
/// `cuMemcpyHtoDAsync` — effectively zero cost.
#[cfg(feature = "cuda")]
pub struct DeviceScalar<T: DeviceRepr + ValidAsZeroBits + Copy> {
    buf: CudaSlice<T>,
    stream: Arc<CudaStream>,
}

#[cfg(feature = "cuda")]
impl<T: DeviceRepr + ValidAsZeroBits + Copy> DeviceScalar<T> {
    /// Allocate a device scalar with the given initial value.
    pub fn new(stream: &Arc<CudaStream>, initial: T) -> GpuResult<Self> {
        let buf = stream.clone_htod(&[initial])?;
        Ok(Self {
            buf,
            stream: Arc::clone(stream),
        })
    }

    /// Update the device value. This is an async H→D memcpy of `size_of::<T>()`
    /// bytes. Must be called on the same stream as the graph to ensure ordering.
    pub fn update(&mut self, value: T) -> GpuResult<()> {
        self.stream.memcpy_htod(&[value], &mut self.buf)?;
        Ok(())
    }

    /// Borrow the underlying `CudaSlice` for use as a kernel parameter.
    /// The graph captures this pointer address; updating the value later
    /// changes what the kernel reads without re-capturing.
    #[inline]
    pub fn inner(&self) -> &CudaSlice<T> {
        &self.buf
    }
}

// ---------------------------------------------------------------------------
// CapturedGraph — a replayable CUDA graph
// ---------------------------------------------------------------------------

/// A captured and instantiated CUDA graph that can be replayed with
/// [`launch`](CapturedGraph::launch).
///
/// Created via [`begin_capture`] + GPU ops + [`end_capture`].
/// The graph holds references to all device memory used during capture.
/// Those buffers must remain allocated for the lifetime of the graph.
///
/// **Allocator pool integration (CL-278).** When created via
/// [`end_capture_with_pool`], the graph holds a strong reference to
/// the [`CapturePool`] that recorded its allocations. The pool keeps
/// every registered buffer alive until the last `CapturedGraph`
/// referencing it is dropped, which guarantees the device pointers
/// recorded in the graph remain valid across replays. Without the
/// pool, callers must manually keep buffers alive (the original
/// [`end_capture`] API).
#[cfg(feature = "cuda")]
pub struct CapturedGraph {
    graph: cudarc::driver::CudaGraph,
    /// Optional reference to the pool that owns the graph's
    /// allocations. Some(pool) when constructed via
    /// [`end_capture_with_pool`]. Dropping the graph drops this
    /// Arc, which (if it's the last reference) drops every buffer
    /// the pool holds. CL-278.
    pool: Option<Arc<CapturePool>>,
    /// Monotonic counter bumped by every successful [`launch`]. Lets
    /// callers assert that a specific replay happened after some
    /// other work completed, useful for graph-aware profilers and
    /// integration tests. CL-454.
    replay_count: AtomicU64,
    /// True after the first successful [`upload`] so subsequent
    /// uploads become cheap no-ops. CL-454.
    uploaded: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "cuda")]
impl CapturedGraph {
    /// Replay all operations captured in this graph.
    ///
    /// Before calling this, update any [`DeviceScalar`] values and perform
    /// any pre-launch memcpys (e.g., position embeddings). All updates must
    /// be on the same stream the graph was captured on.
    ///
    /// Bumps [`num_replays`](Self::num_replays) on success.
    pub fn launch(&self) -> GpuResult<()> {
        self.graph.launch()?;
        self.replay_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }

    /// Pre-upload the graph's executable resources to the device.
    ///
    /// The first [`launch`](Self::launch) of a freshly instantiated graph
    /// pays a one-time cost for the driver to copy the exec into GPU
    /// memory. Calling `upload` up front shifts that cost out of the
    /// hot replay loop. Subsequent uploads are a no-op. CL-454.
    pub fn upload(&self) -> GpuResult<()> {
        if self.uploaded.load(Ordering::Acquire) {
            return Ok(());
        }
        self.graph.upload()?;
        self.uploaded.store(true, Ordering::Release);
        Ok(())
    }

    /// Number of successful replays issued on this graph. CL-454.
    #[inline]
    pub fn num_replays(&self) -> u64 {
        self.replay_count.load(Ordering::Relaxed)
    }

    /// Returns `true` if [`upload`](Self::upload) has been called on
    /// this graph. CL-454.
    #[inline]
    pub fn is_uploaded(&self) -> bool {
        self.uploaded.load(Ordering::Acquire)
    }

    /// Number of buffers held alive by this graph's allocator pool.
    /// Returns 0 if the graph was created without a pool. CL-278.
    pub fn pool_buffer_count(&self) -> usize {
        self.pool.as_ref().map(|p| p.buffer_count()).unwrap_or(0)
    }

    /// True if this graph holds a CapturePool reference. CL-278.
    pub fn has_pool(&self) -> bool {
        self.pool.is_some()
    }

    /// Return the [`Arc<CapturePool>`] this graph is using, if any.
    /// Allows sharing the same pool between multiple graphs so they
    /// all keep the same buffers alive. CL-454.
    pub fn pool(&self) -> Option<&Arc<CapturePool>> {
        self.pool.as_ref()
    }
}

// ---------------------------------------------------------------------------
// Capture API
// ---------------------------------------------------------------------------

/// Begin CUDA graph capture on the given stream.
///
/// All GPU operations (kernel launches, cuBLAS calls, memcpys) issued on this
/// stream after this call are **recorded but not executed**. Call
/// [`end_capture`] to finalize and instantiate the graph.
///
/// # Requirements
///
/// - All output buffers must be pre-allocated before capture begins.
/// - No `alloc_zeros` / `cpu_to_gpu` during capture (use `_into` variants).
/// - No CPU↔GPU synchronization during capture.
/// - Event tracking should be disabled during capture to avoid interference
///   (call `ctx.disable_event_tracking()` before, re-enable after).
#[cfg(feature = "cuda")]
pub fn begin_capture(stream: &Arc<CudaStream>) -> GpuResult<()> {
    begin_capture_with_mode(stream, CaptureMode::default())
}

/// Begin CUDA graph capture with an explicit [`CaptureMode`]. CL-454.
///
/// Prefer [`begin_capture`] for the default (`ThreadLocal`) mode. Use
/// this form when you need `Global` (debugging / strict serialization)
/// or `Relaxed` (max throughput, single-thread ownership).
#[cfg(feature = "cuda")]
pub fn begin_capture_with_mode(stream: &Arc<CudaStream>, mode: CaptureMode) -> GpuResult<()> {
    stream.begin_capture(mode.to_cuda())?;
    Ok(())
}

/// Query the capture status of a CUDA stream. CL-454.
///
/// This is the ferrotorch-gpu equivalent of PyTorch's
/// `torch.cuda.is_current_stream_capturing`. Callers can use this to
/// skip capture-invalid APIs (allocator calls, H↔D copies) when a
/// graph is being recorded.
#[cfg(feature = "cuda")]
pub fn capture_status(stream: &Arc<CudaStream>) -> GpuResult<CaptureStatus> {
    let raw = stream.capture_status()?;
    Ok(CaptureStatus::from_cuda(raw))
}

/// Shorthand for `capture_status(stream)?.is_capturing()`. CL-454.
#[cfg(feature = "cuda")]
pub fn is_stream_capturing(stream: &Arc<CudaStream>) -> GpuResult<bool> {
    Ok(capture_status(stream)?.is_capturing())
}

/// End CUDA graph capture, instantiate, and return the replayable graph.
///
/// Returns `Err` if capture was not active or if instantiation fails.
///
/// The returned graph has no [`CapturePool`] attached. The caller is
/// responsible for keeping the buffers used by the captured kernels
/// alive for the graph's lifetime. Use [`end_capture_with_pool`]
/// for the lifetime-managed variant.
#[cfg(feature = "cuda")]
pub fn end_capture(stream: &Arc<CudaStream>) -> GpuResult<CapturedGraph> {
    let flags = cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    let graph = stream.end_capture(flags)?.ok_or(GpuError::InvalidState {
        message: "CUDA graph capture returned null".to_string(),
    })?;
    Ok(CapturedGraph {
        graph,
        pool: None,
        replay_count: AtomicU64::new(0),
        uploaded: std::sync::atomic::AtomicBool::new(false),
    })
}

/// End CUDA graph capture and attach a [`CapturePool`] reference to
/// the resulting [`CapturedGraph`]. CL-278.
///
/// The pool's tracked buffers are kept alive for the lifetime of the
/// returned graph: dropping the graph drops its `Arc<CapturePool>`,
/// which (if it's the last reference) drops every buffer the pool
/// recorded. This guarantees that the device pointers recorded in
/// the captured graph remain valid across replays.
///
/// Use this in concert with [`CapturePool::record_buffer`]: allocate
/// every buffer used during capture before calling `begin_capture`,
/// register each one with the pool, run the kernels under capture,
/// then call `end_capture_with_pool(stream, pool)` to seal the
/// lifetime relationship.
#[cfg(feature = "cuda")]
pub fn end_capture_with_pool(
    stream: &Arc<CudaStream>,
    pool: Arc<CapturePool>,
) -> GpuResult<CapturedGraph> {
    let mut graph = end_capture(stream)?;
    graph.pool = Some(pool);
    Ok(graph)
}

// ---------------------------------------------------------------------------
// GraphCaptureGuard — RAII wrapper that ends capture on drop
// ---------------------------------------------------------------------------

/// RAII guard that runs CUDA graph capture in a scoped block.
///
/// Call [`GraphCaptureGuard::begin`] (or [`Self::begin_with_mode`] /
/// [`Self::begin_with_pool`]) to start capture; calling [`Self::finish`]
/// returns the instantiated graph. If the guard is dropped without calling
/// `finish` (for example because a kernel returned an error
/// mid-capture), its `Drop` impl best-effort-ends capture and
/// discards the resulting graph so the stream returns to a usable
/// state. CL-454.
///
/// This mirrors PyTorch's `with torch.cuda.graph(g): ...` context
/// manager semantics in Rust's RAII idiom.
///
/// # Example
///
/// ```ignore
/// use ferrotorch_gpu::graph::GraphCaptureGuard;
///
/// let mut guard = GraphCaptureGuard::begin(device.stream())?;
/// run_kernels()?; // any kernel launched on device.stream() is recorded
/// let graph = guard.finish()?;
/// graph.upload()?;
/// for _ in 0..1000 { graph.launch()?; }
/// ```
#[cfg(feature = "cuda")]
pub struct GraphCaptureGuard {
    stream: Arc<CudaStream>,
    /// Optional pool to attach when `finish` is called.
    pool: Option<Arc<CapturePool>>,
    /// Becomes `false` after [`finish`] consumes the guard, so `Drop`
    /// knows capture is already ended.
    active: bool,
}

#[cfg(feature = "cuda")]
impl GraphCaptureGuard {
    /// Begin graph capture on `stream` in the default
    /// [`CaptureMode::ThreadLocal`] mode. CL-454.
    pub fn begin(stream: &Arc<CudaStream>) -> GpuResult<Self> {
        Self::begin_with_mode(stream, CaptureMode::default())
    }

    /// Begin graph capture with an explicit [`CaptureMode`]. CL-454.
    pub fn begin_with_mode(stream: &Arc<CudaStream>, mode: CaptureMode) -> GpuResult<Self> {
        begin_capture_with_mode(stream, mode)?;
        Ok(Self {
            stream: Arc::clone(stream),
            pool: None,
            active: true,
        })
    }

    /// Begin graph capture bound to a [`CapturePool`]. The pool is
    /// attached to the resulting graph by [`Self::finish`]. CL-454.
    pub fn begin_with_pool(stream: &Arc<CudaStream>, pool: Arc<CapturePool>) -> GpuResult<Self> {
        begin_capture_with_pool(&pool, stream)?;
        Ok(Self {
            stream: Arc::clone(stream),
            pool: Some(pool),
            active: true,
        })
    }

    /// Finish capture and return the instantiated [`CapturedGraph`].
    ///
    /// Consumes the guard so `Drop` becomes a no-op. If a pool was
    /// attached at construction, the resulting graph is produced via
    /// [`end_capture_with_pool`] and holds the pool Arc for the
    /// lifetime of the graph.
    pub fn finish(mut self) -> GpuResult<CapturedGraph> {
        self.active = false;
        if let Some(pool) = self.pool.take() {
            end_capture_with_pool(&self.stream, pool)
        } else {
            end_capture(&self.stream)
        }
    }

    /// Report whether the stream this guard is bound to is still
    /// actively capturing. An unexpected `Invalidated` or `None`
    /// usually means a forbidden API call (alloc, sync, host copy)
    /// happened under capture.
    pub fn status(&self) -> GpuResult<CaptureStatus> {
        capture_status(&self.stream)
    }
}

#[cfg(feature = "cuda")]
impl Drop for GraphCaptureGuard {
    fn drop(&mut self) {
        if !self.active {
            return;
        }
        // Best-effort: discard the in-flight capture so the stream
        // becomes usable again. We ignore errors because we're in
        // Drop — the CapturedGraph result is immediately dropped.
        let _ = end_capture(&self.stream);
    }
}

// ---------------------------------------------------------------------------
// Graph pool handle registry — share a CapturePool across multiple graphs
// ---------------------------------------------------------------------------

/// Opaque handle for a pool registered with the process-wide graph
/// pool registry. Used to share the same buffer-lifetime pool across
/// multiple captured graphs without passing `Arc<CapturePool>` around
/// by hand. CL-454.
///
/// Mirrors PyTorch's `torch.cuda.graph_pool_handle()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct GraphPoolHandle(pub u64);

#[cfg(feature = "cuda")]
static NEXT_POOL_HANDLE: AtomicU64 = AtomicU64::new(1);

#[cfg(feature = "cuda")]
static POOL_REGISTRY: std::sync::OnceLock<
    std::sync::Mutex<std::collections::HashMap<u64, Arc<CapturePool>>>,
> = std::sync::OnceLock::new();

#[cfg(feature = "cuda")]
fn pool_registry() -> &'static std::sync::Mutex<std::collections::HashMap<u64, Arc<CapturePool>>> {
    POOL_REGISTRY.get_or_init(|| std::sync::Mutex::new(std::collections::HashMap::new()))
}

/// Allocate a fresh [`GraphPoolHandle`] and register a new
/// [`CapturePool`] under it in the process-wide registry. CL-454.
///
/// The handle can later be passed to [`capture_pool_for_handle`] to
/// retrieve the same `Arc<CapturePool>` from any thread, which lets
/// two independently captured graphs share the same buffer-keeping
/// pool.
#[cfg(feature = "cuda")]
pub fn graph_pool_handle() -> GraphPoolHandle {
    let id = NEXT_POOL_HANDLE.fetch_add(1, Ordering::Relaxed);
    let pool = Arc::new(CapturePool::new());
    let mut reg = pool_registry().lock().unwrap_or_else(|p| p.into_inner());
    reg.insert(id, pool);
    GraphPoolHandle(id)
}

/// Look up the [`CapturePool`] registered under `handle` and return
/// a strong `Arc` to it. Returns `None` if the handle was never
/// allocated or has been released via [`release_graph_pool_handle`].
/// CL-454.
#[cfg(feature = "cuda")]
pub fn capture_pool_for_handle(handle: GraphPoolHandle) -> Option<Arc<CapturePool>> {
    let reg = pool_registry().lock().unwrap_or_else(|p| p.into_inner());
    reg.get(&handle.0).cloned()
}

/// Drop the registry's strong reference to the pool behind `handle`.
/// Any [`CapturedGraph`] that holds its own Arc (for example via
/// [`end_capture_with_pool`]) keeps the pool alive until that graph
/// is dropped too. CL-454.
#[cfg(feature = "cuda")]
pub fn release_graph_pool_handle(handle: GraphPoolHandle) {
    let mut reg = pool_registry().lock().unwrap_or_else(|p| p.into_inner());
    reg.remove(&handle.0);
}

// ---------------------------------------------------------------------------
// make_graphed_callable — scoped capture over a closure
// ---------------------------------------------------------------------------

/// Capture the operations performed by `f` into a CUDA graph and
/// return the replayable graph. CL-454.
///
/// This is the ferrotorch-gpu equivalent of PyTorch's
/// `torch.cuda.make_graphed_callables` for the simple single-callable
/// case: the caller supplies a closure that runs all the GPU work to
/// capture, and the returned [`CapturedGraph`] can be replayed over
/// and over. The closure runs exactly once during capture, so all
/// per-call work (allocations, dtype decisions) that isn't valid
/// under capture must happen outside.
///
/// If the closure returns an error, capture is discarded and the
/// error is propagated.
#[cfg(feature = "cuda")]
pub fn make_graphed_callable<F>(
    stream: &Arc<CudaStream>,
    mode: CaptureMode,
    f: F,
) -> GpuResult<CapturedGraph>
where
    F: FnOnce() -> GpuResult<()>,
{
    let guard = GraphCaptureGuard::begin_with_mode(stream, mode)?;
    match f() {
        Ok(()) => guard.finish(),
        Err(e) => {
            // Guard drop ends capture and discards the graph.
            drop(guard);
            Err(e)
        }
    }
}

// ---------------------------------------------------------------------------
// CapturePool — memory pool for graph capture
// ---------------------------------------------------------------------------

/// A dedicated memory pool for CUDA graph capture.
///
/// Two responsibilities:
///
/// 1. **Sealed flag** — gates [`begin_capture_with_pool`] so the
///    caller can express "no more allocations after this point"
///    semantically. Sealed pools cannot satisfy new allocations
///    during capture.
///
/// 2. **Buffer lifetime tracking (CL-278)** — registered buffers
///    are kept alive by the pool itself, so they outlive any
///    [`CapturedGraph`] that holds an `Arc<CapturePool>`. Dropping
///    the graph drops the Arc, and dropping the last Arc drops
///    every registered buffer in registration order.
///
/// # Usage
///
/// ```ignore
/// use std::sync::Arc;
/// let pool = Arc::new(CapturePool::new());
///
/// // Allocate every buffer the captured kernels will read or
/// // write, and register each one with the pool so it stays alive
/// // for the graph's lifetime.
/// let mut buf_a = alloc_zeros_f32(1024, &device)?;
/// let mut buf_b = alloc_zeros_f32(1024, &device)?;
/// pool.record_buffer(buf_a.try_clone()?);
/// pool.record_buffer(buf_b.try_clone()?);
///
/// pool.seal();
/// begin_capture_with_pool(&pool, stream)?;
/// // ... launch kernels using buf_a and buf_b ...
/// let graph = end_capture_with_pool(stream, Arc::clone(&pool))?;
/// // Dropping `pool` here is safe — the graph holds its own Arc.
/// ```
#[cfg(feature = "cuda")]
pub struct CapturePool {
    sealed: std::sync::atomic::AtomicBool,
    /// Registered buffers (type-erased) kept alive for the graph's
    /// lifetime. Each entry is a Box<dyn Any + Send + Sync> wrapping
    /// the buffer's drop guard. CL-278.
    buffers: std::sync::Mutex<Vec<Box<dyn std::any::Any + Send + Sync + 'static>>>,
}

#[cfg(feature = "cuda")]
impl CapturePool {
    /// Create a new, unsealed capture pool.
    pub fn new() -> Self {
        Self {
            sealed: std::sync::atomic::AtomicBool::new(false),
            buffers: std::sync::Mutex::new(Vec::new()),
        }
    }

    /// Seal the pool, preventing any further allocations.
    pub fn seal(&self) {
        self.sealed
            .store(true, std::sync::atomic::Ordering::Release);
    }

    /// Unseal the pool, allowing allocations again.
    pub fn unseal(&self) {
        self.sealed
            .store(false, std::sync::atomic::Ordering::Release);
    }

    /// Check whether the pool is sealed.
    pub fn is_capture_pool_sealed(&self) -> bool {
        self.sealed.load(std::sync::atomic::Ordering::Acquire)
    }

    /// Register a buffer with the pool so it stays alive for the
    /// lifetime of any [`CapturedGraph`] that holds this pool.
    /// CL-278.
    ///
    /// `buffer` can be any type that owns GPU memory (typically
    /// `CudaBuffer<f32>`, `CudaBuffer<f64>`, or `Arc<CudaBuffer<T>>`).
    /// The pool stores it in a type-erased `Box<dyn Any + Send +
    /// Sync>` and drops it (in registration order) when the pool
    /// itself is dropped.
    ///
    /// Returns the index of the registered buffer for diagnostic
    /// purposes.
    pub fn record_buffer<B>(&self, buffer: B) -> usize
    where
        B: Send + Sync + 'static,
    {
        let mut guard = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
        let idx = guard.len();
        guard.push(Box::new(buffer));
        idx
    }

    /// Number of buffers currently registered with the pool. CL-278.
    pub fn buffer_count(&self) -> usize {
        self.buffers.lock().map(|g| g.len()).unwrap_or(0)
    }

    /// Drop every registered buffer immediately, in registration
    /// order. The pool itself remains usable; new buffers can still
    /// be registered after this call. CL-278.
    ///
    /// Use this when reusing a pool across multiple capture cycles.
    /// Calling clear while a [`CapturedGraph`] still holds an Arc
    /// to this pool is safe — the graph's strong reference keeps
    /// the pool struct alive, but the buffer slots are reset.
    pub fn clear_buffers(&self) {
        let mut guard = self.buffers.lock().unwrap_or_else(|p| p.into_inner());
        guard.clear();
    }
}

#[cfg(feature = "cuda")]
impl Default for CapturePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Begin CUDA graph capture with a capture pool.
///
/// Like [`begin_capture`], but checks that the capture pool is not sealed
/// before starting capture. A sealed pool cannot satisfy allocations
/// during graph capture, which would cause CUDA errors.
///
/// # Errors
///
/// Returns [`GpuError::InvalidArgument`](GpuError) if the pool is sealed.
/// Returns a CUDA driver error if `begin_capture` fails.
#[cfg(feature = "cuda")]
pub fn begin_capture_with_pool(pool: &CapturePool, stream: &Arc<CudaStream>) -> GpuResult<()> {
    if pool.is_capture_pool_sealed() {
        return Err(GpuError::InvalidState {
            message: "cannot begin graph capture: capture pool is sealed".into(),
        });
    }
    begin_capture(stream)
}

/// Stub CapturePool when cuda feature is disabled. Provides the
/// same surface API as the cuda-enabled type so callers compile on
/// both feature configurations.
#[cfg(not(feature = "cuda"))]
pub struct CapturePool;

#[cfg(not(feature = "cuda"))]
impl CapturePool {
    /// Create an empty CapturePool. Without the cuda feature the
    /// pool has no internal state to initialize.
    pub fn new() -> Self {
        Self
    }

    /// No-op without the cuda feature: there is no real CUDA pool
    /// to seal because no real allocations can happen.
    pub fn seal(&self) {
        // Without the cuda feature there is no allocator state to
        // mutate; the CapturePool exists only so callers can write
        // feature-portable code.
    }

    /// No-op without the cuda feature: there is no real CUDA pool
    /// to unseal because no real allocations can happen.
    pub fn unseal(&self) {
        // Without the cuda feature there is no allocator state to
        // mutate; the CapturePool exists only so callers can write
        // feature-portable code.
    }

    /// Always returns `false` without the cuda feature since there
    /// is no real pool that could be in either state.
    pub fn is_capture_pool_sealed(&self) -> bool {
        false
    }

    /// Always returns 0 without the cuda feature since no real
    /// allocations can be tracked. CL-278.
    pub fn buffer_count(&self) -> usize {
        0
    }
}

#[cfg(not(feature = "cuda"))]
impl Default for CapturePool {
    fn default() -> Self {
        Self::new()
    }
}

/// Stub begin_capture_with_pool when cuda feature is disabled.
#[cfg(not(feature = "cuda"))]
pub fn begin_capture_with_pool<T>(_pool: &CapturePool, _stream: &T) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Stubs when cuda feature is disabled
// ---------------------------------------------------------------------------

/// Stub DeviceScalar.
#[cfg(not(feature = "cuda"))]
pub struct DeviceScalar<T: Copy> {
    _phantom: std::marker::PhantomData<T>,
}

/// Stub CapturedGraph.
#[cfg(not(feature = "cuda"))]
pub struct CapturedGraph;

#[cfg(not(feature = "cuda"))]
impl CapturedGraph {
    pub fn launch(&self) -> GpuResult<()> {
        Err(GpuError::NoCudaFeature)
    }

    /// Stub upload for CL-454.
    pub fn upload(&self) -> GpuResult<()> {
        Err(GpuError::NoCudaFeature)
    }

    /// Stub num_replays — always 0 without the cuda feature. CL-454.
    pub fn num_replays(&self) -> u64 {
        0
    }

    /// Stub is_uploaded — always false without the cuda feature. CL-454.
    pub fn is_uploaded(&self) -> bool {
        false
    }
}

#[cfg(not(feature = "cuda"))]
pub fn begin_capture<T>(_stream: &T) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

/// Stub `begin_capture_with_mode` when the cuda feature is not enabled.
/// CL-454.
#[cfg(not(feature = "cuda"))]
pub fn begin_capture_with_mode<T>(_stream: &T, _mode: CaptureMode) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

/// Stub `capture_status` when the cuda feature is not enabled. CL-454.
#[cfg(not(feature = "cuda"))]
pub fn capture_status<T>(_stream: &T) -> GpuResult<CaptureStatus> {
    Err(GpuError::NoCudaFeature)
}

/// Stub `is_stream_capturing` when the cuda feature is not enabled.
/// CL-454.
#[cfg(not(feature = "cuda"))]
pub fn is_stream_capturing<T>(_stream: &T) -> GpuResult<bool> {
    Err(GpuError::NoCudaFeature)
}

#[cfg(not(feature = "cuda"))]
pub fn end_capture<T>(_stream: &T) -> GpuResult<CapturedGraph> {
    Err(GpuError::NoCudaFeature)
}

/// Stub `end_capture_with_pool` when the cuda feature is not enabled.
/// CL-278.
#[cfg(not(feature = "cuda"))]
pub fn end_capture_with_pool<T>(
    _stream: &T,
    _pool: std::sync::Arc<CapturePool>,
) -> GpuResult<CapturedGraph> {
    Err(GpuError::NoCudaFeature)
}

/// Stub `GraphCaptureGuard` when the cuda feature is not enabled. CL-454.
#[cfg(not(feature = "cuda"))]
pub struct GraphCaptureGuard {
    _never: core::convert::Infallible,
}

#[cfg(not(feature = "cuda"))]
impl GraphCaptureGuard {
    pub fn begin<T>(_stream: &T) -> GpuResult<Self> {
        Err(GpuError::NoCudaFeature)
    }

    pub fn begin_with_mode<T>(_stream: &T, _mode: CaptureMode) -> GpuResult<Self> {
        Err(GpuError::NoCudaFeature)
    }

    pub fn begin_with_pool<T>(_stream: &T, _pool: std::sync::Arc<CapturePool>) -> GpuResult<Self> {
        Err(GpuError::NoCudaFeature)
    }

    pub fn finish(self) -> GpuResult<CapturedGraph> {
        match self._never {}
    }

    pub fn status(&self) -> GpuResult<CaptureStatus> {
        match self._never {}
    }
}

/// Stub `graph_pool_handle` when the cuda feature is not enabled. CL-454.
#[cfg(not(feature = "cuda"))]
pub fn graph_pool_handle() -> GraphPoolHandle {
    GraphPoolHandle(0)
}

/// Stub `capture_pool_for_handle` when the cuda feature is not enabled.
/// CL-454.
#[cfg(not(feature = "cuda"))]
pub fn capture_pool_for_handle(_handle: GraphPoolHandle) -> Option<std::sync::Arc<CapturePool>> {
    None
}

/// Stub `release_graph_pool_handle` when the cuda feature is not enabled.
/// CL-454.
#[cfg(not(feature = "cuda"))]
pub fn release_graph_pool_handle(_handle: GraphPoolHandle) {
    // nothing to release
}

/// Stub `make_graphed_callable` when the cuda feature is not enabled.
/// CL-454.
#[cfg(not(feature = "cuda"))]
pub fn make_graphed_callable<T, F>(
    _stream: &T,
    _mode: CaptureMode,
    _f: F,
) -> GpuResult<CapturedGraph>
where
    F: FnOnce() -> GpuResult<()>,
{
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Tests — CL-278 capture pool buffer tracking
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    #[test]
    fn capture_pool_buffer_count_starts_at_zero() {
        let pool = CapturePool::new();
        assert_eq!(pool.buffer_count(), 0);
    }

    #[test]
    fn capture_pool_record_buffer_increments_count() {
        let pool = CapturePool::new();
        let buf_a: Vec<f32> = vec![0.0; 10];
        let idx = pool.record_buffer(buf_a);
        assert_eq!(idx, 0);
        assert_eq!(pool.buffer_count(), 1);

        let buf_b: Vec<f64> = vec![0.0; 5];
        let idx = pool.record_buffer(buf_b);
        assert_eq!(idx, 1);
        assert_eq!(pool.buffer_count(), 2);
    }

    #[test]
    fn capture_pool_clear_buffers_resets_count_but_keeps_pool() {
        let pool = CapturePool::new();
        pool.record_buffer(vec![0u8; 16]);
        pool.record_buffer(vec![0u8; 32]);
        assert_eq!(pool.buffer_count(), 2);
        pool.clear_buffers();
        assert_eq!(pool.buffer_count(), 0);
        // Pool is still usable.
        pool.record_buffer(vec![0u8; 8]);
        assert_eq!(pool.buffer_count(), 1);
    }

    #[test]
    fn capture_pool_drop_releases_registered_buffers() {
        // Use Arc to detect when the inner buffer is dropped.
        let buf = Arc::new(vec![1.0f32, 2.0, 3.0]);
        let pool = CapturePool::new();
        pool.record_buffer(Arc::clone(&buf));
        assert_eq!(Arc::strong_count(&buf), 2);
        drop(pool);
        // Pool dropped → recorded Arc dropped → strong count back to 1.
        assert_eq!(Arc::strong_count(&buf), 1);
    }

    #[test]
    fn capture_pool_records_heterogeneous_types() {
        let pool = CapturePool::new();
        pool.record_buffer(vec![0.0f32; 4]);
        pool.record_buffer(vec![0.0f64; 4]);
        pool.record_buffer(vec![0u8; 4]);
        pool.record_buffer(Arc::new(42i32));
        assert_eq!(pool.buffer_count(), 4);
    }

    #[test]
    fn capture_pool_seal_unseal() {
        let pool = CapturePool::new();
        assert!(!pool.is_capture_pool_sealed());
        pool.seal();
        assert!(pool.is_capture_pool_sealed());
        pool.unseal();
        assert!(!pool.is_capture_pool_sealed());
    }

    // -----------------------------------------------------------------------
    // CL-454 — CaptureMode / CaptureStatus / graph pool handle tests.
    //
    // These tests exercise the typed wrappers and the process-wide
    // pool-handle registry without requiring a real CUDA device.
    // Tests that actually touch a device live under the
    // `feature = "cuda-live"` gate (run them with
    //   cargo test -p ferrotorch-gpu --features cuda,cuda-live
    // on a machine with a functioning CUDA driver).
    // -----------------------------------------------------------------------

    #[test]
    fn capture_mode_default_is_thread_local() {
        assert_eq!(CaptureMode::default(), CaptureMode::ThreadLocal);
    }

    #[test]
    fn capture_mode_to_cuda_round_trip() {
        use cudarc::driver::sys::CUstreamCaptureMode::*;
        assert_eq!(CaptureMode::Global.to_cuda(), CU_STREAM_CAPTURE_MODE_GLOBAL);
        assert_eq!(
            CaptureMode::ThreadLocal.to_cuda(),
            CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        );
        assert_eq!(
            CaptureMode::Relaxed.to_cuda(),
            CU_STREAM_CAPTURE_MODE_RELAXED
        );
    }

    #[test]
    fn capture_status_is_capturing_only_when_active() {
        assert!(!CaptureStatus::None.is_capturing());
        assert!(CaptureStatus::Active.is_capturing());
        assert!(!CaptureStatus::Invalidated.is_capturing());
    }

    #[test]
    fn capture_status_is_invalidated_only_when_broken() {
        assert!(!CaptureStatus::None.is_invalidated());
        assert!(!CaptureStatus::Active.is_invalidated());
        assert!(CaptureStatus::Invalidated.is_invalidated());
    }

    #[test]
    fn capture_status_from_cuda_maps_all_variants() {
        use cudarc::driver::sys::CUstreamCaptureStatus::*;
        assert_eq!(
            CaptureStatus::from_cuda(CU_STREAM_CAPTURE_STATUS_NONE),
            CaptureStatus::None
        );
        assert_eq!(
            CaptureStatus::from_cuda(CU_STREAM_CAPTURE_STATUS_ACTIVE),
            CaptureStatus::Active
        );
        assert_eq!(
            CaptureStatus::from_cuda(CU_STREAM_CAPTURE_STATUS_INVALIDATED),
            CaptureStatus::Invalidated
        );
    }

    #[test]
    fn graph_pool_handle_allocates_unique_ids() {
        let h1 = graph_pool_handle();
        let h2 = graph_pool_handle();
        assert_ne!(h1, h2, "each call should return a fresh id");
        // Both handles should map back to a real pool.
        assert!(capture_pool_for_handle(h1).is_some());
        assert!(capture_pool_for_handle(h2).is_some());
        release_graph_pool_handle(h1);
        release_graph_pool_handle(h2);
    }

    #[test]
    fn graph_pool_handle_shares_single_pool_across_lookups() {
        let h = graph_pool_handle();
        let a = capture_pool_for_handle(h).expect("handle registered");
        let b = capture_pool_for_handle(h).expect("handle still registered");
        assert!(
            Arc::ptr_eq(&a, &b),
            "both lookups should return the same pool Arc"
        );

        // Register a buffer through one lookup; the other should see it.
        a.record_buffer(vec![1.0f32, 2.0]);
        assert_eq!(b.buffer_count(), 1);

        release_graph_pool_handle(h);
        // After release the registry no longer hands out the pool, but
        // existing Arcs keep it alive.
        assert!(capture_pool_for_handle(h).is_none());
        // The existing Arc still has its buffer.
        assert_eq!(a.buffer_count(), 1);
    }

    #[test]
    fn graph_pool_handle_release_is_idempotent() {
        let h = graph_pool_handle();
        assert!(capture_pool_for_handle(h).is_some());
        release_graph_pool_handle(h);
        release_graph_pool_handle(h); // second call is fine
        assert!(capture_pool_for_handle(h).is_none());
    }

    #[test]
    fn graph_pool_handle_unknown_id_returns_none() {
        // A fresh handle ID that was never registered.
        let fake = GraphPoolHandle(u64::MAX);
        assert!(capture_pool_for_handle(fake).is_none());
    }
}

// ---------------------------------------------------------------------------
// CL-454 — tests that don't need cudarc type info.
// ---------------------------------------------------------------------------

#[cfg(all(test, not(feature = "cuda")))]
mod no_cuda_tests {
    use super::*;

    #[test]
    fn capture_mode_and_status_exist_without_cuda_feature() {
        // The feature-portable types compile without the cuda feature
        // so callers can write cfg-free code.
        let _ = CaptureMode::default();
        assert!(!CaptureStatus::None.is_capturing());
        assert!(CaptureStatus::Active.is_capturing());
        assert!(CaptureStatus::Invalidated.is_invalidated());
    }

    #[test]
    fn graph_pool_handle_without_cuda_returns_sentinel() {
        let h = graph_pool_handle();
        assert_eq!(h.0, 0, "stub handle is always zero without cuda feature");
        assert!(capture_pool_for_handle(h).is_none());
        release_graph_pool_handle(h); // no-op
    }

    #[test]
    fn captured_graph_stub_num_replays_and_is_uploaded_are_zero() {
        let g = CapturedGraph;
        assert_eq!(g.num_replays(), 0);
        assert!(!g.is_uploaded());
    }
}
