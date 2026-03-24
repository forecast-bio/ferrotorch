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
use cudarc::driver::{CudaSlice, CudaStream, DeviceRepr, ValidAsZeroBits};

use crate::error::{GpuError, GpuResult};

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
#[cfg(feature = "cuda")]
pub struct CapturedGraph {
    graph: cudarc::driver::CudaGraph,
}

#[cfg(feature = "cuda")]
impl CapturedGraph {
    /// Replay all operations captured in this graph.
    ///
    /// Before calling this, update any [`DeviceScalar`] values and perform
    /// any pre-launch memcpys (e.g., position embeddings). All updates must
    /// be on the same stream the graph was captured on.
    pub fn launch(&self) -> GpuResult<()> {
        self.graph.launch()?;
        Ok(())
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
    stream.begin_capture(
        cudarc::driver::sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL,
    )?;
    Ok(())
}

/// End CUDA graph capture, instantiate, and return the replayable graph.
///
/// Returns `Err` if capture was not active or if instantiation fails.
#[cfg(feature = "cuda")]
pub fn end_capture(stream: &Arc<CudaStream>) -> GpuResult<CapturedGraph> {
    let flags = cudarc::driver::sys::CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH;
    let graph = stream
        .end_capture(flags)?
        .ok_or(GpuError::PtxCompileFailed {
            kernel: "CUDA graph capture returned null",
        })?;
    Ok(CapturedGraph { graph })
}

// ---------------------------------------------------------------------------
// CapturePool — memory pool for graph capture
// ---------------------------------------------------------------------------

/// A dedicated memory pool for CUDA graph capture.
///
/// During graph capture, allocations must come from a pool that is not
/// sealed. Once sealed, the pool rejects new allocations — this is used
/// to ensure that all buffers are pre-allocated before capture begins.
///
/// # Usage
///
/// ```ignore
/// let pool = CapturePool::new();
/// // ... allocate buffers from pool ...
/// pool.seal();  // no more allocations allowed
///
/// // begin_capture_with_pool(&pool, stream) would fail here because
/// // the pool is sealed — you can't allocate during capture from a
/// // sealed pool. Un-seal it first or use a fresh pool.
/// ```
#[cfg(feature = "cuda")]
pub struct CapturePool {
    sealed: std::sync::atomic::AtomicBool,
}

#[cfg(feature = "cuda")]
impl CapturePool {
    /// Create a new, unsealed capture pool.
    pub fn new() -> Self {
        Self {
            sealed: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Seal the pool, preventing any further allocations.
    pub fn seal(&self) {
        self.sealed.store(true, std::sync::atomic::Ordering::Release);
    }

    /// Unseal the pool, allowing allocations again.
    pub fn unseal(&self) {
        self.sealed.store(false, std::sync::atomic::Ordering::Release);
    }

    /// Check whether the pool is sealed.
    pub fn is_capture_pool_sealed(&self) -> bool {
        self.sealed.load(std::sync::atomic::Ordering::Acquire)
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
pub fn begin_capture_with_pool(
    pool: &CapturePool,
    stream: &Arc<CudaStream>,
) -> GpuResult<()> {
    if pool.is_capture_pool_sealed() {
        return Err(GpuError::InvalidState {
            message: "cannot begin graph capture: capture pool is sealed".into(),
        });
    }
    begin_capture(stream)
}

/// Stub CapturePool when cuda feature is disabled.
#[cfg(not(feature = "cuda"))]
pub struct CapturePool;

#[cfg(not(feature = "cuda"))]
impl CapturePool {
    pub fn new() -> Self {
        Self
    }

    pub fn seal(&self) {}

    pub fn unseal(&self) {}

    pub fn is_capture_pool_sealed(&self) -> bool {
        false
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
}

#[cfg(not(feature = "cuda"))]
pub fn begin_capture<T>(_stream: &T) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

#[cfg(not(feature = "cuda"))]
pub fn end_capture<T>(_stream: &T) -> GpuResult<CapturedGraph> {
    Err(GpuError::NoCudaFeature)
}
