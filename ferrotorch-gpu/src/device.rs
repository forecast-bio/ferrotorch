//! CUDA device management.
//!
//! [`GpuDevice`] wraps a `cudarc::driver::CudaContext` and its default stream,
//! providing a safe, ergonomic entry point for all GPU operations.

#[cfg(feature = "cuda")]
use std::sync::Arc;

#[cfg(feature = "cuda")]
use cudarc::cublas::CudaBlas;
#[cfg(feature = "cuda")]
use cudarc::driver::{CudaContext, CudaStream};

#[cfg(not(feature = "cuda"))]
use crate::error::GpuError;
use crate::error::GpuResult;

/// Handle to a single CUDA GPU device.
///
/// Holds a CUDA context, default stream, and a **cached cuBLAS handle**.
/// The cuBLAS handle is created once and reused for all matmul/bmm ops,
/// eliminating the ~1.7ms `cuModuleLoadData` overhead that occurs when
/// creating a new `CudaBlas` per operation.
#[cfg(feature = "cuda")]
pub struct GpuDevice {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    blas: CudaBlas,
    ordinal: usize,
}

#[cfg(feature = "cuda")]
impl GpuDevice {
    /// Initialize the CUDA device at the given ordinal.
    ///
    /// Creates a fresh `CudaContext`, takes its default stream, and
    /// constructs a cached `CudaBlas` handle bound to that stream so
    /// subsequent matmul/bmm ops reuse it instead of paying the
    /// `cuModuleLoadData` cost per call.
    pub fn new(ordinal: usize) -> GpuResult<Self> {
        let ctx = CudaContext::new(ordinal)?;
        let stream = ctx.default_stream();
        let blas = CudaBlas::new(stream.clone())?;
        Ok(Self {
            ctx,
            stream,
            blas,
            ordinal,
        })
    }

    /// Create a `GpuDevice` with a non-blocking stream forked from the
    /// given device's default stream. The forked stream supports CUDA graph
    /// capture (which the legacy default stream does not).
    pub fn fork_for_capture(parent: &GpuDevice) -> GpuResult<Self> {
        let stream = parent.stream.fork()?;
        let blas = CudaBlas::new(stream.clone())?;
        Ok(Self {
            ctx: Arc::clone(&parent.ctx),
            stream,
            blas,
            ordinal: parent.ordinal,
        })
    }

    /// The shared `CudaContext` underlying this device.
    ///
    /// Required by `cudarc::driver::CudaModule` loaders and other low-level
    /// APIs that need a context handle separate from the stream.
    #[inline]
    pub fn context(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// The device's default (legacy) stream.
    ///
    /// Prefer [`current_stream`](Self::current_stream) which respects the
    /// thread-local stream override set by [`StreamGuard`].
    #[inline]
    pub fn default_stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    /// The active stream for this device on the current thread.
    ///
    /// Returns the thread-local stream set by [`StreamGuard`] if one is
    /// active, otherwise falls back to the device's default stream. All
    /// kernel launches and memory operations should use this.
    #[inline]
    pub fn stream(&self) -> Arc<CudaStream> {
        crate::stream::current_stream_or_default(self)
    }

    /// The cached cuBLAS handle — reused for all matmul/bmm operations.
    #[inline]
    pub fn blas(&self) -> &CudaBlas {
        &self.blas
    }

    /// The 0-based ordinal of this CUDA device, as reported by the driver.
    #[inline]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

#[cfg(feature = "cuda")]
impl Clone for GpuDevice {
    fn clone(&self) -> Self {
        let blas =
            CudaBlas::new(self.stream.clone()).expect("CudaBlas::new failed in GpuDevice::clone");
        Self {
            ctx: Arc::clone(&self.ctx),
            stream: Arc::clone(&self.stream),
            blas,
            ordinal: self.ordinal,
        }
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for GpuDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuDevice")
            .field("ordinal", &self.ordinal)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Stub when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub `GpuDevice` when the `cuda` feature is not enabled.
///
/// Every method returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
#[derive(Clone, Debug)]
pub struct GpuDevice {
    ordinal: usize,
}

#[cfg(not(feature = "cuda"))]
impl GpuDevice {
    /// Always returns an error — compile with `features = ["cuda"]`.
    pub fn new(ordinal: usize) -> GpuResult<Self> {
        let _ = ordinal;
        Err(GpuError::NoCudaFeature)
    }

    /// The device ordinal.
    #[inline]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}
