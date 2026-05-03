//! Concrete [`CubeStorageHandle`] implementation for ferrotorch-cubecl.
//!
//! [`CubeclStorageHandle`] wraps a `cubecl::server::Handle` together with an
//! `Arc<CubeRuntime>` so the runtime (and thus the device memory) stays alive
//! as long as any tensor holds a reference to this handle.
//!
//! Core defines the [`CubeStorageHandle`] trait; this crate provides the only
//! concrete implementation, keeping the core→cubecl dependency one-way.
//!
//! Issue #673: device-resident XPU storage.

use std::sync::Arc;

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
use ferrotorch_core::FerrotorchError;
use ferrotorch_core::FerrotorchResult;
use ferrotorch_core::storage::CubeStorageHandle;

use crate::runtime::CubeRuntime;

// ---------------------------------------------------------------------------
// CubeclStorageHandle
// ---------------------------------------------------------------------------

/// Device-resident buffer handle for CubeCL-backed tensors.
///
/// Holds:
/// - A `cubecl::server::Handle` pointing to GPU memory.
/// - An `Arc<CubeRuntime>` so the device client stays alive.
/// - `len`: element count (`f32` elements).
/// - `ordinal`: device ordinal.
///
/// This is the concrete type stored inside `StorageBuffer::Cubecl` for XPU
/// tensors. Constructed by [`upload_f32`].
#[derive(Debug)]
pub struct CubeclStorageHandle {
    handle: cubecl::server::Handle,
    runtime: Arc<CubeRuntime>,
    len: usize,
    ordinal: usize,
}

impl CubeclStorageHandle {
    /// Construct a handle from its parts (internal use by `upload_f32`).
    #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
    fn new(
        handle: cubecl::server::Handle,
        runtime: Arc<CubeRuntime>,
        len: usize,
        ordinal: usize,
    ) -> Self {
        Self {
            handle,
            runtime,
            len,
            ordinal,
        }
    }

    /// Construct a handle from a raw `cubecl::server::Handle` returned by a
    /// kernel launcher.
    ///
    /// Used by `ferrotorch-xpu`'s `wrap_result_handle` to turn the
    /// `(cubecl::server::Handle, shape)` pair from `portable_*` into a
    /// `CubeclStorageHandle` without an extra H2D upload. Issue #673.
    pub fn from_raw(
        handle: cubecl::server::Handle,
        runtime: Arc<CubeRuntime>,
        len: usize,
        ordinal: usize,
    ) -> Self {
        Self {
            handle,
            runtime,
            len,
            ordinal,
        }
    }

    /// Borrow the raw `cubecl::server::Handle`.
    ///
    /// Used by `ops.rs` to pass handles directly to kernel launchers without
    /// an extra H2D upload. Issue #673.
    pub fn raw_handle(&self) -> &cubecl::server::Handle {
        &self.handle
    }

    /// Borrow the `CubeRuntime` this handle belongs to.
    pub fn runtime(&self) -> &Arc<CubeRuntime> {
        &self.runtime
    }
}

impl CubeStorageHandle for CubeclStorageHandle {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn len(&self) -> usize {
        self.len
    }

    fn ordinal(&self) -> usize {
        self.ordinal
    }

    fn read_to_host(&self) -> FerrotorchResult<Vec<f32>> {
        // `read_f32s` is only available when a backend feature is compiled in.
        // Without a backend, this path is unreachable because the handle can
        // only be constructed via `upload_f32`, which also requires a feature.
        #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
        {
            self.runtime.read_f32s(self.handle.clone(), self.len)
        }
        #[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
        {
            Err(FerrotorchError::DeviceUnavailable)
        }
    }

    fn clone_handle(&self) -> Box<dyn CubeStorageHandle> {
        // `cubecl::server::Handle` clone is cheap — it bumps an internal
        // ref count in the cubecl server's handle table, not a buffer copy.
        Box::new(CubeclStorageHandle {
            handle: self.handle.clone(),
            runtime: Arc::clone(&self.runtime),
            len: self.len,
            ordinal: self.ordinal,
        })
    }
}

// ---------------------------------------------------------------------------
// Result-wrapping helper — for ferrotorch-xpu to use without depending on cubecl directly
// ---------------------------------------------------------------------------

/// Wrap the `(cubecl::server::Handle, Vec<usize>)` result of a `portable_*`
/// kernel call into a `CubeclStorageHandle`.
///
/// `ferrotorch-xpu` uses this so it never needs to name `cubecl::server::Handle`
/// directly (cubecl is not a direct dep of that crate). Issue #673.
pub fn wrap_kernel_output(
    handle: cubecl::server::Handle,
    shape: &[usize],
    runtime: Arc<CubeRuntime>,
    ordinal: usize,
) -> CubeclStorageHandle {
    let numel: usize = shape.iter().product();
    CubeclStorageHandle::from_raw(handle, runtime, numel, ordinal)
}

// ---------------------------------------------------------------------------
// H2D upload helper
// ---------------------------------------------------------------------------

/// Upload a host `f32` slice to device memory, returning a
/// [`CubeclStorageHandle`] wrapping the device-resident buffer.
///
/// This is the single H2D upload point for the XPU path. The caller wraps the
/// returned handle in `TensorStorage::xpu_from_handle` to produce XPU storage.
///
/// # Errors
///
/// Returns `DeviceUnavailable` if no backend feature is compiled in.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn upload_f32(
    data: &[f32],
    runtime: Arc<CubeRuntime>,
    ordinal: usize,
) -> FerrotorchResult<CubeclStorageHandle> {
    use crate::runtime::CubeClient;
    use cubecl::prelude::*;

    let bytes = f32::as_bytes(data);
    let handle = match runtime.client() {
        #[cfg(feature = "wgpu")]
        CubeClient::Wgpu(c) => c.create_from_slice(bytes),
        #[cfg(feature = "cuda")]
        CubeClient::Cuda(c) => c.create_from_slice(bytes),
        #[cfg(feature = "rocm")]
        CubeClient::Rocm(c) => c.create_from_slice(bytes),
    };
    Ok(CubeclStorageHandle::new(
        handle,
        runtime,
        data.len(),
        ordinal,
    ))
}

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
pub fn upload_f32(
    _data: &[f32],
    _runtime: Arc<CubeRuntime>,
    _ordinal: usize,
) -> FerrotorchResult<CubeclStorageHandle> {
    Err(FerrotorchError::DeviceUnavailable)
}

// ---------------------------------------------------------------------------
// Handle extraction helper — used by ops.rs to avoid re-uploading
// ---------------------------------------------------------------------------

/// Extract a `&CubeclStorageHandle` from a tensor's storage, if present.
///
/// Returns `None` when the tensor is not backed by a CubeCL device buffer
/// (e.g. it is a CPU tensor). Ops use this to route device-resident inputs
/// through the handle-direct kernel path (no H2D upload) vs. the slice-upload
/// fallback path for CPU tensors passed to an XPU op.
///
/// # Example (in ops.rs)
///
/// ```ignore
/// match (cubecl_handle_of(a), cubecl_handle_of(b)) {
///     (Some(ha), Some(hb)) => run_add_handles(client, ha, hb),
///     _ => { /* slice-upload fallback */ }
/// }
/// ```
pub fn cubecl_handle_of(t: &ferrotorch_core::Tensor<f32>) -> Option<&CubeclStorageHandle> {
    t.inner_storage_arc()
        .cubecl_handle()
        .and_then(|h| h.as_any().downcast_ref::<CubeclStorageHandle>())
}
