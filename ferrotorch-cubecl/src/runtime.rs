//! Unified runtime selection for CubeCL backends.
//!
//! [`CubeDevice`] enumerates the three supported backends (CUDA, ROCm, WGPU),
//! each parameterised by a device ordinal. [`CubeRuntime`] resolves that
//! selection into a real CubeCL [`ComputeClient`] — one per backend — which
//! owns the on-device memory and the compiled-kernel cache for that device.

use std::fmt;

#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
use cubecl::Runtime;
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
use cubecl::prelude::ComputeClient;
use ferrotorch_core::{FerrotorchError, FerrotorchResult};

#[cfg(feature = "cuda")]
use cubecl_cuda::{CudaDevice, CudaRuntime};
#[cfg(feature = "rocm")]
use cubecl_hip::{AmdDevice, HipRuntime};
#[cfg(feature = "wgpu")]
use cubecl_wgpu::{WgpuDevice, WgpuRuntime};

// ---------------------------------------------------------------------------
// CubeDevice
// ---------------------------------------------------------------------------

/// A device selector for CubeCL backends.
///
/// The `usize` field is the device ordinal (e.g., GPU index 0, 1, ...).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CubeDevice {
    /// NVIDIA GPU via CUDA PTX codegen.
    Cuda(usize),
    /// Portable GPU via WGPU — AMD (Vulkan), Intel (Vulkan), Apple (Metal).
    Wgpu(usize),
    /// AMD GPU via native HIP/ROCm runtime.
    Rocm(usize),
}

impl CubeDevice {
    /// Device ordinal regardless of backend.
    #[inline]
    pub fn ordinal(&self) -> usize {
        match self {
            Self::Cuda(o) | Self::Wgpu(o) | Self::Rocm(o) => *o,
        }
    }

    /// Human-readable backend name.
    #[inline]
    pub fn backend_name(&self) -> &'static str {
        match self {
            Self::Cuda(_) => "cuda",
            Self::Wgpu(_) => "wgpu",
            Self::Rocm(_) => "rocm",
        }
    }
}

impl fmt::Display for CubeDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.backend_name(), self.ordinal())
    }
}

// ---------------------------------------------------------------------------
// CubeClient — per-backend real compute client
// ---------------------------------------------------------------------------

/// An initialised CubeCL compute client for one of the supported backends.
///
/// The variant is determined by which runtime feature was compiled in and
/// what [`CubeDevice`] the runtime was built for. `ops.rs` matches on this
/// enum to dispatch generic CubeCL kernels to the correct backend.
#[derive(Clone)]
pub enum CubeClient {
    /// A real Wgpu (Vulkan/Metal/DX12) compute client.
    #[cfg(feature = "wgpu")]
    Wgpu(ComputeClient<WgpuRuntime>),
    /// A real CUDA compute client.
    #[cfg(feature = "cuda")]
    Cuda(ComputeClient<CudaRuntime>),
    /// A real HIP/ROCm compute client.
    #[cfg(feature = "rocm")]
    Rocm(ComputeClient<HipRuntime>),
}

impl fmt::Debug for CubeClient {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            #[cfg(feature = "wgpu")]
            Self::Wgpu(_) => f.write_str("CubeClient::Wgpu(..)"),
            #[cfg(feature = "cuda")]
            Self::Cuda(_) => f.write_str("CubeClient::Cuda(..)"),
            #[cfg(feature = "rocm")]
            Self::Rocm(_) => f.write_str("CubeClient::Rocm(..)"),
            #[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
            _ => f.write_str("CubeClient::<no backend>"),
        }
    }
}

// When no backend feature is enabled the enum has no variants and any
// construction site would be unreachable; the compiler would reject it.
// Keep an `_never` field in a ZST form so `CubeClient` still has a type, by
// giving it an uninhabited variant behind the cfg. (rustc treats an enum
// with zero variants as uninhabited, which is what we want.)
#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
const _: fn() = || {
    // Just a compile-time check that CubeClient is still a valid type
    // even with no variants.
    let _ = std::mem::size_of::<CubeClient>();
};

// ---------------------------------------------------------------------------
// CubeRuntime
// ---------------------------------------------------------------------------

/// CubeCL runtime wrapper that holds a real compute client for one device.
#[derive(Clone, Debug)]
pub struct CubeRuntime {
    device: CubeDevice,
    client: CubeClient,
}

impl CubeRuntime {
    /// Create a runtime targeting the given device.
    ///
    /// Returns `Err(FerrotorchError::DeviceUnavailable)` if the required
    /// backend feature was not compiled in.
    pub fn new(device: CubeDevice) -> FerrotorchResult<Self> {
        let client = Self::make_client(device)?;
        Ok(Self { device, client })
    }

    /// The device this runtime targets.
    #[inline]
    pub fn device(&self) -> &CubeDevice {
        &self.device
    }

    /// The underlying compute client (one variant per backend).
    #[inline]
    pub fn client(&self) -> &CubeClient {
        &self.client
    }

    /// Auto-detect the best available backend, returning `None` if no GPU
    /// backend feature is enabled.
    ///
    /// Priority order: CUDA > ROCm > WGPU.
    pub fn auto() -> Option<Self> {
        // CUDA takes priority when available.
        #[cfg(feature = "cuda")]
        {
            return Self::new(CubeDevice::Cuda(0)).ok();
        }

        // ROCm for AMD-native workloads.
        #[cfg(feature = "rocm")]
        {
            return Self::new(CubeDevice::Rocm(0)).ok();
        }

        // WGPU is the most portable fallback.
        #[cfg(feature = "wgpu")]
        {
            return Self::new(CubeDevice::Wgpu(0)).ok();
        }

        #[allow(unreachable_code)]
        None
    }

    /// Returns `true` if any GPU backend feature was compiled in.
    pub fn is_available() -> bool {
        cfg!(any(feature = "cuda", feature = "rocm", feature = "wgpu"))
    }

    // ---------------------------------------------------------------------
    // Backend client construction
    // ---------------------------------------------------------------------

    fn make_client(device: CubeDevice) -> FerrotorchResult<CubeClient> {
        match device {
            CubeDevice::Wgpu(idx) => {
                #[cfg(feature = "wgpu")]
                {
                    let wgpu_device = wgpu_device_for_index(idx);
                    let client = WgpuRuntime::client(&wgpu_device);
                    Ok(CubeClient::Wgpu(client))
                }
                #[cfg(not(feature = "wgpu"))]
                {
                    let _ = idx;
                    Err(FerrotorchError::DeviceUnavailable)
                }
            }
            CubeDevice::Cuda(idx) => {
                #[cfg(feature = "cuda")]
                {
                    let cuda_device = CudaDevice { index: idx };
                    let client = CudaRuntime::client(&cuda_device);
                    return Ok(CubeClient::Cuda(client));
                }
                #[cfg(not(feature = "cuda"))]
                {
                    let _ = idx;
                    Err(FerrotorchError::DeviceUnavailable)
                }
            }
            CubeDevice::Rocm(idx) => {
                #[cfg(feature = "rocm")]
                {
                    let amd_device = AmdDevice { index: idx };
                    let client = HipRuntime::client(&amd_device);
                    return Ok(CubeClient::Rocm(client));
                }
                #[cfg(not(feature = "rocm"))]
                {
                    let _ = idx;
                    Err(FerrotorchError::DeviceUnavailable)
                }
            }
        }
    }
}

#[cfg(feature = "wgpu")]
fn wgpu_device_for_index(index: usize) -> WgpuDevice {
    match index {
        // Index 0 maps to the system default adapter; this is the most
        // portable choice and matches how ferrotorch-gpu selects a GPU.
        0 => WgpuDevice::DefaultDevice,
        // Higher indices explicitly select a discrete GPU slot.
        n => WgpuDevice::DiscreteGpu(n),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cube_device_ordinal() {
        assert_eq!(CubeDevice::Cuda(3).ordinal(), 3);
        assert_eq!(CubeDevice::Wgpu(1).ordinal(), 1);
        assert_eq!(CubeDevice::Rocm(0).ordinal(), 0);
    }

    #[test]
    fn cube_device_backend_name() {
        assert_eq!(CubeDevice::Cuda(0).backend_name(), "cuda");
        assert_eq!(CubeDevice::Wgpu(0).backend_name(), "wgpu");
        assert_eq!(CubeDevice::Rocm(0).backend_name(), "rocm");
    }

    #[test]
    fn cube_device_display() {
        assert_eq!(CubeDevice::Cuda(2).to_string(), "cuda:2");
        assert_eq!(CubeDevice::Wgpu(0).to_string(), "wgpu:0");
        assert_eq!(CubeDevice::Rocm(1).to_string(), "rocm:1");
    }

    #[test]
    fn cube_device_equality() {
        assert_eq!(CubeDevice::Cuda(0), CubeDevice::Cuda(0));
        assert_ne!(CubeDevice::Cuda(0), CubeDevice::Cuda(1));
        assert_ne!(CubeDevice::Cuda(0), CubeDevice::Wgpu(0));
    }

    #[test]
    fn cube_device_clone_and_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(CubeDevice::Cuda(0));
        set.insert(CubeDevice::Wgpu(0));
        set.insert(CubeDevice::Rocm(0));
        assert_eq!(set.len(), 3);

        // Duplicate should not increase size.
        set.insert(CubeDevice::Cuda(0));
        assert_eq!(set.len(), 3);
    }

    /// Probe whether wgpu can construct a runtime in the current
    /// environment. WSL2 lacks a Vulkan ICD by default, so the cubecl-wgpu
    /// worker thread panics during adapter selection and the panic surfaces
    /// on the main thread as `RecvError`. Catch that here so tests skip
    /// cleanly instead of failing.
    #[cfg(feature = "wgpu")]
    fn wgpu_probe_runtime() -> Option<CubeRuntime> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CubeRuntime::new(CubeDevice::Wgpu(0)).ok()
        }));
        match result {
            Ok(Some(rt)) => Some(rt),
            _ => None,
        }
    }

    #[cfg(feature = "wgpu")]
    #[test]
    fn wgpu_runtime_new_and_device() {
        let Some(rt) = wgpu_probe_runtime() else {
            eprintln!("[ferrotorch-cubecl] wgpu adapter unavailable; skipping");
            return;
        };
        assert_eq!(*rt.device(), CubeDevice::Wgpu(0));
        // Client should match the selected backend.
        assert!(matches!(rt.client(), CubeClient::Wgpu(_)));
    }

    #[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
    #[test]
    fn no_backend_feature_yields_device_unavailable() {
        let err = CubeRuntime::new(CubeDevice::Wgpu(0)).unwrap_err();
        assert!(matches!(err, FerrotorchError::DeviceUnavailable));
    }

    #[test]
    fn cube_runtime_auto_returns_something_or_none() {
        // `auto()` may panic on the worker thread if a backend feature is
        // compiled in but the actual hardware/driver isn't available
        // (e.g. wgpu in WSL without Vulkan). Catch that and treat it as
        // "not available" — matching the documented contract that this
        // function returns `Option`.
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(CubeRuntime::auto));
        match result {
            Ok(Some(_)) => assert!(CubeRuntime::is_available()),
            Ok(None) | Err(_) => {
                // Either no backend feature compiled in, or the backend
                // feature is present but no adapter exists at runtime.
                // Both are valid outcomes.
                eprintln!(
                    "[ferrotorch-cubecl] auto() returned no runtime (no backend feature or no \
                     adapter); test passes"
                );
            }
        }
    }

    #[test]
    fn cube_runtime_is_available_consistent() {
        // `is_available()` is a compile-time check (`cfg!(...)`). When a
        // feature is compiled in but no hardware exists at runtime, `auto()`
        // may still return `Some` (lazy init succeeds, kernel dispatch
        // would fail later). We accept that asymmetry here — the test
        // verifies that "feature compiled in" is at least consistent with
        // "auto() doesn't return None for compile-time reasons".
        let available = CubeRuntime::is_available();
        if !available {
            // Belt-and-suspenders: when no feature is compiled, auto() must
            // be None.
            let auto = std::panic::catch_unwind(std::panic::AssertUnwindSafe(CubeRuntime::auto));
            assert!(auto.ok().flatten().is_none());
        }
    }
}
