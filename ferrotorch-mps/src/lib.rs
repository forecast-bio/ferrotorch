//! Apple Silicon Metal Performance Shaders (MPS) backend marker crate. (#451)
//!
//! This crate ships only the `Device::Mps(_)` marker plumbing — the
//! platform-detection surface, the `MpsDevice` ordinal handle, and the
//! initialization entry point. **No Metal kernels live here yet** and no
//! Metal dependency is wired (`metal` / `objc2-metal` are deliberately
//! absent). All entry points return [`FerrotorchError::DeviceUnavailable`]
//! on every platform until the kernel layer lands; see issue #451 for the
//! tracking plan.
//!
//! # Why every entry point is unavailable
//!
//! ferrotorch is a `PyTorch` reimplementation. `PyTorch`'s MPS backend
//! executes real Metal compute on Apple Silicon. Until ferrotorch ships
//! that — MSL kernels for each `GpuBackend` trait method, a macOS CI
//! runner to validate them, and the `objc2-metal` dep to launch them —
//! the honest implementation of a skeleton is *"explicitly unavailable
//! everywhere,"* not *"platform-conditional silent success."* That avoids
//! the `init_*` lying-success stub anti-pattern (a function whose name
//! implies success-with-side-effect actually returns `Ok(())` without
//! doing the registration, then every downstream op fails).
//!
//! # API surface that already works
//!
//! - [`Device::Mps(_)`](ferrotorch_core::Device::Mps) variant in
//!   `ferrotorch-core` is in place; tensor construction with that device
//!   variant returns an explicit error directing callers to the MPS
//!   backend.
//! - This crate's [`MpsDevice`] type carries the ordinal and implements
//!   the standard derives (`Debug`, `Clone`, `Copy`, `PartialEq`, `Eq`,
//!   `Hash`) plus `Display` (`mps:{ordinal}`), so callers can plumb MPS
//!   device handles through generic code today even though no kernel
//!   will run.
//!
//! # When the kernel layer lands
//!
//! Issue #451 tracks the work. At that point [`is_mps_available`],
//! [`mps_device_count`], [`MpsDevice::new`], and [`init_mps_backend`]
//! gain real macOS-conditional bodies (gated by an actual `metal`
//! dependency, validated by macOS CI). The signatures here are stable
//! so that change is purely additive.

#![warn(clippy::all, clippy::pedantic)]
#![deny(unsafe_code, rust_2018_idioms, missing_debug_implementations)]
// Pedantic lints we explicitly accept across this crate. Each allow names a
// concrete reason — the alternative would be churn-for-zero-benefit or a
// worse API. Add to this list only with a one-line justification.
#![allow(
    // The crate's name is `ferrotorch-mps` and its types naturally repeat the
    // `Mps` token (`MpsDevice`, `mps_device_count`, `init_mps_backend`); the
    // repetition is the disambiguator that prevents glob-import collisions
    // with sibling backends like `ferrotorch-gpu`.
    clippy::module_name_repetitions,
)]
#![deny(missing_docs)]

use core::fmt;

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

/// Returns `true` if this build can run MPS kernels.
///
/// Always returns `false` until the kernel layer (#451) lands. There is no
/// platform-conditional path here — that would be a lying-success stub
/// because no kernels are wired regardless of platform.
#[must_use]
pub fn is_mps_available() -> bool {
    false
}

/// An opaque handle for an Apple-Silicon Metal device.
///
/// `MpsDevice` is `Copy` because it wraps a single `usize`. Construction
/// always fails with [`FerrotorchError::DeviceUnavailable`] until the
/// kernel layer (#451) lands; the type is exported now so downstream
/// generic code can name it in signatures and trait bounds today.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MpsDevice {
    ordinal: usize,
}

impl MpsDevice {
    /// Try to construct a device handle for the given ordinal.
    ///
    /// # Errors
    ///
    /// Always returns [`FerrotorchError::DeviceUnavailable`] until the MPS
    /// kernel layer lands (issue #451).
    pub fn new(_ordinal: usize) -> FerrotorchResult<Self> {
        Err(FerrotorchError::DeviceUnavailable)
    }

    /// Number of MPS devices the system reports.
    ///
    /// Always `0` until the kernel layer (#451) lands. Provided as an
    /// associated function in addition to the free [`mps_device_count`]
    /// for callers that prefer the type-anchored spelling.
    #[must_use]
    pub fn count() -> usize {
        0
    }

    /// Device ordinal (0 = system default GPU).
    #[must_use]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }
}

impl fmt::Display for MpsDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mps:{}", self.ordinal)
    }
}

/// Number of MPS devices the system reports.
///
/// Renamed from `device_count` to `mps_device_count` to avoid colliding
/// with `ferrotorch_gpu::device_count` when both backends are re-exported
/// via the `ferrotorch::{gpu, mps}` namespaces. Mirrors `PyTorch`'s
/// module-scoped `torch.cuda.device_count()` / `torch.mps.device_count()`
/// idiom rather than the type-anchored Rust idiom.
///
/// Always returns `0` until the kernel layer (#451) lands.
#[must_use]
pub fn mps_device_count() -> usize {
    0
}

/// Initialize the MPS backend.
///
/// # Errors
///
/// Always returns [`FerrotorchError::DeviceUnavailable`] until the MPS
/// kernel layer lands. The previous implementation returned `Ok(())`
/// without registering any backend, then every downstream op failed with
/// `DeviceUnavailable` from the empty dispatch table — a lying-success
/// stub. Returning the error here is the honest shape: the dispatcher
/// reports no implementation rather than carrying a fake registration.
///
/// `FerrotorchError::DeviceUnavailable` is a unit variant in
/// `ferrotorch-core` and does not carry a structured message; the
/// pointer to the tracking issue (#451) lives in this crate's docs and
/// in the `FerrotorchError::DeviceUnavailable` `Display` text. Adding a
/// new error variant just to carry the message would be a workspace-level
/// coordination event (cross-crate `match` surface change) that this
/// dispatch is explicitly forbidden from making.
pub fn init_mps_backend() -> FerrotorchResult<()> {
    Err(FerrotorchError::DeviceUnavailable)
}

#[cfg(test)]
mod tests {
    use super::{FerrotorchError, MpsDevice, init_mps_backend, is_mps_available, mps_device_count};

    #[test]
    fn is_mps_available_is_always_false() {
        assert!(!is_mps_available());
    }

    #[test]
    fn mps_device_new_always_unavailable() {
        assert!(matches!(
            MpsDevice::new(0),
            Err(FerrotorchError::DeviceUnavailable)
        ));
    }

    #[test]
    fn mps_device_count_is_zero() {
        assert_eq!(mps_device_count(), 0);
        assert_eq!(MpsDevice::count(), 0);
    }

    #[test]
    fn init_mps_backend_always_unavailable() {
        assert!(matches!(
            init_mps_backend(),
            Err(FerrotorchError::DeviceUnavailable)
        ));
    }

    #[test]
    fn device_mps_marker_round_trips() {
        // ferrotorch-core exposes Device::Mps(_) regardless of MPS
        // availability — the variant just doesn't do anything useful
        // without the backend.
        let d = ferrotorch_core::Device::Mps(0);
        assert!(d.is_mps());
        assert_eq!(format!("{d}"), "mps:0");
    }
}
