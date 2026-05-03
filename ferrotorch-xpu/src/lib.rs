//! Intel XPU backend for ferrotorch.
//!
//! `ferrotorch-xpu` exposes a [`XpuDevice`] handle that wraps a
//! [`ferrotorch_cubecl::CubeRuntime`] running on the wgpu backend, which
//! targets Intel GPUs (Arc series, Data Center GPU Max) through Vulkan.
//!
//! Post-issue #673: tensors with [`Device::Xpu(ordinal)`] hold a
//! **device-resident** CubeCL handle (`StorageBuffer::Cubecl`). Inputs are
//! uploaded once (H2D), kernel results remain on-device as handles, and data
//! is only read back when the user explicitly calls `.cpu()`. This matches
//! PyTorch's behaviour for CUDA tensors.
//!
//! ```ignore
//! use ferrotorch_core::Device;
//! use ferrotorch_xpu::XpuDevice;
//!
//! let xpu = XpuDevice::new(0)?;
//! let a = make_xpu_tensor(vec![1.0_f32, 2.0, 3.0], &[3], &xpu)?;
//! let b = make_xpu_tensor(vec![10.0_f32, 20.0, 30.0], &[3], &xpu)?;
//! let c = ferrotorch_xpu::xpu_add(&a, &b, &xpu)?;
//! // c is device-resident; no readback yet.
//! let data = c.cpu()?.data()?.to_vec();
//! assert_eq!(data, &[11.0, 22.0, 33.0]);
//! ```
//!
//! # Feature flags
//!
//! - `wgpu` (default): enables the cubecl wgpu runtime. Without this
//!   feature [`XpuDevice::new`] returns
//!   [`FerrotorchError::DeviceUnavailable`].

#[cfg(feature = "wgpu")]
use std::sync::Arc;

#[cfg(feature = "wgpu")]
use ferrotorch_core::TensorStorage;
use ferrotorch_core::{Device, FerrotorchError, FerrotorchResult, Tensor};

#[cfg(feature = "wgpu")]
use ferrotorch_cubecl::{CubeDevice, CubeRuntime, upload_f32, wrap_kernel_output};

// ---------------------------------------------------------------------------
// XpuDevice
// ---------------------------------------------------------------------------

/// A handle to an Intel XPU device.
///
/// `XpuDevice` owns a `CubeRuntime` configured for the wgpu backend.
/// All ops in this crate take an `&XpuDevice` so they can dispatch
/// kernels through the runtime's compute client without re-initialising
/// the backend on every call.
///
/// Issue #673.
#[derive(Debug, Clone)]
pub struct XpuDevice {
    ordinal: usize,
    #[cfg(feature = "wgpu")]
    runtime: Arc<CubeRuntime>,
}

impl XpuDevice {
    /// Initialise the XPU runtime for the device with the given ordinal.
    ///
    /// Without the `wgpu` feature this returns
    /// [`FerrotorchError::DeviceUnavailable`].
    pub fn new(ordinal: usize) -> FerrotorchResult<Self> {
        #[cfg(feature = "wgpu")]
        {
            let runtime = CubeRuntime::new(CubeDevice::Wgpu(ordinal))?;
            Ok(Self {
                ordinal,
                runtime: Arc::new(runtime),
            })
        }
        #[cfg(not(feature = "wgpu"))]
        {
            let _ = ordinal;
            Err(FerrotorchError::DeviceUnavailable)
        }
    }

    /// The XPU device ordinal this handle targets.
    #[inline]
    pub fn ordinal(&self) -> usize {
        self.ordinal
    }

    /// The matching `Device::Xpu(ordinal)` for use with
    /// [`Tensor::to`](ferrotorch_core::Tensor::to).
    #[inline]
    pub fn device(&self) -> Device {
        Device::Xpu(self.ordinal)
    }

    /// True when the wgpu feature is compiled in. Without it, every
    /// op in this crate returns `DeviceUnavailable`.
    #[inline]
    pub fn is_available() -> bool {
        cfg!(feature = "wgpu")
    }

    /// Borrow the underlying [`CubeRuntime`] (only with the `wgpu`
    /// feature). Useful for callers that want to drop down to the
    /// portable cubecl API for an op this crate hasn't wrapped yet.
    #[cfg(feature = "wgpu")]
    pub fn runtime(&self) -> &Arc<CubeRuntime> {
        &self.runtime
    }
}

impl core::fmt::Display for XpuDevice {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "xpu:{}", self.ordinal)
    }
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

/// Verify that a tensor lives on the XPU device managed by `xpu`.
#[cfg(feature = "wgpu")]
fn check_xpu_tensor(t: &Tensor<f32>, xpu: &XpuDevice) -> FerrotorchResult<()> {
    let actual = t.device();
    let expected = xpu.device();
    if actual != expected {
        return Err(FerrotorchError::DeviceMismatch {
            expected,
            got: actual,
        });
    }
    Ok(())
}

/// Build a fresh XPU tensor from `data` and `shape`.
///
/// This is the single H2D upload point for user-facing construction.
/// It allocates a device buffer directly — no intermediate CPU tensor.
/// Issue #673: eliminates the double-allocation from the pre-#673 path.
#[cfg(feature = "wgpu")]
pub fn make_xpu_tensor(
    data: Vec<f32>,
    shape: &[usize],
    xpu: &XpuDevice,
) -> FerrotorchResult<Tensor<f32>> {
    let handle = upload_f32(&data, Arc::clone(xpu.runtime()), xpu.ordinal())?;
    let storage = TensorStorage::xpu_from_handle(Box::new(handle), xpu.ordinal());
    Tensor::from_storage(storage, shape.to_vec(), false)
}

#[cfg(not(feature = "wgpu"))]
pub fn make_xpu_tensor(
    _data: Vec<f32>,
    _shape: &[usize],
    _xpu: &XpuDevice,
) -> FerrotorchResult<Tensor<f32>> {
    Err(FerrotorchError::DeviceUnavailable)
}

// ---------------------------------------------------------------------------
// Public ops
// ---------------------------------------------------------------------------

#[cfg(feature = "wgpu")]
macro_rules! xpu_binary {
    ($name:ident, $cubecl:path, $doc:literal) => {
        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns [`FerrotorchError::DeviceMismatch`] if either input tensor
        /// is not on the XPU device managed by `xpu`.  Returns any error
        /// propagated from the underlying CubeCL kernel or from shape
        /// validation.
        ///
        /// The result is device-resident; call `.cpu()` to read back.
        pub fn $name(
            a: &Tensor<f32>,
            b: &Tensor<f32>,
            xpu: &XpuDevice,
        ) -> FerrotorchResult<Tensor<f32>> {
            check_xpu_tensor(a, xpu)?;
            check_xpu_tensor(b, xpu)?;
            // portable_* routes through handle-direct path (no H2D upload)
            // when both inputs are device-resident. Issue #673.
            let (handle, shape) = $cubecl(a, b, xpu.runtime())?;
            let cube_handle =
                wrap_kernel_output(handle, &shape, Arc::clone(xpu.runtime()), xpu.ordinal());
            let storage = TensorStorage::xpu_from_handle(Box::new(cube_handle), xpu.ordinal());
            Tensor::from_storage(storage, shape, false)
        }
    };
}

#[cfg(feature = "wgpu")]
macro_rules! xpu_unary {
    ($name:ident, $cubecl:path, $doc:literal) => {
        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns [`FerrotorchError::DeviceMismatch`] if `x` is not on the
        /// XPU device managed by `xpu`.  Returns any error propagated from
        /// the underlying CubeCL kernel.
        ///
        /// The result is device-resident; call `.cpu()` to read back.
        pub fn $name(x: &Tensor<f32>, xpu: &XpuDevice) -> FerrotorchResult<Tensor<f32>> {
            check_xpu_tensor(x, xpu)?;
            let (handle, shape) = $cubecl(x, xpu.runtime())?;
            let cube_handle =
                wrap_kernel_output(handle, &shape, Arc::clone(xpu.runtime()), xpu.ordinal());
            let storage = TensorStorage::xpu_from_handle(Box::new(cube_handle), xpu.ordinal());
            Tensor::from_storage(storage, shape, false)
        }
    };
}

#[cfg(feature = "wgpu")]
macro_rules! xpu_polynomial {
    ($name:ident, $cubecl:path, $doc:literal) => {
        #[doc = $doc]
        ///
        /// # Errors
        ///
        /// Returns [`FerrotorchError::DeviceMismatch`] if `x` is not on the
        /// XPU device managed by `xpu`. Returns
        /// [`FerrotorchError::InvalidArgument`] if the degree exceeds the
        /// `u32` range. Returns any error propagated from the underlying
        /// CubeCL kernel.
        ///
        /// The result is device-resident; call `.cpu()` to read back.
        pub fn $name(x: &Tensor<f32>, n: usize, xpu: &XpuDevice) -> FerrotorchResult<Tensor<f32>> {
            check_xpu_tensor(x, xpu)?;
            let (handle, shape) = $cubecl(x, n, xpu.runtime())?;
            let cube_handle =
                wrap_kernel_output(handle, &shape, Arc::clone(xpu.runtime()), xpu.ordinal());
            let storage = TensorStorage::xpu_from_handle(Box::new(cube_handle), xpu.ordinal());
            Tensor::from_storage(storage, shape, false)
        }
    };
}

#[cfg(feature = "wgpu")]
xpu_binary!(
    xpu_add,
    ferrotorch_cubecl::ops::portable_add,
    "Elementwise `a + b` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_binary!(
    xpu_sub,
    ferrotorch_cubecl::ops::portable_sub,
    "Elementwise `a - b` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_binary!(
    xpu_mul,
    ferrotorch_cubecl::ops::portable_mul,
    "Elementwise `a * b` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_binary!(
    xpu_div,
    ferrotorch_cubecl::ops::portable_div,
    "Elementwise `a / b` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_binary!(
    xpu_matmul,
    ferrotorch_cubecl::ops::portable_matmul,
    "2-D matrix multiplication on the XPU."
);

#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_neg,
    ferrotorch_cubecl::ops::portable_neg,
    "Elementwise `-x` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_abs,
    ferrotorch_cubecl::ops::portable_abs,
    "Elementwise `|x|` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_relu,
    ferrotorch_cubecl::ops::portable_relu,
    "Elementwise `relu(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_exp,
    ferrotorch_cubecl::ops::portable_exp,
    "Elementwise `exp(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_ln,
    ferrotorch_cubecl::ops::portable_ln,
    "Elementwise `ln(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_sqrt,
    ferrotorch_cubecl::ops::portable_sqrt,
    "Elementwise `sqrt(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_sin,
    ferrotorch_cubecl::ops::portable_sin,
    "Elementwise `sin(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_cos,
    ferrotorch_cubecl::ops::portable_cos,
    "Elementwise `cos(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_tanh,
    ferrotorch_cubecl::ops::portable_tanh,
    "Elementwise `tanh(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_unary!(
    xpu_sigmoid,
    ferrotorch_cubecl::ops::portable_sigmoid,
    "Elementwise `sigmoid(x)` on the XPU."
);

// ---------------------------------------------------------------------------
// Orthogonal polynomial families — three-term recurrences with scalar `n`.
// Mirrors `ferrotorch_cubecl::ops::portable_*_polynomial_*`. (#674, #577)
// ---------------------------------------------------------------------------

#[cfg(feature = "wgpu")]
xpu_polynomial!(
    xpu_chebyshev_polynomial_t,
    ferrotorch_cubecl::ops::portable_chebyshev_polynomial_t,
    "Chebyshev polynomial of the first kind `T_n(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_polynomial!(
    xpu_chebyshev_polynomial_u,
    ferrotorch_cubecl::ops::portable_chebyshev_polynomial_u,
    "Chebyshev polynomial of the second kind `U_n(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_polynomial!(
    xpu_chebyshev_polynomial_v,
    ferrotorch_cubecl::ops::portable_chebyshev_polynomial_v,
    "Chebyshev polynomial of the third kind `V_n(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_polynomial!(
    xpu_chebyshev_polynomial_w,
    ferrotorch_cubecl::ops::portable_chebyshev_polynomial_w,
    "Chebyshev polynomial of the fourth kind `W_n(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_polynomial!(
    xpu_hermite_polynomial_h,
    ferrotorch_cubecl::ops::portable_hermite_polynomial_h,
    "Physicists' Hermite polynomial `H_n(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_polynomial!(
    xpu_hermite_polynomial_he,
    ferrotorch_cubecl::ops::portable_hermite_polynomial_he,
    "Probabilists' Hermite polynomial `He_n(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_polynomial!(
    xpu_laguerre_polynomial_l,
    ferrotorch_cubecl::ops::portable_laguerre_polynomial_l,
    "Laguerre polynomial `L_n(x)` on the XPU."
);
#[cfg(feature = "wgpu")]
xpu_polynomial!(
    xpu_legendre_polynomial_p,
    ferrotorch_cubecl::ops::portable_legendre_polynomial_p,
    "Legendre polynomial `P_n(x)` on the XPU."
);

// ---------------------------------------------------------------------------
// No-feature stubs
// ---------------------------------------------------------------------------

#[cfg(not(feature = "wgpu"))]
macro_rules! xpu_binary_stub {
    ($name:ident) => {
        /// Binary XPU op — unavailable because the `wgpu` feature is not enabled.
        ///
        /// # Errors
        ///
        /// Always returns [`FerrotorchError::DeviceUnavailable`].
        /// Enable the `wgpu` feature to get a real implementation.
        pub fn $name(
            _a: &Tensor<f32>,
            _b: &Tensor<f32>,
            _xpu: &XpuDevice,
        ) -> FerrotorchResult<Tensor<f32>> {
            Err(FerrotorchError::DeviceUnavailable)
        }
    };
}

#[cfg(not(feature = "wgpu"))]
macro_rules! xpu_unary_stub {
    ($name:ident) => {
        /// Unary XPU op — unavailable because the `wgpu` feature is not enabled.
        ///
        /// # Errors
        ///
        /// Always returns [`FerrotorchError::DeviceUnavailable`].
        /// Enable the `wgpu` feature to get a real implementation.
        pub fn $name(_x: &Tensor<f32>, _xpu: &XpuDevice) -> FerrotorchResult<Tensor<f32>> {
            Err(FerrotorchError::DeviceUnavailable)
        }
    };
}

#[cfg(not(feature = "wgpu"))]
macro_rules! xpu_polynomial_stub {
    ($name:ident) => {
        /// Polynomial XPU op — unavailable because the `wgpu` feature is not
        /// enabled.
        ///
        /// # Errors
        ///
        /// Always returns [`FerrotorchError::DeviceUnavailable`].
        /// Enable the `wgpu` feature to get a real implementation.
        pub fn $name(
            _x: &Tensor<f32>,
            _n: usize,
            _xpu: &XpuDevice,
        ) -> FerrotorchResult<Tensor<f32>> {
            Err(FerrotorchError::DeviceUnavailable)
        }
    };
}

#[cfg(not(feature = "wgpu"))]
xpu_binary_stub!(xpu_add);
#[cfg(not(feature = "wgpu"))]
xpu_binary_stub!(xpu_sub);
#[cfg(not(feature = "wgpu"))]
xpu_binary_stub!(xpu_mul);
#[cfg(not(feature = "wgpu"))]
xpu_binary_stub!(xpu_div);
#[cfg(not(feature = "wgpu"))]
xpu_binary_stub!(xpu_matmul);

#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_neg);
#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_abs);
#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_relu);
#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_exp);
#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_ln);
#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_sqrt);
#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_sin);
#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_cos);
#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_tanh);
#[cfg(not(feature = "wgpu"))]
xpu_unary_stub!(xpu_sigmoid);

#[cfg(not(feature = "wgpu"))]
xpu_polynomial_stub!(xpu_chebyshev_polynomial_t);
#[cfg(not(feature = "wgpu"))]
xpu_polynomial_stub!(xpu_chebyshev_polynomial_u);
#[cfg(not(feature = "wgpu"))]
xpu_polynomial_stub!(xpu_chebyshev_polynomial_v);
#[cfg(not(feature = "wgpu"))]
xpu_polynomial_stub!(xpu_chebyshev_polynomial_w);
#[cfg(not(feature = "wgpu"))]
xpu_polynomial_stub!(xpu_hermite_polynomial_h);
#[cfg(not(feature = "wgpu"))]
xpu_polynomial_stub!(xpu_hermite_polynomial_he);
#[cfg(not(feature = "wgpu"))]
xpu_polynomial_stub!(xpu_laguerre_polynomial_l);
#[cfg(not(feature = "wgpu"))]
xpu_polynomial_stub!(xpu_legendre_polynomial_p);

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;

    /// Probe whether an XPU (wgpu under the hood) device can be constructed
    /// in the current environment. Returns `None` when no wgpu adapter is
    /// available (e.g. WSL2 without Vulkan ICDs) so tests skip cleanly.
    fn xpu() -> Option<XpuDevice> {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| XpuDevice::new(0).ok()));
        match result {
            Ok(Some(d)) => Some(d),
            _ => None,
        }
    }

    /// Build a device-resident XPU tensor from a slice.
    fn xpu_tensor(data: &[f32], xpu: &XpuDevice) -> Tensor<f32> {
        make_xpu_tensor(data.to_vec(), &[data.len()], xpu).unwrap()
    }

    #[test]
    fn xpu_device_init_and_metadata() {
        let Some(xpu) = xpu() else { return };
        assert_eq!(xpu.ordinal(), 0);
        assert_eq!(xpu.device(), Device::Xpu(0));
        assert!(XpuDevice::is_available());
    }

    #[test]
    fn xpu_tensor_is_device_resident() {
        // Post-#673: XPU tensors must NOT expose data() without .cpu() first.
        let Some(xpu) = xpu() else { return };
        let t = xpu_tensor(&[1.0, 2.0, 3.0, 4.0], &xpu);
        assert_eq!(t.device(), Device::Xpu(0));
        assert_eq!(t.shape(), &[4]);
        // data() must error — tensor is device-resident.
        assert!(t.data().is_err(), "data() must return Err for XPU tensors");
        // .cpu() then .data() must succeed.
        let host = t.cpu().unwrap();
        assert_eq!(host.device(), Device::Cpu);
        assert_eq!(host.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn xpu_add_runs_on_gpu_and_tags_xpu_storage() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[1.0, 2.0, 3.0, 4.0], &xpu);
        let b = xpu_tensor(&[10.0, 20.0, 30.0, 40.0], &xpu);
        let c = xpu_add(&a, &b, &xpu).unwrap();
        assert_eq!(c.device(), Device::Xpu(0), "result must stay on XPU");
        assert_eq!(c.shape(), &[4]);
        // Must error directly; only readable via .cpu().
        assert!(c.data().is_err(), "data() must return Err on XPU result");
        let data = c.cpu().unwrap().data().unwrap().to_vec();
        assert_eq!(data, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn xpu_sub_mul_div_run_on_gpu() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[10.0, 20.0, 30.0], &xpu);
        let b = xpu_tensor(&[2.0, 4.0, 5.0], &xpu);

        let s = xpu_sub(&a, &b, &xpu).unwrap();
        assert_eq!(s.cpu().unwrap().data().unwrap(), &[8.0, 16.0, 25.0]);

        let m = xpu_mul(&a, &b, &xpu).unwrap();
        assert_eq!(m.cpu().unwrap().data().unwrap(), &[20.0, 80.0, 150.0]);

        let d = xpu_div(&a, &b, &xpu).unwrap();
        assert_eq!(d.cpu().unwrap().data().unwrap(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn xpu_matmul_runs_on_gpu() {
        let Some(xpu) = xpu() else { return };
        let a = make_xpu_tensor(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], &xpu).unwrap();
        let b = make_xpu_tensor(vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2], &xpu).unwrap();

        let c = xpu_matmul(&a, &b, &xpu).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.device(), Device::Xpu(0));
        let data = c.cpu().unwrap().data().unwrap().to_vec();
        assert!((data[0] - 58.0).abs() < 1e-4);
        assert!((data[1] - 64.0).abs() < 1e-4);
        assert!((data[2] - 139.0).abs() < 1e-4);
        assert!((data[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn xpu_unary_kernels_run_on_gpu() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[-3.0, -1.0, 0.0, 1.0, 3.0], &xpu);
        let n = xpu_neg(&a, &xpu).unwrap();
        assert_eq!(
            n.cpu().unwrap().data().unwrap(),
            &[3.0, 1.0, 0.0, -1.0, -3.0]
        );

        let abs = xpu_abs(&a, &xpu).unwrap();
        assert_eq!(
            abs.cpu().unwrap().data().unwrap(),
            &[3.0, 1.0, 0.0, 1.0, 3.0]
        );

        let r = xpu_relu(&a, &xpu).unwrap();
        assert_eq!(r.cpu().unwrap().data().unwrap(), &[0.0, 0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn xpu_transcendentals_run_on_gpu() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[0.0, 1.0, 2.0], &xpu);
        let e = xpu_exp(&a, &xpu).unwrap();
        let ed = e.cpu().unwrap().data().unwrap().to_vec();
        let expected: Vec<f32> = [0.0, 1.0, 2.0].iter().map(|x: &f32| x.exp()).collect();
        for (a, b) in ed.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "exp parity: got {a}, expected {b}");
        }

        let input_ln = xpu_tensor(&[1.0, std::f32::consts::E, 10.0], &xpu);
        let l = xpu_ln(&input_ln, &xpu).unwrap();
        let ld = l.cpu().unwrap().data().unwrap().to_vec();
        let exp_l: Vec<f32> = [1.0_f32, std::f32::consts::E, 10.0]
            .iter()
            .map(|x| x.ln())
            .collect();
        for (a, b) in ld.iter().zip(exp_l.iter()) {
            assert!((a - b).abs() < 1e-3, "ln parity: got {a}, expected {b}");
        }
    }

    #[test]
    fn xpu_add_rejects_cpu_input() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[1.0, 2.0], &xpu);
        let b = ferrotorch_core::tensor(&[1.0_f32, 2.0]).unwrap();
        let err = xpu_add(&a, &b, &xpu).unwrap_err();
        assert!(matches!(err, FerrotorchError::DeviceMismatch { .. }));
    }

    #[test]
    fn xpu_add_rejects_cuda_input_against_xpu_device() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[1.0, 2.0], &xpu);
        let b = ferrotorch_core::tensor(&[1.0_f32, 2.0]).unwrap();
        let err = xpu_add(&a, &b, &xpu).unwrap_err();
        assert!(matches!(err, FerrotorchError::DeviceMismatch { .. }));
    }

    #[test]
    fn xpu_chained_ops_stay_on_device() {
        // Verify that chaining two ops doesn't trigger any H2D/D2H round-trip.
        // a + b → c (device); c + a → d (device); only read d.cpu().
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[1.0, 2.0, 3.0], &xpu);
        let b = xpu_tensor(&[10.0, 20.0, 30.0], &xpu);
        let c = xpu_add(&a, &b, &xpu).unwrap();
        assert_eq!(c.device(), Device::Xpu(0));
        let d = xpu_add(&c, &a, &xpu).unwrap();
        assert_eq!(d.device(), Device::Xpu(0));
        let result = d.cpu().unwrap().data().unwrap().to_vec();
        assert_eq!(result, &[12.0, 24.0, 36.0]);
    }

    #[test]
    fn xpu_chebyshev_t_runs_on_gpu() {
        // T_3(x) = 4x^3 - 3x. At x = [0, 0.5, 1, -1] → [0, -1, 1, -1].
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[0.0, 0.5, 1.0, -1.0], &xpu);
        let r = xpu_chebyshev_polynomial_t(&a, 3, &xpu).unwrap();
        assert_eq!(r.device(), Device::Xpu(0));
        let data = r.cpu().unwrap().data().unwrap().to_vec();
        for (got, want) in data.iter().zip(&[0.0_f32, -1.0, 1.0, -1.0]) {
            assert!(
                (got - want).abs() < 1e-4,
                "T_3 parity: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn xpu_legendre_p_runs_on_gpu() {
        // P_2(x) = (3x^2 - 1)/2. At x = [0, 1, -1] → [-0.5, 1, 1].
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[0.0, 1.0, -1.0], &xpu);
        let r = xpu_legendre_polynomial_p(&a, 2, &xpu).unwrap();
        let data = r.cpu().unwrap().data().unwrap().to_vec();
        for (got, want) in data.iter().zip(&[-0.5_f32, 1.0, 1.0]) {
            assert!(
                (got - want).abs() < 1e-4,
                "P_2 parity: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn xpu_polynomial_rejects_cpu_input() {
        let Some(xpu) = xpu() else { return };
        let a = ferrotorch_core::tensor(&[0.0_f32, 0.5, 1.0]).unwrap();
        let err = xpu_chebyshev_polynomial_t(&a, 2, &xpu).unwrap_err();
        assert!(matches!(err, FerrotorchError::DeviceMismatch { .. }));
    }
}

#[cfg(all(test, not(feature = "wgpu")))]
mod no_backend_tests {
    use super::*;

    #[test]
    fn xpu_device_new_errors_without_wgpu() {
        let err = XpuDevice::new(0).unwrap_err();
        assert!(matches!(err, FerrotorchError::DeviceUnavailable));
        assert!(!XpuDevice::is_available());
    }
}
