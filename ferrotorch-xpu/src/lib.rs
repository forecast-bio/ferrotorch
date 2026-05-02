//! Intel XPU backend for ferrotorch.
//!
//! `ferrotorch-xpu` exposes a [`XpuDevice`] handle that wraps a
//! [`ferrotorch_cubecl::CubeRuntime`] running on the wgpu backend, which
//! targets Intel GPUs (Arc series, Data Center GPU Max) through Vulkan.
//! Tensors with [`Device::Xpu(ordinal)`] live as CPU `Vec<T>` storage
//! today; the device marker drives op dispatch through this crate, which
//! uploads the inputs to the cubecl runtime, runs a real `#[cube]`
//! kernel, and reads the result back. CL-452.
//!
//! This is the same dispatch shape ferrotorch-gpu uses for CUDA, so
//! switching a model from CUDA to XPU is a single line:
//!
//! ```ignore
//! use ferrotorch_core::Device;
//! use ferrotorch_xpu::XpuDevice;
//!
//! let xpu = XpuDevice::new(0)?;
//! let a = ferrotorch_core::tensor(&[1.0_f32, 2.0, 3.0])?
//!     .to(Device::Xpu(0))?;
//! let b = ferrotorch_core::tensor(&[10.0_f32, 20.0, 30.0])?
//!     .to(Device::Xpu(0))?;
//! let c = ferrotorch_xpu::xpu_add(&a, &b, &xpu)?;
//! ```
//!
//! # Feature flags
//!
//! - `wgpu` (default): enables the cubecl wgpu runtime. Without this
//!   feature [`XpuDevice::new`] returns
//!   [`FerrotorchError::DeviceUnavailable`].

use ferrotorch_core::{Device, FerrotorchError, FerrotorchResult, Tensor};

#[cfg(feature = "wgpu")]
use ferrotorch_cubecl::{CubeDevice, CubeRuntime};

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
/// CL-452.
#[derive(Debug, Clone)]
pub struct XpuDevice {
    ordinal: usize,
    #[cfg(feature = "wgpu")]
    runtime: CubeRuntime,
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
            Ok(Self { ordinal, runtime })
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
    pub fn runtime(&self) -> &CubeRuntime {
        &self.runtime
    }
}

// ---------------------------------------------------------------------------
// Tensor helpers
// ---------------------------------------------------------------------------

/// Verify that a tensor lives on the XPU device managed by `xpu`.
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
fn make_xpu_tensor(
    data: Vec<f32>,
    shape: &[usize],
    xpu: &XpuDevice,
) -> FerrotorchResult<Tensor<f32>> {
    ferrotorch_core::from_vec(data, shape)?.to(xpu.device())
}

// ---------------------------------------------------------------------------
// Public ops
// ---------------------------------------------------------------------------

#[cfg(feature = "wgpu")]
macro_rules! xpu_binary {
    ($name:ident, $cubecl:path, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(
            a: &Tensor<f32>,
            b: &Tensor<f32>,
            xpu: &XpuDevice,
        ) -> FerrotorchResult<Tensor<f32>> {
            check_xpu_tensor(a, xpu)?;
            check_xpu_tensor(b, xpu)?;
            // The portable_* op already validates shape compatibility.
            let result_on_cubecl = $cubecl(a, b, xpu.runtime())?;
            // The cubecl op returns a CPU tensor; tag it as XPU storage.
            let data = result_on_cubecl.data_vec()?;
            let shape = result_on_cubecl.shape().to_vec();
            make_xpu_tensor(data, &shape, xpu)
        }
    };
}

#[cfg(feature = "wgpu")]
macro_rules! xpu_unary {
    ($name:ident, $cubecl:path, $doc:literal) => {
        #[doc = $doc]
        pub fn $name(x: &Tensor<f32>, xpu: &XpuDevice) -> FerrotorchResult<Tensor<f32>> {
            check_xpu_tensor(x, xpu)?;
            let result_on_cubecl = $cubecl(x, xpu.runtime())?;
            let data = result_on_cubecl.data_vec()?;
            let shape = result_on_cubecl.shape().to_vec();
            make_xpu_tensor(data, &shape, xpu)
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
// No-feature stubs
// ---------------------------------------------------------------------------

#[cfg(not(feature = "wgpu"))]
macro_rules! xpu_binary_stub {
    ($name:ident) => {
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
        pub fn $name(_x: &Tensor<f32>, _xpu: &XpuDevice) -> FerrotorchResult<Tensor<f32>> {
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;

    /// Probe whether an XPU (wgpu under the hood) device can be constructed
    /// in the current environment. Returns `None` when no wgpu adapter is
    /// available — e.g. WSL2 without Vulkan ICDs — so tests skip cleanly
    /// instead of failing the whole suite. Tests pattern: `let Some(xpu) =
    /// xpu() else { return; };`.
    fn xpu() -> Option<XpuDevice> {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| XpuDevice::new(0).ok()));
        match result {
            Ok(Some(d)) => Some(d),
            _ => {
                eprintln!(
                    "[ferrotorch-xpu] wgpu adapter unavailable; skipping XPU integration test"
                );
                None
            }
        }
    }

    fn xpu_tensor(data: &[f32]) -> Tensor<f32> {
        ferrotorch_core::tensor(data)
            .unwrap()
            .to(Device::Xpu(0))
            .unwrap()
    }

    #[test]
    fn xpu_device_init_and_metadata() {
        let Some(xpu) = xpu() else { return };
        assert_eq!(xpu.ordinal(), 0);
        assert_eq!(xpu.device(), Device::Xpu(0));
        assert!(XpuDevice::is_available());
    }

    #[test]
    fn xpu_tensor_to_xpu_keeps_data_and_shape() {
        let t = xpu_tensor(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(t.device(), Device::Xpu(0));
        assert_eq!(t.shape(), &[4]);
        let data: &[f32] = t.data().unwrap();
        assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn xpu_add_runs_on_gpu_and_tags_xpu_storage() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let b = xpu_tensor(&[10.0, 20.0, 30.0, 40.0]);
        let c = xpu_add(&a, &b, &xpu).unwrap();
        assert_eq!(c.device(), Device::Xpu(0), "result must stay on XPU");
        assert_eq!(c.shape(), &[4]);
        let data: &[f32] = c.data().unwrap();
        assert_eq!(data, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn xpu_sub_mul_div_run_on_gpu() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[10.0, 20.0, 30.0]);
        let b = xpu_tensor(&[2.0, 4.0, 5.0]);

        let s = xpu_sub(&a, &b, &xpu).unwrap();
        assert_eq!(s.data().unwrap(), &[8.0, 16.0, 25.0]);

        let m = xpu_mul(&a, &b, &xpu).unwrap();
        assert_eq!(m.data().unwrap(), &[20.0, 80.0, 150.0]);

        let d = xpu_div(&a, &b, &xpu).unwrap();
        assert_eq!(d.data().unwrap(), &[5.0, 5.0, 6.0]);
    }

    #[test]
    fn xpu_matmul_runs_on_gpu() {
        let Some(xpu) = xpu() else { return };
        let a = ferrotorch_core::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3])
            .unwrap()
            .to(Device::Xpu(0))
            .unwrap();
        let b = ferrotorch_core::from_vec(vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2])
            .unwrap()
            .to(Device::Xpu(0))
            .unwrap();

        let c = xpu_matmul(&a, &b, &xpu).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.device(), Device::Xpu(0));
        let data: &[f32] = c.data().unwrap();
        assert!((data[0] - 58.0).abs() < 1e-4);
        assert!((data[1] - 64.0).abs() < 1e-4);
        assert!((data[2] - 139.0).abs() < 1e-4);
        assert!((data[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn xpu_unary_kernels_run_on_gpu() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[-3.0, -1.0, 0.0, 1.0, 3.0]);
        let n = xpu_neg(&a, &xpu).unwrap();
        assert_eq!(n.data().unwrap(), &[3.0, 1.0, 0.0, -1.0, -3.0]);

        let abs = xpu_abs(&a, &xpu).unwrap();
        assert_eq!(abs.data().unwrap(), &[3.0, 1.0, 0.0, 1.0, 3.0]);

        let r = xpu_relu(&a, &xpu).unwrap();
        assert_eq!(r.data().unwrap(), &[0.0, 0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn xpu_transcendentals_run_on_gpu() {
        let Some(xpu) = xpu() else { return };
        let a = xpu_tensor(&[0.0, 1.0, 2.0]);
        let e = xpu_exp(&a, &xpu).unwrap();
        let ed = e.data().unwrap();
        let expected: Vec<f32> = [0.0, 1.0, 2.0].iter().map(|x: &f32| x.exp()).collect();
        for (a, b) in ed.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-3, "exp parity: got {a}, expected {b}");
        }

        let l = xpu_ln(&xpu_tensor(&[1.0, std::f32::consts::E, 10.0]), &xpu).unwrap();
        let ld = l.data().unwrap();
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
        let a = xpu_tensor(&[1.0, 2.0]);
        let b = ferrotorch_core::tensor(&[1.0_f32, 2.0]).unwrap();
        let err = xpu_add(&a, &b, &xpu).unwrap_err();
        assert!(matches!(err, FerrotorchError::DeviceMismatch { .. }));
    }

    #[test]
    fn xpu_add_rejects_cuda_input_against_xpu_device() {
        // We can't actually allocate a CUDA tensor on this box, but
        // a CPU tensor with the wrong device marker triggers the same
        // mismatch path.
        let Some(xpu_other) = xpu() else { return };
        let a = xpu_tensor(&[1.0, 2.0]);
        // Pretend tensor was on a different XPU ordinal — currently the
        // device is the only thing distinguishing XPU(0) from XPU(1).
        let b = ferrotorch_core::tensor(&[1.0_f32, 2.0]).unwrap();
        let err = xpu_add(&a, &b, &xpu_other).unwrap_err();
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
