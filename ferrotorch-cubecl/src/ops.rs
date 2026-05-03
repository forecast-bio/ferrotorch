//! Portable GPU operations that dispatch through a real CubeCL
//! [`ComputeClient`] and run `#[cube]` kernels on the selected backend.
//!
//! Each function takes a [`CubeRuntime`] (which owns the client) and input
//! `f32` tensors, runs the kernel on-device, and returns a
//! `(cubecl::server::Handle, Vec<usize>)` pair: the **device-resident** result
//! buffer plus its shape. No host readback is performed here.
//!
//! Post-#673: inputs that are already device-resident (XPU tensors backed by
//! `StorageBuffer::Cubecl`) are dispatched through handle-direct kernel paths
//! with no H2D upload. CPU tensors are uploaded on the way in.
//!
//! Only `f32` is supported today — CubeCL's `Float` trait requires
//! `CubeElement`, which `f32` implements on every backend.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor};
// Needed so `.len()` and `.raw_handle()` resolve on `&CubeclStorageHandle`
// inside the dispatch macros (trait methods require the trait to be in scope).
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
use ferrotorch_core::storage::CubeStorageHandle as _;

use crate::runtime::CubeRuntime;

#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
use crate::kernels;
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
use crate::runtime::CubeClient;
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
use crate::storage::cubecl_handle_of;

// ---------------------------------------------------------------------------
// Shape helpers (always available so error paths can use them)
// ---------------------------------------------------------------------------

fn check_same_shape(a: &Tensor<f32>, b: &Tensor<f32>) -> FerrotorchResult<()> {
    if a.shape() != b.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "cubecl elementwise: lhs {:?} vs rhs {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }
    Ok(())
}

/// Get a contiguous host `Vec<f32>` from a tensor.
///
/// Only used for the CPU-input fallback path (when a tensor is not yet
/// device-resident). XPU tensors go through `cubecl_handle_of` instead.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
fn contiguous_data(t: &Tensor<f32>) -> FerrotorchResult<Vec<f32>> {
    match t.data() {
        Ok(slice) => Ok(slice.to_vec()),
        Err(_) => t.data_vec(),
    }
}

// ---------------------------------------------------------------------------
// Dispatcher macros
//
// Each macro returns `(cubecl::server::Handle, usize)` — the device-resident
// output buffer plus its element count. No readback. ADR #663 / issue #673.
// ---------------------------------------------------------------------------

/// Dispatch a binary op, using handle-direct path when both inputs are already
/// device-resident (no H2D upload), otherwise uploading slices.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
macro_rules! dispatch_binary {
    ($rt:expr, $launcher:path, $launcher_handle:path, $a:expr, $b:expr) => {{
        match (cubecl_handle_of($a), cubecl_handle_of($b)) {
            (Some(ha), Some(hb)) => {
                let n = ha.len();
                let ah = ha.raw_handle().clone();
                let bh = hb.raw_handle().clone();
                match $rt.client() {
                    #[cfg(feature = "wgpu")]
                    CubeClient::Wgpu(c) => $launcher_handle(c, ah, bh, n),
                    #[cfg(feature = "cuda")]
                    CubeClient::Cuda(c) => $launcher_handle(c, ah, bh, n),
                    #[cfg(feature = "rocm")]
                    CubeClient::Rocm(c) => $launcher_handle(c, ah, bh, n),
                }
            }
            _ => {
                let a_data = contiguous_data($a)?;
                let b_data = contiguous_data($b)?;
                match $rt.client() {
                    #[cfg(feature = "wgpu")]
                    CubeClient::Wgpu(c) => $launcher(c, &a_data, &b_data),
                    #[cfg(feature = "cuda")]
                    CubeClient::Cuda(c) => $launcher(c, &a_data, &b_data),
                    #[cfg(feature = "rocm")]
                    CubeClient::Rocm(c) => $launcher(c, &a_data, &b_data),
                }
            }
        }
    }};
}

/// Dispatch a unary op, using handle-direct path when the input is already
/// device-resident (no H2D upload), otherwise uploading the slice.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
macro_rules! dispatch_unary {
    ($rt:expr, $launcher:path, $launcher_handle:path, $x:expr) => {{
        match cubecl_handle_of($x) {
            Some(hx) => {
                let n = hx.len();
                let xh = hx.raw_handle().clone();
                match $rt.client() {
                    #[cfg(feature = "wgpu")]
                    CubeClient::Wgpu(c) => $launcher_handle(c, xh, n),
                    #[cfg(feature = "cuda")]
                    CubeClient::Cuda(c) => $launcher_handle(c, xh, n),
                    #[cfg(feature = "rocm")]
                    CubeClient::Rocm(c) => $launcher_handle(c, xh, n),
                }
            }
            None => {
                let x_data = contiguous_data($x)?;
                match $rt.client() {
                    #[cfg(feature = "wgpu")]
                    CubeClient::Wgpu(c) => $launcher(c, &x_data),
                    #[cfg(feature = "cuda")]
                    CubeClient::Cuda(c) => $launcher(c, &x_data),
                    #[cfg(feature = "rocm")]
                    CubeClient::Rocm(c) => $launcher(c, &x_data),
                }
            }
        }
    }};
}

#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
macro_rules! dispatch_matmul {
    ($rt:expr, $a:expr, $b:expr, $m:expr, $k:expr, $n:expr) => {{
        let a_data = contiguous_data($a)?;
        let b_data = contiguous_data($b)?;
        match $rt.client() {
            #[cfg(feature = "wgpu")]
            CubeClient::Wgpu(c) => kernels::run_matmul(c, &a_data, &b_data, $m, $k, $n),
            #[cfg(feature = "cuda")]
            CubeClient::Cuda(c) => kernels::run_matmul(c, &a_data, &b_data, $m, $k, $n),
            #[cfg(feature = "rocm")]
            CubeClient::Rocm(c) => kernels::run_matmul(c, &a_data, &b_data, $m, $k, $n),
        }
    }};
}

// ---------------------------------------------------------------------------
// Public ops — one implementation per backend-enabled build, one no-op
// error stub for the no-backend build.
// ---------------------------------------------------------------------------

/// Elementwise `a + b` on the GPU.
///
/// Returns the device-resident result handle and shape. Call
/// Elementwise `a + b` on the GPU. Inputs may be device-resident (no H2D
/// upload) or CPU tensors (uploaded on the way in). Issue #673.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn portable_add(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    check_same_shape(a, b)?;
    let (handle, _n) = dispatch_binary!(rt, kernels::run_add, kernels::run_add_handle, a, b);
    Ok((handle, a.shape().to_vec()))
}

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
pub fn portable_add(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    _rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    check_same_shape(a, b)?;
    Err(FerrotorchError::DeviceUnavailable)
}

/// Elementwise `a - b` on the GPU.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn portable_sub(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    check_same_shape(a, b)?;
    let (handle, _n) = dispatch_binary!(rt, kernels::run_sub, kernels::run_sub_handle, a, b);
    Ok((handle, a.shape().to_vec()))
}

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
pub fn portable_sub(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    _rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    check_same_shape(a, b)?;
    Err(FerrotorchError::DeviceUnavailable)
}

/// Elementwise `a * b` on the GPU.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn portable_mul(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    check_same_shape(a, b)?;
    let (handle, _n) = dispatch_binary!(rt, kernels::run_mul, kernels::run_mul_handle, a, b);
    Ok((handle, a.shape().to_vec()))
}

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
pub fn portable_mul(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    _rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    check_same_shape(a, b)?;
    Err(FerrotorchError::DeviceUnavailable)
}

/// Elementwise `relu(x)` on the GPU.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn portable_relu(
    x: &Tensor<f32>,
    rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    let (handle, _n) = dispatch_unary!(rt, kernels::run_relu, kernels::run_relu_handle, x);
    Ok((handle, x.shape().to_vec()))
}

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
pub fn portable_relu(
    _x: &Tensor<f32>,
    _rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    Err(FerrotorchError::DeviceUnavailable)
}

/// 2-D matrix multiplication on the GPU.
///
/// `a` must have shape `[M, K]` and `b` must have shape `[K, N]`.
/// Returns device-resident handle + shape `[M, N]`.
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub fn portable_matmul(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    let (m, k, n) = check_matmul_shapes(a, b)?;
    let (handle, _out_len) = dispatch_matmul!(rt, a, b, m, k, n);
    Ok((handle, vec![m, n]))
}

#[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
pub fn portable_matmul(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
    _rt: &CubeRuntime,
) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
    check_matmul_shapes(a, b)?;
    Err(FerrotorchError::DeviceUnavailable)
}

fn check_matmul_shapes(
    a: &Tensor<f32>,
    b: &Tensor<f32>,
) -> FerrotorchResult<(usize, usize, usize)> {
    let a_shape = a.shape();
    let b_shape = b.shape();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "cubecl matmul requires 2-D tensors, got lhs {:?} rhs {:?}",
                a_shape, b_shape
            ),
        });
    }
    if a_shape[1] != b_shape[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "cubecl matmul inner dims mismatch: lhs {:?} vs rhs {:?}",
                a_shape, b_shape
            ),
        });
    }
    Ok((a_shape[0], a_shape[1], b_shape[1]))
}

// ---------------------------------------------------------------------------
// Additional elementwise ops — macro-generated for brevity
// ---------------------------------------------------------------------------

macro_rules! define_portable_unary {
    ($name:ident, $runner:ident, $runner_handle:ident) => {
        #[doc = concat!("Elementwise `", stringify!($name), "(x)` on the GPU. Returns device-resident handle + shape. Issue #673: no H2D upload when input is already device-resident.")]
        #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
        pub fn $name(
            x: &Tensor<f32>,
            rt: &CubeRuntime,
        ) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
            let (handle, _n) = dispatch_unary!(rt, kernels::$runner, kernels::$runner_handle, x);
            Ok((handle, x.shape().to_vec()))
        }

        #[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
        pub fn $name(
            _x: &Tensor<f32>,
            _rt: &CubeRuntime,
        ) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
            Err(FerrotorchError::DeviceUnavailable)
        }
    };
}

macro_rules! define_portable_binary {
    ($name:ident, $runner:ident, $runner_handle:ident) => {
        #[doc = concat!("Elementwise `a ", stringify!($name), " b` on the GPU. Returns device-resident handle + shape. Issue #673: no H2D upload when inputs are already device-resident.")]
        #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
        pub fn $name(
            a: &Tensor<f32>,
            b: &Tensor<f32>,
            rt: &CubeRuntime,
        ) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
            check_same_shape(a, b)?;
            let (handle, _n) = dispatch_binary!(rt, kernels::$runner, kernels::$runner_handle, a, b);
            Ok((handle, a.shape().to_vec()))
        }

        #[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
        pub fn $name(
            a: &Tensor<f32>,
            b: &Tensor<f32>,
            _rt: &CubeRuntime,
        ) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
            check_same_shape(a, b)?;
            Err(FerrotorchError::DeviceUnavailable)
        }
    };
}

define_portable_binary!(portable_div, run_div, run_div_handle);

define_portable_unary!(portable_neg, run_neg, run_neg_handle);
define_portable_unary!(portable_abs, run_abs, run_abs_handle);
define_portable_unary!(portable_exp, run_exp, run_exp_handle);
define_portable_unary!(portable_ln, run_ln, run_ln_handle);
define_portable_unary!(portable_sqrt, run_sqrt, run_sqrt_handle);
define_portable_unary!(portable_sin, run_sin, run_sin_handle);
define_portable_unary!(portable_cos, run_cos, run_cos_handle);
define_portable_unary!(portable_tanh, run_tanh, run_tanh_handle);
define_portable_unary!(portable_sigmoid, run_sigmoid, run_sigmoid_handle);

// ---------------------------------------------------------------------------
// Orthogonal polynomial families — three-term recurrences, scalar `n` arg. (#577)
// Polynomials always go through host upload (no handle-direct variant yet).
// ---------------------------------------------------------------------------

#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
macro_rules! dispatch_unary_with_n {
    ($rt:expr, $launcher:path, $x:expr, $n:expr) => {{
        let x_data = contiguous_data($x)?;
        match $rt.client() {
            #[cfg(feature = "wgpu")]
            CubeClient::Wgpu(c) => $launcher(c, &x_data, $n),
            #[cfg(feature = "cuda")]
            CubeClient::Cuda(c) => $launcher(c, &x_data, $n),
            #[cfg(feature = "rocm")]
            CubeClient::Rocm(c) => $launcher(c, &x_data, $n),
        }
    }};
}

macro_rules! define_portable_polynomial {
    ($name:ident, $runner:ident, $math:literal) => {
        #[doc = concat!("Evaluate ", $math, " elementwise on the GPU at degree `n`. Returns device-resident handle + shape.")]
        #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
        pub fn $name(
            x: &Tensor<f32>,
            n: usize,
            rt: &CubeRuntime,
        ) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
            let n_u32 = u32::try_from(n).map_err(|_| FerrotorchError::InvalidArgument {
                message: format!("polynomial degree {n} exceeds u32 range"),
            })?;
            let (handle, _n) = dispatch_unary_with_n!(rt, kernels::$runner, x, n_u32);
            Ok((handle, x.shape().to_vec()))
        }

        #[cfg(not(any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
        pub fn $name(
            _x: &Tensor<f32>,
            _n: usize,
            _rt: &CubeRuntime,
        ) -> FerrotorchResult<(cubecl::server::Handle, Vec<usize>)> {
            Err(FerrotorchError::DeviceUnavailable)
        }
    };
}

define_portable_polynomial!(
    portable_chebyshev_polynomial_t,
    run_chebyshev_t,
    "Chebyshev T_n(x)"
);
define_portable_polynomial!(
    portable_chebyshev_polynomial_u,
    run_chebyshev_u,
    "Chebyshev U_n(x)"
);
define_portable_polynomial!(
    portable_chebyshev_polynomial_v,
    run_chebyshev_v,
    "Chebyshev V_n(x)"
);
define_portable_polynomial!(
    portable_chebyshev_polynomial_w,
    run_chebyshev_w,
    "Chebyshev W_n(x)"
);
define_portable_polynomial!(
    portable_hermite_polynomial_h,
    run_hermite_h,
    "Hermite (physicist) H_n(x)"
);
define_portable_polynomial!(
    portable_hermite_polynomial_he,
    run_hermite_he,
    "Hermite (probabilist) He_n(x)"
);
define_portable_polynomial!(
    portable_laguerre_polynomial_l,
    run_laguerre_l,
    "Laguerre L_n(x)"
);
define_portable_polynomial!(
    portable_legendre_polynomial_p,
    run_legendre_p,
    "Legendre P_n(x)"
);

// ---------------------------------------------------------------------------
// Tests — only meaningful with a real backend feature
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "wgpu"))]
mod tests {
    use super::*;
    use crate::runtime::CubeDevice;

    /// Attempt to build a wgpu runtime, returning `None` when no adapter is
    /// available. WSL2 does not ship Vulkan ICDs by default, so cubecl-wgpu's
    /// worker thread panics during adapter selection; that surfaces on the
    /// main thread as `RecvError` from the cubecl channel. We catch it so
    /// the wgpu integration tests skip cleanly in headless / no-GPU envs
    /// rather than failing the whole test run. Tests that get `None` use
    /// `let Some(rt) = runtime() else { return; };` to skip gracefully —
    /// install Vulkan drivers (or a wgpu-compatible GPU) to actually exercise
    /// the GPU code path.
    fn runtime() -> Option<CubeRuntime> {
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CubeRuntime::new(CubeDevice::Wgpu(0)).ok()
        }));
        match result {
            Ok(Some(rt)) => Some(rt),
            _ => {
                eprintln!(
                    "[ferrotorch-cubecl] wgpu adapter unavailable; skipping wgpu integration test"
                );
                None
            }
        }
    }

    // Helper: read back f32s from a (Handle, Vec<usize>) result.
    fn readback(rt: &CubeRuntime, handle: cubecl::server::Handle, shape: &[usize]) -> Vec<f32> {
        let n: usize = shape.iter().product();
        rt.read_f32s(handle, n).expect("read_f32s failed")
    }

    #[test]
    fn portable_add_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[1.0_f32, 2.0, 3.0, 4.0]).unwrap();
        let b = ferrotorch_core::tensor(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
        let (handle, shape) = portable_add(&a, &b, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_eq!(data, &[11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn portable_sub_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[10.0_f32, 20.0, 30.0]).unwrap();
        let b = ferrotorch_core::tensor(&[1.0_f32, 2.0, 3.0]).unwrap();
        let (handle, shape) = portable_sub(&a, &b, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_eq!(data, &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn portable_mul_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[2.0_f32, 3.0, 4.0]).unwrap();
        let b = ferrotorch_core::tensor(&[10.0_f32, 10.0, 10.0]).unwrap();
        let (handle, shape) = portable_mul(&a, &b, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_eq!(data, &[20.0, 30.0, 40.0]);
    }

    #[test]
    fn portable_relu_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[-3.0_f32, -1.0, 0.0, 1.0, 3.0]).unwrap();
        let (handle, shape) = portable_relu(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_eq!(data, &[0.0, 0.0, 0.0, 1.0, 3.0]);
    }

    #[test]
    fn portable_matmul_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // B = [[7, 8], [9, 10], [11, 12]]  (3x2)
        // C = [[58, 64], [139, 154]]  (2x2)
        let a = ferrotorch_core::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b =
            ferrotorch_core::from_vec(vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        let (handle, shape) = portable_matmul(&a, &b, &rt).unwrap();
        assert_eq!(shape, &[2, 2]);
        let data = readback(&rt, handle, &shape);
        assert!((data[0] - 58.0).abs() < 1e-4);
        assert!((data[1] - 64.0).abs() < 1e-4);
        assert!((data[2] - 139.0).abs() < 1e-4);
        assert!((data[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn portable_matmul_rejects_rank_mismatch() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[1.0_f32, 2.0, 3.0]).unwrap();
        let b = ferrotorch_core::from_vec(vec![1.0_f32, 2.0, 3.0], &[3, 1]).unwrap();
        let err = portable_matmul(&a, &b, &rt).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn portable_matmul_rejects_inner_dim_mismatch() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::from_vec(vec![1.0_f32; 6], &[2, 3]).unwrap();
        let b = ferrotorch_core::from_vec(vec![1.0_f32; 8], &[4, 2]).unwrap();
        let err = portable_matmul(&a, &b, &rt).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn portable_add_rejects_shape_mismatch() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[1.0_f32, 2.0, 3.0]).unwrap();
        let b = ferrotorch_core::tensor(&[1.0_f32, 2.0]).unwrap();
        let err = portable_add(&a, &b, &rt).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn portable_add_large_shape() {
        // Exercise multi-cube launch (more than 256 elements so we go past
        // one cube).
        let Some(rt) = runtime() else { return };
        let n: usize = 1024;
        let a: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..n).map(|i| (2 * i) as f32).collect();
        let ta = ferrotorch_core::from_vec(a.clone(), &[n]).unwrap();
        let tb = ferrotorch_core::from_vec(b.clone(), &[n]).unwrap();
        let (handle, shape) = portable_add(&ta, &tb, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        for i in 0..n {
            assert_eq!(data[i], a[i] + b[i]);
        }
    }

    #[test]
    fn portable_matmul_square_8x8() {
        let Some(rt) = runtime() else { return };
        let n = 8usize;
        // identity matrix
        let mut a = vec![0.0_f32; n * n];
        for i in 0..n {
            a[i * n + i] = 1.0;
        }
        // arbitrary matrix
        let b: Vec<f32> = (0..n * n).map(|i| i as f32).collect();

        let ta = ferrotorch_core::from_vec(a, &[n, n]).unwrap();
        let tb = ferrotorch_core::from_vec(b.clone(), &[n, n]).unwrap();
        let (handle, shape) = portable_matmul(&ta, &tb, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_eq!(data, b.as_slice(), "I * B should equal B");
    }

    // Helper: approximate-equal for float comparisons in transcendental tests.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len());
        for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "at index {i}: expected {e}, got {a} (tol={tol})"
            );
        }
    }

    #[test]
    fn portable_div_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[10.0_f32, 20.0, 30.0, 40.0]).unwrap();
        let b = ferrotorch_core::tensor(&[2.0_f32, 4.0, 5.0, 8.0]).unwrap();
        let (handle, shape) = portable_div(&a, &b, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[5.0, 5.0, 6.0, 5.0], 1e-4);
    }

    #[test]
    fn portable_neg_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[-3.0_f32, -1.0, 0.0, 1.0, 3.0]).unwrap();
        let (handle, shape) = portable_neg(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[3.0, 1.0, 0.0, -1.0, -3.0], 1e-6);
    }

    #[test]
    fn portable_abs_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[-3.0_f32, -1.0, 0.0, 1.0, 3.0]).unwrap();
        let (handle, shape) = portable_abs(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[3.0, 1.0, 0.0, 1.0, 3.0], 1e-6);
    }

    #[test]
    fn portable_exp_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[0.0_f32, 1.0, 2.0, -1.0]).unwrap();
        let (handle, shape) = portable_exp(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        let expected: Vec<f32> = [0.0, 1.0, 2.0, -1.0]
            .iter()
            .map(|x: &f32| x.exp())
            .collect();
        assert_close(&data, &expected, 1e-3);
    }

    #[test]
    fn portable_ln_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[1.0_f32, std::f32::consts::E, 10.0, 100.0]).unwrap();
        let (handle, shape) = portable_ln(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        let expected: Vec<f32> = [1.0_f32, std::f32::consts::E, 10.0, 100.0]
            .iter()
            .map(|x| x.ln())
            .collect();
        assert_close(&data, &expected, 1e-3);
    }

    #[test]
    fn portable_sqrt_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[0.0_f32, 1.0, 4.0, 9.0, 16.0]).unwrap();
        let (handle, shape) = portable_sqrt(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[0.0, 1.0, 2.0, 3.0, 4.0], 1e-4);
    }

    #[test]
    fn portable_sin_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let pi = std::f32::consts::PI;
        let a = ferrotorch_core::tensor(&[0.0_f32, pi / 2.0, pi, 3.0 * pi / 2.0]).unwrap();
        let (handle, shape) = portable_sin(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[0.0, 1.0, 0.0, -1.0], 1e-3);
    }

    #[test]
    fn portable_cos_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let pi = std::f32::consts::PI;
        let a = ferrotorch_core::tensor(&[0.0_f32, pi / 2.0, pi, 3.0 * pi / 2.0]).unwrap();
        let (handle, shape) = portable_cos(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[1.0, 0.0, -1.0, 0.0], 1e-3);
    }

    #[test]
    fn portable_tanh_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[-2.0_f32, -0.5, 0.0, 0.5, 2.0]).unwrap();
        let (handle, shape) = portable_tanh(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        let expected: Vec<f32> = [-2.0_f32, -0.5, 0.0, 0.5, 2.0]
            .iter()
            .map(|x| x.tanh())
            .collect();
        assert_close(&data, &expected, 1e-4);
    }

    #[test]
    fn portable_sigmoid_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[-4.0_f32, -1.0, 0.0, 1.0, 4.0]).unwrap();
        let (handle, shape) = portable_sigmoid(&a, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        let expected: Vec<f32> = [-4.0_f32, -1.0, 0.0, 1.0, 4.0]
            .iter()
            .map(|x| 1.0 / (1.0 + (-x).exp()))
            .collect();
        assert_close(&data, &expected, 1e-4);
    }

    #[test]
    fn portable_sigmoid_large_shape() {
        // Sigmoid across multiple cubes + verify numerical stability.
        let Some(rt) = runtime() else { return };
        let n: usize = 1024;
        let input: Vec<f32> = (0..n).map(|i| (i as f32 - 512.0) * 0.01).collect();
        let t = ferrotorch_core::from_vec(input.clone(), &[n]).unwrap();
        let (handle, shape) = portable_sigmoid(&t, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        for (i, &x) in input.iter().enumerate() {
            let expected = 1.0 / (1.0 + (-x).exp());
            assert!(
                (data[i] - expected).abs() < 1e-4,
                "sigmoid mismatch at {i}: got {}, expected {}",
                data[i],
                expected
            );
        }
    }

    #[test]
    fn portable_exp_then_ln_is_identity() {
        // End-to-end check: ln(exp(x)) ≈ x.
        // Read back after exp to produce a CPU tensor for the ln input.
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[-2.0_f32, -0.5, 0.1, 1.0, 3.0]).unwrap();
        let (exp_handle, exp_shape) = portable_exp(&a, &rt).unwrap();
        let exp_data = readback(&rt, exp_handle, &exp_shape);
        let e = ferrotorch_core::from_vec(exp_data, &exp_shape).unwrap();
        let (ln_handle, ln_shape) = portable_ln(&e, &rt).unwrap();
        let data = readback(&rt, ln_handle, &ln_shape);
        assert_close(&data, &[-2.0, -0.5, 0.1, 1.0, 3.0], 1e-3);
    }

    #[test]
    fn portable_div_rejects_shape_mismatch() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[1.0_f32, 2.0, 3.0]).unwrap();
        let b = ferrotorch_core::tensor(&[1.0_f32, 2.0]).unwrap();
        let err = portable_div(&a, &b, &rt).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    // -----------------------------------------------------------------------
    // Orthogonal polynomial families (#577)
    // -----------------------------------------------------------------------

    #[test]
    fn portable_chebyshev_t_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        // T_3(x) = 4x^3 - 3x. At x = [0, 0.5, 1, -1] → [0, -1, 1, -1].
        let a = ferrotorch_core::tensor(&[0.0_f32, 0.5, 1.0, -1.0]).unwrap();
        let (handle, shape) = portable_chebyshev_polynomial_t(&a, 3, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[0.0, -1.0, 1.0, -1.0], 1e-4);
    }

    #[test]
    fn portable_chebyshev_t_n0_returns_ones() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[0.3_f32, -0.7, 0.0]).unwrap();
        let (handle, shape) = portable_chebyshev_polynomial_t(&a, 0, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[1.0, 1.0, 1.0], 1e-6);
    }

    #[test]
    fn portable_chebyshev_t_n1_returns_x() {
        let Some(rt) = runtime() else { return };
        let a = ferrotorch_core::tensor(&[0.3_f32, -0.7, 0.0]).unwrap();
        let (handle, shape) = portable_chebyshev_polynomial_t(&a, 1, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[0.3, -0.7, 0.0], 1e-6);
    }

    #[test]
    fn portable_chebyshev_u_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        // U_2(x) = 4x^2 - 1. At x = [0, 0.5, 1] → [-1, 0, 3].
        let a = ferrotorch_core::tensor(&[0.0_f32, 0.5, 1.0]).unwrap();
        let (handle, shape) = portable_chebyshev_polynomial_u(&a, 2, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[-1.0, 0.0, 3.0], 1e-4);
    }

    #[test]
    fn portable_chebyshev_v_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        // V_1(x) = 2x - 1. At x = [0, 0.5, 1] → [-1, 0, 1].
        let a = ferrotorch_core::tensor(&[0.0_f32, 0.5, 1.0]).unwrap();
        let (handle, shape) = portable_chebyshev_polynomial_v(&a, 1, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[-1.0, 0.0, 1.0], 1e-6);
    }

    #[test]
    fn portable_chebyshev_w_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        // W_1(x) = 2x + 1. At x = [0, -0.5, 1] → [1, 0, 3].
        let a = ferrotorch_core::tensor(&[0.0_f32, -0.5, 1.0]).unwrap();
        let (handle, shape) = portable_chebyshev_polynomial_w(&a, 1, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[1.0, 0.0, 3.0], 1e-6);
    }

    #[test]
    fn portable_hermite_h_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        // H_3(x) = 8x^3 - 12x. At x = [0, 1, 2] → [0, -4, 40].
        let a = ferrotorch_core::tensor(&[0.0_f32, 1.0, 2.0]).unwrap();
        let (handle, shape) = portable_hermite_polynomial_h(&a, 3, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[0.0, -4.0, 40.0], 1e-3);
    }

    #[test]
    fn portable_hermite_he_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        // He_3(x) = x^3 - 3x. At x = [0, 1, 2] → [0, -2, 2].
        let a = ferrotorch_core::tensor(&[0.0_f32, 1.0, 2.0]).unwrap();
        let (handle, shape) = portable_hermite_polynomial_he(&a, 3, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[0.0, -2.0, 2.0], 1e-4);
    }

    #[test]
    fn portable_laguerre_l_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        // L_2(x) = (1/2)(x^2 - 4x + 2). At x = [0, 1, 2] → [1, -1/2, -1].
        let a = ferrotorch_core::tensor(&[0.0_f32, 1.0, 2.0]).unwrap();
        let (handle, shape) = portable_laguerre_polynomial_l(&a, 2, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[1.0, -0.5, -1.0], 1e-4);
    }

    #[test]
    fn portable_legendre_p_runs_on_gpu() {
        let Some(rt) = runtime() else { return };
        // P_2(x) = (1/2)(3x^2 - 1). At x = [0, 0.5, 1] → [-1/2, -1/8, 1].
        let a = ferrotorch_core::tensor(&[0.0_f32, 0.5, 1.0]).unwrap();
        let (handle, shape) = portable_legendre_polynomial_p(&a, 2, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        assert_close(&data, &[-0.5, -0.125, 1.0], 1e-4);
    }

    #[test]
    fn portable_polynomial_handles_large_input() {
        // Exercise multi-cube launch (1024 elems > 256 cube dim).
        let Some(rt) = runtime() else { return };
        let n_elems: usize = 1024;
        let a: Vec<f32> = (0..n_elems).map(|i| (i as f32) / 2048.0).collect();
        let ta = ferrotorch_core::from_vec(a.clone(), &[n_elems]).unwrap();
        // T_5 must hold |T_n(x)| <= 1 for |x| <= 1.
        let (handle, shape) = portable_chebyshev_polynomial_t(&ta, 5, &rt).unwrap();
        let data = readback(&rt, handle, &shape);
        for (i, v) in data.iter().enumerate() {
            assert!(v.abs() <= 1.001, "T_5({}) = {}", a[i], v);
        }
    }
}

// ---------------------------------------------------------------------------
// Tests that run on every build (no backend feature needed)
// ---------------------------------------------------------------------------

#[cfg(all(test, not(any(feature = "wgpu", feature = "cuda", feature = "rocm"))))]
mod no_backend_tests {
    use super::*;
    use crate::runtime::CubeDevice;

    #[test]
    fn runtime_construction_errors_without_backend() {
        let err = CubeRuntime::new(CubeDevice::Wgpu(0)).unwrap_err();
        assert!(matches!(err, FerrotorchError::DeviceUnavailable));
    }
}
