//! Portable GPU operations that mirror `ferrotorch-gpu` but dispatch via CubeCL.
//!
//! Each function accepts a [`CubeDevice`] indicating the target backend. The
//! current implementation delegates to the existing CPU autograd kernels in
//! `ferrotorch-core` — this is correct and portable, making the crate usable
//! today. As CubeCL kernel implementations land, each function will be
//! replaced with a real GPU dispatch while the API remains identical.
//!
//! The CPU-fallback strategy mirrors what `ferrotorch-gpu` does for non-f32
//! types: compute on the host, return a `Tensor<T>`. Once a CubeCL kernel is
//! wired up the data stays on-device and the fallback disappears.

use crate::runtime::CubeDevice;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

// ---------------------------------------------------------------------------
// Elementwise arithmetic
// ---------------------------------------------------------------------------

/// Portable GPU elementwise addition.
///
/// Dispatches to the appropriate CubeCL backend identified by `device`.
/// Currently falls back to the CPU autograd kernel.
pub fn portable_add<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    _device: &CubeDevice,
) -> FerrotorchResult<Tensor<T>> {
    // TODO: replace with CubeCL kernel dispatch
    ferrotorch_core::grad_fns::arithmetic::add(a, b)
}

/// Portable GPU elementwise subtraction.
pub fn portable_sub<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    _device: &CubeDevice,
) -> FerrotorchResult<Tensor<T>> {
    ferrotorch_core::grad_fns::arithmetic::sub(a, b)
}

/// Portable GPU elementwise multiplication.
pub fn portable_mul<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    _device: &CubeDevice,
) -> FerrotorchResult<Tensor<T>> {
    ferrotorch_core::grad_fns::arithmetic::mul(a, b)
}

// ---------------------------------------------------------------------------
// Linear algebra
// ---------------------------------------------------------------------------

/// Portable GPU matrix multiplication.
///
/// Both tensors must be 2-D with compatible inner dimensions.
pub fn portable_matmul<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    _device: &CubeDevice,
) -> FerrotorchResult<Tensor<T>> {
    ferrotorch_core::grad_fns::linalg::matmul_differentiable(a, b)
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// Portable GPU ReLU activation.
pub fn portable_relu<T: Float>(
    input: &Tensor<T>,
    _device: &CubeDevice,
) -> FerrotorchResult<Tensor<T>> {
    ferrotorch_core::grad_fns::activation::relu(input)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::CubeDevice;

    fn device() -> CubeDevice {
        CubeDevice::Wgpu(0)
    }

    #[test]
    fn portable_add_basic() {
        let a = ferrotorch_core::tensor(&[1.0_f32, 2.0, 3.0]).unwrap();
        let b = ferrotorch_core::tensor(&[10.0_f32, 20.0, 30.0]).unwrap();
        let c = portable_add(&a, &b, &device()).unwrap();
        let data: &[f32] = c.data().unwrap();
        assert!((data[0] - 11.0).abs() < 1e-6);
        assert!((data[1] - 22.0).abs() < 1e-6);
        assert!((data[2] - 33.0).abs() < 1e-6);
    }

    #[test]
    fn portable_sub_basic() {
        let a = ferrotorch_core::tensor(&[10.0_f32, 20.0, 30.0]).unwrap();
        let b = ferrotorch_core::tensor(&[1.0_f32, 2.0, 3.0]).unwrap();
        let c = portable_sub(&a, &b, &device()).unwrap();
        let data: &[f32] = c.data().unwrap();
        assert!((data[0] - 9.0).abs() < 1e-6);
        assert!((data[1] - 18.0).abs() < 1e-6);
        assert!((data[2] - 27.0).abs() < 1e-6);
    }

    #[test]
    fn portable_mul_basic() {
        let a = ferrotorch_core::tensor(&[2.0_f32, 3.0, 4.0]).unwrap();
        let b = ferrotorch_core::tensor(&[10.0_f32, 10.0, 10.0]).unwrap();
        let c = portable_mul(&a, &b, &device()).unwrap();
        let data: &[f32] = c.data().unwrap();
        assert!((data[0] - 20.0).abs() < 1e-6);
        assert!((data[1] - 30.0).abs() < 1e-6);
        assert!((data[2] - 40.0).abs() < 1e-6);
    }

    #[test]
    fn portable_relu_basic() {
        let a = ferrotorch_core::tensor(&[-3.0_f32, -1.0, 0.0, 1.0, 3.0]).unwrap();
        let c = portable_relu(&a, &device()).unwrap();
        let data: &[f32] = c.data().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
        assert!((data[4] - 3.0).abs() < 1e-6);
    }

    #[test]
    fn portable_matmul_basic() {
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // B = [[7, 8], [9, 10], [11, 12]]  (3x2)
        // C = [[58, 64], [139, 154]]  (2x2)
        let a = ferrotorch_core::from_vec(vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b =
            ferrotorch_core::from_vec(vec![7.0_f32, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]).unwrap();
        let c = portable_matmul(&a, &b, &device()).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let data: &[f32] = c.data().unwrap();
        assert!((data[0] - 58.0).abs() < 1e-4);
        assert!((data[1] - 64.0).abs() < 1e-4);
        assert!((data[2] - 139.0).abs() < 1e-4);
        assert!((data[3] - 154.0).abs() < 1e-4);
    }
}
