//! Shared helpers used by the `step_foreach` paths of the optimizers.
//!
//! These helpers wrap common tensor-op patterns that show up repeatedly when
//! expressing an optimizer update entirely through GPU-aware tensor ops,
//! notably elementwise `max(a, b)` (which ferrotorch_core does not expose as
//! a grad_fn) and scalar-promotion to the parameter's device.

use ferrotorch_core::creation::scalar;
use ferrotorch_core::grad_fns::arithmetic::{abs, add, mul, sub};
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{Device, FerrotorchResult, Float, Tensor};

/// Broadcast-free elementwise `max(a, b)` computed as
/// `0.5 * (a + b + |a - b|)`. Works on any backend that supports add/sub/abs.
pub fn elemwise_max<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    device: Device,
) -> FerrotorchResult<Tensor<T>> {
    let diff = sub(a, b)?;
    let abs_diff = abs(&diff)?;
    let sum_ab = add(a, b)?;
    let sum_plus_abs = add(&sum_ab, &abs_diff)?;
    let half = scalar(cast::<f64, T>(0.5)?)?.to(device)?;
    mul(&sum_plus_abs, &half)
}

/// Create a scalar tensor on the given device. This is the common boilerplate
/// every `step_foreach` function repeats for every hyperparameter.
#[inline]
pub fn scalar_on<T: Float>(value: T, device: Device) -> FerrotorchResult<Tensor<T>> {
    scalar(value)?.to(device)
}

/// Convenience: convert an `f64` hyperparameter into a scalar on the device.
#[inline]
pub fn f64_scalar_on<T: Float>(value: f64, device: Device) -> FerrotorchResult<Tensor<T>> {
    scalar_on(cast::<f64, T>(value)?, device)
}
