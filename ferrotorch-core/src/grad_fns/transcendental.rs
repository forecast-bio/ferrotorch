//! Backward functions for transcendental (exp, log, sin, cos) and clamp
//! operations.
//!
//! Each operation has a backward struct implementing `GradFn<T>` and a public
//! function that performs the forward pass and attaches the grad_fn to the
//! result tensor when gradient tracking is enabled.

use std::sync::Arc;

use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::ops::elementwise::unary_map;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether a single tensor requires grad (and grad is enabled).
#[inline]
fn needs_grad_unary<T: Float>(a: &Tensor<T>) -> bool {
    is_grad_enabled() && a.requires_grad()
}

// ===========================================================================
// exp
// ===========================================================================

/// Backward node for `c = exp(x)`.
///
/// VJP: `dx = grad * exp(x)`. We store the output (= exp(x)) to avoid
/// recomputing the exponential.
#[derive(Debug)]
struct ExpBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> GradFn<T> for ExpBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: dx = grad * output
                Some(no_grad(|| crate::grad_fns::arithmetic::mul(grad_output, &self.output))?)
            } else {
                // CPU path: direct data access for performance.
                let go_data = grad_output.data()?;
                let out_data = self.output.data()?;
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(out_data.iter())
                    .map(|(&g, &o)| g * o)
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ExpBackward"
    }
}

/// Differentiable elementwise exponential: `c[i] = exp(x[i])`.
pub fn exp<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = crate::ops::elementwise::fast_exp(input)?;

    if needs_grad_unary(input) {
        let grad_fn = Arc::new(ExpBackward {
            input: input.clone(),
            output: output.clone(),
        });
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(output)
    }
}

// ===========================================================================
// log
// ===========================================================================

/// Backward node for `c = ln(x)`.
///
/// VJP: `dx = grad / x`.
#[derive(Debug)]
struct LogBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for LogBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: dx = grad / x
                Some(no_grad(|| crate::grad_fns::arithmetic::div(grad_output, &self.input))?)
            } else {
                // CPU path
                let go_data = grad_output.data()?;
                let x_data = self.input.data()?;
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(x_data.iter())
                    .map(|(&g, &x)| g / x)
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LogBackward"
    }
}

/// Differentiable elementwise natural log: `c[i] = ln(x[i])`.
pub fn log<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.ln())?;

    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(LogBackward {
                input: input.clone(),
            }),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// sin
// ===========================================================================

/// Backward node for `c = sin(x)`.
///
/// VJP: `dx = grad * cos(x)`.
#[derive(Debug)]
struct SinBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for SinBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: dx = grad * cos(x)
                let da = no_grad(|| {
                    let cos_x = cos(&self.input)?;
                    crate::grad_fns::arithmetic::mul(grad_output, &cos_x)
                })?;
                Some(da)
            } else {
                // CPU path
                let go_data = grad_output.data()?;
                let x_data = self.input.data()?;
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(x_data.iter())
                    .map(|(&g, &x)| g * x.cos())
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SinBackward"
    }
}

/// Differentiable elementwise sine: `c[i] = sin(x[i])`.
pub fn sin<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.sin())?;

    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(SinBackward {
                input: input.clone(),
            }),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// cos
// ===========================================================================

/// Backward node for `c = cos(x)`.
///
/// VJP: `dx = grad * (-sin(x))`.
#[derive(Debug)]
struct CosBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for CosBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: dx = grad * (-sin(x))
                let da = no_grad(|| {
                    let sin_x = sin(&self.input)?;
                    let neg_sin = crate::grad_fns::arithmetic::neg(&sin_x)?;
                    crate::grad_fns::arithmetic::mul(grad_output, &neg_sin)
                })?;
                Some(da)
            } else {
                // CPU path
                let go_data = grad_output.data()?;
                let x_data = self.input.data()?;
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(x_data.iter())
                    .map(|(&g, &x)| g * (-x.sin()))
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CosBackward"
    }
}

/// Differentiable elementwise cosine: `c[i] = cos(x[i])`.
pub fn cos<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.cos())?;

    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(CosBackward {
                input: input.clone(),
            }),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// tanh (delegated)
// ===========================================================================

// tanh and sigmoid are implemented in `grad_fns::activation` since they are
// also activation functions. Re-exporting here for discoverability:
//
//   use crate::grad_fns::activation::{tanh, sigmoid};

// ===========================================================================
// clamp
// ===========================================================================

/// Backward node for `c = clamp(x, min, max)`.
///
/// VJP: `dx[i] = grad[i]` if `min <= x[i] <= max`, else `0`.
#[derive(Debug)]
struct ClampBackward<T: Float> {
    input: Tensor<T>,
    min: T,
    max: T,
}

impl<T: Float> GradFn<T> for ClampBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: build mask on CPU, move to GPU, multiply.
                let input_cpu = self.input.cpu()?;
                let x_data = input_cpu.data()?;
                let zero = <T as num_traits::Zero>::zero();
                let one = <T as num_traits::One>::one();
                let mask_data: Vec<T> = x_data
                    .iter()
                    .map(|&x| {
                        if x >= self.min && x <= self.max {
                            one
                        } else {
                            zero
                        }
                    })
                    .collect();
                let mask_cpu = Tensor::from_storage(
                    TensorStorage::cpu(mask_data),
                    self.input.shape().to_vec(),
                    false,
                )?;
                let mask_gpu = mask_cpu.to(grad_output.device())?;
                Some(no_grad(|| crate::grad_fns::arithmetic::mul(grad_output, &mask_gpu))?)
            } else {
                // CPU path
                let go_data = grad_output.data()?;
                let x_data = self.input.data()?;
                let zero = <T as num_traits::Zero>::zero();
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(x_data.iter())
                    .map(|(&g, &x)| {
                        if x >= self.min && x <= self.max {
                            g
                        } else {
                            zero
                        }
                    })
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.input.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ClampBackward"
    }
}

/// Differentiable elementwise clamp: `c[i] = x[i].clamp(min, max)`.
///
/// Gradient flows through only where `min <= x[i] <= max`; it is zero at
/// the boundaries where the value was clamped.
pub fn clamp<T: Float>(input: &Tensor<T>, min: T, max: T) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| {
        if x < min {
            min
        } else if x > max {
            max
        } else {
            x
        }
    })?;

    if needs_grad_unary(input) {
        let (storage, shape) = output.into_storage_and_shape()?;
        Tensor::from_operation(
            storage,
            shape,
            Arc::new(ClampBackward {
                input: input.clone(),
                min,
                max,
            }),
        )
    } else {
        Ok(output)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    /// Create a leaf 1-D tensor.
    fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], requires_grad)
            .unwrap()
    }

    /// Assert a scalar tensor is approximately equal to `expected`.
    fn assert_scalar_approx(t: &Tensor<f32>, expected: f32, tol: f32) {
        let val = t.item().unwrap();
        assert!(
            (val - expected).abs() < tol,
            "expected {expected}, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Forward tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_exp_forward() {
        let a = leaf_vec(&[0.0, 1.0, 2.0], false);
        let c = exp(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-5);
        assert!((d[1] - std::f32::consts::E).abs() < 1e-5);
        assert!((d[2] - std::f32::consts::E * std::f32::consts::E).abs() < 1e-4);
    }

    #[test]
    fn test_log_forward() {
        let a = leaf_vec(&[1.0, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E], false);
        let c = log(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-5);
        assert!((d[1] - 1.0).abs() < 1e-5);
        assert!((d[2] - 2.0).abs() < 1e-4);
    }

    #[test]
    fn test_sin_forward() {
        let a = leaf_vec(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI], false);
        let c = sin(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-6);
        assert!((d[1] - 1.0).abs() < 1e-6);
        assert!(d[2].abs() < 1e-6);
    }

    #[test]
    fn test_cos_forward() {
        let a = leaf_vec(&[0.0, std::f32::consts::FRAC_PI_2, std::f32::consts::PI], false);
        let c = cos(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert!(d[1].abs() < 1e-6);
        assert!((d[2] - (-1.0)).abs() < 1e-6);
    }

    #[test]
    fn test_clamp_forward() {
        let a = leaf_vec(&[-2.0, 0.5, 1.5, 3.0], false);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        assert_eq!(c.data().unwrap(), &[0.0, 0.5, 1.5, 2.0]);
    }

    // -----------------------------------------------------------------------
    // Backward tests (scalar tensors for simplicity)
    // -----------------------------------------------------------------------

    #[test]
    fn test_exp_backward() {
        // c = exp(a); dc/da = exp(a).
        // a = 1.0 => dc/da = e ~= 2.7183.
        let a = leaf_scalar(1.0, true);
        let c = exp(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), std::f32::consts::E, 1e-5);
    }

    #[test]
    fn test_log_backward() {
        // c = ln(a); dc/da = 1/a.
        // a = 2.0 => dc/da = 0.5.
        let a = leaf_scalar(2.0, true);
        let c = log(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.5, 1e-6);
    }

    #[test]
    fn test_sin_backward() {
        // c = sin(a); dc/da = cos(a).
        // a = 0.0 => dc/da = cos(0) = 1.0.
        let a = leaf_scalar(0.0, true);
        let c = sin(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_sin_backward_pi_over_3() {
        // a = pi/3 => dc/da = cos(pi/3) = 0.5.
        let a = leaf_scalar(std::f32::consts::FRAC_PI_3, true);
        let c = sin(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.5, 1e-5);
    }

    #[test]
    fn test_cos_backward() {
        // c = cos(a); dc/da = -sin(a).
        // a = 0.0 => dc/da = -sin(0) = 0.
        let a = leaf_scalar(0.0, true);
        let c = cos(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    #[test]
    fn test_cos_backward_pi_over_2() {
        // a = pi/2 => dc/da = -sin(pi/2) = -1.0.
        let a = leaf_scalar(std::f32::consts::FRAC_PI_2, true);
        let c = cos(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -1.0, 1e-5);
    }

    #[test]
    fn test_clamp_backward_interior() {
        // a = 1.5, clamp(0, 2) => interior, so dc/da = 1.
        let a = leaf_scalar(1.5, true);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_clamp_backward_clamped_low() {
        // a = -1.0, clamp(0, 2) => clamped to min, so dc/da = 0.
        let a = leaf_scalar(-1.0, true);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    #[test]
    fn test_clamp_backward_clamped_high() {
        // a = 5.0, clamp(0, 2) => clamped to max, so dc/da = 0.
        let a = leaf_scalar(5.0, true);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.0, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Chain rule tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_exp_log() {
        // c = log(exp(a)) = a. dc/da = 1.
        let a = leaf_scalar(3.0, true);
        let b = exp(&a).unwrap();
        let c = log(&b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-4);
    }

    #[test]
    fn test_chain_sin_cos() {
        // c = sin(a)^2 + cos(a)^2 = 1 for all a.
        // But that requires add/mul/pow which are separate ops.
        // Instead: c = cos(sin(a)).
        // dc/da = -sin(sin(a)) * cos(a).
        // a = 0.5 => dc/da = -sin(sin(0.5)) * cos(0.5)
        //          = -sin(0.4794) * 0.8776
        //          = -0.4611 * 0.8776 ~= -0.4047
        let a = leaf_scalar(0.5, true);
        let b = sin(&a).unwrap();
        let c = cos(&b).unwrap();
        c.backward().unwrap();

        let expected = -(0.5_f32.sin().sin()) * 0.5_f32.cos();
        assert_scalar_approx(&a.grad().unwrap().unwrap(), expected, 1e-4);
    }

    // -----------------------------------------------------------------------
    // No-grad test
    // -----------------------------------------------------------------------

    #[test]
    fn test_exp_no_grad_fn_when_not_tracking() {
        let a = leaf_scalar(1.0, false);
        let c = exp(&a).unwrap();
        assert!(c.grad_fn().is_none());
    }

    #[test]
    fn test_log_no_grad_fn_when_not_tracking() {
        let a = leaf_scalar(1.0, false);
        let c = log(&a).unwrap();
        assert!(c.grad_fn().is_none());
    }

    #[test]
    fn test_clamp_no_grad_fn_when_not_tracking() {
        let a = leaf_scalar(1.0, false);
        let c = clamp(&a, 0.0, 2.0).unwrap();
        assert!(c.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // Numerical gradient check (finite difference)
    // -----------------------------------------------------------------------

    /// Check gradient using central finite differences:
    ///   grad ~= (f(x+h) - f(x-h)) / (2*h)
    fn numerical_grad_check(
        f: impl Fn(f32) -> f32,
        x: f32,
        analytic_grad: f32,
        tol: f32,
    ) {
        let h = 1e-4_f32;
        let numerical = (f(x + h) - f(x - h)) / (2.0 * h);
        assert!(
            (analytic_grad - numerical).abs() < tol,
            "analytic={analytic_grad}, numerical={numerical}",
        );
    }

    #[test]
    fn test_exp_numerical_grad() {
        let x = 1.5_f32;
        let a = leaf_scalar(x, true);
        let c = exp(&a).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.exp(), x, g, 1e-3);
    }

    #[test]
    fn test_log_numerical_grad() {
        let x = 2.0_f32;
        let a = leaf_scalar(x, true);
        let c = log(&a).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.ln(), x, g, 1e-3);
    }

    #[test]
    fn test_sin_numerical_grad() {
        let x = 1.0_f32;
        let a = leaf_scalar(x, true);
        let c = sin(&a).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.sin(), x, g, 1e-3);
    }

    #[test]
    fn test_cos_numerical_grad() {
        let x = 1.0_f32;
        let a = leaf_scalar(x, true);
        let c = cos(&a).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.cos(), x, g, 1e-3);
    }

    #[test]
    fn test_clamp_numerical_grad_interior() {
        let x = 0.5_f32;
        let a = leaf_scalar(x, true);
        let c = clamp(&a, 0.0, 1.0).unwrap();
        c.backward().unwrap();
        let g = a.grad().unwrap().unwrap().item().unwrap();
        numerical_grad_check(|v| v.clamp(0.0, 1.0), x, g, 1e-3);
    }
}
