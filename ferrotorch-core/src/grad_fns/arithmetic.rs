//! Backward functions for elementwise arithmetic operations.
//!
//! Each operation has a backward struct implementing `GradFn<T>` and a public
//! function that performs the forward pass and attaches the grad_fn to the
//! result tensor when gradient tracking is enabled.

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::ops::elementwise::{binary_map, scalar_map, unary_map, fast_add, fast_mul};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether at least one of two tensors requires grad (and grad is enabled).
#[inline]
fn needs_grad<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    is_grad_enabled() && (a.requires_grad() || b.requires_grad())
}

/// Whether a single tensor requires grad (and grad is enabled).
#[inline]
fn needs_grad_unary<T: Float>(a: &Tensor<T>) -> bool {
    is_grad_enabled() && a.requires_grad()
}

// ===========================================================================
// add
// ===========================================================================

/// Backward node for `c = a + b`.
///
/// VJP: da = grad, db = grad.
#[derive(Debug)]
struct AddBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for AddBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            Some(grad_output.clone())
        } else {
            None
        };
        let db = if self.b.requires_grad() {
            Some(grad_output.clone())
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Elementwise addition: `c = a + b`.
pub fn add<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = fast_add(a, b)?;

    if needs_grad(a, b) {
        let storage = TensorStorage::cpu(result.data()?.to_vec());
        Tensor::from_operation(
            storage,
            result.shape().to_vec(),
            Arc::new(AddBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
    } else {
        Ok(result)
    }
}

// ===========================================================================
// sub
// ===========================================================================

/// Backward node for `c = a - b`.
///
/// VJP: da = grad, db = -grad.
#[derive(Debug)]
struct SubBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for SubBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            Some(grad_output.clone())
        } else {
            None
        };
        let db = if self.b.requires_grad() {
            let go_data = grad_output.data()?;
            let neg: Vec<T> = go_data.iter().map(|&g| -g).collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(neg),
                grad_output.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

/// Elementwise subtraction: `c = a - b`.
pub fn sub<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = binary_map(a, b, |x, y| x - y)?;

    if needs_grad(a, b) {
        let storage = TensorStorage::cpu(result.data()?.to_vec());
        Tensor::from_operation(
            storage,
            result.shape().to_vec(),
            Arc::new(SubBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
    } else {
        Ok(result)
    }
}

// ===========================================================================
// mul
// ===========================================================================

/// Backward node for `c = a * b`.
///
/// VJP: da = grad * b, db = grad * a.
#[derive(Debug)]
struct MulBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for MulBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go_data = grad_output.data()?;

        let da = if self.a.requires_grad() {
            let b_data = self.b.data()?;
            let grad_a: Vec<T> = go_data.iter().zip(b_data.iter()).map(|(&g, &b)| g * b).collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.a.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        let db = if self.b.requires_grad() {
            let a_data = self.a.data()?;
            let grad_b: Vec<T> = go_data.iter().zip(a_data.iter()).map(|(&g, &a)| g * a).collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_b),
                self.b.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Elementwise multiplication: `c = a * b`.
pub fn mul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = fast_mul(a, b)?;

    if needs_grad(a, b) {
        let storage = TensorStorage::cpu(result.data()?.to_vec());
        Tensor::from_operation(
            storage,
            result.shape().to_vec(),
            Arc::new(MulBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
    } else {
        Ok(result)
    }
}

// ===========================================================================
// div
// ===========================================================================

/// Backward node for `c = a / b`.
///
/// VJP: da = grad / b, db = -grad * a / (b * b).
#[derive(Debug)]
struct DivBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for DivBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go_data = grad_output.data()?;
        let b_data = self.b.data()?;

        let da = if self.a.requires_grad() {
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(b_data.iter())
                .map(|(&g, &b)| g / b)
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.a.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        let db = if self.b.requires_grad() {
            let a_data = self.a.data()?;
            let grad_b: Vec<T> = go_data
                .iter()
                .zip(a_data.iter().zip(b_data.iter()))
                .map(|(&g, (&a, &b))| -g * a / (b * b))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_b),
                self.b.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

/// Elementwise division: `c = a / b`.
pub fn div<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = binary_map(a, b, |x, y| x / y)?;

    if needs_grad(a, b) {
        let storage = TensorStorage::cpu(result.data()?.to_vec());
        Tensor::from_operation(
            storage,
            result.shape().to_vec(),
            Arc::new(DivBackward {
                a: a.clone(),
                b: b.clone(),
            }),
        )
    } else {
        Ok(result)
    }
}

// ===========================================================================
// neg
// ===========================================================================

/// Backward node for `c = -a`.
///
/// VJP: da = -grad.
#[derive(Debug)]
struct NegBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for NegBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            let go_data = grad_output.data()?;
            let neg: Vec<T> = go_data.iter().map(|&g| -g).collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(neg),
                grad_output.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Elementwise negation: `c = -a`.
pub fn neg<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = unary_map(a, |x| -x)?;

    if needs_grad_unary(a) {
        let storage = TensorStorage::cpu(result.data()?.to_vec());
        Tensor::from_operation(
            storage,
            result.shape().to_vec(),
            Arc::new(NegBackward { a: a.clone() }),
        )
    } else {
        Ok(result)
    }
}

// ===========================================================================
// pow (tensor ^ scalar exponent)
// ===========================================================================

/// Backward node for `c = a ^ exp` where `exp` is a scalar.
///
/// VJP: da = exp * a^(exp-1) * grad.
#[derive(Debug)]
struct PowBackward<T: Float> {
    a: Tensor<T>,
    exp: f64,
}

impl<T: Float> GradFn<T> for PowBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            let go_data = grad_output.data()?;
            let a_data = self.a.data()?;
            let exp_t = T::from(self.exp).unwrap();
            let exp_m1 = T::from(self.exp - 1.0).unwrap();
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(a_data.iter())
                .map(|(&g, &a)| g * exp_t * a.powf(exp_m1))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.a.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "PowBackward"
    }
}

/// Elementwise power: `c = a ^ exp` where `exp` is a scalar `f64`.
pub fn pow<T: Float>(a: &Tensor<T>, exp: f64) -> FerrotorchResult<Tensor<T>> {
    let exp_t = T::from(exp).unwrap();
    let result = scalar_map(a, exp_t, |x, e| x.powf(e))?;

    if needs_grad_unary(a) {
        let storage = TensorStorage::cpu(result.data()?.to_vec());
        Tensor::from_operation(
            storage,
            result.shape().to_vec(),
            Arc::new(PowBackward {
                a: a.clone(),
                exp,
            }),
        )
    } else {
        Ok(result)
    }
}

// ===========================================================================
// sqrt
// ===========================================================================

/// Backward node for `c = sqrt(a)`.
///
/// VJP: da = grad / (2 * sqrt(a)).
#[derive(Debug)]
struct SqrtBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for SqrtBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            let go_data = grad_output.data()?;
            let a_data = self.a.data()?;
            let two = T::from(2.0).unwrap();
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(a_data.iter())
                .map(|(&g, &a)| g / (two * a.sqrt()))
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.a.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Elementwise square root: `c = sqrt(a)`.
pub fn sqrt<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = unary_map(a, |x| x.sqrt())?;

    if needs_grad_unary(a) {
        let storage = TensorStorage::cpu(result.data()?.to_vec());
        Tensor::from_operation(
            storage,
            result.shape().to_vec(),
            Arc::new(SqrtBackward { a: a.clone() }),
        )
    } else {
        Ok(result)
    }
}

// ===========================================================================
// abs
// ===========================================================================

/// Backward node for `c = |a|`.
///
/// VJP: da = grad * sign(a).
#[derive(Debug)]
struct AbsBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for AbsBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            let go_data = grad_output.data()?;
            let a_data = self.a.data()?;
            let zero = <T as num_traits::Zero>::zero();
            let one = <T as num_traits::One>::one();
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(a_data.iter())
                .map(|(&g, &a)| {
                    let sign = if a > zero {
                        one
                    } else if a < zero {
                        -one
                    } else {
                        zero
                    };
                    g * sign
                })
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.a.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "AbsBackward"
    }
}

/// Elementwise absolute value: `c = |a|`.
pub fn abs<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = unary_map(a, |x| x.abs())?;

    if needs_grad_unary(a) {
        let storage = TensorStorage::cpu(result.data()?.to_vec());
        Tensor::from_operation(
            storage,
            result.shape().to_vec(),
            Arc::new(AbsBackward { a: a.clone() }),
        )
    } else {
        Ok(result)
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
    fn test_add_forward() {
        let a = leaf_vec(&[1.0, 2.0, 3.0], false);
        let b = leaf_vec(&[4.0, 5.0, 6.0], false);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_forward() {
        let a = leaf_vec(&[10.0, 20.0, 30.0], false);
        let b = leaf_vec(&[1.0, 2.0, 3.0], false);
        let c = sub(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul_forward() {
        let a = leaf_vec(&[2.0, 3.0, 4.0], false);
        let b = leaf_vec(&[5.0, 6.0, 7.0], false);
        let c = mul(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_forward() {
        let a = leaf_vec(&[10.0, 20.0, 30.0], false);
        let b = leaf_vec(&[2.0, 5.0, 10.0], false);
        let c = div(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_neg_forward() {
        let a = leaf_vec(&[1.0, -2.0, 3.0], false);
        let c = neg(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_pow_forward() {
        let a = leaf_vec(&[2.0, 3.0, 4.0], false);
        let c = pow(&a, 2.0).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 4.0).abs() < 1e-6);
        assert!((d[1] - 9.0).abs() < 1e-6);
        assert!((d[2] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_sqrt_forward() {
        let a = leaf_vec(&[4.0, 9.0, 16.0], false);
        let c = sqrt(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-6);
        assert!((d[1] - 3.0).abs() < 1e-6);
        assert!((d[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_abs_forward() {
        let a = leaf_vec(&[-3.0, 0.0, 5.0], false);
        let c = abs(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[3.0, 0.0, 5.0]);
    }

    // -----------------------------------------------------------------------
    // Backward tests (scalar tensors for simplicity)
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_backward() {
        // c = a + b; dc/da = 1, dc/db = 1.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = add(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_sub_backward() {
        // c = a - b; dc/da = 1, dc/db = -1.
        let a = leaf_scalar(5.0, true);
        let b = leaf_scalar(3.0, true);
        let c = sub(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_mul_backward() {
        // c = a * b; dc/da = b = 3, dc/db = a = 2.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = mul(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 3.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), 2.0, 1e-6);
    }

    #[test]
    fn test_div_backward() {
        // c = a / b; dc/da = 1/b = 1/4, dc/db = -a/b^2 = -6/16 = -0.375.
        let a = leaf_scalar(6.0, true);
        let b = leaf_scalar(4.0, true);
        let c = div(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.25, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), -0.375, 1e-6);
    }

    #[test]
    fn test_neg_backward() {
        // c = -a; dc/da = -1.
        let a = leaf_scalar(7.0, true);
        let c = neg(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_pow_backward() {
        // c = a^3; dc/da = 3 * a^2 = 3 * 4 = 12.
        let a = leaf_scalar(2.0, true);
        let c = pow(&a, 3.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 12.0, 1e-5);
    }

    #[test]
    fn test_sqrt_backward() {
        // c = sqrt(a); dc/da = 1 / (2 * sqrt(a)).
        // a = 4.0 => dc/da = 1 / (2 * 2) = 0.25.
        let a = leaf_scalar(4.0, true);
        let c = sqrt(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.25, 1e-6);
    }

    #[test]
    fn test_abs_backward_positive() {
        // c = |a| where a > 0; dc/da = sign(a) = 1.
        let a = leaf_scalar(3.0, true);
        let c = abs(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_abs_backward_negative() {
        // c = |a| where a < 0; dc/da = sign(a) = -1.
        let a = leaf_scalar(-3.0, true);
        let c = abs(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -1.0, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Tests for no-grad and partial requires_grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_no_grad_fn_when_inputs_detached() {
        let a = leaf_scalar(2.0, false);
        let b = leaf_scalar(3.0, false);
        let c = add(&a, &b).unwrap();
        assert!(c.grad_fn().is_none());
    }

    #[test]
    fn test_mul_partial_requires_grad() {
        // a requires grad, b does not.
        // c = a * b; dc/da = b = 5, dc/db = None.
        let a = leaf_scalar(3.0, true);
        let b = leaf_scalar(5.0, false);
        let c = mul(&a, &b).unwrap();
        assert!(c.grad_fn().is_some());
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 5.0, 1e-6);
        assert!(b.grad().unwrap().is_none());
    }

    #[test]
    fn test_no_grad_context_skips_backward() {
        use crate::autograd::no_grad::no_grad;

        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = no_grad(|| add(&a, &b)).unwrap();
        // Inside no_grad, no grad_fn should be attached.
        assert!(c.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // Chain rule tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_mul_add() {
        // d = a * b + b
        // dd/da = b = 3
        // dd/db = a + 1 = 3
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = mul(&a, &b).unwrap();
        let d = add(&c, &b).unwrap();
        d.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 3.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), 3.0, 1e-6);
    }

    #[test]
    fn test_chain_div_sub() {
        // c = a / b - a
        // dc/da = 1/b - 1 = 1/2 - 1 = -0.5
        // dc/db = -a/b^2 = -3/4 = -0.75
        let a = leaf_scalar(3.0, true);
        let b = leaf_scalar(2.0, true);
        let d = div(&a, &b).unwrap();
        let e = sub(&d, &a).unwrap();
        e.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -0.5, 1e-5);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), -0.75, 1e-5);
    }

    #[test]
    fn test_chain_sqrt_pow() {
        // c = sqrt(a)^2 = a. dc/da = 1.
        // sqrt(9) = 3, pow(3, 2) = 9.
        // d(pow)/d(sqrt) = 2 * sqrt(a) = 6.
        // d(sqrt)/da = 1 / (2*sqrt(a)) = 1/6.
        // dc/da = 6 * 1/6 = 1.
        let a = leaf_scalar(9.0, true);
        let s = sqrt(&a).unwrap();
        let c = pow(&s, 2.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-5);
    }

    #[test]
    fn test_neg_double() {
        // c = -(-a) = a; dc/da = 1.
        let a = leaf_scalar(5.0, true);
        let b = neg(&a).unwrap();
        let c = neg(&b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Vector backward tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul_vector_backward() {
        // c = a * b (elementwise), then sum to scalar for backward.
        // loss = sum(a * b)
        // d(loss)/d(a_i) = b_i, d(loss)/d(b_i) = a_i.
        let a = leaf_vec(&[1.0, 2.0, 3.0], true);
        let b = leaf_vec(&[4.0, 5.0, 6.0], true);
        let c = mul(&a, &b).unwrap();

        // Sum to scalar so we can call backward.
        let c_data = c.data().unwrap().to_vec();
        let total: f32 = c_data.iter().sum();
        let sum_backward = SumBackward { input: c.clone() };
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(sum_backward),
        )
        .unwrap();
        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let a_g = a_grad.data().unwrap();
        assert!((a_g[0] - 4.0).abs() < 1e-6);
        assert!((a_g[1] - 5.0).abs() < 1e-6);
        assert!((a_g[2] - 6.0).abs() < 1e-6);

        let b_grad = b.grad().unwrap().unwrap();
        let b_g = b_grad.data().unwrap();
        assert!((b_g[0] - 1.0).abs() < 1e-6);
        assert!((b_g[1] - 2.0).abs() < 1e-6);
        assert!((b_g[2] - 3.0).abs() < 1e-6);
    }

    /// Helper backward node for sum reduction in tests:
    /// loss = sum(input); d(loss)/d(input_i) = 1.
    #[derive(Debug)]
    struct SumBackward<T: Float> {
        input: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for SumBackward<T> {
        fn backward(&self, _grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let ones_data = vec![<T as num_traits::One>::one(); self.input.numel()];
            let ones = Tensor::from_storage(
                TensorStorage::cpu(ones_data),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(ones)])
        }

        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.input]
        }

        fn name(&self) -> &'static str {
            "SumBackward"
        }
    }
}
