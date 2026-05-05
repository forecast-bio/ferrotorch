//! Operator overloading for Tensor: Add, Sub, Mul, Div, Neg.
//!
//! Enables `let c = &a + &b` syntax instead of `grad_fns::arithmetic::add(&a, &b)`.
//! All overloads delegate to the differentiable grad_fns, so autograd works
//! transparently through operators.

use std::ops;

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::grad_fns::arithmetic;
use crate::tensor::Tensor;

// --- Add ---

impl<T: Float> ops::Add<&Tensor<T>> for &Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        arithmetic::add(self, rhs)
    }
}

impl<T: Float> ops::Add<Tensor<T>> for &Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn add(self, rhs: Tensor<T>) -> Self::Output {
        arithmetic::add(self, &rhs)
    }
}

impl<T: Float> ops::Add<&Tensor<T>> for Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn add(self, rhs: &Tensor<T>) -> Self::Output {
        arithmetic::add(&self, rhs)
    }
}

impl<T: Float> ops::Add<Tensor<T>> for Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn add(self, rhs: Tensor<T>) -> Self::Output {
        arithmetic::add(&self, &rhs)
    }
}

// --- Sub ---

impl<T: Float> ops::Sub<&Tensor<T>> for &Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        arithmetic::sub(self, rhs)
    }
}

impl<T: Float> ops::Sub<Tensor<T>> for &Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn sub(self, rhs: Tensor<T>) -> Self::Output {
        arithmetic::sub(self, &rhs)
    }
}

impl<T: Float> ops::Sub<&Tensor<T>> for Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn sub(self, rhs: &Tensor<T>) -> Self::Output {
        arithmetic::sub(&self, rhs)
    }
}

impl<T: Float> ops::Sub<Tensor<T>> for Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn sub(self, rhs: Tensor<T>) -> Self::Output {
        arithmetic::sub(&self, &rhs)
    }
}

// --- Mul ---

impl<T: Float> ops::Mul<&Tensor<T>> for &Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn mul(self, rhs: &Tensor<T>) -> Self::Output {
        arithmetic::mul(self, rhs)
    }
}

impl<T: Float> ops::Mul<Tensor<T>> for &Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn mul(self, rhs: Tensor<T>) -> Self::Output {
        arithmetic::mul(self, &rhs)
    }
}

impl<T: Float> ops::Mul<&Tensor<T>> for Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn mul(self, rhs: &Tensor<T>) -> Self::Output {
        arithmetic::mul(&self, rhs)
    }
}

impl<T: Float> ops::Mul<Tensor<T>> for Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn mul(self, rhs: Tensor<T>) -> Self::Output {
        arithmetic::mul(&self, &rhs)
    }
}

// --- Div ---

impl<T: Float> ops::Div<&Tensor<T>> for &Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn div(self, rhs: &Tensor<T>) -> Self::Output {
        arithmetic::div(self, rhs)
    }
}

impl<T: Float> ops::Div<Tensor<T>> for &Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn div(self, rhs: Tensor<T>) -> Self::Output {
        arithmetic::div(self, &rhs)
    }
}

impl<T: Float> ops::Div<&Tensor<T>> for Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn div(self, rhs: &Tensor<T>) -> Self::Output {
        arithmetic::div(&self, rhs)
    }
}

impl<T: Float> ops::Div<Tensor<T>> for Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn div(self, rhs: Tensor<T>) -> Self::Output {
        arithmetic::div(&self, &rhs)
    }
}

// --- Neg ---

impl<T: Float> ops::Neg for &Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn neg(self) -> Self::Output {
        arithmetic::neg(self)
    }
}

impl<T: Float> ops::Neg for Tensor<T> {
    type Output = FerrotorchResult<Tensor<T>>;
    fn neg(self) -> Self::Output {
        arithmetic::neg(&self)
    }
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    // reason: 2 + 3 = 5 in f32 is bit-exact (small integers); add-grad
    // is exactly 1.0 by construction. Equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_add_refs() {
        let a = scalar(2.0f32).unwrap().requires_grad_(true);
        let b = scalar(3.0f32).unwrap().requires_grad_(true);
        let c = (&a + &b).unwrap();
        assert_eq!(c.item().unwrap(), 5.0);
        c.backward().unwrap();
        assert_eq!(a.grad().unwrap().unwrap().item().unwrap(), 1.0);
        assert_eq!(b.grad().unwrap().unwrap().item().unwrap(), 1.0);
    }

    #[test]
    // reason: 5 - 3 = 2 in f32 is bit-exact (small integers).
    #[allow(clippy::float_cmp)]
    fn test_sub_refs() {
        let a = scalar(5.0f32).unwrap();
        let b = scalar(3.0f32).unwrap();
        assert_eq!((&a - &b).unwrap().item().unwrap(), 2.0);
    }

    #[test]
    // reason: 4 * 3 = 12 in f32 is bit-exact (small integers); mul-grads
    // are the other operand exactly. Equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_mul_with_autograd() {
        let a = scalar(4.0f32).unwrap().requires_grad_(true);
        let b = scalar(3.0f32).unwrap().requires_grad_(true);
        let c = (&a * &b).unwrap();
        assert_eq!(c.item().unwrap(), 12.0);
        c.backward().unwrap();
        assert_eq!(a.grad().unwrap().unwrap().item().unwrap(), 3.0);
        assert_eq!(b.grad().unwrap().unwrap().item().unwrap(), 4.0);
    }

    #[test]
    // reason: 6 / 2 = 3 in f32 is bit-exact (powers of 2 in division).
    #[allow(clippy::float_cmp)]
    fn test_div_refs() {
        let a = scalar(6.0f32).unwrap();
        let b = scalar(2.0f32).unwrap();
        assert_eq!((&a / &b).unwrap().item().unwrap(), 3.0);
    }

    #[test]
    // reason: negation flips the sign bit only — bit-exact for any finite
    // input including the small integers used here.
    #[allow(clippy::float_cmp)]
    fn test_neg() {
        let a = scalar(5.0f32).unwrap();
        assert_eq!((-&a).unwrap().item().unwrap(), -5.0);
        assert_eq!((-scalar(3.0f32).unwrap()).unwrap().item().unwrap(), -3.0);
    }

    #[test]
    // reason: 2 + 3 = 5 in f32 is bit-exact (small integers).
    #[allow(clippy::float_cmp)]
    fn test_owned_add() {
        let c = (scalar(2.0f32).unwrap() + scalar(3.0f32).unwrap()).unwrap();
        assert_eq!(c.item().unwrap(), 5.0);
    }

    #[test]
    // reason: 2 + 3 = 5 in f32 is bit-exact (small integers).
    #[allow(clippy::float_cmp)]
    fn test_mixed_ownership() {
        let a = scalar(2.0f32).unwrap();
        let b = scalar(3.0f32).unwrap();
        assert_eq!((a + &b).unwrap().item().unwrap(), 5.0);
    }

    #[test]
    // reason: (2+3)*(2-3) = 5 * -1 = -5; every step is bit-exact in f32
    // because operands are small integers.
    #[allow(clippy::float_cmp)]
    fn test_chained_expression() {
        let a = scalar(2.0f32).unwrap().requires_grad_(true);
        let b = scalar(3.0f32).unwrap().requires_grad_(true);
        // (a + b) * (a - b) = (2+3)*(2-3) = 5 * -1 = -5
        let c = (&(&a + &b).unwrap() * &(&a - &b).unwrap()).unwrap();
        assert_eq!(c.item().unwrap(), -5.0);
    }
}
