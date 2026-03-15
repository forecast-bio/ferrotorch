//! Backward functions for reduction operations: sum, mean, prod.
//!
//! Each reduction collapses an input tensor to a scalar. The VJP
//! (vector-Jacobian product) broadcasts the upstream scalar gradient
//! back to the input shape.

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::ops::elementwise;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// SumBackward
// ---------------------------------------------------------------------------

/// Backward node for `sum(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output` for all i (broadcast scalar to input shape).
#[derive(Debug)]
pub struct SumBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for SumBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Extract the scalar value — works for both CPU and GPU by
        // transferring to CPU if needed (it's just one number).
        let go = if grad_output.is_cuda() {
            let cpu = grad_output.cpu()?;
            cpu.data()?[0]
        } else {
            grad_output.data()?[0]
        };
        let numel = self.input.numel();
        let data = vec![go; numel];
        let grad_cpu =
            Tensor::from_storage(TensorStorage::cpu(data), self.input.shape().to_vec(), false)?;
        // Place gradient on the same device as the input.
        let grad_input = grad_cpu.to(self.input.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SumBackward"
    }
}

/// Differentiable sum reduction: returns a scalar that is the sum of all elements.
///
/// When gradient tracking is enabled and the input requires grad, the returned
/// tensor carries a [`SumBackward`] node.
pub fn sum<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        let backend = crate::gpu_dispatch::gpu_backend()
            .ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = backend.sum_f32(input.gpu_handle()?, input.numel())?;
        let storage = TensorStorage::gpu(handle);
        let shape = vec![];

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(SumBackward {
                input: input.clone(),
            });
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = elementwise::sum(input)?;

        if is_grad_enabled() && input.requires_grad() {
            let grad_fn = Arc::new(SumBackward {
                input: input.clone(),
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }
}

// ---------------------------------------------------------------------------
// MeanBackward
// ---------------------------------------------------------------------------

/// Backward node for `mean(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output / numel` for all i.
#[derive(Debug)]
pub struct MeanBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for MeanBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = if grad_output.is_cuda() {
            let cpu = grad_output.cpu()?;
            cpu.data()?[0]
        } else {
            grad_output.data()?[0]
        };
        let numel = self.input.numel();
        let n = T::from(numel).unwrap();
        let val = go / n;
        let data = vec![val; numel];
        let grad_cpu =
            Tensor::from_storage(TensorStorage::cpu(data), self.input.shape().to_vec(), false)?;
        let grad_input = grad_cpu.to(self.input.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MeanBackward"
    }
}

/// Differentiable mean reduction: returns a scalar that is the mean of all elements.
///
/// When gradient tracking is enabled and the input requires grad, the returned
/// tensor carries a [`MeanBackward`] node.
pub fn mean<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = elementwise::mean(input)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(MeanBackward {
            input: input.clone(),
        });
        Tensor::from_operation(
            TensorStorage::cpu(result.data()?.to_vec()),
            result.shape().to_vec(),
            grad_fn,
        )
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// ProdBackward
// ---------------------------------------------------------------------------

/// Backward node for `prod(input) -> scalar`.
///
/// VJP: `grad_input[i] = grad_output * prod(input) / input[i]`.
///
/// When any `input[i]` is zero, we recompute the partial product excluding
/// that element to avoid division by zero. This is done via prefix/suffix
/// products.
#[derive(Debug)]
pub struct ProdBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GradFn<T> for ProdBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let go = if grad_output.is_cuda() {
            let cpu = grad_output.cpu()?;
            cpu.data()?[0]
        } else {
            grad_output.data()?[0]
        };

        // Transfer input to CPU if on GPU for the prefix/suffix computation.
        let input_cpu = if self.input.is_cuda() {
            self.input.cpu()?
        } else {
            self.input.clone()
        };
        let input_data = input_cpu.data()?;
        let n = input_data.len();

        // Use prefix/suffix products to avoid division by zero.
        // prefix[i] = product of input[0..i]
        // suffix[i] = product of input[i+1..n]
        // grad[i] = go * prefix[i] * suffix[i]
        let mut prefix = vec![<T as num_traits::One>::one(); n];
        for i in 1..n {
            prefix[i] = prefix[i - 1] * input_data[i - 1];
        }

        let mut suffix = vec![<T as num_traits::One>::one(); n];
        if n > 1 {
            for i in (0..n - 1).rev() {
                suffix[i] = suffix[i + 1] * input_data[i + 1];
            }
        }

        let grad_data: Vec<T> = (0..n).map(|i| go * prefix[i] * suffix[i]).collect();

        let grad_cpu = Tensor::from_storage(
            TensorStorage::cpu(grad_data),
            self.input.shape().to_vec(),
            false,
        )?;
        // Place gradient on the same device as the input.
        let grad_input = grad_cpu.to(self.input.device())?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ProdBackward"
    }
}

/// Differentiable product reduction: returns a scalar that is the product
/// of all elements.
///
/// When gradient tracking is enabled and the input requires grad, the returned
/// tensor carries a [`ProdBackward`] node.
pub fn prod<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let total = data
        .iter()
        .copied()
        .fold(<T as num_traits::One>::one(), |a, b| a * b);
    let result = Tensor::from_storage(TensorStorage::cpu(vec![total]), vec![], false)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(ProdBackward {
            input: input.clone(),
        });
        Tensor::from_operation(
            TensorStorage::cpu(result.data()?.to_vec()),
            result.shape().to_vec(),
            grad_fn,
        )
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::no_grad::no_grad;
    use crate::storage::TensorStorage;

    /// Helper: create a leaf tensor with given data, shape, and requires_grad.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), requires_grad)
            .unwrap()
    }

    /// Helper: create a leaf scalar.
    fn leaf_scalar(val: f64, requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    // --- Forward tests ---

    #[test]
    fn test_sum_forward_1d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let s = sum(&x).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 10.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_forward_2d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let s = sum(&x).unwrap();
        assert!((s.item().unwrap() - 21.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_forward() {
        let x = leaf(&[2.0, 4.0, 6.0, 8.0], &[4], false);
        let m = mean(&x).unwrap();
        assert!((m.item().unwrap() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_forward() {
        let x = leaf(&[2.0, 3.0, 4.0], &[3], false);
        let p = prod(&x).unwrap();
        assert!((p.item().unwrap() - 24.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_forward_scalar() {
        let x = leaf_scalar(7.0, false);
        let p = prod(&x).unwrap();
        assert!((p.item().unwrap() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_forward_with_zero() {
        let x = leaf(&[3.0, 0.0, 5.0], &[3], false);
        let p = prod(&x).unwrap();
        assert!((p.item().unwrap()).abs() < 1e-12);
    }

    // --- Backward tests ---

    #[test]
    fn test_sum_backward_scalar_input() {
        // sum(x) where x is a scalar = x. Gradient should be 1.
        let x = leaf_scalar(5.0, true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_sum_backward_1d() {
        // sum([a, b, c]) = a + b + c. d/d(each) = 1.
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert_eq!(gd.len(), 3);
        for &v in gd {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sum_backward_2d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        for &v in g.data().unwrap() {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mean_backward_scalar_input() {
        // mean(x) where x is a scalar = x. Gradient should be 1.
        let x = leaf_scalar(5.0, true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_mean_backward_1d() {
        // mean([a, b, c]) = (a + b + c) / 3. d/d(each) = 1/3.
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        let expected = 1.0 / 3.0;
        for &v in gd {
            assert!((v - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_mean_backward_2d() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert_eq!(g.shape(), &[2, 3]);
        let expected = 1.0 / 6.0;
        for &v in g.data().unwrap() {
            assert!((v - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn test_prod_backward_scalar_input() {
        // prod(x) where x is scalar = x. Gradient should be 1.
        let x = leaf_scalar(5.0, true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        assert!((g.item().unwrap() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_prod_backward_1d() {
        // prod([a, b, c]) = a*b*c.
        // d/da = b*c, d/db = a*c, d/dc = a*b.
        let x = leaf(&[2.0, 3.0, 4.0], &[3], true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 12.0).abs() < 1e-12, "d/da = 3*4 = 12, got {}", gd[0]);
        assert!((gd[1] - 8.0).abs() < 1e-12, "d/db = 2*4 = 8, got {}", gd[1]);
        assert!((gd[2] - 6.0).abs() < 1e-12, "d/dc = 2*3 = 6, got {}", gd[2]);
    }

    #[test]
    fn test_prod_backward_with_zero() {
        // prod([3, 0, 5]) = 0.
        // d/d(x0) = 0*5 = 0, d/d(x1) = 3*5 = 15, d/d(x2) = 3*0 = 0.
        let x = leaf(&[3.0, 0.0, 5.0], &[3], true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        assert!((gd[0] - 0.0).abs() < 1e-12, "got {}", gd[0]);
        assert!((gd[1] - 15.0).abs() < 1e-12, "got {}", gd[1]);
        assert!((gd[2] - 0.0).abs() < 1e-12, "got {}", gd[2]);
    }

    #[test]
    fn test_prod_backward_two_zeros() {
        // prod([0, 0, 5]) = 0.
        // All gradients should be 0 (each product-excluding-one still contains a zero).
        let x = leaf(&[0.0, 0.0, 5.0], &[3], true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();

        let g = x.grad().unwrap().unwrap();
        let gd = g.data().unwrap();
        for &v in gd {
            assert!((v).abs() < 1e-12, "expected 0, got {v}");
        }
    }

    // --- Gradient tracking / no_grad tests ---

    #[test]
    fn test_sum_no_grad_fn_when_input_not_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let s = sum(&x).unwrap();
        assert!(s.grad_fn().is_none());
        assert!(!s.requires_grad());
    }

    #[test]
    fn test_sum_has_grad_fn_when_input_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = sum(&x).unwrap();
        assert!(s.grad_fn().is_some());
        assert_eq!(s.grad_fn().unwrap().name(), "SumBackward");
        assert!(s.requires_grad());
    }

    #[test]
    fn test_mean_has_grad_fn_when_input_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = mean(&x).unwrap();
        assert!(m.grad_fn().is_some());
        assert_eq!(m.grad_fn().unwrap().name(), "MeanBackward");
    }

    #[test]
    fn test_prod_has_grad_fn_when_input_requires_grad() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let p = prod(&x).unwrap();
        assert!(p.grad_fn().is_some());
        assert_eq!(p.grad_fn().unwrap().name(), "ProdBackward");
    }

    #[test]
    fn test_sum_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let s = no_grad(|| sum(&x)).unwrap();
        assert!(s.grad_fn().is_none());
        assert!(!s.requires_grad());
    }

    #[test]
    fn test_mean_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let m = no_grad(|| mean(&x)).unwrap();
        assert!(m.grad_fn().is_none());
    }

    #[test]
    fn test_prod_no_grad_fn_in_no_grad_context() {
        let x = leaf(&[2.0, 3.0], &[2], true);
        let p = no_grad(|| prod(&x)).unwrap();
        assert!(p.grad_fn().is_none());
    }

    // --- Numerical gradient checking ---

    /// Finite-difference gradient check for a scalar -> scalar function.
    fn numerical_grad_check(
        f: impl Fn(&Tensor<f64>) -> FerrotorchResult<Tensor<f64>>,
        x_val: f64,
        expected_analytic: f64,
        tol: f64,
    ) {
        let eps = 1e-7;

        let x_plus = leaf_scalar(x_val + eps, false);
        let x_minus = leaf_scalar(x_val - eps, false);

        let f_plus = f(&x_plus).unwrap().item().unwrap();
        let f_minus = f(&x_minus).unwrap().item().unwrap();
        let numerical = (f_plus - f_minus) / (2.0 * eps);

        assert!(
            (numerical - expected_analytic).abs() < tol,
            "numerical gradient {numerical} differs from analytic {expected_analytic} by more than {tol}"
        );
    }

    #[test]
    fn test_sum_numerical_gradient() {
        // sum(x) for scalar x: d/dx = 1.
        let x = leaf_scalar(3.0, true);
        let s = sum(&x).unwrap();
        s.backward().unwrap();
        let analytic = x.grad().unwrap().unwrap().item().unwrap();

        numerical_grad_check(sum, 3.0, analytic, 1e-5);
    }

    #[test]
    fn test_mean_numerical_gradient() {
        // mean(x) for scalar x: d/dx = 1.
        let x = leaf_scalar(3.0, true);
        let m = mean(&x).unwrap();
        m.backward().unwrap();
        let analytic = x.grad().unwrap().unwrap().item().unwrap();

        numerical_grad_check(mean, 3.0, analytic, 1e-5);
    }

    #[test]
    fn test_prod_numerical_gradient() {
        // prod(x) for scalar x: d/dx = 1.
        let x = leaf_scalar(3.0, true);
        let p = prod(&x).unwrap();
        p.backward().unwrap();
        let analytic = x.grad().unwrap().unwrap().item().unwrap();

        numerical_grad_check(prod, 3.0, analytic, 1e-5);
    }
}
