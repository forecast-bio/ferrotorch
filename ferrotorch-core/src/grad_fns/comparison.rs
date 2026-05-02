//! Backward function for the differentiable conditional `where_` operation.
//!
//! `where_(condition, x, y)` selects from `x` where `condition` is true, and
//! from `y` where `condition` is false. The VJP routes the upstream gradient
//! to `x` at true positions and to `y` at false positions.

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

use crate::bool_tensor::BoolTensor;
use crate::error::FerrotorchError;

/// Backward node for `where_(condition, x, y)`.
///
/// Stores the boolean condition mask and references to both input tensors
/// so the autograd engine can traverse the graph.
#[derive(Debug)]
pub struct WhereBackward<T: Float> {
    condition: Vec<bool>,
    x: Tensor<T>,
    y: Tensor<T>,
}

impl<T: Float> GradFn<T> for WhereBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let device = grad_output.device();
        let go = grad_output.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();

        // grad_x = grad_output where condition is true, 0 otherwise
        let grad_x: Vec<T> = go
            .iter()
            .zip(self.condition.iter())
            .map(|(&g, &c)| if c { g } else { zero })
            .collect();

        // grad_y = grad_output where condition is false, 0 otherwise
        let grad_y: Vec<T> = go
            .iter()
            .zip(self.condition.iter())
            .map(|(&g, &c)| if c { zero } else { g })
            .collect();

        let grad_x_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_x), self.x.shape().to_vec(), false)?;
        let grad_y_tensor =
            Tensor::from_storage(TensorStorage::cpu(grad_y), self.y.shape().to_vec(), false)?;

        if device.is_cuda() {
            Ok(vec![
                Some(grad_x_tensor.to(device)?),
                Some(grad_y_tensor.to(device)?),
            ])
        } else {
            Ok(vec![Some(grad_x_tensor), Some(grad_y_tensor)])
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.x, &self.y]
    }

    fn name(&self) -> &'static str {
        "WhereBackward"
    }
}

/// Differentiable conditional selection.
///
/// For each element `i`, the output is `x[i]` if `condition[i]` is true,
/// otherwise `y[i]`. All three inputs must have the same length.
///
/// When gradient tracking is enabled and either input requires grad, the
/// returned tensor carries a [`WhereBackward`] node that routes gradients
/// to the appropriate input during the backward pass.
pub fn where_<T: Float>(
    condition: &[bool],
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let device = x.device();
    let x_data = x.data_vec()?;
    let y_data = y.data_vec()?;

    debug_assert_eq!(condition.len(), x_data.len());
    debug_assert_eq!(condition.len(), y_data.len());

    let result: Vec<T> = condition
        .iter()
        .zip(x_data.iter().zip(y_data.iter()))
        .map(|(&c, (&xv, &yv))| if c { xv } else { yv })
        .collect();

    let needs_grad = is_grad_enabled() && (x.requires_grad() || y.requires_grad());

    let storage = TensorStorage::on_device(result, device)?;
    if needs_grad {
        let grad_fn = Arc::new(WhereBackward {
            condition: condition.to_vec(),
            x: x.clone(),
            y: y.clone(),
        });
        Tensor::from_operation(storage, x.shape().to_vec(), grad_fn)
    } else {
        Tensor::from_storage(storage, x.shape().to_vec(), false)
    }
}

// ---------------------------------------------------------------------------
// First-class BoolTensor wrapper (#615)
// ---------------------------------------------------------------------------

/// Pointwise ternary `where(cond, x, y)` taking a [`BoolTensor`] for
/// the condition. Mirrors `torch.where(cond, x, y)`. The mask must
/// have the same numel as `x` / `y`.
pub fn where_bt<T: Float>(
    cond: &BoolTensor,
    x: &Tensor<T>,
    y: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if cond.numel() != x.numel() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "where_bt: cond numel={} != x numel={}",
                cond.numel(),
                x.numel()
            ),
        });
    }
    if x.shape() != y.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "where_bt: x shape {:?} != y shape {:?}",
                x.shape(),
                y.shape()
            ),
        });
    }
    where_(cond.data(), x, y)
}

#[cfg(test)]
mod first_class_tests {
    use super::*;

    #[test]
    fn where_bt_picks_correctly() {
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f32, 2.0, 3.0, 4.0]),
            vec![4],
            false,
        )
        .unwrap();
        let y = Tensor::from_storage(
            TensorStorage::cpu(vec![10.0_f32, 20.0, 30.0, 40.0]),
            vec![4],
            false,
        )
        .unwrap();
        let cond = BoolTensor::from_vec(vec![true, false, true, false], vec![4]).unwrap();
        let out = where_bt(&cond, &x, &y).unwrap();
        assert_eq!(out.data().unwrap(), &[1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn where_bt_rejects_shape_mismatch() {
        let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f32; 4]), vec![4], false).unwrap();
        let y = Tensor::from_storage(TensorStorage::cpu(vec![0.0_f32; 4]), vec![4], false).unwrap();
        let cond = BoolTensor::from_vec(vec![true; 3], vec![3]).unwrap();
        let err = where_bt(&cond, &x, &y).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::graph::backward;
    use crate::storage::TensorStorage;

    /// Helper to make a leaf tensor from a slice.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    #[test]
    fn test_where_forward() {
        let cond = vec![true, false, true, false];
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], false);

        let out = where_(&cond, &x, &y).unwrap();
        assert_eq!(out.data().unwrap(), &[1.0, 20.0, 3.0, 40.0]);
    }

    #[test]
    fn test_where_backward() {
        // condition = [true, false, true, false]
        // out = where_(cond, x, y) = [x0, y1, x2, y3]
        //
        // To get a scalar for backward, compute sum(out).
        // grad_output for where_ is all 1s (from sum backward).
        //
        // Expected gradients:
        //   grad_x = [1.0, 0.0, 1.0, 0.0]  (gradient flows where condition is true)
        //   grad_y = [0.0, 1.0, 0.0, 1.0]  (gradient flows where condition is false)
        let cond = vec![true, false, true, false];
        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);
        let y = leaf(&[10.0, 20.0, 30.0, 40.0], &[4], true);

        let out = where_(&cond, &x, &y).unwrap();

        // sum(out) to get a scalar for backward
        let out_data = out.data().unwrap();
        let total: f32 = out_data.iter().sum();

        // Build sum node: backward of sum passes ones as grad to its input.
        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
            numel: usize,
        }

        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.numel];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let scalar = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: out.clone(),
                numel: 4,
            }),
        )
        .unwrap();

        backward(&scalar).unwrap();

        let x_grad = x.grad().unwrap().unwrap();
        let y_grad = y.grad().unwrap().unwrap();

        assert_eq!(x_grad.data().unwrap(), &[1.0, 0.0, 1.0, 0.0]);
        assert_eq!(y_grad.data().unwrap(), &[0.0, 1.0, 0.0, 1.0]);
    }

    #[test]
    fn test_where_no_grad() {
        crate::autograd::no_grad::no_grad(|| {
            let cond = vec![true, false];
            let x = leaf(&[1.0, 2.0], &[2], true);
            let y = leaf(&[10.0, 20.0], &[2], true);

            let out = where_(&cond, &x, &y).unwrap();
            assert!(!out.requires_grad());
            assert!(out.grad_fn().is_none());
            assert_eq!(out.data().unwrap(), &[1.0, 20.0]);
        });
    }
}
