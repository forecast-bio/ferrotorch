//! Differentiable optimizer steps for meta-learning.
//!
//! Regular optimizers (`Sgd::step()`, `AdamW::step()`, etc.) update
//! parameters in-place inside a `no_grad()` block for efficiency —
//! the update itself is not tracked by autograd. This is the correct
//! behavior for standard training.
//!
//! Meta-learning methods like MAML (Finn et al. 2017) need gradients
//! to flow *through* the inner-loop optimizer step, so the outer
//! loss of the adapted parameters can be differentiated back to the
//! original parameters. This module provides differentiable step
//! functions that return fresh tensors with autograd edges instead
//! of mutating parameters in place.
//!
//! # Example (MAML inner loop)
//!
//! ```ignore
//! use ferrotorch_optim::differentiable::diff_sgd_step;
//! use ferrotorch_core::grad_fns::arithmetic::{add, sub, mul};
//!
//! // Outer: theta (global params) requires_grad
//! let theta = vec![initial_param.clone()];
//!
//! // Inner loop: compute inner loss, differentiable step
//! let inner_loss = task_loss(&theta, support_data)?;
//! let grads = grad(&inner_loss, &theta)?;
//! let adapted = diff_sgd_step(&theta, &grads, 1e-2)?;
//!
//! // Outer: compute meta-loss on adapted, backprop to theta
//! let outer_loss = task_loss(&adapted, query_data)?;
//! outer_loss.backward()?;
//! // theta.grad() is now populated with the second-order gradient
//! ```
//!
//! CL-389.

use ferrotorch_core::creation;
use ferrotorch_core::grad_fns::arithmetic::{add, mul, sub};
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

/// Differentiable SGD step: `new_param = param - lr * grad`.
///
/// Unlike `Sgd::step()`, this returns a fresh tensor with autograd
/// edges back to both `param` and `grad` (plus the internal scalar
/// `lr`). Use this inside meta-learning inner loops where the outer
/// loss needs to differentiate back through the inner update.
///
/// Input and output lengths must match. Parameters and gradients
/// are paired element-wise in order.
///
/// # Errors
///
/// - `params` and `grads` have different lengths
/// - Any inner arithmetic op fails (shape mismatch, device mismatch)
pub fn diff_sgd_step<T: Float>(
    params: &[Tensor<T>],
    grads: &[Tensor<T>],
    lr: f64,
) -> FerrotorchResult<Vec<Tensor<T>>> {
    if params.len() != grads.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "diff_sgd_step: params.len() ({}) != grads.len() ({})",
                params.len(),
                grads.len()
            ),
        });
    }
    let lr_t = T::from(lr).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("diff_sgd_step: lr {lr} not representable in the tensor dtype"),
    })?;

    let mut updated = Vec::with_capacity(params.len());
    for (p, g) in params.iter().zip(grads.iter()) {
        if p.shape() != g.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "diff_sgd_step: param shape {:?} != grad shape {:?}",
                    p.shape(),
                    g.shape()
                ),
            });
        }
        // Scalar lr tensor on the same device as the parameter.
        let lr_scalar = creation::scalar(lr_t)?.to(p.device())?;
        // update = lr * grad
        let update = mul(g, &lr_scalar)?;
        // new_param = param - update
        let new_param = sub(p, &update)?;
        updated.push(new_param);
    }
    Ok(updated)
}

/// Updated `(params, velocities)` returned by [`diff_sgd_momentum_step`].
pub type DiffSgdMomentumOutput<T> = (Vec<Tensor<T>>, Vec<Tensor<T>>);

/// Differentiable SGD step with momentum: `velocity = momentum * prev_v + grad`
/// and `new_param = param - lr * velocity`.
///
/// Returns both the new parameters AND the new velocity buffers so
/// the caller can chain multiple inner steps. `prev_velocities` may
/// be empty (or all zeros) on the first step.
///
/// # Errors
///
/// - params, grads, and prev_velocities (when non-empty) have
///   mismatched lengths
/// - Any inner arithmetic op fails
pub fn diff_sgd_momentum_step<T: Float>(
    params: &[Tensor<T>],
    grads: &[Tensor<T>],
    prev_velocities: &[Tensor<T>],
    lr: f64,
    momentum: f64,
) -> FerrotorchResult<DiffSgdMomentumOutput<T>> {
    if params.len() != grads.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "diff_sgd_momentum_step: params.len() ({}) != grads.len() ({})",
                params.len(),
                grads.len()
            ),
        });
    }
    if !prev_velocities.is_empty() && prev_velocities.len() != params.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "diff_sgd_momentum_step: prev_velocities.len() ({}) must be 0 or match params.len() ({})",
                prev_velocities.len(),
                params.len()
            ),
        });
    }

    let lr_t = T::from(lr).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("diff_sgd_momentum_step: lr {lr} not representable"),
    })?;
    let mom_t = T::from(momentum).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!("diff_sgd_momentum_step: momentum {momentum} not representable"),
    })?;

    let mut new_params = Vec::with_capacity(params.len());
    let mut new_velocities = Vec::with_capacity(params.len());

    for (i, (p, g)) in params.iter().zip(grads.iter()).enumerate() {
        if p.shape() != g.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "diff_sgd_momentum_step: param shape {:?} != grad shape {:?}",
                    p.shape(),
                    g.shape()
                ),
            });
        }
        let lr_scalar = creation::scalar(lr_t)?.to(p.device())?;
        let mom_scalar = creation::scalar(mom_t)?.to(p.device())?;

        // v_new = momentum * v_prev + grad
        // On the first step (empty prev_velocities), v_new = grad.
        let v_new = if prev_velocities.is_empty() {
            g.clone()
        } else {
            let v_prev = &prev_velocities[i];
            if v_prev.shape() != g.shape() {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "diff_sgd_momentum_step: prev_velocity shape {:?} != grad shape {:?}",
                        v_prev.shape(),
                        g.shape()
                    ),
                });
            }
            let scaled_v = mul(v_prev, &mom_scalar)?;
            add(&scaled_v, g)?
        };
        // new_param = param - lr * v_new
        let update = mul(&v_new, &lr_scalar)?;
        let new_param = sub(p, &update)?;
        new_params.push(new_param);
        new_velocities.push(v_new);
    }
    Ok((new_params, new_velocities))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::grad_fns::reduction::sum;
    use ferrotorch_core::storage::TensorStorage;

    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    #[test]
    fn test_diff_sgd_step_shape_and_values() {
        let p = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let g = leaf(&[0.1, 0.2, 0.3], &[3], false);
        let out = diff_sgd_step(&[p], &[g], 0.5).unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].shape(), &[3]);
        let data = out[0].data().unwrap();
        // new = 1 - 0.5*0.1 = 0.95, 2 - 0.1 = 1.9, 3 - 0.15 = 2.85
        assert!((data[0] - 0.95).abs() < 1e-5);
        assert!((data[1] - 1.9).abs() < 1e-5);
        assert!((data[2] - 2.85).abs() < 1e-5);
    }

    #[test]
    fn test_diff_sgd_step_length_mismatch_errors() {
        let p = leaf(&[1.0], &[1], true);
        let g1 = leaf(&[0.1], &[1], false);
        let g2 = leaf(&[0.2], &[1], false);
        let result = diff_sgd_step(&[p], &[g1, g2], 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_sgd_step_shape_mismatch_errors() {
        let p = leaf(&[1.0, 2.0], &[2], true);
        let g = leaf(&[0.1, 0.2, 0.3], &[3], false);
        let result = diff_sgd_step(&[p], &[g], 0.1);
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_sgd_step_multiple_params() {
        let p1 = leaf(&[1.0, 2.0], &[2], true);
        let p2 = leaf(&[3.0, 4.0], &[2], true);
        let g1 = leaf(&[0.1, 0.1], &[2], false);
        let g2 = leaf(&[0.2, 0.2], &[2], false);
        let out = diff_sgd_step(&[p1, p2], &[g1, g2], 1.0).unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0].data().unwrap(), &[0.9, 1.9]);
        assert_eq!(out[1].data().unwrap(), &[2.8, 3.8]);
    }

    #[test]
    fn test_diff_sgd_step_autograd_edge_to_param() {
        // theta requires grad; adapted = theta - lr*grad; loss = sum(adapted);
        // d(loss)/d(theta) = 1 (since grad is detached).
        let theta = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let g = leaf(&[0.1, 0.2, 0.3], &[3], false);
        let adapted = diff_sgd_step(std::slice::from_ref(&theta), &[g], 0.1).unwrap();
        let loss = sum(&adapted[0]).unwrap();
        loss.backward().unwrap();
        let theta_grad = theta.grad().unwrap().unwrap();
        // d(sum(theta - 0.1*g))/d(theta) = [1, 1, 1]
        let gd = theta_grad.data().unwrap();
        for &v in gd {
            assert!((v - 1.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_diff_sgd_step_second_order_via_grad_tensor() {
        // The inner gradient itself is a tensor that requires_grad —
        // when we differentiate the adapted params w.r.t. the original
        // grad tensor, we get -lr * identity.
        // Test: theta = 1.0; grad_tensor = 0.5 (requires_grad);
        // adapted = theta - lr*grad = 1 - 0.1*0.5 = 0.95;
        // loss = adapted (scalar); d(loss)/d(grad) = -0.1.
        let theta = leaf(&[1.0], &[1], false);
        let g = leaf(&[0.5], &[1], true);
        let adapted = diff_sgd_step(&[theta], std::slice::from_ref(&g), 0.1).unwrap();
        let loss = sum(&adapted[0]).unwrap();
        loss.backward().unwrap();
        let grad_of_g = g.grad().unwrap().unwrap();
        let gd = grad_of_g.data().unwrap();
        // d/d(g) of (theta - 0.1*g) = -0.1
        assert!((gd[0] - (-0.1)).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // diff_sgd_momentum_step
    // -----------------------------------------------------------------------

    #[test]
    fn test_diff_sgd_momentum_first_step_uses_grad_as_velocity() {
        let p = leaf(&[10.0, 20.0], &[2], true);
        let g = leaf(&[1.0, 2.0], &[2], false);
        // Empty prev_velocities → v_new = g; p_new = p - lr*g.
        let (new_p, new_v) = diff_sgd_momentum_step(&[p], &[g], &[], 0.1, 0.9).unwrap();
        assert_eq!(new_p[0].data().unwrap(), &[9.9, 19.8]);
        assert_eq!(new_v[0].data().unwrap(), &[1.0, 2.0]);
    }

    #[test]
    fn test_diff_sgd_momentum_second_step_uses_velocity() {
        let p = leaf(&[10.0], &[1], true);
        let g = leaf(&[1.0], &[1], false);
        let v_prev = leaf(&[2.0], &[1], false);
        // v_new = 0.9*2.0 + 1.0 = 2.8
        // p_new = 10 - 0.1*2.8 = 9.72
        let (new_p, new_v) = diff_sgd_momentum_step(&[p], &[g], &[v_prev], 0.1, 0.9).unwrap();
        assert!((new_v[0].data().unwrap()[0] - 2.8).abs() < 1e-5);
        assert!((new_p[0].data().unwrap()[0] - 9.72).abs() < 1e-5);
    }

    #[test]
    fn test_diff_sgd_momentum_length_mismatch_errors() {
        let p = leaf(&[1.0], &[1], true);
        let g1 = leaf(&[0.1], &[1], false);
        let g2 = leaf(&[0.2], &[1], false);
        let result = diff_sgd_momentum_step(&[p], &[g1, g2], &[], 0.1, 0.9);
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_sgd_momentum_velocity_length_mismatch_errors() {
        let p = leaf(&[1.0], &[1], true);
        let g = leaf(&[0.1], &[1], false);
        let v1 = leaf(&[0.0], &[1], false);
        let v2 = leaf(&[0.0], &[1], false);
        // prev_velocities is non-empty but wrong length.
        let result = diff_sgd_momentum_step(&[p], &[g], &[v1, v2], 0.1, 0.9);
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_sgd_momentum_shape_mismatch_errors() {
        let p = leaf(&[1.0, 2.0], &[2], true);
        let g = leaf(&[0.1], &[1], false);
        let result = diff_sgd_momentum_step(&[p], &[g], &[], 0.1, 0.9);
        assert!(result.is_err());
    }

    #[test]
    fn test_diff_sgd_momentum_maintains_autograd_chain() {
        // Verify gradients flow through the velocity computation.
        let theta = leaf(&[5.0], &[1], true);
        let g = leaf(&[0.5], &[1], false);
        let (adapted, _) =
            diff_sgd_momentum_step(std::slice::from_ref(&theta), &[g], &[], 0.1, 0.9).unwrap();
        let loss = sum(&adapted[0]).unwrap();
        loss.backward().unwrap();
        // d/d(theta) of (theta - 0.1*0.5) = 1
        let theta_grad = theta.grad().unwrap().unwrap();
        assert!((theta_grad.data().unwrap()[0] - 1.0).abs() < 1e-5);
    }
}
