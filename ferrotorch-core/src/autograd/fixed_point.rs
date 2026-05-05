//! Fixed-point implicit differentiation for ferrotorch autograd.
//!
//! Given a fixed-point equation `x* = f(x*, p)` where `x*` is the fixed point
//! and `p` are parameters, the implicit function theorem gives:
//!
//! ```text
//! dx*/dp = (I - df/dx|_{x*})^{-1} @ (df/dp|_{x*})
//! ```
//!
//! This avoids unrolling through the entire iterative process (which can be
//! thousands of steps), making it memory-efficient for:
//!
//! - Deep Equilibrium Models (DEQ)
//! - Long-context RNNs (fixed point of the recurrence)
//! - Neural ODEs (fixed point of the flow)
//! - Neural Cellular Automata (fixed point of the update rule)
//!
//! The backward pass uses the Neumann series approximation:
//!
//! ```text
//! v = (I - J_x^T)^{-1} @ grad_output = sum_{k=0}^{K} (J_x^T)^k @ grad_output
//! ```
//!
//! which converges when the spectral radius of `J_x` is less than 1 (guaranteed
//! when `f` is contractive).

use std::fmt;
use std::sync::Arc;

use crate::autograd::higher_order::grad;
use crate::autograd::no_grad::no_grad;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Type alias for the fixed-point function f(x, params) -> x.
type FixedPointFn<T> =
    Arc<dyn Fn(&Tensor<T>, &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

/// Find a fixed point of `f` starting from `x0`, then compute its derivative
/// w.r.t. `params` using the implicit function theorem.
///
/// This is used for:
/// - Long-context RNNs (fixed point of the recurrence)
/// - Neural ODEs (fixed point of the flow)
/// - Neural Cellular Automata (fixed point of the update rule)
/// - Equilibrium models (DEQ)
///
/// # Arguments
/// - `f`: The function x_{n+1} = f(x_n, params). Must be contractive.
/// - `x0`: Initial guess.
/// - `params`: Parameters to differentiate w.r.t.
/// - `max_iter`: Maximum iterations to find the fixed point.
/// - `tol`: Convergence tolerance (stop when ||x_{n+1} - x_n|| < tol).
///
/// # Returns
/// The fixed point x* as a Tensor with grad_fn attached so that
/// backward() computes dx*/dp via the implicit function theorem.
///
/// # Examples
///
/// ```ignore
/// // f(x, a) = a * x, a = 0.5, x0 = 10
/// // Fixed point: x* = 0
/// let a = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], true)?;
/// let x0 = Tensor::from_storage(TensorStorage::cpu(vec![10.0f32]), vec![], false)?;
/// let x_star = fixed_point(
///     |x, p| {
///         // f(x, a) = a * x
///         crate::grad_fns::arithmetic::mul(x, p[0])
///     },
///     &x0,
///     &[&a],
///     1000,
///     1e-8,
/// )?;
/// ```
pub fn fixed_point<T, F>(
    f: F,
    x0: &Tensor<T>,
    params: &[&Tensor<T>],
    max_iter: usize,
    tol: f64,
) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>, &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    // 1. Find the fixed point by iteration (forward pass, no grad needed).
    let x_star = no_grad(|| -> FerrotorchResult<Tensor<T>> {
        let mut x = x0.clone();
        for _ in 0..max_iter {
            let x_next = f(&x, params)?;
            // Compute L1 norm of the difference.
            let x_data = x.data_vec()?;
            let x_next_data = x_next.data_vec()?;
            let norm: f64 = x_data
                .iter()
                .zip(x_next_data.iter())
                .map(|(&a, &b)| (a - b).to_f64().unwrap().abs())
                .sum();
            if norm < tol {
                return Ok::<Tensor<T>, FerrotorchError>(x_next);
            }
            x = x_next;
        }
        Ok(x) // Didn't converge but return best estimate.
    })?;

    // 2. If any parameter requires grad, attach a FixedPointBackward node
    //    that uses implicit differentiation via the Neumann series.
    if params.iter().any(|p| p.requires_grad()) {
        let x_star_data = x_star.data_vec()?;
        let x_star_shape = x_star.shape().to_vec();
        let storage = TensorStorage::cpu(x_star_data);

        // Clone params for storage in the backward node.
        let params_owned: Vec<Tensor<T>> = params.iter().map(|p| (*p).clone()).collect();

        Tensor::from_operation(
            storage,
            x_star_shape,
            Arc::new(FixedPointBackward {
                f_closure: Arc::new(f),
                x_star: x_star.clone(),
                params: params_owned,
                backward_max_iter: max_iter.min(50), // Cap backward iterations.
                backward_tol: tol,
            }),
        )
    } else {
        Ok(x_star)
    }
}

/// Backward node for fixed-point implicit differentiation.
///
/// Uses the Neumann series to solve the implicit derivative system:
///
/// ```text
/// (I - J_x^T) @ v = grad_output
/// v = sum_{k=0}^{K} (J_x^T)^k @ grad_output
/// ```
///
/// Then distributes `v` through `df/dp` to produce gradients for each parameter.
struct FixedPointBackward<T: Float> {
    /// The function f(x, params) whose fixed point was found.
    f_closure: FixedPointFn<T>,
    /// The fixed point x*.
    x_star: Tensor<T>,
    /// The parameters to differentiate w.r.t.
    params: Vec<Tensor<T>>,
    /// Maximum iterations for the Neumann series in the backward pass.
    backward_max_iter: usize,
    /// Convergence tolerance for the Neumann series.
    backward_tol: f64,
}

impl<T: Float> fmt::Debug for FixedPointBackward<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FixedPointBackward")
            .field("f_closure", &"<closure>")
            .field("x_star_shape", &self.x_star.shape())
            .field("num_params", &self.params.len())
            .field("backward_max_iter", &self.backward_max_iter)
            .field("backward_tol", &self.backward_tol)
            .finish()
    }
}

impl<T: Float> GradFn<T> for FixedPointBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let n = self.x_star.numel();
        let num_params = self.params.len();

        // Step 1: Solve (I - J_x^T) v = grad_output via the Neumann series.
        //
        // v_0 = grad_output
        // v_{k+1} = grad_output + J_x^T @ v_k
        //
        // This converges because f is contractive => spectral_radius(J_x) < 1.
        //
        // We compute J_x^T @ v via a VJP: if y = f(x, p), then
        // VJP(y, x, v) = J_x^T @ v = grad(y, x, grad_output=v).
        let go_data = grad_output.data_vec()?;
        let go_shape = grad_output.shape().to_vec();

        let mut v_data = go_data.clone();

        for _ in 0..self.backward_max_iter {
            // Create a fresh x* that requires grad so we can compute J_x^T @ v.
            let x_fresh = Tensor::from_storage(
                TensorStorage::cpu(self.x_star.data_vec()?),
                self.x_star.shape().to_vec(),
                true,
            )?;

            // Detached params (we only want the Jacobian w.r.t. x here).
            let params_detached: Vec<Tensor<T>> = self
                .params
                .iter()
                .map(|p| {
                    Tensor::from_storage(
                        TensorStorage::cpu(p.data_vec().unwrap()),
                        p.shape().to_vec(),
                        false,
                    )
                    .unwrap()
                })
                .collect();
            let params_ref: Vec<&Tensor<T>> = params_detached.iter().collect();

            // Evaluate f(x*, params) with grad tracking on x.
            let y = (self.f_closure)(&x_fresh, &params_ref)?;

            // Compute VJP: J_x^T @ v via grad(y, x, grad_output=v).
            // We need to make y scalar to use grad(), so we use a dot product:
            // L = sum(y * v), then grad(L, x) = J_x^T @ v.
            let v_tensor =
                Tensor::from_storage(TensorStorage::cpu(v_data.clone()), go_shape.clone(), false)?;
            let yv = elementwise_mul_sum(&y, &v_tensor)?;

            let grads = grad(&yv, &[&x_fresh], false, false)?;

            let jt_v = match &grads[0] {
                Some(g) => g.data_vec()?,
                None => vec![<T as num_traits::Zero>::zero(); n],
            };

            // v_new = grad_output + J_x^T @ v
            let mut v_new = Vec::with_capacity(n);
            let mut diff_norm: f64 = 0.0;
            for i in 0..n {
                let val =
                    T::from(go_data[i].to_f64().unwrap() + jt_v[i].to_f64().unwrap()).unwrap();
                diff_norm += (val.to_f64().unwrap() - v_data[i].to_f64().unwrap()).abs();
                v_new.push(val);
            }
            v_data = v_new;

            if diff_norm < self.backward_tol {
                break;
            }
        }

        // Step 2: Compute gradients for each parameter.
        //
        // For each param p_i, the gradient is:
        //   grad_p_i = J_{p_i}^T @ v = grad(f(x*, p), p_i, grad_output=v)
        //
        // We evaluate f(x*, params) with grad tracking on params, then
        // compute grad(L, params) where L = sum(y * v).

        // Create x* without grad (we don't need x gradients here).
        let x_detached = Tensor::from_storage(
            TensorStorage::cpu(self.x_star.data_vec()?),
            self.x_star.shape().to_vec(),
            false,
        )?;

        // Create params with grad enabled.
        let params_with_grad: Vec<Tensor<T>> = self
            .params
            .iter()
            .map(|p| {
                Tensor::from_storage(
                    TensorStorage::cpu(p.data_vec().unwrap()),
                    p.shape().to_vec(),
                    p.requires_grad(),
                )
                .unwrap()
            })
            .collect();
        let params_ref: Vec<&Tensor<T>> = params_with_grad.iter().collect();

        // Evaluate f(x*, params).
        let y = (self.f_closure)(&x_detached, &params_ref)?;

        // L = sum(y * v)
        let v_tensor = Tensor::from_storage(TensorStorage::cpu(v_data), go_shape, false)?;
        let loss = elementwise_mul_sum(&y, &v_tensor)?;

        // Compute grad(L, params).
        let grad_inputs: Vec<&Tensor<T>> = params_with_grad
            .iter()
            .filter(|p| p.requires_grad())
            .collect();

        let mut result: Vec<Option<Tensor<T>>> = Vec::with_capacity(num_params);

        if grad_inputs.is_empty() {
            for _ in 0..num_params {
                result.push(None);
            }
        } else {
            let param_grads = grad(&loss, &grad_inputs[..], false, false)?;

            // Map back: grad_inputs is a filtered subset; we need to map each
            // param in self.params to its gradient (or None if !requires_grad).
            let mut grad_idx = 0;
            for p in &params_with_grad {
                if p.requires_grad() {
                    result.push(param_grads[grad_idx].clone());
                    grad_idx += 1;
                } else {
                    result.push(None);
                }
            }
        }

        Ok(result)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        self.params.iter().collect()
    }

    fn name(&self) -> &'static str {
        "FixedPointBackward"
    }
}

/// Compute `sum(a * b)` as a scalar tensor, preserving the autograd graph.
///
/// This is equivalent to a dot product when both tensors are 1-D, and a
/// Frobenius inner product for higher-dimensional tensors.
fn elementwise_mul_sum<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let prod = crate::grad_fns::arithmetic::mul(a, b)?;
    crate::grad_fns::reduction::sum(&prod)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::graph::backward;
    use crate::storage::TensorStorage;

    /// Create a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    /// Assert a scalar tensor is approximately equal to `expected`.
    fn assert_approx(actual: f32, expected: f32, tol: f32, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: expected {expected}, got {actual}"
        );
    }

    // -----------------------------------------------------------------------
    // Basic fixed-point convergence tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_fixed_point_affine() {
        // f(x) = 0.5 * x + 1
        // Fixed point: x* = 0.5 * x* + 1 => 0.5 * x* = 1 => x* = 2
        let x0 = leaf_scalar(0.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        let x_star = fixed_point(
            |x, _params| {
                // f(x) = 0.5 * x + 1
                let half = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], false)?;
                let one = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![], false)?;
                let half_x = crate::grad_fns::arithmetic::mul(x, &half)?;
                crate::grad_fns::arithmetic::add(&half_x, &one)
            },
            &x0,
            &[&dummy_param],
            1000,
            1e-8,
        )
        .unwrap();

        assert_approx(x_star.item().unwrap(), 2.0, 1e-4, "fixed point of 0.5x + 1");
    }

    #[test]
    fn test_fixed_point_contractive_to_zero() {
        // f(x, a) = a * x with a = 0.5, starting from x = 10
        // Fixed point: x* = 0.5 * x* => x* = 0
        let x0 = leaf_scalar(10.0, false);
        let a = leaf_scalar(0.5, false);

        let x_star = fixed_point(
            |x, params| crate::grad_fns::arithmetic::mul(x, params[0]),
            &x0,
            &[&a],
            1000,
            1e-8,
        )
        .unwrap();

        assert_approx(x_star.item().unwrap(), 0.0, 1e-4, "fixed point of 0.5*x");
    }

    #[test]
    fn test_fixed_point_tolerance() {
        // f(x) = 0.5 * x + 1, fixed point x* = 2
        // With a loose tolerance, it should converge in fewer iterations.
        let x0 = leaf_scalar(0.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        let x_star = fixed_point(
            |x, _params| {
                let half = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], false)?;
                let one = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![], false)?;
                let half_x = crate::grad_fns::arithmetic::mul(x, &half)?;
                crate::grad_fns::arithmetic::add(&half_x, &one)
            },
            &x0,
            &[&dummy_param],
            1000,
            0.1, // Loose tolerance.
        )
        .unwrap();

        // Should be close to 2.0 but not exact.
        let val = x_star.item().unwrap();
        assert!(
            (val - 2.0).abs() < 0.2,
            "loose tolerance: expected near 2.0, got {val}"
        );
    }

    #[test]
    fn test_fixed_point_max_iter_reached() {
        // f(x) = 0.99 * x + 0.01, fixed point x* = 1
        // With very few iterations, it won't converge fully.
        let x0 = leaf_scalar(0.0, false);
        let dummy_param = leaf_scalar(1.0, false);

        let x_star = fixed_point(
            |x, _params| {
                let scale = Tensor::from_storage(TensorStorage::cpu(vec![0.99f32]), vec![], false)?;
                let bias = Tensor::from_storage(TensorStorage::cpu(vec![0.01f32]), vec![], false)?;
                let sx = crate::grad_fns::arithmetic::mul(x, &scale)?;
                crate::grad_fns::arithmetic::add(&sx, &bias)
            },
            &x0,
            &[&dummy_param],
            5, // Very few iterations.
            1e-10,
        )
        .unwrap();

        // Should return best estimate even though it didn't converge.
        let val = x_star.item().unwrap();
        assert!(val > 0.0, "should have made some progress from x0=0");
        assert!(val < 1.0, "should not have reached x*=1 in 5 iterations");
    }

    // -----------------------------------------------------------------------
    // Gradient tests via implicit differentiation
    // -----------------------------------------------------------------------

    #[test]
    fn test_fixed_point_gradient_affine() {
        // f(x, a) = a * x + (1 - a)
        // Fixed point: x* = a * x* + 1 - a => x*(1 - a) = 1 - a => x* = 1
        // dx*/da = 0 (the fixed point is always 1 regardless of a)
        //
        // Actually for a < 1, let's use a different formulation:
        // f(x, b) = 0.5 * x + b
        // Fixed point: x* = 0.5 * x* + b => x* = 2b
        // dx*/db = 2
        let x0 = leaf_scalar(0.0, false);
        let b = leaf_scalar(3.0, true);

        let x_star = fixed_point(
            |x, params| {
                let half = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], false)?;
                let half_x = crate::grad_fns::arithmetic::mul(x, &half)?;
                crate::grad_fns::arithmetic::add(&half_x, params[0])
            },
            &x0,
            &[&b],
            1000,
            1e-8,
        )
        .unwrap();

        // x* should be 2b = 6
        assert_approx(x_star.item().unwrap(), 6.0, 1e-3, "x* = 2b = 6");

        // Compute gradient: dx*/db should be 2.
        backward(&x_star).unwrap();
        let grad_b = b.grad().unwrap().unwrap();
        assert_approx(grad_b.item().unwrap(), 2.0, 0.2, "dx*/db = 2");
    }

    #[test]
    fn test_fixed_point_gradient_scaling() {
        // f(x, a) = a * x, starting from x0 = 10
        // Fixed point: x* = 0 for any |a| < 1
        // dx*/da = 0 (the fixed point is always 0)
        let x0 = leaf_scalar(10.0, false);
        let a = leaf_scalar(0.5, true);

        let x_star = fixed_point(
            |x, params| crate::grad_fns::arithmetic::mul(x, params[0]),
            &x0,
            &[&a],
            1000,
            1e-8,
        )
        .unwrap();

        // x* should be 0
        assert_approx(x_star.item().unwrap(), 0.0, 1e-3, "x* = 0");

        // Compute gradient: dx*/da = 0 since x* = 0 regardless of a.
        backward(&x_star).unwrap();
        let grad_a = a.grad().unwrap().unwrap();
        assert_approx(grad_a.item().unwrap(), 0.0, 0.1, "dx*/da = 0");
    }

    #[test]
    fn test_fixed_point_no_grad_params() {
        // If no parameter requires grad, no backward node is attached.
        let x0 = leaf_scalar(0.0, false);
        let b = leaf_scalar(3.0, false); // No grad.

        let x_star = fixed_point(
            |x, params| {
                let half = Tensor::from_storage(TensorStorage::cpu(vec![0.5f32]), vec![], false)?;
                let half_x = crate::grad_fns::arithmetic::mul(x, &half)?;
                crate::grad_fns::arithmetic::add(&half_x, params[0])
            },
            &x0,
            &[&b],
            1000,
            1e-8,
        )
        .unwrap();

        assert_approx(x_star.item().unwrap(), 6.0, 1e-3, "x* = 2b = 6");
        // No grad_fn should be attached.
        assert!(
            x_star.grad_fn().is_none(),
            "no grad_fn when params don't require grad"
        );
    }
}
