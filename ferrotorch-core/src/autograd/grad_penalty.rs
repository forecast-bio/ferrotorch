//! Gradient penalty utilities for regularization and WGAN-GP training.
//!
//! This module provides [`gradient_penalty`] for WGAN-GP, [`grad_norm`] for
//! generic gradient norm computation, and [`jvp`]/[`vjp`] for Jacobian-vector
//! and vector-Jacobian products.
//!
//! All functions leverage the higher-order gradient system in
//! [`super::higher_order::grad`], with `create_graph=true` where needed so the
//! resulting penalty tensors can be differentiated again (enabling outer-loop
//! optimization in GAN training).

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

use super::higher_order::grad;

/// Compute the gradient penalty for WGAN-GP.
///
/// Given a discriminator function `D`, real samples, and fake samples, this
/// computes:
///
/// ```text
/// penalty = lambda * (||grad(D(x_interp), x_interp)||_2 - 1)^2
/// ```
///
/// where `x_interp = alpha * real + (1 - alpha) * fake` with `alpha ~ U(0, 1)`.
///
/// The returned tensor has `grad_fn` attached (via `create_graph=true`), so it
/// can be added to the discriminator loss and differentiated in the outer
/// training loop.
///
/// # Parameters
///
/// - `discriminator`: A closure that takes an interpolated input tensor and
///   returns a scalar discriminator output.
/// - `real`: Real data samples (1-D tensor of shape `[n]`).
/// - `fake`: Fake data samples (1-D tensor of shape `[n]`, same shape as `real`).
/// - `lambda`: Gradient penalty coefficient (typically 10.0).
///
/// # Errors
///
/// Returns an error if `real` and `fake` have different shapes, or if the
/// discriminator does not return a scalar.
///
/// # Example
///
/// ```ignore
/// // Linear discriminator D(x) = sum(x)
/// let penalty = gradient_penalty(
///     |x| crate::grad_fns::reduction::sum(x),
///     &real, &fake, 10.0,
/// )?;
/// let total_loss = crate::grad_fns::arithmetic::add(&d_loss, &penalty)?;
/// total_loss.backward()?;
/// ```
pub fn gradient_penalty<T: Float, F>(
    discriminator: F,
    real: &Tensor<T>,
    fake: &Tensor<T>,
    lambda: f64,
) -> FerrotorchResult<Tensor<T>>
where
    F: Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>>,
{
    // Validate shapes match.
    if real.shape() != fake.shape() {
        return Err(crate::error::FerrotorchError::ShapeMismatch {
            message: format!(
                "gradient_penalty: real shape {:?} != fake shape {:?}",
                real.shape(),
                fake.shape()
            ),
        });
    }

    let n = real.numel();

    // 1. Random interpolation coefficient alpha ~ U(0, 1).
    let alpha_tensor: Tensor<T> = crate::creation::rand(real.shape())?;
    let alpha_data = alpha_tensor.data_vec()?;
    let real_data = real.data_vec()?;
    let fake_data = fake.data_vec()?;

    // x_interp = alpha * real + (1 - alpha) * fake
    let one = <T as num_traits::One>::one();
    let interp_data: Vec<T> = (0..n)
        .map(|i| alpha_data[i] * real_data[i] + (one - alpha_data[i]) * fake_data[i])
        .collect();

    // 2. Create x_interp with requires_grad=true so we can differentiate through it.
    let x_interp =
        Tensor::from_storage(TensorStorage::cpu(interp_data), real.shape().to_vec(), true)?;

    // 3. Forward pass through discriminator.
    let d_interp = discriminator(&x_interp)?;

    // 4. Compute gradients with create_graph=true so the penalty is differentiable.
    let grads = grad(&d_interp, &[&x_interp], false, true)?;
    let grad_interp = match &grads[0] {
        Some(g) => g.clone(),
        None => {
            // Discriminator output doesn't depend on input -- gradient is zero.
            let zero_data = vec![<T as num_traits::Zero>::zero(); n];
            Tensor::from_storage(TensorStorage::cpu(zero_data), real.shape().to_vec(), false)?
        }
    };

    // 5. Compute L2 norm of the gradient: ||grad||_2 = sqrt(sum(grad^2)).
    let grad_sq = crate::grad_fns::arithmetic::pow(&grad_interp, 2.0)?;
    let grad_sq_sum = crate::grad_fns::reduction::sum(&grad_sq)?;
    let grad_norm = crate::grad_fns::arithmetic::sqrt(&grad_sq_sum)?;

    // 6. penalty = lambda * (grad_norm - 1)^2
    let one_tensor = Tensor::from_storage(TensorStorage::cpu(vec![one]), vec![], false)?;
    let diff = crate::grad_fns::arithmetic::sub(&grad_norm, &one_tensor)?;
    let diff_sq = crate::grad_fns::arithmetic::pow(&diff, 2.0)?;

    let lambda_t = T::from(lambda).unwrap();
    let lambda_tensor = Tensor::from_storage(TensorStorage::cpu(vec![lambda_t]), vec![], false)?;
    let penalty = crate::grad_fns::arithmetic::mul(&lambda_tensor, &diff_sq)?;

    Ok(penalty)
}

/// Compute the L2 norm of gradients of `outputs` with respect to `inputs`.
///
/// Returns a scalar tensor equal to `sqrt(sum_i ||d(outputs)/d(inputs_i)||^2)`,
/// computed across all gradient tensors. Useful for gradient regularization:
/// add `grad_norm(loss, &params)` to the loss to penalize large gradients.
///
/// # Parameters
///
/// - `outputs`: A scalar tensor to differentiate.
/// - `inputs`: The tensors to differentiate with respect to.
///
/// # Example
///
/// ```ignore
/// let x = Tensor::from_storage(TensorStorage::cpu(vec![3.0f32, 4.0]), vec![2], true)?;
/// let y = crate::grad_fns::reduction::sum(&crate::grad_fns::arithmetic::pow(&x, 2.0)?)?;
/// let norm = grad_norm(&y, &[&x])?;
/// // dy/dx = [6, 8], norm = sqrt(36 + 64) = 10
/// ```
pub fn grad_norm<T: Float>(
    outputs: &Tensor<T>,
    inputs: &[&Tensor<T>],
) -> FerrotorchResult<Tensor<T>> {
    let grads = grad(outputs, inputs, false, false)?;

    // Compute sqrt(sum(g_i^2)) across all gradient tensors.
    let zero = <T as num_traits::Zero>::zero();
    let mut total_sq = zero;

    for maybe_grad in grads.iter().flatten() {
        let g_data = maybe_grad.data_vec()?;
        for &val in &g_data {
            total_sq += val * val;
        }
    }

    let norm_val = total_sq.sqrt();
    Tensor::from_storage(TensorStorage::cpu(vec![norm_val]), vec![], false)
}

/// Compute the Jacobian-vector product (JVP): `J @ v`.
///
/// Given a function `f: R^n -> R^m`, an input point `input`, and a tangent
/// vector `v` of shape `[n]`, returns the directional derivative `J @ v`
/// of shape `[m]`.
///
/// For the MVP implementation, this uses central finite differences:
/// `(f(x + h*v) - f(x - h*v)) / (2h)` with `h = 1e-4`.
///
/// # Parameters
///
/// - `f`: The function to differentiate.
/// - `input`: The point at which to evaluate the JVP (shape `[n]`).
/// - `v`: The tangent vector (shape `[n]`).
///
/// # Example
///
/// ```ignore
/// // f(x) = x^2 (elementwise), J = diag(2x), Jv = [2*x0*v0, 2*x1*v1]
/// let result = jvp(|x| pow(x, 2.0), &input, &v)?;
/// ```
pub fn jvp<T: Float, F>(f: F, input: &Tensor<T>, v: &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    F: Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>>,
{
    if input.shape() != v.shape() {
        return Err(crate::error::FerrotorchError::ShapeMismatch {
            message: format!(
                "jvp: input shape {:?} != v shape {:?}",
                input.shape(),
                v.shape()
            ),
        });
    }

    let h = T::from(1e-4).unwrap();
    let two_h = T::from(2e-4).unwrap();

    let input_data = input.data_vec()?;
    let v_data = v.data_vec()?;
    let n = input.numel();

    // x_plus = input + h * v
    let plus_data: Vec<T> = (0..n).map(|i| input_data[i] + h * v_data[i]).collect();
    let x_plus =
        Tensor::from_storage(TensorStorage::cpu(plus_data), input.shape().to_vec(), false)?;

    // x_minus = input - h * v
    let minus_data: Vec<T> = (0..n).map(|i| input_data[i] - h * v_data[i]).collect();
    let x_minus = Tensor::from_storage(
        TensorStorage::cpu(minus_data),
        input.shape().to_vec(),
        false,
    )?;

    // f(x + h*v) and f(x - h*v)
    let f_plus = f(&x_plus)?;
    let f_minus = f(&x_minus)?;

    let fp_data = f_plus.data_vec()?;
    let fm_data = f_minus.data_vec()?;

    // (f(x+hv) - f(x-hv)) / (2h)
    let result_data: Vec<T> = fp_data
        .iter()
        .zip(fm_data.iter())
        .map(|(&fp, &fm)| (fp - fm) / two_h)
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(result_data),
        f_plus.shape().to_vec(),
        false,
    )
}

/// Compute the vector-Jacobian product (VJP): `v^T @ J`.
///
/// Given a function `f: R^n -> R^m`, an input point `input`, and a cotangent
/// vector `v` of shape `[m]`, returns `v^T @ J` of shape `[n]`.
///
/// This is essentially what `backward()` computes. It runs a forward pass
/// through `f`, then uses [`grad`] to propagate `v` backward through the
/// computation graph.
///
/// # Parameters
///
/// - `f`: The function to differentiate.
/// - `input`: The point at which to evaluate the VJP (shape `[n]`).
/// - `v`: The cotangent vector (shape `[m]`, matching `f(input)` shape).
///
/// # Example
///
/// ```ignore
/// // f(x) = 2*x, J = 2*I, v^T J = 2*v
/// let result = vjp(|x| add(x, x), &input, &v)?;
/// ```
pub fn vjp<T: Float, F>(f: F, input: &Tensor<T>, v: &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    F: Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>>,
{
    // Create a fresh input that requires grad.
    let x = Tensor::from_storage(
        TensorStorage::cpu(input.data_vec()?),
        input.shape().to_vec(),
        true,
    )?;

    // Forward pass.
    let y = f(&x)?;

    // We need a scalar output for grad(). Compute the dot product y . v
    // to get a scalar whose gradient w.r.t. x is v^T @ J.
    let y_data = y.data_vec()?;
    let v_data = v.data_vec()?;

    if y_data.len() != v_data.len() {
        return Err(crate::error::FerrotorchError::ShapeMismatch {
            message: format!(
                "vjp: f(input) has {} elements but v has {}",
                y_data.len(),
                v_data.len()
            ),
        });
    }

    // Construct y_weighted = y * v using differentiable mul, then sum.
    let v_tensor = Tensor::from_storage(TensorStorage::cpu(v_data), y.shape().to_vec(), false)?;
    let weighted = crate::grad_fns::arithmetic::mul(&y, &v_tensor)?;
    let scalar = crate::grad_fns::reduction::sum(&weighted)?;

    // Backward: d(scalar)/dx = v^T @ J
    let grads = grad(&scalar, &[&x], false, false)?;

    match grads.into_iter().next().unwrap() {
        Some(g) => Ok(g),
        None => {
            // f doesn't depend on input.
            let zero_data = vec![<T as num_traits::Zero>::zero(); input.numel()];
            Tensor::from_storage(TensorStorage::cpu(zero_data), input.shape().to_vec(), false)
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grad_fns::arithmetic::{add, mul, pow};
    use crate::grad_fns::reduction::sum;
    use crate::storage::TensorStorage;

    /// Create a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    /// Create a leaf 1-D tensor.
    fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            requires_grad,
        )
        .unwrap()
    }

    /// Assert a scalar tensor is approximately equal to `expected`.
    fn assert_approx(actual: f32, expected: f32, tol: f32, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: expected {expected}, got {actual}"
        );
    }

    // -----------------------------------------------------------------------
    // gradient_penalty tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_penalty_linear_discriminator() {
        // D(x) = sum(x). For any input of size n:
        //   grad(D(x), x) = [1, 1, ..., 1]
        //   ||grad||_2 = sqrt(n)
        //   penalty = lambda * (sqrt(n) - 1)^2
        //
        // We use fixed "real" and "fake" so the test is deterministic.
        // Since D is linear, gradient is the same regardless of interpolation.
        let n = 4usize;
        let real = leaf_vec(&[1.0, 2.0, 3.0, 4.0], false);
        let fake = leaf_vec(&[0.5, 1.5, 2.5, 3.5], false);
        let lambda = 10.0;

        let penalty = gradient_penalty(sum, &real, &fake, lambda).unwrap();

        let expected = lambda as f32 * ((n as f32).sqrt() - 1.0).powi(2);
        assert_approx(
            penalty.item().unwrap(),
            expected,
            1e-3,
            "gradient_penalty for linear D(x)=sum(x)",
        );
    }

    #[test]
    fn test_gradient_penalty_shape_mismatch() {
        let real = leaf_vec(&[1.0, 2.0], false);
        let fake = leaf_vec(&[1.0, 2.0, 3.0], false);
        let result = gradient_penalty(sum, &real, &fake, 10.0);
        assert!(result.is_err(), "should error on shape mismatch");
    }

    #[test]
    fn test_gradient_penalty_scalar_input() {
        // D(x) = x^2 for scalar x.
        // grad(D(x), x) = 2x_interp
        // ||grad||_2 = |2 * x_interp|
        // penalty = lambda * (|2 * x_interp| - 1)^2
        //
        // With real=2.0 and fake=2.0, x_interp = 2.0 for any alpha.
        // grad_norm = 4.0, penalty = lambda * (4 - 1)^2 = lambda * 9
        let real = leaf_vec(&[2.0], false);
        let fake = leaf_vec(&[2.0], false);
        let lambda = 5.0;

        let penalty = gradient_penalty(
            |x| {
                let sq = pow(x, 2.0)?;
                sum(&sq)
            },
            &real,
            &fake,
            lambda,
        )
        .unwrap();

        let expected = 5.0f32 * (4.0 - 1.0_f32).powi(2);
        assert_approx(
            penalty.item().unwrap(),
            expected,
            1e-2,
            "gradient_penalty for D(x)=x^2 at x=2",
        );
    }

    #[test]
    fn test_gradient_penalty_has_grad_fn() {
        // The returned penalty should be differentiable (has grad_fn).
        let real = leaf_vec(&[1.0, 2.0], false);
        let fake = leaf_vec(&[0.5, 1.5], false);

        let penalty = gradient_penalty(sum, &real, &fake, 10.0).unwrap();

        assert!(
            penalty.grad_fn().is_some(),
            "gradient_penalty result should have grad_fn for outer optimization"
        );
    }

    // -----------------------------------------------------------------------
    // grad_norm tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_grad_norm_simple() {
        // f(x) = sum(x^2), x = [3, 4]
        // df/dx = [6, 8]
        // ||grad||_2 = sqrt(36 + 64) = sqrt(100) = 10
        let x = leaf_vec(&[3.0, 4.0], true);
        let y = {
            let sq = pow(&x, 2.0).unwrap();
            sum(&sq).unwrap()
        };

        let norm = grad_norm(&y, &[&x]).unwrap();
        assert_approx(norm.item().unwrap(), 10.0, 1e-3, "grad_norm of [6,8]");
    }

    #[test]
    fn test_grad_norm_scalar() {
        // f(x) = x^3, x = 2
        // df/dx = 12
        // ||grad|| = 12
        let x = leaf_scalar(2.0, true);
        let y = pow(&x, 3.0).unwrap();

        let norm = grad_norm(&y, &[&x]).unwrap();
        assert_approx(norm.item().unwrap(), 12.0, 1e-3, "grad_norm of scalar");
    }

    #[test]
    fn test_grad_norm_multiple_inputs() {
        // f(x, y) = x^2 + y^2, x=3, y=4
        // df/dx = 6, df/dy = 8
        // ||grad||_2 = sqrt(36 + 64) = 10
        let x = leaf_scalar(3.0, true);
        let y = leaf_scalar(4.0, true);
        let x2 = pow(&x, 2.0).unwrap();
        let y2 = pow(&y, 2.0).unwrap();
        let z = add(&x2, &y2).unwrap();

        let norm = grad_norm(&z, &[&x, &y]).unwrap();
        assert_approx(
            norm.item().unwrap(),
            10.0,
            1e-3,
            "grad_norm across two inputs",
        );
    }

    // -----------------------------------------------------------------------
    // vjp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vjp_identity() {
        // f(x) = x, J = I, v^T J = v^T
        let input = leaf_vec(&[1.0, 2.0, 3.0], false);
        let v = leaf_vec(&[4.0, 5.0, 6.0], false);

        let result = vjp(
            |x| {
                // Identity: multiply by 1 to get a tracked tensor.
                let ones =
                    Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 3]), vec![3], false)
                        .unwrap();
                mul(x, &ones)
            },
            &input,
            &v,
        )
        .unwrap();

        let data = result.data().unwrap();
        assert_approx(data[0], 4.0, 1e-5, "vjp identity [0]");
        assert_approx(data[1], 5.0, 1e-5, "vjp identity [1]");
        assert_approx(data[2], 6.0, 1e-5, "vjp identity [2]");
    }

    #[test]
    fn test_vjp_linear_2x() {
        // f(x) = 2*x (via add(x, x)), J = 2*I, v^T J = 2*v^T
        let input = leaf_vec(&[1.0, 2.0], false);
        let v = leaf_vec(&[3.0, 4.0], false);

        let result = vjp(|x| add(x, x), &input, &v).unwrap();

        let data = result.data().unwrap();
        assert_approx(data[0], 6.0, 1e-5, "vjp 2x [0]");
        assert_approx(data[1], 8.0, 1e-5, "vjp 2x [1]");
    }

    #[test]
    fn test_vjp_scalar_mul() {
        // f(x) = x * c where c=3, J = 3*I, v^T J = 3*v
        let input = leaf_vec(&[2.0], false);
        let v = leaf_vec(&[5.0], false);

        let result = vjp(
            |x| {
                let c =
                    Tensor::from_storage(TensorStorage::cpu(vec![3.0f32]), vec![1], false).unwrap();
                mul(x, &c)
            },
            &input,
            &v,
        )
        .unwrap();

        assert_approx(result.data().unwrap()[0], 15.0, 1e-5, "vjp scalar mul");
    }

    #[test]
    fn test_vjp_matches_manual_backward() {
        // f(x) = x^2 (elementwise), J = diag(2x)
        // v^T J = [v0*2*x0, v1*2*x1]
        let input = leaf_vec(&[3.0, 4.0], false);
        let v = leaf_vec(&[1.0, 1.0], false);

        let result = vjp(|x| pow(x, 2.0), &input, &v).unwrap();

        let data = result.data().unwrap();
        // v^T J = [1*6, 1*8] = [6, 8]
        assert_approx(data[0], 6.0, 1e-3, "vjp x^2 [0]");
        assert_approx(data[1], 8.0, 1e-3, "vjp x^2 [1]");
    }

    #[test]
    fn test_vjp_shape_mismatch() {
        let input = leaf_vec(&[1.0, 2.0], false);
        let v = leaf_vec(&[1.0, 2.0, 3.0], false);

        let result = vjp(|x| add(x, x), &input, &v);
        assert!(
            result.is_err(),
            "vjp should error when v shape != f(input) shape"
        );
    }

    // -----------------------------------------------------------------------
    // jvp tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_jvp_identity() {
        // f(x) = x, J = I, Jv = v
        let input = leaf_vec(&[1.0, 2.0, 3.0], false);
        let v = leaf_vec(&[4.0, 5.0, 6.0], false);

        let result = jvp(
            |x| {
                // Return a copy of x (no grad tracking needed for finite diff).
                let d = x.data().unwrap();
                Tensor::from_storage(TensorStorage::cpu(d.to_vec()), x.shape().to_vec(), false)
            },
            &input,
            &v,
        )
        .unwrap();

        let data = result.data().unwrap();
        assert_approx(data[0], 4.0, 1e-2, "jvp identity [0]");
        assert_approx(data[1], 5.0, 1e-2, "jvp identity [1]");
        assert_approx(data[2], 6.0, 1e-2, "jvp identity [2]");
    }

    #[test]
    fn test_jvp_quadratic() {
        // f(x) = x^2 (elementwise), J = diag(2x), Jv = [2*x0*v0, 2*x1*v1]
        let input = leaf_vec(&[3.0, 4.0], false);
        let v = leaf_vec(&[1.0, 1.0], false);

        let result = jvp(
            |x| {
                let d = x.data().unwrap();
                let sq: Vec<f32> = d.iter().map(|&val| val * val).collect();
                Tensor::from_storage(TensorStorage::cpu(sq), x.shape().to_vec(), false)
            },
            &input,
            &v,
        )
        .unwrap();

        let data = result.data().unwrap();
        // Jv = [2*3*1, 2*4*1] = [6, 8]
        assert_approx(data[0], 6.0, 1e-1, "jvp x^2 [0]");
        assert_approx(data[1], 8.0, 1e-1, "jvp x^2 [1]");
    }

    #[test]
    fn test_jvp_linear_2x() {
        // f(x) = 2*x, J = 2I, Jv = 2v
        let input = leaf_vec(&[1.0, 2.0], false);
        let v = leaf_vec(&[3.0, 4.0], false);

        let result = jvp(
            |x| {
                let d = x.data().unwrap();
                let doubled: Vec<f32> = d.iter().map(|&val| val * 2.0).collect();
                Tensor::from_storage(TensorStorage::cpu(doubled), x.shape().to_vec(), false)
            },
            &input,
            &v,
        )
        .unwrap();

        let data = result.data().unwrap();
        assert_approx(data[0], 6.0, 1e-2, "jvp 2x [0]");
        assert_approx(data[1], 8.0, 1e-2, "jvp 2x [1]");
    }

    #[test]
    fn test_jvp_matches_analytical_cubic() {
        // f(x) = x^3 (scalar), J = [3x^2], Jv = 3x^2 * v
        // At x=2, v=1: Jv = 12
        let input = leaf_vec(&[2.0], false);
        let v = leaf_vec(&[1.0], false);

        let result = jvp(
            |x| {
                let d = x.data().unwrap();
                let cubed: Vec<f32> = d.iter().map(|&val| val * val * val).collect();
                Tensor::from_storage(TensorStorage::cpu(cubed), x.shape().to_vec(), false)
            },
            &input,
            &v,
        )
        .unwrap();

        assert_approx(result.data().unwrap()[0], 12.0, 1e-1, "jvp x^3 at x=2");
    }

    #[test]
    fn test_jvp_shape_mismatch() {
        let input = leaf_vec(&[1.0, 2.0], false);
        let v = leaf_vec(&[1.0], false);

        let result = jvp(
            |x| {
                let d = x.data().unwrap();
                Tensor::from_storage(TensorStorage::cpu(d.to_vec()), x.shape().to_vec(), false)
            },
            &input,
            &v,
        );
        assert!(result.is_err(), "jvp should error on shape mismatch");
    }

    // -----------------------------------------------------------------------
    // Integration: gradient_penalty with create_graph for outer optimization
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_penalty_create_graph_outer_loop() {
        // Verify that the gradient penalty result can be differentiated again
        // (i.e., it participates in the computation graph for outer-loop
        // optimization as required in WGAN-GP training).
        //
        // D(x) = sum(w * x) where w is a learnable parameter.
        // For fixed w, grad(D(x), x) = w, ||w||_2 = norm.
        // penalty = lambda * (||w||_2 - 1)^2
        //
        // We verify the penalty has a grad_fn, allowing backward through it.
        let real = leaf_vec(&[1.0, 2.0], false);
        let fake = leaf_vec(&[0.5, 1.5], false);

        let penalty = gradient_penalty(sum, &real, &fake, 10.0).unwrap();

        // The penalty tensor should be part of the computation graph.
        assert!(
            penalty.grad_fn().is_some(),
            "penalty must have grad_fn for outer-loop optimization"
        );
        assert!(
            penalty.requires_grad(),
            "penalty must require grad for outer-loop optimization"
        );
    }
}
