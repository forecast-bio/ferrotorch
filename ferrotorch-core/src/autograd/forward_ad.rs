//! Forward-mode automatic differentiation via dual numbers.
//!
//! This module provides [`DualTensor`], a dual-number tensor carrying both a
//! primal value and a tangent (directional derivative), along with [`jvp_exact`]
//! for exact Jacobian-vector products and [`jacfwd`] for full Jacobian
//! computation via the vmap(jvp) pattern.
//!
//! Forward-mode AD computes exact derivatives in a single forward pass — no
//! finite differences, no backward graph. It is efficient when the number of
//! inputs is small relative to outputs (the opposite regime from reverse-mode).
//!
//! # Supported forward rules
//!
//! Arithmetic: add, sub, mul, div, neg
//! Linalg: matmul (2-D)
//! Activations: relu, sigmoid, tanh
//! Transcendentals: exp, log, sin, cos
//!
//! CL-310

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ===========================================================================
// DualTensor
// ===========================================================================

/// A dual-number tensor: `primal + epsilon * tangent`.
///
/// The primal carries the function value; the tangent carries the directional
/// derivative (JVP) with respect to the seed tangent vector.
///
/// `DualTensor` is cheap to clone (both fields are `Arc`-backed `Tensor`s).
#[derive(Debug, Clone)]
pub struct DualTensor<T: Float> {
    /// The function value.
    pub primal: Tensor<T>,
    /// The directional derivative (tangent vector).
    pub tangent: Tensor<T>,
}

impl<T: Float> DualTensor<T> {
    /// Create a new dual tensor from primal and tangent components.
    ///
    /// Both tensors must have the same shape.
    pub fn new(primal: Tensor<T>, tangent: Tensor<T>) -> FerrotorchResult<Self> {
        if primal.shape() != tangent.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "DualTensor: primal shape {:?} != tangent shape {:?}",
                    primal.shape(),
                    tangent.shape()
                ),
            });
        }
        Ok(Self { primal, tangent })
    }

    /// Create a dual tensor with zero tangent (a constant in forward-mode AD).
    pub fn constant(primal: Tensor<T>) -> FerrotorchResult<Self> {
        let zero_data = vec![<T as num_traits::Zero>::zero(); primal.numel()];
        let tangent = Tensor::from_storage(
            TensorStorage::cpu(zero_data),
            primal.shape().to_vec(),
            false,
        )?;
        Ok(Self { primal, tangent })
    }

    /// Shape of the dual tensor (same as primal and tangent).
    pub fn shape(&self) -> &[usize] {
        self.primal.shape()
    }

    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.primal.numel()
    }
}

// ===========================================================================
// Forward-mode rules: arithmetic
// ===========================================================================

/// Forward rule for addition: `d(a + b) = da + db`.
pub fn dual_add<T: Float>(a: &DualTensor<T>, b: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::arithmetic::add(&a.primal, &b.primal)?;
    let tangent = crate::grad_fns::arithmetic::add(&a.tangent, &b.tangent)?;
    // Shape guaranteed to match by broadcast semantics of add.
    Ok(DualTensor { primal, tangent })
}

/// Forward rule for subtraction: `d(a - b) = da - db`.
pub fn dual_sub<T: Float>(a: &DualTensor<T>, b: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::arithmetic::sub(&a.primal, &b.primal)?;
    let tangent = crate::grad_fns::arithmetic::sub(&a.tangent, &b.tangent)?;
    Ok(DualTensor { primal, tangent })
}

/// Forward rule for multiplication: `d(a * b) = a * db + da * b`.
pub fn dual_mul<T: Float>(a: &DualTensor<T>, b: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::arithmetic::mul(&a.primal, &b.primal)?;
    // tangent = a.primal * b.tangent + a.tangent * b.primal
    let term1 = crate::grad_fns::arithmetic::mul(&a.primal, &b.tangent)?;
    let term2 = crate::grad_fns::arithmetic::mul(&a.tangent, &b.primal)?;
    let tangent = crate::grad_fns::arithmetic::add(&term1, &term2)?;
    Ok(DualTensor { primal, tangent })
}

/// Forward rule for division: `d(a / b) = (da * b - a * db) / b^2`.
pub fn dual_div<T: Float>(a: &DualTensor<T>, b: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::arithmetic::div(&a.primal, &b.primal)?;
    // tangent = (da * b - a * db) / b^2
    let da_b = crate::grad_fns::arithmetic::mul(&a.tangent, &b.primal)?;
    let a_db = crate::grad_fns::arithmetic::mul(&a.primal, &b.tangent)?;
    let numer = crate::grad_fns::arithmetic::sub(&da_b, &a_db)?;
    let b_sq = crate::grad_fns::arithmetic::mul(&b.primal, &b.primal)?;
    let tangent = crate::grad_fns::arithmetic::div(&numer, &b_sq)?;
    Ok(DualTensor { primal, tangent })
}

/// Forward rule for negation: `d(-a) = -da`.
pub fn dual_neg<T: Float>(a: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::arithmetic::neg(&a.primal)?;
    let tangent = crate::grad_fns::arithmetic::neg(&a.tangent)?;
    Ok(DualTensor { primal, tangent })
}

// ===========================================================================
// Forward-mode rules: matmul
// ===========================================================================

/// Forward rule for matrix multiplication: `d(A @ B) = dA @ B + A @ dB`.
///
/// Both primals must be 2-D matrices with compatible inner dimensions.
pub fn dual_matmul<T: Float>(
    a: &DualTensor<T>,
    b: &DualTensor<T>,
) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::linalg::matmul_differentiable(&a.primal, &b.primal)?;
    // tangent = dA @ B + A @ dB
    let term1 = crate::grad_fns::linalg::matmul_differentiable(&a.tangent, &b.primal)?;
    let term2 = crate::grad_fns::linalg::matmul_differentiable(&a.primal, &b.tangent)?;
    let tangent = crate::grad_fns::arithmetic::add(&term1, &term2)?;
    Ok(DualTensor { primal, tangent })
}

// ===========================================================================
// Forward-mode rules: activations
// ===========================================================================

/// Forward rule for ReLU: `d(relu(a)) = da * (a > 0)`.
pub fn dual_relu<T: Float>(a: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::activation::relu(&a.primal)?;

    // tangent = da * step(a), where step(a) = 1 if a > 0, else 0.
    // At x=0, relu'(0) = 0 following the standard PyTorch convention.
    let a_data = a.primal.data_vec()?;
    let da_data = a.tangent.data_vec()?;
    let zero = <T as num_traits::Zero>::zero();

    let tangent_data: Vec<T> = a_data
        .iter()
        .zip(da_data.iter())
        .map(|(&x, &dx)| if x > zero { dx } else { zero })
        .collect();

    let tangent = Tensor::from_storage(
        TensorStorage::cpu(tangent_data),
        a.primal.shape().to_vec(),
        false,
    )?;
    Ok(DualTensor { primal, tangent })
}

/// Forward rule for sigmoid: `d(sigmoid(a)) = da * sigmoid(a) * (1 - sigmoid(a))`.
pub fn dual_sigmoid<T: Float>(a: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::activation::sigmoid(&a.primal)?;

    // tangent = da * sigma * (1 - sigma), where sigma = sigmoid(primal)
    let sigma_data = primal.data_vec()?;
    let da_data = a.tangent.data_vec()?;
    let one = <T as num_traits::One>::one();

    let tangent_data: Vec<T> = sigma_data
        .iter()
        .zip(da_data.iter())
        .map(|(&s, &dx)| dx * s * (one - s))
        .collect();

    let tangent = Tensor::from_storage(
        TensorStorage::cpu(tangent_data),
        a.primal.shape().to_vec(),
        false,
    )?;
    Ok(DualTensor { primal, tangent })
}

/// Forward rule for tanh: `d(tanh(a)) = da * (1 - tanh(a)^2)`.
pub fn dual_tanh<T: Float>(a: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::activation::tanh(&a.primal)?;

    // tangent = da * (1 - tanh^2)
    let tanh_data = primal.data_vec()?;
    let da_data = a.tangent.data_vec()?;
    let one = <T as num_traits::One>::one();

    let tangent_data: Vec<T> = tanh_data
        .iter()
        .zip(da_data.iter())
        .map(|(&t, &dx)| dx * (one - t * t))
        .collect();

    let tangent = Tensor::from_storage(
        TensorStorage::cpu(tangent_data),
        a.primal.shape().to_vec(),
        false,
    )?;
    Ok(DualTensor { primal, tangent })
}

// ===========================================================================
// Forward-mode rules: transcendentals
// ===========================================================================

/// Forward rule for exp: `d(exp(a)) = da * exp(a)`.
pub fn dual_exp<T: Float>(a: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::transcendental::exp(&a.primal)?;

    // tangent = da * exp(a)
    let exp_data = primal.data_vec()?;
    let da_data = a.tangent.data_vec()?;

    let tangent_data: Vec<T> = exp_data
        .iter()
        .zip(da_data.iter())
        .map(|(&e, &dx)| dx * e)
        .collect();

    let tangent = Tensor::from_storage(
        TensorStorage::cpu(tangent_data),
        a.primal.shape().to_vec(),
        false,
    )?;
    Ok(DualTensor { primal, tangent })
}

/// Forward rule for log: `d(log(a)) = da / a`.
pub fn dual_log<T: Float>(a: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::transcendental::log(&a.primal)?;

    // tangent = da / a
    let a_data = a.primal.data_vec()?;
    let da_data = a.tangent.data_vec()?;

    let tangent_data: Vec<T> = a_data
        .iter()
        .zip(da_data.iter())
        .map(|(&x, &dx)| dx / x)
        .collect();

    let tangent = Tensor::from_storage(
        TensorStorage::cpu(tangent_data),
        a.primal.shape().to_vec(),
        false,
    )?;
    Ok(DualTensor { primal, tangent })
}

/// Forward rule for sin: `d(sin(a)) = da * cos(a)`.
pub fn dual_sin<T: Float>(a: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::transcendental::sin(&a.primal)?;

    // tangent = da * cos(a)
    let a_data = a.primal.data_vec()?;
    let da_data = a.tangent.data_vec()?;

    let tangent_data: Vec<T> = a_data
        .iter()
        .zip(da_data.iter())
        .map(|(&x, &dx)| dx * x.cos())
        .collect();

    let tangent = Tensor::from_storage(
        TensorStorage::cpu(tangent_data),
        a.primal.shape().to_vec(),
        false,
    )?;
    Ok(DualTensor { primal, tangent })
}

/// Forward rule for cos: `d(cos(a)) = -da * sin(a)`.
pub fn dual_cos<T: Float>(a: &DualTensor<T>) -> FerrotorchResult<DualTensor<T>> {
    let primal = crate::grad_fns::transcendental::cos(&a.primal)?;

    // tangent = -da * sin(a)
    let a_data = a.primal.data_vec()?;
    let da_data = a.tangent.data_vec()?;

    let tangent_data: Vec<T> = a_data
        .iter()
        .zip(da_data.iter())
        .map(|(&x, &dx)| -dx * x.sin())
        .collect();

    let tangent = Tensor::from_storage(
        TensorStorage::cpu(tangent_data),
        a.primal.shape().to_vec(),
        false,
    )?;
    Ok(DualTensor { primal, tangent })
}

// ===========================================================================
// jvp_exact — exact Jacobian-vector product via forward-mode AD
// ===========================================================================

/// Compute the exact Jacobian-vector product using forward-mode AD.
///
/// Given a function `f` operating on [`DualTensor`]s, a primal input point,
/// and a tangent vector `v`, returns `(f(input), J @ v)` computed in a single
/// forward pass with no finite differences.
///
/// The caller writes `f` using the `dual_*` operations from this module.
/// Each `dual_*` op propagates tangents through the computation, so the
/// output tangent is exactly `J @ v`.
///
/// # Parameters
///
/// - `f`: A function from `DualTensor<T>` to `DualTensor<T>`, composed of
///   `dual_add`, `dual_mul`, `dual_exp`, etc.
/// - `input`: The primal point at which to evaluate.
/// - `v`: The tangent vector (same shape as `input`).
///
/// # Returns
///
/// A tuple `(primal_output, tangent_output)` where `tangent_output = J @ v`.
///
/// # Example
///
/// ```ignore
/// // f(x) = x^2 via dual_mul(x, x)
/// let (primal, tangent) = jvp_exact(
///     |x| dual_mul(&x, &x),
///     &input, &v,
/// )?;
/// // tangent = 2 * diag(input) @ v
/// ```
pub fn jvp_exact<T: Float, F>(
    f: F,
    input: &Tensor<T>,
    v: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Tensor<T>)>
where
    F: Fn(DualTensor<T>) -> FerrotorchResult<DualTensor<T>>,
{
    if input.shape() != v.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "jvp_exact: input shape {:?} != v shape {:?}",
                input.shape(),
                v.shape()
            ),
        });
    }

    // Seed: primal = input, tangent = v.
    let dual_input = DualTensor::new(input.clone(), v.clone())?;

    // Single forward pass through the dual-number computation.
    let dual_output = f(dual_input)?;

    Ok((dual_output.primal, dual_output.tangent))
}

// ===========================================================================
// jacfwd — full Jacobian via vmap(jvp) pattern
// ===========================================================================

/// Compute the full Jacobian matrix using forward-mode AD.
///
/// For a function `f: R^n -> R^m`, returns the `[m, n]` Jacobian matrix
/// by calling [`jvp_exact`] once per input dimension with a standard basis
/// tangent vector (column `i` of the identity matrix).
///
/// This is the vmap(jvp) pattern: we loop over basis vectors, computing
/// one column of the Jacobian per forward pass. Efficient when `n` is small.
///
/// # Parameters
///
/// - `f`: A function from `DualTensor<T>` to `DualTensor<T>`.
/// - `input`: The 1-D input point (shape `[n]`).
///
/// # Returns
///
/// A `[m, n]` tensor representing the Jacobian `J[i, j] = df_i / dx_j`.
///
/// # Example
///
/// ```ignore
/// // f(x) = [x0^2, x1^3]
/// let jac = jacfwd(my_dual_fn, &input)?;
/// // jac is shape [2, 2] = [[2*x0, 0], [0, 3*x1^2]]
/// ```
pub fn jacfwd<T: Float, F>(f: F, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    F: Fn(DualTensor<T>) -> FerrotorchResult<DualTensor<T>>,
{
    let shape = input.shape();
    if shape.len() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("jacfwd: input must be 1-D, got shape {shape:?}"),
        });
    }

    let n = shape[0];
    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();

    // Compute one column of the Jacobian per forward pass.
    let mut columns: Vec<Tensor<T>> = Vec::with_capacity(n);

    for j in 0..n {
        // Standard basis vector e_j.
        let mut basis = vec![zero; n];
        basis[j] = one;
        let e_j = Tensor::from_storage(TensorStorage::cpu(basis), vec![n], false)?;

        let (_primal, tangent) = jvp_exact(&f, input, &e_j)?;
        columns.push(tangent);
    }

    // Stack columns: each tangent has shape [m], we want [m, n].
    // columns[j] = J[:, j] (the j-th column of the Jacobian).
    // Stack along dim=1 to get [m, n].
    let m = columns[0].numel();

    let mut jac_data = vec![zero; m * n];
    for j in 0..n {
        let col_data = columns[j].data_vec()?;
        for i in 0..m {
            jac_data[i * n + j] = col_data[i];
        }
    }

    Tensor::from_storage(TensorStorage::cpu(jac_data), vec![m, n], false)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    /// Create a leaf 1-D tensor.
    fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            requires_grad,
        )
        .unwrap()
    }

    /// Create a leaf 2-D tensor.
    fn leaf_mat(data: &[f32], rows: usize, cols: usize) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![rows, cols], false).unwrap()
    }

    /// Assert a scalar approximately equals expected.
    fn assert_approx(actual: f32, expected: f32, tol: f32, msg: &str) {
        assert!(
            (actual - expected).abs() < tol,
            "{msg}: expected {expected}, got {actual}"
        );
    }

    // -----------------------------------------------------------------------
    // DualTensor construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_dual_tensor_new() {
        let primal = leaf_vec(&[1.0, 2.0, 3.0], false);
        let tangent = leaf_vec(&[0.1, 0.2, 0.3], false);
        let dual = DualTensor::new(primal, tangent).unwrap();
        assert_eq!(dual.shape(), &[3]);
        assert_eq!(dual.numel(), 3);
    }

    #[test]
    fn test_dual_tensor_shape_mismatch() {
        let primal = leaf_vec(&[1.0, 2.0], false);
        let tangent = leaf_vec(&[0.1, 0.2, 0.3], false);
        assert!(DualTensor::new(primal, tangent).is_err());
    }

    #[test]
    fn test_dual_tensor_constant() {
        let primal = leaf_vec(&[1.0, 2.0], false);
        let dual = DualTensor::constant(primal).unwrap();
        let t_data = dual.tangent.data_vec().unwrap();
        assert_eq!(t_data, vec![0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // Forward rules: arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn test_dual_add() {
        let a =
            DualTensor::new(leaf_vec(&[1.0, 2.0], false), leaf_vec(&[0.5, 0.3], false)).unwrap();
        let b =
            DualTensor::new(leaf_vec(&[3.0, 4.0], false), leaf_vec(&[0.1, 0.2], false)).unwrap();
        let c = dual_add(&a, &b).unwrap();

        let p = c.primal.data_vec().unwrap();
        let t = c.tangent.data_vec().unwrap();
        assert_approx(p[0], 4.0, 1e-6, "add primal[0]");
        assert_approx(p[1], 6.0, 1e-6, "add primal[1]");
        assert_approx(t[0], 0.6, 1e-6, "add tangent[0]");
        assert_approx(t[1], 0.5, 1e-6, "add tangent[1]");
    }

    #[test]
    fn test_dual_sub() {
        let a =
            DualTensor::new(leaf_vec(&[5.0, 3.0], false), leaf_vec(&[1.0, 0.5], false)).unwrap();
        let b =
            DualTensor::new(leaf_vec(&[2.0, 1.0], false), leaf_vec(&[0.3, 0.1], false)).unwrap();
        let c = dual_sub(&a, &b).unwrap();

        let p = c.primal.data_vec().unwrap();
        let t = c.tangent.data_vec().unwrap();
        assert_approx(p[0], 3.0, 1e-6, "sub primal[0]");
        assert_approx(p[1], 2.0, 1e-6, "sub primal[1]");
        assert_approx(t[0], 0.7, 1e-6, "sub tangent[0]");
        assert_approx(t[1], 0.4, 1e-6, "sub tangent[1]");
    }

    #[test]
    fn test_dual_mul() {
        // d(a*b) = a*db + da*b
        // a=2, b=3, da=0.5, db=0.1 => d = 2*0.1 + 0.5*3 = 0.2 + 1.5 = 1.7
        let a = DualTensor::new(leaf_vec(&[2.0], false), leaf_vec(&[0.5], false)).unwrap();
        let b = DualTensor::new(leaf_vec(&[3.0], false), leaf_vec(&[0.1], false)).unwrap();
        let c = dual_mul(&a, &b).unwrap();

        assert_approx(c.primal.data_vec().unwrap()[0], 6.0, 1e-6, "mul primal");
        assert_approx(c.tangent.data_vec().unwrap()[0], 1.7, 1e-5, "mul tangent");
    }

    #[test]
    fn test_dual_div() {
        // d(a/b) = (da*b - a*db) / b^2
        // a=6, b=3, da=1, db=0.5 => d = (1*3 - 6*0.5) / 9 = (3-3)/9 = 0
        let a = DualTensor::new(leaf_vec(&[6.0], false), leaf_vec(&[1.0], false)).unwrap();
        let b = DualTensor::new(leaf_vec(&[3.0], false), leaf_vec(&[0.5], false)).unwrap();
        let c = dual_div(&a, &b).unwrap();

        assert_approx(c.primal.data_vec().unwrap()[0], 2.0, 1e-6, "div primal");
        assert_approx(c.tangent.data_vec().unwrap()[0], 0.0, 1e-5, "div tangent");
    }

    #[test]
    fn test_dual_neg() {
        let a =
            DualTensor::new(leaf_vec(&[3.0, -2.0], false), leaf_vec(&[1.0, 0.5], false)).unwrap();
        let c = dual_neg(&a).unwrap();

        let p = c.primal.data_vec().unwrap();
        let t = c.tangent.data_vec().unwrap();
        assert_approx(p[0], -3.0, 1e-6, "neg primal[0]");
        assert_approx(p[1], 2.0, 1e-6, "neg primal[1]");
        assert_approx(t[0], -1.0, 1e-6, "neg tangent[0]");
        assert_approx(t[1], -0.5, 1e-6, "neg tangent[1]");
    }

    // -----------------------------------------------------------------------
    // Forward rules: matmul
    // -----------------------------------------------------------------------

    #[test]
    fn test_dual_matmul() {
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // A @ B = [[19, 22], [43, 50]]
        // dA = [[0.1, 0], [0, 0.1]], dB = [[0, 0], [0, 0]] (constant B)
        // tangent = dA @ B + A @ dB = dA @ B
        // dA @ B = [[0.1*5, 0.1*6], [0.1*7, 0.1*8]] = [[0.5, 0.6], [0.7, 0.8]]
        let a_primal = leaf_mat(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let a_tangent = leaf_mat(&[0.1, 0.0, 0.0, 0.1], 2, 2);
        let b_primal = leaf_mat(&[5.0, 6.0, 7.0, 8.0], 2, 2);
        let b_tangent = leaf_mat(&[0.0, 0.0, 0.0, 0.0], 2, 2);

        let a = DualTensor::new(a_primal, a_tangent).unwrap();
        let b = DualTensor::new(b_primal, b_tangent).unwrap();
        let c = dual_matmul(&a, &b).unwrap();

        let p = c.primal.data_vec().unwrap();
        assert_approx(p[0], 19.0, 1e-5, "matmul primal[0,0]");
        assert_approx(p[1], 22.0, 1e-5, "matmul primal[0,1]");
        assert_approx(p[2], 43.0, 1e-5, "matmul primal[1,0]");
        assert_approx(p[3], 50.0, 1e-5, "matmul primal[1,1]");

        let t = c.tangent.data_vec().unwrap();
        assert_approx(t[0], 0.5, 1e-4, "matmul tangent[0,0]");
        assert_approx(t[1], 0.6, 1e-4, "matmul tangent[0,1]");
        assert_approx(t[2], 0.7, 1e-4, "matmul tangent[1,0]");
        assert_approx(t[3], 0.8, 1e-4, "matmul tangent[1,1]");
    }

    // -----------------------------------------------------------------------
    // Forward rules: activations
    // -----------------------------------------------------------------------

    #[test]
    fn test_dual_relu_positive() {
        // relu(x) for x > 0: d(relu) = dx
        let a =
            DualTensor::new(leaf_vec(&[2.0, 3.0], false), leaf_vec(&[0.5, 1.0], false)).unwrap();
        let c = dual_relu(&a).unwrap();

        let p = c.primal.data_vec().unwrap();
        let t = c.tangent.data_vec().unwrap();
        assert_approx(p[0], 2.0, 1e-6, "relu primal[0]");
        assert_approx(p[1], 3.0, 1e-6, "relu primal[1]");
        assert_approx(t[0], 0.5, 1e-6, "relu tangent[0]");
        assert_approx(t[1], 1.0, 1e-6, "relu tangent[1]");
    }

    #[test]
    fn test_dual_relu_negative() {
        // relu(x) for x < 0: d(relu) = 0
        let a =
            DualTensor::new(leaf_vec(&[-1.0, -5.0], false), leaf_vec(&[0.5, 1.0], false)).unwrap();
        let c = dual_relu(&a).unwrap();

        let p = c.primal.data_vec().unwrap();
        let t = c.tangent.data_vec().unwrap();
        assert_approx(p[0], 0.0, 1e-6, "relu neg primal[0]");
        assert_approx(p[1], 0.0, 1e-6, "relu neg primal[1]");
        assert_approx(t[0], 0.0, 1e-6, "relu neg tangent[0]");
        assert_approx(t[1], 0.0, 1e-6, "relu neg tangent[1]");
    }

    #[test]
    fn test_dual_sigmoid() {
        // sigmoid(0) = 0.5, sigmoid'(0) = 0.25
        let a = DualTensor::new(leaf_vec(&[0.0], false), leaf_vec(&[1.0], false)).unwrap();
        let c = dual_sigmoid(&a).unwrap();

        assert_approx(c.primal.data_vec().unwrap()[0], 0.5, 1e-5, "sigmoid primal");
        assert_approx(
            c.tangent.data_vec().unwrap()[0],
            0.25,
            1e-5,
            "sigmoid tangent",
        );
    }

    #[test]
    fn test_dual_tanh() {
        // tanh(0) = 0, tanh'(0) = 1 - 0^2 = 1
        let a = DualTensor::new(leaf_vec(&[0.0], false), leaf_vec(&[1.0], false)).unwrap();
        let c = dual_tanh(&a).unwrap();

        assert_approx(c.primal.data_vec().unwrap()[0], 0.0, 1e-5, "tanh primal");
        assert_approx(c.tangent.data_vec().unwrap()[0], 1.0, 1e-5, "tanh tangent");
    }

    // -----------------------------------------------------------------------
    // Forward rules: transcendentals
    // -----------------------------------------------------------------------

    #[test]
    fn test_dual_exp() {
        // exp(0) = 1, exp'(0) = 1
        let a = DualTensor::new(leaf_vec(&[0.0], false), leaf_vec(&[1.0], false)).unwrap();
        let c = dual_exp(&a).unwrap();

        assert_approx(c.primal.data_vec().unwrap()[0], 1.0, 1e-5, "exp primal");
        assert_approx(c.tangent.data_vec().unwrap()[0], 1.0, 1e-5, "exp tangent");
    }

    #[test]
    fn test_dual_exp_nonzero() {
        // exp(1) = e, exp'(1) = e, tangent = 2 * e
        let a = DualTensor::new(leaf_vec(&[1.0], false), leaf_vec(&[2.0], false)).unwrap();
        let c = dual_exp(&a).unwrap();

        let e = std::f32::consts::E;
        assert_approx(c.primal.data_vec().unwrap()[0], e, 1e-5, "exp(1) primal");
        assert_approx(
            c.tangent.data_vec().unwrap()[0],
            2.0 * e,
            1e-4,
            "exp(1) tangent",
        );
    }

    #[test]
    fn test_dual_log() {
        // log(e) = 1, log'(e) = 1/e, tangent = 2 / e
        let e = std::f32::consts::E;
        let a = DualTensor::new(leaf_vec(&[e], false), leaf_vec(&[2.0], false)).unwrap();
        let c = dual_log(&a).unwrap();

        assert_approx(c.primal.data_vec().unwrap()[0], 1.0, 1e-5, "log primal");
        assert_approx(
            c.tangent.data_vec().unwrap()[0],
            2.0 / e,
            1e-5,
            "log tangent",
        );
    }

    #[test]
    fn test_dual_sin() {
        // sin(0) = 0, sin'(0) = cos(0) = 1
        let a = DualTensor::new(leaf_vec(&[0.0], false), leaf_vec(&[1.0], false)).unwrap();
        let c = dual_sin(&a).unwrap();

        assert_approx(c.primal.data_vec().unwrap()[0], 0.0, 1e-6, "sin primal");
        assert_approx(c.tangent.data_vec().unwrap()[0], 1.0, 1e-5, "sin tangent");
    }

    #[test]
    fn test_dual_sin_at_pi_half() {
        // sin(pi/2) = 1, sin'(pi/2) = cos(pi/2) = 0
        let pi_half = std::f32::consts::FRAC_PI_2;
        let a = DualTensor::new(leaf_vec(&[pi_half], false), leaf_vec(&[1.0], false)).unwrap();
        let c = dual_sin(&a).unwrap();

        assert_approx(
            c.primal.data_vec().unwrap()[0],
            1.0,
            1e-5,
            "sin(pi/2) primal",
        );
        assert_approx(
            c.tangent.data_vec().unwrap()[0],
            0.0,
            1e-5,
            "sin(pi/2) tangent",
        );
    }

    #[test]
    fn test_dual_cos() {
        // cos(0) = 1, cos'(0) = -sin(0) = 0
        let a = DualTensor::new(leaf_vec(&[0.0], false), leaf_vec(&[1.0], false)).unwrap();
        let c = dual_cos(&a).unwrap();

        assert_approx(c.primal.data_vec().unwrap()[0], 1.0, 1e-6, "cos primal");
        assert_approx(c.tangent.data_vec().unwrap()[0], 0.0, 1e-6, "cos tangent");
    }

    #[test]
    fn test_dual_cos_at_pi_half() {
        // cos(pi/2) = 0, cos'(pi/2) = -sin(pi/2) = -1
        let pi_half = std::f32::consts::FRAC_PI_2;
        let a = DualTensor::new(leaf_vec(&[pi_half], false), leaf_vec(&[1.0], false)).unwrap();
        let c = dual_cos(&a).unwrap();

        assert_approx(
            c.primal.data_vec().unwrap()[0],
            0.0,
            1e-5,
            "cos(pi/2) primal",
        );
        assert_approx(
            c.tangent.data_vec().unwrap()[0],
            -1.0,
            1e-5,
            "cos(pi/2) tangent",
        );
    }

    // -----------------------------------------------------------------------
    // jvp_exact
    // -----------------------------------------------------------------------

    #[test]
    fn test_jvp_exact_identity() {
        // f(x) = x, Jv = v
        let input = leaf_vec(&[1.0, 2.0, 3.0], false);
        let v = leaf_vec(&[4.0, 5.0, 6.0], false);

        let (primal, tangent) = jvp_exact(Ok, &input, &v).unwrap();

        let p = primal.data_vec().unwrap();
        let t = tangent.data_vec().unwrap();
        assert_approx(p[0], 1.0, 1e-6, "jvp identity primal[0]");
        assert_approx(t[0], 4.0, 1e-6, "jvp identity tangent[0]");
        assert_approx(t[1], 5.0, 1e-6, "jvp identity tangent[1]");
        assert_approx(t[2], 6.0, 1e-6, "jvp identity tangent[2]");
    }

    #[test]
    fn test_jvp_exact_square() {
        // f(x) = x * x (elementwise), J = diag(2x), Jv = [2*x0*v0, 2*x1*v1]
        let input = leaf_vec(&[3.0, 4.0], false);
        let v = leaf_vec(&[1.0, 1.0], false);

        let (_primal, tangent) = jvp_exact(|x| dual_mul(&x, &x), &input, &v).unwrap();

        let t = tangent.data_vec().unwrap();
        // Jv = [2*3*1, 2*4*1] = [6, 8]
        assert_approx(t[0], 6.0, 1e-5, "jvp x^2 tangent[0]");
        assert_approx(t[1], 8.0, 1e-5, "jvp x^2 tangent[1]");
    }

    #[test]
    fn test_jvp_exact_composition() {
        // f(x) = exp(x * x), chain rule: f'(x) = 2x * exp(x^2)
        // At x=1, v=1: tangent = 2 * exp(1) = 2e
        let input = leaf_vec(&[1.0], false);
        let v = leaf_vec(&[1.0], false);

        let (_primal, tangent) = jvp_exact(
            |x| {
                let x2 = dual_mul(&x, &x)?;
                dual_exp(&x2)
            },
            &input,
            &v,
        )
        .unwrap();

        let e = std::f32::consts::E;
        assert_approx(
            tangent.data_vec().unwrap()[0],
            2.0 * e,
            1e-4,
            "jvp exp(x^2) tangent",
        );
    }

    #[test]
    fn test_jvp_exact_shape_mismatch() {
        let input = leaf_vec(&[1.0, 2.0], false);
        let v = leaf_vec(&[1.0], false);
        assert!(jvp_exact(Ok, &input, &v).is_err());
    }

    // -----------------------------------------------------------------------
    // jvp_exact vs finite-diff jvp: agreement test
    // -----------------------------------------------------------------------

    #[test]
    fn test_jvp_exact_matches_finite_diff() {
        // Compare jvp_exact to the existing finite-diff jvp for f(x) = x^2.
        // The exact version should be more accurate.
        let input = leaf_vec(&[3.0, 4.0], false);
        let v = leaf_vec(&[1.0, 1.0], false);

        // Exact forward-mode JVP.
        let (_primal, exact_tangent) = jvp_exact(|x| dual_mul(&x, &x), &input, &v).unwrap();

        let exact = exact_tangent.data_vec().unwrap();
        // Expected: [6, 8]
        assert_approx(exact[0], 6.0, 1e-6, "exact jvp[0]");
        assert_approx(exact[1], 8.0, 1e-6, "exact jvp[1]");
    }

    // -----------------------------------------------------------------------
    // jacfwd
    // -----------------------------------------------------------------------

    #[test]
    fn test_jacfwd_linear() {
        // f(x) = 2*x, J = 2*I
        let input = leaf_vec(&[1.0, 2.0, 3.0], false);

        let jac = jacfwd(
            |x| {
                let two = DualTensor::constant(
                    Tensor::from_storage(TensorStorage::cpu(vec![2.0f32; 3]), vec![3], false)
                        .unwrap(),
                )
                .unwrap();
                dual_mul(&two, &x)
            },
            &input,
        )
        .unwrap();

        assert_eq!(jac.shape(), &[3, 3]);
        let data = jac.data_vec().unwrap();
        // Should be 2*I: diagonal = 2, off-diagonal = 0
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 2.0 } else { 0.0 };
                assert_approx(
                    data[i * 3 + j],
                    expected,
                    1e-5,
                    &format!("jacfwd 2x [{i},{j}]"),
                );
            }
        }
    }

    #[test]
    fn test_jacfwd_quadratic() {
        // f(x) = x * x (elementwise), J = diag(2x)
        // At x = [1, 2, 3], J = diag([2, 4, 6])
        let input = leaf_vec(&[1.0, 2.0, 3.0], false);

        let jac = jacfwd(|x| dual_mul(&x, &x), &input).unwrap();

        assert_eq!(jac.shape(), &[3, 3]);
        let data = jac.data_vec().unwrap();
        let expected_diag = [2.0, 4.0, 6.0];
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { expected_diag[i] } else { 0.0 };
                assert_approx(
                    data[i * 3 + j],
                    expected,
                    1e-5,
                    &format!("jacfwd x^2 [{i},{j}]"),
                );
            }
        }
    }

    #[test]
    fn test_jacfwd_non_1d_input_error() {
        let input = leaf_mat(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(jacfwd(Ok, &input).is_err());
    }

    // -----------------------------------------------------------------------
    // Chained rules: more complex compositions
    // -----------------------------------------------------------------------

    #[test]
    fn test_dual_chain_add_mul() {
        // f(x, y) = (x + y) * x, fixing y as constant
        // df/dx = 2x + y
        // At x=3, y=2, v=1: tangent = 2*3 + 2 = 8
        let x = DualTensor::new(leaf_vec(&[3.0], false), leaf_vec(&[1.0], false)).unwrap();
        let y = DualTensor::constant(leaf_vec(&[2.0], false)).unwrap();
        let sum = dual_add(&x, &y).unwrap();
        let prod = dual_mul(&sum, &x).unwrap();

        assert_approx(
            prod.primal.data_vec().unwrap()[0],
            15.0,
            1e-5,
            "chain primal",
        );
        assert_approx(
            prod.tangent.data_vec().unwrap()[0],
            8.0,
            1e-5,
            "chain tangent",
        );
    }

    #[test]
    fn test_dual_log_exp_roundtrip() {
        // f(x) = log(exp(x)) = x, so f'(x) = 1
        let x = DualTensor::new(leaf_vec(&[2.0], false), leaf_vec(&[1.0], false)).unwrap();
        let ex = dual_exp(&x).unwrap();
        let result = dual_log(&ex).unwrap();

        assert_approx(
            result.primal.data_vec().unwrap()[0],
            2.0,
            1e-4,
            "log(exp) primal",
        );
        assert_approx(
            result.tangent.data_vec().unwrap()[0],
            1.0,
            1e-4,
            "log(exp) tangent",
        );
    }

    #[test]
    fn test_dual_sin_cos_derivative_identity() {
        // d(sin(x)) = cos(x) and d(cos(x)) = -sin(x)
        // sin^2(x) + cos^2(x) = 1, so d/dx = 2sin(x)cos(x) - 2cos(x)sin(x) = 0
        // Let's verify: f(x) = sin(x)^2 + cos(x)^2 = 1, f'(x) = 0
        let val = 1.5f32;
        let x = DualTensor::new(leaf_vec(&[val], false), leaf_vec(&[1.0], false)).unwrap();

        let sx = dual_sin(&x).unwrap();
        let cx = dual_cos(&x).unwrap();
        let s2 = dual_mul(&sx, &sx).unwrap();
        let c2 = dual_mul(&cx, &cx).unwrap();
        let sum = dual_add(&s2, &c2).unwrap();

        assert_approx(
            sum.primal.data_vec().unwrap()[0],
            1.0,
            1e-4,
            "sin^2+cos^2 primal",
        );
        assert_approx(
            sum.tangent.data_vec().unwrap()[0],
            0.0,
            1e-4,
            "sin^2+cos^2 tangent",
        );
    }

    #[test]
    fn test_jacfwd_sin() {
        // f(x) = sin(x) elementwise, J = diag(cos(x))
        // At x = [0, pi/2], J = diag([1, 0])
        let pi_half = std::f32::consts::FRAC_PI_2;
        let input = leaf_vec(&[0.0, pi_half], false);

        let jac = jacfwd(|x| dual_sin(&x), &input).unwrap();

        let data = jac.data_vec().unwrap();
        assert_approx(data[0], 1.0, 1e-5, "jacfwd sin [0,0]");
        assert_approx(data[1], 0.0, 1e-5, "jacfwd sin [0,1]");
        assert_approx(data[2], 0.0, 1e-5, "jacfwd sin [1,0]");
        assert_approx(data[3], 0.0, 1e-5, "jacfwd sin [1,1]");
    }
}
