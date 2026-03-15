//! Backward functions for activation operations.
//!
//! Each struct stores the tensors needed for the VJP (vector-Jacobian product)
//! and implements [`GradFn`] to participate in reverse-mode autodiff.

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::ops::elementwise::unary_map;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// ReLU
// ---------------------------------------------------------------------------

/// Backward for `relu(x)`.
///
/// VJP: `grad * (x > 0)` — the step-function mask.
#[derive(Debug)]
pub struct ReluBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> ReluBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for ReluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let zero = <T as num_traits::Zero>::zero();

        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| if x > zero { g } else { zero })
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "ReluBackward"
    }
}

// ---------------------------------------------------------------------------
// Sigmoid
// ---------------------------------------------------------------------------

/// Backward for `sigmoid(x)`.
///
/// VJP: `grad * s * (1 - s)` where `s = sigmoid(x)` (the output).
#[derive(Debug)]
pub struct SigmoidBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> SigmoidBackward<T> {
    pub fn new(input: Tensor<T>, output: Tensor<T>) -> Self {
        Self { input, output }
    }
}

impl<T: Float> GradFn<T> for SigmoidBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let s_data = self.output.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = s_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&s, &g)| g * s * (one - s))
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SigmoidBackward"
    }
}

// ---------------------------------------------------------------------------
// Tanh
// ---------------------------------------------------------------------------

/// Backward for `tanh(x)`.
///
/// VJP: `grad * (1 - t^2)` where `t = tanh(x)` (the output).
#[derive(Debug)]
pub struct TanhBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> TanhBackward<T> {
    pub fn new(input: Tensor<T>, output: Tensor<T>) -> Self {
        Self { input, output }
    }
}

impl<T: Float> GradFn<T> for TanhBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let t_data = self.output.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = t_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&t, &g)| g * (one - t * t))
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "TanhBackward"
    }
}

// ---------------------------------------------------------------------------
// GELU (sigmoid approximation)
// ---------------------------------------------------------------------------

/// Backward for `gelu(x)` using the sigmoid approximation:
///
/// ```text
/// gelu(x) ≈ x * sigmoid(1.702 * x)
/// ```
///
/// Derivative:
/// ```text
/// grad * (s + 1.702 * x * s * (1 - s))
/// ```
/// where `s = sigmoid(1.702 * x)`.
#[derive(Debug)]
pub struct GeluBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> GeluBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for GeluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();
        let k = T::from(1.702).unwrap();

        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let s = one / (one + (-k * x).exp());
                g * (s + k * x * s * (one - s))
            })
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "GeluBackward"
    }
}

// ---------------------------------------------------------------------------
// SiLU (Swish)
// ---------------------------------------------------------------------------

/// Backward for `silu(x) = x * sigmoid(x)`.
///
/// VJP: `grad * (s + x * s * (1 - s))` where `s = sigmoid(x)`.
#[derive(Debug)]
pub struct SiluBackward<T: Float> {
    input: Tensor<T>,
}

impl<T: Float> SiluBackward<T> {
    pub fn new(input: Tensor<T>) -> Self {
        Self { input }
    }
}

impl<T: Float> GradFn<T> for SiluBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let input_data = self.input.data()?;
        let grad_data = grad_output.data()?;
        let one = <T as num_traits::One>::one();

        let result: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                let s = one / (one + (-x).exp());
                g * (s + x * s * (one - s))
            })
            .collect();

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SiluBackward"
    }
}

// ---------------------------------------------------------------------------
// Softmax
// ---------------------------------------------------------------------------

/// Backward for `softmax(x)` along the last axis.
///
/// VJP: `softmax * (grad - sum(grad * softmax, axis=-1, keepdim))`.
///
/// Stores the softmax **output** (not input) for efficiency.
#[derive(Debug)]
pub struct SoftmaxBackward<T: Float> {
    input: Tensor<T>,
    output: Tensor<T>,
}

impl<T: Float> SoftmaxBackward<T> {
    pub fn new(input: Tensor<T>, output: Tensor<T>) -> Self {
        Self { input, output }
    }
}

impl<T: Float> GradFn<T> for SoftmaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let s_data = self.output.data()?;
        let grad_data = grad_output.data()?;
        let shape = self.output.shape();

        if shape.is_empty() {
            // Scalar softmax: derivative is 0 (softmax of scalar is always 1).
            let zero = <T as num_traits::Zero>::zero();
            let grad_input =
                Tensor::from_storage(TensorStorage::cpu(vec![zero]), vec![], false)?;
            return Ok(vec![Some(grad_input)]);
        }

        let last_dim = *shape.last().unwrap();
        let outer = s_data.len() / last_dim.max(1);
        let mut result = vec![<T as num_traits::Zero>::zero(); s_data.len()];

        for i in 0..outer {
            let base = i * last_dim;
            // Compute dot = sum(grad * softmax) along this last-axis slice.
            let mut dot = <T as num_traits::Zero>::zero();
            for j in 0..last_dim {
                dot = dot + grad_data[base + j] * s_data[base + j];
            }
            // grad_input = softmax * (grad - dot)
            for j in 0..last_dim {
                result[base + j] = s_data[base + j] * (grad_data[base + j] - dot);
            }
        }

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "SoftmaxBackward"
    }
}

// ---------------------------------------------------------------------------
// LogSoftmax
// ---------------------------------------------------------------------------

/// Backward for `log_softmax(x)` along the last axis.
///
/// VJP: `grad - softmax * sum(grad, axis=-1, keepdim)`.
///
/// Stores the **softmax output** (= exp(log_softmax)) for efficiency.
#[derive(Debug)]
pub struct LogSoftmaxBackward<T: Float> {
    input: Tensor<T>,
    /// The softmax output, i.e. `exp(log_softmax(x))`.
    softmax_output: Tensor<T>,
}

impl<T: Float> LogSoftmaxBackward<T> {
    pub fn new(input: Tensor<T>, softmax_output: Tensor<T>) -> Self {
        Self {
            input,
            softmax_output,
        }
    }
}

impl<T: Float> GradFn<T> for LogSoftmaxBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let sm_data = self.softmax_output.data()?;
        let grad_data = grad_output.data()?;
        let shape = self.input.shape();

        if shape.is_empty() {
            // Scalar: log_softmax(x) = 0 always, derivative is 0.
            let zero = <T as num_traits::Zero>::zero();
            let grad_input =
                Tensor::from_storage(TensorStorage::cpu(vec![zero]), vec![], false)?;
            return Ok(vec![Some(grad_input)]);
        }

        let last_dim = *shape.last().unwrap();
        let outer = sm_data.len() / last_dim.max(1);
        let mut result = vec![<T as num_traits::Zero>::zero(); sm_data.len()];

        for i in 0..outer {
            let base = i * last_dim;
            // sum_grad = sum(grad) along this last-axis slice.
            let mut sum_grad = <T as num_traits::Zero>::zero();
            for j in 0..last_dim {
                sum_grad = sum_grad + grad_data[base + j];
            }
            // grad_input = grad - softmax * sum_grad
            for j in 0..last_dim {
                result[base + j] = grad_data[base + j] - sm_data[base + j] * sum_grad;
            }
        }

        let grad_input = Tensor::from_storage(
            TensorStorage::cpu(result),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "LogSoftmaxBackward"
    }
}

// ---------------------------------------------------------------------------
// Forward activation helpers (attach grad_fn when grad is enabled)
// ---------------------------------------------------------------------------

/// Compute `relu(x)`, attaching a backward node when gradients are enabled.
pub fn relu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let zero = <T as num_traits::Zero>::zero();
    let output = unary_map(input, |x| if x > zero { x } else { zero })?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(ReluBackward::new(input.clone()));
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            grad_fn,
        )
    } else {
        Ok(output)
    }
}

/// Compute `sigmoid(x)`, attaching a backward node when gradients are enabled.
pub fn sigmoid<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // SIMD-accelerated sigmoid: compute exp(-x) via SIMD, then 1/(1+exp(-x))
    let output = if std::mem::size_of::<T>() == 4 {
        let data = input.data()?;
        let n = data.len();
        let inp: &[f32] = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const f32, n) };
        // Negate
        let neg: Vec<f32> = inp.iter().map(|&x| -x).collect();
        // SIMD exp
        let mut exp_out = vec![0.0f32; n];
        ferray_ufunc::kernels::simd_f32::exp_f32(&neg, &mut exp_out);
        // 1 / (1 + exp(-x))
        let result: Vec<T> = exp_out.iter().map(|&e| T::from(1.0f32 / (1.0 + e)).unwrap()).collect();
        Tensor::from_storage(TensorStorage::cpu(result), input.shape().to_vec(), false)?
    } else {
        let one = <T as num_traits::One>::one();
        unary_map(input, |x| one / (one + (-x).exp()))?
    };

    if is_grad_enabled() && input.requires_grad() {
        let result = Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(SigmoidBackward::new(input.clone(), output.clone())),
        )?;
        Ok(result)
    } else {
        Ok(output)
    }
}

/// Compute `tanh(x)`, attaching a backward node when gradients are enabled.
pub fn tanh<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let output = unary_map(input, |x| x.tanh())?;

    if is_grad_enabled() && input.requires_grad() {
        let result = Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(TanhBackward::new(input.clone(), output.clone())),
        )?;
        Ok(result)
    } else {
        Ok(output)
    }
}

/// Compute `gelu(x)` using the sigmoid approximation, attaching a backward
/// node when gradients are enabled.
///
/// ```text
/// gelu(x) ≈ x * sigmoid(1.702 * x)
/// ```
pub fn gelu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let one = <T as num_traits::One>::one();
    let k = T::from(1.702).unwrap();
    let output = unary_map(input, |x| {
        let s = one / (one + (-k * x).exp());
        x * s
    })?;

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(GeluBackward::new(input.clone())),
        )
    } else {
        Ok(output)
    }
}

/// Compute `silu(x) = x * sigmoid(x)`, attaching a backward node when
/// gradients are enabled.
pub fn silu<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let one = <T as num_traits::One>::one();
    let output = unary_map(input, |x| {
        let s = one / (one + (-x).exp());
        x * s
    })?;

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(SiluBackward::new(input.clone())),
        )
    } else {
        Ok(output)
    }
}

/// Compute `softmax(x)` along the last axis, attaching a backward node when
/// gradients are enabled.
pub fn softmax<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let shape = input.shape();

    let result = if shape.is_empty() {
        // Scalar: softmax is always 1.
        vec![<T as num_traits::One>::one()]
    } else {
        let last_dim = *shape.last().unwrap();
        let outer = data.len() / last_dim.max(1);
        let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];

        for i in 0..outer {
            let base = i * last_dim;
            // Numerical stability: subtract max.
            let mut max_val = data[base];
            for j in 1..last_dim {
                if data[base + j] > max_val {
                    max_val = data[base + j];
                }
            }
            let mut sum_exp = <T as num_traits::Zero>::zero();
            for j in 0..last_dim {
                let e = (data[base + j] - max_val).exp();
                out[base + j] = e;
                sum_exp = sum_exp + e;
            }
            for j in 0..last_dim {
                out[base + j] = out[base + j] / sum_exp;
            }
        }
        out
    };

    let output =
        Tensor::from_storage(TensorStorage::cpu(result), shape.to_vec(), false)?;

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(SoftmaxBackward::new(input.clone(), output.clone())),
        )
    } else {
        Ok(output)
    }
}

/// Compute `log_softmax(x)` along the last axis, attaching a backward node
/// when gradients are enabled.
pub fn log_softmax<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = input.data()?;
    let shape = input.shape();

    // Compute softmax and log_softmax simultaneously for efficiency.
    let (sm_vec, lsm_vec) = if shape.is_empty() {
        // Scalar: softmax = 1, log_softmax = 0.
        (
            vec![<T as num_traits::One>::one()],
            vec![<T as num_traits::Zero>::zero()],
        )
    } else {
        let last_dim = *shape.last().unwrap();
        let outer = data.len() / last_dim.max(1);
        let mut sm = vec![<T as num_traits::Zero>::zero(); data.len()];
        let mut lsm = vec![<T as num_traits::Zero>::zero(); data.len()];

        for i in 0..outer {
            let base = i * last_dim;
            let mut max_val = data[base];
            for j in 1..last_dim {
                if data[base + j] > max_val {
                    max_val = data[base + j];
                }
            }
            let mut sum_exp = <T as num_traits::Zero>::zero();
            for j in 0..last_dim {
                let e = (data[base + j] - max_val).exp();
                sm[base + j] = e;
                sum_exp = sum_exp + e;
            }
            let log_sum = sum_exp.ln();
            for j in 0..last_dim {
                sm[base + j] = sm[base + j] / sum_exp;
                lsm[base + j] = data[base + j] - max_val - log_sum;
            }
        }
        (sm, lsm)
    };

    let softmax_tensor =
        Tensor::from_storage(TensorStorage::cpu(sm_vec), shape.to_vec(), false)?;

    let output =
        Tensor::from_storage(TensorStorage::cpu(lsm_vec), shape.to_vec(), false)?;

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            TensorStorage::cpu(output.data()?.to_vec()),
            output.shape().to_vec(),
            Arc::new(LogSoftmaxBackward::new(input.clone(), softmax_tensor)),
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
    use crate::autograd::graph::backward;
    use crate::storage::TensorStorage;

    /// Helper: create a scalar leaf tensor with `requires_grad = true`.
    fn leaf_scalar(val: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap()
    }

    /// Helper: create a 1-D leaf tensor with `requires_grad = true`.
    fn leaf_vec(vals: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(vals.to_vec()),
            vec![vals.len()],
            true,
        )
        .unwrap()
    }

    /// Numerical gradient via central difference: (f(x+h) - f(x-h)) / (2h).
    fn numerical_grad_scalar(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        let h = 1e-5;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    // -----------------------------------------------------------------------
    // ReLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu_forward_positive() {
        let x = leaf_scalar(2.0);
        let y = relu(&x).unwrap();
        assert!((y.item().unwrap() - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_relu_forward_negative() {
        let x = leaf_scalar(-3.0);
        let y = relu(&x).unwrap();
        assert!((y.item().unwrap()).abs() < 1e-7);
    }

    #[test]
    fn test_relu_backward_positive() {
        let x = leaf_scalar(2.0);
        let y = relu(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d(relu)/dx at x=2 is 1.
        assert!(
            (grad.item().unwrap() - 1.0).abs() < 1e-6,
            "relu grad at x=2: expected 1.0, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_relu_backward_negative() {
        let x = leaf_scalar(-1.5);
        let y = relu(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d(relu)/dx at x=-1.5 is 0.
        assert!(
            grad.item().unwrap().abs() < 1e-6,
            "relu grad at x=-1.5: expected 0.0, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_relu_forward_vector() {
        let x = leaf_vec(&[-1.0, 0.5, 2.0, -0.3]);
        let y = relu(&x).unwrap();
        let y_data = y.data().unwrap();
        assert!((y_data[0] - 0.0).abs() < 1e-7);
        assert!((y_data[1] - 0.5).abs() < 1e-7);
        assert!((y_data[2] - 2.0).abs() < 1e-7);
        assert!((y_data[3] - 0.0).abs() < 1e-7);
    }

    // -----------------------------------------------------------------------
    // Sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn test_sigmoid_forward() {
        let x = leaf_scalar(0.0);
        let y = sigmoid(&x).unwrap();
        assert!((y.item().unwrap() - 0.5).abs() < 1e-7);
    }

    #[test]
    fn test_sigmoid_backward() {
        let x = leaf_scalar(0.0);
        let y = sigmoid(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // sigmoid'(0) = 0.5 * (1 - 0.5) = 0.25.
        assert!(
            (grad.item().unwrap() - 0.25).abs() < 1e-6,
            "sigmoid grad at x=0: expected 0.25, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_sigmoid_backward_nonzero() {
        let val = 1.0_f64;
        let x = leaf_scalar(val);
        let y = sigmoid(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // Compare with numerical gradient.
        let expected = numerical_grad_scalar(|v| 1.0 / (1.0 + (-v).exp()), val);
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-5,
            "sigmoid grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Tanh
    // -----------------------------------------------------------------------

    #[test]
    fn test_tanh_forward() {
        let x = leaf_scalar(0.0);
        let y = tanh(&x).unwrap();
        assert!(y.item().unwrap().abs() < 1e-7);
    }

    #[test]
    fn test_tanh_backward_at_zero() {
        let x = leaf_scalar(0.0);
        let y = tanh(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // tanh'(0) = 1 - tanh(0)^2 = 1.
        assert!(
            (grad.item().unwrap() - 1.0).abs() < 1e-6,
            "tanh grad at x=0: expected 1.0, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_tanh_backward_nonzero() {
        let val = 0.8_f64;
        let x = leaf_scalar(val);
        let y = tanh(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(|v| v.tanh(), val);
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-5,
            "tanh grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // GELU
    // -----------------------------------------------------------------------

    #[test]
    fn test_gelu_forward_zero() {
        let x = leaf_scalar(0.0);
        let y = gelu(&x).unwrap();
        // gelu(0) = 0 * sigmoid(0) = 0.
        assert!(y.item().unwrap().abs() < 1e-7);
    }

    #[test]
    fn test_gelu_backward() {
        let val = 1.0_f64;
        let x = leaf_scalar(val);
        let y = gelu(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let k = 1.702_f64;
        let expected = numerical_grad_scalar(
            |v| {
                let s = 1.0 / (1.0 + (-k * v).exp());
                v * s
            },
            val,
        );
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "gelu grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // SiLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_silu_forward_zero() {
        let x = leaf_scalar(0.0);
        let y = silu(&x).unwrap();
        // silu(0) = 0 * sigmoid(0) = 0.
        assert!(y.item().unwrap().abs() < 1e-7);
    }

    #[test]
    fn test_silu_backward() {
        let val = 1.5_f64;
        let x = leaf_scalar(val);
        let y = silu(&x).unwrap();
        backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad_scalar(
            |v| {
                let s = 1.0 / (1.0 + (-v).exp());
                v * s
            },
            val,
        );
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "silu grad at x={}: expected {}, got {}",
            val,
            expected,
            grad.item().unwrap()
        );
    }

    // -----------------------------------------------------------------------
    // Softmax
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_forward_1d() {
        let x = leaf_vec(&[1.0, 2.0, 3.0]);
        let y = softmax(&x).unwrap();
        let d = y.data().unwrap();
        // Softmax values should sum to 1.
        let total: f64 = d.iter().copied().sum();
        assert!(
            (total - 1.0).abs() < 1e-7,
            "softmax sum: expected 1.0, got {}",
            total
        );
        // Monotonicity: s(1) < s(2) < s(3).
        assert!(d[0] < d[1]);
        assert!(d[1] < d[2]);
    }

    #[test]
    fn test_softmax_backward_1d() {
        // For a 1D softmax, verify the backward struct directly.
        let vals = [1.0_f64, 2.0, 3.0];
        let x = leaf_vec(&vals);
        let y = softmax(&x).unwrap();
        let y_data = y.data().unwrap().to_vec();

        // Use grad_output = [1, 0, 0] to probe d(softmax_0)/d(x_j).
        let grad_output = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 0.0, 0.0]),
            vec![3],
            false,
        )
        .unwrap();

        let bwd = SoftmaxBackward::new(x.clone(), y.clone());
        let grads = bwd.backward(&grad_output).unwrap();
        let gx = grads[0].as_ref().unwrap().data().unwrap().to_vec();

        // Expected: s_0 * (delta_{0j} - s_j)
        let s0 = y_data[0];
        let s1 = y_data[1];
        let s2 = y_data[2];
        let expected = [s0 * (1.0 - s0), s0 * (0.0 - s1), s0 * (0.0 - s2)];

        for (i, (&got, &exp)) in gx.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-7,
                "softmax grad[{}]: expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    // -----------------------------------------------------------------------
    // LogSoftmax
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_softmax_forward_1d() {
        let x = leaf_vec(&[1.0, 2.0, 3.0]);
        let y = log_softmax(&x).unwrap();
        let d = y.data().unwrap();
        // exp(log_softmax) should sum to 1.
        let total: f64 = d.iter().map(|&v| v.exp()).sum();
        assert!(
            (total - 1.0).abs() < 1e-7,
            "exp(log_softmax) sum: expected 1.0, got {}",
            total
        );
    }

    #[test]
    fn test_log_softmax_backward_1d() {
        let vals = [1.0_f64, 2.0, 3.0];
        let x = leaf_vec(&vals);

        // Compute softmax for reference (on a non-grad tensor to avoid
        // entangling computation graphs).
        let x_nograd = Tensor::from_storage(
            TensorStorage::cpu(vals.to_vec()),
            vec![3],
            false,
        )
        .unwrap();
        let sm = softmax(&x_nograd).unwrap();
        let sm_data = sm.data().unwrap().to_vec();

        let grad_output = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 0.0, 0.0]),
            vec![3],
            false,
        )
        .unwrap();

        let bwd = LogSoftmaxBackward::new(x.clone(), sm);
        let grads = bwd.backward(&grad_output).unwrap();
        let gx = grads[0].as_ref().unwrap().data().unwrap().to_vec();

        // Expected: grad - softmax * sum(grad)
        // sum(grad) = 1.0
        // grad_input = [1 - s0, 0 - s1, 0 - s2]
        let expected = [1.0 - sm_data[0], 0.0 - sm_data[1], 0.0 - sm_data[2]];

        for (i, (&got, &exp)) in gx.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-7,
                "log_softmax grad[{}]: expected {}, got {}",
                i,
                exp,
                got
            );
        }
    }

    // -----------------------------------------------------------------------
    // no_grad disables backward nodes
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu_no_grad() {
        crate::autograd::no_grad::no_grad(|| {
            let x = leaf_scalar(2.0);
            let y = relu(&x).unwrap();
            assert!(
                y.grad_fn().is_none(),
                "relu inside no_grad should not attach grad_fn"
            );
        });
    }

    #[test]
    fn test_sigmoid_no_grad() {
        crate::autograd::no_grad::no_grad(|| {
            let x = leaf_scalar(1.0);
            let y = sigmoid(&x).unwrap();
            assert!(
                y.grad_fn().is_none(),
                "sigmoid inside no_grad should not attach grad_fn"
            );
        });
    }
}
