//! Backward functions for FFT operations.
//!
//! The key mathematical identities:
//! - `d/dx FFT(x) = FFT(grad)` (FFT is linear, so its own Jacobian)
//! - `d/dx IFFT(x) = IFFT(grad)` (same reasoning)
//!
//! More precisely, for the backward pass of `y = fft(x)`:
//!   `grad_input = ifft(grad_output) * n`  (because our ifft divides by n)
//!
//! For `y = ifft(x)`:
//!   `grad_input = fft(grad_output) / n`

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::fft;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ---------------------------------------------------------------------------
// FftBackward
// ---------------------------------------------------------------------------

/// Backward for `y = fft(x, n)`.
///
/// VJP: `grad_x = ifft(grad_y) * n` (un-normalized inverse).
/// Since our `ifft` already divides by n, the grad is just `ifft(grad_y) * n`,
/// but actually the correct VJP for a normalized FFT pair where
/// `fft` has no normalization and `ifft` divides by n is:
/// `grad_x = conj(fft(conj(grad_y))) / n = ifft(grad_y) * n / n = ifft(grad_y)` ... wait.
///
/// Let's be precise. Our conventions:
/// - `fft`: no normalization (forward sum without 1/n).
/// - `ifft`: divides by n.
///
/// For `y = FFT(x)` (unnormalized), the Jacobian is the DFT matrix W.
/// The VJP is `grad_x = W^H @ grad_y = n * IFFT(grad_y)`.
///
/// But our `ifft` already computes `(1/n) * W^H @ input`, so
/// `grad_x = n * ifft(grad_y)`.
#[derive(Debug)]
pub struct FftBackward<T: Float> {
    input: Tensor<T>,
    n: Option<usize>,
}

impl<T: Float> FftBackward<T> {
    pub fn new(input: Tensor<T>, n: Option<usize>) -> Self {
        Self { input, n }
    }
}

impl<T: Float> GradFn<T> for FftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            // grad_x = n * ifft(grad_y)
            let inv = fft::ifft(grad_output, self.n)?;
            let fft_n = grad_output.shape()[grad_output.ndim() - 2];
            let scale = T::from(fft_n).unwrap();
            let inv_data = inv.data_vec()?;
            let scaled: Vec<T> = inv_data.iter().map(|&v| v * scale).collect();
            let t = Tensor::from_storage(
                TensorStorage::cpu(scaled),
                inv.shape().to_vec(),
                false,
            )?;
            Some(if device.is_cuda() { t.to(device)? } else { t })
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FftBackward"
    }
}

// ---------------------------------------------------------------------------
// IfftBackward
// ---------------------------------------------------------------------------

/// Backward for `y = ifft(x, n)`.
///
/// Our `ifft(x)` = (1/n) * W^H @ x, so the VJP is:
/// `grad_x = (1/n) * W @ grad_y = (1/n) * fft(grad_y)`.
///
/// Since our `fft` is unnormalized: `grad_x = fft(grad_y) / n`.
#[derive(Debug)]
pub struct IfftBackward<T: Float> {
    input: Tensor<T>,
    n: Option<usize>,
}

impl<T: Float> IfftBackward<T> {
    pub fn new(input: Tensor<T>, n: Option<usize>) -> Self {
        Self { input, n }
    }
}

impl<T: Float> GradFn<T> for IfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            // grad_x = fft(grad_y) / n
            let fwd = fft::fft(grad_output, self.n)?;
            let fft_n = grad_output.shape()[grad_output.ndim() - 2];
            let scale = T::from(1.0).unwrap() / T::from(fft_n).unwrap();
            let fwd_data = fwd.data_vec()?;
            let scaled: Vec<T> = fwd_data.iter().map(|&v| v * scale).collect();
            let t = Tensor::from_storage(
                TensorStorage::cpu(scaled),
                fwd.shape().to_vec(),
                false,
            )?;
            Some(if device.is_cuda() { t.to(device)? } else { t })
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IfftBackward"
    }
}

// ---------------------------------------------------------------------------
// RfftBackward
// ---------------------------------------------------------------------------

/// Backward for `y = rfft(x, n)`.
///
/// `rfft` takes real input of shape `[..., n]` and produces complex output
/// `[..., n/2+1, 2]`. The backward needs to produce a real gradient.
///
/// The VJP is: `grad_x = real(irfft_full(grad_y, n))` where `irfft_full`
/// extends the Hermitian spectrum and does an inverse FFT.
/// More precisely: `grad_x = irfft(grad_y, n)`.
#[derive(Debug)]
pub struct RfftBackward<T: Float> {
    input: Tensor<T>,
    _n: Option<usize>,
    /// The actual FFT length used in the forward pass.
    fft_n: usize,
}

impl<T: Float> RfftBackward<T> {
    pub fn new(input: Tensor<T>, n: Option<usize>, fft_n: usize) -> Self {
        Self { input, _n: n, fft_n }
    }
}

impl<T: Float> GradFn<T> for RfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            // Use irfft to convert the complex gradient back to real.
            // irfft(grad_y, n=fft_n) gives us a real tensor of the right size.
            Some(fft::irfft(grad_output, Some(self.fft_n))?)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "RfftBackward"
    }
}

// ---------------------------------------------------------------------------
// IrfftBackward
// ---------------------------------------------------------------------------

/// Backward for `y = irfft(x, n)`.
///
/// `irfft` takes complex input `[..., n/2+1, 2]` and produces real output
/// `[..., n]`. The backward produces a complex gradient.
///
/// The VJP is: `grad_x = rfft(grad_y, n=n/2+1-related)`.
/// More precisely, `grad_x = rfft(grad_y)` truncated/padded appropriately.
#[derive(Debug)]
pub struct IrfftBackward<T: Float> {
    input: Tensor<T>,
    _n: Option<usize>,
    /// The output length used in the forward pass.
    output_n: usize,
}

impl<T: Float> IrfftBackward<T> {
    pub fn new(input: Tensor<T>, n: Option<usize>, output_n: usize) -> Self {
        Self { input, _n: n, output_n }
    }
}

impl<T: Float> GradFn<T> for IrfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            // rfft(grad_y, n=output_n) gives us [.., output_n/2+1, 2] which
            // should match the input shape.
            Some(fft::rfft(grad_output, Some(self.output_n))?)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IrfftBackward"
    }
}

// ---------------------------------------------------------------------------
// Differentiable forward wrappers
// ---------------------------------------------------------------------------

/// Differentiable 1-D FFT. Attaches `FftBackward` when grad is needed.
pub fn fft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let result = fft::fft(input, n)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(FftBackward::new(input.clone(), n));
        let out = Tensor::from_operation(
            TensorStorage::cpu(result.data_vec()?),
            result.shape().to_vec(),
            grad_fn,
        )?;
        if device.is_cuda() { out.to(device) } else { Ok(out) }
    } else {
        Ok(result)
    }
}

/// Differentiable 1-D inverse FFT. Attaches `IfftBackward` when grad is needed.
pub fn ifft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let result = fft::ifft(input, n)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(IfftBackward::new(input.clone(), n));
        let out = Tensor::from_operation(
            TensorStorage::cpu(result.data_vec()?),
            result.shape().to_vec(),
            grad_fn,
        )?;
        if device.is_cuda() { out.to(device) } else { Ok(out) }
    } else {
        Ok(result)
    }
}

/// Differentiable 1-D real FFT. Attaches `RfftBackward` when grad is needed.
pub fn rfft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let input_n = *input.shape().last().unwrap();
    let fft_n = n.unwrap_or(input_n);
    let result = fft::rfft(input, n)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(RfftBackward::new(input.clone(), n, fft_n));
        let out = Tensor::from_operation(
            TensorStorage::cpu(result.data_vec()?),
            result.shape().to_vec(),
            grad_fn,
        )?;
        if device.is_cuda() { out.to(device) } else { Ok(out) }
    } else {
        Ok(result)
    }
}

/// Differentiable 1-D inverse real FFT. Attaches `IrfftBackward` when grad is needed.
pub fn irfft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let shape = input.shape();
    let half_n = shape[shape.len() - 2];
    let output_n = n.unwrap_or(2 * (half_n - 1));
    let result = fft::irfft(input, n)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(IrfftBackward::new(input.clone(), n, output_n));
        let out = Tensor::from_operation(
            TensorStorage::cpu(result.data_vec()?),
            result.shape().to_vec(),
            grad_fn,
        )?;
        if device.is_cuda() { out.to(device) } else { Ok(out) }
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
    use crate::storage::TensorStorage;

    fn leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    fn no_grad_leaf(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    fn assert_close(actual: &[f64], expected: &[f64], tol: f64) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: {a} vs {e} (diff {})",
                (a - e).abs()
            );
        }
    }

    #[test]
    fn fft_differentiable_attaches_grad_fn() {
        // Complex input [4, 2] with requires_grad.
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let result = fft_differentiable(&input, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "FftBackward");
    }

    #[test]
    fn fft_differentiable_no_grad_when_not_needed() {
        let input = no_grad_leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let result = fft_differentiable(&input, None).unwrap();
        assert!(result.grad_fn().is_none());
    }

    #[test]
    fn fft_backward_identity_check() {
        // For FFT of an impulse [1,0,0,0] -> [1,1,1,1] (all real).
        // grad_output = ones_like(output) = [[1,0],[1,0],[1,0],[1,0]].
        // grad_input = n * ifft(grad_output).
        // ifft([1,1,1,1]) = [1,0,0,0] (impulse).
        // So grad_input = 4 * [1,0,0,0] = [4,0,0,0].
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let result = fft_differentiable(&input, None).unwrap();

        let grad_out = no_grad_leaf(
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            &[4, 2],
        );
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        assert!(grads[0].is_some());

        let g = grads[0].as_ref().unwrap();
        assert_eq!(g.shape(), &[4, 2]);
        let gd = g.data().unwrap();
        // Should be [4, 0, 0, 0, 0, 0, 0, 0].
        assert_close(gd, &[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1e-10);
    }

    #[test]
    fn ifft_backward_identity_check() {
        // ifft([1,1,1,1]) = [1,0,0,0].
        // grad_output = [[1,0],[0,0],[0,0],[0,0]].
        // grad_input = fft(grad_output) / n.
        // fft([1,0,0,0]) = [1,1,1,1].
        // grad_input = [1,1,1,1] / 4 = [0.25, 0.25, 0.25, 0.25].
        let input = leaf(
            &[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            &[4, 2],
        );
        let result = ifft_differentiable(&input, None).unwrap();

        let grad_out = no_grad_leaf(
            &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            &[4, 2],
        );
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        assert!(grads[0].is_some());

        let g = grads[0].as_ref().unwrap();
        let gd = g.data().unwrap();
        // Each complex value should be (0.25, 0.0).
        assert_close(
            gd,
            &[0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0],
            1e-10,
        );
    }

    #[test]
    fn rfft_differentiable_attaches_grad_fn() {
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let result = rfft_differentiable(&input, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "RfftBackward");
    }

    #[test]
    fn irfft_differentiable_attaches_grad_fn() {
        // Input: [3, 2] complex -> irfft -> [4] real.
        let input = leaf(&[10.0, 0.0, -2.0, 2.0, -2.0, 0.0], &[3, 2]);
        let result = irfft_differentiable(&input, Some(4)).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "IrfftBackward");
    }

    #[test]
    fn no_grad_context_disables_tracking() {
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let result = crate::autograd::no_grad::no_grad(|| {
            fft_differentiable(&input, None).unwrap()
        });
        assert!(result.grad_fn().is_none());
    }
}
