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
            let t = Tensor::from_storage(TensorStorage::cpu(scaled), inv.shape().to_vec(), false)?;
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
            let t = Tensor::from_storage(TensorStorage::cpu(scaled), fwd.shape().to_vec(), false)?;
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
        Self {
            input,
            _n: n,
            fft_n,
        }
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
        Self {
            input,
            _n: n,
            output_n,
        }
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
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
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
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
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
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
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
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// FftnBackward / IfftnBackward — N-D complex FFT backward.
// ---------------------------------------------------------------------------
//
// Math (FftNorm::Backward convention, matches torch.fft):
//   y = fftn(x, s, axes)   → grad_x = prod(s) * ifftn(grad_y, s, axes)
//   y = ifftn(x, s, axes)  → grad_x = fftn(grad_y, s, axes) / prod(s)
//
// The shape of the transform output along each transform axis is the value
// in `s` (or the input length if `s` is `None`). We persist `s` and `axes`
// from the forward to keep the backward shape-stable.

#[derive(Debug)]
pub struct FftnBackward<T: Float> {
    input: Tensor<T>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    /// Product of the transform-axis lengths in the forward output.
    norm_n: usize,
}

impl<T: Float> FftnBackward<T> {
    pub fn new(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        norm_n: usize,
    ) -> Self {
        Self {
            input,
            s,
            axes,
            norm_n,
        }
    }
}

impl<T: Float> GradFn<T> for FftnBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            let inv = fft::ifftn(grad_output, self.s.as_deref(), self.axes.as_deref())?;
            let scale = T::from(self.norm_n).unwrap();
            let inv_data = inv.data_vec()?;
            let scaled: Vec<T> = inv_data.iter().map(|&v| v * scale).collect();
            let t = Tensor::from_storage(TensorStorage::cpu(scaled), inv.shape().to_vec(), false)?;
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
        "FftnBackward"
    }
}

#[derive(Debug)]
pub struct IfftnBackward<T: Float> {
    input: Tensor<T>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
    /// Product of the transform-axis lengths.
    norm_n: usize,
}

impl<T: Float> IfftnBackward<T> {
    pub fn new(
        input: Tensor<T>,
        s: Option<Vec<usize>>,
        axes: Option<Vec<isize>>,
        norm_n: usize,
    ) -> Self {
        Self {
            input,
            s,
            axes,
            norm_n,
        }
    }
}

impl<T: Float> GradFn<T> for IfftnBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            let device = grad_output.device();
            let fwd = fft::fftn(grad_output, self.s.as_deref(), self.axes.as_deref())?;
            let scale = T::from(1.0).unwrap() / T::from(self.norm_n).unwrap();
            let fwd_data = fwd.data_vec()?;
            let scaled: Vec<T> = fwd_data.iter().map(|&v| v * scale).collect();
            let t = Tensor::from_storage(TensorStorage::cpu(scaled), fwd.shape().to_vec(), false)?;
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
        "IfftnBackward"
    }
}

// ---------------------------------------------------------------------------
// RfftnBackward / IrfftnBackward — N-D real FFT backward.
// ---------------------------------------------------------------------------
//
// VJPs:
//   y = rfftn(x, s, axes) (real → Hermitian complex)
//     → grad_x = irfftn(grad_y, s=original_real_shape, axes)
//   y = irfftn(x, s, axes) (Hermitian complex → real)
//     → grad_x = rfftn(grad_y, s=original_real_shape, axes)

#[derive(Debug)]
pub struct RfftnBackward<T: Float> {
    input: Tensor<T>,
    /// Output sizes along the transform axes (passed to irfftn for backward).
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
}

impl<T: Float> RfftnBackward<T> {
    pub fn new(input: Tensor<T>, s: Option<Vec<usize>>, axes: Option<Vec<isize>>) -> Self {
        Self { input, s, axes }
    }
}

impl<T: Float> GradFn<T> for RfftnBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            Some(fft::irfftn(
                grad_output,
                self.s.as_deref(),
                self.axes.as_deref(),
            )?)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "RfftnBackward"
    }
}

#[derive(Debug)]
pub struct IrfftnBackward<T: Float> {
    input: Tensor<T>,
    s: Option<Vec<usize>>,
    axes: Option<Vec<isize>>,
}

impl<T: Float> IrfftnBackward<T> {
    pub fn new(input: Tensor<T>, s: Option<Vec<usize>>, axes: Option<Vec<isize>>) -> Self {
        Self { input, s, axes }
    }
}

impl<T: Float> GradFn<T> for IrfftnBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            Some(fft::rfftn(
                grad_output,
                self.s.as_deref(),
                self.axes.as_deref(),
            )?)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IrfftnBackward"
    }
}

// ---------------------------------------------------------------------------
// HfftBackward / IhfftBackward — Hermitian FFT backward.
// ---------------------------------------------------------------------------
//
// hfft maps Hermitian-symmetric complex `[..., n/2+1, 2]` → real `[..., n]`.
// ihfft is the inverse: real `[..., n]` → Hermitian complex `[..., n/2+1, 2]`.
//
// VJPs (matching torch.fft.hfft / ihfft):
//   y = hfft(x, n)  → grad_x = ihfft(grad_y, n=input_n)
//   y = ihfft(x, n) → grad_x = hfft(grad_y, n=input_n)

#[derive(Debug)]
pub struct HfftBackward<T: Float> {
    input: Tensor<T>,
    /// Length of the original Hermitian spectrum (input's second-to-last dim).
    input_n: usize,
}

impl<T: Float> HfftBackward<T> {
    pub fn new(input: Tensor<T>, input_n: usize) -> Self {
        Self { input, input_n }
    }
}

impl<T: Float> GradFn<T> for HfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            // ihfft expects a real-valued input, returns Hermitian complex of
            // length `n/2+1`. Using `n=2*(input_n-1)` recovers the original
            // spectrum length `input_n`.
            let n_forward = 2 * (self.input_n - 1);
            Some(fft::ihfft(grad_output, Some(n_forward))?)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "HfftBackward"
    }
}

#[derive(Debug)]
pub struct IhfftBackward<T: Float> {
    input: Tensor<T>,
    /// Length of the original real signal (input's last dim).
    input_n: usize,
}

impl<T: Float> IhfftBackward<T> {
    pub fn new(input: Tensor<T>, input_n: usize) -> Self {
        Self { input, input_n }
    }
}

impl<T: Float> GradFn<T> for IhfftBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let grad_input = if self.input.requires_grad() {
            // hfft maps Hermitian complex `[..., n/2+1, 2]` → real `[..., n]`.
            // grad_y has shape `[..., n/2+1, 2]`; we want grad_x of shape
            // `[..., input_n]`. Pass `n=input_n` so hfft outputs that length.
            Some(fft::hfft(grad_output, Some(self.input_n))?)
        } else {
            None
        };
        Ok(vec![grad_input])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "IhfftBackward"
    }
}

// ---------------------------------------------------------------------------
// Differentiable forward wrappers — N-D + Hermitian (#580)
// ---------------------------------------------------------------------------

/// Compute the product of transform-axis lengths used for normalization.
/// Mirrors how the forward pass would interpret `s` / `axes`:
///   - If `s` is given, multiply its entries.
///   - Else if `axes` is given, multiply the input's lengths along those axes.
///   - Else multiply the inner dims (excluding the trailing complex pair).
fn fftn_norm_n<T: Float>(input: &Tensor<T>, s: Option<&[usize]>, axes: Option<&[isize]>) -> usize {
    if let Some(s_slice) = s {
        return s_slice.iter().copied().product::<usize>().max(1);
    }
    let shape = input.shape();
    let ndim = shape.len();
    if let Some(axes_slice) = axes {
        let mut prod: usize = 1;
        for &a in axes_slice {
            // Resolve negative axes against `ndim - 1` (excluding trailing
            // complex pair).
            let logical_ndim = ndim.saturating_sub(1);
            let resolved = if a < 0 {
                (logical_ndim as isize + a) as usize
            } else {
                a as usize
            };
            prod = prod.saturating_mul(shape[resolved]);
        }
        return prod.max(1);
    }
    // Default: all inner dims (skip the trailing 2).
    if ndim < 2 {
        1
    } else {
        shape[..ndim - 1].iter().product::<usize>().max(1)
    }
}

/// Same as [`fftn_norm_n`] but for real inputs: there is no trailing complex
/// pair, so all dims except the leading batch are candidates.
fn rfftn_norm_n<T: Float>(input: &Tensor<T>, s: Option<&[usize]>, axes: Option<&[isize]>) -> usize {
    if let Some(s_slice) = s {
        return s_slice.iter().copied().product::<usize>().max(1);
    }
    let shape = input.shape();
    let ndim = shape.len();
    if let Some(axes_slice) = axes {
        let mut prod: usize = 1;
        for &a in axes_slice {
            let resolved = if a < 0 {
                (ndim as isize + a) as usize
            } else {
                a as usize
            };
            prod = prod.saturating_mul(shape[resolved]);
        }
        return prod.max(1);
    }
    shape.iter().product::<usize>().max(1)
}

/// Differentiable N-D FFT. Attaches `FftnBackward` when grad is needed.
pub fn fftn_differentiable<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let result = fft::fftn(input, s, axes)?;

    if is_grad_enabled() && input.requires_grad() {
        let norm_n = fftn_norm_n(input, s, axes);
        let grad_fn = Arc::new(FftnBackward::new(
            input.clone(),
            s.map(|v| v.to_vec()),
            axes.map(|v| v.to_vec()),
            norm_n,
        ));
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable N-D inverse FFT. Attaches `IfftnBackward` when grad is needed.
pub fn ifftn_differentiable<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let result = fft::ifftn(input, s, axes)?;

    if is_grad_enabled() && input.requires_grad() {
        let norm_n = fftn_norm_n(input, s, axes);
        let grad_fn = Arc::new(IfftnBackward::new(
            input.clone(),
            s.map(|v| v.to_vec()),
            axes.map(|v| v.to_vec()),
            norm_n,
        ));
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable N-D real FFT. Attaches `RfftnBackward` when grad is needed.
pub fn rfftn_differentiable<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let _ = rfftn_norm_n::<T>; // keep helper available for symmetry; not needed in fwd
    let result = fft::rfftn(input, s, axes)?;

    if is_grad_enabled() && input.requires_grad() {
        // Backward needs the original real-input shape along the transform
        // axes. We pass the input's shape segment so irfftn can reconstruct.
        let s_back: Vec<usize> = match (s, axes) {
            (Some(s_slice), _) => s_slice.to_vec(),
            (None, Some(axes_slice)) => {
                let shape = input.shape();
                axes_slice
                    .iter()
                    .map(|&a| {
                        let resolved = if a < 0 {
                            (shape.len() as isize + a) as usize
                        } else {
                            a as usize
                        };
                        shape[resolved]
                    })
                    .collect()
            }
            (None, None) => input.shape().to_vec(),
        };
        let grad_fn = Arc::new(RfftnBackward::new(
            input.clone(),
            Some(s_back),
            axes.map(|v| v.to_vec()),
        ));
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable N-D inverse real FFT. Attaches `IrfftnBackward` when grad is needed.
pub fn irfftn_differentiable<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let result = fft::irfftn(input, s, axes)?;

    if is_grad_enabled() && input.requires_grad() {
        // The forward output length along each transform axis becomes the
        // original real shape; back-pass uses the same `s` to reconstruct.
        let s_back: Vec<usize> = match s {
            Some(s_slice) => s_slice.to_vec(),
            None => result.shape().to_vec(),
        };
        let grad_fn = Arc::new(IrfftnBackward::new(
            input.clone(),
            Some(s_back),
            axes.map(|v| v.to_vec()),
        ));
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable Hermitian FFT (complex spectrum → real signal). Attaches
/// `HfftBackward` when grad is needed.
pub fn hfft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let shape = input.shape();
    let input_n = shape[shape.len() - 2];
    let _ = n; // (used at fwd call); backward derives spectrum length from input_n
    let result = fft::hfft(input, n)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(HfftBackward::new(input.clone(), input_n));
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable inverse Hermitian FFT (real signal → Hermitian spectrum).
/// Attaches `IhfftBackward` when grad is needed.
pub fn ihfft_differentiable<T: Float>(
    input: &Tensor<T>,
    n: Option<usize>,
) -> FerrotorchResult<Tensor<T>> {
    let device = input.device();
    let shape = input.shape();
    let input_n = *shape.last().unwrap();
    let _ = n;
    let result = fft::ihfft(input, n)?;

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(IhfftBackward::new(input.clone(), input_n));
        let storage = TensorStorage::on_device(result.data_vec()?, device)?;
        Tensor::from_operation(storage, result.shape().to_vec(), grad_fn)
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

        let grad_out = no_grad_leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[4, 2]);
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
        let input = leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[4, 2]);
        let result = ifft_differentiable(&input, None).unwrap();

        let grad_out = no_grad_leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[4, 2]);
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        assert!(grads[0].is_some());

        let g = grads[0].as_ref().unwrap();
        let gd = g.data().unwrap();
        // Each complex value should be (0.25, 0.0).
        assert_close(gd, &[0.25, 0.0, 0.25, 0.0, 0.25, 0.0, 0.25, 0.0], 1e-10);
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
        let result =
            crate::autograd::no_grad::no_grad(|| fft_differentiable(&input, None).unwrap());
        assert!(result.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // N-D FFT differentiable wrappers (#580)
    // -----------------------------------------------------------------------

    #[test]
    fn fftn_differentiable_attaches_grad_fn() {
        // 2x2 complex input: [[1+0i, 0+0i], [0+0i, 0+0i]] → flat [2, 2, 2].
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2]);
        let result = fftn_differentiable(&input, None, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "FftnBackward");
    }

    #[test]
    fn ifftn_differentiable_attaches_grad_fn() {
        let input = leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 2, 2]);
        let result = ifftn_differentiable(&input, None, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "IfftnBackward");
    }

    #[test]
    fn fftn_no_grad_when_not_needed() {
        let input = no_grad_leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2]);
        let result = fftn_differentiable(&input, None, None).unwrap();
        assert!(result.grad_fn().is_none());
    }

    #[test]
    fn fftn_backward_returns_real_grad_for_impulse() {
        // 2x2 impulse: real [[1,0],[0,0]] (encoded complex as
        // [[1+0i, 0+0i], [0+0i, 0+0i]]). fftn → all-ones 2x2 complex
        // (DFT-2D of a corner impulse). grad_y = ones → grad_x =
        // prod_s * ifftn(ones) = 4 * impulse / 4 → impulse_complex.
        let input = leaf(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], &[2, 2, 2]);
        let result = fftn_differentiable(&input, None, None).unwrap();
        // grad_y = ones (4 complex pairs).
        let grad_out = no_grad_leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 2, 2]);
        let grads = result.grad_fn().unwrap().backward(&grad_out).unwrap();
        let g = grads[0].as_ref().unwrap();
        // Expected: 4 * ifftn(ones) over a 2x2 grid → 4 * impulse / 4 = impulse.
        assert_close(
            g.data().unwrap(),
            &[4.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            1e-9,
        );
    }

    #[test]
    fn rfftn_differentiable_attaches_grad_fn() {
        // Real 2x2 input → rfftn → [2, 2, 2] complex (n/2+1 along last).
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let result = rfftn_differentiable(&input, None, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "RfftnBackward");
    }

    #[test]
    fn irfftn_differentiable_attaches_grad_fn() {
        // Hermitian-shaped complex input [2, 2, 2]: 2 batch × 2 freq × complex.
        let input = leaf(&[1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0], &[2, 2, 2]);
        let result = irfftn_differentiable(&input, Some(&[2, 2]), None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "IrfftnBackward");
    }

    #[test]
    fn hfft_differentiable_attaches_grad_fn() {
        // Hermitian spectrum [3, 2] → real [4]. n=4 means input_n=3 (n/2+1).
        let input = leaf(&[10.0, 0.0, -2.0, 2.0, -2.0, 0.0], &[3, 2]);
        let result = hfft_differentiable(&input, Some(4)).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "HfftBackward");
    }

    #[test]
    fn ihfft_differentiable_attaches_grad_fn() {
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let result = ihfft_differentiable(&input, None).unwrap();
        assert!(result.grad_fn().is_some());
        assert_eq!(result.grad_fn().unwrap().name(), "IhfftBackward");
    }

    #[test]
    fn fftn_norm_n_default_inner_dims() {
        // shape [2, 3, 4, 2] (last dim is complex pair) → norm_n = 2*3*4 = 24.
        let input = no_grad_leaf(&vec![0.0; 2 * 3 * 4 * 2], &[2, 3, 4, 2]);
        let n = fftn_norm_n(&input, None, None);
        assert_eq!(n, 2 * 3 * 4);
    }

    #[test]
    fn fftn_norm_n_with_explicit_s() {
        let input = no_grad_leaf(&[0.0; 8 * 2], &[2, 2, 2, 2]);
        let n = fftn_norm_n(&input, Some(&[3, 5]), None);
        assert_eq!(n, 15);
    }

    #[test]
    fn fftn_norm_n_with_axes() {
        // Axes = [1, 2] → norm_n = shape[1] * shape[2] = 3 * 4 = 12.
        let input = no_grad_leaf(&vec![0.0; 2 * 3 * 4 * 2], &[2, 3, 4, 2]);
        let n = fftn_norm_n(&input, None, Some(&[1, 2]));
        assert_eq!(n, 12);
    }
}
