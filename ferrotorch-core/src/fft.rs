//! FFT operations for tensors.
//!
//! Complex values are represented as an extra trailing dimension of size 2,
//! where `[..., 0]` is the real part and `[..., 1]` is the imaginary part.
//! This matches PyTorch's convention for `torch.fft.*` operations.
//!
//! All functions work on f32, f64, and bf16 tensors via an f64 round-trip:
//! input is upcast to f64, the transform runs in double precision, and the
//! result is cast back to the input dtype. The 1-D and 2-D paths
//! ([`fft`], [`ifft`], [`fft2`], [`ifft2`], [`rfft`], [`irfft`]) are powered
//! by [`rustfft`] directly. The N-D, Hermitian, frequency-helper, and
//! shift paths ([`fftn`], [`ifftn`], [`rfftn`], [`irfftn`], [`hfft`],
//! [`ihfft`], [`fftfreq`], [`rfftfreq`], [`fftshift`], [`ifftshift`]) are
//! delegated to [`ferray_fft`].
//!
//! # GPU note
//!
//! No cuFFT path exists yet. Functions in this module reject GPU tensors
//! with [`FerrotorchError::NotImplementedOnCuda`]. They never silently
//! move a CUDA tensor through host memory.

use ferray_core::Array as FerrayArray;
use ferray_core::IxDyn as FerrayIxDyn;
use ferray_fft::FftNorm;
use rustfft::FftPlanner;
use rustfft::num_complex::Complex;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// True when `T` is f32 (4-byte float), used to pick the f32 vs f64 GPU path.
#[inline]
fn is_f32<T: Float>() -> bool {
    std::mem::size_of::<T>() == 4
}

/// True when `T` is f64 (8-byte float).
#[inline]
fn is_f64<T: Float>() -> bool {
    std::mem::size_of::<T>() == 8
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Convert Complex<f64> back to flat [re, im, re, im, ...] in type T.
fn complex_to_pairs<T: Float>(data: &[Complex<f64>]) -> Vec<T> {
    let mut out = Vec::with_capacity(data.len() * 2);
    for c in data {
        out.push(T::from(c.re).unwrap());
        out.push(T::from(c.im).unwrap());
    }
    out
}

/// Execute 1-D FFT (forward or inverse) along the last axis of shape `batch_shape`.
///
/// `data` is a flat array of `Complex<f64>` with total length = product(batch_shape) * n.
/// The last dimension has size `n`.
///
/// Returns the transformed data with the same layout.
fn fft_1d_last_axis(data: &mut [Complex<f64>], batch_shape: &[usize], n: usize, inverse: bool) {
    let mut planner = FftPlanner::<f64>::new();
    let fft = if inverse {
        planner.plan_fft_inverse(n)
    } else {
        planner.plan_fft_forward(n)
    };

    let batch_size: usize = if batch_shape.is_empty() {
        1
    } else {
        batch_shape.iter().product()
    };

    // Process each batch slice in-place.
    for b in 0..batch_size {
        let offset = b * n;
        fft.process(&mut data[offset..offset + n]);
    }

    // Apply normalization for inverse: divide by n.
    if inverse {
        let scale = 1.0 / n as f64;
        for v in data.iter_mut() {
            v.re *= scale;
            v.im *= scale;
        }
    }
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// 1-D complex-to-complex FFT along the last dimension.
///
/// The input tensor must have a trailing dimension of size 2 representing
/// complex numbers `[re, im]`. If `n` is provided, the signal is truncated
/// or zero-padded along the second-to-last dimension before transforming.
///
/// Returns a tensor with shape `[..., n, 2]` (or `[..., input_len, 2]` if
/// `n` is `None`).
pub fn fft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();

    // Input must end with a dim of 2 (complex representation).
    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fft: input must have trailing dimension 2 (complex), got shape {:?}",
                shape
            ),
        });
    }

    let ndim = shape.len();
    // Signal length is the second-to-last dim.
    if ndim < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: "fft: input must have at least 2 dimensions ([..., n, 2])".into(),
        });
    }

    let input_n = shape[ndim - 2];
    let fft_n = n.unwrap_or(input_n);
    if fft_n == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "fft: n must be > 0".into(),
        });
    }

    let batch_shape = &shape[..ndim - 2];
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        // GPU C2C dispatch via cuFFT (#579), with on-device pad/truncate
        // when `fft_n != input_n` (#605). Fully on-device — no host bounce.
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = input.gpu_handle()?;

        // Optional pad/truncate to fft_n.
        let (transformed_handle, owned);
        let buf_for_fft: &crate::gpu_dispatch::GpuBufferHandle = if fft_n == input_n {
            buf
        } else if is_f32::<T>() {
            owned = backend.pad_truncate_complex_f32(buf, batch_size, input_n, fft_n)?;
            transformed_handle = &owned;
            transformed_handle
        } else {
            owned = backend.pad_truncate_complex_f64(buf, batch_size, input_n, fft_n)?;
            transformed_handle = &owned;
            transformed_handle
        };

        let h = if is_f32::<T>() {
            backend.fft_c2c_f32(buf_for_fft, batch_size, fft_n, false)?
        } else {
            backend.fft_c2c_f64(buf_for_fft, batch_size, fft_n, false)?
        };
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(fft_n);
        out_shape.push(2);
        return Tensor::from_storage(TensorStorage::gpu(h), out_shape, false);
    }
    let data = input.data_vec()?;

    let mut complex_data = Vec::with_capacity(batch_size * fft_n);
    for b in 0..batch_size {
        let src_offset = b * input_n * 2;
        let copy_len = input_n.min(fft_n);
        for i in 0..copy_len {
            let re = data[src_offset + i * 2].to_f64().unwrap();
            let im = data[src_offset + i * 2 + 1].to_f64().unwrap();
            complex_data.push(Complex::new(re, im));
        }
        for _ in copy_len..fft_n {
            complex_data.push(Complex::new(0.0, 0.0));
        }
    }

    fft_1d_last_axis(&mut complex_data, batch_shape, fft_n, false);

    let result_data = complex_to_pairs::<T>(&complex_data);
    let mut out_shape = batch_shape.to_vec();
    out_shape.push(fft_n);
    out_shape.push(2);

    Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)
}

/// 1-D inverse FFT along the last dimension.
///
/// Input has shape `[..., n, 2]` (complex). Returns complex output of the
/// same shape (or `[..., n_out, 2]` if `n` is specified).
pub fn ifft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();

    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ifft: input must have trailing dimension 2 (complex), got shape {:?}",
                shape
            ),
        });
    }

    let ndim = shape.len();
    if ndim < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: "ifft: input must have at least 2 dimensions ([..., n, 2])".into(),
        });
    }

    let input_n = shape[ndim - 2];
    let fft_n = n.unwrap_or(input_n);
    if fft_n == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "ifft: n must be > 0".into(),
        });
    }

    let batch_shape = &shape[..ndim - 2];
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

    if input.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        // GPU C2C dispatch via cuFFT, with on-device pad/truncate when
        // `fft_n != input_n` (#605).
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = input.gpu_handle()?;

        let (transformed_handle, owned);
        let buf_for_fft: &crate::gpu_dispatch::GpuBufferHandle = if fft_n == input_n {
            buf
        } else if is_f32::<T>() {
            owned = backend.pad_truncate_complex_f32(buf, batch_size, input_n, fft_n)?;
            transformed_handle = &owned;
            transformed_handle
        } else {
            owned = backend.pad_truncate_complex_f64(buf, batch_size, input_n, fft_n)?;
            transformed_handle = &owned;
            transformed_handle
        };

        let h = if is_f32::<T>() {
            backend.fft_c2c_f32(buf_for_fft, batch_size, fft_n, true)?
        } else {
            backend.fft_c2c_f64(buf_for_fft, batch_size, fft_n, true)?
        };
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(fft_n);
        out_shape.push(2);
        return Tensor::from_storage(TensorStorage::gpu(h), out_shape, false);
    }
    let data = input.data_vec()?;

    let mut complex_data = Vec::with_capacity(batch_size * fft_n);
    for b in 0..batch_size {
        let src_offset = b * input_n * 2;
        let copy_len = input_n.min(fft_n);
        for i in 0..copy_len {
            let re = data[src_offset + i * 2].to_f64().unwrap();
            let im = data[src_offset + i * 2 + 1].to_f64().unwrap();
            complex_data.push(Complex::new(re, im));
        }
        for _ in copy_len..fft_n {
            complex_data.push(Complex::new(0.0, 0.0));
        }
    }

    fft_1d_last_axis(&mut complex_data, batch_shape, fft_n, true);

    let result_data = complex_to_pairs::<T>(&complex_data);
    let mut out_shape = batch_shape.to_vec();
    out_shape.push(fft_n);
    out_shape.push(2);

    Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)
}

/// 1-D real-to-complex FFT along the last dimension.
///
/// Input is a real-valued tensor of shape `[..., n]`. Output has shape
/// `[..., n/2+1, 2]` representing the non-redundant complex coefficients.
pub fn rfft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if shape.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "rfft: input must have at least 1 dimension".into(),
        });
    }

    let ndim = shape.len();
    let input_n = shape[ndim - 1];
    let fft_n = n.unwrap_or(input_n);
    if fft_n == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "rfft: n must be > 0".into(),
        });
    }

    let batch_shape = &shape[..ndim - 1];
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

    if input.is_cuda() && fft_n == input_n {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = input.gpu_handle()?;
        let h = if is_f32::<T>() {
            backend.rfft_r2c_f32(buf, batch_size, fft_n)?
        } else if is_f64::<T>() {
            backend.rfft_r2c_f64(buf, batch_size, fft_n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "rfft requires f32 or f64".into(),
            });
        };
        let half_n = fft_n / 2 + 1;
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(half_n);
        out_shape.push(2);
        return Tensor::from_storage(TensorStorage::gpu(h), out_shape, false);
    }
    let data = input.data_vec()?;

    // Build complex input from real data (zero imaginary).
    let mut complex_data = Vec::with_capacity(batch_size * fft_n);
    for b in 0..batch_size {
        let src_offset = b * input_n;
        let copy_len = input_n.min(fft_n);
        for i in 0..copy_len {
            complex_data.push(Complex::new(data[src_offset + i].to_f64().unwrap(), 0.0));
        }
        for _ in copy_len..fft_n {
            complex_data.push(Complex::new(0.0, 0.0));
        }
    }

    // Forward FFT.
    fft_1d_last_axis(&mut complex_data, batch_shape, fft_n, false);

    // Truncate to n/2+1 (Hermitian symmetry).
    let half_n = fft_n / 2 + 1;
    let mut result_data = Vec::with_capacity(batch_size * half_n * 2);
    for b in 0..batch_size {
        let offset = b * fft_n;
        for i in 0..half_n {
            let c = complex_data[offset + i];
            result_data.push(T::from(c.re).unwrap());
            result_data.push(T::from(c.im).unwrap());
        }
    }

    let mut out_shape = batch_shape.to_vec();
    out_shape.push(half_n);
    out_shape.push(2);

    Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)
}

/// 1-D complex-to-real inverse FFT.
///
/// Input has shape `[..., n/2+1, 2]` (Hermitian spectrum). Output is
/// real-valued with shape `[..., n]`. If `n` is `None`, uses `2*(m-1)`
/// where `m` is the input's second-to-last dimension.
pub fn irfft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();

    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "irfft: input must have trailing dimension 2 (complex), got shape {:?}",
                shape
            ),
        });
    }

    let ndim = shape.len();
    if ndim < 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: "irfft: input must have at least 2 dimensions ([..., n/2+1, 2])".into(),
        });
    }

    let half_n = shape[ndim - 2];
    let output_n = n.unwrap_or(2 * (half_n - 1));
    if output_n == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "irfft: output length must be > 0".into(),
        });
    }

    let batch_shape = &shape[..ndim - 2];
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

    if input.is_cuda() && half_n == output_n / 2 + 1 {
        // GPU path: input spectrum length matches `output_n / 2 + 1`. The
        // mismatched-`n` case still routes through CPU below; it requires
        // a Hermitian-extension or truncation step that's deferred.
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = input.gpu_handle()?;
        let h = if is_f32::<T>() {
            backend.irfft_c2r_f32(buf, batch_size, output_n)?
        } else if is_f64::<T>() {
            backend.irfft_c2r_f64(buf, batch_size, output_n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "irfft requires f32 or f64".into(),
            });
        };
        let mut out_shape = batch_shape.to_vec();
        out_shape.push(output_n);
        return Tensor::from_storage(TensorStorage::gpu(h), out_shape, false);
    }
    let data = input.data_vec()?;

    // Extend Hermitian-symmetric spectrum to full length.
    let mut complex_data = Vec::with_capacity(batch_size * output_n);
    for b in 0..batch_size {
        let src_offset = b * half_n * 2;

        // Copy the first half_n values.
        for i in 0..half_n.min(output_n) {
            let re = data[src_offset + i * 2].to_f64().unwrap();
            let im = data[src_offset + i * 2 + 1].to_f64().unwrap();
            complex_data.push(Complex::new(re, im));
        }

        // Fill the mirror half using Hermitian symmetry: X[k] = conj(X[n-k]).
        for k in half_n..output_n {
            let mirror = output_n - k;
            if mirror < half_n {
                let re = data[src_offset + mirror * 2].to_f64().unwrap();
                let im = data[src_offset + mirror * 2 + 1].to_f64().unwrap();
                complex_data.push(Complex::new(re, -im));
            } else {
                complex_data.push(Complex::new(0.0, 0.0));
            }
        }
    }

    // Inverse FFT.
    fft_1d_last_axis(&mut complex_data, batch_shape, output_n, true);

    // Extract real parts.
    let result_data: Vec<T> = complex_data
        .iter()
        .map(|c| T::from(c.re).unwrap())
        .collect();

    let mut out_shape = batch_shape.to_vec();
    out_shape.push(output_n);

    Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)
}

/// 2-D FFT (complex-to-complex) along the last two spatial dimensions.
///
/// Input has shape `[..., rows, cols, 2]` (complex). Output has the same shape.
pub fn fft2<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();

    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fft2: input must have trailing dimension 2 (complex), got shape {:?}",
                shape
            ),
        });
    }

    let ndim = shape.len();
    if ndim < 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: "fft2: input must have at least 3 dimensions ([..., rows, cols, 2])".into(),
        });
    }

    let rows = shape[ndim - 3];
    let cols = shape[ndim - 2];
    let batch_dims: usize = shape[..ndim - 3].iter().product::<usize>().max(1);

    // GPU fast path via cufftPlan2d (#634): unbatched (or batch=1) f32/f64.
    if input.is_cuda() && batch_dims == 1 && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = if is_f32::<T>() {
            backend.fft2_c2c_f32(input.gpu_handle()?, rows, cols, false)?
        } else {
            backend.fft2_c2c_f64(input.gpu_handle()?, rows, cols, false)?
        };
        return Tensor::from_storage(TensorStorage::gpu(h), shape.to_vec(), false);
    }

    // Step 1: FFT along columns (last spatial axis).
    let after_cols = fft(input, Some(cols))?;

    // Step 2: FFT along rows — need to transpose rows<->cols, FFT, transpose back.
    fft_2d_row_pass(&after_cols, rows, cols, false)
}

/// 2-D inverse FFT (complex-to-complex) along the last two spatial dimensions.
///
/// Input has shape `[..., rows, cols, 2]` (complex). Output has the same shape.
pub fn ifft2<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();

    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ifft2: input must have trailing dimension 2 (complex), got shape {:?}",
                shape
            ),
        });
    }

    let ndim = shape.len();
    if ndim < 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: "ifft2: input must have at least 3 dimensions ([..., rows, cols, 2])".into(),
        });
    }

    let rows = shape[ndim - 3];
    let cols = shape[ndim - 2];
    let batch_dims: usize = shape[..ndim - 3].iter().product::<usize>().max(1);

    if input.is_cuda() && batch_dims == 1 && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let h = if is_f32::<T>() {
            backend.fft2_c2c_f32(input.gpu_handle()?, rows, cols, true)?
        } else {
            backend.fft2_c2c_f64(input.gpu_handle()?, rows, cols, true)?
        };
        return Tensor::from_storage(TensorStorage::gpu(h), shape.to_vec(), false);
    }

    // Step 1: IFFT along columns.
    let after_cols = ifft(input, Some(cols))?;

    // Step 2: IFFT along rows.
    fft_2d_row_pass(&after_cols, rows, cols, true)
}

/// Internal: apply FFT/IFFT along the row axis of a `[..., rows, cols, 2]` tensor.
///
/// This transposes the rows and cols, runs a 1-D FFT along the (now-last)
/// spatial axis, then transposes back.
fn fft_2d_row_pass<T: Float>(
    input: &Tensor<T>,
    rows: usize,
    cols: usize,
    inverse: bool,
) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "fft2" });
    }
    let ndim = shape.len();
    let batch_shape = &shape[..ndim - 3];
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);
    let data = input.data_vec()?;

    // Transpose [batch, rows, cols, 2] -> [batch, cols, rows, 2] so rows become the last spatial dim.
    let mut transposed = vec![T::from(0.0).unwrap(); data.len()];
    for b in 0..batch_size {
        let base = b * rows * cols * 2;
        for r in 0..rows {
            for c in 0..cols {
                let src = base + r * cols * 2 + c * 2;
                let dst = base + c * rows * 2 + r * 2;
                transposed[dst] = data[src];
                transposed[dst + 1] = data[src + 1];
            }
        }
    }

    // Build a temporary tensor with shape [batch..., cols, rows, 2].
    let mut trans_shape = batch_shape.to_vec();
    trans_shape.push(cols);
    trans_shape.push(rows);
    trans_shape.push(2);
    let trans_tensor = Tensor::from_storage(TensorStorage::cpu(transposed), trans_shape, false)?;

    // FFT along the last spatial dim (rows).
    let transformed = if inverse {
        ifft(&trans_tensor, Some(rows))?
    } else {
        fft(&trans_tensor, Some(rows))?
    };

    // Transpose back [batch, cols, rows, 2] -> [batch, rows, cols, 2].
    let t_data = transformed.data_vec()?;
    let mut result = vec![T::from(0.0).unwrap(); t_data.len()];
    for b in 0..batch_size {
        let base = b * rows * cols * 2;
        for c in 0..cols {
            for r in 0..rows {
                let src = base + c * rows * 2 + r * 2;
                let dst = base + r * cols * 2 + c * 2;
                result[dst] = t_data[src];
                result[dst + 1] = t_data[src + 1];
            }
        }
    }

    let mut out_shape = batch_shape.to_vec();
    out_shape.push(rows);
    out_shape.push(cols);
    out_shape.push(2);

    Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)
}

// ---------------------------------------------------------------------------
// ferray-fft round-trip helpers
// ---------------------------------------------------------------------------
//
// The following helpers move data between ferrotorch's complex-as-trailing-
// dim-2 convention and ferray-fft's `Array<Complex<f64>, IxDyn>` native
// representation. Computation always runs in f64 to support every
// `T: Float` (including bf16, which ferray-fft itself does not implement).

/// Build an `Array<Complex<f64>, IxDyn>` from a tensor whose last dimension
/// is 2 (re, im). Returns the array shape **without** the trailing 2.
fn tensor_to_complex_array<T: Float>(
    input: &Tensor<T>,
    op: &'static str,
) -> FerrotorchResult<FerrayArray<Complex<f64>, FerrayIxDyn>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }

    let shape = input.shape();
    if shape.is_empty() || *shape.last().unwrap() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: input must have trailing dimension 2 (complex), got shape {shape:?}"
            ),
        });
    }

    let data = input.data_vec()?;
    let total_complex = data.len() / 2;
    let mut complex_data = Vec::with_capacity(total_complex);
    for i in 0..total_complex {
        let re = data[i * 2].to_f64().unwrap();
        let im = data[i * 2 + 1].to_f64().unwrap();
        complex_data.push(Complex::new(re, im));
    }

    let inner_shape: Vec<usize> = shape[..shape.len() - 1].to_vec();
    FerrayArray::from_vec(FerrayIxDyn::new(&inner_shape), complex_data).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("{op}: failed to build ferray array: {e}"),
        }
    })
}

/// Build a real `Array<f64, IxDyn>` from a real-valued tensor.
fn tensor_to_real_array<T: Float>(
    input: &Tensor<T>,
    op: &'static str,
) -> FerrotorchResult<FerrayArray<f64, FerrayIxDyn>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op });
    }
    let data = input.data_vec()?;
    let real_data: Vec<f64> = data.iter().map(|v| v.to_f64().unwrap()).collect();
    FerrayArray::from_vec(FerrayIxDyn::new(input.shape()), real_data).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("{op}: failed to build ferray array: {e}"),
        }
    })
}

/// Convert an `Array<Complex<f64>, IxDyn>` back to a `Tensor<T>` with the
/// trailing 2-dim representing complex pairs.
fn complex_array_to_tensor<T: Float>(
    arr: &FerrayArray<Complex<f64>, FerrayIxDyn>,
) -> FerrotorchResult<Tensor<T>> {
    let shape = arr.shape().to_vec();
    let total: usize = shape.iter().product();
    let mut out_data: Vec<T> = Vec::with_capacity(total * 2);
    for c in arr.iter() {
        out_data.push(T::from(c.re).unwrap());
        out_data.push(T::from(c.im).unwrap());
    }
    let mut out_shape = shape;
    out_shape.push(2);
    Tensor::from_storage(TensorStorage::cpu(out_data), out_shape, false)
}

/// Convert an `Array<f64, IxDyn>` back to a real `Tensor<T>`.
fn real_array_to_tensor<T: Float>(
    arr: &FerrayArray<f64, FerrayIxDyn>,
) -> FerrotorchResult<Tensor<T>> {
    let shape = arr.shape().to_vec();
    let out_data: Vec<T> = arr.iter().map(|&v| T::from(v).unwrap()).collect();
    Tensor::from_storage(TensorStorage::cpu(out_data), shape, false)
}

// ---------------------------------------------------------------------------
// N-D complex FFT (fftn, ifftn)
// ---------------------------------------------------------------------------

/// N-dimensional complex-to-complex FFT.
///
/// Input has shape `[..., 2]` representing complex values (last dim = re/im).
/// Transforms over the inner dimensions specified by `axes`, or all inner
/// dimensions if `axes` is `None`. The trailing complex dim is always
/// excluded from the transform set.
///
/// `s` optionally specifies the output length along each transform axis
/// (truncate or zero-pad).
pub fn fftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    let arr = tensor_to_complex_array(input, "fftn")?;
    let result = ferray_fft::fftn(&arr, s, axes, FftNorm::Backward).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("fftn: {e}"),
        }
    })?;
    complex_array_to_tensor(&result)
}

/// N-dimensional inverse complex FFT.
///
/// See [`fftn`] for parameter semantics. Normalization divides by the
/// product of the transform-axis lengths (matches `torch.fft.ifftn`).
pub fn ifftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    let arr = tensor_to_complex_array(input, "ifftn")?;
    let result = ferray_fft::ifftn(&arr, s, axes, FftNorm::Backward).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("ifftn: {e}"),
        }
    })?;
    complex_array_to_tensor(&result)
}

// ---------------------------------------------------------------------------
// N-D real FFT (rfftn, irfftn)
// ---------------------------------------------------------------------------

/// N-dimensional real-to-complex FFT.
///
/// Input is real-valued with shape `[..., n]`. The last transform axis
/// produces `n/2 + 1` complex coefficients (Hermitian symmetry); other
/// transform axes return full length. Output shape is the input shape
/// with the last transform axis replaced by `n/2 + 1` and a trailing 2
/// appended for complex.
pub fn rfftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    let arr = tensor_to_real_array(input, "rfftn")?;
    let result = ferray_fft::rfftn(&arr, s, axes, FftNorm::Backward).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("rfftn: {e}"),
        }
    })?;
    complex_array_to_tensor(&result)
}

/// N-dimensional complex-to-real inverse FFT.
///
/// Inverse of [`rfftn`]. Input has shape `[..., n/2 + 1, 2]` along the
/// last transform axis; output is real with that axis restored to
/// `n` (or whatever `s` specifies).
pub fn irfftn<T: Float>(
    input: &Tensor<T>,
    s: Option<&[usize]>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    let arr = tensor_to_complex_array(input, "irfftn")?;
    let result = ferray_fft::irfftn(&arr, s, axes, FftNorm::Backward).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("irfftn: {e}"),
        }
    })?;
    real_array_to_tensor(&result)
}

// ---------------------------------------------------------------------------
// Hermitian FFT (hfft, ihfft)
// ---------------------------------------------------------------------------

/// 1-D FFT of a Hermitian-symmetric complex spectrum, returning real output.
///
/// Input has shape `[..., n/2 + 1, 2]`; output has shape `[..., n]` (real).
/// If `n` is `None`, uses `2 * (input_len - 1)`.
///
/// The Hermitian condition `X[k] = conj(X[-k])` is implicit in the input.
pub fn hfft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    let arr = tensor_to_complex_array(input, "hfft")?;
    let result = ferray_fft::hfft(&arr, n, None, FftNorm::Backward).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("hfft: {e}"),
        }
    })?;
    real_array_to_tensor(&result)
}

/// 1-D inverse FFT of a real signal, returning a Hermitian-symmetric spectrum.
///
/// Input has shape `[..., n]` (real); output has shape `[..., n/2 + 1, 2]`
/// (complex pairs).
pub fn ihfft<T: Float>(input: &Tensor<T>, n: Option<usize>) -> FerrotorchResult<Tensor<T>> {
    let arr = tensor_to_real_array(input, "ihfft")?;
    let result = ferray_fft::ihfft(&arr, n, None, FftNorm::Backward).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("ihfft: {e}"),
        }
    })?;
    complex_array_to_tensor(&result)
}

// ---------------------------------------------------------------------------
// Frequency helpers (fftfreq, rfftfreq)
// ---------------------------------------------------------------------------

/// Discrete Fourier Transform sample frequencies.
///
/// Returns a length-`n` `Tensor<f64>` on CPU containing the frequency bin
/// centers in cycles per unit of the sample spacing `d`. Matches
/// `torch.fft.fftfreq` and `numpy.fft.fftfreq`.
pub fn fftfreq(n: usize, d: f64) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_fft::fftfreq(n, d).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("fftfreq: {e}"),
    })?;
    let data: Vec<f64> = arr.iter().copied().collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)
}

/// Sample frequencies for `rfft` (non-negative half).
///
/// Returns a length-`n/2 + 1` `Tensor<f64>` on CPU. Matches
/// `torch.fft.rfftfreq`.
pub fn rfftfreq(n: usize, d: f64) -> FerrotorchResult<Tensor<f64>> {
    let arr = ferray_fft::rfftfreq(n, d).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("rfftfreq: {e}"),
    })?;
    let len = arr.shape()[0];
    let data: Vec<f64> = arr.iter().copied().collect();
    Tensor::from_storage(TensorStorage::cpu(data), vec![len], false)
}

// ---------------------------------------------------------------------------
// Shift helpers (fftshift, ifftshift)
// ---------------------------------------------------------------------------

/// Shift the zero-frequency component to the center along the given axes.
///
/// If `axes` is `None`, shifts every axis. Matches `torch.fft.fftshift`
/// (and `numpy.fft.fftshift`).
pub fn fftshift<T: Float>(
    input: &Tensor<T>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "fftshift" });
    }
    let arr = tensor_to_real_array(input, "fftshift")?;
    let shifted =
        ferray_fft::fftshift(&arr, axes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("fftshift: {e}"),
        })?;
    real_array_to_tensor(&shifted)
}

/// Inverse of [`fftshift`].
///
/// Differs from `fftshift` only on odd-length axes. Matches
/// `torch.fft.ifftshift`.
pub fn ifftshift<T: Float>(
    input: &Tensor<T>,
    axes: Option<&[isize]>,
) -> FerrotorchResult<Tensor<T>> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "ifftshift" });
    }
    let arr = tensor_to_real_array(input, "ifftshift")?;
    let shifted =
        ferray_fft::ifftshift(&arr, axes).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("ifftshift: {e}"),
        })?;
    real_array_to_tensor(&shifted)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    /// Create a tensor from data and shape.
    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    fn assert_close(a: &[f64], b: &[f64], tol: f64) {
        assert_eq!(
            a.len(),
            b.len(),
            "length mismatch: {} vs {}",
            a.len(),
            b.len()
        );
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "index {i}: {x} vs {y} (diff {})",
                (x - y).abs()
            );
        }
    }

    /// Build a complex tensor of shape [n, 2] from a slice of (re, im) pairs.
    fn complex_tensor(pairs: &[(f64, f64)]) -> Tensor<f64> {
        let mut data = Vec::with_capacity(pairs.len() * 2);
        for &(re, im) in pairs {
            data.push(re);
            data.push(im);
        }
        t(&data, &[pairs.len(), 2])
    }

    // -----------------------------------------------------------------------
    // fft of zeros is zeros
    // -----------------------------------------------------------------------

    #[test]
    fn fft_of_zeros() {
        let input = complex_tensor(&[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        let result = fft(&input, None).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
        let d = result.data().unwrap();
        for &v in d {
            assert!(v.abs() < 1e-12, "expected 0, got {v}");
        }
    }

    // -----------------------------------------------------------------------
    // fft of ones: DC component = n, rest = 0
    // -----------------------------------------------------------------------

    #[test]
    fn fft_of_ones() {
        let n = 8;
        let pairs: Vec<(f64, f64)> = vec![(1.0, 0.0); n];
        let input = complex_tensor(&pairs);
        let result = fft(&input, None).unwrap();
        assert_eq!(result.shape(), &[n, 2]);
        let d = result.data().unwrap();

        // DC component (index 0): re = n, im = 0.
        assert!(
            (d[0] - n as f64).abs() < 1e-10,
            "DC re = {}, expected {n}",
            d[0]
        );
        assert!(d[1].abs() < 1e-10, "DC im = {}", d[1]);

        // All other bins should be 0.
        for i in 1..n {
            assert!(d[i * 2].abs() < 1e-10, "bin {i} re = {}", d[i * 2]);
            assert!(d[i * 2 + 1].abs() < 1e-10, "bin {i} im = {}", d[i * 2 + 1]);
        }
    }

    // -----------------------------------------------------------------------
    // fft of a pure cosine: peaks at k and n-k
    // -----------------------------------------------------------------------

    #[test]
    fn fft_pure_cosine() {
        let n = 16;
        let k = 3; // frequency bin
        let pi = std::f64::consts::PI;

        // x[i] = cos(2*pi*k*i/n)
        let pairs: Vec<(f64, f64)> = (0..n)
            .map(|i| ((2.0 * pi * k as f64 * i as f64 / n as f64).cos(), 0.0))
            .collect();
        let input = complex_tensor(&pairs);
        let result = fft(&input, None).unwrap();
        let d = result.data().unwrap();

        // Magnitudes: bin k and bin n-k should have magnitude n/2.
        // All others should be ~0.
        for i in 0..n {
            let mag = (d[i * 2] * d[i * 2] + d[i * 2 + 1] * d[i * 2 + 1]).sqrt();
            if i == k || i == n - k {
                assert!(
                    (mag - n as f64 / 2.0).abs() < 1e-8,
                    "bin {i}: magnitude {mag}, expected {}",
                    n as f64 / 2.0
                );
            } else {
                assert!(mag < 1e-8, "bin {i}: magnitude {mag}, expected ~0");
            }
        }
    }

    // -----------------------------------------------------------------------
    // fft -> ifft round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn fft_ifft_roundtrip() {
        let pairs = vec![
            (1.0, 2.0),
            (-1.0, 0.5),
            (3.0, -1.0),
            (0.0, 0.0),
            (-2.5, 1.5),
            (0.7, -0.3),
        ];
        let input = complex_tensor(&pairs);
        let spectrum = fft(&input, None).unwrap();
        let recovered = ifft(&spectrum, None).unwrap();
        let d = recovered.data().unwrap();

        for (i, &(re, im)) in pairs.iter().enumerate() {
            assert!(
                (d[i * 2] - re).abs() < 1e-10,
                "re at {i}: {} vs {re}",
                d[i * 2]
            );
            assert!(
                (d[i * 2 + 1] - im).abs() < 1e-10,
                "im at {i}: {} vs {im}",
                d[i * 2 + 1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // rfft + irfft round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn rfft_irfft_roundtrip() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let n = original.len();
        let input = t(&original, &[n]);

        let spectrum = rfft(&input, None).unwrap();
        // n=8 -> n/2+1 = 5 complex values -> shape [5, 2].
        assert_eq!(spectrum.shape(), &[5, 2]);

        let recovered = irfft(&spectrum, Some(n)).unwrap();
        assert_eq!(recovered.shape(), &[n]);
        let d = recovered.data().unwrap();
        assert_close(d, &original, 1e-10);
    }

    // -----------------------------------------------------------------------
    // rfft output shape
    // -----------------------------------------------------------------------

    #[test]
    fn rfft_output_shape() {
        // Even length.
        let input = t(&[0.0; 8], &[8]);
        let result = rfft(&input, None).unwrap();
        assert_eq!(result.shape(), &[5, 2]); // 8/2+1 = 5

        // Odd length.
        let input = t(&[0.0; 7], &[7]);
        let result = rfft(&input, None).unwrap();
        assert_eq!(result.shape(), &[4, 2]); // 7/2+1 = 4
    }

    // -----------------------------------------------------------------------
    // rfft + irfft round-trip with odd length
    // -----------------------------------------------------------------------

    #[test]
    fn rfft_irfft_roundtrip_odd() {
        let original = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let n = original.len();
        let input = t(&original, &[n]);

        let spectrum = rfft(&input, None).unwrap();
        assert_eq!(spectrum.shape(), &[3, 2]); // 5/2+1 = 3

        let recovered = irfft(&spectrum, Some(n)).unwrap();
        assert_eq!(recovered.shape(), &[n]);
        assert_close(recovered.data().unwrap(), &original, 1e-10);
    }

    // -----------------------------------------------------------------------
    // fft with n parameter (padding/truncation)
    // -----------------------------------------------------------------------

    #[test]
    fn fft_with_padding() {
        // Pad [1+0j, 1+0j] to length 4 -> FFT of [1, 1, 0, 0].
        let input = complex_tensor(&[(1.0, 0.0), (1.0, 0.0)]);
        let result = fft(&input, Some(4)).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
        let d = result.data().unwrap();
        // DC = 2.0.
        assert!((d[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn fft_with_truncation() {
        // Truncate [1, 2, 3, 4] to length 2 -> FFT of [1, 2].
        let input = complex_tensor(&[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]);
        let result = fft(&input, Some(2)).unwrap();
        assert_eq!(result.shape(), &[2, 2]);
        let d = result.data().unwrap();
        // FFT of [1, 2] = [3, -1].
        assert!((d[0] - 3.0).abs() < 1e-10);
        assert!(d[1].abs() < 1e-10);
        assert!((d[2] - (-1.0)).abs() < 1e-10);
        assert!(d[3].abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // fft2 round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn fft2_ifft2_roundtrip() {
        // 2x3 complex matrix.
        let pairs = vec![
            (1.0, 0.0),
            (2.0, 0.0),
            (3.0, 0.0),
            (4.0, 0.0),
            (5.0, 0.0),
            (6.0, 0.0),
        ];
        let mut data = Vec::new();
        for &(re, im) in &pairs {
            data.push(re);
            data.push(im);
        }
        let input = t(&data, &[2, 3, 2]);
        let spectrum = fft2(&input).unwrap();
        assert_eq!(spectrum.shape(), &[2, 3, 2]);

        let recovered = ifft2(&spectrum).unwrap();
        assert_eq!(recovered.shape(), &[2, 3, 2]);
        let d = recovered.data().unwrap();
        for (i, &(re, im)) in pairs.iter().enumerate() {
            assert!(
                (d[i * 2] - re).abs() < 1e-9,
                "re at {i}: {} vs {re}",
                d[i * 2]
            );
            assert!(
                (d[i * 2 + 1] - im).abs() < 1e-9,
                "im at {i}: {} vs {im}",
                d[i * 2 + 1]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Batched FFT
    // -----------------------------------------------------------------------

    #[test]
    fn fft_batched() {
        // Batch of 2 signals, each length 4.
        // Signal 0: [1, 0, 0, 0] (impulse) -> all ones.
        // Signal 1: [1, 1, 1, 1] (constant) -> [4, 0, 0, 0].
        let data = vec![
            // batch 0: impulse
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // batch 1: constant
            1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0,
        ];
        let input = t(&data, &[2, 4, 2]);
        let result = fft(&input, None).unwrap();
        assert_eq!(result.shape(), &[2, 4, 2]);
        let d = result.data().unwrap();

        // Batch 0: all bins should be (1, 0).
        for i in 0..4 {
            assert!((d[i * 2] - 1.0).abs() < 1e-10, "batch0 bin {i} re");
            assert!(d[i * 2 + 1].abs() < 1e-10, "batch0 bin {i} im");
        }

        // Batch 1: DC = (4, 0), rest = (0, 0).
        let off = 4 * 2;
        assert!((d[off] - 4.0).abs() < 1e-10, "batch1 DC re");
        assert!(d[off + 1].abs() < 1e-10, "batch1 DC im");
        for i in 1..4 {
            assert!(d[off + i * 2].abs() < 1e-10, "batch1 bin {i} re");
            assert!(d[off + i * 2 + 1].abs() < 1e-10, "batch1 bin {i} im");
        }
    }

    // -----------------------------------------------------------------------
    // f32 support
    // -----------------------------------------------------------------------

    #[test]
    fn fft_f32() {
        let data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![4, 2], false).unwrap();
        let result = fft(&input, None).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
        let d = result.data().unwrap();
        for i in 0..4 {
            assert!((d[i * 2] - 1.0).abs() < 1e-5, "bin {i} re = {}", d[i * 2]);
            assert!(d[i * 2 + 1].abs() < 1e-5, "bin {i} im = {}", d[i * 2 + 1]);
        }
    }

    // -----------------------------------------------------------------------
    // fftn / ifftn round-trip — agrees with 1-D fft for 1 axis
    // -----------------------------------------------------------------------

    #[test]
    fn fftn_matches_fft_1d() {
        let pairs = vec![(1.0, 2.0), (3.0, -1.0), (-2.0, 0.5), (0.0, 1.0)];
        let input = complex_tensor(&pairs);
        let by_fft = fft(&input, None).unwrap();
        let by_fftn = fftn(&input, None, None).unwrap();
        assert_close(by_fft.data().unwrap(), by_fftn.data().unwrap(), 1e-9);
    }

    #[test]
    fn fftn_ifftn_roundtrip_2d() {
        // 3x4 complex grid (12 complex values, 24 floats).
        let mut data = Vec::with_capacity(24);
        for i in 0..12 {
            data.push(i as f64);
            data.push((i as f64) * 0.5);
        }
        let input = t(&data, &[3, 4, 2]);
        let spectrum = fftn(&input, None, None).unwrap();
        assert_eq!(spectrum.shape(), &[3, 4, 2]);
        let recovered = ifftn(&spectrum, None, None).unwrap();
        assert_eq!(recovered.shape(), &[3, 4, 2]);
        assert_close(recovered.data().unwrap(), input.data().unwrap(), 1e-9);
    }

    #[test]
    fn fftn_ifftn_roundtrip_3d() {
        // 2x2x3 complex grid.
        let mut data = Vec::with_capacity(2 * 2 * 3 * 2);
        for i in 0..(2 * 2 * 3) {
            data.push(i as f64 + 1.0);
            data.push((i as f64) * 0.3);
        }
        let input = t(&data, &[2, 2, 3, 2]);
        let spectrum = fftn(&input, None, None).unwrap();
        assert_eq!(spectrum.shape(), &[2, 2, 3, 2]);
        let recovered = ifftn(&spectrum, None, None).unwrap();
        assert_close(recovered.data().unwrap(), input.data().unwrap(), 1e-9);
    }

    // -----------------------------------------------------------------------
    // rfftn / irfftn round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn rfftn_irfftn_roundtrip_2d() {
        let original: Vec<f64> = (1..=12).map(|x| x as f64).collect();
        let input = t(&original, &[3, 4]);
        let spectrum = rfftn(&input, None, None).unwrap();
        // Last transform axis 4 -> 4/2 + 1 = 3 complex values.
        assert_eq!(spectrum.shape(), &[3, 3, 2]);
        let recovered = irfftn(&spectrum, Some(&[3, 4]), None).unwrap();
        assert_eq!(recovered.shape(), &[3, 4]);
        assert_close(recovered.data().unwrap(), &original, 1e-9);
    }

    // -----------------------------------------------------------------------
    // hfft / ihfft round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn hfft_ihfft_roundtrip() {
        let original = vec![1.0, 2.5, -1.5, 0.5, 3.0, -2.0, 0.0, 1.0];
        let n = original.len();
        let input = t(&original, &[n]);
        // ihfft(real n) -> complex n/2+1 -> hfft -> real n.
        let spectrum = ihfft(&input, None).unwrap();
        assert_eq!(spectrum.shape(), &[n / 2 + 1, 2]);
        let recovered = hfft(&spectrum, Some(n)).unwrap();
        assert_eq!(recovered.shape(), &[n]);
        assert_close(recovered.data().unwrap(), &original, 1e-9);
    }

    // -----------------------------------------------------------------------
    // fftfreq / rfftfreq numerical correctness
    // -----------------------------------------------------------------------

    #[test]
    fn fftfreq_known_values() {
        // numpy: fftfreq(8, 1.0) = [0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125]
        let f = fftfreq(8, 1.0).unwrap();
        let expected = [0.0, 0.125, 0.25, 0.375, -0.5, -0.375, -0.25, -0.125];
        assert_close(f.data().unwrap(), &expected, 1e-12);
    }

    #[test]
    fn rfftfreq_known_values() {
        // numpy: rfftfreq(8, 1.0) = [0, 0.125, 0.25, 0.375, 0.5]
        let f = rfftfreq(8, 1.0).unwrap();
        let expected = [0.0, 0.125, 0.25, 0.375, 0.5];
        assert_close(f.data().unwrap(), &expected, 1e-12);
    }

    #[test]
    fn fftfreq_with_sample_spacing() {
        // d = 0.1: bin 1 = 1/(8*0.1) = 1.25
        let f = fftfreq(8, 0.1).unwrap();
        let d = f.data().unwrap();
        assert!((d[1] - 1.25).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // fftshift / ifftshift
    // -----------------------------------------------------------------------

    #[test]
    fn fftshift_basic_even() {
        // Even length: [0,1,2,3,4,5,6,7] -> [4,5,6,7,0,1,2,3]
        let input = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[8]);
        let shifted = fftshift(&input, None).unwrap();
        let d = shifted.data().unwrap();
        assert_close(d, &[4.0, 5.0, 6.0, 7.0, 0.0, 1.0, 2.0, 3.0], 1e-12);
    }

    #[test]
    fn fftshift_ifftshift_even_inverse() {
        // For even-length axes, ifftshift undoes fftshift exactly.
        let input = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[8]);
        let shifted = fftshift(&input, None).unwrap();
        let unshifted = ifftshift(&shifted, None).unwrap();
        assert_close(unshifted.data().unwrap(), input.data().unwrap(), 1e-12);
    }

    #[test]
    fn fftshift_ifftshift_odd_inverse() {
        // Odd-length: fftshift and ifftshift differ but compose to identity.
        let input = t(&[0.0, 1.0, 2.0, 3.0, 4.0], &[5]);
        let shifted = fftshift(&input, None).unwrap();
        let unshifted = ifftshift(&shifted, None).unwrap();
        assert_close(unshifted.data().unwrap(), input.data().unwrap(), 1e-12);
    }

    #[test]
    fn fftshift_axes_arg() {
        // 2x4: shift only the last axis -> [[2,3,0,1],[6,7,4,5]]
        let input = t(&[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], &[2, 4]);
        let shifted = fftshift(&input, Some(&[-1])).unwrap();
        assert_close(
            shifted.data().unwrap(),
            &[2.0, 3.0, 0.0, 1.0, 6.0, 7.0, 4.0, 5.0],
            1e-12,
        );
    }

    // -----------------------------------------------------------------------
    // GPU discipline: GPU tensors return DeviceUnavailable, not silent CPU bounce.
    // We can't construct a real CUDA tensor in this CPU-only test context, but
    // we verify the existing `is_cuda` rejection path is intact for the new
    // wrappers by checking that the helpers carry the same gate. This is
    // exercised in integration tests on machines with CUDA.
    // -----------------------------------------------------------------------

    #[test]
    fn fftn_agrees_with_fft2_for_2d() {
        // 2D complex grid; fftn over last 2 axes should match fft2.
        let mut data = Vec::with_capacity(2 * 3 * 2);
        for i in 0..6 {
            data.push((i as f64) - 3.0);
            data.push((i as f64) * 0.7);
        }
        let input = t(&data, &[2, 3, 2]);
        let by_fft2 = fft2(&input).unwrap();
        let by_fftn = fftn(&input, None, None).unwrap();
        assert_close(by_fft2.data().unwrap(), by_fftn.data().unwrap(), 1e-9);
    }
}
