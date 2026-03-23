//! FFT operations for tensors, powered by rustfft.
//!
//! Complex values are represented as an extra trailing dimension of size 2,
//! where `[..., 0]` is the real part and `[..., 1]` is the imaginary part.
//! This matches PyTorch's convention for `torch.fft.*` operations.
//!
//! All functions work on f32 and f64 tensors. Internally, computation is
//! performed in f64 via rustfft, then converted back to the input dtype.

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

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
fn fft_1d_last_axis(
    data: &mut [Complex<f64>],
    batch_shape: &[usize],
    n: usize,
    inverse: bool,
) {
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

    let device = input.device();
    let data = input.data_vec()?;
    let batch_shape = &shape[..ndim - 2];
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

    // Parse complex pairs from the flat data.
    let mut complex_data = Vec::with_capacity(batch_size * fft_n);
    for b in 0..batch_size {
        let src_offset = b * input_n * 2;
        let copy_len = input_n.min(fft_n);
        for i in 0..copy_len {
            let re = data[src_offset + i * 2].to_f64().unwrap();
            let im = data[src_offset + i * 2 + 1].to_f64().unwrap();
            complex_data.push(Complex::new(re, im));
        }
        // Zero-pad if needed.
        for _ in copy_len..fft_n {
            complex_data.push(Complex::new(0.0, 0.0));
        }
    }

    fft_1d_last_axis(&mut complex_data, batch_shape, fft_n, false);

    let result_data = complex_to_pairs::<T>(&complex_data);
    let mut out_shape = batch_shape.to_vec();
    out_shape.push(fft_n);
    out_shape.push(2);

    let out = Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)?;
    if device.is_cuda() { out.to(device) } else { Ok(out) }
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

    let device = input.device();
    let data = input.data_vec()?;
    let batch_shape = &shape[..ndim - 2];
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

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

    let out = Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)?;
    if device.is_cuda() { out.to(device) } else { Ok(out) }
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

    let device = input.device();
    let data = input.data_vec()?;
    let batch_shape = &shape[..ndim - 1];
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

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

    let out = Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)?;
    if device.is_cuda() { out.to(device) } else { Ok(out) }
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

    let device = input.device();
    let data = input.data_vec()?;
    let batch_shape = &shape[..ndim - 2];
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

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

    let out = Tensor::from_storage(TensorStorage::cpu(result_data), out_shape, false)?;
    if device.is_cuda() { out.to(device) } else { Ok(out) }
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

    // FFT along cols (last spatial dim) first, then rows.
    let rows = shape[ndim - 3];
    let cols = shape[ndim - 2];

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
    let device = input.device();
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
    let trans_tensor =
        Tensor::from_storage(TensorStorage::cpu(transposed), trans_shape, false)?;

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

    let out = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
    if device.is_cuda() { out.to(device) } else { Ok(out) }
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
        assert_eq!(a.len(), b.len(), "length mismatch: {} vs {}", a.len(), b.len());
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
        assert!((d[0] - n as f64).abs() < 1e-10, "DC re = {}, expected {n}", d[0]);
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
        let input =
            complex_tensor(&[(1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (4.0, 0.0)]);
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
            (1.0, 0.0), (2.0, 0.0), (3.0, 0.0),
            (4.0, 0.0), (5.0, 0.0), (6.0, 0.0),
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
            1.0, 0.0,  0.0, 0.0,  0.0, 0.0,  0.0, 0.0,
            // batch 1: constant
            1.0, 0.0,  1.0, 0.0,  1.0, 0.0,  1.0, 0.0,
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
        let input = Tensor::from_storage(
            TensorStorage::cpu(data),
            vec![4, 2],
            false,
        )
        .unwrap();
        let result = fft(&input, None).unwrap();
        assert_eq!(result.shape(), &[4, 2]);
        let d = result.data().unwrap();
        for i in 0..4 {
            assert!((d[i * 2] - 1.0).abs() < 1e-5, "bin {i} re = {}", d[i * 2]);
            assert!(d[i * 2 + 1].abs() < 1e-5, "bin {i} im = {}", d[i * 2 + 1]);
        }
    }
}
