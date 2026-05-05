//! `Tensor ↔ ndarray` adapter primitives.
//!
//! ferrolearn operates on `ndarray::{Array1, Array2}`. ferrotorch operates
//! on its own [`Tensor<T>`]. These functions bridge the two with a single
//! `memcpy` per call.
//!
//! # GPU tensors are auto-moved to CPU
//!
//! ferrolearn is CPU-only by design. The adapter routines accept tensors
//! on **any** device — if the input lives on a CUDA / XPU device, the
//! data is transparently moved to host memory before conversion. This
//! matches the torch idiom of `loss.cpu().item()` where a single
//! materialisation step crosses the device boundary.
//!
//! The transparent transfer is **only** done by these adapter helpers
//! — `ferrotorch-ml` is the dedicated bridge crate for sklearn-shaped
//! workloads, so the caller has already opted into the CPU world by
//! using this crate. Compute crates (`ferrotorch-core`, `-nn`, `-gpu`)
//! continue to enforce the strict `/rust-gpu-discipline` no-silent-fallback
//! rule. If you want the strict gate here too — e.g. to fail-fast in a
//! training-loop hot path — add an explicit `assert!(t.device().is_cpu())`
//! at your call site.

use ndarray::{Array1, Array2};

use ferrotorch_core::dtype::Float;
use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;

/// Convert a 1-D tensor to a 1-D ndarray. Allocates one contiguous
/// buffer (a single `memcpy`).
///
/// GPU tensors are transparently moved to host memory; see the module
/// docstring for the rationale.
///
/// # Examples
///
/// ```
/// use ferrotorch_core::tensor;
/// use ferrotorch_ml::adapter::tensor_to_array1;
///
/// let t = tensor(&[1.0_f64, 2.0, 3.0]).unwrap();
/// let arr = tensor_to_array1(&t).unwrap();
/// assert_eq!(arr.len(), 3);
/// assert_eq!(arr[0], 1.0);
/// ```
///
/// # Errors
/// - [`FerrotorchError::ShapeMismatch`] when the input is not 1-D.
pub fn tensor_to_array1<T: Float + Clone>(t: &Tensor<T>) -> FerrotorchResult<Array1<T>> {
    if t.shape().len() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "tensor_to_array1: expected 1-D tensor, got shape {:?}",
                t.shape()
            ),
        });
    }
    // `data_vec()` already routes GPU tensors through host transparently
    // (see ferrotorch-core::tensor::Tensor::data_vec); we just hand its
    // result to ndarray.
    Ok(Array1::from(t.data_vec()?))
}

/// Convert a 2-D tensor to a 2-D ndarray. Allocates one contiguous
/// buffer (a single `memcpy`). Row-major layout is preserved on both
/// sides.
///
/// GPU tensors are transparently moved to host memory; see the module
/// docstring for the rationale.
///
/// # Examples
///
/// ```
/// use ferrotorch_core::{Tensor, TensorStorage};
/// use ferrotorch_ml::adapter::tensor_to_array2;
///
/// let t = Tensor::from_storage(
///     TensorStorage::cpu(vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
///     vec![2, 3],
///     false,
/// ).unwrap();
/// let arr = tensor_to_array2(&t).unwrap();
/// assert_eq!(arr.shape(), &[2, 3]);
/// assert_eq!(arr[[1, 2]], 6.0);
/// ```
///
/// # Errors
/// - [`FerrotorchError::ShapeMismatch`] for non-2-D input.
pub fn tensor_to_array2<T: Float + Clone>(t: &Tensor<T>) -> FerrotorchResult<Array2<T>> {
    if t.shape().len() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "tensor_to_array2: expected 2-D tensor, got shape {:?}",
                t.shape()
            ),
        });
    }
    let (rows, cols) = (t.shape()[0], t.shape()[1]);
    let data = t.data_vec()?;
    Array2::from_shape_vec((rows, cols), data).map_err(|e| FerrotorchError::ShapeMismatch {
        message: format!("tensor_to_array2: shape build failed: {e}"),
    })
}

/// Convert a 1-D ndarray back to a CPU tensor.
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use ferrotorch_ml::adapter::array1_to_tensor;
///
/// let arr = Array1::from(vec![1.0_f64, 2.0, 3.0]);
/// let t = array1_to_tensor(arr).unwrap();
/// assert_eq!(t.shape(), &[3]);
/// assert_eq!(t.data().unwrap(), &[1.0, 2.0, 3.0]);
/// ```
///
/// # Errors
///
/// Returns [`FerrotorchError`] when [`Tensor::from_storage`] rejects the
/// shape (e.g. internal storage construction failure).
pub fn array1_to_tensor<T: Float>(arr: Array1<T>) -> FerrotorchResult<Tensor<T>> {
    let len = arr.len();
    let data = arr.into_raw_vec_and_offset().0;
    Tensor::from_storage(TensorStorage::cpu(data), vec![len], false)
}

/// Convert a 2-D ndarray back to a CPU tensor.
///
/// # Examples
///
/// ```
/// use ndarray::Array2;
/// use ferrotorch_ml::adapter::array2_to_tensor;
///
/// let arr = Array2::from_shape_vec((2, 3), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
/// let t = array2_to_tensor(arr).unwrap();
/// assert_eq!(t.shape(), &[2, 3]);
/// assert_eq!(t.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// ```
///
/// # Errors
///
/// Returns [`FerrotorchError`] when [`Tensor::from_storage`] rejects the
/// shape (e.g. internal storage construction failure).
pub fn array2_to_tensor<T: Float>(arr: Array2<T>) -> FerrotorchResult<Tensor<T>> {
    let (rows, cols) = arr.dim();
    // ndarray Array2 may be non-contiguous if it was sliced/transposed —
    // collect into row-major first.
    let data: Vec<T> = if arr.is_standard_layout() {
        arr.into_raw_vec_and_offset().0
    } else {
        arr.iter().copied().collect()
    };
    Tensor::from_storage(TensorStorage::cpu(data), vec![rows, cols], false)
}

/// Convert a 1-D `Array1<usize>` (sklearn label-style) into a CPU tensor
/// of `T`. Useful when ferrolearn returns class predictions and you want
/// them back inside a tensor pipeline.
///
/// # Examples
///
/// ```
/// use ndarray::Array1;
/// use ferrotorch_core::Tensor;
/// use ferrotorch_ml::adapter::array1_usize_to_tensor;
///
/// let labels = Array1::from(vec![0_usize, 1, 2, 1]);
/// let t: Tensor<f64> = array1_usize_to_tensor(labels).unwrap();
/// assert_eq!(t.data().unwrap(), &[0.0, 1.0, 2.0, 1.0]);
/// ```
///
/// # Errors
///
/// - [`FerrotorchError::InvalidArgument`] when an input label cannot be
///   represented as a finite `T` (propagated from
///   [`ferrotorch_core::numeric_cast::cast`]).
/// - [`FerrotorchError`] from [`Tensor::from_storage`] on storage build
///   failure.
pub fn array1_usize_to_tensor<T: Float>(arr: Array1<usize>) -> FerrotorchResult<Tensor<T>> {
    let data: Vec<T> = arr
        .iter()
        .map(|&i| cast::<f64, T>(i as f64))
        .collect::<FerrotorchResult<_>>()?;
    Tensor::from_storage(TensorStorage::cpu(data), vec![arr.len()], false)
}

/// Convert a 1-D tensor of class labels (encoded as floats) into an
/// `Array1<usize>` for ferrolearn classification metrics. Values are
/// rounded via `as usize` after a finite-and-non-negative check.
///
/// GPU tensors are transparently moved to host memory; see the module
/// docstring.
///
/// # Examples
///
/// ```
/// use ferrotorch_core::tensor;
/// use ferrotorch_ml::adapter::tensor_to_array1_usize;
///
/// let t = tensor(&[0.0_f64, 1.0, 2.0, 1.0]).unwrap();
/// let labels = tensor_to_array1_usize(&t).unwrap();
/// assert_eq!(labels.to_vec(), vec![0, 1, 2, 1]);
/// ```
///
/// # Errors
///
/// - [`FerrotorchError::ShapeMismatch`] when the input is not 1-D.
/// - [`FerrotorchError::InvalidArgument`] when an element is non-finite,
///   negative, or otherwise not representable as a non-negative integer.
/// - [`FerrotorchError`] propagated from [`Tensor::data_vec`] on
///   device-transfer failure.
pub fn tensor_to_array1_usize<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Array1<usize>> {
    if t.shape().len() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "tensor_to_array1_usize: expected 1-D tensor, got shape {:?}",
                t.shape()
            ),
        });
    }
    let data = t.data_vec()?;
    let mut out = Vec::with_capacity(data.len());
    for (i, &v) in data.iter().enumerate() {
        let f = v.to_f64().ok_or_else(|| FerrotorchError::InvalidArgument {
            message: format!("tensor_to_array1_usize: element {i} is not finite"),
        })?;
        if !f.is_finite() || f < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "tensor_to_array1_usize: element {i} = {f} is not a non-negative finite \
                     integer"
                ),
            });
        }
        out.push(f as usize);
    }
    Ok(Array1::from(out))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::tensor as new_tensor;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() < tol
    }

    #[test]
    fn array1_round_trip() {
        let t = new_tensor(&[1.0_f64, 2.0, 3.0, 4.0]).unwrap();
        let arr = tensor_to_array1(&t).unwrap();
        assert_eq!(arr.len(), 4);
        let back = array1_to_tensor(arr).unwrap();
        assert_eq!(back.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn array2_round_trip() {
        let data: Vec<f64> = (0..6).map(|x| x as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![2, 3], false).unwrap();
        let arr = tensor_to_array2(&t).unwrap();
        assert_eq!(arr.shape(), &[2, 3]);
        // Row 0: [0, 1, 2]; Row 1: [3, 4, 5]
        assert!(close(arr[[0, 0]], 0.0, 1e-12));
        assert!(close(arr[[0, 2]], 2.0, 1e-12));
        assert!(close(arr[[1, 0]], 3.0, 1e-12));
        let back = array2_to_tensor(arr).unwrap();
        assert_eq!(back.shape(), &[2, 3]);
        assert_eq!(back.data().unwrap(), &data);
    }

    #[test]
    fn array2_handles_transposed_view() {
        // Transposed Array2 is non-contiguous; the adapter must still
        // produce a valid tensor (via collect, not raw vec).
        let arr = Array2::from_shape_vec((2, 3), vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let trans = arr.t().to_owned();
        let t = array2_to_tensor(trans).unwrap();
        assert_eq!(t.shape(), &[3, 2]);
        // Transposed: column 0 of original = (1, 4); column 1 = (2, 5); column 2 = (3, 6)
        // Row-major flat: [1, 4, 2, 5, 3, 6]
        assert_eq!(t.data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn tensor_to_array1_rejects_2d() {
        let t =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64; 6]), vec![2, 3], false).unwrap();
        let err = tensor_to_array1(&t).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn tensor_to_array2_rejects_1d() {
        let t = new_tensor(&[1.0_f64, 2.0]).unwrap();
        let err = tensor_to_array2(&t).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn array1_usize_round_trip() {
        // Build an Array1<usize>, convert to tensor, then back via
        // tensor_to_array1_usize.
        let arr = Array1::from(vec![0_usize, 1, 2, 3]);
        let t: Tensor<f64> = array1_usize_to_tensor(arr).unwrap();
        assert_eq!(t.data().unwrap(), &[0.0, 1.0, 2.0, 3.0]);
        let back = tensor_to_array1_usize(&t).unwrap();
        assert_eq!(back.to_vec(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn tensor_to_array1_usize_rejects_negative() {
        let t = new_tensor(&[0.0_f64, 1.0, -1.0]).unwrap();
        let err = tensor_to_array1_usize(&t).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn tensor_to_array1_usize_rejects_nan() {
        let t = new_tensor(&[0.0_f64, f64::NAN, 1.0]).unwrap();
        let err = tensor_to_array1_usize(&t).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }
}
