//! Linear algebra operations bridging to ferray-linalg.
//!
//! Constructs ferray `Array` views from tensor data slices, calls
//! ferray-linalg operations, and wraps the results back into tensors.

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Matrix multiplication: C = A @ B.
///
/// Supports:
/// - 2D x 2D: standard matrix multiply (M,K) @ (K,N) -> (M,N)
/// - 1D x 1D: dot product -> scalar
/// - 2D x 1D: matrix-vector multiply (M,K) @ (K,) -> (M,)
/// - 1D x 2D: vector-matrix multiply (K,) @ (K,N) -> (N,)
pub fn matmul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    match (a.ndim(), b.ndim()) {
        (1, 1) => dot(a, b),
        (2, 1) => mv(a, b),
        (1, 2) => vm(a, b),
        (2, 2) => mm(a, b),
        _ => Err(FerrotorchError::InvalidArgument {
            message: format!(
                "matmul not supported for shapes {:?} and {:?} (batch matmul not yet implemented)",
                a.shape(),
                b.shape()
            ),
        }),
    }
}

/// Dot product of two 1-D tensors -> scalar.
pub fn dot<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("dot requires 1-D tensors, got {:?} and {:?}", a.shape(), b.shape()),
        });
    }
    if a.shape()[0] != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "dot product dimension mismatch: {} vs {}",
                a.shape()[0],
                b.shape()[0]
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let result = a_data
        .iter()
        .zip(b_data.iter())
        .fold(<T as num_traits::Zero>::zero(), |acc, (&x, &y)| acc + x * y);

    Tensor::from_storage(TensorStorage::cpu(vec![result]), vec![], false)
}

/// Matrix-matrix multiply: (M,K) @ (K,N) -> (M,N).
pub fn mm<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("mm requires 2-D tensors, got {:?} and {:?}", a.shape(), b.shape()),
        });
    }

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    if k != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mm: inner dimensions mismatch: ({},{}) @ ({},{})",
                m, k, b.shape()[0], n
            ),
        });
    }

    // Delegate to ferray-linalg (faer-backed optimized BLAS).
    // LinalgFloat is only f32/f64 — dispatch based on TypeId to avoid
    // unnecessary f64 round-trips for f32 data.
    let a_data = a.data()?;
    let b_data = b.data()?;

    let result = if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
        // Direct f32 path — reinterpret the T slice as f32 without going
        // through to_f64().unwrap() as f32 for every element.
        // SAFETY: We confirmed T is f32 via TypeId check above, so the cast
        // from *const T to *const f32 is sound and layout-compatible.
        let a_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const f32, a_data.len()) };
        let b_f32: &[f32] =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const f32, b_data.len()) };
        let a_arr = ferray_core::Array::from_vec(ferray_core::IxDyn::new(&[m, k]), a_f32.to_vec())
            .map_err(FerrotorchError::Ferray)?;
        let b_arr = ferray_core::Array::from_vec(ferray_core::IxDyn::new(&[k, n]), b_f32.to_vec())
            .map_err(FerrotorchError::Ferray)?;
        let r = ferray_linalg::matmul(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        // Reinterpret the f32 result slice back as T (which is f32).
        let r_slice = r.as_slice().unwrap();
        // SAFETY: T is f32, so casting *const f32 to *const T is sound.
        let r_as_t: &[T] =
            unsafe { std::slice::from_raw_parts(r_slice.as_ptr() as *const T, r_slice.len()) };
        r_as_t.to_vec()
    } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
        // Direct f64 path — reinterpret the T slice as f64 without conversion.
        // SAFETY: We confirmed T is f64 via TypeId check above.
        let a_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(a_data.as_ptr() as *const f64, a_data.len()) };
        let b_f64: &[f64] =
            unsafe { std::slice::from_raw_parts(b_data.as_ptr() as *const f64, b_data.len()) };
        let a_arr = ferray_core::Array::from_vec(ferray_core::IxDyn::new(&[m, k]), a_f64.to_vec())
            .map_err(FerrotorchError::Ferray)?;
        let b_arr = ferray_core::Array::from_vec(ferray_core::IxDyn::new(&[k, n]), b_f64.to_vec())
            .map_err(FerrotorchError::Ferray)?;
        let r = ferray_linalg::matmul(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        // Reinterpret the f64 result slice back as T (which is f64).
        let r_slice = r.as_slice().unwrap();
        // SAFETY: T is f64, so casting *const f64 to *const T is sound.
        let r_as_t: &[T] =
            unsafe { std::slice::from_raw_parts(r_slice.as_ptr() as *const T, r_slice.len()) };
        r_as_t.to_vec()
    } else {
        // Fallback for any other Float type: go through f64.
        let a_f64: Vec<f64> = a_data.iter().map(|&v| v.to_f64().unwrap()).collect();
        let b_f64: Vec<f64> = b_data.iter().map(|&v| v.to_f64().unwrap()).collect();
        let a_arr = ferray_core::Array::from_vec(ferray_core::IxDyn::new(&[m, k]), a_f64)
            .map_err(FerrotorchError::Ferray)?;
        let b_arr = ferray_core::Array::from_vec(ferray_core::IxDyn::new(&[k, n]), b_f64)
            .map_err(FerrotorchError::Ferray)?;
        let r = ferray_linalg::matmul(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        r.as_slice().unwrap().iter().map(|&v| T::from(v).unwrap()).collect::<Vec<T>>()
    };

    Tensor::from_storage(TensorStorage::cpu(result), vec![m, n], false)
}

/// Matrix-vector multiply: (M,K) @ (K,) -> (M,).
pub fn mv<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("mv requires (2-D, 1-D), got {:?} and {:?}", a.shape(), b.shape()),
        });
    }

    let m = a.shape()[0];
    let k = a.shape()[1];

    if k != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("mv: dimension mismatch: ({},{}) @ ({},)", m, k, b.shape()[0]),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); m];

    for i in 0..m {
        let mut acc = <T as num_traits::Zero>::zero();
        for p in 0..k {
            acc = acc + a_data[i * k + p] * b_data[p];
        }
        result[i] = acc;
    }

    Tensor::from_storage(TensorStorage::cpu(result), vec![m], false)
}

/// Vector-matrix multiply: (K,) @ (K,N) -> (N,).
fn vm<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let k = a.shape()[0];
    let n = b.shape()[1];

    if k != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("vm: dimension mismatch: ({},) @ ({},{})", k, b.shape()[0], n),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); n];

    for j in 0..n {
        let mut acc = <T as num_traits::Zero>::zero();
        for p in 0..k {
            acc = acc + a_data[p] * b_data[p * n + j];
        }
        result[j] = acc;
    }

    Tensor::from_storage(TensorStorage::cpu(result), vec![n], false)
}

/// Batched matrix multiply: [B, M, K] @ [B, K, N] -> [B, M, N].
///
/// Loops over the batch dimension and calls `mm` for each slice.
pub fn bmm<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 3 || b.ndim() != 3 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "bmm requires 3-D tensors, got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }

    let batch = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];
    let n = b.shape()[2];

    if b.shape()[0] != batch {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "bmm: batch dimensions mismatch: {} vs {}",
                batch,
                b.shape()[0]
            ),
        });
    }
    if k != b.shape()[1] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "bmm: inner dimensions mismatch: ({},{},{}) @ ({},{},{})",
                batch, m, k, b.shape()[0], b.shape()[1], n
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let slice_a = m * k;
    let slice_b = k * n;
    let slice_c = m * n;
    let mut result = vec![<T as num_traits::Zero>::zero(); batch * slice_c];

    for bi in 0..batch {
        let a_off = bi * slice_a;
        let b_off = bi * slice_b;
        let c_off = bi * slice_c;
        for i in 0..m {
            for j in 0..n {
                let mut acc = <T as num_traits::Zero>::zero();
                for p in 0..k {
                    acc = acc + a_data[a_off + i * k + p] * b_data[b_off + p * n + j];
                }
                result[c_off + i * n + j] = acc;
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(result), vec![batch, m, n], false)
}

/// Transpose a 2-D tensor.
pub fn transpose<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("transpose requires 2-D tensor, got {:?}", input.shape()),
        });
    }

    let m = input.shape()[0];
    let n = input.shape()[1];
    let data = input.data()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); m * n];

    for i in 0..m {
        for j in 0..n {
            result[j * m + i] = data[i * n + j];
        }
    }

    Tensor::from_storage(TensorStorage::cpu(result), vec![n, m], false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_dot() {
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0, 6.0], &[3]);
        let c = dot(&a, &b).unwrap();
        assert!(c.is_scalar());
        assert!((c.item().unwrap() - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_mm() {
        // [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = mm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
        let d = c.data().unwrap();
        assert!((d[0] - 19.0).abs() < 1e-6);
        assert!((d[1] - 22.0).abs() < 1e-6);
        assert!((d[2] - 43.0).abs() < 1e-6);
        assert!((d[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_mv() {
        // [[1, 2], [3, 4]] @ [5, 6] = [17, 39]
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0], &[2]);
        let c = mv(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2]);
        let d = c.data().unwrap();
        assert!((d[0] - 17.0).abs() < 1e-6);
        assert!((d[1] - 39.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_dispatch() {
        // 1D x 1D -> dot
        let a = t(&[1.0, 2.0, 3.0], &[3]);
        let b = t(&[4.0, 5.0, 6.0], &[3]);
        let c = matmul(&a, &b).unwrap();
        assert!(c.is_scalar());

        // 2D x 2D -> mm
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_mm_shape_mismatch() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        assert!(mm(&a, &b).is_err());
    }

    #[test]
    fn test_transpose() {
        // [[1, 2, 3], [4, 5, 6]] -> [[1, 4], [2, 5], [3, 6]]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let b = transpose(&a).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    // -------------------------------------------------------------------
    // bmm tests
    // -------------------------------------------------------------------

    #[test]
    fn test_bmm_forward_shape() {
        // [2, 3, 4] @ [2, 4, 5] -> [2, 3, 5]
        let a = t(&vec![1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&vec![1.0; 2 * 4 * 5], &[2, 4, 5]);
        let c = bmm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3, 5]);
    }

    #[test]
    fn test_bmm_forward_correctness() {
        // Batch 0: [[1, 2], [3, 4]] @ [[5, 6], [7, 8]] = [[19, 22], [43, 50]]
        // Batch 1: [[1, 0], [0, 1]] @ [[9, 10], [11, 12]] = [[9, 10], [11, 12]]
        #[rustfmt::skip]
        let a_data: Vec<f32> = vec![
            // batch 0
            1.0, 2.0, 3.0, 4.0,
            // batch 1 (identity)
            1.0, 0.0, 0.0, 1.0,
        ];
        #[rustfmt::skip]
        let b_data: Vec<f32> = vec![
            // batch 0
            5.0, 6.0, 7.0, 8.0,
            // batch 1
            9.0, 10.0, 11.0, 12.0,
        ];
        let a = t(&a_data, &[2, 2, 2]);
        let b = t(&b_data, &[2, 2, 2]);
        let c = bmm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        let d = c.data().unwrap();
        // batch 0
        assert!((d[0] - 19.0).abs() < 1e-6);
        assert!((d[1] - 22.0).abs() < 1e-6);
        assert!((d[2] - 43.0).abs() < 1e-6);
        assert!((d[3] - 50.0).abs() < 1e-6);
        // batch 1 (identity @ B = B)
        assert!((d[4] - 9.0).abs() < 1e-6);
        assert!((d[5] - 10.0).abs() < 1e-6);
        assert!((d[6] - 11.0).abs() < 1e-6);
        assert!((d[7] - 12.0).abs() < 1e-6);
    }

    #[test]
    fn test_bmm_batch_size_1() {
        // Single batch should behave like mm.
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let b = t(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]);
        let c = bmm(&a, &b).unwrap();
        assert_eq!(c.shape(), &[1, 2, 2]);

        let d = c.data().unwrap();
        // Same result as mm: [[19, 22], [43, 50]]
        assert!((d[0] - 19.0).abs() < 1e-6);
        assert!((d[1] - 22.0).abs() < 1e-6);
        assert!((d[2] - 43.0).abs() < 1e-6);
        assert!((d[3] - 50.0).abs() < 1e-6);
    }

    #[test]
    fn test_bmm_shape_mismatch() {
        // Batch dimension mismatch.
        let a = t(&vec![1.0; 2 * 2 * 2], &[2, 2, 2]);
        let b = t(&vec![1.0; 3 * 2 * 2], &[3, 2, 2]);
        assert!(bmm(&a, &b).is_err());

        // Inner dimension mismatch.
        let a = t(&vec![1.0; 2 * 2 * 3], &[2, 2, 3]);
        let b = t(&vec![1.0; 2 * 4 * 2], &[2, 4, 2]);
        assert!(bmm(&a, &b).is_err());

        // Wrong ndim.
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&vec![1.0; 1 * 2 * 2], &[1, 2, 2]);
        assert!(bmm(&a, &b).is_err());
    }
}
