//! Advanced linear algebra operations bridging to ferray-linalg.
//!
//! Complements `ops::linalg` (matmul, mm, mv, dot, bmm, transpose) with
//! decompositions, solvers, norms, and related functions. Each delegates to
//! the corresponding ferray-linalg routine via the same Array bridge pattern.
//!
//! **Backward support**: These operations currently return non-grad tensors.
//! Gradient functions for SVD, solve, and others will be added in a future
//! pass (the math is well-known but complex).

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

// ---------------------------------------------------------------------------
// Helper: convert Tensor<T> data to ferray Array<T, Ix2> or IxDyn
// ---------------------------------------------------------------------------

/// Build a ferray `Array<f64, Ix2>` from a 2-D tensor's data (f64 path).
fn tensor_to_array2_f64<T: Float>(
    t: &Tensor<T>,
) -> FerrotorchResult<ferray_core::Array<f64, ferray_core::Ix2>> {
    let shape = t.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("expected 2-D tensor, got {:?}", shape),
        });
    }
    let data: Vec<f64> = t.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();
    ferray_core::Array::from_vec(ferray_core::Ix2::new([shape[0], shape[1]]), data)
        .map_err(FerrotorchError::Ferray)
}

/// Build a ferray `Array<f32, Ix2>` from a 2-D tensor's data (f32 path).
fn tensor_to_array2_f32<T: Float>(
    t: &Tensor<T>,
) -> FerrotorchResult<ferray_core::Array<f32, ferray_core::Ix2>> {
    let shape = t.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("expected 2-D tensor, got {:?}", shape),
        });
    }
    let data: Vec<f32> = t.data()?.iter().map(|&v| v.to_f64().unwrap() as f32).collect();
    ferray_core::Array::from_vec(ferray_core::Ix2::new([shape[0], shape[1]]), data)
        .map_err(FerrotorchError::Ferray)
}

/// Build a ferray `Array<f64, IxDyn>` from a tensor's data (any dimensionality).
fn tensor_to_arraydyn_f64<T: Float>(
    t: &Tensor<T>,
) -> FerrotorchResult<ferray_core::Array<f64, ferray_core::IxDyn>> {
    let data: Vec<f64> = t.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();
    ferray_core::Array::from_vec(ferray_core::IxDyn::new(t.shape()), data)
        .map_err(FerrotorchError::Ferray)
}

/// Build a ferray `Array<f32, IxDyn>` from a tensor's data (any dimensionality).
fn tensor_to_arraydyn_f32<T: Float>(
    t: &Tensor<T>,
) -> FerrotorchResult<ferray_core::Array<f32, ferray_core::IxDyn>> {
    let data: Vec<f32> = t.data()?.iter().map(|&v| v.to_f64().unwrap() as f32).collect();
    ferray_core::Array::from_vec(ferray_core::IxDyn::new(t.shape()), data)
        .map_err(FerrotorchError::Ferray)
}

/// Convert a slice of f64 back to `Vec<T>`.
fn slice_to_vec<T: Float>(s: &[f64]) -> Vec<T> {
    s.iter().map(|&v| T::from(v).unwrap()).collect()
}

/// Convert a slice of f32 back to `Vec<T>`.
fn slice_f32_to_vec<T: Float>(s: &[f32]) -> Vec<T> {
    s.iter().map(|&v| T::from(v).unwrap()).collect()
}

/// True when `T` is f32 (4-byte float), used to pick the f32 vs f64 path.
fn is_f32<T: Float>() -> bool {
    std::mem::size_of::<T>() == 4
}

/// Guard: linalg decompositions are CPU-only. Return an explicit error for
/// GPU tensors instead of silently downloading data to host.
fn require_cpu<T: Float>(t: &Tensor<T>, op: &str) -> FerrotorchResult<()> {
    if t.is_cuda() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: GPU tensors are not supported for linalg decompositions. \
                 Call `.cpu()` explicitly before calling `{op}`."
            ),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Singular Value Decomposition
// ---------------------------------------------------------------------------

/// Singular Value Decomposition: `A = U @ diag(S) @ Vh`.
///
/// Returns `(U, S, Vh)` where `U` and `Vh` are unitary and `S` contains
/// singular values in descending order. Uses reduced (thin) SVD.
///
/// # Backward
/// Not yet implemented. Returns non-grad tensors.
pub fn svd<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>, Tensor<T>)> {
    require_cpu(input, "svd")?;
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("svd requires a 2-D tensor, got {:?}", shape),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let (u, s, vh) = ferray_linalg::svd(&arr, false).map_err(FerrotorchError::Ferray)?;
        let u_data = slice_f32_to_vec::<T>(u.as_slice().unwrap());
        let s_data = slice_f32_to_vec::<T>(s.as_slice().unwrap());
        let vh_data = slice_f32_to_vec::<T>(vh.as_slice().unwrap());
        let u_shape = u.shape().to_vec();
        let s_shape = s.shape().to_vec();
        let vh_shape = vh.shape().to_vec();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(u_data), u_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(s_data), s_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(vh_data), vh_shape, false)?,
        ))
    } else {
        let arr = tensor_to_array2_f64(input)?;
        let (u, s, vh) = ferray_linalg::svd(&arr, false).map_err(FerrotorchError::Ferray)?;
        let u_data = slice_to_vec::<T>(u.as_slice().unwrap());
        let s_data = slice_to_vec::<T>(s.as_slice().unwrap());
        let vh_data = slice_to_vec::<T>(vh.as_slice().unwrap());
        let u_shape = u.shape().to_vec();
        let s_shape = s.shape().to_vec();
        let vh_shape = vh.shape().to_vec();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(u_data), u_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(s_data), s_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(vh_data), vh_shape, false)?,
        ))
    }
}

// ---------------------------------------------------------------------------
// Solve linear system
// ---------------------------------------------------------------------------

/// Solve the linear system `A @ x = b`.
///
/// `a` must be a square 2-D tensor. `b` can be 1-D (single RHS) or 2-D
/// (multiple RHS columns).
///
/// # Backward
/// Not yet implemented. Returns non-grad tensors.
pub fn solve<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "solve")?;
    require_cpu(b, "solve")?;
    if a.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("solve: `a` must be 2-D, got {:?}", a.shape()),
        });
    }
    if a.shape()[0] != a.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "solve: `a` must be square, got {}x{}",
                a.shape()[0],
                a.shape()[1]
            ),
        });
    }

    if is_f32::<T>() {
        let a_arr = tensor_to_array2_f32(a)?;
        let b_arr = tensor_to_arraydyn_f32(b)?;
        let x = ferray_linalg::solve(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        let x_data = slice_f32_to_vec::<T>(x.as_slice().unwrap());
        let x_shape = x.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(x_data), x_shape, false)
    } else {
        let a_arr = tensor_to_array2_f64(a)?;
        let b_arr = tensor_to_arraydyn_f64(b)?;
        let x = ferray_linalg::solve(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        let x_data = slice_to_vec::<T>(x.as_slice().unwrap());
        let x_shape = x.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(x_data), x_shape, false)
    }
}

// ---------------------------------------------------------------------------
// Determinant
// ---------------------------------------------------------------------------

/// Matrix determinant of a square 2-D tensor.
///
/// Returns a scalar tensor.
///
/// # Backward
/// Not yet implemented. Returns non-grad tensors.
pub fn det<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(input, "det")?;
    let shape = input.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("det requires a square 2-D tensor, got {:?}", shape),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let d: f32 = ferray_linalg::det(&arr).map_err(FerrotorchError::Ferray)?;
        let val = T::from(d).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else {
        let arr = tensor_to_array2_f64(input)?;
        let d: f64 = ferray_linalg::det(&arr).map_err(FerrotorchError::Ferray)?;
        let val = T::from(d).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    }
}

// ---------------------------------------------------------------------------
// Matrix inverse
// ---------------------------------------------------------------------------

/// Matrix inverse of a square 2-D tensor.
///
/// # Backward
/// Not yet implemented. Returns non-grad tensors.
pub fn inv<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(input, "inv")?;
    let shape = input.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("inv requires a square 2-D tensor, got {:?}", shape),
        });
    }

    let n = shape[0];

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let r = ferray_linalg::inv(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else {
        let arr = tensor_to_array2_f64(input)?;
        let r = ferray_linalg::inv(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    }
}

// ---------------------------------------------------------------------------
// QR decomposition
// ---------------------------------------------------------------------------

/// QR decomposition: `A = Q @ R`.
///
/// Returns `(Q, R)` in reduced form.
///
/// # Backward
/// Not yet implemented. Returns non-grad tensors.
pub fn qr<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    require_cpu(input, "qr")?;
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("qr requires a 2-D tensor, got {:?}", shape),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let (q, r) = ferray_linalg::qr(&arr, ferray_linalg::QrMode::Reduced)
            .map_err(FerrotorchError::Ferray)?;
        let q_data = slice_f32_to_vec::<T>(q.as_slice().unwrap());
        let r_data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        let q_shape = q.shape().to_vec();
        let r_shape = r.shape().to_vec();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(q_data), q_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(r_data), r_shape, false)?,
        ))
    } else {
        let arr = tensor_to_array2_f64(input)?;
        let (q, r) = ferray_linalg::qr(&arr, ferray_linalg::QrMode::Reduced)
            .map_err(FerrotorchError::Ferray)?;
        let q_data = slice_to_vec::<T>(q.as_slice().unwrap());
        let r_data = slice_to_vec::<T>(r.as_slice().unwrap());
        let q_shape = q.shape().to_vec();
        let r_shape = r.shape().to_vec();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(q_data), q_shape, false)?,
            Tensor::from_storage(TensorStorage::cpu(r_data), r_shape, false)?,
        ))
    }
}

// ---------------------------------------------------------------------------
// Cholesky decomposition
// ---------------------------------------------------------------------------

/// Cholesky decomposition of a symmetric positive-definite matrix.
///
/// Returns the lower-triangular factor `L` such that `A = L @ L^T`.
///
/// # Backward
/// Not yet implemented. Returns non-grad tensors.
pub fn cholesky<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(input, "cholesky")?;
    let shape = input.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("cholesky requires a square 2-D tensor, got {:?}", shape),
        });
    }

    let n = shape[0];

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let l = ferray_linalg::cholesky(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(l.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else {
        let arr = tensor_to_array2_f64(input)?;
        let l = ferray_linalg::cholesky(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(l.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    }
}

// ---------------------------------------------------------------------------
// Matrix norm (Frobenius)
// ---------------------------------------------------------------------------

/// Matrix norm (Frobenius by default).
///
/// Returns a scalar tensor containing the Frobenius norm.
///
/// # Backward
/// Not yet implemented. Returns non-grad tensors.
pub fn matrix_norm<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(input, "matrix_norm")?;
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("matrix_norm requires a 2-D tensor, got {:?}", shape),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_arraydyn_f32(input)?;
        let n: f32 = ferray_linalg::norm(&arr, ferray_linalg::NormOrder::Fro)
            .map_err(FerrotorchError::Ferray)?;
        let val = T::from(n).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else {
        let arr = tensor_to_arraydyn_f64(input)?;
        let n: f64 = ferray_linalg::norm(&arr, ferray_linalg::NormOrder::Fro)
            .map_err(FerrotorchError::Ferray)?;
        let val = T::from(n).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    }
}

// ---------------------------------------------------------------------------
// Pseudo-inverse
// ---------------------------------------------------------------------------

/// Moore-Penrose pseudo-inverse of a 2-D tensor.
///
/// # Backward
/// Not yet implemented. Returns non-grad tensors.
pub fn pinv<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(input, "pinv")?;
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("pinv requires a 2-D tensor, got {:?}", shape),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let r = ferray_linalg::pinv(&arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        let r_shape = r.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(data), r_shape, false)
    } else {
        let arr = tensor_to_array2_f64(input)?;
        let r = ferray_linalg::pinv(&arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        let r_shape = r.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(data), r_shape, false)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn t(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // Helper: build a symmetric positive-definite matrix.
    // A = M^T M + I where M = [[2,1,0],[1,3,1],[0,1,2]].
    fn spd_3x3() -> Tensor<f64> {
        #[rustfmt::skip]
        let a: Vec<f64> = vec![
            6.0, 5.0, 1.0,
            5.0, 12.0, 5.0,
            1.0, 5.0, 6.0,
        ];
        t(&a, &[3, 3])
    }

    #[test]
    fn test_svd_reconstructs() {
        // A = [[1, 2], [3, 4], [5, 6]]  (3x2)
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let (u, s, vh) = svd(&a).unwrap();

        // Reconstruct: U @ diag(S) @ Vh
        let u_data = u.data().unwrap();
        let s_data = s.data().unwrap();
        let vh_data = vh.data().unwrap();
        let u_shape = u.shape();
        let vh_shape = vh.shape();

        let m = u_shape[0]; // 3
        let k = u_shape[1]; // 2 (reduced)
        let n = vh_shape[1]; // 2

        // U @ diag(S): scale columns of U by S
        let mut us = vec![0.0f64; m * k];
        for i in 0..m {
            for j in 0..k {
                us[i * k + j] = u_data[i * k + j] * s_data[j];
            }
        }

        // (US) @ Vh
        let mut recon = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += us[i * k + p] * vh_data[p * n + j];
                }
                recon[i * n + j] = acc;
            }
        }

        let a_data = a.data().unwrap();
        for i in 0..m * n {
            assert!(
                (recon[i] - a_data[i]).abs() < 1e-10,
                "SVD reconstruction failed at index {}: {} vs {}",
                i,
                recon[i],
                a_data[i]
            );
        }
    }

    #[test]
    fn test_solve_ax_eq_b() {
        // A = [[2, 1], [1, 3]], b = [5, 10]
        // Solution: x = [1, 3]  (2*1+1*3=5, 1*1+3*3=10)
        let a = t(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
        let b = t(&[5.0, 10.0], &[2]);
        let x = solve(&a, &b).unwrap();
        let x_data = x.data().unwrap();
        assert!((x_data[0] - 1.0).abs() < 1e-10);
        assert!((x_data[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_det_identity() {
        let eye = t(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        );
        let d = det(&eye).unwrap();
        assert!(d.is_scalar());
        assert!((d.item().unwrap() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inv_identity() {
        // inv(A) @ A ~ I
        let a = t(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
        let a_inv = inv(&a).unwrap();
        let a_inv_data = a_inv.data().unwrap();
        let a_data = a.data().unwrap();
        let n = 2;

        // Compute a_inv @ a
        let mut product = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += a_inv_data[i * n + k] * a_data[k * n + j];
                }
                product[i * n + j] = acc;
            }
        }

        // Should be approximately identity
        for i in 0..n {
            for j in 0..n {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (product[i * n + j] - expected).abs() < 1e-10,
                    "inv(A) @ A [{},{}] = {} (expected {})",
                    i,
                    j,
                    product[i * n + j],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_qr_reconstructs() {
        // A = [[1, 2], [3, 4], [5, 6]]
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let (q, r) = qr(&a).unwrap();
        let q_data = q.data().unwrap();
        let r_data = r.data().unwrap();
        let q_shape = q.shape();
        let r_shape = r.shape();

        let m = q_shape[0]; // 3
        let k = q_shape[1]; // 2 (reduced)
        let n = r_shape[1]; // 2

        // Reconstruct Q @ R
        let mut recon = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..k {
                    acc += q_data[i * k + p] * r_data[p * n + j];
                }
                recon[i * n + j] = acc;
            }
        }

        let a_data = a.data().unwrap();
        for i in 0..m * n {
            assert!(
                (recon[i] - a_data[i]).abs() < 1e-10,
                "QR reconstruction failed at index {}: {} vs {}",
                i,
                recon[i],
                a_data[i]
            );
        }

        // Q should be orthogonal: Q^T @ Q ~ I_k
        let mut qtq = vec![0.0f64; k * k];
        for i in 0..k {
            for j in 0..k {
                let mut acc = 0.0;
                for p in 0..m {
                    acc += q_data[p * k + i] * q_data[p * k + j];
                }
                qtq[i * k + j] = acc;
            }
        }
        for i in 0..k {
            for j in 0..k {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (qtq[i * k + j] - expected).abs() < 1e-10,
                    "Q^T Q [{},{}] = {} (expected {})",
                    i,
                    j,
                    qtq[i * k + j],
                    expected,
                );
            }
        }
    }

    #[test]
    fn test_cholesky_spd() {
        let a = spd_3x3();
        let l = cholesky(&a).unwrap();
        let l_data = l.data().unwrap();
        let n = 3;

        // Verify lower-triangular: upper entries should be zero
        for i in 0..n {
            for j in (i + 1)..n {
                assert!(
                    l_data[i * n + j].abs() < 1e-10,
                    "L[{},{}] = {} should be 0",
                    i,
                    j,
                    l_data[i * n + j]
                );
            }
        }

        // Reconstruct: L @ L^T should equal A
        let a_data = a.data().unwrap();
        let mut llt = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for p in 0..n {
                    acc += l_data[i * n + p] * l_data[j * n + p]; // L @ L^T
                }
                llt[i * n + j] = acc;
            }
        }

        for i in 0..n * n {
            assert!(
                (llt[i] - a_data[i]).abs() < 1e-10,
                "L @ L^T failed at index {}: {} vs {}",
                i,
                llt[i],
                a_data[i]
            );
        }
    }

    #[test]
    fn test_matrix_norm_identity() {
        // Frobenius norm of n x n identity = sqrt(n)
        let eye = t(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            &[3, 3],
        );
        let n = matrix_norm(&eye).unwrap();
        assert!(n.is_scalar());
        let expected = (3.0f64).sqrt();
        assert!(
            (n.item().unwrap() - expected).abs() < 1e-10,
            "Frobenius norm of 3x3 identity = {} (expected {})",
            n.item().unwrap(),
            expected,
        );
    }

    #[test]
    fn test_pinv_full_rank_square() {
        // For a full-rank square matrix, pinv(A) == inv(A)
        let a = t(&[2.0, 1.0, 1.0, 3.0], &[2, 2]);
        let a_pinv = pinv(&a).unwrap();
        let a_inv = inv(&a).unwrap();
        let pinv_data = a_pinv.data().unwrap();
        let inv_data = a_inv.data().unwrap();
        for i in 0..4 {
            assert!(
                (pinv_data[i] - inv_data[i]).abs() < 1e-10,
                "pinv vs inv at index {}: {} vs {}",
                i,
                pinv_data[i],
                inv_data[i]
            );
        }
    }
}
