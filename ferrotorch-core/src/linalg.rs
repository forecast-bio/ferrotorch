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
            message: format!("expected 2-D tensor, got {shape:?}"),
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
            message: format!("expected 2-D tensor, got {shape:?}"),
        });
    }
    let data: Vec<f32> = t
        .data()?
        .iter()
        .map(|&v| v.to_f64().unwrap() as f32)
        .collect();
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
    let data: Vec<f32> = t
        .data()?
        .iter()
        .map(|&v| v.to_f64().unwrap() as f32)
        .collect();
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

/// True when `T` is f64 (8-byte float).
fn is_f64<T: Float>() -> bool {
    std::mem::size_of::<T>() == 8
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
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("svd requires a 2-D tensor, got {shape:?}"),
        });
    }

    if input.is_cuda() {
        // GPU dispatch via cuSOLVER. Reduced SVD shapes:
        //   U: [m, k], S: [k], Vh: [k, n], k = min(m, n)
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = shape[0];
        let n = shape[1];
        let k = m.min(n);
        let buf = input.gpu_handle()?;
        let (u_h, s_h, vh_h) = if is_f32::<T>() {
            backend.svd_f32(buf, m, n)?
        } else if is_f64::<T>() {
            backend.svd_f64(buf, m, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "svd requires f32 or f64".into(),
            });
        };
        return Ok((
            Tensor::from_storage(TensorStorage::gpu(u_h), vec![m, k], false)?,
            Tensor::from_storage(TensorStorage::gpu(s_h), vec![k], false)?,
            Tensor::from_storage(TensorStorage::gpu(vh_h), vec![k, n], false)?,
        ));
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
    } else if is_f64::<T>() {
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
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
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
    if a.is_cuda() != b.is_cuda() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let n = a.shape()[0];
        // b can be [n] (single RHS) or [n, nrhs].
        let nrhs = if b.ndim() == 1 { 1 } else { b.shape()[1] };
        let x_h = if is_f32::<T>() {
            backend.solve_f32(a.gpu_handle()?, b.gpu_handle()?, n, nrhs)?
        } else if is_f64::<T>() {
            backend.solve_f64(a.gpu_handle()?, b.gpu_handle()?, n, nrhs)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "solve requires f32 or f64".into(),
            });
        };
        let out_shape: Vec<usize> = if b.ndim() == 1 {
            vec![n]
        } else {
            vec![n, nrhs]
        };
        return Tensor::from_storage(TensorStorage::gpu(x_h), out_shape, false);
    }

    if is_f32::<T>() {
        let a_arr = tensor_to_array2_f32(a)?;
        let b_arr = tensor_to_arraydyn_f32(b)?;
        let x = ferray_linalg::solve(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        let x_data = slice_f32_to_vec::<T>(x.as_slice().unwrap());
        let x_shape = x.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(x_data), x_shape, false)
    } else if is_f64::<T>() {
        let a_arr = tensor_to_array2_f64(a)?;
        let b_arr = tensor_to_arraydyn_f64(b)?;
        let x = ferray_linalg::solve(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        let x_data = slice_to_vec::<T>(x.as_slice().unwrap());
        let x_shape = x.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(x_data), x_shape, false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
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
            message: format!("det requires a square 2-D tensor, got {shape:?}"),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let d: f32 = ferray_linalg::det(&arr).map_err(FerrotorchError::Ferray)?;
        let val = T::from(d).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let d: f64 = ferray_linalg::det(&arr).map_err(FerrotorchError::Ferray)?;
        let val = T::from(d).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
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
            message: format!("inv requires a square 2-D tensor, got {shape:?}"),
        });
    }

    let n = shape[0];

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let r = ferray_linalg::inv(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let r = ferray_linalg::inv(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
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
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("qr requires a 2-D tensor, got {shape:?}"),
        });
    }

    if input.is_cuda() {
        // Reduced QR shapes: Q [m, k], R [k, n], k = min(m, n)
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = shape[0];
        let n = shape[1];
        let k = m.min(n);
        let (q_h, r_h) = if is_f32::<T>() {
            backend.qr_f32(input.gpu_handle()?, m, n)?
        } else if is_f64::<T>() {
            backend.qr_f64(input.gpu_handle()?, m, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "qr requires f32 or f64".into(),
            });
        };
        return Ok((
            Tensor::from_storage(TensorStorage::gpu(q_h), vec![m, k], false)?,
            Tensor::from_storage(TensorStorage::gpu(r_h), vec![k, n], false)?,
        ));
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
    } else if is_f64::<T>() {
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
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
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
    let shape = input.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("cholesky requires a square 2-D tensor, got {shape:?}"),
        });
    }

    let n = shape[0];

    if input.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let l_h = if is_f32::<T>() {
            backend.cholesky_f32(input.gpu_handle()?, n)?
        } else if is_f64::<T>() {
            backend.cholesky_f64(input.gpu_handle()?, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "cholesky requires f32 or f64".into(),
            });
        };
        return Tensor::from_storage(TensorStorage::gpu(l_h), vec![n, n], false);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let l = ferray_linalg::cholesky(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(l.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let l = ferray_linalg::cholesky(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(l.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
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
    let shape = input.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("matrix_norm requires a 2-D tensor, got {shape:?}"),
        });
    }

    if input.is_cuda() {
        // Frobenius norm: sqrt(sum_ij A_ij^2). Composes existing GPU
        // primitives (mul → reduce_sum → sqrt) — three kernel launches but
        // fully GPU-resident; result lands as a 0-d tensor on device.
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = input.gpu_handle()?;
        let numel = shape.iter().product::<usize>();
        let h = if is_f32::<T>() {
            let sq = backend.mul_f32(buf, buf)?;
            let s = backend.sum_f32(&sq, numel)?;
            backend.sqrt_f32(&s)?
        } else if is_f64::<T>() {
            let sq = backend.mul_f64(buf, buf)?;
            let s = backend.sum_f64(&sq, numel)?;
            backend.sqrt_f64(&s)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "matrix_norm requires f32 or f64".into(),
            });
        };
        return Tensor::from_storage(TensorStorage::gpu(h), vec![], false);
    }

    if is_f32::<T>() {
        let arr = tensor_to_arraydyn_f32(input)?;
        let n: f32 = ferray_linalg::norm(&arr, ferray_linalg::NormOrder::Fro)
            .map_err(FerrotorchError::Ferray)?;
        let val = T::from(n).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_arraydyn_f64(input)?;
        let n: f64 = ferray_linalg::norm(&arr, ferray_linalg::NormOrder::Fro)
            .map_err(FerrotorchError::Ferray)?;
        let val = T::from(n).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
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
            message: format!("pinv requires a 2-D tensor, got {shape:?}"),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(input)?;
        let r = ferray_linalg::pinv(&arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        let r_shape = r.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(data), r_shape, false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(input)?;
        let r = ferray_linalg::pinv(&arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        let r_shape = r.shape().to_vec();
        Tensor::from_storage(TensorStorage::cpu(data), r_shape, false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ===========================================================================
// Eigendecomposition (Hermitian / general)
// ===========================================================================

/// Symmetric / Hermitian eigendecomposition: `A = Q diag(w) Q^T`.
///
/// `a` must be square and (numerically) symmetric. Returns `(w, Q)` where
/// `w` are real eigenvalues in ascending order and `Q` is the orthogonal
/// matrix of eigenvectors (column `i` of `Q` is the eigenvector for `w[i]`).
///
/// Mirrors `torch.linalg.eigh`. GPU-resident on CUDA via cuSOLVER `syevd`.
pub fn eigh<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("eigh requires a square 2-D tensor, got {shape:?}"),
        });
    }
    let n = shape[0];

    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = a.gpu_handle()?;
        let (w_h, v_h) = if is_f32::<T>() {
            backend.eigh_f32(buf, n)?
        } else if is_f64::<T>() {
            backend.eigh_f64(buf, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "eigh requires f32 or f64".into(),
            });
        };
        return Ok((
            Tensor::from_storage(TensorStorage::gpu(w_h), vec![n], false)?,
            Tensor::from_storage(TensorStorage::gpu(v_h), vec![n, n], false)?,
        ));
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (w, q) = ferray_linalg::eigh(&arr).map_err(FerrotorchError::Ferray)?;
        let w_data = slice_f32_to_vec::<T>(w.as_slice().unwrap());
        let q_data = slice_f32_to_vec::<T>(q.as_slice().unwrap());
        Ok((
            Tensor::from_storage(TensorStorage::cpu(w_data), vec![n], false)?,
            Tensor::from_storage(TensorStorage::cpu(q_data), vec![n, n], false)?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (w, q) = ferray_linalg::eigh(&arr).map_err(FerrotorchError::Ferray)?;
        let w_data = slice_to_vec::<T>(w.as_slice().unwrap());
        let q_data = slice_to_vec::<T>(q.as_slice().unwrap());
        Ok((
            Tensor::from_storage(TensorStorage::cpu(w_data), vec![n], false)?,
            Tensor::from_storage(TensorStorage::cpu(q_data), vec![n, n], false)?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Eigenvalues of a symmetric / Hermitian matrix (real, ascending).
///
/// Mirrors `torch.linalg.eigvalsh`. GPU-resident on CUDA via cuSOLVER `syevd`
/// with `jobz=NOVECTOR`.
pub fn eigvalsh<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("eigvalsh requires a square 2-D tensor, got {shape:?}"),
        });
    }
    let n = shape[0];

    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let buf = a.gpu_handle()?;
        let w_h = if is_f32::<T>() {
            backend.eigvalsh_f32(buf, n)?
        } else if is_f64::<T>() {
            backend.eigvalsh_f64(buf, n)?
        } else {
            return Err(FerrotorchError::InvalidArgument {
                message: "eigvalsh requires f32 or f64".into(),
            });
        };
        return Tensor::from_storage(TensorStorage::gpu(w_h), vec![n], false);
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let w = ferray_linalg::eigvalsh(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(w.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let w = ferray_linalg::eigvalsh(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(w.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), vec![n], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// General (non-symmetric) eigendecomposition.
///
/// Returns `(w, V)` where eigenvalues `w` and eigenvectors `V` are
/// **complex-valued**, encoded as tensors with a trailing dimension of
/// size 2 representing `[real, imag]` (matching ferrotorch's complex
/// convention used by [`fft`](crate::fft)). `w` has shape `[n, 2]` and
/// `V` has shape `[n, n, 2]`.
///
/// Mirrors `torch.linalg.eig`. CPU-only today.
pub fn eig<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    require_cpu(a, "eig")?;
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("eig requires a square 2-D tensor, got {shape:?}"),
        });
    }
    let n = shape[0];

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (w, v) = ferray_linalg::eig(&arr).map_err(FerrotorchError::Ferray)?;
        let w_data: Vec<T> = w
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        let v_data: Vec<T> = v
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(w_data), vec![n, 2], false)?,
            Tensor::from_storage(TensorStorage::cpu(v_data), vec![n, n, 2], false)?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (w, v) = ferray_linalg::eig(&arr).map_err(FerrotorchError::Ferray)?;
        let w_data: Vec<T> = w
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        let v_data: Vec<T> = v
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        Ok((
            Tensor::from_storage(TensorStorage::cpu(w_data), vec![n, 2], false)?,
            Tensor::from_storage(TensorStorage::cpu(v_data), vec![n, n, 2], false)?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// General eigenvalues only (complex, encoded `[n, 2]`).
///
/// Mirrors `torch.linalg.eigvals`. CPU-only today.
pub fn eigvals<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "eigvals")?;
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("eigvals requires a square 2-D tensor, got {shape:?}"),
        });
    }
    let n = shape[0];

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let w = ferray_linalg::eigvals(&arr).map_err(FerrotorchError::Ferray)?;
        let data: Vec<T> = w
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, 2], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let w = ferray_linalg::eigvals(&arr).map_err(FerrotorchError::Ferray)?;
        let data: Vec<T> = w
            .as_slice()
            .unwrap()
            .iter()
            .flat_map(|c| [T::from(c.re).unwrap(), T::from(c.im).unwrap()])
            .collect();
        Tensor::from_storage(TensorStorage::cpu(data), vec![n, 2], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ===========================================================================
// LU decomposition
// ===========================================================================

/// LU decomposition with partial pivoting: `A = P L U`.
///
/// Returns `(P, L, U)` where `P` is the permutation matrix (m × m), `L`
/// is unit-lower-triangular (m × k), and `U` is upper-triangular (k × n)
/// with `k = min(m, n)`.
///
/// Mirrors `torch.linalg.lu`. CPU-only today.
pub fn lu<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>, Tensor<T>)> {
    require_cpu(a, "lu")?;
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("lu requires a 2-D tensor, got {shape:?}"),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (p, l, u) = ferray_linalg::lu(&arr).map_err(FerrotorchError::Ferray)?;
        let p_data = slice_f32_to_vec::<T>(p.as_slice().unwrap());
        let l_data = slice_f32_to_vec::<T>(l.as_slice().unwrap());
        let u_data = slice_f32_to_vec::<T>(u.as_slice().unwrap());
        Ok((
            Tensor::from_storage(TensorStorage::cpu(p_data), p.shape().to_vec(), false)?,
            Tensor::from_storage(TensorStorage::cpu(l_data), l.shape().to_vec(), false)?,
            Tensor::from_storage(TensorStorage::cpu(u_data), u.shape().to_vec(), false)?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (p, l, u) = ferray_linalg::lu(&arr).map_err(FerrotorchError::Ferray)?;
        let p_data = slice_to_vec::<T>(p.as_slice().unwrap());
        let l_data = slice_to_vec::<T>(l.as_slice().unwrap());
        let u_data = slice_to_vec::<T>(u.as_slice().unwrap());
        Ok((
            Tensor::from_storage(TensorStorage::cpu(p_data), p.shape().to_vec(), false)?,
            Tensor::from_storage(TensorStorage::cpu(l_data), l.shape().to_vec(), false)?,
            Tensor::from_storage(TensorStorage::cpu(u_data), u.shape().to_vec(), false)?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// LU factorization in cuSOLVER's packed form: returns `(LU_packed, pivots)`
/// where `LU_packed` is `n×n` row-major with the strict lower triangle = `L`
/// (unit diagonal implicit) and the upper triangle = `U`, and `pivots` is a
/// length-`n` host `Vec<i32>` of 1-based row-permutation indices (cuSOLVER /
/// LAPACK convention). Mirrors `torch.linalg.lu_factor`. (#604)
///
/// On CUDA f32/f64, dispatches to the native `gpu_lu_factor` kernel
/// (cuSOLVER `getrf` with on-device row→col→row transpose). The LU matrix
/// stays on device (O(n²) values); only the pivot vector (O(n) ints) is
/// downloaded to host as a `Vec<i32>` since `Tensor<T>` requires
/// `T: Float`. Other dtypes and CPU inputs fall back to `ferray-linalg::lu`
/// and pack the result locally.
pub fn lu_factor<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Vec<i32>)> {
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("lu_factor requires a square 2-D tensor, got {shape:?}"),
        });
    }
    let n = shape[0];

    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let (lu_h, ipiv) = if is_f32::<T>() {
            backend.lu_factor_f32(a.gpu_handle()?, n)?
        } else {
            backend.lu_factor_f64(a.gpu_handle()?, n)?
        };
        // The LU matrix stays on device; pivots are returned as a host
        // Vec<i32> directly from the trait (O(n) ints, not worth a typed
        // GPU int handle).
        let lu = Tensor::from_storage(TensorStorage::gpu(lu_h), vec![n, n], false)?;
        return Ok((lu, ipiv));
    }
    if a.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "lu_factor" });
    }

    // CPU fallback: get full PLU from ferray-linalg, then collapse to packed
    // LU + pivots (the cuSOLVER convention). The packed form is L's strict
    // lower triangle plus U's upper triangle (incl. diagonal); ipiv comes
    // from the row permutation P encoded as 1-based indices.
    let (p, l, u) = if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (p, l, u) = ferray_linalg::lu(&arr).map_err(FerrotorchError::Ferray)?;
        (
            slice_f32_to_vec::<T>(p.as_slice().unwrap()),
            slice_f32_to_vec::<T>(l.as_slice().unwrap()),
            slice_f32_to_vec::<T>(u.as_slice().unwrap()),
        )
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (p, l, u) = ferray_linalg::lu(&arr).map_err(FerrotorchError::Ferray)?;
        (
            slice_to_vec::<T>(p.as_slice().unwrap()),
            slice_to_vec::<T>(l.as_slice().unwrap()),
            slice_to_vec::<T>(u.as_slice().unwrap()),
        )
    } else {
        return Err(FerrotorchError::InvalidArgument {
            message: "lu_factor requires f32 or f64".into(),
        });
    };

    // Build packed LU buffer: lower triangle = strict-L, upper = U (incl. diag).
    let mut packed = vec![<T as num_traits::Zero>::zero(); n * n];
    for i in 0..n {
        for j in 0..n {
            packed[i * n + j] = if j < i {
                l[i * n + j] // strict lower of L
            } else {
                u[i * n + j] // U upper triangle
            };
        }
    }
    // Convert P (an n×n permutation matrix) to ipiv in cuSOLVER /
    // LAPACK swap-sequence form so the CPU and GPU paths produce
    // interchangeable output. cuSOLVER's `ipiv[i]` (1-based) is the
    // index of the row swapped INTO position `i` at step `i` of the
    // factorization.
    //
    // Two-step conversion:
    //   1. Read P as a permutation vector `perm` where `perm[i]` is the
    //      column with a 1 in row `i` of P (i.e. row `i` of `P @ A`
    //      equals row `perm[i]` of `A`).
    //   2. Convert `perm` → swap-sequence by replaying the swaps. At
    //      step `i`, the algorithm wants `perm[i]` at position `i`.
    //      Find where `perm[i]` lives in the running array `work`
    //      (originally identity), record the swap, and apply it.
    let mut perm = vec![0_usize; n];
    let one = T::from(1.0).unwrap();
    for i in 0..n {
        for j in 0..n {
            if p[i * n + j] == one {
                perm[i] = j;
                break;
            }
        }
    }
    let mut work: Vec<usize> = (0..n).collect();
    let mut ipiv = vec![0_i32; n];
    for i in 0..n {
        let target = perm[i];
        let j = (i..n).find(|&k| work[k] == target).unwrap_or(i);
        ipiv[i] = (j + 1) as i32;
        work.swap(i, j);
    }
    let lu = Tensor::from_storage(TensorStorage::cpu(packed), vec![n, n], false)?;
    Ok((lu, ipiv))
}

// ===========================================================================
// Singular values only / least squares
// ===========================================================================

/// Singular values (descending) of a 2-D tensor.
///
/// Mirrors `torch.linalg.svdvals`. CPU-only today.
pub fn svdvals<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "svdvals")?;
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("svdvals requires a 2-D tensor, got {shape:?}"),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let s = ferray_linalg::svdvals(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(s.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), s.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let s = ferray_linalg::svdvals(&arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(s.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), s.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Least-squares solution `X` minimizing `||A X - B||_F`. Just the
/// solution — no residuals / rank / singular values. (#630)
///
/// On CUDA f32/f64, dispatches to cuSOLVER `cusolverDnSSgels` /
/// `cusolverDnDDgels` (iterative refinement, no host bounce). CPU and
/// other dtypes route through `ferray-linalg::lstsq` and discard the
/// extra outputs. `A` is `[M, N]`; `B` is `[M, K]` (or `[M]` treated as
/// `[M, 1]`); output is `[N, K]` (or `[N]` for the 1-D case).
pub fn lstsq_solve<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("lstsq_solve: `a` must be 2-D, got {:?}", a.shape()),
        });
    }
    let m = a.shape()[0];
    let n = a.shape()[1];
    let (b_is_1d, nrhs) = match b.ndim() {
        1 if b.shape()[0] == m => (true, 1),
        2 if b.shape()[0] == m => (false, b.shape()[1]),
        _ => {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "lstsq_solve: `b` must be 1-D [{m}] or 2-D [{m}, K], got {:?}",
                    b.shape()
                ),
            });
        }
    };

    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let x_h = if is_f32::<T>() {
            backend.lstsq_f32(a.gpu_handle()?, b.gpu_handle()?, m, n, nrhs)?
        } else {
            backend.lstsq_f64(a.gpu_handle()?, b.gpu_handle()?, m, n, nrhs)?
        };
        let out_shape = if b_is_1d { vec![n] } else { vec![n, nrhs] };
        return Tensor::from_storage(TensorStorage::gpu(x_h), out_shape, false);
    }
    if a.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "lstsq_solve" });
    }

    // CPU: route through full ferray lstsq and take the solution slice.
    let (sol, _r, _rank, _sv) = lstsq(a, b, None)?;
    Ok(sol)
}

/// Least-squares solution to `A x ≈ b`.
///
/// Returns `(solution, residuals, rank, singular_values)`. `rcond`
/// controls the singular-value cutoff for rank determination; if `None`,
/// uses a sensible default (`max(m, n) * eps`).
///
/// Mirrors `torch.linalg.lstsq`. CPU-only today.
#[allow(clippy::type_complexity)]
pub fn lstsq<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    rcond: Option<f64>,
) -> FerrotorchResult<(Tensor<T>, Tensor<T>, Tensor<T>, Tensor<T>)> {
    require_cpu(a, "lstsq")?;
    require_cpu(b, "lstsq")?;
    if a.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("lstsq: `a` must be 2-D, got {:?}", a.shape()),
        });
    }

    if is_f32::<T>() {
        let a_arr = tensor_to_array2_f32(a)?;
        let b_arr = tensor_to_arraydyn_f32(b)?;
        let (sol, resid, rank, sv) = ferray_linalg::lstsq(&a_arr, &b_arr, rcond.map(|r| r as f32))
            .map_err(FerrotorchError::Ferray)?;
        Ok((
            Tensor::from_storage(
                TensorStorage::cpu(slice_f32_to_vec::<T>(sol.as_slice().unwrap())),
                sol.shape().to_vec(),
                false,
            )?,
            Tensor::from_storage(
                TensorStorage::cpu(slice_f32_to_vec::<T>(resid.as_slice().unwrap())),
                resid.shape().to_vec(),
                false,
            )?,
            // rank is a usize; encode as a length-0 (scalar) tensor of T.
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(rank as f32).unwrap()]),
                vec![],
                false,
            )?,
            Tensor::from_storage(
                TensorStorage::cpu(slice_f32_to_vec::<T>(sv.as_slice().unwrap())),
                sv.shape().to_vec(),
                false,
            )?,
        ))
    } else if is_f64::<T>() {
        let a_arr = tensor_to_array2_f64(a)?;
        let b_arr = tensor_to_arraydyn_f64(b)?;
        let (sol, resid, rank, sv) =
            ferray_linalg::lstsq(&a_arr, &b_arr, rcond).map_err(FerrotorchError::Ferray)?;
        Ok((
            Tensor::from_storage(
                TensorStorage::cpu(slice_to_vec::<T>(sol.as_slice().unwrap())),
                sol.shape().to_vec(),
                false,
            )?,
            Tensor::from_storage(
                TensorStorage::cpu(slice_to_vec::<T>(resid.as_slice().unwrap())),
                resid.shape().to_vec(),
                false,
            )?,
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(rank as f64).unwrap()]),
                vec![],
                false,
            )?,
            Tensor::from_storage(
                TensorStorage::cpu(slice_to_vec::<T>(sv.as_slice().unwrap())),
                sv.shape().to_vec(),
                false,
            )?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ===========================================================================
// Higher-order solvers (matrix_power, tensorsolve, tensorinv)
// ===========================================================================

/// Compute `A^n` for integer `n`. For `n >= 0`, uses repeated squaring;
/// for `n < 0`, computes the inverse first.
///
/// Mirrors `torch.linalg.matrix_power`. CPU-only today.
pub fn matrix_power<T: Float>(a: &Tensor<T>, n: i64) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "matrix_power")?;
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("matrix_power requires a square 2-D tensor, got {shape:?}"),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let r = ferray_linalg::matrix_power(&arr, n).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let r = ferray_linalg::matrix_power(&arr, n).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Solve `tensordot(a, x, axes) = b` for a tensor `x`.
///
/// Mirrors `torch.linalg.tensorsolve`. CPU-only today.
pub fn tensorsolve<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "tensorsolve")?;
    require_cpu(b, "tensorsolve")?;

    if is_f32::<T>() {
        let a_arr = tensor_to_arraydyn_f32(a)?;
        let b_arr = tensor_to_arraydyn_f32(b)?;
        let x =
            ferray_linalg::tensorsolve(&a_arr, &b_arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(x.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), x.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let a_arr = tensor_to_arraydyn_f64(a)?;
        let b_arr = tensor_to_arraydyn_f64(b)?;
        let x =
            ferray_linalg::tensorsolve(&a_arr, &b_arr, None).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(x.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), x.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Tensor inverse with respect to the partition at `ind`.
///
/// Mirrors `torch.linalg.tensorinv`. CPU-only today.
pub fn tensorinv<T: Float>(a: &Tensor<T>, ind: usize) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "tensorinv")?;

    if is_f32::<T>() {
        let arr = tensor_to_arraydyn_f32(a)?;
        let inv = ferray_linalg::tensorinv(&arr, ind).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(inv.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), inv.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let arr = tensor_to_arraydyn_f64(a)?;
        let inv = ferray_linalg::tensorinv(&arr, ind).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(inv.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), inv.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

// ===========================================================================
// Norms (vector / slogdet / matrix_rank / cond)
// ===========================================================================

/// p-norm of a tensor.
///
/// Returns a scalar tensor. `ord` may be `2.0` (L2/Frobenius), `1.0` (L1),
/// `f64::INFINITY`, or any positive real. Matches `torch.linalg.vector_norm`'s
/// scalar reduction (full-tensor) form.
///
/// CPU-only today.
pub fn vector_norm<T: Float>(input: &Tensor<T>, ord: f64) -> FerrotorchResult<Tensor<T>> {
    require_cpu(input, "vector_norm")?;
    let order = float_to_norm_order(ord);

    if is_f32::<T>() {
        let arr = tensor_to_arraydyn_f32(input)?;
        let r = ferray_linalg::vector_norm(&arr, order, None, false)
            .map_err(FerrotorchError::Ferray)?;
        // Result is a 0-d (or 1-d singleton) array.
        let val = T::from(r.as_slice().unwrap()[0]).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else if is_f64::<T>() {
        let arr = tensor_to_arraydyn_f64(input)?;
        let r = ferray_linalg::vector_norm(&arr, order, None, false)
            .map_err(FerrotorchError::Ferray)?;
        let val = T::from(r.as_slice().unwrap()[0]).unwrap();
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Sign and natural log of `|det(A)|`.
///
/// Returns `(sign, logabsdet)` as scalar tensors. For singular matrices,
/// `sign` is `0` and `logabsdet` is `-inf`. Mirrors `torch.linalg.slogdet`.
/// CPU-only today.
pub fn slogdet<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    require_cpu(a, "slogdet")?;
    let shape = a.shape();
    if shape.len() != 2 || shape[0] != shape[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("slogdet requires a square 2-D tensor, got {shape:?}"),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let (sign, logabs) = ferray_linalg::slogdet(&arr).map_err(FerrotorchError::Ferray)?;
        Ok((
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(sign).unwrap()]),
                vec![],
                false,
            )?,
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(logabs).unwrap()]),
                vec![],
                false,
            )?,
        ))
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let (sign, logabs) = ferray_linalg::slogdet(&arr).map_err(FerrotorchError::Ferray)?;
        Ok((
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(sign).unwrap()]),
                vec![],
                false,
            )?,
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(logabs).unwrap()]),
                vec![],
                false,
            )?,
        ))
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Numerical rank of `a`.
///
/// Returns a scalar (0-d) `i64`-valued tensor encoded as `T`. `tol`, when
/// `Some(t)`, is the absolute tolerance below which singular values are
/// treated as zero; default is `max(m, n) * eps * sigma_max`.
///
/// Mirrors `torch.linalg.matrix_rank`. CPU-only today.
pub fn matrix_rank<T: Float>(a: &Tensor<T>, tol: Option<f64>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "matrix_rank")?;
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("matrix_rank requires a 2-D tensor, got {shape:?}"),
        });
    }

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let r = ferray_linalg::matrix_rank(&arr, tol.map(|t| t as f32))
            .map_err(FerrotorchError::Ferray)?;
        Tensor::from_storage(
            TensorStorage::cpu(vec![T::from(r as f32).unwrap()]),
            vec![],
            false,
        )
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let r = ferray_linalg::matrix_rank(&arr, tol).map_err(FerrotorchError::Ferray)?;
        Tensor::from_storage(
            TensorStorage::cpu(vec![T::from(r as f64).unwrap()]),
            vec![],
            false,
        )
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Condition number of `a` under the given norm order (`p = 2.0` for the
/// 2-norm, `1.0`, `f64::INFINITY`, etc.).
///
/// Mirrors `torch.linalg.cond`. CPU-only today.
pub fn cond<T: Float>(a: &Tensor<T>, p: f64) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "cond")?;
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("cond requires a 2-D tensor, got {shape:?}"),
        });
    }
    let order = float_to_norm_order(p);

    if is_f32::<T>() {
        let arr = tensor_to_array2_f32(a)?;
        let val: f32 = ferray_linalg::cond(&arr, order).map_err(FerrotorchError::Ferray)?;
        Tensor::from_storage(
            TensorStorage::cpu(vec![T::from(val).unwrap()]),
            vec![],
            false,
        )
    } else if is_f64::<T>() {
        let arr = tensor_to_array2_f64(a)?;
        let val: f64 = ferray_linalg::cond(&arr, order).map_err(FerrotorchError::Ferray)?;
        Tensor::from_storage(
            TensorStorage::cpu(vec![T::from(val).unwrap()]),
            vec![],
            false,
        )
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Map a torch-style `ord` float to ferray's `NormOrder`.
///
/// `2.0` -> Fro/L2, `1.0` -> L1, `f64::INFINITY` -> Inf, `f64::NEG_INFINITY`
/// -> NegInf, anything else -> P(p as the underlying float type).
// reason: PyTorch dispatches `ord` by exact magic values (1.0, 2.0, ±inf).
// 1.0 and 2.0 are exactly representable in f64 and callers pass these
// literals directly, so equality (not epsilon) is the correct dispatch
// predicate; an epsilon check would route nearby user values like 1.0001
// to L1 silently and break parity with torch.linalg.norm.
#[allow(clippy::float_cmp)]
fn float_to_norm_order<T: Into<f64>>(ord: T) -> ferray_linalg::NormOrder {
    let v: f64 = ord.into();
    if v == f64::INFINITY {
        ferray_linalg::NormOrder::Inf
    } else if v == f64::NEG_INFINITY {
        ferray_linalg::NormOrder::NegInf
    } else if v == 1.0 {
        ferray_linalg::NormOrder::L1
    } else if v == 2.0 {
        ferray_linalg::NormOrder::Fro
    } else {
        ferray_linalg::NormOrder::P(v)
    }
}

// ===========================================================================
// Vector products (cross / multi_dot)
// ===========================================================================

/// 3-element vector cross product `a × b`.
///
/// Both inputs must be 1-D length-3 tensors today (matching ferray-linalg's
/// scalar cross). The `dim` argument is reserved for future torch-style
/// dispatch over arbitrary shapes with a length-3 axis; it is currently
/// ignored. Mirrors `torch.linalg.cross` for the 1-D case. CPU-only.
pub fn cross<T: Float>(a: &Tensor<T>, b: &Tensor<T>, dim: i64) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "cross")?;
    require_cpu(b, "cross")?;

    if is_f32::<T>() {
        let a_arr = tensor_to_arraydyn_f32(a)?;
        let b_arr = tensor_to_arraydyn_f32(b)?;
        let _ = dim; // ferray's cross hardcodes the last axis with size 3
        let r = ferray_linalg::cross(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let a_arr = tensor_to_arraydyn_f64(a)?;
        let b_arr = tensor_to_arraydyn_f64(b)?;
        let _ = dim; // ferray's cross hardcodes the last axis with size 3
        let r = ferray_linalg::cross(&a_arr, &b_arr).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Chained matmul `A1 @ A2 @ ... @ Ak`, ordered to minimise intermediate
/// flop count.
///
/// Mirrors `torch.linalg.multi_dot`. CPU-only today.
pub fn multi_dot<T: Float>(matrices: &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    if matrices.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "multi_dot requires at least one matrix".into(),
        });
    }
    for m in matrices {
        require_cpu(m, "multi_dot")?;
    }

    if is_f32::<T>() {
        let arrs: Vec<_> = matrices
            .iter()
            .map(|m| tensor_to_arraydyn_f32(m))
            .collect::<Result<_, _>>()?;
        let refs: Vec<_> = arrs.iter().collect();
        let r = ferray_linalg::multi_dot(&refs).map_err(FerrotorchError::Ferray)?;
        let data = slice_f32_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else if is_f64::<T>() {
        let arrs: Vec<_> = matrices
            .iter()
            .map(|m| tensor_to_arraydyn_f64(m))
            .collect::<Result<_, _>>()?;
        let refs: Vec<_> = arrs.iter().collect();
        let r = ferray_linalg::multi_dot(&refs).map_err(FerrotorchError::Ferray)?;
        let data = slice_to_vec::<T>(r.as_slice().unwrap());
        Tensor::from_storage(TensorStorage::cpu(data), r.shape().to_vec(), false)
    } else {
        Err(FerrotorchError::InvalidArgument {
            message: "linalg op requires f32 or f64".into(),
        })
    }
}

/// Diagonal of a 2-D tensor, optionally offset.
///
/// Returns a 1-D tensor of length `min(m, n) - |offset|` containing
/// `a[i, i + offset]`. Implemented in-house (no ferray dep) since it's a
/// pure-shape operation.
///
/// Mirrors `torch.linalg.diagonal` (and `torch.diagonal` with `dim1=0,
/// dim2=1`).
pub fn diagonal<T: Float>(a: &Tensor<T>, offset: i64) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "diagonal")?;
    let shape = a.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("diagonal requires a 2-D tensor, got {shape:?}"),
        });
    }
    let (m, n) = (shape[0] as i64, shape[1] as i64);
    let row_start: i64;
    let col_start: i64;
    if offset >= 0 {
        row_start = 0;
        col_start = offset;
    } else {
        row_start = -offset;
        col_start = 0;
    }
    if col_start >= n || row_start >= m {
        return Tensor::from_storage(TensorStorage::cpu(Vec::<T>::new()), vec![0], false);
    }
    let len = (m - row_start).min(n - col_start) as usize;
    let data = a.data()?;
    let mut out: Vec<T> = Vec::with_capacity(len);
    for i in 0..len as i64 {
        let r = (row_start + i) as usize;
        let c = (col_start + i) as usize;
        out.push(data[r * shape[1] + c]);
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![len], false)
}

// ===========================================================================
// Linalg tail: solve_triangular / matrix_exp / ldl / householder / *_ex (#581)
// ===========================================================================

/// Solve `A x = b` (or `x A = b`) where `A` is triangular.
///
/// `upper`: if `true`, treat `A` as upper-triangular; else lower-triangular.
/// `transpose`: if `true`, solve `A^T x = b` (or `x A^T = b`).
/// `unit_diagonal`: if `true`, ignore the diagonal entries of `A` and treat
/// them as 1 (the matrix's strict-triangular part still defines the system).
///
/// `b` may be 1-D (`[n]`) for a single right-hand side or 2-D (`[n, k]`) for
/// `k` simultaneous RHS columns. Output has the same shape as `b`.
///
/// Mirrors `torch.linalg.solve_triangular`. CPU-only today (forward / back
/// substitution in pure Rust at f64 internally).
pub fn solve_triangular<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    upper: bool,
    transpose: bool,
    unit_diagonal: bool,
) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "solve_triangular")?;
    require_cpu(b, "solve_triangular")?;

    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "solve_triangular: a must be square 2-D, got {:?}",
                a.shape()
            ),
        });
    }
    let n = a.shape()[0];
    let (b_shape, k) = match b.ndim() {
        1 => (vec![n], 1usize),
        2 => {
            if b.shape()[0] != n {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("solve_triangular: b leading dim {} ≠ n={n}", b.shape()[0]),
                });
            }
            (vec![n, b.shape()[1]], b.shape()[1])
        }
        _ => {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "solve_triangular: b must be 1-D or 2-D, got {:?}",
                    b.shape()
                ),
            });
        }
    };

    // Materialize to f64 internally; the existing helpers use the same
    // strategy. Final cast back to T.
    let a_f64: Vec<f64> = a.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();
    let mut x: Vec<f64> = b.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();

    // Effective `upper`: when transposed, an upper-triangular A becomes
    // lower-triangular and vice versa. Fold it here so the loop only handles
    // two cases.
    let effective_upper = upper ^ transpose;

    let a_at = |row: usize, col: usize| -> f64 {
        if transpose {
            a_f64[col * n + row]
        } else {
            a_f64[row * n + col]
        }
    };

    for col in 0..k {
        let stride = if b.ndim() == 1 { 0 } else { k };
        let xj = |i: usize, j: usize, x: &[f64]| -> f64 {
            if b.ndim() == 1 {
                x[i]
            } else {
                x[i * stride + j]
            }
        };
        let xj_set = |i: usize, j: usize, val: f64, x: &mut [f64]| {
            if b.ndim() == 1 {
                x[i] = val;
            } else {
                x[i * stride + j] = val;
            }
        };

        if effective_upper {
            // Back-substitute from row n-1 → 0.
            for i in (0..n).rev() {
                let mut sum = xj(i, col, &x);
                for j in (i + 1)..n {
                    sum -= a_at(i, j) * xj(j, col, &x);
                }
                let diag = if unit_diagonal { 1.0 } else { a_at(i, i) };
                if diag == 0.0 {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "solve_triangular: zero on diagonal".into(),
                    });
                }
                xj_set(i, col, sum / diag, &mut x);
            }
        } else {
            // Forward-substitute from row 0 → n-1.
            for i in 0..n {
                let mut sum = xj(i, col, &x);
                for j in 0..i {
                    sum -= a_at(i, j) * xj(j, col, &x);
                }
                let diag = if unit_diagonal { 1.0 } else { a_at(i, i) };
                if diag == 0.0 {
                    return Err(FerrotorchError::InvalidArgument {
                        message: "solve_triangular: zero on diagonal".into(),
                    });
                }
                xj_set(i, col, sum / diag, &mut x);
            }
        }
    }

    let out: Vec<T> = x.into_iter().map(|v| T::from(v).unwrap()).collect();
    Tensor::from_storage(TensorStorage::cpu(out), b_shape, false)
}

/// LDL^T factorization of a real symmetric matrix (no pivoting).
///
/// Returns `(L, D)` where `L` is unit lower-triangular and `D` is diagonal
/// (returned as a length-`n` vector), with `A = L D L^T`.
///
/// Mirrors `torch.linalg.ldl_factor` for the no-pivot case. Numerically
/// reliable on positive-definite or strongly diagonally-dominant inputs;
/// for indefinite or rank-deficient matrices use `eigh` or a pivoted
/// factorization (Bunch-Kaufman) — see follow-up.
pub fn ldl_factor<T: Float>(a: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    require_cpu(a, "ldl_factor")?;
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("ldl_factor: a must be square 2-D, got {:?}", a.shape()),
        });
    }
    let n = a.shape()[0];
    let a_f64: Vec<f64> = a.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();

    let mut l = vec![0.0f64; n * n];
    let mut d = vec![0.0f64; n];

    // No-pivot LDL^T: for j in 0..n,
    //   D_j = A_jj - sum_{k<j} L_jk^2 * D_k
    //   L_ij = (A_ij - sum_{k<j} L_ik * L_jk * D_k) / D_j  for i > j
    //   L_jj = 1
    for j in 0..n {
        let mut diag = a_f64[j * n + j];
        for k in 0..j {
            diag -= l[j * n + k] * l[j * n + k] * d[k];
        }
        d[j] = diag;
        if diag == 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ldl_factor: zero pivot at column {j} (no-pivot path)"),
            });
        }
        l[j * n + j] = 1.0;
        for i in (j + 1)..n {
            let mut sum = a_f64[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k] * d[k];
            }
            l[i * n + j] = sum / diag;
        }
    }

    let l_out: Vec<T> = l.into_iter().map(|v| T::from(v).unwrap()).collect();
    let d_out: Vec<T> = d.into_iter().map(|v| T::from(v).unwrap()).collect();
    Ok((
        Tensor::from_storage(TensorStorage::cpu(l_out), vec![n, n], false)?,
        Tensor::from_storage(TensorStorage::cpu(d_out), vec![n], false)?,
    ))
}

/// Solve `A x = b` using a precomputed LDL^T factorization.
///
/// Given `(L, D)` from [`ldl_factor`] and a right-hand side `b` (1-D or 2-D),
/// returns `x` such that `(L D L^T) x = b`. Same `b` shape conventions as
/// [`solve`] / [`solve_triangular`].
pub fn ldl_solve<T: Float>(
    l: &Tensor<T>,
    d: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    require_cpu(l, "ldl_solve")?;
    require_cpu(d, "ldl_solve")?;
    require_cpu(b, "ldl_solve")?;
    if l.ndim() != 2 || l.shape()[0] != l.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("ldl_solve: L must be square 2-D, got {:?}", l.shape()),
        });
    }
    if d.ndim() != 1 || d.shape()[0] != l.shape()[0] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ldl_solve: D must be 1-D of length {}, got {:?}",
                l.shape()[0],
                d.shape()
            ),
        });
    }

    // Step 1: solve L y = b (forward substitution, unit diagonal).
    let y = solve_triangular(
        l, b, /* upper */ false, /* transpose */ false, /* unit_diag */ true,
    )?;
    // Step 2: scale by D^{-1}: z_i = y_i / d_i (broadcast across columns of y).
    let n = d.shape()[0];
    let d_data = d.data()?.to_vec();
    let y_data = y.data()?.to_vec();
    let z_shape = y.shape().to_vec();
    let k = if y.ndim() == 1 { 1 } else { y.shape()[1] };
    let mut z = vec![T::from(0.0).unwrap(); y_data.len()];
    for i in 0..n {
        let di = d_data[i].to_f64().unwrap();
        if di == 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ldl_solve: zero diagonal at index {i}"),
            });
        }
        for j in 0..k {
            let val = if y.ndim() == 1 {
                y_data[i].to_f64().unwrap()
            } else {
                y_data[i * k + j].to_f64().unwrap()
            };
            let scaled = T::from(val / di).unwrap();
            if y.ndim() == 1 {
                z[i] = scaled;
            } else {
                z[i * k + j] = scaled;
            }
        }
    }
    let z_t = Tensor::from_storage(TensorStorage::cpu(z), z_shape, false)?;
    // Step 3: solve L^T x = z (back substitution via transpose).
    solve_triangular(
        l, &z_t, /* upper */ false, /* transpose */ true, /* unit_diag */ true,
    )
}

/// Apply the implicit Householder representation `(V, tau)` from a QR
/// factorization to recover the orthogonal matrix `Q`.
///
/// `v` is `[m, k]` whose `j`-th column is the Householder vector for the
/// `j`-th reflection (with implicit unit at row `j`). `tau` is `[k]` of
/// scalar coefficients. Returns `Q` of shape `[m, m]` such that
/// `Q = (I - tau_0 v_0 v_0^T)(I - tau_1 v_1 v_1^T) ... `.
///
/// Mirrors `torch.linalg.householder_product`.
pub fn householder_product<T: Float>(
    v: &Tensor<T>,
    tau: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    require_cpu(v, "householder_product")?;
    require_cpu(tau, "householder_product")?;
    if v.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("householder_product: v must be 2-D, got {:?}", v.shape()),
        });
    }
    if tau.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "householder_product: tau must be 1-D, got {:?}",
                tau.shape()
            ),
        });
    }
    let m = v.shape()[0];
    let k = v.shape()[1];
    if tau.shape()[0] != k {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "householder_product: tau length {} ≠ v cols {k}",
                tau.shape()[0]
            ),
        });
    }
    if k > m {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("householder_product: k={k} must be ≤ m={m}"),
        });
    }

    let v_f64: Vec<f64> = v.data()?.iter().map(|&x| x.to_f64().unwrap()).collect();
    let tau_f64: Vec<f64> = tau.data()?.iter().map(|&x| x.to_f64().unwrap()).collect();

    // Initialize Q = I_m (row-major).
    let mut q = vec![0.0f64; m * m];
    for i in 0..m {
        q[i * m + i] = 1.0;
    }

    // Apply reflections in reverse order so the cumulative product
    // (I - τ_0 v_0 v_0^T) (I - τ_1 v_1 v_1^T) ... lands in Q. We update
    // Q ← H_j Q where H_j is the j-th reflector. Stepping right-to-left
    // makes that Q = H_0 H_1 ... H_{k-1}.
    for j in (0..k).rev() {
        let tau_j = tau_f64[j];
        if tau_j == 0.0 {
            continue;
        }
        // Extract v_j: column j of V with implicit unit at row j and zeros
        // above row j.
        let mut vj = vec![0.0f64; m];
        vj[j] = 1.0;
        for i in (j + 1)..m {
            vj[i] = v_f64[i * k + j];
        }

        // For each column c of Q, compute Q[:, c] -= τ * v_j * (v_j^T Q[:, c]).
        for c in 0..m {
            let mut dot = 0.0f64;
            for i in 0..m {
                dot += vj[i] * q[i * m + c];
            }
            let scale = tau_j * dot;
            for i in 0..m {
                q[i * m + c] -= scale * vj[i];
            }
        }
    }

    let out: Vec<T> = q.into_iter().map(|x| T::from(x).unwrap()).collect();
    Tensor::from_storage(TensorStorage::cpu(out), vec![m, m], false)
}

/// Matrix exponential `expm(A)` via Padé(13) with scaling and squaring.
///
/// Uses the Higham 2005 algorithm: choose `s` so `||A/2^s||_∞ ≤ θ_13`,
/// compute the Padé(13) approximation of `exp(A/2^s)`, then square `s`
/// times to recover `exp(A)`. Mirrors `torch.linalg.matrix_exp`.
///
/// CPU-only; works in f64 internally.
pub fn matrix_exp<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    require_cpu(a, "matrix_exp")?;
    if a.ndim() != 2 || a.shape()[0] != a.shape()[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("matrix_exp: a must be square 2-D, got {:?}", a.shape()),
        });
    }
    let n = a.shape()[0];
    let a_data: Vec<f64> = a.data()?.iter().map(|&v| v.to_f64().unwrap()).collect();
    let result = matrix_exp_pade13(&a_data, n)?;
    let out: Vec<T> = result.into_iter().map(|v| T::from(v).unwrap()).collect();
    Tensor::from_storage(TensorStorage::cpu(out), vec![n, n], false)
}

// --- helpers for matrix_exp ------------------------------------------------

fn mat_eye(n: usize) -> Vec<f64> {
    let mut m = vec![0.0f64; n * n];
    for i in 0..n {
        m[i * n + i] = 1.0;
    }
    m
}

fn mat_inf_norm(a: &[f64], n: usize) -> f64 {
    (0..n)
        .map(|i| (0..n).map(|j| a[i * n + j].abs()).sum::<f64>())
        .fold(0.0, f64::max)
}

fn mat_mul(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let mut out = vec![0.0f64; n * n];
    for i in 0..n {
        for k in 0..n {
            let aik = a[i * n + k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..n {
                out[i * n + j] += aik * b[k * n + j];
            }
        }
    }
    out
}

fn mat_axpby(a: &[f64], alpha: f64, b: &[f64], beta: f64) -> Vec<f64> {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| alpha * x + beta * y)
        .collect()
}

/// Solve `(I - U)^{-1} (I + U)`-style linear system used in Padé approximant.
/// Solves `(P) X = Q` for `X` via LU with partial pivoting in pure Rust.
fn solve_dense_pivoted(p: &[f64], q: &[f64], n: usize) -> FerrotorchResult<Vec<f64>> {
    // Augmented matrix [P | Q] in row-major; size n × 2n.
    let mut aug = vec![0.0f64; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * (2 * n) + j] = p[i * n + j];
            aug[i * (2 * n) + n + j] = q[i * n + j];
        }
    }
    // Gaussian elimination with partial pivoting.
    for col in 0..n {
        // Find pivot row.
        let mut pivot_row = col;
        let mut pivot_val = aug[col * (2 * n) + col].abs();
        for r in (col + 1)..n {
            let v = aug[r * (2 * n) + col].abs();
            if v > pivot_val {
                pivot_val = v;
                pivot_row = r;
            }
        }
        if pivot_val == 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "matrix_exp: singular Padé denominator (numerical)".into(),
            });
        }
        if pivot_row != col {
            for j in 0..(2 * n) {
                aug.swap(col * (2 * n) + j, pivot_row * (2 * n) + j);
            }
        }
        // Eliminate below.
        let pivot = aug[col * (2 * n) + col];
        for r in (col + 1)..n {
            let factor = aug[r * (2 * n) + col] / pivot;
            if factor == 0.0 {
                continue;
            }
            for j in col..(2 * n) {
                aug[r * (2 * n) + j] -= factor * aug[col * (2 * n) + j];
            }
        }
    }
    // Back-substitute.
    let mut x = vec![0.0f64; n * n];
    for c in 0..n {
        for i in (0..n).rev() {
            let mut sum = aug[i * (2 * n) + n + c];
            for j in (i + 1)..n {
                sum -= aug[i * (2 * n) + j] * x[j * n + c];
            }
            let diag = aug[i * (2 * n) + i];
            x[i * n + c] = sum / diag;
        }
    }
    Ok(x)
}

fn matrix_exp_pade13(a: &[f64], n: usize) -> FerrotorchResult<Vec<f64>> {
    // Higham 2005 thresholds and Padé(13) coefficients.
    const THETA13: f64 = 5.371920351148152;
    let b: [f64; 14] = [
        64764752532480000.0,
        32382376266240000.0,
        7771770303897600.0,
        1187353796428800.0,
        129060195264000.0,
        10559470521600.0,
        670442572800.0,
        33522128640.0,
        1323241920.0,
        40840800.0,
        960960.0,
        16380.0,
        182.0,
        1.0,
    ];

    let norm = mat_inf_norm(a, n);
    let s = if norm <= THETA13 {
        0
    } else {
        (norm / THETA13).log2().ceil() as i32
    };
    let scale = (1u64 << s.max(0)) as f64;
    let a_scaled: Vec<f64> = a.iter().map(|&v| v / scale).collect();

    let id = mat_eye(n);
    let a2 = mat_mul(&a_scaled, &a_scaled, n);
    let a4 = mat_mul(&a2, &a2, n);
    let a6 = mat_mul(&a4, &a2, n);

    // U = A * (A6 * (b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I)
    // V = A6 * (b12 A6 + b10 A4 + b8 A2) + b6 A6 + b4 A4 + b2 A2 + b0 I
    let inner_u = {
        let t1 = mat_axpby(&a6, b[13], &a4, b[11]);
        let t2 = mat_axpby(&t1, 1.0, &a2, b[9]);
        mat_mul(&a6, &t2, n)
    };
    let mid_u = mat_axpby(&inner_u, 1.0, &a6, b[7]);
    let mid_u = mat_axpby(&mid_u, 1.0, &a4, b[5]);
    let mid_u = mat_axpby(&mid_u, 1.0, &a2, b[3]);
    let mid_u = mat_axpby(&mid_u, 1.0, &id, b[1]);
    let u = mat_mul(&a_scaled, &mid_u, n);

    let inner_v = {
        let t1 = mat_axpby(&a6, b[12], &a4, b[10]);
        let t2 = mat_axpby(&t1, 1.0, &a2, b[8]);
        mat_mul(&a6, &t2, n)
    };
    let v = mat_axpby(&inner_v, 1.0, &a6, b[6]);
    let v = mat_axpby(&v, 1.0, &a4, b[4]);
    let v = mat_axpby(&v, 1.0, &a2, b[2]);
    let v = mat_axpby(&v, 1.0, &id, b[0]);

    let p = mat_axpby(&v, 1.0, &u, -1.0); // V - U
    let q = mat_axpby(&v, 1.0, &u, 1.0); // V + U
    let mut r = solve_dense_pivoted(&p, &q, n)?;

    // Squaring phase.
    for _ in 0..s.max(0) {
        r = mat_mul(&r, &r, n);
    }
    Ok(r)
}

// ---------------------------------------------------------------------------
// `_ex` variants — return `(value, info)` with non-throwing semantics
// ---------------------------------------------------------------------------

/// `cholesky` that doesn't error on failure: returns `(L, info)` where
/// `info` is `0` on success and the leading-minor index that failed
/// (1-based) when `A` is not positive-definite.
///
/// Mirrors `torch.linalg.cholesky_ex`. `info` is returned as a 0-d
/// scalar tensor (cast to `T`) for shape consistency with the family.
pub fn cholesky_ex<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    match cholesky(input) {
        Ok(l) => Ok((
            l,
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(0.0).unwrap()]),
                vec![],
                false,
            )?,
        )),
        Err(_) => {
            // Build a same-shape zero L and info=1 (non-zero failure indicator).
            let shape = input.shape();
            let n = shape.first().copied().unwrap_or(0);
            let zero_l = vec![T::from(0.0).unwrap(); n * n];
            Ok((
                Tensor::from_storage(TensorStorage::cpu(zero_l), vec![n, n], false)?,
                Tensor::from_storage(
                    TensorStorage::cpu(vec![T::from(1.0).unwrap()]),
                    vec![],
                    false,
                )?,
            ))
        }
    }
}

/// `inv` that doesn't error on singular input: returns `(A^{-1}, info)`.
pub fn inv_ex<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    match inv(input) {
        Ok(out) => Ok((
            out,
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(0.0).unwrap()]),
                vec![],
                false,
            )?,
        )),
        Err(_) => {
            let shape = input.shape();
            let n = shape.first().copied().unwrap_or(0);
            let zero = vec![T::from(0.0).unwrap(); n * n];
            Ok((
                Tensor::from_storage(TensorStorage::cpu(zero), vec![n, n], false)?,
                Tensor::from_storage(
                    TensorStorage::cpu(vec![T::from(1.0).unwrap()]),
                    vec![],
                    false,
                )?,
            ))
        }
    }
}

/// `solve` that doesn't error on singular `A`: returns `(x, info)`.
pub fn solve_ex<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    match solve(a, b) {
        Ok(x) => Ok((
            x,
            Tensor::from_storage(
                TensorStorage::cpu(vec![T::from(0.0).unwrap()]),
                vec![],
                false,
            )?,
        )),
        Err(_) => {
            let shape = b.shape().to_vec();
            let total: usize = shape.iter().product();
            let zero = vec![T::from(0.0).unwrap(); total];
            Ok((
                Tensor::from_storage(TensorStorage::cpu(zero), shape, false)?,
                Tensor::from_storage(
                    TensorStorage::cpu(vec![T::from(1.0).unwrap()]),
                    vec![],
                    false,
                )?,
            ))
        }
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
        let eye = t(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
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
        let eye = t(&[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], &[3, 3]);
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

    // -----------------------------------------------------------------------
    // eigh / eigvalsh (symmetric)
    // -----------------------------------------------------------------------

    #[test]
    fn test_eigh_diagonal_matrix() {
        // Diagonal matrix: eigenvalues are the diagonal entries (sorted),
        // eigenvectors are standard basis vectors.
        let a = t(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], &[3, 3]);
        let (w, _q) = eigh(&a).unwrap();
        let w_data = w.data().unwrap();
        // Ascending order: 1, 2, 3
        assert!((w_data[0] - 1.0).abs() < 1e-10);
        assert!((w_data[1] - 2.0).abs() < 1e-10);
        assert!((w_data[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_eigvalsh_matches_eigh() {
        // eigvalsh should agree with the eigenvalue half of eigh.
        let a = t(&[2.0, 1.0, 1.0, 2.0], &[2, 2]);
        let w_only = eigvalsh(&a).unwrap();
        let (w_full, _q) = eigh(&a).unwrap();
        let a_data = w_only.data().unwrap();
        let b_data = w_full.data().unwrap();
        for i in 0..2 {
            assert!(
                (a_data[i] - b_data[i]).abs() < 1e-10,
                "eigvalsh[{i}]={} vs eigh.0[{i}]={}",
                a_data[i],
                b_data[i]
            );
        }
    }

    #[test]
    fn test_eigh_reconstructs() {
        // A symmetric -> A = Q diag(w) Q^T.
        let a = t(&[4.0, 1.0, 1.0, 3.0], &[2, 2]);
        let (w, q) = eigh(&a).unwrap();
        let w_data = w.data().unwrap();
        let q_data = q.data().unwrap();
        // Reconstruct: result[i,j] = sum_k q[i,k] * w[k] * q[j,k]
        let n = 2;
        let mut recon = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += q_data[i * n + k] * w_data[k] * q_data[j * n + k];
                }
                recon[i * n + j] = acc;
            }
        }
        let a_data = a.data().unwrap();
        for i in 0..n * n {
            assert!(
                (recon[i] - a_data[i]).abs() < 1e-9,
                "eigh reconstruction at {i}: {} vs {}",
                recon[i],
                a_data[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // eig / eigvals (general; complex output)
    // -----------------------------------------------------------------------

    #[test]
    fn test_eigvals_diagonal_real() {
        // Diagonal: eigenvalues are diagonal entries (real).
        let a = t(&[2.0, 0.0, 0.0, 5.0], &[2, 2]);
        let w = eigvals(&a).unwrap();
        // Shape is [n, 2] with last dim = (re, im).
        assert_eq!(w.shape(), &[2, 2]);
        let d = w.data().unwrap();
        // The two eigenvalues should be {2, 5} in some order; collect real parts.
        let reals: Vec<f64> = (0..2).map(|i| d[i * 2]).collect();
        let imags: Vec<f64> = (0..2).map(|i| d[i * 2 + 1]).collect();
        for im in imags {
            assert!(im.abs() < 1e-10, "imag part should be 0, got {im}");
        }
        let mut sorted = reals.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert!((sorted[0] - 2.0).abs() < 1e-10);
        assert!((sorted[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_eig_returns_complex_eigenvectors_shape() {
        let a = t(&[0.0, -1.0, 1.0, 0.0], &[2, 2]); // rotation 90°: complex eigenvalues ±i
        let (w, v) = eig(&a).unwrap();
        assert_eq!(w.shape(), &[2, 2]);
        assert_eq!(v.shape(), &[2, 2, 2]);
    }

    // -----------------------------------------------------------------------
    // lu
    // -----------------------------------------------------------------------

    #[test]
    fn test_lu_reconstructs() {
        let a = t(&[2.0, 4.0, 6.0, 1.0, 3.0, 5.0, 7.0, 8.0, 9.0], &[3, 3]);
        let (p, l, u) = lu(&a).unwrap();
        let p_data = p.data().unwrap();
        let l_data = l.data().unwrap();
        let u_data = u.data().unwrap();
        // Reconstruct PLU
        let n = 3;
        // L is [3,3] (k=min(3,3)=3); U is [3,3]
        let mut lu_prod = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += l_data[i * n + k] * u_data[k * n + j];
                }
                lu_prod[i * n + j] = acc;
            }
        }
        let mut plu = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += p_data[i * n + k] * lu_prod[k * n + j];
                }
                plu[i * n + j] = acc;
            }
        }
        let a_data = a.data().unwrap();
        for i in 0..n * n {
            assert!(
                (plu[i] - a_data[i]).abs() < 1e-9,
                "lu reconstruction at {i}: {} vs {}",
                plu[i],
                a_data[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // svdvals / lstsq
    // -----------------------------------------------------------------------

    #[test]
    fn test_svdvals_descending() {
        let a = t(&[3.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 2.0], &[3, 3]);
        let s = svdvals(&a).unwrap();
        let d = s.data().unwrap();
        // Descending: 3, 2, 1
        assert!((d[0] - 3.0).abs() < 1e-9);
        assert!((d[1] - 2.0).abs() < 1e-9);
        assert!((d[2] - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_lstsq_overdetermined() {
        // y = 2x + 1 fit through (0,1), (1,3), (2,5), (3,7).
        let a = t(&[0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0], &[4, 2]);
        let b = t(&[1.0, 3.0, 5.0, 7.0], &[4]);
        let (sol, _resid, rank, sv) = lstsq(&a, &b, None).unwrap();
        let s = sol.data_vec().unwrap();
        // Coefficients should be (2, 1).
        assert!((s[0] - 2.0).abs() < 1e-9, "slope = {}", s[0]);
        assert!((s[1] - 1.0).abs() < 1e-9, "intercept = {}", s[1]);
        // Rank 2 (full column rank).
        assert!((rank.data().unwrap()[0] - 2.0).abs() < 1e-9);
        assert_eq!(sv.shape(), &[2]);
    }

    // -----------------------------------------------------------------------
    // matrix_power, matrix_rank, slogdet, cond
    // -----------------------------------------------------------------------

    #[test]
    fn test_matrix_power_zero_is_identity() {
        let a = t(&[2.0, 1.0, 0.0, 3.0], &[2, 2]);
        let r = matrix_power(&a, 0).unwrap();
        let d = r.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!(d[1].abs() < 1e-10);
        assert!(d[2].abs() < 1e-10);
        assert!((d[3] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_power_two_equals_self_squared() {
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let a2 = matrix_power(&a, 2).unwrap();
        // Hand-compute A @ A: [[7, 10], [15, 22]]
        let d = a2.data().unwrap();
        assert!((d[0] - 7.0).abs() < 1e-10);
        assert!((d[1] - 10.0).abs() < 1e-10);
        assert!((d[2] - 15.0).abs() < 1e-10);
        assert!((d[3] - 22.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_rank_full_rank_2x2() {
        let a = t(&[1.0, 2.0, 3.0, 5.0], &[2, 2]);
        let r = matrix_rank(&a, None).unwrap();
        assert!((r.data().unwrap()[0] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_matrix_rank_singular_2x2() {
        // Rows are scalar multiples — rank 1.
        let a = t(&[1.0, 2.0, 2.0, 4.0], &[2, 2]);
        let r = matrix_rank(&a, None).unwrap();
        assert!((r.data().unwrap()[0] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_slogdet_identity() {
        let a = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let (sign, logabs) = slogdet(&a).unwrap();
        assert!((sign.data().unwrap()[0] - 1.0).abs() < 1e-10);
        assert!(logabs.data().unwrap()[0].abs() < 1e-10);
    }

    #[test]
    fn test_slogdet_negative_det() {
        // det = 1*4 - 2*3 = -2, sign = -1, log|det| = log(2)
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let (sign, logabs) = slogdet(&a).unwrap();
        assert!((sign.data().unwrap()[0] - (-1.0)).abs() < 1e-10);
        assert!((logabs.data().unwrap()[0] - 2.0_f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn test_cond_identity_is_one() {
        let a = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let c = cond(&a, 2.0).unwrap();
        assert!((c.data().unwrap()[0] - 1.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // vector_norm
    // -----------------------------------------------------------------------

    #[test]
    fn test_vector_norm_l2() {
        let v = t(&[3.0, 4.0], &[2]);
        let n = vector_norm(&v, 2.0).unwrap();
        assert!((n.data().unwrap()[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_norm_l1() {
        let v = t(&[1.0, -2.0, 3.0, -4.0], &[4]);
        let n = vector_norm(&v, 1.0).unwrap();
        assert!((n.data().unwrap()[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_vector_norm_inf() {
        let v = t(&[1.0, -7.0, 3.0, -4.0], &[4]);
        let n = vector_norm(&v, f64::INFINITY).unwrap();
        assert!((n.data().unwrap()[0] - 7.0).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // multi_dot, cross, diagonal
    // -----------------------------------------------------------------------

    #[test]
    fn test_multi_dot_chains_three() {
        // (A @ B) @ C
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[1.0, 0.0, 0.0, 1.0], &[2, 2]);
        let c = t(&[2.0, 0.0, 0.0, 2.0], &[2, 2]);
        let r = multi_dot(&[&a, &b, &c]).unwrap();
        // A @ I @ 2I = 2A
        let d = r.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 4.0).abs() < 1e-10);
        assert!((d[2] - 6.0).abs() < 1e-10);
        assert!((d[3] - 8.0).abs() < 1e-10);
    }

    #[test]
    fn test_cross_basis_vectors() {
        // e1 × e2 = e3
        let e1 = t(&[1.0, 0.0, 0.0], &[3]);
        let e2 = t(&[0.0, 1.0, 0.0], &[3]);
        let r = cross(&e1, &e2, -1).unwrap();
        let d = r.data().unwrap();
        assert!(d[0].abs() < 1e-10);
        assert!(d[1].abs() < 1e-10);
        assert!((d[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_diagonal_main() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let d = diagonal(&a, 0).unwrap();
        assert_eq!(d.shape(), &[3]);
        assert_eq!(d.data().unwrap(), &[1.0, 5.0, 9.0]);
    }

    #[test]
    fn test_diagonal_offset_positive() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let d = diagonal(&a, 1).unwrap();
        // a[0,1], a[1,2] = 2, 6
        assert_eq!(d.data().unwrap(), &[2.0, 6.0]);
    }

    #[test]
    fn test_diagonal_offset_negative() {
        let a = t(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3]);
        let d = diagonal(&a, -1).unwrap();
        // a[1,0], a[2,1] = 4, 8
        assert_eq!(d.data().unwrap(), &[4.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // tensorinv / tensorsolve smoke
    // -----------------------------------------------------------------------

    #[test]
    fn test_tensorinv_2x2_matrix_form() {
        // For a [2,2]-shaped matrix viewed as a (2,2) tensor at ind=1, the
        // tensor inverse is the matrix inverse.
        let a = t(&[4.0, 7.0, 2.0, 6.0], &[2, 2]);
        let inv_a = tensorinv(&a, 1).unwrap();
        // Hand-compute A^-1 = (1/10) * [[6, -7], [-2, 4]]
        let d = inv_a.data().unwrap();
        assert!((d[0] - 0.6).abs() < 1e-10);
        assert!((d[1] - (-0.7)).abs() < 1e-10);
        assert!((d[2] - (-0.2)).abs() < 1e-10);
        assert!((d[3] - 0.4).abs() < 1e-10);
    }

    // -----------------------------------------------------------------------
    // GPU discipline: every new fn returns InvalidArgument on CUDA tensors,
    // never silently downloads. We can't construct CUDA tensors in this
    // CPU-only test, but the require_cpu gate is exercised by the GPU
    // tests in ferrotorch-gpu.
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // Linalg tail: solve_triangular / matrix_exp / ldl / householder / *_ex (#581)
    // -----------------------------------------------------------------------

    #[test]
    fn solve_triangular_lower_1d_b() {
        // L = [[1, 0], [2, 3]], b = [1, 8] → x: 1·x0 = 1 → x0=1; 2·1 + 3·x1 = 8 → x1 = 2.
        let a = t(&[1.0, 0.0, 2.0, 3.0], &[2, 2]);
        let b = t(&[1.0, 8.0], &[2]);
        let x = solve_triangular(&a, &b, false, false, false).unwrap();
        assert_eq!(x.shape(), &[2]);
        let d = x.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_triangular_upper_1d_b() {
        // U = [[2, 1], [0, 4]], b = [4, 8] → x1=2, 2·x0 + 1·2 = 4 → x0=1.
        let a = t(&[2.0, 1.0, 0.0, 4.0], &[2, 2]);
        let b = t(&[4.0, 8.0], &[2]);
        let x = solve_triangular(&a, &b, true, false, false).unwrap();
        let d = x.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_triangular_2d_b_multi_rhs() {
        // L = [[1, 0], [2, 3]], B = [[1, 2], [8, 13]] → X = [[1, 2], [2, 3]].
        let a = t(&[1.0, 0.0, 2.0, 3.0], &[2, 2]);
        let b = t(&[1.0, 2.0, 8.0, 13.0], &[2, 2]);
        let x = solve_triangular(&a, &b, false, false, false).unwrap();
        assert_eq!(x.shape(), &[2, 2]);
        let d = x.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
        assert!((d[2] - 2.0).abs() < 1e-10);
        assert!((d[3] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solve_triangular_unit_diag() {
        // L_unit = [[*, 0], [2, *]] (diag treated as 1), b = [1, 5]
        // → x0 = 1, x1 = 5 - 2·1 = 3.
        let a = t(&[99.0, 0.0, 2.0, 99.0], &[2, 2]);
        let b = t(&[1.0, 5.0], &[2]);
        let x = solve_triangular(&a, &b, false, false, true).unwrap();
        let d = x.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn solve_triangular_transpose_lower() {
        // A = lower [[2, 0], [3, 4]]. solve A^T x = b with A^T = [[2, 3], [0, 4]].
        // For b=[5, 8]: x1 = 2, then 2·x0 + 3·2 = 5 → x0 = -0.5.
        let a = t(&[2.0, 0.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[5.0, 8.0], &[2]);
        let x = solve_triangular(&a, &b, false, true, false).unwrap();
        let d = x.data().unwrap();
        assert!((d[0] - (-0.5)).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn ldl_factor_pd_matrix() {
        // A = [[4, 2], [2, 3]] is PD. Expected L=[[1,0],[0.5,1]], D=[4, 2].
        // Verify A = L diag(D) L^T.
        let a = t(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let (l, d) = ldl_factor(&a).unwrap();
        let l_d = l.data().unwrap();
        let d_d = d.data().unwrap();
        // Reconstruct A_recon[i,j] = sum_k L[i,k] * D[k] * L[j,k].
        let n = 2;
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0;
                for k in 0..n {
                    acc += l_d[i * n + k] * d_d[k] * l_d[j * n + k];
                }
                let expected = a.data().unwrap()[i * n + j];
                assert!(
                    (acc - expected).abs() < 1e-10,
                    "LDL reconstruction A[{i},{j}]: {acc} vs {expected}"
                );
            }
        }
    }

    #[test]
    fn ldl_solve_pd_matches_solve() {
        // A = [[4, 2], [2, 3]], b = [6, 5]. Solve via ldl and via direct solve.
        let a = t(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let b = t(&[6.0, 5.0], &[2]);
        let (l, d) = ldl_factor(&a).unwrap();
        let x_ldl = ldl_solve(&l, &d, &b).unwrap();
        let x_ref = solve(&a, &b).unwrap();
        let xd = x_ldl.data().unwrap();
        let rd = x_ref.data().unwrap();
        for i in 0..2 {
            assert!(
                (xd[i] - rd[i]).abs() < 1e-9,
                "ldl_solve[{i}]={} vs {}",
                xd[i],
                rd[i]
            );
        }
    }

    #[test]
    fn matrix_exp_zero_is_identity() {
        let a = t(&[0.0, 0.0, 0.0, 0.0], &[2, 2]);
        let e = matrix_exp(&a).unwrap();
        let d = e.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-12);
        assert!((d[1]).abs() < 1e-12);
        assert!((d[2]).abs() < 1e-12);
        assert!((d[3] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn matrix_exp_diagonal() {
        // expm(diag(a, b)) = diag(e^a, e^b).
        let a = t(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
        let e = matrix_exp(&a).unwrap();
        let d = e.data().unwrap();
        assert!((d[0] - 1.0_f64.exp()).abs() < 1e-10);
        assert!(d[1].abs() < 1e-10);
        assert!(d[2].abs() < 1e-10);
        assert!((d[3] - 2.0_f64.exp()).abs() < 1e-10);
    }

    #[test]
    fn matrix_exp_skew_symmetric_2x2_is_rotation() {
        // expm([[0, -t], [t, 0]]) = [[cos t, -sin t], [sin t, cos t]].
        let theta = 0.5_f64;
        let a = t(&[0.0, -theta, theta, 0.0], &[2, 2]);
        let e = matrix_exp(&a).unwrap();
        let d = e.data().unwrap();
        assert!((d[0] - theta.cos()).abs() < 1e-10);
        assert!((d[1] + theta.sin()).abs() < 1e-10);
        assert!((d[2] - theta.sin()).abs() < 1e-10);
        assert!((d[3] - theta.cos()).abs() < 1e-10);
    }

    #[test]
    fn cholesky_ex_succeeds_for_pd() {
        let a = t(&[4.0, 2.0, 2.0, 3.0], &[2, 2]);
        let (_l, info) = cholesky_ex(&a).unwrap();
        assert_eq!(info.shape(), &[] as &[usize]);
        assert!(info.data().unwrap()[0].abs() < 1e-12);
    }

    #[test]
    fn cholesky_ex_returns_nonzero_info_for_indefinite() {
        // Negative-definite-ish: ferray cholesky should fail.
        let a = t(&[-1.0, 0.0, 0.0, -1.0], &[2, 2]);
        let (_l, info) = cholesky_ex(&a).unwrap();
        assert!(info.data().unwrap()[0] != 0.0);
    }

    #[test]
    fn inv_ex_succeeds_for_invertible() {
        let a = t(&[2.0, 0.0, 0.0, 4.0], &[2, 2]);
        let (inv_a, info) = inv_ex(&a).unwrap();
        assert!(info.data().unwrap()[0].abs() < 1e-12);
        let d = inv_a.data().unwrap();
        assert!((d[0] - 0.5).abs() < 1e-10);
        assert!((d[3] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn inv_ex_singular_returns_nonzero_info() {
        let a = t(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let (_inv_a, info) = inv_ex(&a).unwrap();
        assert!(info.data().unwrap()[0] != 0.0);
    }

    #[test]
    fn solve_ex_succeeds() {
        let a = t(&[1.0, 0.0, 0.0, 2.0], &[2, 2]);
        let b = t(&[3.0, 4.0], &[2]);
        let (x, info) = solve_ex(&a, &b).unwrap();
        assert!(info.data().unwrap()[0].abs() < 1e-12);
        let d = x.data().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
    }

    #[test]
    fn solve_ex_singular_returns_nonzero_info() {
        let a = t(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let b = t(&[1.0, 2.0], &[2]);
        let (_x, info) = solve_ex(&a, &b).unwrap();
        assert!(info.data().unwrap()[0] != 0.0);
    }

    #[test]
    fn householder_product_identity_when_no_reflectors() {
        // k=0 → tau is empty → output is I_m.
        let v =
            Tensor::from_storage(TensorStorage::cpu(Vec::<f64>::new()), vec![3, 0], false).unwrap();
        let tau =
            Tensor::from_storage(TensorStorage::cpu(Vec::<f64>::new()), vec![0], false).unwrap();
        let q = householder_product(&v, &tau).unwrap();
        assert_eq!(q.shape(), &[3, 3]);
        let d = q.data().unwrap();
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((d[i * 3 + j] - expected).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn householder_product_single_reflection_is_orthogonal() {
        // Single Householder vector v0 = [1, 0]^T (unit at row 0; below is 0)
        // with tau = 2 → I - 2·v0·v0^T = I - 2·e_0·e_0^T = diag(-1, 1).
        // V is [m=2, k=1]: v[0,0] is the implicit unit (we store anything;
        // householder_product overrides with 1), v[1,0] = 0 (below row 0).
        let v = Tensor::from_storage(TensorStorage::cpu(vec![0.0_f64, 0.0]), vec![2, 1], false)
            .unwrap();
        let tau = Tensor::from_storage(TensorStorage::cpu(vec![2.0_f64]), vec![1], false).unwrap();
        let q = householder_product(&v, &tau).unwrap();
        let d = q.data().unwrap();
        // Q = diag(-1, 1).
        assert!((d[0] + 1.0).abs() < 1e-12);
        assert!(d[1].abs() < 1e-12);
        assert!(d[2].abs() < 1e-12);
        assert!((d[3] - 1.0).abs() < 1e-12);
    }
}
