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
/// Follows PyTorch `torch.matmul` semantics exactly:
///
/// - 1D x 1D: dot product -> scalar
/// - 2D x 1D: matrix-vector multiply (M,K) @ (K,) -> (M,)
/// - 1D x 2D: vector-matrix multiply (K,) @ (K,N) -> (N,)
/// - 2D x 2D: standard matrix multiply (M,K) @ (K,N) -> (M,N)
/// - ≥3D: batched matmul with NumPy-style broadcasting over leading dims.
///   If one input is 1D, it is promoted (prepend dim for LHS, append dim for RHS)
///   and the added dimension is squeezed from the output.
pub fn matmul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::matmul(a, b)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("matmul", "linalg", &[a.shape(), b.shape()], || {
        match (a.ndim(), b.ndim()) {
            (0, _) | (_, 0) => Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "matmul: scalar operands not supported, got shapes {:?} and {:?}",
                    a.shape(),
                    b.shape()
                ),
            }),
            (1, 1) => dot(a, b),
            (2, 1) => mv(a, b),
            (1, 2) => vm(a, b),
            (2, 2) => mm(a, b),
            _ => broadcast_matmul(a, b),
        }
    })
}

/// Broadcast leading dimensions of two shapes according to NumPy rules.
/// Returns the broadcasted batch shape.
fn broadcast_batch_shapes(a: &[usize], b: &[usize]) -> FerrotorchResult<Vec<usize>> {
    let max_len = a.len().max(b.len());
    let mut result = Vec::with_capacity(max_len);
    for i in 0..max_len {
        let da = if i < max_len - a.len() {
            1
        } else {
            a[i - (max_len - a.len())]
        };
        let db = if i < max_len - b.len() {
            1
        } else {
            b[i - (max_len - b.len())]
        };
        if da == db {
            result.push(da);
        } else if da == 1 {
            result.push(db);
        } else if db == 1 {
            result.push(da);
        } else {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "matmul: batch dimensions cannot be broadcast: {:?} vs {:?}",
                    a, b
                ),
            });
        }
    }
    Ok(result)
}

/// Batched matrix multiply with NumPy-style broadcast over leading dimensions.
///
/// Handles all cases where at least one operand has ndim ≥ 3 (and the other
/// is at least 1D). 1D operands are promoted before dispatch and the added
/// dimension is squeezed from the output, matching `torch.matmul`.
fn broadcast_matmul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let device = a.device();
    // --- 1D promotion ------------------------------------------------
    // If a is 1D (K,) → treat as (1, K) and squeeze row from output.
    // If b is 1D (K,) → treat as (K, 1) and squeeze col from output.
    let squeeze_row = a.ndim() == 1;
    let squeeze_col = b.ndim() == 1;

    let a_shape: Vec<usize> = if squeeze_row {
        let mut s = vec![1];
        s.extend_from_slice(a.shape());
        s
    } else {
        a.shape().to_vec()
    };
    let b_shape: Vec<usize> = if squeeze_col {
        let mut s = b.shape().to_vec();
        s.push(1);
        s
    } else {
        b.shape().to_vec()
    };

    let a_nd = a_shape.len();
    let b_nd = b_shape.len();

    // Matrix dims (last two of each).
    let m = a_shape[a_nd - 2];
    let k_a = a_shape[a_nd - 1];
    let k_b = b_shape[b_nd - 2];
    let n = b_shape[b_nd - 1];

    if k_a != k_b {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "matmul: inner dimensions mismatch: {:?} @ {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }
    let k = k_a;

    // Batch dimensions.
    let a_batch = &a_shape[..a_nd - 2];
    let b_batch = &b_shape[..b_nd - 2];
    let batch_shape = broadcast_batch_shapes(a_batch, b_batch)?;
    let batch_size: usize = batch_shape.iter().product::<usize>().max(1);

    // Compute strides for broadcasting iteration.
    let a_batch_strides = broadcast_strides(a_batch, &batch_shape);
    let b_batch_strides = broadcast_strides(b_batch, &batch_shape);

    let a_mat_size = m * k;
    let b_mat_size = k * n;
    let c_mat_size = m * n;

    let a_data = a.data_vec()?;
    let b_data = b.data_vec()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); batch_size * c_mat_size];

    for bi in 0..batch_size {
        // Map flat batch index to a/b offsets using broadcast strides.
        let a_off = batch_linear_index(bi, &a_batch_strides, &batch_shape) * a_mat_size;
        let b_off = batch_linear_index(bi, &b_batch_strides, &batch_shape) * b_mat_size;
        let c_off = bi * c_mat_size;

        for i in 0..m {
            for j in 0..n {
                let mut acc = <T as num_traits::Zero>::zero();
                for p in 0..k {
                    acc += a_data[a_off + i * k + p] * b_data[b_off + p * n + j];
                }
                result[c_off + i * n + j] = acc;
            }
        }
    }

    // Output shape = batch_shape + [m, n], then squeeze promoted dims.
    let mut out_shape = batch_shape;
    out_shape.push(m);
    out_shape.push(n);

    if squeeze_row {
        // Remove the m=1 dimension (second-to-last).
        let pos = out_shape.len() - 2;
        out_shape.remove(pos);
    }
    if squeeze_col {
        // Remove the n=1 dimension (last).
        out_shape.pop();
    }

    let t = Tensor::from_storage(TensorStorage::cpu(result), out_shape, false)?;
    Ok(if device.is_cuda() { t.to(device)? } else { t })
}

/// Compute the strides needed to map a flat index in the broadcast shape
/// back to a flat index in the (possibly smaller) source batch shape.
fn broadcast_strides(src: &[usize], broadcast: &[usize]) -> Vec<usize> {
    let offset = broadcast.len() - src.len();
    let mut strides = vec![0usize; broadcast.len()];

    // Compute row-major strides for the source shape.
    if !src.is_empty() {
        let mut src_strides = vec![1usize; src.len()];
        for i in (0..src.len() - 1).rev() {
            src_strides[i] = src_strides[i + 1] * src[i + 1];
        }

        for (i, stride) in strides.iter_mut().enumerate() {
            if i < offset {
                // Dimension doesn't exist in source — broadcast (stride 0).
                *stride = 0;
            } else {
                let si = i - offset;
                if src[si] == 1 {
                    // Size-1 dimension — broadcast (stride 0).
                    *stride = 0;
                } else {
                    *stride = src_strides[si];
                }
            }
        }
    }

    strides
}

/// Convert a flat batch index into a flat source index using broadcast strides.
fn batch_linear_index(flat: usize, strides: &[usize], shape: &[usize]) -> usize {
    let mut idx = 0;
    let mut remaining = flat;
    // Decompose flat index into multi-index, then dot with strides.
    for i in (0..shape.len()).rev() {
        let coord = remaining % shape[i];
        remaining /= shape[i];
        idx += coord * strides[i];
    }
    idx
}

/// Dot product of two 1-D tensors -> scalar.
pub fn dot<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 1 || b.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "dot requires 1-D tensors, got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
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

/// Threshold for switching from direct ikj loop to faer.
/// For matrices at or below this size, the naive loop avoids faer overhead.
const DIRECT_MM_THRESHOLD: usize = 128;

/// Whether `T` is `half::bf16`. Used to route bf16 kernels through an
/// f32 accumulator to avoid the catastrophic precision loss of summing
/// hundreds of 7-bit-mantissa values in bf16.
#[inline(always)]
fn is_bf16<T: 'static>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>()
}

/// Reinterpret a bf16 slice as `&[half::bf16]`. Only call when
/// `is_bf16::<T>()` is true.
#[inline(always)]
unsafe fn as_bf16_slice<T>(data: &[T]) -> &[half::bf16] {
    // SAFETY: caller guarantees T is half::bf16 (same size, same repr).
    unsafe { &*(data as *const [T] as *const [half::bf16]) }
}

/// Write f32 results into a freshly-zeroed T slice (only valid when T=bf16).
#[inline(always)]
unsafe fn write_f32_as_bf16<T>(dst: &mut [T], src: &[f32]) {
    // SAFETY: caller guarantees T is half::bf16.
    let dst_bf16 = unsafe { &mut *(dst as *mut [T] as *mut [half::bf16]) };
    for (d, &s) in dst_bf16.iter_mut().zip(src.iter()) {
        *d = half::bf16::from_f32(s);
    }
}

/// Choose parallelism for faer matmul.  For medium matrices use sequential
/// to avoid thread-pool overhead; for large matrices use rayon.
#[inline]
fn faer_par(m: usize, k: usize, n: usize) -> faer::Par {
    // Rough heuristic: only pay rayon spawn cost when there's enough work.
    if m * k * n >= 512 * 512 * 512 {
        faer::Par::rayon(0)
    } else {
        faer::Par::Seq
    }
}

/// Raw matrix multiply on borrowed slices: (M,K) @ (K,N) -> Vec<T>.
/// Zero input allocations — operates directly on the borrowed data.
/// This is the hot-path workhorse used by both `mm` and `mm_differentiable`.
pub fn mm_raw<T: Float>(a_data: &[T], b_data: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let max_dim = m.max(n).max(k);
    let zero = <T as num_traits::Zero>::zero();

    if max_dim <= DIRECT_MM_THRESHOLD {
        // Direct ikj loop — cache-friendly, zero intermediate allocations.
        // Uses unsafe get_unchecked to eliminate bounds checks in the hot loop.
        let mut result = vec![zero; m * n];
        if is_bf16::<T>() {
            // bf16 fast path with f32 accumulator. Summing up to
            // DIRECT_MM_THRESHOLD bf16 values in bf16 loses ~7 bits of
            // precision per dot; accumulating in f32 preserves them.
            // SAFETY: is_bf16::<T>() returned true (TypeId::of::<T>() ==
            // TypeId::of::<half::bf16>()), so T == half::bf16. This satisfies
            // the contracts of as_bf16_slice (T == bf16) and write_f32_as_bf16
            // (dst is bf16-layout). All get_unchecked calls index in [0, m*k),
            // [0, k*n), or [0, m*n), which are within the allocated buffers
            // of length m*k, k*n, m*n respectively (acc and result are
            // explicitly sized m*n; a_data/b_data are guaranteed by caller).
            unsafe {
                let a_bf16 = as_bf16_slice(a_data);
                let b_bf16 = as_bf16_slice(b_data);
                let mut acc = vec![0.0f32; m * n];
                for i in 0..m {
                    let a_row = i * k;
                    let r_row = i * n;
                    for p in 0..k {
                        let a_ip = a_bf16.get_unchecked(a_row + p).to_f32();
                        let b_row = p * n;
                        for j in 0..n {
                            *acc.get_unchecked_mut(r_row + j) +=
                                a_ip * b_bf16.get_unchecked(b_row + j).to_f32();
                        }
                    }
                }
                write_f32_as_bf16(&mut result, &acc);
            }
        } else {
            // SAFETY: a_data has length m*k and b_data has length k*n by
            // function contract; result was allocated with `vec![zero; m*n]`.
            // Index arithmetic `i*k + p` with i<m, p<k stays in [0, m*k);
            // `p*n + j` with p<k, j<n stays in [0, k*n); `i*n + j` stays in
            // [0, m*n). All get_unchecked accesses are in bounds.
            unsafe {
                for i in 0..m {
                    let a_row = i * k;
                    let r_row = i * n;
                    for p in 0..k {
                        let a_ip = *a_data.get_unchecked(a_row + p);
                        let b_row = p * n;
                        for j in 0..n {
                            let r = result.get_unchecked_mut(r_row + j);
                            *r += a_ip * *b_data.get_unchecked(b_row + j);
                        }
                    }
                }
            }
        }
        result
    } else {
        // Large matrices — use faer GEMM for high-performance BLAS.
        // faer supports arbitrary strides natively and auto-vectorises with AVX/SSE.
        let mut result = vec![zero; m * n];
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // SAFETY: TypeId guard above proves T == f32, so &[T] and &[f32]
            // have identical layout (size, alignment, niche). The cast is a
            // no-op reinterpretation; lifetimes are tied to a_data/b_data/result.
            let a_f32 = unsafe { &*(a_data as *const [T] as *const [f32]) };
            // SAFETY: same as a_f32 above (T == f32 by the TypeId guard).
            let b_f32 = unsafe { &*(b_data as *const [T] as *const [f32]) };
            // SAFETY: same as a_f32 above (T == f32 by the TypeId guard).
            // `result` was just allocated and is not aliased by a_data/b_data
            // (which are immutable borrows from the caller), so producing a
            // unique &mut [f32] view is sound.
            let c_f32 = unsafe { &mut *(result.as_mut_slice() as *mut [T] as *mut [f32]) };
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f32, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f32, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f32, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f32,
                par,
            );
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // SAFETY: TypeId guard above proves T == f64, so &[T] and &[f64]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f64 = unsafe { &*(a_data as *const [T] as *const [f64]) };
            // SAFETY: same as a_f64 above (T == f64 by the TypeId guard).
            let b_f64 = unsafe { &*(b_data as *const [T] as *const [f64]) };
            // SAFETY: T == f64 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f64] is unique for the duration of this block.
            let c_f64 = unsafe { &mut *(result.as_mut_slice() as *mut [T] as *mut [f64]) };
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f64, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f64, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
        } else {
            // Fallback for f16/bf16: upcast to f64, run faer, downcast.
            let a_f64: Vec<f64> = a_data.iter().map(|&v| v.to_f64().unwrap()).collect();
            let b_f64: Vec<f64> = b_data.iter().map(|&v| v.to_f64().unwrap()).collect();
            let mut r_f64 = vec![0.0f64; m * n];
            let a_mat = faer::mat::MatRef::from_row_major_slice(&a_f64, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(&b_f64, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(&mut r_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
            for (r, &v) in result.iter_mut().zip(r_f64.iter()) {
                *r = T::from(v).unwrap();
            }
        }
        result
    }
}

/// Matrix multiply with B transposed: A @ B^T.
/// A is (M,K), B is (N,K) stored row-major, result is (M,N).
/// For small matrices, uses a direct loop. For large matrices, uses faer GEMM
/// with a zero-copy transposed view of B.
pub fn mm_raw_bt<T: Float>(a_data: &[T], b_data: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let max_dim = m.max(n).max(k);
    let zero = <T as num_traits::Zero>::zero();

    if max_dim <= DIRECT_MM_THRESHOLD {
        // Direct loop — both A row and B row are accessed sequentially.
        // B is (N,K) row-major, so B[j][p] = b_data[j*k + p].
        // C[i][j] = sum_p A[i][p] * B[j][p]
        let mut result = vec![zero; m * n];
        if is_bf16::<T>() {
            // bf16 fast path with f32 accumulator (see note in `mm_raw`).
            // SAFETY: is_bf16::<T>() returned true so T == half::bf16, which
            // satisfies as_bf16_slice and write_f32_as_bf16 contracts. A is
            // (M,K) and B is (N,K), so a_row+p < m*k and b_row+p < n*k for
            // i<m, j<n, p<k. acc_buf and result both have length m*n, so
            // r_row+j = i*n+j < m*n. All get_unchecked accesses are in bounds.
            unsafe {
                let a_bf16 = as_bf16_slice(a_data);
                let b_bf16 = as_bf16_slice(b_data);
                let mut acc_buf = vec![0.0f32; m * n];
                for i in 0..m {
                    let a_row = i * k;
                    let r_row = i * n;
                    for j in 0..n {
                        let b_row = j * k;
                        let mut acc = 0.0f32;
                        for p in 0..k {
                            acc += a_bf16.get_unchecked(a_row + p).to_f32()
                                * b_bf16.get_unchecked(b_row + p).to_f32();
                        }
                        *acc_buf.get_unchecked_mut(r_row + j) = acc;
                    }
                }
                write_f32_as_bf16(&mut result, &acc_buf);
            }
        } else {
            // SAFETY: A is (M,K) and B is (N,K) row-major (function contract);
            // result was allocated with `vec![zero; m*n]`. Index arithmetic
            // i*k+p < m*k for i<m,p<k; j*k+p < n*k for j<n,p<k; i*n+j < m*n.
            // All get_unchecked accesses are in bounds for the respective slices.
            unsafe {
                for i in 0..m {
                    let a_row = i * k;
                    let r_row = i * n;
                    for j in 0..n {
                        let b_row = j * k;
                        let mut acc = zero;
                        for p in 0..k {
                            acc +=
                                *a_data.get_unchecked(a_row + p) * *b_data.get_unchecked(b_row + p);
                        }
                        *result.get_unchecked_mut(r_row + j) = acc;
                    }
                }
            }
        }
        result
    } else {
        // Large matrices — use faer GEMM with zero-copy transposed B view.
        // B is (N,K) row-major. Wrap as (N,K) MatRef then .transpose() to get (K,N).
        let mut result = vec![zero; m * n];
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // SAFETY: TypeId guard above proves T == f32, so &[T] and &[f32]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f32 = unsafe { &*(a_data as *const [T] as *const [f32]) };
            // SAFETY: same as a_f32 above (T == f32 by the TypeId guard).
            let b_f32 = unsafe { &*(b_data as *const [T] as *const [f32]) };
            // SAFETY: T == f32 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f32] is unique for the duration of this block.
            let c_f32 = unsafe { &mut *(result.as_mut_slice() as *mut [T] as *mut [f32]) };
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f32, m, k);
            // B is (N,K) row-major; transpose gives (K,N) view — zero copy.
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f32, n, k).transpose();
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f32, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f32,
                par,
            );
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // SAFETY: TypeId guard above proves T == f64, so &[T] and &[f64]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f64 = unsafe { &*(a_data as *const [T] as *const [f64]) };
            // SAFETY: same as a_f64 above (T == f64 by the TypeId guard).
            let b_f64 = unsafe { &*(b_data as *const [T] as *const [f64]) };
            // SAFETY: T == f64 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f64] is unique for the duration of this block.
            let c_f64 = unsafe { &mut *(result.as_mut_slice() as *mut [T] as *mut [f64]) };
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f64, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f64, n, k).transpose();
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
        } else {
            let a_f64: Vec<f64> = a_data.iter().map(|&v| v.to_f64().unwrap()).collect();
            let b_f64: Vec<f64> = b_data.iter().map(|&v| v.to_f64().unwrap()).collect();
            let mut r_f64 = vec![0.0f64; m * n];
            let a_mat = faer::mat::MatRef::from_row_major_slice(&a_f64, m, k);
            let b_mat = faer::mat::MatRef::from_row_major_slice(&b_f64, n, k).transpose();
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(&mut r_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
            for (r, &v) in result.iter_mut().zip(r_f64.iter()) {
                *r = T::from(v).unwrap();
            }
        }
        result
    }
}

/// Matrix multiply with A transposed: A^T @ B.
/// A is (K,M) stored row-major, B is (K,N) row-major, result is (M,N).
/// Computes C[i,j] = sum_k A[k,i] * B[k,j] = A^T @ B without materializing the transpose.
pub fn mm_raw_at<T: Float>(a_data: &[T], b_data: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let max_dim = m.max(n).max(k);
    let zero = <T as num_traits::Zero>::zero();

    if max_dim <= DIRECT_MM_THRESHOLD {
        // Direct loop: A is (K,M) row-major, B is (K,N) row-major.
        // C[i,j] = sum_p A[p,i] * B[p,j]
        let mut result = vec![zero; m * n];
        if is_bf16::<T>() {
            // bf16 fast path with f32 accumulator (see note in `mm_raw`).
            // SAFETY: is_bf16::<T>() returned true so T == half::bf16,
            // satisfying as_bf16_slice and write_f32_as_bf16 contracts.
            // A is (K,M) and B is (K,N), so for p<k, i<m, j<n we have
            // a_row+i = p*m+i < k*m and b_row+j = p*n+j < k*n. acc_buf and
            // result both have length m*n, so r_row+j = i*n+j < m*n. All
            // get_unchecked accesses are in bounds.
            unsafe {
                let a_bf16 = as_bf16_slice(a_data);
                let b_bf16 = as_bf16_slice(b_data);
                let mut acc_buf = vec![0.0f32; m * n];
                for p in 0..k {
                    let a_row = p * m;
                    let b_row = p * n;
                    for i in 0..m {
                        let a_val = a_bf16.get_unchecked(a_row + i).to_f32();
                        let r_row = i * n;
                        for j in 0..n {
                            *acc_buf.get_unchecked_mut(r_row + j) +=
                                a_val * b_bf16.get_unchecked(b_row + j).to_f32();
                        }
                    }
                }
                write_f32_as_bf16(&mut result, &acc_buf);
            }
        } else {
            // SAFETY: A is (K,M) and B is (K,N) row-major (function contract);
            // result was allocated with `vec![zero; m*n]`. p*m+i < k*m for
            // p<k,i<m; p*n+j < k*n for p<k,j<n; i*n+j < m*n for i<m,j<n.
            // All get_unchecked accesses are in bounds.
            unsafe {
                for p in 0..k {
                    let a_row = p * m;
                    let b_row = p * n;
                    for i in 0..m {
                        let a_val = *a_data.get_unchecked(a_row + i);
                        let r_row = i * n;
                        for j in 0..n {
                            let r = result.get_unchecked_mut(r_row + j);
                            *r += a_val * *b_data.get_unchecked(b_row + j);
                        }
                    }
                }
            }
        }
        result
    } else {
        // Large matrices — use faer GEMM with zero-copy transposed A view.
        // A is (K,M) row-major. Wrap as (K,M) MatRef then .transpose() to get (M,K).
        let mut result = vec![zero; m * n];
        if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>() {
            // SAFETY: TypeId guard above proves T == f32, so &[T] and &[f32]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f32 = unsafe { &*(a_data as *const [T] as *const [f32]) };
            // SAFETY: same as a_f32 above (T == f32 by the TypeId guard).
            let b_f32 = unsafe { &*(b_data as *const [T] as *const [f32]) };
            // SAFETY: T == f32 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f32] is unique for the duration of this block.
            let c_f32 = unsafe { &mut *(result.as_mut_slice() as *mut [T] as *mut [f32]) };
            // A is (K,M) row-major; transpose gives (M,K) view — zero copy.
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f32, k, m).transpose();
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f32, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f32, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f32,
                par,
            );
        } else if std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>() {
            // SAFETY: TypeId guard above proves T == f64, so &[T] and &[f64]
            // have identical layout; the cast is a layout-preserving reinterpret.
            let a_f64 = unsafe { &*(a_data as *const [T] as *const [f64]) };
            // SAFETY: same as a_f64 above (T == f64 by the TypeId guard).
            let b_f64 = unsafe { &*(b_data as *const [T] as *const [f64]) };
            // SAFETY: T == f64 by the TypeId guard. `result` was just allocated
            // and is not aliased by the immutable a_data/b_data borrows, so the
            // produced &mut [f64] is unique for the duration of this block.
            let c_f64 = unsafe { &mut *(result.as_mut_slice() as *mut [T] as *mut [f64]) };
            let a_mat = faer::mat::MatRef::from_row_major_slice(a_f64, k, m).transpose();
            let b_mat = faer::mat::MatRef::from_row_major_slice(b_f64, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(c_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
        } else {
            let a_f64: Vec<f64> = a_data.iter().map(|&v| v.to_f64().unwrap()).collect();
            let b_f64: Vec<f64> = b_data.iter().map(|&v| v.to_f64().unwrap()).collect();
            let mut r_f64 = vec![0.0f64; m * n];
            let a_mat = faer::mat::MatRef::from_row_major_slice(&a_f64, k, m).transpose();
            let b_mat = faer::mat::MatRef::from_row_major_slice(&b_f64, k, n);
            let mut c_mat = faer::mat::MatMut::from_row_major_slice_mut(&mut r_f64, m, n);
            let par = faer_par(m, k, n);
            faer::linalg::matmul::matmul(
                &mut c_mat,
                faer::Accum::Replace,
                &a_mat,
                &b_mat,
                1.0f64,
                par,
            );
            for (r, &v) in result.iter_mut().zip(r_f64.iter()) {
                *r = T::from(v).unwrap();
            }
        }
        result
    }
}

/// Matrix-matrix multiply: (M,K) @ (K,N) -> (M,N).
pub fn mm<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 2 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mm requires 2-D tensors, got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }

    // Materialize non-contiguous views (e.g. from transpose/permute).
    let a = if a.is_contiguous() {
        a.clone()
    } else {
        a.contiguous()?
    };
    let b = if b.is_contiguous() {
        b.clone()
    } else {
        b.contiguous()?
    };

    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[1];

    if k != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mm: inner dimensions mismatch: ({},{}) @ ({},{})",
                m,
                k,
                b.shape()[0],
                n
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let result = mm_raw(a_data, b_data, m, k, n);

    Tensor::from_storage(TensorStorage::cpu(result), vec![m, n], false)
}

/// Matrix-vector multiply: (M,K) @ (K,) -> (M,).
pub fn mv<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 2 || b.ndim() != 1 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mv requires (2-D, 1-D), got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }

    let m = a.shape()[0];
    let k = a.shape()[1];

    if k != b.shape()[0] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mv: dimension mismatch: ({},{}) @ ({},)",
                m,
                k,
                b.shape()[0]
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); m];

    for i in 0..m {
        let mut acc = <T as num_traits::Zero>::zero();
        for p in 0..k {
            acc += a_data[i * k + p] * b_data[p];
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
            message: format!(
                "vm: dimension mismatch: ({},) @ ({},{})",
                k,
                b.shape()[0],
                n
            ),
        });
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut result = vec![<T as num_traits::Zero>::zero(); n];

    for j in 0..n {
        let mut acc = <T as num_traits::Zero>::zero();
        for p in 0..k {
            acc += a_data[p] * b_data[p * n + j];
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
                batch,
                m,
                k,
                b.shape()[0],
                b.shape()[1],
                n
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
                    acc += a_data[a_off + i * k + p] * b_data[b_off + p * n + j];
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

    // -------------------------------------------------------------------
    // broadcast_matmul tests
    // -------------------------------------------------------------------

    #[test]
    fn test_matmul_3d_3d_same_batch() {
        // (2, 2, 3) @ (2, 3, 2) -> (2, 2, 2)
        // Batch 0: [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[1,0]] = [[4,2],[10,5]]
        // Batch 1: identity-like
        #[rustfmt::skip]
        let a = t(&[
            1.0, 2.0, 3.0,  4.0, 5.0, 6.0,   // batch 0
            1.0, 0.0, 0.0,  0.0, 1.0, 0.0,   // batch 1
        ], &[2, 2, 3]);
        #[rustfmt::skip]
        let b = t(&[
            1.0, 0.0,  0.0, 1.0,  1.0, 0.0,  // batch 0
            1.0, 2.0,  3.0, 4.0,  5.0, 6.0,  // batch 1
        ], &[2, 3, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);
        let d = c.data().unwrap();
        // Batch 0: [[1*1+2*0+3*1, 1*0+2*1+3*0], [4*1+5*0+6*1, 4*0+5*1+6*0]]
        //        = [[4, 2], [10, 5]]
        assert!((d[0] - 4.0).abs() < 1e-6);
        assert!((d[1] - 2.0).abs() < 1e-6);
        assert!((d[2] - 10.0).abs() < 1e-6);
        assert!((d[3] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_matmul_3d_2d_broadcast() {
        // (2, 3, 4) @ (4, 2) -> (2, 3, 2)
        // The 2D right operand broadcasts over the batch dim.
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 4 * 2], &[4, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3, 2]);
        // Each element = sum of 4 ones = 4.0
        for &v in c.data().unwrap().iter() {
            assert!((v - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_2d_3d_broadcast() {
        // (3, 4) @ (2, 4, 2) -> (2, 3, 2)
        let a = t(&[1.0; 3 * 4], &[3, 4]);
        let b = t(&[1.0; 2 * 4 * 2], &[2, 4, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3, 2]);
    }

    #[test]
    fn test_matmul_batch_broadcast_1_vs_n() {
        // (1, 2, 3) @ (4, 3, 2) -> (4, 2, 2) — batch dim 1 broadcasts to 4
        let a = t(&[1.0; 2 * 3], &[1, 2, 3]);
        let b = t(&[1.0; 4 * 3 * 2], &[4, 3, 2]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[4, 2, 2]);
    }

    #[test]
    fn test_matmul_4d() {
        // (2, 3, 2, 4) @ (2, 3, 4, 5) -> (2, 3, 2, 5)
        let a = t(&[1.0; 2 * 3 * 2 * 4], &[2, 3, 2, 4]);
        let b = t(&vec![1.0; 2 * 3 * 4 * 5], &[2, 3, 4, 5]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3, 2, 5]);
    }

    #[test]
    fn test_matmul_3d_1d() {
        // (2, 3, 4) @ (4,) -> (2, 3) — 1D promoted to (4,1), col squeezed
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 4], &[4]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
        for &v in c.data().unwrap().iter() {
            assert!((v - 4.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_1d_3d() {
        // (4,) @ (2, 4, 3) -> (2, 3) — 1D promoted to (1,4), row squeezed
        let a = t(&[1.0; 4], &[4]);
        let b = t(&[1.0; 2 * 4 * 3], &[2, 4, 3]);
        let c = matmul(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 3]);
    }

    #[test]
    fn test_matmul_broadcast_mismatch() {
        // (2, 3, 4) @ (3, 4, 2) — batch dims 2 vs 3, not broadcastable
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 3 * 4 * 2], &[3, 4, 2]);
        assert!(matmul(&a, &b).is_err());
    }

    #[test]
    fn test_matmul_inner_dim_mismatch() {
        // (2, 3, 4) @ (2, 5, 2) — inner dims 4 vs 5
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 2 * 5 * 2], &[2, 5, 2]);
        assert!(matmul(&a, &b).is_err());
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
        let a = t(&[1.0; 2 * 3 * 4], &[2, 3, 4]);
        let b = t(&[1.0; 2 * 4 * 5], &[2, 4, 5]);
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
        let a = t(&[1.0; 2 * 2 * 2], &[2, 2, 2]);
        let b = t(&[1.0; 3 * 2 * 2], &[3, 2, 2]);
        assert!(bmm(&a, &b).is_err());

        // Inner dimension mismatch.
        let a = t(&[1.0; 2 * 2 * 3], &[2, 2, 3]);
        let b = t(&[1.0; 2 * 4 * 2], &[2, 4, 2]);
        assert!(bmm(&a, &b).is_err());

        // Wrong ndim.
        let a = t(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t(&[1.0; 2 * 2], &[1, 2, 2]);
        assert!(bmm(&a, &b).is_err());
    }
}
