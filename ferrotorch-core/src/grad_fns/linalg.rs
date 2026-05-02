//! Backward functions for linear algebra operations.
//!
//! Each struct captures the forward inputs and implements `GradFn` to
//! compute VJPs (vector-Jacobian products) for reverse-mode autodiff.

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::autocast_ops::{AutocastCategory, autocast_guard};
use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::gpu_dispatch::gpu_backend;
use crate::ops::linalg::{self, mm, transpose};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Type alias for a pair of optional tensor gradients (used by matmul backward).
type GradPair<T> = FerrotorchResult<(Option<Tensor<T>>, Option<Tensor<T>>)>;

/// Returns `true` if `T` is `f32`.
#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

/// Returns `true` if `T` is `f64`.
#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// GPU-native matmul backward for f32 tensors.
/// dA = grad_C @ B^T, dB = A^T @ grad_C — all on GPU, no CPU roundtrip.
fn mm_backward_gpu<T: Float>(grad_output: &Tensor<T>, a: &Tensor<T>, b: &Tensor<T>) -> GradPair<T> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
    let go_h = grad_output.gpu_handle()?;
    let m = grad_output.shape()[0];
    let n = grad_output.shape()[1];
    let f64_path = is_f64::<T>();

    let grad_a = if a.requires_grad() {
        let k = b.shape()[0];
        let b_h = b.gpu_handle()?;
        let (bt_h, result_h) = if f64_path {
            let bt = backend.transpose_2d_f64(b_h, k, n)?;
            let r = backend.matmul_f64(go_h, &bt, m, n, k)?;
            (bt, r)
        } else {
            let bt = backend.transpose_2d_f32(b_h, k, n)?;
            let r = backend.matmul_f32(go_h, &bt, m, n, k)?;
            (bt, r)
        };
        let _ = bt_h;
        Some(Tensor::from_storage(
            TensorStorage::gpu(result_h),
            vec![m, k],
            false,
        )?)
    } else {
        None
    };

    let grad_b = if b.requires_grad() {
        let k = a.shape()[1];
        let a_h = a.gpu_handle()?;
        let (at_h, result_h) = if f64_path {
            let at = backend.transpose_2d_f64(a_h, m, k)?;
            let r = backend.matmul_f64(&at, go_h, k, m, n)?;
            (at, r)
        } else {
            let at = backend.transpose_2d_f32(a_h, m, k)?;
            let r = backend.matmul_f32(&at, go_h, k, m, n)?;
            (at, r)
        };
        let _ = at_h;
        Some(Tensor::from_storage(
            TensorStorage::gpu(result_h),
            vec![k, n],
            false,
        )?)
    } else {
        None
    };

    Ok((grad_a, grad_b))
}

// ---------------------------------------------------------------------------
// MmBackward — C = A @ B  (2D x 2D)
// ---------------------------------------------------------------------------

/// Backward for matrix-matrix multiply: `C = mm(A, B)`.
///
/// VJP formulas:
/// - `dA = grad_C @ B^T`
/// - `dB = A^T @ grad_C`
#[derive(Debug)]
pub struct MmBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> MmBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for MmBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // GPU-native path for f32/f64.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let (ga, gb) = mm_backward_gpu(grad_output, &self.a, &self.b)?;
            return Ok(vec![ga, gb]);
        }

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "MmBackward" });
        }

        // CPU path.
        let grad_a = if self.a.requires_grad() {
            let gc_data = grad_output.data()?;
            let b_data = self.b.data()?;
            let m = grad_output.shape()[0];
            let n = grad_output.shape()[1];
            let k = self.b.shape()[0];
            let result = crate::ops::linalg::mm_raw_bt(gc_data, b_data, m, n, k);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![m, k],
                false,
            )?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let a_data = self.a.data()?;
            let gc_data = grad_output.data()?;
            let m = self.a.shape()[0];
            let k = self.a.shape()[1];
            let n = grad_output.shape()[1];
            let result = crate::ops::linalg::mm_raw_at(a_data, gc_data, k, m, n);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![k, n],
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MmBackward"
    }
}

// ---------------------------------------------------------------------------
// MvBackward — y = A @ x  (2D x 1D)
// ---------------------------------------------------------------------------

/// Backward for matrix-vector multiply: `y = mv(A, x)`.
///
/// VJP formulas:
/// - `dA = outer(grad_y, x)`   — i.e. `dA[i,j] = grad_y[i] * x[j]`
/// - `dx = A^T @ grad_y`
#[derive(Debug)]
pub struct MvBackward<T: Float> {
    a: Tensor<T>,
    x: Tensor<T>,
}

impl<T: Float> MvBackward<T> {
    pub fn new(a: Tensor<T>, x: Tensor<T>) -> Self {
        Self { a, x }
    }
}

impl<T: Float> GradFn<T> for MvBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.a.is_cuda() || self.x.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "MvBackward" });
        }

        // grad_output is shape (M,) — the upstream gradient on y.
        let grad_a = if self.a.requires_grad() {
            // dA = outer(grad_y, x): shape (M, K)
            let grad_data = grad_output.data()?;
            let x_data = self.x.data()?;
            let m = grad_data.len();
            let k = x_data.len();
            let mut outer = vec![<T as num_traits::Zero>::zero(); m * k];
            for i in 0..m {
                for j in 0..k {
                    outer[i * k + j] = grad_data[i] * x_data[j];
                }
            }
            Some(Tensor::from_storage(
                TensorStorage::cpu(outer),
                vec![m, k],
                false,
            )?)
        } else {
            None
        };

        let grad_x = if self.x.requires_grad() {
            // dx = A^T @ grad_y
            let at = transpose(&self.a)?;
            Some(linalg::mv(&at, grad_output)?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_x])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.x]
    }

    fn name(&self) -> &'static str {
        "MvBackward"
    }
}

// ---------------------------------------------------------------------------
// DotBackward — s = dot(a, b)  (1D x 1D -> scalar)
// ---------------------------------------------------------------------------

/// Backward for dot product: `s = dot(a, b)`.
///
/// VJP formulas:
/// - `da = grad_s * b`
/// - `db = grad_s * a`
#[derive(Debug)]
pub struct DotBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> DotBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for DotBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() || self.a.is_cuda() || self.b.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "DotBackward" });
        }
        let s = grad_output.item()?;

        let grad_a = if self.a.requires_grad() {
            let b_data = self.b.data()?;
            let result: Vec<T> = b_data.iter().map(|&v| s * v).collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                self.a.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let a_data = self.a.data()?;
            let result: Vec<T> = a_data.iter().map(|&v| s * v).collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                self.b.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "DotBackward"
    }
}

// ---------------------------------------------------------------------------
// batch_transpose — swap dims 1 and 2 of a 3D tensor
// ---------------------------------------------------------------------------

/// Transpose dims 1 and 2 of a 3D tensor: `[batch, r, c]` → `[batch, c, r]`.
///
/// This is a data rearrangement (not a view) that works on any device.
/// Used by `BmmBackward` to compute `bmm(grad_C, B^T)` and `bmm(A^T, grad_C)`.
fn batch_transpose<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // Use permute + contiguous to transpose dims 1 and 2 entirely on-device.
    // This avoids the GPU→CPU→GPU roundtrip that dominated BmmBackward cost.
    input.permute(&[0, 2, 1])?.contiguous()
}

// ---------------------------------------------------------------------------
// BmmBackward — C[b] = A[b] @ B[b]  (3D batched matmul)
// ---------------------------------------------------------------------------

/// Backward for batched matrix-matrix multiply: `C = bmm(A, B)`.
///
/// VJP formulas (per batch element `b`):
/// - `dA[b] = grad_C[b] @ B[b]^T`
/// - `dB[b] = A[b]^T @ grad_C[b]`
#[derive(Debug)]
pub struct BmmBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> BmmBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for BmmBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // PyTorch approach: grad_A = bmm(grad_C, B^T), grad_B = bmm(A^T, grad_C)
        // where ^T transposes dims 1 and 2. Uses the same GPU-aware bmm path.
        let grad_a = if self.a.requires_grad() {
            let bt = batch_transpose(&self.b)?;
            Some(crate::autograd::no_grad::no_grad(|| bmm(grad_output, &bt))?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let at = batch_transpose(&self.a)?;
            Some(crate::autograd::no_grad::no_grad(|| bmm(&at, grad_output))?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "BmmBackward"
    }
}

// ---------------------------------------------------------------------------
// MatmulBackward — dispatches based on input shapes
// ---------------------------------------------------------------------------

/// Backward for the general `matmul` dispatcher.
///
/// Internally delegates to `MmBackward`, `MvBackward`, `DotBackward`,
/// or the vm path depending on the operand ranks at forward time.
#[derive(Debug)]
pub struct MatmulBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> MatmulBackward<T> {
    pub fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for MatmulBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        match (self.a.ndim(), self.b.ndim()) {
            (2, 2) => {
                let inner = MmBackward::new(self.a.clone(), self.b.clone());
                inner.backward(grad_output)
            }
            (2, 1) => {
                let inner = MvBackward::new(self.a.clone(), self.b.clone());
                inner.backward(grad_output)
            }
            (1, 1) => {
                let inner = DotBackward::new(self.a.clone(), self.b.clone());
                inner.backward(grad_output)
            }
            (1, 2) => {
                // vm: y = a @ B where a is (K,), B is (K,N), y is (N,)
                if grad_output.is_cuda() || self.a.is_cuda() || self.b.is_cuda() {
                    return Err(FerrotorchError::NotImplementedOnCuda {
                        op: "MatmulBackward(vm)",
                    });
                }

                let grad_a = if self.a.requires_grad() {
                    Some(linalg::mv(&self.b, grad_output)?)
                } else {
                    None
                };

                let grad_b = if self.b.requires_grad() {
                    let a_data = self.a.data()?;
                    let grad_data = grad_output.data()?;
                    let k = a_data.len();
                    let n = grad_data.len();
                    let mut outer = vec![<T as num_traits::Zero>::zero(); k * n];
                    for ki in 0..k {
                        for ni in 0..n {
                            outer[ki * n + ni] = a_data[ki] * grad_data[ni];
                        }
                    }
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(outer),
                        vec![k, n],
                        false,
                    )?)
                } else {
                    None
                };

                Ok(vec![grad_a, grad_b])
            }
            _ => {
                // Batched broadcast matmul backward.
                // For C = matmul(A, B) where shapes may broadcast:
                //   dA = matmul(grad_C, B^T)  — then sum-reduce broadcast dims
                //   dB = matmul(A^T, grad_C)  — then sum-reduce broadcast dims
                //
                // "Transpose" here means swapping the last two dims.
                // After computing the full broadcast gradient, we sum over
                // any dimensions that were broadcast (size-1 in original).
                broadcast_matmul_backward(&self.a, &self.b, grad_output)
            }
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MatmulBackward"
    }
}

/// Backward pass for batched broadcast matmul.
///
/// Given forward: `C = matmul(A, B)` where A and B may have broadcast
/// batch dimensions, computes:
/// - `grad_A = matmul(grad_C, B_transposed)` summed over broadcast dims
/// - `grad_B = matmul(A_transposed, grad_C)` summed over broadcast dims
fn broadcast_matmul_backward<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
    grad_output: &Tensor<T>,
) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
    // Transpose last two dims of a tensor (swap matrix dims in batched tensor).
    let swap_last_two = |t: &Tensor<T>| -> FerrotorchResult<Tensor<T>> {
        let shape = t.shape();
        let nd = shape.len();
        if nd < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Cannot transpose last two dims of tensor with ndim < 2".into(),
            });
        }
        if t.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "MatmulBackward(broadcast)",
            });
        }
        let data = t.data()?;
        let rows = shape[nd - 2];
        let cols = shape[nd - 1];
        let mat_size = rows * cols;
        let n_mats: usize = shape[..nd - 2].iter().product::<usize>().max(1);
        let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];
        for m in 0..n_mats {
            let off = m * mat_size;
            for i in 0..rows {
                for j in 0..cols {
                    out[off + j * rows + i] = data[off + i * cols + j];
                }
            }
        }
        let mut out_shape = shape.to_vec();
        out_shape[nd - 2] = cols;
        out_shape[nd - 1] = rows;
        Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
    };

    // Sum-reduce grad to match the original shape. This handles the case
    // where a dimension was size-1 (broadcast) in the original but expanded
    // in the gradient. We need to sum over those expanded dimensions.
    let reduce_to_shape = |grad: Tensor<T>, target: &[usize]| -> FerrotorchResult<Tensor<T>> {
        let grad_shape = grad.shape().to_vec();
        if grad_shape == target {
            return Ok(grad);
        }
        if grad.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "MatmulBackward(broadcast)",
            });
        }

        let grad_nd = grad_shape.len();
        let target_nd = target.len();
        let offset = grad_nd - target_nd;
        let grad_data = grad.data()?;

        // Compute target total size.
        let target_size: usize = target.iter().product::<usize>().max(1);
        let mut result = vec![<T as num_traits::Zero>::zero(); target_size];

        let grad_total: usize = grad_shape.iter().product::<usize>().max(1);

        // For each element in the gradient, compute which element in the
        // target it maps to, and accumulate.
        // Build stride tables for both shapes.
        let mut grad_strides = vec![1usize; grad_nd];
        for i in (0..grad_nd.saturating_sub(1)).rev() {
            grad_strides[i] = grad_strides[i + 1] * grad_shape[i + 1];
        }

        let mut target_strides = vec![1usize; target_nd];
        if target_nd > 0 {
            for i in (0..target_nd.saturating_sub(1)).rev() {
                target_strides[i] = target_strides[i + 1] * target[i + 1];
            }
        }

        for (flat, &grad_val) in grad_data.iter().enumerate().take(grad_total) {
            // Decompose flat index into grad multi-index.
            let mut remaining = flat;
            let mut target_flat = 0usize;
            for d in (0..grad_nd).rev() {
                let coord = remaining % grad_shape[d];
                remaining /= grad_shape[d];

                // Map to target dimension.
                if d >= offset {
                    let td = d - offset;
                    let target_coord = if target[td] == 1 { 0 } else { coord };
                    target_flat += target_coord * target_strides[td];
                }
                // If d < offset, this dimension doesn't exist in target — summed out.
            }
            result[target_flat] += grad_val;
        }

        Tensor::from_storage(TensorStorage::cpu(result), target.to_vec(), false)
    };

    let grad_a = if a.requires_grad() {
        // grad_A = matmul(grad_C, B^T) reduced to A's shape.
        let bt = swap_last_two(b)?;
        let full_grad = linalg::matmul(grad_output, &bt)?;
        Some(reduce_to_shape(full_grad, a.shape())?)
    } else {
        None
    };

    let grad_b = if b.requires_grad() {
        // grad_B = matmul(A^T, grad_C) reduced to B's shape.
        let at = swap_last_two(a)?;
        let full_grad = linalg::matmul(&at, grad_output)?;
        Some(reduce_to_shape(full_grad, b.shape())?)
    } else {
        None
    };

    Ok(vec![grad_a, grad_b])
}

// ---------------------------------------------------------------------------
// Differentiable forward wrappers
// ---------------------------------------------------------------------------

/// Differentiable matrix-matrix multiply. If either input requires grad and
/// grad is enabled, attaches `MmBackward`.
pub fn mm_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    // Materialize non-contiguous views before linalg ops.
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

    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        // When autocast says ReducedPrecision and inputs are f32 on GPU,
        // use the f16-accumulate path (falls back to f32 if no kernel).
        let handle =
            if is_f32::<T>() && autocast_guard("mm") == Some(AutocastCategory::ReducedPrecision) {
                backend.matmul_f16_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)?
            } else {
                backend.matmul_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)?
            };
        let storage = TensorStorage::gpu(handle);
        let shape = vec![m, n];

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MmBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
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

        // Compute result directly — no intermediate Tensor allocation.
        let result_vec = linalg::mm_raw(a_data, b_data, m, k, n);
        let storage = TensorStorage::cpu(result_vec);
        let shape = vec![m, n];

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MmBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    }
}

// ---------------------------------------------------------------------------
// MmBtBackward — C = A @ B^T  (fused transpose, no materialized B^T)
// ---------------------------------------------------------------------------

/// Backward for fused `C = A @ B^T` (B is NOT transposed in storage).
///
/// Forward: C[i,j] = sum_k A[i,k] * B[j,k]  (B is (N,K) row-major)
///
/// VJP:
/// - `dA = grad_C @ B`   (no transpose — B is already in the right layout)
/// - `dB = grad_C^T @ A` (which is the same as grad_C transposed times A)
#[derive(Debug)]
struct MmBtBackward<T: Float> {
    a: Tensor<T>, // (M, K)
    b: Tensor<T>, // (N, K) — original, not transposed
}

impl<T: Float> MmBtBackward<T> {
    fn new(a: Tensor<T>, b: Tensor<T>) -> Self {
        Self { a, b }
    }
}

impl<T: Float> GradFn<T> for MmBtBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // GPU-native path for f32/f64.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let go_h = grad_output.gpu_handle()?;
            let m = grad_output.shape()[0];
            let n = grad_output.shape()[1];
            let f64_path = is_f64::<T>();

            let grad_a = if self.a.requires_grad() {
                let k = self.b.shape()[1];
                let b_h = self.b.gpu_handle()?;
                let result_h = if f64_path {
                    backend.matmul_f64(go_h, b_h, m, n, k)?
                } else {
                    backend.matmul_f32(go_h, b_h, m, n, k)?
                };
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![m, k],
                    false,
                )?)
            } else {
                None
            };

            let grad_b = if self.b.requires_grad() {
                let k = self.a.shape()[1];
                let a_h = self.a.gpu_handle()?;
                let (got_h, result_h) = if f64_path {
                    let got = backend.transpose_2d_f64(go_h, m, n)?;
                    let r = backend.matmul_f64(&got, a_h, n, m, k)?;
                    (got, r)
                } else {
                    let got = backend.transpose_2d_f32(go_h, m, n)?;
                    let r = backend.matmul_f32(&got, a_h, n, m, k)?;
                    (got, r)
                };
                let _ = got_h;
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![n, k],
                    false,
                )?)
            } else {
                None
            };

            return Ok(vec![grad_a, grad_b]);
        }

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "MmBtBackward" });
        }

        let grad_a = if self.a.requires_grad() {
            Some(mm(grad_output, &self.b)?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            let gc_data = grad_output.data()?;
            let a_data = self.a.data()?;
            let m = grad_output.shape()[0];
            let n = grad_output.shape()[1];
            let k = self.a.shape()[1];
            let result = crate::ops::linalg::mm_raw_at(gc_data, a_data, n, m, k);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![n, k],
                false,
            )?)
        } else {
            None
        };

        Ok(vec![grad_a, grad_b])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MmBtBackward"
    }
}

/// Fused differentiable `A @ B^T`. Avoids materializing the transpose of B.
///
/// A: (M, K), B: (N, K) -> result: (M, N)
/// Linear layer uses this: input @ weight^T where weight is (out, in).
pub fn mm_bt_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let m = a.shape()[0];
    let k = a.shape()[1];
    let n = b.shape()[0];

    if b.ndim() != 2 || b.shape()[1] != k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "mm_bt: A is ({},{}) but B is {:?} (expected ({},{}))",
                m,
                k,
                b.shape(),
                n,
                k
            ),
        });
    }

    // GPU path: transpose B then matmul.
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let bt_handle = backend.transpose_2d_f32(b.gpu_handle()?, n, k)?;
        let handle = backend.matmul_f32(a.gpu_handle()?, &bt_handle, m, k, n)?;
        let storage = TensorStorage::gpu(handle);
        let shape = vec![m, n];

        return if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MmBtBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    let a_data = a.data()?;
    let b_data = b.data()?;
    let result_vec = linalg::mm_raw_bt(a_data, b_data, m, k, n);
    let storage = TensorStorage::cpu(result_vec);
    let shape = vec![m, n];

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(MmBtBackward::new(a.clone(), b.clone()));
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// Fused Linear: C = A @ W^T + bias  (avoids intermediate tensors)
// ---------------------------------------------------------------------------

/// Backward for fused linear: C = A @ W^T + bias.
/// grad_A = grad_C @ W, grad_W = grad_C^T @ A, grad_bias = sum(grad_C, dim=0).
#[derive(Debug)]
struct LinearFusedBackward<T: Float> {
    input: Tensor<T>,  // (M, K)
    weight: Tensor<T>, // (N, K) — not transposed
    has_bias: bool,
    bias: Option<Tensor<T>>, // (N,)
}

impl<T: Float> GradFn<T> for LinearFusedBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let m = grad_output.shape()[0];
        let n = grad_output.shape()[1];

        // GPU-native path for f32/f64 tensors.
        if grad_output.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
            let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
            let go_h = grad_output.gpu_handle()?;
            let f64_path = is_f64::<T>();

            let grad_input = if self.input.requires_grad() {
                let k = self.weight.shape()[1];
                let w_h = self.weight.gpu_handle()?;
                let result_h = if f64_path {
                    backend.matmul_f64(go_h, w_h, m, n, k)?
                } else {
                    backend.matmul_f32(go_h, w_h, m, n, k)?
                };
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![m, k],
                    false,
                )?)
            } else {
                None
            };

            let grad_weight = if self.weight.requires_grad() {
                let k = self.input.shape()[1];
                let inp_h = self.input.gpu_handle()?;
                let result_h = if f64_path {
                    let got_h = backend.transpose_2d_f64(go_h, m, n)?;
                    backend.matmul_f64(&got_h, inp_h, n, m, k)?
                } else {
                    let got_h = backend.transpose_2d_f32(go_h, m, n)?;
                    backend.matmul_f32(&got_h, inp_h, n, m, k)?
                };
                Some(Tensor::from_storage(
                    TensorStorage::gpu(result_h),
                    vec![n, k],
                    false,
                )?)
            } else {
                None
            };

            let grad_bias = if self.has_bias {
                if let Some(ref b) = self.bias {
                    if b.requires_grad() {
                        let go_shape = &[m, n];
                        let summed = if f64_path {
                            backend.sum_axis_f64(go_h, go_shape, 0)?
                        } else {
                            backend.sum_axis_f32(go_h, go_shape, 0)?
                        };
                        Some(Tensor::from_storage(
                            TensorStorage::gpu(summed),
                            vec![n],
                            false,
                        )?)
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            let mut grads = vec![grad_input, grad_weight];
            if self.bias.is_some() {
                grads.push(grad_bias);
            }
            return Ok(grads);
        }

        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "LinearFusedBackward",
            });
        }

        // CPU path.
        let gc_data = grad_output.data()?;

        let grad_input = if self.input.requires_grad() {
            let w_data = self.weight.data()?;
            let k = self.weight.shape()[1];
            let result = crate::ops::linalg::mm_raw(gc_data, w_data, m, n, k);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![m, k],
                false,
            )?)
        } else {
            None
        };

        let grad_weight = if self.weight.requires_grad() {
            let a_data = self.input.data()?;
            let k = self.input.shape()[1];
            let result = crate::ops::linalg::mm_raw_at(gc_data, a_data, n, m, k);
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![n, k],
                false,
            )?)
        } else {
            None
        };

        let grad_bias = if self.has_bias {
            if let Some(ref b) = self.bias {
                if b.requires_grad() {
                    let zero = <T as num_traits::Zero>::zero();
                    let mut gb = vec![zero; n];
                    for i in 0..m {
                        let row = i * n;
                        for j in 0..n {
                            gb[j] += gc_data[row + j];
                        }
                    }
                    Some(Tensor::from_storage(
                        TensorStorage::cpu(gb),
                        vec![n],
                        false,
                    )?)
                } else {
                    None
                }
            } else {
                None
            }
        } else {
            None
        };

        // Return exactly as many gradients as inputs() returns.
        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "LinearFusedBackward"
    }
}

/// Fused differentiable linear: output = input @ weight^T + bias.
/// Creates a single tensor (instead of 3) with a combined backward.
pub fn linear_fused<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
) -> FerrotorchResult<Tensor<T>> {
    let m = input.shape()[0];
    let k = input.shape()[1];
    let n = weight.shape()[0];

    // GPU path: transpose weight, matmul, broadcast_add bias.
    if input.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        // C = input @ weight^T
        let wt_handle = backend.transpose_2d_f32(weight.gpu_handle()?, n, k)?;
        // When autocast says ReducedPrecision and inputs are f32 on GPU,
        // use the f16-accumulate path (falls back to f32 if no kernel).
        let use_f16 =
            is_f32::<T>() && autocast_guard("linear") == Some(AutocastCategory::ReducedPrecision);
        let mut result_handle = if use_f16 {
            backend.matmul_f16_f32(input.gpu_handle()?, &wt_handle, m, k, n)?
        } else {
            backend.matmul_f32(input.gpu_handle()?, &wt_handle, m, k, n)?
        };
        // Add bias if present
        if let Some(b) = bias {
            let out_shape = vec![m, n];
            let b_shape = vec![n];
            result_handle = backend.broadcast_add_f32(
                &result_handle,
                b.gpu_handle()?,
                &out_shape,
                &b_shape,
                &out_shape,
            )?;
        }
        let storage = TensorStorage::gpu(result_handle);
        let shape = vec![m, n];

        let needs_grad = is_grad_enabled()
            && (input.requires_grad()
                || weight.requires_grad()
                || bias.is_some_and(|b| b.requires_grad()));

        return if needs_grad {
            let grad_fn = Arc::new(LinearFusedBackward {
                input: input.clone(),
                weight: weight.clone(),
                has_bias: bias.is_some(),
                bias: bias.cloned(),
            });
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        };
    }

    let a_data = input.data()?;
    let w_data = weight.data()?;
    let mut result_vec = linalg::mm_raw_bt(a_data, w_data, m, k, n);

    // Fuse bias addition
    if let Some(b) = bias {
        let b_data = b.data()?;
        for i in 0..m {
            let row = i * n;
            for j in 0..n {
                result_vec[row + j] += b_data[j];
            }
        }
    }

    let storage = TensorStorage::cpu(result_vec);
    let shape = vec![m, n];

    let needs_grad = is_grad_enabled()
        && (input.requires_grad()
            || weight.requires_grad()
            || bias.is_some_and(|b| b.requires_grad()));

    if needs_grad {
        let grad_fn = Arc::new(LinearFusedBackward {
            input: input.clone(),
            weight: weight.clone(),
            has_bias: bias.is_some(),
            bias: bias.cloned(),
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

/// Differentiable matrix-vector multiply. Attaches `MvBackward` when needed.
pub fn mv_differentiable<T: Float>(a: &Tensor<T>, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let needs_grad = is_grad_enabled() && (a.requires_grad() || x.requires_grad());

    // Compute mv directly from slices to avoid double-copy.
    let a_data = a.data()?;
    let x_data = x.data()?;
    let m = a.shape()[0];
    let k = a.shape()[1];
    let zero = <T as num_traits::Zero>::zero();

    let mut result_vec = vec![zero; m];
    for (i, result_elem) in result_vec.iter_mut().enumerate() {
        let mut acc = zero;
        let row = i * k;
        for p in 0..k {
            acc += a_data[row + p] * x_data[p];
        }
        *result_elem = acc;
    }

    let storage = TensorStorage::cpu(result_vec);
    let shape = vec![m];

    if needs_grad {
        let grad_fn = Arc::new(MvBackward::new(a.clone(), x.clone()));
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

/// Differentiable dot product. Attaches `DotBackward` when needed.
pub fn dot_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let needs_grad = is_grad_enabled() && (a.requires_grad() || b.requires_grad());

    let a_data = a.data()?;
    let b_data = b.data()?;
    let result_val = a_data
        .iter()
        .zip(b_data.iter())
        .fold(<T as num_traits::Zero>::zero(), |acc, (&x, &y)| acc + x * y);

    let storage = TensorStorage::cpu(vec![result_val]);
    let shape = vec![];

    if needs_grad {
        let grad_fn = Arc::new(DotBackward::new(a.clone(), b.clone()));
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

/// Differentiable batched matmul with `BmmBackward`.
///
/// Uses the GPU-aware `bmm()` for the forward pass (dispatches to cuBLAS on
/// GPU, CPU loops otherwise), then attaches `BmmBackward` for autograd.
pub fn bmm_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    // Record autocast decision. Actual f16 dispatch for bmm will be added
    // when the batched f16 GEMM kernel lands; for now the guard ensures the
    // policy is tracked.
    let _autocast_cat = autocast_guard("bmm");
    let result = bmm(a, b)?;

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(BmmBackward::new(a.clone(), b.clone()));
        let (storage, shape) = result.into_storage_and_shape()?;
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Ok(result)
    }
}

/// Differentiable general matmul dispatcher. Attaches `MatmulBackward`
/// when needed. Supports all rank combinations including batched broadcast
/// matmul for ≥3D tensors.
pub fn matmul_differentiable<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    // Materialize non-contiguous views before linalg ops.
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

    if a.is_cuda() && a.ndim() == 2 && b.ndim() == 2 {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        // When autocast says ReducedPrecision and inputs are f32 on GPU,
        // use the f16-accumulate path (falls back to f32 if no kernel).
        let handle = if is_f32::<T>()
            && autocast_guard("matmul") == Some(AutocastCategory::ReducedPrecision)
        {
            backend.matmul_f16_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)?
        } else {
            backend.matmul_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = vec![m, n];

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MatmulBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        // Dispatch to specialized paths that avoid double-copy.
        match (a.ndim(), b.ndim()) {
            (1, 1) => return dot_differentiable(&a, &b),
            (2, 1) => return mv_differentiable(&a, &b),
            (2, 2) => return mm_differentiable(&a, &b),
            (3, 3) if a.shape()[0] == b.shape()[0] => return bmm_differentiable(&a, &b),
            _ => {}
        }

        // Fallback for other shapes — still goes through linalg::matmul.
        let result = linalg::matmul(&a, &b)?;

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MatmulBackward::new(a.clone(), b.clone()));
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// bmm (batched matmul) — GPU-accelerated via strided batch SGEMM
// ===========================================================================

/// Batched matrix multiply: `C[i] = A[i] @ B[i]` for `i` in `0..batch`.
///
/// `a` shape: `[batch, m, k]`, `b` shape: `[batch, k, n]`.
/// Returns `[batch, m, n]`.
///
/// On GPU, dispatches to cuBLAS `SgemmStridedBatched`. On CPU, falls back
/// to per-batch `mm`.
pub fn bmm<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.ndim() != 3 || b.ndim() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "bmm requires 3-D tensors, got {:?} and {:?}",
                a.shape(),
                b.shape()
            ),
        });
    }
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    // Materialize non-contiguous views (e.g. from permute/transpose).
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

    let batch = a.shape()[0];
    let m = a.shape()[1];
    let k = a.shape()[2];
    let n = b.shape()[2];

    if b.shape()[0] != batch || b.shape()[1] != k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!("bmm: a is [{batch},{m},{k}], b is {:?}", b.shape()),
        });
    }

    let out_shape = vec![batch, m, n];

    // GPU path.
    if a.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            // Use f16 Tensor Core path when autocast selects ReducedPrecision.
            let handle = if is_f32::<T>()
                && autocast_guard("bmm") == Some(AutocastCategory::ReducedPrecision)
            {
                backend.bmm_f16_f32(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
            } else {
                backend.bmm_f32(a.gpu_handle()?, b.gpu_handle()?, batch, m, k, n)?
            };
            return Tensor::from_storage(TensorStorage::gpu(handle), out_shape, false);
        }
    }

    // CPU path: loop over batch.
    let a_data = a.data()?;
    let b_data = b.data()?;
    let mut out = Vec::with_capacity(batch * m * n);

    for bi in 0..batch {
        let a_off = bi * m * k;
        let b_off = bi * k * n;
        for i in 0..m {
            for j in 0..n {
                let mut sum = <T as num_traits::Zero>::zero();
                for p in 0..k {
                    sum += a_data[a_off + i * k + p] * b_data[b_off + p * n + j];
                }
                out.push(sum);
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
}

// ===========================================================================
// permute_0213 — swap dims 1 and 2 of a 4D tensor
// ===========================================================================

/// Permute a 4-D tensor from `[d0, d1, d2, d3]` to `[d0, d2, d1, d3]`.
///
/// Primary use: reshape attention heads `[B, S, H, D_h]` → `[B, H, S, D_h]`.
/// On GPU, dispatches to a native PTX kernel. On CPU, does direct index mapping.
pub fn permute_0213<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("permute_0213 requires 4-D tensor, got {:?}", input.shape()),
        });
    }

    let d0 = input.shape()[0];
    let d1 = input.shape()[1];
    let d2 = input.shape()[2];
    let d3 = input.shape()[3];
    let out_shape = vec![d0, d2, d1, d3];

    // GPU path.
    if input.is_cuda() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let handle = backend.permute_0213_f32(input.gpu_handle()?, d0, d1, d2, d3)?;
            return Tensor::from_storage(TensorStorage::gpu(handle), out_shape, false);
        }
    }

    // CPU path.
    let data = input.data()?;
    let total = d0 * d1 * d2 * d3;
    let mut out = vec![<T as num_traits::Zero>::zero(); total];

    for i0 in 0..d0 {
        for i1 in 0..d1 {
            for i2 in 0..d2 {
                for i3 in 0..d3 {
                    let in_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
                    let out_idx = ((i0 * d2 + i2) * d1 + i1) * d3 + i3;
                    out[out_idx] = data[in_idx];
                }
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), out_shape, false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    /// Helper: create a leaf tensor with requires_grad.
    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    /// Helper: create a leaf tensor without requires_grad.
    fn no_grad_leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    /// Assert two slices are element-wise close.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
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

    // -----------------------------------------------------------------------
    // mm backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_mm_backward_both_grads() {
        // A = [[1, 2], [3, 4]]  (2x2)
        // B = [[5, 6], [7, 8]]  (2x2)
        // C = A @ B = [[19, 22], [43, 50]]
        //
        // To get a scalar loss: L = sum(C) = 19 + 22 + 43 + 50 = 134
        // dL/dC = [[1, 1], [1, 1]]
        //
        // dL/dA = dL/dC @ B^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        // dL/dB = A^T @ dL/dC = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = leaf(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);

        let c = mm_differentiable(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);

        // Sum C to get a scalar for backward.
        let c_data = c.data().unwrap();
        let loss_val: f32 = c_data.iter().sum();

        // Build a SumBackward manually: dL/dC = ones_like(C).
        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackward { input: c }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[2, 2]);
        assert_eq!(b_grad.shape(), &[2, 2]);

        // dL/dA = [[11, 15], [11, 15]]
        assert_close(a_grad.data().unwrap(), &[11.0, 15.0, 11.0, 15.0], 1e-5);
        // dL/dB = [[4, 4], [6, 6]]
        assert_close(b_grad.data().unwrap(), &[4.0, 4.0, 6.0, 6.0], 1e-5);
    }

    #[test]
    fn test_mm_backward_one_requires_grad() {
        // Only A requires grad, B does not.
        let a = leaf(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity
        let b = no_grad_leaf(&[2.0, 3.0, 4.0, 5.0], &[2, 2]);

        let c = mm_differentiable(&a, &b).unwrap();
        assert!(c.grad_fn().is_some());

        // grad_output = ones(2,2)
        let grad_out = no_grad_leaf(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let grads = c.grad_fn().unwrap().backward(&grad_out).unwrap();

        // grad_a should be Some, grad_b should be None
        assert!(grads[0].is_some());
        assert!(grads[1].is_none());

        // dA = grad_C @ B^T = [[1,1],[1,1]] @ [[2,4],[3,5]] = [[5,9],[5,9]]
        let ga = grads[0].as_ref().unwrap();
        assert_close(ga.data().unwrap(), &[5.0, 9.0, 5.0, 9.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // dot backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_dot_backward() {
        // a = [1, 2, 3], b = [4, 5, 6]
        // s = dot(a, b) = 4 + 10 + 18 = 32
        // ds/da = b = [4, 5, 6]
        // ds/db = a = [1, 2, 3]
        let a = leaf(&[1.0, 2.0, 3.0], &[3]);
        let b = leaf(&[4.0, 5.0, 6.0], &[3]);

        let s = dot_differentiable(&a, &b).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 32.0).abs() < 1e-5);

        s.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[3]);
        assert_eq!(b_grad.shape(), &[3]);
        assert_close(a_grad.data().unwrap(), &[4.0, 5.0, 6.0], 1e-5);
        assert_close(b_grad.data().unwrap(), &[1.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_dot_backward_one_requires_grad() {
        let a = leaf(&[2.0, 3.0], &[2]);
        let b = no_grad_leaf(&[4.0, 5.0], &[2]);

        let s = dot_differentiable(&a, &b).unwrap();
        let grad_out = no_grad_leaf(&[1.0], &[]);
        let grads = s.grad_fn().unwrap().backward(&grad_out).unwrap();

        assert!(grads[0].is_some());
        assert!(grads[1].is_none());
        assert_close(
            grads[0].as_ref().unwrap().data().unwrap(),
            &[4.0, 5.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // mv backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_mv_backward() {
        // A = [[1, 2], [3, 4]]  (2x2)
        // x = [5, 6]            (2,)
        // y = A @ x = [17, 39]
        //
        // Use L = sum(y) = 56, so dL/dy = [1, 1].
        // dA = outer([1,1], [5,6]) = [[5,6],[5,6]]
        // dx = A^T @ [1,1] = [[1,3],[2,4]] @ [1,1] = [4, 6]
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let x = leaf(&[5.0, 6.0], &[2]);

        let y = mv_differentiable(&a, &x).unwrap();
        assert_eq!(y.shape(), &[2]);

        // Build sum for scalar loss.
        let y_data = y.data().unwrap();
        let loss_val: f32 = y_data.iter().sum();

        #[derive(Debug)]
        struct SumBackward1D<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward1D<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackward1D { input: y }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let x_grad = x.grad().unwrap().expect("x should have grad");

        assert_eq!(a_grad.shape(), &[2, 2]);
        assert_eq!(x_grad.shape(), &[2]);

        // dA = outer([1,1], [5,6]) = [[5,6],[5,6]]
        assert_close(a_grad.data().unwrap(), &[5.0, 6.0, 5.0, 6.0], 1e-5);
        // dx = A^T @ [1,1] = [4, 6]
        assert_close(x_grad.data().unwrap(), &[4.0, 6.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // matmul backward (dispatch)
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_backward_dispatches_to_dot() {
        // matmul(1D, 1D) should use DotBackward path.
        let a = leaf(&[1.0, 2.0], &[2]);
        let b = leaf(&[3.0, 4.0], &[2]);

        let s = matmul_differentiable(&a, &b).unwrap();
        assert!(s.is_scalar());
        assert!((s.item().unwrap() - 11.0).abs() < 1e-5);

        s.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let b_grad = b.grad().unwrap().unwrap();
        assert_close(a_grad.data().unwrap(), &[3.0, 4.0], 1e-5);
        assert_close(b_grad.data().unwrap(), &[1.0, 2.0], 1e-5);
    }

    #[test]
    fn test_matmul_backward_dispatches_to_mm() {
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = leaf(&[1.0, 0.0, 0.0, 1.0], &[2, 2]); // identity

        let c = matmul_differentiable(&a, &b).unwrap();

        // grad_output = ones
        let grad_out = no_grad_leaf(&[1.0, 1.0, 1.0, 1.0], &[2, 2]);
        let grads = c.grad_fn().unwrap().backward(&grad_out).unwrap();

        // dA = ones @ I^T = ones
        assert_close(
            grads[0].as_ref().unwrap().data().unwrap(),
            &[1.0, 1.0, 1.0, 1.0],
            1e-5,
        );
        // dB = A^T @ ones = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        assert_close(
            grads[1].as_ref().unwrap().data().unwrap(),
            &[4.0, 4.0, 6.0, 6.0],
            1e-5,
        );
    }

    #[test]
    fn test_matmul_backward_vm() {
        // a = [1, 2] (K=2), B = [[3, 4, 5], [6, 7, 8]] (2x3)
        // y = a @ B = [1*3+2*6, 1*4+2*7, 1*5+2*8] = [15, 18, 21]
        //
        // dL/dy = [1, 1, 1]  (from sum)
        // da = B @ dL/dy = [[3,4,5],[6,7,8]] @ [1,1,1] = [12, 21]
        // dB = outer(a, dL/dy) = [[1,1,1],[2,2,2]]
        let a = leaf(&[1.0, 2.0], &[2]);
        let b = leaf(&[3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 3]);

        let y = matmul_differentiable(&a, &b).unwrap();
        assert_eq!(y.shape(), &[3]);

        // Build sum for scalar.
        let y_data = y.data().unwrap();
        let loss_val: f32 = y_data.iter().sum();

        #[derive(Debug)]
        struct SumBackwardVec<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackwardVec<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackwardVec { input: y }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[2]);
        assert_eq!(b_grad.shape(), &[2, 3]);

        // da = B @ [1,1,1] = [12, 21]
        assert_close(a_grad.data().unwrap(), &[12.0, 21.0], 1e-5);
        // dB = outer([1,2], [1,1,1]) = [[1,1,1],[2,2,2]]
        assert_close(
            b_grad.data().unwrap(),
            &[1.0, 1.0, 1.0, 2.0, 2.0, 2.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // bmm backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_bmm_backward_both_grads() {
        // Batch 0: A0=[[1,2],[3,4]], B0=[[5,6],[7,8]]
        //   C0 = [[19,22],[43,50]]
        // Batch 1: A1=[[1,0],[0,1]] (identity), B1=[[9,10],[11,12]]
        //   C1 = [[9,10],[11,12]]
        //
        // L = sum(C), dL/dC = ones(2,2,2)
        //
        // dA0 = ones(2,2) @ B0^T = [[1,1],[1,1]] @ [[5,7],[6,8]] = [[11,15],[11,15]]
        // dA1 = ones(2,2) @ B1^T = [[1,1],[1,1]] @ [[9,11],[10,12]] = [[19,23],[19,23]]
        //
        // dB0 = A0^T @ ones(2,2) = [[1,3],[2,4]] @ [[1,1],[1,1]] = [[4,4],[6,6]]
        // dB1 = A1^T @ ones(2,2) = [[1,0],[0,1]] @ [[1,1],[1,1]] = [[1,1],[1,1]]
        #[rustfmt::skip]
        let a = leaf(&[
            1.0, 2.0, 3.0, 4.0,   // batch 0
            1.0, 0.0, 0.0, 1.0,   // batch 1
        ], &[2, 2, 2]);
        #[rustfmt::skip]
        let b = leaf(&[
            5.0, 6.0, 7.0, 8.0,    // batch 0
            9.0, 10.0, 11.0, 12.0, // batch 1
        ], &[2, 2, 2]);

        let c = bmm_differentiable(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        // Sum to scalar for backward.
        let c_data = c.data().unwrap();
        let loss_val: f32 = c_data.iter().sum();

        #[derive(Debug)]
        struct SumBackward3D<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward3D<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let g = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(g)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![loss_val]),
            vec![],
            Arc::new(SumBackward3D { input: c }),
        )
        .unwrap();

        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().expect("a should have grad");
        let b_grad = b.grad().unwrap().expect("b should have grad");

        assert_eq!(a_grad.shape(), &[2, 2, 2]);
        assert_eq!(b_grad.shape(), &[2, 2, 2]);

        #[rustfmt::skip]
        let expected_da: &[f32] = &[
            11.0, 15.0, 11.0, 15.0,  // batch 0
            19.0, 23.0, 19.0, 23.0,  // batch 1
        ];
        #[rustfmt::skip]
        let expected_db: &[f32] = &[
            4.0, 4.0, 6.0, 6.0,  // batch 0
            1.0, 1.0, 1.0, 1.0,  // batch 1
        ];
        assert_close(a_grad.data().unwrap(), expected_da, 1e-5);
        assert_close(b_grad.data().unwrap(), expected_db, 1e-5);
    }

    #[test]
    fn test_bmm_backward_batch_size_1() {
        // Single batch: should match mm backward exactly.
        // A=[[1,2],[3,4]], B=[[5,6],[7,8]]
        // dL/dC = ones(1,2,2)
        // dA = ones @ B^T = [[11,15],[11,15]]
        // dB = A^T @ ones = [[4,4],[6,6]]
        let a = leaf(&[1.0, 2.0, 3.0, 4.0], &[1, 2, 2]);
        let b = leaf(&[5.0, 6.0, 7.0, 8.0], &[1, 2, 2]);

        let c = bmm_differentiable(&a, &b).unwrap();

        let grad_out = no_grad_leaf(&[1.0, 1.0, 1.0, 1.0], &[1, 2, 2]);
        let grads = c.grad_fn().unwrap().backward(&grad_out).unwrap();

        assert!(grads[0].is_some());
        assert!(grads[1].is_some());

        let ga = grads[0].as_ref().unwrap();
        let gb = grads[1].as_ref().unwrap();
        assert_eq!(ga.shape(), &[1, 2, 2]);
        assert_eq!(gb.shape(), &[1, 2, 2]);

        assert_close(ga.data().unwrap(), &[11.0, 15.0, 11.0, 15.0], 1e-5);
        assert_close(gb.data().unwrap(), &[4.0, 4.0, 6.0, 6.0], 1e-5);
    }

    #[test]
    fn test_bmm_backward_one_requires_grad() {
        // Only A requires grad.
        let a = leaf(&[1.0, 0.0, 0.0, 1.0], &[1, 2, 2]);
        let b = no_grad_leaf(&[2.0, 3.0, 4.0, 5.0], &[1, 2, 2]);

        let c = bmm_differentiable(&a, &b).unwrap();
        assert!(c.grad_fn().is_some());

        let grad_out = no_grad_leaf(&[1.0, 1.0, 1.0, 1.0], &[1, 2, 2]);
        let grads = c.grad_fn().unwrap().backward(&grad_out).unwrap();

        assert!(grads[0].is_some());
        assert!(grads[1].is_none());

        // dA = ones @ B^T = [[1,1],[1,1]] @ [[2,4],[3,5]] = [[5,9],[5,9]]
        let ga = grads[0].as_ref().unwrap();
        assert_close(ga.data().unwrap(), &[5.0, 9.0, 5.0, 9.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // no_grad disables backward tracking
    // -----------------------------------------------------------------------

    #[test]
    fn test_no_grad_skips_backward() {
        let a = leaf(&[1.0, 2.0, 3.0], &[3]);
        let b = leaf(&[4.0, 5.0, 6.0], &[3]);

        let s = crate::autograd::no_grad::no_grad(|| dot_differentiable(&a, &b).unwrap());

        // Should have no grad_fn because we were inside no_grad.
        assert!(s.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // broadcast matmul backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_backward_3d_3d_numerical() {
        // Numerical gradient check for (2,2,3) @ (2,3,2).
        let eps = 1e-3f32;

        let a_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 0.1).collect();
        let b_data: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 + 0.5).collect();

        // Forward + backward.
        let a = leaf(&a_data, &[2, 2, 3]);
        let b = leaf(&b_data, &[2, 3, 2]);
        let c = matmul_differentiable(&a, &b).unwrap();
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();

        let analytic_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let analytic_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

        // Check each element of A numerically.
        for idx in 0..a_data.len() {
            let mut a_plus = a_data.clone();
            a_plus[idx] += eps;
            let mut a_minus = a_data.clone();
            a_minus[idx] -= eps;

            let loss_plus = crate::autograd::no_grad::no_grad(|| {
                let ap = no_grad_leaf(&a_plus, &[2, 2, 3]);
                let bp = no_grad_leaf(&b_data, &[2, 3, 2]);
                let c = linalg::matmul(&ap, &bp).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });
            let loss_minus = crate::autograd::no_grad::no_grad(|| {
                let am = no_grad_leaf(&a_minus, &[2, 2, 3]);
                let bm = no_grad_leaf(&b_data, &[2, 3, 2]);
                let c = linalg::matmul(&am, &bm).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic_a[idx]).abs() < 5e-2,
                "grad_a[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic_a[idx],
                (numerical - analytic_a[idx]).abs()
            );
        }

        // Check each element of B numerically.
        for idx in 0..b_data.len() {
            let mut b_plus = b_data.clone();
            b_plus[idx] += eps;
            let mut b_minus = b_data.clone();
            b_minus[idx] -= eps;

            let loss_plus = crate::autograd::no_grad::no_grad(|| {
                let ap = no_grad_leaf(&a_data, &[2, 2, 3]);
                let bp = no_grad_leaf(&b_plus, &[2, 3, 2]);
                let c = linalg::matmul(&ap, &bp).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });
            let loss_minus = crate::autograd::no_grad::no_grad(|| {
                let am = no_grad_leaf(&a_data, &[2, 2, 3]);
                let bm = no_grad_leaf(&b_minus, &[2, 3, 2]);
                let c = linalg::matmul(&am, &bm).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic_b[idx]).abs() < 5e-2,
                "grad_b[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic_b[idx],
                (numerical - analytic_b[idx]).abs()
            );
        }
    }

    #[test]
    fn test_matmul_backward_3d_2d_broadcast_numerical() {
        // (2,3,4) @ (4,2) — B broadcasts over batch dim.
        // Gradient for B must sum over the batch dimension.
        let eps = 1e-4f32;

        let a_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.05 + 0.1).collect();
        let b_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1 + 0.2).collect();

        let a = leaf(&a_data, &[2, 3, 4]);
        let b = leaf(&b_data, &[4, 2]);
        let c = matmul_differentiable(&a, &b).unwrap();
        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();

        let analytic_a = a.grad().unwrap().unwrap().data().unwrap().to_vec();
        let analytic_b = b.grad().unwrap().unwrap().data().unwrap().to_vec();

        // Grad shapes should match input shapes.
        assert_eq!(a.grad().unwrap().unwrap().shape(), &[2, 3, 4]);
        assert_eq!(b.grad().unwrap().unwrap().shape(), &[4, 2]);

        // Numerical check for B (the broadcast operand — most important).
        for idx in 0..b_data.len() {
            let mut b_plus = b_data.clone();
            b_plus[idx] += eps;
            let mut b_minus = b_data.clone();
            b_minus[idx] -= eps;

            let loss_plus = crate::autograd::no_grad::no_grad(|| {
                let ap = no_grad_leaf(&a_data, &[2, 3, 4]);
                let bp = no_grad_leaf(&b_plus, &[4, 2]);
                let c = linalg::matmul(&ap, &bp).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });
            let loss_minus = crate::autograd::no_grad::no_grad(|| {
                let am = no_grad_leaf(&a_data, &[2, 3, 4]);
                let bm = no_grad_leaf(&b_minus, &[4, 2]);
                let c = linalg::matmul(&am, &bm).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic_b[idx]).abs() < 1e-2,
                "grad_b[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic_b[idx],
                (numerical - analytic_b[idx]).abs()
            );
        }

        // Spot-check A gradient too.
        for idx in 0..4 {
            let mut a_plus = a_data.clone();
            a_plus[idx] += eps;
            let mut a_minus = a_data.clone();
            a_minus[idx] -= eps;

            let loss_plus = crate::autograd::no_grad::no_grad(|| {
                let ap = no_grad_leaf(&a_plus, &[2, 3, 4]);
                let bp = no_grad_leaf(&b_data, &[4, 2]);
                let c = linalg::matmul(&ap, &bp).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });
            let loss_minus = crate::autograd::no_grad::no_grad(|| {
                let am = no_grad_leaf(&a_minus, &[2, 3, 4]);
                let bm = no_grad_leaf(&b_data, &[4, 2]);
                let c = linalg::matmul(&am, &bm).unwrap();
                crate::grad_fns::reduction::sum(&c).unwrap().item().unwrap()
            });

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic_a[idx]).abs() < 1e-2,
                "grad_a[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic_a[idx],
                (numerical - analytic_a[idx]).abs()
            );
        }
    }

    #[test]
    fn test_matmul_backward_batch_broadcast_1_vs_n() {
        // (1,2,3) @ (2,3,2) — batch dim 1 broadcasts to 2.
        // grad_a must sum over the broadcast batch dimension.
        let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b_data: Vec<f32> = (0..12).map(|i| (i as f32) + 1.0).collect();

        let a = leaf(&a_data, &[1, 2, 3]);
        let b = leaf(&b_data, &[2, 3, 2]);
        let c = matmul_differentiable(&a, &b).unwrap();
        assert_eq!(c.shape(), &[2, 2, 2]);

        let loss = crate::grad_fns::reduction::sum(&c).unwrap();
        loss.backward().unwrap();

        // Grad shapes must match original shapes, not broadcast shapes.
        assert_eq!(a.grad().unwrap().unwrap().shape(), &[1, 2, 3]);
        assert_eq!(b.grad().unwrap().unwrap().shape(), &[2, 3, 2]);
    }
}
