//! Backward functions for linear algebra operations.
//!
//! Each struct captures the forward inputs and implements `GradFn` to
//! compute VJPs (vector-Jacobian products) for reverse-mode autodiff.

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::ops::linalg::{self, mm, transpose};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

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
        let grad_a = if self.a.requires_grad() {
            // dA = grad_C @ B^T
            let bt = transpose(&self.b)?;
            Some(mm(grad_output, &bt)?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            // dB = A^T @ grad_C
            let at = transpose(&self.a)?;
            Some(mm(&at, grad_output)?)
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
            // A is (M, K), A^T is (K, M), grad_y is (M,) -> result is (K,)
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
        let s = grad_output.item()?;

        let grad_a = if self.a.requires_grad() {
            // da = grad_s * b
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
            // db = grad_s * a
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
        let batch = self.a.shape()[0];
        let m = self.a.shape()[1];
        let k = self.a.shape()[2];
        let n = self.b.shape()[2];

        let grad_a = if self.a.requires_grad() {
            // dA[b] = grad_C[b] @ B[b]^T   — shape per batch: (M,N) @ (N,K) -> (M,K)
            let grad_data = grad_output.data()?;
            let b_data = self.b.data()?;
            let mut result = vec![<T as num_traits::Zero>::zero(); batch * m * k];

            for bi in 0..batch {
                let g_off = bi * m * n;
                let b_off = bi * k * n;
                let r_off = bi * m * k;
                for i in 0..m {
                    for j in 0..k {
                        let mut acc = <T as num_traits::Zero>::zero();
                        for p in 0..n {
                            // grad[b,i,p] * B[b,j,p]  (B transposed: B^T[p,j] = B[j,p])
                            acc = acc + grad_data[g_off + i * n + p] * b_data[b_off + j * n + p];
                        }
                        result[r_off + i * k + j] = acc;
                    }
                }
            }
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![batch, m, k],
                false,
            )?)
        } else {
            None
        };

        let grad_b = if self.b.requires_grad() {
            // dB[b] = A[b]^T @ grad_C[b]   — shape per batch: (K,M) @ (M,N) -> (K,N)
            let a_data = self.a.data()?;
            let grad_data = grad_output.data()?;
            let mut result = vec![<T as num_traits::Zero>::zero(); batch * k * n];

            for bi in 0..batch {
                let a_off = bi * m * k;
                let g_off = bi * m * n;
                let r_off = bi * k * n;
                for i in 0..k {
                    for j in 0..n {
                        let mut acc = <T as num_traits::Zero>::zero();
                        for p in 0..m {
                            // A^T[b,i,p] * grad[b,p,j]  (A^T[i,p] = A[p,i])
                            acc = acc + a_data[a_off + p * k + i] * grad_data[g_off + p * n + j];
                        }
                        result[r_off + i * n + j] = acc;
                    }
                }
            }
            Some(Tensor::from_storage(
                TensorStorage::cpu(result),
                vec![batch, k, n],
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
                // da = B @ grad_y  (K,N) @ (N,) -> (K,)
                // dB = outer(a, grad_y): dB[k,n] = a[k] * grad_y[n]
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
            _ => Err(crate::error::FerrotorchError::InvalidArgument {
                message: format!(
                    "MatmulBackward: unsupported shapes {:?} and {:?}",
                    self.a.shape(),
                    self.b.shape()
                ),
            }),
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MatmulBackward"
    }
}

// ---------------------------------------------------------------------------
// Differentiable forward wrappers
// ---------------------------------------------------------------------------

/// Differentiable matrix-matrix multiply. If either input requires grad and
/// grad is enabled, attaches `MmBackward`.
pub fn mm_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch { expected: a.device(), got: b.device() });
    }

    if a.is_cuda() {
        let backend = crate::gpu_dispatch::gpu_backend()
            .ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        let handle = backend.matmul_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)?;
        let storage = TensorStorage::gpu(handle);
        let shape = vec![m, n];

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MmBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = mm(a, b)?;

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MmBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }
}

/// Differentiable matrix-vector multiply. Attaches `MvBackward` when needed.
pub fn mv_differentiable<T: Float>(a: &Tensor<T>, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = linalg::mv(a, x)?;

    if is_grad_enabled() && (a.requires_grad() || x.requires_grad()) {
        let grad_fn = Arc::new(MvBackward::new(a.clone(), x.clone()));
        Tensor::from_operation(
            TensorStorage::cpu(result.data()?.to_vec()),
            result.shape().to_vec(),
            grad_fn,
        )
    } else {
        Ok(result)
    }
}

/// Differentiable dot product. Attaches `DotBackward` when needed.
pub fn dot_differentiable<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let result = linalg::dot(a, b)?;

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(DotBackward::new(a.clone(), b.clone()));
        Tensor::from_operation(
            TensorStorage::cpu(result.data()?.to_vec()),
            result.shape().to_vec(),
            grad_fn,
        )
    } else {
        Ok(result)
    }
}

/// Differentiable batched matmul with `BmmBackward`.
pub fn bmm_differentiable<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let result = linalg::bmm(a, b)?;

    if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
        let grad_fn = Arc::new(BmmBackward::new(a.clone(), b.clone()));
        Tensor::from_operation(
            TensorStorage::cpu(result.data()?.to_vec()),
            result.shape().to_vec(),
            grad_fn,
        )
    } else {
        Ok(result)
    }
}

/// Differentiable general matmul dispatcher. Attaches `MatmulBackward`
/// when needed.
pub fn matmul_differentiable<T: Float>(
    a: &Tensor<T>,
    b: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch { expected: a.device(), got: b.device() });
    }

    if a.is_cuda() && a.ndim() == 2 && b.ndim() == 2 {
        let backend = crate::gpu_dispatch::gpu_backend()
            .ok_or(FerrotorchError::DeviceUnavailable)?;
        let m = a.shape()[0];
        let k = a.shape()[1];
        let n = b.shape()[1];
        let handle = backend.matmul_f32(a.gpu_handle()?, b.gpu_handle()?, m, k, n)?;
        let storage = TensorStorage::gpu(handle);
        let shape = vec![m, n];

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MatmulBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(storage, shape, grad_fn)
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = linalg::matmul(a, b)?;

        if is_grad_enabled() && (a.requires_grad() || b.requires_grad()) {
            let grad_fn = Arc::new(MatmulBackward::new(a.clone(), b.clone()));
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )
        } else {
            Ok(result)
        }
    }
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
            message: format!("bmm requires 3-D tensors, got {:?} and {:?}", a.shape(), b.shape()),
        });
    }
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch { expected: a.device(), got: b.device() });
    }

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
            let handle = backend.bmm_f32(
                a.gpu_handle()?, b.gpu_handle()?,
                batch, m, k, n,
            )?;
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
                    sum = sum + a_data[a_off + i * k + p] * b_data[b_off + p * n + j];
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
            let handle = backend.permute_0213_f32(
                input.gpu_handle()?, d0, d1, d2, d3,
            )?;
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
