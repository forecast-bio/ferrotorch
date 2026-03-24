//! Method-style API for Tensor operations.
//!
//! Enables `a.matmul(&b)`, `a.relu()`, `a.sum()`, `a.reshape(&[2, 3])` etc.
//! All methods delegate to the corresponding grad_fns or ops functions.

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

impl<T: Float> Tensor<T> {
    // --- Arithmetic ---

    pub fn add_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::add(self, other)
    }

    pub fn sub_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::sub(self, other)
    }

    pub fn mul_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::mul(self, other)
    }

    pub fn div_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::div(self, other)
    }

    pub fn neg_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::neg(self)
    }

    pub fn pow_t(&self, exponent: f64) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::pow(self, exponent)
    }

    pub fn sqrt_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::sqrt(self)
    }

    pub fn abs_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::arithmetic::abs(self)
    }

    // --- Transcendental ---

    pub fn exp_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::exp(self)
    }

    pub fn log_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::log(self)
    }

    pub fn sin_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::sin(self)
    }

    pub fn cos_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::cos(self)
    }

    pub fn clamp_t(&self, min: T, max: T) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::transcendental::clamp(self, min, max)
    }

    // --- Activation ---

    pub fn relu(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::relu(self)
    }

    pub fn sigmoid(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::sigmoid(self)
    }

    pub fn tanh_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::tanh(self)
    }

    pub fn gelu(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::gelu(self)
    }

    pub fn gelu_with(
        &self,
        approximate: crate::grad_fns::activation::GeluApproximate,
    ) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::gelu_with(self, approximate)
    }

    pub fn silu(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::silu(self)
    }

    pub fn softmax(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::softmax(self)
    }

    pub fn log_softmax(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::activation::log_softmax(self)
    }

    // --- Reduction ---

    pub fn sum_all(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::sum(self)
    }

    pub fn mean_all(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::mean(self)
    }

    pub fn prod_all(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::prod(self)
    }

    // --- Linalg ---

    pub fn matmul(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::matmul_differentiable(self, other)
    }

    pub fn mm(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::mm_differentiable(self, other)
    }

    /// Fused A @ B^T — avoids materializing the transpose of B.
    /// A: [M, K], B: [N, K] -> [M, N].
    pub fn mm_bt(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::mm_bt_differentiable(self, other)
    }

    pub fn bmm(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::bmm_differentiable(self, other)
    }

    pub fn mv_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::mv_differentiable(self, other)
    }

    pub fn dot_t(&self, other: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::linalg::dot_differentiable(self, other)
    }

    pub fn t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::transpose_2d(self)
    }

    /// Einstein summation with this tensor as the first operand.
    ///
    /// `others` contains the remaining input tensors (if any). The equation
    /// must include subscripts for `self` followed by the `others`.
    ///
    /// ```ignore
    /// // Matrix multiply: self @ other
    /// let c = a.einsum("ij,jk->ik", &[&b])?;
    ///
    /// // Trace of self
    /// let t = a.einsum("ii->", &[])?;
    /// ```
    pub fn einsum(&self, equation: &str, others: &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
        let mut inputs: Vec<&Tensor<T>> = vec![self];
        inputs.extend_from_slice(others);
        crate::einsum::einsum_differentiable(equation, &inputs)
    }

    // --- Reduction (dim) ---

    pub fn sum_dim(&self, dim: i64, keepdim: bool) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::sum_dim(self, dim, keepdim)
    }

    pub fn mean_dim(&self, dim: i64, keepdim: bool) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::reduction::mean_dim(self, dim, keepdim)
    }

    // --- Shape ---

    pub fn reshape_t(&self, shape: &[isize]) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::reshape(self, shape)
    }

    pub fn flatten_t(&self) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::flatten(self)
    }

    pub fn squeeze_t(&self, axis: isize) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::squeeze(self, axis)
    }

    pub fn unsqueeze_t(&self, axis: isize) -> FerrotorchResult<Tensor<T>> {
        crate::grad_fns::shape::unsqueeze(self, axis)
    }

    /// Permute tensor dimensions. Like PyTorch's `tensor.permute(dims)`.
    ///
    /// `dims` must be a valid permutation of `0..ndim`.
    pub fn permute(&self, dims: &[usize]) -> FerrotorchResult<Tensor<T>> {
        permute_t(self, dims)
    }

    /// View tensor with new shape. Like PyTorch's `tensor.view(shape)`.
    ///
    /// Exactly one dimension may be `-1`, in which case it is inferred.
    /// Requires the tensor to be contiguous.
    pub fn view(&self, shape: &[i64]) -> FerrotorchResult<Tensor<T>> {
        view_t(self, shape)
    }

    /// Make tensor contiguous (copy data if needed).
    ///
    /// Currently all tensors are contiguous, so this is always a cheap clone.
    pub fn contiguous(&self) -> FerrotorchResult<Tensor<T>> {
        contiguous_t(self)
    }

    /// Split tensor into `chunks` roughly equal pieces along `dim`.
    pub fn chunk(&self, chunks: usize, dim: usize) -> FerrotorchResult<Vec<Tensor<T>>> {
        chunk_t(self, chunks, dim)
    }

    /// Split tensor into pieces of given sizes along `dim`.
    pub fn split(&self, split_sizes: &[usize], dim: usize) -> FerrotorchResult<Vec<Tensor<T>>> {
        split_t(self, split_sizes, dim)
    }

    // --- PyTorch compatibility aliases ---

    /// Alias for `shape()`. Returns the tensor dimensions like PyTorch's `Tensor.size()`.
    #[inline]
    pub fn size(&self) -> &[usize] {
        self.shape()
    }

    /// Alias for `ndim()`. Returns the number of dimensions like PyTorch's `Tensor.dim()`.
    #[inline]
    pub fn dim(&self) -> usize {
        self.ndim()
    }

    // --- Utility ---

    /// Print the tensor and return self for chaining.
    pub fn print(&self) -> &Self {
        println!("{self}");
        self
    }
}

// ---------------------------------------------------------------------------
// Free functions: permute, view, contiguous, chunk, split
// ---------------------------------------------------------------------------

/// Permute tensor dimensions. Like PyTorch's `tensor.permute(dims)`.
///
/// `dims` must be a valid permutation of `0..ndim`.
pub fn permute_t<T: Float>(input: &Tensor<T>, dims: &[usize]) -> FerrotorchResult<Tensor<T>> {
    use crate::error::FerrotorchError;
    use crate::storage::TensorStorage;

    let ndim = input.ndim();
    if dims.len() != ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "permute: dims length {} does not match tensor ndim {}",
                dims.len(),
                ndim
            ),
        });
    }

    // Validate that dims is a valid permutation.
    let mut seen = vec![false; ndim];
    for &d in dims {
        if d >= ndim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("permute: dim {} is out of bounds for ndim {}", d, ndim),
            });
        }
        if seen[d] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("permute: duplicate dim {} in permutation", d),
            });
        }
        seen[d] = true;
    }

    let in_shape = input.shape();
    let device = input.device();
    let in_data = input.data_vec()?;

    // Compute output shape.
    let out_shape: Vec<usize> = dims.iter().map(|&d| in_shape[d]).collect();
    let out_numel: usize = out_shape.iter().product();
    let mut out_data = vec![<T as num_traits::Zero>::zero(); out_numel];

    // Compute input strides for coordinate decomposition.
    let in_strides = crate::shape::c_contiguous_strides(in_shape);

    for flat in 0..out_numel {
        // Decompose flat index using output shape to get output coords.
        let mut rem = flat;
        let mut out_coords = vec![0usize; ndim];
        for d in (0..ndim).rev() {
            out_coords[d] = rem % out_shape[d];
            rem /= out_shape[d];
        }
        // Map output coords back to input coords via inverse permutation.
        let mut in_flat = 0usize;
        for (out_d, &in_d) in dims.iter().enumerate() {
            in_flat += out_coords[out_d] as usize * in_strides[in_d] as usize;
        }
        out_data[flat] = in_data[in_flat];
    }

    // Permute is differentiable: backward is the inverse permutation.
    let storage = TensorStorage::on_device(out_data, device)?;
    if crate::autograd::no_grad::is_grad_enabled() && input.requires_grad() {
        let grad_fn = std::sync::Arc::new(PermuteBackward {
            input: input.clone(),
            dims: dims.to_vec(),
        });
        Tensor::from_operation(storage, out_shape, grad_fn)
    } else {
        Tensor::from_storage(storage, out_shape, false)
    }
}

/// Backward for permute: apply the inverse permutation to the gradient.
#[derive(Debug)]
struct PermuteBackward<T: Float> {
    input: Tensor<T>,
    dims: Vec<usize>,
}

impl<T: Float> crate::tensor::GradFn<T> for PermuteBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Compute inverse permutation.
        let mut inv_dims = vec![0usize; self.dims.len()];
        for (i, &d) in self.dims.iter().enumerate() {
            inv_dims[d] = i;
        }
        let grad_input = permute_t(grad_output, &inv_dims)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "PermuteBackward"
    }
}

/// View tensor with new shape. Like PyTorch's `tensor.view(shape)`.
///
/// Exactly one dimension may be `-1`, in which case it is inferred.
/// Requires the tensor to be contiguous (currently all tensors are).
pub fn view_t<T: Float>(input: &Tensor<T>, shape: &[i64]) -> FerrotorchResult<Tensor<T>> {
    use crate::error::FerrotorchError;

    if !input.is_contiguous() {
        return Err(FerrotorchError::InvalidArgument {
            message: "view: tensor must be contiguous; call .contiguous() first".into(),
        });
    }

    // Convert i64 shape to isize for reshape (which handles -1 inference).
    let isize_shape: Vec<isize> = shape.iter().map(|&d| d as isize).collect();
    crate::grad_fns::shape::reshape(input, &isize_shape)
}

/// Make tensor contiguous (copy data if needed).
///
/// If the tensor is already contiguous this returns a cheap clone.
/// Otherwise it gathers the data in C-order and creates a new
/// contiguous tensor, preserving the original device.
pub fn contiguous_t<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if input.is_contiguous() {
        return Ok(input.clone());
    }
    let device = input.device();
    let data = input.data_vec()?;
    let storage = TensorStorage::on_device(data, device)?;
    Tensor::from_storage(storage, input.shape().to_vec(), input.requires_grad())
}

/// Split tensor into `chunks` roughly equal pieces along `dim`.
///
/// If the tensor size along `dim` is not evenly divisible by `chunks`,
/// the last chunk will be smaller.
pub fn chunk_t<T: Float>(
    input: &Tensor<T>,
    chunks: usize,
    dim: usize,
) -> FerrotorchResult<Vec<Tensor<T>>> {
    use crate::error::FerrotorchError;

    if chunks == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "chunk: chunks must be > 0".into(),
        });
    }

    let shape = input.shape();
    if dim >= shape.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "chunk: dim {} is out of bounds for tensor with {} dimensions",
                dim,
                shape.len()
            ),
        });
    }

    let dim_size = shape[dim];
    let chunk_size = (dim_size + chunks - 1) / chunks; // ceiling division
    let mut split_sizes = Vec::new();
    let mut remaining = dim_size;
    while remaining > 0 {
        let s = chunk_size.min(remaining);
        split_sizes.push(s);
        remaining -= s;
    }

    split_t(input, &split_sizes, dim)
}

/// Split tensor into pieces of given sizes along `dim`.
///
/// The sum of `split_sizes` must equal the tensor's size along `dim`.
/// When gradient tracking is enabled and the input requires grad, each
/// output chunk is connected to the autograd graph via `SplitBackward`.
pub fn split_t<T: Float>(
    input: &Tensor<T>,
    split_sizes: &[usize],
    dim: usize,
) -> FerrotorchResult<Vec<Tensor<T>>> {
    use std::any::TypeId;
    use std::sync::Arc;
    use crate::autograd::no_grad::is_grad_enabled;
    use crate::error::FerrotorchError;
    use crate::grad_fns::shape::SplitBackward;
    use crate::storage::TensorStorage;

    let shape = input.shape();
    let ndim = shape.len();

    if dim >= ndim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "split: dim {} is out of bounds for tensor with {} dimensions",
                dim, ndim
            ),
        });
    }

    let total: usize = split_sizes.iter().sum();
    if total != shape[dim] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "split: split_sizes sum {} does not match dim {} size {}",
                total, dim, shape[dim]
            ),
        });
    }

    let device = input.device();
    let needs_grad = is_grad_enabled() && input.requires_grad();

    // GPU fast path: use strided_split to extract each chunk directly on GPU.
    if device.is_cuda() && TypeId::of::<T>() == TypeId::of::<f32>() {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let inner: usize = if dim + 1 < ndim {
                shape[dim + 1..].iter().product()
            } else {
                1
            };
            let total_along_dim = shape[dim];
            let in_handle = input.gpu_handle()?;

            let mut results = Vec::with_capacity(split_sizes.len());
            let mut offset_along_dim = 0usize;

            for &split_size in split_sizes {
                let mut chunk_shape = shape.to_vec();
                chunk_shape[dim] = split_size;
                let chunk_numel: usize = chunk_shape.iter().product();

                let chunk_handle = backend.strided_split_f32(
                    in_handle,
                    total_along_dim,
                    offset_along_dim,
                    split_size,
                    inner,
                    chunk_numel,
                )?;

                let storage = TensorStorage::gpu(chunk_handle);
                let t = if needs_grad {
                    let grad_fn = Arc::new(SplitBackward::new(
                        input.clone(),
                        dim,
                        offset_along_dim,
                        split_size,
                    ));
                    Tensor::from_operation(storage, chunk_shape, grad_fn)?
                } else {
                    Tensor::from_storage(storage, chunk_shape, false)?
                };
                results.push(t);
                offset_along_dim += split_size;
            }

            return Ok(results);
        }
    }

    // CPU path (also serves as fallback for non-f32 or missing backend).
    let in_data = input.data_vec()?;

    let outer: usize = shape[..dim].iter().product();
    let inner: usize = if dim + 1 < ndim {
        shape[dim + 1..].iter().product()
    } else {
        1
    };
    let total_along_dim = shape[dim];

    let mut results = Vec::with_capacity(split_sizes.len());
    let mut offset_along_dim = 0usize;

    for &split_size in split_sizes {
        let mut chunk_shape = shape.to_vec();
        chunk_shape[dim] = split_size;
        let chunk_numel: usize = chunk_shape.iter().product();
        let mut chunk_data = vec![<T as num_traits::Zero>::zero(); chunk_numel];

        for o in 0..outer {
            let src_start = o * total_along_dim * inner + offset_along_dim * inner;
            let dst_start = o * split_size * inner;
            let row_len = split_size * inner;
            chunk_data[dst_start..dst_start + row_len]
                .copy_from_slice(&in_data[src_start..src_start + row_len]);
        }

        let storage = TensorStorage::on_device(chunk_data, device)?;
        let t = if needs_grad {
            let grad_fn = Arc::new(SplitBackward::new(
                input.clone(),
                dim,
                offset_along_dim,
                split_size,
            ));
            Tensor::from_operation(storage, chunk_shape, grad_fn)?
        } else {
            Tensor::from_storage(storage, chunk_shape, false)?
        };
        results.push(t);
        offset_along_dim += split_size;
    }

    Ok(results)
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn test_method_relu() {
        let a = scalar(2.0f32).unwrap();
        assert_eq!(a.relu().unwrap().item().unwrap(), 2.0);

        let b = scalar(-1.0f32).unwrap();
        assert_eq!(b.relu().unwrap().item().unwrap(), 0.0);
    }

    #[test]
    fn test_method_matmul() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let b = from_slice(&[5.0, 6.0, 7.0, 8.0], &[2, 2]).unwrap();
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape(), &[2, 2]);
    }

    #[test]
    fn test_method_sum() {
        let a = tensor(&[1.0f32, 2.0, 3.0]).unwrap();
        let s = a.sum_all().unwrap();
        assert_eq!(s.item().unwrap(), 6.0);
    }

    #[test]
    fn test_method_transpose() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = a.t().unwrap();
        assert_eq!(b.shape(), &[3, 2]);
    }

    #[test]
    fn test_method_chain() {
        let a = scalar(3.0f32).unwrap().requires_grad_(true);
        // a.pow(2).relu().sum() = relu(9) = 9
        let c = a.pow_t(2.0).unwrap().relu().unwrap();
        assert_eq!(c.item().unwrap(), 9.0);
    }

    #[test]
    fn test_method_sigmoid() {
        let a = scalar(0.0f32).unwrap();
        let s = a.sigmoid().unwrap();
        assert!((s.item().unwrap() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_method_flatten() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let f = a.flatten_t().unwrap();
        assert_eq!(f.shape(), &[6]);
    }

    #[test]
    fn test_method_print_chain() {
        let a = scalar(42.0f32).unwrap();
        // .print() returns &Self for chaining
        let _ = a.print();
    }

    // --- sum_dim / mean_dim method wrappers ---

    #[test]
    fn test_method_sum_dim() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let s = a.sum_dim(1, false).unwrap();
        assert_eq!(s.shape(), &[2]);
        assert!((s.data().unwrap()[0] - 6.0).abs() < 1e-6);
        assert!((s.data().unwrap()[1] - 15.0).abs() < 1e-6);
    }

    #[test]
    fn test_method_mean_dim() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let m = a.mean_dim(0, false).unwrap();
        assert_eq!(m.shape(), &[3]);
        assert!((m.data().unwrap()[0] - 2.5).abs() < 1e-6);
    }

    // --- permute ---

    #[test]
    fn test_method_permute_2d() {
        // Transpose via permute
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = a.permute(&[1, 0]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data().unwrap(), &[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_method_permute_3d() {
        let data: Vec<f32> = (1..=24).map(|x| x as f32).collect();
        let a = from_slice(&data, &[2, 3, 4]).unwrap();
        let b = a.permute(&[2, 0, 1]).unwrap();
        assert_eq!(b.shape(), &[4, 2, 3]);
        // Verify element [0,0,0] of output = element [0,0,0] of input = 1.0
        assert_eq!(b.data().unwrap()[0], 1.0);
        // element [1,0,0] of output = input[0,0,1] = 2.0
        assert_eq!(b.data().unwrap()[1 * 2 * 3], 2.0);
    }

    #[test]
    fn test_permute_invalid_dims() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        assert!(a.permute(&[0]).is_err()); // wrong length
        assert!(a.permute(&[0, 0]).is_err()); // duplicate
        assert!(a.permute(&[0, 2]).is_err()); // out of bounds
    }

    // --- view ---

    #[test]
    fn test_method_view() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let b = a.view(&[3, 2]).unwrap();
        assert_eq!(b.shape(), &[3, 2]);
        assert_eq!(b.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_method_view_infer() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
        let b = a.view(&[2, -1]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
    }

    // --- contiguous ---

    #[test]
    fn test_method_contiguous() {
        let a = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = a.contiguous().unwrap();
        assert_eq!(b.shape(), &[3]);
        assert_eq!(b.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    // --- chunk ---

    #[test]
    fn test_method_chunk_even() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[6]).unwrap();
        let chunks = a.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].data().unwrap(), &[1.0, 2.0]);
        assert_eq!(chunks[1].data().unwrap(), &[3.0, 4.0]);
        assert_eq!(chunks[2].data().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_method_chunk_uneven() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let chunks = a.chunk(3, 0).unwrap();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0].shape(), &[2]);
        assert_eq!(chunks[1].shape(), &[2]);
        assert_eq!(chunks[2].shape(), &[1]);
    }

    #[test]
    fn test_method_chunk_2d() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]).unwrap();
        let chunks = a.chunk(2, 0).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[2, 2]);
        assert_eq!(chunks[1].shape(), &[1, 2]);
    }

    // --- split ---

    #[test]
    fn test_method_split() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0], &[5]).unwrap();
        let parts = a.split(&[2, 3], 0).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].data().unwrap(), &[1.0, 2.0]);
        assert_eq!(parts[1].data().unwrap(), &[3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_method_split_2d_axis1() {
        let a = from_slice(
            &[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
            &[2, 4],
        )
        .unwrap();
        let parts = a.split(&[1, 3], 1).unwrap();
        assert_eq!(parts.len(), 2);
        assert_eq!(parts[0].shape(), &[2, 1]);
        assert_eq!(parts[0].data().unwrap(), &[1.0, 5.0]);
        assert_eq!(parts[1].shape(), &[2, 3]);
        assert_eq!(parts[1].data().unwrap(), &[2.0, 3.0, 4.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn test_split_bad_sizes() {
        let a = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        assert!(a.split(&[1, 1], 0).is_err()); // sum != 3
    }

    // --- split/chunk autograd ---

    #[test]
    fn test_split_preserves_grad() {
        // Split a requires-grad tensor and verify chunks have grad_fn.
        let a = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![6],
            true,
        ).unwrap();
        let chunks = a.split(&[2, 4], 0).unwrap();
        assert!(chunks[0].grad_fn().is_some(), "chunk 0 should have grad_fn");
        assert!(chunks[1].grad_fn().is_some(), "chunk 1 should have grad_fn");
    }

    #[test]
    fn test_split_backward_simple() {
        // x = [1, 2, 3, 4, 5, 6], split into [1,2,3] and [4,5,6].
        // loss = sum(chunk0) + 2*sum(chunk1)
        // d_loss/d_x = [1, 1, 1, 2, 2, 2]
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![6],
            true,
        ).unwrap();
        let chunks = x.split(&[3, 3], 0).unwrap();

        let sum0 = crate::grad_fns::reduction::sum(&chunks[0]).unwrap();
        let sum1 = crate::grad_fns::reduction::sum(&chunks[1]).unwrap();

        // 2 * sum1
        let two = Tensor::from_storage(
            TensorStorage::cpu(vec![2.0f64]),
            vec![],
            false,
        ).unwrap();
        let scaled = crate::grad_fns::arithmetic::mul(&sum1, &two).unwrap();
        let loss = crate::grad_fns::arithmetic::add(&sum0, &scaled).unwrap();

        loss.backward().unwrap();

        let grad = x.grad().unwrap().expect("x should have grad");
        assert_eq!(grad.shape(), &[6]);
        let g = grad.data().unwrap();
        // First 3 elements: grad from sum0 = 1.0 each
        // Last 3 elements: grad from 2*sum1 = 2.0 each
        for i in 0..3 {
            assert!(
                (g[i] - 1.0).abs() < 1e-10,
                "grad[{i}] = {}, expected 1.0", g[i]
            );
        }
        for i in 3..6 {
            assert!(
                (g[i] - 2.0).abs() < 1e-10,
                "grad[{i}] = {}, expected 2.0", g[i]
            );
        }
    }

    #[test]
    fn test_chunk_backward_2d() {
        // x shape [2, 4], chunk into 2 along dim=1 -> two [2, 2] tensors.
        // loss = sum(chunk0) * 3 + sum(chunk1)
        // grad_x[:, 0:2] = 3, grad_x[:, 2:4] = 1
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f64; 8]),
            vec![2, 4],
            true,
        ).unwrap();
        let chunks = x.chunk(2, 1).unwrap();
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0].shape(), &[2, 2]);
        assert_eq!(chunks[1].shape(), &[2, 2]);

        let sum0 = crate::grad_fns::reduction::sum(&chunks[0]).unwrap();
        let sum1 = crate::grad_fns::reduction::sum(&chunks[1]).unwrap();

        let three = Tensor::from_storage(
            TensorStorage::cpu(vec![3.0f64]),
            vec![],
            false,
        ).unwrap();
        let scaled = crate::grad_fns::arithmetic::mul(&sum0, &three).unwrap();
        let loss = crate::grad_fns::arithmetic::add(&scaled, &sum1).unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().expect("x should have grad");
        assert_eq!(grad.shape(), &[2, 4]);
        let g = grad.data().unwrap();
        // Row 0: [3, 3, 1, 1], Row 1: [3, 3, 1, 1]
        let expected = [3.0, 3.0, 1.0, 1.0, 3.0, 3.0, 1.0, 1.0];
        for (i, (&actual, &exp)) in g.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - exp).abs() < 1e-10,
                "grad[{i}] = {actual}, expected {exp}"
            );
        }
    }

    #[test]
    fn test_split_no_grad_when_disabled() {
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
            vec![3],
            false, // no grad
        ).unwrap();
        let chunks = x.split(&[1, 2], 0).unwrap();
        assert!(chunks[0].grad_fn().is_none());
        assert!(chunks[1].grad_fn().is_none());
    }

    // --- size / dim aliases ---

    #[test]
    fn test_size_alias() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(a.size(), &[2, 3]);
        assert_eq!(a.size(), a.shape());
    }

    #[test]
    fn test_dim_alias() {
        let a = from_slice(&[1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        assert_eq!(a.dim(), 2);
        assert_eq!(a.dim(), a.ndim());
    }
}
