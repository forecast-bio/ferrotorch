//! Method-style API for Tensor operations.
//!
//! Enables `a.matmul(&b)`, `a.relu()`, `a.sum()`, `a.reshape(&[2, 3])` etc.
//! All methods delegate to the corresponding grad_fns or ops functions.

use crate::dtype::Float;
use crate::error::FerrotorchResult;
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

    // --- Utility ---

    /// Print the tensor and return self for chaining.
    pub fn print(&self) -> &Self {
        println!("{self}");
        self
    }
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
}
