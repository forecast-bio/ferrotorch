//! Identity and Flatten modules.
//!
//! [`Identity`] passes input through unchanged — useful for model composition,
//! conditional layers, and debugging.
//!
//! [`Flatten`] reshapes input by flattening contiguous dimensions from
//! `start_dim` to `end_dim` into a single dimension. The default
//! (`start_dim=1, end_dim=-1`) flattens everything except the batch dimension.

use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::module::Module;
use crate::parameter::Parameter;

// ===========================================================================
// Identity
// ===========================================================================

/// A module that returns its input unchanged.
///
/// Useful as a placeholder in model architectures where a layer is
/// conditionally applied, or for debugging / hook attachment points.
///
/// Has zero learnable parameters.
///
/// # Examples
///
/// ```ignore
/// let id = Identity;
/// let output = id.forward(&input)?; // output == input
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Identity {
    training: bool,
}

impl Identity {
    /// Create a new `Identity` module.
    pub fn new() -> Self {
        Self { training: true }
    }
}

impl Default for Identity {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Module<T> for Identity {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Ok(input.clone())
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Flatten
// ===========================================================================

/// Flattens a contiguous range of dimensions in a tensor.
///
/// By default, flattens all dimensions except the batch dimension
/// (`start_dim=1, end_dim=-1`), producing output of shape `[B, *]`.
///
/// Negative `end_dim` values are resolved relative to the input's
/// number of dimensions (`-1` = last dim).
///
/// # Examples
///
/// ```ignore
/// // Input: [2, 3, 4, 5]
/// let flatten = Flatten::new(1, -1);
/// let output = flatten.forward(&input)?;
/// // Output: [2, 60]
///
/// // Flatten specific range
/// let flatten = Flatten::new(2, 3);
/// let output = flatten.forward(&input)?;
/// // Output: [2, 3, 20]
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Flatten {
    /// First dimension to flatten (inclusive).
    pub start_dim: usize,
    /// Last dimension to flatten (inclusive). Negative values count from the end.
    pub end_dim: isize,
    training: bool,
}

impl Flatten {
    /// Create a new `Flatten` module.
    ///
    /// # Arguments
    ///
    /// * `start_dim` - First dimension to flatten (inclusive, 0-indexed).
    /// * `end_dim` - Last dimension to flatten (inclusive). Use `-1` for the
    ///   last dimension, `-2` for second-to-last, etc.
    pub fn new(start_dim: usize, end_dim: isize) -> Self {
        Self {
            start_dim,
            end_dim,
            training: true,
        }
    }

    /// Resolve `end_dim` to a concrete dimension index.
    fn resolve_end_dim(&self, ndim: usize) -> FerrotorchResult<usize> {
        let resolved = if self.end_dim < 0 {
            let d = ndim as isize + self.end_dim;
            if d < 0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "Flatten: end_dim {} is out of range for input with {} dims",
                        self.end_dim, ndim
                    ),
                });
            }
            d as usize
        } else {
            self.end_dim as usize
        };

        if resolved >= ndim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Flatten: resolved end_dim {} is out of range for input with {} dims",
                    resolved, ndim
                ),
            });
        }

        Ok(resolved)
    }
}

impl Default for Flatten {
    fn default() -> Self {
        Self::new(1, -1)
    }
}

impl<T: Float> Module<T> for Flatten {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        let ndim = shape.len();

        // 0-D tensor: nothing to flatten.
        if ndim == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Flatten: cannot flatten a 0-D (scalar) tensor".into(),
            });
        }

        // 1-D tensor: already flat.
        if ndim == 1 {
            return Ok(input.clone());
        }

        if self.start_dim >= ndim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Flatten: start_dim {} is out of range for input with {} dims",
                    self.start_dim, ndim
                ),
            });
        }

        let end_dim = self.resolve_end_dim(ndim)?;

        if self.start_dim > end_dim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Flatten: start_dim ({}) must be <= end_dim ({})",
                    self.start_dim, end_dim
                ),
            });
        }

        // If start == end, no flattening needed.
        if self.start_dim == end_dim {
            return Ok(input.clone());
        }

        // Build new shape: [dims before start, flattened, dims after end].
        let mut new_shape: Vec<isize> = Vec::with_capacity(ndim - (end_dim - self.start_dim));

        for &d in &shape[..self.start_dim] {
            new_shape.push(d as isize);
        }

        // Flatten the range [start_dim..=end_dim] into one dim.
        let flattened: usize = shape[self.start_dim..=end_dim].iter().product();
        new_shape.push(flattened as isize);

        for &d in &shape[end_dim + 1..] {
            new_shape.push(d as isize);
        }

        reshape(input, &new_shape)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Unflatten
// ===========================================================================

/// Unflattens a dimension, expanding it into multiple dimensions.
///
/// The inverse of [`Flatten`]. Given an input where dimension `dim` has
/// size equal to the product of `unflattened_size`, reshapes that
/// dimension into the specified shape.
///
/// Matches PyTorch's `nn.Unflatten`.
#[derive(Debug, Clone)]
pub struct Unflatten {
    /// The dimension to unflatten.
    pub dim: usize,
    /// The target shape for the unflattened dimension.
    pub unflattened_size: Vec<usize>,
    training: bool,
}

impl Unflatten {
    pub fn new(dim: usize, unflattened_size: Vec<usize>) -> Self {
        Self {
            dim,
            unflattened_size,
            training: true,
        }
    }

    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        if self.dim >= shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Unflatten: dim {} out of range for input with {} dims",
                    self.dim,
                    shape.len()
                ),
            });
        }

        let expected_size: usize = self.unflattened_size.iter().product();
        if expected_size != shape[self.dim] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Unflatten: unflattened_size {:?} (product={}) doesn't match dim {} size {}",
                    self.unflattened_size, expected_size, self.dim, shape[self.dim]
                ),
            });
        }

        let mut new_shape = Vec::with_capacity(shape.len() - 1 + self.unflattened_size.len());
        new_shape.extend_from_slice(&shape[..self.dim]);
        new_shape.extend_from_slice(&self.unflattened_size);
        new_shape.extend_from_slice(&shape[self.dim + 1..]);

        input.view_reshape(new_shape)
    }
}

impl<T: Float> Module<T> for Unflatten {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Unflatten::forward(self, input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// ChannelShuffle
// ===========================================================================

/// Rearranges channels in a [N, C, H, W] tensor by dividing them into
/// groups and interleaving.
///
/// Used in ShuffleNet architectures. With `groups=g`, the channel
/// dimension is reshaped to `[g, C/g]`, transposed to `[C/g, g]`,
/// then flattened back to `[C]`.
///
/// Matches PyTorch's `nn.ChannelShuffle`.
#[derive(Debug, Clone)]
pub struct ChannelShuffle {
    pub groups: usize,
    training: bool,
}

impl ChannelShuffle {
    pub fn new(groups: usize) -> Self {
        Self {
            groups,
            training: true,
        }
    }

    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ChannelShuffle: input must have at least 2 dims, got {:?}",
                    input.shape()
                ),
            });
        }
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "ChannelShuffle",
            });
        }

        let shape = input.shape();
        let channels = shape[1];
        if channels % self.groups != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ChannelShuffle: channels ({}) must be divisible by groups ({})",
                    channels, self.groups
                ),
            });
        }

        let g = self.groups;
        let cpg = channels / g; // channels per group
        let batch = shape[0];
        let spatial: usize = shape[2..].iter().product();
        let data = input.data()?;

        // Reshape [N, C, *] → [N, g, cpg, *] → transpose → [N, cpg, g, *] → [N, C, *]
        let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];
        for n in 0..batch {
            for c_out in 0..channels {
                // c_out in the shuffled order: group index = c_out % g, within-group = c_out / g
                let c_in = (c_out % g) * cpg + (c_out / g);
                for s in 0..spatial {
                    out[n * channels * spatial + c_out * spatial + s] =
                        data[n * channels * spatial + c_in * spatial + s];
                }
            }
        }

        Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(out),
            shape.to_vec(),
            false,
        )
    }
}

impl<T: Float> Module<T> for ChannelShuffle {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        ChannelShuffle::forward(self, input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// CosineSimilarity
// ===========================================================================

/// Computes cosine similarity between two tensors along a dimension.
///
/// `cos(x1, x2) = (x1 . x2) / (||x1|| * ||x2||)`
///
/// Matches PyTorch's `nn.CosineSimilarity`.
#[derive(Debug, Clone)]
pub struct CosineSimilarity {
    /// Dimension along which to compute cosine similarity.
    pub dim: usize,
    /// Small value to avoid division by zero.
    pub eps: f64,
}

impl CosineSimilarity {
    pub fn new(dim: usize, eps: f64) -> Self {
        Self { dim, eps }
    }

    pub fn forward<T: Float>(&self, x1: &Tensor<T>, x2: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if x1.shape() != x2.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "CosineSimilarity: shapes must match, got {:?} and {:?}",
                    x1.shape(),
                    x2.shape()
                ),
            });
        }
        if x1.is_cuda() || x2.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CosineSimilarity",
            });
        }

        let shape = x1.shape();
        if self.dim >= shape.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CosineSimilarity: dim {} out of range for shape {:?}",
                    self.dim, shape
                ),
            });
        }

        let d1 = x1.data()?;
        let d2 = x2.data()?;
        let dim_size = shape[self.dim];
        let outer: usize = shape[..self.dim].iter().product();
        let inner: usize = shape[self.dim + 1..].iter().product();
        let eps_t = T::from(self.eps).unwrap();

        let out_numel = outer * inner;
        let mut result = Vec::with_capacity(out_numel);

        for o in 0..outer {
            for i in 0..inner {
                let mut dot = <T as num_traits::Zero>::zero();
                let mut n1 = <T as num_traits::Zero>::zero();
                let mut n2 = <T as num_traits::Zero>::zero();
                for d in 0..dim_size {
                    let idx = o * dim_size * inner + d * inner + i;
                    dot += d1[idx] * d2[idx];
                    n1 += d1[idx] * d1[idx];
                    n2 += d2[idx] * d2[idx];
                }
                let denom = (n1.sqrt() * n2.sqrt()).max(eps_t);
                result.push(dot / denom);
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape.remove(self.dim);
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(result),
            out_shape,
            false,
        )
    }
}

impl Default for CosineSimilarity {
    fn default() -> Self {
        Self::new(1, 1e-8)
    }
}

// ===========================================================================
// PairwiseDistance
// ===========================================================================

/// Computes the pairwise distance between two tensors using the p-norm.
///
/// `d(x1, x2) = ||x1 - x2||_p`
///
/// Matches PyTorch's `nn.PairwiseDistance`.
#[derive(Debug, Clone)]
pub struct PairwiseDistance {
    /// The norm degree (default: 2.0 for Euclidean).
    pub p: f64,
    /// Small value to avoid division by zero.
    pub eps: f64,
    /// Whether to keep the output dimension.
    pub keepdim: bool,
}

impl PairwiseDistance {
    pub fn new(p: f64, eps: f64, keepdim: bool) -> Self {
        Self { p, eps, keepdim }
    }

    pub fn forward<T: Float>(&self, x1: &Tensor<T>, x2: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if x1.shape() != x2.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "PairwiseDistance: shapes must match, got {:?} and {:?}",
                    x1.shape(),
                    x2.shape()
                ),
            });
        }
        if x1.is_cuda() || x2.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "PairwiseDistance",
            });
        }

        let shape = x1.shape();
        let ndim = shape.len();
        if ndim == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "PairwiseDistance: input must have at least 1 dimension".into(),
            });
        }

        let d1 = x1.data()?;
        let d2 = x2.data()?;
        let last_dim = shape[ndim - 1];
        let outer: usize = d1.len() / last_dim;
        let p_t = T::from(self.p).unwrap();
        let inv_p = T::from(1.0 / self.p).unwrap();
        let eps_t = T::from(self.eps).unwrap();

        let mut result = Vec::with_capacity(outer);
        for o in 0..outer {
            let mut norm = <T as num_traits::Zero>::zero();
            for i in 0..last_dim {
                let diff = d1[o * last_dim + i] - d2[o * last_dim + i];
                let abs_diff = if diff < <T as num_traits::Zero>::zero() {
                    <T as num_traits::Zero>::zero() - diff
                } else {
                    diff
                };
                norm += (abs_diff + eps_t).powf(p_t);
            }
            result.push(norm.powf(inv_p));
        }

        let mut out_shape: Vec<usize> = shape[..ndim - 1].to_vec();
        if self.keepdim {
            out_shape.push(1);
        }
        if out_shape.is_empty() {
            out_shape.push(1);
        }
        Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(result),
            out_shape,
            false,
        )
    }
}

impl Default for PairwiseDistance {
    fn default() -> Self {
        Self::new(2.0, 1e-6, false)
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::autograd::graph::backward;
    use ferrotorch_core::storage::TensorStorage;

    /// Helper: create a leaf tensor with given data, shape, and requires_grad.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Identity tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_identity_forward() {
        let id = Identity::new();
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let output: Tensor<f64> = id.forward(&input).unwrap();
        assert_eq!(output.shape(), input.shape());
        assert_eq!(output.data_vec().unwrap(), input.data_vec().unwrap());
    }

    #[test]
    fn test_identity_no_parameters() {
        let id = Identity::new();
        assert!(Module::<f64>::parameters(&id).is_empty());
        assert!(Module::<f64>::named_parameters(&id).is_empty());
    }

    #[test]
    fn test_identity_preserves_grad() {
        let id = Identity::new();
        let input = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let output: Tensor<f64> = id.forward(&input).unwrap();
        assert!(output.requires_grad());
    }

    #[test]
    fn test_identity_train_eval() {
        let mut id = Identity::new();
        assert!(Module::<f64>::is_training(&id));
        Module::<f64>::eval(&mut id);
        assert!(!Module::<f64>::is_training(&id));
        Module::<f64>::train(&mut id);
        assert!(Module::<f64>::is_training(&id));
    }

    #[test]
    fn test_identity_empty_tensor() {
        let id = Identity::new();
        let input = leaf(&[], &[0], false);
        let output: Tensor<f64> = id.forward(&input).unwrap();
        assert_eq!(output.shape(), &[0]);
        assert_eq!(output.numel(), 0);
    }

    #[test]
    fn test_identity_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Identity>();
    }

    // -----------------------------------------------------------------------
    // Flatten tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_flatten_default() {
        // Default: start_dim=1, end_dim=-1 => flatten all but batch.
        let flatten = Flatten::default();
        let input = leaf(
            &(0..120).map(|i| i as f64).collect::<Vec<_>>(),
            &[2, 3, 4, 5],
            false,
        );
        let output: Tensor<f64> = flatten.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 60]);
    }

    #[test]
    fn test_flatten_specific_range() {
        // Flatten dims 2..3 of [2, 3, 4, 5] => [2, 3, 20].
        let flatten = Flatten::new(2, 3);
        let input = leaf(
            &(0..120).map(|i| i as f64).collect::<Vec<_>>(),
            &[2, 3, 4, 5],
            false,
        );
        let output: Tensor<f64> = flatten.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3, 20]);
    }

    #[test]
    fn test_flatten_all_dims() {
        // start_dim=0, end_dim=-1 => flatten everything.
        let flatten = Flatten::new(0, -1);
        let input = leaf(
            &(0..24).map(|i| i as f64).collect::<Vec<_>>(),
            &[2, 3, 4],
            false,
        );
        let output: Tensor<f64> = flatten.forward(&input).unwrap();
        assert_eq!(output.shape(), &[24]);
    }

    #[test]
    fn test_flatten_noop_single_dim() {
        // start_dim == end_dim => no-op.
        let flatten = Flatten::new(1, 1);
        let input = leaf(
            &(0..12).map(|i| i as f64).collect::<Vec<_>>(),
            &[3, 4],
            false,
        );
        let output: Tensor<f64> = flatten.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3, 4]);
    }

    #[test]
    fn test_flatten_1d_input() {
        // 1-D input: already flat, should return as-is.
        let flatten = Flatten::new(0, -1);
        let input = leaf(&[1.0, 2.0, 3.0], &[3], false);
        let output: Tensor<f64> = flatten.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3]);
    }

    #[test]
    fn test_flatten_0d_error() {
        // 0-D tensor should error.
        let flatten = Flatten::new(0, -1);
        let input = leaf(&[42.0], &[], false);
        assert!(Module::<f64>::forward(&flatten, &input).is_err());
    }

    #[test]
    fn test_flatten_start_dim_out_of_range() {
        let flatten = Flatten::new(5, -1);
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        assert!(Module::<f64>::forward(&flatten, &input).is_err());
    }

    #[test]
    fn test_flatten_end_dim_out_of_range() {
        let flatten = Flatten::new(0, 10);
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        assert!(Module::<f64>::forward(&flatten, &input).is_err());
    }

    #[test]
    fn test_flatten_start_gt_end_error() {
        let flatten = Flatten::new(2, 1);
        let input = leaf(
            &(0..24).map(|i| i as f64).collect::<Vec<_>>(),
            &[2, 3, 4],
            false,
        );
        assert!(Module::<f64>::forward(&flatten, &input).is_err());
    }

    #[test]
    fn test_flatten_preserves_data() {
        let flatten = Flatten::default();
        let data: Vec<f64> = (0..24).map(|i| i as f64).collect();
        let input = leaf(&data, &[2, 3, 4], false);
        let output: Tensor<f64> = flatten.forward(&input).unwrap();
        assert_eq!(output.data_vec().unwrap(), data);
    }

    #[test]
    fn test_flatten_backward() {
        use ferrotorch_core::tensor::GradFn;
        use std::sync::Arc;

        /// Sum backward helper that propagates gradients.
        #[derive(Debug)]
        struct SumBackwardHelper {
            input: Tensor<f64>,
        }

        impl GradFn<f64> for SumBackwardHelper {
            fn backward(
                &self,
                _grad_output: &Tensor<f64>,
            ) -> FerrotorchResult<Vec<Option<Tensor<f64>>>> {
                let ones_data = vec![1.0f64; self.input.numel()];
                let ones = Tensor::from_storage(
                    TensorStorage::cpu(ones_data),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(ones)])
            }

            fn inputs(&self) -> Vec<&Tensor<f64>> {
                vec![&self.input]
            }

            fn name(&self) -> &'static str {
                "SumBackwardHelper"
            }
        }

        let flatten = Flatten::default();
        let input = leaf(
            &(0..24).map(|i| i as f64).collect::<Vec<_>>(),
            &[2, 3, 4],
            true,
        );
        let output: Tensor<f64> = flatten.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 12]);
        assert!(output.requires_grad());

        // Trigger backward through a differentiable sum.
        let out_data = output.data().unwrap();
        let total: f64 = out_data.iter().sum();
        let sum_gf = Arc::new(SumBackwardHelper {
            input: output.clone(),
        });
        let loss = Tensor::from_operation(TensorStorage::cpu(vec![total]), vec![], sum_gf).unwrap();
        backward(&loss).unwrap();

        let grad = input.grad().unwrap().unwrap();
        assert_eq!(grad.shape(), &[2, 3, 4]);
        // Gradient of sum is all ones.
        for &v in grad.data().unwrap().iter() {
            assert!((v - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_flatten_no_parameters() {
        let flatten = Flatten::default();
        assert!(Module::<f64>::parameters(&flatten).is_empty());
        assert!(Module::<f64>::named_parameters(&flatten).is_empty());
    }

    #[test]
    fn test_flatten_zero_size_dim() {
        // Tensor with a zero-size dimension should still work.
        let flatten = Flatten::default();
        let input = leaf(&[], &[2, 0, 4], false);
        let output: Tensor<f64> = flatten.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 0]);
    }

    #[test]
    fn test_flatten_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Flatten>();
    }
}
