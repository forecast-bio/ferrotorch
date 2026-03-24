//! Nested (jagged) tensors for variable-length sequences.
//!
//! A `NestedTensor` packs variable-length tensors into a single contiguous
//! buffer with an offsets array. This avoids the memory waste and compute
//! overhead of padding short sequences to the maximum length.
//!
//! The primary use case is efficient transformer attention on batches where
//! sequences have different lengths (e.g., NLP inputs of varying token counts).
//!
//! # Layout
//!
//! For a batch of sequences with shapes `[s0, D]`, `[s1, D]`, `[s2, D]`:
//!
//! ```text
//! values:  [  s0 * D elements  |  s1 * D elements  |  s2 * D elements  ]
//! offsets: [0,  s0,  s0+s1,  s0+s1+s2]
//! ```
//!
//! The ragged dimension is the one that varies across sequences (typically
//! dimension 1 = sequence length in `[batch, seq_len, ...]` layouts).

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::ops::linalg::mm_raw;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// A nested (jagged) tensor that packs variable-length sequences into
/// contiguous storage with an offsets array.
///
/// All sequences must agree on every dimension except the ragged one.
/// For example, three sequences of shapes `[3, 8]`, `[5, 8]`, `[2, 8]`
/// can be packed with `ragged_dim = 0` (the first dim of each constituent
/// tensor, which corresponds to the seq_len axis).
///
/// # Type parameter
///
/// `T` must implement [`Float`] (currently `f32` or `f64`), matching
/// the constraint on `Tensor<T>`.
#[derive(Debug, Clone)]
pub struct NestedTensor<T: Float> {
    /// Contiguous buffer holding all sequences packed together.
    values: Tensor<T>,
    /// Offsets into the values buffer along the ragged dimension.
    /// Length = num_sequences + 1.
    /// Sequence `i` spans rows `offsets[i]..offsets[i+1]`.
    offsets: Vec<usize>,
    /// The ragged dimension within each constituent tensor (usually 0,
    /// which corresponds to the seq_len axis when the outer batch
    /// dimension is implicit).
    ragged_dim: usize,
    /// The shape of each constituent tensor with the ragged dim removed.
    /// For example, if constituent shapes are `[s_i, D]` with ragged_dim=0,
    /// then `non_ragged_shape = [D]`.
    non_ragged_shape: Vec<usize>,
}

impl<T: Float> NestedTensor<T> {
    /// Pack variable-length tensors into a single `NestedTensor`.
    ///
    /// All tensors must have the same number of dimensions and must agree
    /// on every dimension except `ragged_dim` (default: 0). The ragged
    /// dimension is the one that varies (typically the sequence-length axis).
    ///
    /// # Arguments
    ///
    /// - `tensors` - Slice of tensors to pack. Must be non-empty.
    ///
    /// # Errors
    ///
    /// - Empty input slice
    /// - Tensors with mismatched number of dimensions
    /// - Tensors with mismatched sizes on non-ragged dimensions
    /// - GPU tensors (not yet supported)
    pub fn new(tensors: &[Tensor<T>]) -> FerrotorchResult<Self> {
        Self::with_ragged_dim(tensors, 0)
    }

    /// Pack variable-length tensors with an explicit ragged dimension.
    ///
    /// See [`new`](Self::new) for details; `ragged_dim` specifies which
    /// dimension of the constituent tensors varies in length.
    pub fn with_ragged_dim(tensors: &[Tensor<T>], ragged_dim: usize) -> FerrotorchResult<Self> {
        if tensors.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "NestedTensor::new requires at least one tensor".into(),
            });
        }

        let ndim = tensors[0].ndim();
        if ndim == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "NestedTensor does not support 0-dimensional tensors".into(),
            });
        }
        if ragged_dim >= ndim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ragged_dim {} out of range for {}-dimensional tensors",
                    ragged_dim, ndim
                ),
            });
        }

        // Compute the non-ragged shape from the first tensor.
        let ref_shape = tensors[0].shape();
        let non_ragged_shape: Vec<usize> = ref_shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != ragged_dim)
            .map(|(_, &s)| s)
            .collect();

        // Product of non-ragged dimensions — elements per "row" along the ragged axis.
        let non_ragged_numel: usize = non_ragged_shape.iter().product::<usize>().max(1);

        // Validate all tensors and build offsets.
        let mut offsets = Vec::with_capacity(tensors.len() + 1);
        offsets.push(0usize);
        let mut total_ragged: usize = 0;

        for (idx, t) in tensors.iter().enumerate() {
            if t.ndim() != ndim {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "tensor {} has {} dimensions but tensor 0 has {}",
                        idx,
                        t.ndim(),
                        ndim
                    ),
                });
            }
            // Check non-ragged dims match.
            for (d, &expected) in non_ragged_shape.iter().enumerate() {
                // Map d back to the original axis index.
                let orig_axis = if d < ragged_dim { d } else { d + 1 };
                if t.shape()[orig_axis] != expected {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "tensor {} has size {} on axis {} but expected {} \
                             (all non-ragged dimensions must match)",
                            idx, t.shape()[orig_axis], orig_axis, expected
                        ),
                    });
                }
            }
            let ragged_len = t.shape()[ragged_dim];
            total_ragged += ragged_len;
            offsets.push(total_ragged);
        }

        // Copy all tensor data into a single contiguous buffer.
        let total_elements = total_ragged * non_ragged_numel;
        let mut packed = Vec::with_capacity(total_elements);
        for t in tensors {
            let data = t.data_vec()?;
            packed.extend_from_slice(&data);
        }

        debug_assert_eq!(packed.len(), total_elements);

        let values = Tensor::from_storage(
            TensorStorage::cpu(packed),
            vec![total_elements],
            false,
        )?;

        Ok(Self {
            values,
            offsets,
            ragged_dim,
            non_ragged_shape,
        })
    }

    /// Unpack the nested tensor back into individual tensors.
    ///
    /// This is the inverse of [`new`](Self::new).
    pub fn unbind(&self) -> FerrotorchResult<Vec<Tensor<T>>> {
        let data = self.values.data()?;
        let non_ragged_numel: usize = self.non_ragged_shape.iter().product::<usize>().max(1);
        let n = self.num_sequences();
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let ragged_len = self.offsets[i + 1] - self.offsets[i];
            let elem_start = self.offsets[i] * non_ragged_numel;
            let elem_end = self.offsets[i + 1] * non_ragged_numel;
            let slice = data[elem_start..elem_end].to_vec();

            // Reconstruct the full shape with the ragged dimension inserted.
            let shape = self.constituent_shape(ragged_len);

            result.push(Tensor::from_storage(
                TensorStorage::cpu(slice),
                shape,
                false,
            )?);
        }

        Ok(result)
    }

    /// The number of sequences (batch elements) in this nested tensor.
    #[inline]
    pub fn num_sequences(&self) -> usize {
        self.offsets.len() - 1
    }

    /// The length of sequence `idx` along the ragged dimension.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= num_sequences()`.
    #[inline]
    pub fn sequence_length(&self, idx: usize) -> usize {
        assert!(
            idx < self.num_sequences(),
            "sequence index {} out of range for {} sequences",
            idx,
            self.num_sequences()
        );
        self.offsets[idx + 1] - self.offsets[idx]
    }

    /// The maximum sequence length along the ragged dimension.
    #[inline]
    pub fn max_sequence_length(&self) -> usize {
        let n = self.num_sequences();
        (0..n)
            .map(|i| self.offsets[i + 1] - self.offsets[i])
            .max()
            .unwrap_or(0)
    }

    /// The total number of elements across all sequences.
    #[inline]
    pub fn total_elements(&self) -> usize {
        self.values.numel()
    }

    /// The ragged dimension index within each constituent tensor.
    #[inline]
    pub fn ragged_dim(&self) -> usize {
        self.ragged_dim
    }

    /// The offsets array. Length = `num_sequences() + 1`.
    #[inline]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// A reference to the packed values tensor.
    #[inline]
    pub fn values(&self) -> &Tensor<T> {
        &self.values
    }

    /// The shape of non-ragged dimensions (ragged dim removed).
    #[inline]
    pub fn non_ragged_shape(&self) -> &[usize] {
        &self.non_ragged_shape
    }

    /// Convert to a dense padded tensor of shape
    /// `[num_sequences, max_len, ...non_ragged...]` where shorter sequences
    /// are padded with `padding_value`.
    ///
    /// The output always has the batch dimension as dim 0 and the ragged
    /// dimension as dim 1, regardless of the original `ragged_dim`.
    pub fn to_padded(&self, padding_value: T) -> FerrotorchResult<Tensor<T>> {
        let n = self.num_sequences();
        let max_len = self.max_sequence_length();
        let non_ragged_numel: usize = self.non_ragged_shape.iter().product::<usize>().max(1);
        let row_size = max_len * non_ragged_numel;
        let total = n * row_size;

        let data = self.values.data()?;

        let mut padded = vec![padding_value; total];
        for i in 0..n {
            let src_start = self.offsets[i] * non_ragged_numel;
            let src_end = self.offsets[i + 1] * non_ragged_numel;
            let dst_start = i * row_size;
            padded[dst_start..dst_start + (src_end - src_start)]
                .copy_from_slice(&data[src_start..src_end]);
        }

        // Build output shape: [batch, max_len, ...non_ragged_shape...]
        let mut shape = Vec::with_capacity(2 + self.non_ragged_shape.len());
        shape.push(n);
        shape.push(max_len);
        shape.extend_from_slice(&self.non_ragged_shape);

        Tensor::from_storage(TensorStorage::cpu(padded), shape, false)
    }

    /// Create a `NestedTensor` from a padded tensor and per-sequence lengths.
    ///
    /// # Arguments
    ///
    /// - `tensor` - Dense padded tensor of shape `[batch, max_len, ...]`.
    /// - `lengths` - Actual length of each sequence. Length must equal the
    ///   batch dimension.
    ///
    /// # Errors
    ///
    /// - `lengths.len()` does not match the batch dimension.
    /// - Any length exceeds the padded sequence dimension.
    pub fn from_padded(tensor: &Tensor<T>, lengths: &[usize]) -> FerrotorchResult<Self> {
        let shape = tensor.shape();
        if shape.len() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "from_padded requires at least 2D tensor, got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let max_len = shape[1];

        if lengths.len() != batch {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "lengths has {} entries but tensor batch dimension is {}",
                    lengths.len(),
                    batch
                ),
            });
        }

        // Non-ragged shape is everything after dim 1.
        let non_ragged_shape: Vec<usize> = shape[2..].to_vec();
        let non_ragged_numel: usize = non_ragged_shape.iter().product::<usize>().max(1);
        let row_size = max_len * non_ragged_numel;

        let data = tensor.data_vec()?;

        // Compute total elements and build offsets.
        let mut offsets = Vec::with_capacity(batch + 1);
        offsets.push(0usize);
        let mut total_ragged: usize = 0;

        for (i, &len) in lengths.iter().enumerate() {
            if len > max_len {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "length {} for sequence {} exceeds padded dimension {}",
                        len, i, max_len
                    ),
                });
            }
            total_ragged += len;
            offsets.push(total_ragged);
        }

        let total_elements = total_ragged * non_ragged_numel;
        let mut packed = Vec::with_capacity(total_elements);

        for (i, &len) in lengths.iter().enumerate() {
            let src_start = i * row_size;
            let copy_len = len * non_ragged_numel;
            packed.extend_from_slice(&data[src_start..src_start + copy_len]);
        }

        debug_assert_eq!(packed.len(), total_elements);

        let values = Tensor::from_storage(
            TensorStorage::cpu(packed),
            vec![total_elements],
            false,
        )?;

        Ok(Self {
            values,
            offsets,
            ragged_dim: 0,
            non_ragged_shape,
        })
    }

    // -----------------------------------------------------------------------
    // Element-wise arithmetic
    // -----------------------------------------------------------------------

    /// Element-wise addition of two nested tensors.
    ///
    /// Both must have the same number of sequences, the same offsets, and the
    /// same non-ragged shape.
    pub fn add(&self, other: &NestedTensor<T>) -> FerrotorchResult<NestedTensor<T>> {
        self.elementwise_binary(other, |a, b| a + b, "add")
    }

    /// Element-wise subtraction of two nested tensors.
    pub fn sub(&self, other: &NestedTensor<T>) -> FerrotorchResult<NestedTensor<T>> {
        self.elementwise_binary(other, |a, b| a - b, "sub")
    }

    /// Element-wise multiplication of two nested tensors.
    pub fn mul(&self, other: &NestedTensor<T>) -> FerrotorchResult<NestedTensor<T>> {
        self.elementwise_binary(other, |a, b| a * b, "mul")
    }

    /// Element-wise division of two nested tensors.
    pub fn div(&self, other: &NestedTensor<T>) -> FerrotorchResult<NestedTensor<T>> {
        self.elementwise_binary(other, |a, b| a / b, "div")
    }

    /// Add a scalar to every element.
    pub fn add_scalar(&self, scalar: T) -> FerrotorchResult<NestedTensor<T>> {
        self.elementwise_scalar(|a| a + scalar)
    }

    /// Subtract a scalar from every element.
    pub fn sub_scalar(&self, scalar: T) -> FerrotorchResult<NestedTensor<T>> {
        self.elementwise_scalar(|a| a - scalar)
    }

    /// Multiply every element by a scalar.
    pub fn mul_scalar(&self, scalar: T) -> FerrotorchResult<NestedTensor<T>> {
        self.elementwise_scalar(|a| a * scalar)
    }

    /// Divide every element by a scalar.
    pub fn div_scalar(&self, scalar: T) -> FerrotorchResult<NestedTensor<T>> {
        self.elementwise_scalar(|a| a / scalar)
    }

    // -----------------------------------------------------------------------
    // Matmul
    // -----------------------------------------------------------------------

    /// Batched matrix multiply: each sequence `[s_i, K]` is multiplied by
    /// the same weight matrix `[K, N]`, producing a new nested tensor where
    /// each sequence has shape `[s_i, N]`.
    ///
    /// The weight matrix `other` must be a 2D tensor of shape `[K, N]` where
    /// `K` equals the last dimension of each sequence.
    ///
    /// # Errors
    ///
    /// - `other` is not 2D.
    /// - Inner dimension mismatch between sequences and the weight matrix.
    pub fn matmul(&self, other: &Tensor<T>) -> FerrotorchResult<NestedTensor<T>> {
        if other.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "NestedTensor::matmul requires a 2D weight matrix, got shape {:?}",
                    other.shape()
                ),
            });
        }

        let k = other.shape()[0];
        let n_out = other.shape()[1];

        // The last dimension of the non-ragged shape must equal K.
        if self.non_ragged_shape.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "NestedTensor::matmul requires at least 2D constituent tensors \
                          (ragged_dim + at least one non-ragged dim)"
                    .into(),
            });
        }

        let last_dim = *self.non_ragged_shape.last().unwrap();
        if last_dim != k {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "NestedTensor::matmul inner dimension mismatch: sequences have last dim {}, \
                     weight matrix has first dim {}",
                    last_dim, k
                ),
            });
        }

        let self_data = self.values.data()?;
        let other_data = other.data()?;

        // Product of non-ragged dims except the last (for higher-order tensors
        // we treat each sequence as [s_i * prod(non_ragged[:-1]), last_dim]).
        let leading_non_ragged: usize = self.non_ragged_shape[..self.non_ragged_shape.len() - 1]
            .iter()
            .product::<usize>()
            .max(1);

        let num_seq = self.num_sequences();
        let non_ragged_numel_in: usize = self.non_ragged_shape.iter().product::<usize>().max(1);

        // New non-ragged shape: replace last dim with n_out.
        let mut new_non_ragged = self.non_ragged_shape.clone();
        *new_non_ragged.last_mut().unwrap() = n_out;
        let non_ragged_numel_out: usize = new_non_ragged.iter().product::<usize>().max(1);

        // Compute total output elements.
        let total_ragged: usize = self.offsets[num_seq];
        let total_out = total_ragged * non_ragged_numel_out;
        let mut output = Vec::with_capacity(total_out);

        for i in 0..num_seq {
            let ragged_len = self.offsets[i + 1] - self.offsets[i];
            let m = ragged_len * leading_non_ragged; // rows for this sequence
            let src_start = self.offsets[i] * non_ragged_numel_in;
            let src_end = self.offsets[i + 1] * non_ragged_numel_in;
            let seq_data = &self_data[src_start..src_end];

            let result_vec = mm_raw(seq_data, other_data, m, k, n_out);
            output.extend_from_slice(&result_vec);
        }

        debug_assert_eq!(output.len(), total_out);

        let values = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![total_out],
            false,
        )?;

        Ok(NestedTensor {
            values,
            offsets: self.offsets.clone(),
            ragged_dim: self.ragged_dim,
            non_ragged_shape: new_non_ragged,
        })
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Apply a binary element-wise operation on the packed values buffers.
    fn elementwise_binary(
        &self,
        other: &NestedTensor<T>,
        op: impl Fn(T, T) -> T,
        op_name: &str,
    ) -> FerrotorchResult<NestedTensor<T>> {
        if self.offsets != other.offsets {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "NestedTensor::{}: offsets do not match \
                     (different sequence counts or lengths)",
                    op_name
                ),
            });
        }
        if self.non_ragged_shape != other.non_ragged_shape {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "NestedTensor::{}: non-ragged shapes do not match: {:?} vs {:?}",
                    op_name, self.non_ragged_shape, other.non_ragged_shape
                ),
            });
        }

        let a_data = self.values.data()?;
        let b_data = other.values.data()?;

        let result: Vec<T> = a_data
            .iter()
            .zip(b_data.iter())
            .map(|(&a, &b)| op(a, b))
            .collect();

        let values = Tensor::from_storage(
            TensorStorage::cpu(result),
            vec![a_data.len()],
            false,
        )?;

        Ok(NestedTensor {
            values,
            offsets: self.offsets.clone(),
            ragged_dim: self.ragged_dim,
            non_ragged_shape: self.non_ragged_shape.clone(),
        })
    }

    /// Apply a unary element-wise operation (used for scalar arithmetic).
    fn elementwise_scalar(
        &self,
        op: impl Fn(T) -> T,
    ) -> FerrotorchResult<NestedTensor<T>> {
        let data = self.values.data()?;
        let result: Vec<T> = data.iter().map(|&a| op(a)).collect();

        let values = Tensor::from_storage(
            TensorStorage::cpu(result),
            vec![data.len()],
            false,
        )?;

        Ok(NestedTensor {
            values,
            offsets: self.offsets.clone(),
            ragged_dim: self.ragged_dim,
            non_ragged_shape: self.non_ragged_shape.clone(),
        })
    }

    /// Reconstruct the full shape of a constituent tensor given its ragged length.
    fn constituent_shape(&self, ragged_len: usize) -> Vec<usize> {
        let ndim = self.non_ragged_shape.len() + 1;
        let mut shape = Vec::with_capacity(ndim);
        let mut nr_idx = 0;
        for d in 0..ndim {
            if d == self.ragged_dim {
                shape.push(ragged_len);
            } else {
                shape.push(self.non_ragged_shape[nr_idx]);
                nr_idx += 1;
            }
        }
        shape
    }
}

// ---------------------------------------------------------------------------
// Scaled dot-product attention for nested tensors
// ---------------------------------------------------------------------------

/// Perform scaled dot-product attention on nested (jagged) tensors.
///
/// Each sequence in the batch gets its own independent attention computation,
/// operating only on the actual tokens — no padding waste.
///
/// # Arguments
///
/// - `query` - Nested tensor where each sequence has shape `[s_i, D]`.
/// - `key` - Nested tensor where each sequence has shape `[s_i, D]`.
///   Must have the same offsets as `query`.
/// - `value` - Nested tensor where each sequence has shape `[s_i, D_v]`.
///   Must have the same offsets as `query`.
///
/// # Returns
///
/// A nested tensor where each sequence has shape `[s_i, D_v]`.
///
/// # Attention formula
///
/// For each sequence `i`:
///
/// ```text
/// scores_i = Q_i @ K_i^T / sqrt(D)
/// weights_i = softmax(scores_i, dim=-1)
/// output_i = weights_i @ V_i
/// ```
pub fn nested_scaled_dot_product_attention(
    query: &NestedTensor<f32>,
    key: &NestedTensor<f32>,
    value: &NestedTensor<f32>,
) -> FerrotorchResult<NestedTensor<f32>> {
    // Validate that all three have the same sequence structure.
    if query.offsets != key.offsets {
        return Err(FerrotorchError::ShapeMismatch {
            message: "nested_scaled_dot_product_attention: query and key \
                      have different sequence structures (offsets mismatch)"
                .into(),
        });
    }
    if query.offsets != value.offsets {
        return Err(FerrotorchError::ShapeMismatch {
            message: "nested_scaled_dot_product_attention: query and value \
                      have different sequence structures (offsets mismatch)"
                .into(),
        });
    }

    // Query and key must have the same last dimension (head_dim D).
    if query.non_ragged_shape.is_empty() || key.non_ragged_shape.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "nested_scaled_dot_product_attention requires at least 2D \
                      constituent tensors (ragged + feature dim)"
                .into(),
        });
    }
    let d_q = *query.non_ragged_shape.last().unwrap();
    let d_k = *key.non_ragged_shape.last().unwrap();
    if d_q != d_k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "nested_scaled_dot_product_attention: query last dim ({}) != \
                 key last dim ({})",
                d_q, d_k
            ),
        });
    }

    if value.non_ragged_shape.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "nested_scaled_dot_product_attention: value must have a \
                      feature dimension"
                .into(),
        });
    }
    let d_v = *value.non_ragged_shape.last().unwrap();
    let d = d_q;

    let scale = 1.0 / (d as f32).sqrt();
    let num_seq = query.num_sequences();

    let q_data = query.values.data()?;
    let k_data = key.values.data()?;
    let v_data = value.values.data()?;

    // Pre-compute total output size.
    let total_ragged: usize = query.offsets[num_seq];
    let total_out = total_ragged * d_v;
    let mut output = Vec::with_capacity(total_out);

    for i in 0..num_seq {
        let seq_len = query.offsets[i + 1] - query.offsets[i];

        if seq_len == 0 {
            continue;
        }

        // Extract slices for this sequence.
        let q_start = query.offsets[i] * d;
        let q_end = query.offsets[i + 1] * d;
        let q_seq = &q_data[q_start..q_end];

        let k_start = key.offsets[i] * d;
        let k_end = key.offsets[i + 1] * d;
        let k_seq = &k_data[k_start..k_end];

        let v_start = value.offsets[i] * d_v;
        let v_end = value.offsets[i + 1] * d_v;
        let v_seq = &v_data[v_start..v_end];

        // scores = Q @ K^T : [seq_len, d] @ [d, seq_len] -> [seq_len, seq_len]
        // We need K transposed. mm_raw expects row-major, so we transpose K
        // by doing Q @ K^T via mm_raw_bt (A @ B^T).
        let scores = crate::ops::linalg::mm_raw_bt(q_seq, k_seq, seq_len, d, seq_len);

        // Scale scores.
        let mut scaled_scores: Vec<f32> = scores.iter().map(|&s| s * scale).collect();

        // Softmax along last dimension (each row of the [seq_len, seq_len] matrix).
        softmax_rows_inplace(&mut scaled_scores, seq_len, seq_len);

        // context = weights @ V : [seq_len, seq_len] @ [seq_len, d_v] -> [seq_len, d_v]
        let context = mm_raw(&scaled_scores, v_seq, seq_len, seq_len, d_v);

        output.extend_from_slice(&context);
    }

    debug_assert_eq!(output.len(), total_out);

    let values = Tensor::from_storage(
        TensorStorage::cpu(output),
        vec![total_out],
        false,
    )?;

    Ok(NestedTensor {
        values,
        offsets: query.offsets.clone(),
        ragged_dim: query.ragged_dim,
        non_ragged_shape: value.non_ragged_shape.clone(),
    })
}

/// In-place row-wise softmax on a flat `[rows, cols]` matrix.
///
/// For each row, computes `exp(x - max) / sum(exp(x - max))`.
fn softmax_rows_inplace(data: &mut [f32], rows: usize, cols: usize) {
    debug_assert_eq!(data.len(), rows * cols);

    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];

        // Numerical stability: subtract max.
        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }

        if sum > 0.0 {
            let inv_sum = 1.0 / sum;
            for v in row.iter_mut() {
                *v *= inv_sum;
            }
        }
    }
}

// ===========================================================================
// Display
// ===========================================================================

impl<T: Float> std::fmt::Display for NestedTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let n = self.num_sequences();
        write!(
            f,
            "NestedTensor(num_sequences={}, lengths=[",
            n
        )?;
        for i in 0..n {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}", self.offsets[i + 1] - self.offsets[i])?;
        }
        write!(
            f,
            "], non_ragged_shape={:?}, total_elements={})",
            self.non_ragged_shape,
            self.total_elements()
        )
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::from_vec;

    /// Helper: create a simple 2D tensor [rows, cols] with sequential values.
    fn seq_tensor(rows: usize, cols: usize) -> Tensor<f32> {
        let n = rows * cols;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        from_vec(data, &[rows, cols]).unwrap()
    }

    /// Helper: create a 2D tensor filled with a constant.
    fn const_tensor(rows: usize, cols: usize, val: f32) -> Tensor<f32> {
        let data = vec![val; rows * cols];
        from_vec(data, &[rows, cols]).unwrap()
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_new_basic() {
        let t1 = seq_tensor(3, 4); // [3, 4]
        let t2 = seq_tensor(5, 4); // [5, 4]
        let t3 = seq_tensor(2, 4); // [2, 4]

        let nt = NestedTensor::new(&[t1, t2, t3]).unwrap();

        assert_eq!(nt.num_sequences(), 3);
        assert_eq!(nt.sequence_length(0), 3);
        assert_eq!(nt.sequence_length(1), 5);
        assert_eq!(nt.sequence_length(2), 2);
        assert_eq!(nt.max_sequence_length(), 5);
        assert_eq!(nt.total_elements(), (3 + 5 + 2) * 4);
        assert_eq!(nt.ragged_dim(), 0);
        assert_eq!(nt.non_ragged_shape(), &[4]);
    }

    #[test]
    fn test_new_single_tensor() {
        let t = seq_tensor(7, 3);
        let nt = NestedTensor::new(&[t]).unwrap();

        assert_eq!(nt.num_sequences(), 1);
        assert_eq!(nt.sequence_length(0), 7);
        assert_eq!(nt.max_sequence_length(), 7);
        assert_eq!(nt.total_elements(), 21);
    }

    #[test]
    fn test_new_empty_error() {
        let result = NestedTensor::<f32>::new(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_dim_mismatch_error() {
        let t1 = seq_tensor(3, 4);
        let t2 = from_vec(vec![0.0; 24], &[2, 3, 4]).unwrap();

        let result = NestedTensor::new(&[t1, t2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_new_non_ragged_mismatch_error() {
        let t1 = seq_tensor(3, 4);
        let t2 = seq_tensor(5, 5); // Different non-ragged dim

        let result = NestedTensor::new(&[t1, t2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_ragged_dim_out_of_range() {
        let t1 = seq_tensor(3, 4);
        let result = NestedTensor::with_ragged_dim(&[t1], 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_with_ragged_dim_1() {
        // Tensors of shape [2, s_i] — ragged on dim 1
        let t1 = from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]).unwrap();
        let t2 = from_vec(vec![7.0, 8.0, 9.0, 10.0], &[2, 2]).unwrap();

        let nt = NestedTensor::with_ragged_dim(&[t1, t2], 1).unwrap();

        assert_eq!(nt.num_sequences(), 2);
        assert_eq!(nt.sequence_length(0), 3);
        assert_eq!(nt.sequence_length(1), 2);
        assert_eq!(nt.ragged_dim(), 1);
        assert_eq!(nt.non_ragged_shape(), &[2]);
    }

    // -----------------------------------------------------------------------
    // Unbind (roundtrip)
    // -----------------------------------------------------------------------

    #[test]
    fn test_unbind_roundtrip() {
        let t1 = seq_tensor(3, 4);
        let t2 = seq_tensor(5, 4);
        let t3 = seq_tensor(2, 4);

        let t1_data = t1.data().unwrap().to_vec();
        let t2_data = t2.data().unwrap().to_vec();
        let t3_data = t3.data().unwrap().to_vec();

        let nt = NestedTensor::new(&[t1, t2, t3]).unwrap();
        let unbound = nt.unbind().unwrap();

        assert_eq!(unbound.len(), 3);
        assert_eq!(unbound[0].shape(), &[3, 4]);
        assert_eq!(unbound[1].shape(), &[5, 4]);
        assert_eq!(unbound[2].shape(), &[2, 4]);

        assert_eq!(unbound[0].data().unwrap(), &t1_data);
        assert_eq!(unbound[1].data().unwrap(), &t2_data);
        assert_eq!(unbound[2].data().unwrap(), &t3_data);
    }

    #[test]
    fn test_unbind_single() {
        let t = seq_tensor(4, 6);
        let original_data = t.data().unwrap().to_vec();

        let nt = NestedTensor::new(&[t]).unwrap();
        let unbound = nt.unbind().unwrap();

        assert_eq!(unbound.len(), 1);
        assert_eq!(unbound[0].shape(), &[4, 6]);
        assert_eq!(unbound[0].data().unwrap(), &original_data);
    }

    // -----------------------------------------------------------------------
    // to_padded / from_padded
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_padded_basic() {
        let t1 = const_tensor(2, 3, 1.0);
        let t2 = const_tensor(4, 3, 2.0);
        let t3 = const_tensor(1, 3, 3.0);

        let nt = NestedTensor::new(&[t1, t2, t3]).unwrap();
        let padded = nt.to_padded(0.0).unwrap();

        // Shape: [3, 4, 3] — batch=3, max_len=4, feature=3
        assert_eq!(padded.shape(), &[3, 4, 3]);

        let data = padded.data().unwrap();

        // Sequence 0: 2 rows of 1.0, then 2 rows of 0.0
        for i in 0..6 {
            assert_eq!(data[i], 1.0, "seq 0, elem {i}");
        }
        for i in 6..12 {
            assert_eq!(data[i], 0.0, "seq 0 padding, elem {i}");
        }

        // Sequence 1: 4 rows of 2.0
        for i in 12..24 {
            assert_eq!(data[i], 2.0, "seq 1, elem {i}");
        }

        // Sequence 2: 1 row of 3.0, then 3 rows of 0.0
        for i in 24..27 {
            assert_eq!(data[i], 3.0, "seq 2, elem {i}");
        }
        for i in 27..36 {
            assert_eq!(data[i], 0.0, "seq 2 padding, elem {i}");
        }
    }

    #[test]
    fn test_to_padded_with_custom_padding() {
        let t1 = const_tensor(1, 2, 5.0);
        let t2 = const_tensor(3, 2, 7.0);

        let nt = NestedTensor::new(&[t1, t2]).unwrap();
        let padded = nt.to_padded(-1.0).unwrap();

        assert_eq!(padded.shape(), &[2, 3, 2]);
        let data = padded.data().unwrap();

        // Seq 0: 1 row of 5.0, then 2 rows of -1.0
        assert_eq!(data[0], 5.0);
        assert_eq!(data[1], 5.0);
        assert_eq!(data[2], -1.0);
        assert_eq!(data[3], -1.0);
        assert_eq!(data[4], -1.0);
        assert_eq!(data[5], -1.0);
    }

    #[test]
    fn test_from_padded_basic() {
        // Create a padded tensor [2, 4, 3]
        let mut data = vec![0.0f32; 2 * 4 * 3];
        // Seq 0: 2 rows of actual data
        for i in 0..6 {
            data[i] = 1.0;
        }
        // Seq 1: 3 rows of actual data
        for i in 12..21 {
            data[i] = 2.0;
        }

        let padded = from_vec(data, &[2, 4, 3]).unwrap();
        let lengths = [2, 3];

        let nt = NestedTensor::from_padded(&padded, &lengths).unwrap();

        assert_eq!(nt.num_sequences(), 2);
        assert_eq!(nt.sequence_length(0), 2);
        assert_eq!(nt.sequence_length(1), 3);
        assert_eq!(nt.total_elements(), (2 + 3) * 3);
    }

    #[test]
    fn test_padded_roundtrip() {
        let t1 = seq_tensor(3, 4);
        let t2 = seq_tensor(5, 4);
        let t3 = seq_tensor(1, 4);

        let t1_data = t1.data().unwrap().to_vec();
        let t2_data = t2.data().unwrap().to_vec();
        let t3_data = t3.data().unwrap().to_vec();

        let nt = NestedTensor::new(&[t1, t2, t3]).unwrap();

        // Convert to padded and back.
        let padded = nt.to_padded(0.0).unwrap();
        let lengths = [3, 5, 1];
        let nt2 = NestedTensor::from_padded(&padded, &lengths).unwrap();

        let unbound = nt2.unbind().unwrap();
        assert_eq!(unbound[0].data().unwrap(), &t1_data);
        assert_eq!(unbound[1].data().unwrap(), &t2_data);
        assert_eq!(unbound[2].data().unwrap(), &t3_data);
    }

    #[test]
    fn test_from_padded_length_exceeds_dim_error() {
        let padded = from_vec(vec![0.0; 12], &[2, 3, 2]).unwrap();
        let lengths = [2, 4]; // 4 exceeds dim 1 = 3
        let result = NestedTensor::from_padded(&padded, &lengths);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_padded_wrong_batch_error() {
        let padded = from_vec(vec![0.0; 12], &[2, 3, 2]).unwrap();
        let lengths = [2, 3, 1]; // 3 entries but batch = 2
        let result = NestedTensor::from_padded(&padded, &lengths);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // Element-wise arithmetic
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_nested() {
        let t1 = const_tensor(3, 4, 1.0);
        let t2 = const_tensor(5, 4, 2.0);

        let a = NestedTensor::new(&[t1.clone(), t2.clone()]).unwrap();
        let b = NestedTensor::new(&[
            const_tensor(3, 4, 10.0),
            const_tensor(5, 4, 20.0),
        ])
        .unwrap();

        let c = a.add(&b).unwrap();

        let unbound = c.unbind().unwrap();
        for &v in unbound[0].data().unwrap() {
            assert!((v - 11.0).abs() < 1e-6);
        }
        for &v in unbound[1].data().unwrap() {
            assert!((v - 22.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sub_nested() {
        let a = NestedTensor::new(&[const_tensor(2, 3, 10.0)]).unwrap();
        let b = NestedTensor::new(&[const_tensor(2, 3, 3.0)]).unwrap();

        let c = a.sub(&b).unwrap();
        let unbound = c.unbind().unwrap();
        for &v in unbound[0].data().unwrap() {
            assert!((v - 7.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mul_nested() {
        let a = NestedTensor::new(&[const_tensor(2, 3, 3.0)]).unwrap();
        let b = NestedTensor::new(&[const_tensor(2, 3, 4.0)]).unwrap();

        let c = a.mul(&b).unwrap();
        let unbound = c.unbind().unwrap();
        for &v in unbound[0].data().unwrap() {
            assert!((v - 12.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_div_nested() {
        let a = NestedTensor::new(&[const_tensor(2, 3, 12.0)]).unwrap();
        let b = NestedTensor::new(&[const_tensor(2, 3, 4.0)]).unwrap();

        let c = a.div(&b).unwrap();
        let unbound = c.unbind().unwrap();
        for &v in unbound[0].data().unwrap() {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_arithmetic_shape_mismatch_error() {
        let a = NestedTensor::new(&[const_tensor(3, 4, 1.0)]).unwrap();
        let b = NestedTensor::new(&[const_tensor(5, 4, 1.0)]).unwrap();

        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_add_scalar() {
        let a = NestedTensor::new(&[seq_tensor(3, 2)]).unwrap();
        let b = a.add_scalar(100.0).unwrap();

        let orig = a.unbind().unwrap();
        let shifted = b.unbind().unwrap();

        let orig_data = orig[0].data().unwrap();
        let shifted_data = shifted[0].data().unwrap();

        for (o, s) in orig_data.iter().zip(shifted_data.iter()) {
            assert!((*s - *o - 100.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_sub_scalar() {
        let a = NestedTensor::new(&[const_tensor(2, 3, 10.0)]).unwrap();
        let b = a.sub_scalar(3.0).unwrap();

        for &v in b.unbind().unwrap()[0].data().unwrap() {
            assert!((v - 7.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_mul_scalar() {
        let a = NestedTensor::new(&[const_tensor(2, 3, 5.0)]).unwrap();
        let b = a.mul_scalar(3.0).unwrap();

        for &v in b.unbind().unwrap()[0].data().unwrap() {
            assert!((v - 15.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_div_scalar() {
        let a = NestedTensor::new(&[const_tensor(2, 3, 12.0)]).unwrap();
        let b = a.div_scalar(4.0).unwrap();

        for &v in b.unbind().unwrap()[0].data().unwrap() {
            assert!((v - 3.0).abs() < 1e-6);
        }
    }

    // -----------------------------------------------------------------------
    // Matmul
    // -----------------------------------------------------------------------

    #[test]
    fn test_matmul_basic() {
        // Sequences of shape [s_i, 3], weight matrix [3, 2]
        let t1 = const_tensor(2, 3, 1.0); // [2, 3] all ones
        let t2 = const_tensor(4, 3, 2.0); // [4, 3] all twos

        let nt = NestedTensor::new(&[t1, t2]).unwrap();

        // Weight matrix: identity-like [3, 2] = all ones
        let w = const_tensor(3, 2, 1.0);

        let result = nt.matmul(&w).unwrap();

        assert_eq!(result.num_sequences(), 2);
        assert_eq!(result.sequence_length(0), 2);
        assert_eq!(result.sequence_length(1), 4);

        let unbound = result.unbind().unwrap();
        assert_eq!(unbound[0].shape(), &[2, 2]);
        assert_eq!(unbound[1].shape(), &[4, 2]);

        // [2,3] all 1s @ [3,2] all 1s = [2,2] all 3s
        for &v in unbound[0].data().unwrap() {
            assert!((v - 3.0).abs() < 1e-6);
        }

        // [4,3] all 2s @ [3,2] all 1s = [4,2] all 6s
        for &v in unbound[1].data().unwrap() {
            assert!((v - 6.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_matmul_identity() {
        // Multiply by identity matrix
        let t = seq_tensor(3, 4); // [3, 4]
        let original = t.data().unwrap().to_vec();

        let nt = NestedTensor::new(&[t]).unwrap();

        // 4x4 identity
        let mut eye_data = vec![0.0f32; 16];
        for i in 0..4 {
            eye_data[i * 4 + i] = 1.0;
        }
        let eye = from_vec(eye_data, &[4, 4]).unwrap();

        let result = nt.matmul(&eye).unwrap();
        let unbound = result.unbind().unwrap();

        assert_eq!(unbound[0].shape(), &[3, 4]);
        let result_data = unbound[0].data().unwrap();
        for (a, b) in result_data.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_matmul_dimension_mismatch_error() {
        let nt = NestedTensor::new(&[seq_tensor(3, 4)]).unwrap();
        let w = const_tensor(5, 2, 1.0); // Inner dim 5 != 4

        assert!(nt.matmul(&w).is_err());
    }

    #[test]
    fn test_matmul_non_2d_weight_error() {
        let nt = NestedTensor::new(&[seq_tensor(3, 4)]).unwrap();
        let w = from_vec(vec![1.0; 24], &[2, 3, 4]).unwrap();

        assert!(nt.matmul(&w).is_err());
    }

    // -----------------------------------------------------------------------
    // Attention
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_uniform_input() {
        // All-ones input: attention weights should be uniform, output = value.
        let q1 = const_tensor(3, 4, 0.0); // Zero queries -> uniform attention
        let k1 = const_tensor(3, 4, 0.0);
        let v1 = const_tensor(3, 4, 1.0); // Values all ones

        let q = NestedTensor::new(&[q1]).unwrap();
        let k = NestedTensor::new(&[k1]).unwrap();
        let v = NestedTensor::new(&[v1]).unwrap();

        let out = nested_scaled_dot_product_attention(&q, &k, &v).unwrap();

        assert_eq!(out.num_sequences(), 1);
        assert_eq!(out.sequence_length(0), 3);

        let unbound = out.unbind().unwrap();
        // With zero Q and K, scores are all 0, softmax is uniform (1/3),
        // output = uniform_weights @ V = average of V rows = [1, 1, 1, 1]
        for &v in unbound[0].data().unwrap() {
            assert!(
                (v - 1.0).abs() < 1e-5,
                "expected 1.0, got {v}"
            );
        }
    }

    #[test]
    fn test_attention_variable_lengths() {
        // Two sequences of different lengths.
        let q1 = const_tensor(2, 4, 0.0);
        let q2 = const_tensor(5, 4, 0.0);
        let k1 = const_tensor(2, 4, 0.0);
        let k2 = const_tensor(5, 4, 0.0);
        let v1 = const_tensor(2, 4, 3.0);
        let v2 = const_tensor(5, 4, 7.0);

        let q = NestedTensor::new(&[q1, q2]).unwrap();
        let k = NestedTensor::new(&[k1, k2]).unwrap();
        let v = NestedTensor::new(&[v1, v2]).unwrap();

        let out = nested_scaled_dot_product_attention(&q, &k, &v).unwrap();

        assert_eq!(out.num_sequences(), 2);
        assert_eq!(out.sequence_length(0), 2);
        assert_eq!(out.sequence_length(1), 5);

        let unbound = out.unbind().unwrap();
        // Uniform attention -> output = average of value rows.
        for &v in unbound[0].data().unwrap() {
            assert!((v - 3.0).abs() < 1e-5);
        }
        for &v in unbound[1].data().unwrap() {
            assert!((v - 7.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_single_token() {
        // Single token: attention is trivially the value itself.
        let q = NestedTensor::new(&[const_tensor(1, 8, 0.5)]).unwrap();
        let k = NestedTensor::new(&[const_tensor(1, 8, 0.3)]).unwrap();
        let v = NestedTensor::new(&[const_tensor(1, 8, 42.0)]).unwrap();

        let out = nested_scaled_dot_product_attention(&q, &k, &v).unwrap();

        let unbound = out.unbind().unwrap();
        for &val in unbound[0].data().unwrap() {
            assert!((val - 42.0).abs() < 1e-5);
        }
    }

    #[test]
    fn test_attention_offsets_mismatch_error() {
        let q = NestedTensor::new(&[const_tensor(3, 4, 0.0)]).unwrap();
        let k = NestedTensor::new(&[const_tensor(5, 4, 0.0)]).unwrap();
        let v = NestedTensor::new(&[const_tensor(5, 4, 0.0)]).unwrap();

        assert!(nested_scaled_dot_product_attention(&q, &k, &v).is_err());
    }

    #[test]
    fn test_attention_dim_mismatch_error() {
        let q = NestedTensor::new(&[const_tensor(3, 4, 0.0)]).unwrap();
        let k = NestedTensor::new(&[const_tensor(3, 8, 0.0)]).unwrap(); // Different D
        let v = NestedTensor::new(&[const_tensor(3, 4, 0.0)]).unwrap();

        assert!(nested_scaled_dot_product_attention(&q, &k, &v).is_err());
    }

    #[test]
    fn test_attention_concentrated_on_one_key() {
        // If one key is very different, the query that matches it should
        // attend primarily to it.
        //
        // Q = [[10, 0], [0, 10]]  K = [[10, 0], [0, 10]]  V = [[1, 0], [0, 1]]
        // scores[0] = Q[0] @ K^T / sqrt(2) = [100, 0] / sqrt(2)
        // softmax -> nearly [1, 0] -> output[0] ~ V[0] = [1, 0]
        // scores[1] = Q[1] @ K^T / sqrt(2) = [0, 100] / sqrt(2)
        // softmax -> nearly [0, 1] -> output[1] ~ V[1] = [0, 1]
        let q = NestedTensor::new(&[from_vec(
            vec![10.0, 0.0, 0.0, 10.0],
            &[2, 2],
        )
        .unwrap()])
        .unwrap();
        let k = NestedTensor::new(&[from_vec(
            vec![10.0, 0.0, 0.0, 10.0],
            &[2, 2],
        )
        .unwrap()])
        .unwrap();
        let v = NestedTensor::new(&[from_vec(
            vec![1.0, 0.0, 0.0, 1.0],
            &[2, 2],
        )
        .unwrap()])
        .unwrap();

        let out = nested_scaled_dot_product_attention(&q, &k, &v).unwrap();
        let unbound = out.unbind().unwrap();
        let data = unbound[0].data().unwrap();

        // output[0] ~ [1.0, 0.0]
        assert!(data[0] > 0.99, "expected ~1.0, got {}", data[0]);
        assert!(data[1] < 0.01, "expected ~0.0, got {}", data[1]);

        // output[1] ~ [0.0, 1.0]
        assert!(data[2] < 0.01, "expected ~0.0, got {}", data[2]);
        assert!(data[3] > 0.99, "expected ~1.0, got {}", data[3]);
    }

    #[test]
    fn test_attention_different_value_dim() {
        // Q, K have dim 4, V has dim 6 (D_v != D_q is allowed)
        let q = NestedTensor::new(&[const_tensor(3, 4, 0.0)]).unwrap();
        let k = NestedTensor::new(&[const_tensor(3, 4, 0.0)]).unwrap();
        let v = NestedTensor::new(&[const_tensor(3, 6, 5.0)]).unwrap();

        let out = nested_scaled_dot_product_attention(&q, &k, &v).unwrap();

        assert_eq!(out.sequence_length(0), 3);
        let unbound = out.unbind().unwrap();
        assert_eq!(unbound[0].shape(), &[3, 6]);

        // Uniform attention -> output = average of value rows = [5, 5, 5, 5, 5, 5]
        for &val in unbound[0].data().unwrap() {
            assert!((val - 5.0).abs() < 1e-5);
        }
    }

    // -----------------------------------------------------------------------
    // Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_display() {
        let t1 = seq_tensor(3, 4);
        let t2 = seq_tensor(5, 4);

        let nt = NestedTensor::new(&[t1, t2]).unwrap();
        let s = format!("{nt}");

        assert!(s.contains("num_sequences=2"));
        assert!(s.contains("3, 5"));
        assert!(s.contains("total_elements=32"));
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_length_sequence() {
        // A sequence with 0 rows is valid.
        let t1 = seq_tensor(3, 4);
        let t2 = from_vec(vec![], &[0, 4]).unwrap();

        let nt = NestedTensor::new(&[t1, t2]).unwrap();

        assert_eq!(nt.num_sequences(), 2);
        assert_eq!(nt.sequence_length(0), 3);
        assert_eq!(nt.sequence_length(1), 0);
        assert_eq!(nt.total_elements(), 12);
    }

    #[test]
    fn test_to_padded_with_zero_length() {
        let t1 = const_tensor(2, 3, 5.0);
        let t2 = from_vec(vec![], &[0, 3]).unwrap();

        let nt = NestedTensor::new(&[t1, t2]).unwrap();
        let padded = nt.to_padded(-1.0).unwrap();

        // Shape: [2, 2, 3]
        assert_eq!(padded.shape(), &[2, 2, 3]);

        let data = padded.data().unwrap();
        // Seq 0: [5, 5, 5, 5, 5, 5]
        for i in 0..6 {
            assert_eq!(data[i], 5.0);
        }
        // Seq 1: all padding
        for i in 6..12 {
            assert_eq!(data[i], -1.0);
        }
    }

    #[test]
    fn test_f64_nested_tensor() {
        let t1 = crate::creation::from_vec(vec![1.0_f64, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let t2 = crate::creation::from_vec(vec![5.0_f64, 6.0, 7.0, 8.0, 9.0, 10.0], &[3, 2]).unwrap();

        let nt = NestedTensor::new(&[t1, t2]).unwrap();
        assert_eq!(nt.num_sequences(), 2);
        assert_eq!(nt.total_elements(), 10);

        let unbound = nt.unbind().unwrap();
        assert_eq!(unbound[0].data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(unbound[1].data().unwrap(), &[5.0, 6.0, 7.0, 8.0, 9.0, 10.0]);
    }

    #[test]
    fn test_large_batch_attention() {
        // 10 sequences with varying lengths.
        let lengths = [3, 7, 1, 5, 2, 8, 4, 6, 9, 3];
        let dim = 4;

        let qs: Vec<Tensor<f32>> = lengths
            .iter()
            .map(|&l| const_tensor(l, dim, 0.0))
            .collect();
        let ks: Vec<Tensor<f32>> = lengths
            .iter()
            .map(|&l| const_tensor(l, dim, 0.0))
            .collect();
        let vs: Vec<Tensor<f32>> = lengths
            .iter()
            .map(|&l| const_tensor(l, dim, 1.0))
            .collect();

        let q = NestedTensor::new(&qs).unwrap();
        let k = NestedTensor::new(&ks).unwrap();
        let v = NestedTensor::new(&vs).unwrap();

        let out = nested_scaled_dot_product_attention(&q, &k, &v).unwrap();

        assert_eq!(out.num_sequences(), 10);
        for (i, &len) in lengths.iter().enumerate() {
            assert_eq!(out.sequence_length(i), len);
        }

        // Uniform attention on all-1 values -> output should be all 1s.
        let unbound = out.unbind().unwrap();
        for (i, t) in unbound.iter().enumerate() {
            for &val in t.data().unwrap() {
                assert!(
                    (val - 1.0).abs() < 1e-5,
                    "sequence {i}: expected 1.0, got {val}"
                );
            }
        }
    }

    #[test]
    fn test_equal_lengths_matches_padded() {
        // When all sequences have the same length, nested attention should
        // produce the same result as padded attention.
        let dim = 4;
        let len = 3;

        let q_data: Vec<f32> = (0..len * dim).map(|i| (i as f32) * 0.1).collect();
        let k_data: Vec<f32> = (0..len * dim).map(|i| (i as f32) * 0.05).collect();
        let v_data: Vec<f32> = (0..len * dim).map(|i| (i as f32) * 0.2).collect();

        let qt = from_vec(q_data.clone(), &[len, dim]).unwrap();
        let kt = from_vec(k_data.clone(), &[len, dim]).unwrap();
        let vt = from_vec(v_data.clone(), &[len, dim]).unwrap();

        let q = NestedTensor::new(&[qt]).unwrap();
        let k = NestedTensor::new(&[kt]).unwrap();
        let v = NestedTensor::new(&[vt]).unwrap();

        let out = nested_scaled_dot_product_attention(&q, &k, &v).unwrap();
        let unbound = out.unbind().unwrap();
        let result = unbound[0].data().unwrap();

        // Manually compute expected attention:
        // scores[i,j] = sum_d Q[i,d]*K[j,d] / sqrt(dim)
        let scale = 1.0 / (dim as f32).sqrt();
        let mut expected = vec![0.0f32; len * dim];

        for i in 0..len {
            // Compute scores for row i
            let mut scores = vec![0.0f32; len];
            for j in 0..len {
                let mut dot = 0.0f32;
                for d in 0..dim {
                    dot += q_data[i * dim + d] * k_data[j * dim + d];
                }
                scores[j] = dot * scale;
            }

            // Softmax
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut exp_scores: Vec<f32> = scores.iter().map(|&s| (s - max_s).exp()).collect();
            let sum: f32 = exp_scores.iter().sum();
            for e in &mut exp_scores {
                *e /= sum;
            }

            // Weighted sum of values
            for d in 0..dim {
                let mut val = 0.0f32;
                for j in 0..len {
                    val += exp_scores[j] * v_data[j * dim + d];
                }
                expected[i * dim + d] = val;
            }
        }

        for (i, (r, e)) in result.iter().zip(expected.iter()).enumerate() {
            assert!(
                (r - e).abs() < 1e-4,
                "element {i}: nested result {r} != manual expected {e}"
            );
        }
    }
}
