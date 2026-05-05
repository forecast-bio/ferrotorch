use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// A nested (ragged) tensor — a collection of tensors with differing sizes
/// along one dimension (the "ragged" dimension).
///
/// This is the ferrotorch equivalent of PyTorch's `torch.nested.nested_tensor`.
/// Each component tensor may have a different size along the ragged dimension,
/// but all other dimensions must match.
///
/// # Example
///
/// A batch of sequences with different lengths:
///
/// ```text
/// NestedTensor {
///     tensors: [
///         Tensor([3, 8]),  // sequence length 3, hidden dim 8
///         Tensor([5, 8]),  // sequence length 5, hidden dim 8
///         Tensor([2, 8]),  // sequence length 2, hidden dim 8
///     ],
///     ragged_dim: 0,       // dimension 0 varies across components
/// }
/// ```
#[derive(Debug, Clone)]
pub struct NestedTensor<T: Float> {
    /// The component tensors. All must have the same number of dimensions
    /// and identical sizes on every axis except `ragged_dim`.
    tensors: Vec<Tensor<T>>,
    /// Which dimension is ragged (varies in length across components).
    ragged_dim: usize,
}

impl<T: Float> NestedTensor<T> {
    /// Create a nested tensor from a list of component tensors.
    ///
    /// All tensors must have the same number of dimensions, and identical sizes
    /// on every axis except `ragged_dim`.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `tensors` is empty
    /// - Tensors have differing numbers of dimensions
    /// - Tensors have mismatched sizes on non-ragged dimensions
    /// - `ragged_dim` is out of range
    pub fn new(tensors: Vec<Tensor<T>>, ragged_dim: usize) -> FerrotorchResult<Self> {
        if tensors.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "NestedTensor requires at least one component tensor".into(),
            });
        }

        let ndim = tensors[0].ndim();
        if ragged_dim >= ndim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ragged_dim {ragged_dim} out of range for {ndim}-D tensors"),
            });
        }

        for (i, t) in tensors.iter().enumerate().skip(1) {
            if t.ndim() != ndim {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "tensor {} has {} dims but tensor 0 has {} dims",
                        i,
                        t.ndim(),
                        ndim
                    ),
                });
            }
            for d in 0..ndim {
                if d != ragged_dim && t.shape()[d] != tensors[0].shape()[d] {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "tensor {} has size {} on dim {} but tensor 0 has size {} \
                             (only dim {} may differ)",
                            i,
                            t.shape()[d],
                            d,
                            tensors[0].shape()[d],
                            ragged_dim,
                        ),
                    });
                }
            }
        }

        Ok(Self {
            tensors,
            ragged_dim,
        })
    }

    /// Number of component tensors.
    #[inline]
    pub fn num_components(&self) -> usize {
        self.tensors.len()
    }

    /// The ragged dimension index.
    #[inline]
    pub fn ragged_dim(&self) -> usize {
        self.ragged_dim
    }

    /// References to the component tensors.
    #[inline]
    pub fn tensors(&self) -> &[Tensor<T>] {
        &self.tensors
    }

    /// Number of dimensions of each component tensor.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.tensors[0].ndim()
    }

    /// The size of non-ragged dimensions (taken from the first component).
    pub fn consistent_shape(&self) -> Vec<usize> {
        self.tensors[0].shape().to_vec()
    }

    /// The lengths along the ragged dimension for each component.
    pub fn ragged_lengths(&self) -> Vec<usize> {
        self.tensors
            .iter()
            .map(|t| t.shape()[self.ragged_dim])
            .collect()
    }

    /// Convert to a padded dense tensor.
    ///
    /// Pads each component along the ragged dimension to the maximum length,
    /// filling missing positions with `pad_value`. The result has an extra
    /// leading batch dimension.
    ///
    /// For a nested tensor with components of shape `[L_i, D]` and
    /// `ragged_dim=0`, the output is `[batch, max_L, D]`.
    pub fn to_padded(&self, pad_value: T) -> FerrotorchResult<Tensor<T>> {
        let batch = self.tensors.len();
        let ndim = self.ndim();
        let max_len = self
            .tensors
            .iter()
            .map(|t| t.shape()[self.ragged_dim])
            .max()
            .unwrap_or(0);

        // Build output shape: [batch, d0, d1, ..., d_{ndim-1}] where
        // d_{ragged_dim} = max_len.
        let mut out_shape = Vec::with_capacity(ndim + 1);
        out_shape.push(batch);
        for d in 0..ndim {
            if d == self.ragged_dim {
                out_shape.push(max_len);
            } else {
                out_shape.push(self.tensors[0].shape()[d]);
            }
        }

        let numel: usize = out_shape.iter().product();
        let mut data = vec![pad_value; numel];

        // Compute strides for the output tensor (row-major).
        let mut out_strides = vec![0usize; ndim + 1];
        out_strides[ndim] = 1;
        for d in (0..ndim).rev() {
            out_strides[d] = out_strides[d + 1] * out_shape[d + 1];
        }

        for (b, t) in self.tensors.iter().enumerate() {
            let t_data = t.data()?;
            let t_shape = t.shape();

            // Compute strides for this component tensor (row-major).
            let mut t_strides = vec![0usize; ndim];
            if ndim > 0 {
                t_strides[ndim - 1] = 1;
                for d in (0..ndim - 1).rev() {
                    t_strides[d] = t_strides[d + 1] * t_shape[d + 1];
                }
            }

            let t_numel: usize = t_shape.iter().product();
            for (flat, &val) in t_data.iter().enumerate().take(t_numel) {
                // Convert flat index to multi-dim coords in the component.
                let mut remaining = flat;
                let mut out_flat = b * out_strides[0];
                for d in 0..ndim {
                    let coord = remaining / t_strides[d];
                    remaining %= t_strides[d];
                    out_flat += coord * out_strides[d + 1];
                }
                data[out_flat] = val;
            }
        }

        Tensor::from_storage(TensorStorage::cpu(data), out_shape, false)
    }

    /// Reconstruct a nested tensor from a padded dense tensor and per-component
    /// lengths along the ragged dimension.
    ///
    /// This is the inverse of [`to_padded`](Self::to_padded). The first
    /// dimension of `tensor` is the batch dimension.
    ///
    /// # Arguments
    ///
    /// * `tensor` - Padded tensor with shape `[batch, d0, d1, ..., d_{ndim-1}]`.
    /// * `lengths` - Length of each component along the ragged dimension.
    ///   Must have length equal to the batch dimension.
    /// * `ragged_dim` - Which dimension (in the component tensors) is ragged.
    pub fn from_padded(
        tensor: &Tensor<T>,
        lengths: &[usize],
        ragged_dim: usize,
    ) -> FerrotorchResult<Self> {
        let full_shape = tensor.shape();
        if full_shape.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "from_padded requires at least a batch dimension".into(),
            });
        }

        let batch = full_shape[0];
        if lengths.len() != batch {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "lengths has {} entries but batch dimension is {}",
                    lengths.len(),
                    batch
                ),
            });
        }

        let comp_ndim = full_shape.len() - 1; // number of dims in each component
        if ragged_dim >= comp_ndim {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ragged_dim {ragged_dim} out of range for {comp_ndim}-D component tensors"
                ),
            });
        }

        let padded_data = tensor.data()?;

        // Strides for the full padded tensor (row-major).
        let full_ndim = full_shape.len();
        let mut full_strides = vec![0usize; full_ndim];
        if full_ndim > 0 {
            full_strides[full_ndim - 1] = 1;
            for d in (0..full_ndim - 1).rev() {
                full_strides[d] = full_strides[d + 1] * full_shape[d + 1];
            }
        }

        let mut tensors = Vec::with_capacity(batch);
        for (b, &len_b) in lengths.iter().enumerate().take(batch) {
            // Build component shape: same as full_shape[1..] but with
            // ragged_dim replaced by len_b.
            let mut comp_shape = Vec::with_capacity(comp_ndim);
            for d in 0..comp_ndim {
                if d == ragged_dim {
                    comp_shape.push(len_b);
                } else {
                    comp_shape.push(full_shape[d + 1]);
                }
            }

            // Compute strides for the component (row-major).
            let mut comp_strides = vec![0usize; comp_ndim];
            if comp_ndim > 0 {
                comp_strides[comp_ndim - 1] = 1;
                for d in (0..comp_ndim - 1).rev() {
                    comp_strides[d] = comp_strides[d + 1] * comp_shape[d + 1];
                }
            }

            let comp_numel: usize = comp_shape.iter().product();
            let mut comp_data = Vec::with_capacity(comp_numel);

            for flat in 0..comp_numel {
                // Convert flat index to multi-dim coords in the component.
                let mut remaining = flat;
                let mut full_flat = b * full_strides[0];
                for d in 0..comp_ndim {
                    let coord = remaining.checked_div(comp_strides[d]).unwrap_or(0);
                    if comp_strides[d] > 0 {
                        remaining %= comp_strides[d];
                    }
                    full_flat += coord * full_strides[d + 1];
                }
                comp_data.push(padded_data[full_flat]);
            }

            tensors.push(Tensor::from_storage(
                TensorStorage::cpu(comp_data),
                comp_shape,
                false,
            )?);
        }

        Self::new(tensors, ragged_dim)
    }
}

// --- Attention ---

/// Row-wise softmax in place.
///
/// Each row of `data` (of width `cols`) is independently softmax'd.
/// For numerical stability, the maximum value is subtracted before
/// exponentiation.
///
/// When all values in a row are `-inf` (producing a sum of zero after
/// exponentiation), the row is filled with NaN to match PyTorch semantics.
fn softmax_rows_inplace<T: Float>(data: &mut [T], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut data[r * cols..(r + 1) * cols];

        // Numerical stability: subtract row max.
        let max_val = row
            .iter()
            .copied()
            .fold(<T as num_traits::Float>::neg_infinity(), |a, b| {
                if b > a { b } else { a }
            });

        let mut sum = <T as num_traits::Zero>::zero();
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }

        if sum == <T as num_traits::Zero>::zero() {
            // All inputs were -inf; produce NaN to match PyTorch.
            for val in row.iter_mut() {
                *val = <T as num_traits::Float>::nan();
            }
        } else {
            for val in row.iter_mut() {
                *val = *val / sum;
            }
        }
    }
}

/// Scaled dot-product attention over nested tensors.
///
/// Implements the standard multi-head attention formula:
///
/// ```text
/// Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
/// ```
///
/// Each component in the nested tensor is processed independently (they may
/// have different sequence lengths). This is generic over `T: Float` so it
/// works with both f32 and f64.
///
/// # Arguments
///
/// * `query` - Nested tensor of shape `[seq_q, d_k]` per component.
/// * `key` - Nested tensor of shape `[seq_k, d_k]` per component.
/// * `value` - Nested tensor of shape `[seq_k, d_v]` per component.
///
/// # Returns
///
/// A nested tensor of shape `[seq_q, d_v]` per component.
pub fn nested_scaled_dot_product_attention<T: Float>(
    query: &NestedTensor<T>,
    key: &NestedTensor<T>,
    value: &NestedTensor<T>,
) -> FerrotorchResult<NestedTensor<T>> {
    let n = query.num_components();
    if key.num_components() != n || value.num_components() != n {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "query has {} components but key has {} and value has {}",
                n,
                key.num_components(),
                value.num_components()
            ),
        });
    }

    let mut outputs = Vec::with_capacity(n);

    for i in 0..n {
        let q = &query.tensors()[i];
        let k = &key.tensors()[i];
        let v = &value.tensors()[i];

        if q.ndim() != 2 || k.ndim() != 2 || v.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "attention requires 2-D tensors, component {} has dims ({}, {}, {})",
                    i,
                    q.ndim(),
                    k.ndim(),
                    v.ndim()
                ),
            });
        }

        let seq_q = q.shape()[0];
        let d_k = q.shape()[1];
        let seq_k = k.shape()[0];
        let d_k2 = k.shape()[1];
        let seq_k2 = v.shape()[0];
        let d_v = v.shape()[1];

        if d_k != d_k2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("component {i}: query d_k={d_k} but key d_k={d_k2}"),
            });
        }
        if seq_k != seq_k2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!("component {i}: key seq_len={seq_k} but value seq_len={seq_k2}"),
            });
        }

        let q_data = q.data()?;
        let k_data = k.data()?;
        let v_data = v.data()?;

        let scale = T::from(d_k).unwrap().sqrt().recip();

        // Compute Q @ K^T: [seq_q, seq_k]
        let mut scores = vec![<T as num_traits::Zero>::zero(); seq_q * seq_k];
        for qi in 0..seq_q {
            for ki in 0..seq_k {
                let mut dot = <T as num_traits::Zero>::zero();
                for di in 0..d_k {
                    dot += q_data[qi * d_k + di] * k_data[ki * d_k + di];
                }
                scores[qi * seq_k + ki] = dot * scale;
            }
        }

        // Softmax over each row.
        softmax_rows_inplace(&mut scores, seq_q, seq_k);

        // Multiply by V: [seq_q, d_v]
        let mut out = vec![<T as num_traits::Zero>::zero(); seq_q * d_v];
        for qi in 0..seq_q {
            for dvi in 0..d_v {
                let mut acc = <T as num_traits::Zero>::zero();
                for ki in 0..seq_k {
                    acc += scores[qi * seq_k + ki] * v_data[ki * d_v + dvi];
                }
                out[qi * d_v + dvi] = acc;
            }
        }

        outputs.push(Tensor::from_storage(
            TensorStorage::cpu(out),
            vec![seq_q, d_v],
            false,
        )?);
    }

    NestedTensor::new(outputs, query.ragged_dim())
}

// ===========================================================================
// PackedNestedTensor — packed flat storage + offsets layout. CL-291.
// ===========================================================================

/// A nested (jagged) tensor stored as **one contiguous flat buffer**
/// with an offsets array marking the start of each component.
///
/// This is the efficient storage layout for nested tensors: bulk
/// elementwise ops operate on the whole flat buffer at once without
/// touching the offsets, per-sequence reductions walk the offsets
/// once, and conversion to/from padded dense tensors uses a single
/// linear scan. The companion [`NestedTensor`] list-of-tensors layout
/// is better for ergonomic per-component access; choose based on the
/// workload.
///
/// Every component has the same shape on the **tail** dimensions
/// (everything except `ragged_dim`), and the ragged dimension is
/// always the **leading** dim (dim 0) within each component. That
/// restriction keeps the offsets 1-D and the flat layout unambiguous.
///
/// # Example
///
/// ```ignore
/// // Batch of three sequences with lengths 3, 5, 2 and hidden dim 4.
/// let seqs = vec![
///     vec![1.0, 2.0, 3.0, 4.0,
///          5.0, 6.0, 7.0, 8.0,
///          9.0, 10.0, 11.0, 12.0],    // len=3 → 3*4 = 12 values
///     // ... sequence 1
///     // ... sequence 2
/// ];
/// let lengths = vec![3, 5, 2];
/// let tail_shape = vec![4];
/// let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &tail_shape)?;
/// assert_eq!(pnt.num_components(), 3);
/// assert_eq!(pnt.offsets(), &[0, 12, 32, 40]);
/// ```
///
/// # Layout invariants
///
/// - `offsets.len() == num_components + 1`
/// - `offsets[0] == 0`
/// - `offsets[i+1] - offsets[i] == lengths[i] * tail_numel`
/// - `offsets[num_components] == data.len()`
///
/// where `tail_numel = product(tail_shape)`.
#[derive(Debug, Clone)]
pub struct PackedNestedTensor<T: Float> {
    /// Flat concatenation of every component's data, in component
    /// order. For a nested tensor with components of shape
    /// `[L_i] + tail_shape`, component `i` occupies the slice
    /// `data[offsets[i] .. offsets[i+1]]`.
    data: Vec<T>,
    /// Length-`num_components + 1` offsets array. `offsets[i]` is
    /// the start of component `i` in `data`.
    offsets: Vec<usize>,
    /// Shape of each component's tail (everything after the ragged
    /// dim). For a 1-D ragged sequence the tail is empty; for
    /// `[L, D]` components with ragged dim 0, the tail is `[D]`.
    tail_shape: Vec<usize>,
}

impl<T: Float> PackedNestedTensor<T> {
    /// Create a packed nested tensor from a list of per-component
    /// flat data buffers, their ragged-dim lengths, and the shared
    /// tail shape.
    ///
    /// `sequences[i]` must contain exactly `lengths[i] * tail_numel`
    /// values in row-major order (outer dim ragged, tail dims
    /// c-contiguous).
    ///
    /// # Errors
    ///
    /// - `sequences.len() != lengths.len()`
    /// - Any `sequences[i].len() != lengths[i] * tail_numel`
    /// - Empty input (`sequences.is_empty()`)
    pub fn from_sequences(
        sequences: Vec<Vec<T>>,
        lengths: &[usize],
        tail_shape: &[usize],
    ) -> FerrotorchResult<Self> {
        if sequences.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "PackedNestedTensor requires at least one sequence".into(),
            });
        }
        if sequences.len() != lengths.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor: sequences has {} entries but lengths has {}",
                    sequences.len(),
                    lengths.len()
                ),
            });
        }
        let tail_numel: usize = tail_shape.iter().product::<usize>().max(1);

        let mut total = 0usize;
        for (i, seq) in sequences.iter().enumerate() {
            let expected = lengths[i].checked_mul(tail_numel).ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!(
                        "PackedNestedTensor: length overflow in component {i} \
                         (length={}, tail_numel={tail_numel})",
                        lengths[i]
                    ),
                }
            })?;
            if seq.len() != expected {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "PackedNestedTensor: sequence {i} has {} elements but expected \
                         lengths[{i}] * tail_numel = {}*{} = {}",
                        seq.len(),
                        lengths[i],
                        tail_numel,
                        expected
                    ),
                });
            }
            total += expected;
        }

        let mut data = Vec::with_capacity(total);
        let mut offsets = Vec::with_capacity(sequences.len() + 1);
        offsets.push(0);
        for seq in sequences {
            data.extend(seq);
            offsets.push(data.len());
        }

        Ok(Self {
            data,
            offsets,
            tail_shape: tail_shape.to_vec(),
        })
    }

    /// Create a packed nested tensor from the component tensors of
    /// a [`NestedTensor`]. `ragged_dim` must be 0 — the packed
    /// layout requires the ragged dim to lead.
    ///
    /// # Errors
    ///
    /// - `nested.ragged_dim() != 0`
    /// - `nested.num_components() == 0`
    pub fn from_nested(nested: &NestedTensor<T>) -> FerrotorchResult<Self> {
        if nested.ragged_dim() != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::from_nested requires ragged_dim == 0, got {}",
                    nested.ragged_dim()
                ),
            });
        }
        let comps = nested.tensors();
        if comps.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "PackedNestedTensor::from_nested: no components".into(),
            });
        }
        // Tail shape = shape without the ragged (leading) dim.
        let tail_shape: Vec<usize> = comps[0].shape()[1..].to_vec();
        let lengths: Vec<usize> = comps.iter().map(|t| t.shape()[0]).collect();

        let mut sequences: Vec<Vec<T>> = Vec::with_capacity(comps.len());
        for t in comps {
            sequences.push(t.data()?.to_vec());
        }
        Self::from_sequences(sequences, &lengths, &tail_shape)
    }

    /// Convert back into a component-list [`NestedTensor`] (ragged
    /// dim 0). Each component becomes a fresh [`Tensor<T>`] holding
    /// its slice of the packed data.
    pub fn to_nested(&self) -> FerrotorchResult<NestedTensor<T>> {
        let n = self.num_components();
        let mut tensors = Vec::with_capacity(n);
        for i in 0..n {
            let len = self.length(i);
            let mut shape = vec![len];
            shape.extend_from_slice(&self.tail_shape);
            let slice = self.component_slice(i).to_vec();
            tensors.push(Tensor::from_storage(
                TensorStorage::cpu(slice),
                shape,
                false,
            )?);
        }
        NestedTensor::new(tensors, 0)
    }

    /// Number of component sequences.
    #[inline]
    pub fn num_components(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// The `offsets` array. Always `num_components + 1` long.
    #[inline]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Shared tail shape (non-ragged dims).
    #[inline]
    pub fn tail_shape(&self) -> &[usize] {
        &self.tail_shape
    }

    /// The flat packed data buffer.
    #[inline]
    pub fn data(&self) -> &[T] {
        &self.data
    }

    /// Length (along the ragged dim) of component `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_components()`.
    #[inline]
    pub fn length(&self, i: usize) -> usize {
        let tail_numel: usize = self.tail_shape.iter().product::<usize>().max(1);
        (self.offsets[i + 1] - self.offsets[i]) / tail_numel
    }

    /// Total number of elements in the packed buffer (sum of every
    /// component's element count).
    #[inline]
    pub fn total_numel(&self) -> usize {
        self.data.len()
    }

    /// Borrow the slice of the packed data holding component `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= num_components()`.
    pub fn component_slice(&self, i: usize) -> &[T] {
        &self.data[self.offsets[i]..self.offsets[i + 1]]
    }

    /// Elementwise map that applies `f` to every value in the
    /// packed buffer and returns a new `PackedNestedTensor` with
    /// the same offsets and tail shape.
    ///
    /// This is the workhorse for implementing nested-level
    /// elementwise ops (relu, neg, abs, etc.) without writing per-
    /// component loops. CL-291.
    pub fn map(&self, f: impl Fn(T) -> T) -> Self {
        let data: Vec<T> = self.data.iter().copied().map(f).collect();
        Self {
            data,
            offsets: self.offsets.clone(),
            tail_shape: self.tail_shape.clone(),
        }
    }

    /// Elementwise addition of two packed nested tensors. Both must
    /// have identical offsets and tail shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the offsets or tail shapes don't match.
    pub fn add(&self, other: &Self) -> FerrotorchResult<Self> {
        self.zip_with(other, "add", |a, b| a + b)
    }

    /// Elementwise subtraction.
    pub fn sub(&self, other: &Self) -> FerrotorchResult<Self> {
        self.zip_with(other, "sub", |a, b| a - b)
    }

    /// Elementwise multiplication.
    pub fn mul(&self, other: &Self) -> FerrotorchResult<Self> {
        self.zip_with(other, "mul", |a, b| a * b)
    }

    /// Elementwise division.
    pub fn div(&self, other: &Self) -> FerrotorchResult<Self> {
        self.zip_with(other, "div", |a, b| a / b)
    }

    /// Shared implementation for elementwise binary ops. Validates
    /// that `self` and `other` have matching layouts before
    /// applying `f` over the packed data.
    fn zip_with(
        &self,
        other: &Self,
        op_name: &'static str,
        f: impl Fn(T, T) -> T,
    ) -> FerrotorchResult<Self> {
        if self.offsets != other.offsets {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::{op_name}: offsets mismatch \
                     ({:?} vs {:?})",
                    self.offsets, other.offsets
                ),
            });
        }
        if self.tail_shape != other.tail_shape {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::{op_name}: tail shape mismatch \
                     ({:?} vs {:?})",
                    self.tail_shape, other.tail_shape
                ),
            });
        }
        let data: Vec<T> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(&a, &b)| f(a, b))
            .collect();
        Ok(Self {
            data,
            offsets: self.offsets.clone(),
            tail_shape: self.tail_shape.clone(),
        })
    }

    /// Per-component sum of every element. Returns a 1-D vec with
    /// `num_components` entries. Tail dims are summed into a single
    /// scalar per component. CL-291.
    pub fn sum_per_component(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.num_components());
        for i in 0..self.num_components() {
            let slice = self.component_slice(i);
            let mut acc = <T as num_traits::Zero>::zero();
            for &v in slice {
                acc += v;
            }
            out.push(acc);
        }
        out
    }

    /// Per-component mean of every element. Returns zero for
    /// empty components (length 0) rather than NaN. CL-291.
    pub fn mean_per_component(&self) -> Vec<T> {
        let mut out = Vec::with_capacity(self.num_components());
        for i in 0..self.num_components() {
            let slice = self.component_slice(i);
            if slice.is_empty() {
                out.push(<T as num_traits::Zero>::zero());
                continue;
            }
            let mut acc = <T as num_traits::Zero>::zero();
            for &v in slice {
                acc += v;
            }
            let n = T::from(slice.len()).unwrap();
            out.push(acc / n);
        }
        out
    }

    /// Convert to a padded dense tensor. The output shape is
    /// `[num_components, max_length] + tail_shape`; positions
    /// beyond each component's ragged length are filled with
    /// `pad_value`. CL-291.
    pub fn to_padded(&self, pad_value: T) -> FerrotorchResult<Tensor<T>> {
        let n = self.num_components();
        let mut max_len = 0usize;
        for i in 0..n {
            max_len = max_len.max(self.length(i));
        }
        let tail_numel: usize = self.tail_shape.iter().product::<usize>().max(1);
        let row_stride = max_len * tail_numel;

        let mut out = vec![pad_value; n * row_stride];
        for i in 0..n {
            let dst_base = i * row_stride;
            let slice = self.component_slice(i);
            out[dst_base..dst_base + slice.len()].copy_from_slice(slice);
        }

        let mut shape = vec![n, max_len];
        shape.extend_from_slice(&self.tail_shape);
        Tensor::from_storage(TensorStorage::cpu(out), shape, false)
    }

    /// Reconstruct a packed nested tensor from a padded dense
    /// tensor + per-component lengths. Inverse of [`to_padded`].
    ///
    /// # Errors
    ///
    /// - `tensor.ndim() < 2` — must have at least batch and
    ///   sequence dims.
    /// - `lengths.len() != tensor.shape()[0]`
    /// - Any `lengths[i] > tensor.shape()[1]` (would walk off the
    ///   padded row).
    pub fn from_padded(tensor: &Tensor<T>, lengths: &[usize]) -> FerrotorchResult<Self> {
        let shape = tensor.shape();
        if shape.len() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::from_padded: tensor must have at least \
                     2 dims (batch, sequence), got {shape:?}"
                ),
            });
        }
        let n = shape[0];
        let max_len = shape[1];
        let tail_shape: Vec<usize> = shape[2..].to_vec();
        let tail_numel: usize = tail_shape.iter().product::<usize>().max(1);

        if lengths.len() != n {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PackedNestedTensor::from_padded: lengths has {} entries but \
                     batch dim is {}",
                    lengths.len(),
                    n
                ),
            });
        }
        for (i, &len) in lengths.iter().enumerate() {
            if len > max_len {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "PackedNestedTensor::from_padded: lengths[{i}] = {len} \
                         exceeds max_len = {max_len}"
                    ),
                });
            }
        }

        let padded = tensor.data()?;
        let row_stride = max_len * tail_numel;

        let mut data = Vec::with_capacity(lengths.iter().sum::<usize>() * tail_numel);
        let mut offsets = Vec::with_capacity(n + 1);
        offsets.push(0);
        for (i, &len) in lengths.iter().enumerate() {
            let src_base = i * row_stride;
            let src_end = src_base + len * tail_numel;
            data.extend_from_slice(&padded[src_base..src_end]);
            offsets.push(data.len());
        }

        Ok(Self {
            data,
            offsets,
            tail_shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    fn make_tensor_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    // --- NestedTensor construction ---

    #[test]
    fn test_nested_construction() {
        let t1 = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let t2 = make_tensor(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]);

        let nt = NestedTensor::new(vec![t1, t2], 0).unwrap();

        assert_eq!(nt.num_components(), 2);
        assert_eq!(nt.ragged_dim(), 0);
        assert_eq!(nt.ndim(), 2);
        assert_eq!(nt.ragged_lengths(), vec![3, 2]);
    }

    #[test]
    fn test_nested_rejects_empty() {
        let result = NestedTensor::<f32>::new(vec![], 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_nested_rejects_shape_mismatch() {
        let t1 = make_tensor(vec![1.0; 6], vec![3, 2]);
        let t2 = make_tensor(vec![1.0; 6], vec![2, 3]); // dim 1 differs

        let result = NestedTensor::new(vec![t1, t2], 0);
        assert!(result.is_err());
    }

    // --- to_padded / from_padded round-trip (ragged_dim=0) ---

    #[test]
    fn test_to_padded_from_padded_ragged_dim_0() {
        let t1 = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);
        let t2 = make_tensor(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]);

        let nt = NestedTensor::new(vec![t1, t2], 0).unwrap();
        let padded = nt.to_padded(0.0).unwrap();

        assert_eq!(padded.shape(), &[2, 3, 2]); // batch=2, max_len=3, d=2

        let lengths = nt.ragged_lengths();
        let reconstructed = NestedTensor::from_padded(&padded, &lengths, 0).unwrap();

        assert_eq!(reconstructed.num_components(), 2);

        let r0 = reconstructed.tensors()[0].data().unwrap();
        assert_eq!(r0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let r1 = reconstructed.tensors()[1].data().unwrap();
        assert_eq!(r1, &[7.0, 8.0, 9.0, 10.0]);
    }

    // --- from_padded round-trip for ragged_dim != 0 ---

    #[test]
    // reason: to_padded copies component values verbatim and writes the literal
    // pad value (0.0) into pad slots — no float arithmetic, so bitwise equality
    // with the source literals is the correct assertion.
    #[allow(clippy::float_cmp)]
    fn test_from_padded_round_trip_ragged_dim_1() {
        // Component tensors: shape [2, L_i] where dim 1 is ragged.
        // t1: [2, 3], t2: [2, 2]
        let t1 = make_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let t2 = make_tensor(vec![7.0, 8.0, 9.0, 10.0], vec![2, 2]);

        let nt = NestedTensor::new(vec![t1, t2], 1).unwrap();

        // to_padded: output shape [batch=2, 2, max_col=3]
        let padded = nt.to_padded(0.0).unwrap();
        assert_eq!(padded.shape(), &[2, 2, 3]);

        let padded_data = padded.data().unwrap();
        // batch 0: [[1,2,3],[4,5,6]]
        assert_eq!(padded_data[0], 1.0);
        assert_eq!(padded_data[1], 2.0);
        assert_eq!(padded_data[2], 3.0);
        assert_eq!(padded_data[3], 4.0);
        assert_eq!(padded_data[4], 5.0);
        assert_eq!(padded_data[5], 6.0);
        // batch 1: [[7,8,0],[9,10,0]]
        assert_eq!(padded_data[6], 7.0);
        assert_eq!(padded_data[7], 8.0);
        assert_eq!(padded_data[8], 0.0); // pad
        assert_eq!(padded_data[9], 9.0);
        assert_eq!(padded_data[10], 10.0);
        assert_eq!(padded_data[11], 0.0); // pad

        // Reconstruct from padded.
        let lengths = nt.ragged_lengths();
        assert_eq!(lengths, vec![3, 2]);
        let reconstructed = NestedTensor::from_padded(&padded, &lengths, 1).unwrap();

        assert_eq!(reconstructed.num_components(), 2);
        assert_eq!(reconstructed.tensors()[0].shape(), &[2, 3]);
        assert_eq!(reconstructed.tensors()[1].shape(), &[2, 2]);

        let r0 = reconstructed.tensors()[0].data().unwrap();
        assert_eq!(r0, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let r1 = reconstructed.tensors()[1].data().unwrap();
        assert_eq!(r1, &[7.0, 8.0, 9.0, 10.0]);
    }

    // --- scaled dot-product attention ---

    #[test]
    fn test_sdpa_basic() {
        // Single component: Q=[2,4], K=[3,4], V=[3,5]
        let q = make_tensor(vec![1.0; 8], vec![2, 4]);
        let k = make_tensor(vec![1.0; 12], vec![3, 4]);
        let v = make_tensor(vec![1.0; 15], vec![3, 5]);

        let qn = NestedTensor::new(vec![q], 0).unwrap();
        let kn = NestedTensor::new(vec![k], 0).unwrap();
        let vn = NestedTensor::new(vec![v], 0).unwrap();

        let result = nested_scaled_dot_product_attention(&qn, &kn, &vn).unwrap();

        assert_eq!(result.num_components(), 1);
        assert_eq!(result.tensors()[0].shape(), &[2, 5]);

        // With uniform values, softmax should produce uniform weights,
        // and the output should be close to 1.0 everywhere.
        let out = result.tensors()[0].data().unwrap();
        for &val in out {
            assert!((val - 1.0).abs() < 1e-5, "expected ~1.0, got {val}");
        }
    }

    #[test]
    fn test_sdpa_f64() {
        // Verify it works with f64.
        let q = make_tensor_f64(vec![1.0; 8], vec![2, 4]);
        let k = make_tensor_f64(vec![1.0; 12], vec![3, 4]);
        let v = make_tensor_f64(vec![1.0; 15], vec![3, 5]);

        let qn = NestedTensor::new(vec![q], 0).unwrap();
        let kn = NestedTensor::new(vec![k], 0).unwrap();
        let vn = NestedTensor::new(vec![v], 0).unwrap();

        let result = nested_scaled_dot_product_attention(&qn, &kn, &vn).unwrap();

        assert_eq!(result.num_components(), 1);
        assert_eq!(result.tensors()[0].shape(), &[2, 5]);
    }

    // --- softmax degenerate case: all -inf -> NaN ---

    #[test]
    fn test_softmax_all_neg_inf_produces_nan() {
        let mut data = vec![f32::NEG_INFINITY; 6];
        softmax_rows_inplace(&mut data, 2, 3);

        for val in &data {
            assert!(val.is_nan(), "expected NaN for all -inf input, got {val}");
        }
    }

    #[test]
    fn test_softmax_normal_case() {
        let mut data = vec![1.0f32, 2.0, 3.0];
        softmax_rows_inplace(&mut data, 1, 3);

        let sum: f32 = data.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "softmax should sum to 1, got {sum}"
        );
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
    }

    // ────────────────────────────────────────────────────────────────
    // CL-291: PackedNestedTensor tests
    // ────────────────────────────────────────────────────────────────

    #[test]
    fn packed_from_sequences_1d() {
        // Three 1-D sequences with empty tail shape.
        let seqs = vec![
            vec![1.0f32, 2.0, 3.0],
            vec![4.0, 5.0, 6.0, 7.0, 8.0],
            vec![9.0, 10.0],
        ];
        let lengths = vec![3usize, 5, 2];
        let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &[]).unwrap();

        assert_eq!(pnt.num_components(), 3);
        assert_eq!(pnt.offsets(), &[0, 3, 8, 10]);
        assert_eq!(pnt.total_numel(), 10);
        assert_eq!(pnt.length(0), 3);
        assert_eq!(pnt.length(1), 5);
        assert_eq!(pnt.length(2), 2);
        assert_eq!(pnt.component_slice(0), &[1.0, 2.0, 3.0]);
        assert_eq!(pnt.component_slice(1), &[4.0, 5.0, 6.0, 7.0, 8.0]);
        assert_eq!(pnt.component_slice(2), &[9.0, 10.0]);
    }

    #[test]
    fn packed_from_sequences_with_tail_shape() {
        // Two 2-D sequences, ragged on dim 0, tail shape [4].
        let seqs = vec![
            vec![
                1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ], // len=3
            vec![13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0], // len=2
        ];
        let lengths = vec![3usize, 2];
        let tail = vec![4usize];
        let pnt = PackedNestedTensor::from_sequences(seqs, &lengths, &tail).unwrap();

        assert_eq!(pnt.num_components(), 2);
        assert_eq!(pnt.offsets(), &[0, 12, 20]);
        assert_eq!(pnt.length(0), 3);
        assert_eq!(pnt.length(1), 2);
        assert_eq!(pnt.tail_shape(), &[4]);
    }

    #[test]
    fn packed_rejects_empty_sequences_list() {
        let result = PackedNestedTensor::<f32>::from_sequences(vec![], &[], &[]);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("at least one sequence"));
    }

    #[test]
    fn packed_rejects_mismatched_sequence_length() {
        let seqs = vec![vec![1.0f32, 2.0, 3.0]]; // 3 elements
        let lengths = vec![2usize]; // expects 2*tail_numel=2
        let result = PackedNestedTensor::from_sequences(seqs, &lengths, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn packed_rejects_mismatched_sequences_vs_lengths() {
        let seqs = vec![vec![1.0f32, 2.0]];
        let lengths = vec![2usize, 3];
        let result = PackedNestedTensor::from_sequences(seqs, &lengths, &[]);
        assert!(result.is_err());
    }

    #[test]
    fn packed_map_applies_fn_to_every_element() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![vec![1.0f32, -2.0, 3.0], vec![-4.0, 5.0]],
            &[3usize, 2],
            &[],
        )
        .unwrap();
        // ReLU via map.
        let relu = pnt.map(|x: f32| x.max(0.0));
        assert_eq!(relu.data(), &[1.0, 0.0, 3.0, 0.0, 5.0]);
        // Offsets preserved.
        assert_eq!(relu.offsets(), pnt.offsets());
    }

    #[test]
    fn packed_add_sub_mul_div() {
        let a = PackedNestedTensor::from_sequences(
            vec![vec![10.0f32, 20.0, 30.0], vec![40.0, 50.0]],
            &[3usize, 2],
            &[],
        )
        .unwrap();
        let b = PackedNestedTensor::from_sequences(
            vec![vec![1.0f32, 2.0, 3.0], vec![4.0, 5.0]],
            &[3usize, 2],
            &[],
        )
        .unwrap();

        assert_eq!(a.add(&b).unwrap().data(), &[11.0, 22.0, 33.0, 44.0, 55.0]);
        assert_eq!(a.sub(&b).unwrap().data(), &[9.0, 18.0, 27.0, 36.0, 45.0]);
        assert_eq!(a.mul(&b).unwrap().data(), &[10.0, 40.0, 90.0, 160.0, 250.0]);
        assert_eq!(a.div(&b).unwrap().data(), &[10.0, 10.0, 10.0, 10.0, 10.0]);
    }

    #[test]
    fn packed_add_rejects_mismatched_offsets() {
        let a = PackedNestedTensor::from_sequences(vec![vec![1.0f32, 2.0, 3.0]], &[3usize], &[])
            .unwrap();
        let b =
            PackedNestedTensor::from_sequences(vec![vec![1.0f32, 2.0]], &[2usize], &[]).unwrap();
        let result = a.add(&b);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("offsets mismatch"));
    }

    #[test]
    fn packed_add_rejects_mismatched_tail_shape() {
        let a = PackedNestedTensor::from_sequences(
            vec![vec![1.0f32, 2.0, 3.0, 4.0]],
            &[2usize],
            &[2], // tail [2]
        )
        .unwrap();
        let b = PackedNestedTensor::from_sequences(
            vec![vec![1.0f32, 2.0, 3.0, 4.0]],
            &[4usize],
            &[], // tail []
        )
        .unwrap();
        // Offsets match (both [0, 4]) but tail shapes differ.
        let result = a.add(&b);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("tail shape mismatch"));
    }

    #[test]
    fn packed_sum_per_component() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![
                vec![1.0f32, 2.0, 3.0],       // sum = 6
                vec![10.0, 20.0, 30.0, 40.0], // sum = 100
                vec![5.0],                    // sum = 5
            ],
            &[3usize, 4, 1],
            &[],
        )
        .unwrap();
        let sums = pnt.sum_per_component();
        assert_eq!(sums, vec![6.0, 100.0, 5.0]);
    }

    #[test]
    fn packed_mean_per_component() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![
                vec![2.0f32, 4.0, 6.0],       // mean = 4
                vec![10.0, 20.0, 30.0, 40.0], // mean = 25
                vec![7.0],                    // mean = 7
            ],
            &[3usize, 4, 1],
            &[],
        )
        .unwrap();
        let means = pnt.mean_per_component();
        assert_eq!(means, vec![4.0, 25.0, 7.0]);
    }

    #[test]
    fn packed_mean_handles_empty_component_as_zero() {
        // Zero-length components give zero mean rather than NaN.
        let pnt =
            PackedNestedTensor::from_sequences(vec![vec![1.0f32, 2.0], vec![]], &[2usize, 0], &[])
                .unwrap();
        let means = pnt.mean_per_component();
        assert_eq!(means, vec![1.5, 0.0]);
    }

    #[test]
    fn packed_to_padded_pads_with_value() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![
                vec![1.0f32, 2.0, 3.0],
                vec![4.0, 5.0],
                vec![6.0, 7.0, 8.0, 9.0],
            ],
            &[3usize, 2, 4],
            &[],
        )
        .unwrap();

        let padded = pnt.to_padded(-1.0).unwrap();
        // [3 components × 4 max_len] = 12 elements
        assert_eq!(padded.shape(), &[3, 4]);
        let data = padded.data().unwrap();
        assert_eq!(
            data,
            &[
                1.0, 2.0, 3.0, -1.0, // first: [1,2,3] + pad
                4.0, 5.0, -1.0, -1.0, // second: [4,5] + pad
                6.0, 7.0, 8.0, 9.0, // third: [6,7,8,9]
            ]
        );
    }

    #[test]
    fn packed_to_padded_with_tail_shape() {
        // Two components of shape [L, 2], lengths 2 and 1.
        let pnt = PackedNestedTensor::from_sequences(
            vec![
                vec![1.0f32, 2.0, 3.0, 4.0], // 2 rows of 2
                vec![5.0, 6.0],              // 1 row of 2
            ],
            &[2usize, 1],
            &[2],
        )
        .unwrap();

        let padded = pnt.to_padded(0.0).unwrap();
        // [2 components, 2 max_len, 2 tail] = 8 elements
        assert_eq!(padded.shape(), &[2, 2, 2]);
        let data = padded.data().unwrap();
        assert_eq!(
            data,
            &[
                1.0, 2.0, 3.0, 4.0, // first: [[1,2],[3,4]]
                5.0, 6.0, 0.0, 0.0, // second: [[5,6], pad]
            ]
        );
    }

    #[test]
    fn packed_from_padded_inverse_of_to_padded() {
        // Roundtrip: pack → pad → unpack.
        let orig = PackedNestedTensor::from_sequences(
            vec![
                vec![1.0f32, 2.0, 3.0],
                vec![4.0, 5.0],
                vec![6.0, 7.0, 8.0, 9.0],
            ],
            &[3usize, 2, 4],
            &[],
        )
        .unwrap();

        let padded = orig.to_padded(-99.0).unwrap();
        let recovered = PackedNestedTensor::from_padded(&padded, &[3, 2, 4]).unwrap();

        assert_eq!(recovered.offsets(), orig.offsets());
        assert_eq!(recovered.data(), orig.data());
    }

    #[test]
    fn packed_from_padded_rejects_length_exceeding_max_len() {
        let data: Vec<f32> = vec![0.0; 12];
        let t = make_tensor(data, vec![3, 4]);
        let result = PackedNestedTensor::from_padded(&t, &[3, 5, 2]); // 5 > 4
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("exceeds max_len"));
    }

    #[test]
    fn packed_from_padded_rejects_lengths_count_mismatch() {
        let t = make_tensor(vec![0.0f32; 12], vec![3, 4]);
        let result = PackedNestedTensor::from_padded(&t, &[3, 4]); // 2 lengths for batch 3
        assert!(result.is_err());
    }

    #[test]
    fn packed_from_nested_and_back() {
        // Roundtrip from list-of-tensors NestedTensor to packed and back.
        let t1 = make_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        let t2 = make_tensor(vec![4.0, 5.0], vec![2]);
        let nested = NestedTensor::new(vec![t1, t2], 0).unwrap();

        let packed = PackedNestedTensor::from_nested(&nested).unwrap();
        assert_eq!(packed.num_components(), 2);
        assert_eq!(packed.data(), &[1.0, 2.0, 3.0, 4.0, 5.0]);

        let round_trip = packed.to_nested().unwrap();
        assert_eq!(round_trip.num_components(), 2);
        assert_eq!(round_trip.tensors()[0].shape(), &[3]);
        assert_eq!(round_trip.tensors()[1].shape(), &[2]);
        assert_eq!(round_trip.tensors()[0].data().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(round_trip.tensors()[1].data().unwrap(), &[4.0, 5.0]);
    }

    #[test]
    fn packed_from_nested_rejects_non_zero_ragged_dim() {
        // The packed layout requires ragged_dim == 0.
        let t1 = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let t2 = make_tensor(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![2, 3]);
        let nested = NestedTensor::new(vec![t1, t2], 1).unwrap();

        let result = PackedNestedTensor::from_nested(&nested);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("ragged_dim == 0"));
    }

    #[test]
    fn packed_f64_works_like_f32() {
        let pnt = PackedNestedTensor::from_sequences(
            vec![vec![1.0f64, 2.0], vec![3.0, 4.0, 5.0]],
            &[2usize, 3],
            &[],
        )
        .unwrap();
        assert_eq!(pnt.sum_per_component(), vec![3.0, 12.0]);
        let doubled = pnt.map(|x: f64| x * 2.0);
        assert_eq!(doubled.data(), &[2.0, 4.0, 6.0, 8.0, 10.0]);

        // Silence the unused `make_tensor_f64` warning by exercising it too.
        let dense = make_tensor_f64(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        assert_eq!(dense.shape(), &[2, 2]);
    }
}
