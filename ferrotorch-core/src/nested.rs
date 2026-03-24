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
                message: format!(
                    "ragged_dim {} out of range for {}-D tensors",
                    ragged_dim, ndim
                ),
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
            for flat in 0..t_numel {
                // Convert flat index to multi-dim coords in the component.
                let mut remaining = flat;
                let mut out_flat = b * out_strides[0];
                for d in 0..ndim {
                    let coord = remaining / t_strides[d];
                    remaining %= t_strides[d];
                    out_flat += coord * out_strides[d + 1];
                }
                data[out_flat] = t_data[flat];
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
                    "ragged_dim {} out of range for {}-D component tensors",
                    ragged_dim, comp_ndim
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
        for b in 0..batch {
            // Build component shape: same as full_shape[1..] but with
            // ragged_dim replaced by lengths[b].
            let mut comp_shape = Vec::with_capacity(comp_ndim);
            for d in 0..comp_ndim {
                if d == ragged_dim {
                    comp_shape.push(lengths[b]);
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
                    let coord = if comp_strides[d] > 0 {
                        remaining / comp_strides[d]
                    } else {
                        0
                    };
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
            .fold(<T as num_traits::Float>::neg_infinity(), |a, b| if b > a { b } else { a });

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
                message: format!(
                    "component {}: query d_k={} but key d_k={}",
                    i, d_k, d_k2
                ),
            });
        }
        if seq_k != seq_k2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "component {}: key seq_len={} but value seq_len={}",
                    i, seq_k, seq_k2
                ),
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
    fn test_from_padded_round_trip_ragged_dim_1() {
        // Component tensors: shape [2, L_i] where dim 1 is ragged.
        // t1: [2, 3], t2: [2, 2]
        let t1 = make_tensor(
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            vec![2, 3],
        );
        let t2 = make_tensor(
            vec![7.0, 8.0, 9.0, 10.0],
            vec![2, 2],
        );

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
        for &val in out.iter() {
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
        assert!((sum - 1.0).abs() < 1e-6, "softmax should sum to 1, got {sum}");
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
    }
}
