use std::collections::HashMap;
use std::fmt;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// A sparse tensor in COO (Coordinate List) format.
///
/// Stores only non-zero elements with their indices.
/// Efficient for tensors where most elements are zero (e.g., adjacency matrices,
/// sparse embeddings, one-hot vectors).
///
/// # Format
///
/// Each non-zero element is stored as a pair of `(index, value)` where `index`
/// is a `Vec<usize>` of length `ndim`, specifying the coordinate in the dense
/// tensor. For example, in a 3x4 matrix, the entry at row 1, column 2 has
/// index `[1, 2]`.
///
/// # Duplicate indices
///
/// The COO format permits duplicate indices. When converting to dense or
/// performing arithmetic, duplicates are summed. Call [`coalesce`](Self::coalesce)
/// to merge duplicates into a canonical form.
pub struct SparseTensor<T: Float> {
    /// Indices of non-zero elements: shape [nnz, ndim].
    /// Each element is a coordinate in the dense tensor.
    indices: Vec<Vec<usize>>,
    /// Values of non-zero elements: shape [nnz].
    values: Vec<T>,
    /// Shape of the dense tensor this represents.
    shape: Vec<usize>,
    /// Number of non-zero elements (including duplicates).
    nnz: usize,
}

impl<T: Float> SparseTensor<T> {
    /// Create a new sparse tensor from indices, values, and shape.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - `indices.len() != values.len()`
    /// - Any index vector has a length != `shape.len()`
    /// - Any index component is out of bounds for the corresponding dimension
    pub fn new(
        indices: Vec<Vec<usize>>,
        values: Vec<T>,
        shape: Vec<usize>,
    ) -> FerrotorchResult<Self> {
        if indices.len() != values.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "indices length ({}) must equal values length ({})",
                    indices.len(),
                    values.len()
                ),
            });
        }

        let ndim = shape.len();

        for (i, idx) in indices.iter().enumerate() {
            if idx.len() != ndim {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "index {} has {} dimensions but shape has {}",
                        i,
                        idx.len(),
                        ndim
                    ),
                });
            }
            for (axis, &coord) in idx.iter().enumerate() {
                if coord >= shape[axis] {
                    return Err(FerrotorchError::IndexOutOfBounds {
                        index: coord,
                        axis,
                        size: shape[axis],
                    });
                }
            }
        }

        let nnz = values.len();

        Ok(Self {
            indices,
            values,
            shape,
            nnz,
        })
    }

    /// Create a sparse tensor from a dense tensor.
    ///
    /// Elements whose absolute value is strictly greater than `threshold`
    /// are stored as non-zero entries.
    pub fn from_dense(tensor: &Tensor<T>, threshold: T) -> FerrotorchResult<Self> {
        let data = tensor.data()?;
        let shape = tensor.shape().to_vec();
        let ndim = shape.len();

        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (flat_idx, &val) in data.iter().enumerate() {
            if val.abs() > threshold {
                // Convert flat index to multi-dimensional index.
                let mut coord = vec![0usize; ndim];
                let mut remaining = flat_idx;
                for d in (0..ndim).rev() {
                    if shape[d] > 0 {
                        coord[d] = remaining % shape[d];
                        remaining /= shape[d];
                    }
                }
                indices.push(coord);
                values.push(val);
            }
        }

        let nnz = values.len();

        Ok(Self {
            indices,
            values,
            shape,
            nnz,
        })
    }

    /// Convert this sparse tensor to a dense `Tensor<T>`.
    ///
    /// Duplicate indices are summed during conversion.
    pub fn to_dense(&self) -> FerrotorchResult<Tensor<T>> {
        let numel: usize = self.shape.iter().product();
        let mut data = vec![<T as num_traits::Zero>::zero(); numel];
        let ndim = self.shape.len();

        for (idx, &val) in self.indices.iter().zip(self.values.iter()) {
            // Convert multi-dimensional index to flat index.
            let mut flat = 0usize;
            let mut stride = 1usize;
            for d in (0..ndim).rev() {
                flat += idx[d] * stride;
                stride *= self.shape[d];
            }
            data[flat] = data[flat] + val;
        }

        Tensor::from_storage(TensorStorage::cpu(data), self.shape.clone(), false)
    }

    /// Number of stored non-zero elements (including duplicates).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.nnz
    }

    /// Shape of the dense tensor this represents.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// The stored non-zero values.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// The indices of stored non-zero elements.
    #[inline]
    pub fn indices(&self) -> &[Vec<usize>] {
        &self.indices
    }

    /// Sparse-dense matrix multiply: `sparse [M, K] @ dense [K, N] -> dense [M, N]`.
    ///
    /// The sparse tensor must be 2-D. The dense tensor must be 2-D with its
    /// first dimension matching the sparse tensor's second dimension.
    ///
    /// # Algorithm
    ///
    /// For each non-zero entry `(i, j, v)` in the sparse matrix:
    ///
    /// ```text
    /// output[i, :] += v * dense[j, :]
    /// ```
    ///
    /// This is a scatter-accumulate pattern — the same kernel used in the
    /// backward pass of `nn.Embedding`.
    pub fn spmm(&self, dense: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if self.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "spmm requires 2-D sparse tensor, got {}-D",
                    self.ndim()
                ),
            });
        }
        if dense.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "spmm requires 2-D dense tensor, got {}-D",
                    dense.ndim()
                ),
            });
        }

        let m = self.shape[0];
        let k_sparse = self.shape[1];
        let dense_shape = dense.shape();
        let k_dense = dense_shape[0];
        let n = dense_shape[1];

        if k_sparse != k_dense {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "spmm inner dimensions mismatch: sparse [{}, {}] @ dense [{}, {}]",
                    m, k_sparse, k_dense, n
                ),
            });
        }

        let dense_data = dense.data()?;
        let mut output = vec![<T as num_traits::Zero>::zero(); m * n];

        // Scatter-accumulate: for each (i, j, v), output[i, :] += v * dense[j, :]
        for (idx, &v) in self.indices.iter().zip(self.values.iter()) {
            let i = idx[0];
            let j = idx[1];
            for col in 0..n {
                output[i * n + col] = output[i * n + col] + v * dense_data[j * n + col];
            }
        }

        Tensor::from_storage(TensorStorage::cpu(output), vec![m, n], false)
    }

    /// Element-wise multiply of all stored values by a scalar.
    ///
    /// Returns a new sparse tensor with the same sparsity pattern.
    pub fn mul_scalar(&self, scalar: T) -> Self {
        let new_values: Vec<T> = self.values.iter().map(|&v| v * scalar).collect();
        Self {
            indices: self.indices.clone(),
            values: new_values,
            shape: self.shape.clone(),
            nnz: self.nnz,
        }
    }

    /// Add two sparse tensors element-wise.
    ///
    /// The result contains the union of non-zero positions. Where indices
    /// overlap, values are summed. The result may contain duplicate indices
    /// — call [`coalesce`](Self::coalesce) afterwards if a canonical form is needed.
    ///
    /// Both tensors must have the same shape.
    pub fn add(&self, other: &SparseTensor<T>) -> FerrotorchResult<SparseTensor<T>> {
        if self.shape != other.shape {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "cannot add sparse tensors with shapes {:?} and {:?}",
                    self.shape, other.shape
                ),
            });
        }

        // Concatenate indices and values from both tensors.
        let mut indices = self.indices.clone();
        indices.extend_from_slice(&other.indices);

        let mut values = self.values.clone();
        values.extend_from_slice(&other.values);

        let nnz = values.len();

        Ok(SparseTensor {
            indices,
            values,
            shape: self.shape.clone(),
            nnz,
        })
    }

    /// Coalesce: merge duplicate indices by summing their values.
    ///
    /// Returns a new sparse tensor in canonical form where every index
    /// appears at most once and entries with a zero sum are removed.
    pub fn coalesce(&self) -> SparseTensor<T> {
        let mut map: HashMap<Vec<usize>, T> = HashMap::new();

        for (idx, &val) in self.indices.iter().zip(self.values.iter()) {
            let entry = map.entry(idx.clone()).or_insert_with(<T as num_traits::Zero>::zero);
            *entry = *entry + val;
        }

        // Remove entries that sum to zero.
        let mut indices = Vec::new();
        let mut values = Vec::new();

        for (idx, val) in map {
            if !<T as num_traits::Zero>::is_zero(&val) {
                indices.push(idx);
                values.push(val);
            }
        }

        let nnz = values.len();

        SparseTensor {
            indices,
            values,
            shape: self.shape.clone(),
            nnz,
        }
    }

    /// Transpose a 2-D sparse tensor.
    ///
    /// Swaps the row and column indices and transposes the shape.
    ///
    /// # Errors
    ///
    /// Returns an error if the tensor is not 2-D.
    pub fn t(&self) -> FerrotorchResult<SparseTensor<T>> {
        if self.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "transpose requires a 2-D sparse tensor, got {}-D",
                    self.ndim()
                ),
            });
        }

        let new_indices: Vec<Vec<usize>> = self
            .indices
            .iter()
            .map(|idx| vec![idx[1], idx[0]])
            .collect();

        let new_shape = vec![self.shape[1], self.shape[0]];

        Ok(SparseTensor {
            indices: new_indices,
            values: self.values.clone(),
            shape: new_shape,
            nnz: self.nnz,
        })
    }
}

// --- Trait impls ---

impl<T: Float> Clone for SparseTensor<T> {
    fn clone(&self) -> Self {
        Self {
            indices: self.indices.clone(),
            values: self.values.clone(),
            shape: self.shape.clone(),
            nnz: self.nnz,
        }
    }
}

impl<T: Float> fmt::Debug for SparseTensor<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SparseTensor")
            .field("shape", &self.shape)
            .field("nnz", &self.nnz)
            .field("ndim", &self.shape.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Construction and accessors ---

    #[test]
    fn test_construction_and_accessors() {
        let indices = vec![vec![0, 1], vec![1, 2], vec![2, 0]];
        let values = vec![1.0f32, 2.0, 3.0];
        let shape = vec![3, 3];

        let sp = SparseTensor::new(indices.clone(), values.clone(), shape.clone()).unwrap();

        assert_eq!(sp.nnz(), 3);
        assert_eq!(sp.shape(), &[3, 3]);
        assert_eq!(sp.ndim(), 2);
        assert_eq!(sp.values(), &[1.0, 2.0, 3.0]);
        assert_eq!(sp.indices(), &indices);
    }

    // --- from_dense with threshold ---

    #[test]
    fn test_from_dense_with_threshold() {
        // Dense 3x3 matrix with some near-zero values.
        let data = vec![
            0.0f32, 0.0, 5.0,
            0.0, 0.0, 0.0,
            3.0, 0.0, 0.0,
        ];
        let tensor = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 3], false).unwrap();

        let sp = SparseTensor::from_dense(&tensor, 0.0).unwrap();

        assert_eq!(sp.nnz(), 2);
        assert_eq!(sp.shape(), &[3, 3]);

        // Should contain [0,2] -> 5.0 and [2,0] -> 3.0
        let dense = sp.to_dense().unwrap();
        let d = dense.data().unwrap();
        assert_eq!(d[0 * 3 + 2], 5.0); // [0, 2]
        assert_eq!(d[2 * 3 + 0], 3.0); // [2, 0]
    }

    #[test]
    fn test_from_dense_threshold_filters_small() {
        let data = vec![0.5f32, 1.5, 0.1, 2.0];
        let tensor = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).unwrap();

        // threshold = 1.0: only values with |v| > 1.0 are stored.
        let sp = SparseTensor::from_dense(&tensor, 1.0).unwrap();

        assert_eq!(sp.nnz(), 2);
        let dense = sp.to_dense().unwrap();
        let d = dense.data().unwrap();
        assert_eq!(d[0], 0.0);   // 0.5 <= 1.0, filtered
        assert_eq!(d[1], 1.5);   // 1.5 > 1.0, kept
        assert_eq!(d[2], 0.0);   // 0.1 <= 1.0, filtered
        assert_eq!(d[3], 2.0);   // 2.0 > 1.0, kept
    }

    // --- to_dense round-trip ---

    #[test]
    fn test_to_dense_round_trip() {
        let data = vec![
            1.0f64, 0.0, 0.0,
            0.0, 2.0, 0.0,
            0.0, 0.0, 3.0,
        ];
        let original = Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 3], false).unwrap();

        let sp = SparseTensor::from_dense(&original, 0.0).unwrap();
        let reconstructed = sp.to_dense().unwrap();

        let orig_data = original.data().unwrap();
        let recon_data = reconstructed.data().unwrap();

        for (a, b) in orig_data.iter().zip(recon_data.iter()) {
            assert!((*a - *b).abs() < 1e-10, "mismatch: {} vs {}", a, b);
        }
    }

    // --- spmm matches dense mm ---

    #[test]
    fn test_spmm_matches_dense_mm() {
        // Sparse 2x3 matrix:
        // [[1, 0, 2],
        //  [0, 3, 0]]
        let sp = SparseTensor::new(
            vec![vec![0, 0], vec![0, 2], vec![1, 1]],
            vec![1.0f32, 2.0, 3.0],
            vec![2, 3],
        )
        .unwrap();

        // Dense 3x2 matrix:
        // [[1, 4],
        //  [2, 5],
        //  [3, 6]]
        let dense = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 4.0, 2.0, 5.0, 3.0, 6.0]),
            vec![3, 2],
            false,
        )
        .unwrap();

        let result = sp.spmm(&dense).unwrap();
        let d = result.data().unwrap();

        assert_eq!(result.shape(), &[2, 2]);

        // Row 0: [1, 0, 2] @ [[1, 4], [2, 5], [3, 6]] = [1*1 + 0*2 + 2*3, 1*4 + 0*5 + 2*6] = [7, 16]
        assert!((d[0] - 7.0).abs() < 1e-6);
        assert!((d[1] - 16.0).abs() < 1e-6);

        // Row 1: [0, 3, 0] @ [[1, 4], [2, 5], [3, 6]] = [0*1 + 3*2 + 0*3, 0*4 + 3*5 + 0*6] = [6, 15]
        assert!((d[2] - 6.0).abs() < 1e-6);
        assert!((d[3] - 15.0).abs() < 1e-6);
    }

    // --- spmm with identity sparse matrix ---

    #[test]
    fn test_spmm_identity() {
        // 3x3 identity as sparse.
        let sp = SparseTensor::new(
            vec![vec![0, 0], vec![1, 1], vec![2, 2]],
            vec![1.0f32; 3],
            vec![3, 3],
        )
        .unwrap();

        // Dense 3x2 matrix.
        let dense = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![3, 2],
            false,
        )
        .unwrap();

        let result = sp.spmm(&dense).unwrap();
        let d = result.data().unwrap();
        let expected = dense.data().unwrap();

        assert_eq!(result.shape(), &[3, 2]);
        for (a, b) in d.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // --- coalesce merges duplicates ---

    #[test]
    fn test_coalesce_merges_duplicates() {
        // Two entries at [0, 1] with values 3.0 and 4.0.
        let sp = SparseTensor::new(
            vec![vec![0, 0], vec![0, 1], vec![0, 1]],
            vec![1.0f32, 3.0, 4.0],
            vec![1, 3],
        )
        .unwrap();

        let coalesced = sp.coalesce();

        assert_eq!(coalesced.nnz(), 2); // [0,0] -> 1.0, [0,1] -> 7.0

        let dense = coalesced.to_dense().unwrap();
        let d = dense.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert!((d[1] - 7.0).abs() < 1e-6);
        assert!((d[2] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_coalesce_removes_zero_sum() {
        // Two entries at [0, 0] that cancel out.
        let sp = SparseTensor::new(
            vec![vec![0, 0], vec![0, 0]],
            vec![5.0f32, -5.0],
            vec![1, 1],
        )
        .unwrap();

        let coalesced = sp.coalesce();
        assert_eq!(coalesced.nnz(), 0);
    }

    // --- transpose ---

    #[test]
    fn test_transpose() {
        let sp = SparseTensor::new(
            vec![vec![0, 1], vec![2, 0]],
            vec![5.0f32, 3.0],
            vec![3, 4],
        )
        .unwrap();

        let transposed = sp.t().unwrap();

        assert_eq!(transposed.shape(), &[4, 3]);
        assert_eq!(transposed.nnz(), 2);
        assert_eq!(transposed.indices()[0], vec![1, 0]);
        assert_eq!(transposed.indices()[1], vec![0, 2]);
        assert_eq!(transposed.values(), &[5.0, 3.0]);
    }

    #[test]
    fn test_transpose_not_2d() {
        let sp = SparseTensor::new(
            vec![vec![0, 1, 2]],
            vec![1.0f32],
            vec![3, 3, 3],
        )
        .unwrap();

        assert!(sp.t().is_err());
    }

    // --- mul_scalar ---

    #[test]
    fn test_mul_scalar() {
        let sp = SparseTensor::new(
            vec![vec![0, 0], vec![1, 1]],
            vec![2.0f64, 3.0],
            vec![2, 2],
        )
        .unwrap();

        let scaled = sp.mul_scalar(10.0);

        assert_eq!(scaled.values(), &[20.0, 30.0]);
        assert_eq!(scaled.nnz(), 2);
        assert_eq!(scaled.shape(), &[2, 2]);
        assert_eq!(scaled.indices(), sp.indices());
    }

    // --- add two sparse tensors ---

    #[test]
    fn test_add_sparse_tensors() {
        // a: [0,0] -> 1.0, [0,1] -> 2.0
        let a = SparseTensor::new(
            vec![vec![0, 0], vec![0, 1]],
            vec![1.0f32, 2.0],
            vec![2, 2],
        )
        .unwrap();

        // b: [0,1] -> 3.0, [1,0] -> 4.0
        let b = SparseTensor::new(
            vec![vec![0, 1], vec![1, 0]],
            vec![3.0, 4.0],
            vec![2, 2],
        )
        .unwrap();

        let sum = a.add(&b).unwrap();

        // Uncoalesced: 4 entries ([0,0]->1, [0,1]->2, [0,1]->3, [1,0]->4).
        assert_eq!(sum.nnz(), 4);

        // After coalescing, [0,1] should have value 5.0.
        let coalesced = sum.coalesce();
        assert_eq!(coalesced.nnz(), 3);

        let dense = coalesced.to_dense().unwrap();
        let d = dense.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6); // [0,0]
        assert!((d[1] - 5.0).abs() < 1e-6); // [0,1] = 2 + 3
        assert!((d[2] - 4.0).abs() < 1e-6); // [1,0]
        assert!((d[3] - 0.0).abs() < 1e-6); // [1,1]
    }

    #[test]
    fn test_add_shape_mismatch() {
        let a = SparseTensor::<f32>::new(vec![], vec![], vec![2, 3]).unwrap();
        let b = SparseTensor::<f32>::new(vec![], vec![], vec![3, 2]).unwrap();

        assert!(a.add(&b).is_err());
    }

    // --- Error: index out of bounds ---

    #[test]
    fn test_index_out_of_bounds() {
        let result = SparseTensor::new(
            vec![vec![3, 0]], // row 3 in a 3x3 matrix is out of bounds
            vec![1.0f32],
            vec![3, 3],
        );

        assert!(result.is_err());
        let err = result.unwrap_err();
        match err {
            FerrotorchError::IndexOutOfBounds { index, axis, size } => {
                assert_eq!(index, 3);
                assert_eq!(axis, 0);
                assert_eq!(size, 3);
            }
            other => panic!("expected IndexOutOfBounds, got: {other:?}"),
        }
    }

    #[test]
    fn test_index_out_of_bounds_second_axis() {
        let result = SparseTensor::new(
            vec![vec![0, 5]], // col 5 in a 3x3 matrix is out of bounds
            vec![1.0f64],
            vec![3, 3],
        );

        assert!(result.is_err());
        match result.unwrap_err() {
            FerrotorchError::IndexOutOfBounds { index, axis, size } => {
                assert_eq!(index, 5);
                assert_eq!(axis, 1);
                assert_eq!(size, 3);
            }
            other => panic!("expected IndexOutOfBounds, got: {other:?}"),
        }
    }

    // --- Edge cases ---

    #[test]
    fn test_empty_sparse_tensor() {
        let sp = SparseTensor::<f32>::new(vec![], vec![], vec![5, 5]).unwrap();

        assert_eq!(sp.nnz(), 0);
        assert_eq!(sp.shape(), &[5, 5]);

        let dense = sp.to_dense().unwrap();
        assert!(dense.data().unwrap().iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_indices_values_length_mismatch() {
        let result = SparseTensor::new(
            vec![vec![0, 0], vec![1, 1]],
            vec![1.0f32], // only 1 value for 2 indices
            vec![2, 2],
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_spmm_dimension_mismatch() {
        let sp = SparseTensor::new(
            vec![vec![0, 0]],
            vec![1.0f32],
            vec![2, 3],
        )
        .unwrap();

        // Dense is 4x2, but sparse inner dim is 3.
        let dense = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32; 8]),
            vec![4, 2],
            false,
        )
        .unwrap();

        assert!(sp.spmm(&dense).is_err());
    }

    #[test]
    fn test_debug_format() {
        let sp = SparseTensor::new(
            vec![vec![0, 0]],
            vec![1.0f32],
            vec![3, 3],
        )
        .unwrap();

        let debug = format!("{sp:?}");
        assert!(debug.contains("SparseTensor"));
        assert!(debug.contains("nnz: 1"));
    }

    #[test]
    fn test_clone() {
        let sp = SparseTensor::new(
            vec![vec![0, 1]],
            vec![42.0f32],
            vec![2, 2],
        )
        .unwrap();

        let sp2 = sp.clone();
        assert_eq!(sp2.values(), &[42.0]);
        assert_eq!(sp2.indices(), &[vec![0, 1]]);
        assert_eq!(sp2.shape(), &[2, 2]);
    }

    // --- 1-D, 3-D, and zero-dimension edge cases ---

    #[test]
    fn test_1d_sparse_tensor() {
        let sp = SparseTensor::new(
            vec![vec![1], vec![4]],
            vec![10.0f32, 20.0],
            vec![5],
        )
        .unwrap();

        assert_eq!(sp.ndim(), 1);
        assert_eq!(sp.nnz(), 2);
        assert_eq!(sp.shape(), &[5]);

        let dense = sp.to_dense().unwrap();
        let d = dense.data().unwrap();
        assert_eq!(d.len(), 5);
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], 10.0);
        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], 0.0);
        assert_eq!(d[4], 20.0);
    }

    #[test]
    fn test_3d_sparse_tensor() {
        let sp = SparseTensor::new(
            vec![vec![0, 1, 2], vec![1, 0, 0]],
            vec![5.0f64, 7.0],
            vec![2, 2, 3],
        )
        .unwrap();

        assert_eq!(sp.ndim(), 3);
        assert_eq!(sp.nnz(), 2);
        assert_eq!(sp.shape(), &[2, 2, 3]);

        let dense = sp.to_dense().unwrap();
        let d = dense.data().unwrap();
        assert_eq!(d.len(), 12);
        // [0,1,2] -> flat index = 0*6 + 1*3 + 2 = 5
        assert!((d[5] - 5.0).abs() < 1e-10);
        // [1,0,0] -> flat index = 1*6 + 0*3 + 0 = 6
        assert!((d[6] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_zero_dimension_sparse_tensor() {
        // Shape [0, 5]: zero rows, 5 columns. No elements possible.
        let sp = SparseTensor::<f32>::new(vec![], vec![], vec![0, 5]).unwrap();

        assert_eq!(sp.ndim(), 2);
        assert_eq!(sp.nnz(), 0);
        assert_eq!(sp.shape(), &[0, 5]);

        let dense = sp.to_dense().unwrap();
        assert_eq!(dense.numel(), 0);
        assert!(dense.data().unwrap().is_empty());
    }
}
