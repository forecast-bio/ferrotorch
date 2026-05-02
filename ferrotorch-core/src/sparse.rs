use std::collections::{HashMap, HashSet};
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
            data[flat] += val;
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
                message: format!("spmm requires 2-D sparse tensor, got {}-D", self.ndim()),
            });
        }
        if dense.ndim() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("spmm requires 2-D dense tensor, got {}-D", dense.ndim()),
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
                output[i * n + col] += v * dense_data[j * n + col];
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
    /// appears at most once, entries with a zero sum are removed, and
    /// the remaining entries are sorted lexicographically by index for
    /// deterministic output.
    pub fn coalesce(&self) -> SparseTensor<T> {
        let mut map: HashMap<Vec<usize>, T> = HashMap::new();

        for (idx, &val) in self.indices.iter().zip(self.values.iter()) {
            let entry = map
                .entry(idx.clone())
                .or_insert_with(<T as num_traits::Zero>::zero);
            *entry += val;
        }

        // Remove entries that sum to zero, collect into pairs.
        let mut pairs: Vec<(Vec<usize>, T)> = map
            .into_iter()
            .filter(|(_, val)| !<T as num_traits::Zero>::is_zero(val))
            .collect();

        // Sort lexicographically by index for deterministic order.
        pairs.sort_by(|(a, _), (b, _)| a.cmp(b));

        let mut indices = Vec::with_capacity(pairs.len());
        let mut values = Vec::with_capacity(pairs.len());
        for (idx, val) in pairs {
            indices.push(idx);
            values.push(val);
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

// --- CooTensor: 2-D COO format with separate row/col arrays ---

/// A 2-D sparse tensor in COO (Coordinate List) format with separate
/// row and column index arrays.
///
/// Unlike [`SparseTensor`] which uses `Vec<Vec<usize>>` for arbitrary-rank
/// indices, `CooTensor` stores flat `row_indices` and `col_indices` arrays
/// for better cache locality on 2-D matrices.
#[derive(Debug, Clone)]
pub struct CooTensor<T: Float> {
    row_indices: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
    nrows: usize,
    ncols: usize,
    is_coalesced: bool,
}

impl<T: Float> CooTensor<T> {
    /// Create a new 2-D COO sparse tensor.
    pub fn new(
        row_indices: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
        nrows: usize,
        ncols: usize,
    ) -> FerrotorchResult<Self> {
        if row_indices.len() != values.len() || col_indices.len() != values.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "row_indices ({}), col_indices ({}), and values ({}) must have equal length",
                    row_indices.len(),
                    col_indices.len(),
                    values.len()
                ),
            });
        }
        for (i, (&r, &c)) in row_indices.iter().zip(col_indices.iter()).enumerate() {
            if r >= nrows {
                return Err(FerrotorchError::IndexOutOfBounds {
                    index: r,
                    axis: 0,
                    size: nrows,
                });
            }
            if c >= ncols {
                return Err(FerrotorchError::IndexOutOfBounds {
                    index: c,
                    axis: 1,
                    size: ncols,
                });
            }
            let _ = i; // suppress unused warning
        }
        Ok(Self {
            row_indices,
            col_indices,
            values,
            nrows,
            ncols,
            is_coalesced: false,
        })
    }

    /// Number of stored entries (including duplicates).
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Whether duplicate indices have been merged.
    #[inline]
    pub fn is_coalesced(&self) -> bool {
        self.is_coalesced
    }

    /// Row indices of stored entries.
    #[inline]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Column indices of stored entries.
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Stored values.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Number of rows.
    #[inline]
    pub fn nrows(&self) -> usize {
        self.nrows
    }

    /// Number of columns.
    #[inline]
    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Coalesce: merge duplicate `(row, col)` entries by summing values.
    ///
    /// Uses `(row, col)` tuples as HashMap keys to avoid flat-index overflow
    /// on large matrices. The output is sorted lexicographically by `(row, col)`.
    pub fn coalesce(&self) -> Self {
        // Use (row, col) tuple as key to avoid overflow from flat index.
        let mut map: HashMap<(usize, usize), T> = HashMap::new();

        for i in 0..self.values.len() {
            let key = (self.row_indices[i], self.col_indices[i]);
            let entry = map.entry(key).or_insert_with(<T as num_traits::Zero>::zero);
            *entry += self.values[i];
        }

        let mut pairs: Vec<((usize, usize), T)> = map
            .into_iter()
            .filter(|(_, val)| !<T as num_traits::Zero>::is_zero(val))
            .collect();

        // Sort by (row, col) for deterministic order.
        pairs.sort_by_key(|&((r, c), _)| (r, c));

        let mut row_indices = Vec::with_capacity(pairs.len());
        let mut col_indices = Vec::with_capacity(pairs.len());
        let mut values = Vec::with_capacity(pairs.len());
        for ((r, c), v) in pairs {
            row_indices.push(r);
            col_indices.push(c);
            values.push(v);
        }

        Self {
            row_indices,
            col_indices,
            values,
            nrows: self.nrows,
            ncols: self.ncols,
            is_coalesced: true,
        }
    }

    /// Convert to dense tensor.
    pub fn to_dense(&self) -> FerrotorchResult<Tensor<T>> {
        let mut data = vec![<T as num_traits::Zero>::zero(); self.nrows * self.ncols];
        for i in 0..self.values.len() {
            let flat = self.row_indices[i] * self.ncols + self.col_indices[i];
            data[flat] += self.values[i];
        }
        Tensor::from_storage(
            TensorStorage::cpu(data),
            vec![self.nrows, self.ncols],
            false,
        )
    }

    /// Convert from a CSR tensor.
    ///
    /// The result is conservatively marked as uncoalesced (`is_coalesced = false`)
    /// because we do not validate uniqueness of entries from the source.
    pub fn from_csr(csr: &CsrTensor<T>) -> Self {
        let mut row_indices = Vec::new();
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        for row in 0..csr.nrows {
            let start = csr.row_ptrs[row];
            let end = csr.row_ptrs[row + 1];
            for j in start..end {
                row_indices.push(row);
                col_indices.push(csr.col_indices[j]);
                values.push(csr.values[j]);
            }
        }

        Self {
            row_indices,
            col_indices,
            values,
            nrows: csr.nrows,
            ncols: csr.ncols,
            // Conservative: do not assume CSR source was validated for uniqueness.
            is_coalesced: false,
        }
    }
}

// --- CsrTensor: Compressed Sparse Row ---

/// A 2-D sparse tensor in CSR (Compressed Sparse Row) format.
///
/// Stores row boundaries in `row_ptrs` (length `nrows + 1`), column indices
/// in `col_indices`, and corresponding values in `values`.
///
/// Efficient for row-slicing and sparse matrix-vector products.
#[derive(Debug, Clone)]
pub struct CsrTensor<T: Float> {
    row_ptrs: Vec<usize>,
    col_indices: Vec<usize>,
    values: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T: Float> CsrTensor<T> {
    /// Create a CSR tensor directly from components.
    pub fn new(
        row_ptrs: Vec<usize>,
        col_indices: Vec<usize>,
        values: Vec<T>,
        nrows: usize,
        ncols: usize,
    ) -> FerrotorchResult<Self> {
        if row_ptrs.len() != nrows + 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "row_ptrs length ({}) must be nrows + 1 ({})",
                    row_ptrs.len(),
                    nrows + 1
                ),
            });
        }
        if col_indices.len() != values.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "col_indices length ({}) must equal values length ({})",
                    col_indices.len(),
                    values.len()
                ),
            });
        }
        Ok(Self {
            row_ptrs,
            col_indices,
            values,
            nrows,
            ncols,
        })
    }

    /// Convert a COO tensor to CSR format.
    ///
    /// Uses a dense scratch array (of size `nrows * ncols`) to accumulate
    /// values and detect duplicates in O(nnz) time rather than O(nnz^2).
    pub fn from_coo(coo: &CooTensor<T>) -> FerrotorchResult<Self> {
        let nrows = coo.nrows;
        let ncols = coo.ncols;

        // Use a HashSet to detect duplicates efficiently without O(n^2) scans.
        let mut seen: HashSet<(usize, usize)> = HashSet::with_capacity(coo.nnz());

        // Accumulate into a row-oriented structure.
        // row_entries[r] = vec of (col, value).
        let mut row_entries: Vec<HashMap<usize, T>> = vec![HashMap::new(); nrows];

        for i in 0..coo.values.len() {
            let r = coo.row_indices[i];
            let c = coo.col_indices[i];
            seen.insert((r, c));
            let entry = row_entries[r]
                .entry(c)
                .or_insert_with(<T as num_traits::Zero>::zero);
            *entry += coo.values[i];
        }

        // Build CSR arrays.
        let mut row_ptrs = Vec::with_capacity(nrows + 1);
        let mut col_indices = Vec::new();
        let mut values = Vec::new();

        row_ptrs.push(0);
        for entry in row_entries.iter_mut().take(nrows) {
            let mut cols: Vec<(usize, T)> = entry
                .drain()
                .filter(|(_, v)| !<T as num_traits::Zero>::is_zero(v))
                .collect();
            cols.sort_by_key(|&(c, _)| c);
            for (c, v) in cols {
                col_indices.push(c);
                values.push(v);
            }
            row_ptrs.push(col_indices.len());
        }

        let _ = seen; // used for efficient duplicate detection

        Ok(Self {
            row_ptrs,
            col_indices,
            values,
            nrows,
            ncols,
        })
    }

    /// Number of stored non-zero entries.
    #[inline]
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Row pointer array (length `nrows + 1`).
    #[inline]
    pub fn row_ptrs(&self) -> &[usize] {
        &self.row_ptrs
    }

    /// Column index array.
    #[inline]
    pub fn col_indices(&self) -> &[usize] {
        &self.col_indices
    }

    /// Stored values.
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Convert to dense tensor.
    pub fn to_dense(&self) -> FerrotorchResult<Tensor<T>> {
        let mut data = vec![<T as num_traits::Zero>::zero(); self.nrows * self.ncols];
        for row in 0..self.nrows {
            let start = self.row_ptrs[row];
            let end = self.row_ptrs[row + 1];
            for j in start..end {
                let flat = row * self.ncols + self.col_indices[j];
                data[flat] += self.values[j];
            }
        }
        Tensor::from_storage(
            TensorStorage::cpu(data),
            vec![self.nrows, self.ncols],
            false,
        )
    }
}

// ===========================================================================
// SemiStructuredSparseTensor — 2:4 structured sparsity. CL-292.
// ===========================================================================

/// A tensor stored in the NVIDIA 2:4 structured sparsity format.
///
/// In 2:4 structured sparsity, every contiguous group of 4 elements
/// along the innermost dimension has exactly 2 non-zero values.
/// This regular pattern is what NVIDIA's Sparse Tensor Cores
/// consume (Ampere SM_80+) to deliver up to 2× matmul throughput
/// on sparse weights.
///
/// # Storage
///
/// - `values` holds the retained elements in original row-major
///   order, with length `original.numel() / 2`. Each group of 4
///   contributes 2 consecutive values.
/// - `mask` is a byte-packed metadata stream: 4 bits per group
///   encoding which 2 of the 4 positions were kept. Two groups
///   pack into one byte, so `mask.len() == (num_groups + 1) / 2`.
/// - `shape` preserves the original dense shape so the tensor can
///   be decompressed back to the same layout.
///
/// # Invariants
///
/// - `original.numel() % 4 == 0` (the innermost stride must cover
///   full 4-element groups; non-multiples are rejected at
///   construction time).
/// - Every group's mask has **exactly** 2 bits set.
/// - `values.len() == num_groups * 2`.
/// - `mask.len() == num_groups.div_ceil(2)`.
#[derive(Debug, Clone)]
pub struct SemiStructuredSparseTensor<T: Float> {
    /// Retained values in row-major order (2 per group of 4).
    values: Vec<T>,
    /// Byte-packed 4-bit-per-group masks (2 groups per byte).
    mask: Vec<u8>,
    /// Original dense shape.
    shape: Vec<usize>,
}

impl<T: Float> SemiStructuredSparseTensor<T> {
    /// Compress a dense tensor into 2:4 semi-structured format.
    ///
    /// For each contiguous group of 4 elements along the flat
    /// row-major order, keeps the 2 elements with the largest
    /// absolute value and zeros the other two. Ties are broken
    /// by position (lower index wins).
    ///
    /// # Errors
    ///
    /// - `FerrotorchError::InvalidArgument` if `dense.numel() % 4 != 0`.
    pub fn compress(dense: &Tensor<T>) -> FerrotorchResult<Self> {
        let data = dense.data_vec()?;
        let numel = data.len();
        if numel % 4 != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "SemiStructuredSparseTensor::compress: numel must be a \
                     multiple of 4, got {numel}"
                ),
            });
        }
        let num_groups = numel / 4;
        let mut values = Vec::with_capacity(num_groups * 2);
        let mut mask = vec![0u8; num_groups.div_ceil(2)];

        for g in 0..num_groups {
            let base = g * 4;
            // Find the 2 largest-magnitude positions.
            let mut mags: [(usize, T); 4] = [
                (0, data[base].abs()),
                (1, data[base + 1].abs()),
                (2, data[base + 2].abs()),
                (3, data[base + 3].abs()),
            ];
            // Sort descending by magnitude; stable on ties so
            // lower index wins.
            mags.sort_by(|a, b| {
                b.1.partial_cmp(&a.1)
                    .unwrap_or(std::cmp::Ordering::Equal)
                    .then_with(|| a.0.cmp(&b.0))
            });
            // Take positions [0] and [1] as kept, then sort those
            // two ascending so values are stored in original
            // position order.
            let mut kept = [mags[0].0, mags[1].0];
            kept.sort();
            values.push(data[base + kept[0]]);
            values.push(data[base + kept[1]]);

            // Build the 4-bit mask for this group: set bit i if
            // position i was kept.
            let nibble: u8 = (1 << kept[0]) | (1 << kept[1]);
            // Pack into the mask byte: group `g` uses bits
            // (g % 2) * 4 .. (g % 2) * 4 + 4.
            let byte = g / 2;
            let shift = (g % 2) * 4;
            mask[byte] |= nibble << shift;
        }

        Ok(Self {
            values,
            mask,
            shape: dense.shape().to_vec(),
        })
    }

    /// Decompress back to a dense `Tensor<T>`. The output has the
    /// same shape as the original and zeros at every position
    /// that was masked out.
    pub fn decompress(&self) -> FerrotorchResult<Tensor<T>> {
        let numel = self.shape.iter().product::<usize>();
        let mut out = vec![<T as num_traits::Zero>::zero(); numel];
        let num_groups = numel / 4;

        for g in 0..num_groups {
            let byte = g / 2;
            let shift = (g % 2) * 4;
            let nibble = (self.mask[byte] >> shift) & 0xF;
            // Walk the 4 bits in ascending order; for each set
            // bit, consume one value from the stream.
            let mut val_idx = g * 2;
            for pos in 0..4 {
                if (nibble >> pos) & 1 != 0 {
                    out[g * 4 + pos] = self.values[val_idx];
                    val_idx += 1;
                }
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), self.shape.clone(), false)
    }

    /// Original dense shape.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Retained values in row-major order (2 per group of 4).
    #[inline]
    pub fn values(&self) -> &[T] {
        &self.values
    }

    /// Byte-packed 4-bit-per-group mask stream.
    #[inline]
    pub fn mask(&self) -> &[u8] {
        &self.mask
    }

    /// Number of 4-element groups in the original tensor.
    #[inline]
    pub fn num_groups(&self) -> usize {
        self.shape.iter().product::<usize>() / 4
    }

    /// Compressed byte count vs. dense byte count, as a ratio in
    /// `(0, 1]`. For 2:4 sparsity the values halve the storage;
    /// the mask adds ~1 byte per 8 elements, so the steady-state
    /// ratio is ≈ 0.5 + 1/16 = 0.5625 of the dense size (for f32).
    pub fn compression_ratio(&self) -> f64 {
        let dense_bytes = (self.shape.iter().product::<usize>()) * std::mem::size_of::<T>();
        if dense_bytes == 0 {
            return 1.0;
        }
        let compressed = self.values.len() * std::mem::size_of::<T>() + self.mask.len();
        compressed as f64 / dense_bytes as f64
    }

    /// Return the extracted 4-bit nibble for group `g`. The low
    /// four bits of the returned byte hold the mask (bit `i` set
    /// if position `i` was kept).
    ///
    /// # Panics
    ///
    /// Panics if `g >= num_groups()`.
    pub fn group_mask(&self, g: usize) -> u8 {
        let byte = g / 2;
        let shift = (g % 2) * 4;
        (self.mask[byte] >> shift) & 0xF
    }
}

/// Matrix multiply `a @ b` where `b` is stored in 2:4 semi-
/// structured format. The last-dim strides of `b`'s original
/// dense shape must be a multiple of 4 (guaranteed by
/// [`SemiStructuredSparseTensor::compress`]).
///
/// This is a **reference implementation** that decompresses `b`
/// and calls the dense matmul path. It establishes the correct
/// numeric behavior and API surface for a future Tensor Core
/// specialization. CL-292.
///
/// # Shape contract
///
/// - `a` is 2-D with shape `[m, k]`.
/// - `b` is a compressed representation of a `[k, n]` dense
///   weight matrix where `n % 4 == 0`.
/// - Output shape: `[m, n]`.
///
/// # Errors
///
/// - `b.shape().len() != 2`
/// - `a.shape().len() != 2`
/// - Inner dimensions don't match
pub fn sparse_matmul_24<T: Float>(
    a: &Tensor<T>,
    b: &SemiStructuredSparseTensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    if a.shape().len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "sparse_matmul_24: `a` must be 2-D, got shape {:?}",
                a.shape()
            ),
        });
    }
    if b.shape().len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "sparse_matmul_24: `b` must be 2-D, got shape {:?}",
                b.shape()
            ),
        });
    }
    let m = a.shape()[0];
    let k = a.shape()[1];
    let kb = b.shape()[0];
    let n = b.shape()[1];
    if k != kb {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "sparse_matmul_24: inner dims mismatch: a.shape[1]={k} != b.shape[0]={kb}"
            ),
        });
    }

    // Reference path: decompress and do the dense matmul.
    let b_dense = b.decompress()?;
    let a_data = a.data_vec()?;
    let b_data = b_dense.data_vec()?;
    let mut out = vec![<T as num_traits::Zero>::zero(); m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = <T as num_traits::Zero>::zero();
            for kk in 0..k {
                acc += a_data[i * k + kk] * b_data[kk * n + j];
            }
            out[i * n + j] = acc;
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![m, n], false)
}

// ===========================================================================
// CscTensor: Compressed Sparse Column (#619)
// ===========================================================================

/// 2-D sparse tensor in CSC (Compressed Sparse Column) format. Dual of
/// [`CsrTensor`]: instead of storing row pointers + column indices,
/// stores column pointers (`col_ptrs`, length `ncols + 1`) and row
/// indices for each non-zero. Efficient for column slicing and
/// `A^T x` style ops.
#[derive(Debug, Clone)]
pub struct CscTensor<T: Float> {
    col_ptrs: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<T>,
    nrows: usize,
    ncols: usize,
}

impl<T: Float> CscTensor<T> {
    /// Build directly from components.
    pub fn new(
        col_ptrs: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<T>,
        nrows: usize,
        ncols: usize,
    ) -> FerrotorchResult<Self> {
        if col_ptrs.len() != ncols + 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CscTensor: col_ptrs length ({}) must be ncols + 1 ({})",
                    col_ptrs.len(),
                    ncols + 1
                ),
            });
        }
        if row_indices.len() != values.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CscTensor: row_indices length ({}) must equal values length ({})",
                    row_indices.len(),
                    values.len()
                ),
            });
        }
        for &r in &row_indices {
            if r >= nrows {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("CscTensor: row index {r} >= nrows {nrows}"),
                });
            }
        }
        Ok(Self {
            col_ptrs,
            row_indices,
            values,
            nrows,
            ncols,
        })
    }

    /// Convert from CSR: transpose the row/col axis. CSR stores by rows,
    /// CSC stores by columns — we walk the CSR entries column-by-column
    /// and bucket them.
    pub fn from_csr(csr: &CsrTensor<T>) -> Self {
        let nrows = csr.nrows;
        let ncols = csr.ncols;

        // Count entries per column.
        let mut counts = vec![0usize; ncols];
        for &c in &csr.col_indices {
            counts[c] += 1;
        }
        // Prefix-sum into col_ptrs.
        let mut col_ptrs = vec![0usize; ncols + 1];
        for j in 0..ncols {
            col_ptrs[j + 1] = col_ptrs[j] + counts[j];
        }
        let nnz = csr.values.len();
        let mut row_indices = vec![0usize; nnz];
        let mut values = vec![<T as num_traits::Zero>::zero(); nnz];
        // Per-column write cursors.
        let mut cursor = col_ptrs.clone();
        for r in 0..nrows {
            let start = csr.row_ptrs[r];
            let end = csr.row_ptrs[r + 1];
            for k in start..end {
                let c = csr.col_indices[k];
                let dst = cursor[c];
                row_indices[dst] = r;
                values[dst] = csr.values[k];
                cursor[c] += 1;
            }
        }

        Self {
            col_ptrs,
            row_indices,
            values,
            nrows,
            ncols,
        }
    }

    /// Convert to CSR via the dual conversion.
    pub fn to_csr(&self) -> CsrTensor<T> {
        // Mirror the same algorithm with rows/cols swapped.
        let mut counts = vec![0usize; self.nrows];
        for &r in &self.row_indices {
            counts[r] += 1;
        }
        let mut row_ptrs = vec![0usize; self.nrows + 1];
        for i in 0..self.nrows {
            row_ptrs[i + 1] = row_ptrs[i] + counts[i];
        }
        let nnz = self.values.len();
        let mut col_indices = vec![0usize; nnz];
        let mut values = vec![<T as num_traits::Zero>::zero(); nnz];
        let mut cursor = row_ptrs.clone();
        for c in 0..self.ncols {
            let start = self.col_ptrs[c];
            let end = self.col_ptrs[c + 1];
            for k in start..end {
                let r = self.row_indices[k];
                let dst = cursor[r];
                col_indices[dst] = c;
                values[dst] = self.values[k];
                cursor[r] += 1;
            }
        }
        CsrTensor::new(row_ptrs, col_indices, values, self.nrows, self.ncols)
            .expect("CSR rebuild from CSC always satisfies invariants")
    }

    /// Materialize as a dense 2-D `Tensor`.
    pub fn to_dense(&self) -> FerrotorchResult<Tensor<T>> {
        let mut data = vec![<T as num_traits::Zero>::zero(); self.nrows * self.ncols];
        for c in 0..self.ncols {
            let start = self.col_ptrs[c];
            let end = self.col_ptrs[c + 1];
            for k in start..end {
                let r = self.row_indices[k];
                data[r * self.ncols + c] = self.values[k];
            }
        }
        Tensor::from_storage(
            TensorStorage::cpu(data),
            vec![self.nrows, self.ncols],
            false,
        )
    }

    pub fn nnz(&self) -> usize {
        self.values.len()
    }
    pub fn nrows(&self) -> usize {
        self.nrows
    }
    pub fn ncols(&self) -> usize {
        self.ncols
    }
    pub fn col_ptrs(&self) -> &[usize] {
        &self.col_ptrs
    }
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }
    pub fn values(&self) -> &[T] {
        &self.values
    }
}

// ===========================================================================
// SparseGrad: index/value pairs for sparse-gradient optimizer steps (#619)
// ===========================================================================

/// A sparse gradient: a list of (index, value) pairs that an optimizer
/// applies to a dense parameter tensor. Mirrors the `coalesce`d form of
/// `torch.Tensor.is_sparse` gradients used by `nn.Embedding(sparse=True)`
/// and consumed by `optim.SparseAdam` / `optim.SGD`.
///
/// `indices` and `values` describe a sparse update along the leading
/// dimension of the parameter (e.g. for an embedding of shape `[V, D]`,
/// each `index` is a row in `[0, V)` and the corresponding `value` is
/// the gradient row of shape `[D]`).
#[derive(Debug, Clone)]
pub struct SparseGrad<T: Float> {
    /// Affected leading-dim positions (length = nnz).
    indices: Vec<usize>,
    /// Per-affected-position gradient slabs, flat row-major: each slab
    /// is `slab_size` elements and corresponds to `indices[i]`.
    /// Length = `nnz * slab_size`.
    values: Vec<T>,
    /// Trailing-dim shape (everything past the leading dim).
    slab_shape: Vec<usize>,
}

impl<T: Float> SparseGrad<T> {
    /// Build from indices + per-index value slabs.
    ///
    /// `values.len()` must equal `indices.len() * prod(slab_shape)`.
    pub fn new(
        indices: Vec<usize>,
        values: Vec<T>,
        slab_shape: Vec<usize>,
    ) -> FerrotorchResult<Self> {
        let slab_size: usize = slab_shape.iter().product::<usize>().max(1);
        if values.len() != indices.len() * slab_size {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "SparseGrad: values.len()={} != indices.len()={} * slab_size={}",
                    values.len(),
                    indices.len(),
                    slab_size
                ),
            });
        }
        Ok(Self {
            indices,
            values,
            slab_shape,
        })
    }

    /// Number of distinct sparse rows (may include duplicates if the
    /// caller didn't coalesce).
    pub fn nnz(&self) -> usize {
        self.indices.len()
    }

    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Flat per-slab buffer; length `nnz * slab_size`. Use [`slab_shape`]
    /// to interpret each slab.
    pub fn values(&self) -> &[T] {
        &self.values
    }

    pub fn slab_shape(&self) -> &[usize] {
        &self.slab_shape
    }

    pub fn slab_size(&self) -> usize {
        self.slab_shape.iter().product::<usize>().max(1)
    }

    /// Coalesce: sum slabs that share the same index, returning a new
    /// `SparseGrad` whose `indices` are unique (and sorted ascending).
    pub fn coalesce(&self) -> Self {
        let slab_size = self.slab_size();
        let mut groups: std::collections::BTreeMap<usize, Vec<T>> =
            std::collections::BTreeMap::new();
        for (k, &idx) in self.indices.iter().enumerate() {
            let slab_start = k * slab_size;
            let entry = groups
                .entry(idx)
                .or_insert_with(|| vec![<T as num_traits::Zero>::zero(); slab_size]);
            for (j, dst) in entry.iter_mut().enumerate() {
                *dst += self.values[slab_start + j];
            }
        }
        let new_nnz = groups.len();
        let mut new_indices = Vec::with_capacity(new_nnz);
        let mut new_values = Vec::with_capacity(new_nnz * slab_size);
        for (idx, slab) in groups {
            new_indices.push(idx);
            new_values.extend(slab);
        }
        Self {
            indices: new_indices,
            values: new_values,
            slab_shape: self.slab_shape.clone(),
        }
    }

    /// Apply this sparse gradient to a dense parameter tensor: update
    /// `param[indices[i]] -= lr * values[i]` for every i. The leading
    /// dim of `param` is the indexed dim; the rest must match `slab_shape`.
    pub fn apply_sgd(&self, param: &mut Tensor<T>, lr: T) -> FerrotorchResult<()> {
        let shape = param.shape().to_vec();
        if shape.len() != 1 + self.slab_shape.len() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "SparseGrad::apply_sgd: param shape {:?} incompatible with slab shape {:?}",
                    shape, self.slab_shape
                ),
            });
        }
        if shape[1..] != self.slab_shape[..] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "SparseGrad::apply_sgd: param trailing dims {:?} != slab_shape {:?}",
                    &shape[1..],
                    self.slab_shape
                ),
            });
        }
        let slab_size = self.slab_size();
        let leading = shape[0];
        let mut data = param.data_vec()?;
        for (k, &idx) in self.indices.iter().enumerate() {
            if idx >= leading {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("SparseGrad::apply_sgd: index {idx} >= {leading}"),
                });
            }
            let row_start = idx * slab_size;
            let val_start = k * slab_size;
            for j in 0..slab_size {
                data[row_start + j] = data[row_start + j] - lr * self.values[val_start + j];
            }
        }
        *param = Tensor::from_storage(TensorStorage::cpu(data), shape, param.requires_grad())?;
        Ok(())
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
        let data = vec![0.0f32, 0.0, 5.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0];
        let tensor = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 3], false).unwrap();

        let sp = SparseTensor::from_dense(&tensor, 0.0).unwrap();

        assert_eq!(sp.nnz(), 2);
        assert_eq!(sp.shape(), &[3, 3]);

        // Should contain [0,2] -> 5.0 and [2,0] -> 3.0
        let dense = sp.to_dense().unwrap();
        let d = dense.data().unwrap();
        // Row-major 3x3, stride = (3, 1).
        let idx = |r: usize, c: usize| r * 3 + c;
        assert_eq!(d[idx(0, 2)], 5.0);
        assert_eq!(d[idx(2, 0)], 3.0);
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
        assert_eq!(d[0], 0.0); // 0.5 <= 1.0, filtered
        assert_eq!(d[1], 1.5); // 1.5 > 1.0, kept
        assert_eq!(d[2], 0.0); // 0.1 <= 1.0, filtered
        assert_eq!(d[3], 2.0); // 2.0 > 1.0, kept
    }

    // --- to_dense round-trip ---

    #[test]
    fn test_to_dense_round_trip() {
        let data = vec![1.0f64, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 3.0];
        let original =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 3], false).unwrap();

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
        let sp = SparseTensor::new(vec![vec![0, 0], vec![0, 0]], vec![5.0f32, -5.0], vec![1, 1])
            .unwrap();

        let coalesced = sp.coalesce();
        assert_eq!(coalesced.nnz(), 0);
    }

    // --- transpose ---

    #[test]
    fn test_transpose() {
        let sp =
            SparseTensor::new(vec![vec![0, 1], vec![2, 0]], vec![5.0f32, 3.0], vec![3, 4]).unwrap();

        let transposed = sp.t().unwrap();

        assert_eq!(transposed.shape(), &[4, 3]);
        assert_eq!(transposed.nnz(), 2);
        assert_eq!(transposed.indices()[0], vec![1, 0]);
        assert_eq!(transposed.indices()[1], vec![0, 2]);
        assert_eq!(transposed.values(), &[5.0, 3.0]);
    }

    #[test]
    fn test_transpose_not_2d() {
        let sp = SparseTensor::new(vec![vec![0, 1, 2]], vec![1.0f32], vec![3, 3, 3]).unwrap();

        assert!(sp.t().is_err());
    }

    // --- mul_scalar ---

    #[test]
    fn test_mul_scalar() {
        let sp =
            SparseTensor::new(vec![vec![0, 0], vec![1, 1]], vec![2.0f64, 3.0], vec![2, 2]).unwrap();

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
        let a =
            SparseTensor::new(vec![vec![0, 0], vec![0, 1]], vec![1.0f32, 2.0], vec![2, 2]).unwrap();

        // b: [0,1] -> 3.0, [1,0] -> 4.0
        let b =
            SparseTensor::new(vec![vec![0, 1], vec![1, 0]], vec![3.0, 4.0], vec![2, 2]).unwrap();

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
        let sp = SparseTensor::new(vec![vec![0, 0]], vec![1.0f32], vec![2, 3]).unwrap();

        // Dense is 4x2, but sparse inner dim is 3.
        let dense =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32; 8]), vec![4, 2], false).unwrap();

        assert!(sp.spmm(&dense).is_err());
    }

    #[test]
    fn test_debug_format() {
        let sp = SparseTensor::new(vec![vec![0, 0]], vec![1.0f32], vec![3, 3]).unwrap();

        let debug = format!("{sp:?}");
        assert!(debug.contains("SparseTensor"));
        assert!(debug.contains("nnz: 1"));
    }

    #[test]
    fn test_clone() {
        let sp = SparseTensor::new(vec![vec![0, 1]], vec![42.0f32], vec![2, 2]).unwrap();

        let sp2 = sp.clone();
        assert_eq!(sp2.values(), &[42.0]);
        assert_eq!(sp2.indices(), &[vec![0, 1]]);
        assert_eq!(sp2.shape(), &[2, 2]);
    }

    // --- CooTensor tests ---

    #[test]
    fn test_coo_coalesce_uses_tuple_key() {
        // Duplicate (0, 1) entries.
        let coo =
            CooTensor::new(vec![0, 0, 1], vec![1, 1, 0], vec![3.0f32, 4.0, 5.0], 2, 2).unwrap();

        let coalesced = coo.coalesce();
        assert!(coalesced.is_coalesced());
        assert_eq!(coalesced.nnz(), 2); // (0,1)->7, (1,0)->5

        let dense = coalesced.to_dense().unwrap();
        let d = dense.data().unwrap();
        assert!((d[1] - 7.0).abs() < 1e-6); // (0,1) = 3 + 4
        assert!((d[2] - 5.0).abs() < 1e-6); // (1,0) = 5
    }

    #[test]
    fn test_coo_from_csr_not_coalesced() {
        let csr = CsrTensor::new(vec![0, 1, 2], vec![0, 1], vec![1.0f32, 2.0], 2, 2).unwrap();

        let coo = CooTensor::from_csr(&csr);
        // Should be conservatively marked as not coalesced.
        assert!(!coo.is_coalesced());
        assert_eq!(coo.nnz(), 2);
    }

    // --- CsrTensor tests ---

    #[test]
    fn test_csr_from_coo_with_duplicates() {
        // COO with duplicate (0,0).
        let coo =
            CooTensor::new(vec![0, 0, 1], vec![0, 0, 1], vec![1.0f32, 2.0, 3.0], 2, 2).unwrap();

        let csr = CsrTensor::from_coo(&coo).unwrap();
        assert_eq!(csr.nnz(), 2); // (0,0)->3, (1,1)->3

        let dense = csr.to_dense().unwrap();
        let d = dense.data().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-6); // (0,0) = 1 + 2
        assert!((d[3] - 3.0).abs() < 1e-6); // (1,1) = 3
    }

    #[test]
    fn test_coalesce_deterministic_order() {
        // SparseTensor coalesce should produce deterministic (sorted) output.
        let sp = SparseTensor::new(
            vec![vec![1, 0], vec![0, 1], vec![0, 0]],
            vec![3.0f32, 2.0, 1.0],
            vec![2, 2],
        )
        .unwrap();

        let coalesced = sp.coalesce();
        // Should be sorted: [0,0], [0,1], [1,0].
        assert_eq!(coalesced.indices()[0], vec![0, 0]);
        assert_eq!(coalesced.indices()[1], vec![0, 1]);
        assert_eq!(coalesced.indices()[2], vec![1, 0]);
    }

    // --- 1-D, 3-D, and zero-dimension edge cases ---

    #[test]
    fn test_1d_sparse_tensor() {
        let sp = SparseTensor::new(vec![vec![1], vec![4]], vec![10.0f32, 20.0], vec![5]).unwrap();

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

    // ────────────────────────────────────────────────────────────────
    // CL-292: SemiStructuredSparseTensor (2:4) tests
    // ────────────────────────────────────────────────────────────────

    fn mk(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    #[test]
    fn semi24_compress_keeps_two_largest_magnitudes_per_group() {
        // Group 0: [1, 4, 2, 3] → keep 4 and 3 (positions 1, 3).
        // Group 1: [-5, 2, 0, 1] → keep -5 and 2 (positions 0, 1).
        let t = mk(vec![1.0, 4.0, 2.0, 3.0, -5.0, 2.0, 0.0, 1.0], vec![8]);
        let sp = SemiStructuredSparseTensor::compress(&t).unwrap();

        // Values stored in original position order.
        assert_eq!(sp.values(), &[4.0, 3.0, -5.0, 2.0]);
        // Group 0 mask = bits 1 and 3 → 0b1010 = 0xA.
        // Group 1 mask = bits 0 and 1 → 0b0011 = 0x3.
        // Packed byte = (group1 << 4) | group0 = (0x3 << 4) | 0xA = 0x3A.
        assert_eq!(sp.mask(), &[0x3A]);
        assert_eq!(sp.num_groups(), 2);
        assert_eq!(sp.group_mask(0), 0xA);
        assert_eq!(sp.group_mask(1), 0x3);
    }

    #[test]
    fn semi24_decompress_roundtrips_compressed_values() {
        // After compress → decompress, retained positions have
        // their original values and dropped positions are zero.
        let t = mk(vec![1.0, 4.0, 2.0, 3.0, -5.0, 2.0, 0.0, 1.0], vec![8]);
        let sp = SemiStructuredSparseTensor::compress(&t).unwrap();
        let dense = sp.decompress().unwrap();
        let data = dense.data().unwrap();
        // Group 0 [1,4,2,3] → kept pos 1,3 → [0,4,0,3].
        // Group 1 [-5,2,0,1] → kept pos 0,1 → [-5,2,0,0].
        assert_eq!(data, &[0.0, 4.0, 0.0, 3.0, -5.0, 2.0, 0.0, 0.0]);
        assert_eq!(dense.shape(), &[8]);
    }

    #[test]
    fn semi24_compress_decompress_preserves_shape() {
        // 2-D shape [2, 8] — 4 groups total, 2 per row.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let t = mk(data, vec![2, 8]);
        let sp = SemiStructuredSparseTensor::compress(&t).unwrap();
        assert_eq!(sp.shape(), &[2, 8]);
        let dense = sp.decompress().unwrap();
        assert_eq!(dense.shape(), &[2, 8]);
    }

    #[test]
    fn semi24_rejects_non_multiple_of_4() {
        let t = mk(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![5]);
        let result = SemiStructuredSparseTensor::compress(&t);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("multiple of 4"));
    }

    #[test]
    fn semi24_tie_breaking_prefers_lower_position() {
        // All magnitudes equal → keep positions 0 and 1 (lowest indices).
        let t = mk(vec![1.0, 1.0, 1.0, 1.0], vec![4]);
        let sp = SemiStructuredSparseTensor::compress(&t).unwrap();
        // Mask should be 0b0011 = 0x3 (positions 0 and 1).
        assert_eq!(sp.group_mask(0), 0x3);
        assert_eq!(sp.values(), &[1.0, 1.0]);
    }

    #[test]
    fn semi24_compression_ratio_is_roughly_half() {
        // For any f32 tensor multiple of 4, ratio ≈ (values*4 + mask*1) / (numel*4)
        // = (numel/2 * 4 + ceil(numel/8)) / (numel*4)
        // ≈ 0.5 + small overhead from the mask byte.
        let n = 1024usize;
        let data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let t = mk(data, vec![n]);
        let sp = SemiStructuredSparseTensor::compress(&t).unwrap();
        let ratio = sp.compression_ratio();
        // Values: 512 f32s = 2048 bytes. Mask: 128 bytes.
        // Dense: 4096 bytes. Ratio: (2048+128)/4096 = 0.53125.
        assert!(ratio > 0.5 && ratio < 0.6, "unexpected ratio: {ratio}");
    }

    #[test]
    fn semi24_zero_tensor_has_deterministic_mask() {
        // When all values are zero, the tie-breaker picks
        // positions 0 and 1 uniformly across every group.
        let t = mk(vec![0.0; 16], vec![16]);
        let sp = SemiStructuredSparseTensor::compress(&t).unwrap();
        assert_eq!(sp.values(), &[0.0; 8]);
        for g in 0..4 {
            assert_eq!(sp.group_mask(g), 0x3);
        }
    }

    #[test]
    fn semi24_negative_and_positive_by_magnitude() {
        // Group [-10, 1, -2, 3] → magnitudes [10, 1, 2, 3] →
        // top-2 are positions 0 (-10) and 3 (3). Values stored
        // in ascending position order: [-10, 3]. Mask bits 0, 3
        // → 0b1001 = 0x9.
        let t = mk(vec![-10.0, 1.0, -2.0, 3.0], vec![4]);
        let sp = SemiStructuredSparseTensor::compress(&t).unwrap();
        assert_eq!(sp.values(), &[-10.0, 3.0]);
        assert_eq!(sp.group_mask(0), 0x9);
    }

    #[test]
    fn semi24_sparse_matmul_matches_dense_matmul() {
        // a @ b where b is compressed to 2:4. The reference
        // implementation decompresses b and does the full
        // matmul, so the output should match dense @ (masked b).
        // We verify that by computing dense matmul of the
        // decompressed b and comparing.
        let a = mk(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        // b = [2, 4] with groups along the innermost dim.
        let b_data = vec![
            1.0, 4.0, 2.0, 3.0, // row 0 group → kept 4, 3
            -5.0, 2.0, 0.0, 1.0, // row 1 group → kept -5, 2
        ];
        let b_dense = mk(b_data.clone(), vec![2, 4]);
        let b_sparse = SemiStructuredSparseTensor::compress(&b_dense).unwrap();

        // Compute sparse_matmul: result = a @ decompress(b_sparse).
        let out = sparse_matmul_24(&a, &b_sparse).unwrap();
        assert_eq!(out.shape(), &[2, 4]);

        // Manual reference: a @ decompressed.
        let b_masked = b_sparse.decompress().unwrap();
        let b_m = b_masked.data().unwrap();
        // Row 0 of a @ b_masked:
        //   a[0,:] = [1, 2]
        //   b_masked = [[0,4,0,3],[-5,2,0,0]]
        //   out[0,:] = [1*0 + 2*(-5), 1*4 + 2*2, 0, 1*3 + 0] = [-10, 8, 0, 3]
        let d = out.data().unwrap();
        assert_eq!(d[0], 1.0 * b_m[0] + 2.0 * b_m[4]);
        assert_eq!(d[1], 1.0 * b_m[1] + 2.0 * b_m[5]);
        assert_eq!(d[2], 1.0 * b_m[2] + 2.0 * b_m[6]);
        assert_eq!(d[3], 1.0 * b_m[3] + 2.0 * b_m[7]);
        // Row 1: a[1,:] = [3, 4]
        assert_eq!(d[4], 3.0 * b_m[0] + 4.0 * b_m[4]);
        assert_eq!(d[5], 3.0 * b_m[1] + 4.0 * b_m[5]);
        assert_eq!(d[6], 3.0 * b_m[2] + 4.0 * b_m[6]);
        assert_eq!(d[7], 3.0 * b_m[3] + 4.0 * b_m[7]);
    }

    #[test]
    fn semi24_sparse_matmul_rejects_non_2d_a() {
        let a = mk(vec![1.0, 2.0, 3.0, 4.0], vec![4]); // 1-D
        let b_dense = mk(vec![1.0; 16], vec![4, 4]);
        let b_sparse = SemiStructuredSparseTensor::compress(&b_dense).unwrap();
        let result = sparse_matmul_24(&a, &b_sparse);
        assert!(result.is_err());
    }

    #[test]
    fn semi24_sparse_matmul_rejects_inner_dim_mismatch() {
        let a = mk(vec![1.0, 2.0, 3.0], vec![1, 3]); // k=3
        let b_dense = mk(vec![1.0; 16], vec![4, 4]); // k=4
        let b_sparse = SemiStructuredSparseTensor::compress(&b_dense).unwrap();
        let result = sparse_matmul_24(&a, &b_sparse);
        assert!(result.is_err());
    }

    #[test]
    fn semi24_compress_then_decompress_matches_apply_2_4_mask() {
        // Compressing + decompressing should yield the same result
        // as the existing `apply_2_4_mask` function (which also
        // keeps the 2 largest-magnitude elements per group).
        let t = mk(
            vec![
                0.1, 0.9, 0.3, 0.5, -0.8, 0.2, 0.7, -0.4, 1.5, -2.0, 0.1, 0.3,
            ],
            vec![12],
        );
        let sp = SemiStructuredSparseTensor::compress(&t).unwrap();
        let sp_dense = sp.decompress().unwrap();
        let mask_result = crate::pruning::apply_2_4_mask(&t).unwrap();
        assert_eq!(
            sp_dense.data().unwrap(),
            mask_result.data().unwrap(),
            "compress+decompress must match apply_2_4_mask output"
        );
    }

    #[test]
    fn semi24_f64_parity() {
        let t = Tensor::<f64>::from_storage(
            TensorStorage::cpu(vec![1.0, 4.0, 2.0, 3.0, -5.0, 2.0, 0.0, 1.0]),
            vec![8],
            false,
        )
        .unwrap();
        let sp = SemiStructuredSparseTensor::compress(&t).unwrap();
        assert_eq!(sp.values(), &[4.0, 3.0, -5.0, 2.0]);
        let dense = sp.decompress().unwrap();
        let data = dense.data().unwrap();
        assert_eq!(data, &[0.0, 4.0, 0.0, 3.0, -5.0, 2.0, 0.0, 0.0]);
    }

    // -----------------------------------------------------------------------
    // CscTensor (#619)
    // -----------------------------------------------------------------------

    #[test]
    fn csc_from_csr_roundtrip() {
        // Build a CSR matrix:
        //   [[1, 0, 2],
        //    [0, 0, 3],
        //    [4, 5, 0]]
        // CSR: row_ptrs = [0, 2, 3, 5]; col_idx = [0, 2, 2, 0, 1]; vals = [1, 2, 3, 4, 5]
        let csr = CsrTensor::new(
            vec![0, 2, 3, 5],
            vec![0, 2, 2, 0, 1],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
            3,
            3,
        )
        .unwrap();
        let csc = CscTensor::from_csr(&csr);
        assert_eq!(csc.nrows(), 3);
        assert_eq!(csc.ncols(), 3);
        assert_eq!(csc.nnz(), 5);
        // Round-trip back to CSR.
        let csr2 = csc.to_csr();
        assert_eq!(csr2.values().to_vec(), vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn csc_to_dense_matches_csr() {
        let csr = CsrTensor::new(
            vec![0, 2, 3, 5],
            vec![0, 2, 2, 0, 1],
            vec![1.0_f32, 2.0, 3.0, 4.0, 5.0],
            3,
            3,
        )
        .unwrap();
        let csc = CscTensor::from_csr(&csr);
        let dense = csc.to_dense().unwrap();
        let d = dense.data().unwrap();
        assert_eq!(d, &[1.0, 0.0, 2.0, 0.0, 0.0, 3.0, 4.0, 5.0, 0.0]);
    }

    #[test]
    fn csc_rejects_bad_col_ptrs_length() {
        let err = CscTensor::new(vec![0, 1], vec![0], vec![1.0_f32], 2, 3).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn csc_rejects_oob_row_index() {
        let err = CscTensor::new(vec![0, 1], vec![5], vec![1.0_f32], 2, 1).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    // -----------------------------------------------------------------------
    // SparseGrad (#619)
    // -----------------------------------------------------------------------

    #[test]
    fn sparse_grad_construction_validates_size() {
        // 2 indices, slab_shape [3] → expects 6 values.
        let err = SparseGrad::<f32>::new(vec![0, 2], vec![1.0; 5], vec![3]).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn sparse_grad_coalesce_sums_duplicate_indices() {
        // index 0 appears twice with slabs [1, 2] and [3, 4] → coalesced [4, 6].
        // index 1 once with [5, 6].
        let g = SparseGrad::<f32>::new(vec![0, 1, 0], vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0], vec![2])
            .unwrap();
        let c = g.coalesce();
        assert_eq!(c.indices(), &[0, 1]);
        assert_eq!(c.values(), &[4.0, 6.0, 5.0, 6.0]);
    }

    #[test]
    fn sparse_grad_apply_sgd_updates_only_affected_rows() {
        // Embedding [4, 3] init zeros; sparse grad at rows 1, 3 with lr=1.
        let mut param =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0; 12]), vec![4, 3], false)
                .unwrap();
        let grad = SparseGrad::<f32>::new(vec![1, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3])
            .unwrap();
        grad.apply_sgd(&mut param, 1.0).unwrap();
        let d = param.data().unwrap();
        // Row 0: untouched
        assert_eq!(&d[0..3], &[0.0, 0.0, 0.0]);
        // Row 1: -[1, 2, 3] = [-1, -2, -3]
        assert_eq!(&d[3..6], &[-1.0, -2.0, -3.0]);
        // Row 2: untouched
        assert_eq!(&d[6..9], &[0.0, 0.0, 0.0]);
        // Row 3: -[4, 5, 6] = [-4, -5, -6]
        assert_eq!(&d[9..12], &[-4.0, -5.0, -6.0]);
    }

    #[test]
    fn sparse_grad_apply_sgd_rejects_oob_index() {
        let mut param =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0; 6]), vec![2, 3], false)
                .unwrap();
        let grad = SparseGrad::<f32>::new(vec![5], vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let err = grad.apply_sgd(&mut param, 1.0).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn sparse_grad_apply_sgd_rejects_shape_mismatch() {
        let mut param =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0; 6]), vec![2, 3], false)
                .unwrap();
        // slab_shape [4] != param trailing [3]
        let grad = SparseGrad::<f32>::new(vec![0], vec![1.0; 4], vec![4]).unwrap();
        let err = grad.apply_sgd(&mut param, 1.0).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }
}
