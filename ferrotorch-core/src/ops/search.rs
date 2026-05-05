//! Searching and sorting tensor operations.
//!
//! - [`searchsorted`] — binary search a sorted tensor for insertion points
//! - [`bucketize`] — discretize values into bucket indices
//! - [`unique`] — return unique elements
//! - [`unique_consecutive`] — deduplicate consecutive elements

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Find insertion indices for `values` in a sorted 1-D `boundaries` tensor.
///
/// Returns a tensor of the same shape as `values` containing indices `i`
/// such that `boundaries[i-1] < value <= boundaries[i]` (right=true) or
/// `boundaries[i-1] <= value < boundaries[i]` (right=false).
///
/// Matches PyTorch's `torch.searchsorted`.
pub fn searchsorted<T: Float>(
    boundaries: &Tensor<T>,
    values: &Tensor<T>,
    right: bool,
) -> FerrotorchResult<Vec<usize>> {
    if boundaries.ndim() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "searchsorted: boundaries must be 1-D, got shape {:?}",
                boundaries.shape()
            ),
        });
    }

    if boundaries.is_cuda() || values.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "searchsorted" });
    }

    let bounds = boundaries.data()?;
    let vals = values.data_vec()?;

    let result: Vec<usize> = vals
        .iter()
        .map(|v| {
            if right {
                // Find first index where bounds[i] > v (upper_bound).
                bounds.partition_point(|b| *b <= *v)
            } else {
                // Find first index where bounds[i] >= v (lower_bound).
                bounds.partition_point(|b| *b < *v)
            }
        })
        .collect();

    Ok(result)
}

/// Discretize `input` values into buckets defined by `boundaries`.
///
/// Returns a `Vec<usize>` of bucket indices. Equivalent to
/// `searchsorted(boundaries, input, right=false)`.
///
/// Matches PyTorch's `torch.bucketize`.
pub fn bucketize<T: Float>(
    input: &Tensor<T>,
    boundaries: &Tensor<T>,
    right: bool,
) -> FerrotorchResult<Vec<usize>> {
    searchsorted(boundaries, input, right)
}

/// Return the sorted unique elements of a 1-D tensor.
///
/// Returns `(unique_values, inverse_indices, counts)` where:
/// - `unique_values` — sorted tensor of unique elements
/// - `inverse_indices` — for each input element, its index in `unique_values`
/// - `counts` — how many times each unique element appears
///
/// Matches PyTorch's `torch.unique(sorted=True, return_inverse=True, return_counts=True)`.
pub fn unique<T: Float>(
    input: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Vec<usize>, Vec<usize>)> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "unique" });
    }

    let data = input.data_vec()?;
    let n = data.len();

    if n == 0 {
        return Ok((
            Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0], false)?,
            vec![],
            vec![],
        ));
    }

    // Sort indices by value.
    let mut indices: Vec<usize> = (0..n).collect();
    indices.sort_by(|&a, &b| {
        data[a]
            .partial_cmp(&data[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Extract unique values, inverse mapping, and counts.
    let mut unique_vals: Vec<T> = Vec::new();
    let mut inverse = vec![0usize; n];
    let mut counts: Vec<usize> = Vec::new();

    let mut current_unique_idx = 0;
    unique_vals.push(data[indices[0]]);
    counts.push(0);

    for &orig_idx in &indices {
        let val = data[orig_idx];
        if val != *unique_vals.last().unwrap() {
            unique_vals.push(val);
            counts.push(0);
            current_unique_idx += 1;
        }
        inverse[orig_idx] = current_unique_idx;
        counts[current_unique_idx] += 1;
    }

    let unique_len = unique_vals.len();
    let unique_tensor =
        Tensor::from_storage(TensorStorage::cpu(unique_vals), vec![unique_len], false)?;

    Ok((unique_tensor, inverse, counts))
}

/// Remove consecutive duplicate elements from a 1-D tensor.
///
/// Returns `(output, inverse_indices, counts)` where:
/// - `output` — tensor with consecutive duplicates removed
/// - `inverse_indices` — for each input element, its index in `output`
/// - `counts` — length of each run of consecutive equal elements
///
/// Matches PyTorch's `torch.unique_consecutive`.
pub fn unique_consecutive<T: Float>(
    input: &Tensor<T>,
) -> FerrotorchResult<(Tensor<T>, Vec<usize>, Vec<usize>)> {
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "unique_consecutive",
        });
    }

    let data = input.data_vec()?;
    let n = data.len();

    if n == 0 {
        return Ok((
            Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0], false)?,
            vec![],
            vec![],
        ));
    }

    let mut output: Vec<T> = vec![data[0]];
    let mut inverse = vec![0usize; n];
    let mut counts: Vec<usize> = vec![1];

    for i in 1..n {
        if data[i] != data[i - 1] {
            output.push(data[i]);
            counts.push(1);
        } else {
            *counts.last_mut().unwrap() += 1;
        }
        inverse[i] = output.len() - 1;
    }

    let out_len = output.len();
    let output_tensor = Tensor::from_storage(TensorStorage::cpu(output), vec![out_len], false)?;

    Ok((output_tensor, inverse, counts))
}

/// Histogram — count elements in equal-width bins.
///
/// `input` is flattened. Returns a 1-D tensor of `bins` counts.
/// Elements outside `[min, max]` are clamped to the boundary bins.
///
/// Matches PyTorch's `torch.histc`.
pub fn histc<T: Float>(
    input: &Tensor<T>,
    bins: usize,
    min_val: f64,
    max_val: f64,
) -> FerrotorchResult<Tensor<T>> {
    if bins == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "histc: bins must be > 0".into(),
        });
    }
    if min_val >= max_val {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("histc: min ({min_val}) must be < max ({max_val})"),
        });
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "histc" });
    }

    let data = input.data_vec()?;
    let mut counts = vec![<T as num_traits::Zero>::zero(); bins];
    let range = max_val - min_val;
    let bin_width = range / bins as f64;

    for &v in &data {
        let f = num_traits::ToPrimitive::to_f64(&v).unwrap();
        let clamped = f.clamp(min_val, max_val - 1e-30);
        let idx = ((clamped - min_val) / bin_width) as usize;
        let idx = idx.min(bins - 1);
        counts[idx] += <T as num_traits::One>::one();
    }

    Tensor::from_storage(TensorStorage::cpu(counts), vec![bins], false)
}

/// Create coordinate grids from 1-D coordinate vectors.
///
/// Given N 1-D tensors, returns N tensors of shape `[len0, len1, ..., lenN-1]`
/// where each output tensor contains the coordinates for one axis.
///
/// Matches PyTorch's `torch.meshgrid` with `indexing='ij'`.
pub fn meshgrid<T: Float>(tensors: &[Tensor<T>]) -> FerrotorchResult<Vec<Tensor<T>>> {
    if tensors.is_empty() {
        return Ok(vec![]);
    }

    for t in tensors {
        if t.ndim() != 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "meshgrid: all inputs must be 1-D, got shape {:?}",
                    t.shape()
                ),
            });
        }
        if t.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "meshgrid" });
        }
    }

    let shapes: Vec<usize> = tensors.iter().map(|t| t.shape()[0]).collect();
    let ndim = shapes.len();
    let total: usize = shapes.iter().product();

    let mut result = Vec::with_capacity(ndim);

    for (dim, t) in tensors.iter().enumerate() {
        let data = t.data()?;
        let mut grid = Vec::with_capacity(total);

        // Stride pattern: for dimension `dim`, the value repeats every
        // `product(shapes[dim+1..])` elements and cycles every
        // `product(shapes[..dim]) * product(shapes[dim+1..])` elements.
        let inner: usize = shapes[dim + 1..].iter().product();
        let outer_stride = shapes[dim] * inner;

        for flat in 0..total {
            let coord = (flat / inner) % shapes[dim];
            grid.push(data[coord]);
        }

        // Suppress unused variable warning.
        let _ = outer_stride;

        result.push(Tensor::from_storage(
            TensorStorage::cpu(grid),
            shapes.clone(),
            false,
        )?);
    }

    Ok(result)
}

/// Return the `k` largest elements and their indices along the last dimension.
///
/// Input must be at least 1-D. Returns `(values, indices)` both with the
/// last dimension replaced by `k`.
///
/// Matches PyTorch's `torch.topk`.
pub fn topk<T: Float>(
    input: &Tensor<T>,
    k: usize,
    largest: bool,
) -> FerrotorchResult<(Tensor<T>, Vec<usize>)> {
    if input.ndim() == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "topk: input must have at least 1 dimension".into(),
        });
    }
    if input.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda { op: "topk" });
    }

    let shape = input.shape();
    let last_dim = *shape.last().unwrap();
    if k > last_dim {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("topk: k ({k}) > last dimension size ({last_dim})"),
        });
    }

    let data = input.data_vec()?;
    let outer: usize = data.len() / last_dim;

    let mut out_values = Vec::with_capacity(outer * k);
    let mut out_indices = Vec::with_capacity(outer * k);

    for o in 0..outer {
        let slice = &data[o * last_dim..(o + 1) * last_dim];
        let mut idx: Vec<usize> = (0..last_dim).collect();

        if largest {
            idx.sort_by(|&a, &b| {
                slice[b]
                    .partial_cmp(&slice[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        } else {
            idx.sort_by(|&a, &b| {
                slice[a]
                    .partial_cmp(&slice[b])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }

        for &i in &idx[..k] {
            out_values.push(slice[i]);
            out_indices.push(i);
        }
    }

    let mut out_shape = shape.to_vec();
    *out_shape.last_mut().unwrap() = k;
    let values = Tensor::from_storage(TensorStorage::cpu(out_values), out_shape, false)?;

    Ok((values, out_indices))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor_1d(data: &[f32]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
    }

    // --- searchsorted ---

    #[test]
    fn test_searchsorted_right() {
        let bounds = tensor_1d(&[1.0, 3.0, 5.0, 7.0]);
        let values = tensor_1d(&[0.0, 2.0, 3.0, 6.0, 8.0]);
        let result = searchsorted(&bounds, &values, true).unwrap();
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_searchsorted_left() {
        let bounds = tensor_1d(&[1.0, 3.0, 5.0, 7.0]);
        let values = tensor_1d(&[1.0, 3.0, 5.0, 7.0]);
        let result = searchsorted(&bounds, &values, false).unwrap();
        assert_eq!(result, vec![0, 1, 2, 3]);
    }

    #[test]
    fn test_searchsorted_empty_bounds() {
        let bounds = tensor_1d(&[]);
        let values = tensor_1d(&[1.0, 2.0]);
        let result = searchsorted(&bounds, &values, true).unwrap();
        assert_eq!(result, vec![0, 0]);
    }

    // --- bucketize ---

    #[test]
    fn test_bucketize() {
        let bounds = tensor_1d(&[0.0, 1.0, 2.0, 3.0]);
        let input = tensor_1d(&[-0.5, 0.5, 1.5, 2.5, 3.5]);
        let result = bucketize(&input, &bounds, false).unwrap();
        assert_eq!(result, vec![0, 1, 2, 3, 4]);
    }

    // --- unique ---

    #[test]
    // reason: round-trip bit-equality — unique() copies values without
    // arithmetic, and the inverse map is verified by index lookup. Both
    // sides hold the same bit pattern, so equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_unique_sorted() {
        let input = tensor_1d(&[3.0, 1.0, 2.0, 1.0, 3.0, 2.0]);
        let (unique, inverse, counts) = unique(&input).unwrap();
        let unique_data = unique.data().unwrap();
        assert_eq!(unique_data, &[1.0, 2.0, 3.0]);
        assert_eq!(counts, vec![2, 2, 2]);
        // Verify inverse: unique[inverse[i]] == input[i]
        let input_data = input.data().unwrap();
        for i in 0..6 {
            assert_eq!(unique_data[inverse[i]], input_data[i]);
        }
    }

    #[test]
    fn test_unique_empty() {
        let input = tensor_1d(&[]);
        let (unique, inverse, counts) = unique(&input).unwrap();
        assert_eq!(unique.numel(), 0);
        assert!(inverse.is_empty());
        assert!(counts.is_empty());
    }

    #[test]
    fn test_unique_all_same() {
        let input = tensor_1d(&[5.0, 5.0, 5.0]);
        let (unique, _inverse, counts) = unique(&input).unwrap();
        assert_eq!(unique.data().unwrap(), &[5.0]);
        assert_eq!(counts, vec![3]);
    }

    // --- unique_consecutive ---

    #[test]
    fn test_unique_consecutive_basic() {
        let input = tensor_1d(&[1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 1.0, 1.0]);
        let (output, inverse, counts) = unique_consecutive(&input).unwrap();
        let out_data = output.data().unwrap();
        assert_eq!(out_data, &[1.0, 2.0, 3.0, 1.0]);
        assert_eq!(counts, vec![2, 3, 1, 2]);
        assert_eq!(inverse, vec![0, 0, 1, 1, 1, 2, 3, 3]);
    }

    #[test]
    fn test_unique_consecutive_no_duplicates() {
        let input = tensor_1d(&[1.0, 2.0, 3.0]);
        let (output, _inverse, counts) = unique_consecutive(&input).unwrap();
        assert_eq!(output.data().unwrap(), &[1.0, 2.0, 3.0]);
        assert_eq!(counts, vec![1, 1, 1]);
    }

    #[test]
    fn test_unique_consecutive_empty() {
        let input = tensor_1d(&[]);
        let (output, inverse, counts) = unique_consecutive(&input).unwrap();
        assert_eq!(output.numel(), 0);
        assert!(inverse.is_empty());
        assert!(counts.is_empty());
    }

    // --- histc ---

    #[test]
    fn test_histc_basic() {
        let input = tensor_1d(&[0.5, 1.5, 2.5, 3.5, 1.5]);
        let hist = histc(&input, 4, 0.0, 4.0).unwrap();
        let data = hist.data().unwrap();
        assert_eq!(data, &[1.0, 2.0, 1.0, 1.0]);
    }

    #[test]
    fn test_histc_clamps() {
        let input = tensor_1d(&[-1.0, 5.0, 0.5]);
        let hist = histc(&input, 2, 0.0, 2.0).unwrap();
        let data = hist.data().unwrap();
        // -1.0 clamps to bin 0, 5.0 clamps to bin 1, 0.5 is bin 0
        assert_eq!(data, &[2.0, 1.0]);
    }

    // --- meshgrid ---

    #[test]
    fn test_meshgrid_2d() {
        let x = tensor_1d(&[1.0, 2.0, 3.0]);
        let y = tensor_1d(&[4.0, 5.0]);
        let grids = meshgrid(&[x, y]).unwrap();
        assert_eq!(grids.len(), 2);
        assert_eq!(grids[0].shape(), &[3, 2]);
        assert_eq!(grids[1].shape(), &[3, 2]);
        // grid_x should be [[1,1],[2,2],[3,3]]
        let gx = grids[0].data().unwrap();
        assert_eq!(gx, &[1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        // grid_y should be [[4,5],[4,5],[4,5]]
        let gy = grids[1].data().unwrap();
        assert_eq!(gy, &[4.0, 5.0, 4.0, 5.0, 4.0, 5.0]);
    }

    // --- topk ---

    #[test]
    fn test_topk_largest() {
        let input = tensor_1d(&[3.0, 1.0, 4.0, 1.0, 5.0, 9.0]);
        let (values, indices) = topk(&input, 3, true).unwrap();
        let vdata = values.data().unwrap();
        assert_eq!(vdata, &[9.0, 5.0, 4.0]);
        assert_eq!(indices, vec![5, 4, 2]);
    }

    #[test]
    fn test_topk_smallest() {
        let input = tensor_1d(&[3.0, 1.0, 4.0, 1.0, 5.0]);
        let (values, indices) = topk(&input, 2, false).unwrap();
        let vdata = values.data().unwrap();
        assert_eq!(vdata, &[1.0, 1.0]);
        assert_eq!(indices, vec![1, 3]);
    }

    #[test]
    fn test_topk_k_exceeds_dim() {
        let input = tensor_1d(&[1.0, 2.0]);
        let result = topk(&input, 5, true);
        assert!(result.is_err());
    }
}
