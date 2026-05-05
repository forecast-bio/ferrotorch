use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Unstructured magnitude pruning: zero out the smallest weights.
///
/// Given a weight tensor and a sparsity fraction in `[0, 1)`, computes
/// a magnitude threshold and returns a new tensor with the smallest
/// `sparsity * numel` elements set to zero.
///
/// # Arguments
///
/// * `weights` - The weight tensor to prune.
/// * `sparsity` - Fraction of elements to zero out (e.g. 0.5 for 50% sparsity).
///
/// # Returns
///
/// A new tensor with the same shape and `requires_grad` as the input,
/// with the smallest-magnitude elements zeroed.
pub fn magnitude_prune<T: Float>(
    weights: &Tensor<T>,
    sparsity: f64,
) -> FerrotorchResult<Tensor<T>> {
    if !(0.0..1.0).contains(&sparsity) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("sparsity must be in [0, 1), got {sparsity}"),
        });
    }

    let data = weights.data()?;
    let numel = data.len();
    let n_prune = ((numel as f64) * sparsity).round() as usize;

    if n_prune == 0 {
        // Nothing to prune: return a copy.
        return Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            weights.shape().to_vec(),
            weights.requires_grad(),
        );
    }

    // Sort magnitudes to find the threshold.
    // Use unwrap_or(Ordering::Equal) to handle NaN without panicking.
    let mut magnitudes: Vec<T> = data.iter().map(|&v| v.abs()).collect();
    magnitudes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let threshold = magnitudes[n_prune - 1];

    let pruned: Vec<T> = data
        .iter()
        .map(|&v| {
            if v.abs() <= threshold {
                <T as num_traits::Zero>::zero()
            } else {
                v
            }
        })
        .collect();

    Tensor::from_storage(
        TensorStorage::cpu(pruned),
        weights.shape().to_vec(),
        weights.requires_grad(),
    )
}

/// Apply 2:4 structured sparsity mask.
///
/// For every group of 4 contiguous elements, keeps the 2 with the largest
/// magnitude and zeros the other 2. If the tensor length is not a multiple
/// of 4, the trailing elements are left unchanged.
///
/// The output tensor preserves the input's `requires_grad` flag.
///
/// # Arguments
///
/// * `weights` - The weight tensor to apply the mask to.
///
/// # Returns
///
/// A new tensor with 2:4 structured sparsity applied.
pub fn apply_2_4_mask<T: Float>(weights: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let data = weights.data()?;
    let mut result = data.to_vec();

    let groups = result.len() / 4;
    for g in 0..groups {
        let base = g * 4;
        let group = &mut result[base..base + 4];

        // Find indices of the 2 smallest-magnitude elements and zero them.
        // Use unwrap_or(Ordering::Equal) to handle NaN without panicking.
        let mut idx_mag: Vec<(usize, T)> = group
            .iter()
            .enumerate()
            .map(|(i, &v)| (i, v.abs()))
            .collect();
        idx_mag.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        // Zero the 2 smallest.
        group[idx_mag[0].0] = <T as num_traits::Zero>::zero();
        group[idx_mag[1].0] = <T as num_traits::Zero>::zero();
    }

    Tensor::from_storage(
        TensorStorage::cpu(result),
        weights.shape().to_vec(),
        weights.requires_grad(),
    )
}

/// Compute the sparsity ratio of a tensor: fraction of exact zeros.
pub fn sparsity_ratio<T: Float>(tensor: &Tensor<T>) -> FerrotorchResult<f64> {
    let data = tensor.data()?;
    let zeros = data
        .iter()
        .filter(|&&v| v == <T as num_traits::Zero>::zero())
        .count();
    Ok(zeros as f64 / data.len() as f64)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    fn make_tensor_rg(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, true).unwrap()
    }

    // --- magnitude_prune ---

    #[test]
    // reason: pruning is select-or-zero — kept slots hold the exact input
    // bit pattern (no arithmetic), pruned slots hold the exact zero bit
    // pattern. Equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_magnitude_prune_50_percent() {
        let t = make_tensor(vec![1.0, -4.0, 2.0, -3.0], vec![4]);
        let pruned = magnitude_prune(&t, 0.5).unwrap();
        let d = pruned.data().unwrap();

        // 50% of 4 = 2 elements pruned. Sorted magnitudes: [1, 2, 3, 4].
        // threshold = magnitude[1] = 2.0. Elements with |v| <= 2 are zeroed.
        assert_eq!(d[0], 0.0); // |1| <= 2
        assert_eq!(d[1], -4.0); // |4| > 2
        assert_eq!(d[2], 0.0); // |2| <= 2
        assert_eq!(d[3], -3.0); // |3| > 2
    }

    #[test]
    fn test_magnitude_prune_zero_sparsity() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let pruned = magnitude_prune(&t, 0.0).unwrap();
        let d = pruned.data().unwrap();
        assert_eq!(d, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_magnitude_prune_invalid_sparsity() {
        let t = make_tensor(vec![1.0], vec![1]);
        assert!(magnitude_prune(&t, 1.0).is_err());
        assert!(magnitude_prune(&t, -0.1).is_err());
    }

    // --- NaN edge case for pruning (Issue 11) ---

    #[test]
    fn test_magnitude_prune_nan_no_panic() {
        let t = make_tensor(vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0, 4.0], vec![6]);
        // Should not panic even with NaN values.
        let result = magnitude_prune(&t, 0.5);
        assert!(result.is_ok());
    }

    #[test]
    fn test_apply_2_4_mask_nan_no_panic() {
        let t = make_tensor(
            vec![1.0, f32::NAN, 3.0, f32::NAN, 2.0, 4.0, 0.5, 0.1],
            vec![8],
        );
        // Should not panic even with NaN values.
        let result = apply_2_4_mask(&t);
        assert!(result.is_ok());
    }

    // --- apply_2_4_mask ---

    #[test]
    // reason: 2:4 masking is select-or-zero — kept slots hold the exact
    // input bit pattern (no arithmetic), pruned slots hold exact zero. The
    // 0.9 and 0.8 literals on the RHS produce the same f32 bit pattern as
    // the corresponding input literals, so equality is the right check.
    #[allow(clippy::float_cmp)]
    fn test_apply_2_4_mask_basic() {
        let t = make_tensor(vec![1.0, -4.0, 2.0, -3.0, 0.5, 0.1, 0.9, 0.8], vec![8]);
        let masked = apply_2_4_mask(&t).unwrap();
        let d = masked.data().unwrap();

        // Group 0: [1, -4, 2, -3]. Magnitudes: [1, 4, 2, 3].
        // Smallest two: indices 0 (mag 1) and 2 (mag 2) -> zeroed.
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], -4.0);
        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], -3.0);

        // Group 1: [0.5, 0.1, 0.9, 0.8]. Magnitudes: [0.5, 0.1, 0.9, 0.8].
        // Smallest two: indices 1 (mag 0.1) and 0 (mag 0.5) -> zeroed.
        assert_eq!(d[4], 0.0);
        assert_eq!(d[5], 0.0);
        assert_eq!(d[6], 0.9);
        assert_eq!(d[7], 0.8);
    }

    #[test]
    fn test_apply_2_4_mask_preserves_requires_grad() {
        let t = make_tensor_rg(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        assert!(t.requires_grad());

        let masked = apply_2_4_mask(&t).unwrap();
        assert!(
            masked.requires_grad(),
            "apply_2_4_mask must propagate requires_grad"
        );
    }

    // --- sparsity_ratio ---

    #[test]
    fn test_sparsity_ratio() {
        let t = make_tensor(vec![0.0, 1.0, 0.0, 2.0], vec![4]);
        let ratio = sparsity_ratio(&t).unwrap();
        assert!((ratio - 0.5).abs() < 1e-10);
    }
}
