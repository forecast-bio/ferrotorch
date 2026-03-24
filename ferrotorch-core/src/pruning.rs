//! Weight pruning utilities for structured and unstructured sparsity.
//!
//! This module provides:
//!
//! - **2:4 Semi-Structured Pruning**: NVIDIA Ampere-style magnitude pruning that
//!   keeps exactly 2 of every 4 consecutive weights along the column axis.
//! - **Mask creation**: Binary masks for 2:4 patterns.
//! - **Gradual Pruning**: A scheduler that progressively increases sparsity during
//!   training, following the cubic schedule from Zhu & Gupta (2017).

use num_traits::{One, Zero};

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::sparse::SemiStructuredTensor;
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Prune a 2-D weight tensor to 2:4 semi-structured sparsity.
///
/// For each group of 4 consecutive elements along the column axis, the 2
/// elements with the smallest absolute magnitude are zeroed. The result is
/// returned as a [`SemiStructuredTensor`] in compressed form.
///
/// This is equivalent to `SemiStructuredTensor::from_dense`, but named
/// explicitly as a pruning operation for clarity in training pipelines.
///
/// # Errors
///
/// Returns an error if the tensor is not 2-D or `ncols` is not a multiple of 4.
pub fn magnitude_prune_2_4<T: Float>(tensor: &Tensor<T>) -> FerrotorchResult<SemiStructuredTensor<T>> {
    SemiStructuredTensor::from_dense(tensor)
}

/// Create a binary 0/1 mask encoding the 2:4 sparsity pattern.
///
/// For each group of 4 consecutive elements along the column axis, the 2
/// positions with the largest absolute magnitude get a mask value of `1.0`
/// and the others get `0.0`.
///
/// The returned mask has the same shape as the input tensor. Multiply it
/// element-wise with the original tensor to obtain the pruned tensor.
///
/// # Errors
///
/// Returns an error if the tensor is not 2-D or `ncols` is not a multiple of 4.
pub fn create_2_4_mask<T: Float>(tensor: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if tensor.ndim() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "create_2_4_mask requires a 2-D tensor, got {}-D",
                tensor.ndim()
            ),
        });
    }

    let data = tensor.data()?;
    let nrows = tensor.shape()[0];
    let ncols = tensor.shape()[1];

    if ncols % 4 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "create_2_4_mask: ncols ({}) must be a multiple of 4",
                ncols
            ),
        });
    }

    let groups_per_row = ncols / 4;
    let mut mask = vec![<T as Zero>::zero(); nrows * ncols];

    for i in 0..nrows {
        for g in 0..groups_per_row {
            let base = i * ncols + g * 4;
            let group = [data[base], data[base + 1], data[base + 2], data[base + 3]];

            // Find top-2 by magnitude, tie-break by lower index.
            let mut order: [usize; 4] = [0, 1, 2, 3];
            order.sort_by(|&a, &b| {
                let mag_a = group[a].abs();
                let mag_b = group[b].abs();
                mag_b.partial_cmp(&mag_a).unwrap().then(a.cmp(&b))
            });

            mask[base + order[0]] = <T as One>::one();
            mask[base + order[1]] = <T as One>::one();
        }
    }

    Tensor::from_storage(TensorStorage::cpu(mask), vec![nrows, ncols], false)
}

/// Apply a pre-computed binary mask to a tensor element-wise.
///
/// Returns `tensor * mask` — elements where `mask == 0` become zero. This is
/// used during training to enforce the sparsity pattern on each forward pass
/// (straight-through estimator: gradients flow through the mask unchanged).
///
/// # Errors
///
/// Returns an error if the tensors have different shapes.
pub fn apply_2_4_mask<T: Float>(tensor: &Tensor<T>, mask: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if tensor.shape() != mask.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "apply_2_4_mask: tensor shape {:?} != mask shape {:?}",
                tensor.shape(),
                mask.shape()
            ),
        });
    }

    let t_data = tensor.data()?;
    let m_data = mask.data()?;

    let result: Vec<T> = t_data
        .iter()
        .zip(m_data.iter())
        .map(|(&t, &m)| t * m)
        .collect();

    Tensor::from_storage(TensorStorage::cpu(result), tensor.shape().to_vec(), false)
}

/// Gradual magnitude pruner following the cubic schedule from
/// Zhu & Gupta, "To prune, or not to prune" (2017).
///
/// Sparsity increases from `initial_sparsity` to `final_sparsity` over the
/// interval `[begin_step, end_step)` according to:
///
/// ```text
/// s_t = s_f + (s_i - s_f) * (1 - (t - t_0) / (t_n - t_0))^3
/// ```
///
/// where `s_i` is `initial_sparsity`, `s_f` is `final_sparsity`, `t_0` is
/// `begin_step`, `t_n` is `end_step`, and `t` is the current step.
///
/// Every `frequency` steps the mask is recomputed based on the current
/// weight magnitudes at the scheduled sparsity level.
///
/// # Usage
///
/// ```ignore
/// let mut pruner = GradualPruner::new(0.0, 0.5, 0, 1000, 100);
/// for step in 0..2000 {
///     if pruner.should_prune(step) {
///         let mask = pruner.compute_mask(&weight, step)?;
///         weight = apply_mask(&weight, &mask)?;
///     }
///     // ... training step ...
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GradualPruner {
    /// Sparsity at `begin_step`.
    initial_sparsity: f64,
    /// Target sparsity at `end_step`.
    final_sparsity: f64,
    /// First step at which pruning begins.
    begin_step: u64,
    /// Step at which target sparsity is reached (exclusive).
    end_step: u64,
    /// Re-compute the mask every this many steps within [begin_step, end_step).
    frequency: u64,
}

impl GradualPruner {
    /// Create a new gradual pruner.
    ///
    /// # Panics
    ///
    /// Panics if `begin_step >= end_step`, `frequency == 0`, or sparsity values
    /// are not in `[0.0, 1.0]`.
    pub fn new(
        initial_sparsity: f64,
        final_sparsity: f64,
        begin_step: u64,
        end_step: u64,
        frequency: u64,
    ) -> Self {
        assert!(
            begin_step < end_step,
            "GradualPruner: begin_step ({}) must be < end_step ({})",
            begin_step,
            end_step
        );
        assert!(frequency > 0, "GradualPruner: frequency must be > 0");
        assert!(
            (0.0..=1.0).contains(&initial_sparsity),
            "GradualPruner: initial_sparsity must be in [0, 1], got {}",
            initial_sparsity
        );
        assert!(
            (0.0..=1.0).contains(&final_sparsity),
            "GradualPruner: final_sparsity must be in [0, 1], got {}",
            final_sparsity
        );

        Self {
            initial_sparsity,
            final_sparsity,
            begin_step,
            end_step,
            frequency,
        }
    }

    /// Whether the mask should be recomputed at this step.
    ///
    /// Returns `true` if `step` falls within `[begin_step, end_step)` and is
    /// aligned to `frequency`, or if `step == begin_step`.
    pub fn should_prune(&self, step: u64) -> bool {
        if step < self.begin_step || step >= self.end_step {
            return false;
        }
        let offset = step - self.begin_step;
        offset % self.frequency == 0
    }

    /// Compute the sparsity level for the current step using the cubic schedule.
    ///
    /// Before `begin_step`, returns `initial_sparsity`.
    /// After `end_step`, returns `final_sparsity`.
    pub fn sparsity_at(&self, step: u64) -> f64 {
        if step <= self.begin_step {
            return self.initial_sparsity;
        }
        if step >= self.end_step {
            return self.final_sparsity;
        }

        let progress =
            (step - self.begin_step) as f64 / (self.end_step - self.begin_step) as f64;
        let decay = (1.0 - progress).powi(3);
        self.final_sparsity + (self.initial_sparsity - self.final_sparsity) * decay
    }

    /// Compute a magnitude-based pruning mask at the scheduled sparsity level.
    ///
    /// The mask is a tensor of the same shape as `weights`, with `1.0` for kept
    /// elements and `0.0` for pruned elements. The fraction of zeros equals
    /// `sparsity_at(step)` (rounded to the nearest integer count).
    ///
    /// Elements are ranked by absolute magnitude; the smallest are pruned first.
    /// Ties are broken by flat index (lower index is kept).
    pub fn compute_mask<T: Float>(&self, weights: &Tensor<T>, step: u64) -> FerrotorchResult<Tensor<T>> {
        let sparsity = self.sparsity_at(step);
        let data = weights.data()?;
        let n = data.len();
        let n_prune = ((n as f64) * sparsity).round() as usize;

        if n_prune == 0 {
            // No pruning — all ones.
            return Tensor::from_storage(
                TensorStorage::cpu(vec![<T as One>::one(); n]),
                weights.shape().to_vec(),
                false,
            );
        }

        if n_prune >= n {
            // Full pruning — all zeros.
            return Tensor::from_storage(
                TensorStorage::cpu(vec![<T as Zero>::zero(); n]),
                weights.shape().to_vec(),
                false,
            );
        }

        // Rank elements by magnitude ascending (smallest first), tie-break by index.
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            let mag_a = data[a].abs();
            let mag_b = data[b].abs();
            mag_a.partial_cmp(&mag_b).unwrap().then(a.cmp(&b))
        });

        let mut mask = vec![<T as One>::one(); n];
        for &idx in indices.iter().take(n_prune) {
            mask[idx] = <T as Zero>::zero();
        }

        Tensor::from_storage(TensorStorage::cpu(mask), weights.shape().to_vec(), false)
    }

    /// The initial sparsity level.
    #[inline]
    pub fn initial_sparsity(&self) -> f64 {
        self.initial_sparsity
    }

    /// The target final sparsity level.
    #[inline]
    pub fn final_sparsity(&self) -> f64 {
        self.final_sparsity
    }

    /// The step at which pruning begins.
    #[inline]
    pub fn begin_step(&self) -> u64 {
        self.begin_step
    }

    /// The step at which target sparsity is reached.
    #[inline]
    pub fn end_step(&self) -> u64 {
        self.end_step
    }

    /// How often the mask is recomputed.
    #[inline]
    pub fn frequency(&self) -> u64 {
        self.frequency
    }
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
    }

    // --- magnitude_prune_2_4 ---

    #[test]
    fn test_magnitude_prune_2_4_basic() {
        let t = make_tensor(vec![1.0, 3.0, 0.5, 2.0], vec![1, 4]);
        let ss = magnitude_prune_2_4(&t).unwrap();
        assert_eq!(ss.shape(), [1, 4]);
        assert_eq!(ss.nnz(), 2);
        // Kept: indices 1 (3.0) and 3 (2.0).
        assert_eq!(ss.values(), &[3.0, 2.0]);
    }

    #[test]
    fn test_magnitude_prune_2_4_not_2d() {
        let t = make_tensor(vec![1.0; 8], vec![2, 2, 2]);
        assert!(magnitude_prune_2_4(&t).is_err());
    }

    #[test]
    fn test_magnitude_prune_2_4_not_multiple_of_4() {
        let t = make_tensor(vec![1.0; 6], vec![2, 3]);
        assert!(magnitude_prune_2_4(&t).is_err());
    }

    // --- create_2_4_mask ---

    #[test]
    fn test_create_2_4_mask_basic() {
        let t = make_tensor(vec![1.0, 3.0, 0.5, 2.0], vec![1, 4]);
        let mask = create_2_4_mask(&t).unwrap();
        let d = mask.data().unwrap();

        // Keep indices 1 (3.0) and 3 (2.0).
        assert_eq!(d[0], 0.0);
        assert_eq!(d[1], 1.0);
        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], 1.0);
    }

    #[test]
    fn test_create_2_4_mask_ties() {
        // All equal: keep lowest indices 0, 1.
        let t = make_tensor(vec![5.0; 4], vec![1, 4]);
        let mask = create_2_4_mask(&t).unwrap();
        let d = mask.data().unwrap();
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 1.0);
        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], 0.0);
    }

    #[test]
    fn test_create_2_4_mask_multi_row() {
        #[rustfmt::skip]
        let t = make_tensor(vec![
            10.0, 1.0, 2.0, 9.0,
             5.0, 6.0, 1.0, 2.0,
        ], vec![2, 4]);

        let mask = create_2_4_mask(&t).unwrap();
        let d = mask.data().unwrap();

        // Row 0: keep 10.0 (idx 0) and 9.0 (idx 3).
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 0.0);
        assert_eq!(d[2], 0.0);
        assert_eq!(d[3], 1.0);

        // Row 1: keep 5.0 (idx 0) and 6.0 (idx 1).
        assert_eq!(d[4], 1.0);
        assert_eq!(d[5], 1.0);
        assert_eq!(d[6], 0.0);
        assert_eq!(d[7], 0.0);
    }

    #[test]
    fn test_create_2_4_mask_not_2d() {
        let t = make_tensor(vec![1.0; 4], vec![4]);
        assert!(create_2_4_mask(&t).is_err());
    }

    #[test]
    fn test_create_2_4_mask_not_multiple_of_4() {
        let t = make_tensor(vec![1.0; 6], vec![2, 3]);
        assert!(create_2_4_mask(&t).is_err());
    }

    // --- apply_2_4_mask ---

    #[test]
    fn test_apply_2_4_mask_basic() {
        let t = make_tensor(vec![1.0, 3.0, 0.5, 2.0], vec![1, 4]);
        let mask = create_2_4_mask(&t).unwrap();
        let pruned = apply_2_4_mask(&t, &mask).unwrap();
        let d = pruned.data().unwrap();

        assert_eq!(d[0], 0.0);  // pruned
        assert_eq!(d[1], 3.0);  // kept
        assert_eq!(d[2], 0.0);  // pruned
        assert_eq!(d[3], 2.0);  // kept
    }

    #[test]
    fn test_apply_2_4_mask_shape_mismatch() {
        let t = make_tensor(vec![1.0; 4], vec![1, 4]);
        let m = make_tensor(vec![1.0; 8], vec![2, 4]);
        assert!(apply_2_4_mask(&t, &m).is_err());
    }

    #[test]
    fn test_apply_mask_all_ones() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let m = make_tensor(vec![1.0; 4], vec![2, 2]);
        let result = apply_2_4_mask(&t, &m).unwrap();
        let d = result.data().unwrap();
        assert_eq!(d, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_apply_mask_all_zeros() {
        let t = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let m = make_tensor(vec![0.0; 4], vec![2, 2]);
        let result = apply_2_4_mask(&t, &m).unwrap();
        let d = result.data().unwrap();
        assert_eq!(d, &[0.0, 0.0, 0.0, 0.0]);
    }

    // --- apply_2_4_mask round-trip with SemiStructured ---

    #[test]
    fn test_mask_matches_semi_structured() {
        // Verify that masking then dense-reading matches SemiStructured to_dense.
        #[rustfmt::skip]
        let data = vec![
            4.0f32, 1.0, 3.0, 2.0,
            1.0,    5.0, 6.0, 2.0,
        ];
        let t = make_tensor(data, vec![2, 4]);

        let mask = create_2_4_mask(&t).unwrap();
        let masked = apply_2_4_mask(&t, &mask).unwrap();

        let ss = magnitude_prune_2_4(&t).unwrap();
        let ss_dense = ss.to_dense().unwrap();

        let a = masked.data().unwrap();
        let b = ss_dense.data().unwrap();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "mask vs semi-structured mismatch: {} vs {}", x, y);
        }
    }

    // --- GradualPruner ---

    #[test]
    fn test_gradual_pruner_cubic_schedule() {
        let pruner = GradualPruner::new(0.0, 0.5, 0, 1000, 100);

        // At step 0: initial_sparsity = 0.0
        assert!((pruner.sparsity_at(0) - 0.0).abs() < 1e-10);

        // At step 1000: final_sparsity = 0.5
        assert!((pruner.sparsity_at(1000) - 0.5).abs() < 1e-10);

        // At step 500 (halfway): s = 0.5 + (0.0 - 0.5) * (1 - 0.5)^3 = 0.5 - 0.5*0.125 = 0.4375
        assert!((pruner.sparsity_at(500) - 0.4375).abs() < 1e-10);

        // Monotonically increasing (since initial < final).
        let mut prev = pruner.sparsity_at(0);
        for step in (100..=1000).step_by(100) {
            let curr = pruner.sparsity_at(step);
            assert!(curr >= prev, "sparsity should increase: {} < {} at step {}", curr, prev, step);
            prev = curr;
        }
    }

    #[test]
    fn test_gradual_pruner_should_prune() {
        let pruner = GradualPruner::new(0.0, 0.5, 100, 500, 50);

        // Before begin: no.
        assert!(!pruner.should_prune(0));
        assert!(!pruner.should_prune(50));
        assert!(!pruner.should_prune(99));

        // At begin: yes.
        assert!(pruner.should_prune(100));

        // Not aligned: no.
        assert!(!pruner.should_prune(110));
        assert!(!pruner.should_prune(125));

        // Aligned: yes.
        assert!(pruner.should_prune(150));
        assert!(pruner.should_prune(200));
        assert!(pruner.should_prune(450));

        // At end: no (exclusive).
        assert!(!pruner.should_prune(500));
        assert!(!pruner.should_prune(600));
    }

    #[test]
    fn test_gradual_pruner_compute_mask() {
        let pruner = GradualPruner::new(0.0, 0.5, 0, 100, 10);

        let weights = make_tensor(vec![4.0, 3.0, 2.0, 1.0, 0.5, 0.1, 5.0, 6.0], vec![2, 4]);

        // At step 0: sparsity = 0.0 -> all ones.
        let mask0 = pruner.compute_mask(&weights, 0).unwrap();
        let d0 = mask0.data().unwrap();
        assert!(d0.iter().all(|&x| x == 1.0));

        // At step 100: sparsity = 0.5 -> half zeros.
        let mask100 = pruner.compute_mask(&weights, 100).unwrap();
        let d100 = mask100.data().unwrap();
        let n_zeros = d100.iter().filter(|&&x| x == 0.0).count();
        assert_eq!(n_zeros, 4); // 50% of 8

        // The 4 smallest magnitudes should be pruned: 0.1, 0.5, 1.0, 2.0
        // So indices 5, 4, 3, 2 should be zero.
        assert_eq!(d100[5], 0.0); // 0.1
        assert_eq!(d100[4], 0.0); // 0.5
        assert_eq!(d100[3], 0.0); // 1.0
        assert_eq!(d100[2], 0.0); // 2.0
        // Kept: indices 0 (4.0), 1 (3.0), 6 (5.0), 7 (6.0).
        assert_eq!(d100[0], 1.0);
        assert_eq!(d100[1], 1.0);
        assert_eq!(d100[6], 1.0);
        assert_eq!(d100[7], 1.0);
    }

    #[test]
    fn test_gradual_pruner_full_sparsity() {
        let pruner = GradualPruner::new(0.0, 1.0, 0, 100, 10);
        let weights = make_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![1, 4]);

        // At step 100: sparsity = 1.0 -> all zeros.
        let mask = pruner.compute_mask(&weights, 100).unwrap();
        let d = mask.data().unwrap();
        assert!(d.iter().all(|&x| x == 0.0));
    }

    #[test]
    fn test_gradual_pruner_before_begin() {
        let pruner = GradualPruner::new(0.0, 0.5, 100, 200, 10);

        assert!((pruner.sparsity_at(0) - 0.0).abs() < 1e-10);
        assert!((pruner.sparsity_at(50) - 0.0).abs() < 1e-10);
        assert!((pruner.sparsity_at(100) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_gradual_pruner_after_end() {
        let pruner = GradualPruner::new(0.0, 0.5, 0, 100, 10);

        assert!((pruner.sparsity_at(100) - 0.5).abs() < 1e-10);
        assert!((pruner.sparsity_at(200) - 0.5).abs() < 1e-10);
        assert!((pruner.sparsity_at(1000) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gradual_pruner_accessors() {
        let pruner = GradualPruner::new(0.1, 0.9, 10, 500, 25);

        assert!((pruner.initial_sparsity() - 0.1).abs() < 1e-10);
        assert!((pruner.final_sparsity() - 0.9).abs() < 1e-10);
        assert_eq!(pruner.begin_step(), 10);
        assert_eq!(pruner.end_step(), 500);
        assert_eq!(pruner.frequency(), 25);
    }

    #[test]
    #[should_panic(expected = "begin_step")]
    fn test_gradual_pruner_invalid_steps() {
        GradualPruner::new(0.0, 0.5, 100, 100, 10);
    }

    #[test]
    #[should_panic(expected = "frequency")]
    fn test_gradual_pruner_zero_frequency() {
        GradualPruner::new(0.0, 0.5, 0, 100, 0);
    }

    #[test]
    #[should_panic(expected = "initial_sparsity")]
    fn test_gradual_pruner_invalid_initial_sparsity() {
        GradualPruner::new(-0.1, 0.5, 0, 100, 10);
    }

    #[test]
    #[should_panic(expected = "final_sparsity")]
    fn test_gradual_pruner_invalid_final_sparsity() {
        GradualPruner::new(0.0, 1.5, 0, 100, 10);
    }

    #[test]
    fn test_gradual_pruner_decreasing_sparsity() {
        // Unusual but valid: go from high to low sparsity (densification).
        let pruner = GradualPruner::new(0.9, 0.1, 0, 100, 10);

        assert!((pruner.sparsity_at(0) - 0.9).abs() < 1e-10);
        assert!((pruner.sparsity_at(100) - 0.1).abs() < 1e-10);

        // Monotonically decreasing.
        let mut prev = pruner.sparsity_at(0);
        for step in (10..=100).step_by(10) {
            let curr = pruner.sparsity_at(step);
            assert!(curr <= prev, "sparsity should decrease: {} > {} at step {}", curr, prev, step);
            prev = curr;
        }
    }

    #[test]
    fn test_gradual_pruner_debug_clone() {
        let pruner = GradualPruner::new(0.0, 0.5, 0, 100, 10);
        let debug = format!("{pruner:?}");
        assert!(debug.contains("GradualPruner"));

        let cloned = pruner.clone();
        assert!((cloned.initial_sparsity() - 0.0).abs() < 1e-10);
        assert!((cloned.final_sparsity() - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_gradual_pruner_mask_intermediate_sparsity() {
        let pruner = GradualPruner::new(0.0, 0.75, 0, 100, 10);

        // Create a tensor with 8 elements.
        let weights = make_tensor(
            vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0],
            vec![1, 8],
        );

        // At step 100: sparsity = 0.75 -> 6 pruned out of 8.
        let mask = pruner.compute_mask(&weights, 100).unwrap();
        let d = mask.data().unwrap();
        let n_zeros = d.iter().filter(|&&x| x == 0.0).count();
        assert_eq!(n_zeros, 6);

        // The 2 kept should be the largest: 8.0 (idx 0) and 7.0 (idx 1).
        assert_eq!(d[0], 1.0);
        assert_eq!(d[1], 1.0);
    }
}
