/// A sampler produces a sequence of indices for a `DataLoader` to fetch.
pub trait Sampler: Send + Sync {
    /// Return indices for one epoch.
    fn indices(&self, epoch: usize) -> Vec<usize>;

    /// Total number of samples.
    fn len(&self) -> usize;

    /// Whether the sampler produces zero indices.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Fisher-Yates shuffle with a deterministic xorshift64 PRNG.
///
/// This is the shared shuffle primitive used by [`RandomSampler`] and
/// [`DistributedSampler`].
pub fn shuffle_with_seed(indices: &mut [usize], seed: u64) {
    let mut state = seed;
    if state == 0 {
        state = 0xdeadbeefcafe;
    }
    for i in (1..indices.len()).rev() {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let j = (state as usize) % (i + 1);
        indices.swap(i, j);
    }
}

/// Yields indices in order: 0, 1, 2, ..., n-1.
#[derive(Debug, Clone)]
pub struct SequentialSampler {
    size: usize,
}

impl SequentialSampler {
    pub fn new(size: usize) -> Self {
        Self { size }
    }
}

impl Sampler for SequentialSampler {
    fn indices(&self, _epoch: usize) -> Vec<usize> {
        (0..self.size).collect()
    }

    fn len(&self) -> usize {
        self.size
    }
}

/// Yields indices in a random permutation, seeded by epoch for reproducibility.
#[derive(Debug, Clone)]
pub struct RandomSampler {
    size: usize,
    seed: u64,
}

impl RandomSampler {
    pub fn new(size: usize, seed: u64) -> Self {
        Self { size, seed }
    }
}

impl Sampler for RandomSampler {
    fn indices(&self, epoch: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.size).collect();
        let effective_seed = self.seed ^ (epoch as u64).wrapping_mul(0x9e3779b97f4a7c15);
        shuffle_with_seed(&mut indices, effective_seed);
        indices
    }

    fn len(&self) -> usize {
        self.size
    }
}

/// Sampler that partitions indices across distributed ranks.
///
/// Each rank gets a non-overlapping subset of indices, ensuring all
/// ranks process different data. Supports shuffling with epoch-dependent
/// seeding for reproducibility.
///
/// The total size is padded to be evenly divisible by `num_replicas`
/// so every rank processes the same number of samples (matching PyTorch's
/// `DistributedSampler` behavior).
#[derive(Debug, Clone)]
pub struct DistributedSampler {
    num_samples: usize,
    num_replicas: usize,
    rank: usize,
    shuffle: bool,
    seed: u64,
}

impl DistributedSampler {
    /// Create a new `DistributedSampler`.
    ///
    /// # Panics
    ///
    /// Panics if `rank >= num_replicas`.
    pub fn new(num_samples: usize, num_replicas: usize, rank: usize) -> Self {
        assert!(rank < num_replicas, "rank must be < num_replicas");
        Self {
            num_samples,
            num_replicas,
            rank,
            shuffle: true,
            seed: 0,
        }
    }

    /// Enable or disable shuffling (default: enabled).
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the base seed for shuffling.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }
}

impl Sampler for DistributedSampler {
    fn indices(&self, epoch: usize) -> Vec<usize> {
        let mut indices: Vec<usize> = (0..self.num_samples).collect();

        if self.shuffle {
            let effective_seed = self.seed ^ (epoch as u64).wrapping_mul(0x9e3779b97f4a7c15);
            shuffle_with_seed(&mut indices, effective_seed);
        }

        // Pad to be evenly divisible by num_replicas.
        let total_size = self.num_samples.div_ceil(self.num_replicas) * self.num_replicas;
        while indices.len() < total_size {
            let wrap_idx = indices.len() - self.num_samples;
            indices.push(indices[wrap_idx]);
        }

        // Subsample for this rank: interleaved partitioning.
        indices
            .into_iter()
            .skip(self.rank)
            .step_by(self.num_replicas)
            .collect()
    }

    fn len(&self) -> usize {
        self.num_samples.div_ceil(self.num_replicas)
    }
}

// ---------------------------------------------------------------------------
// WeightedRandomSampler
// ---------------------------------------------------------------------------

/// Samples indices according to given weights, with or without replacement.
///
/// Matches PyTorch's `torch.utils.data.WeightedRandomSampler`.
///
/// Uses a deterministic xorshift64 PRNG seeded by `(base_seed XOR epoch)` for
/// reproducibility. Weights are interpreted as un-normalized probabilities:
/// the chance of drawing index `i` is `weights[i] / sum(weights)`.
#[derive(Debug, Clone)]
pub struct WeightedRandomSampler {
    weights: Vec<f64>,
    num_samples: usize,
    replacement: bool,
    seed: u64,
}

impl WeightedRandomSampler {
    /// Create a new `WeightedRandomSampler`.
    ///
    /// # Arguments
    ///
    /// * `weights` — per-element sampling weight (must be non-negative, at
    ///   least one must be positive).
    /// * `num_samples` — how many indices to draw per epoch.
    /// * `replacement` — if `true`, indices may repeat; if `false`, sampling
    ///   is without replacement (and `num_samples` must be ≤ number of
    ///   non-zero-weight elements).
    /// * `seed` — base seed for the PRNG.
    ///
    /// # Panics
    ///
    /// Panics if `weights` is empty, any weight is negative, all weights are
    /// zero, or `!replacement && num_samples > weights.len()`.
    pub fn new(weights: Vec<f64>, num_samples: usize, replacement: bool, seed: u64) -> Self {
        assert!(
            !weights.is_empty(),
            "WeightedRandomSampler: weights must not be empty"
        );
        assert!(
            weights.iter().all(|&w| w >= 0.0),
            "WeightedRandomSampler: weights must be non-negative"
        );
        let total: f64 = weights.iter().sum();
        assert!(
            total > 0.0,
            "WeightedRandomSampler: at least one weight must be positive"
        );
        if !replacement {
            assert!(
                num_samples <= weights.len(),
                "WeightedRandomSampler: num_samples ({num_samples}) > population ({}) with replacement=false",
                weights.len()
            );
        }
        Self {
            weights,
            num_samples,
            replacement,
            seed,
        }
    }

    /// Advance the xorshift64 state and return the next u64.
    #[inline]
    fn xorshift64(state: &mut u64) -> u64 {
        let mut s = *state;
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *state = s;
        s
    }

    /// Draw one index using the inverse-CDF method on the cumulative weights.
    fn weighted_sample(cumulative: &[f64], total: f64, rng_state: &mut u64) -> usize {
        let u = Self::xorshift64(rng_state);
        // Map to (0, 1].
        let r = ((u >> 11) as f64) / ((1u64 << 53) as f64) * total;
        // Binary search: first index where cumulative[i] > r.
        cumulative
            .partition_point(|&c| c <= r)
            .min(cumulative.len() - 1)
    }
}

impl Sampler for WeightedRandomSampler {
    fn indices(&self, epoch: usize) -> Vec<usize> {
        let mut rng_state = self.seed ^ (epoch as u64).wrapping_mul(0x9e3779b97f4a7c15);
        if rng_state == 0 {
            rng_state = 0xdeadbeefcafe;
        }

        // Build cumulative weight array.
        let mut cumulative = Vec::with_capacity(self.weights.len());
        let mut acc = 0.0f64;
        for &w in &self.weights {
            acc += w;
            cumulative.push(acc);
        }
        let total = acc;

        if self.replacement {
            (0..self.num_samples)
                .map(|_| Self::weighted_sample(&cumulative, total, &mut rng_state))
                .collect()
        } else {
            // Without replacement: draw from shrinking pool.
            let mut live_weights: Vec<f64> = self.weights.clone();
            let mut result = Vec::with_capacity(self.num_samples);

            for _ in 0..self.num_samples {
                // Rebuild cumulative for live weights.
                let mut cum = Vec::with_capacity(live_weights.len());
                let mut a = 0.0f64;
                for &w in &live_weights {
                    a += w;
                    cum.push(a);
                }
                let t = a;

                let idx = Self::weighted_sample(&cum, t, &mut rng_state);
                result.push(idx);
                live_weights[idx] = 0.0; // Remove from pool.
            }
            result
        }
    }

    fn len(&self) -> usize {
        self.num_samples
    }
}

// ---------------------------------------------------------------------------
// BatchSampler
// ---------------------------------------------------------------------------

/// Wraps another sampler and yields batches of indices.
///
/// Matches PyTorch's `torch.utils.data.BatchSampler`.
#[derive(Debug, Clone)]
pub struct BatchSampler<S: Sampler> {
    sampler: S,
    batch_size: usize,
    drop_last: bool,
}

impl<S: Sampler> BatchSampler<S> {
    /// Create a new `BatchSampler`.
    ///
    /// # Panics
    ///
    /// Panics if `batch_size` is 0.
    pub fn new(sampler: S, batch_size: usize, drop_last: bool) -> Self {
        assert!(batch_size > 0, "BatchSampler: batch_size must be > 0");
        Self {
            sampler,
            batch_size,
            drop_last,
        }
    }

    /// Return batches of indices for the given epoch.
    pub fn batches(&self, epoch: usize) -> Vec<Vec<usize>> {
        let all = self.sampler.indices(epoch);
        let mut result = Vec::with_capacity(all.len().div_ceil(self.batch_size));

        for chunk in all.chunks(self.batch_size) {
            if chunk.len() < self.batch_size && self.drop_last {
                break;
            }
            result.push(chunk.to_vec());
        }
        result
    }

    /// Number of batches.
    pub fn num_batches(&self) -> usize {
        let n = self.sampler.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_sampler() {
        let s = SequentialSampler::new(5);
        assert_eq!(s.indices(0), vec![0, 1, 2, 3, 4]);
        assert_eq!(s.indices(1), vec![0, 1, 2, 3, 4]); // Same every epoch.
        assert_eq!(s.len(), 5);
    }

    #[test]
    fn test_random_sampler_permutation() {
        let s = RandomSampler::new(10, 42);
        let idx = s.indices(0);
        assert_eq!(idx.len(), 10);
        // Contains all indices.
        let mut sorted = idx.clone();
        sorted.sort();
        assert_eq!(sorted, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_random_sampler_reproducible() {
        let s = RandomSampler::new(100, 42);
        let a = s.indices(0);
        let b = s.indices(0);
        assert_eq!(a, b); // Same seed+epoch = same order.
    }

    #[test]
    fn test_random_sampler_different_epochs() {
        let s = RandomSampler::new(20, 42);
        let a = s.indices(0);
        let b = s.indices(1);
        assert_ne!(a, b); // Different epochs = different order.
    }

    #[test]
    fn test_random_sampler_shuffled() {
        let s = RandomSampler::new(100, 42);
        let idx = s.indices(0);
        let sequential: Vec<usize> = (0..100).collect();
        assert_ne!(idx, sequential); // Should be shuffled.
    }

    // ── shuffle_with_seed ──────────────────────────────────────────

    #[test]
    fn test_shuffle_with_seed_deterministic() {
        let mut a: Vec<usize> = (0..50).collect();
        let mut b: Vec<usize> = (0..50).collect();
        shuffle_with_seed(&mut a, 123);
        shuffle_with_seed(&mut b, 123);
        assert_eq!(a, b);
    }

    #[test]
    fn test_shuffle_with_seed_different_seeds() {
        let mut a: Vec<usize> = (0..100).collect();
        let mut b: Vec<usize> = (0..100).collect();
        shuffle_with_seed(&mut a, 1);
        shuffle_with_seed(&mut b, 2);
        assert_ne!(a, b);
    }

    // ── DistributedSampler ─────────────────────────────────────────

    #[test]
    fn test_distributed_sampler_len() {
        // 10 samples, 3 replicas => ceil(10/3) = 4 per rank
        let s = DistributedSampler::new(10, 3, 0);
        assert_eq!(s.len(), 4);

        // Exact division: 12 samples, 4 replicas => 3 per rank
        let s = DistributedSampler::new(12, 4, 0);
        assert_eq!(s.len(), 3);
    }

    #[test]
    fn test_distributed_sampler_no_overlap() {
        let n = 10;
        let world_size = 3;
        let mut all_indices = Vec::new();
        for rank in 0..world_size {
            let s = DistributedSampler::new(n, world_size, rank).shuffle(false);
            let indices = s.indices(0);
            assert_eq!(indices.len(), s.len());
            all_indices.extend(indices);
        }
        // Total indices = per_rank * world_size = 4 * 3 = 12 (padded from 10)
        assert_eq!(all_indices.len(), 12);

        // All original indices should be covered.
        for i in 0..n {
            assert!(
                all_indices.contains(&i),
                "index {i} missing from distributed partitions"
            );
        }
    }

    #[test]
    fn test_distributed_sampler_sequential_partitioning() {
        // With shuffle=false, indices are interleaved: rank0 gets 0,3,6,9;
        // rank1 gets 1,4,7,padded; rank2 gets 2,5,8,padded.
        let s0 = DistributedSampler::new(10, 3, 0).shuffle(false);
        let s1 = DistributedSampler::new(10, 3, 1).shuffle(false);
        let s2 = DistributedSampler::new(10, 3, 2).shuffle(false);

        let i0 = s0.indices(0);
        let i1 = s1.indices(0);
        let i2 = s2.indices(0);

        assert_eq!(i0, vec![0, 3, 6, 9]);
        assert_eq!(i1, vec![1, 4, 7, 0]); // padded: index wraps to 0
        assert_eq!(i2, vec![2, 5, 8, 1]); // padded: index wraps to 1
    }

    #[test]
    fn test_distributed_sampler_shuffle_reproducible() {
        let s = DistributedSampler::new(100, 4, 1).seed(42);
        let a = s.indices(0);
        let b = s.indices(0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_distributed_sampler_shuffle_varies_by_epoch() {
        let s = DistributedSampler::new(100, 4, 0).seed(42);
        let a = s.indices(0);
        let b = s.indices(1);
        assert_ne!(a, b);
    }

    #[test]
    fn test_distributed_sampler_different_ranks_differ() {
        let s0 = DistributedSampler::new(100, 4, 0).seed(42);
        let s1 = DistributedSampler::new(100, 4, 1).seed(42);
        let a = s0.indices(0);
        let b = s1.indices(0);
        assert_ne!(a, b);
    }

    #[test]
    fn test_distributed_sampler_exact_division() {
        let s = DistributedSampler::new(12, 4, 2).shuffle(false);
        let indices = s.indices(0);
        assert_eq!(indices.len(), 3);
        assert_eq!(indices, vec![2, 6, 10]);
    }

    #[test]
    #[should_panic(expected = "rank must be < num_replicas")]
    fn test_distributed_sampler_invalid_rank() {
        let _ = DistributedSampler::new(10, 3, 3);
    }

    #[test]
    fn test_distributed_sampler_single_replica() {
        // With 1 replica, should return all indices.
        let s = DistributedSampler::new(5, 1, 0).shuffle(false);
        assert_eq!(s.indices(0), vec![0, 1, 2, 3, 4]);
        assert_eq!(s.len(), 5);
    }

    // ── WeightedRandomSampler ─────────────────────────────────────

    #[test]
    fn test_weighted_sampler_len() {
        let s = WeightedRandomSampler::new(vec![1.0, 2.0, 3.0], 10, true, 42);
        assert_eq!(s.len(), 10);
    }

    #[test]
    fn test_weighted_sampler_replacement_count() {
        let s = WeightedRandomSampler::new(vec![1.0, 1.0, 1.0, 1.0], 20, true, 99);
        let idx = s.indices(0);
        assert_eq!(idx.len(), 20);
        // All indices should be in range [0, 4).
        assert!(idx.iter().all(|&i| i < 4));
    }

    #[test]
    fn test_weighted_sampler_reproducible() {
        let s = WeightedRandomSampler::new(vec![1.0, 2.0, 3.0], 50, true, 42);
        let a = s.indices(0);
        let b = s.indices(0);
        assert_eq!(a, b);
    }

    #[test]
    fn test_weighted_sampler_different_epochs() {
        let s = WeightedRandomSampler::new(vec![1.0, 1.0, 1.0, 1.0], 20, true, 42);
        let a = s.indices(0);
        let b = s.indices(1);
        assert_ne!(a, b);
    }

    #[test]
    fn test_weighted_sampler_zero_weight_excluded() {
        // Weight 0 on index 1 means it should never be sampled.
        let s = WeightedRandomSampler::new(vec![1.0, 0.0, 1.0], 100, true, 42);
        let idx = s.indices(0);
        assert!(
            idx.iter().all(|&i| i != 1),
            "index with zero weight was sampled"
        );
    }

    #[test]
    fn test_weighted_sampler_heavy_bias() {
        // Weight 1000 on index 2, weight 1 on others.
        let s = WeightedRandomSampler::new(vec![1.0, 1.0, 1000.0], 200, true, 7);
        let idx = s.indices(0);
        let count_2 = idx.iter().filter(|&&i| i == 2).count();
        // With such a heavy bias, index 2 should dominate.
        assert!(
            count_2 > 150,
            "expected index 2 to be sampled >150 times but got {count_2}"
        );
    }

    #[test]
    fn test_weighted_sampler_no_replacement() {
        let s = WeightedRandomSampler::new(vec![1.0, 2.0, 3.0, 4.0], 4, false, 42);
        let idx = s.indices(0);
        assert_eq!(idx.len(), 4);
        // All unique.
        let mut sorted = idx.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 4);
    }

    #[test]
    fn test_weighted_sampler_no_replacement_subset() {
        let s = WeightedRandomSampler::new(vec![1.0, 1.0, 1.0, 1.0, 1.0], 3, false, 42);
        let idx = s.indices(0);
        assert_eq!(idx.len(), 3);
        // All unique.
        let mut sorted = idx.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3);
        assert!(idx.iter().all(|&i| i < 5));
    }

    #[test]
    #[should_panic(expected = "weights must not be empty")]
    fn test_weighted_sampler_empty_weights() {
        WeightedRandomSampler::new(vec![], 10, true, 0);
    }

    #[test]
    #[should_panic(expected = "weights must be non-negative")]
    fn test_weighted_sampler_negative_weight() {
        WeightedRandomSampler::new(vec![1.0, -1.0], 5, true, 0);
    }

    #[test]
    #[should_panic(expected = "at least one weight must be positive")]
    fn test_weighted_sampler_all_zero_weights() {
        WeightedRandomSampler::new(vec![0.0, 0.0], 5, true, 0);
    }

    #[test]
    #[should_panic(expected = "num_samples")]
    fn test_weighted_sampler_no_replacement_too_many() {
        WeightedRandomSampler::new(vec![1.0, 1.0], 5, false, 0);
    }

    // ── BatchSampler ──────────────────────────────────────────────

    #[test]
    fn test_batch_sampler_basic() {
        let inner = SequentialSampler::new(10);
        let bs = BatchSampler::new(inner, 3, false);

        let batches = bs.batches(0);
        assert_eq!(batches.len(), 4); // 3 + 3 + 3 + 1
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
        assert_eq!(batches[2], vec![6, 7, 8]);
        assert_eq!(batches[3], vec![9]);
    }

    #[test]
    fn test_batch_sampler_drop_last() {
        let inner = SequentialSampler::new(10);
        let bs = BatchSampler::new(inner, 3, true);

        let batches = bs.batches(0);
        assert_eq!(batches.len(), 3); // last incomplete batch dropped
        assert_eq!(batches[0], vec![0, 1, 2]);
        assert_eq!(batches[1], vec![3, 4, 5]);
        assert_eq!(batches[2], vec![6, 7, 8]);
    }

    #[test]
    fn test_batch_sampler_exact() {
        let inner = SequentialSampler::new(9);
        let bs = BatchSampler::new(inner, 3, false);
        let batches = bs.batches(0);
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_batch_sampler_exact_drop_last() {
        let inner = SequentialSampler::new(9);
        let bs = BatchSampler::new(inner, 3, true);
        let batches = bs.batches(0);
        assert_eq!(batches.len(), 3); // No incomplete batch to drop.
    }

    #[test]
    fn test_batch_sampler_num_batches() {
        let inner = SequentialSampler::new(10);
        assert_eq!(BatchSampler::new(inner.clone(), 3, false).num_batches(), 4);
        assert_eq!(BatchSampler::new(inner.clone(), 3, true).num_batches(), 3);
        assert_eq!(BatchSampler::new(inner.clone(), 10, false).num_batches(), 1);
        assert_eq!(BatchSampler::new(inner, 10, true).num_batches(), 1);
    }

    #[test]
    fn test_batch_sampler_with_random() {
        let inner = RandomSampler::new(7, 42);
        let bs = BatchSampler::new(inner, 3, false);
        let batches = bs.batches(0);
        assert_eq!(batches.len(), 3); // 3 + 3 + 1
        assert_eq!(batches[0].len(), 3);
        assert_eq!(batches[1].len(), 3);
        assert_eq!(batches[2].len(), 1);
        // All 7 indices present.
        let flat: Vec<usize> = batches.into_iter().flatten().collect();
        let mut sorted = flat.clone();
        sorted.sort();
        assert_eq!(sorted, (0..7).collect::<Vec<_>>());
    }

    #[test]
    fn test_batch_sampler_single_element_batches() {
        let inner = SequentialSampler::new(3);
        let bs = BatchSampler::new(inner, 1, false);
        let batches = bs.batches(0);
        assert_eq!(batches, vec![vec![0], vec![1], vec![2]]);
    }

    #[test]
    #[should_panic(expected = "batch_size must be > 0")]
    fn test_batch_sampler_zero_batch_size() {
        let inner = SequentialSampler::new(5);
        BatchSampler::new(inner, 0, false);
    }
}
