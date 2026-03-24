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
}
