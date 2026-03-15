use std::sync::Arc;

use ferrotorch_core::FerrotorchResult;

use crate::dataset::Dataset;
use crate::sampler::{RandomSampler, Sampler, SequentialSampler};

/// A data loader that yields batches of samples from a [`Dataset`].
///
/// Mirrors the core API of PyTorch's `DataLoader`, but returns
/// `Vec<D::Sample>` batches so the caller can collate as needed.
///
/// # Examples
///
/// ```ignore
/// let ds = VecDataset::new(vec![1, 2, 3, 4, 5]);
/// let loader = DataLoader::new(Arc::new(ds), 2);
///
/// for batch in loader.iter(0) {
///     let batch = batch.unwrap();
///     println!("{batch:?}");
/// }
/// // [1, 2], [3, 4], [5]
/// ```
pub struct DataLoader<D: Dataset> {
    dataset: Arc<D>,
    batch_size: usize,
    shuffle: bool,
    drop_last: bool,
    seed: u64,
    collate_fn: Option<Arc<dyn Fn(Vec<D::Sample>) -> FerrotorchResult<D::Sample> + Send + Sync>>,
}

impl<D: Dataset> DataLoader<D> {
    /// Create a new `DataLoader` for the given dataset and batch size.
    ///
    /// Defaults: sequential order, keep the final partial batch, seed 0.
    ///
    /// # Panics
    ///
    /// Panics if `batch_size` is 0.
    pub fn new(dataset: Arc<D>, batch_size: usize) -> Self {
        assert!(batch_size > 0, "batch_size must be > 0");
        Self {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            seed: 0,
            collate_fn: None,
        }
    }

    /// Enable or disable shuffling.
    ///
    /// When enabled, a [`RandomSampler`] is used to permute indices each
    /// epoch. Otherwise a [`SequentialSampler`] yields 0..n in order.
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// If `true`, drop the last batch when it is smaller than `batch_size`.
    pub fn drop_last(mut self, drop_last: bool) -> Self {
        self.drop_last = drop_last;
        self
    }

    /// Set the base seed used by the random sampler.
    ///
    /// The effective seed for each epoch is derived from this value and the
    /// epoch number, so different epochs produce different orderings while
    /// remaining deterministic.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set a collation function that merges a `Vec<Sample>` into a single
    /// `Sample`.
    ///
    /// When set, each batch produced by the iterator will be passed through
    /// the collation function. The iterator's `Item` type changes from
    /// `FerrotorchResult<Vec<D::Sample>>` to `FerrotorchResult<D::Sample>`
    /// when consumed via [`DataLoader::iter_collated`].
    ///
    /// The collate function is also available through the standard
    /// [`DataLoader::iter`] path: if a collate function is **not** set,
    /// the iterator returns raw `Vec<D::Sample>` batches as before.
    pub fn with_collate<F>(mut self, f: F) -> Self
    where
        F: Fn(Vec<D::Sample>) -> FerrotorchResult<D::Sample> + Send + Sync + 'static,
    {
        self.collate_fn = Some(Arc::new(f));
        self
    }

    /// Return a reference to the collation function, if one has been set.
    pub fn collate_fn(
        &self,
    ) -> Option<&(dyn Fn(Vec<D::Sample>) -> FerrotorchResult<D::Sample> + Send + Sync)> {
        self.collate_fn.as_deref()
    }

    /// Produce a collated batch iterator for the given epoch.
    ///
    /// Each call to `next()` returns a single collated `D::Sample` produced
    /// by the collation function set via [`with_collate`](DataLoader::with_collate).
    ///
    /// # Panics
    ///
    /// Panics if no collation function has been set. Use [`iter`](DataLoader::iter)
    /// for the uncollated path.
    pub fn iter_collated(&self, epoch: usize) -> CollatedIter<'_, D> {
        let collate_fn = self
            .collate_fn
            .as_ref()
            .expect("iter_collated called without a collate_fn — use with_collate() first");

        CollatedIter {
            inner: self.iter(epoch),
            collate_fn: collate_fn.as_ref(),
        }
    }

    /// Return the number of batches that will be produced for one epoch.
    pub fn len(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            (n + self.batch_size - 1) / self.batch_size
        }
    }

    /// Whether the loader produces zero batches.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Produce a batch iterator for the given epoch.
    ///
    /// The `epoch` parameter is passed to the sampler so that shuffled
    /// orderings vary across epochs yet remain reproducible.
    pub fn iter(&self, epoch: usize) -> DataLoaderIter<'_, D> {
        let sampler: Box<dyn Sampler> = if self.shuffle {
            Box::new(RandomSampler::new(self.dataset.len(), self.seed))
        } else {
            Box::new(SequentialSampler::new(self.dataset.len()))
        };

        let indices = sampler.indices(epoch);

        DataLoaderIter {
            dataset: &self.dataset,
            indices,
            batch_size: self.batch_size,
            drop_last: self.drop_last,
            pos: 0,
        }
    }
}

/// Iterator over batches produced by a [`DataLoader`].
///
/// Each call to `next()` returns `Some(FerrotorchResult<Vec<D::Sample>>)`.
/// The result is `Err` if any individual `Dataset::get` fails.
pub struct DataLoaderIter<'a, D: Dataset> {
    dataset: &'a D,
    indices: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    pos: usize,
}

impl<D: Dataset> Iterator for DataLoaderIter<'_, D> {
    type Item = FerrotorchResult<Vec<D::Sample>>;

    fn next(&mut self) -> Option<Self::Item> {
        let remaining = self.indices.len().saturating_sub(self.pos);
        if remaining == 0 {
            return None;
        }
        if self.drop_last && remaining < self.batch_size {
            return None;
        }

        let end = (self.pos + self.batch_size).min(self.indices.len());
        let batch_indices = &self.indices[self.pos..end];
        self.pos = end;

        let mut batch = Vec::with_capacity(batch_indices.len());
        for &idx in batch_indices {
            match self.dataset.get(idx) {
                Ok(sample) => batch.push(sample),
                Err(e) => return Some(Err(e)),
            }
        }
        Some(Ok(batch))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len().saturating_sub(self.pos);
        let batches = if self.drop_last {
            remaining / self.batch_size
        } else if remaining == 0 {
            0
        } else {
            (remaining + self.batch_size - 1) / self.batch_size
        };
        (batches, Some(batches))
    }
}

impl<D: Dataset> ExactSizeIterator for DataLoaderIter<'_, D> {}

/// Iterator over collated batches produced by a [`DataLoader`].
///
/// Each call to `next()` returns `Some(FerrotorchResult<D::Sample>)` — a
/// single collated sample produced by the user-supplied collation function.
pub struct CollatedIter<'a, D: Dataset> {
    inner: DataLoaderIter<'a, D>,
    collate_fn: &'a (dyn Fn(Vec<D::Sample>) -> FerrotorchResult<D::Sample> + Send + Sync),
}

impl<D: Dataset> Iterator for CollatedIter<'_, D> {
    type Item = FerrotorchResult<D::Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        Some(batch.and_then(|samples| (self.collate_fn)(samples)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<D: Dataset> ExactSizeIterator for CollatedIter<'_, D> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::VecDataset;

    fn make_dataset(n: usize) -> Arc<VecDataset<i32>> {
        Arc::new(VecDataset::new((0..n as i32).collect()))
    }

    // ── batch count ─────────────────────────────────────────────────

    #[test]
    fn test_batch_count_exact_division() {
        let loader = DataLoader::new(make_dataset(10), 5);
        assert_eq!(loader.len(), 2);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_batch_count_with_remainder() {
        let loader = DataLoader::new(make_dataset(10), 3);
        assert_eq!(loader.len(), 4); // ceil(10/3) = 4
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_batch_count_single_element() {
        let loader = DataLoader::new(make_dataset(1), 5);
        assert_eq!(loader.len(), 1);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn test_empty_dataset() {
        let loader = DataLoader::new(make_dataset(0), 4);
        assert!(loader.is_empty());
        assert_eq!(loader.len(), 0);
        let batches: Vec<_> = loader.iter(0).collect();
        assert!(batches.is_empty());
    }

    // ── batch sizes ─────────────────────────────────────────────────

    #[test]
    fn test_batch_sizes_exact() {
        let loader = DataLoader::new(make_dataset(6), 3);
        let sizes: Vec<usize> = loader
            .iter(0)
            .map(|b| b.unwrap().len())
            .collect();
        assert_eq!(sizes, vec![3, 3]);
    }

    #[test]
    fn test_batch_sizes_with_partial_last() {
        let loader = DataLoader::new(make_dataset(7), 3);
        let sizes: Vec<usize> = loader
            .iter(0)
            .map(|b| b.unwrap().len())
            .collect();
        assert_eq!(sizes, vec![3, 3, 1]);
    }

    #[test]
    fn test_all_samples_present_sequential() {
        let loader = DataLoader::new(make_dataset(10), 3);
        let mut all: Vec<i32> = loader
            .iter(0)
            .flat_map(|b| b.unwrap())
            .collect();
        all.sort();
        assert_eq!(all, (0..10).collect::<Vec<i32>>());
    }

    // ── drop_last ───────────────────────────────────────────────────

    #[test]
    fn test_drop_last_removes_partial_batch() {
        let loader = DataLoader::new(make_dataset(10), 3).drop_last(true);
        assert_eq!(loader.len(), 3); // 10/3 = 3 full batches
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.as_ref().unwrap().len(), 3);
        }
    }

    #[test]
    fn test_drop_last_exact_keeps_all() {
        let loader = DataLoader::new(make_dataset(9), 3).drop_last(true);
        assert_eq!(loader.len(), 3);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_drop_last_smaller_than_batch() {
        let loader = DataLoader::new(make_dataset(2), 5).drop_last(true);
        assert!(loader.is_empty());
        let batches: Vec<_> = loader.iter(0).collect();
        assert!(batches.is_empty());
    }

    // ── shuffle ─────────────────────────────────────────────────────

    #[test]
    fn test_shuffle_produces_different_order() {
        let loader = DataLoader::new(make_dataset(100), 100)
            .shuffle(true)
            .seed(42);
        let batch = loader.iter(0).next().unwrap().unwrap();
        let sequential: Vec<i32> = (0..100).collect();
        assert_ne!(batch, sequential, "shuffled batch should differ from sequential");
    }

    #[test]
    fn test_shuffle_contains_all_elements() {
        let loader = DataLoader::new(make_dataset(20), 5)
            .shuffle(true)
            .seed(7);
        let mut all: Vec<i32> = loader
            .iter(0)
            .flat_map(|b| b.unwrap())
            .collect();
        all.sort();
        assert_eq!(all, (0..20).collect::<Vec<i32>>());
    }

    #[test]
    fn test_shuffle_different_epochs() {
        let loader = DataLoader::new(make_dataset(50), 50)
            .shuffle(true)
            .seed(99);
        let epoch0 = loader.iter(0).next().unwrap().unwrap();
        let epoch1 = loader.iter(1).next().unwrap().unwrap();
        assert_ne!(epoch0, epoch1, "different epochs should produce different order");
    }

    // ── reproducibility ─────────────────────────────────────────────

    #[test]
    fn test_reproducible_with_same_seed_and_epoch() {
        let loader = DataLoader::new(make_dataset(30), 10)
            .shuffle(true)
            .seed(42);
        let run1: Vec<Vec<i32>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();
        let run2: Vec<Vec<i32>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(run1, run2);
    }

    #[test]
    fn test_different_seeds_differ() {
        let ds = make_dataset(100);
        let loader_a = DataLoader::new(Arc::clone(&ds), 100)
            .shuffle(true)
            .seed(1);
        let loader_b = DataLoader::new(ds, 100)
            .shuffle(true)
            .seed(2);
        let a = loader_a.iter(0).next().unwrap().unwrap();
        let b = loader_b.iter(0).next().unwrap().unwrap();
        assert_ne!(a, b);
    }

    // ── size_hint / ExactSizeIterator ───────────────────────────────

    #[test]
    fn test_size_hint_accurate() {
        let loader = DataLoader::new(make_dataset(11), 3);
        let mut it = loader.iter(0);
        assert_eq!(it.len(), 4);
        it.next();
        assert_eq!(it.len(), 3);
        it.next();
        assert_eq!(it.len(), 2);
        it.next();
        assert_eq!(it.len(), 1);
        it.next();
        assert_eq!(it.len(), 0);
        assert!(it.next().is_none());
    }

    #[test]
    fn test_size_hint_drop_last() {
        let loader = DataLoader::new(make_dataset(11), 3).drop_last(true);
        let it = loader.iter(0);
        assert_eq!(it.len(), 3);
    }

    // ── builder ergonomics ──────────────────────────────────────────

    #[test]
    #[should_panic(expected = "batch_size must be > 0")]
    fn test_zero_batch_size_panics() {
        let _ = DataLoader::new(make_dataset(5), 0);
    }

    #[test]
    fn test_builder_chaining() {
        let loader = DataLoader::new(make_dataset(10), 2)
            .shuffle(true)
            .drop_last(true)
            .seed(123);
        assert!(loader.shuffle);
        assert!(loader.drop_last);
        assert_eq!(loader.seed, 123);
    }

    // ── collate_fn ────────────────────────────────────────────────────

    #[test]
    fn test_with_collate_sum() {
        let loader = DataLoader::new(make_dataset(6), 3)
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let collated: Vec<i32> = loader
            .iter_collated(0)
            .map(|r| r.unwrap())
            .collect();
        // Sequential: [0,1,2] -> 3, [3,4,5] -> 12
        assert_eq!(collated, vec![3, 12]);
    }

    #[test]
    fn test_collate_with_remainder() {
        let loader = DataLoader::new(make_dataset(5), 3)
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let collated: Vec<i32> = loader
            .iter_collated(0)
            .map(|r| r.unwrap())
            .collect();
        // [0,1,2] -> 3, [3,4] -> 7
        assert_eq!(collated, vec![3, 7]);
    }

    #[test]
    fn test_collate_with_drop_last() {
        let loader = DataLoader::new(make_dataset(5), 3)
            .drop_last(true)
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let collated: Vec<i32> = loader
            .iter_collated(0)
            .map(|r| r.unwrap())
            .collect();
        // Only [0,1,2] -> 3 (partial batch [3,4] dropped)
        assert_eq!(collated, vec![3]);
    }

    #[test]
    fn test_collate_fn_accessor() {
        let loader = DataLoader::new(make_dataset(5), 3);
        assert!(loader.collate_fn().is_none());

        let loader = loader.with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));
        assert!(loader.collate_fn().is_some());
    }

    #[test]
    fn test_collated_iter_size_hint() {
        let loader = DataLoader::new(make_dataset(10), 3)
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let it = loader.iter_collated(0);
        assert_eq!(it.len(), 4); // ceil(10/3) = 4
    }

    #[test]
    fn test_collate_error_propagation() {
        let loader = DataLoader::new(make_dataset(4), 2)
            .with_collate(|_batch| {
                Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                    message: "test error".into(),
                })
            });

        let results: Vec<_> = loader.iter_collated(0).collect();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_err());
        assert!(results[1].is_err());
    }

    #[test]
    #[should_panic(expected = "iter_collated called without a collate_fn")]
    fn test_collated_iter_panics_without_collate_fn() {
        let loader = DataLoader::new(make_dataset(5), 3);
        let _ = loader.iter_collated(0);
    }

    #[test]
    fn test_uncollated_iter_unaffected_by_collate() {
        // The original iter() path still returns Vec<Sample> even when
        // a collate_fn is set.
        let loader = DataLoader::new(make_dataset(4), 2)
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let batches: Vec<Vec<i32>> = loader
            .iter(0)
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(batches, vec![vec![0, 1], vec![2, 3]]);
    }
}
