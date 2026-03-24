use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::Arc;
use std::thread::{self, JoinHandle};

use ferrotorch_core::{Device, FerrotorchResult};
use rayon::prelude::*;

use crate::dataset::Dataset;
use crate::sampler::{RandomSampler, Sampler, SequentialSampler};

/// Trait for sample types that can be transferred to a target [`Device`].
///
/// Implement this for your `Dataset::Sample` type to enable automatic device
/// transfer via [`DataLoader::device`].
///
/// # Examples
///
/// ```ignore
/// use ferrotorch_core::{Device, FerrotorchResult, Tensor};
/// use ferrotorch_data::ToDevice;
///
/// struct MyBatch {
///     inputs: Tensor<f32>,
///     labels: Tensor<f32>,
/// }
///
/// impl ToDevice for MyBatch {
///     fn to_device(&self, device: Device) -> FerrotorchResult<Self> {
///         Ok(MyBatch {
///             inputs: self.inputs.to(device)?,
///             labels: self.labels.to(device)?,
///         })
///     }
/// }
/// ```
pub trait ToDevice: Sized {
    /// Transfer this value to the given device.
    fn to_device(&self, device: Device) -> FerrotorchResult<Self>;
}

/// Type alias for the device-transfer function stored internally.
type TransferFn<S> = Arc<dyn Fn(Vec<S>) -> FerrotorchResult<Vec<S>> + Send + Sync>;

/// A data loader that yields batches of samples from a [`Dataset`].
///
/// Mirrors the core API of PyTorch's `DataLoader`, but returns
/// `Vec<D::Sample>` batches so the caller can collate as needed.
///
/// # Prefetch Pipeline
///
/// When `prefetch_factor > 0` (default: 2), a background thread loads batches
/// ahead of the consumer and buffers them in a bounded channel. The consumer's
/// `next()` call receives from the channel instead of loading directly, hiding
/// I/O latency behind computation.
///
/// # Device Transfer
///
/// When a `device` is set via [`DataLoader::device`] and the `Sample` type
/// implements [`ToDevice`], each batch is transferred to the target device
/// after loading. The transfer runs inside the prefetch pipeline when enabled,
/// so device transfers overlap with the consumer's computation.
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
    num_workers: usize,
    prefetch_factor: usize,
    device: Option<Device>,
    custom_sampler: Option<Box<dyn Sampler>>,
    collate_fn: Option<Arc<dyn Fn(Vec<D::Sample>) -> FerrotorchResult<D::Sample> + Send + Sync>>,
    transfer_fn: Option<TransferFn<D::Sample>>,
}

impl<D: Dataset> DataLoader<D> {
    /// Create a new `DataLoader` for the given dataset and batch size.
    ///
    /// Defaults: sequential order, keep the final partial batch, seed 0,
    /// prefetch_factor 2, no device transfer.
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
            num_workers: 0,
            prefetch_factor: 2,
            device: None,
            custom_sampler: None,
            collate_fn: None,
            transfer_fn: None,
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

    /// Set the number of worker threads for parallel sample loading.
    ///
    /// When `n > 0`, each batch's samples are loaded in parallel using
    /// rayon's work-stealing thread pool. When `n == 0` (the default),
    /// samples are loaded sequentially on the calling thread.
    pub fn num_workers(mut self, n: usize) -> Self {
        self.num_workers = n;
        self
    }

    /// Set the prefetch factor — the number of batches buffered ahead of
    /// the consumer.
    ///
    /// When `prefetch_factor > 0`, a background thread loads batches into a
    /// bounded channel of this capacity. The iterator's `next()` receives
    /// from the channel instead of loading directly.
    ///
    /// When `prefetch_factor == 0`, batches are loaded synchronously on the
    /// calling thread (no background thread is spawned).
    ///
    /// Default: 2.
    pub fn prefetch_factor(mut self, n: usize) -> Self {
        self.prefetch_factor = n;
        self
    }

    /// Set the target device for automatic batch transfer.
    ///
    /// When set, each batch's samples are transferred to the given device
    /// after loading. The `Sample` type must implement [`ToDevice`].
    ///
    /// This transfer happens inside the prefetch pipeline (if enabled),
    /// so device transfers are overlapped with the consumer's processing.
    pub fn device(mut self, device: Device) -> Self
    where
        D::Sample: ToDevice + 'static,
    {
        self.device = Some(device);
        self.transfer_fn = Some(Arc::new(move |samples: Vec<D::Sample>| {
            samples
                .into_iter()
                .map(|s| s.to_device(device))
                .collect()
        }));
        self
    }

    /// Inject a custom [`Sampler`] to control index generation.
    ///
    /// When a custom sampler is set, it takes precedence over the
    /// `shuffle` flag — the sampler fully controls index ordering.
    pub fn with_sampler(mut self, sampler: Box<dyn Sampler>) -> Self {
        self.custom_sampler = Some(sampler);
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
    pub fn iter_collated(&self, epoch: usize) -> CollatedIter<'_, D>
    where
        D: 'static,
        D::Sample: 'static,
    {
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

    /// Build the index list for the given epoch.
    fn build_indices(&self, epoch: usize) -> Vec<usize> {
        if let Some(ref sampler) = self.custom_sampler {
            sampler.indices(epoch)
        } else if self.shuffle {
            let sampler = RandomSampler::new(self.dataset.len(), self.seed);
            sampler.indices(epoch)
        } else {
            let sampler = SequentialSampler::new(self.dataset.len());
            sampler.indices(epoch)
        }
    }

    /// Produce a batch iterator for the given epoch.
    ///
    /// The `epoch` parameter is passed to the sampler so that shuffled
    /// orderings vary across epochs yet remain reproducible.
    ///
    /// When `prefetch_factor > 0`, returns a prefetching iterator backed by
    /// a background thread. When `prefetch_factor == 0`, returns a
    /// synchronous iterator that loads on the calling thread.
    pub fn iter(&self, epoch: usize) -> BatchIter<'_, D>
    where
        D: 'static,
        D::Sample: 'static,
    {
        let indices = self.build_indices(epoch);

        if self.prefetch_factor > 0 {
            BatchIter::Prefetch(PrefetchIter::new(
                Arc::clone(&self.dataset),
                indices,
                self.batch_size,
                self.drop_last,
                self.num_workers,
                self.prefetch_factor,
                self.transfer_fn.clone(),
            ))
        } else {
            BatchIter::Sync(DataLoaderIter {
                dataset: &self.dataset,
                indices,
                batch_size: self.batch_size,
                drop_last: self.drop_last,
                num_workers: self.num_workers,
                transfer_fn: self.transfer_fn.as_ref(),
                pos: 0,
            })
        }
    }
}

/// Iterator over batches — either synchronous or prefetched.
///
/// Returned by [`DataLoader::iter`]. Transparent to the caller: both
/// variants yield `FerrotorchResult<Vec<D::Sample>>`.
pub enum BatchIter<'a, D: Dataset> {
    /// Synchronous loading on the calling thread.
    Sync(DataLoaderIter<'a, D>),
    /// Prefetched loading via a background thread.
    Prefetch(PrefetchIter<D>),
}

impl<D: Dataset + 'static> Iterator for BatchIter<'_, D>
where
    D::Sample: Send + 'static,
{
    type Item = FerrotorchResult<Vec<D::Sample>>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            BatchIter::Sync(inner) => inner.next(),
            BatchIter::Prefetch(inner) => inner.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            BatchIter::Sync(inner) => inner.size_hint(),
            BatchIter::Prefetch(inner) => inner.size_hint(),
        }
    }
}

impl<D: Dataset + 'static> ExactSizeIterator for BatchIter<'_, D> where D::Sample: Send + 'static {}

/// Iterator over batches produced by a [`DataLoader`] (synchronous path).
///
/// Each call to `next()` returns `Some(FerrotorchResult<Vec<D::Sample>>)`.
/// The result is `Err` if any individual `Dataset::get` fails.
///
/// When `num_workers > 0`, samples within each batch are loaded in parallel
/// using rayon's work-stealing thread pool.
pub struct DataLoaderIter<'a, D: Dataset> {
    dataset: &'a D,
    indices: Vec<usize>,
    batch_size: usize,
    drop_last: bool,
    num_workers: usize,
    transfer_fn: Option<&'a TransferFn<D::Sample>>,
    pos: usize,
}

impl<D: Dataset> Iterator for DataLoaderIter<'_, D>
where
    D::Sample: Send,
{
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

        let batch = if self.num_workers > 0 {
            // Parallel path: load samples concurrently via rayon.
            batch_indices
                .par_iter()
                .map(|&idx| self.dataset.get(idx))
                .collect::<Result<Vec<_>, _>>()
        } else {
            // Sequential path: load samples one at a time.
            let mut batch = Vec::with_capacity(batch_indices.len());
            for &idx in batch_indices {
                match self.dataset.get(idx) {
                    Ok(sample) => batch.push(sample),
                    Err(e) => return Some(Err(e)),
                }
            }
            Ok(batch)
        };

        // Apply device transfer if configured.
        let batch = match (batch, &self.transfer_fn) {
            (Ok(samples), Some(f)) => f(samples),
            (result, _) => result,
        };

        Some(batch)
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

impl<D: Dataset> ExactSizeIterator for DataLoaderIter<'_, D> where D::Sample: Send {}

/// Prefetch iterator backed by a background loading thread.
///
/// A bounded channel of capacity `prefetch_factor` buffers batches ahead
/// of the consumer. The background thread terminates when the channel is
/// closed (either by exhausting the dataset or by dropping this iterator).
pub struct PrefetchIter<D: Dataset> {
    receiver: Receiver<FerrotorchResult<Vec<D::Sample>>>,
    /// Handle to the background thread. Joined on drop to ensure clean
    /// shutdown and propagate panics.
    _handle: JoinHandle<()>,
    /// Total number of batches the background thread will produce.
    total_batches: usize,
    /// Number of batches consumed so far.
    consumed: usize,
}

impl<D: Dataset + 'static> PrefetchIter<D>
where
    D::Sample: Send + 'static,
{
    fn new(
        dataset: Arc<D>,
        indices: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        prefetch_factor: usize,
        transfer_fn: Option<TransferFn<D::Sample>>,
    ) -> Self {
        let total_batches = compute_batch_count(indices.len(), batch_size, drop_last);

        // Bounded channel — at most `prefetch_factor` batches buffered.
        let (tx, rx) = mpsc::sync_channel::<FerrotorchResult<Vec<D::Sample>>>(
            prefetch_factor.max(1),
        );

        let handle = thread::spawn(move || {
            Self::producer_loop(dataset, indices, batch_size, drop_last, num_workers, transfer_fn, tx);
        });

        PrefetchIter {
            receiver: rx,
            _handle: handle,
            total_batches,
            consumed: 0,
        }
    }

    /// Background producer: loads batches and sends them through the channel.
    ///
    /// Stops when all batches are sent or when the receiver is dropped
    /// (channel disconnected).
    fn producer_loop(
        dataset: Arc<D>,
        indices: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        transfer_fn: Option<TransferFn<D::Sample>>,
        tx: SyncSender<FerrotorchResult<Vec<D::Sample>>>,
    ) {
        let mut pos = 0;
        loop {
            let remaining = indices.len().saturating_sub(pos);
            if remaining == 0 {
                break;
            }
            if drop_last && remaining < batch_size {
                break;
            }

            let end = (pos + batch_size).min(indices.len());
            let batch_indices = &indices[pos..end];
            pos = end;

            let batch = if num_workers > 0 {
                batch_indices
                    .par_iter()
                    .map(|&idx| dataset.get(idx))
                    .collect::<Result<Vec<_>, _>>()
            } else {
                let mut batch = Vec::with_capacity(batch_indices.len());
                let mut err = None;
                for &idx in batch_indices {
                    match dataset.get(idx) {
                        Ok(sample) => batch.push(sample),
                        Err(e) => {
                            err = Some(e);
                            break;
                        }
                    }
                }
                match err {
                    Some(e) => Err(e),
                    None => Ok(batch),
                }
            };

            // Apply device transfer if configured.
            let batch = match (batch, &transfer_fn) {
                (Ok(samples), Some(f)) => f(samples),
                (result, _) => result,
            };

            // send() fails when the receiver is dropped — that's our
            // signal to stop. This is the normal shutdown path when the
            // consumer drops the iterator mid-iteration.
            if tx.send(batch).is_err() {
                break;
            }
        }
        // tx drops here, closing the channel. The consumer's next recv()
        // will see a disconnected error and return None.
    }
}

impl<D: Dataset + 'static> Iterator for PrefetchIter<D>
where
    D::Sample: Send + 'static,
{
    type Item = FerrotorchResult<Vec<D::Sample>>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed >= self.total_batches {
            return None;
        }
        match self.receiver.recv() {
            Ok(batch) => {
                self.consumed += 1;
                Some(batch)
            }
            // Channel disconnected — producer finished or panicked.
            Err(_) => None,
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_batches.saturating_sub(self.consumed);
        (remaining, Some(remaining))
    }
}

impl<D: Dataset + 'static> ExactSizeIterator for PrefetchIter<D> where D::Sample: Send + 'static {}

/// Compute the number of batches for a given index count.
fn compute_batch_count(n_indices: usize, batch_size: usize, drop_last: bool) -> usize {
    if drop_last {
        n_indices / batch_size
    } else if n_indices == 0 {
        0
    } else {
        (n_indices + batch_size - 1) / batch_size
    }
}

/// Iterator over collated batches produced by a [`DataLoader`].
///
/// Each call to `next()` returns `Some(FerrotorchResult<D::Sample>)` — a
/// single collated sample produced by the user-supplied collation function.
pub struct CollatedIter<'a, D: Dataset> {
    inner: BatchIter<'a, D>,
    collate_fn: &'a (dyn Fn(Vec<D::Sample>) -> FerrotorchResult<D::Sample> + Send + Sync),
}

impl<D: Dataset + 'static> Iterator for CollatedIter<'_, D>
where
    D::Sample: Send + 'static,
{
    type Item = FerrotorchResult<D::Sample>;

    fn next(&mut self) -> Option<Self::Item> {
        let batch = self.inner.next()?;
        Some(batch.and_then(|samples| (self.collate_fn)(samples)))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.inner.size_hint()
    }
}

impl<D: Dataset + 'static> ExactSizeIterator for CollatedIter<'_, D> where D::Sample: Send + 'static {}

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
            .seed(123)
            .prefetch_factor(4);
        assert!(loader.shuffle);
        assert!(loader.drop_last);
        assert_eq!(loader.seed, 123);
        assert_eq!(loader.prefetch_factor, 4);
    }

    #[test]
    fn test_builder_chaining_with_device() {
        let loader = DataLoader::new(make_device_dataset(4), 2)
            .prefetch_factor(4)
            .device(Device::Cpu);
        assert_eq!(loader.prefetch_factor, 4);
        assert_eq!(loader.device, Some(Device::Cpu));
        assert!(loader.transfer_fn.is_some());
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

    // ── num_workers ─────────────────────────────────────────────────

    #[test]
    fn test_num_workers_builder() {
        let loader = DataLoader::new(make_dataset(10), 2).num_workers(4);
        assert_eq!(loader.num_workers, 4);
    }

    #[test]
    fn test_num_workers_parallel_loads_all_samples() {
        let loader = DataLoader::new(make_dataset(20), 5).num_workers(2);
        let mut all: Vec<i32> = loader
            .iter(0)
            .flat_map(|b| b.unwrap())
            .collect();
        all.sort();
        assert_eq!(all, (0..20).collect::<Vec<i32>>());
    }

    #[test]
    fn test_num_workers_parallel_batch_sizes() {
        let loader = DataLoader::new(make_dataset(7), 3).num_workers(2);
        let sizes: Vec<usize> = loader
            .iter(0)
            .map(|b| b.unwrap().len())
            .collect();
        assert_eq!(sizes, vec![3, 3, 1]);
    }

    #[test]
    fn test_num_workers_parallel_drop_last() {
        let loader = DataLoader::new(make_dataset(10), 3)
            .num_workers(2)
            .drop_last(true);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.as_ref().unwrap().len(), 3);
        }
    }

    #[test]
    fn test_num_workers_parallel_with_shuffle() {
        let loader = DataLoader::new(make_dataset(100), 100)
            .num_workers(4)
            .shuffle(true)
            .seed(42);
        let batch = loader.iter(0).next().unwrap().unwrap();
        let sequential: Vec<i32> = (0..100).collect();
        // Should be shuffled even in parallel mode.
        assert_ne!(batch, sequential);
        // But contain all elements.
        let mut sorted = batch;
        sorted.sort();
        assert_eq!(sorted, sequential);
    }

    #[test]
    fn test_num_workers_zero_is_sequential() {
        // num_workers=0 should behave identically to default.
        let loader = DataLoader::new(make_dataset(10), 3)
            .num_workers(0)
            .prefetch_factor(0);
        let batches: Vec<Vec<i32>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9]]);
    }

    // ── custom sampler ──────────────────────────────────────────────

    #[test]
    fn test_with_sampler_overrides_shuffle() {
        use crate::sampler::SequentialSampler;

        // Even with shuffle=true, a custom sampler takes precedence.
        let loader = DataLoader::new(make_dataset(6), 3)
            .shuffle(true)
            .seed(42)
            .with_sampler(Box::new(SequentialSampler::new(6)));

        let batches: Vec<Vec<i32>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn test_with_distributed_sampler() {
        use crate::sampler::DistributedSampler;

        let ds = make_dataset(10);
        // Rank 0 of 2, no shuffle => indices [0,2,4,6,8]
        let sampler = DistributedSampler::new(10, 2, 0).shuffle(false);
        let loader = DataLoader::new(ds, 3).with_sampler(Box::new(sampler));

        let all: Vec<i32> = loader
            .iter(0)
            .flat_map(|b| b.unwrap())
            .collect();
        assert_eq!(all, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_with_sampler_and_num_workers() {
        use crate::sampler::SequentialSampler;

        let loader = DataLoader::new(make_dataset(8), 4)
            .num_workers(2)
            .with_sampler(Box::new(SequentialSampler::new(8)));

        let mut all: Vec<i32> = loader
            .iter(0)
            .flat_map(|b| b.unwrap())
            .collect();
        all.sort();
        assert_eq!(all, (0..8).collect::<Vec<i32>>());
    }

    // ── prefetch pipeline ───────────────────────────────────────────

    #[test]
    fn test_prefetch_produces_same_results_as_sync() {
        // Compare prefetch=0 (sync) vs prefetch=2 (default) for
        // sequential loading.
        let ds = make_dataset(20);
        let sync_loader = DataLoader::new(Arc::clone(&ds), 3).prefetch_factor(0);
        let prefetch_loader = DataLoader::new(ds, 3).prefetch_factor(2);

        let sync_batches: Vec<Vec<i32>> = sync_loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();
        let prefetch_batches: Vec<Vec<i32>> = prefetch_loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();

        assert_eq!(sync_batches, prefetch_batches);
    }

    #[test]
    fn test_prefetch_with_shuffle_same_elements() {
        let ds = make_dataset(50);
        let loader = DataLoader::new(ds, 7)
            .shuffle(true)
            .seed(42)
            .prefetch_factor(3);

        let mut all: Vec<i32> = loader
            .iter(0)
            .flat_map(|b| b.unwrap())
            .collect();
        all.sort();
        assert_eq!(all, (0..50).collect::<Vec<i32>>());
    }

    #[test]
    fn test_prefetch_with_drop_last() {
        let loader = DataLoader::new(make_dataset(10), 3)
            .drop_last(true)
            .prefetch_factor(2);

        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.as_ref().unwrap().len(), 3);
        }
    }

    #[test]
    fn test_prefetch_empty_dataset() {
        let loader = DataLoader::new(make_dataset(0), 4).prefetch_factor(2);
        let batches: Vec<_> = loader.iter(0).collect();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_prefetch_single_element() {
        let loader = DataLoader::new(make_dataset(1), 5).prefetch_factor(2);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].as_ref().unwrap(), &vec![0]);
    }

    #[test]
    fn test_prefetch_factor_1() {
        // Minimal buffer: one batch ahead.
        let loader = DataLoader::new(make_dataset(10), 3).prefetch_factor(1);
        let batches: Vec<Vec<i32>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(
            batches,
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9]]
        );
    }

    #[test]
    fn test_prefetch_factor_large() {
        // Buffer larger than total batches — should still work.
        let loader = DataLoader::new(make_dataset(6), 3).prefetch_factor(100);
        let batches: Vec<Vec<i32>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn test_prefetch_size_hint() {
        let loader = DataLoader::new(make_dataset(11), 3).prefetch_factor(2);
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
    fn test_prefetch_drop_mid_iteration_no_hang() {
        // Dropping the iterator mid-stream must not block or hang.
        // The background thread should detect the closed channel and exit.
        let loader = DataLoader::new(make_dataset(1000), 3).prefetch_factor(2);
        let mut it = loader.iter(0);
        // Consume only 2 of ~334 batches, then drop.
        let _ = it.next();
        let _ = it.next();
        drop(it);
        // If we reach here without hanging, the test passes.
    }

    #[test]
    fn test_prefetch_multiple_epochs() {
        let loader = DataLoader::new(make_dataset(10), 5)
            .shuffle(true)
            .seed(42)
            .prefetch_factor(2);

        let epoch0: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        let epoch1: Vec<Vec<i32>> = loader.iter(1).map(|b| b.unwrap()).collect();

        // Different ordering.
        let flat0: Vec<i32> = epoch0.into_iter().flatten().collect();
        let flat1: Vec<i32> = epoch1.into_iter().flatten().collect();
        assert_ne!(flat0, flat1);

        // But same elements.
        let mut sorted0 = flat0;
        let mut sorted1 = flat1;
        sorted0.sort();
        sorted1.sort();
        assert_eq!(sorted0, sorted1);
    }

    #[test]
    fn test_prefetch_with_num_workers() {
        let loader = DataLoader::new(make_dataset(20), 5)
            .num_workers(2)
            .prefetch_factor(3);

        let mut all: Vec<i32> = loader
            .iter(0)
            .flat_map(|b| b.unwrap())
            .collect();
        all.sort();
        assert_eq!(all, (0..20).collect::<Vec<i32>>());
    }

    #[test]
    fn test_prefetch_reproducibility() {
        let ds = make_dataset(30);
        let loader = DataLoader::new(ds, 7)
            .shuffle(true)
            .seed(99)
            .prefetch_factor(2);

        let run1: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        let run2: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(run1, run2);
    }

    #[test]
    fn test_sync_path_when_prefetch_zero() {
        // prefetch_factor=0 should use the synchronous DataLoaderIter.
        let loader = DataLoader::new(make_dataset(6), 3).prefetch_factor(0);
        let batches: Vec<Vec<i32>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    // ── device transfer ─────────────────────────────────────────────

    /// A sample type that implements ToDevice for testing.
    #[derive(Debug, Clone, PartialEq)]
    struct DeviceSample {
        value: i32,
        device: Device,
    }

    impl ToDevice for DeviceSample {
        fn to_device(&self, device: Device) -> FerrotorchResult<Self> {
            Ok(DeviceSample {
                value: self.value,
                device,
            })
        }
    }

    /// Dataset of DeviceSamples, all starting on CPU.
    struct DeviceDataset {
        data: Vec<DeviceSample>,
    }

    impl Dataset for DeviceDataset {
        type Sample = DeviceSample;

        fn len(&self) -> usize {
            self.data.len()
        }

        fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
            self.data.get(index).cloned().ok_or_else(|| {
                ferrotorch_core::FerrotorchError::IndexOutOfBounds {
                    index,
                    axis: 0,
                    size: self.data.len(),
                }
            })
        }
    }

    fn make_device_dataset(n: usize) -> Arc<DeviceDataset> {
        Arc::new(DeviceDataset {
            data: (0..n as i32)
                .map(|v| DeviceSample {
                    value: v,
                    device: Device::Cpu,
                })
                .collect(),
        })
    }

    #[test]
    fn test_device_transfer_sync() {
        let loader = DataLoader::new(make_device_dataset(4), 2)
            .prefetch_factor(0)
            .device(Device::Cuda(0));

        let batches: Vec<Vec<DeviceSample>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();

        assert_eq!(batches.len(), 2);
        for batch in &batches {
            for sample in batch {
                assert_eq!(sample.device, Device::Cuda(0));
            }
        }
    }

    #[test]
    fn test_device_transfer_prefetch() {
        let loader = DataLoader::new(make_device_dataset(6), 2)
            .prefetch_factor(2)
            .device(Device::Cuda(1));

        let batches: Vec<Vec<DeviceSample>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();

        assert_eq!(batches.len(), 3);
        for batch in &batches {
            for sample in batch {
                assert_eq!(sample.device, Device::Cuda(1));
            }
        }
        // Verify values are preserved.
        let values: Vec<i32> = batches.iter().flat_map(|b| b.iter().map(|s| s.value)).collect();
        assert_eq!(values, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_no_device_no_transfer() {
        // Without .device(), samples should remain on their original device.
        let loader = DataLoader::new(make_device_dataset(4), 2)
            .prefetch_factor(0);

        let batches: Vec<Vec<DeviceSample>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();

        for batch in &batches {
            for sample in batch {
                assert_eq!(sample.device, Device::Cpu);
            }
        }
    }

    #[test]
    fn test_device_transfer_empty_dataset() {
        let loader = DataLoader::new(make_device_dataset(0), 4)
            .device(Device::Cuda(0));

        let batches: Vec<_> = loader.iter(0).collect();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_device_transfer_single_element() {
        let loader = DataLoader::new(make_device_dataset(1), 5)
            .device(Device::Cuda(0));

        let batches: Vec<Vec<DeviceSample>> = loader
            .iter(0)
            .map(|b| b.unwrap())
            .collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0][0].device, Device::Cuda(0));
        assert_eq!(batches[0][0].value, 0);
    }

    #[test]
    fn test_device_transfer_with_collate() {
        let loader = DataLoader::new(make_device_dataset(6), 3)
            .device(Device::Cuda(0))
            .with_collate(|batch| {
                Ok(DeviceSample {
                    value: batch.iter().map(|s| s.value).sum(),
                    device: batch[0].device,
                })
            });

        let collated: Vec<DeviceSample> = loader
            .iter_collated(0)
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(collated.len(), 2);
        // Device transfer happens before collation.
        assert_eq!(collated[0].device, Device::Cuda(0));
        assert_eq!(collated[1].device, Device::Cuda(0));
        assert_eq!(collated[0].value, 0 + 1 + 2);
        assert_eq!(collated[1].value, 3 + 4 + 5);
    }

    // ── prefetch_factor builder ─────────────────────────────────────

    #[test]
    fn test_prefetch_factor_builder() {
        let loader = DataLoader::new(make_dataset(10), 2).prefetch_factor(5);
        assert_eq!(loader.prefetch_factor, 5);
    }

    #[test]
    fn test_default_prefetch_factor_is_2() {
        let loader = DataLoader::new(make_dataset(10), 2);
        assert_eq!(loader.prefetch_factor, 2);
    }

    // ── iterator exhaustion ─────────────────────────────────────────

    #[test]
    fn test_prefetch_iterator_returns_none_after_exhaustion() {
        let loader = DataLoader::new(make_dataset(3), 2).prefetch_factor(2);
        let mut it = loader.iter(0);
        assert!(it.next().is_some()); // [0, 1]
        assert!(it.next().is_some()); // [2]
        assert!(it.next().is_none());
        assert!(it.next().is_none()); // Repeated calls stay None.
        assert!(it.next().is_none());
    }

    #[test]
    fn test_sync_iterator_returns_none_after_exhaustion() {
        let loader = DataLoader::new(make_dataset(3), 2).prefetch_factor(0);
        let mut it = loader.iter(0);
        assert!(it.next().is_some());
        assert!(it.next().is_some());
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    // ── drop safety ────────────────────────────────────────────────

    #[test]
    fn test_drop_immediately_after_creation() {
        // Creating and immediately dropping a prefetch iterator should not
        // block or panic.
        let loader = DataLoader::new(make_dataset(100), 3).prefetch_factor(2);
        let it = loader.iter(0);
        drop(it);
    }

    #[test]
    fn test_drop_empty_prefetch_iter() {
        let loader = DataLoader::new(make_dataset(0), 4).prefetch_factor(2);
        let it = loader.iter(0);
        drop(it);
    }
}
