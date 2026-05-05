use std::collections::{BinaryHeap, VecDeque};
use std::sync::mpsc::{self, Receiver, SyncSender};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

use ferrotorch_core::{Device, FerrotorchError, FerrotorchResult};
use rayon::prelude::*;

use crate::dataset::Dataset;
use crate::sampler::{RandomSampler, Sampler, SequentialSampler};

// ---------------------------------------------------------------------------
// WorkerMode
// ---------------------------------------------------------------------------

/// Parallelism strategy for worker threads. Mirrors the split between
/// PyTorch `DataLoader`'s process-based workers and the existing
/// rayon-based intra-batch parallelism.
///
/// - [`IntraBatch`](WorkerMode::IntraBatch) (default, preserves
///   existing behavior): batches are loaded one at a time, but samples
///   within each batch are loaded in parallel using rayon's
///   work-stealing thread pool sized by `num_workers`. A single
///   background prefetch thread is used when `prefetch_factor > 0`.
///
/// - [`CrossBatch`](WorkerMode::CrossBatch): spawn `num_workers`
///   dedicated worker threads that independently produce full batches
///   in parallel. This mirrors PyTorch's multi-process DataLoader
///   pipeline (except using threads instead of processes, since Rust
///   has no GIL to work around). Batches are reordered to preserve
///   deterministic output ordering. CL-377.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum WorkerMode {
    /// Load samples within a single batch in parallel (rayon pool).
    #[default]
    IntraBatch,
    /// Load different batches in parallel (N dedicated worker threads).
    CrossBatch,
}

/// Trait for sample types that can be transferred to a target [`Device`].
///
/// Provide an impl on your `Dataset::Sample` type to enable automatic device
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
///
///     // Optional: override for the pinned-memory fast path. The default
///     // forwards to `to_device` so existing impls keep working.
///     fn to_device_pinned(&self, device: Device) -> FerrotorchResult<Self> {
///         Ok(MyBatch {
///             inputs: self.inputs.to_pinned(device)?,
///             labels: self.labels.to_pinned(device)?,
///         })
///     }
/// }
/// ```
pub trait ToDevice: Sized {
    /// Transfer this value to the given device.
    fn to_device(&self, device: Device) -> FerrotorchResult<Self>;

    /// Transfer this value to the given device using pinned (page-locked)
    /// host memory for any CPU→GPU copies.
    ///
    /// The default implementation forwards to [`to_device`](Self::to_device),
    /// preserving compatibility with existing impls. Override this when your
    /// sample type contains tensors and you want the
    /// [`DataLoader::pin_memory`] fast path to actually use pinned memory.
    /// Typical override: replace any `tensor.to(device)` calls with
    /// `tensor.to_pinned(device)`.
    fn to_device_pinned(&self, device: Device) -> FerrotorchResult<Self> {
        self.to_device(device)
    }
}

/// Blanket impl: a `Tensor<T>` is itself a `ToDevice` sample. Useful for
/// loaders that yield raw tensor batches without a wrapper struct, e.g.
/// `DataLoader<TensorDataset<f32>>`.
impl<T: ferrotorch_core::Float> ToDevice for ferrotorch_core::Tensor<T> {
    fn to_device(&self, device: Device) -> FerrotorchResult<Self> {
        self.to(device)
    }
    fn to_device_pinned(&self, device: Device) -> FerrotorchResult<Self> {
        self.to_pinned(device)
    }
}

/// Type alias for the device-transfer function stored internally.
///
/// The boolean argument is the `pin_memory` flag. When `true`, the closure
/// uses `ToDevice::to_device_pinned` (page-locked host memory + DMA);
/// when `false`, it uses `ToDevice::to_device` (regular pageable copy).
type TransferFn<S> = Arc<dyn Fn(Vec<S>, bool) -> FerrotorchResult<Vec<S>> + Send + Sync>;

/// Type alias for the collation function stored internally.
type CollateFn<S> = Arc<dyn Fn(Vec<S>) -> FerrotorchResult<S> + Send + Sync>;

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
/// let loader = DataLoader::new(Arc::new(ds), 2).unwrap();
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
    worker_mode: WorkerMode,
    device: Option<Device>,
    pin_memory: bool,
    custom_sampler: Option<Box<dyn Sampler>>,
    collate_fn: Option<CollateFn<D::Sample>>,
    transfer_fn: Option<TransferFn<D::Sample>>,
}

// Manual `Debug` impl: `Box<dyn Sampler>`, `CollateFn`, and `TransferFn`
// hold trait objects / `Arc<dyn Fn>` closures that do not implement
// `Debug`. We elide their content with a presence indicator so the
// derived format remains useful for diagnostics.
impl<D: Dataset> std::fmt::Debug for DataLoader<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataLoader")
            .field("batch_size", &self.batch_size)
            .field("shuffle", &self.shuffle)
            .field("drop_last", &self.drop_last)
            .field("seed", &self.seed)
            .field("num_workers", &self.num_workers)
            .field("prefetch_factor", &self.prefetch_factor)
            .field("worker_mode", &self.worker_mode)
            .field("device", &self.device)
            .field("pin_memory", &self.pin_memory)
            .field("has_custom_sampler", &self.custom_sampler.is_some())
            .field("has_collate_fn", &self.collate_fn.is_some())
            .field("has_transfer_fn", &self.transfer_fn.is_some())
            .finish()
    }
}

impl<D: Dataset> DataLoader<D> {
    /// Create a new `DataLoader` for the given dataset and batch size.
    ///
    /// Defaults: sequential order, keep the final partial batch, seed 0,
    /// prefetch_factor 2, no device transfer.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `batch_size == 0`.
    pub fn new(dataset: Arc<D>, batch_size: usize) -> FerrotorchResult<Self> {
        if batch_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "DataLoader: batch_size must be > 0".into(),
            });
        }
        Ok(Self {
            dataset,
            batch_size,
            shuffle: false,
            drop_last: false,
            seed: 0,
            num_workers: 0,
            prefetch_factor: 2,
            worker_mode: WorkerMode::IntraBatch,
            device: None,
            pin_memory: false,
            custom_sampler: None,
            collate_fn: None,
            transfer_fn: None,
        })
    }

    /// Select the worker parallelism strategy.
    ///
    /// - [`WorkerMode::IntraBatch`] (default) loads samples within a
    ///   single batch in parallel via rayon.
    /// - [`WorkerMode::CrossBatch`] spawns `num_workers` dedicated
    ///   threads that produce independent batches in parallel, with a
    ///   reorder buffer to preserve ordering. CL-377.
    pub fn worker_mode(mut self, mode: WorkerMode) -> Self {
        self.worker_mode = mode;
        self
    }

    /// Returns the current worker mode.
    pub fn current_worker_mode(&self) -> WorkerMode {
        self.worker_mode
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
        // The closure receives `pin_memory` as a runtime flag so the user
        // can flip pin_memory on or off via the builder after device() has
        // been called. The branch picks the pinned vs. regular ToDevice
        // method based on the flag's value at call time.
        self.transfer_fn = Some(Arc::new(
            move |samples: Vec<D::Sample>, pin_memory: bool| {
                if pin_memory {
                    samples
                        .into_iter()
                        .map(|s| s.to_device_pinned(device))
                        .collect()
                } else {
                    samples.into_iter().map(|s| s.to_device(device)).collect()
                }
            },
        ));
        self
    }

    /// Enable pinned memory for CPU→GPU transfers.
    ///
    /// When `true` and a target device is set, batch tensors are transferred
    /// to GPU via page-locked (pinned) host memory, which enables DMA
    /// transfers (~2x faster for large batches). Has no effect when the
    /// target device is CPU.
    ///
    /// Corresponds to PyTorch's `DataLoader(pin_memory=True)`.
    pub fn pin_memory(mut self, enable: bool) -> Self {
        self.pin_memory = enable;
        self
    }

    /// Returns `true` if pin_memory is enabled.
    pub fn is_pin_memory(&self) -> bool {
        self.pin_memory
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
    pub fn collate_fn(&self) -> Option<&CollateFn<D::Sample>> {
        self.collate_fn.as_ref()
    }

    /// Produce a collated batch iterator for the given epoch.
    ///
    /// Each call to `next()` returns a single collated `D::Sample` produced
    /// by the collation function set via [`with_collate`](DataLoader::with_collate).
    ///
    /// # Errors
    ///
    /// Returns `Err(FerrotorchError::InvalidArgument)` if no collation
    /// function has been set. Use [`iter`](DataLoader::iter) for the
    /// uncollated path, or call [`with_collate`](DataLoader::with_collate)
    /// first.
    pub fn iter_collated(&self, epoch: usize) -> FerrotorchResult<CollatedIter<'_, D>>
    where
        D: 'static,
        D::Sample: 'static,
    {
        let collate_fn =
            self.collate_fn
                .as_ref()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: "iter_collated called without a collate_fn — use with_collate() first"
                        .into(),
                })?;

        Ok(CollatedIter {
            inner: self.iter(epoch),
            collate_fn: collate_fn.as_ref(),
        })
    }

    /// Return the number of batches that will be produced for one epoch.
    pub fn len(&self) -> usize {
        let n = self.dataset.len();
        if self.drop_last {
            n / self.batch_size
        } else {
            n.div_ceil(self.batch_size)
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

        // CrossBatch mode with > 0 workers uses the dedicated
        // multi-worker pipeline. Any other combination falls back to
        // the existing Prefetch / Sync code paths.
        if self.worker_mode == WorkerMode::CrossBatch && self.num_workers > 0 {
            return BatchIter::MultiWorker(MultiWorkerIter::new(
                Arc::clone(&self.dataset),
                indices,
                self.batch_size,
                self.drop_last,
                self.num_workers,
                self.prefetch_factor.max(self.num_workers),
                self.transfer_fn.clone(),
                self.pin_memory,
            ));
        }

        if self.prefetch_factor > 0 {
            BatchIter::Prefetch(PrefetchIter::new(
                Arc::clone(&self.dataset),
                indices,
                self.batch_size,
                self.drop_last,
                self.num_workers,
                self.prefetch_factor,
                self.transfer_fn.clone(),
                self.pin_memory,
            ))
        } else {
            BatchIter::Sync(DataLoaderIter {
                dataset: &self.dataset,
                indices,
                batch_size: self.batch_size,
                drop_last: self.drop_last,
                num_workers: self.num_workers,
                transfer_fn: self.transfer_fn.as_ref(),
                pin_memory: self.pin_memory,
                pos: 0,
            })
        }
    }
}

/// Iterator over batches — synchronous, single-thread prefetched, or
/// multi-worker. Returned by [`DataLoader::iter`]. Transparent to the
/// caller: all variants yield `FerrotorchResult<Vec<D::Sample>>` in
/// sampler order.
pub enum BatchIter<'a, D: Dataset> {
    /// Synchronous loading on the calling thread.
    Sync(DataLoaderIter<'a, D>),
    /// Prefetched loading via a single background thread with
    /// optional rayon-parallel sample loading within each batch.
    Prefetch(PrefetchIter<D>),
    /// Cross-batch parallel loading via `num_workers` dedicated
    /// threads, with a reorder buffer to preserve sampler order.
    /// CL-377.
    MultiWorker(MultiWorkerIter<D>),
}

// Manual `Debug`: variants wrap iterator state with non-`Debug` fields
// (closures, channels, thread handles); print the variant tag only.
impl<D: Dataset> std::fmt::Debug for BatchIter<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BatchIter::Sync(_) => f.debug_tuple("BatchIter::Sync").finish(),
            BatchIter::Prefetch(_) => f.debug_tuple("BatchIter::Prefetch").finish(),
            BatchIter::MultiWorker(_) => f.debug_tuple("BatchIter::MultiWorker").finish(),
        }
    }
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
            BatchIter::MultiWorker(inner) => inner.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            BatchIter::Sync(inner) => inner.size_hint(),
            BatchIter::Prefetch(inner) => inner.size_hint(),
            BatchIter::MultiWorker(inner) => inner.size_hint(),
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
    /// Whether to use pinned-memory CPU→GPU transfers in the transfer_fn.
    /// Only meaningful when transfer_fn is set and the target device is CUDA.
    pin_memory: bool,
    pos: usize,
}

// Manual `Debug`: `dataset: &D` and `transfer_fn: Option<&Arc<dyn Fn ...>>`
// are not `Debug`-bound. Elide both with presence indicators.
impl<D: Dataset> std::fmt::Debug for DataLoaderIter<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DataLoaderIter")
            .field("indices_len", &self.indices.len())
            .field("batch_size", &self.batch_size)
            .field("drop_last", &self.drop_last)
            .field("num_workers", &self.num_workers)
            .field("has_transfer_fn", &self.transfer_fn.is_some())
            .field("pin_memory", &self.pin_memory)
            .field("pos", &self.pos)
            .finish()
    }
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

        // Apply device transfer if configured. Pass the pin_memory flag so
        // the transfer_fn closure picks the pinned vs. regular path.
        let batch = match (batch, &self.transfer_fn) {
            (Ok(samples), Some(f)) => f(samples, self.pin_memory),
            (result, _) => result,
        };

        Some(batch)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.indices.len().saturating_sub(self.pos);
        let batches = if self.drop_last {
            remaining / self.batch_size
        } else {
            remaining.div_ceil(self.batch_size)
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

// Manual `Debug`: `Receiver` and `JoinHandle` are not `Debug` for
// arbitrary `D::Sample`. Print only the bookkeeping counters that are
// useful for diagnostics.
impl<D: Dataset> std::fmt::Debug for PrefetchIter<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PrefetchIter")
            .field("total_batches", &self.total_batches)
            .field("consumed", &self.consumed)
            .finish()
    }
}

impl<D: Dataset + 'static> PrefetchIter<D>
where
    D::Sample: Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset: Arc<D>,
        indices: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        prefetch_factor: usize,
        transfer_fn: Option<TransferFn<D::Sample>>,
        pin_memory: bool,
    ) -> Self {
        let total_batches = compute_batch_count(indices.len(), batch_size, drop_last);

        // Bounded channel — at most `prefetch_factor` batches buffered.
        let (tx, rx) =
            mpsc::sync_channel::<FerrotorchResult<Vec<D::Sample>>>(prefetch_factor.max(1));

        let handle = thread::spawn(move || {
            Self::producer_loop(
                dataset,
                indices,
                batch_size,
                drop_last,
                num_workers,
                transfer_fn,
                pin_memory,
                tx,
            );
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
    #[allow(clippy::too_many_arguments)]
    fn producer_loop(
        dataset: Arc<D>,
        indices: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        transfer_fn: Option<TransferFn<D::Sample>>,
        pin_memory: bool,
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

            // Apply device transfer if configured. Pass pin_memory so the
            // transfer_fn closure picks the pinned-memory CPU→GPU path
            // when both pin_memory and a CUDA target device are set.
            let batch = match (batch, &transfer_fn) {
                (Ok(samples), Some(f)) => f(samples, pin_memory),
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

// ---------------------------------------------------------------------------
// MultiWorkerIter — cross-batch parallel loading
// ---------------------------------------------------------------------------

/// A single work item sent from the dispatcher to a worker thread.
///
/// `seq` is the batch's position in the sampler's output order. It is
/// used by the reorder buffer to yield results deterministically even
/// though worker threads complete batches in arbitrary order.
struct WorkItem {
    seq: usize,
    indices: Vec<usize>,
}

/// A completed batch produced by a worker, tagged with its sequence
/// number so the consumer can reorder.
struct WorkResult<S> {
    seq: usize,
    batch: FerrotorchResult<Vec<S>>,
}

/// Ordering helper for the `BinaryHeap` reorder buffer: we pop the
/// smallest-seq batch first, so we implement `Ord` in reverse.
struct SeqEntry<S> {
    seq: usize,
    batch: FerrotorchResult<Vec<S>>,
}

impl<S> PartialEq for SeqEntry<S> {
    fn eq(&self, other: &Self) -> bool {
        self.seq == other.seq
    }
}

impl<S> Eq for SeqEntry<S> {}

impl<S> PartialOrd for SeqEntry<S> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<S> Ord for SeqEntry<S> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse so BinaryHeap::pop returns the smallest seq.
        other.seq.cmp(&self.seq)
    }
}

/// Cross-batch multi-worker iterator. `num_workers` threads each pull
/// `WorkItem`s from a shared work queue, load the batch (applying
/// device transfer if configured), and push a `WorkResult` back on a
/// result channel. The iterator reorders results by sequence number
/// so the consumer sees them in sampler order.
///
/// Bounded parallelism: at most `in_flight` work items are dispatched
/// at once so memory use stays predictable.
pub struct MultiWorkerIter<D: Dataset> {
    result_rx: Receiver<WorkResult<D::Sample>>,
    /// Work queue shared with the worker threads; the dispatcher
    /// pushes onto the back and workers pop from the front under the
    /// mutex. `None` as a sentinel tells workers to shut down.
    work_queue: Arc<Mutex<VecDeque<Option<WorkItem>>>>,
    /// Condvar signaled when new work is pushed or when the queue is
    /// closed (so workers waiting on an empty queue can wake up).
    work_cv: Arc<std::sync::Condvar>,
    /// Next sequence number the dispatcher should hand out.
    next_dispatch_seq: usize,
    /// Next sequence number the consumer should yield.
    next_yield_seq: usize,
    /// Reorder buffer for out-of-order completions.
    reorder_buf: BinaryHeap<SeqEntry<D::Sample>>,
    /// Number of batches currently out with a worker (dispatched but
    /// not yet collected).
    in_flight_count: usize,
    /// Maximum allowed `in_flight_count`.
    max_in_flight: usize,
    /// Total number of batches this iterator will produce.
    total_batches: usize,
    /// Pre-computed list of batch index slices (seq → Vec<usize>).
    batch_plans: Vec<Vec<usize>>,
    /// Worker thread handles. Joined on drop.
    worker_handles: Vec<JoinHandle<()>>,
    /// Whether the work queue has been closed (sentinel sent).
    closed: bool,
}

// Manual `Debug`: `Receiver`, `Mutex<VecDeque<Option<WorkItem>>>`,
// `BinaryHeap<SeqEntry<D::Sample>>`, and `JoinHandle` either lack
// `Debug` for arbitrary `D::Sample` or carry per-thread state that is
// not useful in diagnostics. Print only the dispatcher bookkeeping.
impl<D: Dataset> std::fmt::Debug for MultiWorkerIter<D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MultiWorkerIter")
            .field("next_dispatch_seq", &self.next_dispatch_seq)
            .field("next_yield_seq", &self.next_yield_seq)
            .field("in_flight_count", &self.in_flight_count)
            .field("max_in_flight", &self.max_in_flight)
            .field("total_batches", &self.total_batches)
            .field("workers", &self.worker_handles.len())
            .field("closed", &self.closed)
            .finish()
    }
}

impl<D: Dataset + 'static> MultiWorkerIter<D>
where
    D::Sample: Send + 'static,
{
    #[allow(clippy::too_many_arguments)]
    fn new(
        dataset: Arc<D>,
        indices: Vec<usize>,
        batch_size: usize,
        drop_last: bool,
        num_workers: usize,
        max_in_flight: usize,
        transfer_fn: Option<TransferFn<D::Sample>>,
        pin_memory: bool,
    ) -> Self {
        // Plan all batches up front so workers just need to receive
        // their slice of indices.
        let total_batches = compute_batch_count(indices.len(), batch_size, drop_last);
        let mut batch_plans: Vec<Vec<usize>> = Vec::with_capacity(total_batches);
        let mut pos = 0;
        while pos < indices.len() {
            let remaining = indices.len() - pos;
            if drop_last && remaining < batch_size {
                break;
            }
            let end = (pos + batch_size).min(indices.len());
            batch_plans.push(indices[pos..end].to_vec());
            pos = end;
        }

        let work_queue: Arc<Mutex<VecDeque<Option<WorkItem>>>> =
            Arc::new(Mutex::new(VecDeque::new()));
        let work_cv = Arc::new(std::sync::Condvar::new());
        let (result_tx, result_rx) =
            mpsc::sync_channel::<WorkResult<D::Sample>>(max_in_flight.max(1));

        // Spawn worker threads.
        let mut worker_handles = Vec::with_capacity(num_workers);
        for _ in 0..num_workers {
            let dataset_w = Arc::clone(&dataset);
            let queue_w = Arc::clone(&work_queue);
            let cv_w = Arc::clone(&work_cv);
            let tx_w = result_tx.clone();
            let transfer_w = transfer_fn.clone();
            let handle = thread::spawn(move || {
                worker_loop(dataset_w, queue_w, cv_w, tx_w, transfer_w, pin_memory);
            });
            worker_handles.push(handle);
        }
        // Drop our clone of the sender so the channel closes when all
        // workers drop theirs.
        drop(result_tx);

        MultiWorkerIter {
            result_rx,
            work_queue,
            work_cv,
            next_dispatch_seq: 0,
            next_yield_seq: 0,
            reorder_buf: BinaryHeap::new(),
            in_flight_count: 0,
            max_in_flight: max_in_flight.max(1),
            total_batches,
            batch_plans,
            worker_handles,
            closed: false,
        }
    }

    /// Push more work items to the queue until we hit max_in_flight or
    /// exhaust the plan.
    ///
    /// Returns `Err(LockPoisoned)` if the work queue mutex is poisoned
    /// (i.e., a worker panicked while holding the lock).
    fn refill_work_queue(&mut self) -> FerrotorchResult<()> {
        while self.in_flight_count < self.max_in_flight
            && self.next_dispatch_seq < self.total_batches
        {
            let seq = self.next_dispatch_seq;
            let indices = self.batch_plans[seq].clone();
            {
                let mut queue =
                    self.work_queue
                        .lock()
                        .map_err(|e| FerrotorchError::LockPoisoned {
                            message: format!("MultiWorkerIter work queue poisoned: {e}"),
                        })?;
                queue.push_back(Some(WorkItem { seq, indices }));
            }
            self.work_cv.notify_one();
            self.next_dispatch_seq += 1;
            self.in_flight_count += 1;
        }
        Ok(())
    }

    /// Signal all workers to exit by pushing one `None` sentinel per
    /// worker and notifying all waiters.
    ///
    /// Returns `Err(LockPoisoned)` if the work queue mutex is poisoned.
    fn close_work_queue(&mut self) -> FerrotorchResult<()> {
        if self.closed {
            return Ok(());
        }
        {
            let mut queue = self
                .work_queue
                .lock()
                .map_err(|e| FerrotorchError::LockPoisoned {
                    message: format!("MultiWorkerIter work queue poisoned on close: {e}"),
                })?;
            for _ in 0..self.worker_handles.len() {
                queue.push_back(None);
            }
        }
        self.work_cv.notify_all();
        self.closed = true;
        Ok(())
    }
}

impl<D: Dataset + 'static> Iterator for MultiWorkerIter<D>
where
    D::Sample: Send + 'static,
{
    type Item = FerrotorchResult<Vec<D::Sample>>;

    fn next(&mut self) -> Option<Self::Item> {
        // Iteration done — close the work queue so workers exit and
        // return None. Idempotent across multiple next() calls.
        if self.next_yield_seq >= self.total_batches {
            // Ignore close errors here: if the mutex is poisoned at this
            // point the workers are already dead, so we just stop.
            let _ = self.close_work_queue();
            return None;
        }

        // Keep the workers busy; propagate lock-poison as an Err batch.
        if let Err(e) = self.refill_work_queue() {
            return Some(Err(e));
        }

        // Serve out-of-order results from the reorder buffer first.
        loop {
            if let Some(top) = self.reorder_buf.peek() {
                if top.seq == self.next_yield_seq {
                    // SAFETY: peek() returned Some, so pop() cannot return
                    // None — the heap is non-empty.
                    let entry = self
                        .reorder_buf
                        .pop()
                        .expect("invariant: heap non-empty after peek");
                    self.next_yield_seq += 1;
                    return Some(entry.batch);
                }
            }
            // Not in buffer yet — receive the next completion.
            match self.result_rx.recv() {
                Ok(WorkResult { seq, batch }) => {
                    self.in_flight_count -= 1;
                    self.reorder_buf.push(SeqEntry { seq, batch });
                    // Dispatch more work as soon as a slot frees up.
                    if let Err(e) = self.refill_work_queue() {
                        return Some(Err(e));
                    }
                }
                Err(_) => {
                    // Channel closed unexpectedly — no more batches.
                    return None;
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total_batches.saturating_sub(self.next_yield_seq);
        (remaining, Some(remaining))
    }
}

impl<D: Dataset + 'static> ExactSizeIterator for MultiWorkerIter<D> where D::Sample: Send + 'static {}

impl<D: Dataset> Drop for MultiWorkerIter<D> {
    fn drop(&mut self) {
        // Close the queue if the consumer dropped us before
        // exhausting the iterator.
        if !self.closed {
            // If the mutex is poisoned (a worker panicked while holding
            // it) we can't push sentinels, but notify_all() still wakes
            // waiting threads so they can observe the broken state.
            if let Ok(mut queue) = self.work_queue.lock() {
                for _ in 0..self.worker_handles.len() {
                    queue.push_back(None);
                }
            }
            self.work_cv.notify_all();
            self.closed = true;
        }
        // Join all workers. Panics propagate as thread-panic messages
        // but are not re-raised — dropping an iterator should never
        // panic.
        for handle in self.worker_handles.drain(..) {
            let _ = handle.join();
        }
    }
}

/// The worker loop executed by every spawned thread. Pulls one
/// `WorkItem` at a time from the shared queue, loads the batch,
/// applies the optional device transfer, and sends the `WorkResult`
/// back to the main thread. Exits when the queue yields a `None`
/// sentinel or when the result channel is closed.
fn worker_loop<D: Dataset + 'static>(
    dataset: Arc<D>,
    queue: Arc<Mutex<VecDeque<Option<WorkItem>>>>,
    cv: Arc<std::sync::Condvar>,
    tx: SyncSender<WorkResult<D::Sample>>,
    transfer_fn: Option<TransferFn<D::Sample>>,
    pin_memory: bool,
) where
    D::Sample: Send + 'static,
{
    loop {
        // Pop one work item or shutdown sentinel.
        let item_opt: Option<WorkItem> = {
            // If the mutex is poisoned, send a LockPoisoned error back
            // to the dispatcher and exit cleanly rather than panicking.
            let mut guard = match queue.lock() {
                Ok(g) => g,
                Err(e) => {
                    let _ = tx.send(WorkResult {
                        seq: usize::MAX,
                        batch: Err(FerrotorchError::LockPoisoned {
                            message: format!("worker queue mutex poisoned: {e}"),
                        }),
                    });
                    return;
                }
            };
            loop {
                if let Some(front) = guard.pop_front() {
                    break front;
                }
                // If the condvar wait returns a poisoned guard, extract
                // it anyway — the poison is on the mutex, and we've
                // already handled the lock-acquire case above; here the
                // guard came from a prior successful lock so we can
                // continue with the (possibly stale) state.
                guard = match cv.wait(guard) {
                    Ok(g) => g,
                    Err(poisoned) => poisoned.into_inner(),
                };
            }
        };

        let item = match item_opt {
            Some(it) => it,
            None => return, // shutdown
        };

        // Load the batch sequentially. Cross-batch workers already
        // give us batch-level parallelism, so intra-batch rayon
        // parallelism is unnecessary here (and counter-productive if
        // num_workers exceeds physical cores).
        let mut batch = Vec::with_capacity(item.indices.len());
        let mut err = None;
        for idx in item.indices {
            match dataset.get(idx) {
                Ok(sample) => batch.push(sample),
                Err(e) => {
                    err = Some(e);
                    break;
                }
            }
        }

        let result: FerrotorchResult<Vec<D::Sample>> = match err {
            Some(e) => Err(e),
            None => match &transfer_fn {
                Some(f) => f(batch, pin_memory),
                None => Ok(batch),
            },
        };

        // Send result back. If the receiver dropped, the consumer has
        // torn down the iterator — exit quietly.
        if tx
            .send(WorkResult {
                seq: item.seq,
                batch: result,
            })
            .is_err()
        {
            return;
        }
    }
}

/// Compute the number of batches for a given index count.
fn compute_batch_count(n_indices: usize, batch_size: usize, drop_last: bool) -> usize {
    if drop_last {
        n_indices / batch_size
    } else {
        n_indices.div_ceil(batch_size)
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

// Manual `Debug`: `collate_fn` is a `dyn Fn` reference and not `Debug`.
// Print the inner `BatchIter` variant tag and a presence indicator.
impl<D: Dataset> std::fmt::Debug for CollatedIter<'_, D> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CollatedIter")
            .field("inner", &self.inner)
            .field("has_collate_fn", &true)
            .finish()
    }
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
        let loader = DataLoader::new(make_dataset(10), 5).unwrap();
        assert_eq!(loader.len(), 2);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 2);
    }

    #[test]
    fn test_batch_count_with_remainder() {
        let loader = DataLoader::new(make_dataset(10), 3).unwrap();
        assert_eq!(loader.len(), 4); // ceil(10/3) = 4
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 4);
    }

    #[test]
    fn test_batch_count_single_element() {
        let loader = DataLoader::new(make_dataset(1), 5).unwrap();
        assert_eq!(loader.len(), 1);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 1);
    }

    #[test]
    fn test_empty_dataset() {
        let loader = DataLoader::new(make_dataset(0), 4).unwrap();
        assert!(loader.is_empty());
        assert_eq!(loader.len(), 0);
        let batches: Vec<_> = loader.iter(0).collect();
        assert!(batches.is_empty());
    }

    // ── batch sizes ─────────────────────────────────────────────────

    #[test]
    fn test_batch_sizes_exact() {
        let loader = DataLoader::new(make_dataset(6), 3).unwrap();
        let sizes: Vec<usize> = loader.iter(0).map(|b| b.unwrap().len()).collect();
        assert_eq!(sizes, vec![3, 3]);
    }

    #[test]
    fn test_batch_sizes_with_partial_last() {
        let loader = DataLoader::new(make_dataset(7), 3).unwrap();
        let sizes: Vec<usize> = loader.iter(0).map(|b| b.unwrap().len()).collect();
        assert_eq!(sizes, vec![3, 3, 1]);
    }

    #[test]
    fn test_all_samples_present_sequential() {
        let loader = DataLoader::new(make_dataset(10), 3).unwrap();
        let mut all: Vec<i32> = loader.iter(0).flat_map(|b| b.unwrap()).collect();
        all.sort();
        assert_eq!(all, (0..10).collect::<Vec<i32>>());
    }

    // ── drop_last ───────────────────────────────────────────────────

    #[test]
    fn test_drop_last_removes_partial_batch() {
        let loader = DataLoader::new(make_dataset(10), 3)
            .unwrap()
            .drop_last(true);
        assert_eq!(loader.len(), 3); // 10/3 = 3 full batches
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 3);
        for b in &batches {
            assert_eq!(b.as_ref().unwrap().len(), 3);
        }
    }

    #[test]
    fn test_drop_last_exact_keeps_all() {
        let loader = DataLoader::new(make_dataset(9), 3).unwrap().drop_last(true);
        assert_eq!(loader.len(), 3);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 3);
    }

    #[test]
    fn test_drop_last_smaller_than_batch() {
        let loader = DataLoader::new(make_dataset(2), 5).unwrap().drop_last(true);
        assert!(loader.is_empty());
        let batches: Vec<_> = loader.iter(0).collect();
        assert!(batches.is_empty());
    }

    // ── shuffle ─────────────────────────────────────────────────────

    #[test]
    fn test_shuffle_produces_different_order() {
        let loader = DataLoader::new(make_dataset(100), 100)
            .unwrap()
            .shuffle(true)
            .seed(42);
        let batch = loader.iter(0).next().unwrap().unwrap();
        let sequential: Vec<i32> = (0..100).collect();
        assert_ne!(
            batch, sequential,
            "shuffled batch should differ from sequential"
        );
    }

    #[test]
    fn test_shuffle_contains_all_elements() {
        let loader = DataLoader::new(make_dataset(20), 5)
            .unwrap()
            .shuffle(true)
            .seed(7);
        let mut all: Vec<i32> = loader.iter(0).flat_map(|b| b.unwrap()).collect();
        all.sort();
        assert_eq!(all, (0..20).collect::<Vec<i32>>());
    }

    #[test]
    fn test_shuffle_different_epochs() {
        let loader = DataLoader::new(make_dataset(50), 50)
            .unwrap()
            .shuffle(true)
            .seed(99);
        let epoch0 = loader.iter(0).next().unwrap().unwrap();
        let epoch1 = loader.iter(1).next().unwrap().unwrap();
        assert_ne!(
            epoch0, epoch1,
            "different epochs should produce different order"
        );
    }

    // ── reproducibility ─────────────────────────────────────────────

    #[test]
    fn test_reproducible_with_same_seed_and_epoch() {
        let loader = DataLoader::new(make_dataset(30), 10)
            .unwrap()
            .shuffle(true)
            .seed(42);
        let run1: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        let run2: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(run1, run2);
    }

    #[test]
    fn test_different_seeds_differ() {
        let ds = make_dataset(100);
        let loader_a = DataLoader::new(Arc::clone(&ds), 100)
            .unwrap()
            .shuffle(true)
            .seed(1);
        let loader_b = DataLoader::new(ds, 100).unwrap().shuffle(true).seed(2);
        let a = loader_a.iter(0).next().unwrap().unwrap();
        let b = loader_b.iter(0).next().unwrap().unwrap();
        assert_ne!(a, b);
    }

    // ── size_hint / ExactSizeIterator ───────────────────────────────

    #[test]
    fn test_size_hint_accurate() {
        let loader = DataLoader::new(make_dataset(11), 3).unwrap();
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
        let loader = DataLoader::new(make_dataset(11), 3)
            .unwrap()
            .drop_last(true);
        let it = loader.iter(0);
        assert_eq!(it.len(), 3);
    }

    // ── builder ergonomics ──────────────────────────────────────────

    #[test]
    fn test_zero_batch_size_returns_err() {
        let result = DataLoader::new(make_dataset(5), 0);
        assert!(result.is_err());
        if let Err(e) = result {
            let msg = format!("{e:?}");
            assert!(msg.contains("batch_size must be > 0"), "got: {msg}");
        }
    }

    #[test]
    fn test_builder_chaining() {
        let loader = DataLoader::new(make_dataset(10), 2)
            .unwrap()
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
            .unwrap()
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
            .unwrap()
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let collated: Vec<i32> = loader
            .iter_collated(0)
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        // Sequential: [0,1,2] -> 3, [3,4,5] -> 12
        assert_eq!(collated, vec![3, 12]);
    }

    #[test]
    fn test_collate_with_remainder() {
        let loader = DataLoader::new(make_dataset(5), 3)
            .unwrap()
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let collated: Vec<i32> = loader
            .iter_collated(0)
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        // [0,1,2] -> 3, [3,4] -> 7
        assert_eq!(collated, vec![3, 7]);
    }

    #[test]
    fn test_collate_with_drop_last() {
        let loader = DataLoader::new(make_dataset(5), 3)
            .unwrap()
            .drop_last(true)
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let collated: Vec<i32> = loader
            .iter_collated(0)
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        // Only [0,1,2] -> 3 (partial batch [3,4] dropped)
        assert_eq!(collated, vec![3]);
    }

    #[test]
    fn test_collate_fn_accessor() {
        let loader = DataLoader::new(make_dataset(5), 3).unwrap();
        assert!(loader.collate_fn().is_none());

        let loader = loader.with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));
        assert!(loader.collate_fn().is_some());
    }

    #[test]
    fn test_collated_iter_size_hint() {
        let loader = DataLoader::new(make_dataset(10), 3)
            .unwrap()
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let it = loader.iter_collated(0).unwrap();
        assert_eq!(it.len(), 4); // ceil(10/3) = 4
    }

    #[test]
    fn test_collate_error_propagation() {
        let loader = DataLoader::new(make_dataset(4), 2)
            .unwrap()
            .with_collate(|_batch| {
                Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                    message: "test error".into(),
                })
            });

        let results: Vec<_> = loader.iter_collated(0).unwrap().collect();
        assert_eq!(results.len(), 2);
        assert!(results[0].is_err());
        assert!(results[1].is_err());
    }

    #[test]
    fn test_collated_iter_err_without_collate_fn() {
        // iter_collated now returns Err instead of panicking when no
        // collate_fn is set — callers can handle the misuse gracefully.
        let loader = DataLoader::new(make_dataset(5), 3).unwrap();
        assert!(loader.iter_collated(0).is_err());
    }

    #[test]
    fn test_uncollated_iter_unaffected_by_collate() {
        // The original iter() path still returns Vec<Sample> even when
        // a collate_fn is set.
        let loader = DataLoader::new(make_dataset(4), 2)
            .unwrap()
            .with_collate(|batch| Ok(batch.into_iter().sum::<i32>()));

        let batches: Vec<Vec<i32>> = loader.iter(0).map(|r| r.unwrap()).collect();
        assert_eq!(batches, vec![vec![0, 1], vec![2, 3]]);
    }

    // ── num_workers ─────────────────────────────────────────────────

    #[test]
    fn test_num_workers_builder() {
        let loader = DataLoader::new(make_dataset(10), 2).unwrap().num_workers(4);
        assert_eq!(loader.num_workers, 4);
    }

    #[test]
    fn test_num_workers_parallel_loads_all_samples() {
        let loader = DataLoader::new(make_dataset(20), 5).unwrap().num_workers(2);
        let mut all: Vec<i32> = loader.iter(0).flat_map(|b| b.unwrap()).collect();
        all.sort();
        assert_eq!(all, (0..20).collect::<Vec<i32>>());
    }

    #[test]
    fn test_num_workers_parallel_batch_sizes() {
        let loader = DataLoader::new(make_dataset(7), 3).unwrap().num_workers(2);
        let sizes: Vec<usize> = loader.iter(0).map(|b| b.unwrap().len()).collect();
        assert_eq!(sizes, vec![3, 3, 1]);
    }

    #[test]
    fn test_num_workers_parallel_drop_last() {
        let loader = DataLoader::new(make_dataset(10), 3)
            .unwrap()
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
            .unwrap()
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
            .unwrap()
            .num_workers(0)
            .prefetch_factor(0);
        let batches: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(
            batches,
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9]]
        );
    }

    // ── custom sampler ──────────────────────────────────────────────

    #[test]
    fn test_with_sampler_overrides_shuffle() {
        use crate::sampler::SequentialSampler;

        // Even with shuffle=true, a custom sampler takes precedence.
        let loader = DataLoader::new(make_dataset(6), 3)
            .unwrap()
            .shuffle(true)
            .seed(42)
            .with_sampler(Box::new(SequentialSampler::new(6)));

        let batches: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn test_with_distributed_sampler() {
        use crate::sampler::DistributedSampler;

        let ds = make_dataset(10);
        // Rank 0 of 2, no shuffle => indices [0,2,4,6,8]
        let sampler = DistributedSampler::new(10, 2, 0).shuffle(false);
        let loader = DataLoader::new(ds, 3)
            .unwrap()
            .with_sampler(Box::new(sampler));

        let all: Vec<i32> = loader.iter(0).flat_map(|b| b.unwrap()).collect();
        assert_eq!(all, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_with_sampler_and_num_workers() {
        use crate::sampler::SequentialSampler;

        let loader = DataLoader::new(make_dataset(8), 4)
            .unwrap()
            .num_workers(2)
            .with_sampler(Box::new(SequentialSampler::new(8)));

        let mut all: Vec<i32> = loader.iter(0).flat_map(|b| b.unwrap()).collect();
        all.sort();
        assert_eq!(all, (0..8).collect::<Vec<i32>>());
    }

    // ── prefetch pipeline ───────────────────────────────────────────

    #[test]
    fn test_prefetch_produces_same_results_as_sync() {
        // Compare prefetch=0 (sync) vs prefetch=2 (default) for
        // sequential loading.
        let ds = make_dataset(20);
        let sync_loader = DataLoader::new(Arc::clone(&ds), 3)
            .unwrap()
            .prefetch_factor(0);
        let prefetch_loader = DataLoader::new(ds, 3).unwrap().prefetch_factor(2);

        let sync_batches: Vec<Vec<i32>> = sync_loader.iter(0).map(|b| b.unwrap()).collect();
        let prefetch_batches: Vec<Vec<i32>> = prefetch_loader.iter(0).map(|b| b.unwrap()).collect();

        assert_eq!(sync_batches, prefetch_batches);
    }

    #[test]
    fn test_prefetch_with_shuffle_same_elements() {
        let ds = make_dataset(50);
        let loader = DataLoader::new(ds, 7)
            .unwrap()
            .shuffle(true)
            .seed(42)
            .prefetch_factor(3);

        let mut all: Vec<i32> = loader.iter(0).flat_map(|b| b.unwrap()).collect();
        all.sort();
        assert_eq!(all, (0..50).collect::<Vec<i32>>());
    }

    #[test]
    fn test_prefetch_with_drop_last() {
        let loader = DataLoader::new(make_dataset(10), 3)
            .unwrap()
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
        let loader = DataLoader::new(make_dataset(0), 4)
            .unwrap()
            .prefetch_factor(2);
        let batches: Vec<_> = loader.iter(0).collect();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_prefetch_single_element() {
        let loader = DataLoader::new(make_dataset(1), 5)
            .unwrap()
            .prefetch_factor(2);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].as_ref().unwrap(), &vec![0]);
    }

    #[test]
    fn test_prefetch_factor_1() {
        // Minimal buffer: one batch ahead.
        let loader = DataLoader::new(make_dataset(10), 3)
            .unwrap()
            .prefetch_factor(1);
        let batches: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(
            batches,
            vec![vec![0, 1, 2], vec![3, 4, 5], vec![6, 7, 8], vec![9]]
        );
    }

    #[test]
    fn test_prefetch_factor_large() {
        // Buffer larger than total batches — should still work.
        let loader = DataLoader::new(make_dataset(6), 3)
            .unwrap()
            .prefetch_factor(100);
        let batches: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    #[test]
    fn test_prefetch_size_hint() {
        let loader = DataLoader::new(make_dataset(11), 3)
            .unwrap()
            .prefetch_factor(2);
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
        let loader = DataLoader::new(make_dataset(1000), 3)
            .unwrap()
            .prefetch_factor(2);
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
            .unwrap()
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
            .unwrap()
            .num_workers(2)
            .prefetch_factor(3);

        let mut all: Vec<i32> = loader.iter(0).flat_map(|b| b.unwrap()).collect();
        all.sort();
        assert_eq!(all, (0..20).collect::<Vec<i32>>());
    }

    #[test]
    fn test_prefetch_reproducibility() {
        let ds = make_dataset(30);
        let loader = DataLoader::new(ds, 7)
            .unwrap()
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
        let loader = DataLoader::new(make_dataset(6), 3)
            .unwrap()
            .prefetch_factor(0);
        let batches: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches, vec![vec![0, 1, 2], vec![3, 4, 5]]);
    }

    // ── device transfer ─────────────────────────────────────────────

    /// A sample type that implements ToDevice for testing.
    /// `pinned` records which transfer method was called: `Some(true)` if
    /// `to_device_pinned`, `Some(false)` if `to_device`, `None` if neither.
    #[derive(Debug, Clone, PartialEq)]
    struct DeviceSample {
        value: i32,
        device: Device,
        pinned: Option<bool>,
    }

    impl ToDevice for DeviceSample {
        fn to_device(&self, device: Device) -> FerrotorchResult<Self> {
            Ok(DeviceSample {
                value: self.value,
                device,
                pinned: Some(false),
            })
        }
        fn to_device_pinned(&self, device: Device) -> FerrotorchResult<Self> {
            Ok(DeviceSample {
                value: self.value,
                device,
                pinned: Some(true),
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
            self.data.get(index).cloned().ok_or({
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
                    pinned: None,
                })
                .collect(),
        })
    }

    #[test]
    fn test_device_transfer_sync() {
        let loader = DataLoader::new(make_device_dataset(4), 2)
            .unwrap()
            .prefetch_factor(0)
            .device(Device::Cuda(0));

        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();

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
            .unwrap()
            .prefetch_factor(2)
            .device(Device::Cuda(1));

        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();

        assert_eq!(batches.len(), 3);
        for batch in &batches {
            for sample in batch {
                assert_eq!(sample.device, Device::Cuda(1));
            }
        }
        // Verify values are preserved.
        let values: Vec<i32> = batches
            .iter()
            .flat_map(|b| b.iter().map(|s| s.value))
            .collect();
        assert_eq!(values, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_no_device_no_transfer() {
        // Without .device(), samples should remain on their original device.
        let loader = DataLoader::new(make_device_dataset(4), 2)
            .unwrap()
            .prefetch_factor(0);

        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();

        for batch in &batches {
            for sample in batch {
                assert_eq!(sample.device, Device::Cpu);
            }
        }
    }

    #[test]
    fn test_device_transfer_empty_dataset() {
        let loader = DataLoader::new(make_device_dataset(0), 4)
            .unwrap()
            .device(Device::Cuda(0));

        let batches: Vec<_> = loader.iter(0).collect();
        assert!(batches.is_empty());
    }

    #[test]
    fn test_device_transfer_single_element() {
        let loader = DataLoader::new(make_device_dataset(1), 5)
            .unwrap()
            .device(Device::Cuda(0));

        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0][0].device, Device::Cuda(0));
        assert_eq!(batches[0][0].value, 0);
    }

    #[test]
    fn test_device_transfer_with_collate() {
        let loader = DataLoader::new(make_device_dataset(6), 3)
            .unwrap()
            .device(Device::Cuda(0))
            .with_collate(|batch| {
                Ok(DeviceSample {
                    value: batch.iter().map(|s| s.value).sum(),
                    device: batch[0].device,
                    pinned: batch[0].pinned,
                })
            });

        let collated: Vec<DeviceSample> = loader
            .iter_collated(0)
            .unwrap()
            .map(|r| r.unwrap())
            .collect();
        assert_eq!(collated.len(), 2);
        // Device transfer happens before collation.
        assert_eq!(collated[0].device, Device::Cuda(0));
        assert_eq!(collated[1].device, Device::Cuda(0));
        assert_eq!(collated[0].value, 1 + 2);
        assert_eq!(collated[1].value, 3 + 4 + 5);
    }

    // ── pin_memory ──────────────────────────────────────────────────
    //
    // pin_memory is a runtime flag passed to the transfer_fn closure built
    // by .device(). When set, the closure dispatches to
    // ToDevice::to_device_pinned instead of ToDevice::to_device. The
    // DeviceSample test impl tracks which method was called via the
    // `pinned` field, letting us verify the dispatch path is correct
    // without needing a real CUDA device. CL-378

    #[test]
    fn test_pin_memory_default_off_uses_to_device() {
        // pin_memory not set -> closure should call to_device (pinned=Some(false))
        let loader = DataLoader::new(make_device_dataset(2), 2)
            .unwrap()
            .prefetch_factor(0)
            .device(Device::Cuda(0));
        assert!(!loader.is_pin_memory());
        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();
        for sample in &batches[0] {
            assert_eq!(sample.pinned, Some(false));
        }
    }

    #[test]
    fn test_pin_memory_on_uses_to_device_pinned_sync() {
        // pin_memory(true) + device(Cuda) + sync iter -> to_device_pinned
        let loader = DataLoader::new(make_device_dataset(4), 2)
            .unwrap()
            .prefetch_factor(0)
            .device(Device::Cuda(0))
            .pin_memory(true);
        assert!(loader.is_pin_memory());
        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();
        for batch in &batches {
            for sample in batch {
                assert_eq!(sample.pinned, Some(true));
                assert_eq!(sample.device, Device::Cuda(0));
            }
        }
    }

    #[test]
    fn test_pin_memory_on_uses_to_device_pinned_prefetch() {
        // pin_memory(true) + device(Cuda) + prefetch iter -> to_device_pinned
        let loader = DataLoader::new(make_device_dataset(6), 2)
            .unwrap()
            .prefetch_factor(2)
            .device(Device::Cuda(0))
            .pin_memory(true);
        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 3);
        for batch in &batches {
            for sample in batch {
                assert_eq!(sample.pinned, Some(true));
            }
        }
    }

    #[test]
    fn test_pin_memory_set_after_device_takes_effect() {
        // The pin_memory flag is read at iter time (via the closure
        // parameter), not at device() construction time. So calling
        // pin_memory(true) AFTER device() must still flip the path.
        let loader = DataLoader::new(make_device_dataset(2), 2)
            .unwrap()
            .device(Device::Cuda(0))
            .pin_memory(true);
        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();
        for sample in &batches[0] {
            assert_eq!(sample.pinned, Some(true));
        }
    }

    #[test]
    fn test_pin_memory_set_before_device_takes_effect() {
        // Order independent: pin_memory before device should also work.
        let loader = DataLoader::new(make_device_dataset(2), 2)
            .unwrap()
            .pin_memory(true)
            .device(Device::Cuda(0));
        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();
        for sample in &batches[0] {
            assert_eq!(sample.pinned, Some(true));
        }
    }

    #[test]
    fn test_pin_memory_without_device_is_noop() {
        // pin_memory without device() should be a no-op (no transfer
        // happens at all). Samples retain their original device and
        // pinned=None.
        let loader = DataLoader::new(make_device_dataset(2), 2)
            .unwrap()
            .pin_memory(true);
        let batches: Vec<Vec<DeviceSample>> = loader.iter(0).map(|b| b.unwrap()).collect();
        for sample in &batches[0] {
            assert_eq!(sample.pinned, None);
            assert_eq!(sample.device, Device::Cpu);
        }
    }

    #[test]
    fn test_pin_memory_default_for_to_device_pinned_trait() {
        // A type that implements only ToDevice::to_device (using the
        // default to_device_pinned forwarder) should still work — the
        // default impl forwards to to_device, so the result device is
        // correct even though the pinned-ness flag isn't tracked.
        #[derive(Debug, Clone)]
        struct NoPinSample {
            value: i32,
            device: Device,
        }
        impl ToDevice for NoPinSample {
            fn to_device(&self, device: Device) -> FerrotorchResult<Self> {
                Ok(NoPinSample {
                    value: self.value,
                    device,
                })
            }
            // No to_device_pinned override -- uses the default forwarder.
        }
        struct NoPinDataset {
            data: Vec<NoPinSample>,
        }
        impl Dataset for NoPinDataset {
            type Sample = NoPinSample;
            fn len(&self) -> usize {
                self.data.len()
            }
            fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
                Ok(self.data[index].clone())
            }
        }

        let ds = Arc::new(NoPinDataset {
            data: vec![
                NoPinSample {
                    value: 1,
                    device: Device::Cpu,
                },
                NoPinSample {
                    value: 2,
                    device: Device::Cpu,
                },
            ],
        });
        let loader = DataLoader::new(ds, 2)
            .unwrap()
            .device(Device::Cuda(0))
            .pin_memory(true);
        let batches: Vec<Vec<NoPinSample>> = loader.iter(0).map(|b| b.unwrap()).collect();
        for sample in &batches[0] {
            // Default impl forwards to to_device, so device is set correctly.
            assert_eq!(sample.device, Device::Cuda(0));
        }
    }

    // ── prefetch_factor builder ─────────────────────────────────────

    #[test]
    fn test_prefetch_factor_builder() {
        let loader = DataLoader::new(make_dataset(10), 2)
            .unwrap()
            .prefetch_factor(5);
        assert_eq!(loader.prefetch_factor, 5);
    }

    #[test]
    fn test_default_prefetch_factor_is_2() {
        let loader = DataLoader::new(make_dataset(10), 2).unwrap();
        assert_eq!(loader.prefetch_factor, 2);
    }

    // ── iterator exhaustion ─────────────────────────────────────────

    #[test]
    fn test_prefetch_iterator_returns_none_after_exhaustion() {
        let loader = DataLoader::new(make_dataset(3), 2)
            .unwrap()
            .prefetch_factor(2);
        let mut it = loader.iter(0);
        assert!(it.next().is_some()); // [0, 1]
        assert!(it.next().is_some()); // [2]
        assert!(it.next().is_none());
        assert!(it.next().is_none()); // Repeated calls stay None.
        assert!(it.next().is_none());
    }

    #[test]
    fn test_sync_iterator_returns_none_after_exhaustion() {
        let loader = DataLoader::new(make_dataset(3), 2)
            .unwrap()
            .prefetch_factor(0);
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
        let loader = DataLoader::new(make_dataset(100), 3)
            .unwrap()
            .prefetch_factor(2);
        let it = loader.iter(0);
        drop(it);
    }

    #[test]
    fn test_drop_empty_prefetch_iter() {
        let loader = DataLoader::new(make_dataset(0), 4)
            .unwrap()
            .prefetch_factor(2);
        let it = loader.iter(0);
        drop(it);
    }

    // ── CL-377: cross-batch multi-worker pipeline ─────────────────

    #[test]
    fn test_multi_worker_default_is_intra_batch() {
        let loader = DataLoader::new(make_dataset(10), 2).unwrap();
        assert_eq!(loader.current_worker_mode(), WorkerMode::IntraBatch);
    }

    #[test]
    fn test_multi_worker_builder_sets_mode() {
        let loader = DataLoader::new(make_dataset(10), 2)
            .unwrap()
            .worker_mode(WorkerMode::CrossBatch);
        assert_eq!(loader.current_worker_mode(), WorkerMode::CrossBatch);
    }

    #[test]
    fn test_multi_worker_preserves_order() {
        // 4 workers, 20 samples, batch_size 2 → 10 batches. Each
        // worker runs independently so batches complete out of order
        // on the wire; the reorder buffer must yield them in order.
        let loader = DataLoader::new(make_dataset(20), 2)
            .unwrap()
            .num_workers(4)
            .worker_mode(WorkerMode::CrossBatch)
            .prefetch_factor(8);

        let batches: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        assert_eq!(batches.len(), 10);
        // Should be exactly [[0,1],[2,3],...[18,19]] in order.
        for (i, batch) in batches.iter().enumerate() {
            let base = (i * 2) as i32;
            assert_eq!(batch, &vec![base, base + 1], "batch {i}");
        }
    }

    #[test]
    fn test_multi_worker_with_drop_last() {
        let loader = DataLoader::new(make_dataset(7), 2)
            .unwrap()
            .num_workers(2)
            .worker_mode(WorkerMode::CrossBatch)
            .drop_last(true);
        let batches: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        // 7 samples, batch=2, drop_last → 3 batches: [0,1],[2,3],[4,5]
        assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4, 5]]);
    }

    #[test]
    fn test_multi_worker_with_keep_last() {
        let loader = DataLoader::new(make_dataset(7), 2)
            .unwrap()
            .num_workers(2)
            .worker_mode(WorkerMode::CrossBatch)
            .drop_last(false);
        let batches: Vec<Vec<i32>> = loader.iter(0).map(|b| b.unwrap()).collect();
        // 7 samples → 4 batches with last being partial [6]
        assert_eq!(batches, vec![vec![0, 1], vec![2, 3], vec![4, 5], vec![6]]);
    }

    #[test]
    fn test_multi_worker_with_shuffle_deterministic() {
        // Shuffled + multi-worker should still yield all samples
        // exactly once, and the same seed should produce the same
        // sequence across runs.
        let loader1 = DataLoader::new(make_dataset(12), 3)
            .unwrap()
            .shuffle(true)
            .seed(42)
            .num_workers(3)
            .worker_mode(WorkerMode::CrossBatch);
        let loader2 = DataLoader::new(make_dataset(12), 3)
            .unwrap()
            .shuffle(true)
            .seed(42)
            .num_workers(3)
            .worker_mode(WorkerMode::CrossBatch);

        let run1: Vec<i32> = loader1.iter(0).flat_map(|b| b.unwrap()).collect();
        let run2: Vec<i32> = loader2.iter(0).flat_map(|b| b.unwrap()).collect();

        assert_eq!(run1, run2);
        // Every sample present exactly once.
        let mut sorted = run1.clone();
        sorted.sort();
        assert_eq!(sorted, (0..12).collect::<Vec<_>>());
    }

    #[test]
    fn test_multi_worker_empty_dataset() {
        let loader = DataLoader::new(make_dataset(0), 4)
            .unwrap()
            .num_workers(2)
            .worker_mode(WorkerMode::CrossBatch);
        let batches: Vec<_> = loader.iter(0).collect();
        assert_eq!(batches.len(), 0);
    }

    #[test]
    fn test_multi_worker_drop_mid_iteration_shuts_down_cleanly() {
        // Spawn many workers on a big dataset, consume one batch,
        // then drop the iterator. Workers should exit cleanly without
        // deadlocking the test.
        let loader = DataLoader::new(make_dataset(1000), 4)
            .unwrap()
            .num_workers(4)
            .worker_mode(WorkerMode::CrossBatch)
            .prefetch_factor(16);
        let mut it = loader.iter(0);
        let _first = it.next().unwrap().unwrap();
        drop(it);
        // If workers had hung the test would time out.
    }

    #[test]
    fn test_multi_worker_zero_workers_falls_back_to_prefetch() {
        // CrossBatch with num_workers=0 should fall back to the
        // existing Prefetch / Sync code paths.
        let loader = DataLoader::new(make_dataset(6), 2)
            .unwrap()
            .num_workers(0)
            .worker_mode(WorkerMode::CrossBatch)
            .prefetch_factor(2);
        let it = loader.iter(0);
        assert!(matches!(it, BatchIter::Prefetch(_)));
    }

    #[test]
    fn test_multi_worker_iter_variant_returned_when_configured() {
        let loader = DataLoader::new(make_dataset(6), 2)
            .unwrap()
            .num_workers(2)
            .worker_mode(WorkerMode::CrossBatch);
        let it = loader.iter(0);
        assert!(matches!(it, BatchIter::MultiWorker(_)));
    }

    #[test]
    fn test_multi_worker_size_hint_decreases_monotonically() {
        let loader = DataLoader::new(make_dataset(8), 2)
            .unwrap()
            .num_workers(2)
            .worker_mode(WorkerMode::CrossBatch);
        let mut it = loader.iter(0);
        assert_eq!(it.size_hint(), (4, Some(4)));
        let _ = it.next().unwrap().unwrap();
        assert_eq!(it.size_hint(), (3, Some(3)));
        let _ = it.next().unwrap().unwrap();
        assert_eq!(it.size_hint(), (2, Some(2)));
        let _ = it.next().unwrap().unwrap();
        let _ = it.next().unwrap().unwrap();
        assert_eq!(it.size_hint(), (0, Some(0)));
        assert!(it.next().is_none());
    }
}
