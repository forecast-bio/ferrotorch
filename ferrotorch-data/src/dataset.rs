use ferrotorch_core::{FerrotorchError, FerrotorchResult};

/// A map-style dataset: provides random access to samples by index.
///
/// This is the primary dataset trait, matching PyTorch's `Dataset`.
/// Implementations must be `Send + Sync` for use with multi-worker
/// data loading.
pub trait Dataset: Send + Sync {
    /// The type of a single sample returned by `get()`.
    type Sample: Send;

    /// Total number of samples in the dataset.
    fn len(&self) -> usize;

    /// Whether the dataset is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Retrieve the sample at the given index.
    ///
    /// Returns an error if the index is out of bounds.
    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample>;
}

/// An iterable-style dataset: provides a streaming iterator over samples.
///
/// Used for large datasets that don't fit in memory or are generated
/// on-the-fly (e.g., streaming from disk, web datasets).
pub trait IterableDataset: Send + Sync {
    /// The type of a single sample.
    type Sample: Send;

    /// Return an iterator over samples.
    ///
    /// When used with multiple workers, each worker receives a
    /// `WorkerInfo` to partition the stream.
    fn iter(
        &self,
        worker_info: Option<&WorkerInfo>,
    ) -> Box<dyn Iterator<Item = FerrotorchResult<Self::Sample>> + Send + '_>;
}

/// Information about the current data loading worker.
///
/// Passed to `IterableDataset::iter()` so the dataset can partition
/// its stream across workers.
#[derive(Debug, Clone)]
pub struct WorkerInfo {
    /// This worker's index (0-based).
    pub worker_id: usize,
    /// Total number of workers.
    pub num_workers: usize,
}

/// A simple in-memory dataset backed by a `Vec<S>`.
///
/// Useful for testing and small datasets.
#[derive(Debug, Clone)]
pub struct VecDataset<S: Send + Sync + Clone> {
    data: Vec<S>,
}

impl<S: Send + Sync + Clone> VecDataset<S> {
    pub fn new(data: Vec<S>) -> Self {
        Self { data }
    }
}

impl<S: Send + Sync + Clone + 'static> Dataset for VecDataset<S> {
    type Sample = S;

    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        self.data
            .get(index)
            .cloned()
            .ok_or(FerrotorchError::IndexOutOfBounds {
                index,
                axis: 0,
                size: self.data.len(),
            })
    }
}

/// A dataset that applies a transform to another dataset's samples.
pub struct MappedDataset<D: Dataset, F> {
    inner: D,
    transform: F,
}

impl<D, F, O> MappedDataset<D, F>
where
    D: Dataset,
    F: Fn(D::Sample) -> FerrotorchResult<O> + Send + Sync,
    O: Send,
{
    pub fn new(dataset: D, transform: F) -> Self {
        Self {
            inner: dataset,
            transform,
        }
    }
}

impl<D, F, O> Dataset for MappedDataset<D, F>
where
    D: Dataset,
    F: Fn(D::Sample) -> FerrotorchResult<O> + Send + Sync,
    O: Send + 'static,
{
    type Sample = O;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        let sample = self.inner.get(index)?;
        (self.transform)(sample)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vec_dataset() {
        let ds = VecDataset::new(vec![1, 2, 3, 4, 5]);
        assert_eq!(ds.len(), 5);
        assert!(!ds.is_empty());
        assert_eq!(ds.get(0).unwrap(), 1);
        assert_eq!(ds.get(4).unwrap(), 5);
        assert!(ds.get(5).is_err());
    }

    #[test]
    fn test_vec_dataset_empty() {
        let ds = VecDataset::<i32>::new(vec![]);
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn test_mapped_dataset() {
        let ds = VecDataset::new(vec![1, 2, 3]);
        let mapped = MappedDataset::new(ds, |x| Ok(x * 10));
        assert_eq!(mapped.len(), 3);
        assert_eq!(mapped.get(0).unwrap(), 10);
        assert_eq!(mapped.get(2).unwrap(), 30);
    }

    #[test]
    fn test_dataset_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VecDataset<i32>>();
    }
}
