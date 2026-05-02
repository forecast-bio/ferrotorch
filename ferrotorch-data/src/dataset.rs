use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, select};

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

// ---------------------------------------------------------------------------
// TensorDataset
// ---------------------------------------------------------------------------

/// A dataset of tensors, all with the same size in dimension 0.
///
/// Matches PyTorch's `torch.utils.data.TensorDataset`. Indexing returns
/// a `Vec<Tensor<T>>` where each element is the i-th slice of each stored
/// tensor along dimension 0.
///
/// # Example
///
/// ```ignore
/// let x = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
/// let y = Tensor::from_vec(vec![0.0, 1.0], &[2]);
/// let ds = TensorDataset::new(vec![x, y]).unwrap();
/// assert_eq!(ds.len(), 2);
/// let sample = ds.get(0).unwrap(); // [x[0], y[0]]
/// ```
pub struct TensorDataset<T: Float> {
    tensors: Vec<Tensor<T>>,
    /// Size of dimension 0 (cached for fast access).
    len: usize,
}

impl<T: Float> TensorDataset<T> {
    /// Create a new `TensorDataset` from a list of tensors.
    ///
    /// All tensors must have the same size in dimension 0. At least one
    /// tensor must be provided, and all tensors must have at least one
    /// dimension.
    pub fn new(tensors: Vec<Tensor<T>>) -> FerrotorchResult<Self> {
        if tensors.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "TensorDataset: need at least one tensor".into(),
            });
        }

        let first_shape = tensors[0].shape();
        if first_shape.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "TensorDataset: tensors must have at least one dimension".into(),
            });
        }

        let len = first_shape[0];

        for (i, t) in tensors.iter().enumerate().skip(1) {
            let s = t.shape();
            if s.is_empty() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "TensorDataset: tensor at index {i} is a scalar (needs >= 1 dim)"
                    ),
                });
            }
            if s[0] != len {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "TensorDataset: tensor at index {i} has dim-0 size {} but expected {len}",
                        s[0]
                    ),
                });
            }
        }

        Ok(Self { tensors, len })
    }

    /// Number of samples (size of dimension 0).
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Retrieve the sample at the given index.
    ///
    /// Returns one tensor per stored tensor, each selected at `index`
    /// along dimension 0 (removing that dimension, like PyTorch indexing).
    pub fn get(&self, index: usize) -> FerrotorchResult<Vec<Tensor<T>>> {
        if index >= self.len {
            return Err(FerrotorchError::IndexOutOfBounds {
                index,
                axis: 0,
                size: self.len,
            });
        }
        self.tensors.iter().map(|t| select(t, 0, index)).collect()
    }
}

impl<T: Float + 'static> Dataset for TensorDataset<T> {
    type Sample = Vec<Tensor<T>>;

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        TensorDataset::get(self, index)
    }
}

// ---------------------------------------------------------------------------
// ConcatDataset
// ---------------------------------------------------------------------------

/// A dataset that concatenates multiple datasets end-to-end.
///
/// Matches PyTorch's `torch.utils.data.ConcatDataset`. The total length
/// is the sum of all sub-dataset lengths, and indexing transparently maps
/// to the correct sub-dataset.
pub struct ConcatDataset<D: Dataset> {
    datasets: Vec<D>,
    /// Cumulative lengths: `cumulative[i]` = sum of lengths of datasets 0..=i.
    cumulative: Vec<usize>,
}

impl<D: Dataset> ConcatDataset<D> {
    /// Create a `ConcatDataset` from a list of datasets.
    ///
    /// # Errors
    ///
    /// Returns an error if the list is empty.
    pub fn new(datasets: Vec<D>) -> FerrotorchResult<Self> {
        if datasets.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "ConcatDataset: need at least one dataset".into(),
            });
        }

        let mut cumulative = Vec::with_capacity(datasets.len());
        let mut total = 0usize;
        for ds in &datasets {
            total += ds.len();
            cumulative.push(total);
        }

        Ok(Self {
            datasets,
            cumulative,
        })
    }

    /// Map a global index to (dataset_index, local_index).
    fn locate(&self, index: usize) -> (usize, usize) {
        // Binary search for the first cumulative length > index.
        let ds_idx = self.cumulative.partition_point(|&cum| cum <= index);
        let local = if ds_idx == 0 {
            index
        } else {
            index - self.cumulative[ds_idx - 1]
        };
        (ds_idx, local)
    }
}

impl<D: Dataset> Dataset for ConcatDataset<D>
where
    D::Sample: 'static,
{
    type Sample = D::Sample;

    fn len(&self) -> usize {
        *self.cumulative.last().unwrap_or(&0)
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        let total = self.len();
        if index >= total {
            return Err(FerrotorchError::IndexOutOfBounds {
                index,
                axis: 0,
                size: total,
            });
        }
        let (ds_idx, local) = self.locate(index);
        self.datasets[ds_idx].get(local)
    }
}

// ---------------------------------------------------------------------------
// ChainDataset
// ---------------------------------------------------------------------------

/// A dataset that chains multiple iterable datasets end-to-end.
///
/// Matches PyTorch's `torch.utils.data.ChainDataset`. It lazily iterates
/// through each sub-dataset in order, yielding all samples from the first
/// dataset, then all from the second, and so on.
///
/// `ChainDataset` implements `IterableDataset` and can also be used as a
/// `Dataset` (map-style) if you need random access — it delegates to the
/// same `ConcatDataset`-style index mapping internally.
pub struct ChainDataset<D: Dataset> {
    inner: ConcatDataset<D>,
}

impl<D: Dataset> ChainDataset<D> {
    /// Create a `ChainDataset` from a list of datasets.
    ///
    /// # Errors
    ///
    /// Returns an error if the list is empty.
    pub fn new(datasets: Vec<D>) -> FerrotorchResult<Self> {
        Ok(Self {
            inner: ConcatDataset::new(datasets)?,
        })
    }
}

impl<D: Dataset> Dataset for ChainDataset<D>
where
    D::Sample: 'static,
{
    type Sample = D::Sample;

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        self.inner.get(index)
    }
}

impl<D: Dataset> IterableDataset for ChainDataset<D>
where
    D::Sample: 'static,
{
    type Sample = D::Sample;

    fn iter(
        &self,
        worker_info: Option<&WorkerInfo>,
    ) -> Box<dyn Iterator<Item = FerrotorchResult<Self::Sample>> + Send + '_> {
        let total = self.inner.len();
        let (start, end) = match worker_info {
            Some(info) => {
                let per_worker = total / info.num_workers;
                let remainder = total % info.num_workers;
                // Workers 0..remainder get one extra sample.
                let s = info.worker_id * per_worker + info.worker_id.min(remainder);
                let extra = if info.worker_id < remainder { 1 } else { 0 };
                (s, s + per_worker + extra)
            }
            None => (0, total),
        };

        Box::new((start..end).map(move |i| self.inner.get(i)))
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

    // ── TensorDataset ─────────────────────────────────────────────

    use ferrotorch_core::TensorStorage;

    fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        let storage = TensorStorage::cpu(data.to_vec());
        Tensor::from_storage(storage, shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_tensor_dataset_basic() {
        let x = t32(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[3, 2]);
        let y = t32(&[10.0, 20.0, 30.0], &[3]);
        let ds = TensorDataset::new(vec![x, y]).unwrap();

        assert_eq!(ds.len(), 3);
        assert!(!ds.is_empty());

        let sample = ds.get(0).unwrap();
        assert_eq!(sample.len(), 2);
        assert_eq!(sample[0].shape(), &[2]);
        assert_eq!(sample[0].data_vec().unwrap(), vec![1.0, 2.0]);
        // Empty-shape literal needs an explicit element type when the
        // serde_json transitive (via polars in `arrow`-feature builds)
        // adds a competing `PartialEq<Value> for usize` impl.
        assert_eq!(sample[1].shape(), &[] as &[usize]);
        assert_eq!(sample[1].data_vec().unwrap(), vec![10.0]);

        let sample2 = ds.get(2).unwrap();
        assert_eq!(sample2[0].data_vec().unwrap(), vec![5.0, 6.0]);
        assert_eq!(sample2[1].data_vec().unwrap(), vec![30.0]);
    }

    #[test]
    fn test_tensor_dataset_single_tensor() {
        let x = t32(&[1.0, 2.0, 3.0], &[3, 1]);
        let ds = TensorDataset::new(vec![x]).unwrap();
        assert_eq!(ds.len(), 3);
        let sample = ds.get(1).unwrap();
        assert_eq!(sample.len(), 1);
        assert_eq!(sample[0].data_vec().unwrap(), vec![2.0]);
    }

    #[test]
    fn test_tensor_dataset_oob() {
        let x = t32(&[1.0, 2.0], &[2, 1]);
        let ds = TensorDataset::new(vec![x]).unwrap();
        assert!(ds.get(2).is_err());
    }

    #[test]
    fn test_tensor_dataset_dim_mismatch() {
        let x = t32(&[1.0, 2.0, 3.0], &[3]);
        let y = t32(&[1.0, 2.0], &[2]);
        assert!(TensorDataset::new(vec![x, y]).is_err());
    }

    #[test]
    fn test_tensor_dataset_empty_tensors() {
        assert!(TensorDataset::<f32>::new(vec![]).is_err());
    }

    #[test]
    fn test_tensor_dataset_scalar_rejected() {
        let s = t32(&[1.0], &[]);
        assert!(TensorDataset::new(vec![s]).is_err());
    }

    #[test]
    fn test_tensor_dataset_as_trait() {
        let x = t32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let y = t32(&[0.0, 1.0], &[2]);
        let ds = TensorDataset::new(vec![x, y]).unwrap();

        // Use through the Dataset trait.
        let d: &dyn Dataset<Sample = Vec<Tensor<f32>>> = &ds;
        assert_eq!(d.len(), 2);
        let sample = d.get(0).unwrap();
        assert_eq!(sample.len(), 2);
    }

    // ── ConcatDataset ─────────────────────────────────────────────

    #[test]
    fn test_concat_dataset_basic() {
        let a = VecDataset::new(vec![10, 20, 30]);
        let b = VecDataset::new(vec![40, 50]);
        let ds = ConcatDataset::new(vec![a, b]).unwrap();

        assert_eq!(ds.len(), 5);
        assert_eq!(ds.get(0).unwrap(), 10);
        assert_eq!(ds.get(2).unwrap(), 30);
        assert_eq!(ds.get(3).unwrap(), 40);
        assert_eq!(ds.get(4).unwrap(), 50);
    }

    #[test]
    fn test_concat_dataset_oob() {
        let a = VecDataset::new(vec![1, 2]);
        let b = VecDataset::new(vec![3]);
        let ds = ConcatDataset::new(vec![a, b]).unwrap();
        assert!(ds.get(3).is_err());
    }

    #[test]
    fn test_concat_dataset_single() {
        let a = VecDataset::new(vec![1, 2, 3]);
        let ds = ConcatDataset::new(vec![a]).unwrap();
        assert_eq!(ds.len(), 3);
        assert_eq!(ds.get(1).unwrap(), 2);
    }

    #[test]
    fn test_concat_dataset_empty_err() {
        let result = ConcatDataset::<VecDataset<i32>>::new(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_concat_dataset_boundary() {
        // Three datasets of sizes 2, 3, 1 -> total 6.
        let a = VecDataset::new(vec![0, 1]);
        let b = VecDataset::new(vec![2, 3, 4]);
        let c = VecDataset::new(vec![5]);
        let ds = ConcatDataset::new(vec![a, b, c]).unwrap();

        assert_eq!(ds.len(), 6);
        for i in 0..6 {
            assert_eq!(ds.get(i).unwrap(), i as i32);
        }
    }

    // ── ChainDataset ──────────────────────────────────────────────

    #[test]
    fn test_chain_dataset_map_style() {
        let a = VecDataset::new(vec![10, 20]);
        let b = VecDataset::new(vec![30, 40, 50]);
        let ds = ChainDataset::new(vec![a, b]).unwrap();

        assert_eq!(ds.len(), 5);
        assert_eq!(Dataset::get(&ds, 0).unwrap(), 10);
        assert_eq!(Dataset::get(&ds, 4).unwrap(), 50);
    }

    #[test]
    fn test_chain_dataset_iterable() {
        let a = VecDataset::new(vec![1, 2]);
        let b = VecDataset::new(vec![3, 4]);
        let ds = ChainDataset::new(vec![a, b]).unwrap();

        let items: Vec<i32> = ds.iter(None).map(|r| r.unwrap()).collect();
        assert_eq!(items, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_chain_dataset_iterable_with_workers() {
        let a = VecDataset::new(vec![0, 1, 2, 3, 4, 5]);
        let ds = ChainDataset::new(vec![a]).unwrap();

        let w0 = WorkerInfo {
            worker_id: 0,
            num_workers: 2,
        };
        let w1 = WorkerInfo {
            worker_id: 1,
            num_workers: 2,
        };

        let items0: Vec<i32> = ds.iter(Some(&w0)).map(|r| r.unwrap()).collect();
        let items1: Vec<i32> = ds.iter(Some(&w1)).map(|r| r.unwrap()).collect();

        assert_eq!(items0.len(), 3);
        assert_eq!(items1.len(), 3);
        // Together they cover all samples.
        let mut all = items0;
        all.extend(items1);
        all.sort();
        assert_eq!(all, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_chain_dataset_empty_err() {
        let result = ChainDataset::<VecDataset<i32>>::new(vec![]);
        assert!(result.is_err());
    }
}
