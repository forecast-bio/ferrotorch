//! Distributed Data Parallel (DDP) wrapper.
//!
//! [`DDP`] wraps a [`Module`] and synchronizes parameter gradients across
//! all ranks after each backward pass by allreducing them. This is the
//! standard data-parallel training strategy: each rank processes a
//! different mini-batch, then gradients are averaged so that all replicas
//! stay in sync.
//!
//! # Gradient bucketing
//!
//! Rather than calling allreduce once per parameter (which incurs high
//! latency overhead for models with many small parameters), gradients are
//! grouped into **buckets** of configurable size (default ~25 MB). Within
//! each bucket, gradients are flattened into a single contiguous buffer,
//! allreduced once, and then scattered back to the individual parameter
//! gradient tensors.

use std::sync::Arc;

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Float, FerrotorchResult, Tensor};
use ferrotorch_nn::{Module, Parameter};

use crate::backend::Backend;
use crate::collective::{allreduce, ReduceOp};

/// Default bucket size in bytes (~25 MB).
const DEFAULT_BUCKET_SIZE_BYTES: usize = 25 * 1024 * 1024;

/// A range of parameter indices that form one communication bucket.
#[derive(Debug, Clone)]
struct Bucket {
    /// Indices into the `parameters()` vec for this bucket.
    param_indices: Vec<usize>,
    /// Total number of elements across all parameters in this bucket.
    total_numel: usize,
}

/// Distributed Data Parallel module wrapper.
///
/// Wraps an inner [`Module`] and provides [`sync_gradients`] to allreduce
/// parameter gradients across all ranks. Typical usage:
///
/// ```ignore
/// let ddp = DDP::new(model, backend);
///
/// loop {
///     let output = ddp.module().forward(&input)?;
///     let loss = criterion.forward(&output, &target)?;
///     ferrotorch_core::backward(&loss)?;
///     ddp.sync_gradients()?;
///     optimizer.step()?;
///     optimizer.zero_grad()?;
/// }
/// ```
///
/// # Gradient bucketing
///
/// Parameters are grouped into buckets of approximately `bucket_size_bytes`
/// (default 25 MB). Use [`DDP::with_bucket_size`] to customize.
pub struct DDP<M: Module<T>, T: Float> {
    module: M,
    backend: Arc<dyn Backend>,
    /// Pre-computed bucket assignments. Computed once at construction and
    /// reused for every `sync_gradients` call.
    buckets: Vec<Bucket>,
    _marker: std::marker::PhantomData<T>,
}

impl<M: Module<T>, T: Float> DDP<M, T> {
    /// Wrap a module for distributed data-parallel training.
    ///
    /// Uses the default bucket size of ~25 MB. All ranks must hold the
    /// same initial model weights (e.g., loaded from a checkpoint or
    /// broadcast from rank 0).
    pub fn new(module: M, backend: Arc<dyn Backend>) -> Self {
        Self::with_bucket_size(module, backend, DEFAULT_BUCKET_SIZE_BYTES)
    }

    /// Wrap a module with a custom bucket size (in bytes).
    ///
    /// A bucket size of 0 puts every parameter into its own bucket
    /// (equivalent to the old per-parameter allreduce behavior).
    pub fn with_bucket_size(
        module: M,
        backend: Arc<dyn Backend>,
        bucket_size_bytes: usize,
    ) -> Self {
        let buckets = compute_buckets::<T, M>(&module, bucket_size_bytes);
        Self {
            module,
            backend,
            buckets,
            _marker: std::marker::PhantomData,
        }
    }

    /// Immutable access to the inner module (for forward pass, etc.).
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Mutable access to the inner module (for train/eval mode, etc.).
    pub fn module_mut(&mut self) -> &mut M {
        &mut self.module
    }

    /// Consume the DDP wrapper and return the inner module.
    pub fn into_inner(self) -> M {
        self.module
    }

    /// The backend used for communication.
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// The number of communication buckets used for gradient synchronization.
    ///
    /// This is the number of allreduce calls per `sync_gradients` invocation
    /// (one per non-empty bucket whose parameters have gradients).
    pub fn bucket_count(&self) -> usize {
        self.buckets.len()
    }

    /// Allreduce all parameter gradients across ranks (mean reduction).
    ///
    /// Call this after `backward()` and before `optimizer.step()`. Each
    /// parameter's `.grad()` is replaced with the mean gradient across
    /// all ranks, ensuring all replicas apply the same update.
    ///
    /// Gradients are bucketed: within each bucket, all gradients are
    /// flattened into one contiguous buffer, allreduced once, and then
    /// scattered back.
    pub fn sync_gradients(&self) -> FerrotorchResult<()> {
        let params = self.module.parameters();

        if params.is_empty() {
            return Ok(());
        }

        for bucket in &self.buckets {
            self.sync_bucket(&params, bucket)?;
        }

        Ok(())
    }

    /// Synchronize one bucket of gradients.
    fn sync_bucket(
        &self,
        params: &[&Parameter<T>],
        bucket: &Bucket,
    ) -> FerrotorchResult<()> {
        if bucket.param_indices.is_empty() || bucket.total_numel == 0 {
            return Ok(());
        }

        // Collect gradient data and metadata for parameters that have
        // gradients. Track which params had gradients so we can scatter
        // back to only those.
        let mut flat_buf: Vec<T> = Vec::with_capacity(bucket.total_numel);
        let mut grad_meta: Vec<GradMeta> = Vec::new();

        for &idx in &bucket.param_indices {
            let param = params[idx];
            if let Some(grad) = param.tensor().grad()? {
                let grad_data = grad.data()?;
                let offset = flat_buf.len();
                let numel = grad_data.len();
                let shape = grad.shape().to_vec();
                flat_buf.extend_from_slice(grad_data);
                grad_meta.push(GradMeta {
                    param_idx: idx,
                    offset,
                    numel,
                    shape,
                });
            }
        }

        // If no parameters in this bucket have gradients, nothing to do.
        if flat_buf.is_empty() {
            return Ok(());
        }

        // Create a flat tensor and allreduce it.
        let flat_tensor = Tensor::from_storage(
            TensorStorage::cpu(flat_buf),
            vec![grad_meta.iter().map(|m| m.numel).sum::<usize>()],
            false,
        )?;

        let synced = allreduce(&flat_tensor, self.backend.as_ref(), ReduceOp::Mean)?;
        let synced_data = synced.data()?;

        // Scatter the allreduced data back to individual parameter gradients.
        for meta in &grad_meta {
            let grad_slice = &synced_data[meta.offset..meta.offset + meta.numel];
            let grad_tensor = Tensor::from_storage(
                TensorStorage::cpu(grad_slice.to_vec()),
                meta.shape.clone(),
                false,
            )?;
            params[meta.param_idx].tensor().set_grad(Some(grad_tensor))?;
        }

        Ok(())
    }

    /// Broadcast model parameters from `root` rank to all other ranks.
    ///
    /// Ensures all ranks start with identical weights. Call once before
    /// the training loop begins.
    pub fn broadcast_parameters(&mut self, root: usize) -> FerrotorchResult<()> {
        let params_mut = self.module.parameters_mut();

        for param in params_mut {
            let tensor = param.tensor().clone();
            let synced = crate::collective::broadcast(&tensor, self.backend.as_ref(), root)?;
            *param = Parameter::new(synced);
        }

        Ok(())
    }
}

/// Metadata for one parameter's gradient within a flattened bucket buffer.
struct GradMeta {
    /// Index into the `parameters()` vec.
    param_idx: usize,
    /// Start offset in the flat buffer.
    offset: usize,
    /// Number of elements.
    numel: usize,
    /// Original shape of the gradient tensor.
    shape: Vec<usize>,
}

/// Compute bucket assignments based on parameter sizes.
///
/// Parameters are grouped greedily in order: each parameter is added to
/// the current bucket until the bucket exceeds `bucket_size_bytes`, then
/// a new bucket is started. This ensures parameters that are contiguous
/// in the model definition are in the same bucket (matching PyTorch's
/// behavior).
///
/// If `bucket_size_bytes` is 0, each parameter gets its own bucket.
fn compute_buckets<T: Float, M: Module<T>>(
    module: &M,
    bucket_size_bytes: usize,
) -> Vec<Bucket> {
    let params = module.parameters();

    if params.is_empty() {
        return vec![];
    }

    let t_size = std::mem::size_of::<T>();
    let mut buckets: Vec<Bucket> = Vec::new();
    let mut current_indices: Vec<usize> = Vec::new();
    let mut current_numel: usize = 0;
    let mut current_bytes: usize = 0;

    for (i, param) in params.iter().enumerate() {
        let numel = param.numel();
        let param_bytes = numel * t_size;

        // If adding this parameter would exceed the bucket size and the
        // current bucket is non-empty, finalize the current bucket first.
        // Special case: bucket_size_bytes == 0 means one param per bucket.
        if !current_indices.is_empty()
            && (bucket_size_bytes == 0 || current_bytes + param_bytes > bucket_size_bytes)
        {
            buckets.push(Bucket {
                param_indices: std::mem::take(&mut current_indices),
                total_numel: current_numel,
            });
            current_numel = 0;
            current_bytes = 0;
        }

        current_indices.push(i);
        current_numel += numel;
        current_bytes += param_bytes;
    }

    // Don't forget the last bucket.
    if !current_indices.is_empty() {
        buckets.push(Bucket {
            param_indices: current_indices,
            total_numel: current_numel,
        });
    }

    buckets
}

// Forward the Module trait through to the inner module so DDP can be
// used as a drop-in replacement.
impl<M: Module<T>, T: Float> Module<T> for DDP<M, T> {
    fn forward(
        &self,
        input: &ferrotorch_core::Tensor<T>,
    ) -> FerrotorchResult<ferrotorch_core::Tensor<T>> {
        self.module.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.module.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.module.parameters_mut()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.module.named_parameters()
    }

    fn train(&mut self) {
        self.module.train();
    }

    fn eval(&mut self) {
        self.module.eval();
    }

    fn is_training(&self) -> bool {
        self.module.is_training()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SimulatedBackend;
    use ferrotorch_core::storage::TensorStorage;
    use ferrotorch_core::{FerrotorchResult, Tensor};
    use ferrotorch_nn::Parameter;
    use std::thread;

    /// Minimal module with one parameter for testing DDP.
    struct TestModule<T: Float> {
        weight: Parameter<T>,
        training: bool,
    }

    impl<T: Float> TestModule<T> {
        fn new(data: &[T]) -> FerrotorchResult<Self> {
            Ok(Self {
                weight: Parameter::from_slice(data, &[data.len()])?,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for TestModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Parameter<T>> {
            vec![&self.weight]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            vec![&mut self.weight]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            vec![("weight".into(), &self.weight)]
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    /// Module with multiple parameters for testing bucketing.
    struct MultiParamModule<T: Float> {
        params: Vec<Parameter<T>>,
        training: bool,
    }

    impl<T: Float> MultiParamModule<T> {
        fn new(param_sizes: &[usize]) -> FerrotorchResult<Self> {
            let mut params = Vec::new();
            for &sz in param_sizes {
                params.push(Parameter::from_slice(
                    &vec![num_traits::One::one(); sz],
                    &[sz],
                )?);
            }
            Ok(Self {
                params,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for MultiParamModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Parameter<T>> {
            self.params.iter().collect()
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            self.params.iter_mut().collect()
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            self.params
                .iter()
                .enumerate()
                .map(|(i, p)| (format!("param_{i}"), p))
                .collect()
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    #[test]
    fn test_ddp_sync_gradients() {
        // 4 ranks. Each rank's parameter has a gradient equal to [rank, rank, rank].
        // After sync_gradients (mean), all should have [1.5, 1.5, 1.5].
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0]).unwrap();
                    let ddp = DDP::new(model, b);

                    // Simulate a backward pass by manually setting gradients.
                    let grad_val = rank as f32;
                    let grad = Tensor::from_storage(
                        TensorStorage::cpu(vec![grad_val, grad_val, grad_val]),
                        vec![3],
                        false,
                    )
                    .unwrap();
                    ddp.module().weight.tensor().set_grad(Some(grad)).unwrap();

                    // Sync gradients across all ranks.
                    ddp.sync_gradients().unwrap();

                    // All ranks should now have mean gradient = (0+1+2+3)/4 = 1.5
                    let synced_grad = ddp.module().weight.tensor().grad().unwrap().unwrap();
                    let data = synced_grad.data().unwrap();
                    for &v in data {
                        assert!(
                            (v - 1.5).abs() < 1e-5,
                            "rank {rank}: expected 1.5, got {v}"
                        );
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_ddp_broadcast_parameters() {
        // Rank 0 has weights [10, 20, 30]. Other ranks have [0, 0, 0].
        // After broadcast_parameters(0), all should have [10, 20, 30].
        let group = SimulatedBackend::create_group(3).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let data: Vec<f32> = if rank == 0 {
                        vec![10.0, 20.0, 30.0]
                    } else {
                        vec![0.0, 0.0, 0.0]
                    };
                    let model = TestModule::<f32>::new(&data).unwrap();
                    let mut ddp = DDP::new(model, b);

                    ddp.broadcast_parameters(0).unwrap();

                    let param_data = ddp.module().weight.tensor().data().unwrap();
                    assert!(
                        (param_data[0] - 10.0).abs() < 1e-5,
                        "rank {rank}: expected 10.0, got {}",
                        param_data[0]
                    );
                    assert!(
                        (param_data[1] - 20.0).abs() < 1e-5,
                        "rank {rank}: expected 20.0, got {}",
                        param_data[1]
                    );
                    assert!(
                        (param_data[2] - 30.0).abs() < 1e-5,
                        "rank {rank}: expected 30.0, got {}",
                        param_data[2]
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_ddp_delegates_module_trait() {
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let model = TestModule::<f32>::new(&[1.0, 2.0]).unwrap();
        let mut ddp = DDP::new(model, b);

        // Module trait methods should delegate.
        assert!(ddp.is_training());
        ddp.eval();
        assert!(!ddp.is_training());
        ddp.train();
        assert!(ddp.is_training());

        assert_eq!(ddp.parameters().len(), 1);
        assert_eq!(ddp.named_parameters()[0].0, "weight");
    }

    // -------------------------------------------------------------------
    // Gradient bucketing tests
    // -------------------------------------------------------------------

    #[test]
    fn test_bucketing_fewer_allreduce_calls_than_params() {
        // Create a module with 10 parameters of 100 f32 elements each
        // (400 bytes each, 4000 bytes total). With bucket_size = 2000 bytes,
        // we should get 2 buckets (5 params * 400 bytes = 2000 per bucket).
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        let model = MultiParamModule::<f32>::new(&[100; 10]).unwrap();
        let ddp = DDP::with_bucket_size(model, b, 2000);

        assert_eq!(ddp.module().params.len(), 10);
        // 10 params * 100 elements * 4 bytes = 4000 bytes total.
        // With bucket_size=2000, we expect 2 buckets (5 params per bucket,
        // each contributing 400 bytes, for 2000 bytes per bucket).
        assert_eq!(ddp.bucket_count(), 2);
        assert!(ddp.bucket_count() < ddp.module().params.len());
    }

    #[test]
    fn test_bucketing_zero_bucket_size_one_per_param() {
        // bucket_size_bytes = 0 means each parameter is its own bucket.
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        let model = MultiParamModule::<f32>::new(&[10, 20, 30]).unwrap();
        let ddp = DDP::with_bucket_size(model, b, 0);

        assert_eq!(ddp.bucket_count(), 3);
    }

    #[test]
    fn test_bucketing_large_bucket_one_bucket() {
        // Very large bucket size: all params in one bucket.
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        let model = MultiParamModule::<f32>::new(&[10, 20, 30, 40]).unwrap();
        let ddp = DDP::with_bucket_size(model, b, usize::MAX);

        assert_eq!(ddp.bucket_count(), 1);
    }

    #[test]
    fn test_bucketing_empty_model() {
        // No parameters: no buckets.
        struct EmptyModule;
        impl Module<f32> for EmptyModule {
            fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
                Ok(input.clone())
            }
            fn parameters(&self) -> Vec<&Parameter<f32>> {
                vec![]
            }
            fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
                vec![]
            }
            fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
                vec![]
            }
            fn train(&mut self) {}
            fn eval(&mut self) {}
            fn is_training(&self) -> bool {
                true
            }
        }

        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        let ddp = DDP::new(EmptyModule, b);
        assert_eq!(ddp.bucket_count(), 0);

        // sync_gradients should be a no-op, not an error.
        ddp.sync_gradients().unwrap();
    }

    #[test]
    fn test_bucketed_sync_correctness_multi_param() {
        // 3 ranks. Module with 4 parameters of varying sizes.
        // Bucket size = 40 bytes (10 f32 elements), so params are bucketed.
        // Each rank's gradient for param i is [rank * (i+1); numel].
        // After mean: [(0*(i+1) + 1*(i+1) + 2*(i+1))/3; numel] = [(i+1); numel]
        let world = 3;
        let param_sizes = vec![3, 5, 2, 4]; // 14 elements total
        let bucket_bytes = 20; // 5 f32 = 20 bytes per bucket

        let group = SimulatedBackend::create_group(world).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                let ps = param_sizes.clone();
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = MultiParamModule::<f32>::new(&ps).unwrap();
                    let ddp = DDP::with_bucket_size(model, b, bucket_bytes);

                    // Verify bucketing: 4 params, bucket ~5 elements.
                    // param 0: 3 elems (12 bytes) -> bucket 0
                    // param 1: 5 elems (20 bytes) -> 12+20=32 > 20, new bucket
                    // param 2: 2 elems (8 bytes) -> bucket 1 had 20 bytes, 20+8=28 > 20, new bucket
                    // param 3: 4 elems (16 bytes) -> bucket 2 had 8 bytes, 8+16=24 > 20, new bucket
                    // So 4 buckets. But wait, param 1 alone is 20 bytes which equals the bucket size.
                    // param 0: 12 bytes fits in first bucket.
                    // param 1: 12+20=32 > 20, start new bucket. param 1: 20 bytes.
                    // param 2: 20+8=28 > 20, start new bucket. param 2: 8 bytes.
                    // param 3: 8+16=24 > 20, start new bucket. param 3: 16 bytes.
                    // So 4 buckets. With such small bucket size, we don't save calls here.
                    // Let's just verify correctness.

                    // Set gradients.
                    for (i, param) in ddp.module().params.iter().enumerate() {
                        let numel = ps[i];
                        let grad_val = rank as f32 * (i + 1) as f32;
                        let grad = Tensor::from_storage(
                            TensorStorage::cpu(vec![grad_val; numel]),
                            vec![numel],
                            false,
                        )
                        .unwrap();
                        param.tensor().set_grad(Some(grad)).unwrap();
                    }

                    ddp.sync_gradients().unwrap();

                    // Verify mean gradients.
                    for (i, param) in ddp.module().params.iter().enumerate() {
                        let expected = (i + 1) as f32; // (0+1+2)/3 * (i+1) = (i+1)
                        let grad = param.tensor().grad().unwrap().unwrap();
                        let data = grad.data().unwrap();
                        assert_eq!(data.len(), ps[i], "param {i} shape mismatch");
                        for (j, &v) in data.iter().enumerate() {
                            assert!(
                                (v - expected).abs() < 1e-5,
                                "rank {rank}, param {i}, elem {j}: expected {expected}, got {v}"
                            );
                        }
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_bucketed_sync_some_params_without_grads() {
        // Edge case: only some parameters have gradients.
        let world = 2;
        let param_sizes = vec![3, 4, 5];

        let group = SimulatedBackend::create_group(world).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                let ps = param_sizes.clone();
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = MultiParamModule::<f32>::new(&ps).unwrap();
                    // Large bucket: everything in one bucket.
                    let ddp = DDP::with_bucket_size(model, b, usize::MAX);

                    // Only set gradient on param 0 and param 2 (skip param 1).
                    let val = (rank + 1) as f32;
                    let grad0 = Tensor::from_storage(
                        TensorStorage::cpu(vec![val; 3]),
                        vec![3],
                        false,
                    )
                    .unwrap();
                    ddp.module().params[0]
                        .tensor()
                        .set_grad(Some(grad0))
                        .unwrap();

                    let grad2 = Tensor::from_storage(
                        TensorStorage::cpu(vec![val * 10.0; 5]),
                        vec![5],
                        false,
                    )
                    .unwrap();
                    ddp.module().params[2]
                        .tensor()
                        .set_grad(Some(grad2))
                        .unwrap();

                    ddp.sync_gradients().unwrap();

                    // Param 0: mean of [1.0, 2.0] = 1.5
                    let g0 = ddp.module().params[0].tensor().grad().unwrap().unwrap();
                    for &v in g0.data().unwrap() {
                        assert!(
                            (v - 1.5).abs() < 1e-5,
                            "rank {rank} param 0: expected 1.5, got {v}"
                        );
                    }

                    // Param 1: no gradient, should remain None.
                    assert!(
                        ddp.module().params[1].tensor().grad().unwrap().is_none(),
                        "rank {rank}: param 1 should have no gradient"
                    );

                    // Param 2: mean of [10.0, 20.0] = 15.0
                    let g2 = ddp.module().params[2].tensor().grad().unwrap().unwrap();
                    for &v in g2.data().unwrap() {
                        assert!(
                            (v - 15.0).abs() < 1e-5,
                            "rank {rank} param 2: expected 15.0, got {v}"
                        );
                    }
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_bucketing_reduces_allreduce_calls() {
        // 10 parameters of 100 elements each = 4000 bytes of f32.
        // Default bucket = 25MB >> 4000 bytes, so all params fit in 1 bucket.
        // Compare: without bucketing, we'd need 10 allreduce calls.
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        let model = MultiParamModule::<f32>::new(&[100; 10]).unwrap();
        let num_params = model.params.len();
        let ddp = DDP::new(model, b); // default 25MB bucket

        // With default bucket size, all 10 small params should fit in 1 bucket.
        assert_eq!(ddp.bucket_count(), 1);
        assert!(
            ddp.bucket_count() < num_params,
            "bucket_count ({}) should be less than num_params ({})",
            ddp.bucket_count(),
            num_params
        );
    }

    #[test]
    fn test_bucketing_single_param_single_bucket() {
        // A single parameter should always result in exactly 1 bucket.
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0]).unwrap();
        let ddp = DDP::new(model, b);

        assert_eq!(ddp.bucket_count(), 1);
    }

    #[test]
    fn test_ddp_single_rank_sync_gradients() {
        // Single rank: sync_gradients should just return the same gradient.
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0]).unwrap();
        let ddp = DDP::new(model, b);

        let grad = Tensor::from_storage(
            TensorStorage::cpu(vec![10.0f32, 20.0, 30.0]),
            vec![3],
            false,
        )
        .unwrap();
        ddp.module().weight.tensor().set_grad(Some(grad)).unwrap();

        ddp.sync_gradients().unwrap();

        let synced = ddp.module().weight.tensor().grad().unwrap().unwrap();
        let data = synced.data().unwrap();
        assert_eq!(data, &[10.0, 20.0, 30.0]);
    }
}
