//! Distributed Data Parallel (DDP) wrapper.
//!
//! [`DDP`] wraps a [`Module`] and synchronizes parameter gradients across
//! all ranks after each backward pass by allreducing them. This is the
//! standard data-parallel training strategy: each rank processes a
//! different mini-batch, then gradients are averaged so that all replicas
//! stay in sync.

use std::sync::Arc;

use crate::backend::Backend;
use crate::collective::{ReduceOp, allreduce};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::{Module, Parameter};

/// Default bucket size for gradient bucketing (25 MB).
const DEFAULT_BUCKET_SIZE_BYTES: usize = 25 * 1024 * 1024;

/// Distributed Data Parallel module wrapper.
///
/// Wraps an inner [`Module`] and provides [`sync_gradients`] to allreduce
/// parameter gradients across all ranks. Parameters are grouped into
/// buckets (default 25 MB) for efficient communication.
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
pub struct DDP<M: Module<T>, T: Float> {
    module: M,
    backend: Arc<dyn Backend>,
    /// Bucket assignments: `buckets[i]` is a list of parameter indices in bucket i.
    buckets: Vec<Vec<usize>>,
    _marker: std::marker::PhantomData<T>,
}

impl<M: Module<T>, T: Float> DDP<M, T> {
    /// Wrap a module for distributed data-parallel training.
    ///
    /// Parameters are assigned to ~25 MB gradient buckets in reverse order
    /// (matching PyTorch's convention — backward computes gradients in
    /// reverse parameter order, so the first bucket fills first).
    pub fn new(module: M, backend: Arc<dyn Backend>) -> Self {
        Self::with_bucket_size(module, backend, DEFAULT_BUCKET_SIZE_BYTES)
    }

    /// Wrap a module with a custom bucket size (in bytes).
    pub fn with_bucket_size(
        module: M,
        backend: Arc<dyn Backend>,
        bucket_size_bytes: usize,
    ) -> Self {
        let params = module.parameters();
        let buckets = compute_buckets::<T>(&params, bucket_size_bytes);
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

    /// Allreduce parameter gradients across ranks using gradient bucketing.
    ///
    /// Parameters are grouped into ~25 MB buckets. Each bucket is
    /// allreduced independently as a single flat buffer. This enables
    /// future overlapped communication where the first bucket can start
    /// transferring while backward is still computing later gradients.
    ///
    /// Call this after `backward()` and before `optimizer.step()`.
    pub fn sync_gradients(&self) -> FerrotorchResult<()> {
        let params = self.module.parameters();
        for bucket in &self.buckets {
            sync_one_bucket::<T>(bucket, &params, self.backend.as_ref())?;
        }
        Ok(())
    }

    /// Allreduce parameter gradients with bucket-level parallelism.
    ///
    /// Like [`sync_gradients`], but processes all buckets concurrently using
    /// `std::thread::scope`. Each bucket's allreduce runs in its own thread,
    /// overlapping communication across buckets. All threads complete before
    /// this method returns.
    ///
    /// This provides communication/computation overlap when backward and
    /// sync run on different threads, and communication overlap across
    /// buckets even in the synchronous case.
    pub fn overlapped_sync_gradients(&self) -> FerrotorchResult<()> {
        let params = self.module.parameters();

        // Collect errors from threads.
        let errors: std::sync::Mutex<Vec<ferrotorch_core::error::FerrotorchError>> =
            std::sync::Mutex::new(Vec::new());

        std::thread::scope(|s| {
            for bucket in &self.buckets {
                let params_ref = &params;
                let backend_ref = self.backend.as_ref();
                let errors_ref = &errors;

                s.spawn(move || {
                    let result = sync_one_bucket::<T>(bucket, params_ref, backend_ref);
                    if let Err(e) = result {
                        errors_ref.lock().unwrap().push(e);
                    }
                });
            }
        });

        let errs = errors.into_inner().unwrap();
        if let Some(e) = errs.into_iter().next() {
            return Err(e);
        }

        Ok(())
    }

    /// Broadcast model parameters from `root` rank to all other ranks.
    ///
    /// Ensures all ranks start with identical weights. Call once before
    /// the training loop begins.
    ///
    /// # Warning
    ///
    /// This replaces the `Parameter` objects in the module. Any optimizer
    /// that holds references to the old parameters must be re-initialized
    /// after calling this method, otherwise optimizer state (momentum,
    /// adaptive learning rates, etc.) will refer to stale parameters.
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

/// Assign parameters to gradient buckets.
///
/// Parameters are added in reverse order (matching PyTorch — backward
/// computes gradients in reverse parameter order, so the last parameters
/// fill the first bucket). Each bucket holds at most `bucket_size_bytes`
/// of gradient data.
fn compute_buckets<T: Float>(
    params: &[&Parameter<T>],
    bucket_size_bytes: usize,
) -> Vec<Vec<usize>> {
    let elem_size = std::mem::size_of::<T>();
    let mut buckets: Vec<Vec<usize>> = Vec::new();
    let mut current_bucket: Vec<usize> = Vec::new();
    let mut current_bytes: usize = 0;

    // Reverse order: last parameter first.
    for i in (0..params.len()).rev() {
        let param_bytes = params[i].tensor().numel() * elem_size;

        if !current_bucket.is_empty() && current_bytes + param_bytes > bucket_size_bytes {
            buckets.push(current_bucket);
            current_bucket = Vec::new();
            current_bytes = 0;
        }

        current_bucket.push(i);
        current_bytes += param_bytes;
    }

    if !current_bucket.is_empty() {
        buckets.push(current_bucket);
    }

    buckets
}

/// Allreduce a single bucket's gradients.
///
/// Builds a flat buffer from the bucket's parameter gradients, allreduces,
/// and scatters the result back. Used by both `sync_gradients` (serial)
/// and `overlapped_sync_gradients` (parallel).
fn sync_one_bucket<T: Float>(
    bucket: &[usize],
    params: &[&Parameter<T>],
    backend: &dyn Backend,
) -> FerrotorchResult<()> {
    let mut flat_data: Vec<T> = Vec::new();
    let mut param_numels: Vec<usize> = Vec::new();

    for &pi in bucket {
        let numel = params[pi].tensor().numel();
        param_numels.push(numel);

        let grad = params[pi].tensor().grad()?;
        match grad {
            Some(g) => flat_data.extend(g.data_vec()?),
            None => {
                flat_data.extend(std::iter::repeat_n(<T as num_traits::Zero>::zero(), numel));
            }
        }
    }

    if flat_data.is_empty() {
        return Ok(());
    }

    let flat_tensor = Tensor::from_storage(
        TensorStorage::cpu(flat_data),
        vec![param_numels.iter().sum()],
        false,
    )?;
    let synced = allreduce(&flat_tensor, backend, ReduceOp::Mean)?;
    let synced_data = synced.data()?;

    let mut offset = 0;
    for (&pi, &numel) in bucket.iter().zip(param_numels.iter()) {
        let grad_slice = &synced_data[offset..offset + numel];
        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_slice.to_vec()),
            params[pi].tensor().shape().to_vec(),
            false,
        )?;
        params[pi].tensor().set_grad(Some(grad_tensor))?;
        offset += numel;
    }

    Ok(())
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
                        assert!((v - 1.5).abs() < 1e-5, "rank {rank}: expected 1.5, got {v}");
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
}
