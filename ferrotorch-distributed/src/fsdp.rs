//! Fully Sharded Data Parallel (FSDP) wrapper.
//!
//! [`FSDP`] wraps a [`Module`] and shards its parameters across ranks.
//! During forward, parameters are all-gathered to form the full tensors.
//! During gradient synchronization, full-parameter gradients are
//! reduce-scattered so each rank only stores its shard's gradient.
//!
//! This reduces per-rank memory from O(params) to O(params / world_size)
//! at the cost of additional communication during forward and backward.

use std::sync::Arc;

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Float, FerrotorchResult, Tensor};
use ferrotorch_nn::{Module, Parameter};

use crate::backend::Backend;
use crate::collective::{all_gather, reduce_scatter, ReduceOp};

/// Fully Sharded Data Parallel module wrapper.
///
/// Wraps an inner [`Module`] and shards each parameter across ranks so that
/// each rank only stores `1 / world_size` of the full parameter tensor.
///
/// # Forward pass
///
/// Before calling the inner module's `forward()`, FSDP all-gathers each
/// shard to reconstruct the full parameter tensor and installs it into the
/// module. The full-parameter tensors are stored in [`full_params`] so
/// that backward can accumulate gradients on them.
///
/// # Gradient synchronization
///
/// After `backward()`, call [`sync_gradients`] to:
/// 1. Read gradients from the full-parameter tensors stored during forward.
/// 2. Reduce-scatter the full gradients so each rank gets only its shard
///    portion of the gradient.
/// 3. Set each shard parameter's gradient from the reduce-scattered result.
///
/// # Example
///
/// ```ignore
/// let mut fsdp = FSDP::new(model, backend)?;
///
/// loop {
///     let output = fsdp.forward(&input)?;
///     let loss = criterion.forward(&output, &target)?;
///     ferrotorch_core::backward(&loss)?;
///     fsdp.sync_gradients()?;
///     optimizer.step()?;
///     optimizer.zero_grad()?;
/// }
/// ```
pub struct FSDP<M: Module<T>, T: Float> {
    module: M,
    backend: Arc<dyn Backend>,
    /// Original full-parameter shapes before sharding.
    original_shapes: Vec<Vec<usize>>,
    /// Full-param tensors from the last forward pass, kept alive so
    /// backward can accumulate gradients on them.
    full_params: Vec<Tensor<T>>,
    _marker: std::marker::PhantomData<T>,
}

impl<M: Module<T>, T: Float> FSDP<M, T> {
    /// Wrap a module for fully-sharded data-parallel training.
    ///
    /// Each parameter is split evenly across `world_size` ranks. This rank
    /// keeps only its shard (the `rank`-th chunk). The original parameter
    /// shapes are recorded for reconstruction during forward.
    ///
    /// # Panics
    ///
    /// Panics if any parameter's element count is not evenly divisible by
    /// `world_size`.
    pub fn new(mut module: M, backend: Arc<dyn Backend>) -> FerrotorchResult<Self> {
        let rank = backend.rank();
        let world_size = backend.world_size();
        let mut original_shapes = Vec::new();

        {
            let params = module.parameters_mut();
            for param in params {
                let tensor = param.tensor();
                let shape = tensor.shape().to_vec();
                let numel = tensor.numel();

                assert!(
                    numel % world_size == 0,
                    "FSDP: parameter with {} elements is not evenly divisible by world_size {}",
                    numel,
                    world_size,
                );

                original_shapes.push(shape);

                let data = tensor.data_vec()?;
                let chunk_size = numel / world_size;
                let start = rank * chunk_size;
                let end = start + chunk_size;
                let shard_data = data[start..end].to_vec();

                let shard_tensor = Tensor::from_storage(
                    TensorStorage::cpu(shard_data),
                    vec![chunk_size],
                    true,
                )?;
                // Shard params need requires_grad=true so the optimizer can
                // update them.
                *param = Parameter::new(shard_tensor);
            }
        }

        Ok(Self {
            module,
            backend,
            original_shapes,
            full_params: Vec::new(),
            _marker: std::marker::PhantomData,
        })
    }

    /// Immutable access to the inner module.
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Mutable access to the inner module.
    pub fn module_mut(&mut self) -> &mut M {
        &mut self.module
    }

    /// Consume the wrapper and return the inner module.
    pub fn into_inner(self) -> M {
        self.module
    }

    /// The backend used for communication.
    pub fn backend(&self) -> &Arc<dyn Backend> {
        &self.backend
    }

    /// Reconstruct full parameters from shards across all ranks and run
    /// the inner module's forward pass.
    ///
    /// The all-gathered full-parameter tensors are stored in `self.full_params`
    /// so their gradients can be read after backward.
    pub fn forward(&mut self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let world_size = self.backend.world_size();
        self.full_params.clear();

        {
            let params = self.module.parameters_mut();
            for (i, param) in params.into_iter().enumerate() {
                let shard = param.tensor().clone();
                let orig_shape = &self.original_shapes[i];

                // All-gather the shard to get the full parameter.
                let full = if world_size == 1 {
                    shard
                } else {
                    all_gather(&shard, self.backend.as_ref())?
                };

                // Reshape to the original parameter shape and enable grad.
                let full = Tensor::from_storage(
                    TensorStorage::cpu(full.data_vec()?),
                    orig_shape.clone(),
                    true,
                )?;

                self.full_params.push(full.clone());

                // Install the full parameter into the module for this forward pass.
                *param = Parameter::new(full);
            }
        }

        let output = self.module.forward(input)?;

        // After forward, restore shard parameters so the module holds only
        // shards at rest (saves memory).
        self.restore_shards()?;

        Ok(output)
    }

    /// Replace full parameters with their local shards to free memory.
    fn restore_shards(&mut self) -> FerrotorchResult<()> {
        let rank = self.backend.rank();
        let world_size = self.backend.world_size();

        let params = self.module.parameters_mut();
        for (i, param) in params.into_iter().enumerate() {
            let tensor = param.tensor();
            let data = tensor.data_vec()?;
            let numel = data.len();
            let chunk_size = numel / world_size;
            let start = rank * chunk_size;
            let end = start + chunk_size;
            let shard_data = data[start..end].to_vec();

            let shard_tensor = Tensor::from_storage(
                TensorStorage::cpu(shard_data),
                vec![chunk_size],
                true,
            )?;
            *param = Parameter::new(shard_tensor);

            // Preserve the original shape metadata.
            let _ = &self.original_shapes[i];
        }

        Ok(())
    }

    /// Reduce-scatter gradients from the full-parameter tensors stored
    /// during forward, then set each shard parameter's gradient.
    ///
    /// Call this after `backward()` and before `optimizer.step()`.
    ///
    /// # How it works
    ///
    /// 1. For each parameter, read the gradient from the full-param tensor
    ///    that was used during forward (stored in `self.full_params`).
    /// 2. Reduce-scatter the full gradient across ranks (mean reduction) so
    ///    each rank gets only its shard portion.
    /// 3. Set the shard parameter's `.grad()` to the reduce-scattered result.
    ///
    /// Using reduce-scatter (not allreduce) is correct for FSDP because each
    /// rank only needs its own shard of the gradient to update its shard of
    /// the parameter.
    pub fn sync_gradients(&mut self) -> FerrotorchResult<()> {
        let world_size = self.backend.world_size();
        let params = self.module.parameters_mut();

        if self.full_params.len() != params.len() {
            return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!(
                    "FSDP sync_gradients: expected {} full_params but have {}. \
                     Was forward() called before backward()?",
                    params.len(),
                    self.full_params.len(),
                ),
            }
            .into());
        }

        for (i, param) in params.into_iter().enumerate() {
            let full_param = &self.full_params[i];

            // Read the gradient from the full-parameter tensor. If no
            // gradient was computed (e.g., parameter was unused in forward),
            // use zeros so all ranks exchange buffers of the same size.
            let grad = full_param.grad()?;
            let full_grad = match grad {
                Some(g) => g,
                None => {
                    let numel = full_param.numel();
                    Tensor::from_storage(
                        TensorStorage::cpu(
                            vec![<T as num_traits::Zero>::zero(); numel],
                        ),
                        full_param.shape().to_vec(),
                        false,
                    )?
                }
            };

            // Flatten for reduce-scatter.
            let grad_data = full_grad.data_vec()?;
            let flat_grad = Tensor::from_storage(
                TensorStorage::cpu(grad_data),
                vec![full_grad.numel()],
                false,
            )?;

            // Reduce-scatter: each rank gets its shard of the averaged gradient.
            let shard_grad = if world_size == 1 {
                flat_grad
            } else {
                reduce_scatter(&flat_grad, self.backend.as_ref(), ReduceOp::Mean)?
            };

            // Set the shard parameter's gradient.
            // Interior mutability: set_grad works on &self via Mutex.
            param.tensor().set_grad(Some(shard_grad))?;
        }

        // Clear full_params to free memory now that gradients have been read.
        self.full_params.clear();

        Ok(())
    }

    /// Update shard parameters from a flat data slice.
    ///
    /// This is used by optimizers that produce a flat parameter buffer.
    /// The slice must have exactly the number of elements expected for
    /// this rank's shards.
    pub fn update_shards(&mut self, flat_data: &[T]) -> FerrotorchResult<()> {
        let params = self.module.parameters_mut();
        let total_shard_numel: usize = params.iter().map(|p| p.tensor().numel()).sum();

        assert!(
            flat_data.len() == total_shard_numel,
            "FSDP update_shards: expected {} elements but got {}",
            total_shard_numel,
            flat_data.len(),
        );

        let mut offset = 0;
        for param in params {
            let numel = param.tensor().numel();
            let shard_data = flat_data[offset..offset + numel].to_vec();
            let shard_tensor = Tensor::from_storage(
                TensorStorage::cpu(shard_data),
                param.tensor().shape().to_vec(),
                true,
            )?;
            *param = Parameter::new(shard_tensor);
            offset += numel;
        }

        Ok(())
    }

    /// Export the module's current sharded state as a state dict.
    ///
    /// Returns the named parameters of the inner module. Because FSDP stores
    /// only shard parameters (1/`world_size` of each full parameter), the
    /// returned tensors are the local shards — not the full parameters.
    ///
    /// This is designed to work with [`save_distributed`](crate::checkpoint::save_distributed)
    /// which saves per-rank shards to separate files.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let state = fsdp.state_dict()?;
    /// save_distributed(&state, &dir, rank, world_size, &metadata)?;
    /// ```
    pub fn state_dict(&self) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let named = self.module.named_parameters();
        let mut result = std::collections::HashMap::with_capacity(named.len());
        for (name, param) in named {
            result.insert(name, param.tensor().clone());
        }
        Ok(result)
    }

    /// Load sharded state into the module's parameters.
    ///
    /// The `state_dict` must contain tensors that match this rank's shard
    /// sizes (not full parameter sizes). Each key in the state dict is matched
    /// to a named parameter by name. Missing or unexpected keys are reported
    /// as errors.
    ///
    /// This is designed to work with [`load_distributed`](crate::checkpoint::load_distributed)
    /// which loads (and optionally reshards) per-rank shard files.
    ///
    /// # Example
    ///
    /// ```ignore
    /// let state = load_distributed(&dir, rank, world_size)?;
    /// fsdp.load_state_dict(&state)?;
    /// ```
    pub fn load_state_dict(
        &mut self,
        state_dict: &std::collections::HashMap<String, Tensor<T>>,
    ) -> FerrotorchResult<()> {
        // Collect the parameter names in order so we can match them with
        // parameters_mut().
        let param_names: Vec<String> = self
            .module
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect();

        // Validate: check that all expected keys are present.
        for name in &param_names {
            if !state_dict.contains_key(name) {
                return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                    message: format!("FSDP load_state_dict: missing key \"{name}\""),
                }
                .into());
            }
        }

        // Validate: check for unexpected keys.
        let known: std::collections::HashSet<&str> =
            param_names.iter().map(|s| s.as_str()).collect();
        for key in state_dict.keys() {
            if !known.contains(key.as_str()) {
                return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                    message: format!("FSDP load_state_dict: unexpected key \"{key}\""),
                }
                .into());
            }
        }

        // Replace each parameter's data with the corresponding state dict tensor.
        let params = self.module.parameters_mut();
        for (name, param) in param_names.iter().zip(params.into_iter()) {
            let tensor = &state_dict[name];

            // Validate shape match: the state dict tensor must have the same
            // number of elements as the current shard parameter.
            if param.tensor().numel() != tensor.numel() {
                return Err(ferrotorch_core::FerrotorchError::ShapeMismatch {
                    message: format!(
                        "FSDP load_state_dict: parameter \"{name}\" has {} elements \
                         but state_dict tensor has {}",
                        param.tensor().numel(),
                        tensor.numel()
                    ),
                }
                .into());
            }

            // Build a new parameter from the loaded data, preserving shard shape.
            let data = tensor.data_vec()?;
            let shard_tensor = Tensor::from_storage(
                TensorStorage::cpu(data),
                param.tensor().shape().to_vec(),
                true,
            )?;
            *param = Parameter::new(shard_tensor);
        }

        Ok(())
    }
}

// FSDP does NOT implement Module<T> because forward() requires &mut self
// (to store full_params). Callers must use fsdp.forward() directly.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SimulatedBackend;
    use ferrotorch_core::storage::TensorStorage;
    use ferrotorch_core::{FerrotorchResult, Tensor};
    use ferrotorch_nn::Parameter;
    use std::thread;

    /// Minimal module with one parameter for testing FSDP.
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
            // Simple forward: multiply input by weight sum (produces a scalar
            // that depends on all weight elements).
            let w_data = self.weight.tensor().data_vec()?;
            let w_sum: T = w_data.iter().copied().fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
            let i_data = input.data_vec()?;
            let out: Vec<T> = i_data.iter().map(|&x| x * w_sum).collect();
            Tensor::from_storage(
                TensorStorage::cpu(out),
                input.shape().to_vec(),
                false,
            )
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
    fn test_fsdp_sharding() {
        // 2 ranks, parameter [10, 20, 30, 40].
        // Rank 0 gets [10, 20], Rank 1 gets [30, 40].
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = TestModule::<f32>::new(&[10.0, 20.0, 30.0, 40.0]).unwrap();
                    let fsdp = FSDP::new(model, b).unwrap();

                    let shard = fsdp.module().weight.tensor().data_vec().unwrap();
                    (rank, shard)
                })
            })
            .collect();

        for h in handles {
            let (rank, shard) = h.join().unwrap();
            if rank == 0 {
                assert_eq!(shard, &[10.0, 20.0]);
            } else {
                assert_eq!(shard, &[30.0, 40.0]);
            }
        }
    }

    #[test]
    fn test_fsdp_shard_requires_grad() {
        // Shard parameters must have requires_grad=true.
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let fsdp = FSDP::new(model, b).unwrap();
                    fsdp.module().weight.tensor().requires_grad()
                })
            })
            .collect();

        for h in handles {
            assert!(h.join().unwrap(), "shard must have requires_grad=true");
        }
    }

    #[test]
    fn test_fsdp_forward_restores_shards() {
        // After forward(), parameters should be back to shard size.
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let mut fsdp = FSDP::new(model, b).unwrap();

                    let input = ferrotorch_core::from_slice(&[1.0f32], &[1]).unwrap();
                    let _output = fsdp.forward(&input).unwrap();

                    // After forward, shard should be size 2 (4 / 2 ranks).
                    let shard = fsdp.module().weight.tensor();
                    assert_eq!(shard.numel(), 2);
                    assert!(shard.requires_grad());
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_fsdp_forward_produces_correct_output() {
        // 2 ranks, param [1, 2, 3, 4], weight_sum = 10.
        // Input [2.0] -> output should be [20.0] on all ranks.
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let mut fsdp = FSDP::new(model, b).unwrap();

                    let input = ferrotorch_core::from_slice(&[2.0f32], &[1]).unwrap();
                    let output = fsdp.forward(&input).unwrap();
                    let data = output.data_vec().unwrap();
                    assert!(
                        (data[0] - 20.0).abs() < 1e-6,
                        "expected 20.0, got {}",
                        data[0]
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_fsdp_update_shards() {
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut fsdp = FSDP::new(model, b).unwrap();

        fsdp.update_shards(&[10.0, 20.0, 30.0, 40.0]).unwrap();
        let data = fsdp.module().weight.tensor().data_vec().unwrap();
        assert_eq!(data, &[10.0, 20.0, 30.0, 40.0]);
    }

    #[test]
    #[should_panic(expected = "expected 4 elements but got 2")]
    fn test_fsdp_update_shards_size_validation() {
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut fsdp = FSDP::new(model, b).unwrap();

        // Wrong size: should panic.
        fsdp.update_shards(&[10.0, 20.0]).unwrap();
    }

    #[test]
    fn test_fsdp_sync_gradients_single_rank() {
        // Single rank: sync_gradients should pass through the gradient
        // from the full param to the shard param.
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut fsdp = FSDP::new(model, b).unwrap();

        // Run forward to populate full_params.
        let input = ferrotorch_core::from_slice(&[1.0f32], &[1]).unwrap();
        let _output = fsdp.forward(&input).unwrap();

        // Manually set gradient on full_params (simulating backward).
        let grad = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1f32, 0.2, 0.3, 0.4]),
            vec![4],
            false,
        )
        .unwrap();
        fsdp.full_params[0].set_grad(Some(grad)).unwrap();

        fsdp.sync_gradients().unwrap();

        // Shard param should now have the full gradient (single rank = no scatter).
        let shard_grad = fsdp.module().weight.tensor().grad().unwrap().unwrap();
        let data = shard_grad.data_vec().unwrap();
        assert_eq!(data, &[0.1, 0.2, 0.3, 0.4]);
    }

    #[test]
    fn test_fsdp_sync_gradients_multi_rank() {
        // 2 ranks, param size 4 -> shard size 2.
        // Both ranks set identical gradients on full_params: [1, 2, 3, 4].
        // reduce_scatter(mean) on [1,2,3,4] -> rank 0 gets [1,2], rank 1 gets [3,4].
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let mut fsdp = FSDP::new(model, b).unwrap();

                    // Run forward.
                    let input = ferrotorch_core::from_slice(&[1.0f32], &[1]).unwrap();
                    let _output = fsdp.forward(&input).unwrap();

                    // Set gradient on full_params.
                    let grad = Tensor::from_storage(
                        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
                        vec![4],
                        false,
                    )
                    .unwrap();
                    fsdp.full_params[0].set_grad(Some(grad)).unwrap();

                    fsdp.sync_gradients().unwrap();

                    let shard_grad = fsdp.module().weight.tensor().grad().unwrap().unwrap();
                    let data = shard_grad.data_vec().unwrap();
                    (rank, data)
                })
            })
            .collect();

        for h in handles {
            let (rank, data) = h.join().unwrap();
            if rank == 0 {
                // Mean of [1,2] from both ranks = [1,2].
                assert_eq!(data.len(), 2);
                assert!((data[0] - 1.0).abs() < 1e-6, "rank 0: expected 1.0, got {}", data[0]);
                assert!((data[1] - 2.0).abs() < 1e-6, "rank 0: expected 2.0, got {}", data[1]);
            } else {
                // Mean of [3,4] from both ranks = [3,4].
                assert_eq!(data.len(), 2);
                assert!((data[0] - 3.0).abs() < 1e-6, "rank 1: expected 3.0, got {}", data[0]);
                assert!((data[1] - 4.0).abs() < 1e-6, "rank 1: expected 4.0, got {}", data[1]);
            }
        }
    }
}
