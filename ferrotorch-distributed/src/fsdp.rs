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
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::{Module, Parameter};

use crate::async_collective::{PendingCollective, async_all_gather};
use crate::backend::{Backend, SubBackend};
use crate::collective::{ReduceOp, all_gather, allreduce, reduce_scatter};

/// Sharding strategy for [`FSDP`]. Mirrors PyTorch's `ShardingStrategy`.
///
/// - [`FullShard`](ShardingStrategy::FullShard) — shard parameters,
///   gradients, and optimizer state. Minimum memory, maximum
///   communication. Equivalent to ZeRO-3.
/// - [`ShardGradOp`](ShardingStrategy::ShardGradOp) — keep parameters
///   replicated on every rank (no parameter all-gather in forward),
///   but shard gradients and optimizer state. Equivalent to ZeRO-2.
///   After `optimizer.step()`, call
///   [`FSDP::broadcast_updated_params`] to re-sync the updated param
///   shards back to every rank.
/// - [`NoShard`](ShardingStrategy::NoShard) — equivalent to DDP: no
///   sharding, allreduce the full gradient. Provided so the FSDP
///   wrapper can be used as a drop-in replacement for DDP during
///   debugging or for single-node experiments.
/// - [`HybridShard`](ShardingStrategy::HybridShard) — shard within a
///   node (intra-node FullShard) and replicate across nodes
///   (inter-node DDP). Uses two [`SubBackend`]s derived from the
///   global backend to run intra-node `all_gather` / `reduce_scatter`
///   and inter-node `allreduce` independently. CL-327.
///
/// CL-372.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ShardingStrategy {
    /// Shard parameters + gradients + optimizer state (ZeRO-3 /
    /// full FSDP). This is the default and matches the behavior
    /// that existed before CL-372.
    #[default]
    FullShard,
    /// Shard gradients + optimizer state only, keep params
    /// replicated (ZeRO-2).
    ShardGradOp,
    /// No sharding (ZeRO-0 / DDP equivalent). gradients are allreduced.
    NoShard,
    /// Hybrid sharding: shard within a node (intra-node FullShard) +
    /// replicate across nodes (inter-node DDP). The `intra_node_size`
    /// is the number of ranks per node (i.e., local GPUs). The total
    /// `world_size` must be a multiple of `intra_node_size`.
    /// CL-327.
    HybridShard { intra_node_size: usize },
}

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
    /// Active sharding strategy. Drives the behavior of `new`,
    /// `forward`, `sync_gradients`, and `broadcast_updated_params`.
    strategy: ShardingStrategy,
    /// Original full-parameter shapes before sharding.
    original_shapes: Vec<Vec<usize>>,
    /// Full-param tensors from the last forward pass, kept alive so
    /// backward can accumulate gradients on them.
    full_params: Vec<Tensor<T>>,
    /// Pending async all-gather handles produced by
    /// [`FSDP::prefetch_forward_params`]. One entry per parameter, in
    /// the same order as [`Module::parameters_mut`]. `None` means no
    /// prefetch is in flight and `forward()` will use the synchronous
    /// all-gather path. CL-373.
    pending_prefetch: Option<Vec<PendingCollective<T>>>,
    /// Intra-node subgroup used by [`ShardingStrategy::HybridShard`].
    /// `None` for all other strategies. CL-327.
    intra_node_group: Option<Arc<SubBackend>>,
    /// Inter-node subgroup used by [`ShardingStrategy::HybridShard`].
    /// `None` for all other strategies. CL-327.
    inter_node_group: Option<Arc<SubBackend>>,
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
    pub fn new(module: M, backend: Arc<dyn Backend>) -> FerrotorchResult<Self> {
        Self::new_with_strategy(module, backend, ShardingStrategy::FullShard)
    }

    /// Wrap a module for data-parallel training with a specific
    /// [`ShardingStrategy`].
    ///
    /// - `FullShard` — shard parameters, gradients, and optimizer state
    ///   (the classic FSDP / ZeRO-3 behavior; identical to [`new`]).
    /// - `ShardGradOp` — keep parameters replicated on every rank and
    ///   only shard gradients + optimizer state (ZeRO-2). After
    ///   calling the optimizer step on the shard gradients, the caller
    ///   must call [`broadcast_updated_params`] to re-sync the updated
    ///   parameter shards back to every rank. CL-372.
    /// - `NoShard` — no sharding (ZeRO-0 / DDP equivalent). Gradients
    ///   are allreduced across ranks in `sync_gradients` and all ranks
    ///   update the full parameters locally.
    pub fn new_with_strategy(
        mut module: M,
        backend: Arc<dyn Backend>,
        strategy: ShardingStrategy,
    ) -> FerrotorchResult<Self> {
        let rank = backend.rank();
        let world_size = backend.world_size();
        let mut original_shapes = Vec::new();

        // For HybridShard, build the intra- and inter-node subgroups
        // upfront so we can use the intra_group's world_size to shard
        // parameters below.
        let (intra_node_group, inter_node_group) = match strategy {
            ShardingStrategy::HybridShard { intra_node_size } => {
                if intra_node_size == 0 || world_size % intra_node_size != 0 {
                    return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                        message: format!(
                            "HybridShard: world_size={world_size} must be a positive multiple of intra_node_size={intra_node_size}"
                        ),
                    });
                }

                // This node's row index in the (inter_size x intra_size)
                // rank grid.
                let node_idx = rank / intra_node_size;
                let local_idx = rank % intra_node_size;

                // Intra-node members: contiguous block of `intra_node_size`
                // ranks starting at `node_idx * intra_node_size`.
                let intra_members: Vec<usize> = (node_idx * intra_node_size
                    ..node_idx * intra_node_size + intra_node_size)
                    .collect();
                let intra = Arc::new(SubBackend::new(Arc::clone(&backend), intra_members)?);

                // Inter-node members: ranks with the same local_idx across
                // every node (i.e., ranks stride-selected by
                // `intra_node_size`).
                let inter_members: Vec<usize> = (0..world_size / intra_node_size)
                    .map(|n| n * intra_node_size + local_idx)
                    .collect();
                let inter = Arc::new(SubBackend::new(Arc::clone(&backend), inter_members)?);

                (Some(intra), Some(inter))
            }
            _ => (None, None),
        };

        {
            let params = module.parameters_mut();
            for param in params {
                let tensor = param.tensor();
                let shape = tensor.shape().to_vec();
                original_shapes.push(shape);

                match strategy {
                    ShardingStrategy::FullShard => {
                        let numel = tensor.numel();
                        assert!(
                            numel % world_size == 0,
                            "FSDP: parameter with {numel} elements is not evenly divisible by world_size {world_size}"
                        );
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
                        *param = Parameter::new(shard_tensor);
                    }
                    ShardingStrategy::HybridShard { .. } => {
                        // Shard within the node only. Each rank keeps
                        // `1 / intra_node_size` of each parameter.
                        let intra = intra_node_group
                            .as_ref()
                            .expect("HybridShard: intra_node_group built above");
                        let intra_size = intra.world_size();
                        let intra_rank = intra.rank();
                        let numel = tensor.numel();
                        assert!(
                            numel % intra_size == 0,
                            "FSDP HybridShard: parameter with {numel} elements is not evenly divisible by intra_node_size {intra_size}"
                        );
                        let data = tensor.data_vec()?;
                        let chunk_size = numel / intra_size;
                        let start = intra_rank * chunk_size;
                        let end = start + chunk_size;
                        let shard_data = data[start..end].to_vec();
                        let shard_tensor = Tensor::from_storage(
                            TensorStorage::cpu(shard_data),
                            vec![chunk_size],
                            true,
                        )?;
                        *param = Parameter::new(shard_tensor);
                    }
                    ShardingStrategy::ShardGradOp | ShardingStrategy::NoShard => {
                        // Keep the full parameter on this rank; only
                        // gradients (and optimizer state, as an
                        // external concern) are sharded for ShardGradOp.
                        // NoShard is a plain DDP-style replication.
                        //
                        // For ShardGradOp, each rank still needs to know
                        // which slice of the flat parameter is "its"
                        // shard for the optimizer step. That's derived
                        // at grad-sync time from world_size + rank.
                    }
                }
            }
        }

        Ok(Self {
            module,
            backend,
            strategy,
            original_shapes,
            full_params: Vec::new(),
            pending_prefetch: None,
            intra_node_group,
            inter_node_group,
            _marker: std::marker::PhantomData,
        })
    }

    /// Return the active sharding strategy.
    pub fn strategy(&self) -> ShardingStrategy {
        self.strategy
    }

    /// Kick off asynchronous all-gathers for every parameter so the
    /// next [`forward`](Self::forward) call consumes the pre-gathered
    /// tensors instead of blocking on a fresh all-gather.
    ///
    /// This is FSDP's equivalent of PyTorch's backward prefetch:
    /// communication for layer N+1 (or the next forward pass) overlaps
    /// with compute for layer N. The caller should insert local
    /// compute (e.g., the previous layer's backward, or input
    /// preprocessing) between `prefetch_forward_params` and `forward`
    /// to realize the overlap.
    ///
    /// Only valid for [`ShardingStrategy::FullShard`] — the other
    /// strategies keep parameters replicated on every rank so there's
    /// nothing to all-gather. Calling this on a non-`FullShard` FSDP
    /// returns an `InvalidArgument` error.
    ///
    /// # Invariant
    ///
    /// Exactly one `prefetch_forward_params` → `forward` pair should be
    /// in flight at any time on a given FSDP instance. Calling
    /// `prefetch_forward_params` twice in a row (without an intervening
    /// `forward`) returns an `InvalidArgument` error.
    ///
    /// CL-373.
    pub fn prefetch_forward_params(&mut self) -> FerrotorchResult<()> {
        if self.strategy != ShardingStrategy::FullShard {
            return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                message: format!(
                    "FSDP::prefetch_forward_params: prefetch is only valid for FullShard, got {:?}",
                    self.strategy
                ),
            });
        }
        if self.pending_prefetch.is_some() {
            return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                message: "FSDP::prefetch_forward_params called twice without intervening forward()"
                    .into(),
            });
        }

        let mut handles = Vec::new();
        {
            let params = self.module.parameters();
            for param in params {
                let shard = param.tensor().clone();
                let h = async_all_gather(shard, Arc::clone(&self.backend));
                handles.push(h);
            }
        }
        self.pending_prefetch = Some(handles);
        Ok(())
    }

    /// True if a prefetch is currently pending. Primarily useful for
    /// tests and diagnostics.
    pub fn has_pending_prefetch(&self) -> bool {
        self.pending_prefetch.is_some()
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
    ///
    /// For `ShardGradOp` and `NoShard` strategies, parameters are already
    /// full on every rank, so no all-gather happens and `full_params` is
    /// populated from the current parameter tensors directly.
    ///
    /// If [`prefetch_forward_params`](Self::prefetch_forward_params) was
    /// called earlier, the pending async all-gather handles are consumed
    /// here instead of running the synchronous all_gather — this is how
    /// FSDP hides all-gather latency behind whatever local compute
    /// happened between `prefetch_forward_params` and `forward`.
    pub fn forward(&mut self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let world_size = self.backend.world_size();
        self.full_params.clear();

        // Grab any pending prefetch handles. They are consumed here so
        // a subsequent forward() without a matching prefetch reverts to
        // synchronous all_gather.
        let mut pending = self.pending_prefetch.take();

        match self.strategy {
            ShardingStrategy::FullShard => {
                if let Some(ref p) = pending {
                    if p.len() != self.module.parameters().len() {
                        return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                            message: format!(
                                "FSDP prefetch: have {} pending handles but module has {} parameters",
                                p.len(),
                                self.module.parameters().len(),
                            ),
                        });
                    }
                }
                // Pop handles from the front by reversing so we can pop from back.
                if let Some(ref mut p) = pending {
                    p.reverse();
                }

                let params = self.module.parameters_mut();
                for (i, param) in params.into_iter().enumerate() {
                    let orig_shape = &self.original_shapes[i];

                    // Get the gathered full tensor either from a pending
                    // async handle or via a fresh synchronous all_gather.
                    let full = if let Some(ref mut handles) = pending {
                        // Consume the last handle (original order via reverse).
                        let handle = handles.pop().ok_or_else(|| {
                            ferrotorch_core::FerrotorchError::InvalidArgument {
                                message: "FSDP prefetch: exhausted pending handles".into(),
                            }
                        })?;
                        handle.wait()?
                    } else {
                        let shard = param.tensor().clone();
                        if world_size == 1 {
                            shard
                        } else {
                            all_gather(&shard, self.backend.as_ref())?
                        }
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
            ShardingStrategy::HybridShard { .. } => {
                // HybridShard: all-gather the shards within the node only.
                // Prefetch is not supported for this strategy yet — the
                // intra-node subgroup needs its own async path. CL-327.
                if pending.is_some() {
                    return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                        message:
                            "FSDP prefetch_forward_params is not yet implemented for HybridShard"
                                .into(),
                    });
                }

                let intra = self
                    .intra_node_group
                    .as_ref()
                    .expect("HybridShard: intra_node_group set in new_with_strategy");
                let intra_ref: &dyn Backend = &**intra;
                let intra_size = intra.world_size();

                let params = self.module.parameters_mut();
                for (i, param) in params.into_iter().enumerate() {
                    let orig_shape = &self.original_shapes[i];

                    let shard = param.tensor().clone();
                    let full = if intra_size == 1 {
                        shard
                    } else {
                        all_gather(&shard, intra_ref)?
                    };

                    let full = Tensor::from_storage(
                        TensorStorage::cpu(full.data_vec()?),
                        orig_shape.clone(),
                        true,
                    )?;

                    self.full_params.push(full.clone());
                    *param = Parameter::new(full);
                }
            }
            ShardingStrategy::ShardGradOp | ShardingStrategy::NoShard => {
                // Parameters are already full on every rank. Prefetch
                // is meaningless for these strategies — surface any
                // accidental misuse.
                if pending.is_some() {
                    return Err(ferrotorch_core::FerrotorchError::InvalidArgument {
                        message: "FSDP prefetch_forward_params is only valid for FullShard; \
                                  use ShardingStrategy::FullShard or don't prefetch"
                            .into(),
                    });
                }
                // We still need to wrap them in requires_grad=true leaves
                // so the autograd graph can flow through the forward
                // pass, and stash them in full_params so sync_gradients
                // can read the backward-accumulated gradients.
                let params = self.module.parameters_mut();
                for param in params.into_iter() {
                    let t = param.tensor().clone();
                    let data = t.data_vec()?;
                    let shape = t.shape().to_vec();
                    let full = Tensor::from_storage(TensorStorage::cpu(data), shape, true)?;
                    self.full_params.push(full.clone());
                    *param = Parameter::new(full);
                }
            }
        }

        let output = self.module.forward(input)?;

        // After forward, restore shard parameters (FullShard /
        // HybridShard) or leave the full params in place
        // (ShardGradOp / NoShard).
        match self.strategy {
            ShardingStrategy::FullShard => self.restore_shards()?,
            ShardingStrategy::HybridShard { .. } => self.restore_hybrid_shards()?,
            ShardingStrategy::ShardGradOp | ShardingStrategy::NoShard => {
                // Nothing to do: params are already full on every rank.
            }
        }

        Ok(output)
    }

    /// Replace the current full parameter tensors with this rank's
    /// intra-node shards. Used by [`ShardingStrategy::HybridShard`] after
    /// forward completes. CL-327.
    fn restore_hybrid_shards(&mut self) -> FerrotorchResult<()> {
        let intra = self
            .intra_node_group
            .as_ref()
            .expect("HybridShard: intra_node_group set")
            .clone();
        let intra_size = intra.world_size();
        let intra_rank = intra.rank();

        let params = self.module.parameters_mut();
        for (i, param) in params.into_iter().enumerate() {
            let tensor = param.tensor();
            let data = tensor.data_vec()?;
            let numel = data.len();
            let chunk_size = numel / intra_size;
            let start = intra_rank * chunk_size;
            let end = start + chunk_size;
            let shard_data = data[start..end].to_vec();

            let shard_tensor =
                Tensor::from_storage(TensorStorage::cpu(shard_data), vec![chunk_size], true)?;
            *param = Parameter::new(shard_tensor);

            // Preserve the original shape metadata.
            let _ = &self.original_shapes[i];
        }

        Ok(())
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

            let shard_tensor =
                Tensor::from_storage(TensorStorage::cpu(shard_data), vec![chunk_size], true)?;
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
        let rank = self.backend.rank();
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
            });
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
                        TensorStorage::cpu(vec![<T as num_traits::Zero>::zero(); numel]),
                        full_param.shape().to_vec(),
                        false,
                    )?
                }
            };

            // Flatten for reduction ops.
            let grad_data = full_grad.data_vec()?;
            let flat_grad = Tensor::from_storage(
                TensorStorage::cpu(grad_data),
                vec![full_grad.numel()],
                false,
            )?;

            match self.strategy {
                ShardingStrategy::FullShard => {
                    // Reduce-scatter: each rank gets its shard of the
                    // averaged gradient. Parameter is already the
                    // shard here (installed by restore_shards after
                    // forward), so set_grad lines up.
                    let shard_grad = if world_size == 1 {
                        flat_grad
                    } else {
                        reduce_scatter(&flat_grad, self.backend.as_ref(), ReduceOp::Mean)?
                    };
                    param.tensor().set_grad(Some(shard_grad))?;
                }
                ShardingStrategy::ShardGradOp => {
                    // ZeRO-2: reduce-scatter the flat gradient into a
                    // per-rank slice, then set the parameter's .grad()
                    // to a tensor shaped like the full parameter but
                    // with only this rank's shard positions populated
                    // and the rest zeroed. The optimizer's update at
                    // the non-shard positions becomes a no-op, so
                    // each rank effectively updates only its shard
                    // slice. After optimizer.step, the caller must
                    // invoke broadcast_updated_params to re-sync.
                    let numel = flat_grad.numel();
                    assert!(
                        numel % world_size == 0,
                        "FSDP ShardGradOp: parameter with {numel} elements is not evenly \
                         divisible by world_size {world_size}"
                    );
                    let shard_grad_flat = if world_size == 1 {
                        flat_grad
                    } else {
                        reduce_scatter(&flat_grad, self.backend.as_ref(), ReduceOp::Mean)?
                    };
                    let chunk_size = numel / world_size;
                    let shard_data = shard_grad_flat.data_vec()?;
                    // Pad with zeros at non-shard positions so the
                    // gradient tensor matches the full parameter shape.
                    let mut padded = vec![<T as num_traits::Zero>::zero(); numel];
                    let start = rank * chunk_size;
                    padded[start..start + chunk_size].copy_from_slice(&shard_data);
                    let padded_grad = Tensor::from_storage(
                        TensorStorage::cpu(padded),
                        full_param.shape().to_vec(),
                        false,
                    )?;
                    param.tensor().set_grad(Some(padded_grad))?;
                }
                ShardingStrategy::HybridShard { .. } => {
                    // HybridShard: reduce-scatter within the node to
                    // get this rank's intra-node shard of the mean
                    // gradient, then allreduce the shard across nodes
                    // so every replica of this intra-rank has the
                    // same gradient. Parameter is already the
                    // intra-node shard (installed by
                    // restore_hybrid_shards after forward).
                    let intra = self
                        .intra_node_group
                        .as_ref()
                        .expect("HybridShard: intra_node_group set");
                    let inter = self
                        .inter_node_group
                        .as_ref()
                        .expect("HybridShard: inter_node_group set");
                    let intra_ref: &dyn Backend = &**intra;
                    let inter_ref: &dyn Backend = &**inter;
                    let intra_size = intra.world_size();
                    let inter_size = inter.world_size();

                    // Step 1: intra-node reduce_scatter (mean).
                    let intra_shard = if intra_size == 1 {
                        flat_grad
                    } else {
                        reduce_scatter(&flat_grad, intra_ref, ReduceOp::Mean)?
                    };

                    // Step 2: inter-node allreduce (mean across
                    // replicas).
                    let replicated = if inter_size == 1 {
                        intra_shard
                    } else {
                        allreduce(&intra_shard, inter_ref, ReduceOp::Mean)?
                    };

                    param.tensor().set_grad(Some(replicated))?;
                }
                ShardingStrategy::NoShard => {
                    // Plain DDP: allreduce the full gradient so every
                    // rank has the same averaged gradient, then set
                    // it on the full parameter.
                    let reduced = if world_size == 1 {
                        flat_grad
                    } else {
                        allreduce(&flat_grad, self.backend.as_ref(), ReduceOp::Mean)?
                    };
                    let reduced_full = Tensor::from_storage(
                        TensorStorage::cpu(reduced.data_vec()?),
                        full_param.shape().to_vec(),
                        false,
                    )?;
                    param.tensor().set_grad(Some(reduced_full))?;
                }
            }
        }

        // Clear full_params to free memory now that gradients have been read.
        self.full_params.clear();

        Ok(())
    }

    /// For `ShardGradOp`: after `optimizer.step()`, each rank has
    /// applied the update to its own shard of the full parameter
    /// (because `sync_gradients` zeroed the non-shard positions of the
    /// gradient). This method re-syncs the parameter tensors so every
    /// rank has the fully updated parameter, by summing contributions
    /// via an allreduce: each rank contributes its updated shard, zero
    /// elsewhere; the sum across ranks is the full updated parameter.
    ///
    /// More precisely, this method reconstructs the full parameter as
    /// an allgather of per-rank shards. It is a no-op for `FullShard`
    /// and `NoShard` strategies (they already have consistent
    /// parameters after step).
    ///
    /// Call this AFTER `optimizer.step()` and BEFORE the next
    /// `forward()`. CL-372.
    pub fn broadcast_updated_params(&mut self) -> FerrotorchResult<()> {
        if self.strategy != ShardingStrategy::ShardGradOp {
            // Nothing to do for FullShard (already shard-local) or
            // NoShard (all ranks already have the same params).
            return Ok(());
        }

        let rank = self.backend.rank();
        let world_size = self.backend.world_size();
        if world_size == 1 {
            return Ok(());
        }

        let params = self.module.parameters_mut();
        for param in params {
            // Extract this rank's shard from the updated full parameter.
            let full = param.tensor();
            let full_data = full.data_vec()?;
            let numel = full_data.len();
            assert!(
                numel % world_size == 0,
                "FSDP broadcast_updated_params: parameter with {numel} elements is not evenly \
                 divisible by world_size {world_size}"
            );
            let chunk_size = numel / world_size;
            let start = rank * chunk_size;
            let end = start + chunk_size;
            let shard = full_data[start..end].to_vec();
            let shard_tensor =
                Tensor::from_storage(TensorStorage::cpu(shard), vec![chunk_size], false)?;

            // All-gather across ranks to get the full updated parameter.
            let gathered = all_gather(&shard_tensor, self.backend.as_ref())?;
            let full_shape = full.shape().to_vec();
            let new_full =
                Tensor::from_storage(TensorStorage::cpu(gathered.data_vec()?), full_shape, true)?;
            *param = Parameter::new(new_full);
        }
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
            let w_sum: T = w_data
                .iter()
                .copied()
                .fold(<T as num_traits::Zero>::zero(), |a, b| a + b);
            let i_data = input.data_vec()?;
            let out: Vec<T> = i_data.iter().map(|&x| x * w_sum).collect();
            Tensor::from_storage(TensorStorage::cpu(out), input.shape().to_vec(), false)
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
    fn test_fsdp_shard_grad_op_keeps_full_params() {
        // ShardGradOp (ZeRO-2): params stay replicated on every rank.
        // Verify each rank has the full parameter after `new_with_strategy`.
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let fsdp =
                        FSDP::new_with_strategy(model, b, ShardingStrategy::ShardGradOp).unwrap();
                    assert_eq!(fsdp.strategy(), ShardingStrategy::ShardGradOp);
                    fsdp.module().weight.tensor().data_vec().unwrap()
                })
            })
            .collect();

        for h in handles {
            let data = h.join().unwrap();
            assert_eq!(data, &[1.0, 2.0, 3.0, 4.0]);
        }
    }

    #[test]
    fn test_fsdp_shard_grad_op_sync_gradients_multi_rank() {
        // ZeRO-2: two ranks, param [1,2,3,4], both ranks produce full grad
        // [1,2,3,4]. reduce_scatter(mean) gives rank 0 the slice [1,2] and
        // rank 1 the slice [3,4]. Each rank's .grad() is then padded so the
        // other-rank positions are zero, giving
        //   rank 0 grad: [1,2,0,0]
        //   rank 1 grad: [0,0,3,4]
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let mut fsdp =
                        FSDP::new_with_strategy(model, b, ShardingStrategy::ShardGradOp).unwrap();

                    let input = ferrotorch_core::from_slice(&[1.0f32], &[1]).unwrap();
                    let _output = fsdp.forward(&input).unwrap();

                    let grad = Tensor::from_storage(
                        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
                        vec![4],
                        false,
                    )
                    .unwrap();
                    fsdp.full_params[0].set_grad(Some(grad)).unwrap();

                    fsdp.sync_gradients().unwrap();

                    // Param remains full (not restored to shard) under
                    // ShardGradOp, and .grad() is a tensor of the full
                    // shape with only this rank's shard populated.
                    let w = fsdp.module().weight.tensor();
                    assert_eq!(w.numel(), 4, "ShardGradOp keeps params full");
                    let g = w.grad().unwrap().unwrap();
                    let gd = g.data_vec().unwrap();
                    assert_eq!(gd.len(), 4, "grad should be full-shape");
                    (rank, gd)
                })
            })
            .collect();

        for h in handles {
            let (rank, gd) = h.join().unwrap();
            if rank == 0 {
                assert!((gd[0] - 1.0).abs() < 1e-6, "rank 0 [0]: {}", gd[0]);
                assert!((gd[1] - 2.0).abs() < 1e-6, "rank 0 [1]: {}", gd[1]);
                assert_eq!(gd[2], 0.0, "rank 0 [2] should be zero");
                assert_eq!(gd[3], 0.0, "rank 0 [3] should be zero");
            } else {
                assert_eq!(gd[0], 0.0, "rank 1 [0] should be zero");
                assert_eq!(gd[1], 0.0, "rank 1 [1] should be zero");
                assert!((gd[2] - 3.0).abs() < 1e-6, "rank 1 [2]: {}", gd[2]);
                assert!((gd[3] - 4.0).abs() < 1e-6, "rank 1 [3]: {}", gd[3]);
            }
        }
    }

    #[test]
    fn test_fsdp_shard_grad_op_broadcast_updated_params() {
        // Full ZeRO-2 loop: each rank simulates an optimizer.step() that
        // applies a per-shard update (adds rank*10 to its slice), then
        // calls broadcast_updated_params. After that, every rank should
        // see the fully updated parameter.
        //
        // Rank 0 slice is [0,1]; rank 1 slice is [2,3].
        // Starting param: [1,2,3,4].
        // After per-rank update:
        //   rank 0 local param: [1+10, 2+10, 3, 4]     = [11, 12, 3, 4]
        //   rank 1 local param: [1, 2, 3+20, 4+20]     = [1, 2, 23, 24]
        // After broadcast_updated_params (allgather of rank-local shards):
        //   both ranks: [11, 12, 23, 24]
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let mut fsdp =
                        FSDP::new_with_strategy(model, b, ShardingStrategy::ShardGradOp).unwrap();

                    // Simulate per-rank optimizer step: each rank overwrites
                    // its shard slice, leaving the other slice untouched.
                    let mut local = fsdp.module().weight.tensor().data_vec().unwrap();
                    if rank == 0 {
                        local[0] += 10.0;
                        local[1] += 10.0;
                    } else {
                        local[2] += 20.0;
                        local[3] += 20.0;
                    }
                    let new_param =
                        Tensor::from_storage(TensorStorage::cpu(local), vec![4], true).unwrap();
                    *fsdp.module.parameters_mut()[0] = Parameter::new(new_param);

                    // Re-sync.
                    fsdp.broadcast_updated_params().unwrap();

                    fsdp.module().weight.tensor().data_vec().unwrap()
                })
            })
            .collect();

        for h in handles {
            let data = h.join().unwrap();
            assert_eq!(data, &[11.0, 12.0, 23.0, 24.0]);
        }
    }

    #[test]
    fn test_fsdp_no_shard_is_ddp_equivalent() {
        // NoShard (ZeRO-0 / DDP): each rank has the full parameter and
        // allreduce-averages gradients. Param [1,2,3,4]; both ranks set
        // identical grads [1,2,3,4]; after sync both ranks should see the
        // same averaged grad [1,2,3,4] (identity since both contributions
        // are equal).
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let mut fsdp =
                        FSDP::new_with_strategy(model, b, ShardingStrategy::NoShard).unwrap();
                    assert_eq!(fsdp.strategy(), ShardingStrategy::NoShard);

                    // Params stay full.
                    assert_eq!(fsdp.module().weight.tensor().numel(), 4);

                    let input = ferrotorch_core::from_slice(&[1.0f32], &[1]).unwrap();
                    let _output = fsdp.forward(&input).unwrap();

                    let grad = Tensor::from_storage(
                        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
                        vec![4],
                        false,
                    )
                    .unwrap();
                    fsdp.full_params[0].set_grad(Some(grad)).unwrap();

                    fsdp.sync_gradients().unwrap();

                    let w = fsdp.module().weight.tensor();
                    assert_eq!(w.numel(), 4, "NoShard keeps params full");
                    w.grad().unwrap().unwrap().data_vec().unwrap()
                })
            })
            .collect();

        for h in handles {
            let gd = h.join().unwrap();
            assert_eq!(gd.len(), 4);
            // Mean of two identical [1,2,3,4] vectors is [1,2,3,4].
            for (i, expected) in [1.0f32, 2.0, 3.0, 4.0].iter().enumerate() {
                assert!(
                    (gd[i] - expected).abs() < 1e-6,
                    "NoShard allreduce: got {} at {}, expected {}",
                    gd[i],
                    i,
                    expected
                );
            }
        }
    }

    #[test]
    fn test_fsdp_prefetched_forward_matches_sync_forward() {
        // Prefetch then forward should produce exactly the same output
        // as a plain synchronous forward. 2 ranks, weight [1,2,3,4],
        // input [2.0] -> expected output [20.0].
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let mut fsdp = FSDP::new(model, b).unwrap();

                    // Kick off prefetch BEFORE doing other work.
                    assert!(!fsdp.has_pending_prefetch());
                    fsdp.prefetch_forward_params().unwrap();
                    assert!(fsdp.has_pending_prefetch());

                    // Simulated "local compute" between prefetch and forward.
                    let _scratch: f32 = (0..100).map(|i| i as f32).sum();

                    let input = ferrotorch_core::from_slice(&[2.0f32], &[1]).unwrap();
                    let output = fsdp.forward(&input).unwrap();
                    // After forward, the prefetch handles have been consumed.
                    assert!(!fsdp.has_pending_prefetch());

                    output.data_vec().unwrap()[0]
                })
            })
            .collect();

        for h in handles {
            let v = h.join().unwrap();
            assert!((v - 20.0).abs() < 1e-6, "expected 20.0, got {v}");
        }
    }

    #[test]
    fn test_fsdp_forward_without_prefetch_still_works() {
        // Smoke test: forward() without a prior prefetch should use the
        // synchronous path and produce the same result.
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
                    let mut fsdp = FSDP::new(model, b).unwrap();
                    assert!(!fsdp.has_pending_prefetch());
                    let input = ferrotorch_core::from_slice(&[2.0f32], &[1]).unwrap();
                    let output = fsdp.forward(&input).unwrap();
                    output.data_vec().unwrap()[0]
                })
            })
            .collect();

        for h in handles {
            assert!((h.join().unwrap() - 20.0).abs() < 1e-6);
        }
    }

    #[test]
    fn test_fsdp_prefetch_rejects_double_call() {
        // Calling prefetch twice without an intervening forward should
        // return an error (stale pending handles would be leaked).
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut fsdp = FSDP::new(model, b).unwrap();
        fsdp.prefetch_forward_params().unwrap();
        let r = fsdp.prefetch_forward_params();
        assert!(r.is_err());
        let err = format!("{}", r.unwrap_err());
        assert!(err.contains("called twice"), "err = {err}");

        // Consume the pending handles so the test doesn't leak a live
        // background thread that waits on channel traffic. Running
        // forward() drains and joins them cleanly.
        let input = ferrotorch_core::from_slice(&[1.0f32], &[1]).unwrap();
        let _ = fsdp.forward(&input).unwrap();
    }

    #[test]
    fn test_fsdp_prefetch_rejects_non_fullshard() {
        // Prefetch only makes sense for FullShard (the other strategies
        // don't do a parameter all_gather). Calling it on ShardGradOp
        // must return a clear error.
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut fsdp = FSDP::new_with_strategy(model, b, ShardingStrategy::ShardGradOp).unwrap();
        let r = fsdp.prefetch_forward_params();
        assert!(r.is_err());
        let err = format!("{}", r.unwrap_err());
        assert!(err.contains("FullShard"), "err = {err}");
    }

    #[test]
    fn test_fsdp_no_shard_broadcast_is_noop() {
        // broadcast_updated_params should be a no-op for NoShard and
        // FullShard strategies.
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let model = TestModule::<f32>::new(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        let mut fsdp = FSDP::new_with_strategy(model, b, ShardingStrategy::NoShard).unwrap();
        fsdp.broadcast_updated_params().unwrap();
        assert_eq!(
            fsdp.module().weight.tensor().data_vec().unwrap(),
            &[1.0, 2.0, 3.0, 4.0]
        );
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
                assert!(
                    (data[0] - 1.0).abs() < 1e-6,
                    "rank 0: expected 1.0, got {}",
                    data[0]
                );
                assert!(
                    (data[1] - 2.0).abs() < 1e-6,
                    "rank 0: expected 2.0, got {}",
                    data[1]
                );
            } else {
                // Mean of [3,4] from both ranks = [3,4].
                assert_eq!(data.len(), 2);
                assert!(
                    (data[0] - 3.0).abs() < 1e-6,
                    "rank 1: expected 3.0, got {}",
                    data[0]
                );
                assert!(
                    (data[1] - 4.0).abs() < 1e-6,
                    "rank 1: expected 4.0, got {}",
                    data[1]
                );
            }
        }
    }

    // -----------------------------------------------------------------------
    // HybridShard tests. CL-327
    // -----------------------------------------------------------------------

    #[test]
    fn test_fsdp_hybrid_shard_rejects_uneven_world_size() {
        // world_size=4, intra_node_size=3 is not a divisor → error.
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let b = arcs[0].clone();
        let model = TestModule::<f32>::new(&[1.0f32; 8]).unwrap();
        let result = FSDP::new_with_strategy(
            model,
            b,
            ShardingStrategy::HybridShard { intra_node_size: 3 },
        );
        assert!(result.is_err(), "expected uneven intra_node_size to fail");
    }

    #[test]
    fn test_fsdp_hybrid_shard_intra_node_sharding() {
        // world_size=4, intra_node_size=2:
        //   node 0 = {rank 0, rank 1}
        //   node 1 = {rank 2, rank 3}
        // Each rank shards a param of size 4 into chunks of size 4/2=2.
        //   rank 0 (intra 0, node 0) -> [weight[0], weight[1]]
        //   rank 1 (intra 1, node 0) -> [weight[2], weight[3]]
        //   rank 2 (intra 0, node 1) -> [weight[0], weight[1]]
        //   rank 3 (intra 1, node 1) -> [weight[2], weight[3]]
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = TestModule::<f32>::new(&[10.0, 20.0, 30.0, 40.0]).unwrap();
                    let fsdp = FSDP::new_with_strategy(
                        model,
                        b,
                        ShardingStrategy::HybridShard { intra_node_size: 2 },
                    )
                    .unwrap();
                    let shard = fsdp.module().weight.tensor().data_vec().unwrap();
                    (rank, shard)
                })
            })
            .collect();

        for h in handles {
            let (rank, shard) = h.join().unwrap();
            assert_eq!(shard.len(), 2, "each rank gets 4/2 = 2 elements");
            let expected: &[f32] = if rank % 2 == 0 {
                &[10.0, 20.0]
            } else {
                &[30.0, 40.0]
            };
            assert_eq!(shard, expected, "rank {} shard mismatch", rank);
        }
    }

    #[test]
    fn test_fsdp_hybrid_shard_sync_gradients() {
        // world_size=4, intra_node_size=2:
        //   node 0 = {rank 0, rank 1}, node 1 = {rank 2, rank 3}.
        // Each rank produces the same full grad [1,2,3,4]. After:
        //   intra-node reduce_scatter(mean) → each intra-rank gets its shard
        //     of the node-local mean gradient: rank 0 gets [1,2], rank 1
        //     gets [3,4], rank 2 gets [1,2], rank 3 gets [3,4].
        //   inter-node allreduce(mean) across replicas of the same intra
        //     position: replicas produce identical shards, so the mean is
        //     unchanged.
        // Final expected shard gradients:
        //   ranks 0, 2 → [1, 2]
        //   ranks 1, 3 → [3, 4]
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = TestModule::<f32>::new(&[10.0, 20.0, 30.0, 40.0]).unwrap();
                    let mut fsdp = FSDP::new_with_strategy(
                        model,
                        b,
                        ShardingStrategy::HybridShard { intra_node_size: 2 },
                    )
                    .unwrap();

                    let input = ferrotorch_core::from_slice(&[1.0f32], &[1]).unwrap();
                    let _output = fsdp.forward(&input).unwrap();

                    // After forward, the module param is back to a shard
                    // (intra-node) but full_params[0] holds the gathered
                    // full tensor. Attach a synthetic full-shape gradient.
                    let grad = Tensor::from_storage(
                        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
                        vec![4],
                        false,
                    )
                    .unwrap();
                    fsdp.full_params[0].set_grad(Some(grad)).unwrap();

                    fsdp.sync_gradients().unwrap();

                    let w = fsdp.module().weight.tensor();
                    assert_eq!(w.numel(), 2, "HybridShard keeps params as intra-shards");
                    let g = w.grad().unwrap().unwrap();
                    let gd = g.data_vec().unwrap();
                    (rank, gd)
                })
            })
            .collect();

        for h in handles {
            let (rank, gd) = h.join().unwrap();
            assert_eq!(gd.len(), 2, "shard grad should have 2 elements");
            let expected: &[f32] = if rank % 2 == 0 {
                &[1.0, 2.0]
            } else {
                &[3.0, 4.0]
            };
            for (i, e) in expected.iter().enumerate() {
                assert!(
                    (gd[i] - e).abs() < 1e-6,
                    "rank {} [{}]: expected {}, got {}",
                    rank,
                    i,
                    e,
                    gd[i]
                );
            }
        }
    }

    #[test]
    fn test_fsdp_hybrid_shard_inter_node_averaging() {
        // Construct a scenario where the two nodes disagree on grads, so
        // the inter-node allreduce has meaningful work:
        //   node 0 (ranks 0,1) full grad = [2,4,6,8]
        //   node 1 (ranks 2,3) full grad = [10,20,30,40]
        // Node means after intra reduce_scatter:
        //   rank 0 [2,4], rank 1 [6,8], rank 2 [10,20], rank 3 [30,40].
        // After inter-node allreduce(mean):
        //   rank 0 (intra 0) = mean(rank 0's shard, rank 2's shard) = [6,12]
        //   rank 1 (intra 1) = mean(rank 1's shard, rank 3's shard) = [18,24]
        //   rank 2 (intra 0) = mean(rank 2's shard, rank 0's shard) = [6,12]
        //   rank 3 (intra 1) = mean(rank 3's shard, rank 1's shard) = [18,24]
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let model = TestModule::<f32>::new(&[0.0f32; 4]).unwrap();
                    let mut fsdp = FSDP::new_with_strategy(
                        model,
                        b,
                        ShardingStrategy::HybridShard { intra_node_size: 2 },
                    )
                    .unwrap();

                    let input = ferrotorch_core::from_slice(&[1.0f32], &[1]).unwrap();
                    let _ = fsdp.forward(&input).unwrap();

                    // Node-specific gradient: node 0 ranks 0/1 get
                    // [2,4,6,8], node 1 ranks 2/3 get [10,20,30,40].
                    let grad_vec: Vec<f32> = if rank < 2 {
                        vec![2.0, 4.0, 6.0, 8.0]
                    } else {
                        vec![10.0, 20.0, 30.0, 40.0]
                    };
                    let grad =
                        Tensor::from_storage(TensorStorage::cpu(grad_vec), vec![4], false).unwrap();
                    fsdp.full_params[0].set_grad(Some(grad)).unwrap();

                    fsdp.sync_gradients().unwrap();

                    let gd = fsdp
                        .module()
                        .weight
                        .tensor()
                        .grad()
                        .unwrap()
                        .unwrap()
                        .data_vec()
                        .unwrap();
                    (rank, gd)
                })
            })
            .collect();

        for h in handles {
            let (rank, gd) = h.join().unwrap();
            assert_eq!(gd.len(), 2);
            // intra-position 0 -> [6, 12]; intra-position 1 -> [18, 24]
            let expected: &[f32] = if rank % 2 == 0 {
                &[6.0, 12.0]
            } else {
                &[18.0, 24.0]
            };
            for (i, e) in expected.iter().enumerate() {
                assert!(
                    (gd[i] - e).abs() < 1e-4,
                    "hybrid inter-node mean rank {} [{}]: expected {}, got {}",
                    rank,
                    i,
                    e,
                    gd[i]
                );
            }
        }
    }
}
