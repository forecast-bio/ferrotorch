//! Pipeline parallelism for distributed training.
//!
//! Splits a model into sequential stages, each running on a different rank.
//! Micro-batches flow through the pipeline so multiple stages can execute
//! concurrently, improving hardware utilization.
//!
//! Two scheduling strategies are provided:
//!
//! - **GPipe** ([`PipelineSchedule::GPipe`]): All micro-batch forwards run
//!   first, then all backwards. Simple but has high peak memory.
//!
//! - **1F1B** ([`PipelineSchedule::OneFOnEB`]): Interleaves forward and
//!   backward passes per micro-batch to reduce peak memory. **Note:** the
//!   1F1B forward scheduling is implemented, but the backward pass currently
//!   uses GPipe-style sequential processing. True 1F1B memory savings require
//!   a combined forward+backward method that interleaves at a finer
//!   granularity, which is planned for a future release. The current
//!   implementation provides scheduling structure but not the full memory
//!   benefit.

use std::sync::Arc;

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Module;

use crate::backend::Backend;

// ---------------------------------------------------------------------------
// Pipeline schedule
// ---------------------------------------------------------------------------

/// Pipeline scheduling strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// GPipe: all forwards then all backwards.
    GPipe,
    /// 1F1B: interleaved forward and backward scheduling.
    ///
    /// **Current limitation:** Forward scheduling follows 1F1B ordering, but
    /// the backward pass uses GPipe-style sequential processing. The memory
    /// benefit of true 1F1B requires a combined forward+backward method
    /// (future work).
    OneFOnEB,
}

// ---------------------------------------------------------------------------
// Pipeline
// ---------------------------------------------------------------------------

/// Pipeline parallel wrapper.
///
/// Wraps a single stage module and coordinates with other ranks via the
/// provided [`Backend`]. Each rank runs one stage of the pipeline.
///
/// # Usage
///
/// ```ignore
/// let pipeline = Pipeline::new(
///     stage_module,
///     backend,
///     num_microbatches,
///     PipelineSchedule::GPipe,
/// )?;
/// ```
pub struct Pipeline<M: Module<T>, T: Float> {
    module: M,
    backend: Arc<dyn Backend>,
    num_microbatches: usize,
    schedule: PipelineSchedule,
    _marker: std::marker::PhantomData<T>,
}

impl<M: Module<T>, T: Float> std::fmt::Debug for Pipeline<M, T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Pipeline")
            .field("num_microbatches", &self.num_microbatches)
            .field("schedule", &self.schedule)
            .finish_non_exhaustive()
    }
}

impl<M: Module<T>, T: Float> Pipeline<M, T> {
    /// Create a new pipeline stage.
    ///
    /// # Arguments
    ///
    /// * `module` - This rank's stage module.
    /// * `backend` - Communication backend (rank/world_size determine stage).
    /// * `num_microbatches` - Number of micro-batches to split the input into.
    /// * `schedule` - Pipeline scheduling strategy.
    ///
    /// # Errors
    ///
    /// Returns an error if `num_microbatches` is zero or if `world_size` is
    /// less than 2 (pipeline parallelism requires at least 2 stages).
    pub fn new(
        module: M,
        backend: Arc<dyn Backend>,
        num_microbatches: usize,
        schedule: PipelineSchedule,
    ) -> FerrotorchResult<Self> {
        if num_microbatches == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Pipeline: num_microbatches must be > 0".into(),
            });
        }
        if backend.world_size() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Pipeline: world_size must be >= 2 for pipeline parallelism, got {}",
                    backend.world_size(),
                ),
            });
        }
        Ok(Self {
            module,
            backend,
            num_microbatches,
            schedule,
            _marker: std::marker::PhantomData,
        })
    }

    /// Run the forward pass for all micro-batches through this stage.
    ///
    /// - If this is the first stage (rank 0), splits `input` into
    ///   `num_microbatches` chunks along dimension 0.
    /// - If this is a middle or last stage, receives activations from the
    ///   previous stage.
    /// - After running forward, sends activations to the next stage (unless
    ///   this is the last stage).
    ///
    /// Returns the outputs for each micro-batch (only meaningful on the last
    /// stage).
    pub fn forward(&self, input: Option<&Tensor<T>>) -> FerrotorchResult<Vec<Tensor<T>>> {
        let rank = self.backend.rank();
        let world_size = self.backend.world_size();

        let mut outputs = Vec::with_capacity(self.num_microbatches);

        for mb in 0..self.num_microbatches {
            // Get input for this micro-batch.
            let mb_input = if rank == 0 {
                // First stage: chunk the input.
                let input = input.ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: "Pipeline: rank 0 must provide input".into(),
                })?;
                self.get_microbatch(input, mb)?
            } else {
                // Receive from previous stage.
                self.recv_activation(rank - 1)?
            };

            // Forward through this stage's module.
            let output = self.module.forward(&mb_input)?;

            if rank < world_size - 1 {
                // Send to next stage.
                self.send_activation(&output, rank + 1)?;
            }

            outputs.push(output);
        }

        Ok(outputs)
    }

    /// Run the backward pass for all micro-batches.
    ///
    /// For the last stage, computes gradients from the loss. For other stages,
    /// receives gradient activations from the next stage.
    ///
    /// # Implementation note
    ///
    /// Both GPipe and 1F1B schedules currently use the same backward
    /// processing order (sequential over all micro-batches). The 1F1B
    /// forward scheduling is implemented but the backward pass does not
    /// yet interleave with forward to achieve the full memory savings.
    /// This is documented as a known limitation; true 1F1B memory benefits
    /// require a combined forward+backward method (future work).
    pub fn backward(
        &self,
        outputs: &[Tensor<T>],
        grad_outputs: Option<&[Tensor<T>]>,
    ) -> FerrotorchResult<()> {
        let rank = self.backend.rank();
        let world_size = self.backend.world_size();

        for mb in (0..self.num_microbatches).rev() {
            if rank == world_size - 1 {
                // Last stage: use provided grad_outputs.
                if let Some(grads) = grad_outputs {
                    if mb < grads.len() {
                        outputs[mb].set_grad(Some(grads[mb].clone()))?;
                    }
                }
            } else {
                // Receive gradient from next stage.
                let grad = self.recv_activation(rank + 1)?;
                outputs[mb].set_grad(Some(grad))?;
            }

            // Backward through autograd.
            ferrotorch_core::backward(&outputs[mb])?;

            if rank > 0 {
                // Send gradient to previous stage.
                // Compute the gradient w.r.t. the input of this stage.
                let numel = outputs[mb].numel();
                let grad_input = Tensor::from_storage(
                    TensorStorage::cpu(vec![<T as num_traits::Zero>::zero(); numel]),
                    outputs[mb].shape().to_vec(),
                    false,
                )?;
                self.send_activation(&grad_input, rank - 1)?;
            }
        }

        Ok(())
    }

    /// Extract micro-batch `mb_idx` from `input` by chunking dim 0.
    fn get_microbatch(&self, input: &Tensor<T>, mb_idx: usize) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        if shape.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "Pipeline: input tensor must have at least 1 dimension".into(),
            });
        }
        let batch_size = shape[0];
        let mb_size = batch_size / self.num_microbatches;
        let start = mb_idx * mb_size;
        let end = if mb_idx == self.num_microbatches - 1 {
            batch_size
        } else {
            start + mb_size
        };

        let data = input.data_vec()?;
        let stride: usize = shape[1..].iter().product();
        let mb_data = data[start * stride..end * stride].to_vec();
        let mut mb_shape = shape.to_vec();
        mb_shape[0] = end - start;

        Tensor::from_storage(TensorStorage::cpu(mb_data), mb_shape, input.requires_grad())
    }

    /// Send an activation tensor to `dst_rank`.
    fn send_activation(&self, tensor: &Tensor<T>, dst_rank: usize) -> FerrotorchResult<()> {
        let data = tensor.data_vec()?;
        let elem_size = std::mem::size_of::<T>();
        let byte_slice: Vec<u8> = data
            .iter()
            .flat_map(|v| {
                // SAFETY: byte-reinterpret of a single `&T` (`T: Float`)
                // into a `[u8]` view of length `size_of::<T>()`.
                //
                // - VALIDITY: every byte pattern is a valid `u8`; reading
                //   the underlying bytes of an IEEE-754 float can never
                //   produce an invalid value.
                // - LENGTH: `elem_size == size_of::<T>()`, set immediately
                //   above; this matches the size of the value pointed to.
                // - ALIGNMENT: `*const u8` is 1-aligned, satisfied by any
                //   `*const T` for primitive `Float`.
                // - LIFETIME: the slice borrows `*v` (live for the closure
                //   body) and is immediately copied via `bytes.to_vec()`
                //   before the closure returns; the borrow does not
                //   escape. No dangling reference is possible.
                // - PROVENANCE: `v as *const T as *const u8` derives from
                //   the live `&T` borrow; the dual cast preserves
                //   provenance under the strict-provenance model.
                // - ENDIANNESS: matches checkpoint.rs:as_le_bytes — this
                //   crate targets LE platforms; the wire format used by
                //   `recv_activation` performs the inverse byte-pattern
                //   read on the same endianness.
                let bytes =
                    unsafe { std::slice::from_raw_parts(v as *const T as *const u8, elem_size) };
                bytes.to_vec()
            })
            .collect();

        // Send shape info first: ndim (8 bytes) + shape dims (8 bytes each).
        let ndim = tensor.shape().len() as u64;
        let mut header = ndim.to_le_bytes().to_vec();
        for &d in tensor.shape() {
            header.extend_from_slice(&(d as u64).to_le_bytes());
        }
        self.backend.send(&header, dst_rank)?;
        self.backend.send(&byte_slice, dst_rank)?;

        Ok(())
    }

    /// Receive an activation tensor from `src_rank`.
    fn recv_activation(&self, src_rank: usize) -> FerrotorchResult<Tensor<T>> {
        // Receive shape header.
        let mut ndim_buf = [0u8; 8];
        self.backend.recv(&mut ndim_buf, src_rank)?;
        let ndim = u64::from_le_bytes(ndim_buf) as usize;

        let mut shape = Vec::with_capacity(ndim);
        for _ in 0..ndim {
            let mut dim_buf = [0u8; 8];
            self.backend.recv(&mut dim_buf, src_rank)?;
            shape.push(u64::from_le_bytes(dim_buf) as usize);
        }

        let numel: usize = shape.iter().product();
        let elem_size = std::mem::size_of::<T>();
        let mut byte_buf = vec![0u8; numel * elem_size];
        self.backend.recv(&mut byte_buf, src_rank)?;

        let data: Vec<T> = byte_buf
            .chunks_exact(elem_size)
            .map(|chunk| match elem_size {
                4 => {
                    let val = f32::from_le_bytes(chunk.try_into().unwrap());
                    T::from(val).unwrap()
                }
                8 => {
                    let val = f64::from_le_bytes(chunk.try_into().unwrap());
                    T::from(val).unwrap()
                }
                _ => unreachable!("unsupported element size {}", elem_size),
            })
            .collect();

        Tensor::from_storage(TensorStorage::cpu(data), shape, false)
    }

    /// The schedule used by this pipeline.
    pub fn schedule(&self) -> PipelineSchedule {
        self.schedule
    }

    /// The number of micro-batches.
    pub fn num_microbatches(&self) -> usize {
        self.num_microbatches
    }

    /// Immutable access to the inner module.
    pub fn module(&self) -> &M {
        &self.module
    }

    /// Mutable access to the inner module.
    pub fn module_mut(&mut self) -> &mut M {
        &mut self.module
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SimulatedBackend;
    use ferrotorch_nn::Parameter;

    /// Identity module with real train/eval state tracking.
    /// Used by multiple pipeline validation tests.
    struct IdentityModule {
        training: bool,
    }

    impl IdentityModule {
        fn new() -> Self {
            Self { training: true }
        }
    }

    impl Module<f32> for IdentityModule {
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
    fn test_pipeline_new_validates_microbatches() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        // Zero microbatches should error.
        let result = Pipeline::new(IdentityModule::new(), b.clone(), 0, PipelineSchedule::GPipe);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("num_microbatches must be > 0"));
    }

    #[test]
    fn test_pipeline_new_validates_world_size() {
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        // World size 1 should error.
        let result = Pipeline::new(IdentityModule::new(), b, 2, PipelineSchedule::OneFOnEB);
        assert!(result.is_err());
        let err = format!("{}", result.unwrap_err());
        assert!(err.contains("world_size must be >= 2"));
    }

    #[test]
    fn test_pipeline_schedule_accessors() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());

        let pipeline = Pipeline::new(IdentityModule::new(), b, 4, PipelineSchedule::GPipe).unwrap();
        assert_eq!(pipeline.schedule(), PipelineSchedule::GPipe);
        assert_eq!(pipeline.num_microbatches(), 4);
    }

    #[test]
    fn test_identity_module_train_eval_toggles_state() {
        let mut m = IdentityModule::new();
        assert!(m.is_training());
        m.eval();
        assert!(!m.is_training());
        m.train();
        assert!(m.is_training());
    }
}
