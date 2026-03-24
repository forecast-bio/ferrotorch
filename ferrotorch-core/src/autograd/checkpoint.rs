use std::sync::Arc;

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::tensor::Tensor;

/// Run a function with gradient checkpointing.
///
/// During the forward pass, intermediate activations are **not** saved.
/// During the backward pass, the forward function is re-executed to
/// recompute them, trading compute for memory.
///
/// This is useful for very deep networks where storing all activations
/// would exceed available memory.
///
/// # Arguments
///
/// * `f` - The forward function to checkpoint. It receives the input tensor
///   and returns the output tensor.
/// * `input` - The input tensor. Must have `requires_grad = true`.
///
/// # Returns
///
/// The output tensor, with a grad_fn that will recompute `f` during backward.
///
/// # Saved inputs and storage sharing
///
/// The checkpoint stores a clone of the input tensor. Because `Tensor` is an
/// `Arc`-wrapped type, the clone shares the same underlying `TensorStorage`.
/// If the caller mutates the storage in-place between the forward and backward
/// passes (which is unusual but possible via unsafe code), the recomputation
/// will see the mutated data. This is the same behavior as PyTorch.
///
/// # RNG reproducibility
///
/// **Warning:** This implementation does not currently save or restore RNG
/// state. If `f` uses stochastic operations (e.g., dropout), the backward
/// recomputation will produce different random values than the forward pass,
/// leading to incorrect gradients. Deterministic checkpoint functions (no
/// dropout, no random sampling) are unaffected.
///
/// # Thread-local state and rayon
///
/// **Warning:** Both [`no_grad`] and `GRAD_ENABLED` use `thread_local!`
/// storage. When `f` spawns work onto rayon worker threads (e.g., via
/// parallel iterators), those threads will **not** inherit the calling
/// thread's gradient-enabled flag. This means operations executed on rayon
/// threads inside a `no_grad` block may still record gradients. This is a
/// known limitation — fixing it properly requires per-rayon-thread state
/// propagation which is a larger effort.
pub fn checkpoint<T, F>(f: F, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    use crate::autograd::no_grad::no_grad;
    use crate::storage::TensorStorage;

    // Forward pass without recording the graph (saves memory).
    let output = no_grad(|| f(input))?;

    if !input.requires_grad() {
        return Ok(output);
    }

    // Wrap in a CheckpointBackward that re-runs f during backward.
    let checkpoint_fn = Arc::new(CheckpointBackward {
        func: Arc::new(f),
        input: input.clone(),
        output_shape: output.shape().to_vec(),
    });

    let device = output.device();
    let storage = TensorStorage::on_device(output.data_vec()?, device)?;
    Tensor::from_operation(storage, output.shape().to_vec(), checkpoint_fn)
}

/// Internal backward node for gradient checkpointing.
///
/// # TensorId aliasing invariant
///
/// The `input` field stores a clone of the original input tensor. Because
/// `Tensor::clone()` is an `Arc` clone, the stored tensor shares the same
/// `TensorId` as the original. This is **intentional**: the autograd engine
/// uses `TensorId` to accumulate gradients, so the checkpoint's input must
/// have the same identity as the user's input tensor. If `TensorId` were
/// reassigned on clone, gradients computed during recomputation would be
/// written to a different identity and the user would never see them.
struct CheckpointBackward<T: Float> {
    func: Arc<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>,
    input: Tensor<T>,
    output_shape: Vec<usize>,
}

impl<T: Float> std::fmt::Debug for CheckpointBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointBackward")
            .field("input_shape", &self.input.shape())
            .field("output_shape", &self.output_shape)
            .finish()
    }
}

impl<T: Float> crate::tensor::GradFn<T> for CheckpointBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Re-run the forward function WITH gradient tracking to build the graph.
        //
        // TODO(RNG): When a proper seeded RNG state system is added (manual_seed +
        // thread-local generator), save the RNG state before forward and restore it
        // here with an RAII guard so that stochastic ops (dropout) produce identical
        // values during recomputation. The guard must use Drop to handle both success
        // and failure paths:
        //
        //   struct RngGuard { prev: RngState }
        //   impl Drop for RngGuard {
        //       fn drop(&mut self) { restore_rng_state(self.prev); }
        //   }
        //   let _rng_guard = RngGuard { prev: save_rng_state() };
        //   set_rng_state(self.saved_rng_state);
        //
        let input_with_grad = self.input.clone().requires_grad_(true);
        let recomputed = (self.func)(&input_with_grad)?;

        // Use autograd to compute gradients with grad_output as the upstream gradient.
        // We need to compute d(recomputed)/d(input) * grad_output.
        // Since backward() on a non-scalar needs an external gradient, we compute
        // the scalar sum(recomputed * grad_output) and backprop through that.
        // This correctly propagates grad_output through the chain rule.
        use crate::grad_fns::arithmetic::mul;
        use crate::grad_fns::reduction::sum;
        let weighted = mul(&recomputed, &grad_output.clone().requires_grad_(false).detach())?;
        let scalar = sum(&weighted)?;
        scalar.backward()?;

        let input_grad = input_with_grad.grad()?;
        Ok(vec![input_grad])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CheckpointBackward"
    }
}
