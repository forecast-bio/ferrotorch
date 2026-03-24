use std::sync::Arc;

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::gpu_dispatch::GpuRngState;
use crate::tensor::Tensor;

/// Type alias for a checkpointable function: takes an input tensor and produces an output tensor.
type CheckpointFn<T> = Arc<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

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
/// If the input is on a CUDA device and a GPU backend is registered, the
/// checkpoint saves the GPU RNG state before the forward pass and restores
/// it during backward recomputation. This ensures stochastic operations
/// like dropout produce identical masks during forward and recomputation,
/// yielding correct gradients.
///
/// If no GPU backend is registered (CPU-only), GPU RNG state is not saved.
/// CPU RNG state for dropout is not currently managed by this checkpoint
/// (the CPU dropout path uses a time-seeded xorshift that is not deterministic
/// across calls).
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

    // Save GPU RNG state before the forward pass so we can restore it during
    // backward recomputation. This ensures dropout masks are identical.
    let saved_gpu_rng = save_gpu_rng_state(input);

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
        saved_gpu_rng,
    });

    let device = output.device();
    let storage = TensorStorage::on_device(output.data_vec()?, device)?;
    Tensor::from_operation(storage, output.shape().to_vec(), checkpoint_fn)
}

/// Save the GPU RNG state for the device the tensor lives on.
///
/// Returns `None` if no GPU backend is registered or the tensor is on CPU.
fn save_gpu_rng_state<T: Float>(tensor: &Tensor<T>) -> Option<GpuRngState> {
    let device_ordinal = match tensor.device() {
        crate::device::Device::Cuda(id) => id,
        crate::device::Device::Cpu => return None,
    };
    let backend = crate::gpu_dispatch::gpu_backend()?;
    backend.save_rng_state(device_ordinal).ok()
}

/// RAII guard that restores GPU RNG state on drop, ensuring cleanup happens
/// on both success and panic paths.
struct GpuRngGuard {
    previous: GpuRngState,
}

impl Drop for GpuRngGuard {
    fn drop(&mut self) {
        if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
            let _ = backend.restore_rng_state(self.previous);
        }
    }
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
    func: CheckpointFn<T>,
    input: Tensor<T>,
    output_shape: Vec<usize>,
    /// GPU RNG state saved before the forward pass. Restored during backward
    /// recomputation so that stochastic ops (dropout) produce identical masks.
    saved_gpu_rng: Option<GpuRngState>,
}

impl<T: Float> std::fmt::Debug for CheckpointBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointBackward")
            .field("input_shape", &self.input.shape())
            .field("output_shape", &self.output_shape)
            .field("has_gpu_rng_state", &self.saved_gpu_rng.is_some())
            .finish()
    }
}

impl<T: Float> crate::tensor::GradFn<T> for CheckpointBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Re-run the forward function WITH gradient tracking to build the graph.
        //
        // If we saved GPU RNG state during the forward pass, restore it now so
        // that stochastic ops (dropout) produce identical masks. The RAII guard
        // saves the current state and restores it when dropped, ensuring the
        // global RNG is not permanently rewound.
        let _rng_guard = if let Some(saved_state) = self.saved_gpu_rng {
            // Save current state to restore after recomputation.
            let current_state = save_gpu_rng_state(&self.input);
            // Restore the state from the forward pass.
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let _ = backend.restore_rng_state(saved_state);
            }
            current_state.map(|prev| GpuRngGuard { previous: prev })
        } else {
            None
        };

        let input_with_grad = self.input.clone().requires_grad_(true);
        let recomputed = (self.func)(&input_with_grad)?;

        // Use autograd to compute gradients with grad_output as the upstream gradient.
        // We need to compute d(recomputed)/d(input) * grad_output.
        // Since backward() on a non-scalar needs an external gradient, we compute
        // the scalar sum(recomputed * grad_output) and backprop through that.
        // This correctly propagates grad_output through the chain rule.
        use crate::grad_fns::arithmetic::mul;
        use crate::grad_fns::reduction::sum;
        let weighted = mul(
            &recomputed,
            &grad_output.clone().requires_grad_(false).detach(),
        )?;
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
