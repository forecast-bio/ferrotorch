use std::sync::Arc;

use crate::autograd::autocast::{AutocastSnapshot, current_autocast_snapshot, with_autocast_state};
use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::gpu_dispatch::GpuRngState;
use crate::tensor::Tensor;

/// Type alias for a checkpointable function: takes an input tensor and produces an output tensor.
type CheckpointFn<T> = Arc<dyn Fn(&Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

/// Type alias for a multi-input checkpointable function.
type CheckpointMultiFn<T> = Arc<dyn Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

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

    // Capture the autocast state at forward time so the recomputation during
    // backward runs in the same mixed-precision context. Without this, a
    // checkpoint declared inside `autocast(F16, ...)` would produce f32
    // matmul outputs during recompute (different from forward) and the
    // gradients would be numerically inconsistent.
    let saved_autocast = current_autocast_snapshot();

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
        saved_autocast,
    });

    let device = output.device();
    let storage = TensorStorage::on_device(output.data_vec()?, device)?;
    Tensor::from_operation(storage, output.shape().to_vec(), checkpoint_fn)
}

/// Gradient checkpointing for functions with multiple tensor inputs.
///
/// Like [`checkpoint`], but the function `f` receives a slice of tensors.
/// Gradients are computed for all inputs that have `requires_grad = true`.
///
/// GPU RNG state is saved/restored using the device of the first input.
pub fn checkpoint_multi<T, F>(f: F, inputs: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    use crate::autograd::no_grad::no_grad;
    use crate::storage::TensorStorage;

    if inputs.is_empty() {
        return Err(crate::error::FerrotorchError::InvalidArgument {
            message: "checkpoint_multi: at least one input required".into(),
        });
    }

    let saved_gpu_rng = save_gpu_rng_state(&inputs[0]);
    let saved_autocast = current_autocast_snapshot();

    let output = no_grad(|| f(inputs))?;

    let any_requires_grad = inputs.iter().any(|t| t.requires_grad());
    if !any_requires_grad {
        return Ok(output);
    }

    let checkpoint_fn = Arc::new(CheckpointMultiBackward {
        func: Arc::new(f),
        inputs: inputs.to_vec(),
        output_shape: output.shape().to_vec(),
        saved_gpu_rng,
        saved_autocast,
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
        // XPU has no CUDA-style RNG state to save: ferrotorch-xpu
        // delegates to the cubecl wgpu runtime which manages its own
        // RNG. Treat XPU like CPU here. CL-452.
        crate::device::Device::Xpu(_)
        | crate::device::Device::Cpu
        | crate::device::Device::Mps(_)
        | crate::device::Device::Meta => return None,
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
    /// Autocast (enabled, dtype) state captured at forward time. Restored
    /// for the duration of the recomputation so mixed-precision ops produce
    /// numerically identical activations.
    saved_autocast: AutocastSnapshot,
}

impl<T: Float> std::fmt::Debug for CheckpointBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointBackward")
            .field("input_shape", &self.input.shape())
            .field("output_shape", &self.output_shape)
            .field("has_gpu_rng_state", &self.saved_gpu_rng.is_some())
            .field("autocast_enabled", &self.saved_autocast.enabled)
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

        // Run the recomputation inside an autocast context that exactly
        // matches the forward pass. with_autocast_state uses an RAII guard,
        // so the surrounding (caller's) autocast state is restored even if
        // the recomputation panics.
        with_autocast_state(self.saved_autocast, || {
            let input_with_grad = self.input.clone().requires_grad_(true);
            let recomputed = (self.func)(&input_with_grad)?;

            // Use autograd to compute gradients with grad_output as the
            // upstream gradient. We compute the scalar
            // sum(recomputed * grad_output) and backprop through that;
            // this correctly propagates grad_output through chain rule.
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
        })
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "CheckpointBackward"
    }
}

// ---------------------------------------------------------------------------
// Multi-input checkpoint backward
// ---------------------------------------------------------------------------

struct CheckpointMultiBackward<T: Float> {
    func: CheckpointMultiFn<T>,
    inputs: Vec<Tensor<T>>,
    output_shape: Vec<usize>,
    saved_gpu_rng: Option<GpuRngState>,
    saved_autocast: AutocastSnapshot,
}

impl<T: Float> std::fmt::Debug for CheckpointMultiBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointMultiBackward")
            .field("num_inputs", &self.inputs.len())
            .field("output_shape", &self.output_shape)
            .field("has_gpu_rng_state", &self.saved_gpu_rng.is_some())
            .field("autocast_enabled", &self.saved_autocast.enabled)
            .finish()
    }
}

impl<T: Float> crate::tensor::GradFn<T> for CheckpointMultiBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Restore GPU RNG state for deterministic recomputation.
        let _rng_guard = if let Some(saved_state) = self.saved_gpu_rng {
            let current_state = save_gpu_rng_state(&self.inputs[0]);
            if let Some(backend) = crate::gpu_dispatch::gpu_backend() {
                let _ = backend.restore_rng_state(saved_state);
            }
            current_state.map(|prev| GpuRngGuard { previous: prev })
        } else {
            None
        };

        // Run recomputation under the same autocast state as the forward
        // pass. The RAII guard inside with_autocast_state restores the
        // caller's state on exit, including panic unwind.
        with_autocast_state(self.saved_autocast, || {
            // Re-run forward with grad tracking on all inputs that need it.
            let inputs_with_grad: Vec<Tensor<T>> = self
                .inputs
                .iter()
                .map(|t| {
                    if t.requires_grad() {
                        t.clone().requires_grad_(true)
                    } else {
                        t.clone()
                    }
                })
                .collect();

            let recomputed = (self.func)(&inputs_with_grad)?;

            // Backprop via weighted sum trick.
            use crate::grad_fns::arithmetic::mul;
            use crate::grad_fns::reduction::sum;
            let weighted = mul(
                &recomputed,
                &grad_output.clone().requires_grad_(false).detach(),
            )?;
            let scalar = sum(&weighted)?;
            scalar.backward()?;

            // Collect gradients for each input.
            let mut grads = Vec::with_capacity(self.inputs.len());
            for (orig, with_grad) in self.inputs.iter().zip(inputs_with_grad.iter()) {
                if orig.requires_grad() {
                    grads.push(with_grad.grad()?);
                } else {
                    grads.push(None);
                }
            }
            Ok(grads)
        })
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        self.inputs.iter().collect()
    }

    fn name(&self) -> &'static str {
        "CheckpointMultiBackward"
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::autocast::{AutocastDtype, autocast, is_autocast_enabled};
    use crate::creation::{from_slice, scalar};
    use crate::grad_fns::arithmetic::{add, mul};
    use crate::grad_fns::reduction::sum;
    use crate::storage::TensorStorage;

    fn leaf_grad(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    // -----------------------------------------------------------------------
    // Single-input checkpoint correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_single_input_basic() {
        // f(x) = (x * x) + x  -- df/dx = 2x + 1
        // For x = [1, 2, 3], grad should be [3, 5, 7].
        let x = leaf_grad(&[1.0, 2.0, 3.0], &[3]);
        let y = checkpoint(
            |t: &Tensor<f32>| {
                let sq = mul(t, t)?;
                add(&sq, t)
            },
            &x,
        )
        .unwrap();
        // sum(y) = 1+1 + 4+2 + 9+3 = 20
        let s = sum(&y).unwrap();
        assert!((s.item().unwrap() - 20.0).abs() < 1e-5);

        s.backward().unwrap();
        let g = x.grad().unwrap().expect("x should have a gradient");
        let gd = g.data().unwrap();
        assert!((gd[0] - 3.0).abs() < 1e-5);
        assert!((gd[1] - 5.0).abs() < 1e-5);
        assert!((gd[2] - 7.0).abs() < 1e-5);
    }

    #[test]
    fn test_checkpoint_no_grad_input_returns_output_only() {
        // When input does not require grad, checkpoint should still produce
        // the correct output but skip wrapping in a backward node.
        let x = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let y = checkpoint(
            |t: &Tensor<f32>| {
                let two = scalar(2.0f32)?;
                mul(t, &two)
            },
            &x,
        )
        .unwrap();
        let yd = y.data().unwrap();
        assert_eq!(yd, &[2.0, 4.0, 6.0]);
        // No grad_fn since input had no grad.
        assert!(y.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // Multi-input checkpoint correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_multi_two_inputs_both_grad() {
        // f(a, b) = a * b + a  -- df/da = b + 1, df/db = a
        let a = leaf_grad(&[1.0, 2.0, 3.0], &[3]);
        let b = leaf_grad(&[4.0, 5.0, 6.0], &[3]);
        let y = checkpoint_multi(
            |ts: &[Tensor<f32>]| {
                let prod = mul(&ts[0], &ts[1])?;
                add(&prod, &ts[0])
            },
            &[a.clone(), b.clone()],
        )
        .unwrap();
        // y = [4+1, 10+2, 18+3] = [5, 12, 21]
        let s = sum(&y).unwrap();
        s.backward().unwrap();

        // df/da = b + 1 = [5, 6, 7]
        let ga = a.grad().unwrap().expect("a should have a gradient");
        let gad = ga.data().unwrap();
        assert!((gad[0] - 5.0).abs() < 1e-5);
        assert!((gad[1] - 6.0).abs() < 1e-5);
        assert!((gad[2] - 7.0).abs() < 1e-5);

        // df/db = a = [1, 2, 3]
        let gb = b.grad().unwrap().expect("b should have a gradient");
        let gbd = gb.data().unwrap();
        assert!((gbd[0] - 1.0).abs() < 1e-5);
        assert!((gbd[1] - 2.0).abs() < 1e-5);
        assert!((gbd[2] - 3.0).abs() < 1e-5);
    }

    #[test]
    fn test_checkpoint_multi_partial_grad() {
        // Only the second input requires grad.
        let a = from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let b = leaf_grad(&[4.0, 5.0, 6.0], &[3]);
        let y = checkpoint_multi(
            |ts: &[Tensor<f32>]| mul(&ts[0], &ts[1]),
            &[a.clone(), b.clone()],
        )
        .unwrap();
        let s = sum(&y).unwrap();
        s.backward().unwrap();

        // a has no grad, b's grad should be a.
        assert!(a.grad().unwrap().is_none());
        let gb = b.grad().unwrap().expect("b should have a gradient");
        let gbd = gb.data().unwrap();
        assert_eq!(gbd, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_checkpoint_multi_empty_inputs_errors() {
        let result = checkpoint_multi(|_: &[Tensor<f32>]| panic!("should not run"), &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_checkpoint_multi_no_grad_inputs_returns_output_only() {
        // None of the inputs need grad — output is computed but no backward.
        let a = from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let b = from_slice(&[3.0f32, 4.0], &[2]).unwrap();
        let y = checkpoint_multi(|ts: &[Tensor<f32>]| add(&ts[0], &ts[1]), &[a, b]).unwrap();
        let yd = y.data().unwrap();
        assert_eq!(yd, &[4.0, 6.0]);
        assert!(y.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // Autocast snapshot helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_current_autocast_snapshot_outside_region() {
        let snap = current_autocast_snapshot();
        assert!(!snap.enabled);
    }

    #[test]
    fn test_current_autocast_snapshot_inside_region() {
        autocast(AutocastDtype::BF16, || {
            let snap = current_autocast_snapshot();
            assert!(snap.enabled);
            assert_eq!(snap.dtype, AutocastDtype::BF16);
        });
    }

    #[test]
    fn test_with_autocast_state_restores_disabled() {
        // Snapshot disabled state, then call with_autocast_state from
        // inside an enabled region — the closure should see disabled.
        let disabled = AutocastSnapshot {
            enabled: false,
            dtype: AutocastDtype::F16,
        };
        autocast(AutocastDtype::F16, || {
            assert!(is_autocast_enabled());
            with_autocast_state(disabled, || {
                assert!(!is_autocast_enabled());
            });
            // After the closure, the surrounding autocast region is restored.
            assert!(is_autocast_enabled());
        });
    }

    #[test]
    fn test_with_autocast_state_overrides_dtype() {
        let f16_state = AutocastSnapshot {
            enabled: true,
            dtype: AutocastDtype::F16,
        };
        autocast(AutocastDtype::BF16, || {
            with_autocast_state(f16_state, || {
                assert!(is_autocast_enabled());
                assert_eq!(
                    crate::autograd::autocast::autocast_dtype(),
                    AutocastDtype::F16
                );
            });
            // Restored.
            assert_eq!(
                crate::autograd::autocast::autocast_dtype(),
                AutocastDtype::BF16
            );
        });
    }

    // -----------------------------------------------------------------------
    // Checkpoint preserves autocast state across backward recomputation
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_captures_autocast_snapshot() {
        // When checkpoint is called inside an autocast region, the saved
        // snapshot should reflect that. We can verify this by checking the
        // Debug output of the backward node — its `autocast_enabled` field
        // tracks the snapshot state.
        let x = leaf_grad(&[1.0f32, 2.0, 3.0], &[3]);
        let y_inside = autocast(AutocastDtype::F16, || {
            checkpoint(|t: &Tensor<f32>| mul(t, t), &x)
        })
        .unwrap();
        let dbg = format!("{:?}", y_inside.grad_fn().unwrap());
        assert!(
            dbg.contains("autocast_enabled: true"),
            "expected captured autocast=true in debug repr, got {}",
            dbg
        );
    }

    #[test]
    fn test_checkpoint_outside_autocast_captures_disabled() {
        let x = leaf_grad(&[1.0f32, 2.0, 3.0], &[3]);
        let y = checkpoint(|t: &Tensor<f32>| mul(t, t), &x).unwrap();
        let dbg = format!("{:?}", y.grad_fn().unwrap());
        assert!(
            dbg.contains("autocast_enabled: false"),
            "expected captured autocast=false in debug repr, got {}",
            dbg
        );
    }

    #[test]
    fn test_checkpoint_recomputation_uses_saved_autocast() {
        // The checkpoint is created inside autocast(F16). Backward is called
        // OUTSIDE any autocast region. During recomputation, autocast must
        // be re-enabled (with F16) so the inner ops see the same context.
        // We verify by inspecting the autocast state from inside the
        // recomputation closure via a shared flag.
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicBool, Ordering};

        let saw_autocast = StdArc::new(AtomicBool::new(false));
        let saw_autocast_clone = StdArc::clone(&saw_autocast);

        let x = leaf_grad(&[1.0f32, 2.0, 3.0], &[3]);
        let y = autocast(AutocastDtype::F16, || {
            checkpoint(
                move |t: &Tensor<f32>| {
                    saw_autocast_clone.store(is_autocast_enabled(), Ordering::SeqCst);
                    mul(t, t)
                },
                &x,
            )
        })
        .unwrap();

        // Reset the flag — forward set it to true (we were in autocast).
        saw_autocast.store(false, Ordering::SeqCst);

        // Backward runs OUTSIDE any autocast region.
        assert!(!is_autocast_enabled());
        let s = sum(&y).unwrap();
        s.backward().unwrap();

        // The recomputation closure should have observed autocast = true,
        // because the saved snapshot was restored before the recomputation.
        assert!(
            saw_autocast.load(Ordering::SeqCst),
            "checkpoint backward should re-enable autocast during recomputation"
        );

        // After backward returns, the caller's autocast state is restored
        // (still disabled outside the region).
        assert!(!is_autocast_enabled());
    }

    #[test]
    fn test_checkpoint_multi_recomputation_uses_saved_autocast() {
        // Same as the single-input test but for checkpoint_multi.
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicUsize, Ordering};

        let observed = StdArc::new(AtomicUsize::new(0));
        let observed_clone = StdArc::clone(&observed);

        let a = leaf_grad(&[1.0f32, 2.0], &[2]);
        let b = leaf_grad(&[3.0f32, 4.0], &[2]);

        let y = autocast(AutocastDtype::BF16, || {
            checkpoint_multi(
                move |ts: &[Tensor<f32>]| {
                    let dtype = crate::autograd::autocast::autocast_dtype();
                    let val = if is_autocast_enabled() {
                        match dtype {
                            AutocastDtype::F16 => 1,
                            AutocastDtype::BF16 => 2,
                        }
                    } else {
                        0
                    };
                    observed_clone.store(val, Ordering::SeqCst);
                    add(&ts[0], &ts[1])
                },
                &[a.clone(), b.clone()],
            )
        })
        .unwrap();

        // Forward observed BF16 (= 2). Reset.
        observed.store(0, Ordering::SeqCst);

        let s = sum(&y).unwrap();
        s.backward().unwrap();

        // Backward recomputation should also observe BF16.
        assert_eq!(
            observed.load(Ordering::SeqCst),
            2,
            "expected recomputation to see autocast(BF16), got code {}",
            observed.load(Ordering::SeqCst)
        );
    }

    #[test]
    fn test_checkpoint_recomputation_does_not_leak_autocast() {
        // If the checkpoint was created INSIDE autocast and backward is
        // called OUTSIDE autocast, after backward returns we should still
        // be outside autocast (the with_autocast_state RAII guard restores).
        let x = leaf_grad(&[1.0f32, 2.0], &[2]);
        let y = autocast(AutocastDtype::F16, || {
            checkpoint(|t: &Tensor<f32>| mul(t, t), &x)
        })
        .unwrap();

        assert!(!is_autocast_enabled());
        let s = sum(&y).unwrap();
        s.backward().unwrap();
        assert!(
            !is_autocast_enabled(),
            "checkpoint backward should not leak autocast state to caller"
        );
    }

    #[test]
    fn test_checkpoint_recomputation_inside_different_autocast() {
        // Forward in F16, backward called from inside BF16 region.
        // Recomputation should TEMPORARILY switch to F16, then restore BF16.
        use std::sync::Arc as StdArc;
        use std::sync::atomic::{AtomicU8, Ordering};

        let observed = StdArc::new(AtomicU8::new(0));
        let observed_clone = StdArc::clone(&observed);

        let x = leaf_grad(&[1.0f32, 2.0], &[2]);
        let y = autocast(AutocastDtype::F16, || {
            checkpoint(
                move |t: &Tensor<f32>| {
                    let code: u8 = if is_autocast_enabled() {
                        match crate::autograd::autocast::autocast_dtype() {
                            AutocastDtype::F16 => 1,
                            AutocastDtype::BF16 => 2,
                        }
                    } else {
                        0
                    };
                    observed_clone.store(code, Ordering::SeqCst);
                    mul(t, t)
                },
                &x,
            )
        })
        .unwrap();

        observed.store(0, Ordering::SeqCst);

        autocast(AutocastDtype::BF16, || {
            let s = sum(&y).unwrap();
            s.backward().unwrap();
            // The surrounding BF16 region must be restored after backward
            // (the saved F16 snapshot only applies during the recomputation
            // closure).
            assert_eq!(
                crate::autograd::autocast::autocast_dtype(),
                AutocastDtype::BF16,
                "with_autocast_state should restore caller's BF16 state"
            );
        });

        assert_eq!(
            observed.load(Ordering::SeqCst),
            1,
            "expected recomputation to see F16 (saved snapshot), got code {}",
            observed.load(Ordering::SeqCst)
        );
    }
}
