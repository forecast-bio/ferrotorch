use std::sync::Arc;

use crate::creation::{restore_rng_state, save_rng_state, RngState};
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::tensor::Tensor;

/// Run a function with gradient checkpointing (multi-tensor input).
///
/// During the forward pass, intermediate activations are **not** saved.
/// During the backward pass, the forward function is re-executed to
/// recompute them, trading compute for memory.
///
/// The CPU RNG state is snapshotted before the forward pass and restored
/// before recomputation so that stochastic operations (e.g., dropout)
/// produce identical results in both passes.
///
/// # Arguments
///
/// * `f` - The forward function to checkpoint. It receives a slice of input
///   tensors and returns a single output tensor.
/// * `inputs` - A slice of input tensors. At least one must have
///   `requires_grad = true` for the backward graph to be recorded.
///
/// # Returns
///
/// The output tensor, with a grad_fn that will recompute `f` during backward.
///
/// # Errors
///
/// Returns `FerrotorchError::InvalidArgument` if `inputs` is empty.
pub fn checkpoint<T, F>(f: F, inputs: &[&Tensor<T>]) -> FerrotorchResult<Tensor<T>>
where
    T: Float,
    F: Fn(&[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync + 'static,
{
    use crate::autograd::no_grad::no_grad;
    use crate::storage::TensorStorage;

    if inputs.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "checkpoint: inputs must not be empty".into(),
        });
    }

    // Snapshot the RNG state *before* the forward pass so we can replay it.
    let rng_state = save_rng_state();

    // Forward pass without recording the graph (saves memory).
    let output = no_grad(|| f(inputs))?;

    // If no input requires grad, just return the output as-is.
    let any_requires_grad = inputs.iter().any(|t| t.requires_grad());
    if !any_requires_grad {
        return Ok(output);
    }

    // Determine the output device — the result must live on the same device.
    let device = output.device();

    // Clone inputs for later recomputation in the backward pass.
    let saved_inputs: Vec<Tensor<T>> = inputs.iter().map(|t| (*t).clone()).collect();
    let output_shape = output.shape().to_vec();

    let checkpoint_fn = Arc::new(CheckpointBackward {
        func: Arc::new(f),
        inputs: saved_inputs,
        output_shape,
        rng_state,
    });

    let storage = TensorStorage::on_device(output.data_vec()?, device)?;
    Tensor::from_operation(storage, output.shape().to_vec(), checkpoint_fn)
}

struct CheckpointBackward<T: Float> {
    func: Arc<dyn Fn(&[&Tensor<T>]) -> FerrotorchResult<Tensor<T>> + Send + Sync>,
    inputs: Vec<Tensor<T>>,
    output_shape: Vec<usize>,
    rng_state: RngState,
}

impl<T: Float> std::fmt::Debug for CheckpointBackward<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CheckpointBackward")
            .field(
                "input_shapes",
                &self
                    .inputs
                    .iter()
                    .map(|t| t.shape().to_vec())
                    .collect::<Vec<_>>(),
            )
            .field("output_shape", &self.output_shape)
            .finish()
    }
}

impl<T: Float> crate::tensor::GradFn<T> for CheckpointBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Prepare inputs with requires_grad = true so the recomputed graph
        // records operations against them.
        let inputs_with_grad: Vec<Tensor<T>> = self
            .inputs
            .iter()
            .map(|t| t.clone().requires_grad_(true))
            .collect();

        let input_refs: Vec<&Tensor<T>> = inputs_with_grad.iter().collect();

        // Restore the RNG state to what it was before the original forward
        // pass so that stochastic ops (dropout, etc.) produce the same masks.
        let prev_rng = restore_rng_state(self.rng_state);

        // Re-run the forward function WITH gradient tracking.
        let recomputed = (self.func)(&input_refs)?;

        // Restore the RNG state that was active before we overwrote it.
        restore_rng_state(prev_rng);

        // Propagate upstream gradient through the recomputed graph directly
        // using backward_with_grad, which correctly handles non-scalar
        // outputs without the multiply-and-sum workaround.
        use crate::autograd::graph::backward_with_grad;
        backward_with_grad(&recomputed, Some(grad_output))?;

        // Collect gradients for each input.
        let grads: Vec<Option<Tensor<T>>> = inputs_with_grad
            .iter()
            .map(|t| t.grad().unwrap_or(None))
            .collect();

        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        self.inputs.iter().collect()
    }

    fn name(&self) -> &'static str {
        "CheckpointBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{from_slice, manual_seed, ones, rand, zeros};
    use crate::grad_fns::arithmetic::{add, mul};
    use crate::grad_fns::reduction::sum;
    use crate::storage::TensorStorage;

    /// Helper: leaf tensor from a flat slice and shape.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Single-input checkpoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_single_input_forward_value() {
        // f(x) = x * x, checkpoint should produce the same forward value.
        let x = leaf(&[2.0, 3.0], &[2], true);
        let out = checkpoint(
            |inputs| mul(inputs[0], inputs[0]),
            &[&x],
        )
        .unwrap();
        let data = out.data().unwrap();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_single_input_backward() {
        // f(x) = x * x, d/dx = 2x
        let x = leaf(&[2.0, 3.0], &[2], true);
        let out = checkpoint(
            |inputs| mul(inputs[0], inputs[0]),
            &[&x],
        )
        .unwrap();
        let loss = sum(&out).unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().unwrap();
        let gd = grad.data().unwrap();
        assert!((gd[0] - 4.0).abs() < 1e-5, "got {}", gd[0]);
        assert!((gd[1] - 6.0).abs() < 1e-5, "got {}", gd[1]);
    }

    // -----------------------------------------------------------------------
    // Multi-input checkpoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_multi_input_forward() {
        // f(a, b) = a + b
        let a = leaf(&[1.0, 2.0], &[2], true);
        let b = leaf(&[3.0, 4.0], &[2], true);
        let out = checkpoint(
            |inputs| add(inputs[0], inputs[1]),
            &[&a, &b],
        )
        .unwrap();
        let data = out.data().unwrap();
        assert!((data[0] - 4.0).abs() < 1e-6);
        assert!((data[1] - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_multi_input_backward() {
        // f(a, b) = a * b, d/da = b, d/db = a
        let a = leaf(&[2.0, 3.0], &[2], true);
        let b = leaf(&[5.0, 7.0], &[2], true);
        let out = checkpoint(
            |inputs| mul(inputs[0], inputs[1]),
            &[&a, &b],
        )
        .unwrap();
        let loss = sum(&out).unwrap();
        loss.backward().unwrap();

        let ga = a.grad().unwrap().unwrap();
        let gb = b.grad().unwrap().unwrap();
        let gad = ga.data().unwrap();
        let gbd = gb.data().unwrap();
        // d(loss)/da = b
        assert!((gad[0] - 5.0).abs() < 1e-5, "got {}", gad[0]);
        assert!((gad[1] - 7.0).abs() < 1e-5, "got {}", gad[1]);
        // d(loss)/db = a
        assert!((gbd[0] - 2.0).abs() < 1e-5, "got {}", gbd[0]);
        assert!((gbd[1] - 3.0).abs() < 1e-5, "got {}", gbd[1]);
    }

    // -----------------------------------------------------------------------
    // No-grad inputs: checkpoint returns plain output, no backward error
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_no_grad_inputs() {
        let x = leaf(&[1.0, 2.0], &[2], false);
        let out = checkpoint(
            |inputs| {
                let doubled_data: Vec<f32> =
                    inputs[0].data().unwrap().iter().map(|&v| v * 2.0).collect();
                from_slice(&doubled_data, inputs[0].shape())
            },
            &[&x],
        )
        .unwrap();
        assert!(!out.requires_grad());
        let data = out.data().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 4.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Empty inputs: should return an error
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_empty_inputs() {
        let result: FerrotorchResult<Tensor<f32>> =
            checkpoint(|_inputs| zeros(&[2]), &[]);
        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // RNG preservation: a function that uses rand() inside checkpoint
    // must produce the same random values during recomputation.
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_rng_preservation() {
        // Simulate dropout: mask = rand() < 0.5, out = x * mask
        // The checkpoint backward must restore the RNG so the same mask
        // is generated during recomputation, producing correct gradients.
        manual_seed(12345);

        let x = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], true);

        // We need to capture what the dropout mask looks like so we can
        // verify the gradient. First, figure out the mask by running the
        // same operation with the same seed outside of checkpoint.
        manual_seed(42);
        let mask_tensor: Tensor<f32> = rand(&[4]).unwrap();
        let mask_data = mask_tensor.data().unwrap().to_vec();
        let binary_mask: Vec<f32> = mask_data.iter().map(|&v| if v < 0.5 { 1.0 } else { 0.0 }).collect();

        // Now run the checkpointed version with the same seed.
        manual_seed(42);
        let out = checkpoint(
            |inputs| {
                // Simulated dropout: generate mask from RNG, multiply elementwise.
                let mask_t: Tensor<f32> = rand(inputs[0].shape())?;
                let mask_vals = mask_t.data()?.to_vec();
                let binary: Vec<f32> = mask_vals.iter().map(|&v| if v < 0.5 { 1.0 } else { 0.0 }).collect();
                let mask = Tensor::from_storage(
                    TensorStorage::cpu(binary),
                    inputs[0].shape().to_vec(),
                    false,
                )?;
                mul(inputs[0], &mask)
            },
            &[&x],
        )
        .unwrap();

        // Verify forward values match expected dropout.
        let out_data = out.data().unwrap();
        for i in 0..4 {
            let expected = (i as f32 + 1.0) * binary_mask[i];
            assert!(
                (out_data[i] - expected).abs() < 1e-6,
                "forward mismatch at {}: got {} expected {}",
                i,
                out_data[i],
                expected,
            );
        }

        // Backward: d(sum(out))/dx = binary_mask (since out = x * mask).
        let loss = sum(&out).unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().unwrap();
        let gd = grad.data().unwrap();
        for i in 0..4 {
            assert!(
                (gd[i] - binary_mask[i]).abs() < 1e-5,
                "grad mismatch at {}: got {} expected {}",
                i,
                gd[i],
                binary_mask[i],
            );
        }
    }

    // -----------------------------------------------------------------------
    // Device preservation: output device matches input device
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_device_preservation() {
        use crate::device::Device;
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let out = checkpoint(
            |inputs| mul(inputs[0], inputs[0]),
            &[&x],
        )
        .unwrap();
        assert_eq!(out.device(), Device::Cpu);
        assert_eq!(out.device(), x.device());
    }

    // -----------------------------------------------------------------------
    // Nested checkpoint: checkpoint inside checkpoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_nested() {
        // outer: f(x) = checkpoint_inner(x * 2)
        // inner: g(y) = y + y = 2y
        // overall: f(x) = 2 * (x * 2) = 4x, gradient = 4
        let x = leaf(&[3.0], &[1], true);

        let out = checkpoint(
            |outer_inputs| {
                // x * 2
                let two = Tensor::from_storage(
                    TensorStorage::cpu(vec![2.0f32]),
                    vec![1],
                    false,
                )?;
                let doubled = mul(outer_inputs[0], &two)?;

                // Inner checkpoint: y + y
                checkpoint(
                    |inner_inputs| add(inner_inputs[0], inner_inputs[0]),
                    &[&doubled],
                )
            },
            &[&x],
        )
        .unwrap();

        let out_val = out.data().unwrap()[0];
        assert!((out_val - 12.0).abs() < 1e-5, "expected 12.0, got {}", out_val);

        let loss = sum(&out).unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().unwrap();
        let gd = grad.data().unwrap()[0];
        assert!((gd - 4.0).abs() < 1e-5, "expected gradient 4.0, got {}", gd);
    }

    // -----------------------------------------------------------------------
    // Checkpoint with mixed requires_grad inputs
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_mixed_requires_grad() {
        // a requires grad, b does not. f(a, b) = a * b.
        // d(loss)/da = b, d(loss)/db = None (b is not a leaf with grad).
        let a = leaf(&[2.0, 3.0], &[2], true);
        let b = leaf(&[5.0, 7.0], &[2], false);

        let out = checkpoint(
            |inputs| mul(inputs[0], inputs[1]),
            &[&a, &b],
        )
        .unwrap();
        let loss = sum(&out).unwrap();
        loss.backward().unwrap();

        let ga = a.grad().unwrap().unwrap();
        let gad = ga.data().unwrap();
        assert!((gad[0] - 5.0).abs() < 1e-5);
        assert!((gad[1] - 7.0).abs() < 1e-5);

        // b has no gradient because it doesn't require grad.
        assert!(b.grad().unwrap().is_none());
    }

    // -----------------------------------------------------------------------
    // RNG state save/restore round-trip
    // -----------------------------------------------------------------------

    #[test]
    fn test_rng_state_round_trip() {
        manual_seed(99);
        let state = save_rng_state();

        let t1: Tensor<f32> = rand(&[5]).unwrap();
        let d1 = t1.data().unwrap().to_vec();

        restore_rng_state(state);
        let t2: Tensor<f32> = rand(&[5]).unwrap();
        let d2 = t2.data().unwrap().to_vec();

        assert_eq!(d1, d2, "RNG state restore should reproduce the same values");
    }

    // -----------------------------------------------------------------------
    // Scalar output checkpoint
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_scalar_output() {
        // f(x) = sum(x * x)
        let x = leaf(&[1.0, 2.0, 3.0], &[3], true);
        let out = checkpoint(
            |inputs| {
                let sq = mul(inputs[0], inputs[0])?;
                sum(&sq)
            },
            &[&x],
        )
        .unwrap();
        assert!(out.is_scalar() || out.numel() == 1);
        out.backward().unwrap();

        let grad = x.grad().unwrap().unwrap();
        let gd = grad.data().unwrap();
        // d(sum(x^2))/dx = 2x
        assert!((gd[0] - 2.0).abs() < 1e-5);
        assert!((gd[1] - 4.0).abs() < 1e-5);
        assert!((gd[2] - 6.0).abs() < 1e-5);
    }

    // -----------------------------------------------------------------------
    // Checkpoint output shape matches direct computation
    // -----------------------------------------------------------------------

    #[test]
    fn test_checkpoint_preserves_shape() {
        let x = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], true);
        let out = checkpoint(
            |inputs| {
                // Just pass through
                let o = ones(&[2, 3]).unwrap();
                add(inputs[0], &o)
            },
            &[&x],
        )
        .unwrap();
        assert_eq!(out.shape(), &[2, 3]);
    }
}
