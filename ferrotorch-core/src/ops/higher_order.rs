//! Higher-order tensor operations: `cond` and `scan`.
//!
//! These operations enable conditional execution and sequential state
//! accumulation within the autograd graph, critical for architectures
//! like state-space models (Mamba, S4) and dynamic control flow.

use std::sync::Arc;

use crate::autograd::no_grad::is_grad_enabled;
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ===========================================================================
// cond — conditional subgraph execution
// ===========================================================================

/// Backward node for `cond`.
///
/// Holds the raw branch output (which already carries the branch function's
/// grad_fn chain). Routes the upstream gradient to it; the autograd engine
/// continues traversal through the branch's grad chain to compute per-operand
/// gradients correctly — even for non-identity branches like
/// `&ops[0] * &ops[1]`.
///
/// Earlier versions held the operands directly and returned identity grads to
/// each one; that bypassed the branch function's grad_fn chain and produced
/// wrong gradients for any branch that wasn't a pure pass-through. The dead
/// `branch` and `operands` fields were the structural evidence of that gap.
#[derive(Debug)]
struct CondBackward<T: Float> {
    branch_output: Tensor<T>,
}

impl<T: Float> GradFn<T> for CondBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(grad_output.clone())])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.branch_output]
    }

    fn name(&self) -> &'static str {
        "CondBackward"
    }
}

/// Conditional subgraph execution.
///
/// Evaluates `true_fn(operands)` if `pred > 0.5`, otherwise `false_fn(operands)`.
/// Only the taken branch is executed — the other branch is never called.
///
/// # Arguments
///
/// - `pred` - A scalar tensor (0-D or single-element) treated as boolean.
///   Values > 0.5 are "true", <= 0.5 are "false".
/// - `true_fn` - Function to call when pred is true. Takes operands, returns
///   a vector of output tensors.
/// - `false_fn` - Function to call when pred is false. Same signature.
/// - `operands` - Input tensors passed to whichever branch executes.
///
/// # Returns
///
/// The vector of tensors returned by the executed branch.
///
/// # Errors
///
/// Returns `FerrotorchError::InvalidArgument` if:
/// - `pred` is not a scalar (more than 1 element)
/// - The two branches return different numbers of tensors
/// - The two branches return tensors with different shapes at corresponding positions
///
/// # Autograd
///
/// Gradients flow only through the taken branch. The untaken branch contributes
/// no gradients (it was never executed).
pub fn cond<T, TF, FF>(
    pred: &Tensor<T>,
    true_fn: TF,
    false_fn: FF,
    operands: &[Tensor<T>],
) -> FerrotorchResult<Vec<Tensor<T>>>
where
    T: Float,
    TF: FnOnce(&[Tensor<T>]) -> Vec<Tensor<T>>,
    FF: FnOnce(&[Tensor<T>]) -> Vec<Tensor<T>>,
{
    // Validate pred is scalar.
    if pred.numel() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "cond: pred must be a scalar tensor (1 element), got shape {:?} ({} elements)",
                pred.shape(),
                pred.numel()
            ),
        });
    }

    let pred_val = pred.data()?[0];
    let threshold = T::from(0.5).expect("Float trait guarantees from(0.5) succeeds");
    let take_true = pred_val > threshold;

    let branch_outputs = if take_true {
        true_fn(operands)
    } else {
        false_fn(operands)
    };

    // Check if any operand requires grad.
    let any_requires_grad = is_grad_enabled() && operands.iter().any(|op| op.requires_grad());

    if !any_requires_grad {
        return Ok(branch_outputs);
    }

    // Wrap each branch output with a CondBackward holding that output. The
    // autograd engine, on backward, will traverse into the output's own
    // grad_fn chain (whatever differentiable ops the branch composed) and
    // route gradients back to the operands correctly.
    let mut result = Vec::with_capacity(branch_outputs.len());
    for out in &branch_outputs {
        let data = out.data_vec()?;
        let shape = out.shape().to_vec();
        let grad_fn = Arc::new(CondBackward {
            branch_output: out.clone(),
        });
        let wrapped = Tensor::from_operation(TensorStorage::cpu(data), shape, grad_fn)?;
        result.push(wrapped);
    }

    Ok(result)
}

/// Validate that two sets of outputs have matching shapes.
///
/// This is a utility for users who want to verify at trace time that both
/// branches of a `cond` produce compatible outputs. Since `cond` only
/// executes one branch, it cannot validate shape agreement at runtime.
/// Call this function to eagerly validate both branches.
pub fn validate_cond_branches<T: Float>(
    true_outputs: &[Tensor<T>],
    false_outputs: &[Tensor<T>],
) -> FerrotorchResult<()> {
    if true_outputs.len() != false_outputs.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "cond: true branch returns {} tensors but false branch returns {}",
                true_outputs.len(),
                false_outputs.len()
            ),
        });
    }

    for (i, (t, f)) in true_outputs.iter().zip(false_outputs.iter()).enumerate() {
        if t.shape() != f.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "cond: output[{i}] shape mismatch: true branch {:?} vs false branch {:?}",
                    t.shape(),
                    f.shape()
                ),
            });
        }
    }

    Ok(())
}

// ===========================================================================
// scan — sequential state accumulation
// ===========================================================================

/// Backward node for one wrapped scan tensor — either a step output or the
/// final carry. Holds the single raw tensor whose grad_fn chain the autograd
/// engine should traverse.
///
/// Each wrapped output / final carry gets its own `ScanBackward` instance.
/// The held `target` tensor was produced by the user's step function (e.g.
/// `crate::grad_fns::arithmetic::add(carry, x)`) and so already carries the
/// correct grad chain. We just route the upstream gradient to it; the
/// engine's topo walk handles the rest, unrolling each step in reverse and
/// ultimately reaching `init_carry` and every `xs[i]`.
///
/// Earlier versions held `Vec<carries>`, `Vec<xs>`, `Vec<outputs>` plus an
/// `OutputKind { FinalCarry, StepOutput(usize) }` enum to disambiguate which
/// instance was which. The held tensor already encodes the role — the enum
/// and the Vec storage were both vestigial.
#[derive(Debug)]
struct ScanBackward<T: Float> {
    target: Tensor<T>,
}

impl<T: Float> GradFn<T> for ScanBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(grad_output.clone())])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.target]
    }

    fn name(&self) -> &'static str {
        "ScanBackward"
    }
}

/// Sequential state accumulation (scan / fold with outputs).
///
/// Iterates over `xs`, calling `fn_step(carry, x) -> (new_carry, output)` at
/// each step. Returns `(final_carry, [outputs...])`.
///
/// This is the core primitive for state-space models (Mamba, S4, RWKV) where
/// a hidden state is sequentially updated while producing an output at each
/// timestep.
///
/// # Arguments
///
/// - `fn_step` - The step function. Called once per element of `xs`. Takes
///   the current carry state and the current input, returns `(new_carry, output)`.
/// - `init` - Initial carry state tensor.
/// - `xs` - Sequence of input tensors to iterate over.
///
/// # Returns
///
/// `(final_carry, outputs)` where:
/// - `final_carry` is the carry state after processing all inputs
/// - `outputs` is a Vec of output tensors, one per step
///
/// # Autograd
///
/// Gradients flow backward through all steps in reverse order, following
/// the autograd graph that the step function built during the forward pass.
/// This is equivalent to backpropagation through time (BPTT).
pub fn scan<T, F>(
    fn_step: F,
    init: &Tensor<T>,
    xs: &[Tensor<T>],
) -> FerrotorchResult<(Tensor<T>, Vec<Tensor<T>>)>
where
    T: Float,
    F: Fn(&Tensor<T>, &Tensor<T>) -> (Tensor<T>, Tensor<T>),
{
    if xs.is_empty() {
        // No steps to execute — return init as final carry with empty outputs.
        return Ok((init.clone(), Vec::new()));
    }

    let mut outputs: Vec<Tensor<T>> = Vec::with_capacity(xs.len());
    let mut current_carry = init.clone();

    for x in xs {
        let (new_carry, output) = fn_step(&current_carry, x);
        outputs.push(output);
        current_carry = new_carry;
    }

    // Check if we need autograd. We don't need to inspect intermediate carries
    // here — if `init` or any `xs[i]` requires grad, the grad chain on each
    // step's outputs / new_carry will reflect that automatically (the step
    // function builds it).
    let any_requires_grad =
        is_grad_enabled() && (init.requires_grad() || xs.iter().any(|x| x.requires_grad()));

    if !any_requires_grad {
        return Ok((current_carry, outputs));
    }

    // Wrap the final carry: route the gradient to the *raw* final carry
    // (i.e. the new_carry produced by the last fn_step call), not to init.
    let final_carry_wrapped = Tensor::from_operation(
        TensorStorage::cpu(current_carry.data_vec()?),
        current_carry.shape().to_vec(),
        Arc::new(ScanBackward {
            target: current_carry.clone(),
        }),
    )?;

    // Wrap each step output: route the gradient to the raw step output. Its
    // grad_fn chain (from `fn_step`) routes back through each step.
    let mut wrapped_outputs = Vec::with_capacity(outputs.len());
    for out in &outputs {
        let wrapped = Tensor::from_operation(
            TensorStorage::cpu(out.data_vec()?),
            out.shape().to_vec(),
            Arc::new(ScanBackward {
                target: out.clone(),
            }),
        )?;
        wrapped_outputs.push(wrapped);
    }

    Ok((final_carry_wrapped, wrapped_outputs))
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::creation::{full, ones, zeros};

    // -----------------------------------------------------------------------
    // cond tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cond_true_branch() {
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![], false).unwrap();

        let x = ones::<f32>(&[3]).unwrap();

        let result = cond(
            &pred,
            |ops| {
                // True branch: multiply by 2
                let data = ops[0].data().unwrap();
                let doubled: Vec<f32> = data.iter().map(|&v| v * 2.0).collect();
                vec![
                    Tensor::from_storage(
                        TensorStorage::cpu(doubled),
                        ops[0].shape().to_vec(),
                        false,
                    )
                    .unwrap(),
                ]
            },
            |ops| {
                // False branch: multiply by 3
                let data = ops[0].data().unwrap();
                let tripled: Vec<f32> = data.iter().map(|&v| v * 3.0).collect();
                vec![
                    Tensor::from_storage(
                        TensorStorage::cpu(tripled),
                        ops[0].shape().to_vec(),
                        false,
                    )
                    .unwrap(),
                ]
            },
            &[x],
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        let data = result[0].data().unwrap();
        // True branch: 1.0 * 2.0 = 2.0
        assert_eq!(data, &[2.0, 2.0, 2.0]);
    }

    #[test]
    fn test_cond_false_branch() {
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0]), vec![], false).unwrap();

        let x = ones::<f32>(&[3]).unwrap();

        let result = cond(
            &pred,
            |ops| {
                let data = ops[0].data().unwrap();
                let doubled: Vec<f32> = data.iter().map(|&v| v * 2.0).collect();
                vec![
                    Tensor::from_storage(
                        TensorStorage::cpu(doubled),
                        ops[0].shape().to_vec(),
                        false,
                    )
                    .unwrap(),
                ]
            },
            |ops| {
                let data = ops[0].data().unwrap();
                let tripled: Vec<f32> = data.iter().map(|&v| v * 3.0).collect();
                vec![
                    Tensor::from_storage(
                        TensorStorage::cpu(tripled),
                        ops[0].shape().to_vec(),
                        false,
                    )
                    .unwrap(),
                ]
            },
            &[x],
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        let data = result[0].data().unwrap();
        // False branch: 1.0 * 3.0 = 3.0
        assert_eq!(data, &[3.0, 3.0, 3.0]);
    }

    #[test]
    fn test_cond_threshold_boundary() {
        // pred = 0.5 exactly => false branch (> 0.5 is true, not >=)
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.5]), vec![], false).unwrap();

        let x = ones::<f32>(&[2]).unwrap();

        let result = cond(
            &pred,
            |_| vec![full::<f32>(&[2], 10.0).unwrap()],
            |_| vec![full::<f32>(&[2], 20.0).unwrap()],
            &[x],
        )
        .unwrap();

        let data = result[0].data().unwrap();
        assert_eq!(data, &[20.0, 20.0]); // false branch
    }

    #[test]
    fn test_cond_just_above_threshold() {
        // pred = 0.51 => true branch
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.51]), vec![], false).unwrap();

        let x = ones::<f32>(&[2]).unwrap();

        let result = cond(
            &pred,
            |_| vec![full::<f32>(&[2], 10.0).unwrap()],
            |_| vec![full::<f32>(&[2], 20.0).unwrap()],
            &[x],
        )
        .unwrap();

        let data = result[0].data().unwrap();
        assert_eq!(data, &[10.0, 10.0]); // true branch
    }

    #[test]
    fn test_cond_non_scalar_pred_error() {
        let pred = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 0.0]), vec![2], false)
            .unwrap();

        let x = ones::<f32>(&[3]).unwrap();

        let result = cond(
            &pred,
            |_| vec![zeros::<f32>(&[3]).unwrap()],
            |_| vec![ones::<f32>(&[3]).unwrap()],
            &[x],
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_cond_multiple_outputs() {
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![], false).unwrap();

        let x = ones::<f32>(&[2]).unwrap();

        let result = cond(
            &pred,
            |ops| {
                let d = ops[0].data().unwrap();
                vec![
                    Tensor::from_storage(
                        TensorStorage::cpu(d.iter().map(|&v| v * 2.0).collect()),
                        ops[0].shape().to_vec(),
                        false,
                    )
                    .unwrap(),
                    Tensor::from_storage(
                        TensorStorage::cpu(d.iter().map(|&v| v * 3.0).collect()),
                        ops[0].shape().to_vec(),
                        false,
                    )
                    .unwrap(),
                ]
            },
            |_| vec![zeros::<f32>(&[2]).unwrap(), zeros::<f32>(&[2]).unwrap()],
            &[x],
        )
        .unwrap();

        assert_eq!(result.len(), 2);
        assert_eq!(result[0].data().unwrap(), &[2.0, 2.0]);
        assert_eq!(result[1].data().unwrap(), &[3.0, 3.0]);
    }

    #[test]
    fn test_cond_empty_operands() {
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![], false).unwrap();

        let result = cond(
            &pred,
            |_| vec![full::<f32>(&[3], 42.0).unwrap()],
            |_| vec![full::<f32>(&[3], 0.0).unwrap()],
            &[],
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        assert_eq!(result[0].data().unwrap(), &[42.0, 42.0, 42.0]);
    }

    #[test]
    fn test_cond_with_requires_grad() {
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![], false).unwrap();

        // Create operand with requires_grad=true.
        let x = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3], true)
            .unwrap();

        let result = cond(
            &pred,
            |ops| {
                let data = ops[0].data().unwrap();
                let doubled: Vec<f32> = data.iter().map(|&v| v * 2.0).collect();
                vec![
                    Tensor::from_storage(
                        TensorStorage::cpu(doubled),
                        ops[0].shape().to_vec(),
                        false,
                    )
                    .unwrap(),
                ]
            },
            |_| vec![zeros::<f32>(&[3]).unwrap()],
            &[x],
        )
        .unwrap();

        // Result should have grad_fn attached since operand requires grad.
        assert!(result[0].requires_grad());
        assert_eq!(result[0].data().unwrap(), &[2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_cond_scalar_pred_single_element() {
        // Shape [1] should also work as "scalar".
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![1], false).unwrap();

        let x = ones::<f32>(&[2]).unwrap();

        let result = cond(
            &pred,
            |_| vec![full::<f32>(&[2], 5.0).unwrap()],
            |_| vec![full::<f32>(&[2], 0.0).unwrap()],
            &[x],
        )
        .unwrap();

        assert_eq!(result[0].data().unwrap(), &[5.0, 5.0]);
    }

    #[test]
    fn test_validate_cond_branches_matching() {
        let a = vec![ones::<f32>(&[3, 4]).unwrap(), zeros::<f32>(&[2]).unwrap()];
        let b = vec![zeros::<f32>(&[3, 4]).unwrap(), ones::<f32>(&[2]).unwrap()];

        assert!(validate_cond_branches(&a, &b).is_ok());
    }

    #[test]
    fn test_validate_cond_branches_count_mismatch() {
        let a = vec![ones::<f32>(&[3]).unwrap()];
        let b = vec![ones::<f32>(&[3]).unwrap(), ones::<f32>(&[3]).unwrap()];

        assert!(validate_cond_branches(&a, &b).is_err());
    }

    #[test]
    fn test_validate_cond_branches_shape_mismatch() {
        let a = vec![ones::<f32>(&[3]).unwrap()];
        let b = vec![ones::<f32>(&[4]).unwrap()];

        assert!(validate_cond_branches(&a, &b).is_err());
    }

    // -----------------------------------------------------------------------
    // scan tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scan_empty_sequence() {
        let init = full::<f32>(&[2], 1.0).unwrap();
        let xs: &[Tensor<f32>] = &[];

        let (final_carry, outputs) =
            scan(|carry, _x| (carry.clone(), carry.clone()), &init, xs).unwrap();

        assert_eq!(final_carry.shape(), &[2]);
        assert_eq!(final_carry.data().unwrap(), &[1.0, 1.0]);
        assert!(outputs.is_empty());
    }

    #[test]
    fn test_scan_cumulative_sum() {
        // Scan that accumulates a running sum.
        let init = zeros::<f32>(&[1]).unwrap();
        let xs: Vec<Tensor<f32>> = vec![
            full::<f32>(&[1], 1.0).unwrap(),
            full::<f32>(&[1], 2.0).unwrap(),
            full::<f32>(&[1], 3.0).unwrap(),
        ];

        let (final_carry, outputs) = scan(
            |carry, x| {
                // new_carry = carry + x
                let c_data = carry.data().unwrap();
                let x_data = x.data().unwrap();
                let sum_val = c_data[0] + x_data[0];
                let new_carry =
                    Tensor::from_storage(TensorStorage::cpu(vec![sum_val]), vec![1], false)
                        .unwrap();
                let output = new_carry.clone();
                (new_carry, output)
            },
            &init,
            &xs,
        )
        .unwrap();

        // Final carry should be 1 + 2 + 3 = 6
        assert_eq!(final_carry.data().unwrap(), &[6.0]);

        // Outputs should be [1, 3, 6] (cumsum)
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].data().unwrap(), &[1.0]);
        assert_eq!(outputs[1].data().unwrap(), &[3.0]);
        assert_eq!(outputs[2].data().unwrap(), &[6.0]);
    }

    #[test]
    fn test_scan_single_step() {
        let init = full::<f32>(&[2], 10.0).unwrap();
        let xs = vec![full::<f32>(&[2], 5.0).unwrap()];

        let (final_carry, outputs) = scan(
            |carry, x| {
                let c = carry.data().unwrap();
                let xd = x.data().unwrap();
                let new_data: Vec<f32> = c.iter().zip(xd.iter()).map(|(&a, &b)| a + b).collect();
                let new_carry = Tensor::from_storage(
                    TensorStorage::cpu(new_data.clone()),
                    carry.shape().to_vec(),
                    false,
                )
                .unwrap();
                let output = Tensor::from_storage(
                    TensorStorage::cpu(new_data),
                    carry.shape().to_vec(),
                    false,
                )
                .unwrap();
                (new_carry, output)
            },
            &init,
            &xs,
        )
        .unwrap();

        assert_eq!(final_carry.data().unwrap(), &[15.0, 15.0]);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0].data().unwrap(), &[15.0, 15.0]);
    }

    #[test]
    fn test_scan_carry_shape_preserved() {
        let init = zeros::<f32>(&[3, 4]).unwrap();
        let xs = vec![ones::<f32>(&[3, 4]).unwrap(), ones::<f32>(&[3, 4]).unwrap()];

        let (final_carry, outputs) = scan(
            |carry, x| {
                let c = carry.data().unwrap();
                let xd = x.data().unwrap();
                let new_data: Vec<f32> = c.iter().zip(xd.iter()).map(|(&a, &b)| a + b).collect();
                let new_carry = Tensor::from_storage(
                    TensorStorage::cpu(new_data.clone()),
                    carry.shape().to_vec(),
                    false,
                )
                .unwrap();
                let output = Tensor::from_storage(
                    TensorStorage::cpu(new_data),
                    carry.shape().to_vec(),
                    false,
                )
                .unwrap();
                (new_carry, output)
            },
            &init,
            &xs,
        )
        .unwrap();

        assert_eq!(final_carry.shape(), &[3, 4]);
        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape(), &[3, 4]);
        assert_eq!(outputs[1].shape(), &[3, 4]);
    }

    #[test]
    fn test_scan_different_carry_and_output_shapes() {
        // Carry is [2], output is [1] — they can differ.
        let init = zeros::<f32>(&[2]).unwrap();
        let xs = vec![
            full::<f32>(&[2], 1.0).unwrap(),
            full::<f32>(&[2], 2.0).unwrap(),
        ];

        let (final_carry, outputs) = scan(
            |carry, x| {
                let c = carry.data().unwrap();
                let xd = x.data().unwrap();
                let new_data: Vec<f32> = c.iter().zip(xd.iter()).map(|(&a, &b)| a + b).collect();

                let new_carry = Tensor::from_storage(
                    TensorStorage::cpu(new_data.clone()),
                    carry.shape().to_vec(),
                    false,
                )
                .unwrap();

                // Output is the sum of carry elements (scalar).
                let sum: f32 = new_data.iter().sum();
                let output =
                    Tensor::from_storage(TensorStorage::cpu(vec![sum]), vec![1], false).unwrap();

                (new_carry, output)
            },
            &init,
            &xs,
        )
        .unwrap();

        assert_eq!(final_carry.shape(), &[2]);
        assert_eq!(final_carry.data().unwrap(), &[3.0, 3.0]); // [0+1+2, 0+1+2]

        assert_eq!(outputs.len(), 2);
        assert_eq!(outputs[0].shape(), &[1]);
        assert_eq!(outputs[0].data().unwrap(), &[2.0]); // 1+1
        assert_eq!(outputs[1].shape(), &[1]);
        assert_eq!(outputs[1].data().unwrap(), &[6.0]); // 3+3
    }

    #[test]
    fn test_scan_multiplicative_accumulation() {
        // Multiply carry by x at each step.
        let init = full::<f32>(&[1], 1.0).unwrap();
        let xs = vec![
            full::<f32>(&[1], 2.0).unwrap(),
            full::<f32>(&[1], 3.0).unwrap(),
            full::<f32>(&[1], 4.0).unwrap(),
        ];

        let (final_carry, outputs) = scan(
            |carry, x| {
                let c = carry.data().unwrap();
                let xd = x.data().unwrap();
                let product = c[0] * xd[0];
                let new_carry =
                    Tensor::from_storage(TensorStorage::cpu(vec![product]), vec![1], false)
                        .unwrap();
                let output = new_carry.clone();
                (new_carry, output)
            },
            &init,
            &xs,
        )
        .unwrap();

        // 1 * 2 * 3 * 4 = 24
        assert_eq!(final_carry.data().unwrap(), &[24.0]);
        assert_eq!(outputs.len(), 3);
        assert_eq!(outputs[0].data().unwrap(), &[2.0]);
        assert_eq!(outputs[1].data().unwrap(), &[6.0]);
        assert_eq!(outputs[2].data().unwrap(), &[24.0]);
    }

    #[test]
    fn test_scan_with_requires_grad() {
        let init =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0]), vec![1], true).unwrap();

        let xs = vec![
            full::<f32>(&[1], 1.0).unwrap(),
            full::<f32>(&[1], 2.0).unwrap(),
        ];

        let (final_carry, outputs) = scan(
            |carry, x| {
                let c = carry.data().unwrap();
                let xd = x.data().unwrap();
                let sum = c[0] + xd[0];
                let new_carry =
                    Tensor::from_storage(TensorStorage::cpu(vec![sum]), vec![1], false).unwrap();
                let output = new_carry.clone();
                (new_carry, output)
            },
            &init,
            &xs,
        )
        .unwrap();

        // With requires_grad init, results should track gradients.
        assert!(final_carry.requires_grad());
        assert_eq!(final_carry.data().unwrap(), &[3.0]);

        assert_eq!(outputs.len(), 2);
        assert!(outputs[0].requires_grad());
        assert!(outputs[1].requires_grad());
    }

    #[test]
    fn test_scan_ema_filter() {
        // Exponential moving average: carry = alpha * x + (1 - alpha) * carry
        let alpha = 0.3f32;
        let init = zeros::<f32>(&[1]).unwrap();
        let xs = vec![
            full::<f32>(&[1], 1.0).unwrap(),
            full::<f32>(&[1], 1.0).unwrap(),
            full::<f32>(&[1], 1.0).unwrap(),
            full::<f32>(&[1], 1.0).unwrap(),
        ];

        let (final_carry, outputs) = scan(
            move |carry, x| {
                let c = carry.data().unwrap();
                let xd = x.data().unwrap();
                let ema = alpha * xd[0] + (1.0 - alpha) * c[0];
                let new_carry =
                    Tensor::from_storage(TensorStorage::cpu(vec![ema]), vec![1], false).unwrap();
                let output = new_carry.clone();
                (new_carry, output)
            },
            &init,
            &xs,
        )
        .unwrap();

        // EMA of constant 1.0 with alpha=0.3:
        // step 0: 0.3*1 + 0.7*0 = 0.3
        // step 1: 0.3*1 + 0.7*0.3 = 0.51
        // step 2: 0.3*1 + 0.7*0.51 = 0.657
        // step 3: 0.3*1 + 0.7*0.657 = 0.7599
        assert_eq!(outputs.len(), 4);

        let eps = 1e-5;
        assert!((outputs[0].data().unwrap()[0] - 0.3).abs() < eps);
        assert!((outputs[1].data().unwrap()[0] - 0.51).abs() < eps);
        assert!((outputs[2].data().unwrap()[0] - 0.657).abs() < eps);
        assert!((outputs[3].data().unwrap()[0] - 0.7599).abs() < eps);
        assert!((final_carry.data().unwrap()[0] - 0.7599).abs() < eps);
    }

    // -----------------------------------------------------------------------
    // Gradient-value tests for cond and scan.
    //
    // The previous CondBackward / ScanBackward held operands / xs / carries
    // directly and returned identity grads, bypassing the branch / step
    // function's own grad chain. That happened to be correct for trivial
    // identity branches but produced wrong gradients for any real op
    // (e.g. multiplication: ∂(a*b)/∂a = b, not 1). These tests pin the
    // analytically-correct VJPs so a regression in routing fails immediately.
    // -----------------------------------------------------------------------

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!((a - e).abs() < tol, "index {i}: got {a}, expected {e}");
        }
    }

    #[test]
    fn test_cond_grad_flows_through_mul_branch() {
        // pred > 0.5 → mul branch. y = a * b, L = sum(y). dL/da = b, dL/db = a.
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0]), vec![], false).unwrap();
        let a =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![2.0, 3.0]), vec![2], true).unwrap();
        let b =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![5.0, 7.0]), vec![2], true).unwrap();

        let result = cond(
            &pred,
            |ops| vec![crate::grad_fns::arithmetic::mul(&ops[0], &ops[1]).unwrap()],
            |ops| vec![crate::grad_fns::arithmetic::add(&ops[0], &ops[1]).unwrap()],
            &[a.clone(), b.clone()],
        )
        .unwrap();

        let loss = crate::grad_fns::reduction::sum(&result[0]).unwrap();
        loss.backward().unwrap();

        let ga = a.grad().unwrap().expect("a should have grad");
        let gb = b.grad().unwrap().expect("b should have grad");
        assert_close(ga.data().unwrap(), &[5.0, 7.0], 1e-6); // grad a == b
        assert_close(gb.data().unwrap(), &[2.0, 3.0], 1e-6); // grad b == a
    }

    #[test]
    fn test_cond_false_branch_grad_flows_through_add() {
        // pred <= 0.5 → add branch. y = a + b, L = sum(y). dL/da = dL/db = 1.
        let pred =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0]), vec![], false).unwrap();
        let a =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![2.0, 3.0]), vec![2], true).unwrap();
        let b =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![5.0, 7.0]), vec![2], true).unwrap();

        let result = cond(
            &pred,
            |ops| vec![crate::grad_fns::arithmetic::mul(&ops[0], &ops[1]).unwrap()],
            |ops| vec![crate::grad_fns::arithmetic::add(&ops[0], &ops[1]).unwrap()],
            &[a.clone(), b.clone()],
        )
        .unwrap();

        let loss = crate::grad_fns::reduction::sum(&result[0]).unwrap();
        loss.backward().unwrap();

        let ga = a.grad().unwrap().expect("a should have grad");
        let gb = b.grad().unwrap().expect("b should have grad");
        assert_close(ga.data().unwrap(), &[1.0, 1.0], 1e-6);
        assert_close(gb.data().unwrap(), &[1.0, 1.0], 1e-6);
    }

    #[test]
    fn test_scan_grad_flows_through_step_function() {
        // step(carry, x) -> (carry + x, carry * x).
        // For init = 0, xs = [3, 5]:
        //   step 0: new_carry = 0 + 3 = 3, output = 0 * 3 = 0
        //   step 1: new_carry = 3 + 5 = 8, output = 3 * 5 = 15
        // L = sum(out0) + sum(out1) = 0 + 15 = 15.
        // dL/dx0 = x1 = 5, dL/dx1 = init + x0 = 3.
        let init =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0]), vec![1], true).unwrap();
        let x0 = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![3.0]), vec![1], true).unwrap();
        let x1 = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![5.0]), vec![1], true).unwrap();

        let (_final_carry, outputs) = scan(
            |carry, x| {
                let new_carry = crate::grad_fns::arithmetic::add(carry, x).unwrap();
                let output = crate::grad_fns::arithmetic::mul(carry, x).unwrap();
                (new_carry, output)
            },
            &init,
            &[x0.clone(), x1.clone()],
        )
        .unwrap();

        let s0 = crate::grad_fns::reduction::sum(&outputs[0]).unwrap();
        let s1 = crate::grad_fns::reduction::sum(&outputs[1]).unwrap();
        let loss = crate::grad_fns::arithmetic::add(&s0, &s1).unwrap();
        loss.backward().unwrap();

        let gx0 = x0.grad().unwrap().expect("x0 should have grad");
        let gx1 = x1.grad().unwrap().expect("x1 should have grad");
        assert_close(gx0.data().unwrap(), &[5.0], 1e-6);
        assert_close(gx1.data().unwrap(), &[3.0], 1e-6);
    }

    #[test]
    fn test_scan_grad_flows_through_final_carry() {
        // step(carry, x) -> (carry + x, detached).
        // final_carry = init + x0 + x1. L = sum(final_carry).
        // dL/d(init) = dL/dx0 = dL/dx1 = 1.
        let init =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0]), vec![1], true).unwrap();
        let x0 = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![2.0]), vec![1], true).unwrap();
        let x1 = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![3.0]), vec![1], true).unwrap();

        let (final_carry, _outputs) = scan(
            |carry, x| {
                let new_carry = crate::grad_fns::arithmetic::add(carry, x).unwrap();
                // Detached output isolates the carry path for this test.
                let output = Tensor::from_storage(
                    TensorStorage::cpu(new_carry.data().unwrap().to_vec()),
                    new_carry.shape().to_vec(),
                    false,
                )
                .unwrap();
                (new_carry, output)
            },
            &init,
            &[x0.clone(), x1.clone()],
        )
        .unwrap();

        let loss = crate::grad_fns::reduction::sum(&final_carry).unwrap();
        loss.backward().unwrap();

        let ginit = init.grad().unwrap().expect("init should have grad");
        let gx0 = x0.grad().unwrap().expect("x0 should have grad");
        let gx1 = x1.grad().unwrap().expect("x1 should have grad");
        assert_close(ginit.data().unwrap(), &[1.0], 1e-6);
        assert_close(gx0.data().unwrap(), &[1.0], 1e-6);
        assert_close(gx1.data().unwrap(), &[1.0], 1e-6);
    }
}
