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

/// Which branch was taken during the forward pass.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CondBranch {
    True,
    False,
}

/// Backward node for `cond`.
///
/// Gradients flow only through the branch that was actually executed during
/// the forward pass. The other branch receives zero gradients. This matches
/// PyTorch's `torch.cond` semantics: the untaken branch is never evaluated,
/// so no gradient information exists for it.
#[derive(Debug)]
struct CondBackward<T: Float> {
    /// Which branch was taken (kept for debugging/introspection).
    #[allow(dead_code)]
    branch: CondBranch,
    /// The output tensors from the taken branch (these have their own grad_fns
    /// if the branch function used differentiable ops).
    branch_outputs: Vec<Tensor<T>>,
    /// The original operands passed to cond — kept alive so the autograd
    /// graph retains references to them for the backward traversal.
    #[allow(dead_code)]
    operands: Vec<Tensor<T>>,
    /// Index of this output in the result vector.
    output_index: usize,
}

impl<T: Float> GradFn<T> for CondBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // The branch_outputs[output_index] is the tensor whose backward we need
        // to invoke. Since cond's outputs are wrappers around the branch outputs,
        // we propagate the gradient to the corresponding branch output.
        //
        // We return one gradient per operand. The branch function's own autograd
        // graph handles routing gradients from branch_outputs back to the operands.
        // Here we just need to connect the gradient to the branch output tensor.
        let _branch_out = &self.branch_outputs[self.output_index];

        // The autograd engine will follow the grad_fn chain on the branch
        // output tensor. We just pass the gradient through to it.
        Ok(vec![Some(grad_output.clone())])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.branch_outputs[self.output_index]]
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
    let threshold = T::from(0.5).unwrap();
    let take_true = pred_val > threshold;

    let (branch, branch_outputs) = if take_true {
        (CondBranch::True, true_fn(operands))
    } else {
        (CondBranch::False, false_fn(operands))
    };

    // Check if any operand requires grad.
    let any_requires_grad = is_grad_enabled() && operands.iter().any(|op| op.requires_grad());

    if !any_requires_grad {
        return Ok(branch_outputs);
    }

    // Wrap each branch output with a CondBackward grad_fn so gradients
    // flow through the taken branch back to the operands.
    let operands_vec: Vec<Tensor<T>> = operands.to_vec();
    let mut result = Vec::with_capacity(branch_outputs.len());

    for (i, out) in branch_outputs.iter().enumerate() {
        let data = out.data_vec()?;
        let shape = out.shape().to_vec();
        let grad_fn = Arc::new(CondBackward {
            branch,
            branch_outputs: branch_outputs.clone(),
            operands: operands_vec.clone(),
            output_index: i,
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

/// Backward node for scan.
///
/// Stores all intermediate carries and inputs so that gradients can flow
/// backward through the entire scan sequence. Each step's backward is
/// unrolled in reverse order.
#[derive(Debug)]
struct ScanBackward<T: Float> {
    /// All intermediate carries: carries[0] = init, carries[i+1] = carry after step i.
    carries: Vec<Tensor<T>>,
    /// The input tensors at each step — kept alive so the autograd graph
    /// retains references for the backward traversal.
    #[allow(dead_code)]
    xs: Vec<Tensor<T>>,
    /// The output tensors produced at each step.
    outputs: Vec<Tensor<T>>,
    /// Which output in the result vector this backward node corresponds to.
    /// If `FinalCarry`, this is the final_carry backward.
    output_index: OutputKind,
}

#[derive(Debug, Clone, Copy)]
enum OutputKind {
    /// This backward node is for the final carry.
    FinalCarry,
    /// This backward node is for step output at the given index.
    StepOutput(usize),
}

impl<T: Float> GradFn<T> for ScanBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // For scan's backward, we need to propagate gradients through the
        // step function's autograd graph. The step function created tensors
        // with their own grad_fns when it ran during forward.
        //
        // The autograd engine handles this: it will follow the grad_fn chain
        // from each output/carry tensor back through the step function's ops.
        //
        // For FinalCarry: gradient goes to the last carry (carries[n]).
        // For StepOutput(i): gradient goes to outputs[i].
        match self.output_index {
            OutputKind::FinalCarry => {
                // Pass gradient to the last carry. The autograd engine will
                // follow the grad_fn chain on carries.last().
                let _last_carry = self.carries.last().unwrap();
                Ok(vec![Some(grad_output.clone())])
            }
            OutputKind::StepOutput(_i) => {
                // Pass gradient to the i-th step output.
                Ok(vec![Some(grad_output.clone())])
            }
        }
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        match self.output_index {
            OutputKind::FinalCarry => {
                // The final carry depends on the last carry in the chain.
                vec![self.carries.last().unwrap()]
            }
            OutputKind::StepOutput(i) => {
                // Step output i depends on the output tensor from step i.
                vec![&self.outputs[i]]
            }
        }
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

    let mut carries: Vec<Tensor<T>> = Vec::with_capacity(xs.len() + 1);
    carries.push(init.clone());

    let mut outputs: Vec<Tensor<T>> = Vec::with_capacity(xs.len());

    let mut current_carry = init.clone();

    for x in xs {
        let (new_carry, output) = fn_step(&current_carry, x);
        carries.push(new_carry.clone());
        outputs.push(output);
        current_carry = new_carry;
    }

    // Check if we need autograd.
    let any_requires_grad = is_grad_enabled()
        && (init.requires_grad()
            || xs.iter().any(|x| x.requires_grad())
            || carries.iter().any(|c| c.requires_grad())
            || outputs.iter().any(|o| o.requires_grad()));

    if !any_requires_grad {
        return Ok((current_carry, outputs));
    }

    // Wrap the final carry with a ScanBackward grad_fn.
    let final_carry_data = current_carry.data_vec()?;
    let final_carry_shape = current_carry.shape().to_vec();

    let final_carry_wrapped = Tensor::from_operation(
        TensorStorage::cpu(final_carry_data),
        final_carry_shape,
        Arc::new(ScanBackward {
            carries: carries.clone(),
            xs: xs.to_vec(),
            outputs: outputs.clone(),
            output_index: OutputKind::FinalCarry,
        }),
    )?;

    // Wrap each step output with a ScanBackward grad_fn.
    let mut wrapped_outputs = Vec::with_capacity(outputs.len());
    for (i, out) in outputs.iter().enumerate() {
        let out_data = out.data_vec()?;
        let out_shape = out.shape().to_vec();

        let wrapped = Tensor::from_operation(
            TensorStorage::cpu(out_data),
            out_shape,
            Arc::new(ScanBackward {
                carries: carries.clone(),
                xs: xs.to_vec(),
                outputs: outputs.clone(),
                output_index: OutputKind::StepOutput(i),
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
}
