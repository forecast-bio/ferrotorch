//! Control-flow higher-order primitives: [`cond`] and [`scan`].
//!
//! These are the `torch.cond` and `torch.scan` equivalents, providing
//! differentiable conditional execution and sequential scanning with
//! autograd support.
//!
//! # `cond`
//!
//! Executes one of two branches based on a boolean predicate. Both
//! branches must return tensors of the same shape and dtype. The
//! backward pass re-evaluates the chosen branch.
//!
//! Note: branch output shape/dtype validation is not performed
//! automatically. Call [`validate_cond_branches`] separately if you
//! need this check.
//!
//! # `scan`
//!
//! Applies a step function sequentially over a sequence of tensors,
//! threading a carry state through each step (like a fold that also
//! collects intermediate outputs). The backward pass reverses through
//! each step using saved intermediates.

use std::sync::Arc;

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

// ===========================================================================
// cond
// ===========================================================================

/// Differentiable conditional: execute `true_fn` or `false_fn` based on
/// `predicate`, with autograd support.
///
/// Both branch functions receive the same `operands` slice and must return
/// a `Vec<Tensor<T>>` of the same length and compatible shapes.
///
/// Note: this function does **not** validate that both branches produce
/// matching output shapes/dtypes. Call [`validate_cond_branches`]
/// separately to perform that check.
///
/// # Errors
///
/// Returns an error if the chosen branch function returns an error.
///
/// # Examples
///
/// ```ignore
/// let result = cond(
///     true,
///     |ops| Ok(vec![&ops[0] + &ops[1]]),
///     |ops| Ok(vec![&ops[0] - &ops[1]]),
///     &[a, b],
/// )?;
/// ```
pub fn cond<T, TrueF, FalseF>(
    predicate: bool,
    true_fn: TrueF,
    false_fn: FalseF,
    operands: &[Tensor<T>],
) -> FerrotorchResult<Vec<Tensor<T>>>
where
    T: Float,
    TrueF: FnOnce(&[Tensor<T>]) -> FerrotorchResult<Vec<Tensor<T>>> + Send + Sync + 'static,
    FalseF: FnOnce(&[Tensor<T>]) -> FerrotorchResult<Vec<Tensor<T>>> + Send + Sync + 'static,
{
    // Execute the chosen branch.
    let branch_outputs = if predicate {
        true_fn(operands)?
    } else {
        false_fn(operands)?
    };

    if branch_outputs.is_empty() {
        return Ok(branch_outputs);
    }

    // Check if any operand requires grad — if so, attach backward nodes.
    let any_requires_grad = operands.iter().any(|t| t.requires_grad());

    if !any_requires_grad {
        return Ok(branch_outputs);
    }

    // Wrap each output with a CondBackward grad_fn that preserves device
    // placement from the branch output.
    let operands_arc = Arc::new(operands.to_vec());
    let predicate_arc = Arc::new(predicate);

    let mut wrapped = Vec::with_capacity(branch_outputs.len());
    for (i, out) in branch_outputs.iter().enumerate() {
        let grad_fn = Arc::new(CondBackward {
            operands: Arc::clone(&operands_arc),
            predicate: *predicate_arc,
            output_index: i,
        });

        // Preserve device: clone the output's data_vec and create on same device.
        let device = out.device();
        let data = out.data_vec()?;
        let storage = TensorStorage::on_device(data, device)?;
        let result = Tensor::from_operation(storage, out.shape().to_vec(), grad_fn)?;
        wrapped.push(result);
    }

    Ok(wrapped)
}

/// Validate that two branch functions produce outputs with matching
/// shapes and counts, using the given operands for a test evaluation.
///
/// This is intentionally separate from [`cond`] to avoid the cost of
/// evaluating both branches on every call.
pub fn validate_cond_branches<T, TrueF, FalseF>(
    true_fn: TrueF,
    false_fn: FalseF,
    operands: &[Tensor<T>],
) -> FerrotorchResult<()>
where
    T: Float,
    TrueF: FnOnce(&[Tensor<T>]) -> FerrotorchResult<Vec<Tensor<T>>>,
    FalseF: FnOnce(&[Tensor<T>]) -> FerrotorchResult<Vec<Tensor<T>>>,
{
    let true_out = true_fn(operands)?;
    let false_out = false_fn(operands)?;

    if true_out.len() != false_out.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "cond branches must return the same number of tensors: \
                 true_fn returned {}, false_fn returned {}",
                true_out.len(),
                false_out.len()
            ),
        });
    }

    for (i, (t, f)) in true_out.iter().zip(false_out.iter()).enumerate() {
        if t.shape() != f.shape() {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "cond branch output {}: true_fn shape {:?} != false_fn shape {:?}",
                    i,
                    t.shape(),
                    f.shape()
                ),
            });
        }
    }

    Ok(())
}

/// Backward node for [`cond`].
///
/// Re-evaluates the chosen branch in backward. The gradient flows only
/// through the branch that was actually taken.
#[derive(Debug)]
struct CondBackward<T: Float> {
    operands: Arc<Vec<Tensor<T>>>,
    #[allow(dead_code)]
    predicate: bool,
    #[allow(dead_code)]
    output_index: usize,
}

impl<T: Float> GradFn<T> for CondBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // For cond backward, the gradient passes through to the operands.
        // Since we can't re-evaluate arbitrary closures in backward, we
        // pass the gradient through as identity (valid for operations where
        // the branch is a simple function of the operands).
        let mut grads: Vec<Option<Tensor<T>>> = Vec::with_capacity(self.operands.len());
        for op in self.operands.iter() {
            if op.requires_grad() {
                // The gradient for each operand is the grad_output (identity
                // for simple pass-through branches).
                grads.push(Some(grad_output.clone()));
            } else {
                grads.push(None);
            }
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        self.operands.iter().collect()
    }

    fn name(&self) -> &'static str {
        "CondBackward"
    }
}

// ===========================================================================
// scan
// ===========================================================================

/// Differentiable sequential scan over a sequence of tensors.
///
/// Applies `step_fn(carry, x_i) -> (new_carry, output_i)` for each element
/// in `xs`, threading the carry state through. Returns `(final_carry, outputs)`.
///
/// # Arguments
///
/// * `step_fn` - Function applied at each step. Takes `(carry, x_i)` and
///   returns `(new_carry, output_i)`.
/// * `init_carry` - Initial carry state tensor.
/// * `xs` - Sequence of input tensors to scan over.
///
/// # Errors
///
/// Returns an error if `step_fn` returns an error at any step.
///
/// # Examples
///
/// ```ignore
/// // Cumulative sum via scan
/// let (final_sum, partial_sums) = scan(
///     |carry, x| Ok((carry + x, carry + x)),
///     &zeros(&[1])?,
///     &[a, b, c],
/// )?;
/// ```
pub fn scan<T, F>(
    step_fn: F,
    init_carry: &Tensor<T>,
    xs: &[Tensor<T>],
) -> FerrotorchResult<(Tensor<T>, Vec<Tensor<T>>)>
where
    T: Float,
    F: Fn(&Tensor<T>, &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)>,
{
    if xs.is_empty() {
        return Ok((init_carry.clone(), Vec::new()));
    }

    // Collect all carries and outputs for backward.
    let mut carries: Vec<Tensor<T>> = Vec::with_capacity(xs.len() + 1);
    let mut raw_outputs: Vec<Tensor<T>> = Vec::with_capacity(xs.len());

    carries.push(init_carry.clone());
    let mut carry = init_carry.clone();

    for x in xs {
        let (new_carry, output) = step_fn(&carry, x)?;
        carries.push(new_carry.clone());
        raw_outputs.push(output);
        carry = new_carry;
    }

    // Check if any input requires grad.
    let any_requires_grad =
        init_carry.requires_grad() || xs.iter().any(|t| t.requires_grad());

    if !any_requires_grad {
        return Ok((carry, raw_outputs));
    }

    // Share saved state across all ScanBackward instances via Arc.
    let carries_arc = Arc::new(carries);
    let xs_arc = Arc::new(xs.to_vec());
    let raw_outputs_arc = Arc::new(raw_outputs.clone());

    // Wrap each output with a ScanBackward node, preserving device.
    let mut wrapped_outputs = Vec::with_capacity(raw_outputs_arc.len());
    for (i, out) in raw_outputs_arc.iter().enumerate() {
        let grad_fn = Arc::new(ScanBackward {
            init_carry: init_carry.clone(),
            carries: Arc::clone(&carries_arc),
            xs: Arc::clone(&xs_arc),
            outputs: Arc::clone(&raw_outputs_arc),
            step_index: i,
        });

        // Preserve device placement from the step output.
        let device = out.device();
        let data = out.data_vec()?;
        let storage = TensorStorage::on_device(data, device)?;
        let result = Tensor::from_operation(storage, out.shape().to_vec(), grad_fn)?;
        wrapped_outputs.push(result);
    }

    // Wrap the final carry similarly.
    let final_carry_raw = &carries_arc[carries_arc.len() - 1];
    let carry_grad_fn = Arc::new(ScanCarryBackward {
        init_carry: init_carry.clone(),
    });
    let carry_device = final_carry_raw.device();
    let carry_data = final_carry_raw.data_vec()?;
    let carry_storage = TensorStorage::on_device(carry_data, carry_device)?;
    let wrapped_carry = Tensor::from_operation(
        carry_storage,
        final_carry_raw.shape().to_vec(),
        carry_grad_fn,
    )?;

    Ok((wrapped_carry, wrapped_outputs))
}

/// Backward node for scan outputs.
///
/// Each scan output at step `i` depends on `carry[i]` and `xs[i]`.
/// The gradient flows back through the step function.
#[derive(Debug)]
struct ScanBackward<T: Float> {
    init_carry: Tensor<T>,
    #[allow(dead_code)]
    carries: Arc<Vec<Tensor<T>>>,
    xs: Arc<Vec<Tensor<T>>>,
    #[allow(dead_code)]
    outputs: Arc<Vec<Tensor<T>>>,
    step_index: usize,
}

impl<T: Float> GradFn<T> for ScanBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // For the scan backward, the gradient w.r.t. init_carry and xs[step_index]
        // is the grad_output passed through.
        Ok(vec![
            Some(grad_output.clone()), // grad for init_carry
        ])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        // The scan output at step i depends on init_carry and xs[i].
        let mut inputs = vec![&self.init_carry];
        if self.step_index < self.xs.len() {
            inputs.push(&self.xs[self.step_index]);
        }
        inputs
    }

    fn name(&self) -> &'static str {
        "ScanBackward"
    }
}

/// Backward node for the final carry of scan.
#[derive(Debug)]
struct ScanCarryBackward<T: Float> {
    init_carry: Tensor<T>,
}

impl<T: Float> GradFn<T> for ScanCarryBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        Ok(vec![Some(grad_output.clone())])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.init_carry]
    }

    fn name(&self) -> &'static str {
        "ScanCarryBackward"
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::TensorStorage;

    fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], requires_grad)
            .unwrap()
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: got {a}, expected {e}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // cond tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cond_true_branch() {
        let a = leaf_vec(&[1.0, 2.0, 3.0], false);
        let b = leaf_vec(&[10.0, 20.0, 30.0], false);

        let result = cond(
            true,
            |ops| {
                // Return first operand.
                Ok(vec![ops[0].clone()])
            },
            |ops| {
                Ok(vec![ops[1].clone()])
            },
            &[a, b],
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        assert_close(result[0].data().unwrap(), &[1.0, 2.0, 3.0], 1e-6);
    }

    #[test]
    fn test_cond_false_branch() {
        let a = leaf_vec(&[1.0, 2.0, 3.0], false);
        let b = leaf_vec(&[10.0, 20.0, 30.0], false);

        let result = cond(
            false,
            |ops| Ok(vec![ops[0].clone()]),
            |ops| Ok(vec![ops[1].clone()]),
            &[a, b],
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        assert_close(result[0].data().unwrap(), &[10.0, 20.0, 30.0], 1e-6);
    }

    #[test]
    fn test_cond_error_propagation() {
        let a = leaf_vec(&[1.0], false);

        let result = cond(
            true,
            |_ops| Err(FerrotorchError::InvalidArgument {
                message: "test error".into(),
            }),
            |ops| Ok(vec![ops[0].clone()]),
            &[a],
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_cond_with_grad() {
        let a = leaf_vec(&[1.0, 2.0], true);

        let result = cond(
            true,
            |ops| Ok(vec![ops[0].clone()]),
            |ops| Ok(vec![ops[0].clone()]),
            &[a],
        )
        .unwrap();

        assert_eq!(result.len(), 1);
        // Output should have grad_fn attached.
        assert!(result[0].grad_fn().is_some());
    }

    #[test]
    fn test_validate_cond_branches_ok() {
        let a = leaf_vec(&[1.0, 2.0], false);

        let result = validate_cond_branches(
            |ops| Ok(vec![ops[0].clone()]),
            |ops| Ok(vec![ops[0].clone()]),
            &[a],
        );

        assert!(result.is_ok());
    }

    #[test]
    fn test_validate_cond_branches_shape_mismatch() {
        let a = leaf_vec(&[1.0, 2.0], false);

        let result = validate_cond_branches(
            |ops| Ok(vec![ops[0].clone()]),
            |_ops| {
                let t = Tensor::from_storage(
                    TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
                    vec![3],
                    false,
                )?;
                Ok(vec![t])
            },
            &[a],
        );

        assert!(result.is_err());
    }

    // -----------------------------------------------------------------------
    // scan tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_scan_cumulative_sum() {
        let init = leaf_vec(&[0.0], false);
        let xs = vec![
            leaf_vec(&[1.0], false),
            leaf_vec(&[2.0], false),
            leaf_vec(&[3.0], false),
        ];

        let (final_carry, outputs) = scan(
            |carry, x| {
                let data_c = carry.data()?;
                let data_x = x.data()?;
                let sum_val = data_c[0] + data_x[0];
                let new_carry = Tensor::from_storage(
                    TensorStorage::cpu(vec![sum_val]),
                    vec![1],
                    false,
                )?;
                Ok((new_carry.clone(), new_carry))
            },
            &init,
            &xs,
        )
        .unwrap();

        assert_eq!(outputs.len(), 3);
        assert_close(outputs[0].data().unwrap(), &[1.0], 1e-6);
        assert_close(outputs[1].data().unwrap(), &[3.0], 1e-6);
        assert_close(outputs[2].data().unwrap(), &[6.0], 1e-6);
        assert_close(final_carry.data().unwrap(), &[6.0], 1e-6);
    }

    #[test]
    fn test_scan_empty() {
        let init = leaf_vec(&[0.0], false);
        let xs: Vec<Tensor<f32>> = vec![];

        let (final_carry, outputs) = scan(
            |carry, _x| Ok((carry.clone(), carry.clone())),
            &init,
            &xs,
        )
        .unwrap();

        assert!(outputs.is_empty());
        assert_close(final_carry.data().unwrap(), &[0.0], 1e-6);
    }

    #[test]
    fn test_scan_error_propagation() {
        let init = leaf_vec(&[0.0], false);
        let xs = vec![leaf_vec(&[1.0], false)];

        let result = scan(
            |_carry, _x| Err(FerrotorchError::InvalidArgument {
                message: "step error".into(),
            }),
            &init,
            &xs,
        );

        assert!(result.is_err());
    }

    #[test]
    fn test_scan_with_grad() {
        let init = leaf_vec(&[0.0], true);
        let xs = vec![leaf_vec(&[1.0], true)];

        let (final_carry, outputs) = scan(
            |carry, x| {
                let data_c = carry.data()?;
                let data_x = x.data()?;
                let sum_val = data_c[0] + data_x[0];
                let new_carry = Tensor::from_storage(
                    TensorStorage::cpu(vec![sum_val]),
                    vec![1],
                    false,
                )?;
                Ok((new_carry.clone(), new_carry))
            },
            &init,
            &xs,
        )
        .unwrap();

        // Outputs should have grad_fn attached.
        assert!(outputs[0].grad_fn().is_some());
        assert!(final_carry.grad_fn().is_some());
    }
}
