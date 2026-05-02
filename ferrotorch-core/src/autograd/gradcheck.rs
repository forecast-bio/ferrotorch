//! Numerical gradient checking utilities.
//!
//! [`gradcheck`] verifies that the analytical gradients computed by autograd
//! match finite-difference numerical gradients. This is essential for testing
//! custom backward implementations.
//!
//! Matches PyTorch's `torch.autograd.gradcheck` API.

use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::storage::TensorStorage;
use crate::tensor::Tensor;

/// Check analytical gradients against numerical (finite-difference) gradients.
///
/// `func` takes a slice of input tensors and returns a scalar output.
/// `inputs` are the tensors to check gradients for (must require grad).
/// `eps` is the finite-difference step size (default: 1e-6).
/// `atol` is the absolute tolerance for comparison (default: 1e-5).
/// `rtol` is the relative tolerance for comparison (default: 1e-3).
///
/// Returns `Ok(true)` if all gradients match, `Err` with a descriptive
/// message if any gradient mismatches are found.
///
/// # Example
///
/// ```ignore
/// use ferrotorch_core::autograd::gradcheck::gradcheck;
///
/// let x = Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]), vec![3], true)?;
/// let result = gradcheck(
///     |inputs| {
///         let x = &inputs[0];
///         // sum(x^2)
///         let x2 = ferrotorch_core::ops::elementwise::unary_map(x, |v| v * v)?;
///         ferrotorch_core::grad_fns::reduction::sum(&x2)
///     },
///     &[x],
///     None, None, None,
/// )?;
/// assert!(result);
/// ```
pub fn gradcheck<T, F>(
    func: F,
    inputs: &[Tensor<T>],
    eps: Option<f64>,
    atol: Option<f64>,
    rtol: Option<f64>,
) -> FerrotorchResult<bool>
where
    T: Float,
    F: Fn(&[Tensor<T>]) -> FerrotorchResult<Tensor<T>>,
{
    // Default eps is larger for f32 to avoid cancellation in finite differences.
    let default_eps = if std::mem::size_of::<T>() <= 4 {
        1e-3
    } else {
        1e-6
    };
    let default_atol = if std::mem::size_of::<T>() <= 4 {
        1e-3
    } else {
        1e-5
    };
    let default_rtol = if std::mem::size_of::<T>() <= 4 {
        1e-2
    } else {
        1e-3
    };
    let eps = eps.unwrap_or(default_eps);
    let atol = atol.unwrap_or(default_atol);
    let rtol = rtol.unwrap_or(default_rtol);

    let eps_t = T::from(eps).unwrap();

    // Step 1: Compute analytical gradients via autograd.
    let output = func(inputs)?;
    if output.numel() != 1 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "gradcheck: function must return a scalar, got shape {:?}",
                output.shape()
            ),
        });
    }
    output.backward()?;

    // Step 2: For each input, compare analytical grad with numerical.
    for (input_idx, input) in inputs.iter().enumerate() {
        let analytical_grad = match input.grad()? {
            Some(g) => g,
            None => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "gradcheck: input {input_idx} has no gradient (requires_grad=false?)"
                    ),
                });
            }
        };
        let analytical_data = analytical_grad.data_vec()?;

        let input_data = input.data_vec()?;
        let n = input_data.len();

        // Compute numerical gradient via central difference.
        for elem_idx in 0..n {
            // f(x + eps)
            let mut perturbed_plus = input_data.clone();
            perturbed_plus[elem_idx] += eps_t;

            // f(x - eps)
            let mut perturbed_minus = input_data.clone();
            perturbed_minus[elem_idx] = perturbed_minus[elem_idx] - eps_t;

            // Create perturbed tensors (no grad needed).
            let plus_tensor = Tensor::from_storage(
                TensorStorage::cpu(perturbed_plus),
                input.shape().to_vec(),
                false,
            )?;
            let minus_tensor = Tensor::from_storage(
                TensorStorage::cpu(perturbed_minus),
                input.shape().to_vec(),
                false,
            )?;

            // Build input slices with the perturbed tensor replacing this input.
            let mut plus_inputs: Vec<Tensor<T>> = Vec::with_capacity(inputs.len());
            let mut minus_inputs: Vec<Tensor<T>> = Vec::with_capacity(inputs.len());
            for (i, inp) in inputs.iter().enumerate() {
                if i == input_idx {
                    plus_inputs.push(plus_tensor.clone());
                    minus_inputs.push(minus_tensor.clone());
                } else {
                    // Use a detached copy.
                    let data = inp.data_vec()?;
                    let t = Tensor::from_storage(
                        TensorStorage::cpu(data),
                        inp.shape().to_vec(),
                        false,
                    )?;
                    plus_inputs.push(t.clone());
                    minus_inputs.push(t);
                }
            }

            let f_plus = func(&plus_inputs)?;
            let f_minus = func(&minus_inputs)?;

            let f_plus_val = f_plus.data_vec()?[0];
            let f_minus_val = f_minus.data_vec()?[0];

            // Numerical gradient: (f(x+eps) - f(x-eps)) / (2*eps)
            let two_eps = T::from(2.0 * eps).unwrap();
            let numerical = (f_plus_val - f_minus_val) / two_eps;
            let analytical = analytical_data[elem_idx];

            // Check closeness: |a - n| <= atol + rtol * |n|
            let diff = if analytical > numerical {
                analytical - numerical
            } else {
                numerical - analytical
            };
            let zero = <T as num_traits::Zero>::zero();
            let abs_numerical = if numerical < zero {
                zero - numerical
            } else {
                numerical
            };
            let tolerance = T::from(atol).unwrap() + T::from(rtol).unwrap() * abs_numerical;

            if diff > tolerance {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "gradcheck failed at input {input_idx}, element {elem_idx}: \
                         analytical={analytical:?}, numerical={numerical:?}, diff={diff:?}, tol={tolerance:?}"
                    ),
                });
            }
        }
    }

    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grad_fns::{arithmetic, reduction};

    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    #[test]
    fn test_gradcheck_sum_of_squares() {
        let x = leaf(&[1.0, 2.0, 3.0], &[3]);
        let result = gradcheck(
            |inputs| {
                // sum(x * x) — uses autograd-aware mul
                let x2 = arithmetic::mul(&inputs[0], &inputs[0])?;
                reduction::sum(&x2)
            },
            &[x],
            None,
            None,
            None,
        );
        assert!(result.is_ok(), "gradcheck failed: {:?}", result.err());
        assert!(result.unwrap());
    }

    #[test]
    fn test_gradcheck_linear_combination() {
        let a = leaf(&[2.0, 3.0], &[2]);
        let b = leaf(&[4.0, 5.0], &[2]);
        let result = gradcheck(
            |inputs| {
                let prod = arithmetic::mul(&inputs[0], &inputs[1])?;
                reduction::sum(&prod)
            },
            &[a, b],
            None,
            None,
            None,
        );
        assert!(result.is_ok(), "gradcheck failed: {:?}", result.err());
    }

    #[test]
    fn test_gradcheck_add() {
        let a = leaf(&[1.0, 2.0, 3.0], &[3]);
        let b = leaf(&[4.0, 5.0, 6.0], &[3]);
        let result = gradcheck(
            |inputs| {
                let s = arithmetic::add(&inputs[0], &inputs[1])?;
                reduction::sum(&s)
            },
            &[a, b],
            None,
            None,
            None,
        );
        assert!(result.is_ok(), "gradcheck failed: {:?}", result.err());
    }

    #[test]
    fn test_gradcheck_non_scalar_fails() {
        let x = leaf(&[1.0, 2.0], &[2]);
        let result = gradcheck(|inputs| Ok(inputs[0].clone()), &[x], None, None, None);
        assert!(result.is_err());
    }
}
