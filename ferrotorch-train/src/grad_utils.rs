//! Gradient clipping utilities.
//!
//! These functions mirror PyTorch's `torch.nn.utils.clip_grad_norm_` and
//! `torch.nn.utils.clip_grad_value_`:
//!
//! | Function | Description |
//! |----------|-------------|
//! | [`clip_grad_norm_`] | Clip total gradient norm across all parameters |
//! | [`clip_grad_value_`] | Clamp each gradient element to `[-clip_value, clip_value]` |
//!
//! [CL-334] Add gradient checkpointing, autocast context, gradient clipping, and EMA callback

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::Parameter;

/// Clip the total gradient norm of an iterable of parameters.
///
/// The gradients are modified **in-place**. The norm is computed over all
/// gradients concatenated, as if they were a single vector.
///
/// # Arguments
///
/// * `params` - An iterable of parameters whose `.grad()` will be clipped.
/// * `max_norm` - Maximum allowed norm value.
/// * `norm_type` - Type of the p-norm used. Common values:
///   - `2.0` for L2 norm (Euclidean, the default in PyTorch)
///   - `1.0` for L1 norm
///   - `f64::INFINITY` for max norm
///
/// # Returns
///
/// The total (unclipped) norm of all gradients as a scalar `f64`.
///
/// # Algorithm
///
/// 1. Compute the total p-norm of all parameter gradients concatenated.
/// 2. Compute `clip_coef = max_norm / (total_norm + 1e-6)`.
/// 3. If `clip_coef < 1`, multiply every gradient element by `clip_coef`.
///
/// This matches PyTorch's `torch.nn.utils.clip_grad_norm_` behaviour exactly.
///
/// # Errors
///
/// Returns `FerrotorchError::InvalidArgument` when `max_norm < 0.0` or
/// `norm_type <= 0.0`. Also propagates any `FerrotorchError` from the
/// underlying tensor reads / writes (`grad()`, `data_vec()`, `set_grad()`).
/// May additionally return `FerrotorchError::InvalidArgument` from the
/// numeric `cast` helper when a gradient value (e.g. an `f16`/`bf16` NaN
/// or out-of-range value) cannot be represented as `f64`, or when the
/// computed `clip_coef` cannot be converted back to `T`.
///
/// # Examples
///
/// ```ignore
/// use ferrotorch_train::clip_grad_norm_;
///
/// let total_norm = clip_grad_norm_(&model.parameters(), 1.0, 2.0)?;
/// println!("Total gradient norm: {total_norm}");
/// ```
pub fn clip_grad_norm_<T: Float>(
    params: &[&Parameter<T>],
    max_norm: f64,
    norm_type: f64,
) -> FerrotorchResult<f64> {
    if max_norm < 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("max_norm must be non-negative, got {max_norm}"),
        });
    }
    if norm_type <= 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("norm_type must be positive, got {norm_type}"),
        });
    }

    // Collect all gradient data into a flat list of f64 values for norm computation.
    // This matches PyTorch's approach of treating all grads as a single vector.
    let mut grad_tensors: Vec<(Tensor<T>, Vec<T>)> = Vec::new();

    for param in params {
        if let Some(grad) = param.grad()? {
            let data = grad.data_vec()?;
            grad_tensors.push((grad, data));
        }
    }

    if grad_tensors.is_empty() {
        return Ok(0.0);
    }

    // Compute the total norm.
    let total_norm = if norm_type == f64::INFINITY {
        // Max norm: find the maximum absolute value across all gradients.
        let mut max_val = 0.0_f64;
        for (_, data) in &grad_tensors {
            for &val in data.iter() {
                let abs = cast::<T, f64>(val)?.abs();
                if abs > max_val {
                    max_val = abs;
                }
            }
        }
        max_val
    } else {
        // p-norm: (sum |g_i|^p)^(1/p)
        let mut norm_sum = 0.0_f64;
        for (_, data) in &grad_tensors {
            for &val in data.iter() {
                let abs = cast::<T, f64>(val)?.abs();
                norm_sum += abs.powf(norm_type);
            }
        }
        norm_sum.powf(1.0 / norm_type)
    };

    // Compute clipping coefficient: min(max_norm / (total_norm + eps), 1.0)
    // The epsilon prevents division by zero when all gradients are zero.
    let clip_coef = max_norm / (total_norm + 1e-6);

    if clip_coef < 1.0 {
        let clip_coef_t: T = cast(clip_coef)?;

        // Scale all gradients in-place.
        for (i, param) in params.iter().enumerate() {
            if let Some(grad) = param.grad()? {
                let data = &grad_tensors[i].1;
                // Find the correct data entry — we only pushed entries for
                // params that had gradients, so we need to track the mapping.
                // Since we iterate params in the same order, use a separate
                // counter.
                let _ = data; // suppress unused warning, we use grad_data_idx below
                let _ = grad; // same
            }
        }

        // Actually perform the clipping using the collected grad_tensors.
        // We need to set the clipped gradients back onto the parameters.
        let mut grad_idx = 0;
        for param in params {
            if let Some(_grad) = param.grad()? {
                let (ref grad_tensor, ref data) = grad_tensors[grad_idx];
                let clipped: Vec<T> = data.iter().map(|&v| v * clip_coef_t).collect();
                let new_grad = Tensor::from_storage(
                    TensorStorage::cpu(clipped),
                    grad_tensor.shape().to_vec(),
                    false,
                )?;
                param.set_grad(Some(new_grad))?;
                grad_idx += 1;
            }
        }
    }

    Ok(total_norm)
}

/// Clip each gradient element to `[-clip_value, clip_value]`.
///
/// Gradients are modified **in-place**. Unlike [`clip_grad_norm_`], this
/// function clips individual elements independently, which is simpler but
/// less principled (it changes the direction of the gradient vector).
///
/// # Arguments
///
/// * `params` - An iterable of parameters whose `.grad()` will be clipped.
/// * `clip_value` - Maximum absolute value for any gradient element.
///
/// # Errors
///
/// Returns `FerrotorchError::InvalidArgument` when `clip_value < 0.0`.
/// Also propagates any `FerrotorchError` from the underlying tensor reads
/// / writes (`grad()`, `data_vec()`, `set_grad()`). Also returns
/// `FerrotorchError::InvalidArgument` from the numeric `cast` helper when
/// `clip_value` (or its negation) cannot be represented as `T`.
///
/// # Examples
///
/// ```ignore
/// use ferrotorch_train::clip_grad_value_;
///
/// clip_grad_value_(&model.parameters(), 0.5)?;
/// ```
pub fn clip_grad_value_<T: Float>(
    params: &[&Parameter<T>],
    clip_value: f64,
) -> FerrotorchResult<()> {
    if clip_value < 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("clip_value must be non-negative, got {clip_value}"),
        });
    }

    let clip_pos: T = cast(clip_value)?;
    let clip_neg: T = cast(-clip_value)?;

    for param in params {
        if let Some(grad) = param.grad()? {
            let data = grad.data_vec()?;
            let clamped: Vec<T> = data
                .iter()
                .map(|&v| {
                    if v > clip_pos {
                        clip_pos
                    } else if v < clip_neg {
                        clip_neg
                    } else {
                        v
                    }
                })
                .collect();
            let new_grad =
                Tensor::from_storage(TensorStorage::cpu(clamped), grad.shape().to_vec(), false)?;
            param.set_grad(Some(new_grad))?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_nn::Parameter;

    fn make_param_with_grad(data: &[f32], grad_data: &[f32], shape: &[usize]) -> Parameter<f32> {
        let p = Parameter::from_slice(data, shape).unwrap();
        let grad = Tensor::from_storage(
            TensorStorage::cpu(grad_data.to_vec()),
            shape.to_vec(),
            false,
        )
        .unwrap();
        p.set_grad(Some(grad)).unwrap();
        p
    }

    // -- clip_grad_norm_ with L2 norm ----------------------------------------

    #[test]
    fn test_clip_grad_norm_l2_clips_when_above() {
        // Gradient = [3.0, 4.0], L2 norm = 5.0. Max norm = 2.5.
        let p = make_param_with_grad(&[1.0, 2.0], &[3.0, 4.0], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 2.5, 2.0).unwrap();

        assert!(
            (total_norm - 5.0).abs() < 1e-5,
            "total norm should be 5.0, got {total_norm}"
        );

        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        // clip_coef = 2.5 / (5.0 + 1e-6) ~= 0.5
        // clipped = [3.0 * 0.5, 4.0 * 0.5] = [1.5, 2.0]
        let expected_coef = 2.5 / (5.0 + 1e-6);
        assert!(
            (data[0] - 3.0 * expected_coef as f32).abs() < 1e-4,
            "expected ~1.5, got {}",
            data[0]
        );
        assert!(
            (data[1] - 4.0 * expected_coef as f32).abs() < 1e-4,
            "expected ~2.0, got {}",
            data[1]
        );
    }

    #[test]
    fn test_clip_grad_norm_l2_no_clip_when_below() {
        // Gradient = [0.1, 0.2], L2 norm ~= 0.2236. Max norm = 1.0.
        let p = make_param_with_grad(&[1.0, 2.0], &[0.1, 0.2], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 1.0, 2.0).unwrap();

        let expected_norm = (0.01_f64 + 0.04).sqrt();
        assert!(
            (total_norm - expected_norm).abs() < 1e-5,
            "total norm should be {expected_norm}, got {total_norm}"
        );

        // Gradients should be unchanged.
        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 0.1).abs() < 1e-6);
        assert!((data[1] - 0.2).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_multiple_params() {
        // Two parameters: grads = [3.0] and [4.0]. Total L2 norm = 5.0.
        let p1 = make_param_with_grad(&[1.0], &[3.0], &[1]);
        let p2 = make_param_with_grad(&[2.0], &[4.0], &[1]);
        let params: Vec<&Parameter<f32>> = vec![&p1, &p2];

        let total_norm = clip_grad_norm_(&params, 2.5, 2.0).unwrap();
        assert!((total_norm - 5.0).abs() < 1e-5);

        let coef = 2.5 / (5.0 + 1e-6);
        let g1 = p1.grad().unwrap().unwrap().data().unwrap()[0];
        let g2 = p2.grad().unwrap().unwrap().data().unwrap()[0];
        assert!((g1 - 3.0 * coef as f32).abs() < 1e-4);
        assert!((g2 - 4.0 * coef as f32).abs() < 1e-4);
    }

    // -- clip_grad_norm_ with L1 norm ----------------------------------------

    #[test]
    fn test_clip_grad_norm_l1() {
        // Gradient = [3.0, -4.0], L1 norm = 7.0. Max norm = 3.5.
        let p = make_param_with_grad(&[1.0, 2.0], &[3.0, -4.0], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 3.5, 1.0).unwrap();
        assert!((total_norm - 7.0).abs() < 1e-5);

        let coef = 3.5 / (7.0 + 1e-6);
        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 3.0 * coef as f32).abs() < 1e-4);
        assert!((data[1] - (-4.0 * coef as f32)).abs() < 1e-4);
    }

    // -- clip_grad_norm_ with inf norm ---------------------------------------

    #[test]
    fn test_clip_grad_norm_inf() {
        // Gradient = [3.0, -7.0], inf norm = 7.0. Max norm = 3.5.
        let p = make_param_with_grad(&[1.0, 2.0], &[3.0, -7.0], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 3.5, f64::INFINITY).unwrap();
        assert!((total_norm - 7.0).abs() < 1e-5);

        let coef = 3.5 / (7.0 + 1e-6);
        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 3.0 * coef as f32).abs() < 1e-4);
        assert!((data[1] - (-7.0 * coef as f32)).abs() < 1e-4);
    }

    // -- clip_grad_norm_ with no gradients -----------------------------------

    #[test]
    fn test_clip_grad_norm_no_gradients() {
        let p = Parameter::<f32>::zeros(&[3]).unwrap();
        // No gradient set.
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 1.0, 2.0).unwrap();
        assert!((total_norm - 0.0).abs() < 1e-12);
    }

    // -- clip_grad_norm_ with zero gradients ---------------------------------

    #[test]
    fn test_clip_grad_norm_zero_gradients() {
        let p = make_param_with_grad(&[1.0, 2.0], &[0.0, 0.0], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        let total_norm = clip_grad_norm_(&params, 1.0, 2.0).unwrap();
        assert!(total_norm < 1e-12);

        // Gradients remain zero.
        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0]).abs() < 1e-12);
        assert!((data[1]).abs() < 1e-12);
    }

    // -- clip_grad_value_ ----------------------------------------------------

    #[test]
    fn test_clip_grad_value_clips_large() {
        let p = make_param_with_grad(&[1.0, 2.0, 3.0], &[10.0, -10.0, 0.5], &[3]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        clip_grad_value_(&params, 1.0).unwrap();

        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!(
            (data[0] - 1.0).abs() < 1e-6,
            "expected 1.0, got {}",
            data[0]
        );
        assert!(
            (data[1] - (-1.0)).abs() < 1e-6,
            "expected -1.0, got {}",
            data[1]
        );
        assert!(
            (data[2] - 0.5).abs() < 1e-6,
            "expected 0.5, got {}",
            data[2]
        );
    }

    #[test]
    fn test_clip_grad_value_no_clip_needed() {
        let p = make_param_with_grad(&[1.0, 2.0], &[0.3, -0.3], &[2]);
        let params: Vec<&Parameter<f32>> = vec![&p];

        clip_grad_value_(&params, 1.0).unwrap();

        let grad = p.grad().unwrap().unwrap();
        let data = grad.data().unwrap();
        assert!((data[0] - 0.3).abs() < 1e-6);
        assert!((data[1] - (-0.3)).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_value_no_gradients() {
        let p = Parameter::<f32>::zeros(&[2]).unwrap();
        let params: Vec<&Parameter<f32>> = vec![&p];

        // Should succeed with no-op.
        clip_grad_value_(&params, 1.0).unwrap();
    }

    #[test]
    fn test_clip_grad_value_multiple_params() {
        let p1 = make_param_with_grad(&[1.0], &[5.0], &[1]);
        let p2 = make_param_with_grad(&[2.0], &[-5.0], &[1]);
        let params: Vec<&Parameter<f32>> = vec![&p1, &p2];

        clip_grad_value_(&params, 2.0).unwrap();

        assert!((p1.grad().unwrap().unwrap().data().unwrap()[0] - 2.0).abs() < 1e-6);
        assert!((p2.grad().unwrap().unwrap().data().unwrap()[0] - (-2.0)).abs() < 1e-6);
    }

    // -- Send + Sync ---------------------------------------------------------

    #[test]
    fn test_functions_are_callable() {
        // Smoke test: the functions exist and have the expected signatures.
        fn _test_norm<T: Float>(params: &[&Parameter<T>]) {
            let _ = clip_grad_norm_(params, 1.0, 2.0);
        }
        fn _test_value<T: Float>(params: &[&Parameter<T>]) {
            let _ = clip_grad_value_(params, 1.0);
        }
    }
}
