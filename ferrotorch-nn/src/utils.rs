//! Gradient clipping utilities.
//!
//! These mirror PyTorch's `torch.nn.utils.clip_grad_norm_` and
//! `torch.nn.utils.clip_grad_value_`, operating in-place on the
//! gradients of a parameter slice.

use ferrotorch_core::{Device, Float, FerrotorchResult, Tensor, TensorStorage};

use crate::parameter::Parameter;

/// Clip the total gradient norm of an iterable of parameters.
///
/// Computes the total norm of all parameter gradients (treating them as
/// a single concatenated vector), and if the total norm exceeds
/// `max_norm`, rescales every gradient by `max_norm / total_norm`.
///
/// # Arguments
///
/// * `params` — Slice of parameter references whose gradients will be clipped.
/// * `max_norm` — Maximum allowed norm.
/// * `norm_type` — Type of the norm (e.g. `2.0` for L2 norm).
///
/// # Returns
///
/// The total gradient norm **before** any clipping is applied.
///
/// Parameters whose gradient is `None` are silently skipped.
pub fn clip_grad_norm_<T: Float>(
    params: &[&Parameter<T>],
    max_norm: f64,
    norm_type: f64,
) -> FerrotorchResult<f64> {
    // Collect all gradient tensors (skip None).
    let grads: Vec<Tensor<T>> = params
        .iter()
        .filter_map(|p| p.grad().ok().flatten())
        .collect();

    // Compute total norm.
    let total_norm: f64 = if norm_type == f64::INFINITY {
        // Infinity norm: max absolute value across all gradients.
        let mut max_val: f64 = 0.0;
        for g in &grads {
            let data = g.data_vec()?;
            for v in &data {
                let abs_v = v.to_f64().unwrap().abs();
                if abs_v > max_val {
                    max_val = abs_v;
                }
            }
        }
        max_val
    } else {
        // General p-norm: (sum |g_i|^p)^(1/p).
        let mut accum: f64 = 0.0;
        for g in &grads {
            let data = g.data_vec()?;
            for v in &data {
                accum += v.to_f64().unwrap().abs().powf(norm_type);
            }
        }
        accum.powf(1.0 / norm_type)
    };

    // Clip if necessary.
    if total_norm > max_norm {
        let clip_coef = max_norm / total_norm;
        let clip_t = T::from(clip_coef).unwrap();

        for param in params {
            if let Some(g) = param.grad()? {
                let data = g.data_vec()?;
                let scaled: Vec<T> = data.iter().map(|&v| v * clip_t).collect();
                let device = g.device();
                let new_grad =
                    Tensor::from_storage(TensorStorage::cpu(scaled), g.shape().to_vec(), false)?;
                let new_grad = if device != Device::Cpu { new_grad.to(device)? } else { new_grad };
                param.set_grad(Some(new_grad))?;
            }
        }
    }

    Ok(total_norm)
}

/// Clamp all gradient values to the range `[-clip_value, clip_value]`.
///
/// Each element of every parameter gradient is clamped independently.
/// Parameters whose gradient is `None` are silently skipped.
///
/// # Arguments
///
/// * `params` — Slice of parameter references whose gradients will be clamped.
/// * `clip_value` — Maximum absolute value for gradient elements.
pub fn clip_grad_value_<T: Float>(
    params: &[&Parameter<T>],
    clip_value: f64,
) -> FerrotorchResult<()> {
    let lo = T::from(-clip_value).unwrap();
    let hi = T::from(clip_value).unwrap();

    for param in params {
        if let Some(g) = param.grad()? {
            let data = g.data_vec()?;
            let clamped: Vec<T> = data
                .iter()
                .map(|&v| {
                    if v < lo {
                        lo
                    } else if v > hi {
                        hi
                    } else {
                        v
                    }
                })
                .collect();
            let device = g.device();
            let new_grad =
                Tensor::from_storage(TensorStorage::cpu(clamped), g.shape().to_vec(), false)?;
            let new_grad = if device != Device::Cpu { new_grad.to(device)? } else { new_grad };
            param.set_grad(Some(new_grad))?;
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a parameter and manually set its gradient.
    fn param_with_grad(shape: &[usize], grad_data: &[f32]) -> Parameter<f32> {
        let p = Parameter::<f32>::zeros(shape).unwrap();
        let grad = Tensor::from_storage(
            TensorStorage::cpu(grad_data.to_vec()),
            shape.to_vec(),
            false,
        )
        .unwrap();
        p.set_grad(Some(grad)).unwrap();
        p
    }

    // -----------------------------------------------------------------------
    // clip_grad_norm_
    // -----------------------------------------------------------------------

    #[test]
    fn test_clip_grad_norm_reduces_norm() {
        // Gradient = [3.0, 4.0] => L2 norm = 5.0
        let p = param_with_grad(&[2], &[3.0, 4.0]);
        let total = clip_grad_norm_(&[&p], 2.5, 2.0).unwrap();

        // Total norm before clipping should be 5.0.
        assert!((total - 5.0).abs() < 1e-6);

        // After clipping, the new norm should be ~2.5.
        let g = p.grad().unwrap().unwrap();
        let d = g.data().unwrap();
        let new_norm = (d[0] as f64 * d[0] as f64 + d[1] as f64 * d[1] as f64).sqrt();
        assert!((new_norm - 2.5).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_no_clip_when_below() {
        // Gradient = [1.0, 0.0] => L2 norm = 1.0
        let p = param_with_grad(&[2], &[1.0, 0.0]);
        let total = clip_grad_norm_(&[&p], 10.0, 2.0).unwrap();

        assert!((total - 1.0).abs() < 1e-6);

        // Gradient should be unchanged.
        let g = p.grad().unwrap().unwrap();
        let d = g.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert!((d[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_multiple_params() {
        // p1 grad = [3.0], p2 grad = [4.0] => joint L2 norm = 5.0
        let p1 = param_with_grad(&[1], &[3.0]);
        let p2 = param_with_grad(&[1], &[4.0]);
        let total = clip_grad_norm_(&[&p1, &p2], 2.5, 2.0).unwrap();

        assert!((total - 5.0).abs() < 1e-6);

        let g1 = p1.grad().unwrap().unwrap().data().unwrap()[0] as f64;
        let g2 = p2.grad().unwrap().unwrap().data().unwrap()[0] as f64;
        let new_norm = (g1 * g1 + g2 * g2).sqrt();
        assert!((new_norm - 2.5).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_returns_total_norm() {
        let p = param_with_grad(&[3], &[1.0, 2.0, 2.0]);
        // L2 norm = sqrt(1 + 4 + 4) = 3.0
        let total = clip_grad_norm_(&[&p], 100.0, 2.0).unwrap();
        assert!((total - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_skips_none_grads() {
        let p_with = param_with_grad(&[2], &[3.0, 4.0]);
        let p_without = Parameter::<f32>::zeros(&[2]).unwrap();
        // No grad set on p_without — should be silently skipped.
        let total = clip_grad_norm_(&[&p_with, &p_without], 2.5, 2.0).unwrap();
        assert!((total - 5.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // clip_grad_value_
    // -----------------------------------------------------------------------

    #[test]
    fn test_clip_grad_value_clamps_elements() {
        let p = param_with_grad(&[4], &[-5.0, 0.5, 3.0, -0.1]);
        clip_grad_value_(&[&p], 1.0).unwrap();

        let g = p.grad().unwrap().unwrap();
        let d = g.data().unwrap();
        assert!((d[0] - (-1.0)).abs() < 1e-6);
        assert!((d[1] - 0.5).abs() < 1e-6);
        assert!((d[2] - 1.0).abs() < 1e-6);
        assert!((d[3] - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_value_skips_none_grads() {
        let p = Parameter::<f32>::zeros(&[3]).unwrap();
        // No gradient — should succeed without error.
        clip_grad_value_(&[&p], 1.0).unwrap();
        assert!(p.grad().unwrap().is_none());
    }

    #[test]
    fn test_clip_grad_value_preserves_within_range() {
        let p = param_with_grad(&[3], &[0.1, -0.2, 0.3]);
        clip_grad_value_(&[&p], 1.0).unwrap();

        let g = p.grad().unwrap().unwrap();
        let d = g.data().unwrap();
        assert!((d[0] - 0.1).abs() < 1e-6);
        assert!((d[1] - (-0.2)).abs() < 1e-6);
        assert!((d[2] - 0.3).abs() < 1e-6);
    }
}
