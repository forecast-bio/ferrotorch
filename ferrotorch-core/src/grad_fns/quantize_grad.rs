//! Differentiable fake-quantization for quantization-aware training
//! (QAT). CL-293.
//!
//! Provides a tensor-level `fake_quantize_differentiable` op that
//! wraps `FakeQuantize` in a proper autograd node using the
//! straight-through estimator (STE):
//!
//! ```text
//! forward(x) = dequantize(quantize(x))
//! backward(grad) = grad * (x >= range_min && x <= range_max ? 1 : 0)
//! ```
//!
//! This is the clipped STE used by PyTorch's
//! `torch.fake_quantize_per_tensor_affine`. Values outside the
//! representable range have zero gradient (matching the behavior of
//! clamp at the range boundaries), while in-range values pass the
//! gradient through unchanged.
//!
//! The forward quantization is non-differentiable (it contains
//! `round` and `clamp`), so the STE substitutes a piecewise-linear
//! surrogate gradient that lets models train through quantization
//! noise.

use std::sync::Arc;

use crate::dtype::Float;
use crate::error::FerrotorchResult;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Differentiable fake quantize per-tensor (affine).
///
/// Forward: `dequantize(round(x / scale + zp).clamp(qmin, qmax))`.
/// Backward: clipped STE — `grad * mask` where `mask` is 1 for
/// values within `[dequantize(qmin), dequantize(qmax)]` and 0 for
/// out-of-range values.
///
/// # Arguments
///
/// * `input` — the tensor to fake-quantize.
/// * `scale` — quantization scale (positive, non-zero).
/// * `zero_point` — integer zero point for affine quantization.
///   For symmetric schemes pass `0`.
/// * `qmin` — minimum integer value of the target dtype
///   (e.g. `-128` for int8 affine or `0` for uint8).
/// * `qmax` — maximum integer value of the target dtype.
///
/// # Errors
///
/// - `FerrotorchError::InvalidArgument` if `scale <= 0`.
pub fn fake_quantize_differentiable<T: Float>(
    input: &Tensor<T>,
    scale: f64,
    zero_point: i32,
    qmin: i32,
    qmax: i32,
) -> FerrotorchResult<Tensor<T>> {
    use crate::error::FerrotorchError;
    if scale.is_nan() || scale <= 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("fake_quantize_differentiable: scale must be > 0, got {scale}"),
        });
    }
    if qmin >= qmax {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("fake_quantize_differentiable: qmin ({qmin}) must be < qmax ({qmax})"),
        });
    }

    let data = input.data_vec()?;
    let scale_f = T::from(scale).unwrap();
    let zp_f = T::from(zero_point as f64).unwrap();
    let qmin_f = T::from(qmin as f64).unwrap();
    let qmax_f = T::from(qmax as f64).unwrap();

    // Dequantized range boundaries for the STE mask.
    let range_min: T = (qmin_f - zp_f) * scale_f;
    let range_max: T = (qmax_f - zp_f) * scale_f;

    let mut out = Vec::with_capacity(data.len());
    for &x in &data {
        // Fake quantize: q = round(x / scale + zp).clamp(qmin, qmax)
        let scaled = x / scale_f + zp_f;
        let rounded = scaled.round();
        let clamped = if rounded < qmin_f {
            qmin_f
        } else if rounded > qmax_f {
            qmax_f
        } else {
            rounded
        };
        // Dequantize: dq = (q - zp) * scale
        let dq = (clamped - zp_f) * scale_f;
        out.push(dq);
    }

    let storage = TensorStorage::cpu(out);
    let shape = input.shape().to_vec();

    if input.requires_grad() && crate::autograd::no_grad::is_grad_enabled() {
        let grad_fn = Arc::new(FakeQuantizeBackward::<T> {
            input: input.clone(),
            range_min,
            range_max,
        });
        Tensor::from_operation(storage, shape, grad_fn)
    } else {
        Tensor::from_storage(storage, shape, false)
    }
}

/// Backward node for fake_quantize using clipped STE.
#[derive(Debug)]
struct FakeQuantizeBackward<T: Float> {
    input: Tensor<T>,
    range_min: T,
    range_max: T,
}

impl<T: Float> GradFn<T> for FakeQuantizeBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        let grad_data = grad_output.data_vec()?;
        let input_data = self.input.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let grad: Vec<T> = input_data
            .iter()
            .zip(grad_data.iter())
            .map(|(&x, &g)| {
                if x >= self.range_min && x <= self.range_max {
                    g
                } else {
                    zero
                }
            })
            .collect();
        let storage = TensorStorage::cpu(grad);
        let shape = self.input.shape().to_vec();
        Ok(vec![Some(Tensor::from_storage(storage, shape, false)?)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FakeQuantizeBackward"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::backward;

    fn t(data: Vec<f32>, shape: Vec<usize>, req_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape, req_grad).unwrap()
    }

    // ── forward correctness ────────────────────────────────────────

    #[test]
    fn fake_quantize_round_trips_representable_values() {
        // int8 symmetric: qmin=-128, qmax=127, scale chosen so
        // exact multiples of scale are fixed points.
        let scale = 0.1;
        let zp = 0;
        let qmin = -128;
        let qmax = 127;

        // Values that are exact multiples of scale should round-trip.
        let input = t(vec![0.0, 0.1, 0.2, -0.1, -0.2], vec![5], false);
        let out = fake_quantize_differentiable(&input, scale, zp, qmin, qmax).unwrap();
        let data = out.data().unwrap();
        for (got, expected) in data.iter().zip([0.0, 0.1, 0.2, -0.1, -0.2].iter()) {
            assert!(
                (got - expected).abs() < 1e-5,
                "expected {expected}, got {got}"
            );
        }
    }

    #[test]
    // reason: with scale=1.0, zp=0 the quantize-then-dequantize round-trip
    // is exact for integer-valued inputs in range, and clamping snaps
    // out-of-range inputs to exact integer boundaries. Every expected
    // value is bit-exactly representable in f32, so equality is correct.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_clamps_out_of_range_values() {
        // int8: [-128, 127] with scale 1.0, zp 0 → representable
        // range is [-128.0, 127.0]. Values outside should be
        // clamped to the nearest boundary.
        let input = t(vec![-200.0, -100.0, 0.0, 100.0, 200.0], vec![5], false);
        let out = fake_quantize_differentiable(&input, 1.0, 0, -128, 127).unwrap();
        let data = out.data().unwrap();
        assert_eq!(data[0], -128.0); // clamped
        assert_eq!(data[1], -100.0);
        assert_eq!(data[2], 0.0);
        assert_eq!(data[3], 100.0);
        assert_eq!(data[4], 127.0); // clamped
    }

    #[test]
    fn fake_quantize_rejects_zero_scale() {
        let input = t(vec![1.0], vec![1], false);
        let result = fake_quantize_differentiable(&input, 0.0, 0, -128, 127);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("scale must be > 0"));
    }

    #[test]
    fn fake_quantize_rejects_negative_scale() {
        let input = t(vec![1.0], vec![1], false);
        let result = fake_quantize_differentiable(&input, -0.1, 0, -128, 127);
        assert!(result.is_err());
    }

    #[test]
    fn fake_quantize_rejects_inverted_range() {
        let input = t(vec![1.0], vec![1], false);
        let result = fake_quantize_differentiable(&input, 1.0, 0, 128, -128);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("qmin"));
    }

    #[test]
    fn fake_quantize_asymmetric_with_zero_point() {
        // uint8: [0, 255] with a non-zero zero-point shifts the
        // representable range into the positives.
        // scale=1.0, zp=128 → range is [(-128)*1, (127)*1] = [-128, 127].
        // Actually with qmin=0, qmax=255, zp=128, range =
        // [(0-128)*1, (255-128)*1] = [-128, 127].
        let input = t(vec![-128.0, 0.0, 127.0], vec![3], false);
        let out = fake_quantize_differentiable(&input, 1.0, 128, 0, 255).unwrap();
        let data = out.data().unwrap();
        assert_eq!(data, &[-128.0, 0.0, 127.0]);
    }

    // ── backward / STE ─────────────────────────────────────────────

    #[test]
    // reason: STE passes gradient 1.0 through for in-range values; this is
    // a binary mask (1.0 for in-range, 0.0 for out-of-range), written as an
    // exact bit pattern, never the result of arithmetic.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_ste_passes_grad_for_in_range_values() {
        // scale=1.0, zp=0, range=[-128, 127]. Values inside this
        // range should have gradient 1.0 passed through unchanged.
        let input = t(vec![-10.0, 0.0, 10.0, 50.0], vec![4], true);
        let out = fake_quantize_differentiable(&input, 1.0, 0, -128, 127).unwrap();
        // Sum for a scalar backward seed.
        let loss = out
            .data_vec()
            .unwrap()
            .into_iter()
            .fold(0.0f32, |a, b| a + b);
        // Manually trigger backward via autograd. Use sum to get a
        // scalar root.
        let sum = crate::grad_fns::reduction::sum(&out).unwrap();
        backward(&sum).unwrap();
        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        for &g in grad_data {
            assert_eq!(g, 1.0);
        }
        let _ = loss;
    }

    #[test]
    // reason: STE gradient mask is binary (1.0 for in-range, 0.0 for clipped);
    // each grad slot holds the exact bit pattern of the chosen sentinel,
    // never the result of arithmetic — equality is the right check.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_ste_zeros_grad_for_out_of_range_values() {
        // Only values in [-1.0, 1.0] (scale=0.01, range=[-1.28, 1.27])
        // get grad 1, others get 0. Use scale=0.01, qmin=-128, qmax=127.
        let input = t(vec![-5.0, -1.0, 0.0, 1.0, 5.0, 100.0], vec![6], true);
        let out = fake_quantize_differentiable(&input, 0.01, 0, -128, 127).unwrap();
        let sum = crate::grad_fns::reduction::sum(&out).unwrap();
        backward(&sum).unwrap();
        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        // -5.0 is below -1.28 → grad 0
        assert_eq!(grad_data[0], 0.0);
        // -1.0 is within [-1.28, 1.27] → grad 1
        assert_eq!(grad_data[1], 1.0);
        // 0.0 in range → 1
        assert_eq!(grad_data[2], 1.0);
        // 1.0 in range → 1
        assert_eq!(grad_data[3], 1.0);
        // 5.0 above 1.27 → 0
        assert_eq!(grad_data[4], 0.0);
        // 100.0 above 1.27 → 0
        assert_eq!(grad_data[5], 0.0);
    }

    #[test]
    fn fake_quantize_no_grad_when_input_doesnt_require_grad() {
        let input = t(vec![1.0, 2.0], vec![2], false);
        let out = fake_quantize_differentiable(&input, 1.0, 0, -128, 127).unwrap();
        assert!(!out.requires_grad());
        assert!(out.grad_fn().is_none());
    }

    #[test]
    fn fake_quantize_preserves_grad_fn_when_input_requires_grad() {
        let input = t(vec![1.0, 2.0], vec![2], true);
        let out = fake_quantize_differentiable(&input, 1.0, 0, -128, 127).unwrap();
        assert!(out.requires_grad());
        assert!(out.grad_fn().is_some());
    }

    #[test]
    fn fake_quantize_no_grad_context_skips_grad_fn() {
        use crate::autograd::no_grad::no_grad;
        let input = t(vec![1.0, 2.0], vec![2], true);
        let out = no_grad(|| fake_quantize_differentiable(&input, 1.0, 0, -128, 127)).unwrap();
        // Inside no_grad, even a requires_grad input produces an
        // output with no grad_fn.
        assert!(out.grad_fn().is_none());
    }

    #[test]
    // reason: chained STE × relu mask product is still binary (0.0 or 1.0
    // — multiplying two binary masks). Each grad slot holds an exact bit
    // pattern, never a non-trivial arithmetic result.
    #[allow(clippy::float_cmp)]
    fn fake_quantize_chains_through_autograd_with_relu() {
        // y = relu(fake_quantize(x)); backward should flow through
        // both layers and give the expected combined gradient.
        let input = t(vec![-2.0, -0.5, 0.5, 2.0], vec![4], true);
        let fq = fake_quantize_differentiable(&input, 0.01, 0, -128, 127).unwrap();
        let relu_out = crate::grad_fns::activation::relu(&fq).unwrap();
        let sum = crate::grad_fns::reduction::sum(&relu_out).unwrap();
        backward(&sum).unwrap();
        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        // x=-2.0: out of range (range=[-1.28, 1.27]) → STE mask 0 → grad 0.
        assert_eq!(grad_data[0], 0.0);
        // x=-0.5: in range, but relu zeros negatives → grad 0 (relu mask).
        assert_eq!(grad_data[1], 0.0);
        // x=0.5: in range, relu passes → grad 1.
        assert_eq!(grad_data[2], 1.0);
        // x=2.0: out of range → STE mask 0 → grad 0.
        assert_eq!(grad_data[3], 0.0);
    }
}
