//! Dropout regularization layers.
//!
//! [`Dropout`] randomly zeroes individual elements during training with
//! probability `p`, scaling surviving elements by `1/(1-p)` (inverted
//! dropout).  [`Dropout2d`] drops entire channels instead of individual
//! elements, useful for convolutional feature maps.
//!
//! Both modules are identity in eval mode and have zero learnable parameters.

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{Float, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};

use crate::module::Module;
use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// Internal xorshift PRNG (matches ferrotorch_core::creation::rand)
// ---------------------------------------------------------------------------

/// Seed a xorshift64 state from system time and thread id.
fn xorshift_seed() -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    use std::time::SystemTime;

    let mut hasher = DefaultHasher::new();
    SystemTime::now().hash(&mut hasher);
    std::thread::current().id().hash(&mut hasher);
    let mut state = hasher.finish();
    if state == 0 {
        state = 0xdeadbeefcafe;
    }
    state
}

/// Advance xorshift64 state and return a uniform value in [0, 1).
#[inline]
fn xorshift_next(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

// ---------------------------------------------------------------------------
// DropoutBackward
// ---------------------------------------------------------------------------

/// Backward node for elementwise dropout.
///
/// Reapplies the same binary mask scaled by `1/(1-p)` to the upstream
/// gradient, routing gradients only through surviving elements.
#[derive(Debug)]
struct DropoutBackward<T: Float> {
    input: Tensor<T>,
    /// The mask already includes the `1/(1-p)` scaling factor: each element
    /// is either `0` or `1/(1-p)`.
    scaled_mask: Vec<T>,
}

impl<T: Float> GradFn<T> for DropoutBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
            let go_data = cpu_go.data()?;
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(self.scaled_mask.iter())
                .map(|(&g, &m)| g * m)
                .collect();
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.input.shape().to_vec(),
                false,
            )?;
            Some(if self.input.is_cuda() { g.to(self.input.device())? } else { g })
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "DropoutBackward"
    }
}

// ---------------------------------------------------------------------------
// Dropout2dBackward
// ---------------------------------------------------------------------------

/// Backward node for channel-wise dropout.
///
/// Identical to [`DropoutBackward`] — the mask shape already encodes the
/// channel-level structure (all spatial positions in a dropped channel are 0).
#[derive(Debug)]
struct Dropout2dBackward<T: Float> {
    input: Tensor<T>,
    scaled_mask: Vec<T>,
}

impl<T: Float> GradFn<T> for Dropout2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let cpu_go = if grad_output.is_cuda() { grad_output.cpu()? } else { grad_output.clone() };
            let go_data = cpu_go.data()?;
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(self.scaled_mask.iter())
                .map(|(&g, &m)| g * m)
                .collect();
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.input.shape().to_vec(),
                false,
            )?;
            Some(if self.input.is_cuda() { g.to(self.input.device())? } else { g })
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "Dropout2dBackward"
    }
}

// ===========================================================================
// Dropout
// ===========================================================================

/// Randomly zeroes elements with probability `p` during training.
///
/// During training, each element is independently set to zero with probability
/// `p` and scaled by `1/(1-p)` so that the expected value is preserved
/// (inverted dropout).  During evaluation (`eval()` mode), the input is
/// returned unchanged.
///
/// # Panics
///
/// The constructor returns an error if `p` is outside `[0, 1)`.
#[derive(Debug)]
pub struct Dropout<T: Float> {
    p: f64,
    training: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Dropout<T> {
    /// Create a new `Dropout` layer.
    ///
    /// `p` is the probability of an element being zeroed. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("dropout probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: Float> Module<T> for Dropout<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Eval mode or p == 0: identity.
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let numel = input.numel();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        // GPU fast path: run dropout kernel entirely on device.
        if input.is_cuda() {
            if let Some(backend) = ferrotorch_core::gpu_dispatch::gpu_backend() {
                let threshold = (self.p * u32::MAX as f64) as u32;
                let scale_f32 = 1.0f32 / (1.0 - self.p as f32);
                let seed = xorshift_seed() as u32;
                let handle = backend.dropout_f32(
                    input.gpu_handle()?,
                    threshold,
                    scale_f32,
                    seed,
                )?;

                // For backward, we need the mask. Reconstruct it from the same
                // seed on CPU (the kernel is deterministic for a given seed).
                if is_grad_enabled() && input.requires_grad() {
                    let scaled_mask: Vec<T> = (0..numel)
                        .map(|i| {
                            let mut r = (i as u32).wrapping_mul(2654435761) ^ seed;
                            r ^= r << 13; r ^= r >> 17; r ^= r << 5;
                            if r < threshold { zero } else { scale }
                        })
                        .collect();
                    return Tensor::from_operation(
                        TensorStorage::gpu(handle),
                        input.shape().to_vec(),
                        Arc::new(DropoutBackward {
                            input: input.clone(),
                            scaled_mask,
                        }),
                    );
                } else {
                    return Tensor::from_storage(
                        TensorStorage::gpu(handle),
                        input.shape().to_vec(),
                        false,
                    );
                }
            }
        }

        // CPU path.
        let mut state = xorshift_seed();
        let scaled_mask: Vec<T> = (0..numel)
            .map(|_| {
                if xorshift_next(&mut state) < self.p {
                    zero
                } else {
                    scale
                }
            })
            .collect();

        let input_data = input.data()?;
        let output_data: Vec<T> = input_data
            .iter()
            .zip(scaled_mask.iter())
            .map(|(&x, &m)| x * m)
            .collect();

        if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(DropoutBackward {
                    input: input.clone(),
                    scaled_mask,
                }),
            )
        } else {
            Tensor::from_storage(TensorStorage::cpu(output_data), input.shape().to_vec(), false)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Dropout2d
// ===========================================================================

/// Randomly zeroes entire channels with probability `p` during training.
///
/// Expects input of shape `[B, C, ...]` (at least 2 dimensions). During
/// training, each channel (the entire `[H, W, ...]` slice for a given `b, c`)
/// is independently set to zero with probability `p` and surviving channels
/// are scaled by `1/(1-p)`.  During evaluation the input is returned unchanged.
///
/// # Panics
///
/// The constructor returns an error if `p` is outside `[0, 1)`.
#[derive(Debug)]
pub struct Dropout2d<T: Float> {
    p: f64,
    training: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Dropout2d<T> {
    /// Create a new `Dropout2d` layer.
    ///
    /// `p` is the probability of an entire channel being zeroed. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("dropout2d probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: Float> Module<T> for Dropout2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Eval mode or p == 0: identity.
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let device = input.device();
        let shape = input.shape();
        if shape.len() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dropout2d expects at least 2D input [B, C, ...], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let spatial: usize = shape[2..].iter().product();
        // If there are no spatial dims, spatial == 1 (product of empty iter).
        let spatial = if shape.len() == 2 { 1 } else { spatial };

        let numel = input.numel();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        // Generate per-channel keep/drop decisions.
        let mut state = xorshift_seed();
        let channel_mask: Vec<bool> = (0..batch * channels)
            .map(|_| xorshift_next(&mut state) >= self.p)
            .collect();

        // Expand channel mask to full element mask.
        let scaled_mask: Vec<T> = {
            let mut mask = Vec::with_capacity(numel);
            for bc in 0..batch * channels {
                let val = if channel_mask[bc] { scale } else { zero };
                for _ in 0..spatial {
                    mask.push(val);
                }
            }
            mask
        };

        let input_data = input.data()?;
        let output_data: Vec<T> = input_data
            .iter()
            .zip(scaled_mask.iter())
            .map(|(&x, &m)| x * m)
            .collect();

        let result = if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(Dropout2dBackward {
                    input: input.clone(),
                    scaled_mask,
                }),
            )?
        } else {
            Tensor::from_storage(TensorStorage::cpu(output_data), input.shape().to_vec(), false)?
        };
        if device.is_cuda() { result.to(device) } else { Ok(result) }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a leaf tensor with given data and shape.
    fn leaf_tensor(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Dropout
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout_rate_approximately_correct() {
        let d = Dropout::<f32>::new(0.5).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[100_000]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        // Count zeros — should be roughly 50%.
        let zeros = data.iter().filter(|&&x| x == 0.0).count();
        let rate = zeros as f64 / data.len() as f64;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "dropout rate = {rate}, expected ~0.5"
        );

        // Surviving elements should be scaled by 1/(1-0.5) = 2.0.
        let non_zero: Vec<f32> = data.iter().copied().filter(|&x| x != 0.0).collect();
        assert!(!non_zero.is_empty());
        for &v in &non_zero {
            assert!(
                (v - 2.0).abs() < 1e-6,
                "surviving element = {v}, expected 2.0"
            );
        }
    }

    #[test]
    fn test_dropout_eval_is_identity() {
        let mut d = Dropout::<f32>::new(0.5).unwrap();
        d.eval();
        assert!(!d.is_training());

        let input = ferrotorch_core::ones::<f32>(&[100]).unwrap();
        let output = d.forward(&input).unwrap();

        // In eval mode the output should be the exact same Arc (identity).
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout_zero_prob_is_identity() {
        let d = Dropout::<f32>::new(0.0).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[100]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout_invalid_p() {
        assert!(Dropout::<f32>::new(1.0).is_err());
        assert!(Dropout::<f32>::new(-0.1).is_err());
        assert!(Dropout::<f32>::new(1.5).is_err());
    }

    #[test]
    fn test_dropout_backward_routes_through_surviving() {
        let d = Dropout::<f32>::new(0.5).unwrap();
        let input = leaf_tensor(&[1.0; 1000], &[1000], true);
        let output = d.forward(&input).unwrap();

        // To backward we need a scalar loss. Sum the output manually.
        let out_data = output.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();

        // Build a SumBackward so we can call backward.
        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: output.clone(),
            }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();

        // Every gradient element should be either 0 (dropped) or 1/(1-p) = 2.0 (survived).
        for &g in grad_data {
            assert!(
                g == 0.0 || (g - 2.0).abs() < 1e-6,
                "gradient element = {g}, expected 0.0 or 2.0"
            );
        }

        // The dropout mask for forward and backward should match: output zero
        // iff gradient zero.
        let out_data = output.data().unwrap();
        for (i, (&o, &g)) in out_data.iter().zip(grad_data.iter()).enumerate() {
            assert_eq!(
                o == 0.0,
                g == 0.0,
                "mismatch at index {i}: output={o}, grad={g}"
            );
        }
    }

    #[test]
    fn test_dropout_no_parameters() {
        let d = Dropout::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
        assert!(d.named_parameters().is_empty());
    }

    #[test]
    fn test_dropout_train_eval_toggle() {
        let mut d = Dropout::<f32>::new(0.5).unwrap();
        assert!(d.is_training());
        d.eval();
        assert!(!d.is_training());
        d.train();
        assert!(d.is_training());
    }

    #[test]
    fn test_dropout_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dropout<f32>>();
        assert_send_sync::<Dropout<f64>>();
    }

    // -----------------------------------------------------------------------
    // Dropout2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout2d_drops_whole_channels() {
        let d = Dropout2d::<f32>::new(0.5).unwrap();
        // Shape: [2, 10, 4, 4] — 2 batches, 10 channels, 4x4 spatial.
        let input = ferrotorch_core::ones::<f32>(&[2, 10, 4, 4]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let spatial = 4 * 4;
        // Check that each channel is either entirely zero or entirely scaled.
        for b in 0..2 {
            for c in 0..10 {
                let start = (b * 10 + c) * spatial;
                let end = start + spatial;
                let channel = &data[start..end];

                let first = channel[0];
                assert!(
                    channel.iter().all(|&x| (x - first).abs() < 1e-6),
                    "channel (b={b}, c={c}) is not uniform: first={first}, channel={channel:?}"
                );
                // Value should be 0 or 1/(1-0.5) = 2.0.
                assert!(
                    first == 0.0 || (first - 2.0).abs() < 1e-6,
                    "channel value = {first}, expected 0.0 or 2.0"
                );
            }
        }
    }

    #[test]
    fn test_dropout2d_rate_approximately_correct() {
        let d = Dropout2d::<f32>::new(0.5).unwrap();
        // Many channels to get a good statistical sample.
        let input = ferrotorch_core::ones::<f32>(&[1, 1000, 2, 2]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let spatial = 2 * 2;
        let mut dropped = 0;
        for c in 0..1000 {
            let start = c * spatial;
            if data[start] == 0.0 {
                dropped += 1;
            }
        }
        let rate = dropped as f64 / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "dropout2d rate = {rate}, expected ~0.5"
        );
    }

    #[test]
    fn test_dropout2d_eval_is_identity() {
        let mut d = Dropout2d::<f32>::new(0.5).unwrap();
        d.eval();
        let input = ferrotorch_core::ones::<f32>(&[2, 3, 4, 4]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout2d_invalid_p() {
        assert!(Dropout2d::<f32>::new(1.0).is_err());
        assert!(Dropout2d::<f32>::new(-0.1).is_err());
    }

    #[test]
    fn test_dropout2d_requires_2d_input() {
        let d = Dropout2d::<f32>::new(0.3).unwrap();
        let input_1d = ferrotorch_core::ones::<f32>(&[10]).unwrap();
        assert!(d.forward(&input_1d).is_err());
    }

    #[test]
    fn test_dropout2d_backward_routes_through_surviving_channels() {
        let d = Dropout2d::<f32>::new(0.5).unwrap();
        // [1, 20, 3, 3]
        let input = leaf_tensor(&[1.0; 1 * 20 * 3 * 3], &[1, 20, 3, 3], true);
        let output = d.forward(&input).unwrap();

        let out_data = output.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();

        #[derive(Debug)]
        struct SumBackward<T: Float> {
            input: Tensor<T>,
        }
        impl<T: Float> GradFn<T> for SumBackward<T> {
            fn backward(
                &self,
                _grad_output: &Tensor<T>,
            ) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
                let ones = vec![<T as num_traits::One>::one(); self.input.numel()];
                let t = Tensor::from_storage(
                    TensorStorage::cpu(ones),
                    self.input.shape().to_vec(),
                    false,
                )?;
                Ok(vec![Some(t)])
            }
            fn inputs(&self) -> Vec<&Tensor<T>> {
                vec![&self.input]
            }
            fn name(&self) -> &'static str {
                "SumBackward"
            }
        }

        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward {
                input: output.clone(),
            }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();
        let out_data = output.data().unwrap();

        // Gradient mask must match output mask.
        for (i, (&o, &g)) in out_data.iter().zip(grad_data.iter()).enumerate() {
            assert_eq!(
                o == 0.0,
                g == 0.0,
                "mismatch at index {i}: output={o}, grad={g}"
            );
        }

        // Gradients should be channel-uniform.
        let spatial = 3 * 3;
        for c in 0..20 {
            let start = c * spatial;
            let end = start + spatial;
            let channel_grad = &grad_data[start..end];
            let first = channel_grad[0];
            assert!(
                channel_grad.iter().all(|&g| (g - first).abs() < 1e-6),
                "gradient channel {c} is not uniform"
            );
        }
    }

    #[test]
    fn test_dropout2d_no_parameters() {
        let d = Dropout2d::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
        assert!(d.named_parameters().is_empty());
    }

    #[test]
    fn test_dropout2d_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dropout2d<f32>>();
        assert_send_sync::<Dropout2d<f64>>();
    }
}
