//! Dropout regularization layers.
//!
//! [`Dropout`] randomly zeroes individual elements during training with
//! probability `p`, scaling surviving elements by `1/(1-p)` (inverted
//! dropout). [`Dropout1d`], [`Dropout2d`], and [`Dropout3d`] drop entire
//! channels instead of individual elements, for 3D, 4D, and 5D inputs
//! respectively. [`AlphaDropout`] preserves mean and variance for use
//! with SELU activations.
//!
//! All modules are identity in eval mode and have zero learnable parameters.

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::gpu_dispatch::GpuRngState;
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

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
// Philox 4x32-10 for CPU-side mask regeneration
// ---------------------------------------------------------------------------
// We need the Philox algorithm on CPU to regenerate dropout masks during
// backward for GPU tensors (the forward mask was generated on GPU using
// the Philox state). This is a copy of the core algorithm from
// ferrotorch-gpu/src/rng.rs to avoid a dependency on the GPU crate.

#[allow(dead_code)]
const PHILOX_M0: u32 = 0xD2511F53;
#[allow(dead_code)]
const PHILOX_M1: u32 = 0xCD9E8D57;
#[allow(dead_code)]
const PHILOX_W0: u32 = 0x9E3779B9;
#[allow(dead_code)]
const PHILOX_W1: u32 = 0xBB67AE85;

#[allow(dead_code)]
#[inline]
fn philox_round(c0: u32, c1: u32, c2: u32, c3: u32, k0: u32, k1: u32) -> (u32, u32, u32, u32) {
    let prod0 = (PHILOX_M0 as u64) * (c0 as u64);
    let hi0 = (prod0 >> 32) as u32;
    let lo0 = prod0 as u32;

    let prod1 = (PHILOX_M1 as u64) * (c2 as u64);
    let hi1 = (prod1 >> 32) as u32;
    let lo1 = prod1 as u32;

    let new_c0 = hi1 ^ c1 ^ k0;
    let new_c1 = lo1;
    let new_c2 = hi0 ^ c3 ^ k1;
    let new_c3 = lo0;

    (new_c0, new_c1, new_c2, new_c3)
}

/// Philox 4x32-10: produces 4 uniform u32 values from (counter, key).
#[allow(dead_code)]
fn philox_4x32_10(counter: u64, key: u64) -> [u32; 4] {
    let mut c0 = counter as u32;
    let mut c1 = (counter >> 32) as u32;
    let mut c2 = 0u32;
    let mut c3 = 0u32;

    let mut k0 = key as u32;
    let mut k1 = (key >> 32) as u32;

    for _ in 0..9 {
        (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);
        k0 = k0.wrapping_add(PHILOX_W0);
        k1 = k1.wrapping_add(PHILOX_W1);
    }
    // Round 10 (final, no key advance)
    (c0, c1, c2, c3) = philox_round(c0, c1, c2, c3, k0, k1);

    [c0, c1, c2, c3]
}

/// Generate a dropout mask using the Philox algorithm, matching the GPU kernel's
/// behavior. The mask uses `(counter ^ seed)` as a derived u32 seed and applies
/// the same xorshift-multiply hash that the GPU dropout kernel uses.
///
/// This ensures backward mask matches the forward mask generated on GPU.
fn philox_dropout_mask<T: Float>(
    numel: usize,
    threshold: u32,
    scale: T,
    rng_state: &GpuRngState,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let derived_seed = (rng_state.counter ^ rng_state.seed) as u32;

    (0..numel)
        .map(|i| {
            let mut r = (i as u32).wrapping_mul(2654435761) ^ derived_seed;
            r ^= r << 13;
            r ^= r >> 17;
            r ^= r << 5;
            if r < threshold { zero } else { scale }
        })
        .collect()
}

// ---------------------------------------------------------------------------
// DropoutBackward
// ---------------------------------------------------------------------------

/// Backward node for elementwise dropout.
///
/// Reapplies the same binary mask scaled by `1/(1-p)` to the upstream
/// gradient, routing gradients only through surviving elements.
///
/// The mask is stored as a [`Tensor<T>`] on the same device as the
/// forward input so backward reduces to a single `mul` that stays
/// GPU-native when the input is on CUDA.
#[derive(Debug)]
struct DropoutBackward<T: Float> {
    input: Tensor<T>,
    /// Mask tensor with elements in `{0, 1/(1-p)}`. Lives on the same
    /// device as `input`, so `mul(grad_output, scaled_mask)` in the
    /// backward routes entirely through GPU ops when training on CUDA.
    scaled_mask: Tensor<T>,
}

impl<T: Float> GradFn<T> for DropoutBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.input.requires_grad() {
            let g = ferrotorch_core::grad_fns::arithmetic::mul(grad_output, &self.scaled_mask)?;
            Some(g)
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
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "dropout2d backward",
            });
        }
        let da = if self.input.requires_grad() {
            let go_data = grad_output.data_vec()?;
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
            Some(g)
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

        // GPU fast path: run dropout kernel entirely on device using the
        // Philox CBRNG. This integrates with the global GPU RNG state so
        // that gradient checkpointing can reproduce identical masks.
        if input.is_cuda() {
            if let Some(backend) = ferrotorch_core::gpu_dispatch::gpu_backend() {
                let threshold = (self.p * u32::MAX as f64) as u32;
                let scale_f32 = 1.0f32 / (1.0 - self.p as f32);

                let (handle, rng_state) =
                    backend.dropout_philox_f32(input.gpu_handle()?, threshold, scale_f32)?;

                // For backward, we need the mask. Regenerate it from the saved
                // Philox RNG state using the same deterministic hash that the
                // GPU kernel uses. This is reproducible across checkpoint
                // save/restore because the Philox state is deterministic.
                if is_grad_enabled() && input.requires_grad() {
                    let scaled_mask_vec = philox_dropout_mask(numel, threshold, scale, &rng_state);
                    // Upload the mask to the input's device so the
                    // backward `mul` runs on-device without a CPU
                    // round-trip.
                    let mask_cpu = Tensor::from_storage(
                        TensorStorage::cpu(scaled_mask_vec),
                        input.shape().to_vec(),
                        false,
                    )?;
                    let scaled_mask = mask_cpu.to(input.device())?;
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
        let scaled_mask_vec: Vec<T> = (0..numel)
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
            .zip(scaled_mask_vec.iter())
            .map(|(&x, &m)| x * m)
            .collect();

        if is_grad_enabled() && input.requires_grad() {
            let scaled_mask = Tensor::from_storage(
                TensorStorage::cpu(scaled_mask_vec),
                input.shape().to_vec(),
                false,
            )?;
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(DropoutBackward {
                    input: input.clone(),
                    scaled_mask,
                }),
            )
        } else {
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )
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
        // Product of empty slice is 1, so no special case needed for 2-D inputs.
        let spatial: usize = shape[2..].iter().product();

        let numel = input.numel();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        // GPU tensors are not yet supported for Dropout2d — needs a fused
        // channel-broadcast dropout kernel.
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "Dropout2d" });
        }

        // CPU path.
        // Generate per-channel keep/drop decisions.
        let mut state = xorshift_seed();
        let channel_mask: Vec<bool> = (0..batch * channels)
            .map(|_| xorshift_next(&mut state) >= self.p)
            .collect();

        // Expand channel mask to full element mask.
        let scaled_mask: Vec<T> = {
            let mut mask = Vec::with_capacity(numel);
            for &cm in &channel_mask {
                let val = if cm { scale } else { zero };
                for _ in 0..spatial {
                    mask.push(val);
                }
            }
            mask
        };

        let input_data = input.data_vec()?;
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
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )?
        };
        Ok(result)
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
// Dropout1d — CL-433
// ===========================================================================

/// Randomly zeroes entire 1D channels with probability `p` during training.
///
/// Expects input of shape `[B, C, L]` (3 dimensions). During training,
/// each channel (the entire length-`L` slice for a given `b, c`) is
/// independently set to zero with probability `p` and surviving channels
/// are scaled by `1/(1-p)`. During evaluation the input is returned unchanged.
///
/// This is the 1D analogue of [`Dropout2d`].
///
/// Matches `torch.nn.Dropout1d`.
#[derive(Debug)]
pub struct Dropout1d<T: Float> {
    p: f64,
    training: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Dropout1d<T> {
    /// Create a new `Dropout1d` layer.
    ///
    /// `p` is the probability of an entire channel being zeroed. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("dropout1d probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: Float> Module<T> for Dropout1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dropout1d expects 3D input [B, C, L], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let length = shape[2];

        let numel = input.numel();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "Dropout1d" });
        }

        let mut state = xorshift_seed();
        let channel_mask: Vec<bool> = (0..batch * channels)
            .map(|_| xorshift_next(&mut state) >= self.p)
            .collect();

        let scaled_mask: Vec<T> = {
            let mut mask = Vec::with_capacity(numel);
            for &cm in &channel_mask {
                let val = if cm { scale } else { zero };
                for _ in 0..length {
                    mask.push(val);
                }
            }
            mask
        };

        let input_data = input.data_vec()?;
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
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )?
        };
        Ok(result)
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
// Dropout3d — CL-433
// ===========================================================================

/// Randomly zeroes entire 3D channels with probability `p` during training.
///
/// Expects input of shape `[B, C, D, H, W]` (5 dimensions). During training,
/// each channel (the entire `D * H * W` volume for a given `b, c`) is
/// independently set to zero with probability `p` and surviving channels
/// are scaled by `1/(1-p)`. During evaluation the input is returned unchanged.
///
/// Matches `torch.nn.Dropout3d`.
#[derive(Debug)]
pub struct Dropout3d<T: Float> {
    p: f64,
    training: bool,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Dropout3d<T> {
    /// Create a new `Dropout3d` layer.
    ///
    /// `p` is the probability of an entire channel being zeroed. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("dropout3d probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: Float> Module<T> for Dropout3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        let shape = input.shape();
        if shape.len() != 5 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Dropout3d expects 5D input [B, C, D, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let batch = shape[0];
        let channels = shape[1];
        let spatial: usize = shape[2..].iter().product();

        let numel = input.numel();
        let scale = T::from(1.0 / (1.0 - self.p)).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "Dropout3d" });
        }

        let mut state = xorshift_seed();
        let channel_mask: Vec<bool> = (0..batch * channels)
            .map(|_| xorshift_next(&mut state) >= self.p)
            .collect();

        let scaled_mask: Vec<T> = {
            let mut mask = Vec::with_capacity(numel);
            for &cm in &channel_mask {
                let val = if cm { scale } else { zero };
                for _ in 0..spatial {
                    mask.push(val);
                }
            }
            mask
        };

        let input_data = input.data_vec()?;
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
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )?
        };
        Ok(result)
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
// AlphaDropout — CL-433
// ===========================================================================

/// Alpha Dropout for use with SELU activations.
///
/// Unlike standard dropout, `AlphaDropout` preserves the self-normalizing
/// property of SELU by maintaining the mean and variance of the input.
/// Dropped elements are set to the SELU saturation value rather than zero,
/// and the output is affinely transformed to restore the original mean and
/// variance.
///
/// During training:
/// 1. Elements are dropped with probability `p`.
/// 2. Dropped elements are set to `alpha' = -lambda * alpha` (the SELU
///    saturation value).
/// 3. The result is affinely transformed: `output = a * (masked + alpha' * drop_mask) + b`
///    where `a` and `b` are chosen to preserve the input's mean and variance.
///
/// During evaluation, the input is returned unchanged.
///
/// Matches `torch.nn.AlphaDropout`.
#[derive(Debug)]
pub struct AlphaDropout<T: Float> {
    p: f64,
    training: bool,
    _marker: std::marker::PhantomData<T>,
}

// SELU constants (from PyTorch / Klambauer et al. 2017).
const SELU_ALPHA: f64 = 1.6732632423543772;
const SELU_LAMBDA: f64 = 1.0507009873554805;

impl<T: Float> AlphaDropout<T> {
    /// Create a new `AlphaDropout` layer.
    ///
    /// `p` is the probability of an element being dropped. Must be in `[0, 1)`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("alpha_dropout probability must be in [0, 1), got {p}"),
            });
        }
        Ok(Self {
            p,
            training: true,
            _marker: std::marker::PhantomData,
        })
    }
}

/// Backward node for AlphaDropout.
///
/// The affine correction factor `a` is baked into the scaled_mask:
/// surviving elements get `a`, dropped elements get `0`.
/// Gradient routing: grad_input = grad_output * scaled_mask.
#[derive(Debug)]
struct AlphaDropoutBackward<T: Float> {
    input: Tensor<T>,
    /// Mask with `a` for kept elements and `0` for dropped elements.
    grad_mask: Vec<T>,
}

impl<T: Float> GradFn<T> for AlphaDropoutBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if grad_output.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "AlphaDropout backward",
            });
        }
        let da = if self.input.requires_grad() {
            let go_data = grad_output.data_vec()?;
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(self.grad_mask.iter())
                .map(|(&g, &m)| g * m)
                .collect();
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.input.shape().to_vec(),
                false,
            )?;
            Some(g)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AlphaDropoutBackward"
    }
}

impl<T: Float> Module<T> for AlphaDropout<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if !self.training || self.p == 0.0 {
            return Ok(input.clone());
        }

        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda { op: "AlphaDropout" });
        }

        let numel = input.numel();
        let p = self.p;

        // The SELU saturation value for dropped elements.
        let alpha_prime = -SELU_LAMBDA * SELU_ALPHA;

        // Affine correction factors to preserve mean and variance.
        // a = (1 / sqrt(q + alpha'^2 * p * q))  where q = 1 - p
        // b = -a * (p * alpha')
        let q = 1.0 - p;
        let a_f64 = 1.0 / (q + alpha_prime * alpha_prime * p * q).sqrt();
        let b_f64 = -a_f64 * p * alpha_prime;

        let a = T::from(a_f64).unwrap();
        let b = T::from(b_f64).unwrap();
        let alpha_prime_t = T::from(alpha_prime).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        // Generate drop mask.
        let mut state = xorshift_seed();
        let keep: Vec<bool> = (0..numel).map(|_| xorshift_next(&mut state) >= p).collect();

        let input_data = input.data()?;
        let mut output_data = Vec::with_capacity(numel);
        let mut grad_mask = Vec::with_capacity(numel);

        for (i, &x) in input_data.iter().enumerate() {
            if keep[i] {
                // Kept element: a * x + b
                output_data.push(a * x + b);
                grad_mask.push(a);
            } else {
                // Dropped element: a * alpha' + b
                output_data.push(a * alpha_prime_t + b);
                grad_mask.push(zero);
            }
        }

        if is_grad_enabled() && input.requires_grad() {
            Tensor::from_operation(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                Arc::new(AlphaDropoutBackward {
                    input: input.clone(),
                    grad_mask,
                }),
            )
        } else {
            Tensor::from_storage(
                TensorStorage::cpu(output_data),
                input.shape().to_vec(),
                false,
            )
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
        let input = leaf_tensor(&[1.0; 20 * 3 * 3], &[1, 20, 3, 3], true);
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

    // -----------------------------------------------------------------------
    // Dropout1d — CL-433
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout1d_drops_whole_channels() {
        let d = Dropout1d::<f32>::new(0.5).unwrap();
        // Shape: [2, 10, 8] — 2 batches, 10 channels, length 8.
        let input = ferrotorch_core::ones::<f32>(&[2, 10, 8]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let length = 8;
        for b in 0..2 {
            for c in 0..10 {
                let start = (b * 10 + c) * length;
                let end = start + length;
                let channel = &data[start..end];

                let first = channel[0];
                assert!(
                    channel.iter().all(|&x| (x - first).abs() < 1e-6),
                    "channel (b={b}, c={c}) is not uniform"
                );
                assert!(
                    first == 0.0 || (first - 2.0).abs() < 1e-6,
                    "channel value = {first}, expected 0.0 or 2.0"
                );
            }
        }
    }

    #[test]
    fn test_dropout1d_rate_approximately_correct() {
        let d = Dropout1d::<f32>::new(0.5).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[1, 1000, 4]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let length = 4;
        let mut dropped = 0;
        for c in 0..1000 {
            if data[c * length] == 0.0 {
                dropped += 1;
            }
        }
        let rate = dropped as f64 / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "dropout1d rate = {rate}, expected ~0.5"
        );
    }

    #[test]
    fn test_dropout1d_eval_is_identity() {
        let mut d = Dropout1d::<f32>::new(0.5).unwrap();
        d.eval();
        let input = ferrotorch_core::ones::<f32>(&[2, 3, 8]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout1d_invalid_p() {
        assert!(Dropout1d::<f32>::new(1.0).is_err());
        assert!(Dropout1d::<f32>::new(-0.1).is_err());
    }

    #[test]
    fn test_dropout1d_requires_3d_input() {
        let d = Dropout1d::<f32>::new(0.3).unwrap();
        let input_2d = ferrotorch_core::ones::<f32>(&[10, 5]).unwrap();
        assert!(d.forward(&input_2d).is_err());
    }

    #[test]
    fn test_dropout1d_no_parameters() {
        let d = Dropout1d::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
    }

    #[test]
    fn test_dropout1d_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dropout1d<f32>>();
        assert_send_sync::<Dropout1d<f64>>();
    }

    // -----------------------------------------------------------------------
    // Dropout3d — CL-433
    // -----------------------------------------------------------------------

    #[test]
    fn test_dropout3d_drops_whole_channels() {
        let d = Dropout3d::<f32>::new(0.5).unwrap();
        // Shape: [2, 10, 2, 2, 2] — 2 batches, 10 channels, 2x2x2 spatial.
        let input = ferrotorch_core::ones::<f32>(&[2, 10, 2, 2, 2]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let spatial = 2 * 2 * 2;
        for b in 0..2 {
            for c in 0..10 {
                let start = (b * 10 + c) * spatial;
                let end = start + spatial;
                let channel = &data[start..end];

                let first = channel[0];
                assert!(
                    channel.iter().all(|&x| (x - first).abs() < 1e-6),
                    "channel (b={b}, c={c}) is not uniform"
                );
                assert!(
                    first == 0.0 || (first - 2.0).abs() < 1e-6,
                    "channel value = {first}, expected 0.0 or 2.0"
                );
            }
        }
    }

    #[test]
    fn test_dropout3d_rate_approximately_correct() {
        let d = Dropout3d::<f32>::new(0.5).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[1, 1000, 2, 2, 2]).unwrap();
        let output = d.forward(&input).unwrap();
        let data = output.data().unwrap();

        let spatial = 2 * 2 * 2;
        let mut dropped = 0;
        for c in 0..1000 {
            if data[c * spatial] == 0.0 {
                dropped += 1;
            }
        }
        let rate = dropped as f64 / 1000.0;
        assert!(
            (rate - 0.5).abs() < 0.05,
            "dropout3d rate = {rate}, expected ~0.5"
        );
    }

    #[test]
    fn test_dropout3d_eval_is_identity() {
        let mut d = Dropout3d::<f32>::new(0.5).unwrap();
        d.eval();
        let input = ferrotorch_core::ones::<f32>(&[2, 3, 2, 2, 2]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_dropout3d_invalid_p() {
        assert!(Dropout3d::<f32>::new(1.0).is_err());
        assert!(Dropout3d::<f32>::new(-0.1).is_err());
    }

    #[test]
    fn test_dropout3d_requires_5d_input() {
        let d = Dropout3d::<f32>::new(0.3).unwrap();
        let input_4d = ferrotorch_core::ones::<f32>(&[2, 3, 4, 4]).unwrap();
        assert!(d.forward(&input_4d).is_err());
    }

    #[test]
    fn test_dropout3d_no_parameters() {
        let d = Dropout3d::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
    }

    #[test]
    fn test_dropout3d_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Dropout3d<f32>>();
        assert_send_sync::<Dropout3d<f64>>();
    }

    // -----------------------------------------------------------------------
    // AlphaDropout — CL-433
    // -----------------------------------------------------------------------

    #[test]
    fn test_alpha_dropout_preserves_mean_approx() {
        // With large sample, mean should be approximately preserved.
        let d = AlphaDropout::<f64>::new(0.5).unwrap();
        // Generate input with known mean.
        let n = 100_000;
        let data: Vec<f64> = (0..n).map(|i| (i as f64 / n as f64) - 0.5).collect();
        let input_mean: f64 = data.iter().sum::<f64>() / n as f64;

        let input = Tensor::from_storage(TensorStorage::cpu(data), vec![1, n], false).unwrap();
        let output = d.forward(&input).unwrap();
        let out_data = output.data().unwrap();
        let out_mean: f64 = out_data.iter().sum::<f64>() / n as f64;

        // Mean should be roughly preserved (within statistical tolerance).
        assert!(
            (out_mean - input_mean).abs() < 0.05,
            "AlphaDropout mean = {out_mean}, input mean = {input_mean}"
        );
    }

    #[test]
    fn test_alpha_dropout_eval_is_identity() {
        let mut d = AlphaDropout::<f32>::new(0.5).unwrap();
        d.eval();
        let input = ferrotorch_core::ones::<f32>(&[100]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_alpha_dropout_zero_prob_is_identity() {
        let d = AlphaDropout::<f32>::new(0.0).unwrap();
        let input = ferrotorch_core::ones::<f32>(&[100]).unwrap();
        let output = d.forward(&input).unwrap();
        assert!(output.is_same(&input));
    }

    #[test]
    fn test_alpha_dropout_invalid_p() {
        assert!(AlphaDropout::<f32>::new(1.0).is_err());
        assert!(AlphaDropout::<f32>::new(-0.1).is_err());
        assert!(AlphaDropout::<f32>::new(1.5).is_err());
    }

    #[test]
    fn test_alpha_dropout_no_parameters() {
        let d = AlphaDropout::<f32>::new(0.3).unwrap();
        assert!(d.parameters().is_empty());
    }

    #[test]
    fn test_alpha_dropout_backward_routes_gradient() {
        let d = AlphaDropout::<f32>::new(0.5).unwrap();
        let input = leaf_tensor(&[1.0; 1000], &[1000], true);
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

        // Gradient should have two types of values: 0 for dropped, `a` for kept.
        let mut seen_zero = false;
        let mut seen_nonzero = false;
        for &g in grad_data {
            if g == 0.0 {
                seen_zero = true;
            } else {
                seen_nonzero = true;
            }
        }
        assert!(
            seen_zero,
            "some elements should have zero gradient (dropped)"
        );
        assert!(
            seen_nonzero,
            "some elements should have nonzero gradient (kept)"
        );
    }

    #[test]
    fn test_alpha_dropout_train_eval_toggle() {
        let mut d = AlphaDropout::<f32>::new(0.5).unwrap();
        assert!(d.is_training());
        d.eval();
        assert!(!d.is_training());
        d.train();
        assert!(d.is_training());
    }

    #[test]
    fn test_alpha_dropout_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<AlphaDropout<f32>>();
        assert_send_sync::<AlphaDropout<f64>>();
    }
}
