//! Activation function wrapper modules.
//!
//! Each struct is a zero-parameter [`Module`] that applies the corresponding
//! elementwise non-linearity in its [`forward`](Module::forward) method.
//! They carry a `training` flag for API consistency but their behaviour is
//! identical in train and eval modes.

use ferrotorch_core::grad_fns::activation as act;
use ferrotorch_core::grad_fns::arithmetic;
use ferrotorch_core::grad_fns::transcendental;
use ferrotorch_core::ops::elementwise::unary_map;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, normalize_axis};

use crate::module::Module;
use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// Macro: implements the full `Module` trait for a zero-parameter activation.
// ---------------------------------------------------------------------------

macro_rules! impl_activation_module {
    ($ty:ident) => {
        impl<T: Float> Module<T> for $ty {
            fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
                self.forward(input)
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
    };
}

// ===========================================================================
// ReLU
// ===========================================================================

/// Applies the rectified linear unit function elementwise:
///
/// `ReLU(x) = max(0, x)`
#[derive(Debug, Clone)]
pub struct ReLU {
    training: bool,
}

impl ReLU {
    /// Create a new `ReLU` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        act::relu(input)
    }
}

impl Default for ReLU {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(ReLU);

// ===========================================================================
// Softmax2d
// ===========================================================================

/// Applies softmax over the channel dimension of 4-D input [N, C, H, W].
///
/// `Softmax2d(x)[n, c, h, w] = exp(x[n,c,h,w]) / sum_c'(exp(x[n,c',h,w]))`
///
/// Matches PyTorch's `nn.Softmax2d`.
#[derive(Debug, Clone)]
pub struct Softmax2d {
    training: bool,
}

impl Softmax2d {
    pub fn new() -> Self {
        Self { training: true }
    }

    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 4 {
            return Err(ferrotorch_core::error::FerrotorchError::InvalidArgument {
                message: format!(
                    "Softmax2d expects 4-D input [N,C,H,W], got {:?}",
                    input.shape()
                ),
            });
        }

        if input.is_cuda() {
            return Err(
                ferrotorch_core::error::FerrotorchError::NotImplementedOnCuda { op: "Softmax2d" },
            );
        }

        let shape = input.shape();
        let n = shape[0];
        let c = shape[1];
        let h = shape[2];
        let w = shape[3];
        let data = input.data()?;
        let mut out = vec![<T as num_traits::Zero>::zero(); n * c * h * w];

        // Softmax over channel dim (dim=1) for each (n, h, w) position.
        for batch in 0..n {
            for row in 0..h {
                for col in 0..w {
                    // Find max for stability.
                    let mut max_val = T::neg_infinity();
                    for ch in 0..c {
                        let idx = batch * c * h * w + ch * h * w + row * w + col;
                        if data[idx] > max_val {
                            max_val = data[idx];
                        }
                    }
                    // Compute exp and sum.
                    let mut sum_exp = <T as num_traits::Zero>::zero();
                    for ch in 0..c {
                        let idx = batch * c * h * w + ch * h * w + row * w + col;
                        let e = (data[idx] - max_val).exp();
                        out[idx] = e;
                        sum_exp += e;
                    }
                    // Normalize.
                    for ch in 0..c {
                        let idx = batch * c * h * w + ch * h * w + row * w + col;
                        out[idx] = out[idx] / sum_exp;
                    }
                }
            }
        }

        Tensor::from_storage(
            ferrotorch_core::storage::TensorStorage::cpu(out),
            shape.to_vec(),
            false,
        )
    }
}

impl Default for Softmax2d {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(Softmax2d);

// ===========================================================================
// GELU
// ===========================================================================

pub use act::GeluApproximate;

/// Applies the Gaussian Error Linear Unit activation function.
///
/// Three approximation modes are available (see [`GeluApproximate`]):
///
/// - **`None`** (default) — exact erf-based, matches PyTorch `approximate="none"`.
/// - **`Tanh`** — tanh approximation, matches PyTorch `approximate="tanh"`.
/// - **`Sigmoid`** — fast `x * sigmoid(1.702 * x)`.
#[derive(Debug, Clone)]
pub struct GELU {
    approximate: GeluApproximate,
    training: bool,
}

impl GELU {
    /// Create a new `GELU` module with the default exact (erf) mode.
    pub fn new() -> Self {
        Self {
            approximate: GeluApproximate::default(),
            training: true,
        }
    }

    /// Create a new `GELU` module with the specified approximation mode.
    pub fn with_approximate(approximate: GeluApproximate) -> Self {
        Self {
            approximate,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        act::gelu_with(input, self.approximate)
    }
}

impl Default for GELU {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(GELU);

// ===========================================================================
// SiLU (Swish)
// ===========================================================================

/// Applies the Sigmoid Linear Unit (Swish) function:
///
/// `SiLU(x) = x * sigmoid(x)`
#[derive(Debug, Clone)]
pub struct SiLU {
    training: bool,
}

impl SiLU {
    /// Create a new `SiLU` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        act::silu(input)
    }
}

impl Default for SiLU {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(SiLU);

// ===========================================================================
// Sigmoid
// ===========================================================================

/// Applies the logistic sigmoid function elementwise:
///
/// `Sigmoid(x) = 1 / (1 + exp(-x))`
#[derive(Debug, Clone)]
pub struct Sigmoid {
    training: bool,
}

impl Sigmoid {
    /// Create a new `Sigmoid` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        act::sigmoid(input)
    }
}

impl Default for Sigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(Sigmoid);

// ===========================================================================
// Tanh
// ===========================================================================

/// Applies the hyperbolic tangent function elementwise.
#[derive(Debug, Clone)]
pub struct Tanh {
    training: bool,
}

impl Tanh {
    /// Create a new `Tanh` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        act::tanh(input)
    }
}

impl Default for Tanh {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(Tanh);

// ===========================================================================
// Softmax
// ===========================================================================

/// Applies the softmax function along a given dimension.
///
/// Currently only the last axis (`dim = -1`) is supported because the
/// underlying `ferrotorch_core::grad_fns::activation::softmax` operates on
/// the last axis. Passing any other dimension returns an error.
#[derive(Debug, Clone)]
pub struct Softmax {
    /// The dimension along which to compute softmax.
    pub dim: isize,
    training: bool,
}

impl Softmax {
    /// Create a new `Softmax` module operating along `dim`.
    ///
    /// Defaults to `dim = -1` (last axis), matching PyTorch convention.
    pub fn new(dim: isize) -> Self {
        Self {
            dim,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let ndim = input.ndim();
        if ndim == 0 {
            // Scalar: softmax is always 1.
            return act::softmax(input);
        }

        let axis = normalize_axis(self.dim, ndim)?;
        if axis != ndim - 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Softmax currently only supports dim=-1 (last axis), \
                     but got dim={} (axis={}) for a {}-D tensor",
                    self.dim, axis, ndim,
                ),
            });
        }

        act::softmax(input)
    }
}

impl Default for Softmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl_activation_module!(Softmax);

// ===========================================================================
// LogSoftmax
// ===========================================================================

/// Applies log(softmax(x)) along a given dimension.
///
/// More numerically stable than computing `log(softmax(x))` separately.
/// Currently only the last axis (`dim = -1`) is supported.
#[derive(Debug, Clone)]
pub struct LogSoftmax {
    /// The dimension along which to compute log-softmax.
    pub dim: isize,
    training: bool,
}

impl LogSoftmax {
    /// Create a new `LogSoftmax` module operating along `dim`.
    pub fn new(dim: isize) -> Self {
        Self {
            dim,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let ndim = input.ndim();
        if ndim == 0 {
            return act::log_softmax(input);
        }

        let axis = normalize_axis(self.dim, ndim)?;
        if axis != ndim - 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LogSoftmax currently only supports dim=-1 (last axis), \
                     but got dim={} (axis={}) for a {}-D tensor",
                    self.dim, axis, ndim,
                ),
            });
        }

        act::log_softmax(input)
    }
}

impl Default for LogSoftmax {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl_activation_module!(LogSoftmax);

// ===========================================================================
// LeakyReLU
// ===========================================================================

/// Applies the leaky rectified linear unit function:
///
/// `LeakyReLU(x) = max(0, x) + negative_slope * min(0, x)`
///
/// This is implemented by composing differentiable primitives so that
/// autograd works automatically:
///
/// ```text
/// forward(x) = (1 - negative_slope) * relu(x) + negative_slope * x
/// ```
#[derive(Debug, Clone)]
pub struct LeakyReLU {
    /// Slope for negative inputs. Default: 0.01.
    pub negative_slope: f64,
    training: bool,
}

impl LeakyReLU {
    /// Create a new `LeakyReLU` with the given negative slope.
    pub fn new(negative_slope: f64) -> Self {
        Self {
            negative_slope,
            training: true,
        }
    }

    /// Forward pass.
    ///
    /// Computes `(1 - negative_slope) * relu(x) + negative_slope * x`
    /// using differentiable core operations so gradients propagate correctly.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if (self.negative_slope - 0.0).abs() < f64::EPSILON {
            // Degenerate case: standard ReLU.
            return act::relu(input);
        }
        if (self.negative_slope - 1.0).abs() < f64::EPSILON {
            // Degenerate case: identity.
            return Ok(input.clone());
        }

        // relu_x = relu(input)
        let relu_x = act::relu(input)?;

        // scale = (1 - negative_slope)
        let scale = T::from(1.0 - self.negative_slope).unwrap();
        let slope = T::from(self.negative_slope).unwrap();

        // scale_tensor = scalar(1 - negative_slope)
        let scale_tensor = ferrotorch_core::scalar(scale)?;
        // slope_tensor = scalar(negative_slope)
        let slope_tensor = ferrotorch_core::scalar(slope)?;

        // result = scale * relu(x) + slope * x
        let scaled_relu = arithmetic::mul(&relu_x, &scale_tensor)?;
        let scaled_x = arithmetic::mul(input, &slope_tensor)?;
        arithmetic::add(&scaled_relu, &scaled_x)
    }
}

impl Default for LeakyReLU {
    fn default() -> Self {
        Self::new(0.01)
    }
}

impl_activation_module!(LeakyReLU);

// ===========================================================================
// ELU
// ===========================================================================

/// Applies the Exponential Linear Unit function:
///
/// ```text
/// ELU(x) = x            if x > 0
///        = alpha * (exp(x) - 1)  if x <= 0
/// ```
///
/// Differentiable: autograd backward is supported via `EluBackward`.
#[derive(Debug, Clone)]
pub struct ELU {
    /// Scale for the negative region. Default: 1.0.
    pub alpha: f64,
    training: bool,
}

impl ELU {
    /// Create a new `ELU` module with the given alpha.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        act::elu(input, self.alpha)
    }
}

impl Default for ELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl_activation_module!(ELU);

// ===========================================================================
// Mish
// ===========================================================================

/// Applies the Mish activation function:
///
/// `Mish(x) = x * tanh(softplus(x))`
///
/// where `softplus(x) = ln(1 + exp(x))`.
///
/// Differentiable: autograd backward is supported via `MishBackward`.
#[derive(Debug, Clone)]
pub struct Mish {
    training: bool,
}

impl Mish {
    /// Create a new `Mish` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        act::mish(input)
    }
}

impl Default for Mish {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(Mish);

// ===========================================================================
// PReLU (Parametric ReLU)
// ===========================================================================

/// Parametric Rectified Linear Unit with a learnable negative slope.
///
/// `PReLU(x) = max(0, x) + alpha * min(0, x)`
///
/// where `alpha` is a learnable [`Parameter`]. This is equivalent to
/// `(1 - alpha) * relu(x) + alpha * x` for differentiable composition.
#[derive(Debug, Clone)]
pub struct PReLU<T: Float> {
    /// Learnable negative slope parameter.
    pub alpha: Parameter<T>,
    training: bool,
}

impl<T: Float> PReLU<T> {
    /// Create a new `PReLU` module with the given initial negative slope.
    pub fn new(init_alpha: f64) -> FerrotorchResult<Self> {
        let alpha_val = T::from(init_alpha).unwrap();
        let alpha_tensor = ferrotorch_core::from_slice(&[alpha_val], &[1])?;
        Ok(Self {
            alpha: Parameter::new(alpha_tensor),
            training: true,
        })
    }

    /// Forward pass.
    ///
    /// Computes `prelu(x, alpha) = max(0, x) + alpha * min(0, x)` via the
    /// native fused [`act::prelu`] op (single forward, single backward).
    pub fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if self.alpha.tensor().is_cuda() {
            return Err(
                ferrotorch_core::error::FerrotorchError::NotImplementedOnCuda { op: "PReLU" },
            );
        }
        act::prelu(input, self.alpha.tensor())
    }
}

impl<T: Float> Module<T> for PReLU<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![&self.alpha]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![&mut self.alpha]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![("alpha".to_string(), &self.alpha)]
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
// CELU
// ===========================================================================

/// Continuously Differentiable Exponential Linear Unit:
///
/// ```text
/// CELU(x) = max(0, x) + min(0, alpha * (exp(x / alpha) - 1))
/// ```
///
/// Unlike ELU, CELU is continuously differentiable everywhere.
#[derive(Debug, Clone)]
pub struct CELU {
    /// Scale for the negative region. Default: 1.0.
    pub alpha: f64,
    training: bool,
}

impl CELU {
    /// Create a new `CELU` module with the given alpha.
    pub fn new(alpha: f64) -> Self {
        Self {
            alpha,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let alpha = T::from(self.alpha).unwrap();

        unary_map(input, |x| {
            let pos = if x > zero { x } else { zero };
            let neg = if x < zero {
                alpha * ((x / alpha).exp() - one)
            } else {
                zero
            };
            pos + neg
        })
    }
}

impl Default for CELU {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl_activation_module!(CELU);

// ===========================================================================
// SELU
// ===========================================================================

/// Scaled Exponential Linear Unit with fixed constants:
///
/// ```text
/// SELU(x) = lambda * (x                    if x > 0)
///         = lambda * (alpha * (exp(x) - 1)  if x <= 0)
/// ```
///
/// where `alpha = 1.6732632423543772` and `lambda = 1.0507009873554805`.
/// These constants enable self-normalizing behaviour when used with
/// properly initialized weights (LeCun normal).
#[derive(Debug, Clone)]
pub struct SELU {
    training: bool,
}

/// SELU alpha constant.
const SELU_ALPHA: f64 = 1.6732632423543772;
/// SELU lambda (scale) constant.
const SELU_LAMBDA: f64 = 1.0507009873554805;

impl SELU {
    /// Create a new `SELU` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let alpha = T::from(SELU_ALPHA).unwrap();
        let lambda = T::from(SELU_LAMBDA).unwrap();

        unary_map(input, |x| {
            if x > zero {
                lambda * x
            } else {
                lambda * alpha * (x.exp() - one)
            }
        })
    }
}

impl Default for SELU {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(SELU);

// ===========================================================================
// HardSigmoid
// ===========================================================================

/// Hard Sigmoid activation:
///
/// `HardSigmoid(x) = clamp((x + 3) / 6, 0, 1)`
///
/// A piecewise-linear approximation of the sigmoid function.
#[derive(Debug, Clone)]
pub struct HardSigmoid {
    training: bool,
}

impl HardSigmoid {
    /// Create a new `HardSigmoid` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let three = T::from(3.0).unwrap();
        let six = T::from(6.0).unwrap();

        unary_map(input, |x| {
            let v = (x + three) / six;
            if v < zero {
                zero
            } else if v > one {
                one
            } else {
                v
            }
        })
    }
}

impl Default for HardSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(HardSigmoid);

// ===========================================================================
// HardSwish
// ===========================================================================

/// Hard Swish activation:
///
/// `HardSwish(x) = x * HardSigmoid(x) = x * clamp((x + 3) / 6, 0, 1)`
///
/// A piecewise-linear approximation of SiLU (Swish).
#[derive(Debug, Clone)]
pub struct HardSwish {
    training: bool,
}

impl HardSwish {
    /// Create a new `HardSwish` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let zero = <T as num_traits::Zero>::zero();
        let one = <T as num_traits::One>::one();
        let three = T::from(3.0).unwrap();
        let six = T::from(6.0).unwrap();

        unary_map(input, |x| {
            let hard_sig = {
                let v = (x + three) / six;
                if v < zero {
                    zero
                } else if v > one {
                    one
                } else {
                    v
                }
            };
            x * hard_sig
        })
    }
}

impl Default for HardSwish {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(HardSwish);

// ===========================================================================
// Softplus
// ===========================================================================

/// Softplus activation:
///
/// `Softplus(x) = log(1 + exp(beta * x)) / beta`
///
/// A smooth approximation of ReLU. As `beta` increases, Softplus converges
/// to ReLU.
#[derive(Debug, Clone)]
pub struct Softplus {
    /// Sharpness parameter. Default: 1.0.
    pub beta: f64,
    /// Threshold above which the function reverts to a linear function
    /// for numerical stability. Default: 20.0.
    pub threshold: f64,
    training: bool,
}

impl Softplus {
    /// Create a new `Softplus` module with the given beta.
    pub fn new(beta: f64) -> Self {
        Self {
            beta,
            threshold: 20.0,
            training: true,
        }
    }

    /// Forward pass.
    ///
    /// Differentiable: autograd backward is supported via `SoftplusBackward`.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        act::softplus(input, self.beta, self.threshold)
    }
}

impl Default for Softplus {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl_activation_module!(Softplus);

// ===========================================================================
// GLU (Gated Linear Unit)
// ===========================================================================

/// Gated Linear Unit:
///
/// `GLU(x) = a * sigmoid(b)`
///
/// where `a` and `b` are the two halves of the input split along the last
/// dimension. The input's last dimension must be even.
///
/// Reference: *Language Modeling with Gated Convolutional Networks* (Dauphin et al., 2017).
#[derive(Debug, Clone)]
pub struct GLU {
    training: bool,
}

impl GLU {
    /// Create a new `GLU` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    ///
    /// Splits the input along the last dimension into two equal halves,
    /// then computes `first_half * sigmoid(second_half)`.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape();
        let ndim = shape.len();
        if ndim == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "GLU requires at least 1D input".to_string(),
            });
        }

        let last_dim = shape[ndim - 1];
        if last_dim % 2 != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "GLU requires the last dimension to be even, got {}",
                    last_dim
                ),
            });
        }

        let half = last_dim / 2;
        let device = input.device();
        let data = input.data_vec()?;

        // Compute the stride of the last dimension (number of elements per
        // "row" in the last dimension).
        let outer_size: usize = shape[..ndim - 1].iter().product();
        let outer_size = if outer_size == 0 { 1 } else { outer_size };

        let one = <T as num_traits::One>::one();

        let mut result = Vec::with_capacity(outer_size * half);
        for i in 0..outer_size {
            let base = i * last_dim;
            for j in 0..half {
                let a = data[base + j];
                let b = data[base + half + j];
                let sig_b = one / (one + (-b).exp());
                result.push(a * sig_b);
            }
        }

        let mut out_shape = shape.to_vec();
        out_shape[ndim - 1] = half;

        let out = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(result),
            out_shape,
            false,
        )?;
        if device.is_cuda() {
            out.to(device)
        } else {
            Ok(out)
        }
    }
}

impl Default for GLU {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(GLU);

// ===========================================================================
// ReLU6
// ===========================================================================

/// Applies `ReLU6(x) = min(max(0, x), 6)` elementwise.
///
/// A ReLU clamped to `[0, 6]`, commonly used in MobileNet architectures.
///
/// Differentiable: uses [`transcendental::clamp`] which tracks gradients.
#[derive(Debug, Clone)]
pub struct ReLU6 {
    training: bool,
}

impl ReLU6 {
    /// Create a new `ReLU6` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let zero = <T as num_traits::Zero>::zero();
        let six = T::from(6.0).unwrap();
        transcendental::clamp(input, zero, six)
    }
}

impl Default for ReLU6 {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(ReLU6);

// ===========================================================================
// Hardtanh
// ===========================================================================

/// Applies the hard tanh function elementwise:
///
/// ```text
/// Hardtanh(x) = min_val  if x < min_val
///             = max_val  if x > max_val
///             = x        otherwise
/// ```
///
/// Differentiable: uses [`transcendental::clamp`] which tracks gradients.
#[derive(Debug, Clone)]
pub struct Hardtanh {
    /// Minimum value. Default: -1.0.
    pub min_val: f64,
    /// Maximum value. Default: 1.0.
    pub max_val: f64,
    training: bool,
}

impl Hardtanh {
    /// Create a new `Hardtanh` module with the given min and max values.
    pub fn new(min_val: f64, max_val: f64) -> Self {
        Self {
            min_val,
            max_val,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let min = T::from(self.min_val).unwrap();
        let max = T::from(self.max_val).unwrap();
        transcendental::clamp(input, min, max)
    }
}

impl Default for Hardtanh {
    fn default() -> Self {
        Self::new(-1.0, 1.0)
    }
}

impl_activation_module!(Hardtanh);

// ===========================================================================
// LogSigmoid
// ===========================================================================

/// Applies `LogSigmoid(x) = log(sigmoid(x))` elementwise.
///
/// Numerically stable: implemented as `-softplus(-x)` to avoid overflow.
///
/// Differentiable: composes differentiable primitives (softplus, neg).
#[derive(Debug, Clone)]
pub struct LogSigmoid {
    training: bool,
}

impl LogSigmoid {
    /// Create a new `LogSigmoid` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    ///
    /// Uses the identity `log(sigmoid(x)) = -softplus(-x)` for numerical
    /// stability (avoids computing `exp(x)` for large positive `x`).
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // log(sigmoid(x)) = log(1/(1+exp(-x))) = -log(1+exp(-x)) = -softplus(-x)
        let neg_input = arithmetic::neg(input)?;
        let sp = act::softplus(&neg_input, 1.0, 20.0)?;
        arithmetic::neg(&sp)
    }
}

impl Default for LogSigmoid {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(LogSigmoid);

// ===========================================================================
// Softmin
// ===========================================================================

/// Applies `Softmin(x) = Softmax(-x)` along a given dimension.
///
/// Reverses the ordering: the smallest input gets the largest probability.
/// Currently only the last axis (`dim = -1`) is supported.
#[derive(Debug, Clone)]
pub struct Softmin {
    /// The dimension along which to compute softmin.
    pub dim: isize,
    training: bool,
}

impl Softmin {
    /// Create a new `Softmin` module operating along `dim`.
    pub fn new(dim: isize) -> Self {
        Self {
            dim,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let ndim = input.ndim();
        if ndim == 0 {
            let neg_input = arithmetic::neg(input)?;
            return act::softmax(&neg_input);
        }

        let axis = normalize_axis(self.dim, ndim)?;
        if axis != ndim - 1 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Softmin currently only supports dim=-1 (last axis), \
                     but got dim={} (axis={}) for a {}-D tensor",
                    self.dim, axis, ndim,
                ),
            });
        }

        let neg_input = arithmetic::neg(input)?;
        act::softmax(&neg_input)
    }
}

impl Default for Softmin {
    fn default() -> Self {
        Self::new(-1)
    }
}

impl_activation_module!(Softmin);

// ===========================================================================
// Threshold
// ===========================================================================

/// Applies the threshold function:
///
/// ```text
/// Threshold(x) = x      if x > threshold
///              = value   otherwise
/// ```
///
/// Matches PyTorch `nn.Threshold(threshold, value)`.
#[derive(Debug, Clone)]
pub struct Threshold {
    /// Threshold value.
    pub threshold: f64,
    /// Replacement value for inputs at or below the threshold.
    pub value: f64,
    training: bool,
}

impl Threshold {
    /// Create a new `Threshold` module.
    pub fn new(threshold: f64, value: f64) -> Self {
        Self {
            threshold,
            value,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let thresh = T::from(self.threshold).unwrap();
        let val = T::from(self.value).unwrap();
        unary_map(input, |x| if x > thresh { x } else { val })
    }
}

impl_activation_module!(Threshold);

// ===========================================================================
// Softshrink
// ===========================================================================

/// Applies the soft shrinkage function elementwise:
///
/// ```text
/// Softshrink(x) = x - lambda  if x > lambda
///               = x + lambda  if x < -lambda
///               = 0           otherwise
/// ```
///
/// Default `lambda = 0.5`.
#[derive(Debug, Clone)]
pub struct Softshrink {
    /// Shrinkage threshold. Default: 0.5.
    pub lambda: f64,
    training: bool,
}

impl Softshrink {
    /// Create a new `Softshrink` module with the given lambda.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let lam = T::from(self.lambda).unwrap();
        let neg_lam = T::from(-self.lambda).unwrap();
        let zero = <T as num_traits::Zero>::zero();
        unary_map(input, |x| {
            if x > lam {
                x - lam
            } else if x < neg_lam {
                x + lam
            } else {
                zero
            }
        })
    }
}

impl Default for Softshrink {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl_activation_module!(Softshrink);

// ===========================================================================
// Hardshrink
// ===========================================================================

/// Applies the hard shrinkage function elementwise:
///
/// ```text
/// Hardshrink(x) = x  if x > lambda  or  x < -lambda
///               = 0  otherwise
/// ```
///
/// Default `lambda = 0.5`.
#[derive(Debug, Clone)]
pub struct Hardshrink {
    /// Shrinkage threshold. Default: 0.5.
    pub lambda: f64,
    training: bool,
}

impl Hardshrink {
    /// Create a new `Hardshrink` module with the given lambda.
    pub fn new(lambda: f64) -> Self {
        Self {
            lambda,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let lam = T::from(self.lambda).unwrap();
        let neg_lam = T::from(-self.lambda).unwrap();
        let zero = <T as num_traits::Zero>::zero();
        unary_map(input, |x| if x > lam || x < neg_lam { x } else { zero })
    }
}

impl Default for Hardshrink {
    fn default() -> Self {
        Self::new(0.5)
    }
}

impl_activation_module!(Hardshrink);

// ===========================================================================
// Tanhshrink
// ===========================================================================

/// Applies `Tanhshrink(x) = x - tanh(x)` elementwise.
///
/// Differentiable: composes differentiable primitives (tanh, sub).
#[derive(Debug, Clone)]
pub struct Tanhshrink {
    training: bool,
}

impl Tanhshrink {
    /// Create a new `Tanhshrink` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let tanh_x = act::tanh(input)?;
        arithmetic::sub(input, &tanh_x)
    }
}

impl Default for Tanhshrink {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(Tanhshrink);

// ===========================================================================
// Softsign
// ===========================================================================

/// Applies `Softsign(x) = x / (1 + |x|)` elementwise.
///
/// A smooth, bounded activation similar to tanh but with lighter tails.
#[derive(Debug, Clone)]
pub struct Softsign {
    training: bool,
}

impl Softsign {
    /// Create a new `Softsign` module.
    pub fn new() -> Self {
        Self { training: true }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let one = <T as num_traits::One>::one();
        unary_map(input, |x| x / (one + x.abs()))
    }
}

impl Default for Softsign {
    fn default() -> Self {
        Self::new()
    }
}

impl_activation_module!(Softsign);

// ===========================================================================
// RReLU (Randomized Leaky ReLU)
// ===========================================================================

/// Applies the Randomized Leaky ReLU function:
///
/// ```text
/// RReLU(x) = x                              if x >= 0
///          = a * x  (a ~ Uniform[lower, upper])  if x < 0   (training)
///          = ((lower + upper) / 2) * x       if x < 0       (eval)
/// ```
///
/// In training mode, each negative element gets an independent random slope
/// drawn from `Uniform(lower, upper)`. In eval mode, the deterministic mean
/// slope `(lower + upper) / 2` is used.
///
/// Default: `lower = 1/8`, `upper = 1/3`, matching PyTorch.
#[derive(Debug, Clone)]
pub struct RReLU {
    /// Lower bound for the random slope. Default: 1/8.
    pub lower: f64,
    /// Upper bound for the random slope. Default: 1/3.
    pub upper: f64,
    training: bool,
}

/// Seed a xorshift64 state from system time and thread id.
fn rrelu_xorshift_seed() -> u64 {
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
fn rrelu_xorshift_next(state: &mut u64) -> f64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    (*state as f64) / (u64::MAX as f64)
}

impl RReLU {
    /// Create a new `RReLU` module with the given lower and upper bounds.
    pub fn new(lower: f64, upper: f64) -> Self {
        Self {
            lower,
            upper,
            training: true,
        }
    }

    /// Forward pass.
    pub fn forward<T: Float>(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let zero = <T as num_traits::Zero>::zero();

        if self.training {
            // Stochastic: per-element random slope in [lower, upper].
            // Use Cell for interior mutability since unary_map requires Fn.
            let rng_state = std::cell::Cell::new(rrelu_xorshift_seed());
            let lower = self.lower;
            let upper = self.upper;
            let range = upper - lower;

            unary_map(input, |x| {
                if x >= zero {
                    x
                } else {
                    let mut st = rng_state.get();
                    let u = rrelu_xorshift_next(&mut st);
                    rng_state.set(st);
                    let slope = T::from(lower + u * range).unwrap();
                    slope * x
                }
            })
        } else {
            // Deterministic: mean slope.
            let mean_slope = T::from((self.lower + self.upper) / 2.0).unwrap();
            unary_map(input, |x| if x >= zero { x } else { mean_slope * x })
        }
    }
}

impl Default for RReLU {
    fn default() -> Self {
        Self::new(1.0 / 8.0, 1.0 / 3.0)
    }
}

impl_activation_module!(RReLU);

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    /// Helper: 1-D tensor from a slice (no grad).
    fn t(data: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
    }

    /// Helper: 2-D tensor (no grad).
    fn t2d(data: &[f64], rows: usize, cols: usize) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![rows, cols], false).unwrap()
    }

    // -----------------------------------------------------------------------
    // Module trait compliance helpers
    // -----------------------------------------------------------------------

    /// Verify that a module has zero parameters and responds to train/eval.
    fn assert_zero_param_module<M, T: Float>(module: &mut M)
    where
        M: Module<T>,
    {
        assert!(module.parameters().is_empty(), "should have no parameters");
        assert!(
            module.parameters_mut().is_empty(),
            "should have no mutable parameters"
        );
        assert!(
            module.named_parameters().is_empty(),
            "should have no named parameters"
        );
        assert!(module.is_training(), "default should be training mode");
        module.eval();
        assert!(!module.is_training(), "eval() should set training=false");
        module.train();
        assert!(module.is_training(), "train() should set training=true");
    }

    // -----------------------------------------------------------------------
    // ReLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu_forward() {
        let m = ReLU::new();
        let x = t(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-7);
        assert!((d[1] - 0.0).abs() < 1e-7);
        assert!((d[2] - 0.0).abs() < 1e-7);
        assert!((d[3] - 1.0).abs() < 1e-7);
        assert!((d[4] - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_relu_module_trait() {
        let mut m = ReLU::new();
        assert_zero_param_module::<ReLU, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // GELU
    // -----------------------------------------------------------------------

    #[test]
    fn test_gelu_forward() {
        let m = GELU::new();
        // gelu(0) = 0
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        assert!(y.data().unwrap()[0].abs() < 1e-7);

        // gelu(x) > 0 for x > 0
        let x = t(&[1.0, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!(d[0] > 0.0);
        assert!(d[1] > 0.0);

        // gelu(x) is close to x for large positive x
        let x = t(&[10.0]);
        let y = m.forward(&x).unwrap();
        assert!((y.data().unwrap()[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_gelu_module_trait() {
        let mut m = GELU::new();
        assert_zero_param_module::<GELU, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // SiLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_silu_forward() {
        let m = SiLU::new();
        // silu(0) = 0 * sigmoid(0) = 0
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        assert!(y.data().unwrap()[0].abs() < 1e-7);

        // silu(x) = x * sigmoid(x); for large x, sigmoid(x) -> 1 so silu(x) -> x
        let x = t(&[10.0]);
        let y = m.forward(&x).unwrap();
        assert!((y.data().unwrap()[0] - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_silu_module_trait() {
        let mut m = SiLU::new();
        assert_zero_param_module::<SiLU, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Sigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn test_sigmoid_forward() {
        let m = Sigmoid::new();
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        assert!((y.data().unwrap()[0] - 0.5).abs() < 1e-7);

        let x = t(&[-100.0, 100.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!(d[0] < 1e-10, "sigmoid(-100) should be ~0");
        assert!((d[1] - 1.0).abs() < 1e-10, "sigmoid(100) should be ~1");
    }

    #[test]
    fn test_sigmoid_module_trait() {
        let mut m = Sigmoid::new();
        assert_zero_param_module::<Sigmoid, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Tanh
    // -----------------------------------------------------------------------

    #[test]
    fn test_tanh_forward() {
        let m = Tanh::new();
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        assert!(y.data().unwrap()[0].abs() < 1e-7);

        let x = t(&[-100.0, 100.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] + 1.0).abs() < 1e-10, "tanh(-100) should be ~-1");
        assert!((d[1] - 1.0).abs() < 1e-10, "tanh(100) should be ~1");
    }

    #[test]
    fn test_tanh_module_trait() {
        let mut m = Tanh::new();
        assert_zero_param_module::<Tanh, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Softmax
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmax_forward_1d() {
        let m = Softmax::new(-1);
        let x = t(&[1.0, 2.0, 3.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        // Sum should be 1.
        let total: f64 = d.iter().sum();
        assert!((total - 1.0).abs() < 1e-7);

        // Monotonicity.
        assert!(d[0] < d[1]);
        assert!(d[1] < d[2]);
    }

    #[test]
    fn test_softmax_forward_2d() {
        let m = Softmax::new(-1);
        // [[1, 2], [3, 4]]
        let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        // Each row should sum to 1.
        let row0_sum = d[0] + d[1];
        let row1_sum = d[2] + d[3];
        assert!((row0_sum - 1.0).abs() < 1e-7);
        assert!((row1_sum - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_softmax_wrong_dim() {
        let m = Softmax::new(0);
        let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        // dim=0 is not the last axis for a 2-D tensor, should error.
        assert!(m.forward(&x).is_err());
    }

    #[test]
    fn test_softmax_module_trait() {
        let mut m = Softmax::new(-1);
        assert_zero_param_module::<Softmax, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // LogSoftmax
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_softmax_forward_1d() {
        let m = LogSoftmax::new(-1);
        let x = t(&[1.0, 2.0, 3.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        // exp(log_softmax) should sum to 1.
        let total: f64 = d.iter().map(|&v| v.exp()).sum();
        assert!((total - 1.0).abs() < 1e-7, "exp(log_softmax) sum = {total}");

        // All log-probabilities should be negative.
        assert!(d.iter().all(|&v| v <= 0.0));
    }

    #[test]
    fn test_log_softmax_module_trait() {
        let mut m = LogSoftmax::new(-1);
        assert_zero_param_module::<LogSoftmax, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // LeakyReLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_leaky_relu_forward() {
        let m = LeakyReLU::new(0.01);
        let x = t(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        assert!((d[0] - (-0.02)).abs() < 1e-7, "LeakyReLU(-2) = {}", d[0]);
        assert!((d[1] - (-0.01)).abs() < 1e-7, "LeakyReLU(-1) = {}", d[1]);
        assert!((d[2] - 0.0).abs() < 1e-7, "LeakyReLU(0) = {}", d[2]);
        assert!((d[3] - 1.0).abs() < 1e-7, "LeakyReLU(1) = {}", d[3]);
        assert!((d[4] - 2.0).abs() < 1e-7, "LeakyReLU(2) = {}", d[4]);
    }

    #[test]
    fn test_leaky_relu_large_slope() {
        let m = LeakyReLU::new(0.2);
        let x = t(&[-5.0, 3.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        assert!(
            (d[0] - (-1.0)).abs() < 1e-7,
            "LeakyReLU(-5, slope=0.2) = {}",
            d[0]
        );
        assert!(
            (d[1] - 3.0).abs() < 1e-7,
            "LeakyReLU(3, slope=0.2) = {}",
            d[1]
        );
    }

    #[test]
    fn test_leaky_relu_zero_slope_is_relu() {
        let m = LeakyReLU::new(0.0);
        let x = t(&[-2.0, 0.0, 3.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        assert!((d[0] - 0.0).abs() < 1e-7);
        assert!((d[1] - 0.0).abs() < 1e-7);
        assert!((d[2] - 3.0).abs() < 1e-7);
    }

    #[test]
    fn test_leaky_relu_module_trait() {
        let mut m = LeakyReLU::new(0.01);
        assert_zero_param_module::<LeakyReLU, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // ELU
    // -----------------------------------------------------------------------

    #[test]
    fn test_elu_forward() {
        let m = ELU::new(1.0);
        let x = t(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        // For x > 0, ELU(x) = x.
        assert!((d[3] - 1.0).abs() < 1e-7);
        assert!((d[4] - 2.0).abs() < 1e-7);

        // For x = 0, ELU(0) = 0.
        assert!((d[2] - 0.0).abs() < 1e-7);

        // For x < 0, ELU(x) = alpha * (exp(x) - 1) < 0.
        let expected_m1 = 1.0 * ((-1.0_f64).exp() - 1.0);
        assert!(
            (d[1] - expected_m1).abs() < 1e-7,
            "ELU(-1) expected {}, got {}",
            expected_m1,
            d[1]
        );

        let expected_m2 = 1.0 * ((-2.0_f64).exp() - 1.0);
        assert!(
            (d[0] - expected_m2).abs() < 1e-7,
            "ELU(-2) expected {}, got {}",
            expected_m2,
            d[0]
        );

        // ELU approaches -alpha from below for very negative x.
        let x = t(&[-100.0]);
        let y = m.forward(&x).unwrap();
        assert!((y.data().unwrap()[0] + 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_elu_custom_alpha() {
        let m = ELU::new(2.0);
        let x = t(&[-1.0, 1.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        let expected = 2.0 * ((-1.0_f64).exp() - 1.0);
        assert!((d[0] - expected).abs() < 1e-7);
        assert!((d[1] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_elu_module_trait() {
        let mut m = ELU::new(1.0);
        assert_zero_param_module::<ELU, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Mish
    // -----------------------------------------------------------------------

    #[test]
    fn test_mish_forward() {
        let m = Mish::new();
        // mish(0) = 0 * tanh(softplus(0)) = 0 * tanh(ln(2)) = 0
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        assert!(y.data().unwrap()[0].abs() < 1e-7, "mish(0) should be 0");

        // For large positive x, mish(x) -> x (softplus(x) -> x, tanh(x) -> 1).
        let x = t(&[20.0]);
        let y = m.forward(&x).unwrap();
        assert!(
            (y.data().unwrap()[0] - 20.0).abs() < 0.01,
            "mish(20) should be ~20"
        );

        // mish is slightly negative for negative inputs.
        let x = t(&[-1.0]);
        let y = m.forward(&x).unwrap();
        let val = y.data().unwrap()[0];
        let softplus = (1.0 + (-1.0_f64).exp()).ln();
        let expected = -softplus.tanh();
        assert!(
            (val - expected).abs() < 1e-7,
            "mish(-1) expected {}, got {}",
            expected,
            val
        );
    }

    #[test]
    fn test_mish_module_trait() {
        let mut m = Mish::new();
        assert_zero_param_module::<Mish, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Default constructors
    // -----------------------------------------------------------------------

    // -----------------------------------------------------------------------
    // PReLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_prelu_forward_default() {
        let m = PReLU::<f64>::new(0.25).unwrap();
        let x = t(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        // For x > 0: output = x. For x < 0: output = 0.25 * x.
        assert!((d[0] - (-0.5)).abs() < 1e-6, "PReLU(-2) = {}", d[0]);
        assert!((d[1] - (-0.25)).abs() < 1e-6, "PReLU(-1) = {}", d[1]);
        assert!((d[2] - 0.0).abs() < 1e-6, "PReLU(0) = {}", d[2]);
        assert!((d[3] - 1.0).abs() < 1e-6, "PReLU(1) = {}", d[3]);
        assert!((d[4] - 2.0).abs() < 1e-6, "PReLU(2) = {}", d[4]);
    }

    #[test]
    fn test_prelu_has_parameter() {
        let m = PReLU::<f64>::new(0.25).unwrap();
        assert_eq!(m.parameters().len(), 1, "PReLU should have 1 parameter");
        let named = m.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "alpha");
    }

    #[test]
    fn test_prelu_module_trait() {
        let mut m = PReLU::<f64>::new(0.25).unwrap();
        assert_eq!(m.parameters().len(), 1);
        assert!(m.is_training());
        m.eval();
        assert!(!m.is_training());
        m.train();
        assert!(m.is_training());
    }

    // -----------------------------------------------------------------------
    // CELU
    // -----------------------------------------------------------------------

    #[test]
    fn test_celu_forward() {
        let m = CELU::new(1.0);
        let x = t(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        // For x > 0: CELU(x) = x
        assert!((d[3] - 1.0).abs() < 1e-7);
        assert!((d[4] - 2.0).abs() < 1e-7);
        assert!((d[2] - 0.0).abs() < 1e-7);

        // For x < 0: CELU(x) = alpha * (exp(x/alpha) - 1)
        let expected_m1 = 1.0 * ((-1.0_f64).exp() - 1.0);
        assert!((d[1] - expected_m1).abs() < 1e-7, "CELU(-1) = {}", d[1]);
    }

    #[test]
    fn test_celu_module_trait() {
        let mut m = CELU::new(1.0);
        assert_zero_param_module::<CELU, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // SELU
    // -----------------------------------------------------------------------

    #[test]
    fn test_selu_forward() {
        let m = SELU::new();
        let x = t(&[-1.0, 0.0, 1.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        // For x > 0: SELU(x) = lambda * x
        let lambda = 1.0507009873554805_f64;
        let alpha = 1.6732632423543772_f64;
        assert!((d[2] - lambda * 1.0).abs() < 1e-7, "SELU(1) = {}", d[2]);
        assert!((d[1] - 0.0).abs() < 1e-7, "SELU(0) = {}", d[1]);

        // For x < 0: SELU(x) = lambda * alpha * (exp(x) - 1)
        let expected_m1 = lambda * alpha * ((-1.0_f64).exp() - 1.0);
        assert!((d[0] - expected_m1).abs() < 1e-7, "SELU(-1) = {}", d[0]);
    }

    #[test]
    fn test_selu_module_trait() {
        let mut m = SELU::new();
        assert_zero_param_module::<SELU, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // HardSigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn test_hard_sigmoid_forward() {
        let m = HardSigmoid::new();
        // clamp((x+3)/6, 0, 1)
        // x = -4: (−4+3)/6 = −1/6 < 0 -> 0
        // x = -3: (-3+3)/6 = 0
        // x = 0: (0+3)/6 = 0.5
        // x = 3: (3+3)/6 = 1.0
        // x = 5: (5+3)/6 > 1 -> 1
        let x = t(&[-4.0, -3.0, 0.0, 3.0, 5.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-7, "HardSigmoid(-4) = {}", d[0]);
        assert!((d[1] - 0.0).abs() < 1e-7, "HardSigmoid(-3) = {}", d[1]);
        assert!((d[2] - 0.5).abs() < 1e-7, "HardSigmoid(0) = {}", d[2]);
        assert!((d[3] - 1.0).abs() < 1e-7, "HardSigmoid(3) = {}", d[3]);
        assert!((d[4] - 1.0).abs() < 1e-7, "HardSigmoid(5) = {}", d[4]);
    }

    #[test]
    fn test_hard_sigmoid_module_trait() {
        let mut m = HardSigmoid::new();
        assert_zero_param_module::<HardSigmoid, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // HardSwish
    // -----------------------------------------------------------------------

    #[test]
    fn test_hard_swish_forward() {
        let m = HardSwish::new();
        // HardSwish(x) = x * clamp((x+3)/6, 0, 1)
        // x = -4: -4 * 0 = 0
        // x = 0: 0 * 0.5 = 0
        // x = 3: 3 * 1.0 = 3
        // x = 5: 5 * 1.0 = 5
        // x = -1: -1 * ((-1+3)/6) = -1 * (1/3) = -1/3
        let x = t(&[-4.0, 0.0, 3.0, 5.0, -1.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-7, "HardSwish(-4) = {}", d[0]);
        assert!((d[1] - 0.0).abs() < 1e-7, "HardSwish(0) = {}", d[1]);
        assert!((d[2] - 3.0).abs() < 1e-7, "HardSwish(3) = {}", d[2]);
        assert!((d[3] - 5.0).abs() < 1e-7, "HardSwish(5) = {}", d[3]);
        assert!(
            (d[4] - (-1.0 / 3.0)).abs() < 1e-7,
            "HardSwish(-1) = {}",
            d[4]
        );
    }

    #[test]
    fn test_hard_swish_module_trait() {
        let mut m = HardSwish::new();
        assert_zero_param_module::<HardSwish, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Softplus
    // -----------------------------------------------------------------------

    #[test]
    fn test_softplus_forward() {
        let m = Softplus::new(1.0);
        // softplus(0) = ln(1 + 1) = ln(2)
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - 2.0_f64.ln()).abs() < 1e-7, "Softplus(0) = {}", d[0]);

        // For large x, softplus(x) -> x (threshold mode).
        let x = t(&[25.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - 25.0).abs() < 1e-5, "Softplus(25) = {}", d[0]);

        // softplus(1) = ln(1 + e) ~ 1.3133
        let x = t(&[1.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        let expected = (1.0 + 1.0_f64.exp()).ln();
        assert!((d[0] - expected).abs() < 1e-7, "Softplus(1) = {}", d[0]);
    }

    #[test]
    fn test_softplus_custom_beta() {
        let m = Softplus::new(2.0);
        // softplus(x, beta=2) = ln(1 + exp(2*x)) / 2
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        let expected = 2.0_f64.ln() / 2.0;
        assert!(
            (d[0] - expected).abs() < 1e-7,
            "Softplus(0, beta=2) = {}",
            d[0]
        );
    }

    #[test]
    fn test_softplus_module_trait() {
        let mut m = Softplus::new(1.0);
        assert_zero_param_module::<Softplus, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // GLU
    // -----------------------------------------------------------------------

    #[test]
    fn test_glu_forward_1d() {
        let m = GLU::new();
        // input = [1.0, 0.0, 2.0, 0.0]  (last dim = 4, split into [1,0] and [2,0])
        // a = [1.0, 0.0], b = [2.0, 0.0]
        // output = a * sigmoid(b) = [1.0 * sigmoid(2.0), 0.0 * sigmoid(0.0)]
        let x = t(&[1.0, 0.0, 2.0, 0.0]);
        let y = m.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2]);
        let d = y.data().unwrap();
        let sig_2 = 1.0 / (1.0 + (-2.0_f64).exp());
        assert!((d[0] - sig_2).abs() < 1e-7, "GLU[0] = {}", d[0]);
        assert!((d[1] - 0.0).abs() < 1e-7, "GLU[1] = {}", d[1]);
    }

    #[test]
    fn test_glu_forward_2d() {
        let m = GLU::new();
        // [[1.0, 0.0, 2.0, 0.0]] -> shape [1, 4], splits last dim
        let x = t2d(&[1.0, 0.0, 2.0, 0.0], 1, 4);
        let y = m.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 2]);
        let d = y.data().unwrap();
        let sig_2 = 1.0 / (1.0 + (-2.0_f64).exp());
        assert!((d[0] - sig_2).abs() < 1e-7);
        assert!((d[1] - 0.0).abs() < 1e-7);
    }

    #[test]
    fn test_glu_odd_dim_error() {
        let m = GLU::new();
        let x = t(&[1.0, 2.0, 3.0]); // last dim = 3 (odd)
        assert!(m.forward(&x).is_err());
    }

    #[test]
    fn test_glu_module_trait() {
        let mut m = GLU::new();
        assert_zero_param_module::<GLU, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // ReLU6
    // -----------------------------------------------------------------------

    #[test]
    fn test_relu6_forward() {
        let m = ReLU6::new();
        let x = t(&[-2.0, 0.0, 3.0, 6.0, 10.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-7, "ReLU6(-2) = {}", d[0]);
        assert!((d[1] - 0.0).abs() < 1e-7, "ReLU6(0) = {}", d[1]);
        assert!((d[2] - 3.0).abs() < 1e-7, "ReLU6(3) = {}", d[2]);
        assert!((d[3] - 6.0).abs() < 1e-7, "ReLU6(6) = {}", d[3]);
        assert!((d[4] - 6.0).abs() < 1e-7, "ReLU6(10) = {}", d[4]);
    }

    #[test]
    fn test_relu6_module_trait() {
        let mut m = ReLU6::new();
        assert_zero_param_module::<ReLU6, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Hardtanh
    // -----------------------------------------------------------------------

    #[test]
    fn test_hardtanh_forward_default() {
        let m = Hardtanh::default();
        // clamp(x, -1, 1)
        let x = t(&[-5.0, -1.0, 0.0, 0.5, 1.0, 3.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - (-1.0)).abs() < 1e-7, "Hardtanh(-5) = {}", d[0]);
        assert!((d[1] - (-1.0)).abs() < 1e-7, "Hardtanh(-1) = {}", d[1]);
        assert!((d[2] - 0.0).abs() < 1e-7, "Hardtanh(0) = {}", d[2]);
        assert!((d[3] - 0.5).abs() < 1e-7, "Hardtanh(0.5) = {}", d[3]);
        assert!((d[4] - 1.0).abs() < 1e-7, "Hardtanh(1) = {}", d[4]);
        assert!((d[5] - 1.0).abs() < 1e-7, "Hardtanh(3) = {}", d[5]);
    }

    #[test]
    fn test_hardtanh_custom_range() {
        let m = Hardtanh::new(-2.0, 2.0);
        let x = t(&[-5.0, -2.0, 0.0, 2.0, 5.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - (-2.0)).abs() < 1e-7);
        assert!((d[1] - (-2.0)).abs() < 1e-7);
        assert!((d[2] - 0.0).abs() < 1e-7);
        assert!((d[3] - 2.0).abs() < 1e-7);
        assert!((d[4] - 2.0).abs() < 1e-7);
    }

    #[test]
    fn test_hardtanh_module_trait() {
        let mut m = Hardtanh::default();
        assert_zero_param_module::<Hardtanh, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // LogSigmoid
    // -----------------------------------------------------------------------

    #[test]
    fn test_log_sigmoid_forward() {
        let m = LogSigmoid::new();
        // log(sigmoid(0)) = log(0.5) = -ln(2)
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!(
            (d[0] - (-2.0_f64.ln())).abs() < 1e-6,
            "LogSigmoid(0) = {}, expected {}",
            d[0],
            -2.0_f64.ln()
        );

        // All outputs should be <= 0 (log of a probability).
        let x = t(&[-10.0, -1.0, 0.0, 1.0, 10.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!(
            d.iter().all(|&v| v <= 0.0),
            "All LogSigmoid values should be <= 0"
        );

        // For large positive x, log(sigmoid(x)) -> 0.
        assert!(
            d[4].abs() < 1e-4,
            "LogSigmoid(10) should be ~0, got {}",
            d[4]
        );

        // For large negative x, log(sigmoid(x)) -> x.
        assert!(
            (d[0] - (-10.0)).abs() < 0.1,
            "LogSigmoid(-10) should be ~-10, got {}",
            d[0]
        );
    }

    #[test]
    fn test_log_sigmoid_module_trait() {
        let mut m = LogSigmoid::new();
        assert_zero_param_module::<LogSigmoid, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Softmin
    // -----------------------------------------------------------------------

    #[test]
    fn test_softmin_forward_1d() {
        let m = Softmin::new(-1);
        let x = t(&[1.0, 2.0, 3.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        // Sum should be 1.
        let total: f64 = d.iter().sum();
        assert!((total - 1.0).abs() < 1e-7, "Softmin sum = {}", total);

        // Softmin reverses ordering: smallest input gets largest probability.
        assert!(d[0] > d[1], "softmin(1) > softmin(2)");
        assert!(d[1] > d[2], "softmin(2) > softmin(3)");
    }

    #[test]
    fn test_softmin_wrong_dim() {
        let m = Softmin::new(0);
        let x = t2d(&[1.0, 2.0, 3.0, 4.0], 2, 2);
        assert!(m.forward(&x).is_err());
    }

    #[test]
    fn test_softmin_module_trait() {
        let mut m = Softmin::new(-1);
        assert_zero_param_module::<Softmin, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Threshold
    // -----------------------------------------------------------------------

    #[test]
    fn test_threshold_forward() {
        let m = Threshold::new(0.5, -1.0);
        let x = t(&[-1.0, 0.0, 0.5, 1.0, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        // x <= threshold -> value
        assert!((d[0] - (-1.0)).abs() < 1e-7, "Threshold(-1) = {}", d[0]);
        assert!((d[1] - (-1.0)).abs() < 1e-7, "Threshold(0) = {}", d[1]);
        assert!((d[2] - (-1.0)).abs() < 1e-7, "Threshold(0.5) = {}", d[2]);
        // x > threshold -> x
        assert!((d[3] - 1.0).abs() < 1e-7, "Threshold(1) = {}", d[3]);
        assert!((d[4] - 2.0).abs() < 1e-7, "Threshold(2) = {}", d[4]);
    }

    #[test]
    fn test_threshold_module_trait() {
        let mut m = Threshold::new(0.5, -1.0);
        assert_zero_param_module::<Threshold, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Softshrink
    // -----------------------------------------------------------------------

    #[test]
    fn test_softshrink_forward() {
        let m = Softshrink::default(); // lambda = 0.5
        let x = t(&[-2.0, -0.5, -0.3, 0.0, 0.3, 0.5, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        // x > lambda: x - lambda
        assert!((d[6] - 1.5).abs() < 1e-7, "Softshrink(2) = {}", d[6]);
        // x < -lambda: x + lambda
        assert!((d[0] - (-1.5)).abs() < 1e-7, "Softshrink(-2) = {}", d[0]);
        // -lambda <= x <= lambda: 0
        assert!((d[2] - 0.0).abs() < 1e-7, "Softshrink(-0.3) = {}", d[2]);
        assert!((d[3] - 0.0).abs() < 1e-7, "Softshrink(0) = {}", d[3]);
        assert!((d[4] - 0.0).abs() < 1e-7, "Softshrink(0.3) = {}", d[4]);
        // Boundary: x == lambda or x == -lambda -> 0
        assert!((d[1] - 0.0).abs() < 1e-7, "Softshrink(-0.5) = {}", d[1]);
        assert!((d[5] - 0.0).abs() < 1e-7, "Softshrink(0.5) = {}", d[5]);
    }

    #[test]
    fn test_softshrink_custom_lambda() {
        let m = Softshrink::new(1.0);
        let x = t(&[-2.0, -0.5, 0.5, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - (-1.0)).abs() < 1e-7);
        assert!((d[1] - 0.0).abs() < 1e-7);
        assert!((d[2] - 0.0).abs() < 1e-7);
        assert!((d[3] - 1.0).abs() < 1e-7);
    }

    #[test]
    fn test_softshrink_module_trait() {
        let mut m = Softshrink::default();
        assert_zero_param_module::<Softshrink, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Hardshrink
    // -----------------------------------------------------------------------

    #[test]
    fn test_hardshrink_forward() {
        let m = Hardshrink::default(); // lambda = 0.5
        let x = t(&[-2.0, -0.5, -0.3, 0.0, 0.3, 0.5, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        // |x| > lambda: x
        assert!((d[0] - (-2.0)).abs() < 1e-7, "Hardshrink(-2) = {}", d[0]);
        assert!((d[6] - 2.0).abs() < 1e-7, "Hardshrink(2) = {}", d[6]);
        // |x| <= lambda: 0
        assert!((d[2] - 0.0).abs() < 1e-7, "Hardshrink(-0.3) = {}", d[2]);
        assert!((d[3] - 0.0).abs() < 1e-7, "Hardshrink(0) = {}", d[3]);
        assert!((d[4] - 0.0).abs() < 1e-7, "Hardshrink(0.3) = {}", d[4]);
        // Boundary: x == lambda or x == -lambda -> 0
        assert!((d[1] - 0.0).abs() < 1e-7, "Hardshrink(-0.5) = {}", d[1]);
        assert!((d[5] - 0.0).abs() < 1e-7, "Hardshrink(0.5) = {}", d[5]);
    }

    #[test]
    fn test_hardshrink_module_trait() {
        let mut m = Hardshrink::default();
        assert_zero_param_module::<Hardshrink, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Tanhshrink
    // -----------------------------------------------------------------------

    #[test]
    fn test_tanhshrink_forward() {
        let m = Tanhshrink::new();
        // tanhshrink(0) = 0 - tanh(0) = 0
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        assert!(
            y.data().unwrap()[0].abs() < 1e-7,
            "Tanhshrink(0) should be 0"
        );

        // For large |x|, tanh(x) -> sign(x), so tanhshrink(x) -> x - sign(x).
        let x = t(&[10.0, -10.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!(
            (d[0] - 9.0).abs() < 0.01,
            "Tanhshrink(10) should be ~9, got {}",
            d[0]
        );
        assert!(
            (d[1] - (-9.0)).abs() < 0.01,
            "Tanhshrink(-10) should be ~-9, got {}",
            d[1]
        );

        // Exact check: tanhshrink(1) = 1 - tanh(1)
        let x = t(&[1.0]);
        let y = m.forward(&x).unwrap();
        let expected = 1.0 - 1.0_f64.tanh();
        assert!(
            (y.data().unwrap()[0] - expected).abs() < 1e-7,
            "Tanhshrink(1) expected {}, got {}",
            expected,
            y.data().unwrap()[0]
        );
    }

    #[test]
    fn test_tanhshrink_module_trait() {
        let mut m = Tanhshrink::new();
        assert_zero_param_module::<Tanhshrink, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Softsign
    // -----------------------------------------------------------------------

    #[test]
    fn test_softsign_forward() {
        let m = Softsign::new();
        // softsign(0) = 0
        let x = t(&[0.0]);
        let y = m.forward(&x).unwrap();
        assert!(y.data().unwrap()[0].abs() < 1e-7, "Softsign(0) should be 0");

        // softsign(1) = 1/2 = 0.5
        let x = t(&[1.0]);
        let y = m.forward(&x).unwrap();
        assert!(
            (y.data().unwrap()[0] - 0.5).abs() < 1e-7,
            "Softsign(1) should be 0.5"
        );

        // softsign(-1) = -1/2 = -0.5
        let x = t(&[-1.0]);
        let y = m.forward(&x).unwrap();
        assert!(
            (y.data().unwrap()[0] - (-0.5)).abs() < 1e-7,
            "Softsign(-1) should be -0.5"
        );

        // Bounded in (-1, 1) for large values.
        let x = t(&[100.0, -100.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!(
            d[0] > 0.99 && d[0] < 1.0,
            "Softsign(100) should be ~1, got {}",
            d[0]
        );
        assert!(
            d[1] < -0.99 && d[1] > -1.0,
            "Softsign(-100) should be ~-1, got {}",
            d[1]
        );
    }

    #[test]
    fn test_softsign_module_trait() {
        let mut m = Softsign::new();
        assert_zero_param_module::<Softsign, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // RReLU
    // -----------------------------------------------------------------------

    #[test]
    #[allow(clippy::field_reassign_with_default)]
    fn test_rrelu_eval_forward() {
        // In eval mode, RReLU uses deterministic mean slope.
        let mut m = RReLU::default(); // lower=1/8, upper=1/3
        m.training = false;
        let mean_slope = (1.0 / 8.0 + 1.0 / 3.0) / 2.0;

        let x = t(&[-2.0, -1.0, 0.0, 1.0, 2.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        assert!(
            (d[0] - (-2.0 * mean_slope)).abs() < 1e-7,
            "RReLU(-2,eval) = {}",
            d[0]
        );
        assert!(
            (d[1] - (-mean_slope)).abs() < 1e-7,
            "RReLU(-1,eval) = {}",
            d[1]
        );
        assert!((d[2] - 0.0).abs() < 1e-7, "RReLU(0,eval) = {}", d[2]);
        assert!((d[3] - 1.0).abs() < 1e-7, "RReLU(1,eval) = {}", d[3]);
        assert!((d[4] - 2.0).abs() < 1e-7, "RReLU(2,eval) = {}", d[4]);
    }

    #[test]
    fn test_rrelu_training_positive_passthrough() {
        // In training mode, positive values should pass through unchanged.
        let m = RReLU::default();
        let x = t(&[0.0, 1.0, 5.0, 100.0]);
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();
        assert!((d[0] - 0.0).abs() < 1e-7);
        assert!((d[1] - 1.0).abs() < 1e-7);
        assert!((d[2] - 5.0).abs() < 1e-7);
        assert!((d[3] - 100.0).abs() < 1e-7);
    }

    #[test]
    fn test_rrelu_training_negative_bounded() {
        // In training mode, negative outputs should be scaled by a slope in [lower, upper].
        let m = RReLU::new(0.1, 0.5);
        let x = t(&[-1.0; 100]); // 100 copies of -1
        let y = m.forward(&x).unwrap();
        let d = y.data().unwrap();

        for (i, &val) in d.iter().enumerate() {
            // slope * (-1) should be in [-0.5, -0.1]
            assert!(
                (-0.5 - 1e-7..=-0.1 + 1e-7).contains(&val),
                "RReLU(-1, train)[{}] = {} not in [-0.5, -0.1]",
                i,
                val
            );
        }

        // With 100 samples, we should see some variance (not all the same).
        let first = d[0];
        let has_variance = d.iter().any(|&v| (v - first).abs() > 1e-10);
        assert!(has_variance, "RReLU training should produce varying slopes");
    }

    #[test]
    fn test_rrelu_module_trait() {
        let mut m = RReLU::default();
        assert_zero_param_module::<RReLU, f64>(&mut m);
    }

    // -----------------------------------------------------------------------
    // Default constructors
    // -----------------------------------------------------------------------

    #[test]
    fn test_defaults() {
        let _relu = ReLU::default();
        let _gelu = GELU::default();
        let _silu = SiLU::default();
        let _sigmoid = Sigmoid::default();
        let _tanh = Tanh::default();
        let _softmax = Softmax::default();
        let _log_softmax = LogSoftmax::default();

        let lrelu = LeakyReLU::default();
        assert!((lrelu.negative_slope - 0.01).abs() < f64::EPSILON);

        let elu = ELU::default();
        assert!((elu.alpha - 1.0).abs() < f64::EPSILON);

        let _mish = Mish::default();

        let celu = CELU::default();
        assert!((celu.alpha - 1.0).abs() < f64::EPSILON);

        let _selu = SELU::default();
        let _hard_sigmoid = HardSigmoid::default();
        let _hard_swish = HardSwish::default();

        let softplus = Softplus::default();
        assert!((softplus.beta - 1.0).abs() < f64::EPSILON);

        let _glu = GLU::default();

        // New activations
        let _relu6 = ReLU6::default();

        let hardtanh = Hardtanh::default();
        assert!((hardtanh.min_val - (-1.0)).abs() < f64::EPSILON);
        assert!((hardtanh.max_val - 1.0).abs() < f64::EPSILON);

        let _log_sigmoid = LogSigmoid::default();
        let _softmin = Softmin::default();

        let softshrink = Softshrink::default();
        assert!((softshrink.lambda - 0.5).abs() < f64::EPSILON);

        let hardshrink = Hardshrink::default();
        assert!((hardshrink.lambda - 0.5).abs() < f64::EPSILON);

        let _tanhshrink = Tanhshrink::default();
        let _softsign = Softsign::default();

        let rrelu = RReLU::default();
        assert!((rrelu.lower - 1.0 / 8.0).abs() < f64::EPSILON);
        assert!((rrelu.upper - 1.0 / 3.0).abs() < f64::EPSILON);
    }

    // -----------------------------------------------------------------------
    // Send + Sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ReLU>();
        assert_send_sync::<GELU>();
        assert_send_sync::<SiLU>();
        assert_send_sync::<Sigmoid>();
        assert_send_sync::<Tanh>();
        assert_send_sync::<Softmax>();
        assert_send_sync::<LogSoftmax>();
        assert_send_sync::<LeakyReLU>();
        assert_send_sync::<ELU>();
        assert_send_sync::<Mish>();
        assert_send_sync::<PReLU<f64>>();
        assert_send_sync::<CELU>();
        assert_send_sync::<SELU>();
        assert_send_sync::<HardSigmoid>();
        assert_send_sync::<HardSwish>();
        assert_send_sync::<Softplus>();
        assert_send_sync::<GLU>();
        // New activations
        assert_send_sync::<ReLU6>();
        assert_send_sync::<Hardtanh>();
        assert_send_sync::<LogSigmoid>();
        assert_send_sync::<Softmin>();
        assert_send_sync::<Threshold>();
        assert_send_sync::<Softshrink>();
        assert_send_sync::<Hardshrink>();
        assert_send_sync::<Tanhshrink>();
        assert_send_sync::<Softsign>();
        assert_send_sync::<RReLU>();
    }

    // -----------------------------------------------------------------------
    // Backward (autograd) tests for Softplus, ELU, Mish
    // -----------------------------------------------------------------------

    /// Helper: 1-D tensor from a slice with `requires_grad = true`.
    fn t_grad(data: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], true).unwrap()
    }

    /// Helper: scalar leaf tensor with `requires_grad = true`.
    fn t_scalar_grad(val: f64) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap()
    }

    /// Numerical gradient via central difference: (f(x+h) - f(x-h)) / (2h).
    fn numerical_grad(f: impl Fn(f64) -> f64, x: f64) -> f64 {
        let h = 1e-5;
        (f(x + h) - f(x - h)) / (2.0 * h)
    }

    // -- Softplus backward --

    #[test]
    fn test_softplus_backward_produces_grad() {
        let x = t_scalar_grad(1.0);
        let m = Softplus::new(1.0);
        let y = m.forward(&x).unwrap();
        ferrotorch_core::backward(&y).unwrap();

        let grad = x.grad().unwrap();
        assert!(
            grad.is_some(),
            "Softplus backward should produce a gradient"
        );
    }

    #[test]
    fn test_softplus_backward_at_zero() {
        let x = t_scalar_grad(0.0);
        let m = Softplus::new(1.0);
        let y = m.forward(&x).unwrap();
        ferrotorch_core::backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d/dx softplus(0) = sigmoid(0) = 0.5
        assert!(
            (grad.item().unwrap() - 0.5).abs() < 1e-6,
            "Softplus grad at x=0: expected 0.5, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_softplus_backward_matches_numerical() {
        for &val in &[-2.0, -0.5, 0.0, 1.0, 3.0] {
            let x = t_scalar_grad(val);
            let m = Softplus::new(1.0);
            let y = m.forward(&x).unwrap();
            ferrotorch_core::backward(&y).unwrap();

            let grad = x.grad().unwrap().unwrap();
            let expected = numerical_grad(|v| (1.0 + v.exp()).ln(), val);
            assert!(
                (grad.item().unwrap() - expected).abs() < 1e-4,
                "Softplus grad at x={}: expected {}, got {}",
                val,
                expected,
                grad.item().unwrap()
            );
        }
    }

    #[test]
    fn test_softplus_backward_custom_beta() {
        let val = 1.0;
        let beta = 2.0;
        let x = t_scalar_grad(val);
        let m = Softplus::new(beta);
        let y = m.forward(&x).unwrap();
        ferrotorch_core::backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let expected = numerical_grad(|v| (1.0 + (beta * v).exp()).ln() / beta, val);
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-4,
            "Softplus grad at x={}, beta={}: expected {}, got {}",
            val,
            beta,
            expected,
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_softplus_backward_vector() {
        let x = t_grad(&[-2.0, -0.5, 0.0, 1.0, 3.0]);
        let m = Softplus::new(1.0);
        let y = m.forward(&x).unwrap();
        // Sum to get a scalar for backward.
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();

        for (i, &val) in [-2.0_f64, -0.5, 0.0, 1.0, 3.0].iter().enumerate() {
            let expected = numerical_grad(|v| (1.0 + v.exp()).ln(), val);
            assert!(
                (grad_data[i] - expected).abs() < 1e-4,
                "Softplus grad[{}] at x={}: expected {}, got {}",
                i,
                val,
                expected,
                grad_data[i]
            );
        }
    }

    // -- ELU backward --

    #[test]
    fn test_elu_backward_produces_grad() {
        let x = t_scalar_grad(-1.0);
        let m = ELU::new(1.0);
        let y = m.forward(&x).unwrap();
        ferrotorch_core::backward(&y).unwrap();

        let grad = x.grad().unwrap();
        assert!(grad.is_some(), "ELU backward should produce a gradient");
    }

    #[test]
    fn test_elu_backward_positive() {
        let x = t_scalar_grad(2.0);
        let m = ELU::new(1.0);
        let y = m.forward(&x).unwrap();
        ferrotorch_core::backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d/dx elu(x) at x=2 (positive) = 1.
        assert!(
            (grad.item().unwrap() - 1.0).abs() < 1e-6,
            "ELU grad at x=2: expected 1.0, got {}",
            grad.item().unwrap()
        );
    }

    #[test]
    fn test_elu_backward_matches_numerical() {
        let alpha = 1.0;
        for &val in &[-2.0, -1.0, -0.5, 0.5, 2.0] {
            let x = t_scalar_grad(val);
            let m = ELU::new(alpha);
            let y = m.forward(&x).unwrap();
            ferrotorch_core::backward(&y).unwrap();

            let grad = x.grad().unwrap().unwrap();
            let expected =
                numerical_grad(|v| if v > 0.0 { v } else { alpha * (v.exp() - 1.0) }, val);
            assert!(
                (grad.item().unwrap() - expected).abs() < 1e-4,
                "ELU grad at x={}: expected {}, got {}",
                val,
                expected,
                grad.item().unwrap()
            );
        }
    }

    #[test]
    fn test_elu_backward_custom_alpha() {
        let alpha = 2.0;
        let val = -0.5;
        let x = t_scalar_grad(val);
        let m = ELU::new(alpha);
        let y = m.forward(&x).unwrap();
        ferrotorch_core::backward(&y).unwrap();

        let grad = x.grad().unwrap().unwrap();
        // d/dx [alpha * (exp(x) - 1)] = alpha * exp(x)
        let expected = alpha * val.exp();
        assert!(
            (grad.item().unwrap() - expected).abs() < 1e-5,
            "ELU grad at x={}, alpha={}: expected {}, got {}",
            val,
            alpha,
            expected,
            grad.item().unwrap()
        );
    }

    // -- Mish backward --

    #[test]
    fn test_mish_backward_produces_grad() {
        let x = t_scalar_grad(1.0);
        let m = Mish::new();
        let y = m.forward(&x).unwrap();
        ferrotorch_core::backward(&y).unwrap();

        let grad = x.grad().unwrap();
        assert!(grad.is_some(), "Mish backward should produce a gradient");
    }

    #[test]
    fn test_mish_backward_matches_numerical() {
        let mish_fn = |v: f64| {
            let sp = (1.0 + v.exp()).ln();
            v * sp.tanh()
        };

        for &val in &[-2.0, -1.0, 0.0, 0.5, 1.5, 3.0] {
            let x = t_scalar_grad(val);
            let m = Mish::new();
            let y = m.forward(&x).unwrap();
            ferrotorch_core::backward(&y).unwrap();

            let grad = x.grad().unwrap().unwrap();
            let expected = numerical_grad(mish_fn, val);
            assert!(
                (grad.item().unwrap() - expected).abs() < 1e-4,
                "Mish grad at x={}: expected {}, got {}",
                val,
                expected,
                grad.item().unwrap()
            );
        }
    }

    #[test]
    fn test_mish_backward_vector() {
        let x = t_grad(&[-1.0, 0.0, 1.0, 2.0]);
        let m = Mish::new();
        let y = m.forward(&x).unwrap();
        let sum = ferrotorch_core::grad_fns::reduction::sum(&y).unwrap();
        ferrotorch_core::backward(&sum).unwrap();

        let grad = x.grad().unwrap().unwrap();
        let grad_data = grad.data().unwrap();

        let mish_fn = |v: f64| {
            let sp = (1.0 + v.exp()).ln();
            v * sp.tanh()
        };

        for (i, &val) in [-1.0_f64, 0.0, 1.0, 2.0].iter().enumerate() {
            let expected = numerical_grad(mish_fn, val);
            assert!(
                (grad_data[i] - expected).abs() < 1e-4,
                "Mish grad[{}] at x={}: expected {}, got {}",
                i,
                val,
                expected,
                grad_data[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Backward (autograd) tests for new activations
    // -----------------------------------------------------------------------

    // -- ReLU6 backward --

    #[test]
    fn test_relu6_backward_matches_numerical() {
        let relu6_fn = |v: f64| v.clamp(0.0, 6.0);

        for &val in &[-2.0, 0.5, 3.0, 5.5, 8.0] {
            let x = t_scalar_grad(val);
            let m = ReLU6::new();
            let y = m.forward(&x).unwrap();
            ferrotorch_core::backward(&y).unwrap();

            let grad = x.grad().unwrap().unwrap();
            let expected = numerical_grad(relu6_fn, val);
            assert!(
                (grad.item().unwrap() - expected).abs() < 1e-4,
                "ReLU6 grad at x={}: expected {}, got {}",
                val,
                expected,
                grad.item().unwrap()
            );
        }
    }

    // -- Hardtanh backward --

    #[test]
    fn test_hardtanh_backward_matches_numerical() {
        let hardtanh_fn = |v: f64| v.clamp(-1.0, 1.0);

        for &val in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            let x = t_scalar_grad(val);
            let m = Hardtanh::default();
            let y = m.forward(&x).unwrap();
            ferrotorch_core::backward(&y).unwrap();

            let grad = x.grad().unwrap().unwrap();
            let expected = numerical_grad(hardtanh_fn, val);
            assert!(
                (grad.item().unwrap() - expected).abs() < 1e-4,
                "Hardtanh grad at x={}: expected {}, got {}",
                val,
                expected,
                grad.item().unwrap()
            );
        }
    }

    // -- LogSigmoid backward --

    #[test]
    fn test_log_sigmoid_backward_matches_numerical() {
        let logsigmoid_fn = |v: f64| {
            // log(sigmoid(v)) = -softplus(-v) = -ln(1+exp(-v))
            -(1.0 + (-v).exp()).ln()
        };

        for &val in &[-3.0, -1.0, 0.0, 1.0, 3.0] {
            let x = t_scalar_grad(val);
            let m = LogSigmoid::new();
            let y = m.forward(&x).unwrap();
            ferrotorch_core::backward(&y).unwrap();

            let grad = x.grad().unwrap().unwrap();
            let expected = numerical_grad(logsigmoid_fn, val);
            assert!(
                (grad.item().unwrap() - expected).abs() < 1e-4,
                "LogSigmoid grad at x={}: expected {}, got {}",
                val,
                expected,
                grad.item().unwrap()
            );
        }
    }

    // -- Tanhshrink backward --

    #[test]
    fn test_tanhshrink_backward_matches_numerical() {
        let tanhshrink_fn = |v: f64| v - v.tanh();

        for &val in &[-2.0, -0.5, 0.0, 0.5, 2.0] {
            let x = t_scalar_grad(val);
            let m = Tanhshrink::new();
            let y = m.forward(&x).unwrap();
            ferrotorch_core::backward(&y).unwrap();

            let grad = x.grad().unwrap().unwrap();
            let expected = numerical_grad(tanhshrink_fn, val);
            assert!(
                (grad.item().unwrap() - expected).abs() < 1e-4,
                "Tanhshrink grad at x={}: expected {}, got {}",
                val,
                expected,
                grad.item().unwrap()
            );
        }
    }

    // -----------------------------------------------------------------------
    // State dict round-trip (empty for all activations)
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_dict_empty() {
        let m = ReLU::new();
        let sd = Module::<f64>::state_dict(&m);
        assert!(sd.is_empty());
    }
}
