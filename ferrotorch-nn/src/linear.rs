//! Fully connected (dense) linear layer: `y = input @ weight^T + bias`.
//!
//! This is the fundamental building block for feedforward networks. The
//! weight matrix has shape `[out_features, in_features]` (same convention
//! as PyTorch) and the optional bias has shape `[out_features]`.
//!
//! # Autograd
//!
//! The forward pass is built from composable differentiable operations
//! (`mm_differentiable`, `add`), so the backward graph is constructed
//! automatically:
//!
//! - `grad_weight` is accumulated through `MmBackward`
//! - `grad_bias` is accumulated through `AddBackward` (broadcast reduction)
//! - `grad_input` is accumulated through `MmBackward`

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::linalg::mm_differentiable;
use ferrotorch_core::grad_fns::shape::{reshape, transpose_2d};
use ferrotorch_core::{Float, FerrotorchError, FerrotorchResult, Tensor};

use crate::init::{kaiming_uniform, NonLinearity};
use crate::module::Module;
use crate::parameter::Parameter;

/// A fully connected (dense) linear layer.
///
/// Applies the transformation `y = x @ W^T + b` where `W` has shape
/// `[out_features, in_features]` and `b` (if present) has shape
/// `[out_features]`.
///
/// # Initialization
///
/// - **Weight**: Kaiming uniform with `gain = sqrt(2)` (ReLU). This is
///   the PyTorch default for `nn.Linear`.
/// - **Bias**: Zeros.
///
/// # Examples
///
/// ```ignore
/// let layer = Linear::<f32>::new(784, 256, true)?;
/// let output = layer.forward(&input)?; // input: [batch, 784] -> output: [batch, 256]
/// ```
#[derive(Debug)]
pub struct Linear<T: Float> {
    /// Weight matrix of shape `[out_features, in_features]`.
    pub weight: Parameter<T>,
    /// Optional bias vector of shape `[out_features]`.
    pub bias: Option<Parameter<T>>,
    /// Number of input features.
    in_features: usize,
    /// Number of output features.
    out_features: usize,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> Linear<T> {
    /// Create a new linear layer.
    ///
    /// # Arguments
    ///
    /// - `in_features` — Size of each input sample.
    /// - `out_features` — Size of each output sample.
    /// - `bias` — If `true`, adds a learnable bias to the output.
    ///
    /// # Errors
    ///
    /// Returns an error if `in_features` or `out_features` is zero, or if
    /// parameter allocation fails.
    pub fn new(in_features: usize, out_features: usize, bias: bool) -> FerrotorchResult<Self> {
        if in_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Linear: in_features must be > 0".into(),
            });
        }
        if out_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "Linear: out_features must be > 0".into(),
            });
        }

        // Initialize weight with Kaiming uniform (fan_in mode, ReLU gain).
        let mut weight = Parameter::zeros(&[out_features, in_features])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        // Initialize bias with zeros.
        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_features])?;
            crate::init::zeros(&mut b)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_features,
            out_features,
            training: true,
        })
    }

    /// Number of input features.
    #[inline]
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Number of output features.
    #[inline]
    pub fn out_features(&self) -> usize {
        self.out_features
    }
}

impl<T: Float> Module<T> for Linear<T> {
    /// Forward pass: `y = input @ weight^T + bias`.
    ///
    /// # Input shape
    ///
    /// - 2D: `[batch, in_features]` — standard batched forward.
    ///
    /// # Output shape
    ///
    /// - 2D input: `[batch, out_features]`.
    ///
    /// # Autograd
    ///
    /// When gradient tracking is enabled, the returned tensor participates
    /// in the computation graph through the composed differentiable
    /// operations (`mm_differentiable` + `add`). Calling `.backward()` on
    /// a downstream scalar loss will propagate gradients to `weight` and
    /// `bias` automatically.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Validate input shape.
        if input.ndim() != 2 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Linear expects 2D input [batch, in_features], got shape {:?}",
                    input.shape()
                ),
            });
        }

        if input.shape()[1] != self.in_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Linear: input has {} features but layer expects {}",
                    input.shape()[1],
                    self.in_features
                ),
            });
        }

        // Compute weight^T: [in_features, out_features]
        // Uses differentiable transpose so gradients flow back to the weight parameter.
        let weight_t = transpose_2d(self.weight.tensor())?;

        // Compute input @ weight^T: [batch, out_features]
        // Uses differentiable mm which attaches MmBackward when grad is enabled.
        let output = mm_differentiable(input, &weight_t)?;

        // Add bias if present: [batch, out_features] + [out_features] (broadcast)
        // Uses the differentiable `add` op which handles broadcasting and GPU
        // dispatch, so gradients flow to the bias parameter automatically.
        match &self.bias {
            Some(bias) => {
                let bias_2d = reshape(bias.tensor(), &[1, self.out_features as isize])?;
                add(&output, &bias_2d)
            }
            None => Ok(output),
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
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

// ---------------------------------------------------------------------------
// Display
// ---------------------------------------------------------------------------

impl<T: Float> std::fmt::Display for Linear<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Linear(in_features={}, out_features={}, bias={})",
            self.in_features,
            self.out_features,
            self.bias.is_some()
        )
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{TensorStorage, Tensor};

    /// Create a leaf tensor with given data and shape, optionally with grad.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    /// Assert two float slices are element-wise close.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: actual={a} expected={e} diff={}",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_construction_with_bias() {
        let layer = Linear::<f32>::new(10, 5, true).unwrap();
        assert_eq!(layer.in_features(), 10);
        assert_eq!(layer.out_features(), 5);
        assert_eq!(layer.weight.shape(), &[5, 10]);
        assert!(layer.bias.is_some());
        assert_eq!(layer.bias.as_ref().unwrap().shape(), &[5]);
    }

    #[test]
    fn test_construction_without_bias() {
        let layer = Linear::<f32>::new(8, 4, false).unwrap();
        assert_eq!(layer.weight.shape(), &[4, 8]);
        assert!(layer.bias.is_none());
    }

    #[test]
    fn test_construction_zero_in_features() {
        assert!(Linear::<f32>::new(0, 5, true).is_err());
    }

    #[test]
    fn test_construction_zero_out_features() {
        assert!(Linear::<f32>::new(5, 0, true).is_err());
    }

    #[test]
    fn test_weight_requires_grad() {
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        assert!(layer.weight.requires_grad());
        assert!(layer.bias.as_ref().unwrap().requires_grad());
    }

    // -----------------------------------------------------------------------
    // Forward shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_shape() {
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        let input = leaf(&[0.0; 8], &[2, 4], false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 3]);
    }

    #[test]
    fn test_forward_shape_no_bias() {
        let layer = Linear::<f32>::new(6, 2, false).unwrap();
        let input = leaf(&[0.0; 18], &[3, 6], false);
        let output = layer.forward(&input).unwrap();
        assert_eq!(output.shape(), &[3, 2]);
    }

    #[test]
    fn test_forward_wrong_input_features() {
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        let input = leaf(&[0.0; 15], &[3, 5], false);
        assert!(layer.forward(&input).is_err());
    }

    #[test]
    fn test_forward_1d_input_rejected() {
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[4], false);
        assert!(layer.forward(&input).is_err());
    }

    // -----------------------------------------------------------------------
    // Forward correctness (manual weight/bias)
    // -----------------------------------------------------------------------

    #[test]
    fn test_forward_correctness_no_bias() {
        // Build a layer then manually set the weight.
        let mut layer = Linear::<f32>::new(3, 2, false).unwrap();

        // weight = [[1, 0, 0], [0, 1, 0]]  (2x3)
        // This selects the first two features.
        layer.weight = Parameter::from_slice(
            &[1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            &[2, 3],
        )
        .unwrap();

        // input = [[1, 2, 3], [4, 5, 6]]  (2x3)
        let input = leaf(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3], false);
        let output = layer.forward(&input).unwrap();

        // output = input @ weight^T = [[1,2,3],[4,5,6]] @ [[1,0],[0,1],[0,0]]
        //        = [[1, 2], [4, 5]]
        assert_eq!(output.shape(), &[2, 2]);
        assert_close(output.data().unwrap(), &[1.0, 2.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn test_forward_correctness_with_bias() {
        let mut layer = Linear::<f32>::new(2, 2, true).unwrap();

        // weight = [[1, 0], [0, 1]]  (identity)
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        // bias = [10, 20]
        *layer.bias.as_mut().unwrap() =
            Parameter::from_slice(&[10.0, 20.0], &[2]).unwrap();

        let input = leaf(&[1.0, 2.0, 3.0, 4.0], &[2, 2], false);
        let output = layer.forward(&input).unwrap();

        // output = input @ I + [10, 20] = [[11, 22], [13, 24]]
        assert_close(output.data().unwrap(), &[11.0, 22.0, 13.0, 24.0], 1e-6);
    }

    // -----------------------------------------------------------------------
    // Backward gradients
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_gradients_no_bias() {
        // Linear: y = input @ W^T, loss = sum(y)
        // W = [[1, 2], [3, 4]]  (out=2, in=2)
        // input = [[1, 0], [0, 1]]  (batch=2, in=2)
        //
        // W^T = [[1, 3], [2, 4]]
        // y = input @ W^T = [[1, 3], [2, 4]]  shape [2, 2]
        // loss = 1 + 3 + 2 + 4 = 10
        //
        // dL/dy = ones(2, 2)
        // dL/d(input) = dL/dy @ W = [[1,1],[1,1]] @ [[1,2],[3,4]] = [[4,6],[4,6]]
        // dL/d(W^T) = input^T @ dL/dy = [[1,0],[0,1]] @ [[1,1],[1,1]] = [[1,1],[1,1]]
        // => dL/d(W) = [[1,1],[1,1]]^T = [[1,1],[1,1]]
        let mut layer = Linear::<f32>::new(2, 2, false).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();

        let input = leaf(&[1.0, 0.0, 0.0, 1.0], &[2, 2], true);
        let output = layer.forward(&input).unwrap();

        // Reduce to scalar via differentiable sum.
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        // Check input grad.
        let input_grad = input.grad().unwrap().expect("input should have grad");
        assert_eq!(input_grad.shape(), &[2, 2]);
        assert_close(input_grad.data().unwrap(), &[4.0, 6.0, 4.0, 6.0], 1e-5);
    }

    #[test]
    fn test_backward_weight_grad() {
        // Use a known configuration to verify weight gradients.
        // W = [[1, 0], [0, 1]]  (out=2, in=2) — identity
        // input = [[2, 3]]  (batch=1, in=2)
        // y = [[2, 3]] @ I = [[2, 3]]
        // loss = sum(y) = 5
        // dL/dy = ones(1, 2) = [[1, 1]]
        //
        // For mm(input, W^T):
        //   dL/d(W^T) = input^T @ dL/dy = [[2],[3]] @ [[1,1]] = [[2,2],[3,3]]
        //   => dL/d(W) by chain through transpose
        //
        // PyTorch reference: W.grad = dL/dy^T @ input = [[1],[1]] @ [[2,3]] = [[2,3],[2,3]]
        let mut layer = Linear::<f32>::new(2, 2, false).unwrap();
        layer.weight = Parameter::from_slice(&[1.0, 0.0, 0.0, 1.0], &[2, 2]).unwrap();

        let input = leaf(&[2.0, 3.0], &[1, 2], false);
        let output = layer.forward(&input).unwrap();
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        // The weight gradient flows through mm(input, W^T):
        // dL/d(W^T) = input^T @ dL/dy = [[2],[3]] @ [[1,1]] = [[2,2],[3,3]]
        // Since W^T was created via transpose(W), the gradient accumulates on
        // the original W parameter through the transpose operation.
        // The transpose of [[2,2],[3,3]] is [[2,3],[2,3]], matching W's shape.
        let w_grad = layer.weight.grad().unwrap().expect("weight should have grad");
        assert_eq!(w_grad.shape(), &[2, 2]);
        assert_close(w_grad.data().unwrap(), &[2.0, 3.0, 2.0, 3.0], 1e-5);
    }

    #[test]
    fn test_backward_numerical_gradient() {
        // Numerical gradient check for a small Linear layer.
        // Perturb each weight element by eps and check finite-difference
        // gradient matches autograd gradient.
        let eps = 1e-4f32;

        let mut layer = Linear::<f32>::new(2, 2, false).unwrap();
        layer.weight = Parameter::from_slice(&[0.5, -0.3, 0.2, 0.8], &[2, 2]).unwrap();

        let input_data = [1.0f32, 2.0, 3.0, 4.0];
        let input = leaf(&input_data, &[2, 2], false);

        // Forward + backward for analytic gradient.
        let output = layer.forward(&input).unwrap();
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        let analytic_grad = layer.weight.grad().unwrap().unwrap();
        let analytic = analytic_grad.data().unwrap().to_vec();

        // Numerical gradient for each weight element.
        let base_weight = [0.5f32, -0.3, 0.2, 0.8];
        for idx in 0..4 {
            let mut w_plus = base_weight;
            w_plus[idx] += eps;
            let mut layer_plus = Linear::<f32>::new(2, 2, false).unwrap();
            layer_plus.weight = Parameter::from_slice(&w_plus, &[2, 2]).unwrap();
            let input_ng = leaf(&input_data, &[2, 2], false);
            let out_plus = ferrotorch_core::no_grad(|| {
                let o = layer_plus.forward(&input_ng).unwrap();
                ferrotorch_core::grad_fns::reduction::sum(&o).unwrap()
            });
            let loss_plus = out_plus.item().unwrap();

            let mut w_minus = base_weight;
            w_minus[idx] -= eps;
            let mut layer_minus = Linear::<f32>::new(2, 2, false).unwrap();
            layer_minus.weight = Parameter::from_slice(&w_minus, &[2, 2]).unwrap();
            let input_ng2 = leaf(&input_data, &[2, 2], false);
            let out_minus = ferrotorch_core::no_grad(|| {
                let o = layer_minus.forward(&input_ng2).unwrap();
                ferrotorch_core::grad_fns::reduction::sum(&o).unwrap()
            });
            let loss_minus = out_minus.item().unwrap();

            let numerical = (loss_plus - loss_minus) / (2.0 * eps);
            assert!(
                (numerical - analytic[idx]).abs() < 1e-2,
                "weight[{idx}]: numerical={numerical}, analytic={}, diff={}",
                analytic[idx],
                (numerical - analytic[idx]).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Parameter count
    // -----------------------------------------------------------------------

    #[test]
    fn test_parameter_count_with_bias() {
        let layer = Linear::<f32>::new(10, 5, true).unwrap();
        let params = layer.parameters();
        assert_eq!(params.len(), 2);
        // weight: 10 * 5 = 50 elements, bias: 5 elements
        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total, 55);
    }

    #[test]
    fn test_parameter_count_without_bias() {
        let layer = Linear::<f32>::new(10, 5, false).unwrap();
        let params = layer.parameters();
        assert_eq!(params.len(), 1);
        let total: usize = params.iter().map(|p| p.numel()).sum();
        assert_eq!(total, 50);
    }

    // -----------------------------------------------------------------------
    // State dict roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_dict_roundtrip_with_bias() {
        let layer = Linear::<f32>::new(4, 3, true).unwrap();
        let sd = layer.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(sd.contains_key("bias"));
        assert_eq!(sd["weight"].shape(), &[3, 4]);
        assert_eq!(sd["bias"].shape(), &[3]);

        let mut layer2 = Linear::<f32>::new(4, 3, true).unwrap();
        layer2.load_state_dict(&sd, true).unwrap();

        // Verify loaded weights match.
        assert_close(
            layer2.weight.data().unwrap(),
            layer.weight.data().unwrap(),
            1e-7,
        );
        assert_close(
            layer2.bias.as_ref().unwrap().data().unwrap(),
            layer.bias.as_ref().unwrap().data().unwrap(),
            1e-7,
        );
    }

    #[test]
    fn test_state_dict_roundtrip_without_bias() {
        let layer = Linear::<f32>::new(6, 2, false).unwrap();
        let sd = layer.state_dict();
        assert!(sd.contains_key("weight"));
        assert!(!sd.contains_key("bias"));

        let mut layer2 = Linear::<f32>::new(6, 2, false).unwrap();
        layer2.load_state_dict(&sd, true).unwrap();

        assert_close(
            layer2.weight.data().unwrap(),
            layer.weight.data().unwrap(),
            1e-7,
        );
    }

    #[test]
    fn test_state_dict_shape_mismatch_rejected() {
        let layer_a = Linear::<f32>::new(4, 3, true).unwrap();
        let sd = layer_a.state_dict();

        let mut layer_b = Linear::<f32>::new(4, 5, true).unwrap();
        assert!(layer_b.load_state_dict(&sd, true).is_err());
    }

    // -----------------------------------------------------------------------
    // Named parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_named_parameters_with_bias() {
        let layer = Linear::<f32>::new(3, 2, true).unwrap();
        let named = layer.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_named_parameters_without_bias() {
        let layer = Linear::<f32>::new(3, 2, false).unwrap();
        let named = layer.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    // -----------------------------------------------------------------------
    // Train / Eval
    // -----------------------------------------------------------------------

    #[test]
    fn test_train_eval() {
        let mut layer = Linear::<f32>::new(4, 3, true).unwrap();
        assert!(layer.is_training());
        layer.eval();
        assert!(!layer.is_training());
        layer.train();
        assert!(layer.is_training());
    }

    // -----------------------------------------------------------------------
    // Display
    // -----------------------------------------------------------------------

    #[test]
    fn test_display() {
        let layer = Linear::<f32>::new(10, 5, true).unwrap();
        let s = format!("{layer}");
        assert_eq!(s, "Linear(in_features=10, out_features=5, bias=true)");
    }

    #[test]
    fn test_display_no_bias() {
        let layer = Linear::<f32>::new(10, 5, false).unwrap();
        let s = format!("{layer}");
        assert_eq!(s, "Linear(in_features=10, out_features=5, bias=false)");
    }

    // -----------------------------------------------------------------------
    // Send + Sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_linear_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Linear<f32>>();
        assert_send_sync::<Linear<f64>>();
    }

    // -----------------------------------------------------------------------
    // Device transfer
    // -----------------------------------------------------------------------

    #[test]
    fn test_to_device_cpu_preserves_weights() {
        let mut layer = Linear::<f32>::new(4, 3, true).unwrap();
        layer.weight = Parameter::from_slice(
            &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
            &[3, 4],
        )
        .unwrap();
        *layer.bias.as_mut().unwrap() =
            Parameter::from_slice(&[0.1, 0.2, 0.3], &[3]).unwrap();

        layer.to_device(ferrotorch_core::Device::Cpu).unwrap();

        assert_eq!(layer.weight.shape(), &[3, 4]);
        assert_close(layer.weight.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0], 1e-7);
        assert_close(layer.bias.as_ref().unwrap().data().unwrap(), &[0.1, 0.2, 0.3], 1e-7);
        assert!(layer.weight.requires_grad());
        assert!(layer.bias.as_ref().unwrap().requires_grad());
    }

    #[test]
    fn test_to_device_cuda_returns_device_unavailable() {
        let mut layer = Linear::<f32>::new(4, 3, true).unwrap();
        let result = layer.to_device(ferrotorch_core::Device::Cuda(0));
        assert!(result.is_err());
    }
}
