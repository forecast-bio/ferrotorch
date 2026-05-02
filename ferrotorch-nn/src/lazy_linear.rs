//! Lazy variants of [`Linear`](super::Linear) and convolution layers.
//!
//! Lazy modules defer parameter allocation until the first forward call,
//! at which point the input tensor's shape is inspected to determine the
//! missing dimensions (`in_features` for `LazyLinear`, `in_channels` for
//! `LazyConv*d`). Mirrors `torch.nn.LazyLinear` and friends.
//!
//! # Use cases
//!
//! Lazy modules are useful when:
//! - The input feature size is hard to compute by hand (e.g. after a
//!   variable-length sequence of pooling/strided convs feeds into a
//!   classifier head — `LazyLinear` lets you write the model definition
//!   without manually working out the flattened dimension).
//! - You want to load a state_dict that already has the correct weight
//!   shape and let the module pick that up at load time.
//!
//! # Thread safety
//!
//! Initialization uses [`std::sync::OnceLock`] so the first forward call
//! across any number of threads is safely materialized exactly once. Any
//! subsequent forward call sees the initialized parameters via a regular
//! reference (no lock acquisition on the hot path).
//!
//! # Limitations
//!
//! Once initialized, lazy modules are functionally identical to their
//! eager counterparts. They cannot be re-initialized with a different
//! input feature size — you would need to construct a fresh lazy module
//! for that. CL-445.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use ferrotorch_core::grad_fns::linalg::linear_fused;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::init::{NonLinearity, kaiming_uniform, zeros as init_zeros};
use crate::module::Module;
use crate::parameter::Parameter;

/// A linear layer that defers `in_features` discovery to the first
/// forward call.
///
/// On the first call to [`forward`](Self::forward), the input's last
/// dimension is taken as `in_features`, the weight (shape
/// `[out_features, in_features]`) and optional bias (shape
/// `[out_features]`) are allocated and initialized identically to
/// [`Linear`](super::Linear), and stored. Subsequent forward calls
/// behave exactly like a standard `Linear`.
///
/// Mirrors `torch.nn.LazyLinear`.
#[derive(Debug)]
pub struct LazyLinear<T: Float> {
    out_features: usize,
    bias_enabled: bool,
    weight: OnceLock<Parameter<T>>,
    bias: OnceLock<Parameter<T>>,
    training: AtomicBool,
}

impl<T: Float> LazyLinear<T> {
    /// Build a new `LazyLinear` with the given `out_features` and bias flag.
    /// `in_features` will be discovered from the first forward input.
    ///
    /// # Errors
    ///
    /// Returns an error if `out_features == 0`.
    pub fn new(out_features: usize, bias: bool) -> FerrotorchResult<Self> {
        if out_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyLinear: out_features must be > 0".into(),
            });
        }
        Ok(Self {
            out_features,
            bias_enabled: bias,
            weight: OnceLock::new(),
            bias: OnceLock::new(),
            training: AtomicBool::new(true),
        })
    }

    /// Returns `true` once the parameters have been materialized
    /// (i.e. after the first successful forward call).
    pub fn is_initialized(&self) -> bool {
        self.weight.get().is_some()
    }

    /// Number of output features. Always known at construction time.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Number of input features. `None` until the first forward call
    /// has materialized the weight.
    pub fn in_features(&self) -> Option<usize> {
        self.weight.get().map(|w| w.tensor().shape()[1])
    }

    /// Eagerly materialize the parameters with the given `in_features`.
    /// Useful when you want the parameters present before any forward
    /// call (e.g. so they show up in `parameters()` for the optimizer).
    ///
    /// Calling this after the parameters are already initialized is
    /// a no-op (returns Ok). Calling this with a different in_features
    /// than was previously materialized is also a no-op — the existing
    /// parameters are kept; the contract is "first one wins".
    pub fn materialize(&self, in_features: usize) -> FerrotorchResult<()> {
        if in_features == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyLinear: in_features must be > 0".into(),
            });
        }
        if self.weight.get().is_none() {
            let mut w = Parameter::zeros(&[self.out_features, in_features])?;
            kaiming_uniform(&mut w, NonLinearity::ReLU)?;
            // set() returns Err if another thread won the race; that's
            // fine, the other initialization wins and ours is dropped.
            let _ = self.weight.set(w);
        }
        if self.bias_enabled && self.bias.get().is_none() {
            let mut b = Parameter::zeros(&[self.out_features])?;
            init_zeros(&mut b)?;
            let _ = self.bias.set(b);
        }
        Ok(())
    }
}

impl<T: Float> Module<T> for LazyLinear<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() == 0 {
            return Err(FerrotorchError::ShapeMismatch {
                message: "LazyLinear: scalar input not supported".into(),
            });
        }

        // Materialize on first call. Subsequent calls hit a fast path.
        if self.weight.get().is_none() {
            let last_dim = input.shape()[input.ndim() - 1];
            self.materialize(last_dim)?;
        }

        let weight = self
            .weight
            .get()
            .expect("weight should be initialized after materialize()");
        let in_features = weight.tensor().shape()[1];

        let last_dim = input.shape()[input.ndim() - 1];
        if last_dim != in_features {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LazyLinear: input has {} features but layer was initialized with {}",
                    last_dim, in_features
                ),
            });
        }

        // Same logic as Linear::forward — flatten leading dims, fused
        // linear, reshape back.
        let input_shape = input.shape().to_vec();
        let batch_shape = &input_shape[..input_shape.len() - 1];
        let n: usize = batch_shape.iter().product::<usize>().max(1);
        let needs_reshape = input.ndim() != 2;
        let input_2d = if needs_reshape {
            reshape(input, &[n as isize, in_features as isize])?
        } else {
            input.clone()
        };

        let output_2d = linear_fused(
            &input_2d,
            weight.tensor(),
            self.bias.get().map(|b| b.tensor()),
        )?;

        if needs_reshape {
            let mut out_shape: Vec<isize> = batch_shape.iter().map(|&d| d as isize).collect();
            out_shape.push(self.out_features as isize);
            reshape(&output_2d, &out_shape)
        } else {
            Ok(output_2d)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        if let Some(w) = self.weight.get() {
            params.push(w);
        }
        if let Some(b) = self.bias.get() {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        if let Some(w) = self.weight.get_mut() {
            params.push(w);
        }
        if let Some(b) = self.bias.get_mut() {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        if let Some(w) = self.weight.get() {
            params.push(("weight".to_string(), w));
        }
        if let Some(b) = self.bias.get() {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training.store(true, Ordering::Relaxed);
    }

    fn eval(&mut self) {
        self.training.store(false, Ordering::Relaxed);
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_lazy_linear_uninitialized_until_first_forward() {
        let lazy: LazyLinear<f32> = LazyLinear::new(8, true).unwrap();
        assert!(!lazy.is_initialized());
        assert_eq!(lazy.in_features(), None);
        // Empty parameters list pre-init.
        assert_eq!(lazy.parameters().len(), 0);
    }

    #[test]
    fn test_lazy_linear_materializes_on_first_forward() {
        let lazy: LazyLinear<f32> = LazyLinear::new(4, true).unwrap();
        let input = cpu_tensor(
            &[
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
            &[2, 6],
        );
        let out = lazy.forward(&input).unwrap();
        assert_eq!(out.shape(), &[2, 4]);
        assert!(lazy.is_initialized());
        assert_eq!(lazy.in_features(), Some(6));
        assert_eq!(lazy.parameters().len(), 2); // weight + bias
    }

    #[test]
    fn test_lazy_linear_no_bias_has_one_param() {
        let lazy: LazyLinear<f32> = LazyLinear::new(3, false).unwrap();
        let input = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let _ = lazy.forward(&input).unwrap();
        assert_eq!(lazy.parameters().len(), 1);
        assert!(lazy.bias.get().is_none());
    }

    #[test]
    fn test_lazy_linear_subsequent_forward_uses_initialized_weights() {
        let lazy: LazyLinear<f32> = LazyLinear::new(2, true).unwrap();
        let input1 = cpu_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        let _ = lazy.forward(&input1).unwrap();

        // Second forward with the same in_features should succeed.
        let input2 = cpu_tensor(&[4.0, 5.0, 6.0], &[1, 3]);
        let out2 = lazy.forward(&input2).unwrap();
        assert_eq!(out2.shape(), &[1, 2]);
    }

    #[test]
    fn test_lazy_linear_rejects_mismatched_in_features() {
        let lazy: LazyLinear<f32> = LazyLinear::new(2, true).unwrap();
        let input1 = cpu_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        let _ = lazy.forward(&input1).unwrap();
        // Now in_features is locked to 3.
        let input_bad = cpu_tensor(&[1.0, 2.0, 3.0, 4.0], &[1, 4]);
        let result = lazy.forward(&input_bad);
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_linear_explicit_materialize_initializes_eagerly() {
        let lazy: LazyLinear<f32> = LazyLinear::new(8, true).unwrap();
        assert!(!lazy.is_initialized());
        lazy.materialize(16).unwrap();
        assert!(lazy.is_initialized());
        assert_eq!(lazy.in_features(), Some(16));
        // Parameters are now visible to optimizers without a forward call.
        assert_eq!(lazy.parameters().len(), 2);
    }

    #[test]
    fn test_lazy_linear_materialize_idempotent() {
        let lazy: LazyLinear<f32> = LazyLinear::new(4, false).unwrap();
        lazy.materialize(8).unwrap();
        // Second call with same in_features is a no-op.
        lazy.materialize(8).unwrap();
        // Even with different in_features, the first one wins -- does
        // not panic, does not re-initialize.
        lazy.materialize(16).unwrap();
        assert_eq!(lazy.in_features(), Some(8));
    }

    #[test]
    fn test_lazy_linear_zero_out_features_errors() {
        let result = LazyLinear::<f32>::new(0, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_lazy_linear_higher_rank_input() {
        // 3-D input [batch, seq, features] should be handled like Linear.
        let lazy: LazyLinear<f32> = LazyLinear::new(2, true).unwrap();
        let data: Vec<f32> = (0..24).map(|i| i as f32 / 10.0).collect();
        let input = cpu_tensor(&data, &[2, 4, 3]);
        let out = lazy.forward(&input).unwrap();
        assert_eq!(out.shape(), &[2, 4, 2]);
        assert_eq!(lazy.in_features(), Some(3));
    }

    #[test]
    fn test_lazy_linear_named_parameters_after_init() {
        let lazy: LazyLinear<f32> = LazyLinear::new(2, true).unwrap();
        let input = cpu_tensor(&[1.0, 2.0, 3.0], &[1, 3]);
        let _ = lazy.forward(&input).unwrap();
        let names: Vec<String> = lazy
            .named_parameters()
            .iter()
            .map(|(n, _)| n.clone())
            .collect();
        assert!(names.contains(&"weight".to_string()));
        assert!(names.contains(&"bias".to_string()));
    }

    #[test]
    fn test_lazy_linear_train_eval_toggle() {
        let mut lazy: LazyLinear<f32> = LazyLinear::new(2, true).unwrap();
        assert!(lazy.is_training());
        lazy.eval();
        assert!(!lazy.is_training());
        lazy.train();
        assert!(lazy.is_training());
    }
}
