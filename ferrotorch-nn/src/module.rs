use std::collections::HashMap;

use ferrotorch_core::{Device, FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::parameter::Parameter;

/// A map from parameter names to tensors, used for serialization.
pub type StateDict<T> = HashMap<String, Tensor<T>>;

/// Reduction mode for loss functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reduction {
    /// Return the mean of all losses.
    Mean,
    /// Return the sum of all losses.
    Sum,
    /// Return the unreduced loss tensor.
    None,
}

/// The trait that all neural network layers implement.
///
/// Requires `Send + Sync` to match `Tensor<T>`'s thread-safety guarantees.
pub trait Module<T: Float>: Send + Sync {
    /// Forward pass. Takes input tensor, returns output tensor.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Iterate over all learnable parameters.
    fn parameters(&self) -> Vec<&Parameter<T>>;

    /// Iterate over all learnable parameters mutably.
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>>;

    /// Named parameters for state dict serialization.
    ///
    /// Keys use dot-separated paths for nested modules
    /// (e.g., `"layer1.weight"`, `"layer1.bias"`).
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)>;

    /// Set training mode. Affects dropout, batchnorm, etc.
    fn train(&mut self);

    /// Set evaluation mode.
    fn eval(&mut self);

    /// Whether the module is in training mode.
    fn is_training(&self) -> bool;

    /// Move all parameters to a device.
    ///
    /// Default implementation iterates `parameters_mut()` and transfers each.
    fn to_device(&mut self, device: Device) -> FerrotorchResult<()> {
        for param in self.parameters_mut() {
            *param = param.to(device)?;
        }
        Ok(())
    }

    /// Export parameters as a state dict.
    fn state_dict(&self) -> StateDict<T> {
        self.named_parameters()
            .into_iter()
            .map(|(name, param)| {
                // Clone the tensor data (not just the Arc) for serialization.
                let data = param.tensor().clone();
                (name, data)
            })
            .collect()
    }

    /// Load parameters from a state dict.
    ///
    /// When `strict` is `true` (default), unexpected keys are an error.
    /// When `false`, unexpected keys are silently ignored and missing
    /// keys leave existing parameter values unchanged.
    fn load_state_dict(&mut self, state: &StateDict<T>, strict: bool) -> FerrotorchResult<()> {
        let named = self.named_parameters();
        let known_keys: std::collections::HashSet<&str> =
            named.iter().map(|(k, _)| k.as_str()).collect();

        if strict {
            for key in state.keys() {
                if !known_keys.contains(key.as_str()) {
                    return Err(FerrotorchError::InvalidArgument {
                        message: format!("unexpected key in state_dict: \"{key}\""),
                    });
                }
            }
        }

        // We need mutable access to parameters. Use named_parameters to get
        // the mapping, then parameters_mut to actually update.
        // This two-pass approach avoids borrowing issues.
        let param_names: Vec<String> = self
            .named_parameters()
            .into_iter()
            .map(|(name, _)| name)
            .collect();

        let params_mut = self.parameters_mut();

        for (name, param) in param_names.iter().zip(params_mut.into_iter()) {
            if let Some(tensor) = state.get(name) {
                if param.shape() != tensor.shape() {
                    return Err(FerrotorchError::ShapeMismatch {
                        message: format!(
                            "state_dict shape mismatch for \"{name}\": expected {:?}, got {:?}",
                            param.shape(),
                            tensor.shape()
                        ),
                    });
                }
                // Replace the parameter data with the loaded tensor.
                *param = Parameter::new(tensor.clone());
            } else if strict {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("missing key in state_dict: \"{name}\""),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    /// A minimal test module with one parameter.
    struct SimpleModule<T: Float> {
        weight: Parameter<T>,
        training: bool,
    }

    impl<T: Float> SimpleModule<T> {
        fn new(size: usize) -> FerrotorchResult<Self> {
            Ok(Self {
                weight: Parameter::zeros(&[size])?,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for SimpleModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            // Just return input for testing.
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Parameter<T>> {
            vec![&self.weight]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            vec![&mut self.weight]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            vec![("weight".to_string(), &self.weight)]
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

    #[test]
    fn test_module_parameters() {
        let m = SimpleModule::<f32>::new(5).unwrap();
        assert_eq!(m.parameters().len(), 1);
        assert_eq!(m.parameters()[0].shape(), &[5]);
    }

    #[test]
    fn test_module_named_parameters() {
        let m = SimpleModule::<f32>::new(3).unwrap();
        let named = m.named_parameters();
        assert_eq!(named.len(), 1);
        assert_eq!(named[0].0, "weight");
    }

    #[test]
    fn test_module_train_eval() {
        let mut m = SimpleModule::<f32>::new(2).unwrap();
        assert!(m.is_training());
        m.eval();
        assert!(!m.is_training());
        m.train();
        assert!(m.is_training());
    }

    #[test]
    fn test_module_state_dict_roundtrip() {
        let m = SimpleModule::<f32>::new(4).unwrap();
        let sd = m.state_dict();
        assert!(sd.contains_key("weight"));
        assert_eq!(sd["weight"].shape(), &[4]);

        let mut m2 = SimpleModule::<f32>::new(4).unwrap();
        m2.load_state_dict(&sd, true).unwrap();
    }

    #[test]
    fn test_module_state_dict_strict_extra_key() {
        let mut m = SimpleModule::<f32>::new(3).unwrap();
        let mut sd = HashMap::new();
        sd.insert(
            "weight".to_string(),
            ferrotorch_core::zeros::<f32>(&[3]).unwrap(),
        );
        sd.insert(
            "extra".to_string(),
            ferrotorch_core::zeros::<f32>(&[1]).unwrap(),
        );

        assert!(m.load_state_dict(&sd, true).is_err());
        assert!(m.load_state_dict(&sd, false).is_ok());
    }

    #[test]
    fn test_module_state_dict_shape_mismatch() {
        let mut m = SimpleModule::<f32>::new(3).unwrap();
        let mut sd = HashMap::new();
        sd.insert(
            "weight".to_string(),
            ferrotorch_core::zeros::<f32>(&[5]).unwrap(),
        );

        assert!(m.load_state_dict(&sd, true).is_err());
    }

    #[test]
    fn test_module_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SimpleModule<f32>>();
    }

    #[test]
    fn test_reduction_enum() {
        assert_eq!(Reduction::Mean, Reduction::Mean);
        assert_ne!(Reduction::Mean, Reduction::Sum);
    }

    #[test]
    fn test_to_device_cpu_preserves_weights() {
        let mut m = SimpleModule::<f32>::new(4).unwrap();
        m.to_device(ferrotorch_core::Device::Cpu).unwrap();
        assert_eq!(m.parameters().len(), 1);
        assert_eq!(m.parameters()[0].shape(), &[4]);
    }

    #[test]
    fn test_to_device_cuda_without_backend() {
        let mut m = SimpleModule::<f32>::new(3).unwrap();
        let result = m.to_device(ferrotorch_core::Device::Cuda(0));
        assert!(result.is_err());
    }
}
