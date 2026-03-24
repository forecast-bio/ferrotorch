use std::collections::HashMap;

use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Parameter;

/// Serializable optimizer state for checkpointing.
pub type OptimizerState = HashMap<String, HashMap<String, Vec<f64>>>;

/// A group of parameters with shared hyperparameters.
#[derive(Debug)]
pub struct ParamGroup<T: Float> {
    /// The parameters in this group.
    pub params: Vec<Parameter<T>>,
    /// Learning rate for this group.
    pub lr: f64,
    /// Weight decay for this group (L2 penalty).
    pub weight_decay: f64,
}

impl<T: Float> ParamGroup<T> {
    pub fn new(params: Vec<Parameter<T>>, lr: f64) -> Self {
        Self {
            params,
            lr,
            weight_decay: 0.0,
        }
    }

    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }
}

/// The trait that all optimizers implement.
pub trait Optimizer<T: Float> {
    /// Perform one optimization step (update parameters using their .grad).
    ///
    /// All parameter updates execute inside `no_grad()`.
    fn step(&mut self) -> FerrotorchResult<()>;

    /// Zero out all parameter gradients.
    fn zero_grad(&mut self) -> FerrotorchResult<()>;

    /// Get the current learning rate (from the first parameter group).
    fn lr(&self) -> f64;

    /// Set the learning rate for all parameter groups.
    fn set_lr(&mut self, lr: f64);

    /// Get a reference to the parameter groups.
    fn param_groups(&self) -> &[ParamGroup<T>];

    /// Get mutable parameter group references (for scheduler LR updates).
    fn param_groups_mut(&mut self) -> &mut [ParamGroup<T>];

    /// Add a new parameter group.
    fn add_param_group(&mut self, group: ParamGroup<T>);

    /// Export optimizer state for checkpointing.
    fn state_dict(&self) -> OptimizerState;

    /// Load optimizer state from a checkpoint.
    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()>;
}

/// Helper: collect parameter data as a flat f64 vec (for state serialization).
pub(crate) fn tensor_to_f64_vec<T: Float>(t: &Tensor<T>) -> FerrotorchResult<Vec<f64>> {
    let data = t.data()?;
    Ok(data.iter().map(|&v| v.to_f64().unwrap()).collect())
}

/// Helper: create a tensor from f64 vec (for state deserialization).
pub(crate) fn f64_vec_to_tensor<T: Float>(
    data: &[f64],
    shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    let converted: Vec<T> = data.iter().map(|&v| T::from(v).unwrap()).collect();
    ferrotorch_core::from_vec(converted, shape)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_param_group_construction() {
        let p = Parameter::<f32>::zeros(&[3, 4]).unwrap();
        let group = ParamGroup::new(vec![p], 0.01);
        assert_eq!(group.lr, 0.01);
        assert_eq!(group.weight_decay, 0.0);
        assert_eq!(group.params.len(), 1);
    }

    #[test]
    fn test_param_group_with_weight_decay() {
        let p = Parameter::<f32>::zeros(&[5]).unwrap();
        let group = ParamGroup::new(vec![p], 0.1).with_weight_decay(1e-4);
        assert_eq!(group.weight_decay, 1e-4);
    }
}
