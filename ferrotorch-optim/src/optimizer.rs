use std::collections::HashMap;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Parameter;

/// Serializable optimizer state for checkpointing.
pub type OptimizerState = HashMap<String, HashMap<String, Vec<f64>>>;

/// A group of parameters with shared hyperparameters.
///
/// The `params` field is `pub(crate)` so that internal optimizer
/// implementations can index into the vector directly while external
/// callers go through [`ParamGroup::params`] (read-only) and
/// [`ParamGroup::add_param`] (fallible append). Direct external access
/// would otherwise let downstream code bypass the optimizer's
/// invariants — most importantly, the requirement that all parameters
/// in a group be on a single device.
#[derive(Debug)]
pub struct ParamGroup<T: Float> {
    /// The parameters in this group. Crate-internal optimizer code
    /// indexes this vector directly; external code should go through
    /// [`ParamGroup::params`] / [`ParamGroup::add_param`].
    pub(crate) params: Vec<Parameter<T>>,
    /// Learning rate for this group.
    pub lr: f64,
    /// Weight decay for this group (L2 penalty).
    pub weight_decay: f64,
}

impl<T: Float> ParamGroup<T> {
    /// Create a new parameter group sharing the given learning rate.
    pub fn new(params: Vec<Parameter<T>>, lr: f64) -> Self {
        Self {
            params,
            lr,
            weight_decay: 0.0,
        }
    }

    /// Set the L2 weight decay for this group, builder-style.
    pub fn with_weight_decay(mut self, wd: f64) -> Self {
        self.weight_decay = wd;
        self
    }

    /// Read-only access to the parameters in this group.
    #[inline]
    #[must_use]
    pub fn params(&self) -> &[Parameter<T>] {
        &self.params
    }

    /// Append a parameter to this group.
    ///
    /// # Errors
    ///
    /// Currently always succeeds; the `Result` return is reserved so
    /// that future invariants (e.g. dtype/device homogeneity within a
    /// group) can be enforced without a breaking change.
    pub fn add_param(&mut self, param: Parameter<T>) -> FerrotorchResult<()> {
        self.params.push(param);
        Ok(())
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
    data.iter()
        .map(|&v| cast::<T, f64>(v))
        .collect::<FerrotorchResult<Vec<f64>>>()
}

/// Helper: create a tensor from f64 vec (for state deserialization).
pub(crate) fn f64_vec_to_tensor<T: Float>(
    data: &[f64],
    shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    let converted: Vec<T> = data
        .iter()
        .map(|&v| cast::<f64, T>(v))
        .collect::<FerrotorchResult<Vec<T>>>()?;
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
