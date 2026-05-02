//! SparseAdam optimizer — Adam variant for sparse gradients.
//!
//! Only updates the moment estimates for parameters whose gradients are
//! non-zero. This is efficient for large embedding tables where only a few
//! rows are accessed per batch.
//!
//! Matches PyTorch's `torch.optim.SparseAdam`.

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

/// Hyperparameters for the SparseAdam optimizer.
#[derive(Debug, Clone, Copy)]
pub struct SparseAdamConfig {
    /// Learning rate (default: 0.001).
    pub lr: f64,
    /// Exponential decay rates for first/second moment (default: (0.9, 0.999)).
    pub betas: (f64, f64),
    /// Numerical stability term (default: 1e-8).
    pub eps: f64,
}

impl Default for SparseAdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
        }
    }
}

#[derive(Debug)]
struct SparseAdamState {
    step_count: u64,
    exp_avg: Vec<f64>,
    exp_avg_sq: Vec<f64>,
}

/// SparseAdam optimizer.
///
/// Like Adam, but only updates moment estimates for elements with non-zero
/// gradients. This avoids unnecessary computation for large sparse parameter
/// matrices (e.g., embedding tables).
///
/// # Differences from Adam
///
/// - No weight decay support (sparse updates and weight decay don't mix well).
/// - No AMSGrad variant.
/// - Moment estimates are only updated at indices where the gradient is non-zero.
#[derive(Debug)]
pub struct SparseAdam<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: SparseAdamConfig,
    state: HashMap<String, SparseAdamState>,
}

impl<T: Float> SparseAdam<T> {
    /// Create a new SparseAdam optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: SparseAdamConfig) -> Self {
        let group = ParamGroup::new(params, config.lr);
        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
        }
    }

    fn param_key(gi: usize, pi: usize) -> String {
        format!("g{gi}_p{pi}")
    }
}

impl<T: Float> Optimizer<T> for SparseAdam<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        let (beta1, beta2) = self.config.betas;
        let eps = self.config.eps;

        for gi in 0..self.param_groups.len() {
            let lr = self.param_groups[gi].lr;

            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];
                let tensor = param.tensor();

                let grad_tensor = match tensor.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                if tensor.is_cuda() {
                    return Err(FerrotorchError::NotImplementedOnCuda { op: "SparseAdam" });
                }

                let grad_data = grad_tensor.data_vec()?;
                let numel = tensor.numel();
                let key = Self::param_key(gi, pi);

                let state = self.state.entry(key).or_insert_with(|| SparseAdamState {
                    step_count: 0,
                    exp_avg: vec![0.0; numel],
                    exp_avg_sq: vec![0.0; numel],
                });

                state.step_count += 1;
                let t = state.step_count as f64;

                // Bias correction factors.
                let bc1 = 1.0 - beta1.powf(t);
                let bc2 = 1.0 - beta2.powf(t);

                // Update only non-zero gradient entries.
                no_grad(|| -> FerrotorchResult<()> {
                    let mut param_data = tensor.data_vec()?;

                    for i in 0..numel {
                        let g = num_traits::ToPrimitive::to_f64(&grad_data[i]).unwrap();
                        if g == 0.0 {
                            continue; // Skip zero gradients — this is the "sparse" part.
                        }

                        // Update moments at this index.
                        state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g;
                        state.exp_avg_sq[i] = beta2 * state.exp_avg_sq[i] + (1.0 - beta2) * g * g;

                        // Bias-corrected estimates.
                        let m_hat = state.exp_avg[i] / bc1;
                        let v_hat = state.exp_avg_sq[i] / bc2;

                        // Parameter update.
                        let update = lr * m_hat / (v_hat.sqrt() + eps);
                        let p = num_traits::ToPrimitive::to_f64(&param_data[i]).unwrap();
                        param_data[i] = T::from(p - update).unwrap();
                    }

                    unsafe { param.tensor().update_data(&param_data)? };
                    Ok(())
                })?;
            }
        }
        Ok(())
    }

    fn zero_grad(&mut self) -> FerrotorchResult<()> {
        for group in &mut self.param_groups {
            for param in &mut group.params {
                param.zero_grad()?;
            }
        }
        Ok(())
    }

    fn lr(&self) -> f64 {
        self.param_groups.first().map(|g| g.lr).unwrap_or(0.0)
    }

    fn set_lr(&mut self, lr: f64) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
    }

    fn param_groups(&self) -> &[ParamGroup<T>] {
        &self.param_groups
    }

    fn param_groups_mut(&mut self) -> &mut [ParamGroup<T>] {
        &mut self.param_groups
    }

    fn add_param_group(&mut self, group: ParamGroup<T>) {
        self.param_groups.push(group);
    }

    fn state_dict(&self) -> OptimizerState {
        OptimizerState::default()
    }

    fn load_state_dict(&mut self, _state: &OptimizerState) -> FerrotorchResult<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;

    fn make_param(data: &[f32]) -> Parameter<f32> {
        let t = Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], true)
            .unwrap();
        Parameter::new(t)
    }

    #[test]
    fn test_sparse_adam_skips_zero_gradients() {
        // Create a param and set a sparse gradient (only index 1 is non-zero).
        let param = make_param(&[1.0, 2.0, 3.0]);
        let initial = param.tensor().data_vec().unwrap();

        // Manually set gradient: [0, 1, 0]
        let grad = Tensor::from_storage(TensorStorage::cpu(vec![0.0f32, 1.0, 0.0]), vec![3], false)
            .unwrap();
        param.tensor().set_grad(Some(grad)).unwrap();

        let mut opt = SparseAdam::new(vec![param.clone()], SparseAdamConfig::default());
        opt.step().unwrap();

        let updated = opt.param_groups[0].params[0].tensor().data_vec().unwrap();

        // Index 0 and 2 should be unchanged (zero gradient).
        assert_eq!(updated[0], initial[0], "index 0 should not change");
        assert_eq!(updated[2], initial[2], "index 2 should not change");
        // Index 1 should have moved.
        assert_ne!(updated[1], initial[1], "index 1 should change");
    }

    #[test]
    fn test_sparse_adam_dense_matches_direction() {
        // With all non-zero gradients, SparseAdam should still update all params.
        let param = make_param(&[1.0, 2.0]);
        let grad =
            Tensor::from_storage(TensorStorage::cpu(vec![0.5f32, -0.5]), vec![2], false).unwrap();
        param.tensor().set_grad(Some(grad)).unwrap();

        let mut opt = SparseAdam::new(vec![param.clone()], SparseAdamConfig::default());
        opt.step().unwrap();

        let updated = opt.param_groups[0].params[0].tensor().data_vec().unwrap();

        // Positive gradient -> param should decrease.
        assert!(updated[0] < 1.0, "positive grad should decrease param");
        // Negative gradient -> param should increase.
        assert!(updated[1] > 2.0, "negative grad should increase param");
    }

    #[test]
    fn test_sparse_adam_multiple_steps() {
        let param = make_param(&[5.0]);
        let mut opt = SparseAdam::new(vec![param.clone()], SparseAdamConfig::default());

        for _ in 0..10 {
            let grad =
                Tensor::from_storage(TensorStorage::cpu(vec![1.0f32]), vec![1], false).unwrap();
            opt.param_groups[0].params[0]
                .tensor()
                .set_grad(Some(grad))
                .unwrap();
            opt.step().unwrap();
        }

        let val = opt.param_groups[0].params[0].tensor().data_vec().unwrap()[0];
        // After 10 steps with constant positive gradient, param should decrease.
        assert!(val < 5.0, "param should decrease after 10 steps, got {val}");
    }

    #[test]
    fn test_sparse_adam_zero_grad() {
        let param = make_param(&[1.0, 2.0]);
        let grad =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0f32, 1.0]), vec![2], false).unwrap();
        param.tensor().set_grad(Some(grad)).unwrap();

        let mut opt = SparseAdam::new(vec![param], SparseAdamConfig::default());
        opt.zero_grad().unwrap();

        // Grad should be cleared.
        let g = opt.param_groups[0].params[0].tensor().grad().unwrap();
        assert!(g.is_none(), "grad should be None after zero_grad");
    }
}
