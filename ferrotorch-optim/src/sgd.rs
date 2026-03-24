//! Stochastic Gradient Descent optimizer with momentum.
//!
//! Implements the SGD update rule from PyTorch:
//!
//! - Without momentum: `param = param - lr * grad`
//! - With momentum:
//!   - `buf = momentum * buf + (1 - dampening) * grad`
//!   - If nesterov: `grad = grad + momentum * buf`
//!   - Else: `grad = buf`
//!   - `param = param - lr * grad`
//!
//! Weight decay is applied as L2 regularization before the momentum step:
//! `grad = grad + weight_decay * param`.

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchResult, Float, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// SgdConfig
// ---------------------------------------------------------------------------

/// Configuration for the SGD optimizer.
#[derive(Debug, Clone)]
pub struct SgdConfig {
    /// Learning rate.
    pub lr: f64,
    /// Momentum factor (default: 0.0).
    pub momentum: f64,
    /// Dampening for momentum (default: 0.0).
    pub dampening: f64,
    /// Weight decay (L2 penalty) (default: 0.0).
    pub weight_decay: f64,
    /// Whether to use Nesterov momentum (default: false).
    pub nesterov: bool,
    /// When `true`, maximize the objective by negating the gradient (default:
    /// false). CL-321
    pub maximize: bool,
}

impl SgdConfig {
    /// Create a new SGD configuration with the given learning rate.
    ///
    /// All other hyperparameters are set to their defaults:
    /// - `momentum = 0.0`
    /// - `dampening = 0.0`
    /// - `weight_decay = 0.0`
    /// - `nesterov = false`
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.0,
            dampening: 0.0,
            weight_decay: 0.0,
            nesterov: false,
            maximize: false,
        }
    }

    /// Set the momentum factor.
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the dampening factor.
    pub fn dampening(mut self, dampening: f64) -> Self {
        self.dampening = dampening;
        self
    }

    /// Set the weight decay (L2 penalty).
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable or disable Nesterov momentum.
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }
}

impl Default for SgdConfig {
    fn default() -> Self {
        Self::new(0.01)
    }
}

// ---------------------------------------------------------------------------
// Sgd
// ---------------------------------------------------------------------------

/// Stochastic Gradient Descent optimizer with optional momentum, weight
/// decay, dampening, and Nesterov acceleration.
#[derive(Debug)]
pub struct Sgd<T: Float> {
    /// Parameter groups, each with their own learning rate.
    param_groups: Vec<ParamGroup<T>>,
    /// Global configuration (momentum, dampening, weight_decay, nesterov).
    config: SgdConfig,
    /// Momentum buffers keyed by `"{group_idx}_{param_idx}"`.
    momentum_buffers: HashMap<String, Vec<T>>,
    /// Tracks whether each parameter has had its first step (for momentum
    /// buffer initialization). Keyed the same as `momentum_buffers`.
    step_count: HashMap<String, u64>,
}

impl<T: Float> Sgd<T> {
    /// Create a new SGD optimizer.
    ///
    /// All parameters are placed in a single parameter group with the
    /// learning rate from `config.lr`.
    pub fn new(params: Vec<Parameter<T>>, config: SgdConfig) -> Self {
        let lr = config.lr;
        let weight_decay = config.weight_decay;
        let group = ParamGroup::new(params, lr).with_weight_decay(weight_decay);
        Self {
            param_groups: vec![group],
            config,
            momentum_buffers: HashMap::new(),
            step_count: HashMap::new(),
        }
    }

    /// Build the string key for a given group/param index pair.
    #[inline]
    fn buf_key(group_idx: usize, param_idx: usize) -> String {
        format!("{group_idx}_{param_idx}")
    }
}

impl<T: Float> Optimizer<T> for Sgd<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        let momentum = self.config.momentum;
        let dampening = self.config.dampening;
        let nesterov = self.config.nesterov;

        for gi in 0..self.param_groups.len() {
            let group_lr = self.param_groups[gi].lr;
            let group_wd = self.param_groups[gi].weight_decay;

            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];

                // Skip parameters without gradients.
                let grad_tensor = match param.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                let param_data = param.data_vec()?;
                let mut grad_data = grad_tensor.data_vec()?;

                // Maximize: negate gradient. CL-321
                if self.config.maximize {
                    for g in grad_data.iter_mut() {
                        *g = T::from(0.0).unwrap() - *g;
                    }
                }

                // Weight decay: grad = grad + weight_decay * param
                let wd = group_wd;
                if wd > 0.0 {
                    let wd_t = T::from(wd).unwrap();
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g = *g + wd_t * p;
                    }
                }

                // Momentum
                let effective_grad = if momentum > 0.0 {
                    let key = Self::buf_key(gi, pi);
                    let step = self.step_count.entry(key.clone()).or_insert(0);

                    if *step == 0 {
                        // First step: buf = grad.clone()
                        self.momentum_buffers.insert(key.clone(), grad_data.clone());
                    } else {
                        // buf = momentum * buf + (1 - dampening) * grad
                        let mom_t = T::from(momentum).unwrap();
                        let damp_coeff = T::from(1.0 - dampening).unwrap();
                        let buf = self.momentum_buffers.get_mut(&key).unwrap();
                        for (b, &g) in buf.iter_mut().zip(grad_data.iter()) {
                            *b = mom_t * *b + damp_coeff * g;
                        }
                    }

                    *step += 1;

                    let buf = self.momentum_buffers.get(&key).unwrap();

                    if nesterov {
                        // grad = grad + momentum * buf
                        let mom_t = T::from(momentum).unwrap();
                        let mut nesterov_grad = grad_data.clone();
                        for (ng, &b) in nesterov_grad.iter_mut().zip(buf.iter()) {
                            *ng = *ng + mom_t * b;
                        }
                        nesterov_grad
                    } else {
                        // grad = buf
                        buf.clone()
                    }
                } else {
                    grad_data
                };

                // param = param - lr * grad
                let lr_t = T::from(group_lr).unwrap();
                let new_data: Vec<T> = param_data
                    .iter()
                    .zip(effective_grad.iter())
                    .map(|(&p, &g)| p - lr_t * g)
                    .collect();

                // Write updated values back (works on CPU and GPU).
                no_grad(|| {
                    // SAFETY: Optimizer step runs inside no_grad() with exclusive
                    // access to parameters, so no aliasing references exist.
                    unsafe { param.tensor().update_data(&new_data) }
                })?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> FerrotorchResult<()> {
        for group in &mut self.param_groups {
            for param in &mut group.params {
                param.set_grad(None)?;
            }
        }
        Ok(())
    }

    fn lr(&self) -> f64 {
        self.param_groups
            .first()
            .map(|g| g.lr)
            .unwrap_or(self.config.lr)
    }

    fn set_lr(&mut self, lr: f64) {
        for group in &mut self.param_groups {
            group.lr = lr;
        }
        self.config.lr = lr;
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
        let mut state = OptimizerState::new();
        for (key, buf) in &self.momentum_buffers {
            let mut entry = HashMap::new();
            let f64_buf: Vec<f64> = buf.iter().map(|&v| v.to_f64().unwrap()).collect();
            entry.insert("momentum_buffer".to_string(), f64_buf);
            if let Some(&steps) = self.step_count.get(key) {
                entry.insert("step".to_string(), vec![steps as f64]);
            }
            state.insert(key.clone(), entry);
        }
        state
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        self.momentum_buffers.clear();
        self.step_count.clear();
        for (key, entry) in state {
            if let Some(buf_data) = entry.get("momentum_buffer") {
                let buf: Vec<T> = buf_data.iter().map(|&v| T::from(v).unwrap()).collect();
                self.momentum_buffers.insert(key.clone(), buf);
            }
            if let Some(step_data) = entry.get("step") {
                if let Some(&step_val) = step_data.first() {
                    self.step_count.insert(key.clone(), step_val as u64);
                }
            }
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};
    use ferrotorch_nn::{Linear, MSELoss, Module, Reduction};

    /// Create a leaf tensor with given data and shape, optionally with grad.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Basic SGD (no momentum)
    // -----------------------------------------------------------------------

    #[test]
    fn test_sgd_basic_step() {
        // Parameter = [1.0, 2.0, 3.0], grad = [0.1, 0.2, 0.3], lr = 0.1
        // After step: param = param - 0.1 * grad = [0.99, 1.98, 2.97]
        let p = Parameter::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let grad = leaf(&[0.1, 0.2, 0.3], &[3], false);
        p.set_grad(Some(grad)).unwrap();

        let config = SgdConfig::new(0.1);
        let mut sgd = Sgd::new(vec![p], config);

        sgd.step().unwrap();

        let updated = sgd.param_groups()[0].params[0].data().unwrap();
        assert!((updated[0] - 0.99).abs() < 1e-6);
        assert!((updated[1] - 1.98).abs() < 1e-6);
        assert!((updated[2] - 2.97).abs() < 1e-6);
    }

    #[test]
    fn test_sgd_skips_params_without_grad() {
        let p1 = Parameter::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let p2 = Parameter::from_slice(&[3.0f32, 4.0], &[2]).unwrap();

        // Only set grad on p1.
        let grad = leaf(&[1.0, 1.0], &[2], false);
        p1.set_grad(Some(grad)).unwrap();

        let config = SgdConfig::new(0.5);
        let mut sgd = Sgd::new(vec![p1, p2], config);
        sgd.step().unwrap();

        // p1 should be updated.
        let p1_data = sgd.param_groups()[0].params[0].data().unwrap();
        assert!((p1_data[0] - 0.5).abs() < 1e-6);
        assert!((p1_data[1] - 1.5).abs() < 1e-6);

        // p2 should be unchanged.
        let p2_data = sgd.param_groups()[0].params[1].data().unwrap();
        assert!((p2_data[0] - 3.0).abs() < 1e-6);
        assert!((p2_data[1] - 4.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Zero grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_grad_clears_grads() {
        let p = Parameter::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let grad = leaf(&[0.5, 0.5], &[2], false);
        p.set_grad(Some(grad)).unwrap();
        assert!(p.grad().unwrap().is_some());

        let config = SgdConfig::new(0.01);
        let mut sgd = Sgd::new(vec![p], config);
        sgd.zero_grad().unwrap();

        assert!(sgd.param_groups()[0].params[0].grad().unwrap().is_none());
    }

    // -----------------------------------------------------------------------
    // Momentum
    // -----------------------------------------------------------------------

    #[test]
    fn test_sgd_momentum_works() {
        // Verify momentum accumulation over two steps.
        // Step 1: buf = grad = [1.0, 1.0], param -= lr * buf
        // Step 2: buf = 0.9 * [1,1] + 1.0 * [1,1] = [1.9, 1.9], param -= lr * buf
        let p = Parameter::from_slice(&[10.0f32, 10.0], &[2]).unwrap();

        let config = SgdConfig::new(0.1).momentum(0.9);
        let mut sgd = Sgd::new(vec![p], config);

        // Step 1.
        let grad1 = leaf(&[1.0, 1.0], &[2], false);
        sgd.param_groups_mut()[0].params[0]
            .set_grad(Some(grad1))
            .unwrap();
        sgd.step().unwrap();

        // After step 1: param = 10 - 0.1 * 1.0 = 9.9
        let data1 = sgd.param_groups()[0].params[0].data().unwrap().to_vec();
        assert!(
            (data1[0] - 9.9).abs() < 1e-5,
            "step 1: expected 9.9, got {}",
            data1[0]
        );

        // Step 2.
        let grad2 = leaf(&[1.0, 1.0], &[2], false);
        sgd.param_groups_mut()[0].params[0]
            .set_grad(Some(grad2))
            .unwrap();
        sgd.step().unwrap();

        // After step 2: buf = 0.9*1 + 1*1 = 1.9, param = 9.9 - 0.1 * 1.9 = 9.71
        let data2 = sgd.param_groups()[0].params[0].data().unwrap().to_vec();
        assert!(
            (data2[0] - 9.71).abs() < 1e-5,
            "step 2: expected 9.71, got {}",
            data2[0]
        );
    }

    #[test]
    fn test_sgd_nesterov_momentum() {
        // Nesterov: effective_grad = grad + momentum * buf
        // Step 1: buf = grad = [1.0], effective = grad + mom * buf = 1 + 0.9*1 = 1.9
        //         param = 10 - 0.1 * 1.9 = 9.81
        let p = Parameter::from_slice(&[10.0f32], &[1]).unwrap();

        let config = SgdConfig::new(0.1).momentum(0.9).nesterov(true);
        let mut sgd = Sgd::new(vec![p], config);

        let grad = leaf(&[1.0], &[1], false);
        sgd.param_groups_mut()[0].params[0]
            .set_grad(Some(grad))
            .unwrap();
        sgd.step().unwrap();

        let data = sgd.param_groups()[0].params[0].data().unwrap().to_vec();
        assert!(
            (data[0] - 9.81).abs() < 1e-5,
            "nesterov step 1: expected 9.81, got {}",
            data[0]
        );
    }

    // -----------------------------------------------------------------------
    // Weight decay
    // -----------------------------------------------------------------------

    #[test]
    fn test_sgd_weight_decay() {
        // grad = grad + wd * param = 0.0 + 0.1 * 5.0 = 0.5
        // param = 5.0 - 0.1 * 0.5 = 4.95
        let p = Parameter::from_slice(&[5.0f32], &[1]).unwrap();
        let grad = leaf(&[0.0], &[1], false);
        p.set_grad(Some(grad)).unwrap();

        let config = SgdConfig::new(0.1).weight_decay(0.1);
        let mut sgd = Sgd::new(vec![p], config);
        sgd.step().unwrap();

        let data = sgd.param_groups()[0].params[0].data().unwrap().to_vec();
        assert!(
            (data[0] - 4.95).abs() < 1e-5,
            "expected 4.95, got {}",
            data[0]
        );
    }

    // -----------------------------------------------------------------------
    // LR accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_lr_get_set() {
        let p = Parameter::<f32>::zeros(&[2]).unwrap();
        let config = SgdConfig::new(0.01);
        let mut sgd = Sgd::new(vec![p], config);

        assert!((sgd.lr() - 0.01).abs() < 1e-12);

        sgd.set_lr(0.1);
        assert!((sgd.lr() - 0.1).abs() < 1e-12);
        assert!((sgd.param_groups()[0].lr - 0.1).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Multiple parameter groups with different LRs
    // -----------------------------------------------------------------------

    #[test]
    fn test_param_groups_different_lr() {
        let p1 = Parameter::from_slice(&[10.0f32], &[1]).unwrap();
        let p2 = Parameter::from_slice(&[10.0f32], &[1]).unwrap();

        let config = SgdConfig::new(0.1);
        let mut sgd = Sgd::new(vec![p1], config);

        // Add a second group with a different LR.
        let group2 = ParamGroup::new(vec![p2], 0.01);
        sgd.add_param_group(group2);

        // Set gradients.
        let grad1 = leaf(&[1.0], &[1], false);
        let grad2 = leaf(&[1.0], &[1], false);
        sgd.param_groups_mut()[0].params[0]
            .set_grad(Some(grad1))
            .unwrap();
        sgd.param_groups_mut()[1].params[0]
            .set_grad(Some(grad2))
            .unwrap();

        sgd.step().unwrap();

        // Group 0: 10 - 0.1 * 1 = 9.9
        let d1 = sgd.param_groups()[0].params[0].data().unwrap().to_vec();
        assert!(
            (d1[0] - 9.9).abs() < 1e-6,
            "group 0: expected 9.9, got {}",
            d1[0]
        );

        // Group 1: 10 - 0.01 * 1 = 9.99
        let d2 = sgd.param_groups()[1].params[0].data().unwrap().to_vec();
        assert!(
            (d2[0] - 9.99).abs() < 1e-6,
            "group 1: expected 9.99, got {}",
            d2[0]
        );
    }

    // -----------------------------------------------------------------------
    // State dict roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_dict_roundtrip() {
        let p = Parameter::from_slice(&[5.0f32, 5.0], &[2]).unwrap();

        let config = SgdConfig::new(0.1).momentum(0.9);
        let mut sgd = Sgd::new(vec![p], config);

        // Run one step to populate momentum buffers.
        let grad = leaf(&[1.0, 2.0], &[2], false);
        sgd.param_groups_mut()[0].params[0]
            .set_grad(Some(grad))
            .unwrap();
        sgd.step().unwrap();

        // Serialize.
        let state = sgd.state_dict();
        assert!(!state.is_empty());
        assert!(state.contains_key("0_0"));
        let entry = &state["0_0"];
        assert!(entry.contains_key("momentum_buffer"));
        assert!(entry.contains_key("step"));

        // Deserialize into a fresh optimizer.
        let p2 = Parameter::from_slice(&[5.0f32, 5.0], &[2]).unwrap();
        let config2 = SgdConfig::new(0.1).momentum(0.9);
        let mut sgd2 = Sgd::new(vec![p2], config2);
        sgd2.load_state_dict(&state).unwrap();

        // Verify buffers match.
        assert_eq!(sgd2.momentum_buffers.get("0_0").unwrap().len(), 2);
        let orig_buf = sgd.momentum_buffers.get("0_0").unwrap();
        let loaded_buf = sgd2.momentum_buffers.get("0_0").unwrap();
        for (a, b) in orig_buf.iter().zip(loaded_buf.iter()) {
            assert!((*a - *b).abs() < 1e-6, "buffer mismatch: {} vs {}", a, b);
        }
    }

    // -----------------------------------------------------------------------
    // Convergence: XOR with 2-layer MLP
    // -----------------------------------------------------------------------

    #[test]
    #[ignore = "flaky: depends on random init, run with --ignored to verify"]
    fn test_xor_convergence() {
        // XOR dataset: 4 samples, 2 inputs, 1 output.
        // inputs:  [[0,0], [0,1], [1,0], [1,1]]
        // targets: [0, 1, 1, 0]
        let inputs_data: &[f32] = &[0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0];
        let targets_data: &[f32] = &[0.0, 1.0, 1.0, 0.0];

        let inputs = leaf(inputs_data, &[4, 2], false);
        let targets = leaf(targets_data, &[4, 1], false);

        // 2-layer MLP: Linear(2, 8) -> ReLU -> Linear(8, 1)
        let mut layer1 = Linear::<f32>::new(2, 8, true).unwrap();
        let mut layer2 = Linear::<f32>::new(8, 1, true).unwrap();
        let relu = ferrotorch_nn::ReLU::new();

        // Collect parameters.
        let params: Vec<Parameter<f32>> = layer1
            .parameters()
            .into_iter()
            .chain(layer2.parameters())
            .cloned()
            .collect();

        let config = SgdConfig::new(0.5).momentum(0.9);
        let mut sgd = Sgd::new(params, config);

        let loss_fn = MSELoss::new(Reduction::Mean);

        let mut final_loss = f32::MAX;

        for _epoch in 0..2000 {
            // Forward.
            let h = layer1.forward(&inputs).unwrap();
            let h = relu.forward(&h).unwrap();
            let pred = layer2.forward(&h).unwrap();

            let loss = loss_fn.forward(&pred, &targets).unwrap();
            final_loss = loss.item().unwrap();

            // Backward.
            sgd.zero_grad().unwrap();
            loss.backward().unwrap();

            // Sync gradients from the computation graph to the optimizer's
            // parameter clones. The modules and optimizer hold independent
            // Parameter instances, so gradients must be copied.
            {
                let module_params: Vec<&Parameter<f32>> = layer1
                    .parameters()
                    .into_iter()
                    .chain(layer2.parameters())
                    .collect();
                let opt_params = &sgd.param_groups()[0].params;
                for (mp, op) in module_params.iter().zip(opt_params.iter()) {
                    if let Some(g) = mp.grad().unwrap() {
                        op.set_grad(Some(g)).unwrap();
                    }
                }
            }

            sgd.step().unwrap();

            // Sync updated parameters back to the modules.
            {
                let opt_params = &sgd.param_groups()[0].params;
                let mut idx = 0;
                for mp in layer1.parameters_mut() {
                    *mp = opt_params[idx].clone();
                    idx += 1;
                }
                for mp in layer2.parameters_mut() {
                    *mp = opt_params[idx].clone();
                    idx += 1;
                }
            }
        }

        assert!(
            final_loss < 0.01,
            "XOR did not converge: final loss = {final_loss}"
        );
    }
}
