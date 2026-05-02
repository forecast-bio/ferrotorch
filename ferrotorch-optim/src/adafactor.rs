//! Adafactor optimizer — memory-efficient adaptive optimizer.
//!
//! Implements the algorithm from Shazeer & Stern, "Adafactor: Adaptive
//! Learning Rates with Sublinear Memory Cost" (ICML 2018).
//!
//! Key features:
//! - Factored second-moment estimation: for 2D+ parameters, stores row and
//!   column factors instead of the full second-moment matrix, reducing memory
//!   from O(mn) to O(m+n).
//! - Optional relative step sizing (no explicit learning rate needed).
//! - First-moment (beta1) is optional — when disabled, saves more memory.

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

/// Hyperparameters for the Adafactor optimizer.
#[derive(Debug, Clone, Copy)]
pub struct AdafactorConfig {
    /// Learning rate (default: None = use relative step size).
    pub lr: Option<f64>,
    /// Coefficient for computing running averages of gradient (default: None = no first moment).
    pub beta1: Option<f64>,
    /// Decay rate for second moment row/column factors (default: 0.8 — the
    /// paper uses `rho = min(rho_max, 1 - step^(-0.8))`).
    pub decay_rate: f64,
    /// Numerical stability epsilon for second moment (default: 1e-30).
    pub eps_sq: f64,
    /// Numerical stability epsilon for final update (default: 1e-3).
    pub eps_rms: f64,
    /// Weight decay coefficient (default: 0.0).
    pub weight_decay: f64,
    /// Whether to use relative step sizes (default: true).
    pub relative_step: bool,
    /// Warmup initialization (default: false).
    pub warmup_init: bool,
}

impl Default for AdafactorConfig {
    fn default() -> Self {
        Self {
            lr: None,
            beta1: None,
            decay_rate: -0.8,
            eps_sq: 1e-30,
            eps_rms: 1e-3,
            weight_decay: 0.0,
            relative_step: true,
            warmup_init: false,
        }
    }
}

#[derive(Debug)]
struct AdafactorState {
    step_count: u64,
    /// Row factor of second moment (for 2D+ params), shape [rows].
    row_factor: Vec<f64>,
    /// Column factor of second moment (for 2D+ params), shape [cols].
    col_factor: Vec<f64>,
    /// Full second moment (for 1D params or when factoring is off).
    full_sq: Vec<f64>,
    /// First moment estimate (when beta1 is Some).
    exp_avg: Vec<f64>,
    /// RMS of the parameter values (for relative step sizing).
    rms: f64,
}

/// Adafactor optimizer.
///
/// Memory-efficient alternative to Adam that uses factored second moments
/// for matrix parameters and optional relative step sizing.
#[derive(Debug)]
pub struct Adafactor<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdafactorConfig,
    state: HashMap<String, AdafactorState>,
}

impl<T: Float> Adafactor<T> {
    /// Create a new Adafactor optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: AdafactorConfig) -> Self {
        let lr = config.lr.unwrap_or(1e-3);
        let group = ParamGroup::new(params, lr);
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

impl<T: Float> Optimizer<T> for Adafactor<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        let config = self.config;

        for gi in 0..self.param_groups.len() {
            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];
                let tensor = param.tensor();

                let grad_tensor = match tensor.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                if tensor.is_cuda() {
                    return Err(FerrotorchError::NotImplementedOnCuda { op: "Adafactor" });
                }

                let grad_data = grad_tensor.data_vec()?;
                let param_data = tensor.data_vec()?;
                let shape = tensor.shape().to_vec();
                let numel = tensor.numel();
                let key = Self::param_key(gi, pi);

                let use_factored = shape.len() >= 2;
                let rows = if use_factored {
                    shape[shape.len() - 2]
                } else {
                    0
                };
                let cols = if use_factored {
                    shape[shape.len() - 1]
                } else {
                    0
                };

                let state = self.state.entry(key).or_insert_with(|| {
                    let rms = {
                        let sum_sq: f64 = param_data
                            .iter()
                            .map(|v| {
                                let f = num_traits::ToPrimitive::to_f64(v).unwrap();
                                f * f
                            })
                            .sum();
                        (sum_sq / numel as f64).sqrt()
                    };
                    AdafactorState {
                        step_count: 0,
                        row_factor: if use_factored {
                            vec![0.0; rows]
                        } else {
                            vec![]
                        },
                        col_factor: if use_factored {
                            vec![0.0; cols]
                        } else {
                            vec![]
                        },
                        full_sq: if use_factored {
                            vec![]
                        } else {
                            vec![0.0; numel]
                        },
                        exp_avg: if config.beta1.is_some() {
                            vec![0.0; numel]
                        } else {
                            vec![]
                        },
                        rms,
                    }
                });

                state.step_count += 1;
                let step = state.step_count;
                let rho = {
                    let rho_inf = 1.0 - (step as f64).powf(config.decay_rate);
                    rho_inf.min(1.0 - 1e-8)
                };

                // Compute effective learning rate.
                let group_lr = self.param_groups[gi].lr;
                let lr = if config.relative_step {
                    let rel = if config.warmup_init {
                        (step as f64).powf(-0.5).min(1e-6 * step as f64)
                    } else {
                        (step as f64).powf(-0.5)
                    };
                    let rms_val = state.rms.max(config.eps_rms);
                    rel * rms_val
                } else {
                    config.lr.unwrap_or(group_lr)
                };

                no_grad(|| -> FerrotorchResult<()> {
                    let mut new_param = param_data.clone();

                    // Weight decay.
                    if config.weight_decay != 0.0 {
                        for slot in &mut new_param[..numel] {
                            let p = num_traits::ToPrimitive::to_f64(slot).unwrap();
                            *slot = T::from(p * (1.0 - lr * config.weight_decay)).unwrap();
                        }
                    }

                    // Compute squared gradients.
                    let grad_sq: Vec<f64> = grad_data
                        .iter()
                        .map(|v| {
                            let g = num_traits::ToPrimitive::to_f64(v).unwrap();
                            g * g + config.eps_sq
                        })
                        .collect();

                    if use_factored {
                        // Update row and column factors.
                        // row_factor[r] = rho * row_factor[r] + (1-rho) * mean(grad_sq[r, :])
                        let batch = numel / (rows * cols);
                        for r in 0..rows {
                            let mut row_mean = 0.0;
                            for b in 0..batch {
                                for c in 0..cols {
                                    row_mean += grad_sq[b * rows * cols + r * cols + c];
                                }
                            }
                            row_mean /= (batch * cols) as f64;
                            state.row_factor[r] =
                                rho * state.row_factor[r] + (1.0 - rho) * row_mean;
                        }

                        for c in 0..cols {
                            let mut col_mean = 0.0;
                            for b in 0..batch {
                                for r in 0..rows {
                                    col_mean += grad_sq[b * rows * cols + r * cols + c];
                                }
                            }
                            col_mean /= (batch * rows) as f64;
                            state.col_factor[c] =
                                rho * state.col_factor[c] + (1.0 - rho) * col_mean;
                        }

                        // Reconstruct second moment estimate: v[r,c] = row[r] * col[c] / mean(row)
                        let row_mean: f64 = state.row_factor.iter().sum::<f64>() / rows as f64;
                        let row_mean = row_mean.max(1e-30);

                        // Compute update.
                        for i in 0..numel {
                            let r = (i / cols) % rows;
                            let c = i % cols;
                            let v_est = state.row_factor[r] * state.col_factor[c] / row_mean;
                            let g = num_traits::ToPrimitive::to_f64(&grad_data[i]).unwrap();

                            let update = if let Some(beta1) = config.beta1 {
                                state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g;
                                state.exp_avg[i] / (v_est.sqrt() + 1e-30)
                            } else {
                                g / (v_est.sqrt() + 1e-30)
                            };

                            let p = num_traits::ToPrimitive::to_f64(&new_param[i]).unwrap();
                            new_param[i] = T::from(p - lr * update).unwrap();
                        }
                    } else {
                        // Non-factored: full second moment.
                        for i in 0..numel {
                            state.full_sq[i] = rho * state.full_sq[i] + (1.0 - rho) * grad_sq[i];

                            let g = num_traits::ToPrimitive::to_f64(&grad_data[i]).unwrap();
                            let update = if let Some(beta1) = config.beta1 {
                                state.exp_avg[i] = beta1 * state.exp_avg[i] + (1.0 - beta1) * g;
                                state.exp_avg[i] / (state.full_sq[i].sqrt() + 1e-30)
                            } else {
                                g / (state.full_sq[i].sqrt() + 1e-30)
                            };

                            let p = num_traits::ToPrimitive::to_f64(&new_param[i]).unwrap();
                            new_param[i] = T::from(p - lr * update).unwrap();
                        }
                    }

                    // Update RMS for relative step sizing.
                    let sum_sq: f64 = new_param
                        .iter()
                        .map(|v| {
                            let f = num_traits::ToPrimitive::to_f64(v).unwrap();
                            f * f
                        })
                        .sum();
                    state.rms = (sum_sq / numel as f64).sqrt();

                    unsafe { param.tensor().update_data(&new_param)? };
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

    fn make_param(data: &[f32], shape: &[usize]) -> Parameter<f32> {
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap();
        Parameter::new(t)
    }

    fn set_grad(param: &Parameter<f32>, data: &[f32]) {
        let g = Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            param.tensor().shape().to_vec(),
            false,
        )
        .unwrap();
        param.tensor().set_grad(Some(g)).unwrap();
    }

    #[test]
    fn test_adafactor_1d_param_decreases_loss() {
        let param = make_param(&[5.0, 3.0], &[2]);
        let config = AdafactorConfig {
            lr: Some(0.1),
            relative_step: false,
            ..Default::default()
        };
        let mut opt = Adafactor::new(vec![param.clone()], config);

        for _ in 0..20 {
            // Gradient = param value (minimize sum(x^2)/2).
            let vals = opt.param_groups[0].params[0].tensor().data_vec().unwrap();
            set_grad(&opt.param_groups[0].params[0], &vals);
            opt.step().unwrap();
        }

        let final_vals = opt.param_groups[0].params[0].tensor().data_vec().unwrap();
        assert!(
            final_vals[0].abs() < 5.0,
            "param[0] should decrease, got {}",
            final_vals[0]
        );
        assert!(
            final_vals[1].abs() < 3.0,
            "param[1] should decrease, got {}",
            final_vals[1]
        );
    }

    #[test]
    fn test_adafactor_2d_factored() {
        // 2D param should use factored second moments.
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let param = make_param(&data, &[3, 4]);
        let config = AdafactorConfig {
            lr: Some(0.01),
            relative_step: false,
            ..Default::default()
        };
        let mut opt = Adafactor::new(vec![param.clone()], config);

        let grad: Vec<f32> = vec![1.0; 12];
        set_grad(&opt.param_groups[0].params[0], &grad);
        opt.step().unwrap();

        let updated = opt.param_groups[0].params[0].tensor().data_vec().unwrap();
        // All should have decreased (positive gradient with lr > 0).
        for (i, &v) in updated.iter().enumerate() {
            assert!(
                v < data[i],
                "param[{i}] should decrease: {v} >= {}",
                data[i]
            );
        }
    }

    #[test]
    fn test_adafactor_with_beta1() {
        let param = make_param(&[1.0, 2.0, 3.0], &[3]);
        let config = AdafactorConfig {
            lr: Some(0.01),
            beta1: Some(0.9),
            relative_step: false,
            ..Default::default()
        };
        let mut opt = Adafactor::new(vec![param.clone()], config);

        for _ in 0..5 {
            set_grad(&opt.param_groups[0].params[0], &[1.0, 1.0, 1.0]);
            opt.step().unwrap();
        }

        let vals = opt.param_groups[0].params[0].tensor().data_vec().unwrap();
        assert!(vals[0] < 1.0, "should decrease with momentum");
    }

    #[test]
    fn test_adafactor_relative_step() {
        let param = make_param(&[10.0], &[1]);
        let config = AdafactorConfig {
            relative_step: true,
            ..Default::default()
        };
        let mut opt = Adafactor::new(vec![param.clone()], config);

        set_grad(&opt.param_groups[0].params[0], &[1.0]);
        opt.step().unwrap();

        let val = opt.param_groups[0].params[0].tensor().data_vec().unwrap()[0];
        assert!(val < 10.0, "relative step should still decrease param");
    }

    #[test]
    fn test_adafactor_zero_grad() {
        let param = make_param(&[1.0], &[1]);
        set_grad(&param, &[1.0]);
        let mut opt = Adafactor::new(vec![param], AdafactorConfig::default());
        opt.zero_grad().unwrap();

        let g = opt.param_groups[0].params[0].tensor().grad().unwrap();
        assert!(g.is_none());
    }
}
