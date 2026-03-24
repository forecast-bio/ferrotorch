//! AdamW optimizer — Adam with decoupled weight decay.
//!
//! Unlike standard Adam (which adds L2 regularization to the gradient before
//! moment estimation), AdamW applies weight decay directly to the parameters
//! *before* the gradient update. This decoupling produces better
//! generalization in practice and is the default optimizer for most modern
//! deep learning workloads.
//!
//! Reference: Loshchilov & Hutter, "Decoupled Weight Decay Regularization"
//! (ICLR 2019).
//!
//! # GPU-resident state
//!
//! When parameters are f32 tensors on CUDA, the optimizer stores its moment
//! estimates (`exp_avg`, `exp_avg_sq`) as GPU buffers and composes the entire
//! update step from GPU backend operations — no CPU↔GPU transfers.
//! For a 70M-param model this eliminates ~560 MB of PCIe traffic per step.

use std::any::TypeId;
use std::collections::HashMap;

use ferrotorch_core::gpu_dispatch::{gpu_backend, GpuBufferHandle};
use ferrotorch_core::{no_grad, Float, FerrotorchError, FerrotorchResult};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Hyperparameters for the [`AdamW`] optimizer.
#[derive(Debug, Clone, Copy)]
pub struct AdamWConfig {
    /// Learning rate (default: 0.001).
    pub lr: f64,
    /// Exponential decay rates for the first and second moment estimates
    /// (default: `(0.9, 0.999)`).
    pub betas: (f64, f64),
    /// Term added to the denominator for numerical stability (default: 1e-8).
    pub eps: f64,
    /// Decoupled weight decay coefficient (default: 0.01).
    ///
    /// Note: this default is higher than Adam's typical `0.0` because
    /// decoupled weight decay is the whole point of AdamW.
    pub weight_decay: f64,
}

impl Default for AdamWConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// Per-parameter state
// ---------------------------------------------------------------------------

/// Internal state for a single parameter tracked by AdamW.
///
/// CPU parameters use `Vec<f64>` moment buffers on the host.
/// GPU f32 parameters use `GpuBufferHandle` moment buffers that stay
/// device-resident, avoiding any PCIe transfer during `step()`.
#[derive(Debug)]
enum AdamWParamState {
    /// CPU-resident state (original path).
    Cpu {
        step_count: u64,
        exp_avg: Vec<f64>,
        exp_avg_sq: Vec<f64>,
    },
    /// GPU-resident state for f32 params on CUDA.
    Gpu {
        step_count: u64,
        /// First moment (m) — lives on device.
        exp_avg: GpuBufferHandle,
        /// Second moment (v) — lives on device.
        exp_avg_sq: GpuBufferHandle,
        /// Pre-uploaded constant buffer of `eps` values for the denominator.
        eps_buf: GpuBufferHandle,
    },
}

impl AdamWParamState {
    fn step_count(&self) -> u64 {
        match self {
            Self::Cpu { step_count, .. } => *step_count,
            Self::Gpu { step_count, .. } => *step_count,
        }
    }

    /// Download moment vectors to CPU `Vec<f64>` for serialization.
    /// CPU variant returns the vecs directly; GPU variant downloads and
    /// converts f32 → f64.
    fn to_cpu_vecs(&self) -> FerrotorchResult<(Vec<f64>, Vec<f64>)> {
        match self {
            Self::Cpu {
                exp_avg,
                exp_avg_sq,
                ..
            } => Ok((exp_avg.clone(), exp_avg_sq.clone())),
            Self::Gpu {
                exp_avg,
                exp_avg_sq,
                ..
            } => {
                let backend =
                    gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

                let m_bytes = backend.gpu_to_cpu(exp_avg)?;
                let v_bytes = backend.gpu_to_cpu(exp_avg_sq)?;

                let m_f32: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        m_bytes.as_ptr() as *const f32,
                        m_bytes.len() / std::mem::size_of::<f32>(),
                    )
                };
                let v_f32: &[f32] = unsafe {
                    std::slice::from_raw_parts(
                        v_bytes.as_ptr() as *const f32,
                        v_bytes.len() / std::mem::size_of::<f32>(),
                    )
                };

                let m_f64: Vec<f64> = m_f32.iter().map(|&x| x as f64).collect();
                let v_f64: Vec<f64> = v_f32.iter().map(|&x| x as f64).collect();

                Ok((m_f64, v_f64))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// AdamW
// ---------------------------------------------------------------------------

/// AdamW optimizer — Adam with **decoupled** weight decay.
///
/// # Algorithm
///
/// For each parameter `p` with gradient `g`:
///
/// 1. **Decoupled weight decay**: `p = p * (1 - lr * weight_decay)`
/// 2. **First moment update**: `m = beta1 * m + (1 - beta1) * g`
/// 3. **Second moment update**: `v = beta2 * v + (1 - beta2) * g^2`
/// 4. **Bias-corrected estimates**:
///    - `m_hat = m / (1 - beta1^t)`
///    - `v_hat = v / (1 - beta2^t)`
/// 5. **Parameter update**: `p = p - lr * m_hat / (sqrt(v_hat) + eps)`
///
/// The key difference from [`Adam`] is step 1: weight decay is applied
/// directly to the parameter, *not* added to the gradient. This means the
/// regularization strength is independent of the adaptive learning rate.
///
/// # GPU-resident state
///
/// When `T = f32` and the parameter lives on a CUDA device, the optimizer
/// keeps moment buffers on the GPU and composes the entire AdamW update
/// from GPU backend kernels. No CPU↔GPU data transfers occur during
/// `step()`.
#[derive(Debug)]
pub struct AdamW<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdamWConfig,
    /// Per-parameter optimizer state, keyed by `"g{group_idx}_p{param_idx}"`.
    state: HashMap<String, AdamWParamState>,
}

impl<T: Float> AdamW<T> {
    /// Create a new AdamW optimizer with the given parameters and config.
    pub fn new(params: Vec<Parameter<T>>, config: AdamWConfig) -> Self {
        let mut group = ParamGroup::new(params, config.lr);
        group.weight_decay = config.weight_decay;
        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
        }
    }

    /// Create a new AdamW optimizer with pre-configured parameter groups.
    pub fn new_with_groups(groups: Vec<ParamGroup<T>>, config: AdamWConfig) -> Self {
        Self {
            param_groups: groups,
            config,
            state: HashMap::new(),
        }
    }

    /// Generate the state key for a parameter.
    #[inline]
    fn param_key(group_idx: usize, param_idx: usize) -> String {
        format!("g{group_idx}_p{param_idx}")
    }

    /// Returns `true` when the GPU-resident fast path should be used.
    ///
    /// Requirements: T is f32, the parameter is on a CUDA device, and
    /// a GPU backend is registered.
    #[inline]
    fn use_gpu_path(tensor: &ferrotorch_core::Tensor<T>) -> bool {
        TypeId::of::<T>() == TypeId::of::<f32>()
            && tensor.is_cuda()
            && gpu_backend().is_some()
    }
}

// ---------------------------------------------------------------------------
// GPU-resident AdamW step (f32 only)
// ---------------------------------------------------------------------------

/// Execute the full AdamW update on GPU without any CPU↔GPU transfer.
///
/// All buffers (param, grad, m, v, eps) stay device-resident. The update
/// is composed from the `GpuBackend` kernel primitives:
///
/// ```text
/// 1. p  = scale(p, 1 - lr * wd)            // weight decay
/// 2. m  = scale(m, beta1) + scale(g, 1-beta1)   // 1st moment
/// 3. v  = scale(v, beta2) + scale(g*g, 1-beta2)  // 2nd moment
/// 4. m_hat = scale(m, 1/bc1)               // bias correction
/// 5. v_hat = scale(v, 1/bc2)
/// 6. denom = sqrt(v_hat) + eps_buf
/// 7. step  = div(m_hat, denom)
/// 8. p  = p + scale(step, -lr)             // apply update
/// ```
fn gpu_adamw_step(
    param_handle: &GpuBufferHandle,
    grad_handle: &GpuBufferHandle,
    state: &mut AdamWParamState,
    beta1: f64,
    beta2: f64,
    lr: f64,
    wd: f64,
) -> FerrotorchResult<GpuBufferHandle> {
    let backend =
        gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

    let (step_count, m_handle, v_handle, eps_handle) = match state {
        AdamWParamState::Gpu {
            step_count,
            exp_avg,
            exp_avg_sq,
            eps_buf,
        } => (step_count, exp_avg, exp_avg_sq, eps_buf),
        _ => {
            return Err(FerrotorchError::InvalidArgument {
                message: "gpu_adamw_step called with CPU state".into(),
            });
        }
    };

    *step_count += 1;
    let t = *step_count;

    // 1. Weight decay: p_decayed = p * (1 - lr * wd)
    let decay_factor = (1.0 - lr * wd) as f32;
    let p_decayed = backend.scale_f32(param_handle, decay_factor)?;

    // 2. First moment: m = beta1 * m + (1 - beta1) * g
    let m_scaled = backend.scale_f32(m_handle, beta1 as f32)?;
    let g_scaled = backend.scale_f32(grad_handle, (1.0 - beta1) as f32)?;
    let m_new = backend.add_f32(&m_scaled, &g_scaled)?;
    // Update state in place.
    *m_handle = m_new;

    // 3. Second moment: v = beta2 * v + (1 - beta2) * g^2
    let g_sq = backend.mul_f32(grad_handle, grad_handle)?;
    let v_scaled = backend.scale_f32(v_handle, beta2 as f32)?;
    let g_sq_scaled = backend.scale_f32(&g_sq, (1.0 - beta2) as f32)?;
    let v_new = backend.add_f32(&v_scaled, &g_sq_scaled)?;
    *v_handle = v_new;

    // 4-5. Bias correction: m_hat = m / (1 - beta1^t), v_hat = v / (1 - beta2^t)
    let bc1 = 1.0 - beta1.powi(t as i32);
    let bc2 = 1.0 - beta2.powi(t as i32);
    let m_hat = backend.scale_f32(m_handle, (1.0 / bc1) as f32)?;
    let v_hat = backend.scale_f32(v_handle, (1.0 / bc2) as f32)?;

    // 6. denom = sqrt(v_hat) + eps
    let sqrt_v = backend.sqrt_f32(&v_hat)?;
    let denom = backend.add_f32(&sqrt_v, eps_handle)?;

    // 7. step_update = m_hat / denom
    let step_update = backend.div_f32(&m_hat, &denom)?;

    // 8. p = p_decayed - lr * step_update  (= p_decayed + (-lr) * step_update)
    let lr_step = backend.scale_f32(&step_update, -(lr as f32))?;
    let p_new = backend.add_f32(&p_decayed, &lr_step)?;

    Ok(p_new)
}

impl<T: Float> Optimizer<T> for AdamW<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        let config = self.config;
        let (beta1, beta2) = config.betas;

        for gi in 0..self.param_groups.len() {
            let group_lr = self.param_groups[gi].lr;
            let group_wd = self.param_groups[gi].weight_decay;

            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];
                let tensor = param.tensor();

                // Skip parameters without gradients.
                let grad_tensor = match tensor.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                let key = Self::param_key(gi, pi);
                let numel = tensor.numel();

                // ==========================================================
                // GPU fast path: f32 on CUDA — zero CPU↔GPU transfer
                // ==========================================================
                if Self::use_gpu_path(tensor) {
                    let param_handle = tensor.gpu_handle()?;
                    let grad_handle = grad_tensor.gpu_handle()?;
                    let device_ordinal = param_handle.device_ordinal();

                    // Lazily initialize GPU state.
                    if !self.state.contains_key(&key) {
                        let backend = gpu_backend()
                            .ok_or(FerrotorchError::DeviceUnavailable)?;

                        // Allocate zero-filled moment buffers on device.
                        let m_buf = backend.alloc_zeros(
                            numel,
                            std::mem::size_of::<f32>(),
                            device_ordinal,
                        )?;
                        let v_buf = backend.alloc_zeros(
                            numel,
                            std::mem::size_of::<f32>(),
                            device_ordinal,
                        )?;

                        // Upload constant eps buffer (one-time cost).
                        let eps_vec: Vec<f32> = vec![config.eps as f32; numel];
                        let eps_bytes: &[u8] = unsafe {
                            std::slice::from_raw_parts(
                                eps_vec.as_ptr() as *const u8,
                                eps_vec.len() * std::mem::size_of::<f32>(),
                            )
                        };
                        let eps_buf = backend.cpu_to_gpu(
                            eps_bytes,
                            std::mem::size_of::<f32>(),
                            device_ordinal,
                        )?;

                        self.state.insert(
                            key.clone(),
                            AdamWParamState::Gpu {
                                step_count: 0,
                                exp_avg: m_buf,
                                exp_avg_sq: v_buf,
                                eps_buf,
                            },
                        );
                    }

                    let state = self.state.get_mut(&key).unwrap();

                    let p_new = gpu_adamw_step(
                        param_handle,
                        grad_handle,
                        state,
                        beta1,
                        beta2,
                        group_lr,
                        group_wd,
                    )?;

                    no_grad(|| {
                        // SAFETY: Optimizer step runs inside no_grad() with
                        // exclusive access to parameters.
                        unsafe { tensor.update_gpu_buffer(p_new) }
                    })?;

                    continue;
                }

                // ==========================================================
                // CPU path (original): download, compute in f64, upload
                // ==========================================================

                // Read parameter data and gradient data into f64 workspace.
                // data_vec() handles GPU→CPU transfer transparently.
                let param_data: Vec<f64> = tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| v.to_f64().unwrap())
                    .collect();
                let grad_data: Vec<f64> = grad_tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| v.to_f64().unwrap())
                    .collect();

                // ----------------------------------------------------------
                // 1. Decoupled weight decay: p = p * (1 - lr * wd)
                //
                // This is the defining characteristic of AdamW. Unlike Adam
                // which adds wd*param to the gradient (L2 regularization),
                // AdamW decays the parameter values directly. The gradient
                // is NOT modified.
                // ----------------------------------------------------------
                let decay_factor = 1.0 - group_lr * group_wd;

                // ----------------------------------------------------------
                // 2-3. Moment updates (standard Adam, no L2 in gradient)
                // ----------------------------------------------------------
                let state =
                    self.state.entry(key).or_insert_with(|| AdamWParamState::Cpu {
                        step_count: 0,
                        exp_avg: vec![0.0; numel],
                        exp_avg_sq: vec![0.0; numel],
                    });

                // Extract CPU buffers; error if somehow GPU state is here.
                let (step_count, exp_avg, exp_avg_sq) = match state {
                    AdamWParamState::Cpu {
                        step_count,
                        exp_avg,
                        exp_avg_sq,
                    } => (step_count, exp_avg, exp_avg_sq),
                    AdamWParamState::Gpu { .. } => {
                        return Err(FerrotorchError::InvalidArgument {
                            message:
                                "CPU tensor has GPU optimizer state — this is a bug"
                                    .into(),
                        });
                    }
                };

                *step_count += 1;
                let step = *step_count;

                // exp_avg  = beta1 * exp_avg  + (1 - beta1) * grad
                // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                for i in 0..numel {
                    let g = grad_data[i];
                    exp_avg[i] = beta1 * exp_avg[i] + (1.0 - beta1) * g;
                    exp_avg_sq[i] =
                        beta2 * exp_avg_sq[i] + (1.0 - beta2) * g * g;
                }

                // ----------------------------------------------------------
                // 4-5. Bias correction and in-place parameter update
                // ----------------------------------------------------------
                let bc1 = 1.0 - beta1.powi(step as i32);
                let bc2 = 1.0 - beta2.powi(step as i32);

                let new_values: Vec<T> = (0..numel)
                    .map(|i| {
                        let m_hat = exp_avg[i] / bc1;
                        let v_hat = exp_avg_sq[i] / bc2;
                        let decayed = param_data[i] * decay_factor;
                        let updated =
                            decayed - group_lr * m_hat / (v_hat.sqrt() + config.eps);
                        T::from(updated).unwrap()
                    })
                    .collect();

                no_grad(|| {
                    // SAFETY: Optimizer step runs inside no_grad() with exclusive
                    // access to parameters, so no aliasing references exist.
                    unsafe { param.tensor().update_data(&new_values) }
                })?;
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) -> FerrotorchResult<()> {
        for group in &self.param_groups {
            for param in &group.params {
                param.tensor().set_grad(None)?;
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
        let mut out = OptimizerState::new();
        for (key, ps) in &self.state {
            let mut entry = HashMap::new();
            entry.insert("step_count".to_string(), vec![ps.step_count() as f64]);

            // Both CPU and GPU variants serialize to Vec<f64>.
            match ps.to_cpu_vecs() {
                Ok((m, v)) => {
                    entry.insert("exp_avg".to_string(), m);
                    entry.insert("exp_avg_sq".to_string(), v);
                }
                Err(_) => {
                    // If GPU download fails, store empty vecs rather than
                    // panic — the caller can detect this.
                    entry.insert("exp_avg".to_string(), vec![]);
                    entry.insert("exp_avg_sq".to_string(), vec![]);
                }
            }

            out.insert(key.clone(), entry);
        }
        out
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        for (key, entry) in state {
            let step_count = entry
                .get("step_count")
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(0.0) as u64;

            let exp_avg = entry
                .get("exp_avg")
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing exp_avg in state for key {key}"),
                })?;

            let exp_avg_sq = entry
                .get("exp_avg_sq")
                .cloned()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing exp_avg_sq in state for key {key}"),
                })?;

            // Always load as CPU. If the corresponding parameter is on GPU,
            // the first `step()` call will see no state for that key (the
            // CPU state was loaded under the same key) — but we handle the
            // mixed case: we keep the CPU state and the CPU path will be
            // used. If the user wants full GPU-resident state after loading,
            // they can take one step to trigger lazy migration.
            self.state.insert(
                key.clone(),
                AdamWParamState::Cpu {
                    step_count,
                    exp_avg,
                    exp_avg_sq,
                },
            );
        }
        Ok(())
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::grad_fns::arithmetic::{add, pow};
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Create a scalar parameter from a single f64 value.
    fn scalar_param(val: f64) -> Parameter<f64> {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap();
        Parameter::new(t)
    }

    /// Create a 1-D parameter from a slice.
    fn vec_param(data: &[f64]) -> Parameter<f64> {
        let t = Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            true,
        )
        .unwrap();
        Parameter::new(t)
    }

    /// Read a scalar parameter's current value from the optimizer.
    fn param_val(opt: &AdamW<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx]
            .tensor()
            .data()
            .unwrap()[0]
    }

    /// Read a vector parameter's data from the optimizer.
    fn param_vec(opt: &AdamW<f64>, group: usize, idx: usize) -> Vec<f64> {
        opt.param_groups[group].params[idx]
            .tensor()
            .data()
            .unwrap()
            .to_vec()
    }

    /// Set gradient on a parameter inside the optimizer.
    fn set_grad_scalar(opt: &AdamW<f64>, group: usize, idx: usize, val: f64) {
        let grad = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], false).unwrap();
        opt.param_groups[group].params[idx]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();
    }

    /// Set gradient on a vector parameter inside the optimizer.
    fn set_grad_vec(opt: &AdamW<f64>, group: usize, idx: usize, data: &[f64]) {
        let grad = Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            false,
        )
        .unwrap();
        opt.param_groups[group].params[idx]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();
    }

    /// L2 norm of a slice.
    fn l2_norm(data: &[f64]) -> f64 {
        data.iter().map(|&x| x * x).sum::<f64>().sqrt()
    }

    // -------------------------------------------------------------------
    // Default config values
    // -------------------------------------------------------------------

    #[test]
    fn test_default_config() {
        let config = AdamWConfig::default();
        assert_eq!(config.lr, 1e-3);
        assert_eq!(config.betas, (0.9, 0.999));
        assert_eq!(config.eps, 1e-8);
        assert_eq!(config.weight_decay, 0.01);
    }

    // -------------------------------------------------------------------
    // Basic step test
    // -------------------------------------------------------------------

    #[test]
    fn test_adamw_single_step() {
        let p = vec_param(&[1.0, 2.0, 3.0]);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.01,
                weight_decay: 0.1,
                ..Default::default()
            },
        );

        set_grad_vec(&opt, 0, 0, &[0.1, 0.2, 0.3]);
        opt.step().unwrap();

        let data = param_vec(&opt, 0, 0);
        // After one step, each element should have decreased
        // (weight decay pulls toward zero, gradient update also pulls down).
        assert!(data[0] < 1.0, "param[0] should decrease, got {}", data[0]);
        assert!(data[1] < 2.0, "param[1] should decrease, got {}", data[1]);
        assert!(data[2] < 3.0, "param[2] should decrease, got {}", data[2]);
    }

    // -------------------------------------------------------------------
    // Decoupled weight decay verification
    // -------------------------------------------------------------------

    #[test]
    fn test_weight_decay_with_zero_gradient() {
        // With zero gradient, only decoupled weight decay should shrink params.
        // With zero grad, AdamW still decays: p *= (1 - lr * wd).
        // The Adam moment update with zero grad contributes nothing to the update.
        let p = vec_param(&[5.0, -3.0, 10.0]);
        let lr = 0.01;
        let wd = 0.1;
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr,
                weight_decay: wd,
                ..Default::default()
            },
        );

        let initial = param_vec(&opt, 0, 0);
        set_grad_vec(&opt, 0, 0, &[0.0, 0.0, 0.0]);
        opt.step().unwrap();

        let after = param_vec(&opt, 0, 0);
        let decay_factor = 1.0 - lr * wd;

        for i in 0..3 {
            let expected = initial[i] * decay_factor;
            assert!(
                (after[i] - expected).abs() < 1e-10,
                "param[{i}]: expected {expected}, got {}",
                after[i]
            );
        }
    }

    #[test]
    fn test_param_norm_decreases_with_zero_gradient() {
        // Even with zero gradients, param norm should decrease due to
        // decoupled weight decay. This is the hallmark of AdamW.
        let p = vec_param(&[3.0, 4.0]);
        let lr = 0.1;
        let wd = 0.5;
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr,
                weight_decay: wd,
                ..Default::default()
            },
        );

        let norm_before = l2_norm(&param_vec(&opt, 0, 0));

        for _ in 0..10 {
            set_grad_vec(&opt, 0, 0, &[0.0, 0.0]);
            opt.step().unwrap();
        }

        let norm_after = l2_norm(&param_vec(&opt, 0, 0));
        assert!(
            norm_after < norm_before,
            "norm should decrease with zero grad due to decoupled wd: {norm_before} -> {norm_after}"
        );

        // Verify the decay is multiplicative: (1 - 0.1 * 0.5)^10 = 0.95^10
        let expected_ratio = (1.0 - lr * wd).powi(10);
        let actual_ratio = norm_after / norm_before;
        assert!(
            (actual_ratio - expected_ratio).abs() < 1e-10,
            "expected ratio {expected_ratio}, got {actual_ratio}"
        );
    }

    // -------------------------------------------------------------------
    // Convergence test — minimize f(x) = x^2 starting from x = 5.0
    // -------------------------------------------------------------------

    #[test]
    fn test_convergence_quadratic() {
        let p = scalar_param(5.0);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.1,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        for _ in 0..1000 {
            opt.zero_grad().unwrap();
            let val = param_val(&opt, 0, 0);
            // grad of x^2 = 2x
            set_grad_scalar(&opt, 0, 0, 2.0 * val);
            opt.step().unwrap();
        }

        let final_val = param_val(&opt, 0, 0);
        assert!(
            final_val.abs() < 0.01,
            "should converge near zero, got {final_val}"
        );
    }

    // -------------------------------------------------------------------
    // Convergence on Rosenbrock: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    // -------------------------------------------------------------------

    #[test]
    fn test_convergence_rosenbrock() {
        let p = vec_param(&[-1.0, 1.0]);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.001,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        for _ in 0..10000 {
            opt.zero_grad().unwrap();
            let vals = param_vec(&opt, 0, 0);
            let x = vals[0];
            let y = vals[1];

            // grad_x = 2(x-1) - 400x(y-x^2)
            // grad_y = 200(y-x^2)
            let gx = 2.0 * (x - 1.0) - 400.0 * x * (y - x * x);
            let gy = 200.0 * (y - x * x);

            set_grad_vec(&opt, 0, 0, &[gx, gy]);
            opt.step().unwrap();
        }

        let vals = param_vec(&opt, 0, 0);
        let dist = ((vals[0] - 1.0).powi(2) + (vals[1] - 1.0).powi(2)).sqrt();
        assert!(
            dist < 0.1,
            "should converge near (1,1), got ({}, {}), dist={dist}",
            vals[0],
            vals[1]
        );
    }

    // -------------------------------------------------------------------
    // Bias correction test
    // -------------------------------------------------------------------

    #[test]
    fn test_bias_correction_early_steps() {
        // After one step with grad=1.0, param starts at 0.0:
        //   m = 0.1 * 1.0 = 0.1, v = 0.001 * 1.0 = 0.001
        //   bc1 = 0.1, bc2 = 0.001
        //   m_hat = 0.1 / 0.1 = 1.0, v_hat = 0.001 / 0.001 = 1.0
        //   update = 0.001 * 1.0 / (1.0 + 1e-8) ~= 0.001
        //   param = 0.0 - 0.001 = -0.001
        let p = scalar_param(0.0);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.001,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();

        let val = param_val(&opt, 0, 0);
        assert!(
            (val - (-0.001)).abs() < 1e-8,
            "expected ~-0.001, got {val}"
        );
    }

    // -------------------------------------------------------------------
    // Zero grad test
    // -------------------------------------------------------------------

    #[test]
    fn test_zero_grad() {
        let p = vec_param(&[1.0, 2.0]);
        let mut opt = AdamW::new(vec![p], AdamWConfig::default());

        // Set a gradient.
        set_grad_vec(&opt, 0, 0, &[0.5, 0.5]);
        assert!(opt.param_groups[0].params[0]
            .tensor()
            .grad()
            .unwrap()
            .is_some());

        opt.zero_grad().unwrap();
        assert!(opt.param_groups[0].params[0]
            .tensor()
            .grad()
            .unwrap()
            .is_none());
    }

    // -------------------------------------------------------------------
    // LR get/set test
    // -------------------------------------------------------------------

    #[test]
    fn test_lr_get_set() {
        let p = scalar_param(1.0);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.01,
                ..Default::default()
            },
        );

        assert!((opt.lr() - 0.01).abs() < 1e-12);
        opt.set_lr(0.001);
        assert!((opt.lr() - 0.001).abs() < 1e-12);
        assert!((opt.param_groups()[0].lr - 0.001).abs() < 1e-12);
    }

    // -------------------------------------------------------------------
    // State dict round-trip
    // -------------------------------------------------------------------

    #[test]
    fn test_state_dict_round_trip() {
        let p = vec_param(&[1.0, 2.0, 3.0]);
        let mut opt = AdamW::new(vec![p], AdamWConfig::default());

        // Take a few steps to build state.
        for _ in 0..3 {
            set_grad_vec(&opt, 0, 0, &[0.1, 0.2, 0.3]);
            opt.step().unwrap();
        }

        let saved = opt.state_dict();
        assert!(!saved.is_empty(), "state dict should be non-empty");

        // Verify the saved state has expected keys.
        let key = AdamW::<f64>::param_key(0, 0);
        assert!(saved.contains_key(&key), "expected key {key} in state dict");

        let entry = &saved[&key];
        assert!(entry.contains_key("step_count"));
        assert!(entry.contains_key("exp_avg"));
        assert!(entry.contains_key("exp_avg_sq"));

        let step_count = entry["step_count"][0] as u64;
        assert_eq!(step_count, 3);

        // Load state into a fresh optimizer.
        let p2 = vec_param(&[1.0, 2.0, 3.0]);
        let mut opt2 = AdamW::new(vec![p2], AdamWConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2.state_dict();
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["exp_avg"], saved[&key]["exp_avg"]);
        assert_eq!(loaded[&key]["exp_avg_sq"], saved[&key]["exp_avg_sq"]);
    }

    // -------------------------------------------------------------------
    // Multiple parameter groups
    // -------------------------------------------------------------------

    #[test]
    fn test_multiple_param_groups() {
        let p1 = scalar_param(5.0);
        let p2 = scalar_param(5.0);

        let group1 = ParamGroup::new(vec![p1], 0.01).with_weight_decay(0.0);
        let group2 = ParamGroup::new(vec![p2], 0.01).with_weight_decay(1.0);

        let mut opt = AdamW::new_with_groups(
            vec![group1, group2],
            AdamWConfig {
                lr: 0.01,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        // With zero gradient, only group2 (wd=1.0) should see decay.
        set_grad_scalar(&opt, 0, 0, 0.0);
        set_grad_scalar(&opt, 1, 0, 0.0);
        opt.step().unwrap();

        let v1 = param_val(&opt, 0, 0);
        let v2 = param_val(&opt, 1, 0);

        // p1 should be unchanged (no wd, no grad update with zero grad).
        assert!(
            (v1 - 5.0).abs() < 1e-10,
            "p1 should stay at 5.0, got {v1}"
        );

        // p2 should have decayed: 5.0 * (1 - 0.01 * 1.0) = 5.0 * 0.99 = 4.95
        assert!(
            (v2 - 4.95).abs() < 1e-10,
            "p2 should decay to ~4.95, got {v2}"
        );
    }

    // -------------------------------------------------------------------
    // Skip params with no gradient
    // -------------------------------------------------------------------

    #[test]
    fn test_skip_params_without_grad() {
        let p1 = scalar_param(1.0);
        let p2 = scalar_param(2.0);

        let mut opt = AdamW::new(
            vec![p1, p2],
            AdamWConfig {
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        // Only set grad on p1.
        set_grad_scalar(&opt, 0, 0, 0.5);
        // p2 has no gradient.

        opt.step().unwrap();

        // p1 should have been updated.
        let v1 = param_val(&opt, 0, 0);
        assert!(v1 != 1.0, "p1 should have been updated, got {v1}");

        // p2 should be unchanged (no grad = skipped).
        let v2 = param_val(&opt, 0, 1);
        assert_eq!(v2, 2.0, "p2 should be unchanged (no grad)");
    }

    // -------------------------------------------------------------------
    // Multiple steps produce monotonic loss decrease on quadratic
    // -------------------------------------------------------------------

    #[test]
    fn test_monotonic_loss_decrease() {
        let p = scalar_param(10.0);
        let mut opt = AdamW::new(
            vec![p],
            AdamWConfig {
                lr: 0.1,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        let mut prev_loss = f64::MAX;
        for _ in 0..300 {
            opt.zero_grad().unwrap();
            let x = param_val(&opt, 0, 0);
            let loss = x * x;
            assert!(
                loss <= prev_loss + 1e-10,
                "loss should decrease: {prev_loss} -> {loss}"
            );
            prev_loss = loss;
            set_grad_scalar(&opt, 0, 0, 2.0 * x);
            opt.step().unwrap();
        }
        assert!(prev_loss < 0.01, "should converge, final loss = {prev_loss}");
    }

    // -------------------------------------------------------------------
    // Add param group mid-training
    // -------------------------------------------------------------------

    #[test]
    fn test_add_param_group() {
        let p1 = scalar_param(5.0);
        let mut opt = AdamW::new(
            vec![p1],
            AdamWConfig {
                lr: 0.01,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        set_grad_scalar(&opt, 0, 0, 1.0);
        opt.step().unwrap();
        assert!(param_val(&opt, 0, 0) < 5.0);

        // Add a second group.
        let p2 = scalar_param(10.0);
        let group2 = ParamGroup::new(vec![p2], 0.01);
        opt.add_param_group(group2);

        set_grad_scalar(&opt, 0, 0, 1.0);
        set_grad_scalar(&opt, 1, 0, 1.0);
        opt.step().unwrap();

        let v2 = param_val(&opt, 1, 0);
        assert!(
            v2 < 10.0,
            "p2 should have been updated after add_param_group, got {v2}"
        );
    }

    // -------------------------------------------------------------------
    // Convergence with autograd backward (end-to-end)
    // -------------------------------------------------------------------

    #[test]
    fn test_convergence_with_autograd() {
        // Minimize f(x,y) = x^2 + y^2 using autograd backward.
        let px = scalar_param(3.0);
        let py = scalar_param(-2.0);

        let mut opt = AdamW::new(
            vec![px, py],
            AdamWConfig {
                lr: 0.01,
                weight_decay: 0.0,
                ..Default::default()
            },
        );

        for _ in 0..2000 {
            opt.zero_grad().unwrap();

            let x = opt.param_groups[0].params[0].tensor().clone();
            let y = opt.param_groups[0].params[1].tensor().clone();

            let x_sq = pow(&x, 2.0).unwrap();
            let y_sq = pow(&y, 2.0).unwrap();
            let loss = add(&x_sq, &y_sq).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        let vx = param_val(&opt, 0, 0);
        let vy = param_val(&opt, 0, 1);
        assert!(
            vx.abs() < 0.1,
            "expected x near 0, got {vx}"
        );
        assert!(
            vy.abs() < 0.1,
            "expected y near 0, got {vy}"
        );
    }
}
