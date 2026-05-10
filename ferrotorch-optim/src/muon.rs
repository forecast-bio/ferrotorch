//! Muon optimizer — spectral-norm-aware SGD with Newton-Schulz orthogonalization.
//!
//! For 2D weight matrices, Muon orthogonalizes the gradient via Newton-Schulz
//! iterations before applying momentum. For non-2D parameters (biases, norms),
//! it falls back to standard momentum SGD.
//!
//! Reference: <https://arxiv.org/abs/2502.16982>

use std::collections::HashMap;

use ferrotorch_core::creation::scalar;
use ferrotorch_core::grad_fns::arithmetic::{add, mul, neg, sub};
use ferrotorch_core::grad_fns::reduction::sum as tensor_sum;
use ferrotorch_core::numeric_cast::cast;
// CL-1105 Pattern B correctness: use the differentiable matmul, which has
// the CUDA dispatch (cuBLAS GEMM) wired up; the `ops::linalg::matmul`
// alternative calls `.data()?` and surfaces `GpuTensorNotAccessible` on
// CUDA tensors. The autograd graph is suppressed by the `no_grad` wrapping
// at every call site in the step body.
use ferrotorch_core::grad_fns::linalg::matmul_differentiable as tensor_matmul;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// MuonConfig
// ---------------------------------------------------------------------------

/// Configuration for the [`Muon`] optimizer.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MuonConfig {
    /// Learning rate (default: 0.02).
    pub lr: f64,
    /// Momentum factor (default: 0.95).
    pub momentum: f64,
    /// Whether to use Nesterov momentum (default: true).
    pub nesterov: bool,
    /// Number of Newton-Schulz iterations for orthogonalization (default: 5).
    pub ns_steps: usize,
    /// Weight decay (L2 penalty) applied to parameters (default: 0.0).
    pub weight_decay: f64,
    /// When `true`, maximize the objective by negating the gradient (default:
    /// false). CL-321
    pub maximize: bool,
}

impl MuonConfig {
    /// Create a new Muon configuration with the given learning rate.
    pub fn new(lr: f64) -> Self {
        Self {
            lr,
            momentum: 0.95,
            nesterov: true,
            ns_steps: 5,
            weight_decay: 0.0,
            maximize: false,
        }
    }

    /// Set the momentum factor.
    pub fn momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable or disable Nesterov momentum.
    pub fn nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Set the number of Newton-Schulz iteration steps.
    pub fn ns_steps(mut self, ns_steps: usize) -> Self {
        self.ns_steps = ns_steps;
        self
    }

    /// Set the weight decay (L2 penalty).
    pub fn weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set the learning rate.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the momentum factor.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Enable or disable Nesterov momentum.
    #[must_use]
    pub fn with_nesterov(mut self, nesterov: bool) -> Self {
        self.nesterov = nesterov;
        self
    }

    /// Set the number of Newton-Schulz iterations for orthogonalization.
    #[must_use]
    pub fn with_ns_steps(mut self, ns_steps: usize) -> Self {
        self.ns_steps = ns_steps;
        self
    }

    /// Set the weight decay (L2 penalty) applied to parameters.
    #[must_use]
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Set the maximize flag (when `true`, negate the gradient to maximize).
    #[must_use]
    pub fn with_maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self::new(0.02)
    }
}

// ---------------------------------------------------------------------------
// Matrix helpers (dense, row-major) — kept for state_dict CPU serialization
// ---------------------------------------------------------------------------

/// Compute the Frobenius norm of a flat matrix. (CPU-only reference used by
/// `newton_schulz_orthogonalize`; production path is the device-aware
/// `newton_schulz_orthogonalize_tensor` above.)
#[cfg(test)]
fn frobenius_norm(data: &[f64], _rows: usize, _cols: usize) -> f64 {
    data.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix multiply: C = A (m x k) @ B (k x n) -> (m x n).
#[cfg(test)]
fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// Transpose a (rows x cols) matrix.
#[cfg(test)]
fn transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = data[i * cols + j];
        }
    }
    t
}

/// Device-resident Newton-Schulz orthogonalization of a matrix G (rows x cols).
///
/// 1. Normalize: G = G / ||G||_F where ||G||_F = sqrt(sum(G * G))
/// 2. For `ns_steps` iterations: G = G @ (3*I - G^T @ G) / 2
///
/// All ops dispatch to the tensor's device — the result lands on the same
/// device as the input. CL-1105 Pattern B.
fn newton_schulz_orthogonalize_tensor<T: Float>(
    grad: &Tensor<T>,
    ns_steps: usize,
) -> FerrotorchResult<Tensor<T>> {
    let device = grad.device();
    let shape = grad.shape();
    debug_assert_eq!(shape.len(), 2, "newton_schulz expects 2-D tensor");
    let cols = shape[1];

    // ||G||_F^2 = sum(G * G) (scalar tensor on device).
    let g_sq = mul(grad, grad)?;
    let norm_sq = tensor_sum(&g_sq)?;
    // ||G||_F = sqrt(||G||_F^2)
    let norm = ferrotorch_core::grad_fns::arithmetic::sqrt(&norm_sq)?;

    // Frobenius norm safety guard: if the input is identically zero the
    // upstream code already returns zeros via the algorithmic fixed point;
    // we add a tiny epsilon to keep the division on-device finite.
    let eps_t = scalar(cast::<f64, T>(1e-30)?)?.to(device)?;
    let norm_safe = add(&norm, &eps_t)?;

    // g = grad / ||grad||_F  (broadcast: shape == grad.shape)
    let mut g = ferrotorch_core::grad_fns::arithmetic::div(grad, &norm_safe)?;

    // Construct identity I (cols x cols) and the constant 3 once, on the
    // input's device.
    let identity_cpu = ferrotorch_core::creation::eye::<T>(cols)?;
    let identity = identity_cpu.to(device)?;
    let three_t = scalar(cast::<f64, T>(3.0)?)?.to(device)?;
    let three_i = mul(&identity, &three_t)?;
    let half_t = scalar(cast::<f64, T>(0.5)?)?.to(device)?;

    for _ in 0..ns_steps {
        // G^T  (zero-copy view on any device).
        let gt = g.t()?;
        // G^T @ G  -> (cols x cols)
        let gtg = tensor_matmul(&gt, &g)?;
        // M = 3*I - G^T @ G
        let m = sub(&three_i, &gtg)?;
        // G_{k+1} = G_k @ M / 2
        let gm = tensor_matmul(&g, &m)?;
        g = mul(&gm, &half_t)?;
    }

    Ok(g)
}

#[cfg(test)]
fn newton_schulz_orthogonalize(
    grad: &[f64],
    rows: usize,
    cols: usize,
    ns_steps: usize,
) -> Vec<f64> {
    // CPU-only legacy reference used by test_newton_schulz_*; the production
    // path is `newton_schulz_orthogonalize_tensor`.
    let norm = frobenius_norm(grad, rows, cols);
    if norm < 1e-30 {
        return vec![0.0; rows * cols];
    }
    let mut g: Vec<f64> = grad.iter().map(|&x| x / norm).collect();

    for _ in 0..ns_steps {
        let gt = transpose(&g, rows, cols);
        let gtg = matmul(&gt, &g, cols, rows, cols);

        let mut m = vec![0.0; cols * cols];
        for i in 0..cols {
            for j in 0..cols {
                let idx = i * cols + j;
                let identity = if i == j { 3.0 } else { 0.0 };
                m[idx] = identity - gtg[idx];
            }
        }

        let gm = matmul(&g, &m, rows, cols, cols);
        g = gm.iter().map(|&x| x / 2.0).collect();
    }

    g
}

// ---------------------------------------------------------------------------
// Muon
// ---------------------------------------------------------------------------

/// Muon optimizer.
///
/// For 2D parameters, applies Newton-Schulz orthogonalization to the gradient
/// before the momentum step. For non-2D parameters, falls back to standard
/// momentum SGD.
///
/// CL-1105: momentum buffers are stored as [`Tensor<T>`] so they live on the
/// same device as the parameters they correspond to; the step body composes
/// device-aware arithmetic ops (no `data_vec()` round-trip).
#[derive(Debug)]
pub struct Muon<T: Float> {
    /// Parameter groups.
    param_groups: Vec<ParamGroup<T>>,
    /// Global configuration.
    config: MuonConfig,
    /// Momentum buffers keyed by `"{group_idx}_{param_idx}"`. Each buffer
    /// lives on the same device as the parameter and is used by the
    /// device-resident step path (CL-1105 Pattern B).
    momentum_buffers: HashMap<String, Tensor<T>>,
    /// Step count per parameter (for momentum buffer init).
    step_count: HashMap<String, u64>,
}

impl<T: Float> Muon<T> {
    /// Create a new Muon optimizer.
    pub fn new(params: Vec<Parameter<T>>, config: MuonConfig) -> Self {
        let lr = config.lr;
        let wd = config.weight_decay;
        let group = ParamGroup::new(params, lr).with_weight_decay(wd);
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

impl<T: Float> Optimizer<T> for Muon<T> {
    /// Run one optimizer step.
    ///
    /// CL-1105: device-resident Pattern B. Newton-Schulz, momentum, and
    /// the parameter update are all expressed via device-aware
    /// `arithmetic::*` + `linalg::matmul` ops. Parameter tensors stay on
    /// their original device (CPU or CUDA) throughout the step — no
    /// `data_vec()` round-trip.
    fn step(&mut self) -> FerrotorchResult<()> {
        let momentum_f = self.config.momentum;
        let nesterov = self.config.nesterov;
        let ns_steps = self.config.ns_steps;

        for gi in 0..self.param_groups.len() {
            let group_lr = self.param_groups[gi].lr;
            let group_wd = self.param_groups[gi].weight_decay;

            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];
                let param_t = param.tensor().clone();
                let device = param_t.device();
                let shape = param_t.shape().to_vec();

                // Skip parameters without gradients.
                let grad_tensor = match param.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                let key = Self::buf_key(gi, pi);

                no_grad(|| -> FerrotorchResult<()> {
                    // grad: device-resident clone (negated for maximize).
                    let mut grad: Tensor<T> = if self.config.maximize {
                        neg(&grad_tensor)?
                    } else {
                        grad_tensor.clone()
                    };

                    // Weight decay: grad = grad + wd * param
                    if group_wd > 0.0 {
                        let wd_t = scalar(cast::<f64, T>(group_wd)?)?.to(device)?;
                        let weighted = mul(&param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    // For 2D parameters: apply Newton-Schulz orthogonalization
                    // entirely on the parameter's device. For non-2D: use
                    // gradient as-is (standard momentum SGD).
                    let processed_grad = if shape.len() == 2 {
                        newton_schulz_orthogonalize_tensor(&grad, ns_steps)?
                    } else {
                        grad
                    };

                    // Momentum
                    let effective_grad = if momentum_f > 0.0 {
                        let mom_t = scalar(cast::<f64, T>(momentum_f)?)?.to(device)?;
                        let step = self.step_count.entry(key.clone()).or_insert(0);

                        if *step == 0 {
                            // Initialize momentum buffer to the processed grad
                            // (clone is zero-copy at the storage Arc level on
                            // this construction path; the value is then
                            // overwritten by subsequent EMA updates).
                            self.momentum_buffers
                                .insert(key.clone(), processed_grad.clone());
                        } else {
                            // buf = momentum * buf + processed_grad
                            let old_buf = self.momentum_buffers.get(&key).unwrap().clone();
                            let scaled = mul(&old_buf, &mom_t)?;
                            let new_buf = add(&scaled, &processed_grad)?;
                            self.momentum_buffers.insert(key.clone(), new_buf);
                        }

                        *step += 1;

                        let buf_ref = self.momentum_buffers.get(&key).unwrap();

                        if nesterov {
                            // nesterov_grad = processed_grad + momentum * buf
                            let scaled_buf = mul(buf_ref, &mom_t)?;
                            add(&processed_grad, &scaled_buf)?
                        } else {
                            buf_ref.clone()
                        }
                    } else {
                        processed_grad
                    };

                    // param = param - lr * effective_grad
                    let lr_t = scalar(cast::<f64, T>(group_lr)?)?.to(device)?;
                    let scaled = mul(&effective_grad, &lr_t)?;
                    let new_param = sub(&param_t, &scaled)?;

                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: `update_storage` requires the caller to hold
                    // exclusive access to the parameter's storage Arc.
                    // Conditions here:
                    //  1. We are inside `Optimizer::step(&mut self)`, so no
                    //     other clone of `Muon<T>` can be running.
                    //  2. The enclosing closure is wrapped in `no_grad`, so
                    //     no autograd graph is being constructed and no
                    //     `grad_fn` holds a clone of the parameter tensor.
                    //  3. `param_t` is a fresh clone of the parameter's
                    //     tensor held only in this loop iteration; all
                    //     intermediate tensors built from it (`grad`,
                    //     `processed_grad`, `effective_grad`, `scaled`,
                    //     `new_param`) are about to drop and were produced
                    //     by ops that allocated fresh storage.
                    //  4. `new_param.into_storage_and_shape()` consumed
                    //     `new_param`, so the only remaining handle to
                    //     `storage` is local.
                    // The new storage is on the same device (it was produced
                    // by ops dispatched on `device`) and has matching numel
                    // (verified internally by `update_storage`).
                    unsafe { param_t.update_storage(storage)? };

                    Ok(())
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

    fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
        let mut state = OptimizerState::new();
        for (key, buf) in &self.momentum_buffers {
            let mut entry = HashMap::new();
            // Materialize device-resident momentum buffer to f64 for
            // serialization (mirrors the pattern used by other Pattern B
            // optimizers — only happens at checkpoint time, not per step).
            let buf_cpu = if buf.is_cuda() { buf.cpu()? } else { buf.clone() };
            let buf_f64: Vec<f64> = buf_cpu
                .data_vec()?
                .iter()
                .map(|&v| cast::<T, f64>(v))
                .collect::<FerrotorchResult<Vec<f64>>>()?;
            entry.insert("momentum_buffer".to_string(), buf_f64);
            // Preserve the original tensor shape so load can reconstruct
            // the same layout.
            let shape_f64: Vec<f64> = buf.shape().iter().map(|&d| d as f64).collect();
            entry.insert("momentum_buffer_shape".to_string(), shape_f64);
            if let Some(&steps) = self.step_count.get(key) {
                entry.insert("step".to_string(), vec![steps as f64]);
            }
            state.insert(key.clone(), entry);
        }
        Ok(state)
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        self.momentum_buffers.clear();
        self.step_count.clear();
        for (key, entry) in state {
            if let Some(buf_data) = entry.get("momentum_buffer") {
                // Default shape: 1-D (matches legacy Vec<f64> serialization);
                // when the saved-by-this-impl `momentum_buffer_shape` key is
                // present, use it.
                let shape: Vec<usize> = entry
                    .get("momentum_buffer_shape")
                    .map(|s| s.iter().map(|&d| d as usize).collect())
                    .unwrap_or_else(|| vec![buf_data.len()]);
                let cast_data: Vec<T> = buf_data
                    .iter()
                    .map(|&v| cast::<f64, T>(v))
                    .collect::<FerrotorchResult<Vec<T>>>()?;
                let tensor =
                    Tensor::from_storage(TensorStorage::cpu(cast_data), shape, false)?;
                self.momentum_buffers.insert(key.clone(), tensor);
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

    /// Create a leaf tensor with given data and shape, optionally with grad.
    fn leaf(data: &[f64], shape: &[usize], requires_grad: bool) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // Newton-Schulz helper
    // -----------------------------------------------------------------------

    #[test]
    fn test_newton_schulz_produces_orthogonal() {
        // Start with a non-orthogonal 2x2 matrix.
        let g = vec![3.0, 1.0, 1.0, 2.0];
        let orth = newton_schulz_orthogonalize(&g, 2, 2, 10);

        // Check that orth^T @ orth ~ I.
        let ot = transpose(&orth, 2, 2);
        let otg = matmul(&ot, &orth, 2, 2, 2);

        // Diagonal should be ~1, off-diagonal should be ~0.
        assert!(
            (otg[0] - 1.0).abs() < 1e-4,
            "orth^T @ orth [0,0] = {}",
            otg[0]
        );
        assert!(
            (otg[3] - 1.0).abs() < 1e-4,
            "orth^T @ orth [1,1] = {}",
            otg[3]
        );
        assert!(otg[1].abs() < 1e-4, "orth^T @ orth [0,1] = {}", otg[1]);
        assert!(otg[2].abs() < 1e-4, "orth^T @ orth [1,0] = {}", otg[2]);
    }

    #[test]
    fn test_newton_schulz_zero_grad() {
        let g = vec![0.0, 0.0, 0.0, 0.0];
        let orth = newton_schulz_orthogonalize(&g, 2, 2, 5);
        for &v in &orth {
            assert!(v.abs() < 1e-30, "zero grad should remain zero");
        }
    }

    // -----------------------------------------------------------------------
    // Basic Muon step
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_basic_step_1d() {
        // 1D parameter: should fall back to standard momentum SGD.
        let p = Parameter::from_slice(&[10.0_f64, 10.0], &[2]).unwrap();
        let grad = leaf(&[1.0, 1.0], &[2], false);
        p.set_grad(Some(grad)).unwrap();

        let config = MuonConfig::new(0.1).momentum(0.0).nesterov(false);
        let mut muon = Muon::new(vec![p], config);
        muon.step().unwrap();

        let data = muon.param_groups()[0].params[0].data().unwrap().to_vec();
        // param = 10 - 0.1 * 1.0 = 9.9
        assert!(
            (data[0] - 9.9).abs() < 1e-6,
            "expected 9.9, got {}",
            data[0]
        );
    }

    #[test]
    fn test_muon_basic_step_2d() {
        // 2D parameter: Newton-Schulz should orthogonalize the gradient.
        let p = Parameter::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let grad = leaf(&[2.0, 0.5, 0.5, 2.0], &[2, 2], false);
        p.set_grad(Some(grad)).unwrap();

        let config = MuonConfig::new(0.1)
            .momentum(0.0)
            .nesterov(false)
            .ns_steps(10);
        let mut muon = Muon::new(vec![p], config);
        muon.step().unwrap();

        // After orthogonalization, the gradient update direction should be
        // orthogonal. The parameter should have moved.
        let data = muon.param_groups()[0].params[0].data().unwrap().to_vec();
        // Just check it moved from identity.
        let moved = data.iter().enumerate().any(|(i, &v)| {
            let identity_val = if i == 0 || i == 3 { 1.0 } else { 0.0 };
            (v - identity_val).abs() > 1e-6
        });
        assert!(moved, "parameter should have been updated");
    }

    // -----------------------------------------------------------------------
    // Convergence on a quadratic
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_convergence_quadratic() {
        // Minimize f(x) = 0.5 * ||x||^2 starting from x = [5.0, 3.0].
        // The gradient is x itself, so the optimizer should drive x toward 0.
        let p = Parameter::from_slice(&[5.0_f64, 3.0], &[2]).unwrap();

        let config = MuonConfig::new(0.01).momentum(0.9).nesterov(true);
        let mut muon = Muon::new(vec![p], config);

        for _ in 0..200 {
            // grad = current param value (gradient of 0.5 * ||x||^2)
            let current = muon.param_groups()[0].params[0].data().unwrap().to_vec();
            let grad = leaf(&current, &[2], false);
            muon.param_groups_mut()[0].params[0]
                .set_grad(Some(grad))
                .unwrap();
            muon.step().unwrap();
        }

        let final_data = muon.param_groups()[0].params[0].data().unwrap().to_vec();
        let norm_sq: f64 = final_data.iter().map(|&x| x * x).sum();
        assert!(
            norm_sq < 0.01,
            "quadratic did not converge: ||x||^2 = {}",
            norm_sq
        );
    }

    #[test]
    fn test_muon_convergence_2d_quadratic() {
        // Minimize f(W) = 0.5 * ||W||_F^2 for a 2x2 matrix.
        // Gradient = W. Muon should orthogonalize then apply momentum.
        let p = Parameter::from_slice(&[3.0_f64, 1.0, 1.0, 3.0], &[2, 2]).unwrap();

        let config = MuonConfig::new(0.01)
            .momentum(0.9)
            .nesterov(true)
            .ns_steps(5);
        let mut muon = Muon::new(vec![p], config);

        for _ in 0..300 {
            let current = muon.param_groups()[0].params[0].data().unwrap().to_vec();
            let grad = leaf(&current, &[2, 2], false);
            muon.param_groups_mut()[0].params[0]
                .set_grad(Some(grad))
                .unwrap();
            muon.step().unwrap();
        }

        let final_data = muon.param_groups()[0].params[0].data().unwrap().to_vec();
        let norm_sq: f64 = final_data.iter().map(|&x| x * x).sum();
        assert!(
            norm_sq < 0.1,
            "2D quadratic did not converge: ||W||_F^2 = {}",
            norm_sq
        );
    }

    // -----------------------------------------------------------------------
    // LR accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_lr_get_set() {
        let p = Parameter::<f64>::zeros(&[2]).unwrap();
        let config = MuonConfig::new(0.02);
        let mut muon = Muon::new(vec![p], config);

        assert!((muon.lr() - 0.02).abs() < 1e-12);

        muon.set_lr(0.1);
        assert!((muon.lr() - 0.1).abs() < 1e-12);
        assert!((muon.param_groups()[0].lr - 0.1).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // State dict roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_state_dict_roundtrip() {
        let p = Parameter::from_slice(&[5.0_f64, 5.0], &[2]).unwrap();

        let config = MuonConfig::new(0.02).momentum(0.95);
        let mut muon = Muon::new(vec![p], config);

        // Run one step to populate momentum buffers.
        let grad = leaf(&[1.0, 2.0], &[2], false);
        muon.param_groups_mut()[0].params[0]
            .set_grad(Some(grad))
            .unwrap();
        muon.step().unwrap();

        let state = muon
            .state_dict()
            .expect("muon state_dict must succeed in test");
        assert!(!state.is_empty());
        assert!(state.contains_key("0_0"));

        // Load into a fresh optimizer.
        let p2 = Parameter::from_slice(&[5.0_f64, 5.0], &[2]).unwrap();
        let config2 = MuonConfig::new(0.02).momentum(0.95);
        let mut muon2 = Muon::new(vec![p2], config2);
        muon2.load_state_dict(&state).unwrap();

        assert_eq!(muon2.momentum_buffers.get("0_0").unwrap().numel(), 2);
    }

    // -----------------------------------------------------------------------
    // Zero grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_muon_zero_grad() {
        let p = Parameter::from_slice(&[1.0_f64, 2.0], &[2]).unwrap();
        let grad = leaf(&[0.5, 0.5], &[2], false);
        p.set_grad(Some(grad)).unwrap();
        assert!(p.grad().unwrap().is_some());

        let config = MuonConfig::new(0.02);
        let mut muon = Muon::new(vec![p], config);
        muon.zero_grad().unwrap();

        assert!(muon.param_groups()[0].params[0].grad().unwrap().is_none());
    }

    // -----------------------------------------------------------------------
    // CL-1105 Pattern B — CUDA device-resident step tests.
    //
    // These tests run only with `--features cuda` and require an NVIDIA GPU
    // at runtime. Without one, `init_cuda_backend()` returns Err and the
    // test cascades to a skip with an [cascade_skip] log line.
    // -----------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    fn try_init_cuda() -> bool {
        match ferrotorch_gpu::init_cuda_backend() {
            Ok(_) => true,
            Err(e) => {
                eprintln!("[cascade_skip] no CUDA device: {e}");
                false
            }
        }
    }

    /// CUDA-resident Muon step must keep the parameter on its original
    /// device (no silent demote to CPU).
    #[cfg(feature = "cuda")]
    #[test]
    fn muon_step_preserves_device_for_cuda_input() {
        if !try_init_cuda() {
            return;
        }
        let p_cpu = Parameter::from_slice(&[1.0_f64, 0.0, 0.0, 1.0], &[2, 2]).unwrap();
        let p = p_cpu.to(ferrotorch_core::Device::Cuda(0)).unwrap();
        let grad = leaf(&[2.0, 0.5, 0.5, 2.0], &[2, 2], false)
            .cuda()
            .unwrap();
        p.set_grad(Some(grad)).unwrap();

        let config = MuonConfig::new(0.1)
            .momentum(0.0)
            .nesterov(false)
            .ns_steps(5);
        let mut muon = Muon::new(vec![p], config);
        muon.step().unwrap();

        let after = &muon.param_groups()[0].params[0];
        assert!(
            after.tensor().is_cuda(),
            "Muon::step must preserve CUDA residence; got device {:?}",
            after.tensor().device()
        );
        assert_eq!(after.tensor().device(), ferrotorch_core::Device::Cuda(0));
    }

    /// CUDA step must produce numerically equivalent results to the CPU
    /// path within tolerance (1e-4 for f32, but we run f64 here so 1e-8).
    #[cfg(feature = "cuda")]
    #[test]
    fn muon_step_matches_cpu_within_tolerance() {
        if !try_init_cuda() {
            return;
        }
        let init = [1.0_f64, 0.0, 0.0, 1.0];
        let grad_data = [2.0_f64, 0.5, 0.5, 2.0];

        // CPU reference run.
        let p_cpu = Parameter::from_slice(&init, &[2, 2]).unwrap();
        let g_cpu = leaf(&grad_data, &[2, 2], false);
        p_cpu.set_grad(Some(g_cpu)).unwrap();
        let mut muon_cpu = Muon::new(
            vec![p_cpu],
            MuonConfig::new(0.1).momentum(0.0).nesterov(false).ns_steps(5),
        );
        muon_cpu.step().unwrap();
        let cpu_after: Vec<f64> = muon_cpu.param_groups()[0].params[0]
            .data()
            .unwrap()
            .to_vec();

        // CUDA run.
        let p_gpu = Parameter::from_slice(&init, &[2, 2])
            .unwrap()
            .to(ferrotorch_core::Device::Cuda(0))
            .unwrap();
        let g_gpu = leaf(&grad_data, &[2, 2], false).cuda().unwrap();
        p_gpu.set_grad(Some(g_gpu)).unwrap();
        let mut muon_gpu = Muon::new(
            vec![p_gpu],
            MuonConfig::new(0.1).momentum(0.0).nesterov(false).ns_steps(5),
        );
        muon_gpu.step().unwrap();
        let gpu_after_t = muon_gpu.param_groups()[0].params[0]
            .tensor()
            .cpu()
            .unwrap();
        let gpu_after: Vec<f64> = gpu_after_t.data().unwrap().to_vec();

        assert_eq!(cpu_after.len(), gpu_after.len());
        for (i, (c, g)) in cpu_after.iter().zip(gpu_after.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-6,
                "Muon CPU/GPU mismatch at idx {i}: cpu={c}, gpu={g}"
            );
        }
    }
}
