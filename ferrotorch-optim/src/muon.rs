//! Muon optimizer — spectral-norm-aware SGD with Newton-Schulz orthogonalization.
//!
//! For 2D weight matrices, Muon orthogonalizes the gradient via Newton-Schulz
//! iterations before applying momentum. For non-2D parameters (biases, norms),
//! it falls back to standard momentum SGD.
//!
//! Reference: <https://arxiv.org/abs/2502.16982>

use std::collections::HashMap;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, no_grad};
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
}

impl Default for MuonConfig {
    fn default() -> Self {
        Self::new(0.02)
    }
}

// ---------------------------------------------------------------------------
// Matrix helpers (dense, row-major)
// ---------------------------------------------------------------------------

/// Compute the Frobenius norm of a flat matrix.
fn frobenius_norm(data: &[f64], _rows: usize, _cols: usize) -> f64 {
    data.iter().map(|&x| x * x).sum::<f64>().sqrt()
}

/// Matrix multiply: C = A (m x k) @ B (k x n) -> (m x n).
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
fn transpose(data: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = data[i * cols + j];
        }
    }
    t
}

/// Newton-Schulz orthogonalization of a matrix G (rows x cols).
///
/// 1. Normalize: G = G / ||G||_F
/// 2. For `ns_steps` iterations: G = G * (3*I - G^T @ G) / 2
///
/// This converges to the polar factor (orthogonal component) of G.
fn newton_schulz_orthogonalize(
    grad: &[f64],
    rows: usize,
    cols: usize,
    ns_steps: usize,
) -> Vec<f64> {
    // Normalize by Frobenius norm.
    let norm = frobenius_norm(grad, rows, cols);
    if norm < 1e-30 {
        return vec![0.0; rows * cols];
    }
    let mut g: Vec<f64> = grad.iter().map(|&x| x / norm).collect();

    for _ in 0..ns_steps {
        // G^T @ G  -> (cols x rows) @ (rows x cols) = (cols x cols)
        let gt = transpose(&g, rows, cols);
        let gtg = matmul(&gt, &g, cols, rows, cols);

        // M = 3*I - G^T @ G  (cols x cols)
        let mut m = vec![0.0; cols * cols];
        for i in 0..cols {
            for j in 0..cols {
                let idx = i * cols + j;
                let identity = if i == j { 3.0 } else { 0.0 };
                m[idx] = identity - gtg[idx];
            }
        }

        // G = G @ M / 2  -> (rows x cols) @ (cols x cols) = (rows x cols)
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
#[derive(Debug)]
pub struct Muon<T: Float> {
    /// Parameter groups.
    param_groups: Vec<ParamGroup<T>>,
    /// Global configuration.
    config: MuonConfig,
    /// Momentum buffers keyed by `"{group_idx}_{param_idx}"`.
    momentum_buffers: HashMap<String, Vec<f64>>,
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
    fn step(&mut self) -> FerrotorchResult<()> {
        let momentum = self.config.momentum;
        let nesterov = self.config.nesterov;
        let ns_steps = self.config.ns_steps;

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

                let param_data: Vec<f64> = param
                    .data_vec()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                let mut grad_data: Vec<f64> = grad_tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                let shape = param.shape().to_vec();

                // Maximize: negate gradient. CL-321
                if self.config.maximize {
                    for g in grad_data.iter_mut() {
                        *g = -*g;
                    }
                }

                // Weight decay: grad = grad + wd * param
                let wd = group_wd;
                if wd > 0.0 {
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g += wd * p;
                    }
                }

                // For 2D parameters: apply Newton-Schulz orthogonalization.
                // For non-2D: use gradient as-is (standard momentum SGD).
                let processed_grad = if shape.len() == 2 {
                    let rows = shape[0];
                    let cols = shape[1];
                    newton_schulz_orthogonalize(&grad_data, rows, cols, ns_steps)
                } else {
                    grad_data
                };

                // Momentum
                let effective_grad = if momentum > 0.0 {
                    let key = Self::buf_key(gi, pi);
                    let step = self.step_count.entry(key.clone()).or_insert(0);

                    if *step == 0 {
                        self.momentum_buffers
                            .insert(key.clone(), processed_grad.clone());
                    } else {
                        let buf = self.momentum_buffers.get_mut(&key).unwrap();
                        for (b, &g) in buf.iter_mut().zip(processed_grad.iter()) {
                            *b = momentum * *b + g;
                        }
                    }

                    *step += 1;

                    let buf = self.momentum_buffers.get(&key).unwrap();

                    if nesterov {
                        let mut nesterov_grad = processed_grad.clone();
                        for (ng, &b) in nesterov_grad.iter_mut().zip(buf.iter()) {
                            *ng += momentum * b;
                        }
                        nesterov_grad
                    } else {
                        buf.clone()
                    }
                } else {
                    processed_grad
                };

                // param = param - lr * grad
                let new_data: Vec<T> = param_data
                    .iter()
                    .zip(effective_grad.iter())
                    .map(|(&p, &g)| cast::<f64, T>(p - group_lr * g))
                    .collect::<FerrotorchResult<Vec<T>>>()?;

                no_grad(|| {
                    // SAFETY: `update_data` mutates the parameter's storage
                    // via `Arc::as_ptr`; soundness depends on a single
                    // exclusive writer.
                    //  1. Muon::step is called via `Optimizer::step(&mut
                    //     self)`, ruling out concurrent invocations through
                    //     this optimiser handle.
                    //  2. The closure body sits inside `no_grad`, so this
                    //     write does not record a `grad_fn` that would
                    //     retain a clone of the storage Arc.
                    //  3. All of the inputs we read from the parameter
                    //     earlier in the iteration (`param_data`,
                    //     `processed_grad`) were materialised as owned
                    //     `Vec<T>` values via `data_vec()`; no live `&[T]`
                    //     into the param's storage remains. `new_data` is a
                    //     fresh `Vec<T>` produced by the iterator chain
                    //     above.
                    //  4. The per-parameter loop iterates `(gi, pi)` keys
                    //     sequentially, so two iterations cannot hold
                    //     overlapping borrows.
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
            entry.insert("momentum_buffer".to_string(), buf.clone());
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
                self.momentum_buffers.insert(key.clone(), buf_data.clone());
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

        let state = muon.state_dict();
        assert!(!state.is_empty());
        assert!(state.contains_key("0_0"));

        // Load into a fresh optimizer.
        let p2 = Parameter::from_slice(&[5.0_f64, 5.0], &[2]).unwrap();
        let config2 = MuonConfig::new(0.02).momentum(0.95);
        let mut muon2 = Muon::new(vec![p2], config2);
        muon2.load_state_dict(&state).unwrap();

        assert_eq!(muon2.momentum_buffers.get("0_0").unwrap().len(), 2);
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
}
