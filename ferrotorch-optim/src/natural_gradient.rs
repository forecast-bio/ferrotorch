//! K-FAC (Kronecker-Factored Approximate Curvature) natural gradient optimizer.
//!
//! Standard gradient descent uses the Euclidean gradient.  Natural gradient
//! descent preconditions the gradient with the inverse Fisher Information
//! Matrix (FIM), yielding updates that are invariant to parameterization:
//!
//! ```text
//! theta -= lr * F^{-1} @ gradient
//! ```
//!
//! Computing and inverting the full Fisher is intractable for large networks.
//! K-FAC (Martens & Grosse, 2015) approximates the Fisher block for each
//! Linear layer as a Kronecker product:
//!
//! ```text
//! F_W  ≈  E[a a^T] ⊗ E[g g^T]           (A ⊗ G)
//! F_W^{-1} ≈ A^{-1} ⊗ G^{-1}
//! natural_grad_W = G^{-1} @ grad_W @ A^{-1}
//! ```
//!
//! where `a` is the input activation and `g` is the output gradient.
//!
//! This module provides an MVP implementation for dense (Linear) layers.
//! Convolutional layers need a different Kronecker factorization and are not
//! yet supported.
//!
//! All parameter updates execute inside `no_grad()` so the optimizer step is
//! never recorded in the autograd graph.

use std::collections::HashMap;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage, no_grad};
use ferrotorch_nn::Parameter;

use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// KfacConfig
// ---------------------------------------------------------------------------

/// Hyperparameters for the K-FAC optimizer.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct KfacConfig {
    /// Learning rate (default: 0.001).
    pub lr: f64,
    /// Tikhonov damping added to Kronecker factors before inversion
    /// (default: 1e-3).
    pub damping: f64,
    /// Momentum coefficient for the preconditioned gradient (default: 0.9).
    pub momentum: f64,
    /// How often (in steps) to recompute the Kronecker factor inverses
    /// (default: 10).  The running-average factors themselves are updated
    /// every time `update_factors` is called.
    pub update_freq: usize,
    /// Weight decay coefficient for L2 regularization (default: 0.0).
    pub weight_decay: f64,
    /// When `true`, maximize the objective by negating the gradient (default:
    /// false). CL-321
    pub maximize: bool,
}

impl Default for KfacConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            damping: 1e-3,
            momentum: 0.9,
            update_freq: 10,
            weight_decay: 0.0,
            maximize: false,
        }
    }
}

impl KfacConfig {
    /// Set the learning rate.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the Tikhonov damping added to Kronecker factors before inversion.
    #[must_use]
    pub fn with_damping(mut self, damping: f64) -> Self {
        self.damping = damping;
        self
    }

    /// Set the momentum coefficient for the preconditioned gradient.
    #[must_use]
    pub fn with_momentum(mut self, momentum: f64) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set how often (in steps) to recompute the Kronecker factor inverses.
    #[must_use]
    pub fn with_update_freq(mut self, update_freq: usize) -> Self {
        self.update_freq = update_freq;
        self
    }

    /// Set the weight decay coefficient for L2 regularization.
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

// ---------------------------------------------------------------------------
// KroneckerFactors
// ---------------------------------------------------------------------------

/// Per-parameter Kronecker factor state.
///
/// Stores the running averages of input outer products (A) and gradient outer
/// products (G), their cached inverses, and an optional momentum buffer.
#[derive(Debug, Clone)]
struct KroneckerFactors {
    /// Running average of input outer product: E[a a^T], stored row-major
    /// with shape `[in_features, in_features]`.
    a_factor: Vec<f64>,
    a_size: usize,
    /// Running average of gradient outer product: E[g g^T], stored row-major
    /// with shape `[out_features, out_features]`.
    g_factor: Vec<f64>,
    g_size: usize,
    /// Cached inverse of (A + damping * I).
    a_inv: Option<Vec<f64>>,
    /// Cached inverse of (G + damping * I).
    g_inv: Option<Vec<f64>>,
    /// Momentum buffer for the preconditioned gradient (flat, row-major).
    momentum_buf: Option<Vec<f64>>,
}

// ---------------------------------------------------------------------------
// Kfac
// ---------------------------------------------------------------------------

/// K-FAC natural gradient optimizer.
///
/// Approximates the Fisher Information Matrix as Kronecker products per
/// layer, enabling efficient natural gradient descent for Linear layers.
#[derive(Debug)]
pub struct Kfac<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: KfacConfig,
    step_count: usize,
    /// Per-parameter Kronecker factors, keyed by a user-supplied name
    /// (typically the layer/parameter name).
    factors: HashMap<String, KroneckerFactors>,
}

impl<T: Float> Kfac<T> {
    /// Create a new K-FAC optimizer for the given parameters.
    pub fn new(params: Vec<Parameter<T>>, config: KfacConfig) -> Self {
        let mut group = ParamGroup::new(params, config.lr);
        group.weight_decay = config.weight_decay;

        Self {
            param_groups: vec![group],
            config,
            step_count: 0,
            factors: HashMap::new(),
        }
    }

    /// Generate the state key for a parameter (group index, param index).
    #[inline]
    fn param_key(group_idx: usize, param_idx: usize) -> String {
        format!("g{group_idx}_p{param_idx}")
    }

    /// Update the Kronecker factors for a named parameter from the current
    /// mini-batch's input activations and output gradients.
    ///
    /// * `param_name` — an arbitrary key that links this factor update to a
    ///   specific parameter.  Must match the key used during `step`.
    /// * `input_activation` — `[batch, in_features]` tensor of layer inputs.
    /// * `output_gradient` — `[batch, out_features]` tensor of layer output
    ///   gradients (∂L/∂y).
    ///
    /// The factors are updated as exponential moving averages with the
    /// momentum coefficient from `KfacConfig::momentum`:
    ///
    /// ```text
    /// A = momentum * A + (1 - momentum) * (a^T a) / batch_size
    /// G = momentum * G + (1 - momentum) * (g^T g) / batch_size
    /// ```
    pub fn update_factors(
        &mut self,
        param_name: &str,
        input_activation: &Tensor<T>,
        output_gradient: &Tensor<T>,
    ) -> FerrotorchResult<()> {
        let a_shape = input_activation.shape();
        let g_shape = output_gradient.shape();

        if a_shape.len() != 2 || g_shape.len() != 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "update_factors expects 2-D tensors, got activation {:?} and gradient {:?}",
                    a_shape, g_shape
                ),
            });
        }

        let batch = a_shape[0];
        if g_shape[0] != batch {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "batch size mismatch: activation batch {} vs gradient batch {}",
                    batch, g_shape[0]
                ),
            });
        }

        let in_features = a_shape[1];
        let out_features = g_shape[1];

        // Read activation and gradient data as f64.
        let a_data: Vec<f64> = input_activation
            .data()?
            .iter()
            .map(|&v| cast::<T, f64>(v))
            .collect::<FerrotorchResult<Vec<f64>>>()?;
        let g_data: Vec<f64> = output_gradient
            .data()?
            .iter()
            .map(|&v| cast::<T, f64>(v))
            .collect::<FerrotorchResult<Vec<f64>>>()?;

        let batch_f = batch as f64;

        // Compute A_batch = (a^T @ a) / batch_size  [in_features x in_features]
        let mut a_batch = vec![0.0f64; in_features * in_features];
        for b in 0..batch {
            for i in 0..in_features {
                for j in 0..in_features {
                    a_batch[i * in_features + j] +=
                        a_data[b * in_features + i] * a_data[b * in_features + j];
                }
            }
        }
        for v in &mut a_batch {
            *v /= batch_f;
        }

        // Compute G_batch = (g^T @ g) / batch_size  [out_features x out_features]
        let mut g_batch = vec![0.0f64; out_features * out_features];
        for b in 0..batch {
            for i in 0..out_features {
                for j in 0..out_features {
                    g_batch[i * out_features + j] +=
                        g_data[b * out_features + i] * g_data[b * out_features + j];
                }
            }
        }
        for v in &mut g_batch {
            *v /= batch_f;
        }

        // Exponential moving average update.
        let mom = self.config.momentum;

        let entry = self.factors.entry(param_name.to_string());
        let factors = entry.or_insert_with(|| KroneckerFactors {
            a_factor: vec![0.0; in_features * in_features],
            a_size: in_features,
            g_factor: vec![0.0; out_features * out_features],
            g_size: out_features,
            a_inv: None,
            g_inv: None,
            momentum_buf: None,
        });

        // A = mom * A_old + (1 - mom) * A_batch
        for (a, &ab) in factors.a_factor.iter_mut().zip(a_batch.iter()) {
            *a = mom * *a + (1.0 - mom) * ab;
        }

        // G = mom * G_old + (1 - mom) * G_batch
        for (g, &gb) in factors.g_factor.iter_mut().zip(g_batch.iter()) {
            *g = mom * *g + (1.0 - mom) * gb;
        }

        // Invalidate cached inverses — they will be recomputed on the next
        // step (or when `update_freq` triggers).
        factors.a_inv = None;
        factors.g_inv = None;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Dense matrix helpers (f64, row-major)
// ---------------------------------------------------------------------------

/// Invert a square matrix (with damping) using ferrotorch_core::linalg::inv.
fn invert_damped(matrix: &[f64], n: usize, damping: f64) -> FerrotorchResult<Vec<f64>> {
    // Add damping * I to the matrix.
    let mut damped = matrix.to_vec();
    for i in 0..n {
        damped[i * n + i] += damping;
    }

    // Build a Tensor, call linalg::inv, extract the result.
    let data: Vec<f64> = damped;
    let t = Tensor::from_storage(TensorStorage::cpu(data), vec![n, n], false)?;
    let inv_t = ferrotorch_core::linalg::inv(&t)?;
    let inv_data = inv_t.data()?.to_vec();
    Ok(inv_data)
}

/// Matrix multiply C = A @ B (row-major, f64).
/// A: [m, k], B: [k, n] -> C: [m, n]
fn matmul_f64(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

// ---------------------------------------------------------------------------
// Optimizer trait implementation
// ---------------------------------------------------------------------------

impl<T: Float> Optimizer<T> for Kfac<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        let config = self.config;
        self.step_count += 1;
        let should_update_inv =
            self.step_count % config.update_freq == 1 || config.update_freq <= 1;

        // Precompute inverses if needed.
        if should_update_inv {
            // Collect names first to avoid borrow conflict.
            let names: Vec<String> = self.factors.keys().cloned().collect();
            for name in &names {
                let f = self.factors.get(name).unwrap();
                let a_size = f.a_size;
                let g_size = f.g_size;
                let a_factor = f.a_factor.clone();
                let g_factor = f.g_factor.clone();

                let a_inv = invert_damped(&a_factor, a_size, config.damping)?;
                let g_inv = invert_damped(&g_factor, g_size, config.damping)?;

                let f = self.factors.get_mut(name).unwrap();
                f.a_inv = Some(a_inv);
                f.g_inv = Some(g_inv);
            }
        }

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

                // Read parameter data and gradient data as f64.
                let param_data: Vec<f64> = tensor
                    .data()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                let mut grad_data: Vec<f64> = grad_tensor
                    .data()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                let shape = tensor.shape().to_vec();

                // Maximize: negate gradient. CL-321
                if self.config.maximize {
                    for g in grad_data.iter_mut() {
                        *g = -*g;
                    }
                }

                // L2 weight decay: grad = grad + weight_decay * param.
                if group_wd > 0.0 {
                    for (g, &p) in grad_data.iter_mut().zip(param_data.iter()) {
                        *g += group_wd * p;
                    }
                }

                // Ensure Kronecker factor inverses are computed (mutable
                // access, then released before the immutable borrow below).
                if let Some(f) = self.factors.get_mut(&key) {
                    if f.a_inv.is_none() {
                        f.a_inv = Some(invert_damped(&f.a_factor, f.a_size, config.damping)?);
                    }
                    if f.g_inv.is_none() {
                        f.g_inv = Some(invert_damped(&f.g_factor, f.g_size, config.damping)?);
                    }
                }

                // Now borrow immutably to compute the preconditioned gradient.
                let preconditioned = if let Some(factors) = self.factors.get(&key) {
                    let a_inv = factors.a_inv.as_ref().unwrap();
                    let g_inv = factors.g_inv.as_ref().unwrap();

                    let out_f = factors.g_size;
                    let in_f = factors.a_size;

                    // grad_W is [out_features, in_features] (row-major).
                    // natural_grad = G^{-1} @ grad_W @ A^{-1}
                    // Step 1: temp = G^{-1} @ grad_W  [out_f, in_f]
                    let temp = matmul_f64(g_inv, &grad_data, out_f, out_f, in_f);
                    // Step 2: natural_grad = temp @ A^{-1}  [out_f, in_f]

                    matmul_f64(&temp, a_inv, out_f, in_f, in_f)
                } else {
                    // No Kronecker factors registered — fall back to vanilla
                    // gradient (behaves like SGD).
                    grad_data
                };

                // Apply momentum to the preconditioned gradient.
                let effective_grad = if config.momentum > 0.0 {
                    // We need a mutable reference to factors for momentum buf.
                    // Use the same key.
                    let entry = self.factors.entry(key.clone());
                    let factors = entry.or_insert_with(|| KroneckerFactors {
                        a_factor: Vec::new(),
                        a_size: 0,
                        g_factor: Vec::new(),
                        g_size: 0,
                        a_inv: None,
                        g_inv: None,
                        momentum_buf: None,
                    });

                    let mom = config.momentum;
                    if let Some(ref mut buf) = factors.momentum_buf {
                        // buf = mom * buf + preconditioned
                        for (b, &g) in buf.iter_mut().zip(preconditioned.iter()) {
                            *b = mom * *b + g;
                        }
                    } else {
                        factors.momentum_buf = Some(preconditioned.clone());
                    }

                    factors.momentum_buf.as_ref().unwrap().clone()
                } else {
                    preconditioned
                };

                // Update: param -= lr * effective_grad
                let numel = param_data.len();
                let new_param_data: Vec<T> = (0..numel)
                    .map(|i| {
                        let updated = param_data[i] - group_lr * effective_grad[i];
                        cast::<f64, T>(updated)
                    })
                    .collect::<FerrotorchResult<Vec<T>>>()?;

                // Write updated parameter data inside no_grad.
                let shape_clone = shape.clone();
                let new_tensor = no_grad(|| {
                    Tensor::from_storage(TensorStorage::cpu(new_param_data), shape_clone, true)
                })?;

                self.param_groups[gi].params[pi] = Parameter::new(new_tensor);
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

    fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
        let mut out = OptimizerState::new();

        // Serialize step count as a pseudo-entry.
        {
            let mut meta = HashMap::new();
            meta.insert("step_count".to_string(), vec![self.step_count as f64]);
            out.insert("__kfac_meta__".to_string(), meta);
        }

        // Serialize each factor set.
        for (name, factors) in &self.factors {
            let mut entry = HashMap::new();
            entry.insert("a_size".to_string(), vec![factors.a_size as f64]);
            entry.insert("g_size".to_string(), vec![factors.g_size as f64]);
            entry.insert("a_factor".to_string(), factors.a_factor.clone());
            entry.insert("g_factor".to_string(), factors.g_factor.clone());
            if let Some(ref buf) = factors.momentum_buf {
                entry.insert("momentum_buf".to_string(), buf.clone());
            }
            out.insert(name.clone(), entry);
        }

        Ok(out)
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        // Restore step count.
        if let Some(meta) = state.get("__kfac_meta__") {
            if let Some(sc) = meta.get("step_count").and_then(|v| v.first()) {
                self.step_count = *sc as usize;
            }
        }

        // Restore factors.
        self.factors.clear();
        for (name, entry) in state {
            if name == "__kfac_meta__" {
                continue;
            }

            let a_size = entry
                .get("a_size")
                .and_then(|v| v.first())
                .copied()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing a_size in state for key {name}"),
                })? as usize;

            let g_size = entry
                .get("g_size")
                .and_then(|v| v.first())
                .copied()
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("missing g_size in state for key {name}"),
                })? as usize;

            let a_factor =
                entry
                    .get("a_factor")
                    .cloned()
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!("missing a_factor in state for key {name}"),
                    })?;

            let g_factor =
                entry
                    .get("g_factor")
                    .cloned()
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!("missing g_factor in state for key {name}"),
                    })?;

            let momentum_buf = entry.get("momentum_buf").cloned();

            self.factors.insert(
                name.clone(),
                KroneckerFactors {
                    a_factor,
                    a_size,
                    g_factor,
                    g_size,
                    a_inv: None,
                    g_inv: None,
                    momentum_buf,
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
    use ferrotorch_core::grad_fns::arithmetic::pow;

    /// Create a scalar parameter from a single f64 value.
    fn scalar_param(val: f64) -> Parameter<f64> {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap();
        Parameter::new(t)
    }

    /// Create a parameter from flat data and shape.
    fn param_from(data: &[f64], shape: &[usize]) -> Parameter<f64> {
        Parameter::from_slice(data, shape).unwrap()
    }

    /// Read a scalar parameter's current value.
    fn param_val(opt: &Kfac<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    // -----------------------------------------------------------------------
    // KfacConfig defaults
    // -----------------------------------------------------------------------

    #[test]
    fn test_kfac_config_defaults() {
        let config = KfacConfig::default();
        assert!((config.lr - 1e-3).abs() < 1e-12);
        assert!((config.damping - 1e-3).abs() < 1e-12);
        assert!((config.momentum - 0.9).abs() < 1e-12);
        assert_eq!(config.update_freq, 10);
        assert!((config.weight_decay - 0.0).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Kfac construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_kfac_construction() {
        let p1 = param_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let p2 = param_from(&[5.0, 6.0], &[2]);

        let config = KfacConfig {
            lr: 0.01,
            damping: 1e-4,
            ..Default::default()
        };

        let kfac = Kfac::new(vec![p1, p2], config);

        assert_eq!(kfac.param_groups.len(), 1);
        assert_eq!(kfac.param_groups[0].params.len(), 2);
        assert!((kfac.lr() - 0.01).abs() < 1e-12);
        assert_eq!(kfac.step_count, 0);
        assert!(kfac.factors.is_empty());
    }

    // -----------------------------------------------------------------------
    // update_factors stores running averages
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_factors_stores_running_averages() {
        let p = param_from(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let config = KfacConfig {
            momentum: 0.0, // No EMA blending — just take the batch estimate.
            ..Default::default()
        };
        let mut kfac = Kfac::new(vec![p], config);

        // Batch of 2 samples, in_features=3.
        let activation = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            vec![2, 3],
            false,
        )
        .unwrap();

        // out_features=2.
        let gradient = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
            vec![2, 2],
            false,
        )
        .unwrap();

        kfac.update_factors("g0_p0", &activation, &gradient)
            .unwrap();

        let factors = kfac.factors.get("g0_p0").unwrap();

        // A = (a^T a) / 2 with the given data:
        //   a = [[1,0,0],[0,1,0]]
        //   a^T a = [[1,0,0],[0,1,0],[0,0,0]]
        //   / 2 = [[0.5,0,0],[0,0.5,0],[0,0,0]]
        assert_eq!(factors.a_size, 3);
        assert!((factors.a_factor[0] - 0.5).abs() < 1e-12); // (0,0)
        assert!((factors.a_factor[4] - 0.5).abs() < 1e-12); // (1,1)
        assert!((factors.a_factor[8] - 0.0).abs() < 1e-12); // (2,2)
        assert!((factors.a_factor[1] - 0.0).abs() < 1e-12); // off-diag

        // G = (g^T g) / 2 with the given data:
        //   g = [[1,0],[0,1]]
        //   g^T g = [[1,0],[0,1]]
        //   / 2 = [[0.5,0],[0,0.5]]
        assert_eq!(factors.g_size, 2);
        assert!((factors.g_factor[0] - 0.5).abs() < 1e-12);
        assert!((factors.g_factor[3] - 0.5).abs() < 1e-12);
        assert!((factors.g_factor[1] - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_update_factors_ema_blending() {
        let p = param_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let config = KfacConfig {
            momentum: 0.5, // 50/50 blend
            ..Default::default()
        };
        let mut kfac = Kfac::new(vec![p], config);

        // First update: identity-like activations and gradients.
        let act1 = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let grad1 = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
            vec![2, 2],
            false,
        )
        .unwrap();

        kfac.update_factors("test", &act1, &grad1).unwrap();

        // First update: A = 0.5 * 0 + 0.5 * [[0.5,0],[0,0.5]] = [[0.25,0],[0,0.25]]
        let f = kfac.factors.get("test").unwrap();
        assert!((f.a_factor[0] - 0.25).abs() < 1e-12);

        // Second update with same data:
        // A = 0.5 * [[0.25,0],[0,0.25]] + 0.5 * [[0.5,0],[0,0.5]]
        //   = [[0.125,0],[0,0.125]] + [[0.25,0],[0,0.25]]
        //   = [[0.375,0],[0,0.375]]
        kfac.update_factors("test", &act1, &grad1).unwrap();

        let f = kfac.factors.get("test").unwrap();
        assert!((f.a_factor[0] - 0.375).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // step with identity factors matches SGD
    // -----------------------------------------------------------------------

    #[test]
    fn test_step_with_identity_factors_matches_sgd() {
        // When A = I and G = I, the preconditioned gradient should equal the
        // raw gradient (up to damping).  With damping=0 and identity factors,
        // K-FAC step = SGD step exactly.
        //
        // We set up a 2x3 weight parameter, manually set its gradient, set
        // up identity Kronecker factors, and verify the update matches plain
        // SGD: param -= lr * grad.
        let lr = 0.1;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let grad_data = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let p = param_from(&data, &[2, 3]);

        // Set gradient manually.
        let grad_t =
            Tensor::from_storage(TensorStorage::cpu(grad_data.clone()), vec![2, 3], false).unwrap();
        p.tensor().set_grad(Some(grad_t)).unwrap();

        let config = KfacConfig {
            lr,
            damping: 0.0,
            momentum: 0.0,
            update_freq: 1,
            weight_decay: 0.0,
            maximize: false,
        };
        let mut kfac = Kfac::new(vec![p], config);

        // Manually insert identity Kronecker factors.
        //   A = I_3 (in_features=3), G = I_2 (out_features=2)
        // These represent an identity Fisher, so F^{-1} = I and
        // natural_grad = G^{-1} @ grad @ A^{-1} = I @ grad @ I = grad.
        let key = Kfac::<f64>::param_key(0, 0);
        kfac.factors.insert(
            key.clone(),
            KroneckerFactors {
                a_factor: vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
                a_size: 3,
                g_factor: vec![1.0, 0.0, 0.0, 1.0],
                g_size: 2,
                a_inv: None,
                g_inv: None,
                momentum_buf: None,
            },
        );

        kfac.step().unwrap();

        // Expected: param_i - lr * grad_i
        let updated = kfac.param_groups[0].params[0].data().unwrap().to_vec();
        for i in 0..6 {
            let expected = data[i] - lr * grad_data[i];
            assert!(
                (updated[i] - expected).abs() < 1e-10,
                "element {i}: expected {expected}, got {}",
                updated[i]
            );
        }
    }

    // -----------------------------------------------------------------------
    // Convergence on a simple quadratic
    // -----------------------------------------------------------------------

    #[test]
    fn test_convergence_quadratic() {
        // Minimize f(x) = x^2 starting from x=5.
        // K-FAC (even without meaningful Kronecker factors) should converge
        // since it falls back to SGD when no factors are registered.
        let p = scalar_param(5.0);
        let config = KfacConfig {
            lr: 0.01,
            damping: 1e-3,
            momentum: 0.0,
            update_freq: 1,
            weight_decay: 0.0,
            maximize: false,
        };
        let mut kfac = Kfac::new(vec![p], config);

        for _ in 0..1000 {
            kfac.zero_grad().unwrap();

            let tensor = kfac.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();

            kfac.step().unwrap();
        }

        let final_val = param_val(&kfac, 0, 0);
        assert!(
            final_val.abs() < 0.1,
            "expected convergence near 0, got {final_val}"
        );
    }

    #[test]
    fn test_convergence_with_kfac_factors() {
        // Minimize a 2-D quadratic f(w) = 0.5 * (W[0]^2 + W[1]^2) with
        // K-FAC preconditioning.  We simulate a single Linear layer
        // y = Wx with W: [1, 2].
        //
        // We provide well-conditioned Kronecker factors (identity-like
        // activations from a batch of 2), which makes the natural gradient
        // equivalent to a well-preconditioned update.

        let lr = 0.01;
        let w_init = vec![3.0, -4.0];
        let p = param_from(&w_init, &[1, 2]);

        let config = KfacConfig {
            lr,
            damping: 0.1,
            momentum: 0.0,
            update_freq: 1,
            weight_decay: 0.0,
            maximize: false,
        };
        let mut kfac = Kfac::new(vec![p], config);

        let key = Kfac::<f64>::param_key(0, 0);

        for _step in 0..500 {
            kfac.zero_grad().unwrap();

            let tensor = kfac.param_groups[0].params[0].tensor().clone();

            // Manual gradient: grad = W (gradient of 0.5 * ||W||^2)
            let w_data = tensor.data().unwrap().to_vec();
            let grad_t =
                Tensor::from_storage(TensorStorage::cpu(w_data.clone()), vec![1, 2], false)
                    .unwrap();
            tensor.set_grad(Some(grad_t)).unwrap();

            // Update Kronecker factors with well-conditioned activations.
            // Batch of 2 with identity-like rows => A ≈ I.
            let act = Tensor::from_storage(
                TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
                vec![2, 2],
                false,
            )
            .unwrap();
            // Output gradient: batch of 2, out_features=1 => G = scalar.
            let grad_out =
                Tensor::from_storage(TensorStorage::cpu(vec![1.0, 1.0]), vec![2, 1], false)
                    .unwrap();

            kfac.update_factors(&key, &act, &grad_out).unwrap();

            kfac.step().unwrap();
        }

        let final_w = kfac.param_groups[0].params[0].data().unwrap().to_vec();
        assert!(
            final_w[0].abs() < 0.1,
            "expected W[0] near 0, got {}",
            final_w[0]
        );
        assert!(
            final_w[1].abs() < 0.1,
            "expected W[1] near 0, got {}",
            final_w[1]
        );
    }

    // -----------------------------------------------------------------------
    // state_dict / load_state_dict roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_dict_roundtrip() {
        let p = param_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let config = KfacConfig {
            momentum: 0.0,
            update_freq: 1,
            ..Default::default()
        };
        let mut kfac = Kfac::new(vec![p], config);

        // Register some factors.
        let key = Kfac::<f64>::param_key(0, 0);
        let act = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let grad_out = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        kfac.update_factors(&key, &act, &grad_out).unwrap();

        // Do a step to advance step_count and populate momentum.
        let grad_t = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1, 0.2, 0.3, 0.4]),
            vec![2, 2],
            false,
        )
        .unwrap();
        kfac.param_groups[0].params[0]
            .tensor()
            .set_grad(Some(grad_t))
            .unwrap();
        kfac.step().unwrap();

        // Save state.
        let saved = kfac
            .state_dict()
            .expect("kfac state_dict must succeed in test");
        assert!(!saved.is_empty());
        assert!(saved.contains_key("__kfac_meta__"));
        assert!(saved.contains_key(&key));

        let meta = &saved["__kfac_meta__"];
        assert_eq!(meta["step_count"][0] as usize, 1);

        let entry = &saved[&key];
        assert!(entry.contains_key("a_factor"));
        assert!(entry.contains_key("g_factor"));
        assert!(entry.contains_key("a_size"));
        assert!(entry.contains_key("g_size"));

        // Load into a fresh optimizer.
        let p2 = param_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mut kfac2 = Kfac::new(vec![p2], config);
        kfac2.load_state_dict(&saved).unwrap();

        assert_eq!(kfac2.step_count, 1);
        assert!(kfac2.factors.contains_key(&key));

        let loaded = kfac2
            .state_dict()
            .expect("kfac state_dict round-trip must succeed in test");
        assert_eq!(loaded[&key]["a_factor"], saved[&key]["a_factor"]);
        assert_eq!(loaded[&key]["g_factor"], saved[&key]["g_factor"]);
    }

    // -----------------------------------------------------------------------
    // Learning rate accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_kfac_lr_accessors() {
        let p = scalar_param(1.0);
        let mut kfac = Kfac::new(
            vec![p],
            KfacConfig {
                lr: 0.05,
                ..Default::default()
            },
        );

        assert!((kfac.lr() - 0.05).abs() < 1e-12);

        kfac.set_lr(0.01);
        assert!((kfac.lr() - 0.01).abs() < 1e-12);
        assert!((kfac.param_groups()[0].lr - 0.01).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Weight decay
    // -----------------------------------------------------------------------

    #[test]
    fn test_kfac_weight_decay() {
        // Without Kronecker factors, K-FAC falls back to SGD with weight decay.
        // grad = grad + wd * param = 0 + 0.1 * 5 = 0.5
        // param = 5 - 0.1 * 0.5 = 4.95
        let p = Parameter::from_slice(&[5.0_f64], &[1]).unwrap();
        let grad = Tensor::from_storage(TensorStorage::cpu(vec![0.0_f64]), vec![1], false).unwrap();
        p.tensor().set_grad(Some(grad)).unwrap();

        let config = KfacConfig {
            lr: 0.1,
            weight_decay: 0.1,
            momentum: 0.0,
            ..Default::default()
        };
        let mut kfac = Kfac::new(vec![p], config);
        kfac.step().unwrap();

        let data = kfac.param_groups[0].params[0].data().unwrap().to_vec();
        assert!(
            (data[0] - 4.95).abs() < 1e-10,
            "expected 4.95, got {}",
            data[0]
        );
    }

    // -----------------------------------------------------------------------
    // update_factors rejects bad shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_update_factors_rejects_1d() {
        let p = param_from(&[1.0, 2.0], &[2]);
        let mut kfac = Kfac::new(vec![p], KfacConfig::default());

        let act = Tensor::from_storage(TensorStorage::cpu(vec![1.0, 2.0]), vec![2], false).unwrap();
        let grad =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 2.0]), vec![2], false).unwrap();

        let result = kfac.update_factors("test", &act, &grad);
        assert!(result.is_err());
    }

    #[test]
    fn test_update_factors_rejects_batch_mismatch() {
        let p = param_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let mut kfac = Kfac::new(vec![p], KfacConfig::default());

        let act = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
            vec![2, 2],
            false,
        )
        .unwrap();
        let grad = Tensor::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3, 1], false)
            .unwrap();

        let result = kfac.update_factors("test", &act, &grad);
        assert!(result.is_err());
    }
}
