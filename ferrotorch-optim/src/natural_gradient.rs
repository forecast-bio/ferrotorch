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

use ferrotorch_core::creation::{eye, scalar};
use ferrotorch_core::grad_fns::arithmetic::{add, mul, sub};
use ferrotorch_core::numeric_cast::cast;
// CL-1105 Pattern B correctness: use the differentiable matmul, which has
// the CUDA dispatch (cuBLAS GEMM) wired up; the `ops::linalg::matmul`
// alternative calls `.data()?` and surfaces `GpuTensorNotAccessible` on
// CUDA tensors. The autograd graph is suppressed by `no_grad` at every
// call site in `update_factors` and `step`.
use ferrotorch_core::grad_fns::linalg::matmul_differentiable as tensor_matmul;
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
/// CL-1105 Pattern B: factors live as device-resident [`Tensor<T>`] so the
/// step body composes device-aware arithmetic + matmul + solve ops with no
/// `data_vec()` round-trip when the parameters live on CUDA. The factors are
/// `[a_size, a_size]` and `[g_size, g_size]` 2-D tensors.
#[derive(Debug, Clone)]
struct KroneckerFactors<T: Float> {
    /// Running average of input outer product: E[a a^T], 2-D tensor of
    /// shape `[in_features, in_features]` on the parameter's device.
    a_factor: Tensor<T>,
    a_size: usize,
    /// Running average of gradient outer product: E[g g^T], 2-D tensor of
    /// shape `[out_features, out_features]` on the parameter's device.
    g_factor: Tensor<T>,
    g_size: usize,
    /// Cached inverse of `(A + damping * I)` (only used when no GPU
    /// backend is available; on GPU the step uses `solve` directly which
    /// is faster than forming the inverse).
    a_inv: Option<Tensor<T>>,
    /// Cached inverse of `(G + damping * I)` (CPU-only cache).
    g_inv: Option<Tensor<T>>,
    /// Momentum buffer for the preconditioned gradient (same shape as the
    /// gradient, on the parameter's device).
    momentum_buf: Option<Tensor<T>>,
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
    factors: HashMap<String, KroneckerFactors<T>>,
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
        let device = input_activation.device();

        // CL-1105 Pattern B: device-resident outer-product accumulation.
        // A_batch = (a^T @ a) / batch  uses tensor matmul, runs on the
        // input's device (cuBLAS GEMM on CUDA).
        let inv_batch_t = scalar(cast::<f64, T>(1.0 / (batch as f64))?)?.to(device)?;
        let a_t = no_grad(|| input_activation.t())?;
        let a_batch_unscaled = no_grad(|| tensor_matmul(&a_t, input_activation))?;
        let a_batch = no_grad(|| mul(&a_batch_unscaled, &inv_batch_t))?;

        let g_t = no_grad(|| output_gradient.t())?;
        let g_batch_unscaled = no_grad(|| tensor_matmul(&g_t, output_gradient))?;
        let g_batch = no_grad(|| mul(&g_batch_unscaled, &inv_batch_t))?;

        // Exponential moving average update (device-resident).
        let mom = self.config.momentum;
        let mom_t = scalar(cast::<f64, T>(mom)?)?.to(device)?;
        let one_minus_mom_t = scalar(cast::<f64, T>(1.0 - mom)?)?.to(device)?;

        // Initialize factor tensors on the activation's device when first
        // seen — zero-initialized to match the scalar-loop semantics.
        let key_owned = param_name.to_string();
        let needs_init = !self.factors.contains_key(&key_owned);
        if needs_init {
            let zero_a = ferrotorch_core::creation::zeros::<T>(&[in_features, in_features])?
                .to(device)?;
            let zero_g = ferrotorch_core::creation::zeros::<T>(&[out_features, out_features])?
                .to(device)?;
            self.factors.insert(
                key_owned.clone(),
                KroneckerFactors {
                    a_factor: zero_a,
                    a_size: in_features,
                    g_factor: zero_g,
                    g_size: out_features,
                    a_inv: None,
                    g_inv: None,
                    momentum_buf: None,
                },
            );
        }

        let factors = self.factors.get_mut(&key_owned).unwrap();
        // If the existing factors live on a different device (e.g. loaded
        // from a state_dict on CPU then params moved to CUDA), migrate
        // them once on first reuse.
        if factors.a_factor.device() != device {
            factors.a_factor = factors.a_factor.clone().to(device)?;
        }
        if factors.g_factor.device() != device {
            factors.g_factor = factors.g_factor.clone().to(device)?;
        }

        // A = mom * A_old + (1 - mom) * A_batch
        let a_old_scaled = no_grad(|| mul(&factors.a_factor, &mom_t))?;
        let a_batch_scaled = no_grad(|| mul(&a_batch, &one_minus_mom_t))?;
        factors.a_factor = no_grad(|| add(&a_old_scaled, &a_batch_scaled))?;

        // G = mom * G_old + (1 - mom) * G_batch
        let g_old_scaled = no_grad(|| mul(&factors.g_factor, &mom_t))?;
        let g_batch_scaled = no_grad(|| mul(&g_batch, &one_minus_mom_t))?;
        factors.g_factor = no_grad(|| add(&g_old_scaled, &g_batch_scaled))?;

        // Invalidate cached inverses — they will be recomputed on the next
        // step (or when `update_freq` triggers).
        factors.a_inv = None;
        factors.g_inv = None;

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tensor matrix helpers — device-aware (CL-1105 Pattern B)
// ---------------------------------------------------------------------------

/// Compute `(matrix + damping * I)^{-1}` as a [`Tensor<T>`] on the input's
/// device.
///
/// On CUDA, `linalg::solve` dispatches to cuSOLVER's
/// `cusolverDnXgetrf` + `cusolverDnXgetrs` (LU + triangular solve), so this
/// routine stays GPU-resident from input to output. On CPU it falls
/// through to `linalg::inv` (which routes to LAPACK via ferray-linalg).
fn invert_damped_tensor<T: Float>(matrix: &Tensor<T>, damping: f64) -> FerrotorchResult<Tensor<T>> {
    let shape = matrix.shape();
    debug_assert_eq!(
        shape.len(),
        2,
        "invert_damped_tensor expects a 2-D square tensor"
    );
    debug_assert_eq!(
        shape[0], shape[1],
        "invert_damped_tensor expects a square matrix"
    );
    let n = shape[0];
    let device = matrix.device();

    // Build identity on the matrix's device.
    let identity_cpu = eye::<T>(n)?;
    let identity = identity_cpu.to(device)?;

    // damped = matrix + damping * I
    let damping_t = scalar(cast::<f64, T>(damping)?)?.to(device)?;
    let damping_i = mul(&identity, &damping_t)?;
    let damped = add(matrix, &damping_i)?;

    // Solve `damped @ X = I` -> X = damped^{-1}. On CUDA this dispatches
    // to the GPU `solve` path (`cusolver::gpu_solve_*`); on CPU it falls
    // through to LAPACK `getrs`.
    ferrotorch_core::linalg::solve(&damped, &identity)
}

// ---------------------------------------------------------------------------
// Optimizer trait implementation
// ---------------------------------------------------------------------------

impl<T: Float> Optimizer<T> for Kfac<T> {
    /// Run one optimizer step.
    ///
    /// CL-1105: device-resident Pattern B. Factor inversion uses
    /// device-aware [`ferrotorch_core::linalg::solve`] (cuSOLVER on CUDA);
    /// the preconditioned gradient is composed via [`tensor_matmul`]
    /// (cuBLAS on CUDA). Parameter tensors stay on their original device
    /// throughout the step.
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
                let (a_factor, g_factor) = {
                    let f = self.factors.get(name).unwrap();
                    (f.a_factor.clone(), f.g_factor.clone())
                };

                let a_inv = no_grad(|| invert_damped_tensor(&a_factor, config.damping))?;
                let g_inv = no_grad(|| invert_damped_tensor(&g_factor, config.damping))?;

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
                let param_t = param.tensor().clone();
                let device = param_t.device();
                let shape = param_t.shape().to_vec();

                // Skip parameters without gradients.
                let grad_tensor = match param_t.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                let key = Self::param_key(gi, pi);

                no_grad(|| -> FerrotorchResult<()> {
                    // grad: device-resident (negated for maximize).
                    let mut grad: Tensor<T> = if self.config.maximize {
                        ferrotorch_core::grad_fns::arithmetic::neg(&grad_tensor)?
                    } else {
                        grad_tensor.clone()
                    };

                    // L2 weight decay: grad = grad + weight_decay * param.
                    if group_wd > 0.0 {
                        let wd_t = scalar(cast::<f64, T>(group_wd)?)?.to(device)?;
                        let weighted = mul(&param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    // Ensure Kronecker factor inverses are computed and
                    // resident on `device` (mutable access; released before
                    // the immutable read below).
                    if let Some(f) = self.factors.get_mut(&key) {
                        if f.a_factor.device() != device {
                            f.a_factor = f.a_factor.clone().to(device)?;
                        }
                        if f.g_factor.device() != device {
                            f.g_factor = f.g_factor.clone().to(device)?;
                        }
                        if f.a_inv.is_none() {
                            f.a_inv = Some(invert_damped_tensor(&f.a_factor, config.damping)?);
                        } else if f.a_inv.as_ref().unwrap().device() != device {
                            let a_inv = f.a_inv.take().unwrap();
                            f.a_inv = Some(a_inv.to(device)?);
                        }
                        if f.g_inv.is_none() {
                            f.g_inv = Some(invert_damped_tensor(&f.g_factor, config.damping)?);
                        } else if f.g_inv.as_ref().unwrap().device() != device {
                            let g_inv = f.g_inv.take().unwrap();
                            f.g_inv = Some(g_inv.to(device)?);
                        }
                    }

                    // Compute the preconditioned gradient. When factors exist:
                    //   natural_grad = G^{-1} @ grad_W @ A^{-1}
                    // For non-2D parameters we skip preconditioning entirely
                    // (Kronecker factorization is only defined for Linear
                    // weights; the original implementation also fell back
                    // here whenever the user did not register factors).
                    let preconditioned = if let Some(factors) = self.factors.get(&key) {
                        if shape.len() != 2 {
                            grad
                        } else {
                            let a_inv = factors.a_inv.as_ref().unwrap();
                            let g_inv = factors.g_inv.as_ref().unwrap();
                            // temp = G^{-1} @ grad   [out_f, in_f]
                            let temp = tensor_matmul(g_inv, &grad)?;
                            // natural_grad = temp @ A^{-1}  [out_f, in_f]
                            tensor_matmul(&temp, a_inv)?
                        }
                    } else {
                        // No factors registered — vanilla gradient (SGD).
                        grad
                    };

                    // Apply momentum to the preconditioned gradient.
                    let effective_grad = if config.momentum > 0.0 {
                        let mom_t = scalar(cast::<f64, T>(config.momentum)?)?.to(device)?;

                        // Ensure a factors entry exists for the momentum buf
                        // (matches the legacy code, which created a
                        // zero-sized factors record purely as a momentum
                        // container when no factors were registered).
                        let needs_init = !self.factors.contains_key(&key);
                        if needs_init {
                            let zero_a = ferrotorch_core::creation::zeros::<T>(&[0, 0])?;
                            let zero_g = ferrotorch_core::creation::zeros::<T>(&[0, 0])?;
                            self.factors.insert(
                                key.clone(),
                                KroneckerFactors {
                                    a_factor: zero_a,
                                    a_size: 0,
                                    g_factor: zero_g,
                                    g_size: 0,
                                    a_inv: None,
                                    g_inv: None,
                                    momentum_buf: None,
                                },
                            );
                        }

                        let factors = self.factors.get_mut(&key).unwrap();
                        if let Some(ref buf) = factors.momentum_buf {
                            // buf = mom * buf + preconditioned
                            let scaled = mul(buf, &mom_t)?;
                            let new_buf = add(&scaled, &preconditioned)?;
                            factors.momentum_buf = Some(new_buf);
                        } else {
                            factors.momentum_buf = Some(preconditioned.clone());
                        }

                        factors.momentum_buf.as_ref().unwrap().clone()
                    } else {
                        preconditioned
                    };

                    // param = param - lr * effective_grad
                    let lr_t = scalar(cast::<f64, T>(group_lr)?)?.to(device)?;
                    let scaled = mul(&effective_grad, &lr_t)?;
                    let new_param = sub(&param_t, &scaled)?;

                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: same as Muon::step — we own a `&mut self`
                    // handle, the closure runs inside `no_grad`, all
                    // intermediate tensors are freshly allocated and about
                    // to drop, and `new_param` was consumed into `storage`.
                    // The new storage is on `device` and has matching numel.
                    unsafe { param_t.update_storage(storage)? };

                    Ok(())
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

    fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
        let mut out = OptimizerState::new();

        // Serialize step count as a pseudo-entry.
        {
            let mut meta = HashMap::new();
            meta.insert("step_count".to_string(), vec![self.step_count as f64]);
            out.insert("__kfac_meta__".to_string(), meta);
        }

        // Serialize each factor set. Tensors are downloaded to host once at
        // checkpoint time (not per step) and serialised as f64 to keep the
        // on-disk format dtype-agnostic.
        for (name, factors) in &self.factors {
            let mut entry = HashMap::new();
            entry.insert("a_size".to_string(), vec![factors.a_size as f64]);
            entry.insert("g_size".to_string(), vec![factors.g_size as f64]);

            let a_cpu = if factors.a_factor.is_cuda() {
                factors.a_factor.cpu()?
            } else {
                factors.a_factor.clone()
            };
            let a_f64: Vec<f64> = a_cpu
                .data_vec()?
                .iter()
                .map(|&v| cast::<T, f64>(v))
                .collect::<FerrotorchResult<Vec<f64>>>()?;
            entry.insert("a_factor".to_string(), a_f64);

            let g_cpu = if factors.g_factor.is_cuda() {
                factors.g_factor.cpu()?
            } else {
                factors.g_factor.clone()
            };
            let g_f64: Vec<f64> = g_cpu
                .data_vec()?
                .iter()
                .map(|&v| cast::<T, f64>(v))
                .collect::<FerrotorchResult<Vec<f64>>>()?;
            entry.insert("g_factor".to_string(), g_f64);

            if let Some(ref buf) = factors.momentum_buf {
                let buf_cpu = if buf.is_cuda() { buf.cpu()? } else { buf.clone() };
                let buf_f64: Vec<f64> = buf_cpu
                    .data_vec()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                entry.insert("momentum_buf".to_string(), buf_f64);
                let buf_shape: Vec<f64> = buf.shape().iter().map(|&d| d as f64).collect();
                entry.insert("momentum_buf_shape".to_string(), buf_shape);
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

            let a_factor_f64 =
                entry
                    .get("a_factor")
                    .cloned()
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!("missing a_factor in state for key {name}"),
                    })?;
            let a_factor_t: Vec<T> = a_factor_f64
                .iter()
                .map(|&v| cast::<f64, T>(v))
                .collect::<FerrotorchResult<Vec<T>>>()?;
            let a_factor = Tensor::from_storage(
                TensorStorage::cpu(a_factor_t),
                vec![a_size, a_size],
                false,
            )?;

            let g_factor_f64 =
                entry
                    .get("g_factor")
                    .cloned()
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!("missing g_factor in state for key {name}"),
                    })?;
            let g_factor_t: Vec<T> = g_factor_f64
                .iter()
                .map(|&v| cast::<f64, T>(v))
                .collect::<FerrotorchResult<Vec<T>>>()?;
            let g_factor = Tensor::from_storage(
                TensorStorage::cpu(g_factor_t),
                vec![g_size, g_size],
                false,
            )?;

            let momentum_buf = if let Some(buf_f64) = entry.get("momentum_buf") {
                let buf_t: Vec<T> = buf_f64
                    .iter()
                    .map(|&v| cast::<f64, T>(v))
                    .collect::<FerrotorchResult<Vec<T>>>()?;
                let buf_shape: Vec<usize> = entry
                    .get("momentum_buf_shape")
                    .map(|s| s.iter().map(|&d| d as usize).collect())
                    .unwrap_or_else(|| vec![buf_t.len()]);
                Some(Tensor::from_storage(
                    TensorStorage::cpu(buf_t),
                    buf_shape,
                    false,
                )?)
            } else {
                None
            };

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
        let a_data = factors.a_factor.data().unwrap().to_vec();
        assert!((a_data[0] - 0.5).abs() < 1e-12); // (0,0)
        assert!((a_data[4] - 0.5).abs() < 1e-12); // (1,1)
        assert!((a_data[8] - 0.0).abs() < 1e-12); // (2,2)
        assert!((a_data[1] - 0.0).abs() < 1e-12); // off-diag

        // G = (g^T g) / 2 with the given data:
        //   g = [[1,0],[0,1]]
        //   g^T g = [[1,0],[0,1]]
        //   / 2 = [[0.5,0],[0,0.5]]
        assert_eq!(factors.g_size, 2);
        let g_data = factors.g_factor.data().unwrap().to_vec();
        assert!((g_data[0] - 0.5).abs() < 1e-12);
        assert!((g_data[3] - 0.5).abs() < 1e-12);
        assert!((g_data[1] - 0.0).abs() < 1e-12);
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
        {
            let f = kfac.factors.get("test").unwrap();
            let a_data = f.a_factor.data().unwrap().to_vec();
            assert!((a_data[0] - 0.25).abs() < 1e-12);
        }

        // Second update with same data:
        // A = 0.5 * [[0.25,0],[0,0.25]] + 0.5 * [[0.5,0],[0,0.5]]
        //   = [[0.125,0],[0,0.125]] + [[0.25,0],[0,0.25]]
        //   = [[0.375,0],[0,0.375]]
        kfac.update_factors("test", &act1, &grad1).unwrap();

        let f = kfac.factors.get("test").unwrap();
        let a_data = f.a_factor.data().unwrap().to_vec();
        assert!((a_data[0] - 0.375).abs() < 1e-12);
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
                a_factor: Tensor::from_storage(
                    TensorStorage::cpu(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
                    vec![3, 3],
                    false,
                )
                .unwrap(),
                a_size: 3,
                g_factor: Tensor::from_storage(
                    TensorStorage::cpu(vec![1.0, 0.0, 0.0, 1.0]),
                    vec![2, 2],
                    false,
                )
                .unwrap(),
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

    // -----------------------------------------------------------------------
    // CL-1105 Pattern B — CUDA device-resident step tests.
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

    /// KFAC step must preserve the device residence of its parameter
    /// (factors and inverses are uploaded to the parameter's device).
    #[cfg(feature = "cuda")]
    #[test]
    fn kfac_step_preserves_device_for_cuda_input() {
        if !try_init_cuda() {
            return;
        }
        let p = param_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
            .to(ferrotorch_core::Device::Cuda(0))
            .unwrap();
        let grad = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1_f64, 0.2, 0.3, 0.4]),
            vec![2, 2],
            false,
        )
        .unwrap()
        .cuda()
        .unwrap();
        p.tensor().set_grad(Some(grad)).unwrap();

        let mut kfac = Kfac::new(
            vec![p],
            KfacConfig {
                lr: 0.01,
                damping: 1e-3,
                momentum: 0.0,
                update_freq: 1,
                ..Default::default()
            },
        );

        // Register factors via CUDA activation/gradient — exercises the
        // device-resident outer-product accumulation path.
        let act = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f64, 0.0, 0.0, 1.0]),
            vec![2, 2],
            false,
        )
        .unwrap()
        .cuda()
        .unwrap();
        let outgrad = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f64, 0.0, 0.0, 1.0]),
            vec![2, 2],
            false,
        )
        .unwrap()
        .cuda()
        .unwrap();
        kfac.update_factors("g0_p0", &act, &outgrad).unwrap();

        kfac.step().unwrap();

        let after = &kfac.param_groups[0].params[0];
        assert!(
            after.tensor().is_cuda(),
            "KFAC::step must preserve CUDA residence; got device {:?}",
            after.tensor().device()
        );
        assert_eq!(after.tensor().device(), ferrotorch_core::Device::Cuda(0));
    }

    /// KFAC factor inversion uses cuSOLVER on CUDA. After the inversion
    /// step, the cached `a_inv`/`g_inv` tensors must live on the CUDA
    /// device — proving the inversion path went through the GPU solver
    /// rather than silently demoting to CPU LAPACK.
    #[cfg(feature = "cuda")]
    #[test]
    #[allow(non_snake_case)]
    fn kfac_factor_inversion_uses_cuSOLVER_on_cuda() {
        if !try_init_cuda() {
            return;
        }
        let p = param_from(&[1.0, 2.0, 3.0, 4.0], &[2, 2])
            .to(ferrotorch_core::Device::Cuda(0))
            .unwrap();
        let grad = Tensor::from_storage(
            TensorStorage::cpu(vec![0.1_f64, 0.2, 0.3, 0.4]),
            vec![2, 2],
            false,
        )
        .unwrap()
        .cuda()
        .unwrap();
        p.tensor().set_grad(Some(grad)).unwrap();

        let mut kfac = Kfac::new(
            vec![p],
            KfacConfig {
                lr: 0.01,
                damping: 1e-3,
                momentum: 0.0,
                update_freq: 1,
                ..Default::default()
            },
        );

        let act = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f64, 0.0, 0.0, 1.0]),
            vec![2, 2],
            false,
        )
        .unwrap()
        .cuda()
        .unwrap();
        let outgrad = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f64, 0.0, 0.0, 1.0]),
            vec![2, 2],
            false,
        )
        .unwrap()
        .cuda()
        .unwrap();
        kfac.update_factors("g0_p0", &act, &outgrad).unwrap();

        kfac.step().unwrap();

        // After the step, the cached factor inverses must be CUDA-resident.
        let factors = kfac.factors.get("g0_p0").expect("factors registered");
        let a_inv = factors.a_inv.as_ref().expect("a_inv computed");
        let g_inv = factors.g_inv.as_ref().expect("g_inv computed");
        assert!(
            a_inv.is_cuda(),
            "A^-1 must be CUDA-resident (cuSOLVER path); got {:?}",
            a_inv.device()
        );
        assert!(
            g_inv.is_cuda(),
            "G^-1 must be CUDA-resident (cuSOLVER path); got {:?}",
            g_inv.device()
        );
    }

    /// KFAC numerical equivalence: CPU vs CUDA run, same init/config,
    /// must converge to the same updated parameter vector.
    #[cfg(feature = "cuda")]
    #[test]
    fn kfac_step_matches_cpu_within_tolerance() {
        if !try_init_cuda() {
            return;
        }
        let init = vec![1.0_f64, 2.0, 3.0, 4.0];
        let grad_data = vec![0.1_f64, 0.2, 0.3, 0.4];
        let act_data = vec![1.0_f64, 0.0, 0.0, 1.0];
        let outg_data = vec![1.0_f64, 0.0, 0.0, 1.0];
        let cfg = KfacConfig {
            lr: 0.01,
            damping: 1e-3,
            momentum: 0.0,
            update_freq: 1,
            ..Default::default()
        };

        // CPU reference.
        let p_cpu = param_from(&init, &[2, 2]);
        p_cpu
            .tensor()
            .set_grad(Some(
                Tensor::from_storage(TensorStorage::cpu(grad_data.clone()), vec![2, 2], false)
                    .unwrap(),
            ))
            .unwrap();
        let mut kfac_cpu = Kfac::new(vec![p_cpu], cfg);
        let act_cpu =
            Tensor::from_storage(TensorStorage::cpu(act_data.clone()), vec![2, 2], false).unwrap();
        let outg_cpu =
            Tensor::from_storage(TensorStorage::cpu(outg_data.clone()), vec![2, 2], false).unwrap();
        kfac_cpu.update_factors("g0_p0", &act_cpu, &outg_cpu).unwrap();
        kfac_cpu.step().unwrap();
        let cpu_after: Vec<f64> = kfac_cpu.param_groups[0].params[0]
            .data()
            .unwrap()
            .to_vec();

        // CUDA run.
        let p_gpu = param_from(&init, &[2, 2])
            .to(ferrotorch_core::Device::Cuda(0))
            .unwrap();
        let g_gpu =
            Tensor::from_storage(TensorStorage::cpu(grad_data), vec![2, 2], false)
                .unwrap()
                .cuda()
                .unwrap();
        p_gpu.tensor().set_grad(Some(g_gpu)).unwrap();
        let mut kfac_gpu = Kfac::new(vec![p_gpu], cfg);
        let act_gpu = Tensor::from_storage(TensorStorage::cpu(act_data), vec![2, 2], false)
            .unwrap()
            .cuda()
            .unwrap();
        let outg_gpu = Tensor::from_storage(TensorStorage::cpu(outg_data), vec![2, 2], false)
            .unwrap()
            .cuda()
            .unwrap();
        kfac_gpu.update_factors("g0_p0", &act_gpu, &outg_gpu).unwrap();
        kfac_gpu.step().unwrap();
        let gpu_after: Vec<f64> = kfac_gpu.param_groups[0].params[0]
            .tensor()
            .cpu()
            .unwrap()
            .data()
            .unwrap()
            .to_vec();

        assert_eq!(cpu_after.len(), gpu_after.len());
        for (i, (c, g)) in cpu_after.iter().zip(gpu_after.iter()).enumerate() {
            assert!(
                (c - g).abs() < 1e-6,
                "KFAC CPU/GPU mismatch at idx {i}: cpu={c}, gpu={g}"
            );
        }
    }
}
