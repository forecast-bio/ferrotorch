//! Adam optimizer (Adaptive Moment Estimation).
//!
//! Implements the algorithm from Kingma & Ba, "Adam: A Method for Stochastic
//! Optimization" (ICLR 2015), including the AMSGrad variant from Reddi et al.
//!
//! All parameter updates execute inside `no_grad()` so the optimizer step is
//! never recorded in the autograd graph.

use std::any::TypeId;
use std::collections::HashMap;
use std::collections::hash_map::Entry;

use ferrotorch_core::gpu_dispatch::{GpuBufferHandle, gpu_backend};
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, no_grad};
use ferrotorch_nn::Parameter;

use crate::foreach_utils::elemwise_max;
use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};

// ---------------------------------------------------------------------------
// AdamConfig
// ---------------------------------------------------------------------------

/// Hyperparameters for the Adam optimizer.
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct AdamConfig {
    /// Learning rate (default: 0.001).
    pub lr: f64,
    /// Exponential decay rates for the first and second moment estimates
    /// (default: (0.9, 0.999)).
    pub betas: (f64, f64),
    /// Term added to the denominator for numerical stability (default: 1e-8).
    pub eps: f64,
    /// Weight decay coefficient for L2 regularization (default: 0.0).
    pub weight_decay: f64,
    /// Whether to use the AMSGrad variant (default: false).
    pub amsgrad: bool,
    /// When `true`, maximize the objective by negating the gradient (default:
    /// false). CL-321
    pub maximize: bool,
    /// When `true`, use the on-device tensor-op update path. Moments live
    /// on the parameter's device and are updated via GPU-aware tensor ops,
    /// avoiding per-step CPU round-trips. Computes in `T` precision instead
    /// of the legacy `f64` accumulation — a documented accuracy tradeoff
    /// users opt into. Default: false. CL-497
    pub foreach: bool,
}

impl Default for AdamConfig {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.0,
            amsgrad: false,
            maximize: false,
            foreach: false,
        }
    }
}

impl AdamConfig {
    /// Set the learning rate.
    #[must_use]
    pub fn with_lr(mut self, lr: f64) -> Self {
        self.lr = lr;
        self
    }

    /// Set the exponential decay rates for the first and second moment estimates.
    #[must_use]
    pub fn with_betas(mut self, betas: (f64, f64)) -> Self {
        self.betas = betas;
        self
    }

    /// Set the term added to the denominator for numerical stability.
    #[must_use]
    pub fn with_eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the weight decay coefficient for L2 regularization.
    #[must_use]
    pub fn with_weight_decay(mut self, weight_decay: f64) -> Self {
        self.weight_decay = weight_decay;
        self
    }

    /// Enable or disable the AMSGrad variant.
    #[must_use]
    pub fn with_amsgrad(mut self, amsgrad: bool) -> Self {
        self.amsgrad = amsgrad;
        self
    }

    /// Set the maximize flag (when `true`, negate the gradient to maximize).
    #[must_use]
    pub fn with_maximize(mut self, maximize: bool) -> Self {
        self.maximize = maximize;
        self
    }

    /// Enable or disable the on-device tensor-op (foreach) update path.
    #[must_use]
    pub fn with_foreach(mut self, foreach: bool) -> Self {
        self.foreach = foreach;
        self
    }
}

// ---------------------------------------------------------------------------
// Per-parameter state
// ---------------------------------------------------------------------------

/// Mutable state tracked for each parameter across steps.
///
/// For GPU f32 parameters, moments are stored as `GpuBufferHandle`s on-device
/// and the fused Adam kernel updates param + moments in a single launch. For
/// CPU parameters (or f64), moments are `Vec<f64>` with the original scalar
/// loop.
#[derive(Debug)]
struct AdamParamState {
    /// Number of optimizer steps taken for this parameter.
    step_count: u64,
    /// CPU first moment estimate (exponential moving average of gradients).
    exp_avg: Vec<f64>,
    /// CPU second moment estimate (exponential moving average of squared gradients).
    exp_avg_sq: Vec<f64>,
    /// Maximum of bias-corrected second moment estimates (AMSGrad only).
    max_exp_avg_sq: Option<Vec<f64>>,
    /// GPU first moment (present when param is on CUDA and T is f32).
    gpu_exp_avg: Option<GpuBufferHandle>,
    /// GPU second moment (present when param is on CUDA and T is f32).
    gpu_exp_avg_sq: Option<GpuBufferHandle>,
}

/// On-device state used by the foreach (tensor-op) update path. Moments
/// live on the parameter's device and are updated via GPU-aware tensor ops.
#[derive(Debug)]
struct AdamForeachState<T: Float> {
    step_count: u64,
    exp_avg: Tensor<T>,
    exp_avg_sq: Tensor<T>,
    max_exp_avg_sq: Option<Tensor<T>>,
}

// ---------------------------------------------------------------------------
// Adam
// ---------------------------------------------------------------------------

/// The Adam optimizer.
///
/// Maintains per-parameter exponential moving averages of the gradient (first
/// moment) and the squared gradient (second moment). With `amsgrad = true`, it
/// additionally tracks the running maximum of the bias-corrected second moment.
#[derive(Debug)]
pub struct Adam<T: Float> {
    param_groups: Vec<ParamGroup<T>>,
    config: AdamConfig,
    /// Per-parameter state (legacy CPU path), keyed by
    /// `"g{group_idx}_p{param_idx}"`. Used when `config.foreach == false`.
    state: HashMap<String, AdamParamState>,
    /// Foreach (on-device) per-parameter state. Used when
    /// `config.foreach == true`.
    foreach_state: HashMap<String, AdamForeachState<T>>,
}

impl<T: Float> Adam<T> {
    /// Create a new Adam optimizer for the given parameters.
    pub fn new(params: Vec<Parameter<T>>, config: AdamConfig) -> Self {
        let mut group = ParamGroup::new(params, config.lr);
        group.weight_decay = config.weight_decay;

        Self {
            param_groups: vec![group],
            config,
            state: HashMap::new(),
            foreach_state: HashMap::new(),
        }
    }

    /// Generate the state key for a parameter.
    #[inline]
    fn param_key(group_idx: usize, param_idx: usize) -> String {
        format!("g{group_idx}_p{param_idx}")
    }

    /// Foreach (on-device, tensor-op) update path used when
    /// `config.foreach == true`. Mirrors the legacy CPU path numerically
    /// within `T` precision.
    fn step_foreach(&mut self) -> FerrotorchResult<()> {
        use ferrotorch_core::creation::{scalar, zeros};
        use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, neg, sqrt, sub};

        let config = self.config;
        let (beta1, beta2) = config.betas;

        for gi in 0..self.param_groups.len() {
            let group_lr = self.param_groups[gi].lr;
            let group_wd = self.param_groups[gi].weight_decay;

            for pi in 0..self.param_groups[gi].params.len() {
                let param = &self.param_groups[gi].params[pi];
                let param_t = param.tensor();

                let grad_tensor = match param.grad()? {
                    Some(g) => g,
                    None => continue,
                };

                let device = param_t.device();
                let key = Self::param_key(gi, pi);

                no_grad(|| {
                    // grad (possibly negated for maximize, possibly L2-augmented
                    // with wd*param — classic Adam uses coupled weight decay).
                    let mut grad: Tensor<T> = if config.maximize {
                        neg(&grad_tensor)?
                    } else {
                        grad_tensor.clone()
                    };
                    if group_wd > 0.0 {
                        let wd_t = scalar(cast::<f64, T>(group_wd)?)?.to(device)?;
                        let weighted = mul(param_t, &wd_t)?;
                        grad = add(&grad, &weighted)?;
                    }

                    // Initialize state on first step. Use `Entry::Vacant` +
                    // explicit `?` propagation rather than `or_insert_with`
                    // (whose closure cannot return `Result`); a fallible
                    // zeros / .to(device) here would otherwise have to
                    // panic.
                    let next_step = match self.foreach_state.entry(key.clone()) {
                        Entry::Vacant(slot) => {
                            let exp_avg = zeros::<T>(param_t.shape())?.to(device)?;
                            let exp_avg_sq = zeros::<T>(param_t.shape())?.to(device)?;
                            let max_exp_avg_sq = if config.amsgrad {
                                Some(zeros::<T>(param_t.shape())?.to(device)?)
                            } else {
                                None
                            };
                            slot.insert(AdamForeachState {
                                step_count: 0,
                                exp_avg,
                                exp_avg_sq,
                                max_exp_avg_sq,
                            });
                            1
                        }
                        Entry::Occupied(slot) => slot.get().step_count + 1,
                    };

                    // Read current moment tensors.
                    let exp_avg_old = self.foreach_state[&key].exp_avg.clone();
                    let exp_avg_sq_old = self.foreach_state[&key].exp_avg_sq.clone();

                    // exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                    let beta1_t = scalar(cast::<f64, T>(beta1)?)?.to(device)?;
                    let one_minus_beta1 = scalar(cast::<f64, T>(1.0 - beta1)?)?.to(device)?;
                    let exp_avg_new = add(
                        &mul(&exp_avg_old, &beta1_t)?,
                        &mul(&grad, &one_minus_beta1)?,
                    )?;

                    // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
                    let beta2_t = scalar(cast::<f64, T>(beta2)?)?.to(device)?;
                    let one_minus_beta2 = scalar(cast::<f64, T>(1.0 - beta2)?)?.to(device)?;
                    let grad_sq = mul(&grad, &grad)?;
                    let exp_avg_sq_new = add(
                        &mul(&exp_avg_sq_old, &beta2_t)?,
                        &mul(&grad_sq, &one_minus_beta2)?,
                    )?;

                    // Bias-correction factors.
                    let bc1 = 1.0 - beta1.powi(next_step as i32);
                    let bc2 = 1.0 - beta2.powi(next_step as i32);
                    let inv_bc1 = scalar(cast::<f64, T>(1.0 / bc1)?)?.to(device)?;
                    let inv_bc2 = scalar(cast::<f64, T>(1.0 / bc2)?)?.to(device)?;

                    let m_hat = mul(&exp_avg_new, &inv_bc1)?;

                    // AMSGrad: take elementwise max of bias-corrected v.
                    let (v_hat, new_max) = if config.amsgrad {
                        let old_max = self.foreach_state[&key]
                            .max_exp_avg_sq
                            .as_ref()
                            .ok_or_else(|| FerrotorchError::Internal {
                                message:
                                    "adam foreach: amsgrad enabled but max_exp_avg_sq slot empty"
                                        .to_string(),
                            })?
                            .clone();
                        let corrected_v = mul(&exp_avg_sq_new, &inv_bc2)?;
                        let new_max = elemwise_max(&old_max, &corrected_v, device)?;
                        (new_max.clone(), Some(new_max))
                    } else {
                        (mul(&exp_avg_sq_new, &inv_bc2)?, None)
                    };

                    // sqrt(v_hat) + eps
                    let sqrt_v = sqrt(&v_hat)?;
                    let eps_t = scalar(cast::<f64, T>(config.eps)?)?.to(device)?;
                    let denom = add(&sqrt_v, &eps_t)?;
                    let update = div(&m_hat, &denom)?;

                    // param = param - lr * update
                    let lr_t = scalar(cast::<f64, T>(group_lr)?)?.to(device)?;
                    let scaled_update = mul(&update, &lr_t)?;
                    let new_param = sub(param_t, &scaled_update)?;

                    // Commit parameter update.
                    let (storage, _) = new_param.into_storage_and_shape()?;
                    // SAFETY: `update_storage` requires exclusive access to
                    // the parameter tensor's `Arc<TensorStorage<T>>`. We
                    // satisfy that here:
                    //   1. `step_foreach` borrows `&mut self` on the
                    //      optimizer for the entire iteration, so no other
                    //      method on this optimizer instance can run in
                    //      parallel.
                    //   2. We are inside `no_grad()`, so no autograd graph
                    //      is being recorded — no `grad_fn` clone of this
                    //      storage is being constructed concurrently.
                    //   3. All reads of the parameter tensor's data
                    //      (`exp_avg_old`, `exp_avg_sq_old`, `m_hat`,
                    //      `denom`, `update`, `new_param`) computed earlier
                    //      in this iteration produced *new* tensors via
                    //      tensor ops; once `new_param.into_storage_and_shape()`
                    //      consumed `new_param`, no live `&` to this
                    //      tensor's storage exists.
                    //   4. The new storage shares this tensor's element
                    //      count and device (it was produced by tensor ops
                    //      on `param_t`), so the `update_storage` length
                    //      check passes.
                    unsafe { param_t.update_storage(storage)? };

                    // Commit state after the parameter update succeeded.
                    // The slot was inserted (or pre-existed) above by the
                    // `Entry::Vacant`/`Occupied` match — out-of-scope of
                    // this re-read can only happen if state mutation
                    // races with the optimizer step, which `&mut self`
                    // forbids.
                    let st = self.foreach_state.get_mut(&key).ok_or_else(|| {
                        FerrotorchError::Internal {
                            message: format!(
                                "adam foreach: state slot for {key} disappeared mid-step"
                            ),
                        }
                    })?;
                    st.step_count = next_step;
                    st.exp_avg = exp_avg_new;
                    st.exp_avg_sq = exp_avg_sq_new;
                    if let Some(m) = new_max {
                        st.max_exp_avg_sq = Some(m);
                    }

                    Ok::<(), ferrotorch_core::FerrotorchError>(())
                })?;
            }
        }

        Ok(())
    }
}

impl<T: Float> Optimizer<T> for Adam<T> {
    fn step(&mut self) -> FerrotorchResult<()> {
        // Foreach path: stay on-device throughout, no CPU roundtrip.
        if self.config.foreach {
            return self.step_foreach();
        }

        let config = self.config;
        let (beta1, beta2) = config.betas;
        let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();

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

                // ---- GPU fast-path: fused Adam kernel (f32, non-AMSGrad) ----
                let use_gpu = is_f32
                    && tensor.is_cuda()
                    && grad_tensor.is_cuda()
                    && !config.amsgrad
                    && !config.maximize;

                if use_gpu {
                    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                    let ordinal = match tensor.device() {
                        ferrotorch_core::Device::Cuda(o) => o,
                        _ => 0,
                    };

                    // Lazy-init GPU state. `or_insert_with` cannot propagate
                    // `Err` from a fallible allocation; use an explicit
                    // Vacant/Occupied match so an OOM at first step
                    // bubbles out instead of being swallowed into `None` and
                    // silently routing to the CPU path.
                    //
                    // First: ensure the slot exists (and is GPU-populated)
                    // OR decide what to do when alloc fails. Doing this in
                    // a separate pre-step keeps borrow scopes simple — the
                    // compute branch later borrows `state` mutably without
                    // overlapping the alloc-decision borrow.
                    if !self.state.contains_key(&key) {
                        let gpu_m = backend.alloc_zeros(numel, std::mem::size_of::<f32>(), ordinal);
                        let gpu_v = backend.alloc_zeros(numel, std::mem::size_of::<f32>(), ordinal);
                        match (gpu_m, gpu_v) {
                            (Ok(m), Ok(v)) => {
                                self.state.insert(
                                    key.clone(),
                                    AdamParamState {
                                        step_count: 0,
                                        exp_avg: Vec::new(),
                                        exp_avg_sq: Vec::new(),
                                        max_exp_avg_sq: None,
                                        gpu_exp_avg: Some(m),
                                        gpu_exp_avg_sq: Some(v),
                                    },
                                );
                            }
                            (Err(e), _) | (_, Err(e)) => {
                                // Match PyTorch parity: GPU op failure is
                                // `Err` by default. Opt-in CPU fallback
                                // only when the user has explicitly set
                                // `FERROTORCH_ENABLE_GPU_FALLBACK`.
                                // Default refuses silent backend
                                // degradation.
                                if std::env::var("FERROTORCH_ENABLE_GPU_FALLBACK").is_ok() {
                                    tracing::warn!(
                                        target: "ferrotorch::gpu_fallback",
                                        kernel = "fused_adam_f32",
                                        error = %e,
                                        "Adam GPU state allocation failed; falling back to \
                                         CPU. Unset FERROTORCH_ENABLE_GPU_FALLBACK to make \
                                         this an error instead.",
                                    );
                                    // Re-acquire `key` for the CPU path
                                    // below to use, and skip the GPU
                                    // compute branch.
                                    // (Fall-through into the CPU branch.)
                                } else {
                                    return Err(FerrotorchError::Internal {
                                        message: format!(
                                            "fused_adam_f32: GPU state alloc failed: {e:?} \
                                             (set FERROTORCH_ENABLE_GPU_FALLBACK=1 to opt \
                                             into a CPU fallback)"
                                        ),
                                    });
                                }
                            }
                        }
                    }

                    if let Some(state) = self.state.get_mut(&key) {
                        state.step_count += 1;
                        let step = state.step_count;
                        let bc1 = 1.0 - (beta1 as f32).powi(step as i32);
                        let bc2 = 1.0 - (beta2 as f32).powi(step as i32);

                        // Both moments are guaranteed `Some` by the
                        // Vacant-branch insertion above. If somehow they
                        // are `None` (e.g. a future bug or
                        // load_state_dict dropping them), surface that as
                        // `Internal` rather than silently downgrading.
                        let gpu_m = state.gpu_exp_avg.as_mut().ok_or_else(|| {
                            FerrotorchError::Internal {
                                message: "fused_adam_f32: gpu_exp_avg missing post-init"
                                    .to_string(),
                            }
                        })?;
                        let gpu_v = state.gpu_exp_avg_sq.as_mut().ok_or_else(|| {
                            FerrotorchError::Internal {
                                message: "fused_adam_f32: gpu_exp_avg_sq missing post-init"
                                    .to_string(),
                            }
                        })?;

                        no_grad(|| {
                            // No `unsafe` here: `with_gpu_handle_mut` is
                            // the safe wrapper that centralizes the
                            // `Arc::as_ptr -> *mut TensorStorage<T>`
                            // pointer cast inside `ferrotorch-core`.
                            tensor.with_gpu_handle_mut(|param_handle| {
                                backend.fused_adam_f32(
                                    param_handle,
                                    grad_tensor.gpu_handle()?,
                                    gpu_m,
                                    gpu_v,
                                    beta1 as f32,
                                    beta2 as f32,
                                    group_lr as f32,
                                    config.eps as f32,
                                    bc1,
                                    bc2,
                                    group_wd as f32,
                                )
                            })
                        })?;
                        continue;
                    }
                    // GPU state alloc failed and FERROTORCH_ENABLE_GPU_FALLBACK
                    // was set: fall through to the CPU path below.
                }

                // ---- CPU path (or f64 / AMSGrad / maximize) ----

                let param_data: Vec<f64> = tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;
                let mut grad_data: Vec<f64> = grad_tensor
                    .data_vec()?
                    .iter()
                    .map(|&v| cast::<T, f64>(v))
                    .collect::<FerrotorchResult<Vec<f64>>>()?;

                // Maximize: negate gradient. CL-321
                if config.maximize {
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

                // Lazy-init state.
                let state = self.state.entry(key).or_insert_with(|| AdamParamState {
                    step_count: 0,
                    exp_avg: vec![0.0; numel],
                    exp_avg_sq: vec![0.0; numel],
                    max_exp_avg_sq: if config.amsgrad {
                        Some(vec![0.0; numel])
                    } else {
                        None
                    },
                    gpu_exp_avg: None,
                    gpu_exp_avg_sq: None,
                });

                state.step_count += 1;
                let step = state.step_count;

                // Update first moment: exp_avg = beta1 * exp_avg + (1 - beta1) * grad.
                for (m, &g) in state.exp_avg.iter_mut().zip(grad_data.iter()) {
                    *m = beta1 * *m + (1.0 - beta1) * g;
                }

                // Update second moment: exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2.
                for (v, &g) in state.exp_avg_sq.iter_mut().zip(grad_data.iter()) {
                    *v = beta2 * *v + (1.0 - beta2) * g * g;
                }

                // Bias correction.
                let bc1 = 1.0 - beta1.powi(step as i32);
                let bc2 = 1.0 - beta2.powi(step as i32);

                // Compute updated parameter values.
                if config.amsgrad {
                    let max_sq = state.max_exp_avg_sq.as_mut().unwrap();
                    for (ms, &ea) in max_sq.iter_mut().zip(state.exp_avg_sq.iter()) {
                        if ea > *ms {
                            *ms = ea;
                        }
                    }
                }

                let new_values: Vec<T> = (0..numel)
                    .map(|i| {
                        let corrected_avg = state.exp_avg[i] / bc1;
                        let denom = if config.amsgrad {
                            let max_sq = state.max_exp_avg_sq.as_ref().unwrap();
                            (max_sq[i] / bc2).sqrt() + config.eps
                        } else {
                            let corrected_sq = state.exp_avg_sq[i] / bc2;
                            corrected_sq.sqrt() + config.eps
                        };
                        let updated = param_data[i] - group_lr * corrected_avg / denom;
                        cast::<f64, T>(updated)
                    })
                    .collect::<FerrotorchResult<Vec<T>>>()?;

                no_grad(|| {
                    // SAFETY: `update_data` requires exclusive access to the
                    // parameter tensor's `Arc<TensorStorage<T>>` for the
                    // duration of the call. We satisfy that here:
                    //   1. `Adam::step` borrows `&mut self` on the
                    //      optimizer for the entire step, so no other
                    //      method can run on this optimizer instance.
                    //   2. We are inside `no_grad()` — no autograd graph
                    //      is being constructed, so no `grad_fn` is taking
                    //      a parallel `Arc` clone of this storage.
                    //   3. All reads of the parameter's data
                    //      (`param_data` above) and gradient (`grad_data`)
                    //      have already completed and were copied into
                    //      owned `Vec<f64>` workspaces — there is no live
                    //      `&[T]` into the storage at the call site.
                    //   4. `new_values.len() == numel == tensor.numel()` so
                    //      the inner length check passes.
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

    fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
        let mut out = OptimizerState::new();
        for (key, pstate) in &self.state {
            let mut entry = HashMap::new();
            entry.insert("step_count".to_string(), vec![pstate.step_count as f64]);

            // For GPU state, download moments to CPU f64 for serialization.
            if let (Some(gpu_m), Some(gpu_v)) = (&pstate.gpu_exp_avg, &pstate.gpu_exp_avg_sq) {
                if let Some(backend) = gpu_backend() {
                    if let (Ok(m_bytes), Ok(v_bytes)) =
                        (backend.gpu_to_cpu(gpu_m), backend.gpu_to_cpu(gpu_v))
                    {
                        // Decode bytes -> f32 -> f64 without `unsafe`.
                        // The raw `slice::from_raw_parts` reinterpretation
                        // we used to do here had two distinct hazards
                        // (alignment of the byte buffer and partial-length
                        // truncation) that `f32::from_ne_bytes` over fixed
                        // 4-byte windows sidesteps entirely; this also
                        // matches PyTorch's behavior of native-endian f32
                        // serialization.
                        let m_f64: Vec<f64> = m_bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f64)
                            .collect();
                        let v_f64: Vec<f64> = v_bytes
                            .chunks_exact(4)
                            .map(|c| f32::from_ne_bytes([c[0], c[1], c[2], c[3]]) as f64)
                            .collect();
                        entry.insert("exp_avg".to_string(), m_f64);
                        entry.insert("exp_avg_sq".to_string(), v_f64);
                        out.insert(key.clone(), entry);
                        continue;
                    }
                }
            }

            entry.insert("exp_avg".to_string(), pstate.exp_avg.clone());
            entry.insert("exp_avg_sq".to_string(), pstate.exp_avg_sq.clone());
            if let Some(max_sq) = &pstate.max_exp_avg_sq {
                entry.insert("max_exp_avg_sq".to_string(), max_sq.clone());
            }
            out.insert(key.clone(), entry);
        }
        Ok(out)
    }

    fn load_state_dict(&mut self, state: &OptimizerState) -> FerrotorchResult<()> {
        for (key, entry) in state {
            let step_count = entry
                .get("step_count")
                .and_then(|v| v.first())
                .copied()
                .unwrap_or(0.0) as u64;

            let exp_avg =
                entry
                    .get("exp_avg")
                    .cloned()
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!("missing exp_avg in state for key {key}"),
                    })?;

            let exp_avg_sq = entry.get("exp_avg_sq").cloned().ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: format!("missing exp_avg_sq in state for key {key}"),
                }
            })?;

            let max_exp_avg_sq = entry.get("max_exp_avg_sq").cloned();

            self.state.insert(
                key.clone(),
                AdamParamState {
                    step_count,
                    exp_avg,
                    exp_avg_sq,
                    max_exp_avg_sq,
                    gpu_exp_avg: None,
                    gpu_exp_avg_sq: None,
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
    use ferrotorch_core::grad_fns::arithmetic::{add, mul, pow, sub};
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Create a scalar parameter from a single f64 value.
    fn scalar_param(val: f64) -> Parameter<f64> {
        let t = Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], true).unwrap();
        Parameter::new(t)
    }

    /// Read a scalar parameter's current value.
    fn param_val(opt: &Adam<f64>, group: usize, idx: usize) -> f64 {
        opt.param_groups[group].params[idx].tensor().data().unwrap()[0]
    }

    // -----------------------------------------------------------------------
    // Rosenbrock convergence test
    // -----------------------------------------------------------------------

    /// Rosenbrock function: f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
    /// Global minimum at (1, 1) with f(1,1) = 0.
    #[test]
    fn test_adam_rosenbrock_convergence() {
        let px = scalar_param(-1.0);
        let py = scalar_param(1.0);

        let mut opt = Adam::new(
            vec![px, py],
            AdamConfig {
                lr: 0.01,
                ..Default::default()
            },
        );

        for _ in 0..5000 {
            opt.zero_grad().unwrap();

            // Read current parameter tensors from the optimizer (they get
            // replaced on each step).
            let x_tensor = opt.param_groups[0].params[0].tensor().clone();
            let y_tensor = opt.param_groups[0].params[1].tensor().clone();

            // f(x, y) = (1 - x)^2 + 100*(y - x^2)^2
            let one =
                Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
            let hundred =
                Tensor::from_storage(TensorStorage::cpu(vec![100.0_f64]), vec![], false).unwrap();

            // term1 = (1 - x)^2
            let diff1 = sub(&one, &x_tensor).unwrap();
            let term1 = pow(&diff1, 2.0).unwrap();

            // term2 = 100 * (y - x^2)^2
            let x_sq = pow(&x_tensor, 2.0).unwrap();
            let diff2 = sub(&y_tensor, &x_sq).unwrap();
            let diff2_sq = pow(&diff2, 2.0).unwrap();
            let term2 = mul(&hundred, &diff2_sq).unwrap();

            let loss = add(&term1, &term2).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        let final_x = param_val(&opt, 0, 0);
        let final_y = param_val(&opt, 0, 1);

        assert!(
            (final_x - 1.0).abs() < 0.05,
            "expected x near 1.0, got {final_x}"
        );
        assert!(
            (final_y - 1.0).abs() < 0.05,
            "expected y near 1.0, got {final_y}"
        );
    }

    // -----------------------------------------------------------------------
    // zero_grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_zero_grad() {
        let p = scalar_param(3.0);
        let mut opt = Adam::new(vec![p], AdamConfig::default());

        // Manually set a gradient.
        let grad = Tensor::from_storage(TensorStorage::cpu(vec![1.0_f64]), vec![], false).unwrap();
        opt.param_groups[0].params[0]
            .tensor()
            .set_grad(Some(grad))
            .unwrap();

        // Verify gradient exists.
        assert!(
            opt.param_groups[0].params[0]
                .tensor()
                .grad()
                .unwrap()
                .is_some()
        );

        // Zero it out.
        opt.zero_grad().unwrap();

        assert!(
            opt.param_groups[0].params[0]
                .tensor()
                .grad()
                .unwrap()
                .is_none()
        );
    }

    // -----------------------------------------------------------------------
    // state_dict / load_state_dict
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_state_dict_roundtrip() {
        let p = scalar_param(2.0);
        let mut opt = Adam::new(vec![p], AdamConfig::default());

        // Run a few steps so the state is populated.
        for _ in 0..3 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        // Save state.
        let saved = opt
            .state_dict()
            .expect("adam state_dict must succeed in test");
        assert!(
            !saved.is_empty(),
            "state dict should be non-empty after steps"
        );

        // Verify the state contains expected keys.
        let key = Adam::<f64>::param_key(0, 0);
        assert!(saved.contains_key(&key), "expected key {key} in state dict");

        let entry = &saved[&key];
        assert!(entry.contains_key("step_count"));
        assert!(entry.contains_key("exp_avg"));
        assert!(entry.contains_key("exp_avg_sq"));

        let step_count = entry["step_count"][0] as u64;
        assert_eq!(step_count, 3);

        // Load state into a fresh optimizer.
        let p2 = scalar_param(2.0);
        let mut opt2 = Adam::new(vec![p2], AdamConfig::default());
        opt2.load_state_dict(&saved).unwrap();

        let loaded = opt2
            .state_dict()
            .expect("adam state_dict round-trip must succeed in test");
        assert_eq!(loaded[&key]["step_count"], saved[&key]["step_count"]);
        assert_eq!(loaded[&key]["exp_avg"], saved[&key]["exp_avg"]);
        assert_eq!(loaded[&key]["exp_avg_sq"], saved[&key]["exp_avg_sq"]);
    }

    // -----------------------------------------------------------------------
    // AMSGrad variant
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_amsgrad() {
        let p = scalar_param(5.0);
        let mut opt = Adam::new(
            vec![p],
            AdamConfig {
                amsgrad: true,
                ..Default::default()
            },
        );

        // A few steps to exercise the amsgrad path.
        for _ in 0..10 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        // Verify state contains max_exp_avg_sq.
        let saved = opt
            .state_dict()
            .expect("adam amsgrad state_dict must succeed in test");
        let key = Adam::<f64>::param_key(0, 0);
        assert!(saved[&key].contains_key("max_exp_avg_sq"));

        // The parameter should have moved towards zero (minimizing x^2).
        let val = param_val(&opt, 0, 0);
        assert!(
            val.abs() < 5.0,
            "parameter should have decreased from 5.0, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Weight decay
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_weight_decay() {
        let p = scalar_param(5.0);
        let config = AdamConfig {
            lr: 0.1,
            weight_decay: 0.1,
            ..Default::default()
        };
        let mut opt = Adam::new(vec![p], config);

        for _ in 0..50 {
            let tensor = opt.param_groups[0].params[0].tensor().clone();
            let loss = pow(&tensor, 2.0).unwrap();
            loss.backward().unwrap();
            opt.step().unwrap();
            opt.zero_grad().unwrap();
        }

        // With weight decay, the parameter should still converge towards zero.
        let val = param_val(&opt, 0, 0);
        assert!(
            val.abs() < 1.0,
            "parameter should have moved towards 0 with weight decay, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Learning rate accessors
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_lr_accessors() {
        let p = scalar_param(1.0);
        let mut opt = Adam::new(
            vec![p],
            AdamConfig {
                lr: 0.05,
                ..Default::default()
            },
        );

        assert!((opt.lr() - 0.05).abs() < 1e-12);

        opt.set_lr(0.01);
        assert!((opt.lr() - 0.01).abs() < 1e-12);
        assert!((opt.param_groups()[0].lr - 0.01).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Multiple parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_adam_multiple_params() {
        let p1 = scalar_param(3.0);
        let p2 = scalar_param(-2.0);

        let mut opt = Adam::new(
            vec![p1, p2],
            AdamConfig {
                lr: 0.01,
                ..Default::default()
            },
        );

        for _ in 0..1000 {
            // Minimize f(a, b) = a^2 + b^2.
            opt.zero_grad().unwrap();
            let a = opt.param_groups[0].params[0].tensor().clone();
            let b = opt.param_groups[0].params[1].tensor().clone();

            let a_sq = pow(&a, 2.0).unwrap();
            let b_sq = pow(&b, 2.0).unwrap();
            let loss = add(&a_sq, &b_sq).unwrap();
            loss.backward().unwrap();

            opt.step().unwrap();
        }

        let v1 = param_val(&opt, 0, 0);
        let v2 = param_val(&opt, 0, 1);
        assert!(v1.abs() < 0.5, "expected p1 near 0, got {v1}");
        assert!(v2.abs() < 0.5, "expected p2 near 0, got {v2}");
    }

    // -----------------------------------------------------------------------
    // Foreach mode parity tests. CL-497
    // -----------------------------------------------------------------------

    /// Create a pair of parameters with identical initial values so the
    /// legacy and foreach paths can be compared side by side.
    fn paired_params(data: &[f32]) -> (Parameter<f32>, Parameter<f32>) {
        (
            Parameter::from_slice(data, &[data.len()]).unwrap(),
            Parameter::from_slice(data, &[data.len()]).unwrap(),
        )
    }

    fn leaf_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_adam_foreach_basic_parity() {
        let (p_legacy, p_foreach) = paired_params(&[1.0, 2.0, 3.0, 4.0]);
        let mut legacy = Adam::new(vec![p_legacy.clone()], AdamConfig::default());
        let mut foreach = Adam::new(
            vec![p_foreach.clone()],
            AdamConfig {
                foreach: true,
                ..Default::default()
            },
        );

        for _ in 0..8 {
            let g = leaf_f32(&[0.1, 0.2, -0.3, 0.4], &[4]);
            p_legacy.set_grad(Some(g.clone())).unwrap();
            p_foreach.set_grad(Some(g)).unwrap();
            legacy.step().unwrap();
            foreach.step().unwrap();
        }

        let l = legacy.param_groups()[0].params[0].data().unwrap().to_vec();
        let f = foreach.param_groups()[0].params[0].data().unwrap().to_vec();
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "adam foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_adam_foreach_parity_with_weight_decay() {
        let (p_legacy, p_foreach) = paired_params(&[5.0, -3.0, 2.0]);
        let cfg = AdamConfig {
            lr: 0.01,
            weight_decay: 0.05,
            ..Default::default()
        };
        let mut legacy = Adam::new(vec![p_legacy.clone()], cfg);
        let mut foreach = Adam::new(
            vec![p_foreach.clone()],
            AdamConfig {
                foreach: true,
                ..cfg
            },
        );

        for _ in 0..5 {
            let g = leaf_f32(&[0.5, -0.5, 1.0], &[3]);
            p_legacy.set_grad(Some(g.clone())).unwrap();
            p_foreach.set_grad(Some(g)).unwrap();
            legacy.step().unwrap();
            foreach.step().unwrap();
        }

        let l = legacy.param_groups()[0].params[0].data().unwrap().to_vec();
        let f = foreach.param_groups()[0].params[0].data().unwrap().to_vec();
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "adam weight decay parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_adam_foreach_parity_with_amsgrad() {
        let (p_legacy, p_foreach) = paired_params(&[2.0, -1.0, 0.5, 0.0]);
        let cfg = AdamConfig {
            amsgrad: true,
            ..Default::default()
        };
        let mut legacy = Adam::new(vec![p_legacy.clone()], cfg);
        let mut foreach = Adam::new(
            vec![p_foreach.clone()],
            AdamConfig {
                foreach: true,
                ..cfg
            },
        );

        for _ in 0..6 {
            let g = leaf_f32(&[0.2, 0.1, -0.05, 0.3], &[4]);
            p_legacy.set_grad(Some(g.clone())).unwrap();
            p_foreach.set_grad(Some(g)).unwrap();
            legacy.step().unwrap();
            foreach.step().unwrap();
        }

        let l = legacy.param_groups()[0].params[0].data().unwrap().to_vec();
        let f = foreach.param_groups()[0].params[0].data().unwrap().to_vec();
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-3,
                "adam amsgrad parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_adam_foreach_skips_params_without_grad() {
        let p1 = Parameter::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let p2 = Parameter::from_slice(&[3.0f32, 4.0], &[2]).unwrap();
        let g = leaf_f32(&[1.0, 1.0], &[2]);
        p1.set_grad(Some(g)).unwrap();

        let mut opt = Adam::new(
            vec![p1, p2],
            AdamConfig {
                foreach: true,
                ..Default::default()
            },
        );
        opt.step().unwrap();

        let p1_data = opt.param_groups()[0].params[0].data().unwrap();
        let p2_data = opt.param_groups()[0].params[1].data().unwrap();
        // p1 updated by Adam; p2 unchanged.
        assert!(p1_data[0] < 1.0);
        assert_eq!(p2_data, &[3.0, 4.0]);
    }
}
