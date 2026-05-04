//! Stochastic Weight Averaging (SWA) utilities.
//!
//! Provides [`AveragedModel`] for maintaining running averages of model
//! parameters and [`SWALR`] for the SWA-specific learning rate schedule.
//!
//! # SWA overview
//!
//! SWA maintains a running equal-weight average of model checkpoints taken
//! during training. At step `n`, the averaged parameters are:
//!
//! ```text
//! avg = avg + (param - avg) / (n + 1)
//! ```
//!
//! This is numerically stable and avoids storing all past checkpoints.
//!
//! When combined with a cyclical or constant learning rate (via [`SWALR`]),
//! SWA explores a wider region of the loss landscape and converges to
//! flatter minima that generalize better.
//!
//! # References
//!
//! - Izmailov et al., "Averaging Weights Leads to Wider Optima and Better
//!   Generalization" (UAI 2018)
//!
//! # CL-321

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, no_grad};
use ferrotorch_nn::Parameter;

use crate::foreach_utils::f64_scalar_on;
use crate::optimizer::Optimizer;
use crate::scheduler::LrScheduler;

// ---------------------------------------------------------------------------
// Averaging strategy
// ---------------------------------------------------------------------------

/// Averaging strategy for [`AveragedModel`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AveragingStrategy {
    /// Equal-weight running mean (SWA):
    /// `avg = avg + (param - avg) / (n + 1)`
    Swa,
    /// Exponential moving average:
    /// `avg = decay * avg + (1 - decay) * param`
    Ema(f64),
}

// ---------------------------------------------------------------------------
// AveragedModel
// ---------------------------------------------------------------------------

/// Maintains a running average of model parameters for SWA or EMA.
///
/// Unlike [`ExponentialMovingAverage`](crate::ema::ExponentialMovingAverage),
/// which is a standalone utility, `AveragedModel` is designed to integrate
/// into the training loop with a configurable averaging strategy and supports
/// both SWA (equal-weight) and EMA averaging.
///
/// # Example
///
/// ```ignore
/// // SWA: equal-weight average
/// let mut swa = AveragedModel::new(&params, AveragingStrategy::Swa);
///
/// // EMA: exponential decay
/// let mut ema = AveragedModel::new(&params, AveragingStrategy::Ema(0.999));
///
/// for epoch in 0..total_epochs {
///     train_one_epoch(&mut model, &mut optimizer);
///     if epoch >= swa_start {
///         swa.update_parameters(&params)?;
///     }
/// }
///
/// swa.apply_to(&params)?;  // copy averaged weights into model
/// ```
#[derive(Debug, Clone)]
pub struct AveragedModel<T: Float> {
    /// The averaging strategy (SWA or EMA).
    strategy: AveragingStrategy,
    /// Averaged parameter values. One `Vec<T>` per parameter. Used when
    /// `foreach == false`.
    averaged_params: Vec<Vec<T>>,
    /// On-device averaged tensors. Used when `foreach == true`.
    averaged_tensors: Vec<Tensor<T>>,
    /// Number of times `update_parameters` has been called.
    n_averaged: u64,
    /// When `true`, keep averaged state on-device and update via tensor ops.
    /// CL-497
    foreach: bool,
}

impl<T: Float> AveragedModel<T> {
    /// Create a new `AveragedModel` by cloning the current parameter values.
    ///
    /// # Arguments
    ///
    /// * `params` — Parameters to average over.
    /// * `strategy` — [`AveragingStrategy::Swa`] for equal-weight averaging,
    ///   or [`AveragingStrategy::Ema`] with a decay coefficient.
    ///
    /// # Panics
    ///
    /// Panics if `Ema(decay)` has `decay` outside `[0, 1]`.
    pub fn new(params: &[Parameter<T>], strategy: AveragingStrategy) -> Self {
        if let AveragingStrategy::Ema(decay) = strategy {
            assert!(
                (0.0..=1.0).contains(&decay),
                "EMA decay must be in [0, 1], got {decay}"
            );
        }

        let averaged_params: Vec<Vec<T>> = params
            .iter()
            .map(|p| p.data().unwrap_or_default().to_vec())
            .collect();

        Self {
            strategy,
            averaged_params,
            averaged_tensors: Vec::new(),
            n_averaged: 0,
            foreach: false,
        }
    }

    /// Enable the on-device (foreach) averaging path.
    ///
    /// When enabled, the running average is kept as `Tensor<T>` on the
    /// parameter's native device and updated via GPU-aware tensor ops
    /// instead of a CPU scalar loop, avoiding per-step CPU↔GPU roundtrips.
    ///
    /// Must be called with the same `params` that `new` was called with.
    /// The averaged tensors are deep-copied so they don't alias the
    /// parameter storage. CL-497
    pub fn with_foreach(mut self, params: &[Parameter<T>]) -> Self {
        assert_eq!(
            params.len(),
            self.averaged_params.len(),
            "with_foreach: parameter count mismatch"
        );

        self.averaged_tensors = params
            .iter()
            .map(|p| {
                let data = p.data_vec().expect("with_foreach: read parameter data");
                let shape = p.tensor().shape().to_vec();
                let device = p.tensor().device();
                let t =
                    Tensor::from_storage(ferrotorch_core::TensorStorage::cpu(data), shape, false)
                        .expect("with_foreach: construct averaged tensor");
                t.to(device).expect("with_foreach: move to device")
            })
            .collect();
        self.averaged_params.clear();
        self.foreach = true;
        self
    }

    /// Return the number of averaging updates performed.
    pub fn n_averaged(&self) -> u64 {
        self.n_averaged
    }

    /// Return the averaging strategy.
    pub fn strategy(&self) -> AveragingStrategy {
        self.strategy
    }

    /// Return the averaged parameter values (CPU path only).
    pub fn averaged_params(&self) -> &[Vec<T>] {
        &self.averaged_params
    }

    /// Read the averaged values for a given parameter index as a Vec.
    /// Works for both the CPU and foreach paths.
    pub fn averaged_values(&self, index: usize) -> FerrotorchResult<Vec<T>> {
        if self.foreach {
            self.averaged_tensors[index].data_vec()
        } else {
            Ok(self.averaged_params[index].clone())
        }
    }

    /// Update the averaged parameters from the current model parameters.
    ///
    /// On the first call (`n_averaged == 0`), the current parameters are
    /// simply copied. Subsequent calls apply the configured averaging
    /// strategy.
    pub fn update_parameters(&mut self, params: &[Parameter<T>]) -> FerrotorchResult<()> {
        if self.foreach {
            return self.update_parameters_foreach(params);
        }

        assert_eq!(
            params.len(),
            self.averaged_params.len(),
            "parameter count mismatch: expected {}, got {}",
            self.averaged_params.len(),
            params.len()
        );

        if self.n_averaged == 0 {
            // First update: just copy the parameters.
            for (avg, param) in self.averaged_params.iter_mut().zip(params.iter()) {
                let param_data = param.data()?;
                avg.copy_from_slice(param_data);
            }
        } else {
            match self.strategy {
                AveragingStrategy::Swa => {
                    // avg = avg + (param - avg) / (n + 1)
                    let n_plus_1 = cast::<u64, T>(self.n_averaged + 1)?;
                    for (avg, param) in self.averaged_params.iter_mut().zip(params.iter()) {
                        let param_data = param.data()?;
                        for (a, &p) in avg.iter_mut().zip(param_data.iter()) {
                            *a = *a + (p - *a) / n_plus_1;
                        }
                    }
                }
                AveragingStrategy::Ema(decay) => {
                    let decay_t = cast::<f64, T>(decay)?;
                    let one_minus_decay_t = cast::<f64, T>(1.0 - decay)?;
                    for (avg, param) in self.averaged_params.iter_mut().zip(params.iter()) {
                        let param_data = param.data()?;
                        for (a, &p) in avg.iter_mut().zip(param_data.iter()) {
                            *a = decay_t * *a + one_minus_decay_t * p;
                        }
                    }
                }
            }
        }

        self.n_averaged += 1;
        Ok(())
    }

    /// Foreach (on-device) averaging update. CL-497
    fn update_parameters_foreach(&mut self, params: &[Parameter<T>]) -> FerrotorchResult<()> {
        use ferrotorch_core::grad_fns::arithmetic::{add, div, mul, sub};

        assert_eq!(
            params.len(),
            self.averaged_tensors.len(),
            "foreach averaged model: parameter count mismatch: expected {}, got {}",
            self.averaged_tensors.len(),
            params.len()
        );

        no_grad(|| {
            if self.n_averaged == 0 {
                // First update: deep-copy parameter data into the averaged
                // tensors to avoid aliasing storage.
                for (avg, param) in self.averaged_tensors.iter_mut().zip(params.iter()) {
                    let data = param.data_vec()?;
                    let shape = param.tensor().shape().to_vec();
                    let device = param.tensor().device();
                    let fresh = Tensor::from_storage(
                        ferrotorch_core::TensorStorage::cpu(data),
                        shape,
                        false,
                    )?
                    .to(device)?;
                    *avg = fresh;
                }
            } else {
                match self.strategy {
                    AveragingStrategy::Swa => {
                        // avg = avg + (param - avg) / (n + 1)
                        for (avg, param) in self.averaged_tensors.iter_mut().zip(params.iter()) {
                            let param_t = param.tensor().clone();
                            let device = param_t.device();
                            let n_plus_1 =
                                f64_scalar_on::<T>((self.n_averaged + 1) as f64, device)?;
                            let diff = sub(&param_t, &*avg)?;
                            let delta = div(&diff, &n_plus_1)?;
                            let new_avg = add(&*avg, &delta)?;
                            *avg = new_avg;
                        }
                    }
                    AveragingStrategy::Ema(decay) => {
                        let one_minus_decay = 1.0 - decay;
                        for (avg, param) in self.averaged_tensors.iter_mut().zip(params.iter()) {
                            let param_t = param.tensor().clone();
                            let device = param_t.device();
                            let decay_t = f64_scalar_on::<T>(decay, device)?;
                            let one_minus_decay_t = f64_scalar_on::<T>(one_minus_decay, device)?;
                            let scaled_avg = mul(&*avg, &decay_t)?;
                            let scaled_param = mul(&param_t, &one_minus_decay_t)?;
                            let new_avg = add(&scaled_avg, &scaled_param)?;
                            *avg = new_avg;
                        }
                    }
                }
            }
            Ok::<(), ferrotorch_core::FerrotorchError>(())
        })?;

        self.n_averaged += 1;
        Ok(())
    }

    /// Copy the averaged parameter values into the given parameters.
    ///
    /// Use this after training to load the averaged weights for inference.
    ///
    /// Takes `&mut self` even though it does not mutate `self`'s data: the
    /// `&mut` receiver is a borrow-checker-enforced witness that no other
    /// caller is concurrently observing the parameter storage that this
    /// method writes to via `update_data` / `update_storage`. This is the
    /// borrow-checker analog of the `sole-writer` contract documented on
    /// the unsafe blocks below.
    pub fn apply_to(&mut self, params: &[Parameter<T>]) -> FerrotorchResult<()> {
        if self.foreach {
            assert_eq!(params.len(), self.averaged_tensors.len());
            no_grad(|| {
                for (avg, param) in self.averaged_tensors.iter().zip(params.iter()) {
                    let avg_clone = avg.clone();
                    let (storage, _) = avg_clone.into_storage_and_shape()?;
                    // SAFETY: `update_storage` swaps the parameter's storage
                    // Arc; sole-writer required.
                    //  1. `apply_to(&mut self, params)` takes `&mut self`,
                    //     so the borrow checker enforces that no other
                    //     `apply_to` / `update_parameters` call on this
                    //     `AveragedModel` runs concurrently. The caller
                    //     still owes the `Tensor::update_storage` contract
                    //     on the parameter side: no other live handle may
                    //     observe `params[i].tensor()`'s storage during
                    //     this call.
                    //  2. We are inside `no_grad`, so no autograd `grad_fn`
                    //     records a clone of the param's storage Arc as
                    //     part of this swap.
                    //  3. `avg_clone` is a fresh clone of the averaged
                    //     tensor, disjoint from any parameter's storage;
                    //     `into_storage_and_shape` consumes `avg_clone` and
                    //     yields the storage we install into the param.
                    //  4. `self.averaged_tensors[i]` was constructed on the
                    //     parameter's device in `update_parameters_foreach`
                    //     so device + numel match.
                    unsafe { param.tensor().update_storage(storage)? };
                }
                Ok::<(), ferrotorch_core::FerrotorchError>(())
            })?;
            return Ok(());
        }

        assert_eq!(params.len(), self.averaged_params.len());

        for (avg, param) in self.averaged_params.iter().zip(params.iter()) {
            // SAFETY: `update_data` writes through `Arc::as_ptr`. Same
            // contract as the foreach branch above:
            //  1. `apply_to(&mut self, ..)` takes `&mut self`, so the
            //     borrow checker enforces that no other `apply_to` /
            //     `update_parameters` call runs concurrently. The caller
            //     still owes the `Tensor::update_data` contract on the
            //     parameter side: no other live handle may observe the
            //     parameter's storage during this call.
            //  2. The `no_grad` closure suppresses `grad_fn` recording.
            //  3. `avg: &Vec<T>` borrows from `self.averaged_params` and
            //     is disjoint from any parameter's storage.
            no_grad(|| unsafe { param.tensor().update_data(avg) })?;
        }

        Ok(())
    }
}

// ---------------------------------------------------------------------------
// SWALR — SWA learning rate schedule
// ---------------------------------------------------------------------------

/// Annealing strategy for [`SWALR`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnnealStrategy {
    /// Cosine annealing: `alpha = (1 - cos(pi * t)) / 2`
    Cosine,
    /// Linear annealing: `alpha = t`
    Linear,
}

/// SWA learning rate scheduler.
///
/// Anneals the learning rate from its current value to `swa_lr` over
/// `anneal_epochs` steps using either cosine or linear interpolation.
/// After the annealing phase, the learning rate stays constant at `swa_lr`.
///
/// This scheduler is meant to be used during the SWA phase of training,
/// typically switched to after a warmup + cosine/step decay phase.
///
/// # Example
///
/// ```ignore
/// let mut swa_scheduler = SWALR::new(0.05, 10, AnnealStrategy::Cosine);
///
/// for epoch in swa_start..total_epochs {
///     train_one_epoch(&mut model, &mut optimizer);
///     swa_model.update_parameters(&params)?;
///     swa_scheduler.step(&mut optimizer);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct Swalr {
    /// Target SWA learning rate.
    swa_lr: f64,
    /// Number of epochs for the annealing phase.
    anneal_epochs: usize,
    /// Annealing strategy (cosine or linear).
    anneal_strategy: AnnealStrategy,
    /// Current step counter.
    current_step: usize,
    /// The initial learning rate captured at the first step.
    initial_lr: Option<f64>,
    /// Current computed learning rate.
    current_lr: f64,
}

impl Swalr {
    /// Create a new SWA learning rate scheduler.
    ///
    /// # Arguments
    ///
    /// * `swa_lr` — Target learning rate for the SWA phase.
    /// * `anneal_epochs` — Number of steps over which to anneal from the
    ///   current LR to `swa_lr`. Set to 0 for immediate switch.
    /// * `anneal_strategy` — [`AnnealStrategy::Cosine`] or
    ///   [`AnnealStrategy::Linear`].
    pub fn new(swa_lr: f64, anneal_epochs: usize, anneal_strategy: AnnealStrategy) -> Self {
        Self {
            swa_lr,
            anneal_epochs,
            anneal_strategy,
            current_step: 0,
            initial_lr: None,
            current_lr: swa_lr,
        }
    }

    /// Return the target SWA learning rate.
    pub fn swa_lr(&self) -> f64 {
        self.swa_lr
    }

    /// Return the annealing epoch count.
    pub fn anneal_epochs(&self) -> usize {
        self.anneal_epochs
    }

    /// Compute the interpolation factor at progress `t` in [0, 1].
    fn anneal_factor(&self, t: f64) -> f64 {
        match self.anneal_strategy {
            AnnealStrategy::Cosine => (1.0 - (std::f64::consts::PI * t).cos()) / 2.0,
            AnnealStrategy::Linear => t,
        }
    }
}

impl<T: Float> LrScheduler<T> for Swalr {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
        // Capture the initial LR on the first call.
        if self.initial_lr.is_none() {
            self.initial_lr = Some(optimizer.lr());
        }
        let initial_lr = self.initial_lr.unwrap();

        self.current_step += 1;

        let lr = if self.anneal_epochs == 0 {
            // Immediate switch.
            self.swa_lr
        } else {
            let t = (self.current_step as f64 / self.anneal_epochs as f64).clamp(0.0, 1.0);
            let alpha = self.anneal_factor(t);
            // Interpolate: initial_lr * (1 - alpha) + swa_lr * alpha
            initial_lr * (1.0 - alpha) + self.swa_lr * alpha
        };

        self.current_lr = lr;
        optimizer.set_lr(lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{OptimizerState, ParamGroup};

    // ---- SAFETY note shared by all `update_data` test sites below --------
    //
    // Each test that uses `no_grad(|| unsafe { p.tensor().update_data(&[..]) })`
    // is simulating "an optimizer step has changed the parameter" before
    // driving SWA logic. The call requires sole-writer access to `p`'s
    // storage:
    //  1. Each `Parameter<T>` is constructed locally in the test (e.g. via
    //     `Parameter::from_slice`/`Parameter::zeros`); the only outstanding
    //     handles are `p` itself plus the shared slices passed to SWA
    //     methods (`std::slice::from_ref(&p)`, `&[p]`).
    //  2. The `no_grad` closure suppresses autograd `grad_fn` recording for
    //     the write — no node retains a clone of the storage Arc.
    //  3. `AveragedModel` keeps its averaged state in disjoint
    //     `Vec<T>` / `Tensor<T>` storages and never aliases parameter
    //     storage.
    //  4. Each test runs single-threaded relative to its own parameters;
    //     `cargo test` may schedule tests in parallel, but each test owns
    //     its own `Parameter`s.
    // Per-site comments below cite this block by name where they need to
    // point out anything site-specific.

    // -----------------------------------------------------------------------
    // Mock optimizer for scheduler tests
    // -----------------------------------------------------------------------

    struct MockOptimizer {
        lr: f64,
    }

    impl MockOptimizer {
        fn new(lr: f64) -> Self {
            Self { lr }
        }
    }

    impl Optimizer<f32> for MockOptimizer {
        fn step(&mut self) -> FerrotorchResult<()> {
            Ok(())
        }
        fn zero_grad(&mut self) -> FerrotorchResult<()> {
            Ok(())
        }
        fn lr(&self) -> f64 {
            self.lr
        }
        fn set_lr(&mut self, lr: f64) {
            self.lr = lr;
        }
        fn param_groups(&self) -> &[ParamGroup<f32>] {
            &[]
        }
        fn param_groups_mut(&mut self) -> &mut [ParamGroup<f32>] {
            &mut []
        }
        fn add_param_group(&mut self, _group: ParamGroup<f32>) {
            // MockOptimizer is test-only and doesn't manage real parameter
            // groups; its param_groups() always returns an empty slice. If
            // any test tries to add a group, it's a bug in the test.
            panic!("MockOptimizer::add_param_group called — this mock holds no real param groups");
        }
        fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
            Ok(Default::default())
        }
        fn load_state_dict(&mut self, _state: &OptimizerState) -> FerrotorchResult<()> {
            Ok(())
        }
    }

    // =======================================================================
    // AveragedModel — SWA
    // =======================================================================

    #[test]
    fn test_averaged_model_swa_first_update_copies() {
        let p = Parameter::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let mut avg = AveragedModel::new(std::slice::from_ref(&p), AveragingStrategy::Swa);

        // Change param before first update.
        // SAFETY: see test-module note above; `p` is local to this test.
        no_grad(|| unsafe { p.tensor().update_data(&[10.0f32, 20.0, 30.0]) }).unwrap();
        avg.update_parameters(&[p]).unwrap();

        assert_eq!(avg.averaged_params()[0], vec![10.0f32, 20.0, 30.0]);
        assert_eq!(avg.n_averaged(), 1);
    }

    #[test]
    fn test_averaged_model_swa_running_mean() {
        // n=0: avg = [1] (copy)
        // n=1: avg = [1] + ([3] - [1]) / 2 = [2]
        // n=2: avg = [2] + ([6] - [2]) / 3 = [2 + 4/3] = [10/3]
        let p = Parameter::from_slice(&[0.0f32], &[1]).unwrap();
        let mut avg = AveragedModel::new(std::slice::from_ref(&p), AveragingStrategy::Swa);

        // Update 1: param = 1
        // SAFETY: see test-module note above; `p` is local to this test.
        no_grad(|| unsafe { p.tensor().update_data(&[1.0f32]) }).unwrap();
        avg.update_parameters(std::slice::from_ref(&p)).unwrap();
        assert!((avg.averaged_params()[0][0] - 1.0).abs() < 1e-6);

        // Update 2: param = 3
        // SAFETY: see test-module note above; `p` is local to this test.
        no_grad(|| unsafe { p.tensor().update_data(&[3.0f32]) }).unwrap();
        avg.update_parameters(std::slice::from_ref(&p)).unwrap();
        // avg = 1 + (3 - 1) / 2 = 2
        assert!(
            (avg.averaged_params()[0][0] - 2.0).abs() < 1e-6,
            "expected 2.0, got {}",
            avg.averaged_params()[0][0]
        );

        // Update 3: param = 6
        // SAFETY: see test-module note above; `p` is local to this test.
        no_grad(|| unsafe { p.tensor().update_data(&[6.0f32]) }).unwrap();
        avg.update_parameters(&[p]).unwrap();
        // avg = 2 + (6 - 2) / 3 = 10/3 ≈ 3.333
        let expected = 10.0 / 3.0;
        assert!(
            (avg.averaged_params()[0][0] - expected).abs() < 1e-5,
            "expected {expected}, got {}",
            avg.averaged_params()[0][0]
        );
        assert_eq!(avg.n_averaged(), 3);
    }

    #[test]
    fn test_averaged_model_swa_equal_weight_mean() {
        // If we average 4 identical values, the mean should be that value.
        let p = Parameter::from_slice(&[5.0f32, 10.0], &[2]).unwrap();
        let mut avg = AveragedModel::new(std::slice::from_ref(&p), AveragingStrategy::Swa);

        for _ in 0..4 {
            avg.update_parameters(std::slice::from_ref(&p)).unwrap();
        }

        assert!((avg.averaged_params()[0][0] - 5.0).abs() < 1e-6);
        assert!((avg.averaged_params()[0][1] - 10.0).abs() < 1e-6);
    }

    // =======================================================================
    // AveragedModel — EMA
    // =======================================================================

    #[test]
    fn test_averaged_model_ema() {
        let p = Parameter::from_slice(&[0.0f32], &[1]).unwrap();
        let mut avg = AveragedModel::new(std::slice::from_ref(&p), AveragingStrategy::Ema(0.5));

        // Update 1 (first call copies): avg = 10
        // SAFETY: see test-module note above; `p` is local to this test.
        no_grad(|| unsafe { p.tensor().update_data(&[10.0f32]) }).unwrap();
        avg.update_parameters(std::slice::from_ref(&p)).unwrap();
        assert!((avg.averaged_params()[0][0] - 10.0).abs() < 1e-6);

        // Update 2: avg = 0.5 * 10 + 0.5 * 20 = 15
        // SAFETY: see test-module note above; `p` is local to this test.
        no_grad(|| unsafe { p.tensor().update_data(&[20.0f32]) }).unwrap();
        avg.update_parameters(&[p]).unwrap();
        assert!(
            (avg.averaged_params()[0][0] - 15.0).abs() < 1e-6,
            "expected 15.0, got {}",
            avg.averaged_params()[0][0]
        );
    }

    // =======================================================================
    // AveragedModel — apply_to
    // =======================================================================

    #[test]
    fn test_averaged_model_apply_to() {
        let p = Parameter::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let mut avg = AveragedModel::new(std::slice::from_ref(&p), AveragingStrategy::Swa);

        // Update with [1, 2], then [3, 4].
        avg.update_parameters(std::slice::from_ref(&p)).unwrap();

        // SAFETY: see test-module note above; `p` is local to this test.
        no_grad(|| unsafe { p.tensor().update_data(&[3.0f32, 4.0]) }).unwrap();
        avg.update_parameters(std::slice::from_ref(&p)).unwrap();
        // avg = [1, 2] + ([3, 4] - [1, 2]) / 2 = [2, 3]

        // Apply averaged weights to the parameter.
        avg.apply_to(std::slice::from_ref(&p)).unwrap();
        let data = p.data().unwrap();
        assert!((data[0] - 2.0).abs() < 1e-6);
        assert!((data[1] - 3.0).abs() < 1e-6);
    }

    // =======================================================================
    // SWALR — cosine annealing
    // =======================================================================

    #[test]
    fn test_swalr_cosine_annealing() {
        let mut sched = Swalr::new(0.05, 10, AnnealStrategy::Cosine);
        let mut opt = MockOptimizer::new(0.1);

        // After annealing, LR should be at swa_lr.
        for _ in 0..10 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.05).abs() < 1e-10,
            "expected 0.05, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_swalr_cosine_midpoint() {
        let mut sched = Swalr::new(0.0, 100, AnnealStrategy::Cosine);
        let mut opt = MockOptimizer::new(1.0);

        for _ in 0..50 {
            sched.step(&mut opt);
        }
        // At midpoint: alpha = (1 - cos(pi * 0.5)) / 2 = 0.5
        // lr = 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        assert!(
            (opt.lr - 0.5).abs() < 1e-10,
            "cosine midpoint: expected 0.5, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_swalr_stays_at_swa_lr_after_anneal() {
        let mut sched = Swalr::new(0.01, 5, AnnealStrategy::Cosine);
        let mut opt = MockOptimizer::new(0.1);

        // Run well past anneal phase.
        for _ in 0..20 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.01).abs() < 1e-10,
            "post-anneal: expected 0.01, got {}",
            opt.lr
        );
    }

    // =======================================================================
    // SWALR — linear annealing
    // =======================================================================

    #[test]
    fn test_swalr_linear_annealing() {
        let mut sched = Swalr::new(0.0, 10, AnnealStrategy::Linear);
        let mut opt = MockOptimizer::new(1.0);

        // Step 5: t = 5/10 = 0.5, alpha = 0.5
        // lr = 1.0 * 0.5 + 0.0 * 0.5 = 0.5
        for _ in 0..5 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.5).abs() < 1e-10,
            "linear midpoint: expected 0.5, got {}",
            opt.lr
        );

        // Step 10: t = 1.0, alpha = 1.0, lr = swa_lr = 0.0
        for _ in 0..5 {
            sched.step(&mut opt);
        }
        assert!(
            opt.lr.abs() < 1e-10,
            "linear end: expected 0.0, got {}",
            opt.lr
        );
    }

    // =======================================================================
    // SWALR — immediate switch (anneal_epochs = 0)
    // =======================================================================

    #[test]
    fn test_swalr_immediate_switch() {
        let mut sched = Swalr::new(0.05, 0, AnnealStrategy::Cosine);
        let mut opt = MockOptimizer::new(0.1);

        sched.step(&mut opt);
        assert!(
            (opt.lr - 0.05).abs() < 1e-10,
            "immediate: expected 0.05, got {}",
            opt.lr
        );
    }

    // =======================================================================
    // SWALR — get_lr
    // =======================================================================

    #[test]
    fn test_swalr_get_lr() {
        let mut sched = Swalr::new(0.01, 10, AnnealStrategy::Linear);
        let mut opt = MockOptimizer::new(0.1);

        sched.step(&mut opt);
        // t = 1/10 = 0.1, alpha = 0.1
        // lr = 0.1 * 0.9 + 0.01 * 0.1 = 0.09 + 0.001 = 0.091
        let expected = 0.1 * 0.9 + 0.01 * 0.1;
        let actual = <Swalr as LrScheduler<f32>>::get_lr(&sched);
        assert!(
            (actual - expected).abs() < 1e-10,
            "get_lr: expected {expected}, got {actual}",
        );
    }

    // =======================================================================
    // Edge cases
    // =======================================================================

    #[test]
    #[should_panic(expected = "EMA decay must be in [0, 1]")]
    fn test_averaged_model_ema_invalid_decay() {
        let p = Parameter::<f32>::zeros(&[1]).unwrap();
        let _avg = AveragedModel::new(&[p], AveragingStrategy::Ema(1.5));
    }

    #[test]
    #[should_panic(expected = "parameter count mismatch")]
    fn test_averaged_model_param_count_mismatch() {
        let p1 = Parameter::<f32>::zeros(&[1]).unwrap();
        let p2 = Parameter::<f32>::zeros(&[1]).unwrap();
        let mut avg = AveragedModel::new(&[p1], AveragingStrategy::Swa);
        avg.update_parameters(&[p2.clone(), p2]).unwrap();
    }

    // =======================================================================
    // Foreach mode parity tests. CL-497
    // =======================================================================

    #[test]
    fn test_averaged_model_swa_foreach_parity() {
        let p_legacy = Parameter::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
        let p_foreach = Parameter::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();

        let mut legacy =
            AveragedModel::new(std::slice::from_ref(&p_legacy), AveragingStrategy::Swa);
        let mut foreach =
            AveragedModel::new(std::slice::from_ref(&p_foreach), AveragingStrategy::Swa)
                .with_foreach(std::slice::from_ref(&p_foreach));

        for step in 0..5 {
            let new_vals: Vec<f32> = (0..4).map(|i| 1.0 + i as f32 + step as f32).collect();
            // SAFETY: see test-module note above; `p_legacy` and `p_foreach`
            // are disjoint local parameters with their own storage Arcs.
            no_grad(|| unsafe { p_legacy.tensor().update_data(&new_vals) }).unwrap();
            // SAFETY: see test-module note above; `p_foreach` is local.
            no_grad(|| unsafe { p_foreach.tensor().update_data(&new_vals) }).unwrap();

            legacy
                .update_parameters(std::slice::from_ref(&p_legacy))
                .unwrap();
            foreach
                .update_parameters(std::slice::from_ref(&p_foreach))
                .unwrap();
        }

        let l = legacy.averaged_values(0).unwrap();
        let f = foreach.averaged_values(0).unwrap();
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "swa foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_averaged_model_ema_foreach_parity() {
        let p_legacy = Parameter::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();
        let p_foreach = Parameter::from_slice(&[0.0f32, 0.0, 0.0], &[3]).unwrap();

        let mut legacy =
            AveragedModel::new(std::slice::from_ref(&p_legacy), AveragingStrategy::Ema(0.9));
        let mut foreach = AveragedModel::new(
            std::slice::from_ref(&p_foreach),
            AveragingStrategy::Ema(0.9),
        )
        .with_foreach(std::slice::from_ref(&p_foreach));

        for step in 0..6 {
            let v = 10.0 + step as f32;
            let new_vals = vec![v, v + 1.0, v + 2.0];
            // SAFETY: see test-module note above; `p_legacy` and `p_foreach`
            // are disjoint local parameters with their own storage Arcs.
            no_grad(|| unsafe { p_legacy.tensor().update_data(&new_vals) }).unwrap();
            // SAFETY: see test-module note above; `p_foreach` is local.
            no_grad(|| unsafe { p_foreach.tensor().update_data(&new_vals) }).unwrap();

            legacy
                .update_parameters(std::slice::from_ref(&p_legacy))
                .unwrap();
            foreach
                .update_parameters(std::slice::from_ref(&p_foreach))
                .unwrap();
        }

        let l = legacy.averaged_values(0).unwrap();
        let f = foreach.averaged_values(0).unwrap();
        for (a, b) in l.iter().zip(f.iter()) {
            assert!(
                (a - b).abs() < 1e-4,
                "avg-ema foreach parity: legacy={a}, foreach={b}"
            );
        }
    }

    #[test]
    fn test_averaged_model_swa_foreach_apply_to() {
        let p = Parameter::from_slice(&[10.0f32, 20.0], &[2]).unwrap();
        let mut avg = AveragedModel::new(std::slice::from_ref(&p), AveragingStrategy::Swa)
            .with_foreach(std::slice::from_ref(&p));

        // Step 1: first update copies [10, 20].
        avg.update_parameters(std::slice::from_ref(&p)).unwrap();

        // Step 2: param = [2, 4]. avg = 10 + (2 - 10)/2 = 6; 20 + (4-20)/2 = 12
        // SAFETY: see test-module note above; `p` is local to this test.
        no_grad(|| unsafe { p.tensor().update_data(&[2.0f32, 4.0]) }).unwrap();
        avg.update_parameters(std::slice::from_ref(&p)).unwrap();

        // Apply averaged values into param.
        avg.apply_to(std::slice::from_ref(&p)).unwrap();
        let data = p.data().unwrap();
        assert!(
            (data[0] - 6.0).abs() < 1e-5,
            "expected 6.0, got {}",
            data[0]
        );
        assert!(
            (data[1] - 12.0).abs() < 1e-5,
            "expected 12.0, got {}",
            data[1]
        );
    }
}
