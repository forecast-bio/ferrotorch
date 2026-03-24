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

use ferrotorch_core::{no_grad, Float, FerrotorchResult};
use ferrotorch_nn::Parameter;

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
    /// Averaged parameter values. One `Vec<T>` per parameter.
    averaged_params: Vec<Vec<T>>,
    /// Number of times `update_parameters` has been called.
    n_averaged: u64,
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
            n_averaged: 0,
        }
    }

    /// Return the number of averaging updates performed.
    pub fn n_averaged(&self) -> u64 {
        self.n_averaged
    }

    /// Return the averaging strategy.
    pub fn strategy(&self) -> AveragingStrategy {
        self.strategy
    }

    /// Return the averaged parameter values.
    pub fn averaged_params(&self) -> &[Vec<T>] {
        &self.averaged_params
    }

    /// Update the averaged parameters from the current model parameters.
    ///
    /// On the first call (`n_averaged == 0`), the current parameters are
    /// simply copied. Subsequent calls apply the configured averaging
    /// strategy.
    pub fn update_parameters(&mut self, params: &[Parameter<T>]) -> FerrotorchResult<()> {
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
                avg.copy_from_slice(&param_data);
            }
        } else {
            match self.strategy {
                AveragingStrategy::Swa => {
                    // avg = avg + (param - avg) / (n + 1)
                    let n_plus_1 = T::from(self.n_averaged + 1).unwrap();
                    for (avg, param) in self.averaged_params.iter_mut().zip(params.iter()) {
                        let param_data = param.data()?;
                        for (a, &p) in avg.iter_mut().zip(param_data.iter()) {
                            *a = *a + (p - *a) / n_plus_1;
                        }
                    }
                }
                AveragingStrategy::Ema(decay) => {
                    let decay_t = T::from(decay).unwrap();
                    let one_minus_decay_t = T::from(1.0 - decay).unwrap();
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

    /// Copy the averaged parameter values into the given parameters.
    ///
    /// Use this after training to load the averaged weights for inference.
    pub fn apply_to(&self, params: &[Parameter<T>]) -> FerrotorchResult<()> {
        assert_eq!(params.len(), self.averaged_params.len());

        for (avg, param) in self.averaged_params.iter().zip(params.iter()) {
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
            let t = (self.current_step as f64 / self.anneal_epochs as f64).min(1.0).max(0.0);
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
        fn add_param_group(&mut self, _group: ParamGroup<f32>) {}
        fn state_dict(&self) -> OptimizerState {
            Default::default()
        }
        fn load_state_dict(
            &mut self,
            _state: &OptimizerState,
        ) -> FerrotorchResult<()> {
            Ok(())
        }
    }

    // =======================================================================
    // AveragedModel — SWA
    // =======================================================================

    #[test]
    fn test_averaged_model_swa_first_update_copies() {
        let p = Parameter::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let mut avg = AveragedModel::new(&[p.clone()], AveragingStrategy::Swa);

        // Change param before first update.
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
        let mut avg = AveragedModel::new(&[p.clone()], AveragingStrategy::Swa);

        // Update 1: param = 1
        no_grad(|| unsafe { p.tensor().update_data(&[1.0f32]) }).unwrap();
        avg.update_parameters(&[p.clone()]).unwrap();
        assert!((avg.averaged_params()[0][0] - 1.0).abs() < 1e-6);

        // Update 2: param = 3
        no_grad(|| unsafe { p.tensor().update_data(&[3.0f32]) }).unwrap();
        avg.update_parameters(&[p.clone()]).unwrap();
        // avg = 1 + (3 - 1) / 2 = 2
        assert!(
            (avg.averaged_params()[0][0] - 2.0).abs() < 1e-6,
            "expected 2.0, got {}",
            avg.averaged_params()[0][0]
        );

        // Update 3: param = 6
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
        let mut avg = AveragedModel::new(&[p.clone()], AveragingStrategy::Swa);

        for _ in 0..4 {
            avg.update_parameters(&[p.clone()]).unwrap();
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
        let mut avg = AveragedModel::new(&[p.clone()], AveragingStrategy::Ema(0.5));

        // Update 1 (first call copies): avg = 10
        no_grad(|| unsafe { p.tensor().update_data(&[10.0f32]) }).unwrap();
        avg.update_parameters(&[p.clone()]).unwrap();
        assert!((avg.averaged_params()[0][0] - 10.0).abs() < 1e-6);

        // Update 2: avg = 0.5 * 10 + 0.5 * 20 = 15
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
        let mut avg = AveragedModel::new(&[p.clone()], AveragingStrategy::Swa);

        // Update with [1, 2], then [3, 4].
        avg.update_parameters(&[p.clone()]).unwrap();

        no_grad(|| unsafe { p.tensor().update_data(&[3.0f32, 4.0]) }).unwrap();
        avg.update_parameters(&[p.clone()]).unwrap();
        // avg = [1, 2] + ([3, 4] - [1, 2]) / 2 = [2, 3]

        // Apply averaged weights to the parameter.
        avg.apply_to(&[p.clone()]).unwrap();
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
}
