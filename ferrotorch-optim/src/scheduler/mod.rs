//! Learning rate schedulers for ferrotorch optimizers.
//!
//! This module provides schedulers that adjust the learning rate during
//! training. Two scheduler traits are defined:
//!
//! - [`LrScheduler`] — step-based schedulers that adjust LR every step.
//! - [`MetricScheduler`] — metric-aware schedulers (e.g., reduce on plateau).
//!
//! # Provided schedulers
//!
//! | Scheduler | Strategy |
//! |-----------|----------|
//! | [`StepLR`] | Multiply LR by `gamma` every `step_size` steps |
//! | [`MultiStepLR`] | Multiply LR by `gamma` at specific milestones |
//! | [`ExponentialLR`] | Multiply LR by `gamma` every step |
//! | [`CosineAnnealingLR`] | Cosine decay from `base_lr` to `eta_min` |
//! | [`CosineAnnealingWarmRestarts`] | Cosine decay with periodic warm restarts (SGDR) |
//! | [`CyclicLR`] | Cycle LR between boundaries (triangular/exp_range) |
//! | [`OneCycleLR`] | Super-convergence one-cycle policy |
//! | [`PolynomialLR`] | Polynomial decay over a fixed number of steps |
//! | [`ConstantLR`] | Constant factor for a fixed number of steps |
//! | [`LinearLR`] | Linear factor ramp between two endpoints |
//! | [`LambdaLR`] | User-provided lambda function per step |
//! | [`LinearWarmup`] | Linear ramp from 0 to `base_lr` |
//! | [`ReduceLROnPlateau`] | Reduce LR when a metric stops improving |
//! | [`MultiplicativeLR`] | Multiply LR by a user-provided function each step |
//! | [`ChainedScheduler`] | Apply multiple schedulers in order every step |
//! | [`SequentialLr`] | Chain multiple schedulers with milestone switches |
//!
//! # Convenience constructors
//!
//! [`cosine_warmup_scheduler`] returns a [`SequentialLr`] that combines
//! [`LinearWarmup`] followed by [`CosineAnnealingLR`].
//!
//! [CL-320]

pub mod chained_scheduler;
pub mod constant_lr;
pub mod cosine;
pub mod cosine_warm_restarts;
pub mod cyclic_lr;
pub mod exponential_lr;
pub mod lambda_lr;
pub mod linear_lr;
pub mod multi_step_lr;
pub mod multiplicative_lr;
pub mod one_cycle_lr;
pub mod plateau;
pub mod polynomial_lr;
pub mod step;
pub mod warmup;

pub use chained_scheduler::ChainedScheduler;
pub use constant_lr::ConstantLR;
pub use cosine::CosineAnnealingLR;
pub use cosine_warm_restarts::CosineAnnealingWarmRestarts;
pub use cyclic_lr::{CyclicLR, CyclicMode};
pub use exponential_lr::ExponentialLR;
pub use lambda_lr::LambdaLR;
pub use linear_lr::LinearLR;
pub use multi_step_lr::MultiStepLR;
pub use multiplicative_lr::MultiplicativeLR;
pub use one_cycle_lr::{AnnealStrategy, OneCycleLR};
pub use plateau::{MetricScheduler, PlateauMode, ReduceLROnPlateau};
pub use polynomial_lr::PolynomialLR;
pub use step::StepLR;
pub use warmup::LinearWarmup;

use ferrotorch_core::Float;

use crate::optimizer::Optimizer;

// ---------------------------------------------------------------------------
// LrScheduler trait
// ---------------------------------------------------------------------------

/// A learning rate scheduler that adjusts the optimizer's LR each step.
///
/// Implementors track their own internal step counter and compute the new
/// learning rate from it.
pub trait LrScheduler<T: Float> {
    /// Advance the scheduler by one step and update the optimizer's LR.
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>);

    /// Return the current learning rate (without advancing the step).
    fn get_lr(&self) -> f64;
}

// ---------------------------------------------------------------------------
// SequentialLr
// ---------------------------------------------------------------------------

/// Chains multiple [`LrScheduler`]s, switching at specified milestones.
///
/// Each entry is a `(scheduler, milestone)` pair where `milestone` is the
/// **last global step** handled by that scheduler. The next scheduler
/// takes over at step `milestone + 1`.
///
/// # Example
///
/// ```ignore
/// use ferrotorch_optim::scheduler::{LinearWarmup, StepLR, SequentialLr};
///
/// let warmup = LinearWarmup::new(0.1, 1000);
/// let decay  = StepLR::new(0.1, 5000, 0.5);
///
/// // Warmup handles steps 1..=1000, then StepLR from step 1001 onward.
/// let scheduler = SequentialLr::new(vec![
///     (Box::new(warmup), 1000),
///     (Box::new(decay),  usize::MAX),
/// ]);
/// ```
pub struct SequentialLr<T: Float> {
    /// Scheduler / milestone pairs, ordered by milestone.
    schedulers: Vec<(Box<dyn LrScheduler<T>>, usize)>,
    /// Global step counter.
    current_step: usize,
}

impl<T: Float> SequentialLr<T> {
    /// Create a new `SequentialLr` from an ordered list of
    /// `(scheduler, milestone)` pairs.
    ///
    /// The `milestone` value is the global step at which the scheduler
    /// *stops* being active and the next one takes over. The last
    /// scheduler's milestone is typically `usize::MAX`.
    pub fn new(schedulers: Vec<(Box<dyn LrScheduler<T>>, usize)>) -> Self {
        Self {
            schedulers,
            current_step: 0,
        }
    }

    /// Return the index of the currently active scheduler.
    fn active_index(&self) -> usize {
        for (i, &(_, milestone)) in self.schedulers.iter().enumerate() {
            if self.current_step <= milestone {
                return i;
            }
        }
        // Past all milestones: use the last scheduler.
        self.schedulers.len().saturating_sub(1)
    }
}

impl<T: Float> LrScheduler<T> for SequentialLr<T> {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
        self.current_step += 1;
        let idx = self.active_index();
        if let Some((scheduler, _)) = self.schedulers.get_mut(idx) {
            scheduler.step(optimizer);
        }
    }

    fn get_lr(&self) -> f64 {
        let idx = self.active_index();
        self.schedulers
            .get(idx)
            .map(|(s, _)| s.get_lr())
            .unwrap_or(0.0)
    }
}

// ---------------------------------------------------------------------------
// Convenience: cosine warmup scheduler
// ---------------------------------------------------------------------------

/// Create a [`SequentialLr`] that linearly warms up and then cosine-decays.
///
/// # Arguments
///
/// * `base_lr` - Peak learning rate (reached at end of warmup).
/// * `warmup_steps` - Number of warmup steps.
/// * `total_steps` - Total number of training steps (warmup + decay).
/// * `min_lr` - Minimum learning rate at the end of cosine decay.
///
/// # Panics
///
/// Panics if `warmup_steps >= total_steps`.
pub fn cosine_warmup_scheduler<T: Float>(
    base_lr: f64,
    warmup_steps: usize,
    total_steps: usize,
    min_lr: f64,
) -> SequentialLr<T> {
    assert!(
        warmup_steps < total_steps,
        "warmup_steps ({warmup_steps}) must be less than total_steps ({total_steps})"
    );

    let warmup = LinearWarmup::new(base_lr, warmup_steps);
    let cosine_steps = total_steps - warmup_steps;
    let cosine = CosineAnnealingLR::new(base_lr, cosine_steps, min_lr);

    SequentialLr::new(vec![
        (Box::new(warmup), warmup_steps),
        (Box::new(cosine), usize::MAX),
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    struct MockOptimizer {
        lr: f64,
    }

    impl MockOptimizer {
        fn new(lr: f64) -> Self {
            Self { lr }
        }
    }

    impl Optimizer<f32> for MockOptimizer {
        fn step(&mut self) -> ferrotorch_core::FerrotorchResult<()> {
            Ok(())
        }
        fn zero_grad(&mut self) -> ferrotorch_core::FerrotorchResult<()> {
            Ok(())
        }
        fn lr(&self) -> f64 {
            self.lr
        }
        fn set_lr(&mut self, lr: f64) {
            self.lr = lr;
        }
        fn param_groups(&self) -> &[crate::optimizer::ParamGroup<f32>] {
            &[]
        }
        fn param_groups_mut(&mut self) -> &mut [crate::optimizer::ParamGroup<f32>] {
            &mut []
        }
        fn add_param_group(&mut self, _group: crate::optimizer::ParamGroup<f32>) {}
        fn state_dict(&self) -> ferrotorch_core::FerrotorchResult<crate::optimizer::OptimizerState> {
            Ok(Default::default())
        }
        fn load_state_dict(
            &mut self,
            _state: &crate::optimizer::OptimizerState,
        ) -> ferrotorch_core::FerrotorchResult<()> {
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // SequentialLr tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_sequential_warmup_then_step() {
        let warmup = LinearWarmup::new(0.1, 5);
        let step_sched = StepLR::new(0.1, 5, 0.5);

        let mut seq: SequentialLr<f32> = SequentialLr::new(vec![
            (Box::new(warmup), 5),
            (Box::new(step_sched), usize::MAX),
        ]);
        let mut opt = MockOptimizer::new(0.0);

        // Warmup phase: steps 1..5
        for step in 1..=5 {
            seq.step(&mut opt);
            let expected_warmup = 0.1 * (step as f64 / 5.0);
            assert!(
                (opt.lr - expected_warmup).abs() < 1e-12,
                "warmup step {step}: expected {expected_warmup}, got {}",
                opt.lr
            );
        }

        // Decay phase: steps 6..15, StepLR with step_size=5 gamma=0.5
        // StepLR internal steps: 1..5 -> gamma^0 = 1.0 (lr=0.1)
        for _ in 0..5 {
            seq.step(&mut opt);
        }
        // StepLR at internal step 5: 0.1 * 0.5^1 = 0.05
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "decay step 5: expected 0.05, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_sequential_get_lr_reflects_active() {
        let warmup = LinearWarmup::new(0.5, 10);
        let cosine = CosineAnnealingLR::new(0.5, 90, 0.0);

        let mut seq: SequentialLr<f32> =
            SequentialLr::new(vec![(Box::new(warmup), 10), (Box::new(cosine), usize::MAX)]);
        let mut opt = MockOptimizer::new(0.0);

        // Take 1 step in warmup phase.
        seq.step(&mut opt);
        assert!(
            (seq.get_lr() - 0.05).abs() < 1e-12,
            "expected 0.05, got {}",
            seq.get_lr()
        );
    }

    // -----------------------------------------------------------------------
    // cosine_warmup_scheduler tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_cosine_warmup_end_to_end() {
        let base_lr = 0.1;
        let warmup_steps = 10;
        let total_steps = 110;
        let min_lr = 0.001;

        let mut sched: SequentialLr<f32> =
            cosine_warmup_scheduler(base_lr, warmup_steps, total_steps, min_lr);
        let mut opt = MockOptimizer::new(0.0);

        // Warmup: LR should reach base_lr at step warmup_steps.
        for _ in 0..warmup_steps {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - base_lr).abs() < 1e-12,
            "after warmup: expected {base_lr}, got {}",
            opt.lr
        );

        // Cosine decay: run remaining steps.
        let cosine_steps = total_steps - warmup_steps;
        for _ in 0..cosine_steps {
            sched.step(&mut opt);
        }
        // At the end of cosine decay, LR should be at min_lr.
        assert!(
            (opt.lr - min_lr).abs() < 1e-10,
            "after cosine: expected {min_lr}, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_cosine_warmup_midpoint() {
        let base_lr = 1.0;
        let warmup_steps = 10;
        let total_steps = 110;
        let min_lr = 0.0;

        let mut sched: SequentialLr<f32> =
            cosine_warmup_scheduler(base_lr, warmup_steps, total_steps, min_lr);
        let mut opt = MockOptimizer::new(0.0);

        // Run through warmup.
        for _ in 0..warmup_steps {
            sched.step(&mut opt);
        }

        // Run to cosine midpoint (50 of 100 cosine steps).
        let cosine_steps = total_steps - warmup_steps;
        for _ in 0..(cosine_steps / 2) {
            sched.step(&mut opt);
        }
        // At midpoint: lr = 0.5 * (1 + cos(pi/2)) = 0.5 * 1 = 0.5
        let expected = (base_lr + min_lr) / 2.0;
        assert!(
            (opt.lr - expected).abs() < 1e-10,
            "cosine midpoint: expected {expected}, got {}",
            opt.lr
        );
    }

    #[test]
    #[should_panic(expected = "warmup_steps")]
    fn test_cosine_warmup_panics_on_bad_args() {
        let _: SequentialLr<f32> = cosine_warmup_scheduler(0.1, 100, 50, 0.0);
    }

    // -----------------------------------------------------------------------
    // SequentialLr: three-phase schedule
    // -----------------------------------------------------------------------

    #[test]
    fn test_sequential_three_phases() {
        // Phase 1: warmup 0..5
        // Phase 2: constant (via warmup that's already done) 5..10
        // Phase 3: step decay 10..
        let phase1 = LinearWarmup::new(0.1, 5);
        let phase2 = LinearWarmup::new(0.1, 1); // warmup_steps=1, will be base_lr immediately
        let phase3 = StepLR::new(0.1, 3, 0.5);

        let mut seq: SequentialLr<f32> = SequentialLr::new(vec![
            (Box::new(phase1), 5),
            (Box::new(phase2), 10),
            (Box::new(phase3), usize::MAX),
        ]);
        let mut opt = MockOptimizer::new(0.0);

        // Phase 1: linear warmup.
        for _ in 0..5 {
            seq.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "end of phase 1: expected 0.1, got {}",
            opt.lr
        );

        // Phase 2: constant at base_lr.
        for _ in 0..5 {
            seq.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "end of phase 2: expected 0.1, got {}",
            opt.lr
        );

        // Phase 3: StepLR. After 3 internal steps, lr = 0.1 * 0.5 = 0.05.
        for _ in 0..3 {
            seq.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "phase 3 after 3 steps: expected 0.05, got {}",
            opt.lr
        );
    }
}
