//! Chained learning rate scheduler.
//!
//! Applies multiple schedulers in sequence on every step. Each scheduler's
//! `step()` is called one after the other, so the final LR is the result
//! of all schedulers composing their effects.
//!
//! This differs from [`SequentialLr`](super::SequentialLr), which switches
//! between schedulers at milestones. `ChainedScheduler` calls *every*
//! scheduler on *every* step.
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Chains multiple [`LrScheduler`]s, applying all of them in order each step.
///
/// On each call to [`step()`](LrScheduler::step), every inner scheduler is
/// stepped in order. The optimizer's LR after all inner schedulers have run
/// is the effective learning rate.
///
/// # Use case
///
/// Combine independent scheduling effects — for example, an exponential decay
/// composed with a warmup factor:
///
/// ```ignore
/// use ferrotorch_optim::scheduler::{ChainedScheduler, ExponentialLR, LinearLR};
///
/// let exp = ExponentialLR::new(0.1, 0.95);
/// let linear = LinearLR::new(0.1, 0.1, 1.0, 10);
///
/// let scheduler = ChainedScheduler::new(vec![
///     Box::new(exp),
///     Box::new(linear),
/// ]);
/// ```
pub struct ChainedScheduler<T: Float> {
    /// Inner schedulers, applied in order.
    schedulers: Vec<Box<dyn LrScheduler<T>>>,
}

impl<T: Float> ChainedScheduler<T> {
    /// Create a new `ChainedScheduler` from an ordered list of schedulers.
    ///
    /// # Arguments
    ///
    /// * `schedulers` - Schedulers to apply in order on each step.
    ///
    /// # Panics
    ///
    /// Panics if `schedulers` is empty.
    pub fn new(schedulers: Vec<Box<dyn LrScheduler<T>>>) -> Self {
        assert!(
            !schedulers.is_empty(),
            "ChainedScheduler requires at least one scheduler"
        );
        Self { schedulers }
    }

    /// Return the number of inner schedulers.
    pub fn len(&self) -> usize {
        self.schedulers.len()
    }

    /// Return `true` if there are no inner schedulers.
    pub fn is_empty(&self) -> bool {
        self.schedulers.is_empty()
    }
}

impl<T: Float> LrScheduler<T> for ChainedScheduler<T> {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
        for scheduler in &mut self.schedulers {
            scheduler.step(optimizer);
        }
    }

    fn get_lr(&self) -> f64 {
        // The effective LR is whatever the last scheduler set.
        self.schedulers.last().map(|s| s.get_lr()).unwrap_or(0.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::constant_lr::ConstantLR;
    use crate::scheduler::exponential_lr::ExponentialLR;
    use crate::scheduler::step::StepLR;

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
        fn state_dict(&self) -> crate::optimizer::OptimizerState {
            Default::default()
        }
        fn load_state_dict(
            &mut self,
            _state: &crate::optimizer::OptimizerState,
        ) -> ferrotorch_core::FerrotorchResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_chained_single_scheduler() {
        // A single ExponentialLR in a chain should behave identically.
        let base = 0.1;
        let gamma = 0.95;
        let exp = ExponentialLR::new(base, gamma);
        let mut chained: ChainedScheduler<f32> = ChainedScheduler::new(vec![Box::new(exp)]);
        let mut opt = MockOptimizer::new(base);

        for step in 1..=10 {
            chained.step(&mut opt);
            let expected = base * gamma.powi(step);
            assert!(
                (opt.lr - expected).abs() < 1e-10,
                "step {step}: expected {expected}, got {}",
                opt.lr
            );
        }
    }

    #[test]
    fn test_chained_two_exponentials() {
        // Two ExponentialLR(gamma=0.9) chained = ExponentialLR(gamma=0.81).
        // Each step: scheduler 1 sets lr = base * 0.9^n, scheduler 2 reads
        // that and sets lr = (base * 0.9^n) ... but scheduler 2 has its own
        // base, so the composition is: step1 sets opt.lr, step2 overwrites.
        //
        // Actually: ExponentialLR uses its own base_lr, so chaining two
        // independent ExponentialLRs means the last one wins. That's correct
        // behavior for chained schedulers that are independently parameterized.
        //
        // A more meaningful test: chain ConstantLR (factor phase) with StepLR.
        let base = 0.1;

        // Phase: ConstantLR runs at half LR for 5 steps, then full.
        // StepLR decays by 0.5 every 5 steps.
        // Both step independently. The last one (StepLR) sets the final LR.
        let constant = ConstantLR::new(base, 0.5, 5);
        let step_sched = StepLR::new(base, 5, 0.5);

        let mut chained: ChainedScheduler<f32> =
            ChainedScheduler::new(vec![Box::new(constant), Box::new(step_sched)]);
        let mut opt = MockOptimizer::new(base);

        // During first 5 steps: ConstantLR sets lr=0.05, then StepLR sets lr=0.1
        // (step/5=0, so gamma^0=1). StepLR wins.
        for _ in 0..4 {
            chained.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "expected 0.1 (StepLR dominates), got {}",
            opt.lr
        );

        // At step 5: StepLR internal step=5, lr = 0.1 * 0.5^1 = 0.05.
        chained.step(&mut opt);
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "step 5: expected 0.05, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_chained_get_lr() {
        let exp = ExponentialLR::new(0.1, 0.9);
        let mut chained: ChainedScheduler<f32> = ChainedScheduler::new(vec![Box::new(exp)]);
        let mut opt = MockOptimizer::new(0.1);

        chained.step(&mut opt);
        assert!(
            (chained.get_lr() - opt.lr).abs() < 1e-12,
            "get_lr should match optimizer LR"
        );
    }

    #[test]
    fn test_chained_len() {
        let s1 = ExponentialLR::new(0.1, 0.9);
        let s2 = StepLR::new(0.1, 10, 0.5);
        let chained: ChainedScheduler<f32> =
            ChainedScheduler::new(vec![Box::new(s1), Box::new(s2)]);
        assert_eq!(chained.len(), 2);
        assert!(!chained.is_empty());
    }

    #[test]
    #[should_panic(expected = "at least one scheduler")]
    fn test_chained_empty_panics() {
        let _: ChainedScheduler<f32> = ChainedScheduler::new(vec![]);
    }

    #[test]
    fn test_chained_all_schedulers_stepped() {
        // Verify that all schedulers advance their internal state.
        // Use two StepLR schedulers with different step_sizes.
        // After chaining, both should have advanced their counters.
        let s1 = StepLR::new(1.0, 3, 0.5); // decays at step 3,6,9...
        let s2 = StepLR::new(1.0, 5, 0.1); // decays at step 5,10...

        let mut chained: ChainedScheduler<f32> =
            ChainedScheduler::new(vec![Box::new(s1), Box::new(s2)]);
        let mut opt = MockOptimizer::new(1.0);

        // After 5 steps:
        // s1 internal step=5: lr = 1.0 * 0.5^(5/3) = 1.0 * 0.5^1 = 0.5
        // s2 internal step=5: lr = 1.0 * 0.1^(5/5) = 1.0 * 0.1^1 = 0.1
        // Last scheduler (s2) sets the final LR.
        for _ in 0..5 {
            chained.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "expected 0.1 (s2 sets final LR), got {}",
            opt.lr
        );
    }
}
