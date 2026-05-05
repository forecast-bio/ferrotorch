//! Multi-step learning rate scheduler.
//!
//! Decays the learning rate by `gamma` at each milestone step.
//! At step `n`, `lr = base_lr * gamma^(number of milestones <= n)`.
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Decays the learning rate by `gamma` at specified milestone steps.
///
/// # Formula
///
/// ```text
/// lr = base_lr * gamma ^ (count of milestones <= current_step)
/// ```
///
/// # Example
///
/// ```ignore
/// // lr = 0.05     if step < 30
/// // lr = 0.005    if 30 <= step < 80
/// // lr = 0.0005   if step >= 80
/// let scheduler = MultiStepLR::new(0.05, vec![30, 80], 0.1);
/// ```
#[derive(Debug, Clone)]
pub struct MultiStepLR {
    /// Initial learning rate.
    base_lr: f64,
    /// Sorted list of milestone steps at which LR is multiplied by gamma.
    milestones: Vec<usize>,
    /// Multiplicative factor of learning rate decay.
    gamma: f64,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl MultiStepLR {
    /// Create a new `MultiStepLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Initial learning rate.
    /// * `milestones` - List of step indices at which LR is decayed.
    ///   Will be sorted internally.
    /// * `gamma` - Multiplicative factor of learning rate decay.
    pub fn new(base_lr: f64, mut milestones: Vec<usize>, gamma: f64) -> Self {
        milestones.sort_unstable();
        Self {
            base_lr,
            milestones,
            gamma,
            current_step: 0,
            current_lr: base_lr,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Compute the learning rate at the given step using bisect.
    fn compute_lr(&self, step: usize) -> f64 {
        // Count how many milestones have been passed (milestone <= step).
        let count = self.milestones.partition_point(|&m| m <= step);
        self.base_lr * self.gamma.powi(count as i32)
    }
}

impl<T: Float> LrScheduler<T> for MultiStepLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
        self.current_step += 1;
        self.current_lr = self.compute_lr(self.current_step);
        optimizer.set_lr(self.current_lr);
    }

    fn get_lr(&self) -> f64 {
        self.current_lr
    }
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
        fn state_dict(
            &self,
        ) -> ferrotorch_core::FerrotorchResult<crate::optimizer::OptimizerState> {
            Ok(Default::default())
        }
        fn load_state_dict(
            &mut self,
            _state: &crate::optimizer::OptimizerState,
        ) -> ferrotorch_core::FerrotorchResult<()> {
            Ok(())
        }
    }

    #[test]
    fn test_multi_step_before_first_milestone() {
        let mut sched = MultiStepLR::new(0.05, vec![30, 80], 0.1);
        let mut opt = MockOptimizer::new(0.05);

        for _ in 0..29 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "expected 0.05, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_multi_step_at_first_milestone() {
        let mut sched = MultiStepLR::new(0.05, vec![30, 80], 0.1);
        let mut opt = MockOptimizer::new(0.05);

        for _ in 0..30 {
            sched.step(&mut opt);
        }
        // At step 30: gamma^1 = 0.1, lr = 0.005
        assert!(
            (opt.lr - 0.005).abs() < 1e-12,
            "expected 0.005, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_multi_step_between_milestones() {
        let mut sched = MultiStepLR::new(0.05, vec![30, 80], 0.1);
        let mut opt = MockOptimizer::new(0.05);

        for _ in 0..50 {
            sched.step(&mut opt);
        }
        // Between milestones 30 and 80: gamma^1 = 0.1, lr = 0.005
        assert!(
            (opt.lr - 0.005).abs() < 1e-12,
            "expected 0.005, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_multi_step_at_second_milestone() {
        let mut sched = MultiStepLR::new(0.05, vec![30, 80], 0.1);
        let mut opt = MockOptimizer::new(0.05);

        for _ in 0..80 {
            sched.step(&mut opt);
        }
        // At step 80: gamma^2 = 0.01, lr = 0.0005
        assert!(
            (opt.lr - 0.0005).abs() < 1e-12,
            "expected 0.0005, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_multi_step_past_all_milestones() {
        let mut sched = MultiStepLR::new(0.05, vec![30, 80], 0.1);
        let mut opt = MockOptimizer::new(0.05);

        for _ in 0..100 {
            sched.step(&mut opt);
        }
        // Past all milestones: gamma^2 = 0.01, lr = 0.0005
        assert!(
            (opt.lr - 0.0005).abs() < 1e-12,
            "expected 0.0005, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_multi_step_unsorted_milestones() {
        // Milestones given out of order should be sorted internally.
        let mut sched = MultiStepLR::new(1.0, vec![80, 30], 0.5);
        let mut opt = MockOptimizer::new(1.0);

        for _ in 0..30 {
            sched.step(&mut opt);
        }
        assert!((opt.lr - 0.5).abs() < 1e-12, "expected 0.5, got {}", opt.lr);

        for _ in 0..50 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.25).abs() < 1e-12,
            "expected 0.25, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_multi_step_single_milestone() {
        let mut sched = MultiStepLR::new(0.1, vec![10], 0.5);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..9 {
            sched.step(&mut opt);
        }
        assert!((opt.lr - 0.1).abs() < 1e-12);

        sched.step(&mut opt);
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "expected 0.05, got {}",
            opt.lr
        );
    }
}
