//! Constant learning rate scheduler.
//!
//! Multiplies the learning rate by a constant `factor` for `total_iters`
//! steps, then restores it to `base_lr`.
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Constant factor learning rate scheduler.
///
/// During the first `total_iters` steps, `lr = base_lr * factor`.
/// After that, `lr = base_lr`.
///
/// This is useful for a "warm-up at reduced LR" pattern before switching
/// to the full learning rate.
///
/// # Example
///
/// ```ignore
/// // Run at half LR for 40 steps, then full LR.
/// let scheduler = ConstantLR::new(0.1, 0.5, 40);
/// ```
#[derive(Debug, Clone)]
pub struct ConstantLR {
    /// Base (full) learning rate.
    base_lr: f64,
    /// Multiplicative factor applied during the constant phase.
    factor: f64,
    /// Number of steps to apply the factor.
    total_iters: usize,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl ConstantLR {
    /// Create a new `ConstantLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - The full learning rate (restored after `total_iters`).
    /// * `factor` - Multiplicative factor in `[0, 1]`.
    /// * `total_iters` - Number of steps during which `factor` is applied.
    ///
    /// # Panics
    ///
    /// Panics if `factor` is not in `[0, 1]`.
    pub fn new(base_lr: f64, factor: f64, total_iters: usize) -> Self {
        assert!(
            (0.0..=1.0).contains(&factor),
            "factor must be in [0, 1], got {factor}"
        );
        Self {
            base_lr,
            factor,
            total_iters,
            current_step: 0,
            current_lr: base_lr * factor,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Compute the learning rate at the given step.
    fn compute_lr(&self, step: usize) -> f64 {
        if step >= self.total_iters {
            self.base_lr
        } else {
            self.base_lr * self.factor
        }
    }
}

impl<T: Float> LrScheduler<T> for ConstantLR {
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

    #[test]
    fn test_constant_lr_initial() {
        let sched = ConstantLR::new(0.1, 0.5, 10);
        // Initial LR should be base_lr * factor.
        assert!(
            (sched.get_lr() - 0.05).abs() < 1e-12,
            "expected 0.05, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_constant_lr_during_phase() {
        let mut sched = ConstantLR::new(0.1, 0.5, 10);
        let mut opt = MockOptimizer::new(0.05);

        for _ in 0..9 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "expected 0.05, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_constant_lr_restores_after_total_iters() {
        let mut sched = ConstantLR::new(0.1, 0.5, 10);
        let mut opt = MockOptimizer::new(0.05);

        for _ in 0..10 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "expected 0.1 after total_iters, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_constant_lr_stays_at_base_after_total_iters() {
        let mut sched = ConstantLR::new(0.1, 0.5, 5);
        let mut opt = MockOptimizer::new(0.05);

        for _ in 0..20 {
            sched.step(&mut opt);
        }
        assert!((opt.lr - 0.1).abs() < 1e-12, "expected 0.1, got {}", opt.lr);
    }

    #[test]
    fn test_constant_lr_factor_one() {
        // factor=1.0 => no change at any point.
        let mut sched = ConstantLR::new(0.1, 1.0, 10);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..20 {
            sched.step(&mut opt);
        }
        assert!((opt.lr - 0.1).abs() < 1e-12, "expected 0.1, got {}", opt.lr);
    }

    #[test]
    fn test_constant_lr_factor_zero() {
        let mut sched = ConstantLR::new(0.1, 0.0, 5);
        let mut opt = MockOptimizer::new(0.0);

        // During phase: lr = 0.
        for _ in 0..4 {
            sched.step(&mut opt);
        }
        assert!(opt.lr.abs() < 1e-12);

        // After phase: lr = base_lr.
        sched.step(&mut opt);
        assert!((opt.lr - 0.1).abs() < 1e-12, "expected 0.1, got {}", opt.lr);
    }

    #[test]
    #[should_panic(expected = "factor must be in [0, 1]")]
    fn test_constant_lr_invalid_factor() {
        ConstantLR::new(0.1, 1.5, 10);
    }
}
