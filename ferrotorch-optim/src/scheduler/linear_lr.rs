//! Linear learning rate scheduler.
//!
//! Linearly interpolates the multiplicative factor from `start_factor` to
//! `end_factor` over `total_iters` steps. After `total_iters`, the LR
//! stays at `base_lr * end_factor`.
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Linear multiplicative factor scheduler.
///
/// # Formula
///
/// ```text
/// factor = start_factor + (end_factor - start_factor) * min(step, total_iters) / total_iters
/// lr = base_lr * factor
/// ```
///
/// # Example
///
/// ```ignore
/// // Ramp factor from 1/3 to 1.0 over 5 steps.
/// let scheduler = LinearLR::new(0.1, 1.0 / 3.0, 1.0, 5);
/// ```
#[derive(Debug, Clone)]
pub struct LinearLR {
    /// Base learning rate.
    base_lr: f64,
    /// Starting multiplicative factor.
    start_factor: f64,
    /// Ending multiplicative factor.
    end_factor: f64,
    /// Number of iterations over which to interpolate.
    total_iters: usize,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl LinearLR {
    /// Create a new `LinearLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Base learning rate.
    /// * `start_factor` - Starting multiplicative factor (applied at step 0).
    /// * `end_factor` - Ending multiplicative factor (reached at `total_iters`).
    /// * `total_iters` - Number of steps to linearly interpolate.
    ///
    /// # Panics
    ///
    /// Panics if `start_factor` is not in `(0, 1]` or `end_factor` is not in `[0, 1]`.
    pub fn new(base_lr: f64, start_factor: f64, end_factor: f64, total_iters: usize) -> Self {
        assert!(
            start_factor > 0.0 && start_factor <= 1.0,
            "start_factor must be in (0, 1], got {start_factor}"
        );
        assert!(
            (0.0..=1.0).contains(&end_factor),
            "end_factor must be in [0, 1], got {end_factor}"
        );
        let current_lr = base_lr * start_factor;
        Self {
            base_lr,
            start_factor,
            end_factor,
            total_iters,
            current_step: 0,
            current_lr,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Compute the learning rate at the given step (closed-form).
    fn compute_lr(&self, step: usize) -> f64 {
        if self.total_iters == 0 {
            return self.base_lr * self.end_factor;
        }
        let clamped = step.min(self.total_iters);
        let factor = self.start_factor
            + (self.end_factor - self.start_factor) * clamped as f64 / self.total_iters as f64;
        self.base_lr * factor
    }
}

impl<T: Float> LrScheduler<T> for LinearLR {
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
    fn test_linear_lr_initial() {
        let sched = LinearLR::new(0.1, 1.0 / 3.0, 1.0, 5);
        let expected = 0.1 / 3.0;
        assert!(
            (sched.get_lr() - expected).abs() < 1e-12,
            "expected {expected}, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_linear_lr_ramp_to_end() {
        let base = 0.1;
        let start = 0.5;
        let end = 1.0;
        let total = 10;
        let mut sched = LinearLR::new(base, start, end, total);
        let mut opt = MockOptimizer::new(base * start);

        for _ in 0..total {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - base * end).abs() < 1e-12,
            "expected {}, got {}",
            base * end,
            opt.lr
        );
    }

    #[test]
    fn test_linear_lr_analytical() {
        let base = 1.0;
        let start = 0.2;
        let end = 0.8;
        let total = 10;
        let mut sched = LinearLR::new(base, start, end, total);
        let mut opt = MockOptimizer::new(base * start);

        for step in 1..=total {
            sched.step(&mut opt);
            let expected_factor = start + (end - start) * step as f64 / total as f64;
            let expected = base * expected_factor;
            assert!(
                (opt.lr - expected).abs() < 1e-12,
                "step {step}: expected {expected}, got {}",
                opt.lr
            );
        }
    }

    #[test]
    fn test_linear_lr_stays_after_total_iters() {
        let base = 0.1;
        let start = 0.1;
        let end = 1.0;
        let total = 5;
        let mut sched = LinearLR::new(base, start, end, total);
        let mut opt = MockOptimizer::new(base * start);

        for _ in 0..20 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - base * end).abs() < 1e-12,
            "expected {}, got {}",
            base * end,
            opt.lr
        );
    }

    #[test]
    fn test_linear_lr_decreasing_factor() {
        // Factor goes from 1.0 to 0.1 -> LR decreases.
        let base = 1.0;
        let start = 1.0;
        let end = 0.1;
        let total = 10;
        let mut sched = LinearLR::new(base, start, end, total);
        let mut opt = MockOptimizer::new(base);

        for _ in 0..total {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - base * end).abs() < 1e-12,
            "expected {}, got {}",
            base * end,
            opt.lr
        );
    }

    #[test]
    fn test_linear_lr_midpoint() {
        let base = 1.0;
        let start = 0.0 + 1e-10; // Near zero but valid.
        let end = 1.0;
        let total = 10;
        let mut sched = LinearLR::new(base, start, end, total);
        let mut opt = MockOptimizer::new(base * start);

        for _ in 0..5 {
            sched.step(&mut opt);
        }
        let expected_factor = start + (end - start) * 0.5;
        let expected = base * expected_factor;
        assert!(
            (opt.lr - expected).abs() < 1e-10,
            "midpoint: expected {expected}, got {}",
            opt.lr
        );
    }

    #[test]
    #[should_panic(expected = "start_factor must be in (0, 1]")]
    fn test_linear_lr_invalid_start_factor() {
        LinearLR::new(0.1, 0.0, 1.0, 5);
    }
}
