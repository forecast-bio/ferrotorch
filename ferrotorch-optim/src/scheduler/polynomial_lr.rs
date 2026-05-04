//! Polynomial learning rate scheduler.
//!
//! Decays the learning rate using a polynomial function over `total_iters`
//! steps. After `total_iters`, the learning rate remains at 0 (or the final
//! computed value).
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Polynomial learning rate scheduler.
///
/// # Formula
///
/// ```text
/// lr = base_lr * (1 - min(current_step, total_iters) / total_iters) ^ power
/// ```
///
/// After `total_iters` steps, the learning rate stays at 0 (for power >= 1)
/// or the final value.
///
/// # Example
///
/// ```ignore
/// let scheduler = PolynomialLR::new(0.1, 100, 1.0);
/// // Linear decay over 100 steps: lr = 0.1 * (1 - step/100)
/// ```
#[derive(Debug, Clone)]
pub struct PolynomialLR {
    /// Initial learning rate.
    base_lr: f64,
    /// Total number of decay steps.
    total_iters: usize,
    /// Power of the polynomial.
    power: f64,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl PolynomialLR {
    /// Create a new `PolynomialLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Initial learning rate.
    /// * `total_iters` - Number of steps over which to decay.
    /// * `power` - Power of the polynomial (1.0 = linear decay).
    pub fn new(base_lr: f64, total_iters: usize, power: f64) -> Self {
        Self {
            base_lr,
            total_iters,
            power,
            current_step: 0,
            current_lr: base_lr,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Compute the learning rate at the given step (closed-form).
    fn compute_lr(&self, step: usize) -> f64 {
        let clamped = step.min(self.total_iters);
        if self.total_iters == 0 {
            return 0.0;
        }
        self.base_lr * (1.0 - clamped as f64 / self.total_iters as f64).powf(self.power)
    }
}

impl<T: Float> LrScheduler<T> for PolynomialLR {
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
    fn test_polynomial_initial() {
        let sched = PolynomialLR::new(0.1, 100, 1.0);
        assert!((sched.get_lr() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_polynomial_linear_decay() {
        // power=1.0 => linear decay.
        let base = 1.0;
        let total = 10;
        let mut sched = PolynomialLR::new(base, total, 1.0);
        let mut opt = MockOptimizer::new(base);

        for step in 1..=total {
            sched.step(&mut opt);
            let expected = base * (1.0 - step as f64 / total as f64);
            assert!(
                (opt.lr - expected).abs() < 1e-12,
                "step {step}: expected {expected}, got {}",
                opt.lr
            );
        }
        // At total_iters, lr should be 0.
        assert!(opt.lr.abs() < 1e-12);
    }

    #[test]
    fn test_polynomial_quadratic_decay() {
        let base = 1.0;
        let total = 100;
        let power = 2.0;
        let mut sched = PolynomialLR::new(base, total, power);
        let mut opt = MockOptimizer::new(base);

        // At step 50: lr = 1.0 * (1 - 50/100)^2 = 0.25.
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
    fn test_polynomial_beyond_total_iters() {
        let mut sched = PolynomialLR::new(1.0, 10, 1.0);
        let mut opt = MockOptimizer::new(1.0);

        // Run past total_iters.
        for _ in 0..20 {
            sched.step(&mut opt);
        }
        // Should stay at 0.
        assert!(opt.lr.abs() < 1e-12, "expected ~0.0, got {}", opt.lr);
    }

    #[test]
    fn test_polynomial_fractional_power() {
        let base = 1.0;
        let total = 100;
        let power = 0.5;
        let mut sched = PolynomialLR::new(base, total, power);
        let mut opt = MockOptimizer::new(base);

        for step in 1..=total {
            sched.step(&mut opt);
            let expected = base * (1.0 - step as f64 / total as f64).powf(power);
            assert!(
                (opt.lr - expected).abs() < 1e-10,
                "step {step}: expected {expected}, got {}",
                opt.lr
            );
        }
    }

    #[test]
    fn test_polynomial_midpoint() {
        // At midpoint with power=1: lr = base * 0.5
        let mut sched = PolynomialLR::new(0.1, 100, 1.0);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..50 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "expected 0.05, got {}",
            opt.lr
        );
    }
}
