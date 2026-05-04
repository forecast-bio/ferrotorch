//! Cosine annealing learning rate scheduler.
//!
//! Decays the learning rate following a cosine curve from `base_lr` down to
//! `eta_min` over `t_max` steps.

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Cosine annealing learning rate scheduler.
///
/// # Formula
///
/// ```text
/// lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * current_step / T_max))
/// ```
///
/// After `T_max` steps the learning rate stays at `eta_min`.
///
/// # Example
///
/// ```ignore
/// let scheduler = CosineAnnealingLR::new(0.1, 100, 1e-6);
/// // Smoothly decays from 0.1 to 1e-6 over 100 steps.
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingLR {
    /// Initial (maximum) learning rate.
    base_lr: f64,
    /// Maximum number of steps for the cosine schedule.
    t_max: usize,
    /// Minimum learning rate.
    eta_min: f64,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl CosineAnnealingLR {
    /// Create a new cosine annealing scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Initial (maximum) learning rate.
    /// * `t_max` - Number of steps for one cosine half-cycle.
    /// * `eta_min` - Minimum learning rate (default in PyTorch: 0.0).
    pub fn new(base_lr: f64, t_max: usize, eta_min: f64) -> Self {
        Self {
            base_lr,
            t_max,
            eta_min,
            current_step: 0,
            current_lr: base_lr,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Compute the learning rate at the given step.
    fn compute_lr(&self, step: usize) -> f64 {
        if step >= self.t_max {
            return self.eta_min;
        }
        let progress = std::f64::consts::PI * (step as f64) / (self.t_max as f64);
        self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + progress.cos())
    }
}

impl<T: Float> LrScheduler<T> for CosineAnnealingLR {
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
    fn test_cosine_initial_lr() {
        let sched = CosineAnnealingLR::new(0.1, 100, 0.0);
        assert!((sched.get_lr() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_cosine_at_t_max() {
        let mut sched = CosineAnnealingLR::new(0.1, 10, 0.0);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..10 {
            sched.step(&mut opt);
        }
        // At t_max, lr should be eta_min = 0.0.
        assert!(
            sched.get_lr().abs() < 1e-12,
            "expected ~0.0, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_cosine_beyond_t_max() {
        let mut sched = CosineAnnealingLR::new(0.1, 10, 0.001);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..20 {
            sched.step(&mut opt);
        }
        // Beyond t_max, lr stays at eta_min.
        assert!(
            (sched.get_lr() - 0.001).abs() < 1e-12,
            "expected 0.001, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_cosine_midpoint() {
        // At t_max/2, cos(pi/2) = 0, so lr = eta_min + 0.5 * (base - eta_min) * (1 + 0)
        // = eta_min + 0.5 * (base - eta_min) = (base + eta_min) / 2
        let base = 0.1;
        let eta_min = 0.0;
        let t_max = 100;
        let mut sched = CosineAnnealingLR::new(base, t_max, eta_min);
        let mut opt = MockOptimizer::new(base);

        for _ in 0..50 {
            sched.step(&mut opt);
        }
        let expected = (base + eta_min) / 2.0;
        assert!(
            (sched.get_lr() - expected).abs() < 1e-12,
            "midpoint: expected {expected}, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_cosine_analytical_values() {
        let base = 1.0;
        let eta_min = 0.0;
        let t_max = 100;
        let mut sched = CosineAnnealingLR::new(base, t_max, eta_min);
        let mut opt = MockOptimizer::new(base);

        for step in 1..=t_max {
            sched.step(&mut opt);
            let expected = eta_min
                + 0.5
                    * (base - eta_min)
                    * (1.0 + (std::f64::consts::PI * step as f64 / t_max as f64).cos());
            assert!(
                (sched.get_lr() - expected).abs() < 1e-10,
                "step {step}: expected {expected}, got {}",
                sched.get_lr()
            );
        }
    }

    #[test]
    fn test_cosine_with_nonzero_eta_min() {
        let base = 0.1;
        let eta_min = 0.01;
        let t_max = 10;
        let mut sched = CosineAnnealingLR::new(base, t_max, eta_min);
        let mut opt = MockOptimizer::new(base);

        for _ in 0..t_max {
            sched.step(&mut opt);
        }
        assert!(
            (sched.get_lr() - eta_min).abs() < 1e-12,
            "at t_max with eta_min: expected {eta_min}, got {}",
            sched.get_lr()
        );
    }
}
