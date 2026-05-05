//! Exponential learning rate scheduler.
//!
//! Decays the learning rate by `gamma` every step.
//! At step `n`, `lr = base_lr * gamma^n`.
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Decays the learning rate by `gamma` every step.
///
/// # Formula
///
/// ```text
/// lr = base_lr * gamma ^ current_step
/// ```
///
/// # Example
///
/// ```ignore
/// let scheduler = ExponentialLR::new(0.1, 0.95);
/// // step 0  -> lr = 0.1
/// // step 1  -> lr = 0.095
/// // step 10 -> lr = 0.1 * 0.95^10 ≈ 0.0598
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialLR {
    /// Initial learning rate.
    base_lr: f64,
    /// Multiplicative factor of learning rate decay per step.
    gamma: f64,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl ExponentialLR {
    /// Create a new `ExponentialLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Initial learning rate.
    /// * `gamma` - Multiplicative factor applied every step.
    pub fn new(base_lr: f64, gamma: f64) -> Self {
        Self {
            base_lr,
            gamma,
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
        self.base_lr * self.gamma.powi(step as i32)
    }
}

impl<T: Float> LrScheduler<T> for ExponentialLR {
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
    fn test_exponential_initial() {
        let sched = ExponentialLR::new(0.1, 0.95);
        assert!((sched.get_lr() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_exponential_one_step() {
        let mut sched = ExponentialLR::new(0.1, 0.95);
        let mut opt = MockOptimizer::new(0.1);
        sched.step(&mut opt);
        assert!(
            (opt.lr - 0.095).abs() < 1e-12,
            "expected 0.095, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_exponential_analytical() {
        let base = 1.0;
        let gamma = 0.9;
        let mut sched = ExponentialLR::new(base, gamma);
        let mut opt = MockOptimizer::new(base);

        for step in 1..=20 {
            sched.step(&mut opt);
            let expected = base * gamma.powi(step);
            assert!(
                (opt.lr - expected).abs() < 1e-10,
                "step {step}: expected {expected}, got {}",
                opt.lr
            );
        }
    }

    #[test]
    fn test_exponential_gamma_one() {
        // gamma=1.0 => no decay.
        let mut sched = ExponentialLR::new(0.5, 1.0);
        let mut opt = MockOptimizer::new(0.5);

        for _ in 0..50 {
            sched.step(&mut opt);
        }
        assert!((opt.lr - 0.5).abs() < 1e-12, "expected 0.5, got {}", opt.lr);
    }

    #[test]
    fn test_exponential_rapid_decay() {
        let mut sched = ExponentialLR::new(1.0, 0.1);
        let mut opt = MockOptimizer::new(1.0);

        sched.step(&mut opt);
        assert!((opt.lr - 0.1).abs() < 1e-12);

        sched.step(&mut opt);
        assert!((opt.lr - 0.01).abs() < 1e-12);

        sched.step(&mut opt);
        assert!((opt.lr - 0.001).abs() < 1e-10);
    }
}
