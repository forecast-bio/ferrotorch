//! Step learning rate scheduler.
//!
//! Decays the learning rate by `gamma` every `step_size` epochs.
//! At step `n`, the learning rate is `base_lr * gamma^(n / step_size)`.

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Decays the learning rate by `gamma` every `step_size` steps.
///
/// # Formula
///
/// ```text
/// lr = base_lr * gamma ^ (current_step / step_size)
/// ```
///
/// where `/` is integer (floor) division.
///
/// # Example
///
/// ```ignore
/// let scheduler = StepLR::new(0.1, 30, 0.1);
/// // step  0..29  -> lr = 0.1
/// // step 30..59  -> lr = 0.01
/// // step 60..89  -> lr = 0.001
/// ```
#[derive(Debug, Clone)]
pub struct StepLR {
    /// Initial learning rate.
    base_lr: f64,
    /// Period of learning rate decay (in steps).
    step_size: usize,
    /// Multiplicative factor of learning rate decay.
    gamma: f64,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl StepLR {
    /// Create a new `StepLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Initial learning rate.
    /// * `step_size` - Period of learning rate decay.
    /// * `gamma` - Multiplicative factor of learning rate decay (default in PyTorch: 0.1).
    pub fn new(base_lr: f64, step_size: usize, gamma: f64) -> Self {
        Self {
            base_lr,
            step_size,
            gamma,
            current_step: 0,
            current_lr: base_lr,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Compute the learning rate for the given step.
    fn compute_lr(&self, step: usize) -> f64 {
        let exponent = (step / self.step_size) as f64;
        self.base_lr * self.gamma.powf(exponent)
    }
}

impl<T: Float> LrScheduler<T> for StepLR {
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

    // Minimal mock optimizer for testing schedulers without a full Parameter stack.
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
    fn test_step_lr_initial() {
        let sched = StepLR::new(0.1, 10, 0.1);
        assert!((sched.get_lr() - 0.1).abs() < 1e-12);
    }

    #[test]
    fn test_step_lr_before_first_decay() {
        let mut sched = StepLR::new(0.1, 10, 0.1);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..9 {
            sched.step(&mut opt);
        }
        // Steps 1..9: exponent = step/10 = 0 for all, lr = 0.1
        assert!(
            (sched.get_lr() - 0.1).abs() < 1e-12,
            "expected 0.1, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_step_lr_at_decay_boundary() {
        let mut sched = StepLR::new(0.1, 10, 0.1);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..10 {
            sched.step(&mut opt);
        }
        // Step 10: exponent = 10/10 = 1, lr = 0.1 * 0.1 = 0.01
        assert!(
            (sched.get_lr() - 0.01).abs() < 1e-12,
            "expected 0.01, got {}",
            sched.get_lr()
        );
        assert!((opt.lr - 0.01).abs() < 1e-12);
    }

    #[test]
    fn test_step_lr_multiple_decays() {
        let mut sched = StepLR::new(1.0, 5, 0.5);
        let mut opt = MockOptimizer::new(1.0);

        // Step through 15 steps and check at boundaries.
        for _ in 0..5 {
            sched.step(&mut opt);
        }
        // step=5: lr = 1.0 * 0.5^1 = 0.5
        assert!(
            (sched.get_lr() - 0.5).abs() < 1e-12,
            "step 5: expected 0.5, got {}",
            sched.get_lr()
        );

        for _ in 0..5 {
            sched.step(&mut opt);
        }
        // step=10: lr = 1.0 * 0.5^2 = 0.25
        assert!(
            (sched.get_lr() - 0.25).abs() < 1e-12,
            "step 10: expected 0.25, got {}",
            sched.get_lr()
        );

        for _ in 0..5 {
            sched.step(&mut opt);
        }
        // step=15: lr = 1.0 * 0.5^3 = 0.125
        assert!(
            (sched.get_lr() - 0.125).abs() < 1e-12,
            "step 15: expected 0.125, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_step_lr_optimizer_lr_synced() {
        let mut sched = StepLR::new(0.1, 3, 0.5);
        let mut opt = MockOptimizer::new(0.1);

        for i in 1..=9 {
            sched.step(&mut opt);
            let expected = 0.1 * 0.5_f64.powf((i / 3) as f64);
            assert!(
                (opt.lr - expected).abs() < 1e-12,
                "step {i}: opt.lr = {}, expected {expected}",
                opt.lr
            );
        }
    }
}
