//! Lambda learning rate scheduler.
//!
//! Sets the learning rate to `base_lr * lr_lambda(epoch)` each step,
//! where `lr_lambda` is a user-provided function.
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Lambda learning rate scheduler.
///
/// At each step, computes `lr = base_lr * lr_lambda(current_step)`.
///
/// # Example
///
/// ```ignore
/// // Decay by 0.95 every epoch.
/// let scheduler = LambdaLR::new(0.1, |epoch| 0.95_f64.powi(epoch as i32));
/// ```
pub struct LambdaLR {
    /// Initial learning rate.
    base_lr: f64,
    /// User-provided lambda: maps epoch/step to a multiplicative factor.
    lr_lambda: Box<dyn Fn(usize) -> f64>,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl LambdaLR {
    /// Create a new `LambdaLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Initial learning rate.
    /// * `lr_lambda` - A function `f(epoch) -> factor` that returns a
    ///   multiplicative factor applied to `base_lr`.
    pub fn new(base_lr: f64, lr_lambda: impl Fn(usize) -> f64 + 'static) -> Self {
        Self {
            base_lr,
            lr_lambda: Box::new(lr_lambda),
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
        self.base_lr * (self.lr_lambda)(step)
    }
}

impl<T: Float> LrScheduler<T> for LambdaLR {
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
    fn test_lambda_lr_constant_factor() {
        // Lambda that always returns 1.0 -> LR should stay at base_lr.
        let mut sched = LambdaLR::new(0.1, |_| 1.0);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..10 {
            sched.step(&mut opt);
        }
        assert!((opt.lr - 0.1).abs() < 1e-12, "expected 0.1, got {}", opt.lr);
    }

    #[test]
    fn test_lambda_lr_exponential_decay() {
        let gamma = 0.95_f64;
        let mut sched = LambdaLR::new(0.1, move |epoch| gamma.powi(epoch as i32));
        let mut opt = MockOptimizer::new(0.1);

        for step in 1..=10 {
            sched.step(&mut opt);
            let expected = 0.1 * gamma.powi(step);
            assert!(
                (opt.lr - expected).abs() < 1e-12,
                "step {step}: expected {expected}, got {}",
                opt.lr
            );
        }
    }

    #[test]
    fn test_lambda_lr_step_function() {
        // Lambda that returns epoch / 30 (integer division).
        let mut sched = LambdaLR::new(0.1, |epoch| (epoch / 30) as f64);
        let mut opt = MockOptimizer::new(0.1);

        // Steps 1..29: epoch/30 = 0, so lr = 0.0
        for _ in 0..29 {
            sched.step(&mut opt);
        }
        assert!(opt.lr.abs() < 1e-12, "expected 0.0, got {}", opt.lr);

        // Step 30: epoch/30 = 1, so lr = 0.1
        sched.step(&mut opt);
        assert!((opt.lr - 0.1).abs() < 1e-12, "expected 0.1, got {}", opt.lr);
    }

    #[test]
    fn test_lambda_lr_get_lr_matches_optimizer() {
        let mut sched = LambdaLR::new(0.5, |epoch| 1.0 / (1.0 + epoch as f64));
        let mut opt = MockOptimizer::new(0.5);

        for _ in 0..5 {
            sched.step(&mut opt);
            assert!(
                (LrScheduler::<f32>::get_lr(&sched) - opt.lr).abs() < 1e-12,
                "get_lr and optimizer LR diverged"
            );
        }
    }
}
