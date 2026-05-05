//! Linear warmup learning rate scheduler.
//!
//! Linearly ramps the learning rate from 0 to `base_lr` over
//! `warmup_steps` steps. After the warmup phase, the LR stays at `base_lr`.

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Linear warmup scheduler.
///
/// # Formula
///
/// ```text
/// lr = base_lr * min(1.0, current_step / warmup_steps)
/// ```
///
/// # Example
///
/// ```ignore
/// let scheduler = LinearWarmup::new(0.1, 1000);
/// // step   0  -> lr = 0.0
/// // step 500  -> lr = 0.05
/// // step 1000 -> lr = 0.1
/// // step 2000 -> lr = 0.1  (clamped)
/// ```
#[derive(Debug, Clone)]
pub struct LinearWarmup {
    /// Target learning rate after warmup.
    base_lr: f64,
    /// Number of warmup steps.
    warmup_steps: usize,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl LinearWarmup {
    /// Create a new linear warmup scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Target learning rate at the end of warmup.
    /// * `warmup_steps` - Number of steps over which to linearly ramp up.
    pub fn new(base_lr: f64, warmup_steps: usize) -> Self {
        Self {
            base_lr,
            warmup_steps,
            current_step: 0,
            current_lr: 0.0,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Compute the learning rate at the given step.
    fn compute_lr(&self, step: usize) -> f64 {
        if self.warmup_steps == 0 {
            return self.base_lr;
        }
        let ratio = (step as f64 / self.warmup_steps as f64).min(1.0);
        self.base_lr * ratio
    }
}

impl<T: Float> LrScheduler<T> for LinearWarmup {
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
    fn test_warmup_initial_lr_is_zero() {
        let sched = LinearWarmup::new(0.1, 100);
        assert!((sched.get_lr() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_warmup_linear_ramp() {
        let base_lr = 0.1;
        let warmup_steps = 10;
        let mut sched = LinearWarmup::new(base_lr, warmup_steps);
        let mut opt = MockOptimizer::new(0.0);

        for step in 1..=warmup_steps {
            sched.step(&mut opt);
            let expected = base_lr * (step as f64 / warmup_steps as f64);
            assert!(
                (sched.get_lr() - expected).abs() < 1e-12,
                "step {step}: expected {expected}, got {}",
                sched.get_lr()
            );
            assert!(
                (opt.lr - expected).abs() < 1e-12,
                "step {step}: opt.lr expected {expected}, got {}",
                opt.lr
            );
        }
    }

    #[test]
    fn test_warmup_reaches_base_lr() {
        let mut sched = LinearWarmup::new(0.5, 100);
        let mut opt = MockOptimizer::new(0.0);

        for _ in 0..100 {
            sched.step(&mut opt);
        }
        assert!(
            (sched.get_lr() - 0.5).abs() < 1e-12,
            "expected 0.5, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_warmup_stays_at_base_lr_after_completion() {
        let mut sched = LinearWarmup::new(0.1, 10);
        let mut opt = MockOptimizer::new(0.0);

        for _ in 0..20 {
            sched.step(&mut opt);
        }
        // After warmup_steps, LR should be clamped at base_lr.
        assert!(
            (sched.get_lr() - 0.1).abs() < 1e-12,
            "expected 0.1, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_warmup_zero_steps() {
        let mut sched = LinearWarmup::new(0.1, 0);
        let mut opt = MockOptimizer::new(0.0);

        sched.step(&mut opt);
        // With zero warmup steps, should immediately be at base_lr.
        assert!(
            (sched.get_lr() - 0.1).abs() < 1e-12,
            "expected 0.1, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_warmup_halfway() {
        let mut sched = LinearWarmup::new(1.0, 100);
        let mut opt = MockOptimizer::new(0.0);

        for _ in 0..50 {
            sched.step(&mut opt);
        }
        assert!(
            (sched.get_lr() - 0.5).abs() < 1e-12,
            "expected 0.5, got {}",
            sched.get_lr()
        );
    }
}
