//! Multiplicative learning rate scheduler.
//!
//! Multiplies the current learning rate by a user-provided function each step.
//! Unlike [`LambdaLR`](super::LambdaLR), which computes `base_lr * f(epoch)`,
//! this scheduler computes `lr *= f(epoch)` — the factor is applied
//! cumulatively to whatever the current LR is.
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Multiplicative learning rate scheduler.
///
/// At each step, computes `lr = lr * lr_lambda(current_step)`.
///
/// # Difference from `LambdaLR`
///
/// - `LambdaLR`: `lr = base_lr * f(step)` (absolute — recomputed from base each step)
/// - `MultiplicativeLR`: `lr = lr * f(step)` (relative — compounds on previous LR)
///
/// # Example
///
/// ```ignore
/// // Decay by 5% every epoch (compound).
/// let scheduler = MultiplicativeLR::new(0.1, |_epoch| 0.95);
/// // step 1: 0.1 * 0.95 = 0.095
/// // step 2: 0.095 * 0.95 = 0.09025
/// // step 3: 0.09025 * 0.95 = 0.0857375
/// ```
pub struct MultiplicativeLR {
    /// User-provided lambda: maps epoch/step to a multiplicative factor.
    lr_lambda: Box<dyn Fn(usize) -> f64>,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl MultiplicativeLR {
    /// Create a new `MultiplicativeLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Initial learning rate.
    /// * `lr_lambda` - A function `f(epoch) -> factor` that returns a
    ///   multiplicative factor applied to the *current* LR each step.
    pub fn new(base_lr: f64, lr_lambda: impl Fn(usize) -> f64 + 'static) -> Self {
        Self {
            lr_lambda: Box::new(lr_lambda),
            current_step: 0,
            current_lr: base_lr,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }
}

impl<T: Float> LrScheduler<T> for MultiplicativeLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
        self.current_step += 1;
        let factor = (self.lr_lambda)(self.current_step);
        self.current_lr *= factor;
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
    fn test_multiplicative_constant_factor() {
        // Constant factor 0.95 -> compounds each step.
        let base = 0.1;
        let gamma = 0.95;
        let mut sched = MultiplicativeLR::new(base, move |_| gamma);
        let mut opt = MockOptimizer::new(base);

        let mut expected = base;
        for step in 1..=10 {
            sched.step(&mut opt);
            expected *= gamma;
            assert!(
                (opt.lr - expected).abs() < 1e-12,
                "step {step}: expected {expected}, got {}",
                opt.lr
            );
        }
    }

    #[test]
    fn test_multiplicative_factor_one() {
        // Factor 1.0 -> LR unchanged.
        let mut sched = MultiplicativeLR::new(0.1, |_| 1.0);
        let mut opt = MockOptimizer::new(0.1);

        for _ in 0..20 {
            sched.step(&mut opt);
        }
        assert!((opt.lr - 0.1).abs() < 1e-12, "expected 0.1, got {}", opt.lr);
    }

    #[test]
    fn test_multiplicative_epoch_dependent() {
        // Factor depends on epoch: f(epoch) = 1.0 / epoch.
        // step 1: 0.1 * (1/1) = 0.1
        // step 2: 0.1 * (1/2) = 0.05
        // step 3: 0.05 * (1/3) = 0.01666...
        let mut sched = MultiplicativeLR::new(0.1, |epoch| 1.0 / epoch as f64);
        let mut opt = MockOptimizer::new(0.1);

        sched.step(&mut opt);
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "step 1: expected 0.1, got {}",
            opt.lr
        );

        sched.step(&mut opt);
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "step 2: expected 0.05, got {}",
            opt.lr
        );

        sched.step(&mut opt);
        let expected = 0.05 / 3.0;
        assert!(
            (opt.lr - expected).abs() < 1e-12,
            "step 3: expected {expected}, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_multiplicative_get_lr_matches_optimizer() {
        let mut sched = MultiplicativeLR::new(0.5, |_| 0.9);
        let mut opt = MockOptimizer::new(0.5);

        for _ in 0..5 {
            sched.step(&mut opt);
            assert!(
                (LrScheduler::<f32>::get_lr(&sched) - opt.lr).abs() < 1e-12,
                "get_lr and optimizer LR diverged"
            );
        }
    }

    #[test]
    fn test_multiplicative_vs_exponential() {
        // MultiplicativeLR with constant factor gamma should produce same
        // result as ExponentialLR: base * gamma^n.
        let base = 1.0;
        let gamma = 0.9;
        let mut sched = MultiplicativeLR::new(base, move |_| gamma);
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
    fn test_multiplicative_halving() {
        // Halve the LR at epochs 5 and 10.
        let mut sched =
            MultiplicativeLR::new(
                1.0,
                |epoch| {
                    if epoch == 5 || epoch == 10 { 0.5 } else { 1.0 }
                },
            );
        let mut opt = MockOptimizer::new(1.0);

        for _ in 0..4 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 1.0).abs() < 1e-12,
            "before epoch 5: expected 1.0, got {}",
            opt.lr
        );

        sched.step(&mut opt); // epoch 5
        assert!(
            (opt.lr - 0.5).abs() < 1e-12,
            "at epoch 5: expected 0.5, got {}",
            opt.lr
        );

        for _ in 0..4 {
            sched.step(&mut opt); // epochs 6..9
        }
        assert!(
            (opt.lr - 0.5).abs() < 1e-12,
            "before epoch 10: expected 0.5, got {}",
            opt.lr
        );

        sched.step(&mut opt); // epoch 10
        assert!(
            (opt.lr - 0.25).abs() < 1e-12,
            "at epoch 10: expected 0.25, got {}",
            opt.lr
        );
    }
}
