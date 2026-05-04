//! Cosine annealing with warm restarts (SGDR) scheduler.
//!
//! Implements the SGDR schedule from "Stochastic Gradient Descent with Warm
//! Restarts" (Loshchilov & Hutter, 2016). The learning rate follows a cosine
//! curve from `base_lr` to `eta_min`, then snaps back to `base_lr` and starts
//! a new cycle. Each successive cycle can be longer by a factor of `t_mult`.
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Cosine annealing with warm restarts.
///
/// # Formula
///
/// ```text
/// lr = eta_min + 0.5 * (base_lr - eta_min) * (1 + cos(pi * T_cur / T_i))
/// ```
///
/// where `T_cur` is the number of steps since the last restart, and `T_i`
/// is the length of the current cycle. After each cycle completes,
/// `T_i = T_i * T_mult`.
///
/// # Example
///
/// ```ignore
/// let scheduler = CosineAnnealingWarmRestarts::new(0.1, 20, 1, 0.0);
/// // Cycles every 20 steps, restarting at base_lr each time.
/// ```
#[derive(Debug, Clone)]
pub struct CosineAnnealingWarmRestarts {
    /// Initial (maximum) learning rate.
    base_lr: f64,
    /// Number of steps in the first cycle (retained for state introspection).
    #[allow(dead_code)]
    t_0: usize,
    /// Factor by which T_i grows after each restart.
    t_mult: usize,
    /// Minimum learning rate.
    eta_min: f64,
    /// Current position within the active cycle.
    t_cur: usize,
    /// Length of the current cycle.
    t_i: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl CosineAnnealingWarmRestarts {
    /// Create a new cosine annealing warm restarts scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Initial (maximum) learning rate.
    /// * `t_0` - Number of steps in the first cycle.
    /// * `t_mult` - Cycle length multiplier (>= 1). After each restart, the
    ///   cycle length becomes `T_i * t_mult`.
    /// * `eta_min` - Minimum learning rate.
    ///
    /// # Panics
    ///
    /// Panics if `t_0 == 0` or `t_mult == 0`.
    pub fn new(base_lr: f64, t_0: usize, t_mult: usize, eta_min: f64) -> Self {
        assert!(t_0 > 0, "t_0 must be positive, got {t_0}");
        assert!(t_mult >= 1, "t_mult must be >= 1, got {t_mult}");
        Self {
            base_lr,
            t_0,
            t_mult,
            eta_min,
            t_cur: 0,
            t_i: t_0,
            current_lr: base_lr,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Compute the learning rate from `t_cur` and `t_i`.
    fn compute_lr(&self) -> f64 {
        let progress = std::f64::consts::PI * (self.t_cur as f64) / (self.t_i as f64);
        self.eta_min + 0.5 * (self.base_lr - self.eta_min) * (1.0 + progress.cos())
    }
}

impl<T: Float> LrScheduler<T> for CosineAnnealingWarmRestarts {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
        self.t_cur += 1;
        if self.t_cur >= self.t_i {
            // Restart: reset t_cur, grow cycle length.
            self.t_cur = 0;
            self.t_i *= self.t_mult;
        }
        self.current_lr = self.compute_lr();
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
    fn test_warm_restarts_first_cycle_end() {
        // T_0=10, t_mult=1, eta_min=0.0
        // After 10 steps we should be at eta_min, then restart.
        let mut sched = CosineAnnealingWarmRestarts::new(0.1, 10, 1, 0.0);
        let mut opt = MockOptimizer::new(0.1);

        // Step to the last step of the first cycle (step 9, t_cur=9 before restart).
        for _ in 0..9 {
            sched.step(&mut opt);
        }
        // t_cur=9 of t_i=10: close to eta_min but not quite.
        let expected_9 = 0.5 * 0.1 * (1.0 + (std::f64::consts::PI * 9.0 / 10.0).cos());
        assert!(
            (opt.lr - expected_9).abs() < 1e-10,
            "step 9: expected {expected_9}, got {}",
            opt.lr
        );

        // Step 10: triggers restart (t_cur goes 10 -> 0), then compute.
        sched.step(&mut opt);
        // After restart, t_cur=0, lr = base_lr.
        assert!(
            (opt.lr - 0.1).abs() < 1e-10,
            "after restart: expected 0.1, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_warm_restarts_t_mult_2() {
        // T_0=5, t_mult=2, eta_min=0.0
        // Cycle 1: 5 steps, cycle 2: 10 steps, cycle 3: 20 steps.
        let mut sched = CosineAnnealingWarmRestarts::new(1.0, 5, 2, 0.0);
        let mut opt = MockOptimizer::new(1.0);

        // Run through cycle 1 (5 steps).
        for _ in 0..5 {
            sched.step(&mut opt);
        }
        // Should have restarted: t_cur=0, lr=base_lr.
        assert!(
            (opt.lr - 1.0).abs() < 1e-10,
            "after cycle 1 restart: expected 1.0, got {}",
            opt.lr
        );

        // Run through cycle 2 (10 steps).
        for _ in 0..10 {
            sched.step(&mut opt);
        }
        // Should have restarted again.
        assert!(
            (opt.lr - 1.0).abs() < 1e-10,
            "after cycle 2 restart: expected 1.0, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_warm_restarts_midpoint() {
        // At midpoint of a cycle, lr should be (base + eta_min) / 2.
        let base = 1.0;
        let eta_min = 0.0;
        let t_0 = 20;
        let mut sched = CosineAnnealingWarmRestarts::new(base, t_0, 1, eta_min);
        let mut opt = MockOptimizer::new(base);

        for _ in 0..10 {
            sched.step(&mut opt);
        }
        let expected = (base + eta_min) / 2.0;
        assert!(
            (opt.lr - expected).abs() < 1e-10,
            "midpoint: expected {expected}, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_warm_restarts_with_eta_min() {
        let base = 0.1;
        let eta_min = 0.01;
        let t_0 = 10;
        let mut sched = CosineAnnealingWarmRestarts::new(base, t_0, 1, eta_min);
        let mut opt = MockOptimizer::new(base);

        // Run to just before restart (step 9: t_cur=9).
        for _ in 0..9 {
            sched.step(&mut opt);
        }
        let expected =
            eta_min + 0.5 * (base - eta_min) * (1.0 + (std::f64::consts::PI * 9.0 / 10.0).cos());
        assert!(
            (opt.lr - expected).abs() < 1e-10,
            "near end: expected {expected}, got {}",
            opt.lr
        );
        // LR should be close to eta_min but not below.
        assert!(opt.lr >= eta_min - 1e-12);
    }

    #[test]
    fn test_warm_restarts_multiple_cycles_analytical() {
        let base = 0.5;
        let eta_min = 0.0;
        let t_0 = 4;
        let mut sched = CosineAnnealingWarmRestarts::new(base, t_0, 1, eta_min);
        let mut opt = MockOptimizer::new(base);

        // Two full cycles: 8 steps.
        let mut lrs = Vec::new();
        for _ in 0..8 {
            sched.step(&mut opt);
            lrs.push(opt.lr);
        }

        // The pattern should repeat every t_0=4 steps.
        // Steps 0..3 of each cycle are at t_cur=1,2,3,0(restart).
        for i in 0..4 {
            assert!(
                (lrs[i] - lrs[i + 4]).abs() < 1e-10,
                "cycle pattern mismatch at offset {i}: {} vs {}",
                lrs[i],
                lrs[i + 4]
            );
        }
    }

    #[test]
    #[should_panic(expected = "t_0 must be positive")]
    fn test_warm_restarts_zero_t0_panics() {
        CosineAnnealingWarmRestarts::new(0.1, 0, 1, 0.0);
    }
}
