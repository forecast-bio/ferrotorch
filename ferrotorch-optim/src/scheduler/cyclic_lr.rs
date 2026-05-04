//! Cyclic learning rate scheduler.
//!
//! Cycles the learning rate between `base_lr` and `max_lr` with a triangular
//! wave. Three built-in policies are provided: `triangular`, `triangular2`,
//! and `exp_range`.
//!
//! Reference: "Cyclical Learning Rates for Training Neural Networks"
//! (Smith, 2017).
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Policy for the cyclic learning rate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CyclicMode {
    /// Basic triangular cycle without amplitude scaling.
    Triangular,
    /// Triangular cycle that halves the amplitude each cycle.
    Triangular2,
    /// Cycle that scales amplitude by `gamma^(iteration)` each iteration.
    ExpRange,
}

/// Cyclic learning rate scheduler.
///
/// Cycles the learning rate between `base_lr` and `max_lr` using a
/// triangular wave pattern. The amplitude can be scaled per-cycle or
/// per-iteration depending on the mode.
///
/// # Policies
///
/// - **Triangular**: constant amplitude `(max_lr - base_lr)`.
/// - **Triangular2**: amplitude is halved each complete cycle.
/// - **ExpRange**: amplitude scaled by `gamma^iteration`.
///
/// # Example
///
/// ```ignore
/// let scheduler = CyclicLR::new(0.001, 0.01, 2000, None, CyclicMode::Triangular, 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct CyclicLR {
    /// Lower boundary learning rate.
    base_lr: f64,
    /// Upper boundary learning rate.
    max_lr: f64,
    /// Total cycle size (step_size_up + step_size_down).
    total_size: f64,
    /// Ratio of the up-phase to the full cycle.
    step_ratio: f64,
    /// Scaling mode.
    mode: CyclicMode,
    /// Gamma for exp_range mode.
    gamma: f64,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl CyclicLR {
    /// Create a new `CyclicLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `base_lr` - Lower learning rate boundary.
    /// * `max_lr` - Upper learning rate boundary.
    /// * `step_size_up` - Number of iterations in the increasing half of a cycle.
    /// * `step_size_down` - Number of iterations in the decreasing half.
    ///   If `None`, defaults to `step_size_up`.
    /// * `mode` - One of `Triangular`, `Triangular2`, or `ExpRange`.
    /// * `gamma` - Constant for `ExpRange` mode: `gamma^(iteration)`.
    pub fn new(
        base_lr: f64,
        max_lr: f64,
        step_size_up: usize,
        step_size_down: Option<usize>,
        mode: CyclicMode,
        gamma: f64,
    ) -> Self {
        let step_size_down = step_size_down.unwrap_or(step_size_up);
        let total_size = (step_size_up + step_size_down) as f64;
        let step_ratio = step_size_up as f64 / total_size;

        Self {
            base_lr,
            max_lr,
            total_size,
            step_ratio,
            mode,
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
        // Cycle number (1-indexed).
        let cycle = (1.0 + step as f64 / self.total_size).floor();
        // Position within the cycle [0, 1).
        let x = 1.0 + step as f64 / self.total_size - cycle;

        // Triangular wave: ramp up then ramp down.
        let scale_factor = if x <= self.step_ratio {
            x / self.step_ratio
        } else {
            (x - 1.0) / (self.step_ratio - 1.0)
        };

        let base_height = (self.max_lr - self.base_lr) * scale_factor;

        match self.mode {
            CyclicMode::Triangular => self.base_lr + base_height,
            CyclicMode::Triangular2 => {
                // Scale by 1 / 2^(cycle-1).
                self.base_lr + base_height / 2.0_f64.powf(cycle - 1.0)
            }
            CyclicMode::ExpRange => {
                // Scale by gamma^iteration.
                self.base_lr + base_height * self.gamma.powi(step as i32)
            }
        }
    }
}

impl<T: Float> LrScheduler<T> for CyclicLR {
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
    fn test_cyclic_triangular_peak() {
        // step_size_up=10, step_size_down=10 => total=20, step_ratio=0.5
        // At step 10, x = 1 + 10/20 - 1 = 0.5, scale_factor = 0.5/0.5 = 1.0
        // => lr = base + (max - base) * 1.0 = max_lr
        let mut sched = CyclicLR::new(0.001, 0.01, 10, None, CyclicMode::Triangular, 1.0);
        let mut opt = MockOptimizer::new(0.001);

        for _ in 0..10 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 0.01).abs() < 1e-10,
            "expected peak 0.01, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_cyclic_triangular_valley() {
        // After a full cycle (20 steps), should be back at base_lr.
        let mut sched = CyclicLR::new(0.001, 0.01, 10, None, CyclicMode::Triangular, 1.0);
        let mut opt = MockOptimizer::new(0.001);

        for _ in 0..20 {
            sched.step(&mut opt);
        }
        // At step 20: cycle=2, x = 1 + 20/20 - 2 = 0.0
        // scale_factor = 0.0/0.5 = 0.0 => lr = base_lr
        assert!(
            (opt.lr - 0.001).abs() < 1e-10,
            "expected valley 0.001, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_cyclic_triangular2_halves_amplitude() {
        let base = 0.0;
        let max = 1.0;
        let mut sched = CyclicLR::new(base, max, 10, None, CyclicMode::Triangular2, 1.0);
        let mut opt = MockOptimizer::new(base);

        // Peak of cycle 1 (step 10): full amplitude.
        for _ in 0..10 {
            sched.step(&mut opt);
        }
        let peak1 = opt.lr;

        // Complete cycle 1, then peak of cycle 2 (step 30).
        for _ in 0..20 {
            sched.step(&mut opt);
        }
        let peak2 = opt.lr;

        // Peak 2 should be half of peak 1.
        assert!(
            (peak2 - peak1 / 2.0).abs() < 1e-10,
            "expected peak2={}, got {}",
            peak1 / 2.0,
            peak2
        );
    }

    #[test]
    fn test_cyclic_exp_range_decays() {
        let base = 0.0;
        let max = 1.0;
        let gamma = 0.99;
        let mut sched = CyclicLR::new(base, max, 10, None, CyclicMode::ExpRange, gamma);
        let mut opt = MockOptimizer::new(base);

        // Peak of cycle 1.
        for _ in 0..10 {
            sched.step(&mut opt);
        }
        let peak1 = opt.lr;

        // Peak of cycle 2.
        for _ in 0..20 {
            sched.step(&mut opt);
        }
        let peak2 = opt.lr;

        // exp_range should cause peak2 < peak1.
        assert!(
            peak2 < peak1,
            "exp_range should decay: peak1={peak1}, peak2={peak2}"
        );
    }

    #[test]
    fn test_cyclic_asymmetric_cycle() {
        // step_size_up=5, step_size_down=15 => faster ascent, slower descent.
        let mut sched = CyclicLR::new(0.0, 1.0, 5, Some(15), CyclicMode::Triangular, 1.0);
        let mut opt = MockOptimizer::new(0.0);

        // At step 5: should be at peak.
        for _ in 0..5 {
            sched.step(&mut opt);
        }
        assert!(
            (opt.lr - 1.0).abs() < 1e-10,
            "expected peak 1.0, got {}",
            opt.lr
        );

        // At step 20: should be back at base.
        for _ in 0..15 {
            sched.step(&mut opt);
        }
        assert!(opt.lr.abs() < 1e-10, "expected valley 0.0, got {}", opt.lr);
    }

    #[test]
    fn test_cyclic_midpoint_ramp_up() {
        // step_size_up=10, symmetric.
        // At step 5: halfway up.
        let base = 0.0;
        let max = 1.0;
        let mut sched = CyclicLR::new(base, max, 10, None, CyclicMode::Triangular, 1.0);
        let mut opt = MockOptimizer::new(base);

        for _ in 0..5 {
            sched.step(&mut opt);
        }
        // x = 1 + 5/20 - 1 = 0.25, scale_factor = 0.25/0.5 = 0.5
        assert!((opt.lr - 0.5).abs() < 1e-10, "expected 0.5, got {}", opt.lr);
    }
}
