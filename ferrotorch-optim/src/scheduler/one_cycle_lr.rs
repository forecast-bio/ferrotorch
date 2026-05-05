//! One-cycle learning rate scheduler.
//!
//! Implements the 1cycle policy from "Super-Convergence: Very Fast Training
//! of Neural Networks Using Large Learning Rates" (Smith & Topin, 2018).
//!
//! The learning rate anneals from `initial_lr = max_lr / div_factor` up to
//! `max_lr`, then back down to `min_lr = initial_lr / final_div_factor`.
//! Supports both cosine and linear annealing strategies, and an optional
//! three-phase variant.
//!
//! [CL-320]

use ferrotorch_core::Float;

use super::LrScheduler;
use crate::optimizer::Optimizer;

/// Annealing strategy for the one-cycle policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AnnealStrategy {
    /// Cosine annealing between phase endpoints.
    Cos,
    /// Linear annealing between phase endpoints.
    Linear,
}

/// A single phase of the one-cycle schedule.
#[derive(Debug, Clone)]
struct SchedulePhase {
    /// End step of this phase (inclusive).
    end_step: f64,
    /// Starting LR for this phase.
    start_lr: f64,
    /// Ending LR for this phase.
    end_lr: f64,
}

/// One-cycle learning rate scheduler.
///
/// # Two-phase mode (default)
///
/// 1. Ramp from `initial_lr` to `max_lr` over `pct_start * total_steps`.
/// 2. Anneal from `max_lr` to `min_lr` over the remaining steps.
///
/// # Three-phase mode
///
/// 1. Ramp from `initial_lr` to `max_lr` over `pct_start * total_steps`.
/// 2. Anneal from `max_lr` back to `initial_lr` over `pct_start * total_steps`.
/// 3. Anneal from `initial_lr` to `min_lr` over the remaining steps.
///
/// # Example
///
/// ```ignore
/// let scheduler = OneCycleLR::new(
///     0.01,   // max_lr
///     1000,   // total_steps
///     0.3,    // pct_start
///     AnnealStrategy::Cos,
///     25.0,   // div_factor
///     1e4,    // final_div_factor
///     false,  // three_phase
/// );
/// ```
#[derive(Debug, Clone)]
pub struct OneCycleLR {
    /// Schedule phases.
    phases: Vec<SchedulePhase>,
    /// Total number of steps (retained for state introspection).
    #[allow(dead_code)]
    total_steps: usize,
    /// Annealing strategy.
    anneal_strategy: AnnealStrategy,
    /// Current step count.
    current_step: usize,
    /// Current computed learning rate.
    current_lr: f64,
}

impl OneCycleLR {
    /// Create a new `OneCycleLR` scheduler.
    ///
    /// # Arguments
    ///
    /// * `max_lr` - Upper learning rate boundary.
    /// * `total_steps` - Total number of training steps.
    /// * `pct_start` - Fraction of steps spent in the increasing phase (0..1).
    /// * `anneal_strategy` - `Cos` or `Linear` annealing.
    /// * `div_factor` - Determines initial LR: `initial_lr = max_lr / div_factor`.
    /// * `final_div_factor` - Determines final LR: `min_lr = initial_lr / final_div_factor`.
    /// * `three_phase` - If `true`, use the three-phase variant.
    ///
    /// # Panics
    ///
    /// Panics if `total_steps == 0` or `pct_start` is not in `[0, 1]`.
    pub fn new(
        max_lr: f64,
        total_steps: usize,
        pct_start: f64,
        anneal_strategy: AnnealStrategy,
        div_factor: f64,
        final_div_factor: f64,
        three_phase: bool,
    ) -> Self {
        assert!(total_steps > 0, "total_steps must be > 0");
        assert!(
            (0.0..=1.0).contains(&pct_start),
            "pct_start must be in [0, 1], got {pct_start}"
        );

        let initial_lr = max_lr / div_factor;
        let min_lr = initial_lr / final_div_factor;

        let phases = if three_phase {
            vec![
                SchedulePhase {
                    end_step: pct_start * total_steps as f64 - 1.0,
                    start_lr: initial_lr,
                    end_lr: max_lr,
                },
                SchedulePhase {
                    end_step: 2.0 * pct_start * total_steps as f64 - 2.0,
                    start_lr: max_lr,
                    end_lr: initial_lr,
                },
                SchedulePhase {
                    end_step: total_steps as f64 - 1.0,
                    start_lr: initial_lr,
                    end_lr: min_lr,
                },
            ]
        } else {
            vec![
                SchedulePhase {
                    end_step: pct_start * total_steps as f64 - 1.0,
                    start_lr: initial_lr,
                    end_lr: max_lr,
                },
                SchedulePhase {
                    end_step: total_steps as f64 - 1.0,
                    start_lr: max_lr,
                    end_lr: min_lr,
                },
            ]
        };

        Self {
            phases,
            total_steps,
            anneal_strategy,
            current_step: 0,
            current_lr: initial_lr,
        }
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Cosine anneal from `start` to `end` as `pct` goes from 0 to 1.
    fn anneal_cos(start: f64, end: f64, pct: f64) -> f64 {
        let cos_out = (std::f64::consts::PI * pct).cos() + 1.0;
        end + (start - end) / 2.0 * cos_out
    }

    /// Linear anneal from `start` to `end` as `pct` goes from 0 to 1.
    fn anneal_linear(start: f64, end: f64, pct: f64) -> f64 {
        (end - start) * pct + start
    }

    /// Compute the learning rate at the given step.
    fn compute_lr(&self, step: usize) -> f64 {
        let step_num = step as f64;
        let mut start_step = 0.0_f64;

        for (i, phase) in self.phases.iter().enumerate() {
            if step_num <= phase.end_step || i == self.phases.len() - 1 {
                let pct = if (phase.end_step - start_step).abs() < 1e-12 {
                    1.0
                } else {
                    (step_num - start_step) / (phase.end_step - start_step)
                };
                return match self.anneal_strategy {
                    AnnealStrategy::Cos => Self::anneal_cos(phase.start_lr, phase.end_lr, pct),
                    AnnealStrategy::Linear => {
                        Self::anneal_linear(phase.start_lr, phase.end_lr, pct)
                    }
                };
            }
            start_step = phase.end_step;
        }

        // Shouldn't reach here, but return the last phase's end_lr.
        self.phases.last().map(|p| p.end_lr).unwrap_or(0.0)
    }
}

impl<T: Float> LrScheduler<T> for OneCycleLR {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>) {
        self.current_lr = self.compute_lr(self.current_step);
        optimizer.set_lr(self.current_lr);
        self.current_step += 1;
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
    fn test_one_cycle_initial_lr() {
        let max_lr = 0.01;
        let div_factor = 25.0;
        let sched = OneCycleLR::new(
            max_lr,
            100,
            0.3,
            AnnealStrategy::Cos,
            div_factor,
            1e4,
            false,
        );
        let expected_initial = max_lr / div_factor;
        assert!(
            (sched.get_lr() - expected_initial).abs() < 1e-12,
            "expected {expected_initial}, got {}",
            sched.get_lr()
        );
    }

    #[test]
    fn test_one_cycle_reaches_max_lr_cos() {
        let max_lr = 0.1;
        let total_steps = 100;
        let pct_start = 0.3;
        let mut sched = OneCycleLR::new(
            max_lr,
            total_steps,
            pct_start,
            AnnealStrategy::Cos,
            25.0,
            1e4,
            false,
        );
        let mut opt = MockOptimizer::new(0.004);

        // Step to the end of phase 1.
        let phase1_end = (pct_start * total_steps as f64) as usize;
        for _ in 0..phase1_end {
            sched.step(&mut opt);
        }

        // At the boundary, LR should be close to max_lr.
        // It won't be exact due to phase boundary math, but should be within tolerance.
        assert!(
            (opt.lr - max_lr).abs() < 0.01,
            "at phase boundary: expected ~{max_lr}, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_one_cycle_end_lr() {
        let max_lr = 0.1;
        let total_steps = 100;
        let div_factor = 25.0;
        let final_div_factor = 1e4;
        let mut sched = OneCycleLR::new(
            max_lr,
            total_steps,
            0.3,
            AnnealStrategy::Cos,
            div_factor,
            final_div_factor,
            false,
        );
        let mut opt = MockOptimizer::new(0.004);

        // Run all steps.
        for _ in 0..total_steps {
            sched.step(&mut opt);
        }

        let initial_lr = max_lr / div_factor;
        let min_lr = initial_lr / final_div_factor;
        assert!(
            (opt.lr - min_lr).abs() < 1e-10,
            "expected min_lr={min_lr}, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_one_cycle_linear_monotonic_ramp() {
        let max_lr = 1.0;
        let total_steps = 100;
        let pct_start = 0.3;
        let mut sched = OneCycleLR::new(
            max_lr,
            total_steps,
            pct_start,
            AnnealStrategy::Linear,
            25.0,
            1e4,
            false,
        );
        let mut opt = MockOptimizer::new(0.04);

        let phase1_steps = (pct_start * total_steps as f64) as usize;
        let mut prev_lr = 0.0;
        for i in 0..phase1_steps {
            sched.step(&mut opt);
            if i > 0 {
                assert!(
                    opt.lr >= prev_lr - 1e-12,
                    "step {i}: LR should be monotonically increasing in ramp phase"
                );
            }
            prev_lr = opt.lr;
        }
    }

    #[test]
    fn test_one_cycle_three_phase() {
        let max_lr = 0.1;
        let total_steps = 100;
        let pct_start = 0.3;
        let div_factor = 25.0;
        let final_div_factor = 1e4;
        let mut sched = OneCycleLR::new(
            max_lr,
            total_steps,
            pct_start,
            AnnealStrategy::Cos,
            div_factor,
            final_div_factor,
            true,
        );
        let mut opt = MockOptimizer::new(0.004);

        let initial_lr = max_lr / div_factor;
        let min_lr = initial_lr / final_div_factor;

        // Run all steps.
        for _ in 0..total_steps {
            sched.step(&mut opt);
        }

        // At the end of three-phase, LR should approach min_lr.
        assert!(
            (opt.lr - min_lr).abs() < 1e-10,
            "three_phase end: expected min_lr={min_lr}, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_one_cycle_lr_never_negative() {
        let mut sched = OneCycleLR::new(0.01, 200, 0.3, AnnealStrategy::Cos, 25.0, 1e4, false);
        let mut opt = MockOptimizer::new(0.0004);

        for step in 0..200 {
            sched.step(&mut opt);
            assert!(
                opt.lr >= 0.0,
                "step {step}: LR should never be negative, got {}",
                opt.lr
            );
        }
    }

    #[test]
    #[should_panic(expected = "total_steps must be > 0")]
    fn test_one_cycle_zero_steps_panics() {
        OneCycleLR::new(0.01, 0, 0.3, AnnealStrategy::Cos, 25.0, 1e4, false);
    }
}
