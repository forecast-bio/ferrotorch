//! Reduce learning rate on plateau scheduler.
//!
//! Monitors a metric and reduces the learning rate when the metric has
//! stopped improving for `patience` steps.

use ferrotorch_core::Float;

use crate::optimizer::Optimizer;

/// Mode for plateau detection: minimize or maximize the metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlateauMode {
    /// Reduce LR when the metric stops decreasing.
    Min,
    /// Reduce LR when the metric stops increasing.
    Max,
}

/// A metric-aware scheduler that reduces the learning rate when a metric
/// plateaus.
///
/// Unlike [`LrScheduler`](super::LrScheduler), this scheduler requires
/// a metric value at each step, so it implements its own
/// [`MetricScheduler`] trait instead.
///
/// # Algorithm
///
/// 1. Track the best metric value seen so far.
/// 2. If the metric has not improved for `patience` consecutive calls,
///    multiply the current learning rate by `factor`.
/// 3. Optionally enforce a minimum learning rate (`min_lr`).
///
/// # Example
///
/// ```ignore
/// let mut scheduler = ReduceLROnPlateau::new(PlateauMode::Min)
///     .factor(0.1)
///     .patience(10)
///     .min_lr(1e-6);
///
/// for epoch in 0..100 {
///     let val_loss = train_one_epoch();
///     scheduler.step(&mut optimizer, val_loss);
/// }
/// ```
#[derive(Debug, Clone)]
pub struct ReduceLROnPlateau {
    /// Whether we are minimizing or maximizing the metric.
    mode: PlateauMode,
    /// Factor by which the learning rate is reduced. `new_lr = lr * factor`.
    factor: f64,
    /// Number of steps with no improvement before reducing LR.
    patience: usize,
    /// Minimum learning rate. LR will not be reduced below this value.
    min_lr: f64,
    /// Threshold for measuring improvement (relative).
    threshold: f64,
    /// Best metric value seen so far.
    best: f64,
    /// Number of steps since the last improvement.
    num_bad_steps: usize,
    /// Current learning rate (tracked for `get_lr()`).
    current_lr: f64,
    /// Whether we have received at least one metric value.
    initialized: bool,
}

impl ReduceLROnPlateau {
    /// Create a new plateau scheduler with the given mode.
    ///
    /// Defaults:
    /// - `factor = 0.1`
    /// - `patience = 10`
    /// - `min_lr = 0.0`
    /// - `threshold = 1e-4`
    pub fn new(mode: PlateauMode) -> Self {
        let best = match mode {
            PlateauMode::Min => f64::INFINITY,
            PlateauMode::Max => f64::NEG_INFINITY,
        };
        Self {
            mode,
            factor: 0.1,
            patience: 10,
            min_lr: 0.0,
            threshold: 1e-4,
            best,
            num_bad_steps: 0,
            current_lr: 0.0,
            initialized: false,
        }
    }

    /// Set the multiplicative factor for LR reduction.
    pub fn factor(mut self, factor: f64) -> Self {
        self.factor = factor;
        self
    }

    /// Set the number of patience steps.
    pub fn patience(mut self, patience: usize) -> Self {
        self.patience = patience;
        self
    }

    /// Set the minimum learning rate.
    pub fn min_lr(mut self, min_lr: f64) -> Self {
        self.min_lr = min_lr;
        self
    }

    /// Set the threshold for measuring improvement.
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Return the current learning rate.
    pub fn get_lr(&self) -> f64 {
        self.current_lr
    }

    /// Check whether the metric has improved relative to the best.
    fn is_better(&self, metric: f64) -> bool {
        match self.mode {
            PlateauMode::Min => metric < self.best * (1.0 - self.threshold),
            PlateauMode::Max => metric > self.best * (1.0 + self.threshold),
        }
    }
}

/// Trait for schedulers that require a metric value each step.
///
/// This is separate from [`LrScheduler`](super::LrScheduler) because the
/// signature differs -- plateau schedulers need a metric to decide whether
/// to reduce the learning rate.
pub trait MetricScheduler<T: Float> {
    /// Perform one scheduler step with the given metric value.
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>, metric: f64);

    /// Return the current learning rate.
    fn get_lr(&self) -> f64;
}

impl<T: Float> MetricScheduler<T> for ReduceLROnPlateau {
    fn step(&mut self, optimizer: &mut dyn Optimizer<T>, metric: f64) {
        // On first call, snapshot the optimizer's current LR.
        if !self.initialized {
            self.current_lr = optimizer.lr();
            self.initialized = true;
        }

        if self.is_better(metric) {
            self.best = metric;
            self.num_bad_steps = 0;
        } else {
            self.num_bad_steps += 1;
        }

        if self.num_bad_steps > self.patience {
            let new_lr = (self.current_lr * self.factor).max(self.min_lr);
            if new_lr < self.current_lr {
                self.current_lr = new_lr;
                optimizer.set_lr(new_lr);
            }
            self.num_bad_steps = 0;
        }
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
    fn test_plateau_no_reduction_when_improving() {
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(3)
            .factor(0.5);
        let mut opt = MockOptimizer::new(0.1);

        // Steadily improving metric (decreasing for Min mode).
        for i in 0..10 {
            let metric = 1.0 - 0.1 * i as f64;
            <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, metric);
        }
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "LR should not change when improving; got {}",
            opt.lr
        );
    }

    #[test]
    fn test_plateau_reduces_after_patience() {
        let patience = 3;
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(patience)
            .factor(0.5)
            .threshold(0.0);
        let mut opt = MockOptimizer::new(0.1);

        // Give it one good value, then plateau.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        assert!((opt.lr - 0.1).abs() < 1e-12);

        // patience + 1 steps of no improvement triggers reduction.
        for _ in 0..=patience {
            <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        }
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "expected 0.05, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_plateau_respects_min_lr() {
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(0)
            .factor(0.1)
            .min_lr(0.01)
            .threshold(0.0);
        let mut opt = MockOptimizer::new(0.1);

        // Each step with a non-improving metric should reduce LR, but not below min.
        for _ in 0..20 {
            <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 999.0);
        }
        assert!(
            opt.lr >= 0.01 - 1e-12,
            "LR should not go below min_lr; got {}",
            opt.lr
        );
    }

    #[test]
    fn test_plateau_max_mode() {
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Max)
            .patience(2)
            .factor(0.5)
            .threshold(0.0);
        let mut opt = MockOptimizer::new(0.1);

        // Improving metric in max mode (increasing).
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 2.0);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 3.0);
        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "should not reduce when improving in max mode"
        );

        // Stagnant metric.
        for _ in 0..=2 {
            <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 3.0);
        }
        assert!(
            (opt.lr - 0.05).abs() < 1e-12,
            "expected 0.05 after plateau in max mode, got {}",
            opt.lr
        );
    }

    #[test]
    fn test_plateau_resets_bad_count_on_improvement() {
        let mut sched = ReduceLROnPlateau::new(PlateauMode::Min)
            .patience(3)
            .factor(0.5)
            .threshold(0.0);
        let mut opt = MockOptimizer::new(0.1);

        // Start with a good value.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);

        // 2 bad steps (below patience).
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 1.0);

        // Improvement resets the counter.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.5);

        // 3 more bad steps -- should NOT trigger because counter was reset.
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.5);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.5);
        <ReduceLROnPlateau as MetricScheduler<f32>>::step(&mut sched, &mut opt, 0.5);

        assert!(
            (opt.lr - 0.1).abs() < 1e-12,
            "LR should not have been reduced; got {}",
            opt.lr
        );
    }
}
