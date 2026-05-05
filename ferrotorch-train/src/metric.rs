//! Metric trait and built-in training metrics.
//!
//! Metrics accumulate values over batches and compute a summary at the end of
//! an epoch (or evaluation pass). All metrics are [`Send + Sync`] so they can
//! be shared across threads in a distributed setting.
//!
//! # Provided metrics
//!
//! | Metric | Description |
//! |--------|-------------|
//! | [`LossMetric`] | Running mean of scalar loss values |
//! | [`AccuracyMetric`] | Fraction of correct predictions |
//! | [`TopKAccuracy`] | Fraction of correct within top-k predictions |
//! | [`RunningAverage`] | Windowed average of arbitrary f64 values |

// ---------------------------------------------------------------------------
// Metric trait
// ---------------------------------------------------------------------------

/// A metric that accumulates values over batches and produces a scalar summary.
///
/// The associated `Input` type determines what is fed into [`update`](Metric::update).
/// For simple loss tracking this is `f64`; for accuracy metrics it could be
/// a `(predicted, target)` pair, etc.
pub trait Metric: Send + Sync {
    /// The type of data passed to [`update`](Metric::update) each batch.
    type Input;

    /// Accumulate one batch of observations.
    fn update(&mut self, input: &Self::Input);

    /// Compute the current metric value from accumulated data.
    fn compute(&self) -> f64;

    /// Reset all internal state (typically called at the start of each epoch).
    fn reset(&mut self);

    /// Human-readable name of this metric (used in logging and history).
    fn name(&self) -> &str;
}

// ---------------------------------------------------------------------------
// LossMetric
// ---------------------------------------------------------------------------

/// Tracks the running mean of scalar loss values.
///
/// # Examples
///
/// ```
/// use ferrotorch_train::LossMetric;
/// use ferrotorch_train::Metric;
///
/// let mut m = LossMetric::new();
/// m.update(&2.0);
/// m.update(&4.0);
/// assert!((m.compute() - 3.0).abs() < 1e-12);
/// ```
pub struct LossMetric {
    sum: f64,
    count: usize,
}

impl LossMetric {
    /// Create a new `LossMetric` with zero accumulated state.
    pub fn new() -> Self {
        Self { sum: 0.0, count: 0 }
    }
}

impl Default for LossMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for LossMetric {
    type Input = f64;

    fn update(&mut self, input: &f64) {
        self.sum += *input;
        self.count += 1;
    }

    fn compute(&self) -> f64 {
        if self.count == 0 {
            0.0
        } else {
            self.sum / self.count as f64
        }
    }

    fn reset(&mut self) {
        self.sum = 0.0;
        self.count = 0;
    }

    fn name(&self) -> &'static str {
        "loss"
    }
}

// ---------------------------------------------------------------------------
// AccuracyMetric
// ---------------------------------------------------------------------------

/// Tracks classification accuracy as `correct / total`.
///
/// Each [`update`](Metric::update) call receives a `(correct_count, batch_size)`
/// pair so the caller can compute correctness however it sees fit (argmax,
/// threshold, etc.) before feeding the metric.
///
/// # Examples
///
/// ```
/// use ferrotorch_train::AccuracyMetric;
/// use ferrotorch_train::Metric;
///
/// let mut m = AccuracyMetric::new();
/// m.update(&(8, 10)); // 8 of 10 correct
/// m.update(&(9, 10)); // 9 of 10 correct
/// assert!((m.compute() - 0.85).abs() < 1e-12);
/// ```
pub struct AccuracyMetric {
    correct: usize,
    total: usize,
}

impl AccuracyMetric {
    /// Create a new `AccuracyMetric` with zero accumulated state.
    pub fn new() -> Self {
        Self {
            correct: 0,
            total: 0,
        }
    }
}

impl Default for AccuracyMetric {
    fn default() -> Self {
        Self::new()
    }
}

impl Metric for AccuracyMetric {
    /// `(correct_count, batch_size)`.
    type Input = (usize, usize);

    fn update(&mut self, input: &(usize, usize)) {
        self.correct += input.0;
        self.total += input.1;
    }

    fn compute(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn name(&self) -> &'static str {
        "accuracy"
    }
}

// ---------------------------------------------------------------------------
// TopKAccuracy
// ---------------------------------------------------------------------------

/// Tracks top-k classification accuracy.
///
/// Each [`update`](Metric::update) call receives a `(correct_in_top_k, batch_size)`
/// pair, analogous to [`AccuracyMetric`] but for top-k predictions.
///
/// # Examples
///
/// ```
/// use ferrotorch_train::TopKAccuracy;
/// use ferrotorch_train::Metric;
///
/// let mut m = TopKAccuracy::new(5);
/// m.update(&(9, 10));
/// assert!((m.compute() - 0.9).abs() < 1e-12);
/// assert_eq!(m.k(), 5);
/// ```
pub struct TopKAccuracy {
    k: usize,
    correct: usize,
    total: usize,
}

impl TopKAccuracy {
    /// Create a new `TopKAccuracy` metric for the given `k`.
    ///
    /// # Panics
    ///
    /// Panics if `k` is 0.
    pub fn new(k: usize) -> Self {
        assert!(k > 0, "k must be > 0");
        Self {
            k,
            correct: 0,
            total: 0,
        }
    }

    /// Return the `k` value this metric tracks.
    pub fn k(&self) -> usize {
        self.k
    }
}

impl Metric for TopKAccuracy {
    /// `(correct_in_top_k, batch_size)`.
    type Input = (usize, usize);

    fn update(&mut self, input: &(usize, usize)) {
        self.correct += input.0;
        self.total += input.1;
    }

    fn compute(&self) -> f64 {
        if self.total == 0 {
            0.0
        } else {
            self.correct as f64 / self.total as f64
        }
    }

    fn reset(&mut self) {
        self.correct = 0;
        self.total = 0;
    }

    fn name(&self) -> &'static str {
        "top_k_accuracy"
    }
}

// ---------------------------------------------------------------------------
// RunningAverage
// ---------------------------------------------------------------------------

/// Tracks a windowed running average of `f64` values.
///
/// Unlike [`LossMetric`] (which averages *all* values since the last reset),
/// `RunningAverage` keeps only the most recent `window_size` values. This is
/// useful for smoothing noisy batch metrics during training.
///
/// # Examples
///
/// ```
/// use ferrotorch_train::RunningAverage;
/// use ferrotorch_train::Metric;
///
/// let mut m = RunningAverage::new(3);
/// m.update(&1.0);
/// m.update(&2.0);
/// m.update(&3.0);
/// assert!((m.compute() - 2.0).abs() < 1e-12);
/// m.update(&6.0); // window: [2.0, 3.0, 6.0]
/// assert!((m.compute() - (11.0 / 3.0)).abs() < 1e-12);
/// ```
pub struct RunningAverage {
    values: Vec<f64>,
    window_size: usize,
}

impl RunningAverage {
    /// Create a new `RunningAverage` with the given window size.
    ///
    /// # Panics
    ///
    /// Panics if `window_size` is 0.
    pub fn new(window_size: usize) -> Self {
        assert!(window_size > 0, "window_size must be > 0");
        Self {
            values: Vec::with_capacity(window_size),
            window_size,
        }
    }

    /// Return the window size.
    pub fn window_size(&self) -> usize {
        self.window_size
    }
}

impl Metric for RunningAverage {
    type Input = f64;

    fn update(&mut self, input: &f64) {
        self.values.push(*input);
        if self.values.len() > self.window_size {
            self.values.remove(0);
        }
    }

    fn compute(&self) -> f64 {
        if self.values.is_empty() {
            0.0
        } else {
            self.values.iter().sum::<f64>() / self.values.len() as f64
        }
    }

    fn reset(&mut self) {
        self.values.clear();
    }

    fn name(&self) -> &'static str {
        "running_avg"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- LossMetric ----------------------------------------------------------

    #[test]
    fn test_loss_metric_update_compute() {
        let mut m = LossMetric::new();
        m.update(&1.0);
        m.update(&3.0);
        m.update(&5.0);
        assert!((m.compute() - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_loss_metric_empty() {
        let m = LossMetric::new();
        assert!((m.compute() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_loss_metric_reset() {
        let mut m = LossMetric::new();
        m.update(&10.0);
        m.update(&20.0);
        assert!((m.compute() - 15.0).abs() < 1e-12);
        m.reset();
        assert!((m.compute() - 0.0).abs() < 1e-12);
        m.update(&5.0);
        assert!((m.compute() - 5.0).abs() < 1e-12);
    }

    #[test]
    fn test_loss_metric_name() {
        let m = LossMetric::new();
        assert_eq!(m.name(), "loss");
    }

    #[test]
    fn test_loss_metric_single_value() {
        let mut m = LossMetric::new();
        m.update(&42.0);
        assert!((m.compute() - 42.0).abs() < 1e-12);
    }

    // -- AccuracyMetric ------------------------------------------------------

    #[test]
    fn test_accuracy_metric_update_compute() {
        let mut m = AccuracyMetric::new();
        m.update(&(8, 10));
        m.update(&(9, 10));
        assert!((m.compute() - 0.85).abs() < 1e-12);
    }

    #[test]
    fn test_accuracy_metric_perfect() {
        let mut m = AccuracyMetric::new();
        m.update(&(10, 10));
        assert!((m.compute() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_accuracy_metric_zero() {
        let mut m = AccuracyMetric::new();
        m.update(&(0, 10));
        assert!((m.compute() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_accuracy_metric_empty() {
        let m = AccuracyMetric::new();
        assert!((m.compute() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_accuracy_metric_reset() {
        let mut m = AccuracyMetric::new();
        m.update(&(5, 10));
        m.reset();
        assert!((m.compute() - 0.0).abs() < 1e-12);
        m.update(&(7, 10));
        assert!((m.compute() - 0.7).abs() < 1e-12);
    }

    #[test]
    fn test_accuracy_metric_name() {
        let m = AccuracyMetric::new();
        assert_eq!(m.name(), "accuracy");
    }

    // -- TopKAccuracy --------------------------------------------------------

    #[test]
    fn test_topk_accuracy_basic() {
        let mut m = TopKAccuracy::new(5);
        m.update(&(9, 10));
        assert!((m.compute() - 0.9).abs() < 1e-12);
    }

    #[test]
    fn test_topk_accuracy_k_value() {
        let m = TopKAccuracy::new(3);
        assert_eq!(m.k(), 3);
    }

    #[test]
    fn test_topk_accuracy_reset() {
        let mut m = TopKAccuracy::new(5);
        m.update(&(8, 10));
        m.reset();
        assert!((m.compute() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_topk_accuracy_name() {
        let m = TopKAccuracy::new(5);
        assert_eq!(m.name(), "top_k_accuracy");
    }

    #[test]
    #[should_panic(expected = "k must be > 0")]
    fn test_topk_accuracy_zero_k_panics() {
        let _ = TopKAccuracy::new(0);
    }

    // -- RunningAverage ------------------------------------------------------

    #[test]
    fn test_running_average_within_window() {
        let mut m = RunningAverage::new(5);
        m.update(&1.0);
        m.update(&2.0);
        m.update(&3.0);
        assert!((m.compute() - 2.0).abs() < 1e-12);
    }

    #[test]
    fn test_running_average_exceeds_window() {
        let mut m = RunningAverage::new(3);
        m.update(&1.0);
        m.update(&2.0);
        m.update(&3.0);
        m.update(&6.0); // drops 1.0, window: [2.0, 3.0, 6.0]
        assert!((m.compute() - (11.0 / 3.0)).abs() < 1e-12);
    }

    #[test]
    fn test_running_average_empty() {
        let m = RunningAverage::new(10);
        assert!((m.compute() - 0.0).abs() < 1e-12);
    }

    #[test]
    fn test_running_average_reset() {
        let mut m = RunningAverage::new(3);
        m.update(&5.0);
        m.update(&10.0);
        m.reset();
        assert!((m.compute() - 0.0).abs() < 1e-12);
        m.update(&7.0);
        assert!((m.compute() - 7.0).abs() < 1e-12);
    }

    #[test]
    fn test_running_average_name() {
        let m = RunningAverage::new(5);
        assert_eq!(m.name(), "running_avg");
    }

    #[test]
    fn test_running_average_window_size() {
        let m = RunningAverage::new(42);
        assert_eq!(m.window_size(), 42);
    }

    #[test]
    #[should_panic(expected = "window_size must be > 0")]
    fn test_running_average_zero_window_panics() {
        let _ = RunningAverage::new(0);
    }

    // -- Send + Sync ---------------------------------------------------------

    #[test]
    fn test_metrics_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<LossMetric>();
        assert_send_sync::<AccuracyMetric>();
        assert_send_sync::<TopKAccuracy>();
        assert_send_sync::<RunningAverage>();
    }
}
