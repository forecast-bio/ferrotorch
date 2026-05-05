//! Training history tracking.
//!
//! [`TrainingHistory`] accumulates per-epoch results ([`EpochResult`]) produced
//! by the [`Learner`](crate::Learner) during a call to
//! [`fit`](crate::Learner::fit).

use std::collections::HashMap;
use std::fmt;

// ---------------------------------------------------------------------------
// EpochResult
// ---------------------------------------------------------------------------

/// Summary of a single training epoch.
///
/// Marked `#[non_exhaustive]` so that fields can be added in a minor
/// release without breaking external struct-literal construction.
/// Construct internal callers with the literal syntax in this crate;
/// downstream consumers should obtain instances from
/// [`Learner::fit`](crate::Learner::fit) and read fields by access.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EpochResult {
    /// Zero-indexed epoch number.
    pub epoch: usize,
    /// Mean training loss over all batches in this epoch.
    pub train_loss: f64,
    /// Mean validation loss (if a validation loader was provided).
    pub val_loss: Option<f64>,
    /// Named metric values computed at the end of this epoch.
    pub metrics: HashMap<String, f64>,
    /// Learning rate at the end of this epoch.
    pub lr: f64,
    /// Wall-clock duration of this epoch in seconds.
    pub duration_secs: f64,
}

impl fmt::Display for EpochResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "epoch {}: train_loss={:.6}", self.epoch, self.train_loss)?;
        if let Some(vl) = self.val_loss {
            write!(f, ", val_loss={vl:.6}")?;
        }
        for (name, value) in &self.metrics {
            write!(f, ", {name}={value:.6}")?;
        }
        write!(f, ", lr={:.2e}, {:.1}s", self.lr, self.duration_secs)
    }
}

// ---------------------------------------------------------------------------
// EvalResult
// ---------------------------------------------------------------------------

/// Summary of an evaluation pass.
///
/// Marked `#[non_exhaustive]` so that fields can be added in a minor
/// release without breaking external struct-literal construction.
/// Downstream consumers should obtain instances from
/// [`Learner::evaluate`](crate::Learner::evaluate) and read fields by access.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct EvalResult {
    /// Mean loss over the evaluation dataset.
    pub loss: f64,
    /// Named metric values.
    pub metrics: HashMap<String, f64>,
}

impl fmt::Display for EvalResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "eval_loss={:.6}", self.loss)?;
        for (name, value) in &self.metrics {
            write!(f, ", {name}={value:.6}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// TrainingHistory
// ---------------------------------------------------------------------------

/// Accumulated results from an entire training run.
///
/// Returned by [`Learner::fit`](crate::Learner::fit). Marked
/// `#[non_exhaustive]` so that fields can be added in a minor release
/// without breaking external struct-literal construction; downstream
/// consumers should obtain instances from `fit()` (or via
/// [`TrainingHistory::new`] / [`TrainingHistory::default`]).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct TrainingHistory {
    /// Per-epoch results, in chronological order.
    pub epochs: Vec<EpochResult>,
}

impl TrainingHistory {
    /// Create an empty history.
    pub fn new() -> Self {
        Self { epochs: Vec::new() }
    }

    /// Push a completed epoch result.
    pub fn push(&mut self, result: EpochResult) {
        self.epochs.push(result);
    }

    /// Total number of completed epochs.
    pub fn len(&self) -> usize {
        self.epochs.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.epochs.is_empty()
    }

    /// Return the best (lowest) training loss and the epoch it occurred in.
    ///
    /// Returns `None` if the history is empty.
    pub fn best_train_loss(&self) -> Option<(usize, f64)> {
        self.epochs
            .iter()
            .map(|e| (e.epoch, e.train_loss))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Return the best (lowest) validation loss and the epoch it occurred in.
    ///
    /// Returns `None` if no epochs have validation loss.
    pub fn best_val_loss(&self) -> Option<(usize, f64)> {
        self.epochs
            .iter()
            .filter_map(|e| e.val_loss.map(|vl| (e.epoch, vl)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
    }

    /// Collect all training losses as a `Vec<f64>`.
    pub fn train_losses(&self) -> Vec<f64> {
        self.epochs.iter().map(|e| e.train_loss).collect()
    }

    /// Collect all validation losses as a `Vec<Option<f64>>`.
    pub fn val_losses(&self) -> Vec<Option<f64>> {
        self.epochs.iter().map(|e| e.val_loss).collect()
    }
}

impl Default for TrainingHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Display for TrainingHistory {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for epoch in &self.epochs {
            writeln!(f, "{epoch}")?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_epoch(epoch: usize, train_loss: f64, val_loss: Option<f64>) -> EpochResult {
        EpochResult {
            epoch,
            train_loss,
            val_loss,
            metrics: HashMap::new(),
            lr: 0.001,
            duration_secs: 1.0,
        }
    }

    #[test]
    fn test_history_new_is_empty() {
        let h = TrainingHistory::new();
        assert!(h.is_empty());
        assert_eq!(h.len(), 0);
    }

    #[test]
    fn test_history_push_and_len() {
        let mut h = TrainingHistory::new();
        h.push(make_epoch(0, 1.0, None));
        h.push(make_epoch(1, 0.5, None));
        assert_eq!(h.len(), 2);
        assert!(!h.is_empty());
    }

    #[test]
    fn test_best_train_loss() {
        let mut h = TrainingHistory::new();
        h.push(make_epoch(0, 2.0, None));
        h.push(make_epoch(1, 0.5, None));
        h.push(make_epoch(2, 1.0, None));
        let (epoch, loss) = h.best_train_loss().unwrap();
        assert_eq!(epoch, 1);
        assert!((loss - 0.5).abs() < 1e-12);
    }

    #[test]
    fn test_best_val_loss() {
        let mut h = TrainingHistory::new();
        h.push(make_epoch(0, 2.0, Some(1.5)));
        h.push(make_epoch(1, 1.0, Some(0.8)));
        h.push(make_epoch(2, 0.5, Some(0.9)));
        let (epoch, loss) = h.best_val_loss().unwrap();
        assert_eq!(epoch, 1);
        assert!((loss - 0.8).abs() < 1e-12);
    }

    #[test]
    fn test_best_val_loss_none_when_no_val() {
        let mut h = TrainingHistory::new();
        h.push(make_epoch(0, 1.0, None));
        assert!(h.best_val_loss().is_none());
    }

    #[test]
    fn test_best_train_loss_empty() {
        let h = TrainingHistory::new();
        assert!(h.best_train_loss().is_none());
    }

    #[test]
    fn test_train_losses() {
        let mut h = TrainingHistory::new();
        h.push(make_epoch(0, 3.0, None));
        h.push(make_epoch(1, 2.0, None));
        h.push(make_epoch(2, 1.0, None));
        assert_eq!(h.train_losses(), vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_val_losses() {
        let mut h = TrainingHistory::new();
        h.push(make_epoch(0, 1.0, Some(1.5)));
        h.push(make_epoch(1, 0.5, None));
        assert_eq!(h.val_losses(), vec![Some(1.5), None]);
    }

    #[test]
    fn test_epoch_result_display() {
        let e = make_epoch(0, 0.5, Some(0.6));
        let s = format!("{e}");
        assert!(s.contains("epoch 0"));
        assert!(s.contains("train_loss="));
        assert!(s.contains("val_loss="));
    }

    #[test]
    fn test_epoch_result_display_no_val() {
        let e = make_epoch(0, 0.5, None);
        let s = format!("{e}");
        assert!(!s.contains("val_loss="));
    }

    #[test]
    fn test_eval_result_display() {
        let e = EvalResult {
            loss: 0.42,
            metrics: HashMap::from([("accuracy".to_string(), 0.95)]),
        };
        let s = format!("{e}");
        assert!(s.contains("eval_loss="));
        assert!(s.contains("accuracy="));
    }
}
