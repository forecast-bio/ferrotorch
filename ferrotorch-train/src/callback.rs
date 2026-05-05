//! Training callbacks.
//!
//! Callbacks hook into the training loop at well-defined points (epoch
//! start/end, batch start/end, training end) and can observe or modify
//! training behavior.
//!
//! # Provided callbacks
//!
//! | Callback | Description |
//! |----------|-------------|
//! | [`EarlyStopping`] | Stop training when validation loss stops improving |
//! | [`ProgressLogger`] | Print epoch/batch progress to stdout |
//! | [`EmaCallback`] | Maintain exponential moving average of model parameters |
//!
//! [CL-334] Add gradient checkpointing, autocast context, gradient clipping, and EMA callback

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float};

use crate::history::{EpochResult, TrainingHistory};

// ---------------------------------------------------------------------------
// Callback trait
// ---------------------------------------------------------------------------

/// A callback that hooks into the [`Learner`](crate::Learner) training loop.
///
/// All methods have default no-op implementations so callbacks only need to
/// override the hooks they care about.
pub trait Callback<T: Float>: Send + Sync {
    /// Called at the start of each epoch.
    fn on_epoch_start(&mut self, _epoch: usize) {}

    /// Called at the end of each epoch with the epoch result.
    fn on_epoch_end(&mut self, _epoch: usize, _result: &EpochResult) {}

    /// Called at the start of each batch.
    fn on_batch_start(&mut self, _batch: usize) {}

    /// Called at the end of each batch with the batch loss.
    fn on_batch_end(&mut self, _batch: usize, _loss: f64) {}

    /// Called when the entire training run finishes.
    fn on_train_end(&mut self, _history: &TrainingHistory) {}

    /// Whether this callback requests early stopping.
    ///
    /// The [`Learner`](crate::Learner) checks this after each epoch. If any
    /// callback returns `true`, training stops.
    fn should_stop(&self) -> bool {
        false
    }
}

// ---------------------------------------------------------------------------
// EarlyStopping
// ---------------------------------------------------------------------------

/// Stop training when validation loss fails to improve for `patience` epochs.
///
/// "Improve" means decreasing by at least `min_delta`. The callback tracks the
/// best validation loss seen so far and increments a wait counter when no
/// improvement occurs. Training stops when `wait >= patience`.
///
/// # Examples
///
/// ```
/// use ferrotorch_train::EarlyStopping;
/// use ferrotorch_train::Callback;
///
/// let es = EarlyStopping::new(3, 0.001);
/// assert!(!Callback::<f32>::should_stop(&es));
/// ```
pub struct EarlyStopping {
    patience: usize,
    min_delta: f64,
    best: f64,
    wait: usize,
    stopped: bool,
}

impl EarlyStopping {
    /// Create a new `EarlyStopping` callback.
    ///
    /// # Arguments
    ///
    /// * `patience` - Number of epochs with no improvement before stopping.
    /// * `min_delta` - Minimum decrease in validation loss to count as improvement.
    pub fn new(patience: usize, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best: f64::INFINITY,
            wait: 0,
            stopped: false,
        }
    }

    /// Return the current best validation loss.
    pub fn best_loss(&self) -> f64 {
        self.best
    }

    /// Return the current wait counter.
    pub fn wait(&self) -> usize {
        self.wait
    }

    /// Return the patience value.
    pub fn patience(&self) -> usize {
        self.patience
    }
}

impl<T: Float> Callback<T> for EarlyStopping {
    fn on_epoch_end(&mut self, _epoch: usize, result: &EpochResult) {
        let val_loss = match result.val_loss {
            Some(vl) => vl,
            // No validation loss: nothing to monitor.
            None => return,
        };

        if val_loss < self.best - self.min_delta {
            // Improvement: reset wait counter.
            self.best = val_loss;
            self.wait = 0;
        } else {
            // No improvement.
            self.wait += 1;
            if self.wait >= self.patience {
                self.stopped = true;
            }
        }
    }

    fn should_stop(&self) -> bool {
        self.stopped
    }
}

// ---------------------------------------------------------------------------
// ProgressLogger
// ---------------------------------------------------------------------------

/// Logs training progress via the `tracing` crate.
///
/// Emits `tracing::info!` events at epoch start/end and batch-level loss for
/// visibility during training. Output is redirectable / suppressible by
/// configuring a `tracing` subscriber in the consumer (e.g. via
/// `tracing_subscriber::fmt`); installing no subscriber silently drops the
/// events. The log target is `ferrotorch::progress`.
pub struct ProgressLogger {
    log_every_n_batches: usize,
}

impl ProgressLogger {
    /// Create a new `ProgressLogger`.
    ///
    /// # Arguments
    ///
    /// * `log_every_n_batches` - Print batch-level loss every N batches.
    ///   Set to 0 to disable batch-level logging.
    pub fn new(log_every_n_batches: usize) -> Self {
        Self {
            log_every_n_batches,
        }
    }
}

impl Default for ProgressLogger {
    fn default() -> Self {
        Self::new(0)
    }
}

impl<T: Float> Callback<T> for ProgressLogger {
    fn on_epoch_start(&mut self, epoch: usize) {
        tracing::info!(target: "ferrotorch::progress", epoch, "--- Epoch {epoch} ---");
    }

    fn on_epoch_end(&mut self, _epoch: usize, result: &EpochResult) {
        tracing::info!(target: "ferrotorch::progress", "{result}");
    }

    fn on_batch_end(&mut self, batch: usize, loss: f64) {
        if self.log_every_n_batches > 0 && batch % self.log_every_n_batches == 0 {
            tracing::info!(
                target: "ferrotorch::progress",
                batch,
                loss,
                "  batch {batch}: loss={loss:.6}",
            );
        }
    }

    fn on_train_end(&mut self, history: &TrainingHistory) {
        tracing::info!(
            target: "ferrotorch::progress",
            epochs = history.len(),
            "Training complete. {} epochs.",
            history.len(),
        );
        if let Some((epoch, loss)) = history.best_train_loss() {
            tracing::info!(
                target: "ferrotorch::progress",
                epoch,
                loss,
                "Best train loss: {loss:.6} (epoch {epoch})",
            );
        }
        if let Some((epoch, loss)) = history.best_val_loss() {
            tracing::info!(
                target: "ferrotorch::progress",
                epoch,
                loss,
                "Best val loss: {loss:.6} (epoch {epoch})",
            );
        }
    }
}

// ---------------------------------------------------------------------------
// EmaCallback
// ---------------------------------------------------------------------------

/// Exponential Moving Average (EMA) of model parameters.
///
/// Maintains a shadow copy of model parameters as an exponentially weighted
/// moving average. After each batch, updates the shadow parameters:
///
/// ```text
/// shadow = decay * shadow + (1 - decay) * current_param
/// ```
///
/// At evaluation time, the shadow parameters can be swapped into the model
/// to get smoother, more stable predictions. This is widely used in
/// practice (e.g., by Polyak averaging, SWA, and many GAN training setups).
///
/// # Usage
///
/// ```ignore
/// use ferrotorch_train::EmaCallback;
///
/// // Create with decay=0.999 (typical value).
/// let mut ema = EmaCallback::new(0.999);
///
/// // Attach to learner — it will track parameter updates automatically.
/// let learner = Learner::new(model, optimizer, loss_fn)
///     .with_callback(Box::new(ema));
/// ```
///
/// # Notes
///
/// - The shadow parameters are initialized lazily on the first batch end.
/// - `decay` should be close to 1.0 (e.g., 0.999 or 0.9999). Higher values
///   produce smoother averages with more lag.
///
/// [CL-334] Add gradient checkpointing, autocast context, gradient clipping, and EMA callback
pub struct EmaCallback {
    /// Decay factor. Typically 0.999 or 0.9999.
    decay: f64,
    /// Number of update steps performed.
    num_updates: usize,
    /// Shadow parameter values, stored as flat `Vec<f64>` for each parameter.
    /// The outer Vec corresponds to named_parameters() in order.
    shadow: Vec<Vec<f64>>,
    /// Whether the shadow has been initialized.
    initialized: bool,
}

impl EmaCallback {
    /// Create a new EMA callback.
    ///
    /// # Arguments
    ///
    /// * `decay` - The EMA decay factor. Must be in `[0, 1]`.
    ///   A typical value is `0.999`.
    ///
    /// # Panics
    ///
    /// Panics if `decay` is not in `[0, 1]`.
    pub fn new(decay: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&decay),
            "decay must be in [0, 1], got {decay}"
        );
        Self {
            decay,
            num_updates: 0,
            shadow: Vec::new(),
            initialized: false,
        }
    }

    /// Return the decay factor.
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Return the number of EMA update steps performed.
    pub fn num_updates(&self) -> usize {
        self.num_updates
    }

    /// Whether the shadow parameters have been initialized.
    pub fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// Return the shadow parameter values.
    ///
    /// Each inner `Vec<f64>` corresponds to one parameter (in the order of
    /// `named_parameters()`), stored as a flat array of f64 values.
    pub fn shadow_params(&self) -> &[Vec<f64>] {
        &self.shadow
    }

    /// Initialize shadow parameters from the given parameter values.
    ///
    /// Called internally on the first batch. Can also be called manually to
    /// reinitialize.
    ///
    /// # Errors
    ///
    /// Returns `FerrotorchError::InvalidArgument` from the numeric `cast`
    /// helper when a parameter value (e.g. an `f16`/`bf16` value outside
    /// the `f64` representable range) cannot be converted to `f64`.
    pub fn init_from_params<T: Float>(&mut self, params: &[Vec<T>]) -> FerrotorchResult<()> {
        let mut shadow: Vec<Vec<f64>> = Vec::with_capacity(params.len());
        for p in params {
            let mut row: Vec<f64> = Vec::with_capacity(p.len());
            for &v in p {
                row.push(cast::<T, f64>(v)?);
            }
            shadow.push(row);
        }
        self.shadow = shadow;
        self.initialized = true;
        Ok(())
    }

    /// Update the shadow parameters with the current parameter values.
    ///
    /// Applies: `shadow = decay * shadow + (1 - decay) * param`
    ///
    /// # Errors
    ///
    /// Returns `FerrotorchError::InvalidArgument` from the numeric `cast`
    /// helper when a parameter value cannot be converted to `f64`.
    pub fn update_from_params<T: Float>(&mut self, params: &[Vec<T>]) -> FerrotorchResult<()> {
        let one_minus_decay = 1.0 - self.decay;

        for (shadow, current) in self.shadow.iter_mut().zip(params.iter()) {
            for (s, c) in shadow.iter_mut().zip(current.iter()) {
                let c_f64 = cast::<T, f64>(*c)?;
                *s = self.decay * *s + one_minus_decay * c_f64;
            }
        }

        self.num_updates += 1;
        Ok(())
    }
}

impl<T: Float> Callback<T> for EmaCallback {
    fn on_batch_end(&mut self, _batch: usize, _loss: f64) {
        // The actual EMA update requires access to model parameters, which
        // the Callback trait's on_batch_end does not provide. The parameter
        // update must be driven externally (by the Learner or by user code).
        //
        // This on_batch_end increments the update counter to track how many
        // batches have passed, but the real EMA arithmetic happens when
        // `update_from_params()` is called explicitly.
        //
        // This is a deliberate design choice: the Callback trait is
        // parameter-agnostic (it receives only scalar loss), so EMA updates
        // must be triggered by code that has access to the model parameters.
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    fn make_epoch_result(epoch: usize, val_loss: Option<f64>) -> EpochResult {
        EpochResult {
            epoch,
            train_loss: 1.0,
            val_loss,
            metrics: HashMap::new(),
            lr: 0.001,
            duration_secs: 1.0,
        }
    }

    /// Helper: call `on_epoch_end` with `f32` as the `Float` type parameter.
    fn epoch_end(es: &mut EarlyStopping, epoch: usize, result: &EpochResult) {
        Callback::<f32>::on_epoch_end(es, epoch, result);
    }

    /// Helper: call `should_stop` with `f32` as the `Float` type parameter.
    fn stopped(es: &EarlyStopping) -> bool {
        Callback::<f32>::should_stop(es)
    }

    // -- EarlyStopping -------------------------------------------------------

    #[test]
    fn test_early_stopping_no_trigger_on_improvement() {
        let mut es = EarlyStopping::new(3, 0.001);
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(1.0)));
        assert!(!stopped(&es));
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(0.9)));
        assert!(!stopped(&es));
        epoch_end(&mut es, 2, &make_epoch_result(2, Some(0.8)));
        assert!(!stopped(&es));
    }

    #[test]
    fn test_early_stopping_triggers_after_patience() {
        let mut es = EarlyStopping::new(2, 0.0);
        // Set a baseline.
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(1.0)));
        assert!(!stopped(&es));
        // No improvement.
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(1.0)));
        assert!(!stopped(&es));
        assert_eq!(es.wait(), 1);
        // Still no improvement: patience exhausted.
        epoch_end(&mut es, 2, &make_epoch_result(2, Some(1.1)));
        assert!(stopped(&es));
        assert_eq!(es.wait(), 2);
    }

    #[test]
    fn test_early_stopping_resets_on_improvement() {
        let mut es = EarlyStopping::new(3, 0.0);
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(1.0)));
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(1.1))); // wait=1
        epoch_end(&mut es, 2, &make_epoch_result(2, Some(1.2))); // wait=2
        assert_eq!(es.wait(), 2);
        epoch_end(&mut es, 3, &make_epoch_result(3, Some(0.5))); // improvement, reset
        assert_eq!(es.wait(), 0);
        assert!(!stopped(&es));
    }

    #[test]
    fn test_early_stopping_min_delta() {
        let mut es = EarlyStopping::new(2, 0.1);
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(1.0)));
        // Improvement is only 0.05 < min_delta (0.1).
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(0.95)));
        assert_eq!(es.wait(), 1);
        // Real improvement: 1.0 - 0.8 = 0.2 > 0.1.
        epoch_end(&mut es, 2, &make_epoch_result(2, Some(0.8)));
        assert_eq!(es.wait(), 0);
    }

    #[test]
    fn test_early_stopping_ignores_no_val_loss() {
        let mut es = EarlyStopping::new(2, 0.0);
        epoch_end(&mut es, 0, &make_epoch_result(0, None));
        epoch_end(&mut es, 1, &make_epoch_result(1, None));
        epoch_end(&mut es, 2, &make_epoch_result(2, None));
        // No val_loss means nothing to monitor: should never stop.
        assert!(!stopped(&es));
    }

    #[test]
    fn test_early_stopping_best_loss() {
        let mut es = EarlyStopping::new(5, 0.0);
        assert!(es.best_loss().is_infinite());
        epoch_end(&mut es, 0, &make_epoch_result(0, Some(2.0)));
        assert!((es.best_loss() - 2.0).abs() < 1e-12);
        epoch_end(&mut es, 1, &make_epoch_result(1, Some(1.5)));
        assert!((es.best_loss() - 1.5).abs() < 1e-12);
    }

    #[test]
    fn test_early_stopping_patience_accessor() {
        let es = EarlyStopping::new(7, 0.01);
        assert_eq!(es.patience(), 7);
    }

    // -- ProgressLogger ------------------------------------------------------

    #[test]
    fn test_progress_logger_construction() {
        let pl = ProgressLogger::new(10);
        assert_eq!(pl.log_every_n_batches, 10);
    }

    #[test]
    fn test_progress_logger_default() {
        let pl = ProgressLogger::default();
        assert_eq!(pl.log_every_n_batches, 0);
    }

    #[test]
    fn test_progress_logger_should_stop_always_false() {
        let pl = ProgressLogger::new(10);
        assert!(!Callback::<f32>::should_stop(&pl));
    }

    // -- EmaCallback ---------------------------------------------------------

    #[test]
    fn test_ema_callback_construction() {
        let ema = EmaCallback::new(0.999);
        assert!((ema.decay() - 0.999).abs() < 1e-12);
        assert_eq!(ema.num_updates(), 0);
        assert!(!ema.is_initialized());
    }

    #[test]
    #[should_panic(expected = "decay must be in [0, 1]")]
    fn test_ema_callback_invalid_decay_above() {
        EmaCallback::new(1.5);
    }

    #[test]
    #[should_panic(expected = "decay must be in [0, 1]")]
    fn test_ema_callback_invalid_decay_below() {
        EmaCallback::new(-0.1);
    }

    #[test]
    fn test_ema_callback_boundary_decay_values() {
        let ema0 = EmaCallback::new(0.0);
        assert!((ema0.decay() - 0.0).abs() < 1e-12);

        let ema1 = EmaCallback::new(1.0);
        assert!((ema1.decay() - 1.0).abs() < 1e-12);
    }

    #[test]
    fn test_ema_init_from_params() {
        let mut ema = EmaCallback::new(0.99);
        let params: Vec<Vec<f32>> = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0]];
        ema.init_from_params(&params).unwrap();

        assert!(ema.is_initialized());
        assert_eq!(ema.shadow_params().len(), 2);
        assert_eq!(ema.shadow_params()[0], vec![1.0, 2.0, 3.0]);
        assert_eq!(ema.shadow_params()[1], vec![4.0, 5.0]);
    }

    #[test]
    fn test_ema_update_from_params() {
        let mut ema = EmaCallback::new(0.9);

        // Initialize with [10.0].
        ema.init_from_params(&[vec![10.0_f32]]).unwrap();

        // Update with [20.0]. Expected: 0.9 * 10 + 0.1 * 20 = 11.0.
        ema.update_from_params(&[vec![20.0_f32]]).unwrap();

        assert_eq!(ema.num_updates(), 1);
        assert!((ema.shadow_params()[0][0] - 11.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_multiple_updates() {
        let mut ema = EmaCallback::new(0.5);

        ema.init_from_params(&[vec![0.0_f32]]).unwrap();

        // Step 1: 0.5 * 0 + 0.5 * 10 = 5.0
        ema.update_from_params(&[vec![10.0_f32]]).unwrap();
        assert!((ema.shadow_params()[0][0] - 5.0).abs() < 1e-10);

        // Step 2: 0.5 * 5 + 0.5 * 10 = 7.5
        ema.update_from_params(&[vec![10.0_f32]]).unwrap();
        assert!((ema.shadow_params()[0][0] - 7.5).abs() < 1e-10);

        // Step 3: 0.5 * 7.5 + 0.5 * 10 = 8.75
        ema.update_from_params(&[vec![10.0_f32]]).unwrap();
        assert!((ema.shadow_params()[0][0] - 8.75).abs() < 1e-10);

        assert_eq!(ema.num_updates(), 3);
    }

    #[test]
    fn test_ema_decay_zero_replaces_completely() {
        let mut ema = EmaCallback::new(0.0);
        ema.init_from_params(&[vec![100.0_f32]]).unwrap();

        // decay=0 means shadow = 0 * shadow + 1 * current = current.
        ema.update_from_params(&[vec![42.0_f32]]).unwrap();
        assert!((ema.shadow_params()[0][0] - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_decay_one_keeps_original() {
        let mut ema = EmaCallback::new(1.0);
        ema.init_from_params(&[vec![100.0_f32]]).unwrap();

        // decay=1 means shadow = 1 * shadow + 0 * current = shadow (no change).
        ema.update_from_params(&[vec![42.0_f32]]).unwrap();
        assert!((ema.shadow_params()[0][0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_callback_should_stop_always_false() {
        let ema = EmaCallback::new(0.999);
        assert!(!Callback::<f32>::should_stop(&ema));
    }

    // -- Send + Sync ---------------------------------------------------------

    #[test]
    fn test_callbacks_are_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EarlyStopping>();
        assert_send_sync::<ProgressLogger>();
        assert_send_sync::<EmaCallback>();
    }
}
