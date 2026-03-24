//! The [`Learner`] — a high-level training loop abstraction.
//!
//! `Learner` owns a model, optimizer, optional scheduler, metrics, and
//! callbacks. It orchestrates the training loop: forward pass, loss
//! computation, backward pass, optimizer step, metric tracking, and
//! callback dispatch.
//!
//! # Examples
//!
//! ```ignore
//! use ferrotorch_train::Learner;
//!
//! let learner = Learner::new(model, optimizer, loss_fn)
//!     .with_scheduler(scheduler)
//!     .with_metric(Box::new(AccuracyMetric::new()))
//!     .with_callback(Box::new(EarlyStopping::new(5, 0.001)))
//!     .with_callback(Box::new(ProgressLogger::new(100)));
//!
//! let history = learner.fit(&train_loader, Some(&val_loader), 100)?;
//! ```

use std::path::PathBuf;
use std::time::Instant;

use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Module;
use ferrotorch_optim::Optimizer;
use ferrotorch_optim::scheduler::LrScheduler;
use ferrotorch_serialize::{TrainingCheckpoint, load_checkpoint, save_checkpoint};

use crate::callback::Callback;
use crate::history::{EpochResult, EvalResult, TrainingHistory};
use crate::metric::Metric;

// ---------------------------------------------------------------------------
// LossFn
// ---------------------------------------------------------------------------

/// A loss function that takes `(prediction, target)` and returns a scalar loss.
///
/// This is a simple function-pointer type alias rather than a trait object so
/// that any closure or function with the right signature works.
pub type LossFn<T> =
    Box<dyn Fn(&Tensor<T>, &Tensor<T>) -> FerrotorchResult<Tensor<T>> + Send + Sync>;

// ---------------------------------------------------------------------------
// Learner
// ---------------------------------------------------------------------------

/// A high-level training loop abstraction.
///
/// `Learner` combines a model, optimizer, loss function, and optional
/// scheduler, metrics, and callbacks into a single struct that drives the
/// training loop via [`fit`](Learner::fit).
pub struct Learner<M, T: Float> {
    model: M,
    optimizer: Box<dyn Optimizer<T>>,
    loss_fn: LossFn<T>,
    scheduler: Option<Box<dyn LrScheduler<T>>>,
    train_metrics: Vec<Box<dyn Metric<Input = f64>>>,
    val_metrics: Vec<Box<dyn Metric<Input = f64>>>,
    callbacks: Vec<Box<dyn Callback<T>>>,
    checkpoint_dir: Option<PathBuf>,
    epoch: usize,
    step: usize,
}

impl<M: Module<T>, T: Float> Learner<M, T> {
    /// Create a new `Learner`.
    ///
    /// # Arguments
    ///
    /// * `model` - The neural network module to train.
    /// * `optimizer` - The optimizer that updates model parameters.
    /// * `loss_fn` - A function `(prediction, target) -> scalar_loss`.
    pub fn new(model: M, optimizer: Box<dyn Optimizer<T>>, loss_fn: LossFn<T>) -> Self {
        Self {
            model,
            optimizer,
            loss_fn,
            scheduler: None,
            train_metrics: Vec::new(),
            val_metrics: Vec::new(),
            callbacks: Vec::new(),
            checkpoint_dir: None,
            epoch: 0,
            step: 0,
        }
    }

    /// Attach a learning rate scheduler.
    pub fn with_scheduler(mut self, scheduler: Box<dyn LrScheduler<T>>) -> Self {
        self.scheduler = Some(scheduler);
        self
    }

    /// Add a training metric.
    ///
    /// Training metrics receive the scalar loss value (as `f64`) after each
    /// batch and are summarized at the end of each epoch.
    pub fn with_train_metric(mut self, metric: Box<dyn Metric<Input = f64>>) -> Self {
        self.train_metrics.push(metric);
        self
    }

    /// Add a validation metric.
    pub fn with_val_metric(mut self, metric: Box<dyn Metric<Input = f64>>) -> Self {
        self.val_metrics.push(metric);
        self
    }

    /// Add a training callback.
    pub fn with_callback(mut self, callback: Box<dyn Callback<T>>) -> Self {
        self.callbacks.push(callback);
        self
    }

    /// Enable checkpointing to the given directory.
    ///
    /// When set, the [`Learner`] will save model state dicts at the end of
    /// each epoch.
    pub fn with_checkpointing(mut self, dir: PathBuf) -> Self {
        self.checkpoint_dir = Some(dir);
        self
    }

    /// Return a reference to the model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Return a mutable reference to the model.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// Return the current epoch counter.
    pub fn epoch(&self) -> usize {
        self.epoch
    }

    /// Return the current global step counter.
    pub fn step(&self) -> usize {
        self.step
    }

    /// Load a training checkpoint from disk, restoring model weights,
    /// optimizer state, and the epoch/step counters.
    ///
    /// The model's `load_state_dict` is called with `strict = true`.
    pub fn load_checkpoint(&mut self, path: impl AsRef<std::path::Path>) -> FerrotorchResult<()> {
        let ckpt: TrainingCheckpoint<T> = load_checkpoint(path)?;
        self.model.load_state_dict(&ckpt.model_state, true)?;
        self.optimizer.load_state_dict(&ckpt.optimizer_state)?;
        self.epoch = ckpt.epoch;
        self.step = ckpt.step;
        Ok(())
    }

    /// Run the training loop.
    ///
    /// Trains for `num_epochs` epochs on `train_data`. If `val_data` is
    /// provided, runs an evaluation pass after each training epoch.
    ///
    /// # Arguments
    ///
    /// * `train_data` - Iterator of `(input, target)` batches for training.
    /// * `val_data` - Optional iterator of `(input, target)` batches for
    ///   validation.
    /// * `num_epochs` - Number of epochs to train.
    ///
    /// # Returns
    ///
    /// A [`TrainingHistory`] containing per-epoch results.
    pub fn fit<I, V>(
        &mut self,
        train_data: &dyn Fn() -> I,
        val_data: Option<&dyn Fn() -> V>,
        num_epochs: usize,
    ) -> FerrotorchResult<TrainingHistory>
    where
        I: Iterator<Item = FerrotorchResult<(Tensor<T>, Tensor<T>)>>,
        V: Iterator<Item = FerrotorchResult<(Tensor<T>, Tensor<T>)>>,
    {
        let mut history = TrainingHistory::new();

        for _ in 0..num_epochs {
            let epoch_start = Instant::now();

            // Notify callbacks.
            for cb in &mut self.callbacks {
                cb.on_epoch_start(self.epoch);
            }

            // Reset training metrics.
            for m in &mut self.train_metrics {
                m.reset();
            }

            // ── Training phase ──────────────────────────────────────────
            self.model.train();
            let mut train_loss_sum = 0.0_f64;
            let mut train_batch_count = 0_usize;

            for (batch_idx, batch_result) in train_data().enumerate() {
                let (input, target) = batch_result?;

                // Notify callbacks.
                for cb in &mut self.callbacks {
                    cb.on_batch_start(batch_idx);
                }

                // Forward.
                let output = self.model.forward(&input)?;
                let loss = (self.loss_fn)(&output, &target)?;
                let loss_val = loss.item()?.to_f64().unwrap();

                // Backward.
                loss.backward()?;

                // Optimizer step.
                self.optimizer.step()?;
                self.optimizer.zero_grad()?;

                // Track loss.
                train_loss_sum += loss_val;
                train_batch_count += 1;
                self.step += 1;

                // Update training metrics.
                for m in &mut self.train_metrics {
                    m.update(&loss_val);
                }

                // Notify callbacks.
                for cb in &mut self.callbacks {
                    cb.on_batch_end(batch_idx, loss_val);
                }
            }

            // Scheduler step (per-epoch).
            if let Some(ref mut sched) = self.scheduler {
                sched.step(self.optimizer.as_mut());
            }

            let train_loss = if train_batch_count > 0 {
                train_loss_sum / train_batch_count as f64
            } else {
                0.0
            };

            // ── Validation phase ────────────────────────────────────────
            let val_loss = if let Some(val_fn) = val_data {
                // Reset validation metrics.
                for m in &mut self.val_metrics {
                    m.reset();
                }

                let eval_result = self.evaluate_iter(val_fn)?;

                Some(eval_result.loss)
            } else {
                None
            };

            // ── Epoch result ────────────────────────────────────────────
            let mut metrics = std::collections::HashMap::new();
            for m in &self.train_metrics {
                metrics.insert(format!("train_{}", m.name()), m.compute());
            }
            for m in &self.val_metrics {
                metrics.insert(format!("val_{}", m.name()), m.compute());
            }

            // Save checkpoint if checkpointing is enabled.
            if let Some(ref dir) = self.checkpoint_dir {
                let _ = std::fs::create_dir_all(dir);
                let checkpoint = TrainingCheckpoint {
                    model_state: self.model.state_dict(),
                    optimizer_state: self.optimizer.state_dict(),
                    epoch: self.epoch,
                    step: self.step,
                };
                let path = dir.join(format!("checkpoint_epoch_{}.ftc", self.epoch));
                let _ = save_checkpoint(&checkpoint, &path);
            }

            let epoch_result = EpochResult {
                epoch: self.epoch,
                train_loss,
                val_loss,
                metrics,
                lr: self.optimizer.lr(),
                duration_secs: epoch_start.elapsed().as_secs_f64(),
            };

            // Notify callbacks.
            for cb in &mut self.callbacks {
                cb.on_epoch_end(self.epoch, &epoch_result);
            }

            history.push(epoch_result);
            self.epoch += 1;

            // Check for early stopping.
            if self.callbacks.iter().any(|cb| cb.should_stop()) {
                break;
            }
        }

        // Notify callbacks of training end.
        for cb in &mut self.callbacks {
            cb.on_train_end(&history);
        }

        Ok(history)
    }

    /// Run an evaluation pass on the given data.
    ///
    /// Sets the model to eval mode, runs forward passes (no gradient),
    /// and returns an [`EvalResult`] with mean loss and metric values.
    pub fn evaluate<V>(&mut self, val_data: &dyn Fn() -> V) -> FerrotorchResult<EvalResult>
    where
        V: Iterator<Item = FerrotorchResult<(Tensor<T>, Tensor<T>)>>,
    {
        // Reset validation metrics.
        for m in &mut self.val_metrics {
            m.reset();
        }

        self.evaluate_iter(val_data)
    }

    /// Internal evaluation helper that does not reset metrics.
    fn evaluate_iter<V>(&mut self, val_data: &dyn Fn() -> V) -> FerrotorchResult<EvalResult>
    where
        V: Iterator<Item = FerrotorchResult<(Tensor<T>, Tensor<T>)>>,
    {
        self.model.eval();

        let mut val_loss_sum = 0.0_f64;
        let mut val_batch_count = 0_usize;

        ferrotorch_core::no_grad(|| -> FerrotorchResult<()> {
            for batch_result in val_data() {
                let (input, target) = batch_result?;
                let output = self.model.forward(&input)?;
                let loss = (self.loss_fn)(&output, &target)?;
                let loss_val = loss.item()?.to_f64().unwrap();

                val_loss_sum += loss_val;
                val_batch_count += 1;

                for m in &mut self.val_metrics {
                    m.update(&loss_val);
                }
            }
            Ok(())
        })?;

        let loss = if val_batch_count > 0 {
            val_loss_sum / val_batch_count as f64
        } else {
            0.0
        };

        let mut metrics = std::collections::HashMap::new();
        for m in &self.val_metrics {
            metrics.insert(m.name().to_string(), m.compute());
        }

        Ok(EvalResult { loss, metrics })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_nn::Parameter;

    // -- Minimal test module -------------------------------------------------

    struct DummyModule<T: Float> {
        weight: Parameter<T>,
        training: bool,
    }

    impl<T: Float> DummyModule<T> {
        fn new() -> FerrotorchResult<Self> {
            Ok(Self {
                weight: Parameter::zeros(&[1])?,
                training: true,
            })
        }
    }

    impl<T: Float> Module<T> for DummyModule<T> {
        fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
            Ok(input.clone())
        }
        fn parameters(&self) -> Vec<&Parameter<T>> {
            vec![&self.weight]
        }
        fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
            vec![&mut self.weight]
        }
        fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
            vec![("weight".to_string(), &self.weight)]
        }
        fn train(&mut self) {
            self.training = true;
        }
        fn eval(&mut self) {
            self.training = false;
        }
        fn is_training(&self) -> bool {
            self.training
        }
    }

    // -- Minimal test optimizer ----------------------------------------------

    struct DummyOptimizer {
        lr: f64,
    }

    impl Optimizer<f32> for DummyOptimizer {
        fn step(&mut self) -> FerrotorchResult<()> {
            Ok(())
        }
        fn zero_grad(&mut self) -> FerrotorchResult<()> {
            Ok(())
        }
        fn lr(&self) -> f64 {
            self.lr
        }
        fn set_lr(&mut self, lr: f64) {
            self.lr = lr;
        }
        fn param_groups(&self) -> &[ferrotorch_optim::ParamGroup<f32>] {
            &[]
        }
        fn param_groups_mut(&mut self) -> &mut [ferrotorch_optim::ParamGroup<f32>] {
            &mut []
        }
        fn add_param_group(&mut self, _group: ferrotorch_optim::ParamGroup<f32>) {}
        fn state_dict(&self) -> ferrotorch_optim::OptimizerState {
            Default::default()
        }
        fn load_state_dict(
            &mut self,
            _state: &ferrotorch_optim::OptimizerState,
        ) -> FerrotorchResult<()> {
            Ok(())
        }
    }

    // -- Construction tests --------------------------------------------------

    #[test]
    fn test_learner_construction() {
        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));

        let learner = Learner::new(model, optimizer, loss_fn);
        assert_eq!(learner.epoch(), 0);
        assert_eq!(learner.step(), 0);
    }

    #[test]
    fn test_learner_with_checkpoint_dir() {
        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));

        let learner = Learner::new(model, optimizer, loss_fn)
            .with_checkpointing(PathBuf::from("/tmp/checkpoints"));

        assert_eq!(
            learner.checkpoint_dir,
            Some(PathBuf::from("/tmp/checkpoints"))
        );
    }

    #[test]
    fn test_learner_with_metrics() {
        use crate::metric::LossMetric;

        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));

        let learner = Learner::new(model, optimizer, loss_fn)
            .with_train_metric(Box::new(LossMetric::new()))
            .with_val_metric(Box::new(LossMetric::new()));

        assert_eq!(learner.train_metrics.len(), 1);
        assert_eq!(learner.val_metrics.len(), 1);
    }

    #[test]
    fn test_learner_with_callbacks() {
        use crate::callback::EarlyStopping;

        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));

        let learner = Learner::new(model, optimizer, loss_fn)
            .with_callback(Box::new(EarlyStopping::new(5, 0.001)));

        assert_eq!(learner.callbacks.len(), 1);
    }

    #[test]
    fn test_learner_model_accessors() {
        let model = DummyModule::<f32>::new().unwrap();
        let optimizer: Box<dyn Optimizer<f32>> = Box::new(DummyOptimizer { lr: 0.01 });
        let loss_fn: LossFn<f32> = Box::new(|pred, _target| Ok(pred.clone()));

        let mut learner = Learner::new(model, optimizer, loss_fn);
        assert!(learner.model().is_training());
        learner.model_mut().eval();
        assert!(!learner.model().is_training());
    }
}
