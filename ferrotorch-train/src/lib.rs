//! Training loop, metrics, and callbacks for ferrotorch.
//!
//! This crate provides the [`Learner`] abstraction that ties together a model,
//! optimizer, loss function, metrics, and callbacks into a complete training
//! pipeline.
//!
//! # Overview
//!
//! | Component | Description |
//! |-----------|-------------|
//! | [`Learner`] | High-level training loop: `fit()`, `evaluate()` |
//! | [`Metric`] | Accumulate batch-level values into epoch-level summaries |
//! | [`Callback`] | Hook into epoch/batch boundaries (early stopping, logging) |
//! | [`TrainingHistory`] | Record of per-epoch results |
//!
//! # Quick start
//!
//! ```ignore
//! use ferrotorch_train::{Learner, LossMetric, EarlyStopping, ProgressLogger};
//!
//! let mut learner = Learner::new(model, optimizer, loss_fn)
//!     .with_train_metric(Box::new(LossMetric::new()))
//!     .with_callback(Box::new(EarlyStopping::new(5, 0.001)))
//!     .with_callback(Box::new(ProgressLogger::new(100)));
//!
//! let history = learner.fit(&train_loader, Some(&val_loader), 50)?;
//! println!("Best val loss: {:?}", history.best_val_loss());
//! ```

pub mod callback;
pub mod history;
pub mod learner;
pub mod metric;
pub mod tensorboard;

pub use callback::{Callback, EarlyStopping, ProgressLogger};
pub use history::{EpochResult, EvalResult, TrainingHistory};
pub use learner::{Learner, LossFn};
pub use metric::{AccuracyMetric, LossMetric, Metric, RunningAverage, TopKAccuracy};
pub use tensorboard::{TensorBoardCallback, TensorBoardWriter};
