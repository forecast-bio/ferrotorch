# ferrotorch-train

Training loop, metrics, and callbacks for ferrotorch.

## What it provides

- **`Learner`** -- high-level training loop that ties together model, optimizer, loss, metrics, and callbacks
- **Metrics** -- `LossMetric`, `AccuracyMetric`, `TopKAccuracy`, `RunningAverage`, and the `Metric` trait
- **Callbacks** -- `EarlyStopping`, `ProgressLogger`, and the `Callback` trait for hooking into epoch/batch boundaries
- **`TrainingHistory`** -- record of per-epoch results with `EpochResult` and `EvalResult`
- **`LossFn`** -- loss function abstraction for the training loop

## Quick start

```rust
use ferrotorch_train::{Learner, LossMetric, EarlyStopping, ProgressLogger};

let mut learner = Learner::new(model, optimizer, loss_fn)
    .with_train_metric(Box::new(LossMetric::new()))
    .with_callback(Box::new(EarlyStopping::new(5, 0.001)))
    .with_callback(Box::new(ProgressLogger::new(100)));

let history = learner.fit(&train_loader, Some(&val_loader), 50)?;
println!("Best val loss: {:?}", history.best_val_loss());
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
