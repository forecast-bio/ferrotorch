# ferrotorch-optim

Optimizers and learning rate schedulers for ferrotorch.

## What it provides

- **Optimizer trait** -- `Optimizer` with `step`, `zero_grad`, and `ParamGroup` support
- **Optimizers**:
  - `SGD` (with momentum, weight decay, Nesterov)
  - `Adam`, `AdamW` (with in-place parameter updates for performance)
  - `Adagrad`
  - `RMSprop`
  - `L-BFGS` (with line search)
  - `Muon`
- **Gradient clipping** -- `clip_grad_norm_`, `clip_grad_value_` utilities
- **Learning rate schedulers** -- `StepLR`, `CosineAnnealingLR`, `ReduceLROnPlateau`, `LinearWarmup`, `SequentialLr`, `cosine_warmup_scheduler`
- **Gradient accumulation** -- `GradientAccumulator` for micro-batching with optimized accumulation
- **Gradient scaling** -- `GradScaler` for mixed-precision training

## Quick start

```rust
use ferrotorch_optim::{Adam, AdamConfig, Optimizer};

let config = AdamConfig { lr: 1e-3, ..Default::default() };
let mut optimizer = Adam::new(model.parameters_mut(), config);

// Training loop
optimizer.zero_grad();
let loss = model.forward(&input)?;
loss.backward()?;
optimizer.step()?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
