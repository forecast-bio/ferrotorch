# ferrotorch-optim

Optimizers and learning rate schedulers for ferrotorch.

## What it provides

The `Optimizer<T>` trait with `step`, `zero_grad`, `lr`/`set_lr`, parameter groups, and `state_dict`/`load_state_dict` for checkpointing.

### Optimizers

| Optimizer        | Notes                                                               |
|------------------|---------------------------------------------------------------------|
| `Sgd`            | Momentum, dampening, Nesterov, weight decay, maximize, **foreach**  |
| `Adam`           | Standard Adam with bias correction                                  |
| `AdamW`          | Adam + decoupled weight decay, **foreach** mode                     |
| `Adamax`         | Adam variant using L-infinity norm                                  |
| `NAdam`          | Nesterov-accelerated Adam                                           |
| `RAdam`          | Rectified Adam with warmup-free variance correction                 |
| `Adagrad`        | Per-parameter learning rate accumulation                            |
| `Adadelta`       | Adagrad with running window                                         |
| `Adafactor`      | Memory-efficient Adam (row/column factorization)                    |
| `RMSprop`        | RMSprop with momentum                                               |
| `Rprop`          | Resilient backprop                                                  |
| `ASGD`           | Averaged SGD                                                        |
| `SparseAdam`     | Adam variant for sparse gradients                                   |
| `LBFGS`          | L-BFGS with Strong Wolfe line search                                |
| `Muon`           | Momentum + Newton-Schulz orthogonalization                          |
| `NaturalGradient`| K-FAC approximation (Kronecker-factored Fisher)                     |
| `Ema`            | Exponential moving average of parameters                            |
| `Swa`            | Stochastic Weight Averaging                                         |

### Fused / foreach mode (CL-388)

`Sgd` and `AdamW` support a `foreach: true` config flag that routes the
parameter update through GPU-aware tensor ops instead of the legacy
CPU `f64` workspace path. On CUDA this avoids the per-step
`data_vec()` round-trip. Moment buffers live on the parameter's native
device. Parity tests verify identical results to the legacy path.

### Gradient utilities

- **Gradient clipping** — `clip_grad_norm_`, `clip_grad_value_`
- **Gradient accumulation** — `GradientAccumulator` for micro-batching
- **Gradient scaling** — `GradScaler` for mixed-precision training

### Learning rate schedulers

`StepLR`, `MultiStepLR`, `ExponentialLR`, `CosineAnnealingLR`,
`ReduceLROnPlateau`, `LinearWarmup`, `SequentialLr`,
`cosine_warmup_scheduler`, `OneCycleLR`, `CyclicLR`, `PolynomialLR`.

## Quick start

```rust
use ferrotorch_optim::{AdamW, AdamWConfig, Optimizer};

// Standard (legacy CPU path)
let mut opt = AdamW::new(model.parameters(), AdamWConfig::default());

// Foreach mode (on-device tensor-op path, faster on CUDA)
let mut opt = AdamW::new(
    model.parameters(),
    AdamWConfig { foreach: true, ..Default::default() },
);

// Training step
opt.zero_grad()?;
let loss = model.forward(&input)?;
loss.backward()?;
opt.step()?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
