# ferrotorch-distributed

Distributed training for ferrotorch -- backends, collectives, and DDP.

## What it provides

- **Backends** -- `TcpBackend` for real multi-process training, `SimulatedBackend` for in-process testing, and the `Backend` trait
- **Collectives** -- `allreduce`, `broadcast`, `barrier` with `ReduceOp` (Sum, Mean, Min, Max)
- **DDP** -- `DDP` wraps any `Module` and synchronizes gradients across ranks after each backward pass
- **GPU collectives** (requires `gpu` feature) -- `gpu_allreduce`, `gpu_broadcast` for GPU tensor communication

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `gpu`   | no      | Enable GPU-aware collectives via ferrotorch-gpu |

## Quick start

```rust
use ferrotorch_distributed::{TcpBackend, Backend, allreduce, ReduceOp, DDP};

let backend = TcpBackend::init(rank, world_size, &addr)?;
let mut ddp_model = DDP::new(model, &backend)?;

// Training loop -- gradients are synchronized automatically
let loss = ddp_model.forward(&input)?;
backward(&loss)?;
allreduce(&backend, &mut grad_tensor, ReduceOp::Mean)?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
