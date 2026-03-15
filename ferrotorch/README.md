# ferrotorch

Top-level re-export crate for the ferrotorch deep learning framework.

## What it provides

This is the umbrella crate that re-exports the most commonly used types from:

- **ferrotorch-core** -- Tensor, autograd, differentiable ops, quantization
- **ferrotorch-nn** -- Module trait, layers, losses, activations
- **ferrotorch-optim** -- Optimizers and learning rate schedulers
- **ferrotorch-data** -- Dataset, DataLoader, samplers, transforms
- **ferrotorch-vision** -- Vision model architectures, datasets, image I/O

## Quick start

```rust
use ferrotorch::*;

let a = scalar(2.0f32)?.requires_grad_(true);
let b = scalar(3.0f32)?.requires_grad_(true);
let c = (&a * &b)?;

c.backward()?;
println!("{}", a.grad()?.unwrap()); // tensor(3.)
```

## Part of ferrotorch

This is the top-level crate of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
