# ferrotorch-distributions

Probability distributions for ferrotorch -- differentiable sampling and log-probabilities.

## What it provides

- **`Distribution<T>` trait** -- `sample`, `rsample` (reparameterized), `log_prob`, and `entropy`
- **Normal** -- Gaussian distribution with `loc` and `scale` (reparameterized)
- **Uniform** -- continuous uniform distribution with `low` and `high` (reparameterized)
- **Bernoulli** -- binary distribution with `probs` (discrete, no reparameterization)
- **Categorical** -- multinomial distribution with `probs` (discrete, no reparameterization)

Reparameterized distributions allow gradients to flow through `rsample`, enabling variational inference (VAE, policy gradient, etc.).

## Quick start

```rust
use ferrotorch_distributions::{Normal, Distribution};

let dist = Normal::new(0.0_f32, 1.0)?;
let samples = dist.rsample(&[32, 64])?;   // gradient flows through
let log_p = dist.log_prob(&samples)?;
let h = dist.entropy()?;
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
