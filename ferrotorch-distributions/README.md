# ferrotorch-distributions

Probability distributions for ferrotorch — differentiable sampling, log-probabilities, KL divergences, and bijective transforms.

## What it provides

The `Distribution<T>` trait with `sample`, `rsample` (reparameterized), `log_prob`, and `entropy` methods, implemented for 25+ distributions.

### Univariate continuous

| Distribution           | Parameters                       | Reparameterized |
|------------------------|----------------------------------|-----------------|
| `Normal`               | `loc`, `scale`                   | yes             |
| `Uniform`              | `low`, `high`                    | yes             |
| `Beta`                 | `concentration1`, `concentration0` | yes           |
| `Gamma`                | `concentration`, `rate`          | yes             |
| `Exponential`          | `rate`                           | yes             |
| `Laplace`              | `loc`, `scale`                   | yes             |
| `Cauchy`               | `loc`, `scale`                   | yes             |
| `Gumbel`               | `loc`, `scale`                   | yes             |
| `HalfNormal`           | `scale`                          | yes             |
| `LogNormal`            | `loc`, `scale`                   | yes             |
| `StudentT`             | `df`, `loc`, `scale`             | yes             |
| `VonMises`             | `loc`, `concentration`           | no              |
| `Weibull`              | `scale`, `concentration`         | yes             |
| `Pareto`               | `scale`, `alpha`                 | yes             |
| `Kumaraswamy`          | `concentration1`, `concentration0` | yes           |

### Discrete

| Distribution           | Parameters                       | Reparameterized |
|------------------------|----------------------------------|-----------------|
| `Bernoulli`            | `probs`                          | no              |
| `Categorical`          | `probs`                          | no              |
| `OneHotCategorical`    | `probs`                          | no              |
| `Multinomial`          | `total_count`, `probs`           | no              |
| `Poisson`              | `rate`                           | no              |

### Multivariate

| Distribution                    | Parameters                            | Reparameterized |
|---------------------------------|---------------------------------------|-----------------|
| `MultivariateNormal`            | `loc`, `scale_tril` / `cov` / `prec` | yes             |
| `LowRankMultivariateNormal`     | `loc`, `cov_factor`, `cov_diag`      | yes             |
| `Dirichlet`                     | `concentration`                       | yes             |

### Wrappers and transforms

| Type                            | Description                                                  |
|---------------------------------|--------------------------------------------------------------|
| `Independent`                   | Reinterprets rightmost batch dims as event dims              |
| `MixtureSameFamily`             | Finite mixture with same-family components                   |
| `RelaxedBernoulli`              | Concrete (Gumbel-softmax) relaxation of Bernoulli            |
| `RelaxedOneHotCategorical`      | Concrete relaxation of Categorical (simplex-valued samples)  |
| `TransformedDistribution`       | Apply bijective transforms with log-det-Jacobian             |

### Infrastructure

- **`constraints`** — constraint objects for parameter and support validation
- **`transforms`** — bijective transforms (Affine, Sigmoid, Exp, StickBreaking, …)
- **`kl`** — analytical KL divergence registry for same-family distribution pairs

## Quick start

```rust
use ferrotorch_core::Tensor;
use ferrotorch_distributions::{Normal, Distribution};

// Construct from leaf tensors
let loc = Tensor::from_storage(/* ... */)?;   // shape [d]
let scale = Tensor::from_storage(/* ... */)?; // shape [d]

let dist = Normal::new(loc, scale)?;
let samples = dist.rsample(&[32, 64])?;       // gradient flows through
let log_p = dist.log_prob(&samples)?;
let h = dist.entropy()?;
```

## Differentiable variational inference

```rust
use ferrotorch_distributions::{Normal, Independent, Distribution};

// Diagonal Gaussian over a 16-dim latent — log_prob returns shape []
// (single scalar per sample) instead of [16].
let posterior = Independent::new(Normal::new(mu, sigma)?, 1)?;
let z = posterior.rsample(&[batch])?;          // [batch, 16]
let log_q = posterior.log_prob(&z)?;           // [batch]
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
