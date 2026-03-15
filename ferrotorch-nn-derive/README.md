# ferrotorch-nn-derive

Derive macro for the ferrotorch `Module` trait -- auto-generates parameter collection and train/eval boilerplate.

## What it provides

- **`#[derive(Module)]`** -- generates `parameters`, `parameters_mut`, `named_parameters`, `train`, `eval`, and `is_training` methods
- **Field attributes**:
  - `#[param]` -- marks a `Parameter<T>` field for registration
  - `#[submodule]` -- marks a nested `Module<T>` for recursive traversal
  - `#[skip]` -- ignores a field entirely
  - `training: bool` -- required field, managed automatically by the derive

## Quick start

```rust
use ferrotorch_nn::{Module, Parameter, Linear};
use ferrotorch_nn_derive::Module;

#[derive(Module)]
struct MyModel<T: Float> {
    #[param]     weight: Parameter<T>,
    #[submodule] layer1: Linear<T>,
    #[submodule] layer2: Linear<T>,
    #[skip]      hidden_size: usize,
    training: bool,
}

// Only `forward()` needs manual implementation -- all other
// Module methods are derived automatically.
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
