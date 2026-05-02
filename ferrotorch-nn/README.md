# ferrotorch-nn

Neural network modules for ferrotorch — layers, losses, hooks, and parameter management.

## What it provides

- **`Module<T>` trait** — `forward`, `parameters` / `parameters_mut`, `train`/`eval`, `state_dict` / `load_state_dict`
- **`#[derive(Module)]`** proc macro — annotate fields with `#[param]`, `#[submodule]`, `#[skip]` and parameter collection + train/eval propagation are generated for you

### Layers

| Category        | Modules                                                                  |
|-----------------|--------------------------------------------------------------------------|
| Linear          | `Linear`, `LazyLinear`, `Bilinear`, `LoRALinear`                         |
| Convolution     | `Conv1d`, `Conv2d`, `Conv3d`, `ConvTranspose1d`/`2d`/`3d`, `LazyConv1d`/`2d`/`3d` |
| Buffer          | `Buffer<T>` non-trainable persistent state with state_dict round-trip    |
| Pooling         | `MaxPool1d`/`2d`/`3d`, `AvgPool1d`/`2d`/`3d`, `AdaptiveAvgPool1d`/`2d`/`3d`, `AdaptiveMaxPool*` |
| Normalization   | `BatchNorm1d`/`2d`/`3d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `InstanceNorm*`, `LocalResponseNorm` |
| Activation      | `ReLU`, `LeakyReLU`, `PReLU`, `ELU`, `GELU`, `SiLU`, `Mish`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, `Softplus`, `Hardswish`, `Hardsigmoid` |
| Recurrent       | `LSTM`, `LSTMCell`, `GRU`, `GRUCell`, `RNN`, `RNNCell`                   |
| Transformer     | `MultiheadAttention`, `TransformerEncoderLayer`, `TransformerDecoderLayer`, `RotaryPositionEmbedding`, `KVCache`, `SwiGLU` |
| Embedding       | `Embedding`, `EmbeddingBag`                                              |
| Dropout         | `Dropout`, `Dropout1d`/`2d`/`3d`, `AlphaDropout`                         |
| Container       | `Sequential`, `ModuleList`, `ModuleDict`                                 |
| Utility         | `Flatten`, `Unflatten`, `Identity`, `PixelShuffle`, `PixelUnshuffle`     |

### Losses

`CrossEntropyLoss`, `NLLLoss`, `BCELoss`, `BCEWithLogitsLoss`, `MSELoss`,
`L1Loss`, `HuberLoss`, `SmoothL1Loss`, `KLDivLoss`, `CosineEmbeddingLoss`,
`MarginRankingLoss`, `TripletMarginLoss`, `HingeEmbeddingLoss`,
`MultiLabelMarginLoss`, `PoissonNLLLoss`, `GaussianNLLLoss`,
`CTCLoss`.

### Initialization

`init::xavier_uniform_`, `init::xavier_normal_`, `init::kaiming_uniform_`,
`init::kaiming_normal_`, `init::orthogonal_`, `init::uniform_`,
`init::normal_`, `init::constant_`, `init::zeros_`, `init::ones_`,
with `NonLinearity` enum for gain selection.

### Hooks

`ForwardHook`, `BackwardHook`, `ForwardPreHook`, `HookedModule` for
inspection, debugging, and gradient surgery.

## Quick start

```rust
use ferrotorch_core::{Float, Tensor};
use ferrotorch_nn::{Linear, Module, ReLU, Sequential};

let model: Sequential<f32> = Sequential::new(vec![
    Box::new(Linear::new(784, 256, /* bias */ true)?),
    Box::new(ReLU::default()),
    Box::new(Linear::new(256, 10, true)?),
]);

let x = ferrotorch_core::randn(&[32, 784])?;
let out = model.forward(&x)?;
assert_eq!(out.shape(), &[32, 10]);
```

## With `#[derive(Module)]`

```rust
use ferrotorch_nn::{Linear, Module, ReLU};

#[derive(Module)]
struct MLP<T: ferrotorch_core::Float> {
    #[submodule] fc1: Linear<T>,
    #[submodule] act: ReLU,
    #[submodule] fc2: Linear<T>,
}

impl<T: ferrotorch_core::Float> MLP<T> {
    fn new(d_in: usize, d_hidden: usize, d_out: usize) -> ferrotorch_core::FerrotorchResult<Self> {
        Ok(Self {
            fc1: Linear::new(d_in, d_hidden, true)?,
            act: ReLU::default(),
            fc2: Linear::new(d_hidden, d_out, true)?,
        })
    }
}
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
