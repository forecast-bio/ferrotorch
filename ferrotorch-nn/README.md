# ferrotorch-nn

Neural network modules for ferrotorch -- layers, losses, and initialization.

## What it provides

- **Module trait** -- `Module<T>` with `forward`, `parameters`, `train`/`eval`, and `StateDict`
- **`#[derive(Module)]`** -- auto-generates parameter collection and train/eval boilerplate
- **Linear layers** -- `Linear`
- **Convolutions** -- `Conv1d`, `Conv2d`, `ConvTranspose2d`
- **Normalization** -- `LayerNorm`, `BatchNorm2d`, `GroupNorm`, `RMSNorm`
- **Activations** -- `ReLU`, `GELU`, `SiLU`, `Mish`, `Sigmoid`, `Tanh`, `Softmax`, `LogSoftmax`, and more
- **Pooling** -- `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d` (functional and module forms)
- **Recurrent** -- `LSTM`, `GRU`
- **Transformer** -- `TransformerEncoderLayer`, `TransformerDecoderLayer`, `MultiheadAttention`, `RotaryPositionEmbedding`, `KVCache`, `SwiGLU`
- **Utility modules** -- `Flatten`, `Identity`
- **Containers** -- `Sequential`, `ModuleList`, `ModuleDict`
- **Losses** -- `CrossEntropyLoss`, `MSELoss`, `BCEWithLogitsLoss`, `HuberLoss`, `KLDivLoss`, `SmoothL1Loss`, `CosineEmbeddingLoss`
- **Hooks** -- `ForwardHook`, `BackwardHook`, `ForwardPreHook`, `HookedModule`
- **Initialization** -- `init` module with `NonLinearity` for weight init schemes
- **Dropout** -- `Dropout`, `Dropout2d`
- **Embedding** -- `Embedding`
- **Gradient clipping** -- `clip_grad_norm_`, `clip_grad_value_`

## Quick start

```rust
use ferrotorch_core::{tensor, Float};
use ferrotorch_nn::{Linear, Module, Sequential};

fn main() {
    let model = Sequential::new(vec![
        Box::new(Linear::<f32>::new(784, 256, true)),
        Box::new(Linear::<f32>::new(256, 10, true)),
    ]);
    let x = tensor(&[0.0_f32; 784]).reshape(&[1, 784]);
    let out = model.forward(&x).unwrap();
}
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
