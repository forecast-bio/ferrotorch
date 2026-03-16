# ferrotorch

Top-level re-export crate for the ferrotorch deep learning framework.

## What it provides

This is the umbrella crate that re-exports all ferrotorch sub-crates through a single dependency. Add `ferrotorch` to your `Cargo.toml` and access everything via submodules:

### Always included
- **ferrotorch-core** — Tensor, autograd, differentiable ops (40+), quantization, einops
- **ferrotorch-nn** — Module trait, 26+ layers (Linear, Conv1d/2d, LSTM, GRU, Attention, norms, activations), losses, LoRA
- **ferrotorch-optim** — 8 optimizers (SGD, Adam, AdamW, RMSprop, Adagrad, L-BFGS, Muon, K-FAC), gradient clipping, schedulers
- **ferrotorch-data** — Dataset, DataLoader (parallel via rayon), DistributedSampler, collation, transforms
- **ferrotorch-vision** — ResNet, VGG, ViT, Swin, ConvNeXt, EfficientNet, YOLO; MNIST/CIFAR datasets; image I/O

### Default features (opt-out with `default-features = false`)
- **ferrotorch-train** — Learner training loop, callbacks, metrics, checkpointing
- **ferrotorch-serialize** — ONNX export, PyTorch .pt import, safetensors, GGUF
- **ferrotorch-jit** — Tracing JIT, IR graph, optimization passes, code generation
- **ferrotorch-distributions** — Probability distributions for sampling and VI
- **ferrotorch-profiler** — Performance profiling with Chrome trace export
- **ferrotorch-hub** — Model hub for downloading and caching pretrained weights

### Optional features (opt-in)
- **`gpu`** — CUDA backend with hand-written PTX kernels and cuBLAS (`cargo add ferrotorch --features gpu`)
- **`cubecl`** — Portable GPU via CubeCL: CUDA + WGPU/AMD + ROCm (`--features cubecl`)
- **`distributed`** — DDP, collective ops, TCP backend (`--features distributed`)

## Quick start

```rust
use ferrotorch::prelude::*;

fn main() -> FerrotorchResult<()> {
    // Autograd
    let a = scalar(2.0f32)?.requires_grad_(true);
    let b = scalar(3.0f32)?.requires_grad_(true);
    let c = (&a * &b)?;
    c.backward()?;
    println!("{}", a.grad()?.unwrap()); // tensor(3.)

    // Neural network
    let model = Sequential::new(vec![
        Box::new(Linear::new(784, 256, true)?),
        Box::new(ReLU::default()),
        Box::new(Linear::new(256, 10, true)?),
    ]);
    let x = rand::<f32>(&[32, 784])?;
    let out = model.forward(&x)?;
    Ok(())
}
```

## Submodules

```rust
use ferrotorch::nn::*;           // layers, losses, activations
use ferrotorch::optim::*;        // optimizers, schedulers
use ferrotorch::data::*;         // datasets, dataloaders
use ferrotorch::vision::*;       // models, transforms
use ferrotorch::train::*;        // Learner, callbacks
use ferrotorch::serialize::*;    // ONNX, safetensors
use ferrotorch::jit::*;          // tracing, IR
use ferrotorch::distributions::*; // probability distributions
```

## Part of ferrotorch

This is the top-level crate of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
