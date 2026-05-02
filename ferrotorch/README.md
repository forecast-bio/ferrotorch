# ferrotorch

Top-level re-export crate for the ferrotorch deep learning framework.

## What it provides

This is the umbrella crate that re-exports all ferrotorch sub-crates through a single dependency. Add `ferrotorch` to your `Cargo.toml` and access everything via submodules:

### Always included
- **ferrotorch-core** — Tensor, autograd, 80+ differentiable ops, complex / sparse / named tensors, FFT, signal, masked, quantization, einops
- **ferrotorch-nn** — Module trait, 30+ layers (Linear, Conv1d/2d, LSTM, GRU, Attention, norms, activations, lazy modules), losses, LoRA
- **ferrotorch-optim** — 19 optimizers (SGD, Adam, AdamW, Adamax, NAdam, RAdam, Adagrad, Adadelta, Adafactor, RMSprop, Rprop, ASGD, SparseAdam, L-BFGS, Muon, K-FAC, EMA, SWA), 12+ LR schedulers, gradient clipping
- **ferrotorch-data** — Dataset, DataLoader (parallel via rayon), DistributedSampler, collation, transforms, NumPy/Arrow interop
- **ferrotorch-vision** — ResNet, VGG, ViT, Swin, ConvNeXt, EfficientNet, U-Net, YOLO; MNIST/CIFAR/ImageFolder datasets; image I/O

### Default features (opt-out with `default-features = false`)
- **ferrotorch-train** — Learner training loop, callbacks, metrics, checkpointing
- **ferrotorch-serialize** — ONNX export, PyTorch .pt import, safetensors, GGUF
- **ferrotorch-jit** — Tracing JIT, IR graph, optimization passes, code generation
- **ferrotorch-jit-script** — `#[script]` proc macro for source-based graph capture
- **ferrotorch-distributions** — Probability distributions for sampling and VI
- **ferrotorch-profiler** — Performance profiling with Chrome trace export
- **ferrotorch-hub** — Model hub for downloading and caching pretrained weights
- **ferrotorch-tokenize** — HuggingFace tokenizer wrapper (BPE, WordPiece, Unigram)

### Optional features (opt-in)
- **`gpu`** — NVIDIA CUDA backend with PTX kernels, cuBLAS, cuSOLVER, cuFFT (`cargo add ferrotorch --features gpu`)
- **`cubecl`** — Portable GPU via CubeCL: CUDA + WGPU/AMD + ROCm (`--features cubecl`)
- **`mps`** — Apple Silicon Metal Performance Shaders backend (`--features mps`)
- **`xpu`** — Intel Arc / Data Center GPU Max via CubeCL wgpu (`--features xpu`)
- **`distributed`** — DDP, collective ops, TCP / Gloo backends (`--features distributed`)
- **`llama`** — Llama 3 model composition + GPU bf16 inference (`--features llama`)
- **`ml`** — Sklearn-compatible adapter and classic-ML datasets (`--features ml`)

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
use ferrotorch::tokenize::*;     // HF tokenizer wrapper
// feature-gated:
// use ferrotorch::llama::*;     // Llama 3 model + inferencer
// use ferrotorch::ml::*;        // sklearn-compatible adapter
```

## Part of ferrotorch

This is the top-level crate of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
