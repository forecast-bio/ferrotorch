# ferrotorch

**PyTorch, rewritten from scratch in pure Rust.**

[![crates.io](https://img.shields.io/crates/v/ferrotorch.svg)](https://crates.io/crates/ferrotorch)
[![docs.rs](https://docs.rs/ferrotorch/badge.svg)](https://docs.rs/ferrotorch)
[![license](https://img.shields.io/crates/l/ferrotorch.svg)](https://github.com/dollspace-gay/ferrotorch#license)
[![tests](https://img.shields.io/badge/tests-1%2C350%2B_passing-brightgreen.svg)](#)

---

ferrotorch is a deep learning framework with reverse-mode automatic differentiation, neural network modules, optimizers, GPU acceleration, and a JIT compiler --- all in pure Rust with no C++ dependencies. It provides the same eager-mode, dynamic-graph experience that researchers know from PyTorch, backed by [ferray](https://crates.io/crates/ferray) as its NumPy-equivalent array engine.

If you have ever wanted to train a ResNet or a transformer in Rust without pulling in libtorch, wrapping C++ with FFI, or giving up autograd --- this is it.

## Key Features

- **Pure Rust, no C/C++ FFI** --- the only foreign call is cudarc for the CUDA driver API. Everything else compiles with `cargo build`.
- **Reverse-mode autograd** with 30+ differentiable operations, topological-sort backward pass, gradient accumulation, and checkpointing.
- **Operator overloading** --- write `&a + &b`, `&x * &y`, `-z` with natural Rust syntax. All ownership combinations supported.
- **24+ neural network layers** including Linear, Conv1d/2d, LSTM, MultiheadAttention, BatchNorm, LayerNorm, RMSNorm, and LLM modules (RoPE, SwiGLU, KV cache, TransformerEncoder/DecoderLayer).
- **7 optimizers** --- SGD, Adam, AdamW, RMSprop, Adagrad, L-BFGS, and Muon, with parameter groups and 5 LR schedulers.
- **JIT compiler** --- trace a forward pass into a static IR, then run constant folding, dead code elimination, operator fusion, and memory planning. `compile()` API mirrors `torch.compile`.
- **GPU acceleration** --- NVIDIA via cudarc + cuBLAS (81.8x matmul speedup on RTX 3090), AMD/Intel/Apple via CubeCL (WGPU, ROCm, Vulkan, Metal).
- **SafeTensors + PyTorch .pt import** --- load HuggingFace models directly; pure-Rust pickle parser for PyTorch checkpoints.
- **8 vision model architectures** --- ResNet, VGG, ViT, EfficientNet, ConvNeXt, Swin Transformer, U-Net, YOLO.
- **INT8/INT4 quantization** --- per-tensor and per-channel post-training quantization with quantized matmul.
- **Distributed training** --- DDP with gradient synchronization over a TCP backend, GPU-aware collectives.
- **Training loop** --- `Learner` abstraction with metrics (loss, accuracy, top-k), callbacks (early stopping, progress logging), and training history.
- **`#[derive(Module)]` proc macro** --- annotate fields with `#[param]`, `#[submodule]`, `#[skip]` and the Module trait is implemented for you.
- **Zero-panic guarantee** --- every public function returns `Result<T, FerrotorchError>`.

## Quick Start

```rust
use ferrotorch_core::*;

// Build a small computation graph
let a = scalar(2.0f32)?.requires_grad_(true);
let b = scalar(3.0f32)?.requires_grad_(true);
let c = (&a * &b)?;

// Reverse-mode autodiff
c.backward()?;
println!("{}", a.grad()?.unwrap()); // tensor(3.)
println!("{}", b.grad()?.unwrap()); // tensor(2.)
```

## Training Example

A condensed version of `ferrotorch/examples/train_mnist.rs`:

```rust
use ferrotorch_core::*;
use ferrotorch_nn::*;
use ferrotorch_optim::*;

// 3-layer MLP: 784 -> 128 -> 64 -> 10
let mut model = Sequential::new(vec![
    Box::new(Linear::<f32>::new(784, 128, true)?),
    Box::new(ReLU::default()),
    Box::new(Linear::<f32>::new(128, 64, true)?),
    Box::new(ReLU::default()),
    Box::new(Linear::<f32>::new(64, 10, true)?),
]);

let mut optimizer = Adam::new(
    model.parameters().into_iter().cloned().collect(),
    AdamConfig::default(),
);
let loss_fn = CrossEntropyLoss::new(Reduction::Mean, 0.0);

for epoch in 0..10 {
    for batch in train_loader.iter(epoch) {
        let batch = batch?;
        let logits = model.forward(&input)?;
        let loss = loss_fn.forward(&logits, &target)?;

        optimizer.zero_grad()?;
        loss.backward()?;
        optimizer.step()?;
    }
}
```

Or use the high-level `Learner` API:

```rust
use ferrotorch_train::*;

let mut learner = Learner::new(model, optimizer, loss_fn)
    .with_train_metric(Box::new(LossMetric::new()))
    .with_callback(Box::new(EarlyStopping::new(5, 0.001)))
    .with_callback(Box::new(ProgressLogger::new(100)));

let history = learner.fit(&train_loader, Some(&val_loader), 50)?;
```

## Crate Overview

ferrotorch is a workspace of 12 crates. Use the umbrella crate for convenience, or depend on individual crates for minimal compile times.

| Crate | Description |
|---|---|
| **ferrotorch** | Top-level re-export crate (`cargo add ferrotorch`) |
| **ferrotorch-core** | Tensor, autograd engine, 30+ differentiable ops, quantization |
| **ferrotorch-nn** | Module trait, 24+ layers, losses, activations, `#[derive(Module)]` |
| **ferrotorch-nn-derive** | Proc macro for `#[derive(Module)]` |
| **ferrotorch-optim** | 7 optimizers, 5 LR schedulers, GradScaler for mixed precision |
| **ferrotorch-data** | Dataset, DataLoader, samplers, transforms |
| **ferrotorch-train** | Learner, metrics, callbacks, training history |
| **ferrotorch-vision** | 8 model architectures, MNIST/CIFAR datasets, image I/O |
| **ferrotorch-jit** | Tracing, IR graph, optimization passes, codegen backends |
| **ferrotorch-serialize** | SafeTensors, PyTorch .pt import, training checkpoints |
| **ferrotorch-gpu** | NVIDIA CUDA backend via cudarc + cuBLAS |
| **ferrotorch-cubecl** | Portable GPU via CubeCL (NVIDIA, AMD, Intel, Apple) |
| **ferrotorch-distributed** | DDP, allreduce, broadcast, TCP backend |

## GPU Support

ferrotorch supports GPU acceleration through two backends:

### NVIDIA (ferrotorch-gpu)

Uses [cudarc](https://crates.io/crates/cudarc) for safe Rust bindings to the CUDA driver API, with cuBLAS for matmul/GEMM and custom PTX kernels for elementwise ops. Includes a caching memory allocator modeled after PyTorch's `CUDACachingAllocator`.

```rust
let x = tensor.cuda()?;          // Move to GPU 0
let y = x.matmul(&weights)?;     // cuBLAS GEMM
let z = y.cpu()?;                // Move back
```

### AMD / Intel / Apple (ferrotorch-cubecl)

Uses [CubeCL](https://crates.io/crates/cubecl) to compile a single kernel definition to multiple backends:

| Feature flag | Backend | GPU vendors |
|---|---|---|
| `cuda` | NVIDIA CUDA via PTX | NVIDIA |
| `wgpu` | WGPU (Vulkan / Metal / DX12) | AMD, Intel, Apple |
| `rocm` | AMD HIP (native) | AMD |

```rust
use ferrotorch_cubecl::CubeRuntime;

if let Some(rt) = CubeRuntime::auto() {
    println!("Using device: {:?}", rt.device());
}
```

## Model Zoo

Pre-built architectures in ferrotorch-vision, ready to use or fine-tune:

| Architecture | Variants | Task |
|---|---|---|
| **ResNet** | 18, 34, 50 | Image classification |
| **VGG** | 11, 16 | Image classification |
| **Vision Transformer (ViT)** | B/16 | Image classification |
| **EfficientNet** | B0 | Efficient mobile classification |
| **ConvNeXt** | Tiny | Modern ConvNet |
| **Swin Transformer** | Tiny | Hierarchical vision transformer |
| **U-Net** | --- | Semantic segmentation |
| **YOLO** | --- | Object detection |

```rust
use ferrotorch_vision::models::{list_models, get_model};

for name in list_models() {
    println!("{name}");
}
let resnet = get_model::<f32>("resnet50", 1000)?;
```

## Comparison with Alternatives

| | ferrotorch | PyTorch | Burn | tch-rs | candle |
|---|---|---|---|---|---|
| **Language** | Rust | Python/C++ | Rust | Rust (C++ FFI) | Rust |
| **C++ dependency** | None | libtorch | None | libtorch | None |
| **Autograd** | Reverse-mode, dynamic graph | Reverse-mode, dynamic graph | Reverse-mode | Via libtorch | Forward ops only |
| **GPU** | CUDA + CubeCL (AMD/Intel/Apple) | CUDA + ROCm | CubeCL | Via libtorch | CUDA |
| **Eager mode** | Yes | Yes | Yes | Yes | Yes |
| **JIT / compile** | Tracing + fusion + codegen | TorchScript / torch.compile | No | Via libtorch | No |
| **Distributed** | DDP | DDP / FSDP / Pipeline | No | Via libtorch | Partial |
| **Quantization** | INT8 / INT4 | INT8 / INT4 / FP8 | INT8 | Via libtorch | GGUF |
| **Model zoo** | 8 architectures | Thousands | Limited | Via libtorch | LLM-focused |
| **Training loop** | Learner + callbacks | Manual / Lightning | Learner | Manual | Manual |
| **Proc macro** | `#[derive(Module)]` | No (dynamic) | `#[derive(Module)]` | No | No |

## Installation

Add ferrotorch to your project:

```sh
cargo add ferrotorch
```

Or pick individual crates:

```sh
cargo add ferrotorch-core ferrotorch-nn ferrotorch-optim
```

For GPU support:

```sh
# NVIDIA CUDA
cargo add ferrotorch-gpu

# Portable GPU (AMD, Intel, Apple)
cargo add ferrotorch-cubecl --features wgpu
```

## Minimum Supported Rust Version

Rust **1.85+** (Edition 2024).

ferrotorch tracks the latest stable Rust edition. The MSRV will only be bumped in minor version releases.

## License

Licensed under either of

- [Apache License, Version 2.0](LICENSE-APACHE)
- [MIT License](LICENSE-MIT)

at your option.

## Contributing

Contributions are welcome. Whether it is a bug report, a new layer, an optimizer, a model architecture, or a performance improvement --- open an issue or submit a pull request at [github.com/dollspace-gay/ferrotorch](https://github.com/dollspace-gay/ferrotorch).

If you are unsure where to start, look for issues labeled **good first issue** or check the [CHANGELOG](CHANGELOG.md) for recently shipped features that could use more tests or documentation.
