# ferrotorch

**PyTorch, rewritten from scratch in pure Rust.**

[![crates.io](https://img.shields.io/crates/v/ferrotorch.svg)](https://crates.io/crates/ferrotorch)
[![docs.rs](https://docs.rs/ferrotorch/badge.svg)](https://docs.rs/ferrotorch)
[![license](https://img.shields.io/crates/l/ferrotorch.svg)](https://github.com/dollspace-gay/ferrotorch#license)
[![tests](https://img.shields.io/badge/tests-4%2C800%2B_passing-brightgreen.svg)](#)

---

ferrotorch is a deep learning framework with reverse-mode automatic differentiation, neural network modules, optimizers, GPU acceleration, and a JIT compiler --- all in pure Rust with no C++ dependencies. It provides the same eager-mode, dynamic-graph experience that researchers know from PyTorch, backed by [ferray](https://crates.io/crates/ferray) as its NumPy-equivalent array engine.

If you have ever wanted to train a ResNet or a transformer in Rust without pulling in libtorch, wrapping C++ with FFI, or giving up autograd --- this is it.

## Key Features

- **Pure Rust, no C/C++ FFI** --- the only foreign call is cudarc for the CUDA driver API. Everything else compiles with `cargo build`.
- **Reverse-mode autograd** with 80+ differentiable operations (including exp, log, sin, cos, clamp, FFT, eigh, signal processing), topological-sort backward pass, gradient accumulation, broadcast gradient reduction, checkpointing, sparse gradients, and `backward_with_gradient()` for non-scalar tensors.
- **Operator overloading** --- write `&a + &b`, `&x * &y`, `-z` with natural Rust syntax. All ownership combinations supported.
- **30+ neural network layers** including Linear, Conv1d/2d, LSTM, GRU, MultiheadAttention, BatchNorm, LayerNorm, RMSNorm, Flatten, Identity, lazy modules (`LazyLinear`, `LazyConv2d`), and LLM modules (RoPE, SwiGLU, KV cache, TransformerEncoder/DecoderLayer).
- **19 optimizers** --- SGD, Adam, AdamW, Adamax, NAdam, RAdam, Adagrad, Adadelta, Adafactor, RMSprop, Rprop, ASGD, SparseAdam, L-BFGS, Muon, NaturalGradient (K-FAC), EMA, SWA — with parameter groups, foreach (on-device) update mode for SGD/AdamW, 12+ LR schedulers, and gradient clipping (`clip_grad_norm`, `clip_grad_value`).
- **JIT compiler** --- both `trace` (capture-based) and `script` (source-based, in `ferrotorch-jit-script`) frontends. Lower into a static IR, then run constant folding, dead code elimination, operator fusion, and memory planning. `compile()` API mirrors `torch.compile`.
- **GPU acceleration** --- unified device-aware tensors (`tensor.cuda()`, `model.to_device(Device::Cuda(0))`) with auto-dispatch to CPU or GPU, f32/f64 dispatch. NVIDIA via cudarc + cuBLAS (81.8x matmul speedup on RTX 3090), AMD/Intel/Apple via CubeCL (WGPU, ROCm, Vulkan, Metal), Apple Silicon native via ferrotorch-mps, Intel Arc via ferrotorch-xpu. No separate `GpuTensor` type.
- **Llama 3 inference stack** --- full GQA + RoPE + SwiGLU decoder, KV cache, GPU bf16 inference, HuggingFace SafeTensors loader, GPTQ/AWQ/HQQ quantized loaders.
- **Complex tensor support** --- interleaved-real complex storage, complex-aware FFT/eig, autograd through complex math.
- **Named tensors** --- `NamedTensor<T>` with `refine_names`, `align_to`, `rename` for advisory dim labels.
- **GPU memory safety** --- pre-OOM hooks, VRAM reservation, budget enforcement, pressure watchdog, and emergency checkpointing. Never lose a training run to a Steam game again.
- **ONNX export** --- trace a model and emit a standard `.onnx` file loadable by onnxruntime, TensorRT, CoreML. Hand-written protobuf encoder, no external dependency.
- **Operation fusion** --- chain elementwise ops into a single kernel with PTX codegen. 2-5x GPU speedup for fused chains.
- **SafeTensors + PyTorch .pt import** --- load HuggingFace models directly; pure-Rust pickle parser (28 opcodes) for PyTorch checkpoints.
- **8 vision model architectures** --- ResNet, VGG, ViT, EfficientNet, ConvNeXt, Swin Transformer, U-Net, YOLO.
- **INT8/INT4 quantization** --- per-tensor and per-channel post-training quantization with quantized matmul.
- **Distributed training** --- DDP with gradient synchronization over a TCP backend, GPU-aware collectives, `DistributedSampler` for multi-rank training.
- **Training loop** --- `Learner` abstraction with metrics (loss, accuracy, top-k), callbacks (early stopping, progress logging), training history, and model checkpointing (save/load).
- **`#[derive(Module)]` proc macro** --- annotate fields with `#[param]`, `#[submodule]`, `#[skip]` and the Module trait is implemented for you.
- **Einops** --- `rearrange("b c h w -> b (c h w)")`, `repeat`, `reduce` with readable string patterns.
- **LoRA** --- parameter-efficient fine-tuning via `LoRALinear` with trainable low-rank A/B matrices and `merge()` for zero-overhead inference.
- **Tensor operations** --- `sum_dim`/`mean_dim` with axis and keepdim, `permute`/`view`/`contiguous`/`chunk`/`split`, `zeros_like`/`ones_like`/`rand_like`/`randn_like`/`full_like` creation ops.
- **Grad context managers** --- `enable_grad()`, `set_grad_enabled()` for fine-grained autograd control.
- **Parallel DataLoader** --- rayon-based `num_workers`, custom `collate_fn`, `DistributedSampler`, seedable transform RNG via `manual_seed`.
- **Fixed-point derivatives** --- implicit differentiation for equilibrium models, Neural CAs, and DEQ networks.
- **K-FAC natural gradient** --- Kronecker-factored Fisher approximation for second-order optimization.
- **Prelude** --- `use ferrotorch::prelude::*` for convenient one-line imports.
- **Zero-panic guarantee** --- every public function returns `Result<T, FerrotorchError>`.

## Quick Start

```rust
use ferrotorch::prelude::*;

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
use ferrotorch::prelude::*;

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

ferrotorch is a workspace of 22 crates. Use the umbrella crate for convenience, or depend on individual crates for minimal compile times.

| Crate | Description |
|---|---|
| **ferrotorch** | Top-level re-export crate (`cargo add ferrotorch`) |
| **ferrotorch-core** | Tensor, autograd engine, 80+ differentiable ops, complex / sparse / named tensors, FFT, signal, masked, quantization |
| **ferrotorch-nn** | Module trait, 30+ layers, lazy modules, losses, activations, `#[derive(Module)]` |
| **ferrotorch-nn-derive** | Proc macro for `#[derive(Module)]` |
| **ferrotorch-optim** | 19 optimizers (foreach mode for SGD/AdamW), 12+ LR schedulers, gradient clipping, GradScaler |
| **ferrotorch-data** | Dataset, parallel DataLoader, samplers, transforms, collate_fn, NumPy/Arrow interop |
| **ferrotorch-train** | Learner, metrics, callbacks, training history, checkpointing |
| **ferrotorch-vision** | 8 model architectures, MNIST/CIFAR/ImageFolder datasets, image I/O |
| **ferrotorch-jit** | Tracing, IR graph, optimization passes, codegen backends |
| **ferrotorch-jit-script** | Script-frontend (Rust-source-to-IR) compiler for ahead-of-time graph lowering |
| **ferrotorch-serialize** | SafeTensors, PyTorch .pt import, ONNX export, checkpoints |
| **ferrotorch-gpu** | NVIDIA CUDA backend, cuBLAS, cuSOLVER, cuFFT, memory guard, pre-OOM hooks |
| **ferrotorch-cubecl** | Portable GPU via CubeCL (NVIDIA, AMD, Intel, Apple) |
| **ferrotorch-mps** | Apple Silicon Metal Performance Shaders backend (M-series GPUs) |
| **ferrotorch-xpu** | Intel Arc / Data Center GPU Max backend via CubeCL wgpu |
| **ferrotorch-distributed** | DDP, allreduce, broadcast, TCP / Gloo backends |
| **ferrotorch-distributions** | 25+ probability distributions (Normal, MultivariateNormal, Bernoulli, Categorical, Beta, Gamma, Kumaraswamy, Weibull, Concrete relaxations, Independent, MixtureSameFamily, …), KL registry, bijective transforms |
| **ferrotorch-hub** | Pretrained model registry, download, and caching |
| **ferrotorch-profiler** | Operation profiling and Chrome trace export |
| **ferrotorch-tokenize** | HuggingFace `tokenizers` wrapper (BPE, WordPiece, Unigram) |
| **ferrotorch-llama** | Llama 3 / Meta LLaMA model composition, GPU bf16 inference, GPTQ/AWQ/HQQ quant loaders |
| **ferrotorch-ml** | Sklearn-compatible adapter, ferrolearn bridge, classic-ML datasets and metrics |

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

### Apple Silicon native (ferrotorch-mps)

`ferrotorch-mps` targets M-series GPUs through the Metal Performance Shaders framework directly, for users on macOS who want a native Metal path independent of WGPU.

### Intel Arc / DC GPU Max (ferrotorch-xpu)

`ferrotorch-xpu` provides Intel-targeted kernels via CubeCL's wgpu backend, with a dedicated `Device::Xpu(_)` device variant.

```rust
use ferrotorch_cubecl::CubeRuntime;

if let Some(rt) = CubeRuntime::auto() {
    println!("Using device: {:?}", rt.device());
}
```

### Unified Device Model

Unlike PyTorch's history of separate CPU/CUDA tensor types, ferrotorch uses
a single `Tensor<T>` that is device-aware internally. Operations auto-dispatch:

```rust
let mut model = Linear::new(784, 10, true)?;
model.to_device(Device::Cuda(0))?;    // Move weights to GPU
let x = rand::<f32>(&[32, 784])?.cuda()?;
let y = model.forward(&x)?;            // Auto-dispatches to GPU
y.backward()?;                          // Autograd on GPU
```

## GPU Memory Safety

Unlike PyTorch, ferrotorch provides **proactive** GPU memory management. No more lost training runs.

```rust
use ferrotorch_gpu::*;

let device = Arc::new(GpuDevice::new(0)?);

// Reserve 22GB upfront — other apps can't steal it
let guard = MemoryGuardBuilder::new(device.clone())
    .budget_bytes(22 * 1024 * 1024 * 1024)
    .reserve_bytes(22 * 1024 * 1024 * 1024)
    .oom_policy(OomPolicy::WaitAndRetry { timeout_secs: 60 })
    .build()?;

// Register a pre-OOM hook: "halve the batch before crashing"
guard.register_hook(MemoryHook {
    name: "halve_batch".into(),
    estimated_free_bytes: 2 * 1024 * 1024 * 1024,
    execution_overhead_bytes: 50 * 1024 * 1024, // metadata setup cost
    priority: 0,
    callback: Box::new(|| { /* split batch, free old tensors */ 2_000_000_000 }),
});

// Emergency checkpoint on unrecoverable OOM
guard.on_oom(|| save_checkpoint(&model, "emergency.ckpt").unwrap());

// Background watchdog pauses training when VRAM gets tight
let watchdog = Arc::new(MemoryWatchdog::new(device, 512 * 1024 * 1024, Duration::from_secs(1)));
watchdog.clone().start();

// In training loop
for batch in loader.iter(epoch) {
    watchdog.wait_if_paused();  // blocks until VRAM pressure lifts
    let buf = guard.safe_alloc_with_hooks::<f32>(batch_size)?;  // hooks fire before OOM
}
```

| Layer | What it prevents |
|---|---|
| **MemoryReservation** | Other processes stealing VRAM mid-training |
| **Budget enforcement** | Allocations beyond your declared limit |
| **Pre-OOM hooks** | Batch splitting, cache clearing — called *before* failure |
| **OomPolicy** | Retry, wait, checkpoint-and-fail, or crash (your choice) |
| **MemoryWatchdog** | Pauses training when free VRAM drops below threshold |
| **Emergency checkpoint** | Saves model state before crash so you don't lose progress |

## ONNX Export

Export models to the ONNX standard format for deployment on any inference runtime:

```rust
use ferrotorch_serialize::export_onnx;

export_onnx(
    |inputs| model.forward(&inputs[0]),
    &[example_input],
    "model.onnx",
    OnnxExportConfig { opset_version: 17, model_name: "my_model".into() },
)?;
```

The exported `.onnx` file works with onnxruntime (C++/Python), NVIDIA TensorRT, Apple CoreML, and ONNX.js (browser). No external protobuf dependency — hand-written encoder in pure Rust.

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
| **GPU** | Unified tensors, CUDA + CubeCL | CUDA + ROCm | CubeCL | Via libtorch | CUDA |
| **Eager mode** | Yes | Yes | Yes | Yes | Yes |
| **JIT / compile** | Tracing + fusion + codegen | TorchScript / torch.compile | No | Via libtorch | No |
| **Distributed** | DDP | DDP / FSDP / Pipeline | No | Via libtorch | Partial |
| **Quantization** | INT8 / INT4 | INT8 / INT4 / FP8 | INT8 | Via libtorch | GGUF |
| **ONNX export** | Yes (pure Rust) | Yes | No | Yes | No |
| **GPU memory safety** | Pre-OOM hooks, budget, watchdog | Basic caching | No | Via libtorch | No |
| **Model zoo** | 8 architectures | Thousands | Limited | Via libtorch | LLM-focused |
| **Training loop** | Learner + callbacks | Manual / Lightning | Learner | Manual | Manual |
| **Proc macro** | `#[derive(Module)]` | No (dynamic) | `#[derive(Module)]` | No | No |
| **LoRA** | Yes (`LoRALinear` + merge) | Via libraries | No | No | Yes |
| **Einops** | Yes (rearrange/repeat/reduce) | Via library | No | No | No |
| **Tests** | 4,800+ | Extensive | Growing | Via libtorch | Growing |

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
