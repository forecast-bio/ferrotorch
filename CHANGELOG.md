# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.3] - 2026-03-17

### Fixed
- Fix bmm_differentiable GPU crash from .data() on GPU tensors (#212)
- Fix view/reshape on GPU tensors dropping requires_grad and breaking autograd graph (#211)
- Fix index_select and masked_fill to use proper GPU kernels instead of CPU fallback (#210)
- Update rustls-webpki to 0.103.10 (#204)
- Update rustls-webpki to 0.103.10 (#204)
- Fix PTX register name collision (`%tid` → `%r_tid`) — all elementwise kernels were silently falling back to CPU due to `CUDA_ERROR_INVALID_PTX`
- Fix softmax PTX: wrong hex prefix for float literals (`0xff` → `0f`), undeclared shared memory registers (`%saddr`, `%sbase`)
- Fix CUDA graph capture on legacy default stream — fork non-blocking stream via `GpuDevice::fork_for_capture()`

### Added
- Add GELU approximation modes matching PyTorch (none, tanh) plus existing sigmoid (#205)
- Add GELU approximation modes matching PyTorch (none, tanh) plus existing sigmoid (#205)
- Add GELU approximation modes matching PyTorch (none, tanh) plus existing sigmoid (#205)
- Update differentiable matmul wrappers and backward passes for broadcast (#203)
- Update nn::Linear to accept arbitrary-rank inputs (#201)
- Add batched broadcast matmul for arbitrary-rank tensors (#200)
- **GPU buffer pool** (`pool.rs`): caching allocator that reuses freed `CudaSlice`s by element count, eliminating `cuMemAllocAsync`/`cuMemFreeAsync` per op
- **CUDA graph capture** (`graph.rs`): `DeviceScalar<T>`, `CapturedGraph`, `begin_capture`/`end_capture` API for replaying entire decode passes as a single driver call
- **`_into` kernel variants**: non-allocating versions of all decode-path kernels (add, mul, scale, gelu, layernorm, softmax, permute, embed_lookup, matmul, bmm, slice_read) for pre-allocated output buffers
- **Indirect-parameter PTX kernels**: `slice_write_indirect` and `causal_mask_indirect` read variable parameters (pos, total_len) from device memory for CUDA graph compatibility
- **`scale_f32` PTX kernel**: scalar multiply (`out[i] = a[i] * scalar`) exposed via `GpuBackend::scale_f32()`
- **`GpuBackend::as_any()`**: downcast trait method for backend-specific access
- **`GpuBufferHandle::into_inner()`**: consume handle and extract concrete type
- **`GpuDevice::fork_for_capture()`**: create non-blocking stream for CUDA graph capture
- **`get_cuda_device()`**: retrieve the shared `GpuDevice` from the registered backend
- **`precompile_decode_kernels()`**: pre-compile all decode-path PTX modules before graph capture
- **`CudaBuffer` pool-aware Drop**: returns `CudaSlice` to pool via function pointer dispatch (f32/f64)
- **`GpuError::PtxCompileFailed`** variant for explicit PTX compilation failure reporting
- M≤4 cuBLAS bypass: route vector-matrix multiplies through PTX `small_matmul` kernel instead of cuBLAS SGEMM

### Changed
- `CudaBuffer<T>.data` is now `Option<CudaSlice<T>>` with custom Drop for pool integration
- `alloc_zeros_f32` / `alloc_zeros_f64` check pool before allocating from CUDA driver
- All kernel output allocations in `kernels.rs` and `blas.rs` use pool-aware `alloc_zeros_f32`/`alloc_zeros_f64`

### Performance
- **GPT-2 124M decode: 3.5 tok/s → 100 tok/s (29x) on WSL2/RTX 3090**
  - PTX bug fixes alone: 3.5 → 90 tok/s (elementwise ops moved from CPU fallback to GPU)
  - CUDA graph capture: 90 → 100 tok/s (300 kernel launches collapsed to 1 graph replay)

## [0.1.0] - 2026-03-15

### Fixed
- Fix PTX kernel recompilation on every call — add module cache (#178)
- Fix flaky backend_impl OnceLock test ordering (#173)
- Fix FlashAttention GPU PTX register name collision (#168)
- Rewrite GPU conv2d as pure GPU — im2col PTX kernel, no CPU roundtrip (#163)
- Fix flaky watchdog timing test (#144)
- Wire up ferray-ufunc SIMD kernels for elementwise ops (#141)
- Wire up ferray-linalg for CPU matmul and fix crates.io dependency versions (#140)

### Added
- Update README and prep all crates for crates.io with per-crate READMEs (#176)
- Add einops, LoRA, fixed-point derivatives, and natural gradient (#174)
- Commit unified device-aware Tensor Steps 2-4 (#172)
- Implement unified device-aware Tensor — Step 1: core infrastructure (#170)
- Design unified device-aware Tensor architecture (#169)
- Phase 10 Wave 4: FlashAttention GPU, gradient penalty, PagedAttention, GGUF (#165)
- Phase 10 Wave 3: higher-order grads, FlashAttention, autocast wiring, model hub (#160)
- Phase 10 Wave 2: linalg, vmap, pack_padded_seq, special functions, TensorBoard (#154)
- Add per-crate READMEs and prep new crates for publishing (#153)
- Phase 10 Wave 1: einsum, hooks, distributions, profiler, sparse, FFT (#149)
- Design document for remaining PyTorch feature parity (#148)
- Update README with pre-OOM hooks, ONNX export, and latest features (#147)
- Add pre-OOM hooks system and ONNX export (#145)
- Add GPU memory reservation, OOM recovery, and graceful pause-on-pressure (#142)
- Add performance benchmarks vs PyTorch (#139)
- Add comprehensive README and prepare crates for crates.io publishing (#138)
- Phase 9: CubeCL + AMD GPU + Training + LLM + Quantization (#136)
- Update CHANGELOG, add licenses, analyze Burn, plan AMD GPU support (#135)

Initial release of ferrotorch — a complete deep learning framework in pure Rust, built on ferray.

### Core Engine (ferrotorch-core)

- **Tensor type** with dynamic shapes, Arc-based identity sharing, and Mutex-guarded gradients (Send + Sync)
- **Reverse-mode autograd** with Kahn's topological sort, gradient accumulation, and computation graph
- **30+ differentiable operations**: arithmetic (add, sub, mul, div, neg, pow, sqrt, abs), reductions (sum, mean, prod), linalg (matmul, mm, mv, dot, bmm, transpose), activations (relu, sigmoid, tanh, gelu, silu, softmax, log_softmax), shape (reshape, flatten, squeeze, unsqueeze, cat, expand), indexing (index_select, masked_fill), comparison (where_)
- **Operator overloading**: `&a + &b`, `a * b`, `-a` with all ownership combinations
- **Method-style API**: `tensor.matmul(&other)`, `tensor.relu()`, `tensor.sum_all()`
- **In-place operations**: `zero_()`, `fill_()`, `add_scalar_()`, `mul_scalar_()`, `clamp_()` with autograd safety guards
- **Display formatting** matching PyTorch style with grad_fn names
- **Gradient checkpointing** for memory-efficient deep networks
- **no_grad() and autocast()** context managers
- **Tensor creation**: zeros, ones, full, rand, randn, eye, arange, linspace, from_slice, scalar

### Neural Network Modules (ferrotorch-nn)

- **Module trait** with forward, parameters, named_parameters, train/eval, state_dict, load_state_dict (strict mode)
- **`#[derive(Module)]` proc macro** with `#[param]`, `#[submodule]`, `#[skip]` attributes
- **Layers**: Linear, Conv1d, Conv2d, ConvTranspose2d, BatchNorm2d, LayerNorm, GroupNorm, RMSNorm, Dropout, Dropout2d, Embedding, MultiheadAttention, LSTM, MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
- **Activations**: ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LogSoftmax, LeakyReLU, ELU, Mish
- **Containers**: Sequential, ModuleList, ModuleDict (insertion-order preserving)
- **Loss functions**: CrossEntropyLoss (label smoothing), MSELoss, BCEWithLogitsLoss, HuberLoss
- **Weight initialization**: xavier_uniform/normal, kaiming_uniform/normal, uniform, normal, zeros, ones
- **Functional API**: `functional::linear`, `functional::relu`, `functional::dropout`, `functional::cross_entropy`
- **Gradient clipping**: `clip_grad_norm_()`, `clip_grad_value_()`

### Optimizers (ferrotorch-optim)

- **6 optimizers**: SGD (momentum, Nesterov, weight decay), Adam (AMSGrad), AdamW (decoupled weight decay), RMSprop (centered mode), Adagrad (LR decay), L-BFGS (two-loop recursion)
- **Parameter groups** with per-group hyperparameters
- **5 LR schedulers**: StepLR, CosineAnnealingLR, LinearWarmup, ReduceLROnPlateau, SequentialLr
- **GradScaler** for mixed-precision training with dynamic loss scaling

### Data Loading (ferrotorch-data)

- **Dataset and IterableDataset traits** with VecDataset and MappedDataset
- **DataLoader** with batching, shuffling, drop_last, seeded reproducibility
- **Samplers**: SequentialSampler, RandomSampler (deterministic Fisher-Yates)
- **Transforms**: Compose, Normalize, ToTensor, RandomCrop, RandomHorizontalFlip

### Vision (ferrotorch-vision)

- **8 model architectures**: ResNet-18/34/50, VGG-11/16, ViT-B/16, EfficientNet-B0, ConvNeXt-Tiny, Swin Transformer Tiny, U-Net, YOLO
- **Model registry**: `list_models()`, `get_model()`, `register_model()`
- **Datasets**: MNIST (real IDX parsing + synthetic), CIFAR-10/100 (synthetic)
- **Image transforms**: Resize, CenterCrop, VisionToTensor, VisionNormalize
- **Image I/O**: read_image, write_image, read_image_as_tensor (PNG/JPEG)

### JIT Compiler (ferrotorch-jit)

- **Tracing**: captures autograd graphs into static IR
- **IR graph** with 27+ operation kinds and binary serialization
- **4 optimization passes**: constant folding, dead code elimination, operator fusion, memory planning
- **compile() API** (torch.compile equivalent)
- **Codegen backends**: InterpreterBackend, NativeBackend
- **Graph break handling**: SegmentedModule with compiled + eager segments

### GPU Backend (ferrotorch-gpu)

- **CUDA via cudarc 0.19** (pure Rust, no C FFI)
- **cuBLAS matmul**: 81.8x speedup on RTX 3090
- **GPU Conv2d**: im2col + cuBLAS GEMM
- **PTX kernels**: add, sub, mul, neg, relu
- **Caching allocator** with memory tracking

### Serialization (ferrotorch-serialize)

- **SafeTensors** via the real `safetensors` crate (HuggingFace compatible)
- **PyTorch .pt import** with custom pickle parser (pure Rust)
- **Training checkpoints** (model + optimizer + epoch)

### Distributed (ferrotorch-distributed)

- **TCP backend** with allreduce, broadcast, barrier
- **DDP wrapper** with gradient synchronization
- **GPU-aware collectives** via transfer transport
