# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.3] - 2026-03-17

### Fixed
- Fix incomplete cache miss tracking in cpu_pool thread pool (#471)
- Fix unsigned hub commits blocking crosslink sync (#413)
- T1.1: GPU gradient accumulation — use backend.add_f32 instead of CPU roundtrip (#259)
- Add GPU dispatch for div, exp, log, sqrt, pow, abs elementwise ops (#218)
- Fix into_storage_and_shape panic on shared GPU tensors (#216)
- QA review: audit all Tier 3 changes for bugs, lazy shortcuts, and correctness issues (#286)
- QA review: audit all Tier 2 changes for bugs, lazy shortcuts, and correctness issues (#274)
- QA review: audit all Tier 1 changes for lazy shortcuts, bugs, and correctness issues (#266)
- Fix shape ops breaking GPU autograd — upload to device before from_operation, not after (#247)
- Fix Conv2d, Conv1d, and all pooling layers to work on GPU (#221)
- Fix GPU backend f64 path — hardcoded f32 in cpu_to_gpu, gpu_to_cpu, clone_buffer (#238)
- Fix batched matmul and broadcast matmul backward crash on GPU (#228)
- Fix einsum and einops crash on GPU tensors (#226)
- Fix checkpoint, higher_order grad, grad_penalty, fixed_point crash on GPU (#227)
- Fix all probability distributions crash on GPU tensors (#234)
- Fix vision models GPU crashes — ViT, Swin, ConvNeXt, UNet (#235)
- Fix Dropout2d and functional dropout crash on GPU (#240)
- Fix LSTM autograd graph severed by from_storage leaf creation (#222)
- Fix FlashAttention, RoPE, KVCache, SwiGLU crash on GPU (#229)
- Fix FFT operations crash on GPU tensors (#230)
- Fix JIT fusion engine crash on GPU tensors (#237)
- Fix permute, split, chunk, where crash on GPU tensors (#225)
- Fix distributed collective ops and DDP crash on GPU tensors (#236)
- Fix serialization save crash on GPU tensors (#233)
- Fix KFAC optimizer and GradScaler GPU crashes (#232)
- Fix all loss functions to work on GPU tensors (#220)
- Fix gradient clipping utilities crash on GPU tensors (#224)
- Fix in-place ops crash on GPU — add_scalar_, mul_scalar_, fill_, clamp_ (#223)
- Fix backward device restore in GroupNorm, RMSNorm, Softplus, reduce_grad (#231)
- Fix tanh, silu, softplus returning CPU tensors when GPU input has grad (#219)
- Fix into_storage_and_shape panic on shared GPU tensors (#217)
- Fix optimizer step on GPU parameters — add in-place GPU write path (#215)
- Fix Tensor::to() to preserve autograd graph across device transfers (#214)
- Fix Embedding backward to produce weight gradients on GPU (#213)
- Fix bmm_differentiable GPU crash from .data() on GPU tensors (#212)
- Fix view/reshape on GPU tensors dropping requires_grad and breaking autograd graph (#211)
- Fix index_select and masked_fill to use proper GPU kernels instead of CPU fallback (#210)
- Update rustls-webpki to 0.103.10 (#204)
- Update rustls-webpki to 0.103.10 (#204)
- Fix PTX register name collision (`%tid` → `%r_tid`) — all elementwise kernels were silently falling back to CPU due to `CUDA_ERROR_INVALID_PTX`
- Fix softmax PTX: wrong hex prefix for float literals (`0xff` → `0f`), undeclared shared memory registers (`%saddr`, `%sbase`)
- Fix CUDA graph capture on legacy default stream — fork non-blocking stream via `GpuDevice::fork_for_capture()`

### Added
- WU-20: GPU kernel expansion — GroupNorm, BatchNorm backward, MaxPool2d, AvgPool2d GPU kernels (#358)
- T3.5: DataLoader prefetch pipeline + pin_memory (#279)
- Multi-threaded backward engine with per-device worker threads and priority queue (#354)
- JIT pattern fusion — fuse_attention, fuse_linear, fuse_conv_bn (3-5x transformer speedup) (#366)
- DDP communication/computation overlap — allreduce during backward (#371)
- DDP gradient bucketing — group params into 25MB buckets for async allreduce (#370)
- SavedVariable hooks — pack/unpack for memory offloading in autograd (#355)
- T3.10: MultiheadAttention batched matmul — eliminate serial batch/head loops (#284)
- T3.2: Checkpoint RNG preservation + multi-tensor input (#276)
- T3.1: JIT multi-input fusion + GPU kernel execution (#275)
- T3.3: GradScaler GPU kernel for fused unscale+inf-check (#277)
- WU-04: In-place tensor ops (add_, mul_, sub_, div_, zero_, fill_) with version counter (#357)
- T2.2: fp16/bf16 kernels and cublasGemmEx for Tensor Cores (#268)
- T2.1: CUDA stream pool with thread-local current stream and events (#267)
- T1.3: Zero-copy stride-based views for transpose/permute/slice (#261)
- T1.2: GPU-resident optimizer state — store exp_avg/exp_avg_sq as GPU tensors (#260)
- Perf Phase 5B: Wire backward GPU kernels — eliminate all CPU roundtrips in backward passes (#255)
- Perf Phase 3C: Fused SIMD sigmoid, sin, cos kernels — eliminate intermediate allocations (#254)
- Perf Phase 3B: Wire fast_sigmoid and fast_tanh into activation forward paths (#253)
- Perf Phase 5: Wire GPU kernels into grad_fns to eliminate CPU roundtrips (#252)
- Perf Phase 4B: Add backward and reduction GPU kernels (#251)
- Perf Phase 4: Add missing GPU kernels via CubeCL — div, exp, log, sqrt, sigmoid, tanh, axis reductions (#250)
- Add Pythia-70M architecture integration test for GPU training validation (#249)
- Add GPU integration test suite — 5 end-to-end training experiments (#248)
- Add CPU tensor buffer pool to eliminate allocation overhead in elementwise ops (#246)
- Perf Phase 3: Rewrite elementwise kernels with pulp SIMD + rayon parallelism (#245)
- Perf Phase 2: Migrate CPU matmul from matrixmultiply to faer GEMM (#242)
- Perf Phase 1: Add data_ref() zero-copy CPU path to eliminate data_vec() copies (#241)
- Perf Phase 8: Add .cargo/config.toml with target-cpu=native (#244)
- Perf Phase 7: Switch global allocator to mimalloc (#243)
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
- Add GPU kernels for cumulative ops (cumsum, cumprod, cummax, cummin, logcumsumexp) (#478)
- Eliminate CPU roundtrip in unary_map for GPU tensors (foundation function for many ops) (#480)
- Add GPU backward kernels for SiLU, ELU, Mish, LogSoftmax (no GPU path, force CPU roundtrip) (#477)
- Add GPU forward kernel for log_softmax (currently explicit .cpu() download) (#476)
- Add GPU forward kernels for SiLU, ELU, Mish activations (currently use unary_map CPU roundtrip) (#475)
- Eliminate GPU→CPU roundtrips in norm.rs forward/backward (LayerNorm, GroupNorm, RMSNorm, BatchNorm) (#474)
- Eliminate GPU→CPU roundtrips in all loss function backward passes (13+ losses, 42 .cpu() calls) (#473)
- Add GPU backward kernel for GELU Tanh approximation mode (#465)
- Add GPU forward kernels for GELU Tanh and erf approximation modes (#469)
- GPU Conv2d backward pass — forward-only GPU kernel, backward falls to CPU (#349)
- Fused GRU/LSTM kernels — GRU forward is 10.7x slower than PyTorch (gate-level fusion needed) (#345)
- Optimized Conv2d — im2col is 7.4x slower than PyTorch, needs cache-friendly tiling (#344)
- GPU vectorized loads — ld.global.f32 (32-bit) should be ld.global.v4.f32 (128-bit) (#351)
- Lower CPU parallel threshold from 2M to 32K elements to match PyTorch grain size (#343)
- SLEEF vectorized transcendentals — exp/sin/cos are 22-33x slower than PyTorch (scalar libm) (#341)
- TensorIterator abstraction for broadcasting — broadcast ops are 166-185x slower than PyTorch (#339)
- Vectorized reduction kernels — sum/mean/max/min along axis are 500-670x slower than PyTorch (#338)
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
