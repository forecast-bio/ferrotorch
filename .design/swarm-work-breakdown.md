# Swarmable Work Breakdown — All PyTorch Gaps

*30 independent work units. Each touches exactly one crate. All can run in parallel.*

---

## How to use this

Each unit is a self-contained agent task. Launch them all at once:
```
/kickoff "WU-XX: <title>"
```

Every unit specifies:
- **Crate**: Which directory to work in
- **Files**: Exact files to create or edit
- **Deliverables**: What to implement
- **Tests**: What to verify
- **PyTorch reference**: Where to look for implementation guidance

Units are numbered WU-01 through WU-30. No unit depends on another.

---

## WU-01: Missing Tensor Indexing Ops
**Crate**: `ferrotorch-core`
**Files**: `src/ops/indexing.rs` (edit), `src/grad_fns/indexing.rs` (edit)
**Deliverables**:
- `gather(input, dim, index)` — Gather along axis by index tensor
- `scatter(input, dim, index, src)` — Scatter src into input at index positions
- `scatter_add(input, dim, index, src)` — Scatter with addition
- `where_cond(condition, x, y)` — Ternary selection
- All with backward implementations (GradFn)
**Tests**: Shape correctness, gradient flow, edge cases (empty, single-element)
**PyTorch ref**: `aten/src/ATen/native/ScatterGather.cpp`, `aten/src/ATen/native/TensorAdvancedIndexing.cpp`

---

## WU-02: Cumulative & Scan Ops
**Crate**: `ferrotorch-core`
**Files**: `src/ops/cumulative.rs` (new), `src/grad_fns/cumulative.rs` (new), `src/lib.rs` (add mod)
**Deliverables**:
- `cumsum(input, dim)` with backward
- `cumprod(input, dim)` with backward
- `cummax(input, dim)` returning (values, indices)
- `cummin(input, dim)` returning (values, indices)
- `logcumsumexp(input, dim)` with backward
**Tests**: Compare against manual loop, gradient correctness, multi-dim
**PyTorch ref**: `aten/src/ATen/native/ReduceOps.cpp`

---

## WU-03: Statistical & Special Functions
**Crate**: `ferrotorch-core`
**Files**: `src/special.rs` (edit), `src/ops/reduction.rs` (edit)
**Deliverables**:
- `nanmean(input, dim)`, `nansum(input, dim)` — NaN-aware reductions
- `logsumexp(input, dim)` with backward — numerically stable
- Extend `special.rs`: `erfinv`, `polygamma(n, input)`, `lgamma`, `digamma`
**Tests**: NaN handling, numerical stability near overflow, gradient correctness
**PyTorch ref**: `aten/src/ATen/native/ReduceOps.cpp`, `aten/src/ATen/native/Math.h`

---

## WU-04: In-Place Ops & Version Counter
**Crate**: `ferrotorch-core`
**Files**: `src/tensor.rs` (edit), `src/ops/elementwise.rs` (edit)
**Deliverables**:
- Add `version: Arc<AtomicU32>` to `TensorInner`
- `bump_version()` on in-place mutation
- `add_(other)`, `mul_(scalar)`, `sub_(other)`, `div_(scalar)` — in-place variants
- `zero_()`, `fill_(value)` — in-place initialization
- Safety: check version hasn't changed since saved for backward
**Tests**: Version increment, error on stale saved tensor, in-place + backward correctness
**PyTorch ref**: `c10/core/TensorImpl.h` (version_counter), `aten/src/ATen/native/BinaryOps.cpp`

---

## WU-05: Channels-Last Memory Format
**Crate**: `ferrotorch-core`
**Files**: `src/tensor.rs` (edit), `src/storage.rs` (edit)
**Deliverables**:
- `MemoryFormat` enum: `Contiguous`, `ChannelsLast`, `ChannelsLast3d`
- `to_memory_format(format)` — rearrange strides without copying data when possible
- `is_contiguous(memory_format)` — check specific format
- `contiguous(memory_format)` — materialize to target format
**Tests**: NCHW→NHWC stride calculation, roundtrip, contiguity checks
**PyTorch ref**: `c10/core/MemoryFormat.h`, `aten/src/ATen/native/TensorShape.cpp`

---

## WU-06: Autograd Engine — Forward-Mode AD
**Crate**: `ferrotorch-core`
**Files**: `src/autograd/forward_ad.rs` (new), `src/autograd/mod.rs` (edit), `src/lib.rs` (add re-export)
**Deliverables**:
- Dual number `DualTensor<T>` with primal + tangent
- `jvp(f, primals, tangents)` — exact Jacobian-vector product (replace finite-diff)
- Forward-mode rules for: add, sub, mul, div, matmul, relu, sigmoid, tanh, exp, log, sin, cos
- `jacfwd(f, input)` using vmap(jvp) pattern
**Tests**: Compare against finite-diff (should be exact), higher-order composition
**PyTorch ref**: `torch/_functorch/eager_transforms.py` (jvp), `torch/csrc/autograd/forward_grad.h`

---

## WU-07: Autograd Hooks & Anomaly Detection
**Crate**: `ferrotorch-core`
**Files**: `src/autograd/hooks.rs` (new), `src/autograd/anomaly.rs` (new), `src/tensor.rs` (edit)
**Deliverables**:
- `tensor.register_hook(fn)` — called when gradient is computed for this tensor
- `tensor.register_post_accumulate_grad_hook(fn)` — called after grad accumulation
- `AnomalyMode::enable()` / `disable()` — global toggle
- When enabled: store backtrace at forward, print at backward error
- `remove_hook(handle)` for cleanup
**Tests**: Hook firing order, anomaly trace on NaN gradient, hook removal
**PyTorch ref**: `torch/csrc/autograd/function_hook.h`, `torch/autograd/anomaly_mode.py`

---

## WU-08: vmap Composability & Batching Rules
**Crate**: `ferrotorch-core`
**Files**: `src/vmap.rs` (edit)
**Deliverables**:
- Fix vmap signature to support multi-arg, multi-output patterns
- `vmap_n(f, in_dims, out_dims)` — N-ary input/output
- Dedicated batching rules for: matmul (→bmm), add/sub/mul/div (broadcast), transpose, reshape
- `per_sample_grad(loss_fn, params, batch)` convenience wrapper
- `make_functional(module)` — extract params, return (fn, params) tuple
**Tests**: vmap(grad) composition, matmul batching vs loop (correctness + speed), per_sample_grad on MLP
**PyTorch ref**: `torch/_functorch/vmap.py`, `torch/_functorch/eager_transforms.py`

---

## WU-09: Missing Activations
**Crate**: `ferrotorch-nn`
**Files**: `src/activation.rs` (edit)
**Deliverables**:
- `ReLU6` — clamp(x, 0, 6)
- `Hardtanh` — clamp(x, min_val, max_val)
- `LogSigmoid` — log(sigmoid(x)) numerically stable
- `Softmin` — softmax(-x)
- `Threshold` — threshold function
- `Softshrink`, `Hardshrink`, `Tanhshrink`, `Softsign`
- `RReLU` — random leaky ReLU (stochastic in training)
- All with forward + backward
**Tests**: Forward values, gradient correctness, training/eval mode for RReLU
**PyTorch ref**: `torch/nn/modules/activation.py`

---

## WU-10: Missing Conv & Padding Layers
**Crate**: `ferrotorch-nn`
**Files**: `src/conv.rs` (edit), `src/padding.rs` (new), `src/lib.rs` (add mod)
**Deliverables**:
- `Conv3d` — 3D convolution via vol2col + matmul
- `ConvTranspose1d` — 1D transposed convolution
- `ConvTranspose3d` — 3D transposed convolution
- Padding modules: `ConstantPad1d/2d/3d`, `ReflectionPad1d/2d/3d`, `ReplicationPad1d/2d/3d`, `ZeroPad1d/2d/3d`
- Add padding_mode parameter to existing Conv1d/2d: 'reflect', 'replicate', 'circular'
**Tests**: Shape correctness, gradient flow, padding mode equivalence
**PyTorch ref**: `torch/nn/modules/conv.py`, `torch/nn/modules/padding.py`

---

## WU-11: Missing Norm & Pooling Layers
**Crate**: `ferrotorch-nn`
**Files**: `src/norm.rs` (edit), `src/pooling.rs` (edit)
**Deliverables**:
- `InstanceNorm1d/2d/3d` — per-sample normalization
- `SyncBatchNorm` — stub with note about distributed requirement
- `MaxPool1d`, `MaxPool3d`, `AvgPool1d`, `AvgPool3d`
- `AdaptiveMaxPool2d`, `AdaptiveAvgPool1d/3d`
- `MaxUnpool2d` — inverse of MaxPool2d using saved indices
**Tests**: Shape correctness, gradient flow, adaptive output size calculation
**PyTorch ref**: `torch/nn/modules/instancenorm.py`, `torch/nn/modules/pooling.py`

---

## WU-12: Missing Loss Functions
**Crate**: `ferrotorch-nn`
**Files**: `src/loss.rs` (edit)
**Deliverables**:
- `BCELoss` — binary cross-entropy (requires sigmoid input)
- `TripletMarginLoss` — triplet loss for metric learning
- `MarginRankingLoss` — ranking loss with margin
- `CTCLoss` — connectionist temporal classification
- `PoissonNLLLoss` — Poisson regression loss
- `MultiMarginLoss`, `MultiLabelSoftMarginLoss`
- `HingeEmbeddingLoss`
**Tests**: Forward values vs manual computation, gradient correctness, reduction modes
**PyTorch ref**: `torch/nn/modules/loss.py`

---

## WU-13: Upsample, Interpolation & Vision Ops
**Crate**: `ferrotorch-nn`
**Files**: `src/upsample.rs` (new), `src/functional.rs` (edit), `src/lib.rs` (add mod)
**Deliverables**:
- `Upsample` module with modes: nearest, bilinear, bicubic
- `F.interpolate(input, size, scale_factor, mode, align_corners)` — functional API
- `F.grid_sample(input, grid, mode, padding_mode, align_corners)` — spatial transformer
- `F.affine_grid(theta, size, align_corners)` — generate affine grids
- `PixelShuffle`, `PixelUnshuffle` — sub-pixel convolution
- `Unfold`, `Fold` — sliding window patch extraction/reconstruction
**Tests**: Bilinear upsample 2x matches expected, grid_sample identity grid, roundtrip fold/unfold
**PyTorch ref**: `torch/nn/functional.py` (interpolate, grid_sample, affine_grid)

---

## WU-14: Weight Initialization & Module Utils
**Crate**: `ferrotorch-nn`
**Files**: `src/init.rs` (edit), `src/utils.rs` (edit)
**Deliverables**:
- `trunc_normal_(tensor, mean, std, a, b)` — truncated normal init
- `orthogonal_(tensor, gain)` — orthogonal initialization via QR
- `sparse_(tensor, sparsity, std)` — sparse initialization
- `dirac_(tensor, groups)` — Dirac delta initialization for conv
- `eye_(tensor)` — identity matrix initialization
- `weight_norm(module, name, dim)` — weight normalization wrapper
- `spectral_norm(module, name)` — spectral normalization wrapper
**Tests**: Distribution statistics, orthogonality check, spectral norm ≤ 1
**PyTorch ref**: `torch/nn/init.py`, `torch/nn/utils/weight_norm.py`, `torch/nn/utils/spectral_norm.py`

---

## WU-15: Missing Optimizers
**Crate**: `ferrotorch-optim`
**Files**: `src/radam.rs` (new), `src/nadam.rs` (new), `src/adamax.rs` (new), `src/adadelta.rs` (new), `src/rprop.rs` (new), `src/asgd.rs` (new), `src/lib.rs` (add mods)
**Deliverables**:
- `RAdam` — Rectified Adam with variance rectification term
- `NAdam` — Nesterov-accelerated Adam
- `Adamax` — Adam variant using L-infinity norm
- `Adadelta` — adaptive learning rate per-parameter
- `Rprop` — resilient backpropagation
- `ASGD` — averaged stochastic gradient descent
- All implementing `Optimizer<T>` trait with state dict support
**Tests**: Convergence on Rosenbrock, parameter update correctness, state dict save/load
**PyTorch ref**: `torch/optim/radam.py`, `torch/optim/nadam.py`, `torch/optim/adamax.py`, etc.

---

## WU-16: Missing LR Schedulers
**Crate**: `ferrotorch-optim`
**Files**: `src/scheduler/` (add new files), `src/scheduler/mod.rs` (edit)
**Deliverables**:
- `LambdaLR` — user-provided lambda function
- `MultiStepLR` — decay at specific epoch milestones
- `ExponentialLR` — exponential decay per epoch
- `CosineAnnealingWarmRestarts` — cosine with warm restarts (SGDR)
- `CyclicLR` — cyclic learning rate (triangular, triangular2, exp_range)
- `OneCycleLR` — super-convergence one-cycle policy
- `PolynomialLR` — polynomial decay
- `ConstantLR`, `LinearLR` — simple ramps
**Tests**: LR values at specific steps match expected, warmup/decay curves
**PyTorch ref**: `torch/optim/lr_scheduler.py`

---

## WU-17: EMA, SWA & Optimizer Utilities
**Crate**: `ferrotorch-optim`
**Files**: `src/swa.rs` (new), `src/ema.rs` (new), `src/lib.rs` (add mods)
**Deliverables**:
- `ExponentialMovingAverage` — maintains EMA copy of model params, decay parameter
- `AveragedModel` — wraps model, applies EMA/SWA update after each step
- `SWALR` — SWA-specific learning rate schedule
- `update_bn(model, dataloader)` — recalculate batch norm stats after averaging
- `maximize` flag support on all optimizers (negate gradient for maximization)
**Tests**: EMA decay correctness, SWA averaging matches manual, BN update changes running stats
**PyTorch ref**: `torch/optim/swa_utils.py`

---

## WU-18: GPU Caching Allocator
**Crate**: `ferrotorch-gpu`
**Files**: `src/allocator.rs` (rewrite), `src/pool.rs` (edit)
**Deliverables**:
- `Block` struct: size, stream, stream_uses set, device
- `BlockPool` with power-of-two size binning (small <1MB, large ≥1MB)
- Block splitting: oversized blocks split, remainder returned to pool
- Block coalescing: adjacent freed blocks merged
- Stream-aware reuse: check stream_uses before returning block
- `record_stream(buffer, stream)` — track cross-stream usage
- `empty_cache()` — release all cached blocks back to CUDA driver
- Fallback: on allocation failure, try GC then retry
**Tests**: Alloc/free cycles don't leak, stream safety, fragmentation under repeated alloc/free
**PyTorch ref**: `c10/cuda/CUDACachingAllocator.cpp`

---

## WU-19: GPU Stream Priority & Graph Isolation
**Crate**: `ferrotorch-gpu`
**Files**: `src/stream.rs` (edit), `src/graph.rs` (edit)
**Deliverables**:
- Expand stream pool: 32 low-priority + 32 high-priority streams per device
- `StreamPriority` enum: Low, High
- `get_stream(priority)` — allocate from correct pool
- Route matmul/attention to high-priority streams
- Graph memory isolation: allocations during capture go to private pool
- Private pool released only when graph is destroyed
**Tests**: Priority streams created with correct CUDA priority, graph capture doesn't leak
**PyTorch ref**: `c10/cuda/CUDAStream.h`, `torch/cuda/graphs.py`

---

## WU-20: GPU Kernel Expansion
**Crate**: `ferrotorch-gpu`
**Files**: `src/kernels.rs` (edit), `src/conv.rs` (edit)
**Deliverables**:
- Reduction kernels: `gpu_sum(dim)`, `gpu_mean(dim)`, `gpu_max(dim)`, `gpu_min(dim)`
- GroupNorm GPU kernel (forward + backward)
- BatchNorm GPU backward kernel
- MaxPool2d GPU kernel
- AvgPool2d GPU kernel
- Expand Conv2d GPU to include backward pass
**Tests**: Numerical match against CPU implementations, gradient correctness
**PyTorch ref**: `aten/src/ATen/native/cuda/` (Reduce.cu, NormKernels.cu, Pool2d.cu)

---

## WU-21: Data Loading Enhancements
**Crate**: `ferrotorch-data`
**Files**: `src/sampler.rs` (edit), `src/dataset.rs` (edit), `src/collate.rs` (new), `src/lib.rs` (add mod)
**Deliverables**:
- `WeightedRandomSampler` — multinomial sampling by weight vector
- `SubsetRandomSampler` — shuffle within index subset
- `BatchSampler` — wraps any sampler to yield batches
- `ConcatDataset`, `ChainDataset`, `SubsetDataset` — composite datasets
- `default_collate(samples)` — stack tensors, handle nested types
- Worker init function support: `worker_init_fn` parameter on DataLoader
**Tests**: Weighted sampling distribution, concat dataset indexing, collate shape correctness
**PyTorch ref**: `torch/utils/data/sampler.py`, `torch/utils/data/dataset.py`, `torch/utils/data/_utils/collate.py`

---

## WU-22: Distributed DDP Optimization
**Crate**: `ferrotorch-distributed`
**Files**: `src/ddp.rs` (edit), `src/collective.rs` (edit)
**Deliverables**:
- `GradBucket` struct — groups parameters by configurable size (default 25MB)
- Bucket-level async allreduce: trigger allreduce per bucket as gradients arrive
- Gradient compression hooks: `FP16CompressHook`, `PowerSGDHook` (basic)
- `Join` context for handling uneven inputs across ranks
- Communication/computation overlap via background futures
**Tests**: Bucketed allreduce matches flat allreduce numerically, compression roundtrip
**PyTorch ref**: `torch/distributed/algorithms/ddp_comm_hooks/`

---

## WU-23: FSDP Sharding Strategies
**Crate**: `ferrotorch-distributed`
**Files**: `src/fsdp.rs` (edit)
**Deliverables**:
- `ShardingStrategy` enum: `FullShard`, `ShardGradOp`, `NoShard`, `HybridShard`
- `ShardGradOp`: only shard gradients + optimizer states, keep params replicated
- `NoShard`: pure DDP behavior through FSDP interface
- Backward prefetch: all-gather next layer's params during current backward
- CPU offloading: params on CPU, prefetch to GPU on demand
**Tests**: Memory reduction matches expected for each strategy, training loss matches DDP
**PyTorch ref**: `torch/distributed/fsdp/`

---

## WU-24: Serialization Expansion
**Crate**: `ferrotorch-serialize`
**Files**: `src/pytorch_export.rs` (new), `src/onnx_export.rs` (edit), `src/state_dict.rs` (edit), `src/lib.rs` (add mod)
**Deliverables**:
- `save_pytorch(state_dict, path)` — ZIP + pickle protocol 2 writer (mirror of pytorch_import)
- Expand dtype support in state_dict: f16, bf16, i8, i16, i32, i64, u8
- ONNX op registry: systematic mapping of all ferrotorch-nn ops to ONNX operators
- Support opset versions 14, 17, 18
- Dynamic shape annotations in ONNX export
- `validate_checkpoint(path)` — CRC32 integrity check
**Tests**: Round-trip PyTorch import→export, ONNX model loads in onnxruntime, corrupt file detection
**PyTorch ref**: `torch/serialization.py`, `torch/onnx/_internal/exporter/`

---

## WU-25: Distribution Families (Univariate)
**Crate**: `ferrotorch-distributions`
**Files**: `src/beta.rs`, `src/gamma.rs`, `src/exponential.rs`, `src/laplace.rs`, `src/poisson.rs`, `src/log_normal.rs`, `src/student_t.rs`, `src/gumbel.rs`, `src/half_normal.rs` (all new), `src/lib.rs` (add mods)
**Deliverables**:
- 9 distributions: Beta, Gamma, Exponential, Laplace, Poisson, LogNormal, StudentT, Gumbel, HalfNormal
- All implement `Distribution<T>` trait: sample, log_prob, entropy
- rsample (reparameterized) for all continuous distributions
- Correct parameter validation
**Tests**: Sample mean/variance converge to theoretical, log_prob matches manual computation, rsample gradients flow
**PyTorch ref**: `torch/distributions/` (beta.py, gamma.py, etc.)

---

## WU-26: Distribution Infrastructure
**Crate**: `ferrotorch-distributions`
**Files**: `src/transforms.rs` (new), `src/constraints.rs` (new), `src/kl.rs` (new), `src/lib.rs` (add mods)
**Deliverables**:
- `Transform` trait: forward, inverse, log_abs_det_jacobian
- Transforms: ExpTransform, AffineTransform, SigmoidTransform, TanhTransform, SoftplusTransform, ComposeTransform
- `Constraint` trait: check(value) → bool, is_discrete, event_dim
- Constraints: real, positive, unit_interval, simplex, greater_than, interval
- `TransformedDistribution` — apply bijective transforms to base distribution
- `kl_divergence(p, q)` — registry for same-family KL (Normal↔Normal, Bernoulli↔Bernoulli, etc.)
- At least 10 KL pairs for implemented distributions
**Tests**: Transform invertibility, constraint checking, KL divergence vs Monte Carlo estimate
**PyTorch ref**: `torch/distributions/transforms.py`, `torch/distributions/constraints.py`, `torch/distributions/kl.py`

---

## WU-27: Multivariate Distributions
**Crate**: `ferrotorch-distributions`
**Files**: `src/multivariate_normal.rs`, `src/dirichlet.rs`, `src/multinomial.rs`, `src/independent.rs` (all new), `src/lib.rs` (add mods)
**Deliverables**:
- `MultivariateNormal` — full covariance, precision, or scale_tril parameterization
- `Dirichlet` — Bayesian simplex distribution
- `Multinomial` — multi-trial categorical
- `Independent` — reinterpret batch dims as event dims
- `MixtureSameFamily` — mixture of same-family distributions
- rsample for MultivariateNormal and Dirichlet
**Tests**: MVN sample covariance matches input, Dirichlet samples on simplex, mixture log_prob
**PyTorch ref**: `torch/distributions/multivariate_normal.py`, `torch/distributions/dirichlet.py`

---

## WU-28: Vision Transforms & Augmentation
**Crate**: `ferrotorch-vision`
**Files**: `src/transforms/` (add new files), `src/transforms/mod.rs` (edit)
**Deliverables**:
- `Compose` — chain transforms
- `RandomHorizontalFlip(p)`, `RandomVerticalFlip(p)` — stochastic flips
- `RandomCrop(size, padding)` — random spatial crop with padding
- `RandomResizedCrop(size, scale, ratio)` — crop + resize
- `RandomRotation(degrees)` — rotation by random angle
- `ColorJitter(brightness, contrast, saturation, hue)` — photometric augmentation
- `RandomGaussianBlur(kernel_size, sigma)` — Gaussian blur
- `RandomAffine(degrees, translate, scale, shear)` — affine transform
- `RandomApply(transforms, p)`, `RandomChoice(transforms)` — stochastic composition
**Tests**: Output shape preservation, deterministic with seed, augmentation diversity
**PyTorch ref**: `torchvision/transforms/` (though external, patterns are standard)

---

## WU-29: Profiler Auto-Instrumentation & Scheduling
**Crate**: `ferrotorch-profiler`
**Files**: `src/profiler.rs` (edit), `src/event.rs` (edit), `src/report.rs` (edit), `src/schedule.rs` (new)
**Deliverables**:
- `ProfileSchedule` — wait/warmup/active/repeat cycle configuration
- `profiler.step()` — mark iteration boundaries
- `on_trace_ready` callback — called when recording cycle completes
- Memory tracking: `record_memory(name, bytes, category)` with categories (Parameter, Activation, Gradient, Temporary)
- FLOPS estimation for matmul and conv2d ops
- Stack trace capture via `backtrace` crate (complete the `with_stack` flag)
- Enhanced Chrome trace: parent-child hierarchy, self-time computation
- `report.save_chrome_trace_gz(path)` — gzip-compressed export
**Tests**: Schedule cycle correctness, memory category aggregation, FLOPS estimates match expected
**PyTorch ref**: `torch/profiler/profiler.py`, `torch/profiler/_memory_profiler.py`

---

## WU-30: Training Utilities — Checkpointing, AMP & Clipping
**Crate**: `ferrotorch-train`
**Files**: `src/checkpoint.rs` (new), `src/amp.rs` (new), `src/grad_utils.rs` (new), `src/lib.rs` (add mods)
**Deliverables**:
- **Gradient checkpointing**: `checkpoint(function, *args)` — recompute forward in backward to save memory
  - Save RNG state before checkpoint region, restore during recompute
  - Support nested checkpointing
- **Autocast context**: `autocast(device, dtype, |closure|)` — cast ops to lower precision
  - Op-level casting table: which ops run in fp16 vs fp32
  - Integration with GradScaler
- **Gradient clipping**: `clip_grad_norm_(params, max_norm, norm_type)`, `clip_grad_value_(params, clip_value)`
- **EMA integration**: `EmaCallback` for Learner — update EMA params after each step
**Tests**: Checkpoint reduces peak memory, autocast produces fp16 intermediates, clip_grad bounds gradient norm
**PyTorch ref**: `torch/utils/checkpoint.py`, `torch/amp/autocast_mode.py`, `torch/nn/utils/clip_grad.py`

---

## Summary: Agent Launch Order

All 30 units are independent. For maximum parallelism, launch all at once. If you need to prioritize, here's the grouping by impact tier:

### Tier 0 — Performance (launch first)
| Unit | Title | Crate |
|------|-------|-------|
| WU-04 | In-Place Ops & Version Counter | ferrotorch-core |
| WU-05 | Channels-Last Memory Format | ferrotorch-core |
| WU-18 | GPU Caching Allocator | ferrotorch-gpu |
| WU-19 | GPU Stream Priority & Graph Isolation | ferrotorch-gpu |
| WU-20 | GPU Kernel Expansion | ferrotorch-gpu |

### Tier 1 — Feature-Critical (blocks real workloads)
| Unit | Title | Crate |
|------|-------|-------|
| WU-06 | Forward-Mode AD | ferrotorch-core |
| WU-10 | Conv3d & Padding Layers | ferrotorch-nn |
| WU-11 | Norm & Pooling Layers | ferrotorch-nn |
| WU-13 | Upsample, Interpolation & Vision Ops | ferrotorch-nn |
| WU-22 | DDP Optimization | ferrotorch-distributed |
| WU-30 | Training: Checkpointing, AMP & Clipping | ferrotorch-train |

### Tier 2 — Ecosystem Completeness
| Unit | Title | Crate |
|------|-------|-------|
| WU-01 | Tensor Indexing Ops | ferrotorch-core |
| WU-02 | Cumulative & Scan Ops | ferrotorch-core |
| WU-03 | Statistical & Special Functions | ferrotorch-core |
| WU-08 | vmap Composability & Batching Rules | ferrotorch-core |
| WU-09 | Missing Activations | ferrotorch-nn |
| WU-12 | Missing Loss Functions | ferrotorch-nn |
| WU-14 | Weight Init & Module Utils | ferrotorch-nn |
| WU-15 | Missing Optimizers | ferrotorch-optim |
| WU-16 | Missing LR Schedulers | ferrotorch-optim |
| WU-17 | EMA, SWA & Optimizer Utils | ferrotorch-optim |
| WU-21 | Data Loading Enhancements | ferrotorch-data |
| WU-25 | Distribution Families (Univariate) | ferrotorch-distributions |
| WU-26 | Distribution Infrastructure | ferrotorch-distributions |
| WU-27 | Multivariate Distributions | ferrotorch-distributions |
| WU-28 | Vision Transforms & Augmentation | ferrotorch-vision |

### Tier 3 — Production Hardening
| Unit | Title | Crate |
|------|-------|-------|
| WU-07 | Autograd Hooks & Anomaly Detection | ferrotorch-core |
| WU-23 | FSDP Sharding Strategies | ferrotorch-distributed |
| WU-24 | Serialization Expansion | ferrotorch-serialize |
| WU-29 | Profiler Auto-Instrumentation | ferrotorch-profiler |

---

## Crate Conflict Matrix

Multiple units touch the same crate. When running in parallel, use **git worktrees** to isolate:

| Crate | Units | Max Parallel |
|-------|-------|-------------|
| ferrotorch-core | WU-01, 02, 03, 04, 05, 06, 07, 08 | 8 (all different files) |
| ferrotorch-nn | WU-09, 10, 11, 12, 13, 14 | 6 (all different files) |
| ferrotorch-optim | WU-15, 16, 17 | 3 (all different files) |
| ferrotorch-gpu | WU-18, 19, 20 | 3 (all different files) |
| ferrotorch-distributed | WU-22, 23 | 2 (different files) |
| ferrotorch-distributions | WU-25, 26, 27 | 3 (all different files) |
| ferrotorch-data | WU-21 | 1 |
| ferrotorch-serialize | WU-24 | 1 |
| ferrotorch-vision | WU-28 | 1 |
| ferrotorch-profiler | WU-29 | 1 |
| ferrotorch-train | WU-30 | 1 |

**Total: 30 units, all parallelizable with worktree isolation.**
