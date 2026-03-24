# Full PyTorch Gap Analysis — Complete Subsystem Comparison

## Overview

17 PyTorch subsystems analyzed across 5+12 agent investigations. This document catalogs every gap between ferrotorch and PyTorch, organized into swarmable implementation tasks.

---

## TIER 1: Blocks GPU Training Performance (Do First)

### 1.1 GPU Gradient Accumulation Roundtrips Through CPU
- **PyTorch**: In-place `tensor.add_()` on GPU with stream synchronization
- **ferrotorch**: `graph.rs:148-163` downloads EVERY gradient to CPU, adds as scalars, re-uploads
- **Impact**: THE reason GPU training is only 3x faster than CPU
- **Fix**: Use `backend.add_f32()` for same-device GPU gradients in `graph.rs`
- **Files**: `ferrotorch-core/src/autograd/graph.rs`

### 1.2 GPU-Resident Optimizer State
- **PyTorch**: Optimizer state (exp_avg, exp_avg_sq) are GPU tensors; FusedAdam does ALL params in one kernel
- **ferrotorch**: State is `Vec<f64>` on CPU; every step downloads all params/grads, does scalar f64 loops, uploads back
- **Impact**: ~200ms per step for 70M params (should be ~5ms)
- **Fix**: Store state as `Tensor<f32>` on GPU, compose optimizer step from GPU ops
- **Files**: `ferrotorch-optim/src/adam.rs`, `adamw.rs`, `rmsprop.rs`, `sgd.rs`

### 1.3 Zero-Copy Stride-Based Views (Transpose/Permute/Slice)
- **PyTorch**: O(1) metadata change — swap strides, no data copy. TensorIterator handles non-contiguous strides
- **ferrotorch**: O(N) full data copy for every permute/transpose. Stride fields exist but are decorative
- **Impact**: Every attention layer does permute([0,2,1,3]) — 100us wasted per layer
- **Fix**: Make transpose/permute swap strides, add `is_contiguous()` + `contiguous()`, update kernels for stride awareness
- **Files**: `ferrotorch-core/src/tensor.rs`, `methods.rs`, `ops/elementwise.rs`, `ops/linalg.rs`

### 1.4 SLEEF Vectorized Transcendentals
- **PyTorch**: `Sleef_expf8_u10` — 8-wide AVX2 exp in one call, plus OpenMP at 32K grain
- **ferrotorch**: Scalar `f32::exp()` one at a time, parallel threshold at 2M
- **Impact**: sigmoid is 8.4x slower than PyTorch
- **Fix**: Integrate `sleef-rs` or write polynomial SIMD approximations via pulp; lower parallel threshold to 32K
- **Files**: `ferrotorch-core/src/ops/elementwise.rs`

### 1.5 GPU Kernel Vectorized Loads
- **PyTorch**: `ld.global.v4.f32` — 128-bit loads, 8 elements per thread, 128 threads = 1024 per block
- **ferrotorch**: `ld.global.f32` — 32-bit loads, 1 element per thread, 256 threads = 256 per block
- **Impact**: 4x memory bandwidth waste for all GPU elementwise ops
- **Fix**: Rewrite PTX kernels with vec4 loads, 4-8 elements per thread
- **Files**: `ferrotorch-gpu/src/kernels.rs`

---

## TIER 2: Blocks Production GPU Training (Do Second)

### 2.1 CUDA Stream Management
- **PyTorch**: Per-device stream pool (32 streams), thread-local current stream, CUDA events, stream-aware autograd
- **ferrotorch**: One stream per device, no events, no stream awareness in autograd or allocator
- **Impact**: No compute/communication overlap, correctness bugs with multi-stream
- **Files**: `ferrotorch-gpu/src/device.rs`, new `stream.rs`

### 2.2 fp16/bf16 + Tensor Cores
- **PyTorch**: All kernels templated for f16/bf16, `cublasGemmEx` with `CUBLAS_COMPUTE_32F` for Tensor Cores (8-16x throughput)
- **ferrotorch**: Everything f32-only, no Tensor Core utilization
- **Impact**: 8-16x matmul throughput left on the table
- **Files**: `ferrotorch-gpu/src/blas.rs`, `ferrotorch-core/src/dtype.rs`

### 2.3 Autocast (AMP)
- **PyTorch**: Dispatch-key interception casts ops to fp16/bf16 automatically with weight caching
- **ferrotorch**: Policy engine exists (`autocast.rs`) but never actually casts tensors
- **Impact**: No automatic mixed precision — users must manually manage dtypes
- **Files**: `ferrotorch-core/src/autograd/autocast.rs`, `autocast_ops.rs`

### 2.4 GPU Allocator Block Splitting + Stream Awareness
- **PyTorch**: 2-pool design, block splitting/coalescing, stream-aware deallocation, OOM recovery
- **ferrotorch**: Exact-size matching only, no stream tracking (correctness bug), no OOM recovery
- **Impact**: Low pool hit rates, potential data corruption with multi-stream
- **Files**: `ferrotorch-gpu/src/pool.rs`

### 2.5 Multi-Threaded Backward Engine
- **PyTorch**: Per-device worker threads, priority queue scheduling, in-place gradient accumulation, gradient stealing
- **ferrotorch**: Single-threaded Kahn's algorithm, always allocates new tensors for accumulation
- **Impact**: Cannot overlap CPU and GPU backward work; excessive allocations
- **Files**: `ferrotorch-core/src/autograd/graph.rs`

### 2.6 DDP Gradient Bucketing + Async AllReduce
- **PyTorch**: 25MB buckets with autograd hooks, async allreduce overlaps backward computation, ring topology
- **ferrotorch**: Per-parameter sequential allreduce after backward, star topology through rank 0
- **Impact**: No communication/computation overlap, O(N) bottleneck at rank 0
- **Files**: `ferrotorch-distributed/src/ddp.rs`, `collective.rs`

---

## TIER 3: Feature Completeness (Do Third)

### 3.1 Kernel Fusion / JIT Compilation
- **PyTorch Inductor**: FX graph → lowering → IR → scheduler/fusion → Triton/C++ codegen. Fuses arbitrary pointwise DAGs.
- **ferrotorch-jit**: Linear chain fusion only (`FusedElementwise`), PTX generation for simple chains, no multi-input DAG fusion
- **Minimum viable**: Extend fusion to multi-input DAGs, wire PTX generation to actual GPU execution
- **Files**: `ferrotorch-jit/src/fusion.rs`, `codegen.rs`, `optimize.rs`

### 3.2 Gradient Checkpointing
- **PyTorch**: Multi-tensor input, RNG state preservation, autocast state preservation, non-reentrant mode
- **ferrotorch**: Single-tensor input, no RNG/autocast state preservation, scalar reduction trick for backward
- **Files**: `ferrotorch-core/src/autograd/checkpoint.rs`

### 3.3 GradScaler Improvements
- **PyTorch**: Fused unscale+inf-check CUDA kernel, fused optimizer integration, device-tensor found_inf
- **ferrotorch**: CPU scalar loop for unscale, bool found_inf (forces GPU sync), no optimizer integration
- **Files**: `ferrotorch-optim/src/grad_scaler.rs`

### 3.4 CUDA Graph Improvements
- **PyTorch**: Allocator pool redirection during capture, graph pool sharing, RNG state management, capture status query
- **ferrotorch**: Basic capture/replay works but no allocator integration, no RNG management
- **Files**: `ferrotorch-gpu/src/graph.rs`

### 3.5 DataLoader Prefetching + pin_memory
- **PyTorch**: Multi-process workers, prefetch pipeline, pin_memory thread for async H2D
- **ferrotorch**: Synchronous rayon-based loading, no prefetch, no pin_memory
- **Files**: `ferrotorch-data/src/dataloader.rs`

### 3.6 GPU Profiler Events
- **PyTorch**: CUPTI/Kineto integration, per-kernel CUDA event timing, memory lifecycle tracking
- **ferrotorch**: CPU wall-clock only, no GPU kernel timing
- **Files**: `ferrotorch-profiler/src/profiler.rs`

### 3.7 Missing NN Modules
- Identity, Flatten, L1Loss, NLLLoss, CTCLoss, BatchNorm1d, SyncBatchNorm, EmbeddingBag
- MultiheadAttention serial batch/head loops need batched matmul
- ParamGroup only stores lr/weight_decay (not betas/eps per group)
- **Files**: `ferrotorch-nn/src/`

### 3.8 GPU Linalg via cuSOLVER
- **PyTorch**: SVD, Cholesky, LU, QR, eigendecomp all on GPU via cuSOLVER
- **ferrotorch**: CPU-only via faer, no autograd for linalg ops
- **Files**: `ferrotorch-core/src/linalg.rs`, new `ferrotorch-gpu/src/cusolver.rs`

### 3.9 FSDP (Fully Sharded Data Parallel)
- **PyTorch**: Parameter/gradient/optimizer sharding, all-gather before forward, reduce-scatter after backward
- **ferrotorch**: Nothing
- **Files**: new `ferrotorch-distributed/src/fsdp.rs`

---

## TIER 4: Advanced Features (Do Later)

### 4.1 Higher-Order Ops (cond, scan, flex_attention)
### 4.2 AOT Autograd (backward graph compilation)
### 4.3 Inductor-Style Codegen (Triton/C++ backends)
### 4.4 Nested/Jagged Tensors (variable-length sequences)
### 4.5 Semi-Structured 2:4 Sparsity
### 4.6 Quantization-Aware Training
### 4.7 Distributed Checkpointing + Resharding
### 4.8 RPC / Pipeline Parallelism
### 4.9 Model Export (ExportedProgram, dynamic shapes)
### 4.10 CUDA RNG State Management

---

## Swarmable Task Breakdown

Each task below is independent and can be assigned to a separate agent:

### TIER 1 (6 tasks — highest impact)
| Task | Files | Complexity | Dependencies |
|------|-------|-----------|-------------|
| T1.1 GPU gradient accumulation | graph.rs | Medium | None |
| T1.2 GPU-resident optimizer | adam.rs, adamw.rs, rmsprop.rs, sgd.rs | Large | None |
| T1.3 Zero-copy stride views | tensor.rs, methods.rs | Large | None |
| T1.4 SLEEF vectorized math | elementwise.rs | Medium | None |
| T1.5 Vectorized GPU kernel loads | kernels.rs | Medium | None |
| T1.6 Lower parallel threshold to 32K | elementwise.rs | Trivial | None |

### TIER 2 (6 tasks — production GPU)
| Task | Files | Complexity | Dependencies |
|------|-------|-----------|-------------|
| T2.1 CUDA stream pool + events | device.rs, new stream.rs | Large | None |
| T2.2 fp16/bf16 kernels + cublasGemmEx | blas.rs, dtype.rs, kernels.rs | Large | None |
| T2.3 Wire autocast to actually cast | autocast.rs, autocast_ops.rs | Medium | T2.2 |
| T2.4 GPU allocator block splitting | pool.rs | Large | T2.1 |
| T2.5 Multi-threaded backward engine | graph.rs | Large | T2.1 |
| T2.6 DDP bucketing + ring allreduce | ddp.rs, collective.rs | Large | None |

### TIER 3 (10 tasks — feature completeness)
| Task | Files | Complexity | Dependencies |
|------|-------|-----------|-------------|
| T3.1 JIT multi-input fusion + GPU exec | fusion.rs, codegen.rs | Large | None |
| T3.2 Checkpoint RNG + multi-tensor | checkpoint.rs | Medium | None |
| T3.3 GradScaler GPU kernel | grad_scaler.rs | Medium | None |
| T3.4 CUDA graph allocator integration | graph.rs, pool.rs | Medium | T2.1, T2.4 |
| T3.5 DataLoader prefetch + pin_memory | dataloader.rs | Medium | None |
| T3.6 GPU profiler events | profiler.rs | Medium | T2.1 |
| T3.7 Missing nn modules | nn/src/ | Medium | None |
| T3.8 GPU linalg via cuSOLVER | linalg.rs, new cusolver.rs | Large | None |
| T3.9 FSDP | new fsdp.rs | Large | T2.6 |
| T3.10 MHA batched matmul | attention.rs | Medium | None |
