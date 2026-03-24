---
title: "PyTorch Gap Analysis — What ferrotorch Must Do to Be a Real Replacement"
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-24
updated: 2026-03-24
---


## Design Specification

### executive summary

Analysis of PyTorch's source code (cloned at `/home/doll/pytorch-source/`) reveals 7 architectural gaps between ferrotorch and PyTorch, ordered by training performance impact. Closing these gaps would bring ferrotorch from 4.2x slower than PyTorch on Pythia-70M to competitive performance.

### gap 1: sleef vectorized transcendentals (8-19x impact on sigmoid/exp/tanh)

**What PyTorch does:** Uses SLEEF library (`Sleef_expf8_u10`) for 8-wide AVX2 or 16-wide AVX-512 vectorized exp, log, sin, cos. The sigmoid kernel processes 8 floats simultaneously: `neg() → exp() → add(1) → reciprocal()`, all in SIMD registers. Combined with 2x loop unrolling and OpenMP parallelism at 32K grain size.

**What ferrotorch does:** Calls scalar `f32::exp()` (libm) one element at a time. LLVM cannot auto-vectorize libm calls. Parallel threshold is 2M elements — 61x higher than PyTorch's 32K.

**Fix:**
- REQ-1: Integrate `sleef-rs` crate or write hand-rolled SIMD polynomial approximations for exp, log, sin, cos using pulp
- REQ-2: Lower PARALLEL_THRESHOLD from 2M to 32K elements
- REQ-3: Add 2x loop unrolling in SIMD fast paths

**Expected improvement:** sigmoid 1,413 → ~200us (7x), closing 85% of the gap with PyTorch's 169us.

**Files:** `ferrotorch-core/src/ops/elementwise.rs`

### gap 2: zero-copy stride-based views (o(n) → o(1) for transpose/permute)

**What PyTorch does:** `transpose()`, `permute()`, `slice()`, `narrow()`, `expand()` are all O(1) metadata changes — they modify strides and offsets on `TensorImpl` without touching data. A transposed `[512,512]` tensor swaps two integers in the stride array. TensorIterator handles non-contiguous strides transparently in all kernels.

**What ferrotorch does:** `permute_t()` and `transpose()` allocate a new Vec and copy every element with nested index decomposition. A `[512,512]` transpose does 262K copies. The stride field exists on `TensorInner` but is purely decorative — `data()` always returns a contiguous slice ignoring strides.

**Fix:**
- REQ-4: Make transpose/permute return views (swap strides, share Arc<TensorStorage>)
- REQ-5: Add `is_contiguous()` check and `contiguous()` materialization
- REQ-6: Update all SIMD/GPU kernels to handle non-contiguous strides via offset computation (TensorIterator equivalent)

**Expected improvement:** Every attention layer does permute([0,2,1,3]) on [B,S,H,D]. Eliminating this copy saves ~100us per layer on Pythia-70M (6 layers × forward + backward = 12 calls).

**Files:** `ferrotorch-core/src/tensor.rs`, `ferrotorch-core/src/methods.rs`, `ferrotorch-core/src/ops/elementwise.rs`

### gap 3: gpu gradient accumulation roundtrips (gpu training bottleneck)

**What PyTorch does:** Gradient accumulation uses in-place `tensor.add_()` on GPU. `InputBuffer::add()` detects when it holds the last reference and does in-place addition — no new allocation. The backward engine dispatches GPU nodes to per-device worker threads. CUDA stream synchronization ensures correct ordering without global sync.

**What ferrotorch does:** `graph.rs:148-163` downloads EVERY gradient to CPU via `.cpu()`, adds them as CPU scalars, creates a new TensorStorage, and uploads back. For a 70M parameter model, this is ~280MB of GPU→CPU→GPU transfer per backward pass.

**Fix:**
- REQ-7: Gradient accumulation must stay on GPU — use `backend.add_f32()` for same-device tensors
- REQ-8: Use in-place accumulation when the gradient tensor has refcount=1 (steal pattern)
- REQ-9: Add CUDA stream tracking to the backward engine for correct multi-stream ordering

**Expected improvement:** This is THE reason the reporter in issue #14 got only 3x GPU speedup. Fixing this should give ~10-20x GPU backward speedup.

**Files:** `ferrotorch-core/src/autograd/graph.rs`, `ferrotorch-core/src/tensor.rs`

### gap 4: gpu-resident optimizer state (optimizer step bottleneck)

**What PyTorch does:** Optimizer state (exp_avg, exp_avg_sq for Adam) are GPU tensors. FusedAdam processes ALL parameters in a single kernel launch via multi-tensor apply. The kernel reads params, grads, and state from GPU memory, computes updates, and writes back — zero CPU involvement.

**What ferrotorch does:** Optimizer state is `Vec<f64>` on CPU. Every `step()` downloads all parameters and gradients via `data_vec()`, does scalar f64 arithmetic in a Rust loop, then uploads via `update_data()`. For 70M parameters, this is ~560MB of transfers per step.

**Fix:**
- REQ-10: Store optimizer state as GPU `Tensor<f32>` (not `Vec<f64>`)
- REQ-11: Implement optimizer step as composed GPU ops: `exp_avg = beta1 * exp_avg + (1-beta1) * grad` using `scale_f32` + `add_f32`
- REQ-12: Long-term: implement a fused multi-tensor Adam CUDA kernel

**Expected improvement:** Optimizer step from ~200ms → ~5ms for 70M params on GPU.

**Files:** `ferrotorch-optim/src/adam.rs`, `ferrotorch-optim/src/adamw.rs`

### gap 5: gpu kernel vectorized loads (4-8x throughput for memory-bound ops)

**What PyTorch does:** CUDA kernels use `aligned_vector<float, 4>` for 128-bit coalesced loads — each thread processes 4-8 elements. `thread_work_size = 8`, so 128 threads handle 1024 elements per block. For memory-bandwidth-bound elementwise ops, this maximizes PCIe/HBM throughput.

**What ferrotorch does:** Every PTX kernel does scalar `ld.global.f32` — one 32-bit load per thread. 256 threads handle 256 elements per block. This wastes 75% of the memory bus bandwidth since GPU memory transactions are 128 bytes (32 floats) but each thread only uses 4 bytes.

**Fix:**
- REQ-13: Rewrite PTX kernels to use `ld.global.v4.f32` (128-bit vectorized loads) with 4 elements per thread
- REQ-14: Use 128 threads × 8 elements = 1024 elements per block (matching PyTorch)

**Expected improvement:** 2-4x for all GPU elementwise ops.

**Files:** `ferrotorch-gpu/src/kernels.rs`

### gap 6: fp16/bf16 and tensor core support

**What PyTorch does:** All kernels are templated for `float`, `Half`, `BFloat16`. cuBLAS uses `cublasGemmEx` with `CUBLAS_COMPUTE_32F` for fp16 matmul on Tensor Cores — 8-16x throughput on A100/H100. `autocast` context manager automatically manages dtype.

**What ferrotorch does:** Everything is f32-only. No fp16/bf16 kernels, no Tensor Core utilization, no autocast. The GPU matmul uses `cublasSgemm` (f32 only).

**Fix:**
- REQ-15: Add `cublasGemmEx` path for fp16 matmul with f32 accumulation
- REQ-16: Add `autocast` context manager that wraps forward ops in dtype conversion
- REQ-17: Add fp16 storage support in `TensorStorage`

**Expected improvement:** 8-16x matmul throughput on Tensor Core GPUs. Critical for any model >1B params.

**Files:** `ferrotorch-gpu/src/blas.rs`, `ferrotorch-core/src/dtype.rs`, `ferrotorch-core/src/storage.rs`

### gap 7: gpu memory allocator — block splitting and stream awareness

**What PyTorch does:** CUDACachingAllocator allocates large segments (2-20MB), carves them into blocks, and coalesces adjacent free blocks. Size rounding to 512-byte boundaries. Stream-aware deallocation with CUDA events prevents use-after-free across streams. Two pools (small ≤1MB, large >1MB) with best-fit selection.

**What ferrotorch does:** Exact-size matching only — a 1024-element buffer cannot satisfy a 512-element request. No stream tracking (correctness bug). No block splitting or coalescing. Single pool with no size classes.

**Fix:**
- REQ-18: Add size rounding to pool keys (round to next multiple of 128 elements or 512 bytes)
- REQ-19: Add stream field to pool keys — only reuse buffers from the same CUDA stream
- REQ-20: Implement block splitting for the large pool (allocate 2MB segments, carve into smaller blocks)
- REQ-21: Add OOM recovery path that empties the cache and retries

**Expected improvement:** Higher pool hit rate (currently near-zero for variable-size intermediates), elimination of stream-related correctness bugs.

**Files:** `ferrotorch-gpu/src/pool.rs`

### priority ordering for implementation

| Priority | Gap | Impact | Effort | Phase |
|----------|-----|--------|--------|-------|
| 1 | GPU gradient accumulation (#3) | Fixes GPU training being only 3x faster than CPU | Medium | Next sprint |
| 2 | GPU-resident optimizer (#4) | Fixes optimizer being bottleneck for large models | Medium | Next sprint |
| 3 | Zero-copy views (#2) | Eliminates O(N) copies in every attention layer | Large | Architecture change |
| 4 | SLEEF transcendentals (#1) | Closes 85% of CPU sigmoid/exp gap | Medium | Performance sprint |
| 5 | GPU vectorized loads (#5) | 2-4x all GPU elementwise ops | Medium | GPU kernel rewrite |
| 6 | fp16/Tensor Cores (#6) | 8-16x matmul for large models | Large | Feature addition |
| 7 | GPU allocator (#7) | Higher pool hit rates, correctness fix | Medium | Infrastructure |

### missing features (not performance, but blocking adoption)

From the nn module analysis:
- No `autocast` / automatic mixed precision
- No GPU FlashAttention kernel (CPU-only tiled attention exists)
- MultiheadAttention processes batch×heads in serial loops
- ParamGroup only stores lr/weight_decay (not betas/eps per group)
- Missing: Identity, Flatten, L1Loss, NLLLoss, CTCLoss, BatchNorm1d, SyncBatchNorm, EmbeddingBag
- No `torch.compile` / graph-level optimization

ferrotorch has features PyTorch lacks: LoRA, SwiGLU, RoPE, PagedKVCache, Muon optimizer, KFAC.

