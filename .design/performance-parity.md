# Feature: Performance Parity with PyTorch — Eliminate CPU Roundtrips and Close Compute Gaps

## Summary

ferrotorch is 18-28x slower than PyTorch on CPU elementwise ops (scalar loops vs AVX-512), 2.7-5.7x slower on matmul (matrixmultiply vs MKL), and has ~30 GPU backward paths that roundtrip through CPU. This design eliminates all three gaps through: (1) migrating CPU GEMM to faer with pulp SIMD, (2) rewriting elementwise kernels with pulp + rayon, (3) porting all GPU backward ops to CubeCL, and (4) adding JIT operator fusion for elementwise chains. The goal is within 1.5x of PyTorch on CPU and parity on GPU.

## Requirements

- REQ-1: CPU matmul [1024,1024] must complete in under 5,000 us (currently 16,208 us; PyTorch: 2,860 us)
- REQ-2: CPU elementwise add/mul [1000,1000] must complete in under 100 us (currently 917/939 us; PyTorch: 32/52 us)
- REQ-3: CPU transcendental ops (exp, sin, cos, tanh, sigmoid) [1000,1000] must complete in under 500 us each (currently 1,500-3,200 us; PyTorch: 169 us for sigmoid)
- REQ-4: GPU backward passes for softmax, sigmoid, tanh, layernorm, and all arithmetic broadcast ops must execute entirely on GPU with zero CPU roundtrips
- REQ-5: GPU must have native kernels for div, exp, log, sqrt, pow, abs, and per-axis reduction (sum/mean/max along a dimension)
- REQ-6: Elementwise operation chains (e.g., `x.mul(w).add(b).relu()`) must be fusible into a single GPU kernel via the JIT
- REQ-7: `data_vec()` on CPU tensors must not allocate — return a borrowed view or cow type
- REQ-8: Elementwise ops on tensors larger than 32K elements must use rayon parallel iteration
- REQ-9: All GPU kernels must be written in CubeCL for portability across CUDA, ROCm, and WebGPU
- REQ-10: The global allocator must be switched to mimalloc for reduced allocation overhead in autograd graph construction

## Acceptance Criteria

- [ ] AC-1: `cargo run --release --example ferrotorch_bench` reports matmul [1024,1024] < 5,000 us
- [ ] AC-2: Benchmark reports add [1000,1000] < 100 us and mul [1000,1000] < 100 us
- [ ] AC-3: Benchmark reports sigmoid [1000,1000] < 500 us, exp < 300 us, sin/cos < 500 us
- [ ] AC-4: `SoftmaxBackward`, `SigmoidBackward`, `TanhBackward`, `LayerNormBackward`, and `reduce_grad_to_shape` have no `ensure_cpu`/`.cpu()?` calls in their implementations
- [ ] AC-5: `GpuBackend` trait has methods for `div_f32`, `exp_f32`, `log_f32`, `sqrt_f32`, `pow_f32`, `abs_f32`, `sum_axis_f32`, `mean_axis_f32`, `max_axis_f32`
- [ ] AC-6: `ferrotorch-jit/src/fusion.rs` can fuse a chain of 3+ elementwise ops into a single CubeCL kernel and execute it on GPU
- [ ] AC-7: `Tensor::data_vec()` returns `Cow<[T]>` — `Cow::Borrowed` for CPU, `Cow::Owned` for GPU
- [ ] AC-8: Adding two [100000,100] tensors uses rayon (verified by thread count or explicit test)
- [ ] AC-9: All new GPU kernels in ferrotorch-gpu compile and run via CubeCL's CUDA backend, with `#[cube]` annotated functions
- [ ] AC-10: `ferrotorch/src/lib.rs` or the umbrella crate sets `#[global_allocator] static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;`
- [ ] AC-11: Full test suite (`cargo test`) passes with zero regressions
- [ ] AC-12: MLP training step B=32 benchmark < 800 us (currently 1,254 us; PyTorch: 535 us)

## Architecture

### Phase 1: Zero-Copy CPU Data Access (REQ-7)

**File: `ferrotorch-core/src/tensor.rs`**

Change `data_vec()` return type from `Vec<T>` to `Cow<'_, [T]>`:

```rust
pub fn data_vec(&self) -> FerrotorchResult<Cow<'_, [T]>> {
    if self.is_cuda() {
        let cpu_tensor = self.cpu()?;
        Ok(Cow::Owned(cpu_tensor.data()?.to_vec()))
    } else {
        Ok(Cow::Borrowed(self.data()?))
    }
}
```

This eliminates the unnecessary `.to_vec()` copy for every CPU tensor operation. Every call site that does `data_vec()?.iter()` continues to work unchanged since `Cow<[T]>` derefs to `&[T]`. Call sites that need ownership call `.into_owned()`.

**Impact:** Eliminates ~1M element copies per elementwise op on [1000,1000] tensors. This single change should recover the 4-5x regression introduced by the GPU safety audit.

**Propagation:** All 39 files modified in the GPU audit use `data_vec()` — they all benefit automatically.

### Phase 2: CPU Matmul Migration to faer (REQ-1)

**File: `ferrotorch-core/src/ops/linalg.rs`**

Replace `matrixmultiply::sgemm`/`dgemm` in `mm_raw()` (line 246) with `faer::linalg::matmul::matmul()`:

```rust
pub fn mm_raw<T: Float>(a_data: &[T], b_data: &[T], m: usize, k: usize, n: usize) -> Vec<T> {
    let mut c = vec![T::zero(); m * n];
    // faer uses MatRef/MatMut views over raw slices — zero-copy
    let a_mat = faer::mat::from_raw_parts::<T>(a_data.as_ptr(), m, k, k as isize, 1);
    let b_mat = faer::mat::from_raw_parts::<T>(b_data.as_ptr(), k, n, n as isize, 1);
    let mut c_mat = faer::mat::from_raw_parts_mut::<T>(c.as_mut_ptr(), m, n, n as isize, 1);
    faer::linalg::matmul::matmul(c_mat.as_mut(), a_mat.as_ref(), b_mat.as_ref(), None, T::one(), faer::Parallelism::Rayon(0));
    c
}
```

faer 0.24 uses pulp for SIMD dispatch (SSE2/AVX2/AVX-512 at runtime) and rayon for threading. It matches OpenBLAS and reaches ~80-90% of MKL on Intel. The `Parallelism::Rayon(0)` parameter uses all available cores.

**Dependency change:** `ferray-linalg 0.2.5` already depends on `faer 0.24`. Add a direct `faer = "0.24"` dependency to `ferrotorch-core/Cargo.toml` for the matmul API.

Remove `matrixmultiply` from `Cargo.toml` after migration.

The `DIRECT_MM_THRESHOLD = 128` can be removed — faer handles small matrices efficiently internally.

### Phase 3: Elementwise Kernel Rewrite with pulp + rayon (REQ-2, REQ-3, REQ-8)

**File: `ferrotorch-core/src/ops/elementwise.rs`**

#### 3a: pulp SIMD for all elementwise ops

Replace ferray-ufunc kernel calls with direct pulp SIMD. pulp 0.22 (already a transitive dep via faer) provides runtime SIMD dispatch:

```rust
use pulp::Arch;

fn simd_add_f32(a: &[f32], b: &[f32], out: &mut [f32]) {
    let arch = Arch::new();
    arch.dispatch(|| {
        // pulp auto-selects SSE2/AVX2/AVX-512
        for i in 0..a.len() {
            out[i] = a[i] + b[i];
        }
    });
}
```

pulp's `Arch::dispatch` compiles the closure for multiple SIMD widths and selects the best at runtime. For transcendentals (sin, cos, tanh, sigmoid), implement SIMD-friendly polynomial approximations:

- **exp:** Cody-Waite range reduction + degree-6 minimax polynomial (matches glibc accuracy)
- **sin/cos:** Payne-Hanek reduction + degree-11 Chebyshev polynomial
- **tanh:** `tanh(x) = 1 - 2/(exp(2x)+1)` using the SIMD exp
- **sigmoid:** `sigmoid(x) = 1/(1+exp(-x))` using the SIMD exp

These approximations vectorize perfectly and achieve 4-8x speedup over scalar `f32::sin()` etc.

#### 3b: rayon parallelism for large tensors

Wrap SIMD kernels in rayon parallel iteration above a threshold:

```rust
const PARALLEL_THRESHOLD: usize = 32_768; // 32K elements

fn fast_add_impl(a: &[f32], b: &[f32], out: &mut [f32]) {
    if a.len() >= PARALLEL_THRESHOLD {
        let chunk_size = a.len() / rayon::current_num_threads();
        a.par_chunks(chunk_size)
            .zip(b.par_chunks(chunk_size))
            .zip(out.par_chunks_mut(chunk_size))
            .for_each(|((a_chunk, b_chunk), out_chunk)| {
                simd_add_f32(a_chunk, b_chunk, out_chunk);
            });
    } else {
        simd_add_f32(a, b, out);
    }
}
```

#### 3c: broadcast loop fusion

Replace the per-element `broadcast_index()` call in `binary_map` with precomputed stride arrays:

```rust
fn broadcast_binary_op<T, F>(a: &[T], b: &[T], a_shape: &[usize], b_shape: &[usize], out_shape: &[usize], f: F) -> Vec<T>
where F: Fn(T, T) -> T
{
    let a_strides = broadcast_strides(a_shape, out_shape);
    let b_strides = broadcast_strides(b_shape, out_shape);
    let mut out = Vec::with_capacity(out_numel);
    // Single pass with precomputed strides — no per-element modulo
    for i in 0..out_numel {
        let a_idx = compute_index(i, &a_strides, out_shape);
        let b_idx = compute_index(i, &b_strides, out_shape);
        out.push(f(a[a_idx], b[b_idx]));
    }
    out
}
```

The key insight: `broadcast_strides()` precomputes a stride of 0 for broadcast dimensions and the real stride otherwise. `compute_index` then uses only multiplication and addition (no modulo), which is 3-5x faster.

### Phase 4: CubeCL GPU Kernels (REQ-5, REQ-9)

**New file: `ferrotorch-gpu/src/cubecl_kernels.rs`**

Write all missing GPU kernels using CubeCL's `#[cube]` macro:

```rust
use cubecl::prelude::*;

#[cube(launch)]
fn div_kernel(input_a: &Tensor<f32>, input_b: &Tensor<f32>, output: &mut Tensor<f32>) {
    let idx = ABSOLUTE_POS;
    output[idx] = input_a[idx] / input_b[idx];
}

#[cube(launch)]
fn exp_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>) {
    let idx = ABSOLUTE_POS;
    output[idx] = f32::exp(input[idx]);
}

#[cube(launch)]
fn sum_axis_kernel(input: &Tensor<f32>, output: &mut Tensor<f32>, #[comptime] axis: u32, #[comptime] axis_size: u32) {
    let out_idx = ABSOLUTE_POS;
    let mut sum = 0.0f32;
    for i in 0..axis_size {
        // Compute input index from output index + axis position
        sum += input[compute_input_idx(out_idx, i, axis)];
    }
    output[out_idx] = sum;
}
```

**Kernels to implement:**
1. `div_f32` — elementwise division
2. `exp_f32`, `log_f32`, `sqrt_f32`, `pow_f32`, `abs_f32` — unary transcendentals
3. `sigmoid_f32`, `tanh_f32` — activation forward (currently CPU-only)
4. `sum_axis_f32`, `mean_axis_f32`, `max_axis_f32` — per-axis reductions
5. `sigmoid_backward_f32`, `tanh_backward_f32`, `softmax_backward_f32` — activation backward
6. `layernorm_backward_f32` — normalization backward
7. `reduce_to_shape_f32` — broadcast gradient reduction (eliminates the #1 GPU roundtrip source)

**Integration with GpuBackend trait:**

Add new methods to `ferrotorch-core/src/gpu_dispatch.rs`:
```rust
fn div_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
fn exp_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
fn sum_axis_f32(&self, a: &GpuBufferHandle, shape: &[usize], axis: usize) -> FerrotorchResult<GpuBufferHandle>;
// ... etc
```

CubeCL kernels compile to CUDA PTX, ROCm HIP, or WGPU compute shaders depending on the runtime. The `ferrotorch-cubecl` crate already has the CubeCL dependency wired up — the `cubecl_kernels.rs` implementations register with the existing `GpuBackend` trait.

### Phase 5: Eliminate GPU Backward Roundtrips (REQ-4)

**Files: `ferrotorch-core/src/grad_fns/activation.rs`, `arithmetic.rs`, `reduction.rs`**

With the Phase 4 GPU kernels available, rewrite backward implementations to compose GPU ops:

**SoftmaxBackward** (currently 3 roundtrips):
```rust
// BEFORE: ensure_cpu → CPU dot product → CPU broadcast → restore_device
// AFTER: all GPU ops
fn backward(&self, grad_output: &Tensor<T>) -> ... {
    if grad_output.is_cuda() {
        no_grad(|| {
            // dot = sum(grad * softmax, dim=-1, keepdim=true)
            let gs = mul(grad_output, &self.output)?;
            let dot = sum_axis(&gs, -1, true)?;
            // grad_input = softmax * (grad - dot)
            let diff = sub(grad_output, &dot)?;
            mul(&self.output, &diff)
        })
    } else { /* existing CPU path */ }
}
```

**reduce_grad_to_shape** (currently 2 roundtrips per broadcast backward):
```rust
// BEFORE: gpu_grad.cpu() → CPU sum over axes → result.to(device)
// AFTER: GPU sum_axis kernel
fn reduce_grad_to_shape(grad: &Tensor<T>, target_shape: &[usize]) -> ... {
    if grad.is_cuda() {
        // Use GPU sum_axis_f32 for each dimension that was broadcast
        for axis in broadcast_axes(grad.shape(), target_shape) {
            grad = sum_axis(&grad, axis, false)?;
        }
        Ok(grad)
    } else { /* existing CPU path */ }
}
```

Same pattern for `SigmoidBackward` (`grad * output * (1 - output)` using GPU mul/sub), `TanhBackward` (`grad * (1 - output^2)` using GPU mul/sub/pow), and `LayerNormBackward`.

### Phase 6: JIT Operator Fusion (REQ-6)

**Files: `ferrotorch-jit/src/fusion.rs`, new `ferrotorch-jit/src/gpu_fusion.rs`**

Extend the existing JIT fusion engine to generate fused CubeCL kernels:

**Architecture:**
1. **Trace capture:** During eager execution, buffer a sequence of elementwise operations into an op tape (already partially implemented in `fusion.rs`)
2. **Pattern detection:** Identify fusible chains — sequences of unary/binary elementwise ops with no shape changes or reductions
3. **Code generation:** Generate a single CubeCL `#[cube]` function that applies the entire chain in one kernel launch
4. **Compilation and caching:** Use CubeCL's JIT compilation to produce a CUDA PTX module, cache it by op-chain signature
5. **Dispatch:** Replace the eager op sequence with a single fused kernel call

**Example fusion:**
```rust
// Eager: 3 kernel launches, 3 global memory round-trips
let y = x.mul(&w)?.add(&b)?.relu()?;

// Fused: 1 kernel launch, 1 read + 1 write
// Generated CubeCL kernel:
#[cube(launch)]
fn fused_mul_add_relu(x: &Tensor<f32>, w: &Tensor<f32>, b: &Tensor<f32>, out: &mut Tensor<f32>) {
    let i = ABSOLUTE_POS;
    let v = x[i] * w[i] + b[i];
    out[i] = if v > 0.0 { v } else { 0.0 };
}
```

**Scope constraint:** Start with GPU-only fusion for elementwise chains. CPU fusion and reduction fusion are future work.

### Phase 7: Global Allocator (REQ-10)

**File: `ferrotorch/src/lib.rs`**

```rust
#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;
```

Add `mimalloc = { version = "0.1", default-features = false }` to `ferrotorch/Cargo.toml`.

This is a 2-line change that reduces allocation overhead for the many small `Vec`, `Arc`, and `Box` allocations in autograd graph construction. Expected 5-15% improvement in training step throughput based on mimalloc benchmarks.

### Phase 8: Build Configuration

**New file: `.cargo/config.toml`**

```toml
[target.'cfg(target_arch = "x86_64")']
rustflags = ["-C", "target-cpu=native"]
```

This enables the compiler to use all SIMD extensions available on the build machine for auto-vectorization of non-hot-path code. The hot paths use explicit pulp SIMD and do not depend on this flag, but secondary code paths benefit. Document that this is for development/local builds — release builds should target a broader ISA baseline.

### Dependency Changes Summary

| Crate | Action | Cargo.toml |
|-------|--------|------------|
| `faer` | Add direct dep | `ferrotorch-core/Cargo.toml` |
| `pulp` | Add direct dep | `ferrotorch-core/Cargo.toml` |
| `mimalloc` | Add dep | `ferrotorch/Cargo.toml` |
| `matrixmultiply` | Remove | `ferrotorch-core/Cargo.toml` |
| `ferray-ufunc` | Keep (used elsewhere) | unchanged |
| `cubecl` | Already present | unchanged |

### Execution Order

Phases are ordered by impact-to-effort ratio:

1. **Phase 1** (data_vec Cow) — 1 file, instant 3-5x elementwise recovery
2. **Phase 7** (mimalloc) — 2 lines, free 5-15% training throughput
3. **Phase 2** (faer matmul) — 1 function, 3-5x matmul speedup
4. **Phase 3** (pulp elementwise) — 1 file, 5-10x elementwise speedup
5. **Phase 8** (target-cpu=native) — 1 file, free auto-vectorization bonus
6. **Phase 4** (CubeCL GPU kernels) — new file, unblocks Phase 5
7. **Phase 5** (GPU backward elimination) — ~10 files, eliminates 30 roundtrips
8. **Phase 6** (JIT fusion) — 2 files, biggest long-term win

## Open Questions

### Q1: faer f32 API availability — RESOLVED
Benchmarked faer 0.24 f32 GEMM on [1024,1024]:
- matrixmultiply sgemm: 15,621 us
- faer sequential: 13,219 us (1.18x faster)
- faer + rayon: 3,794 us (4.12x faster)
- PyTorch MKL: 2,860 us (1.33x faster than faer+rayon)

faer f32 works natively via `MatRef::from_row_major_slice` — no conversion overhead. Max numerical difference: 0.000244. **Decision: full migration to faer for f32 and f64 GEMM.**

### Q2: CubeCL compilation model — RESOLVED
**Decision: Accept JIT.** First training step will be slower, subsequent steps use cached kernels. This matches PyTorch's torch.compile/triton model and avoids startup complexity.

### Q3: data access pattern — RESOLVED
**Decision: Option (a).** Add `data_ref() -> &[T]` for CPU-only zero-copy access (alias for existing `data()`). Keep `data_vec() -> Vec<T>` for GPU-safe access. Teach hot paths to branch: `is_cuda()` → tensor ops stay on device, else → `data_ref()` for zero-copy CPU.

## Out of Scope

- AMD ROCm or Apple Metal backend testing (CubeCL supports them but we only validate CUDA)
- f16/bf16 mixed-precision kernels (separate feature)
- Multi-GPU (NCCL) communication optimization
- Custom CUDA graph capture for inference
- Autograd tape arena allocation with bumpalo (investigate separately)
- CPU operator fusion in ferrotorch-jit (GPU fusion only in this design)
- Convolution kernel optimization (cuDNN integration is a separate effort)
- MKL optional backend behind feature flag (faer is sufficient for now)
