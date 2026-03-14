---
title: "Phase 1 — Autograd Engine (ferrotorch-core)"
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-14
updated: 2026-03-14
---


## Design Specification

### Summary

The foundational crate of ferrotorch: a `Tensor<T>` type with dynamic shapes, owned-buffer storage with device tagging, and a reverse-mode automatic differentiation engine. Every downstream crate (nn, optim, data, vision, gpu) depends on this. Nothing else ships until this works and is numerically verified against PyTorch.

### Requirements

- REQ-1: `Tensor<T>` must be a dynamically-shaped tensor type parameterized only by element type `T`, with shape represented as `Vec<usize>` and strides as `Vec<isize>`. It must support f32, f64, and bf16 (once ferray-core adds bf16 to the `Element` trait).
- REQ-2: `TensorStorage<T>` must own its data buffer directly (`Vec<T>` for CPU) rather than wrapping `ferray_core::Array`. It must carry a `Device` tag. The storage design must be extensible to GPU device pointers without refactoring the `Tensor` type.
- REQ-3: The autograd engine must implement reverse-mode automatic differentiation via a dynamic computation graph. Graphs are built during the forward pass and consumed (dropped) during backward. Graph nodes must be reference-counted (`Arc`) for shared inputs.
- REQ-4: `tensor.backward()` must compute gradients for all leaf tensors that contributed to the output, using topological sort and reverse traversal. Gradients on shared inputs must accumulate additively.
- REQ-5: A `GradFn<T>` trait must define the backward (VJP) contract. Every differentiable operation in the grad_fns module must implement this trait with a correct vector-Jacobian product.
- REQ-6: VJP implementations must exist for all operations listed in the grad_fns table below (arithmetic, reduction, linalg, activation, shape, indexing, comparison categories — ~25 math operations). Layer-level grad_fns (conv, pool, norm, dropout, embedding, loss, attention) are deferred to ferrotorch-nn (Phase 2).
- REQ-7: `no_grad()` must disable gradient tracking for its closure scope using thread-local state. Tensors created inside `no_grad()` must have `requires_grad = false` regardless of input tensors.
- REQ-8: Gradient checkpointing must allow trading compute for memory by recomputing intermediate activations during backward instead of storing them.
- REQ-9: All public functions must return `Result<T, FerrotorchError>`. No panics on invalid input (shape mismatch, dimension out of bounds, type mismatch, etc.).
- REQ-10: `Tensor<T>` and all autograd types must be `Send + Sync`. Computation graphs must be safe to build on one thread and backward on another.
- REQ-11: The crate must provide factory functions for tensor creation: `zeros`, `ones`, `full`, `rand`, `randn`, `from_slice`, `from_vec`, `eye`, `arange`, `linspace` — matching PyTorch's creation API.
- REQ-12: Tensor operations must delegate to ferray-ufunc for CPU elementwise math (exp, log, sin, cos, etc.) and ferray-linalg for linear algebra (matmul, decompositions) for both f32 and f64. ferray-linalg f32 support is an external prerequisite that must be completed before Phase 1 starts.

### Acceptance Criteria

- [ ] AC-1: `Tensor<f32>` and `Tensor<f64>` can be constructed via `zeros([2, 3])`, `ones([2, 3])`, `rand([2, 3])`, `from_slice(&[1.0, 2.0, 3.0], &[3])` and the resulting shape, strides, and data are correct.
- [ ] AC-2: `TensorStorage` holds a `Vec<T>` for CPU and carries `Device::Cpu`. Adding a `Device::Cuda(0)` variant compiles without changing `Tensor<T>`'s public API.
- [ ] AC-3: Forward operations (add, mul, matmul, relu, etc.) on tensors with `requires_grad = true` build a computation graph. The graph can be inspected to verify the correct number of nodes and edges.
- [ ] AC-4: `tensor.backward()` on a scalar output computes correct gradients on all leaf inputs. Verified numerically against PyTorch for at least 20 representative computation graphs (see test matrix below).
- [ ] AC-5: Gradients through shared inputs accumulate correctly: if `c = a + a`, then `c.backward()` produces `a.grad = 2.0`.
- [ ] AC-6: `no_grad(|| { ... })` produces tensors with no grad_fn, even when inputs have `requires_grad = true`. Exiting the closure restores gradient tracking.
- [ ] AC-7: Gradient checkpointing reduces peak memory usage by at least 40% on a 50-layer deep chain (e.g., 50 sequential matmuls) compared to standard backward, while producing identical gradients.
- [ ] AC-8: Every VJP in the grad_fns table passes a numerical gradient check (finite differences with tolerance `rtol=1e-4, atol=1e-6` for f32, `rtol=1e-7, atol=1e-10` for f64).
- [ ] AC-9: All public functions return `Result`. Passing a shape-mismatched pair to `add()` returns `Err(FerrotorchError::ShapeMismatch { .. })`, not a panic.
- [ ] AC-10: `Tensor<f32>` matmul produces correct results via ferray-linalg's f32 path (not f64 promotion). Verified against PyTorch `torch.mm()` output.
- [ ] AC-11: `cargo test -p ferrotorch-core` passes with 0 failures. Minimum 200 tests covering all grad_fns, creation functions, error paths, and edge cases (empty tensors, scalar tensors, 0-dim, large rank).
- [ ] AC-12: `Tensor` is `Send + Sync` — a test spawns a thread, builds a graph, sends the output tensor to another thread, and calls `backward()` successfully.

### Architecture

### Crate Layout

```
ferrotorch-core/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports
│   ├── tensor.rs                 # Tensor<T> — the central type
│   ├── storage.rs                # TensorStorage<T> — owned buffer + device
│   ├── device.rs                 # Device enum (Cpu, Cuda(usize))
│   ├── dtype.rs                  # DType re-export from ferray + bf16
│   ├── shape.rs                  # Shape utilities: broadcast, stride calc, indexing
│   ├── error.rs                  # FerrotorchError enum
│   ├── creation.rs               # zeros, ones, rand, randn, from_slice, etc.
│   ├── autograd/
│   │   ├── mod.rs
│   │   ├── graph.rs              # GraphNode, topological sort, backward engine
│   │   ├── function.rs           # GradFn<T> trait
│   │   ├── no_grad.rs            # Thread-local gradient enable/disable
│   │   └── checkpoint.rs         # Gradient checkpointing (recompute vs store)
│   ├── grad_fns/                 # Math ops only — layer ops (conv, pool, norm, etc.) move to ferrotorch-nn
│   │   ├── mod.rs
│   │   ├── arithmetic.rs         # add, sub, mul, div, neg, pow, sqrt, abs
│   │   ├── reduction.rs          # sum, mean, prod
│   │   ├── linalg.rs             # matmul, bmm, mm, mv, dot
│   │   ├── activation.rs         # relu, sigmoid, tanh, gelu, silu, softmax, log_softmax
│   │   ├── shape.rs              # reshape, transpose, permute, expand, contiguous, cat, stack, split, squeeze, unsqueeze, flatten
│   │   ├── indexing.rs           # gather, scatter_add, index_select, masked_fill
│   │   └── comparison.rs         # where_ (differentiable through selected branch)
│   └── ops/
│       ├── mod.rs
│       ├── elementwise.rs        # Tensor method wrappers that call ferray-ufunc and record grad_fns
│       ├── linalg.rs             # matmul via ferray-linalg (f32/f64 unified)
│       └── creation.rs           # Tensor factory methods
└── tests/
    ├── test_tensor.rs            # Construction, shape, dtype, device
    ├── test_autograd.rs          # Graph building, backward, accumulation
    ├── test_grad_fns.rs          # Numerical gradient checks for every VJP
    ├── test_no_grad.rs           # Gradient suppression
    ├── test_checkpoint.rs        # Memory savings verification
    └── test_thread_safety.rs     # Send + Sync across threads
```

### Core Types

**Tensor<T>** (`tensor.rs`):
```rust
pub struct Tensor<T: Element = f32> {
    storage: Arc<TensorStorage<T>>,   // Shared ownership for graph references
    shape: Vec<usize>,
    strides: Vec<isize>,
    offset: usize,                    // Offset into storage for views
    grad: Mutex<Option<Box<Tensor<T>>>>,     // Interior mutability for backward (Sync-safe)
    grad_fn: Option<Arc<dyn GradFn<T>>>,
    requires_grad: bool,
    is_leaf: bool,
}
```

`Arc<TensorStorage<T>>` enables multiple graph nodes to reference the same data without cloning. `Mutex` on `grad` allows backward to write gradients without `&mut self` while keeping `Tensor<T>` `Send + Sync`. Lock contention is negligible — backward is single-threaded by default and math ops dominate runtime.

**TensorStorage<T>** (`storage.rs`):
```rust
pub struct TensorStorage<T: Element> {
    data: StorageBuffer<T>,
    device: Device,
}

pub enum StorageBuffer<T: Element> {
    Cpu(Vec<T>),
    // Future: Cuda(CudaBuffer<T>),
    // Future: Metal(MetalBuffer<T>),
}
```

Owning the buffer directly (not wrapping ferray's Array) gives us:
- Freedom to add GPU backends as new enum variants
- Control over memory allocation strategy (future: caching allocator)
- No dependency on ferray's ndarray internals for storage layout

**Device** (`device.rs`):
```rust
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    Cpu,
    Cuda(usize),
}
```

Defined in Phase 1 even though only `Cpu` is functional. This ensures the type is baked into every API from day one.

**GradFn<T>** (`autograd/function.rs`):
```rust
pub trait GradFn<T: Element>: Send + Sync + fmt::Debug {
    fn backward(&self, grad_output: &Tensor<T>) -> Result<Vec<Option<Tensor<T>>>, FerrotorchError>;
    fn inputs(&self) -> Vec<Arc<TensorStorage<T>>>;
    fn name(&self) -> &'static str;
}
```

Returns `Vec<Option<Tensor<T>>>` — `None` for inputs that don't require gradients. The `name()` method aids debugging (e.g., "AddBackward", "MatmulBackward").

### Backward Engine (`autograd/graph.rs`)

1. Starting from the output tensor, collect all nodes with `grad_fn.is_some()` via DFS
2. Topological sort (Kahn's algorithm — iterative, no stack overflow on deep graphs)
3. Walk in reverse order, calling each node's `GradFn::backward()`
4. Accumulate gradients additively on leaf tensors (handles shared inputs like `a + a`)
5. Drop the computation graph after backward completes (matches PyTorch eager mode)

### ferray Integration Boundary

ferrotorch does **not** wrap ferray Arrays for storage. Instead, it calls ferray functions by constructing temporary Arrays from raw pointers when needed:

```
Tensor<T> → extract raw slice → construct ferray ArrayView<T, IxDyn> → call ferray op → write result into new TensorStorage
```

This happens in `ops/elementwise.rs` and `ops/linalg.rs`. The conversion is zero-copy for reads (ArrayView borrows the slice) and requires one allocation for the output.

**Linalg path**: With ferray-linalg extended to support f32 (external prerequisite), `ops/linalg.rs` calls `ferray_linalg::matmul()` uniformly for both f32 and f64. No dtype dispatch needed in ferrotorch.
- `bf16`: promotes to f32, calls ferray-linalg, demotes back (until native bf16 SIMD is available)

### Grad Functions Table (Phase 1 scope — math ops only)

| File | Operations | VJP Strategy |
|------|-----------|-------------|
| `arithmetic.rs` | add, sub, mul, div, neg, pow, sqrt, abs | Elementwise: grad flows through unchanged (add/sub), or scaled by the other operand (mul/div) |
| `reduction.rs` | sum, mean, prod | Broadcast gradient back to input shape; prod uses log-sum-exp trick for numerical stability |
| `linalg.rs` | matmul, bmm, mm, mv, dot | `d(A @ B)/dA = grad @ B^T`, `d(A @ B)/dB = A^T @ grad`; bmm applies per-batch |
| `activation.rs` | relu, sigmoid, tanh, gelu, silu, softmax, log_softmax | relu: mask where input > 0; sigmoid: `grad * s * (1-s)`; softmax: Jacobian-vector product |
| `shape.rs` | reshape, transpose, permute, expand, contiguous, cat, stack, split, squeeze, unsqueeze, flatten | Inverse shape ops: reshape grad back, transpose grad, split grad at cat boundaries |
| `indexing.rs` | gather, scatter_add, index_select, masked_fill | Sparse accumulation: scatter grad to source indices |
| `comparison.rs` | where_ | Route grad to selected branch, zero to other |

**Deferred to ferrotorch-nn (Phase 2):** conv, pool, norm, dropout, embedding, loss, attention. These implement `GradFn<T>` (defined in core) but live in the nn crate alongside their corresponding `Module` wrappers.

### Error Type (`error.rs`)

```rust
#[derive(Debug, thiserror::Error)]
pub enum FerrotorchError {
    #[error("shape mismatch: {message}")]
    ShapeMismatch { message: String },
    #[error("device mismatch: expected {expected}, got {got}")]
    DeviceMismatch { expected: Device, got: Device },
    #[error("backward called on non-scalar tensor with shape {shape:?}")]
    BackwardNonScalar { shape: Vec<usize> },
    #[error("no gradient function on non-leaf tensor")]
    NoGradFn,
    #[error("dtype mismatch: expected {expected}, got {got}")]
    DtypeMismatch { expected: String, got: String },
    #[error("index out of bounds: index {index} on axis {axis} with size {size}")]
    IndexOutOfBounds { index: usize, axis: usize, size: usize },
    #[error("invalid argument: {message}")]
    InvalidArgument { message: String },
    #[error(transparent)]
    Ferray(#[from] ferray_core::FerrayError),
}
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferray-core` | workspace | `Element` trait, `DType`, `Array`/`ArrayView` for ferray interop |
| `ferray-ufunc` | workspace | Elementwise math (exp, log, sin, cos, etc.) |
| `ferray-linalg` | workspace | f32/f64 matmul/decompositions (requires f32 extension — external prereq) |
| `ferray-random` | workspace | Random tensor creation (rand, randn) |
| `thiserror` | 2.0 | Error derive macros |
| `rayon` | 1.11 | Parallel elementwise operations |

### Test Strategy

1. **Numerical gradient checks**: For every `GradFn`, compare analytic gradient against finite-difference approximation: `(f(x+h) - f(x-h)) / 2h`. Use `h = 1e-5` for f32, `h = 1e-8` for f64.
2. **PyTorch reference tests**: For 20+ representative graphs, compute forward + backward in PyTorch, serialize the inputs/outputs/gradients as `.npy` files (via ferray-io), and assert ferrotorch matches within tolerance.
3. **Edge cases**: Empty tensors, scalar (0-dim) tensors, tensors with size-0 dimensions, very high rank (8+), contiguous vs non-contiguous inputs.
4. **Thread safety**: Build graph on thread A, send output to thread B, backward on thread B. Verify gradients are correct.
5. **Memory**: Verify gradient checkpointing reduces peak allocation on deep chains.

### Out of Scope

- GPU execution — `Device::Cuda` is defined but only `Device::Cpu` is functional in Phase 1
- Neural network modules (Linear, Conv2d, etc.) — that is Phase 2 (ferrotorch-nn)
- Optimizers (SGD, Adam) — that is Phase 3 (ferrotorch-optim)
- Model serialization — that is Phase 3 (ferrotorch-serialize)
- Data loading — that is Phase 4 (ferrotorch-data)
- Python bindings — that is a late phase (ferrotorch-python)
- Higher-order gradients (grad of grad) — not needed for standard training
- Lazy/compiled graph mode — that is Phase 8 (ferrotorch-jit)
- Mixed-precision training (autocast) — future feature after bf16 is stable
- Custom autograd function registration by users — defer to Phase 2+

### resolved questions

### Q1: f32 linalg — faer direct or extend ferray-linalg?
**Decision**: Extend ferray-linalg to support f32 before starting Phase 1 (Option B).

This is a prerequisite alongside the bf16 ferray-core extension. ferray-linalg's ~30 public functions will be made generic over `T` where faer supports it (f32, f64), using the same faer bridge pattern. Once complete, ferrotorch calls `ferray_linalg::matmul()` for both f32 and f64 with no dtype dispatch in `ops/linalg.rs`. This eliminates the dual-path maintenance burden and benefits the ferray ecosystem.

**External prerequisites before Phase 1 can start:**
1. ferray-core: add bf16 to `Element` trait and `DType` enum
2. ferray-linalg: make all public functions generic over f32/f64

### Q2: Interior mutability strategy for gradients
**Decision**: `Mutex<Option<Tensor<T>>>` on the tensor (Option B).

This satisfies REQ-10 (Send + Sync) without refactoring later. The lock overhead is negligible compared to the cost of the math operations in backward. `Mutex::lock()` returns `Result`, fitting the zero-panic guarantee — use `.map_err()` to convert `PoisonError` into `FerrotorchError`. Single-threaded backward simply never contends. Multi-threaded backward (parallel gradient accumulation for independent subgraphs) becomes possible in the future without changing the core type.

### Q3: Scope of grad_fns in Phase 1
**Decision**: Math ops in core, layer ops in nn (Option B).

ferrotorch-core keeps: arithmetic, reduction, linalg, activation, shape, indexing, comparison (~7 grad_fn files, ~25 operations). These are pure math operations whose VJPs are well-defined and don't require nn-specific concepts (kernel size, dropout probability, vocabulary size).

ferrotorch-nn (Phase 2) gets: conv, pool, norm, dropout, embedding, loss, attention (~8 grad_fn files). These are neural network primitives. nn defines structs implementing `GradFn<T>` (the trait from core) — no registration mechanism needed. Core's backward engine calls `grad_fn.backward()` via trait dispatch without knowing the concrete type. This is just standard Rust dynamic dispatch on `Arc<dyn GradFn<T>>`.

