---
title: "Phase 7 — Distributed Training (ferrotorch-distributed)"
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-15
updated: 2026-03-15
---


## Design Specification

### Summary

A multi-GPU and multi-node distributed training crate for ferrotorch, providing communication primitives, process group management, and high-level parallelism strategies. Wraps ferrotorch-nn modules with automatic gradient synchronization (DDP), parameter sharding (FSDP), pipeline parallelism (GPipe-style microbatching), and tensor parallelism (Megatron-style column/row splitting). Communication is abstracted behind a `Backend` trait with NCCL (GPU), Gloo (CPU), and MPI implementations, allowing users to scale training from a single machine with multiple GPUs to clusters of hundreds of nodes without changing model code.

### Requirements

- REQ-1: A `Backend` trait must abstract inter-process communication, with concrete implementations for NCCL (GPU-to-GPU via NVLink/PCIe/InfiniBand), Gloo (CPU via TCP/shared memory), and MPI (via system MPI library). Each backend must support the full set of collective operations defined in REQ-3. Backend selection must be explicit at initialization time, not auto-detected.
- REQ-2: A `ProcessGroup` must manage a set of ranks participating in collective communication. The default group spans all ranks. Users must be able to create sub-groups (e.g., for tensor-parallel communication within a node vs. data-parallel communication across nodes). Each rank has a unique integer ID within its group, and the group tracks the total world size. Process groups must be `Send + Sync` and reference-counted (`Arc`).
- REQ-3: The following collective operations must be implemented for all backends: `allreduce` (sum, mean, min, max), `broadcast` (root to all), `allgather` (concatenate from all ranks), `reduce_scatter` (reduce then scatter), `barrier` (synchronization fence), and point-to-point `send`/`recv`. All collectives must operate on `Tensor<T>` directly and return `Result<(), FerrotorchError>`. Async variants must return a `Work` handle that can be `wait()`-ed.
- REQ-4: `DistributedDataParallel<M>` must wrap any `Module<T>` and synchronize gradients across ranks after each backward pass via allreduce on the default process group. It must bucket small parameter gradients into larger buffers before allreduce to amortize communication latency (matching PyTorch's gradient bucketing strategy). Forward calls must broadcast parameters from rank 0 on the first iteration to ensure all ranks start with identical weights.
- REQ-5: `FullyShardedDataParallel<M>` must shard model parameters, gradients, and optimizer states across ranks in the process group, materializing full parameters only during forward and backward via allgather, then re-sharding via reduce_scatter after backward. It must support configurable sharding strategies: full shard (ZeRO-3 equivalent, parameters + gradients + optimizer state), grad-only shard (ZeRO-2 equivalent, gradients + optimizer state), and no shard (DDP fallback). Mixed-precision with f32 parameter copies for the optimizer and bf16/f16 for computation must be supported.
- REQ-6: Pipeline parallelism must partition a `Sequential` module across devices (one stage per device group) and execute forward/backward using GPipe-style microbatching to fill the pipeline. The number of microbatches must be configurable. The implementation must handle the 1F1B (one-forward-one-backward) schedule to reduce peak memory compared to naive GPipe fill-drain.
- REQ-7: Tensor parallelism must provide `ColumnParallelLinear` and `RowParallelLinear` modules that split weight matrices across ranks along the output and input dimensions respectively. Column-parallel gathers outputs via allgather after forward; row-parallel reduces inputs via reduce_scatter. These modules must be drop-in replacements for `Linear` in transformer architectures.
- REQ-8: All public functions must return `Result<T, FerrotorchError>`. Failures in communication (rank timeout, NCCL error, connection refused) must produce descriptive errors including the failing rank, operation, and backend. No panics on communication failure.
- REQ-9: An `init_process_group` entry point must initialize the distributed runtime from environment variables (`RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT`) or explicit configuration, matching PyTorch's `torch.distributed.init_process_group` interface. Cleanup must happen via `Drop` on the process group, releasing backend resources.

### Acceptance Criteria

- [ ] AC-1: `init_process_group(backend, rank, world_size, master_addr, master_port)` initializes a process group. All ranks can execute `barrier()` and return successfully. Destroying the group (via `Drop`) releases all backend resources without leaks.
- [ ] AC-2: `allreduce` on a `Tensor<f32>` across 4 ranks produces the correct sum (verified numerically). `broadcast` from rank 0 results in identical tensors on all ranks. `allgather` concatenates rank-local tensors into the correct global tensor. `reduce_scatter` produces correct sharded results. All collectives tested for both NCCL and Gloo backends.
- [ ] AC-3: `DistributedDataParallel` wrapping a 3-layer MLP produces identical gradients on all ranks after backward (within f32 tolerance), and parameter updates are identical after optimizer step. Training loss curves match single-GPU training within statistical noise over 100 iterations.
- [ ] AC-4: `DistributedDataParallel` gradient bucketing reduces the number of allreduce calls — a model with 50 small parameters (each under 1KB) issues fewer than 10 allreduce calls per backward pass, not 50.
- [ ] AC-5: `FullyShardedDataParallel` with full sharding reduces per-rank memory usage by at least 60% compared to DDP for a 100M-parameter model across 4 ranks, while producing numerically identical training results.
- [ ] AC-6: Pipeline parallelism with 4 stages and 8 microbatches trains a 4-segment Sequential model across 4 devices. The 1F1B schedule achieves at least 70% pipeline utilization (measured as compute time / wall time per rank) and produces correct gradients verified against single-device training on the same model.
- [ ] AC-7: `ColumnParallelLinear` and `RowParallelLinear` produce numerically identical outputs to a standard `Linear` layer when gathered across ranks. A transformer block using tensor-parallel attention and MLP produces the same logits as a single-rank transformer block (within f32 tolerance).
- [ ] AC-8: All public functions return `Result`. Attempting `allreduce` on a destroyed process group returns `Err(FerrotorchError::DistributedError { .. })`. Timeout on a hung rank returns an error with the rank ID and operation name within the configured timeout window.
- [ ] AC-9: `cargo test -p ferrotorch-distributed` passes with 0 failures. Multi-rank tests use `std::process::Command` to spawn child processes simulating ranks (no external launcher required for testing). Minimum 80 tests covering all collectives, DDP, FSDP, pipeline, and tensor parallelism.

### Architecture

### Crate Layout

```
ferrotorch-distributed/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports, init_process_group
│   ├── backend/
│   │   ├── mod.rs                # Backend trait definition
│   │   ├── nccl.rs               # NCCL backend (GPU, via nccl-rs bindings)
│   │   ├── gloo.rs               # Gloo backend (CPU, TCP + shared memory)
│   │   └── mpi.rs                # MPI backend (via rsmpi bindings)
│   ├── process_group.rs          # ProcessGroup, sub-group creation, rank/world_size
│   ├── collective.rs             # allreduce, broadcast, allgather, reduce_scatter, barrier, send, recv
│   ├── ddp.rs                    # DistributedDataParallel<M> wrapper
│   ├── fsdp.rs                   # FullyShardedDataParallel<M> wrapper
│   ├── pipeline.rs               # PipelineParallel, GPipe schedule, 1F1B schedule
│   └── tensor_parallel.rs        # ColumnParallelLinear, RowParallelLinear
└── tests/
    ├── test_process_group.rs     # Init, sub-groups, cleanup
    ├── test_collectives.rs       # allreduce, broadcast, allgather, reduce_scatter, barrier
    ├── test_ddp.rs               # Gradient sync, bucketing, convergence
    ├── test_fsdp.rs              # Sharding, memory, numerical equivalence
    ├── test_pipeline.rs          # Microbatch scheduling, utilization, correctness
    └── test_tensor_parallel.rs   # Column/row parallel, transformer integration
```

### Backend Trait (`backend/mod.rs`)

```rust
/// Reduction operations for collective communication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    Sum,
    Mean,
    Min,
    Max,
}

/// Handle for an in-flight async collective operation.
pub trait Work: Send + Sync {
    fn wait(&self) -> Result<(), FerrotorchError>;
    fn is_completed(&self) -> bool;
}

/// Communication backend abstraction.
pub trait Backend: Send + Sync {
    fn allreduce(&self, tensor: &mut Tensor<f32>, op: ReduceOp, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn allreduce_async(&self, tensor: &mut Tensor<f32>, op: ReduceOp, group: &ProcessGroup)
        -> Result<Box<dyn Work>, FerrotorchError>;

    fn broadcast(&self, tensor: &mut Tensor<f32>, src: usize, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn allgather(&self, output: &mut Tensor<f32>, input: &Tensor<f32>, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn reduce_scatter(&self, output: &mut Tensor<f32>, input: &Tensor<f32>, op: ReduceOp, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn barrier(&self, group: &ProcessGroup) -> Result<(), FerrotorchError>;

    fn send(&self, tensor: &Tensor<f32>, dst: usize, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn recv(&self, tensor: &mut Tensor<f32>, src: usize, group: &ProcessGroup)
        -> Result<(), FerrotorchError>;

    fn backend_name(&self) -> &'static str;
}
```

The trait is defined over `Tensor<f32>` in the signatures above for clarity, but the actual implementation is generic over `T: Element` using a sealed helper trait `DistElement` that restricts to types the backend can transmit (f32, f64, bf16, f16, i32, i64).

### NCCL Backend (`backend/nccl.rs`)

Wraps the NCCL library via Rust bindings (either `cudarc`'s NCCL support or a dedicated `nccl-rs` crate). Each `NcclBackend` instance holds a `ncclComm_t` communicator per process group. Tensors must reside on `Device::Cuda` — passing a CPU tensor returns `Err(FerrotorchError::DeviceMismatch)`. All operations are launched on the CUDA stream associated with the tensor's device, enabling overlap with compute kernels.

### Gloo Backend (`backend/gloo.rs`)

A pure-Rust TCP-based backend for CPU tensors. Uses `std::net::TcpStream` connections between ranks established during `init_process_group`. Implements allreduce via ring-reduce (each rank sends/receives from neighbors in a logical ring, requiring `2 * (world_size - 1)` communication steps for reduce-scatter + allgather phases). Suitable for development, testing, and CPU-only training.

Shared-memory transport is used when ranks detect they are on the same host (comparing `MASTER_ADDR` against local interfaces), avoiding TCP overhead for intra-node communication.

### MPI Backend (`backend/mpi.rs`)

Wraps the system MPI library via `rsmpi`. Delegates collectives to `MPI_Allreduce`, `MPI_Bcast`, `MPI_Allgather`, etc. Intended for HPC clusters where MPI is the standard communication fabric. The MPI environment must be initialized before `init_process_group` (via `mpi::initialize()` from rsmpi). Feature-gated behind `features = ["mpi"]` since rsmpi requires a system MPI installation.

### Process Group (`process_group.rs`)

```rust
pub struct ProcessGroup {
    rank: usize,
    world_size: usize,
    ranks: Vec<usize>,              // Global rank IDs in this group
    backend: Arc<dyn Backend>,
    sub_groups: Vec<Arc<ProcessGroup>>,
}

impl ProcessGroup {
    /// Create a sub-group from a subset of ranks.
    pub fn new_group(&self, ranks: &[usize]) -> Result<Arc<ProcessGroup>, FerrotorchError>;

    /// This rank's ID within the group.
    pub fn rank(&self) -> usize;

    /// Total number of ranks in the group.
    pub fn world_size(&self) -> usize;
}
```

Sub-groups are created by calling `new_group` with a subset of global ranks. The backend creates a new communicator scoped to those ranks (e.g., `ncclCommSplit` for NCCL, `MPI_Comm_create_group` for MPI). Sub-groups enable hybrid parallelism: a tensor-parallel group within each node (e.g., ranks [0,1,2,3]) and a data-parallel group across nodes (e.g., ranks [0,4,8,12]).

### Collective Operations (`collective.rs`)

Module-level functions that dispatch to the backend attached to the given process group:

```rust
pub fn allreduce(tensor: &mut Tensor<f32>, op: ReduceOp, group: &ProcessGroup)
    -> Result<(), FerrotorchError>
{
    group.backend.allreduce(tensor, op, group)
}
```

Async variants return a `Box<dyn Work>` handle. The caller must `wait()` before reading the tensor. This enables overlapping communication with computation — DDP issues async allreduce on one gradient bucket while backward is still computing gradients for the next bucket.

### DistributedDataParallel (`ddp.rs`)

```rust
pub struct DistributedDataParallel<M: Module<f32>> {
    module: M,
    process_group: Arc<ProcessGroup>,
    buckets: Vec<GradientBucket>,
    bucket_size_mb: f64,
}

struct GradientBucket {
    params: Vec<usize>,             // Indices into module.parameters()
    buffer: Tensor<f32>,            // Flat buffer for allreduce
    work: Option<Box<dyn Work>>,    // In-flight async allreduce handle
}
```

**Initialization**: Parameters are assigned to buckets in reverse declaration order (matching PyTorch's strategy — parameters used later in forward are communicated first in backward). Bucket boundaries are placed when cumulative parameter size exceeds `bucket_size_mb` (default: 25 MB). On the first forward call, rank 0's parameters are broadcast to all ranks to ensure identical starting weights.

**Backward hook**: After each parameter's gradient is computed during `backward()`, the DDP wrapper checks whether the gradient's bucket is full. If so, it flattens all gradients in the bucket into the contiguous buffer and issues an async allreduce. This overlaps communication of early buckets with gradient computation for later layers.

**Synchronization**: Before `optimizer.step()`, DDP calls `wait()` on all in-flight allreduce handles, then copies the averaged gradients from bucket buffers back to each parameter's `.grad`. The optimizer then steps on the synchronized gradients.

### FullyShardedDataParallel (`fsdp.rs`)

```rust
pub enum ShardingStrategy {
    FullShard,      // ZeRO-3: shard params + grads + optimizer state
    GradOnlyShard,  // ZeRO-2: shard grads + optimizer state, replicate params
    NoShard,        // DDP-equivalent: replicate everything
}

pub struct FullyShardedDataParallel<M: Module<f32>> {
    module: M,
    process_group: Arc<ProcessGroup>,
    strategy: ShardingStrategy,
    sharded_params: Vec<ShardedParameter>,
    mixed_precision: Option<MixedPrecisionConfig>,
}

struct ShardedParameter {
    local_shard: Tensor<f32>,       // This rank's slice of the parameter
    full_param: Option<Tensor<f32>>,// Materialized during forward/backward, then dropped
    shard_offsets: (usize, usize),  // Start/end indices in the flat parameter
}

pub struct MixedPrecisionConfig {
    param_dtype: DType,             // Storage dtype (f32)
    compute_dtype: DType,           // Forward/backward dtype (bf16 or f16)
    reduce_dtype: DType,            // Communication dtype (f32 for accuracy)
}
```

**Forward**: Before the wrapped module's forward, FSDP calls `allgather` to materialize full parameters from shards across ranks. The full parameter tensors are attached to the module for the duration of forward.

**Backward**: Full parameters are re-materialized via `allgather` (if they were freed after forward). After gradient computation, `reduce_scatter` distributes gradient shards — each rank keeps only its shard of the gradient. Full parameter tensors are freed immediately after use.

**Memory lifecycle**: In `FullShard` mode, each rank holds only `1/world_size` of the parameters at rest. Peak memory during forward/backward includes one full parameter set at a time (the layer currently executing). FSDP processes layers one at a time, prefetching the next layer's allgather while the current layer executes.

### Pipeline Parallelism (`pipeline.rs`)

```rust
pub struct PipelineParallel<T: Element = f32> {
    stages: Vec<PipelineStage<T>>,
    num_microbatches: usize,
    schedule: PipelineSchedule,
    process_group: Arc<ProcessGroup>,
}

pub enum PipelineSchedule {
    /// Fill all microbatches forward, then drain all backward. Simple but high memory.
    FillDrain,
    /// Interleave forward and backward passes. Lower memory, better utilization.
    OneFOneBSchedule,
}

struct PipelineStage<T: Element> {
    module: Box<dyn Module<T>>,
    device: Device,
    rank: usize,
}
```

**FillDrain (GPipe)**: All microbatches execute forward through all stages, then all microbatches execute backward. Peak memory is proportional to `num_microbatches` since all intermediate activations are held simultaneously.

**1F1B schedule**: After the pipeline fills (first `num_stages` microbatches in forward), each rank alternates one forward and one backward pass. This limits peak activation memory to `num_stages` microbatches rather than `num_microbatches`, at the cost of slightly more complex scheduling logic.

**Inter-stage communication**: Stage `i` sends its output tensor to stage `i+1` via point-to-point `send`/`recv`. During backward, stage `i+1` sends the gradient of its input back to stage `i`. Each stage runs on its assigned device, so send/recv crosses device boundaries (GPU-to-GPU via NCCL or CPU-to-CPU via Gloo).

### Tensor Parallelism (`tensor_parallel.rs`)

```rust
pub struct ColumnParallelLinear<T: Element = f32> {
    weight_shard: Parameter<T>,     // Shape: [out_features / world_size, in_features]
    bias_shard: Option<Parameter<T>>,
    process_group: Arc<ProcessGroup>,
    gather_output: bool,
}

pub struct RowParallelLinear<T: Element = f32> {
    weight_shard: Parameter<T>,     // Shape: [out_features, in_features / world_size]
    bias: Option<Parameter<T>>,     // Only rank 0 holds bias (added after reduce)
    process_group: Arc<ProcessGroup>,
    input_is_parallel: bool,
}
```

**ColumnParallelLinear**: Splits the weight along the output dimension. Each rank computes `Y_local = X @ W_local^T`. If `gather_output` is true, an allgather concatenates `Y_local` across ranks to produce the full output. In transformer architectures, `gather_output` is false for the first linear in a paired column+row sequence — the partial outputs feed directly into `RowParallelLinear`.

**RowParallelLinear**: Splits the weight along the input dimension. Each rank computes `Y_local = X_local @ W_local^T` where `X_local` is the rank-local slice of the input. A reduce_scatter (or allreduce) combines partial sums across ranks. When `input_is_parallel` is true (typical in a transformer), the input is already split across ranks from a preceding `ColumnParallelLinear`.

**Transformer pattern**: A typical Megatron-style transformer block uses:
1. `ColumnParallelLinear` for Q/K/V projections (split heads across ranks)
2. Local attention computation on each rank's head subset
3. `RowParallelLinear` for the output projection (reduce across ranks)
4. `ColumnParallelLinear` for MLP up-projection
5. `RowParallelLinear` for MLP down-projection

This requires only 2 allreduce operations per transformer block (one after attention output projection, one after MLP down-projection) regardless of the number of ranks.

### Initialization (`lib.rs`)

```rust
pub fn init_process_group(
    backend: BackendKind,
    rank: usize,
    world_size: usize,
    master_addr: &str,
    master_port: u16,
) -> Result<Arc<ProcessGroup>, FerrotorchError>;

pub fn init_process_group_from_env(
    backend: BackendKind,
) -> Result<Arc<ProcessGroup>, FerrotorchError>;

#[derive(Clone, Copy, Debug)]
pub enum BackendKind {
    Nccl,
    Gloo,
    Mpi,
}
```

`init_process_group_from_env` reads `RANK`, `WORLD_SIZE`, `MASTER_ADDR`, `MASTER_PORT` from environment variables, matching PyTorch's `torchrun` conventions. This allows ferrotorch distributed programs to be launched with the same tools (`torchrun`, `mpirun`, SLURM `srun`).

### Error Handling

The `FerrotorchError` enum (defined in ferrotorch-core) is extended with distributed variants:

```rust
#[error("distributed error on rank {rank}, operation {operation}: {message}")]
DistributedError { rank: usize, operation: String, message: String },

#[error("rank {rank} timed out after {timeout_secs}s during {operation}")]
DistributedTimeout { rank: usize, timeout_secs: u64, operation: String },

#[error("backend {backend} not available: {reason}")]
BackendUnavailable { backend: String, reason: String },
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `Device`, `FerrotorchError` |
| `ferrotorch-nn` | workspace | `Module<T>`, `Parameter<T>`, `Sequential` |
| `ferrotorch-gpu` | workspace | CUDA device, NCCL communicator access |
| `cudarc` | latest | NCCL bindings for GPU communication |
| `rsmpi` | latest | MPI bindings (feature-gated behind `mpi`) |
| `crossbeam-channel` | 0.5 | Lock-free channels for async work notification |

### Test Strategy

Multi-rank tests spawn child processes via `std::process::Command`, each running with different `RANK` environment variables. The test binary is re-invoked with a special flag (`--distributed-worker`) to enter worker mode. This avoids requiring `mpirun` or `torchrun` in the test harness.

```rust
#[test]
fn test_allreduce_4_ranks() {
    let world_size = 4;
    let handles: Vec<_> = (0..world_size).map(|rank| {
        Command::new(env::current_exe().unwrap())
            .env("RANK", rank.to_string())
            .env("WORLD_SIZE", world_size.to_string())
            .env("MASTER_ADDR", "127.0.0.1")
            .env("MASTER_PORT", "29500")
            .arg("--distributed-worker")
            .arg("allreduce")
            .spawn()
            .unwrap()
    }).collect();
    for mut h in handles { assert!(h.wait().unwrap().success()); }
}
```

NCCL tests require multiple GPUs and are gated behind `#[cfg(feature = "nccl-tests")]`. CI runs Gloo-backend tests by default.

### Out of Scope

- Model parallelism strategies beyond pipeline and tensor parallelism (e.g., expert parallelism / MoE routing) — future work
- Elastic training (adding/removing ranks at runtime) — requires a separate coordination service
- Gradient compression (quantized allreduce, sparsification) — optimization pass after correctness is established
- Heterogeneous device training (mixing GPU and CPU ranks in the same group) — each process group is tied to one backend
- Automatic parallelism planning (deciding how to split a model across devices) — users specify the strategy explicitly
- RDMA/InfiniBand transport for Gloo — initial implementation uses TCP and shared memory only
- Cross-framework distributed communication (e.g., interop with PyTorch distributed processes) — not a goal

