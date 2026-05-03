/// The device type that executed an operation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Operation executed on the CPU.
    #[default]
    Cpu,
    /// Operation executed on a CUDA GPU (or timed by CPU as a fallback
    /// when GPU event timing is unavailable).
    Cuda,
}

impl std::fmt::Display for DeviceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DeviceType::Cpu => write!(f, "CPU"),
            DeviceType::Cuda => write!(f, "CUDA"),
        }
    }
}

/// Category for memory allocation/free events. Mirrors the broad
/// categories used by `PyTorch`'s memory profiler so users can tell where
/// VRAM is going. CL-333.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum MemoryCategory {
    /// Forward-pass activations and intermediate tensors. Typically
    /// the dominant category during training, especially without
    /// gradient checkpointing.
    Activations,
    /// Learnable model parameters (weight, bias). Usually fixed for
    /// the lifetime of training.
    Parameters,
    /// Optimizer state (Adam first/second moments, SGD momentum
    /// buffers, etc.). Roughly 2x the parameter footprint for Adam-style
    /// optimizers.
    OptimizerState,
    /// Gradient buffers held on parameters during the backward pass.
    Gradients,
    /// Any allocation that doesn't fit the above buckets.
    #[default]
    Other,
}

impl std::fmt::Display for MemoryCategory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MemoryCategory::Activations => write!(f, "activations"),
            MemoryCategory::Parameters => write!(f, "parameters"),
            MemoryCategory::OptimizerState => write!(f, "optimizer_state"),
            MemoryCategory::Gradients => write!(f, "gradients"),
            MemoryCategory::Other => write!(f, "other"),
        }
    }
}

/// A single profiling event recorded during execution.
#[derive(Debug, Clone)]
pub struct ProfileEvent {
    /// Operation name (e.g. `"matmul"`, `"conv2d"`).
    pub name: String,
    /// Category tag (e.g. `"tensor_op"`, `"cuda_kernel"`, `"memory"`).
    pub category: String,
    /// Microseconds elapsed since the profiler was started.
    pub start_us: u64,
    /// Duration of the operation in microseconds.
    pub duration_us: u64,
    /// Shapes of the input tensors (empty when `record_shapes` is off).
    pub input_shapes: Vec<Vec<usize>>,
    /// Bytes allocated (positive) or freed (negative).  `None` when
    /// `record_memory` is off or the event is not a memory event.
    pub memory_bytes: Option<i64>,
    /// Memory category for memory events (Activations, Parameters,
    /// `OptimizerState`, Gradients, Other). `None` for non-memory events.
    /// CL-333.
    pub memory_category: Option<MemoryCategory>,
    /// OS thread id that recorded this event.
    pub thread_id: u64,
    /// The device type that executed this operation.
    pub device_type: DeviceType,
    /// Estimated FLOPS for the operation, computed from name + input
    /// shapes. `None` when shapes were not recorded or when the op
    /// has no FLOPS estimator. CL-333.
    pub flops: Option<u64>,
    /// Captured stack trace (one frame per `\n`-separated line).
    /// `None` when `with_stack` is off in the profiler config. Filled
    /// from `std::backtrace::Backtrace::capture()` which is a no-op
    /// unless the `RUST_BACKTRACE` env var is set, so the cost is
    /// negligible when stack capture is not requested. CL-333.
    pub stack_trace: Option<String>,
}

/// A pair of GPU timing values (start and end) in microseconds.
///
/// Returned by [`Profiler::push_gpu_event`] when the event is
/// successfully recorded. The caller can use the index to correlate
/// GPU events with CPU-side bookkeeping.
#[derive(Debug, Clone, Copy)]
pub struct GpuTimingPair {
    /// Start time in microseconds relative to the profiler's epoch.
    pub start_us: u64,
    /// End time in microseconds relative to the profiler's epoch.
    pub end_us: u64,
}
