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
    /// OS thread id that recorded this event.
    pub thread_id: u64,
    /// The device type that executed this operation.
    pub device_type: DeviceType,
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
