/// Which device an operation was timed on.
///
/// CPU events use `std::time::Instant` (wall-clock). CUDA events use
/// `cuEventElapsedTime` for accurate GPU-side timing that accounts for
/// asynchronous kernel execution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DeviceType {
    /// Timed with CPU wall clock (`std::time::Instant`).
    Cpu,
    /// Timed with CUDA events (`cuEventRecord` / `cuEventElapsedTime`).
    /// The `u64` payload holds the duration in microseconds, resolved
    /// lazily when the profiling report is generated.
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

impl Default for DeviceType {
    fn default() -> Self {
        DeviceType::Cpu
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
    /// Which device this event was timed on.
    pub device_type: DeviceType,
}
