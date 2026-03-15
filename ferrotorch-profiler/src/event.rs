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
}
