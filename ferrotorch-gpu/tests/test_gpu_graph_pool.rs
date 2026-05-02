//! End-to-end CUDA graph + capture pool integration tests.
//!
//! Verifies that a real CUDA graph can be captured with buffers
//! registered in a `CapturePool`, that the resulting `CapturedGraph`
//! holds the pool, and that replays read the recorded device
//! pointers without crashing. CL-278.

#![cfg(feature = "cuda")]

use std::sync::{Arc, Mutex, MutexGuard};

use ferrotorch_gpu::buffer::CudaBuffer;
use ferrotorch_gpu::device::GpuDevice;
use ferrotorch_gpu::graph::{CapturePool, begin_capture_with_pool, end_capture_with_pool};
use ferrotorch_gpu::kernels::gpu_add_into;
use ferrotorch_gpu::transfer::{alloc_zeros_f32, cpu_to_gpu};

/// Serialise CUDA stream-capture sections across this test binary's
/// thread pool. cargo runs tests within one binary in parallel, and
/// CUDA's stream-capture state machine has process-wide invariants
/// that can be violated by concurrent capture work on different
/// threads/streams (manifests as `CUDA_ERROR_STREAM_CAPTURE_INVALIDATED`
/// in `end_capture`). Holding this mutex across `begin_capture` …
/// `end_capture` makes the tests deterministic. (#602)
fn capture_lock() -> MutexGuard<'static, ()> {
    static CAPTURE_MUTEX: Mutex<()> = Mutex::new(());
    // Recover from a poisoned lock — a panic inside another test's
    // capture section shouldn't poison the test that follows.
    CAPTURE_MUTEX
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn dev() -> GpuDevice {
    GpuDevice::new(0).expect("CUDA device 0 must be available for these tests")
}

#[test]
fn captured_graph_holds_pool_buffers_alive() {
    let _guard = capture_lock();
    // Build a tiny graph that adds two pre-allocated buffers into a
    // third. Register all three buffers with the pool so the
    // resulting CapturedGraph keeps them alive across replays.
    let device = dev();

    // Pre-allocate buffers BEFORE capture (CUDA graph requirement).
    let a_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
    let b_data: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
    let mut out: CudaBuffer<f32> = alloc_zeros_f32(4, &device).expect("alloc out");
    let a = cpu_to_gpu(&a_data, &device).expect("upload a");
    let b = cpu_to_gpu(&b_data, &device).expect("upload b");

    let pool = Arc::new(CapturePool::new());
    // We need raw pointers from the buffers during capture, so we
    // wrap each in an Arc so we can both register it with the pool
    // (for lifetime extension) and use it for the kernel launch.
    // Easiest path: keep the originals on the stack until after
    // capture, then donate them.
    let stream = device.context().new_stream().expect("non-blocking stream");

    begin_capture_with_pool(&pool, &stream).expect("begin capture");
    gpu_add_into(&a, &b, &mut out, &device).expect("add_into during capture");
    let graph = end_capture_with_pool(&stream, Arc::clone(&pool)).expect("end_capture_with_pool");

    // Donate the buffers to the pool now so they outlive the
    // local variables we're about to drop.
    pool.record_buffer(a);
    pool.record_buffer(b);
    pool.record_buffer(out);
    assert_eq!(pool.buffer_count(), 3);
    assert!(graph.has_pool());
    assert_eq!(graph.pool_buffer_count(), 3);

    // Replay the graph — the recorded pointers must still be valid
    // because the pool kept them alive.
    graph.launch().expect("replay 1");
    graph.launch().expect("replay 2");
    graph.launch().expect("replay 3");
}

#[test]
fn dropping_graph_releases_pool_buffers() {
    let _guard = capture_lock();
    // Use a sentinel Arc to detect when the buffer is actually
    // dropped after the graph is destroyed.
    let device = dev();
    let _stream = device.context().new_stream().expect("stream");

    let sentinel = Arc::new(42u8);
    let pool = Arc::new(CapturePool::new());
    pool.record_buffer(Arc::clone(&sentinel));
    assert_eq!(Arc::strong_count(&sentinel), 2);

    // Build a no-op graph that holds the pool.
    let stream = device.context().new_stream().expect("non-blocking stream");
    begin_capture_with_pool(&pool, &stream).expect("begin");
    let graph = end_capture_with_pool(&stream, Arc::clone(&pool)).expect("end");
    assert!(graph.has_pool());

    // Strong refs: sentinel (1) + pool (1) = 2 still.
    assert_eq!(Arc::strong_count(&sentinel), 2);

    // Drop the graph. The graph holds an Arc<CapturePool>, which
    // holds the sentinel. Dropping the graph drops *its* Arc but
    // we still hold our local `pool` Arc, so the pool itself is
    // still alive.
    drop(graph);
    assert_eq!(Arc::strong_count(&sentinel), 2);

    // Now drop the local pool reference. Last Arc<CapturePool>
    // gone → pool dropped → sentinel Arc dropped.
    drop(pool);
    assert_eq!(Arc::strong_count(&sentinel), 1);
}

#[test]
fn pool_seal_blocks_begin_capture_with_pool() {
    let device = dev();
    let stream = device.context().new_stream().expect("non-blocking stream");
    let pool = Arc::new(CapturePool::new());
    pool.seal();
    let result = begin_capture_with_pool(&pool, &stream);
    assert!(
        result.is_err(),
        "sealed pool must reject begin_capture_with_pool"
    );
}

#[test]
fn captured_graph_without_pool_has_zero_buffer_count() {
    let _guard = capture_lock();
    // The legacy end_capture path produces a graph with no pool.
    let device = dev();
    let stream = device.context().new_stream().expect("non-blocking stream");
    use ferrotorch_gpu::graph::{begin_capture, end_capture};

    begin_capture(&stream).expect("begin");
    let graph = end_capture(&stream).expect("end");
    assert!(!graph.has_pool());
    assert_eq!(graph.pool_buffer_count(), 0);
}

#[test]
fn pool_buffer_count_grows_with_record_buffer() {
    let pool = Arc::new(CapturePool::new());
    assert_eq!(pool.buffer_count(), 0);
    pool.record_buffer(vec![0u8; 16]);
    pool.record_buffer(vec![0u8; 32]);
    pool.record_buffer(vec![0u8; 64]);
    assert_eq!(pool.buffer_count(), 3);
}
