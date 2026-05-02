//! Real-GPU integration tests for CUDA event-based kernel timing.
//!
//! Verifies the full pipeline:
//!   1. CudaKernelScope::new records a start CUDA event on a real
//!      stream.
//!   2. A CUDA kernel runs on that stream.
//!   3. CudaKernelScope::stop records the end event and queues the
//!      pair on the profiler.
//!   4. Profiler::flush_cuda_kernels synchronizes the events and
//!      converts them to ProfileEvents with GPU-measured duration.
//!
//! These tests require a real CUDA device and only run with the
//! `cuda` feature enabled. CL-380.

#![cfg(feature = "cuda")]

use std::sync::Arc;

use cudarc::driver::CudaContext;
use ferrotorch_profiler::{CudaKernelScope, ProfileConfig, with_profiler};

fn ctx() -> Option<Arc<CudaContext>> {
    CudaContext::new(0).ok()
}

#[test]
fn cuda_kernel_scope_basic_lifecycle() {
    let Some(c) = ctx() else { return };
    let stream = c.default_stream();

    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        let scope =
            CudaKernelScope::new(&c, &stream, "test_kernel", "cuda_kernel").expect("create scope");
        // No actual kernel — just measure the empty interval. The
        // GPU duration should be near zero but the event still gets
        // recorded.
        scope.stop(p).expect("stop scope");
        // Pending count should reflect the queued scope.
        assert_eq!(p.pending_cuda_count(), 1);
        p.flush_cuda_kernels();
        assert_eq!(p.pending_cuda_count(), 0);
    });

    // After flush, the report should contain exactly one CUDA event
    // with the right name and category.
    let events = report.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].name, "test_kernel");
    assert_eq!(events[0].category, "cuda_kernel");
    use ferrotorch_profiler::ProfileEvent;
    let _: &ProfileEvent = &events[0]; // type sanity check
}

#[test]
fn cuda_kernel_scope_measures_real_gpu_time() {
    use ferrotorch_profiler::ProfileEvent;
    let Some(c) = ctx() else { return };
    let stream = c.default_stream();

    // Run a meaningful amount of GPU work so the elapsed time is
    // non-zero. cudarc's `synchronize()` flushes any pending kernels;
    // launching nothing means the events fire immediately and we get
    // a near-zero duration. To get a measurable interval, we use the
    // strided_copy kernel from ferrotorch-gpu (CL-496) which we know
    // takes a few microseconds.
    use ferrotorch_gpu::device::GpuDevice;
    use ferrotorch_gpu::kernels::gpu_strided_copy;
    use ferrotorch_gpu::transfer::cpu_to_gpu;

    let dev = GpuDevice::new(0).expect("CUDA device 0");
    let n = 1024 * 1024; // 1M elements — guarantees non-trivial elapsed time
    let host: Vec<f32> = (0..n).map(|i| i as f32).collect();
    let input = cpu_to_gpu(&host, &dev).expect("upload");

    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        let scope =
            CudaKernelScope::new(&c, &stream, "strided_copy", "cuda_kernel").expect("scope");

        // Permute via strided copy: shape [1024, 1024] -> [1024, 1024]
        // with strides [1, 1024] (transpose). This launches a real
        // PTX kernel on the default stream.
        let _out =
            gpu_strided_copy(&input, &[1024, 1024], &[1, 1024], 0, &dev).expect("strided_copy");

        scope.stop(p).expect("stop scope");
        p.flush_cuda_kernels();
    });

    let events: &[ProfileEvent] = report.events();
    assert_eq!(events.len(), 1);
    let evt = &events[0];
    assert_eq!(evt.name, "strided_copy");
    // The GPU-measured duration must be > 0 — if it's zero, the
    // CUDA event timing fell back to wall-clock or we measured an
    // empty interval. 1M-element strided copy takes ~10-100 µs on
    // an RTX 3090.
    assert!(
        evt.duration_us > 0,
        "expected non-zero GPU duration for 1M-element strided_copy, got {}",
        evt.duration_us
    );
}

#[test]
fn flush_cuda_kernels_is_idempotent() {
    let Some(c) = ctx() else { return };
    let stream = c.default_stream();

    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        let scope = CudaKernelScope::new(&c, &stream, "k1", "cuda_kernel").unwrap();
        scope.stop(p).unwrap();
        p.flush_cuda_kernels();
        // Second flush should be a no-op since the queue is empty.
        p.flush_cuda_kernels();
        p.flush_cuda_kernels();
    });

    assert_eq!(
        report.events().len(),
        1,
        "duplicate flushes must not duplicate events"
    );
}

#[test]
fn flush_cuda_kernels_handles_empty_queue() {
    // Calling flush with no pending scopes should not error or panic.
    let _c_present = ctx().is_some();
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.flush_cuda_kernels();
        assert_eq!(p.pending_cuda_count(), 0);
    });
    assert!(report.events().is_empty());
}

#[test]
fn multiple_kernels_finalize_in_registration_order() {
    use ferrotorch_profiler::ProfileEvent;
    let Some(c) = ctx() else { return };
    let stream = c.default_stream();

    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        for i in 0..3 {
            let scope = CudaKernelScope::new(&c, &stream, format!("k{i}"), "cuda_kernel").unwrap();
            scope.stop(p).unwrap();
        }
        assert_eq!(p.pending_cuda_count(), 3);
        p.flush_cuda_kernels();
        assert_eq!(p.pending_cuda_count(), 0);
    });

    let events: &[ProfileEvent] = report.events();
    assert_eq!(events.len(), 3);
    assert_eq!(events[0].name, "k0");
    assert_eq!(events[1].name, "k1");
    assert_eq!(events[2].name, "k2");
}
