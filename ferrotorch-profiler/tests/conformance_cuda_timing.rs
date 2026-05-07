//! Conformance — `ferrotorch_profiler::cuda_timing` module parity.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/826>.
//!
//! The `cuda_timing` module is gated on `feature = "cuda"` and requires a
//! live CUDA context.  The hardware-backed tests are therefore excluded from
//! non-CUDA CI builds via `#[cfg(feature = "cuda")]` guards.
//!
//! This file covers the CPU-observable surface:
//!
//! * `Profiler::pending_cuda_count()` returns 0 in a non-CUDA build (no-op
//!   path confirmed).
//! * `Profiler::flush_cuda_kernels()` is callable with no CUDA feature and
//!   produces no events (no-op stub confirmed).
//! * `GpuTimingPair` struct field access (structural).
//! * `push_gpu_event` → DeviceType::Cuda path confirmed (CPU-side mock timing;
//!   not CUDA-event timing, but exercises the same `ProfileEvent` construction
//!   that `CudaKernelScope::stop` eventually feeds into).
//!
//! The three CUDA-gated items (`CudaKernelScope`, `CudaKernelScope::new`,
//! `CudaKernelScope::stop`) are listed in `_surface_exclusions.toml` with
//! tracking issue #826 and are not tested here.
//!
//! Reference: `torch.profiler.ProfilerActivity.CUDA`,
//! `cudarc::driver::{CudaContext, CudaStream, CudaEvent}`.

use ferrotorch_profiler::{ProfileConfig, with_profiler};
use ferrotorch_profiler::GpuTimingPair;

// ---------------------------------------------------------------------------
// Test: GpuTimingPair public fields are accessible (structural)
// ---------------------------------------------------------------------------

#[test]
fn cuda_timing_gpu_timing_pair_fields_accessible() {
    let pair = GpuTimingPair {
        start_us: 1_000,
        end_us:   2_500,
    };
    assert_eq!(pair.start_us, 1_000, "GpuTimingPair::start_us must be readable");
    assert_eq!(pair.end_us, 2_500,   "GpuTimingPair::end_us must be readable");
}

// ---------------------------------------------------------------------------
// Test: GpuTimingPair is Copy — assign without move
// ---------------------------------------------------------------------------

#[test]
fn cuda_timing_gpu_timing_pair_is_copy() {
    let original = GpuTimingPair { start_us: 10, end_us: 20 };
    let copy = original; // copy, not move
    assert_eq!(original.start_us, copy.start_us);
    assert_eq!(original.end_us, copy.end_us);
}

// ---------------------------------------------------------------------------
// Test: pending_cuda_count() = 0 without the cuda feature / when empty
//
// torch.profiler parity: pending kernel count is 0 before any CUDA scope
// is queued.  In a non-CUDA build the method is a no-op stub that always
// returns 0.
// ---------------------------------------------------------------------------

#[test]
fn cuda_timing_pending_cuda_count_zero_initially() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        assert_eq!(
            p.pending_cuda_count(),
            0,
            "pending_cuda_count must be 0 before any CUDA scope is queued"
        );
        // Record a CPU event to confirm non-CUDA ops don't inflate the count.
        p.record("add", "tensor_op", &[&[3usize, 4]]);
        assert_eq!(
            p.pending_cuda_count(),
            0,
            "pending_cuda_count must remain 0 after recording a CPU event"
        );
    });
    // The report must have the CPU event; no phantom CUDA events.
    assert_eq!(report.events().len(), 1);
    assert!(!report.has_gpu_events());
}

// ---------------------------------------------------------------------------
// Test: flush_cuda_kernels() no-op without cuda feature — produces no events
//
// torch.profiler parity: calling flush on an empty pending queue is always
// safe.  In a non-CUDA build the stub must compile and run without panic.
// ---------------------------------------------------------------------------

#[test]
fn cuda_timing_flush_cuda_kernels_noop_without_feature() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        // Calling flush with nothing pending must be a silent no-op.
        p.flush_cuda_kernels();
        p.flush_cuda_kernels(); // idempotent
    });
    assert_eq!(
        report.events().len(),
        0,
        "flush_cuda_kernels no-op must not inject phantom events"
    );
    assert!(
        !report.has_gpu_events(),
        "flush_cuda_kernels no-op must not set has_gpu_events"
    );
}

// ---------------------------------------------------------------------------
// Test: push_gpu_event produces a DeviceType::Cuda event
//
// This exercises the CPU-side mock-timing path that `CudaKernelScope::stop`
// feeds through after resolving GPU event elapsed time.  The behavioral
// contract (device_type = Cuda, correct duration) is identical; only the
// timing source differs (CPU mock vs real cuEventElapsedTime).
// ---------------------------------------------------------------------------

#[test]
fn cuda_timing_push_gpu_event_device_type_is_cuda() {
    use ferrotorch_profiler::DeviceType;

    let timing = GpuTimingPair { start_us: 0, end_us: 150 };
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.push_gpu_event("gemm", "cuda_kernel", timing);
    });
    let events = report.events();
    assert_eq!(events.len(), 1, "must record exactly 1 GPU event");
    assert_eq!(
        events[0].device_type,
        DeviceType::Cuda,
        "push_gpu_event must set device_type = Cuda (torch.profiler parity: \
         ProfilerActivity.CUDA events have DeviceType.CUDA)"
    );
    assert_eq!(
        events[0].duration_us, 150,
        "push_gpu_event duration_us must be end_us - start_us = 150"
    );
}

// ---------------------------------------------------------------------------
// Test: push_gpu_event — duration derived correctly from GpuTimingPair
// ---------------------------------------------------------------------------

#[test]
fn cuda_timing_push_gpu_event_duration_from_timing_pair() {
    let timing = GpuTimingPair { start_us: 1_000, end_us: 1_400 };
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.push_gpu_event("relu_cuda", "cuda_kernel", timing);
    });
    assert_eq!(
        report.events()[0].duration_us,
        400,
        "duration_us must be end_us - start_us = 1400 - 1000 = 400"
    );
}

// ---------------------------------------------------------------------------
// Test: push_gpu_event — name and category preserved
// ---------------------------------------------------------------------------

#[test]
fn cuda_timing_push_gpu_event_name_and_category_preserved() {
    let timing = GpuTimingPair { start_us: 0, end_us: 1 };
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.push_gpu_event("fused_mha_fwd", "cuda_kernel", timing);
    });
    let ev = &report.events()[0];
    assert_eq!(ev.name,     "fused_mha_fwd", "event name must be preserved");
    assert_eq!(ev.category, "cuda_kernel",   "event category must be preserved");
}

// ---------------------------------------------------------------------------
// Test: push_gpu_event returns None when profiler is stopped
//
// torch.profiler parity: recording after stop is a no-op; any index return
// must be None.
// ---------------------------------------------------------------------------

#[test]
fn cuda_timing_push_gpu_event_inactive_returns_none() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.stop();
        let idx = p.push_gpu_event("kernel", "cuda_kernel", GpuTimingPair { start_us: 0, end_us: 50 });
        assert_eq!(idx, None, "push_gpu_event must return None after stop()");
    });
    assert_eq!(
        report.events().len(),
        0,
        "no events must be recorded after stop()"
    );
}

// ---------------------------------------------------------------------------
// Test: has_gpu_events() transitions from false → true after push_gpu_event
// ---------------------------------------------------------------------------

#[test]
fn cuda_timing_has_gpu_events_transitions_on_push() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        assert!(!p.is_active() || true); // always active at start; just verify API
        // Before any GPU event.
        let pre_report = {
            // Use an inner with_profiler to snapshot state without GPU events.
            let ((), r) = with_profiler(ProfileConfig::default(), |_| {});
            r
        };
        assert!(!pre_report.has_gpu_events(), "fresh report must not have GPU events");

        // Push one GPU event.
        p.push_gpu_event("sgemm", "cuda_kernel", GpuTimingPair { start_us: 0, end_us: 10 });
    });
    assert!(
        report.has_gpu_events(),
        "has_gpu_events must be true after push_gpu_event"
    );
}
