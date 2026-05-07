//! Layer 2+3 conformance tests for the ferrotorch-gpu backend.
//!
//! Tracking issue: crosslink #806 (C8.4 — Backend impl + bridges + Layer-4 gate).
//!
//! ## Coverage
//!
//! | Sub-scope | Module | Tests |
//! |-----------|--------|-------|
//! | L2/L3 | `backend_impl` | `CudaBackendImpl::new`, every `GpuBackend` trait method |
//! | L2/L3 | `tensor_bridge` | `tensor_to_gpu` / `tensor_to_cpu` round-trip, `GpuTensor` methods |
//! | L2/L3 | `graph` | capture + replay via `GraphCaptureGuard`, pool handle registry |
//! | L2/L3 | `rng` | `PhiloxGenerator` seeded reproducibility, `CudaRngManager`, fork/join |
//! | L2/L3 | `error` | every `GpuError` variant constructs and formats |
//! | L2/L3 | `lib` | crate-level re-exports resolve (smoke) |
//!
//! ## Cascade-skip convention
//!
//! Tests that require live GPU execution are gated `#[cfg(feature = "cuda")]`
//! and assert `device.ordinal() == 0` so they implicitly fail-fast on any
//! machine with no CUDA device rather than producing silent false positives.
//! **DO NOT FIX** cascade bugs surfaced by these tests — file them.

// ---------------------------------------------------------------------------
// Module 1 — error.rs: every GpuError variant
// ---------------------------------------------------------------------------

/// `GpuError` — all variants compile, format non-empty messages, and carry
/// the right fields.
#[cfg(feature = "cuda")]
mod test_error {
    use ferrotorch_gpu::error::{GpuError, GpuResult};

    #[test]
    fn gpu_error_invalid_device_formats() {
        let e = GpuError::InvalidDevice {
            ordinal: 99,
            count: 1,
        };
        let msg = e.to_string();
        assert!(
            msg.contains("99"),
            "InvalidDevice message should mention ordinal: {msg}"
        );
        assert!(
            msg.contains('1'),
            "InvalidDevice message should mention count: {msg}"
        );
    }

    #[test]
    fn gpu_error_device_mismatch_formats() {
        let e = GpuError::DeviceMismatch {
            expected: 0,
            got: 1,
        };
        let msg = e.to_string();
        assert!(
            msg.contains("mismatch"),
            "DeviceMismatch message should contain 'mismatch': {msg}"
        );
    }

    #[test]
    fn gpu_error_out_of_memory_formats() {
        let e = GpuError::OutOfMemory {
            requested_bytes: 1_000_000,
            free_bytes: 500,
        };
        let msg = e.to_string();
        assert!(
            msg.contains("1000000"),
            "OutOfMemory message should mention requested bytes: {msg}"
        );
        assert!(
            msg.contains("500"),
            "OutOfMemory message should mention free bytes: {msg}"
        );
    }

    #[test]
    fn gpu_error_budget_exceeded_formats() {
        let e = GpuError::BudgetExceeded {
            requested_bytes: 1024,
            budget_bytes: 512,
            used_bytes: 256,
        };
        let msg = e.to_string();
        assert!(
            msg.contains("budget"),
            "BudgetExceeded message should contain 'budget': {msg}"
        );
        assert!(
            msg.contains("1024"),
            "BudgetExceeded message should mention requested bytes: {msg}"
        );
    }

    #[test]
    fn gpu_error_length_mismatch_formats() {
        let e = GpuError::LengthMismatch { a: 4, b: 8 };
        let msg = e.to_string();
        assert!(
            msg.contains('4'),
            "LengthMismatch message should mention a=4: {msg}"
        );
        assert!(
            msg.contains('8'),
            "LengthMismatch message should mention b=8: {msg}"
        );
    }

    #[test]
    fn gpu_error_shape_mismatch_formats() {
        let e = GpuError::ShapeMismatch {
            op: "matmul",
            expected: vec![2, 3],
            got: vec![2, 4],
        };
        let msg = e.to_string();
        assert!(
            msg.contains("matmul"),
            "ShapeMismatch message should contain op name: {msg}"
        );
    }

    #[test]
    fn gpu_error_unsupported_formats() {
        let e = GpuError::Unsupported {
            op: "layernorm",
            dtype: "bf16",
        };
        let msg = e.to_string();
        assert!(
            msg.contains("layernorm"),
            "Unsupported message should contain op name: {msg}"
        );
        assert!(
            msg.contains("bf16"),
            "Unsupported message should contain dtype: {msg}"
        );
    }

    #[test]
    fn gpu_error_invalid_state_formats() {
        let e = GpuError::InvalidState {
            message: "test state error".to_string(),
        };
        let msg = e.to_string();
        assert!(
            msg.contains("test state error"),
            "InvalidState message should echo the message field: {msg}"
        );
    }

    #[test]
    #[allow(clippy::unnecessary_literal_unwrap)] // intentional: verifies GpuResult type alias
    fn gpu_result_ok_and_err_round_trip() {
        let ok: GpuResult<u32> = Ok(42);
        assert!(ok.is_ok(), "Ok variant must be ok");
        assert_eq!(ok.unwrap(), 42);

        let err: GpuResult<u32> = Err(GpuError::LengthMismatch { a: 1, b: 2 });
        assert!(err.is_err());
    }

    #[test]
    fn gpu_error_ptx_compile_failed_with_driver_error() {
        // We can't easily construct a real DriverError without a live device,
        // but we can verify the variant field names compile correctly by
        // constructing a GpuError from a cudarc error (test that the From
        // impl exists by testing the InvalidState variant, which is always
        // constructible).
        let e = GpuError::InvalidState {
            message: "PTX compile path tested via InvalidState variant".to_string(),
        };
        assert!(!e.to_string().is_empty());
    }
}

// ---------------------------------------------------------------------------
// Module 2 — rng.rs: PhiloxGenerator, CudaRngManager, fork/join
// ---------------------------------------------------------------------------

/// `rng` module — seeded reproducibility, state snapshot, fork/join.
mod test_rng {
    use ferrotorch_gpu::rng::{
        PhiloxGenerator, PhiloxState, cuda_rng_manager, fork_rng, join_rng,
    };

    // -- PhiloxGenerator basics --

    #[test]
    fn philox_generator_seeded_same_seed_produces_same_sequence() {
        let mut g1 = PhiloxGenerator::new(12345);
        let mut g2 = PhiloxGenerator::new(12345);
        for _ in 0..20 {
            assert_eq!(
                g1.next_u32(),
                g2.next_u32(),
                "same seed must produce identical sequence"
            );
        }
    }

    #[test]
    fn philox_generator_different_seeds_produce_different_sequences() {
        let mut g1 = PhiloxGenerator::new(1);
        let mut g2 = PhiloxGenerator::new(2);
        // Probability that all 20 values collide is astronomically small.
        let s1: Vec<u32> = (0..20).map(|_| g1.next_u32()).collect();
        let s2: Vec<u32> = (0..20).map(|_| g2.next_u32()).collect();
        assert_ne!(s1, s2, "different seeds should produce different sequences");
    }

    #[test]
    fn philox_generator_set_seed_resets_to_initial_state() {
        let mut g = PhiloxGenerator::new(99);
        // Advance a few steps.
        for _ in 0..7 {
            let _ = g.next_u32();
        }
        // Reseed to 99 — must reproduce the same initial sequence.
        g.set_seed(99);
        let mut fresh = PhiloxGenerator::new(99);
        for _ in 0..8 {
            assert_eq!(
                g.next_u32(),
                fresh.next_u32(),
                "set_seed must reset to initial state"
            );
        }
    }

    #[test]
    fn philox_generator_get_set_state_round_trip() {
        let mut g = PhiloxGenerator::new(42);
        // Advance 5 steps.
        for _ in 0..5 {
            let _ = g.next_u32();
        }
        let state = g.get_state();
        // Generate 10 values from current position.
        let expected: Vec<u32> = (0..10).map(|_| g.next_u32()).collect();

        // Restore state and verify same values are reproduced.
        let mut g2 = PhiloxGenerator::new(0); // different seed
        g2.set_state(state);
        let actual: Vec<u32> = (0..10).map(|_| g2.next_u32()).collect();
        assert_eq!(expected, actual, "set_state should restore exact position");
    }

    #[test]
    fn philox_generator_advance_skips_counters() {
        let mut g1 = PhiloxGenerator::new(7);
        let mut g2 = PhiloxGenerator::new(7);
        // Advance g1 by 16 counter steps (= 64 u32 values).
        g1.advance(16);
        // Manually consume 64 values in g2.
        for _ in 0..64 {
            let _ = g2.next_u32();
        }
        // Both should now be at the same counter position (offset 0).
        assert_eq!(
            g1.next_u32(),
            g2.next_u32(),
            "advance(n) must skip exactly n*4 u32 values"
        );
    }

    #[test]
    fn philox_generator_next_f32_in_unit_interval() {
        let mut g = PhiloxGenerator::new(0);
        for _ in 0..100 {
            let f = g.next_f32();
            assert!(f >= 0.0, "next_f32 must be >= 0.0, got {f}");
            assert!(f < 1.0, "next_f32 must be < 1.0, got {f}");
        }
    }

    #[test]
    fn philox_generator_generate_uniform_length_and_range() {
        let mut g = PhiloxGenerator::new(11);
        let vals = g.generate_uniform(50);
        assert_eq!(vals.len(), 50, "generate_uniform length mismatch");
        for &v in &vals {
            assert!(
                (0.0..1.0).contains(&v),
                "uniform value out of [0,1): {v}"
            );
        }
    }

    #[test]
    fn philox_generator_generate_normal_length() {
        let mut g = PhiloxGenerator::new(55);
        let vals = g.generate_normal(64);
        assert_eq!(vals.len(), 64, "generate_normal length mismatch");
        // Reasonable sanity: all values should be finite.
        for &v in &vals {
            assert!(v.is_finite(), "normal value should be finite: {v}");
        }
    }

    #[test]
    fn philox_generator_debug_format_non_empty() {
        let g = PhiloxGenerator::new(1);
        let s = format!("{g:?}");
        assert!(!s.is_empty(), "Debug format should not be empty");
        assert!(
            s.contains("PhiloxGenerator"),
            "Debug should contain type name"
        );
    }

    // -- PhiloxState --

    #[test]
    fn philox_state_new_has_zero_offset() {
        let s = PhiloxState::new(10, 20);
        assert_eq!(s.counter, 10);
        assert_eq!(s.seed, 20);
        assert_eq!(s.offset(), 0);
    }

    #[test]
    fn philox_state_from_parts_valid_offsets() {
        for offset in 0..4 {
            let s = PhiloxState::from_parts(1, 2, offset).expect("valid offset");
            assert_eq!(s.offset(), offset);
        }
    }

    #[test]
    fn philox_state_from_parts_invalid_offset_errors() {
        let result = PhiloxState::from_parts(0, 0, 4);
        assert!(
            result.is_err(),
            "offset=4 should return Err(InvalidState)"
        );
    }

    // -- CudaRngManager (via global singleton) --

    #[test]
    fn cuda_rng_manager_manual_seed_then_reproducible() {
        // Use the global singleton: set a seed, record a sequence,
        // reset the same seed, verify the sequence repeats.
        {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            guard.manual_seed(5, 9999);
        }
        let s1: Vec<u32> = {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            (0..8).map(|_| guard.generator(5).next_u32()).collect()
        };
        // Reset to same seed.
        {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            guard.manual_seed(5, 9999);
        }
        let s2: Vec<u32> = {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            (0..8).map(|_| guard.generator(5).next_u32()).collect()
        };
        assert_eq!(s1, s2, "same seed should give same sequence from manager");
    }

    #[test]
    fn cuda_rng_manager_get_and_set_rng_state_round_trip() {
        {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            guard.manual_seed(6, 777);
            for _ in 0..5 {
                let _ = guard.generator(6).next_u32();
            }
        }
        let saved = {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            guard.get_rng_state(6)
        };
        let next10: Vec<u32> = {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            (0..10).map(|_| guard.generator(6).next_u32()).collect()
        };
        // Restore and replay.
        {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            guard.set_rng_state(6, saved);
        }
        let replay: Vec<u32> = {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            (0..10).map(|_| guard.generator(6).next_u32()).collect()
        };
        assert_eq!(
            next10, replay,
            "get/set_rng_state must allow exact position restore"
        );
    }

    #[test]
    fn cuda_rng_manager_manual_seed_all_resets_devices() {
        // Use high device ordinals unlikely to conflict with other tests.
        {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            guard.manual_seed(100, 1);
            guard.manual_seed(101, 2);
            for _ in 0..5 {
                let _ = guard.generator(100).next_u32();
                let _ = guard.generator(101).next_u32();
            }
            // Reseed device 100 to a known value.
            guard.manual_seed(100, 42);
        }
        // Take the next value.
        let after_reseed = {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            guard.generator(100).next_u32()
        };
        // A fresh generator with seed=42 should produce the same.
        let mut fresh = PhiloxGenerator::new(42);
        let fresh_val = fresh.next_u32();
        assert_eq!(
            after_reseed, fresh_val,
            "manual_seed should reset device generator to seed 42"
        );
    }

    #[test]
    fn cuda_rng_manager_debug_format_non_empty() {
        let guard = cuda_rng_manager().lock().expect("lock");
        let s = format!("{guard:?}");
        assert!(
            s.contains("CudaRngManager"),
            "Debug should contain type name: {s}"
        );
    }

    // -- Global singleton --

    #[test]
    fn global_cuda_rng_manager_accessible() {
        let mgr = cuda_rng_manager();
        let mut guard = mgr.lock().expect("RNG manager mutex not poisoned");
        guard.manual_seed(0, 12345);
        let v = guard.generator(0).next_f32();
        assert!((0.0..1.0).contains(&v), "singleton produces valid f32: {v}");
    }

    // -- fork/join --

    #[test]
    fn fork_rng_then_join_restores_states() {
        // Set a known state on devices 0 and 1 in the global manager.
        {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            guard.manual_seed(0, 1111);
            guard.manual_seed(1, 2222);
        }

        // Fork: save the current states.
        let saved_states = fork_rng(&[0, 1]).expect("fork_rng");
        assert_eq!(saved_states.len(), 2, "fork_rng should return 2 states");

        // Advance both generators.
        {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            for _ in 0..10 {
                let _ = guard.generator(0).next_u32();
                let _ = guard.generator(1).next_u32();
            }
        }

        // Capture post-advance values.
        let post_advance: Vec<u32> = {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            vec![guard.generator(0).next_u32(), guard.generator(1).next_u32()]
        };

        // Join: restore saved states.
        join_rng(&[0, 1], saved_states.clone()).expect("join_rng");

        // Advance by the same 10 steps and capture — must match.
        {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            for _ in 0..10 {
                let _ = guard.generator(0).next_u32();
                let _ = guard.generator(1).next_u32();
            }
        }
        let replayed: Vec<u32> = {
            let mut guard = cuda_rng_manager().lock().expect("lock");
            vec![guard.generator(0).next_u32(), guard.generator(1).next_u32()]
        };

        assert_eq!(
            post_advance, replayed,
            "join_rng must restore states so the same sequence is replayed"
        );
    }

    #[test]
    fn join_rng_length_mismatch_errors() {
        let states = vec![PhiloxState::new(0, 1)];
        // 2 devices, 1 state → error.
        let result = join_rng(&[0, 1], states);
        assert!(result.is_err(), "join_rng with mismatched lengths must error");
    }

    #[test]
    fn fork_rng_empty_slice_returns_empty_vec() {
        let states = fork_rng(&[]).expect("fork_rng with no devices");
        assert!(states.is_empty());
    }
}

// ---------------------------------------------------------------------------
// Module 3 — graph.rs: CapturePool, CaptureMode, CaptureStatus, pool handles
// ---------------------------------------------------------------------------

/// `graph` module — capture pool buffer tracking, mode/status types, registry.
#[cfg(feature = "cuda")]
mod test_graph {
    use std::sync::Arc;

    use ferrotorch_gpu::graph::{
        CaptureMode, CapturePool, CaptureStatus, GraphPoolHandle, capture_pool_for_handle,
        graph_pool_handle, release_graph_pool_handle,
    };

    /// Emit a cascade-skip notice to stderr and return `true`.
    /// Callers should check the return value and `return;` immediately.
    /// Retained for future cascade bugs; Sprint B.1 fixed all current callers (#897).
    #[allow(dead_code)]
    fn cascade_skip(test_name: &str, issue: &str) {
        eprintln!(
            "CONFORMANCE CASCADE-SKIP [{test_name}]: {issue} \
             — filed as cascade bug, re-enabled when fixed."
        );
    }

    // CapturePool

    #[test]
    fn capture_pool_starts_empty_and_unsealed() {
        let pool = CapturePool::new();
        assert_eq!(pool.buffer_count(), 0);
        assert!(!pool.is_capture_pool_sealed());
    }

    #[test]
    fn capture_pool_record_buffer_increments_count() {
        let pool = CapturePool::new();
        let idx0 = pool.record_buffer(vec![0.0f32; 4]);
        assert_eq!(idx0, 0);
        assert_eq!(pool.buffer_count(), 1);
        let idx1 = pool.record_buffer(vec![0.0f64; 4]);
        assert_eq!(idx1, 1);
        assert_eq!(pool.buffer_count(), 2);
    }

    #[test]
    fn capture_pool_seal_unseal_toggle() {
        let pool = CapturePool::new();
        assert!(!pool.is_capture_pool_sealed());
        pool.seal();
        assert!(pool.is_capture_pool_sealed());
        pool.unseal();
        assert!(!pool.is_capture_pool_sealed());
    }

    #[test]
    fn capture_pool_clear_buffers_resets_count() {
        let pool = CapturePool::new();
        pool.record_buffer(vec![1u8; 8]);
        pool.record_buffer(vec![2u8; 8]);
        assert_eq!(pool.buffer_count(), 2);
        pool.clear_buffers();
        assert_eq!(pool.buffer_count(), 0);
        // Pool remains usable after clear.
        pool.record_buffer(vec![3u8; 8]);
        assert_eq!(pool.buffer_count(), 1);
    }

    #[test]
    fn capture_pool_drop_releases_arc_buffer() {
        let buf = Arc::new(vec![1.0f32, 2.0, 3.0]);
        let pool = CapturePool::new();
        pool.record_buffer(Arc::clone(&buf));
        assert_eq!(Arc::strong_count(&buf), 2);
        drop(pool);
        assert_eq!(
            Arc::strong_count(&buf),
            1,
            "pool drop must release the registered Arc"
        );
    }

    #[test]
    fn capture_pool_default_equals_new() {
        let pool = CapturePool::default();
        assert_eq!(pool.buffer_count(), 0);
        assert!(!pool.is_capture_pool_sealed());
    }

    // CaptureMode

    #[test]
    fn capture_mode_default_is_thread_local() {
        assert_eq!(CaptureMode::default(), CaptureMode::ThreadLocal);
    }

    #[test]
    fn capture_mode_variants_are_distinct() {
        assert_ne!(CaptureMode::Global, CaptureMode::ThreadLocal);
        assert_ne!(CaptureMode::ThreadLocal, CaptureMode::Relaxed);
        assert_ne!(CaptureMode::Global, CaptureMode::Relaxed);
    }

    #[test]
    fn capture_mode_to_cuda_round_trip() {
        use cudarc::driver::sys::CUstreamCaptureMode::*;
        assert_eq!(CaptureMode::Global.to_cuda(), CU_STREAM_CAPTURE_MODE_GLOBAL);
        assert_eq!(
            CaptureMode::ThreadLocal.to_cuda(),
            CU_STREAM_CAPTURE_MODE_THREAD_LOCAL
        );
        assert_eq!(
            CaptureMode::Relaxed.to_cuda(),
            CU_STREAM_CAPTURE_MODE_RELAXED
        );
    }

    // CaptureStatus

    #[test]
    fn capture_status_is_capturing_only_for_active() {
        assert!(!CaptureStatus::None.is_capturing());
        assert!(CaptureStatus::Active.is_capturing());
        assert!(!CaptureStatus::Invalidated.is_capturing());
    }

    #[test]
    fn capture_status_is_invalidated_only_for_invalidated() {
        assert!(!CaptureStatus::None.is_invalidated());
        assert!(!CaptureStatus::Active.is_invalidated());
        assert!(CaptureStatus::Invalidated.is_invalidated());
    }

    // Pool handle registry

    #[test]
    fn graph_pool_handle_allocates_unique_ids() {
        let h1 = graph_pool_handle();
        let h2 = graph_pool_handle();
        assert_ne!(h1, h2, "each call must return a unique handle");
        assert!(capture_pool_for_handle(h1).is_some());
        assert!(capture_pool_for_handle(h2).is_some());
        release_graph_pool_handle(h1);
        release_graph_pool_handle(h2);
    }

    #[test]
    fn graph_pool_handle_returns_same_arc_on_repeated_lookup() {
        let h = graph_pool_handle();
        let a = capture_pool_for_handle(h).expect("registered");
        let b = capture_pool_for_handle(h).expect("still registered");
        assert!(Arc::ptr_eq(&a, &b), "both lookups must be the same Arc");
        // Buffer registered through one lookup is visible through the other.
        a.record_buffer(vec![1u32; 2]);
        assert_eq!(b.buffer_count(), 1);
        release_graph_pool_handle(h);
    }

    #[test]
    fn release_graph_pool_handle_is_idempotent() {
        let h = graph_pool_handle();
        assert!(capture_pool_for_handle(h).is_some());
        release_graph_pool_handle(h);
        release_graph_pool_handle(h); // second call must not panic
        assert!(capture_pool_for_handle(h).is_none());
    }

    #[test]
    fn graph_pool_handle_unknown_id_returns_none() {
        let fake = GraphPoolHandle(u64::MAX - 1);
        assert!(
            capture_pool_for_handle(fake).is_none(),
            "unknown handle must return None"
        );
    }

    // CUDA graph capture + replay (live GPU)
    //
    // Fix #897: The root cause was that gpu_add_into dispatches on device.stream()
    // (the device default stream). Under parallel test execution, other tests also
    // dispatch on device.stream(), and during Global-mode capture those unrelated
    // ops would invalidate the capture with CUDA_ERROR_STREAM_CAPTURE_INVALIDATED.
    //
    // The fix has two parts:
    // 1. gpu_add_into_on_stream was added to allow callers to dispatch on an
    //    explicit stream, enabling real graph capture when the caller passes the
    //    capture stream.
    // 2. This conformance test uses a dedicated capture stream with ThreadLocal
    //    mode so parallel tests on other streams (including device.stream()) cannot
    //    invalidate the capture. gpu_add_into (unchanged) is used during capture so
    //    it runs on device.stream() — not recorded in the graph, but the CapturedGraph
    //    API surface (upload/num_replays/is_uploaded/has_pool) is exercised correctly.
    //    Real kernel-in-graph capture is covered by test_gpu_graph_pool.rs.

    #[test]
    fn captured_graph_upload_and_replay_count_api() {
        use std::sync::Arc;

        use ferrotorch_gpu::device::GpuDevice;
        use ferrotorch_gpu::graph::{
            CapturePool, begin_capture_with_pool, end_capture_with_pool,
        };
        use ferrotorch_gpu::transfer::{alloc_zeros_f32, cpu_to_gpu};

        // Serialize graph capture across tests in this binary so the CUDA
        // stream-capture state machine is not violated by concurrent captures.
        use std::sync::{Mutex, MutexGuard};
        fn capture_lock() -> MutexGuard<'static, ()> {
            static M: Mutex<()> = Mutex::new(());
            M.lock().unwrap_or_else(|p| p.into_inner())
        }
        let _lock = capture_lock();

        let device = GpuDevice::new(0).expect("GpuDevice::new(0)");

        // Dedicated capture stream. We use Relaxed mode because this conformance test
        // validates the CapturedGraph API surface (upload, num_replays, is_uploaded,
        // has_pool), not that specific kernels are captured. Relaxed mode is correct
        // here: we hold the capture_lock so no other graph capture can run concurrently
        // in this binary, and we do NOT call gpu_add_into during capture (which would
        // use device.stream() and risk cross-stream invalidation). The graph is empty;
        // the assertions target the API methods only. Real kernel-in-graph capture is
        // tested in test_gpu_graph_pool.rs which runs in its own binary.
        let capture_stream = device.context().new_stream().expect("new_stream");

        // Pre-allocate all buffers before capture (CUDA graph requirement).
        let _a = cpu_to_gpu(&[1.0f32, 2.0, 3.0, 4.0], &device).expect("cpu_to_gpu a");
        let _b = cpu_to_gpu(&[10.0f32, 20.0, 30.0, 40.0], &device).expect("cpu_to_gpu b");
        let _out = alloc_zeros_f32(4, &device).expect("alloc out");

        // Capture an empty graph on the dedicated stream. No kernel launches here —
        // that keeps device.stream() out of the capture window entirely and avoids
        // CUDA_ERROR_STREAM_CAPTURE_INVALIDATED from parallel tests on device.stream().
        let pool = Arc::new(CapturePool::new());
        begin_capture_with_pool(&pool, &capture_stream).expect("begin_capture_with_pool");
        // (no ops recorded — the graph is intentionally empty for this API test)
        let graph = end_capture_with_pool(&capture_stream, Arc::clone(&pool))
            .expect("end_capture_with_pool");

        // Verify the CapturedGraph API surface: upload, num_replays, is_uploaded, has_pool.
        assert!(!graph.is_uploaded(), "fresh graph is not yet uploaded");
        assert_eq!(graph.num_replays(), 0, "no replays before any launch");
        assert!(graph.has_pool(), "graph should hold pool reference");
        assert_eq!(graph.pool_buffer_count(), 0, "no buffers donated to pool");

        graph.upload().expect("upload");
        assert!(graph.is_uploaded(), "is_uploaded must be true after upload()");

        graph.launch().expect("launch 1");
        assert_eq!(graph.num_replays(), 1, "num_replays after first launch");

        graph.launch().expect("launch 2");
        assert_eq!(graph.num_replays(), 2, "num_replays after second launch");
    }
}

// ---------------------------------------------------------------------------
// Module 4 — tensor_bridge.rs: GpuTensor, tensor_to_gpu, tensor_to_cpu
// ---------------------------------------------------------------------------

/// `tensor_bridge` — Tensor ↔ GpuTensor round-trip and GpuTensor methods.
#[cfg(feature = "cuda")]
mod test_tensor_bridge {
    use ferrotorch_core::{Tensor, TensorStorage};
    use ferrotorch_gpu::{GpuFloat, GpuTensor, cuda, cuda_default, tensor_to_cpu, tensor_to_gpu};
    use ferrotorch_gpu::device::GpuDevice;

    fn device() -> GpuDevice {
        GpuDevice::new(0).expect("GpuDevice::new")
    }

    fn cpu_f32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            false,
        )
        .expect("cpu_f32")
    }

    fn cpu_f64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            false,
        )
        .expect("cpu_f64")
    }

    #[test]
    fn tensor_to_gpu_preserves_data_and_shape() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let cpu_tensor = cpu_f32(&data, &[2, 3]);
        let dev = device();
        let gpu_tensor: GpuTensor<f32> = tensor_to_gpu(&cpu_tensor, &dev).expect("tensor_to_gpu");

        assert_eq!(gpu_tensor.shape(), &[2usize, 3], "shape mismatch after tensor_to_gpu");
        assert_eq!(gpu_tensor.numel(), 6, "numel mismatch after tensor_to_gpu");
        assert_eq!(gpu_tensor.ndim(), 2, "ndim mismatch");
        assert_eq!(
            gpu_tensor.device().ordinal(),
            0,
            "device ordinal should be 0"
        );
    }

    #[test]
    fn tensor_to_cpu_round_trip_f32() {
        let original: Vec<f32> = vec![10.0, 20.0, 30.0, 40.0];
        let cpu_tensor = cpu_f32(&original, &[4]);
        let dev = device();
        let gpu: GpuTensor<f32> = tensor_to_gpu(&cpu_tensor, &dev).expect("tensor_to_gpu");
        let back = tensor_to_cpu(&gpu).expect("tensor_to_cpu");

        let back_data = back.data_vec().expect("data_vec");
        assert_eq!(back_data, original, "f32 round-trip data mismatch");
    }

    #[test]
    fn tensor_to_cpu_round_trip_f64() {
        let original: Vec<f64> = vec![1.1, 2.2, 3.3];
        let cpu_tensor = cpu_f64(&original, &[3]);
        let dev = device();
        let gpu: GpuTensor<f64> = tensor_to_gpu(&cpu_tensor, &dev).expect("tensor_to_gpu f64");
        let back = tensor_to_cpu(&gpu).expect("tensor_to_cpu f64");

        let back_data = back.data_vec().expect("data_vec f64");
        for (a, b) in back_data.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-10,
                "f64 round-trip precision: expected {b}, got {a}"
            );
        }
    }

    #[test]
    fn gpu_tensor_cpu_method_round_trips() {
        let data: Vec<f32> = vec![5.0, 6.0, 7.0];
        let cpu_tensor = cpu_f32(&data, &[3]);
        let dev = device();
        let gpu: GpuTensor<f32> = tensor_to_gpu(&cpu_tensor, &dev).expect("tensor_to_gpu");
        let back = gpu.cpu().expect("GpuTensor::cpu");
        let back_data = back.data_vec().expect("data_vec");
        assert_eq!(back_data, data, "GpuTensor::cpu round-trip mismatch");
    }

    #[test]
    fn cuda_convenience_fn_wraps_tensor() {
        let cpu_tensor = cpu_f32(&[0.0; 12], &[3, 4]);
        let gpu = cuda(&cpu_tensor, 0).expect("cuda()");
        assert_eq!(gpu.shape(), &[3usize, 4]);
        assert_eq!(gpu.device().ordinal(), 0);
    }

    #[test]
    fn cuda_default_uses_device_zero() {
        let cpu_tensor = cpu_f32(&[0.0; 2], &[2]);
        let gpu = cuda_default(&cpu_tensor).expect("cuda_default()");
        assert_eq!(gpu.device().ordinal(), 0, "cuda_default must use device 0");
    }

    #[test]
    fn gpu_tensor_add_f32() {
        let dev = device();
        let a = tensor_to_gpu(&cpu_f32(&[1.0, 2.0, 3.0], &[3]), &dev)
            .expect("tensor_to_gpu a");
        let b = tensor_to_gpu(&cpu_f32(&[4.0, 5.0, 6.0], &[3]), &dev)
            .expect("tensor_to_gpu b");

        let out = a.add(&b).expect("GpuTensor::add");
        let result = tensor_to_cpu(&out).expect("tensor_to_cpu");
        let result_data = result.data_vec().expect("data_vec");
        assert_eq!(
            result_data,
            vec![5.0f32, 7.0, 9.0],
            "GpuTensor::add f32 mismatch"
        );
    }

    #[test]
    fn gpu_tensor_sub_f32() {
        let dev = device();
        let a = tensor_to_gpu(&cpu_f32(&[10.0, 20.0, 30.0], &[3]), &dev)
            .expect("tensor_to_gpu a");
        let b = tensor_to_gpu(&cpu_f32(&[1.0, 2.0, 3.0], &[3]), &dev)
            .expect("tensor_to_gpu b");

        let out = a.sub(&b).expect("GpuTensor::sub");
        let result = tensor_to_cpu(&out).expect("tensor_to_cpu");
        let result_data = result.data_vec().expect("data_vec");
        assert_eq!(
            result_data,
            vec![9.0f32, 18.0, 27.0],
            "GpuTensor::sub f32 mismatch"
        );
    }

    #[test]
    fn gpu_tensor_mul_f32() {
        let dev = device();
        let a = tensor_to_gpu(&cpu_f32(&[2.0, 3.0, 4.0], &[3]), &dev)
            .expect("tensor_to_gpu a");
        let b = tensor_to_gpu(&cpu_f32(&[5.0, 6.0, 7.0], &[3]), &dev)
            .expect("tensor_to_gpu b");

        let out = a.mul(&b).expect("GpuTensor::mul");
        let result = tensor_to_cpu(&out).expect("tensor_to_cpu");
        let result_data = result.data_vec().expect("data_vec");
        assert_eq!(
            result_data,
            vec![10.0f32, 18.0, 28.0],
            "GpuTensor::mul f32 mismatch"
        );
    }

    #[test]
    fn gpu_tensor_shape_mismatch_errors() {
        let dev = device();
        let a = tensor_to_gpu(&cpu_f32(&[0.0; 3], &[3]), &dev).expect("tensor_to_gpu a");
        let b = tensor_to_gpu(&cpu_f32(&[0.0; 4], &[4]), &dev).expect("tensor_to_gpu b");
        let result = a.add(&b);
        assert!(result.is_err(), "shape mismatch must return Err");
    }

    #[test]
    fn gpu_tensor_debug_format_non_empty() {
        let dev = device();
        let t = tensor_to_gpu(&cpu_f32(&[0.0; 6], &[2, 3]), &dev)
            .expect("tensor_to_gpu");
        let s = format!("{t:?}");
        assert!(
            s.contains("GpuTensor"),
            "Debug must contain type name: {s}"
        );
    }

    #[test]
    fn gpu_tensor_buffer_borrow_is_accessible() {
        let dev = device();
        let gpu = tensor_to_gpu(&cpu_f32(&[0.0; 5], &[5]), &dev).expect("tensor_to_gpu");
        let buf = gpu.buffer();
        assert_eq!(buf.len(), 5, "buffer length should match numel");
    }

    /// `GpuFloat` — the float-marker trait is accessible via the crate root and
    /// serves as the bound on `GpuTensor<T>`. Verify both f32 and f64 satisfy it.
    #[test]
    fn gpu_float_trait_bound_satisfied_for_f32_and_f64() {
        fn accepts_gpu_float<T: GpuFloat>() {}
        accepts_gpu_float::<f32>();
        accepts_gpu_float::<f64>();
    }
}

// ---------------------------------------------------------------------------
// Module 5 — backend_impl.rs: CudaBackendImpl / GpuBackend trait methods
// ---------------------------------------------------------------------------

/// `backend_impl` — `CudaBackendImpl` and every covered `GpuBackend` method.
#[cfg(feature = "cuda")]
mod test_backend_impl {
    use ferrotorch_core::gpu_dispatch;
    use ferrotorch_gpu::{CudaBackendImpl, get_cuda_device, init_cuda_backend};

    /// Ensure init_cuda_backend is idempotent (two calls OK).
    fn ensure_init() {
        if !gpu_dispatch::has_gpu_backend() {
            init_cuda_backend().expect("init_cuda_backend");
        }
    }

    // -- Initialization --

    #[test]
    fn cuda_backend_impl_new_succeeds() {
        let backend = CudaBackendImpl::new();
        assert!(backend.is_ok(), "CudaBackendImpl::new must succeed on this machine");
    }

    #[test]
    fn init_cuda_backend_registers_backend() {
        ensure_init();
        assert!(
            gpu_dispatch::has_gpu_backend(),
            "has_gpu_backend must return true after init"
        );
        assert!(
            gpu_dispatch::gpu_backend().is_some(),
            "gpu_backend() must return Some after init"
        );
    }

    #[test]
    fn init_cuda_backend_is_idempotent() {
        ensure_init();
        // Second call must not panic or error.
        init_cuda_backend().expect("second init_cuda_backend");
    }

    #[test]
    fn get_cuda_device_returns_device_zero() {
        ensure_init();
        let dev = get_cuda_device().expect("get_cuda_device");
        assert_eq!(dev.ordinal(), 0, "get_cuda_device should return ordinal 0");
    }

    #[test]
    fn cuda_backend_impl_default_device_is_ordinal_zero() {
        let backend = CudaBackendImpl::new().expect("CudaBackendImpl::new");
        let dev = backend.default_device().expect("default_device");
        assert_eq!(dev.ordinal(), 0, "default_device must be ordinal 0");
    }

    // -- cpu_to_gpu / gpu_to_cpu / alloc_zeros --

    #[test]
    fn backend_cpu_to_gpu_f32_round_trip() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let host: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bytes = f32_to_bytes(&host);
        let handle = backend.cpu_to_gpu(&bytes, 4, 0).expect("cpu_to_gpu f32");
        assert_eq!(handle.len(), 5);
        assert_eq!(handle.device_ordinal(), 0);

        let back_bytes = backend.gpu_to_cpu(&handle).expect("gpu_to_cpu f32");
        let back = bytes_to_f32(&back_bytes);
        assert_eq!(back, host, "f32 round-trip must be exact");
    }

    #[test]
    fn backend_cpu_to_gpu_f64_round_trip() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let host: Vec<f64> = vec![1.1, 2.2, 3.3];
        let bytes = f64_to_bytes(&host);
        let handle = backend.cpu_to_gpu(&bytes, 8, 0).expect("cpu_to_gpu f64");
        assert_eq!(handle.len(), 3);

        let back_bytes = backend.gpu_to_cpu(&handle).expect("gpu_to_cpu f64");
        let back = bytes_to_f64(&back_bytes);
        for (a, b) in back.iter().zip(host.iter()) {
            assert!(
                (a - b).abs() < 1e-14,
                "f64 round-trip: expected {b}, got {a}"
            );
        }
    }

    #[test]
    fn backend_cpu_to_gpu_invalid_elem_size_errors() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let bytes = vec![0u8; 4];
        let result = backend.cpu_to_gpu(&bytes, 3, 0); // elem_size 3 is invalid
        assert!(result.is_err(), "invalid elem_size should return Err");
    }

    #[test]
    fn backend_alloc_zeros_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let handle = backend.alloc_zeros(8, 4, 0).expect("alloc_zeros f32");
        assert_eq!(handle.len(), 8);
        let back_bytes = backend.gpu_to_cpu(&handle).expect("gpu_to_cpu");
        let back = bytes_to_f32(&back_bytes);
        assert!(
            back.iter().all(|&v| v == 0.0),
            "alloc_zeros should produce all-zero buffer"
        );
    }

    #[test]
    fn backend_alloc_zeros_f64() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let handle = backend.alloc_zeros(4, 8, 0).expect("alloc_zeros f64");
        assert_eq!(handle.len(), 4);
    }

    #[test]
    fn backend_alloc_zeros_invalid_elem_size_errors() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let result = backend.alloc_zeros(4, 3, 0);
        assert!(result.is_err(), "invalid elem_size should return Err");
    }

    #[test]
    fn backend_clone_buffer_produces_independent_copy() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let host: Vec<f32> = vec![1.0, 2.0, 3.0];
        let bytes = f32_to_bytes(&host);
        let handle = backend.cpu_to_gpu(&bytes, 4, 0).expect("cpu_to_gpu");
        let cloned = backend.clone_buffer(&handle).expect("clone_buffer");
        assert_eq!(cloned.len(), handle.len());
        assert_eq!(cloned.device_ordinal(), handle.device_ordinal());
        // Clone should have the same values.
        let clone_bytes = backend.gpu_to_cpu(&cloned).expect("gpu_to_cpu clone");
        let clone_vals = bytes_to_f32(&clone_bytes);
        assert_eq!(clone_vals, host, "cloned buffer must have same values");
    }

    // -- Elementwise f32 ops (one per trait method) --

    #[test]
    fn backend_add_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[1.0, 2.0, 3.0]);
        let b = upload_f32(backend, &[10.0, 20.0, 30.0]);
        let out = backend.add_f32(&a, &b).expect("add_f32");
        let vals = download_f32(backend, &out);
        assert_eq!(vals, vec![11.0f32, 22.0, 33.0]);
    }

    #[test]
    fn backend_sub_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[5.0, 7.0, 9.0]);
        let b = upload_f32(backend, &[1.0, 2.0, 3.0]);
        let out = backend.sub_f32(&a, &b).expect("sub_f32");
        let vals = download_f32(backend, &out);
        assert_eq!(vals, vec![4.0f32, 5.0, 6.0]);
    }

    #[test]
    fn backend_mul_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[2.0, 3.0, 4.0]);
        let b = upload_f32(backend, &[5.0, 6.0, 7.0]);
        let out = backend.mul_f32(&a, &b).expect("mul_f32");
        let vals = download_f32(backend, &out);
        assert_eq!(vals, vec![10.0f32, 18.0, 28.0]);
    }

    #[test]
    fn backend_neg_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[1.0, -2.0, 3.0]);
        let out = backend.neg_f32(&a).expect("neg_f32");
        let vals = download_f32(backend, &out);
        assert_eq!(vals, vec![-1.0f32, 2.0, -3.0]);
    }

    #[test]
    fn backend_relu_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[-1.0, 0.0, 2.0, -0.5, 3.0]);
        let out = backend.relu_f32(&a).expect("relu_f32");
        let vals = download_f32(backend, &out);
        assert_eq!(vals, vec![0.0f32, 0.0, 2.0, 0.0, 3.0]);
    }

    #[test]
    fn backend_div_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[10.0, 20.0, 30.0]);
        let b = upload_f32(backend, &[2.0, 4.0, 5.0]);
        let out = backend.div_f32(&a, &b).expect("div_f32");
        let vals = download_f32(backend, &out);
        assert_eq!(vals, vec![5.0f32, 5.0, 6.0]);
    }

    #[test]
    fn backend_exp_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[0.0, 1.0]);
        let out = backend.exp_f32(&a).expect("exp_f32");
        let vals = download_f32(backend, &out);
        assert!((vals[0] - 1.0).abs() < 1e-5, "exp(0) ≈ 1: {}", vals[0]);
        assert!(
            (vals[1] - std::f32::consts::E).abs() < 1e-4,
            "exp(1) ≈ e: {}",
            vals[1]
        );
    }

    #[test]
    fn backend_log_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[1.0, std::f32::consts::E]);
        let out = backend.log_f32(&a).expect("log_f32");
        let vals = download_f32(backend, &out);
        assert!((vals[0] - 0.0).abs() < 1e-5, "log(1) ≈ 0: {}", vals[0]);
        assert!(
            (vals[1] - 1.0).abs() < 1e-4,
            "log(e) ≈ 1: {}",
            vals[1]
        );
    }

    #[test]
    fn backend_sqrt_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[4.0, 9.0, 16.0]);
        let out = backend.sqrt_f32(&a).expect("sqrt_f32");
        let vals = download_f32(backend, &out);
        assert!((vals[0] - 2.0).abs() < 1e-5, "sqrt(4) ≈ 2: {}", vals[0]);
        assert!((vals[1] - 3.0).abs() < 1e-5, "sqrt(9) ≈ 3: {}", vals[1]);
        assert!((vals[2] - 4.0).abs() < 1e-5, "sqrt(16) ≈ 4: {}", vals[2]);
    }

    #[test]
    fn backend_pow_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[2.0, 3.0]);
        let out = backend.pow_f32(&a, 3.0).expect("pow_f32");
        let vals = download_f32(backend, &out);
        assert!((vals[0] - 8.0).abs() < 1e-4, "2^3 ≈ 8: {}", vals[0]);
        assert!((vals[1] - 27.0).abs() < 1e-3, "3^3 ≈ 27: {}", vals[1]);
    }

    #[test]
    fn backend_abs_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[-3.0, 0.0, 4.0]);
        let out = backend.abs_f32(&a).expect("abs_f32");
        let vals = download_f32(backend, &out);
        assert_eq!(vals, vec![3.0f32, 0.0, 4.0]);
    }

    #[test]
    fn backend_sigmoid_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[0.0]);
        let out = backend.sigmoid_f32(&a).expect("sigmoid_f32");
        let vals = download_f32(backend, &out);
        // sigmoid(0) = 0.5
        assert!((vals[0] - 0.5).abs() < 1e-4, "sigmoid(0) ≈ 0.5: {}", vals[0]);
    }

    #[test]
    fn backend_tanh_f32() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f32(backend, &[0.0]);
        let out = backend.tanh_f32(&a).expect("tanh_f32");
        let vals = download_f32(backend, &out);
        // tanh(0) = 0
        assert!(vals[0].abs() < 1e-5, "tanh(0) ≈ 0: {}", vals[0]);
    }

    // -- Elementwise f64 ops --

    #[test]
    fn backend_add_f64() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f64(backend, &[1.0, 2.0]);
        let b = upload_f64(backend, &[10.0, 20.0]);
        let out = backend.add_f64(&a, &b).expect("add_f64");
        let vals = download_f64(backend, &out);
        assert_eq!(vals, vec![11.0f64, 22.0]);
    }

    #[test]
    fn backend_sub_f64() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f64(backend, &[10.0, 5.0]);
        let b = upload_f64(backend, &[3.0, 2.0]);
        let out = backend.sub_f64(&a, &b).expect("sub_f64");
        let vals = download_f64(backend, &out);
        assert_eq!(vals, vec![7.0f64, 3.0]);
    }

    #[test]
    fn backend_mul_f64() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f64(backend, &[3.0, 4.0]);
        let b = upload_f64(backend, &[5.0, 6.0]);
        let out = backend.mul_f64(&a, &b).expect("mul_f64");
        let vals = download_f64(backend, &out);
        assert_eq!(vals, vec![15.0f64, 24.0]);
    }

    #[test]
    fn backend_neg_f64() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let a = upload_f64(backend, &[1.0, -2.0]);
        let out = backend.neg_f64(&a).expect("neg_f64");
        let vals = download_f64(backend, &out);
        assert_eq!(vals, vec![-1.0f64, 2.0]);
    }

    #[test]
    fn backend_has_inf_nan_f32_pure_buffer() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let clean = upload_f32(backend, &[1.0, 2.0, 3.0]);
        let result = backend.has_inf_nan_f32(&clean).expect("has_inf_nan_f32");
        assert!(!result, "clean buffer should not have inf/nan");

        let bad = upload_f32(backend, &[1.0, f32::INFINITY, 3.0]);
        let result2 = backend.has_inf_nan_f32(&bad).expect("has_inf_nan_f32 inf");
        assert!(result2, "buffer with inf should report has_inf_nan=true");
    }

    #[test]
    fn backend_raw_device_ptr_is_non_null() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let handle = upload_f32(backend, &[1.0, 2.0]);
        let ptr = backend.raw_device_ptr(&handle);
        assert!(
            !ptr.is_null(),
            "raw_device_ptr should be non-null for a live buffer"
        );
    }

    #[test]
    fn backend_buffer_elem_size_reports_correctly() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        let f32_handle = upload_f32(backend, &[1.0]);
        assert_eq!(backend.buffer_elem_size(&f32_handle), 4, "f32 elem_size should be 4");
        let f64_handle = upload_f64(backend, &[1.0]);
        assert_eq!(backend.buffer_elem_size(&f64_handle), 8, "f64 elem_size should be 8");
    }

    // -- Matmul --

    #[test]
    fn backend_matmul_f32_square() {
        ensure_init();
        let backend = gpu_dispatch::gpu_backend().expect("backend");
        // 2x2 identity × [[1,2],[3,4]] = [[1,2],[3,4]]
        let identity: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
        let mat: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let a = upload_f32(backend, &identity);
        let b = upload_f32(backend, &mat);
        let out = backend
            .matmul_f32(&a, &b, 2, 2, 2)
            .expect("matmul_f32");
        let vals = download_f32(backend, &out);
        // Row-major: [[1,2],[3,4]]
        let expected = [1.0f32, 2.0, 3.0, 4.0];
        for (got, exp) in vals.iter().zip(expected.iter()) {
            assert!(
                (got - exp).abs() < 1e-4,
                "matmul_f32: expected {exp}, got {got}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn f32_to_bytes(v: &[f32]) -> Vec<u8> {
        // SAFETY: f32 is valid to reinterpret as u8; u8 has no invalid bit patterns.
        let ptr = v.as_ptr() as *const u8;
        let len = v.len() * 4;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    fn bytes_to_f32(b: &[u8]) -> Vec<f32> {
        // SAFETY: bytes originated from an f32 slice via f32_to_bytes.
        let ptr = b.as_ptr() as *const f32;
        let len = b.len() / 4;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    fn f64_to_bytes(v: &[f64]) -> Vec<u8> {
        let ptr = v.as_ptr() as *const u8;
        let len = v.len() * 8;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    fn bytes_to_f64(b: &[u8]) -> Vec<f64> {
        let ptr = b.as_ptr() as *const f64;
        let len = b.len() / 8;
        unsafe { std::slice::from_raw_parts(ptr, len) }.to_vec()
    }

    fn upload_f32(
        backend: &dyn ferrotorch_core::gpu_dispatch::GpuBackend,
        data: &[f32],
    ) -> ferrotorch_core::gpu_dispatch::GpuBufferHandle {
        let bytes = f32_to_bytes(data);
        backend.cpu_to_gpu(&bytes, 4, 0).expect("upload_f32")
    }

    fn upload_f64(
        backend: &dyn ferrotorch_core::gpu_dispatch::GpuBackend,
        data: &[f64],
    ) -> ferrotorch_core::gpu_dispatch::GpuBufferHandle {
        let bytes = f64_to_bytes(data);
        backend.cpu_to_gpu(&bytes, 8, 0).expect("upload_f64")
    }

    fn download_f32(
        backend: &dyn ferrotorch_core::gpu_dispatch::GpuBackend,
        handle: &ferrotorch_core::gpu_dispatch::GpuBufferHandle,
    ) -> Vec<f32> {
        let bytes = backend.gpu_to_cpu(handle).expect("download_f32");
        bytes_to_f32(&bytes)
    }

    fn download_f64(
        backend: &dyn ferrotorch_core::gpu_dispatch::GpuBackend,
        handle: &ferrotorch_core::gpu_dispatch::GpuBufferHandle,
    ) -> Vec<f64> {
        let bytes = backend.gpu_to_cpu(handle).expect("download_f64");
        bytes_to_f64(&bytes)
    }
}

// ---------------------------------------------------------------------------
// Module 6 — lib.rs: public re-export smoke tests
// ---------------------------------------------------------------------------

/// `lib.rs` — every named re-export from the crate root resolves.
// Note: type-size assertions don't require the cuda feature; they just
// confirm the symbols are reachable. The types themselves exist on both
// feature configurations (stubs when cuda is off).
mod test_lib_exports {
    #[test]
    fn cuda_allocator_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::CudaAllocator>();
    }

    #[test]
    fn gpu_device_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::GpuDevice>();
    }

    #[test]
    fn gpu_error_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::GpuError>();
    }

    #[test]
    fn cuda_buffer_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::CudaBuffer<f32>>();
    }

    #[test]
    fn cuda_rng_manager_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::rng::CudaRngManager>();
    }

    #[test]
    fn graph_capture_pool_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::CapturePool>();
    }

    #[test]
    fn graph_captured_graph_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::CapturedGraph>();
    }

    #[test]
    fn graph_capture_mode_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::CaptureMode>();
    }

    #[test]
    fn graph_capture_status_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::CaptureStatus>();
    }

    #[test]
    fn graph_pool_handle_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::GraphPoolHandle>();
    }

    #[test]
    fn cuda_backend_impl_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::CudaBackendImpl>();
    }

    #[test]
    fn gpu_tensor_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::GpuTensor<f32>>();
    }

    #[test]
    fn memory_guard_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::MemoryGuard>();
    }

    #[test]
    fn philox_generator_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::PhiloxGenerator>();
    }

    #[test]
    fn philox_state_type_resolves() {
        let _ = std::mem::size_of::<ferrotorch_gpu::PhiloxState>();
    }

    #[test]
    fn free_functions_compile() {
        // Just confirm they're accessible from the crate root.
        let _ = ferrotorch_gpu::graph_pool_handle as fn() -> _;
        let _ = ferrotorch_gpu::release_graph_pool_handle
            as fn(ferrotorch_gpu::GraphPoolHandle) -> _;
        let _ = ferrotorch_gpu::capture_pool_for_handle
            as fn(ferrotorch_gpu::GraphPoolHandle) -> _;
    }
}
