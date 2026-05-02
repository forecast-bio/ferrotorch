use ferrotorch_profiler::{ProfileConfig, with_profiler};

// ---------------------------------------------------------------------------
// Record events, verify count
// ---------------------------------------------------------------------------

#[test]
fn record_events_count() {
    let config = ProfileConfig::default();
    let (_result, report) = with_profiler(config, |p| {
        p.record("matmul", "tensor_op", &[&[32, 784], &[784, 256]]);
        p.record("relu", "tensor_op", &[&[32, 256]]);
        p.record("matmul", "tensor_op", &[&[32, 256], &[256, 10]]);
    });
    assert_eq!(report.events().len(), 3);
}

#[test]
fn record_with_duration_events_count() {
    let config = ProfileConfig::default();
    let (_, report) = with_profiler(config, |p| {
        p.record_with_duration("conv2d", "tensor_op", 1200);
        p.record_with_duration("batchnorm", "tensor_op", 300);
    });
    assert_eq!(report.events().len(), 2);
    // Durations should be preserved.
    assert_eq!(report.events()[0].duration_us, 1200);
    assert_eq!(report.events()[1].duration_us, 300);
}

// ---------------------------------------------------------------------------
// table() output format
// ---------------------------------------------------------------------------

#[test]
fn table_output_format() {
    let config = ProfileConfig::default();
    let (_, report) = with_profiler(config, |p| {
        p.record_with_duration("matmul", "tensor_op", 500);
        p.record_with_duration("relu", "tensor_op", 100);
    });
    let table = report.table(10);
    // Should contain header columns.
    assert!(table.contains("Op"));
    assert!(table.contains("Count"));
    assert!(table.contains("Total us"));
    assert!(table.contains("Avg us"));
    assert!(table.contains("Max us"));
    // Should contain the operation names.
    assert!(table.contains("matmul"));
    assert!(table.contains("relu"));
    // Should contain table borders.
    assert!(table.contains("+--"));
    assert!(table.contains("--+"));
}

#[test]
fn table_empty_report() {
    let config = ProfileConfig::default();
    let (_, report) = with_profiler(config, |_p| {});
    let table = report.table(10);
    assert_eq!(table, "(no events recorded)");
}

// ---------------------------------------------------------------------------
// chrome_trace_json() is valid JSON
// ---------------------------------------------------------------------------

#[test]
fn chrome_trace_json_valid() {
    let config = ProfileConfig::default();
    let (_, report) = with_profiler(config, |p| {
        p.record_with_duration("mm", "tensor_op", 500);
        p.record_with_duration("relu", "tensor_op", 100);
    });
    let json = report.chrome_trace_json();
    // Basic structural validation.
    assert!(json.starts_with("{\"traceEvents\":["));
    assert!(json.ends_with("]}"));
    // Contains expected fields.
    assert!(json.contains("\"name\":\"mm\""));
    assert!(json.contains("\"cat\":\"tensor_op\""));
    assert!(json.contains("\"ph\":\"X\""));
    assert!(json.contains("\"dur\":500"));
    assert!(json.contains("\"dur\":100"));
    assert!(json.contains("\"pid\":1"));
    assert!(json.contains("\"shapes\""));
}

#[test]
fn chrome_trace_json_empty() {
    let config = ProfileConfig::default();
    let (_, report) = with_profiler(config, |_p| {});
    let json = report.chrome_trace_json();
    assert_eq!(json, "{\"traceEvents\":[]}");
}

#[test]
fn chrome_trace_json_shapes_recorded() {
    let config = ProfileConfig {
        record_shapes: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record("mm", "tensor_op", &[&[32, 784], &[784, 256]]);
    });
    let json = report.chrome_trace_json();
    assert!(json.contains("[[32,784],[784,256]]"));
}

// ---------------------------------------------------------------------------
// ProfileConfig defaults
// ---------------------------------------------------------------------------

#[test]
fn profile_config_defaults() {
    let config = ProfileConfig::default();
    assert!(config.record_shapes);
    assert!(!config.record_memory);
    assert!(!config.with_stack);
}

// ---------------------------------------------------------------------------
// with_profiler captures events and returns closure result
// ---------------------------------------------------------------------------

#[test]
fn with_profiler_captures_events() {
    let config = ProfileConfig::default();
    let (result, report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[4, 4]]);
        p.record("mul", "tensor_op", &[&[4, 4]]);
        42
    });
    assert_eq!(result, 42);
    assert_eq!(report.events().len(), 2);
    assert_eq!(report.events()[0].name, "add");
    assert_eq!(report.events()[1].name, "mul");
}

#[test]
fn with_profiler_stops_after_closure() {
    let config = ProfileConfig::default();
    let (_, report) = with_profiler(config, |p| {
        p.record("a", "tensor_op", &[]);
        // The profiler is stopped after the closure returns, so all events
        // recorded inside should be present.
    });
    assert_eq!(report.events().len(), 1);
}

// ---------------------------------------------------------------------------
// top_ops sorting
// ---------------------------------------------------------------------------

#[test]
fn top_ops_sorted_by_total_time() {
    let config = ProfileConfig::default();
    let (_, report) = with_profiler(config, |p| {
        // relu: 3 calls totalling 300us
        p.record_with_duration("relu", "tensor_op", 100);
        p.record_with_duration("relu", "tensor_op", 100);
        p.record_with_duration("relu", "tensor_op", 100);
        // matmul: 2 calls totalling 1000us
        p.record_with_duration("matmul", "tensor_op", 500);
        p.record_with_duration("matmul", "tensor_op", 500);
        // conv2d: 1 call totalling 600us
        p.record_with_duration("conv2d", "tensor_op", 600);
    });

    let top = report.top_ops(10);
    assert_eq!(top.len(), 3);
    // matmul (1000) > conv2d (600) > relu (300)
    assert_eq!(top[0].name, "matmul");
    assert_eq!(top[0].total_us, 1000);
    assert_eq!(top[0].count, 2);
    assert_eq!(top[0].avg_us, 500);
    assert_eq!(top[0].max_us, 500);

    assert_eq!(top[1].name, "conv2d");
    assert_eq!(top[1].total_us, 600);
    assert_eq!(top[1].count, 1);

    assert_eq!(top[2].name, "relu");
    assert_eq!(top[2].total_us, 300);
    assert_eq!(top[2].count, 3);
    assert_eq!(top[2].avg_us, 100);
    assert_eq!(top[2].max_us, 100);
}

#[test]
fn top_ops_truncated() {
    let config = ProfileConfig::default();
    let (_, report) = with_profiler(config, |p| {
        p.record_with_duration("a", "tensor_op", 100);
        p.record_with_duration("b", "tensor_op", 200);
        p.record_with_duration("c", "tensor_op", 300);
        p.record_with_duration("d", "tensor_op", 400);
    });
    let top = report.top_ops(2);
    assert_eq!(top.len(), 2);
    assert_eq!(top[0].name, "d");
    assert_eq!(top[1].name, "c");
}

#[test]
fn total_time_us_sums_durations() {
    let config = ProfileConfig::default();
    let (_, report) = with_profiler(config, |p| {
        p.record_with_duration("a", "tensor_op", 100);
        p.record_with_duration("b", "tensor_op", 250);
    });
    assert_eq!(report.total_time_us(), 350);
}

// ---------------------------------------------------------------------------
// Memory recording
// ---------------------------------------------------------------------------

#[test]
fn record_memory_when_enabled() {
    let config = ProfileConfig {
        record_memory: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record_memory("alloc_tensor", 4096);
        p.record_memory("free_tensor", -4096);
    });
    assert_eq!(report.events().len(), 2);
    assert_eq!(report.events()[0].memory_bytes, Some(4096));
    assert_eq!(report.events()[1].memory_bytes, Some(-4096));
    assert_eq!(report.events()[0].category, "memory");
}

#[test]
fn record_memory_ignored_when_disabled() {
    let config = ProfileConfig {
        record_memory: false,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record_memory("alloc_tensor", 4096);
    });
    assert_eq!(report.events().len(), 0);
}

// ---------------------------------------------------------------------------
// Shapes recording respect config
// ---------------------------------------------------------------------------

#[test]
fn shapes_not_recorded_when_disabled() {
    let config = ProfileConfig {
        record_shapes: false,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record("mm", "tensor_op", &[&[32, 784], &[784, 256]]);
    });
    assert_eq!(report.events().len(), 1);
    assert!(report.events()[0].input_shapes.is_empty());
}

// ---------------------------------------------------------------------------
// Auto-instrumentation: tensor ops invoked from inside with_profiler are
// recorded automatically via the ferrotorch_core::profiler_hook thread-local.
// CL-379.
// ---------------------------------------------------------------------------

#[test]
fn auto_instrumentation_records_add() {
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;

    let a: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3], false).unwrap();
    let b: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![10.0, 20.0, 30.0]), vec![3], false).unwrap();

    let (_result, report) = with_profiler(ProfileConfig::default(), |_p| {
        // Note: we don't call _p.record() at all -- the tensor op below
        // should record itself via the profiler_hook.
        ferrotorch_core::grad_fns::arithmetic::add(&a, &b).unwrap()
    });

    let events = report.events();
    assert!(
        events.iter().any(|e| e.name == "add"),
        "expected at least one auto-recorded 'add' event, got: {:?}",
        events.iter().map(|e| &e.name).collect::<Vec<_>>()
    );
    let add_event = events.iter().find(|e| e.name == "add").unwrap();
    assert_eq!(add_event.category, "tensor_op");
    assert_eq!(add_event.input_shapes, vec![vec![3], vec![3]]);
}

#[test]
fn auto_instrumentation_records_matmul_and_sum() {
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;

    let a: Tensor<f32> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        vec![2, 3],
        false,
    )
    .unwrap();
    let b: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0; 12]), vec![3, 4], false).unwrap();

    let (_, report) = with_profiler(ProfileConfig::default(), |_| {
        let c = ferrotorch_core::ops::linalg::matmul(&a, &b).unwrap();
        let _s = ferrotorch_core::grad_fns::reduction::sum(&c).unwrap();
    });

    let events = report.events();
    let names: Vec<&str> = events.iter().map(|e| e.name.as_str()).collect();
    assert!(
        names.contains(&"matmul"),
        "expected matmul in events, got {:?}",
        names
    );
    assert!(
        names.contains(&"sum"),
        "expected sum in events, got {:?}",
        names
    );

    let matmul_ev = events.iter().find(|e| e.name == "matmul").unwrap();
    assert_eq!(matmul_ev.category, "linalg");
    assert_eq!(matmul_ev.input_shapes, vec![vec![2, 3], vec![3, 4]]);
}

#[test]
fn auto_instrumentation_no_profiler_outside_with_profiler() {
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;

    // First call: outside any profiler -- should be a complete no-op.
    let a: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0, 2.0]), vec![2], false).unwrap();
    let b: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![3.0, 4.0]), vec![2], false).unwrap();
    let _ = ferrotorch_core::grad_fns::arithmetic::add(&a, &b).unwrap();

    // Second call: inside a profiler -- this one should be recorded.
    let (_, report) = with_profiler(ProfileConfig::default(), |_| {
        let _ = ferrotorch_core::grad_fns::arithmetic::add(&a, &b).unwrap();
    });

    // Only the in-profile call should appear.
    let add_count = report.events().iter().filter(|e| e.name == "add").count();
    assert_eq!(
        add_count, 1,
        "expected exactly 1 'add' event, got {add_count}"
    );
}

#[test]
fn auto_instrumentation_hook_cleared_after_with_profiler() {
    use ferrotorch_core::profiler_hook;

    // Before with_profiler, no hook.
    assert!(profiler_hook::current().is_none());

    let (_, _report) = with_profiler(ProfileConfig::default(), |_| {
        // Inside, the hook is set.
        assert!(profiler_hook::current().is_some());
    });

    // After with_profiler returns (even if the closure panicked), the
    // hook is cleared via the RAII guard.
    assert!(
        profiler_hook::current().is_none(),
        "profiler_hook should be cleared after with_profiler"
    );
}

#[test]
fn auto_instrumentation_records_multiple_ops_in_one_session() {
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;

    let x: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3], false).unwrap();
    let y: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![4.0, 5.0, 6.0]), vec![3], false).unwrap();

    let (_, report) = with_profiler(ProfileConfig::default(), |_| {
        let s = ferrotorch_core::grad_fns::arithmetic::add(&x, &y).unwrap();
        let p = ferrotorch_core::grad_fns::arithmetic::mul(&s, &y).unwrap();
        let d = ferrotorch_core::grad_fns::arithmetic::sub(&p, &x).unwrap();
        let _ = ferrotorch_core::grad_fns::reduction::sum(&d).unwrap();
    });

    let events = report.events();
    let names: Vec<&str> = events.iter().map(|e| e.name.as_str()).collect();
    // All four ops should appear, in execution order.
    assert!(names.contains(&"add"));
    assert!(names.contains(&"mul"));
    assert!(names.contains(&"sub"));
    assert!(names.contains(&"sum"));
    assert!(events.len() >= 4);
}

// ---------------------------------------------------------------------------
// Memory categories, FLOPS estimation, stack traces. CL-333.
// ---------------------------------------------------------------------------

#[test]
fn memory_category_recorded_when_specified() {
    use ferrotorch_profiler::MemoryCategory;
    let config = ProfileConfig {
        record_memory: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record_memory_categorized("alloc_weight", 4096, MemoryCategory::Parameters);
        p.record_memory_categorized("alloc_grad", 4096, MemoryCategory::Gradients);
        p.record_memory_categorized("alloc_act", 8192, MemoryCategory::Activations);
        p.record_memory_categorized("alloc_optim", 8192, MemoryCategory::OptimizerState);
    });
    let events = report.events();
    assert_eq!(events.len(), 4);
    assert_eq!(events[0].memory_category, Some(MemoryCategory::Parameters));
    assert_eq!(events[1].memory_category, Some(MemoryCategory::Gradients));
    assert_eq!(events[2].memory_category, Some(MemoryCategory::Activations));
    assert_eq!(
        events[3].memory_category,
        Some(MemoryCategory::OptimizerState)
    );
}

#[test]
fn memory_by_category_aggregates_totals() {
    use ferrotorch_profiler::MemoryCategory;
    let config = ProfileConfig {
        record_memory: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        // Two parameter allocations: total +6000
        p.record_memory_categorized("w1", 1000, MemoryCategory::Parameters);
        p.record_memory_categorized("w2", 5000, MemoryCategory::Parameters);
        // One activation allocation
        p.record_memory_categorized("a1", 12000, MemoryCategory::Activations);
        // A free as a negative
        p.record_memory_categorized("a1_free", -2000, MemoryCategory::Activations);
    });
    let by_cat = report.memory_by_category();
    // Sorted by absolute byte size: Activations 10000, Parameters 6000.
    assert_eq!(by_cat.len(), 2);
    assert_eq!(by_cat[0].0, MemoryCategory::Activations);
    assert_eq!(by_cat[0].1, 10000);
    assert_eq!(by_cat[1].0, MemoryCategory::Parameters);
    assert_eq!(by_cat[1].1, 6000);
}

#[test]
fn record_memory_default_uses_other_category() {
    use ferrotorch_profiler::MemoryCategory;
    let config = ProfileConfig {
        record_memory: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record_memory("alloc", 4096);
    });
    let events = report.events();
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].memory_category, Some(MemoryCategory::Other));
}

#[test]
fn flops_estimated_for_recorded_ops() {
    let config = ProfileConfig {
        record_shapes: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3, 4], &[3, 4]]);
        p.record("matmul", "linalg", &[&[4, 5], &[5, 6]]);
        p.record("relu", "activation", &[&[10, 10]]);
    });
    let events = report.events();
    assert_eq!(events[0].flops, Some(12), "add[3,4]+[3,4] = 12 FLOPS");
    assert_eq!(events[1].flops, Some(240), "matmul[4,5]@[5,6] = 240 FLOPS");
    assert_eq!(events[2].flops, Some(100), "relu[10,10] = 100 FLOPS");
}

#[test]
fn flops_none_when_record_shapes_off() {
    let config = ProfileConfig {
        record_shapes: false,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3, 4], &[3, 4]]);
    });
    assert!(report.events()[0].flops.is_none());
}

#[test]
fn total_flops_sums_event_flops() {
    let config = ProfileConfig {
        record_shapes: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3, 4], &[3, 4]]); // 12
        p.record("matmul", "linalg", &[&[4, 5], &[5, 6]]); // 240
    });
    assert_eq!(report.total_flops(), 252);
}

#[test]
fn flops_per_second_zero_when_no_time() {
    // total_time_us is the SUM of event durations, which is 0 for
    // record() events (they have duration 0).
    let config = ProfileConfig {
        record_shapes: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3, 4], &[3, 4]]);
    });
    // Event duration is 0, so flops_per_second returns 0.
    assert_eq!(report.flops_per_second(), 0.0);
}

#[test]
fn flops_per_second_nonzero_with_durations() {
    let config = ProfileConfig {
        record_shapes: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        // Use record() to get the FLOPS estimate, but also use
        // record_with_duration to get a non-zero total_time. Since
        // record_with_duration doesn't take shapes, we can't get
        // flops on it -- but we can manually verify the math.
        p.record_with_duration("add", "tensor_op", 1000); // 1ms
    });
    // FLOPS = 0 (record_with_duration doesn't estimate), so per/sec = 0.
    assert_eq!(report.flops_per_second(), 0.0);
}

#[test]
fn auto_instrumented_op_gets_flops_estimate() {
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;

    let a: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![1.0; 12]), vec![3, 4], false).unwrap();
    let b: Tensor<f32> =
        Tensor::from_storage(TensorStorage::cpu(vec![2.0; 12]), vec![3, 4], false).unwrap();

    let config = ProfileConfig {
        record_shapes: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |_| {
        let _ = ferrotorch_core::grad_fns::arithmetic::add(&a, &b).unwrap();
    });
    let add_event = report.events().iter().find(|e| e.name == "add").unwrap();
    assert_eq!(add_event.flops, Some(12));
}

#[test]
fn stack_trace_captured_when_with_stack_enabled() {
    let config = ProfileConfig {
        with_stack: true,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3, 4]]);
    });
    let event = &report.events()[0];
    assert!(
        event.stack_trace.is_some(),
        "expected stack_trace when with_stack=true"
    );
    assert!(report.has_stack_traces());
}

#[test]
fn stack_trace_none_when_with_stack_disabled() {
    let config = ProfileConfig {
        with_stack: false,
        ..ProfileConfig::default()
    };
    let (_, report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3, 4]]);
    });
    assert!(report.events()[0].stack_trace.is_none());
    assert!(!report.has_stack_traces());
}

#[test]
fn memory_by_category_empty_when_no_memory_events() {
    let (_, report) = with_profiler(ProfileConfig::default(), |p| {
        p.record("add", "tensor_op", &[&[3, 4]]);
    });
    assert_eq!(report.memory_by_category().len(), 0);
}

// ---------------------------------------------------------------------------
// TensorBoard export. CL-381.
// ---------------------------------------------------------------------------

#[test]
fn tensorboard_export_creates_expected_layout() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, report) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("matmul", "linalg", 500);
        p.record_with_duration("relu", "activation", 50);
    });
    let path = report
        .save_tensorboard_trace(tmp.path(), Some("run0"), Some("testhost"))
        .unwrap();

    // Expected path: {logdir}/plugins/profile/run0/testhost.pt.trace.json
    let expected = tmp
        .path()
        .join("plugins")
        .join("profile")
        .join("run0")
        .join("testhost.pt.trace.json");
    assert_eq!(path, expected);
    assert!(
        expected.exists(),
        "expected trace file to exist at {expected:?}"
    );

    // Contents should be valid Chrome trace JSON (starts with {"traceEvents":[)
    let contents = std::fs::read_to_string(&expected).unwrap();
    assert!(
        contents.contains("traceEvents"),
        "expected Chrome trace JSON, got: {contents}"
    );
    assert!(contents.contains("matmul"));
    assert!(contents.contains("relu"));
}

#[test]
fn tensorboard_export_default_run_id() {
    let tmp = tempfile::tempdir().unwrap();
    let (_, report) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("add", "tensor_op", 10);
    });
    let path = report
        .save_tensorboard_trace(tmp.path(), None, Some("host"))
        .unwrap();
    // None run_id defaults to "run0"
    let expected = tmp
        .path()
        .join("plugins")
        .join("profile")
        .join("run0")
        .join("host.pt.trace.json");
    assert_eq!(path, expected);
}

#[test]
fn tensorboard_export_creates_nested_directories() {
    // logdir doesn't exist yet; the export should create all three
    // levels: logdir, logdir/plugins, logdir/plugins/profile/run_id.
    let tmp = tempfile::tempdir().unwrap();
    let nonexistent = tmp.path().join("new").join("deeply").join("nested");
    assert!(!nonexistent.exists());

    let (_, report) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("op", "test", 1);
    });
    let result = report
        .save_tensorboard_trace(&nonexistent, Some("myrun"), Some("h1"))
        .unwrap();
    assert!(result.exists());
    assert!(
        nonexistent
            .join("plugins")
            .join("profile")
            .join("myrun")
            .exists()
    );
}

#[test]
fn tensorboard_export_multiple_runs_do_not_collide() {
    let tmp = tempfile::tempdir().unwrap();

    // Export two separate runs with different run_ids.
    let (_, report1) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("forward_pass_1", "training", 100);
    });
    let p1 = report1
        .save_tensorboard_trace(tmp.path(), Some("epoch1"), Some("host"))
        .unwrap();

    let (_, report2) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("forward_pass_2", "training", 200);
    });
    let p2 = report2
        .save_tensorboard_trace(tmp.path(), Some("epoch2"), Some("host"))
        .unwrap();

    assert_ne!(p1, p2);
    assert!(p1.exists());
    assert!(p2.exists());

    // Each file should contain only its own events.
    let c1 = std::fs::read_to_string(&p1).unwrap();
    let c2 = std::fs::read_to_string(&p2).unwrap();
    assert!(c1.contains("forward_pass_1"));
    assert!(!c1.contains("forward_pass_2"));
    assert!(c2.contains("forward_pass_2"));
    assert!(!c2.contains("forward_pass_1"));
}

#[test]
fn tensorboard_export_multiple_hosts_same_run() {
    // Multi-node / multi-GPU runs write one file per host into the
    // same run directory.
    let tmp = tempfile::tempdir().unwrap();

    let (_, report_a) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("rank0_op", "distributed", 50);
    });
    let (_, report_b) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("rank1_op", "distributed", 60);
    });

    let p_a = report_a
        .save_tensorboard_trace(tmp.path(), Some("ddp_run"), Some("rank0"))
        .unwrap();
    let p_b = report_b
        .save_tensorboard_trace(tmp.path(), Some("ddp_run"), Some("rank1"))
        .unwrap();

    // Both in the same run directory, different filenames.
    assert_eq!(p_a.parent(), p_b.parent());
    assert_eq!(
        p_a.file_name().unwrap().to_string_lossy(),
        "rank0.pt.trace.json"
    );
    assert_eq!(
        p_b.file_name().unwrap().to_string_lossy(),
        "rank1.pt.trace.json"
    );
}
