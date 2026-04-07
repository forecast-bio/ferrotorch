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

    let a: Tensor<f32> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0, 2.0, 3.0]),
        vec![3],
        false,
    )
    .unwrap();
    let b: Tensor<f32> = Tensor::from_storage(
        TensorStorage::cpu(vec![10.0, 20.0, 30.0]),
        vec![3],
        false,
    )
    .unwrap();

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
    let b: Tensor<f32> = Tensor::from_storage(
        TensorStorage::cpu(vec![1.0; 12]),
        vec![3, 4],
        false,
    )
    .unwrap();

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
    assert!(names.contains(&"sum"), "expected sum in events, got {:?}", names);

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
    assert_eq!(add_count, 1, "expected exactly 1 'add' event, got {add_count}");
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
