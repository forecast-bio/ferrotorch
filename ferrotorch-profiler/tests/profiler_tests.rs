use ferrotorch_profiler::{with_profiler, ProfileConfig};

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
