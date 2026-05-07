//! Conformance — `ferrotorch_profiler::report` module parity against `torch.profiler`.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/826>.
//!
//! Verifies that [`ProfileReport`] and [`OpSummary`] match the behavioral
//! contract of `torch.profiler.profile.key_averages()` and the Chrome trace
//! export.
//!
//! What is pinned:
//! * `top_ops()` sort order (descending by cumulative duration).
//! * `top_ops()` aggregated counts and total durations.
//! * `chrome_trace_json()` structural format (traceEvents array, X-type events,
//!   required fields: name, cat, ph, ts, dur, pid, tid, args).
//! * `memory_by_category()` aggregation and sort order.
//! * `total_time_us()` sum.
//! * `total_flops()` sum.
//! * `has_gpu_events()` flag.
//! * `has_stack_traces()` flag.
//! * `table()` header row and separator presence (structural, not pixel-perfect).
//! * `save_chrome_trace()` round-trips through the filesystem cleanly.
//! * `save_tensorboard_trace()` creates the expected directory structure and
//!   file content.
//!
//! What is NOT pinned: exact string layout of `table()`, `ts` / `start_us`
//! values (wall-clock dependent), `pid` / `tid` values (OS-determined).
//!
//! Reference: `torch.profiler.profile`, `torch.profiler.ProfilerActivity`.

use ferrotorch_profiler::{MemoryCategory, ProfileConfig, with_profiler};

// ---------------------------------------------------------------------------
// Fixture loader (Layer 2)
// ---------------------------------------------------------------------------

fn fixtures_json() -> serde_json::Value {
    let p = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures.json");
    let bytes = std::fs::read(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. Run scripts/regenerate_profiler_fixtures.py first.",
            p.display()
        )
    });
    serde_json::from_slice(&bytes)
        .unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

// ---------------------------------------------------------------------------
// Helper: build a ProfileReport from a fixture's event list.
// Each event specifies name, category, duration_us.  Memory events also
// specify memory_bytes and memory_category.
// ---------------------------------------------------------------------------

fn build_report_from_fixture(fixture_id: &str) -> ferrotorch_profiler::ProfileReport {
    let all = fixtures_json();
    let fixtures = all["report_fixtures"]
        .as_array()
        .expect("fixtures.json missing report_fixtures");
    let fix = fixtures
        .iter()
        .find(|f| f["id"].as_str() == Some(fixture_id))
        .unwrap_or_else(|| panic!("fixture {fixture_id:?} not found"));

    // Prefer "events" key; fall back to "input_events" for memory-aggregation fixtures.
    let events_arr = fix
        .get("events")
        .or_else(|| fix.get("input_events"))
        .and_then(|v| v.as_array())
        .unwrap_or_else(|| panic!("fixture {fixture_id:?} has no events or input_events"));

    let config = ProfileConfig {
        record_memory: true,
        record_shapes: false,
        with_stack: false,
    };

    let ((), report) = with_profiler(config, |p| {
        for ev in events_arr {
            let name = ev["name"].as_str().expect("event.name");
            let mem_bytes = ev.get("memory_bytes").and_then(|v| v.as_i64());
            let mem_cat_str = ev.get("memory_category").and_then(|v| v.as_str());

            if let Some(bytes) = mem_bytes {
                let cat = match mem_cat_str.unwrap_or("Other") {
                    "Activations" => MemoryCategory::Activations,
                    "Parameters" => MemoryCategory::Parameters,
                    "Gradients" => MemoryCategory::Gradients,
                    "OptimizerState" => MemoryCategory::OptimizerState,
                    _ => MemoryCategory::Other,
                };
                p.record_memory_categorized(name, bytes, cat);
            } else {
                let category = ev
                    .get("category")
                    .and_then(|v| v.as_str())
                    .unwrap_or("tensor_op");
                let duration_us = ev
                    .get("duration_us")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(0);
                p.record_with_duration(name, category, duration_us);
            }
        }
    });

    report
}

// ---------------------------------------------------------------------------
// Test: top_ops() sort order and aggregated stats (fixture-driven)
//
// torch.profiler.key_averages() sorts by self CPU time descending.
// ferrotorch.top_ops() sorts by cumulative duration descending.
// ---------------------------------------------------------------------------

#[test]
fn report_top_ops_sorted_by_cumulative_duration() {
    let all = fixtures_json();
    let fix = all["report_fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["id"].as_str() == Some("report_top_ops_sorted"))
        .expect("report_top_ops_sorted fixture");

    let expected_names: Vec<&str> = fix["expected_top3_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let expected_totals: Vec<u64> = fix["expected_top3_total_us"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap())
        .collect();
    let expected_counts: Vec<usize> = fix["expected_top3_counts"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let report = build_report_from_fixture("report_top_ops_sorted");
    let top3 = report.top_ops(3);

    assert_eq!(
        top3.len(),
        3,
        "top_ops(3) must return exactly 3 summaries"
    );
    for (i, summary) in top3.iter().enumerate() {
        assert_eq!(
            summary.name, expected_names[i],
            "top_ops[{i}].name mismatch: got {:?} expected {:?}",
            summary.name, expected_names[i]
        );
        assert_eq!(
            summary.total_us, expected_totals[i],
            "top_ops[{i}].total_us mismatch: got {} expected {}",
            summary.total_us, expected_totals[i]
        );
        assert_eq!(
            summary.count, expected_counts[i],
            "top_ops[{i}].count mismatch: got {} expected {}",
            summary.count, expected_counts[i]
        );
    }
}

// ---------------------------------------------------------------------------
// Test: top_ops() truncates to N (no over-return)
// ---------------------------------------------------------------------------

#[test]
fn report_top_ops_truncates_to_n() {
    let report = build_report_from_fixture("report_top_ops_sorted");
    // There are 3 distinct op names in the fixture; asking for 1 must
    // return only 1.
    let top1 = report.top_ops(1);
    assert_eq!(top1.len(), 1, "top_ops(1) must return exactly 1 summary");
    // The single result must be the highest-total op (matmul = 1000 µs).
    assert_eq!(top1[0].name, "matmul");
    assert_eq!(top1[0].total_us, 1000);
}

// ---------------------------------------------------------------------------
// Test: top_ops() with zero n returns empty
// ---------------------------------------------------------------------------

#[test]
fn report_top_ops_zero_returns_empty() {
    let report = build_report_from_fixture("report_top_ops_sorted");
    assert!(
        report.top_ops(0).is_empty(),
        "top_ops(0) must return an empty Vec"
    );
}

// ---------------------------------------------------------------------------
// Test: chrome_trace_json() structural format
//
// torch.profiler generates Chrome trace JSON with a "traceEvents" array;
// each event must have: name, cat, ph="X", ts, dur, pid, tid, args.
// ---------------------------------------------------------------------------

#[test]
fn report_chrome_trace_json_structure() {
    let all = fixtures_json();
    let fix = all["report_fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["id"].as_str() == Some("report_chrome_trace_structure"))
        .expect("report_chrome_trace_structure fixture");

    let expected_prefix = fix["expected_json_prefix"].as_str().expect("prefix");
    let expected_suffix = fix["expected_json_suffix"].as_str().expect("suffix");
    let required_fields: Vec<&str> = fix["expected_fields"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();

    let report = build_report_from_fixture("report_chrome_trace_structure");
    let json = report.chrome_trace_json();

    assert!(
        json.starts_with(expected_prefix),
        "chrome_trace_json must start with {expected_prefix:?}; got: {json}"
    );
    assert!(
        json.ends_with(expected_suffix),
        "chrome_trace_json must end with {expected_suffix:?}; got: {json}"
    );
    for field in &required_fields {
        assert!(
            json.contains(&format!("\"{field}\":")),
            "chrome_trace_json must contain field {field:?}; full json: {json}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test: chrome_trace_json() has exactly one event per recorded event
// ---------------------------------------------------------------------------

#[test]
fn report_chrome_trace_event_count_matches_recorded() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("add", "tensor_op", 100);
        p.record_with_duration("relu", "tensor_op", 200);
    });
    let json = report.chrome_trace_json();
    // Each event serializes as {"ph":"X",...}; count "ph" occurrences.
    let ph_count = json.matches("\"ph\"").count();
    assert_eq!(
        ph_count, 2,
        "chrome_trace_json must contain exactly 2 events (one \"ph\" per event); got json: {json}"
    );
}

// ---------------------------------------------------------------------------
// Test: memory_by_category() aggregation — net bytes per category
//
// torch.profiler.key_averages() groups memory events by category; net bytes
// are allocs minus frees.
// ---------------------------------------------------------------------------

#[test]
fn report_memory_by_category_aggregation() {
    let all = fixtures_json();
    let fix = all["report_fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["id"].as_str() == Some("report_memory_by_category_aggregation"))
        .expect("report_memory_by_category_aggregation fixture");

    let expected_cats: Vec<&str> = fix["expected_categories"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap())
        .collect();
    let expected_net: Vec<i64> = fix["expected_net_bytes"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_i64().unwrap())
        .collect();

    let report = build_report_from_fixture("report_memory_by_category_aggregation");
    let by_cat = report.memory_by_category();

    assert_eq!(
        by_cat.len(),
        expected_cats.len(),
        "memory_by_category must return {} entries; got {}",
        expected_cats.len(),
        by_cat.len()
    );

    let cat_to_net: std::collections::HashMap<String, i64> = by_cat
        .into_iter()
        .map(|(cat, net)| (format!("{cat:?}"), net))
        .collect();

    for (cat_str, expected_bytes) in expected_cats.iter().zip(expected_net.iter()) {
        // MemoryCategory::Debug representation; match by variant name substring.
        let found = cat_to_net.iter().find(|(k, _)| k.contains(cat_str));
        assert!(
            found.is_some(),
            "memory_by_category must include category {cat_str:?}; got keys: {:?}",
            cat_to_net.keys().collect::<Vec<_>>()
        );
        let actual_bytes = found.unwrap().1;
        assert_eq!(
            *actual_bytes, *expected_bytes,
            "net bytes for {cat_str} mismatch: got {actual_bytes} expected {expected_bytes}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test: total_time_us() sums all event durations
// ---------------------------------------------------------------------------

#[test]
fn report_total_time_us_sums_all_events() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("a", "tensor_op", 100);
        p.record_with_duration("b", "tensor_op", 250);
        p.record_with_duration("c", "tensor_op", 75);
    });
    assert_eq!(
        report.total_time_us(),
        425,
        "total_time_us must be the sum of all event durations: 100+250+75=425"
    );
}

// ---------------------------------------------------------------------------
// Test: total_time_us() = 0 for empty report
// ---------------------------------------------------------------------------

#[test]
fn report_total_time_us_zero_when_no_events() {
    let ((), report) = with_profiler(ProfileConfig::default(), |_| {});
    assert_eq!(
        report.total_time_us(),
        0,
        "total_time_us must be 0 when no events were recorded"
    );
}

// ---------------------------------------------------------------------------
// Test: total_flops() sums flops from all events with estimates
// ---------------------------------------------------------------------------

#[test]
fn report_total_flops_sums_recognized_ops() {
    // add [3,4]: 12 FLOPs; relu [5,6]: 30 FLOPs; total = 42.
    let config = ProfileConfig {
        record_shapes: true,
        ..ProfileConfig::default()
    };
    let ((), report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3usize, 4], &[3, 4]]);
        p.record("relu", "tensor_op", &[&[5usize, 6]]);
    });
    assert_eq!(
        report.total_flops(),
        42,
        "total_flops must be 12 (add) + 30 (relu) = 42"
    );
}

// ---------------------------------------------------------------------------
// Test: total_flops() ignores events with no estimate (None flops)
// ---------------------------------------------------------------------------

#[test]
fn report_total_flops_skips_none_events() {
    let config = ProfileConfig {
        record_shapes: true,
        ..ProfileConfig::default()
    };
    let ((), report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[2usize, 2], &[2, 2]]); // 4 FLOPs
        p.record("custom_unknown_op_xyz", "tensor_op", &[&[100usize, 100]]); // None
    });
    assert_eq!(
        report.total_flops(),
        4,
        "total_flops must skip events with no FLOPS estimate (None)"
    );
}

// ---------------------------------------------------------------------------
// Test: flops_per_second() = 0 when no time elapsed
// ---------------------------------------------------------------------------

#[test]
fn report_flops_per_second_zero_when_no_events() {
    let ((), report) = with_profiler(ProfileConfig::default(), |_| {});
    assert_eq!(
        report.flops_per_second(),
        0.0,
        "flops_per_second must be 0.0 when there are no events (no elapsed time)"
    );
}

// ---------------------------------------------------------------------------
// Test: has_gpu_events() = false when all events are CPU
// ---------------------------------------------------------------------------

#[test]
fn report_has_gpu_events_false_for_cpu_only() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.record("add", "tensor_op", &[&[3usize, 4]]);
    });
    assert!(
        !report.has_gpu_events(),
        "has_gpu_events must be false when all events are CPU"
    );
}

// ---------------------------------------------------------------------------
// Test: has_gpu_events() = true when a GPU event is present
// ---------------------------------------------------------------------------

#[test]
fn report_has_gpu_events_true_when_gpu_event_pushed() {
    use ferrotorch_profiler::GpuTimingPair;

    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        let timing = GpuTimingPair {
            start_us: 0,
            end_us: 100,
        };
        p.push_gpu_event("conv2d", "cuda_kernel", timing);
    });
    assert!(
        report.has_gpu_events(),
        "has_gpu_events must be true after push_gpu_event"
    );
}

// ---------------------------------------------------------------------------
// Test: has_stack_traces() = false when with_stack = false
// ---------------------------------------------------------------------------

#[test]
fn report_has_stack_traces_false_when_disabled() {
    let config = ProfileConfig {
        with_stack: false,
        ..ProfileConfig::default()
    };
    let ((), report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3usize, 4]]);
    });
    assert!(
        !report.has_stack_traces(),
        "has_stack_traces must be false when with_stack=false"
    );
}

// ---------------------------------------------------------------------------
// Test: has_stack_traces() = true when with_stack = true
// ---------------------------------------------------------------------------

#[test]
fn report_has_stack_traces_true_when_enabled() {
    let config = ProfileConfig {
        with_stack: true,
        ..ProfileConfig::default()
    };
    let ((), report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3usize, 4]]);
    });
    assert!(
        report.has_stack_traces(),
        "has_stack_traces must be true when with_stack=true and RUST_BACKTRACE is enabled"
    );
}

// ---------------------------------------------------------------------------
// Test: table() — non-empty report produces a non-empty, bordered string
// ---------------------------------------------------------------------------

#[test]
fn report_table_nonempty_contains_separator_and_header() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("matmul", "linalg", 500);
        p.record_with_duration("relu", "activation", 100);
    });
    let t = report.table(10);
    // The table must not be the no-events fallback.
    assert_ne!(
        t, "(no events recorded)",
        "table must not be fallback when events exist"
    );
    // Must contain table borders (+---+).
    assert!(
        t.contains("+-"),
        "table must contain border separators (+-); got:\n{t}"
    );
    // Must contain the Op column header.
    assert!(
        t.contains("Op"),
        "table must contain 'Op' column header; got:\n{t}"
    );
    // Both recorded op names must appear.
    assert!(t.contains("matmul"), "table must contain 'matmul'; got:\n{t}");
    assert!(t.contains("relu"), "table must contain 'relu'; got:\n{t}");
}

// ---------------------------------------------------------------------------
// Test: table() — empty report returns fallback string
// ---------------------------------------------------------------------------

#[test]
fn report_table_empty_report_fallback() {
    let ((), report) = with_profiler(ProfileConfig::default(), |_| {});
    assert_eq!(
        report.table(10),
        "(no events recorded)",
        "table must return fallback string when no events were recorded"
    );
}

// ---------------------------------------------------------------------------
// Test: save_chrome_trace() round-trips JSON through the filesystem
// ---------------------------------------------------------------------------

#[test]
fn report_save_chrome_trace_roundtrip() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("trace.json");

    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("mm", "tensor_op", 500);
    });
    report
        .save_chrome_trace(&path)
        .expect("save_chrome_trace must not fail on a writable path");

    let content = std::fs::read_to_string(&path).expect("read back trace file");
    assert!(
        content.starts_with("{\"traceEvents\":["),
        "saved Chrome trace must start with {{\"traceEvents\":["
    );
    assert!(
        content.ends_with("]}"),
        "saved Chrome trace must end with ]}}: got\n{content}"
    );
}

// ---------------------------------------------------------------------------
// Test: save_tensorboard_trace() creates the expected directory layout
// ---------------------------------------------------------------------------

#[test]
fn report_save_tensorboard_trace_directory_layout() {
    let dir = tempfile::tempdir().expect("tempdir");
    let logdir = dir.path().join("tb_logs");

    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("relu", "tensor_op", 200);
    });
    let written_path = report
        .save_tensorboard_trace(&logdir, Some("run0"), Some("test-host"))
        .expect("save_tensorboard_trace must not fail on a writable path");

    // Must exist and be a file.
    assert!(
        written_path.exists(),
        "save_tensorboard_trace must create the trace file"
    );

    // Path must follow: {logdir}/plugins/profile/{run_id}/{hostname}.pt.trace.json
    let expected = logdir
        .join("plugins")
        .join("profile")
        .join("run0")
        .join("test-host.pt.trace.json");
    assert_eq!(
        written_path, expected,
        "TensorBoard trace path must follow the plugins/profile/{{run_id}}/{{hostname}}.pt.trace.json convention"
    );

    // File must contain valid Chrome trace JSON.
    let content = std::fs::read_to_string(&written_path).expect("read trace");
    assert!(
        content.starts_with("{\"traceEvents\":["),
        "TensorBoard trace file must be Chrome trace JSON"
    );
}

// ---------------------------------------------------------------------------
// Test: OpSummary fields accessible (structural)
// ---------------------------------------------------------------------------

#[test]
fn report_op_summary_fields_accessible() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.record_with_duration("add", "tensor_op", 300);
        p.record_with_duration("add", "tensor_op", 100);
    });
    let ops = report.top_ops(1);
    assert_eq!(ops.len(), 1);
    let s = &ops[0];
    // Structural: all public fields must be accessible without panic.
    assert_eq!(s.name, "add");
    assert_eq!(s.count, 2);
    assert_eq!(s.total_us, 400);
    assert_eq!(s.cpu_total_us, 400);
    assert_eq!(s.cpu_count, 2);
    assert_eq!(s.cpu_avg_us, 200);
    assert_eq!(s.cpu_max_us, 300);
    assert_eq!(s.gpu_total_us, 0);
    assert_eq!(s.gpu_count, 0);
    assert_eq!(s.gpu_avg_us, 0);
    assert_eq!(s.avg_us, 200);
    assert_eq!(s.max_us, 300);
}
