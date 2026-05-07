//! Conformance — `ferrotorch_profiler::event` module parity against `torch.profiler`.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/826>.
//!
//! Verifies that [`ProfileEvent`] and [`MemoryCategory`] match the structural
//! contract of `torch.profiler`'s event objects.  Absolute timing fields
//! (`start_us`, `duration_us`) are intentionally NOT pinned — they are
//! wall-clock-dependent and non-deterministic.  What is pinned:
//!
//! * Field presence and types (structural).
//! * `device_type` for CPU vs GPU events.
//! * `memory_bytes` sign convention (+ = alloc, − = free).
//! * `memory_category` values.
//! * `flops` estimation from shapes.
//! * `stack_trace` presence/absence controlled by `with_stack` flag.
//!
//! Reference: `torch.profiler.profile`, `torch.profiler.ProfilerActivity`,
//! `torch.profiler.record_function`.

use ferrotorch_profiler::{MemoryCategory, ProfileConfig, ProfileEvent, with_profiler};

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

fn event_fixtures(all: &serde_json::Value) -> &Vec<serde_json::Value> {
    all["event_fixtures"]
        .as_array()
        .expect("fixtures.json missing event_fixtures array")
}

// ---------------------------------------------------------------------------
// Helper: build a profile event via with_profiler for a given fixture id.
// ---------------------------------------------------------------------------

fn run_and_get_events(fixture_id: &str, all: &serde_json::Value) -> Vec<ProfileEvent> {
    let fixtures = event_fixtures(all);
    let fix = fixtures
        .iter()
        .find(|f| f["id"].as_str() == Some(fixture_id))
        .unwrap_or_else(|| panic!("fixture {fixture_id:?} not found"));

    let op = fix["op"].as_str().expect("op");
    let category = fix["category"].as_str().unwrap_or("tensor_op");
    let memory_bytes = fix["memory_bytes"].as_i64();
    let memory_category_str = fix["memory_category"].as_str();

    let config = ProfileConfig {
        record_shapes: true,
        record_memory: memory_bytes.is_some(),
        ..ProfileConfig::default()
    };

    let ((), report) = with_profiler(config, |p| {
        if let Some(bytes) = memory_bytes {
            let cat = match memory_category_str.unwrap_or("Other") {
                "Parameters" => MemoryCategory::Parameters,
                "Activations" => MemoryCategory::Activations,
                "Gradients" => MemoryCategory::Gradients,
                "OptimizerState" => MemoryCategory::OptimizerState,
                _ => MemoryCategory::Other,
            };
            if memory_category_str.is_some_and(|s| s != "Other") {
                p.record_memory_categorized(op, bytes, cat);
            } else {
                p.record_memory(op, bytes);
            }
        } else {
            let shapes: Vec<Vec<usize>> = fix["shapes"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .map(|s| {
                            s.as_array()
                                .unwrap()
                                .iter()
                                .map(|d| d.as_u64().unwrap() as usize)
                                .collect()
                        })
                        .collect()
                })
                .unwrap_or_default();
            let shape_refs: Vec<&[usize]> = shapes.iter().map(Vec::as_slice).collect();
            p.record(op, category, &shape_refs);
        }
    });

    report.events().to_vec()
}

// ---------------------------------------------------------------------------
// Test: CPU op event has device_type = CPU
// ---------------------------------------------------------------------------

#[test]
fn event_cpu_op_device_type_is_cpu() {
    use ferrotorch_profiler::DeviceType;

    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.record("add", "tensor_op", &[&[3usize, 4], &[3, 4]]);
    });
    let events = report.events();
    assert_eq!(events.len(), 1, "expected 1 event");
    assert_eq!(
        events[0].device_type,
        DeviceType::Cpu,
        "CPU op must have device_type = Cpu"
    );
}

// ---------------------------------------------------------------------------
// Test: event name and category are preserved
// ---------------------------------------------------------------------------

#[test]
fn event_name_and_category_preserved() {
    let all = fixtures_json();
    let events = run_and_get_events("event_cpu_op_record", &all);
    assert!(!events.is_empty(), "expected at least 1 event");
    let ev = &events[0];
    assert_eq!(ev.name, "add", "event name mismatch");
    assert_eq!(ev.category, "tensor_op", "event category mismatch");
}

// ---------------------------------------------------------------------------
// Test: flops estimated from shapes matches fixture
// ---------------------------------------------------------------------------

#[test]
fn event_flops_estimated_from_shapes() {
    let all = fixtures_json();
    let fix = event_fixtures(&all)
        .iter()
        .find(|f| f["id"].as_str() == Some("event_cpu_op_record"))
        .expect("event_cpu_op_record fixture");
    let expected_flops = fix["expected_flops"].as_u64();

    let events = run_and_get_events("event_cpu_op_record", &all);
    assert!(!events.is_empty());
    assert_eq!(
        events[0].flops, expected_flops,
        "flops mismatch: got {:?}, expected {:?}",
        events[0].flops, expected_flops
    );
}

// ---------------------------------------------------------------------------
// Test: memory event — positive bytes = allocation
// ---------------------------------------------------------------------------

#[test]
fn event_memory_alloc_positive_bytes() {
    let all = fixtures_json();
    let events = run_and_get_events("event_memory_alloc", &all);
    assert_eq!(events.len(), 1, "expected 1 memory event");
    let ev = &events[0];
    assert_eq!(ev.category, "memory", "memory events must have category = memory");
    assert_eq!(
        ev.memory_bytes,
        Some(4096),
        "alloc event must have positive memory_bytes"
    );
    assert_eq!(
        ev.memory_category,
        Some(MemoryCategory::Other),
        "default memory category is Other"
    );
}

// ---------------------------------------------------------------------------
// Test: memory event — negative bytes = free
// ---------------------------------------------------------------------------

#[test]
fn event_memory_free_negative_bytes() {
    let all = fixtures_json();
    let events = run_and_get_events("event_memory_free", &all);
    assert_eq!(events.len(), 1, "expected 1 memory free event");
    assert_eq!(
        events[0].memory_bytes,
        Some(-4096),
        "free event must have negative memory_bytes"
    );
}

// ---------------------------------------------------------------------------
// Test: categorized memory events — all five categories
// ---------------------------------------------------------------------------

#[test]
fn event_memory_categorized_parameters() {
    let all = fixtures_json();
    let events = run_and_get_events("event_memory_categorized_parameters", &all);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].memory_category, Some(MemoryCategory::Parameters));
}

#[test]
fn event_memory_categorized_activations() {
    let all = fixtures_json();
    let events = run_and_get_events("event_memory_categorized_activations", &all);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].memory_category, Some(MemoryCategory::Activations));
}

#[test]
fn event_memory_categorized_gradients() {
    let all = fixtures_json();
    let events = run_and_get_events("event_memory_categorized_gradients", &all);
    assert_eq!(events.len(), 1);
    assert_eq!(events[0].memory_category, Some(MemoryCategory::Gradients));
}

#[test]
fn event_memory_categorized_optimizer_state() {
    let all = fixtures_json();
    let events = run_and_get_events("event_memory_categorized_optimizer_state", &all);
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].memory_category,
        Some(MemoryCategory::OptimizerState)
    );
}

// ---------------------------------------------------------------------------
// Test: non-memory events have memory_bytes = None
// ---------------------------------------------------------------------------

#[test]
fn event_non_memory_has_no_memory_bytes() {
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.record("matmul", "linalg", &[&[4usize, 5], &[5, 6]]);
    });
    let events = report.events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].memory_bytes, None,
        "non-memory event must not carry memory_bytes"
    );
    assert_eq!(
        events[0].memory_category, None,
        "non-memory event must not carry memory_category"
    );
}

// ---------------------------------------------------------------------------
// Test: stack trace — present when with_stack = true
// ---------------------------------------------------------------------------

#[test]
fn event_stack_trace_present_when_with_stack_enabled() {
    let config = ProfileConfig {
        with_stack: true,
        ..ProfileConfig::default()
    };
    let ((), report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3usize, 4]]);
    });
    let ev = &report.events()[0];
    // torch.profiler with with_stack=True attaches a stack trace.
    // ferrotorch matches: stack_trace must be Some when with_stack=true.
    assert!(
        ev.stack_trace.is_some(),
        "stack_trace must be Some when with_stack=true (torch.profiler parity)"
    );
}

// ---------------------------------------------------------------------------
// Test: stack trace — absent when with_stack = false
// ---------------------------------------------------------------------------

#[test]
fn event_stack_trace_absent_when_with_stack_disabled() {
    let config = ProfileConfig {
        with_stack: false,
        ..ProfileConfig::default()
    };
    let ((), report) = with_profiler(config, |p| {
        p.record("add", "tensor_op", &[&[3usize, 4]]);
    });
    assert_eq!(
        report.events()[0].stack_trace,
        None,
        "stack_trace must be None when with_stack=false"
    );
}

// ---------------------------------------------------------------------------
// Test: shapes not recorded when record_shapes = false
// ---------------------------------------------------------------------------

#[test]
fn event_shapes_absent_when_record_shapes_disabled() {
    let config = ProfileConfig {
        record_shapes: false,
        ..ProfileConfig::default()
    };
    let ((), report) = with_profiler(config, |p| {
        p.record("mm", "tensor_op", &[&[32usize, 784], &[784, 256]]);
    });
    assert!(
        report.events()[0].input_shapes.is_empty(),
        "input_shapes must be empty when record_shapes=false"
    );
}

// ---------------------------------------------------------------------------
// Test: shapes recorded when record_shapes = true
// ---------------------------------------------------------------------------

#[test]
fn event_shapes_recorded_when_record_shapes_enabled() {
    let config = ProfileConfig {
        record_shapes: true,
        ..ProfileConfig::default()
    };
    let ((), report) = with_profiler(config, |p| {
        p.record("mm", "tensor_op", &[&[32usize, 784], &[784, 256]]);
    });
    assert_eq!(
        report.events()[0].input_shapes,
        vec![vec![32usize, 784], vec![784, 256]],
        "input_shapes must be recorded when record_shapes=true"
    );
}

// ---------------------------------------------------------------------------
// Test: MemoryCategory Display matches torch.profiler category strings
// ---------------------------------------------------------------------------

#[test]
fn memory_category_display_strings() {
    // torch.profiler uses category strings for its memory breakdown table.
    // ferrotorch must match the same human-readable names so exported traces
    // are interpretable by standard tooling.
    assert_eq!(MemoryCategory::Activations.to_string(), "activations");
    assert_eq!(MemoryCategory::Parameters.to_string(), "parameters");
    assert_eq!(MemoryCategory::OptimizerState.to_string(), "optimizer_state");
    assert_eq!(MemoryCategory::Gradients.to_string(), "gradients");
    assert_eq!(MemoryCategory::Other.to_string(), "other");
}

// ---------------------------------------------------------------------------
// Test: memory events NOT recorded when record_memory = false (torch parity)
// ---------------------------------------------------------------------------

#[test]
fn event_memory_not_recorded_when_disabled() {
    // torch.profiler with profile_memory=False silently drops memory events.
    let config = ProfileConfig {
        record_memory: false,
        ..ProfileConfig::default()
    };
    let ((), report) = with_profiler(config, |p| {
        p.record_memory("alloc", 4096);
    });
    assert_eq!(
        report.events().len(),
        0,
        "memory events must be dropped when record_memory=false (torch.profiler parity: \
         profile_memory=False)"
    );
}

// ---------------------------------------------------------------------------
// Test: GpuTimingPair fields accessible (structural)
// ---------------------------------------------------------------------------

#[test]
fn gpu_timing_pair_fields_accessible() {
    use ferrotorch_profiler::GpuTimingPair;

    let pair = GpuTimingPair {
        start_us: 100,
        end_us: 250,
    };
    assert_eq!(pair.start_us, 100);
    assert_eq!(pair.end_us, 250);
}

// ---------------------------------------------------------------------------
// Test: push_gpu_event produces a Cuda device_type event
// ---------------------------------------------------------------------------

#[test]
fn event_push_gpu_event_device_type_is_cuda() {
    use ferrotorch_profiler::{DeviceType, GpuTimingPair};

    let timing = GpuTimingPair {
        start_us: 0,
        end_us: 50,
    };
    let ((), report) = with_profiler(ProfileConfig::default(), |p| {
        p.push_gpu_event("conv2d", "cuda_kernel", timing);
    });
    let events = report.events();
    assert_eq!(events.len(), 1);
    assert_eq!(
        events[0].device_type,
        DeviceType::Cuda,
        "push_gpu_event must set device_type = Cuda (torch.profiler parity: \
         ProfilerActivity.CUDA events have DeviceType.CUDA)"
    );
    assert_eq!(events[0].duration_us, 50);
}
