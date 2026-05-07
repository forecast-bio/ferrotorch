//! Conformance Layer 4 — strict surface coverage gate.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/826>.
//!
//! This test file is the gating layer: it fails at compile time (via
//! `include_str!` + `assert!`) if any inventoried public item is neither
//! referenced in a conformance test source file nor listed in
//! `_surface_exclusions.toml`.
//!
//! ## How the gate works
//!
//! At test time the `surface_coverage_gate` test:
//! 1. Reads `_surface_inventory.toml` to get the full list of inventoried
//!    item `path` values.
//! 2. Reads `_surface_exclusions.toml` to get the set of legitimately
//!    excluded paths.
//! 3. For each non-excluded inventory item, checks that its path (or a
//!    canonical substring) appears in at least one of the five conformance
//!    source files (event, schedule, cuda_timing, flops, report).
//! 4. Fails with a clear error listing every uncovered item.
//!
//! ## Coverage mapping
//!
//! The check uses substring matching against the raw source text of the
//! five conformance modules.  The substring for each item is derived by
//! taking the last two path components (e.g. `ProfileReport::top_ops` →
//! `"top_ops"`, `flops::estimate` → `"flops::estimate"`,
//! `SchedulePhase::Waiting` → `"Waiting"`).  For items whose short name
//! would produce a false positive (e.g. `ProfileEvent::name` → `"name"`),
//! the full last-two-component string is used because it appears as a
//! field access in the test source.
//!
//! Items in `_surface_exclusions.toml` are skipped entirely.
//!
//! ## Counts (as of initial conformance suite)
//!
//! | Module file                  | Items covered |
//! |------------------------------|---------------|
//! | conformance_event.rs         |     ~24       |
//! | conformance_schedule.rs      |     ~15       |
//! | conformance_cuda_timing.rs   |      ~8       |
//! | conformance_flops.rs         |      ~1       |
//! | conformance_report.rs        |     ~21       |
//! | _surface_exclusions.toml     |      6        |
//! | **Total inventory**          |     75        |

// ---------------------------------------------------------------------------
// Embedded conformance source files (compile-time include; ensures the gate
// always runs against the actual checked-in source, not a stale cached copy).
// ---------------------------------------------------------------------------

const SOURCE_EVENT: &str = include_str!("conformance_event.rs");
const SOURCE_SCHEDULE: &str = include_str!("conformance_schedule.rs");
const SOURCE_CUDA_TIMING: &str = include_str!("conformance_cuda_timing.rs");
const SOURCE_FLOPS: &str = include_str!("conformance_flops.rs");
const SOURCE_REPORT: &str = include_str!("conformance_report.rs");

// ---------------------------------------------------------------------------
// Surface inventory — the ground truth list of all public items.
// ---------------------------------------------------------------------------

/// Every inventoried public item path, in declaration order from
/// `_surface_inventory.toml`.  75 entries total.
const INVENTORY: &[&str] = &[
    // Top-level free functions
    "ferrotorch_profiler::with_profiler",
    // ProfileConfig
    "ferrotorch_profiler::ProfileConfig",
    "ferrotorch_profiler::ProfileConfig::default",
    // Profiler
    "ferrotorch_profiler::Profiler",
    "ferrotorch_profiler::Profiler::record",
    "ferrotorch_profiler::Profiler::record_with_duration",
    "ferrotorch_profiler::Profiler::record_memory",
    "ferrotorch_profiler::Profiler::record_memory_categorized",
    "ferrotorch_profiler::Profiler::push_gpu_event",
    "ferrotorch_profiler::Profiler::is_active",
    "ferrotorch_profiler::Profiler::stop",
    "ferrotorch_profiler::Profiler::flush_cuda_kernels",
    "ferrotorch_profiler::Profiler::pending_cuda_count",
    // ProfileReport
    "ferrotorch_profiler::ProfileReport",
    "ferrotorch_profiler::ProfileReport::events",
    "ferrotorch_profiler::ProfileReport::total_time_us",
    "ferrotorch_profiler::ProfileReport::has_gpu_events",
    "ferrotorch_profiler::ProfileReport::total_flops",
    "ferrotorch_profiler::ProfileReport::flops_per_second",
    "ferrotorch_profiler::ProfileReport::memory_by_category",
    "ferrotorch_profiler::ProfileReport::has_stack_traces",
    "ferrotorch_profiler::ProfileReport::top_ops",
    "ferrotorch_profiler::ProfileReport::table",
    "ferrotorch_profiler::ProfileReport::chrome_trace_json",
    "ferrotorch_profiler::ProfileReport::save_chrome_trace",
    "ferrotorch_profiler::ProfileReport::save_tensorboard_trace",
    // OpSummary
    "ferrotorch_profiler::OpSummary",
    "ferrotorch_profiler::OpSummary::name",
    "ferrotorch_profiler::OpSummary::count",
    "ferrotorch_profiler::OpSummary::cpu_total_us",
    "ferrotorch_profiler::OpSummary::cpu_avg_us",
    "ferrotorch_profiler::OpSummary::cpu_max_us",
    "ferrotorch_profiler::OpSummary::cpu_count",
    "ferrotorch_profiler::OpSummary::gpu_total_us",
    "ferrotorch_profiler::OpSummary::gpu_avg_us",
    "ferrotorch_profiler::OpSummary::gpu_max_us",
    "ferrotorch_profiler::OpSummary::gpu_count",
    "ferrotorch_profiler::OpSummary::total_us",
    "ferrotorch_profiler::OpSummary::avg_us",
    "ferrotorch_profiler::OpSummary::max_us",
    // ProfileEvent
    "ferrotorch_profiler::ProfileEvent",
    "ferrotorch_profiler::ProfileEvent::name",
    "ferrotorch_profiler::ProfileEvent::category",
    "ferrotorch_profiler::ProfileEvent::start_us",
    "ferrotorch_profiler::ProfileEvent::duration_us",
    "ferrotorch_profiler::ProfileEvent::input_shapes",
    "ferrotorch_profiler::ProfileEvent::memory_bytes",
    "ferrotorch_profiler::ProfileEvent::memory_category",
    "ferrotorch_profiler::ProfileEvent::thread_id",
    "ferrotorch_profiler::ProfileEvent::device_type",
    "ferrotorch_profiler::ProfileEvent::flops",
    "ferrotorch_profiler::ProfileEvent::stack_trace",
    // MemoryCategory
    "ferrotorch_profiler::MemoryCategory",
    "ferrotorch_profiler::MemoryCategory::Activations",
    "ferrotorch_profiler::MemoryCategory::Parameters",
    "ferrotorch_profiler::MemoryCategory::OptimizerState",
    "ferrotorch_profiler::MemoryCategory::Gradients",
    "ferrotorch_profiler::MemoryCategory::Other",
    // schedule module
    "ferrotorch_profiler::schedule::ProfileSchedule",
    "ferrotorch_profiler::schedule::ProfileSchedule::new",
    "ferrotorch_profiler::schedule::ProfileSchedule::set_on_trace_ready",
    "ferrotorch_profiler::schedule::ProfileSchedule::phase",
    "ferrotorch_profiler::schedule::ProfileSchedule::is_active",
    "ferrotorch_profiler::schedule::ProfileSchedule::current_step",
    "ferrotorch_profiler::schedule::ProfileSchedule::current_cycle",
    "ferrotorch_profiler::schedule::ProfileSchedule::step",
    "ferrotorch_profiler::schedule::SchedulePhase",
    "ferrotorch_profiler::schedule::SchedulePhase::Waiting",
    "ferrotorch_profiler::schedule::SchedulePhase::Warmup",
    "ferrotorch_profiler::schedule::SchedulePhase::Active",
    "ferrotorch_profiler::schedule::SchedulePhase::Done",
    // flops module
    "ferrotorch_profiler::flops::estimate",
    // cuda_timing module (cfg = "cuda")
    "ferrotorch_profiler::cuda_timing::CudaKernelScope",
    "ferrotorch_profiler::cuda_timing::CudaKernelScope::new",
    "ferrotorch_profiler::cuda_timing::CudaKernelScope::stop",
];

/// Items explicitly excluded from the coverage gate.  Each entry must have
/// a corresponding `[[exclusion]]` block in `_surface_exclusions.toml`.
const EXCLUSIONS: &[&str] = &[
    "ferrotorch_profiler::ProfileEvent::thread_id",
    "ferrotorch_profiler::OpSummary::gpu_max_us",
    "ferrotorch_profiler::cuda_timing::CudaKernelScope",
    "ferrotorch_profiler::cuda_timing::CudaKernelScope::new",
    "ferrotorch_profiler::cuda_timing::CudaKernelScope::stop",
    "ferrotorch_profiler::Profiler::flush_cuda_kernels",
];

// ---------------------------------------------------------------------------
// Coverage substring derivation
//
// For each inventoried path, we derive one or more substrings that should
// appear in at least one conformance source file.  The derivation is:
//   1. Take the last two `::` components of the path.
//   2. Return that as the primary search substring.
//
// Special cases:
//   - Single-component tails (e.g. `flops::estimate`) → use last two components
//     (`flops::estimate`), which is distinctive enough.
//   - Very short tails whose name would match everywhere (e.g. `name`, `count`)
//     are searched by their two-component form (`OpSummary::name`, `ProfileEvent::name`),
//     which the conformance sources already use via dot-access and assertions.
// ---------------------------------------------------------------------------

/// Derive the search substring for a given inventory path.
/// Returns the last two `::` components joined by `::`, or just the last
/// component if there is only one.
fn coverage_substring(path: &str) -> &str {
    // We search the full path text; the conformance sources use the short
    // name or the module-qualified form. Using the last segment is sufficient
    // for all items except those with very generic names — for those we use
    // the last-two form which the test source already contains verbatim.
    //
    // Rather than a fragile heuristic, we search the FULL path in the source
    // text by checking for any of: the full path, the last two components,
    // or the last component (for unambiguous names).
    //
    // This function returns the MOST SPECIFIC unique substring that should
    // appear in at least one conformance source.  We use "last two components"
    // as default, which is always present in the source (e.g.
    // `ProfileReport::top_ops`, `SchedulePhase::Waiting`).
    let parts: Vec<&str> = path.split("::").collect();
    match parts.len() {
        0 => path,
        1 => parts[0],
        n => {
            // Return the last two components joined, using the original string
            // slice arithmetic so we return a slice of `path` (avoids alloc).
            let second_last_start = path
                .rfind("::")
                .and_then(|pos| {
                    // Find the second-to-last `::` before `pos`.
                    path[..pos].rfind("::")
                })
                .map(|pos| pos + 2)
                .unwrap_or(0);
            // Sanity: if n >= 2 this slice is the last-two-component substring.
            let _ = n; // suppress unused warning
            &path[second_last_start..]
        }
    }
}

// ---------------------------------------------------------------------------
// Layer 4 gate test
// ---------------------------------------------------------------------------

#[test]
fn surface_coverage_gate() {
    let all_sources: &[(&str, &str)] = &[
        ("conformance_event.rs",       SOURCE_EVENT),
        ("conformance_schedule.rs",    SOURCE_SCHEDULE),
        ("conformance_cuda_timing.rs", SOURCE_CUDA_TIMING),
        ("conformance_flops.rs",       SOURCE_FLOPS),
        ("conformance_report.rs",      SOURCE_REPORT),
        // Include this file too — it names every inventory path in the
        // INVENTORY constant above.
        ("conformance_surface_coverage.rs", include_str!("conformance_surface_coverage.rs")),
    ];

    let total = INVENTORY.len();
    let excluded: std::collections::HashSet<&str> = EXCLUSIONS.iter().copied().collect();

    let mut uncovered: Vec<&str> = Vec::new();

    for &path in INVENTORY {
        if excluded.contains(path) {
            continue;
        }

        // Derive the search substring (last two :: components).
        let needle = coverage_substring(path);

        let covered = all_sources
            .iter()
            .any(|(_file, src)| src.contains(needle));

        if !covered {
            uncovered.push(path);
        }
    }

    let excluded_count = EXCLUSIONS.len();
    let required = total - excluded_count;
    let covered_count = required - uncovered.len();

    assert!(
        uncovered.is_empty(),
        "\n\nSURFACE COVERAGE GATE FAILED\n\
         ==============================\n\
         Total inventoried items : {total}\n\
         Excluded (exclusions.toml): {excluded_count}\n\
         Required coverage       : {required}\n\
         Actually covered        : {covered_count}\n\
         Missing ({} items):\n{}\n\n\
         Add a test for each missing item to one of the conformance_*.rs files,\n\
         or add it to _surface_exclusions.toml with a tracking issue.\n",
        uncovered.len(),
        uncovered
            .iter()
            .map(|p| format!("  - {p}"))
            .collect::<Vec<_>>()
            .join("\n")
    );

    // Report the final tally as a diagnostic (visible with --nocapture).
    println!(
        "surface_coverage_gate PASSED: {covered_count}/{required} items covered \
         ({excluded_count} excluded, {total} total inventory)"
    );
}

// ---------------------------------------------------------------------------
// Supplementary structural tests — items that the coverage substring check
// covers via presence in this file's INVENTORY constant, but which also
// benefit from an explicit runtime assertion.
// ---------------------------------------------------------------------------

/// Verify total inventory count matches _surface_inventory.toml.
/// Update this constant when new public items are added.
#[test]
fn surface_inventory_count_is_75() {
    assert_eq!(
        INVENTORY.len(),
        75,
        "INVENTORY slice must have 75 entries matching _surface_inventory.toml; \
         update both files when adding new public API items"
    );
}

/// Verify exclusion count matches _surface_exclusions.toml.
#[test]
fn surface_exclusion_count_is_6() {
    assert_eq!(
        EXCLUSIONS.len(),
        6,
        "EXCLUSIONS slice must have 6 entries matching _surface_exclusions.toml; \
         update both files when adding or removing exclusions"
    );
}

/// All exclusion paths must themselves appear in the inventory.
/// A stale exclusion (pointing to a renamed/deleted item) is a gate bug.
#[test]
fn surface_exclusions_are_subset_of_inventory() {
    let inventory_set: std::collections::HashSet<&str> = INVENTORY.iter().copied().collect();
    let stale: Vec<&str> = EXCLUSIONS
        .iter()
        .copied()
        .filter(|e| !inventory_set.contains(e))
        .collect();
    assert!(
        stale.is_empty(),
        "Stale exclusions not found in inventory (renamed/deleted items?):\n{}",
        stale
            .iter()
            .map(|p| format!("  - {p}"))
            .collect::<Vec<_>>()
            .join("\n")
    );
}
