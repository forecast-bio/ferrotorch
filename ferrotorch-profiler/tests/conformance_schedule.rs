//! Conformance — `ferrotorch_profiler::schedule` parity against `torch.profiler`.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/826>.
//!
//! Verifies that [`ProfileSchedule`] and [`SchedulePhase`] match the
//! behavioral contract of `torch.profiler.schedule(wait, warmup, active,
//! repeat)`.
//!
//! PyTorch reference mapping:
//!
//! | torch `ProfilerAction`  | ferrotorch `SchedulePhase` |
//! |-------------------------|----------------------------|
//! | `NONE` (initial / post) | `Waiting` / `Done`         |
//! | `WARMUP`                | `Warmup`                   |
//! | `RECORD`                | `Active`                   |
//! | `RECORD_AND_SAVE`       | `Active` (+ callback fires)|
//!
//! What is pinned: phase sequence per step count (pure arithmetic state
//! machine, no wall-clock).  `on_trace_ready` callback fire count.
//!
//! What is NOT pinned: absolute timing, step latency.

use ferrotorch_profiler::schedule::{ProfileSchedule, SchedulePhase};

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

/// Parse a JSON phase string into `SchedulePhase`.
fn parse_phase(s: &str) -> SchedulePhase {
    match s {
        "Waiting" => SchedulePhase::Waiting,
        "Warmup" => SchedulePhase::Warmup,
        "Active" => SchedulePhase::Active,
        "Done" => SchedulePhase::Done,
        other => panic!("unknown phase in fixture: {other:?}"),
    }
}

/// Drive a schedule through N steps and collect the phase after each step,
/// prepended with the initial phase (before any step()).
fn drive_schedule(sched: &mut ProfileSchedule, n_steps: usize) -> Vec<SchedulePhase> {
    let mut phases = vec![sched.phase()];
    for _ in 0..n_steps {
        phases.push(sched.step());
    }
    phases
}

// ---------------------------------------------------------------------------
// Fixture-driven helper: run one schedule fixture end-to-end.
// ---------------------------------------------------------------------------

fn run_schedule_fixture(fixture_id: &str) {
    let all = fixtures_json();
    let fixtures = all["schedule_fixtures"]
        .as_array()
        .expect("fixtures.json missing schedule_fixtures");
    let fix = fixtures
        .iter()
        .find(|f| f["id"].as_str() == Some(fixture_id))
        .unwrap_or_else(|| panic!("fixture {fixture_id:?} not found"));

    let wait = fix["wait"].as_u64().expect("wait");
    let warmup = fix["warmup"].as_u64().expect("warmup");
    let active = fix["active"].as_u64().expect("active");
    let repeat = fix["repeat"].as_u64().expect("repeat");

    let expected: Vec<SchedulePhase> = fix["expected_phase_sequence"]
        .as_array()
        .expect("expected_phase_sequence")
        .iter()
        .map(|v| parse_phase(v.as_str().expect("phase string")))
        .collect();

    let mut sched = ProfileSchedule::new(wait, warmup, active, repeat);
    // Drive n_steps = expected.len() - 1 (first element is initial phase).
    let actual = drive_schedule(&mut sched, expected.len() - 1);

    assert_eq!(
        actual, expected,
        "fixture {fixture_id}: phase sequence mismatch.\n  actual  : {actual:?}\n  expected: {expected:?}"
    );
}

// ---------------------------------------------------------------------------
// Tests: one per schedule fixture
// ---------------------------------------------------------------------------

#[test]
fn schedule_wait1_warmup1_active2_repeat1() {
    run_schedule_fixture("schedule_wait1_warmup1_active2_repeat1");
}

#[test]
fn schedule_no_wait_no_warmup_active3_repeat1() {
    run_schedule_fixture("schedule_no_wait_no_warmup_active3_repeat1");
}

#[test]
fn schedule_warmup_only_active1_repeat1() {
    run_schedule_fixture("schedule_warmup_only_active1_repeat1");
}

#[test]
fn schedule_active1_repeat3_phase_sequence() {
    run_schedule_fixture("schedule_active1_repeat3");
}

// ---------------------------------------------------------------------------
// Test: on_trace_ready fires once per completed active window
//
// torch.profiler parity: the on_trace_ready callback fires at the end of
// each active window (ProfilerAction.RECORD_AND_SAVE), once per repeat cycle.
// ---------------------------------------------------------------------------

#[test]
fn schedule_on_trace_ready_fires_once_per_cycle() {
    let all = fixtures_json();
    let fix = all["schedule_fixtures"]
        .as_array()
        .unwrap()
        .iter()
        .find(|f| f["id"].as_str() == Some("schedule_active1_repeat3"))
        .expect("schedule_active1_repeat3 fixture");

    let wait   = fix["wait"].as_u64().unwrap();
    let warmup = fix["warmup"].as_u64().unwrap();
    let active = fix["active"].as_u64().unwrap();
    let repeat = fix["repeat"].as_u64().unwrap();
    let expected_calls = fix["expected_on_trace_ready_calls"]
        .as_u64()
        .expect("expected_on_trace_ready_calls") as usize;

    use std::sync::{Arc, Mutex};
    let fired: Arc<Mutex<Vec<u64>>> = Arc::new(Mutex::new(Vec::new()));
    let fired_clone = fired.clone();

    let mut sched = ProfileSchedule::new(wait, warmup, active, repeat);
    sched.set_on_trace_ready(move |cycle| {
        fired_clone.lock().unwrap().push(cycle);
    });

    // Drive to Done.
    loop {
        let phase = sched.step();
        if phase == SchedulePhase::Done {
            break;
        }
    }

    let calls = fired.lock().unwrap();
    assert_eq!(
        calls.len(),
        expected_calls,
        "on_trace_ready fired {} times, expected {} (torch.profiler fires once per cycle)",
        calls.len(),
        expected_calls
    );
    // Cycle indices should be 0-based and monotonically increasing.
    for (i, &cycle) in calls.iter().enumerate() {
        assert_eq!(
            cycle, i as u64,
            "on_trace_ready cycle index must be 0-based monotone: got {cycle} at position {i}"
        );
    }
}

// ---------------------------------------------------------------------------
// Test: step() returns Done and stays Done after all cycles exhausted
// ---------------------------------------------------------------------------

#[test]
fn schedule_done_is_terminal() {
    // torch.profiler: once Done, subsequent steps are no-ops.
    let mut sched = ProfileSchedule::new(0, 0, 1, 1);
    // One cycle of one active step: step once → Done.
    assert_eq!(sched.step(), SchedulePhase::Done);
    // Additional steps must stay Done, not panic or wrap.
    assert_eq!(sched.step(), SchedulePhase::Done);
    assert_eq!(sched.step(), SchedulePhase::Done);
}

// ---------------------------------------------------------------------------
// Test: is_active() returns true only during Active phase
// ---------------------------------------------------------------------------

#[test]
fn schedule_is_active_only_during_active_phase() {
    // wait=1, warmup=1, active=2, repeat=1
    let mut sched = ProfileSchedule::new(1, 1, 2, 1);
    // Initial phase: Waiting
    assert!(!sched.is_active(), "Waiting phase must not be active");
    // step → Warmup
    sched.step();
    assert!(!sched.is_active(), "Warmup phase must not be active");
    // step → Active
    sched.step();
    assert!(sched.is_active(), "Active phase must be active");
    sched.step(); // still Active
    assert!(sched.is_active());
    // step → Done
    sched.step();
    assert!(!sched.is_active(), "Done phase must not be active");
}

// ---------------------------------------------------------------------------
// Test: current_step() and current_cycle() tracking
// ---------------------------------------------------------------------------

#[test]
fn schedule_step_and_cycle_tracking() {
    // wait=1, warmup=0, active=1, repeat=2
    let mut sched = ProfileSchedule::new(1, 0, 1, 2);
    assert_eq!(sched.current_step(), 0);
    assert_eq!(sched.current_cycle(), 0);

    sched.step(); // step 1 in cycle 0 (active)
    assert_eq!(sched.current_step(), 1);
    assert_eq!(sched.current_cycle(), 0);

    sched.step(); // end of cycle 0 → cycle 1 starts, step resets to 0
    assert_eq!(sched.current_step(), 0);
    assert_eq!(sched.current_cycle(), 1);
}

// ---------------------------------------------------------------------------
// Test: SchedulePhase Display matches torch.profiler convention
// ---------------------------------------------------------------------------

#[test]
fn schedule_phase_display_strings() {
    // torch.profiler uses string labels for its schedule phases in logs/exports.
    assert_eq!(SchedulePhase::Waiting.to_string(), "Waiting");
    assert_eq!(SchedulePhase::Warmup.to_string(),  "Warmup");
    assert_eq!(SchedulePhase::Active.to_string(),  "Active");
    assert_eq!(SchedulePhase::Done.to_string(),    "Done");
}

// ---------------------------------------------------------------------------
// Test: clone preserves config and position, drops callback
//
// torch.profiler.schedule() returns a stateless callable; ferrotorch wraps
// the state in ProfileSchedule.  The clone contract (config+position kept,
// callback dropped) is tested here as a structural parity assertion.
// ---------------------------------------------------------------------------

#[test]
fn schedule_clone_preserves_config_drops_callback() {
    use std::sync::{Arc, Mutex};

    let mut sched = ProfileSchedule::new(0, 0, 4, 1);
    let original_fires: Arc<Mutex<u64>> = Arc::new(Mutex::new(0));
    {
        let c = original_fires.clone();
        sched.set_on_trace_ready(move |_| {
            *c.lock().unwrap() += 1;
        });
    }
    sched.step();
    sched.step(); // now at step 2 of 4

    let mut cloned = sched.clone();
    // Config and position preserved.
    assert_eq!(cloned.phase(), sched.phase());
    assert_eq!(cloned.current_step(), sched.current_step());

    // Drive clone to completion — original's counter must NOT increment.
    while cloned.step() != SchedulePhase::Done {}
    assert_eq!(
        *original_fires.lock().unwrap(),
        0,
        "clone must not share the original's on_trace_ready callback"
    );

    // Original still fires its own callback.
    sched.step();
    sched.step(); // end of cycle
    assert_eq!(
        *original_fires.lock().unwrap(),
        1,
        "original's on_trace_ready must still fire after clone is driven"
    );
}

// ---------------------------------------------------------------------------
// Test: zero-active panics (parity with torch.profiler.schedule invariant)
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "`active` must be > 0")]
fn schedule_zero_active_panics() {
    // torch.profiler.schedule requires active > 0; ferrotorch must match.
    let _ = ProfileSchedule::new(0, 0, 0, 1);
}

// ---------------------------------------------------------------------------
// Test: zero-repeat panics
// ---------------------------------------------------------------------------

#[test]
#[should_panic(expected = "`repeat` must be > 0")]
fn schedule_zero_repeat_panics() {
    let _ = ProfileSchedule::new(0, 0, 1, 0);
}
