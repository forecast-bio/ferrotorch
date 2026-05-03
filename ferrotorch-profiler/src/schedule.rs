// CL-333: ProfileSchedule — wait/warmup/active/repeat cycle

/// State machine phase for the profiler schedule.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulePhase {
    /// Waiting — profiler is idle, events are discarded.
    Waiting,
    /// Warmup — profiler is idle, events are discarded, but the
    /// runtime is warming up (JIT, caches, etc.).
    Warmup,
    /// Active — profiler is recording events.
    Active,
    /// Done — all repeat cycles exhausted, profiler is idle.
    Done,
}

impl std::fmt::Display for SchedulePhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulePhase::Waiting => write!(f, "Waiting"),
            SchedulePhase::Warmup => write!(f, "Warmup"),
            SchedulePhase::Active => write!(f, "Active"),
            SchedulePhase::Done => write!(f, "Done"),
        }
    }
}

/// Controls the profiler's wait/warmup/active/repeat lifecycle.
///
/// The schedule advances through phases as [`step()`](ProfileSchedule::step)
/// is called at iteration boundaries.
///
/// ```text
///  ┌──────────────────────────── one cycle ────────────────────────────┐
///  │  wait steps  │  warmup steps  │  active steps  │                 │
///  └──────────────┴────────────────┴────────────────┘                 │
///                                                    repeat × N ──────┘
/// ```
///
/// After `repeat` cycles (default 1), the schedule enters [`Done`](SchedulePhase::Done).
pub struct ProfileSchedule {
    /// Number of steps to wait before the first warmup.
    wait: u64,
    /// Number of warmup steps (events discarded, just cache priming).
    warmup: u64,
    /// Number of active profiling steps.
    active: u64,
    /// How many wait→warmup→active cycles to run.
    repeat: u64,

    // --- mutable runtime state ---
    /// Current step within the current cycle (0-based).
    current_step: u64,
    /// Current cycle (0-based).
    current_cycle: u64,
    /// Current phase (derived from step position).
    phase: SchedulePhase,
    /// Callback invoked when an active window ends.
    on_trace_ready: Option<Box<dyn FnMut(u64) + Send>>,
}

impl std::fmt::Debug for ProfileSchedule {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfileSchedule")
            .field("wait", &self.wait)
            .field("warmup", &self.warmup)
            .field("active", &self.active)
            .field("repeat", &self.repeat)
            .field("current_step", &self.current_step)
            .field("current_cycle", &self.current_cycle)
            .field("phase", &self.phase)
            .field(
                "on_trace_ready",
                &self.on_trace_ready.as_ref().map(|_| ".."),
            )
            .finish()
    }
}

impl ProfileSchedule {
    /// Create a new schedule.
    ///
    /// # Panics
    ///
    /// Panics if `active` is zero (there must be at least one active step).
    #[must_use]
    pub fn new(wait: u64, warmup: u64, active: u64, repeat: u64) -> Self {
        assert!(active > 0, "ProfileSchedule: `active` must be > 0");
        assert!(repeat > 0, "ProfileSchedule: `repeat` must be > 0");
        let phase = if wait > 0 {
            SchedulePhase::Waiting
        } else if warmup > 0 {
            SchedulePhase::Warmup
        } else {
            SchedulePhase::Active
        };
        Self {
            wait,
            warmup,
            active,
            repeat,
            current_step: 0,
            current_cycle: 0,
            phase,
            on_trace_ready: None,
        }
    }

    /// Set a callback that fires when each active window completes.
    ///
    /// The callback receives the cycle index (0-based).
    pub fn set_on_trace_ready(&mut self, cb: impl FnMut(u64) + Send + 'static) {
        self.on_trace_ready = Some(Box::new(cb));
    }

    /// Current phase of the schedule.
    #[must_use]
    pub fn phase(&self) -> SchedulePhase {
        self.phase
    }

    /// Whether the profiler should be recording events right now.
    #[must_use]
    pub fn is_active(&self) -> bool {
        self.phase == SchedulePhase::Active
    }

    /// Current step within the current cycle.
    #[must_use]
    pub fn current_step(&self) -> u64 {
        self.current_step
    }

    /// Current cycle (0-based).
    #[must_use]
    pub fn current_cycle(&self) -> u64 {
        self.current_cycle
    }

    /// Total steps per cycle.
    fn cycle_length(&self) -> u64 {
        self.wait + self.warmup + self.active
    }

    /// Advance the schedule by one step. Call this at each iteration
    /// boundary (e.g., after each training step).
    ///
    /// Returns the phase *after* advancing.
    pub fn step(&mut self) -> SchedulePhase {
        if self.phase == SchedulePhase::Done {
            return SchedulePhase::Done;
        }

        self.current_step += 1;

        let cycle_len = self.cycle_length();
        let pos = self.current_step;

        if pos >= cycle_len {
            // End of cycle — fire callback if we were active.
            if let Some(ref mut cb) = self.on_trace_ready {
                cb(self.current_cycle);
            }

            self.current_cycle += 1;
            self.current_step = 0;

            if self.current_cycle >= self.repeat {
                self.phase = SchedulePhase::Done;
                return SchedulePhase::Done;
            }

            // Start new cycle.
            self.phase = if self.wait > 0 {
                SchedulePhase::Waiting
            } else if self.warmup > 0 {
                SchedulePhase::Warmup
            } else {
                SchedulePhase::Active
            };
            return self.phase;
        }

        // Within the cycle, determine phase from position.
        self.phase = if pos < self.wait {
            SchedulePhase::Waiting
        } else if pos < self.wait + self.warmup {
            SchedulePhase::Warmup
        } else {
            SchedulePhase::Active
        };

        self.phase
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_phases() {
        // wait=1, warmup=1, active=2, repeat=1
        let mut sched = ProfileSchedule::new(1, 1, 2, 1);
        assert_eq!(sched.phase(), SchedulePhase::Waiting);
        assert!(!sched.is_active());

        // Step 0→1: still in wait (pos=1, wait=1 → pos < wait is false,
        // pos < wait+warmup=2 → true → Warmup)
        let p = sched.step();
        assert_eq!(p, SchedulePhase::Warmup);
        assert!(!sched.is_active());

        // Step 1→2: active
        let p = sched.step();
        assert_eq!(p, SchedulePhase::Active);
        assert!(sched.is_active());

        // Step 2→3: still active
        let p = sched.step();
        assert_eq!(p, SchedulePhase::Active);

        // Step 3→4: end of cycle, repeat=1, done
        let p = sched.step();
        assert_eq!(p, SchedulePhase::Done);
    }

    #[test]
    fn repeat_cycles() {
        // wait=0, warmup=0, active=2, repeat=2
        let mut sched = ProfileSchedule::new(0, 0, 2, 2);
        assert_eq!(sched.phase(), SchedulePhase::Active);

        // Cycle 0: step 0→1 active
        assert_eq!(sched.step(), SchedulePhase::Active);
        // Cycle 0: step 1→2 → end of cycle → cycle 1 starts
        assert_eq!(sched.step(), SchedulePhase::Active);
        // Cycle 1: step 0→1 active
        assert_eq!(sched.step(), SchedulePhase::Active);
        // Cycle 1: step 1→2 → end of cycle → done
        assert_eq!(sched.step(), SchedulePhase::Done);
        // Stays done
        assert_eq!(sched.step(), SchedulePhase::Done);
    }

    #[test]
    fn on_trace_ready_fires() {
        use std::sync::{Arc, Mutex};
        let fired = Arc::new(Mutex::new(Vec::new()));
        let fired_clone = fired.clone();

        let mut sched = ProfileSchedule::new(0, 0, 1, 3);
        sched.set_on_trace_ready(move |cycle| {
            fired_clone.lock().unwrap().push(cycle);
        });

        // 3 cycles of 1 active step each
        sched.step(); // end cycle 0
        sched.step(); // end cycle 1
        sched.step(); // end cycle 2 → done

        let calls = fired.lock().unwrap();
        assert_eq!(*calls, vec![0, 1, 2]);
    }

    #[test]
    fn no_wait_no_warmup() {
        let mut sched = ProfileSchedule::new(0, 0, 3, 1);
        assert_eq!(sched.phase(), SchedulePhase::Active);
        assert_eq!(sched.step(), SchedulePhase::Active);
        assert_eq!(sched.step(), SchedulePhase::Active);
        assert_eq!(sched.step(), SchedulePhase::Done);
    }

    #[test]
    fn only_warmup() {
        let mut sched = ProfileSchedule::new(0, 2, 1, 1);
        assert_eq!(sched.phase(), SchedulePhase::Warmup);
        assert_eq!(sched.step(), SchedulePhase::Warmup);
        assert_eq!(sched.step(), SchedulePhase::Active);
        assert_eq!(sched.step(), SchedulePhase::Done);
    }

    #[test]
    #[should_panic(expected = "`active` must be > 0")]
    fn zero_active_panics() {
        let _ = ProfileSchedule::new(1, 1, 0, 1);
    }

    #[test]
    #[should_panic(expected = "`repeat` must be > 0")]
    fn zero_repeat_panics() {
        let _ = ProfileSchedule::new(0, 0, 1, 0);
    }

    #[test]
    fn current_step_and_cycle_tracking() {
        let mut sched = ProfileSchedule::new(1, 0, 1, 2);
        assert_eq!(sched.current_step(), 0);
        assert_eq!(sched.current_cycle(), 0);

        sched.step(); // step 1 in cycle 0 (active)
        assert_eq!(sched.current_step(), 1);
        assert_eq!(sched.current_cycle(), 0);

        sched.step(); // end cycle 0 → cycle 1 starts
        assert_eq!(sched.current_step(), 0);
        assert_eq!(sched.current_cycle(), 1);
    }
}
