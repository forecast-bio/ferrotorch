//! Thread-local profiler hook for auto-instrumented tensor ops.
//!
//! ferrotorch-core cannot depend on ferrotorch-profiler (the dependency
//! goes the other way around), so this module defines a minimal trait
//! and thread-local that ferrotorch-profiler can plug into.
//!
//! When a profiler is active on the current thread (set via
//! [`set_current`]), any tensor op wrapped in [`profile_op_scope`]
//! reports its execution time and input shapes to the profiler. This
//! is the building block for the auto-instrumentation path requested
//! in CL-379.
//!
//! # Example (from inside ferrotorch-profiler::with_profiler)
//!
//! ```ignore
//! use std::sync::Arc;
//! use ferrotorch_core::profiler_hook;
//!
//! struct MyProfiler { /* ... */ }
//! impl profiler_hook::OpProfiler for MyProfiler {
//!     fn record_op(&self, name: &str, category: &str, shapes: &[&[usize]], duration_us: u64) {
//!         /* push event into the profiler's event list */
//!     }
//! }
//!
//! let p: Arc<dyn profiler_hook::OpProfiler> = Arc::new(MyProfiler { /* ... */ });
//! profiler_hook::set_current(Some(p));
//! // ... user code that calls into ferrotorch tensor ops ...
//! profiler_hook::set_current(None);
//! ```

use std::cell::RefCell;
use std::sync::Arc;
use std::time::Instant;

/// Minimal trait that ferrotorch-profiler implements so that tensor ops
/// in ferrotorch-core can record themselves without depending on the
/// profiler crate directly.
pub trait OpProfiler: Send + Sync {
    /// Record an op event with the given name, category, input shapes,
    /// and measured duration in microseconds.
    fn record_op(&self, name: &str, category: &str, shapes: &[&[usize]], duration_us: u64);
}

thread_local! {
    /// The currently active profiler on this thread, if any. Tensor ops
    /// check this via [`current`] and report their timing if the
    /// profiler is set. The cell holds an `Arc<dyn OpProfiler>` so the
    /// profiler can outlive the closure that set it (used by the
    /// non-blocking auto-instrumentation API).
    static CURRENT: RefCell<Option<Arc<dyn OpProfiler>>> = const { RefCell::new(None) };
}

/// Set or clear the current thread's profiler. Pass `None` to clear.
///
/// `ferrotorch_profiler::with_profiler` calls this with `Some(...)`
/// before invoking the user closure and with `None` after, in an
/// RAII-style guard pattern (so even on panic the profiler is cleared).
pub fn set_current(profiler: Option<Arc<dyn OpProfiler>>) {
    CURRENT.with(|c| *c.borrow_mut() = profiler);
}

/// Return the currently active profiler on this thread, if any.
///
/// Returns a clone of the `Arc`, so the caller can hold the profiler
/// reference for the duration of one op without taking the cell's
/// borrow lock for the closure body. This avoids deadlocks if the
/// op body itself tries to record nested ops (which is normal — e.g.
/// `matmul` is composed of GEMM + bias add).
pub fn current() -> Option<Arc<dyn OpProfiler>> {
    CURRENT.with(|c| c.borrow().clone())
}

/// Run `f` and record its execution time under the current profiler
/// if one is active. When no profiler is set, the closure is called
/// directly with no overhead beyond a thread-local read and an
/// `Option::is_none()` check.
///
/// `name` should match the canonical op name (e.g. `"matmul"`,
/// `"add"`, `"softmax"`). `category` typically describes the op
/// family — common values are `"tensor_op"` for elementwise math,
/// `"linalg"` for matmul/bmm/etc., `"reduction"` for sum/mean,
/// `"activation"` for relu/gelu/softmax, `"shape"` for view/permute.
///
/// `shapes` carries the input tensor shapes for diagnostic display.
/// Pass an empty slice if shape recording is not relevant.
pub fn profile_op_scope<F, R>(name: &str, category: &str, shapes: &[&[usize]], f: F) -> R
where
    F: FnOnce() -> R,
{
    let p = current();
    if let Some(profiler) = p {
        let start = Instant::now();
        let result = f();
        let elapsed_us = start.elapsed().as_micros() as u64;
        profiler.record_op(name, category, shapes, elapsed_us);
        result
    } else {
        f()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    type TestProfilerEvent = (String, String, Vec<Vec<usize>>, u64);

    #[derive(Default)]
    struct TestProfiler {
        events: Mutex<Vec<TestProfilerEvent>>,
    }
    impl OpProfiler for TestProfiler {
        fn record_op(&self, name: &str, category: &str, shapes: &[&[usize]], duration_us: u64) {
            let shapes_owned: Vec<Vec<usize>> = shapes.iter().map(|s| s.to_vec()).collect();
            self.events.lock().unwrap().push((
                name.to_string(),
                category.to_string(),
                shapes_owned,
                duration_us,
            ));
        }
    }

    #[test]
    fn test_no_profiler_active_by_default() {
        // Use a fresh thread to guarantee a clean thread-local.
        std::thread::spawn(|| {
            assert!(current().is_none());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn test_profile_op_scope_no_profiler_runs_closure() {
        std::thread::spawn(|| {
            let result = profile_op_scope("test_op", "test", &[], || 42);
            assert_eq!(result, 42);
            assert!(current().is_none());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn test_profile_op_scope_records_when_active() {
        std::thread::spawn(|| {
            let p = Arc::new(TestProfiler::default());
            set_current(Some(p.clone() as Arc<dyn OpProfiler>));

            let result = profile_op_scope("matmul", "linalg", &[&[2, 3], &[3, 4]], || {
                // Simulate a tiny op body.
                std::thread::sleep(std::time::Duration::from_micros(1));
                "ok"
            });
            assert_eq!(result, "ok");

            set_current(None);

            let events = p.events.lock().unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(events[0].0, "matmul");
            assert_eq!(events[0].1, "linalg");
            assert_eq!(events[0].2, vec![vec![2, 3], vec![3, 4]]);
            // Don't assert on the actual duration -- timing is flaky.
        })
        .join()
        .unwrap();
    }

    #[test]
    fn test_set_current_can_be_cleared() {
        std::thread::spawn(|| {
            let p = Arc::new(TestProfiler::default());
            set_current(Some(p as Arc<dyn OpProfiler>));
            assert!(current().is_some());
            set_current(None);
            assert!(current().is_none());
        })
        .join()
        .unwrap();
    }

    #[test]
    fn test_nested_profile_op_scope_records_inner_op() {
        // The op body itself can call profile_op_scope -- both events
        // should land in the profiler.
        std::thread::spawn(|| {
            let p = Arc::new(TestProfiler::default());
            set_current(Some(p.clone() as Arc<dyn OpProfiler>));

            profile_op_scope("outer", "test", &[&[2, 2]], || {
                profile_op_scope("inner", "test", &[&[2, 2]], || {});
            });

            set_current(None);

            let events = p.events.lock().unwrap();
            assert_eq!(events.len(), 2);
            // Inner op completes first (LIFO scope timing), so it's
            // recorded before the outer op.
            assert_eq!(events[0].0, "inner");
            assert_eq!(events[1].0, "outer");
        })
        .join()
        .unwrap();
    }

    #[test]
    fn test_thread_local_isolation() {
        // Setting a profiler on one thread does not affect another.
        let h = std::thread::spawn(|| {
            let p = Arc::new(TestProfiler::default());
            set_current(Some(p as Arc<dyn OpProfiler>));
            assert!(current().is_some());
            // Spawn a child thread; it should see no profiler.
            std::thread::spawn(|| {
                assert!(current().is_none());
            })
            .join()
            .unwrap();
            set_current(None);
        });
        h.join().unwrap();
    }
}
