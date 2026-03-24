//! Autocast operation categorization and policy engine.
//!
//! This module classifies operations into categories that determine how they
//! behave under mixed-precision autocast:
//!
//! - **`ReducedPrecision`**: operations that benefit from f16/bf16 (matmul, conv, linear).
//! - **`FullPrecision`**: operations that need f32 for numerical accuracy (reductions, norms, loss).
//! - **`Passthrough`**: elementwise ops that follow whatever dtype their inputs have.
//!
//! When autocast is enabled via [`super::autocast::autocast`], the functions here
//! let individual ops query whether they should cast their inputs. The primary
//! entry point for op implementations is [`autocast_guard`], which returns the
//! [`AutocastCategory`] for an op and optionally records debug events.

use super::autocast::{is_autocast_debug, is_autocast_enabled};

/// Policy: which operations should be cast to reduced precision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutocastCategory {
    /// Cast to reduced precision (matmul, conv, linear).
    ReducedPrecision,
    /// Keep in full precision (reductions, norms, softmax, loss).
    FullPrecision,
    /// No casting (elementwise -- follow input dtype).
    Passthrough,
}

/// Get the autocast category for an operation.
pub fn autocast_category(op_name: &str) -> AutocastCategory {
    match op_name {
        // Reduced precision (benefits from f16/bf16)
        "mm" | "matmul" | "bmm" | "linear" | "conv1d" | "conv2d" | "conv_transpose2d"
        // addmm and einsum are not wired yet but belong here for when they are.
        | "addmm" | "einsum" => {
            AutocastCategory::ReducedPrecision
        }
        // Full precision (needs f32 for accuracy)
        "sum" | "mean" | "prod" | "softmax" | "log_softmax" | "layer_norm" | "batch_norm"
        | "group_norm" | "rms_norm" | "cross_entropy" | "mse_loss" | "bce_with_logits" => {
            AutocastCategory::FullPrecision
        }
        // Everything else: passthrough
        _ => AutocastCategory::Passthrough,
    }
}

/// Check if the current autocast context wants this op to run in reduced precision.
pub fn should_cast_to_reduced(op_name: &str) -> bool {
    is_autocast_enabled() && autocast_category(op_name) == AutocastCategory::ReducedPrecision
}

/// Check if the current autocast context wants this op to stay in full precision.
///
/// This function will be used once f16 tensor storage is added, at which point
/// ops in the `FullPrecision` category will cast f16 inputs up to f32 before
/// computing. Until then, all tensors are already f32 so the upcast is a no-op.
pub fn should_keep_full_precision(op_name: &str) -> bool {
    is_autocast_enabled() && autocast_category(op_name) == AutocastCategory::FullPrecision
}

/// Primary entry point for op implementations to query autocast policy.
///
/// Returns the [`AutocastCategory`] for `op_name`. Forward ops should branch
/// on this: when the result is [`AutocastCategory::ReducedPrecision`] and the
/// tensor is on a GPU in f32, the op should use a reduced-precision matmul
/// path (e.g., `backend.matmul_f16_f32()`) instead of `backend.matmul_f32()`.
///
/// When [`set_autocast_debug`](super::autocast::set_autocast_debug) is
/// enabled, this also records the decision to the per-thread event log for
/// test inspection. In production, no events are recorded.
///
/// Returns `None` when autocast is not enabled.
pub fn autocast_guard(op_name: &str) -> Option<AutocastCategory> {
    if !is_autocast_enabled() {
        return None;
    }
    let cat = autocast_category(op_name);
    if is_autocast_debug() {
        AUTOCAST_EVENTS.with(|events| {
            events.borrow_mut().push(AutocastEvent {
                op: op_name.to_owned(),
                category: cat,
            });
        });
    }
    Some(cat)
}

/// Log that an autocast decision was made (for testing/debugging).
///
/// Returns `Some(category)` when autocast is active, `None` otherwise.
///
/// Prefer [`autocast_guard`] in new code; this function exists for backward
/// compatibility.
pub fn autocast_log(op_name: &str) -> Option<AutocastCategory> {
    autocast_guard(op_name)
}

// ---------------------------------------------------------------------------
// Debug event log (gated behind AUTOCAST_DEBUG flag)
// ---------------------------------------------------------------------------

/// A recorded autocast decision, for test/debug inspection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutocastEvent {
    pub op: String,
    pub category: AutocastCategory,
}

thread_local! {
    static AUTOCAST_EVENTS: std::cell::RefCell<Vec<AutocastEvent>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

/// Drain and return all recorded autocast events on this thread.
///
/// Only meaningful when [`set_autocast_debug`](super::autocast::set_autocast_debug)
/// has been called with `true`.
pub fn drain_autocast_events() -> Vec<AutocastEvent> {
    AUTOCAST_EVENTS.with(|events| events.borrow_mut().drain(..).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::autocast::{AutocastDtype, autocast};

    // -------------------------------------------------------------------
    // Category classification
    // -------------------------------------------------------------------

    #[test]
    fn test_mm_is_reduced_precision() {
        assert_eq!(autocast_category("mm"), AutocastCategory::ReducedPrecision);
    }

    #[test]
    fn test_matmul_is_reduced_precision() {
        assert_eq!(
            autocast_category("matmul"),
            AutocastCategory::ReducedPrecision
        );
    }

    #[test]
    fn test_bmm_is_reduced_precision() {
        assert_eq!(autocast_category("bmm"), AutocastCategory::ReducedPrecision);
    }

    #[test]
    fn test_linear_is_reduced_precision() {
        assert_eq!(
            autocast_category("linear"),
            AutocastCategory::ReducedPrecision
        );
    }

    #[test]
    fn test_conv2d_is_reduced_precision() {
        assert_eq!(
            autocast_category("conv2d"),
            AutocastCategory::ReducedPrecision
        );
    }

    #[test]
    fn test_softmax_is_full_precision() {
        assert_eq!(
            autocast_category("softmax"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_log_softmax_is_full_precision() {
        assert_eq!(
            autocast_category("log_softmax"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_layer_norm_is_full_precision() {
        assert_eq!(
            autocast_category("layer_norm"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_batch_norm_is_full_precision() {
        assert_eq!(
            autocast_category("batch_norm"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_cross_entropy_is_full_precision() {
        assert_eq!(
            autocast_category("cross_entropy"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_mse_loss_is_full_precision() {
        assert_eq!(
            autocast_category("mse_loss"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_sum_is_full_precision() {
        assert_eq!(autocast_category("sum"), AutocastCategory::FullPrecision);
    }

    #[test]
    fn test_mean_is_full_precision() {
        assert_eq!(autocast_category("mean"), AutocastCategory::FullPrecision);
    }

    #[test]
    fn test_add_is_passthrough() {
        assert_eq!(autocast_category("add"), AutocastCategory::Passthrough);
    }

    #[test]
    fn test_mul_is_passthrough() {
        assert_eq!(autocast_category("mul"), AutocastCategory::Passthrough);
    }

    #[test]
    fn test_relu_is_passthrough() {
        assert_eq!(autocast_category("relu"), AutocastCategory::Passthrough);
    }

    #[test]
    fn test_unknown_op_is_passthrough() {
        assert_eq!(
            autocast_category("some_custom_op"),
            AutocastCategory::Passthrough
        );
    }

    // -------------------------------------------------------------------
    // should_cast_to_reduced
    // -------------------------------------------------------------------

    #[test]
    fn test_should_cast_to_reduced_false_when_disabled() {
        // Outside any autocast context, autocast is disabled.
        assert!(!should_cast_to_reduced("mm"));
        assert!(!should_cast_to_reduced("matmul"));
        assert!(!should_cast_to_reduced("linear"));
    }

    #[test]
    fn test_should_cast_to_reduced_true_for_mm_when_enabled() {
        autocast(AutocastDtype::F16, || {
            assert!(should_cast_to_reduced("mm"));
            assert!(should_cast_to_reduced("matmul"));
            assert!(should_cast_to_reduced("linear"));
            assert!(should_cast_to_reduced("conv2d"));
        });
    }

    #[test]
    fn test_should_cast_to_reduced_false_for_passthrough_when_enabled() {
        autocast(AutocastDtype::F16, || {
            assert!(!should_cast_to_reduced("add"));
            assert!(!should_cast_to_reduced("relu"));
        });
    }

    #[test]
    fn test_should_cast_to_reduced_false_for_full_precision_when_enabled() {
        autocast(AutocastDtype::F16, || {
            assert!(!should_cast_to_reduced("softmax"));
            assert!(!should_cast_to_reduced("layer_norm"));
        });
    }

    // -------------------------------------------------------------------
    // should_keep_full_precision
    // -------------------------------------------------------------------

    #[test]
    fn test_should_keep_full_precision_false_when_disabled() {
        assert!(!should_keep_full_precision("softmax"));
        assert!(!should_keep_full_precision("layer_norm"));
    }

    #[test]
    fn test_should_keep_full_precision_true_when_enabled() {
        autocast(AutocastDtype::BF16, || {
            assert!(should_keep_full_precision("softmax"));
            assert!(should_keep_full_precision("layer_norm"));
            assert!(should_keep_full_precision("cross_entropy"));
        });
    }

    #[test]
    fn test_should_keep_full_precision_false_for_reduced_when_enabled() {
        autocast(AutocastDtype::F16, || {
            assert!(!should_keep_full_precision("mm"));
            assert!(!should_keep_full_precision("matmul"));
        });
    }

    // -------------------------------------------------------------------
    // autocast_log
    // -------------------------------------------------------------------

    #[test]
    fn test_autocast_log_none_when_disabled() {
        assert!(autocast_log("mm").is_none());
        assert!(autocast_log("softmax").is_none());
        assert!(autocast_log("add").is_none());
    }

    #[test]
    fn test_autocast_log_returns_category_when_enabled() {
        autocast(AutocastDtype::F16, || {
            assert_eq!(autocast_log("mm"), Some(AutocastCategory::ReducedPrecision));
            assert_eq!(
                autocast_log("softmax"),
                Some(AutocastCategory::FullPrecision)
            );
            assert_eq!(autocast_log("add"), Some(AutocastCategory::Passthrough));
        });
    }

    // -------------------------------------------------------------------
    // Context boundary: policy active inside, inactive outside
    // -------------------------------------------------------------------

    #[test]
    fn test_policy_active_inside_context_inactive_outside() {
        // Before: disabled.
        assert!(!should_cast_to_reduced("mm"));
        assert!(autocast_log("mm").is_none());

        autocast(AutocastDtype::F16, || {
            // Inside: enabled.
            assert!(should_cast_to_reduced("mm"));
            assert!(autocast_log("mm").is_some());
        });

        // After: disabled again.
        assert!(!should_cast_to_reduced("mm"));
        assert!(autocast_log("mm").is_none());
    }

    #[test]
    fn test_nested_autocast_policy_still_works() {
        autocast(AutocastDtype::F16, || {
            assert!(should_cast_to_reduced("mm"));

            autocast(AutocastDtype::BF16, || {
                // Still enabled in the inner context.
                assert!(should_cast_to_reduced("mm"));
                assert!(should_keep_full_precision("softmax"));
            });

            // Restored to outer context, still enabled.
            assert!(should_cast_to_reduced("mm"));
        });

        // Outside all contexts: disabled.
        assert!(!should_cast_to_reduced("mm"));
    }

    // -------------------------------------------------------------------
    // autocast_guard
    // -------------------------------------------------------------------

    #[test]
    fn test_autocast_guard_none_when_disabled() {
        assert!(autocast_guard("mm").is_none());
        assert!(autocast_guard("softmax").is_none());
    }

    #[test]
    fn test_autocast_guard_returns_category() {
        autocast(AutocastDtype::F16, || {
            assert_eq!(
                autocast_guard("mm"),
                Some(AutocastCategory::ReducedPrecision)
            );
            assert_eq!(
                autocast_guard("softmax"),
                Some(AutocastCategory::FullPrecision)
            );
            assert_eq!(autocast_guard("add"), Some(AutocastCategory::Passthrough));
        });
    }

    #[test]
    fn test_autocast_guard_debug_events() {
        use crate::autograd::autocast::set_autocast_debug;

        // Drain any stale events from other tests.
        drain_autocast_events();

        set_autocast_debug(true);
        let events = autocast(AutocastDtype::F16, || {
            autocast_guard("mm");
            autocast_guard("softmax");
            autocast_guard("relu");
            // Drain inside the closure to avoid race with other tests
            // that may toggle the debug flag.
            drain_autocast_events()
        });
        set_autocast_debug(false);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].op, "mm");
        assert_eq!(events[0].category, AutocastCategory::ReducedPrecision);
        assert_eq!(events[1].op, "softmax");
        assert_eq!(events[1].category, AutocastCategory::FullPrecision);
        assert_eq!(events[2].op, "relu");
        assert_eq!(events[2].category, AutocastCategory::Passthrough);
    }

    #[test]
    fn test_autocast_guard_no_events_without_debug() {
        use crate::autograd::autocast::set_autocast_debug;

        drain_autocast_events();
        set_autocast_debug(false);
        autocast(AutocastDtype::F16, || {
            autocast_guard("mm");
            autocast_guard("linear");
        });
        let events = drain_autocast_events();
        assert!(
            events.is_empty(),
            "no events should be recorded when debug is off"
        );
    }

    // -------------------------------------------------------------------
    // addmm / einsum entries (not wired yet, but classified)
    // -------------------------------------------------------------------

    #[test]
    fn test_addmm_is_reduced_precision() {
        assert_eq!(
            autocast_category("addmm"),
            AutocastCategory::ReducedPrecision
        );
    }

    #[test]
    fn test_einsum_is_reduced_precision() {
        assert_eq!(
            autocast_category("einsum"),
            AutocastCategory::ReducedPrecision
        );
    }
}
