//! Autocast operation categorization and policy engine.
//!
//! This module classifies operations into categories that determine how they
//! behave under mixed-precision autocast:
//!
//! - **`ReducedPrecision`**: operations that benefit from f16/bf16 (matmul, conv, linear).
//! - **`FullPrecision`**: operations that need f32 for numerical accuracy (reductions, norms, loss).
//! - **`Passthrough`**: elementwise ops that follow whatever dtype their inputs have.
//!
//! When autocast is enabled via [`super::autocast::autocast`], forward ops call
//! [`autocast_guard`] to record the policy decision for each operation. Once
//! f16/bf16 tensor storage lands, the guard will perform the actual dtype cast;
//! until then it records events into a thread-local log that tests can drain
//! via [`drain_autocast_events`].

use std::cell::RefCell;

use super::autocast::is_autocast_enabled;

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

/// A recorded autocast policy decision from [`autocast_guard`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AutocastEvent {
    /// The operation name passed to `autocast_guard`.
    pub op_name: String,
    /// The category the operation was classified into.
    pub category: AutocastCategory,
}

thread_local! {
    static AUTOCAST_EVENTS: RefCell<Vec<AutocastEvent>> = const { RefCell::new(Vec::new()) };
}

/// Get the autocast category for an operation.
///
/// Reduced-precision ops are those where tensor cores / half-precision ALUs
/// provide a speedup with minimal accuracy loss (matmul, convolution, linear,
/// addmm, einsum). Full-precision ops are numerically sensitive and must stay
/// in f32 (reductions, norms, loss functions, softmax). Everything else is
/// passthrough and follows the dtype of its inputs.
pub fn autocast_category(op_name: &str) -> AutocastCategory {
    match op_name {
        // Reduced precision (benefits from f16/bf16)
        "mm" | "matmul" | "bmm" | "linear" | "conv1d" | "conv2d" | "conv_transpose2d"
        | "addmm" | "einsum" => AutocastCategory::ReducedPrecision,
        // Full precision (needs f32 for accuracy)
        "sum" | "mean" | "prod" | "softmax" | "log_softmax" | "layer_norm" | "batch_norm"
        | "group_norm" | "rms_norm" | "cross_entropy" | "mse_loss" | "binary_cross_entropy"
        | "bce_with_logits" => AutocastCategory::FullPrecision,
        // Everything else: passthrough
        _ => AutocastCategory::Passthrough,
    }
}

/// Check if the current autocast context wants this op to run in reduced precision.
pub fn should_cast_to_reduced(op_name: &str) -> bool {
    is_autocast_enabled() && autocast_category(op_name) == AutocastCategory::ReducedPrecision
}

/// Check if the current autocast context wants this op to stay in full precision.
pub fn should_keep_full_precision(op_name: &str) -> bool {
    is_autocast_enabled() && autocast_category(op_name) == AutocastCategory::FullPrecision
}

/// Log that an autocast decision was made (for testing/debugging).
///
/// Returns `Some(category)` when autocast is active, `None` otherwise.
/// In a real f16 implementation, this would cast the tensor to/from the
/// reduced-precision dtype; for now it records the policy decision.
pub fn autocast_log(op_name: &str) -> Option<AutocastCategory> {
    if is_autocast_enabled() {
        Some(autocast_category(op_name))
    } else {
        None
    }
}

/// Guard that forward ops call to participate in autocast.
///
/// When autocast is **disabled**, this is a no-op and returns immediately.
///
/// When autocast is **enabled**, the guard classifies `op_name` into its
/// [`AutocastCategory`] and records an [`AutocastEvent`] in the thread-local
/// event log. Tests can drain these events with [`drain_autocast_events`].
///
/// Once f16/bf16 tensor storage is implemented, the `ReducedPrecision` branch
/// will cast inputs to the target reduced-precision dtype and the
/// `FullPrecision` branch will ensure inputs are promoted back to f32. Until
/// then, the guard validates the policy wiring without performing actual casts.
///
/// Passthrough ops do not generate events since they require no dtype action.
pub fn autocast_guard(op_name: &str) {
    if !is_autocast_enabled() {
        return;
    }
    let category = autocast_category(op_name);
    // Passthrough ops need no action — skip event recording.
    if category == AutocastCategory::Passthrough {
        return;
    }
    AUTOCAST_EVENTS.with(|events| {
        events.borrow_mut().push(AutocastEvent {
            op_name: op_name.to_owned(),
            category,
        });
    });
}

/// Drain all recorded autocast events from the thread-local log.
///
/// Returns the events accumulated since the last drain (or since the thread
/// started). This is the primary mechanism for tests to verify that forward
/// ops correctly call [`autocast_guard`].
pub fn drain_autocast_events() -> Vec<AutocastEvent> {
    AUTOCAST_EVENTS.with(|events| events.borrow_mut().drain(..).collect())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::autograd::autocast::{autocast, AutocastDtype};

    // -------------------------------------------------------------------
    // Category classification
    // -------------------------------------------------------------------

    #[test]
    fn test_mm_is_reduced_precision() {
        assert_eq!(
            autocast_category("mm"),
            AutocastCategory::ReducedPrecision
        );
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
        assert_eq!(
            autocast_category("bmm"),
            AutocastCategory::ReducedPrecision
        );
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

    #[test]
    fn test_conv1d_is_reduced_precision() {
        assert_eq!(
            autocast_category("conv1d"),
            AutocastCategory::ReducedPrecision
        );
    }

    #[test]
    fn test_conv_transpose2d_is_reduced_precision() {
        assert_eq!(
            autocast_category("conv_transpose2d"),
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
    fn test_group_norm_is_full_precision() {
        assert_eq!(
            autocast_category("group_norm"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_rms_norm_is_full_precision() {
        assert_eq!(
            autocast_category("rms_norm"),
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
    fn test_binary_cross_entropy_is_full_precision() {
        assert_eq!(
            autocast_category("binary_cross_entropy"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_bce_with_logits_is_full_precision() {
        assert_eq!(
            autocast_category("bce_with_logits"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_sum_is_full_precision() {
        assert_eq!(
            autocast_category("sum"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_mean_is_full_precision() {
        assert_eq!(
            autocast_category("mean"),
            AutocastCategory::FullPrecision
        );
    }

    #[test]
    fn test_prod_is_full_precision() {
        assert_eq!(
            autocast_category("prod"),
            AutocastCategory::FullPrecision
        );
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
    fn test_should_cast_to_reduced_true_for_new_ops_when_enabled() {
        autocast(AutocastDtype::F16, || {
            assert!(should_cast_to_reduced("addmm"));
            assert!(should_cast_to_reduced("einsum"));
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
    fn test_should_keep_full_precision_true_for_new_ops_when_enabled() {
        autocast(AutocastDtype::BF16, || {
            assert!(should_keep_full_precision("binary_cross_entropy"));
            assert!(should_keep_full_precision("bce_with_logits"));
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
    // autocast_guard — no-op when disabled
    // -------------------------------------------------------------------

    #[test]
    fn test_autocast_guard_noop_when_disabled() {
        // Drain any leftover events from prior tests.
        drain_autocast_events();

        // Call guard outside any autocast context.
        autocast_guard("mm");
        autocast_guard("softmax");
        autocast_guard("add");

        // No events should be recorded.
        let events = drain_autocast_events();
        assert!(events.is_empty(), "expected no events when autocast is disabled, got {events:?}");
    }

    // -------------------------------------------------------------------
    // autocast_guard — logs events when enabled
    // -------------------------------------------------------------------

    #[test]
    fn test_autocast_guard_logs_reduced_precision() {
        drain_autocast_events();

        autocast(AutocastDtype::F16, || {
            autocast_guard("mm");
            autocast_guard("matmul");
            autocast_guard("linear");
        });

        let events = drain_autocast_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0], AutocastEvent { op_name: "mm".into(), category: AutocastCategory::ReducedPrecision });
        assert_eq!(events[1], AutocastEvent { op_name: "matmul".into(), category: AutocastCategory::ReducedPrecision });
        assert_eq!(events[2], AutocastEvent { op_name: "linear".into(), category: AutocastCategory::ReducedPrecision });
    }

    #[test]
    fn test_autocast_guard_logs_full_precision() {
        drain_autocast_events();

        autocast(AutocastDtype::BF16, || {
            autocast_guard("softmax");
            autocast_guard("cross_entropy");
            autocast_guard("mse_loss");
        });

        let events = drain_autocast_events();
        assert_eq!(events.len(), 3);
        assert_eq!(events[0].op_name, "softmax");
        assert_eq!(events[0].category, AutocastCategory::FullPrecision);
        assert_eq!(events[1].op_name, "cross_entropy");
        assert_eq!(events[1].category, AutocastCategory::FullPrecision);
        assert_eq!(events[2].op_name, "mse_loss");
        assert_eq!(events[2].category, AutocastCategory::FullPrecision);
    }

    #[test]
    fn test_autocast_guard_skips_passthrough() {
        drain_autocast_events();

        autocast(AutocastDtype::F16, || {
            autocast_guard("add");
            autocast_guard("relu");
            autocast_guard("some_custom_op");
        });

        // Passthrough ops should NOT generate events.
        let events = drain_autocast_events();
        assert!(events.is_empty(), "passthrough ops should not generate events, got {events:?}");
    }

    #[test]
    fn test_autocast_guard_events_cleared_by_drain() {
        drain_autocast_events();

        autocast(AutocastDtype::F16, || {
            autocast_guard("mm");
        });

        let first = drain_autocast_events();
        assert_eq!(first.len(), 1);

        // Second drain should be empty.
        let second = drain_autocast_events();
        assert!(second.is_empty());
    }

    #[test]
    fn test_autocast_guard_new_reduced_ops() {
        drain_autocast_events();

        autocast(AutocastDtype::F16, || {
            autocast_guard("addmm");
            autocast_guard("einsum");
        });

        let events = drain_autocast_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], AutocastEvent { op_name: "addmm".into(), category: AutocastCategory::ReducedPrecision });
        assert_eq!(events[1], AutocastEvent { op_name: "einsum".into(), category: AutocastCategory::ReducedPrecision });
    }

    #[test]
    fn test_autocast_guard_new_full_precision_ops() {
        drain_autocast_events();

        autocast(AutocastDtype::BF16, || {
            autocast_guard("binary_cross_entropy");
            autocast_guard("bce_with_logits");
        });

        let events = drain_autocast_events();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], AutocastEvent { op_name: "binary_cross_entropy".into(), category: AutocastCategory::FullPrecision });
        assert_eq!(events[1], AutocastEvent { op_name: "bce_with_logits".into(), category: AutocastCategory::FullPrecision });
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
    // autocast_guard inside nested contexts
    // -------------------------------------------------------------------

    #[test]
    fn test_autocast_guard_nested_contexts() {
        drain_autocast_events();

        autocast(AutocastDtype::F16, || {
            autocast_guard("mm");

            autocast(AutocastDtype::BF16, || {
                autocast_guard("linear");
                autocast_guard("softmax");
            });

            autocast_guard("bmm");
        });

        // Guard outside: no-op.
        autocast_guard("mm");

        let events = drain_autocast_events();
        assert_eq!(events.len(), 4);
        assert_eq!(events[0].op_name, "mm");
        assert_eq!(events[0].category, AutocastCategory::ReducedPrecision);
        assert_eq!(events[1].op_name, "linear");
        assert_eq!(events[1].category, AutocastCategory::ReducedPrecision);
        assert_eq!(events[2].op_name, "softmax");
        assert_eq!(events[2].category, AutocastCategory::FullPrecision);
        assert_eq!(events[3].op_name, "bmm");
        assert_eq!(events[3].category, AutocastCategory::ReducedPrecision);
    }
}
