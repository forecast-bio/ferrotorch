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
//! let individual ops query whether they should cast their inputs. The actual
//! dtype casting requires f16 tensor support (a larger change); this module
//! provides the policy engine and category classification that will drive it.

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

/// Get the autocast category for an operation.
pub fn autocast_category(op_name: &str) -> AutocastCategory {
    match op_name {
        // Reduced precision (benefits from f16/bf16)
        "mm" | "matmul" | "bmm" | "linear" | "conv1d" | "conv2d" | "conv_transpose2d" => {
            AutocastCategory::ReducedPrecision
        }
        // Full precision (needs f32 for accuracy)
        "sum" | "mean" | "prod" | "softmax" | "log_softmax" | "layer_norm" | "batch_norm"
        | "group_norm" | "rms_norm" | "cross_entropy" | "mse_loss" => {
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
}
