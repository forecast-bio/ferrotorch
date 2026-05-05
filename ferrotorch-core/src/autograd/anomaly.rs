// [CL-311] Anomaly detection mode for the autograd engine.
//
// When enabled, every tensor created by a differentiable operation stores
// a backtrace at construction time. If the backward pass encounters an
// error (e.g., NaN gradient, shape mismatch), the stored backtrace is
// printed alongside the error message so the user can see exactly which
// forward-pass operation produced the problematic node.
//
// Follows PyTorch's `torch.autograd.set_detect_anomaly(True)` semantics.

use std::cell::Cell;
use std::fmt;

thread_local! {
    static ANOMALY_ENABLED: Cell<bool> = const { Cell::new(false) };
}

/// Global anomaly detection mode.
///
/// When enabled:
/// - Forward operations capture a `std::backtrace::Backtrace` and store it
///   on the resulting tensor's metadata.
/// - The backward pass checks gradients for NaN/Inf and reports the stored
///   backtrace to help locate the source of numerical issues.
///
/// Anomaly mode has significant runtime overhead (backtrace capture is slow),
/// so it should only be enabled during debugging.
#[derive(Debug)]
pub struct AnomalyMode;

impl AnomalyMode {
    /// Enable anomaly detection on the current thread.
    pub fn enable() {
        ANOMALY_ENABLED.with(|c| c.set(true));
    }

    /// Disable anomaly detection on the current thread.
    pub fn disable() {
        ANOMALY_ENABLED.with(|c| c.set(false));
    }

    /// Returns `true` if anomaly detection is currently enabled.
    pub fn is_enabled() -> bool {
        ANOMALY_ENABLED.with(|c| c.get())
    }
}

/// Execute a closure with anomaly detection enabled.
///
/// The previous anomaly-detection state is restored when the closure returns
/// (even on panic), matching the RAII pattern used by `no_grad` / `enable_grad`.
///
/// # Example
///
/// ```
/// use ferrotorch_core::autograd::anomaly::detect_anomaly;
///
/// detect_anomaly(|| {
///     // All forward ops here capture backtraces.
/// });
/// ```
pub fn detect_anomaly<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    struct AnomalyGuard {
        prev: bool,
    }
    impl Drop for AnomalyGuard {
        fn drop(&mut self) {
            ANOMALY_ENABLED.with(|c| c.set(self.prev));
        }
    }
    let _guard = AnomalyGuard {
        prev: AnomalyMode::is_enabled(),
    };
    AnomalyMode::enable();
    f()
}

/// A captured forward-pass backtrace, stored on tensors when anomaly mode is on.
///
/// Uses `std::backtrace::Backtrace` under the hood. Display formatting
/// produces a human-readable stack trace.
#[derive(Clone)]
pub struct ForwardBacktrace {
    trace: String,
}

impl ForwardBacktrace {
    /// Capture the current backtrace if anomaly mode is enabled.
    ///
    /// Returns `None` if anomaly mode is off (zero overhead in the common case).
    pub fn capture_if_enabled() -> Option<Self> {
        if !AnomalyMode::is_enabled() {
            return None;
        }
        // Use std::backtrace::Backtrace which is stable since Rust 1.65.
        let bt = std::backtrace::Backtrace::capture();
        Some(Self {
            trace: bt.to_string(),
        })
    }

    /// Get the backtrace string.
    pub fn trace(&self) -> &str {
        &self.trace
    }
}

impl fmt::Debug for ForwardBacktrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ForwardBacktrace")
            .field("trace", &"<backtrace>")
            .finish()
    }
}

impl fmt::Display for ForwardBacktrace {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Forward-pass backtrace:\n{}", self.trace)
    }
}

/// Check a gradient tensor for NaN or Inf values (anomaly check).
///
/// Called during backward when anomaly mode is enabled. If the gradient
/// contains NaN or Inf, returns an error message including the
/// `forward_backtrace` (if available) from the tensor that produced
/// this gradient.
pub fn check_gradient_anomaly<T: crate::dtype::Float>(
    grad: &crate::tensor::Tensor<T>,
    op_name: &str,
    forward_bt: Option<&ForwardBacktrace>,
) -> crate::error::FerrotorchResult<()> {
    // Only check when anomaly mode is on — this function should only be
    // called in that case, but guard defensively.
    if !AnomalyMode::is_enabled() {
        return Ok(());
    }

    // For GPU tensors we skip the check (would require a D2H transfer
    // just for validation). Users can call .cpu() in a hook if needed.
    if grad.is_cuda() {
        return Ok(());
    }

    let data = grad.data()?;
    let has_nan = data.iter().any(|v| v.is_nan());
    let has_inf = data.iter().any(|v| v.is_infinite());

    if has_nan || has_inf {
        let anomaly_kind = if has_nan && has_inf {
            "NaN and Inf"
        } else if has_nan {
            "NaN"
        } else {
            "Inf"
        };

        let bt_msg = match forward_bt {
            Some(bt) => format!("\n\n{bt}"),
            None => String::from(
                "\n\n(no forward backtrace available — was anomaly mode enabled during forward pass?)",
            ),
        };

        return Err(crate::error::FerrotorchError::InvalidArgument {
            message: format!("anomaly detected: {anomaly_kind} in gradient of {op_name}{bt_msg}"),
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anomaly_mode_default_off() {
        // Reset in case another test left it on.
        AnomalyMode::disable();
        assert!(!AnomalyMode::is_enabled());
    }

    #[test]
    fn test_anomaly_mode_enable_disable() {
        AnomalyMode::enable();
        assert!(AnomalyMode::is_enabled());
        AnomalyMode::disable();
        assert!(!AnomalyMode::is_enabled());
    }

    #[test]
    fn test_detect_anomaly_scoped() {
        AnomalyMode::disable();
        assert!(!AnomalyMode::is_enabled());
        detect_anomaly(|| {
            assert!(AnomalyMode::is_enabled());
        });
        assert!(!AnomalyMode::is_enabled());
    }

    #[test]
    fn test_detect_anomaly_panic_safety() {
        AnomalyMode::disable();
        let result = std::panic::catch_unwind(|| {
            detect_anomaly(|| {
                assert!(AnomalyMode::is_enabled());
                panic!("intentional panic inside detect_anomaly");
            });
        });
        assert!(result.is_err());
        assert!(!AnomalyMode::is_enabled());
    }

    #[test]
    fn test_detect_anomaly_nested() {
        AnomalyMode::disable();
        detect_anomaly(|| {
            assert!(AnomalyMode::is_enabled());
            detect_anomaly(|| {
                assert!(AnomalyMode::is_enabled());
            });
            // Still enabled — inner detect_anomaly restores to "enabled"
            // (the outer's state).
            assert!(AnomalyMode::is_enabled());
        });
        assert!(!AnomalyMode::is_enabled());
    }

    #[test]
    fn test_forward_backtrace_capture_when_disabled() {
        AnomalyMode::disable();
        assert!(ForwardBacktrace::capture_if_enabled().is_none());
    }

    #[test]
    fn test_forward_backtrace_capture_when_enabled() {
        AnomalyMode::enable();
        let bt = ForwardBacktrace::capture_if_enabled();
        AnomalyMode::disable();
        assert!(bt.is_some());
        // The trace should contain something — at minimum our test function name.
        assert!(!bt.unwrap().trace().is_empty());
    }

    #[test]
    fn test_check_gradient_anomaly_clean() {
        use crate::storage::TensorStorage;
        use crate::tensor::Tensor;

        AnomalyMode::enable();
        let grad =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3], false)
                .unwrap();
        let result = check_gradient_anomaly(&grad, "TestOp", None);
        AnomalyMode::disable();
        assert!(result.is_ok());
    }

    #[test]
    fn test_check_gradient_anomaly_nan() {
        use crate::storage::TensorStorage;
        use crate::tensor::Tensor;

        AnomalyMode::enable();
        let grad = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0, f32::NAN, 3.0]),
            vec![3],
            false,
        )
        .unwrap();
        let result = check_gradient_anomaly(&grad, "TestOp", None);
        AnomalyMode::disable();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("NaN"));
        assert!(msg.contains("TestOp"));
    }

    #[test]
    fn test_check_gradient_anomaly_inf() {
        use crate::storage::TensorStorage;
        use crate::tensor::Tensor;

        AnomalyMode::enable();
        let grad = Tensor::<f32>::from_storage(
            TensorStorage::cpu(vec![1.0, f32::INFINITY, 3.0]),
            vec![3],
            false,
        )
        .unwrap();
        let result = check_gradient_anomaly(&grad, "TestOp", None);
        AnomalyMode::disable();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("Inf"));
    }

    #[test]
    fn test_check_gradient_anomaly_with_backtrace() {
        use crate::storage::TensorStorage;
        use crate::tensor::Tensor;

        AnomalyMode::enable();
        let bt = ForwardBacktrace::capture_if_enabled().unwrap();
        let grad =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![f32::NAN]), vec![], false).unwrap();
        let result = check_gradient_anomaly(&grad, "BadOp", Some(&bt));
        AnomalyMode::disable();
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(msg.contains("Forward-pass backtrace"));
    }

    #[test]
    fn test_check_gradient_anomaly_skipped_when_disabled() {
        use crate::storage::TensorStorage;
        use crate::tensor::Tensor;

        AnomalyMode::disable();
        // Even with NaN, should pass because anomaly mode is off.
        let grad =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![f32::NAN]), vec![], false).unwrap();
        assert!(check_gradient_anomaly(&grad, "TestOp", None).is_ok());
    }
}
