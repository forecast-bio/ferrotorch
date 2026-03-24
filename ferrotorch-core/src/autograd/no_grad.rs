use std::cell::Cell;

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

/// Returns `true` if gradient tracking is currently enabled on this thread.
pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Execute a closure with gradient tracking disabled.
///
/// Tensors created inside the closure will have `requires_grad = false`
/// regardless of their inputs. This is used for inference and for
/// manually updating parameters (e.g., SGD step) without recording
/// the update in the computation graph.
///
/// Calls can be nested safely — the outermost `no_grad` restores the
/// previous state.
///
/// # Panic safety
///
/// The previous gradient-enabled state is restored via an RAII drop guard,
/// so it is correctly restored even if `f` panics.
///
/// # Example
///
/// ```
/// use ferrotorch_core::autograd::no_grad;
///
/// no_grad(|| {
///     // All tensor operations here are untracked.
/// });
/// ```
pub fn no_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    struct NoGradGuard {
        prev: bool,
    }
    impl Drop for NoGradGuard {
        fn drop(&mut self) {
            GRAD_ENABLED.with(|g| g.set(self.prev));
        }
    }
    let _guard = NoGradGuard {
        prev: is_grad_enabled(),
    };
    GRAD_ENABLED.with(|g| g.set(false));
    f()
}

/// Re-enable gradient computation inside a `no_grad` block.
///
/// This is useful for gradient checkpointing where you need to
/// re-enable gradients for recomputation inside a `no_grad` context.
///
/// # Panic safety
///
/// The previous gradient-enabled state is restored via an RAII drop guard,
/// so it is correctly restored even if `f` panics.
///
/// # Example
///
/// ```
/// use ferrotorch_core::autograd::no_grad::{no_grad, enable_grad, is_grad_enabled};
///
/// no_grad(|| {
///     assert!(!is_grad_enabled());
///     enable_grad(|| {
///         assert!(is_grad_enabled());
///     });
///     assert!(!is_grad_enabled());
/// });
/// ```
pub fn enable_grad<F, R>(f: F) -> R
where
    F: FnOnce() -> R,
{
    struct EnableGradGuard {
        prev: bool,
    }
    impl Drop for EnableGradGuard {
        fn drop(&mut self) {
            GRAD_ENABLED.with(|g| g.set(self.prev));
        }
    }
    let _guard = EnableGradGuard {
        prev: is_grad_enabled(),
    };
    GRAD_ENABLED.with(|g| g.set(true));
    f()
}

/// Programmatically set whether gradients are enabled.
///
/// Prefer [`no_grad`] or [`enable_grad`] over this function, as they
/// correctly restore the previous state. This function is provided for
/// cases where a closure-based API is inconvenient (e.g., C FFI).
pub fn set_grad_enabled(enabled: bool) {
    GRAD_ENABLED.with(|cell| cell.set(enabled));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_grad_enabled_default() {
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_disables() {
        assert!(is_grad_enabled());
        no_grad(|| {
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_nested() {
        assert!(is_grad_enabled());
        no_grad(|| {
            assert!(!is_grad_enabled());
            no_grad(|| {
                assert!(!is_grad_enabled());
            });
            // Inner no_grad restores to false (the outer no_grad's state).
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_enable_grad_inside_no_grad() {
        no_grad(|| {
            assert!(!is_grad_enabled());
            enable_grad(|| {
                assert!(is_grad_enabled());
            });
            // Restored to false after enable_grad returns.
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_enable_grad_returns_value() {
        let val = no_grad(|| {
            enable_grad(|| 42)
        });
        assert_eq!(val, 42);
    }

    #[test]
    fn test_enable_grad_when_already_enabled() {
        assert!(is_grad_enabled());
        let result = enable_grad(|| {
            assert!(is_grad_enabled());
            99
        });
        assert_eq!(result, 99);
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_set_grad_enabled() {
        assert!(is_grad_enabled());
        set_grad_enabled(false);
        assert!(!is_grad_enabled());
        set_grad_enabled(true);
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_set_grad_enabled_inside_no_grad() {
        no_grad(|| {
            assert!(!is_grad_enabled());
            set_grad_enabled(true);
            assert!(is_grad_enabled());
            set_grad_enabled(false);
            assert!(!is_grad_enabled());
        });
        // no_grad restores the previous state (true).
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_no_grad_panic_safety() {
        assert!(is_grad_enabled());
        let result = std::panic::catch_unwind(|| {
            no_grad(|| {
                assert!(!is_grad_enabled());
                panic!("intentional panic inside no_grad");
            });
        });
        assert!(result.is_err());
        // RAII guard must have restored grad_enabled to true.
        assert!(is_grad_enabled());
    }

    #[test]
    fn test_enable_grad_panic_safety() {
        no_grad(|| {
            assert!(!is_grad_enabled());
            let result = std::panic::catch_unwind(|| {
                enable_grad(|| {
                    assert!(is_grad_enabled());
                    panic!("intentional panic inside enable_grad");
                });
            });
            assert!(result.is_err());
            // RAII guard must have restored grad_enabled to false.
            assert!(!is_grad_enabled());
        });
        assert!(is_grad_enabled());
    }
}
