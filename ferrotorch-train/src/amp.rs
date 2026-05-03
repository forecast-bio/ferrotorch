//! Automatic mixed precision (AMP) training utilities.
//!
//! This module ties together the autocast context from
//! [`ferrotorch_core::autograd::autocast`] and the
//! [`GradScaler`](ferrotorch_optim::GradScaler) into a convenient API for
//! mixed-precision training.
//!
//! | Item | Description |
//! |------|-------------|
//! | [`autocast`] | Re-export: run a closure with autocast enabled |
//! | [`AutocastDtype`] | Re-export: `F16` or `BF16` |
//! | [`AmpContext`] | Combined autocast + grad scaler management |
//!
//! # How autocast works
//!
//! When autocast is enabled, operations are automatically dispatched to
//! reduced-precision or full-precision paths based on their category:
//!
//! | Category | Examples | Behaviour |
//! |----------|----------|-----------|
//! | ReducedPrecision | matmul, conv, linear | Cast to f16/bf16 |
//! | FullPrecision | softmax, norms, losses | Keep in f32 |
//! | Passthrough | relu, add, mul | Follow input dtype |
//!
//! # Usage
//!
//! ```ignore
//! use ferrotorch_train::amp::{AmpContext, AutocastDtype};
//!
//! let mut amp = AmpContext::new(AutocastDtype::F16, Default::default());
//!
//! for (input, target) in batches {
//!     optimizer.zero_grad()?;
//!
//!     // Forward in autocast context.
//!     let loss = amp.autocast_forward(|| {
//!         let out = model.forward(&input)?;
//!         loss_fn(&out, &target)
//!     })?;
//!
//!     // Scale loss, backward, unscale, step, update.
//!     amp.backward_step(&loss, &mut optimizer)?;
//! }
//! ```
//!
//! [CL-334] Add gradient checkpointing, autocast context, gradient clipping, and EMA callback

// Re-export the core autocast primitives.
pub use ferrotorch_core::autograd::autocast::{AutocastDtype, autocast};
pub use ferrotorch_core::autograd::autocast::{autocast_dtype, is_autocast_enabled};
pub use ferrotorch_core::autograd::autocast_ops::{
    AutocastCategory, autocast_category, autocast_guard, should_cast_to_reduced,
    should_keep_full_precision,
};
pub use ferrotorch_optim::{GradScaler, GradScalerConfig, GradScalerState};

use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_optim::Optimizer;

// ---------------------------------------------------------------------------
// AmpContext
// ---------------------------------------------------------------------------

/// Combined autocast + gradient scaler context for mixed-precision training.
///
/// Wraps a [`GradScaler`] and an [`AutocastDtype`] into a single object
/// that manages the full AMP workflow: autocast forward pass, loss scaling,
/// backward, unscale, optimizer step, and scale update.
///
/// # Disabled mode
///
/// If the [`GradScalerConfig`] has `enabled = false`, the `AmpContext` still
/// runs the autocast context but the scaler operations are passthrough no-ops.
/// This allows the same training loop to work with or without AMP.
pub struct AmpContext<T: Float> {
    /// The target reduced-precision dtype for autocast regions.
    dtype: AutocastDtype,
    /// Dynamic loss scaler.
    scaler: GradScaler<T>,
}

impl<T: Float> AmpContext<T> {
    /// Create a new AMP context.
    ///
    /// # Arguments
    ///
    /// * `dtype` - The reduced-precision dtype to use (`F16` or `BF16`).
    /// * `scaler_config` - Configuration for the gradient scaler.
    pub fn new(dtype: AutocastDtype, scaler_config: GradScalerConfig) -> Self {
        Self {
            dtype,
            scaler: GradScaler::new(scaler_config),
        }
    }

    /// Run a closure inside an autocast context.
    ///
    /// Operations that benefit from reduced precision (matmul, conv, linear)
    /// will be dispatched to f16/bf16 paths. Operations requiring full
    /// precision (norms, softmax, losses) remain in f32.
    pub fn autocast_forward<F, R>(&self, f: F) -> R
    where
        F: FnOnce() -> R,
    {
        autocast(self.dtype, f)
    }

    /// Scale the loss, run backward, unscale gradients, step the optimizer,
    /// and update the scale factor.
    ///
    /// This is the "everything after the forward pass" part of mixed-precision
    /// training. Returns `true` if the optimizer step was actually taken
    /// (i.e. no inf/NaN in gradients), `false` if it was skipped.
    ///
    /// # Arguments
    ///
    /// * `loss` - The unscaled loss tensor from the forward pass.
    /// * `optimizer` - The optimizer to step.
    pub fn backward_step(
        &mut self,
        loss: &Tensor<T>,
        optimizer: &mut dyn Optimizer<T>,
    ) -> FerrotorchResult<bool> {
        // Scale loss and compute backward.
        let scaled_loss = self.scaler.scale(loss)?;
        scaled_loss.backward()?;

        // Unscale, step (skip if inf/NaN), update scale factor.
        let stepped = self.scaler.step(optimizer)?;
        self.scaler.update();
        optimizer.zero_grad()?;

        Ok(stepped)
    }

    /// Access the underlying grad scaler.
    pub fn scaler(&self) -> &GradScaler<T> {
        &self.scaler
    }

    /// Mutable access to the underlying grad scaler.
    pub fn scaler_mut(&mut self) -> &mut GradScaler<T> {
        &mut self.scaler
    }

    /// The autocast dtype this context uses.
    pub fn dtype(&self) -> AutocastDtype {
        self.dtype
    }

    /// Current loss scale factor.
    pub fn get_scale(&self) -> f64 {
        self.scaler.get_scale()
    }

    /// Whether the scaler is enabled.
    pub fn is_enabled(&self) -> bool {
        self.scaler.is_enabled()
    }

    /// Export the scaler state for checkpointing.
    pub fn scaler_state_dict(&self) -> GradScalerState {
        self.scaler.state_dict()
    }

    /// Restore the scaler state from a checkpoint.
    pub fn load_scaler_state_dict(&mut self, state: &GradScalerState) {
        self.scaler.load_state_dict(state);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- Re-exports exist ----------------------------------------------------

    #[test]
    fn test_autocast_reexported() {
        // Just verify the types are accessible.
        let _: AutocastDtype = AutocastDtype::F16;
        let _: AutocastDtype = AutocastDtype::BF16;
    }

    #[test]
    fn test_autocast_category_reexported() {
        let cat = autocast_category("mm");
        assert_eq!(cat, AutocastCategory::ReducedPrecision);
    }

    // -- AmpContext construction ---------------------------------------------

    #[test]
    fn test_amp_context_construction() {
        let ctx = AmpContext::<f32>::new(AutocastDtype::F16, GradScalerConfig::default());
        assert_eq!(ctx.dtype(), AutocastDtype::F16);
        assert!(ctx.is_enabled());
        assert!((ctx.get_scale() - 65536.0).abs() < 1e-6);
    }

    #[test]
    fn test_amp_context_disabled() {
        let mut config = GradScalerConfig::default();
        config.enabled = false;
        let ctx = AmpContext::<f32>::new(AutocastDtype::BF16, config);
        assert!(!ctx.is_enabled());
    }

    // -- autocast_forward wraps closure in autocast --------------------------

    #[test]
    fn test_autocast_forward_enables_autocast() {
        let ctx = AmpContext::<f32>::new(AutocastDtype::F16, GradScalerConfig::default());

        assert!(!is_autocast_enabled());

        let was_enabled = ctx.autocast_forward(is_autocast_enabled);

        assert!(
            was_enabled,
            "autocast should be enabled inside autocast_forward"
        );
        assert!(!is_autocast_enabled(), "autocast should be disabled after");
    }

    #[test]
    fn test_autocast_forward_sets_dtype() {
        let ctx = AmpContext::<f32>::new(AutocastDtype::BF16, GradScalerConfig::default());

        let dtype = ctx.autocast_forward(autocast_dtype);
        assert_eq!(dtype, AutocastDtype::BF16);
    }

    // -- scaler state dict round-trip ----------------------------------------

    #[test]
    fn test_scaler_state_dict_roundtrip() {
        let mut cfg = GradScalerConfig::default();
        cfg.init_scale = 1024.0;
        let ctx = AmpContext::<f32>::new(AutocastDtype::F16, cfg);

        let state = ctx.scaler_state_dict();
        assert!((state.scale_factor - 1024.0).abs() < 1e-6);

        // Load into a different context.
        let mut ctx2 = AmpContext::<f32>::new(AutocastDtype::F16, GradScalerConfig::default());
        ctx2.load_scaler_state_dict(&state);
        assert!((ctx2.get_scale() - 1024.0).abs() < 1e-6);
    }

    // -- scaler accessor -----------------------------------------------------

    #[test]
    fn test_scaler_accessor() {
        let mut cfg = GradScalerConfig::default();
        cfg.init_scale = 512.0;
        let ctx = AmpContext::<f32>::new(AutocastDtype::F16, cfg);
        assert!((ctx.scaler().get_scale() - 512.0).abs() < 1e-6);
    }

    #[test]
    fn test_scaler_mut_accessor() {
        let mut ctx = AmpContext::<f32>::new(AutocastDtype::F16, GradScalerConfig::default());
        // Load a state through the mutable accessor.
        let state = GradScalerState {
            scale_factor: 999.0,
            growth_tracker: 42,
        };
        ctx.scaler_mut().load_state_dict(&state);
        assert!((ctx.get_scale() - 999.0).abs() < 1e-6);
    }
}
