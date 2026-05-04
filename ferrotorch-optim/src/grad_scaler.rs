//! Gradient scaler for mixed-precision training.
//!
//! Implements the standard loss-scaling approach used with float16 / bfloat16
//! training. The scaler multiplies the loss by a large factor before
//! `backward()` so that small gradients survive quantization to half
//! precision. Before the optimizer step, gradients are divided by the same
//! factor, restoring their true magnitude. If any gradient contains inf/NaN
//! (indicating the scale is too high), the optimizer step is skipped and the
//! scale is reduced; otherwise, consecutive healthy steps gradually increase
//! the scale.
//!
//! This mirrors the behaviour of `torch.cuda.amp.GradScaler`.

use ferrotorch_core::grad_fns::arithmetic::mul;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, scalar};

use crate::optimizer::Optimizer;

// ---------------------------------------------------------------------------
// GradScalerConfig
// ---------------------------------------------------------------------------

/// Configuration for [`GradScaler`].
#[derive(Debug, Clone, Copy)]
#[non_exhaustive]
pub struct GradScalerConfig {
    /// Initial loss-scale factor (default: 65536.0).
    pub init_scale: f64,
    /// Factor by which the scale grows on healthy intervals (default: 2.0).
    pub growth_factor: f64,
    /// Factor by which the scale shrinks when inf/NaN is detected (default: 0.5).
    pub backoff_factor: f64,
    /// Number of consecutive healthy steps before growing the scale (default: 2000).
    pub growth_interval: usize,
    /// Whether gradient scaling is enabled. When `false`, all methods are
    /// pass-through no-ops, allowing the same training loop to work with or
    /// without mixed precision.
    pub enabled: bool,
}

impl Default for GradScalerConfig {
    fn default() -> Self {
        Self {
            init_scale: 65536.0,
            growth_factor: 2.0,
            backoff_factor: 0.5,
            growth_interval: 2000,
            enabled: true,
        }
    }
}

impl GradScalerConfig {
    /// Set the initial loss-scale factor.
    #[must_use]
    pub fn with_init_scale(mut self, init_scale: f64) -> Self {
        self.init_scale = init_scale;
        self
    }

    /// Set the factor by which the scale grows on healthy intervals.
    #[must_use]
    pub fn with_growth_factor(mut self, growth_factor: f64) -> Self {
        self.growth_factor = growth_factor;
        self
    }

    /// Set the factor by which the scale shrinks when inf/NaN is detected.
    #[must_use]
    pub fn with_backoff_factor(mut self, backoff_factor: f64) -> Self {
        self.backoff_factor = backoff_factor;
        self
    }

    /// Set the number of consecutive healthy steps before growing the scale.
    #[must_use]
    pub fn with_growth_interval(mut self, growth_interval: usize) -> Self {
        self.growth_interval = growth_interval;
        self
    }

    /// Enable or disable gradient scaling. When `false`, all methods are pass-through no-ops.
    #[must_use]
    pub fn with_enabled(mut self, enabled: bool) -> Self {
        self.enabled = enabled;
        self
    }
}

// ---------------------------------------------------------------------------
// GradScalerState (for serialization)
// ---------------------------------------------------------------------------

/// Checkpoint-serializable state of a [`GradScaler`].
#[derive(Debug, Clone, PartialEq)]
pub struct GradScalerState {
    /// Current loss-scale factor.
    pub scale_factor: f64,
    /// Number of consecutive steps without inf/NaN since the last scale change.
    pub growth_tracker: usize,
}

// ---------------------------------------------------------------------------
// GradScaler
// ---------------------------------------------------------------------------

/// Dynamic loss scaler for mixed-precision training.
///
/// # Typical usage
///
/// ```ignore
/// let mut scaler = GradScaler::new(GradScalerConfig::default());
///
/// // Forward pass (in half precision).
/// let loss = model.forward(&input);
///
/// // Scale the loss and call backward.
/// let scaled_loss = scaler.scale(&loss)?;
/// optimizer.zero_grad()?;
/// scaled_loss.backward()?;
///
/// // Unscale, step (skipping if inf/NaN), and update the scale factor.
/// scaler.step(&mut optimizer)?;
/// scaler.update();
/// ```
///
/// # Implementation notes
///
/// - **CPU download for inf/NaN check:** `unscale_` downloads all gradient
///   data to CPU (via `.data()`) to check for non-finite values. For GPU
///   tensors this incurs a device-to-host transfer per parameter per step.
///   This matches PyTorch's behaviour where the check also synchronizes.
///
/// - **No short-circuit after inf:** When `unscale_` detects inf/NaN in one
///   parameter group, it continues processing remaining groups rather than
///   returning early. This matches PyTorch's `GradScaler` which always
///   unscales all parameter groups before checking `found_inf`.
#[derive(Debug)]
pub struct GradScaler<T: Float> {
    /// Current multiplicative scale applied to the loss.
    scale_factor: f64,
    /// Number of consecutive healthy steps since the last scale change.
    growth_tracker: usize,
    /// Configuration knobs.
    config: GradScalerConfig,
    /// Set to `true` by [`unscale_`](Self::unscale_) when any gradient
    /// element is non-finite.
    found_inf: bool,
    /// Whether `unscale_` has already been called for the current step.
    /// Reset by [`update`](Self::update).
    already_unscaled: bool,
    /// Phantom marker for the element type.
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> GradScaler<T> {
    /// Create a new `GradScaler` from the given configuration.
    pub fn new(config: GradScalerConfig) -> Self {
        Self {
            scale_factor: config.init_scale,
            growth_tracker: 0,
            config,
            found_inf: false,
            already_unscaled: false,
            _marker: std::marker::PhantomData,
        }
    }

    // -----------------------------------------------------------------------
    // scale
    // -----------------------------------------------------------------------

    /// Multiply the loss by the current scale factor.
    ///
    /// The multiplication participates in the autograd graph so that
    /// `backward()` on the result propagates scaled gradients.
    ///
    /// When the scaler is disabled, the loss is returned unchanged (cloned).
    pub fn scale(&self, loss: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if !self.config.enabled {
            return Ok(loss.clone());
        }
        let scale_tensor = scalar(cast::<f64, T>(self.scale_factor)?)?;
        mul(loss, &scale_tensor)
    }

    // -----------------------------------------------------------------------
    // unscale_
    // -----------------------------------------------------------------------

    /// Divide all parameter gradients by the current scale factor and check
    /// for non-finite values.
    ///
    /// After this call, [`found_inf`](Self::found_inf) is set if **any**
    /// gradient element is inf or NaN.
    ///
    /// This method is idempotent within a single step — calling it twice is
    /// safe (the second call is a no-op).
    pub fn unscale_(&mut self, optimizer: &mut dyn Optimizer<T>) -> FerrotorchResult<()> {
        use std::any::TypeId;

        if !self.config.enabled || self.already_unscaled {
            return Ok(());
        }

        self.found_inf = false;
        let inv_scale = cast::<f64, T>(1.0 / self.scale_factor)?;
        let is_f32 = TypeId::of::<T>() == TypeId::of::<f32>();

        for group in optimizer.param_groups() {
            for param in &group.params {
                let grad_opt = param.grad()?;
                if let Some(grad) = grad_opt {
                    // GPU f32 fast path: scale on-device, check inf via GPU sum.
                    if is_f32 && grad.is_cuda() {
                        if let Some(backend) = ferrotorch_core::gpu_dispatch::gpu_backend() {
                            let inv_f32 = 1.0f32 / self.scale_factor as f32;
                            let scaled = backend.scale_f32(grad.gpu_handle()?, inv_f32)?;

                            // Check for inf/NaN: sum all elements — if any
                            // element is inf/NaN the sum will be non-finite.
                            let sum_handle = backend.sum_f32(&scaled, grad.numel())?;
                            let sum_bytes = backend.gpu_to_cpu(&sum_handle)?;
                            let sum_val = f32::from_le_bytes([
                                sum_bytes[0],
                                sum_bytes[1],
                                sum_bytes[2],
                                sum_bytes[3],
                            ]);
                            if !sum_val.is_finite() {
                                self.found_inf = true;
                            }

                            let new_grad = Tensor::from_storage(
                                ferrotorch_core::TensorStorage::gpu(scaled),
                                grad.shape().to_vec(),
                                false,
                            )?;
                            param.set_grad(Some(new_grad))?;
                            continue;
                        }
                    }

                    // CPU path.
                    let grad_data = grad.data_vec()?;
                    let mut new_data = Vec::with_capacity(grad_data.len());

                    for &val in grad_data.iter() {
                        let unscaled = val * inv_scale;
                        if !unscaled.is_finite() {
                            self.found_inf = true;
                        }
                        new_data.push(unscaled);
                    }

                    let device = grad.device();
                    let new_grad = Tensor::from_storage(
                        ferrotorch_core::TensorStorage::on_device(new_data, device)?,
                        grad.shape().to_vec(),
                        false,
                    )?;
                    param.set_grad(Some(new_grad))?;
                }
            }
        }

        self.already_unscaled = true;
        Ok(())
    }

    // -----------------------------------------------------------------------
    // step
    // -----------------------------------------------------------------------

    /// Unscale gradients (if not already done) and perform an optimizer step,
    /// **unless** inf/NaN was detected.
    ///
    /// Returns `true` when the optimizer step was actually taken, `false`
    /// when it was skipped due to non-finite gradients.
    pub fn step(&mut self, optimizer: &mut dyn Optimizer<T>) -> FerrotorchResult<bool> {
        if !self.config.enabled {
            optimizer.step()?;
            return Ok(true);
        }

        // Ensure gradients are unscaled.
        if !self.already_unscaled {
            self.unscale_(optimizer)?;
        }

        if self.found_inf {
            return Ok(false);
        }

        optimizer.step()?;
        Ok(true)
    }

    // -----------------------------------------------------------------------
    // update
    // -----------------------------------------------------------------------

    /// Adjust the scale factor based on whether inf/NaN was encountered.
    ///
    /// Must be called after every `step()`.
    ///
    /// - If inf/NaN was found: `scale *= backoff_factor` and the growth
    ///   tracker is reset.
    /// - Otherwise: the growth tracker is incremented, and once it reaches
    ///   `growth_interval` the scale is multiplied by `growth_factor` and the
    ///   tracker is reset.
    pub fn update(&mut self) {
        if !self.config.enabled {
            // Reset per-step flags even when disabled.
            self.found_inf = false;
            self.already_unscaled = false;
            return;
        }

        if self.found_inf {
            self.scale_factor *= self.config.backoff_factor;
            self.growth_tracker = 0;
        } else {
            self.growth_tracker += 1;
            if self.growth_tracker >= self.config.growth_interval {
                self.scale_factor *= self.config.growth_factor;
                self.growth_tracker = 0;
            }
        }

        // Reset per-step flags.
        self.found_inf = false;
        self.already_unscaled = false;
    }

    // -----------------------------------------------------------------------
    // Accessors
    // -----------------------------------------------------------------------

    /// Return the current loss-scale factor.
    #[inline]
    pub fn get_scale(&self) -> f64 {
        self.scale_factor
    }

    /// Whether the scaler is enabled.
    #[inline]
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Export the scaler state for checkpoint serialization.
    pub fn state_dict(&self) -> GradScalerState {
        GradScalerState {
            scale_factor: self.scale_factor,
            growth_tracker: self.growth_tracker,
        }
    }

    /// Restore scaler state from a checkpoint.
    pub fn load_state_dict(&mut self, state: &GradScalerState) {
        self.scale_factor = state.scale_factor;
        self.growth_tracker = state.growth_tracker;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::{Optimizer, OptimizerState, ParamGroup};
    use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage};
    use ferrotorch_nn::Parameter;

    // -----------------------------------------------------------------------
    // Minimal optimizer for testing
    // -----------------------------------------------------------------------

    /// A trivial optimizer that records whether `step()` was called.
    struct MockOptimizer<T: Float> {
        param_groups: Vec<ParamGroup<T>>,
        step_called: bool,
    }

    impl<T: Float> MockOptimizer<T> {
        fn new(params: Vec<Parameter<T>>) -> Self {
            let group = ParamGroup::new(params, 0.01);
            Self {
                param_groups: vec![group],
                step_called: false,
            }
        }
    }

    impl<T: Float> Optimizer<T> for MockOptimizer<T> {
        fn step(&mut self) -> FerrotorchResult<()> {
            self.step_called = true;
            Ok(())
        }

        fn zero_grad(&mut self) -> FerrotorchResult<()> {
            for group in &self.param_groups {
                for param in &group.params {
                    param.set_grad(None)?;
                }
            }
            Ok(())
        }

        fn lr(&self) -> f64 {
            self.param_groups.first().map(|g| g.lr).unwrap_or(0.01)
        }

        fn set_lr(&mut self, lr: f64) {
            for g in &mut self.param_groups {
                g.lr = lr;
            }
        }

        fn param_groups(&self) -> &[ParamGroup<T>] {
            &self.param_groups
        }

        fn param_groups_mut(&mut self) -> &mut [ParamGroup<T>] {
            &mut self.param_groups
        }

        fn add_param_group(&mut self, group: ParamGroup<T>) {
            self.param_groups.push(group);
        }

        fn state_dict(&self) -> FerrotorchResult<OptimizerState> {
            Ok(OptimizerState::new())
        }

        fn load_state_dict(&mut self, _state: &OptimizerState) -> FerrotorchResult<()> {
            Ok(())
        }
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn make_param(data: &[f32], shape: &[usize]) -> Parameter<f32> {
        Parameter::from_slice(data, shape).unwrap()
    }

    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // -----------------------------------------------------------------------
    // scale() multiplies loss by scale_factor
    // -----------------------------------------------------------------------

    #[test]
    fn test_scale_multiplies_loss() {
        let config = GradScalerConfig {
            init_scale: 1024.0,
            ..Default::default()
        };
        let scaler = GradScaler::<f32>::new(config);

        let loss = leaf(&[2.0], &[1]);
        let scaled = scaler.scale(&loss).unwrap();
        let val = scaled.data().unwrap()[0];

        assert!(
            (val - 2048.0).abs() < 1e-3,
            "expected 2.0 * 1024.0 = 2048.0, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // unscale_ divides gradients by scale_factor
    // -----------------------------------------------------------------------

    #[test]
    fn test_unscale_divides_gradients() {
        let config = GradScalerConfig {
            init_scale: 256.0,
            ..Default::default()
        };
        let mut scaler = GradScaler::<f32>::new(config);

        // Parameter with a "scaled" gradient.
        let p = make_param(&[1.0, 2.0, 3.0], &[3]);
        let grad = leaf(&[256.0, 512.0, 768.0], &[3]);
        p.set_grad(Some(grad)).unwrap();

        let mut opt = MockOptimizer::new(vec![p]);
        scaler.unscale_(&mut opt).unwrap();

        let unscaled = opt.param_groups[0].params[0].grad().unwrap().unwrap();
        let data = unscaled.data().unwrap();
        assert!(
            (data[0] - 1.0).abs() < 1e-5,
            "expected 256/256 = 1.0, got {}",
            data[0]
        );
        assert!(
            (data[1] - 2.0).abs() < 1e-5,
            "expected 512/256 = 2.0, got {}",
            data[1]
        );
        assert!(
            (data[2] - 3.0).abs() < 1e-5,
            "expected 768/256 = 3.0, got {}",
            data[2]
        );
        assert!(!scaler.found_inf, "no inf expected in healthy gradients");
    }

    // -----------------------------------------------------------------------
    // inf detection skips step and halves scale
    // -----------------------------------------------------------------------

    #[test]
    fn test_inf_skips_step_and_halves_scale() {
        let config = GradScalerConfig {
            init_scale: 1024.0,
            backoff_factor: 0.5,
            ..Default::default()
        };
        let mut scaler = GradScaler::<f32>::new(config);

        // Parameter with an inf gradient.
        let p = make_param(&[1.0], &[1]);
        let grad = leaf(&[f32::INFINITY], &[1]);
        p.set_grad(Some(grad)).unwrap();

        let mut opt = MockOptimizer::new(vec![p]);

        let stepped = scaler.step(&mut opt).unwrap();
        assert!(!stepped, "step should be skipped when inf is found");
        assert!(
            !opt.step_called,
            "optimizer step should not have been called"
        );

        scaler.update();
        assert!(
            (scaler.get_scale() - 512.0).abs() < 1e-6,
            "scale should halve from 1024 to 512, got {}",
            scaler.get_scale()
        );
    }

    // -----------------------------------------------------------------------
    // NaN detection also triggers skip
    // -----------------------------------------------------------------------

    #[test]
    fn test_nan_skips_step() {
        let config = GradScalerConfig {
            init_scale: 1024.0,
            ..Default::default()
        };
        let mut scaler = GradScaler::<f32>::new(config);

        let p = make_param(&[1.0], &[1]);
        let grad = leaf(&[f32::NAN], &[1]);
        p.set_grad(Some(grad)).unwrap();

        let mut opt = MockOptimizer::new(vec![p]);
        let stepped = scaler.step(&mut opt).unwrap();
        assert!(!stepped, "step should be skipped when NaN is found");
    }

    // -----------------------------------------------------------------------
    // growth after consecutive healthy steps
    // -----------------------------------------------------------------------

    #[test]
    fn test_growth_after_healthy_interval() {
        let config = GradScalerConfig {
            init_scale: 128.0,
            growth_factor: 2.0,
            growth_interval: 3, // short interval for testing
            ..Default::default()
        };
        let mut scaler = GradScaler::<f32>::new(config);

        // Simulate 3 healthy steps.
        for _ in 0..3 {
            let p = make_param(&[1.0], &[1]);
            let grad = leaf(&[1.0], &[1]);
            p.set_grad(Some(grad)).unwrap();

            let mut opt = MockOptimizer::new(vec![p]);
            let stepped = scaler.step(&mut opt).unwrap();
            assert!(stepped, "healthy step should proceed");
            scaler.update();
        }

        assert!(
            (scaler.get_scale() - 256.0).abs() < 1e-6,
            "scale should double from 128 to 256 after 3 healthy steps, got {}",
            scaler.get_scale()
        );
    }

    // -----------------------------------------------------------------------
    // growth tracker resets on inf
    // -----------------------------------------------------------------------

    #[test]
    fn test_growth_tracker_resets_on_inf() {
        let config = GradScalerConfig {
            init_scale: 128.0,
            growth_factor: 2.0,
            growth_interval: 3,
            ..Default::default()
        };
        let mut scaler = GradScaler::<f32>::new(config);

        // Two healthy steps, then an inf step.
        for _ in 0..2 {
            let p = make_param(&[1.0], &[1]);
            let grad = leaf(&[1.0], &[1]);
            p.set_grad(Some(grad)).unwrap();
            let mut opt = MockOptimizer::new(vec![p]);
            scaler.step(&mut opt).unwrap();
            scaler.update();
        }

        // Inject an inf.
        let p = make_param(&[1.0], &[1]);
        let grad = leaf(&[f32::INFINITY], &[1]);
        p.set_grad(Some(grad)).unwrap();
        let mut opt = MockOptimizer::new(vec![p]);
        scaler.step(&mut opt).unwrap();
        scaler.update();

        // Scale should have halved, tracker back to zero.
        assert!(
            (scaler.get_scale() - 64.0).abs() < 1e-6,
            "scale should be 128 * 0.5 = 64, got {}",
            scaler.get_scale()
        );

        // Now need 3 more healthy steps to grow.
        for _ in 0..3 {
            let p = make_param(&[1.0], &[1]);
            let grad = leaf(&[1.0], &[1]);
            p.set_grad(Some(grad)).unwrap();
            let mut opt = MockOptimizer::new(vec![p]);
            scaler.step(&mut opt).unwrap();
            scaler.update();
        }

        assert!(
            (scaler.get_scale() - 128.0).abs() < 1e-6,
            "scale should have grown from 64 to 128, got {}",
            scaler.get_scale()
        );
    }

    // -----------------------------------------------------------------------
    // disabled mode is passthrough
    // -----------------------------------------------------------------------

    #[test]
    fn test_disabled_passthrough() {
        let config = GradScalerConfig {
            enabled: false,
            ..Default::default()
        };
        let mut scaler = GradScaler::<f32>::new(config);

        // scale() should return the tensor unchanged.
        let loss = leaf(&[5.0], &[1]);
        let scaled = scaler.scale(&loss).unwrap();
        assert!(
            (scaled.data().unwrap()[0] - 5.0).abs() < 1e-6,
            "disabled scale should not modify the loss"
        );

        // step() should always call optimizer.step().
        let p = make_param(&[1.0], &[1]);
        let grad = leaf(&[f32::INFINITY], &[1]);
        p.set_grad(Some(grad)).unwrap();

        let mut opt = MockOptimizer::new(vec![p]);
        let stepped = scaler.step(&mut opt).unwrap();
        assert!(stepped, "disabled scaler should always step");
        assert!(
            opt.step_called,
            "optimizer step should be called when disabled"
        );
    }

    // -----------------------------------------------------------------------
    // state_dict / load_state_dict roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_state_dict_roundtrip() {
        let config = GradScalerConfig {
            init_scale: 512.0,
            growth_interval: 3,
            ..Default::default()
        };
        let mut scaler = GradScaler::<f32>::new(config);

        // Do a couple healthy steps to advance the tracker.
        for _ in 0..2 {
            let p = make_param(&[1.0], &[1]);
            let grad = leaf(&[1.0], &[1]);
            p.set_grad(Some(grad)).unwrap();
            let mut opt = MockOptimizer::new(vec![p]);
            scaler.step(&mut opt).unwrap();
            scaler.update();
        }

        let state = scaler.state_dict();
        assert!((state.scale_factor - 512.0).abs() < 1e-6);
        assert_eq!(state.growth_tracker, 2);

        // Load into a fresh scaler.
        let mut scaler2 = GradScaler::<f32>::new(GradScalerConfig::default());
        scaler2.load_state_dict(&state);

        assert!((scaler2.get_scale() - 512.0).abs() < 1e-6);
        assert_eq!(scaler2.state_dict().growth_tracker, 2);
    }

    // -----------------------------------------------------------------------
    // unscale_ is idempotent within a step
    // -----------------------------------------------------------------------

    #[test]
    fn test_unscale_idempotent() {
        let config = GradScalerConfig {
            init_scale: 100.0,
            ..Default::default()
        };
        let mut scaler = GradScaler::<f32>::new(config);

        let p = make_param(&[1.0], &[1]);
        let grad = leaf(&[200.0], &[1]);
        p.set_grad(Some(grad)).unwrap();

        let mut opt = MockOptimizer::new(vec![p]);

        // First unscale_: 200 / 100 = 2.
        scaler.unscale_(&mut opt).unwrap();
        let val1 = opt.param_groups[0].params[0]
            .grad()
            .unwrap()
            .unwrap()
            .data()
            .unwrap()[0];
        assert!((val1 - 2.0).abs() < 1e-5);

        // Second unscale_: should be a no-op.
        scaler.unscale_(&mut opt).unwrap();
        let val2 = opt.param_groups[0].params[0]
            .grad()
            .unwrap()
            .unwrap()
            .data()
            .unwrap()[0];
        assert!(
            (val2 - 2.0).abs() < 1e-5,
            "second unscale_ should be a no-op, got {}",
            val2
        );
    }
}
