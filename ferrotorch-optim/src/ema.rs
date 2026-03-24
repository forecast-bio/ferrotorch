//! Exponential Moving Average (EMA) of model parameters.
//!
//! Maintains a shadow copy of parameters updated with exponential decay:
//!
//! ```text
//! shadow = decay * shadow + (1 - decay) * param
//! ```
//!
//! EMA is commonly used to stabilize training and produce better-performing
//! weights for inference (Polyak averaging). The shadow parameters are never
//! part of the autograd graph.
//!
//! # CL-321

use ferrotorch_core::{FerrotorchResult, Float, no_grad};
use ferrotorch_nn::Parameter;

// ---------------------------------------------------------------------------
// ExponentialMovingAverage
// ---------------------------------------------------------------------------

/// Maintains an exponential moving average of a set of parameters.
///
/// After each optimizer step, call [`update`](Self::update) to update the
/// shadow parameters. Use [`apply_shadow`](Self::apply_shadow) to swap the
/// EMA weights into the model for inference, and
/// [`restore`](Self::restore) to swap back.
///
/// # Example
///
/// ```ignore
/// let mut ema = ExponentialMovingAverage::new(&params, 0.999);
/// for batch in data {
///     optimizer.step()?;
///     ema.update(&params)?;
/// }
/// ema.apply_shadow(&mut params)?; // swap EMA weights in for eval
/// // ... evaluate ...
/// ema.restore(&mut params)?;      // restore original weights
/// ```
#[derive(Debug, Clone)]
pub struct ExponentialMovingAverage<T: Float> {
    /// Decay coefficient in [0, 1]. Higher = slower update.
    decay: f64,
    /// Shadow (averaged) parameter values. Flat Vec<T> per parameter.
    shadow_params: Vec<Vec<T>>,
    /// Backup of original parameter values (populated by `apply_shadow`).
    backup_params: Vec<Vec<T>>,
    /// Number of updates performed.
    num_updates: u64,
    /// Whether to use "inverse decay warmup" (bias correction).
    ///
    /// When enabled, the effective decay at step `t` is:
    /// `min(decay, (1 + t) / (10 + t))` which ramps up from a small value
    /// to the target decay, preventing the initial average from being
    /// dominated by the (often random) initial parameters.
    use_decay_warmup: bool,
}

impl<T: Float> ExponentialMovingAverage<T> {
    /// Create a new EMA tracker by cloning the current parameter values
    /// as the initial shadow state.
    ///
    /// # Arguments
    ///
    /// * `params` — Slice of parameters to track.
    /// * `decay` — Decay rate in `[0, 1]`. Typical values: 0.999 or 0.9999.
    ///
    /// # Panics
    ///
    /// Panics if `decay` is outside `[0, 1]`.
    pub fn new(params: &[Parameter<T>], decay: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&decay),
            "decay must be in [0, 1], got {decay}"
        );

        let shadow_params: Vec<Vec<T>> = params
            .iter()
            .map(|p| p.data().unwrap_or_default().to_vec())
            .collect();

        Self {
            decay,
            shadow_params,
            backup_params: Vec::new(),
            num_updates: 0,
            use_decay_warmup: false,
        }
    }

    /// Enable inverse-decay warmup (bias correction).
    ///
    /// When enabled, the effective decay ramps up from a small value to
    /// the target `decay`, preventing early EMA from being dominated by
    /// the initial (often random) parameter values.
    pub fn with_decay_warmup(mut self, use_warmup: bool) -> Self {
        self.use_decay_warmup = use_warmup;
        self
    }

    /// Return the decay rate.
    pub fn decay(&self) -> f64 {
        self.decay
    }

    /// Return the number of updates performed so far.
    pub fn num_updates(&self) -> u64 {
        self.num_updates
    }

    /// Compute the effective decay for the current step.
    fn effective_decay(&self) -> f64 {
        if self.use_decay_warmup {
            let t = self.num_updates as f64;
            self.decay.min((1.0 + t) / (10.0 + t))
        } else {
            self.decay
        }
    }

    /// Update shadow parameters from the current model parameters.
    ///
    /// ```text
    /// shadow = decay * shadow + (1 - decay) * param
    /// ```
    pub fn update(&mut self, params: &[Parameter<T>]) -> FerrotorchResult<()> {
        assert_eq!(
            params.len(),
            self.shadow_params.len(),
            "parameter count mismatch: expected {}, got {}",
            self.shadow_params.len(),
            params.len()
        );

        let decay = self.effective_decay();
        let one_minus_decay = 1.0 - decay;
        let decay_t = T::from(decay).unwrap();
        let one_minus_decay_t = T::from(one_minus_decay).unwrap();

        for (shadow, param) in self.shadow_params.iter_mut().zip(params.iter()) {
            let param_data = param.data()?;
            assert_eq!(
                shadow.len(),
                param_data.len(),
                "parameter size changed during training"
            );
            for (s, &p) in shadow.iter_mut().zip(param_data.iter()) {
                *s = decay_t * *s + one_minus_decay_t * p;
            }
        }

        self.num_updates += 1;
        Ok(())
    }

    /// Copy shadow parameters into the model parameters (for evaluation).
    ///
    /// The original parameter values are saved in an internal backup so they
    /// can be restored with [`restore`](Self::restore).
    pub fn apply_shadow(&mut self, params: &[Parameter<T>]) -> FerrotorchResult<()> {
        assert_eq!(params.len(), self.shadow_params.len());

        // Save current params as backup.
        self.backup_params = params
            .iter()
            .map(|p| p.data().unwrap_or_default().to_vec())
            .collect();

        // Write shadow values into parameters.
        for (shadow, param) in self.shadow_params.iter().zip(params.iter()) {
            no_grad(|| unsafe { param.tensor().update_data(shadow) })?;
        }

        Ok(())
    }

    /// Restore original parameter values after an `apply_shadow` call.
    ///
    /// # Panics
    ///
    /// Panics if called without a prior `apply_shadow`.
    pub fn restore(&mut self, params: &[Parameter<T>]) -> FerrotorchResult<()> {
        assert!(
            !self.backup_params.is_empty(),
            "restore() called without a prior apply_shadow()"
        );
        assert_eq!(params.len(), self.backup_params.len());

        for (backup, param) in self.backup_params.iter().zip(params.iter()) {
            no_grad(|| unsafe { param.tensor().update_data(backup) })?;
        }

        self.backup_params.clear();
        Ok(())
    }

    /// Return a reference to the shadow parameter values.
    pub fn shadow_params(&self) -> &[Vec<T>] {
        &self.shadow_params
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Construction
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_construction() {
        let p = Parameter::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let ema = ExponentialMovingAverage::new(&[p], 0.999);
        assert_eq!(ema.decay(), 0.999);
        assert_eq!(ema.num_updates(), 0);
        assert_eq!(ema.shadow_params().len(), 1);
        assert_eq!(ema.shadow_params()[0], vec![1.0f32, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "decay must be in [0, 1]")]
    fn test_ema_invalid_decay() {
        let p = Parameter::<f32>::zeros(&[2]).unwrap();
        let _ema = ExponentialMovingAverage::new(&[p], 1.5);
    }

    // -----------------------------------------------------------------------
    // Basic update
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_single_update() {
        // shadow starts at [1.0, 2.0], param moves to [3.0, 4.0]
        // decay = 0.9
        // shadow = 0.9 * [1, 2] + 0.1 * [3, 4] = [0.9+0.3, 1.8+0.4] = [1.2, 2.2]
        let p = Parameter::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let mut ema = ExponentialMovingAverage::new(&[p.clone()], 0.9);

        // Simulate optimizer step changing params.
        ferrotorch_core::no_grad(|| unsafe { p.tensor().update_data(&[3.0f32, 4.0]) }).unwrap();

        ema.update(&[p]).unwrap();

        let shadow = &ema.shadow_params()[0];
        assert!((shadow[0] - 1.2).abs() < 1e-6, "got {}", shadow[0]);
        assert!((shadow[1] - 2.2).abs() < 1e-6, "got {}", shadow[1]);
        assert_eq!(ema.num_updates(), 1);
    }

    #[test]
    fn test_ema_two_updates() {
        let p = Parameter::from_slice(&[0.0f32], &[1]).unwrap();
        let mut ema = ExponentialMovingAverage::new(&[p.clone()], 0.5);

        // Update 1: param = 1.0
        // shadow = 0.5 * 0.0 + 0.5 * 1.0 = 0.5
        ferrotorch_core::no_grad(|| unsafe { p.tensor().update_data(&[1.0f32]) }).unwrap();
        ema.update(&[p.clone()]).unwrap();
        assert!((ema.shadow_params()[0][0] - 0.5).abs() < 1e-6);

        // Update 2: param = 1.0
        // shadow = 0.5 * 0.5 + 0.5 * 1.0 = 0.75
        ema.update(&[p]).unwrap();
        assert!((ema.shadow_params()[0][0] - 0.75).abs() < 1e-6);
        assert_eq!(ema.num_updates(), 2);
    }

    // -----------------------------------------------------------------------
    // Apply / Restore
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_apply_and_restore() {
        let p = Parameter::from_slice(&[10.0f32, 20.0], &[2]).unwrap();
        let mut ema = ExponentialMovingAverage::new(&[p.clone()], 0.5);

        // Update once with param = [0.0, 0.0]
        ferrotorch_core::no_grad(|| unsafe { p.tensor().update_data(&[0.0f32, 0.0]) }).unwrap();
        ema.update(&[p.clone()]).unwrap();
        // shadow = 0.5 * [10, 20] + 0.5 * [0, 0] = [5, 10]

        // Apply shadow — params should become [5, 10].
        ema.apply_shadow(&[p.clone()]).unwrap();
        let data = p.data().unwrap();
        assert!((data[0] - 5.0).abs() < 1e-6);
        assert!((data[1] - 10.0).abs() < 1e-6);

        // Restore — params should go back to [0, 0].
        ema.restore(&[p.clone()]).unwrap();
        let data = p.data().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    #[should_panic(expected = "restore() called without a prior apply_shadow")]
    fn test_ema_restore_without_apply_panics() {
        let p = Parameter::from_slice(&[1.0f32], &[1]).unwrap();
        let mut ema = ExponentialMovingAverage::new(&[p.clone()], 0.99);
        ema.restore(&[p]).unwrap();
    }

    // -----------------------------------------------------------------------
    // Decay warmup
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_decay_warmup() {
        let p = Parameter::from_slice(&[0.0f32], &[1]).unwrap();
        let mut ema = ExponentialMovingAverage::new(&[p.clone()], 0.999).with_decay_warmup(true);

        // Step 0: effective_decay = min(0.999, 1/10) = 0.1
        ferrotorch_core::no_grad(|| unsafe { p.tensor().update_data(&[10.0f32]) }).unwrap();
        ema.update(&[p.clone()]).unwrap();

        // shadow = 0.1 * 0.0 + 0.9 * 10.0 = 9.0
        let shadow = ema.shadow_params()[0][0];
        assert!(
            (shadow - 9.0).abs() < 1e-5,
            "warmup step 0: expected 9.0, got {shadow}"
        );

        // Step 1: effective_decay = min(0.999, 2/11) ≈ 0.1818
        ema.update(&[p.clone()]).unwrap();
        // shadow = 0.1818 * 9.0 + 0.8182 * 10.0 ≈ 1.636 + 8.182 = 9.818
        let shadow = ema.shadow_params()[0][0];
        assert!(
            (shadow - 9.818).abs() < 0.01,
            "warmup step 1: expected ~9.818, got {shadow}"
        );
    }

    // -----------------------------------------------------------------------
    // Multiple parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_multiple_params() {
        let p1 = Parameter::from_slice(&[1.0f32, 2.0], &[2]).unwrap();
        let p2 = Parameter::from_slice(&[3.0f32], &[1]).unwrap();
        let mut ema = ExponentialMovingAverage::new(&[p1.clone(), p2.clone()], 0.0);

        // decay = 0: shadow = 0 * shadow + 1 * param = param (immediate copy)
        ferrotorch_core::no_grad(|| unsafe { p1.tensor().update_data(&[10.0f32, 20.0]) }).unwrap();
        ferrotorch_core::no_grad(|| unsafe { p2.tensor().update_data(&[30.0f32]) }).unwrap();
        ema.update(&[p1, p2]).unwrap();

        assert_eq!(ema.shadow_params()[0], vec![10.0f32, 20.0]);
        assert_eq!(ema.shadow_params()[1], vec![30.0f32]);
    }

    // -----------------------------------------------------------------------
    // Edge case: decay = 1 (shadow never changes)
    // -----------------------------------------------------------------------

    #[test]
    fn test_ema_decay_one_freezes_shadow() {
        let p = Parameter::from_slice(&[5.0f32], &[1]).unwrap();
        let mut ema = ExponentialMovingAverage::new(&[p.clone()], 1.0);

        ferrotorch_core::no_grad(|| unsafe { p.tensor().update_data(&[100.0f32]) }).unwrap();
        ema.update(&[p]).unwrap();

        // shadow = 1.0 * 5.0 + 0.0 * 100.0 = 5.0
        assert!((ema.shadow_params()[0][0] - 5.0).abs() < 1e-6);
    }
}
