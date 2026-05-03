//! Gradient accumulation for training with effective batch sizes larger than
//! physical memory allows.
//!
//! Usage:
//! ```text
//! let mut accum = GradientAccumulator::new(4); // accumulate over 4 micro-batches
//! for micro_batch in loader {
//!     let loss = model.forward(&micro_batch)?;
//!     let scaled = accum.scale_loss(&loss)?;
//!     scaled.backward()?;
//!     if accum.should_step() {
//!         optimizer.step()?;
//!         optimizer.zero_grad()?;
//!     }
//! }
//! ```

use ferrotorch_core::grad_fns::arithmetic;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, scalar};

/// Manages gradient accumulation across multiple micro-batches.
///
/// Gradients are automatically averaged by dividing the loss by the number
/// of accumulation steps, so accumulated gradients have the same magnitude
/// as a single large-batch gradient.
#[derive(Debug, Clone)]
pub struct GradientAccumulator {
    /// How many micro-batches to accumulate before stepping.
    accumulation_steps: usize,
    /// Current micro-batch index within the accumulation window.
    current_step: usize,
}

impl GradientAccumulator {
    /// Create a new accumulator that triggers an optimizer step every
    /// `steps` micro-batches.
    ///
    /// # Panics
    ///
    /// Panics if `steps` is zero.
    pub fn new(steps: usize) -> Self {
        assert!(steps > 0, "accumulation_steps must be >= 1");
        Self {
            accumulation_steps: steps,
            current_step: 0,
        }
    }

    /// Record one micro-batch and return `true` when the optimizer should step.
    ///
    /// The internal counter is automatically reset when it reaches
    /// `accumulation_steps`.
    pub fn should_step(&mut self) -> bool {
        self.current_step += 1;
        if self.current_step >= self.accumulation_steps {
            self.current_step = 0;
            true
        } else {
            false
        }
    }

    /// Scale a loss tensor so that gradients average correctly over
    /// `accumulation_steps` micro-batches.
    ///
    /// Returns `loss / accumulation_steps`.
    pub fn scale_loss<T: Float>(&self, loss: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let scale = scalar(cast::<f64, T>(1.0 / self.accumulation_steps as f64)?)?;
        arithmetic::mul(loss, &scale)
    }

    /// The number of accumulation steps this accumulator was configured with.
    pub fn accumulation_steps(&self) -> usize {
        self.accumulation_steps
    }

    /// The current step index within the accumulation window (0-based).
    pub fn current_step(&self) -> usize {
        self.current_step
    }

    /// Reset the internal counter to zero.
    pub fn reset(&mut self) {
        self.current_step = 0;
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    /// Helper: 1-D tensor from a slice (no grad).
    fn t(data: &[f64]) -> Tensor<f64> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), vec![data.len()], false).unwrap()
    }

    #[test]
    fn test_should_step_cycles() {
        let mut acc = GradientAccumulator::new(3);
        assert!(!acc.should_step()); // step 1 of 3
        assert!(!acc.should_step()); // step 2 of 3
        assert!(acc.should_step()); // step 3 of 3 -> true, resets
        assert!(!acc.should_step()); // step 1 of 3 (new cycle)
        assert!(!acc.should_step()); // step 2 of 3
        assert!(acc.should_step()); // step 3 of 3 -> true
    }

    #[test]
    fn test_should_step_one() {
        // With steps=1 every call returns true.
        let mut acc = GradientAccumulator::new(1);
        assert!(acc.should_step());
        assert!(acc.should_step());
        assert!(acc.should_step());
    }

    #[test]
    fn test_scale_loss_divides_by_steps() {
        let acc = GradientAccumulator::new(4);
        let loss = t(&[8.0]);
        let scaled = acc.scale_loss(&loss).unwrap();
        let d = scaled.data().unwrap();
        // 8.0 / 4 = 2.0
        assert!((d[0] - 2.0).abs() < 1e-10, "expected 2.0, got {}", d[0]);
    }

    #[test]
    fn test_scale_loss_vector() {
        let acc = GradientAccumulator::new(2);
        let loss = t(&[4.0, 6.0, 10.0]);
        let scaled = acc.scale_loss(&loss).unwrap();
        let d = scaled.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-10);
        assert!((d[1] - 3.0).abs() < 1e-10);
        assert!((d[2] - 5.0).abs() < 1e-10);
    }

    #[test]
    #[allow(clippy::approx_constant)] // 3.14 is an arbitrary loss value, not π.
    fn test_scale_loss_steps_one_identity() {
        let acc = GradientAccumulator::new(1);
        let loss = t(&[3.14]);
        let scaled = acc.scale_loss(&loss).unwrap();
        let d = scaled.data().unwrap();
        assert!(
            (d[0] - 3.14).abs() < 1e-10,
            "steps=1 should not change loss, got {}",
            d[0]
        );
    }

    #[test]
    fn test_reset() {
        let mut acc = GradientAccumulator::new(3);
        acc.should_step(); // 1
        acc.should_step(); // 2
        assert_eq!(acc.current_step(), 2);
        acc.reset();
        assert_eq!(acc.current_step(), 0);
    }

    #[test]
    #[should_panic(expected = "accumulation_steps must be >= 1")]
    fn test_zero_steps_panics() {
        GradientAccumulator::new(0);
    }
}
