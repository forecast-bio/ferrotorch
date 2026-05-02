//! `Buffer<T>` — non-trainable persistent module state. (#583)
//!
//! Mirrors `torch.nn.Module.register_buffer`. A buffer is a tensor that:
//! - is part of the module's persistent state (saved / loaded with the
//!   module via `state_dict`),
//! - moves with the module across devices (`to_device`),
//! - does **not** participate in gradient descent (no `requires_grad`).
//!
//! Typical uses: running mean / running variance in `BatchNorm`, position
//! tables in attention layers, masks, scaling constants — anything the
//! module needs to remember across forward passes that should not be
//! optimized.
//!
//! Buffers are exposed via the `Module` trait through `buffers()`,
//! `buffers_mut()`, and `named_buffers()`. Concrete modules opt in by
//! storing `Buffer<T>` fields and overriding the relevant trait methods.

use ferrotorch_core::{Device, FerrotorchResult, Float, Tensor};

/// A non-trainable tensor that is part of a module's persistent state.
///
/// Like [`crate::Parameter`], `Buffer<T>` derefs to `Tensor<T>` for all
/// tensor operations and clones share the same underlying Arc identity.
/// Unlike `Parameter`, `requires_grad` is **always false**.
#[derive(Debug, Clone)]
pub struct Buffer<T: Float> {
    data: Tensor<T>,
}

impl<T: Float> Buffer<T> {
    /// Wrap a tensor as a buffer. `requires_grad` is forced to `false`.
    pub fn new(tensor: Tensor<T>) -> Self {
        Self {
            data: tensor.requires_grad_(false),
        }
    }

    /// Create a zero-filled buffer with the given shape.
    pub fn zeros(shape: &[usize]) -> FerrotorchResult<Self> {
        let t = ferrotorch_core::zeros::<T>(shape)?;
        Ok(Self::new(t))
    }

    /// Create a one-filled buffer with the given shape.
    pub fn ones(shape: &[usize]) -> FerrotorchResult<Self> {
        let t = ferrotorch_core::ones::<T>(shape)?;
        Ok(Self::new(t))
    }

    /// Create a buffer from a slice + shape.
    pub fn from_slice(data: &[T], shape: &[usize]) -> FerrotorchResult<Self> {
        let t = ferrotorch_core::from_slice(data, shape)?;
        Ok(Self::new(t))
    }

    /// Borrow the underlying tensor.
    #[inline]
    pub fn tensor(&self) -> &Tensor<T> {
        &self.data
    }

    /// Consume and return the underlying tensor.
    pub fn into_tensor(self) -> Tensor<T> {
        self.data
    }

    /// Replace the buffer's data. The new tensor is set to
    /// `requires_grad = false` regardless of its input state.
    pub fn set_data(&mut self, tensor: Tensor<T>) {
        self.data = tensor.requires_grad_(false);
    }

    /// Move this buffer to a device.
    pub fn to(&self, device: Device) -> FerrotorchResult<Self> {
        Ok(Self::new(self.data.to(device)?))
    }
}

impl<T: Float> std::ops::Deref for Buffer<T> {
    type Target = Tensor<T>;

    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_does_not_require_grad() {
        let b = Buffer::<f32>::zeros(&[3, 4]).unwrap();
        assert!(!b.requires_grad());
    }

    #[test]
    fn buffer_derefs_to_tensor() {
        let b = Buffer::<f32>::ones(&[2, 3]).unwrap();
        assert_eq!(b.shape(), &[2, 3]);
        assert_eq!(b.numel(), 6);
    }

    #[test]
    fn buffer_clone_shares_identity() {
        let b = Buffer::<f32>::zeros(&[4]).unwrap();
        let b2 = b.clone();
        assert!(b.tensor().is_same(b2.tensor()));
    }

    #[test]
    fn buffer_set_data_keeps_no_grad() {
        let mut b = Buffer::<f32>::zeros(&[3]).unwrap();
        let t = ferrotorch_core::ones::<f32>(&[3])
            .unwrap()
            .requires_grad_(true);
        assert!(t.requires_grad());
        b.set_data(t);
        assert!(!b.requires_grad());
    }

    #[test]
    fn buffer_to_cpu_preserves_data() {
        let b = Buffer::<f32>::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let b2 = b.to(ferrotorch_core::Device::Cpu).unwrap();
        assert_eq!(b2.data().unwrap(), &[1.0, 2.0, 3.0]);
        assert!(!b2.requires_grad());
    }
}
