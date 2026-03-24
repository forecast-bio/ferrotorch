use ferrotorch_core::{Device, FerrotorchResult, Float, Tensor};

/// A tensor registered for gradient descent.
///
/// Always has `requires_grad = true`. Stored inside `Module` implementations
/// as the unit of registration for optimizer consumption.
///
/// `Parameter<T>` is a thin wrapper — it derefs to `Tensor<T>` for all
/// tensor operations, and cloning shares the same underlying identity
/// (Arc-based, like Tensor).
#[derive(Debug, Clone)]
pub struct Parameter<T: Float> {
    data: Tensor<T>,
}

impl<T: Float> Parameter<T> {
    /// Create a new parameter from a tensor.
    ///
    /// The tensor is set to `requires_grad = true` regardless of its
    /// current state.
    pub fn new(tensor: Tensor<T>) -> Self {
        Self {
            data: tensor.requires_grad_(true),
        }
    }

    /// Create a parameter initialized with zeros.
    pub fn zeros(shape: &[usize]) -> FerrotorchResult<Self> {
        let t = ferrotorch_core::zeros::<T>(shape)?;
        Ok(Self::new(t))
    }

    /// Create a parameter initialized with ones.
    pub fn ones(shape: &[usize]) -> FerrotorchResult<Self> {
        let t = ferrotorch_core::ones::<T>(shape)?;
        Ok(Self::new(t))
    }

    /// Create a parameter from a data slice.
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

    /// Replace the underlying tensor data while preserving `requires_grad`.
    ///
    /// Used by optimizers to update parameter values without breaking the
    /// parameter identity semantics. The new tensor is set to
    /// `requires_grad = true` regardless of its input state.
    pub fn set_data(&mut self, tensor: Tensor<T>) {
        self.data = tensor.requires_grad_(true);
    }

    /// Move this parameter to a device.
    pub fn to(&self, device: Device) -> FerrotorchResult<Self> {
        Ok(Self::new(self.data.to(device)?))
    }
}

impl<T: Float> std::ops::Deref for Parameter<T> {
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
    fn test_parameter_requires_grad() {
        let p = Parameter::<f32>::zeros(&[3, 4]).unwrap();
        assert!(p.requires_grad());
    }

    #[test]
    fn test_parameter_deref_to_tensor() {
        let p = Parameter::<f32>::zeros(&[2, 3]).unwrap();
        assert_eq!(p.shape(), &[2, 3]);
        assert_eq!(p.numel(), 6);
    }

    #[test]
    fn test_parameter_clone_shares_identity() {
        let p = Parameter::<f32>::zeros(&[4]).unwrap();
        let p2 = p.clone();
        assert!(p.tensor().is_same(p2.tensor()));
    }

    #[test]
    fn test_parameter_to_cpu_preserves_data() {
        let p = Parameter::<f32>::from_slice(&[1.0, 2.0, 3.0], &[3]).unwrap();
        let p2 = p.to(ferrotorch_core::Device::Cpu).unwrap();
        assert_eq!(p2.shape(), &[3]);
        assert_eq!(p2.data().unwrap(), &[1.0, 2.0, 3.0]);
        assert!(p2.requires_grad());
    }

    #[test]
    fn test_parameter_to_cuda_without_backend() {
        let p = Parameter::<f32>::zeros(&[2]).unwrap();
        let result = p.to(ferrotorch_core::Device::Cuda(0));
        assert!(result.is_err());
    }
}
