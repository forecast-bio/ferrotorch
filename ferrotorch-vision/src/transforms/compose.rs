//! Compose — chain multiple transforms into a sequential pipeline.

use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_data::Transform;

/// Applies a sequence of transforms in order.
///
/// Matches `torchvision.transforms.Compose`.
pub struct Compose<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
}

impl<T: Float> Compose<T> {
    /// Create a new composition from an ordered list of transforms.
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> Self {
        Self { transforms }
    }

    /// Number of transforms in the pipeline.
    pub fn len(&self) -> usize {
        self.transforms.len()
    }

    /// Whether the pipeline is empty.
    pub fn is_empty(&self) -> bool {
        self.transforms.is_empty()
    }
}

impl<T: Float> Transform<T> for Compose<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = input;
        for t in &self.transforms {
            x = t.apply(x)?;
        }
        Ok(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::storage::TensorStorage;

    struct DoubleTransform;
    impl Transform<f32> for DoubleTransform {
        fn apply(&self, input: Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
            let data: Vec<f32> = input.data()?.iter().map(|&x| x * 2.0).collect();
            Tensor::from_storage(TensorStorage::cpu(data), input.shape().to_vec(), false)
        }
    }

    struct AddOneTransform;
    impl Transform<f32> for AddOneTransform {
        fn apply(&self, input: Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
            let data: Vec<f32> = input.data()?.iter().map(|&x| x + 1.0).collect();
            Tensor::from_storage(TensorStorage::cpu(data), input.shape().to_vec(), false)
        }
    }

    #[test]
    fn test_compose_chains() {
        let c = Compose::new(vec![Box::new(DoubleTransform), Box::new(AddOneTransform)]);
        let input = Tensor::from_storage(TensorStorage::cpu(vec![3.0f32]), vec![1], false).unwrap();
        let out = c.apply(input).unwrap();
        // 3 * 2 + 1 = 7
        assert_eq!(out.data().unwrap(), &[7.0]);
    }

    #[test]
    fn test_compose_empty() {
        let c: Compose<f32> = Compose::new(vec![]);
        assert!(c.is_empty());
        let input = Tensor::from_storage(TensorStorage::cpu(vec![5.0f32]), vec![1], false).unwrap();
        let out = c.apply(input).unwrap();
        assert_eq!(out.data().unwrap(), &[5.0]);
    }
}
