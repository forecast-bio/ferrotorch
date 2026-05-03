//! RandomHorizontalFlip — flip a [C, H, W] tensor along the W axis.

use super::rng::random_f64;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Randomly flip a `[C, H, W]` tensor along the horizontal axis (W dimension)
/// with probability `p`.
///
/// Matches `torchvision.transforms.RandomHorizontalFlip`.
pub struct RandomHorizontalFlip<T: Float> {
    p: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomHorizontalFlip<T> {
    /// Create a new `RandomHorizontalFlip` with the given probability.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `p` is outside `[0, 1]`.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("RandomHorizontalFlip: p must be in [0.0, 1.0], got {p}"),
            });
        }
        Ok(Self {
            p,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: Float> Default for RandomHorizontalFlip<T> {
    fn default() -> Self {
        // Default p=0.5 is in [0, 1]; expect documents the invariant.
        Self::new(0.5).expect("invariant: default p=0.5 is in [0, 1]")
    }
}

impl<T: Float> Transform<T> for RandomHorizontalFlip<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomHorizontalFlip: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        if random_f64() >= self.p {
            return Ok(input);
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];
        let data = input.data()?;
        let mut out = vec![<T as num_traits::Zero>::zero(); c * h * w];

        for ch in 0..c {
            for row in 0..h {
                for col in 0..w {
                    out[ch * h * w + row * w + col] = data[ch * h * w + row * w + (w - 1 - col)];
                }
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), shape, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_horizontal_flip_shape() {
        let flip: RandomHorizontalFlip<f32> = RandomHorizontalFlip::new(1.0).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![1, 2, 3],
            false,
        )
        .unwrap();
        let out = flip.apply(input).unwrap();
        assert_eq!(out.shape(), &[1, 2, 3]);
        // Row [1,2,3] -> [3,2,1], Row [4,5,6] -> [6,5,4]
        assert_eq!(out.data().unwrap(), &[3.0, 2.0, 1.0, 6.0, 5.0, 4.0]);
    }

    #[test]
    fn test_horizontal_flip_zero_prob() {
        let flip: RandomHorizontalFlip<f32> = RandomHorizontalFlip::new(0.0).unwrap();
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f32, 2.0, 3.0]),
            vec![1, 1, 3],
            false,
        )
        .unwrap();
        let out = flip.apply(input).unwrap();
        assert_eq!(out.data().unwrap(), &[1.0, 2.0, 3.0]);
    }
}
