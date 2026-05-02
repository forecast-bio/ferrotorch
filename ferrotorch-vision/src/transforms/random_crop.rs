//! RandomCrop — randomly crop a [C, H, W] tensor to a target size.

use super::rng::random_usize;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Randomly crop a `[C, H, W]` tensor to `(crop_h, crop_w)`.
///
/// A random top-left corner is chosen uniformly. If the input is smaller
/// than the crop size, an error is returned.
///
/// Matches `torchvision.transforms.RandomCrop`.
pub struct RandomCrop<T: Float> {
    crop_h: usize,
    crop_w: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomCrop<T> {
    pub fn new(crop_h: usize, crop_w: usize) -> Self {
        Self {
            crop_h,
            crop_w,
            _marker: std::marker::PhantomData,
        }
    }

    /// Square crop.
    pub fn square(size: usize) -> Self {
        Self::new(size, size)
    }
}

impl<T: Float> Transform<T> for RandomCrop<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomCrop: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let c = shape[0];
        let h = shape[1];
        let w = shape[2];

        if h < self.crop_h || w < self.crop_w {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomCrop: input ({h}x{w}) is smaller than crop size ({}x{})",
                    self.crop_h, self.crop_w
                ),
            });
        }

        let top = if h == self.crop_h {
            0
        } else {
            random_usize(h - self.crop_h)
        };
        let left = if w == self.crop_w {
            0
        } else {
            random_usize(w - self.crop_w)
        };

        let data = input.data()?;
        let mut out = Vec::with_capacity(c * self.crop_h * self.crop_w);

        for ch in 0..c {
            for row in top..top + self.crop_h {
                for col in left..left + self.crop_w {
                    out.push(data[ch * h * w + row * w + col]);
                }
            }
        }

        Tensor::from_storage(
            TensorStorage::cpu(out),
            vec![c, self.crop_h, self.crop_w],
            false,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_crop_shape() {
        let crop: RandomCrop<f32> = RandomCrop::new(2, 3);
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32; 3 * 5 * 7]),
            vec![3, 5, 7],
            false,
        )
        .unwrap();
        let out = crop.apply(input).unwrap();
        assert_eq!(out.shape(), &[3, 2, 3]);
    }

    #[test]
    fn test_random_crop_exact_size() {
        let crop: RandomCrop<f32> = RandomCrop::square(3);
        let data: Vec<f32> = (0..27).map(|i| i as f32).collect();
        let input =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 3, 3], false).unwrap();
        let out = crop.apply(input).unwrap();
        // Exact size — no cropping needed, should return same data.
        assert_eq!(out.shape(), &[3, 3, 3]);
        assert_eq!(out.data().unwrap(), &data[..]);
    }

    #[test]
    fn test_random_crop_too_small() {
        let crop: RandomCrop<f32> = RandomCrop::new(10, 10);
        let input = Tensor::from_storage(
            TensorStorage::cpu(vec![0.0f32; 3 * 5 * 5]),
            vec![3, 5, 5],
            false,
        )
        .unwrap();
        assert!(crop.apply(input).is_err());
    }
}
