// CL-332: Vision Transforms & Augmentation — RandomVerticalFlip
use super::rng::random_f64;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Randomly flip a `[C, H, W]` tensor along the vertical axis (H dimension)
/// with probability `p`.
///
/// When applied, rows are reversed — the top row becomes the bottom row and
/// vice-versa. This is the spatial complement to [`RandomHorizontalFlip`].
pub struct RandomVerticalFlip<T: Float> {
    p: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomVerticalFlip<T> {
    /// Create a new `RandomVerticalFlip` with the given probability.
    ///
    /// `p` must be in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `p` is outside `[0, 1]`
    /// or non-finite.
    pub fn new(p: f64) -> FerrotorchResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("RandomVerticalFlip: p must be in [0.0, 1.0], got {p}"),
            });
        }
        Ok(Self {
            p,
            _marker: std::marker::PhantomData,
        })
    }
}

impl<T: Float> Default for RandomVerticalFlip<T> {
    fn default() -> Self {
        // The default p=0.5 is in [0, 1], so `new` never returns Err here;
        // unwrap documents the invariant rather than guarding a real path.
        Self::new(0.5).expect("invariant: default p=0.5 is in [0, 1]")
    }
}

impl<T: Float> Transform<T> for RandomVerticalFlip<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomVerticalFlip: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        if random_f64() >= self.p {
            return Ok(input);
        }

        let channels = shape[0];
        let h = shape[1];
        let w = shape[2];

        if h <= 1 {
            let storage = TensorStorage::cpu(input.data_vec()?);
            return Tensor::from_storage(storage, shape, false);
        }

        let data = input.data()?;
        let mut output = Vec::with_capacity(data.len());

        for c in 0..channels {
            let ch_off = c * h * w;
            for row in (0..h).rev() {
                let start = ch_off + row * w;
                output.extend_from_slice(&data[start..start + w]);
            }
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, shape, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_vertical_flip_always() {
        // p=1.0 should always flip.
        // 1-channel 3x2:
        //   1 2
        //   3 4
        //   5 6
        // Flipped vertically:
        //   5 6
        //   3 4
        //   1 2
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 3, 2], false).unwrap();
        let flip = RandomVerticalFlip::<f64>::new(1.0).unwrap();
        let out = flip.apply(t).unwrap();
        assert_eq!(out.shape(), &[1, 3, 2]);
        let d = out.data().unwrap();
        assert_eq!(d, &[5.0, 6.0, 3.0, 4.0, 1.0, 2.0]);
    }

    #[test]
    fn test_random_vertical_flip_never() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2, 2], false).unwrap();
        let flip = RandomVerticalFlip::<f64>::new(0.0).unwrap();
        let out = flip.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_random_vertical_flip_multichannel() {
        // 2-channel 2x2:
        // Ch0: [1,2,3,4], Ch1: [5,6,7,8]
        // Flipped: Ch0: [3,4,1,2], Ch1: [7,8,5,6]
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2, 2], false).unwrap();
        let flip = RandomVerticalFlip::<f64>::new(1.0).unwrap();
        let out = flip.apply(t).unwrap();
        assert_eq!(
            out.data().unwrap(),
            &[3.0, 4.0, 1.0, 2.0, 7.0, 8.0, 5.0, 6.0]
        );
    }

    #[test]
    fn test_random_vertical_flip_rejects_non_3d() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).unwrap();
        let flip = RandomVerticalFlip::<f64>::new(0.5).unwrap();
        assert!(flip.apply(t).is_err());
    }

    #[test]
    fn test_random_vertical_flip_single_row() {
        // Single row: flip is a no-op.
        let data = vec![1.0_f64, 2.0, 3.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1, 3], false).unwrap();
        let flip = RandomVerticalFlip::<f64>::new(1.0).unwrap();
        let out = flip.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &[1.0, 2.0, 3.0]);
    }
}
