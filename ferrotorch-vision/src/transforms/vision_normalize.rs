use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_data::{Normalize, Transform};

use super::{IMAGENET_MEAN, IMAGENET_STD};

/// Per-channel normalization for vision tensors.
///
/// This is a thin wrapper around [`ferrotorch_data::Normalize`] that provides
/// vision-specific constructors (e.g. [`VisionNormalize::imagenet`]) and
/// accepts the mean/std as fixed-size `[f64; 3]` arrays matching the RGB
/// convention.
///
/// For a `[C, H, W]` tensor, each channel *c* is normalized as:
///
/// ```text
/// output[c] = (input[c] - mean[c]) / std[c]
/// ```
pub struct VisionNormalize<T: Float> {
    inner: Normalize<T>,
}

impl<T: Float> VisionNormalize<T> {
    /// Create a new `VisionNormalize` with the given per-channel mean and std.
    ///
    /// # Errors
    ///
    /// Propagates errors from [`Normalize::new`] — most commonly when an
    /// element of `mean` or `std` cannot be represented in `T` (e.g. a value
    /// outside `f32` range when `T = f32`).
    pub fn new(mean: [f64; 3], std: [f64; 3]) -> FerrotorchResult<Self> {
        Ok(Self {
            inner: Normalize::new(mean.to_vec(), std.to_vec())?,
        })
    }

    /// Create a `VisionNormalize` using ImageNet statistics.
    ///
    /// Mean: `[0.485, 0.456, 0.406]`
    /// Std:  `[0.229, 0.224, 0.225]`
    ///
    /// # Panics
    ///
    /// Cannot fail in practice: the ImageNet constants are within both `f32`
    /// and `f64` range. The `.expect` documents the invariant rather than
    /// guarding a real panic path.
    pub fn imagenet() -> Self {
        Self::new(IMAGENET_MEAN, IMAGENET_STD)
            .expect("invariant: ImageNet constants are within Float range")
    }
}

impl<T: Float> Transform<T> for VisionNormalize<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.inner.apply(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;

    #[test]
    fn test_vision_normalize_known_values() {
        // 3 channels, 1 pixel each.
        // Channel 0: val=0.485 => (0.485 - 0.485) / 0.229 = 0.0
        // Channel 1: val=0.456 => (0.456 - 0.456) / 0.224 = 0.0
        // Channel 2: val=0.406 => (0.406 - 0.406) / 0.225 = 0.0
        let data = vec![0.485_f64, 0.456, 0.406];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 1, 1], false).unwrap();
        let norm = VisionNormalize::<f64>::imagenet();
        let out = norm.apply(t).unwrap();
        let d = out.data().unwrap();
        for &val in d {
            assert!(val.abs() < 1e-10, "expected 0.0, got {val}");
        }
    }

    #[test]
    fn test_vision_normalize_non_zero_result() {
        // Channel 0: val=1.0 => (1.0 - 0.485) / 0.229
        let expected_ch0 = (1.0 - 0.485) / 0.229;
        // Channel 1: val=1.0 => (1.0 - 0.456) / 0.224
        let expected_ch1 = (1.0 - 0.456) / 0.224;
        // Channel 2: val=1.0 => (1.0 - 0.406) / 0.225
        let expected_ch2 = (1.0 - 0.406) / 0.225;

        let data = vec![1.0_f64, 1.0, 1.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 1, 1], false).unwrap();
        let norm = VisionNormalize::<f64>::imagenet();
        let out = norm.apply(t).unwrap();
        let d = out.data().unwrap();

        assert!(
            (d[0] - expected_ch0).abs() < 1e-10,
            "channel 0: expected {expected_ch0}, got {}",
            d[0]
        );
        assert!(
            (d[1] - expected_ch1).abs() < 1e-10,
            "channel 1: expected {expected_ch1}, got {}",
            d[1]
        );
        assert!(
            (d[2] - expected_ch2).abs() < 1e-10,
            "channel 2: expected {expected_ch2}, got {}",
            d[2]
        );
    }

    #[test]
    fn test_vision_normalize_custom_stats() {
        // Custom mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        // val=1.0 => (1.0 - 0.5) / 0.5 = 1.0 for all channels
        let data = vec![1.0_f64; 3];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 1, 1], false).unwrap();
        let norm = VisionNormalize::<f64>::new([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]).unwrap();
        let out = norm.apply(t).unwrap();
        let d = out.data().unwrap();
        for &val in d {
            assert!((val - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_vision_normalize_spatial() {
        // 3-channel 2x2 image, all pixels = 0.0.
        // After normalization: (0 - mean) / std for each channel.
        let data = vec![0.0_f64; 12]; // 3 * 2 * 2
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 2, 2], false).unwrap();
        let norm = VisionNormalize::<f64>::imagenet();
        let out = norm.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 2, 2]);

        let d = out.data().unwrap();
        let expected_ch0 = -0.485 / 0.229;
        let expected_ch1 = -0.456 / 0.224;
        let expected_ch2 = -0.406 / 0.225;

        // Channel 0 pixels (indices 0..4)
        for i in 0..4 {
            assert!(
                (d[i] - expected_ch0).abs() < 1e-10,
                "pixel {i}: expected {expected_ch0}, got {}",
                d[i]
            );
        }
        // Channel 1 pixels (indices 4..8)
        for i in 4..8 {
            assert!(
                (d[i] - expected_ch1).abs() < 1e-10,
                "pixel {i}: expected {expected_ch1}, got {}",
                d[i]
            );
        }
        // Channel 2 pixels (indices 8..12)
        for i in 8..12 {
            assert!(
                (d[i] - expected_ch2).abs() < 1e-10,
                "pixel {i}: expected {expected_ch2}, got {}",
                d[i]
            );
        }
    }

    #[test]
    fn test_vision_normalize_f32() {
        let data = vec![0.485_f32, 0.456, 0.406];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 1, 1], false).unwrap();
        let norm = VisionNormalize::<f32>::imagenet();
        let out = norm.apply(t).unwrap();
        let d = out.data().unwrap();
        for &val in d {
            assert!(val.abs() < 1e-5, "expected ~0.0, got {val}");
        }
    }
}
