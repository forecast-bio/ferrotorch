// CL-332: Vision Transforms & Augmentation — RandomResizedCrop
use super::rng::random_f64;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Crop a random region of the input, then resize to a target size.
///
/// This mirrors `torchvision.transforms.RandomResizedCrop`. A rectangular
/// region whose area is a random fraction (within `scale`) of the original
/// area and whose aspect ratio falls within `ratio` is sampled. The region
/// is then resized to `(height, width)` using nearest-neighbor interpolation.
///
/// If no valid crop can be found after a fixed number of attempts, a center
/// crop at the target aspect ratio is used as a fallback.
pub struct RandomResizedCrop<T: Float> {
    height: usize,
    width: usize,
    scale_lo: f64,
    scale_hi: f64,
    ratio_lo: f64,
    ratio_hi: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomResizedCrop<T> {
    /// Create a new `RandomResizedCrop`.
    ///
    /// * `height`, `width` — output spatial size.
    /// * `scale` — range of area fraction `(lo, hi)` relative to the input,
    ///   e.g. `(0.08, 1.0)`.
    /// * `ratio` — range of aspect ratio `(lo, hi)`, e.g. `(3.0/4.0, 4.0/3.0)`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `scale` is not
    /// `(0, 1] × (0, 1]` with `lo <= hi`, or `ratio` is not positive with
    /// `lo <= hi`.
    pub fn new(
        height: usize,
        width: usize,
        scale: (f64, f64),
        ratio: (f64, f64),
    ) -> FerrotorchResult<Self> {
        if !(scale.0 > 0.0 && scale.0 <= scale.1 && scale.1 <= 1.0) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomResizedCrop: scale must satisfy 0 < lo <= hi <= 1, got ({}, {})",
                    scale.0, scale.1,
                ),
            });
        }
        if !(ratio.0 > 0.0 && ratio.0 <= ratio.1) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomResizedCrop: ratio must satisfy 0 < lo <= hi, got ({}, {})",
                    ratio.0, ratio.1,
                ),
            });
        }
        Ok(Self {
            height,
            width,
            scale_lo: scale.0,
            scale_hi: scale.1,
            ratio_lo: ratio.0,
            ratio_hi: ratio.1,
            _marker: std::marker::PhantomData,
        })
    }
}

/// Nearest-neighbor resize of a single channel from `(in_h, in_w)` to
/// `(out_h, out_w)`, reading from `src` (length `in_h * in_w`).
pub(crate) fn nn_resize_channel<T: Float>(
    src: &[T],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    dst: &mut Vec<T>,
) {
    for oh in 0..out_h {
        let ih = if in_h == 1 { 0 } else { (oh * in_h) / out_h };
        for ow in 0..out_w {
            let iw = if in_w == 1 { 0 } else { (ow * in_w) / out_w };
            dst.push(src[ih * in_w + iw]);
        }
    }
}

impl<T: Float> Transform<T> for RandomResizedCrop<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomResizedCrop: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let channels = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];
        let area = (in_h * in_w) as f64;

        let data = input.data()?;

        // Try up to 10 times to find a valid crop.
        let mut crop_top = 0usize;
        let mut crop_left = 0usize;
        let mut crop_h = in_h;
        let mut crop_w = in_w;
        let mut found = false;

        for _ in 0..10 {
            let target_area =
                area * (self.scale_lo + random_f64() * (self.scale_hi - self.scale_lo));
            let log_lo = self.ratio_lo.ln();
            let log_hi = self.ratio_hi.ln();
            let aspect = (log_lo + random_f64() * (log_hi - log_lo)).exp();

            let w_f = (target_area * aspect).sqrt();
            let h_f = (target_area / aspect).sqrt();
            let w_candidate = w_f.round() as usize;
            let h_candidate = h_f.round() as usize;

            if w_candidate >= 1 && h_candidate >= 1 && w_candidate <= in_w && h_candidate <= in_h {
                crop_h = h_candidate;
                crop_w = w_candidate;
                crop_top = if in_h == crop_h {
                    0
                } else {
                    (random_f64() * (in_h - crop_h) as f64) as usize
                };
                crop_left = if in_w == crop_w {
                    0
                } else {
                    (random_f64() * (in_w - crop_w) as f64) as usize
                };
                found = true;
                break;
            }
        }

        if !found {
            // Fallback: center crop at the target aspect ratio.
            let target_ratio = self.width as f64 / self.height as f64;
            let in_ratio = in_w as f64 / in_h as f64;
            if in_ratio < target_ratio {
                crop_w = in_w;
                crop_h = ((in_w as f64 / target_ratio).round() as usize)
                    .max(1)
                    .min(in_h);
            } else {
                crop_h = in_h;
                crop_w = ((in_h as f64 * target_ratio).round() as usize)
                    .max(1)
                    .min(in_w);
            }
            crop_top = (in_h - crop_h) / 2;
            crop_left = (in_w - crop_w) / 2;
        }

        // Extract the crop, then resize to (self.height, self.width).
        let mut output = Vec::with_capacity(channels * self.height * self.width);

        for c in 0..channels {
            let ch_off = c * in_h * in_w;
            // Extract cropped channel into a temporary buffer.
            let mut cropped = Vec::with_capacity(crop_h * crop_w);
            for row in crop_top..crop_top + crop_h {
                let start = ch_off + row * in_w + crop_left;
                cropped.extend_from_slice(&data[start..start + crop_w]);
            }
            nn_resize_channel(
                &cropped,
                crop_h,
                crop_w,
                self.height,
                self.width,
                &mut output,
            );
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, vec![channels, self.height, self.width], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_random_resized_crop_output_shape() {
        let data: Vec<f64> = (0..300).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 10, 10], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(5, 5, (0.08, 1.0), (0.75, 1.333)).unwrap();
        let out = rrc.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 5, 5]);
    }

    #[test]
    fn test_random_resized_crop_full_scale() {
        // scale=(1.0, 1.0), ratio=(1.0, 1.0): should crop the entire image
        // and resize to target.
        let data: Vec<f64> = (0..48).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 4, 4], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(2, 2, (1.0, 1.0), (1.0, 1.0)).unwrap();
        let out = rrc.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 2, 2]);
    }

    #[test]
    fn test_random_resized_crop_values_from_input() {
        let data: Vec<f64> = (0..75).map(|i| i as f64).collect();
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 5, 5], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(3, 3, (0.5, 1.0), (0.75, 1.333)).unwrap();
        let out = rrc.apply(t).unwrap();
        let out_data = out.data().unwrap();
        let original: std::collections::HashSet<u64> = data.iter().map(|&v| v.to_bits()).collect();
        for &val in out_data {
            assert!(
                original.contains(&val.to_bits()),
                "Output value {val} not found in original"
            );
        }
    }

    #[test]
    fn test_random_resized_crop_rejects_non_3d() {
        let data = vec![1.0_f64; 8];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 4], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(2, 2, (0.08, 1.0), (0.75, 1.333)).unwrap();
        assert!(rrc.apply(t).is_err());
    }

    #[test]
    fn test_random_resized_crop_multichannel() {
        let data: Vec<f64> = (0..192).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 8, 8], false).unwrap();
        let rrc = RandomResizedCrop::<f64>::new(4, 4, (0.2, 0.8), (0.75, 1.333)).unwrap();
        let out = rrc.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
        assert_eq!(out.numel(), 48);
    }

    #[test]
    fn test_nn_resize_channel_identity() {
        let src = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = Vec::new();
        nn_resize_channel(&src, 2, 2, 2, 2, &mut dst);
        assert_eq!(dst, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_nn_resize_channel_upscale() {
        // 2x2 -> 4x4
        let src = vec![1.0_f64, 2.0, 3.0, 4.0];
        let mut dst = Vec::new();
        nn_resize_channel(&src, 2, 2, 4, 4, &mut dst);
        let expected = vec![
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(dst, expected);
    }
}
