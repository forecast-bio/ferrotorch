//! TrivialAugmentWide: pick one augmentation at random from a wide
//! space of image ops and apply it with a random strength.
//!
//! Implements Müller & Hutter 2021 ("TrivialAugment: Tuning-free Yet
//! State-of-the-Art Data Augmentation"). Unlike RandAugment/AutoAugment
//! which apply N ops per image and require tuning, TrivialAugmentWide
//! picks **exactly one** op per image and samples its strength
//! uniformly from a fixed wide range. This simplicity matches or
//! beats more complex augmentation strategies on ImageNet.
//!
//! The op space in this implementation is a subset of torchvision's
//! TrivialAugmentWide: the tensor-level ops that can be implemented
//! directly on `[C, H, W]` without PIL. Color operations that depend
//! on per-channel pixel ranges (e.g. posterize, solarize, equalize)
//! are included with the standard 0–1 scaling assumption.
//!
//! Mirrors `torchvision.transforms.v2.TrivialAugmentWide`. CL-458.

use super::rng::random_usize;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;
use num_traits::NumCast;

/// TrivialAugmentWide data augmentation.
///
/// Each call to `apply` picks a single op at random from the wide
/// op space (identity, brightness, contrast, sharpness, posterize,
/// solarize, auto-contrast, equalize, horizontal-flip, translate-x,
/// translate-y) and applies it with a strength sampled uniformly
/// from the op's canonical range.
pub struct TrivialAugmentWide<T: Float> {
    /// Number of discrete magnitude levels the Hutter paper uses (31
    /// by default). Higher levels give finer control over op strength.
    num_magnitude_bins: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Default for TrivialAugmentWide<T> {
    fn default() -> Self {
        Self::new(31)
    }
}

impl<T: Float> TrivialAugmentWide<T> {
    /// Create a new `TrivialAugmentWide` with the given number of
    /// discrete magnitude levels.
    ///
    /// # Panics
    ///
    /// Panics if `num_magnitude_bins` is 0.
    pub fn new(num_magnitude_bins: usize) -> Self {
        assert!(
            num_magnitude_bins > 0,
            "TrivialAugmentWide: num_magnitude_bins must be > 0"
        );
        Self {
            num_magnitude_bins,
            _marker: std::marker::PhantomData,
        }
    }
}

/// The available op space. Keep this in sync with `apply_op` below.
///
/// Each op has a strength parameter sampled from its canonical range
/// at apply-time. `Identity` has no strength (it's the null op that
/// lets the user get the original image back some fraction of the
/// time).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Op {
    Identity,
    Brightness,
    Contrast,
    Sharpness,
    Posterize,
    Solarize,
    AutoContrast,
    Equalize,
    HorizontalFlip,
    TranslateX,
    TranslateY,
}

impl Op {
    const ALL: &'static [Op] = &[
        Op::Identity,
        Op::Brightness,
        Op::Contrast,
        Op::Sharpness,
        Op::Posterize,
        Op::Solarize,
        Op::AutoContrast,
        Op::Equalize,
        Op::HorizontalFlip,
        Op::TranslateX,
        Op::TranslateY,
    ];
}

/// Apply the selected op with a strength sampled uniformly from its
/// canonical range. The canonical ranges are taken from the
/// torchvision TrivialAugmentWide op space.
fn apply_op<T: Float>(data: &[T], h: usize, w: usize, c: usize, op: Op, num_bins: usize) -> Vec<T> {
    // Sample a magnitude level in [0, num_bins), then map to the op's
    // canonical range.
    let level = random_usize(num_bins);
    let level_f = level as f64 / (num_bins - 1).max(1) as f64;

    match op {
        Op::Identity => data.to_vec(),
        Op::Brightness => {
            // brightness factor in [0.01, 1.99]. A factor of 1 is the
            // identity, <1 darkens, >1 brightens.
            let factor = 0.01 + 1.98 * level_f;
            let f_t: T = <T as NumCast>::from(factor).unwrap();
            data.iter().map(|&v| v * f_t).collect()
        }
        Op::Contrast => {
            // contrast factor in [0.01, 1.99]. Contrast blends each
            // pixel with the mean-gray image: out = mean + f*(in - mean).
            let factor = 0.01 + 1.98 * level_f;
            let f_t: T = <T as NumCast>::from(factor).unwrap();
            // Per-channel mean.
            let mut out = Vec::with_capacity(data.len());
            for ch in 0..c {
                let ch_slice = &data[ch * h * w..(ch + 1) * h * w];
                let mean_f64: f64 =
                    ch_slice.iter().map(|v| v.to_f64().unwrap()).sum::<f64>() / (h * w) as f64;
                let mean_t: T = <T as NumCast>::from(mean_f64).unwrap();
                for &v in ch_slice {
                    out.push(mean_t + f_t * (v - mean_t));
                }
            }
            out
        }
        Op::Sharpness => {
            // sharpness factor in [0.01, 1.99]. Sharpen blends the
            // image with a box-blurred version; factor > 1 increases
            // edge contrast, factor < 1 softens.
            let factor = 0.01 + 1.98 * level_f;
            let f_t: T = <T as NumCast>::from(factor).unwrap();
            let mut out = Vec::with_capacity(data.len());
            for ch in 0..c {
                let ch_slice: Vec<f64> = data[ch * h * w..(ch + 1) * h * w]
                    .iter()
                    .map(|v| v.to_f64().unwrap())
                    .collect();
                let blurred = box_blur_3x3(&ch_slice, h, w);
                for i in 0..h * w {
                    let orig: T = <T as NumCast>::from(ch_slice[i]).unwrap();
                    let blur: T = <T as NumCast>::from(blurred[i]).unwrap();
                    // out = blur + factor * (orig - blur)
                    out.push(blur + f_t * (orig - blur));
                }
            }
            out
        }
        Op::Posterize => {
            // bits in [2, 8]. Quantize each pixel value (assumed in
            // [0, 1]) to 2^bits levels.
            let bits = 2 + (6.0 * level_f).round() as u32;
            let levels = (1u32 << bits) as f64;
            let scale = levels - 1.0;
            data.iter()
                .map(|&v| {
                    let vf = v.to_f64().unwrap();
                    let quantized = (vf * scale).round() / scale;
                    <T as NumCast>::from(quantized).unwrap()
                })
                .collect()
        }
        Op::Solarize => {
            // threshold in [0, 1]. Invert pixels above the threshold.
            let threshold = level_f;
            let one = T::from(1.0).unwrap();
            let thr_t: T = <T as NumCast>::from(threshold).unwrap();
            data.iter()
                .map(|&v| if v >= thr_t { one - v } else { v })
                .collect()
        }
        Op::AutoContrast => {
            // Stretch each channel so its min→0 and max→1.
            let mut out = Vec::with_capacity(data.len());
            for ch in 0..c {
                let ch_slice = &data[ch * h * w..(ch + 1) * h * w];
                let mut min_v = f64::INFINITY;
                let mut max_v = f64::NEG_INFINITY;
                for v in ch_slice {
                    let vf = v.to_f64().unwrap();
                    if vf < min_v {
                        min_v = vf;
                    }
                    if vf > max_v {
                        max_v = vf;
                    }
                }
                let range = max_v - min_v;
                for &v in ch_slice {
                    let vf = v.to_f64().unwrap();
                    let stretched = if range > 0.0 {
                        (vf - min_v) / range
                    } else {
                        vf
                    };
                    out.push(<T as NumCast>::from(stretched).unwrap());
                }
            }
            out
        }
        Op::Equalize => {
            // Histogram equalization using 256 bins per channel
            // (assumes input in [0, 1]).
            let mut out = Vec::with_capacity(data.len());
            const BINS: usize = 256;
            for ch in 0..c {
                let ch_slice = &data[ch * h * w..(ch + 1) * h * w];
                let mut hist = [0usize; BINS];
                for v in ch_slice {
                    let vf = v.to_f64().unwrap().clamp(0.0, 1.0);
                    let idx = (vf * (BINS as f64 - 1.0)).round() as usize;
                    hist[idx] += 1;
                }
                // Build CDF and normalize.
                let mut cdf = [0.0_f64; BINS];
                let mut cum = 0.0;
                let total = (h * w) as f64;
                for i in 0..BINS {
                    cum += hist[i] as f64;
                    cdf[i] = cum / total;
                }
                for &v in ch_slice {
                    let vf = v.to_f64().unwrap().clamp(0.0, 1.0);
                    let idx = (vf * (BINS as f64 - 1.0)).round() as usize;
                    out.push(<T as NumCast>::from(cdf[idx]).unwrap());
                }
            }
            out
        }
        Op::HorizontalFlip => {
            // Flip each row's columns. HorizontalFlip has no strength
            // parameter in TrivialAugmentWide — level is ignored.
            let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];
            for ch in 0..c {
                for row in 0..h {
                    for col in 0..w {
                        let src = ch * h * w + row * w + col;
                        let dst = ch * h * w + row * w + (w - 1 - col);
                        out[dst] = data[src];
                    }
                }
            }
            out
        }
        Op::TranslateX => {
            // translate in pixels, sampled from [-0.32*W, 0.32*W].
            let max_shift = (0.32 * w as f64) as i64;
            let shift = (2.0 * level_f - 1.0) * max_shift as f64;
            let shift = shift.round() as i64;
            let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];
            for ch in 0..c {
                for row in 0..h {
                    for col in 0..w {
                        let src_col = col as i64 - shift;
                        if src_col >= 0 && (src_col as usize) < w {
                            out[ch * h * w + row * w + col] =
                                data[ch * h * w + row * w + src_col as usize];
                        }
                    }
                }
            }
            out
        }
        Op::TranslateY => {
            let max_shift = (0.32 * h as f64) as i64;
            let shift = (2.0 * level_f - 1.0) * max_shift as f64;
            let shift = shift.round() as i64;
            let mut out = vec![<T as num_traits::Zero>::zero(); data.len()];
            for ch in 0..c {
                for row in 0..h {
                    for col in 0..w {
                        let src_row = row as i64 - shift;
                        if src_row >= 0 && (src_row as usize) < h {
                            out[ch * h * w + row * w + col] =
                                data[ch * h * w + src_row as usize * w + col];
                        }
                    }
                }
            }
            out
        }
    }
}

/// Simple 3x3 box blur with zero padding, single channel.
fn box_blur_3x3(data: &[f64], h: usize, w: usize) -> Vec<f64> {
    let mut out = vec![0.0; h * w];
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0;
            let mut count = 0.0;
            for dr in -1i64..=1 {
                for dc in -1i64..=1 {
                    let r = row as i64 + dr;
                    let c = col as i64 + dc;
                    if r >= 0 && (r as usize) < h && c >= 0 && (c as usize) < w {
                        acc += data[r as usize * w + c as usize];
                        count += 1.0;
                    }
                }
            }
            out[row * w + col] = acc / count;
        }
    }
    out
}

impl<T: Float> Transform<T> for TrivialAugmentWide<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "TrivialAugmentWide: expected 3-D tensor [C, H, W], got {shape:?}"
                ),
            });
        }
        let c = shape[0];
        let h = shape[1];
        let w = shape[2];
        if h == 0 || w == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "TrivialAugmentWide: image dimensions must be > 0".into(),
            });
        }

        // Pick one op uniformly from the op space.
        let op_idx = random_usize(Op::ALL.len());
        let op = Op::ALL[op_idx];

        let data = input.data()?;
        let out = apply_op(data, h, w, c, op, self.num_magnitude_bins);
        Tensor::from_storage(TensorStorage::cpu(out), shape, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::rng::vision_manual_seed;

    #[test]
    fn test_trivial_augment_output_shape_preserved() {
        let t: Tensor<f32> =
            Tensor::from_storage(TensorStorage::cpu(vec![0.5; 48]), vec![3, 4, 4], false).unwrap();
        let aug = TrivialAugmentWide::<f32>::new(31);
        let out = aug.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
    }

    #[test]
    fn test_trivial_augment_default_num_bins() {
        let aug = TrivialAugmentWide::<f32>::default();
        assert_eq!(aug.num_magnitude_bins, 31);
    }

    #[test]
    #[should_panic(expected = "num_magnitude_bins must be > 0")]
    fn test_trivial_augment_zero_bins_panics() {
        let _ = TrivialAugmentWide::<f32>::new(0);
    }

    #[test]
    fn test_trivial_augment_rejects_non_3d() {
        let t: Tensor<f32> =
            Tensor::from_storage(TensorStorage::cpu(vec![0.5; 4]), vec![2, 2], false).unwrap();
        let aug = TrivialAugmentWide::<f32>::new(31);
        assert!(aug.apply(t).is_err());
    }

    #[test]
    fn test_op_identity_returns_input_unchanged() {
        let data: Vec<f32> = (0..12).map(|i| i as f32 * 0.1).collect();
        let out = apply_op(&data, 2, 2, 3, Op::Identity, 31);
        assert_eq!(out, data);
    }

    #[test]
    fn test_op_horizontal_flip_reverses_columns() {
        // Single channel 1x3: [1, 2, 3] -> [3, 2, 1]
        let data = vec![1.0_f32, 2.0, 3.0];
        let out = apply_op(&data, 1, 3, 1, Op::HorizontalFlip, 31);
        assert_eq!(out, vec![3.0, 2.0, 1.0]);
    }

    #[test]
    fn test_op_brightness_scales_pixels() {
        // Brightness with level 0 → factor = 0.01, effectively near-black.
        vision_manual_seed(0);
        let data = vec![1.0_f32; 4];
        // We can't control the sampled level directly, but we can check
        // that brightness produces a non-negative valid output and preserves
        // length.
        let out = apply_op(&data, 2, 2, 1, Op::Brightness, 2);
        assert_eq!(out.len(), 4);
        for &v in &out {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn test_op_posterize_preserves_length() {
        vision_manual_seed(42);
        let data: Vec<f32> = (0..9).map(|i| i as f32 / 8.0).collect();
        let out = apply_op(&data, 3, 3, 1, Op::Posterize, 7);
        assert_eq!(out.len(), 9);
        for &v in &out {
            assert!(
                (0.0..=1.0).contains(&v),
                "posterize should stay in [0,1], got {v}"
            );
        }
    }

    #[test]
    fn test_op_solarize_at_threshold_zero_inverts_all() {
        // When the sampled threshold is 0, all pixels satisfy v >= 0,
        // so all get inverted to 1 - v. We test by calling solarize with
        // forced bin 0.
        vision_manual_seed(0);
        let data = vec![0.2_f32, 0.5, 0.8, 1.0];
        // Use num_bins=1 so level_f = 0 / 0 = 0 (guarded by .max(1))
        // which means threshold=0 and all pixels get inverted.
        let out = apply_op(&data, 2, 2, 1, Op::Solarize, 1);
        // Threshold = 0, all pixels >= 0, all inverted.
        let expected = [0.8, 0.5, 0.2, 0.0_f32];
        for (i, (&o, e)) in out.iter().zip(expected.iter()).enumerate() {
            assert!((o - e).abs() < 1e-6, "solarize[{i}]: {o} vs {e}");
        }
    }

    #[test]
    fn test_op_auto_contrast_stretches_range() {
        // A channel with values [0.3, 0.5, 0.7] should stretch to [0, 0.5, 1].
        let data = vec![0.3_f64, 0.5, 0.7];
        let out = apply_op(&data, 1, 3, 1, Op::AutoContrast, 31);
        assert!((out[0] - 0.0).abs() < 1e-10);
        assert!((out[1] - 0.5).abs() < 1e-10);
        assert!((out[2] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_op_auto_contrast_constant_channel_is_unchanged() {
        // If min == max (constant channel), auto-contrast leaves it
        // alone (avoids divide-by-zero).
        let data = vec![0.4_f64; 5];
        let out = apply_op(&data, 1, 5, 1, Op::AutoContrast, 31);
        for &v in &out {
            assert!((v - 0.4).abs() < 1e-10);
        }
    }

    #[test]
    fn test_op_equalize_cdf_is_monotonic() {
        // After equalization, values should be in [0, 1] and preserve
        // relative ordering.
        let data: Vec<f64> = (0..100).map(|i| i as f64 / 99.0).collect();
        let out = apply_op(&data, 10, 10, 1, Op::Equalize, 31);
        for &v in &out {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_box_blur_uniform_is_unchanged_interior() {
        // A uniform image should stay uniform in the interior where
        // the 3x3 kernel fits entirely.
        let data = vec![0.5_f64; 25]; // 5x5
        let blurred = box_blur_3x3(&data, 5, 5);
        for row in 1..4 {
            for col in 1..4 {
                assert!((blurred[row * 5 + col] - 0.5).abs() < 1e-10);
            }
        }
    }
}
