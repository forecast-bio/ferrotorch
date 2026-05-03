// CL-332: Vision Transforms & Augmentation — ColorJitter
use super::rng::random_f64;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Randomly adjust the brightness, contrast, saturation, and hue of an image.
///
/// Expects a `[3, H, W]` tensor in **RGB** channel order with values in
/// `[0, 1]`. Each parameter specifies a range `[max(0, 1 - v), 1 + v]` from
/// which a multiplicative/additive factor is uniformly sampled.
///
/// Processing order is randomised per call (matching PyTorch):
///
/// 1. **Brightness** — scale all channels by a factor in `[1 - b, 1 + b]`.
/// 2. **Contrast** — blend towards the per-channel mean by a factor.
/// 3. **Saturation** — blend towards the luminance (grayscale) image.
/// 4. **Hue** — rotate the hue angle in HSV space by a shift in `[-h, h]`
///    (measured in fraction of a full circle, range `(-0.5, 0.5)`).
///
/// This mirrors `torchvision.transforms.ColorJitter`.
pub struct ColorJitter<T: Float> {
    brightness: f64,
    contrast: f64,
    saturation: f64,
    hue: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> ColorJitter<T> {
    /// Create a new `ColorJitter`.
    ///
    /// * `brightness` — non-negative. 0 means no change. Factor sampled from
    ///   `[max(0, 1 - brightness), 1 + brightness]`.
    /// * `contrast` — same convention.
    /// * `saturation` — same convention.
    /// * `hue` — in `[0, 0.5)`. Hue shift sampled from `[-hue, +hue]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if any of `brightness`,
    /// `contrast`, `saturation` is negative, or if `hue` is outside `[0, 0.5)`.
    pub fn new(
        brightness: f64,
        contrast: f64,
        saturation: f64,
        hue: f64,
    ) -> FerrotorchResult<Self> {
        if brightness < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ColorJitter: brightness must be >= 0, got {brightness}"),
            });
        }
        if contrast < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ColorJitter: contrast must be >= 0, got {contrast}"),
            });
        }
        if saturation < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ColorJitter: saturation must be >= 0, got {saturation}"),
            });
        }
        if !((0.0..0.5).contains(&hue) || hue == 0.0) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("ColorJitter: hue must be in [0, 0.5), got {hue}"),
            });
        }
        Ok(Self {
            brightness,
            contrast,
            saturation,
            hue,
            _marker: std::marker::PhantomData,
        })
    }
}

/// Fisher-Yates shuffle using the global PRNG.
fn shuffle_order(n: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..n).collect();
    for i in (1..n).rev() {
        let j = (random_f64() * (i + 1) as f64) as usize;
        let j = j.min(i); // Clamp in case random_f64() returns exactly 1.0.
        order.swap(i, j);
    }
    order
}

/// Sample a uniform factor from `[max(0, 1 - v), 1 + v]`.
fn uniform_factor(v: f64) -> f64 {
    let lo = (1.0 - v).max(0.0);
    let hi = 1.0 + v;
    lo + random_f64() * (hi - lo)
}

impl<T: Float> Transform<T> for ColorJitter<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 || shape[0] != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ColorJitter: expected 3-D RGB tensor [3, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let h = shape[1];
        let w = shape[2];
        let spatial = h * w;
        let data = input.data_vec()?;
        // Work with per-channel slices as mutable f64 buffers for precision.
        let mut r: Vec<f64> = data[..spatial]
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect();
        let mut g: Vec<f64> = data[spatial..2 * spatial]
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect();
        let mut b: Vec<f64> = data[2 * spatial..]
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect();

        // Determine random order for the four adjustments.
        let order = shuffle_order(4);

        for &op in &order {
            match op {
                0 if self.brightness > 0.0 => {
                    let factor = uniform_factor(self.brightness);
                    for i in 0..spatial {
                        r[i] *= factor;
                        g[i] *= factor;
                        b[i] *= factor;
                    }
                }
                1 if self.contrast > 0.0 => {
                    let factor = uniform_factor(self.contrast);
                    // Compute per-channel mean.
                    let mean_r: f64 = r.iter().sum::<f64>() / spatial as f64;
                    let mean_g: f64 = g.iter().sum::<f64>() / spatial as f64;
                    let mean_b: f64 = b.iter().sum::<f64>() / spatial as f64;
                    for i in 0..spatial {
                        r[i] = mean_r + (r[i] - mean_r) * factor;
                        g[i] = mean_g + (g[i] - mean_g) * factor;
                        b[i] = mean_b + (b[i] - mean_b) * factor;
                    }
                }
                2 if self.saturation > 0.0 => {
                    let factor = uniform_factor(self.saturation);
                    // Grayscale via ITU-R BT.601 luma coefficients.
                    for i in 0..spatial {
                        let gray = 0.2989 * r[i] + 0.5870 * g[i] + 0.1140 * b[i];
                        r[i] = gray + (r[i] - gray) * factor;
                        g[i] = gray + (g[i] - gray) * factor;
                        b[i] = gray + (b[i] - gray) * factor;
                    }
                }
                3 if self.hue > 0.0 => {
                    let hue_shift = self.hue * (2.0 * random_f64() - 1.0);
                    for i in 0..spatial {
                        let (hue, sat, val) = rgb_to_hsv(r[i], g[i], b[i]);
                        let new_hue = (hue + hue_shift).rem_euclid(1.0);
                        let (nr, ng, nb) = hsv_to_rgb(new_hue, sat, val);
                        r[i] = nr;
                        g[i] = ng;
                        b[i] = nb;
                    }
                }
                _ => {}
            }
        }

        // Clamp to [0, 1] and convert back to T.
        let mut output = Vec::with_capacity(data.len());
        for v in r.iter().chain(g.iter()).chain(b.iter()) {
            let clamped = v.clamp(0.0, 1.0);
            output.push(cast::<f64, T>(clamped)?);
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// RGB <-> HSV conversion helpers
// ---------------------------------------------------------------------------

fn rgb_to_hsv(r: f64, g: f64, b: f64) -> (f64, f64, f64) {
    let max = r.max(g).max(b);
    let min = r.min(g).min(b);
    let delta = max - min;

    let v = max;
    let s = if max == 0.0 { 0.0 } else { delta / max };

    let h = if delta == 0.0 {
        0.0
    } else if (max - r).abs() < 1e-15 {
        ((g - b) / delta).rem_euclid(6.0) / 6.0
    } else if (max - g).abs() < 1e-15 {
        ((b - r) / delta + 2.0) / 6.0
    } else {
        ((r - g) / delta + 4.0) / 6.0
    };

    (h, s, v)
}

fn hsv_to_rgb(h: f64, s: f64, v: f64) -> (f64, f64, f64) {
    if s == 0.0 {
        return (v, v, v);
    }

    let h6 = h * 6.0;
    let sector = h6.floor() as usize % 6;
    let f = h6 - h6.floor();
    let p = v * (1.0 - s);
    let q = v * (1.0 - s * f);
    let t = v * (1.0 - s * (1.0 - f));

    match sector {
        0 => (v, t, p),
        1 => (q, v, p),
        2 => (p, v, t),
        3 => (p, q, v),
        4 => (t, p, v),
        _ => (v, p, q),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgb_tensor(r: &[f64], g: &[f64], b: &[f64]) -> Tensor<f64> {
        let spatial = r.len();
        let mut data = Vec::with_capacity(3 * spatial);
        data.extend_from_slice(r);
        data.extend_from_slice(g);
        data.extend_from_slice(b);
        Tensor::from_storage(TensorStorage::cpu(data), vec![3, 1, spatial], false).unwrap()
    }

    #[test]
    fn test_color_jitter_output_shape() {
        let data: Vec<f64> = vec![0.5; 48]; // 3x4x4
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 4, 4], false).unwrap();
        let jitter = ColorJitter::<f64>::new(0.2, 0.2, 0.2, 0.1).unwrap();
        let out = jitter.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
    }

    #[test]
    fn test_color_jitter_zero_params() {
        // All parameters zero: output should equal input.
        let data: Vec<f64> = (0..12).map(|i| i as f64 / 12.0).collect();
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![3, 2, 2], false).unwrap();
        let jitter = ColorJitter::<f64>::new(0.0, 0.0, 0.0, 0.0).unwrap();
        let out = jitter.apply(t).unwrap();
        let d = out.data().unwrap();
        for (a, b) in d.iter().zip(data.iter()) {
            assert!((a - b).abs() < 1e-10, "Expected {b}, got {a}");
        }
    }

    #[test]
    fn test_color_jitter_output_clamped() {
        // Even with extreme parameters, output should be in [0, 1].
        let data: Vec<f64> = vec![0.9; 12]; // 3x2x2
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 2, 2], false).unwrap();
        let jitter = ColorJitter::<f64>::new(0.9, 0.9, 0.9, 0.4).unwrap();
        let out = jitter.apply(t).unwrap();
        for &val in out.data().unwrap() {
            assert!(
                (0.0..=1.0).contains(&val),
                "Output value {val} out of [0, 1]"
            );
        }
    }

    #[test]
    fn test_color_jitter_rejects_non_rgb() {
        // 1-channel tensor should be rejected.
        let data = vec![0.5_f64; 4];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2, 2], false).unwrap();
        let jitter = ColorJitter::<f64>::new(0.2, 0.2, 0.2, 0.1).unwrap();
        assert!(jitter.apply(t).is_err());
    }

    #[test]
    fn test_rgb_hsv_roundtrip() {
        let test_colors = vec![
            (1.0, 0.0, 0.0), // Red
            (0.0, 1.0, 0.0), // Green
            (0.0, 0.0, 1.0), // Blue
            (0.5, 0.5, 0.5), // Gray
            (0.0, 0.0, 0.0), // Black
            (1.0, 1.0, 1.0), // White
            (0.3, 0.6, 0.9), // Arbitrary
        ];

        for (r, g, b) in test_colors {
            let (h, s, v) = rgb_to_hsv(r, g, b);
            let (r2, g2, b2) = hsv_to_rgb(h, s, v);
            assert!(
                (r - r2).abs() < 1e-10 && (g - g2).abs() < 1e-10 && (b - b2).abs() < 1e-10,
                "Roundtrip failed for ({r}, {g}, {b}) -> ({h}, {s}, {v}) -> ({r2}, {g2}, {b2})"
            );
        }
    }

    #[test]
    fn test_color_jitter_brightness_only() {
        // With only brightness, all pixels should be scaled uniformly.
        let r = vec![0.5; 4];
        let g = vec![0.4; 4];
        let b = vec![0.3; 4];
        let t = rgb_tensor(&r, &g, &b);
        let jitter = ColorJitter::<f64>::new(0.3, 0.0, 0.0, 0.0).unwrap();
        let out = jitter.apply(t).unwrap();
        let d = out.data().unwrap();
        // All R pixels should have the same value (scaled by the same factor).
        let r_val = d[0];
        for &v in &d[..4] {
            assert!((v - r_val).abs() < 1e-10);
        }
    }

    #[test]
    fn test_color_jitter_f32() {
        let data: Vec<f32> = vec![0.5; 12];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 2, 2], false).unwrap();
        let jitter = ColorJitter::<f32>::new(0.2, 0.2, 0.2, 0.1).unwrap();
        let out = jitter.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 2, 2]);
        for &val in out.data().unwrap() {
            assert!((0.0..=1.0).contains(&val));
        }
    }
}
