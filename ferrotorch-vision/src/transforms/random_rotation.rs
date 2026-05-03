// CL-332: Vision Transforms & Augmentation — RandomRotation
use super::rng::random_f64;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Rotate a `[C, H, W]` tensor by a random angle using bilinear interpolation.
///
/// The angle is sampled uniformly from `[-degrees, +degrees]`. Pixels outside
/// the rotated source region are filled with zero.
///
/// Rotation is performed around the image center, preserving the spatial
/// dimensions. This mirrors `torchvision.transforms.RandomRotation`.
pub struct RandomRotation<T: Float> {
    degrees: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomRotation<T> {
    /// Create a new `RandomRotation` with the given maximum angle in degrees.
    ///
    /// The actual rotation angle for each application is sampled uniformly from
    /// `[-degrees, +degrees]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `degrees` is negative.
    pub fn new(degrees: f64) -> FerrotorchResult<Self> {
        if degrees < 0.0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("RandomRotation: degrees must be non-negative, got {degrees}"),
            });
        }
        Ok(Self {
            degrees,
            _marker: std::marker::PhantomData,
        })
    }
}

/// Bilinear interpolation sample from a single channel stored in row-major
/// order with dimensions `(h, w)`. Returns zero for out-of-bounds coordinates.
fn bilinear_sample<T: Float>(
    data: &[T],
    h: usize,
    w: usize,
    y: f64,
    x: f64,
) -> FerrotorchResult<T> {
    let zero = <T as num_traits::Zero>::zero();
    if x < 0.0 || y < 0.0 {
        return Ok(zero);
    }

    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let x1 = x0 + 1;
    let y1 = y0 + 1;

    if x0 >= w || y0 >= h {
        return Ok(zero);
    }

    let dx: f64 = x - x0 as f64;
    let dy: f64 = y - y0 as f64;

    let v00 = data[y0 * w + x0];
    let v10 = if x1 < w { data[y0 * w + x1] } else { zero };
    let v01 = if y1 < h { data[y1 * w + x0] } else { zero };
    let v11 = if x1 < w && y1 < h {
        data[y1 * w + x1]
    } else {
        zero
    };

    // Bilinear weights.
    let w00: T = cast::<f64, T>((1.0 - dx) * (1.0 - dy))?;
    let w10: T = cast::<f64, T>(dx * (1.0 - dy))?;
    let w01: T = cast::<f64, T>((1.0 - dx) * dy)?;
    let w11: T = cast::<f64, T>(dx * dy)?;

    Ok(v00 * w00 + v10 * w10 + v01 * w01 + v11 * w11)
}

impl<T: Float> Transform<T> for RandomRotation<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomRotation: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        if self.degrees == 0.0 {
            return Ok(input);
        }

        let channels = shape[0];
        let h = shape[1];
        let w = shape[2];

        // Sample angle uniformly from [-degrees, +degrees].
        let angle_deg = self.degrees * (2.0 * random_f64() - 1.0);
        let angle_rad = angle_deg.to_radians();
        let cos_a = angle_rad.cos();
        let sin_a = angle_rad.sin();

        // Rotation center.
        let cx = (w as f64 - 1.0) / 2.0;
        let cy = (h as f64 - 1.0) / 2.0;

        let data = input.data()?;
        let mut output = Vec::with_capacity(data.len());

        for c in 0..channels {
            let ch_data = &data[c * h * w..(c + 1) * h * w];
            for oy in 0..h {
                for ox in 0..w {
                    // Map output (ox, oy) back to input via inverse rotation.
                    let dx = ox as f64 - cx;
                    let dy = oy as f64 - cy;
                    let sx = cos_a * dx + sin_a * dy + cx;
                    let sy = -sin_a * dx + cos_a * dy + cy;
                    output.push(bilinear_sample(ch_data, h, w, sy, sx)?);
                }
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
    fn test_random_rotation_output_shape() {
        let data: Vec<f64> = (0..75).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 5, 5], false).unwrap();
        let rot = RandomRotation::<f64>::new(30.0).unwrap();
        let out = rot.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 5, 5]);
    }

    #[test]
    fn test_random_rotation_zero_degrees() {
        // Zero degrees should return input unchanged.
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t =
            Tensor::from_storage(TensorStorage::cpu(data.clone()), vec![1, 3, 4], false).unwrap();
        let rot = RandomRotation::<f64>::new(0.0).unwrap();
        let out = rot.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &data);
    }

    #[test]
    fn test_random_rotation_preserves_center_pixel() {
        // The center pixel should be approximately preserved after any rotation.
        // Use a 5x5 image with a distinctive center value.
        let mut data = vec![0.0_f64; 25];
        data[12] = 100.0; // center of 5x5
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 5, 5], false).unwrap();
        let rot = RandomRotation::<f64>::new(45.0).unwrap();
        let out = rot.apply(t).unwrap();
        let d = out.data().unwrap();
        // Center pixel (index 12) should still be close to 100.
        assert!(
            d[12] > 50.0,
            "Center pixel after rotation should be close to original, got {}",
            d[12]
        );
    }

    #[test]
    fn test_random_rotation_rejects_non_3d() {
        let data = vec![1.0_f64; 4];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).unwrap();
        let rot = RandomRotation::<f64>::new(10.0).unwrap();
        assert!(rot.apply(t).is_err());
    }

    #[test]
    fn test_bilinear_sample_exact_pixel() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0]; // 2x2
        let val = bilinear_sample(&data, 2, 2, 0.0, 0.0).unwrap();
        assert!((val - 1.0).abs() < 1e-10);
        let val = bilinear_sample(&data, 2, 2, 0.0, 1.0).unwrap();
        assert!((val - 2.0).abs() < 1e-10);
        let val = bilinear_sample(&data, 2, 2, 1.0, 0.0).unwrap();
        assert!((val - 3.0).abs() < 1e-10);
        let val = bilinear_sample(&data, 2, 2, 1.0, 1.0).unwrap();
        assert!((val - 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear_sample_midpoint() {
        let data = vec![0.0_f64, 2.0, 4.0, 6.0]; // 2x2
        // Midpoint (0.5, 0.5) should be average of all 4: (0+2+4+6)/4 = 3
        let val = bilinear_sample(&data, 2, 2, 0.5, 0.5).unwrap();
        assert!((val - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_bilinear_sample_out_of_bounds() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0]; // 2x2
        let val = bilinear_sample(&data, 2, 2, -1.0, 0.0).unwrap();
        assert!((val - 0.0).abs() < 1e-10);
        let val = bilinear_sample(&data, 2, 2, 0.0, -1.0).unwrap();
        assert!((val - 0.0).abs() < 1e-10);
    }
}
