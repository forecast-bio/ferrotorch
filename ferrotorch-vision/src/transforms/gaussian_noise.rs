//! GaussianNoise: additive Gaussian noise for robustness training.
//!
//! Adds i.i.d. Gaussian noise `N(mean, std^2)` to every element of a
//! `[C, H, W]` tensor. Mirrors `torchvision.transforms.v2.GaussianNoise`.
//! CL-458.

use super::rng::random_f64;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;
use num_traits::NumCast;

/// Add i.i.d. Gaussian noise `N(mean, std^2)` to every element of a
/// `[C, H, W]` image tensor.
///
/// `std` can optionally be clamped to the non-negative range; callers
/// are responsible for passing sensible values.
pub struct GaussianNoise<T: Float> {
    mean: f64,
    std: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> GaussianNoise<T> {
    /// Create a new `GaussianNoise` transform with the given mean and
    /// standard deviation.
    ///
    /// # Panics
    ///
    /// Panics if `std < 0`. Zero is allowed (and is a no-op).
    pub fn new(mean: f64, std: f64) -> Self {
        assert!(std >= 0.0, "GaussianNoise: std must be >= 0, got {std}");
        Self {
            mean,
            std,
            _marker: std::marker::PhantomData,
        }
    }
}

/// Draw one sample from N(0, 1) using Box-Muller over two uniform draws.
fn standard_normal_sample() -> f64 {
    // Use Box-Muller. u1 is clamped away from 0 to avoid log(0).
    let u1 = random_f64().max(1e-12);
    let u2 = random_f64();
    let r = (-2.0 * u1.ln()).sqrt();
    let theta = 2.0 * std::f64::consts::PI * u2;
    r * theta.cos()
}

impl<T: Float> Transform<T> for GaussianNoise<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "GaussianNoise: expected 3-D tensor [C, H, W], got shape {shape:?}"
                ),
            });
        }
        if self.std == 0.0 {
            // Degenerate: adding N(mean, 0) is just a constant shift.
            // Return a new tensor with the shift baked in.
            let data = input.data()?;
            let mean_t: T = <T as NumCast>::from(self.mean).unwrap();
            let out: Vec<T> = data.iter().map(|&v| v + mean_t).collect();
            return Tensor::from_storage(TensorStorage::cpu(out), shape, false);
        }

        let data = input.data()?;
        let mut out = Vec::with_capacity(data.len());
        for &v in data {
            let noise = self.mean + self.std * standard_normal_sample();
            let noise_t: T = <T as NumCast>::from(noise).unwrap();
            out.push(v + noise_t);
        }
        Tensor::from_storage(TensorStorage::cpu(out), shape, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transforms::rng::vision_manual_seed;

    #[test]
    fn test_gaussian_noise_output_shape_preserved() {
        let t: Tensor<f32> =
            Tensor::from_storage(TensorStorage::cpu(vec![0.5; 48]), vec![3, 4, 4], false).unwrap();
        let noise = GaussianNoise::<f32>::new(0.0, 0.1);
        let out = noise.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
    }

    #[test]
    fn test_gaussian_noise_zero_std_is_constant_shift() {
        // With std=0, every output pixel should be (input + mean) exactly.
        let t: Tensor<f32> = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0]),
            vec![1, 2, 2],
            false,
        )
        .unwrap();
        let noise = GaussianNoise::<f32>::new(0.5, 0.0);
        let out = noise.apply(t).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d, &[1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn test_gaussian_noise_std_zero_mean_zero_is_identity() {
        let t: Tensor<f64> = Tensor::from_storage(
            TensorStorage::cpu(vec![0.3, -0.7, 2.1]),
            vec![3, 1, 1],
            false,
        )
        .unwrap();
        let noise = GaussianNoise::<f64>::new(0.0, 0.0);
        let out = noise.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &[0.3, -0.7, 2.1]);
    }

    #[test]
    fn test_gaussian_noise_has_approximate_mean_and_std() {
        // Over a large number of elements, the noise should have
        // approximately the requested mean and std. We use std=0.5 so
        // the signal-to-noise ratio makes this check robust.
        vision_manual_seed(12345);
        let zeros: Vec<f64> = vec![0.0; 10_000];
        let t: Tensor<f64> =
            Tensor::from_storage(TensorStorage::cpu(zeros), vec![1, 100, 100], false).unwrap();
        let noise = GaussianNoise::<f64>::new(0.0, 0.5);
        let out = noise.apply(t).unwrap();
        let d = out.data().unwrap();
        let mean: f64 = d.iter().sum::<f64>() / d.len() as f64;
        let var: f64 = d.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / d.len() as f64;
        let std = var.sqrt();
        // Expect mean near 0 and std near 0.5. Loose tolerance because
        // we only have 10k samples.
        assert!(mean.abs() < 0.05, "mean drift: got {mean}");
        assert!((std - 0.5).abs() < 0.05, "std drift: got {std}");
    }

    #[test]
    fn test_gaussian_noise_rejects_non_3d() {
        let t: Tensor<f32> =
            Tensor::from_storage(TensorStorage::cpu(vec![1.0, 2.0]), vec![2], false).unwrap();
        let noise = GaussianNoise::<f32>::new(0.0, 0.1);
        assert!(noise.apply(t).is_err());
    }

    #[test]
    #[should_panic(expected = "std must be >= 0")]
    fn test_gaussian_noise_negative_std_panics() {
        let _ = GaussianNoise::<f32>::new(0.0, -1.0);
    }
}
