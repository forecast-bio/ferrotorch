use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use num_traits::NumCast;

// ---------------------------------------------------------------------------
// Transform trait
// ---------------------------------------------------------------------------

/// A composable transformation applied to a tensor.
///
/// Transforms form the building blocks of data augmentation pipelines.
/// They are chained together via [`Compose`] and applied to individual
/// samples before batching.
pub trait Transform<T: Float>: Send + Sync {
    /// Apply this transform to the input tensor, returning the transformed
    /// tensor or an error.
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>>;
}

// Allow `Box<dyn Transform<T>>` to be used as a `Transform<T>`.
impl<T: Float> Transform<T> for Box<dyn Transform<T>> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        (**self).apply(input)
    }
}

// ---------------------------------------------------------------------------
// Compose — chain multiple transforms
// ---------------------------------------------------------------------------

/// Sequentially applies a list of transforms.
///
/// ```text
/// let pipeline = Compose::new(vec![
///     Box::new(ToTensor),
///     Box::new(Normalize::new(mean, std)?),
///     Box::new(RandomHorizontalFlip::new(0.5)?),
/// ]);
/// let output = pipeline.apply(input)?;
/// ```
pub struct Compose<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
}

// Manual `Debug`: `Box<dyn Transform<T>>` does not implement `Debug`.
// Print the transform count for diagnostics.
impl<T: Float> std::fmt::Debug for Compose<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Compose")
            .field(
                "transforms",
                &format_args!("[{} transform(s)]", self.transforms.len()),
            )
            .finish()
    }
}

impl<T: Float> Compose<T> {
    /// Create a new composition from the given transform list.
    ///
    /// Transforms are applied in order: the output of transform *i* is
    /// passed as input to transform *i + 1*.
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> Self {
        Self { transforms }
    }
}

impl<T: Float> Transform<T> for Compose<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut current = input;
        for transform in &self.transforms {
            current = transform.apply(current)?;
        }
        Ok(current)
    }
}

// ---------------------------------------------------------------------------
// Normalize — per-channel normalization
// ---------------------------------------------------------------------------

/// Normalize a tensor channel-wise with given mean and standard deviation.
///
/// Given a tensor of shape `[C, ...]`, for each channel *c*:
///
/// ```text
/// output[c] = (input[c] - mean[c]) / std[c]
/// ```
///
/// The number of mean/std values must match the first dimension (channel
/// count) of the input tensor.
#[derive(Debug, Clone)]
pub struct Normalize<T: Float> {
    mean: Vec<T>,
    std: Vec<T>,
}

impl<T: Float> Normalize<T> {
    /// Create a new `Normalize` transform.
    ///
    /// `mean` and `std` are given as `f64` slices for ergonomic construction
    /// (matching PyTorch's convention). They are cast to `T` internally.
    ///
    /// # Errors
    ///
    /// Returns an error if any value in `mean` or `std` is out of range for
    /// `T` (e.g., an `f64` value that overflows `f32`), or if `mean.len() !=
    /// std.len()`.
    pub fn new(mean: Vec<f64>, std: Vec<f64>) -> FerrotorchResult<Self> {
        if mean.len() != std.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Normalize: mean and std must have the same length ({} vs {})",
                    mean.len(),
                    std.len()
                ),
            });
        }
        let mean = mean
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                <T as NumCast>::from(v).ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "Normalize: mean[{i}] = {v} is out of range for the target type"
                    ),
                })
            })
            .collect::<FerrotorchResult<Vec<T>>>()?;
        let std = std
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                <T as NumCast>::from(v).ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "Normalize: std[{i}] = {v} is out of range for the target type"
                    ),
                })
            })
            .collect::<FerrotorchResult<Vec<T>>>()?;
        Ok(Self { mean, std })
    }
}

// `Normalize` is Send + Sync because it only holds `Vec<T>` and `T: Float`
// which is `Send + Sync`.
impl<T: Float> Transform<T> for Normalize<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "Normalize: input must be at least 1-D".into(),
            });
        }

        let channels = shape[0];
        if channels != self.mean.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Normalize: expected {} channels but input has {} (shape {:?})",
                    self.mean.len(),
                    channels,
                    shape
                ),
            });
        }

        let data = input.data()?;
        let channel_numel: usize = shape[1..].iter().product();
        let mut output = Vec::with_capacity(data.len());

        for c in 0..channels {
            let m = self.mean[c];
            let s = self.std[c];
            let start = c * channel_numel;
            let end = start + channel_numel;
            for &val in &data[start..end] {
                output.push((val - m) / s);
            }
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// ToTensor — identity / placeholder transform
// ---------------------------------------------------------------------------

/// Identity transform that returns the input unchanged.
///
/// This is a placeholder for a future image-to-tensor conversion. In the
/// current API everything is already a `Tensor`, so `ToTensor` is a no-op.
#[derive(Debug, Clone, Copy, Default)]
pub struct ToTensor;

impl<T: Float> Transform<T> for ToTensor {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        Ok(input)
    }
}

// `ToTensor` is a zero-size unit struct with no fields, so the compiler
// automatically derives `Send + Sync`. The explicit `unsafe impl` blocks
// that were here were unnecessary and have been removed.

// ---------------------------------------------------------------------------
// RandomHorizontalFlip — flip along the last dimension
// ---------------------------------------------------------------------------

/// Randomly flip the tensor along its last dimension with probability `p`.
///
/// For a 3-D tensor `[C, H, W]` this reverses the W dimension, producing
/// a horizontal flip of an image in channel-first layout.
#[derive(Debug, Clone, Copy)]
pub struct RandomHorizontalFlip<T: Float> {
    p: f64,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomHorizontalFlip<T> {
    /// Create a new `RandomHorizontalFlip` with the given probability.
    ///
    /// `p` must be in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `p` is outside
    /// `[0.0, 1.0]`.
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

use std::sync::atomic::{AtomicU64, Ordering};

static GLOBAL_SEED: AtomicU64 = AtomicU64::new(42);
static RNG_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Set the global random seed for reproducibility.
///
/// Resets the internal counter so that subsequent random operations produce
/// the same sequence as a fresh start with this seed.
pub fn manual_seed(seed: u64) {
    GLOBAL_SEED.store(seed, Ordering::SeqCst);
    RNG_COUNTER.store(0, Ordering::SeqCst);
}

/// Generate a random `f64` in [0, 1) using a seedable splitmix64 PRNG.
///
/// Each call atomically increments a global counter, ensuring unique outputs
/// across threads. Use [`manual_seed`] to reset the sequence for
/// reproducibility.
fn random_f64() -> f64 {
    let seed = GLOBAL_SEED.load(Ordering::Relaxed);
    let counter = RNG_COUNTER.fetch_add(1, Ordering::Relaxed);
    // splitmix64 — good statistical properties for a counter-based PRNG.
    let mut state = seed.wrapping_add(counter.wrapping_mul(0x9E3779B97F4A7C15));
    state = (state ^ (state >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
    state = (state ^ (state >> 27)).wrapping_mul(0x94D049BB133111EB);
    state = state ^ (state >> 31);
    (state as f64) / (u64::MAX as f64)
}

impl<T: Float> Transform<T> for RandomHorizontalFlip<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "RandomHorizontalFlip: input must be at least 1-D".into(),
            });
        }

        if random_f64() >= self.p {
            return Ok(input);
        }

        let shape = input.shape().to_vec();
        let data = input.data()?;
        // The `ndim() == 0` guard above returns early with Err, so the
        // shape has at least one dimension here. Propagate as Internal
        // rather than panic if that invariant ever breaks under refactor.
        let last_dim = *shape.last().ok_or_else(|| FerrotorchError::Internal {
            message: "RandomHorizontalFlip: shape unexpectedly empty after ndim guard".into(),
        })?;

        if last_dim <= 1 {
            // Nothing to flip.
            let storage = TensorStorage::cpu(data.to_vec());
            return Tensor::from_storage(storage, shape, false);
        }

        // Reverse along the last dimension by working in chunks of `last_dim`.
        let mut output = Vec::with_capacity(data.len());
        for chunk in data.chunks(last_dim) {
            for i in (0..last_dim).rev() {
                output.push(chunk[i]);
            }
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, shape, false)
    }
}

// ---------------------------------------------------------------------------
// RandomCrop — extract a random spatial crop
// ---------------------------------------------------------------------------

/// Extract a random crop of size `[crop_h, crop_w]` from the last two
/// spatial dimensions of a tensor.
///
/// Expects input of shape `[C, H, W]`. If the input is already the
/// target size the tensor is returned unchanged.
#[derive(Debug, Clone, Copy)]
pub struct RandomCrop<T: Float> {
    height: usize,
    width: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> RandomCrop<T> {
    /// Create a new `RandomCrop` with the desired output spatial size.
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Float> Transform<T> for RandomCrop<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomCrop: input must be at least 2-D, got shape {:?}",
                    shape
                ),
            });
        }

        let ndim = shape.len();
        let in_h = shape[ndim - 2];
        let in_w = shape[ndim - 1];

        if self.height > in_h || self.width > in_w {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RandomCrop: crop size ({}x{}) is larger than input spatial size ({}x{})",
                    self.height, self.width, in_h, in_w
                ),
            });
        }

        // Determine random crop origin.
        let top = if in_h == self.height {
            0
        } else {
            let r = random_f64();
            (r * (in_h - self.height) as f64) as usize
        };
        let left = if in_w == self.width {
            0
        } else {
            let r = random_f64();
            (r * (in_w - self.width) as f64) as usize
        };

        let data = input.data()?;

        // Number of independent "planes" (product of all dims before H, W).
        let planes: usize = shape[..ndim - 2].iter().product();

        let mut output = Vec::with_capacity(planes * self.height * self.width);
        let plane_size = in_h * in_w;

        for p in 0..planes {
            let plane_offset = p * plane_size;
            for row in top..top + self.height {
                let row_start = plane_offset + row * in_w + left;
                output.extend_from_slice(&data[row_start..row_start + self.width]);
            }
        }

        let mut out_shape = shape;
        let n = out_shape.len();
        out_shape[n - 2] = self.height;
        out_shape[n - 1] = self.width;

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, out_shape, false)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: create a [C, H, W] tensor with sequential values.
    fn sequential_tensor(c: usize, h: usize, w: usize) -> Tensor<f64> {
        let numel = c * h * w;
        let data: Vec<f64> = (0..numel).map(|i| i as f64).collect();
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, vec![c, h, w], false).unwrap()
    }

    // ----- Compose -----

    #[test]
    fn test_compose_chains_correctly() {
        // Compose two normalizations: first subtracts 1.0, then subtracts 2.0.
        let t = Tensor::<f64>::from_storage(
            TensorStorage::cpu(vec![10.0, 20.0, 30.0]),
            vec![1, 3],
            false,
        )
        .unwrap();

        let compose = Compose::new(vec![
            Box::new(Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap()),
            Box::new(Normalize::<f64>::new(vec![2.0], vec![1.0]).unwrap()),
        ]);

        let out = compose.apply(t).unwrap();
        let data = out.data().unwrap();
        // (10 - 1)/1 = 9, then (9 - 2)/1 = 7
        assert!((data[0] - 7.0).abs() < 1e-10);
        // (20 - 1)/1 = 19, then (19 - 2)/1 = 17
        assert!((data[1] - 17.0).abs() < 1e-10);
        // (30 - 1)/1 = 29, then (29 - 2)/1 = 27
        assert!((data[2] - 27.0).abs() < 1e-10);
    }

    #[test]
    fn test_compose_empty() {
        let t = Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0]), vec![2], false)
            .unwrap();
        let compose = Compose::new(vec![]);
        let out = compose.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &[1.0, 2.0]);
    }

    // ----- Normalize -----

    #[test]
    fn test_normalize_produces_expected_values() {
        // Two channels, each with 3 elements.
        // Channel 0: [2, 4, 6], mean=4, std=2 => [-1, 0, 1]
        // Channel 1: [10, 20, 30], mean=20, std=10 => [-1, 0, 1]
        let data = vec![2.0, 4.0, 6.0, 10.0, 20.0, 30.0];
        let t = Tensor::<f64>::from_storage(TensorStorage::cpu(data), vec![2, 3], false).unwrap();

        let norm = Normalize::<f64>::new(vec![4.0, 20.0], vec![2.0, 10.0]).unwrap();
        let out = norm.apply(t).unwrap();
        let d = out.data().unwrap();

        assert!((d[0] - -1.0).abs() < 1e-10);
        assert!((d[1] - 0.0).abs() < 1e-10);
        assert!((d[2] - 1.0).abs() < 1e-10);
        assert!((d[3] - -1.0).abs() < 1e-10);
        assert!((d[4] - 0.0).abs() < 1e-10);
        assert!((d[5] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_identity() {
        // mean=0, std=1 should be identity.
        let data = vec![1.0, 2.0, 3.0];
        let t = Tensor::<f64>::from_storage(TensorStorage::cpu(data), vec![1, 3], false).unwrap();

        let norm = Normalize::<f64>::new(vec![0.0], vec![1.0]).unwrap();
        let out = norm.apply(t).unwrap();
        let d = out.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-10);
        assert!((d[1] - 2.0).abs() < 1e-10);
        assert!((d[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_channel_mismatch() {
        // `new` itself now returns Err when mean.len() != std.len().
        // This tests input validation in the constructor.
        assert!(Normalize::<f32>::new(vec![0.0, 0.0], vec![1.0]).is_err());
        // When lengths match but the input tensor's channel count differs, apply() returns Err.
        let t =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3], false)
                .unwrap();
        let norm = Normalize::<f32>::new(vec![0.0, 0.0], vec![1.0, 1.0]).unwrap();
        assert!(norm.apply(t).is_err());
    }

    // ----- ToTensor -----

    #[test]
    fn test_to_tensor_identity() {
        let t =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![3], false)
                .unwrap();
        let out = ToTensor.apply(t.clone()).unwrap();
        assert!(out.is_same(&t));
    }

    // ----- RandomHorizontalFlip -----

    #[test]
    fn test_random_horizontal_flip_always() {
        // p=1.0 should always flip.
        let t = Tensor::<f64>::from_storage(
            TensorStorage::cpu(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            vec![2, 3],
            false,
        )
        .unwrap();
        let flip = RandomHorizontalFlip::<f64>::new(1.0).unwrap();
        let out = flip.apply(t).unwrap();
        let d = out.data().unwrap();
        // Row 0: [1,2,3] -> [3,2,1]
        assert_eq!(d[0], 3.0);
        assert_eq!(d[1], 2.0);
        assert_eq!(d[2], 1.0);
        // Row 1: [4,5,6] -> [6,5,4]
        assert_eq!(d[3], 6.0);
        assert_eq!(d[4], 5.0);
        assert_eq!(d[5], 4.0);
    }

    #[test]
    fn test_random_horizontal_flip_never() {
        // p=0.0 should never flip.
        let t =
            Tensor::<f64>::from_storage(TensorStorage::cpu(vec![1.0, 2.0, 3.0]), vec![1, 3], false)
                .unwrap();
        let flip = RandomHorizontalFlip::<f64>::new(0.0).unwrap();
        let out = flip.apply(t).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_random_horizontal_flip_approximate_fraction() {
        // With p=0.5, run many trials and check the fraction is roughly 0.5.
        let flip = RandomHorizontalFlip::<f64>::new(0.5).unwrap();
        let mut flipped_count = 0;
        let trials = 1000;

        for _ in 0..trials {
            let t = Tensor::<f64>::from_storage(
                TensorStorage::cpu(vec![1.0, 2.0, 3.0]),
                vec![1, 3],
                false,
            )
            .unwrap();
            let out = flip.apply(t).unwrap();
            let d = out.data().unwrap();
            // If flipped, first element becomes 3.0.
            if (d[0] - 3.0).abs() < 1e-10 {
                flipped_count += 1;
            }
        }

        let fraction = flipped_count as f64 / trials as f64;
        // Allow generous margin: p=0.5, fraction should be in [0.3, 0.7].
        assert!(
            fraction > 0.3 && fraction < 0.7,
            "Expected flip fraction near 0.5, got {fraction}"
        );
    }

    // ----- RandomCrop -----

    #[test]
    fn test_random_crop_output_shape() {
        let t = sequential_tensor(3, 10, 10);
        let crop = RandomCrop::<f64>::new(5, 5);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 5, 5]);
    }

    #[test]
    fn test_random_crop_exact_size() {
        // When crop size equals input size, output equals input.
        let t = sequential_tensor(2, 4, 4);
        let expected = t.data().unwrap().to_vec();
        let crop = RandomCrop::<f64>::new(4, 4);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.shape(), &[2, 4, 4]);
        assert_eq!(out.data().unwrap(), &expected);
    }

    #[test]
    fn test_random_crop_too_large() {
        let t = sequential_tensor(1, 4, 4);
        let crop = RandomCrop::<f64>::new(5, 4);
        assert!(crop.apply(t).is_err());
    }

    #[test]
    fn test_random_crop_preserves_channel_count() {
        let t = sequential_tensor(5, 8, 8);
        let crop = RandomCrop::<f64>::new(3, 3);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.shape()[0], 5);
        assert_eq!(out.numel(), 5 * 3 * 3);
    }

    #[test]
    fn test_random_crop_values_are_subset() {
        // Every value in the cropped output should exist in the original.
        let t = sequential_tensor(1, 8, 8);
        let original: std::collections::HashSet<u64> =
            t.data().unwrap().iter().map(|&v| v.to_bits()).collect();
        let crop = RandomCrop::<f64>::new(4, 4);
        let out = crop.apply(t).unwrap();
        for &val in out.data().unwrap() {
            assert!(
                original.contains(&val.to_bits()),
                "Cropped value {val} not found in original"
            );
        }
    }

    // ----- Send + Sync -----

    #[test]
    fn test_transforms_are_send_sync() {
        fn assert_send_sync<U: Send + Sync>() {}
        assert_send_sync::<Compose<f32>>();
        assert_send_sync::<Normalize<f32>>();
        assert_send_sync::<ToTensor>();
        assert_send_sync::<RandomHorizontalFlip<f32>>();
        assert_send_sync::<RandomCrop<f32>>();
    }
}
