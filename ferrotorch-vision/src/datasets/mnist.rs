//! MNIST handwritten digit dataset.
//!
//! Provides the classic MNIST dataset of 28x28 grayscale handwritten digits
//! (0-9). Supports both loading from IDX files on disk and generating
//! synthetic data for pipeline testing.
//!
//! # Example
//!
//! ```ignore
//! use ferrotorch_vision::datasets::{Mnist, Split};
//!
//! // Generate synthetic data for testing a training pipeline.
//! let train = Mnist::<f32>::synthetic(Split::Train, 1000);
//! let test  = Mnist::<f32>::synthetic(Split::Test, 200);
//!
//! assert_eq!(train.len(), 1000);
//! let sample = train.get(0).unwrap();
//! assert_eq!(sample.image.shape(), &[1, 28, 28]);
//! assert!(sample.label < 10);
//! ```

use std::path::Path;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Dataset;

/// A single MNIST sample: a 1x28x28 grayscale image and its digit label.
///
/// Marked `#[non_exhaustive]` so future per-sample metadata can be added
/// without breaking struct-literal construction outside this crate.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MnistSample<T: Float> {
    /// Grayscale image tensor with shape `[1, 28, 28]`, values in `[0, 1]`.
    pub image: Tensor<T>,
    /// Digit label in `0..=9`.
    pub label: u8,
}

/// Dataset split selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Split {
    /// Training split (60 000 samples in real MNIST).
    Train,
    /// Test split (10 000 samples in real MNIST).
    Test,
}

/// The MNIST handwritten digit dataset.
///
/// Supports two construction modes:
///
/// - [`Mnist::synthetic`]: generates random images and labels in the correct
///   format, useful for testing training pipelines without downloading data.
/// - [`Mnist::from_dir`]: reads real MNIST IDX files from a directory.
#[derive(Debug)]
pub struct Mnist<T: Float> {
    images: Vec<Tensor<T>>,
    labels: Vec<u8>,
    split: Split,
}

impl<T: Float> Mnist<T> {
    /// Number of pixel rows in an MNIST image.
    pub const HEIGHT: usize = 28;
    /// Number of pixel columns in an MNIST image.
    pub const WIDTH: usize = 28;
    /// Number of channels (grayscale).
    pub const CHANNELS: usize = 1;
    /// Number of digit classes.
    pub const NUM_CLASSES: usize = 10;

    /// Create a synthetic MNIST dataset with `num_samples` randomly generated
    /// samples.
    ///
    /// Each image is filled with random values in `[0, 1]` and assigned a
    /// random label in `0..10`. This is useful for smoke-testing data
    /// pipelines and training loops without needing the real dataset files.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if the generated `f64`
    /// pixel values cannot be cast to the target float type `T` (in practice
    /// this never happens for IEEE 754 floats given values in `[0, 1]`).
    pub fn synthetic(split: Split, num_samples: usize) -> FerrotorchResult<Self> {
        let mut images = Vec::with_capacity(num_samples);
        let mut labels = Vec::with_capacity(num_samples);

        // Simple seeded xorshift so tests are deterministic per-split.
        let mut state: u64 = match split {
            Split::Train => 0xdead_beef_cafe_0001,
            Split::Test => 0xdead_beef_cafe_0002,
        };

        let numel = Self::CHANNELS * Self::HEIGHT * Self::WIDTH;

        for _ in 0..num_samples {
            // Generate pixel data in [0, 1].
            let mut data = Vec::with_capacity(numel);
            for _ in 0..numel {
                state = xorshift64(state);
                let f = (state as f64) / (u64::MAX as f64);
                data.push(cast::<f64, T>(f)?);
            }
            let storage = TensorStorage::cpu(data);
            let tensor = Tensor::from_storage(
                storage,
                vec![Self::CHANNELS, Self::HEIGHT, Self::WIDTH],
                false,
            )?;
            images.push(tensor);

            state = xorshift64(state);
            let label = (state % Self::NUM_CLASSES as u64) as u8;
            labels.push(label);
        }

        Ok(Self {
            images,
            labels,
            split,
        })
    }

    /// Load MNIST from IDX files in `root`.
    ///
    /// Expects the standard IDX file layout:
    /// - `train-images-idx3-ubyte` / `train-labels-idx1-ubyte` for [`Split::Train`]
    /// - `t10k-images-idx3-ubyte` / `t10k-labels-idx1-ubyte` for [`Split::Test`]
    ///
    /// The IDX binary format stores big-endian header fields followed by raw
    /// unsigned-byte pixel / label data. Pixel values are normalized to
    /// `[0, 1]` by dividing by 255.
    ///
    /// Returns an error if the files do not exist or contain invalid data.
    /// Automatic download is not yet supported.
    pub fn from_dir<P: AsRef<Path>>(root: P, split: Split) -> FerrotorchResult<Self> {
        let root = root.as_ref();
        let (images_file, labels_file) = match split {
            Split::Train => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            Split::Test => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        };

        let images_path = root.join(images_file);
        let labels_path = root.join(labels_file);

        if !images_path.exists() || !labels_path.exists() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MNIST IDX files not found in '{}'. \
                     Expected '{}' and '{}'. \
                     Automatic download is not yet supported -- \
                     please download from http://yann.lecun.com/exdb/mnist/ \
                     and extract into the root directory.",
                    root.display(),
                    images_file,
                    labels_file,
                ),
            });
        }

        let img_bytes =
            std::fs::read(&images_path).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to read '{}': {e}", images_path.display()),
            })?;
        let lbl_bytes =
            std::fs::read(&labels_path).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to read '{}': {e}", labels_path.display()),
            })?;

        // --- Parse image IDX3 file ---
        if img_bytes.len() < 16 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "images file '{}' is too short ({} bytes, need at least 16)",
                    images_path.display(),
                    img_bytes.len(),
                ),
            });
        }

        let img_magic = read_u32_be(&img_bytes, 0);
        if img_magic != 2051 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "images file has wrong magic number: expected 2051, got {img_magic}",
                ),
            });
        }

        let num_images = read_u32_be(&img_bytes, 4) as usize;
        let rows = read_u32_be(&img_bytes, 8) as usize;
        let cols = read_u32_be(&img_bytes, 12) as usize;
        let pixels_per_image = rows * cols;
        let expected_img_len = 16 + num_images * pixels_per_image;

        if img_bytes.len() < expected_img_len {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "images file is truncated: expected at least {expected_img_len} bytes \
                     for {num_images} images of {rows}x{cols}, got {}",
                    img_bytes.len(),
                ),
            });
        }

        // --- Parse label IDX1 file ---
        if lbl_bytes.len() < 8 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "labels file '{}' is too short ({} bytes, need at least 8)",
                    labels_path.display(),
                    lbl_bytes.len(),
                ),
            });
        }

        let lbl_magic = read_u32_be(&lbl_bytes, 0);
        if lbl_magic != 2049 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "labels file has wrong magic number: expected 2049, got {lbl_magic}",
                ),
            });
        }

        let num_labels = read_u32_be(&lbl_bytes, 4) as usize;
        let expected_lbl_len = 8 + num_labels;

        if lbl_bytes.len() < expected_lbl_len {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "labels file is truncated: expected at least {expected_lbl_len} bytes \
                     for {num_labels} labels, got {}",
                    lbl_bytes.len(),
                ),
            });
        }

        if num_images != num_labels {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "image/label count mismatch: {num_images} images but {num_labels} labels",
                ),
            });
        }

        // --- Build tensors ---
        let inv_255 = 1.0_f64 / 255.0;
        let mut images = Vec::with_capacity(num_images);
        let mut labels = Vec::with_capacity(num_labels);

        for i in 0..num_images {
            let pixel_offset = 16 + i * pixels_per_image;
            let raw_pixels = &img_bytes[pixel_offset..pixel_offset + pixels_per_image];

            let data: Vec<T> = raw_pixels
                .iter()
                .map(|&b| cast::<f64, T>(b as f64 * inv_255))
                .collect::<FerrotorchResult<Vec<T>>>()?;

            let storage = TensorStorage::cpu(data);
            let tensor = Tensor::from_storage(storage, vec![Self::CHANNELS, rows, cols], false)?;
            images.push(tensor);

            let label = lbl_bytes[8 + i];
            labels.push(label);
        }

        Ok(Self {
            images,
            labels,
            split,
        })
    }

    /// Which split this dataset represents.
    pub fn split(&self) -> Split {
        self.split
    }
}

impl<T: Float + 'static> Dataset for Mnist<T> {
    type Sample = MnistSample<T>;

    fn len(&self) -> usize {
        self.images.len()
    }

    fn get(&self, index: usize) -> FerrotorchResult<Self::Sample> {
        if index >= self.images.len() {
            return Err(FerrotorchError::IndexOutOfBounds {
                index,
                axis: 0,
                size: self.images.len(),
            });
        }
        Ok(MnistSample {
            image: self.images[index].clone(),
            label: self.labels[index],
        })
    }
}

/// xorshift64 PRNG step.
#[inline]
fn xorshift64(mut state: u64) -> u64 {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    state
}

/// Read a big-endian `u32` from `buf` starting at `offset`.
#[inline]
fn read_u32_be(buf: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_synthetic_train_len() {
        let ds = Mnist::<f32>::synthetic(Split::Train, 100).unwrap();
        assert_eq!(ds.len(), 100);
        assert!(!ds.is_empty());
    }

    #[test]
    fn test_synthetic_test_len() {
        let ds = Mnist::<f32>::synthetic(Split::Test, 50).unwrap();
        assert_eq!(ds.len(), 50);
    }

    #[test]
    fn test_synthetic_empty() {
        let ds = Mnist::<f32>::synthetic(Split::Train, 0).unwrap();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn test_sample_image_shape() {
        let ds = Mnist::<f32>::synthetic(Split::Train, 10).unwrap();
        let sample = ds.get(0).unwrap();
        assert_eq!(sample.image.shape(), &[1, 28, 28]);
        assert_eq!(sample.image.numel(), 28 * 28);
    }

    #[test]
    fn test_sample_image_values_in_range() {
        let ds = Mnist::<f32>::synthetic(Split::Train, 5).unwrap();
        for i in 0..5 {
            let sample = ds.get(i).unwrap();
            let data = sample.image.data().unwrap();
            for &v in data {
                assert!((0.0..=1.0).contains(&v), "pixel value out of [0,1]: {v}");
            }
        }
    }

    #[test]
    fn test_label_range() {
        let ds = Mnist::<f32>::synthetic(Split::Train, 200).unwrap();
        for i in 0..200 {
            let sample = ds.get(i).unwrap();
            assert!(sample.label < 10, "label out of range: {}", sample.label);
        }
    }

    #[test]
    fn test_out_of_bounds() {
        let ds = Mnist::<f32>::synthetic(Split::Train, 10).unwrap();
        assert!(ds.get(10).is_err());
        assert!(ds.get(100).is_err());
    }

    #[test]
    fn test_split_accessor() {
        let train = Mnist::<f32>::synthetic(Split::Train, 1).unwrap();
        let test = Mnist::<f32>::synthetic(Split::Test, 1).unwrap();
        assert_eq!(train.split(), Split::Train);
        assert_eq!(test.split(), Split::Test);
    }

    #[test]
    fn test_f64_support() {
        let ds = Mnist::<f64>::synthetic(Split::Train, 5).unwrap();
        let sample = ds.get(0).unwrap();
        assert_eq!(sample.image.shape(), &[1, 28, 28]);
        let data = sample.image.data().unwrap();
        for &v in data {
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn test_train_test_different_data() {
        let train = Mnist::<f32>::synthetic(Split::Train, 5).unwrap();
        let test = Mnist::<f32>::synthetic(Split::Test, 5).unwrap();
        // Different seeds should produce different first samples.
        let t0 = train.get(0).unwrap();
        let e0 = test.get(0).unwrap();
        let t_data = t0.image.data().unwrap();
        let e_data = e0.image.data().unwrap();
        assert_ne!(t_data, e_data, "train and test splits should differ");
    }

    #[test]
    fn test_from_dir_missing() {
        let result = Mnist::<f32>::from_dir("/nonexistent/path", Split::Train);
        assert!(result.is_err());
    }

    #[test]
    fn test_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Mnist<f32>>();
        assert_send_sync::<MnistSample<f32>>();
    }

    // --- IDX parsing tests ---

    /// Build a synthetic IDX3 images file (magic 2051).
    fn make_idx3_images(num: u32, rows: u32, cols: u32, pixels: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&2051u32.to_be_bytes());
        buf.extend_from_slice(&num.to_be_bytes());
        buf.extend_from_slice(&rows.to_be_bytes());
        buf.extend_from_slice(&cols.to_be_bytes());
        buf.extend_from_slice(pixels);
        buf
    }

    /// Build a synthetic IDX1 labels file (magic 2049).
    fn make_idx1_labels(num: u32, labels: &[u8]) -> Vec<u8> {
        let mut buf = Vec::new();
        buf.extend_from_slice(&2049u32.to_be_bytes());
        buf.extend_from_slice(&num.to_be_bytes());
        buf.extend_from_slice(labels);
        buf
    }

    /// Write a pair of tiny synthetic IDX files into a temp directory and
    /// return the directory path.
    fn write_synthetic_idx(
        dir: &std::path::Path,
        split: Split,
        num: u32,
        rows: u32,
        cols: u32,
        pixels: &[u8],
        labels: &[u8],
    ) {
        let (img_name, lbl_name) = match split {
            Split::Train => ("train-images-idx3-ubyte", "train-labels-idx1-ubyte"),
            Split::Test => ("t10k-images-idx3-ubyte", "t10k-labels-idx1-ubyte"),
        };
        std::fs::write(
            dir.join(img_name),
            make_idx3_images(num, rows, cols, pixels),
        )
        .unwrap();
        std::fs::write(dir.join(lbl_name), make_idx1_labels(num, labels)).unwrap();
    }

    #[test]
    fn test_from_dir_parses_single_image() {
        let dir = tempfile::tempdir().unwrap();
        // 1 image, 2x3 pixels (small for testing, not 28x28).
        let pixels: Vec<u8> = vec![0, 51, 102, 153, 204, 255];
        let labels: Vec<u8> = vec![7];
        write_synthetic_idx(dir.path(), Split::Train, 1, 2, 3, &pixels, &labels);

        let ds = Mnist::<f32>::from_dir(dir.path(), Split::Train).unwrap();
        assert_eq!(ds.len(), 1);

        let sample = ds.get(0).unwrap();
        assert_eq!(sample.label, 7);
        // Shape: [1, 2, 3] (channels=1, rows=2, cols=3).
        assert_eq!(sample.image.shape(), &[1, 2, 3]);

        let data = sample.image.data().unwrap();
        assert_eq!(data.len(), 6);
        // Verify normalization: pixel / 255.
        let expected: Vec<f32> = pixels.iter().map(|&b| b as f32 / 255.0).collect();
        for (got, want) in data.iter().zip(expected.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "pixel mismatch: got {got}, want {want}",
            );
        }
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_from_dir_parses_multiple_images() {
        let dir = tempfile::tempdir().unwrap();
        // 3 images of 2x2.
        let pixels: Vec<u8> = vec![
            10, 20, 30, 40, // image 0
            50, 60, 70, 80, // image 1
            90, 100, 110, 120, // image 2
        ];
        let labels: Vec<u8> = vec![0, 5, 9];
        write_synthetic_idx(dir.path(), Split::Train, 3, 2, 2, &pixels, &labels);

        let ds = Mnist::<f32>::from_dir(dir.path(), Split::Train).unwrap();
        assert_eq!(ds.len(), 3);

        for i in 0..3 {
            let sample = ds.get(i).unwrap();
            assert_eq!(sample.image.shape(), &[1, 2, 2]);
            assert_eq!(sample.label, labels[i]);

            let data = sample.image.data().unwrap();
            let offset = i * 4;
            for j in 0..4 {
                let expected = pixels[offset + j] as f32 / 255.0;
                assert!(
                    (data[j] - expected).abs() < 1e-6,
                    "image {i}, pixel {j}: got {}, want {expected}",
                    data[j],
                );
            }
        }
    }

    #[test]
    fn test_from_dir_test_split() {
        let dir = tempfile::tempdir().unwrap();
        let pixels: Vec<u8> = vec![128; 4];
        let labels: Vec<u8> = vec![3];
        write_synthetic_idx(dir.path(), Split::Test, 1, 2, 2, &pixels, &labels);

        let ds = Mnist::<f32>::from_dir(dir.path(), Split::Test).unwrap();
        assert_eq!(ds.len(), 1);
        assert_eq!(ds.split(), Split::Test);
        assert_eq!(ds.get(0).unwrap().label, 3);
    }

    #[test]
    fn test_from_dir_f64() {
        let dir = tempfile::tempdir().unwrap();
        let pixels: Vec<u8> = vec![0, 127, 255, 64];
        let labels: Vec<u8> = vec![2];
        write_synthetic_idx(dir.path(), Split::Train, 1, 2, 2, &pixels, &labels);

        let ds = Mnist::<f64>::from_dir(dir.path(), Split::Train).unwrap();
        let sample = ds.get(0).unwrap();
        let data = sample.image.data().unwrap();

        for (i, &pixel) in pixels.iter().enumerate() {
            let expected = pixel as f64 / 255.0;
            assert!(
                (data[i] - expected).abs() < 1e-12,
                "f64 pixel {i}: got {}, want {expected}",
                data[i],
            );
        }
    }

    #[test]
    fn test_from_dir_pixel_normalization_boundaries() {
        let dir = tempfile::tempdir().unwrap();
        // Test exact boundary values: 0 -> 0.0, 255 -> 1.0.
        let pixels: Vec<u8> = vec![0, 255];
        let labels: Vec<u8> = vec![0];
        write_synthetic_idx(dir.path(), Split::Train, 1, 1, 2, &pixels, &labels);

        let ds = Mnist::<f64>::from_dir(dir.path(), Split::Train).unwrap();
        let sample = ds.get(0).unwrap();
        let data = sample.image.data().unwrap();
        assert_eq!(data[0], 0.0);
        assert_eq!(data[1], 1.0);
    }

    #[test]
    fn test_from_dir_wrong_image_magic() {
        let dir = tempfile::tempdir().unwrap();
        // Write images file with wrong magic, valid labels file.
        let mut bad_img = Vec::new();
        bad_img.extend_from_slice(&9999u32.to_be_bytes());
        bad_img.extend_from_slice(&1u32.to_be_bytes()); // num
        bad_img.extend_from_slice(&2u32.to_be_bytes()); // rows
        bad_img.extend_from_slice(&2u32.to_be_bytes()); // cols
        bad_img.extend_from_slice(&[0; 4]); // pixels
        std::fs::write(dir.path().join("train-images-idx3-ubyte"), bad_img).unwrap();
        std::fs::write(
            dir.path().join("train-labels-idx1-ubyte"),
            make_idx1_labels(1, &[0]),
        )
        .unwrap();

        let result = Mnist::<f32>::from_dir(dir.path(), Split::Train);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("magic"),
            "error should mention magic number, got: {msg}",
        );
    }

    #[test]
    fn test_from_dir_wrong_label_magic() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("train-images-idx3-ubyte"),
            make_idx3_images(1, 2, 2, &[0; 4]),
        )
        .unwrap();

        let mut bad_lbl = Vec::new();
        bad_lbl.extend_from_slice(&1234u32.to_be_bytes());
        bad_lbl.extend_from_slice(&1u32.to_be_bytes());
        bad_lbl.push(0);
        std::fs::write(dir.path().join("train-labels-idx1-ubyte"), bad_lbl).unwrap();

        let result = Mnist::<f32>::from_dir(dir.path(), Split::Train);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("magic"),
            "error should mention magic number, got: {msg}",
        );
    }

    #[test]
    fn test_from_dir_truncated_images() {
        let dir = tempfile::tempdir().unwrap();
        // Claim 2 images of 2x2 but only provide 4 bytes of pixel data (enough for 1).
        let pixels: Vec<u8> = vec![0; 4];
        write_synthetic_idx(dir.path(), Split::Train, 2, 2, 2, &pixels, &[0, 1]);

        let result = Mnist::<f32>::from_dir(dir.path(), Split::Train);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("truncated"),
            "error should mention truncation, got: {msg}",
        );
    }

    #[test]
    fn test_from_dir_truncated_labels() {
        let dir = tempfile::tempdir().unwrap();
        // Images claim 2 samples, labels file claims 2 but only has 1 byte of data.
        let pixels: Vec<u8> = vec![0; 8];
        std::fs::write(
            dir.path().join("train-images-idx3-ubyte"),
            make_idx3_images(2, 2, 2, &pixels),
        )
        .unwrap();

        let mut short_lbl = Vec::new();
        short_lbl.extend_from_slice(&2049u32.to_be_bytes());
        short_lbl.extend_from_slice(&2u32.to_be_bytes());
        short_lbl.push(0); // only 1 label byte instead of 2
        std::fs::write(dir.path().join("train-labels-idx1-ubyte"), short_lbl).unwrap();

        let result = Mnist::<f32>::from_dir(dir.path(), Split::Train);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("truncated"),
            "error should mention truncation, got: {msg}",
        );
    }

    #[test]
    fn test_from_dir_count_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        // 2 images but 3 labels.
        let pixels: Vec<u8> = vec![0; 8]; // 2 images of 2x2
        std::fs::write(
            dir.path().join("train-images-idx3-ubyte"),
            make_idx3_images(2, 2, 2, &pixels),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("train-labels-idx1-ubyte"),
            make_idx1_labels(3, &[0, 1, 2]),
        )
        .unwrap();

        let result = Mnist::<f32>::from_dir(dir.path(), Split::Train);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("mismatch"),
            "error should mention count mismatch, got: {msg}",
        );
    }

    #[test]
    fn test_from_dir_images_file_too_short() {
        let dir = tempfile::tempdir().unwrap();
        // Write a file that is too short to even contain the header.
        std::fs::write(dir.path().join("train-images-idx3-ubyte"), [0u8; 10]).unwrap();
        std::fs::write(
            dir.path().join("train-labels-idx1-ubyte"),
            make_idx1_labels(0, &[]),
        )
        .unwrap();

        let result = Mnist::<f32>::from_dir(dir.path(), Split::Train);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("too short"),
            "error should mention file too short, got: {msg}",
        );
    }

    #[test]
    fn test_from_dir_labels_file_too_short() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("train-images-idx3-ubyte"),
            make_idx3_images(0, 1, 1, &[]),
        )
        .unwrap();
        // Write a labels file with only 4 bytes (need at least 8 for header).
        std::fs::write(dir.path().join("train-labels-idx1-ubyte"), [0u8; 4]).unwrap();

        let result = Mnist::<f32>::from_dir(dir.path(), Split::Train);
        assert!(result.is_err());
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains("too short"),
            "error should mention file too short, got: {msg}",
        );
    }

    #[test]
    fn test_from_dir_zero_images() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_idx(dir.path(), Split::Train, 0, 28, 28, &[], &[]);

        let ds = Mnist::<f32>::from_dir(dir.path(), Split::Train).unwrap();
        assert!(ds.is_empty());
        assert_eq!(ds.len(), 0);
    }

    #[test]
    fn test_read_u32_be() {
        let buf = [0x00, 0x00, 0x08, 0x03]; // 2051
        assert_eq!(read_u32_be(&buf, 0), 2051);

        let buf2 = [0x00, 0x00, 0x08, 0x01]; // 2049
        assert_eq!(read_u32_be(&buf2, 0), 2049);

        // Test with offset.
        let buf3 = [0xFF, 0xFF, 0x00, 0x00, 0x00, 0x0A]; // 10 at offset 2
        assert_eq!(read_u32_be(&buf3, 2), 10);
    }
}
