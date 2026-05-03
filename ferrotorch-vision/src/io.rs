//! Image I/O utilities for reading and writing image files.
//!
//! Provides functions to load images from disk into tensors (and vice versa),
//! using the [`image`] crate for format support (PNG, JPEG, BMP, GIF, etc.).

use std::path::Path;

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use image::{ImageBuffer, Rgb, Rgba};

/// Raw image data in HWC (height, width, channels) u8 format.
///
/// This is an intermediate representation between on-disk image files and
/// ferrotorch tensors. Use [`read_image`] to load from disk and
/// [`read_image_as_tensor`] to go directly to a `Tensor`.
///
/// Marked `#[non_exhaustive]` so future format extensions (e.g. a stride
/// field for non-tightly-packed images) can land without breaking
/// struct-literal construction in this crate's tests.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct RawImage {
    /// Pixel data in row-major HWC order, values in `[0, 255]`.
    pub data: Vec<u8>,
    /// Image width in pixels.
    pub width: u32,
    /// Image height in pixels.
    pub height: u32,
    /// Number of channels (3 for RGB, 4 for RGBA).
    pub channels: u32,
}

/// Read an image file (PNG, JPEG, BMP, GIF, etc.) into a [`RawImage`].
///
/// The image is always converted to RGB8 (3 channels). Alpha channels are
/// discarded. Use [`read_image_rgba`] if you need the alpha channel.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the file cannot be opened
/// or decoded.
pub fn read_image(path: impl AsRef<Path>) -> FerrotorchResult<RawImage> {
    let img = image::open(path.as_ref()).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read image '{}': {e}", path.as_ref().display()),
    })?;
    let rgb = img.to_rgb8();
    Ok(RawImage {
        width: rgb.width(),
        height: rgb.height(),
        channels: 3,
        data: rgb.into_raw(),
    })
}

/// Read an image file into a [`RawImage`] preserving the alpha channel (RGBA).
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the file cannot be opened
/// or decoded.
pub fn read_image_rgba(path: impl AsRef<Path>) -> FerrotorchResult<RawImage> {
    let img = image::open(path.as_ref()).map_err(|e| FerrotorchError::InvalidArgument {
        message: format!("failed to read image '{}': {e}", path.as_ref().display()),
    })?;
    let rgba = img.to_rgba8();
    Ok(RawImage {
        width: rgba.width(),
        height: rgba.height(),
        channels: 4,
        data: rgba.into_raw(),
    })
}

/// Read an image file and convert it to a `[C, H, W]` float tensor with
/// values normalized to `[0.0, 1.0]`.
///
/// The image is loaded as RGB (3 channels) and converted to the target float
/// type `T` (typically `f32` or `f64`).
///
/// # Example
///
/// ```no_run
/// use ferrotorch_vision::io::read_image_as_tensor;
/// use ferrotorch_core::{FerrotorchResult, Tensor};
///
/// fn load() -> FerrotorchResult<()> {
///     let tensor: Tensor<f32> = read_image_as_tensor("photo.jpg")?;
///     assert_eq!(tensor.ndim(), 3);
///     assert_eq!(tensor.shape()[0], 3); // C=3 (RGB)
///     Ok(())
/// }
/// ```
///
/// # Errors
///
/// Returns an error if the image cannot be read or decoded.
pub fn read_image_as_tensor<T: Float>(path: impl AsRef<Path>) -> FerrotorchResult<Tensor<T>> {
    let raw = read_image(path)?;
    raw_image_to_tensor(&raw)
}

/// Convert a [`RawImage`] to a `[C, H, W]` float tensor in `[0.0, 1.0]`.
///
/// Performs the HWC -> CHW transposition and scales `u8` values by `1/255`.
pub fn raw_image_to_tensor<T: Float>(raw: &RawImage) -> FerrotorchResult<Tensor<T>> {
    let h = raw.height as usize;
    let w = raw.width as usize;
    let c = raw.channels as usize;

    let scale: T = cast::<f64, T>(255.0)?;
    let mut output = vec![<T as num_traits::Zero>::zero(); c * h * w];

    for row in 0..h {
        for col in 0..w {
            for ch in 0..c {
                let src_idx = row * w * c + col * c + ch;
                let dst_idx = ch * h * w + row * w + col;
                let pixel: T = cast::<u8, T>(raw.data[src_idx])?;
                output[dst_idx] = pixel / scale;
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(output), vec![c, h, w], false)
}

/// Write a [`RawImage`] to a file. The output format is inferred from the
/// file extension (`.png`, `.jpg`, `.bmp`, etc.).
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] if the image cannot be
/// encoded or the file cannot be written.
pub fn write_image(path: impl AsRef<Path>, image: &RawImage) -> FerrotorchResult<()> {
    match image.channels {
        3 => {
            let buf: ImageBuffer<Rgb<u8>, _> =
                ImageBuffer::from_raw(image.width, image.height, image.data.as_slice())
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!(
                            "image data length {} does not match {}x{}x{}",
                            image.data.len(),
                            image.width,
                            image.height,
                            image.channels,
                        ),
                    })?;
            buf.save(path.as_ref())
                .map_err(|e| FerrotorchError::InvalidArgument {
                    message: format!("failed to write image '{}': {e}", path.as_ref().display()),
                })?;
        }
        4 => {
            let buf: ImageBuffer<Rgba<u8>, _> =
                ImageBuffer::from_raw(image.width, image.height, image.data.as_slice())
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!(
                            "image data length {} does not match {}x{}x{}",
                            image.data.len(),
                            image.width,
                            image.height,
                            image.channels,
                        ),
                    })?;
            buf.save(path.as_ref())
                .map_err(|e| FerrotorchError::InvalidArgument {
                    message: format!("failed to write image '{}': {e}", path.as_ref().display()),
                })?;
        }
        _ => {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "write_image supports 3 (RGB) or 4 (RGBA) channels, got {}",
                    image.channels,
                ),
            });
        }
    }
    Ok(())
}

/// Write a `[C, H, W]` float tensor (values in `[0.0, 1.0]`) to an image file.
///
/// The tensor must have 3 or 4 channels. Values are clamped to `[0, 1]` and
/// scaled to `[0, 255]`.
///
/// # Errors
///
/// Returns an error if the tensor is not 3-D, has unsupported channel count,
/// or the file cannot be written.
pub fn write_tensor_as_image<T: Float>(
    path: impl AsRef<Path>,
    tensor: &Tensor<T>,
) -> FerrotorchResult<()> {
    let raw = tensor_to_raw_image(tensor)?;
    write_image(path, &raw)
}

/// Convert a `[C, H, W]` float tensor to a [`RawImage`].
///
/// Values are clamped to `[0.0, 1.0]` and scaled to `[0, 255]`.
pub fn tensor_to_raw_image<T: Float>(tensor: &Tensor<T>) -> FerrotorchResult<RawImage> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "tensor_to_raw_image: expected 3-D tensor [C, H, W], got shape {:?}",
                shape,
            ),
        });
    }

    let c = shape[0];
    let h = shape[1];
    let w = shape[2];

    if c != 3 && c != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("tensor_to_raw_image: expected 3 or 4 channels, got {c}"),
        });
    }

    let zero = <T as num_traits::Zero>::zero();
    let one = <T as num_traits::One>::one();
    let scale: f64 = 255.0;
    let data_slice = tensor.data_vec()?;

    let mut output = vec![0u8; h * w * c];
    for row in 0..h {
        for col in 0..w {
            for ch in 0..c {
                let src_idx = ch * h * w + row * w + col;
                let dst_idx = row * w * c + col * c + ch;
                // Clamp to [0, 1], scale to [0, 255].
                let val = data_slice[src_idx].max(zero).min(one);
                let byte: f64 = cast::<T, f64>(val)? * scale;
                output[dst_idx] = byte.round() as u8;
            }
        }
    }

    Ok(RawImage {
        data: output,
        width: w as u32,
        height: h as u32,
        channels: c as u32,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_read_image_from_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.png");

        // Write a known PNG.
        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_fn(4, 3, |x, y| {
            Rgb([(x as u8) * 60, (y as u8) * 80, 128])
        });
        img.save(&path).unwrap();

        let raw = read_image(&path).unwrap();
        assert_eq!(raw.width, 4);
        assert_eq!(raw.height, 3);
        assert_eq!(raw.channels, 3);
        assert_eq!(raw.data.len(), 4 * 3 * 3);

        // Spot-check pixel (0, 0).
        assert_eq!(raw.data[0], 0); // R: 0*60
        assert_eq!(raw.data[1], 0); // G: 0*80
        assert_eq!(raw.data[2], 128); // B: 128
    }

    #[test]
    fn test_write_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("roundtrip.png");

        // Create a RawImage with known data.
        let w = 3u32;
        let h = 2u32;
        let c = 3u32;
        let mut data = vec![0u8; (w * h * c) as usize];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 17 % 256) as u8;
        }

        let original = RawImage {
            data: data.clone(),
            width: w,
            height: h,
            channels: c,
        };

        write_image(&path, &original).unwrap();
        let loaded = read_image(&path).unwrap();

        assert_eq!(loaded.width, w);
        assert_eq!(loaded.height, h);
        assert_eq!(loaded.channels, c);
        // PNG is lossless, so data should match exactly.
        assert_eq!(loaded.data, data);
    }

    #[test]
    fn test_read_image_as_tensor_shape_and_range() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tensor_test.png");

        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_fn(5, 4, |x, y| {
            Rgb([((x + y) % 256) as u8, ((x * 2 + y) % 256) as u8, 255])
        });
        img.save(&path).unwrap();

        let tensor: Tensor<f32> = read_image_as_tensor(&path).unwrap();

        // Shape should be [C=3, H=4, W=5].
        assert_eq!(tensor.shape(), &[3, 4, 5]);

        // All values should be in [0.0, 1.0].
        let data = tensor.data().unwrap();
        for &val in data {
            assert!(val >= 0.0, "value {val} < 0.0");
            assert!(val <= 1.0, "value {val} > 1.0");
        }

        // The blue channel was all 255 -> should be 1.0.
        // Blue channel starts at index 2 * 4 * 5 = 40.
        let blue_start = 2 * 4 * 5;
        for (j, &val) in data[blue_start..blue_start + 4 * 5].iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-6,
                "blue channel pixel {j} was {val} not 1.0",
            );
        }
    }

    #[test]
    fn test_read_image_as_tensor_f64() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("f64_test.png");

        let img = ImageBuffer::<Rgb<u8>, Vec<u8>>::from_fn(2, 2, |_, _| Rgb([128, 64, 0]));
        img.save(&path).unwrap();

        let tensor: Tensor<f64> = read_image_as_tensor(&path).unwrap();
        assert_eq!(tensor.shape(), &[3, 2, 2]);

        let data = tensor.data().unwrap();
        // Red channel: 128/255 ~ 0.502
        assert!((data[0] - 128.0 / 255.0).abs() < 1e-10);
        // Green channel: 64/255 ~ 0.251
        assert!((data[4] - 64.0 / 255.0).abs() < 1e-10);
        // Blue channel: 0/255 = 0.0
        assert!(data[8].abs() < 1e-10);
    }

    #[test]
    fn test_tensor_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("tensor_rt.png");

        // Create a [3, 2, 2] tensor with known values.
        #[rustfmt::skip]
        let values: Vec<f32> = vec![
            // R channel (2x2)
            0.0, 1.0,
            0.5, 0.25,
            // G channel (2x2)
            1.0, 0.0,
            0.75, 0.5,
            // B channel (2x2)
            0.5, 0.5,
            0.0, 1.0,
        ];
        let tensor =
            Tensor::from_storage(TensorStorage::cpu(values.clone()), vec![3, 2, 2], false).unwrap();

        write_tensor_as_image(&path, &tensor).unwrap();
        let loaded: Tensor<f32> = read_image_as_tensor(&path).unwrap();

        assert_eq!(loaded.shape(), &[3, 2, 2]);

        // Check that values survive the roundtrip within quantization error (1/255).
        let orig = tensor.data().unwrap();
        let back = loaded.data().unwrap();
        for (i, (&a, &b)) in orig.iter().zip(back.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1.0 / 255.0 + 1e-6,
                "element {i}: original={a}, roundtripped={b}",
            );
        }
    }

    #[test]
    fn test_raw_image_data_length_mismatch() {
        let bad = RawImage {
            data: vec![0u8; 10], // Wrong length for 3x2x3.
            width: 3,
            height: 2,
            channels: 3,
        };
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.png");
        assert!(write_image(&path, &bad).is_err());
    }

    #[test]
    fn test_read_nonexistent_file() {
        let result = read_image("/tmp/this_file_does_not_exist_ferrotorch.png");
        assert!(result.is_err());
    }

    #[test]
    fn test_tensor_to_raw_image_clamps() {
        // Values outside [0, 1] should be clamped.
        #[rustfmt::skip]
        let values: Vec<f32> = vec![
            // R
            -0.5, 1.5,
            0.0,  1.0,
            // G
            0.5, 0.5,
            0.5, 0.5,
            // B
            0.0, 0.0,
            0.0, 0.0,
        ];
        let tensor =
            Tensor::from_storage(TensorStorage::cpu(values), vec![3, 2, 2], false).unwrap();

        let raw = tensor_to_raw_image(&tensor).unwrap();
        assert_eq!(raw.width, 2);
        assert_eq!(raw.height, 2);

        // Pixel (0,0): R was -0.5 -> clamped to 0.
        assert_eq!(raw.data[0], 0);
        // Pixel (0,1): R was 1.5 -> clamped to 255.
        assert_eq!(raw.data[3], 255);
    }

    #[test]
    fn test_write_tensor_rejects_non_3d() {
        let tensor =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0; 12]), vec![2, 6], false)
                .unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.png");
        assert!(write_tensor_as_image(&path, &tensor).is_err());
    }
}
