use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Convert an image-like tensor from `[H, W, C]` layout with values in
/// `[0, 255]` to `[C, H, W]` layout with values in `[0.0, 1.0]`.
///
/// This mirrors `torchvision.transforms.ToTensor` which:
/// 1. Transposes HWC -> CHW.
/// 2. Divides by 255 to normalize into `[0, 1]`.
///
/// The input must be a 3-D tensor. If the input is already in CHW float
/// format, use the identity `ferrotorch_data::ToTensor` instead.
pub struct VisionToTensor<T: Float> {
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> VisionToTensor<T> {
    /// Create a new `VisionToTensor` transform.
    pub fn new() -> Self {
        Self {
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Float> Default for VisionToTensor<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Float> Transform<T> for VisionToTensor<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "VisionToTensor: expected 3-D tensor [H, W, C], got shape {:?}",
                    shape
                ),
            });
        }

        let h = shape[0];
        let w = shape[1];
        let c = shape[2];

        let scale: T = cast::<f64, T>(255.0)?;
        let data = input.data_vec()?;

        // Transpose HWC -> CHW and divide by 255.
        let mut output = vec![<T as num_traits::Zero>::zero(); c * h * w];
        for row in 0..h {
            for col in 0..w {
                for ch in 0..c {
                    let src_idx = row * w * c + col * c + ch;
                    let dst_idx = ch * h * w + row * w + col;
                    output[dst_idx] = data[src_idx] / scale;
                }
            }
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, vec![c, h, w], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_to_tensor_transposes_hwc_to_chw() {
        // Input: [2, 3, 3] in HWC (H=2, W=3, C=3)
        // Each pixel has (R, G, B) = channel values.
        // Pixel (0,0): (255, 0, 0)
        // Pixel (0,1): (0, 255, 0)
        // Pixel (0,2): (0, 0, 255)
        // Pixel (1,0): (128, 128, 128)
        // Pixel (1,1): (64, 64, 64)
        // Pixel (1,2): (0, 0, 0)
        #[rustfmt::skip]
        let data: Vec<f64> = vec![
            255.0, 0.0, 0.0,    0.0, 255.0, 0.0,    0.0, 0.0, 255.0,
            128.0, 128.0, 128.0, 64.0, 64.0, 64.0,  0.0, 0.0, 0.0,
        ];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 3, 3], false).unwrap();
        let transform = VisionToTensor::<f64>::new();
        let out = transform.apply(t).unwrap();

        // Output shape should be [C=3, H=2, W=3].
        assert_eq!(out.shape(), &[3, 2, 3]);

        let d = out.data().unwrap();

        // Channel 0 (Red): [255/255, 0, 0, 128/255, 64/255, 0]
        assert!((d[0] - 1.0).abs() < 1e-10); // (0,0) R
        assert!((d[1] - 0.0).abs() < 1e-10); // (0,1) R
        assert!((d[2] - 0.0).abs() < 1e-10); // (0,2) R
        assert!((d[3] - 128.0 / 255.0).abs() < 1e-10); // (1,0) R
        assert!((d[4] - 64.0 / 255.0).abs() < 1e-10); // (1,1) R
        assert!((d[5] - 0.0).abs() < 1e-10); // (1,2) R
    }

    #[test]
    fn test_to_tensor_scales_to_unit_range() {
        // All 255s should become 1.0.
        let data: Vec<f64> = vec![255.0; 12]; // [2, 2, 3]
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2, 3], false).unwrap();
        let transform = VisionToTensor::<f64>::new();
        let out = transform.apply(t).unwrap();
        for &val in out.data().unwrap() {
            assert!((val - 1.0).abs() < 1e-10);
        }

        // All 0s should stay 0.0.
        let data: Vec<f64> = vec![0.0; 12];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2, 3], false).unwrap();
        let out = transform.apply(t).unwrap();
        for &val in out.data().unwrap() {
            assert!(val.abs() < 1e-10);
        }
    }

    #[test]
    fn test_to_tensor_single_pixel() {
        // [1, 1, 3] -> [3, 1, 1], values /255
        let data = vec![51.0_f64, 102.0, 153.0]; // ~0.2, ~0.4, ~0.6
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1, 3], false).unwrap();
        let transform = VisionToTensor::<f64>::new();
        let out = transform.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 1, 1]);
        let d = out.data().unwrap();
        assert!((d[0] - 51.0 / 255.0).abs() < 1e-10);
        assert!((d[1] - 102.0 / 255.0).abs() < 1e-10);
        assert!((d[2] - 153.0 / 255.0).abs() < 1e-10);
    }

    #[test]
    fn test_to_tensor_rejects_non_3d() {
        let data = vec![1.0_f64; 8];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 4], false).unwrap();
        let transform = VisionToTensor::<f64>::new();
        assert!(transform.apply(t).is_err());
    }

    #[test]
    fn test_to_tensor_f32() {
        let data = vec![127.5_f32; 3]; // [1, 1, 3]
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1, 3], false).unwrap();
        let transform = VisionToTensor::<f32>::new();
        let out = transform.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 1, 1]);
        let d = out.data().unwrap();
        assert!((d[0] - 0.5).abs() < 1e-5);
    }
}
