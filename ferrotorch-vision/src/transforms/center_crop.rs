use ferrotorch_core::{Float, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Extract the center region of size `(height, width)` from a `[C, H, W]`
/// tensor.
///
/// If the input spatial dimensions exactly match the requested crop size the
/// tensor data is copied without offset.
pub struct CenterCrop<T: Float> {
    height: usize,
    width: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> CenterCrop<T> {
    /// Create a new `CenterCrop` with the desired output spatial size.
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Float> Transform<T> for CenterCrop<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CenterCrop: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let channels = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];

        if self.height > in_h || self.width > in_w {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CenterCrop: crop size ({}x{}) is larger than input spatial size ({}x{})",
                    self.height, self.width, in_h, in_w
                ),
            });
        }

        // Compute the top-left corner of the center crop.
        let top = (in_h - self.height) / 2;
        let left = (in_w - self.width) / 2;

        let data = input.data_vec()?;
        let mut output = Vec::with_capacity(channels * self.height * self.width);

        for c in 0..channels {
            let channel_offset = c * in_h * in_w;
            for row in top..top + self.height {
                let row_start = channel_offset + row * in_w + left;
                output.extend_from_slice(&data[row_start..row_start + self.width]);
            }
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, vec![channels, self.height, self.width], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_center_crop_output_shape() {
        let data: Vec<f64> = (0..75).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 5, 5], false).unwrap();
        let crop = CenterCrop::<f64>::new(3, 3);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 3, 3]);
    }

    #[test]
    fn test_center_crop_values() {
        // 1-channel 4x4 grid:
        //  0  1  2  3
        //  4  5  6  7
        //  8  9 10 11
        // 12 13 14 15
        //
        // Center 2x2 crop (top=1, left=1):
        //  5  6
        //  9 10
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 4, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(2, 2);
        let out = crop.apply(t).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d, &[5.0, 6.0, 9.0, 10.0]);
    }

    #[test]
    fn test_center_crop_exact_size() {
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = Tensor::from_storage(
            TensorStorage::cpu(data.clone()),
            vec![1, 3, 4],
            false,
        )
        .unwrap();
        let crop = CenterCrop::<f64>::new(3, 4);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.shape(), &[1, 3, 4]);
        assert_eq!(out.data().unwrap(), &data);
    }

    #[test]
    fn test_center_crop_multichannel() {
        // 2-channel 4x4:
        // Channel 0: 0..16
        // Channel 1: 16..32
        // Center 2x2 from each channel.
        let data: Vec<f64> = (0..32).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 4, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(2, 2);
        let out = crop.apply(t).unwrap();
        assert_eq!(out.shape(), &[2, 2, 2]);
        let d = out.data().unwrap();
        // Channel 0: rows 1-2, cols 1-2 -> [5, 6, 9, 10]
        // Channel 1: rows 1-2, cols 1-2 -> [21, 22, 25, 26]
        assert_eq!(d, &[5.0, 6.0, 9.0, 10.0, 21.0, 22.0, 25.0, 26.0]);
    }

    #[test]
    fn test_center_crop_too_large() {
        let data: Vec<f64> = (0..12).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 3, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(5, 4);
        assert!(crop.apply(t).is_err());
    }

    #[test]
    fn test_center_crop_rejects_non_3d() {
        let data: Vec<f64> = (0..16).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![4, 4], false).unwrap();
        let crop = CenterCrop::<f64>::new(2, 2);
        assert!(crop.apply(t).is_err());
    }
}
