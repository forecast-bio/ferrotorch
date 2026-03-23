use ferrotorch_core::{Float, FerrotorchError, FerrotorchResult, Tensor, TensorStorage};
use ferrotorch_data::Transform;

/// Resize spatial dimensions of a `[C, H, W]` tensor to a target `(height, width)`.
///
/// Uses nearest-neighbor interpolation: each output pixel maps to the closest
/// input pixel. This is fast and sufficient for many pipelines; bilinear
/// interpolation can be added later as a separate transform.
pub struct Resize<T: Float> {
    height: usize,
    width: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> Resize<T> {
    /// Create a new `Resize` transform targeting the given spatial size.
    pub fn new(height: usize, width: usize) -> Self {
        Self {
            height,
            width,
            _marker: std::marker::PhantomData,
        }
    }
}

impl<T: Float> Transform<T> for Resize<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let shape = input.shape().to_vec();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Resize: expected 3-D tensor [C, H, W], got shape {:?}",
                    shape
                ),
            });
        }

        let channels = shape[0];
        let in_h = shape[1];
        let in_w = shape[2];
        let out_h = self.height;
        let out_w = self.width;

        let data = input.data_vec()?;
        let mut output = Vec::with_capacity(channels * out_h * out_w);

        for c in 0..channels {
            let channel_offset = c * in_h * in_w;
            for oh in 0..out_h {
                // Nearest-neighbor: map output row to input row.
                let ih = if in_h == 1 {
                    0
                } else {
                    (oh * in_h) / out_h
                };
                for ow in 0..out_w {
                    let iw = if in_w == 1 {
                        0
                    } else {
                        (ow * in_w) / out_w
                    };
                    output.push(data[channel_offset + ih * in_w + iw]);
                }
            }
        }

        let storage = TensorStorage::cpu(output);
        Tensor::from_storage(storage, vec![channels, out_h, out_w], false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_resize_output_shape() {
        // 3x8x8 -> 3x4x4
        let data: Vec<f64> = (0..192).map(|i| i as f64).collect();
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![3, 8, 8], false).unwrap();
        let resize = Resize::<f64>::new(4, 4);
        let out = resize.apply(t).unwrap();
        assert_eq!(out.shape(), &[3, 4, 4]);
    }

    #[test]
    fn test_resize_upscale_shape() {
        // 1x2x2 -> 1x6x6
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2, 2], false).unwrap();
        let resize = Resize::<f64>::new(6, 6);
        let out = resize.apply(t).unwrap();
        assert_eq!(out.shape(), &[1, 6, 6]);
        assert_eq!(out.numel(), 36);
    }

    #[test]
    fn test_resize_identity() {
        // Resize to same size should preserve values.
        let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let t = Tensor::from_storage(
            TensorStorage::cpu(data.clone()),
            vec![1, 3, 3],
            false,
        )
        .unwrap();
        let resize = Resize::<f64>::new(3, 3);
        let out = resize.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &data);
    }

    #[test]
    fn test_resize_nearest_neighbor_values() {
        // 1x2x2 -> 1x4x4 with nearest neighbor should replicate pixels.
        // Input:
        //   1 2
        //   3 4
        // Expected 4x4 output (each pixel maps to nearest):
        //   1 1 2 2
        //   1 1 2 2
        //   3 3 4 4
        //   3 3 4 4
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2, 2], false).unwrap();
        let resize = Resize::<f64>::new(4, 4);
        let out = resize.apply(t).unwrap();
        let d = out.data().unwrap();
        let expected = vec![
            1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(d, &expected);
    }

    #[test]
    fn test_resize_rejects_non_3d() {
        let data = vec![1.0_f64, 2.0, 3.0, 4.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![2, 2], false).unwrap();
        let resize = Resize::<f64>::new(4, 4);
        assert!(resize.apply(t).is_err());
    }
}
