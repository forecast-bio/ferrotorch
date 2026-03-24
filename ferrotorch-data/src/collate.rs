// CL-325 — default_collate: stack tensors into batches

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, stack};

/// Stack a `Vec<Tensor<T>>` along a new leading dimension (dim 0).
///
/// This is the default collation strategy: given N tensors of shape `[*S]`,
/// produce a single tensor of shape `[N, *S]`.
///
/// Matches PyTorch's `default_collate` for the tensor case.
///
/// # Errors
///
/// Returns an error if:
/// - `samples` is empty
/// - Any two tensors have different shapes
/// - Any two tensors are on different devices
pub fn default_collate<T: Float>(samples: Vec<Tensor<T>>) -> FerrotorchResult<Tensor<T>> {
    if samples.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "default_collate: empty sample list".into(),
        });
    }
    stack(&samples, 0)
}

/// Collate a batch of `(input, target)` tensor pairs.
///
/// Stacks all inputs along dim 0 and all targets along dim 0, returning
/// a single `(batched_input, batched_target)` pair. This handles the most
/// common supervised-learning pattern.
pub fn default_collate_pair<T: Float>(
    samples: Vec<(Tensor<T>, Tensor<T>)>,
) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
    if samples.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "default_collate_pair: empty sample list".into(),
        });
    }
    let (inputs, targets): (Vec<_>, Vec<_>) = samples.into_iter().unzip();
    Ok((stack(&inputs, 0)?, stack(&targets, 0)?))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};

    fn t32(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        let storage = TensorStorage::cpu(data.to_vec());
        Tensor::from_storage(storage, shape.to_vec(), false).unwrap()
    }

    fn t64(data: &[f64], shape: &[usize]) -> Tensor<f64> {
        let storage = TensorStorage::cpu(data.to_vec());
        Tensor::from_storage(storage, shape.to_vec(), false).unwrap()
    }

    #[test]
    fn test_default_collate_1d() {
        let a = t32(&[1.0, 2.0, 3.0], &[3]);
        let b = t32(&[4.0, 5.0, 6.0], &[3]);
        let batch = default_collate(vec![a, b]).unwrap();
        assert_eq!(batch.shape(), &[2, 3]);
        let data = batch.data_vec().unwrap();
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_default_collate_2d() {
        let a = t32(&[1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let b = t32(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let batch = default_collate(vec![a, b]).unwrap();
        assert_eq!(batch.shape(), &[2, 2, 2]);
    }

    #[test]
    fn test_default_collate_scalars() {
        let a = t32(&[1.0], &[]);
        let b = t32(&[2.0], &[]);
        let c = t32(&[3.0], &[]);
        let batch = default_collate(vec![a, b, c]).unwrap();
        assert_eq!(batch.shape(), &[3]);
        assert_eq!(batch.data_vec().unwrap(), vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_default_collate_empty() {
        let result = default_collate::<f32>(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_collate_pair() {
        let x1 = t32(&[1.0, 2.0], &[2]);
        let y1 = t32(&[0.0], &[1]);
        let x2 = t32(&[3.0, 4.0], &[2]);
        let y2 = t32(&[1.0], &[1]);
        let (bx, by) = default_collate_pair(vec![(x1, y1), (x2, y2)]).unwrap();
        assert_eq!(bx.shape(), &[2, 2]);
        assert_eq!(by.shape(), &[2, 1]);
    }

    #[test]
    fn test_default_collate_pair_empty() {
        let result = default_collate_pair::<f32>(vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_default_collate_f64() {
        let a = t64(&[1.0, 2.0], &[2]);
        let b = t64(&[3.0, 4.0], &[2]);
        let batch = default_collate(vec![a, b]).unwrap();
        assert_eq!(batch.shape(), &[2, 2]);
        assert_eq!(batch.data_vec().unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
    }
}
