// CL-332: Vision Transforms & Augmentation — RandomApply / RandomChoice
use super::rng::random_f64;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_data::Transform;

/// Apply a list of transforms sequentially with probability `p`.
///
/// With probability `p`, all contained transforms are applied in order
/// (like [`Compose`]). With probability `1 - p`, the input is returned
/// unchanged.
///
/// This mirrors `torchvision.transforms.RandomApply`.
pub struct RandomApply<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
    p: f64,
}

impl<T: Float> RandomApply<T> {
    /// Create a new `RandomApply`.
    ///
    /// * `transforms` — the transforms to apply when triggered.
    /// * `p` — probability that the transforms are applied. Must be in `[0.0, 1.0]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `p` is outside `[0, 1]`.
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>, p: f64) -> FerrotorchResult<Self> {
        if !(0.0..=1.0).contains(&p) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("RandomApply: p must be in [0.0, 1.0], got {p}"),
            });
        }
        Ok(Self { transforms, p })
    }
}

impl<T: Float> Transform<T> for RandomApply<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if random_f64() >= self.p {
            return Ok(input);
        }
        let mut current = input;
        for t in &self.transforms {
            current = t.apply(current)?;
        }
        Ok(current)
    }
}

/// Randomly pick one transform from a list and apply it.
///
/// Each contained transform has equal probability `1/n` of being selected.
///
/// This mirrors `torchvision.transforms.RandomChoice`.
pub struct RandomChoice<T: Float> {
    transforms: Vec<Box<dyn Transform<T>>>,
}

impl<T: Float> RandomChoice<T> {
    /// Create a new `RandomChoice`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] if `transforms` is empty.
    pub fn new(transforms: Vec<Box<dyn Transform<T>>>) -> FerrotorchResult<Self> {
        if transforms.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "RandomChoice: transforms list must not be empty".into(),
            });
        }
        Ok(Self { transforms })
    }
}

impl<T: Float> Transform<T> for RandomChoice<T> {
    fn apply(&self, input: Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let n = self.transforms.len();
        let idx = (random_f64() * n as f64) as usize;
        let idx = idx.min(n - 1); // Clamp in case random_f64() yields exactly 1.0.
        self.transforms[idx].apply(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::TensorStorage;
    use ferrotorch_data::Normalize;

    #[test]
    fn test_random_apply_always() {
        // p=1.0: transforms should always be applied.
        let data = vec![10.0_f64, 20.0, 30.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 3], false).unwrap();
        let ra = RandomApply::<f64>::new(
            vec![Box::new(
                Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap(),
            )],
            1.0,
        )
        .unwrap();
        let out = ra.apply(t).unwrap();
        let d = out.data().unwrap();
        // (10 - 1)/1 = 9
        assert!((d[0] - 9.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_apply_never() {
        // p=0.0: transforms should never be applied.
        let data = vec![10.0_f64, 20.0, 30.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 3], false).unwrap();
        let ra = RandomApply::<f64>::new(
            vec![Box::new(
                Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap(),
            )],
            0.0,
        )
        .unwrap();
        let out = ra.apply(t).unwrap();
        let d = out.data().unwrap();
        assert!((d[0] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_apply_empty_transforms() {
        // Even with p=1.0, empty transforms should act as identity.
        let data = vec![5.0_f64, 6.0];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 2], false).unwrap();
        let ra = RandomApply::<f64>::new(vec![], 1.0).unwrap();
        let out = ra.apply(t).unwrap();
        assert_eq!(out.data().unwrap(), &[5.0, 6.0]);
    }

    #[test]
    fn test_random_choice_selects_one() {
        // Two transforms: one subtracts 1.0, the other subtracts 100.0.
        // Over many trials, both should be selected at least once.
        let mut saw_small = false;
        let mut saw_large = false;

        for _ in 0..200 {
            let data = vec![500.0_f64];
            let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1], false).unwrap();
            let rc = RandomChoice::<f64>::new(vec![
                Box::new(Normalize::<f64>::new(vec![1.0], vec![1.0]).unwrap()),
                Box::new(Normalize::<f64>::new(vec![100.0], vec![1.0]).unwrap()),
            ])
            .unwrap();
            let out = rc.apply(t).unwrap();
            let d = out.data().unwrap();
            if (d[0] - 499.0).abs() < 1e-10 {
                saw_small = true;
            }
            if (d[0] - 400.0).abs() < 1e-10 {
                saw_large = true;
            }
        }

        assert!(saw_small, "RandomChoice never selected first transform");
        assert!(saw_large, "RandomChoice never selected second transform");
    }

    #[test]
    fn test_random_choice_single_transform() {
        let data = vec![10.0_f64];
        let t = Tensor::from_storage(TensorStorage::cpu(data), vec![1, 1], false).unwrap();
        let rc = RandomChoice::<f64>::new(vec![Box::new(
            Normalize::<f64>::new(vec![5.0], vec![1.0]).unwrap(),
        )])
        .unwrap();
        let out = rc.apply(t).unwrap();
        let d = out.data().unwrap();
        assert!((d[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_random_apply_is_send_sync() {
        fn assert_send_sync<U: Send + Sync>() {}
        assert_send_sync::<RandomApply<f32>>();
        assert_send_sync::<RandomChoice<f32>>();
    }
}
