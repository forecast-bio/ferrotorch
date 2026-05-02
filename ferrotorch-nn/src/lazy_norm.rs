//! Lazy normalization modules. (#622)
//!
//! `LazyBatchNorm{1,2,3}d` and `LazyInstanceNorm{1,2,3}d` defer
//! `num_features` discovery to the first forward call, then materialize
//! a regular `BatchNorm*d` / `InstanceNorm*d` and forward to it.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::module::Module;
use crate::norm::{
    BatchNorm1d, BatchNorm2d, BatchNorm3d, InstanceNorm1d, InstanceNorm2d, InstanceNorm3d,
};
use crate::parameter::Parameter;

/// Generic helper: extract the channel dim (dim 1) from input shape `[N, C, ...]`.
fn channels_from_input<T: Float>(
    input: &Tensor<T>,
    op: &str,
    expected_ndim: usize,
) -> FerrotorchResult<usize> {
    if input.ndim() != expected_ndim {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op}: expected {expected_ndim}-D input, got {}-D",
                input.ndim()
            ),
        });
    }
    Ok(input.shape()[1])
}

macro_rules! lazy_batchnorm {
    ($name:ident, $inner:ident, $expected_ndim:expr, $kind:literal) => {
        #[doc = concat!("Lazy variant of [`", stringify!($inner), "`] — `num_features` is")]
        #[doc = "discovered from the input's channel dim on the first forward call."]
        #[derive(Debug)]
        pub struct $name<T: Float> {
            eps: f64,
            momentum: f64,
            affine: bool,
            inner: OnceLock<$inner<T>>,
            training: AtomicBool,
        }

        impl<T: Float> $name<T> {
            pub fn new(eps: f64, momentum: f64, affine: bool) -> Self {
                Self {
                    eps,
                    momentum,
                    affine,
                    inner: OnceLock::new(),
                    training: AtomicBool::new(true),
                }
            }

            pub fn is_initialized(&self) -> bool {
                self.inner.get().is_some()
            }

            pub fn num_features(&self) -> Option<usize> {
                self.inner.get().map(|m| {
                    m.parameters()
                        .first()
                        .map(|p| p.tensor().shape()[0])
                        .unwrap_or(0)
                })
            }

            pub fn materialize(&self, num_features: usize) -> FerrotorchResult<()> {
                if self.inner.get().is_none() {
                    let inner =
                        $inner::<T>::new(num_features, self.eps, self.momentum, self.affine)?;
                    let _ = self.inner.set(inner);
                }
                Ok(())
            }
        }

        impl<T: Float> Module<T> for $name<T> {
            fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
                if self.inner.get().is_none() {
                    let c = channels_from_input(input, $kind, $expected_ndim)?;
                    self.materialize(c)?;
                }
                let inner = self.inner.get().expect("inner initialized");
                inner.forward(input)
            }

            fn parameters(&self) -> Vec<&Parameter<T>> {
                self.inner.get().map(|m| m.parameters()).unwrap_or_default()
            }

            fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
                self.inner
                    .get_mut()
                    .map(|m| m.parameters_mut())
                    .unwrap_or_default()
            }

            fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
                self.inner
                    .get()
                    .map(|m| m.named_parameters())
                    .unwrap_or_default()
            }

            fn train(&mut self) {
                self.training.store(true, Ordering::Relaxed);
                if let Some(m) = self.inner.get_mut() {
                    m.train();
                }
            }

            fn eval(&mut self) {
                self.training.store(false, Ordering::Relaxed);
                if let Some(m) = self.inner.get_mut() {
                    m.eval();
                }
            }

            fn is_training(&self) -> bool {
                self.training.load(Ordering::Relaxed)
            }
        }
    };
}

lazy_batchnorm!(LazyBatchNorm1d, BatchNorm1d, 2, "LazyBatchNorm1d"); // BatchNorm1d also accepts 3D
lazy_batchnorm!(LazyBatchNorm2d, BatchNorm2d, 4, "LazyBatchNorm2d");
lazy_batchnorm!(LazyBatchNorm3d, BatchNorm3d, 5, "LazyBatchNorm3d");

// InstanceNorm has a 3-arg ctor (no momentum); use a separate macro.
macro_rules! lazy_instancenorm {
    ($name:ident, $inner:ident, $expected_ndim:expr, $kind:literal) => {
        #[doc = concat!("Lazy variant of [`", stringify!($inner), "`].")]
        #[derive(Debug)]
        pub struct $name<T: Float> {
            eps: f64,
            affine: bool,
            inner: OnceLock<$inner<T>>,
            training: AtomicBool,
        }

        impl<T: Float> $name<T> {
            pub fn new(eps: f64, affine: bool) -> Self {
                Self {
                    eps,
                    affine,
                    inner: OnceLock::new(),
                    training: AtomicBool::new(true),
                }
            }

            pub fn is_initialized(&self) -> bool {
                self.inner.get().is_some()
            }

            pub fn materialize(&self, num_features: usize) -> FerrotorchResult<()> {
                if self.inner.get().is_none() {
                    let inner = $inner::<T>::new(num_features, self.eps, self.affine)?;
                    let _ = self.inner.set(inner);
                }
                Ok(())
            }
        }

        impl<T: Float> Module<T> for $name<T> {
            fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
                if self.inner.get().is_none() {
                    let c = channels_from_input(input, $kind, $expected_ndim)?;
                    self.materialize(c)?;
                }
                self.inner.get().expect("inner").forward(input)
            }

            fn parameters(&self) -> Vec<&Parameter<T>> {
                self.inner.get().map(|m| m.parameters()).unwrap_or_default()
            }

            fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
                self.inner
                    .get_mut()
                    .map(|m| m.parameters_mut())
                    .unwrap_or_default()
            }

            fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
                self.inner
                    .get()
                    .map(|m| m.named_parameters())
                    .unwrap_or_default()
            }

            fn train(&mut self) {
                self.training.store(true, Ordering::Relaxed);
                if let Some(m) = self.inner.get_mut() {
                    m.train();
                }
            }

            fn eval(&mut self) {
                self.training.store(false, Ordering::Relaxed);
                if let Some(m) = self.inner.get_mut() {
                    m.eval();
                }
            }

            fn is_training(&self) -> bool {
                self.training.load(Ordering::Relaxed)
            }
        }
    };
}

lazy_instancenorm!(LazyInstanceNorm1d, InstanceNorm1d, 3, "LazyInstanceNorm1d");
lazy_instancenorm!(LazyInstanceNorm2d, InstanceNorm2d, 4, "LazyInstanceNorm2d");
lazy_instancenorm!(LazyInstanceNorm3d, InstanceNorm3d, 5, "LazyInstanceNorm3d");

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::storage::TensorStorage;

    fn cpu_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn lazy_batchnorm2d_materializes_on_first_forward() {
        let bn: LazyBatchNorm2d<f32> = LazyBatchNorm2d::new(1e-5, 0.1, true);
        assert!(!bn.is_initialized());
        // Input: [N=2, C=4, H=3, W=3] = 72 elements.
        let data: Vec<f32> = (0..72).map(|i| i as f32).collect();
        let input = cpu_tensor(data, &[2, 4, 3, 3]);
        let _out = bn.forward(&input).unwrap();
        assert!(bn.is_initialized());
        assert_eq!(bn.num_features(), Some(4));
    }

    #[test]
    fn lazy_batchnorm2d_rejects_wrong_rank() {
        let bn: LazyBatchNorm2d<f32> = LazyBatchNorm2d::new(1e-5, 0.1, true);
        let input = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let err = bn.forward(&input).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn lazy_batchnorm2d_explicit_materialize() {
        let bn: LazyBatchNorm2d<f32> = LazyBatchNorm2d::new(1e-5, 0.1, true);
        bn.materialize(8).unwrap();
        assert!(bn.is_initialized());
        assert_eq!(bn.num_features(), Some(8));
    }

    #[test]
    fn lazy_instancenorm2d_materializes() {
        let inn: LazyInstanceNorm2d<f32> = LazyInstanceNorm2d::new(1e-5, true);
        assert!(!inn.is_initialized());
        let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let input = cpu_tensor(data, &[1, 4, 3, 3]);
        let _out = inn.forward(&input).unwrap();
        assert!(inn.is_initialized());
    }

    #[test]
    fn lazy_batchnorm3d_materializes_on_5d_input() {
        let bn: LazyBatchNorm3d<f32> = LazyBatchNorm3d::new(1e-5, 0.1, true);
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        // [N=1, C=2, D=2, H=2, W=2] = 16 elements
        let input = cpu_tensor(data, &[1, 2, 2, 2, 2]);
        let _ = bn.forward(&input).unwrap();
        assert!(bn.is_initialized());
    }

    #[test]
    fn lazy_instancenorm3d_explicit_materialize() {
        let inn: LazyInstanceNorm3d<f32> = LazyInstanceNorm3d::new(1e-5, true);
        inn.materialize(4).unwrap();
        assert!(inn.is_initialized());
    }

    #[test]
    fn lazy_norm_train_eval_toggle() {
        let mut bn: LazyBatchNorm2d<f32> = LazyBatchNorm2d::new(1e-5, 0.1, true);
        assert!(bn.is_training());
        bn.eval();
        assert!(!bn.is_training());
        bn.train();
        assert!(bn.is_training());
    }
}
