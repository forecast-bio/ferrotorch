//! Lazy variants of [`Conv1d`](super::Conv1d), [`Conv2d`](super::Conv2d),
//! and [`Conv3d`](super::Conv3d).
//!
//! Like [`LazyLinear`](super::LazyLinear), these modules defer parameter
//! allocation until the first forward call, at which point the input
//! tensor's channel dimension (`dim 1` for the standard `[B, C, ...]`
//! layout) is taken as `in_channels`, and the underlying `Conv{1,2,3}d`
//! is constructed and stored.
//!
//! Mirrors `torch.nn.LazyConv1d`, `torch.nn.LazyConv2d`, and
//! `torch.nn.LazyConv3d`.
//!
//! # Thread safety
//!
//! Materialization uses [`std::sync::OnceLock`] so the first forward call
//! across any number of threads initializes the parameters exactly once.
//!
//! # Design
//!
//! Each lazy conv wraps a `OnceLock<ConvNd<T>>`. On first forward, the
//! input's `dim 1` is inspected and a `ConvNd::new(...)` is constructed
//! with the user's kernel_size / stride / padding and the newly-discovered
//! in_channels. Subsequent forward calls delegate directly to the inner
//! module. All parameter accessors (`parameters()`, `named_parameters()`,
//! etc.) forward through the `OnceLock::get()` path and return an empty
//! list before materialization. CL-445.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::conv::{Conv1d, Conv2d, Conv3d};
use crate::module::Module;
use crate::parameter::Parameter;

// ===========================================================================
// LazyConv1d
// ===========================================================================

/// 1-D convolution layer that defers `in_channels` discovery to the first
/// forward call. Mirrors `torch.nn.LazyConv1d`.
#[derive(Debug)]
pub struct LazyConv1d<T: Float> {
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    bias_enabled: bool,
    inner: OnceLock<Conv1d<T>>,
    training: AtomicBool,
}

impl<T: Float> LazyConv1d<T> {
    /// Build a new `LazyConv1d`. `in_channels` will be discovered from
    /// the first forward input (dim 1 of the `[B, C_in, L]` tensor).
    pub fn new(
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv1d: out_channels must be > 0".into(),
            });
        }
        if kernel_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv1d: kernel_size must be > 0".into(),
            });
        }
        if stride == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv1d: stride must be > 0".into(),
            });
        }
        Ok(Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            bias_enabled: bias,
            inner: OnceLock::new(),
            training: AtomicBool::new(true),
        })
    }

    /// Returns `true` once `in_channels` has been discovered and the
    /// inner [`Conv1d`] has been constructed.
    pub fn is_initialized(&self) -> bool {
        self.inner.get().is_some()
    }

    /// Eagerly materialize the inner Conv1d with the given `in_channels`.
    pub fn materialize(&self, in_channels: usize) -> FerrotorchResult<()> {
        if in_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv1d: in_channels must be > 0".into(),
            });
        }
        if self.inner.get().is_none() {
            let conv = Conv1d::new(
                in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.bias_enabled,
            )?;
            let _ = self.inner.set(conv);
        }
        Ok(())
    }
}

impl<T: Float> Module<T> for LazyConv1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LazyConv1d expects 3-D input [B, C, L], got {:?}",
                    input.shape()
                ),
            });
        }
        if self.inner.get().is_none() {
            let in_channels = input.shape()[1];
            self.materialize(in_channels)?;
        }
        let conv = self.inner.get().expect("initialized after materialize()");
        conv.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.inner.get().map(|c| c.parameters()).unwrap_or_default()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.inner
            .get_mut()
            .map(|c| c.parameters_mut())
            .unwrap_or_default()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.inner
            .get()
            .map(|c| c.named_parameters())
            .unwrap_or_default()
    }

    fn train(&mut self) {
        self.training.store(true, Ordering::Relaxed);
        if let Some(c) = self.inner.get_mut() {
            c.train();
        }
    }

    fn eval(&mut self) {
        self.training.store(false, Ordering::Relaxed);
        if let Some(c) = self.inner.get_mut() {
            c.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }
}

// ===========================================================================
// LazyConv2d
// ===========================================================================

/// 2-D convolution layer that defers `in_channels` discovery to the first
/// forward call. Mirrors `torch.nn.LazyConv2d`.
#[derive(Debug)]
pub struct LazyConv2d<T: Float> {
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    bias_enabled: bool,
    inner: OnceLock<Conv2d<T>>,
    training: AtomicBool,
}

impl<T: Float> LazyConv2d<T> {
    /// Build a new `LazyConv2d`. `in_channels` will be discovered from
    /// the first forward input (dim 1 of the `[B, C_in, H, W]` tensor).
    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv2d: out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv2d: kernel_size must be > 0 in both dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv2d: stride must be > 0 in both dimensions".into(),
            });
        }
        Ok(Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            bias_enabled: bias,
            inner: OnceLock::new(),
            training: AtomicBool::new(true),
        })
    }

    pub fn is_initialized(&self) -> bool {
        self.inner.get().is_some()
    }

    pub fn materialize(&self, in_channels: usize) -> FerrotorchResult<()> {
        if in_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv2d: in_channels must be > 0".into(),
            });
        }
        if self.inner.get().is_none() {
            let conv = Conv2d::new(
                in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.bias_enabled,
            )?;
            let _ = self.inner.set(conv);
        }
        Ok(())
    }
}

impl<T: Float> Module<T> for LazyConv2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LazyConv2d expects 4-D input [B, C, H, W], got {:?}",
                    input.shape()
                ),
            });
        }
        if self.inner.get().is_none() {
            let in_channels = input.shape()[1];
            self.materialize(in_channels)?;
        }
        let conv = self.inner.get().expect("initialized after materialize()");
        conv.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.inner.get().map(|c| c.parameters()).unwrap_or_default()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.inner
            .get_mut()
            .map(|c| c.parameters_mut())
            .unwrap_or_default()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.inner
            .get()
            .map(|c| c.named_parameters())
            .unwrap_or_default()
    }

    fn train(&mut self) {
        self.training.store(true, Ordering::Relaxed);
        if let Some(c) = self.inner.get_mut() {
            c.train();
        }
    }

    fn eval(&mut self) {
        self.training.store(false, Ordering::Relaxed);
        if let Some(c) = self.inner.get_mut() {
            c.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }
}

// ===========================================================================
// LazyConv3d
// ===========================================================================

/// 3-D convolution layer that defers `in_channels` discovery to the first
/// forward call. Mirrors `torch.nn.LazyConv3d`.
#[derive(Debug)]
pub struct LazyConv3d<T: Float> {
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    bias_enabled: bool,
    inner: OnceLock<Conv3d<T>>,
    training: AtomicBool,
}

impl<T: Float> LazyConv3d<T> {
    /// Build a new `LazyConv3d`. `in_channels` will be discovered from
    /// the first forward input (dim 1 of the `[B, C_in, D, H, W]` tensor).
    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv3d: out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 || kernel_size.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv3d: kernel_size must be > 0 in all dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv3d: stride must be > 0 in all dimensions".into(),
            });
        }
        Ok(Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            bias_enabled: bias,
            inner: OnceLock::new(),
            training: AtomicBool::new(true),
        })
    }

    pub fn is_initialized(&self) -> bool {
        self.inner.get().is_some()
    }

    pub fn materialize(&self, in_channels: usize) -> FerrotorchResult<()> {
        if in_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "LazyConv3d: in_channels must be > 0".into(),
            });
        }
        if self.inner.get().is_none() {
            let conv = Conv3d::new(
                in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.bias_enabled,
            )?;
            let _ = self.inner.set(conv);
        }
        Ok(())
    }
}

impl<T: Float> Module<T> for LazyConv3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 5 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LazyConv3d expects 5-D input [B, C, D, H, W], got {:?}",
                    input.shape()
                ),
            });
        }
        if self.inner.get().is_none() {
            let in_channels = input.shape()[1];
            self.materialize(in_channels)?;
        }
        let conv = self.inner.get().expect("initialized after materialize()");
        conv.forward(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.inner.get().map(|c| c.parameters()).unwrap_or_default()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.inner
            .get_mut()
            .map(|c| c.parameters_mut())
            .unwrap_or_default()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.inner
            .get()
            .map(|c| c.named_parameters())
            .unwrap_or_default()
    }

    fn train(&mut self) {
        self.training.store(true, Ordering::Relaxed);
        if let Some(c) = self.inner.get_mut() {
            c.train();
        }
    }

    fn eval(&mut self) {
        self.training.store(false, Ordering::Relaxed);
        if let Some(c) = self.inner.get_mut() {
            c.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::Tensor;
    use ferrotorch_core::storage::TensorStorage;

    fn cpu_tensor(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    // -----------------------------------------------------------------------
    // LazyConv1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_lazy_conv1d_uninitialized_until_first_forward() {
        let lazy: LazyConv1d<f32> = LazyConv1d::new(8, 3, 1, 0, true).unwrap();
        assert!(!lazy.is_initialized());
        assert_eq!(lazy.parameters().len(), 0);
    }

    #[test]
    fn test_lazy_conv1d_materializes_on_first_forward() {
        let lazy: LazyConv1d<f32> = LazyConv1d::new(4, 3, 1, 1, true).unwrap();
        // Input shape: [batch=1, C_in=2, L=5]
        let input = cpu_tensor(&(0..10).map(|i| i as f32).collect::<Vec<_>>(), &[1, 2, 5]);
        let out = lazy.forward(&input).unwrap();
        // [1, 4, 5] with padding=1, stride=1, kernel=3
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 4);
        assert!(lazy.is_initialized());
        // weight + bias
        assert_eq!(lazy.parameters().len(), 2);
    }

    #[test]
    fn test_lazy_conv1d_rejects_wrong_input_ndim() {
        let lazy: LazyConv1d<f32> = LazyConv1d::new(2, 3, 1, 0, true).unwrap();
        let bad = cpu_tensor(&[1.0, 2.0, 3.0], &[3]);
        assert!(lazy.forward(&bad).is_err());
    }

    #[test]
    fn test_lazy_conv1d_explicit_materialize() {
        let lazy: LazyConv1d<f32> = LazyConv1d::new(8, 3, 1, 0, true).unwrap();
        lazy.materialize(16).unwrap();
        assert!(lazy.is_initialized());
        assert_eq!(lazy.parameters().len(), 2);
    }

    #[test]
    fn test_lazy_conv1d_zero_out_channels_errors() {
        assert!(LazyConv1d::<f32>::new(0, 3, 1, 0, true).is_err());
    }

    // -----------------------------------------------------------------------
    // LazyConv2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_lazy_conv2d_uninitialized_until_first_forward() {
        let lazy: LazyConv2d<f32> = LazyConv2d::new(16, (3, 3), (1, 1), (1, 1), true).unwrap();
        assert!(!lazy.is_initialized());
        assert_eq!(lazy.parameters().len(), 0);
    }

    #[test]
    fn test_lazy_conv2d_materializes_on_first_forward() {
        let lazy: LazyConv2d<f32> = LazyConv2d::new(4, (3, 3), (1, 1), (1, 1), true).unwrap();
        // Input: [batch=1, C_in=3, H=4, W=4]
        let data: Vec<f32> = (0..48).map(|i| i as f32 / 10.0).collect();
        let input = cpu_tensor(&data, &[1, 3, 4, 4]);
        let out = lazy.forward(&input).unwrap();
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 4);
        assert_eq!(out.shape()[2], 4); // padding keeps H
        assert_eq!(out.shape()[3], 4);
        assert!(lazy.is_initialized());
        assert_eq!(lazy.parameters().len(), 2);
    }

    #[test]
    fn test_lazy_conv2d_no_bias() {
        let lazy: LazyConv2d<f32> = LazyConv2d::new(2, (3, 3), (1, 1), (1, 1), false).unwrap();
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let input = cpu_tensor(&data, &[1, 3, 4, 4]);
        let _ = lazy.forward(&input).unwrap();
        assert_eq!(lazy.parameters().len(), 1);
    }

    #[test]
    fn test_lazy_conv2d_subsequent_forward_reuses_inner() {
        let lazy: LazyConv2d<f32> = LazyConv2d::new(2, (3, 3), (1, 1), (1, 1), true).unwrap();
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let input1 = cpu_tensor(&data, &[1, 3, 4, 4]);
        let out1 = lazy.forward(&input1).unwrap();

        // Snapshot weight pointer to verify the inner module is not
        // re-initialized on the second call.
        let first_weight_ptr = lazy.parameters()[0].tensor().data().unwrap().as_ptr();

        let input2 = cpu_tensor(&data, &[1, 3, 4, 4]);
        let out2 = lazy.forward(&input2).unwrap();
        let second_weight_ptr = lazy.parameters()[0].tensor().data().unwrap().as_ptr();
        assert_eq!(first_weight_ptr, second_weight_ptr);
        assert_eq!(out1.shape(), out2.shape());
    }

    #[test]
    fn test_lazy_conv2d_rejects_wrong_ndim() {
        let lazy: LazyConv2d<f32> = LazyConv2d::new(2, (3, 3), (1, 1), (1, 1), true).unwrap();
        let bad = cpu_tensor(&[1.0; 9], &[3, 3]);
        assert!(lazy.forward(&bad).is_err());
    }

    #[test]
    fn test_lazy_conv2d_train_eval_propagates_to_inner() {
        let mut lazy: LazyConv2d<f32> = LazyConv2d::new(2, (3, 3), (1, 1), (1, 1), true).unwrap();
        let data: Vec<f32> = (0..48).map(|i| i as f32).collect();
        let input = cpu_tensor(&data, &[1, 3, 4, 4]);
        let _ = lazy.forward(&input).unwrap();
        lazy.eval();
        assert!(!lazy.is_training());
        lazy.train();
        assert!(lazy.is_training());
    }

    // -----------------------------------------------------------------------
    // LazyConv3d
    // -----------------------------------------------------------------------

    #[test]
    fn test_lazy_conv3d_uninitialized_until_first_forward() {
        let lazy: LazyConv3d<f32> =
            LazyConv3d::new(4, (3, 3, 3), (1, 1, 1), (1, 1, 1), true).unwrap();
        assert!(!lazy.is_initialized());
    }

    #[test]
    fn test_lazy_conv3d_materializes_on_first_forward() {
        let lazy: LazyConv3d<f32> =
            LazyConv3d::new(2, (3, 3, 3), (1, 1, 1), (1, 1, 1), true).unwrap();
        // Input: [batch=1, C_in=2, D=4, H=4, W=4]
        let data: Vec<f32> = (0..128).map(|i| i as f32 / 10.0).collect();
        let input = cpu_tensor(&data, &[1, 2, 4, 4, 4]);
        let out = lazy.forward(&input).unwrap();
        assert_eq!(out.shape()[0], 1);
        assert_eq!(out.shape()[1], 2);
        assert!(lazy.is_initialized());
    }

    #[test]
    fn test_lazy_conv3d_rejects_wrong_ndim() {
        let lazy: LazyConv3d<f32> =
            LazyConv3d::new(2, (3, 3, 3), (1, 1, 1), (1, 1, 1), true).unwrap();
        let bad = cpu_tensor(&[0.0; 48], &[1, 3, 4, 4]);
        assert!(lazy.forward(&bad).is_err());
    }

    #[test]
    fn test_lazy_conv3d_zero_kernel_errors() {
        assert!(LazyConv3d::<f32>::new(2, (3, 0, 3), (1, 1, 1), (1, 1, 1), true).is_err());
    }
}
