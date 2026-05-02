//! Lazy variants of [`ConvTranspose{1,2,3}d`]. (#622)
//!
//! `in_channels` is discovered from the input's channel dim on the first
//! forward call; everything else is provided up front.

use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::conv::{ConvTranspose1d, ConvTranspose2d, ConvTranspose3d};
use crate::module::Module;
use crate::parameter::Parameter;

fn channels_from_input<T: Float>(
    input: &Tensor<T>,
    op: &str,
    expected_ndim: usize,
) -> FerrotorchResult<usize> {
    if input.ndim() != expected_ndim {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "{op}: expected {expected_ndim}-D input [N, C, ...], got {}-D",
                input.ndim()
            ),
        });
    }
    Ok(input.shape()[1])
}

/// Lazy 1-D transposed convolution. `in_channels` discovered at first forward.
#[derive(Debug)]
pub struct LazyConvTranspose1d<T: Float> {
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
    bias_enabled: bool,
    inner: OnceLock<ConvTranspose1d<T>>,
    training: AtomicBool,
}

impl<T: Float> LazyConvTranspose1d<T> {
    pub fn new(
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        bias: bool,
    ) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias_enabled: bias,
            inner: OnceLock::new(),
            training: AtomicBool::new(true),
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.inner.get().is_some()
    }

    pub fn materialize(&self, in_channels: usize) -> FerrotorchResult<()> {
        if self.inner.get().is_none() {
            let inner = ConvTranspose1d::<T>::new(
                in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_padding,
                self.bias_enabled,
            )?;
            let _ = self.inner.set(inner);
        }
        Ok(())
    }
}

impl<T: Float> Module<T> for LazyConvTranspose1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if self.inner.get().is_none() {
            let c = channels_from_input(input, "LazyConvTranspose1d", 3)?;
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

/// Lazy 2-D transposed convolution.
#[derive(Debug)]
pub struct LazyConvTranspose2d<T: Float> {
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    output_padding: (usize, usize),
    bias_enabled: bool,
    inner: OnceLock<ConvTranspose2d<T>>,
    training: AtomicBool,
}

impl<T: Float> LazyConvTranspose2d<T> {
    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        bias: bool,
    ) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias_enabled: bias,
            inner: OnceLock::new(),
            training: AtomicBool::new(true),
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.inner.get().is_some()
    }

    pub fn materialize(&self, in_channels: usize) -> FerrotorchResult<()> {
        if self.inner.get().is_none() {
            let inner = ConvTranspose2d::<T>::new(
                in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_padding,
                self.bias_enabled,
            )?;
            let _ = self.inner.set(inner);
        }
        Ok(())
    }
}

impl<T: Float> Module<T> for LazyConvTranspose2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if self.inner.get().is_none() {
            let c = channels_from_input(input, "LazyConvTranspose2d", 4)?;
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

/// Lazy 3-D transposed convolution.
#[derive(Debug)]
pub struct LazyConvTranspose3d<T: Float> {
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    output_padding: (usize, usize, usize),
    bias_enabled: bool,
    inner: OnceLock<ConvTranspose3d<T>>,
    training: AtomicBool,
}

impl<T: Float> LazyConvTranspose3d<T> {
    pub fn new(
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        bias: bool,
    ) -> Self {
        Self {
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            bias_enabled: bias,
            inner: OnceLock::new(),
            training: AtomicBool::new(true),
        }
    }

    pub fn is_initialized(&self) -> bool {
        self.inner.get().is_some()
    }

    pub fn materialize(&self, in_channels: usize) -> FerrotorchResult<()> {
        if self.inner.get().is_none() {
            let inner = ConvTranspose3d::<T>::new(
                in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
                self.padding,
                self.output_padding,
                self.bias_enabled,
            )?;
            let _ = self.inner.set(inner);
        }
        Ok(())
    }
}

impl<T: Float> Module<T> for LazyConvTranspose3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if self.inner.get().is_none() {
            let c = channels_from_input(input, "LazyConvTranspose3d", 5)?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::storage::TensorStorage;

    fn cpu_tensor(data: Vec<f32>, shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
    }

    #[test]
    fn lazy_conv_transpose2d_explicit_materialize() {
        let m: LazyConvTranspose2d<f32> =
            LazyConvTranspose2d::new(8, (3, 3), (1, 1), (1, 1), (0, 0), true);
        assert!(!m.is_initialized());
        m.materialize(4).unwrap();
        assert!(m.is_initialized());
        // weight + bias = 2 params
        assert_eq!(m.parameters().len(), 2);
    }

    #[test]
    fn lazy_conv_transpose1d_rejects_wrong_rank() {
        let m: LazyConvTranspose1d<f32> = LazyConvTranspose1d::new(4, 3, 1, 0, 0, true);
        let input = cpu_tensor(vec![1.0, 2.0], &[2]);
        let err = m.forward(&input).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn lazy_conv_transpose3d_explicit_materialize() {
        let m: LazyConvTranspose3d<f32> =
            LazyConvTranspose3d::new(2, (2, 2, 2), (1, 1, 1), (0, 0, 0), (0, 0, 0), false);
        m.materialize(3).unwrap();
        assert!(m.is_initialized());
    }

    #[test]
    fn lazy_conv_transpose_train_eval_toggle() {
        let mut m: LazyConvTranspose2d<f32> =
            LazyConvTranspose2d::new(4, (3, 3), (1, 1), (0, 0), (0, 0), true);
        assert!(m.is_training());
        m.eval();
        assert!(!m.is_training());
        m.train();
        assert!(m.is_training());
    }
}
