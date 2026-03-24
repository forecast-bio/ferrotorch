//! Pooling layers: MaxPool1d/2d/3d, AvgPool1d/2d/3d, AdaptiveAvgPool1d/2d/3d,
//! AdaptiveMaxPool2d, MaxUnpool2d — CL-315.
//!
//! All are zero-parameter modules operating on `[B, C, *spatial]` tensors.
//! Each forward pass attaches a `GradFn<T>` for reverse-mode autodiff
//! when gradient tracking is enabled.

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::module::Module;
use crate::parameter::Parameter;

// ===========================================================================
// Helpers
// ===========================================================================

/// Compute the output spatial dimension for a standard pooling operation.
///
/// `out = (input + 2 * padding - kernel_size) / stride + 1`
#[inline]
fn pool_output_size(input: usize, kernel_size: usize, stride: usize, padding: usize) -> usize {
    (input + 2 * padding - kernel_size) / stride + 1
}

/// Validate that the input tensor has shape `[B, C, H, W]`.
fn validate_4d<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(usize, usize, usize, usize)> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "pooling expects 4D input [B, C, H, W], got shape {:?}",
                shape
            ),
        });
    }
    Ok((shape[0], shape[1], shape[2], shape[3]))
}

/// Validate pooling parameters and compute output spatial dimensions.
fn validate_pool_params(
    h: usize,
    w: usize,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> FerrotorchResult<(usize, usize)> {
    if kernel_size[0] == 0 || kernel_size[1] == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "kernel_size must be > 0".into(),
        });
    }
    if stride[0] == 0 || stride[1] == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "stride must be > 0".into(),
        });
    }
    let padded_h = h + 2 * padding[0];
    let padded_w = w + 2 * padding[1];
    if padded_h < kernel_size[0] || padded_w < kernel_size[1] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "padded input ({padded_h}, {padded_w}) smaller than kernel ({}, {})",
                kernel_size[0], kernel_size[1]
            ),
        });
    }
    let out_h = pool_output_size(h, kernel_size[0], stride[0], padding[0]);
    let out_w = pool_output_size(w, kernel_size[1], stride[1], padding[1]);
    Ok((out_h, out_w))
}

// ===========================================================================
// MaxPool2d
// ===========================================================================

/// 2D max pooling layer.
///
/// Slides a kernel window over each `[H, W]` spatial plane, taking the
/// maximum value in each window. Zero parameters.
///
/// Input shape: `[B, C, H, W]`
/// Output shape: `[B, C, H_out, W_out]`
#[derive(Debug, Clone)]
pub struct MaxPool2d {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}

impl MaxPool2d {
    /// Create a new `MaxPool2d` layer.
    ///
    /// `stride` defaults to `kernel_size` when set to `[0, 0]` (PyTorch convention).
    pub fn new(kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> Self {
        let stride = if stride == [0, 0] {
            kernel_size
        } else {
            stride
        };
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<T: Float> Module<T> for MaxPool2d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        max_pool2d_forward(input, self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Forward computation for max pooling, returns the output tensor with
/// gradient tracking when enabled.
fn max_pool2d_forward<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, h, w) = validate_4d(input)?;
    let (out_h, out_w) = validate_pool_params(h, w, kernel_size, stride, padding)?;

    // Save device for restoring on output.
    let input_device = input.device();

    let data = input.data_vec()?;
    let total = batch * channels * out_h * out_w;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];
    // Store the flat index of the max element within the input for each output element.
    let mut indices = vec![0usize; total];

    let neg_inf = T::from(-1e38).unwrap();

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    let mut max_val = neg_inf;
                    let mut max_idx = 0usize;

                    for kh in 0..kernel_size[0] {
                        for kw in 0..kernel_size[1] {
                            let ih = oh * stride[0] + kh;
                            let iw = ow * stride[1] + kw;

                            // Account for padding.
                            let ih = ih as isize - padding[0] as isize;
                            let iw = iw as isize - padding[1] as isize;

                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let ih = ih as usize;
                                let iw = iw as usize;
                                let in_idx = ((b * channels + c) * h + ih) * w + iw;
                                let val = data[in_idx];
                                if val > max_val {
                                    max_val = val;
                                    max_idx = in_idx;
                                }
                            }
                            // Padded positions have -inf, so they never win.
                        }
                    }

                    output[out_idx] = max_val;
                    indices[out_idx] = max_idx;
                }
            }
        }
    }

    let out_shape = vec![batch, channels, out_h, out_w];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(MaxPool2dBackward {
                input: input.clone(),
                indices,
            }),
        )?
        .to(input_device) // restore device
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device) // restore device
    }
}

/// Backward for `MaxPool2d`.
///
/// Routes the upstream gradient to the position of the max element in each
/// pooling window. All other positions receive zero gradient.
#[derive(Debug)]
struct MaxPool2dBackward<T: Float> {
    input: Tensor<T>,
    /// For each output element, the flat index into the input where the max lives.
    indices: Vec<usize>,
}

impl<T: Float> GradFn<T> for MaxPool2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let input_numel = self.input.numel();
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_numel];

        for (out_idx, &in_idx) in self.indices.iter().enumerate() {
            grad_input[in_idx] += go_data[out_idx];
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MaxPool2dBackward"
    }
}

// ===========================================================================
// AvgPool2d
// ===========================================================================

/// 2D average pooling layer.
///
/// Slides a kernel window over each `[H, W]` spatial plane, computing
/// the arithmetic mean of each window. Zero parameters.
///
/// Input shape: `[B, C, H, W]`
/// Output shape: `[B, C, H_out, W_out]`
#[derive(Debug, Clone)]
pub struct AvgPool2d {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}

impl AvgPool2d {
    /// Create a new `AvgPool2d` layer.
    ///
    /// `stride` defaults to `kernel_size` when set to `[0, 0]`.
    pub fn new(kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> Self {
        let stride = if stride == [0, 0] {
            kernel_size
        } else {
            stride
        };
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<T: Float> Module<T> for AvgPool2d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        avg_pool2d_forward(input, self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Forward computation for average pooling.
fn avg_pool2d_forward<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, h, w) = validate_4d(input)?;
    let (out_h, out_w) = validate_pool_params(h, w, kernel_size, stride, padding)?;

    // Save device for restoring on output.
    let input_device = input.device();

    let data = input.data_vec()?;
    let total = batch * channels * out_h * out_w;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];

    let kernel_area = T::from(kernel_size[0] * kernel_size[1]).unwrap();

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                for ow in 0..out_w {
                    let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    let mut sum = <T as num_traits::Zero>::zero();

                    for kh in 0..kernel_size[0] {
                        for kw in 0..kernel_size[1] {
                            let ih = oh * stride[0] + kh;
                            let iw = ow * stride[1] + kw;
                            let ih = ih as isize - padding[0] as isize;
                            let iw = iw as isize - padding[1] as isize;

                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let ih = ih as usize;
                                let iw = iw as usize;
                                let in_idx = ((b * channels + c) * h + ih) * w + iw;
                                sum += data[in_idx];
                            }
                            // Padded positions contribute 0, but we still divide
                            // by the full kernel area (count_include_pad = true).
                        }
                    }

                    output[out_idx] = sum / kernel_area;
                }
            }
        }
    }

    let out_shape = vec![batch, channels, out_h, out_w];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(AvgPool2dBackward {
                input: input.clone(),
                kernel_size,
                stride,
                padding,
            }),
        )?
        .to(input_device) // restore device
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device) // restore device
    }
}

/// Backward for `AvgPool2d`.
///
/// Distributes the upstream gradient evenly to all input positions that
/// contributed to each output window, dividing by the kernel area.
#[derive(Debug)]
struct AvgPool2dBackward<T: Float> {
    input: Tensor<T>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
}

impl<T: Float> GradFn<T> for AvgPool2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let in_shape = self.input.shape();
        let (batch, channels, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let out_h = pool_output_size(h, self.kernel_size[0], self.stride[0], self.padding[0]);
        let out_w = pool_output_size(w, self.kernel_size[1], self.stride[1], self.padding[1]);

        let mut grad_input = vec![<T as num_traits::Zero>::zero(); batch * channels * h * w];
        let kernel_area = T::from(self.kernel_size[0] * self.kernel_size[1]).unwrap();

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                        let grad_val = go_data[out_idx] / kernel_area;

                        for kh in 0..self.kernel_size[0] {
                            for kw in 0..self.kernel_size[1] {
                                let ih =
                                    (oh * self.stride[0] + kh) as isize - self.padding[0] as isize;
                                let iw =
                                    (ow * self.stride[1] + kw) as isize - self.padding[1] as isize;

                                if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                    let ih = ih as usize;
                                    let iw = iw as usize;
                                    let in_idx = ((b * channels + c) * h + ih) * w + iw;
                                    grad_input[in_idx] += grad_val;
                                }
                            }
                        }
                    }
                }
            }
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AvgPool2dBackward"
    }
}

// ===========================================================================
// AdaptiveAvgPool2d
// ===========================================================================

/// 2D adaptive average pooling layer.
///
/// Dynamically computes kernel size and stride to produce the target
/// `output_size` regardless of input spatial dimensions. Zero parameters.
///
/// Input shape: `[B, C, H, W]`
/// Output shape: `[B, C, output_size.0, output_size.1]`
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveAvgPool2d {
    /// Create a new `AdaptiveAvgPool2d` targeting the given output spatial size.
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

impl<T: Float> Module<T> for AdaptiveAvgPool2d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        adaptive_avg_pool2d_forward(input, self.output_size)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Compute the start index for adaptive pooling window.
///
/// Uses the same formula as PyTorch: `start_i = floor(i * input_size / output_size)`.
#[inline]
fn adaptive_start(idx: usize, input_size: usize, output_size: usize) -> usize {
    (idx * input_size) / output_size
}

/// Compute the end index for adaptive pooling window.
///
/// `end_i = ceil((i + 1) * input_size / output_size)`
#[inline]
fn adaptive_end(idx: usize, input_size: usize, output_size: usize) -> usize {
    ((idx + 1) * input_size).div_ceil(output_size)
}

/// Forward computation for adaptive average pooling.
fn adaptive_avg_pool2d_forward<T: Float>(
    input: &Tensor<T>,
    output_size: (usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, h, w) = validate_4d(input)?;
    let (out_h, out_w) = output_size;

    if out_h == 0 || out_w == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "adaptive output_size must be > 0".into(),
        });
    }

    // Save device for restoring on output.
    let input_device = input.device();

    let data = input.data_vec()?;
    let total = batch * channels * out_h * out_w;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                let h_start = adaptive_start(oh, h, out_h);
                let h_end = adaptive_end(oh, h, out_h);

                for ow in 0..out_w {
                    let w_start = adaptive_start(ow, w, out_w);
                    let w_end = adaptive_end(ow, w, out_w);

                    let window_area = (h_end - h_start) * (w_end - w_start);
                    let mut sum = <T as num_traits::Zero>::zero();

                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let in_idx = ((b * channels + c) * h + ih) * w + iw;
                            sum += data[in_idx];
                        }
                    }

                    let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    output[out_idx] = sum / T::from(window_area).unwrap();
                }
            }
        }
    }

    let out_shape = vec![batch, channels, out_h, out_w];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(AdaptiveAvgPool2dBackward {
                input: input.clone(),
                output_size,
            }),
        )?
        .to(input_device) // restore device
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device) // restore device
    }
}

/// Backward for `AdaptiveAvgPool2d`.
///
/// For each output element, distributes the upstream gradient evenly across
/// the input positions in its adaptive window.
#[derive(Debug)]
struct AdaptiveAvgPool2dBackward<T: Float> {
    input: Tensor<T>,
    output_size: (usize, usize),
}

impl<T: Float> GradFn<T> for AdaptiveAvgPool2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let in_shape = self.input.shape();
        let (batch, channels, h, w) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let (out_h, out_w) = self.output_size;

        let mut grad_input = vec![<T as num_traits::Zero>::zero(); batch * channels * h * w];

        for b in 0..batch {
            for c in 0..channels {
                for oh in 0..out_h {
                    let h_start = adaptive_start(oh, h, out_h);
                    let h_end = adaptive_end(oh, h, out_h);

                    for ow in 0..out_w {
                        let w_start = adaptive_start(ow, w, out_w);
                        let w_end = adaptive_end(ow, w, out_w);

                        let window_area = (h_end - h_start) * (w_end - w_start);
                        let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                        let grad_val = go_data[out_idx] / T::from(window_area).unwrap();

                        for ih in h_start..h_end {
                            for iw in w_start..w_end {
                                let in_idx = ((b * channels + c) * h + ih) * w + iw;
                                grad_input[in_idx] += grad_val;
                            }
                        }
                    }
                }
            }
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AdaptiveAvgPool2dBackward"
    }
}

// ===========================================================================
// 1-D helpers — CL-315
// ===========================================================================

/// Validate that the input tensor has shape `[B, C, L]`.
fn validate_3d<T: Float>(input: &Tensor<T>) -> FerrotorchResult<(usize, usize, usize)> {
    let shape = input.shape();
    if shape.len() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "1D pooling expects 3D input [B, C, L], got shape {:?}",
                shape
            ),
        });
    }
    Ok((shape[0], shape[1], shape[2]))
}

/// Validate 1D pooling parameters and compute output length.
fn validate_pool_params_1d(
    l: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> FerrotorchResult<usize> {
    if kernel_size == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "kernel_size must be > 0".into(),
        });
    }
    if stride == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "stride must be > 0".into(),
        });
    }
    let padded = l + 2 * padding;
    if padded < kernel_size {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("padded input ({padded}) smaller than kernel ({kernel_size})"),
        });
    }
    Ok(pool_output_size(l, kernel_size, stride, padding))
}

// ===========================================================================
// 3-D helpers — CL-315
// ===========================================================================

/// Validate that the input tensor has shape `[B, C, D, H, W]`.
fn validate_5d<T: Float>(
    input: &Tensor<T>,
) -> FerrotorchResult<(usize, usize, usize, usize, usize)> {
    let shape = input.shape();
    if shape.len() != 5 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "3D pooling expects 5D input [B, C, D, H, W], got shape {:?}",
                shape
            ),
        });
    }
    Ok((shape[0], shape[1], shape[2], shape[3], shape[4]))
}

/// Validate 3D pooling parameters and compute output spatial dimensions.
fn validate_pool_params_3d(
    d: usize,
    h: usize,
    w: usize,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
) -> FerrotorchResult<(usize, usize, usize)> {
    for i in 0..3 {
        if kernel_size[i] == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0".into(),
            });
        }
        if stride[i] == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0".into(),
            });
        }
    }
    let sizes = [d, h, w];
    for i in 0..3 {
        let padded = sizes[i] + 2 * padding[i];
        if padded < kernel_size[i] {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "padded input dim {i} ({padded}) smaller than kernel ({})",
                    kernel_size[i]
                ),
            });
        }
    }
    let out_d = pool_output_size(d, kernel_size[0], stride[0], padding[0]);
    let out_h = pool_output_size(h, kernel_size[1], stride[1], padding[1]);
    let out_w = pool_output_size(w, kernel_size[2], stride[2], padding[2]);
    Ok((out_d, out_h, out_w))
}

// ===========================================================================
// MaxPool1d — CL-315
// ===========================================================================

/// 1D max pooling layer.
///
/// Slides a kernel window over each `[L]` spatial dimension, taking the
/// maximum value in each window. Zero parameters.
///
/// Input shape: `[B, C, L]`
/// Output shape: `[B, C, L_out]`
#[derive(Debug, Clone)]
pub struct MaxPool1d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl MaxPool1d {
    /// Create a new `MaxPool1d` layer.
    ///
    /// `stride` defaults to `kernel_size` when set to `0` (PyTorch convention).
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        let stride = if stride == 0 { kernel_size } else { stride };
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<T: Float> Module<T> for MaxPool1d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        max_pool1d_forward(input, self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Forward computation for 1D max pooling.
fn max_pool1d_forward<T: Float>(
    input: &Tensor<T>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, l) = validate_3d(input)?;
    let out_l = validate_pool_params_1d(l, kernel_size, stride, padding)?;

    let input_device = input.device();
    let data = input.data_vec()?;
    let total = batch * channels * out_l;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];
    let mut indices = vec![0usize; total];
    let neg_inf = T::from(-1e38).unwrap();

    for b in 0..batch {
        for c in 0..channels {
            for ol in 0..out_l {
                let out_idx = (b * channels + c) * out_l + ol;
                let mut max_val = neg_inf;
                let mut max_idx = 0usize;

                for k in 0..kernel_size {
                    let il = ol * stride + k;
                    let il = il as isize - padding as isize;
                    if il >= 0 && il < l as isize {
                        let il = il as usize;
                        let in_idx = (b * channels + c) * l + il;
                        let val = data[in_idx];
                        if val > max_val {
                            max_val = val;
                            max_idx = in_idx;
                        }
                    }
                }

                output[out_idx] = max_val;
                indices[out_idx] = max_idx;
            }
        }
    }

    let out_shape = vec![batch, channels, out_l];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(MaxPool1dBackward {
                input: input.clone(),
                indices,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Backward for `MaxPool1d`.
#[derive(Debug)]
struct MaxPool1dBackward<T: Float> {
    input: Tensor<T>,
    indices: Vec<usize>,
}

impl<T: Float> GradFn<T> for MaxPool1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let input_numel = self.input.numel();
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_numel];

        for (out_idx, &in_idx) in self.indices.iter().enumerate() {
            grad_input[in_idx] += go_data[out_idx];
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MaxPool1dBackward"
    }
}

// ===========================================================================
// MaxPool3d — CL-315
// ===========================================================================

/// 3D max pooling layer.
///
/// Slides a kernel window over each `[D, H, W]` spatial volume, taking the
/// maximum value in each window. Zero parameters.
///
/// Input shape: `[B, C, D, H, W]`
/// Output shape: `[B, C, D_out, H_out, W_out]`
#[derive(Debug, Clone)]
pub struct MaxPool3d {
    pub kernel_size: [usize; 3],
    pub stride: [usize; 3],
    pub padding: [usize; 3],
}

impl MaxPool3d {
    /// Create a new `MaxPool3d` layer.
    ///
    /// `stride` defaults to `kernel_size` when set to `[0, 0, 0]`.
    pub fn new(kernel_size: [usize; 3], stride: [usize; 3], padding: [usize; 3]) -> Self {
        let stride = if stride == [0, 0, 0] {
            kernel_size
        } else {
            stride
        };
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<T: Float> Module<T> for MaxPool3d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        max_pool3d_forward(input, self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Forward computation for 3D max pooling.
fn max_pool3d_forward<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, d, h, w) = validate_5d(input)?;
    let (out_d, out_h, out_w) = validate_pool_params_3d(d, h, w, kernel_size, stride, padding)?;

    let input_device = input.device();
    let data = input.data_vec()?;
    let total = batch * channels * out_d * out_h * out_w;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];
    let mut indices = vec![0usize; total];
    let neg_inf = T::from(-1e38).unwrap();

    for b in 0..batch {
        for c in 0..channels {
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let out_idx = (((b * channels + c) * out_d + od) * out_h + oh) * out_w + ow;
                        let mut max_val = neg_inf;
                        let mut max_idx = 0usize;

                        for kd in 0..kernel_size[0] {
                            let id = (od * stride[0] + kd) as isize - padding[0] as isize;
                            if id < 0 || id >= d as isize {
                                continue;
                            }
                            let id = id as usize;
                            for kh in 0..kernel_size[1] {
                                let ih = (oh * stride[1] + kh) as isize - padding[1] as isize;
                                if ih < 0 || ih >= h as isize {
                                    continue;
                                }
                                let ih = ih as usize;
                                for kw in 0..kernel_size[2] {
                                    let iw = (ow * stride[2] + kw) as isize - padding[2] as isize;
                                    if iw < 0 || iw >= w as isize {
                                        continue;
                                    }
                                    let iw = iw as usize;
                                    let in_idx = (((b * channels + c) * d + id) * h + ih) * w + iw;
                                    let val = data[in_idx];
                                    if val > max_val {
                                        max_val = val;
                                        max_idx = in_idx;
                                    }
                                }
                            }
                        }

                        output[out_idx] = max_val;
                        indices[out_idx] = max_idx;
                    }
                }
            }
        }
    }

    let out_shape = vec![batch, channels, out_d, out_h, out_w];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(MaxPool3dBackward {
                input: input.clone(),
                indices,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Backward for `MaxPool3d`.
#[derive(Debug)]
struct MaxPool3dBackward<T: Float> {
    input: Tensor<T>,
    indices: Vec<usize>,
}

impl<T: Float> GradFn<T> for MaxPool3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let input_numel = self.input.numel();
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_numel];

        for (out_idx, &in_idx) in self.indices.iter().enumerate() {
            grad_input[in_idx] += go_data[out_idx];
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MaxPool3dBackward"
    }
}

// ===========================================================================
// AvgPool1d — CL-315
// ===========================================================================

/// 1D average pooling layer.
///
/// Input shape: `[B, C, L]`
/// Output shape: `[B, C, L_out]`
#[derive(Debug, Clone)]
pub struct AvgPool1d {
    pub kernel_size: usize,
    pub stride: usize,
    pub padding: usize,
}

impl AvgPool1d {
    /// Create a new `AvgPool1d` layer.
    ///
    /// `stride` defaults to `kernel_size` when set to `0`.
    pub fn new(kernel_size: usize, stride: usize, padding: usize) -> Self {
        let stride = if stride == 0 { kernel_size } else { stride };
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<T: Float> Module<T> for AvgPool1d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        avg_pool1d_forward(input, self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Forward computation for 1D average pooling.
fn avg_pool1d_forward<T: Float>(
    input: &Tensor<T>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, l) = validate_3d(input)?;
    let out_l = validate_pool_params_1d(l, kernel_size, stride, padding)?;

    let input_device = input.device();
    let data = input.data_vec()?;
    let total = batch * channels * out_l;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];
    let kernel_area = T::from(kernel_size).unwrap();

    for b in 0..batch {
        for c in 0..channels {
            for ol in 0..out_l {
                let out_idx = (b * channels + c) * out_l + ol;
                let mut sum = <T as num_traits::Zero>::zero();

                for k in 0..kernel_size {
                    let il = (ol * stride + k) as isize - padding as isize;
                    if il >= 0 && il < l as isize {
                        let il = il as usize;
                        let in_idx = (b * channels + c) * l + il;
                        sum += data[in_idx];
                    }
                }

                output[out_idx] = sum / kernel_area;
            }
        }
    }

    let out_shape = vec![batch, channels, out_l];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(AvgPool1dBackward {
                input: input.clone(),
                kernel_size,
                stride,
                padding,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Backward for `AvgPool1d`.
#[derive(Debug)]
struct AvgPool1dBackward<T: Float> {
    input: Tensor<T>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
}

impl<T: Float> GradFn<T> for AvgPool1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let in_shape = self.input.shape();
        let (batch, channels, l) = (in_shape[0], in_shape[1], in_shape[2]);
        let out_l = pool_output_size(l, self.kernel_size, self.stride, self.padding);

        let mut grad_input = vec![<T as num_traits::Zero>::zero(); batch * channels * l];
        let kernel_area = T::from(self.kernel_size).unwrap();

        for b in 0..batch {
            for c in 0..channels {
                for ol in 0..out_l {
                    let out_idx = (b * channels + c) * out_l + ol;
                    let grad_val = go_data[out_idx] / kernel_area;

                    for k in 0..self.kernel_size {
                        let il = (ol * self.stride + k) as isize - self.padding as isize;
                        if il >= 0 && il < l as isize {
                            let il = il as usize;
                            let in_idx = (b * channels + c) * l + il;
                            grad_input[in_idx] += grad_val;
                        }
                    }
                }
            }
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AvgPool1dBackward"
    }
}

// ===========================================================================
// AvgPool3d — CL-315
// ===========================================================================

/// 3D average pooling layer.
///
/// Input shape: `[B, C, D, H, W]`
/// Output shape: `[B, C, D_out, H_out, W_out]`
#[derive(Debug, Clone)]
pub struct AvgPool3d {
    pub kernel_size: [usize; 3],
    pub stride: [usize; 3],
    pub padding: [usize; 3],
}

impl AvgPool3d {
    /// Create a new `AvgPool3d` layer.
    ///
    /// `stride` defaults to `kernel_size` when set to `[0, 0, 0]`.
    pub fn new(kernel_size: [usize; 3], stride: [usize; 3], padding: [usize; 3]) -> Self {
        let stride = if stride == [0, 0, 0] {
            kernel_size
        } else {
            stride
        };
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl<T: Float> Module<T> for AvgPool3d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        avg_pool3d_forward(input, self.kernel_size, self.stride, self.padding)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Forward computation for 3D average pooling.
fn avg_pool3d_forward<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, d, h, w) = validate_5d(input)?;
    let (out_d, out_h, out_w) = validate_pool_params_3d(d, h, w, kernel_size, stride, padding)?;

    let input_device = input.device();
    let data = input.data_vec()?;
    let total = batch * channels * out_d * out_h * out_w;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];
    let kernel_vol = T::from(kernel_size[0] * kernel_size[1] * kernel_size[2]).unwrap();

    for b in 0..batch {
        for c in 0..channels {
            for od in 0..out_d {
                for oh in 0..out_h {
                    for ow in 0..out_w {
                        let out_idx = (((b * channels + c) * out_d + od) * out_h + oh) * out_w + ow;
                        let mut sum = <T as num_traits::Zero>::zero();

                        for kd in 0..kernel_size[0] {
                            let id = (od * stride[0] + kd) as isize - padding[0] as isize;
                            if id < 0 || id >= d as isize {
                                continue;
                            }
                            let id = id as usize;
                            for kh in 0..kernel_size[1] {
                                let ih = (oh * stride[1] + kh) as isize - padding[1] as isize;
                                if ih < 0 || ih >= h as isize {
                                    continue;
                                }
                                let ih = ih as usize;
                                for kw in 0..kernel_size[2] {
                                    let iw = (ow * stride[2] + kw) as isize - padding[2] as isize;
                                    if iw < 0 || iw >= w as isize {
                                        continue;
                                    }
                                    let iw = iw as usize;
                                    let in_idx = (((b * channels + c) * d + id) * h + ih) * w + iw;
                                    sum += data[in_idx];
                                }
                            }
                        }

                        output[out_idx] = sum / kernel_vol;
                    }
                }
            }
        }
    }

    let out_shape = vec![batch, channels, out_d, out_h, out_w];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(AvgPool3dBackward {
                input: input.clone(),
                kernel_size,
                stride,
                padding,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Backward for `AvgPool3d`.
#[derive(Debug)]
struct AvgPool3dBackward<T: Float> {
    input: Tensor<T>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
}

impl<T: Float> GradFn<T> for AvgPool3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let in_shape = self.input.shape();
        let (batch, channels, d, h, w) = (
            in_shape[0],
            in_shape[1],
            in_shape[2],
            in_shape[3],
            in_shape[4],
        );
        let out_d = pool_output_size(d, self.kernel_size[0], self.stride[0], self.padding[0]);
        let out_h = pool_output_size(h, self.kernel_size[1], self.stride[1], self.padding[1]);
        let out_w = pool_output_size(w, self.kernel_size[2], self.stride[2], self.padding[2]);

        let mut grad_input = vec![<T as num_traits::Zero>::zero(); batch * channels * d * h * w];
        let kernel_vol =
            T::from(self.kernel_size[0] * self.kernel_size[1] * self.kernel_size[2]).unwrap();

        for b in 0..batch {
            for c in 0..channels {
                for od in 0..out_d {
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let out_idx =
                                (((b * channels + c) * out_d + od) * out_h + oh) * out_w + ow;
                            let grad_val = go_data[out_idx] / kernel_vol;

                            for kd in 0..self.kernel_size[0] {
                                let id =
                                    (od * self.stride[0] + kd) as isize - self.padding[0] as isize;
                                if id < 0 || id >= d as isize {
                                    continue;
                                }
                                let id = id as usize;
                                for kh in 0..self.kernel_size[1] {
                                    let ih = (oh * self.stride[1] + kh) as isize
                                        - self.padding[1] as isize;
                                    if ih < 0 || ih >= h as isize {
                                        continue;
                                    }
                                    let ih = ih as usize;
                                    for kw in 0..self.kernel_size[2] {
                                        let iw = (ow * self.stride[2] + kw) as isize
                                            - self.padding[2] as isize;
                                        if iw < 0 || iw >= w as isize {
                                            continue;
                                        }
                                        let iw = iw as usize;
                                        let in_idx =
                                            (((b * channels + c) * d + id) * h + ih) * w + iw;
                                        grad_input[in_idx] += grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AvgPool3dBackward"
    }
}

// ===========================================================================
// AdaptiveMaxPool2d — CL-315
// ===========================================================================

/// 2D adaptive max pooling layer.
///
/// Dynamically computes window boundaries to produce the target `output_size`
/// regardless of input spatial dimensions. Returns the pooled tensor and
/// stores indices internally for gradient routing.
///
/// Input shape: `[B, C, H, W]`
/// Output shape: `[B, C, output_size.0, output_size.1]`
#[derive(Debug, Clone)]
pub struct AdaptiveMaxPool2d {
    pub output_size: (usize, usize),
}

impl AdaptiveMaxPool2d {
    /// Create a new `AdaptiveMaxPool2d` targeting the given output spatial size.
    pub fn new(output_size: (usize, usize)) -> Self {
        Self { output_size }
    }
}

impl<T: Float> Module<T> for AdaptiveMaxPool2d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        adaptive_max_pool2d_forward(input, self.output_size)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Forward computation for adaptive max pooling.
fn adaptive_max_pool2d_forward<T: Float>(
    input: &Tensor<T>,
    output_size: (usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, h, w) = validate_4d(input)?;
    let (out_h, out_w) = output_size;

    if out_h == 0 || out_w == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "adaptive output_size must be > 0".into(),
        });
    }

    let input_device = input.device();
    let data = input.data_vec()?;
    let total = batch * channels * out_h * out_w;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];
    let mut indices = vec![0usize; total];
    let neg_inf = T::from(-1e38).unwrap();

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..out_h {
                let h_start = adaptive_start(oh, h, out_h);
                let h_end = adaptive_end(oh, h, out_h);

                for ow in 0..out_w {
                    let w_start = adaptive_start(ow, w, out_w);
                    let w_end = adaptive_end(ow, w, out_w);

                    let out_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                    let mut max_val = neg_inf;
                    let mut max_idx = 0usize;

                    for ih in h_start..h_end {
                        for iw in w_start..w_end {
                            let in_idx = ((b * channels + c) * h + ih) * w + iw;
                            let val = data[in_idx];
                            if val > max_val {
                                max_val = val;
                                max_idx = in_idx;
                            }
                        }
                    }

                    output[out_idx] = max_val;
                    indices[out_idx] = max_idx;
                }
            }
        }
    }

    let out_shape = vec![batch, channels, out_h, out_w];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(AdaptiveMaxPool2dBackward {
                input: input.clone(),
                indices,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Backward for `AdaptiveMaxPool2d`.
#[derive(Debug)]
struct AdaptiveMaxPool2dBackward<T: Float> {
    input: Tensor<T>,
    indices: Vec<usize>,
}

impl<T: Float> GradFn<T> for AdaptiveMaxPool2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let input_numel = self.input.numel();
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); input_numel];

        for (out_idx, &in_idx) in self.indices.iter().enumerate() {
            grad_input[in_idx] += go_data[out_idx];
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AdaptiveMaxPool2dBackward"
    }
}

// ===========================================================================
// AdaptiveAvgPool1d — CL-315
// ===========================================================================

/// 1D adaptive average pooling layer.
///
/// Input shape: `[B, C, L]`
/// Output shape: `[B, C, output_size]`
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool1d {
    pub output_size: usize,
}

impl AdaptiveAvgPool1d {
    /// Create a new `AdaptiveAvgPool1d` targeting the given output length.
    pub fn new(output_size: usize) -> Self {
        Self { output_size }
    }
}

impl<T: Float> Module<T> for AdaptiveAvgPool1d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        adaptive_avg_pool1d_forward(input, self.output_size)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Forward computation for 1D adaptive average pooling.
fn adaptive_avg_pool1d_forward<T: Float>(
    input: &Tensor<T>,
    output_size: usize,
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, l) = validate_3d(input)?;
    let out_l = output_size;

    if out_l == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "adaptive output_size must be > 0".into(),
        });
    }

    let input_device = input.device();
    let data = input.data_vec()?;
    let total = batch * channels * out_l;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];

    for b in 0..batch {
        for c in 0..channels {
            for ol in 0..out_l {
                let l_start = adaptive_start(ol, l, out_l);
                let l_end = adaptive_end(ol, l, out_l);
                let window = l_end - l_start;
                let mut sum = <T as num_traits::Zero>::zero();

                for il in l_start..l_end {
                    let in_idx = (b * channels + c) * l + il;
                    sum += data[in_idx];
                }

                let out_idx = (b * channels + c) * out_l + ol;
                output[out_idx] = sum / T::from(window).unwrap();
            }
        }
    }

    let out_shape = vec![batch, channels, out_l];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(AdaptiveAvgPool1dBackward {
                input: input.clone(),
                output_size,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Backward for `AdaptiveAvgPool1d`.
#[derive(Debug)]
struct AdaptiveAvgPool1dBackward<T: Float> {
    input: Tensor<T>,
    output_size: usize,
}

impl<T: Float> GradFn<T> for AdaptiveAvgPool1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let in_shape = self.input.shape();
        let (batch, channels, l) = (in_shape[0], in_shape[1], in_shape[2]);
        let out_l = self.output_size;

        let mut grad_input = vec![<T as num_traits::Zero>::zero(); batch * channels * l];

        for b in 0..batch {
            for c in 0..channels {
                for ol in 0..out_l {
                    let l_start = adaptive_start(ol, l, out_l);
                    let l_end = adaptive_end(ol, l, out_l);
                    let window = l_end - l_start;
                    let out_idx = (b * channels + c) * out_l + ol;
                    let grad_val = go_data[out_idx] / T::from(window).unwrap();

                    for il in l_start..l_end {
                        let in_idx = (b * channels + c) * l + il;
                        grad_input[in_idx] += grad_val;
                    }
                }
            }
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AdaptiveAvgPool1dBackward"
    }
}

// ===========================================================================
// AdaptiveAvgPool3d — CL-315
// ===========================================================================

/// 3D adaptive average pooling layer.
///
/// Input shape: `[B, C, D, H, W]`
/// Output shape: `[B, C, output_size.0, output_size.1, output_size.2]`
#[derive(Debug, Clone)]
pub struct AdaptiveAvgPool3d {
    pub output_size: (usize, usize, usize),
}

impl AdaptiveAvgPool3d {
    /// Create a new `AdaptiveAvgPool3d` targeting the given output spatial size.
    pub fn new(output_size: (usize, usize, usize)) -> Self {
        Self { output_size }
    }
}

impl<T: Float> Module<T> for AdaptiveAvgPool3d {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        adaptive_avg_pool3d_forward(input, self.output_size)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        vec![]
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        vec![]
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        vec![]
    }

    fn train(&mut self) {}
    fn eval(&mut self) {}

    fn is_training(&self) -> bool {
        false
    }
}

/// Forward computation for 3D adaptive average pooling.
fn adaptive_avg_pool3d_forward<T: Float>(
    input: &Tensor<T>,
    output_size: (usize, usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, d, h, w) = validate_5d(input)?;
    let (out_d, out_h, out_w) = output_size;

    if out_d == 0 || out_h == 0 || out_w == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "adaptive output_size must be > 0".into(),
        });
    }

    let input_device = input.device();
    let data = input.data_vec()?;
    let total = batch * channels * out_d * out_h * out_w;
    let mut output = vec![<T as num_traits::Zero>::zero(); total];

    for b in 0..batch {
        for c in 0..channels {
            for od in 0..out_d {
                let d_start = adaptive_start(od, d, out_d);
                let d_end = adaptive_end(od, d, out_d);

                for oh in 0..out_h {
                    let h_start = adaptive_start(oh, h, out_h);
                    let h_end = adaptive_end(oh, h, out_h);

                    for ow in 0..out_w {
                        let w_start = adaptive_start(ow, w, out_w);
                        let w_end = adaptive_end(ow, w, out_w);

                        let window_vol = (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
                        let mut sum = <T as num_traits::Zero>::zero();

                        for id in d_start..d_end {
                            for ih in h_start..h_end {
                                for iw in w_start..w_end {
                                    let in_idx = (((b * channels + c) * d + id) * h + ih) * w + iw;
                                    sum += data[in_idx];
                                }
                            }
                        }

                        let out_idx = (((b * channels + c) * out_d + od) * out_h + oh) * out_w + ow;
                        output[out_idx] = sum / T::from(window_vol).unwrap();
                    }
                }
            }
        }
    }

    let out_shape = vec![batch, channels, out_d, out_h, out_w];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(AdaptiveAvgPool3dBackward {
                input: input.clone(),
                output_size,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Backward for `AdaptiveAvgPool3d`.
#[derive(Debug)]
struct AdaptiveAvgPool3dBackward<T: Float> {
    input: Tensor<T>,
    output_size: (usize, usize, usize),
}

impl<T: Float> GradFn<T> for AdaptiveAvgPool3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let in_shape = self.input.shape();
        let (batch, channels, d, h, w) = (
            in_shape[0],
            in_shape[1],
            in_shape[2],
            in_shape[3],
            in_shape[4],
        );
        let (out_d, out_h, out_w) = self.output_size;

        let mut grad_input = vec![<T as num_traits::Zero>::zero(); batch * channels * d * h * w];

        for b in 0..batch {
            for c in 0..channels {
                for od in 0..out_d {
                    let d_start = adaptive_start(od, d, out_d);
                    let d_end = adaptive_end(od, d, out_d);

                    for oh in 0..out_h {
                        let h_start = adaptive_start(oh, h, out_h);
                        let h_end = adaptive_end(oh, h, out_h);

                        for ow in 0..out_w {
                            let w_start = adaptive_start(ow, w, out_w);
                            let w_end = adaptive_end(ow, w, out_w);

                            let window_vol =
                                (d_end - d_start) * (h_end - h_start) * (w_end - w_start);
                            let out_idx =
                                (((b * channels + c) * out_d + od) * out_h + oh) * out_w + ow;
                            let grad_val = go_data[out_idx] / T::from(window_vol).unwrap();

                            for id in d_start..d_end {
                                for ih in h_start..h_end {
                                    for iw in w_start..w_end {
                                        let in_idx =
                                            (((b * channels + c) * d + id) * h + ih) * w + iw;
                                        grad_input[in_idx] += grad_val;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "AdaptiveAvgPool3dBackward"
    }
}

// ===========================================================================
// MaxUnpool2d — CL-315
// ===========================================================================

/// Inverse of `MaxPool2d`.
///
/// Given an output from `MaxPool2d` and the indices of the max positions,
/// scatters the values back into an output tensor of the specified
/// `output_size`. Positions not pointed to by any index remain zero.
///
/// This is commonly used in encoder-decoder architectures (e.g. SegNet)
/// where the pooling indices from the encoder are reused in the decoder.
///
/// Input shape: `[B, C, H, W]` (the pooled tensor)
/// Output shape: `[B, C, output_size.0, output_size.1]`
#[derive(Debug, Clone)]
pub struct MaxUnpool2d {
    pub kernel_size: [usize; 2],
    pub stride: [usize; 2],
    pub padding: [usize; 2],
}

impl MaxUnpool2d {
    /// Create a new `MaxUnpool2d` layer.
    ///
    /// `stride` defaults to `kernel_size` when set to `[0, 0]`.
    pub fn new(kernel_size: [usize; 2], stride: [usize; 2], padding: [usize; 2]) -> Self {
        let stride = if stride == [0, 0] {
            kernel_size
        } else {
            stride
        };
        Self {
            kernel_size,
            stride,
            padding,
        }
    }
}

impl MaxUnpool2d {
    /// Forward pass for `MaxUnpool2d`.
    ///
    /// # Arguments
    ///
    /// * `input` - The pooled tensor, shape `[B, C, H, W]`.
    /// * `indices` - Flat indices from the corresponding `MaxPool2d` forward.
    /// * `output_size` - The desired spatial output size `(H_out, W_out)`.
    pub fn forward_with_indices<T: Float>(
        &self,
        input: &Tensor<T>,
        indices: &[usize],
        output_size: (usize, usize),
    ) -> FerrotorchResult<Tensor<T>> {
        max_unpool2d_forward(input, indices, output_size)
    }
}

/// Functional API for `MaxUnpool2d`.
///
/// Scatters `input` values into an output of shape
/// `[B, C, output_size.0, output_size.1]` using the given flat `indices`.
pub fn max_unpool2d<T: Float>(
    input: &Tensor<T>,
    indices: &[usize],
    output_size: (usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    max_unpool2d_forward(input, indices, output_size)
}

/// Forward computation for max unpooling.
fn max_unpool2d_forward<T: Float>(
    input: &Tensor<T>,
    indices: &[usize],
    output_size: (usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, _h, _w) = validate_4d(input)?;
    let (out_h, out_w) = output_size;

    if input.numel() != indices.len() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "MaxUnpool2d: input numel ({}) != indices len ({})",
                input.numel(),
                indices.len()
            ),
        });
    }

    let input_device = input.device();
    let data = input.data_vec()?;
    let output_numel = batch * channels * out_h * out_w;
    let mut output = vec![<T as num_traits::Zero>::zero(); output_numel];

    // Scatter values to the positions indicated by indices.
    for (i, &idx) in indices.iter().enumerate() {
        if idx >= output_numel {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MaxUnpool2d: index {} out of bounds for output size {}",
                    idx, output_numel
                ),
            });
        }
        output[idx] = data[i];
    }

    let out_shape = vec![batch, channels, out_h, out_w];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(MaxUnpool2dBackward {
                input: input.clone(),
                indices: indices.to_vec(),
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Backward for `MaxUnpool2d`.
///
/// The gradient simply gathers values from the upstream gradient at the
/// index positions (the reverse of the scatter in forward).
#[derive(Debug)]
struct MaxUnpool2dBackward<T: Float> {
    input: Tensor<T>,
    indices: Vec<usize>,
}

impl<T: Float> GradFn<T> for MaxUnpool2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let go_data = grad_output.data_vec()?;
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); self.input.numel()];

        // Gather: grad_input[i] = grad_output[indices[i]]
        for (i, &idx) in self.indices.iter().enumerate() {
            grad_input[i] = go_data[idx];
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "MaxUnpool2dBackward"
    }
}

// ===========================================================================
// Public functional API
// ===========================================================================

/// Functional 1D max pooling. See [`MaxPool1d`] for details.
pub fn max_pool1d<T: Float>(
    input: &Tensor<T>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> FerrotorchResult<Tensor<T>> {
    max_pool1d_forward(input, kernel_size, stride, padding)
}

/// Functional 2D max pooling. See [`MaxPool2d`] for details.
pub fn max_pool2d<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> FerrotorchResult<Tensor<T>> {
    max_pool2d_forward(input, kernel_size, stride, padding)
}

/// Functional 3D max pooling. See [`MaxPool3d`] for details.
pub fn max_pool3d<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
) -> FerrotorchResult<Tensor<T>> {
    max_pool3d_forward(input, kernel_size, stride, padding)
}

/// Functional 1D average pooling. See [`AvgPool1d`] for details.
pub fn avg_pool1d<T: Float>(
    input: &Tensor<T>,
    kernel_size: usize,
    stride: usize,
    padding: usize,
) -> FerrotorchResult<Tensor<T>> {
    avg_pool1d_forward(input, kernel_size, stride, padding)
}

/// Functional 2D average pooling. See [`AvgPool2d`] for details.
pub fn avg_pool2d<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
) -> FerrotorchResult<Tensor<T>> {
    avg_pool2d_forward(input, kernel_size, stride, padding)
}

/// Functional 3D average pooling. See [`AvgPool3d`] for details.
pub fn avg_pool3d<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 3],
    stride: [usize; 3],
    padding: [usize; 3],
) -> FerrotorchResult<Tensor<T>> {
    avg_pool3d_forward(input, kernel_size, stride, padding)
}

/// Functional 1D adaptive average pooling. See [`AdaptiveAvgPool1d`] for details.
pub fn adaptive_avg_pool1d<T: Float>(
    input: &Tensor<T>,
    output_size: usize,
) -> FerrotorchResult<Tensor<T>> {
    adaptive_avg_pool1d_forward(input, output_size)
}

/// Functional 2D adaptive average pooling. See [`AdaptiveAvgPool2d`] for details.
pub fn adaptive_avg_pool2d<T: Float>(
    input: &Tensor<T>,
    output_size: (usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    adaptive_avg_pool2d_forward(input, output_size)
}

/// Functional 3D adaptive average pooling. See [`AdaptiveAvgPool3d`] for details.
pub fn adaptive_avg_pool3d<T: Float>(
    input: &Tensor<T>,
    output_size: (usize, usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    adaptive_avg_pool3d_forward(input, output_size)
}

/// Functional 2D adaptive max pooling. See [`AdaptiveMaxPool2d`] for details.
pub fn adaptive_max_pool2d<T: Float>(
    input: &Tensor<T>,
    output_size: (usize, usize),
) -> FerrotorchResult<Tensor<T>> {
    adaptive_max_pool2d_forward(input, output_size)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a leaf 4D tensor from flat data.
    fn leaf_4d(data: &[f32], shape: [usize; 4], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // MaxPool2d tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_maxpool2d_output_shape() {
        // Input: [1, 1, 4, 4], kernel 2x2, stride 2, no padding
        // Output: [1, 1, 2, 2]
        let input = leaf_4d(&[0.0; 16], [1, 1, 4, 4], false);
        let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
    }

    #[test]
    fn test_maxpool2d_output_shape_with_padding() {
        // Input: [2, 3, 5, 5], kernel 3x3, stride 1, padding 1
        // H_out = (5 + 2*1 - 3) / 1 + 1 = 5
        let input = leaf_4d(&[0.0; 150], [2, 3, 5, 5], false);
        let pool = MaxPool2d::new([3, 3], [1, 1], [1, 1]);
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        assert_eq!(out.shape(), &[2, 3, 5, 5]);
    }

    #[test]
    fn test_maxpool2d_forward_correctness() {
        // Input [1, 1, 4, 4]:
        //  1  2  3  4
        //  5  6  7  8
        //  9 10 11 12
        // 13 14 15 16
        //
        // kernel 2x2, stride 2 => output [1, 1, 2, 2]:
        //  6  8
        // 14 16
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = leaf_4d(&data, [1, 1, 4, 4], false);
        let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        assert_eq!(out.data().unwrap(), &[6.0, 8.0, 14.0, 16.0]);
    }

    #[test]
    fn test_maxpool2d_forward_stride1() {
        // Input [1, 1, 3, 3]:
        // 1 3 2
        // 4 6 5
        // 7 9 8
        //
        // kernel 2x2, stride 1 => output [1, 1, 2, 2]:
        //  max(1,3,4,6)=6  max(3,2,6,5)=6
        //  max(4,6,7,9)=9  max(6,5,9,8)=9
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 3.0, 2.0,
            4.0, 6.0, 5.0,
            7.0, 9.0, 8.0,
        ];
        let input = leaf_4d(&data, [1, 1, 3, 3], false);
        let pool = MaxPool2d::new([2, 2], [1, 1], [0, 0]);
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        assert_eq!(out.data().unwrap(), &[6.0, 6.0, 9.0, 9.0]);
    }

    #[test]
    fn test_maxpool2d_backward() {
        // Input [1, 1, 4, 4], kernel 2x2, stride 2
        // Max indices: (1,1)=6, (1,3)=8, (3,1)=14, (3,3)=16
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = leaf_4d(&data, [1, 1, 4, 4], true);
        let out = max_pool2d(&input, [2, 2], [2, 2], [0, 0]).unwrap();

        // Manually construct a scalar loss = sum(out) for backward.
        let out_data = out.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward { input: out }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // Gradient should be 1.0 at max positions, 0.0 elsewhere.
        #[rustfmt::skip]
        let expected: Vec<f32> = vec![
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 1.0,
        ];
        for (i, (&got, &exp)) in g.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "grad[{i}]: expected {exp}, got {got}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // AvgPool2d tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_avgpool2d_output_shape() {
        let input = leaf_4d(&[0.0; 48], [1, 3, 4, 4], false);
        let pool = AvgPool2d::new([2, 2], [2, 2], [0, 0]);
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        assert_eq!(out.shape(), &[1, 3, 2, 2]);
    }

    #[test]
    fn test_avgpool2d_forward_correctness() {
        // Input [1, 1, 4, 4]:
        //  1  2  3  4
        //  5  6  7  8
        //  9 10 11 12
        // 13 14 15 16
        //
        // kernel 2x2, stride 2 => output [1, 1, 2, 2]:
        //  avg(1,2,5,6)=3.5    avg(3,4,7,8)=5.5
        //  avg(9,10,13,14)=11.5 avg(11,12,15,16)=13.5
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = leaf_4d(&data, [1, 1, 4, 4], false);
        let pool = AvgPool2d::new([2, 2], [2, 2], [0, 0]);
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        let d = out.data().unwrap();
        assert!((d[0] - 3.5).abs() < 1e-6);
        assert!((d[1] - 5.5).abs() < 1e-6);
        assert!((d[2] - 11.5).abs() < 1e-6);
        assert!((d[3] - 13.5).abs() < 1e-6);
    }

    #[test]
    fn test_avgpool2d_forward_stride1() {
        // Input [1, 1, 3, 3]:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        //
        // kernel 2x2, stride 1 => output [1, 1, 2, 2]:
        //  avg(1,2,4,5)=3.0  avg(2,3,5,6)=4.0
        //  avg(4,5,7,8)=6.0  avg(5,6,8,9)=7.0
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = leaf_4d(&data, [1, 1, 3, 3], false);
        let pool = AvgPool2d::new([2, 2], [1, 1], [0, 0]);
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        let d = out.data().unwrap();
        assert!((d[0] - 3.0).abs() < 1e-6);
        assert!((d[1] - 4.0).abs() < 1e-6);
        assert!((d[2] - 6.0).abs() < 1e-6);
        assert!((d[3] - 7.0).abs() < 1e-6);
    }

    #[test]
    fn test_avgpool2d_backward() {
        // Input [1, 1, 4, 4], kernel 2x2, stride 2
        // Each output element distributes grad / 4 to its 4 input positions.
        // With grad_output = all 1s, each input position that's covered gets 0.25.
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = leaf_4d(&data, [1, 1, 4, 4], true);
        let out = avg_pool2d(&input, [2, 2], [2, 2], [0, 0]).unwrap();

        let out_data = out.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(SumBackward { input: out }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // Every input position is covered by exactly one window (stride = kernel_size).
        // grad = 1.0 / 4 = 0.25 for all positions.
        for (i, &val) in g.iter().enumerate() {
            assert!(
                (val - 0.25).abs() < 1e-6,
                "grad[{i}]: expected 0.25, got {val}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // AdaptiveAvgPool2d tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_adaptive_avgpool2d_output_shape() {
        let input = leaf_4d(&[0.0; 75], [1, 3, 5, 5], false);
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        assert_eq!(out.shape(), &[1, 3, 1, 1]);
    }

    #[test]
    fn test_adaptive_avgpool2d_global() {
        // Global average pooling: output (1, 1) => mean of entire spatial plane.
        // Input [1, 1, 2, 2]: 1, 2, 3, 4 => mean = 2.5
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], false);
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert!((out.data().unwrap()[0] - 2.5).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_avgpool2d_identity() {
        // Output size matches input => identity.
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = leaf_4d(&data, [1, 1, 3, 3], false);
        let pool = AdaptiveAvgPool2d::new((3, 3));
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 3, 3]);
        let d = out.data().unwrap();
        for (i, (&got, &exp)) in d.iter().zip(data.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "output[{i}]: expected {exp}, got {got}"
            );
        }
    }

    #[test]
    fn test_adaptive_avgpool2d_2x2() {
        // Input [1, 1, 4, 4] => output (2, 2).
        // PyTorch adaptive formula:
        //   h_start(0) = 0*4/2=0, h_end(0) = ceil(1*4/2)=2
        //   h_start(1) = 1*4/2=2, h_end(1) = ceil(2*4/2)=4
        //   Same for w. So windows are [0..2, 0..2], [0..2, 2..4], [2..4, 0..2], [2..4, 2..4].
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
             1.0,  2.0,  3.0,  4.0,
             5.0,  6.0,  7.0,  8.0,
             9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = leaf_4d(&data, [1, 1, 4, 4], false);
        let pool = AdaptiveAvgPool2d::new((2, 2));
        let out: Tensor<f32> = Module::<f32>::forward(&pool, &input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        let d = out.data().unwrap();
        // Window [0..2, 0..2]: avg(1,2,5,6) = 3.5
        assert!((d[0] - 3.5).abs() < 1e-6);
        // Window [0..2, 2..4]: avg(3,4,7,8) = 5.5
        assert!((d[1] - 5.5).abs() < 1e-6);
        // Window [2..4, 0..2]: avg(9,10,13,14) = 11.5
        assert!((d[2] - 11.5).abs() < 1e-6);
        // Window [2..4, 2..4]: avg(11,12,15,16) = 13.5
        assert!((d[3] - 13.5).abs() < 1e-6);
    }

    #[test]
    fn test_adaptive_avgpool2d_backward() {
        // Input [1, 1, 4, 4] => output (1, 1) = global avg.
        // loss = output[0] (scalar).
        // d(loss)/d(input[i]) = 1/16 for all i.
        let data: Vec<f32> = (1..=16).map(|x| x as f32).collect();
        let input = leaf_4d(&data, [1, 1, 4, 4], true);
        let out = adaptive_avg_pool2d(&input, (1, 1)).unwrap();

        // out is [1, 1, 1, 1], so item() works after reshape to scalar.
        let out_val = out.data().unwrap()[0];
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![out_val]),
            vec![],
            Arc::new(SumBackward { input: out }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        let expected = 1.0 / 16.0;
        for (i, &val) in g.iter().enumerate() {
            assert!(
                (val - expected).abs() < 1e-6,
                "grad[{i}]: expected {expected}, got {val}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Error handling tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_pooling_rejects_3d_input() {
        let input =
            Tensor::<f32>::from_storage(TensorStorage::cpu(vec![0.0; 12]), vec![2, 3, 2], false)
                .unwrap();
        assert!(max_pool2d(&input, [2, 2], [1, 1], [0, 0]).is_err());
        assert!(avg_pool2d(&input, [2, 2], [1, 1], [0, 0]).is_err());
        assert!(adaptive_avg_pool2d(&input, (1, 1)).is_err());
    }

    #[test]
    fn test_pooling_zero_kernel_rejected() {
        let input = leaf_4d(&[0.0; 16], [1, 1, 4, 4], false);
        assert!(max_pool2d(&input, [0, 2], [1, 1], [0, 0]).is_err());
        assert!(avg_pool2d(&input, [2, 0], [1, 1], [0, 0]).is_err());
    }

    #[test]
    fn test_pooling_zero_stride_defaults_to_kernel() {
        let _input = leaf_4d(&[0.0; 16], [1, 1, 4, 4], false);
        let pool = MaxPool2d::new([2, 2], [0, 0], [0, 0]);
        assert_eq!(pool.stride, [2, 2]);
    }

    #[test]
    fn test_maxpool2d_zero_parameters() {
        let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);
        let params: Vec<&Parameter<f32>> = Module::<f32>::parameters(&pool);
        assert!(params.is_empty());
    }

    #[test]
    fn test_avgpool2d_zero_parameters() {
        let pool = AvgPool2d::new([2, 2], [2, 2], [0, 0]);
        let params: Vec<&Parameter<f32>> = Module::<f32>::parameters(&pool);
        assert!(params.is_empty());
    }

    #[test]
    fn test_adaptive_avgpool2d_zero_parameters() {
        let pool = AdaptiveAvgPool2d::new((1, 1));
        let params: Vec<&Parameter<f32>> = Module::<f32>::parameters(&pool);
        assert!(params.is_empty());
    }

    #[test]
    fn test_maxpool2d_batch_channels() {
        // Verify pooling works independently per batch/channel.
        // [2, 2, 4, 4] with known data.
        let mut data = Vec::with_capacity(64);
        for b in 0..2 {
            for c in 0..2 {
                let offset = (b * 2 + c) as f32 * 100.0;
                for i in 0..16 {
                    data.push(offset + i as f32);
                }
            }
        }
        let input = leaf_4d(&data, [2, 2, 4, 4], false);
        let out = max_pool2d(&input, [2, 2], [2, 2], [0, 0]).unwrap();
        assert_eq!(out.shape(), &[2, 2, 2, 2]);

        let d = out.data().unwrap();
        // Batch 0, Channel 0: offset=0, max of [0..3,4..7] etc.
        assert!((d[0] - 5.0).abs() < 1e-6); // max(0,1,4,5)=5
        assert!((d[1] - 7.0).abs() < 1e-6); // max(2,3,6,7)=7
    }

    // -----------------------------------------------------------------------
    // Helper backward node for tests
    // -----------------------------------------------------------------------

    /// Sum reduction backward for test use.
    /// loss = sum(input); d(loss)/d(input_i) = 1.
    #[derive(Debug)]
    struct SumBackward<T: Float> {
        input: Tensor<T>,
    }

    impl<T: Float> GradFn<T> for SumBackward<T> {
        fn backward(&self, _grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
            let ones_data = vec![<T as num_traits::One>::one(); self.input.numel()];
            let ones = Tensor::from_storage(
                TensorStorage::cpu(ones_data),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(ones)])
        }

        fn inputs(&self) -> Vec<&Tensor<T>> {
            vec![&self.input]
        }

        fn name(&self) -> &'static str {
            "SumBackward"
        }
    }
}
