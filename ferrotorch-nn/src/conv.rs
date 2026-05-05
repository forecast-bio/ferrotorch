//! Convolution layers: 1-D, 2-D, 3-D and their transposed variants.
//!
//! Implements `Conv1d<T>`, `Conv2d<T>`, `Conv3d<T>`, `ConvTranspose1d<T>`,
//! `ConvTranspose2d<T>`, and `ConvTranspose3d<T>`.
//! Forward passes use the im2col + matmul approach; backward follows the
//! same structure in reverse.

use std::sync::Arc;

use ferrotorch_core::autograd::autocast_ops::autocast_guard;
use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::ops::linalg::{mm, transpose};
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};

use crate::init::{NonLinearity, kaiming_uniform, zeros as zeros_init};
use crate::module::Module;
use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// im2col / col2im helpers
// ---------------------------------------------------------------------------

/// Extract image patches into columns.
///
/// Given a 4-D input `[B, C, H, W]`, produces a 3-D output
/// `[B, C * kH * kW, H_out * W_out]` where each column is one
/// flattened receptive-field patch.
// Internal kernel: argument set mirrors the 2-D convolution descriptor
// (B, C, H, W, kH, kW, padH, padW, strideH, strideW, dilH, dilW); a config
// struct would force allocation on every call in convolution hot paths.
#[allow(clippy::too_many_arguments)]
fn im2col<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> (Vec<T>, usize, usize) {
    let h_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    let w_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    let col_rows = channels * kernel_h * kernel_w;
    let col_cols = h_out * w_out;

    let zero = <T as num_traits::Zero>::zero();
    let mut cols = vec![zero; batch * col_rows * col_cols];

    for b in 0..batch {
        for c in 0..channels {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;
                            let col = oh * w_out + ow;

                            // Account for padding: the "virtual" input coordinate
                            // must be shifted back by the padding amount.
                            let val = if ih >= pad_h
                                && iw >= pad_w
                                && (ih - pad_h) < height
                                && (iw - pad_w) < width
                            {
                                let real_h = ih - pad_h;
                                let real_w = iw - pad_w;
                                input[b * channels * height * width
                                    + c * height * width
                                    + real_h * width
                                    + real_w]
                            } else {
                                zero
                            };

                            cols[b * col_rows * col_cols + row * col_cols + col] = val;
                        }
                    }
                }
            }
        }
    }

    (cols, col_rows, col_cols)
}

/// Scatter columns back into an image tensor (adjoint of im2col).
///
/// Given columns of shape `[B, C * kH * kW, H_out * W_out]`, accumulates
/// values back into a `[B, C, H, W]` tensor (with padding stripped).
// Internal kernel: argument set is the adjoint of `im2col` (same descriptor
// inputs); refactoring to a config struct would diverge the two helpers'
// signatures unhelpfully.
#[allow(clippy::too_many_arguments)]
fn col2im<T: Float>(
    cols: &[T],
    batch: usize,
    channels: usize,
    height: usize,
    width: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut output = vec![zero; batch * channels * height * width];

    let col_rows = channels * kernel_h * kernel_w;
    let col_cols = h_out * w_out;

    for b in 0..batch {
        for c in 0..channels {
            for kh in 0..kernel_h {
                for kw in 0..kernel_w {
                    let row = c * kernel_h * kernel_w + kh * kernel_w + kw;
                    for oh in 0..h_out {
                        for ow in 0..w_out {
                            let ih = oh * stride_h + kh;
                            let iw = ow * stride_w + kw;
                            let col = oh * w_out + ow;

                            if ih >= pad_h
                                && iw >= pad_w
                                && (ih - pad_h) < height
                                && (iw - pad_w) < width
                            {
                                let real_h = ih - pad_h;
                                let real_w = iw - pad_w;
                                output[b * channels * height * width
                                    + c * height * width
                                    + real_h * width
                                    + real_w] +=
                                    cols[b * col_rows * col_cols + row * col_cols + col];
                            }
                        }
                    }
                }
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Conv2d
// ---------------------------------------------------------------------------

/// A 2-D convolution layer.
///
/// Applies a spatial convolution over an input `[B, C_in, H, W]` using
/// the im2col + matmul algorithm. Equivalent to `torch.nn.Conv2d`.
///
/// # Shape
///
/// - Input: `[B, in_channels, H, W]`
/// - Output: `[B, out_channels, H_out, W_out]`
///
/// where `H_out = (H + 2 * padding.0 - kernel_size.0) / stride.0 + 1`.
#[derive(Debug)]
pub struct Conv2d<T: Float> {
    /// Learnable kernel weights `[out_channels, in_channels, kH, kW]`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels (filters).
    out_channels: usize,
    /// Kernel spatial size `(kH, kW)`.
    kernel_size: (usize, usize),
    /// Stride `(sH, sW)`.
    stride: (usize, usize),
    /// Zero-padding `(pH, pW)` applied to both sides.
    padding: (usize, usize),
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> Conv2d<T> {
    /// Create a new `Conv2d` layer.
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized to zeros.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0 in both dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0 in both dimensions".into(),
            });
        }

        let (kh, kw) = kernel_size;
        let mut weight = Parameter::zeros(&[out_channels, in_channels, kh, kw])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            zeros_init(&mut b)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        })
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.out_channels * self.in_channels * self.kernel_size.0 * self.kernel_size.1;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `Conv2d` from caller-supplied weight and optional bias tensors.
    ///
    /// `weight` must have shape `[out_channels, in_channels, kH, kW]`. If
    /// `bias` is provided, it must be 1-D of length `out_channels`. The
    /// stride and padding are passed through unchanged. This is the constructor
    /// used by `nn::functional::conv2d` so callers can drive the existing
    /// im2col + matmul forward path with their own parameters (e.g. for
    /// stateless functional dispatch or weight sharing across modules).
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 4 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv2d::from_parts: weight must be 4-D [out, in, kH, kW], got {:?}",
                    weight.shape()
                ),
            });
        }
        let out_channels = weight.shape()[0];
        let in_channels = weight.shape()[1];
        let kernel_size = (weight.shape()[2], weight.shape()[3]);
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "Conv2d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Conv2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv2d.
        let _autocast_cat = autocast_guard("conv2d");

        // Validate input shape: [B, C_in, H, W].
        if input.ndim() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv2d expects 4-D input [B, C, H, W], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let h = input.shape()[2];
        let w = input.shape()[3];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv2d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // Check that the kernel fits.
        let h_padded = h + 2 * ph;
        let w_padded = w + 2 * pw;
        if h_padded < kh || w_padded < kw {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv2d: padded input ({h_padded}, {w_padded}) is smaller than kernel ({kh}, {kw})"
                ),
            });
        }

        let h_out = (h_padded - kh) / sh + 1;
        let w_out = (w_padded - kw) / sw + 1;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // ---- GPU fast path: fully on-device conv2d ----
        let is_f32 = std::mem::size_of::<T>() == 4;
        if is_f32 && input.is_cuda() {
            if let Some(backend) = ferrotorch_core::gpu_dispatch::gpu_backend() {
                let bias_handle = self
                    .bias
                    .as_ref()
                    .and_then(|b| b.tensor().gpu_handle().ok());
                let (out_handle, out_shape) = backend.conv2d_f32(
                    input.gpu_handle()?,
                    self.weight.tensor().gpu_handle()?,
                    bias_handle,
                    [batch, c_in, h, w],
                    [self.out_channels, self.in_channels, kh, kw],
                    self.stride,
                    self.padding,
                )?;

                let result = Tensor::from_storage(
                    TensorStorage::gpu(out_handle),
                    out_shape.to_vec(),
                    false,
                )?;

                // For backward, fall through to CPU path if gradients needed
                // (GPU backward not yet implemented — stores input for recomputation)
                if is_grad_enabled()
                    && (input.requires_grad()
                        || self.weight.requires_grad()
                        || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
                {
                    // Download cols for backward (CPU backward path).
                    let input_data = input.data_vec()?;
                    let (cols, col_rows, col_cols) =
                        im2col(&input_data, batch, c_in, h, w, kh, kw, sh, sw, ph, pw);
                    let grad_fn = Arc::new(Conv2dBackward {
                        input: input.clone(),
                        weight: self.weight.tensor().clone(),
                        bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                        in_channels: self.in_channels,
                        out_channels: self.out_channels,
                        kernel_size: self.kernel_size,
                        stride: self.stride,
                        padding: self.padding,
                        cols,
                        col_rows,
                        col_cols,
                        h_out,
                        w_out,
                    });
                    return Tensor::from_operation(
                        result.into_storage_and_shape()?.0,
                        out_shape.to_vec(),
                        grad_fn,
                    );
                }

                return Ok(result);
            }
        }

        // ---- CPU path ----
        let input_data = input.data_vec()?;

        // im2col: [B, C_in * kH * kW, H_out * W_out]
        let (cols, col_rows, col_cols) =
            im2col(&input_data, batch, c_in, h, w, kh, kw, sh, sw, ph, pw);

        // Reshape weight to 2D: [C_out, C_in * kH * kW]
        let weight_data = self.weight.data_vec()?;
        let weight_2d = Tensor::from_storage(
            TensorStorage::cpu(weight_data),
            vec![self.out_channels, col_rows],
            false,
        )?;

        // Per-batch matmul: weight_2d @ cols_b -> [C_out, H_out * W_out]
        let zero = <T as num_traits::Zero>::zero();
        let mut output = vec![zero; batch * self.out_channels * h_out * w_out];

        for b in 0..batch {
            let col_start = b * col_rows * col_cols;
            let col_end = col_start + col_rows * col_cols;
            let cols_b = Tensor::from_storage(
                TensorStorage::cpu(cols[col_start..col_end].to_vec()),
                vec![col_rows, col_cols],
                false,
            )?;

            let out_b = mm(&weight_2d, &cols_b)?;
            let out_data = out_b.data()?;
            let out_start = b * self.out_channels * h_out * w_out;
            output[out_start..out_start + self.out_channels * h_out * w_out]
                .copy_from_slice(out_data);
        }

        // Add bias if present: broadcast [C_out] over [B, C_out, H_out, W_out].
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for hw in 0..(h_out * w_out) {
                        output[b * self.out_channels * h_out * w_out + c * h_out * w_out + hw] +=
                            bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, h_out, w_out],
            false,
        )?;

        // Attach backward if gradients are enabled and any input/param requires grad.
        if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            let grad_fn = Arc::new(Conv2dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                cols,
                col_rows,
                col_cols,
                h_out,
                w_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Conv2dBackward
// ---------------------------------------------------------------------------

/// Backward function for `Conv2d` forward pass.
///
/// Saved tensors:
/// - `input`: the original 4-D input
/// - `weight`: the 4-D kernel
/// - `bias`: optional 1-D bias
/// - `cols`: the im2col columns from the forward pass (avoids recomputation)
#[derive(Debug)]
struct Conv2dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    cols: Vec<T>,
    col_rows: usize,
    col_cols: usize,
    h_out: usize,
    w_out: usize,
}

impl<T: Float> GradFn<T> for Conv2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, H_out, W_out]
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let h = self.input.shape()[2];
        let w = self.input.shape()[3];
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // --- grad_weight ---
        // For each batch element:
        //   grad_output_b: [C_out, H_out * W_out]
        //   cols_b:        [col_rows, col_cols] = [C_in * kH * kW, H_out * W_out]
        //   grad_weight += grad_output_b @ cols_b^T
        // Result shape: [C_out, C_in * kH * kW] -> reshape to [C_out, C_in, kH, kW]
        let grad_weight = if self.weight.requires_grad() {
            let zero = <T as num_traits::Zero>::zero();
            let weight_numel = self.out_channels * self.col_rows;
            let mut gw_accum = vec![zero; weight_numel];

            for b in 0..batch {
                // grad_output for this batch: [C_out, H_out * W_out]
                let go_start = b * self.out_channels * self.h_out * self.w_out;
                let go_end = go_start + self.out_channels * self.h_out * self.w_out;
                let go_b = Tensor::from_storage(
                    TensorStorage::cpu(go_data[go_start..go_end].to_vec()),
                    vec![self.out_channels, self.h_out * self.w_out],
                    false,
                )?;

                // cols for this batch: [col_rows, col_cols]
                let col_start = b * self.col_rows * self.col_cols;
                let col_end = col_start + self.col_rows * self.col_cols;
                let cols_b = Tensor::from_storage(
                    TensorStorage::cpu(self.cols[col_start..col_end].to_vec()),
                    vec![self.col_rows, self.col_cols],
                    false,
                )?;

                // go_b @ cols_b^T -> [C_out, col_rows]
                let cols_bt = transpose(&cols_b)?;
                let gw_b = mm(&go_b, &cols_bt)?;
                let gw_data = gw_b.data()?;

                for i in 0..weight_numel {
                    gw_accum[i] += gw_data[i];
                }
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gw_accum),
                vec![self.out_channels, self.in_channels, kh, kw],
                false,
            )?)
        } else {
            None
        };

        // --- grad_bias ---
        // Sum grad_output over batch, height, width: sum over [B, *, H_out, W_out]
        // Result shape: [C_out]
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let zero = <T as num_traits::Zero>::zero();
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for hw in 0..(self.h_out * self.w_out) {
                            gb[c] +=
                                go_data[batch_idx * self.out_channels * self.h_out * self.w_out
                                    + c * self.h_out * self.w_out
                                    + hw];
                        }
                    }
                }
                Some(Tensor::from_storage(
                    TensorStorage::cpu(gb),
                    vec![self.out_channels],
                    false,
                )?)
            }
            _ => None,
        };

        // --- grad_input ---
        // For each batch element:
        //   weight_2d^T @ grad_output_b -> [col_rows, H_out * W_out]
        //   then col2im to get [C_in, H, W]
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let weight_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data),
                vec![self.out_channels, self.col_rows],
                false,
            )?;
            let weight_2d_t = transpose(&weight_2d)?;

            let zero = <T as num_traits::Zero>::zero();
            let mut grad_cols = vec![zero; batch * self.col_rows * self.col_cols];

            for b in 0..batch {
                let go_start = b * self.out_channels * self.h_out * self.w_out;
                let go_end = go_start + self.out_channels * self.h_out * self.w_out;
                let go_b = Tensor::from_storage(
                    TensorStorage::cpu(go_data[go_start..go_end].to_vec()),
                    vec![self.out_channels, self.h_out * self.w_out],
                    false,
                )?;

                // weight_2d^T @ go_b -> [col_rows, H_out * W_out]
                let gc_b = mm(&weight_2d_t, &go_b)?;
                let gc_data = gc_b.data()?;

                let gc_start = b * self.col_rows * self.col_cols;
                grad_cols[gc_start..gc_start + self.col_rows * self.col_cols]
                    .copy_from_slice(gc_data);
            }

            // col2im to scatter back to [B, C_in, H, W]
            let gi = col2im(
                &grad_cols,
                batch,
                self.in_channels,
                h,
                w,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
                self.h_out,
                self.w_out,
            );

            Some(Tensor::from_storage(
                TensorStorage::cpu(gi),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        // Return exactly as many gradients as inputs() returns.
        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "Conv2dBackward"
    }
}

// ---------------------------------------------------------------------------
// Conv1d
// ---------------------------------------------------------------------------

/// A 1-D convolution layer for sequence data.
///
/// Applies a temporal convolution over an input `[B, C_in, L]` using
/// the im2col + matmul algorithm (delegates to the 2-D helpers with H=1).
/// Equivalent to `torch.nn.Conv1d`.
///
/// # Shape
///
/// - Input: `[B, in_channels, L]`
/// - Output: `[B, out_channels, L_out]`
///
/// where `L_out = (L + 2 * padding - kernel_size) / stride + 1`.
#[derive(Debug)]
pub struct Conv1d<T: Float> {
    /// Learnable kernel weights `[out_channels, in_channels, kernel_size]`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels (filters).
    out_channels: usize,
    /// Kernel length.
    kernel_size: usize,
    /// Stride.
    stride: usize,
    /// Zero-padding applied to both sides.
    padding: usize,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> Conv1d<T> {
    /// Create a new `Conv1d` layer.
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized to zeros.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
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

        let mut weight = Parameter::zeros(&[out_channels, in_channels, kernel_size])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            zeros_init(&mut b)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        })
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.out_channels * self.in_channels * self.kernel_size;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `Conv1d` from caller-supplied weight and optional bias tensors.
    ///
    /// `weight` must have shape `[out_channels, in_channels, kernel_size]`.
    /// Used by `nn::functional::conv1d`.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: usize,
        padding: usize,
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 3 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv1d::from_parts: weight must be 3-D [out, in, k], got {:?}",
                    weight.shape()
                ),
            });
        }
        let out_channels = weight.shape()[0];
        let in_channels = weight.shape()[1];
        let kernel_size = weight.shape()[2];
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "Conv1d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Conv1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv1d.
        let _autocast_cat = autocast_guard("conv1d");

        // Validate input shape: [B, C_in, L].
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv1d expects 3-D input [B, C, L], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let length = input.shape()[2];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv1d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;

        let l_padded = length + 2 * p;
        if l_padded < k {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv1d: padded input length ({l_padded}) is smaller than kernel ({k})"
                ),
            });
        }

        let l_out = (l_padded - k) / s + 1;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // Reshape input [B, C_in, L] -> [B, C_in, 1, L] and use 2-D im2col
        // with kernel (1, k), stride (1, s), padding (0, p).
        let input_data = input.data_vec()?;

        let (cols, col_rows, col_cols) =
            im2col(&input_data, batch, c_in, 1, length, 1, k, 1, s, 0, p);

        // Reshape weight [C_out, C_in, k] to 2-D: [C_out, C_in * k]
        let weight_data = self.weight.data_vec()?;
        let weight_2d = Tensor::from_storage(
            TensorStorage::cpu(weight_data),
            vec![self.out_channels, col_rows],
            false,
        )?;

        // Per-batch matmul: weight_2d @ cols_b -> [C_out, L_out]
        let zero = <T as num_traits::Zero>::zero();
        let mut output = vec![zero; batch * self.out_channels * l_out];

        for b in 0..batch {
            let col_start = b * col_rows * col_cols;
            let col_end = col_start + col_rows * col_cols;
            let cols_b = Tensor::from_storage(
                TensorStorage::cpu(cols[col_start..col_end].to_vec()),
                vec![col_rows, col_cols],
                false,
            )?;

            let out_b = mm(&weight_2d, &cols_b)?;
            let out_data = out_b.data()?;
            let out_start = b * self.out_channels * l_out;
            output[out_start..out_start + self.out_channels * l_out].copy_from_slice(out_data);
        }

        // Add bias if present: broadcast [C_out] over [B, C_out, L_out].
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for l in 0..l_out {
                        output[b * self.out_channels * l_out + c * l_out + l] += bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, l_out],
            false,
        )?;

        // Attach backward if gradients are enabled.
        if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            let grad_fn = Arc::new(Conv1dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                cols,
                col_rows,
                col_cols,
                l_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Conv1dBackward
// ---------------------------------------------------------------------------

/// Backward function for `Conv1d` forward pass.
#[derive(Debug)]
struct Conv1dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    cols: Vec<T>,
    col_rows: usize,
    col_cols: usize,
    l_out: usize,
}

impl<T: Float> GradFn<T> for Conv1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, L_out]
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let length = self.input.shape()[2];
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;

        // --- grad_weight ---
        let grad_weight = if self.weight.requires_grad() {
            let zero = <T as num_traits::Zero>::zero();
            let weight_numel = self.out_channels * self.col_rows;
            let mut gw_accum = vec![zero; weight_numel];

            for b in 0..batch {
                let go_start = b * self.out_channels * self.l_out;
                let go_end = go_start + self.out_channels * self.l_out;
                let go_b = Tensor::from_storage(
                    TensorStorage::cpu(go_data[go_start..go_end].to_vec()),
                    vec![self.out_channels, self.l_out],
                    false,
                )?;

                let col_start = b * self.col_rows * self.col_cols;
                let col_end = col_start + self.col_rows * self.col_cols;
                let cols_b = Tensor::from_storage(
                    TensorStorage::cpu(self.cols[col_start..col_end].to_vec()),
                    vec![self.col_rows, self.col_cols],
                    false,
                )?;

                let cols_bt = transpose(&cols_b)?;
                let gw_b = mm(&go_b, &cols_bt)?;
                let gw_data = gw_b.data()?;

                for i in 0..weight_numel {
                    gw_accum[i] += gw_data[i];
                }
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gw_accum),
                vec![self.out_channels, self.in_channels, k],
                false,
            )?)
        } else {
            None
        };

        // --- grad_bias ---
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let zero = <T as num_traits::Zero>::zero();
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for l in 0..self.l_out {
                            gb[c] += go_data
                                [batch_idx * self.out_channels * self.l_out + c * self.l_out + l];
                        }
                    }
                }
                Some(Tensor::from_storage(
                    TensorStorage::cpu(gb),
                    vec![self.out_channels],
                    false,
                )?)
            }
            _ => None,
        };

        // --- grad_input ---
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let weight_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data),
                vec![self.out_channels, self.col_rows],
                false,
            )?;
            let weight_2d_t = transpose(&weight_2d)?;

            let zero = <T as num_traits::Zero>::zero();
            let mut grad_cols = vec![zero; batch * self.col_rows * self.col_cols];

            for b in 0..batch {
                let go_start = b * self.out_channels * self.l_out;
                let go_end = go_start + self.out_channels * self.l_out;
                let go_b = Tensor::from_storage(
                    TensorStorage::cpu(go_data[go_start..go_end].to_vec()),
                    vec![self.out_channels, self.l_out],
                    false,
                )?;

                let gc_b = mm(&weight_2d_t, &go_b)?;
                let gc_data = gc_b.data()?;

                let gc_start = b * self.col_rows * self.col_cols;
                grad_cols[gc_start..gc_start + self.col_rows * self.col_cols]
                    .copy_from_slice(gc_data);
            }

            // col2im back to [B, C_in, 1, L], then reshape to [B, C_in, L]
            let gi = col2im(
                &grad_cols,
                batch,
                self.in_channels,
                1,
                length,
                1,
                k,
                1,
                s,
                0,
                p,
                1,
                self.l_out,
            );

            Some(Tensor::from_storage(
                TensorStorage::cpu(gi),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "Conv1dBackward"
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose2d
// ---------------------------------------------------------------------------

/// A 2-D transposed convolution (deconvolution) layer.
///
/// Applies a transposed spatial convolution over an input `[B, C_in, H, W]`.
/// Used for upsampling in generative models and decoder networks.
/// Equivalent to `torch.nn.ConvTranspose2d`.
///
/// # Implementation
///
/// The forward pass inserts `(stride - 1)` zeros between each input element
/// (fractionally-strided convolution), then applies a standard convolution
/// with the kernel flipped along both spatial axes.
///
/// # Shape
///
/// - Input: `[B, in_channels, H, W]`
/// - Output: `[B, out_channels, H_out, W_out]`
///
/// where `H_out = (H - 1) * stride.0 - 2 * padding.0 + kernel_size.0 + output_padding.0`.
#[derive(Debug)]
pub struct ConvTranspose2d<T: Float> {
    /// Learnable kernel weights `[in_channels, out_channels, kH, kW]`.
    ///
    /// Note: the channel ordering is transposed compared to `Conv2d`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Kernel spatial size `(kH, kW)`.
    kernel_size: (usize, usize),
    /// Stride `(sH, sW)`.
    stride: (usize, usize),
    /// Zero-padding `(pH, pW)` removed from both sides of the output.
    padding: (usize, usize),
    /// Additional size added to one side of the output `(opH, opW)`.
    output_padding: (usize, usize),
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> ConvTranspose2d<T> {
    /// Create a new `ConvTranspose2d` layer.
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized to zeros.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize),
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0 in both dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0 in both dimensions".into(),
            });
        }
        if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride".into(),
            });
        }

        // Weight shape: [in_channels, out_channels, kH, kW]
        let (kh, kw) = kernel_size;
        let mut weight = Parameter::zeros(&[in_channels, out_channels, kh, kw])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            zeros_init(&mut b)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.in_channels * self.out_channels * self.kernel_size.0 * self.kernel_size.1;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `ConvTranspose2d` from caller-supplied weight and optional bias.
    ///
    /// `weight` must have shape `[in_channels, out_channels, kH, kW]` (note the
    /// transposed channel ordering vs `Conv2d`). Used by
    /// `nn::functional::conv_transpose2d`.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: (usize, usize),
        padding: (usize, usize),
        output_padding: (usize, usize),
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 4 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose2d::from_parts: weight must be 4-D [in, out, kH, kW], got {:?}",
                    weight.shape()
                ),
            });
        }
        let in_channels = weight.shape()[0];
        let out_channels = weight.shape()[1];
        let kernel_size = (weight.shape()[2], weight.shape()[3]);
        if output_padding.0 >= stride.0 || output_padding.1 >= stride.1 {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride".into(),
            });
        }
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "ConvTranspose2d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }
}

/// Insert `(stride - 1)` zeros between each element along both spatial axes.
///
/// Given input `[B, C, H, W]`, produces `[B, C, H_up, W_up]` where
/// `H_up = (H - 1) * stride_h + 1` and `W_up = (W - 1) * stride_w + 1`.
fn stride_insert_zeros<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    h: usize,
    w: usize,
    stride_h: usize,
    stride_w: usize,
) -> (Vec<T>, usize, usize) {
    let h_up = (h - 1) * stride_h + 1;
    let w_up = (w - 1) * stride_w + 1;
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; batch * channels * h_up * w_up];

    for b in 0..batch {
        for c in 0..channels {
            for ih in 0..h {
                for iw in 0..w {
                    let oh = ih * stride_h;
                    let ow = iw * stride_w;
                    out[b * channels * h_up * w_up + c * h_up * w_up + oh * w_up + ow] =
                        input[b * channels * h * w + c * h * w + ih * w + iw];
                }
            }
        }
    }

    (out, h_up, w_up)
}

/// Flip a kernel along both spatial axes: `kernel[c_in, c_out, kh, kw]` ->
/// `kernel[c_out, c_in, kH-1-kh, kW-1-kw]` (also transposes channel dims).
fn flip_kernel<T: Float>(kernel: &[T], c_in: usize, c_out: usize, kh: usize, kw: usize) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut flipped = vec![zero; c_out * c_in * kh * kw];

    for ci in 0..c_in {
        for co in 0..c_out {
            for h in 0..kh {
                for w in 0..kw {
                    // Source: [c_in, c_out, h, w]
                    let src = ci * c_out * kh * kw + co * kh * kw + h * kw + w;
                    // Dest: [c_out, c_in, kH-1-h, kW-1-w]
                    let dst = co * c_in * kh * kw + ci * kh * kw + (kh - 1 - h) * kw + (kw - 1 - w);
                    flipped[dst] = kernel[src];
                }
            }
        }
    }

    flipped
}

impl<T: Float> Module<T> for ConvTranspose2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv_transpose2d.
        let _autocast_cat = autocast_guard("conv_transpose2d");

        // Validate input shape: [B, C_in, H, W].
        if input.ndim() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConvTranspose2d expects 4-D input [B, C, H, W], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let h = input.shape()[2];
        let w = input.shape()[3];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose2d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;
        let (oph, opw) = self.output_padding;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // Step 1: Insert zeros between input elements (stride insertion).
        let input_data = input.data_vec()?;
        let (upsampled, h_up, w_up) = stride_insert_zeros(&input_data, batch, c_in, h, w, sh, sw);

        // Step 2: Flip the kernel and transpose channel dimensions.
        // Weight: [in_channels, out_channels, kH, kW]
        // Flipped: [out_channels, in_channels, kH, kW] with spatial flip.
        let weight_data = self.weight.data_vec()?;
        let flipped = flip_kernel(&weight_data, self.in_channels, self.out_channels, kh, kw);

        // Step 3: Apply a regular convolution on the upsampled input using the
        // flipped kernel. The "padding" for this internal convolution is
        // `kernel_size - 1 - padding` to achieve the correct output size.
        let internal_pad_h = kh - 1 - ph;
        let internal_pad_w = kw - 1 - pw;

        // im2col on the upsampled input with stride=1.
        let (cols, col_rows, col_cols) = im2col(
            &upsampled,
            batch,
            c_in,
            h_up,
            w_up,
            kh,
            kw,
            1,
            1,
            internal_pad_h,
            internal_pad_w,
        );

        // h_out_base and w_out_base from the internal convolution.
        let h_out_base = (h_up + 2 * internal_pad_h - kh) + 1;
        let w_out_base = (w_up + 2 * internal_pad_w - kw) + 1;

        // The final output size includes output_padding.
        let h_out = h_out_base + oph;
        let w_out = w_out_base + opw;

        // Reshape flipped kernel to 2-D: [C_out, C_in * kH * kW]
        let flipped_2d = Tensor::from_storage(
            TensorStorage::cpu(flipped),
            vec![self.out_channels, col_rows],
            false,
        )?;

        // Per-batch matmul.
        let zero = <T as num_traits::Zero>::zero();
        let mut output = vec![zero; batch * self.out_channels * h_out * w_out];

        for b in 0..batch {
            let col_start = b * col_rows * col_cols;
            let col_end = col_start + col_rows * col_cols;
            let cols_b = Tensor::from_storage(
                TensorStorage::cpu(cols[col_start..col_end].to_vec()),
                vec![col_rows, col_cols],
                false,
            )?;

            let out_b = mm(&flipped_2d, &cols_b)?;
            let out_data = out_b.data()?;

            // Copy the base convolution result; extra output_padding rows/cols
            // remain zero (which is correct by definition).
            let out_start = b * self.out_channels * h_out * w_out;
            for c in 0..self.out_channels {
                for oh in 0..h_out_base {
                    for ow in 0..w_out_base {
                        output[out_start + c * h_out * w_out + oh * w_out + ow] =
                            out_data[c * h_out_base * w_out_base + oh * w_out_base + ow];
                    }
                }
            }
        }

        // Add bias if present.
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for hw in 0..(h_out * w_out) {
                        output[b * self.out_channels * h_out * w_out + c * h_out * w_out + hw] +=
                            bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, h_out, w_out],
            false,
        )?;

        // Attach backward if gradients are enabled.
        if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            let grad_fn = Arc::new(ConvTranspose2dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                _output_padding: self.output_padding,
                h_out,
                w_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose2dBackward
// ---------------------------------------------------------------------------

/// Backward function for `ConvTranspose2d` forward pass.
///
/// The backward of a transposed convolution is a regular convolution.
#[derive(Debug)]
struct ConvTranspose2dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize),
    stride: (usize, usize),
    padding: (usize, usize),
    _output_padding: (usize, usize),
    h_out: usize,
    w_out: usize,
}

impl<T: Float> GradFn<T> for ConvTranspose2dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, H_out, W_out]
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let h_in = self.input.shape()[2];
        let w_in = self.input.shape()[3];
        let (kh, kw) = self.kernel_size;
        let (sh, sw) = self.stride;
        let (ph, pw) = self.padding;

        // --- grad_input ---
        // The backward of ConvTranspose2d w.r.t. input is a regular Conv2d
        // of grad_output with the *original* (non-flipped) weight.
        // Weight is [in_channels, out_channels, kH, kW], we need it as
        // [in_channels, out_channels, kH, kW] reshaped to [in_channels, out_channels * kH * kW]
        // but actually we need a regular conv: grad_output [B, C_out, H_out, W_out]
        // convolved with weight^T [in_channels, out_channels, kH, kW] -> transposed to
        // [in_channels as filters over C_out channels].
        //
        // Reshape weight [C_in, C_out, kH, kW] -> [C_in, C_out * kH * kW] for matmul.
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let col_rows = self.out_channels * kh * kw;

            // Reshape weight to [C_in, C_out * kH * kW]
            let weight_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data),
                vec![self.in_channels, col_rows],
                false,
            )?;

            // im2col on grad_output with the conv parameters
            let (go_cols, _go_col_rows, go_col_cols) = im2col(
                &go_data,
                batch,
                self.out_channels,
                self.h_out,
                self.w_out,
                kh,
                kw,
                sh,
                sw,
                ph,
                pw,
            );

            let zero = <T as num_traits::Zero>::zero();
            let mut gi = vec![zero; batch * self.in_channels * h_in * w_in];

            for b in 0..batch {
                let col_start = b * col_rows * go_col_cols;
                let col_end = col_start + col_rows * go_col_cols;
                let go_cols_b = Tensor::from_storage(
                    TensorStorage::cpu(go_cols[col_start..col_end].to_vec()),
                    vec![col_rows, go_col_cols],
                    false,
                )?;

                let gi_b = mm(&weight_2d, &go_cols_b)?;
                let gi_data = gi_b.data()?;

                let out_start = b * self.in_channels * h_in * w_in;
                let copy_len = self.in_channels * h_in * w_in;
                gi[out_start..out_start + copy_len].copy_from_slice(&gi_data[..copy_len]);
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gi),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        // --- grad_weight ---
        // grad_weight[c_in, c_out, kh, kw] = sum_b input_b (x) grad_output_b
        // where (x) is the cross-correlation with stride.
        let grad_weight = if self.weight.requires_grad() {
            let zero = <T as num_traits::Zero>::zero();
            let weight_numel = self.in_channels * self.out_channels * kh * kw;
            let mut gw = vec![zero; weight_numel];
            let input_data = self.input.data_vec()?;

            for b in 0..batch {
                for ci in 0..self.in_channels {
                    for co in 0..self.out_channels {
                        for dh in 0..kh {
                            for dw in 0..kw {
                                let mut acc = zero;
                                for ih in 0..h_in {
                                    for iw in 0..w_in {
                                        let oh = ih * sh + dh;
                                        let ow = iw * sw + dw;
                                        // Account for padding removal
                                        if oh >= ph
                                            && ow >= pw
                                            && (oh - ph) < self.h_out
                                            && (ow - pw) < self.w_out
                                        {
                                            let go_idx =
                                                b * self.out_channels * self.h_out * self.w_out
                                                    + co * self.h_out * self.w_out
                                                    + (oh - ph) * self.w_out
                                                    + (ow - pw);
                                            let in_idx = b * self.in_channels * h_in * w_in
                                                + ci * h_in * w_in
                                                + ih * w_in
                                                + iw;
                                            acc += input_data[in_idx] * go_data[go_idx];
                                        }
                                    }
                                }
                                gw[ci * self.out_channels * kh * kw
                                    + co * kh * kw
                                    + dh * kw
                                    + dw] += acc;
                            }
                        }
                    }
                }
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gw),
                vec![self.in_channels, self.out_channels, kh, kw],
                false,
            )?)
        } else {
            None
        };

        // --- grad_bias ---
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let zero = <T as num_traits::Zero>::zero();
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for hw in 0..(self.h_out * self.w_out) {
                            gb[c] +=
                                go_data[batch_idx * self.out_channels * self.h_out * self.w_out
                                    + c * self.h_out * self.w_out
                                    + hw];
                        }
                    }
                }
                Some(Tensor::from_storage(
                    TensorStorage::cpu(gb),
                    vec![self.out_channels],
                    false,
                )?)
            }
            _ => None,
        };

        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "ConvTranspose2dBackward"
    }
}

// ---------------------------------------------------------------------------
// im2col_3d / col2im_3d helpers
// ---------------------------------------------------------------------------

/// Extract volume patches into columns for 3-D convolution.
///
/// Given a 5-D input `[B, C, D, H, W]`, produces a 3-D output
/// `[B, C * kD * kH * kW, D_out * H_out * W_out]` where each column is one
/// flattened receptive-field patch.
// Internal kernel: argument set mirrors the 3-D convolution descriptor
// (B, C, D, H, W, kD, kH, kW, ...); the 3-D extension of `im2col` carries
// proportionally more arguments than the 2-D version.
#[allow(clippy::too_many_arguments)]
fn im2col_3d<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
) -> (Vec<T>, usize, usize) {
    let d_out = (depth + 2 * pad_d - kernel_d) / stride_d + 1;
    let h_out = (height + 2 * pad_h - kernel_h) / stride_h + 1;
    let w_out = (width + 2 * pad_w - kernel_w) / stride_w + 1;
    let col_rows = channels * kernel_d * kernel_h * kernel_w;
    let col_cols = d_out * h_out * w_out;

    let zero = <T as num_traits::Zero>::zero();
    let mut cols = vec![zero; batch * col_rows * col_cols];

    for b in 0..batch {
        for c in 0..channels {
            for kd in 0..kernel_d {
                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {
                        let row = c * kernel_d * kernel_h * kernel_w
                            + kd * kernel_h * kernel_w
                            + kh * kernel_w
                            + kw;
                        for od in 0..d_out {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let id = od * stride_d + kd;
                                    let ih = oh * stride_h + kh;
                                    let iw = ow * stride_w + kw;
                                    let col = od * h_out * w_out + oh * w_out + ow;

                                    let val = if id >= pad_d
                                        && ih >= pad_h
                                        && iw >= pad_w
                                        && (id - pad_d) < depth
                                        && (ih - pad_h) < height
                                        && (iw - pad_w) < width
                                    {
                                        let real_d = id - pad_d;
                                        let real_h = ih - pad_h;
                                        let real_w = iw - pad_w;
                                        input[b * channels * depth * height * width
                                            + c * depth * height * width
                                            + real_d * height * width
                                            + real_h * width
                                            + real_w]
                                    } else {
                                        zero
                                    };

                                    cols[b * col_rows * col_cols + row * col_cols + col] = val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (cols, col_rows, col_cols)
}

/// Scatter columns back into a volume tensor (adjoint of `im2col_3d`).
///
/// Given columns of shape `[B, C * kD * kH * kW, D_out * H_out * W_out]`,
/// accumulates values back into a `[B, C, D, H, W]` tensor (with padding
/// stripped).
// Internal kernel: adjoint of `im2col_3d`; same descriptor signature.
#[allow(clippy::too_many_arguments)]
fn col2im_3d<T: Float>(
    cols: &[T],
    batch: usize,
    channels: usize,
    depth: usize,
    height: usize,
    width: usize,
    kernel_d: usize,
    kernel_h: usize,
    kernel_w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
    pad_d: usize,
    pad_h: usize,
    pad_w: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut output = vec![zero; batch * channels * depth * height * width];

    let col_rows = channels * kernel_d * kernel_h * kernel_w;
    let col_cols = d_out * h_out * w_out;

    for b in 0..batch {
        for c in 0..channels {
            for kd in 0..kernel_d {
                for kh in 0..kernel_h {
                    for kw in 0..kernel_w {
                        let row = c * kernel_d * kernel_h * kernel_w
                            + kd * kernel_h * kernel_w
                            + kh * kernel_w
                            + kw;
                        for od in 0..d_out {
                            for oh in 0..h_out {
                                for ow in 0..w_out {
                                    let id = od * stride_d + kd;
                                    let ih = oh * stride_h + kh;
                                    let iw = ow * stride_w + kw;
                                    let col = od * h_out * w_out + oh * w_out + ow;

                                    if id >= pad_d
                                        && ih >= pad_h
                                        && iw >= pad_w
                                        && (id - pad_d) < depth
                                        && (ih - pad_h) < height
                                        && (iw - pad_w) < width
                                    {
                                        let real_d = id - pad_d;
                                        let real_h = ih - pad_h;
                                        let real_w = iw - pad_w;
                                        output[b * channels * depth * height * width
                                            + c * depth * height * width
                                            + real_d * height * width
                                            + real_h * width
                                            + real_w] +=
                                            cols[b * col_rows * col_cols + row * col_cols + col];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    output
}

// ---------------------------------------------------------------------------
// Conv3d
// ---------------------------------------------------------------------------

/// A 3-D convolution layer for volumetric data.
///
/// Applies a spatial convolution over an input `[B, C_in, D, H, W]` using
/// the im2col + matmul algorithm. Equivalent to `torch.nn.Conv3d`.
///
/// # Shape
///
/// - Input: `[B, in_channels, D, H, W]`
/// - Output: `[B, out_channels, D_out, H_out, W_out]`
///
/// where `D_out = (D + 2 * padding.0 - kernel_size.0) / stride.0 + 1` (and
/// analogously for H and W).
#[derive(Debug)]
pub struct Conv3d<T: Float> {
    /// Learnable kernel weights `[out_channels, in_channels, kD, kH, kW]`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels (filters).
    out_channels: usize,
    /// Kernel spatial size `(kD, kH, kW)`.
    kernel_size: (usize, usize, usize),
    /// Stride `(sD, sH, sW)`.
    stride: (usize, usize, usize),
    /// Zero-padding `(pD, pH, pW)` applied to both sides.
    padding: (usize, usize, usize),
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> Conv3d<T> {
    /// Create a new `Conv3d` layer.
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized to zeros.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 || kernel_size.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0 in all dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0 in all dimensions".into(),
            });
        }

        let (kd, kh, kw) = kernel_size;
        let mut weight = Parameter::zeros(&[out_channels, in_channels, kd, kh, kw])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            zeros_init(&mut b)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        })
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.out_channels
            * self.in_channels
            * self.kernel_size.0
            * self.kernel_size.1
            * self.kernel_size.2;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `Conv3d` from caller-supplied weight and optional bias tensors.
    ///
    /// `weight` must have shape `[out_channels, in_channels, kD, kH, kW]`.
    /// Used by `nn::functional::conv3d`.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 5 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv3d::from_parts: weight must be 5-D [out, in, kD, kH, kW], got {:?}",
                    weight.shape()
                ),
            });
        }
        let out_channels = weight.shape()[0];
        let in_channels = weight.shape()[1];
        let kernel_size = (weight.shape()[2], weight.shape()[3], weight.shape()[4]);
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "Conv3d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Conv3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv3d.
        let _autocast_cat = autocast_guard("conv3d");

        // Validate input shape: [B, C_in, D, H, W].
        if input.ndim() != 5 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv3d expects 5-D input [B, C, D, H, W], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let d = input.shape()[2];
        let h = input.shape()[3];
        let w = input.shape()[4];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "Conv3d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;

        // Check that the kernel fits.
        let d_padded = d + 2 * pd;
        let h_padded = h + 2 * ph;
        let w_padded = w + 2 * pw;
        if d_padded < kd || h_padded < kh || w_padded < kw {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "Conv3d: padded input ({d_padded}, {h_padded}, {w_padded}) is smaller than kernel ({kd}, {kh}, {kw})"
                ),
            });
        }

        let d_out = (d_padded - kd) / sd + 1;
        let h_out = (h_padded - kh) / sh + 1;
        let w_out = (w_padded - kw) / sw + 1;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // ---- CPU path ----
        let input_data = input.data_vec()?;

        // im2col_3d: [B, C_in * kD * kH * kW, D_out * H_out * W_out]
        let (cols, col_rows, col_cols) = im2col_3d(
            &input_data,
            batch,
            c_in,
            d,
            h,
            w,
            kd,
            kh,
            kw,
            sd,
            sh,
            sw,
            pd,
            ph,
            pw,
        );

        // Reshape weight to 2D: [C_out, C_in * kD * kH * kW]
        let weight_data = self.weight.data_vec()?;
        let weight_2d = Tensor::from_storage(
            TensorStorage::cpu(weight_data),
            vec![self.out_channels, col_rows],
            false,
        )?;

        // Per-batch matmul: weight_2d @ cols_b -> [C_out, D_out * H_out * W_out]
        let zero = <T as num_traits::Zero>::zero();
        let spatial_out = d_out * h_out * w_out;
        let mut output = vec![zero; batch * self.out_channels * spatial_out];

        for b in 0..batch {
            let col_start = b * col_rows * col_cols;
            let col_end = col_start + col_rows * col_cols;
            let cols_b = Tensor::from_storage(
                TensorStorage::cpu(cols[col_start..col_end].to_vec()),
                vec![col_rows, col_cols],
                false,
            )?;

            let out_b = mm(&weight_2d, &cols_b)?;
            let out_data = out_b.data()?;
            let out_start = b * self.out_channels * spatial_out;
            output[out_start..out_start + self.out_channels * spatial_out]
                .copy_from_slice(out_data);
        }

        // Add bias if present: broadcast [C_out] over [B, C_out, D_out, H_out, W_out].
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for s in 0..spatial_out {
                        output[b * self.out_channels * spatial_out + c * spatial_out + s] += bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, d_out, h_out, w_out],
            false,
        )?;

        // Attach backward if gradients are enabled and any input/param requires grad.
        if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            let grad_fn = Arc::new(Conv3dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                cols,
                col_rows,
                col_cols,
                d_out,
                h_out,
                w_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Conv3dBackward
// ---------------------------------------------------------------------------

/// Backward function for `Conv3d` forward pass.
///
/// Saved tensors:
/// - `input`: the original 5-D input
/// - `weight`: the 5-D kernel
/// - `bias`: optional 1-D bias
/// - `cols`: the im2col_3d columns from the forward pass (avoids recomputation)
#[derive(Debug)]
struct Conv3dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    cols: Vec<T>,
    col_rows: usize,
    col_cols: usize,
    d_out: usize,
    h_out: usize,
    w_out: usize,
}

impl<T: Float> GradFn<T> for Conv3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, D_out, H_out, W_out]
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let d = self.input.shape()[2];
        let h = self.input.shape()[3];
        let w = self.input.shape()[4];
        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let spatial_out = self.d_out * self.h_out * self.w_out;

        // --- grad_weight ---
        // For each batch element:
        //   grad_output_b: [C_out, D_out * H_out * W_out]
        //   cols_b:        [col_rows, col_cols]
        //   grad_weight += grad_output_b @ cols_b^T
        let grad_weight = if self.weight.requires_grad() {
            let zero = <T as num_traits::Zero>::zero();
            let weight_numel = self.out_channels * self.col_rows;
            let mut gw_accum = vec![zero; weight_numel];

            for b in 0..batch {
                let go_start = b * self.out_channels * spatial_out;
                let go_end = go_start + self.out_channels * spatial_out;
                let go_b = Tensor::from_storage(
                    TensorStorage::cpu(go_data[go_start..go_end].to_vec()),
                    vec![self.out_channels, spatial_out],
                    false,
                )?;

                let col_start = b * self.col_rows * self.col_cols;
                let col_end = col_start + self.col_rows * self.col_cols;
                let cols_b = Tensor::from_storage(
                    TensorStorage::cpu(self.cols[col_start..col_end].to_vec()),
                    vec![self.col_rows, self.col_cols],
                    false,
                )?;

                let cols_bt = transpose(&cols_b)?;
                let gw_b = mm(&go_b, &cols_bt)?;
                let gw_data = gw_b.data()?;

                for i in 0..weight_numel {
                    gw_accum[i] += gw_data[i];
                }
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gw_accum),
                vec![self.out_channels, self.in_channels, kd, kh, kw],
                false,
            )?)
        } else {
            None
        };

        // --- grad_bias ---
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let zero = <T as num_traits::Zero>::zero();
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for s in 0..spatial_out {
                            gb[c] += go_data
                                [batch_idx * self.out_channels * spatial_out + c * spatial_out + s];
                        }
                    }
                }
                Some(Tensor::from_storage(
                    TensorStorage::cpu(gb),
                    vec![self.out_channels],
                    false,
                )?)
            }
            _ => None,
        };

        // --- grad_input ---
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let weight_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data),
                vec![self.out_channels, self.col_rows],
                false,
            )?;
            let weight_2d_t = transpose(&weight_2d)?;

            let zero = <T as num_traits::Zero>::zero();
            let mut grad_cols = vec![zero; batch * self.col_rows * self.col_cols];

            for b in 0..batch {
                let go_start = b * self.out_channels * spatial_out;
                let go_end = go_start + self.out_channels * spatial_out;
                let go_b = Tensor::from_storage(
                    TensorStorage::cpu(go_data[go_start..go_end].to_vec()),
                    vec![self.out_channels, spatial_out],
                    false,
                )?;

                let gc_b = mm(&weight_2d_t, &go_b)?;
                let gc_data = gc_b.data()?;

                let gc_start = b * self.col_rows * self.col_cols;
                grad_cols[gc_start..gc_start + self.col_rows * self.col_cols]
                    .copy_from_slice(gc_data);
            }

            // col2im_3d to scatter back to [B, C_in, D, H, W]
            let gi = col2im_3d(
                &grad_cols,
                batch,
                self.in_channels,
                d,
                h,
                w,
                kd,
                kh,
                kw,
                sd,
                sh,
                sw,
                pd,
                ph,
                pw,
                self.d_out,
                self.h_out,
                self.w_out,
            );

            Some(Tensor::from_storage(
                TensorStorage::cpu(gi),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        // Return exactly as many gradients as inputs() returns.
        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "Conv3dBackward"
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose1d
// ---------------------------------------------------------------------------

/// A 1-D transposed convolution (deconvolution) layer.
///
/// Applies a transposed temporal convolution over an input `[B, C_in, L]`.
/// Used for upsampling in generative models and decoder networks.
/// Equivalent to `torch.nn.ConvTranspose1d`.
///
/// # Implementation
///
/// Delegates to the 2-D transposed convolution by adding a dummy spatial
/// dimension (H=1), then squeezes the output back to 3-D.
///
/// # Shape
///
/// - Input: `[B, in_channels, L]`
/// - Output: `[B, out_channels, L_out]`
///
/// where `L_out = (L - 1) * stride - 2 * padding + kernel_size + output_padding`.
#[derive(Debug)]
pub struct ConvTranspose1d<T: Float> {
    /// Learnable kernel weights `[in_channels, out_channels, kernel_size]`.
    ///
    /// Note: the channel ordering is transposed compared to `Conv1d`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Kernel length.
    kernel_size: usize,
    /// Stride.
    stride: usize,
    /// Zero-padding removed from both sides of the output.
    padding: usize,
    /// Additional size added to one side of the output.
    output_padding: usize,
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> ConvTranspose1d<T> {
    /// Create a new `ConvTranspose1d` layer.
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized to zeros.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
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
        if output_padding >= stride {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride".into(),
            });
        }

        // Weight shape: [in_channels, out_channels, kernel_size]
        let mut weight = Parameter::zeros(&[in_channels, out_channels, kernel_size])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            zeros_init(&mut b)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.in_channels * self.out_channels * self.kernel_size;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `ConvTranspose1d` from caller-supplied weight and optional bias.
    ///
    /// `weight` must have shape `[in_channels, out_channels, kernel_size]`
    /// (transposed channel ordering vs `Conv1d`). Used by
    /// `nn::functional::conv_transpose1d`.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 3 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose1d::from_parts: weight must be 3-D [in, out, k], got {:?}",
                    weight.shape()
                ),
            });
        }
        let in_channels = weight.shape()[0];
        let out_channels = weight.shape()[1];
        let kernel_size = weight.shape()[2];
        if output_padding >= stride {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride".into(),
            });
        }
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "ConvTranspose1d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for ConvTranspose1d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv_transpose1d.
        let _autocast_cat = autocast_guard("conv_transpose1d");

        // Validate input shape: [B, C_in, L].
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConvTranspose1d expects 3-D input [B, C, L], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let length = input.shape()[2];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose1d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;
        let op = self.output_padding;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // Step 1: Insert zeros between input elements (stride insertion).
        // Treat [B, C, L] as [B, C, 1, L] for the 2-D helper.
        let input_data = input.data_vec()?;
        let (upsampled, _h_up, w_up) =
            stride_insert_zeros(&input_data, batch, c_in, 1, length, 1, s);

        // Step 2: Flip the kernel and transpose channel dimensions.
        // Weight: [in_channels, out_channels, k] -> treat as [in_channels, out_channels, 1, k]
        let weight_data = self.weight.data_vec()?;
        let flipped = flip_kernel(&weight_data, self.in_channels, self.out_channels, 1, k);

        // Step 3: Apply a regular convolution on the upsampled input using the
        // flipped kernel with internal padding.
        let internal_pad_w = k - 1 - p;

        // im2col on the upsampled input [B, C, 1, w_up] with kernel (1, k), stride (1, 1).
        let (cols, col_rows, col_cols) = im2col(
            &upsampled,
            batch,
            c_in,
            1,
            w_up,
            1,
            k,
            1,
            1,
            0,
            internal_pad_w,
        );

        // w_out_base from the internal convolution.
        let w_out_base = (w_up + 2 * internal_pad_w - k) + 1;

        // The final output size includes output_padding.
        let l_out = w_out_base + op;

        // Reshape flipped kernel to 2-D: [C_out, C_in * 1 * k]
        let flipped_2d = Tensor::from_storage(
            TensorStorage::cpu(flipped),
            vec![self.out_channels, col_rows],
            false,
        )?;

        // Per-batch matmul.
        let zero = <T as num_traits::Zero>::zero();
        let mut output = vec![zero; batch * self.out_channels * l_out];

        for b in 0..batch {
            let col_start = b * col_rows * col_cols;
            let col_end = col_start + col_rows * col_cols;
            let cols_b = Tensor::from_storage(
                TensorStorage::cpu(cols[col_start..col_end].to_vec()),
                vec![col_rows, col_cols],
                false,
            )?;

            let out_b = mm(&flipped_2d, &cols_b)?;
            let out_data = out_b.data()?;

            // Copy the base convolution result; extra output_padding positions
            // remain zero (which is correct by definition).
            let out_start = b * self.out_channels * l_out;
            for c in 0..self.out_channels {
                for ow in 0..w_out_base {
                    output[out_start + c * l_out + ow] = out_data[c * w_out_base + ow];
                }
            }
        }

        // Add bias if present.
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for l in 0..l_out {
                        output[b * self.out_channels * l_out + c * l_out + l] += bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, l_out],
            false,
        )?;

        // Attach backward if gradients are enabled.
        if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            let grad_fn = Arc::new(ConvTranspose1dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                _output_padding: self.output_padding,
                l_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose1dBackward
// ---------------------------------------------------------------------------

/// Backward function for `ConvTranspose1d` forward pass.
///
/// The backward of a transposed convolution is a regular convolution.
#[derive(Debug)]
struct ConvTranspose1dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    padding: usize,
    _output_padding: usize,
    l_out: usize,
}

impl<T: Float> GradFn<T> for ConvTranspose1dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, L_out]
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let l_in = self.input.shape()[2];
        let k = self.kernel_size;
        let s = self.stride;
        let p = self.padding;

        // --- grad_input ---
        // The backward of ConvTranspose1d w.r.t. input is a regular Conv1d
        // of grad_output with the original (non-flipped) weight.
        // Weight is [C_in, C_out, k], treat as [C_in, C_out, 1, k].
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let col_rows = self.out_channels * k;

            // Reshape weight to [C_in, C_out * k]
            let weight_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data),
                vec![self.in_channels, col_rows],
                false,
            )?;

            // im2col on grad_output [B, C_out, L_out] treated as [B, C_out, 1, L_out]
            let (go_cols, _go_col_rows, go_col_cols) = im2col(
                &go_data,
                batch,
                self.out_channels,
                1,
                self.l_out,
                1,
                k,
                1,
                s,
                0,
                p,
            );

            let zero = <T as num_traits::Zero>::zero();
            let mut gi = vec![zero; batch * self.in_channels * l_in];

            for b in 0..batch {
                let col_start = b * col_rows * go_col_cols;
                let col_end = col_start + col_rows * go_col_cols;
                let go_cols_b = Tensor::from_storage(
                    TensorStorage::cpu(go_cols[col_start..col_end].to_vec()),
                    vec![col_rows, go_col_cols],
                    false,
                )?;

                let gi_b = mm(&weight_2d, &go_cols_b)?;
                let gi_data = gi_b.data()?;

                let out_start = b * self.in_channels * l_in;
                let copy_len = self.in_channels * l_in;
                gi[out_start..out_start + copy_len].copy_from_slice(&gi_data[..copy_len]);
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gi),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        // --- grad_weight ---
        // grad_weight[c_in, c_out, kw] = sum_b input_b cross-correlated with grad_output_b
        let grad_weight = if self.weight.requires_grad() {
            let zero = <T as num_traits::Zero>::zero();
            let weight_numel = self.in_channels * self.out_channels * k;
            let mut gw = vec![zero; weight_numel];
            let input_data = self.input.data_vec()?;

            for b in 0..batch {
                for ci in 0..self.in_channels {
                    for co in 0..self.out_channels {
                        for dw in 0..k {
                            let mut acc = zero;
                            for il in 0..l_in {
                                let ow = il * s + dw;
                                if ow >= p && (ow - p) < self.l_out {
                                    let go_idx = b * self.out_channels * self.l_out
                                        + co * self.l_out
                                        + (ow - p);
                                    let in_idx = b * self.in_channels * l_in + ci * l_in + il;
                                    acc += input_data[in_idx] * go_data[go_idx];
                                }
                            }
                            gw[ci * self.out_channels * k + co * k + dw] += acc;
                        }
                    }
                }
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gw),
                vec![self.in_channels, self.out_channels, k],
                false,
            )?)
        } else {
            None
        };

        // --- grad_bias ---
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let zero = <T as num_traits::Zero>::zero();
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for l in 0..self.l_out {
                            gb[c] += go_data
                                [batch_idx * self.out_channels * self.l_out + c * self.l_out + l];
                        }
                    }
                }
                Some(Tensor::from_storage(
                    TensorStorage::cpu(gb),
                    vec![self.out_channels],
                    false,
                )?)
            }
            _ => None,
        };

        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "ConvTranspose1dBackward"
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose3d
// ---------------------------------------------------------------------------

/// A 3-D transposed convolution (deconvolution) layer.
///
/// Applies a transposed volumetric convolution over an input `[B, C_in, D, H, W]`.
/// Used for upsampling in generative models and 3-D decoder networks.
/// Equivalent to `torch.nn.ConvTranspose3d`.
///
/// # Implementation
///
/// The forward pass inserts `(stride - 1)` zeros between each input element
/// along all three spatial axes (fractionally-strided convolution), then applies
/// a standard 3-D convolution with the kernel flipped along all spatial axes.
///
/// # Shape
///
/// - Input: `[B, in_channels, D, H, W]`
/// - Output: `[B, out_channels, D_out, H_out, W_out]`
///
/// where `D_out = (D - 1) * stride.0 - 2 * padding.0 + kernel_size.0 + output_padding.0`
/// (and analogously for H and W).
#[derive(Debug)]
pub struct ConvTranspose3d<T: Float> {
    /// Learnable kernel weights `[in_channels, out_channels, kD, kH, kW]`.
    ///
    /// Note: the channel ordering is transposed compared to `Conv3d`.
    weight: Parameter<T>,
    /// Optional learnable bias `[out_channels]`.
    bias: Option<Parameter<T>>,
    /// Number of input channels.
    in_channels: usize,
    /// Number of output channels.
    out_channels: usize,
    /// Kernel spatial size `(kD, kH, kW)`.
    kernel_size: (usize, usize, usize),
    /// Stride `(sD, sH, sW)`.
    stride: (usize, usize, usize),
    /// Zero-padding `(pD, pH, pW)` removed from both sides of the output.
    padding: (usize, usize, usize),
    /// Additional size added to one side of the output `(opD, opH, opW)`.
    output_padding: (usize, usize, usize),
    /// Whether the module is in training mode.
    training: bool,
}

impl<T: Float> ConvTranspose3d<T> {
    /// Create a new `ConvTranspose3d` layer.
    ///
    /// Weight is initialized with Kaiming uniform (ReLU gain).
    /// Bias, if enabled, is initialized to zeros.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: (usize, usize, usize),
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
        bias: bool,
    ) -> FerrotorchResult<Self> {
        if in_channels == 0 || out_channels == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "in_channels and out_channels must be > 0".into(),
            });
        }
        if kernel_size.0 == 0 || kernel_size.1 == 0 || kernel_size.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "kernel_size must be > 0 in all dimensions".into(),
            });
        }
        if stride.0 == 0 || stride.1 == 0 || stride.2 == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "stride must be > 0 in all dimensions".into(),
            });
        }
        if output_padding.0 >= stride.0
            || output_padding.1 >= stride.1
            || output_padding.2 >= stride.2
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride in all dimensions"
                    .into(),
            });
        }

        // Weight shape: [in_channels, out_channels, kD, kH, kW]
        let (kd, kh, kw) = kernel_size;
        let mut weight = Parameter::zeros(&[in_channels, out_channels, kd, kh, kw])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;

        let bias_param = if bias {
            let mut b = Parameter::zeros(&[out_channels])?;
            zeros_init(&mut b)?;
            Some(b)
        } else {
            None
        };

        Ok(Self {
            weight,
            bias: bias_param,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }

    /// The number of learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        let w = self.in_channels
            * self.out_channels
            * self.kernel_size.0
            * self.kernel_size.1
            * self.kernel_size.2;
        let b = if self.bias.is_some() {
            self.out_channels
        } else {
            0
        };
        w + b
    }

    /// Build a `ConvTranspose3d` from caller-supplied weight and optional bias.
    ///
    /// `weight` must have shape `[in_channels, out_channels, kD, kH, kW]`
    /// (transposed channel ordering vs `Conv3d`). Used by
    /// `nn::functional::conv_transpose3d`.
    pub fn from_parts(
        weight: Tensor<T>,
        bias: Option<Tensor<T>>,
        stride: (usize, usize, usize),
        padding: (usize, usize, usize),
        output_padding: (usize, usize, usize),
    ) -> FerrotorchResult<Self> {
        if weight.ndim() != 5 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose3d::from_parts: weight must be 5-D [in, out, kD, kH, kW], got {:?}",
                    weight.shape()
                ),
            });
        }
        let in_channels = weight.shape()[0];
        let out_channels = weight.shape()[1];
        let kernel_size = (weight.shape()[2], weight.shape()[3], weight.shape()[4]);
        if output_padding.0 >= stride.0
            || output_padding.1 >= stride.1
            || output_padding.2 >= stride.2
        {
            return Err(FerrotorchError::InvalidArgument {
                message: "output_padding must be strictly less than stride in all dimensions"
                    .into(),
            });
        }
        if let Some(b) = &bias {
            if b.ndim() != 1 || b.shape()[0] != out_channels {
                return Err(FerrotorchError::ShapeMismatch {
                    message: format!(
                        "ConvTranspose3d::from_parts: bias shape {:?} != [{}]",
                        b.shape(),
                        out_channels
                    ),
                });
            }
        }
        Ok(Self {
            weight: Parameter::new(weight),
            bias: bias.map(Parameter::new),
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            training: true,
        })
    }
}

/// Insert `(stride - 1)` zeros between each element along three spatial axes.
///
/// Given input `[B, C, D, H, W]`, produces `[B, C, D_up, H_up, W_up]` where
/// `D_up = (D - 1) * stride_d + 1` (and analogously for H, W).
// Internal kernel for ConvTranspose3d backward: arguments are the 3-D
// shape descriptor + per-axis stride; refactoring to a config struct would
// add allocation in a hot path.
#[allow(clippy::too_many_arguments)]
fn stride_insert_zeros_3d<T: Float>(
    input: &[T],
    batch: usize,
    channels: usize,
    d: usize,
    h: usize,
    w: usize,
    stride_d: usize,
    stride_h: usize,
    stride_w: usize,
) -> (Vec<T>, usize, usize, usize) {
    let d_up = (d - 1) * stride_d + 1;
    let h_up = (h - 1) * stride_h + 1;
    let w_up = (w - 1) * stride_w + 1;
    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; batch * channels * d_up * h_up * w_up];

    for b in 0..batch {
        for c in 0..channels {
            for id in 0..d {
                for ih in 0..h {
                    for iw in 0..w {
                        let od = id * stride_d;
                        let oh = ih * stride_h;
                        let ow = iw * stride_w;
                        out[b * channels * d_up * h_up * w_up
                            + c * d_up * h_up * w_up
                            + od * h_up * w_up
                            + oh * w_up
                            + ow] = input
                            [b * channels * d * h * w + c * d * h * w + id * h * w + ih * w + iw];
                    }
                }
            }
        }
    }

    (out, d_up, h_up, w_up)
}

/// Flip a 3-D kernel along all spatial axes and transpose channel dimensions:
/// `kernel[c_in, c_out, kD, kH, kW]` ->
/// `kernel[c_out, c_in, kD-1-kd, kH-1-kh, kW-1-kw]`.
fn flip_kernel_3d<T: Float>(
    kernel: &[T],
    c_in: usize,
    c_out: usize,
    kd: usize,
    kh: usize,
    kw: usize,
) -> Vec<T> {
    let zero = <T as num_traits::Zero>::zero();
    let mut flipped = vec![zero; c_out * c_in * kd * kh * kw];

    for ci in 0..c_in {
        for co in 0..c_out {
            for dd in 0..kd {
                for dh in 0..kh {
                    for dw in 0..kw {
                        // Source: [c_in, c_out, dd, dh, dw]
                        let src = ci * c_out * kd * kh * kw
                            + co * kd * kh * kw
                            + dd * kh * kw
                            + dh * kw
                            + dw;
                        // Dest: [c_out, c_in, kD-1-dd, kH-1-dh, kW-1-dw]
                        let dst = co * c_in * kd * kh * kw
                            + ci * kd * kh * kw
                            + (kd - 1 - dd) * kh * kw
                            + (kh - 1 - dh) * kw
                            + (kw - 1 - dw);
                        flipped[dst] = kernel[src];
                    }
                }
            }
        }
    }

    flipped
}

impl<T: Float> Module<T> for ConvTranspose3d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Record autocast decision for conv_transpose3d.
        let _autocast_cat = autocast_guard("conv_transpose3d");

        // Validate input shape: [B, C_in, D, H, W].
        if input.ndim() != 5 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConvTranspose3d expects 5-D input [B, C, D, H, W], got {:?}",
                    input.shape()
                ),
            });
        }

        let batch = input.shape()[0];
        let c_in = input.shape()[1];
        let d = input.shape()[2];
        let h = input.shape()[3];
        let w = input.shape()[4];

        if c_in != self.in_channels {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "ConvTranspose3d: expected {} input channels, got {}",
                    self.in_channels, c_in
                ),
            });
        }

        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let (opd, oph, opw) = self.output_padding;

        // Save the input device so we can restore it on the output.
        let input_device = input.device();

        // Step 1: Insert zeros between input elements (stride insertion).
        let input_data = input.data_vec()?;
        let (upsampled, d_up, h_up, w_up) =
            stride_insert_zeros_3d(&input_data, batch, c_in, d, h, w, sd, sh, sw);

        // Step 2: Flip the kernel and transpose channel dimensions.
        let weight_data = self.weight.data_vec()?;
        let flipped = flip_kernel_3d(
            &weight_data,
            self.in_channels,
            self.out_channels,
            kd,
            kh,
            kw,
        );

        // Step 3: Apply a regular 3-D convolution on the upsampled input using the
        // flipped kernel. The "padding" for this internal convolution is
        // `kernel_size - 1 - padding` to achieve the correct output size.
        let internal_pad_d = kd - 1 - pd;
        let internal_pad_h = kh - 1 - ph;
        let internal_pad_w = kw - 1 - pw;

        // im2col_3d on the upsampled input with stride=1.
        let (cols, col_rows, col_cols) = im2col_3d(
            &upsampled,
            batch,
            c_in,
            d_up,
            h_up,
            w_up,
            kd,
            kh,
            kw,
            1,
            1,
            1,
            internal_pad_d,
            internal_pad_h,
            internal_pad_w,
        );

        // Base output sizes from the internal convolution.
        let d_out_base = (d_up + 2 * internal_pad_d - kd) + 1;
        let h_out_base = (h_up + 2 * internal_pad_h - kh) + 1;
        let w_out_base = (w_up + 2 * internal_pad_w - kw) + 1;

        // The final output size includes output_padding.
        let d_out = d_out_base + opd;
        let h_out = h_out_base + oph;
        let w_out = w_out_base + opw;

        // Reshape flipped kernel to 2-D: [C_out, C_in * kD * kH * kW]
        let flipped_2d = Tensor::from_storage(
            TensorStorage::cpu(flipped),
            vec![self.out_channels, col_rows],
            false,
        )?;

        // Per-batch matmul.
        let zero = <T as num_traits::Zero>::zero();
        let spatial_out = d_out * h_out * w_out;
        let spatial_base = d_out_base * h_out_base * w_out_base;
        let mut output = vec![zero; batch * self.out_channels * spatial_out];

        for b in 0..batch {
            let col_start = b * col_rows * col_cols;
            let col_end = col_start + col_rows * col_cols;
            let cols_b = Tensor::from_storage(
                TensorStorage::cpu(cols[col_start..col_end].to_vec()),
                vec![col_rows, col_cols],
                false,
            )?;

            let out_b = mm(&flipped_2d, &cols_b)?;
            let out_data = out_b.data()?;

            // Copy the base convolution result; extra output_padding positions
            // remain zero (which is correct by definition).
            let out_start = b * self.out_channels * spatial_out;
            for c in 0..self.out_channels {
                for od in 0..d_out_base {
                    for oh in 0..h_out_base {
                        for ow in 0..w_out_base {
                            output[out_start
                                + c * spatial_out
                                + od * h_out * w_out
                                + oh * w_out
                                + ow] = out_data[c * spatial_base
                                + od * h_out_base * w_out_base
                                + oh * w_out_base
                                + ow];
                        }
                    }
                }
            }
        }

        // Add bias if present.
        if let Some(ref bias) = self.bias {
            let bias_data = bias.data_vec()?;
            for b in 0..batch {
                for c in 0..self.out_channels {
                    let bval = bias_data[c];
                    for s in 0..spatial_out {
                        output[b * self.out_channels * spatial_out + c * spatial_out + s] += bval;
                    }
                }
            }
        }

        let result = Tensor::from_storage(
            TensorStorage::cpu(output),
            vec![batch, self.out_channels, d_out, h_out, w_out],
            false,
        )?;

        // Attach backward if gradients are enabled.
        if is_grad_enabled()
            && (input.requires_grad()
                || self.weight.requires_grad()
                || self.bias.as_ref().is_some_and(|b| b.requires_grad()))
        {
            let grad_fn = Arc::new(ConvTranspose3dBackward {
                input: input.clone(),
                weight: self.weight.tensor().clone(),
                bias: self.bias.as_ref().map(|b| b.tensor().clone()),
                in_channels: self.in_channels,
                out_channels: self.out_channels,
                kernel_size: self.kernel_size,
                stride: self.stride,
                padding: self.padding,
                _output_padding: self.output_padding,
                d_out,
                h_out,
                w_out,
            });
            Tensor::from_operation(
                TensorStorage::cpu(result.data()?.to_vec()),
                result.shape().to_vec(),
                grad_fn,
            )?
            .to(input_device) // restore device
        } else {
            result.to(input_device)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = vec![&self.weight];
        if let Some(ref b) = self.bias {
            params.push(b);
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = vec![&mut self.weight];
        if let Some(ref mut b) = self.bias {
            params.push(b);
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = vec![("weight".to_string(), &self.weight)];
        if let Some(ref b) = self.bias {
            params.push(("bias".to_string(), b));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
    }

    fn eval(&mut self) {
        self.training = false;
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ConvTranspose3dBackward
// ---------------------------------------------------------------------------

/// Backward function for `ConvTranspose3d` forward pass.
///
/// The backward of a transposed 3-D convolution is a regular 3-D convolution.
#[derive(Debug)]
struct ConvTranspose3dBackward<T: Float> {
    input: Tensor<T>,
    weight: Tensor<T>,
    bias: Option<Tensor<T>>,
    in_channels: usize,
    out_channels: usize,
    kernel_size: (usize, usize, usize),
    stride: (usize, usize, usize),
    padding: (usize, usize, usize),
    _output_padding: (usize, usize, usize),
    d_out: usize,
    h_out: usize,
    w_out: usize,
}

impl<T: Float> GradFn<T> for ConvTranspose3dBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output shape: [B, C_out, D_out, H_out, W_out]
        let go_data = grad_output.data_vec()?;
        let batch = self.input.shape()[0];
        let d_in = self.input.shape()[2];
        let h_in = self.input.shape()[3];
        let w_in = self.input.shape()[4];
        let (kd, kh, kw) = self.kernel_size;
        let (sd, sh, sw) = self.stride;
        let (pd, ph, pw) = self.padding;
        let spatial_out = self.d_out * self.h_out * self.w_out;

        // --- grad_input ---
        // The backward of ConvTranspose3d w.r.t. input is a regular Conv3d
        // of grad_output with the original (non-flipped) weight.
        let grad_input = if self.input.requires_grad() {
            let weight_data = self.weight.data_vec()?;
            let col_rows = self.out_channels * kd * kh * kw;

            // Reshape weight to [C_in, C_out * kD * kH * kW]
            let weight_2d = Tensor::from_storage(
                TensorStorage::cpu(weight_data),
                vec![self.in_channels, col_rows],
                false,
            )?;

            // im2col_3d on grad_output with the conv parameters
            let (go_cols, _go_col_rows, go_col_cols) = im2col_3d(
                &go_data,
                batch,
                self.out_channels,
                self.d_out,
                self.h_out,
                self.w_out,
                kd,
                kh,
                kw,
                sd,
                sh,
                sw,
                pd,
                ph,
                pw,
            );

            let zero = <T as num_traits::Zero>::zero();
            let spatial_in = d_in * h_in * w_in;
            let mut gi = vec![zero; batch * self.in_channels * spatial_in];

            for b in 0..batch {
                let col_start = b * col_rows * go_col_cols;
                let col_end = col_start + col_rows * go_col_cols;
                let go_cols_b = Tensor::from_storage(
                    TensorStorage::cpu(go_cols[col_start..col_end].to_vec()),
                    vec![col_rows, go_col_cols],
                    false,
                )?;

                let gi_b = mm(&weight_2d, &go_cols_b)?;
                let gi_data = gi_b.data()?;

                let out_start = b * self.in_channels * spatial_in;
                let copy_len = self.in_channels * spatial_in;
                gi[out_start..out_start + copy_len].copy_from_slice(&gi_data[..copy_len]);
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gi),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        // --- grad_weight ---
        // grad_weight[c_in, c_out, kd, kh, kw] = sum_b input_b (x) grad_output_b
        let grad_weight = if self.weight.requires_grad() {
            let zero = <T as num_traits::Zero>::zero();
            let weight_numel = self.in_channels * self.out_channels * kd * kh * kw;
            let mut gw = vec![zero; weight_numel];
            let input_data = self.input.data_vec()?;
            let spatial_in = d_in * h_in * w_in;

            for b in 0..batch {
                for ci in 0..self.in_channels {
                    for co in 0..self.out_channels {
                        for dd in 0..kd {
                            for dh in 0..kh {
                                for dw in 0..kw {
                                    let mut acc = zero;
                                    for id in 0..d_in {
                                        for ih in 0..h_in {
                                            for iw in 0..w_in {
                                                let od = id * sd + dd;
                                                let oh = ih * sh + dh;
                                                let ow = iw * sw + dw;
                                                if od >= pd
                                                    && oh >= ph
                                                    && ow >= pw
                                                    && (od - pd) < self.d_out
                                                    && (oh - ph) < self.h_out
                                                    && (ow - pw) < self.w_out
                                                {
                                                    let go_idx =
                                                        b * self.out_channels * spatial_out
                                                            + co * spatial_out
                                                            + (od - pd) * self.h_out * self.w_out
                                                            + (oh - ph) * self.w_out
                                                            + (ow - pw);
                                                    let in_idx = b * self.in_channels * spatial_in
                                                        + ci * spatial_in
                                                        + id * h_in * w_in
                                                        + ih * w_in
                                                        + iw;
                                                    acc += input_data[in_idx] * go_data[go_idx];
                                                }
                                            }
                                        }
                                    }
                                    gw[ci * self.out_channels * kd * kh * kw
                                        + co * kd * kh * kw
                                        + dd * kh * kw
                                        + dh * kw
                                        + dw] += acc;
                                }
                            }
                        }
                    }
                }
            }

            Some(Tensor::from_storage(
                TensorStorage::cpu(gw),
                vec![self.in_channels, self.out_channels, kd, kh, kw],
                false,
            )?)
        } else {
            None
        };

        // --- grad_bias ---
        let grad_bias = match &self.bias {
            Some(b) if b.requires_grad() => {
                let zero = <T as num_traits::Zero>::zero();
                let mut gb = vec![zero; self.out_channels];
                for batch_idx in 0..batch {
                    for c in 0..self.out_channels {
                        for s in 0..spatial_out {
                            gb[c] += go_data
                                [batch_idx * self.out_channels * spatial_out + c * spatial_out + s];
                        }
                    }
                }
                Some(Tensor::from_storage(
                    TensorStorage::cpu(gb),
                    vec![self.out_channels],
                    false,
                )?)
            }
            _ => None,
        };

        let mut grads = vec![grad_input, grad_weight];
        if self.bias.is_some() {
            grads.push(grad_bias);
        }
        Ok(grads)
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        let mut v = vec![&self.input, &self.weight];
        if let Some(ref b) = self.bias {
            v.push(b);
        }
        v
    }

    fn name(&self) -> &'static str {
        "ConvTranspose3dBackward"
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module;

    /// Helper: create a tensor from flat data and shape.
    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
    }

    /// Helper: create a leaf tensor that requires grad.
    fn leaf(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), true).unwrap()
    }

    /// Assert two slices are element-wise close.
    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: {} vs {}",
            actual.len(),
            expected.len()
        );
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: actual={a} expected={e} (diff {})",
                (a - e).abs()
            );
        }
    }

    // -----------------------------------------------------------------------
    // Output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_output_shape_no_padding() {
        // Input: [1, 1, 5, 5], kernel 3x3, stride 1, padding 0
        // H_out = (5 - 3) / 1 + 1 = 3, W_out = 3
        let conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), false).unwrap();
        let input = t(&[0.0; 25], &[1, 1, 5, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3, 3]);
    }

    #[test]
    fn test_output_shape_with_padding() {
        // Input: [2, 3, 8, 8], kernel 3x3, stride 1, padding 1
        // H_out = (8 + 2 - 3) / 1 + 1 = 8
        let conv = Conv2d::<f32>::new(3, 16, (3, 3), (1, 1), (1, 1), true).unwrap();
        let input = t(&vec![0.0; 2 * 3 * 8 * 8], &[2, 3, 8, 8]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16, 8, 8]);
    }

    #[test]
    fn test_output_shape_with_stride() {
        // Input: [1, 1, 6, 6], kernel 3x3, stride 2, padding 0
        // H_out = (6 - 3) / 2 + 1 = 2
        let conv = Conv2d::<f32>::new(1, 4, (3, 3), (2, 2), (0, 0), false).unwrap();
        let input = t(&[0.0; 36], &[1, 1, 6, 6]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 2, 2]);
    }

    // -----------------------------------------------------------------------
    // 1x1 convolution == linear (per-pixel)
    // -----------------------------------------------------------------------

    #[test]
    fn test_1x1_conv_equals_linear() {
        // A 1x1 conv with 2 input channels and 3 output channels is equivalent
        // to a linear layer applied independently at each spatial position.
        //
        // weight shape: [3, 2, 1, 1] -- interpreted as a [3, 2] matrix
        // input shape: [1, 2, 2, 2]  -- 2 channels, 2x2 spatial
        //
        // For each pixel (h, w): output[:, h, w] = weight.squeeze() @ input[:, h, w]

        let weight_data: Vec<f32> = vec![
            1.0, 2.0, // out_channel 0: [1, 2]
            3.0, 4.0, // out_channel 1: [3, 4]
            5.0, 6.0, // out_channel 2: [5, 6]
        ];
        // Input: channel 0 = [[1, 2], [3, 4]], channel 1 = [[5, 6], [7, 8]]
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0, // channel 0
            5.0, 6.0, 7.0, 8.0, // channel 1
        ];

        // Manually construct Conv2d with known weights.
        let weight_param = Parameter::from_slice(&weight_data, &[3, 2, 1, 1]).unwrap();
        let conv = Conv2d {
            weight: weight_param,
            bias: None,
            in_channels: 2,
            out_channels: 3,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            training: false,
        };

        let input = t(&input_data, &[1, 2, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 3, 2, 2]);

        let out = output.data().unwrap();

        // Pixel (0,0): in = [1, 5], out = [1*1+2*5, 3*1+4*5, 5*1+6*5] = [11, 23, 35]
        // Pixel (0,1): in = [2, 6], out = [1*2+2*6, 3*2+4*6, 5*2+6*6] = [14, 30, 46]
        // Pixel (1,0): in = [3, 7], out = [1*3+2*7, 3*3+4*7, 5*3+6*7] = [17, 37, 57]
        // Pixel (1,1): in = [4, 8], out = [1*4+2*8, 3*4+4*8, 5*4+6*8] = [20, 44, 68]

        // Output layout: [C_out, H, W] = [3, 2, 2]
        // Channel 0: [11, 14, 17, 20]
        // Channel 1: [23, 30, 37, 44]
        // Channel 2: [35, 46, 57, 68]
        let expected = [
            11.0, 14.0, 17.0, 20.0, // out channel 0
            23.0, 30.0, 37.0, 44.0, // out channel 1
            35.0, 46.0, 57.0, 68.0, // out channel 2
        ];
        assert_close(out, &expected, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_bias_addition() {
        // 1x1 conv with bias.
        let weight_data = vec![1.0f32]; // [1, 1, 1, 1]
        let bias_data = vec![10.0f32]; // [1]

        let conv = Conv2d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            training: false,
        };

        let input = t(&[2.0, 3.0, 4.0, 5.0], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        // output = input * 1.0 + 10.0
        assert_close(output.data().unwrap(), &[12.0, 13.0, 14.0, 15.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Backward shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_produces_correct_shapes() {
        // We manually invoke the backward function and check shapes.
        let weight_data = vec![1.0f32; 2 * 3 * 3]; // [2, 1, 3, 3]
        let input_data = vec![1.0f32; 5 * 5]; // [1, 1, 5, 5]
        let bias_data = vec![0.0f32; 2];

        let weight_param = Parameter::from_slice(&weight_data, &[2, 1, 3, 3]).unwrap();
        let bias_param = Parameter::from_slice(&bias_data, &[2]).unwrap();

        let conv = Conv2d {
            weight: weight_param,
            bias: Some(bias_param),
            in_channels: 1,
            out_channels: 2,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            training: false,
        };

        // Forward to get the grad_fn.
        let input = leaf(&input_data, &[1, 1, 5, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 3, 3]);

        // Make sure grad_fn is attached.
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "Conv2dBackward");

        // Construct a grad_output of the right shape.
        let grad_output = t(&[1.0; 2 * 3 * 3], &[1, 2, 3, 3]);
        let grads = output.grad_fn().unwrap().backward(&grad_output).unwrap();

        // grad_input shape should be [1, 1, 5, 5]
        assert!(grads[0].is_some());
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[1, 1, 5, 5]);

        // grad_weight shape should be [2, 1, 3, 3]
        assert!(grads[1].is_some());
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 1, 3, 3]);

        // grad_bias shape should be [2]
        assert!(grads[2].is_some());
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[2]);
    }

    // -----------------------------------------------------------------------
    // Parameter count
    // -----------------------------------------------------------------------

    #[test]
    fn test_parameter_count_with_bias() {
        let conv = Conv2d::<f32>::new(3, 16, (3, 3), (1, 1), (0, 0), true).unwrap();
        // weight: 16 * 3 * 3 * 3 = 432
        // bias:   16
        // total:  448
        assert_eq!(conv.num_parameters(), 448);
        assert_eq!(conv.parameters().len(), 2);
    }

    #[test]
    fn test_parameter_count_without_bias() {
        let conv = Conv2d::<f32>::new(3, 16, (3, 3), (1, 1), (0, 0), false).unwrap();
        assert_eq!(conv.num_parameters(), 432);
        assert_eq!(conv.parameters().len(), 1);
    }

    // -----------------------------------------------------------------------
    // Module trait
    // -----------------------------------------------------------------------

    #[test]
    fn test_named_parameters() {
        let conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), true).unwrap();
        let named = conv.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    #[test]
    fn test_train_eval() {
        let mut conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), false).unwrap();
        assert!(conv.is_training());
        conv.eval();
        assert!(!conv.is_training());
        conv.train();
        assert!(conv.is_training());
    }

    // -----------------------------------------------------------------------
    // Edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_invalid_input_ndim() {
        let conv = Conv2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), false).unwrap();
        let input = t(&[1.0, 2.0, 3.0], &[3]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_channel_mismatch() {
        let conv = Conv2d::<f32>::new(3, 1, (3, 3), (1, 1), (0, 0), false).unwrap();
        let input = t(&[0.0; 5 * 5], &[1, 1, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_zero_channels_rejected() {
        assert!(Conv2d::<f32>::new(0, 16, (3, 3), (1, 1), (0, 0), false).is_err());
        assert!(Conv2d::<f32>::new(3, 0, (3, 3), (1, 1), (0, 0), false).is_err());
    }

    #[test]
    fn test_zero_kernel_rejected() {
        assert!(Conv2d::<f32>::new(1, 1, (0, 3), (1, 1), (0, 0), false).is_err());
    }

    #[test]
    fn test_zero_stride_rejected() {
        assert!(Conv2d::<f32>::new(1, 1, (3, 3), (0, 1), (0, 0), false).is_err());
    }

    // -----------------------------------------------------------------------
    // im2col / col2im roundtrip
    // -----------------------------------------------------------------------

    #[test]
    fn test_im2col_basic() {
        // 1 batch, 1 channel, 3x3 input, 2x2 kernel, stride 1, no padding
        // H_out = 2, W_out = 2
        // Columns: each column is a flattened 2x2 patch
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let (cols, rows, n_cols) = im2col(&input, 1, 1, 3, 3, 2, 2, 1, 1, 0, 0);
        assert_eq!(rows, 4); // 1 * 2 * 2
        assert_eq!(n_cols, 4); // 2 * 2

        // Patch (0,0): [1, 2, 4, 5]
        // Patch (0,1): [2, 3, 5, 6]
        // Patch (1,0): [4, 5, 7, 8]
        // Patch (1,1): [5, 6, 8, 9]
        //
        // cols layout: [row][col] where row = c*kH*kW+kh*kW+kw, col = oh*W_out+ow
        // Row 0 (kh=0,kw=0): [1, 2, 4, 5]
        // Row 1 (kh=0,kw=1): [2, 3, 5, 6]
        // Row 2 (kh=1,kw=0): [4, 5, 7, 8]
        // Row 3 (kh=1,kw=1): [5, 6, 8, 9]
        assert_close(
            &cols,
            &[
                1.0, 2.0, 4.0, 5.0, // row 0
                2.0, 3.0, 5.0, 6.0, // row 1
                4.0, 5.0, 7.0, 8.0, // row 2
                5.0, 6.0, 8.0, 9.0, // row 3
            ],
            1e-7,
        );
    }

    #[test]
    fn test_col2im_roundtrip_no_overlap() {
        // With stride == kernel_size and no padding, im2col + col2im is lossless.
        // 1 batch, 1 channel, 4x4, kernel 2x2, stride 2, no padding
        // H_out = 2, W_out = 2
        #[rustfmt::skip]
        let input: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];

        let (cols, _rows, _n_cols) = im2col(&input, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0);
        let recovered = col2im(&cols, 1, 1, 4, 4, 2, 2, 2, 2, 0, 0, 2, 2);

        assert_close(&recovered, &input, 1e-7);
    }

    // -----------------------------------------------------------------------
    // Forward correctness with a simple 3x3 kernel
    // -----------------------------------------------------------------------

    #[test]
    fn test_3x3_conv_forward() {
        // 1 batch, 1 channel, 3x3 input, 3x3 kernel, stride 1, no padding
        // Output: 1x1 (single value = sum of element-wise product)
        #[rustfmt::skip]
        let input_data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        #[rustfmt::skip]
        let weight_data: Vec<f32> = vec![
            1.0, 0.0, -1.0,
            1.0, 0.0, -1.0,
            1.0, 0.0, -1.0,
        ];

        let conv = Conv2d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 3, 3]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (0, 0),
            training: false,
        };

        let input = t(&input_data, &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 1, 1]);

        // Expected: 1*1 + 0*2 + (-1)*3 + 1*4 + 0*5 + (-1)*6 + 1*7 + 0*8 + (-1)*9
        //         = 1 - 3 + 4 - 6 + 7 - 9 = -6
        assert_close(output.data().unwrap(), &[-6.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Padding correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_padding_preserves_spatial_size() {
        // Input: [1, 1, 3, 3], kernel 3x3, stride 1, padding 1
        // H_out = (3 + 2 - 3) / 1 + 1 = 3 (same size!)
        let weight_data = vec![0.0f32; 9];
        let mut weight_data_center = weight_data;
        weight_data_center[4] = 1.0; // Center of 3x3 = identity

        let conv = Conv2d {
            weight: Parameter::from_slice(&weight_data_center, &[1, 1, 3, 3]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3),
            stride: (1, 1),
            padding: (1, 1),
            training: false,
        };

        let input_data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = t(&input_data, &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3, 3]);

        // With center-only kernel + padding, output should equal input.
        assert_close(output.data().unwrap(), &input_data, 1e-5);
    }

    // ===================================================================
    // Conv1d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // Conv1d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_output_shape_no_padding() {
        // Input: [1, 1, 10], kernel 3, stride 1, padding 0
        // L_out = (10 - 3) / 1 + 1 = 8
        let conv = Conv1d::<f32>::new(1, 4, 3, 1, 0, false).unwrap();
        let input = t(&[0.0; 10], &[1, 1, 10]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_conv1d_output_shape_with_padding() {
        // Input: [2, 3, 16], kernel 3, stride 1, padding 1
        // L_out = (16 + 2 - 3) / 1 + 1 = 16
        let conv = Conv1d::<f32>::new(3, 8, 3, 1, 1, true).unwrap();
        let input = t(&vec![0.0; 2 * 3 * 16], &[2, 3, 16]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 8, 16]);
    }

    #[test]
    fn test_conv1d_output_shape_with_stride() {
        // Input: [1, 1, 10], kernel 3, stride 2, padding 0
        // L_out = (10 - 3) / 2 + 1 = 4
        let conv = Conv1d::<f32>::new(1, 2, 3, 2, 0, false).unwrap();
        let input = t(&[0.0; 10], &[1, 1, 10]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 4]);
    }

    // -----------------------------------------------------------------------
    // Conv1d: 1x1 kernel correctness (acts as per-position linear)
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_1x1_kernel_correctness() {
        // A kernel_size=1 Conv1d is equivalent to a linear layer applied at
        // each position independently.
        //
        // weight: [2, 1, 1] = [[3.0], [5.0]]
        // input:  [1, 1, 4] = [1, 2, 3, 4]
        // output: [1, 2, 4]
        //   out_ch 0: [3, 6, 9, 12]
        //   out_ch 1: [5, 10, 15, 20]
        let weight_data = vec![3.0f32, 5.0];
        let conv = Conv1d {
            weight: Parameter::from_slice(&weight_data, &[2, 1, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 4]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 4]);
        assert_close(
            output.data().unwrap(),
            &[3.0, 6.0, 9.0, 12.0, 5.0, 10.0, 15.0, 20.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // Conv1d: forward correctness with a 3-wide kernel
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_3_kernel_forward() {
        // Input: [1, 1, 5] = [1, 2, 3, 4, 5]
        // Kernel: [1, 1, 3] = [1, 0, -1]
        // Stride 1, padding 0 => L_out = 3
        // Expected: [1*1+0*2+(-1)*3, 1*2+0*3+(-1)*4, 1*3+0*4+(-1)*5] = [-2, -2, -2]
        let conv = Conv1d {
            weight: Parameter::from_slice(&[1.0f32, 0.0, -1.0], &[1, 1, 3]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0, 5.0], &[1, 1, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3]);
        assert_close(output.data().unwrap(), &[-2.0, -2.0, -2.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv1d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_bias() {
        let conv = Conv1d {
            weight: Parameter::from_slice(&[1.0f32], &[1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&[10.0f32], &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            training: false,
        };

        let input = t(&[2.0, 3.0, 4.0], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[12.0, 13.0, 14.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv1d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv1d_invalid_ndim() {
        let conv = Conv1d::<f32>::new(1, 1, 3, 1, 0, false).unwrap();
        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv1d_channel_mismatch() {
        let conv = Conv1d::<f32>::new(3, 1, 3, 1, 0, false).unwrap();
        let input = t(&[0.0; 10], &[1, 1, 10]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv1d_zero_channels_rejected() {
        assert!(Conv1d::<f32>::new(0, 4, 3, 1, 0, false).is_err());
        assert!(Conv1d::<f32>::new(1, 0, 3, 1, 0, false).is_err());
    }

    #[test]
    fn test_conv1d_zero_kernel_rejected() {
        assert!(Conv1d::<f32>::new(1, 1, 0, 1, 0, false).is_err());
    }

    #[test]
    fn test_conv1d_zero_stride_rejected() {
        assert!(Conv1d::<f32>::new(1, 1, 3, 0, 0, false).is_err());
    }

    #[test]
    fn test_conv1d_parameter_count() {
        let conv = Conv1d::<f32>::new(3, 8, 5, 1, 0, true).unwrap();
        // weight: 8 * 3 * 5 = 120, bias: 8, total: 128
        assert_eq!(conv.num_parameters(), 128);
        assert_eq!(conv.parameters().len(), 2);
    }

    // ===================================================================
    // ConvTranspose2d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // ConvTranspose2d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_output_shape_basic() {
        // Input: [1, 1, 3, 3], kernel 3x3, stride 1, padding 0, output_padding 0
        // H_out = (3 - 1) * 1 - 0 + 3 + 0 = 5
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), (0, 0), false).unwrap();
        let input = t(&[0.0; 9], &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn test_conv_transpose2d_output_shape_stride2() {
        // Input: [1, 1, 2, 2], kernel 3x3, stride 2, padding 0, output_padding 0
        // H_out = (2 - 1) * 2 - 0 + 3 + 0 = 5
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (2, 2), (0, 0), (0, 0), false).unwrap();
        let input = t(&[0.0; 4], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn test_conv_transpose2d_output_shape_with_padding() {
        // Input: [1, 1, 3, 3], kernel 3x3, stride 2, padding 1, output_padding 0
        // H_out = (3 - 1) * 2 - 2 + 3 + 0 = 5
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (2, 2), (1, 1), (0, 0), false).unwrap();
        let input = t(&[0.0; 9], &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5]);
    }

    #[test]
    fn test_conv_transpose2d_output_shape_with_output_padding() {
        // Input: [1, 1, 3, 3], kernel 3x3, stride 2, padding 1, output_padding 1
        // H_out = (3 - 1) * 2 - 2 + 3 + 1 = 6
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (2, 2), (1, 1), (1, 1), false).unwrap();
        let input = t(&[0.0; 9], &[1, 1, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 6, 6]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: stride=2 doubles spatial dims (upsampling)
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_stride2_upsamples() {
        // With stride=2, kernel=2x2, padding=0, output_padding=0:
        // H_out = (H - 1) * 2 + 2 = 2 * H
        // So a 4x4 input becomes 8x8 — doubling spatial dims.
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (2, 2), (2, 2), (0, 0), (0, 0), false).unwrap();
        let input = t(&[0.0; 4 * 4], &[1, 1, 4, 4]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 8, 8]);
    }

    #[test]
    fn test_conv_transpose2d_stride2_upsamples_multichannel() {
        // [2, 8, 4, 4] -> [2, 16, 8, 8] with stride=2, kernel=2x2
        let conv =
            ConvTranspose2d::<f32>::new(8, 16, (2, 2), (2, 2), (0, 0), (0, 0), true).unwrap();
        let input = t(&vec![0.0; 2 * 8 * 4 * 4], &[2, 8, 4, 4]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16, 8, 8]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: 1x1 kernel correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_1x1_kernel() {
        // With a 1x1 kernel, stride 1, no padding, the transposed conv is
        // equivalent to a regular 1x1 conv (just a per-pixel linear transform),
        // but with channels transposed:
        // weight shape: [in_channels=1, out_channels=2, 1, 1]
        // input: [1, 1, 2, 2]
        // Each output channel c gets: input * weight[0, c, 0, 0]
        let weight_data = vec![3.0f32, 7.0]; // [1, 2, 1, 1]
        let conv = ConvTranspose2d {
            weight: Parameter::from_slice(&weight_data, &[1, 2, 1, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2, 2]);

        // out_ch 0: input * 3 = [3, 6, 9, 12]
        // out_ch 1: input * 7 = [7, 14, 21, 28]
        assert_close(
            output.data().unwrap(),
            &[3.0, 6.0, 9.0, 12.0, 7.0, 14.0, 21.0, 28.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: correctness with stride insertion
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_stride2_correctness() {
        // Input: [1, 1, 2, 2] = [[1, 2], [3, 4]]
        // Kernel: [1, 1, 2, 2] = [[1, 1], [1, 1]]  (all ones)
        // Stride=2, padding=0, output_padding=0
        // H_out = (2-1)*2 + 2 = 4, W_out = 4
        //
        // Stride insertion produces 3x3:
        //   [[1, 0, 2],
        //    [0, 0, 0],
        //    [3, 0, 4]]
        //
        // Flipped kernel (all ones, still all ones): [[1,1],[1,1]]
        // Internal conv with pad = kernel-1 = 1, stride=1 on 3x3:
        // Padded to 5x5:
        //   [[0, 0, 0, 0, 0],
        //    [0, 1, 0, 2, 0],
        //    [0, 0, 0, 0, 0],
        //    [0, 3, 0, 4, 0],
        //    [0, 0, 0, 0, 0]]
        // Convolve with 2x2 all-ones kernel, output 4x4:
        //   row 0: [1, 0+1, 2+0, 2] = [1, 1, 2, 2]
        //   row 1: [0+1, 1+0+0+0, 0+2+0+0, 0+2] = [1, 1, 2, 2]
        //   row 2: [3, 0+3, 4+0, 4] = [3, 3, 4, 4]
        //   row 3: [3, 3, 4, 4]
        //
        // Wait, let me recalculate more carefully.
        // After padding, we convolve (sum of 2x2 window at each position):
        // pos(0,0): 0+0+0+1 = 1
        // pos(0,1): 0+0+1+0 = 1
        // pos(0,2): 0+0+0+2 = 2
        // pos(0,3): 0+0+2+0 = 2
        // pos(1,0): 0+1+0+0 = 1
        // pos(1,1): 1+0+0+0 = 1
        // pos(1,2): 0+2+0+0 = 2
        // pos(1,3): 2+0+0+0 = 2
        // pos(2,0): 0+0+0+3 = 3
        // pos(2,1): 0+0+3+0 = 3
        // pos(2,2): 0+0+0+4 = 4
        // pos(2,3): 0+0+4+0 = 4
        // pos(3,0): 0+3+0+0 = 3
        // pos(3,1): 3+0+0+0 = 3
        // pos(3,2): 0+4+0+0 = 4
        // pos(3,3): 4+0+0+0 = 4

        let weight_data = vec![1.0f32; 4]; // [1, 1, 2, 2]
        let conv = ConvTranspose2d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 2, 2]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: (2, 2),
            stride: (2, 2),
            padding: (0, 0),
            output_padding: (0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4]);

        #[rustfmt::skip]
        let expected = [
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_close(output.data().unwrap(), &expected, 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_bias() {
        let weight_data = vec![1.0f32]; // [1, 1, 1, 1] identity
        let bias_data = vec![5.0f32];
        let conv = ConvTranspose2d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (1, 1),
            stride: (1, 1),
            padding: (0, 0),
            output_padding: (0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[6.0, 7.0, 8.0, 9.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose2d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose2d_invalid_ndim() {
        let conv =
            ConvTranspose2d::<f32>::new(1, 1, (3, 3), (1, 1), (0, 0), (0, 0), false).unwrap();
        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose2d_channel_mismatch() {
        let conv =
            ConvTranspose2d::<f32>::new(3, 1, (3, 3), (1, 1), (0, 0), (0, 0), false).unwrap();
        let input = t(&[0.0; 5 * 5], &[1, 1, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose2d_zero_channels_rejected() {
        assert!(ConvTranspose2d::<f32>::new(0, 1, (3, 3), (1, 1), (0, 0), (0, 0), false).is_err());
        assert!(ConvTranspose2d::<f32>::new(1, 0, (3, 3), (1, 1), (0, 0), (0, 0), false).is_err());
    }

    #[test]
    fn test_conv_transpose2d_output_padding_too_large() {
        // output_padding must be < stride
        assert!(ConvTranspose2d::<f32>::new(1, 1, (3, 3), (2, 2), (0, 0), (2, 2), false).is_err());
    }

    #[test]
    fn test_conv_transpose2d_parameter_count() {
        let conv =
            ConvTranspose2d::<f32>::new(8, 16, (3, 3), (2, 2), (1, 1), (0, 0), true).unwrap();
        // weight: 8 * 16 * 3 * 3 = 1152, bias: 16, total: 1168
        assert_eq!(conv.num_parameters(), 1168);
        assert_eq!(conv.parameters().len(), 2);
    }

    // ===================================================================
    // Conv3d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // Conv3d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_output_shape_no_padding() {
        // Input: [1, 1, 5, 5, 5], kernel 3x3x3, stride 1, padding 0
        // D_out = (5 - 3) / 1 + 1 = 3
        let conv = Conv3d::<f32>::new(1, 4, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).unwrap();
        let input = t(&vec![0.0; 5 * 5 * 5], &[1, 1, 5, 5, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 3, 3, 3]);
    }

    #[test]
    fn test_conv3d_output_shape_with_padding() {
        // Input: [2, 3, 8, 8, 8], kernel 3x3x3, stride 1, padding 1
        // D_out = (8 + 2 - 3) / 1 + 1 = 8
        let conv = Conv3d::<f32>::new(3, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1), true).unwrap();
        let input = t(&vec![0.0; 2 * 3 * 8 * 8 * 8], &[2, 3, 8, 8, 8]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[2, 16, 8, 8, 8]);
    }

    #[test]
    fn test_conv3d_output_shape_with_stride() {
        // Input: [1, 1, 6, 6, 6], kernel 3x3x3, stride 2, padding 0
        // D_out = (6 - 3) / 2 + 1 = 2
        let conv = Conv3d::<f32>::new(1, 4, (3, 3, 3), (2, 2, 2), (0, 0, 0), false).unwrap();
        let input = t(&vec![0.0; 6 * 6 * 6], &[1, 1, 6, 6, 6]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 4, 2, 2, 2]);
    }

    // -----------------------------------------------------------------------
    // Conv3d: 1x1x1 kernel correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_1x1x1_kernel_correctness() {
        // weight: [2, 1, 1, 1, 1] = [3.0, 5.0]
        // input:  [1, 1, 2, 1, 1] = [1.0, 2.0]
        // output: [1, 2, 2, 1, 1]
        //   out_ch 0: [3.0, 6.0]
        //   out_ch 1: [5.0, 10.0]
        let weight_data = vec![3.0f32, 5.0];
        let conv = Conv3d {
            weight: Parameter::from_slice(&weight_data, &[2, 1, 1, 1, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: (1, 1, 1),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0], &[1, 1, 2, 1, 1]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2, 1, 1]);
        assert_close(output.data().unwrap(), &[3.0, 6.0, 5.0, 10.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv3d: forward correctness with a 3x3x3 kernel
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_3x3x3_kernel_forward() {
        // Input: [1, 1, 3, 3, 3] (all ones), kernel: [1, 1, 3, 3, 3] (all ones)
        // Output: [1, 1, 1, 1, 1] = sum of 27 ones = 27.0
        let input_data = vec![1.0f32; 27];
        let weight_data = vec![1.0f32; 27];
        let conv = Conv3d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 3, 3, 3]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: (3, 3, 3),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            training: false,
        };

        let input = t(&input_data, &[1, 1, 3, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 1, 1, 1]);
        assert_close(output.data().unwrap(), &[27.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv3d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_bias() {
        let conv = Conv3d {
            weight: Parameter::from_slice(&[1.0f32], &[1, 1, 1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&[10.0f32], &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (1, 1, 1),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            training: false,
        };

        let input = t(&[2.0, 3.0], &[1, 1, 2, 1, 1]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[12.0, 13.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // Conv3d: backward produces correct shapes
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_backward_produces_correct_shapes() {
        let weight_data = vec![1.0f32; 2 * 3 * 3 * 3]; // [2, 1, 3, 3, 3]
        let input_data = vec![1.0f32; 5 * 5 * 5]; // [1, 1, 5, 5, 5]
        let bias_data = vec![0.0f32; 2];

        let conv = Conv3d {
            weight: Parameter::from_slice(&weight_data, &[2, 1, 3, 3, 3]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[2]).unwrap()),
            in_channels: 1,
            out_channels: 2,
            kernel_size: (3, 3, 3),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            training: false,
        };

        let input = leaf(&input_data, &[1, 1, 5, 5, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 3, 3, 3]);
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "Conv3dBackward");

        let grad_output = t(&vec![1.0; 2 * 3 * 3 * 3], &[1, 2, 3, 3, 3]);
        let grads = output.grad_fn().unwrap().backward(&grad_output).unwrap();

        assert!(grads[0].is_some());
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[1, 1, 5, 5, 5]);
        assert!(grads[1].is_some());
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[2, 1, 3, 3, 3]);
        assert!(grads[2].is_some());
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[2]);
    }

    // -----------------------------------------------------------------------
    // Conv3d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv3d_invalid_ndim() {
        let conv = Conv3d::<f32>::new(1, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).unwrap();
        let input = t(&[0.0; 25], &[1, 1, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv3d_channel_mismatch() {
        let conv = Conv3d::<f32>::new(3, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).unwrap();
        let input = t(&vec![0.0; 5 * 5 * 5], &[1, 1, 5, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv3d_zero_channels_rejected() {
        assert!(Conv3d::<f32>::new(0, 16, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).is_err());
        assert!(Conv3d::<f32>::new(3, 0, (3, 3, 3), (1, 1, 1), (0, 0, 0), false).is_err());
    }

    #[test]
    fn test_conv3d_zero_kernel_rejected() {
        assert!(Conv3d::<f32>::new(1, 1, (0, 3, 3), (1, 1, 1), (0, 0, 0), false).is_err());
    }

    #[test]
    fn test_conv3d_zero_stride_rejected() {
        assert!(Conv3d::<f32>::new(1, 1, (3, 3, 3), (0, 1, 1), (0, 0, 0), false).is_err());
    }

    #[test]
    fn test_conv3d_parameter_count() {
        let conv = Conv3d::<f32>::new(3, 8, (3, 3, 3), (1, 1, 1), (0, 0, 0), true).unwrap();
        // weight: 8 * 3 * 3 * 3 * 3 = 648, bias: 8, total: 656
        assert_eq!(conv.num_parameters(), 656);
        assert_eq!(conv.parameters().len(), 2);
    }

    #[test]
    fn test_conv3d_named_parameters() {
        let conv = Conv3d::<f32>::new(1, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), true).unwrap();
        let named = conv.named_parameters();
        assert_eq!(named.len(), 2);
        assert_eq!(named[0].0, "weight");
        assert_eq!(named[1].0, "bias");
    }

    // ===================================================================
    // ConvTranspose1d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // ConvTranspose1d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_output_shape_basic() {
        // Input: [1, 1, 5], kernel 3, stride 1, padding 0, output_padding 0
        // L_out = (5 - 1) * 1 - 0 + 3 + 0 = 7
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 1, 0, 0, false).unwrap();
        let input = t(&[0.0; 5], &[1, 1, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 7]);
    }

    #[test]
    fn test_conv_transpose1d_output_shape_stride2() {
        // Input: [1, 1, 3], kernel 3, stride 2, padding 0, output_padding 0
        // L_out = (3 - 1) * 2 - 0 + 3 + 0 = 7
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 2, 0, 0, false).unwrap();
        let input = t(&[0.0; 3], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 7]);
    }

    #[test]
    fn test_conv_transpose1d_output_shape_with_padding() {
        // Input: [1, 1, 5], kernel 3, stride 2, padding 1, output_padding 0
        // L_out = (5 - 1) * 2 - 2 + 3 + 0 = 9
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 2, 1, 0, false).unwrap();
        let input = t(&[0.0; 5], &[1, 1, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 9]);
    }

    #[test]
    fn test_conv_transpose1d_output_shape_with_output_padding() {
        // Input: [1, 1, 5], kernel 3, stride 2, padding 1, output_padding 1
        // L_out = (5 - 1) * 2 - 2 + 3 + 1 = 10
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 2, 1, 1, false).unwrap();
        let input = t(&[0.0; 5], &[1, 1, 5]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 10]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: 1x1 kernel correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_1x1_kernel() {
        // With a kernel_size=1, stride 1, no padding, the transposed conv is
        // a per-position linear transform with channels transposed.
        // weight shape: [1, 2, 1] (in_channels=1, out_channels=2, k=1)
        let weight_data = vec![3.0f32, 7.0]; // [1, 2, 1]
        let conv = ConvTranspose1d {
            weight: Parameter::from_slice(&weight_data, &[1, 2, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            output_padding: 0,
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 3]);

        // out_ch 0: input * 3 = [3, 6, 9]
        // out_ch 1: input * 7 = [7, 14, 21]
        assert_close(
            output.data().unwrap(),
            &[3.0, 6.0, 9.0, 7.0, 14.0, 21.0],
            1e-5,
        );
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: stride=2 correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_stride2_correctness() {
        // Input: [1, 1, 2] = [1, 2]
        // Kernel: [1, 1, 2] = [1, 1] (all ones)
        // Stride=2, padding=0, output_padding=0
        // L_out = (2-1)*2 + 2 = 4
        //
        // Stride insertion produces [1, 0, 2]
        // Flipped kernel (all ones): [1, 1]
        // Internal conv with pad = 2-1 = 1, stride=1 on [1, 0, 2]:
        // Padded to [0, 1, 0, 2, 0]
        // Convolve with [1, 1] kernel, output 4:
        //   pos 0: 0+1 = 1
        //   pos 1: 1+0 = 1
        //   pos 2: 0+2 = 2
        //   pos 3: 2+0 = 2
        let weight_data = vec![1.0f32; 2]; // [1, 1, 2]
        let conv = ConvTranspose1d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 2]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 1,
            kernel_size: 2,
            stride: 2,
            padding: 0,
            output_padding: 0,
            training: false,
        };

        let input = t(&[1.0, 2.0], &[1, 1, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4]);
        assert_close(output.data().unwrap(), &[1.0, 1.0, 2.0, 2.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_bias() {
        let conv = ConvTranspose1d {
            weight: Parameter::from_slice(&[1.0f32], &[1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&[5.0f32], &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: 1,
            stride: 1,
            padding: 0,
            output_padding: 0,
            training: false,
        };

        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[6.0, 7.0, 8.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: backward produces gradients
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_backward_produces_gradients() {
        let weight_data = vec![1.0f32; 3]; // [1, 1, 3]
        let bias_data = vec![0.0f32; 1];

        let conv = ConvTranspose1d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 3]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: 3,
            stride: 1,
            padding: 0,
            output_padding: 0,
            training: false,
        };

        let input = leaf(&[1.0f32, 2.0, 3.0], &[1, 1, 3]);
        let output = conv.forward(&input).unwrap();
        // L_out = (3 - 1) * 1 - 0 + 3 + 0 = 5
        assert_eq!(output.shape(), &[1, 1, 5]);
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "ConvTranspose1dBackward");

        let grad_output = t(&[1.0; 5], &[1, 1, 5]);
        let grads = output.grad_fn().unwrap().backward(&grad_output).unwrap();

        // grad_input shape: [1, 1, 3]
        assert!(grads[0].is_some());
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[1, 1, 3]);
        // grad_weight shape: [1, 1, 3]
        assert!(grads[1].is_some());
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[1, 1, 3]);
        // grad_bias shape: [1]
        assert!(grads[2].is_some());
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[1]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose1d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose1d_invalid_ndim() {
        let conv = ConvTranspose1d::<f32>::new(1, 1, 3, 1, 0, 0, false).unwrap();
        let input = t(&[0.0; 4], &[1, 1, 2, 2]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose1d_channel_mismatch() {
        let conv = ConvTranspose1d::<f32>::new(3, 1, 3, 1, 0, 0, false).unwrap();
        let input = t(&[0.0; 10], &[1, 1, 10]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose1d_zero_channels_rejected() {
        assert!(ConvTranspose1d::<f32>::new(0, 1, 3, 1, 0, 0, false).is_err());
        assert!(ConvTranspose1d::<f32>::new(1, 0, 3, 1, 0, 0, false).is_err());
    }

    #[test]
    fn test_conv_transpose1d_output_padding_too_large() {
        assert!(ConvTranspose1d::<f32>::new(1, 1, 3, 2, 0, 2, false).is_err());
    }

    #[test]
    fn test_conv_transpose1d_parameter_count() {
        let conv = ConvTranspose1d::<f32>::new(8, 16, 5, 2, 1, 0, true).unwrap();
        // weight: 8 * 16 * 5 = 640, bias: 16, total: 656
        assert_eq!(conv.num_parameters(), 656);
        assert_eq!(conv.parameters().len(), 2);
    }

    // ===================================================================
    // ConvTranspose3d tests
    // ===================================================================

    // -----------------------------------------------------------------------
    // ConvTranspose3d: output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_output_shape_basic() {
        // Input: [1, 1, 3, 3, 3], kernel 3x3x3, stride 1, padding 0, output_padding 0
        // D_out = (3 - 1) * 1 - 0 + 3 + 0 = 5
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&[0.0; 27], &[1, 1, 3, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5, 5]);
    }

    #[test]
    fn test_conv_transpose3d_output_shape_stride2() {
        // Input: [1, 1, 2, 2, 2], kernel 3x3x3, stride 2, padding 0, output_padding 0
        // D_out = (2 - 1) * 2 - 0 + 3 + 0 = 5
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (2, 2, 2), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&[0.0; 8], &[1, 1, 2, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5, 5]);
    }

    #[test]
    fn test_conv_transpose3d_output_shape_with_padding() {
        // Input: [1, 1, 3, 3, 3], kernel 3x3x3, stride 2, padding 1, output_padding 0
        // D_out = (3 - 1) * 2 - 2 + 3 + 0 = 5
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), false)
                .unwrap();
        let input = t(&[0.0; 27], &[1, 1, 3, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5, 5]);
    }

    #[test]
    fn test_conv_transpose3d_output_shape_with_output_padding() {
        // Input: [1, 1, 3, 3, 3], kernel 3x3x3, stride 2, padding 1, output_padding 1
        // D_out = (3 - 1) * 2 - 2 + 3 + 1 = 6
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (2, 2, 2), (1, 1, 1), (1, 1, 1), false)
                .unwrap();
        let input = t(&[0.0; 27], &[1, 1, 3, 3, 3]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 6, 6, 6]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: stride=2 upsamples (doubles spatial dims)
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_stride2_upsamples() {
        // With stride=2, kernel=2x2x2, padding=0, output_padding=0:
        // D_out = (D - 1) * 2 + 2 = 2 * D
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (2, 2, 2), (2, 2, 2), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&vec![0.0; 4 * 4 * 4], &[1, 1, 4, 4, 4]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 8, 8, 8]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: 1x1x1 kernel correctness
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_1x1x1_kernel() {
        // weight shape: [in=1, out=2, 1, 1, 1]
        let weight_data = vec![3.0f32, 7.0]; // [1, 2, 1, 1, 1]
        let conv = ConvTranspose3d {
            weight: Parameter::from_slice(&weight_data, &[1, 2, 1, 1, 1]).unwrap(),
            bias: None,
            in_channels: 1,
            out_channels: 2,
            kernel_size: (1, 1, 1),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            output_padding: (0, 0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0], &[1, 1, 2, 1, 1]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 2, 2, 1, 1]);
        assert_close(output.data().unwrap(), &[3.0, 6.0, 7.0, 14.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: bias
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_bias() {
        let conv = ConvTranspose3d {
            weight: Parameter::from_slice(&[1.0f32], &[1, 1, 1, 1, 1]).unwrap(),
            bias: Some(Parameter::from_slice(&[5.0f32], &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (1, 1, 1),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            output_padding: (0, 0, 0),
            training: false,
        };

        let input = t(&[1.0, 2.0], &[1, 1, 2, 1, 1]);
        let output = conv.forward(&input).unwrap();
        assert_close(output.data().unwrap(), &[6.0, 7.0], 1e-5);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: backward produces gradients
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_backward_produces_gradients() {
        let weight_data = vec![1.0f32; 2 * 2 * 2]; // [1, 1, 2, 2, 2]
        let bias_data = vec![0.0f32; 1];

        let conv = ConvTranspose3d {
            weight: Parameter::from_slice(&weight_data, &[1, 1, 2, 2, 2]).unwrap(),
            bias: Some(Parameter::from_slice(&bias_data, &[1]).unwrap()),
            in_channels: 1,
            out_channels: 1,
            kernel_size: (2, 2, 2),
            stride: (1, 1, 1),
            padding: (0, 0, 0),
            output_padding: (0, 0, 0),
            training: false,
        };

        // D_out = (2-1)*1 - 0 + 2 + 0 = 3
        let input = leaf(&[1.0f32; 8], &[1, 1, 2, 2, 2]);
        let output = conv.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 3, 3, 3]);
        assert!(output.grad_fn().is_some());
        assert_eq!(output.grad_fn().unwrap().name(), "ConvTranspose3dBackward");

        let grad_output = t(&[1.0; 27], &[1, 1, 3, 3, 3]);
        let grads = output.grad_fn().unwrap().backward(&grad_output).unwrap();

        assert!(grads[0].is_some());
        assert_eq!(grads[0].as_ref().unwrap().shape(), &[1, 1, 2, 2, 2]);
        assert!(grads[1].is_some());
        assert_eq!(grads[1].as_ref().unwrap().shape(), &[1, 1, 2, 2, 2]);
        assert!(grads[2].is_some());
        assert_eq!(grads[2].as_ref().unwrap().shape(), &[1]);
    }

    // -----------------------------------------------------------------------
    // ConvTranspose3d: edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_transpose3d_invalid_ndim() {
        let conv =
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&[0.0; 25], &[1, 1, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose3d_channel_mismatch() {
        let conv =
            ConvTranspose3d::<f32>::new(3, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .unwrap();
        let input = t(&vec![0.0; 5 * 5 * 5], &[1, 1, 5, 5, 5]);
        assert!(conv.forward(&input).is_err());
    }

    #[test]
    fn test_conv_transpose3d_zero_channels_rejected() {
        assert!(
            ConvTranspose3d::<f32>::new(0, 1, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .is_err()
        );
        assert!(
            ConvTranspose3d::<f32>::new(1, 0, (3, 3, 3), (1, 1, 1), (0, 0, 0), (0, 0, 0), false)
                .is_err()
        );
    }

    #[test]
    fn test_conv_transpose3d_output_padding_too_large() {
        assert!(
            ConvTranspose3d::<f32>::new(1, 1, (3, 3, 3), (2, 2, 2), (0, 0, 0), (2, 2, 2), false)
                .is_err()
        );
    }

    #[test]
    fn test_conv_transpose3d_parameter_count() {
        let conv =
            ConvTranspose3d::<f32>::new(8, 16, (3, 3, 3), (2, 2, 2), (1, 1, 1), (0, 0, 0), true)
                .unwrap();
        // weight: 8 * 16 * 3 * 3 * 3 = 3456, bias: 16, total: 3472
        assert_eq!(conv.num_parameters(), 3472);
        assert_eq!(conv.parameters().len(), 2);
    }
}
