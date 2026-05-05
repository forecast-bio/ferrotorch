//! Upsample, interpolation, and vision ops.
//!
//! This module provides spatial resizing and transformation modules for
//! vision workloads:
//!
//! - [`Upsample`] — Upsamples a `[B, C, H, W]` tensor using nearest, bilinear,
//!   or bicubic interpolation.
//! - [`PixelShuffle`] / [`PixelUnshuffle`] — Sub-pixel convolution for
//!   efficient super-resolution (`[B, C*r*r, H, W]` <-> `[B, C, H*r, W*r]`).
//! - [`Fold`] / [`Unfold`] — Sliding-window patch extraction and reconstruction.
//!
//! All autograd-tracked operations attach a `GradFn<T>` when gradient tracking
//! is enabled so reverse-mode differentiation works out of the box.
//!
//! CL-317: Upsample, Interpolation & Vision Ops

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::module::Module;
use crate::parameter::Parameter;

// ===========================================================================
// Interpolation mode
// ===========================================================================

/// Interpolation mode used by [`Upsample`] and [`interpolate`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpolateMode {
    /// Nearest-neighbor interpolation.
    Nearest,
    /// Bilinear interpolation (4-neighbor weighted average).
    Bilinear,
    /// Bicubic interpolation (16-neighbor cubic kernel).
    Bicubic,
}

/// Padding mode for [`grid_sample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridSamplePaddingMode {
    /// Zero-pad outside the input boundary.
    Zeros,
    /// Clamp coordinates to the border of the input.
    Border,
    /// Reflect coordinates at the border.
    Reflection,
}

/// Sampling mode for [`grid_sample`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GridSampleMode {
    /// Bilinear interpolation.
    Bilinear,
    /// Nearest-neighbor sampling.
    Nearest,
}

// ===========================================================================
// Helpers
// ===========================================================================

/// Validate that the input tensor has shape `[B, C, H, W]`.
fn validate_4d<T: Float>(
    input: &Tensor<T>,
    fn_name: &str,
) -> FerrotorchResult<(usize, usize, usize, usize)> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{fn_name} expects 4D input [B, C, H, W], got shape {:?}",
                shape
            ),
        });
    }
    Ok((shape[0], shape[1], shape[2], shape[3]))
}

/// Cubic interpolation kernel (Keys' cubic, a = -0.75).
///
/// For a distance `t` from the pixel center, this returns the weight
/// used in bicubic interpolation. All arithmetic is done in `f64`.
#[inline]
fn cubic_weight(t: f64) -> f64 {
    let abs_t = t.abs();
    let a: f64 = -0.75;

    if abs_t <= 1.0 {
        (a + 2.0) * abs_t * abs_t * abs_t - (a + 3.0) * abs_t * abs_t + 1.0
    } else if abs_t < 2.0 {
        a * abs_t * abs_t * abs_t - 5.0 * a * abs_t * abs_t + 8.0 * a * abs_t - 4.0 * a
    } else {
        0.0
    }
}

/// Compute source coordinate for `align_corners=true`.
///
/// Maps the output index `i` in `[0, out_size-1]` to the input space `[0, in_size-1]`
/// using a linear mapping that aligns the corners.
#[inline]
fn align_corners_coord(i: usize, in_size: usize, out_size: usize) -> f64 {
    if out_size <= 1 {
        return 0.0;
    }
    (i as f64) * ((in_size - 1) as f64) / ((out_size - 1) as f64)
}

/// Compute source coordinate for `align_corners=false`.
///
/// Uses the half-pixel convention: map center of output pixel to input space.
#[inline]
fn half_pixel_coord(i: usize, in_size: usize, out_size: usize) -> f64 {
    (i as f64 + 0.5) * (in_size as f64 / out_size as f64) - 0.5
}

/// Clamp a value to `[0, max]`.
#[inline]
fn clamp_coord(val: isize, max: usize) -> usize {
    if val < 0 {
        0
    } else if val as usize > max {
        max
    } else {
        val as usize
    }
}

// ===========================================================================
// interpolate — functional API
// ===========================================================================

/// Interpolation target size. Exactly one of `size` or `scale_factor` must be
/// provided (the other is `None`).
///
/// CL-317
pub fn interpolate<T: Float>(
    input: &Tensor<T>,
    size: Option<[usize; 2]>,
    scale_factor: Option<[f64; 2]>,
    mode: InterpolateMode,
    align_corners: bool,
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, h_in, w_in) = validate_4d(input, "interpolate")?;

    // Resolve target size.
    let (h_out, w_out) = match (size, scale_factor) {
        (Some(s), None) => (s[0], s[1]),
        (None, Some(sf)) => {
            let h = (h_in as f64 * sf[0]).round() as usize;
            let w = (w_in as f64 * sf[1]).round() as usize;
            if h == 0 || w == 0 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "interpolate: scale_factor {sf:?} with input ({h_in}, {w_in}) produces zero output"
                    ),
                });
            }
            (h, w)
        }
        _ => {
            return Err(FerrotorchError::InvalidArgument {
                message: "interpolate: exactly one of size or scale_factor must be provided".into(),
            });
        }
    };

    if h_out == 0 || w_out == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("interpolate: output size ({h_out}, {w_out}) must be > 0"),
        });
    }

    if mode == InterpolateMode::Nearest && align_corners {
        return Err(FerrotorchError::InvalidArgument {
            message: "interpolate: align_corners is not supported with nearest mode".into(),
        });
    }

    let input_device = input.device();
    let data = input.data_vec()?;

    let total = batch * channels * h_out * w_out;
    let mut output = vec![T::from(0.0).unwrap(); total];

    match mode {
        InterpolateMode::Nearest => {
            nearest_forward(
                &data,
                &mut output,
                batch,
                channels,
                h_in,
                w_in,
                h_out,
                w_out,
            );
        }
        InterpolateMode::Bilinear => {
            bilinear_forward(
                &data,
                &mut output,
                batch,
                channels,
                h_in,
                w_in,
                h_out,
                w_out,
                align_corners,
            );
        }
        InterpolateMode::Bicubic => {
            bicubic_forward(
                &data,
                &mut output,
                batch,
                channels,
                h_in,
                w_in,
                h_out,
                w_out,
                align_corners,
            );
        }
    }

    let out_shape = vec![batch, channels, h_out, w_out];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(InterpolateBackward {
                input: input.clone(),
                h_out,
                w_out,
                mode,
                align_corners,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

// ---------------------------------------------------------------------------
// Forward kernels
// ---------------------------------------------------------------------------

// Internal kernel: argument set is the upsample descriptor
// (B, C, H_in, W_in, H_out, W_out, scale_h, scale_w); a config struct
// would force allocation in the hot interpolate path.
#[allow(clippy::too_many_arguments)]
fn nearest_forward<T: Float>(
    data: &[T],
    output: &mut [T],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) {
    let h_scale = h_in as f64 / h_out as f64;
    let w_scale = w_in as f64 / w_out as f64;

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                let ih = ((oh as f64 * h_scale).floor() as usize).min(h_in - 1);
                for ow in 0..w_out {
                    let iw = ((ow as f64 * w_scale).floor() as usize).min(w_in - 1);
                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    let in_idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                    output[out_idx] = data[in_idx];
                }
            }
        }
    }
}

// Internal kernel: same upsample descriptor as `nearest_forward`.
#[allow(clippy::too_many_arguments)]
fn bilinear_forward<T: Float>(
    data: &[T],
    output: &mut [T],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    align_corners: bool,
) {
    let one = T::from(1.0).unwrap();

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                let src_h = if align_corners {
                    align_corners_coord(oh, h_in, h_out)
                } else {
                    half_pixel_coord(oh, h_in, h_out)
                };

                let h0 = src_h.floor() as isize;
                let h1 = h0 + 1;
                let th = T::from(src_h - h0 as f64).unwrap();

                for ow in 0..w_out {
                    let src_w = if align_corners {
                        align_corners_coord(ow, w_in, w_out)
                    } else {
                        half_pixel_coord(ow, w_in, w_out)
                    };

                    let w0 = src_w.floor() as isize;
                    let w1 = w0 + 1;
                    let tw = T::from(src_w - w0 as f64).unwrap();

                    let ch0 = clamp_coord(h0, h_in - 1);
                    let ch1 = clamp_coord(h1, h_in - 1);
                    let cw0 = clamp_coord(w0, w_in - 1);
                    let cw1 = clamp_coord(w1, w_in - 1);

                    let base = (b * channels + c) * h_in;
                    let v00 = data[(base + ch0) * w_in + cw0];
                    let v01 = data[(base + ch0) * w_in + cw1];
                    let v10 = data[(base + ch1) * w_in + cw0];
                    let v11 = data[(base + ch1) * w_in + cw1];

                    let val = v00 * (one - th) * (one - tw)
                        + v01 * (one - th) * tw
                        + v10 * th * (one - tw)
                        + v11 * th * tw;

                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    output[out_idx] = val;
                }
            }
        }
    }
}

// Internal kernel: same upsample descriptor as `nearest_forward`.
#[allow(clippy::too_many_arguments)]
fn bicubic_forward<T: Float>(
    data: &[T],
    output: &mut [T],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    align_corners: bool,
) {
    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                let src_h = if align_corners {
                    align_corners_coord(oh, h_in, h_out)
                } else {
                    half_pixel_coord(oh, h_in, h_out)
                };

                let h_floor = src_h.floor() as isize;
                let frac_h = src_h - h_floor as f64;

                // Precompute 4 vertical kernel weights.
                let wh: [T; 4] = [
                    T::from(cubic_weight(frac_h + 1.0)).unwrap(),
                    T::from(cubic_weight(frac_h)).unwrap(),
                    T::from(cubic_weight(frac_h - 1.0)).unwrap(),
                    T::from(cubic_weight(frac_h - 2.0)).unwrap(),
                ];

                for ow in 0..w_out {
                    let src_w = if align_corners {
                        align_corners_coord(ow, w_in, w_out)
                    } else {
                        half_pixel_coord(ow, w_in, w_out)
                    };

                    let w_floor = src_w.floor() as isize;
                    let frac_w = src_w - w_floor as f64;

                    let ww: [T; 4] = [
                        T::from(cubic_weight(frac_w + 1.0)).unwrap(),
                        T::from(cubic_weight(frac_w)).unwrap(),
                        T::from(cubic_weight(frac_w - 1.0)).unwrap(),
                        T::from(cubic_weight(frac_w - 2.0)).unwrap(),
                    ];

                    let mut val = T::from(0.0).unwrap();
                    let base = (b * channels + c) * h_in;

                    for dy in 0..4isize {
                        let iy = clamp_coord(h_floor - 1 + dy, h_in - 1);
                        for dx in 0..4isize {
                            let ix = clamp_coord(w_floor - 1 + dx, w_in - 1);
                            let pixel = data[(base + iy) * w_in + ix];
                            val += pixel * wh[dy as usize] * ww[dx as usize];
                        }
                    }

                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    output[out_idx] = val;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backward for interpolate
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct InterpolateBackward<T: Float> {
    input: Tensor<T>,
    h_out: usize,
    w_out: usize,
    mode: InterpolateMode,
    align_corners: bool,
}

impl<T: Float> GradFn<T> for InterpolateBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let in_shape = self.input.shape();
        let (batch, channels, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let h_out = self.h_out;
        let w_out = self.w_out;

        let go_data = grad_output.data_vec()?;
        let mut grad_input = vec![T::from(0.0).unwrap(); batch * channels * h_in * w_in];

        match self.mode {
            InterpolateMode::Nearest => {
                nearest_backward(
                    &go_data,
                    &mut grad_input,
                    batch,
                    channels,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                );
            }
            InterpolateMode::Bilinear => {
                bilinear_backward(
                    &go_data,
                    &mut grad_input,
                    batch,
                    channels,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    self.align_corners,
                );
            }
            InterpolateMode::Bicubic => {
                bicubic_backward(
                    &go_data,
                    &mut grad_input,
                    batch,
                    channels,
                    h_in,
                    w_in,
                    h_out,
                    w_out,
                    self.align_corners,
                );
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
        "InterpolateBackward"
    }
}

// Internal kernel: adjoint of `nearest_forward`; same descriptor.
#[allow(clippy::too_many_arguments)]
fn nearest_backward<T: Float>(
    go: &[T],
    grad_input: &mut [T],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
) {
    let h_scale = h_in as f64 / h_out as f64;
    let w_scale = w_in as f64 / w_out as f64;

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                let ih = ((oh as f64 * h_scale).floor() as usize).min(h_in - 1);
                for ow in 0..w_out {
                    let iw = ((ow as f64 * w_scale).floor() as usize).min(w_in - 1);
                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    let in_idx = ((b * channels + c) * h_in + ih) * w_in + iw;
                    grad_input[in_idx] += go[out_idx];
                }
            }
        }
    }
}

// Internal kernel: adjoint of `bilinear_forward`; same descriptor.
#[allow(clippy::too_many_arguments)]
fn bilinear_backward<T: Float>(
    go: &[T],
    grad_input: &mut [T],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    align_corners: bool,
) {
    let one = T::from(1.0).unwrap();

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                let src_h = if align_corners {
                    align_corners_coord(oh, h_in, h_out)
                } else {
                    half_pixel_coord(oh, h_in, h_out)
                };

                let h0 = src_h.floor() as isize;
                let h1 = h0 + 1;
                let th = T::from(src_h - h0 as f64).unwrap();

                for ow in 0..w_out {
                    let src_w = if align_corners {
                        align_corners_coord(ow, w_in, w_out)
                    } else {
                        half_pixel_coord(ow, w_in, w_out)
                    };

                    let w0 = src_w.floor() as isize;
                    let w1 = w0 + 1;
                    let tw = T::from(src_w - w0 as f64).unwrap();

                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    let g = go[out_idx];

                    let ch0 = clamp_coord(h0, h_in - 1);
                    let ch1 = clamp_coord(h1, h_in - 1);
                    let cw0 = clamp_coord(w0, w_in - 1);
                    let cw1 = clamp_coord(w1, w_in - 1);

                    let base = (b * channels + c) * h_in;

                    grad_input[(base + ch0) * w_in + cw0] += g * (one - th) * (one - tw);
                    grad_input[(base + ch0) * w_in + cw1] += g * (one - th) * tw;
                    grad_input[(base + ch1) * w_in + cw0] += g * th * (one - tw);
                    grad_input[(base + ch1) * w_in + cw1] += g * th * tw;
                }
            }
        }
    }
}

// Internal kernel: adjoint of `bicubic_forward`; same descriptor.
#[allow(clippy::too_many_arguments)]
fn bicubic_backward<T: Float>(
    go: &[T],
    grad_input: &mut [T],
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    h_out: usize,
    w_out: usize,
    align_corners: bool,
) {
    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                let src_h: f64 = if align_corners {
                    align_corners_coord(oh, h_in, h_out)
                } else {
                    half_pixel_coord(oh, h_in, h_out)
                };

                let h_floor = src_h.floor() as isize;
                let frac_h = src_h - h_floor as f64;

                let wh: [T; 4] = [
                    T::from(cubic_weight(frac_h + 1.0)).unwrap(),
                    T::from(cubic_weight(frac_h)).unwrap(),
                    T::from(cubic_weight(frac_h - 1.0)).unwrap(),
                    T::from(cubic_weight(frac_h - 2.0)).unwrap(),
                ];

                for ow in 0..w_out {
                    let src_w: f64 = if align_corners {
                        align_corners_coord(ow, w_in, w_out)
                    } else {
                        half_pixel_coord(ow, w_in, w_out)
                    };

                    let w_floor = src_w.floor() as isize;
                    let frac_w = src_w - w_floor as f64;

                    let ww: [T; 4] = [
                        T::from(cubic_weight(frac_w + 1.0)).unwrap(),
                        T::from(cubic_weight(frac_w)).unwrap(),
                        T::from(cubic_weight(frac_w - 1.0)).unwrap(),
                        T::from(cubic_weight(frac_w - 2.0)).unwrap(),
                    ];

                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    let g = go[out_idx];
                    let base = (b * channels + c) * h_in;

                    for dy in 0..4isize {
                        let iy = clamp_coord(h_floor - 1 + dy, h_in - 1);
                        for dx in 0..4isize {
                            let ix = clamp_coord(w_floor - 1 + dx, w_in - 1);
                            grad_input[(base + iy) * w_in + ix] +=
                                g * wh[dy as usize] * ww[dx as usize];
                        }
                    }
                }
            }
        }
    }
}

// ===========================================================================
// Upsample module
// ===========================================================================

/// Upsamples a `[B, C, H, W]` tensor to a target spatial size.
///
/// Supports nearest, bilinear, and bicubic interpolation. This is the
/// module-based wrapper around [`interpolate`].
///
/// CL-317
#[derive(Debug, Clone)]
pub struct Upsample {
    /// Target output spatial size `[H, W]`. If `None`, `scale_factor` is used.
    pub size: Option<[usize; 2]>,
    /// Scaling factor `[scale_h, scale_w]`. If `None`, `size` is used.
    pub scale_factor: Option<[f64; 2]>,
    /// Interpolation mode.
    pub mode: InterpolateMode,
    /// Whether to align corners (bilinear/bicubic only).
    pub align_corners: bool,
}

impl Upsample {
    /// Create a new `Upsample` with target `size`.
    pub fn new(size: [usize; 2], mode: InterpolateMode) -> Self {
        Self {
            size: Some(size),
            scale_factor: None,
            mode,
            align_corners: false,
        }
    }

    /// Create a new `Upsample` with a `scale_factor`.
    pub fn with_scale_factor(scale_factor: [f64; 2], mode: InterpolateMode) -> Self {
        Self {
            size: None,
            scale_factor: Some(scale_factor),
            mode,
            align_corners: false,
        }
    }

    /// Set `align_corners` (meaningful for bilinear/bicubic only).
    pub fn align_corners(mut self, align_corners: bool) -> Self {
        self.align_corners = align_corners;
        self
    }
}

impl<T: Float> Module<T> for Upsample {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        interpolate(
            input,
            self.size,
            self.scale_factor,
            self.mode,
            self.align_corners,
        )
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

// ===========================================================================
// grid_sample
// ===========================================================================

/// Samples `input` at spatial locations specified by `grid`.
///
/// This implements the spatial transformer network sampling operation.
///
/// # Shapes
///
/// - `input`: `[B, C, H_in, W_in]`
/// - `grid`: `[B, H_out, W_out, 2]` — normalized coordinates in `[-1, 1]`
/// - **returns**: `[B, C, H_out, W_out]`
///
/// CL-317
pub fn grid_sample<T: Float>(
    input: &Tensor<T>,
    grid: &Tensor<T>,
    mode: GridSampleMode,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, h_in, w_in) = validate_4d(input, "grid_sample")?;

    let grid_shape = grid.shape();
    if grid_shape.len() != 4 || grid_shape[3] != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "grid_sample: grid must be [B, H_out, W_out, 2], got {:?}",
                grid_shape
            ),
        });
    }
    if grid_shape[0] != batch {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "grid_sample: batch mismatch between input ({batch}) and grid ({})",
                grid_shape[0]
            ),
        });
    }
    let h_out = grid_shape[1];
    let w_out = grid_shape[2];

    let input_device = input.device();
    let in_data = input.data_vec()?;
    let grid_data = grid.data_vec()?;

    let total = batch * channels * h_out * w_out;
    let mut output = vec![T::from(0.0).unwrap(); total];

    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();
    let zero = T::from(0.0).unwrap();

    for b in 0..batch {
        for oh in 0..h_out {
            for ow in 0..w_out {
                let grid_base = ((b * h_out + oh) * w_out + ow) * 2;
                let gx = grid_data[grid_base]; // normalized x
                let gy = grid_data[grid_base + 1]; // normalized y

                // Denormalize grid coordinates from [-1, 1] to pixel space.
                let (src_x, src_y) = if align_corners {
                    let sx = (gx + one) * T::from(w_in - 1).unwrap() / two;
                    let sy = (gy + one) * T::from(h_in - 1).unwrap() / two;
                    (sx, sy)
                } else {
                    let sx = ((gx + one) * T::from(w_in).unwrap() - one) / two;
                    let sy = ((gy + one) * T::from(h_in).unwrap() - one) / two;
                    (sx, sy)
                };

                for c in 0..channels {
                    let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                    let in_base = (b * channels + c) * h_in;

                    match mode {
                        GridSampleMode::Nearest => {
                            let ix = src_x.to_f64().unwrap().round() as isize;
                            let iy = src_y.to_f64().unwrap().round() as isize;
                            let (ix, iy) = apply_padding_mode(ix, iy, w_in, h_in, padding_mode);

                            if ix >= 0 && ix < w_in as isize && iy >= 0 && iy < h_in as isize {
                                output[out_idx] =
                                    in_data[(in_base + iy as usize) * w_in + ix as usize];
                            }
                            // else stays zero (for Zeros padding)
                        }
                        GridSampleMode::Bilinear => {
                            let sx = src_x.to_f64().unwrap();
                            let sy = src_y.to_f64().unwrap();
                            let x0 = sx.floor() as isize;
                            let y0 = sy.floor() as isize;
                            let x1 = x0 + 1;
                            let y1 = y0 + 1;
                            let tx = T::from(sx - x0 as f64).unwrap();
                            let ty = T::from(sy - y0 as f64).unwrap();

                            let get_pixel = |iy: isize, ix: isize| -> T {
                                let (ix, iy) = apply_padding_mode(ix, iy, w_in, h_in, padding_mode);
                                if ix >= 0 && ix < w_in as isize && iy >= 0 && iy < h_in as isize {
                                    in_data[(in_base + iy as usize) * w_in + ix as usize]
                                } else {
                                    zero
                                }
                            };

                            let v00 = get_pixel(y0, x0);
                            let v01 = get_pixel(y0, x1);
                            let v10 = get_pixel(y1, x0);
                            let v11 = get_pixel(y1, x1);

                            output[out_idx] = v00 * (one - ty) * (one - tx)
                                + v01 * (one - ty) * tx
                                + v10 * ty * (one - tx)
                                + v11 * ty * tx;
                        }
                    }
                }
            }
        }
    }

    let out_shape = vec![batch, channels, h_out, w_out];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && (input.requires_grad() || grid.requires_grad()) {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(GridSampleBackward {
                input: input.clone(),
                grid: grid.clone(),
                mode,
                padding_mode,
                align_corners,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Apply padding mode to grid coordinates.
fn apply_padding_mode(
    ix: isize,
    iy: isize,
    w: usize,
    h: usize,
    padding_mode: GridSamplePaddingMode,
) -> (isize, isize) {
    match padding_mode {
        GridSamplePaddingMode::Zeros => (ix, iy),
        GridSamplePaddingMode::Border => {
            let cx = ix.max(0).min(w as isize - 1);
            let cy = iy.max(0).min(h as isize - 1);
            (cx, cy)
        }
        GridSamplePaddingMode::Reflection => {
            let reflect = |v: isize, size: usize| -> isize {
                if size <= 1 {
                    return 0;
                }
                let max = size as isize - 1;
                let mut v = v;
                if v < 0 {
                    v = -v;
                }
                // Fold via period 2*(size-1)
                let period = 2 * max;
                v %= period;
                if v > max {
                    v = period - v;
                }
                v
            };
            (reflect(ix, w), reflect(iy, h))
        }
    }
}

// ---------------------------------------------------------------------------
// GridSample backward
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct GridSampleBackward<T: Float> {
    input: Tensor<T>,
    grid: Tensor<T>,
    mode: GridSampleMode,
    padding_mode: GridSamplePaddingMode,
    align_corners: bool,
}

impl<T: Float> GradFn<T> for GridSampleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let in_shape = self.input.shape();
        let (batch, channels, h_in, w_in) = (in_shape[0], in_shape[1], in_shape[2], in_shape[3]);
        let grid_shape = self.grid.shape();
        let h_out = grid_shape[1];
        let w_out = grid_shape[2];

        let go_data = grad_output.data_vec()?;
        let in_data = self.input.data_vec()?;
        let grid_data = self.grid.data_vec()?;

        let one = T::from(1.0).unwrap();
        let two = T::from(2.0).unwrap();
        let zero = T::from(0.0).unwrap();

        let grad_input_needed = self.input.requires_grad();
        let grad_grid_needed = self.grid.requires_grad();

        let mut grad_input = if grad_input_needed {
            vec![zero; batch * channels * h_in * w_in]
        } else {
            vec![]
        };
        let mut grad_grid = if grad_grid_needed {
            vec![zero; batch * h_out * w_out * 2]
        } else {
            vec![]
        };

        for b in 0..batch {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let grid_base = ((b * h_out + oh) * w_out + ow) * 2;
                    let gx = grid_data[grid_base];
                    let gy = grid_data[grid_base + 1];

                    let (src_x, src_y) = if self.align_corners {
                        let sx = (gx + one) * T::from(w_in - 1).unwrap() / two;
                        let sy = (gy + one) * T::from(h_in - 1).unwrap() / two;
                        (sx, sy)
                    } else {
                        let sx = ((gx + one) * T::from(w_in).unwrap() - one) / two;
                        let sy = ((gy + one) * T::from(h_in).unwrap() - one) / two;
                        (sx, sy)
                    };

                    match self.mode {
                        GridSampleMode::Bilinear => {
                            let sx = src_x.to_f64().unwrap();
                            let sy = src_y.to_f64().unwrap();
                            let x0 = sx.floor() as isize;
                            let y0 = sy.floor() as isize;
                            let x1 = x0 + 1;
                            let y1 = y0 + 1;
                            let tx = T::from(sx - x0 as f64).unwrap();
                            let ty = T::from(sy - y0 as f64).unwrap();

                            let get_clamped = |iy: isize, ix: isize| -> (isize, isize) {
                                apply_padding_mode(ix, iy, w_in, h_in, self.padding_mode)
                            };

                            for c in 0..channels {
                                let out_idx = ((b * channels + c) * h_out + oh) * w_out + ow;
                                let g = go_data[out_idx];
                                let in_base = (b * channels + c) * h_in;

                                // Gradient w.r.t. input
                                if grad_input_needed {
                                    let coords = [
                                        (y0, x0, (one - ty) * (one - tx)),
                                        (y0, x1, (one - ty) * tx),
                                        (y1, x0, ty * (one - tx)),
                                        (y1, x1, ty * tx),
                                    ];
                                    for (iy, ix, w) in coords {
                                        let (ix, iy) = get_clamped(iy, ix);
                                        if ix >= 0
                                            && ix < w_in as isize
                                            && iy >= 0
                                            && iy < h_in as isize
                                        {
                                            grad_input
                                                [(in_base + iy as usize) * w_in + ix as usize] +=
                                                g * w;
                                        }
                                    }
                                }

                                // Gradient w.r.t. grid
                                if grad_grid_needed {
                                    let get_pixel = |iy: isize, ix: isize| -> T {
                                        let (ix, iy) = get_clamped(iy, ix);
                                        if ix >= 0
                                            && ix < w_in as isize
                                            && iy >= 0
                                            && iy < h_in as isize
                                        {
                                            in_data[(in_base + iy as usize) * w_in + ix as usize]
                                        } else {
                                            zero
                                        }
                                    };

                                    let v00 = get_pixel(y0, x0);
                                    let v01 = get_pixel(y0, x1);
                                    let v10 = get_pixel(y1, x0);
                                    let v11 = get_pixel(y1, x1);

                                    // dout/d(src_x) = (1-ty)*(v01-v00) + ty*(v11-v10)
                                    let dout_dsx = (one - ty) * (v01 - v00) + ty * (v11 - v10);
                                    // dout/d(src_y) = (1-tx)*(v10-v00) + tx*(v11-v01)
                                    let dout_dsy = (one - tx) * (v10 - v00) + tx * (v11 - v01);

                                    // d(src_x)/d(gx)
                                    let dsx_dgx = if self.align_corners {
                                        T::from(w_in - 1).unwrap() / two
                                    } else {
                                        T::from(w_in).unwrap() / two
                                    };
                                    let dsy_dgy = if self.align_corners {
                                        T::from(h_in - 1).unwrap() / two
                                    } else {
                                        T::from(h_in).unwrap() / two
                                    };

                                    grad_grid[grid_base] += g * dout_dsx * dsx_dgx;
                                    grad_grid[grid_base + 1] += g * dout_dsy * dsy_dgy;
                                }
                            }
                        }
                        GridSampleMode::Nearest => {
                            // Nearest has zero gradient w.r.t. grid coordinates.
                            // Only accumulate gradient for input.
                            if grad_input_needed {
                                let ix = src_x.to_f64().unwrap().round() as isize;
                                let iy = src_y.to_f64().unwrap().round() as isize;
                                let (ix, iy) =
                                    apply_padding_mode(ix, iy, w_in, h_in, self.padding_mode);

                                if ix >= 0 && ix < w_in as isize && iy >= 0 && iy < h_in as isize {
                                    for c in 0..channels {
                                        let out_idx =
                                            ((b * channels + c) * h_out + oh) * w_out + ow;
                                        let in_base = (b * channels + c) * h_in;
                                        grad_input[(in_base + iy as usize) * w_in + ix as usize] +=
                                            go_data[out_idx];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        let gi = if grad_input_needed {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_input),
                self.input.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        let gg = if grad_grid_needed {
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_grid),
                self.grid.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![gi, gg])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input, &self.grid]
    }

    fn name(&self) -> &'static str {
        "GridSampleBackward"
    }
}

// ===========================================================================
// affine_grid
// ===========================================================================

/// Generate a 2D affine grid for use with [`grid_sample`].
///
/// # Shapes
///
/// - `theta`: `[B, 2, 3]` — 2D affine transformation matrices.
/// - `size`: `[B, C, H, W]` — the target output size.
/// - **returns**: `[B, H, W, 2]` — normalized grid coordinates.
///
/// CL-317
pub fn affine_grid<T: Float>(
    theta: &Tensor<T>,
    size: [usize; 4],
    align_corners: bool,
) -> FerrotorchResult<Tensor<T>> {
    let theta_shape = theta.shape();
    if theta_shape.len() != 3 || theta_shape[1] != 2 || theta_shape[2] != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "affine_grid: theta must be [B, 2, 3], got {:?}",
                theta_shape
            ),
        });
    }
    let batch = theta_shape[0];
    if size[0] != batch {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "affine_grid: batch mismatch: theta batch {batch}, size batch {}",
                size[0]
            ),
        });
    }

    let h = size[2];
    let w = size[3];
    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();

    let theta_data = theta.data_vec()?;
    let theta_device = theta.device();
    let total = batch * h * w * 2;
    let mut grid = vec![T::from(0.0).unwrap(); total];

    for b in 0..batch {
        let t_base = b * 6;
        let t00 = theta_data[t_base];
        let t01 = theta_data[t_base + 1];
        let t02 = theta_data[t_base + 2];
        let t10 = theta_data[t_base + 3];
        let t11 = theta_data[t_base + 4];
        let t12 = theta_data[t_base + 5];

        for iy in 0..h {
            let y_norm = if align_corners {
                if h <= 1 {
                    T::from(0.0).unwrap()
                } else {
                    two * T::from(iy).unwrap() / T::from(h - 1).unwrap() - one
                }
            } else {
                (two * T::from(iy).unwrap() + one) / T::from(h).unwrap() - one
            };

            for ix in 0..w {
                let x_norm = if align_corners {
                    if w <= 1 {
                        T::from(0.0).unwrap()
                    } else {
                        two * T::from(ix).unwrap() / T::from(w - 1).unwrap() - one
                    }
                } else {
                    (two * T::from(ix).unwrap() + one) / T::from(w).unwrap() - one
                };

                let out_base = ((b * h + iy) * w + ix) * 2;
                grid[out_base] = t00 * x_norm + t01 * y_norm + t02;
                grid[out_base + 1] = t10 * x_norm + t11 * y_norm + t12;
            }
        }
    }

    let out_shape = vec![batch, h, w, 2];
    let storage = TensorStorage::cpu(grid);

    if is_grad_enabled() && theta.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(AffineGridBackward {
                theta: theta.clone(),
                size,
                align_corners,
            }),
        )?
        .to(theta_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(theta_device)
    }
}

#[derive(Debug)]
struct AffineGridBackward<T: Float> {
    theta: Tensor<T>,
    size: [usize; 4],
    align_corners: bool,
}

impl<T: Float> GradFn<T> for AffineGridBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.theta.requires_grad() {
            return Ok(vec![None]);
        }

        let batch = self.size[0];
        let h = self.size[2];
        let w = self.size[3];
        let one = T::from(1.0).unwrap();
        let two = T::from(2.0).unwrap();
        let zero = T::from(0.0).unwrap();

        let go_data = grad_output.data_vec()?;
        let mut grad_theta = vec![zero; batch * 6];

        for b in 0..batch {
            for iy in 0..h {
                let y_norm = if self.align_corners {
                    if h <= 1 {
                        zero
                    } else {
                        two * T::from(iy).unwrap() / T::from(h - 1).unwrap() - one
                    }
                } else {
                    (two * T::from(iy).unwrap() + one) / T::from(h).unwrap() - one
                };

                for ix in 0..w {
                    let x_norm = if self.align_corners {
                        if w <= 1 {
                            zero
                        } else {
                            two * T::from(ix).unwrap() / T::from(w - 1).unwrap() - one
                        }
                    } else {
                        (two * T::from(ix).unwrap() + one) / T::from(w).unwrap() - one
                    };

                    let go_base = ((b * h + iy) * w + ix) * 2;
                    let gx = go_data[go_base];
                    let gy = go_data[go_base + 1];

                    let t_base = b * 6;
                    // d(grid_x) / d(t00) = x_norm, d(grid_x) / d(t01) = y_norm, d(grid_x) / d(t02) = 1
                    grad_theta[t_base] += gx * x_norm;
                    grad_theta[t_base + 1] += gx * y_norm;
                    grad_theta[t_base + 2] += gx;
                    // d(grid_y) / d(t10) = x_norm, d(grid_y) / d(t11) = y_norm, d(grid_y) / d(t12) = 1
                    grad_theta[t_base + 3] += gy * x_norm;
                    grad_theta[t_base + 4] += gy * y_norm;
                    grad_theta[t_base + 5] += gy;
                }
            }
        }

        let grad_tensor = Tensor::from_storage(
            TensorStorage::cpu(grad_theta),
            self.theta.shape().to_vec(),
            false,
        )?;
        Ok(vec![Some(grad_tensor)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.theta]
    }

    fn name(&self) -> &'static str {
        "AffineGridBackward"
    }
}

// ===========================================================================
// PixelShuffle / PixelUnshuffle
// ===========================================================================

/// Rearranges `[B, C*r*r, H, W]` to `[B, C, H*r, W*r]` (sub-pixel convolution).
///
/// Used in super-resolution networks to upsample feature maps without
/// transposed convolutions.
///
/// CL-317
#[derive(Debug, Clone, Copy)]
pub struct PixelShuffle {
    /// Upscale factor.
    pub upscale_factor: usize,
}

impl PixelShuffle {
    pub fn new(upscale_factor: usize) -> Self {
        Self { upscale_factor }
    }
}

impl<T: Float> Module<T> for PixelShuffle {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        pixel_shuffle(input, self.upscale_factor)
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

/// Rearranges `[B, C, H*r, W*r]` to `[B, C*r*r, H, W]` (inverse sub-pixel convolution).
///
/// CL-317
#[derive(Debug, Clone, Copy)]
pub struct PixelUnshuffle {
    /// Downscale factor.
    pub downscale_factor: usize,
}

impl PixelUnshuffle {
    pub fn new(downscale_factor: usize) -> Self {
        Self { downscale_factor }
    }
}

impl<T: Float> Module<T> for PixelUnshuffle {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        pixel_unshuffle(input, self.downscale_factor)
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

/// Functional pixel shuffle: `[B, C*r*r, H, W]` -> `[B, C, H*r, W*r]`.
///
/// CL-317
pub fn pixel_shuffle<T: Float>(
    input: &Tensor<T>,
    upscale_factor: usize,
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels_in, h, w) = validate_4d(input, "pixel_shuffle")?;
    let r = upscale_factor;

    if r == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "pixel_shuffle: upscale_factor must be > 0".into(),
        });
    }
    if channels_in % (r * r) != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "pixel_shuffle: channels ({channels_in}) must be divisible by r^2 ({})",
                r * r
            ),
        });
    }

    let c_out = channels_in / (r * r);
    let h_out = h * r;
    let w_out = w * r;

    let input_device = input.device();
    let data = input.data_vec()?;

    let total = batch * c_out * h_out * w_out;
    let mut output = vec![T::from(0.0).unwrap(); total];

    // Layout: input channels are organized as [c, r_h, r_w] sub-groups.
    for b in 0..batch {
        for c in 0..c_out {
            for ih in 0..h {
                for iw in 0..w {
                    for rh in 0..r {
                        for rw in 0..r {
                            let in_c = c * r * r + rh * r + rw;
                            let in_idx = ((b * channels_in + in_c) * h + ih) * w + iw;

                            let oh = ih * r + rh;
                            let ow_pos = iw * r + rw;
                            let out_idx = ((b * c_out + c) * h_out + oh) * w_out + ow_pos;

                            output[out_idx] = data[in_idx];
                        }
                    }
                }
            }
        }
    }

    let out_shape = vec![batch, c_out, h_out, w_out];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(PixelShuffleBackward {
                input: input.clone(),
                upscale_factor: r,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Functional pixel unshuffle: `[B, C, H*r, W*r]` -> `[B, C*r*r, H, W]`.
///
/// CL-317
pub fn pixel_unshuffle<T: Float>(
    input: &Tensor<T>,
    downscale_factor: usize,
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, h_in, w_in) = validate_4d(input, "pixel_unshuffle")?;
    let r = downscale_factor;

    if r == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "pixel_unshuffle: downscale_factor must be > 0".into(),
        });
    }
    if h_in % r != 0 || w_in % r != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "pixel_unshuffle: spatial dims ({h_in}, {w_in}) must be divisible by r={r}"
            ),
        });
    }

    let h_out = h_in / r;
    let w_out = w_in / r;
    let c_out = channels * r * r;

    let input_device = input.device();
    let data = input.data_vec()?;

    let total = batch * c_out * h_out * w_out;
    let mut output = vec![T::from(0.0).unwrap(); total];

    for b in 0..batch {
        for c in 0..channels {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    for rh in 0..r {
                        for rw in 0..r {
                            let in_h = oh * r + rh;
                            let in_w = ow * r + rw;
                            let in_idx = ((b * channels + c) * h_in + in_h) * w_in + in_w;

                            let out_c = c * r * r + rh * r + rw;
                            let out_idx = ((b * c_out + out_c) * h_out + oh) * w_out + ow;

                            output[out_idx] = data[in_idx];
                        }
                    }
                }
            }
        }
    }

    let out_shape = vec![batch, c_out, h_out, w_out];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(PixelUnshuffleBackward {
                input: input.clone(),
                downscale_factor: r,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

#[derive(Debug)]
struct PixelShuffleBackward<T: Float> {
    input: Tensor<T>,
    upscale_factor: usize,
}

impl<T: Float> GradFn<T> for PixelShuffleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Backward of pixel_shuffle is pixel_unshuffle.
        let grad_input = pixel_unshuffle(grad_output, self.upscale_factor)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "PixelShuffleBackward"
    }
}

#[derive(Debug)]
struct PixelUnshuffleBackward<T: Float> {
    input: Tensor<T>,
    downscale_factor: usize,
}

impl<T: Float> GradFn<T> for PixelUnshuffleBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Backward of pixel_unshuffle is pixel_shuffle.
        let grad_input = pixel_shuffle(grad_output, self.downscale_factor)?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "PixelUnshuffleBackward"
    }
}

// ===========================================================================
// Unfold / Fold
// ===========================================================================

/// Extracts sliding-window patches from a `[B, C, H, W]` tensor and
/// reshapes them into columns: output `[B, C * kH * kW, L]` where
/// `L = out_h * out_w`.
///
/// This is the im2col operation used in efficient convolution implementations.
///
/// CL-317
#[derive(Debug, Clone, Copy)]
pub struct Unfold {
    pub kernel_size: [usize; 2],
    pub dilation: [usize; 2],
    pub padding: [usize; 2],
    pub stride: [usize; 2],
}

impl Unfold {
    pub fn new(
        kernel_size: [usize; 2],
        dilation: [usize; 2],
        padding: [usize; 2],
        stride: [usize; 2],
    ) -> Self {
        Self {
            kernel_size,
            dilation,
            padding,
            stride,
        }
    }
}

impl<T: Float> Module<T> for Unfold {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        unfold(
            input,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )
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

/// Reconstructs a `[B, C, H, W]` tensor from sliding-window columns
/// `[B, C * kH * kW, L]`, the inverse of [`Unfold`].
///
/// `output_size` specifies the original `[H, W]` spatial dimensions.
///
/// CL-317
#[derive(Debug, Clone, Copy)]
pub struct Fold {
    pub output_size: [usize; 2],
    pub kernel_size: [usize; 2],
    pub dilation: [usize; 2],
    pub padding: [usize; 2],
    pub stride: [usize; 2],
}

impl Fold {
    pub fn new(
        output_size: [usize; 2],
        kernel_size: [usize; 2],
        dilation: [usize; 2],
        padding: [usize; 2],
        stride: [usize; 2],
    ) -> Self {
        Self {
            output_size,
            kernel_size,
            dilation,
            padding,
            stride,
        }
    }
}

impl<T: Float> Module<T> for Fold {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        fold(
            input,
            self.output_size,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )
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

/// Compute the output spatial dim for unfold given the parameters.
#[inline]
fn unfold_output_size(
    input_size: usize,
    kernel_size: usize,
    dilation: usize,
    padding: usize,
    stride: usize,
) -> usize {
    (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1
}

/// Functional unfold (im2col): `[B, C, H, W]` -> `[B, C*kH*kW, L]`.
///
/// CL-317
pub fn unfold<T: Float>(
    input: &Tensor<T>,
    kernel_size: [usize; 2],
    dilation: [usize; 2],
    padding: [usize; 2],
    stride: [usize; 2],
) -> FerrotorchResult<Tensor<T>> {
    let (batch, channels, h, w) = validate_4d(input, "unfold")?;

    if kernel_size[0] == 0
        || kernel_size[1] == 0
        || stride[0] == 0
        || stride[1] == 0
        || dilation[0] == 0
        || dilation[1] == 0
    {
        return Err(FerrotorchError::InvalidArgument {
            message: "unfold: kernel_size, stride, dilation must all be > 0".into(),
        });
    }

    let out_h = unfold_output_size(h, kernel_size[0], dilation[0], padding[0], stride[0]);
    let out_w = unfold_output_size(w, kernel_size[1], dilation[1], padding[1], stride[1]);
    let l = out_h * out_w;
    let k = channels * kernel_size[0] * kernel_size[1];

    let input_device = input.device();
    let data = input.data_vec()?;

    let total = batch * k * l;
    let mut output = vec![T::from(0.0).unwrap(); total];

    for b in 0..batch {
        for c in 0..channels {
            for kh in 0..kernel_size[0] {
                for kw in 0..kernel_size[1] {
                    let k_idx = (c * kernel_size[0] + kh) * kernel_size[1] + kw;
                    for oh in 0..out_h {
                        for ow in 0..out_w {
                            let ih = oh * stride[0] + kh * dilation[0];
                            let iw = ow * stride[1] + kw * dilation[1];
                            let ih = ih as isize - padding[0] as isize;
                            let iw = iw as isize - padding[1] as isize;

                            let l_idx = oh * out_w + ow;
                            let out_idx = (b * k + k_idx) * l + l_idx;

                            if ih >= 0 && ih < h as isize && iw >= 0 && iw < w as isize {
                                let in_idx =
                                    ((b * channels + c) * h + ih as usize) * w + iw as usize;
                                output[out_idx] = data[in_idx];
                            }
                            // else stays zero (padded)
                        }
                    }
                }
            }
        }
    }

    let out_shape = vec![batch, k, l];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(UnfoldBackward {
                input: input.clone(),
                kernel_size,
                dilation,
                padding,
                stride,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

/// Functional fold (col2im): `[B, C*kH*kW, L]` -> `[B, C, H, W]`.
///
/// CL-317
pub fn fold<T: Float>(
    input: &Tensor<T>,
    output_size: [usize; 2],
    kernel_size: [usize; 2],
    dilation: [usize; 2],
    padding: [usize; 2],
    stride: [usize; 2],
) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if shape.len() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fold expects 3D input [B, C*kH*kW, L], got shape {:?}",
                shape
            ),
        });
    }

    if kernel_size[0] == 0
        || kernel_size[1] == 0
        || stride[0] == 0
        || stride[1] == 0
        || dilation[0] == 0
        || dilation[1] == 0
    {
        return Err(FerrotorchError::InvalidArgument {
            message: "fold: kernel_size, stride, dilation must all be > 0".into(),
        });
    }

    let batch = shape[0];
    let k = shape[1]; // C * kH * kW
    let l = shape[2]; // out_h * out_w

    let [h_out, w_out] = output_size;
    let k_area = kernel_size[0] * kernel_size[1];

    if k % k_area != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("fold: dim 1 ({k}) must be divisible by kH*kW ({})", k_area),
        });
    }
    let channels = k / k_area;

    let expected_out_h =
        unfold_output_size(h_out, kernel_size[0], dilation[0], padding[0], stride[0]);
    let expected_out_w =
        unfold_output_size(w_out, kernel_size[1], dilation[1], padding[1], stride[1]);
    let expected_l = expected_out_h * expected_out_w;

    if l != expected_l {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "fold: L={l} does not match expected {expected_l} for output_size ({h_out}, {w_out})"
            ),
        });
    }

    let input_device = input.device();
    let data = input.data_vec()?;

    let total = batch * channels * h_out * w_out;
    let mut output = vec![T::from(0.0).unwrap(); total];

    for b in 0..batch {
        for c in 0..channels {
            for kh in 0..kernel_size[0] {
                for kw in 0..kernel_size[1] {
                    let k_idx = (c * kernel_size[0] + kh) * kernel_size[1] + kw;
                    for oh in 0..expected_out_h {
                        for ow in 0..expected_out_w {
                            let ih = oh * stride[0] + kh * dilation[0];
                            let iw = ow * stride[1] + kw * dilation[1];
                            let ih = ih as isize - padding[0] as isize;
                            let iw = iw as isize - padding[1] as isize;

                            if ih >= 0 && ih < h_out as isize && iw >= 0 && iw < w_out as isize {
                                let l_idx = oh * expected_out_w + ow;
                                let in_idx = (b * k + k_idx) * l + l_idx;
                                let out_idx = ((b * channels + c) * h_out + ih as usize) * w_out
                                    + iw as usize;
                                output[out_idx] += data[in_idx];
                            }
                        }
                    }
                }
            }
        }
    }

    let out_shape = vec![batch, channels, h_out, w_out];
    let storage = TensorStorage::cpu(output);

    if is_grad_enabled() && input.requires_grad() {
        Tensor::from_operation(
            storage,
            out_shape,
            Arc::new(FoldBackward {
                input: input.clone(),
                kernel_size,
                dilation,
                padding,
                stride,
            }),
        )?
        .to(input_device)
    } else {
        Tensor::from_storage(storage, out_shape, false)?.to(input_device)
    }
}

#[derive(Debug)]
struct UnfoldBackward<T: Float> {
    input: Tensor<T>,
    kernel_size: [usize; 2],
    dilation: [usize; 2],
    padding: [usize; 2],
    stride: [usize; 2],
}

impl<T: Float> GradFn<T> for UnfoldBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Backward of unfold is fold.
        let in_shape = self.input.shape();
        let h = in_shape[2];
        let w = in_shape[3];
        let grad_input = fold(
            grad_output,
            [h, w],
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "UnfoldBackward"
    }
}

#[derive(Debug)]
struct FoldBackward<T: Float> {
    input: Tensor<T>,
    kernel_size: [usize; 2],
    dilation: [usize; 2],
    padding: [usize; 2],
    stride: [usize; 2],
}

impl<T: Float> GradFn<T> for FoldBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }
        // Backward of fold is unfold.
        let grad_input = unfold(
            grad_output,
            self.kernel_size,
            self.dilation,
            self.padding,
            self.stride,
        )?;
        Ok(vec![Some(grad_input)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "FoldBackward"
    }
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

    /// Create a leaf tensor with any shape.
    fn leaf(data: &[f32], shape: &[usize], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

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
                "index {i}: actual={a} expected={e} diff={}",
                (a - e).abs(),
            );
        }
    }

    // -----------------------------------------------------------------------
    // Interpolation: Nearest
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_nearest_upsample_2x() {
        // [1, 1, 2, 2] -> [1, 1, 4, 4]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], false);
        let out = interpolate(&input, Some([4, 4]), None, InterpolateMode::Nearest, false).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);

        let d = out.data().unwrap();
        // Each 2x2 block should repeat the source pixel.
        #[rustfmt::skip]
        let expected: Vec<f32> = vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_close(d, &expected, 1e-6);
    }

    #[test]
    fn test_interpolate_nearest_downsample() {
        // [1, 1, 4, 4] -> [1, 1, 2, 2]
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = leaf_4d(&data, [1, 1, 4, 4], false);
        let out = interpolate(&input, Some([2, 2]), None, InterpolateMode::Nearest, false).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        let d = out.data().unwrap();
        // floor(0 * 2) = 0, floor(1 * 2) = 2; so we pick (0,0)=1, (0,2)=3, (2,0)=9, (2,2)=11
        assert_close(d, &[1.0, 3.0, 9.0, 11.0], 1e-6);
    }

    #[test]
    fn test_interpolate_nearest_scale_factor() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], false);
        let out = interpolate(
            &input,
            None,
            Some([2.0, 2.0]),
            InterpolateMode::Nearest,
            false,
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
    }

    // -----------------------------------------------------------------------
    // Interpolation: Bilinear
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_bilinear_upsample() {
        // [1, 1, 2, 2] with align_corners=true -> [1, 1, 3, 3]
        let data: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], false);
        let out = interpolate(&input, Some([3, 3]), None, InterpolateMode::Bilinear, true).unwrap();
        assert_eq!(out.shape(), &[1, 1, 3, 3]);

        let d = out.data().unwrap();
        // Corners should match exactly.
        assert!((d[0] - 0.0).abs() < 1e-5); // top-left
        assert!((d[2] - 1.0).abs() < 1e-5); // top-right
        assert!((d[6] - 2.0).abs() < 1e-5); // bottom-left
        assert!((d[8] - 3.0).abs() < 1e-5); // bottom-right
        // Center should be average of all corners.
        assert!((d[4] - 1.5).abs() < 1e-5);
    }

    #[test]
    fn test_interpolate_bilinear_identity() {
        // Same size should be approximately identity.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let input = leaf_4d(&data, [1, 1, 3, 3], false);
        let out = interpolate(&input, Some([3, 3]), None, InterpolateMode::Bilinear, true).unwrap();
        assert_eq!(out.shape(), &[1, 1, 3, 3]);
        assert_close(out.data().unwrap(), &data, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Interpolation: Bicubic
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_bicubic_output_shape() {
        let data: Vec<f32> = vec![0.0; 64];
        let input = leaf_4d(&data, [1, 1, 8, 8], false);
        let out = interpolate(
            &input,
            Some([16, 16]),
            None,
            InterpolateMode::Bicubic,
            false,
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 1, 16, 16]);
    }

    #[test]
    fn test_interpolate_bicubic_corners_align() {
        // With align_corners=true, the 4 corners of a 2x2 input should
        // map exactly to the 4 corners of the output.
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], false);
        let out = interpolate(&input, Some([5, 5]), None, InterpolateMode::Bicubic, true).unwrap();
        assert_eq!(out.shape(), &[1, 1, 5, 5]);
        let d = out.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-4); // top-left
        assert!((d[4] - 2.0).abs() < 1e-4); // top-right
        assert!((d[20] - 3.0).abs() < 1e-4); // bottom-left
        assert!((d[24] - 4.0).abs() < 1e-4); // bottom-right
    }

    // -----------------------------------------------------------------------
    // Upsample module
    // -----------------------------------------------------------------------

    #[test]
    fn test_upsample_module_nearest() {
        let up = Upsample::new([6, 6], InterpolateMode::Nearest);
        let input = leaf_4d(&[0.0; 9], [1, 1, 3, 3], false);
        let out: Tensor<f32> = Module::<f32>::forward(&up, &input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 6, 6]);
    }

    #[test]
    fn test_upsample_module_bilinear_scale() {
        let up = Upsample::with_scale_factor([2.0, 2.0], InterpolateMode::Bilinear);
        let input = leaf_4d(&[0.0; 4], [1, 1, 2, 2], false);
        let out: Tensor<f32> = Module::<f32>::forward(&up, &input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_upsample_no_parameters() {
        let up = Upsample::new([4, 4], InterpolateMode::Nearest);
        assert!(Module::<f32>::parameters(&up).is_empty());
    }

    // -----------------------------------------------------------------------
    // Interpolate errors
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_no_size_no_scale() {
        let input = leaf_4d(&[0.0; 4], [1, 1, 2, 2], false);
        assert!(interpolate(&input, None, None, InterpolateMode::Nearest, false).is_err());
    }

    #[test]
    fn test_interpolate_both_size_and_scale() {
        let input = leaf_4d(&[0.0; 4], [1, 1, 2, 2], false);
        assert!(
            interpolate(
                &input,
                Some([4, 4]),
                Some([2.0, 2.0]),
                InterpolateMode::Nearest,
                false
            )
            .is_err()
        );
    }

    #[test]
    fn test_interpolate_nearest_align_corners_rejected() {
        let input = leaf_4d(&[0.0; 4], [1, 1, 2, 2], false);
        assert!(interpolate(&input, Some([4, 4]), None, InterpolateMode::Nearest, true).is_err());
    }

    #[test]
    fn test_interpolate_3d_rejected() {
        let input = leaf(&[0.0; 6], &[2, 3], false);
        assert!(interpolate(&input, Some([4, 4]), None, InterpolateMode::Nearest, false).is_err());
    }

    // -----------------------------------------------------------------------
    // Interpolate backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_nearest_backward() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], true);
        let out = interpolate(&input, Some([4, 4]), None, InterpolateMode::Nearest, false).unwrap();

        let out_data = out.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(TestSumBackward { input: out }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // Each input pixel maps to 4 output pixels (2x2 block), so grad = 4.0 for each.
        for (i, &val) in g.iter().enumerate() {
            assert!(
                (val - 4.0).abs() < 1e-5,
                "grad[{i}]: expected 4.0, got {val}"
            );
        }
    }

    #[test]
    fn test_interpolate_bilinear_backward() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], true);
        let out =
            interpolate(&input, Some([4, 4]), None, InterpolateMode::Bilinear, false).unwrap();

        let out_data = out.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(TestSumBackward { input: out }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // Check gradient is non-zero and sums correctly.
        let grad_sum: f32 = g.iter().sum();
        // Sum of gradient = number of output elements (16).
        assert!(
            (grad_sum - 16.0).abs() < 1e-3,
            "grad sum = {grad_sum}, expected 16.0"
        );
    }

    // -----------------------------------------------------------------------
    // PixelShuffle
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_shuffle_shape() {
        // [1, 4, 2, 2] with r=2 -> [1, 1, 4, 4]
        let data = vec![0.0f32; 16];
        let input = leaf_4d(&data, [1, 4, 2, 2], false);
        let out = pixel_shuffle(&input, 2).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pixel_shuffle_values() {
        // [1, 4, 1, 1] with r=2 -> [1, 1, 2, 2]
        // Input channels: [c0_r0c0, c0_r0c1, c0_r1c0, c0_r1c1] = [1, 2, 3, 4]
        // Output: [[1, 2], [3, 4]]
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 4, 1, 1], false);
        let out = pixel_shuffle(&input, 2).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_close(out.data().unwrap(), &[1.0, 2.0, 3.0, 4.0], 1e-6);
    }

    #[test]
    fn test_pixel_shuffle_not_divisible() {
        // channels=3 not divisible by r^2=4
        let input = leaf_4d(&[0.0; 12], [1, 3, 2, 2], false);
        assert!(pixel_shuffle(&input, 2).is_err());
    }

    // -----------------------------------------------------------------------
    // PixelUnshuffle
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_unshuffle_shape() {
        // [1, 1, 4, 4] with r=2 -> [1, 4, 2, 2]
        let data = vec![0.0f32; 16];
        let input = leaf_4d(&data, [1, 1, 4, 4], false);
        let out = pixel_unshuffle(&input, 2).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2, 2]);
    }

    #[test]
    fn test_pixel_shuffle_unshuffle_roundtrip() {
        // Shuffle then unshuffle should give back the original.
        let data: Vec<f32> = (0..36).map(|i| i as f32).collect();
        let input = leaf_4d(&data, [1, 4, 3, 3], false);
        let shuffled = pixel_shuffle(&input, 2).unwrap();
        assert_eq!(shuffled.shape(), &[1, 1, 6, 6]);
        let roundtrip = pixel_unshuffle(&shuffled, 2).unwrap();
        assert_eq!(roundtrip.shape(), &[1, 4, 3, 3]);
        assert_close(roundtrip.data().unwrap(), &data, 1e-6);
    }

    #[test]
    fn test_pixel_unshuffle_spatial_not_divisible() {
        // H=3 not divisible by r=2
        let input = leaf_4d(&[0.0; 9], [1, 1, 3, 3], false);
        assert!(pixel_unshuffle(&input, 2).is_err());
    }

    // -----------------------------------------------------------------------
    // PixelShuffle backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_shuffle_backward() {
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let input = leaf_4d(&data, [1, 4, 2, 2], true);
        let out = pixel_shuffle(&input, 2).unwrap();

        let out_data = out.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(TestSumBackward { input: out }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // Gradient of sum = 1 everywhere.
        for (i, &val) in g.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "grad[{i}]: expected 1.0, got {val}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Unfold
    // -----------------------------------------------------------------------

    #[test]
    fn test_unfold_shape() {
        // [1, 1, 4, 4], kernel 2x2, stride 1, no padding, no dilation
        // out_h = (4 - 2) / 1 + 1 = 3, out_w = 3, L = 9, k = 1*2*2 = 4
        let input = leaf_4d(&[0.0; 16], [1, 1, 4, 4], false);
        let out = unfold(&input, [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
        assert_eq!(out.shape(), &[1, 4, 9]);
    }

    #[test]
    fn test_unfold_values() {
        // [1, 1, 3, 3], kernel 2x2, stride 1
        // Input:
        // 1 2 3
        // 4 5 6
        // 7 8 9
        //
        // Patches (each window is 4 elements, 4 windows):
        // window (0,0): [1, 2, 4, 5]
        // window (0,1): [2, 3, 5, 6]
        // window (1,0): [4, 5, 7, 8]
        // window (1,1): [5, 6, 8, 9]
        //
        // Output [1, 4, 4]:
        // k=0 (c=0, kh=0, kw=0): [1, 2, 4, 5]
        // k=1 (c=0, kh=0, kw=1): [2, 3, 5, 6]
        // k=2 (c=0, kh=1, kw=0): [4, 5, 7, 8]
        // k=3 (c=0, kh=1, kw=1): [5, 6, 8, 9]
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ];
        let input = leaf_4d(&data, [1, 1, 3, 3], false);
        let out = unfold(&input, [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
        assert_eq!(out.shape(), &[1, 4, 4]);

        let d = out.data().unwrap();
        assert_close(&d[0..4], &[1.0, 2.0, 4.0, 5.0], 1e-6);
        assert_close(&d[4..8], &[2.0, 3.0, 5.0, 6.0], 1e-6);
        assert_close(&d[8..12], &[4.0, 5.0, 7.0, 8.0], 1e-6);
        assert_close(&d[12..16], &[5.0, 6.0, 8.0, 9.0], 1e-6);
    }

    #[test]
    fn test_unfold_with_padding() {
        // [1, 1, 2, 2], kernel 2x2, stride 1, padding 1
        // Padded: [4, 4], out_h = (2 + 2*1 - 2)/1 + 1 = 3, L = 9, k = 4
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], false);
        let out = unfold(&input, [2, 2], [1, 1], [1, 1], [1, 1]).unwrap();
        assert_eq!(out.shape(), &[1, 4, 9]);
    }

    #[test]
    fn test_unfold_with_stride() {
        // [1, 1, 4, 4], kernel 2x2, stride 2
        // out_h = (4 - 2)/2 + 1 = 2, L = 4, k = 4
        let input = leaf_4d(&[0.0; 16], [1, 1, 4, 4], false);
        let out = unfold(&input, [2, 2], [1, 1], [0, 0], [2, 2]).unwrap();
        assert_eq!(out.shape(), &[1, 4, 4]);
    }

    #[test]
    fn test_unfold_zero_kernel_rejected() {
        let input = leaf_4d(&[0.0; 16], [1, 1, 4, 4], false);
        assert!(unfold(&input, [0, 2], [1, 1], [0, 0], [1, 1]).is_err());
    }

    // -----------------------------------------------------------------------
    // Fold
    // -----------------------------------------------------------------------

    #[test]
    fn test_fold_shape() {
        // [1, 4, 9] -> [1, 1, 4, 4] with kernel 2x2, stride 1
        let data = vec![0.0f32; 36];
        let input = leaf(&data, &[1, 4, 9], false);
        let out = fold(&input, [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_unfold_fold_roundtrip() {
        // For non-overlapping patches (stride=kernel), fold(unfold(x)) == x.
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        ];
        let input = leaf_4d(&data, [1, 1, 4, 4], false);
        let unfolded = unfold(&input, [2, 2], [1, 1], [0, 0], [2, 2]).unwrap();
        let refolded = fold(&unfolded, [4, 4], [2, 2], [1, 1], [0, 0], [2, 2]).unwrap();
        assert_eq!(refolded.shape(), &[1, 1, 4, 4]);
        assert_close(refolded.data().unwrap(), &data, 1e-6);
    }

    #[test]
    fn test_fold_l_mismatch() {
        // L doesn't match output_size
        let data = vec![0.0f32; 20];
        let input = leaf(&data, &[1, 4, 5], false);
        assert!(fold(&input, [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]).is_err());
    }

    // -----------------------------------------------------------------------
    // grid_sample
    // -----------------------------------------------------------------------

    #[test]
    fn test_grid_sample_identity() {
        // Create a grid that maps each output pixel to itself (identity transform).
        // align_corners=true, so normalized coords are [-1, 1].
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], false);

        // Grid for 2x2 output:
        // (0,0) -> (-1,-1), (0,1) -> (1,-1), (1,0) -> (-1,1), (1,1) -> (1,1)
        let grid_data: Vec<f32> = vec![-1.0, -1.0, 1.0, -1.0, -1.0, 1.0, 1.0, 1.0];
        let grid = leaf(&grid_data, &[1, 2, 2, 2], false);

        let out = grid_sample(
            &input,
            &grid,
            GridSampleMode::Bilinear,
            GridSamplePaddingMode::Zeros,
            true,
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        assert_close(out.data().unwrap(), &data, 1e-5);
    }

    #[test]
    fn test_grid_sample_nearest() {
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let input = leaf_4d(&data, [1, 1, 2, 2], false);

        // Grid that samples center of pixel (0,0) -> should get 1.0
        let grid_data: Vec<f32> = vec![-1.0, -1.0];
        let grid = leaf(&grid_data, &[1, 1, 1, 2], false);

        let out = grid_sample(
            &input,
            &grid,
            GridSampleMode::Nearest,
            GridSamplePaddingMode::Zeros,
            true,
        )
        .unwrap();
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        assert!((out.data().unwrap()[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_grid_sample_batch_mismatch() {
        let input = leaf_4d(&[0.0; 8], [2, 1, 2, 2], false);
        let grid = leaf(&[0.0; 8], &[1, 2, 2, 2], false);
        assert!(
            grid_sample(
                &input,
                &grid,
                GridSampleMode::Bilinear,
                GridSamplePaddingMode::Zeros,
                true
            )
            .is_err()
        );
    }

    #[test]
    fn test_grid_sample_wrong_grid_shape() {
        let input = leaf_4d(&[0.0; 4], [1, 1, 2, 2], false);
        let grid = leaf(&[0.0; 8], &[1, 2, 4], false);
        assert!(
            grid_sample(
                &input,
                &grid,
                GridSampleMode::Bilinear,
                GridSamplePaddingMode::Zeros,
                true
            )
            .is_err()
        );
    }

    // -----------------------------------------------------------------------
    // affine_grid
    // -----------------------------------------------------------------------

    #[test]
    fn test_affine_grid_identity() {
        // Identity transform: [[1, 0, 0], [0, 1, 0]]
        let theta_data: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0];
        let theta = leaf(&theta_data, &[1, 2, 3], false);
        let grid = affine_grid(&theta, [1, 1, 3, 3], true).unwrap();
        assert_eq!(grid.shape(), &[1, 3, 3, 2]);

        let d = grid.data().unwrap();
        // Corners of the grid should be at (-1,-1), (1,-1), (-1,1), (1,1)
        // Top-left: (iy=0, ix=0) -> x=-1, y=-1
        assert!((d[0] - (-1.0)).abs() < 1e-5); // x
        assert!((d[1] - (-1.0)).abs() < 1e-5); // y
        // Top-right: (iy=0, ix=2) -> x=1, y=-1
        assert!((d[4] - 1.0).abs() < 1e-5); // x
        assert!((d[5] - (-1.0)).abs() < 1e-5); // y
    }

    #[test]
    fn test_affine_grid_theta_shape_error() {
        let theta = leaf(&[0.0; 12], &[2, 3, 2], false);
        assert!(affine_grid(&theta, [2, 1, 3, 3], true).is_err());
    }

    #[test]
    fn test_affine_grid_batch_mismatch() {
        let theta = leaf(&[0.0; 6], &[1, 2, 3], false);
        assert!(affine_grid(&theta, [2, 1, 3, 3], true).is_err());
    }

    // -----------------------------------------------------------------------
    // PixelShuffle / PixelUnshuffle module
    // -----------------------------------------------------------------------

    #[test]
    fn test_pixel_shuffle_module() {
        let ps = PixelShuffle::new(2);
        let input = leaf_4d(&[0.0; 16], [1, 4, 2, 2], false);
        let out: Tensor<f32> = Module::<f32>::forward(&ps, &input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
    }

    #[test]
    fn test_pixel_unshuffle_module() {
        let pus = PixelUnshuffle::new(2);
        let input = leaf_4d(&[0.0; 16], [1, 1, 4, 4], false);
        let out: Tensor<f32> = Module::<f32>::forward(&pus, &input).unwrap();
        assert_eq!(out.shape(), &[1, 4, 2, 2]);
    }

    // -----------------------------------------------------------------------
    // Unfold / Fold modules
    // -----------------------------------------------------------------------

    #[test]
    fn test_unfold_module() {
        let uf = Unfold::new([2, 2], [1, 1], [0, 0], [1, 1]);
        let input = leaf_4d(&[0.0; 16], [1, 1, 4, 4], false);
        let out: Tensor<f32> = Module::<f32>::forward(&uf, &input).unwrap();
        assert_eq!(out.shape(), &[1, 4, 9]);
    }

    #[test]
    fn test_fold_module() {
        let f = Fold::new([4, 4], [2, 2], [1, 1], [0, 0], [1, 1]);
        let data = vec![0.0f32; 36];
        let input = leaf(&data, &[1, 4, 9], false);
        let out: Tensor<f32> = Module::<f32>::forward(&f, &input).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4, 4]);
    }

    // -----------------------------------------------------------------------
    // Unfold backward
    // -----------------------------------------------------------------------

    #[test]
    fn test_unfold_backward() {
        // Non-overlapping unfold (stride = kernel): fold(unfold(x)) = x, so
        // gradient through sum(unfold(x)) should be all 1s.
        let data: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let input = leaf_4d(&data, [1, 1, 4, 4], true);
        let out = unfold(&input, [2, 2], [1, 1], [0, 0], [2, 2]).unwrap();

        let out_data = out.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(TestSumBackward { input: out }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        for (i, &val) in g.iter().enumerate() {
            assert!(
                (val - 1.0).abs() < 1e-5,
                "grad[{i}]: expected 1.0, got {val}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Unfold backward with overlapping patches
    // -----------------------------------------------------------------------

    #[test]
    fn test_unfold_backward_overlapping() {
        // Overlapping unfold (stride < kernel): each input pixel appears in
        // multiple patches, so gradient should be > 1 for interior pixels.
        let data: Vec<f32> = (0..9).map(|i| i as f32).collect();
        let input = leaf_4d(&data, [1, 1, 3, 3], true);
        let out = unfold(&input, [2, 2], [1, 1], [0, 0], [1, 1]).unwrap();
        // out shape: [1, 4, 4]

        let out_data = out.data().unwrap().to_vec();
        let total: f32 = out_data.iter().sum();
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(TestSumBackward { input: out }),
        )
        .unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap().unwrap();
        let g = grad.data().unwrap();
        // Corner pixels appear in 1 patch: grad=1
        // Edge pixels appear in 2 patches: grad=2
        // Center pixel appears in 4 patches: grad=4
        #[rustfmt::skip]
        let expected: Vec<f32> = vec![
            1.0, 2.0, 1.0,
            2.0, 4.0, 2.0,
            1.0, 2.0, 1.0,
        ];
        assert_close(g, &expected, 1e-5);
    }

    // -----------------------------------------------------------------------
    // Multichannel batch tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_interpolate_multichannel_batch() {
        // [2, 3, 2, 2] -> [2, 3, 4, 4]
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let input = leaf_4d(&data, [2, 3, 2, 2], false);
        let out = interpolate(&input, Some([4, 4]), None, InterpolateMode::Nearest, false).unwrap();
        assert_eq!(out.shape(), &[2, 3, 4, 4]);
    }

    // -----------------------------------------------------------------------
    // Helper backward node for tests
    // -----------------------------------------------------------------------

    #[derive(Debug)]
    struct TestSumBackward {
        input: Tensor<f32>,
    }

    impl GradFn<f32> for TestSumBackward {
        fn backward(
            &self,
            _grad_output: &Tensor<f32>,
        ) -> FerrotorchResult<Vec<Option<Tensor<f32>>>> {
            let ones_data = vec![1.0f32; self.input.numel()];
            let ones = Tensor::from_storage(
                TensorStorage::cpu(ones_data),
                self.input.shape().to_vec(),
                false,
            )?;
            Ok(vec![Some(ones)])
        }

        fn inputs(&self) -> Vec<&Tensor<f32>> {
            vec![&self.input]
        }

        fn name(&self) -> &'static str {
            "TestSumBackward"
        }
    }
}
