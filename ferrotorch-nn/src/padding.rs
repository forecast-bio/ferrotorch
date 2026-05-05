//! Padding layers: constant, reflection, replication, and zero padding in 1-D, 2-D, 3-D.
//!
//! [CL-314] Add Conv3d, ConvTranspose1d/3d, and padding modules
//!
//! Each module pads the **last N** dimensions of the input tensor, matching
//! PyTorch semantics exactly.  Padding tuples specify *(left, right)* for 1-D,
//! *(left, right, top, bottom)* for 2-D, and
//! *(left, right, top, bottom, front, back)* for 3-D.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::Tensor;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};

use crate::module::Module;
use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// Padding mode enum (used by conv layers with padding_mode)
// ---------------------------------------------------------------------------

/// Padding mode for convolution layers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddingMode {
    /// Zero padding (default).
    Zeros,
    /// Reflect padding.
    Reflect,
    /// Replicate padding (edge padding).
    Replicate,
    /// Circular padding (wrap-around).
    Circular,
}

// ---------------------------------------------------------------------------
// Low-level pad helpers (operate on raw data)
// ---------------------------------------------------------------------------

/// Pad the last dimension of a contiguous tensor.
///
/// `shape` has at least 1 dimension. The padding values `(left, right)` are
/// added to dimension `ndim-1`.
fn pad_1d_constant<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    value: T,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let inner = shape[ndim - 1];
    let new_inner = inner + pad_left + pad_right;

    // Number of "rows" = product of all dimensions except the last.
    let rows: usize = shape[..ndim - 1].iter().product();
    let rows = if rows == 0 { 1 } else { rows };

    let mut out = vec![value; rows * new_inner];
    for r in 0..rows {
        let src_start = r * inner;
        let dst_start = r * new_inner + pad_left;
        out[dst_start..dst_start + inner].copy_from_slice(&data[src_start..src_start + inner]);
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 1] = new_inner;
    (out, new_shape)
}

/// Pad the last 2 dimensions of a contiguous tensor with a constant value.
fn pad_2d_constant<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    value: T,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;

    let outer: usize = shape[..ndim - 2].iter().product();
    let outer = if outer == 0 { 1 } else { outer };

    let mut out = vec![value; outer * new_h * new_w];
    for o in 0..outer {
        for row in 0..h {
            let src_off = o * h * w + row * w;
            let dst_off = o * new_h * new_w + (row + pad_top) * new_w + pad_left;
            out[dst_off..dst_off + w].copy_from_slice(&data[src_off..src_off + w]);
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

/// Pad the last 3 dimensions of a contiguous tensor with a constant value.
// Internal kernel: signature mirrors PyTorch's `F.pad` 3-axis layout
// (left, right, top, bottom, front, back); a config struct adds nothing.
#[allow(clippy::too_many_arguments)]
fn pad_3d_constant<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
    value: T,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let d = shape[ndim - 3];
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_d = d + pad_front + pad_back;
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;

    let outer: usize = shape[..ndim - 3].iter().product();
    let outer = if outer == 0 { 1 } else { outer };

    let mut out = vec![value; outer * new_d * new_h * new_w];
    for o in 0..outer {
        for dep in 0..d {
            for row in 0..h {
                let src_off = o * d * h * w + dep * h * w + row * w;
                let dst_off = o * new_d * new_h * new_w
                    + (dep + pad_front) * new_h * new_w
                    + (row + pad_top) * new_w
                    + pad_left;
                out[dst_off..dst_off + w].copy_from_slice(&data[src_off..src_off + w]);
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 3] = new_d;
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

// ---------------------------------------------------------------------------
// Reflection padding helpers
// ---------------------------------------------------------------------------

/// Reflect-pad the last dimension.
fn pad_1d_reflect<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
) -> FerrotorchResult<(Vec<T>, Vec<usize>)> {
    let ndim = shape.len();
    let inner = shape[ndim - 1];
    if pad_left >= inner || pad_right >= inner {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Reflection padding ({pad_left}, {pad_right}) must be less than input size ({inner})"
            ),
        });
    }
    let new_inner = inner + pad_left + pad_right;
    let rows: usize = shape[..ndim - 1].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; rows * new_inner];
    for r in 0..rows {
        let src = &data[r * inner..(r + 1) * inner];
        let dst = &mut out[r * new_inner..(r + 1) * new_inner];
        // Left reflection
        for i in 0..pad_left {
            dst[pad_left - 1 - i] = src[i + 1];
        }
        // Copy original
        dst[pad_left..pad_left + inner].copy_from_slice(src);
        // Right reflection
        for i in 0..pad_right {
            dst[pad_left + inner + i] = src[inner - 2 - i];
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 1] = new_inner;
    Ok((out, new_shape))
}

/// Reflect-pad the last 2 dimensions.
fn pad_2d_reflect<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
) -> FerrotorchResult<(Vec<T>, Vec<usize>)> {
    let ndim = shape.len();
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    if pad_left >= w || pad_right >= w || pad_top >= h || pad_bottom >= h {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Reflection padding ({pad_left}, {pad_right}, {pad_top}, {pad_bottom}) must be less than input size ({h}, {w})"
            ),
        });
    }
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 2].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_h * new_w];

    for o in 0..outer {
        let src_base = o * h * w;
        let dst_base = o * new_h * new_w;

        for new_row in 0..new_h {
            // Map new_row to source row via reflection
            let src_row = if new_row < pad_top {
                pad_top - new_row
            } else if new_row >= pad_top + h {
                h - 2 - (new_row - pad_top - h)
            } else {
                new_row - pad_top
            };

            for new_col in 0..new_w {
                let src_col = if new_col < pad_left {
                    pad_left - new_col
                } else if new_col >= pad_left + w {
                    w - 2 - (new_col - pad_left - w)
                } else {
                    new_col - pad_left
                };

                out[dst_base + new_row * new_w + new_col] = data[src_base + src_row * w + src_col];
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    Ok((out, new_shape))
}

/// Reflect-pad the last 3 dimensions.
// Internal kernel: same 3-axis pad descriptor as `pad_3d_constant`.
#[allow(clippy::too_many_arguments)]
fn pad_3d_reflect<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
) -> FerrotorchResult<(Vec<T>, Vec<usize>)> {
    let ndim = shape.len();
    let d = shape[ndim - 3];
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    if pad_left >= w
        || pad_right >= w
        || pad_top >= h
        || pad_bottom >= h
        || pad_front >= d
        || pad_back >= d
    {
        return Err(FerrotorchError::InvalidArgument {
            message: "Reflection padding must be less than corresponding input dimension".into(),
        });
    }
    let new_d = d + pad_front + pad_back;
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 3].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_d * new_h * new_w];

    for o in 0..outer {
        let src_base = o * d * h * w;
        let dst_base = o * new_d * new_h * new_w;

        for nd in 0..new_d {
            let sd = if nd < pad_front {
                pad_front - nd
            } else if nd >= pad_front + d {
                d - 2 - (nd - pad_front - d)
            } else {
                nd - pad_front
            };
            for nh in 0..new_h {
                let sh = if nh < pad_top {
                    pad_top - nh
                } else if nh >= pad_top + h {
                    h - 2 - (nh - pad_top - h)
                } else {
                    nh - pad_top
                };
                for nw in 0..new_w {
                    let sw = if nw < pad_left {
                        pad_left - nw
                    } else if nw >= pad_left + w {
                        w - 2 - (nw - pad_left - w)
                    } else {
                        nw - pad_left
                    };
                    out[dst_base + nd * new_h * new_w + nh * new_w + nw] =
                        data[src_base + sd * h * w + sh * w + sw];
                }
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 3] = new_d;
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    Ok((out, new_shape))
}

// ---------------------------------------------------------------------------
// Replication padding helpers
// ---------------------------------------------------------------------------

/// Replicate-pad the last dimension (clamp to edges).
fn pad_1d_replicate<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let inner = shape[ndim - 1];
    let new_inner = inner + pad_left + pad_right;
    let rows: usize = shape[..ndim - 1].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; rows * new_inner];
    for r in 0..rows {
        let src = &data[r * inner..(r + 1) * inner];
        let dst = &mut out[r * new_inner..(r + 1) * new_inner];
        for (i, d) in dst.iter_mut().enumerate() {
            let src_idx = if i < pad_left {
                0
            } else if i >= pad_left + inner {
                inner - 1
            } else {
                i - pad_left
            };
            *d = src[src_idx];
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 1] = new_inner;
    (out, new_shape)
}

/// Replicate-pad the last 2 dimensions.
fn pad_2d_replicate<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 2].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_h * new_w];

    for o in 0..outer {
        let src_base = o * h * w;
        let dst_base = o * new_h * new_w;
        for nr in 0..new_h {
            let sr = nr.saturating_sub(pad_top).min(h - 1);
            for nc in 0..new_w {
                let sc = nc.saturating_sub(pad_left).min(w - 1);
                out[dst_base + nr * new_w + nc] = data[src_base + sr * w + sc];
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

/// Replicate-pad the last 3 dimensions.
// Internal kernel: same 3-axis pad descriptor as `pad_3d_constant`.
#[allow(clippy::too_many_arguments)]
fn pad_3d_replicate<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let d = shape[ndim - 3];
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_d = d + pad_front + pad_back;
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 3].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_d * new_h * new_w];

    for o in 0..outer {
        let src_base = o * d * h * w;
        let dst_base = o * new_d * new_h * new_w;
        for nd in 0..new_d {
            let sd = nd.saturating_sub(pad_front).min(d - 1);
            for nh in 0..new_h {
                let sh = nh.saturating_sub(pad_top).min(h - 1);
                for nw in 0..new_w {
                    let sw = nw.saturating_sub(pad_left).min(w - 1);
                    out[dst_base + nd * new_h * new_w + nh * new_w + nw] =
                        data[src_base + sd * h * w + sh * w + sw];
                }
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 3] = new_d;
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

// ---------------------------------------------------------------------------
// Circular padding helpers
// ---------------------------------------------------------------------------

/// Circular-pad the last dimension (wrap-around).
fn pad_1d_circular<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let inner = shape[ndim - 1];
    let new_inner = inner + pad_left + pad_right;
    let rows: usize = shape[..ndim - 1].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; rows * new_inner];
    for r in 0..rows {
        let src = &data[r * inner..(r + 1) * inner];
        let dst = &mut out[r * new_inner..(r + 1) * new_inner];
        for (i, d) in dst.iter_mut().enumerate() {
            // Map to source via modulo
            let src_idx = ((i as isize - pad_left as isize).rem_euclid(inner as isize)) as usize;
            *d = src[src_idx];
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 1] = new_inner;
    (out, new_shape)
}

/// Circular-pad the last 2 dimensions.
fn pad_2d_circular<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 2].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_h * new_w];

    for o in 0..outer {
        let src_base = o * h * w;
        let dst_base = o * new_h * new_w;
        for nr in 0..new_h {
            let sr = ((nr as isize - pad_top as isize).rem_euclid(h as isize)) as usize;
            for nc in 0..new_w {
                let sc = ((nc as isize - pad_left as isize).rem_euclid(w as isize)) as usize;
                out[dst_base + nr * new_w + nc] = data[src_base + sr * w + sc];
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

/// Circular-pad the last 3 dimensions.
// Internal kernel: same 3-axis pad descriptor as `pad_3d_constant`.
#[allow(clippy::too_many_arguments)]
fn pad_3d_circular<T: Float>(
    data: &[T],
    shape: &[usize],
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
) -> (Vec<T>, Vec<usize>) {
    let ndim = shape.len();
    let d = shape[ndim - 3];
    let h = shape[ndim - 2];
    let w = shape[ndim - 1];
    let new_d = d + pad_front + pad_back;
    let new_h = h + pad_top + pad_bottom;
    let new_w = w + pad_left + pad_right;
    let outer: usize = shape[..ndim - 3].iter().copied().product::<usize>().max(1);

    let zero = <T as num_traits::Zero>::zero();
    let mut out = vec![zero; outer * new_d * new_h * new_w];

    for o in 0..outer {
        let src_base = o * d * h * w;
        let dst_base = o * new_d * new_h * new_w;
        for nd in 0..new_d {
            let sd = ((nd as isize - pad_front as isize).rem_euclid(d as isize)) as usize;
            for nh in 0..new_h {
                let sh = ((nh as isize - pad_top as isize).rem_euclid(h as isize)) as usize;
                for nw in 0..new_w {
                    let sw = ((nw as isize - pad_left as isize).rem_euclid(w as isize)) as usize;
                    out[dst_base + nd * new_h * new_w + nh * new_w + nw] =
                        data[src_base + sd * h * w + sh * w + sw];
                }
            }
        }
    }

    let mut new_shape = shape.to_vec();
    new_shape[ndim - 3] = new_d;
    new_shape[ndim - 2] = new_h;
    new_shape[ndim - 1] = new_w;
    (out, new_shape)
}

// ===========================================================================
// Public functional API — apply arbitrary padding to a Tensor
// ===========================================================================

/// Apply padding to the last dimension of a tensor using the given mode.
///
/// This is the functional version used internally by conv layers with
/// `padding_mode`.
pub fn functional_pad_1d<T: Float>(
    input: &Tensor<T>,
    pad_left: usize,
    pad_right: usize,
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    let data = input.data_vec()?;
    let shape = input.shape();
    let (out_data, new_shape) = match mode {
        PaddingMode::Zeros => pad_1d_constant(
            &data,
            shape,
            pad_left,
            pad_right,
            <T as num_traits::Zero>::zero(),
        ),
        PaddingMode::Reflect => pad_1d_reflect(&data, shape, pad_left, pad_right)?,
        PaddingMode::Replicate => pad_1d_replicate(&data, shape, pad_left, pad_right),
        PaddingMode::Circular => pad_1d_circular(&data, shape, pad_left, pad_right),
    };
    let _ = value; // only used for ConstantPad, not this code path
    Tensor::from_storage(TensorStorage::cpu(out_data), new_shape, false)
}

/// Apply padding to the last 2 dimensions of a tensor using the given mode.
pub fn functional_pad_2d<T: Float>(
    input: &Tensor<T>,
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    let data = input.data_vec()?;
    let shape = input.shape();
    let (out_data, new_shape) = match mode {
        PaddingMode::Zeros => pad_2d_constant(
            &data,
            shape,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            <T as num_traits::Zero>::zero(),
        ),
        PaddingMode::Reflect => {
            pad_2d_reflect(&data, shape, pad_left, pad_right, pad_top, pad_bottom)?
        }
        PaddingMode::Replicate => {
            pad_2d_replicate(&data, shape, pad_left, pad_right, pad_top, pad_bottom)
        }
        PaddingMode::Circular => {
            pad_2d_circular(&data, shape, pad_left, pad_right, pad_top, pad_bottom)
        }
    };
    let _ = value;
    Tensor::from_storage(TensorStorage::cpu(out_data), new_shape, false)
}

/// Apply padding to the last 3 dimensions of a tensor using the given mode.
// Public API: matches PyTorch's `torch.nn.functional.pad` signature for the
// 3-axis case (input + 6 pad amounts + mode + value); divergence would
// break parity with the upstream reference.
#[allow(clippy::too_many_arguments)]
pub fn functional_pad_3d<T: Float>(
    input: &Tensor<T>,
    pad_left: usize,
    pad_right: usize,
    pad_top: usize,
    pad_bottom: usize,
    pad_front: usize,
    pad_back: usize,
    mode: PaddingMode,
    value: T,
) -> FerrotorchResult<Tensor<T>> {
    let data = input.data_vec()?;
    let shape = input.shape();
    let (out_data, new_shape) = match mode {
        PaddingMode::Zeros => pad_3d_constant(
            &data,
            shape,
            pad_left,
            pad_right,
            pad_top,
            pad_bottom,
            pad_front,
            pad_back,
            <T as num_traits::Zero>::zero(),
        ),
        PaddingMode::Reflect => pad_3d_reflect(
            &data, shape, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back,
        )?,
        PaddingMode::Replicate => pad_3d_replicate(
            &data, shape, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back,
        ),
        PaddingMode::Circular => pad_3d_circular(
            &data, shape, pad_left, pad_right, pad_top, pad_bottom, pad_front, pad_back,
        ),
    };
    let _ = value;
    Tensor::from_storage(TensorStorage::cpu(out_data), new_shape, false)
}

// ===========================================================================
// Macro to reduce boilerplate for Module implementations on padding layers
// ===========================================================================

macro_rules! impl_padding_module {
    ($name:ident) => {
        impl<T: Float> Module<T> for $name<T> {
            fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
                self.pad(input)
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
    };
}

// ===========================================================================
// ConstantPad1d / ConstantPad2d / ConstantPad3d
// ===========================================================================

/// Pads the last dimension of the input tensor with a constant value.
///
/// # Shape
/// - Input: `[*, L]`
/// - Output: `[*, L + pad_left + pad_right]`
#[derive(Debug)]
pub struct ConstantPad1d<T: Float> {
    /// Padding `(left, right)`.
    pub padding: (usize, usize),
    /// Constant fill value.
    pub value: T,
    training: bool,
}

impl<T: Float> ConstantPad1d<T> {
    pub fn new(padding: (usize, usize), value: T) -> Self {
        Self {
            padding,
            value,
            training: true,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let data = input.data_vec()?;
        let (out, new_shape) = pad_1d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.value,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ConstantPad1d);

/// Pads the last 2 dimensions with a constant value.
///
/// # Shape
/// - Input: `[*, H, W]`
/// - Output: `[*, H + top + bottom, W + left + right]`
#[derive(Debug)]
pub struct ConstantPad2d<T: Float> {
    /// Padding `(left, right, top, bottom)`.
    pub padding: (usize, usize, usize, usize),
    /// Constant fill value.
    pub value: T,
    training: bool,
}

impl<T: Float> ConstantPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize), value: T) -> Self {
        Self {
            padding,
            value,
            training: true,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConstantPad2d expects at least 2-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_2d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.value,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ConstantPad2d);

/// Pads the last 3 dimensions with a constant value.
///
/// # Shape
/// - Input: `[*, D, H, W]`
/// - Output: `[*, D + front + back, H + top + bottom, W + left + right]`
#[derive(Debug)]
pub struct ConstantPad3d<T: Float> {
    /// Padding `(left, right, top, bottom, front, back)`.
    pub padding: (usize, usize, usize, usize, usize, usize),
    /// Constant fill value.
    pub value: T,
    training: bool,
}

impl<T: Float> ConstantPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize), value: T) -> Self {
        Self {
            padding,
            value,
            training: true,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConstantPad3d expects at least 3-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_3d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.padding.4,
            self.padding.5,
            self.value,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ConstantPad3d);

// ===========================================================================
// ZeroPad1d / ZeroPad2d / ZeroPad3d
// ===========================================================================

/// Pads the last dimension with zeros.
#[derive(Debug)]
pub struct ZeroPad1d<T: Float> {
    pub padding: (usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ZeroPad1d<T> {
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let data = input.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let (out, new_shape) =
            pad_1d_constant(&data, input.shape(), self.padding.0, self.padding.1, zero);
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ZeroPad1d);

/// Pads the last 2 dimensions with zeros.
#[derive(Debug)]
pub struct ZeroPad2d<T: Float> {
    pub padding: (usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ZeroPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ZeroPad2d expects at least 2-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let (out, new_shape) = pad_2d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            zero,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ZeroPad2d);

/// Pads the last 3 dimensions with zeros.
#[derive(Debug)]
pub struct ZeroPad3d<T: Float> {
    pub padding: (usize, usize, usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ZeroPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ZeroPad3d expects at least 3-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let zero = <T as num_traits::Zero>::zero();
        let (out, new_shape) = pad_3d_constant(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.padding.4,
            self.padding.5,
            zero,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ZeroPad3d);

// ===========================================================================
// ReflectionPad1d / ReflectionPad2d / ReflectionPad3d
// ===========================================================================

/// Pads the last dimension using reflection of the input boundary.
#[derive(Debug)]
pub struct ReflectionPad1d<T: Float> {
    pub padding: (usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReflectionPad1d<T> {
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let data = input.data_vec()?;
        let (out, new_shape) =
            pad_1d_reflect(&data, input.shape(), self.padding.0, self.padding.1)?;
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReflectionPad1d);

/// Pads the last 2 dimensions using reflection.
#[derive(Debug)]
pub struct ReflectionPad2d<T: Float> {
    pub padding: (usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReflectionPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ReflectionPad2d expects at least 2-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_2d_reflect(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
        )?;
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReflectionPad2d);

/// Pads the last 3 dimensions using reflection.
#[derive(Debug)]
pub struct ReflectionPad3d<T: Float> {
    pub padding: (usize, usize, usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReflectionPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ReflectionPad3d expects at least 3-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_3d_reflect(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.padding.4,
            self.padding.5,
        )?;
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReflectionPad3d);

// ===========================================================================
// ReplicationPad1d / ReplicationPad2d / ReplicationPad3d
// ===========================================================================

/// Pads the last dimension by replicating the edge values.
#[derive(Debug)]
pub struct ReplicationPad1d<T: Float> {
    pub padding: (usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReplicationPad1d<T> {
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let data = input.data_vec()?;
        let (out, new_shape) =
            pad_1d_replicate(&data, input.shape(), self.padding.0, self.padding.1);
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReplicationPad1d);

/// Pads the last 2 dimensions by replicating edge values.
#[derive(Debug)]
pub struct ReplicationPad2d<T: Float> {
    pub padding: (usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReplicationPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 2 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ReplicationPad2d expects at least 2-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_2d_replicate(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReplicationPad2d);

/// Pads the last 3 dimensions by replicating edge values.
#[derive(Debug)]
pub struct ReplicationPad3d<T: Float> {
    pub padding: (usize, usize, usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> ReplicationPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() < 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ReplicationPad3d expects at least 3-D input, got {:?}",
                    input.shape()
                ),
            });
        }
        let data = input.data_vec()?;
        let (out, new_shape) = pad_3d_replicate(
            &data,
            input.shape(),
            self.padding.0,
            self.padding.1,
            self.padding.2,
            self.padding.3,
            self.padding.4,
            self.padding.5,
        );
        Tensor::from_storage(TensorStorage::cpu(out), new_shape, false)
    }
}

impl_padding_module!(ReplicationPad3d);

// ===========================================================================
// CircularPad — wraps data circularly (periodic boundary conditions)
// ===========================================================================

/// 1-D circular padding: wraps the input circularly.
///
/// Input: [N, C, W]. Pads the W dimension with circular (periodic) values.
/// Matches PyTorch's `nn.CircularPad1d`.
#[derive(Debug, Clone)]
pub struct CircularPad1d<T: Float> {
    pub padding: (usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> CircularPad1d<T> {
    pub fn new(padding: (usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CircularPad1d: expected 3-D input [N,C,W], got {:?}",
                    input.shape()
                ),
            });
        }
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CircularPad1d",
            });
        }
        let shape = input.shape();
        let (n, c, w) = (shape[0], shape[1], shape[2]);
        let (pl, pr) = self.padding;
        let new_w = w + pl + pr;
        let data = input.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; n * c * new_w];

        for batch in 0..n {
            for ch in 0..c {
                for ow in 0..new_w {
                    let iw = ((ow as isize - pl as isize).rem_euclid(w as isize)) as usize;
                    out[batch * c * new_w + ch * new_w + ow] = data[batch * c * w + ch * w + iw];
                }
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), vec![n, c, new_w], false)
    }
}

impl<T: Float> Default for CircularPad1d<T> {
    fn default() -> Self {
        Self::new((0, 0))
    }
}

impl_padding_module!(CircularPad1d);

/// 2-D circular padding. Input: [N, C, H, W].
/// Matches PyTorch's `nn.CircularPad2d`.
#[derive(Debug, Clone)]
pub struct CircularPad2d<T: Float> {
    pub padding: (usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> CircularPad2d<T> {
    pub fn new(padding: (usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CircularPad2d: expected 4-D input [N,C,H,W], got {:?}",
                    input.shape()
                ),
            });
        }
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CircularPad2d",
            });
        }
        let shape = input.shape();
        let (n, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
        let (pl, pr, pt, pb) = self.padding;
        let new_h = h + pt + pb;
        let new_w = w + pl + pr;
        let data = input.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; n * c * new_h * new_w];

        for batch in 0..n {
            for ch in 0..c {
                for oh in 0..new_h {
                    let ih = ((oh as isize - pt as isize).rem_euclid(h as isize)) as usize;
                    for ow in 0..new_w {
                        let iw = ((ow as isize - pl as isize).rem_euclid(w as isize)) as usize;
                        out[batch * c * new_h * new_w + ch * new_h * new_w + oh * new_w + ow] =
                            data[batch * c * h * w + ch * h * w + ih * w + iw];
                    }
                }
            }
        }

        Tensor::from_storage(TensorStorage::cpu(out), vec![n, c, new_h, new_w], false)
    }
}

impl<T: Float> Default for CircularPad2d<T> {
    fn default() -> Self {
        Self::new((0, 0, 0, 0))
    }
}

impl_padding_module!(CircularPad2d);

/// 3-D circular padding. Input: [N, C, D, H, W].
/// Matches PyTorch's `nn.CircularPad3d`.
#[derive(Debug, Clone)]
pub struct CircularPad3d<T: Float> {
    pub padding: (usize, usize, usize, usize, usize, usize),
    training: bool,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float> CircularPad3d<T> {
    pub fn new(padding: (usize, usize, usize, usize, usize, usize)) -> Self {
        Self {
            padding,
            training: true,
            _phantom: std::marker::PhantomData,
        }
    }

    fn pad(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        if input.ndim() != 5 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "CircularPad3d: expected 5-D input [N,C,D,H,W], got {:?}",
                    input.shape()
                ),
            });
        }
        if input.is_cuda() {
            return Err(FerrotorchError::NotImplementedOnCuda {
                op: "CircularPad3d",
            });
        }
        let shape = input.shape();
        let (n, c, d, h, w) = (shape[0], shape[1], shape[2], shape[3], shape[4]);
        let (pl, pr, pt, pb, pf, pk) = self.padding;
        let (new_d, new_h, new_w) = (d + pf + pk, h + pt + pb, w + pl + pr);
        let data = input.data()?;
        let zero = <T as num_traits::Zero>::zero();
        let mut out = vec![zero; n * c * new_d * new_h * new_w];

        for batch in 0..n {
            for ch in 0..c {
                for od in 0..new_d {
                    let id = ((od as isize - pf as isize).rem_euclid(d as isize)) as usize;
                    for oh in 0..new_h {
                        let ih = ((oh as isize - pt as isize).rem_euclid(h as isize)) as usize;
                        for ow in 0..new_w {
                            let iw = ((ow as isize - pl as isize).rem_euclid(w as isize)) as usize;
                            out[batch * c * new_d * new_h * new_w
                                + ch * new_d * new_h * new_w
                                + od * new_h * new_w
                                + oh * new_w
                                + ow] = data
                                [batch * c * d * h * w + ch * d * h * w + id * h * w + ih * w + iw];
                        }
                    }
                }
            }
        }

        Tensor::from_storage(
            TensorStorage::cpu(out),
            vec![n, c, new_d, new_h, new_w],
            false,
        )
    }
}

impl<T: Float> Default for CircularPad3d<T> {
    fn default() -> Self {
        Self::new((0, 0, 0, 0, 0, 0))
    }
}

impl_padding_module!(CircularPad3d);

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::module::Module;

    fn t(data: &[f32], shape: &[usize]) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(data.to_vec()), shape.to_vec(), false).unwrap()
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
            assert!((a - e).abs() < tol, "index {i}: actual={a} expected={e}");
        }
    }

    // -----------------------------------------------------------------------
    // ConstantPad1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_pad1d_basic() {
        let pad = ConstantPad1d::<f32>::new((2, 3), 9.0);
        let input = t(&[1.0, 2.0, 3.0], &[1, 1, 3]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 8]);
        assert_close(
            output.data().unwrap(),
            &[9.0, 9.0, 1.0, 2.0, 3.0, 9.0, 9.0, 9.0],
            1e-7,
        );
    }

    // -----------------------------------------------------------------------
    // ZeroPad1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_pad1d() {
        let pad = ZeroPad1d::<f32>::new((1, 2));
        let input = t(&[1.0, 2.0, 3.0], &[3]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[6]);
        assert_close(
            output.data().unwrap(),
            &[0.0, 1.0, 2.0, 3.0, 0.0, 0.0],
            1e-7,
        );
    }

    // -----------------------------------------------------------------------
    // ZeroPad2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_pad2d() {
        let pad = ZeroPad2d::<f32>::new((1, 1, 1, 1));
        let input = t(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        #[rustfmt::skip]
        let expected = [
            0.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 2.0, 0.0,
            0.0, 3.0, 4.0, 0.0,
            0.0, 0.0, 0.0, 0.0,
        ];
        assert_close(output.data().unwrap(), &expected, 1e-7);
    }

    // -----------------------------------------------------------------------
    // ZeroPad3d
    // -----------------------------------------------------------------------

    #[test]
    fn test_zero_pad3d_shape() {
        let pad = ZeroPad3d::<f32>::new((1, 1, 1, 1, 1, 1));
        let input = t(&[1.0; 2 * 2 * 2], &[1, 1, 2, 2, 2]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4, 4]);
    }

    // -----------------------------------------------------------------------
    // ReflectionPad1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflection_pad1d() {
        let pad = ReflectionPad1d::<f32>::new((2, 2));
        // input = [1, 2, 3, 4]
        let input = t(&[1.0, 2.0, 3.0, 4.0], &[4]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8]);
        // Reflect left: [3, 2, | 1, 2, 3, 4 | 3, 2]
        assert_close(
            output.data().unwrap(),
            &[3.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 2.0],
            1e-7,
        );
    }

    #[test]
    fn test_reflection_pad1d_too_large() {
        let pad = ReflectionPad1d::<f32>::new((4, 0));
        let input = t(&[1.0, 2.0, 3.0], &[3]); // size 3, pad 4 >= 3
        assert!(pad.forward(&input).is_err());
    }

    // -----------------------------------------------------------------------
    // ReflectionPad2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_reflection_pad2d() {
        let pad = ReflectionPad2d::<f32>::new((1, 1, 1, 1));
        #[rustfmt::skip]
        let input = t(&[
            1.0, 2.0, 3.0,
            4.0, 5.0, 6.0,
            7.0, 8.0, 9.0,
        ], &[1, 1, 3, 3]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 5, 5]);
        // Corner (0,0) should reflect to (1,1) in src = 5.0
        let out = output.data().unwrap();
        assert_close(&out[0..1], &[5.0], 1e-7); // top-left corner
    }

    // -----------------------------------------------------------------------
    // ReplicationPad1d
    // -----------------------------------------------------------------------

    #[test]
    fn test_replication_pad1d() {
        let pad = ReplicationPad1d::<f32>::new((2, 3));
        let input = t(&[1.0, 2.0, 3.0], &[3]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[8]);
        assert_close(
            output.data().unwrap(),
            &[1.0, 1.0, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0],
            1e-7,
        );
    }

    // -----------------------------------------------------------------------
    // ReplicationPad2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_replication_pad2d() {
        let pad = ReplicationPad2d::<f32>::new((1, 1, 1, 1));
        #[rustfmt::skip]
        let input = t(&[
            1.0, 2.0,
            3.0, 4.0,
        ], &[1, 1, 2, 2]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 4, 4]);
        #[rustfmt::skip]
        let expected = [
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_close(output.data().unwrap(), &expected, 1e-7);
    }

    // -----------------------------------------------------------------------
    // ConstantPad2d
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_pad2d() {
        let pad = ConstantPad2d::<f32>::new((1, 1, 1, 1), -1.0);
        let input = t(&[5.0, 6.0, 7.0, 8.0], &[2, 2]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[4, 4]);
        #[rustfmt::skip]
        let expected = [
            -1.0, -1.0, -1.0, -1.0,
            -1.0, 5.0, 6.0, -1.0,
            -1.0, 7.0, 8.0, -1.0,
            -1.0, -1.0, -1.0, -1.0,
        ];
        assert_close(output.data().unwrap(), &expected, 1e-7);
    }

    // -----------------------------------------------------------------------
    // ConstantPad3d
    // -----------------------------------------------------------------------

    #[test]
    fn test_constant_pad3d_shape() {
        let pad = ConstantPad3d::<f32>::new((1, 2, 1, 2, 1, 2), 0.0);
        let input = t(&vec![1.0; 3 * 4 * 5], &[1, 1, 3, 4, 5]);
        let output = pad.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1, 6, 7, 8]);
    }

    // -----------------------------------------------------------------------
    // Circular padding (1D)
    // -----------------------------------------------------------------------

    #[test]
    fn test_circular_pad_1d() {
        // input = [1, 2, 3, 4], pad_left=1, pad_right=2
        // circular: [4, 1, 2, 3, 4, 1, 2]
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let (out, new_shape) = pad_1d_circular(&data, &[4], 1, 2);
        assert_eq!(new_shape, &[7]);
        assert_close(&out, &[4.0, 1.0, 2.0, 3.0, 4.0, 1.0, 2.0], 1e-7);
    }

    // -----------------------------------------------------------------------
    // Padding mode enum
    // -----------------------------------------------------------------------

    #[test]
    fn test_padding_mode_eq() {
        assert_eq!(PaddingMode::Zeros, PaddingMode::Zeros);
        assert_ne!(PaddingMode::Zeros, PaddingMode::Reflect);
    }

    // -----------------------------------------------------------------------
    // Module trait: no parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_padding_module_no_params() {
        let pad = ZeroPad2d::<f32>::new((1, 1, 1, 1));
        assert!(pad.parameters().is_empty());
        assert!(pad.named_parameters().is_empty());
    }

    #[test]
    fn test_padding_module_train_eval() {
        let mut pad = ReflectionPad1d::<f32>::new((1, 1));
        assert!(pad.is_training());
        pad.eval();
        assert!(!pad.is_training());
        pad.train();
        assert!(pad.is_training());
    }
}
