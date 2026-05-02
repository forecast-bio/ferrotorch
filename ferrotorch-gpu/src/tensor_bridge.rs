//! Bridge between ferrotorch-core `Tensor<T>` and GPU operations.
//!
//! Because ferrotorch-core cannot depend on ferrotorch-gpu (that would be a
//! circular dependency), all GPU integration lives here. This module provides:
//!
//! - [`GpuTensor<T>`] — a wrapper combining a [`CudaBuffer<T>`] with shape
//!   metadata and the originating [`GpuDevice`].
//! - Transfer functions to move data between CPU [`Tensor`] and GPU.
//! - Elementwise arithmetic on [`GpuTensor`] backed by PTX kernels.
//!
//! # f32-only kernels
//!
//! The PTX kernels are currently f32-only. Operations on `GpuTensor<f64>` fall
//! back to a CPU round-trip (copy to host, compute, copy back). The type
//! parameter is kept for API consistency — once f64 PTX kernels are added,
//! the fallback disappears transparently.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::blas::{gpu_matmul_f32, gpu_matmul_f64};
use crate::buffer::CudaBuffer;
use crate::conv::gpu_conv2d_f32;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::kernels::{
    gpu_add, gpu_add_f64, gpu_mul, gpu_mul_f64, gpu_neg, gpu_neg_f64, gpu_relu, gpu_relu_f64,
    gpu_sub, gpu_sub_f64,
};
use crate::transfer::{cpu_to_gpu, gpu_to_cpu};

// ---------------------------------------------------------------------------
// GpuFloat — Float + DeviceRepr (when cuda is enabled)
// ---------------------------------------------------------------------------
//
// cudarc's transfer functions require `T: DeviceRepr`. Both f32 and f64
// implement it, but we can't unconditionally name the trait when the `cuda`
// feature is off (cudarc isn't compiled). This helper trait bridges the gap.

/// Trait alias: `Float` types that can be transferred to/from GPU.
///
/// When the `cuda` feature is enabled this adds `cudarc::driver::DeviceRepr`.
/// When disabled, it is identical to [`Float`].
#[cfg(feature = "cuda")]
pub trait GpuFloat: Float + cudarc::driver::DeviceRepr {}

#[cfg(feature = "cuda")]
impl GpuFloat for f32 {}
#[cfg(feature = "cuda")]
impl GpuFloat for f64 {}

#[cfg(not(feature = "cuda"))]
pub trait GpuFloat: Float {}

#[cfg(not(feature = "cuda"))]
impl GpuFloat for f32 {}
#[cfg(not(feature = "cuda"))]
impl GpuFloat for f64 {}

// ---------------------------------------------------------------------------
// GpuTensor
// ---------------------------------------------------------------------------

/// A tensor residing on a CUDA GPU.
///
/// Wraps a [`CudaBuffer<T>`] with shape metadata and a reference to the
/// [`GpuDevice`] that owns the memory. Created by [`tensor_to_gpu`] or
/// the convenience functions [`cuda`] / [`cuda_default`].
///
/// Convert back to a CPU [`Tensor`] with [`GpuTensor::cpu`] or the free
/// function [`tensor_to_cpu`].
pub struct GpuTensor<T: GpuFloat> {
    buffer: CudaBuffer<T>,
    shape: Vec<usize>,
    device: GpuDevice,
}

impl<T: GpuFloat> GpuTensor<T> {
    /// The shape of this tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Total number of elements.
    #[inline]
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// The GPU device that holds this tensor's data.
    #[inline]
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    /// Borrow the underlying [`CudaBuffer`].
    #[inline]
    pub fn buffer(&self) -> &CudaBuffer<T> {
        &self.buffer
    }

    /// Number of dimensions.
    #[inline]
    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Copy this tensor back to CPU, returning a [`Tensor<T>`].
    ///
    /// This is a convenience wrapper around [`tensor_to_cpu`].
    pub fn cpu(&self) -> FerrotorchResult<Tensor<T>> {
        tensor_to_cpu(self)
    }
}

impl<T: GpuFloat> std::fmt::Debug for GpuTensor<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuTensor")
            .field("shape", &self.shape)
            .field("numel", &self.numel())
            .field("device_ordinal", &self.device.ordinal())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Arithmetic operations
// ---------------------------------------------------------------------------

/// Returns `true` if `T` is `f32` (the type our PTX kernels support).
#[inline]
fn is_f32<T: GpuFloat>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<f32>()
}

/// Returns `true` if `T` is `f64`.
#[inline]
fn is_f64<T: GpuFloat>() -> bool {
    std::any::TypeId::of::<T>() == std::any::TypeId::of::<f64>()
}

/// Shape-validation helper for binary operations.
fn validate_shapes<T: GpuFloat>(a: &GpuTensor<T>, b: &GpuTensor<T>) -> GpuResult<()> {
    if a.shape() != b.shape() {
        return Err(GpuError::LengthMismatch {
            a: a.numel(),
            b: b.numel(),
        });
    }
    if a.device.ordinal() != b.device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: a.device.ordinal(),
            got: b.device.ordinal(),
        });
    }
    Ok(())
}

impl<T: GpuFloat> GpuTensor<T> {
    /// Elementwise addition: `out[i] = self[i] + other[i]`.
    ///
    /// Uses a PTX kernel for `f32`; falls back to CPU round-trip for `f64`.
    ///
    /// # Errors
    ///
    /// - [`GpuError::LengthMismatch`] if shapes differ.
    /// - [`GpuError::DeviceMismatch`] if tensors are on different devices.
    /// - [`GpuError::Driver`] on CUDA runtime errors.
    pub fn add(&self, other: &GpuTensor<T>) -> GpuResult<GpuTensor<T>> {
        validate_shapes(self, other)?;
        if is_f32::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f32>(&self.buffer) };
            let b_buf = unsafe { transmute_buffer_ref::<T, f32>(&other.buffer) };
            let out_buf = gpu_add(a_buf, b_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f32, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else if is_f64::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f64>(&self.buffer) };
            let b_buf = unsafe { transmute_buffer_ref::<T, f64>(&other.buffer) };
            let out_buf = gpu_add_f64(a_buf, b_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f64, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else {
            binary_cpu_fallback(self, other, |a, b| a + b)
        }
    }

    /// Elementwise subtraction: `out[i] = self[i] - other[i]`.
    ///
    /// Uses a PTX kernel for `f32`; falls back to CPU round-trip for `f64`.
    pub fn sub(&self, other: &GpuTensor<T>) -> GpuResult<GpuTensor<T>> {
        validate_shapes(self, other)?;
        if is_f32::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f32>(&self.buffer) };
            let b_buf = unsafe { transmute_buffer_ref::<T, f32>(&other.buffer) };
            let out_buf = gpu_sub(a_buf, b_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f32, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else if is_f64::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f64>(&self.buffer) };
            let b_buf = unsafe { transmute_buffer_ref::<T, f64>(&other.buffer) };
            let out_buf = gpu_sub_f64(a_buf, b_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f64, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else {
            binary_cpu_fallback(self, other, |a, b| a - b)
        }
    }

    /// Elementwise multiplication: `out[i] = self[i] * other[i]`.
    ///
    /// Uses a PTX kernel for `f32`; falls back to CPU round-trip for `f64`.
    pub fn mul(&self, other: &GpuTensor<T>) -> GpuResult<GpuTensor<T>> {
        validate_shapes(self, other)?;
        if is_f32::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f32>(&self.buffer) };
            let b_buf = unsafe { transmute_buffer_ref::<T, f32>(&other.buffer) };
            let out_buf = gpu_mul(a_buf, b_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f32, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else if is_f64::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f64>(&self.buffer) };
            let b_buf = unsafe { transmute_buffer_ref::<T, f64>(&other.buffer) };
            let out_buf = gpu_mul_f64(a_buf, b_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f64, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else {
            binary_cpu_fallback(self, other, |a, b| a * b)
        }
    }

    /// Elementwise negation: `out[i] = -self[i]`.
    ///
    /// Uses a PTX kernel for `f32`; falls back to CPU round-trip for `f64`.
    pub fn neg(&self) -> GpuResult<GpuTensor<T>> {
        if is_f32::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f32>(&self.buffer) };
            let out_buf = gpu_neg(a_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f32, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else if is_f64::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f64>(&self.buffer) };
            let out_buf = gpu_neg_f64(a_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f64, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else {
            unary_cpu_fallback(self, |x| -x)
        }
    }

    /// Elementwise ReLU: `out[i] = max(self[i], 0)`.
    ///
    /// Uses a PTX kernel for `f32`; falls back to CPU round-trip for `f64`.
    pub fn relu(&self) -> GpuResult<GpuTensor<T>> {
        if is_f32::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f32>(&self.buffer) };
            let out_buf = gpu_relu(a_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f32, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else if is_f64::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f64>(&self.buffer) };
            let out_buf = gpu_relu_f64(a_buf, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f64, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: self.shape.clone(),
                device: self.device.clone(),
            })
        } else {
            unary_cpu_fallback(self, |x| {
                let z = <T as num_traits::Zero>::zero();
                if x > z { x } else { z }
            })
        }
    }

    /// Matrix multiplication: `C = self @ other`.
    ///
    /// Both tensors must be 2-D. `self` has shape `[m, k]` and `other` has
    /// shape `[k, n]`. The result has shape `[m, n]`.
    ///
    /// Uses cuBLAS SGEMM for `f32` and DGEMM for `f64`.
    ///
    /// # Errors
    ///
    /// - [`GpuError::ShapeMismatch`] if either tensor is not 2-D or if the
    ///   inner dimensions do not match (`self.shape[1] != other.shape[0]`).
    /// - [`GpuError::DeviceMismatch`] if tensors are on different devices.
    /// - [`GpuError::Blas`] on cuBLAS runtime errors.
    pub fn matmul(&self, other: &GpuTensor<T>) -> GpuResult<GpuTensor<T>> {
        // Validate 2-D shapes.
        if self.ndim() != 2 {
            return Err(GpuError::ShapeMismatch {
                op: "matmul",
                expected: vec![0, 0], // placeholder: "expected 2-D"
                got: self.shape.clone(),
            });
        }
        if other.ndim() != 2 {
            return Err(GpuError::ShapeMismatch {
                op: "matmul",
                expected: vec![0, 0],
                got: other.shape.clone(),
            });
        }

        let m = self.shape[0];
        let k = self.shape[1];
        let k2 = other.shape[0];
        let n = other.shape[1];

        if k != k2 {
            return Err(GpuError::ShapeMismatch {
                op: "matmul",
                expected: vec![k, n],
                got: vec![k2, n],
            });
        }

        if self.device.ordinal() != other.device.ordinal() {
            return Err(GpuError::DeviceMismatch {
                expected: self.device.ordinal(),
                got: other.device.ordinal(),
            });
        }

        if is_f32::<T>() {
            let a_buf = unsafe { transmute_buffer_ref::<T, f32>(&self.buffer) };
            let b_buf = unsafe { transmute_buffer_ref::<T, f32>(&other.buffer) };
            let out_buf = gpu_matmul_f32(a_buf, b_buf, m, k, n, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f32, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: vec![m, n],
                device: self.device.clone(),
            })
        } else {
            // f64 path
            let a_buf = unsafe { transmute_buffer_ref::<T, f64>(&self.buffer) };
            let b_buf = unsafe { transmute_buffer_ref::<T, f64>(&other.buffer) };
            let out_buf = gpu_matmul_f64(a_buf, b_buf, m, k, n, &self.device)?;
            let out_buf = unsafe { transmute_buffer::<f64, T>(out_buf) };
            Ok(GpuTensor {
                buffer: out_buf,
                shape: vec![m, n],
                device: self.device.clone(),
            })
        }
    }

    /// 2-D convolution: `output = conv2d(self, weight, bias)`.
    ///
    /// Uses im2col (CPU) + cuBLAS GEMM (GPU) — no cuDNN required.
    ///
    /// `self` must have shape `[B, C_in, H, W]` and `weight` must have
    /// shape `[C_out, C_in, kH, kW]`. `bias`, if provided, must have
    /// shape `[C_out]`. The result has shape `[B, C_out, H_out, W_out]`.
    ///
    /// Currently only supports `f32`. For `f64` tensors, returns
    /// [`GpuError::ShapeMismatch`] (f64 conv path not yet implemented).
    ///
    /// # Errors
    ///
    /// - [`GpuError::ShapeMismatch`] if tensor dimensions are wrong, channel
    ///   counts don't match, or if `T` is not `f32`.
    /// - [`GpuError::DeviceMismatch`] if tensors are on different devices.
    /// - [`GpuError::Blas`] on cuBLAS runtime errors.
    pub fn conv2d(
        &self,
        weight: &GpuTensor<T>,
        bias: Option<&GpuTensor<T>>,
        stride: (usize, usize),
        padding: (usize, usize),
    ) -> GpuResult<GpuTensor<T>> {
        // Validate 4-D input.
        if self.ndim() != 4 {
            return Err(GpuError::ShapeMismatch {
                op: "conv2d",
                expected: vec![0, 0, 0, 0],
                got: self.shape.clone(),
            });
        }
        // Validate 4-D weight.
        if weight.ndim() != 4 {
            return Err(GpuError::ShapeMismatch {
                op: "conv2d",
                expected: vec![0, 0, 0, 0],
                got: weight.shape.clone(),
            });
        }
        // Validate 1-D bias.
        if let Some(b) = bias {
            if b.ndim() != 1 {
                return Err(GpuError::ShapeMismatch {
                    op: "conv2d",
                    expected: vec![weight.shape[0]],
                    got: b.shape.clone(),
                });
            }
        }
        // Device consistency.
        if self.device.ordinal() != weight.device.ordinal() {
            return Err(GpuError::DeviceMismatch {
                expected: self.device.ordinal(),
                got: weight.device.ordinal(),
            });
        }
        if let Some(b) = bias {
            if self.device.ordinal() != b.device.ordinal() {
                return Err(GpuError::DeviceMismatch {
                    expected: self.device.ordinal(),
                    got: b.device.ordinal(),
                });
            }
        }

        if !is_f32::<T>() {
            return Err(GpuError::ShapeMismatch {
                op: "conv2d",
                expected: vec![],
                got: vec![],
            });
        }

        let input_shape: [usize; 4] = [self.shape[0], self.shape[1], self.shape[2], self.shape[3]];
        let weight_shape: [usize; 4] = [
            weight.shape[0],
            weight.shape[1],
            weight.shape[2],
            weight.shape[3],
        ];

        let a_buf = unsafe { transmute_buffer_ref::<T, f32>(&self.buffer) };
        let w_buf = unsafe { transmute_buffer_ref::<T, f32>(&weight.buffer) };
        let b_buf = bias.map(|b| unsafe { transmute_buffer_ref::<T, f32>(&b.buffer) });

        let (out_buf, out_shape) = gpu_conv2d_f32(
            a_buf,
            w_buf,
            b_buf,
            input_shape,
            weight_shape,
            stride,
            padding,
            &self.device,
        )?;

        let out_buf = unsafe { transmute_buffer::<f32, T>(out_buf) };
        Ok(GpuTensor {
            buffer: out_buf,
            shape: out_shape.to_vec(),
            device: self.device.clone(),
        })
    }
}

// ---------------------------------------------------------------------------
// Transmute helpers for CudaBuffer<T> <-> CudaBuffer<f32>
// ---------------------------------------------------------------------------
//
// The PTX kernel functions take `&CudaBuffer<f32>`. When T == f32 the
// CudaBuffer<T> has an identical layout, so we can safely reinterpret.
// These helpers are only called after an `is_f32::<T>()` guard.

/// Reinterpret a `&CudaBuffer<T>` as `&CudaBuffer<U>`.
///
/// # Safety
///
/// Caller must have verified `size_of::<T>() == size_of::<U>()` and
/// `align_of::<T>() == align_of::<U>()`.
#[cfg(feature = "cuda")]
unsafe fn transmute_buffer_ref<T, U>(buf: &CudaBuffer<T>) -> &CudaBuffer<U> {
    debug_assert_eq!(std::mem::size_of::<T>(), std::mem::size_of::<U>());
    debug_assert_eq!(std::mem::align_of::<T>(), std::mem::align_of::<U>());
    // CudaBuffer<T> and CudaBuffer<U> have identical layout when T and U
    // are the same size — CudaSlice is a device pointer + length, both
    // size-independent.
    unsafe { &*(buf as *const CudaBuffer<T> as *const CudaBuffer<U>) }
}

/// Reinterpret an owned `CudaBuffer<U>` as `CudaBuffer<T>`.
///
/// # Safety
///
/// Caller must have verified `size_of::<U>() == size_of::<T>()` and
/// `align_of::<U>() == align_of::<T>()`.
#[cfg(feature = "cuda")]
unsafe fn transmute_buffer<U, T>(buf: CudaBuffer<U>) -> CudaBuffer<T> {
    debug_assert_eq!(std::mem::size_of::<U>(), std::mem::size_of::<T>());
    debug_assert_eq!(std::mem::align_of::<U>(), std::mem::align_of::<T>());
    // Move the buffer without running U's drop — T's drop will handle it.
    let result = unsafe { std::ptr::read(&buf as *const CudaBuffer<U> as *const CudaBuffer<T>) };
    std::mem::forget(buf);
    result
}

// Stubs when cuda is not enabled — the transmute helpers are never called
// because all kernel functions return NoCudaFeature, but we still need them
// to exist so the module compiles.

#[cfg(not(feature = "cuda"))]
unsafe fn transmute_buffer_ref<T, U>(buf: &CudaBuffer<T>) -> &CudaBuffer<U> {
    let _ = buf;
    unreachable!("transmute_buffer_ref called without cuda feature")
}

#[cfg(not(feature = "cuda"))]
unsafe fn transmute_buffer<U, T>(buf: CudaBuffer<U>) -> CudaBuffer<T> {
    let _ = buf;
    unreachable!("transmute_buffer called without cuda feature")
}

// ---------------------------------------------------------------------------
// CPU fallback helpers for non-f32 types
// ---------------------------------------------------------------------------

/// Binary operation fallback: copy both operands to CPU, apply `op`, copy back.
fn binary_cpu_fallback<T: GpuFloat>(
    a: &GpuTensor<T>,
    b: &GpuTensor<T>,
    op: fn(T, T) -> T,
) -> GpuResult<GpuTensor<T>> {
    let a_cpu = gpu_to_cpu(&a.buffer, &a.device)?;
    let b_cpu = gpu_to_cpu(&b.buffer, &b.device)?;
    let result: Vec<T> = a_cpu
        .iter()
        .zip(b_cpu.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();
    let out_buf = cpu_to_gpu(&result, &a.device)?;
    Ok(GpuTensor {
        buffer: out_buf,
        shape: a.shape.clone(),
        device: a.device.clone(),
    })
}

/// Unary operation fallback: copy operand to CPU, apply `op`, copy back.
fn unary_cpu_fallback<T: GpuFloat>(a: &GpuTensor<T>, op: fn(T) -> T) -> GpuResult<GpuTensor<T>> {
    let a_cpu = gpu_to_cpu(&a.buffer, &a.device)?;
    let result: Vec<T> = a_cpu.iter().map(|&x| op(x)).collect();
    let out_buf = cpu_to_gpu(&result, &a.device)?;
    Ok(GpuTensor {
        buffer: out_buf,
        shape: a.shape.clone(),
        device: a.device.clone(),
    })
}

// ---------------------------------------------------------------------------
// Transfer functions: Tensor <-> GpuTensor
// ---------------------------------------------------------------------------

/// Move a CPU [`Tensor<T>`] to a GPU, returning a [`GpuTensor<T>`].
///
/// The tensor must be contiguous and reside on `Device::Cpu`. Its flat data
/// is copied to the given [`GpuDevice`] via a host-to-device transfer.
///
/// # Errors
///
/// - Returns [`GpuError::Driver`] on CUDA allocation or copy failures.
/// - Returns [`GpuError::LengthMismatch`] if the tensor is not contiguous.
pub fn tensor_to_gpu<T: GpuFloat>(
    tensor: &Tensor<T>,
    device: &GpuDevice,
) -> GpuResult<GpuTensor<T>> {
    // Ensure the tensor is contiguous so data() gives a proper flat slice.
    if !tensor.is_contiguous() {
        return Err(GpuError::LengthMismatch {
            a: tensor.numel(),
            b: tensor.data().map_or(0, |d| d.len()),
        });
    }

    // Extract the flat data from the CPU tensor.
    let data = tensor.data().map_err(|_e| GpuError::InvalidDevice {
        ordinal: device.ordinal(),
        count: 0,
    })?;

    let buffer = cpu_to_gpu(data, device)?;
    Ok(GpuTensor {
        buffer,
        shape: tensor.shape().to_vec(),
        device: device.clone(),
    })
}

/// Move a [`GpuTensor<T>`] back to CPU, returning a [`Tensor<T>`].
///
/// Performs a device-to-host copy and wraps the resulting `Vec<T>` in a
/// new leaf [`Tensor`] with `requires_grad = false`.
///
/// # Errors
///
/// - Returns an error if the device-to-host copy fails.
pub fn tensor_to_cpu<T: GpuFloat>(gpu_tensor: &GpuTensor<T>) -> FerrotorchResult<Tensor<T>> {
    let host_data = gpu_to_cpu(&gpu_tensor.buffer, &gpu_tensor.device).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("GPU-to-CPU transfer failed: {e}"),
        }
    })?;

    let storage = TensorStorage::cpu(host_data);
    Tensor::from_storage(storage, gpu_tensor.shape.clone(), false)
}

// ---------------------------------------------------------------------------
// Convenience free functions
// ---------------------------------------------------------------------------

/// Move a CPU [`Tensor<T>`] to CUDA device with the given ordinal.
///
/// Shorthand for creating a [`GpuDevice`] and calling [`tensor_to_gpu`].
///
/// # Errors
///
/// Returns a [`GpuError`] if the device cannot be initialized or the
/// transfer fails.
pub fn cuda<T: GpuFloat>(tensor: &Tensor<T>, ordinal: usize) -> GpuResult<GpuTensor<T>> {
    let device = GpuDevice::new(ordinal)?;
    tensor_to_gpu(tensor, &device)
}

/// Move a CPU [`Tensor<T>`] to CUDA device 0.
///
/// Equivalent to `cuda(tensor, 0)`.
pub fn cuda_default<T: GpuFloat>(tensor: &Tensor<T>) -> GpuResult<GpuTensor<T>> {
    cuda(tensor, 0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use ferrotorch_core::{Tensor, TensorStorage};

    /// Helper: create a CPU tensor from a flat vec with the given shape.
    fn cpu_tensor(data: Vec<f32>, shape: Vec<usize>) -> Tensor<f32> {
        let storage = TensorStorage::cpu(data);
        Tensor::from_storage(storage, shape, false).expect("cpu_tensor")
    }

    // -- round-trip -----------------------------------------------------------

    #[test]
    fn tensor_to_gpu_round_trip() {
        let t = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let gpu = cuda_default(&t).expect("cuda_default");
        let back = gpu.cpu().expect("cpu()");

        assert_eq!(back.shape(), &[2, 3]);
        assert_eq!(back.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // -- shape preservation ---------------------------------------------------

    #[test]
    fn gpu_tensor_shape_preserved() {
        let t = cpu_tensor(vec![1.0; 24], vec![2, 3, 4]);
        let gpu = cuda_default(&t).expect("cuda_default");

        assert_eq!(gpu.shape(), &[2, 3, 4]);
        assert_eq!(gpu.numel(), 24);
        assert_eq!(gpu.ndim(), 3);
    }

    // -- add ------------------------------------------------------------------

    #[test]
    fn gpu_tensor_add() {
        let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![4]);
        let b = cpu_tensor(vec![10.0, 20.0, 30.0, 40.0], vec![4]);

        let device = GpuDevice::new(0).expect("CUDA device 0");
        let ga = tensor_to_gpu(&a, &device).expect("a to gpu");
        let gb = tensor_to_gpu(&b, &device).expect("b to gpu");

        let gc = ga.add(&gb).expect("gpu add");
        let result = gc.cpu().expect("cpu");

        assert_eq!(result.shape(), &[4]);
        let data = result.data().unwrap();
        assert!((data[0] - 11.0).abs() < 1e-6);
        assert!((data[1] - 22.0).abs() < 1e-6);
        assert!((data[2] - 33.0).abs() < 1e-6);
        assert!((data[3] - 44.0).abs() < 1e-6);
    }

    // -- relu -----------------------------------------------------------------

    #[test]
    fn gpu_tensor_relu() {
        let t = cpu_tensor(vec![-3.0, -1.0, 0.0, 1.0, 3.0], vec![5]);
        let gpu = cuda_default(&t).expect("cuda_default");
        let out = gpu.relu().expect("relu");
        let result = out.cpu().expect("cpu");

        let data = result.data().unwrap();
        assert!((data[0] - 0.0).abs() < 1e-6);
        assert!((data[1] - 0.0).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - 1.0).abs() < 1e-6);
        assert!((data[4] - 3.0).abs() < 1e-6);
    }

    // -- tensor_to_cpu values -------------------------------------------------

    #[test]
    fn tensor_to_cpu_correct_values() {
        let original = vec![0.5, -1.5, 2.25, 0.0, 100.0, -0.001];
        let t = cpu_tensor(original.clone(), vec![2, 3]);
        let gpu = cuda_default(&t).expect("cuda_default");
        let back = tensor_to_cpu(&gpu).expect("tensor_to_cpu");

        let data = back.data().unwrap();
        for (i, (&got, &expected)) in data.iter().zip(original.iter()).enumerate() {
            assert!(
                (got - expected).abs() < 1e-6,
                "element {i}: got {got}, expected {expected}",
            );
        }
    }

    // -- sub ------------------------------------------------------------------

    #[test]
    fn gpu_tensor_sub() {
        let a = cpu_tensor(vec![10.0, 20.0, 30.0], vec![3]);
        let b = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);

        let device = GpuDevice::new(0).expect("CUDA device 0");
        let ga = tensor_to_gpu(&a, &device).expect("a to gpu");
        let gb = tensor_to_gpu(&b, &device).expect("b to gpu");

        let gc = ga.sub(&gb).expect("gpu sub");
        let result = gc.cpu().expect("cpu");
        let data = result.data().unwrap();
        assert!((data[0] - 9.0).abs() < 1e-6);
        assert!((data[1] - 18.0).abs() < 1e-6);
        assert!((data[2] - 27.0).abs() < 1e-6);
    }

    // -- mul ------------------------------------------------------------------

    #[test]
    fn gpu_tensor_mul() {
        let a = cpu_tensor(vec![2.0, 3.0, 4.0], vec![3]);
        let b = cpu_tensor(vec![10.0, 10.0, 10.0], vec![3]);

        let device = GpuDevice::new(0).expect("CUDA device 0");
        let ga = tensor_to_gpu(&a, &device).expect("a to gpu");
        let gb = tensor_to_gpu(&b, &device).expect("b to gpu");

        let gc = ga.mul(&gb).expect("gpu mul");
        let result = gc.cpu().expect("cpu");
        let data = result.data().unwrap();
        assert!((data[0] - 20.0).abs() < 1e-6);
        assert!((data[1] - 30.0).abs() < 1e-6);
        assert!((data[2] - 40.0).abs() < 1e-6);
    }

    // -- neg ------------------------------------------------------------------

    #[test]
    fn gpu_tensor_neg() {
        let t = cpu_tensor(vec![1.0, -2.0, 0.0, 3.5], vec![4]);
        let gpu = cuda_default(&t).expect("cuda_default");
        let out = gpu.neg().expect("neg");
        let result = out.cpu().expect("cpu");
        let data = result.data().unwrap();
        assert!((data[0] - (-1.0)).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 0.0).abs() < 1e-6);
        assert!((data[3] - (-3.5)).abs() < 1e-6);
    }

    // -- matmul ---------------------------------------------------------------

    #[test]
    fn gpu_tensor_matmul_basic() {
        // A = [[1, 2, 3], [4, 5, 6]]  (2x3)
        // B = [[7, 8], [9, 10], [11, 12]]  (3x2)
        // C = [[58, 64], [139, 154]]  (2x2)
        let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = cpu_tensor(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], vec![3, 2]);

        let device = GpuDevice::new(0).expect("CUDA device 0");
        let ga = tensor_to_gpu(&a, &device).expect("a to gpu");
        let gb = tensor_to_gpu(&b, &device).expect("b to gpu");

        let gc = ga.matmul(&gb).expect("gpu matmul");
        assert_eq!(gc.shape(), &[2, 2]);

        let result = gc.cpu().expect("cpu");
        let data = result.data().unwrap();
        assert!((data[0] - 58.0).abs() < 1e-4);
        assert!((data[1] - 64.0).abs() < 1e-4);
        assert!((data[2] - 139.0).abs() < 1e-4);
        assert!((data[3] - 154.0).abs() < 1e-4);
    }

    #[test]
    fn gpu_tensor_matmul_identity() {
        let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let i = cpu_tensor(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]);

        let device = GpuDevice::new(0).expect("CUDA device 0");
        let ga = tensor_to_gpu(&a, &device).expect("a to gpu");
        let gi = tensor_to_gpu(&i, &device).expect("i to gpu");

        let gc = ga.matmul(&gi).expect("gpu matmul identity");
        let result = gc.cpu().expect("cpu");
        let data = result.data().unwrap();
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn gpu_tensor_matmul_inner_dim_mismatch() {
        // A is [2, 3], B is [2, 2] -- inner dims 3 != 2
        let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        let b = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);

        let device = GpuDevice::new(0).expect("CUDA device 0");
        let ga = tensor_to_gpu(&a, &device).expect("a to gpu");
        let gb = tensor_to_gpu(&b, &device).expect("b to gpu");

        let err = ga.matmul(&gb).unwrap_err();
        match err {
            GpuError::ShapeMismatch { op: "matmul", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    #[test]
    fn gpu_tensor_matmul_not_2d() {
        // A is [6] (1-D), should fail
        let a = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![6]);
        let b = cpu_tensor(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![3, 2]);

        let device = GpuDevice::new(0).expect("CUDA device 0");
        let ga = tensor_to_gpu(&a, &device).expect("a to gpu");
        let gb = tensor_to_gpu(&b, &device).expect("b to gpu");

        let err = ga.matmul(&gb).unwrap_err();
        match err {
            GpuError::ShapeMismatch { op: "matmul", .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- shape mismatch error -------------------------------------------------

    #[test]
    fn gpu_tensor_add_shape_mismatch() {
        let a = cpu_tensor(vec![1.0, 2.0, 3.0], vec![3]);
        let b = cpu_tensor(vec![1.0, 2.0], vec![2]);

        let device = GpuDevice::new(0).expect("CUDA device 0");
        let ga = tensor_to_gpu(&a, &device).expect("a to gpu");
        let gb = tensor_to_gpu(&b, &device).expect("b to gpu");

        let err = ga.add(&gb).unwrap_err();
        match err {
            GpuError::LengthMismatch { .. } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- empty tensor ---------------------------------------------------------

    #[test]
    fn gpu_tensor_empty_round_trip() {
        let t = cpu_tensor(vec![], vec![0]);
        let gpu = cuda_default(&t).expect("cuda_default");
        assert_eq!(gpu.numel(), 0);
        assert_eq!(gpu.shape(), &[0]);

        let back = gpu.cpu().expect("cpu");
        assert_eq!(back.shape(), &[0]);
        assert_eq!(back.data().unwrap().len(), 0);
    }

    // -- scalar tensor --------------------------------------------------------

    #[test]
    fn gpu_tensor_scalar_round_trip() {
        let storage = TensorStorage::cpu(vec![42.0f32]);
        let t = Tensor::from_storage(storage, vec![], false).expect("scalar");
        let gpu = cuda_default(&t).expect("cuda_default");
        assert_eq!(gpu.shape(), &[] as &[usize]);
        assert_eq!(gpu.numel(), 1);

        let back = gpu.cpu().expect("cpu");
        assert!(back.is_scalar());
        assert!((back.item().unwrap() - 42.0).abs() < 1e-6);
    }
}
