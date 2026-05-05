//! Backward functions for elementwise arithmetic operations.
//!
//! Each operation has a backward struct implementing `GradFn<T>` and a public
//! function that performs the forward pass and attaches the grad_fn to the
//! result tensor when gradient tracking is enabled.

use std::any::TypeId;
use std::sync::Arc;

use crate::autograd::no_grad::{is_grad_enabled, no_grad};
use crate::dtype::Float;
use crate::error::{FerrotorchError, FerrotorchResult};
use crate::ops::elementwise::{fast_add, fast_mul, scalar_map, unary_map};
use crate::shape::broadcast_shapes;
use crate::storage::TensorStorage;
use crate::tensor::{GradFn, Tensor};

/// Returns `true` if `T` is `f64`.
#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

/// Returns `true` if `T` is `f32`.
#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Whether at least one of two tensors requires grad (and grad is enabled).
#[inline]
fn needs_grad<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> bool {
    is_grad_enabled() && (a.requires_grad() || b.requires_grad())
}

/// Whether a single tensor requires grad (and grad is enabled).
#[inline]
fn needs_grad_unary<T: Float>(a: &Tensor<T>) -> bool {
    is_grad_enabled() && a.requires_grad()
}

/// Reduce a gradient tensor to a target shape by summing over broadcast
/// dimensions.
///
/// When two tensors with different shapes are combined via broadcasting,
/// the backward pass must sum the gradient over the dimensions that were
/// broadcast to recover the correct gradient shape for each input.
///
/// Algorithm:
/// 1. If shapes already match, return `grad` as-is (clone).
/// 2. Left-pad `target_shape` with 1s to match grad's ndim.
/// 3. For each dimension where target has size 1 but grad has size > 1,
///    sum over that dimension.
/// 4. Reshape to `target_shape`.
///
/// For f32 GPU tensors, reduction is performed entirely on GPU via
/// `sum_axis_f32` — no CPU roundtrip.  Other dtypes fall back to a
/// CPU reduction loop and re-upload.
pub(crate) fn reduce_grad_to_shape<T: Float>(
    grad: &Tensor<T>,
    target_shape: &[usize],
) -> FerrotorchResult<Tensor<T>> {
    let grad_shape = grad.shape();

    // Fast path: shapes already match.
    if grad_shape == target_shape {
        return Ok(grad.clone());
    }

    // Scalar target: sum everything.
    if target_shape.is_empty() {
        // Use the reduction forward op which already handles GPU.
        return crate::grad_fns::reduction::sum(grad);
    }

    // GPU fast path for f32/f64: reduce each broadcast axis on-device.
    if grad.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let mut handle = backend.clone_buffer(grad.gpu_handle()?)?;
        let mut current_shape = grad.shape().to_vec();

        let target_ndim = target_shape.len();

        // First reduce leading dimensions that don't exist in target.
        while current_shape.len() > target_ndim {
            handle = if is_f32::<T>() {
                backend.sum_axis_f32(&handle, &current_shape, 0)?
            } else {
                backend.sum_axis_f64(&handle, &current_shape, 0)?
            };
            current_shape.remove(0);
        }

        // Then reduce dimensions where target has size 1 but grad has size > 1.
        for axis in 0..current_shape.len() {
            if axis < target_shape.len() && target_shape[axis] == 1 && current_shape[axis] > 1 {
                handle = if is_f32::<T>() {
                    backend.sum_axis_f32(&handle, &current_shape, axis)?
                } else {
                    backend.sum_axis_f64(&handle, &current_shape, axis)?
                };
                current_shape[axis] = 1;
            }
        }

        return Tensor::from_storage(TensorStorage::gpu(handle), target_shape.to_vec(), false);
    }

    // CPU path — non-f32/f64 GPU tensors have no GPU kernel, error out.
    if grad.is_cuda() {
        return Err(FerrotorchError::NotImplementedOnCuda {
            op: "broadcast_grad",
        });
    }

    let grad_data = grad.data()?;
    let grad_ndim = grad_shape.len();
    let target_ndim = target_shape.len();

    // Standard broadcasting requires grad_ndim >= target_ndim. The reverse
    // case (gradient has fewer dims than target) used to trigger an integer
    // underflow at `grad_ndim - target_ndim`. The graph::backward seed
    // construction in #498 fixed the most common cause (root.shape() instead
    // of a scalar []), but keep an explicit check here as a defense in
    // depth — better a clean error than a panic if any other path produces
    // a misshapen gradient. CL-498.
    if grad_ndim < target_ndim {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "reduce_grad_to_shape: gradient has {grad_ndim} dim(s) but target has {target_ndim} dim(s) ({grad_shape:?} -> {target_shape:?}). \
                 Standard broadcasting backward requires grad_ndim >= target_ndim."
            ),
        });
    }

    // Left-pad target_shape with 1s to match grad_ndim.
    let padded_target: Vec<usize> = if target_ndim < grad_ndim {
        let mut p = vec![1usize; grad_ndim - target_ndim];
        p.extend_from_slice(target_shape);
        p
    } else {
        target_shape.to_vec()
    };

    let out_numel: usize = target_shape.iter().product();
    let mut result = vec![<T as num_traits::Zero>::zero(); out_numel.max(1)];

    // Precompute target strides for flat index calculation.
    let mut target_strides = vec![1usize; target_ndim];
    for td in (0..target_ndim.saturating_sub(1)).rev() {
        target_strides[td] = target_strides[td + 1] * target_shape[td + 1];
    }

    let offset = grad_ndim - target_ndim; // number of leading 1-padded dims

    for (i, &grad_val) in grad_data.iter().enumerate() {
        // Decompose grad flat index into per-axis coordinates.
        let mut coords = [0usize; 16]; // support up to 16 dims
        let mut rem = i;
        for d in (0..grad_ndim).rev() {
            coords[d] = rem % grad_shape[d];
            rem /= grad_shape[d];
        }

        // Compute flat index in target by mapping each grad coord to
        // the corresponding target coord (collapsing broadcast dims).
        let mut flat = 0usize;
        for (td, &target_stride) in target_strides.iter().enumerate() {
            let gd = td + offset;
            let coord = if padded_target[gd] == 1 {
                0
            } else {
                coords[gd]
            };
            flat += coord * target_stride;
        }

        result[flat] += grad_val;
    }

    Tensor::from_storage(TensorStorage::cpu(result), target_shape.to_vec(), false)
}

// ===========================================================================
// add
// ===========================================================================

/// Backward node for `c = a + b`.
///
/// VJP: da = grad, db = grad.
#[derive(Debug)]
struct AddBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for AddBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            Some(reduce_grad_to_shape(grad_output, self.a.shape())?)
        } else {
            None
        };
        let db = if self.b.requires_grad() {
            Some(reduce_grad_to_shape(grad_output, self.b.shape())?)
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "AddBackward"
    }
}

/// Elementwise addition: `c = a + b`.
pub fn add<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    if let Some(out) = crate::meta_propagate::binary_broadcast(a, b)? {
        return Ok(out);
    }

    crate::profiler_hook::profile_op_scope("add", "tensor_op", &[a.shape(), b.shape()], || {
        add_inner(a, b)
    })
}

fn add_inner<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

        let needs_broadcast = a.shape() != b.shape();
        let (handle, out_shape) = if needs_broadcast {
            let out_shape = broadcast_shapes(a.shape(), b.shape())?;
            let h = if is_f64::<T>() {
                backend.broadcast_add_f64(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    a.shape(),
                    b.shape(),
                    &out_shape,
                )?
            } else {
                backend.broadcast_add_f32(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    a.shape(),
                    b.shape(),
                    &out_shape,
                )?
            };
            (h, out_shape)
        } else if is_f64::<T>() {
            (
                backend.add_f64(a.gpu_handle()?, b.gpu_handle()?)?,
                a.shape().to_vec(),
            )
        } else {
            (
                backend.add_f32(a.gpu_handle()?, b.gpu_handle()?)?,
                a.shape().to_vec(),
            )
        };
        let storage = TensorStorage::gpu(handle);

        if needs_grad(a, b) {
            Tensor::from_operation(
                storage,
                out_shape,
                Arc::new(AddBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Tensor::from_storage(storage, out_shape, false)
        }
    } else {
        let result = fast_add(a, b)?;

        if needs_grad(a, b) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(AddBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// sub
// ===========================================================================

/// Backward node for `c = a - b`.
///
/// VJP: da = grad, db = -grad.
#[derive(Debug)]
struct SubBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for SubBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            Some(reduce_grad_to_shape(grad_output, self.a.shape())?)
        } else {
            None
        };
        let db = if self.b.requires_grad() {
            let neg_grad = no_grad(|| neg(grad_output))?;
            Some(reduce_grad_to_shape(&neg_grad, self.b.shape())?)
        } else {
            None
        };
        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "SubBackward"
    }
}

/// Elementwise subtraction: `c = a - b`.
pub fn sub<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    if let Some(out) = crate::meta_propagate::binary_broadcast(a, b)? {
        return Ok(out);
    }

    crate::profiler_hook::profile_op_scope("sub", "tensor_op", &[a.shape(), b.shape()], || {
        sub_inner(a, b)
    })
}

fn sub_inner<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

        let needs_broadcast = a.shape() != b.shape();
        let (handle, out_shape) = if needs_broadcast {
            let out_shape = broadcast_shapes(a.shape(), b.shape())?;
            let h = if is_f64::<T>() {
                backend.broadcast_sub_f64(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    a.shape(),
                    b.shape(),
                    &out_shape,
                )?
            } else {
                backend.broadcast_sub_f32(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    a.shape(),
                    b.shape(),
                    &out_shape,
                )?
            };
            (h, out_shape)
        } else if is_f64::<T>() {
            (
                backend.sub_f64(a.gpu_handle()?, b.gpu_handle()?)?,
                a.shape().to_vec(),
            )
        } else {
            (
                backend.sub_f32(a.gpu_handle()?, b.gpu_handle()?)?,
                a.shape().to_vec(),
            )
        };
        let storage = TensorStorage::gpu(handle);

        if needs_grad(a, b) {
            Tensor::from_operation(
                storage,
                out_shape,
                Arc::new(SubBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Tensor::from_storage(storage, out_shape, false)
        }
    } else {
        let result = crate::ops::elementwise::fast_sub(a, b)?;

        if needs_grad(a, b) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(SubBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// mul
// ===========================================================================

/// Backward node for `c = a * b`.
///
/// VJP: da = grad * b, db = grad * a.
#[derive(Debug)]
struct MulBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for MulBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // When grad_output requires_grad (i.e., create_graph=true), use
        // differentiable operations so the backward pass itself is recorded
        // in the computation graph for higher-order gradients.
        if grad_output.requires_grad() || grad_output.grad_fn().is_some() {
            // Higher-order: use differentiable ops so the backward pass
            // itself is recorded in the graph.
            let da = if self.a.requires_grad() {
                let raw = mul(grad_output, &self.b)?;
                Some(reduce_grad_to_shape(&raw, self.a.shape())?)
            } else {
                None
            };

            let db = if self.b.requires_grad() {
                let raw = mul(grad_output, &self.a)?;
                Some(reduce_grad_to_shape(&raw, self.b.shape())?)
            } else {
                None
            };

            return Ok(vec![da, db]);
        }

        // Standard (non-higher-order) path: use no_grad + op functions
        // so it works on both CPU and GPU tensors.
        let da = if self.a.requires_grad() {
            let raw = no_grad(|| mul(grad_output, &self.b))?;
            Some(reduce_grad_to_shape(&raw, self.a.shape())?)
        } else {
            None
        };

        let db = if self.b.requires_grad() {
            let raw = no_grad(|| mul(grad_output, &self.a))?;
            Some(reduce_grad_to_shape(&raw, self.b.shape())?)
        } else {
            None
        };

        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "MulBackward"
    }
}

/// Elementwise multiplication: `c = a * b`.
pub fn mul<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    if let Some(out) = crate::meta_propagate::binary_broadcast(a, b)? {
        return Ok(out);
    }

    crate::profiler_hook::profile_op_scope("mul", "tensor_op", &[a.shape(), b.shape()], || {
        mul_inner(a, b)
    })
}

fn mul_inner<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

        let needs_broadcast = a.shape() != b.shape();
        let (handle, out_shape) = if needs_broadcast {
            let out_shape = broadcast_shapes(a.shape(), b.shape())?;
            let h = if is_f64::<T>() {
                backend.broadcast_mul_f64(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    a.shape(),
                    b.shape(),
                    &out_shape,
                )?
            } else {
                backend.broadcast_mul_f32(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    a.shape(),
                    b.shape(),
                    &out_shape,
                )?
            };
            (h, out_shape)
        } else if is_f64::<T>() {
            (
                backend.mul_f64(a.gpu_handle()?, b.gpu_handle()?)?,
                a.shape().to_vec(),
            )
        } else {
            (
                backend.mul_f32(a.gpu_handle()?, b.gpu_handle()?)?,
                a.shape().to_vec(),
            )
        };
        let storage = TensorStorage::gpu(handle);

        if needs_grad(a, b) {
            Tensor::from_operation(
                storage,
                out_shape,
                Arc::new(MulBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Tensor::from_storage(storage, out_shape, false)
        }
    } else {
        let result = fast_mul(a, b)?;

        if needs_grad(a, b) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(MulBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// div
// ===========================================================================

/// Backward node for `c = a / b`.
///
/// VJP: da = grad / b, db = -grad * a / (b * b).
#[derive(Debug)]
struct DivBackward<T: Float> {
    a: Tensor<T>,
    b: Tensor<T>,
}

impl<T: Float> GradFn<T> for DivBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // Use op-level functions which handle broadcasting correctly.
        // da = grad / b
        let da = if self.a.requires_grad() {
            let raw = no_grad(|| div(grad_output, &self.b))?;
            Some(reduce_grad_to_shape(&raw, self.a.shape())?)
        } else {
            None
        };
        // db = -grad * a / (b * b)
        let db = if self.b.requires_grad() {
            let raw = no_grad(|| {
                let neg_go = neg(grad_output)?;
                let neg_go_a = mul(&neg_go, &self.a)?;
                let b_sq = mul(&self.b, &self.b)?;
                div(&neg_go_a, &b_sq)
            })?;
            Some(reduce_grad_to_shape(&raw, self.b.shape())?)
        } else {
            None
        };

        Ok(vec![da, db])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a, &self.b]
    }

    fn name(&self) -> &'static str {
        "DivBackward"
    }
}

/// Elementwise division: `c = a / b`.
///
/// Division by zero follows IEEE 754 semantics: `x / 0.0` produces `+inf`
/// or `-inf` depending on the sign of `x`, and `0.0 / 0.0` produces `NaN`.
pub fn div<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.device() != b.device() {
        return Err(FerrotorchError::DeviceMismatch {
            expected: a.device(),
            got: b.device(),
        });
    }

    if let Some(out) = crate::meta_propagate::binary_broadcast(a, b)? {
        return Ok(out);
    }

    crate::profiler_hook::profile_op_scope("div", "tensor_op", &[a.shape(), b.shape()], || {
        div_inner(a, b)
    })
}

fn div_inner<T: Float>(a: &Tensor<T>, b: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

        let needs_broadcast = a.shape() != b.shape();
        let (handle, out_shape) = if needs_broadcast {
            let out_shape = broadcast_shapes(a.shape(), b.shape())?;
            let h = if is_f64::<T>() {
                backend.broadcast_div_f64(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    a.shape(),
                    b.shape(),
                    &out_shape,
                )?
            } else {
                backend.broadcast_div_f32(
                    a.gpu_handle()?,
                    b.gpu_handle()?,
                    a.shape(),
                    b.shape(),
                    &out_shape,
                )?
            };
            (h, out_shape)
        } else {
            let h = if is_f32::<T>() {
                backend.div_f32(a.gpu_handle()?, b.gpu_handle()?)?
            } else {
                backend.div_f64(a.gpu_handle()?, b.gpu_handle()?)?
            };
            (h, a.shape().to_vec())
        };
        let storage = TensorStorage::gpu(handle);

        if needs_grad(a, b) {
            Tensor::from_operation(
                storage,
                out_shape,
                Arc::new(DivBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Tensor::from_storage(storage, out_shape, false)
        }
    } else {
        let result = crate::ops::elementwise::fast_div(a, b)?;

        if needs_grad(a, b) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(
                storage,
                shape,
                Arc::new(DivBackward {
                    a: a.clone(),
                    b: b.clone(),
                }),
            )
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// neg
// ===========================================================================

/// Backward node for `c = -a`.
///
/// VJP: da = -grad.
#[derive(Debug)]
struct NegBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for NegBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            Some(no_grad(|| neg(grad_output))?)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "NegBackward"
    }
}

/// Elementwise negation: `c = -a`.
pub fn neg<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::unary_same_shape(a)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("neg", "tensor_op", &[a.shape()], || neg_inner(a))
}

fn neg_inner<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if is_f64::<T>() {
            backend.neg_f64(a.gpu_handle()?)?
        } else {
            backend.neg_f32(a.gpu_handle()?)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = a.shape().to_vec();

        if needs_grad_unary(a) {
            Tensor::from_operation(storage, shape, Arc::new(NegBackward { a: a.clone() }))
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = unary_map(a, |x| -x)?;

        if needs_grad_unary(a) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, Arc::new(NegBackward { a: a.clone() }))
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// pow (tensor ^ scalar exponent)
// ===========================================================================

/// Backward node for `c = a ^ exp` where `exp` is a scalar.
///
/// VJP: da = exp * a^(exp-1) * grad.
#[derive(Debug)]
struct PowBackward<T: Float> {
    a: Tensor<T>,
    exp: f64,
}

impl<T: Float> GradFn<T> for PowBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            // When grad_output requires_grad (create_graph=true), use
            // differentiable operations so the backward pass itself is
            // tracked in the computation graph for higher-order gradients.
            if grad_output.requires_grad() || grad_output.grad_fn().is_some() {
                // da = grad_output * exp * a^(exp-1)
                // Using differentiable pow and mul.
                let a_pow = pow(&self.a, self.exp - 1.0)?; // a^(exp-1)
                let exp_t = T::from(self.exp).unwrap();
                let exp_tensor = Tensor::from_storage(
                    TensorStorage::cpu(vec![exp_t; self.a.numel().max(1)]),
                    self.a.shape().to_vec(),
                    false,
                )?;
                let scaled = mul(&exp_tensor, &a_pow)?; // exp * a^(exp-1)
                Some(mul(grad_output, &scaled)?) // grad_output * exp * a^(exp-1)
            } else if grad_output.is_cuda() {
                // GPU path: use op-level functions in no_grad.
                // da = grad_output * exp * a^(exp-1)
                let da = no_grad(|| {
                    let a_pow = pow(&self.a, self.exp - 1.0)?;
                    let exp_t = T::from(self.exp).unwrap();
                    let exp_tensor = Tensor::from_storage(
                        TensorStorage::cpu(vec![exp_t; self.a.numel().max(1)]),
                        self.a.shape().to_vec(),
                        false,
                    )?;
                    let exp_gpu = exp_tensor.to(self.a.device())?;
                    let scaled = mul(&exp_gpu, &a_pow)?;
                    mul(grad_output, &scaled)
                })?;
                Some(da)
            } else {
                // CPU path: direct data access for performance.
                let go_data = grad_output.data()?;
                let a_data = self.a.data()?;
                let exp_t = T::from(self.exp).unwrap();
                let exp_m1 = T::from(self.exp - 1.0).unwrap();
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(a_data.iter())
                    .map(|(&g, &a)| g * exp_t * a.powf(exp_m1))
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.a.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "PowBackward"
    }
}

/// Elementwise power: `c = a ^ exp` where `exp` is a scalar `f64`.
pub fn pow<T: Float>(a: &Tensor<T>, exp: f64) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::unary_same_shape(a)? {
        let _ = exp;
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("pow", "tensor_op", &[a.shape()], || pow_inner(a, exp))
}

fn pow_inner<T: Float>(a: &Tensor<T>, exp: f64) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if is_f32::<T>() {
            backend.pow_f32(a.gpu_handle()?, exp as f32)?
        } else {
            backend.pow_f64(a.gpu_handle()?, exp)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = a.shape().to_vec();

        if needs_grad_unary(a) {
            Tensor::from_operation(storage, shape, Arc::new(PowBackward { a: a.clone(), exp }))
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let exp_t = T::from(exp).unwrap();
        let result = scalar_map(a, exp_t, |x, e| x.powf(e))?;

        if needs_grad_unary(a) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, Arc::new(PowBackward { a: a.clone(), exp }))
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// sqrt
// ===========================================================================

/// Backward node for `c = sqrt(a)`.
///
/// VJP: da = grad / (2 * sqrt(a)).
#[derive(Debug)]
struct SqrtBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for SqrtBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let da = if self.a.requires_grad() {
            if grad_output.is_cuda() {
                // GPU path: da = grad / (2 * sqrt(a))
                let da = no_grad(|| {
                    let sqrt_a = sqrt(&self.a)?;
                    let two_t = T::from(2.0).unwrap();
                    let two_tensor = Tensor::from_storage(
                        TensorStorage::cpu(vec![two_t; self.a.numel().max(1)]),
                        self.a.shape().to_vec(),
                        false,
                    )?;
                    let two_gpu = two_tensor.to(self.a.device())?;
                    let denom = mul(&two_gpu, &sqrt_a)?;
                    div(grad_output, &denom)
                })?;
                Some(da)
            } else {
                // CPU path: direct data access for performance.
                let go_data = grad_output.data()?;
                let a_data = self.a.data()?;
                let two = T::from(2.0).unwrap();
                let grad_a: Vec<T> = go_data
                    .iter()
                    .zip(a_data.iter())
                    .map(|(&g, &a)| g / (two * a.sqrt()))
                    .collect();
                Some(Tensor::from_storage(
                    TensorStorage::cpu(grad_a),
                    self.a.shape().to_vec(),
                    false,
                )?)
            }
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "SqrtBackward"
    }
}

/// Elementwise square root: `c = sqrt(a)`.
pub fn sqrt<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::unary_same_shape(a)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("sqrt", "tensor_op", &[a.shape()], || sqrt_inner(a))
}

fn sqrt_inner<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if is_f32::<T>() {
            backend.sqrt_f32(a.gpu_handle()?)?
        } else {
            backend.sqrt_f64(a.gpu_handle()?)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = a.shape().to_vec();

        if needs_grad_unary(a) {
            Tensor::from_operation(storage, shape, Arc::new(SqrtBackward { a: a.clone() }))
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = unary_map(a, |x| x.sqrt())?;

        if needs_grad_unary(a) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, Arc::new(SqrtBackward { a: a.clone() }))
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// abs
// ===========================================================================

/// Backward node for `c = |a|`.
///
/// VJP: da = grad * sign(a).
#[derive(Debug)]
struct AbsBackward<T: Float> {
    a: Tensor<T>,
}

impl<T: Float> GradFn<T> for AbsBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        use crate::gpu_dispatch::gpu_backend;

        let da = if self.a.requires_grad() {
            // GPU-native path for f32/f64 when both tensors live on CUDA.
            if grad_output.is_cuda() && self.a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
                let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
                let handle = if is_f32::<T>() {
                    backend.abs_backward_f32(grad_output.gpu_handle()?, self.a.gpu_handle()?)?
                } else {
                    backend.abs_backward_f64(grad_output.gpu_handle()?, self.a.gpu_handle()?)?
                };
                let grad_a = Tensor::from_storage(
                    TensorStorage::gpu(handle),
                    self.a.shape().to_vec(),
                    false,
                )?;
                return Ok(vec![Some(grad_a)]);
            }

            if grad_output.is_cuda() || self.a.is_cuda() {
                return Err(FerrotorchError::NotImplementedOnCuda { op: "AbsBackward" });
            }
            // CPU path: direct data access for performance.
            let go_data = grad_output.data()?;
            let a_data = self.a.data()?;
            let zero = <T as num_traits::Zero>::zero();
            let one = <T as num_traits::One>::one();
            let grad_a: Vec<T> = go_data
                .iter()
                .zip(a_data.iter())
                .map(|(&g, &a)| {
                    let sign = if a > zero {
                        one
                    } else if a < zero {
                        -one
                    } else {
                        zero
                    };
                    g * sign
                })
                .collect();
            Some(Tensor::from_storage(
                TensorStorage::cpu(grad_a),
                self.a.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };
        Ok(vec![da])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.a]
    }

    fn name(&self) -> &'static str {
        "AbsBackward"
    }
}

/// Elementwise absolute value: `c = |a|`.
pub fn abs<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if let Some(out) = crate::meta_propagate::unary_same_shape(a)? {
        return Ok(out);
    }
    crate::profiler_hook::profile_op_scope("abs", "tensor_op", &[a.shape()], || abs_inner(a))
}

fn abs_inner<T: Float>(a: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    if a.is_cuda() && (is_f32::<T>() || is_f64::<T>()) {
        let backend =
            crate::gpu_dispatch::gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;
        let handle = if is_f32::<T>() {
            backend.abs_f32(a.gpu_handle()?)?
        } else {
            backend.abs_f64(a.gpu_handle()?)?
        };
        let storage = TensorStorage::gpu(handle);
        let shape = a.shape().to_vec();

        if needs_grad_unary(a) {
            Tensor::from_operation(storage, shape, Arc::new(AbsBackward { a: a.clone() }))
        } else {
            Tensor::from_storage(storage, shape, false)
        }
    } else {
        let result = unary_map(a, |x| x.abs())?;

        if needs_grad_unary(a) {
            let (storage, shape) = result.into_storage_and_shape()?;
            Tensor::from_operation(storage, shape, Arc::new(AbsBackward { a: a.clone() }))
        } else {
            Ok(result)
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a leaf scalar tensor.
    fn leaf_scalar(val: f32, requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(TensorStorage::cpu(vec![val]), vec![], requires_grad).unwrap()
    }

    /// Create a leaf 1-D tensor.
    fn leaf_vec(data: &[f32], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            vec![data.len()],
            requires_grad,
        )
        .unwrap()
    }

    /// Assert a scalar tensor is approximately equal to `expected`.
    fn assert_scalar_approx(t: &Tensor<f32>, expected: f32, tol: f32) {
        let val = t.item().unwrap();
        assert!(
            (val - expected).abs() < tol,
            "expected {expected}, got {val}"
        );
    }

    // -----------------------------------------------------------------------
    // Forward tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_forward() {
        let a = leaf_vec(&[1.0, 2.0, 3.0], false);
        let b = leaf_vec(&[4.0, 5.0, 6.0], false);
        let c = add(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_sub_forward() {
        let a = leaf_vec(&[10.0, 20.0, 30.0], false);
        let b = leaf_vec(&[1.0, 2.0, 3.0], false);
        let c = sub(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[9.0, 18.0, 27.0]);
    }

    #[test]
    fn test_mul_forward() {
        let a = leaf_vec(&[2.0, 3.0, 4.0], false);
        let b = leaf_vec(&[5.0, 6.0, 7.0], false);
        let c = mul(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_div_forward() {
        let a = leaf_vec(&[10.0, 20.0, 30.0], false);
        let b = leaf_vec(&[2.0, 5.0, 10.0], false);
        let c = div(&a, &b).unwrap();
        assert_eq!(c.data().unwrap(), &[5.0, 4.0, 3.0]);
    }

    #[test]
    fn test_neg_forward() {
        let a = leaf_vec(&[1.0, -2.0, 3.0], false);
        let c = neg(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[-1.0, 2.0, -3.0]);
    }

    #[test]
    fn test_pow_forward() {
        let a = leaf_vec(&[2.0, 3.0, 4.0], false);
        let c = pow(&a, 2.0).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 4.0).abs() < 1e-6);
        assert!((d[1] - 9.0).abs() < 1e-6);
        assert!((d[2] - 16.0).abs() < 1e-6);
    }

    #[test]
    fn test_sqrt_forward() {
        let a = leaf_vec(&[4.0, 9.0, 16.0], false);
        let c = sqrt(&a).unwrap();
        let d = c.data().unwrap();
        assert!((d[0] - 2.0).abs() < 1e-6);
        assert!((d[1] - 3.0).abs() < 1e-6);
        assert!((d[2] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_abs_forward() {
        let a = leaf_vec(&[-3.0, 0.0, 5.0], false);
        let c = abs(&a).unwrap();
        assert_eq!(c.data().unwrap(), &[3.0, 0.0, 5.0]);
    }

    // -----------------------------------------------------------------------
    // Backward tests (scalar tensors for simplicity)
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_backward() {
        // c = a + b; dc/da = 1, dc/db = 1.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = add(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_sub_backward() {
        // c = a - b; dc/da = 1, dc/db = -1.
        let a = leaf_scalar(5.0, true);
        let b = leaf_scalar(3.0, true);
        let c = sub(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_mul_backward() {
        // c = a * b; dc/da = b = 3, dc/db = a = 2.
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = mul(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 3.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), 2.0, 1e-6);
    }

    #[test]
    fn test_div_backward() {
        // c = a / b; dc/da = 1/b = 1/4, dc/db = -a/b^2 = -6/16 = -0.375.
        let a = leaf_scalar(6.0, true);
        let b = leaf_scalar(4.0, true);
        let c = div(&a, &b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.25, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), -0.375, 1e-6);
    }

    #[test]
    fn test_div_backward_tensor_by_scalar() {
        // Reproducer from GitHub issue #7:
        // x = [1, 2, 3, 4] (shape [2,2]), s = 2.0 (scalar)
        // y = x / s = [0.5, 1.0, 1.5, 2.0]
        // loss = sum(y) = 5.0
        // d_loss/d_x = 1/s = 0.5 for all elements
        let x = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0f64, 2.0, 3.0, 4.0]),
            vec![2, 2],
            true,
        )
        .unwrap();
        let s = Tensor::from_storage(TensorStorage::cpu(vec![2.0f64]), vec![], false).unwrap();
        let y = div(&x, &s).unwrap();
        let loss = crate::grad_fns::reduction::sum(&y).unwrap();
        loss.backward().unwrap();

        let grad = x.grad().unwrap().expect("x should have grad");
        assert_eq!(grad.shape(), &[2, 2]);
        let g = grad.data().unwrap();
        for (i, &v) in g.iter().enumerate() {
            assert!((v - 0.5).abs() < 1e-10, "grad[{i}] = {v}, expected 0.5");
        }
    }

    #[test]
    fn test_neg_backward() {
        // c = -a; dc/da = -1.
        let a = leaf_scalar(7.0, true);
        let c = neg(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -1.0, 1e-6);
    }

    #[test]
    fn test_pow_backward() {
        // c = a^3; dc/da = 3 * a^2 = 3 * 4 = 12.
        let a = leaf_scalar(2.0, true);
        let c = pow(&a, 3.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 12.0, 1e-5);
    }

    #[test]
    fn test_sqrt_backward() {
        // c = sqrt(a); dc/da = 1 / (2 * sqrt(a)).
        // a = 4.0 => dc/da = 1 / (2 * 2) = 0.25.
        let a = leaf_scalar(4.0, true);
        let c = sqrt(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 0.25, 1e-6);
    }

    #[test]
    fn test_abs_backward_positive() {
        // c = |a| where a > 0; dc/da = sign(a) = 1.
        let a = leaf_scalar(3.0, true);
        let c = abs(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    #[test]
    fn test_abs_backward_negative() {
        // c = |a| where a < 0; dc/da = sign(a) = -1.
        let a = leaf_scalar(-3.0, true);
        let c = abs(&a).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -1.0, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Tests for no-grad and partial requires_grad
    // -----------------------------------------------------------------------

    #[test]
    fn test_add_no_grad_fn_when_inputs_detached() {
        let a = leaf_scalar(2.0, false);
        let b = leaf_scalar(3.0, false);
        let c = add(&a, &b).unwrap();
        assert!(c.grad_fn().is_none());
    }

    #[test]
    fn test_mul_partial_requires_grad() {
        // a requires grad, b does not.
        // c = a * b; dc/da = b = 5, dc/db = None.
        let a = leaf_scalar(3.0, true);
        let b = leaf_scalar(5.0, false);
        let c = mul(&a, &b).unwrap();
        assert!(c.grad_fn().is_some());
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 5.0, 1e-6);
        assert!(b.grad().unwrap().is_none());
    }

    #[test]
    fn test_no_grad_context_skips_backward() {
        use crate::autograd::no_grad::no_grad;

        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = no_grad(|| add(&a, &b)).unwrap();
        // Inside no_grad, no grad_fn should be attached.
        assert!(c.grad_fn().is_none());
    }

    // -----------------------------------------------------------------------
    // Chain rule tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_chain_mul_add() {
        // d = a * b + b
        // dd/da = b = 3
        // dd/db = a + 1 = 3
        let a = leaf_scalar(2.0, true);
        let b = leaf_scalar(3.0, true);
        let c = mul(&a, &b).unwrap();
        let d = add(&c, &b).unwrap();
        d.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 3.0, 1e-6);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), 3.0, 1e-6);
    }

    #[test]
    fn test_chain_div_sub() {
        // c = a / b - a
        // dc/da = 1/b - 1 = 1/2 - 1 = -0.5
        // dc/db = -a/b^2 = -3/4 = -0.75
        let a = leaf_scalar(3.0, true);
        let b = leaf_scalar(2.0, true);
        let d = div(&a, &b).unwrap();
        let e = sub(&d, &a).unwrap();
        e.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), -0.5, 1e-5);
        assert_scalar_approx(&b.grad().unwrap().unwrap(), -0.75, 1e-5);
    }

    #[test]
    fn test_chain_sqrt_pow() {
        // c = sqrt(a)^2 = a. dc/da = 1.
        // sqrt(9) = 3, pow(3, 2) = 9.
        // d(pow)/d(sqrt) = 2 * sqrt(a) = 6.
        // d(sqrt)/da = 1 / (2*sqrt(a)) = 1/6.
        // dc/da = 6 * 1/6 = 1.
        let a = leaf_scalar(9.0, true);
        let s = sqrt(&a).unwrap();
        let c = pow(&s, 2.0).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-5);
    }

    #[test]
    fn test_neg_double() {
        // c = -(-a) = a; dc/da = 1.
        let a = leaf_scalar(5.0, true);
        let b = neg(&a).unwrap();
        let c = neg(&b).unwrap();
        c.backward().unwrap();

        assert_scalar_approx(&a.grad().unwrap().unwrap(), 1.0, 1e-6);
    }

    // -----------------------------------------------------------------------
    // Vector backward tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_mul_vector_backward() {
        // c = a * b (elementwise), then sum to scalar for backward.
        // loss = sum(a * b)
        // d(loss)/d(a_i) = b_i, d(loss)/d(b_i) = a_i.
        let a = leaf_vec(&[1.0, 2.0, 3.0], true);
        let b = leaf_vec(&[4.0, 5.0, 6.0], true);
        let c = mul(&a, &b).unwrap();

        // Sum to scalar so we can call backward.
        let c_data = c.data().unwrap().to_vec();
        let total: f32 = c_data.iter().sum();
        let sum_backward = SumBackward { input: c.clone() };
        let loss = Tensor::from_operation(
            TensorStorage::cpu(vec![total]),
            vec![],
            Arc::new(sum_backward),
        )
        .unwrap();
        loss.backward().unwrap();

        let a_grad = a.grad().unwrap().unwrap();
        let a_g = a_grad.data().unwrap();
        assert!((a_g[0] - 4.0).abs() < 1e-6);
        assert!((a_g[1] - 5.0).abs() < 1e-6);
        assert!((a_g[2] - 6.0).abs() < 1e-6);

        let b_grad = b.grad().unwrap().unwrap();
        let b_g = b_grad.data().unwrap();
        assert!((b_g[0] - 1.0).abs() < 1e-6);
        assert!((b_g[1] - 2.0).abs() < 1e-6);
        assert!((b_g[2] - 3.0).abs() < 1e-6);
    }

    /// Helper backward node for sum reduction in tests:
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
