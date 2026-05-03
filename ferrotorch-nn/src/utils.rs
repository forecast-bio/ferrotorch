//! Gradient clipping utilities.
//!
//! These mirror PyTorch's `torch.nn.utils.clip_grad_norm_` and
//! `torch.nn.utils.clip_grad_value_`, operating in-place on the
//! gradients of a parameter slice.
//!
//! ## Device dispatch policy
//!
//! When all gradients are on **`Device::Cpu`**, the CPU path runs entirely on
//! the host (unchanged from the original implementation).
//!
//! When all gradients are on **`Device::Cuda(n)`** and `T` is `f32` or `f64`:
//! - Per-tensor squared-sum reductions run via `gpu_reduce_sum` /
//!   `gpu_reduce_sum_f64` (real GPU kernel launches — no host readback of the
//!   full gradient tensor).
//! - After the per-tensor reduction, a **single f32/f64 scalar** (one
//!   `GpuBufferHandle` with 1 element) is transferred to the host so that
//!   the cross-tensor square-root and clip-coefficient division can be
//!   computed in one host instruction. This is the legitimate per-tensor
//!   scalar boundary: N scalars, not N full gradient tensors. PyTorch's
//!   own `clip_grad_norm_` implementation does the same (accumulate
//!   per-parameter norm on host, final `sqrt` on host).
//! - The optional in-place scaling uses `backend.scale_f32` /
//!   `backend.scale_f64` (real GPU kernel launch via `gpu_scale`).
//!
//! When gradients are on **different devices** (mixed CPU + CUDA, or CUDA
//! ordinal mismatch) the function returns
//! `Err(FerrotorchError::DeviceMismatch)` — matching PyTorch, which errors on
//! mixed-device gradient sets.
//!
//! For `Device::Mps`, `Device::Xpu`, or other unrecognised backends, the
//! function returns `Err(FerrotorchError::DeviceUnavailable)`.

use std::any::TypeId;

use ferrotorch_core::gpu_dispatch::gpu_backend;
use ferrotorch_core::{Device, FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::parameter::Parameter;

// ---------------------------------------------------------------------------
// Dtype helpers (match the pattern in ferrotorch-nn/src/embedding.rs)
// ---------------------------------------------------------------------------

#[inline]
fn is_f32<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f32>()
}

#[inline]
fn is_f64<T: Float>() -> bool {
    TypeId::of::<T>() == TypeId::of::<f64>()
}

// ---------------------------------------------------------------------------
// Scalar readback from a 1-element GpuBufferHandle
// ---------------------------------------------------------------------------

/// Read a single `f32` from a 1-element `GpuBufferHandle`.
///
/// The handle must contain exactly 4 bytes (one `f32`). This is the
/// per-tensor scalar boundary: after `sum_f32` on a gradient tensor we
/// get one f32 on device; this transfers only that scalar to the host.
fn readback_scalar_f32(
    handle: &ferrotorch_core::gpu_dispatch::GpuBufferHandle,
    backend: &dyn ferrotorch_core::gpu_dispatch::GpuBackend,
) -> FerrotorchResult<f32> {
    let bytes = backend.gpu_to_cpu(handle)?;
    if bytes.len() < 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: "readback_scalar_f32: buffer is empty".into(),
        });
    }
    let val = f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    Ok(val)
}

/// Read a single `f64` from a 1-element `GpuBufferHandle`.
///
/// The handle must contain exactly 8 bytes (one `f64`). Same boundary
/// rationale as `readback_scalar_f32`.
fn readback_scalar_f64(
    handle: &ferrotorch_core::gpu_dispatch::GpuBufferHandle,
    backend: &dyn ferrotorch_core::gpu_dispatch::GpuBackend,
) -> FerrotorchResult<f64> {
    let bytes = backend.gpu_to_cpu(handle)?;
    if bytes.len() < 8 {
        return Err(FerrotorchError::InvalidArgument {
            message: "readback_scalar_f64: buffer is empty".into(),
        });
    }
    let val = f64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]);
    Ok(val)
}

// ---------------------------------------------------------------------------
// clip_grad_norm_
// ---------------------------------------------------------------------------

/// Clip the total gradient norm of an iterable of parameters.
///
/// Computes the total norm of all parameter gradients (treating them as
/// a single concatenated vector), and if the total norm exceeds
/// `max_norm`, rescales every gradient by `max_norm / total_norm`.
///
/// # Arguments
///
/// * `params` — Slice of parameter references whose gradients will be clipped.
/// * `max_norm` — Maximum allowed norm.
/// * `norm_type` — Type of the norm (e.g. `2.0` for L2 norm).
///
/// # Returns
///
/// The total gradient norm **before** any clipping is applied.
///
/// Parameters whose gradient is `None` are silently skipped.
///
/// # Device dispatch
///
/// - All-CPU grads: runs on the host (unchanged).
/// - All-CUDA f32/f64 grads with `norm_type == 2.0`: per-tensor
///   `sum_of_squares` reduction runs on GPU via `gpu_reduce_sum`;
///   N per-tensor scalars are read to host for the final `sqrt`; the
///   optional `scale` runs on GPU via `gpu_scale`.
/// - All-CUDA grads with `norm_type != 2.0`: returns
///   `Err(FerrotorchError::NotImplementedOnCuda)` — PyTorch also restricts
///   the GPU kernel to L2; other norms must use `.cpu()` explicitly.
/// - Mixed-device grads: returns `Err(FerrotorchError::DeviceMismatch)`.
/// - Unsupported single-device backends (MPS, XPU, …): returns
///   `Err(FerrotorchError::DeviceUnavailable)`.
pub fn clip_grad_norm_<T: Float>(
    params: &[&Parameter<T>],
    max_norm: f64,
    norm_type: f64,
) -> FerrotorchResult<f64> {
    // Collect all gradient tensors (skip None).
    let grads: Vec<Tensor<T>> = params
        .iter()
        .filter_map(|p| p.grad().ok().flatten())
        .collect();

    if grads.is_empty() {
        return Ok(0.0);
    }

    // Determine the common device (error on mismatch).
    let common_device = grads[0].device();
    for g in &grads[1..] {
        if g.device() != common_device {
            return Err(FerrotorchError::DeviceMismatch {
                expected: common_device,
                got: g.device(),
            });
        }
    }

    match common_device {
        Device::Cpu => clip_grad_norm_cpu(params, &grads, max_norm, norm_type),
        Device::Cuda(_ordinal) => {
            if !is_f32::<T>() && !is_f64::<T>() {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "clip_grad_norm_ on CUDA requires f32 or f64 gradients",
                });
            }
            if norm_type != 2.0 {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "clip_grad_norm_ on CUDA only supports norm_type == 2.0; \
                         move gradients to CPU for other norms",
                });
            }
            clip_grad_norm_cuda(params, &grads, max_norm)
        }
        _ => Err(FerrotorchError::DeviceUnavailable),
    }
}

// ---------------------------------------------------------------------------
// clip_grad_norm_ — CPU path (unchanged logic from original)
// ---------------------------------------------------------------------------

fn clip_grad_norm_cpu<T: Float>(
    params: &[&Parameter<T>],
    grads: &[Tensor<T>],
    max_norm: f64,
    norm_type: f64,
) -> FerrotorchResult<f64> {
    let total_norm: f64 = if norm_type == f64::INFINITY {
        let mut max_val: f64 = 0.0;
        for g in grads {
            let data = g.data_vec()?;
            for v in &data {
                let abs_v = v.to_f64().unwrap().abs();
                if abs_v > max_val {
                    max_val = abs_v;
                }
            }
        }
        max_val
    } else {
        let mut accum: f64 = 0.0;
        for g in grads {
            let data = g.data_vec()?;
            for v in &data {
                accum += v.to_f64().unwrap().abs().powf(norm_type);
            }
        }
        accum.powf(1.0 / norm_type)
    };

    if total_norm > max_norm {
        let clip_coef = max_norm / total_norm;
        let clip_t = T::from(clip_coef).unwrap();

        for param in params {
            if let Some(g) = param.grad()? {
                let data = g.data_vec()?;
                let scaled: Vec<T> = data.iter().map(|&v| v * clip_t).collect();
                let new_grad =
                    Tensor::from_storage(TensorStorage::cpu(scaled), g.shape().to_vec(), false)?;
                param.set_grad(Some(new_grad))?;
            }
        }
    }

    Ok(total_norm)
}

// ---------------------------------------------------------------------------
// clip_grad_norm_ — CUDA path (GPU kernel launches, no full-tensor D2H)
// ---------------------------------------------------------------------------

/// CUDA path for L2 `clip_grad_norm_` on f32 and f64 gradients.
///
/// GPU compute performed:
/// - Per-tensor: `sum_of_squares = gpu_reduce_sum(grad * grad)` (two kernel
///   launches per tensor: `mul_f32/f64` + `reduce_sum`). Actually we use the
///   backend's `sum_f32/sum_f64` which calls `gpu_reduce_sum` directly on the
///   buffer, but the squaring step needs a separate `mul`. We compose as:
///   `sq_handle = backend.mul_f32(g_handle, g_handle)`, then
///   `sum_handle = backend.sum_f32(sq_handle, numel)`.
/// - The 1-element `sum_handle` is transferred to host (4 or 8 bytes per
///   tensor — the allowed per-tensor scalar boundary).
/// - `total_norm = sqrt(sum_of_all_per_tensor_sum_of_squares)` — one `f64`
///   sqrt on host.
/// - If clipping needed: per-tensor `backend.scale_f32(g_handle, coef)` (one
///   kernel launch per tensor). New GPU handle replaces the gradient.
fn clip_grad_norm_cuda<T: Float>(
    params: &[&Parameter<T>],
    grads: &[Tensor<T>],
    max_norm: f64,
) -> FerrotorchResult<f64> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

    // Phase 1: compute per-tensor sum-of-squares on GPU; read back one scalar.
    // Per-tensor boundary: N floats transferred, not N tensors.
    let mut total_sq: f64 = 0.0;
    for g in grads {
        let g_handle = g.gpu_handle()?;
        let numel = g.numel();

        let per_tensor_sq: f64 = if is_f32::<T>() {
            // sq = g * g on GPU
            let sq_handle = backend.mul_f32(g_handle, g_handle)?;
            // sum(sq) on GPU — returns a 1-element buffer
            let sum_handle = backend.sum_f32(&sq_handle, numel)?;
            // Transfer the single f32 scalar to host (4 bytes — the legitimate boundary)
            // BOUNDARY: per-tensor scalar readback; documented at module top.
            let scalar = readback_scalar_f32(&sum_handle, backend)?;
            scalar as f64
        } else {
            // f64 path
            let sq_handle = backend.mul_f64(g_handle, g_handle)?;
            let sum_handle = backend.sum_f64(&sq_handle, numel)?;
            // Transfer the single f64 scalar to host (8 bytes — the legitimate boundary)
            // BOUNDARY: per-tensor scalar readback; documented at module top.
            readback_scalar_f64(&sum_handle, backend)?
        };

        total_sq += per_tensor_sq;
    }

    // Single sqrt on host (one floating-point instruction, no GPU needed).
    let total_norm = total_sq.sqrt();

    // Phase 2: if clipping needed, scale each gradient in-place on GPU.
    if total_norm > max_norm {
        let clip_coef = max_norm / total_norm;

        for param in params {
            if let Some(g) = param.grad()? {
                let g_handle = g.gpu_handle()?;
                let shape = g.shape().to_vec();
                let ordinal = match g.device() {
                    Device::Cuda(o) => o,
                    _ => unreachable!(),
                };

                // Scale on GPU — real kernel launch via gpu_scale / gpu_scale_f64.
                let scaled_handle = if is_f32::<T>() {
                    // SAFETY for cast: clip_coef is a finite positive f64; the
                    // truncation to f32 stays in-range because max_norm ≤ total_norm ≤ f64::MAX.
                    #[allow(clippy::cast_possible_truncation)]
                    backend.scale_f32(g_handle, clip_coef as f32)?
                } else {
                    backend.scale_f64(g_handle, clip_coef)?
                };

                // Reconstruct a Tensor from the scaled GPU handle.
                let new_storage = TensorStorage::gpu(scaled_handle);
                // Suppress the "ordinal unused" warning: the handle already
                // encodes the device; TensorStorage::gpu reads it internally.
                let _ = ordinal;
                let new_grad = Tensor::from_storage(new_storage, shape, false)?;
                param.set_grad(Some(new_grad))?;
            }
        }
    }

    Ok(total_norm)
}

// ---------------------------------------------------------------------------
// clip_grad_value_
// ---------------------------------------------------------------------------

/// Clamp all gradient values to the range `[-clip_value, clip_value]`.
///
/// Each element of every parameter gradient is clamped independently.
/// Parameters whose gradient is `None` are silently skipped.
///
/// # Arguments
///
/// * `params` — Slice of parameter references whose gradients will be clamped.
/// * `clip_value` — Maximum absolute value for gradient elements.
///
/// # Device dispatch
///
/// - All-CPU grads: runs on the host (unchanged).
/// - All-CUDA f32/f64 grads: `backend.clamp_f32` / `backend.clamp_f64`
///   (real GPU kernel launch via `gpu_clamp` / `gpu_clamp_f64`).
/// - Mixed-device grads: returns `Err(FerrotorchError::DeviceMismatch)`.
/// - Unsupported single-device backends: `Err(FerrotorchError::DeviceUnavailable)`.
pub fn clip_grad_value_<T: Float>(
    params: &[&Parameter<T>],
    clip_value: f64,
) -> FerrotorchResult<()> {
    // Collect grads that exist (skip None) to determine common device.
    let grads: Vec<Tensor<T>> = params
        .iter()
        .filter_map(|p| p.grad().ok().flatten())
        .collect();

    if grads.is_empty() {
        return Ok(());
    }

    let common_device = grads[0].device();
    for g in &grads[1..] {
        if g.device() != common_device {
            return Err(FerrotorchError::DeviceMismatch {
                expected: common_device,
                got: g.device(),
            });
        }
    }

    match common_device {
        Device::Cpu => clip_grad_value_cpu(params, clip_value),
        Device::Cuda(_) => {
            if !is_f32::<T>() && !is_f64::<T>() {
                return Err(FerrotorchError::NotImplementedOnCuda {
                    op: "clip_grad_value_ on CUDA requires f32 or f64 gradients",
                });
            }
            clip_grad_value_cuda(params, clip_value)
        }
        _ => Err(FerrotorchError::DeviceUnavailable),
    }
}

// ---------------------------------------------------------------------------
// clip_grad_value_ — CPU path (unchanged logic from original)
// ---------------------------------------------------------------------------

fn clip_grad_value_cpu<T: Float>(
    params: &[&Parameter<T>],
    clip_value: f64,
) -> FerrotorchResult<()> {
    let lo = T::from(-clip_value).unwrap();
    let hi = T::from(clip_value).unwrap();

    for param in params {
        if let Some(g) = param.grad()? {
            let data = g.data_vec()?;
            let clamped: Vec<T> = data
                .iter()
                .map(|&v| {
                    if v < lo {
                        lo
                    } else if v > hi {
                        hi
                    } else {
                        v
                    }
                })
                .collect();
            let new_grad =
                Tensor::from_storage(TensorStorage::cpu(clamped), g.shape().to_vec(), false)?;
            param.set_grad(Some(new_grad))?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// clip_grad_value_ — CUDA path (GPU clamp kernel)
// ---------------------------------------------------------------------------

/// CUDA path for `clip_grad_value_`.
///
/// For each gradient: `backend.clamp_f32(g_handle, -clip_value, clip_value)`
/// (real GPU kernel launch via `gpu_clamp` PTX kernel). The clamped result
/// stays on device; only the new `GpuBufferHandle` is materialized.
fn clip_grad_value_cuda<T: Float>(
    params: &[&Parameter<T>],
    clip_value: f64,
) -> FerrotorchResult<()> {
    let backend = gpu_backend().ok_or(FerrotorchError::DeviceUnavailable)?;

    for param in params {
        if let Some(g) = param.grad()? {
            let g_handle = g.gpu_handle()?;
            let shape = g.shape().to_vec();

            // Clamp on GPU — real kernel launch via gpu_clamp / gpu_clamp_f64.
            let clamped_handle = if is_f32::<T>() {
                // SAFETY for cast: clip_value is a finite non-negative f64 with values
                // expected in the neural-net regime (1e-6..1e3); truncation to f32 is safe.
                #[allow(clippy::cast_possible_truncation)]
                backend.clamp_f32(g_handle, -(clip_value as f32), clip_value as f32)?
            } else {
                backend.clamp_f64(g_handle, -clip_value, clip_value)?
            };

            let new_storage = TensorStorage::gpu(clamped_handle);
            let new_grad = Tensor::from_storage(new_storage, shape, false)?;
            param.set_grad(Some(new_grad))?;
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a parameter and manually set its gradient.
    fn param_with_grad(shape: &[usize], grad_data: &[f32]) -> Parameter<f32> {
        let p = Parameter::<f32>::zeros(shape).unwrap();
        let grad = Tensor::from_storage(
            TensorStorage::cpu(grad_data.to_vec()),
            shape.to_vec(),
            false,
        )
        .unwrap();
        p.set_grad(Some(grad)).unwrap();
        p
    }

    // -----------------------------------------------------------------------
    // clip_grad_norm_ — CPU path
    // -----------------------------------------------------------------------

    #[test]
    fn test_clip_grad_norm_reduces_norm() {
        // Gradient = [3.0, 4.0] => L2 norm = 5.0
        let p = param_with_grad(&[2], &[3.0, 4.0]);
        let total = clip_grad_norm_(&[&p], 2.5, 2.0).unwrap();

        // Total norm before clipping should be 5.0.
        assert!((total - 5.0).abs() < 1e-6);

        // After clipping, the new norm should be ~2.5.
        let g = p.grad().unwrap().unwrap();
        let d = g.data().unwrap();
        let new_norm = (d[0] as f64 * d[0] as f64 + d[1] as f64 * d[1] as f64).sqrt();
        assert!((new_norm - 2.5).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_no_clip_when_below() {
        // Gradient = [1.0, 0.0] => L2 norm = 1.0
        let p = param_with_grad(&[2], &[1.0, 0.0]);
        let total = clip_grad_norm_(&[&p], 10.0, 2.0).unwrap();

        assert!((total - 1.0).abs() < 1e-6);

        // Gradient should be unchanged.
        let g = p.grad().unwrap().unwrap();
        let d = g.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-6);
        assert!((d[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_multiple_params() {
        // p1 grad = [3.0], p2 grad = [4.0] => joint L2 norm = 5.0
        let p1 = param_with_grad(&[1], &[3.0]);
        let p2 = param_with_grad(&[1], &[4.0]);
        let total = clip_grad_norm_(&[&p1, &p2], 2.5, 2.0).unwrap();

        assert!((total - 5.0).abs() < 1e-6);

        let g1 = p1.grad().unwrap().unwrap().data().unwrap()[0] as f64;
        let g2 = p2.grad().unwrap().unwrap().data().unwrap()[0] as f64;
        let new_norm = (g1 * g1 + g2 * g2).sqrt();
        assert!((new_norm - 2.5).abs() < 1e-4);
    }

    #[test]
    fn test_clip_grad_norm_returns_total_norm() {
        let p = param_with_grad(&[3], &[1.0, 2.0, 2.0]);
        // L2 norm = sqrt(1 + 4 + 4) = 3.0
        let total = clip_grad_norm_(&[&p], 100.0, 2.0).unwrap();
        assert!((total - 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_norm_skips_none_grads() {
        let p_with = param_with_grad(&[2], &[3.0, 4.0]);
        let p_without = Parameter::<f32>::zeros(&[2]).unwrap();
        // No grad set on p_without — should be silently skipped.
        let total = clip_grad_norm_(&[&p_with, &p_without], 2.5, 2.0).unwrap();
        assert!((total - 5.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // clip_grad_value_ — CPU path
    // -----------------------------------------------------------------------

    #[test]
    fn test_clip_grad_value_clamps_elements() {
        let p = param_with_grad(&[4], &[-5.0, 0.5, 3.0, -0.1]);
        clip_grad_value_(&[&p], 1.0).unwrap();

        let g = p.grad().unwrap().unwrap();
        let d = g.data().unwrap();
        assert!((d[0] - (-1.0)).abs() < 1e-6);
        assert!((d[1] - 0.5).abs() < 1e-6);
        assert!((d[2] - 1.0).abs() < 1e-6);
        assert!((d[3] - (-0.1)).abs() < 1e-6);
    }

    #[test]
    fn test_clip_grad_value_skips_none_grads() {
        let p = Parameter::<f32>::zeros(&[3]).unwrap();
        // No gradient — should succeed without error.
        clip_grad_value_(&[&p], 1.0).unwrap();
        assert!(p.grad().unwrap().is_none());
    }

    #[test]
    fn test_clip_grad_value_preserves_within_range() {
        let p = param_with_grad(&[3], &[0.1, -0.2, 0.3]);
        clip_grad_value_(&[&p], 1.0).unwrap();

        let g = p.grad().unwrap().unwrap();
        let d = g.data().unwrap();
        assert!((d[0] - 0.1).abs() < 1e-6);
        assert!((d[1] - (-0.2)).abs() < 1e-6);
        assert!((d[2] - 0.3).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // GPU tests — require a real CUDA device
    // -----------------------------------------------------------------------

    /// Helper: create a parameter whose gradient lives on `Device::Cuda(0)`.
    ///
    /// Constructs the gradient on CPU then transfers it via `.to(Device::Cuda(0))`.
    #[cfg(feature = "cuda")]
    fn param_with_gpu_grad(shape: &[usize], grad_data: &[f32]) -> Parameter<f32> {
        let p = Parameter::<f32>::zeros(shape).unwrap();
        let grad_cpu = Tensor::from_storage(
            TensorStorage::cpu(grad_data.to_vec()),
            shape.to_vec(),
            false,
        )
        .unwrap();
        let grad_gpu = grad_cpu.to(Device::Cuda(0)).unwrap();
        p.set_grad(Some(grad_gpu)).unwrap();
        p
    }

    /// GPU L2 `clip_grad_norm_` — numerical correctness vs CPU reference.
    ///
    /// Constructs two parameters with CUDA-resident gradients, clips with
    /// `max_norm = 2.5`, and asserts:
    /// 1. Returned `total_norm` matches the CPU reference within 1e-5.
    /// 2. Post-clip gradients are still on CUDA.
    /// 3. Post-clip gradient values match the CPU-reference values within 1e-5.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_clip_grad_norm_l2_f32() {
        use ferrotorch_gpu::init_cuda_backend;
        init_cuda_backend().expect("CUDA init failed");

        // CPU reference: grad = [3.0, 4.0], norm = 5.0, clip to 2.5.
        let p_cpu = param_with_grad(&[2], &[3.0, 4.0]);
        let ref_norm = clip_grad_norm_(&[&p_cpu], 2.5, 2.0).unwrap();
        let ref_g = p_cpu.grad().unwrap().unwrap();
        let ref_d = ref_g.data().unwrap().to_vec();

        // GPU version with the same data.
        let p_gpu = param_with_gpu_grad(&[2], &[3.0, 4.0]);
        let gpu_norm = clip_grad_norm_(&[&p_gpu], 2.5, 2.0).unwrap();

        // Returned norm matches reference.
        assert!(
            (gpu_norm - ref_norm).abs() < 1e-5,
            "GPU norm {gpu_norm} != CPU norm {ref_norm}"
        );

        // Gradient is still on CUDA after clipping.
        let g_after = p_gpu.grad().unwrap().unwrap();
        assert!(
            g_after.is_cuda(),
            "gradient should stay on CUDA after clip_grad_norm_"
        );

        // Numerical values match.
        let gpu_d = g_after.data_vec().unwrap();
        for (i, (&gv, &rv)) in gpu_d.iter().zip(ref_d.iter()).enumerate() {
            assert!((gv - rv).abs() < 1e-5, "gradient[{i}]: GPU={gv} CPU={rv}");
        }
    }

    /// GPU L2 `clip_grad_norm_` with f64 gradients.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_clip_grad_norm_l2_f64() {
        use ferrotorch_gpu::init_cuda_backend;
        init_cuda_backend().expect("CUDA init failed");

        // Build a GPU f64 parameter by promoting f32 data.
        let p_gpu = {
            let p = Parameter::<f64>::zeros(&[3]).unwrap();
            let grad_cpu = Tensor::<f64>::from_storage(
                TensorStorage::cpu(vec![1.0_f64, 2.0, 2.0]),
                vec![3],
                false,
            )
            .unwrap();
            let grad_gpu = grad_cpu.to(Device::Cuda(0)).unwrap();
            p.set_grad(Some(grad_gpu)).unwrap();
            p
        };

        // Reference: L2 norm = sqrt(1+4+4) = 3.0, max_norm = 10.0 → no clip.
        let norm = clip_grad_norm_(&[&p_gpu], 10.0, 2.0).unwrap();
        assert!(
            (norm - 3.0).abs() < 1e-9,
            "expected L2 norm 3.0, got {norm}"
        );
        // Gradient must remain on CUDA.
        assert!(
            p_gpu.grad().unwrap().unwrap().is_cuda(),
            "f64 gradient should stay on CUDA after clip_grad_norm_"
        );
    }

    /// GPU `clip_grad_value_` — numerical correctness vs CPU reference.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_gpu_clip_grad_value_f32() {
        use ferrotorch_gpu::init_cuda_backend;
        init_cuda_backend().expect("CUDA init failed");

        let data = [-5.0_f32, 0.5, 3.0, -0.1];

        // CPU reference.
        let p_cpu = param_with_grad(&[4], &data);
        clip_grad_value_(&[&p_cpu], 1.0).unwrap();
        let ref_d = p_cpu.grad().unwrap().unwrap().data().unwrap().to_vec();

        // GPU version.
        let p_gpu = param_with_gpu_grad(&[4], &data);
        clip_grad_value_(&[&p_gpu], 1.0).unwrap();

        let g_after = p_gpu.grad().unwrap().unwrap();
        assert!(
            g_after.is_cuda(),
            "gradient should stay on CUDA after clip_grad_value_"
        );

        let gpu_d = g_after.data_vec().unwrap();
        for (i, (&gv, &rv)) in gpu_d.iter().zip(ref_d.iter()).enumerate() {
            assert!((gv - rv).abs() < 1e-5, "clamped[{i}]: GPU={gv} CPU={rv}");
        }
    }

    /// Mixed-device error test: one CPU grad, one CUDA grad → DeviceMismatch.
    #[test]
    #[cfg(feature = "cuda")]
    fn test_mixed_device_returns_device_mismatch() {
        use ferrotorch_gpu::init_cuda_backend;
        init_cuda_backend().expect("CUDA init failed");

        let p_cpu = param_with_grad(&[2], &[1.0, 2.0]);
        let p_gpu = param_with_gpu_grad(&[2], &[3.0, 4.0]);

        let result = clip_grad_norm_(&[&p_cpu, &p_gpu], 5.0, 2.0);
        assert!(
            matches!(result, Err(FerrotorchError::DeviceMismatch { .. })),
            "expected DeviceMismatch, got {result:?}"
        );

        // Same for clip_grad_value_.
        let result2 = clip_grad_value_(&[&p_cpu, &p_gpu], 1.0);
        assert!(
            matches!(result2, Err(FerrotorchError::DeviceMismatch { .. })),
            "expected DeviceMismatch for clip_grad_value_, got {result2:?}"
        );
    }

    /// Non-L2 norm on CUDA → NotImplementedOnCuda error (no silent CPU fallback).
    #[test]
    #[cfg(feature = "cuda")]
    fn test_non_l2_cuda_returns_error() {
        use ferrotorch_gpu::init_cuda_backend;
        init_cuda_backend().expect("CUDA init failed");

        let p_gpu = param_with_gpu_grad(&[2], &[1.0, 2.0]);

        // norm_type = 1.0 (L1) on CUDA should be an explicit error.
        let result = clip_grad_norm_(&[&p_gpu], 5.0, 1.0);
        assert!(
            matches!(result, Err(FerrotorchError::NotImplementedOnCuda { .. })),
            "expected NotImplementedOnCuda for L1 norm on CUDA, got {result:?}"
        );

        // norm_type = inf on CUDA should also be an explicit error.
        let p_gpu2 = param_with_gpu_grad(&[2], &[1.0, 2.0]);
        let result2 = clip_grad_norm_(&[&p_gpu2], 5.0, f64::INFINITY);
        assert!(
            matches!(result2, Err(FerrotorchError::NotImplementedOnCuda { .. })),
            "expected NotImplementedOnCuda for inf norm on CUDA, got {result2:?}"
        );
    }
}
