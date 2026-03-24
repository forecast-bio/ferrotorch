//! GPU backend dispatch layer.
//!
//! ferrotorch-core defines the [`GpuBackend`] trait and [`GpuBufferHandle`].
//! ferrotorch-gpu (or any other GPU crate) implements and registers a backend.
//! This avoids circular dependencies: core doesn't depend on gpu.

use std::any::Any;
use std::sync::OnceLock;

use crate::error::{FerrotorchError, FerrotorchResult};

// ---------------------------------------------------------------------------
// GpuRngState — serializable GPU RNG state for checkpoint save/restore
// ---------------------------------------------------------------------------

/// Serializable snapshot of a GPU device's RNG state.
///
/// This is defined in `ferrotorch-core` (not `ferrotorch-gpu`) so that the
/// checkpoint module can save/restore GPU RNG state without depending on the
/// GPU crate directly. The GPU backend implementation is responsible for
/// converting this to/from its internal representation (e.g., `PhiloxState`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRngState {
    /// RNG counter value.
    pub counter: u64,
    /// RNG seed.
    pub seed: u64,
    /// Offset within the current random number group.
    pub offset: u64,
    /// Device ordinal this state belongs to.
    pub device: usize,
}

/// Opaque handle to GPU memory.
///
/// ferrotorch-core doesn't know what's inside -- the GPU backend provides
/// the concrete type (e.g., `CudaBuffer<f32>`). We store it as
/// `Box<dyn Any + Send + Sync>` for type erasure.
pub struct GpuBufferHandle {
    pub(crate) inner: Box<dyn Any + Send + Sync>,
    pub(crate) device_ordinal: usize,
    pub(crate) len: usize,
}

impl GpuBufferHandle {
    pub fn new(inner: Box<dyn Any + Send + Sync>, device_ordinal: usize, len: usize) -> Self {
        Self { inner, device_ordinal, len }
    }

    #[inline]
    pub fn device_ordinal(&self) -> usize { self.device_ordinal }

    #[inline]
    pub fn len(&self) -> usize { self.len }

    #[inline]
    pub fn is_empty(&self) -> bool { self.len == 0 }

    pub fn downcast_ref<T: 'static>(&self) -> Option<&T> {
        self.inner.downcast_ref()
    }

    pub fn downcast_mut<T: 'static>(&mut self) -> Option<&mut T> {
        self.inner.downcast_mut()
    }

    /// Consume the handle and extract the inner value as a concrete type.
    pub fn into_inner<T: 'static>(self) -> Result<T, Box<dyn Any + Send + Sync>> {
        self.inner.downcast::<T>().map(|b| *b)
    }
}

impl std::fmt::Debug for GpuBufferHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuBufferHandle")
            .field("device", &self.device_ordinal)
            .field("len", &self.len)
            .finish()
    }
}

/// Trait that GPU backends implement to handle tensor operations.
///
/// ferrotorch-core calls these methods; ferrotorch-gpu provides the implementation.
pub trait GpuBackend: Send + Sync {
    /// Downcast to `&dyn Any` for backend-specific access (e.g., getting the
    /// underlying `GpuDevice` for CUDA graph capture).
    fn as_any(&self) -> &dyn std::any::Any;
    fn cpu_to_gpu(&self, data: &[u8], elem_size: usize, device: usize) -> FerrotorchResult<GpuBufferHandle>;
    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>>;
    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn alloc_zeros(&self, len: usize, elem_size: usize, device: usize) -> FerrotorchResult<GpuBufferHandle>;

    // Elementwise f32
    fn add_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn sub_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn mul_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn neg_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn relu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;

    // Linalg f32
    fn matmul_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, m: usize, k: usize, n: usize) -> FerrotorchResult<GpuBufferHandle>;

    /// Mixed-precision matmul: cast f32 inputs to f16, multiply, accumulate
    /// back to f32. Used by autocast when the category is `ReducedPrecision`.
    ///
    /// Default implementation falls back to `matmul_f32` (no precision
    /// reduction) until a real f16 GEMM kernel is available.
    ///
    /// # NaN / Inf propagation
    ///
    /// f16 has a much smaller dynamic range than f32 (max ~65504). Values
    /// outside that range will overflow to inf or underflow to zero when cast.
    /// Callers relying on autocast should ensure their model weights stay
    /// within f16-representable bounds (which is normal for trained networks).
    fn matmul_f16_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, m: usize, k: usize, n: usize) -> FerrotorchResult<GpuBufferHandle> {
        // Fallback: no f16 kernel available, use full-precision f32.
        self.matmul_f32(a, b, m, k, n)
    }

    // Reduction f32
    fn sum_f32(&self, a: &GpuBufferHandle, len: usize) -> FerrotorchResult<GpuBufferHandle>;

    // Elementwise f64 (default impls return "not yet implemented" errors)
    fn add_f64(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }
    fn sub_f64(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }
    fn mul_f64(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }
    fn neg_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }
    fn relu_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }

    // Linalg f64
    fn matmul_f64(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle, _m: usize, _k: usize, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }

    // Reduction f64
    fn sum_f64(&self, _a: &GpuBufferHandle, _numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "f64 GPU ops not yet implemented".into() })
    }

    // Broadcast binary f32
    fn broadcast_add_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, a_shape: &[usize], b_shape: &[usize], out_shape: &[usize]) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_sub_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, a_shape: &[usize], b_shape: &[usize], out_shape: &[usize]) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_mul_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, a_shape: &[usize], b_shape: &[usize], out_shape: &[usize]) -> FerrotorchResult<GpuBufferHandle>;

    // Softmax f32 (row-wise over last dim)
    fn softmax_f32(&self, a: &GpuBufferHandle, rows: usize, cols: usize) -> FerrotorchResult<GpuBufferHandle>;

    // Dropout f32 (inverted dropout)
    fn dropout_f32(&self, a: &GpuBufferHandle, threshold: u32, scale: f32, seed: u32) -> FerrotorchResult<GpuBufferHandle>;

    /// Dropout using the Philox CBRNG for deterministic, reproducible mask generation.
    ///
    /// Instead of a simple u32 seed, this takes a `GpuRngState` that specifies the
    /// exact Philox counter and key to use. This enables gradient checkpointing to
    /// reproduce identical dropout masks by restoring the RNG state.
    ///
    /// The method also advances the global GPU RNG state by `ceil(n/4)` counters.
    ///
    /// Returns the dropped-out buffer and the Philox state that was used (for
    /// backward mask regeneration).
    fn dropout_philox_f32(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuRngState)> {
        // Default: fall back to the non-Philox version with a dummy seed.
        // The returned state has device=0 as a placeholder.
        let result = self.dropout_f32(a, threshold, scale, 0)?;
        Ok((result, GpuRngState { counter: 0, seed: 0, offset: 0, device: 0 }))
    }

    // 2D transpose f32
    fn transpose_2d_f32(&self, a: &GpuBufferHandle, m: usize, n: usize) -> FerrotorchResult<GpuBufferHandle>;

    // 4D permute (0,2,1,3) f32 — swap dims 1 and 2
    fn permute_0213_f32(&self, a: &GpuBufferHandle, d0: usize, d1: usize, d2: usize, d3: usize) -> FerrotorchResult<GpuBufferHandle>;

    // Batched matmul f32: C[i] = A[i] @ B[i] for i in 0..batch
    fn bmm_f32(&self, a: &GpuBufferHandle, b: &GpuBufferHandle, batch: usize, m: usize, k: usize, n: usize) -> FerrotorchResult<GpuBufferHandle>;

    // GELU activation f32
    fn gelu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;

    // LayerNorm f32 (row-wise, with affine)
    fn layernorm_f32(&self, input: &GpuBufferHandle, weight: &GpuBufferHandle, bias: &GpuBufferHandle, rows: usize, cols: usize, eps: f32) -> FerrotorchResult<GpuBufferHandle>;

    // Slice write: write [N, D] into row `pos` of [N, max_len, D] (in-place)
    fn slice_write_f32(&self, src: &GpuBufferHandle, dst: &mut GpuBufferHandle, n_batch: usize, d: usize, max_len: usize, pos: usize) -> FerrotorchResult<()>;

    // Slice read: read first `len` rows from [N, max_len, D] → [N, len, D]
    fn slice_read_f32(&self, src: &GpuBufferHandle, n_batch: usize, d: usize, len: usize, max_len: usize) -> FerrotorchResult<GpuBufferHandle>;

    // Embedding lookup: gather row `idx` from weight [V, D] → [D]
    fn embed_lookup_f32(&self, idx: &GpuBufferHandle, weight: &GpuBufferHandle, d: usize) -> FerrotorchResult<GpuBufferHandle>;

    // Scalar multiply: out[i] = a[i] * scalar
    fn scale_f32(&self, a: &GpuBufferHandle, scalar: f32) -> FerrotorchResult<GpuBufferHandle>;

    // Backward activation kernels
    // relu_backward: out[i] = (input[i] > 0) ? grad[i] : 0
    fn relu_backward_f32(&self, grad: &GpuBufferHandle, input: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    // gelu_backward: out[i] = grad[i] * (sig + 1.702*x*sig*(1-sig)) where sig = sigmoid(1.702*x)
    fn gelu_backward_f32(&self, grad: &GpuBufferHandle, input: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;

    // Indexing operations
    // index_select_1d: out[i] = input[indices[i]]  (indices stored as f32)
    fn index_select_1d_f32(&self, input: &GpuBufferHandle, indices: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    // scatter_add_1d: out = zeros(input_len); for i: out[indices[i]] += grad_output[i]  (atomic)
    fn scatter_add_1d_f32(&self, grad_output: &GpuBufferHandle, indices: &GpuBufferHandle, input_len: usize) -> FerrotorchResult<GpuBufferHandle>;
    // masked_fill: out[i] = mask[i] ? value : input[i]  (mask stored as f32, 1.0/0.0)
    fn masked_fill_f32(&self, input: &GpuBufferHandle, mask: &GpuBufferHandle, value: f32) -> FerrotorchResult<GpuBufferHandle>;
    // masked_zero: out[i] = mask[i] ? 0.0 : grad[i]  (backward of masked_fill)
    fn masked_zero_f32(&self, grad: &GpuBufferHandle, mask: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;

    // Elementwise unary/binary f32 (default impls for forward ops)
    fn div_f32(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "div_f32 GPU op not yet implemented".into() })
    }
    fn exp_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "exp_f32 GPU op not yet implemented".into() })
    }
    fn log_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "log_f32 GPU op not yet implemented".into() })
    }
    fn sqrt_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "sqrt_f32 GPU op not yet implemented".into() })
    }
    fn pow_f32(&self, _a: &GpuBufferHandle, _exponent: f32) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "pow_f32 GPU op not yet implemented".into() })
    }
    fn abs_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "abs_f32 GPU op not yet implemented".into() })
    }
    fn sigmoid_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "sigmoid_f32 GPU op not yet implemented".into() })
    }
    fn tanh_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "tanh_f32 GPU op not yet implemented".into() })
    }

    // Sigmoid backward: out[i] = grad[i] * output[i] * (1 - output[i])
    fn sigmoid_backward_f32(&self, _grad: &GpuBufferHandle, _output: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "sigmoid_backward_f32 GPU op not yet implemented".into() })
    }

    // Tanh backward: out[i] = grad[i] * (1 - output[i]^2)
    fn tanh_backward_f32(&self, _grad: &GpuBufferHandle, _output: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "tanh_backward_f32 GPU op not yet implemented".into() })
    }

    // Softmax backward: out[i] = output[i] * (grad[i] - dot(grad_row, output_row))
    fn softmax_backward_f32(&self, _grad: &GpuBufferHandle, _output: &GpuBufferHandle, _cols: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "softmax_backward_f32 GPU op not yet implemented".into() })
    }

    // LayerNorm backward: computes grad_input, grad_weight, grad_bias on GPU
    fn layernorm_backward_f32(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument { message: "layernorm_backward_f32 GPU op not yet implemented".into() })
    }

    // Sum along one axis of a tensor
    fn sum_axis_f32(&self, _a: &GpuBufferHandle, _shape: &[usize], _axis: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "sum_axis_f32 GPU op not yet implemented".into() })
    }

    /// Check if a GPU buffer contains any inf or NaN values.
    fn has_inf_nan_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<bool> {
        // Default: download to CPU and scan
        let bytes = self.gpu_to_cpu(a)?;
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, bytes.len() / 4)
        };
        Ok(floats.iter().any(|v| !v.is_finite()))
    }

    // GPU RNG state management (for gradient checkpointing)
    /// Save the current GPU RNG state for a device. Used by checkpoint to
    /// ensure dropout masks are identical on recomputation.
    fn save_rng_state(&self, device: usize) -> FerrotorchResult<GpuRngState> {
        Err(FerrotorchError::InvalidArgument {
            message: format!("save_rng_state not implemented for device {device}"),
        })
    }

    /// Restore a previously saved GPU RNG state for a device.
    fn restore_rng_state(&self, state: GpuRngState) -> FerrotorchResult<()> {
        let _ = state;
        Err(FerrotorchError::InvalidArgument {
            message: "restore_rng_state not implemented".into(),
        })
    }

    // GPU linear algebra via cuSOLVER
    fn svd_f32(&self, _a: &GpuBufferHandle, _m: usize, _n: usize) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument { message: "svd_f32 GPU op not yet implemented".into() })
    }
    fn cholesky_f32(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "cholesky_f32 GPU op not yet implemented".into() })
    }
    fn solve_f32(&self, _a: &GpuBufferHandle, _b: &GpuBufferHandle, _n: usize, _nrhs: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument { message: "solve_f32 GPU op not yet implemented".into() })
    }
    fn qr_f32(&self, _a: &GpuBufferHandle, _m: usize, _n: usize) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument { message: "qr_f32 GPU op not yet implemented".into() })
    }
}

static GPU_BACKEND: OnceLock<Box<dyn GpuBackend>> = OnceLock::new();

/// Register a GPU backend. Called once by the GPU crate on init.
pub fn register_gpu_backend(backend: Box<dyn GpuBackend>) -> Result<(), Box<dyn GpuBackend>> {
    GPU_BACKEND.set(backend)
}

/// Get the registered GPU backend, if any.
pub fn gpu_backend() -> Option<&'static dyn GpuBackend> {
    GPU_BACKEND.get().map(|b| b.as_ref())
}

/// Returns `true` if a GPU backend has been registered.
pub fn has_gpu_backend() -> bool {
    GPU_BACKEND.get().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_buffer_handle() {
        let handle = GpuBufferHandle::new(Box::new(42u64), 0, 100);
        assert_eq!(handle.device_ordinal(), 0);
        assert_eq!(handle.len(), 100);
        assert!(!handle.is_empty());
        assert_eq!(handle.downcast_ref::<u64>(), Some(&42));
    }

    #[test]
    fn test_gpu_buffer_handle_debug() {
        let handle = GpuBufferHandle::new(Box::new(()), 1, 50);
        let s = format!("{handle:?}");
        assert!(s.contains("device: 1"));
    }
}
