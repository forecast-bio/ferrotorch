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
///
/// Fields are crate-private; construct via [`GpuRngState::new`] and read via
/// the [`Self::counter`], [`Self::seed`], [`Self::offset`], [`Self::device`]
/// accessors. Encapsulation lets the layout evolve (e.g. add a Philox-key
/// pair) without a workspace-wide breaking change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GpuRngState {
    /// RNG counter value.
    pub(crate) counter: u64,
    /// RNG seed.
    pub(crate) seed: u64,
    /// Offset within the current random number group.
    pub(crate) offset: u64,
    /// Device ordinal this state belongs to.
    pub(crate) device: usize,
}

impl GpuRngState {
    /// Construct a new RNG state snapshot.
    #[inline]
    #[must_use]
    pub fn new(counter: u64, seed: u64, offset: u64, device: usize) -> Self {
        Self {
            counter,
            seed,
            offset,
            device,
        }
    }

    /// RNG counter value at the time of the snapshot.
    #[inline]
    #[must_use]
    pub fn counter(&self) -> u64 {
        self.counter
    }

    /// RNG seed used by this generator.
    #[inline]
    #[must_use]
    pub fn seed(&self) -> u64 {
        self.seed
    }

    /// Offset within the current random number group.
    #[inline]
    #[must_use]
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// Device ordinal this RNG state belongs to.
    #[inline]
    #[must_use]
    pub fn device(&self) -> usize {
        self.device
    }
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
        Self {
            inner,
            device_ordinal,
            len,
        }
    }

    #[inline]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

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
            .field("inner", &"<dyn Any + Send + Sync>")
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
    fn cpu_to_gpu(
        &self,
        data: &[u8],
        elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>>;

    /// Get the raw CUDA device pointer from a buffer handle.
    ///
    /// Returns null if the handle type is not recognized or the backend
    /// doesn't support raw pointer access.
    fn raw_device_ptr(&self, _handle: &GpuBufferHandle) -> *const std::ffi::c_void {
        std::ptr::null()
    }

    /// Get a mutable raw CUDA device pointer from a buffer handle.
    fn raw_device_ptr_mut(&self, _handle: &mut GpuBufferHandle) -> *mut std::ffi::c_void {
        std::ptr::null_mut()
    }

    /// Get the element size (in bytes) of the data stored in a buffer handle.
    /// Returns 0 if unknown.
    fn buffer_elem_size(&self, _handle: &GpuBufferHandle) -> usize {
        0
    }

    /// Copy CPU data to GPU via pinned (page-locked) host memory.
    ///
    /// ~2x faster than [`cpu_to_gpu`] for large buffers due to DMA transfers.
    /// Falls back to regular `cpu_to_gpu` by default.
    fn cpu_to_gpu_pinned(
        &self,
        data: &[u8],
        elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.cpu_to_gpu(data, elem_size, device)
    }
    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn alloc_zeros(
        &self,
        len: usize,
        elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;

    // Elementwise f32
    fn add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn neg_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn relu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;

    // Linalg f32
    fn matmul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;

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
    fn matmul_f16_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        // Fallback: no f16 kernel available, use full-precision f32.
        self.matmul_f32(a, b, m, k, n)
    }

    // Reduction f32
    fn sum_f32(&self, a: &GpuBufferHandle, len: usize) -> FerrotorchResult<GpuBufferHandle>;

    /// f32 product reduction. Returns a 1-element buffer holding the
    /// product of all elements. (#524)
    fn prod_f32(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU reduce_prod not implemented for this backend".into(),
        })
    }

    /// f32 parallel min reduction. Returns a 1-element buffer holding the
    /// minimum element of `a`. Default impl returns the
    /// "not yet implemented" error so existing backends compile unchanged
    /// — concrete backends override. (#627)
    fn min_f32(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU reduce_min not implemented for this backend".into(),
        })
    }

    /// f32 parallel max reduction. Counterpart of [`Self::min_f32`]. (#627)
    fn max_f32(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU reduce_max not implemented for this backend".into(),
        })
    }

    /// f32 fused masked-min reduction (#627). Single-pass kernel that
    /// folds `(data, mask_f) -> min` directly, where `mask_f[i]` is 1.0
    /// for valid entries and 0.0 for masked. Avoids the
    /// `mul + add + reduce` chain that the unfused path requires.
    fn masked_min_f32(
        &self,
        _data: &GpuBufferHandle,
        _mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU masked_reduce_min not implemented for this backend".into(),
        })
    }

    /// f32 fused masked-max counterpart of [`Self::masked_min_f32`]. (#627)
    fn masked_max_f32(
        &self,
        _data: &GpuBufferHandle,
        _mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "GPU masked_reduce_max not implemented for this backend".into(),
        })
    }

    // Elementwise f64 (default impls return "not yet implemented" errors)
    fn add_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }
    fn sub_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }
    fn mul_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }
    fn neg_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }
    fn relu_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }

    // Linalg f64
    fn matmul_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }

    // -- Vector matmul kernels (#816 / #817 / #818) ---------------------------
    //
    // These cover the rank combinations that PyTorch's `torch.matmul` (and
    // `torch.dot` / `torch.mv`) accepts on CUDA but ferrotorch previously
    // routed through CPU-only specialised paths, surfacing as
    // `GpuTensorNotAccessible` for CUDA inputs.
    //
    // - `dot_*` : 1D x 1D inner product (cuBLAS `{S,D}dot`)
    // - `mv_*`  : 2D x 1D matrix-vector product (cuBLAS `{S,D}gemv`, OP_T)
    // - `vm_*`  : 1D x 2D vector-matrix product (cuBLAS `{S,D}gemv`, OP_N)
    //
    // CUDA is the primary GPU backend; backends that don't implement these
    // (or aren't CUDA at all) inherit the default `Err` impl. CUDA itself
    // overrides them with real cuBLAS kernels.

    /// 1D x 1D dot product on GPU. Returns a 1-element buffer.
    /// `n` is the shared length of both inputs (each buffer holds `n` elements).
    fn dot_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "dot_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// 1D x 1D dot product on GPU (f64 dtype).
    fn dot_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "dot_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// 2D x 1D matrix-vector product `y[m] = A[m,k] @ x[k]`. Returns a
    /// buffer of length `m`.
    fn mv_f32(
        &self,
        _a: &GpuBufferHandle,
        _x: &GpuBufferHandle,
        _m: usize,
        _k: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mv_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// 2D x 1D matrix-vector product on GPU (f64 dtype).
    fn mv_f64(
        &self,
        _a: &GpuBufferHandle,
        _x: &GpuBufferHandle,
        _m: usize,
        _k: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mv_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// 1D x 2D vector-matrix product `y[n] = x[k] @ B[k,n]`. Returns a
    /// buffer of length `n`. Implemented via `gemv` with the transpose flag
    /// — does NOT materialise a transposed copy of `B`.
    fn vm_f32(
        &self,
        _x: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "vm_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// 1D x 2D vector-matrix product on GPU (f64 dtype).
    fn vm_f64(
        &self,
        _x: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "vm_f64 GPU op not implemented for this backend".into(),
        })
    }

    // -- Broadcast / 4D matmul kernel (#819) ----------------------------------
    //
    // PyTorch's `torch.matmul` supports arbitrary leading-dim broadcast on
    // CUDA — `(B, M, K) @ (K, N)`, `(M, K) @ (B, K, N)`, full 4D bmm,
    // `(2, 1, M, K) @ (2, 4, K, N)`, etc. Pre-fix these shapes fell through
    // to `linalg::matmul` (CPU path) and surfaced as `GpuTensorNotAccessible`
    // for CUDA inputs. Post-fix, `matmul_differentiable` routes them to
    // `broadcast_bmm_f{32,64}` which lower to `cublas{S,D}gemmStridedBatched`
    // — stride=0 on broadcasted axes, no `expand` materialisation.
    //
    // Inputs:
    // - `a`/`b`: GPU buffers, contiguous, in row-major batch layout. The
    //   caller has already ensured non-broadcasted dims match and that
    //   `a.len() == a_batch_count * m * k`, `b.len() == b_batch_count * k * n`.
    // - `out_lead`: the broadcasted leading-dim shape (excluding `m, n`).
    //   `batch = product(out_lead)`. Output shape is `out_lead + [m, n]`.
    // - `a_lead`/`b_lead`: per-leading-axis sizes for A and B. Where the
    //   axis size is 1 vs `out_lead[i]`, that axis is treated as broadcast.
    //   Lengths may be shorter than `out_lead` (implicit batch=1 prefix).
    // - `m`, `k`, `n`: per-batch matmul dims.

    /// Broadcast / batched matmul on GPU (f32 dtype).
    fn broadcast_bmm_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_lead: &[usize],
        _b_lead: &[usize],
        _out_lead: &[usize],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_bmm_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Broadcast / batched matmul on GPU (f64 dtype).
    fn broadcast_bmm_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_lead: &[usize],
        _b_lead: &[usize],
        _out_lead: &[usize],
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_bmm_f64 GPU op not implemented for this backend".into(),
        })
    }

    // Reduction f64
    fn sum_f64(&self, _a: &GpuBufferHandle, _numel: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU ops not yet implemented".into(),
        })
    }

    /// f64 product reduction. (#524)
    fn prod_f64(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU reduce_prod not implemented for this backend".into(),
        })
    }

    /// f32 backward of the global `prod` reduction (#785).
    ///
    /// Returns `grad_input[i] = grad_output * (prod_{j != i} input[j])`,
    /// which matches PyTorch's exact zero-handling semantics:
    /// no zeros → `grad_input = grad_output * total / input`; one zero
    /// at index z → only `grad_input[z]` is nonzero (the product of the
    /// remaining elements); two or more zeros → all zero.
    ///
    /// `grad_output` is a scalar (`numel == 1`).
    fn prod_backward_f32(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "prod_backward_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 backward of the global `prod` reduction (#785). Companion of
    /// [`Self::prod_backward_f32`].
    fn prod_backward_f64(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "prod_backward_f64 GPU op not yet implemented".into(),
        })
    }

    /// f64 parallel min reduction. (#627)
    fn min_f64(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU reduce_min not implemented for this backend".into(),
        })
    }

    /// f64 parallel max reduction. (#627)
    fn max_f64(&self, _a: &GpuBufferHandle, _len: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU reduce_max not implemented for this backend".into(),
        })
    }

    /// f64 fused masked-min reduction (#627).
    fn masked_min_f64(
        &self,
        _data: &GpuBufferHandle,
        _mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU masked_reduce_min not implemented for this backend".into(),
        })
    }

    /// f64 fused masked-max reduction (#627).
    fn masked_max_f64(
        &self,
        _data: &GpuBufferHandle,
        _mask_f: &GpuBufferHandle,
        _len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "f64 GPU masked_reduce_max not implemented for this backend".into(),
        })
    }

    // Broadcast binary f32
    fn broadcast_add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_add_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_add_f64 GPU op not yet implemented".into(),
        })
    }
    fn broadcast_sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_sub_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_sub_f64 GPU op not yet implemented".into(),
        })
    }
    fn broadcast_mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_mul_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_mul_f64 GPU op not yet implemented".into(),
        })
    }
    fn broadcast_div_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        a_shape: &[usize],
        b_shape: &[usize],
        out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn broadcast_div_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _a_shape: &[usize],
        _b_shape: &[usize],
        _out_shape: &[usize],
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "broadcast_div_f64 GPU op not yet implemented".into(),
        })
    }

    // Softmax f32 (row-wise over last dim)
    fn softmax_f32(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn softmax_f64(
        &self,
        _a: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax_f64 GPU op not yet implemented".into(),
        })
    }

    // Dropout f32 (inverted dropout)
    fn dropout_f32(
        &self,
        a: &GpuBufferHandle,
        threshold: u32,
        scale: f32,
        seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn dropout_f64(
        &self,
        _a: &GpuBufferHandle,
        _threshold: u32,
        _scale: f64,
        _seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "dropout_f64 GPU op not yet implemented".into(),
        })
    }

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
        Ok((
            result,
            GpuRngState {
                counter: 0,
                seed: 0,
                offset: 0,
                device: 0,
            },
        ))
    }
    fn dropout_philox_f64(
        &self,
        _a: &GpuBufferHandle,
        _threshold: u32,
        _scale: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuRngState)> {
        Err(FerrotorchError::InvalidArgument {
            message: "dropout_philox_f64 GPU op not yet implemented".into(),
        })
    }

    // 2D transpose f32
    fn transpose_2d_f32(
        &self,
        a: &GpuBufferHandle,
        m: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn transpose_2d_f64(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "transpose_2d_f64 GPU op not yet implemented".into(),
        })
    }

    // 4D permute (0,2,1,3) f32 — swap dims 1 and 2
    fn permute_0213_f32(
        &self,
        a: &GpuBufferHandle,
        d0: usize,
        d1: usize,
        d2: usize,
        d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn permute_0213_f64(
        &self,
        _a: &GpuBufferHandle,
        _d0: usize,
        _d1: usize,
        _d2: usize,
        _d3: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "permute_0213_f64 GPU op not yet implemented".into(),
        })
    }

    // Batched matmul f32: C[i] = A[i] @ B[i] for i in 0..batch
    fn bmm_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn bmm_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "bmm_f64 GPU op not yet implemented".into(),
        })
    }

    /// Batched matmul with f16 Tensor Core acceleration.
    /// Takes f32 handles, converts to f16 internally, accumulates in f32.
    fn bmm_f16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "bmm_f16_f32 GPU op not yet implemented".into(),
        })
    }

    // -- bf16 × f32-accumulator mixed-precision kernels (#518) ---------------
    //
    // PyTorch parity (rust-gpu-discipline §3): under `torch.autocast(device_type
    // ="cuda", dtype=torch.bfloat16)`, `torch.matmul`, `torch.bmm`, and
    // `torch.softmax` on CUDA tensors use bf16 inputs with f32 accumulation via
    // cuBLAS GemmEx (CUDA_R_16BF / CUBLAS_COMPUTE_32F).
    //
    // Default impls return `Err` so existing backends compile unchanged. The
    // CUDA backend overrides these three methods with real cuBLAS / PTX kernels.
    // There is NO silent CPU fallback (§3 hard requirement).

    /// Matrix multiply: bf16 inputs (f32-buffers converted to bf16 on-device)
    /// → f32 output via cuBLAS GemmEx (CUDA_R_16BF / CUBLAS_COMPUTE_32F).
    ///
    /// Signature mirrors [`Self::matmul_f16_f32`]; dtype is bf16 instead of f16.
    /// bf16 has a wider exponent range than f16 (same 8-bit exponent as f32),
    /// making it more robust to large weight values typical in transformers.
    fn matmul_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "matmul_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Batched matrix multiply: bf16 inputs → f32 output.
    ///
    /// `a` is `[batch, m, k]`, `b` is `[batch, k, n]`, result is `[batch, m, n]`.
    /// All tensors are passed as f32 handles; inputs are converted to bf16
    /// on-device before the cuBLAS GemmStridedBatchedEx call.
    fn bmm_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _batch: usize,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "bmm_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Row-wise softmax: bf16 input (stored as `u16` bit-pattern buffer) →
    /// f32 output via PTX kernel with f32 accumulator.
    ///
    /// `rows` = product of all dims except the last; `cols` = last dim size.
    /// The input handle must contain a `CudaSlice<u16>` (bf16 bit patterns);
    /// the output is a `CudaBuffer<f32>` of the same shape.
    ///
    /// All phases (max-find, exp-sum, normalize) accumulate in f32 for
    /// numerical stability — matches PyTorch's bf16 softmax contract.
    fn softmax_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    // -- bf16 elementwise (#963) ---------------------------------------------

    /// Elementwise add: bf16 inputs (u16 bit-pattern buffers) -> f32 output.
    ///
    /// PyTorch parity: `torch.add(a.bfloat16(), b.bfloat16())` under autocast
    /// uses f32 accumulators on CUDA. Both handles must contain
    /// `CudaSlice<u16>` (bf16 bit patterns) of the same length `n`.
    fn add_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "add_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Elementwise subtract: bf16 inputs -> f32 output.
    fn sub_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sub_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Elementwise multiply: bf16 inputs -> f32 output.
    fn mul_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mul_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Elementwise divide: bf16 inputs -> f32 output.
    fn div_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "div_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    // -- bf16 reductions (#963) ----------------------------------------------

    /// Sum along an axis: bf16 input [outer, axis_size, inner] (u16) -> f32
    /// [outer, inner]. Accumulates in f32.
    fn sum_axis_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _axis_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sum_axis_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Mean along an axis: bf16 input [outer, axis_size, inner] (u16) -> f32
    /// [outer, inner]. Accumulates in f32, divides by axis_size in f32.
    fn mean_axis_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _axis_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mean_axis_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    // -- bf16 activations (#963) ---------------------------------------------

    /// ReLU activation: bf16 input (u16) -> f32 output.
    /// `out[i] = max(0.0, bf16_to_f32(a[i]))`.
    fn relu_bf16_f32(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "relu_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Sigmoid activation: bf16 input (u16) -> f32 output.
    /// `out[i] = 1 / (1 + exp(-bf16_to_f32(a[i])))`.
    fn sigmoid_bf16_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_bf16_f32 GPU op not implemented for this backend".into(),
        })
    }

    // GELU activation f32 (sigmoid approximation)
    fn gelu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle>;
    fn gelu_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_f64 GPU op not yet implemented".into(),
        })
    }
    // GELU activation f32 (tanh approximation: PyTorch approximate="tanh")
    fn gelu_tanh_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_tanh_f32 GPU op not yet implemented".into(),
        })
    }
    fn gelu_tanh_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_tanh_f64 GPU op not yet implemented".into(),
        })
    }
    // GELU activation f32 (exact erf: PyTorch approximate="none")
    fn gelu_erf_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_erf_f32 GPU op not yet implemented".into(),
        })
    }
    fn gelu_erf_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_erf_f64 GPU op not yet implemented".into(),
        })
    }

    // LayerNorm f32 (row-wise, with affine)
    fn layernorm_f32(
        &self,
        input: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        bias: &GpuBufferHandle,
        rows: usize,
        cols: usize,
        eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn layernorm_f64(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "layernorm_f64 GPU op not yet implemented".into(),
        })
    }

    // RMSNorm f32 (row-wise, weight only — no bias, no mean centering)
    fn rmsnorm_f32(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "rmsnorm_f32 GPU op not yet implemented".into(),
        })
    }
    fn rmsnorm_f64(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "rmsnorm_f64 GPU op not yet implemented".into(),
        })
    }

    // RMSNorm backward f32: returns (grad_input, grad_weight)
    fn rmsnorm_backward_f32(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f32,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "rmsnorm_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn rmsnorm_backward_f64(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "rmsnorm_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Slice write: write [N, D] into row `pos` of [N, max_len, D] (in-place)
    fn slice_write_f32(
        &self,
        src: &GpuBufferHandle,
        dst: &mut GpuBufferHandle,
        n_batch: usize,
        d: usize,
        max_len: usize,
        pos: usize,
    ) -> FerrotorchResult<()>;
    fn slice_write_f64(
        &self,
        _src: &GpuBufferHandle,
        _dst: &mut GpuBufferHandle,
        _n_batch: usize,
        _d: usize,
        _max_len: usize,
        _pos: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "slice_write_f64 GPU op not yet implemented".into(),
        })
    }

    // Slice read: read first `len` rows from [N, max_len, D] → [N, len, D]
    fn slice_read_f32(
        &self,
        src: &GpuBufferHandle,
        n_batch: usize,
        d: usize,
        len: usize,
        max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn slice_read_f64(
        &self,
        _src: &GpuBufferHandle,
        _n_batch: usize,
        _d: usize,
        _len: usize,
        _max_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "slice_read_f64 GPU op not yet implemented".into(),
        })
    }

    // Embedding lookup: gather row `idx` from weight [V, D] → [D]
    fn embed_lookup_f32(
        &self,
        idx: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn embed_lookup_f64(
        &self,
        _idx: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "embed_lookup_f64 GPU op not yet implemented".into(),
        })
    }

    // Batch embedding lookup: gather N rows from weight [V, D] → [N, D]
    // `indices` contains N f32 values encoding integer row indices.
    fn embed_lookup_batch_f32(
        &self,
        indices: &GpuBufferHandle,
        weight: &GpuBufferHandle,
        n: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn embed_lookup_batch_f64(
        &self,
        _indices: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _n: usize,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "embed_lookup_batch_f64 GPU op not yet implemented".into(),
        })
    }

    // Scatter-add rows: grad_weight[indices[i], :] += grad_output[i, :] for embedding backward
    // `indices` contains N f32 values, grad_output is [N, D], output is [num_embeddings, D]
    fn scatter_add_rows_f32(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        num_embeddings: usize,
        d: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn scatter_add_rows_f64(
        &self,
        _grad_output: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _num_embeddings: usize,
        _d: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "scatter_add_rows_f64 GPU op not yet implemented".into(),
        })
    }

    // Scalar multiply: out[i] = a[i] * scalar
    fn scale_f32(&self, a: &GpuBufferHandle, scalar: f32) -> FerrotorchResult<GpuBufferHandle>;
    fn scale_f64(&self, _a: &GpuBufferHandle, _scalar: f64) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "scale_f64 GPU op not yet implemented".into(),
        })
    }

    // Backward activation kernels
    // relu_backward: out[i] = (input[i] > 0) ? grad[i] : 0
    fn relu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn relu_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "relu_backward_f64 GPU op not yet implemented".into(),
        })
    }
    // abs_backward: out[i] = grad[i] * sign(input[i])  (sign(0) = 0)
    fn abs_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "abs_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn abs_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "abs_backward_f64 GPU op not yet implemented".into(),
        })
    }
    // fill: allocate an n-element device buffer filled with `scalar`.
    // Used by sum/mean backward so the grad is built entirely on-device.
    fn fill_f32(
        &self,
        _n: usize,
        _scalar: f32,
        _ordinal: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fill_f32 GPU op not yet implemented".into(),
        })
    }
    fn fill_f64(
        &self,
        _n: usize,
        _scalar: f64,
        _ordinal: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fill_f64 GPU op not yet implemented".into(),
        })
    }
    // gelu_backward (sigmoid approx): out[i] = grad[i] * (sig + 1.702*x*sig*(1-sig))
    fn gelu_backward_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn gelu_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_backward_f64 GPU op not yet implemented".into(),
        })
    }
    // gelu_backward (tanh approx)
    fn gelu_backward_tanh_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_backward_tanh_f32 GPU op not yet implemented".into(),
        })
    }
    fn gelu_backward_tanh_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_backward_tanh_f64 GPU op not yet implemented".into(),
        })
    }
    // gelu_backward (exact erf): out[i] = grad[i] * (Φ(x) + x·φ(x))
    // where Φ = normal CDF, φ = normal PDF
    fn gelu_backward_erf_f32(
        &self,
        grad: &GpuBufferHandle,
        input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn gelu_backward_erf_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "gelu_backward_erf_f64 GPU op not yet implemented".into(),
        })
    }

    // Cumulative scan operations along a dimension.
    // Parameters: (input, outer, dim_size, inner) factorize the tensor shape.
    fn cumsum_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cumsum_f32 GPU op not yet implemented".into(),
        })
    }
    fn cumsum_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cumsum_f64 GPU op not yet implemented".into(),
        })
    }
    fn cumprod_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cumprod_f32 GPU op not yet implemented".into(),
        })
    }
    fn cumprod_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cumprod_f64 GPU op not yet implemented".into(),
        })
    }
    // Returns (values, indices_as_f32)
    fn cummax_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "cummax_f32 GPU op not yet implemented".into(),
        })
    }
    fn cummax_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "cummax_f64 GPU op not yet implemented".into(),
        })
    }
    // Returns (values, indices_as_f32)
    fn cummin_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "cummin_f32 GPU op not yet implemented".into(),
        })
    }
    fn cummin_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "cummin_f64 GPU op not yet implemented".into(),
        })
    }
    fn logcumsumexp_f32(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "logcumsumexp_f32 GPU op not yet implemented".into(),
        })
    }
    fn logcumsumexp_f64(
        &self,
        _a: &GpuBufferHandle,
        _outer: usize,
        _dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "logcumsumexp_f64 GPU op not yet implemented".into(),
        })
    }

    // Clamp: out[i] = max(min_val, min(max_val, x[i]))
    fn clamp_f32(
        &self,
        _a: &GpuBufferHandle,
        _min_val: f32,
        _max_val: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "clamp_f32 GPU op not yet implemented".into(),
        })
    }
    fn clamp_f64(
        &self,
        _a: &GpuBufferHandle,
        _min_val: f64,
        _max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "clamp_f64 GPU op not yet implemented".into(),
        })
    }

    /// VJP for `clamp(x, min, max)`: `out[i] = grad[i]` when `x[i]` is in
    /// `[min, max]`, else `0`. (#524)
    fn clamp_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
        _min_val: f32,
        _max_val: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "clamp_backward_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart. (#524)
    fn clamp_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
        _min_val: f64,
        _max_val: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "clamp_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // SiLU activation: out[i] = x * sigmoid(x)
    fn silu_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "silu_f32 GPU op not yet implemented".into(),
        })
    }
    fn silu_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "silu_f64 GPU op not yet implemented".into(),
        })
    }
    fn silu_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "silu_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn silu_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "silu_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // ELU activation: out[i] = x > 0 ? x : alpha*(exp(x)-1)
    fn elu_f32(&self, _a: &GpuBufferHandle, _alpha: f32) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "elu_f32 GPU op not yet implemented".into(),
        })
    }
    fn elu_f64(&self, _a: &GpuBufferHandle, _alpha: f64) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "elu_f64 GPU op not yet implemented".into(),
        })
    }
    fn elu_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
        _alpha: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "elu_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn elu_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
        _alpha: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "elu_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Mish activation: out[i] = x * tanh(softplus(x))
    fn mish_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mish_f32 GPU op not yet implemented".into(),
        })
    }
    fn mish_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mish_f64 GPU op not yet implemented".into(),
        })
    }
    fn mish_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mish_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn mish_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _input: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "mish_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // LogSoftmax: out[i] = x[i] - log(sum(exp(x))) (row-wise)
    fn log_softmax_f32(
        &self,
        _a: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_softmax_f32 GPU op not yet implemented".into(),
        })
    }
    fn log_softmax_f64(
        &self,
        _a: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_softmax_f64 GPU op not yet implemented".into(),
        })
    }
    // LogSoftmax backward: out[i] = grad[i] - softmax[i] * sum(grad) (row-wise)
    fn log_softmax_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_softmax_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn log_softmax_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_softmax_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Indexing operations
    // index_select_1d: out[i] = input[indices[i]]  (indices stored as f32)
    fn index_select_1d_f32(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn index_select_1d_f64(
        &self,
        _input: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "index_select_1d_f64 GPU op not yet implemented".into(),
        })
    }
    // scatter_add_1d: out = zeros(input_len); for i: out[indices[i]] += grad_output[i]  (atomic)
    fn scatter_add_1d_f32(
        &self,
        grad_output: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn scatter_add_1d_f64(
        &self,
        _grad_output: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _input_len: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "scatter_add_1d_f64 GPU op not yet implemented".into(),
        })
    }
    // index_select_dim: gather slices along an arbitrary axis (N-D).
    //
    // Forward layout (contract):
    //   `input`  has logical shape `[outer, in_dim_size, inner]` after
    //   collapsing the axes before/after `dim`. `indices` is an
    //   `out_dim_size`-long f32 buffer encoding integer offsets into
    //   the `in_dim_size` axis (caller-validated, non-negative,
    //   in-range). The kernel writes
    //     `output[o, i, k] = input[o, indices[i], k]`
    //   for `o in [0, outer), i in [0, out_dim_size), k in [0, inner)`.
    // The output buffer has length `outer * out_dim_size * inner`.
    //
    // This subsumes the 1-D `index_select_1d_*` ops for ndim>=2 with
    // arbitrary `dim`. The 1-D ops are kept for the
    // `IndexSelectBackward` (single-axis 1-D index) call site, which
    // predates this op.
    fn index_select_dim_f32(
        &self,
        input: &GpuBufferHandle,
        indices: &GpuBufferHandle,
        outer: usize,
        in_dim_size: usize,
        out_dim_size: usize,
        inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn index_select_dim_f64(
        &self,
        _input: &GpuBufferHandle,
        _indices: &GpuBufferHandle,
        _outer: usize,
        _in_dim_size: usize,
        _out_dim_size: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "index_select_dim_f64 GPU op not yet implemented".into(),
        })
    }
    // masked_fill: out[i] = mask[i] ? value : input[i]  (mask stored as f32, 1.0/0.0)
    fn masked_fill_f32(
        &self,
        input: &GpuBufferHandle,
        mask: &GpuBufferHandle,
        value: f32,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn masked_fill_f64(
        &self,
        _input: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
        _value: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "masked_fill_f64 GPU op not yet implemented".into(),
        })
    }
    // masked_zero: out[i] = mask[i] ? 0.0 : grad[i]  (backward of masked_fill)
    fn masked_zero_f32(
        &self,
        grad: &GpuBufferHandle,
        mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle>;
    fn masked_zero_f64(
        &self,
        _grad: &GpuBufferHandle,
        _mask: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "masked_zero_f64 GPU op not yet implemented".into(),
        })
    }

    // Elementwise unary/binary f32 (default impls for forward ops)
    fn div_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "div_f32 GPU op not yet implemented".into(),
        })
    }
    fn div_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "div_f64 GPU op not yet implemented".into(),
        })
    }
    fn exp_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "exp_f32 GPU op not yet implemented".into(),
        })
    }
    fn exp_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "exp_f64 GPU op not yet implemented".into(),
        })
    }
    fn log_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_f32 GPU op not yet implemented".into(),
        })
    }
    fn log_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "log_f64 GPU op not yet implemented".into(),
        })
    }
    fn sqrt_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sqrt_f32 GPU op not yet implemented".into(),
        })
    }
    fn sqrt_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sqrt_f64 GPU op not yet implemented".into(),
        })
    }
    fn pow_f32(&self, _a: &GpuBufferHandle, _exponent: f32) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "pow_f32 GPU op not yet implemented".into(),
        })
    }
    fn pow_f64(&self, _a: &GpuBufferHandle, _exponent: f64) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "pow_f64 GPU op not yet implemented".into(),
        })
    }
    fn abs_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "abs_f32 GPU op not yet implemented".into(),
        })
    }
    fn abs_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "abs_f64 GPU op not yet implemented".into(),
        })
    }
    fn sigmoid_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_f32 GPU op not yet implemented".into(),
        })
    }
    fn sigmoid_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_f64 GPU op not yet implemented".into(),
        })
    }
    fn tanh_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "tanh_f32 GPU op not yet implemented".into(),
        })
    }
    fn tanh_f64(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "tanh_f64 GPU op not yet implemented".into(),
        })
    }

    // Sigmoid backward: out[i] = grad[i] * output[i] * (1 - output[i])
    fn sigmoid_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn sigmoid_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sigmoid_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Tanh backward: out[i] = grad[i] * (1 - output[i]^2)
    fn tanh_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "tanh_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn tanh_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "tanh_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Softmax backward: out[i] = output[i] * (grad[i] - dot(grad_row, output_row))
    fn softmax_backward_f32(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn softmax_backward_f64(
        &self,
        _grad: &GpuBufferHandle,
        _output: &GpuBufferHandle,
        _cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "softmax_backward_f64 GPU op not yet implemented".into(),
        })
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
        Err(FerrotorchError::InvalidArgument {
            message: "layernorm_backward_f32 GPU op not yet implemented".into(),
        })
    }
    fn layernorm_backward_f64(
        &self,
        _input: &GpuBufferHandle,
        _grad_output: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _rows: usize,
        _cols: usize,
        _eps: f64,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "layernorm_backward_f64 GPU op not yet implemented".into(),
        })
    }

    // Sum along one axis of a tensor
    fn sum_axis_f32(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sum_axis_f32 GPU op not yet implemented".into(),
        })
    }
    fn sum_axis_f64(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sum_axis_f64 GPU op not yet implemented".into(),
        })
    }

    // Strided split: extract a sub-tensor along one axis entirely on GPU.
    fn strided_split_f32(
        &self,
        _input: &GpuBufferHandle,
        _total_along_axis: usize,
        _split_offset: usize,
        _split_size: usize,
        _inner_size: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_split_f32 GPU op not yet implemented".into(),
        })
    }
    fn strided_split_f64(
        &self,
        _input: &GpuBufferHandle,
        _total_along_axis: usize,
        _split_offset: usize,
        _split_size: usize,
        _inner_size: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_split_f64 GPU op not yet implemented".into(),
        })
    }

    // Strided copy: gather an N-d strided view into a contiguous
    // output buffer entirely on GPU. CL-496.
    fn strided_copy_f32(
        &self,
        _input: &GpuBufferHandle,
        _out_shape: &[usize],
        _src_strides: &[isize],
        _src_offset: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_copy_f32 GPU op not yet implemented".into(),
        })
    }
    fn strided_copy_f64(
        &self,
        _input: &GpuBufferHandle,
        _out_shape: &[usize],
        _src_strides: &[isize],
        _src_offset: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_copy_f64 GPU op not yet implemented".into(),
        })
    }

    // Strided scatter: write a contiguous src into strided positions of
    // dst (in-place). Inverse of strided_copy. Used by
    // `Tensor::as_strided_scatter` for CUDA tensors. (#574)
    fn strided_scatter_f32(
        &self,
        _src: &GpuBufferHandle,
        _dst: &mut GpuBufferHandle,
        _view_shape: &[usize],
        _dst_strides: &[isize],
        _dst_offset: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_scatter_f32 GPU op not yet implemented".into(),
        })
    }
    fn strided_scatter_f64(
        &self,
        _src: &GpuBufferHandle,
        _dst: &mut GpuBufferHandle,
        _view_shape: &[usize],
        _dst_strides: &[isize],
        _dst_offset: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_scatter_f64 GPU op not yet implemented".into(),
        })
    }

    // Strided cat: write a sub-tensor into a larger buffer at an offset along one axis on GPU.
    #[allow(clippy::too_many_arguments)]
    fn strided_cat_f32(
        &self,
        _input: &GpuBufferHandle,
        _output: &mut GpuBufferHandle,
        _total_along_axis: usize,
        _cat_offset: usize,
        _part_size: usize,
        _inner_size: usize,
        _n: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_cat_f32 GPU op not yet implemented".into(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn strided_cat_f64(
        &self,
        _input: &GpuBufferHandle,
        _output: &mut GpuBufferHandle,
        _total_along_axis: usize,
        _cat_offset: usize,
        _part_size: usize,
        _inner_size: usize,
        _n: usize,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "strided_cat_f64 GPU op not yet implemented".into(),
        })
    }

    /// Check if a GPU buffer contains any inf or NaN values.
    ///
    /// Required method (no default impl): backends must provide an
    /// implementation rather than silently fall back to a host-readback
    /// scan. The previous default impl made the host detour invisible at
    /// the call site, hiding a synchronous device-to-host round-trip behind
    /// a trait-method default. Removing it forces the badness to be visible
    /// at each backend's impl site.
    fn has_inf_nan_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<bool>;

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
    fn svd_f32(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "svd_f32 GPU op not yet implemented".into(),
        })
    }
    fn svd_f64(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "svd_f64 GPU op not yet implemented".into(),
        })
    }
    fn cholesky_f32(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cholesky_f32 GPU op not yet implemented".into(),
        })
    }
    fn cholesky_f64(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "cholesky_f64 GPU op not yet implemented".into(),
        })
    }
    fn solve_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
        _nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "solve_f32 GPU op not yet implemented".into(),
        })
    }
    fn solve_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _n: usize,
        _nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "solve_f64 GPU op not yet implemented".into(),
        })
    }
    fn qr_f32(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "qr_f32 GPU op not yet implemented".into(),
        })
    }
    fn qr_f64(
        &self,
        _a: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "qr_f64 GPU op not yet implemented".into(),
        })
    }

    /// LU factorization in cuSOLVER's packed form: returns
    /// `(LU_packed, pivots)` where `LU_packed` is an `n×n` row-major GPU
    /// tensor handle (strict lower = `L`, upper = `U`), and `pivots` is a
    /// host `Vec<i32>` of length `n` (1-based row-permutation indices,
    /// LAPACK convention). The pivot vector is small (O(n)) and inherently
    /// host-readable, so we return it materialized on host rather than
    /// inventing a typed-int GPU handle. Mirrors `torch.linalg.lu_factor`.
    /// (#604)
    fn lu_factor_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<i32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "lu_factor_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart of [`Self::lu_factor_f32`]. (#604)
    fn lu_factor_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, Vec<i32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "lu_factor_f64 GPU op not yet implemented".into(),
        })
    }

    /// GPU-resident least-squares solver via cuSOLVER `cusolverDnSSgels`
    /// (iterative refinement). Solves `min ||A X - B||_F` for `A: m×n`,
    /// `B: m×nrhs`. Returns `X: n×nrhs`. Mirrors `torch.linalg.lstsq`'s
    /// solution output. (#630)
    fn lstsq_f32(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _n: usize,
        _nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "lstsq_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart. (#630)
    fn lstsq_f64(
        &self,
        _a: &GpuBufferHandle,
        _b: &GpuBufferHandle,
        _m: usize,
        _n: usize,
        _nrhs: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "lstsq_f64 GPU op not yet implemented".into(),
        })
    }

    /// Non-symmetric eigendecomposition via cuSOLVER `cusolverDnXgeev`.
    /// Returns `(eigenvalues, eigenvectors)` as **complex** GPU tensors:
    ///   - eigenvalues: length `2n` interleaved re/im (logical `[n, 2]`)
    ///   - eigenvectors: length `2 * n * n` row-major interleaved
    ///     (logical `[n, n, 2]`)
    ///
    /// Mirrors `torch.linalg.eig`. (#631)
    fn eig_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "eig_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart. (#631)
    fn eig_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "eig_f64 GPU op not yet implemented".into(),
        })
    }
    /// Symmetric eigendecomposition (eigenvalues + eigenvectors) of an
    /// `n × n` real symmetric matrix. Returns `(eigenvalues, eigenvectors)`
    /// where eigenvectors is row-major with column `j` the `j`-th eigenvector.
    fn eigh_f32(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "eigh_f32 GPU op not yet implemented".into(),
        })
    }
    fn eigh_f64(
        &self,
        _a: &GpuBufferHandle,
        _n: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "eigh_f64 GPU op not yet implemented".into(),
        })
    }
    /// Eigenvalues only of an `n × n` real symmetric matrix.
    fn eigvalsh_f32(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "eigvalsh_f32 GPU op not yet implemented".into(),
        })
    }
    fn eigvalsh_f64(&self, _a: &GpuBufferHandle, _n: usize) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "eigvalsh_f64 GPU op not yet implemented".into(),
        })
    }

    // GPU 1-D FFT primitives via cuFFT. (#579)
    //
    // - C2C: input/output layout `[batch * n * 2]` interleaved (re, im).
    // - R2C: input `[batch * n]` real → output `[batch * (n/2+1) * 2]` complex.
    // - C2R: input `[batch * (n_out/2+1) * 2]` complex → output `[batch * n_out]` real.
    // - Inverse transforms include 1/n normalization to match torch / numpy.
    fn fft_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fft_c2c_f32 GPU op not yet implemented".into(),
        })
    }
    fn fft_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fft_c2c_f64 GPU op not yet implemented".into(),
        })
    }

    /// GPU pad/truncate for complex tensors stored as `[batch, n, 2]`
    /// (#605). Used by the FFT path when the user passes `n != input_n` —
    /// allocates a `[batch, dst_n, 2]` output, copies the visible portion
    /// from `src`, and zero-fills the tail. Single PTX kernel, no host
    /// bounce.
    fn pad_truncate_complex_f32(
        &self,
        _src: &GpuBufferHandle,
        _batch: usize,
        _src_n: usize,
        _dst_n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "pad_truncate_complex_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart of [`Self::pad_truncate_complex_f32`]. (#605)
    fn pad_truncate_complex_f64(
        &self,
        _src: &GpuBufferHandle,
        _batch: usize,
        _src_n: usize,
        _dst_n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "pad_truncate_complex_f64 GPU op not yet implemented".into(),
        })
    }

    /// 2-D complex-to-complex FFT via cufftPlan2d. Input/output layout
    /// `[h, w, 2]` interleaved complex. `inverse=true` divides by `h*w`.
    /// (#634)
    fn fft2_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fft2_c2c_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 2-D FFT counterpart. (#634)
    fn fft2_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fft2_c2c_f64 GPU op not yet implemented".into(),
        })
    }

    /// Broadcast a `[outer, inner]` tensor into `[outer, repeat_count, inner]`
    /// by replicating along the inserted middle dim. Used for sum_dim /
    /// mean_dim backward where the gradient must be expanded along the
    /// previously-reduced dim. (#524)
    fn repeat_along_dim_f32(
        &self,
        _input: &GpuBufferHandle,
        _outer: usize,
        _repeat_count: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "repeat_along_dim_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 counterpart. (#524)
    fn repeat_along_dim_f64(
        &self,
        _input: &GpuBufferHandle,
        _outer: usize,
        _repeat_count: usize,
        _inner: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "repeat_along_dim_f64 GPU op not yet implemented".into(),
        })
    }
    fn rfft_r2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "rfft_r2c_f32 GPU op not yet implemented".into(),
        })
    }
    fn rfft_r2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "rfft_r2c_f64 GPU op not yet implemented".into(),
        })
    }
    fn irfft_c2r_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "irfft_c2r_f32 GPU op not yet implemented".into(),
        })
    }
    fn irfft_c2r_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "irfft_c2r_f64 GPU op not yet implemented".into(),
        })
    }

    /// Hermitian FFT: `hfft(x, n) = irfft(conj(x), n)` on GPU via cuFFT. (#636)
    ///
    /// Input `[batch, half_in, 2]` complex; output `[batch * n_out]` real.
    /// `half_in` must equal `n_out / 2 + 1`.
    fn hfft_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _half_in: usize,
        _n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "hfft_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 Hermitian FFT counterpart. (#636)
    fn hfft_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _half_in: usize,
        _n_out: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "hfft_f64 GPU op not yet implemented".into(),
        })
    }

    /// Inverse Hermitian FFT: `ihfft(x) = conj(rfft(x)) / n` on GPU. (#636)
    ///
    /// Input `[batch * n]` real; output `[batch, n/2+1, 2]` complex.
    fn ihfft_f32(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "ihfft_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 inverse Hermitian FFT counterpart. (#636)
    fn ihfft_f64(
        &self,
        _a: &GpuBufferHandle,
        _batch: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "ihfft_f64 GPU op not yet implemented".into(),
        })
    }

    /// 3-D complex-to-complex FFT via `cufftPlan3d`. (#636)
    ///
    /// Input/output layout `[d, h, w, 2]` interleaved complex.
    /// `inverse=true` divides by `d*h*w`.
    fn fftn3d_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _d: usize,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn3d_c2c_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 3-D FFT counterpart. (#636)
    fn fftn3d_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _d: usize,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn3d_c2c_f64 GPU op not yet implemented".into(),
        })
    }

    /// 2-D complex-to-complex FFT via `cufftPlanMany` for f32. (#636)
    ///
    /// Input/output layout `[h, w, 2]` interleaved complex.
    /// `inverse=true` divides by `h*w`.
    fn fftn2d_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn2d_c2c_f32 GPU op not yet implemented".into(),
        })
    }

    /// f64 2-D FFT counterpart. (#636)
    fn fftn2d_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _h: usize,
        _w: usize,
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn2d_c2c_f64 GPU op not yet implemented".into(),
        })
    }

    // -- axes-aware N-D FFT via cufftPlanMany (#966) -------------------------

    /// Axes-aware N-D complex-to-complex FFT for f32 via `cufftPlanMany`. (#966)
    ///
    /// Transforms over the specified `axes` (normalized to non-negative in
    /// `[0, ndim)`) of the complex input tensor. Input layout: interleaved
    /// `[..., 2]` (re/im pairs); the trailing complex dim is always the last
    /// and is NOT included in `axes`. `shape` is the tensor's spatial dims
    /// (excluding the trailing 2).
    ///
    /// `inverse=true` applies `1 / product(shape[ax] for ax in axes)`
    /// normalization to match `torch.fft.ifftn`.
    ///
    /// Default impl returns `Err` so existing backends compile unchanged.
    fn fftn_axes_c2c_f32(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axes: &[usize],
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn_axes_c2c_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// f64 variant of [`Self::fftn_axes_c2c_f32`]. (#966)
    fn fftn_axes_c2c_f64(
        &self,
        _a: &GpuBufferHandle,
        _shape: &[usize],
        _axes: &[usize],
        _inverse: bool,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "fftn_axes_c2c_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// Fused Adam optimizer step: updates param, exp_avg, and exp_avg_sq
    /// in a single kernel launch.
    ///
    /// All four buffers (`param`, `grad`, `exp_avg`, `exp_avg_sq`) must have
    /// the same length. `param`, `exp_avg`, and `exp_avg_sq` are modified
    /// in-place.
    #[allow(clippy::too_many_arguments)]
    fn fused_adam_f32(
        &self,
        _param: &mut GpuBufferHandle,
        _grad: &GpuBufferHandle,
        _exp_avg: &mut GpuBufferHandle,
        _exp_avg_sq: &mut GpuBufferHandle,
        _beta1: f32,
        _beta2: f32,
        _lr: f32,
        _eps: f32,
        _bc1: f32,
        _bc2: f32,
        _weight_decay: f32,
    ) -> FerrotorchResult<()> {
        Err(FerrotorchError::InvalidArgument {
            message: "fused_adam_f32 GPU op not yet implemented".into(),
        })
    }

    /// Fused GRU cell forward: pointwise gate computation on pre-computed
    /// gate matrices. Returns `(hy_handle, workspace_handle)`.
    ///
    /// `input_gates` and `hidden_gates` are `[batch, 3*hsz]` from cuBLAS GEMMs.
    /// `bias_ih` and `bias_hh` are `[3*hsz]`. `hx` is `[batch, hsz]`.
    /// `workspace` is `[batch, 5*hsz]` saved for backward.
    fn fused_gru_cell_f32(
        &self,
        _input_gates: &GpuBufferHandle,
        _hidden_gates: &GpuBufferHandle,
        _bias_ih: &GpuBufferHandle,
        _bias_hh: &GpuBufferHandle,
        _hx: &GpuBufferHandle,
        _hidden_size: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, GpuBufferHandle)> {
        Err(FerrotorchError::InvalidArgument {
            message: "fused_gru_cell_f32 GPU op not yet implemented".into(),
        })
    }

    /// GPU MaxPool2d forward.
    #[allow(clippy::too_many_arguments)]
    fn maxpool2d_f32(
        &self,
        _input: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _h_in: usize,
        _w_in: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _ph: usize,
        _pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "maxpool2d_f32 GPU op not yet implemented".into(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn maxpool2d_f64(
        &self,
        _input: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _h_in: usize,
        _w_in: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _ph: usize,
        _pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "maxpool2d_f64 GPU op not yet implemented".into(),
        })
    }

    /// GPU AvgPool2d forward.
    #[allow(clippy::too_many_arguments)]
    fn avgpool2d_f32(
        &self,
        _input: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _h_in: usize,
        _w_in: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _ph: usize,
        _pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "avgpool2d_f32 GPU op not yet implemented".into(),
        })
    }
    #[allow(clippy::too_many_arguments)]
    fn avgpool2d_f64(
        &self,
        _input: &GpuBufferHandle,
        _batch: usize,
        _channels: usize,
        _h_in: usize,
        _w_in: usize,
        _kh: usize,
        _kw: usize,
        _sh: usize,
        _sw: usize,
        _ph: usize,
        _pw: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "avgpool2d_f64 GPU op not yet implemented".into(),
        })
    }

    /// GPU Conv2d forward: im2col + GEMM + bias add, entirely on-device.
    ///
    /// Supports the full `Conv2d::new_full` parameter surface: `groups`
    /// partitions input/output channels and `dilation` spaces the kernel
    /// taps. The dispatch happens on the GPU for every value of these
    /// parameters; there is no CPU detour. Pass `groups = 1` and
    /// `dilation = (1, 1)` for the dense convolution case.
    ///
    /// Returns `(output_handle, output_shape)` where output_shape is `[B, C_out, H_out, W_out]`.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_f32(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: Option<&GpuBufferHandle>,
        _input_shape: [usize; 4],
        _weight_shape: [usize; 4],
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
        _groups: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "conv2d_f32 GPU op not yet implemented".into(),
        })
    }
    /// GPU Conv2d forward (f64). See [`Self::conv2d_f32`] for parameter
    /// semantics — this is the f64 companion.
    #[allow(clippy::too_many_arguments)]
    fn conv2d_f64(
        &self,
        _input: &GpuBufferHandle,
        _weight: &GpuBufferHandle,
        _bias: Option<&GpuBufferHandle>,
        _input_shape: [usize; 4],
        _weight_shape: [usize; 4],
        _stride: (usize, usize),
        _padding: (usize, usize),
        _dilation: (usize, usize),
        _groups: usize,
    ) -> FerrotorchResult<(GpuBufferHandle, [usize; 4])> {
        Err(FerrotorchError::InvalidArgument {
            message: "conv2d_f64 GPU op not yet implemented".into(),
        })
    }

    // -- Sparse SpMM (cuSPARSE, CSR format) -----------------------------------
    //
    // These cover `SparseTensor::spmm` when the dense operand is a CUDA
    // tensor — PyTorch's `torch.sparse.mm` runs on cuSPARSE in that case.
    // Just-in-time CSR upload from the caller's host-side `(crow_indices,
    // col_indices, values)`; the dense operand is already device-resident.
    // Output `[m, n]` row-major lives on the same device as the dense input.
    //
    // See ferrotorch-core/src/sparse.rs `SparseTensor::spmm` for the call
    // site and ferrotorch-gpu/src/sparse.rs for the cuSPARSE implementation.

    /// CSR sparse-dense matmul on GPU (f32 dtype).
    ///
    /// `crow_indices`: `m + 1` host `u32` row pointers in CSR order.
    /// `col_indices`: `nnz` host `u32` column indices.
    /// `values`: `nnz` host f32 non-zero values.
    /// `dense`: device buffer holding a `[k, n]` row-major dense matrix.
    /// Returns a device buffer holding the `[m, n]` row-major dense result.
    fn spmm_csr_f32(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _dense: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "spmm_csr_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR sparse-dense matmul on GPU (f64 dtype). Companion of
    /// [`Self::spmm_csr_f32`].
    fn spmm_csr_f64(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _dense: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "spmm_csr_f64 GPU op not implemented for this backend".into(),
        })
    }

    // -- Sparse <-> Dense conversion (cuSPARSE) -------------------------------
    //
    // P3 covers `SparseTensor::to_dense_on(Device::Cuda)` and the GPU branch
    // of `SparseTensor::from_dense` when the dense tensor lives on CUDA.
    // PyTorch parity (rust-gpu-discipline §3): `torch.Tensor.to_dense()` and
    // `torch.Tensor.to_sparse()` keep the result on the input device and
    // dispatch to cuSPARSE on CUDA. We mirror that.
    //
    // These are CSR-shaped: ferrotorch's internal SpMM path is CSR, and
    // cuSPARSE's `*_to_dense`/`*_to_sparse` accept CSR descriptors. The COO
    // → CSR build (host-side row-pointer prefix sum) reuses the same code
    // path as `SparseTensor::spmm`.
    //
    // See ferrotorch-gpu/src/sparse.rs for the implementation.

    /// CSR-form sparse → dense materialization on GPU (f32 dtype).
    ///
    /// Inputs:
    /// - `crow_indices`: `m + 1` host `u32` row pointers.
    /// - `col_indices`: `nnz` host `u32` column indices.
    /// - `values`: `nnz` host f32 non-zero values.
    /// - `device_ordinal`: target CUDA ordinal; output buffer lives there.
    /// - `m`, `n`: output dense shape `[m, n]`, row-major.
    ///
    /// Returns a device buffer holding the `[m, n]` row-major dense result.
    fn sparse_to_dense_csr_f32(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_to_dense_csr_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR-form sparse → dense materialization on GPU (f64 dtype).
    /// Companion of [`Self::sparse_to_dense_csr_f32`].
    fn sparse_to_dense_csr_f64(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_to_dense_csr_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// Dense → CSR-form sparse extraction on GPU (f32 dtype).
    ///
    /// Reads a row-major `[m, n]` device dense matrix and returns the CSR
    /// triplet `(crow_indices, col_indices, values)` with **only exact-zero**
    /// entries dropped (PyTorch's `torch.Tensor.to_sparse()` semantics — non-
    /// zero thresholds must be applied by the caller).
    ///
    /// Returns host-side `Vec`s — the caller decides whether to coalesce or
    /// store on device. ferrotorch's `SparseTensor` is CPU-resident so the
    /// host-side return matches that storage model.
    fn dense_to_sparse_csr_f32(
        &self,
        _dense: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "dense_to_sparse_csr_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// Dense → CSR-form sparse extraction on GPU (f64 dtype).
    /// Companion of [`Self::dense_to_sparse_csr_f32`].
    fn dense_to_sparse_csr_f64(
        &self,
        _dense: &GpuBufferHandle,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "dense_to_sparse_csr_f64 GPU op not implemented for this backend".into(),
        })
    }

    // -- CSR/CSC/COO format-conversion + to_dense (cuSPARSE) -- P7 ------------
    //
    // PyTorch parity (rust-gpu-discipline §3): `torch.sparse_csr_tensor` /
    // `torch.sparse_csc_tensor` / `torch.sparse_coo_tensor` keep the result on
    // the input device, and the format-conversion helpers (`.to_sparse_csr()`,
    // `.to_sparse_csc()`, `.to_dense()` on a CSR/CSC/COO tensor) run on
    // cuSPARSE when the data lives on CUDA. ferrotorch routes CSR↔CSC via
    // `cusparseCsr2cscEx2` and COO↔CSR via `cusparseXcoo2csr` /
    // `cusparseXcsr2coo` (host inputs uploaded JIT, dense output stays on
    // device).
    //
    // The CSR-shaped sparse-to-dense path already exists as
    // `sparse_to_dense_csr_f{32,64}` (P3); the CSC variants below are dual
    // paths that take a CSC triplet and materialise the dense matrix on
    // device. Per-component dispatch from `CscTensor::to_dense_on` and
    // `CooTensor::to_dense_on`.
    //
    // See ferrotorch-gpu/src/sparse.rs for the cuSPARSE implementations.

    /// CSC-form sparse → dense materialization on GPU (f32 dtype).
    ///
    /// Inputs:
    /// - `col_ptrs`: `n + 1` host `u32` column pointers in CSC order.
    /// - `row_indices`: `nnz` host `u32` row indices.
    /// - `values`: `nnz` host f32 non-zero values.
    /// - `device_ordinal`: target CUDA ordinal; output buffer lives there.
    /// - `m`, `n`: output dense shape `[m, n]`, row-major.
    ///
    /// Returns a device buffer holding the `[m, n]` row-major dense result.
    fn csc_to_dense_f32(
        &self,
        _col_ptrs: &[u32],
        _row_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "csc_to_dense_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSC-form sparse → dense materialization on GPU (f64 dtype).
    /// Companion of [`Self::csc_to_dense_f32`].
    fn csc_to_dense_f64(
        &self,
        _col_ptrs: &[u32],
        _row_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "csc_to_dense_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR → CSC format conversion on GPU (f32 dtype).
    ///
    /// Uses `cusparseCsr2cscEx2`. Returns the CSC triplet
    /// `(col_ptrs, row_indices, values)` in host buffers — ferrotorch's
    /// `CscTensor` is CPU-resident so the host return matches that storage
    /// model. `m`, `n` are the source CSR shape.
    fn csr_to_csc_f32(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "csr_to_csc_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR → CSC format conversion on GPU (f64 dtype).
    /// Companion of [`Self::csr_to_csc_f32`].
    fn csr_to_csc_f64(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "csr_to_csc_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// COO → CSR format conversion on GPU (f32 dtype).
    ///
    /// Wraps `cusparseXcoo2csr`. Caller supplies row-sorted COO (cuSPARSE
    /// requires it); ferrotorch's `CooTensor` is host-resident so the
    /// caller pre-sorts on the host before invoking. Values are passed
    /// through unchanged (only the row indices are compacted into a
    /// `crow_indices` row-pointer array).
    ///
    /// Returns the host CSR triplet `(crow_indices, col_indices, values)`.
    fn coo_to_csr_f32(
        &self,
        _row_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "coo_to_csr_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// COO → CSR format conversion on GPU (f64 dtype).
    /// Companion of [`Self::coo_to_csr_f32`].
    fn coo_to_csr_f64(
        &self,
        _row_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "coo_to_csr_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR → COO row-index expansion on GPU (f32 dtype).
    ///
    /// Wraps `cusparseXcsr2coo`. Returns the host COO triplet
    /// `(row_indices, col_indices, values)`. `col_indices` and `values`
    /// pass through unchanged from the source CSR; only the `crow_indices`
    /// row-pointer array is expanded into per-entry `row_indices`.
    fn csr_to_coo_f32(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f32],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f32>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "csr_to_coo_f32 GPU op not implemented for this backend".into(),
        })
    }

    /// CSR → COO row-index expansion on GPU (f64 dtype).
    /// Companion of [`Self::csr_to_coo_f32`].
    fn csr_to_coo_f64(
        &self,
        _crow_indices: &[u32],
        _col_indices: &[u32],
        _values: &[f64],
        _device_ordinal: usize,
        _m: usize,
        _n: usize,
    ) -> FerrotorchResult<(Vec<u32>, Vec<u32>, Vec<f64>)> {
        Err(FerrotorchError::InvalidArgument {
            message: "csr_to_coo_f64 GPU op not implemented for this backend".into(),
        })
    }

    /// Synchronize the current stream on the given device, blocking until
    /// all enqueued operations have completed.
    fn synchronize(&self, _device: usize) -> FerrotorchResult<()> {
        Err(FerrotorchError::DeviceUnavailable)
    }

    /// Return the number of streams in the pool for the given device.
    fn stream_count(&self, _device: usize) -> usize {
        1
    }

    // ---------------------------------------------------------------------
    // FlashAttention forward (P5).
    //
    // Per-component dispatch from `nested_scaled_dot_product_attention`.
    // Computes `softmax(Q @ K^T * scale) @ V` with on-device tiled
    // online-softmax (FlashAttention-2 forward) without materialising the
    // full [seq_q, seq_k] scores matrix.
    //
    // Default impls return `InvalidArgument` so backends without CUDA
    // declare unsupported, and the caller falls through to the GPU
    // composite path (`bmm + softmax_rows + bmm`).
    // ---------------------------------------------------------------------

    /// FlashAttention forward (f32). `query`/`key` are `[seq_q, d]` /
    /// `[seq_k, d]`; `value` is `[seq_k, d_v]`. Returns `[seq_q, d_v]`.
    /// `scale` is typically `1 / sqrt(d)`. Single-head (no batch dim) —
    /// the caller folds batch into per-component dispatch.
    fn flash_attention_forward_f32(
        &self,
        _query: &GpuBufferHandle,
        _key: &GpuBufferHandle,
        _value: &GpuBufferHandle,
        _seq_q: usize,
        _seq_k: usize,
        _d: usize,
        _d_v: usize,
        _scale: f32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "flash_attention_forward_f32 GPU op not yet implemented".into(),
        })
    }

    /// FlashAttention forward (f64). See [`flash_attention_forward_f32`].
    fn flash_attention_forward_f64(
        &self,
        _query: &GpuBufferHandle,
        _key: &GpuBufferHandle,
        _value: &GpuBufferHandle,
        _seq_q: usize,
        _seq_k: usize,
        _d: usize,
        _d_v: usize,
        _scale: f64,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "flash_attention_forward_f64 GPU op not yet implemented".into(),
        })
    }

    // ---------------------------------------------------------------------
    // P6: 2:4 structured sparse matmul (cuSPARSELt).
    //
    // PyTorch parity (rust-gpu-discipline §3): `torch._C._sparse_semi_
    // structured_apply` (and the `SparseSemiStructuredTensor` user API)
    // dispatches to NVIDIA cuSPARSELt on Ampere+ Tensor Cores when the
    // 2:4 sparse weight is on CUDA. ferrotorch mirrors that by routing
    // `SemiStructuredSparseTensor::sparse_matmul_24` through these
    // hooks.
    //
    // The dense `b_dense_decompressed` operand is the dense
    // representation of the structured 2:4 matrix (mask applied → zeros
    // in non-retained positions). cuSPARSELt repacks it into the
    // Tensor-Core-friendly layout via `cusparseLtSpMMACompress`
    // internally.
    //
    // Default impls return `InvalidArgument` so backends without the
    // `cusparselt` cargo feature declare unsupported and the caller
    // falls through to the existing decompress + dense matmul reference
    // path. This is the §3-correct opt-in mechanism — no silent CPU
    // fallback. See `ferrotorch-gpu/src/cusparselt.rs` for the live
    // implementation under the feature.
    // ---------------------------------------------------------------------

    /// 2:4 structured sparse matmul, FP32 (TF32-Tensor-Core compute).
    /// Computes `[m, n] = a @ b_dense_decompressed` where `b` carries
    /// 2:4 structured zeros along its inner dimension.
    fn sparse_matmul_24_f32(
        &self,
        _a: &GpuBufferHandle,
        _b_dense_decompressed: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_matmul_24_f32 GPU op not implemented for this backend (build with --features cusparselt)".into(),
        })
    }

    /// 2:4 structured sparse matmul, FP16 (FP32 accumulator).
    /// f16 inputs/outputs are passed as raw `u16` buffers (bf16/f16 in
    /// ferrotorch are `u16`-bit-pattern carriers).
    fn sparse_matmul_24_f16(
        &self,
        _a: &GpuBufferHandle,
        _b_dense_decompressed: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_matmul_24_f16 GPU op not implemented for this backend (build with --features cusparselt)".into(),
        })
    }

    /// 2:4 structured sparse matmul, BF16 (FP32 accumulator).
    /// See [`Self::sparse_matmul_24_f16`].
    fn sparse_matmul_24_bf16(
        &self,
        _a: &GpuBufferHandle,
        _b_dense_decompressed: &GpuBufferHandle,
        _m: usize,
        _k: usize,
        _n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "sparse_matmul_24_bf16 GPU op not implemented for this backend (build with --features cusparselt)".into(),
        })
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
