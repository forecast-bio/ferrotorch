//! GPU memory buffer with pool-aware Drop.
//!
//! [`CudaBuffer`] owns a region of device memory via `cudarc::driver::CudaSlice`
//! and tracks its length and originating device ordinal. When dropped, pooled
//! buffers are returned to the global GPU memory pool for reuse instead of
//! being freed back to the CUDA driver.

#[cfg(feature = "cuda")]
use cudarc::driver::CudaSlice;

/// Type-erased function pointer that returns a `CudaSlice<T>` to the pool.
/// Stored as `Option` — `None` means "don't pool, just drop normally."
#[cfg(feature = "cuda")]
type PoolReturnFn<T> = Option<fn(usize, usize, CudaSlice<T>)>;

/// Return a `CudaSlice<f32>` to the global pool.
///
/// `alloc_len` is the actual allocated element count (rounded), not the
/// logical length the user requested. This ensures the pool key matches
/// the true CudaSlice capacity.
#[cfg(feature = "cuda")]
fn return_f32(device: usize, alloc_len: usize, slice: CudaSlice<f32>) {
    crate::pool::pool_return::<CudaSlice<f32>>(device, alloc_len, 4, slice);
}

/// Return a `CudaSlice<f64>` to the global pool.
#[cfg(feature = "cuda")]
fn return_f64(device: usize, alloc_len: usize, slice: CudaSlice<f64>) {
    crate::pool::pool_return::<CudaSlice<f64>>(device, alloc_len, 8, slice);
}

/// Owned GPU memory buffer holding `len` elements of type `T`.
///
/// When `pool_fn` is `Some`, dropping returns the inner `CudaSlice` to the
/// global pool ([`crate::pool`]) instead of freeing GPU memory.
///
/// `alloc_len` tracks the actual number of elements allocated in the
/// underlying `CudaSlice`, which may be larger than `len` due to size
/// rounding or block reuse. This value is passed to `pool_return` on drop
/// so the pool can correctly key the returned buffer.
#[cfg(feature = "cuda")]
pub struct CudaBuffer<T> {
    /// The underlying CUDA device memory. Wrapped in `Option` so
    /// `Drop` can `take()` it without double-free.
    pub(crate) data: Option<CudaSlice<T>>,
    /// Logical element count requested by the caller.
    pub(crate) len: usize,
    /// Actual element count allocated in the CudaSlice (>= len, rounded).
    pub(crate) alloc_len: usize,
    pub(crate) device_ordinal: usize,
    /// If `Some`, this function is called in Drop to return the slice
    /// to the pool. If `None`, CudaSlice::Drop frees normally.
    pub(crate) pool_fn: PoolReturnFn<T>,
}

/// Helper to create a pooled f32 buffer.
#[cfg(feature = "cuda")]
impl CudaBuffer<f32> {
    /// Create a pooled f32 buffer that returns to the global pool on drop.
    ///
    /// `len` is the logical element count (what the caller requested).
    /// `alloc_len` is the actual element count in the CudaSlice (may be
    /// larger due to rounding or block reuse).
    pub(crate) fn new_pooled(
        slice: CudaSlice<f32>,
        len: usize,
        alloc_len: usize,
        device: usize,
    ) -> Self {
        Self {
            data: Some(slice),
            len,
            alloc_len,
            device_ordinal: device,
            pool_fn: Some(return_f32),
        }
    }
}

#[cfg(feature = "cuda")]
impl CudaBuffer<f64> {
    /// Create a pooled f64 buffer that returns to the global pool on drop.
    pub(crate) fn new_pooled(
        slice: CudaSlice<f64>,
        len: usize,
        alloc_len: usize,
        device: usize,
    ) -> Self {
        Self {
            data: Some(slice),
            len,
            alloc_len,
            device_ordinal: device,
            pool_fn: Some(return_f64),
        }
    }
}

#[cfg(feature = "cuda")]
impl<T> Drop for CudaBuffer<T> {
    fn drop(&mut self) {
        if let Some(slice) = self.data.take() {
            if let Some(return_fn) = self.pool_fn {
                // Pass alloc_len (the actual CudaSlice capacity) so the pool
                // can correctly key and track this buffer's true size.
                return_fn(self.device_ordinal, self.alloc_len, slice);
            }
            // else: CudaSlice::Drop fires naturally (cuMemFreeAsync)
        }
    }
}

#[cfg(feature = "cuda")]
impl<T> CudaBuffer<T> {
    /// Number of elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The ordinal of the device that owns this memory.
    #[inline]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }

    /// Borrow the underlying `CudaSlice` for use with cudarc APIs.
    #[inline]
    pub fn inner(&self) -> &CudaSlice<T> {
        self.data.as_ref().expect("CudaBuffer: inner slice already taken")
    }

    /// Mutably borrow the underlying `CudaSlice`.
    #[inline]
    pub fn inner_mut(&mut self) -> &mut CudaSlice<T> {
        self.data.as_mut().expect("CudaBuffer: inner slice already taken")
    }
}

#[cfg(feature = "cuda")]
impl<T> std::fmt::Debug for CudaBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaBuffer")
            .field("len", &self.len)
            .field("device_ordinal", &self.device_ordinal)
            .field("pooled", &self.pool_fn.is_some())
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// Stub when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub `CudaBuffer` when the `cuda` feature is not enabled.
#[cfg(not(feature = "cuda"))]
#[derive(Debug)]
pub struct CudaBuffer<T> {
    pub(crate) _phantom: std::marker::PhantomData<T>,
    pub(crate) len: usize,
    pub(crate) device_ordinal: usize,
}

#[cfg(not(feature = "cuda"))]
impl<T> CudaBuffer<T> {
    /// Number of elements in this buffer.
    #[inline]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Whether the buffer is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The ordinal of the device that owns this memory.
    #[inline]
    pub fn device_ordinal(&self) -> usize {
        self.device_ordinal
    }
}
