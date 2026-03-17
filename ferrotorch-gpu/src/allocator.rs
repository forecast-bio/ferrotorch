//! Caching CUDA memory allocator.
//!
//! [`CudaAllocator`] wraps a [`GpuDevice`] and tracks allocation statistics
//! (bytes currently live on the GPU, peak usage). Every allocation goes
//! through the allocator so the rest of the framework can query memory
//! pressure without talking to the driver.
//!
//! # Future work
//!
//! The current implementation delegates directly to cudarc for every
//! alloc/free. A true caching layer (power-of-two binning with block reuse)
//! is deferred until cudarc's `CudaSlice` ownership model is better
//! understood — the statistics-tracking API is stable and usable today.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::GpuResult;

/// A statistics-tracking GPU memory allocator.
///
/// Wrap a [`GpuDevice`] in a `CudaAllocator` to gain memory accounting:
///
/// - [`memory_allocated`](CudaAllocator::memory_allocated) — bytes currently
///   live on the device.
/// - [`max_memory_allocated`](CudaAllocator::max_memory_allocated) — high-water
///   mark since the allocator was created (or since the last
///   [`reset_peak_stats`](CudaAllocator::reset_peak_stats)).
///
/// # Thread safety
///
/// `CudaAllocator` is `Send + Sync`. The atomic counters are updated with
/// `Relaxed` ordering — they are advisory statistics, not synchronization
/// primitives.
pub struct CudaAllocator {
    device: Arc<GpuDevice>,
    /// Total bytes currently live on the device (not yet freed through us).
    allocated_bytes: AtomicUsize,
    /// Peak allocated bytes since creation or last reset.
    peak_bytes: AtomicUsize,
}

impl CudaAllocator {
    /// Create a new allocator for the given device.
    pub fn new(device: Arc<GpuDevice>) -> Self {
        Self {
            device,
            allocated_bytes: AtomicUsize::new(0),
            peak_bytes: AtomicUsize::new(0),
        }
    }

    /// Allocate `count` zero-initialized elements of type `T` on the device.
    ///
    /// The returned [`CudaBuffer`] is tracked by this allocator. When you are
    /// done with it, pass it to [`free`](CudaAllocator::free) so the
    /// statistics stay accurate. (Dropping the buffer directly still frees
    /// GPU memory, but the `allocated_bytes` counter will be too high.)
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Driver`] if the underlying CUDA allocation fails.
    #[cfg(feature = "cuda")]
    pub fn alloc_zeros<T>(&self, count: usize) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let bytes = count.checked_mul(std::mem::size_of::<T>()).unwrap_or(usize::MAX);
        let slice = self.device.stream().alloc_zeros::<T>(count)?;

        // Update statistics after the allocation succeeds.
        let prev = self.allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.peak_bytes.fetch_max(prev + bytes, Ordering::Relaxed);

        Ok(CudaBuffer {
            data: Some(slice),
            len: count,
            device_ordinal: self.device.ordinal(),
            pool_fn: None,
        })
    }

    /// Copy a host slice to device memory, tracking the allocation.
    ///
    /// This is the allocator-aware equivalent of [`crate::transfer::cpu_to_gpu`].
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Driver`] if the CUDA memcpy or allocation fails.
    #[cfg(feature = "cuda")]
    pub fn alloc_copy<T>(&self, data: &[T]) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        let bytes = data.len().checked_mul(std::mem::size_of::<T>()).unwrap_or(usize::MAX);
        let slice = self.device.stream().clone_htod(data)?;

        let prev = self.allocated_bytes.fetch_add(bytes, Ordering::Relaxed);
        self.peak_bytes.fetch_max(prev + bytes, Ordering::Relaxed);

        Ok(CudaBuffer {
            data: Some(slice),
            len: data.len(),
            device_ordinal: self.device.ordinal(),
            pool_fn: None,
        })
    }

    /// Return a buffer to the allocator, freeing the GPU memory and updating
    /// the statistics.
    ///
    /// This is preferred over simply dropping the buffer so that
    /// [`memory_allocated`](CudaAllocator::memory_allocated) stays accurate.
    pub fn free<T>(&self, buffer: CudaBuffer<T>) {
        let bytes = buffer.len().checked_mul(std::mem::size_of::<T>()).unwrap_or(0);
        self.allocated_bytes.fetch_sub(bytes, Ordering::Relaxed);
        drop(buffer);
    }

    // ------------------------------------------------------------------
    // Statistics queries
    // ------------------------------------------------------------------

    /// Bytes currently allocated (live) on the device through this allocator.
    #[inline]
    pub fn memory_allocated(&self) -> usize {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Peak bytes ever allocated since creation or the last
    /// [`reset_peak_stats`](CudaAllocator::reset_peak_stats).
    #[inline]
    pub fn max_memory_allocated(&self) -> usize {
        self.peak_bytes.load(Ordering::Relaxed)
    }

    /// Reset the peak counter to the current allocation level.
    pub fn reset_peak_stats(&self) {
        let current = self.allocated_bytes.load(Ordering::Relaxed);
        self.peak_bytes.store(current, Ordering::Relaxed);
    }

    /// Drop all cached blocks.
    ///
    /// Currently a no-op because the allocator does not yet maintain a free
    /// cache. Provided for API completeness — callers can start using it
    /// today and it will become effective once block caching is implemented.
    pub fn empty_cache(&self) {
        // No-op: no cached blocks yet.
    }

    /// The underlying device.
    #[inline]
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }
}

impl std::fmt::Debug for CudaAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaAllocator")
            .field("device_ordinal", &self.device.ordinal())
            .field("allocated_bytes", &self.allocated_bytes.load(Ordering::Relaxed))
            .field("peak_bytes", &self.peak_bytes.load(Ordering::Relaxed))
            .finish()
    }
}

// ---------------------------------------------------------------------------
// Stub when `cuda` feature is disabled
// ---------------------------------------------------------------------------

#[cfg(not(feature = "cuda"))]
impl CudaAllocator {
    /// Stub — allocates nothing, only updates statistics.
    pub fn alloc_zeros<T>(&self, _count: usize) -> GpuResult<CudaBuffer<T>> {
        Err(crate::error::GpuError::NoCudaFeature)
    }

    /// Stub — allocates nothing, only updates statistics.
    pub fn alloc_copy<T>(&self, _data: &[T]) -> GpuResult<CudaBuffer<T>> {
        Err(crate::error::GpuError::NoCudaFeature)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    /// Helper: build a `CudaAllocator` around device 0.
    ///
    /// Panics if no CUDA device is available.
    fn make_allocator() -> CudaAllocator {
        let device = GpuDevice::new(0).expect("CUDA device 0");
        CudaAllocator::new(Arc::new(device))
    }

    #[test]
    fn new_allocator_starts_at_zero() {
        let alloc = make_allocator();
        assert_eq!(alloc.memory_allocated(), 0);
        assert_eq!(alloc.max_memory_allocated(), 0);
    }

    #[test]
    fn empty_cache_is_harmless() {
        let alloc = make_allocator();
        alloc.empty_cache(); // should not panic
    }

    #[test]
    fn debug_impl() {
        let alloc = make_allocator();
        let s = format!("{alloc:?}");
        assert!(s.contains("CudaAllocator"));
        assert!(s.contains("allocated_bytes"));
    }

    #[test]
    fn alloc_increases_allocated_bytes() {
        let alloc = make_allocator();
        let buf = alloc.alloc_zeros::<f32>(256).expect("alloc_zeros");
        assert_eq!(alloc.memory_allocated(), 256 * std::mem::size_of::<f32>());
        assert_eq!(alloc.max_memory_allocated(), 256 * std::mem::size_of::<f32>());
        alloc.free(buf);
    }

    #[test]
    fn free_decreases_allocated_bytes() {
        let alloc = make_allocator();
        let buf = alloc.alloc_zeros::<f32>(128).expect("alloc_zeros");
        let expected = 128 * std::mem::size_of::<f32>();
        assert_eq!(alloc.memory_allocated(), expected);

        alloc.free(buf);
        assert_eq!(alloc.memory_allocated(), 0);
    }

    #[test]
    fn peak_tracks_maximum() {
        let alloc = make_allocator();

        let buf1 = alloc.alloc_zeros::<f32>(100).expect("alloc 1");
        let buf2 = alloc.alloc_zeros::<f32>(200).expect("alloc 2");
        let peak_after_two = alloc.max_memory_allocated();

        // Free the first buffer — current goes down, peak stays.
        alloc.free(buf1);
        assert_eq!(alloc.max_memory_allocated(), peak_after_two);
        assert!(alloc.memory_allocated() < peak_after_two);

        alloc.free(buf2);
        assert_eq!(alloc.memory_allocated(), 0);
        assert_eq!(alloc.max_memory_allocated(), peak_after_two);
    }

    #[test]
    fn reset_peak_stats_lowers_peak() {
        let alloc = make_allocator();

        let buf = alloc.alloc_zeros::<f32>(512).expect("alloc");
        let high = alloc.max_memory_allocated();
        alloc.free(buf);

        // Peak is still high.
        assert_eq!(alloc.max_memory_allocated(), high);

        // Reset brings it to current (which is 0).
        alloc.reset_peak_stats();
        assert_eq!(alloc.max_memory_allocated(), 0);
    }

    #[test]
    fn alloc_copy_tracks_bytes() {
        let alloc = make_allocator();
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0];
        let buf = alloc.alloc_copy(&data).expect("alloc_copy");
        assert_eq!(alloc.memory_allocated(), 4 * std::mem::size_of::<f64>());
        alloc.free(buf);
        assert_eq!(alloc.memory_allocated(), 0);
    }

    #[test]
    fn zero_element_alloc() {
        let alloc = make_allocator();
        let buf = alloc.alloc_zeros::<f32>(0).expect("alloc_zeros empty");
        assert_eq!(alloc.memory_allocated(), 0);
        assert_eq!(buf.len(), 0);
        assert!(buf.is_empty());
        alloc.free(buf);
        assert_eq!(alloc.memory_allocated(), 0);
    }
}
