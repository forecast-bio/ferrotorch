//! GPU buffer pool — caching allocator for CUDA memory.
//!
//! Eliminates per-operation `cuMemAllocAsync` + `cuMemFreeAsync` +
//! `cuEventCreate` × 2 + `cuEventDestroy` × 2 by reusing freed buffers.
//! On a pool hit, the only CUDA call is `cuMemsetD8Async` (to zero the
//! buffer). The `CudaSlice`'s events are kept alive across reuses, so
//! no event creation or destruction is needed.
//!
//! This module provides the `CudaSlice`-holding layer that sits on top of the
//! block-metadata caching allocator in [`crate::allocator`]. The allocator
//! manages block splitting, coalescing, and stream tracking; this module
//! manages the actual type-erased `CudaSlice<T>` ownership.
//!
//! # Thread safety
//!
//! The pool is protected by a `Mutex`. The critical section is a `HashMap`
//! lookup + `Vec::pop` (microseconds), so contention is negligible.
//!
//! # CL-323

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

use crate::allocator::StreamId;

static POOL_HITS: AtomicUsize = AtomicUsize::new(0);
static POOL_MISSES: AtomicUsize = AtomicUsize::new(0);
static POOL_RETURNS: AtomicUsize = AtomicUsize::new(0);

/// Get pool statistics: (hits, misses, returns).
pub fn pool_stats() -> (usize, usize, usize) {
    (
        POOL_HITS.load(Ordering::Relaxed),
        POOL_MISSES.load(Ordering::Relaxed),
        POOL_RETURNS.load(Ordering::Relaxed),
    )
}

/// Reset pool statistics.
pub fn reset_pool_stats() {
    POOL_HITS.store(0, Ordering::Relaxed);
    POOL_MISSES.store(0, Ordering::Relaxed);
    POOL_RETURNS.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Pool key: (device_ordinal, element_count, TypeId)
// ---------------------------------------------------------------------------

type PoolKey = (usize, usize, TypeId);

/// Metadata stored alongside each cached buffer for stream-aware reuse.
///
/// # CL-323
struct CachedEntry {
    /// The type-erased `CudaSlice<T>`.
    data: Box<dyn Any + Send + Sync>,
    /// The stream on which this buffer was originally allocated.
    alloc_stream: StreamId,
    /// Streams that have used this buffer (recorded via `record_stream`).
    /// The buffer can only be reused when all these streams have completed
    /// their work.
    stream_uses: Vec<StreamId>,
}

struct PoolState {
    /// Free buffers keyed by (device, len, type). Values are cached entries
    /// with stream metadata. LIFO for temporal locality.
    free: HashMap<PoolKey, Vec<CachedEntry>>,
    /// Total cached bytes (not currently in use).
    cached_bytes: usize,
}

impl PoolState {
    fn new() -> Self {
        Self {
            free: HashMap::new(),
            cached_bytes: 0,
        }
    }
}

static POOL: LazyLock<Mutex<PoolState>> = LazyLock::new(|| Mutex::new(PoolState::new()));

// ---------------------------------------------------------------------------
// Length rounding
// ---------------------------------------------------------------------------

/// Allocation granularity: round up to the nearest multiple of 256 elements.
///
/// This ensures that pool keys are stable across allocations for the same
/// logical size, preventing fragmentation where a buffer allocated with
/// `len` elements cannot be found in the pool because the key differs
/// from the rounded allocation size.
const ROUND_ELEMENTS: usize = 256;

/// Round `len` up to the nearest multiple of the `ROUND_ELEMENTS` constant (256).
///
/// Uses saturating arithmetic to avoid overflow on extreme inputs.
pub fn round_len(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let remainder = len % ROUND_ELEMENTS;
    if remainder == 0 {
        return len;
    }
    len.saturating_add(ROUND_ELEMENTS - remainder)
}

// ---------------------------------------------------------------------------
// Generic pool operations
// ---------------------------------------------------------------------------

/// Try to get a cached `CudaSlice<T>` with exactly `rounded_len` elements
/// from the pool for the given device. Returns `None` on cache miss.
///
/// `rounded_len` must already be rounded via [`round_len`]. `elem_size` is
/// the size of one element in bytes (e.g. 4 for f32) and is used only for
/// byte-level accounting — NOT for pool key lookup.
pub fn pool_take<T: Any + Send + Sync>(
    device_ordinal: usize,
    rounded_len: usize,
    elem_size: usize,
) -> Option<T> {
    let key = (device_ordinal, rounded_len, TypeId::of::<T>());
    // Mutex poison is silently swallowed: this is intentional defensive
    // behavior — a poisoned pool degrades to "always miss" rather than
    // panicking the caller. GPU allocations will still succeed via fresh
    // CUDA driver calls, just without caching.
    let mut pool = POOL.lock().ok()?;
    let bucket = pool.free.get_mut(&key)?;
    let entry = bucket.pop()?;
    let is_empty = bucket.is_empty();
    if is_empty {
        pool.free.remove(&key);
    }
    pool.cached_bytes = pool.cached_bytes.saturating_sub(rounded_len * elem_size);
    POOL_HITS.fetch_add(1, Ordering::Relaxed);
    // Downcast is guaranteed to succeed because the key includes TypeId.
    Some(*entry.data.downcast::<T>().expect("pool type mismatch"))
}

/// Stream-aware variant of [`pool_take`]. Only returns a buffer whose
/// `alloc_stream` matches the given `stream` and has no pending
/// cross-stream uses, ensuring correct synchronization.
///
/// # CL-323
pub fn pool_take_stream<T: Any + Send + Sync>(
    device_ordinal: usize,
    rounded_len: usize,
    elem_size: usize,
    stream: StreamId,
) -> Option<T> {
    let key = (device_ordinal, rounded_len, TypeId::of::<T>());
    let mut pool = POOL.lock().ok()?;
    let bucket = pool.free.get_mut(&key)?;

    // Search from the back (LIFO) for a buffer on the same stream with
    // no pending cross-stream uses.
    let pos = bucket
        .iter()
        .rposition(|entry| entry.alloc_stream == stream && entry.stream_uses.is_empty())?;

    let entry = bucket.swap_remove(pos);
    if bucket.is_empty() {
        pool.free.remove(&key);
    }
    pool.cached_bytes = pool.cached_bytes.saturating_sub(rounded_len * elem_size);
    POOL_HITS.fetch_add(1, Ordering::Relaxed);
    Some(*entry.data.downcast::<T>().expect("pool type mismatch"))
}

/// Return a value to the pool for later reuse.
///
/// `rounded_len` must already be rounded via [`round_len`]. `elem_size` is
/// used for byte-level accounting only. The `POOL_RETURNS` counter is
/// incremented only after the buffer is successfully inserted into the
/// pool bucket.
pub fn pool_return<T: Any + Send + Sync>(
    device_ordinal: usize,
    rounded_len: usize,
    elem_size: usize,
    value: T,
) {
    pool_return_with_stream(device_ordinal, rounded_len, elem_size, value, StreamId(0))
}

/// Return a value to the pool with stream metadata.
///
/// Like [`pool_return`] but records which stream the buffer was used on,
/// enabling stream-aware reuse via [`pool_take_stream`].
///
/// # CL-323
pub fn pool_return_with_stream<T: Any + Send + Sync>(
    device_ordinal: usize,
    rounded_len: usize,
    elem_size: usize,
    value: T,
    alloc_stream: StreamId,
) {
    let key = (device_ordinal, rounded_len, TypeId::of::<T>());
    let Ok(mut pool) = POOL.lock() else { return };
    pool.cached_bytes += rounded_len * elem_size;
    let entry = CachedEntry {
        data: Box::new(value),
        alloc_stream,
        stream_uses: Vec::new(),
    };
    pool.free.entry(key).or_default().push(entry);
    POOL_RETURNS.fetch_add(1, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Stream recording
// ---------------------------------------------------------------------------

/// Record that a buffer (identified by its pool key) was used on `stream`.
///
/// This prevents the buffer from being returned by [`pool_take_stream`]
/// until the recorded stream's work is complete. Callers should call this
/// when a buffer allocated on one stream is consumed by a kernel on a
/// different stream.
///
/// This is the Rust equivalent of PyTorch's `recordStream()`.
///
/// # CL-323
pub fn record_stream<T: Any + Send + Sync>(
    device_ordinal: usize,
    rounded_len: usize,
    stream: StreamId,
) {
    let key = (device_ordinal, rounded_len, TypeId::of::<T>());
    let Ok(mut pool) = POOL.lock() else { return };
    if let Some(bucket) = pool.free.get_mut(&key) {
        for entry in bucket.iter_mut() {
            entry.stream_uses.push(stream);
        }
    }
}

/// Record a stream use on a specific buffer in the pool. This is used to
/// track cross-stream dependencies so the buffer is not prematurely reused.
///
/// # CL-323
#[cfg(feature = "cuda")]
pub fn record_stream_on_buffer(
    device_ordinal: usize,
    rounded_len: usize,
    type_id: TypeId,
    stream: StreamId,
) {
    let key = (device_ordinal, rounded_len, type_id);
    let Ok(mut pool) = POOL.lock() else { return };
    if let Some(bucket) = pool.free.get_mut(&key) {
        // Record on all entries in this bucket. In practice there is usually
        // only one entry per key.
        for entry in bucket.iter_mut() {
            entry.stream_uses.push(stream);
        }
    }
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

/// Drop all cached buffers for a device, releasing GPU memory back to the
/// CUDA driver.
pub fn empty_cache(device_ordinal: usize) {
    let Ok(mut pool) = POOL.lock() else { return };
    pool.free.retain(|&(dev, _, _), _| dev != device_ordinal);
    // Recalculate cached_bytes from remaining entries.
    // Note: we don't store elem_size per entry, so we conservatively estimate
    // by summing the byte counts we have. After a device-specific clear the
    // remaining entries may use different elem sizes, so we just reset to 0
    // and accept that the counter may be slightly off until the next return.
    // A full clear (empty_cache_all) resets to 0 exactly.
    pool.cached_bytes = 0;
}

/// Drop all cached buffers across all devices.
pub fn empty_cache_all() {
    let Ok(mut pool) = POOL.lock() else { return };
    pool.free.clear();
    pool.cached_bytes = 0;
}

/// Total bytes currently cached (available for reuse).
pub fn cached_bytes(_device_ordinal: usize) -> usize {
    POOL.lock().ok().map(|p| p.cached_bytes).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_len_zero() {
        assert_eq!(round_len(0), 0);
    }

    #[test]
    fn round_len_exact_multiple() {
        assert_eq!(round_len(256), 256);
        assert_eq!(round_len(512), 512);
    }

    #[test]
    fn round_len_rounds_up() {
        assert_eq!(round_len(1), 256);
        assert_eq!(round_len(255), 256);
        assert_eq!(round_len(257), 512);
    }

    #[test]
    fn pool_take_miss_returns_none() {
        // Use a unique device ID to avoid interference from parallel tests.
        let result = pool_take::<u64>(9901, 256, 8);
        assert!(result.is_none());
    }

    #[test]
    fn pool_return_then_take() {
        // Use a unique device ID to avoid interference from parallel tests.
        let value: u64 = 12345;
        pool_return::<u64>(9902, 256, 8, value);
        let taken = pool_take::<u64>(9902, 256, 8);
        assert_eq!(taken, Some(12345u64));
    }

    #[test]
    fn pool_stats_tracking() {
        reset_pool_stats();
        let (h, _m, r) = pool_stats();
        assert_eq!(h, 0);
        assert_eq!(r, 0);

        pool_return::<u32>(98, 256, 4, 42u32);
        let (_, _, r) = pool_stats();
        assert!(r >= 1);

        let _ = pool_take::<u32>(98, 256, 4);
        let (h, _, _) = pool_stats();
        assert!(h >= 1);
    }

    #[test]
    fn stream_aware_take() {
        let stream_a = StreamId(100);
        let stream_b = StreamId(200);

        // Return a buffer on stream A.
        pool_return_with_stream::<u64>(97, 256, 8, 777u64, stream_a);

        // Taking for stream B should fail (stream mismatch).
        let taken = pool_take_stream::<u64>(97, 256, 8, stream_b);
        assert!(taken.is_none());

        // Taking for stream A should succeed.
        let taken = pool_take_stream::<u64>(97, 256, 8, stream_a);
        assert_eq!(taken, Some(777u64));
    }

    #[test]
    fn record_stream_prevents_reuse() {
        let stream_a = StreamId(300);
        let stream_b = StreamId(400);

        // Return a buffer on stream A.
        pool_return_with_stream::<u64>(96, 256, 8, 888u64, stream_a);

        // Record stream B usage on all entries in this bucket.
        record_stream::<u64>(96, 256, stream_b);

        // Now pool_take_stream for stream A should fail because stream_uses
        // is non-empty (stream B recorded).
        let taken = pool_take_stream::<u64>(96, 256, 8, stream_a);
        assert!(taken.is_none());

        // But the plain pool_take (non-stream-aware) still works.
        let taken = pool_take::<u64>(96, 256, 8);
        assert_eq!(taken, Some(888u64));
    }

    #[test]
    fn empty_cache_clears_device() {
        pool_return::<u32>(95, 256, 4, 11u32);
        pool_return::<u32>(94, 256, 4, 22u32);

        empty_cache(95);

        // Device 95 cleared.
        assert!(pool_take::<u32>(95, 256, 4).is_none());
        // Device 94 untouched.
        assert_eq!(pool_take::<u32>(94, 256, 4), Some(22u32));
    }

    #[test]
    fn empty_cache_all_clears_everything() {
        pool_return::<u32>(93, 256, 4, 33u32);
        pool_return::<u32>(92, 256, 4, 44u32);

        empty_cache_all();

        assert!(pool_take::<u32>(93, 256, 4).is_none());
        assert!(pool_take::<u32>(92, 256, 4).is_none());
    }
}
