//! GPU buffer pool — caching allocator for CUDA memory.
//!
//! Eliminates per-operation `cuMemAllocAsync` + `cuMemFreeAsync` +
//! `cuEventCreate` × 2 + `cuEventDestroy` × 2 by reusing freed buffers.
//! On a pool hit, the only CUDA call is `cuMemsetD8Async` (to zero the
//! buffer). The `CudaSlice`'s events are kept alive across reuses, so
//! no event creation or destruction is needed.
//!
//! This is the same design principle as PyTorch's `CUDACachingAllocator`:
//! never actually free GPU memory, just return it to a free list.
//!
//! # Thread safety
//!
//! The pool is protected by a `Mutex`. The critical section is a `HashMap`
//! lookup + `Vec::pop` (microseconds), so contention is negligible.

use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{LazyLock, Mutex};

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

struct PoolState {
    /// Free buffers keyed by (device, len, type). Values are `Box<CudaSlice<T>>`
    /// type-erased as `Box<dyn Any + Send + Sync>`. LIFO for temporal locality.
    free: HashMap<PoolKey, Vec<Box<dyn Any + Send + Sync>>>,
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

/// Round `len` up to the nearest multiple of [`ROUND_ELEMENTS`].
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
    let boxed = bucket.pop()?;
    let is_empty = bucket.is_empty();
    if is_empty {
        pool.free.remove(&key);
    }
    pool.cached_bytes = pool.cached_bytes.saturating_sub(rounded_len * elem_size);
    POOL_HITS.fetch_add(1, Ordering::Relaxed);
    // Downcast is guaranteed to succeed because the key includes TypeId.
    Some(*boxed.downcast::<T>().expect("pool type mismatch"))
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
    let key = (device_ordinal, rounded_len, TypeId::of::<T>());
    // Mutex poison is silently swallowed: intentional defensive behavior.
    // A poisoned pool degrades to "leak this buffer" rather than panicking.
    let Ok(mut pool) = POOL.lock() else { return };
    pool.cached_bytes += rounded_len * elem_size;
    pool.free.entry(key).or_default().push(Box::new(value));
    // P9 fix: increment counter AFTER the bucket push, not before the
    // limit check, so the counter accurately reflects successful returns.
    POOL_RETURNS.fetch_add(1, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

/// Drop all cached buffers, releasing GPU memory back to the CUDA driver.
pub fn empty_cache(device_ordinal: usize) {
    let Ok(mut pool) = POOL.lock() else { return };
    pool.free.retain(|&(dev, _, _), _| dev != device_ordinal);
    // Recalculate cached_bytes from remaining entries.
    let remaining: usize = pool.free.iter()
        .map(|((_, len, _), bucket)| len * bucket.len() * 4) // approximate
        .sum();
    pool.cached_bytes = remaining;
}

/// Drop all cached buffers across all devices.
pub fn empty_cache_all() {
    let Ok(mut pool) = POOL.lock() else { return };
    pool.free.clear();
    pool.cached_bytes = 0;
}

/// Total bytes currently cached (available for reuse).
pub fn cached_bytes(_device_ordinal: usize) -> usize {
    POOL.lock()
        .ok()
        .map(|p| p.cached_bytes)
        .unwrap_or(0)
}
