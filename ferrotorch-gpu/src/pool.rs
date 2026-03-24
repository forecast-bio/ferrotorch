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
//! # CUDA graph capture pool isolation
//!
//! During CUDA graph capture, allocations must not alias with the normal
//! pool. When [`set_capturing`] sets the thread-local capture flag, all
//! [`pool_take`] and [`pool_return`] calls are redirected to a separate
//! capture-private pool keyed by a [`CapturePoolId`]. After capture ends,
//! the capture pool is sealed and its buffers are held for the lifetime of
//! the graph — preventing use-after-free when the graph replays.
//!
//! # Thread safety
//!
//! The pool is protected by a `Mutex`. The critical section is a `HashMap`
//! lookup + `Vec::pop` (microseconds), so contention is negligible.

use std::any::{Any, TypeId};
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
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
// Capture-mode flag (thread-local)
// ---------------------------------------------------------------------------

/// Opaque identifier for a capture-private pool.
///
/// Each CUDA graph capture session gets a unique pool ID. Multiple graphs
/// can share a pool by using the same ID (see [`CapturedGraph::pool_id`]).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CapturePoolId(u64);

impl CapturePoolId {
    /// Generate a fresh, globally unique capture pool ID.
    pub fn next() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(1);
        Self(COUNTER.fetch_add(1, Ordering::Relaxed))
    }

    /// Raw numeric value of this pool ID. Useful for diagnostics.
    #[inline]
    pub fn raw(&self) -> u64 {
        self.0
    }
}

thread_local! {
    /// Whether the current thread is inside a CUDA graph capture session.
    /// When `true`, pool operations are redirected to the capture-private pool.
    static CAPTURING: Cell<bool> = const { Cell::new(false) };

    /// The capture pool ID for the current capture session. Only meaningful
    /// when `CAPTURING` is `true`.
    static CAPTURE_POOL_ID: Cell<Option<CapturePoolId>> = const { Cell::new(None) };
}

/// Returns `true` if the current thread is inside a CUDA graph capture.
pub fn is_capturing() -> bool {
    CAPTURING.with(|c| c.get())
}

/// Begin capture-mode pool redirection for this thread.
///
/// Returns `false` if already capturing (nested capture attempt), in which
/// case the flag is NOT changed — the caller must handle the error.
pub fn set_capturing(pool_id: CapturePoolId) -> bool {
    CAPTURING.with(|c| {
        if c.get() {
            // Nested capture: refuse to overwrite.
            return false;
        }
        c.set(true);
        CAPTURE_POOL_ID.with(|p| p.set(Some(pool_id)));
        true
    })
}

/// End capture-mode pool redirection for this thread.
///
/// Returns the pool ID that was active, or `None` if capture was not active.
pub fn clear_capturing() -> Option<CapturePoolId> {
    CAPTURING.with(|c| {
        if !c.get() {
            return None;
        }
        c.set(false);
        CAPTURE_POOL_ID.with(|p| p.take())
    })
}

/// Get the current capture pool ID, if capturing.
pub fn current_capture_pool_id() -> Option<CapturePoolId> {
    if !is_capturing() {
        return None;
    }
    CAPTURE_POOL_ID.with(|p| p.get())
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
// Capture-private pools
// ---------------------------------------------------------------------------

/// State for a single capture-private pool. Identical structure to the
/// normal pool but scoped to a single capture session.
struct CapturePoolState {
    free: HashMap<PoolKey, Vec<Box<dyn Any + Send + Sync>>>,
    cached_bytes: usize,
    /// Once sealed, no further takes/returns are allowed. The buffers are
    /// held alive for the graph's lifetime.
    sealed: bool,
}

impl CapturePoolState {
    fn new() -> Self {
        Self {
            free: HashMap::new(),
            cached_bytes: 0,
            sealed: false,
        }
    }
}

/// Global registry of capture-private pools, keyed by `CapturePoolId`.
static CAPTURE_POOLS: LazyLock<Mutex<HashMap<CapturePoolId, CapturePoolState>>> =
    LazyLock::new(|| Mutex::new(HashMap::new()));

/// Seal a capture pool, preventing further allocations from it.
///
/// Called by `end_capture` after the graph is instantiated. The pool's
/// buffers remain alive — they are the backing memory for the graph.
/// If the pool does not exist (empty capture with no allocations), it
/// is created in sealed state.
pub fn seal_capture_pool(pool_id: CapturePoolId) {
    let Ok(mut pools) = CAPTURE_POOLS.lock() else { return };
    let state = pools.entry(pool_id).or_insert_with(CapturePoolState::new);
    state.sealed = true;
}

/// Drop a capture pool entirely, freeing all its buffers.
///
/// Called when a `CapturedGraph` is dropped and no longer needs its memory.
pub fn drop_capture_pool(pool_id: CapturePoolId) {
    let Ok(mut pools) = CAPTURE_POOLS.lock() else { return };
    pools.remove(&pool_id);
}

/// Get the number of buffers held in a capture pool.
///
/// Returns 0 if the pool does not exist.
pub fn capture_pool_buffer_count(pool_id: CapturePoolId) -> usize {
    let Ok(pools) = CAPTURE_POOLS.lock() else { return 0 };
    pools
        .get(&pool_id)
        .map(|s| s.free.values().map(|v| v.len()).sum())
        .unwrap_or(0)
}

/// Check if a capture pool is sealed.
pub fn is_capture_pool_sealed(pool_id: CapturePoolId) -> bool {
    let Ok(pools) = CAPTURE_POOLS.lock() else { return false };
    pools.get(&pool_id).is_some_and(|s| s.sealed)
}

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
///
/// During CUDA graph capture, this reads from the capture-private pool
/// instead of the global pool. This prevents graph replay from aliasing
/// with non-graph allocations.
pub fn pool_take<T: Any + Send + Sync>(
    device_ordinal: usize,
    rounded_len: usize,
    elem_size: usize,
) -> Option<T> {
    // Capture-mode redirect: use capture-private pool.
    if let Some(capture_id) = current_capture_pool_id() {
        return capture_pool_take::<T>(capture_id, device_ordinal, rounded_len, elem_size);
    }

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
///
/// During CUDA graph capture, this stores into the capture-private pool
/// instead of the global pool.
pub fn pool_return<T: Any + Send + Sync>(
    device_ordinal: usize,
    rounded_len: usize,
    elem_size: usize,
    value: T,
) {
    // Capture-mode redirect: use capture-private pool.
    if let Some(capture_id) = current_capture_pool_id() {
        capture_pool_return::<T>(capture_id, device_ordinal, rounded_len, elem_size, value);
        return;
    }

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
// Capture-private pool take / return
// ---------------------------------------------------------------------------

/// Take from a capture-private pool. Returns `None` if the pool doesn't
/// exist, is sealed, or has no matching buffer.
fn capture_pool_take<T: Any + Send + Sync>(
    pool_id: CapturePoolId,
    device_ordinal: usize,
    rounded_len: usize,
    elem_size: usize,
) -> Option<T> {
    let key = (device_ordinal, rounded_len, TypeId::of::<T>());
    let mut pools = CAPTURE_POOLS.lock().ok()?;
    let state = pools.get_mut(&pool_id)?;
    if state.sealed {
        return None;
    }
    let bucket = state.free.get_mut(&key)?;
    let boxed = bucket.pop()?;
    if bucket.is_empty() {
        state.free.remove(&key);
    }
    state.cached_bytes = state.cached_bytes.saturating_sub(rounded_len * elem_size);
    POOL_HITS.fetch_add(1, Ordering::Relaxed);
    Some(*boxed.downcast::<T>().expect("capture pool type mismatch"))
}

/// Return to a capture-private pool. Creates the pool entry if it doesn't
/// exist yet. Refuses to insert if the pool is sealed.
fn capture_pool_return<T: Any + Send + Sync>(
    pool_id: CapturePoolId,
    device_ordinal: usize,
    rounded_len: usize,
    elem_size: usize,
    value: T,
) {
    let key = (device_ordinal, rounded_len, TypeId::of::<T>());
    let Ok(mut pools) = CAPTURE_POOLS.lock() else { return };
    let state = pools.entry(pool_id).or_insert_with(CapturePoolState::new);
    if state.sealed {
        // Pool is sealed — drop the buffer rather than silently returning
        // it to a finalized graph's pool.
        return;
    }
    state.cached_bytes += rounded_len * elem_size;
    state.free.entry(key).or_default().push(Box::new(value));
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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: drain the global pool for a given key so tests don't
    /// interfere with each other via static state.
    fn drain_global_pool(device: usize, rounded_len: usize) {
        while pool_take::<Vec<u8>>(device, rounded_len, 1).is_some() {}
    }

    // -- Capture-mode flag toggle -------------------------------------------

    #[test]
    fn capturing_flag_defaults_to_false() {
        assert!(!is_capturing());
        assert!(current_capture_pool_id().is_none());
    }

    #[test]
    fn set_and_clear_capturing() {
        let id = CapturePoolId::next();
        assert!(set_capturing(id));
        assert!(is_capturing());
        assert_eq!(current_capture_pool_id(), Some(id));

        let cleared = clear_capturing();
        assert_eq!(cleared, Some(id));
        assert!(!is_capturing());
        assert!(current_capture_pool_id().is_none());
    }

    #[test]
    fn nested_capture_refused() {
        let id1 = CapturePoolId::next();
        let id2 = CapturePoolId::next();

        assert!(set_capturing(id1));
        // Second set_capturing must return false and leave the original
        // capture in place.
        assert!(!set_capturing(id2));
        assert_eq!(current_capture_pool_id(), Some(id1));

        // Clean up.
        clear_capturing();
    }

    #[test]
    fn clear_capturing_when_not_capturing_returns_none() {
        assert!(!is_capturing());
        assert!(clear_capturing().is_none());
    }

    // -- Allocations during capture use separate pool -----------------------

    #[test]
    fn capture_pool_return_isolates_from_global() {
        let device = 99; // unlikely to conflict with real device
        let rlen = 256;

        // Ensure global pool is empty for this key.
        drain_global_pool(device, rlen);

        // Return a buffer to the global pool.
        pool_return::<Vec<u8>>(device, rlen, 1, vec![1u8; rlen]);

        // Now start capture and return a different buffer.
        let capture_id = CapturePoolId::next();
        assert!(set_capturing(capture_id));
        pool_return::<Vec<u8>>(device, rlen, 1, vec![2u8; rlen]);

        // Taking during capture should get the capture buffer (value 2),
        // NOT the global buffer (value 1).
        let taken: Vec<u8> = pool_take(device, rlen, 1).expect("capture pool should hit");
        assert_eq!(taken[0], 2);

        clear_capturing();

        // After capture, taking should get the global buffer (value 1).
        let taken: Vec<u8> = pool_take(device, rlen, 1).expect("global pool should hit");
        assert_eq!(taken[0], 1);

        // Clean up capture pool.
        drop_capture_pool(capture_id);
    }

    #[test]
    fn capture_pool_take_misses_do_not_touch_global() {
        let device = 98;
        let rlen = 512;

        drain_global_pool(device, rlen);

        // Put something in global.
        pool_return::<Vec<u8>>(device, rlen, 1, vec![10u8; rlen]);

        let capture_id = CapturePoolId::next();
        assert!(set_capturing(capture_id));

        // Capture pool is empty — take should miss.
        let result: Option<Vec<u8>> = pool_take(device, rlen, 1);
        assert!(result.is_none());

        clear_capturing();

        // Global pool should still have its buffer.
        let global: Vec<u8> = pool_take(device, rlen, 1).expect("global pool untouched");
        assert_eq!(global[0], 10);

        drop_capture_pool(capture_id);
    }

    // -- Normal allocations unaffected by capture pool ----------------------

    #[test]
    fn normal_pool_unaffected_outside_capture() {
        let device = 97;
        let rlen = 256;

        drain_global_pool(device, rlen);

        pool_return::<Vec<u8>>(device, rlen, 1, vec![42u8; rlen]);
        let taken: Vec<u8> = pool_take(device, rlen, 1).expect("normal pool hit");
        assert_eq!(taken[0], 42);
    }

    // -- Pool sealing -------------------------------------------------------

    #[test]
    fn sealed_pool_refuses_take_and_return() {
        let capture_id = CapturePoolId::next();
        let device = 96;
        let rlen = 256;

        assert!(set_capturing(capture_id));
        pool_return::<Vec<u8>>(device, rlen, 1, vec![7u8; rlen]);
        clear_capturing();

        // Seal the pool.
        seal_capture_pool(capture_id);
        assert!(is_capture_pool_sealed(capture_id));

        // Manually try to take from the sealed pool — should fail.
        assert!(set_capturing(capture_id));
        let result: Option<Vec<u8>> = pool_take(device, rlen, 1);
        assert!(result.is_none());

        // Return to sealed pool should be silently dropped.
        pool_return::<Vec<u8>>(device, rlen, 1, vec![8u8; rlen]);
        // Buffer count should still be 1 (the original, not the dropped one).
        clear_capturing();
        assert_eq!(capture_pool_buffer_count(capture_id), 1);

        drop_capture_pool(capture_id);
    }

    // -- Pool drop ----------------------------------------------------------

    #[test]
    fn drop_capture_pool_cleans_up() {
        let capture_id = CapturePoolId::next();
        let device = 95;
        let rlen = 256;

        assert!(set_capturing(capture_id));
        pool_return::<Vec<u8>>(device, rlen, 1, vec![0u8; rlen]);
        pool_return::<Vec<u8>>(device, rlen, 1, vec![0u8; rlen]);
        clear_capturing();

        assert_eq!(capture_pool_buffer_count(capture_id), 2);

        drop_capture_pool(capture_id);
        assert_eq!(capture_pool_buffer_count(capture_id), 0);
    }

    // -- CapturePoolId uniqueness -------------------------------------------

    #[test]
    fn capture_pool_ids_are_unique() {
        let a = CapturePoolId::next();
        let b = CapturePoolId::next();
        let c = CapturePoolId::next();
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    // -- Empty capture (no allocations during capture) ----------------------

    #[test]
    fn empty_capture_pool_is_valid() {
        let capture_id = CapturePoolId::next();

        assert!(set_capturing(capture_id));
        // No pool_return / pool_take calls — empty capture.
        clear_capturing();

        seal_capture_pool(capture_id);
        assert_eq!(capture_pool_buffer_count(capture_id), 0);
        assert!(is_capture_pool_sealed(capture_id));

        drop_capture_pool(capture_id);
    }
}
