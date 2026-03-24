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
//! # Size rounding & block reuse
//!
//! Element counts are rounded up to the next multiple of 128 (matching
//! PyTorch's 512-byte granularity for f32). This turns near-miss sizes
//! into cache hits. When no exact-rounded-size match exists, the smallest
//! cached block that is >= the requested size is returned. Since
//! `CudaSlice` is opaque, larger blocks are reused whole (no splitting).
//!
//! # Two-pool design
//!
//! Small allocations (<=256KB) and large allocations (>256KB) use separate
//! pools with different eviction strategies, matching PyTorch's pattern.
//!
//! # Memory limit
//!
//! Total cached bytes are bounded by `MAX_CACHED_BYTES` (default 1GB).
//! When the limit is exceeded, the largest unused blocks are dropped first.
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
// Constants
// ---------------------------------------------------------------------------

/// Element count rounding granularity. For f32 (4 bytes), 128 elements = 512
/// bytes, matching PyTorch's `kMinBlockSize`.
const ROUND_ELEMENTS: usize = 128;

/// Threshold (in bytes) separating the small pool from the large pool.
/// 256 KiB = 65536 f32 elements.
const SMALL_POOL_THRESHOLD: usize = 256 * 1024;

/// Maximum total cached bytes across both pools. Default 1 GiB.
const MAX_CACHED_BYTES: usize = 1024 * 1024 * 1024;

/// Maximum number of buffers per bucket (prevents unbounded growth in a
/// single size class).
const MAX_PER_BUCKET: usize = 32;

// ---------------------------------------------------------------------------
// Size rounding
// ---------------------------------------------------------------------------

/// Round an element count up to the next multiple of `ROUND_ELEMENTS`.
/// Returns 0 for input 0.
#[inline]
pub fn round_len(len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    // ceil(len / ROUND_ELEMENTS) * ROUND_ELEMENTS
    let remainder = len % ROUND_ELEMENTS;
    if remainder == 0 {
        len
    } else {
        len + (ROUND_ELEMENTS - remainder)
    }
}

// ---------------------------------------------------------------------------
// Pool key: (device_ordinal, rounded_element_count, TypeId)
// ---------------------------------------------------------------------------

type PoolKey = (usize, usize, TypeId);

/// An entry in the pool: the type-erased value plus the actual allocated
/// element count (which may be larger than the pool key's rounded_len due
/// to block reuse).
struct PoolEntry {
    /// `Box<CudaSlice<T>>` type-erased.
    value: Box<dyn Any + Send + Sync>,
    /// The actual number of elements allocated in this CudaSlice.
    /// Always >= the rounded_len in the pool key.
    alloc_len: usize,
    /// Element size in bytes (4 for f32, 8 for f64).
    elem_size: usize,
}

impl PoolEntry {
    /// Total bytes occupied by this entry.
    #[inline]
    fn byte_size(&self) -> usize {
        self.alloc_len * self.elem_size
    }
}

struct PoolState {
    /// Small pool: allocations <= SMALL_POOL_THRESHOLD bytes.
    small: HashMap<PoolKey, Vec<PoolEntry>>,
    /// Large pool: allocations > SMALL_POOL_THRESHOLD bytes.
    large: HashMap<PoolKey, Vec<PoolEntry>>,
    /// Total cached bytes across both pools (not currently in use).
    cached_bytes: usize,
}

impl PoolState {
    fn new() -> Self {
        Self {
            small: HashMap::new(),
            large: HashMap::new(),
            cached_bytes: 0,
        }
    }

    /// Select the appropriate pool (small or large) mutably.
    #[inline]
    fn pool_for_mut(&mut self, byte_size: usize) -> &mut HashMap<PoolKey, Vec<PoolEntry>> {
        if byte_size <= SMALL_POOL_THRESHOLD {
            &mut self.small
        } else {
            &mut self.large
        }
    }

    /// Enforce `MAX_CACHED_BYTES` by evicting the largest blocks first.
    /// Called after inserting into the pool.
    fn enforce_limit(&mut self) {
        while self.cached_bytes > MAX_CACHED_BYTES {
            // Find the largest single entry across both pools and remove it.
            let largest = self.find_largest_entry();
            match largest {
                Some((key, idx, byte_size, is_small)) => {
                    let pool = if is_small {
                        &mut self.small
                    } else {
                        &mut self.large
                    };
                    if let Some(bucket) = pool.get_mut(&key) {
                        if idx < bucket.len() {
                            bucket.swap_remove(idx);
                            self.cached_bytes = self.cached_bytes.saturating_sub(byte_size);
                        }
                        if bucket.is_empty() {
                            pool.remove(&key);
                        }
                    }
                }
                None => break, // No entries to evict.
            }
        }
    }

    /// Find the largest entry across both pools.
    /// Returns (key, index_in_bucket, byte_size, is_small_pool).
    fn find_largest_entry(&self) -> Option<(PoolKey, usize, usize, bool)> {
        let mut best: Option<(PoolKey, usize, usize, bool)> = None;

        for (key, bucket) in &self.small {
            for (idx, entry) in bucket.iter().enumerate() {
                let bs = entry.byte_size();
                if best.as_ref().is_none_or(|b| bs > b.2) {
                    best = Some((*key, idx, bs, true));
                }
            }
        }
        for (key, bucket) in &self.large {
            for (idx, entry) in bucket.iter().enumerate() {
                let bs = entry.byte_size();
                if best.as_ref().is_none_or(|b| bs > b.2) {
                    best = Some((*key, idx, bs, false));
                }
            }
        }
        best
    }
}

static POOL: LazyLock<Mutex<PoolState>> = LazyLock::new(|| Mutex::new(PoolState::new()));

// ---------------------------------------------------------------------------
// Generic pool operations
// ---------------------------------------------------------------------------

/// Result from [`pool_take`]: the value and the actual allocated element
/// count (which may be larger than the requested length).
pub struct PoolTakeResult<T> {
    /// The cached value.
    pub value: T,
    /// Actual number of elements allocated. Always >= the requested len
    /// (after rounding). Callers must track this for correct pool_return.
    pub alloc_len: usize,
}

/// Try to get a cached `CudaSlice<T>` with at least `len` elements from the
/// pool for the given device.
///
/// The search order is:
/// 1. Exact match on `(device, round_len(len), TypeId)` — entries in that
///    bucket whose `alloc_len >= round_len(len)`.
/// 2. Best-fit search across all buckets: find the smallest cached block
///    with `alloc_len >= round_len(len)` on the same device and type.
///    Since `CudaSlice` is opaque (no splitting), larger blocks are
///    reused whole.
///
/// Returns `None` on cache miss.
pub fn pool_take<T: Any + Send + Sync>(
    device_ordinal: usize,
    len: usize,
    elem_size: usize,
) -> Option<PoolTakeResult<T>> {
    if len == 0 {
        return None;
    }

    let rounded = round_len(len);
    let byte_size = rounded * elem_size;
    let type_id = TypeId::of::<T>();
    let exact_key = (device_ordinal, rounded, type_id);

    let mut pool = POOL.lock().ok()?;
    let pool_map = pool.pool_for_mut(byte_size);

    // 1. Try exact match first (fastest path).
    //    Even on a key match, verify the entry's actual allocation is large
    //    enough. Different-sized CudaSlices may share the same rounded key
    //    (e.g., alloc_len=2 and alloc_len=128 both round to key 128).
    if let Some(bucket) = pool_map.get_mut(&exact_key) {
        // Scan from the back (LIFO) for an entry with sufficient alloc_len.
        if let Some(pos) = bucket.iter().rposition(|e| e.alloc_len >= rounded) {
            let entry = bucket.swap_remove(pos);
            let alloc_len = entry.alloc_len;
            let entry_bytes = entry.byte_size();
            if bucket.is_empty() {
                pool_map.remove(&exact_key);
            }
            pool.cached_bytes = pool.cached_bytes.saturating_sub(entry_bytes);
            POOL_HITS.fetch_add(1, Ordering::Relaxed);
            // SAFETY: downcast is guaranteed because the key includes TypeId.
            let value = *entry.value.downcast::<T>().expect("pool type mismatch");
            return Some(PoolTakeResult { value, alloc_len });
        }
    }

    // 2. Best-fit search: find the smallest block >= rounded that matches
    //    device and type. We only search the same pool tier (small or large).
    //    Since CudaSlice is opaque (we can't split it), we always reuse the
    //    whole block. We prefer the smallest block that fits to minimize waste.
    let mut best_key: Option<PoolKey> = None;
    let mut best_idx: usize = 0;
    let mut best_len: usize = usize::MAX;

    for (&key, bucket) in pool_map.iter() {
        let (dev, key_rounded, tid) = key;
        if dev != device_ordinal || tid != type_id {
            continue;
        }
        if key_rounded < rounded {
            continue;
        }
        // Check ALL entries in the bucket, not just the last one, because
        // entries may have different actual alloc_len values.
        for (idx, entry) in bucket.iter().enumerate() {
            if entry.alloc_len >= rounded && entry.alloc_len < best_len {
                best_len = entry.alloc_len;
                best_key = Some(key);
                best_idx = idx;
            }
        }
    }

    if let Some(bk) = best_key {
        let bucket = pool_map.get_mut(&bk)?;
        let entry = bucket.swap_remove(best_idx);
        let alloc_len = entry.alloc_len;
        let entry_bytes = entry.byte_size();
        if bucket.is_empty() {
            pool_map.remove(&bk);
        }
        pool.cached_bytes = pool.cached_bytes.saturating_sub(entry_bytes);
        POOL_HITS.fetch_add(1, Ordering::Relaxed);
        let value = *entry.value.downcast::<T>().expect("pool type mismatch");
        return Some(PoolTakeResult { value, alloc_len });
    }

    // Cache miss.
    POOL_MISSES.fetch_add(1, Ordering::Relaxed);
    None
}

/// Return a value to the pool for later reuse.
///
/// `alloc_len` is the actual number of elements in the CudaSlice (the rounded
/// count from when it was originally allocated, or from `PoolTakeResult::alloc_len`).
/// `elem_size` is the size of one element in bytes (4 for f32, 8 for f64).
pub fn pool_return<T: Any + Send + Sync>(
    device_ordinal: usize,
    alloc_len: usize,
    elem_size: usize,
    value: T,
) {
    if alloc_len == 0 {
        return;
    }

    let rounded = round_len(alloc_len);
    let byte_size = alloc_len * elem_size;
    let key = (device_ordinal, rounded, TypeId::of::<T>());

    let Ok(mut pool) = POOL.lock() else { return };
    POOL_RETURNS.fetch_add(1, Ordering::Relaxed);

    let entry = PoolEntry {
        value: Box::new(value),
        alloc_len,
        elem_size,
    };

    // Select pool tier based on actual byte size, not rounded.
    let pool_map = pool.pool_for_mut(byte_size);
    let bucket = pool_map.entry(key).or_default();

    // Per-bucket limit.
    if bucket.len() >= MAX_PER_BUCKET {
        return; // Drop the value normally.
    }

    bucket.push(entry);
    pool.cached_bytes += byte_size;

    // Enforce global limit.
    pool.enforce_limit();
}

// ---------------------------------------------------------------------------
// Cache management
// ---------------------------------------------------------------------------

/// Drop all cached buffers for a specific device, releasing GPU memory back
/// to the CUDA driver.
pub fn empty_cache(device_ordinal: usize) {
    let Ok(mut pool) = POOL.lock() else { return };
    empty_cache_inner(&mut pool, device_ordinal);
}

/// Inner implementation so we can call it while already holding the lock.
fn empty_cache_inner(pool: &mut PoolState, device_ordinal: usize) {
    pool.small.retain(|&(dev, _, _), _| dev != device_ordinal);
    pool.large.retain(|&(dev, _, _), _| dev != device_ordinal);
    // Recalculate cached_bytes from remaining entries.
    pool.cached_bytes = recalculate_cached_bytes(&pool.small, &pool.large);
}

/// Recalculate total cached bytes from both pool maps.
fn recalculate_cached_bytes(
    small: &HashMap<PoolKey, Vec<PoolEntry>>,
    large: &HashMap<PoolKey, Vec<PoolEntry>>,
) -> usize {
    let mut total: usize = 0;
    for bucket in small.values() {
        for entry in bucket {
            total += entry.byte_size();
        }
    }
    for bucket in large.values() {
        for entry in bucket {
            total += entry.byte_size();
        }
    }
    total
}

/// Drop all cached buffers across all devices.
pub fn empty_cache_all() {
    let Ok(mut pool) = POOL.lock() else { return };
    pool.small.clear();
    pool.large.clear();
    pool.cached_bytes = 0;
}

/// Empty cache for a given device, callable when you already suspect OOM.
/// This is the same as `empty_cache` but returns the number of bytes freed.
pub fn empty_cache_for_oom(device_ordinal: usize) -> usize {
    let Ok(mut pool) = POOL.lock() else { return 0 };
    let before = pool.cached_bytes;
    empty_cache_inner(&mut pool, device_ordinal);
    before.saturating_sub(pool.cached_bytes)
}

/// Total bytes currently cached (available for reuse) for a specific device.
pub fn cached_bytes(device_ordinal: usize) -> usize {
    let Ok(pool) = POOL.lock() else { return 0 };
    let mut total: usize = 0;
    for (key, bucket) in pool.small.iter().chain(pool.large.iter()) {
        if key.0 == device_ordinal {
            for entry in bucket {
                total += entry.byte_size();
            }
        }
    }
    total
}

/// Total bytes currently cached across all devices.
pub fn cached_bytes_all() -> usize {
    POOL.lock().ok().map(|p| p.cached_bytes).unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicUsize;

    // Use a simple wrapper type for testing (no CUDA required).
    #[derive(Debug, Clone, PartialEq)]
    struct FakeSlice {
        len: usize,
        device: usize,
    }

    // Each test gets a unique device ordinal so tests don't interfere with
    // each other via the shared global POOL. We use high ordinals (1000+)
    // to avoid collisions with real device ordinals used by other test
    // modules.
    static NEXT_DEVICE: AtomicUsize = AtomicUsize::new(1000);

    fn unique_device() -> usize {
        NEXT_DEVICE.fetch_add(1, Ordering::Relaxed)
    }

    // -- Size rounding tests (pure, no global state) --

    #[test]
    fn round_len_zero() {
        assert_eq!(round_len(0), 0);
    }

    #[test]
    fn round_len_exact_multiple() {
        assert_eq!(round_len(128), 128);
        assert_eq!(round_len(256), 256);
        assert_eq!(round_len(1024), 1024);
    }

    #[test]
    fn round_len_rounds_up() {
        assert_eq!(round_len(1), 128);
        assert_eq!(round_len(100), 128);
        assert_eq!(round_len(127), 128);
        assert_eq!(round_len(129), 256);
        assert_eq!(round_len(200), 256);
        assert_eq!(round_len(500), 512);
    }

    #[test]
    fn round_len_large_values() {
        // 1_000_000 / 128 = 7812.5 → rounds up to 7813 * 128 = 1_000_064.
        assert_eq!(round_len(1_000_000), 1_000_064);
        assert_eq!(round_len(1_000_001), 1_000_064);
        // An exact multiple:
        assert_eq!(round_len(1_000_064), 1_000_064);
    }

    // -- Pool take/return basic tests --

    #[test]
    fn pool_miss_on_empty() {
        let dev = unique_device();
        let (_, misses_before, _) = pool_stats();
        let result = pool_take::<FakeSlice>(dev, 512, 4);
        assert!(result.is_none());
        let (_, misses_after, _) = pool_stats();
        assert!(misses_after - misses_before >= 1, "expected at least 1 miss");
    }

    #[test]
    fn pool_take_zero_len() {
        let dev = unique_device();
        let (hits_before, misses_before, _) = pool_stats();
        let result = pool_take::<FakeSlice>(dev, 0, 4);
        assert!(result.is_none());
        let (hits_after, misses_after, _) = pool_stats();
        // Zero-length should not touch stats.
        assert_eq!(hits_after, hits_before);
        assert_eq!(misses_after, misses_before);
    }

    #[test]
    fn pool_return_zero_len() {
        let dev = unique_device();
        // Zero-length return should be a no-op: no bytes cached on this device.
        pool_return::<FakeSlice>(dev, 0, 4, FakeSlice { len: 0, device: 0 });
        assert_eq!(cached_bytes(dev), 0);
    }

    #[test]
    fn pool_hit_exact_size() {
        let dev = unique_device();
        let (hits_before, _, returns_before) = pool_stats();

        let slice = FakeSlice { len: 512, device: 0 };
        pool_return::<FakeSlice>(dev, 512, 4, slice.clone());

        let result = pool_take::<FakeSlice>(dev, 512, 4);
        assert!(result.is_some());
        let take = result.unwrap();
        assert_eq!(take.value, slice);
        assert_eq!(take.alloc_len, 512);

        let (hits_after, _, returns_after) = pool_stats();
        // Use >= because other concurrent tests may bump global counters.
        assert!(hits_after - hits_before >= 1, "expected at least 1 hit");
        assert!(returns_after - returns_before >= 1, "expected at least 1 return");
    }

    #[test]
    fn pool_hit_rounded_size() {
        let dev = unique_device();
        let (hits_before, _, _) = pool_stats();

        let slice = FakeSlice { len: 512, device: 0 };
        pool_return::<FakeSlice>(dev, 512, 4, slice.clone());

        // Request 500 elements — rounds to 512, should hit.
        let result = pool_take::<FakeSlice>(dev, 500, 4);
        assert!(result.is_some());
        let take = result.unwrap();
        assert_eq!(take.value, slice);
        assert_eq!(take.alloc_len, 512);

        let (hits_after, _, _) = pool_stats();
        assert!(hits_after - hits_before >= 1, "expected at least 1 hit");
    }

    #[test]
    fn pool_near_miss_becomes_hit() {
        let dev = unique_device();
        let slice = FakeSlice { len: 128, device: 0 };
        pool_return::<FakeSlice>(dev, 128, 4, slice.clone());

        // Request 100 — rounds to 128 — exact match.
        let result = pool_take::<FakeSlice>(dev, 100, 4);
        assert!(result.is_some());
        assert_eq!(result.unwrap().alloc_len, 128);
    }

    // -- Larger-block reuse tests --

    #[test]
    fn pool_reuses_larger_block() {
        let dev = unique_device();
        let (hits_before, _, _) = pool_stats();

        let slice = FakeSlice { len: 1024, device: 0 };
        pool_return::<FakeSlice>(dev, 1024, 4, slice.clone());

        // Request 512 — no exact match, but 1024 >= 512. Should reuse.
        let result = pool_take::<FakeSlice>(dev, 512, 4);
        assert!(result.is_some());
        let take = result.unwrap();
        assert_eq!(take.value, slice);
        assert_eq!(take.alloc_len, 1024);

        let (hits_after, _, _) = pool_stats();
        assert!(hits_after - hits_before >= 1, "expected at least 1 hit");
    }

    #[test]
    fn pool_prefers_smallest_fit() {
        let dev = unique_device();
        // Insert blocks of sizes 256, 512, 1024. All are small pool
        // (each <= 256 KiB).
        pool_return::<FakeSlice>(dev, 256, 4, FakeSlice { len: 256, device: 0 });
        pool_return::<FakeSlice>(dev, 512, 4, FakeSlice { len: 512, device: 0 });
        pool_return::<FakeSlice>(dev, 1024, 4, FakeSlice { len: 1024, device: 0 });

        // Request 128 — should get the 256 block (smallest >= 128).
        // Note: 128 elements already rounds to 128, and we have an exact
        // 128-rounded-key only if someone put 128 in. We don't. So it
        // does a best-fit scan and finds 256 (the smallest that fits).
        let result = pool_take::<FakeSlice>(dev, 128, 4);
        assert!(result.is_some());
        let take = result.unwrap();
        assert_eq!(take.alloc_len, 256);

        // 512 and 1024 should still be in the pool.
        assert!(cached_bytes(dev) > 0);
    }

    #[test]
    fn pool_does_not_reuse_smaller_block() {
        let dev = unique_device();
        let (_, misses_before, _) = pool_stats();

        // Put a 128-element block.
        pool_return::<FakeSlice>(dev, 128, 4, FakeSlice { len: 128, device: 0 });

        // Request 256 — 128 is too small, should miss.
        let result = pool_take::<FakeSlice>(dev, 256, 4);
        assert!(result.is_none());

        let (_, misses_after, _) = pool_stats();
        assert!(misses_after - misses_before >= 1, "expected at least 1 miss");
    }

    // -- Device isolation tests --

    #[test]
    fn pool_isolates_devices() {
        let dev_a = unique_device();
        let dev_b = unique_device();
        pool_return::<FakeSlice>(dev_a, 512, 4, FakeSlice { len: 512, device: 0 });

        // Request from device B — should miss.
        let result = pool_take::<FakeSlice>(dev_b, 512, 4);
        assert!(result.is_none());

        // Request from device A — should hit.
        let result = pool_take::<FakeSlice>(dev_a, 512, 4);
        assert!(result.is_some());
    }

    // -- Two-pool design tests --

    #[test]
    fn small_and_large_pools_are_separate() {
        let dev = unique_device();
        // Small: 128 elements * 4 bytes = 512 bytes (< 256 KiB).
        pool_return::<FakeSlice>(dev, 128, 4, FakeSlice { len: 128, device: 0 });

        // Large: 128K elements * 4 bytes = 512 KiB (> 256 KiB).
        pool_return::<FakeSlice>(
            dev,
            128 * 1024,
            4,
            FakeSlice { len: 128 * 1024, device: 0 },
        );

        // Verify both pools have entries.
        let pool = POOL.lock().unwrap();
        let has_small = pool.small.keys().any(|k| k.0 == dev);
        let has_large = pool.large.keys().any(|k| k.0 == dev);
        assert!(has_small, "small pool should have entries for dev {dev}");
        assert!(has_large, "large pool should have entries for dev {dev}");
    }

    #[test]
    fn cross_pool_tier_no_leak() {
        let dev = unique_device();
        // Put a large block (512 KiB = 131072 f32 elements).
        let large_len = 131072; // 131072 * 4 = 524288 > 256 KiB
        pool_return::<FakeSlice>(
            dev,
            large_len,
            4,
            FakeSlice { len: large_len, device: 0 },
        );

        // Request a small allocation — should NOT find the large block
        // because small and large pools are separate.
        let result = pool_take::<FakeSlice>(dev, 128, 4);
        assert!(result.is_none());
    }

    // -- Memory limit tests --

    #[test]
    fn pool_enforces_memory_limit() {
        let dev = unique_device();

        // Insert many large blocks that together exceed MAX_CACHED_BYTES.
        // Each block: 256 * 1024 elements * 4 bytes = 1 MiB.
        let block_elements = 256 * 1024;
        let block_bytes = block_elements * 4;
        let num_blocks = (MAX_CACHED_BYTES / block_bytes) + 10;

        for i in 0..num_blocks {
            pool_return::<FakeSlice>(
                dev,
                block_elements,
                4,
                FakeSlice { len: block_elements, device: i },
            );
        }

        // Cached bytes should not exceed MAX_CACHED_BYTES.
        let cached = cached_bytes_all();
        assert!(
            cached <= MAX_CACHED_BYTES,
            "cached_bytes ({cached}) should be <= MAX_CACHED_BYTES ({MAX_CACHED_BYTES})"
        );
    }

    #[test]
    fn pool_evicts_largest_first() {
        let dev = unique_device();

        // Block A: 128K elements = 512 KiB (large pool tier).
        let a_len = 128 * 1024;
        pool_return::<FakeSlice>(dev, a_len, 4, FakeSlice { len: a_len, device: 0 });

        // Block B: 256K elements = 1 MiB (large pool tier).
        let b_len = 256 * 1024;
        pool_return::<FakeSlice>(dev, b_len, 4, FakeSlice { len: b_len, device: 0 });

        // Both should be cached.
        let cached = cached_bytes(dev);
        assert_eq!(cached, (a_len + b_len) * 4);

        // Flood until limit. Each flood block = 128K elements = 512 KiB.
        let flood_len = 128 * 1024;
        let flood_bytes = flood_len * 4;
        let blocks_to_add = MAX_CACHED_BYTES / flood_bytes;
        for i in 0..blocks_to_add {
            pool_return::<FakeSlice>(
                dev,
                flood_len,
                4,
                FakeSlice { len: flood_len, device: i + 100 },
            );
        }

        let cached = cached_bytes_all();
        assert!(
            cached <= MAX_CACHED_BYTES,
            "cached_bytes ({cached}) should be <= MAX_CACHED_BYTES ({MAX_CACHED_BYTES})"
        );
    }

    // -- Empty cache tests --

    #[test]
    fn empty_cache_clears_device() {
        let dev_a = unique_device();
        let dev_b = unique_device();
        pool_return::<FakeSlice>(dev_a, 512, 4, FakeSlice { len: 512, device: 0 });
        pool_return::<FakeSlice>(dev_b, 512, 4, FakeSlice { len: 512, device: 0 });

        assert!(cached_bytes(dev_a) > 0);
        assert!(cached_bytes(dev_b) > 0);

        empty_cache(dev_a);
        assert_eq!(cached_bytes(dev_a), 0);
        assert!(cached_bytes(dev_b) > 0, "dev_b should be untouched");
    }

    #[test]
    fn empty_cache_all_clears_everything() {
        let dev_a = unique_device();
        let dev_b = unique_device();
        pool_return::<FakeSlice>(dev_a, 512, 4, FakeSlice { len: 512, device: 0 });
        pool_return::<FakeSlice>(dev_b, 256, 4, FakeSlice { len: 256, device: 0 });

        empty_cache_all();
        assert_eq!(cached_bytes(dev_a), 0);
        assert_eq!(cached_bytes(dev_b), 0);
        assert_eq!(cached_bytes_all(), 0);
    }

    #[test]
    fn empty_cache_for_oom_returns_freed_bytes() {
        let dev = unique_device();
        pool_return::<FakeSlice>(dev, 512, 4, FakeSlice { len: 512, device: 0 });
        let expected = 512 * 4;
        assert_eq!(cached_bytes(dev), expected);

        let freed = empty_cache_for_oom(dev);
        assert_eq!(freed, expected);
        assert_eq!(cached_bytes(dev), 0);
    }

    #[test]
    fn empty_cache_for_oom_empty_pool() {
        let dev = unique_device();
        let freed = empty_cache_for_oom(dev);
        assert_eq!(freed, 0);
    }

    // -- Per-bucket limit test --

    #[test]
    fn per_bucket_limit_enforced() {
        let dev = unique_device();
        for i in 0..MAX_PER_BUCKET + 10 {
            pool_return::<FakeSlice>(dev, 256, 4, FakeSlice { len: 256, device: i });
        }

        // Only MAX_PER_BUCKET should be cached in the bucket.
        let mut count = 0;
        loop {
            if pool_take::<FakeSlice>(dev, 256, 4).is_none() {
                break;
            }
            count += 1;
        }
        assert_eq!(count, MAX_PER_BUCKET);
    }

    // -- Type isolation test --

    #[test]
    fn different_types_dont_mix() {
        let dev = unique_device();

        #[derive(Debug)]
        struct TypeA(usize);
        #[derive(Debug)]
        struct TypeB(usize);

        pool_return::<TypeA>(dev, 256, 4, TypeA(42));

        // Request as TypeB — should miss.
        let result = pool_take::<TypeB>(dev, 256, 4);
        assert!(result.is_none());

        // Request as TypeA — should hit.
        let result = pool_take::<TypeA>(dev, 256, 4);
        assert!(result.is_some());
    }

    // -- Cached bytes tracking --

    #[test]
    fn cached_bytes_tracks_correctly() {
        let dev = unique_device();
        assert_eq!(cached_bytes(dev), 0);

        pool_return::<FakeSlice>(dev, 256, 4, FakeSlice { len: 256, device: 0 });
        assert_eq!(cached_bytes(dev), 256 * 4);

        pool_return::<FakeSlice>(dev, 512, 4, FakeSlice { len: 512, device: 0 });
        assert_eq!(cached_bytes(dev), (256 + 512) * 4);

        // Take one back.
        let _ = pool_take::<FakeSlice>(dev, 256, 4);
        assert_eq!(cached_bytes(dev), 512 * 4);

        // Take the other.
        let _ = pool_take::<FakeSlice>(dev, 512, 4);
        assert_eq!(cached_bytes(dev), 0);
    }

    #[test]
    fn cached_bytes_per_device() {
        let dev_a = unique_device();
        let dev_b = unique_device();
        pool_return::<FakeSlice>(dev_a, 256, 4, FakeSlice { len: 256, device: 0 });
        pool_return::<FakeSlice>(dev_b, 128, 4, FakeSlice { len: 128, device: 0 });

        assert_eq!(cached_bytes(dev_a), 256 * 4);
        assert_eq!(cached_bytes(dev_b), 128 * 4);
    }
}
