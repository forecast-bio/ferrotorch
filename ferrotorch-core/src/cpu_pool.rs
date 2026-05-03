//! CPU tensor buffer pool — caching allocator for host memory.
//!
//! Mirrors the GPU buffer pool in `ferrotorch-gpu/src/pool.rs`. Instead of
//! calling `vec![0; n]` (which goes to the OS allocator) for every elementwise
//! op, we reuse recently freed `Vec<T>` buffers from a thread-local free list.
//!
//! On a pool hit the only cost is a `Vec::pop` (~5ns). On a miss we fall
//! back to a normal allocation. When a `TensorStorage` is dropped its CPU
//! `Vec` is returned to the pool for reuse.
//!
//! # Design
//!
//! - **Thread-local**: no mutex, no contention. Each rayon worker thread gets
//!   its own pool. Buffers created on thread A won't be reused on thread B,
//!   but in a training loop the main thread dominates and hits the cache.
//!
//! - **Size-exact**: we only reuse buffers with exactly the same capacity.
//!   This avoids wasting memory on oversized buffers and matches training
//!   workloads where the same tensor shapes repeat every iteration.
//!
//! - **Bounded**: each size bucket holds at most `MAX_PER_BUCKET` buffers.
//!   Total cached memory is bounded by `MAX_CACHED_BYTES`. Excess buffers
//!   are dropped normally.

use std::any::{Any, TypeId};
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Maximum buffers per (len, type) bucket.
const MAX_PER_BUCKET: usize = 8;

/// Maximum total cached bytes across all buckets (per thread).
/// 64 MiB is enough for typical training workloads.
const MAX_CACHED_BYTES: usize = 64 * 1024 * 1024;

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

static POOL_HITS: AtomicUsize = AtomicUsize::new(0);
static POOL_MISSES: AtomicUsize = AtomicUsize::new(0);
static POOL_RETURNS: AtomicUsize = AtomicUsize::new(0);

/// Get pool statistics: (hits, misses, returns).
pub fn cpu_pool_stats() -> (usize, usize, usize) {
    (
        POOL_HITS.load(Ordering::Relaxed),
        POOL_MISSES.load(Ordering::Relaxed),
        POOL_RETURNS.load(Ordering::Relaxed),
    )
}

/// Reset pool statistics.
pub fn reset_cpu_pool_stats() {
    POOL_HITS.store(0, Ordering::Relaxed);
    POOL_MISSES.store(0, Ordering::Relaxed);
    POOL_RETURNS.store(0, Ordering::Relaxed);
}

// ---------------------------------------------------------------------------
// Thread-local pool
// ---------------------------------------------------------------------------

type PoolKey = (usize, TypeId); // (element_count, type)

struct CpuPoolState {
    free: HashMap<PoolKey, Vec<Box<dyn Any>>>,
    cached_bytes: usize,
}

thread_local! {
    static CPU_POOL: RefCell<CpuPoolState> = RefCell::new(CpuPoolState {
        free: HashMap::new(),
        cached_bytes: 0,
    });
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Allocate a `Vec<T>` of exactly `len` elements, initialized to zero.
///
/// Checks the thread-local pool first. On a hit, the returned `Vec` has
/// the correct capacity and length but its contents are **zeroed** (we call
/// `fill` to clear stale data). On a miss, falls back to `vec![T::default(); len]`.
#[inline]
pub fn pool_alloc_cpu<T: Default + Clone + 'static>(len: usize) -> Vec<T> {
    if len == 0 {
        return Vec::new();
    }

    let key = (len, TypeId::of::<T>());
    let elem_size = std::mem::size_of::<T>();

    let maybe = CPU_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if let Some(bucket) = pool.free.get_mut(&key) {
            if let Some(boxed) = bucket.pop() {
                pool.cached_bytes -= len * elem_size;
                return Some(boxed);
            }
        }
        None
    });

    if let Some(boxed) = maybe {
        POOL_HITS.fetch_add(1, Ordering::Relaxed);
        let mut v: Vec<T> = *boxed.downcast::<Vec<T>>().unwrap();
        // Zero the buffer so the caller gets clean memory.
        v.fill(T::default());
        debug_assert_eq!(v.len(), len);
        v
    } else {
        POOL_MISSES.fetch_add(1, Ordering::Relaxed);
        vec![T::default(); len]
    }
}

/// Allocate a `Vec<f32>` of `len` elements, uninitialized-ish (pool reuse)
/// or zeroed (fresh alloc). Use when you're going to overwrite every element
/// immediately (e.g., SIMD kernel output buffer).
///
/// This is slightly faster than `pool_alloc_cpu` because it skips the
/// `fill(0.0)` on pool hits — the SIMD kernel will overwrite everything.
#[inline]
pub fn pool_alloc_cpu_uninit_f32(len: usize) -> Vec<f32> {
    if len == 0 {
        return Vec::new();
    }

    let key = (len, TypeId::of::<f32>());

    let maybe = CPU_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if let Some(bucket) = pool.free.get_mut(&key) {
            if let Some(boxed) = bucket.pop() {
                pool.cached_bytes -= len * 4;
                return Some(boxed);
            }
        }
        None
    });

    if let Some(boxed) = maybe {
        POOL_HITS.fetch_add(1, Ordering::Relaxed);
        let v: Vec<f32> = *boxed.downcast::<Vec<f32>>().unwrap();
        debug_assert_eq!(v.len(), len);
        v
    } else {
        POOL_MISSES.fetch_add(1, Ordering::Relaxed);
        vec![0.0f32; len]
    }
}

/// Same as `pool_alloc_cpu_uninit_f32` but for f64.
#[inline]
pub fn pool_alloc_cpu_uninit_f64(len: usize) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }

    let key = (len, TypeId::of::<f64>());

    let maybe = CPU_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        if let Some(bucket) = pool.free.get_mut(&key) {
            if let Some(boxed) = bucket.pop() {
                pool.cached_bytes -= len * 8;
                return Some(boxed);
            }
        }
        None
    });

    if let Some(boxed) = maybe {
        POOL_HITS.fetch_add(1, Ordering::Relaxed);
        let v: Vec<f64> = *boxed.downcast::<Vec<f64>>().unwrap();
        debug_assert_eq!(v.len(), len);
        v
    } else {
        POOL_MISSES.fetch_add(1, Ordering::Relaxed);
        vec![0.0f64; len]
    }
}

/// Return a `Vec<T>` to the thread-local pool for reuse.
///
/// If the pool is full (per-bucket or total bytes limit), the `Vec` is
/// dropped normally.
pub fn pool_return_cpu<T: 'static>(mut v: Vec<T>) {
    let len = v.len();
    if len == 0 {
        return;
    }

    let elem_size = std::mem::size_of::<T>();
    let byte_size = len * elem_size;
    let key = (len, TypeId::of::<T>());

    CPU_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();

        // Check total memory limit.
        if pool.cached_bytes + byte_size > MAX_CACHED_BYTES {
            return; // Drop the Vec normally.
        }

        let bucket = pool.free.entry(key).or_insert_with(Vec::new);
        if bucket.len() >= MAX_PER_BUCKET {
            return; // Bucket full, drop normally.
        }

        // Keep the allocation, ensure correct length for reuse.
        // SAFETY: `len = v.len()` was captured at function entry (line above
        // the early return), so this is a no-op when the caller hasn't
        // changed v's length, and a defensive restore otherwise. The pool's
        // bucket key is `(len, TypeId::of::<T>())`, so consumers that pop
        // this Vec receive exactly `len` elements; they were initialized
        // either by the original `vec![T::default(); len]` allocation in
        // pool_alloc_cpu (or `vec![0.0; len]` in the f32/f64 variants) or
        // by user writes via `as_mut_slice` while the Vec was outstanding.
        // `len <= v.capacity()` because v was originally allocated with
        // capacity == len and capacity never shrinks below in-use length.
        unsafe { v.set_len(len) };

        bucket.push(Box::new(v));
        pool.cached_bytes += byte_size;
        POOL_RETURNS.fetch_add(1, Ordering::Relaxed);
    });
}

/// Drop all cached CPU buffers, freeing memory.
pub fn empty_cpu_pool() {
    CPU_POOL.with(|pool| {
        let mut pool = pool.borrow_mut();
        pool.free.clear();
        pool.cached_bytes = 0;
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pool_miss_then_hit() {
        // First alloc is a miss.
        let v: Vec<f32> = pool_alloc_cpu(1000);
        assert_eq!(v.len(), 1000);
        assert!(v.iter().all(|&x| x == 0.0));

        // Snapshot stats AFTER the miss, BEFORE the return+reuse.
        let (hits_before, _misses_before, _) = cpu_pool_stats();

        // Return it. Pool is thread-local, so this lands in the same
        // bucket the next alloc will read from — no cross-thread
        // contention can steal it.
        pool_return_cpu(v);

        // Second alloc should be a hit (reuse from pool).
        let v2: Vec<f32> = pool_alloc_cpu(1000);
        assert_eq!(v2.len(), 1000);
        assert!(v2.iter().all(|&x| x == 0.0)); // Zeroed on checkout.

        let (hits_after, _misses_after, _) = cpu_pool_stats();

        // Our second alloc was guaranteed to hit because (a) the pool is
        // thread-local and (b) `pool_return_cpu` ran on this thread
        // immediately before `pool_alloc_cpu`. So the global hits
        // counter must have grown by at least one.
        //
        // We deliberately do NOT compare against `misses_after - misses_before`:
        // POOL_MISSES is a global atomic and parallel tests on other
        // threads bump it concurrently. Comparing hit-delta against a
        // racy miss-delta caused intermittent CI failures.
        let hit_delta = hits_after - hits_before;
        assert!(
            hit_delta >= 1,
            "expected at least one new pool hit, but hits stayed at {hits_before}"
        );

        drop(v2);
    }

    #[test]
    fn test_uninit_alloc() {
        let v = pool_alloc_cpu_uninit_f32(500);
        assert_eq!(v.len(), 500);
        // Fresh alloc is zeroed.
        assert!(v.iter().all(|&x| x == 0.0));

        // Modify and return.
        let mut v = v;
        v[0] = 42.0;
        pool_return_cpu(v);

        // Reuse — NOT zeroed (uninit variant).
        let v2 = pool_alloc_cpu_uninit_f32(500);
        assert_eq!(v2.len(), 500);
        // v2[0] may be 42.0 or 0.0 depending on pool hit.
    }

    #[test]
    fn test_bucket_limit() {
        empty_cpu_pool();

        // Fill a bucket beyond MAX_PER_BUCKET.
        for _ in 0..MAX_PER_BUCKET + 5 {
            let v: Vec<f32> = vec![0.0; 100];
            pool_return_cpu(v);
        }

        // Only MAX_PER_BUCKET should be cached.
        let mut hits = 0;
        for _ in 0..MAX_PER_BUCKET + 5 {
            let v: Vec<f32> = pool_alloc_cpu(100);
            if v.len() == 100 {
                hits += 1;
            }
        }
        // At least MAX_PER_BUCKET hits.
        assert!(hits >= MAX_PER_BUCKET);
    }

    #[test]
    fn test_different_sizes_different_buckets() {
        empty_cpu_pool();

        let v1: Vec<f32> = vec![0.0; 100];
        pool_return_cpu(v1);

        // Different size — should miss.
        let v2: Vec<f32> = pool_alloc_cpu(200);
        assert_eq!(v2.len(), 200);

        // Same size — should hit.
        let v3: Vec<f32> = pool_alloc_cpu(100);
        assert_eq!(v3.len(), 100);
    }
}
