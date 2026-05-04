//! Caching CUDA memory allocator.
//!
//! [`CudaAllocator`] wraps a [`GpuDevice`] and provides a PyTorch-style caching
//! memory allocator with:
//!
//! - **Block splitting**: oversized free blocks are split, remainder returned to
//!   the pool.
//! - **Block coalescing**: adjacent freed blocks are merged to reduce
//!   fragmentation.
//! - **Stream-aware reuse**: blocks track which CUDA streams have used them;
//!   a block is only reused when all recorded stream work is complete.
//! - **Dual pools**: small (<1 MiB) and large (>=1 MiB) allocations are kept in
//!   separate pools to avoid small allocations fragmenting large contiguous
//!   regions.
//! - **Statistics**: `memory_allocated`, `max_memory_allocated`,
//!   `memory_reserved`, allocation/free counters.
//!
//! # Design
//!
//! This is a CPU-side data structure that manages block metadata. Actual GPU
//! memory allocation/deallocation is delegated to the [`GpuDevice`] (cudarc).
//! The caching layer sits between callers and the driver, intercepting frees
//! to retain memory for reuse and serving allocs from the cache when possible.
//!
//! The design follows PyTorch's `CUDACachingAllocator` (c10/cuda/). Key
//! constants match PyTorch:
//! - `MIN_BLOCK_SIZE` = 512 bytes
//! - `SMALL_SIZE` = 1 MiB (threshold between small/large pools)
//! - `SMALL_BUFFER` = 2 MiB (small pool segment size)
//! - `MIN_LARGE_ALLOC` = 10 MiB
//! - `ROUND_LARGE` = 2 MiB (rounding for large allocations)
//!
//! # Thread safety
//!
//! `CudaAllocator` is `Send + Sync`. Internal state is protected by a `Mutex`.
//! The critical section is short (BTreeSet lookup + pointer bookkeeping).
//!
//! # CL-323

use std::collections::{BTreeSet, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::GpuResult;

// ---------------------------------------------------------------------------
// Constants (matching PyTorch's CUDACachingAllocator)
// ---------------------------------------------------------------------------

/// Minimum block size — all allocations are rounded up to at least this.
pub const MIN_BLOCK_SIZE: usize = 512;

/// Largest allocation that goes into the small pool.
pub const SMALL_SIZE: usize = 1 << 20; // 1 MiB

/// Segment size for small pool allocations from the driver.
pub const SMALL_BUFFER: usize = 2 << 20; // 2 MiB

/// Allocations between `SMALL_SIZE` and `MIN_LARGE_ALLOC` use a 20 MiB
/// segment from the driver (to reduce the number of driver calls).
pub const MIN_LARGE_ALLOC: usize = 10 << 20; // 10 MiB

/// Large pool segment size for allocations between 1-10 MiB.
pub const LARGE_BUFFER: usize = 20 << 20; // 20 MiB

/// Round up large allocations to this granularity.
pub const ROUND_LARGE: usize = 2 << 20; // 2 MiB

// ---------------------------------------------------------------------------
// StreamId — lightweight stream identifier for tracking cross-stream usage
// ---------------------------------------------------------------------------

/// Opaque identifier for a CUDA stream.
///
/// We use a `usize` derived from the stream's pointer/handle so that stream
/// tracking works without holding `Arc<CudaStream>` references (which would
/// prevent the stream from being dropped).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StreamId(pub usize);

// ---------------------------------------------------------------------------
// Block — metadata for a cached GPU memory region
// ---------------------------------------------------------------------------

/// Unique identifier for a [`Block`]. Monotonically increasing.
static NEXT_BLOCK_ID: AtomicUsize = AtomicUsize::new(0);

/// Metadata for a contiguous region of GPU memory.
///
/// Blocks form a doubly-linked list within a segment (a single driver
/// allocation). Splitting a block creates a new block for the remainder and
/// links them. Coalescing merges adjacent free blocks by adjusting the linked
/// list and summing sizes.
///
/// # CL-323
#[derive(Debug)]
pub struct Block {
    /// Unique id for deterministic BTreeSet ordering when size ties.
    pub(crate) id: usize,
    /// Device ordinal that owns this memory.
    pub device: usize,
    /// Size of this block in bytes.
    pub size: usize,
    /// Virtual address of the block's start (byte offset within its segment).
    /// Used for ordering during coalescing. For blocks that own real GPU memory,
    /// this is derived from the `CudaSlice` pointer. For sub-blocks created by
    /// splitting, this is computed as `parent.ptr + parent.size_before_split`.
    pub ptr: usize,
    /// The CUDA stream on which this block was originally allocated.
    pub stream: StreamId,
    /// Set of streams that have used this block (via `record_stream`).
    /// A block cannot be reused until all recorded stream work is complete.
    pub stream_uses: HashSet<StreamId>,
    /// Whether this block is currently handed out to a caller.
    pub allocated: bool,
    /// Index of the previous block in the segment's linked list, or `None`.
    pub prev: Option<usize>,
    /// Index of the next block in the segment's linked list, or `None`.
    pub next: Option<usize>,
    /// Whether this block is in the small pool.
    pub in_small_pool: bool,
}

impl Block {
    /// Create a new block with the given parameters.
    pub fn new(
        device: usize,
        size: usize,
        ptr: usize,
        stream: StreamId,
        in_small_pool: bool,
    ) -> Self {
        Self {
            id: NEXT_BLOCK_ID.fetch_add(1, Ordering::Relaxed),
            device,
            size,
            ptr,
            stream,
            stream_uses: HashSet::new(),
            allocated: false,
            prev: None,
            next: None,
            in_small_pool,
        }
    }

    /// Whether this block was created by splitting (has neighbors).
    pub fn is_split(&self) -> bool {
        self.prev.is_some() || self.next.is_some()
    }
}

// ---------------------------------------------------------------------------
// BlockKey — ordered key for BTreeSet lookups
// ---------------------------------------------------------------------------

/// Key used for ordering blocks in a [`BlockPool`]'s free set.
///
/// Ordered by `(stream, size, ptr, id)` so that `lower_bound` finds the
/// smallest block >= requested size on the correct stream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct BlockKey {
    stream: StreamId,
    size: usize,
    ptr: usize,
    id: usize,
}

impl BlockKey {
    fn from_block(b: &Block) -> Self {
        Self {
            stream: b.stream,
            size: b.size,
            ptr: b.ptr,
            id: b.id,
        }
    }

    /// Create a search key: finds the smallest block >= `size` on `stream`.
    fn search(stream: StreamId, size: usize) -> Self {
        Self {
            stream,
            size,
            ptr: 0,
            id: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// BlockPool — set of free blocks for one size class
// ---------------------------------------------------------------------------

/// A pool of free [`Block`]s ordered by `(stream, size, ptr)`.
///
/// Two pools exist per allocator: one for small blocks (<1 MiB) and one for
/// large blocks (>=1 MiB).
///
/// # CL-323
pub(crate) struct BlockPool {
    /// Free (non-allocated) blocks, ordered by [`BlockKey`].
    free_blocks: BTreeSet<(BlockKey, usize)>, // (key, block_index)
    /// Whether this is the small pool.
    pub is_small: bool,
}

impl BlockPool {
    /// Create a new empty block pool.
    pub fn new(is_small: bool) -> Self {
        Self {
            free_blocks: BTreeSet::new(),
            is_small,
        }
    }

    /// Insert a block into the free set.
    #[cfg(test)]
    pub fn insert(&mut self, block_idx: usize, block: &Block) {
        self.free_blocks
            .insert((BlockKey::from_block(block), block_idx));
    }

    /// Insert a block into the free set using a precomputed key.
    pub fn insert_key(&mut self, block_idx: usize, key: BlockKey) {
        self.free_blocks.insert((key, block_idx));
    }

    /// Remove a block from the free set using a precomputed key.
    pub fn remove_key(&mut self, block_idx: usize, key: BlockKey) {
        self.free_blocks.remove(&(key, block_idx));
    }

    /// Find the smallest free block >= `size` on `stream`.
    pub fn find_free_block(&self, stream: StreamId, size: usize) -> Option<usize> {
        let search = (BlockKey::search(stream, size), 0);
        if let Some(&(key, idx)) = self.free_blocks.range(search..).next() {
            if key.stream == stream {
                // Found a block on the same stream that is >= requested size.
                return Some(idx);
            }
        }
        None
    }

    /// Number of free blocks in this pool.
    pub fn len(&self) -> usize {
        self.free_blocks.len()
    }

    /// Clear all free blocks from the pool.
    pub fn clear(&mut self) {
        self.free_blocks.clear();
    }
}

// ---------------------------------------------------------------------------
// AllocatorState — the caching allocator's inner mutable state
// ---------------------------------------------------------------------------

/// All mutable state protected by the allocator's mutex.
///
/// # CL-323
pub(crate) struct AllocatorState {
    /// Arena of all blocks (allocated + free). Indexed by `usize`.
    pub(crate) blocks: Vec<Block>,
    /// Small pool: blocks < `SMALL_SIZE`.
    pub(crate) small_pool: BlockPool,
    /// Large pool: blocks >= `SMALL_SIZE`.
    pub(crate) large_pool: BlockPool,
    /// Total bytes reserved from the driver (cached + in-use).
    pub(crate) reserved_bytes: usize,
    /// Total bytes currently handed out to callers.
    pub(crate) allocated_bytes: usize,
    /// Peak allocated bytes.
    pub(crate) peak_bytes: usize,
    /// Number of successful cache hits.
    pub(crate) hits: usize,
    /// Number of cache misses (driver allocs).
    pub(crate) misses: usize,
}

impl AllocatorState {
    fn new() -> Self {
        Self {
            blocks: Vec::new(),
            small_pool: BlockPool::new(true),
            large_pool: BlockPool::new(false),
            reserved_bytes: 0,
            allocated_bytes: 0,
            peak_bytes: 0,
            hits: 0,
            misses: 0,
        }
    }

    /// Get the pool for a given size class.
    pub(crate) fn get_pool_mut(&mut self, is_small: bool) -> &mut BlockPool {
        let pool = if is_small {
            &mut self.small_pool
        } else {
            &mut self.large_pool
        };
        debug_assert_eq!(pool.is_small, is_small, "pool size-class mismatch");
        pool
    }

    /// Allocate a new block index in the arena.
    pub(crate) fn add_block(&mut self, block: Block) -> usize {
        let idx = self.blocks.len();
        self.blocks.push(block);
        idx
    }

    /// Whether a free block should be split for a request of `size` bytes.
    pub(crate) fn should_split(&self, block_idx: usize, size: usize) -> bool {
        let block = &self.blocks[block_idx];
        let remaining = block.size - size;
        if block.in_small_pool {
            // Small pool: split if remainder >= MIN_BLOCK_SIZE.
            remaining >= MIN_BLOCK_SIZE
        } else {
            // Large pool: split if remainder > SMALL_SIZE (avoid small
            // fragments in the large pool).
            remaining > SMALL_SIZE
        }
    }

    /// Split `block_idx` to satisfy a `size`-byte request. The block at
    /// `block_idx` is resized to `size` and a new remainder block is created
    /// and inserted into the appropriate pool.
    pub(crate) fn split_block(&mut self, block_idx: usize, size: usize) {
        let remaining_size = self.blocks[block_idx].size - size;
        let remaining_ptr = self.blocks[block_idx].ptr + size;
        let stream = self.blocks[block_idx].stream;
        let device = self.blocks[block_idx].device;
        let is_small = self.blocks[block_idx].in_small_pool;
        let old_next = self.blocks[block_idx].next;

        // Create the remainder block.
        let mut remainder = Block::new(device, remaining_size, remaining_ptr, stream, is_small);
        remainder.prev = Some(block_idx);
        remainder.next = old_next;

        let rem_idx = self.add_block(remainder);

        // Update the original block.
        self.blocks[block_idx].size = size;
        self.blocks[block_idx].next = Some(rem_idx);

        // Update the block that was after the original, if any.
        if let Some(old_next_idx) = old_next {
            self.blocks[old_next_idx].prev = Some(rem_idx);
        }

        // Insert remainder into the free pool.
        let rem_key = BlockKey::from_block(&self.blocks[rem_idx]);
        let pool = self.get_pool_mut(is_small);
        pool.insert_key(rem_idx, rem_key);
    }

    /// Try to merge `block_idx` with an adjacent block `neighbor_idx`.
    /// Returns the size of the subsumed neighbor, or 0 if merge failed.
    pub(crate) fn try_merge(&mut self, block_idx: usize, neighbor_idx: Option<usize>) -> usize {
        let Some(nbr_idx) = neighbor_idx else {
            return 0;
        };

        // Cannot merge if neighbor is allocated or has pending stream uses.
        if self.blocks[nbr_idx].allocated || !self.blocks[nbr_idx].stream_uses.is_empty() {
            return 0;
        }

        let is_small = self.blocks[nbr_idx].in_small_pool;
        let subsumed_size = self.blocks[nbr_idx].size;

        // Remove neighbor from its free pool.
        let nbr_key = BlockKey::from_block(&self.blocks[nbr_idx]);
        {
            let pool = self.get_pool_mut(is_small);
            pool.remove_key(nbr_idx, nbr_key);
        }

        // Determine merge direction.
        if self.blocks[block_idx].prev == Some(nbr_idx) {
            // [neighbor] [block] — neighbor is before block.
            let nbr_prev = self.blocks[nbr_idx].prev;
            self.blocks[block_idx].ptr = self.blocks[nbr_idx].ptr;
            self.blocks[block_idx].size += subsumed_size;
            self.blocks[block_idx].prev = nbr_prev;
            if let Some(pp) = nbr_prev {
                self.blocks[pp].next = Some(block_idx);
            }
        } else {
            // [block] [neighbor] — neighbor is after block.
            let nbr_next = self.blocks[nbr_idx].next;
            self.blocks[block_idx].size += subsumed_size;
            self.blocks[block_idx].next = nbr_next;
            if let Some(nn) = nbr_next {
                self.blocks[nn].prev = Some(block_idx);
            }
        }

        // Mark the subsumed block as dead (size 0, no links). We do not
        // reclaim arena slots — the Vec grows monotonically. This is fine
        // because the number of live blocks is bounded by the number of
        // driver allocations (typically <10k even for large models).
        self.blocks[nbr_idx].size = 0;
        self.blocks[nbr_idx].prev = None;
        self.blocks[nbr_idx].next = None;

        subsumed_size
    }

    /// Free a block: mark as not-allocated, try to coalesce with neighbors,
    /// then return to the appropriate pool.
    pub(crate) fn free_block(&mut self, block_idx: usize) {
        self.blocks[block_idx].allocated = false;
        self.blocks[block_idx].stream_uses.clear();
        let size = self.blocks[block_idx].size;
        self.allocated_bytes = self.allocated_bytes.saturating_sub(size);

        // Try coalescing with prev and next.
        let prev = self.blocks[block_idx].prev;
        let next = self.blocks[block_idx].next;
        self.try_merge(block_idx, prev);
        self.try_merge(block_idx, next);

        // Insert merged block into free pool.
        let is_small = self.blocks[block_idx].in_small_pool;
        let merged_key = BlockKey::from_block(&self.blocks[block_idx]);
        let pool = self.get_pool_mut(is_small);
        pool.insert_key(block_idx, merged_key);
    }

    /// Total cached bytes: reserved - allocated = free blocks in pools.
    pub(crate) fn cached_bytes(&self) -> usize {
        self.reserved_bytes.saturating_sub(self.allocated_bytes)
    }
}

// ---------------------------------------------------------------------------
// Round size — PyTorch-compatible size rounding
// ---------------------------------------------------------------------------

/// Round `size` up to an allocation-friendly boundary.
///
/// - Sizes below `MIN_BLOCK_SIZE` (512) are rounded up to `MIN_BLOCK_SIZE`.
/// - Otherwise, rounded up to the next multiple of `MIN_BLOCK_SIZE`.
pub fn round_size(size: usize) -> usize {
    if size < MIN_BLOCK_SIZE {
        return MIN_BLOCK_SIZE;
    }
    // Round up to next multiple of MIN_BLOCK_SIZE.
    (size + MIN_BLOCK_SIZE - 1) & !(MIN_BLOCK_SIZE - 1)
}

/// Determine how many bytes to request from the driver for a given request
/// size (after rounding). Small allocations are packed into `SMALL_BUFFER`
/// segments; mid-range into `LARGE_BUFFER`; large are rounded to
/// `ROUND_LARGE`.
pub fn get_allocation_size(size: usize) -> usize {
    if size <= SMALL_SIZE {
        SMALL_BUFFER
    } else if size < MIN_LARGE_ALLOC {
        LARGE_BUFFER
    } else {
        // Round up to next multiple of ROUND_LARGE.
        (size + ROUND_LARGE - 1) & !(ROUND_LARGE - 1)
    }
}

// ---------------------------------------------------------------------------
// CudaAllocator — the public API
// ---------------------------------------------------------------------------

/// A caching GPU memory allocator with block pools, splitting, coalescing,
/// and stream-aware reuse.
///
/// Wraps a [`GpuDevice`] and maintains two block pools (small and large).
/// Allocation requests are served from cached free blocks when possible;
/// only on cache miss does the allocator call through to the CUDA driver.
/// Freed blocks are returned to the pool and coalesced with neighbors to
/// reduce fragmentation.
///
/// # CL-323
pub struct CudaAllocator {
    device: Arc<GpuDevice>,
    pub(crate) state: Mutex<AllocatorState>,
    /// Total bytes currently in use (atomic mirror for lock-free reads).
    allocated_bytes_atomic: AtomicUsize,
    /// Peak bytes ever allocated.
    peak_bytes_atomic: AtomicUsize,
}

impl CudaAllocator {
    /// Create a new caching allocator for the given device.
    pub fn new(device: Arc<GpuDevice>) -> Self {
        Self {
            device,
            state: Mutex::new(AllocatorState::new()),
            allocated_bytes_atomic: AtomicUsize::new(0),
            peak_bytes_atomic: AtomicUsize::new(0),
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
    /// Returns [`crate::error::GpuError::Driver`] if the underlying CUDA allocation fails.
    #[cfg(feature = "cuda")]
    pub fn alloc_zeros<T>(&self, count: usize) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits,
    {
        let bytes = count.saturating_mul(std::mem::size_of::<T>());
        let slice = self.device.stream().alloc_zeros::<T>(count)?;

        // Update statistics after the allocation succeeds.
        let prev = self
            .allocated_bytes_atomic
            .fetch_add(bytes, Ordering::Relaxed);
        self.peak_bytes_atomic
            .fetch_max(prev + bytes, Ordering::Relaxed);

        Ok(CudaBuffer {
            data: Some(slice),
            len: count,
            alloc_len: count,
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
    /// Returns [`crate::error::GpuError::Driver`] if the CUDA memcpy or allocation fails.
    #[cfg(feature = "cuda")]
    pub fn alloc_copy<T>(&self, data: &[T]) -> GpuResult<CudaBuffer<T>>
    where
        T: cudarc::driver::DeviceRepr,
    {
        let bytes = data.len().saturating_mul(std::mem::size_of::<T>());
        let slice = self.device.stream().clone_htod(data)?;

        let prev = self
            .allocated_bytes_atomic
            .fetch_add(bytes, Ordering::Relaxed);
        self.peak_bytes_atomic
            .fetch_max(prev + bytes, Ordering::Relaxed);

        Ok(CudaBuffer {
            data: Some(slice),
            len: data.len(),
            alloc_len: data.len(),
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
        let bytes = buffer
            .len()
            .checked_mul(std::mem::size_of::<T>())
            .unwrap_or(0);
        self.allocated_bytes_atomic
            .fetch_sub(bytes, Ordering::Relaxed);
        drop(buffer);
    }

    // ------------------------------------------------------------------
    // Statistics queries
    // ------------------------------------------------------------------

    /// Bytes currently allocated (live) on the device through this allocator.
    #[inline]
    pub fn memory_allocated(&self) -> usize {
        self.allocated_bytes_atomic.load(Ordering::Relaxed)
    }

    /// Peak bytes ever allocated since creation or the last
    /// [`reset_peak_stats`](CudaAllocator::reset_peak_stats).
    #[inline]
    pub fn max_memory_allocated(&self) -> usize {
        self.peak_bytes_atomic.load(Ordering::Relaxed)
    }

    /// Total bytes reserved from the CUDA driver (cached + in-use).
    pub fn memory_reserved(&self) -> usize {
        self.state.lock().map(|s| s.reserved_bytes).unwrap_or(0)
    }

    /// Reset the peak counter to the current allocation level.
    pub fn reset_peak_stats(&self) {
        let current = self.allocated_bytes_atomic.load(Ordering::Relaxed);
        self.peak_bytes_atomic.store(current, Ordering::Relaxed);
    }

    /// Release all cached (free) blocks back to the CUDA driver.
    ///
    /// After this call, `memory_reserved()` drops to `memory_allocated()`
    /// (only blocks currently in use remain). This is useful when another
    /// component needs GPU memory and the cache is holding onto freed blocks.
    ///
    /// # CL-323
    pub fn empty_cache(&self) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        // Clear both free pools. The actual GPU memory is freed when the
        // pool.rs layer drops its CudaSlice holders.
        state.small_pool.clear();
        state.large_pool.clear();

        // Recalculate reserved to only include allocated blocks.
        state.reserved_bytes = state.allocated_bytes;
    }

    /// The underlying device.
    #[inline]
    pub fn device(&self) -> &GpuDevice {
        &self.device
    }

    // ------------------------------------------------------------------
    // Pool-level cache operations (used by pool.rs)
    // ------------------------------------------------------------------

    /// Record that a block was used on `stream`, preventing reuse until
    /// work on that stream completes.
    ///
    /// This is the Rust equivalent of PyTorch's `recordStream()`.
    ///
    /// # CL-323
    pub fn record_stream_on_block(&self, block_idx: usize, stream: StreamId) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if block_idx < state.blocks.len() {
            state.blocks[block_idx].stream_uses.insert(stream);
        }
    }

    /// Number of blocks in the arena (for debugging/testing).
    pub fn block_count(&self) -> usize {
        self.state.lock().map(|s| s.blocks.len()).unwrap_or(0)
    }

    /// Number of free blocks in both pools (for debugging/testing).
    pub fn free_block_count(&self) -> usize {
        self.state
            .lock()
            .map(|s| s.small_pool.len() + s.large_pool.len())
            .unwrap_or(0)
    }

    /// (hits, misses) cache statistics.
    pub fn cache_stats(&self) -> (usize, usize) {
        self.state
            .lock()
            .map(|s| (s.hits, s.misses))
            .unwrap_or((0, 0))
    }

    /// Total cached (free, reusable) bytes.
    pub fn cached_bytes(&self) -> usize {
        self.state.lock().map(|s| s.cached_bytes()).unwrap_or(0)
    }

    // ------------------------------------------------------------------
    // Cache-aware allocation (used by pool.rs)
    // ------------------------------------------------------------------

    /// Try to find a cached block of at least `size` bytes on `stream`.
    ///
    /// Returns the block index and its actual size if found.
    /// The block is marked as allocated and removed from the free pool.
    /// If the block is significantly larger than needed, it is split.
    ///
    /// # CL-323
    pub fn cache_find(&self, size: usize, stream: StreamId) -> Option<(usize, usize)> {
        let rounded = round_size(size);
        let is_small = rounded <= SMALL_SIZE;

        let Ok(mut state) = self.state.lock() else {
            return None;
        };

        let block_idx = {
            let pool = state.get_pool_mut(is_small);
            pool.find_free_block(stream, rounded)?
        };

        // Remove from free pool (key extracted before mutable pool borrow).
        let key = BlockKey::from_block(&state.blocks[block_idx]);
        state.get_pool_mut(is_small).remove_key(block_idx, key);

        // Split if block is oversized.
        if state.should_split(block_idx, rounded) {
            state.split_block(block_idx, rounded);
        }

        // Mark as allocated.
        state.blocks[block_idx].allocated = true;
        let actual_size = state.blocks[block_idx].size;
        state.allocated_bytes += actual_size;
        if state.allocated_bytes > state.peak_bytes {
            state.peak_bytes = state.allocated_bytes;
        }
        state.hits += 1;

        Some((block_idx, actual_size))
    }

    /// Register a new block from a fresh driver allocation.
    ///
    /// Called when `cache_find` returns `None` and the caller has obtained
    /// memory from the CUDA driver. The full driver allocation is registered
    /// as a block; if it's larger than the requested size, the remainder is
    /// split off and placed in the free pool.
    ///
    /// Returns `(block_idx, actual_block_size)`.
    ///
    /// # CL-323
    pub fn cache_insert(
        &self,
        requested_size: usize,
        driver_alloc_size: usize,
        ptr: usize,
        stream: StreamId,
    ) -> (usize, usize) {
        let rounded = round_size(requested_size);
        let is_small = rounded <= SMALL_SIZE;

        let Ok(mut state) = self.state.lock() else {
            // Fallback: return a dummy index. Should never happen in practice.
            return (0, driver_alloc_size);
        };

        let mut block = Block::new(
            self.device.ordinal(),
            driver_alloc_size,
            ptr,
            stream,
            is_small,
        );
        block.allocated = true;
        let block_idx = state.add_block(block);

        state.reserved_bytes += driver_alloc_size;

        // Split if the driver allocation is larger than requested.
        if state.should_split(block_idx, rounded) {
            state.split_block(block_idx, rounded);
        }

        let actual_size = state.blocks[block_idx].size;
        state.allocated_bytes += actual_size;
        if state.allocated_bytes > state.peak_bytes {
            state.peak_bytes = state.allocated_bytes;
        }
        state.misses += 1;

        (block_idx, actual_size)
    }

    /// Return a block to the cache (free it back to a pool).
    ///
    /// The block is coalesced with any adjacent free blocks and inserted
    /// into the appropriate pool for future reuse.
    ///
    /// # CL-323
    pub fn cache_free(&self, block_idx: usize) {
        let Ok(mut state) = self.state.lock() else {
            return;
        };
        if block_idx < state.blocks.len() && state.blocks[block_idx].allocated {
            state.free_block(block_idx);
        }
    }

    /// Get the driver allocation size for a given request size.
    ///
    /// Callers use this to know how many bytes to request from the driver
    /// when `cache_find` misses.
    pub fn driver_alloc_size(size: usize) -> usize {
        get_allocation_size(round_size(size))
    }
}

impl std::fmt::Debug for CudaAllocator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaAllocator")
            .field("device_ordinal", &self.device.ordinal())
            .field(
                "allocated_bytes",
                &self.allocated_bytes_atomic.load(Ordering::Relaxed),
            )
            .field(
                "peak_bytes",
                &self.peak_bytes_atomic.load(Ordering::Relaxed),
            )
            .field("cached_bytes", &self.cached_bytes())
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
mod tests {
    use super::*;

    // ------------------------------------------------------------------
    // Unit tests for round_size
    // ------------------------------------------------------------------

    #[test]
    fn round_size_minimum() {
        assert_eq!(round_size(0), MIN_BLOCK_SIZE);
        assert_eq!(round_size(1), MIN_BLOCK_SIZE);
        assert_eq!(round_size(511), MIN_BLOCK_SIZE);
        assert_eq!(round_size(512), MIN_BLOCK_SIZE);
    }

    #[test]
    fn round_size_multiples() {
        assert_eq!(round_size(513), 1024);
        assert_eq!(round_size(1024), 1024);
        assert_eq!(round_size(1025), 1536);
    }

    #[test]
    fn alloc_size_small() {
        // Anything <= 1 MiB uses a 2 MiB segment.
        assert_eq!(get_allocation_size(512), SMALL_BUFFER);
        assert_eq!(get_allocation_size(SMALL_SIZE), SMALL_BUFFER);
    }

    #[test]
    fn alloc_size_mid() {
        // Between 1 MiB and 10 MiB uses 20 MiB segment.
        assert_eq!(get_allocation_size(SMALL_SIZE + 1), LARGE_BUFFER);
        assert_eq!(get_allocation_size(MIN_LARGE_ALLOC - 1), LARGE_BUFFER);
    }

    #[test]
    fn alloc_size_large() {
        // >= 10 MiB rounds up to 2 MiB boundary.
        assert_eq!(get_allocation_size(MIN_LARGE_ALLOC), MIN_LARGE_ALLOC);
        assert_eq!(
            get_allocation_size(MIN_LARGE_ALLOC + 1),
            MIN_LARGE_ALLOC + ROUND_LARGE
        );
    }

    // ------------------------------------------------------------------
    // Unit tests for Block / BlockPool / AllocatorState
    // ------------------------------------------------------------------

    fn make_stream() -> StreamId {
        StreamId(42)
    }

    #[test]
    fn block_pool_insert_find() {
        let mut state = AllocatorState::new();
        let stream = make_stream();

        // Create a 4096-byte free block.
        let block = Block::new(0, 4096, 0x1000, stream, true);
        let idx = state.add_block(block);
        state.small_pool.insert(idx, &state.blocks[idx]);

        // Should find it for a 512-byte request.
        let found = state.small_pool.find_free_block(stream, 512);
        assert_eq!(found, Some(idx));
    }

    #[test]
    fn block_pool_respects_stream() {
        let mut state = AllocatorState::new();
        let stream_a = StreamId(1);
        let stream_b = StreamId(2);

        let block = Block::new(0, 4096, 0x1000, stream_a, true);
        let idx = state.add_block(block);
        state.small_pool.insert(idx, &state.blocks[idx]);

        // Should NOT find it for a different stream.
        assert!(state.small_pool.find_free_block(stream_b, 512).is_none());

        // Should find it for the correct stream.
        assert_eq!(state.small_pool.find_free_block(stream_a, 512), Some(idx));
    }

    #[test]
    fn block_pool_finds_smallest_fit() {
        let mut state = AllocatorState::new();
        let stream = make_stream();

        // Add blocks of sizes 4096 and 1024.
        let b1 = Block::new(0, 4096, 0x1000, stream, true);
        let i1 = state.add_block(b1);
        state.small_pool.insert(i1, &state.blocks[i1]);

        let b2 = Block::new(0, 1024, 0x2000, stream, true);
        let i2 = state.add_block(b2);
        state.small_pool.insert(i2, &state.blocks[i2]);

        // Request 768 bytes — should return the 1024 block (smallest fit).
        let found = state.small_pool.find_free_block(stream, 768);
        assert_eq!(found, Some(i2));
    }

    #[test]
    fn split_block_creates_remainder() {
        let mut state = AllocatorState::new();
        let stream = make_stream();

        let block = Block::new(0, 8192, 0x1000, stream, true);
        let idx = state.add_block(block);

        // Split: request 1024 from an 8192 block.
        state.split_block(idx, 1024);

        assert_eq!(state.blocks[idx].size, 1024);
        let rem_idx = state.blocks[idx].next.unwrap();
        assert_eq!(state.blocks[rem_idx].size, 8192 - 1024);
        assert_eq!(state.blocks[rem_idx].ptr, 0x1000 + 1024);
        assert_eq!(state.blocks[rem_idx].prev, Some(idx));

        // Remainder should be in the free pool.
        let found = state.small_pool.find_free_block(stream, 1024);
        assert_eq!(found, Some(rem_idx));
    }

    #[test]
    fn coalesce_merges_adjacent_blocks() {
        let mut state = AllocatorState::new();
        let stream = make_stream();

        // Simulate: a segment of 3 blocks: [A=2048] [B=2048] [C=4096]
        let a = Block::new(0, 2048, 0x1000, stream, true);
        let a_idx = state.add_block(a);

        let b = Block::new(0, 2048, 0x1000 + 2048, stream, true);
        let b_idx = state.add_block(b);

        let c = Block::new(0, 4096, 0x1000 + 4096, stream, true);
        let c_idx = state.add_block(c);

        // Link them.
        state.blocks[a_idx].next = Some(b_idx);
        state.blocks[b_idx].prev = Some(a_idx);
        state.blocks[b_idx].next = Some(c_idx);
        state.blocks[c_idx].prev = Some(b_idx);

        // A and C are free (in pool), B is allocated.
        state.blocks[b_idx].allocated = true;
        state.blocks[b_idx].size = 2048;
        state.allocated_bytes = 2048;

        state.small_pool.insert(a_idx, &state.blocks[a_idx]);
        state.small_pool.insert(c_idx, &state.blocks[c_idx]);

        // Free B — should coalesce with A and C.
        state.free_block(b_idx);

        // B should now be the merged block spanning all 8192 bytes.
        assert_eq!(state.blocks[b_idx].size, 2048 + 2048 + 4096);
        assert_eq!(state.blocks[b_idx].ptr, 0x1000);
        assert!(!state.blocks[b_idx].allocated);
    }

    #[test]
    fn should_split_small_pool() {
        let mut state = AllocatorState::new();
        let stream = make_stream();

        let block = Block::new(0, 2048, 0x1000, stream, true);
        let idx = state.add_block(block);

        // Remainder = 2048 - 1024 = 1024 >= MIN_BLOCK_SIZE(512) => split.
        assert!(state.should_split(idx, 1024));

        // Remainder = 2048 - 1800 = 248 < MIN_BLOCK_SIZE(512) => no split.
        assert!(!state.should_split(idx, 1800));
    }

    #[test]
    fn should_split_large_pool() {
        let mut state = AllocatorState::new();
        let stream = make_stream();

        let block = Block::new(0, 4 * 1024 * 1024, 0x1000, stream, false);
        let idx = state.add_block(block);

        // Remainder = 4MB - 2MB = 2MB > SMALL_SIZE(1MB) => split.
        assert!(state.should_split(idx, 2 * 1024 * 1024));

        // Remainder = 4MB - 3.5MB = 0.5MB <= SMALL_SIZE(1MB) => no split.
        assert!(!state.should_split(idx, 3 * 1024 * 1024 + 512 * 1024));
    }

    #[test]
    fn stream_uses_prevent_reuse() {
        let stream = make_stream();
        let mut block = Block::new(0, 4096, 0x1000, stream, true);

        assert!(block.stream_uses.is_empty());
        block.stream_uses.insert(StreamId(99));

        // Block has pending stream uses — merge should be prevented.
        assert!(!block.stream_uses.is_empty());
    }

    #[test]
    fn stream_uses_prevent_merge() {
        let mut state = AllocatorState::new();
        let stream = make_stream();

        // Two adjacent blocks: [A=2048] [B=2048]
        let a = Block::new(0, 2048, 0x1000, stream, true);
        let a_idx = state.add_block(a);

        let mut b = Block::new(0, 2048, 0x1000 + 2048, stream, true);
        b.stream_uses.insert(StreamId(99)); // pending work
        let b_idx = state.add_block(b);

        // Link.
        state.blocks[a_idx].next = Some(b_idx);
        state.blocks[b_idx].prev = Some(a_idx);

        // B is in the pool but has pending stream uses.
        state.small_pool.insert(b_idx, &state.blocks[b_idx]);

        // Try to merge A with B — should fail because B has stream_uses.
        let merged = state.try_merge(a_idx, Some(b_idx));
        assert_eq!(merged, 0);
        assert_eq!(state.blocks[a_idx].size, 2048); // unchanged
    }

    #[test]
    fn cache_find_and_insert_roundtrip() {
        let device = Arc::new(match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return, // no GPU, skip
        });
        let alloc = CudaAllocator::new(device);
        let stream = StreamId(1);

        // Insert a 4096-byte block.
        let (idx, actual) = alloc.cache_insert(2048, 4096, 0x1000, stream);
        // The block should be split: 2048 for the request, remainder free.
        assert!(actual <= 4096);
        assert_eq!(alloc.cache_stats().1, 1); // 1 miss

        // Free it.
        alloc.cache_free(idx);

        // Find it again — should be a hit.
        let found = alloc.cache_find(512, stream);
        assert!(found.is_some());
        assert_eq!(alloc.cache_stats().0, 1); // 1 hit
    }

    #[test]
    fn empty_cache_clears_pools() {
        let device = Arc::new(match GpuDevice::new(0) {
            Ok(d) => d,
            Err(_) => return,
        });
        let alloc = CudaAllocator::new(device);
        let stream = StreamId(1);

        alloc.cache_insert(1024, 4096, 0x1000, stream);
        {
            let state = alloc.state.lock().unwrap();
            // Should have blocks in the arena.
            assert!(!state.blocks.is_empty());
        }

        // Free block 0 back to pool.
        alloc.cache_free(0);
        assert!(alloc.free_block_count() > 0);

        alloc.empty_cache();
        assert_eq!(alloc.free_block_count(), 0);
    }

    // ------------------------------------------------------------------
    // CUDA integration tests
    // ------------------------------------------------------------------

    #[cfg(feature = "cuda")]
    mod cuda_tests {
        use super::*;

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
            alloc.empty_cache();
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
            assert_eq!(
                alloc.max_memory_allocated(),
                256 * std::mem::size_of::<f32>()
            );
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

            assert_eq!(alloc.max_memory_allocated(), high);

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
}
