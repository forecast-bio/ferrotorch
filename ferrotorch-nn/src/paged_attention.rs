//! PagedAttention — efficient KV cache management for LLM serving.
//!
//! Standard KV caches allocate contiguous memory per sequence. When serving
//! many concurrent sequences of varying length the memory fragments badly:
//! short sequences waste pre-allocated space, and long sequences cannot grow
//! without expensive reallocation.
//!
//! **PagedAttention** (Kwon et al., 2023) solves this by managing the KV cache
//! as fixed-size *pages* drawn from a shared pool. Pages are allocated on
//! demand and freed when a sequence completes, so memory usage scales with
//! the *total* number of tokens across all sequences rather than the *maximum*
//! per sequence.
//!
//! # Components
//!
//! - [`KVPage`] — A single page holding key/value vectors for up to
//!   `page_size` tokens.
//! - [`PagePool`] — A pool of reusable pages with O(1) alloc/free.
//! - [`PagedKVCache`] — A single sequence's KV cache backed by pages from
//!   the pool.
//! - [`PagedAttentionManager`] — Multi-sequence manager that owns the pool
//!   and all active sequences.
//!
//! # Example
//!
//! ```rust,no_run
//! use ferrotorch_nn::paged_attention::PagedAttentionManager;
//!
//! // 64 pages of 256 tokens each, 8 heads, head_dim=64
//! let mut mgr: PagedAttentionManager<f32> =
//!     PagedAttentionManager::new(64, 256, 8, 64);
//!
//! let seq = mgr.add_sequence();
//! // append KV for 10 new tokens
//! let k = vec![0.0f32; 10 * 8 * 64];
//! let v = vec![0.0f32; 10 * 8 * 64];
//! mgr.append_kv(seq, &k, &v).unwrap();
//! let (keys, vals) = mgr.get_kv(seq).unwrap();
//! assert_eq!(keys.shape(), &[10, 8, 64]);
//! ```
//!
//! # Reference
//!
//! Kwon et al., "Efficient Memory Management for Large Language Model Serving
//! with PagedAttention" (SOSP 2023).

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

// ===========================================================================
// KVPage
// ===========================================================================

/// A fixed-size page holding key/value vectors for up to `page_size` tokens.
///
/// Data is stored in row-major order: `[page_size, num_heads, head_dim]`.
/// Only the first `len` token-slots are occupied; the rest are allocated but
/// unused.
#[derive(Debug, Clone)]
pub struct KVPage<T: Float> {
    /// Key data: capacity for `[page_size, num_heads, head_dim]`.
    key: Vec<T>,
    /// Value data: same capacity as `key`.
    value: Vec<T>,
    /// Number of tokens currently stored (0..=page_size).
    len: usize,
    /// Page capacity (tokens per page).
    page_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl<T: Float> KVPage<T> {
    /// Create a new empty page with the given dimensions.
    fn new(page_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let cap = page_size * num_heads * head_dim;
        let zero = <T as num_traits::Zero>::zero();
        Self {
            key: vec![zero; cap],
            value: vec![zero; cap],
            len: 0,
            page_size,
            num_heads,
            head_dim,
        }
    }

    /// Number of elements per token (num_heads * head_dim).
    #[inline]
    fn token_stride(&self) -> usize {
        self.num_heads * self.head_dim
    }

    /// Remaining token-slots available.
    #[inline]
    fn remaining(&self) -> usize {
        self.page_size - self.len
    }

    /// Append key/value data for `count` tokens starting at the given offset
    /// within the provided slices.
    ///
    /// # Panics
    ///
    /// Panics (debug) if `count > self.remaining()`.
    fn append(&mut self, key: &[T], value: &[T], src_offset: usize, count: usize) {
        debug_assert!(count <= self.remaining());
        let stride = self.token_stride();
        let dst_start = self.len * stride;
        let src_start = src_offset * stride;
        let n = count * stride;
        self.key[dst_start..dst_start + n].copy_from_slice(&key[src_start..src_start + n]);
        self.value[dst_start..dst_start + n].copy_from_slice(&value[src_start..src_start + n]);
        self.len += count;
    }

    /// Read key data for all occupied slots as a slice.
    fn key_data(&self) -> &[T] {
        &self.key[..self.len * self.token_stride()]
    }

    /// Read value data for all occupied slots as a slice.
    fn value_data(&self) -> &[T] {
        &self.value[..self.len * self.token_stride()]
    }

    /// Reset this page to empty (does not deallocate).
    fn clear(&mut self) {
        self.len = 0;
    }
}

// ===========================================================================
// PagePool
// ===========================================================================

/// A pool of reusable KV pages.
///
/// Pages are pre-allocated at construction time and recycled via a free-list.
/// Allocation and deallocation are both O(1).
#[derive(Debug)]
pub struct PagePool<T: Float> {
    /// Free pages available for allocation (stored as page IDs).
    free_pages: Vec<usize>,
    /// All pages (indexed by page_id).
    pages: Vec<KVPage<T>>,
    /// Page size (tokens per page).
    page_size: usize,
    num_heads: usize,
    head_dim: usize,
}

impl<T: Float> PagePool<T> {
    /// Create a new pool with `num_pages` pre-allocated pages.
    ///
    /// Each page can hold up to `page_size` tokens, with key/value vectors
    /// of shape `[num_heads, head_dim]` per token.
    ///
    /// # Panics
    ///
    /// Panics if any dimension is zero.
    pub fn new(num_pages: usize, page_size: usize, num_heads: usize, head_dim: usize) -> Self {
        assert!(num_pages > 0, "PagePool: num_pages must be positive");
        assert!(page_size > 0, "PagePool: page_size must be positive");
        assert!(num_heads > 0, "PagePool: num_heads must be positive");
        assert!(head_dim > 0, "PagePool: head_dim must be positive");

        let pages: Vec<KVPage<T>> = (0..num_pages)
            .map(|_| KVPage::new(page_size, num_heads, head_dim))
            .collect();

        // All pages start free; push in reverse so the lowest IDs are allocated first.
        let free_pages: Vec<usize> = (0..num_pages).rev().collect();

        Self {
            free_pages,
            pages,
            page_size,
            num_heads,
            head_dim,
        }
    }

    /// Allocate a page from the pool, returning its page ID.
    ///
    /// Returns `None` if the pool is exhausted.
    pub fn alloc_page(&mut self) -> Option<usize> {
        self.free_pages.pop()
    }

    /// Return a page to the pool.
    ///
    /// The page is cleared (token count reset to 0) before being made
    /// available for reuse.
    ///
    /// # Panics
    ///
    /// Panics if `page_id` is out of range.
    pub fn free_page(&mut self, page_id: usize) {
        assert!(
            page_id < self.pages.len(),
            "PagePool::free_page: page_id {page_id} out of range (pool has {} pages)",
            self.pages.len()
        );
        self.pages[page_id].clear();
        self.free_pages.push(page_id);
    }

    /// Number of pages currently available for allocation.
    #[inline]
    pub fn num_free(&self) -> usize {
        self.free_pages.len()
    }

    /// Number of pages currently in use.
    #[inline]
    pub fn num_used(&self) -> usize {
        self.pages.len() - self.free_pages.len()
    }

    /// Total number of pages in the pool.
    #[inline]
    pub fn num_total(&self) -> usize {
        self.pages.len()
    }

    /// Get a reference to a page by ID.
    #[inline]
    fn page(&self, page_id: usize) -> &KVPage<T> {
        &self.pages[page_id]
    }

    /// Get a mutable reference to a page by ID.
    #[inline]
    fn page_mut(&mut self, page_id: usize) -> &mut KVPage<T> {
        &mut self.pages[page_id]
    }
}

// ===========================================================================
// PagedKVCache
// ===========================================================================

/// A single sequence's KV cache backed by pages from a [`PagePool`].
///
/// Pages are allocated on demand as tokens are appended. The cache does not
/// own the pages — it holds page IDs that index into the shared pool.
#[derive(Debug, Clone)]
pub struct PagedKVCache<T: Float> {
    /// Page IDs in order (each holds up to page_size tokens).
    page_ids: Vec<usize>,
    /// Total tokens stored across all pages.
    total_tokens: usize,
    /// Phantom to carry the type parameter.
    _marker: std::marker::PhantomData<T>,
}

impl<T: Float> PagedKVCache<T> {
    /// Create a new empty paged KV cache.
    pub fn new() -> Self {
        Self {
            page_ids: Vec::new(),
            total_tokens: 0,
            _marker: std::marker::PhantomData,
        }
    }

    /// Append key/value data for new tokens.
    ///
    /// Allocates new pages from the pool as needed. The input slices must
    /// have length `num_new_tokens * num_heads * head_dim`.
    ///
    /// # Errors
    ///
    /// Returns an error if the pool is exhausted before all tokens are stored,
    /// or if the input slice lengths are invalid.
    pub fn append(
        &mut self,
        pool: &mut PagePool<T>,
        key: &[T],
        value: &[T],
    ) -> FerrotorchResult<()> {
        let stride = pool.num_heads * pool.head_dim;
        if key.len() != value.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PagedKVCache::append: key len ({}) != value len ({})",
                    key.len(),
                    value.len()
                ),
            });
        }
        if key.len() % stride != 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "PagedKVCache::append: key len ({}) is not divisible by \
                     num_heads * head_dim ({})",
                    key.len(),
                    stride
                ),
            });
        }

        let num_new_tokens = key.len() / stride;
        let mut tokens_written = 0usize;

        while tokens_written < num_new_tokens {
            // Try to fill the last page first.
            let remaining_in_last = self
                .page_ids
                .last()
                .map(|&pid| pool.page(pid).remaining())
                .unwrap_or(0);

            if remaining_in_last > 0 {
                let pid = *self.page_ids.last().unwrap();
                let to_write = remaining_in_last.min(num_new_tokens - tokens_written);
                pool.page_mut(pid)
                    .append(key, value, tokens_written, to_write);
                tokens_written += to_write;
            } else {
                // Need a fresh page.
                let pid = pool
                    .alloc_page()
                    .ok_or_else(|| FerrotorchError::InvalidArgument {
                        message: format!(
                            "PagedKVCache::append: page pool exhausted \
                         (needed tokens for {} more tokens, pool has 0 free pages)",
                            num_new_tokens - tokens_written
                        ),
                    })?;
                self.page_ids.push(pid);
                let to_write = pool.page_size.min(num_new_tokens - tokens_written);
                pool.page_mut(pid)
                    .append(key, value, tokens_written, to_write);
                tokens_written += to_write;
            }
        }

        self.total_tokens += num_new_tokens;
        Ok(())
    }

    /// Read all cached keys as a contiguous tensor of shape
    /// `[total_tokens, num_heads, head_dim]`.
    pub fn get_keys(&self, pool: &PagePool<T>) -> Tensor<T> {
        self.gather_data(pool, |page| page.key_data())
    }

    /// Read all cached values as a contiguous tensor of shape
    /// `[total_tokens, num_heads, head_dim]`.
    pub fn get_values(&self, pool: &PagePool<T>) -> Tensor<T> {
        self.gather_data(pool, |page| page.value_data())
    }

    /// Total tokens currently cached.
    #[inline]
    pub fn len(&self) -> usize {
        self.total_tokens
    }

    /// Whether the cache is empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.total_tokens == 0
    }

    /// Release all pages back to the pool.
    pub fn free_all(&mut self, pool: &mut PagePool<T>) {
        for &pid in &self.page_ids {
            pool.free_page(pid);
        }
        self.page_ids.clear();
        self.total_tokens = 0;
    }

    /// Number of pages currently held by this cache.
    #[inline]
    pub fn num_pages(&self) -> usize {
        self.page_ids.len()
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Gather key or value data from all pages into a contiguous tensor.
    fn gather_data<F>(&self, pool: &PagePool<T>, extractor: F) -> Tensor<T>
    where
        F: Fn(&KVPage<T>) -> &[T],
    {
        if self.total_tokens == 0 {
            // Return a zero-element tensor with the correct shape.
            return Tensor::from_storage(
                TensorStorage::cpu(Vec::new()),
                vec![0, pool.num_heads, pool.head_dim],
                false,
            )
            .expect("creating zero-length tensor should not fail");
        }

        let stride = pool.num_heads * pool.head_dim;
        let mut data = Vec::with_capacity(self.total_tokens * stride);

        for &pid in &self.page_ids {
            let page = pool.page(pid);
            data.extend_from_slice(extractor(page));
        }

        Tensor::from_storage(
            TensorStorage::cpu(data),
            vec![self.total_tokens, pool.num_heads, pool.head_dim],
            false,
        )
        .expect("tensor shape matches gathered data")
    }
}

impl<T: Float> Default for PagedKVCache<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ===========================================================================
// PagedAttentionManager
// ===========================================================================

/// Multi-sequence manager for concurrent LLM serving.
///
/// Owns the [`PagePool`] and all active [`PagedKVCache`] instances. Provides
/// a simple ID-based API for adding, querying, and removing sequences.
#[derive(Debug)]
pub struct PagedAttentionManager<T: Float> {
    pool: PagePool<T>,
    sequences: Vec<Option<PagedKVCache<T>>>,
}

impl<T: Float> PagedAttentionManager<T> {
    /// Create a new manager with a fresh page pool.
    ///
    /// # Arguments
    ///
    /// - `num_pages`  — Total pages in the pool.
    /// - `page_size`  — Tokens per page.
    /// - `num_heads`  — Number of attention heads.
    /// - `head_dim`   — Dimension per head.
    pub fn new(num_pages: usize, page_size: usize, num_heads: usize, head_dim: usize) -> Self {
        Self {
            pool: PagePool::new(num_pages, page_size, num_heads, head_dim),
            sequences: Vec::new(),
        }
    }

    /// Add a new empty sequence and return its ID.
    pub fn add_sequence(&mut self) -> usize {
        // Reuse a previously-removed slot if one exists.
        for (id, slot) in self.sequences.iter_mut().enumerate() {
            if slot.is_none() {
                *slot = Some(PagedKVCache::new());
                return id;
            }
        }
        let id = self.sequences.len();
        self.sequences.push(Some(PagedKVCache::new()));
        id
    }

    /// Append key/value data for new tokens to a sequence.
    ///
    /// `key` and `value` must have length `num_new_tokens * num_heads * head_dim`.
    pub fn append_kv(&mut self, seq_id: usize, key: &[T], value: &[T]) -> FerrotorchResult<()> {
        let cache = self
            .sequences
            .get_mut(seq_id)
            .and_then(|s| s.as_mut())
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("PagedAttentionManager: invalid sequence id {seq_id}"),
            })?;
        cache.append(&mut self.pool, key, value)
    }

    /// Get the cached keys and values for a sequence.
    ///
    /// Returns tensors of shape `[total_tokens, num_heads, head_dim]`.
    pub fn get_kv(&self, seq_id: usize) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        let cache = self
            .sequences
            .get(seq_id)
            .and_then(|s| s.as_ref())
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("PagedAttentionManager: invalid sequence id {seq_id}"),
            })?;
        Ok((cache.get_keys(&self.pool), cache.get_values(&self.pool)))
    }

    /// Remove a sequence, freeing all of its pages back to the pool.
    pub fn remove_sequence(&mut self, seq_id: usize) {
        if let Some(slot) = self.sequences.get_mut(seq_id) {
            if let Some(mut cache) = slot.take() {
                cache.free_all(&mut self.pool);
            }
        }
    }

    /// Number of active sequences.
    pub fn num_sequences(&self) -> usize {
        self.sequences.iter().filter(|s| s.is_some()).count()
    }

    /// Pool utilization ratio (used pages / total pages).
    pub fn pool_utilization(&self) -> f64 {
        let total = self.pool.num_total() as f64;
        let used = self.pool.num_used() as f64;
        used / total
    }

    /// Read-only access to the underlying pool.
    pub fn pool(&self) -> &PagePool<T> {
        &self.pool
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // PagePool
    // -----------------------------------------------------------------------

    #[test]
    fn pool_alloc_free_cycle() {
        let mut pool: PagePool<f32> = PagePool::new(4, 16, 2, 8);
        assert_eq!(pool.num_free(), 4);
        assert_eq!(pool.num_used(), 0);

        let p0 = pool.alloc_page().unwrap();
        let p1 = pool.alloc_page().unwrap();
        assert_eq!(pool.num_free(), 2);
        assert_eq!(pool.num_used(), 2);

        pool.free_page(p0);
        assert_eq!(pool.num_free(), 3);
        assert_eq!(pool.num_used(), 1);

        pool.free_page(p1);
        assert_eq!(pool.num_free(), 4);
        assert_eq!(pool.num_used(), 0);
    }

    #[test]
    fn pool_exhaustion_returns_none() {
        let mut pool: PagePool<f32> = PagePool::new(2, 4, 1, 1);
        assert!(pool.alloc_page().is_some());
        assert!(pool.alloc_page().is_some());
        assert!(pool.alloc_page().is_none());
    }

    // -----------------------------------------------------------------------
    // PagedKVCache — append & read
    // -----------------------------------------------------------------------

    #[test]
    fn cache_append_grows_pages() {
        let mut pool: PagePool<f32> = PagePool::new(8, 4, 2, 3);
        let mut cache = PagedKVCache::<f32>::new();

        // 4 tokens per page, stride = 2*3 = 6
        // Append 3 tokens => fits in 1 page
        let key: Vec<f32> = (0..18).map(|i| i as f32).collect(); // 3 * 6
        let val: Vec<f32> = (100..118).map(|i| i as f32).collect();
        cache.append(&mut pool, &key, &val).unwrap();
        assert_eq!(cache.len(), 3);
        assert_eq!(cache.num_pages(), 1);

        // Append 3 more tokens => spills into 2nd page (1 remaining + 2 new)
        let key2: Vec<f32> = (18..36).map(|i| i as f32).collect();
        let val2: Vec<f32> = (118..136).map(|i| i as f32).collect();
        cache.append(&mut pool, &key2, &val2).unwrap();
        assert_eq!(cache.len(), 6);
        assert_eq!(cache.num_pages(), 2);
    }

    #[test]
    fn cache_get_keys_values_correct_data() {
        let mut pool: PagePool<f32> = PagePool::new(4, 2, 1, 2);
        let mut cache = PagedKVCache::<f32>::new();

        // stride = 1*2 = 2, page_size = 2 tokens
        // Append 3 tokens => 2 pages
        let key = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 tokens * 2
        let val = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        cache.append(&mut pool, &key, &val).unwrap();

        let keys_tensor = cache.get_keys(&pool);
        let vals_tensor = cache.get_values(&pool);

        assert_eq!(keys_tensor.shape(), &[3, 1, 2]);
        assert_eq!(vals_tensor.shape(), &[3, 1, 2]);

        let k_data = keys_tensor.data().unwrap();
        assert_eq!(k_data, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);

        let v_data = vals_tensor.data().unwrap();
        assert_eq!(v_data, &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn cache_empty_returns_zero_tensor() {
        let pool: PagePool<f32> = PagePool::new(2, 4, 2, 3);
        let cache = PagedKVCache::<f32>::new();

        let keys = cache.get_keys(&pool);
        assert_eq!(keys.shape(), &[0, 2, 3]);
        assert!(cache.is_empty());
    }

    #[test]
    fn cache_free_all_returns_pages() {
        let mut pool: PagePool<f32> = PagePool::new(4, 2, 1, 1);
        let mut cache = PagedKVCache::<f32>::new();

        let key = vec![1.0; 5]; // 5 tokens, 3 pages needed
        let val = vec![2.0; 5];
        cache.append(&mut pool, &key, &val).unwrap();
        assert_eq!(pool.num_used(), 3);

        cache.free_all(&mut pool);
        assert_eq!(pool.num_used(), 0);
        assert_eq!(pool.num_free(), 4);
        assert_eq!(cache.len(), 0);
    }

    // -----------------------------------------------------------------------
    // Pool exhaustion error
    // -----------------------------------------------------------------------

    #[test]
    fn cache_append_fails_on_pool_exhaustion() {
        let mut pool: PagePool<f32> = PagePool::new(1, 2, 1, 1);
        let mut cache = PagedKVCache::<f32>::new();

        // Fill the only page (2 tokens).
        cache.append(&mut pool, &[1.0, 2.0], &[3.0, 4.0]).unwrap();

        // Try to append more — pool is exhausted.
        let result = cache.append(&mut pool, &[5.0], &[6.0]);
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("exhausted"),
            "error should mention exhaustion, got: {err_msg}"
        );
    }

    // -----------------------------------------------------------------------
    // Multiple sequences sharing the pool
    // -----------------------------------------------------------------------

    #[test]
    fn multiple_sequences_share_pool() {
        let mut pool: PagePool<f32> = PagePool::new(8, 4, 1, 2);
        let mut cache_a = PagedKVCache::<f32>::new();
        let mut cache_b = PagedKVCache::<f32>::new();

        // Sequence A: 3 tokens (1 page)
        let ka = vec![1.0; 6];
        let va = vec![2.0; 6];
        cache_a.append(&mut pool, &ka, &va).unwrap();

        // Sequence B: 5 tokens (2 pages)
        let kb = vec![3.0; 10];
        let vb = vec![4.0; 10];
        cache_b.append(&mut pool, &kb, &vb).unwrap();

        assert_eq!(pool.num_used(), 3); // 1 + 2
        assert_eq!(cache_a.len(), 3);
        assert_eq!(cache_b.len(), 5);

        // Free A, B still works.
        cache_a.free_all(&mut pool);
        assert_eq!(pool.num_used(), 2);

        let bkeys = cache_b.get_keys(&pool);
        assert_eq!(bkeys.shape(), &[5, 1, 2]);
    }

    // -----------------------------------------------------------------------
    // Memory grows linearly with total tokens
    // -----------------------------------------------------------------------

    #[test]
    fn memory_grows_linearly_with_total_tokens() {
        let page_size = 8;
        let num_heads = 2;
        let head_dim = 4;
        let stride = num_heads * head_dim;

        let mut pool: PagePool<f32> = PagePool::new(100, page_size, num_heads, head_dim);
        let mut caches: Vec<PagedKVCache<f32>> = Vec::new();

        // Create 10 sequences of varying length.
        let mut total_tokens = 0usize;
        for i in 1..=10 {
            let n = i * 3; // 3, 6, 9, ..., 30 tokens
            let key = vec![1.0f32; n * stride];
            let val = vec![2.0f32; n * stride];
            let mut c = PagedKVCache::<f32>::new();
            c.append(&mut pool, &key, &val).unwrap();
            caches.push(c);
            total_tokens += n;
        }

        // Pages used should be ceil(tokens / page_size) per sequence, which
        // in total equals ceil(total_tokens_per_seq / page_size) summed.
        // The key property: used pages scale with total tokens, not with
        // max-per-sequence * num_sequences.
        let expected_pages: usize = (1..=10)
            .map(|i| {
                let n = i * 3;
                (n + page_size - 1) / page_size
            })
            .sum();
        assert_eq!(pool.num_used(), expected_pages);

        // Total tokens across all caches.
        let cached_total: usize = caches.iter().map(|c| c.len()).sum();
        assert_eq!(cached_total, total_tokens);
    }

    // -----------------------------------------------------------------------
    // PagedAttentionManager
    // -----------------------------------------------------------------------

    #[test]
    fn manager_add_and_remove_sequences() {
        let mut mgr: PagedAttentionManager<f32> = PagedAttentionManager::new(16, 4, 2, 3);

        let s0 = mgr.add_sequence();
        let s1 = mgr.add_sequence();
        assert_eq!(mgr.num_sequences(), 2);

        mgr.remove_sequence(s0);
        assert_eq!(mgr.num_sequences(), 1);

        // Slot is reused.
        let s2 = mgr.add_sequence();
        assert_eq!(s2, s0);
        assert_eq!(mgr.num_sequences(), 2);

        mgr.remove_sequence(s1);
        mgr.remove_sequence(s2);
        assert_eq!(mgr.num_sequences(), 0);
    }

    #[test]
    fn manager_append_and_get_kv() {
        let mut mgr: PagedAttentionManager<f32> = PagedAttentionManager::new(8, 4, 1, 2);

        let seq = mgr.add_sequence();
        let key = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 3 tokens
        let val = vec![10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        mgr.append_kv(seq, &key, &val).unwrap();

        let (k, v) = mgr.get_kv(seq).unwrap();
        assert_eq!(k.shape(), &[3, 1, 2]);
        assert_eq!(v.shape(), &[3, 1, 2]);
        assert_eq!(k.data().unwrap(), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(v.data().unwrap(), &[10.0, 20.0, 30.0, 40.0, 50.0, 60.0]);
    }

    #[test]
    fn manager_invalid_seq_id_returns_error() {
        let mut mgr: PagedAttentionManager<f32> = PagedAttentionManager::new(4, 4, 1, 1);

        let result = mgr.append_kv(999, &[1.0], &[2.0]);
        assert!(result.is_err());

        let result = mgr.get_kv(999);
        assert!(result.is_err());
    }

    #[test]
    fn manager_pool_utilization() {
        let mut mgr: PagedAttentionManager<f32> = PagedAttentionManager::new(10, 4, 1, 1);

        assert_eq!(mgr.pool_utilization(), 0.0);

        let seq = mgr.add_sequence();
        // 4 tokens fills exactly 1 page
        mgr.append_kv(seq, &[1.0; 4], &[2.0; 4]).unwrap();
        assert!((mgr.pool_utilization() - 0.1).abs() < 1e-10);

        mgr.remove_sequence(seq);
        assert_eq!(mgr.pool_utilization(), 0.0);
    }

    #[test]
    fn manager_free_returns_pages() {
        let mut mgr: PagedAttentionManager<f32> = PagedAttentionManager::new(4, 2, 1, 1);

        let s0 = mgr.add_sequence();
        let s1 = mgr.add_sequence();

        // s0: 3 tokens (2 pages), s1: 2 tokens (1 page)
        mgr.append_kv(s0, &[1.0; 3], &[2.0; 3]).unwrap();
        mgr.append_kv(s1, &[3.0; 2], &[4.0; 2]).unwrap();
        assert_eq!(mgr.pool().num_used(), 3);
        assert_eq!(mgr.pool().num_free(), 1);

        mgr.remove_sequence(s0);
        assert_eq!(mgr.pool().num_used(), 1);
        assert_eq!(mgr.pool().num_free(), 3);

        mgr.remove_sequence(s1);
        assert_eq!(mgr.pool().num_used(), 0);
        assert_eq!(mgr.pool().num_free(), 4);
    }
}
