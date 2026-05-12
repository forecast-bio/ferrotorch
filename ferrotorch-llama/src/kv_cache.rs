//! Per-layer key/value cache for incremental Llama decoding (#1129).
//!
//! A `LlamaKvCache` stores the rotated `K` and unrotated `V` tensors at
//! every decoder layer, indexed by their absolute position in the
//! sequence. With a cache in hand, callers can ask
//! [`crate::model::LlamaForCausalLM::forward_one_with_cache`] to feed a
//! single new token: the attention block only matmuls Q against the
//! cached K/V slabs and appends the new K/V rows in place, so the
//! per-step cost drops from `O(seq²·d)` (full-prefix forward) to
//! `O(seq·d)`.
//!
//! # Why per-layer
//!
//! Each decoder layer has its own K/V tensors (different projections,
//! different head counts after GQA broadcast). The cache mirrors the
//! structure of [`crate::model::LlamaModel::layers`] one-to-one.
//!
//! # Shape contract
//!
//! For every layer the cache stores two tensors:
//!
//! - `k`: `[num_kv_heads, seq_len, head_dim]` — post-RoPE, pre-GQA-broadcast.
//! - `v`: `[num_kv_heads, seq_len, head_dim]` — value rows.
//!
//! `seq_len` is the cache's `len()` and grows by one per
//! [`LlamaKvCache::extend`] call. Cloning a cache is `O(num_layers ·
//! seq_len · num_kv_heads · head_dim)` because the underlying tensor
//! storage is copied — this is intentional so that beam-search clones
//! one parent cache into N child slots without aliasing surprises.
//!
//! [`crate::model::LlamaForCausalLM::forward_one_with_cache`]: crate::model::LlamaForCausalLM::forward_one_with_cache

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

/// One layer's worth of cached K and V.
#[derive(Debug, Clone)]
pub struct LayerKvCache<T: Float> {
    /// Post-RoPE keys, shape `[num_kv_heads, seq_len, head_dim]`.
    pub k: Tensor<T>,
    /// Raw values, shape `[num_kv_heads, seq_len, head_dim]`.
    pub v: Tensor<T>,
}

impl<T: Float> LayerKvCache<T> {
    /// Current sequence length stored in this layer's cache.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] if the underlying
    /// tensors do not have the expected 3-D `[num_kv_heads, seq_len,
    /// head_dim]` shape.
    pub fn seq_len(&self) -> FerrotorchResult<usize> {
        let shape = self.k.shape();
        if shape.len() != 3 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LayerKvCache: expected 3-D K shape [Hkv, S, d], got {shape:?}"
                ),
            });
        }
        Ok(shape[1])
    }

    /// Append one position's K and V rows to this layer's cache.
    ///
    /// Both `new_k` and `new_v` must have shape
    /// `[num_kv_heads, 1, head_dim]` matching the cache. The returned
    /// `LayerKvCache` has `seq_len` increased by exactly one.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] when the cached and
    /// new tensors disagree on `num_kv_heads` or `head_dim`, or when
    /// `new_k` / `new_v` are not 3-D with a length-1 sequence axis.
    pub fn append(&self, new_k: &Tensor<T>, new_v: &Tensor<T>) -> FerrotorchResult<Self> {
        let ks = self.k.shape();
        let vs = self.v.shape();
        let nks = new_k.shape();
        let nvs = new_v.shape();
        if ks.len() != 3 || vs.len() != 3 || nks.len() != 3 || nvs.len() != 3 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LayerKvCache::append: all of cache.k {ks:?}, cache.v {vs:?}, \
                     new_k {nks:?}, new_v {nvs:?} must be 3-D [Hkv, S, d]"
                ),
            });
        }
        if nks[1] != 1 || nvs[1] != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LayerKvCache::append: new K/V must have seq_len=1, got K={nks:?}, V={nvs:?}"
                ),
            });
        }
        if ks[0] != nks[0] || ks[2] != nks[2] || vs[0] != nvs[0] || vs[2] != nvs[2] {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LayerKvCache::append: head/dim mismatch — cache K {ks:?} V {vs:?} \
                     vs new K {nks:?} V {nvs:?}"
                ),
            });
        }
        let h_kv = ks[0];
        let d = ks[2];
        let old_seq = ks[1];
        let new_seq = old_seq + 1;
        let k_old = self.k.data_vec()?;
        let v_old = self.v.data_vec()?;
        let k_new = new_k.data_vec()?;
        let v_new = new_v.data_vec()?;
        let mut k_buf = Vec::with_capacity(h_kv * new_seq * d);
        let mut v_buf = Vec::with_capacity(h_kv * new_seq * d);
        for h in 0..h_kv {
            // existing rows for this head: old_seq * d
            let old_start_k = h * old_seq * d;
            let old_start_v = h * old_seq * d;
            k_buf.extend_from_slice(&k_old[old_start_k..old_start_k + old_seq * d]);
            v_buf.extend_from_slice(&v_old[old_start_v..old_start_v + old_seq * d]);
            // new row for this head: d entries at offset h * d in new_k / new_v
            let new_start = h * d;
            k_buf.extend_from_slice(&k_new[new_start..new_start + d]);
            v_buf.extend_from_slice(&v_new[new_start..new_start + d]);
        }
        Ok(Self {
            k: Tensor::from_storage(TensorStorage::cpu(k_buf), vec![h_kv, new_seq, d], false)?,
            v: Tensor::from_storage(TensorStorage::cpu(v_buf), vec![h_kv, new_seq, d], false)?,
        })
    }

    /// Construct a fresh `LayerKvCache` directly from new K and V tensors
    /// (no prior cache to extend). Both tensors must already be 3-D
    /// `[num_kv_heads, 1, head_dim]`.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] when `new_k` / `new_v`
    /// are not 3-D with a length-1 sequence axis.
    pub fn from_single_token(new_k: Tensor<T>, new_v: Tensor<T>) -> FerrotorchResult<Self> {
        let ks = new_k.shape();
        let vs = new_v.shape();
        if ks.len() != 3 || vs.len() != 3 || ks[1] != 1 || vs[1] != 1 {
            return Err(FerrotorchError::ShapeMismatch {
                message: format!(
                    "LayerKvCache::from_single_token: K {ks:?} and V {vs:?} must be \
                     3-D [Hkv, 1, d]"
                ),
            });
        }
        Ok(Self { k: new_k, v: new_v })
    }
}

/// Full per-layer K/V cache for one Llama call site.
///
/// Construct one with [`LlamaKvCache::empty`] and grow it through
/// [`crate::model::LlamaForCausalLM::forward_one_with_cache`]. Clone
/// produces a deep-copy suitable for branching beam-search states.
#[derive(Debug, Clone, Default)]
pub struct LlamaKvCache<T: Float> {
    /// One entry per decoder layer, in `model.layers` order.
    pub layers: Vec<LayerKvCache<T>>,
    /// Number of positions accumulated; equals `layers[0].seq_len()`
    /// when the cache is non-empty.
    pub seq_len: usize,
}

impl<T: Float> LlamaKvCache<T> {
    /// Build an empty cache for a model with `num_layers` decoder
    /// layers. No tensor storage is allocated yet — the first
    /// `forward_one_with_cache` call seeds each layer with its initial
    /// K/V slab.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            layers: Vec::new(),
            seq_len: 0,
        }
    }

    /// Current sequence length (`0` for a fresh cache).
    #[must_use]
    pub fn len(&self) -> usize {
        self.seq_len
    }

    /// `true` when no tokens have been pushed yet.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.seq_len == 0
    }

    /// Append a single token's per-layer K/V slabs to the cache.
    ///
    /// `new_layer_kv` must have exactly one entry per existing decoder
    /// layer (or be the first call, in which case the cache is seeded
    /// and `layers` is grown to `new_layer_kv.len()`). Each entry is a
    /// `(K, V)` pair shaped `[num_kv_heads, 1, head_dim]`.
    ///
    /// Returns a new `LlamaKvCache` with `seq_len + 1` length.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::ShapeMismatch`] when the per-layer
    /// shapes don't match the existing cache, or
    /// [`FerrotorchError::InvalidArgument`] when the layer count
    /// changes between calls.
    pub fn extend(&self, new_layer_kv: &[(Tensor<T>, Tensor<T>)]) -> FerrotorchResult<Self> {
        if self.layers.is_empty() {
            // Seed phase.
            let layers: Vec<LayerKvCache<T>> = new_layer_kv
                .iter()
                .map(|(k, v)| LayerKvCache::from_single_token(k.clone(), v.clone()))
                .collect::<FerrotorchResult<Vec<_>>>()?;
            return Ok(Self {
                layers,
                seq_len: 1,
            });
        }
        if new_layer_kv.len() != self.layers.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "LlamaKvCache::extend: layer-count mismatch — cache has {} layers but \
                     {} were supplied",
                    self.layers.len(),
                    new_layer_kv.len()
                ),
            });
        }
        let layers: Vec<LayerKvCache<T>> = self
            .layers
            .iter()
            .zip(new_layer_kv.iter())
            .map(|(lc, (k, v))| lc.append(k, v))
            .collect::<FerrotorchResult<Vec<_>>>()?;
        Ok(Self {
            layers,
            seq_len: self.seq_len + 1,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::Tensor;

    fn make_kv(h_kv: usize, s: usize, d: usize, fill: f32) -> Tensor<f32> {
        let data = vec![fill; h_kv * s * d];
        Tensor::from_storage(TensorStorage::cpu(data), vec![h_kv, s, d], false).unwrap()
    }

    #[test]
    fn empty_cache_has_zero_len() {
        let c = LlamaKvCache::<f32>::empty();
        assert!(c.is_empty());
        assert_eq!(c.len(), 0);
    }

    #[test]
    fn extend_seeds_then_appends() {
        let cache = LlamaKvCache::<f32>::empty();
        // Seed with 2 layers, 1 head, 4-dim head.
        let k1 = make_kv(1, 1, 4, 1.0);
        let v1 = make_kv(1, 1, 4, 2.0);
        let k2 = make_kv(1, 1, 4, 3.0);
        let v2 = make_kv(1, 1, 4, 4.0);
        let cache = cache
            .extend(&[(k1, v1), (k2, v2)])
            .expect("seed extend succeeds");
        assert_eq!(cache.layers.len(), 2);
        assert_eq!(cache.len(), 1);
        // Append a second token: layers grow to seq_len=2.
        let k1b = make_kv(1, 1, 4, 5.0);
        let v1b = make_kv(1, 1, 4, 6.0);
        let k2b = make_kv(1, 1, 4, 7.0);
        let v2b = make_kv(1, 1, 4, 8.0);
        let cache2 = cache
            .extend(&[(k1b, v1b), (k2b, v2b)])
            .expect("append extend succeeds");
        assert_eq!(cache2.len(), 2);
        // Check layer 0 K has 1 head, 2 positions, 4 dim, with first
        // row = 1.0 and second row = 5.0.
        let l0k = cache2.layers[0].k.data_vec().unwrap();
        // h_kv * seq_len * head_dim = 1 * 2 * 4.
        assert_eq!(l0k.len(), 8);
        // First position rows.
        for v in l0k.iter().take(4) {
            assert_eq!(*v, 1.0);
        }
        // Second position rows.
        for v in l0k.iter().skip(4).take(4) {
            assert_eq!(*v, 5.0);
        }
    }

    #[test]
    fn extend_rejects_layer_count_change() {
        let cache = LlamaKvCache::<f32>::empty();
        let k = make_kv(1, 1, 4, 1.0);
        let v = make_kv(1, 1, 4, 1.0);
        let cache = cache.extend(&[(k.clone(), v.clone())]).unwrap();
        // Second call with two layers — error.
        let err = cache.extend(&[(k.clone(), v.clone()), (k, v)]).unwrap_err();
        assert!(
            matches!(err, FerrotorchError::InvalidArgument { .. }),
            "expected InvalidArgument, got {err:?}"
        );
    }

    #[test]
    fn append_rejects_shape_mismatch() {
        let k = make_kv(2, 1, 4, 0.5);
        let v = make_kv(2, 1, 4, 0.5);
        let lc = LayerKvCache::from_single_token(k, v).unwrap();
        let bad_k = make_kv(3, 1, 4, 0.0);
        let bad_v = make_kv(3, 1, 4, 0.0);
        let err = lc.append(&bad_k, &bad_v).unwrap_err();
        assert!(
            matches!(err, FerrotorchError::ShapeMismatch { .. }),
            "expected ShapeMismatch, got {err:?}"
        );
    }
}
