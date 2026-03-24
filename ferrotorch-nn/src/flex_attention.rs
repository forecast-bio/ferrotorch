//! Flexible attention with composable score modification.
//!
//! Implements `flex_attention` — a generalized attention primitive that supports
//! arbitrary score modifications (causal masks, ALiBi, relative position biases)
//! and sparse block masks. This follows the design of PyTorch's
//! `torch.nn.attention.flex_attention`.
//!
//! The base computation is standard scaled dot-product attention:
//!
//! ```text
//! attn_weights = softmax(score_mod(Q @ K^T / sqrt(d_k)))
//! output = attn_weights @ V
//! ```
//!
//! Where `score_mod` is an optional user-provided function that transforms
//! the raw attention scores before softmax.

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::tensor::GradFn;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

/// Score modification function signature: `(score, b, h, q_idx, kv_idx) -> modified_score`.
type ScoreModFn<T> =
    dyn Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Tensor<T>;

// ===========================================================================
// BlockMask — sparse attention patterns
// ===========================================================================

/// A block-level mask for sparse attention patterns.
///
/// Defines which (query_block, key_block) pairs should participate in
/// attention. Blocks outside the mask are skipped entirely, saving
/// computation for structured sparsity patterns like causal, sliding
/// window, or block-sparse attention.
///
/// The mask is stored as a 2D boolean array indexed by
/// `[query_block_idx][key_block_idx]`.
#[derive(Debug, Clone)]
pub struct BlockMask {
    /// `mask[q_block][k_block]` is `true` if the block pair participates.
    mask: Vec<Vec<bool>>,
    /// Block size for queries (number of query positions per block).
    q_block_size: usize,
    /// Block size for keys (number of key positions per block).
    k_block_size: usize,
    /// Total number of query positions.
    n_q: usize,
    /// Total number of key positions.
    n_k: usize,
}

impl BlockMask {
    /// Create a new `BlockMask` from an explicit mask grid.
    ///
    /// # Arguments
    ///
    /// - `mask` - 2D boolean array `[num_q_blocks][num_k_blocks]`.
    /// - `q_block_size` - Positions per query block.
    /// - `k_block_size` - Positions per key block.
    /// - `n_q` - Total query positions.
    /// - `n_k` - Total key positions.
    pub fn new(
        mask: Vec<Vec<bool>>,
        q_block_size: usize,
        k_block_size: usize,
        n_q: usize,
        n_k: usize,
    ) -> FerrotorchResult<Self> {
        if q_block_size == 0 || k_block_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "BlockMask: block sizes must be positive".into(),
            });
        }

        let num_q_blocks = n_q.div_ceil(q_block_size);
        let num_k_blocks = n_k.div_ceil(k_block_size);

        if mask.len() != num_q_blocks {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "BlockMask: expected {} q_blocks but mask has {} rows",
                    num_q_blocks,
                    mask.len()
                ),
            });
        }

        for (i, row) in mask.iter().enumerate() {
            if row.len() != num_k_blocks {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "BlockMask: row {i} has {} entries, expected {num_k_blocks}",
                        row.len()
                    ),
                });
            }
        }

        Ok(Self {
            mask,
            q_block_size,
            k_block_size,
            n_q,
            n_k,
        })
    }

    /// Create a full (dense) attention mask — all blocks participate.
    ///
    /// Equivalent to no masking at all.
    pub fn full_mask(n_q: usize, n_k: usize, block_size: usize) -> FerrotorchResult<Self> {
        if block_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "BlockMask: block_size must be positive".into(),
            });
        }
        let num_q_blocks = n_q.div_ceil(block_size);
        let num_k_blocks = n_k.div_ceil(block_size);
        let mask = vec![vec![true; num_k_blocks]; num_q_blocks];
        Ok(Self {
            mask,
            q_block_size: block_size,
            k_block_size: block_size,
            n_q,
            n_k,
        })
    }

    /// Create a causal (lower-triangular) mask.
    ///
    /// Block `(q_block, k_block)` is active if any position in the query block
    /// can attend to any position in the key block under causal constraints
    /// (i.e., `q_pos >= k_pos`). This means a block is active if:
    /// `q_block_end - 1 >= k_block_start`, i.e., `q_block * bs + bs - 1 >= k_block * bs`.
    ///
    /// Requires `n_q == n_k`.
    pub fn causal_mask(n: usize, block_size: usize) -> FerrotorchResult<Self> {
        if block_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "BlockMask: block_size must be positive".into(),
            });
        }
        let num_blocks = n.div_ceil(block_size);
        let mut mask = Vec::with_capacity(num_blocks);

        for q_blk in 0..num_blocks {
            let mut row = Vec::with_capacity(num_blocks);
            let q_end = (q_blk + 1) * block_size; // exclusive, but we want last valid pos

            for k_blk in 0..num_blocks {
                let k_start = k_blk * block_size;
                // Block is active if there exists any valid (q, k) pair
                // where q >= k. That means the last q position >= first k position.
                row.push(q_end > k_start);
            }
            mask.push(row);
        }

        Ok(Self {
            mask,
            q_block_size: block_size,
            k_block_size: block_size,
            n_q: n,
            n_k: n,
        })
    }

    /// Create a sliding window mask.
    ///
    /// Each query position can only attend to key positions within
    /// `window_size` positions (inclusive on both sides). A block is active if
    /// any position in the query block is within `window_size` of any position
    /// in the key block.
    ///
    /// Requires `n_q == n_k`.
    pub fn sliding_window_mask(
        n: usize,
        window_size: usize,
        block_size: usize,
    ) -> FerrotorchResult<Self> {
        if block_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "BlockMask: block_size must be positive".into(),
            });
        }
        if window_size == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "BlockMask: window_size must be positive".into(),
            });
        }

        let num_blocks = n.div_ceil(block_size);
        let mut mask = Vec::with_capacity(num_blocks);

        for q_blk in 0..num_blocks {
            let mut row = Vec::with_capacity(num_blocks);
            let q_start = q_blk * block_size;
            let q_end = ((q_blk + 1) * block_size).min(n);

            for k_blk in 0..num_blocks {
                let k_start = k_blk * block_size;
                let k_end = ((k_blk + 1) * block_size).min(n);

                // Block is active if any q in [q_start, q_end) is within
                // window_size of any k in [k_start, k_end).
                // Closest pair: min distance is max(0, max(q_start, k_start) - min(q_end-1, k_end-1))
                // Simpler: blocks overlap in window if the closest positions are <= window_size apart.
                let closest = if q_end <= k_start {
                    // Q block entirely before K block.
                    k_start - (q_end - 1)
                } else if k_end <= q_start {
                    // K block entirely before Q block.
                    q_start - (k_end - 1)
                } else {
                    // Blocks overlap.
                    0
                };

                row.push(closest <= window_size);
            }
            mask.push(row);
        }

        Ok(Self {
            mask,
            q_block_size: block_size,
            k_block_size: block_size,
            n_q: n,
            n_k: n,
        })
    }

    /// Check if block `(q_block, k_block)` is active.
    #[inline]
    pub fn is_active(&self, q_block: usize, k_block: usize) -> bool {
        self.mask
            .get(q_block)
            .and_then(|row| row.get(k_block))
            .copied()
            .unwrap_or(false)
    }

    /// Number of query blocks.
    #[inline]
    pub fn num_q_blocks(&self) -> usize {
        self.mask.len()
    }

    /// Number of key blocks.
    #[inline]
    pub fn num_k_blocks(&self) -> usize {
        self.mask.first().map_or(0, |row| row.len())
    }

    /// Check if a specific (query_pos, key_pos) pair falls in an active block.
    #[inline]
    pub fn allows_position(&self, q_pos: usize, k_pos: usize) -> bool {
        let q_block = q_pos / self.q_block_size;
        let k_block = k_pos / self.k_block_size;
        self.is_active(q_block, k_block)
    }
}

// ===========================================================================
// FlexAttentionBackward — autograd support
// ===========================================================================

/// Backward node for `flex_attention`.
///
/// Recomputes the standard attention forward (Q @ K^T, softmax, @ V) during
/// backward to obtain gradients for Q, K, V. Score modifications are
/// currently treated as detached during backward (no gradient through
/// score_mod parameters), which matches PyTorch's default behavior for
/// non-materialized score_mod.
#[derive(Debug)]
struct FlexAttentionBackward<T: Float> {
    query: Tensor<T>,
    key: Tensor<T>,
    value: Tensor<T>,
    /// The softmax-normalized attention weights from the forward pass.
    /// Shape: `[batch, n_q, n_k]`.
    attn_weights: Tensor<T>,
}

impl<T: Float> GradFn<T> for FlexAttentionBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        // grad_output: [batch, n_q, d_v]
        // attn_weights: [batch, n_q, n_k]
        // Q: [batch, n_q, d], K: [batch, n_k, d], V: [batch, n_k, d_v]
        //
        // Forward: S = Q @ K^T / sqrt(d), A = softmax(S), O = A @ V
        //
        // dV = A^T @ dO           : [batch, n_k, d_v]
        // dA = dO @ V^T           : [batch, n_q, n_k]
        // dS = A * (dA - sum(dA * A, axis=-1, keepdim))  (softmax backward)
        // dQ = dS @ K / sqrt(d)   : [batch, n_q, d]
        // dK = dS^T @ Q / sqrt(d) : [batch, n_k, d]

        let batch = self.query.shape()[0];
        let n_q = self.query.shape()[1];
        let d = self.query.shape()[2];
        let n_k = self.key.shape()[1];
        let d_v = self.value.shape()[2];

        let scale = T::from(1.0 / (d as f64).sqrt()).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        let go_data = grad_output.data()?;
        let attn_data = self.attn_weights.data()?;
        let q_data = self.query.data()?;
        let k_data = self.key.data()?;
        let v_data = self.value.data()?;

        let mut dq_data = vec![zero; batch * n_q * d];
        let mut dk_data = vec![zero; batch * n_k * d];
        let mut dv_data = vec![zero; batch * n_k * d_v];

        for b in 0..batch {
            let go_base = b * n_q * d_v;
            let attn_base = b * n_q * n_k;
            let q_base = b * n_q * d;
            let k_base = b * n_k * d;
            let v_base = b * n_k * d_v;

            // dV = A^T @ dO : [n_k, d_v]
            for j in 0..n_k {
                for dv in 0..d_v {
                    let mut acc = zero;
                    for i in 0..n_q {
                        acc += attn_data[attn_base + i * n_k + j] * go_data[go_base + i * d_v + dv];
                    }
                    dv_data[v_base + j * d_v + dv] = acc;
                }
            }

            // dA = dO @ V^T : [n_q, n_k]
            let mut da = vec![zero; n_q * n_k];
            for i in 0..n_q {
                for j in 0..n_k {
                    let mut acc = zero;
                    for dv in 0..d_v {
                        acc += go_data[go_base + i * d_v + dv] * v_data[v_base + j * d_v + dv];
                    }
                    da[i * n_k + j] = acc;
                }
            }

            // Softmax backward: dS = A * (dA - sum(dA * A, axis=-1, keepdim))
            let mut ds = vec![zero; n_q * n_k];
            for i in 0..n_q {
                // sum_j(dA_ij * A_ij)
                let mut dot_sum = zero;
                for j in 0..n_k {
                    dot_sum += da[i * n_k + j] * attn_data[attn_base + i * n_k + j];
                }
                for j in 0..n_k {
                    ds[i * n_k + j] =
                        attn_data[attn_base + i * n_k + j] * (da[i * n_k + j] - dot_sum);
                }
            }

            // dQ = dS @ K * scale : [n_q, d]
            for i in 0..n_q {
                for dd in 0..d {
                    let mut acc = zero;
                    for j in 0..n_k {
                        acc += ds[i * n_k + j] * k_data[k_base + j * d + dd];
                    }
                    dq_data[q_base + i * d + dd] = acc * scale;
                }
            }

            // dK = dS^T @ Q * scale : [n_k, d]
            for j in 0..n_k {
                for dd in 0..d {
                    let mut acc = zero;
                    for i in 0..n_q {
                        acc += ds[i * n_k + j] * q_data[q_base + i * d + dd];
                    }
                    dk_data[k_base + j * d + dd] = acc * scale;
                }
            }
        }

        let dq = if self.query.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(dq_data),
                self.query.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        let dk = if self.key.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(dk_data),
                self.key.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        let dv = if self.value.requires_grad() {
            Some(Tensor::from_storage(
                TensorStorage::cpu(dv_data),
                self.value.shape().to_vec(),
                false,
            )?)
        } else {
            None
        };

        Ok(vec![dq, dk, dv])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.query, &self.key, &self.value]
    }

    fn name(&self) -> &'static str {
        "FlexAttentionBackward"
    }
}

// ===========================================================================
// flex_attention — public API
// ===========================================================================

/// Flexible scaled dot-product attention with composable score modifications.
///
/// Computes:
/// ```text
/// scores = Q @ K^T / sqrt(d_k)
/// scores = score_mod(scores, b, h, q_idx, kv_idx)  // if score_mod is provided
/// attn_weights = softmax(scores, mask)
/// output = attn_weights @ V
/// ```
///
/// # Arguments
///
/// - `query` - Query tensor of shape `[batch, n_q, d]`.
/// - `key` - Key tensor of shape `[batch, n_k, d]`.
/// - `value` - Value tensor of shape `[batch, n_k, d_v]`.
/// - `score_mod` - Optional function that modifies attention scores before softmax.
///   Takes `(score, batch_idx, head_idx, q_idx, kv_idx)` tensors and returns
///   modified score. All index tensors are scalar (shape `[]` or `[1]`).
///   The `score` tensor is also scalar. This is called per-element.
/// - `block_mask` - Optional [`BlockMask`] defining sparse attention patterns.
///   Blocks not in the mask are skipped (treated as -inf before softmax).
///
/// # Returns
///
/// Output tensor of shape `[batch, n_q, d_v]`.
///
/// # Errors
///
/// Returns an error if input shapes are incompatible (batch mismatch, etc.).
///
/// # Autograd
///
/// Supports backward through Q, K, V. Score modifications are treated as
/// elementwise transforms on detached scores during backward.
pub fn flex_attention<T: Float>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    score_mod: Option<&ScoreModFn<T>>,
    block_mask: Option<&BlockMask>,
) -> FerrotorchResult<Tensor<T>> {
    // --- Validate shapes ---
    if query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "flex_attention: expected 3-D inputs [batch, seq, dim], \
                 got query {:?}, key {:?}, value {:?}",
                query.shape(),
                key.shape(),
                value.shape()
            ),
        });
    }

    let batch = query.shape()[0];
    let n_q = query.shape()[1];
    let d = query.shape()[2];
    let n_k = key.shape()[1];
    let d_v = value.shape()[2];

    if key.shape()[0] != batch || value.shape()[0] != batch {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "flex_attention: batch size mismatch: query={}, key={}, value={}",
                batch,
                key.shape()[0],
                value.shape()[0]
            ),
        });
    }

    if key.shape()[2] != d {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "flex_attention: Q dim ({d}) must match K dim ({})",
                key.shape()[2]
            ),
        });
    }

    if value.shape()[1] != n_k {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "flex_attention: K seq_len ({n_k}) must match V seq_len ({})",
                value.shape()[1]
            ),
        });
    }

    // Validate block_mask dimensions if provided.
    if let Some(bm) = block_mask {
        if bm.n_q != n_q || bm.n_k != n_k {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "flex_attention: BlockMask dimensions (n_q={}, n_k={}) \
                     don't match input (n_q={n_q}, n_k={n_k})",
                    bm.n_q, bm.n_k
                ),
            });
        }
    }

    let scale = T::from(1.0 / (d as f64).sqrt()).unwrap();
    let neg_inf = T::from(-1e30).unwrap();
    let zero = <T as num_traits::Zero>::zero();

    let q_data = query.data()?;
    let k_data = key.data()?;
    let v_data = value.data()?;

    let mut output_data = vec![zero; batch * n_q * d_v];
    // Store attention weights for backward pass.
    let mut all_attn_weights = vec![zero; batch * n_q * n_k];

    for b in 0..batch {
        let q_base = b * n_q * d;
        let k_base = b * n_k * d;
        let v_base = b * n_k * d_v;
        let o_base = b * n_q * d_v;
        let attn_base = b * n_q * n_k;

        // Compute scores = Q @ K^T * scale : [n_q, n_k]
        let mut scores = vec![zero; n_q * n_k];
        for i in 0..n_q {
            for j in 0..n_k {
                // Check block mask.
                if let Some(bm) = block_mask {
                    if !bm.allows_position(i, j) {
                        scores[i * n_k + j] = neg_inf;
                        continue;
                    }
                }

                let mut dot = zero;
                for dd in 0..d {
                    dot += q_data[q_base + i * d + dd] * k_data[k_base + j * d + dd];
                }
                scores[i * n_k + j] = dot * scale;
            }
        }

        // Apply score_mod if provided.
        if let Some(modifier) = &score_mod {
            for i in 0..n_q {
                for j in 0..n_k {
                    // Skip masked positions.
                    if let Some(bm) = block_mask {
                        if !bm.allows_position(i, j) {
                            continue;
                        }
                    }

                    let score_val = scores[i * n_k + j];
                    let score_tensor =
                        Tensor::from_storage(TensorStorage::cpu(vec![score_val]), vec![1], false)?;
                    let b_tensor = Tensor::from_storage(
                        TensorStorage::cpu(vec![T::from(b).unwrap()]),
                        vec![1],
                        false,
                    )?;
                    // Head index is 0 since flex_attention operates on a single head's
                    // Q/K/V. Multi-head dispatch is handled by the caller.
                    let h_tensor = Tensor::from_storage(
                        TensorStorage::cpu(vec![T::from(0).unwrap()]),
                        vec![1],
                        false,
                    )?;
                    let q_idx_tensor = Tensor::from_storage(
                        TensorStorage::cpu(vec![T::from(i).unwrap()]),
                        vec![1],
                        false,
                    )?;
                    let kv_idx_tensor = Tensor::from_storage(
                        TensorStorage::cpu(vec![T::from(j).unwrap()]),
                        vec![1],
                        false,
                    )?;

                    let modified = modifier(
                        &score_tensor,
                        &b_tensor,
                        &h_tensor,
                        &q_idx_tensor,
                        &kv_idx_tensor,
                    );
                    scores[i * n_k + j] = modified.data()?[0];
                }
            }
        }

        // Softmax per row.
        for i in 0..n_q {
            let row_start = i * n_k;

            // Find max for numerical stability.
            let mut row_max = neg_inf;
            for j in 0..n_k {
                if scores[row_start + j] > row_max {
                    row_max = scores[row_start + j];
                }
            }

            // Exp and sum.
            let mut sum_exp = zero;
            for j in 0..n_k {
                let e = (scores[row_start + j] - row_max).exp();
                all_attn_weights[attn_base + row_start + j] = e;
                sum_exp += e;
            }

            // Normalize.
            if sum_exp > zero {
                for j in 0..n_k {
                    all_attn_weights[attn_base + row_start + j] =
                        all_attn_weights[attn_base + row_start + j] / sum_exp;
                }
            }
        }

        // Output = attn_weights @ V : [n_q, d_v]
        for i in 0..n_q {
            for dv in 0..d_v {
                let mut acc = zero;
                for j in 0..n_k {
                    acc +=
                        all_attn_weights[attn_base + i * n_k + j] * v_data[v_base + j * d_v + dv];
                }
                output_data[o_base + i * d_v + dv] = acc;
            }
        }
    }

    let output_shape = vec![batch, n_q, d_v];

    // Attach backward if any input requires grad.
    let needs_grad = is_grad_enabled()
        && (query.requires_grad() || key.requires_grad() || value.requires_grad());

    if needs_grad {
        let attn_weights_tensor = Tensor::from_storage(
            TensorStorage::cpu(all_attn_weights),
            vec![batch, n_q, n_k],
            false,
        )?;

        Tensor::from_operation(
            TensorStorage::cpu(output_data),
            output_shape,
            Arc::new(FlexAttentionBackward {
                query: query.clone(),
                key: key.clone(),
                value: value.clone(),
                attn_weights: attn_weights_tensor,
            }),
        )
    } else {
        Tensor::from_storage(TensorStorage::cpu(output_data), output_shape, false)
    }
}

// ===========================================================================
// Common score_mod presets
// ===========================================================================

/// Creates a causal score modifier.
///
/// Returns a closure that sets `score = -inf` when `q_idx < kv_idx`,
/// implementing causal (autoregressive) masking.
///
/// # Example
///
/// ```ignore
/// let output = flex_attention(&q, &k, &v, Some(&causal_score_mod()), None)?;
/// ```
#[allow(clippy::type_complexity)]
pub fn causal_score_mod<T: Float>()
-> impl Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Tensor<T> {
    move |score, _b, _h, q_idx, kv_idx| {
        let q_pos = q_idx.data().unwrap()[0];
        let k_pos = kv_idx.data().unwrap()[0];
        if q_pos < k_pos {
            let neg_inf = T::from(-1e30).unwrap();
            Tensor::from_storage(TensorStorage::cpu(vec![neg_inf]), vec![1], false).unwrap()
        } else {
            score.clone()
        }
    }
}

/// Creates an ALiBi (Attention with Linear Biases) score modifier.
///
/// Adds a linear position-dependent bias: `score + slope * (q_idx - kv_idx)`.
/// Negative slope values penalize distant positions, implementing the
/// ALiBi mechanism from "Train Short, Test Long" (Press et al., 2022).
///
/// # Arguments
///
/// - `slope` - The per-head slope value. Typically geometric: `1/2^(8*h/H)`.
#[allow(clippy::type_complexity)]
pub fn alibi_score_mod<T: Float>(
    slope: T,
) -> impl Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Tensor<T> {
    move |score, _b, _h, q_idx, kv_idx| {
        let q_pos = q_idx.data().unwrap()[0];
        let k_pos = kv_idx.data().unwrap()[0];
        let bias = slope * (q_pos - k_pos);
        let new_score = score.data().unwrap()[0] + bias;
        Tensor::from_storage(TensorStorage::cpu(vec![new_score]), vec![1], false).unwrap()
    }
}

/// Creates a relative position bias score modifier.
///
/// Adds a bias looked up from a table indexed by `q_idx - kv_idx + max_dist`.
/// Positions outside `[-max_dist, max_dist]` are clamped.
///
/// # Arguments
///
/// - `bias_table` - 1-D tensor of shape `[2 * max_dist + 1]`.
/// - `max_dist` - Maximum relative distance.
#[allow(clippy::type_complexity)]
pub fn relative_position_bias_score_mod<T: Float>(
    bias_table: Tensor<T>,
    max_dist: usize,
) -> impl Fn(&Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>, &Tensor<T>) -> Tensor<T> {
    move |score, _b, _h, q_idx, kv_idx| {
        let q_pos = q_idx.data().unwrap()[0].to_f64().unwrap();
        let k_pos = kv_idx.data().unwrap()[0].to_f64().unwrap();
        let rel = (q_pos - k_pos) as isize;
        let clamped = rel.max(-(max_dist as isize)).min(max_dist as isize);
        let table_idx = (clamped + max_dist as isize) as usize;
        let table_data = bias_table.data().unwrap();
        let bias = table_data[table_idx.min(table_data.len() - 1)];
        let new_score = score.data().unwrap()[0] + bias;
        Tensor::from_storage(TensorStorage::cpu(vec![new_score]), vec![1], false).unwrap()
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::zeros;

    /// Absolute tolerance for floating-point comparisons.
    const ATOL: f64 = 1e-6;

    /// Create a deterministic test tensor.
    fn det_tensor(shape: &[usize], seed: u64) -> Tensor<f64> {
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        let mut state = seed;
        for _ in 0..numel {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let val = (state >> 33) as f64 / (u32::MAX as f64) * 2.0 - 1.0;
            data.push(val);
        }
        Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
    }

    /// Create a deterministic test tensor with requires_grad=true.
    fn det_tensor_grad(shape: &[usize], seed: u64) -> Tensor<f64> {
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        let mut state = seed;
        for _ in 0..numel {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let val = (state >> 33) as f64 / (u32::MAX as f64) * 2.0 - 1.0;
            data.push(val);
        }
        Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), true).unwrap()
    }

    // -----------------------------------------------------------------------
    // BlockMask tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_full_mask_all_active() {
        let bm = BlockMask::full_mask(8, 8, 4).unwrap();
        assert_eq!(bm.num_q_blocks(), 2);
        assert_eq!(bm.num_k_blocks(), 2);
        for q in 0..8 {
            for k in 0..8 {
                assert!(bm.allows_position(q, k));
            }
        }
    }

    #[test]
    fn test_causal_mask_lower_triangular() {
        let bm = BlockMask::causal_mask(4, 1).unwrap();
        // With block_size=1, each position is its own block.
        // Causal: allows (q, k) iff q >= k.
        assert!(bm.allows_position(0, 0));
        assert!(!bm.allows_position(0, 1));
        assert!(bm.allows_position(1, 0));
        assert!(bm.allows_position(1, 1));
        assert!(!bm.allows_position(1, 2));
        assert!(bm.allows_position(3, 0));
        assert!(bm.allows_position(3, 3));
    }

    #[test]
    fn test_causal_mask_block_level() {
        let bm = BlockMask::causal_mask(8, 4).unwrap();
        // 2 blocks: [0-3] and [4-7]
        // Block (0,0) active: q in [0,3], k in [0,3], so q=3 >= k=0 -> yes
        assert!(bm.is_active(0, 0));
        // Block (0,1) inactive: q in [0,3], k in [4,7], q=3 < k=4 -> no
        assert!(!bm.is_active(0, 1));
        // Block (1,0) active: q in [4,7], k in [0,3], q=4 >= k=0 -> yes
        assert!(bm.is_active(1, 0));
        // Block (1,1) active: q in [4,7], k in [4,7], q=7 >= k=4 -> yes
        assert!(bm.is_active(1, 1));
    }

    #[test]
    fn test_sliding_window_mask() {
        let bm = BlockMask::sliding_window_mask(6, 1, 1).unwrap();
        // Window size 1: each pos can attend to pos-1, pos, pos+1
        assert!(bm.allows_position(0, 0));
        assert!(bm.allows_position(0, 1));
        assert!(!bm.allows_position(0, 2));
        assert!(bm.allows_position(2, 1));
        assert!(bm.allows_position(2, 2));
        assert!(bm.allows_position(2, 3));
        assert!(!bm.allows_position(2, 4));
    }

    #[test]
    fn test_sliding_window_wider() {
        let bm = BlockMask::sliding_window_mask(8, 3, 1).unwrap();
        // Window size 3: attend within +/- 3 positions
        assert!(bm.allows_position(4, 1));
        assert!(bm.allows_position(4, 4));
        assert!(bm.allows_position(4, 7));
        assert!(!bm.allows_position(0, 4));
        assert!(!bm.allows_position(7, 3));
    }

    #[test]
    fn test_block_mask_zero_block_size_error() {
        assert!(BlockMask::full_mask(8, 8, 0).is_err());
        assert!(BlockMask::causal_mask(8, 0).is_err());
        assert!(BlockMask::sliding_window_mask(8, 2, 0).is_err());
    }

    #[test]
    fn test_block_mask_zero_window_error() {
        assert!(BlockMask::sliding_window_mask(8, 0, 4).is_err());
    }

    // -----------------------------------------------------------------------
    // flex_attention forward tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_flex_attention_basic_shape() {
        let q = det_tensor(&[2, 4, 8], 42);
        let k = det_tensor(&[2, 6, 8], 99);
        let v = det_tensor(&[2, 6, 8], 137);

        let out = flex_attention(&q, &k, &v, None, None).unwrap();
        assert_eq!(out.shape(), &[2, 4, 8]);
    }

    #[test]
    fn test_flex_attention_self_attention() {
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 4, 8], 99);
        let v = det_tensor(&[1, 4, 8], 137);

        let out = flex_attention(&q, &k, &v, None, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);

        // All values should be finite.
        let data = out.data().unwrap();
        for &v in data {
            assert!(v.is_finite(), "non-finite value in output: {v}");
        }
    }

    #[test]
    fn test_flex_attention_matches_standard() {
        // Without score_mod or block_mask, flex_attention should match
        // standard scaled dot-product attention.
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 4, 8], 99);
        let v = det_tensor(&[1, 4, 8], 137);

        let flex_out = flex_attention(&q, &k, &v, None, None).unwrap();

        // Compute reference standard attention manually.
        let std_out = crate::standard_attention(&q, &k, &v, false).unwrap();

        let flex_data = flex_out.data().unwrap();
        let std_data = std_out.data().unwrap();

        assert_eq!(flex_data.len(), std_data.len());
        for (i, (&f, &s)) in flex_data.iter().zip(std_data.iter()).enumerate() {
            let diff = (f - s).abs();
            assert!(
                diff < ATOL,
                "mismatch at index {i}: flex={f}, std={s}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_flex_attention_with_causal_score_mod() {
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 4, 8], 99);
        let v = det_tensor(&[1, 4, 8], 137);

        let causal = causal_score_mod::<f64>();
        let out = flex_attention(&q, &k, &v, Some(&causal), None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);

        // Compare with standard causal attention.
        let std_out = crate::standard_attention(&q, &k, &v, true).unwrap();

        let flex_data = out.data().unwrap();
        let std_data = std_out.data().unwrap();

        for (i, (&f, &s)) in flex_data.iter().zip(std_data.iter()).enumerate() {
            let diff = (f - s).abs();
            assert!(
                diff < ATOL,
                "causal mismatch at {i}: flex={f}, std={s}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_flex_attention_with_block_mask_full() {
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 4, 8], 99);
        let v = det_tensor(&[1, 4, 8], 137);

        let bm = BlockMask::full_mask(4, 4, 2).unwrap();
        let out_masked = flex_attention(&q, &k, &v, None, Some(&bm)).unwrap();
        let out_plain = flex_attention(&q, &k, &v, None, None).unwrap();

        // Full mask should give identical results.
        let a = out_masked.data().unwrap();
        let b = out_plain.data().unwrap();
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            assert!(diff < ATOL, "full mask mismatch at {i}: {va} vs {vb}");
        }
    }

    #[test]
    fn test_flex_attention_with_causal_block_mask() {
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 4, 8], 99);
        let v = det_tensor(&[1, 4, 8], 137);

        // Block-level causal mask with block_size=1 is exact causal masking.
        let bm = BlockMask::causal_mask(4, 1).unwrap();
        let out = flex_attention(&q, &k, &v, None, Some(&bm)).unwrap();

        let std_out = crate::standard_attention(&q, &k, &v, true).unwrap();

        let a = out.data().unwrap();
        let b = std_out.data().unwrap();
        for (i, (&va, &vb)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (va - vb).abs();
            assert!(
                diff < ATOL,
                "causal block mask mismatch at {i}: flex={va}, std={vb}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_flex_attention_with_alibi() {
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 4, 8], 99);
        let v = det_tensor(&[1, 4, 8], 137);

        let alibi = alibi_score_mod(-0.1f64);
        let out = flex_attention(&q, &k, &v, Some(&alibi), None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 8]);

        let data = out.data().unwrap();
        for &val in data {
            assert!(val.is_finite(), "ALiBi output has non-finite: {val}");
        }
    }

    #[test]
    fn test_flex_attention_shape_mismatch_errors() {
        // Batch mismatch.
        let q = det_tensor(&[2, 4, 8], 42);
        let k = det_tensor(&[3, 4, 8], 99);
        let v = det_tensor(&[3, 4, 8], 137);
        assert!(flex_attention(&q, &k, &v, None, None).is_err());

        // Q/K dim mismatch.
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 4, 16], 99);
        let v = det_tensor(&[1, 4, 16], 137);
        assert!(flex_attention(&q, &k, &v, None, None).is_err());

        // K/V seq_len mismatch.
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 6, 8], 99);
        let v = det_tensor(&[1, 4, 8], 137);
        assert!(flex_attention(&q, &k, &v, None, None).is_err());
    }

    #[test]
    fn test_flex_attention_2d_input_error() {
        let q = det_tensor(&[4, 8], 42);
        let k = det_tensor(&[4, 8], 99);
        let v = det_tensor(&[4, 8], 137);
        assert!(flex_attention(&q, &k, &v, None, None).is_err());
    }

    #[test]
    fn test_flex_attention_block_mask_dim_mismatch() {
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 6, 8], 99);
        let v = det_tensor(&[1, 6, 8], 137);

        // Block mask with n_q=4 but n_k=4, mismatches k seq_len of 6.
        let bm = BlockMask::full_mask(4, 4, 2).unwrap();
        assert!(flex_attention(&q, &k, &v, None, Some(&bm)).is_err());
    }

    #[test]
    fn test_flex_attention_single_position() {
        // seq_len=1 for both Q and K.
        let q = det_tensor(&[1, 1, 4], 42);
        let k = det_tensor(&[1, 1, 4], 99);
        let v = det_tensor(&[1, 1, 4], 137);

        let out = flex_attention(&q, &k, &v, None, None).unwrap();
        assert_eq!(out.shape(), &[1, 1, 4]);

        // With a single key position, attention weight = 1.0, so output = V.
        let out_data = out.data().unwrap();
        let v_data = v.data().unwrap();
        for (i, (&o, &v_val)) in out_data.iter().zip(v_data.iter()).enumerate() {
            let diff = (o - v_val).abs();
            assert!(
                diff < ATOL,
                "single position: out[{i}]={o} vs v[{i}]={v_val}"
            );
        }
    }

    #[test]
    fn test_flex_attention_with_requires_grad() {
        let q = det_tensor_grad(&[1, 4, 8], 42);
        let k = det_tensor_grad(&[1, 4, 8], 99);
        let v = det_tensor_grad(&[1, 4, 8], 137);

        let out = flex_attention(&q, &k, &v, None, None).unwrap();
        assert!(out.requires_grad());
        assert_eq!(out.shape(), &[1, 4, 8]);
    }

    #[test]
    fn test_flex_attention_different_dv() {
        // Q,K dim=8, V dim=16.
        let q = det_tensor(&[1, 4, 8], 42);
        let k = det_tensor(&[1, 6, 8], 99);
        let v = det_tensor(&[1, 6, 16], 137);

        let out = flex_attention(&q, &k, &v, None, None).unwrap();
        assert_eq!(out.shape(), &[1, 4, 16]);
    }

    #[test]
    fn test_flex_attention_uniform_attention() {
        // If Q and K are all zeros, scores are zero everywhere, softmax is
        // uniform (1/n_k), and output should be the mean of V rows.
        let q = zeros::<f64>(&[1, 2, 4]).unwrap();
        let k = zeros::<f64>(&[1, 3, 4]).unwrap();

        let v_data = vec![
            1.0, 0.0, 0.0, 0.0, // row 0
            0.0, 1.0, 0.0, 0.0, // row 1
            0.0, 0.0, 1.0, 0.0, // row 2
        ];
        let v = Tensor::from_storage(TensorStorage::cpu(v_data), vec![1, 3, 4], false).unwrap();

        let out = flex_attention(&q, &k, &v, None, None).unwrap();
        let out_data = out.data().unwrap();

        // Each output row should be mean of V rows = [1/3, 1/3, 1/3, 0]
        let expected = 1.0 / 3.0;
        for row in 0..2 {
            for col in 0..3 {
                let val = out_data[row * 4 + col];
                assert!(
                    (val - expected).abs() < ATOL,
                    "uniform attention [{row}][{col}]: {val} vs {expected}"
                );
            }
            assert!(
                out_data[row * 4 + 3].abs() < ATOL,
                "uniform attention [{row}][3] should be 0"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Score mod preset tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_relative_position_bias() {
        // Create a simple bias table: [0.0, 0.1, 0.2, 0.3, 0.4] for max_dist=2
        let bias_data = vec![0.0f64, 0.1, 0.2, 0.3, 0.4];
        let bias_table =
            Tensor::from_storage(TensorStorage::cpu(bias_data), vec![5], false).unwrap();

        let rpb = relative_position_bias_score_mod(bias_table, 2);

        // q_idx=2, kv_idx=0: rel=2, index=2+2=4 -> bias=0.4
        let score = Tensor::from_storage(TensorStorage::cpu(vec![1.0f64]), vec![1], false).unwrap();
        let b = Tensor::from_storage(TensorStorage::cpu(vec![0.0f64]), vec![1], false).unwrap();
        let h = Tensor::from_storage(TensorStorage::cpu(vec![0.0f64]), vec![1], false).unwrap();
        let q_idx = Tensor::from_storage(TensorStorage::cpu(vec![2.0f64]), vec![1], false).unwrap();
        let kv_idx =
            Tensor::from_storage(TensorStorage::cpu(vec![0.0f64]), vec![1], false).unwrap();

        let result = rpb(&score, &b, &h, &q_idx, &kv_idx);
        let val = result.data().unwrap()[0];
        assert!((val - 1.4).abs() < ATOL, "RPB: expected 1.4, got {val}");
    }

    #[test]
    fn test_alibi_mod_zero_distance() {
        let alibi = alibi_score_mod(-0.5f64);
        let score = Tensor::from_storage(TensorStorage::cpu(vec![1.0f64]), vec![1], false).unwrap();
        let b = Tensor::from_storage(TensorStorage::cpu(vec![0.0f64]), vec![1], false).unwrap();
        let h = Tensor::from_storage(TensorStorage::cpu(vec![0.0f64]), vec![1], false).unwrap();
        let q = Tensor::from_storage(TensorStorage::cpu(vec![3.0f64]), vec![1], false).unwrap();
        let k = Tensor::from_storage(TensorStorage::cpu(vec![3.0f64]), vec![1], false).unwrap();

        let result = alibi(&score, &b, &h, &q, &k);
        let val = result.data().unwrap()[0];
        // distance=0, so bias=0, score stays 1.0
        assert!(
            (val - 1.0).abs() < ATOL,
            "ALiBi zero dist: expected 1.0, got {val}"
        );
    }

    #[test]
    fn test_alibi_mod_negative_distance() {
        let alibi = alibi_score_mod(-0.5f64);
        let score = Tensor::from_storage(TensorStorage::cpu(vec![1.0f64]), vec![1], false).unwrap();
        let b = Tensor::from_storage(TensorStorage::cpu(vec![0.0f64]), vec![1], false).unwrap();
        let h = Tensor::from_storage(TensorStorage::cpu(vec![0.0f64]), vec![1], false).unwrap();
        let q = Tensor::from_storage(TensorStorage::cpu(vec![2.0f64]), vec![1], false).unwrap();
        let k = Tensor::from_storage(TensorStorage::cpu(vec![5.0f64]), vec![1], false).unwrap();

        let result = alibi(&score, &b, &h, &q, &k);
        let val = result.data().unwrap()[0];
        // distance = 2 - 5 = -3, bias = -0.5 * -3 = 1.5, score = 1.0 + 1.5 = 2.5
        assert!(
            (val - 2.5).abs() < ATOL,
            "ALiBi negative dist: expected 2.5, got {val}"
        );
    }
}
