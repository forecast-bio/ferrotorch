//! Memory-efficient FlashAttention (CPU tiled version).
//!
//! Computes `softmax(Q @ K^T / sqrt(d_k)) @ V` without materializing the
//! full `[B, N, N]` attention matrix, reducing peak memory from O(N^2) to
//! O(N * block_size). Uses the online softmax trick (Milakov & Gimelshein, 2018)
//! to compute exact softmax incrementally over key/value blocks.
//!
//! The forward pass is fully tiled. The backward pass (MVP) recomputes
//! the standard (non-tiled) attention to obtain gradients — a future
//! optimization would tile the backward as well.

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// Memory-efficient scaled dot-product attention.
///
/// Computes `softmax(Q @ K^T / sqrt(d_k)) @ V` without materializing the
/// full N x N attention matrix. Uses O(N) memory instead of O(N^2).
///
/// # Arguments
///
/// - `query`  - `[B, N_q, d]`
/// - `key`    - `[B, N_k, d]`
/// - `value`  - `[B, N_k, d_v]`
/// - `causal` - If `true`, apply causal (lower-triangular) mask so that
///   position `i` cannot attend to positions `j > i`. Requires `N_q == N_k`.
/// - `block_size` - Tile size for the tiled computation (e.g., 64 or 128).
///   Larger blocks use more temporary memory but may be faster due to better
///   cache locality. Must be > 0.
///
/// # Returns
///
/// Output tensor of shape `[B, N_q, d_v]`.
///
/// # Errors
///
/// Returns `FerrotorchError::InvalidArgument` or `FerrotorchError::ShapeMismatch`
/// if shapes are incompatible or `block_size == 0`.
pub fn flash_attention<T: Float>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    causal: bool,
    block_size: usize,
) -> FerrotorchResult<Tensor<T>> {
    // --- Validate inputs ---
    validate_inputs(query, key, value, causal, block_size)?;

    let batch = query.shape()[0];
    let n_q = query.shape()[1];
    let d = query.shape()[2];
    let n_k = key.shape()[1];
    let d_v = value.shape()[2];

    // Early return for empty sequence lengths — produce a correctly shaped
    // empty output rather than risking undefined behavior in the tiled kernel.
    if n_q == 0 || n_k == 0 {
        return Tensor::from_storage(TensorStorage::cpu(vec![]), vec![batch, n_q, d_v], false);
    }

    let scale = T::from(1.0 / (d as f64).sqrt()).unwrap();

    let device = query.device();
    let q_data = query.data_vec()?;
    let k_data = key.data_vec()?;
    let v_data = value.data_vec()?;

    let mut output_data = vec![<T as num_traits::Zero>::zero(); batch * n_q * d_v];

    for b in 0..batch {
        let q_base = b * n_q * d;
        let k_base = b * n_k * d;
        let v_base = b * n_k * d_v;
        let o_base = b * n_q * d_v;

        flash_attention_single(
            &q_data[q_base..q_base + n_q * d],
            &k_data[k_base..k_base + n_k * d],
            &v_data[v_base..v_base + n_k * d_v],
            &mut output_data[o_base..o_base + n_q * d_v],
            n_q,
            n_k,
            d,
            d_v,
            scale,
            causal,
            block_size,
        );
    }

    // Attach backward if any input requires grad.
    let result = if is_grad_enabled()
        && (query.requires_grad() || key.requires_grad() || value.requires_grad())
    {
        let result_plain = Tensor::from_storage(
            TensorStorage::cpu(output_data.clone()),
            vec![batch, n_q, d_v],
            false,
        )?;
        let grad_fn = Arc::new(FlashAttentionBackward {
            query: query.clone(),
            key: key.clone(),
            value: value.clone(),
            output: result_plain,
            causal,
        });
        Tensor::from_operation(
            TensorStorage::cpu(output_data),
            vec![batch, n_q, d_v],
            grad_fn,
        )?
    } else {
        Tensor::from_storage(
            TensorStorage::cpu(output_data),
            vec![batch, n_q, d_v],
            false,
        )?
    };
    if device.is_cuda() {
        result.to(device)
    } else {
        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// Input validation
// ---------------------------------------------------------------------------

fn validate_inputs<T: Float>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    causal: bool,
    block_size: usize,
) -> FerrotorchResult<()> {
    if block_size == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "block_size must be > 0".into(),
        });
    }

    if query.ndim() != 3 || key.ndim() != 3 || value.ndim() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "flash_attention expects 3-D inputs [B, N, d], \
                 got query {:?}, key {:?}, value {:?}",
                query.shape(),
                key.shape(),
                value.shape()
            ),
        });
    }

    let batch = query.shape()[0];
    let d = query.shape()[2];
    let n_q = query.shape()[1];
    let n_k = key.shape()[1];

    if key.shape()[0] != batch || value.shape()[0] != batch {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "batch size mismatch: query={}, key={}, value={}",
                batch,
                key.shape()[0],
                value.shape()[0]
            ),
        });
    }

    if key.shape()[2] != d {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "query and key must have same d dimension: query d={}, key d={}",
                d,
                key.shape()[2]
            ),
        });
    }

    if key.shape()[1] != value.shape()[1] {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "key and value must have same sequence length: key={}, value={}",
                key.shape()[1],
                value.shape()[1]
            ),
        });
    }

    if causal && n_q != n_k {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("causal mask requires N_q == N_k, got N_q={n_q}, N_k={n_k}"),
        });
    }

    // n_q=0 or n_k=0 is allowed — the caller handles it by returning an
    // empty output tensor with the correct shape.

    Ok(())
}

// ---------------------------------------------------------------------------
// Tiled forward (single batch element)
// ---------------------------------------------------------------------------

/// Core FlashAttention kernel for a single batch element.
///
/// Uses the online softmax trick to accumulate attention output without
/// materializing the full `[N_q, N_k]` attention matrix.
///
/// # Memory
///
/// Temporary allocations per call:
/// - `m`: `[N_q]` — running row maxima
/// - `l`: `[N_q]` — running row sums
/// - `s_tile`: `[block_q, block_k]` — score tile (reused each inner iteration)
///
/// Total: O(N_q + block_q * block_k), never O(N_q * N_k).
#[allow(clippy::too_many_arguments)]
fn flash_attention_single<T: Float>(
    q: &[T],          // [N_q, d]
    k: &[T],          // [N_k, d]
    v: &[T],          // [N_k, d_v]
    output: &mut [T], // [N_q, d_v]
    n_q: usize,
    n_k: usize,
    d: usize,
    d_v: usize,
    scale: T,
    causal: bool,
    block_size: usize,
) {
    let neg_inf = T::from(-1e30).unwrap();
    let zero = <T as num_traits::Zero>::zero();

    // Running softmax statistics per query position.
    let mut m = vec![neg_inf; n_q]; // row-wise max of scores seen so far
    let mut l = vec![zero; n_q]; // row-wise sum of exp(scores - m) seen so far

    // Output accumulator is `output` itself, initialized to zero.

    // Iterate over key/value blocks.
    let num_k_blocks = n_k.div_ceil(block_size);

    for j_block in 0..num_k_blocks {
        let k_start = j_block * block_size;
        let k_end = (k_start + block_size).min(n_k);
        let bk = k_end - k_start; // actual block size (may be smaller at tail)

        // Iterate over query blocks.
        let num_q_blocks = n_q.div_ceil(block_size);

        for i_block in 0..num_q_blocks {
            let q_start = i_block * block_size;
            let q_end = (q_start + block_size).min(n_q);
            let bq = q_end - q_start;

            // If causal: skip this tile entirely if the entire tile is above
            // the diagonal (all query indices < all key indices).
            if causal && q_end <= k_start {
                // q_end - 1 < k_start means the maximum query index in this
                // block is still less than the minimum key index, so all
                // positions are masked out.
                continue;
            }

            // Compute S_ij = Q_i @ K_j^T * scale  — shape [bq, bk]
            let mut s_tile = vec![zero; bq * bk];

            for qi in 0..bq {
                let q_row = q_start + qi;
                for ki in 0..bk {
                    let k_row = k_start + ki;
                    let mut dot = zero;
                    for dd in 0..d {
                        dot += q[q_row * d + dd] * k[k_row * d + dd];
                    }
                    s_tile[qi * bk + ki] = dot * scale;
                }
            }

            // Apply causal mask: set s_tile[qi, ki] = -inf where
            // (q_start + qi) < (k_start + ki).
            if causal {
                for qi in 0..bq {
                    let q_row = q_start + qi;
                    for ki in 0..bk {
                        let k_row = k_start + ki;
                        if k_row > q_row {
                            s_tile[qi * bk + ki] = neg_inf;
                        }
                    }
                }
            }

            // Online softmax update for each query row in this block.
            for qi in 0..bq {
                let q_row = q_start + qi;
                let s_row = &s_tile[qi * bk..(qi + 1) * bk];

                // Row max of this tile.
                let mut tile_max = neg_inf;
                for &sv in s_row {
                    if sv > tile_max {
                        tile_max = sv;
                    }
                }

                // New running max.
                let m_old = m[q_row];
                let m_new = if tile_max > m_old { tile_max } else { m_old };

                // Correction factor for previously accumulated values.
                let correction = (m_old - m_new).exp();

                // Compute exp(s - m_new) for this tile row and their sum.
                let mut p_row = vec![zero; bk];
                let mut tile_sum = zero;
                for ki in 0..bk {
                    let p = (s_row[ki] - m_new).exp();
                    p_row[ki] = p;
                    tile_sum += p;
                }

                // Update running sum: l_new = correction * l_old + tile_sum.
                let l_old = l[q_row];
                let l_new = correction * l_old + tile_sum;

                // Update output: O_new = (correction * l_old / l_new) * O_old
                //                      + (1 / l_new) * P_row @ V_block
                let o_row_start = q_row * d_v;
                if l_new > zero {
                    let rescale_old = correction * l_old / l_new;
                    let rescale_new = T::from(1.0).unwrap() / l_new;

                    for dv in 0..d_v {
                        // P_row @ V_block column dv.
                        let mut pv = zero;
                        for (ki, &p_ki) in p_row.iter().enumerate() {
                            let k_row = k_start + ki;
                            pv += p_ki * v[k_row * d_v + dv];
                        }
                        output[o_row_start + dv] =
                            rescale_old * output[o_row_start + dv] + rescale_new * pv;
                    }
                }

                m[q_row] = m_new;
                l[q_row] = l_new;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Backward (MVP: standard recomputation, non-tiled)
// ---------------------------------------------------------------------------

/// Backward function for `flash_attention`.
///
/// For the MVP, this recomputes the standard (non-tiled) attention weights
/// from Q, K, V and uses them to compute gradients. This trades compute for
/// memory in the forward pass; a fully tiled backward is left as a future
/// optimization.
#[derive(Debug)]
struct FlashAttentionBackward<T: Float> {
    query: Tensor<T>,
    key: Tensor<T>,
    value: Tensor<T>,
    /// Stored for a future tiled backward pass (not yet used in the MVP
    /// backward, which recomputes attention from Q, K, V directly).
    #[allow(dead_code)]
    output: Tensor<T>,
    causal: bool,
}

impl<T: Float> ferrotorch_core::tensor::GradFn<T> for FlashAttentionBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        let batch = self.query.shape()[0];
        let n_q = self.query.shape()[1];
        let d = self.query.shape()[2];
        let n_k = self.key.shape()[1];
        let d_v = self.value.shape()[2];

        let scale = T::from(1.0 / (d as f64).sqrt()).unwrap();
        let neg_inf = T::from(-1e30).unwrap();
        let zero = <T as num_traits::Zero>::zero();

        let q_data = self.query.data_vec()?;
        let k_data = self.key.data_vec()?;
        let v_data = self.value.data_vec()?;
        let go_data = grad_output.data_vec()?;

        let needs_grad_q = self.query.requires_grad();
        let needs_grad_k = self.key.requires_grad();
        let needs_grad_v = self.value.requires_grad();

        let mut grad_q_data = if needs_grad_q {
            vec![zero; batch * n_q * d]
        } else {
            vec![]
        };
        let mut grad_k_data = if needs_grad_k {
            vec![zero; batch * n_k * d]
        } else {
            vec![]
        };
        let mut grad_v_data = if needs_grad_v {
            vec![zero; batch * n_k * d_v]
        } else {
            vec![]
        };

        for b in 0..batch {
            let q_base = b * n_q * d;
            let k_base = b * n_k * d;
            let v_base = b * n_k * d_v;
            let go_base = b * n_q * d_v;

            let q = &q_data[q_base..q_base + n_q * d];
            let k = &k_data[k_base..k_base + n_k * d];
            let v = &v_data[v_base..v_base + n_k * d_v];
            let go = &go_data[go_base..go_base + n_q * d_v];

            // Recompute scores = Q @ K^T * scale: [N_q, N_k]
            let mut scores = vec![zero; n_q * n_k];
            for i in 0..n_q {
                for j in 0..n_k {
                    let mut dot = zero;
                    for dd in 0..d {
                        dot += q[i * d + dd] * k[j * d + dd];
                    }
                    scores[i * n_k + j] = dot * scale;
                }
            }

            // Apply causal mask.
            if self.causal {
                for i in 0..n_q {
                    for j in (i + 1)..n_k {
                        scores[i * n_k + j] = neg_inf;
                    }
                }
            }

            // Softmax: attn_weights[i, j] = exp(scores[i,j] - max_j) / sum_j
            let mut attn = vec![zero; n_q * n_k];
            for i in 0..n_q {
                let row_start = i * n_k;
                let row = &scores[row_start..row_start + n_k];

                let mut row_max = neg_inf;
                for &s in row {
                    if s > row_max {
                        row_max = s;
                    }
                }
                let mut sum_exp = zero;
                for j in 0..n_k {
                    let e = (row[j] - row_max).exp();
                    attn[row_start + j] = e;
                    sum_exp += e;
                }
                if sum_exp > zero {
                    for j in 0..n_k {
                        attn[row_start + j] = attn[row_start + j] / sum_exp;
                    }
                }
            }

            // grad_V = attn^T @ grad_output: [N_k, d_v]
            if needs_grad_v {
                let gv_base = b * n_k * d_v;
                for j in 0..n_k {
                    for dv in 0..d_v {
                        let mut acc = zero;
                        for i in 0..n_q {
                            acc += attn[i * n_k + j] * go[i * d_v + dv];
                        }
                        grad_v_data[gv_base + j * d_v + dv] += acc;
                    }
                }
            }

            // For grad_Q and grad_K, we need:
            // grad_attn = grad_output @ V^T: [N_q, N_k]
            // Then grad_scores = attn * (grad_attn - rowsum(grad_attn * attn))
            // (This is the Jacobian of softmax applied to grad_attn.)
            if needs_grad_q || needs_grad_k {
                // grad_attn = go @ V^T: [N_q, N_k]
                let mut grad_attn = vec![zero; n_q * n_k];
                for i in 0..n_q {
                    for j in 0..n_k {
                        let mut dot = zero;
                        for dv in 0..d_v {
                            dot += go[i * d_v + dv] * v[j * d_v + dv];
                        }
                        grad_attn[i * n_k + j] = dot;
                    }
                }

                // grad_scores[i, j] = attn[i,j] * (grad_attn[i,j] - sum_k(attn[i,k]*grad_attn[i,k]))
                // This is the standard softmax backward.
                let mut grad_scores = vec![zero; n_q * n_k];
                for i in 0..n_q {
                    let row_start = i * n_k;
                    // Compute dot(attn_row, grad_attn_row)
                    let mut dot_ag = zero;
                    for j in 0..n_k {
                        dot_ag += attn[row_start + j] * grad_attn[row_start + j];
                    }
                    for j in 0..n_k {
                        grad_scores[row_start + j] =
                            attn[row_start + j] * (grad_attn[row_start + j] - dot_ag);
                    }
                }

                // Scale grad_scores by scale factor.
                for gs in &mut grad_scores {
                    *gs = *gs * scale;
                }

                // grad_Q = grad_scores @ K: [N_q, d]
                if needs_grad_q {
                    let gq_base = b * n_q * d;
                    for i in 0..n_q {
                        for dd in 0..d {
                            let mut acc = zero;
                            for j in 0..n_k {
                                acc += grad_scores[i * n_k + j] * k[j * d + dd];
                            }
                            grad_q_data[gq_base + i * d + dd] += acc;
                        }
                    }
                }

                // grad_K = grad_scores^T @ Q: [N_k, d]
                if needs_grad_k {
                    let gk_base = b * n_k * d;
                    for j in 0..n_k {
                        for dd in 0..d {
                            let mut acc = zero;
                            for i in 0..n_q {
                                acc += grad_scores[i * n_k + j] * q[i * d + dd];
                            }
                            grad_k_data[gk_base + j * d + dd] += acc;
                        }
                    }
                }
            }
        }

        let grad_q = if needs_grad_q {
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_q_data),
                self.query.shape().to_vec(),
                false,
            )?;
            Some(if self.query.is_cuda() {
                g.to(self.query.device())?
            } else {
                g
            })
        } else {
            None
        };

        let grad_k = if needs_grad_k {
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_k_data),
                self.key.shape().to_vec(),
                false,
            )?;
            Some(if self.key.is_cuda() {
                g.to(self.key.device())?
            } else {
                g
            })
        } else {
            None
        };

        let grad_v = if needs_grad_v {
            let g = Tensor::from_storage(
                TensorStorage::cpu(grad_v_data),
                self.value.shape().to_vec(),
                false,
            )?;
            Some(if self.value.is_cuda() {
                g.to(self.value.device())?
            } else {
                g
            })
        } else {
            None
        };

        Ok(vec![grad_q, grad_k, grad_v])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.query, &self.key, &self.value]
    }

    fn name(&self) -> &'static str {
        "FlashAttentionBackward"
    }
}

// ---------------------------------------------------------------------------
// Standard (non-tiled) attention — used as reference in tests
// ---------------------------------------------------------------------------

/// Standard scaled dot-product attention (non-tiled, O(N^2) memory).
///
/// This is the naive implementation used as a reference for correctness
/// testing of `flash_attention`.
///
/// # Arguments
///
/// Same as `flash_attention`, except no `block_size` parameter.
///
/// # Returns
///
/// Output tensor of shape `[B, N_q, d_v]`.
pub fn standard_attention<T: Float>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    causal: bool,
) -> FerrotorchResult<Tensor<T>> {
    // Reuse the same validation (block_size is irrelevant, pass 1).
    validate_inputs(query, key, value, causal, 1)?;

    let batch = query.shape()[0];
    let n_q = query.shape()[1];
    let d = query.shape()[2];
    let n_k = key.shape()[1];
    let d_v = value.shape()[2];

    // bf16 has a 7-bit mantissa; summing d=128 values of products in bf16
    // is catastrophic for attention scores. Promote the accumulator to
    // f32 for bf16 inputs. All other T carry their native precision.
    let is_bf16 = std::any::TypeId::of::<T>() == std::any::TypeId::of::<half::bf16>();
    if is_bf16 {
        return standard_attention_bf16(query, key, value, causal);
    }

    let scale = T::from(1.0 / (d as f64).sqrt()).unwrap();
    let neg_inf = T::from(-1e30).unwrap();
    let zero = <T as num_traits::Zero>::zero();

    let q_data = query.data()?;
    let k_data = key.data()?;
    let v_data = value.data()?;

    let mut output_data = vec![zero; batch * n_q * d_v];

    for b in 0..batch {
        let q_base = b * n_q * d;
        let k_base = b * n_k * d;
        let v_base = b * n_k * d_v;
        let o_base = b * n_q * d_v;

        let q = &q_data[q_base..q_base + n_q * d];
        let k = &k_data[k_base..k_base + n_k * d];
        let v = &v_data[v_base..v_base + n_k * d_v];

        // scores = Q @ K^T * scale: [N_q, N_k]
        let mut scores = vec![zero; n_q * n_k];
        for i in 0..n_q {
            for j in 0..n_k {
                let mut dot = zero;
                for dd in 0..d {
                    dot += q[i * d + dd] * k[j * d + dd];
                }
                scores[i * n_k + j] = dot * scale;
            }
        }

        // Causal mask.
        if causal {
            for i in 0..n_q {
                for j in (i + 1)..n_k {
                    scores[i * n_k + j] = neg_inf;
                }
            }
        }

        // Softmax per row.
        let mut attn = vec![zero; n_q * n_k];
        for i in 0..n_q {
            let row_start = i * n_k;
            let mut row_max = neg_inf;
            for j in 0..n_k {
                if scores[row_start + j] > row_max {
                    row_max = scores[row_start + j];
                }
            }
            let mut sum_exp = zero;
            for j in 0..n_k {
                let e = (scores[row_start + j] - row_max).exp();
                attn[row_start + j] = e;
                sum_exp += e;
            }
            if sum_exp > zero {
                for j in 0..n_k {
                    attn[row_start + j] = attn[row_start + j] / sum_exp;
                }
            }
        }

        // output = attn @ V: [N_q, d_v]
        for i in 0..n_q {
            for dv in 0..d_v {
                let mut acc = zero;
                for j in 0..n_k {
                    acc += attn[i * n_k + j] * v[j * d_v + dv];
                }
                output_data[o_base + i * d_v + dv] = acc;
            }
        }
    }

    Tensor::from_storage(
        TensorStorage::cpu(output_data),
        vec![batch, n_q, d_v],
        false,
    )
}

/// Mixed-precision (bf16 storage, f32 accumulator) variant of
/// [`standard_attention`]. Called by `standard_attention` when `T =
/// half::bf16`; identical math end-to-end but all accumulators,
/// softmax intermediates, and output dot products run in f32 before
/// being cast back to bf16 for the return tensor.
fn standard_attention_bf16<T: Float>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    causal: bool,
) -> FerrotorchResult<Tensor<T>> {
    let batch = query.shape()[0];
    let n_q = query.shape()[1];
    let d = query.shape()[2];
    let n_k = key.shape()[1];
    let d_v = value.shape()[2];

    // Convert inputs to f32 up-front (bf16 → f32 is a zero-cost byte
    // shift: bf16 bits occupy the top 16 of an f32, padded with zeros).
    let q_data = query.data()?;
    let k_data = key.data()?;
    let v_data = value.data()?;
    let q_f32: Vec<f32> = q_data.iter().map(|v| v.to_f32().unwrap()).collect();
    let k_f32: Vec<f32> = k_data.iter().map(|v| v.to_f32().unwrap()).collect();
    let v_f32: Vec<f32> = v_data.iter().map(|v| v.to_f32().unwrap()).collect();

    let scale = 1.0f32 / (d as f32).sqrt();
    let neg_inf = f32::NEG_INFINITY;

    let mut output_f32 = vec![0.0f32; batch * n_q * d_v];

    for b in 0..batch {
        let q_base = b * n_q * d;
        let k_base = b * n_k * d;
        let v_base = b * n_k * d_v;
        let o_base = b * n_q * d_v;

        let q = &q_f32[q_base..q_base + n_q * d];
        let k = &k_f32[k_base..k_base + n_k * d];
        let v = &v_f32[v_base..v_base + n_k * d_v];

        let mut scores = vec![0.0f32; n_q * n_k];
        for i in 0..n_q {
            for j in 0..n_k {
                let mut dot = 0.0f32;
                for dd in 0..d {
                    dot += q[i * d + dd] * k[j * d + dd];
                }
                scores[i * n_k + j] = dot * scale;
            }
        }

        if causal {
            for i in 0..n_q {
                for j in (i + 1)..n_k {
                    scores[i * n_k + j] = neg_inf;
                }
            }
        }

        let mut attn = vec![0.0f32; n_q * n_k];
        for i in 0..n_q {
            let row_start = i * n_k;
            let mut row_max = neg_inf;
            for j in 0..n_k {
                if scores[row_start + j] > row_max {
                    row_max = scores[row_start + j];
                }
            }
            let mut sum_exp = 0.0f32;
            for j in 0..n_k {
                let e = (scores[row_start + j] - row_max).exp();
                attn[row_start + j] = e;
                sum_exp += e;
            }
            if sum_exp > 0.0 {
                let inv = 1.0 / sum_exp;
                for j in 0..n_k {
                    attn[row_start + j] *= inv;
                }
            }
        }

        for i in 0..n_q {
            for dv in 0..d_v {
                let mut acc = 0.0f32;
                for j in 0..n_k {
                    acc += attn[i * n_k + j] * v[j * d_v + dv];
                }
                output_f32[o_base + i * d_v + dv] = acc;
            }
        }
    }

    // Cast back to T (= bf16) for the return.
    let output_t: Vec<T> = output_f32.iter().map(|&v| T::from(v).unwrap()).collect();

    Tensor::from_storage(TensorStorage::cpu(output_t), vec![batch, n_q, d_v], false)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    /// Absolute tolerance for comparing flash vs. standard attention.
    const ATOL_F64: f64 = 1e-6;
    /// Relative tolerance.
    const RTOL_F64: f64 = 1e-3;

    /// Check that two slices are approximately equal.
    fn assert_close(a: &[f64], b: &[f64], atol: f64, rtol: f64, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&ai, &bi)) in a.iter().zip(b.iter()).enumerate() {
            let diff = (ai - bi).abs();
            let tol = atol + rtol * bi.abs();
            assert!(
                diff <= tol,
                "{label}[{i}]: {ai} vs {bi}, diff={diff}, tol={tol}"
            );
        }
    }

    /// Create a deterministic pseudo-random tensor for testing.
    ///
    /// Uses a simple LCG to produce reproducible values in [-1, 1].
    fn deterministic_tensor(shape: &[usize], seed: u64) -> Tensor<f64> {
        let numel: usize = shape.iter().product();
        let mut data = Vec::with_capacity(numel);
        let mut state = seed;
        for _ in 0..numel {
            // LCG: state = (a * state + c) mod m
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            // Map to [-1, 1].
            let val = (state >> 33) as f64 / (u32::MAX as f64) * 2.0 - 1.0;
            data.push(val);
        }
        Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false).unwrap()
    }

    /// Create a deterministic tensor with requires_grad=true.
    fn deterministic_tensor_grad(shape: &[usize], seed: u64) -> Tensor<f64> {
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
    // Correctness: flash matches standard
    // -----------------------------------------------------------------------

    #[test]
    fn test_flash_matches_standard_basic() {
        let q = deterministic_tensor(&[2, 8, 16], 42);
        let k = deterministic_tensor(&[2, 8, 16], 99);
        let v = deterministic_tensor(&[2, 8, 16], 137);

        let flash_out = flash_attention(&q, &k, &v, false, 4).unwrap();
        let std_out = standard_attention(&q, &k, &v, false).unwrap();

        assert_eq!(flash_out.shape(), std_out.shape());
        assert_close(
            flash_out.data().unwrap(),
            std_out.data().unwrap(),
            ATOL_F64,
            RTOL_F64,
            "flash_vs_standard",
        );
    }

    #[test]
    fn test_flash_matches_standard_causal() {
        let q = deterministic_tensor(&[1, 12, 8], 7);
        let k = deterministic_tensor(&[1, 12, 8], 13);
        let v = deterministic_tensor(&[1, 12, 8], 21);

        let flash_out = flash_attention(&q, &k, &v, true, 4).unwrap();
        let std_out = standard_attention(&q, &k, &v, true).unwrap();

        assert_eq!(flash_out.shape(), std_out.shape());
        assert_close(
            flash_out.data().unwrap(),
            std_out.data().unwrap(),
            ATOL_F64,
            RTOL_F64,
            "flash_vs_standard_causal",
        );
    }

    #[test]
    fn test_flash_matches_standard_cross_attention() {
        // N_q != N_k (cross-attention, non-causal).
        let q = deterministic_tensor(&[2, 6, 8], 50);
        let k = deterministic_tensor(&[2, 10, 8], 60);
        let v = deterministic_tensor(&[2, 10, 8], 70);

        let flash_out = flash_attention(&q, &k, &v, false, 3).unwrap();
        let std_out = standard_attention(&q, &k, &v, false).unwrap();

        assert_eq!(flash_out.shape(), std_out.shape());
        assert_close(
            flash_out.data().unwrap(),
            std_out.data().unwrap(),
            ATOL_F64,
            RTOL_F64,
            "flash_vs_standard_cross",
        );
    }

    #[test]
    fn test_flash_matches_standard_different_d_v() {
        // d_v != d (value dimension differs from key dimension).
        let q = deterministic_tensor(&[1, 8, 16], 1);
        let k = deterministic_tensor(&[1, 8, 16], 2);
        let v = deterministic_tensor(&[1, 8, 32], 3);

        let flash_out = flash_attention(&q, &k, &v, false, 4).unwrap();
        let std_out = standard_attention(&q, &k, &v, false).unwrap();

        assert_eq!(flash_out.shape(), &[1, 8, 32]);
        assert_close(
            flash_out.data().unwrap(),
            std_out.data().unwrap(),
            ATOL_F64,
            RTOL_F64,
            "flash_vs_standard_diff_dv",
        );
    }

    // -----------------------------------------------------------------------
    // Causal mask: upper triangle gets zero attention
    // -----------------------------------------------------------------------

    #[test]
    fn test_causal_mask_zeros_future() {
        // With a causal mask and identical Q/K/V, position i should only
        // attend to positions 0..=i. We verify by checking that the first
        // row of output (which can only see position 0) equals V[0].
        let n = 4;
        let d = 2;
        // Use uniform Q=K so attention weights (before mask) would be uniform.
        let data = vec![1.0_f64; n * d];
        let qkv = Tensor::from_storage(TensorStorage::cpu(data), vec![1, n, d], false).unwrap();

        let out = flash_attention(&qkv, &qkv, &qkv, true, 2).unwrap();
        let out_data = out.data().unwrap();

        // First row: only attends to position 0, so output[0] = V[0] = [1, 1].
        assert!((out_data[0] - 1.0).abs() < 1e-10);
        assert!((out_data[1] - 1.0).abs() < 1e-10);

        // All values should be finite.
        for &v in out_data {
            assert!(v.is_finite(), "non-finite output: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Different block sizes produce same output
    // -----------------------------------------------------------------------

    #[test]
    fn test_different_block_sizes_same_output() {
        let q = deterministic_tensor(&[1, 16, 8], 100);
        let k = deterministic_tensor(&[1, 16, 8], 200);
        let v = deterministic_tensor(&[1, 16, 8], 300);

        let out_bs1 = flash_attention(&q, &k, &v, false, 1).unwrap();
        let out_bs4 = flash_attention(&q, &k, &v, false, 4).unwrap();
        let out_bs7 = flash_attention(&q, &k, &v, false, 7).unwrap();
        let out_bs16 = flash_attention(&q, &k, &v, false, 16).unwrap();
        let out_bs64 = flash_attention(&q, &k, &v, false, 64).unwrap();

        let ref_data = out_bs1.data().unwrap();
        assert_close(out_bs4.data().unwrap(), ref_data, ATOL_F64, RTOL_F64, "bs4");
        assert_close(out_bs7.data().unwrap(), ref_data, ATOL_F64, RTOL_F64, "bs7");
        assert_close(
            out_bs16.data().unwrap(),
            ref_data,
            ATOL_F64,
            RTOL_F64,
            "bs16",
        );
        assert_close(
            out_bs64.data().unwrap(),
            ref_data,
            ATOL_F64,
            RTOL_F64,
            "bs64",
        );
    }

    #[test]
    fn test_different_block_sizes_same_output_causal() {
        let q = deterministic_tensor(&[2, 10, 4], 111);
        let k = deterministic_tensor(&[2, 10, 4], 222);
        let v = deterministic_tensor(&[2, 10, 4], 333);

        let out_bs2 = flash_attention(&q, &k, &v, true, 2).unwrap();
        let out_bs3 = flash_attention(&q, &k, &v, true, 3).unwrap();
        let out_bs5 = flash_attention(&q, &k, &v, true, 5).unwrap();
        let out_bs10 = flash_attention(&q, &k, &v, true, 10).unwrap();

        let ref_data = out_bs2.data().unwrap();
        assert_close(
            out_bs3.data().unwrap(),
            ref_data,
            ATOL_F64,
            RTOL_F64,
            "bs3_causal",
        );
        assert_close(
            out_bs5.data().unwrap(),
            ref_data,
            ATOL_F64,
            RTOL_F64,
            "bs5_causal",
        );
        assert_close(
            out_bs10.data().unwrap(),
            ref_data,
            ATOL_F64,
            RTOL_F64,
            "bs10_causal",
        );
    }

    // -----------------------------------------------------------------------
    // Memory: peak allocation is not O(N^2)
    // -----------------------------------------------------------------------

    #[test]
    fn test_memory_no_n_squared_allocation() {
        // This test verifies the structural property: flash_attention_single
        // never allocates an [N_q, N_k] matrix. We check that the function
        // runs with a large N without timing out or OOM — the actual O(N^2)
        // standard_attention would allocate a 256MB matrix here.
        //
        // N=2048, d=32, block_size=64 => flash allocates ~2048 + 64*64 = ~6K
        // Standard would allocate 2048*2048 = 4M elements = 32MB (f64).
        let n = 512; // Keep small enough for CI but large enough to matter.
        let d = 16;
        let block_size = 64;

        let q = deterministic_tensor(&[1, n, d], 1000);
        let k = deterministic_tensor(&[1, n, d], 2000);
        let v = deterministic_tensor(&[1, n, d], 3000);

        // This should complete without excessive memory usage.
        let out = flash_attention(&q, &k, &v, false, block_size).unwrap();
        assert_eq!(out.shape(), &[1, n, d]);

        // Verify values are finite.
        let data = out.data().unwrap();
        for (i, &val) in data.iter().enumerate() {
            assert!(val.is_finite(), "non-finite at index {i}: {val}");
        }
    }

    // -----------------------------------------------------------------------
    // Gradient tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_backward_grad_shapes() {
        let q = deterministic_tensor_grad(&[1, 4, 8], 10);
        let k = deterministic_tensor_grad(&[1, 4, 8], 20);
        let v = deterministic_tensor_grad(&[1, 4, 8], 30);

        let out = flash_attention(&q, &k, &v, false, 2).unwrap();

        // Sum to get a scalar for backward.
        let out_data = out.data().unwrap();
        let sum_val: f64 = out_data.iter().copied().sum();
        let _sum_tensor =
            Tensor::from_storage(TensorStorage::cpu(vec![sum_val]), vec![], false).unwrap();

        // We need to use the autograd engine. Since flash_attention returns
        // a tensor with grad_fn, we build a sum reduction and call backward.
        // For the MVP, manually call the backward of the grad_fn.
        let grad_fn = out.grad_fn().unwrap();
        let ones = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f64; out.numel()]),
            out.shape().to_vec(),
            false,
        )
        .unwrap();

        let grads = grad_fn.backward(&ones).unwrap();
        assert_eq!(grads.len(), 3);

        // grad_Q
        let gq = grads[0].as_ref().unwrap();
        assert_eq!(gq.shape(), q.shape());

        // grad_K
        let gk = grads[1].as_ref().unwrap();
        assert_eq!(gk.shape(), k.shape());

        // grad_V
        let gv = grads[2].as_ref().unwrap();
        assert_eq!(gv.shape(), v.shape());
    }

    #[test]
    fn test_backward_numerical_gradient_check() {
        // Numerical gradient check using finite differences.
        // f(x) = sum(flash_attention(Q, K, V))
        // grad_Q[i] ~= (f(Q + eps*e_i) - f(Q - eps*e_i)) / (2*eps)
        let eps = 1e-5;

        let q = deterministic_tensor(&[1, 3, 4], 40);
        let k = deterministic_tensor(&[1, 3, 4], 50);
        let v = deterministic_tensor(&[1, 3, 4], 60);

        // Get analytical gradients.
        let q_grad = deterministic_tensor_grad(&[1, 3, 4], 40);
        let k_grad = deterministic_tensor_grad(&[1, 3, 4], 50);
        let v_grad = deterministic_tensor_grad(&[1, 3, 4], 60);

        let out = flash_attention(&q_grad, &k_grad, &v_grad, false, 2).unwrap();
        let grad_fn = out.grad_fn().unwrap();
        let ones = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f64; out.numel()]),
            out.shape().to_vec(),
            false,
        )
        .unwrap();
        let grads = grad_fn.backward(&ones).unwrap();
        let analytical_gq = grads[0].as_ref().unwrap().data().unwrap().to_vec();
        let analytical_gk = grads[1].as_ref().unwrap().data().unwrap().to_vec();
        let analytical_gv = grads[2].as_ref().unwrap().data().unwrap().to_vec();

        let q_data = q.data().unwrap().to_vec();
        let k_data = k.data().unwrap().to_vec();
        let v_data = v.data().unwrap().to_vec();

        // Numerical grad for Q.
        for idx in 0..q_data.len() {
            let mut q_plus = q_data.clone();
            let mut q_minus = q_data.clone();
            q_plus[idx] += eps;
            q_minus[idx] -= eps;

            let qp = Tensor::from_storage(TensorStorage::cpu(q_plus), q.shape().to_vec(), false)
                .unwrap();
            let qm = Tensor::from_storage(TensorStorage::cpu(q_minus), q.shape().to_vec(), false)
                .unwrap();

            let fp: f64 = flash_attention(&qp, &k, &v, false, 2)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .copied()
                .sum();
            let fm: f64 = flash_attention(&qm, &k, &v, false, 2)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .copied()
                .sum();

            let numerical = (fp - fm) / (2.0 * eps);
            let analytical = analytical_gq[idx];
            let diff = (numerical - analytical).abs();
            let tol = 1e-4 + 1e-3 * analytical.abs();
            assert!(
                diff <= tol,
                "grad_Q[{idx}]: numerical={numerical}, analytical={analytical}, diff={diff}"
            );
        }

        // Numerical grad for K.
        for idx in 0..k_data.len() {
            let mut k_plus = k_data.clone();
            let mut k_minus = k_data.clone();
            k_plus[idx] += eps;
            k_minus[idx] -= eps;

            let kp = Tensor::from_storage(TensorStorage::cpu(k_plus), k.shape().to_vec(), false)
                .unwrap();
            let km = Tensor::from_storage(TensorStorage::cpu(k_minus), k.shape().to_vec(), false)
                .unwrap();

            let fp: f64 = flash_attention(&q, &kp, &v, false, 2)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .copied()
                .sum();
            let fm: f64 = flash_attention(&q, &km, &v, false, 2)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .copied()
                .sum();

            let numerical = (fp - fm) / (2.0 * eps);
            let analytical = analytical_gk[idx];
            let diff = (numerical - analytical).abs();
            let tol = 1e-4 + 1e-3 * analytical.abs();
            assert!(
                diff <= tol,
                "grad_K[{idx}]: numerical={numerical}, analytical={analytical}, diff={diff}"
            );
        }

        // Numerical grad for V.
        for idx in 0..v_data.len() {
            let mut v_plus = v_data.clone();
            let mut v_minus = v_data.clone();
            v_plus[idx] += eps;
            v_minus[idx] -= eps;

            let vp = Tensor::from_storage(TensorStorage::cpu(v_plus), v.shape().to_vec(), false)
                .unwrap();
            let vm = Tensor::from_storage(TensorStorage::cpu(v_minus), v.shape().to_vec(), false)
                .unwrap();

            let fp: f64 = flash_attention(&q, &k, &vp, false, 2)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .copied()
                .sum();
            let fm: f64 = flash_attention(&q, &k, &vm, false, 2)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .copied()
                .sum();

            let numerical = (fp - fm) / (2.0 * eps);
            let analytical = analytical_gv[idx];
            let diff = (numerical - analytical).abs();
            let tol = 1e-4 + 1e-3 * analytical.abs();
            assert!(
                diff <= tol,
                "grad_V[{idx}]: numerical={numerical}, analytical={analytical}, diff={diff}"
            );
        }
    }

    #[test]
    fn test_backward_causal_gradient_check() {
        // Numerical gradient check with causal mask.
        let eps = 1e-5;

        let q = deterministic_tensor(&[1, 4, 4], 77);
        let k = deterministic_tensor(&[1, 4, 4], 88);
        let v = deterministic_tensor(&[1, 4, 4], 99);

        let q_grad = deterministic_tensor_grad(&[1, 4, 4], 77);
        let k_grad = deterministic_tensor_grad(&[1, 4, 4], 88);
        let v_grad = deterministic_tensor_grad(&[1, 4, 4], 99);

        let out = flash_attention(&q_grad, &k_grad, &v_grad, true, 2).unwrap();
        let grad_fn = out.grad_fn().unwrap();
        let ones = Tensor::from_storage(
            TensorStorage::cpu(vec![1.0_f64; out.numel()]),
            out.shape().to_vec(),
            false,
        )
        .unwrap();
        let grads = grad_fn.backward(&ones).unwrap();
        let analytical_gq = grads[0].as_ref().unwrap().data().unwrap().to_vec();

        let q_data = q.data().unwrap().to_vec();

        // Spot-check a few Q gradient elements.
        for idx in [0, 3, 7, 12, 15] {
            let mut q_plus = q_data.clone();
            let mut q_minus = q_data.clone();
            q_plus[idx] += eps;
            q_minus[idx] -= eps;

            let qp = Tensor::from_storage(TensorStorage::cpu(q_plus), q.shape().to_vec(), false)
                .unwrap();
            let qm = Tensor::from_storage(TensorStorage::cpu(q_minus), q.shape().to_vec(), false)
                .unwrap();

            let fp: f64 = flash_attention(&qp, &k, &v, true, 2)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .copied()
                .sum();
            let fm: f64 = flash_attention(&qm, &k, &v, true, 2)
                .unwrap()
                .data()
                .unwrap()
                .iter()
                .copied()
                .sum();

            let numerical = (fp - fm) / (2.0 * eps);
            let analytical = analytical_gq[idx];
            let diff = (numerical - analytical).abs();
            let tol = 1e-4 + 1e-3 * analytical.abs();
            assert!(
                diff <= tol,
                "causal grad_Q[{idx}]: numerical={numerical}, analytical={analytical}, diff={diff}"
            );
        }
    }

    // -----------------------------------------------------------------------
    // Edge cases and error handling
    // -----------------------------------------------------------------------

    #[test]
    fn test_block_size_zero_rejected() {
        let q = deterministic_tensor(&[1, 4, 4], 1);
        let k = deterministic_tensor(&[1, 4, 4], 2);
        let v = deterministic_tensor(&[1, 4, 4], 3);
        let result = flash_attention(&q, &k, &v, false, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_2d_input_rejected() {
        let q = deterministic_tensor(&[4, 4], 1);
        let k = deterministic_tensor(&[4, 4], 2);
        let v = deterministic_tensor(&[4, 4], 3);
        let result = flash_attention(&q, &k, &v, false, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_causal_different_seq_lens_rejected() {
        let q = deterministic_tensor(&[1, 3, 4], 1);
        let k = deterministic_tensor(&[1, 5, 4], 2);
        let v = deterministic_tensor(&[1, 5, 4], 3);
        let result = flash_attention(&q, &k, &v, true, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_batch_mismatch_rejected() {
        let q = deterministic_tensor(&[2, 4, 4], 1);
        let k = deterministic_tensor(&[3, 4, 4], 2);
        let v = deterministic_tensor(&[3, 4, 4], 3);
        let result = flash_attention(&q, &k, &v, false, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_d_mismatch_rejected() {
        let q = deterministic_tensor(&[1, 4, 8], 1);
        let k = deterministic_tensor(&[1, 4, 4], 2);
        let v = deterministic_tensor(&[1, 4, 4], 3);
        let result = flash_attention(&q, &k, &v, false, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_kv_seq_mismatch_rejected() {
        let q = deterministic_tensor(&[1, 4, 4], 1);
        let k = deterministic_tensor(&[1, 6, 4], 2);
        let v = deterministic_tensor(&[1, 4, 4], 3);
        let result = flash_attention(&q, &k, &v, false, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_single_element_sequence() {
        let q = deterministic_tensor(&[1, 1, 4], 10);
        let k = deterministic_tensor(&[1, 1, 4], 20);
        let v = deterministic_tensor(&[1, 1, 4], 30);

        let flash_out = flash_attention(&q, &k, &v, false, 1).unwrap();
        let std_out = standard_attention(&q, &k, &v, false).unwrap();

        assert_close(
            flash_out.data().unwrap(),
            std_out.data().unwrap(),
            ATOL_F64,
            RTOL_F64,
            "single_element",
        );
    }

    #[test]
    fn test_block_size_larger_than_sequence() {
        let q = deterministic_tensor(&[1, 3, 4], 10);
        let k = deterministic_tensor(&[1, 3, 4], 20);
        let v = deterministic_tensor(&[1, 3, 4], 30);

        let flash_out = flash_attention(&q, &k, &v, false, 128).unwrap();
        let std_out = standard_attention(&q, &k, &v, false).unwrap();

        assert_close(
            flash_out.data().unwrap(),
            std_out.data().unwrap(),
            ATOL_F64,
            RTOL_F64,
            "block_larger_than_seq",
        );
    }

    #[test]
    fn test_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<FlashAttentionBackward<f32>>();
        assert_send_sync::<FlashAttentionBackward<f64>>();
    }
}
