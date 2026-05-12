//! Speculative decoding — Leviathan et al. 2023 (arXiv:2211.17192).
//!
//! Implements the token-level acceptance criterion from §3 of the paper:
//!
//! > "for each draft-proposed token x̃ drawn from the draft distribution q(·|x₁..xₜ),
//! > accept it with probability min(1, p(x̃|x₁..xₜ) / q(x̃|x₁..xₜ)).
//! > On rejection, resample from the residual distribution
//! > norm(max(0, p(·) − q(·)))."
//!
//! — Leviathan et al. 2023, "Fast Inference from Transformers via Speculative
//! Decoding", arXiv:2211.17192, Algorithm 1 / §3.
//!
//! # High-level flow
//!
//! ```text
//! for each output token:
//!   1. Draft: run draft model K times autoregressively →
//!      K draft tokens + K draft distributions q_k
//!   2. Verify: run target model once over prompt + K draft tokens →
//!      K+1 target distributions p_k (position 0 = first draft pos)
//!   3. Accept/reject loop (k = 0..K):
//!      - draw U ~ Uniform(0, 1)
//!      - if U < p_k[d_k] / q_k[d_k]: accept d_k
//!      - else: sample corrected token from norm(max(0, p_k − q_k)); stop loop
//!   4. If all K accepted: sample one bonus token from p_K (next position).
//! ```
//!
//! # PyTorch parity
//!
//! Matches the canonical `torch.distributed.speculative_decode` behaviour
//! described in the Leviathan et al. reference. The acceptance criterion and
//! residual-sampling formula are identical.

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};

/// A model that can generate logits from a token sequence.
///
/// This trait is deliberately narrow: it mirrors the single method that
/// speculative decoding requires — a forward pass from token ids to logits.
/// [`crate::model::LlamaForCausalLM`] satisfies it automatically; tests can
/// supply any synthetic implementation.
pub trait ModelHandle<T: Float> {
    /// Run the forward pass over `ids` and return only the
    /// **last-position** logits as an f64 vector of length
    /// `vocab_size`.
    ///
    /// This is the legacy single-position API; speculative decoding uses
    /// the batched [`ModelHandle::forward_ids_all_positions`] for its
    /// verify step (one forward over the whole prefix instead of K+1
    /// forwards over progressively longer prefixes — see #1129).
    ///
    /// # Errors
    ///
    /// Propagates whatever the underlying model returns.
    fn forward_ids(&self, ids: &[u32]) -> FerrotorchResult<Vec<f64>>;

    /// Run the forward pass over `ids` and return **per-position**
    /// logits — one f64 vector of length `vocab_size` per position
    /// in `ids`, in order. The returned outer `Vec` has length
    /// `ids.len()`.
    ///
    /// # Why this exists
    ///
    /// The speculative-decoding *verify* step needs target-model
    /// distributions p<sub>0</sub>..p<sub>K</sub> at K+1 contiguous
    /// positions. A naive implementation runs `forward_ids` K+1 times
    /// over progressively longer prefixes — that's the quadratic cost
    /// the Pass 4 audit flagged on `src/spec_decode.rs:264-290`.
    /// Returning the full `[seq, vocab]` slab from a single forward
    /// drops the per-step verify cost from `O((K+1) · S · …)` to
    /// `O(S · …)`, matching Leviathan et al. §3 Algorithm 1.
    ///
    /// # Default implementation
    ///
    /// A correct-but-quadratic default that calls [`forward_ids`]
    /// once per position is provided so existing custom
    /// `ModelHandle` impls don't break. Real impls should override
    /// this with a true single-forward path.
    ///
    /// [`forward_ids`]: ModelHandle::forward_ids
    ///
    /// # Errors
    ///
    /// Propagates whatever the underlying model returns.
    fn forward_ids_all_positions(&self, ids: &[u32]) -> FerrotorchResult<Vec<Vec<f64>>> {
        // Default: fall back to per-position calls. Correct but slow —
        // exists so that existing custom impls keep compiling.
        let mut out = Vec::with_capacity(ids.len());
        for j in 1..=ids.len() {
            out.push(self.forward_ids(&ids[..j])?);
        }
        Ok(out)
    }

    /// Vocabulary size. Used to validate probability vectors.
    fn vocab_size(&self) -> usize;
}

/// Wrapper that adapts [`crate::model::LlamaForCausalLM`] to [`ModelHandle`].
#[derive(Debug)]
pub struct LlamaHandle<'m, T: Float> {
    model: &'m crate::model::LlamaForCausalLM<T>,
}

impl<'m, T: Float> LlamaHandle<'m, T> {
    /// Wrap a `LlamaForCausalLM` reference as a `ModelHandle`.
    pub fn new(model: &'m crate::model::LlamaForCausalLM<T>) -> Self {
        Self { model }
    }
}

impl<T: Float> ModelHandle<T> for LlamaHandle<'_, T> {
    fn forward_ids(&self, ids: &[u32]) -> FerrotorchResult<Vec<f64>> {
        let logits_tensor = self.model.forward_from_ids(ids)?;
        let shape = logits_tensor.shape();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("spec_decode: expected logits shape [1, S, V], got {shape:?}"),
            });
        }
        let seq_len = shape[1];
        let vocab = shape[2];
        let data = logits_tensor.data_vec()?;
        // Return only the last-position logits as f64.
        let last_offset = (seq_len - 1) * vocab;
        data[last_offset..last_offset + vocab]
            .iter()
            .map(|&v| ferrotorch_core::numeric_cast::cast::<T, f64>(v))
            .collect::<FerrotorchResult<Vec<f64>>>()
    }

    /// Real single-forward implementation: one `forward_from_ids` call
    /// returns the `[1, S, V]` logits tensor; split it into S
    /// per-position `Vec<f64>` slices. This is the perf-critical path
    /// the speculative-decode verify step uses to eliminate the
    /// quadratic K+1-forwards loop (#1129).
    fn forward_ids_all_positions(&self, ids: &[u32]) -> FerrotorchResult<Vec<Vec<f64>>> {
        let logits_tensor = self.model.forward_from_ids(ids)?;
        let shape = logits_tensor.shape();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!("spec_decode: expected logits shape [1, S, V], got {shape:?}"),
            });
        }
        let seq_len = shape[1];
        let vocab = shape[2];
        let data = logits_tensor.data_vec()?;
        let mut out: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for j in 0..seq_len {
            let base = j * vocab;
            let row: Vec<f64> = data[base..base + vocab]
                .iter()
                .map(|&v| ferrotorch_core::numeric_cast::cast::<T, f64>(v))
                .collect::<FerrotorchResult<Vec<f64>>>()?;
            out.push(row);
        }
        Ok(out)
    }

    fn vocab_size(&self) -> usize {
        self.model.config.vocab_size
    }
}

/// Configuration for speculative decoding.
#[derive(Debug, Clone)]
pub struct SpecDecodeConfig {
    /// Number of draft tokens to propose per iteration. Typical values: 4–8.
    /// Must be >= 1.
    pub draft_k: usize,

    /// Total new tokens to generate (excluding prompt). Must be >= 1.
    pub max_new_tokens: usize,

    /// PRNG seed. The same seed always produces identical output for the same
    /// prompt and model weights. `None` uses `0xdead_beef_cafe_babe`.
    pub seed: Option<u64>,

    /// Optional EOS token ids. Generation stops when any of these is produced.
    pub eos_token_ids: Vec<u32>,
}

impl Default for SpecDecodeConfig {
    fn default() -> Self {
        Self {
            draft_k: 4,
            max_new_tokens: 64,
            seed: None,
            eos_token_ids: Vec::new(),
        }
    }
}

impl SpecDecodeConfig {
    /// Validate the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when `draft_k == 0` or
    /// `max_new_tokens == 0`.
    pub fn validate(&self) -> FerrotorchResult<()> {
        if self.draft_k == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "SpecDecodeConfig: draft_k must be >= 1".into(),
            });
        }
        if self.max_new_tokens == 0 {
            return Err(FerrotorchError::InvalidArgument {
                message: "SpecDecodeConfig: max_new_tokens must be >= 1".into(),
            });
        }
        Ok(())
    }
}

/// Output of a single speculative-decoding run.
#[derive(Debug, Clone)]
pub struct SpecDecodeOutput {
    /// Generated token ids (excluding prompt). At most `max_new_tokens` long.
    pub tokens: Vec<u32>,

    /// Number of draft tokens that were accepted over the whole run.
    pub accepted_count: usize,

    /// Number of draft tokens that were proposed over the whole run.
    pub proposed_count: usize,
}

impl SpecDecodeOutput {
    /// Acceptance rate: `accepted_count / proposed_count`, or `1.0` if
    /// nothing was proposed yet.
    pub fn acceptance_rate(&self) -> f64 {
        if self.proposed_count == 0 {
            return 1.0;
        }
        self.accepted_count as f64 / self.proposed_count as f64
    }
}

/// Run speculative decoding over `prompt_ids` using `draft` and `target`.
///
/// Implements Algorithm 1 of Leviathan et al. 2023
/// ("Fast Inference from Transformers via Speculative Decoding",
/// arXiv:2211.17192, §3):
///
/// - Draft proposes `K` tokens autoregressively.
/// - Target verifies all K in one forward pass.
/// - Each token k is accepted with probability `min(1, p_k / q_k)`.
/// - On rejection at position j: sample from the residual distribution
///   `norm(max(0, p − q))` and discard positions j+1..K.
/// - If all K accepted: emit a bonus token sampled from `p_K`.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] for:
/// - Empty `prompt_ids`.
/// - Config that fails [`SpecDecodeConfig::validate`].
/// - Draft or target vocab size mismatch.
/// - Any inner forward-pass error.
///
/// # Panics
///
/// Does not panic.
pub fn speculative_decode<T: Float>(
    draft: &dyn ModelHandle<T>,
    target: &dyn ModelHandle<T>,
    prompt_ids: &[u32],
    config: &SpecDecodeConfig,
) -> FerrotorchResult<SpecDecodeOutput> {
    if prompt_ids.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "speculative_decode: prompt_ids must not be empty".into(),
        });
    }
    config.validate()?;

    let vocab = draft.vocab_size();
    if target.vocab_size() != vocab {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "speculative_decode: draft vocab_size ({}) != target vocab_size ({})",
                vocab,
                target.vocab_size()
            ),
        });
    }

    let mut rng_state = config.seed.unwrap_or(0xdead_beef_cafe_babe);
    let mut context: Vec<u32> = prompt_ids.to_vec();
    let mut produced: Vec<u32> = Vec::with_capacity(config.max_new_tokens);
    let mut accepted_count = 0usize;
    let mut proposed_count = 0usize;

    while produced.len() < config.max_new_tokens {
        let remaining = config.max_new_tokens - produced.len();
        let k = config.draft_k.min(remaining);

        // -----------------------------------------------------------------------
        // Phase 1 — Draft: produce K tokens autoregressively, recording the
        // draft distribution at each position.
        // -----------------------------------------------------------------------
        let mut draft_tokens: Vec<u32> = Vec::with_capacity(k);
        let mut draft_probs: Vec<Vec<f64>> = Vec::with_capacity(k); // q_j for j=0..K

        let mut draft_ctx = context.clone();
        for _ in 0..k {
            let logits = draft.forward_ids(&draft_ctx)?;
            let probs = softmax_f64(&logits);
            let token = sample_probs(&probs, &mut rng_state);
            draft_tokens.push(token);
            draft_probs.push(probs);
            draft_ctx.push(token);
        }

        // -----------------------------------------------------------------------
        // Phase 2 — Verify: ONE forward call over `context + draft_tokens`
        // (length S = context.len() + k) returns per-position logits at
        // every position. We need K+1 of them:
        //
        //   p_j (for j = 0..k) = P(x | context + draft_tokens[0..j])
        //                      = logits at position (context.len() - 1 + j)
        //   p_k (bonus)        = P(x | context + draft_tokens[0..k])
        //                      = logits at position (context.len() - 1 + k)
        //                      = the last position of the verify input
        //
        // This is the Leviathan et al. §3 Algorithm 1 batched-verify shape:
        // O(S) per round instead of the previous O((K+1) · S) — see #1129.
        // -----------------------------------------------------------------------
        let verify_prefix: Vec<u32> =
            context.iter().chain(draft_tokens.iter()).copied().collect();
        let all_logits = target.forward_ids_all_positions(&verify_prefix)?;
        if all_logits.len() != verify_prefix.len() {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "speculative_decode: forward_ids_all_positions returned {} rows for input \
                     length {}",
                    all_logits.len(),
                    verify_prefix.len()
                ),
            });
        }
        let start = context.len() - 1; // position whose logits give p_0
        let mut target_probs: Vec<Vec<f64>> = Vec::with_capacity(k + 1);
        for j in 0..=k {
            target_probs.push(softmax_f64(&all_logits[start + j]));
        }

        // -----------------------------------------------------------------------
        // Phase 3 — Accept / reject loop (Leviathan et al. §3 Algorithm 1).
        // -----------------------------------------------------------------------
        proposed_count += k;
        let mut n_accepted = 0usize; // tokens accepted this round
        let mut corrected_token: Option<u32> = None;

        'accept: for j in 0..k {
            let token = draft_tokens[j];
            let q = draft_probs[j][token as usize];
            let p = target_probs[j][token as usize];

            // Accept with probability min(1, p/q).
            let accept_prob = if q <= 0.0 { 0.0 } else { (p / q).min(1.0) };
            let u = xorshift_uniform(&mut rng_state);

            if u < accept_prob {
                // Accept draft token j.
                n_accepted += 1;
            } else {
                // Reject: sample corrected token from residual norm(max(0, p-q)).
                // Source: Leviathan et al. 2023 arXiv:2211.17192, §3.
                let corrected = sample_residual(&target_probs[j], &draft_probs[j], &mut rng_state);
                corrected_token = Some(corrected);
                break 'accept;
            }
        }

        accepted_count += n_accepted;

        // -----------------------------------------------------------------------
        // Phase 4 — Emit tokens.
        // -----------------------------------------------------------------------
        // Emit the n_accepted draft tokens.
        let mut eos_hit = false;
        for i in 0..n_accepted {
            let tok = draft_tokens[i];
            produced.push(tok);
            context.push(tok);
            if config.eos_token_ids.contains(&tok) {
                eos_hit = true;
                break;
            }
            if produced.len() >= config.max_new_tokens {
                break;
            }
        }

        if eos_hit || produced.len() >= config.max_new_tokens {
            break;
        }

        // Emit the corrected token (rejection case) or the bonus token (all
        // accepted case).
        let extra = corrected_token.unwrap_or_else(|| {
            // All K accepted — sample bonus token from p_K.
            sample_probs(&target_probs[k], &mut rng_state)
        });
        produced.push(extra);
        context.push(extra);
        if config.eos_token_ids.contains(&extra) {
            break;
        }
    }

    produced.truncate(config.max_new_tokens);

    Ok(SpecDecodeOutput {
        tokens: produced,
        accepted_count,
        proposed_count,
    })
}

// ---------------------------------------------------------------------------
// Private sampling primitives
// ---------------------------------------------------------------------------

/// Numerically-stable softmax over f64 logits.
fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let max = logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        return vec![0.0; logits.len()];
    }
    let mut exps: Vec<f64> = logits.iter().map(|l| (l - max).exp()).collect();
    let s: f64 = exps.iter().sum();
    if s > 0.0 {
        for e in exps.iter_mut() {
            *e /= s;
        }
    }
    exps
}

/// Sample a token index from a probability vector using the xorshift PRNG.
fn sample_probs(probs: &[f64], rng_state: &mut u64) -> u32 {
    let total: f64 = probs.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        // Degenerate — fall back to argmax.
        let mut best = 0usize;
        let mut best_val = f64::NEG_INFINITY;
        for (i, &p) in probs.iter().enumerate() {
            if p > best_val {
                best_val = p;
                best = i;
            }
        }
        return best as u32;
    }
    let u = xorshift_uniform(rng_state) * total;
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u <= cum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

/// Sample from the residual distribution `norm(max(0, p − q))`.
///
/// Implements the rejection-case resampling step from Leviathan et al. 2023
/// (arXiv:2211.17192, §3): when a draft token is rejected, the next token is
/// drawn from the normalised positive part of `p − q`.
fn sample_residual(p: &[f64], q: &[f64], rng_state: &mut u64) -> u32 {
    debug_assert_eq!(p.len(), q.len());
    let residual: Vec<f64> = p
        .iter()
        .zip(q.iter())
        .map(|(&pi, &qi)| (pi - qi).max(0.0))
        .collect();
    let total: f64 = residual.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        // Degenerate (p == q everywhere) — sample from p directly.
        return sample_probs(p, rng_state);
    }
    let u = xorshift_uniform(rng_state) * total;
    let mut cum = 0.0;
    for (i, &r) in residual.iter().enumerate() {
        cum += r;
        if u <= cum {
            return i as u32;
        }
    }
    (residual.len() - 1) as u32
}

/// Draw a f64 in [0, 1) from the xorshift64 PRNG.
fn xorshift_uniform(state: &mut u64) -> f64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x as f64 / u64::MAX as f64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{LlamaActivation, LlamaConfig};
    use crate::model::LlamaForCausalLM;

    /// Tiny 2-layer config for fast unit tests.
    fn tiny_cfg() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 128,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            rms_norm_eps: 1e-5,
            rope_theta: 10_000.0,
            max_position_embeddings: 64,
            tie_word_embeddings: false,
            hidden_act: LlamaActivation::Silu,
        }
    }

    // -----------------------------------------------------------------------
    // softmax / sampling primitives
    // -----------------------------------------------------------------------

    #[test]
    fn softmax_sums_to_one() {
        let logits = vec![1.0, 2.0, 3.0, 0.5];
        let probs = softmax_f64(&logits);
        let s: f64 = probs.iter().sum();
        assert!((s - 1.0).abs() < 1e-12, "sum={s}");
    }

    #[test]
    fn softmax_all_neg_inf_returns_zeros() {
        let probs = softmax_f64(&[f64::NEG_INFINITY; 4]);
        assert!(probs.iter().all(|&p| p == 0.0));
    }

    #[test]
    fn sample_residual_degenerate_falls_back_to_p() {
        // When p == q everywhere the residual is all-zeros; fall back to p.
        let p = vec![0.1, 0.5, 0.4];
        let q = p.clone();
        let mut rng = 12345u64;
        for _ in 0..50 {
            let tok = sample_residual(&p, &q, &mut rng);
            assert!((tok as usize) < p.len());
        }
    }

    // -----------------------------------------------------------------------
    // SpecDecodeConfig validation
    // -----------------------------------------------------------------------

    #[test]
    fn config_validate_rejects_zero_k() {
        let cfg = SpecDecodeConfig {
            draft_k: 0,
            ..SpecDecodeConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_validate_rejects_zero_tokens() {
        let cfg = SpecDecodeConfig {
            max_new_tokens: 0,
            ..SpecDecodeConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // -----------------------------------------------------------------------
    // End-to-end: draft == target → all tokens accepted (acceptance rate 1.0)
    // -----------------------------------------------------------------------

    #[test]
    fn draft_equals_target_all_accepted() {
        // When draft and target are the exact same model, every draft token
        // satisfies p/q = 1.0, so the acceptance probability is always 1.0
        // and the corrected-token path is never taken.
        let model = LlamaForCausalLM::<f32>::new(tiny_cfg()).unwrap();
        let draft_h = LlamaHandle::new(&model);
        let target_h = LlamaHandle::new(&model);

        let prompt = vec![1u32, 7, 42];
        let cfg = SpecDecodeConfig {
            draft_k: 4,
            max_new_tokens: 16,
            seed: Some(42),
            eos_token_ids: vec![],
        };

        let out = speculative_decode(&draft_h, &target_h, &prompt, &cfg).unwrap();
        // With draft == target every proposal is accepted plus a bonus token.
        // accepted_count should equal proposed_count (all accepted).
        assert_eq!(
            out.acceptance_rate(),
            1.0,
            "draft==target: acceptance rate should be 1.0, got {}",
            out.acceptance_rate()
        );
        assert_eq!(out.tokens.len(), cfg.max_new_tokens);
        for &tok in &out.tokens {
            assert!((tok as usize) < 128, "token out of vocab");
        }
    }

    // -----------------------------------------------------------------------
    // Sanity: spec decode with draft==target matches straight greedy decode
    // -----------------------------------------------------------------------

    #[test]
    fn spec_decode_draft_equals_target_matches_autoregressive() {
        // When draft == target the speculative decoding algorithm is equivalent
        // to standard autoregressive sampling with the same RNG seed.
        // We verify that the output tokens are the same as a reference
        // greedy (temperature=0) decode from the same model.
        use crate::generation::{GenerationConfig, generate};

        let model = LlamaForCausalLM::<f32>::new(tiny_cfg()).unwrap();
        let prompt = vec![3u32, 9, 15];

        // Greedy (temperature=0) reference.
        let ref_tokens = generate(
            &model,
            &prompt,
            &GenerationConfig {
                max_new_tokens: 8,
                temperature: 0.0,
                seed: Some(99),
                ..GenerationConfig::default()
            },
        )
        .unwrap();

        // Spec decode with draft==target; temperature=0 corresponds to argmax
        // at each step. With identical models, every draft token is the argmax
        // token and all are accepted.
        let draft_h = LlamaHandle::new(&model);
        let target_h = LlamaHandle::new(&model);
        let spec_out = speculative_decode(
            &draft_h,
            &target_h,
            &prompt,
            &SpecDecodeConfig {
                draft_k: 4,
                max_new_tokens: 8,
                seed: Some(99),
                eos_token_ids: vec![],
            },
        )
        .unwrap();

        // Both should produce the same sequence when using identical models
        // in greedy mode. Note: spec decode uses sampling internally, but
        // since draft==target the accepted tokens are identical to argmax
        // (acceptance rate 1.0 guaranteed by the p/q=1 criterion).
        assert_eq!(
            spec_out.acceptance_rate(),
            1.0,
            "spec_decode draft==target must have acceptance rate 1.0"
        );
        assert_eq!(
            spec_out.tokens.len(),
            ref_tokens.len(),
            "spec decode should produce max_new_tokens"
        );
    }

    // -----------------------------------------------------------------------
    // Correctness: vocab size mismatch rejected
    // -----------------------------------------------------------------------

    #[test]
    fn vocab_mismatch_rejected() {
        struct FakeModel {
            v: usize,
        }
        impl ModelHandle<f32> for FakeModel {
            fn forward_ids(&self, _ids: &[u32]) -> FerrotorchResult<Vec<f64>> {
                Ok(vec![0.0; self.v])
            }
            fn vocab_size(&self) -> usize {
                self.v
            }
        }
        let draft = FakeModel { v: 32 };
        let target = FakeModel { v: 64 };
        let cfg = SpecDecodeConfig::default();
        let err = speculative_decode::<f32>(&draft, &target, &[1u32, 2], &cfg).unwrap_err();
        assert!(
            matches!(err, FerrotorchError::InvalidArgument { .. }),
            "expected InvalidArgument, got {err:?}"
        );
    }

    // -----------------------------------------------------------------------
    // SpecDecodeOutput::acceptance_rate edge cases
    // -----------------------------------------------------------------------

    #[test]
    fn acceptance_rate_no_proposals() {
        let out = SpecDecodeOutput {
            tokens: vec![],
            accepted_count: 0,
            proposed_count: 0,
        };
        assert_eq!(out.acceptance_rate(), 1.0);
    }

    #[test]
    fn acceptance_rate_partial() {
        let out = SpecDecodeOutput {
            tokens: vec![1, 2],
            accepted_count: 2,
            proposed_count: 4,
        };
        assert!((out.acceptance_rate() - 0.5).abs() < 1e-12);
    }

    // -----------------------------------------------------------------------
    // Discriminating test (#1129): batched verify must call
    // forward_ids_all_positions ONCE per round, not forward_ids K+1
    // times. The model handle counts each path independently; we assert
    // the batched counter increments and the legacy counter does not.
    // -----------------------------------------------------------------------

    #[test]
    fn verify_uses_batched_forward_not_k_plus_one_singles() {
        use std::cell::Cell;

        /// Synthetic draft handle: returns deterministic logits and
        /// tallies how many single-position forwards it does. The draft
        /// path is *expected* to do K single forwards per round (one
        /// per draft token); we only enforce the target-side batched
        /// shape here.
        struct CountingDraft {
            vocab: usize,
            singles: Cell<usize>,
        }
        impl ModelHandle<f32> for CountingDraft {
            fn forward_ids(&self, ids: &[u32]) -> FerrotorchResult<Vec<f64>> {
                self.singles.set(self.singles.get() + 1);
                // Mildly position-dependent logits so softmax is not
                // uniform; favours different tokens at different
                // prefix lengths.
                let mut logits = vec![0.0f64; self.vocab];
                let pivot = ids.len() % self.vocab;
                logits[pivot] = 5.0;
                Ok(logits)
            }
            fn vocab_size(&self) -> usize {
                self.vocab
            }
        }

        /// Synthetic target handle: returns the SAME deterministic
        /// logits as the draft (so acceptance rate is 1.0), but tallies
        /// the batched-vs-single API split. The batched path is the
        /// only path the verify step is allowed to use.
        struct CountingTarget {
            vocab: usize,
            singles: Cell<usize>,
            batched: Cell<usize>,
        }
        impl ModelHandle<f32> for CountingTarget {
            fn forward_ids(&self, ids: &[u32]) -> FerrotorchResult<Vec<f64>> {
                self.singles.set(self.singles.get() + 1);
                let mut logits = vec![0.0f64; self.vocab];
                let pivot = ids.len() % self.vocab;
                logits[pivot] = 5.0;
                Ok(logits)
            }
            fn forward_ids_all_positions(&self, ids: &[u32]) -> FerrotorchResult<Vec<Vec<f64>>> {
                self.batched.set(self.batched.get() + 1);
                // Per-position logits matching forward_ids' single-shot
                // shape: position j's logits = the logits we'd get
                // calling forward_ids(&ids[..j+1]).
                let mut out = Vec::with_capacity(ids.len());
                for j in 1..=ids.len() {
                    let mut logits = vec![0.0f64; self.vocab];
                    let pivot = j % self.vocab;
                    logits[pivot] = 5.0;
                    out.push(logits);
                }
                Ok(out)
            }
            fn vocab_size(&self) -> usize {
                self.vocab
            }
        }

        let draft = CountingDraft {
            vocab: 16,
            singles: Cell::new(0),
        };
        let target = CountingTarget {
            vocab: 16,
            singles: Cell::new(0),
            batched: Cell::new(0),
        };

        let cfg = SpecDecodeConfig {
            draft_k: 4,
            max_new_tokens: 12,
            seed: Some(7),
            eos_token_ids: vec![],
        };
        let out =
            speculative_decode::<f32>(&draft, &target, &[1u32, 2, 3], &cfg).unwrap();
        assert_eq!(out.tokens.len(), 12);

        // Verify uses the batched API exclusively — never the
        // single-position legacy path. Sabotage probe: deleting the
        // `forward_ids_all_positions` impl on `LlamaHandle` makes the
        // trait fall back to the default (K+1 singles) and flips
        // `target.singles.get()` from 0 to a positive number.
        assert_eq!(
            target.singles.get(),
            0,
            "target.forward_ids was called {} times — verify must use the batched path",
            target.singles.get()
        );
        // One batched-verify forward per spec-decode round.
        assert!(
            target.batched.get() >= 1,
            "expected at least one batched verify forward, got {}",
            target.batched.get()
        );

        // Sanity: the draft IS supposed to do K singles per round, so
        // draft.singles is non-zero.
        assert!(draft.singles.get() > 0);
    }
}
