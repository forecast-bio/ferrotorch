//! Token-level generation for `LlamaForCausalLM`. (#592)
//!
//! Provides greedy / temperature / top-k / top-p (nucleus) sampling on top
//! of the existing `LlamaForCausalLM::forward_from_ids` autoregressive
//! step. Beam search and speculative decoding are deferred — see #592 for
//! the design notes.
//!
//! # Streaming
//!
//! [`generate_with_streamer`] takes a `&mut dyn FnMut(u32) -> bool` that
//! is called for every new token. Return `false` to stop early (e.g. to
//! react to a sentinel token outside the EOS list).

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};

use crate::model::LlamaForCausalLM;

/// All knobs that control text generation.
#[derive(Debug, Clone)]
pub struct GenerationConfig {
    /// Hard cap on output length (excluding prompt). Generation always
    /// stops at this many *new* tokens regardless of EOS.
    pub max_new_tokens: usize,

    /// `0.0` triggers the greedy path; `> 0.0` enables stochastic
    /// sampling. Logits are divided by `temperature` before softmax.
    pub temperature: f64,

    /// Keep only the top-k logits. `0` disables this filter.
    pub top_k: usize,

    /// Nucleus sampling: keep the smallest set of top-prob tokens whose
    /// cumulative probability is at least `top_p`. `1.0` disables this
    /// filter (passes all tokens through).
    pub top_p: f64,

    /// Repetition penalty (Keskar et al. 2019). Values > 1 down-weight
    /// already-emitted tokens; `1.0` disables.
    pub repetition_penalty: f64,

    /// Tokens that terminate generation early. Llama 3 typically uses
    /// `[128001, 128009]` (`<|end_of_text|>`, `<|eot_id|>`).
    pub eos_token_ids: Vec<u32>,

    /// Seed for the xorshift PRNG. `None` uses an env-based seed for
    /// reproducibility; the same seed always produces the same output
    /// for the same prompt and config.
    pub seed: Option<u64>,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            repetition_penalty: 1.0,
            eos_token_ids: Vec::new(),
            seed: None,
        }
    }
}

impl GenerationConfig {
    /// Greedy decoding (always pick the argmax).
    pub fn greedy(max_new_tokens: usize) -> Self {
        Self {
            max_new_tokens,
            temperature: 0.0,
            ..Self::default()
        }
    }

    /// Temperature-only sampling.
    pub fn sampling(max_new_tokens: usize, temperature: f64) -> Self {
        Self {
            max_new_tokens,
            temperature,
            ..Self::default()
        }
    }

    /// Standard "creative" preset: temperature 0.8, top-p 0.95, no top-k.
    pub fn nucleus(max_new_tokens: usize, top_p: f64, temperature: f64) -> Self {
        Self {
            max_new_tokens,
            temperature,
            top_p,
            ..Self::default()
        }
    }
}

/// Generate up to `config.max_new_tokens` new tokens after `prompt_ids`.
/// Returns the generated tokens only (the prompt is not included).
pub fn generate<T: Float>(
    model: &LlamaForCausalLM<T>,
    prompt_ids: &[u32],
    config: &GenerationConfig,
) -> FerrotorchResult<Vec<u32>> {
    generate_with_streamer(model, prompt_ids, config, &mut |_| true)
}

/// `generate` with a streaming callback. The callback is invoked once
/// per new token, in order. Return `false` to stop early.
pub fn generate_with_streamer<T: Float>(
    model: &LlamaForCausalLM<T>,
    prompt_ids: &[u32],
    config: &GenerationConfig,
    streamer: &mut dyn FnMut(u32) -> bool,
) -> FerrotorchResult<Vec<u32>> {
    if prompt_ids.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "generate: prompt_ids must not be empty".into(),
        });
    }
    if config.temperature < 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "generate: temperature must be >= 0, got {}",
                config.temperature
            ),
        });
    }
    if !(0.0..=1.0).contains(&config.top_p) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("generate: top_p must be in [0, 1], got {}", config.top_p),
        });
    }
    if config.repetition_penalty <= 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "generate: repetition_penalty must be > 0, got {}",
                config.repetition_penalty
            ),
        });
    }

    let mut ids = prompt_ids.to_vec();
    let mut produced: Vec<u32> = Vec::with_capacity(config.max_new_tokens);
    let mut rng_state = config.seed.unwrap_or(0xdead_beef_cafe_babe);

    for _ in 0..config.max_new_tokens {
        let logits_tensor = model.forward_from_ids(&ids)?;
        // logits_tensor: [1, seq_len, vocab_size] → take the last position.
        let shape = logits_tensor.shape();
        if shape.len() != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "generate: expected logits shape [1, S, V], got {shape:?}"
                ),
            });
        }
        let seq_len = shape[1];
        let vocab = shape[2];
        let data = logits_tensor.data_vec()?;
        let last_offset = (seq_len - 1) * vocab;
        let mut logits: Vec<f64> = data[last_offset..last_offset + vocab]
            .iter()
            .map(|v| v.to_f64().unwrap())
            .collect();

        if (config.repetition_penalty - 1.0).abs() > f64::EPSILON {
            apply_repetition_penalty(&mut logits, &ids, config.repetition_penalty);
        }

        let next = if config.temperature == 0.0 {
            argmax(&logits)
        } else {
            apply_temperature(&mut logits, config.temperature);
            if config.top_k > 0 && config.top_k < vocab {
                top_k_filter(&mut logits, config.top_k);
            }
            if config.top_p < 1.0 {
                top_p_filter(&mut logits, config.top_p);
            }
            sample_softmax(&logits, &mut rng_state)
        };

        produced.push(next);
        if !streamer(next) {
            break;
        }
        if config.eos_token_ids.contains(&next) {
            break;
        }
        ids.push(next);
    }

    Ok(produced)
}

// ---------------------------------------------------------------------------
// Sampling primitives — exposed publicly so callers can roll their own
// generation loops on top of `model.forward_from_ids`.
// ---------------------------------------------------------------------------

/// In-place divide every logit by `temperature`. `temperature == 0.0`
/// is a contract violation here (the caller picks argmax instead) and
/// is checked at the public entry point.
pub fn apply_temperature(logits: &mut [f64], temperature: f64) {
    let inv = 1.0 / temperature;
    for l in logits.iter_mut() {
        *l *= inv;
    }
}

/// Set every logit outside the top-k to `-inf`. `k == 0` is a no-op.
pub fn top_k_filter(logits: &mut [f64], k: usize) {
    if k == 0 || k >= logits.len() {
        return;
    }
    // Find the k-th largest threshold by partial sort of indices.
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let cutoff = logits[idx[k - 1]];
    for v in logits.iter_mut() {
        if *v < cutoff {
            *v = f64::NEG_INFINITY;
        }
    }
}

/// Nucleus filter: zero out the smallest-prob tail until the kept mass
/// is at least `top_p`. Operates on logits (in-place); the kept set
/// is whatever's needed to reach cumulative `top_p` after softmaxing.
pub fn top_p_filter(logits: &mut [f64], top_p: f64) {
    if top_p >= 1.0 {
        return;
    }
    let probs = softmax_f64(logits);
    let mut idx: Vec<usize> = (0..probs.len()).collect();
    idx.sort_by(|&a, &b| {
        probs[b]
            .partial_cmp(&probs[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut cum = 0.0;
    let mut keep = vec![false; probs.len()];
    for &i in &idx {
        keep[i] = true;
        cum += probs[i];
        if cum >= top_p {
            break;
        }
    }
    for (l, k) in logits.iter_mut().zip(keep.iter()) {
        if !*k {
            *l = f64::NEG_INFINITY;
        }
    }
}

/// In-place repetition penalty (Keskar et al. 2019). For each previously
/// generated token, divide its logit by `penalty` if positive, multiply
/// otherwise. `> 1.0` discourages repeats.
pub fn apply_repetition_penalty(logits: &mut [f64], context: &[u32], penalty: f64) {
    let vocab = logits.len();
    for &tok in context {
        let i = tok as usize;
        if i >= vocab {
            continue;
        }
        let v = logits[i];
        logits[i] = if v > 0.0 { v / penalty } else { v * penalty };
    }
}

/// Argmax — used by greedy decoding.
pub fn argmax(logits: &[f64]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f64::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}

/// Softmax → categorical sample using the given xorshift state.
pub fn sample_softmax(logits: &[f64], rng_state: &mut u64) -> u32 {
    let probs = softmax_f64(logits);
    let total: f64 = probs.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        // All -inf (every token filtered) → fall back to argmax.
        return argmax(logits);
    }
    let u = (xorshift_next(rng_state) as f64 / u64::MAX as f64) * total;
    let mut cum = 0.0;
    for (i, &p) in probs.iter().enumerate() {
        cum += p;
        if u <= cum {
            return i as u32;
        }
    }
    (probs.len() - 1) as u32
}

// ===========================================================================
// Beam search (#612)
// ===========================================================================

/// Configuration for [`beam_search`]. Mirrors the relevant subset of HF
/// transformers `GenerationConfig` for beam search.
#[derive(Debug, Clone)]
pub struct BeamSearchConfig {
    /// Number of beams kept at each step. Must be > 0.
    pub num_beams: usize,
    /// Maximum number of new tokens to generate per beam.
    pub max_new_tokens: usize,
    /// Length-penalty exponent. The score of a finished beam of length L
    /// is `cum_log_prob / L^length_penalty`. Use `1.0` to bias towards
    /// shorter completions, `>1.0` for longer. Default: `1.0`.
    pub length_penalty: f64,
    /// EOS token IDs — beams that produce one of these tokens finalize
    /// and stop competing for slots.
    pub eos_token_ids: Vec<u32>,
}

impl Default for BeamSearchConfig {
    fn default() -> Self {
        Self {
            num_beams: 4,
            max_new_tokens: 32,
            length_penalty: 1.0,
            eos_token_ids: Vec::new(),
        }
    }
}

/// Beam search decoding. Maintains `num_beams` running candidates ranked
/// by cumulative log-probability; at each step expands every live beam to
/// every vocab continuation and keeps the top `num_beams` total.
///
/// Returns the finalised beams (continuation tokens only — prompt is not
/// included) sorted by length-normalised score, best first. The returned
/// `Vec` always has `num_beams` entries; if generation fails to produce
/// `num_beams` finalised candidates within `max_new_tokens`, the live
/// beams are returned as-is.
///
/// # Cost
///
/// Memory and compute are `num_beams ×` the corresponding cost of greedy
/// decoding. Each step we run `num_beams` forward passes (one per beam)
/// and then a top-`num_beams × vocab` argmax. (#612)
pub fn beam_search<T: Float>(
    model: &LlamaForCausalLM<T>,
    prompt_ids: &[u32],
    config: &BeamSearchConfig,
) -> FerrotorchResult<Vec<Vec<u32>>> {
    if prompt_ids.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "beam_search: prompt_ids must not be empty".into(),
        });
    }
    if config.num_beams == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "beam_search: num_beams must be > 0".into(),
        });
    }
    if config.length_penalty <= 0.0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "beam_search: length_penalty must be > 0, got {}",
                config.length_penalty
            ),
        });
    }

    /// One live beam: the produced continuation + cumulative log-prob.
    #[derive(Clone, Debug)]
    struct Beam {
        produced: Vec<u32>,
        score: f64,
        finished: bool,
    }

    let mut live: Vec<Beam> = vec![Beam {
        produced: Vec::new(),
        score: 0.0,
        finished: false,
    }];

    for _step in 0..config.max_new_tokens {
        // Stop early when every beam is finished.
        if live.iter().all(|b| b.finished) {
            break;
        }

        // Expand each non-finished beam to its top continuations. We collect
        // (beam_idx, token, new_score) triples then keep the top `num_beams`.
        let mut candidates: Vec<(usize, u32, f64)> =
            Vec::with_capacity(live.len() * config.num_beams);

        for (bi, beam) in live.iter().enumerate() {
            if beam.finished {
                // Finished beams pass through unchanged with the same score.
                candidates.push((bi, u32::MAX, beam.score));
                continue;
            }
            // Forward pass over prompt + this beam's continuation.
            let mut ids = prompt_ids.to_vec();
            ids.extend_from_slice(&beam.produced);
            let logits_tensor = model.forward_from_ids(&ids)?;
            let shape = logits_tensor.shape();
            if shape.len() != 3 {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "beam_search: expected logits shape [1, S, V], got {shape:?}"
                    ),
                });
            }
            let seq_len = shape[1];
            let vocab = shape[2];
            let data = logits_tensor.data_vec()?;
            let last_offset = (seq_len - 1) * vocab;
            let logits: Vec<f64> = data[last_offset..last_offset + vocab]
                .iter()
                .map(|v| v.to_f64().unwrap())
                .collect();

            // Softmax → log-prob in the numerically-stable way.
            let max_logit = logits
                .iter()
                .copied()
                .fold(f64::NEG_INFINITY, f64::max);
            let log_sum_exp = max_logit
                + logits
                    .iter()
                    .map(|&l| (l - max_logit).exp())
                    .sum::<f64>()
                    .ln();

            // Pick top `num_beams` continuations from this beam by partial sort.
            // Build (token, log_prob) pairs and partial-select the top-k.
            let mut log_probs: Vec<(u32, f64)> = (0..vocab as u32)
                .map(|i| (i, logits[i as usize] - log_sum_exp))
                .collect();
            let pivot = config.num_beams.min(log_probs.len().saturating_sub(1));
            log_probs.select_nth_unstable_by(pivot, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            for &(tok, lp) in log_probs.iter().take(config.num_beams) {
                candidates.push((bi, tok, beam.score + lp));
            }
        }

        // Take global top `num_beams` by score.
        candidates.sort_unstable_by(|a, b| {
            b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.truncate(config.num_beams);

        // Build the next round of beams.
        let mut next_live: Vec<Beam> = Vec::with_capacity(candidates.len());
        for (bi, tok, score) in candidates {
            if tok == u32::MAX {
                // Pass-through finished beam.
                next_live.push(live[bi].clone());
                continue;
            }
            let mut produced = live[bi].produced.clone();
            produced.push(tok);
            let finished = config.eos_token_ids.contains(&tok);
            next_live.push(Beam {
                produced,
                score,
                finished,
            });
        }
        live = next_live;
    }

    // Sort by length-normalised score, best first.
    let lp = config.length_penalty;
    live.sort_by(|a, b| {
        let la = (a.produced.len() as f64).max(1.0).powf(lp);
        let lb = (b.produced.len() as f64).max(1.0).powf(lp);
        let na = a.score / la;
        let nb = b.score / lb;
        nb.partial_cmp(&na).unwrap_or(std::cmp::Ordering::Equal)
    });

    Ok(live.into_iter().map(|b| b.produced).collect())
}

/// Numerically-stable softmax (private helper).
fn softmax_f64(logits: &[f64]) -> Vec<f64> {
    let max = logits
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if !max.is_finite() {
        // All -inf — degenerate; return zeros so the caller falls back.
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

fn xorshift_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn argmax_picks_highest() {
        assert_eq!(argmax(&[0.1, 0.5, 0.2, 0.9, -1.0]), 3);
    }

    #[test]
    fn temperature_scales_logits() {
        let mut l = vec![1.0, 2.0, 3.0];
        apply_temperature(&mut l, 2.0);
        assert_eq!(l, vec![0.5, 1.0, 1.5]);
    }

    #[test]
    fn top_k_keeps_only_k() {
        let mut l = vec![1.0, 5.0, 2.0, 8.0, 3.0, 7.0];
        top_k_filter(&mut l, 2);
        // Top 2 are 8.0 and 7.0; everything else → -inf.
        assert_eq!(l[3], 8.0);
        assert_eq!(l[5], 7.0);
        for i in [0, 1, 2, 4] {
            assert!(l[i].is_infinite() && l[i] < 0.0);
        }
    }

    #[test]
    fn top_k_zero_is_noop() {
        let mut l = vec![1.0, 5.0, 2.0];
        top_k_filter(&mut l, 0);
        assert_eq!(l, vec![1.0, 5.0, 2.0]);
    }

    #[test]
    fn top_p_keeps_just_enough_mass() {
        // logits → probs uniform at first; with top_p=0.5 we should
        // keep enough top tokens to reach 0.5 cumulative mass.
        let mut l = vec![0.0, 0.0, 0.0, 0.0]; // uniform 0.25 each
        top_p_filter(&mut l, 0.5);
        let kept: usize = l.iter().filter(|v| v.is_finite()).count();
        assert_eq!(kept, 2, "got {l:?}");
    }

    #[test]
    fn top_p_one_is_noop() {
        let mut l = vec![0.0, 1.0, -1.0];
        top_p_filter(&mut l, 1.0);
        assert_eq!(l, vec![0.0, 1.0, -1.0]);
    }

    #[test]
    fn repetition_penalty_downweights_seen_tokens() {
        // logits all 1.0 → after applying penalty 2.0 to context [0, 2],
        // logits[0] = 0.5, logits[2] = 0.5, others stay 1.0.
        let mut l = vec![1.0; 4];
        apply_repetition_penalty(&mut l, &[0, 2], 2.0);
        assert_eq!(l, vec![0.5, 1.0, 0.5, 1.0]);
    }

    #[test]
    fn repetition_penalty_negative_logits() {
        // Negative logits are made *more* negative (i.e., multiplied).
        let mut l = vec![-1.0; 3];
        apply_repetition_penalty(&mut l, &[1], 2.0);
        assert_eq!(l, vec![-1.0, -2.0, -1.0]);
    }

    #[test]
    fn sample_softmax_with_one_finite_logit_picks_it() {
        let l = vec![f64::NEG_INFINITY, 1.0, f64::NEG_INFINITY];
        let mut rng = 12345u64;
        for _ in 0..100 {
            assert_eq!(sample_softmax(&l, &mut rng), 1);
        }
    }

    #[test]
    fn sample_softmax_all_neg_inf_falls_back_to_argmax() {
        // Every logit is -inf → softmax returns zeros → fallback to argmax.
        // argmax of all-equal -inf returns the first index.
        let l = vec![f64::NEG_INFINITY; 4];
        let mut rng = 1u64;
        let r = sample_softmax(&l, &mut rng);
        assert_eq!(r, 0);
    }

    #[test]
    fn sample_softmax_distribution_matches_probs_loosely() {
        // Skewed logits: token 1 should dominate.
        let l = vec![0.0, 5.0, 0.0]; // softmax ≈ [0.0066, 0.987, 0.0066]
        let mut rng = 0xdead_beef_u64;
        let mut counts = [0u32; 3];
        for _ in 0..1000 {
            counts[sample_softmax(&l, &mut rng) as usize] += 1;
        }
        // Token 1 should win > 90% of the draws.
        assert!(counts[1] > 900, "got counts {counts:?}");
    }

    #[test]
    fn generation_config_helpers() {
        let g = GenerationConfig::greedy(10);
        assert_eq!(g.temperature, 0.0);
        assert_eq!(g.max_new_tokens, 10);

        let s = GenerationConfig::sampling(20, 0.7);
        assert_eq!(s.temperature, 0.7);

        let n = GenerationConfig::nucleus(30, 0.9, 0.8);
        assert_eq!(n.top_p, 0.9);
        assert_eq!(n.temperature, 0.8);
    }

    // -----------------------------------------------------------------
    // Beam search (#612)
    // -----------------------------------------------------------------

    #[test]
    fn beam_search_config_defaults_sensible() {
        let c = BeamSearchConfig::default();
        assert_eq!(c.num_beams, 4);
        assert_eq!(c.max_new_tokens, 32);
        assert!((c.length_penalty - 1.0).abs() < 1e-12);
        assert!(c.eos_token_ids.is_empty());
    }

    #[test]
    fn beam_search_validates_inputs() {
        // Build a model only enough to hit the API surface; the validation
        // happens before any forward pass.
        use crate::config::LlamaConfig;
        // Tiny config so module construction is cheap.
        let cfg = LlamaConfig {
            vocab_size: 4,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 8,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            hidden_act: crate::config::LlamaActivation::Silu,
        };
        let model = LlamaForCausalLM::<f32>::new(cfg).unwrap();

        // Empty prompt rejected.
        let err =
            beam_search(&model, &[], &BeamSearchConfig::default()).unwrap_err();
        assert!(matches!(
            err,
            ferrotorch_core::FerrotorchError::InvalidArgument { .. }
        ));

        // num_beams = 0 rejected.
        let err = beam_search(
            &model,
            &[1, 2],
            &BeamSearchConfig {
                num_beams: 0,
                ..BeamSearchConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ferrotorch_core::FerrotorchError::InvalidArgument { .. }
        ));

        // length_penalty 0 rejected.
        let err = beam_search(
            &model,
            &[1, 2],
            &BeamSearchConfig {
                length_penalty: 0.0,
                ..BeamSearchConfig::default()
            },
        )
        .unwrap_err();
        assert!(matches!(
            err,
            ferrotorch_core::FerrotorchError::InvalidArgument { .. }
        ));
    }

    #[test]
    fn beam_search_returns_num_beams_results() {
        // Tiny model + short max_new_tokens — just verify the shape contract.
        use crate::config::LlamaConfig;
        let cfg = LlamaConfig {
            vocab_size: 8,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            hidden_act: crate::config::LlamaActivation::Silu,
        };
        let model = LlamaForCausalLM::<f32>::new(cfg).unwrap();
        let beams = beam_search(
            &model,
            &[1, 2, 3],
            &BeamSearchConfig {
                num_beams: 3,
                max_new_tokens: 2,
                length_penalty: 1.0,
                eos_token_ids: vec![],
            },
        )
        .unwrap();
        assert_eq!(beams.len(), 3);
        for beam in &beams {
            assert_eq!(beam.len(), 2, "each beam has max_new_tokens entries");
            for &tok in beam {
                assert!(tok < cfg.vocab_size as u32);
            }
        }
    }

    #[test]
    fn beam_search_eos_finalizes_beam() {
        // If we set EVERY token as an EOS, every beam stops after its first
        // token.
        use crate::config::LlamaConfig;
        let cfg = LlamaConfig {
            vocab_size: 4,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            tie_word_embeddings: false,
            hidden_act: crate::config::LlamaActivation::Silu,
        };
        let model = LlamaForCausalLM::<f32>::new(cfg).unwrap();
        let eos: Vec<u32> = (0..4).collect();
        let beams = beam_search(
            &model,
            &[1, 2],
            &BeamSearchConfig {
                num_beams: 2,
                max_new_tokens: 5,
                length_penalty: 1.0,
                eos_token_ids: eos,
            },
        )
        .unwrap();
        assert_eq!(beams.len(), 2);
        // Each beam should be exactly 1 token long because the first token
        // is always an EOS.
        for beam in &beams {
            assert_eq!(beam.len(), 1);
        }
    }
}
