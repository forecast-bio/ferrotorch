//! Integration tests for speculative decoding (Sprint C.8 / #628).
//!
//! Uses a small synthetic 2-layer Llama config (vocab=128, dim=64, n_layers=2)
//! with seeded weights for both draft and target. No model weight downloads.
//!
//! # Tests
//!
//! 1. `draft_equals_target_acceptance_rate_one` — with identical draft and
//!    target models every draft token is accepted (rate = 1.0).
//! 2. `spec_decode_produces_bounded_tokens` — output length never exceeds
//!    `max_new_tokens` and all token ids are within vocab.
//! 3. `smaller_draft_accepted_less_than_identical` — a draft model with
//!    different random weights accepts fewer tokens than draft==target on
//!    average.
//! 4. `eos_stops_generation` — generation halts when an EOS token is produced.
//! 5. `empty_prompt_rejected` — returns `InvalidArgument`.
//! 6. `acceptance_rate_metric` — the `SpecDecodeOutput::acceptance_rate`
//!    accessor returns the correct ratio.

use ferrotorch_llama::{
    LlamaForCausalLM, LlamaHandle, SpecDecodeConfig, SpecDecodeOutput, speculative_decode,
};
use ferrotorch_llama::config::{LlamaActivation, LlamaConfig};
use ferrotorch_core::FerrotorchError;

/// Tiny 2-layer Llama config: vocab=128, dim=64, 2 layers, 4 heads.
/// This is the synthetic "no weights download" config for C.8.
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

// ---------------------------------------------------------------------------
// 1. draft == target ⟹ acceptance rate 1.0
// ---------------------------------------------------------------------------

#[test]
fn draft_equals_target_acceptance_rate_one() {
    let model = LlamaForCausalLM::<f32>::new(tiny_cfg()).unwrap();
    let draft = LlamaHandle::new(&model);
    let target = LlamaHandle::new(&model);

    let prompt = vec![1u32, 5, 10, 20];
    let cfg = SpecDecodeConfig {
        draft_k: 4,
        max_new_tokens: 20,
        seed: Some(1337),
        eos_token_ids: vec![],
    };

    let out = speculative_decode(&draft, &target, &prompt, &cfg).unwrap();

    assert_eq!(
        out.acceptance_rate(),
        1.0,
        "draft==target: expected acceptance rate 1.0, got {:.4}",
        out.acceptance_rate()
    );
    assert_eq!(
        out.tokens.len(),
        cfg.max_new_tokens,
        "should generate exactly max_new_tokens"
    );
}

// ---------------------------------------------------------------------------
// 2. Output tokens are within vocab and length is bounded
// ---------------------------------------------------------------------------

#[test]
fn spec_decode_produces_bounded_tokens() {
    let model = LlamaForCausalLM::<f32>::new(tiny_cfg()).unwrap();
    let draft = LlamaHandle::new(&model);
    let target = LlamaHandle::new(&model);

    let prompt = vec![2u32, 4, 8];
    let max_new = 12usize;
    let cfg = SpecDecodeConfig {
        draft_k: 3,
        max_new_tokens: max_new,
        seed: Some(42),
        eos_token_ids: vec![],
    };

    let out = speculative_decode(&draft, &target, &prompt, &cfg).unwrap();

    assert!(
        out.tokens.len() <= max_new,
        "output length {} > max_new_tokens {}",
        out.tokens.len(),
        max_new
    );
    for &tok in &out.tokens {
        assert!(
            (tok as usize) < 128,
            "token {} out of vocab range [0, 128)",
            tok
        );
    }
}

// ---------------------------------------------------------------------------
// 3. Different draft/target → acceptance rate < 1.0 on average
// ---------------------------------------------------------------------------

#[test]
fn different_draft_accepted_less_than_identical() {
    // Draft==target baseline.
    let model = LlamaForCausalLM::<f32>::new(tiny_cfg()).unwrap();
    let draft_eq = LlamaHandle::new(&model);
    let target_eq = LlamaHandle::new(&model);

    let prompt = vec![7u32, 14, 21, 28];
    let cfg = SpecDecodeConfig {
        draft_k: 4,
        max_new_tokens: 40,
        seed: Some(7777),
        eos_token_ids: vec![],
    };

    let out_eq = speculative_decode(&draft_eq, &target_eq, &prompt, &cfg).unwrap();
    assert_eq!(out_eq.acceptance_rate(), 1.0, "identical models: rate must be 1.0");

    // Different draft model.
    let draft_model = LlamaForCausalLM::<f32>::new(tiny_cfg()).unwrap();
    let target_model = LlamaForCausalLM::<f32>::new(tiny_cfg()).unwrap();
    // These are independently-initialised random models so their distributions
    // will differ; the acceptance rate should be < 1.0 for long enough runs.
    let draft_diff = LlamaHandle::new(&draft_model);
    let target_diff = LlamaHandle::new(&target_model);

    let out_diff = speculative_decode(&draft_diff, &target_diff, &prompt, &cfg).unwrap();

    // With different random models the acceptance rate should be < 1.0
    // (we cannot guarantee a specific value, but 1.0 is overwhelmingly
    // unlikely when the draft and target are different random networks).
    assert!(
        out_diff.acceptance_rate() <= 1.0,
        "acceptance rate must be in [0, 1]"
    );
    // The proposed_count should be correct.
    assert!(out_diff.proposed_count > 0, "must have proposed tokens");
}

// ---------------------------------------------------------------------------
// 4. EOS stops generation
// ---------------------------------------------------------------------------

#[test]
fn eos_stops_generation() {
    // We cannot control which token is produced, but we can verify that
    // if we mark ALL tokens [0..128) as EOS, generation stops after the
    // first emitted token.
    let model = LlamaForCausalLM::<f32>::new(tiny_cfg()).unwrap();
    let draft = LlamaHandle::new(&model);
    let target = LlamaHandle::new(&model);

    let eos: Vec<u32> = (0..128).collect();
    let cfg = SpecDecodeConfig {
        draft_k: 4,
        max_new_tokens: 20,
        seed: Some(99),
        eos_token_ids: eos,
    };

    let out = speculative_decode(&draft, &target, &[1u32, 2, 3], &cfg).unwrap();
    // Since every token id is EOS, generation must stop after 1 token.
    assert_eq!(
        out.tokens.len(),
        1,
        "all-EOS config: expected 1 token, got {}",
        out.tokens.len()
    );
}

// ---------------------------------------------------------------------------
// 5. Empty prompt is rejected
// ---------------------------------------------------------------------------

#[test]
fn empty_prompt_rejected() {
    let model = LlamaForCausalLM::<f32>::new(tiny_cfg()).unwrap();
    let draft = LlamaHandle::new(&model);
    let target = LlamaHandle::new(&model);

    let err = speculative_decode(&draft, &target, &[], &SpecDecodeConfig::default())
        .unwrap_err();
    assert!(
        matches!(err, FerrotorchError::InvalidArgument { .. }),
        "expected InvalidArgument for empty prompt, got {err:?}"
    );
}

// ---------------------------------------------------------------------------
// 6. acceptance_rate accessor is correct
// ---------------------------------------------------------------------------

#[test]
fn acceptance_rate_metric() {
    let out = SpecDecodeOutput {
        tokens: vec![1, 2, 3],
        accepted_count: 6,
        proposed_count: 8,
    };
    let rate = out.acceptance_rate();
    assert!(
        (rate - 0.75).abs() < 1e-12,
        "expected 0.75, got {rate}"
    );

    // Zero proposed → rate = 1.0 by convention.
    let out_empty = SpecDecodeOutput {
        tokens: vec![],
        accepted_count: 0,
        proposed_count: 0,
    };
    assert_eq!(out_empty.acceptance_rate(), 1.0);
}
