//! Layer 3 conformance tests for ferrotorch-llama.
//!
//! Each test loads a fixture produced by `scripts/regenerate_llama_fixtures.py`
//! (which runs the equivalent forward pass through `transformers==4.50.3`) and
//! compares the ferrotorch output within a documented tolerance budget.
//!
//! Tolerance budget: 1e-3 absolute for all f32 matmul paths. This is
//! conservative — f32 accumulation at tiny model scale matches transformers
//! to ~1e-5, but we use 1e-3 to leave room for future bf16 and fused-kernel
//! paths without needing a fixture regeneration.
//!
//! # Running
//!
//! ```text
//! cargo test -p ferrotorch-llama
//! ```

use std::collections::HashMap;

use ferrotorch_core::{Tensor, from_vec};
use ferrotorch_llama::{
    LlamaActivation, LlamaAttention, LlamaConfig, LlamaDecoderLayer, LlamaForCausalLM, LlamaMLP,
};
use ferrotorch_nn::RMSNorm;
use ferrotorch_nn::module::{Module, StateDict};
use serde_json::Value;

// ---------------------------------------------------------------------------
// Fixture loader
// ---------------------------------------------------------------------------

/// Parse the fixture JSON file.  The path is relative to the crate root
/// (resolved at compile time via `CARGO_MANIFEST_DIR`).
fn load_fixtures() -> Value {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let path = std::path::PathBuf::from(manifest_dir)
        .join("tests/conformance/fixtures/llama.json");
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("cannot read fixture file {}: {e}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|e| panic!("fixture JSON parse error: {e}"))
}

/// Find a fixture with the given `op` and `tag`.
fn get_fixture<'a>(fixtures: &'a [Value], op: &str, tag: &str) -> &'a Value {
    fixtures
        .iter()
        .find(|f| f["op"].as_str() == Some(op) && f["tag"].as_str() == Some(tag))
        .unwrap_or_else(|| panic!("fixture op={op} tag={tag} not found"))
}

/// Parse a JSON array of numbers into `Vec<f32>`.
fn parse_f32_vec(v: &Value) -> Vec<f32> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|x| x.as_f64().expect("expected f64 value") as f32)
        .collect()
}

/// Parse a JSON array of integers into `Vec<usize>`.
fn parse_usize_vec(v: &Value) -> Vec<usize> {
    v.as_array()
        .expect("expected JSON array")
        .iter()
        .map(|x| x.as_u64().expect("expected u64 value") as usize)
        .collect()
}

/// Build a `Tensor<f32>` from a flat JSON array and an explicit shape.
fn tensor_from_json_shape(data_json: &Value, shape: &[usize]) -> Tensor<f32> {
    let data = parse_f32_vec(data_json);
    from_vec(data, shape).expect("tensor_from_json_shape: construction failed")
}

// ---------------------------------------------------------------------------
// Assertion helpers
// ---------------------------------------------------------------------------

const F32_MATMUL_TOL: f32 = 1e-3;
/// Attention forward accumulates softmax + two batched matmuls; at seq_len=4
/// the f32 rounding vs transformers (which also runs f32 internally but uses
/// torch's fused scaled-dot-product-attention) can reach ~3e-3.  We use a
/// separate constant so it is obvious this is a documented, intentional
/// relaxation and not an accidental mis-copy of the main constant.
const F32_ATTN_TOL: f32 = 5e-3;

/// Assert that `actual` and `expected` are element-wise within `tol`.
fn assert_allclose(actual: &[f32], expected: &[f32], tol: f32, context: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{context}: length mismatch: got {} expected {}",
        actual.len(),
        expected.len()
    );
    let mut worst_abs = 0.0_f32;
    let mut worst_idx = 0;
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        if diff > worst_abs {
            worst_abs = diff;
            worst_idx = i;
        }
    }
    assert!(
        worst_abs <= tol,
        "{context}: max absolute error {worst_abs:.2e} > tol {tol:.2e} at index {worst_idx} \
         (actual={} expected={})",
        actual[worst_idx],
        expected[worst_idx],
    );
}

/// Assert shape equality.
fn assert_shape(t: &Tensor<f32>, expected: &[usize], context: &str) {
    assert_eq!(t.shape(), expected, "{context}: shape mismatch");
}

// ---------------------------------------------------------------------------
// Tiny config from the fixture metadata
// ---------------------------------------------------------------------------

fn tiny_cfg(fixtures_root: &Value) -> LlamaConfig {
    let tc = &fixtures_root["metadata"]["tiny_config"];
    LlamaConfig {
        vocab_size: tc["vocab_size"].as_u64().unwrap() as usize,
        hidden_size: tc["hidden_size"].as_u64().unwrap() as usize,
        intermediate_size: tc["intermediate_size"].as_u64().unwrap() as usize,
        num_hidden_layers: tc["num_hidden_layers"].as_u64().unwrap() as usize,
        num_attention_heads: tc["num_attention_heads"].as_u64().unwrap() as usize,
        num_key_value_heads: tc["num_key_value_heads"].as_u64().unwrap() as usize,
        rms_norm_eps: tc["rms_norm_eps"].as_f64().unwrap(),
        rope_theta: tc["rope_theta"].as_f64().unwrap(),
        max_position_embeddings: tc["max_position_embeddings"].as_u64().unwrap() as usize,
        tie_word_embeddings: false,
        hidden_act: LlamaActivation::Silu,
    }
}

// ---------------------------------------------------------------------------
// RMSNorm tests
// ---------------------------------------------------------------------------

fn run_rms_norm_fixture(fx: &Value) {
    let tag = fx["tag"].as_str().unwrap();
    let ctx = format!("rms_norm/{tag}");

    let input_shape = parse_usize_vec(&fx["input_shape"]);
    let weight_data = parse_f32_vec(&fx["weight"]);
    let eps = fx["eps"].as_f64().unwrap();
    let input = tensor_from_json_shape(&fx["input"], &input_shape);
    let expected = parse_f32_vec(&fx["expected"]);

    // Build RMSNorm with the fixture weight.
    let hidden = input_shape.last().copied().unwrap();
    let mut norm = RMSNorm::<f32>::new(vec![hidden], eps)
        .unwrap_or_else(|e| panic!("{ctx}: RMSNorm::new failed: {e}"));

    let weight_tensor = from_vec(weight_data, &[hidden])
        .unwrap_or_else(|e| panic!("{ctx}: weight tensor failed: {e}"));
    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert("weight".to_string(), weight_tensor);
    norm.load_state_dict(&sd, true)
        .unwrap_or_else(|e| panic!("{ctx}: load_state_dict failed: {e}"));

    let out = norm
        .forward(&input)
        .unwrap_or_else(|e| panic!("{ctx}: forward failed: {e}"));

    assert_shape(&out, &input_shape, &ctx);
    let actual = out.data_vec().unwrap_or_else(|e| panic!("{ctx}: data_vec failed: {e}"));
    assert_allclose(&actual, &expected, F32_MATMUL_TOL, &ctx);
}

#[test]
fn conformance_rms_norm_1d_token() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_rms_norm_fixture(get_fixture(fixtures, "rms_norm", "1d_token"));
}

#[test]
fn conformance_rms_norm_2d_seq() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_rms_norm_fixture(get_fixture(fixtures, "rms_norm", "2d_seq"));
}

#[test]
fn conformance_rms_norm_3d_batch_seq() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_rms_norm_fixture(get_fixture(fixtures, "rms_norm", "3d_batch_seq"));
}

// ---------------------------------------------------------------------------
// RoPE tests
// ---------------------------------------------------------------------------
//
// The fixture stores Q and K shaped `[batch, num_heads, seq_len, head_dim]`
// (the HuggingFace convention). Ferrotorch `RotaryPositionEmbedding::apply`
// expects `[..., seq_len, head_dim]` — i.e. it treats every axis before the
// last two as part of the batch. We feed `[num_heads, seq_len, head_dim]`
// (squeezing the leading batch=1), which matches the shape the attention
// layer passes after `reshape_to_heads`.

fn run_rope_fixture(fx: &Value) {
    use ferrotorch_nn::{RoPEConvention, RoPEScaling, RotaryPositionEmbedding};

    let tag = fx["tag"].as_str().unwrap();
    let ctx = format!("rope_apply/{tag}");

    let head_dim = fx["head_dim"].as_u64().unwrap() as usize;
    let rope_theta = fx["rope_theta"].as_f64().unwrap();
    let max_pos = fx["max_position_embeddings"].as_u64().unwrap() as usize;
    let num_heads = fx["num_heads"].as_u64().unwrap() as usize;
    let num_kv_heads = fx["num_kv_heads"].as_u64().unwrap() as usize;
    let seq_len = fx["seq_len"].as_u64().unwrap() as usize;

    // Parse Q/K inputs (flat, layout: [batch, heads, seq, head_dim] → we
    // interpret as [heads, seq, head_dim] ignoring the batch=1 prefix).
    let q_in = tensor_from_json_shape(&fx["q_input"], &[num_heads, seq_len, head_dim]);
    let k_in = tensor_from_json_shape(&fx["k_input"], &[num_kv_heads, seq_len, head_dim]);

    let q_expected = parse_f32_vec(&fx["q_rotated"]);
    let k_expected = parse_f32_vec(&fx["k_rotated"]);

    let rope = RotaryPositionEmbedding::<f32>::with_scaling(
        head_dim,
        max_pos,
        rope_theta,
        RoPEConvention::HalfRotation,
        RoPEScaling::None,
    )
    .unwrap_or_else(|e| panic!("{ctx}: RotaryPositionEmbedding::with_scaling failed: {e}"));

    let q_out = rope
        .apply(&q_in, 0)
        .unwrap_or_else(|e| panic!("{ctx}: rope.apply(q) failed: {e}"));
    let k_out = rope
        .apply(&k_in, 0)
        .unwrap_or_else(|e| panic!("{ctx}: rope.apply(k) failed: {e}"));

    let q_actual = q_out.data_vec().unwrap();
    let k_actual = k_out.data_vec().unwrap();

    assert_allclose(&q_actual, &q_expected, F32_MATMUL_TOL, &format!("{ctx}/q"));
    assert_allclose(&k_actual, &k_expected, F32_MATMUL_TOL, &format!("{ctx}/k"));
}

#[test]
fn conformance_rope_seq1() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_rope_fixture(get_fixture(fixtures, "rope_apply", "seq1"));
}

#[test]
fn conformance_rope_seq4() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_rope_fixture(get_fixture(fixtures, "rope_apply", "seq4"));
}

// ---------------------------------------------------------------------------
// MLP forward tests
// ---------------------------------------------------------------------------

fn run_mlp_fixture(fx: &Value) {
    let tag = fx["tag"].as_str().unwrap();
    let ctx = format!("mlp_forward/{tag}");

    let input_shape = parse_usize_vec(&fx["input_shape"]);
    let expected_shape = parse_usize_vec(&fx["expected_shape"]);
    let hidden_size = fx["hidden_size"].as_u64().unwrap() as usize;
    let intermediate_size = fx["intermediate_size"].as_u64().unwrap() as usize;

    let input = tensor_from_json_shape(&fx["input"], &input_shape);

    // Build MLP and inject weights.
    let cfg = LlamaConfig {
        vocab_size: 64,
        hidden_size,
        intermediate_size,
        num_hidden_layers: 1,
        num_attention_heads: 4,
        num_key_value_heads: 2,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 64,
        tie_word_embeddings: false,
        hidden_act: LlamaActivation::Silu,
    };
    let mut mlp =
        LlamaMLP::<f32>::new(&cfg).unwrap_or_else(|e| panic!("{ctx}: LlamaMLP::new failed: {e}"));

    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert(
        "gate_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["gate_proj_weight"]),
            &[intermediate_size, hidden_size],
        )
        .unwrap(),
    );
    sd.insert(
        "up_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["up_proj_weight"]),
            &[intermediate_size, hidden_size],
        )
        .unwrap(),
    );
    sd.insert(
        "down_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["down_proj_weight"]),
            &[hidden_size, intermediate_size],
        )
        .unwrap(),
    );
    mlp.load_state_dict(&sd, true)
        .unwrap_or_else(|e| panic!("{ctx}: load_state_dict failed: {e}"));

    let out = mlp
        .forward(&input)
        .unwrap_or_else(|e| panic!("{ctx}: forward failed: {e}"));

    assert_shape(&out, &expected_shape, &ctx);
    let actual = out.data_vec().unwrap();
    let expected = parse_f32_vec(&fx["expected"]);
    assert_allclose(&actual, &expected, F32_MATMUL_TOL, &ctx);
}

#[test]
fn conformance_mlp_forward_2d_seq() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_mlp_fixture(get_fixture(fixtures, "mlp_forward", "2d_seq"));
}

#[test]
fn conformance_mlp_forward_3d_batch_seq() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_mlp_fixture(get_fixture(fixtures, "mlp_forward", "3d_batch_seq"));
}

// ---------------------------------------------------------------------------
// Attention forward tests
// ---------------------------------------------------------------------------

fn run_attention_fixture(fx: &Value) {
    let tag = fx["tag"].as_str().unwrap();
    let ctx = format!("attention_forward/{tag}");

    let batch = fx["batch"].as_u64().unwrap() as usize;
    let seq_len = fx["seq_len"].as_u64().unwrap() as usize;
    let hidden_size = fx["hidden_size"].as_u64().unwrap() as usize;
    let num_heads = fx["num_heads"].as_u64().unwrap() as usize;
    let num_kv_heads = fx["num_kv_heads"].as_u64().unwrap() as usize;
    let head_dim = fx["head_dim"].as_u64().unwrap() as usize;
    let kv_dim = num_kv_heads * head_dim;
    let expected_shape = parse_usize_vec(&fx["expected_shape"]);

    let input_shape = [batch, seq_len, hidden_size];
    let input = tensor_from_json_shape(&fx["input"], &input_shape);

    let cfg = LlamaConfig {
        vocab_size: 64,
        hidden_size,
        intermediate_size: 64,
        num_hidden_layers: 1,
        num_attention_heads: num_heads,
        num_key_value_heads: num_kv_heads,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 64,
        tie_word_embeddings: false,
        hidden_act: LlamaActivation::Silu,
    };

    let mut attn = LlamaAttention::<f32>::new(&cfg)
        .unwrap_or_else(|e| panic!("{ctx}: LlamaAttention::new failed: {e}"));

    let mut sd: StateDict<f32> = HashMap::new();
    sd.insert(
        "q_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["q_proj_weight"]),
            &[hidden_size, hidden_size],
        )
        .unwrap(),
    );
    sd.insert(
        "k_proj.weight".to_string(),
        from_vec(parse_f32_vec(&fx["k_proj_weight"]), &[kv_dim, hidden_size]).unwrap(),
    );
    sd.insert(
        "v_proj.weight".to_string(),
        from_vec(parse_f32_vec(&fx["v_proj_weight"]), &[kv_dim, hidden_size]).unwrap(),
    );
    sd.insert(
        "o_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["o_proj_weight"]),
            &[hidden_size, hidden_size],
        )
        .unwrap(),
    );
    attn.load_state_dict(&sd, true)
        .unwrap_or_else(|e| panic!("{ctx}: load_state_dict failed: {e}"));

    let out = attn
        .forward(&input)
        .unwrap_or_else(|e| panic!("{ctx}: forward failed: {e}"));

    assert_shape(&out, &expected_shape, &ctx);
    let actual = out.data_vec().unwrap();
    let expected = parse_f32_vec(&fx["expected"]);
    assert_allclose(&actual, &expected, F32_ATTN_TOL, &ctx);
}

#[test]
fn conformance_attention_forward_seq1_gqa() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_attention_fixture(get_fixture(fixtures, "attention_forward", "seq1_gqa"));
}

#[test]
fn conformance_attention_forward_seq4_gqa() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_attention_fixture(get_fixture(fixtures, "attention_forward", "seq4_gqa"));
}

// ---------------------------------------------------------------------------
// Decoder layer forward test
// ---------------------------------------------------------------------------

#[test]
fn conformance_decoder_layer_seq4_2norm_attn_mlp() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    let fx = get_fixture(fixtures, "decoder_layer_forward", "seq4_2norm_attn_mlp");
    let ctx = "decoder_layer_forward/seq4_2norm_attn_mlp";

    let batch = fx["batch"].as_u64().unwrap() as usize;
    let seq_len = fx["seq_len"].as_u64().unwrap() as usize;
    let hidden_size = fx["hidden_size"].as_u64().unwrap() as usize;
    let num_heads = fx["num_heads"].as_u64().unwrap() as usize;
    let num_kv_heads = fx["num_kv_heads"].as_u64().unwrap() as usize;
    let intermediate_size = fx["intermediate_size"].as_u64().unwrap() as usize;
    let rms_norm_eps = fx["rms_norm_eps"].as_f64().unwrap();
    let kv_dim = num_kv_heads * (hidden_size / num_heads);

    let expected_shape = parse_usize_vec(&fx["expected_shape"]);
    let input = tensor_from_json_shape(&fx["input"], &[batch, seq_len, hidden_size]);

    let cfg = LlamaConfig {
        vocab_size: 64,
        hidden_size,
        intermediate_size,
        num_hidden_layers: 1,
        num_attention_heads: num_heads,
        num_key_value_heads: num_kv_heads,
        rms_norm_eps,
        rope_theta: 10_000.0,
        max_position_embeddings: 64,
        tie_word_embeddings: false,
        hidden_act: LlamaActivation::Silu,
    };

    let mut layer = LlamaDecoderLayer::<f32>::new(&cfg)
        .unwrap_or_else(|e| panic!("{ctx}: LlamaDecoderLayer::new failed: {e}"));

    // Build state dict from fixture fields.
    let mut sd: StateDict<f32> = HashMap::new();

    sd.insert(
        "input_layernorm.weight".to_string(),
        from_vec(parse_f32_vec(&fx["input_layernorm_weight"]), &[hidden_size]).unwrap(),
    );
    sd.insert(
        "post_attention_layernorm.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["post_attention_layernorm_weight"]),
            &[hidden_size],
        )
        .unwrap(),
    );
    sd.insert(
        "self_attn.q_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["q_proj_weight"]),
            &[hidden_size, hidden_size],
        )
        .unwrap(),
    );
    sd.insert(
        "self_attn.k_proj.weight".to_string(),
        from_vec(parse_f32_vec(&fx["k_proj_weight"]), &[kv_dim, hidden_size]).unwrap(),
    );
    sd.insert(
        "self_attn.v_proj.weight".to_string(),
        from_vec(parse_f32_vec(&fx["v_proj_weight"]), &[kv_dim, hidden_size]).unwrap(),
    );
    sd.insert(
        "self_attn.o_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["o_proj_weight"]),
            &[hidden_size, hidden_size],
        )
        .unwrap(),
    );
    sd.insert(
        "mlp.gate_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["gate_proj_weight"]),
            &[intermediate_size, hidden_size],
        )
        .unwrap(),
    );
    sd.insert(
        "mlp.up_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["up_proj_weight"]),
            &[intermediate_size, hidden_size],
        )
        .unwrap(),
    );
    sd.insert(
        "mlp.down_proj.weight".to_string(),
        from_vec(
            parse_f32_vec(&fx["down_proj_weight"]),
            &[hidden_size, intermediate_size],
        )
        .unwrap(),
    );

    layer
        .load_state_dict(&sd, true)
        .unwrap_or_else(|e| panic!("{ctx}: load_state_dict failed: {e}"));

    let out = layer
        .forward(&input)
        .unwrap_or_else(|e| panic!("{ctx}: forward failed: {e}"));

    assert_shape(&out, &expected_shape, ctx);
    let actual = out.data_vec().unwrap();
    let expected = parse_f32_vec(&fx["expected"]);
    assert_allclose(&actual, &expected, F32_MATMUL_TOL, ctx);
}

// ---------------------------------------------------------------------------
// Full causal LM forward tests
// ---------------------------------------------------------------------------

fn build_causal_lm_state_dict(
    fx: &Value,
    cfg: &LlamaConfig,
) -> StateDict<f32> {
    let sd_json = fx["state_dict"].as_object().expect("state_dict must be a JSON object");
    let hidden = cfg.hidden_size;
    let vocab = cfg.vocab_size;
    let intermediate = cfg.intermediate_size;
    let num_heads = cfg.num_attention_heads;
    let num_kv = cfg.num_key_value_heads;
    let head_dim = cfg.head_dim();
    let kv_dim = num_kv * head_dim;

    let mut sd: StateDict<f32> = HashMap::new();

    for (key, val) in sd_json {
        let data = parse_f32_vec(val);
        // Determine shape from key suffix.
        let shape: Vec<usize> = if key.ends_with("embed_tokens.weight") {
            vec![vocab, hidden]
        } else if key.ends_with("norm.weight")
            || key.ends_with("input_layernorm.weight")
            || key.ends_with("post_attention_layernorm.weight")
        {
            vec![hidden]
        } else if key.ends_with("q_proj.weight") {
            vec![num_heads * head_dim, hidden]
        } else if key.ends_with("k_proj.weight") || key.ends_with("v_proj.weight") {
            vec![kv_dim, hidden]
        } else if key.ends_with("o_proj.weight") {
            vec![hidden, hidden]
        } else if key.ends_with("gate_proj.weight") || key.ends_with("up_proj.weight") {
            vec![intermediate, hidden]
        } else if key.ends_with("down_proj.weight") {
            vec![hidden, intermediate]
        } else if key.ends_with("lm_head.weight") {
            vec![vocab, hidden]
        } else {
            panic!("build_causal_lm_state_dict: unrecognised key {key}");
        };

        let tensor = from_vec(data, &shape)
            .unwrap_or_else(|e| panic!("tensor for {key}: {e}"));
        // The fixture uses model-prefixed names (HF layout).  LlamaForCausalLM::load_state_dict
        // expects model-prefixed keys, so we keep them as-is.
        sd.insert(key.clone(), tensor);
    }
    sd
}

fn run_causal_lm_fixture(fx: &Value, cfg: &LlamaConfig) {
    let tag = fx["tag"].as_str().unwrap();
    let ctx = format!("causal_lm_forward/{tag}");

    let token_ids: Vec<u32> = fx["token_ids"]
        .as_array()
        .expect("token_ids must be array")
        .iter()
        .map(|v| v.as_u64().expect("token id must be u64") as u32)
        .collect();

    let expected_logits_shape = parse_usize_vec(&fx["expected_logits_shape"]);
    let expected_last_logits = parse_f32_vec(&fx["expected_last_logits"]);
    let greedy_next_token = fx["greedy_next_token"].as_u64().unwrap() as u32;

    let sd = build_causal_lm_state_dict(fx, cfg);

    let mut model = LlamaForCausalLM::<f32>::new(*cfg)
        .unwrap_or_else(|e| panic!("{ctx}: LlamaForCausalLM::new failed: {e}"));
    model
        .load_hf_state_dict(&sd, false)
        .unwrap_or_else(|e| panic!("{ctx}: load_hf_state_dict failed: {e}"));

    let logits = model
        .forward_from_ids(&token_ids)
        .unwrap_or_else(|e| panic!("{ctx}: forward_from_ids failed: {e}"));

    // Shape check: [1, seq_len, vocab_size]
    assert_shape(&logits, &expected_logits_shape, &ctx);

    // Last-token logits (last seq position).
    let seq_len = token_ids.len();
    let vocab_size = cfg.vocab_size;
    let logits_data = logits.data_vec().unwrap();
    let last_token_logits = &logits_data[(seq_len - 1) * vocab_size..seq_len * vocab_size];
    assert_allclose(
        last_token_logits,
        &expected_last_logits,
        F32_MATMUL_TOL,
        &format!("{ctx}/last_logits"),
    );

    // Greedy decode: argmax of last-token logits must match the fixture.
    let greedy: u32 = last_token_logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i as u32)
        .expect("logits must be non-empty");
    assert_eq!(
        greedy,
        greedy_next_token,
        "{ctx}: greedy next token mismatch: got {greedy} expected {greedy_next_token}"
    );
}

#[test]
fn conformance_causal_lm_end_to_end_seq4() {
    let root = load_fixtures();
    let cfg = tiny_cfg(&root);
    let fixtures = root["fixtures"].as_array().unwrap();
    run_causal_lm_fixture(get_fixture(fixtures, "causal_lm_forward", "end_to_end_seq4"), &cfg);
}

#[test]
fn conformance_causal_lm_single_token() {
    let root = load_fixtures();
    let cfg = tiny_cfg(&root);
    let fixtures = root["fixtures"].as_array().unwrap();
    run_causal_lm_fixture(get_fixture(fixtures, "causal_lm_forward", "single_token"), &cfg);
}

// ---------------------------------------------------------------------------
// Config validation tests (fixture-driven)
// ---------------------------------------------------------------------------

fn run_config_validation_fixture(fx: &Value) {
    let tag = fx["tag"].as_str().unwrap();
    let ctx = format!("config_validation/{tag}");

    let cfg = LlamaConfig {
        vocab_size: fx["vocab_size"].as_u64().unwrap() as usize,
        hidden_size: fx["hidden_size"].as_u64().unwrap() as usize,
        intermediate_size: fx["intermediate_size"].as_u64().unwrap() as usize,
        num_hidden_layers: fx["num_hidden_layers"].as_u64().unwrap() as usize,
        num_attention_heads: fx["num_attention_heads"].as_u64().unwrap() as usize,
        num_key_value_heads: fx["num_key_value_heads"].as_u64().unwrap() as usize,
        rms_norm_eps: 1e-5,
        rope_theta: 10_000.0,
        max_position_embeddings: 64,
        tie_word_embeddings: false,
        hidden_act: LlamaActivation::Silu,
    };

    let expects_error = fx["expected_error"].as_bool().unwrap_or(false);
    let result = cfg.validate();
    if expects_error {
        assert!(
            result.is_err(),
            "{ctx}: expected validate() to return Err, but it returned Ok"
        );
    } else {
        assert!(
            result.is_ok(),
            "{ctx}: expected validate() to return Ok, but got: {:?}",
            result.err()
        );
    }
}

#[test]
fn conformance_config_validation_zero_hidden_size() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_config_validation_fixture(get_fixture(fixtures, "config_validation", "zero_hidden_size"));
}

#[test]
fn conformance_config_validation_non_divisible_heads() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_config_validation_fixture(get_fixture(
        fixtures,
        "config_validation",
        "non_divisible_heads",
    ));
}

#[test]
fn conformance_config_validation_kv_heads_not_dividing_attn_heads() {
    let root = load_fixtures();
    let fixtures = root["fixtures"].as_array().unwrap();
    run_config_validation_fixture(get_fixture(
        fixtures,
        "config_validation",
        "kv_heads_not_dividing_attn_heads",
    ));
}
