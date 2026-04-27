//! GPU dispatch for [`super::JsonSchemaProcessor::compute_mask`].
//!
//! Bridges the CPU-side `JsonGrammar` state to the
//! [`ferrotorch_cubecl::compute_token_mask_dfa_to_gpu`] kernel:
//!
//! 1. Inspect the current grammar via [`super::state::JsonGrammar::boolean_emission_stage`].
//! 2. If the state is DFA-compilable (stage 2 supports `Schema::Boolean`
//!    only), build the per-token DFA tables on the host.
//! 3. Pack the processor's vocab as `(offsets, chars)` u32 buffers.
//! 4. Dispatch the CubeCL kernel; read back the allow mask.
//! 5. If the state isn't compilable, return `None` so callers fall through
//!    to the existing CPU loop in `JsonSchemaProcessor::compute_mask`.
//!
//! Compiled only when the `cuda` feature is enabled.

use cubecl::prelude::{ComputeClient, Runtime};
use ferrotorch_cubecl::{DfaMaskInputs, compute_token_mask_dfa_to_gpu};

use super::state::{
    BooleanEmissionStage, IntegerEmissionStage, NullEmissionStage, NumberEmissionStage,
    StringEmissionStage,
};
use super::json_schema::{JsonSchemaProcessor, TokenMask};

/// One DFA built from a grammar state. All buffers are owned `Vec<u32>`s
/// because the kernel launcher takes them by reference, and they need to
/// outlive the launcher call.
struct CompiledDfa {
    transitions: Vec<u32>,
    char_classes: Vec<u32>,
    num_classes: u32,
    start_state: u32,
    reject_state: u32,
}

/// Compile a [`NullEmissionStage`] into a finite DFA. Mirrors
/// [`compile_dfa_for_boolean`] but for the single literal `"null"`.
fn compile_dfa_for_null(stage: &NullEmissionStage) -> CompiledDfa {
    match stage {
        NullEmissionStage::Start => compile_linear_literal("null"),
        NullEmissionStage::Partial { remaining } => compile_linear_literal(remaining),
    }
}

/// Compile an [`IntegerEmissionStage`] into a finite DFA. Top-level
/// integers only (no parent terminator chars). The DFA covers all
/// reachable forward states from the given starting stage.
///
/// Char classes:
/// - 0 = `'-'`
/// - 1 = `'0'` (zero specifically)
/// - 2 = `'1'..='9'` (non-zero digit)
/// - 3 = OTHER (anything else → REJECT)
///
/// States:
/// - 0 = start (need `'-'` or any digit)
/// - 1 = after `'-'` (need any digit)
/// - 2 = after a single `'0'` (REJECT on any further char — JSON forbids `01`)
/// - 3 = after one or more non-zero digits (more digits valid; nothing else)
/// - 4 = REJECT
fn compile_dfa_for_integer(stage: &IntegerEmissionStage) -> CompiledDfa {
    let class_minus = 0u32;
    let class_zero = 1u32;
    let class_pos_digit = 2u32;
    let class_other = 3u32;
    let num_classes = 4u32;

    let mut char_classes = vec![class_other; 128];
    char_classes[b'-' as usize] = class_minus;
    char_classes[b'0' as usize] = class_zero;
    for d in b'1'..=b'9' {
        char_classes[d as usize] = class_pos_digit;
    }

    let num_states = 5usize;
    let reject = 4u32;
    let nc = num_classes as usize;
    let mut transitions = vec![reject; num_states * nc];
    let row = |s: usize, c: u32| s * nc + c as usize;

    transitions[row(0, class_minus)] = 1;
    transitions[row(0, class_zero)] = 2;
    transitions[row(0, class_pos_digit)] = 3;
    transitions[row(1, class_zero)] = 2;
    transitions[row(1, class_pos_digit)] = 3;
    // state 2 (after-zero): every class → REJECT (already set).
    transitions[row(3, class_zero)] = 3;
    transitions[row(3, class_pos_digit)] = 3;
    // state 4 (REJECT): every class → REJECT (already set).

    let start_state = match stage {
        IntegerEmissionStage::Start => 0,
        IntegerEmissionStage::AfterSign => 1,
        IntegerEmissionStage::AfterZero => 2,
        IntegerEmissionStage::AfterDigits => 3,
    };

    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state,
        reject_state: reject,
    }
}

/// Compile a [`NumberEmissionStage`] into a finite DFA. Top-level
/// numbers only. Adds the decimal-point + fractional-digit dimensions
/// over [`compile_dfa_for_integer`].
///
/// Char classes:
/// - 0 = `'-'`
/// - 1 = `'0'`
/// - 2 = `'1'..='9'`
/// - 3 = `'.'`
/// - 4 = OTHER
///
/// States:
/// - 0 = start
/// - 1 = after `'-'`
/// - 2 = after `'0'`, no decimal — only `'.'` is valid
/// - 3 = after non-zero integer digits, no decimal — digits or `'.'`
/// - 4 = after `'.'` with no fractional digit yet (mid_decimal) — only digits
/// - 5 = after one or more fractional digits — only digits
/// - 6 = REJECT
fn compile_dfa_for_number(stage: &NumberEmissionStage) -> CompiledDfa {
    let class_minus = 0u32;
    let class_zero = 1u32;
    let class_pos_digit = 2u32;
    let class_dot = 3u32;
    let class_other = 4u32;
    let num_classes = 5u32;

    let mut char_classes = vec![class_other; 128];
    char_classes[b'-' as usize] = class_minus;
    char_classes[b'0' as usize] = class_zero;
    for d in b'1'..=b'9' {
        char_classes[d as usize] = class_pos_digit;
    }
    char_classes[b'.' as usize] = class_dot;

    let num_states = 7usize;
    let reject = 6u32;
    let nc = num_classes as usize;
    let mut transitions = vec![reject; num_states * nc];
    let row = |s: usize, c: u32| s * nc + c as usize;

    // state 0 (start): '-' / '0' / '1'-'9'
    transitions[row(0, class_minus)] = 1;
    transitions[row(0, class_zero)] = 2;
    transitions[row(0, class_pos_digit)] = 3;
    // state 1 (after '-'): '0' / '1'-'9'
    transitions[row(1, class_zero)] = 2;
    transitions[row(1, class_pos_digit)] = 3;
    // state 2 (after '0', no decimal): only '.'
    transitions[row(2, class_dot)] = 4;
    // state 3 (after non-zero integer digits, no decimal): digits or '.'
    transitions[row(3, class_zero)] = 3;
    transitions[row(3, class_pos_digit)] = 3;
    transitions[row(3, class_dot)] = 4;
    // state 4 (mid-decimal, no fractional digit): only digits
    transitions[row(4, class_zero)] = 5;
    transitions[row(4, class_pos_digit)] = 5;
    // state 5 (after fractional digit): more digits
    transitions[row(5, class_zero)] = 5;
    transitions[row(5, class_pos_digit)] = 5;
    // state 6 (REJECT): already set.

    let start_state = match stage {
        NumberEmissionStage::Start => 0,
        NumberEmissionStage::AfterSign => 1,
        NumberEmissionStage::AfterZeroNoDecimal => 2,
        NumberEmissionStage::AfterDigitsNoDecimal => 3,
        NumberEmissionStage::AfterDecimalNoFrac => 4,
        NumberEmissionStage::AfterFractionalDigits => 5,
    };

    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state,
        reject_state: reject,
    }
}

/// Compile a [`StringEmissionStage`] into a DFA for `Schema::String`
/// (non-enum). Two grammar states map onto three DFA states.
///
/// Char classes:
/// - 0 = `'"'` (string delimiter)
/// - 1 = `'\\'` (escapes are intentionally unsupported per the grammar)
/// - 2 = printable ASCII other than `'"'` / `'\\'` (i.e. `0x20..=0x7E` minus those two)
/// - 3 = OTHER (control chars, non-ASCII — REJECT)
///
/// States:
/// - 0 = `Phase::Start`, expects opening `'"'`
/// - 1 = inside body, accepts content chars or closing `'"'`
/// - 2 = closed (after `'"'`), any further char rejects
/// - 3 = REJECT
fn compile_dfa_for_string(stage: &StringEmissionStage) -> CompiledDfa {
    let class_quote = 0u32;
    let class_backslash = 1u32;
    let class_content = 2u32;
    let class_other = 3u32;
    let num_classes = 4u32;

    let mut char_classes = vec![class_other; 128];
    for b in 0x20u8..=0x7Eu8 {
        char_classes[b as usize] = class_content;
    }
    char_classes[b'"' as usize] = class_quote;
    char_classes[b'\\' as usize] = class_backslash;

    let num_states = 4usize;
    let reject = 3u32;
    let nc = num_classes as usize;
    let mut transitions = vec![reject; num_states * nc];
    let row = |s: usize, c: u32| s * nc + c as usize;

    // state 0 (Start): only opening '"' is valid
    transitions[row(0, class_quote)] = 1;
    // state 1 (in body): content chars stay in body, closing '"' goes to closed
    transitions[row(1, class_content)] = 1;
    transitions[row(1, class_quote)] = 2;
    // backslash → REJECT (already set; escapes unsupported by the grammar).
    // state 2 (closed): any further char → REJECT (already set).
    // state 3 (REJECT): already set.

    let start_state = match stage {
        StringEmissionStage::Start => 0,
        StringEmissionStage::InBody => 1,
    };

    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state,
        reject_state: reject,
    }
}

/// Compile a [`BooleanEmissionStage`] into a finite DFA.
///
/// State numbering convention:
///
/// - `0` is the start state.
/// - `1..=N` are intermediate states corresponding to characters
///   already accepted along the literal.
/// - `REJECT = num_states - 1`. Defined explicitly so the kernel's
///   `state == reject_state` short-circuit fires.
///
/// For [`BooleanEmissionStage::Start`] the DFA branches: from `0`,
/// `'t'` → state 1 (head of "rue"), `'f'` → state 5 (head of "alse"),
/// any other class → REJECT. Then both branches walk linearly to their
/// respective accept positions, after which any further char rejects
/// (the grammar would be `done`, but the kernel still needs to handle
/// tokens that try to emit past the literal's end).
///
/// For [`BooleanEmissionStage::PartialTrue { remaining }`] the DFA is
/// just the linear walk over `remaining`'s chars, plus REJECT. Same for
/// `PartialFalse`.
fn compile_dfa_for_boolean(stage: &BooleanEmissionStage) -> CompiledDfa {
    match stage {
        BooleanEmissionStage::Start => compile_boolean_full(),
        BooleanEmissionStage::PartialTrue { remaining } => compile_linear_literal(remaining),
        BooleanEmissionStage::PartialFalse { remaining } => compile_linear_literal(remaining),
    }
}

/// DFA for the full `Schema::Boolean` at `Phase::Start`: accept any
/// prefix of `"true"` or `"false"`, reject everything else.
fn compile_boolean_full() -> CompiledDfa {
    // Char classes: t r u e f a l s OTHER  →  9 classes.
    let class_t = 0u32;
    let class_r = 1u32;
    let class_u = 2u32;
    let class_e = 3u32;
    let class_f = 4u32;
    let class_a = 5u32;
    let class_l = 6u32;
    let class_s = 7u32;
    let class_other = 8u32;
    let num_classes = 9u32;

    let mut char_classes = vec![class_other; 128];
    char_classes[b't' as usize] = class_t;
    char_classes[b'r' as usize] = class_r;
    char_classes[b'u' as usize] = class_u;
    char_classes[b'e' as usize] = class_e;
    char_classes[b'f' as usize] = class_f;
    char_classes[b'a' as usize] = class_a;
    char_classes[b'l' as usize] = class_l;
    char_classes[b's' as usize] = class_s;

    // States:
    //  0 = start (need 't' or 'f')
    //  1 = saw "t" (need 'r')
    //  2 = saw "tr" (need 'u')
    //  3 = saw "tru" (need 'e')
    //  4 = saw "true" (any further char rejects)
    //  5 = saw "f" (need 'a')
    //  6 = saw "fa" (need 'l')
    //  7 = saw "fal" (need 's')
    //  8 = saw "fals" (need 'e')
    //  9 = saw "false" (any further char rejects)
    // 10 = REJECT
    let num_states = 11usize;
    let reject = 10u32;
    let mut transitions = vec![reject; num_states * num_classes as usize];

    let nc = num_classes as usize;
    // Index helper makes the row * nc pattern explicit and avoids the
    // clippy::erasing_op false-positive on `0 * nc`.
    let row = |state: usize, class: u32| state * nc + class as usize;
    transitions[row(0, class_t)] = 1;
    transitions[row(0, class_f)] = 5;
    transitions[row(1, class_r)] = 2;
    transitions[row(2, class_u)] = 3;
    transitions[row(3, class_e)] = 4;
    // state 4 (= "true" complete): every class falls through to REJECT (already set).
    transitions[row(5, class_a)] = 6;
    transitions[row(6, class_l)] = 7;
    transitions[row(7, class_s)] = 8;
    transitions[row(8, class_e)] = 9;
    // state 9 (= "false" complete): every class → REJECT (already set).
    // state 10 (REJECT): every class → REJECT (already set).

    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state: 0,
        reject_state: reject,
    }
}

/// DFA accepting any prefix of `literal`. Used for the
/// `PartialTrue` / `PartialFalse` stages: we've already emitted the head
/// of "true" or "false", and `literal` is the remaining suffix to match.
fn compile_linear_literal(literal: &str) -> CompiledDfa {
    // Build a per-char class table over literal's distinct chars.
    // 8 distinct ASCII letters at most for boolean ("true" / "false"); this
    // generalises cleanly to any short literal.
    let mut classes_for_char = std::collections::BTreeMap::<char, u32>::new();
    let mut next_class: u32 = 0;
    for c in literal.chars() {
        classes_for_char.entry(c).or_insert_with(|| {
            let id = next_class;
            next_class += 1;
            id
        });
    }
    let class_other = next_class;
    let num_classes = next_class + 1;

    let mut char_classes = vec![class_other; 128];
    for (&c, &id) in &classes_for_char {
        if (c as u32) < 128 {
            char_classes[c as usize] = id;
        }
    }

    // States: 0 .. literal.len() are intermediate (state `n` is the accept
    // state — we land on it when the literal completes), literal.len() + 1
    // is REJECT. Every char emitted *past* the accept state lands on
    // REJECT, matching the CPU grammar's "already complete" rejection.
    let n = literal.chars().count();
    let reject = (n + 1) as u32;
    let num_states = n + 2;
    let nc = num_classes as usize;
    let mut transitions = vec![reject; num_states * nc];

    for (i, c) in literal.chars().enumerate() {
        let class = *classes_for_char.get(&c).expect("class table built above");
        transitions[i * nc + class as usize] = (i + 1) as u32;
    }
    // state `n` (accept): every class → REJECT (already set).
    // state `reject`: every class → REJECT (already set).

    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state: 0,
        reject_state: reject,
    }
}

/// Pre-packed vocab buffers ready for upload. Computed once per
/// (processor, vocab) and cached on the call site since vocabularies are
/// large (Llama-3 = 128k entries).
pub struct PackedVocab {
    pub offsets: Vec<u32>,
    pub chars: Vec<u32>,
    pub max_token_len: u32,
}

impl PackedVocab {
    /// Pack a string vocabulary into `(offsets, chars, max_token_len)`.
    ///
    /// `offsets[i] .. offsets[i+1]` is the slice of `chars` holding
    /// token `i`'s codepoints (one `u32` per Unicode scalar).
    /// `max_token_len` is the longest token's char count, used as the
    /// kernel's bounded-loop cap.
    pub fn pack(vocab: &[String]) -> Self {
        let mut offsets = Vec::with_capacity(vocab.len() + 1);
        let mut chars = Vec::new();
        let mut max_token_len: usize = 0;
        offsets.push(0u32);
        for tok in vocab {
            let mut tok_len = 0usize;
            for c in tok.chars() {
                chars.push(c as u32);
                tok_len += 1;
            }
            offsets.push(chars.len() as u32);
            if tok_len > max_token_len {
                max_token_len = tok_len;
            }
        }
        Self {
            offsets,
            chars,
            max_token_len: max_token_len as u32,
        }
    }
}

/// Try to compute the token-allow mask on GPU. Returns `None` if the
/// current grammar state isn't DFA-compilable (caller should fall
/// through to the CPU `compute_mask` path).
///
/// Stage 2 supports `Schema::Boolean` only; future stages extend the
/// match in [`compile_dfa_for_boolean`]'s caller to cover Null, Number,
/// StringEnum, etc.
pub fn compute_mask_gpu<R: Runtime>(
    processor: &JsonSchemaProcessor,
    client: &ComputeClient<R>,
    packed: &PackedVocab,
) -> Option<TokenMask> {
    // Try each supported emission-stage accessor in turn. Returning early
    // on the first match gives a deterministic priority — but at most one
    // accessor returns Some for any given grammar state (single-frame +
    // schema-discriminated), so the order is observationally irrelevant.
    let grammar = processor.grammar();
    let dfa = if let Some(stage) = grammar.boolean_emission_stage() {
        compile_dfa_for_boolean(&stage)
    } else if let Some(stage) = grammar.null_emission_stage() {
        compile_dfa_for_null(&stage)
    } else if let Some(stage) = grammar.integer_emission_stage() {
        compile_dfa_for_integer(&stage)
    } else if let Some(stage) = grammar.number_emission_stage() {
        compile_dfa_for_number(&stage)
    } else if let Some(stage) = grammar.string_emission_stage() {
        compile_dfa_for_string(&stage)
    } else {
        return None;
    };

    let inputs = DfaMaskInputs {
        transitions: &dfa.transitions,
        char_classes: &dfa.char_classes,
        vocab_offsets: &packed.offsets,
        vocab_chars: &packed.chars,
        num_classes: dfa.num_classes,
        start_state: dfa.start_state,
        reject_state: dfa.reject_state,
        max_token_len: packed.max_token_len,
    };
    let (handle, n) = compute_token_mask_dfa_to_gpu::<R>(client, &inputs);
    let bytes = client.read_one(handle).ok()?;
    if bytes.len() != n * std::mem::size_of::<u32>() {
        return None;
    }
    let mut allow = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        allow.push(u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    Some(TokenMask { allow })
}

// ---------------------------------------------------------------------------
// CUDA runtime tests — real GPU dispatch with byte-equality vs CPU.
// ---------------------------------------------------------------------------

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use super::*;
    use crate::grammar::schema::Schema;
    use cubecl_cuda::{CudaDevice, CudaRuntime};
    use serde_json::json;

    fn cuda_client() -> ComputeClient<CudaRuntime> {
        let device = CudaDevice { index: 0 };
        CudaRuntime::client(&device)
    }

    fn ascii_char_vocab() -> Vec<String> {
        (0x20u8..=0x7Eu8).map(|b| (b as char).to_string()).collect()
    }

    /// Parity: a fresh Boolean processor at Phase::Start. GPU mask must
    /// be byte-equal to the existing CPU `compute_mask` over the same
    /// vocab.
    #[test]
    fn boolean_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor =
            JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();

        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask =
            compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed).expect(
                "Schema::Boolean at Phase::Start must be DFA-compilable",
            );

        assert_eq!(
            cpu_mask.allow, gpu_mask.allow,
            "GPU mask must equal CPU mask byte-for-byte for Boolean@Start",
        );
        // Sanity: 't' and 'f' should be the only ASCII single-char tokens
        // accepted at the start of a boolean.
        let allowed_chars: Vec<char> = (0..gpu_mask.allow.len())
            .filter(|&i| gpu_mask.allow[i] != 0)
            .map(|i| vocab[i].chars().next().unwrap())
            .collect();
        assert!(allowed_chars.contains(&'t'));
        assert!(allowed_chars.contains(&'f'));
        assert!(!allowed_chars.contains(&'a'));
    }

    /// Parity after the first character has already been emitted.
    /// Stepping 't' moves the grammar into PartialTrue { remaining: "rue" }.
    #[test]
    fn boolean_gpu_mask_matches_cpu_after_t() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
        let t_id = vocab.iter().position(|s| s == "t").unwrap() as u32;
        processor.step_token(t_id).unwrap();

        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("PartialTrue must be DFA-compilable");

        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // After 't', only 'r' should be allowed among single-char tokens.
        let allowed_chars: Vec<char> = (0..gpu_mask.allow.len())
            .filter(|&i| gpu_mask.allow[i] != 0)
            .map(|i| vocab[i].chars().next().unwrap())
            .collect();
        assert!(allowed_chars.contains(&'r'));
        assert!(!allowed_chars.contains(&'t'));
        assert!(!allowed_chars.contains(&'u'));
    }

    /// Parity after stepping 'f': PartialFalse { remaining: "alse" }.
    #[test]
    fn boolean_gpu_mask_matches_cpu_after_f() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "boolean"}), vocab.clone()).unwrap();
        let f_id = vocab.iter().position(|s| s == "f").unwrap() as u32;
        processor.step_token(f_id).unwrap();

        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("PartialFalse must be DFA-compilable");

        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        let allowed_chars: Vec<char> = (0..gpu_mask.allow.len())
            .filter(|&i| gpu_mask.allow[i] != 0)
            .map(|i| vocab[i].chars().next().unwrap())
            .collect();
        assert!(allowed_chars.contains(&'a'));
        assert!(!allowed_chars.contains(&'f'));
    }

    // -----------------------------------------------------------------
    // Stage 3 — Null
    // -----------------------------------------------------------------

    #[test]
    fn null_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor =
            JsonSchemaProcessor::new(&json!({"type": "null"}), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Schema::Null at Phase::Start must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // Sanity: only 'n' should be allowed at the start of `null`.
        let n_id = vocab.iter().position(|s| s == "n").unwrap();
        let m_id = vocab.iter().position(|s| s == "m").unwrap();
        assert_eq!(gpu_mask.allow[n_id], 1);
        assert_eq!(gpu_mask.allow[m_id], 0);
    }

    #[test]
    fn null_gpu_mask_matches_cpu_after_n() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "null"}), vocab.clone()).unwrap();
        let n_id = vocab.iter().position(|s| s == "n").unwrap() as u32;
        processor.step_token(n_id).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Null Partial must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        let u_id = vocab.iter().position(|s| s == "u").unwrap();
        assert_eq!(gpu_mask.allow[u_id], 1);
    }

    // -----------------------------------------------------------------
    // Stage 3 — Integer
    // -----------------------------------------------------------------

    #[test]
    fn integer_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor =
            JsonSchemaProcessor::new(&json!({"type": "integer"}), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Integer Start must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // Sanity: '-' and '0'-'9' are allowed; letters aren't.
        for d in '0'..='9' {
            let i = vocab.iter().position(|s| s == &d.to_string()).unwrap();
            assert_eq!(gpu_mask.allow[i], 1, "digit {d} should be allowed");
        }
        let minus = vocab.iter().position(|s| s == "-").unwrap();
        assert_eq!(gpu_mask.allow[minus], 1);
        let a = vocab.iter().position(|s| s == "a").unwrap();
        assert_eq!(gpu_mask.allow[a], 0);
    }

    #[test]
    fn integer_gpu_mask_matches_cpu_after_sign() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "integer"}), vocab.clone()).unwrap();
        let minus = vocab.iter().position(|s| s == "-").unwrap() as u32;
        processor.step_token(minus).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Integer AfterSign must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
    }

    #[test]
    fn integer_gpu_mask_matches_cpu_after_zero() {
        // Leading-zero edge case: after '0', no further chars are valid
        // for a top-level integer (the JSON forbids "01").
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "integer"}), vocab.clone()).unwrap();
        let zero = vocab.iter().position(|s| s == "0").unwrap() as u32;
        processor.step_token(zero).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Integer AfterZero must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // CPU and GPU both reject every single-char token here.
        assert!(
            gpu_mask.allow.iter().all(|&a| a == 0),
            "AfterZero with no parent must reject everything",
        );
    }

    #[test]
    fn integer_gpu_mask_matches_cpu_after_digits() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "integer"}), vocab.clone()).unwrap();
        let four = vocab.iter().position(|s| s == "4").unwrap() as u32;
        let two = vocab.iter().position(|s| s == "2").unwrap() as u32;
        processor.step_token(four).unwrap();
        processor.step_token(two).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Integer AfterDigits must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
    }

    // -----------------------------------------------------------------
    // Stage 3 — Number
    // -----------------------------------------------------------------

    #[test]
    fn number_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor =
            JsonSchemaProcessor::new(&json!({"type": "number"}), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Number Start must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
    }

    #[test]
    fn number_gpu_mask_matches_cpu_after_zero_is_only_dot() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "number"}), vocab.clone()).unwrap();
        let zero = vocab.iter().position(|s| s == "0").unwrap() as u32;
        processor.step_token(zero).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Number AfterZeroNoDecimal must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // Only '.' should be allowed.
        let dot = vocab.iter().position(|s| s == ".").unwrap();
        assert_eq!(gpu_mask.allow[dot], 1);
        let one = vocab.iter().position(|s| s == "1").unwrap();
        assert_eq!(gpu_mask.allow[one], 0);
    }

    #[test]
    fn number_gpu_mask_matches_cpu_mid_decimal() {
        // After "3." the grammar is in NumberDigits with had_decimal=true,
        // had_fractional_digit=false (mid_decimal). Only digits are valid.
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "number"}), vocab.clone()).unwrap();
        let three = vocab.iter().position(|s| s == "3").unwrap() as u32;
        let dot = vocab.iter().position(|s| s == ".").unwrap() as u32;
        processor.step_token(three).unwrap();
        processor.step_token(dot).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Number AfterDecimalNoFrac must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // Mid-decimal: only digits valid; '.' specifically rejected.
        let dot_idx = vocab.iter().position(|s| s == ".").unwrap();
        assert_eq!(gpu_mask.allow[dot_idx], 0, "second '.' must reject mid-decimal");
        let one = vocab.iter().position(|s| s == "1").unwrap();
        assert_eq!(gpu_mask.allow[one], 1);
    }

    #[test]
    fn number_gpu_mask_matches_cpu_after_fractional() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "number"}), vocab.clone()).unwrap();
        for s in ["3", ".", "1"] {
            let id = vocab.iter().position(|t| t == s).unwrap() as u32;
            processor.step_token(id).unwrap();
        }
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Number AfterFractionalDigits must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
    }

    // -----------------------------------------------------------------
    // Stage 3 — String (non-enum)
    // -----------------------------------------------------------------

    #[test]
    fn string_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor =
            JsonSchemaProcessor::new(&json!({"type": "string"}), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("String Start must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // Only the opening '"' should be allowed.
        let dq = vocab.iter().position(|s| s == "\"").unwrap();
        assert_eq!(gpu_mask.allow[dq], 1);
        let a = vocab.iter().position(|s| s == "a").unwrap();
        assert_eq!(gpu_mask.allow[a], 0);
    }

    #[test]
    fn string_gpu_mask_matches_cpu_in_body() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": "string"}), vocab.clone()).unwrap();
        let dq = vocab.iter().position(|s| s == "\"").unwrap() as u32;
        processor.step_token(dq).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("String InBody must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // Backslash is intentionally rejected (escapes unsupported).
        let bs = vocab.iter().position(|s| s == "\\").unwrap();
        assert_eq!(gpu_mask.allow[bs], 0, "backslash must reject (no escapes)");
    }

    // -----------------------------------------------------------------
    // Gate verification: non-Boolean / non-stage-3 schemas fall through.
    // -----------------------------------------------------------------

    /// Schemas not yet supported (Object, StringEnum, Array, Nullable)
    /// must return `None` so the caller falls back to CPU.
    #[test]
    fn unsupported_schema_returns_none() {
        let vocab = ascii_char_vocab();
        let processor = JsonSchemaProcessor::new(
            &json!({
                "type": "object",
                "properties": {"v": {"type": "boolean"}},
                "required": ["v"]
            }),
            vocab.clone(),
        )
        .unwrap();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let res = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed);
        assert!(
            res.is_none(),
            "Object schema must return None; stage 3 doesn't handle it",
        );

        // Sanity: directly verify the underlying API too.
        assert!(matches!(
            crate::grammar::schema::Schema::from_json_schema(&json!({
                "type": "object",
                "properties": {"v": {"type": "boolean"}},
                "required": ["v"]
            })),
            Ok(Schema::Object { .. })
        ));
    }
}
