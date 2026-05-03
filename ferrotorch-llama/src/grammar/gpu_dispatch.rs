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

use super::json_schema::{JsonSchemaProcessor, TokenMask};
use super::schema::Schema;
use super::state::{
    BooleanEmissionStage, IntegerEmissionStage, NullEmissionStage, NullableEmissionStage,
    NumberEmissionStage, ObjectKeyEmissionStage, StringEmissionStage, StringEnumEmissionStage,
};

/// One DFA built from a grammar state. All buffers are owned `Vec<u32>`s
/// because the kernel launcher takes them by reference, and they need to
/// outlive the launcher call.
///
/// `complete_states` is non-empty only when the wrapped schema has
/// states that are syntactically valid completion points (e.g. after
/// `"true"` for a Boolean, or after at least one digit for an Integer).
/// Multi-frame dispatch uses this to know which states should accept
/// the parent's terminator chars (`,`, `}`, `]`). Single-frame
/// dispatch leaves it empty.
struct CompiledDfa {
    transitions: Vec<u32>,
    char_classes: Vec<u32>,
    num_classes: u32,
    start_state: u32,
    reject_state: u32,
    complete_states: Vec<u32>,
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

    // Integer is complete (a valid integer has been emitted) at
    // AfterZero (just "0") and AfterDigits (one or more non-zero
    // digits, possibly preceded by '-'). AfterSign (just "-") is not
    // complete.
    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state,
        reject_state: reject,
        complete_states: vec![2, 3],
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

    // Number is complete at every digit-emitting state EXCEPT
    // AfterDecimalNoFrac, where the grammar requires at least one
    // fractional digit before the value can terminate.
    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state,
        reject_state: reject,
        complete_states: vec![2, 3, 5],
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

    // String is complete only after the closing '"' (state 2).
    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state,
        reject_state: reject,
        complete_states: vec![2],
    }
}

/// Compile a `Schema::StringEnum` DFA from a prefix trie over the
/// allowed values. State numbering:
///
/// - `0` = `Phase::Start` (only `'"'` is valid).
/// - `1` = trie root (after the opening `'"'`).
/// - `2..K` = trie nodes, one per distinct prefix that appears in any
///   allowed value.
/// - `K` = closed-string accept state (after the closing `'"'`; nothing
///   further is valid).
/// - `K + 1` = REJECT.
///
/// Char classes:
///
/// - one per distinct ASCII char appearing in any allowed value;
/// - `'"'` gets its own class (always present, used for open + close);
/// - `OTHER` (final class) covers anything else.
///
/// `start_partial` selects which trie node the DFA's start_state points
/// at. Empty `start_partial` ⇒ trie root (state 1, used for both
/// `Phase::Start` after the open quote and the just-opened body).
/// Non-empty ⇒ the trie node reached by walking those chars; `None`
/// returned from the parent call when `start_partial` doesn't match any
/// node (which should be unreachable for a valid grammar).
fn compile_dfa_for_string_enum(
    stage: &StringEnumEmissionStage<'_>,
    values: &[String],
) -> Option<CompiledDfa> {
    // Trie node = ordered sequence of chars walked from the root.
    // `prefix_to_state` maps a known prefix to its state id (offset by 1
    // because state 0 is Phase::Start).
    let mut prefix_to_state: std::collections::BTreeMap<String, u32> =
        std::collections::BTreeMap::new();
    prefix_to_state.insert(String::new(), 1);
    let mut all_prefixes: Vec<String> = vec![String::new()];
    for v in values {
        let mut acc = String::new();
        for c in v.chars() {
            acc.push(c);
            if !prefix_to_state.contains_key(&acc) {
                let id = (prefix_to_state.len() + 1) as u32; // +1 for state 0
                prefix_to_state.insert(acc.clone(), id);
                all_prefixes.push(acc.clone());
            }
        }
    }

    // Closed accept and REJECT states sit after every trie node.
    let closed_state = (1 + all_prefixes.len()) as u32;
    let reject = closed_state + 1;
    let num_states = (reject + 1) as usize;

    // Char classes: each distinct char in any value gets its own class,
    // plus '"' and OTHER.
    let mut classes_for_char: std::collections::BTreeMap<char, u32> =
        std::collections::BTreeMap::new();
    let class_quote = 0u32;
    let mut next_class: u32 = 1;
    for v in values {
        for c in v.chars() {
            if c == '"' {
                continue; // Reserved for the literal quote class.
            }
            classes_for_char.entry(c).or_insert_with(|| {
                let id = next_class;
                next_class += 1;
                id
            });
        }
    }
    let class_other = next_class;
    let num_classes = next_class + 1;

    let mut char_classes = vec![class_other; 128];
    char_classes[b'"' as usize] = class_quote;
    for (&c, &id) in &classes_for_char {
        if (c as u32) < 128 {
            char_classes[c as usize] = id;
        }
    }

    let nc = num_classes as usize;
    let mut transitions = vec![reject; num_states * nc];
    let row = |s: u32, c: u32| s as usize * nc + c as usize;

    // State 0 (Phase::Start): only '"' → trie root.
    transitions[row(0, class_quote)] = 1;

    // Each trie node: for every char that extends the prefix toward a
    // value, transition to the deeper trie node. If the prefix is
    // itself a complete value, '"' transitions to closed_state.
    for prefix in &all_prefixes {
        let from_state = *prefix_to_state.get(prefix).unwrap();
        // Per-char extensions.
        for c in classes_for_char.keys().copied() {
            let mut extended = prefix.clone();
            extended.push(c);
            if let Some(&to_state) = prefix_to_state.get(&extended) {
                let class = *classes_for_char.get(&c).unwrap();
                transitions[row(from_state, class)] = to_state;
            }
        }
        // Close-quote allowed iff prefix is a complete enum value.
        if values.iter().any(|v| v == prefix) {
            transitions[row(from_state, class_quote)] = closed_state;
        }
    }
    // closed_state and reject_state: every class → REJECT (already set).

    let start_state = match stage {
        StringEnumEmissionStage::Start => 0,
        StringEnumEmissionStage::InBody { partial } => *prefix_to_state.get(*partial)?,
    };

    // StringEnum is complete only after the closing '"' lands on
    // closed_state.
    Some(CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state,
        reject_state: reject,
        complete_states: vec![closed_state],
    })
}

/// Append parent terminator support to an existing DFA.
///
/// At each state listed in `dfa.complete_states` (states that
/// represent a syntactically-valid scalar completion — e.g. for
/// Integer: AfterZero and AfterDigits), every `terminator` char gets
/// a dedicated class (via `split_class_for_char`) and transitions to a
/// fresh "popped" sink state. The popped state rejects any further
/// character.
///
/// This is the multi-frame extension: when the scalar lives inside an
/// Object property value or Array element, the parent contributes
/// terminator chars (`,`, `}`, `]`) that legally end the value. The
/// kernel walks one token's chars, so a token consisting of "value
/// chars + one terminator" is accepted; a token spanning the value
/// boundary into a *new* parent state (e.g. `,"` in BPE) is
/// conservatively rejected (CPU still accepts it). For ASCII single-
/// char vocabularies this is byte-equal; for real BPE vocabs it's a
/// known under-allow on rare cross-boundary structural tokens.
fn add_terminators_to_states(mut dfa: CompiledDfa, terminators: &[char]) -> CompiledDfa {
    if terminators.is_empty() || dfa.complete_states.is_empty() {
        return dfa;
    }

    let mut term_classes: Vec<u32> = Vec::with_capacity(terminators.len());
    for &c in terminators {
        if (c as u32) < 128 {
            term_classes.push(split_class_for_char(&mut dfa, c as u8));
        }
    }

    // Append a single "popped" sink state. Any class transitions to
    // REJECT (default fill). Multiple terminators all funnel into the
    // same popped state — that's correct because a token can include
    // at most one terminator char before its parent's transitions take
    // over, and we don't model the parent here.
    let nc = dfa.num_classes as usize;
    let n_old = dfa.transitions.len() / nc;
    let popped = n_old as u32;
    let new_total = n_old + 1;
    let mut new_t = vec![dfa.reject_state; new_total * nc];
    new_t[..n_old * nc].copy_from_slice(&dfa.transitions);

    for &complete in &dfa.complete_states {
        for &cls in &term_classes {
            new_t[complete as usize * nc + cls as usize] = popped;
        }
    }

    dfa.transitions = new_t;
    dfa
}

/// Compile a `Phase::ObjectKey` DFA from the still-unseen-property
/// candidates. Structurally identical to a `Schema::StringEnum` at
/// `Phase::StringChars` — a prefix trie over the candidates with
/// closing `'"'` enabled at trie nodes matching a complete value. We
/// reuse the StringEnum compiler with `StringEnumEmissionStage::InBody`
/// since by the time we're in `ObjectKey`, the opening `'"'` has
/// already been emitted (consumed by the grammar's transition into
/// `ObjectKey`).
fn compile_dfa_for_object_key(stage: &ObjectKeyEmissionStage<'_>) -> Option<CompiledDfa> {
    compile_dfa_for_string_enum(
        &StringEnumEmissionStage::InBody {
            partial: stage.partial,
        },
        stage.candidates,
    )
}

/// Ensure the byte-level char `c` has its own class in `dfa`. If the
/// class assigned to `c` is currently shared with at least one other
/// char (in `0..128`), split it: introduce a new class, point `c` at
/// the new class, and copy `c`'s old transition column into the new
/// column for every state. Returns `c`'s class (whether new or
/// pre-existing).
fn split_class_for_char(dfa: &mut CompiledDfa, c: u8) -> u32 {
    let original_class = dfa.char_classes[c as usize];
    let other_using =
        (0..128usize).any(|i| i != c as usize && dfa.char_classes[i] == original_class);
    if !other_using {
        return original_class;
    }
    let new_class = dfa.num_classes;
    let new_nc = (dfa.num_classes + 1) as usize;
    let old_nc = dfa.num_classes as usize;
    let n_states = dfa.transitions.len() / old_nc;
    let mut new_t = vec![dfa.reject_state; n_states * new_nc];
    for s in 0..n_states {
        for cl in 0..old_nc {
            new_t[s * new_nc + cl] = dfa.transitions[s * old_nc + cl];
        }
        new_t[s * new_nc + new_class as usize] =
            dfa.transitions[s * old_nc + original_class as usize];
    }
    dfa.transitions = new_t;
    dfa.num_classes = new_class + 1;
    dfa.char_classes[c as usize] = new_class;
    new_class
}

/// Merge an inner schema's DFA with a "null" branch: at the inner DFA's
/// start state, the char `'n'` triggers a 4-state walk through `"ull"`.
/// All other char-class transitions of the inner DFA are preserved.
///
/// Adds dedicated classes for `'n'`, `'u'`, and `'l'` if those chars
/// were sharing a class with other chars in the inner DFA. Each newly-
/// dedicated class inherits the inner state-by-state transitions for
/// the original char, so behaviour past state 0 doesn't change.
fn merge_null_branch(mut inner: CompiledDfa) -> CompiledDfa {
    let class_n = split_class_for_char(&mut inner, b'n');
    let class_u = split_class_for_char(&mut inner, b'u');
    let class_l = split_class_for_char(&mut inner, b'l');

    let nc = inner.num_classes as usize;
    let n_old_states = inner.transitions.len() / nc;

    let walk_u_state = n_old_states as u32;
    let walk_l_state = walk_u_state + 1;
    let walk_l2_state = walk_l_state + 1;
    let accept_null_state = walk_l2_state + 1;

    let new_total_states = n_old_states + 4;
    let mut new_t = vec![inner.reject_state; new_total_states * nc];
    new_t[..n_old_states * nc].copy_from_slice(&inner.transitions);

    // State 0: the 'n' class now jumps into the null-branch walk.
    new_t[class_n as usize] = walk_u_state;
    // Null-walk transitions (all other classes default to REJECT).
    new_t[walk_u_state as usize * nc + class_u as usize] = walk_l_state;
    new_t[walk_l_state as usize * nc + class_l as usize] = walk_l2_state;
    new_t[walk_l2_state as usize * nc + class_l as usize] = accept_null_state;

    inner.transitions = new_t;
    // Merged completion set: inner's existing complete states plus the
    // freshly-added "null" accept. Multi-frame dispatch attaches parent
    // terminators to all of them.
    inner.complete_states.push(accept_null_state);
    inner
}

/// Build a `Schema::Nullable(inner)` DFA at `Phase::Start`. Returns
/// `None` if `inner` is itself an unsupported schema (Object, Array, or
/// nested Nullable).
fn compile_dfa_for_nullable(inner: &Schema) -> Option<CompiledDfa> {
    let inner_dfa = match inner {
        Schema::Null => {
            // Nullable(Null) is degenerate but still valid; both paths
            // walk "null". Just compile the null DFA directly.
            return Some(compile_dfa_for_null(&NullEmissionStage::Start));
        }
        Schema::Boolean => compile_dfa_for_boolean(&BooleanEmissionStage::Start),
        Schema::Integer => compile_dfa_for_integer(&IntegerEmissionStage::Start),
        Schema::Number => compile_dfa_for_number(&NumberEmissionStage::Start),
        Schema::String => compile_dfa_for_string(&StringEmissionStage::Start),
        Schema::StringEnum(values) => {
            compile_dfa_for_string_enum(&StringEnumEmissionStage::Start, values)?
        }
        // Object / Array / nested Nullable / unknown → return None so
        // callers fall through to CPU compute_mask.
        _ => return None,
    };
    Some(merge_null_branch(inner_dfa))
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

    // Boolean is complete in two places: state 4 ("true" emitted) and
    // state 9 ("false" emitted).
    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state: 0,
        reject_state: reject,
        complete_states: vec![4, 9],
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

    // The literal walk is complete only when the full literal has been
    // emitted — state index `n`.
    CompiledDfa {
        transitions,
        char_classes,
        num_classes,
        start_state: 0,
        reject_state: reject,
        complete_states: vec![n as u32],
    }
}

/// Pre-packed vocab buffers ready for upload. Computed once per
/// (processor, vocab) and cached on the call site since vocabularies are
/// large (Llama-3 = 128k entries).
pub struct PackedVocab {
    /// Per-token offsets into `chars`. `offsets[i] .. offsets[i+1]` is
    /// token `i`'s codepoint slice. Length = `vocab.len() + 1`.
    pub offsets: Vec<u32>,
    /// Flat codepoint storage (one `u32` per Unicode scalar). Total
    /// length = sum of token char-lengths.
    pub chars: Vec<u32>,
    /// Longest token's char count, used as the kernel's bounded-loop cap.
    pub max_token_len: u32,
}

// Manual Debug — printing the full `offsets` (`vocab_size + 1` entries)
// or `chars` (token-summed) for a 128k-entry vocabulary makes Debug
// output unusable. Show only the lengths plus `max_token_len`, which
// is enough to reason about the upload size and per-step kernel cost.
impl std::fmt::Debug for PackedVocab {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PackedVocab")
            .field(
                "offsets",
                &format_args!("<Vec<u32> {} entries>", self.offsets.len()),
            )
            .field(
                "chars",
                &format_args!("<Vec<u32> {} entries>", self.chars.len()),
            )
            .field("max_token_len", &self.max_token_len)
            .finish()
    }
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
    // Try each supported emission-stage accessor in turn. Each scalar
    // accessor returns the *top* frame's stage (multi-frame aware); the
    // helper `top_frame_parent_terminators` then surfaces the parent
    // frame's terminator chars, which `add_terminators_to_states` bakes
    // into the compiled DFA at every completion state. For single-
    // frame grammars `terminators` is empty and the helper is a no-op,
    // so the same dispatch handles both single-frame (top-level scalar)
    // and nested (scalar inside Object/Array) shapes.
    let grammar = processor.grammar();
    let terminators = grammar.top_frame_parent_terminators();

    // ObjectKey is a non-scalar accessor (still multi-frame, but the
    // current top frame is an Object — not a scalar — and it carries
    // its own self-contained DFA shape). It's checked first so its
    // dispatch is independent of the scalar-with-terminators chain.
    if let Some(stage) = grammar.object_key_emission_stage() {
        let dfa = compile_dfa_for_object_key(&stage)?;
        return run_dfa_on_gpu(client, packed, &dfa);
    }

    let dfa = if let Some(stage) = grammar.boolean_emission_stage_top() {
        add_terminators_to_states(compile_dfa_for_boolean(&stage), &terminators)
    } else if let Some(stage) = grammar.null_emission_stage_top() {
        add_terminators_to_states(compile_dfa_for_null(&stage), &terminators)
    } else if let Some(stage) = grammar.integer_emission_stage_top() {
        add_terminators_to_states(compile_dfa_for_integer(&stage), &terminators)
    } else if let Some(stage) = grammar.number_emission_stage_top() {
        add_terminators_to_states(compile_dfa_for_number(&stage), &terminators)
    } else if let Some(stage) = grammar.string_emission_stage_top() {
        add_terminators_to_states(compile_dfa_for_string(&stage), &terminators)
    } else if let Some((stage, values)) = grammar.string_enum_emission_stage_top() {
        add_terminators_to_states(compile_dfa_for_string_enum(&stage, values)?, &terminators)
    } else if let Some(NullableEmissionStage::Start { inner }) = grammar.nullable_emission_stage() {
        add_terminators_to_states(compile_dfa_for_nullable(inner)?, &terminators)
    } else {
        return None;
    };

    run_dfa_on_gpu(client, packed, &dfa)
}

/// Build [`DfaMaskInputs`] from a compiled DFA + the host-packed
/// vocab, dispatch the kernel, and read the mask back. Returns `None`
/// if the device-side read returns the wrong byte count (corrupt
/// transfer).
fn run_dfa_on_gpu<R: Runtime>(
    client: &ComputeClient<R>,
    packed: &PackedVocab,
    dfa: &CompiledDfa,
) -> Option<TokenMask> {
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
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Schema::Boolean at Phase::Start must be DFA-compilable");

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
        let processor = JsonSchemaProcessor::new(&json!({"type": "null"}), vocab.clone()).unwrap();
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
        assert_eq!(
            gpu_mask.allow[dot_idx], 0,
            "second '.' must reject mid-decimal"
        );
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
    // Stage 4 — StringEnum
    // -----------------------------------------------------------------

    fn string_enum_schema() -> serde_json::Value {
        json!({"enum": ["high", "medium", "low"]})
    }

    #[test]
    fn string_enum_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor = JsonSchemaProcessor::new(&string_enum_schema(), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("StringEnum Start must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // Sanity: only '"' is allowed at Phase::Start.
        let dq = vocab.iter().position(|s| s == "\"").unwrap();
        assert_eq!(gpu_mask.allow[dq], 1);
        let h = vocab.iter().position(|s| s == "h").unwrap();
        assert_eq!(gpu_mask.allow[h], 0);
    }

    #[test]
    fn string_enum_gpu_mask_matches_cpu_after_open_quote() {
        let vocab = ascii_char_vocab();
        let mut processor = JsonSchemaProcessor::new(&string_enum_schema(), vocab.clone()).unwrap();
        let dq = vocab.iter().position(|s| s == "\"").unwrap() as u32;
        processor.step_token(dq).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("StringEnum InBody empty-partial must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // Only first chars of 'high'/'medium'/'low' are valid.
        for c in ['h', 'm', 'l'] {
            let i = vocab.iter().position(|s| s == &c.to_string()).unwrap();
            assert_eq!(gpu_mask.allow[i], 1, "first char {c} must be allowed");
        }
        let x = vocab.iter().position(|s| s == "x").unwrap();
        assert_eq!(gpu_mask.allow[x], 0);
    }

    #[test]
    fn string_enum_gpu_mask_matches_cpu_after_h() {
        let vocab = ascii_char_vocab();
        let mut processor = JsonSchemaProcessor::new(&string_enum_schema(), vocab.clone()).unwrap();
        let dq = vocab.iter().position(|s| s == "\"").unwrap() as u32;
        let h = vocab.iter().position(|s| s == "h").unwrap() as u32;
        processor.step_token(dq).unwrap();
        processor.step_token(h).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("StringEnum InBody{partial='h'} must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // After 'h' only 'i' continues toward 'high'; closing '"' is
        // forbidden because 'h' isn't a complete value.
        let i = vocab.iter().position(|s| s == "i").unwrap();
        assert_eq!(gpu_mask.allow[i], 1);
        let dq_idx = vocab.iter().position(|s| s == "\"").unwrap();
        assert_eq!(gpu_mask.allow[dq_idx], 0);
    }

    #[test]
    fn string_enum_gpu_mask_matches_cpu_after_complete_value() {
        // After "low" the partial matches a complete value, so the
        // closing '"' becomes valid (and the only valid char).
        let vocab = ascii_char_vocab();
        let mut processor = JsonSchemaProcessor::new(&string_enum_schema(), vocab.clone()).unwrap();
        for s in ["\"", "l", "o", "w"] {
            let id = vocab.iter().position(|t| t == s).unwrap() as u32;
            processor.step_token(id).unwrap();
        }
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("StringEnum InBody{partial='low'} must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        let dq = vocab.iter().position(|s| s == "\"").unwrap();
        assert_eq!(
            gpu_mask.allow[dq], 1,
            "closing quote must be allowed when partial is a complete value"
        );
    }

    // -----------------------------------------------------------------
    // Stage 4 — Nullable
    // -----------------------------------------------------------------

    #[test]
    fn nullable_boolean_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor =
            JsonSchemaProcessor::new(&json!({"type": ["boolean", "null"]}), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Nullable(Boolean) Start must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // Sanity: 't', 'f', 'n' all allowed.
        for c in ['t', 'f', 'n'] {
            let i = vocab.iter().position(|s| s == &c.to_string()).unwrap();
            assert_eq!(
                gpu_mask.allow[i], 1,
                "char {c} must be allowed for Nullable(Boolean)"
            );
        }
        let a = vocab.iter().position(|s| s == "a").unwrap();
        assert_eq!(gpu_mask.allow[a], 0);
    }

    #[test]
    fn nullable_integer_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor =
            JsonSchemaProcessor::new(&json!({"type": ["integer", "null"]}), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Nullable(Integer) Start must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        let n = vocab.iter().position(|s| s == "n").unwrap();
        let one = vocab.iter().position(|s| s == "1").unwrap();
        let minus = vocab.iter().position(|s| s == "-").unwrap();
        assert_eq!(gpu_mask.allow[n], 1);
        assert_eq!(gpu_mask.allow[one], 1);
        assert_eq!(gpu_mask.allow[minus], 1);
    }

    #[test]
    fn nullable_string_gpu_mask_matches_cpu_at_start() {
        let vocab = ascii_char_vocab();
        let processor =
            JsonSchemaProcessor::new(&json!({"type": ["string", "null"]}), vocab.clone()).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Nullable(String) Start must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        let n = vocab.iter().position(|s| s == "n").unwrap();
        let dq = vocab.iter().position(|s| s == "\"").unwrap();
        assert_eq!(gpu_mask.allow[n], 1);
        assert_eq!(gpu_mask.allow[dq], 1);
    }

    /// After committing to the inner schema (here: emitting 't' for
    /// Nullable(Boolean)), the grammar transitions out of the
    /// `Nullable` wrapper into `Schema::Boolean`. Subsequent
    /// `compute_mask_gpu` must still produce parity, this time via the
    /// Boolean accessor, not the Nullable one.
    #[test]
    fn nullable_boolean_gpu_mask_matches_cpu_after_committing_to_inner() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": ["boolean", "null"]}), vocab.clone()).unwrap();
        let t_id = vocab.iter().position(|s| s == "t").unwrap() as u32;
        processor.step_token(t_id).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Boolean PartialTrue must be DFA-compilable post-commit");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
    }

    /// After committing to the null branch, the grammar is now in
    /// `Schema::Null` literal mode, hit by `null_emission_stage`.
    #[test]
    fn nullable_boolean_gpu_mask_matches_cpu_after_committing_to_null() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&json!({"type": ["boolean", "null"]}), vocab.clone()).unwrap();
        let n_id = vocab.iter().position(|s| s == "n").unwrap() as u32;
        processor.step_token(n_id).unwrap();
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Null Partial after commit must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
    }

    // -----------------------------------------------------------------
    // Stage 5 — Nested scalars (Integer / String inside Array)
    // -----------------------------------------------------------------

    /// Walk an Integer schema nested inside an Array element. After
    /// "[1" the grammar is at NumberDigits inside an Array frame, with
    /// parent_terminators = [',', ']']. Both terminators must be
    /// allowed; CPU-equal byte-for-byte on ASCII single-char vocab.
    #[test]
    fn nested_integer_in_array_after_digit() {
        let vocab = ascii_char_vocab();
        let mut processor = JsonSchemaProcessor::new(
            &json!({"type": "array", "items": {"type": "integer"}}),
            vocab.clone(),
        )
        .unwrap();
        let lb = vocab.iter().position(|s| s == "[").unwrap() as u32;
        let one = vocab.iter().position(|s| s == "1").unwrap() as u32;
        processor.step_token(lb).unwrap();
        processor.step_token(one).unwrap();

        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Integer in Array after digit must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);

        // Both ',' and ']' should be allowed — they're parent
        // terminators, and Integer at AfterDigits is a complete value.
        for c in [',', ']'] {
            let i = vocab.iter().position(|s| s == &c.to_string()).unwrap();
            assert_eq!(
                gpu_mask.allow[i], 1,
                "terminator {c} must be accepted by nested Integer DFA",
            );
        }
        // '5' is also valid (more digits before terminating).
        let five = vocab.iter().position(|s| s == "5").unwrap();
        assert_eq!(gpu_mask.allow[five], 1);
    }

    /// Walk a String schema nested inside an Array. Inside the body,
    /// the closing '"' completes the value; ',' / ']' are NOT valid
    /// directly because the closing quote must come first. After the
    /// closing quote the multi-frame DFA's "popped" sink kicks in.
    #[test]
    fn nested_string_in_array_after_open_quote() {
        let vocab = ascii_char_vocab();
        let mut processor = JsonSchemaProcessor::new(
            &json!({"type": "array", "items": {"type": "string"}}),
            vocab.clone(),
        )
        .unwrap();
        let lb = vocab.iter().position(|s| s == "[").unwrap() as u32;
        let dq = vocab.iter().position(|s| s == "\"").unwrap() as u32;
        processor.step_token(lb).unwrap();
        processor.step_token(dq).unwrap();

        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("String in Array InBody must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);

        // Closing '"' is allowed (will close the body, completing the
        // string); content chars (including ',' which is a literal
        // comma inside the string body, not a terminator) are also
        // allowed. The terminator-class transitions only fire at the
        // String DFA's complete_states (state 2 = closed-after-quote),
        // not mid-body.
        let dq_idx = vocab.iter().position(|s| s == "\"").unwrap();
        assert_eq!(gpu_mask.allow[dq_idx], 1);
        let a = vocab.iter().position(|s| s == "a").unwrap();
        assert_eq!(gpu_mask.allow[a], 1);
        let comma = vocab.iter().position(|s| s == ",").unwrap();
        assert_eq!(
            gpu_mask.allow[comma], 1,
            "comma is valid string-body content"
        );
        // The backslash is still rejected (escapes unsupported).
        let bs = vocab.iter().position(|s| s == "\\").unwrap();
        assert_eq!(gpu_mask.allow[bs], 0);
    }

    /// Walk a Boolean nested inside an Array. After "[t", we're at
    /// PartialTrue {remaining: "rue"}; nothing should change vs the
    /// top-level case at this state because the Boolean is mid-emission
    /// (not at a completion state, so no terminators apply yet).
    #[test]
    fn nested_boolean_in_array_after_t() {
        let vocab = ascii_char_vocab();
        let mut processor = JsonSchemaProcessor::new(
            &json!({"type": "array", "items": {"type": "boolean"}}),
            vocab.clone(),
        )
        .unwrap();
        let lb = vocab.iter().position(|s| s == "[").unwrap() as u32;
        let t = vocab.iter().position(|s| s == "t").unwrap() as u32;
        processor.step_token(lb).unwrap();
        processor.step_token(t).unwrap();

        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("Boolean in Array after 't' must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
    }

    // -----------------------------------------------------------------
    // Stage 5 — ObjectKey (prefix trie over unseen properties)
    // -----------------------------------------------------------------

    fn extraction_response_shaped_object() -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "value": {"type": "number"},
                "valor": {"type": "number"},
                "name": {"type": "string"},
                "note": {"type": "string"},
            },
            "required": ["name"]
        })
    }

    /// Right after '{"', the grammar is in ObjectKey with empty partial.
    /// Valid first chars: 'v' (toward "value" or "valor"), 'n' (toward
    /// "name" or "note"). Other letters reject.
    #[test]
    fn object_key_gpu_mask_matches_cpu_at_empty_partial() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&extraction_response_shaped_object(), vocab.clone()).unwrap();
        let lb = vocab.iter().position(|s| s == "{").unwrap() as u32;
        let dq = vocab.iter().position(|s| s == "\"").unwrap() as u32;
        processor.step_token(lb).unwrap();
        processor.step_token(dq).unwrap();

        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("ObjectKey at empty partial must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        // 'v' and 'n' lead to candidates; 'x' doesn't.
        for c in ['v', 'n'] {
            let i = vocab.iter().position(|s| s == &c.to_string()).unwrap();
            assert_eq!(gpu_mask.allow[i], 1);
        }
        let x = vocab.iter().position(|s| s == "x").unwrap();
        assert_eq!(gpu_mask.allow[x], 0);
        // Closing '"' is forbidden — empty partial isn't a complete key.
        let dq_idx = vocab.iter().position(|s| s == "\"").unwrap();
        assert_eq!(gpu_mask.allow[dq_idx], 0);
    }

    /// After '{"v', valid chars are those that extend toward "value"
    /// or "valor" — 'a' is the only one. Closing '"' still forbidden
    /// (partial isn't complete).
    #[test]
    fn object_key_gpu_mask_matches_cpu_after_v() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&extraction_response_shaped_object(), vocab.clone()).unwrap();
        for s in ["{", "\"", "v"] {
            let id = vocab.iter().position(|t| t == s).unwrap() as u32;
            processor.step_token(id).unwrap();
        }
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("ObjectKey at partial='v' must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        let a = vocab.iter().position(|s| s == "a").unwrap();
        assert_eq!(gpu_mask.allow[a], 1);
        let dq = vocab.iter().position(|s| s == "\"").unwrap();
        assert_eq!(gpu_mask.allow[dq], 0);
    }

    /// After '{"name', the partial matches the complete property name
    /// "name" — closing '"' becomes valid (and is the only valid char,
    /// since no other property starts with "name…").
    #[test]
    fn object_key_gpu_mask_matches_cpu_after_complete_name() {
        let vocab = ascii_char_vocab();
        let mut processor =
            JsonSchemaProcessor::new(&extraction_response_shaped_object(), vocab.clone()).unwrap();
        for s in ["{", "\"", "n", "a", "m", "e"] {
            let id = vocab.iter().position(|t| t == s).unwrap() as u32;
            processor.step_token(id).unwrap();
        }
        let cpu_mask = processor.compute_mask();
        let client = cuda_client();
        let packed = PackedVocab::pack(&vocab);
        let gpu_mask = compute_mask_gpu::<CudaRuntime>(&processor, &client, &packed)
            .expect("ObjectKey at partial='name' must be DFA-compilable");
        assert_eq!(cpu_mask.allow, gpu_mask.allow);
        let dq = vocab.iter().position(|s| s == "\"").unwrap();
        assert_eq!(
            gpu_mask.allow[dq], 1,
            "closing quote must be valid when partial == 'name'"
        );
    }

    // -----------------------------------------------------------------
    // Gate verification: still-unsupported schemas fall through.
    // -----------------------------------------------------------------

    /// Phase::ObjectFreshOpen / ObjectExpectKey / ObjectAfterValue /
    /// ObjectColon and Array structural phases still return `None`.
    /// The user-visible behaviour: GPU dispatch handles ObjectKey and
    /// nested scalars; the structural transitions stay on CPU
    /// (cheap there, kernel-launch overhead would exceed savings).
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
            "Object schema must return None; Object support is post-stage-4 work",
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
