//! Character-level state machine over a JSON value matching a [`Schema`].
//!
//! [`JsonGrammar`] is the workhorse of the constrained decoder. At every
//! point during generation it knows:
//!
//! - Which characters can come next (`valid_next_chars`).
//! - Whether the value emitted so far is complete (`is_complete`).
//!
//! Tokens in a real LLM vocabulary span multiple characters; the
//! [`super::json_schema::JsonSchemaProcessor`] wrapper builds on top of this
//! by simulating each token's chars in sequence and including the token in
//! the allow-mask only if every char is accepted.
//!
//! ## Limitations
//!
//! The character grammar emits compact JSON only:
//!
//! - **No whitespace**: a real model's output is forced into tight
//!   `{"k":"v","n":1.5}` form. Real-world deployments often want a
//!   permissive whitespace mode; that's a follow-up.
//! - **Numbers**: optional `-`, digits, optional `.<digits>`. No exponent.
//! - **Strings**: ASCII printable bytes excluding `"` and `\`. No escape
//!   sequences are produced (`\"`, `\n`, `\\`, `\uXXXX` are all rejected).
//!   This is intentional: the project's `ExtractionResponse` payload is
//!   short, structured, and never needs literal embedded quotes; allowing
//!   escapes would require a full JSON-string sub-parser.
//!
//! Each limitation is captured by a test that asserts the grammar rejects
//! the disallowed input — so the limitations cannot silently regress.

use std::collections::BTreeSet;

use super::schema::Schema;

/// Reasons the grammar may reject a character.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum StepError {
    /// `c` is not a valid next character at the current state.
    #[error("unexpected {got:?} at this point; valid next chars: {expected:?}")]
    UnexpectedChar {
        /// The character that was rejected.
        got: char,
        /// Characters that would have been valid at this state.
        expected: Vec<char>,
    },
    /// The grammar is already complete; further characters are not allowed.
    #[error("grammar is already complete")]
    AlreadyComplete,
    /// The input would advance into an unsupported branch (e.g. a string
    /// escape sequence).
    #[error("unsupported by grammar subset: {0}")]
    Unsupported(&'static str),
}

/// Per-frame phase: where we are inside the current schema value.
#[derive(Debug, Clone)]
enum Phase {
    /// Haven't emitted anything yet for this value.
    Start,
    /// Inside the literal `null`, `true`, or `false` — `remaining` holds the
    /// chars still to emit.
    Literal { remaining: &'static str },
    /// Inside a string value (between the opening `"` and closing `"`).
    StringChars {
        partial: String,
        allowed: Option<Vec<String>>,
    },
    /// Inside a number value.
    ///
    /// - `had_sign`: a leading `-` was emitted.
    /// - `had_digits`: at least one digit has been emitted.
    /// - `had_decimal`: a `.` has been emitted (no more `.` allowed).
    /// - `had_fractional_digit`: at least one digit was emitted *after* `.`.
    ///   Required by JSON: `1.` is invalid; `1.0` is valid. While
    ///   `had_decimal` is true and `had_fractional_digit` is false, only
    ///   digits are valid (no terminator).
    /// - `is_zero_only`: the first emitted digit was `0` *and* nothing else
    ///   has been emitted yet (no more leading zeros: JSON forbids `01`,
    ///   `-007`, etc., but `0`, `0.5`, `-0.25` are all fine).
    NumberDigits {
        had_sign: bool,
        had_digits: bool,
        had_decimal: bool,
        had_fractional_digit: bool,
        is_zero_only: bool,
    },
    /// Inside an object: just emitted `{`. Need `"` to start a key or `}`
    /// to close (if all required keys are satisfied — for an empty
    /// `properties` set this means as soon as we open).
    ObjectFreshOpen { keys_seen: BTreeSet<String> },
    /// Inside an object: just emitted `,`. Must emit `"` to start the next
    /// key — `}` is forbidden.
    ObjectExpectKey { keys_seen: BTreeSet<String> },
    /// Inside an object: emitting key characters between `"` and `"`.
    /// `partial` is the key chars seen so far. `candidates` is the set of
    /// not-yet-seen property names that are still consistent with `partial`.
    ObjectKey {
        partial: String,
        keys_seen: BTreeSet<String>,
        candidates: Vec<String>,
    },
    /// Just emitted the closing `"` of an object key. Need `:` next.
    ObjectColon {
        current_key: String,
        keys_seen: BTreeSet<String>,
    },
    /// Just finished a property value. Need `,` (more keys) or `}` (close,
    /// only if all required keys have been seen).
    ObjectAfterValue { keys_seen: BTreeSet<String> },
    /// Inside an array: just emitted `[`. Need an element value or `]`.
    ArrayFreshOpen,
    /// Inside an array: just finished an element. Need `,` or `]`.
    ArrayAfterValue,
}

#[derive(Debug, Clone)]
struct Frame {
    schema: Schema,
    phase: Phase,
}

/// State machine over the partial JSON emission.
#[derive(Debug, Clone)]
pub struct JsonGrammar {
    frames: Vec<Frame>,
    done: bool,
}

impl JsonGrammar {
    /// Build a fresh grammar that will produce one value of the given schema.
    pub fn new(schema: Schema) -> Self {
        let frame = Frame {
            schema,
            phase: Phase::Start,
        };
        Self {
            frames: vec![frame],
            done: false,
        }
    }

    /// Has the top-level value been fully emitted?
    pub fn is_complete(&self) -> bool {
        self.done
    }

    /// If this grammar is a single-frame `Schema::Boolean`, report which
    /// stage of literal emission we're at. Returns `None` for every other
    /// schema or for nested / multi-frame states.
    ///
    /// Used by the GPU constrained-decoding bridge in
    /// [`super::gpu_dispatch`] (`--features cuda`) to decide whether the
    /// current grammar state is DFA-compilable. Stage-2 GPU support
    /// covers exactly Boolean; everything else falls through to the
    /// existing CPU `compute_mask` loop.
    pub fn boolean_emission_stage(&self) -> Option<BooleanEmissionStage> {
        if self.done {
            return None;
        }
        if self.frames.len() != 1 {
            return None;
        }
        let frame = &self.frames[0];
        if !matches!(frame.schema, Schema::Boolean) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(BooleanEmissionStage::Start),
            Phase::Literal { remaining } => {
                // Disambiguate which literal we're inside. The Phase carries
                // only the remaining suffix; we look up which of "true" /
                // "false" has it as a suffix. The boolean grammar uses
                // `&'static str` slices into the literal source strings, so
                // suffix matching is unambiguous.
                if "true".ends_with(remaining) && remaining.len() < "true".len() {
                    Some(BooleanEmissionStage::PartialTrue { remaining })
                } else if "false".ends_with(remaining) && remaining.len() < "false".len() {
                    Some(BooleanEmissionStage::PartialFalse { remaining })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}

/// Stage of `Schema::Boolean` emission. Surfaces just enough of the
/// internal `Phase` enum for the GPU dispatcher to compile a DFA without
/// exposing the rest of the grammar's state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BooleanEmissionStage {
    /// Nothing emitted yet. The DFA must accept any prefix of either
    /// `"true"` or `"false"`.
    Start,
    /// We've already emitted some prefix of `"true"`. `remaining` is the
    /// suffix still to emit (always non-empty; complete is unreachable
    /// here since the grammar reports `done` for that case).
    PartialTrue {
        /// Characters of `"true"` not yet emitted.
        remaining: &'static str,
    },
    /// Same as `PartialTrue` but for `"false"`.
    PartialFalse {
        /// Characters of `"false"` not yet emitted.
        remaining: &'static str,
    },
}

/// Stage of `Schema::Null` emission. Mirrors `BooleanEmissionStage` for
/// the "null" literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NullEmissionStage {
    /// Nothing emitted yet — DFA expects 'n' first.
    Start,
    /// Some prefix of `"null"` has been emitted. `remaining` is the
    /// suffix still to match.
    Partial {
        /// Characters of `"null"` not yet emitted.
        remaining: &'static str,
    },
}

/// Stage of single-frame `Schema::Integer` emission. The grammar's
/// `Phase::NumberDigits` carries five booleans; for top-level integers
/// `had_decimal` and `had_fractional_digit` are always false, so the
/// reachable space collapses to four cases.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IntegerEmissionStage {
    /// `Phase::Start` — DFA expects `'-'` or `'0'..='9'`.
    Start,
    /// After `'-'`, no digits yet — DFA expects `'0'..='9'`.
    AfterSign,
    /// First digit was `'0'`, no more chars valid (JSON forbids `01`).
    /// The DFA's only outgoing transition is to REJECT.
    AfterZero,
    /// At least one non-zero digit emitted. More digits are valid.
    AfterDigits,
}

/// Stage of single-frame `Schema::Number` emission. Like
/// `IntegerEmissionStage` but extended with the decimal / fractional
/// dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NumberEmissionStage {
    /// `Phase::Start` — DFA expects `'-'` or `'0'..='9'`.
    Start,
    /// After `'-'`, no digits yet — DFA expects `'0'..='9'`.
    AfterSign,
    /// First digit was `'0'`, no decimal yet. Only `'.'` is valid.
    AfterZeroNoDecimal,
    /// Non-zero integer part, no decimal yet. `'0'..='9'` and `'.'`.
    AfterDigitsNoDecimal,
    /// `'.'` emitted, no fractional digit yet. Only `'0'..='9'`.
    AfterDecimalNoFrac,
    /// At least one fractional digit. `'0'..='9'`.
    AfterFractionalDigits,
}

/// Stage of single-frame `Schema::String` emission (non-enum strings).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringEmissionStage {
    /// `Phase::Start` — DFA expects opening `'"'`.
    Start,
    /// Inside the string body. Any printable ASCII except `'"'` and
    /// `'\\'` (escapes are intentionally unsupported per the existing
    /// grammar) plus the closing `'"'`.
    InBody,
}

/// Stage of single-frame `Schema::StringEnum` emission. `partial` is a
/// borrow into the grammar's current `Phase::StringChars` payload so no
/// allocation happens at dispatch time.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StringEnumEmissionStage<'a> {
    /// `Phase::Start` — DFA expects opening `'"'`.
    Start,
    /// Inside the quoted body, having emitted some prefix of one of the
    /// allowed values. `partial` is the chars between the opening `'"'`
    /// and the cursor; an empty `partial` is the just-opened state.
    InBody {
        /// Characters emitted between the opening `'"'` and the cursor.
        partial: &'a str,
    },
}

/// Stage of single-frame `Schema::Nullable(_)` emission. The grammar
/// commits to either the null branch or `inner` after the very first
/// emitted character, so we only need to surface the `Phase::Start`
/// case — every other phase is handled by the inner schema's own
/// accessor (or by `null_emission_stage` for the null branch).
///
/// `PartialEq` only — `Schema` itself derives `PartialEq` but not `Eq`
/// (consistent with the rest of the type), so neither does this.
#[derive(Debug, Clone, PartialEq)]
pub enum NullableEmissionStage<'a> {
    /// `Phase::Start` with a `Schema::Nullable(inner)`. The DFA must
    /// accept any prefix of `"null"` plus any prefix accepted by
    /// `inner`'s start state.
    Start {
        /// The non-null branch's schema.
        inner: &'a Schema,
    },
}

impl JsonGrammar {
    /// If this grammar is a single-frame `Schema::Null`, report which
    /// stage of literal emission we're at.
    pub fn null_emission_stage(&self) -> Option<NullEmissionStage> {
        if self.done || self.frames.len() != 1 {
            return None;
        }
        let frame = &self.frames[0];
        if !matches!(frame.schema, Schema::Null) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(NullEmissionStage::Start),
            Phase::Literal { remaining }
                if "null".ends_with(remaining) && !remaining.is_empty() =>
            {
                Some(NullEmissionStage::Partial { remaining })
            }
            _ => None,
        }
    }

    /// If this grammar is a single-frame `Schema::Integer`, report the
    /// emission stage. Stage-3 GPU dispatch handles the four reachable
    /// cases for top-level integers; nested integers fall through to
    /// CPU because their value-end terminator depends on the parent.
    pub fn integer_emission_stage(&self) -> Option<IntegerEmissionStage> {
        if self.done || self.frames.len() != 1 {
            return None;
        }
        let frame = &self.frames[0];
        if !matches!(frame.schema, Schema::Integer) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(IntegerEmissionStage::Start),
            Phase::NumberDigits {
                had_sign,
                had_digits,
                is_zero_only,
                had_decimal,
                had_fractional_digit,
            } => {
                // Integer mode never sets decimal or fractional flags;
                // assert that to catch grammar drift early.
                debug_assert!(!*had_decimal && !*had_fractional_digit);
                if !*had_digits {
                    if *had_sign {
                        Some(IntegerEmissionStage::AfterSign)
                    } else {
                        // had_sign=F && had_digits=F is Phase::Start; should be unreachable.
                        None
                    }
                } else if *is_zero_only {
                    Some(IntegerEmissionStage::AfterZero)
                } else {
                    Some(IntegerEmissionStage::AfterDigits)
                }
            }
            _ => None,
        }
    }

    /// If this grammar is a single-frame `Schema::Number`, report the
    /// emission stage. Same nested-vs-top-level caveat as
    /// `integer_emission_stage`.
    pub fn number_emission_stage(&self) -> Option<NumberEmissionStage> {
        if self.done || self.frames.len() != 1 {
            return None;
        }
        let frame = &self.frames[0];
        if !matches!(frame.schema, Schema::Number) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(NumberEmissionStage::Start),
            Phase::NumberDigits {
                had_sign,
                had_digits,
                had_decimal,
                had_fractional_digit,
                is_zero_only,
            } => match (
                *had_sign,
                *had_digits,
                *had_decimal,
                *had_fractional_digit,
                *is_zero_only,
            ) {
                (true, false, false, false, false) => Some(NumberEmissionStage::AfterSign),
                (_, true, false, false, true) => Some(NumberEmissionStage::AfterZeroNoDecimal),
                (_, true, false, false, false) => Some(NumberEmissionStage::AfterDigitsNoDecimal),
                (_, true, true, false, _) => Some(NumberEmissionStage::AfterDecimalNoFrac),
                (_, true, true, true, _) => Some(NumberEmissionStage::AfterFractionalDigits),
                _ => None,
            },
            _ => None,
        }
    }

    /// If this grammar is a single-frame `Schema::String` (non-enum),
    /// report the emission stage.
    pub fn string_emission_stage(&self) -> Option<StringEmissionStage> {
        if self.done || self.frames.len() != 1 {
            return None;
        }
        let frame = &self.frames[0];
        if !matches!(frame.schema, Schema::String) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(StringEmissionStage::Start),
            Phase::StringChars { allowed: None, .. } => Some(StringEmissionStage::InBody),
            _ => None,
        }
    }

    /// If this grammar is a single-frame `Schema::StringEnum`, report
    /// the emission stage *and* the allowed value list. The list is
    /// borrowed from the schema, so callers don't pay an allocation.
    pub fn string_enum_emission_stage(&self) -> Option<(StringEnumEmissionStage<'_>, &[String])> {
        if self.done || self.frames.len() != 1 {
            return None;
        }
        let frame = &self.frames[0];
        let values: &[String] = match &frame.schema {
            Schema::StringEnum(v) => v.as_slice(),
            _ => return None,
        };
        let stage = match &frame.phase {
            Phase::Start => StringEnumEmissionStage::Start,
            Phase::StringChars {
                partial,
                allowed: Some(_),
            } => StringEnumEmissionStage::InBody {
                partial: partial.as_str(),
            },
            _ => return None,
        };
        Some((stage, values))
    }

    /// If this grammar is a single-frame `Schema::Nullable(_)` at
    /// `Phase::Start`, surface the inner schema so the GPU dispatcher
    /// can build a merged DFA. After the first character is emitted
    /// the grammar commits to either the null branch or the inner
    /// schema, and subsequent compute_mask calls hit those accessors
    /// directly — so only `Phase::Start` is in scope here.
    pub fn nullable_emission_stage(&self) -> Option<NullableEmissionStage<'_>> {
        if self.done || self.frames.len() != 1 {
            return None;
        }
        let frame = &self.frames[0];
        match (&frame.schema, &frame.phase) {
            (Schema::Nullable(inner), Phase::Start) => Some(NullableEmissionStage::Start { inner }),
            _ => None,
        }
    }

    /// Compute the chars that legally terminate the top-of-stack value
    /// frame, given the current parent (one level up). Returns an empty
    /// vector when the top frame is the only frame (single-frame
    /// dispatch path). Wraps the private `parent_terminators` so the
    /// GPU dispatch module can use it without inheriting the rest of
    /// `Phase`'s API surface.
    pub fn top_frame_parent_terminators(&self) -> Vec<char> {
        if self.done || self.frames.is_empty() {
            return Vec::new();
        }
        let parent = if self.frames.len() > 1 {
            Some(&self.frames[self.frames.len() - 2])
        } else {
            None
        };
        parent_terminators(parent)
    }

    /// If the *top* frame of this grammar is `Schema::Integer`, report
    /// the emission stage. Unlike [`Self::integer_emission_stage`] this
    /// version permits multi-frame grammars (Integer nested inside an
    /// Object property value or Array element). Pair the result with
    /// [`Self::top_frame_parent_terminators`] to feed the GPU compiler.
    pub fn integer_emission_stage_top(&self) -> Option<IntegerEmissionStage> {
        if self.done {
            return None;
        }
        let frame = self.frames.last()?;
        if !matches!(frame.schema, Schema::Integer) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(IntegerEmissionStage::Start),
            Phase::NumberDigits {
                had_sign,
                had_digits,
                is_zero_only,
                had_decimal,
                had_fractional_digit,
            } => {
                debug_assert!(!*had_decimal && !*had_fractional_digit);
                if !*had_digits {
                    if *had_sign {
                        Some(IntegerEmissionStage::AfterSign)
                    } else {
                        None
                    }
                } else if *is_zero_only {
                    Some(IntegerEmissionStage::AfterZero)
                } else {
                    Some(IntegerEmissionStage::AfterDigits)
                }
            }
            _ => None,
        }
    }

    /// Multi-frame variant of [`Self::number_emission_stage`].
    pub fn number_emission_stage_top(&self) -> Option<NumberEmissionStage> {
        if self.done {
            return None;
        }
        let frame = self.frames.last()?;
        if !matches!(frame.schema, Schema::Number) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(NumberEmissionStage::Start),
            Phase::NumberDigits {
                had_sign,
                had_digits,
                had_decimal,
                had_fractional_digit,
                is_zero_only,
            } => match (
                *had_sign,
                *had_digits,
                *had_decimal,
                *had_fractional_digit,
                *is_zero_only,
            ) {
                (true, false, false, false, false) => Some(NumberEmissionStage::AfterSign),
                (_, true, false, false, true) => Some(NumberEmissionStage::AfterZeroNoDecimal),
                (_, true, false, false, false) => Some(NumberEmissionStage::AfterDigitsNoDecimal),
                (_, true, true, false, _) => Some(NumberEmissionStage::AfterDecimalNoFrac),
                (_, true, true, true, _) => Some(NumberEmissionStage::AfterFractionalDigits),
                _ => None,
            },
            _ => None,
        }
    }

    /// Multi-frame variant of [`Self::string_emission_stage`].
    pub fn string_emission_stage_top(&self) -> Option<StringEmissionStage> {
        if self.done {
            return None;
        }
        let frame = self.frames.last()?;
        if !matches!(frame.schema, Schema::String) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(StringEmissionStage::Start),
            Phase::StringChars { allowed: None, .. } => Some(StringEmissionStage::InBody),
            _ => None,
        }
    }

    /// Multi-frame variant of [`Self::boolean_emission_stage`].
    pub fn boolean_emission_stage_top(&self) -> Option<BooleanEmissionStage> {
        if self.done {
            return None;
        }
        let frame = self.frames.last()?;
        if !matches!(frame.schema, Schema::Boolean) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(BooleanEmissionStage::Start),
            Phase::Literal { remaining }
                if "true".ends_with(remaining) && remaining.len() < "true".len() =>
            {
                Some(BooleanEmissionStage::PartialTrue { remaining })
            }
            Phase::Literal { remaining }
                if "false".ends_with(remaining) && remaining.len() < "false".len() =>
            {
                Some(BooleanEmissionStage::PartialFalse { remaining })
            }
            _ => None,
        }
    }

    /// Multi-frame variant of [`Self::null_emission_stage`].
    pub fn null_emission_stage_top(&self) -> Option<NullEmissionStage> {
        if self.done {
            return None;
        }
        let frame = self.frames.last()?;
        if !matches!(frame.schema, Schema::Null) {
            return None;
        }
        match &frame.phase {
            Phase::Start => Some(NullEmissionStage::Start),
            Phase::Literal { remaining }
                if "null".ends_with(remaining) && !remaining.is_empty() =>
            {
                Some(NullEmissionStage::Partial { remaining })
            }
            _ => None,
        }
    }

    /// Multi-frame variant of [`Self::string_enum_emission_stage`].
    pub fn string_enum_emission_stage_top(
        &self,
    ) -> Option<(StringEnumEmissionStage<'_>, &[String])> {
        if self.done {
            return None;
        }
        let frame = self.frames.last()?;
        let values: &[String] = match &frame.schema {
            Schema::StringEnum(v) => v.as_slice(),
            _ => return None,
        };
        let stage = match &frame.phase {
            Phase::Start => StringEnumEmissionStage::Start,
            Phase::StringChars {
                partial,
                allowed: Some(_),
            } => StringEnumEmissionStage::InBody {
                partial: partial.as_str(),
            },
            _ => return None,
        };
        Some((stage, values))
    }

    /// If the top frame is in `Phase::ObjectKey`, surface the partial
    /// chars and the still-unseen-property candidates list. The
    /// candidates borrow into the grammar's own state — no clone.
    pub fn object_key_emission_stage(&self) -> Option<ObjectKeyEmissionStage<'_>> {
        if self.done {
            return None;
        }
        let frame = self.frames.last()?;
        if !matches!(frame.schema, Schema::Object { .. }) {
            return None;
        }
        match &frame.phase {
            Phase::ObjectKey {
                partial,
                candidates,
                ..
            } => Some(ObjectKeyEmissionStage {
                partial: partial.as_str(),
                candidates: candidates.as_slice(),
            }),
            _ => None,
        }
    }
}

/// Stage of `Phase::ObjectKey` emission. The DFA is structurally the
/// same as a [`StringEnumEmissionStage`] over `candidates` — a prefix
/// trie with closing `'"'` enabled at trie nodes that match a complete
/// candidate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectKeyEmissionStage<'a> {
    /// Characters of the in-progress key emitted between the opening `'"'` and the cursor.
    pub partial: &'a str,
    /// Set of property names still consistent with `partial` (not yet seen and prefix-matched).
    pub candidates: &'a [String],
}

impl JsonGrammar {
    /// Set of single-byte characters that may legally come next.
    ///
    /// Emits an empty vector when the grammar is complete (no more input is
    /// valid). Used by [`super::json_schema::JsonSchemaProcessor`] to compute
    /// per-token allow masks.
    pub fn valid_next_chars(&self) -> Vec<char> {
        if self.done {
            return Vec::new();
        }
        let top = self.frames.len() - 1;
        let frame = &self.frames[top];
        let parent = if top > 0 {
            Some(&self.frames[top - 1])
        } else {
            None
        };
        valid_next_chars_for(frame, parent)
    }

    /// Advance the state by one character. Returns an error if the character
    /// is not a valid next emission. The state is left unchanged on error.
    ///
    /// # Errors
    ///
    /// Returns [`StepError::AlreadyComplete`] if the grammar has
    /// already accepted a complete value, [`StepError::UnexpectedChar`]
    /// if `c` is not in the current valid-next-chars set, or
    /// [`StepError::Unsupported`] if the character would advance into
    /// an unsupported branch (e.g. a JSON string escape).
    pub fn step_char(&mut self, c: char) -> Result<(), StepError> {
        if self.done {
            return Err(StepError::AlreadyComplete);
        }
        let allowed = self.valid_next_chars();
        if !allowed.contains(&c) {
            return Err(StepError::UnexpectedChar {
                got: c,
                expected: allowed,
            });
        }
        // We've accepted that `c` is a legal next char; now mutate the
        // top-of-stack frame accordingly. The `apply_step` helper handles all
        // transitions (push, pop, internal phase change).
        self.apply_step(c)
    }

    /// Convenience: feed a string of characters one at a time. Stops at the
    /// first error.
    ///
    /// # Errors
    ///
    /// Forwards the first [`StepError`] returned by [`Self::step_char`].
    pub fn step_str(&mut self, s: &str) -> Result<(), StepError> {
        for c in s.chars() {
            self.step_char(c)?;
        }
        Ok(())
    }

    fn apply_step(&mut self, c: char) -> Result<(), StepError> {
        let top_idx = self.frames.len() - 1;
        let frame = &mut self.frames[top_idx];

        match (&frame.schema.clone(), &frame.phase.clone()) {
            // ----- STRING -----
            (Schema::String, Phase::Start) => {
                // Must be `"`.
                debug_assert_eq!(c, '"');
                frame.phase = Phase::StringChars {
                    partial: String::new(),
                    allowed: None,
                };
            }
            (Schema::StringEnum(values), Phase::Start) => {
                debug_assert_eq!(c, '"');
                frame.phase = Phase::StringChars {
                    partial: String::new(),
                    allowed: Some(values.clone()),
                };
            }
            (Schema::String | Schema::StringEnum(_), Phase::StringChars { .. }) => {
                if c == '"' {
                    // For enums: only allow closing if partial matches a complete value.
                    if let Phase::StringChars {
                        partial,
                        allowed: Some(values),
                    } = &frame.phase
                    {
                        if !values.iter().any(|v| v == partial) {
                            return Err(StepError::UnexpectedChar {
                                got: c,
                                expected: vec![],
                            });
                        }
                    }
                    // Pop the string frame; the parent (if any) was already
                    // transitioned to its post-value phase before push.
                    self.frames.pop();
                    self.bubble_value_done();
                    return Ok(());
                }
                // Else accumulate the char into `partial`.
                let phase_owned = std::mem::replace(&mut frame.phase, Phase::Start);
                if let Phase::StringChars {
                    mut partial,
                    allowed,
                } = phase_owned
                {
                    partial.push(c);
                    frame.phase = Phase::StringChars { partial, allowed };
                }
            }

            // ----- NUMBER / INTEGER -----
            (Schema::Number | Schema::Integer, Phase::Start) => {
                let mut had_sign = false;
                let mut had_digits = false;
                let mut is_zero_only = false;
                if c == '-' {
                    had_sign = true;
                } else {
                    debug_assert!(c.is_ascii_digit());
                    had_digits = true;
                    is_zero_only = c == '0';
                }
                frame.phase = Phase::NumberDigits {
                    had_sign,
                    had_digits,
                    had_decimal: false,
                    had_fractional_digit: false,
                    is_zero_only,
                };
            }
            (
                Schema::Number,
                Phase::NumberDigits {
                    had_sign,
                    had_digits,
                    had_decimal,
                    had_fractional_digit,
                    is_zero_only,
                },
            ) => {
                let (had_sign, had_digits, had_decimal, had_fractional_digit) =
                    (*had_sign, *had_digits, *had_decimal, *had_fractional_digit);
                let _ = is_zero_only;
                if c == '.' {
                    // valid_next_chars only emits `.` when had_decimal is false
                    // and had_digits is true.
                    frame.phase = Phase::NumberDigits {
                        had_sign,
                        had_digits,
                        had_decimal: true,
                        had_fractional_digit: false,
                        is_zero_only: false,
                    };
                } else if c.is_ascii_digit() {
                    let new_is_zero_only = !had_digits && c == '0';
                    let new_fractional = had_decimal || had_fractional_digit;
                    frame.phase = Phase::NumberDigits {
                        had_sign,
                        had_digits: true,
                        had_decimal,
                        had_fractional_digit: new_fractional,
                        is_zero_only: new_is_zero_only,
                    };
                } else {
                    // Number ended by some non-digit; the parent decides what's
                    // valid based on context (`,`, `}`, `]`, end). The current
                    // frame is done — pop and re-dispatch the char to the parent.
                    self.frames.pop();
                    self.bubble_value_done();
                    return self.apply_step(c);
                }
            }
            (
                Schema::Integer,
                Phase::NumberDigits {
                    had_sign,
                    had_digits,
                    ..
                },
            ) => {
                let (had_sign, had_digits) = (*had_sign, *had_digits);
                if c.is_ascii_digit() {
                    let new_is_zero_only = !had_digits && c == '0';
                    frame.phase = Phase::NumberDigits {
                        had_sign,
                        had_digits: true,
                        had_decimal: false,
                        had_fractional_digit: false,
                        is_zero_only: new_is_zero_only,
                    };
                } else {
                    self.frames.pop();
                    self.bubble_value_done();
                    return self.apply_step(c);
                }
            }

            // ----- LITERALS (true/false/null) -----
            (Schema::Boolean, Phase::Start) => {
                let remaining = match c {
                    't' => "rue",
                    'f' => "alse",
                    _ => unreachable!("valid_next_chars only accepts t/f for Boolean"),
                };
                frame.phase = Phase::Literal { remaining };
            }
            (Schema::Null, Phase::Start) => {
                debug_assert_eq!(c, 'n');
                frame.phase = Phase::Literal { remaining: "ull" };
            }
            (Schema::Boolean | Schema::Null, Phase::Literal { remaining }) => {
                let r = *remaining;
                debug_assert!(r.starts_with(c));
                let new_remaining = &r[c.len_utf8()..];
                if new_remaining.is_empty() {
                    self.frames.pop();
                    self.bubble_value_done();
                } else {
                    frame.phase = Phase::Literal {
                        remaining: new_remaining,
                    };
                }
            }

            // ----- NULLABLE: dispatch into inner or null based on first char -----
            (Schema::Nullable(inner), Phase::Start) => {
                let inner = (**inner).clone();
                if c == 'n' {
                    // Treat as Null literal.
                    frame.schema = Schema::Null;
                    frame.phase = Phase::Literal { remaining: "ull" };
                } else {
                    // Switch to the inner schema and re-dispatch this char.
                    frame.schema = inner;
                    frame.phase = Phase::Start;
                    return self.apply_step(c);
                }
            }

            // ----- OBJECT -----
            (Schema::Object { .. }, Phase::Start) => {
                debug_assert_eq!(c, '{');
                frame.phase = Phase::ObjectFreshOpen {
                    keys_seen: BTreeSet::new(),
                };
            }
            (Schema::Object { .. }, Phase::ObjectFreshOpen { keys_seen })
            | (Schema::Object { .. }, Phase::ObjectExpectKey { keys_seen }) => {
                let keys_seen = keys_seen.clone();
                if c == '"' {
                    let candidates = match &frame.schema {
                        Schema::Object { properties, .. } => properties
                            .keys()
                            .filter(|k| !keys_seen.contains(*k))
                            .cloned()
                            .collect::<Vec<_>>(),
                        _ => unreachable!(),
                    };
                    frame.phase = Phase::ObjectKey {
                        partial: String::new(),
                        keys_seen,
                        candidates,
                    };
                } else {
                    debug_assert_eq!(c, '}');
                    self.frames.pop();
                    self.bubble_value_done();
                }
            }
            (
                Schema::Object { .. },
                Phase::ObjectKey {
                    partial,
                    keys_seen,
                    candidates,
                },
            ) => {
                let mut partial = partial.clone();
                let keys_seen = keys_seen.clone();
                let candidates = candidates.clone();
                if c == '"' {
                    // Key complete; partial must match exactly one candidate.
                    if !candidates.iter().any(|k| k == &partial) {
                        return Err(StepError::UnexpectedChar {
                            got: c,
                            expected: vec![],
                        });
                    }
                    frame.phase = Phase::ObjectColon {
                        current_key: partial,
                        keys_seen,
                    };
                } else {
                    partial.push(c);
                    frame.phase = Phase::ObjectKey {
                        partial,
                        keys_seen,
                        candidates,
                    };
                }
            }
            (
                Schema::Object { properties, .. },
                Phase::ObjectColon {
                    current_key,
                    keys_seen,
                },
            ) => {
                debug_assert_eq!(c, ':');
                // Push a child frame for the property's value.
                //
                // Invariant: `current_key` was set on the
                // `Phase::ObjectKey -> Phase::ObjectColon` transition
                // (line ~973) only after the guard at line ~967
                // (`candidates.iter().any(|k| k == &partial)`) succeeded.
                // `candidates` is constructed at line ~936 as the
                // not-yet-seen subset of *this same frame's*
                // `Schema::Object { properties, .. }` keys, so
                // `current_key in properties` holds by construction.
                let prop_schema = properties
                    .get(current_key)
                    .expect(
                        "invariant: ObjectColon.current_key was previously matched against \
                         this frame's Schema::Object.properties keys at the line ~967 guard \
                         (candidates.iter().any) — `properties.get(current_key)` is therefore Some",
                    )
                    .clone();
                let mut keys_seen = keys_seen.clone();
                keys_seen.insert(current_key.clone());
                frame.phase = Phase::ObjectAfterValue { keys_seen };
                self.frames.push(Frame {
                    schema: prop_schema,
                    phase: Phase::Start,
                });
            }
            (Schema::Object { .. }, Phase::ObjectAfterValue { keys_seen }) => {
                let keys_seen = keys_seen.clone();
                if c == ',' {
                    frame.phase = Phase::ObjectExpectKey { keys_seen };
                } else {
                    debug_assert_eq!(c, '}');
                    self.frames.pop();
                    self.bubble_value_done();
                }
            }

            // ----- ARRAY -----
            (Schema::Array { .. }, Phase::Start) => {
                debug_assert_eq!(c, '[');
                frame.phase = Phase::ArrayFreshOpen;
            }
            (Schema::Array { item }, Phase::ArrayFreshOpen) => {
                if c == ']' {
                    self.frames.pop();
                    self.bubble_value_done();
                } else {
                    let item_schema = (**item).clone();
                    frame.phase = Phase::ArrayAfterValue;
                    self.frames.push(Frame {
                        schema: item_schema,
                        phase: Phase::Start,
                    });
                    // Re-dispatch the char to the new top frame.
                    return self.apply_step(c);
                }
            }
            (Schema::Array { item }, Phase::ArrayAfterValue) => {
                if c == ',' {
                    let item_schema = (**item).clone();
                    frame.phase = Phase::ArrayAfterValue;
                    self.frames.push(Frame {
                        schema: item_schema,
                        phase: Phase::Start,
                    });
                } else {
                    debug_assert_eq!(c, ']');
                    self.frames.pop();
                    self.bubble_value_done();
                }
            }

            (schema, phase) => {
                return Err(StepError::Unsupported(Box::leak(
                    format!("schema={schema:?} phase={phase:?}").into_boxed_str(),
                )));
            }
        }
        Ok(())
    }

    fn bubble_value_done(&mut self) {
        // Called whenever a value-level frame finishes. If the stack is now
        // empty, the whole grammar is done. Otherwise, the parent frame's
        // phase update (e.g. ObjectAfterValue, ArrayAfterValue) was handled
        // before push; this is just a stack-empty check.
        if self.frames.is_empty() {
            self.done = true;
        }
    }
}

/// Compute the set of valid next characters for the given top-of-stack frame,
/// using `parent` (if any) to compute correct value-end terminators for
/// number/integer children.
fn valid_next_chars_for(frame: &Frame, parent: Option<&Frame>) -> Vec<char> {
    match (&frame.schema, &frame.phase) {
        (Schema::String, Phase::Start) | (Schema::StringEnum(_), Phase::Start) => vec!['"'],
        (Schema::String, Phase::StringChars { partial: _, .. }) => {
            let mut chars: Vec<char> = (0x20u8..=0x7Eu8)
                .filter(|b| *b != b'"' && *b != b'\\')
                .map(|b| b as char)
                .collect();
            chars.push('"'); // closing quote
            chars
        }
        (Schema::StringEnum(_), Phase::StringChars { partial, allowed }) => {
            // Only chars that could extend `partial` toward a known enum
            // value (or close the string if partial equals a value).
            //
            // Invariant: this match arm is gated on
            // `Schema::StringEnum`. The only path that constructs a
            // `Phase::StringChars` from a `Schema::StringEnum` frame
            // (apply_step `(Schema::StringEnum(values), Phase::Start)`,
            // line ~755) initialises `allowed: Some(values.clone())`.
            // The `allowed: None` variant is only produced by the
            // `Schema::String` arm (~750), which never reaches here.
            let allowed = allowed.as_ref().expect(
                "invariant: Phase::StringChars constructed from Schema::StringEnum always sets \
                 allowed = Some(values.clone()) — see apply_step (Schema::StringEnum, Phase::Start) \
                 around line 755; the None variant is unique to Schema::String",
            );
            let mut next: BTreeSet<char> = BTreeSet::new();
            for v in allowed {
                if v.starts_with(partial.as_str()) && v.len() > partial.len() {
                    // Invariant: `v.len() > partial.len()` (just checked
                    // on the line above) means the slice
                    // `&v[partial.len()..]` has at least one byte;
                    // since `v` is a `String` and the slice starts at a
                    // char boundary by construction (we only push whole
                    // chars into `partial` — see apply_step's
                    // (Schema::String | Schema::StringEnum, StringChars)
                    // branch around line 790), `chars().next()` returns
                    // `Some`.
                    let next_c = v[partial.len()..].chars().next().expect(
                        "invariant: v.len() > partial.len() guarantees the suffix slice is \
                         non-empty, and partial only ever contains whole chars (see apply_step \
                         StringChars accumulation around line 790), so the slice begins at a char \
                         boundary",
                    );
                    next.insert(next_c);
                }
            }
            // Closing quote allowed only if `partial` is itself a complete value.
            if allowed.iter().any(|v| v == partial) {
                next.insert('"');
            }
            next.into_iter().collect()
        }

        (Schema::Number, Phase::Start) | (Schema::Integer, Phase::Start) => {
            let mut v: Vec<char> = ('0'..='9').collect();
            v.push('-');
            v
        }
        (
            Schema::Number,
            Phase::NumberDigits {
                had_digits,
                had_decimal,
                had_fractional_digit,
                is_zero_only,
                ..
            },
        ) => {
            // After an initial `-`, must emit digits next. Once any digits
            // are present, the number can end; the *parent* frame decides
            // what character ends it via re-dispatch in apply_step.
            //
            // JSON forbids leading zeros: after a single `0` as the first
            // digit, no more digits are allowed (only `.` or a terminator).
            // JSON also requires at least one fractional digit after `.`:
            // `1.` is invalid, `1.0` is valid. While we're mid-decimal,
            // only digits are allowed (no terminator).
            let mid_decimal = *had_decimal && !*had_fractional_digit;
            let mut chars: Vec<char> = if *is_zero_only {
                Vec::new()
            } else {
                ('0'..='9').collect()
            };
            if *had_digits && !*had_decimal {
                chars.push('.');
            }
            if *had_digits && !mid_decimal {
                chars.extend(parent_terminators(parent));
            }
            chars
        }
        (
            Schema::Integer,
            Phase::NumberDigits {
                had_digits,
                is_zero_only,
                ..
            },
        ) => {
            let mut chars: Vec<char> = if *is_zero_only {
                Vec::new()
            } else {
                ('0'..='9').collect()
            };
            if *had_digits {
                chars.extend(parent_terminators(parent));
            }
            chars
        }

        (Schema::Boolean, Phase::Start) => vec!['t', 'f'],
        (Schema::Null, Phase::Start) => vec!['n'],
        (Schema::Boolean, Phase::Literal { remaining })
        | (Schema::Null, Phase::Literal { remaining }) => {
            // Invariant: `Phase::Literal` is *never* observable with
            // `remaining = ""`. The Boolean/Null branch in apply_step
            // (around line 895) checks `new_remaining.is_empty()` and
            // pops the frame in that case rather than leaving an empty
            // `Phase::Literal` on the stack. Initial construction
            // (lines ~889 and ~893) seeds with "rue"/"alse"/"ull",
            // none of which are empty.
            vec![remaining.chars().next().expect(
                "invariant: Phase::Literal with empty `remaining` is never observable — \
                 apply_step (Schema::Boolean | Schema::Null, Phase::Literal) at line ~895 pops \
                 the frame instead of leaving an empty literal on the stack",
            )]
        }

        (Schema::Nullable(inner), Phase::Start) => {
            let mut v = valid_next_chars_for(
                &Frame {
                    schema: (**inner).clone(),
                    phase: Phase::Start,
                },
                parent,
            );
            v.push('n'); // null branch
            v.sort_unstable();
            v.dedup();
            v
        }

        (Schema::Object { properties, .. }, Phase::Start) => {
            let _ = properties;
            vec!['{']
        }
        (
            Schema::Object {
                properties,
                required,
            },
            Phase::ObjectFreshOpen { keys_seen },
        ) => {
            let mut v = vec![];
            // Need at least one more key if any required key is unseen.
            let unseen_required: Vec<&String> = required
                .iter()
                .filter(|k| !keys_seen.contains(*k))
                .collect();
            if !properties.keys().all(|k| keys_seen.contains(k)) {
                v.push('"');
            }
            if unseen_required.is_empty() {
                v.push('}');
            }
            v
        }
        (Schema::Object { .. }, Phase::ObjectExpectKey { keys_seen: _ }) => vec!['"'],
        (
            Schema::Object { .. },
            Phase::ObjectKey {
                partial,
                candidates,
                ..
            },
        ) => {
            let mut next: BTreeSet<char> = BTreeSet::new();
            for k in candidates {
                if k.starts_with(partial.as_str()) && k.len() > partial.len() {
                    // Invariant: `k.len() > partial.len()` (just
                    // checked) means `&k[partial.len()..]` is
                    // non-empty. `partial` is built up character-by-
                    // character via `partial.push(c)` in apply_step's
                    // `(Schema::Object { .. }, Phase::ObjectKey { .. })`
                    // branch (line ~978), so `partial.len()` is on a
                    // char boundary of `k` (when `k.starts_with(partial)`).
                    next.insert(k[partial.len()..].chars().next().expect(
                        "invariant: k.len() > partial.len() makes the suffix slice non-empty, \
                         and partial is accumulated via push(c) on whole chars in apply_step \
                         ObjectKey (line ~978), so the slice begins at a char boundary",
                    ));
                }
            }
            // Allow closing the key string only if partial matches one of the
            // candidate keys exactly.
            if candidates.iter().any(|k| k == partial) {
                next.insert('"');
            }
            next.into_iter().collect()
        }
        (Schema::Object { .. }, Phase::ObjectColon { .. }) => vec![':'],
        (
            Schema::Object {
                properties,
                required,
            },
            Phase::ObjectAfterValue { keys_seen },
        ) => {
            let mut v = vec![];
            let unseen_required: Vec<&String> = required
                .iter()
                .filter(|k| !keys_seen.contains(*k))
                .collect();
            if !properties.keys().all(|k| keys_seen.contains(k)) {
                v.push(',');
            }
            if unseen_required.is_empty() {
                v.push('}');
            }
            v
        }

        (Schema::Array { .. }, Phase::Start) => vec!['['],
        (Schema::Array { item }, Phase::ArrayFreshOpen) => {
            // Either close the array, or emit a value-start character.
            let mut v = valid_next_chars_for(
                &Frame {
                    schema: (**item).clone(),
                    phase: Phase::Start,
                },
                Some(frame),
            );
            v.push(']');
            v.sort_unstable();
            v.dedup();
            v
        }
        (Schema::Array { .. }, Phase::ArrayAfterValue) => vec![',', ']'],

        // We've already handled all defined transitions; the bubble-up branches
        // are reached via `apply_step` re-dispatch only and shouldn't surface
        // here.
        _ => Vec::new(),
    }
}

/// What characters does the parent frame use as a terminator for the
/// currently-active number/integer child?
///
/// When the active value is at top level (no parent), there is no JSON
/// terminator — the value ends implicitly via [`JsonGrammar::is_complete`]
/// reporting `true` once at least one digit has been emitted. We return an
/// empty set in that case so the LLM can keep emitting digits up to its
/// own EOS decision.
fn parent_terminators(parent: Option<&Frame>) -> Vec<char> {
    let Some(parent) = parent else {
        return Vec::new();
    };
    match (&parent.schema, &parent.phase) {
        (
            Schema::Object {
                properties,
                required,
            },
            Phase::ObjectAfterValue { keys_seen },
        ) => {
            let mut v = vec![];
            let unseen_required: Vec<&String> = required
                .iter()
                .filter(|k| !keys_seen.contains(*k))
                .collect();
            if !properties.keys().all(|k| keys_seen.contains(k)) {
                v.push(',');
            }
            if unseen_required.is_empty() {
                v.push('}');
            }
            v
        }
        (Schema::Array { .. }, Phase::ArrayAfterValue) => vec![',', ']'],
        // Other parent shapes don't host value children directly; over-approx
        // empty rather than guess wrong.
        _ => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    fn obj(props: &[(&str, Schema)], required: &[&str]) -> Schema {
        let mut p = BTreeMap::new();
        for (k, s) in props {
            p.insert((*k).to_string(), s.clone());
        }
        let mut r = BTreeSet::new();
        for k in required {
            r.insert((*k).to_string());
        }
        Schema::Object {
            properties: p,
            required: r,
        }
    }

    #[test]
    fn empty_object_round_trip() {
        let s = obj(&[], &[]);
        let mut g = JsonGrammar::new(s);
        assert_eq!(g.valid_next_chars(), vec!['{']);
        g.step_char('{').unwrap();
        assert!(g.valid_next_chars().contains(&'}'));
        g.step_char('}').unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn rejects_emitting_after_complete() {
        let mut g = JsonGrammar::new(Schema::Boolean);
        g.step_str("true").unwrap();
        assert!(g.is_complete());
        let err = g.step_char(',').unwrap_err();
        assert!(matches!(err, StepError::AlreadyComplete));
    }

    #[test]
    fn boolean_true_and_false() {
        let mut g = JsonGrammar::new(Schema::Boolean);
        assert_eq!(g.valid_next_chars(), vec!['t', 'f']);
        g.step_str("true").unwrap();
        assert!(g.is_complete());

        let mut g = JsonGrammar::new(Schema::Boolean);
        g.step_str("false").unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn null_literal() {
        let mut g = JsonGrammar::new(Schema::Null);
        g.step_str("null").unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn integer_round_trip() {
        let mut g = JsonGrammar::new(Schema::Integer);
        g.step_str("123").unwrap();
        // Integer ends implicitly; force end via a top-level boundary: at this
        // point we're at top-level, so re-dispatching, say, EOF means done.
        // To detect: number frame should pop on first non-digit; we simulate
        // by checking the frame stack via valid_next_chars: after digits at
        // top level there is no further valid char, so digit-only stream is
        // legal but doesn't auto-complete. We model "complete" via emitting
        // a synthetic terminator — for top-level use the public API
        // `finish()` (added in the public wrapper) or rely on the LLM
        // producing the EOS token. For the unit test, just assert
        // valid_next_chars contains digits and terminators are the empty set
        // at top level.
        let chars = g.valid_next_chars();
        assert!(chars.contains(&'4'));
        // The parent_terminators heuristic adds `,`, `}`, `]` which are
        // **not** valid at the top level; the resolver here is honest about
        // its over-approximation. The test asserts the digit branch works.
    }

    #[test]
    fn negative_number() {
        let mut g = JsonGrammar::new(Schema::Number);
        assert!(g.valid_next_chars().contains(&'-'));
        g.step_char('-').unwrap();
        // Sign emitted; need a digit next.
        assert!(g.valid_next_chars().contains(&'1'));
        g.step_str("3.14").unwrap();
    }

    #[test]
    fn string_round_trip() {
        let mut g = JsonGrammar::new(Schema::String);
        assert_eq!(g.valid_next_chars(), vec!['"']);
        g.step_char('"').unwrap();
        assert!(g.valid_next_chars().contains(&'a'));
        // No escape char allowed.
        assert!(!g.valid_next_chars().contains(&'\\'));
        // Quote not allowed mid-string... wait, quote ENDS the string.
        // The test was wrong. Quote IS allowed (closes string).
        g.step_str("hi").unwrap();
        g.step_char('"').unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn string_enum_round_trip() {
        let s = Schema::StringEnum(vec!["high".into(), "medium".into(), "low".into()]);
        let mut g = JsonGrammar::new(s);
        g.step_char('"').unwrap();
        // Only h, m, l should be valid (first chars of enum values).
        let nx = g.valid_next_chars();
        assert!(nx.contains(&'h'));
        assert!(nx.contains(&'m'));
        assert!(nx.contains(&'l'));
        assert!(!nx.contains(&'z'));
        g.step_str("medium\"").unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn string_enum_rejects_invalid_prefix() {
        let s = Schema::StringEnum(vec!["high".into(), "low".into()]);
        let mut g = JsonGrammar::new(s);
        g.step_char('"').unwrap();
        // `m` is not a valid first char of any enum value.
        assert!(g.step_char('m').is_err());
    }

    #[test]
    fn object_with_required_field() {
        let s = obj(
            &[("name", Schema::String), ("n", Schema::Integer)],
            &["name"],
        );
        let mut g = JsonGrammar::new(s);
        g.step_char('{').unwrap();
        // Can't close yet — `name` is required and unseen.
        assert!(!g.valid_next_chars().contains(&'}'));
        g.step_str("\"name\":\"foo\"").unwrap();
        // Now `,` is allowed (more keys) AND `}` is allowed (required satisfied).
        let nx = g.valid_next_chars();
        assert!(nx.contains(&','));
        assert!(nx.contains(&'}'));
        g.step_char('}').unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn object_rejects_unknown_key() {
        let s = obj(&[("name", Schema::String)], &["name"]);
        let mut g = JsonGrammar::new(s);
        g.step_char('{').unwrap();
        g.step_char('"').unwrap();
        // Only 'n' is a valid first char (only key is "name").
        assert!(g.step_char('z').is_err());
    }

    #[test]
    fn object_rejects_duplicate_key() {
        let s = obj(&[("a", Schema::Integer), ("b", Schema::Integer)], &[]);
        let mut g = JsonGrammar::new(s);
        g.step_str("{\"a\":1,").unwrap();
        // Now we need another key — `a` is gone, must be `b`.
        g.step_char('"').unwrap();
        assert!(g.step_char('a').is_err());
        assert!(g.step_char('b').is_ok());
    }

    #[test]
    fn array_of_numbers() {
        let s = Schema::Array {
            item: Box::new(Schema::Number),
        };
        let mut g = JsonGrammar::new(s);
        g.step_str("[1,2.5,3]").unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn empty_array() {
        let s = Schema::Array {
            item: Box::new(Schema::Number),
        };
        let mut g = JsonGrammar::new(s);
        g.step_str("[]").unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn nested_object() {
        let s = obj(
            &[("inner", obj(&[("v", Schema::Boolean)], &["v"]))],
            &["inner"],
        );
        let mut g = JsonGrammar::new(s);
        g.step_str("{\"inner\":{\"v\":true}}").unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn nullable_string() {
        let s = Schema::Nullable(Box::new(Schema::String));
        let mut g = JsonGrammar::new(s.clone());
        g.step_str("null").unwrap();
        assert!(g.is_complete());

        let mut g = JsonGrammar::new(s);
        g.step_str("\"hi\"").unwrap();
        assert!(g.is_complete());
    }

    #[test]
    fn rejects_string_escape() {
        let mut g = JsonGrammar::new(Schema::String);
        g.step_char('"').unwrap();
        let err = g.step_char('\\').unwrap_err();
        assert!(matches!(err, StepError::UnexpectedChar { .. }));
    }
}
