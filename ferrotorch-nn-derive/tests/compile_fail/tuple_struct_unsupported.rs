//! Triggers: `src/lib.rs` line 109 — "only supports structs with named fields".
//!
//! The derive matches `Data::Struct(_)` but then requires `Fields::Named(_)`.
//! A tuple struct lands in the `_` arm of the inner match.
//! `fn main() {}` exists only to satisfy `trybuild`; the file is expected
//! to fail at the `#[derive(Module)]` step before main is ever reached.

use ferrotorch_nn_derive::Module;

#[derive(Module)]
struct TupleStruct<T>(T, bool);

fn main() {}
