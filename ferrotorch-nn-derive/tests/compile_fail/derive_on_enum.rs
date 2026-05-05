//! Triggers: `src/lib.rs` line 116 — "#[derive(Module)] only supports structs".
//!
//! The derive accepts only `Data::Struct`. Applying it to an `enum` (or `union`)
//! short-circuits before any field classification.

use ferrotorch_nn_derive::Module;

#[derive(Module)]
enum NotAStruct<T> {
    Variant(T),
}

fn main() {}
