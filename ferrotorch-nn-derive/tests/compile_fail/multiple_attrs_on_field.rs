//! Triggers: `src/lib.rs` line 152 — "field cannot have more than one of
//! #[param], #[submodule], #[skip]".
//!
//! Each field may carry at most one of the three classification attributes;
//! `attr_count > 1` is rejected.

use ferrotorch_nn_derive::Module;

#[derive(Module)]
struct DoubleAttr<T> {
    #[param]
    #[submodule]
    weight: T,
    training: bool,
}

fn main() {}
