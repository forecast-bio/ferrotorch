//! Triggers: `src/lib.rs` line 410 — "#[derive(Module)] requires at least one
//! type parameter (e.g., `T: Float`)".
//!
//! `find_float_param` walks `generics.params`; if no type parameter is
//! declared, no float type can be inferred and the derive errors out.

use ferrotorch_nn_derive::Module;

#[derive(Module)]
struct NoGenerics {
    training: bool,
}

fn main() {}
