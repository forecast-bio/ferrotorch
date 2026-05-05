//! Triggers: `src/lib.rs` line 176 — "#[derive(Module)] requires a `training:
//! bool` field".
//!
//! The derive auto-generates `train()`/`eval()`/`is_training()` from a
//! `training: bool` field. A struct without one cannot have those methods
//! generated and is rejected at the call site span.

use ferrotorch_nn_derive::Module;

#[derive(Module)]
struct NoTraining<T> {
    #[param]
    weight: T,
}

fn main() {}
