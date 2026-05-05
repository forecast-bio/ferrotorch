//! `trybuild` harness for `#[derive(Module)]`'s error paths.
//!
//! Each `tests/compile_fail/*.rs` fixture is a minimal reproduction that
//! triggers exactly one `syn::Error` site in `src/lib.rs`. The matching
//! `*.stderr` snapshot pins the rendered `compile_error!` text so any
//! regression in the error message or its span pointer fails the test.
//!
//! To regenerate snapshots after an intentional message change:
//!   `TRYBUILD=overwrite cargo test -p ferrotorch-nn-derive`

#[test]
fn compile_fail() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
