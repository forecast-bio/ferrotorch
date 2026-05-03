//! Pins every module path the umbrella crate's README documents.
//!
//! The umbrella crate's only job is re-exports. The README enumerates the
//! submodules a user can `use ferrotorch::*;` import. If a feature flag or
//! sub-crate dep is removed without updating the README, this test fails at
//! compile time instead of at the user's first build.

// Each test asserts a module path resolves; the `use … as _;` pattern is the
// minimum-friction way to do that without forcing every sub-crate to keep a
// specific named symbol stable. The unused-import warning is intrinsic to the
// pattern, not a code smell.
#![allow(unused_imports)]

#[test]
fn always_on_modules_resolve() {
    use ferrotorch::data as _;
    use ferrotorch::nn as _;
    use ferrotorch::optim as _;
    use ferrotorch::vision as _;
}

#[cfg(feature = "train")]
#[test]
fn train_module_resolves() {
    use ferrotorch::train as _;
}

#[cfg(feature = "serialize")]
#[test]
fn serialize_module_resolves() {
    use ferrotorch::serialize as _;
}

#[cfg(feature = "jit")]
#[test]
fn jit_module_resolves() {
    use ferrotorch::jit as _;
}

#[cfg(feature = "jit-script")]
#[test]
fn jit_script_module_resolves() {
    use ferrotorch::jit_script as _;
}

#[cfg(feature = "distributions")]
#[test]
fn distributions_module_resolves() {
    use ferrotorch::distributions as _;
}

#[cfg(feature = "profiler")]
#[test]
fn profiler_module_resolves() {
    use ferrotorch::profiler as _;
}

#[cfg(feature = "hub")]
#[test]
fn hub_module_resolves() {
    use ferrotorch::hub as _;
}

#[cfg(feature = "tokenize")]
#[test]
fn tokenize_module_resolves() {
    use ferrotorch::tokenize as _;
}

#[cfg(feature = "gpu")]
#[test]
fn gpu_module_resolves() {
    use ferrotorch::gpu as _;
}

#[cfg(feature = "cubecl")]
#[test]
fn cubecl_module_resolves() {
    use ferrotorch::cubecl as _;
}

#[cfg(feature = "mps")]
#[test]
fn mps_module_resolves() {
    use ferrotorch::mps as _;
}

#[cfg(feature = "xpu")]
#[test]
fn xpu_module_resolves() {
    use ferrotorch::xpu as _;
}

#[cfg(feature = "distributed")]
#[test]
fn distributed_module_resolves() {
    use ferrotorch::distributed as _;
}

#[cfg(feature = "llama")]
#[test]
fn llama_module_resolves() {
    use ferrotorch::llama as _;
}

#[cfg(feature = "ml")]
#[test]
fn ml_module_resolves() {
    use ferrotorch::ml as _;
}
