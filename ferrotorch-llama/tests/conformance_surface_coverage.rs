//! Layer 4 — public surface coverage gate.
//!
//! Verifies that every item recorded in `_surface_inventory.toml` is either:
//!
//! a) covered by a fixture test in `conformance_llama.rs` (listed in the
//!    `COVERED` set below), **or**
//! b) explicitly excluded with a reason in `_surface_exclusions.toml`.
//!
//! The test fails if any inventory item appears in neither list, preventing
//! silent drift when new public items are added without a corresponding
//! conformance test or documented exclusion.
//!
//! # Design note
//!
//! The check is done purely in Rust at test-time: both TOML files are parsed
//! and the path sets are reconciled.  No code generation, no proc-macro — the
//! gate is a regular `#[test]` so it runs on every `cargo test` invocation.

use std::collections::HashSet;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// TOML schema
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct InventoryFile {
    item: Vec<InventoryItem>,
}

#[derive(Debug, Deserialize)]
struct InventoryItem {
    path: String,
}

#[derive(Debug, Deserialize)]
struct ExclusionsFile {
    excluded: Vec<ExcludedItem>,
}

#[derive(Debug, Deserialize)]
struct ExcludedItem {
    path: String,
}

// ---------------------------------------------------------------------------
// Items covered by conformance_llama.rs fixture tests (Layer 3)
// ---------------------------------------------------------------------------
//
// Each entry corresponds to the public API path exercised by at least one
// `conformance_*` test in `conformance_llama.rs`.

const COVERED: &[&str] = &[
    // RMSNorm (via ferrotorch_nn but exercised for the Llama norm contract)
    "ferrotorch_llama::LlamaConfig",
    "ferrotorch_llama::LlamaConfig::validate",
    // RMSNorm forward is exercised through all norm fixtures
    // Attention
    "ferrotorch_llama::LlamaAttention",
    "ferrotorch_llama::LlamaAttention::new",
    "ferrotorch_llama::LlamaAttention::num_heads",
    "ferrotorch_llama::LlamaAttention::num_kv_heads",
    "ferrotorch_llama::LlamaAttention::head_dim",
    // MLP
    "ferrotorch_llama::LlamaMLP",
    "ferrotorch_llama::LlamaMLP::new",
    // Decoder layer
    "ferrotorch_llama::LlamaDecoderLayer",
    "ferrotorch_llama::LlamaDecoderLayer::new",
    // Model
    "ferrotorch_llama::LlamaModel",
    "ferrotorch_llama::LlamaModel::new",
    // Causal LM
    "ferrotorch_llama::LlamaForCausalLM",
    "ferrotorch_llama::LlamaForCausalLM::new",
    "ferrotorch_llama::LlamaForCausalLM::forward_from_ids",
    "ferrotorch_llama::LlamaForCausalLM::load_hf_state_dict",
    // Config
    "ferrotorch_llama::LlamaActivation",
];

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

#[test]
fn surface_coverage_gate() {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");

    // Load inventory.
    let inv_path = std::path::PathBuf::from(manifest_dir)
        .join("tests/conformance/_surface_inventory.toml");
    let inv_text = std::fs::read_to_string(&inv_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", inv_path.display()));
    let inventory: InventoryFile = toml::from_str(&inv_text)
        .unwrap_or_else(|e| panic!("inventory TOML parse error: {e}"));

    // Load exclusions.
    let excl_path = std::path::PathBuf::from(manifest_dir)
        .join("tests/conformance/_surface_exclusions.toml");
    let excl_text = std::fs::read_to_string(&excl_path)
        .unwrap_or_else(|e| panic!("cannot read {}: {e}", excl_path.display()));
    let exclusions: ExclusionsFile = toml::from_str(&excl_text)
        .unwrap_or_else(|e| panic!("exclusions TOML parse error: {e}"));

    // Build lookup sets.
    let covered: HashSet<&str> = COVERED.iter().copied().collect();
    let excluded: HashSet<String> = exclusions
        .excluded
        .iter()
        .map(|e| e.path.clone())
        .collect();

    // Validate: every inventory item must be covered or excluded.
    let mut ungated: Vec<String> = Vec::new();
    for item in &inventory.item {
        let path = &item.path;
        if !covered.contains(path.as_str()) && !excluded.contains(path) {
            ungated.push(path.clone());
        }
    }

    // Validate the inverse: nothing in COVERED is missing from the inventory
    // (catches stale entries in the COVERED list that were renamed/removed).
    let inv_paths: HashSet<&str> = inventory.item.iter().map(|i| i.path.as_str()).collect();
    let mut phantom: Vec<&str> = Vec::new();
    for &path in COVERED {
        if !inv_paths.contains(path) {
            phantom.push(path);
        }
    }

    // Validate the inverse for exclusions: nothing in exclusions is missing
    // from the inventory (catches stale exclusions after API removals).
    let mut phantom_excl: Vec<String> = Vec::new();
    for excl in &exclusions.excluded {
        if !inv_paths.contains(excl.path.as_str()) {
            phantom_excl.push(excl.path.clone());
        }
    }

    // Report all problems together.
    let mut failures: Vec<String> = Vec::new();

    if !ungated.is_empty() {
        failures.push(format!(
            "UNGATED: {} public API item(s) appear in the inventory but are neither \
             covered by a conformance test nor explicitly excluded:\n  {}",
            ungated.len(),
            ungated.join("\n  ")
        ));
    }

    if !phantom.is_empty() {
        failures.push(format!(
            "PHANTOM (COVERED): {} item(s) listed in COVERED[] are not in the \
             inventory — stale after a rename or removal:\n  {}",
            phantom.len(),
            phantom.join("\n  ")
        ));
    }

    if !phantom_excl.is_empty() {
        failures.push(format!(
            "PHANTOM (exclusions): {} item(s) in _surface_exclusions.toml are not \
             in the inventory — stale after a rename or removal:\n  {}",
            phantom_excl.len(),
            phantom_excl.join("\n  ")
        ));
    }

    assert!(
        failures.is_empty(),
        "surface_coverage_gate FAILED:\n\n{}",
        failures.join("\n\n")
    );
}
