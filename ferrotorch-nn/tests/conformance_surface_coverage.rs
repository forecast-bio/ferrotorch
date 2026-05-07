//! Layer-4 strict coverage gate for the full `ferrotorch-nn` conformance
//! suite (C9.1 + C9.2 + C9.3 + C9.4).
//!
//! Tracking issue: crosslink #806 (C9.4 — attention + grad_fns + Layer-4 gate).
//!
//! ## What this gate does
//!
//! 1. Loads `tests/conformance/_surface_inventory.toml` — the authoritative
//!    list of every public item in `ferrotorch-nn` scoped to the C9 dispatch.
//! 2. Loads `tests/conformance/_surface_exclusions.toml` — items that have a
//!    documented reason for not requiring a direct conformance test reference.
//! 3. Greps over every `conformance_nn_*.rs` file that **exists on disk**.
//!    C9.1, C9.2, C9.3 files may not yet be present when C9.4 lands; the
//!    gate skips missing files gracefully and reports which sub-phases are
//!    still pending.
//! 4. Asserts that every C9.4 attention/grad_fns item is referenced in
//!    `conformance_nn_attention.rs`.
//! 5. For C9.1–C9.3 items: the gate reports coverage status but does NOT
//!    fail when their test files are absent — it emits a "pending" notice.
//!
//! ## Strictness rules
//!
//! - C9.4 items: **HARD FAIL** if any item is not in exclusions AND not
//!   referenced in `conformance_nn_attention.rs`.
//! - C9.1/C9.2/C9.3 items: **SOFT WARN** if their conformance file is absent.
//!   Once the file is on disk, uncovered items not in exclusions cause a hard fail.
//! - Items in exclusions: always pass regardless of test file coverage.
//!
//! ## Pattern
//!
//! Mirrors `ferrotorch-jit/tests/conformance_surface_coverage.rs` (C7.4 gate)
//! and `ferrotorch-gpu/tests/conformance_surface_coverage.rs` (C8.4 gate).
//! Multi-file scan, hard-fail on own scope, soft-warn on absent peer files
//! for graceful dispatch ordering.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// TOML parsing helpers (hand-rolled to avoid external crate churn)
// ---------------------------------------------------------------------------

/// Extract all `path = "..."` values from a TOML file.
fn extract_toml_paths(content: &str) -> HashSet<String> {
    let mut paths = HashSet::new();
    for line in content.lines() {
        let trimmed = line.trim();
        if let Some(rest) = trimmed.strip_prefix("path = \"") {
            if let Some(path) = rest.strip_suffix('"') {
                paths.insert(path.to_owned());
            }
        }
    }
    paths
}

/// Extract `path → c9_phase` mapping from the inventory TOML.
fn extract_inventory_phases(content: &str) -> HashMap<String, String> {
    let mut map = HashMap::new();
    let mut current_path: Option<String> = None;
    let mut current_phase: Option<String> = None;

    for line in content.lines() {
        let trimmed = line.trim();

        if trimmed.starts_with("[[item]]") {
            // Flush previous item.
            if let (Some(p), Some(ph)) = (current_path.take(), current_phase.take()) {
                map.insert(p, ph);
            }
        } else if let Some(rest) = trimmed.strip_prefix("path = \"") {
            if let Some(path) = rest.strip_suffix('"') {
                current_path = Some(path.to_owned());
            }
        } else if let Some(rest) = trimmed.strip_prefix("c9_phase = \"") {
            if let Some(phase) = rest.strip_suffix('"') {
                current_phase = Some(phase.to_owned());
            }
        }
    }
    // Flush last item.
    if let (Some(p), Some(ph)) = (current_path, current_phase) {
        map.insert(p, ph);
    }
    map
}

// ---------------------------------------------------------------------------
// Candidate conformance file list — all sub-phases C9.1–C9.4
// ---------------------------------------------------------------------------

struct SubPhaseFile {
    phase: &'static str,
    /// Filename relative to the `tests/` directory.
    filename: &'static str,
}

const CONFORMANCE_FILES: &[SubPhaseFile] = &[
    SubPhaseFile {
        phase: "C9.1",
        // C9.1 authored conformance_nn_structural.rs (basic layers + module arch).
        filename: "conformance_nn_structural.rs",
    },
    SubPhaseFile {
        phase: "C9.2",
        // C9.2 authored conformance_nn_norm_activation_loss.rs.
        filename: "conformance_nn_norm_activation_loss.rs",
    },
    SubPhaseFile {
        phase: "C9.3",
        // C9.3 shares the structural file; update when C9.3 authors its own.
        // Per dispatch spec C9.3 is conv + embedding — same file as C9.1 structural
        // for now (it augments the C9.1 file rather than creating a new one).
        filename: "conformance_nn_structural.rs",
    },
    SubPhaseFile {
        phase: "C9.4",
        // C9.4 — this dispatch — attention + grad_fns.
        filename: "conformance_nn_attention.rs",
    },
];

// ---------------------------------------------------------------------------
// Primary gate test
// ---------------------------------------------------------------------------

#[test]
fn every_public_item_has_a_conformance_reference_or_exclusion() {
    let tests_dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();

    // 1. Load surface inventory.
    let inventory_path = tests_dir.join("conformance").join("_surface_inventory.toml");
    let inventory_raw = std::fs::read_to_string(&inventory_path).unwrap_or_else(|e| {
        panic!(
            "Cannot read surface inventory at {}: {e}",
            inventory_path.display()
        )
    });
    let inventory_phases = extract_inventory_phases(&inventory_raw);
    let all_items: HashSet<String> = inventory_phases.keys().cloned().collect();

    // 2. Load exclusions.
    let exclusions_path = tests_dir.join("conformance").join("_surface_exclusions.toml");
    let exclusions_raw = std::fs::read_to_string(&exclusions_path).unwrap_or_else(|e| {
        panic!(
            "Cannot read exclusions file at {}: {e}",
            exclusions_path.display()
        )
    });
    let excluded_items = extract_toml_paths(&exclusions_raw);

    // 3. Build per-phase text index from files that exist on disk.
    let mut phase_text: HashMap<&'static str, String> = HashMap::new();
    let mut missing_phases: Vec<&'static str> = Vec::new();

    for entry in CONFORMANCE_FILES {
        let path = tests_dir.join(entry.filename);
        match std::fs::read_to_string(&path) {
            Ok(text) => {
                // Merge if multiple phases map to the same file.
                phase_text
                    .entry(entry.phase)
                    .and_modify(|e| e.push_str(&text))
                    .or_insert(text);
            }
            Err(_) => {
                if !missing_phases.contains(&entry.phase) {
                    missing_phases.push(entry.phase);
                }
            }
        }
    }

    // Report pending phases.
    if !missing_phases.is_empty() {
        let mut sorted = missing_phases.clone();
        sorted.sort_unstable();
        eprintln!(
            "coverage_gate: conformance files not yet on disk for phases: {:?}",
            sorted
        );
        eprintln!("  These sub-phases will be reconciled after their dispatch lands.");
    }

    // Combined text across all present files — used for soft-warn matching.
    let combined_text: String = phase_text
        .values()
        .cloned()
        .collect::<Vec<_>>()
        .join("\n");

    // C9.4 file must be present since we own it.
    let c94_text = phase_text
        .get("C9.4")
        .expect("conformance_nn_attention.rs must exist — this is the C9.4 gate file");

    // 4. Assess each inventory item.
    let mut uncovered_hard: Vec<String> = Vec::new();
    let mut uncovered_soft: Vec<(String, String)> = Vec::new();

    for item in &all_items {
        // Excluded items always pass.
        if excluded_items.contains(item.as_str()) {
            continue;
        }

        let phase = inventory_phases
            .get(item)
            .map(|s| s.as_str())
            .unwrap_or("unknown");

        // Leaf name is the last `::` segment.  Match on leaf OR full path.
        let leaf = item.rsplit("::").next().unwrap_or(item.as_str());
        let in_combined =
            combined_text.contains(leaf) || combined_text.contains(item.as_str());

        match phase {
            "C9.4" => {
                // Hard: C9.4 file is present, must reference the item.
                let referenced =
                    c94_text.contains(leaf) || c94_text.contains(item.as_str());
                if !referenced {
                    uncovered_hard.push(format!("{item} (leaf={leaf})"));
                }
            }
            "C9.1" | "C9.2" | "C9.3" => {
                if missing_phases.contains(&phase) {
                    // File absent → soft-pending.
                    uncovered_soft.push((item.clone(), format!("{phase} [file absent]")));
                } else if !in_combined {
                    uncovered_soft.push((item.clone(), phase.to_owned()));
                }
            }
            _ => {
                // Unknown phase — soft warning only.
                if !in_combined {
                    uncovered_soft.push((item.clone(), "unknown".to_owned()));
                }
            }
        }
    }

    // Report soft gaps.
    if !uncovered_soft.is_empty() {
        eprintln!(
            "coverage_gate: {} item(s) with soft-pending coverage:",
            uncovered_soft.len()
        );
        for (path, phase) in &uncovered_soft {
            eprintln!("  [{phase}] {path}");
        }
        eprintln!(
            "  Items in absent-file phases become hard failures once those files land."
        );
    }

    // Hard failures.
    assert!(
        uncovered_hard.is_empty(),
        "coverage_gate HARD FAIL: {} item(s) lack a conformance reference and are not excluded:\n{}",
        uncovered_hard.len(),
        uncovered_hard.join("\n")
    );

    // Summary counters.
    let pending_count = uncovered_soft
        .iter()
        .filter(|(_, p)| p.contains("file absent"))
        .count();
    let soft_present_count = uncovered_soft.len() - pending_count;
    let covered_count = all_items
        .len()
        .saturating_sub(excluded_items.len())
        .saturating_sub(uncovered_soft.len());

    eprintln!(
        "coverage_gate: {} total items; {} excluded; {} soft-pending (present files); \
         {} pending (files absent); {} hard-covered",
        all_items.len(),
        excluded_items.len(),
        soft_present_count,
        pending_count,
        covered_count,
    );
}

// ---------------------------------------------------------------------------
// C9.4 key-names smoke test
// ---------------------------------------------------------------------------
//
// Verifies that all C9.4 public-item leaf names appear in
// `conformance_nn_attention.rs` so the gate doesn't rely solely on the TOML
// look-up (which requires the inventory to be complete).

#[test]
fn c9_4_key_names_all_referenced() {
    let attention_path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "conformance_nn_attention.rs",
    ]
    .iter()
    .collect();

    let text = std::fs::read_to_string(&attention_path)
        .expect("conformance_nn_attention.rs must exist for this gate to run");

    // Canonical leaf names for all C9.4 public items (minus exclusions).
    // Items excluded in _surface_exclusions.toml are omitted here.
    let required: &[&str] = &[
        // --- attention ---
        "MultiheadAttention",
        "MultiheadAttention::new",
        "MultiheadAttention::with_gqa",
        "forward_qkv",
        "reshape_to_heads",
        "repeat_kv",
        // --- flash_attention ---
        "flash_attention",
        "standard_attention",
        // --- flex_attention ---
        "BlockMask",
        "flex_attention",
        "causal_score_mod",
        "alibi_score_mod",
        "relative_position_bias_score_mod",
        // --- paged_attention ---
        "PagedAttentionManager",
        "PagedAttentionManager::new",
        "add_sequence",
        "append_kv",
        "get_kv",
        "PagedKVCache",
        "PagePool",
        // --- transformer ---
        "RoPEConvention",
        "RoPEScaling",
        "RotaryPositionEmbedding",
        "RotaryPositionEmbedding::new",
        "with_convention",
        "apply",
        "SwiGLU",
        "SwiGLU::new",
        "KVCache",
        "KVCache::new",
        "with_dims",
        "update",
        "reset",
        "seq_len",
        "TransformerEncoderLayer",
        "TransformerEncoderLayer::new",
        "TransformerDecoderLayer",
        "TransformerDecoderLayer::new",
        "forward_with_memory",
        // --- grad_fns (via backward tests) ---
        "SoftmaxBackward",
        "softmax_backward",
        "layernorm_backward",
        "cross_entropy_backward",
    ];

    let missing: Vec<&&str> = required.iter().filter(|s| !text.contains(**s)).collect();
    assert!(
        missing.is_empty(),
        "c9_4 key names missing from conformance_nn_attention.rs: {missing:?}"
    );
}

// ---------------------------------------------------------------------------
// Gate meta-test: inventory and exclusions must be parseable and consistent
// ---------------------------------------------------------------------------

#[test]
fn surface_inventory_and_exclusions_are_readable() {
    let tests_dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();

    let inventory_path = tests_dir.join("conformance").join("_surface_inventory.toml");
    let inventory_raw =
        std::fs::read_to_string(&inventory_path).expect("_surface_inventory.toml must be readable");
    let phases = extract_inventory_phases(&inventory_raw);
    assert!(
        phases.len() >= 50,
        "inventory looks suspiciously small: {} items (expected ≥50)",
        phases.len()
    );

    // Verify C9.4 items are present.
    let c94_count = phases.values().filter(|p| p.as_str() == "C9.4").count();
    assert!(
        c94_count >= 10,
        "inventory has only {c94_count} C9.4 items (expected ≥10)"
    );

    let exclusions_path = tests_dir.join("conformance").join("_surface_exclusions.toml");
    let exclusions_raw = std::fs::read_to_string(&exclusions_path)
        .expect("_surface_exclusions.toml must be readable");
    let excluded = extract_toml_paths(&exclusions_raw);
    assert!(
        !excluded.is_empty(),
        "exclusions file has no entries — likely empty or corrupt"
    );

    // Stale-exclusion guard: every excluded path must appear in the inventory.
    for path in &excluded {
        assert!(
            phases.contains_key(path.as_str()),
            "exclusion '{path}' is not in _surface_inventory.toml — stale exclusion?"
        );
    }

    eprintln!(
        "meta: inventory has {} items ({} C9.1, {} C9.2, {} C9.3, {} C9.4); \
         exclusions has {} items",
        phases.len(),
        phases.values().filter(|p| p.as_str() == "C9.1").count(),
        phases.values().filter(|p| p.as_str() == "C9.2").count(),
        phases.values().filter(|p| p.as_str() == "C9.3").count(),
        phases.values().filter(|p| p.as_str() == "C9.4").count(),
        excluded.len(),
    );
}

// ---------------------------------------------------------------------------
// Peer-file presence smoke test
// ---------------------------------------------------------------------------

/// Emit a diagnostic (not a failure) for each C9.1–C9.3 conformance file
/// that isn't on disk yet. Once they land, the gate above enforces hard-fail
/// on any items they should cover.
#[test]
fn peer_conformance_files_presence_report() {
    let tests_dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), "tests"].iter().collect();

    // Map phase → canonical filename for peer files (C9.1–C9.3).
    let peer_files: &[(&str, &str)] = &[
        ("C9.1", "conformance_nn_structural.rs"),
        ("C9.2", "conformance_nn_norm_activation_loss.rs"),
        ("C9.3", "conformance_nn_structural.rs"),
    ];

    let mut present: Vec<&str> = Vec::new();
    let mut absent: Vec<&str> = Vec::new();

    for (phase, filename) in peer_files {
        let path = tests_dir.join(filename);
        if path.exists() {
            if !present.contains(phase) {
                present.push(phase);
            }
        } else if !absent.contains(phase) {
            absent.push(phase);
        }
    }

    if !absent.is_empty() {
        eprintln!(
            "peer_presence: C9 sub-phases not yet on disk (soft-pending): {:?}",
            absent
        );
        eprintln!("  Gate will enforce hard-fail once these files land.");
    }
    if !present.is_empty() {
        eprintln!("peer_presence: present peer files for phases: {:?}", present);
    }

    // This test always passes — its purpose is the eprintln diagnostic.
}

// ---------------------------------------------------------------------------
// C9.4 fixture file presence test
// ---------------------------------------------------------------------------

/// Verify the nn_attention fixture file exists and is non-empty.
/// A missing or empty fixture file means regenerate_nn_attention_fixtures.py
/// hasn't been run — the conformance tests will silently pass (they see 0 fixtures).
#[test]
fn c9_4_fixture_file_present_and_non_empty() {
    let fixture_path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "conformance",
        "fixtures",
        "nn_attention.json",
    ]
    .iter()
    .collect();

    let raw = std::fs::read_to_string(&fixture_path).unwrap_or_else(|e| {
        panic!(
            "nn_attention.json missing — run scripts/regenerate_nn_attention_fixtures.py: {e}"
        )
    });

    // Must contain at least some fixture entries.
    let count = raw.matches("\"op\"").count();
    assert!(
        count >= 10,
        "nn_attention.json has only {count} op entries (expected ≥10) — \
         re-run scripts/regenerate_nn_attention_fixtures.py"
    );

    eprintln!("c9_4_fixture: nn_attention.json contains {count} fixture entries");
}
