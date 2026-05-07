//! Conformance Layer 4 — strict per-item coverage gate for `ferrotorch-distributed`.
//!
//! Tracking issue: #882 (ferrotorch-distributed conformance suite).
//!
//! Loads `tests/conformance/_surface_inventory.toml` and scans every
//! `tests/conformance_*.rs` file (other than this one) for references to
//! each public item. Fails the build if any item is neither (a) referenced
//! in a conformance test, nor (b) explicitly excluded in
//! `tests/conformance/_surface_exclusions.toml` with a `reason` **and** a
//! `tracking_issue` field.
//!
//! Mirrors the pattern in `ferrotorch-mps/tests/conformance_surface_coverage.rs`
//! and `ferrotorch-train/tests/conformance_surface_coverage.rs`.

#![allow(clippy::uninlined_format_args)]

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Surface inventory types (TOML — matches ferrotorch-distributed Layer 1)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SurfaceInventory {
    item: Vec<SurfaceItem>,
}

#[derive(Debug, Deserialize)]
struct SurfaceItem {
    path: String,
    kind: String,
    #[allow(dead_code, reason = "deserialized for forward-compat with future reporting")]
    signature: String,
}

// ---------------------------------------------------------------------------
// Exclusions types (same TOML shape as ferrotorch-core / ferrotorch-mps / ferrotorch-train)
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ExclusionsFile {
    #[serde(default, rename = "exclusion")]
    exclusions: Vec<Exclusion>,
}

#[derive(Debug, Deserialize)]
struct Exclusion {
    path: String,
    reason: String,
    /// Tracking issue ref. Required: an exclusion without a follow-up issue
    /// is "indefinite deferral" and the gate rejects it.
    tracking_issue: String,
}

// ---------------------------------------------------------------------------
// Path helpers
// ---------------------------------------------------------------------------

fn conformance_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
}

fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

// ---------------------------------------------------------------------------
// Loaders
// ---------------------------------------------------------------------------

fn read_inventory() -> SurfaceInventory {
    let p = conformance_dir().join("_surface_inventory.toml");
    let body =
        fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {} failed: {e}", p.display()));
    toml::from_str(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn read_exclusions() -> Vec<Exclusion> {
    let p = conformance_dir().join("_surface_exclusions.toml");
    if !p.exists() {
        return Vec::new();
    }
    let body =
        fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let parsed: ExclusionsFile =
        toml::from_str(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()));
    parsed.exclusions
}

/// Validate that a tracking_issue field looks like a real issue reference.
/// Accepts `#NNN` or a full https:// URL. Rejects "TBD", "pending", empty, etc.
fn tracking_issue_valid(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    // Accept #NNN format
    if s.starts_with('#') && s[1..].chars().all(|c| c.is_ascii_digit()) && s.len() > 1 {
        return true;
    }
    // Accept https:// URLs
    if s.starts_with("https://") {
        return true;
    }
    false
}

/// Generate coverage keys for an item path. Returns the shortest unique
/// suffix that a test can reference without necessarily spelling out the
/// full qualified path. Mirrors the ferrotorch-train gate heuristic.
fn coverage_keys(path: &str) -> Vec<String> {
    let mut keys = vec![path.to_string()];
    // Also accept the last component (e.g. "allreduce" from
    // "ferrotorch_distributed::allreduce"), but only if it's a simple
    // identifier without `::`-nesting that could collide.
    if let Some(last) = path.rsplit("::").next() {
        if !last.contains('<') {
            keys.push(last.to_string());
        }
    }
    keys
}

/// Read every `tests/conformance_*.rs` except this gate; return concatenated source.
fn read_conformance_test_sources() -> String {
    let mut combined = String::new();
    let root = tests_dir();
    let entries =
        fs::read_dir(&root).unwrap_or_else(|e| panic!("read tests dir {}: {e}", root.display()));
    for entry in entries {
        let entry = entry.expect("readdir entry");
        let name = entry.file_name();
        let name_str = name.to_string_lossy();
        if name_str.starts_with("conformance_")
            && name_str.ends_with(".rs")
            && name_str != "conformance_surface_coverage.rs"
        {
            let path = entry.path();
            let src = fs::read_to_string(&path)
                .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
            combined.push_str(&src);
            combined.push('\n');
        }
    }
    combined
}

// ---------------------------------------------------------------------------
// Gate test
// ---------------------------------------------------------------------------

#[test]
fn every_public_item_has_a_conformance_reference() {
    let inventory = read_inventory();
    let exclusions = read_exclusions();

    // Validate exclusion entries are well-formed.
    let mut bad_entries: Vec<String> = Vec::new();
    for e in &exclusions {
        if !tracking_issue_valid(&e.tracking_issue) {
            bad_entries.push(format!(
                "{} — invalid `tracking_issue` field: {:?}",
                e.path, e.tracking_issue
            ));
        }
        if e.reason.trim().is_empty() {
            bad_entries.push(format!("{} — empty `reason` field", e.path));
        }
    }
    assert!(
        bad_entries.is_empty(),
        "_surface_exclusions.toml has {} malformed entries:\n  {}",
        bad_entries.len(),
        bad_entries.join("\n  ")
    );

    let exclusion_set: BTreeMap<String, (String, String)> = exclusions
        .into_iter()
        .map(|e| (e.path, (e.reason, e.tracking_issue)))
        .collect();

    let test_sources = read_conformance_test_sources();
    assert!(
        !test_sources.is_empty(),
        "no conformance test source files found in tests/. Expected at least \
         `tests/conformance_distributed.rs` to exist."
    );

    let mut covered: Vec<&str> = Vec::new();
    let mut excluded: Vec<(&str, &str, &str)> = Vec::new();
    let mut uncovered: Vec<&SurfaceItem> = Vec::new();

    for item in &inventory.item {
        // Glob re-exports (`pub use foo::*`) are never auto-coverable.
        if item.path.ends_with("::*") {
            if let Some((reason, issue)) = exclusion_set.get(&item.path) {
                excluded.push((item.path.as_str(), reason.as_str(), issue.as_str()));
            } else {
                uncovered.push(item);
            }
            continue;
        }

        if let Some((reason, issue)) = exclusion_set.get(&item.path) {
            excluded.push((item.path.as_str(), reason.as_str(), issue.as_str()));
            continue;
        }

        let keys = coverage_keys(&item.path);
        let referenced = keys.iter().any(|k| test_sources.contains(k.as_str()));
        if referenced {
            covered.push(item.path.as_str());
        } else {
            uncovered.push(item);
        }
    }

    eprintln!("--- conformance surface coverage (ferrotorch-distributed) ---");
    eprintln!(
        "covered {}/{} (excluded: {}; uncovered: {})",
        covered.len(),
        inventory.item.len(),
        excluded.len(),
        uncovered.len()
    );

    if !uncovered.is_empty() {
        eprintln!("\n  UNCOVERED items (need a conformance test OR an exclusion entry):");
        for item in &uncovered {
            eprintln!("    {}  (kind={})", item.path, item.kind);
        }
    }

    assert!(
        uncovered.is_empty(),
        "{} ferrotorch-distributed public item(s) lack a conformance reference. \
         Either author a test in tests/conformance_*.rs that references the \
         item by name, OR add it to tests/conformance/_surface_exclusions.toml \
         with `reason` and `tracking_issue` fields.",
        uncovered.len()
    );

    // Stale-exclusion guard: an exclusion for an item that no longer exists
    // in the inventory is suspect (renamed or removed; exclusion was forgotten).
    let inventory_paths: std::collections::BTreeSet<&str> =
        inventory.item.iter().map(|i| i.path.as_str()).collect();
    let stale: Vec<&str> = exclusion_set
        .keys()
        .filter(|k| !inventory_paths.contains(k.as_str()))
        .map(String::as_str)
        .collect();
    assert!(
        stale.is_empty(),
        "_surface_exclusions.toml lists items that no longer exist in the \
         surface inventory (stale entries — remove or update): {stale:?}"
    );
}
