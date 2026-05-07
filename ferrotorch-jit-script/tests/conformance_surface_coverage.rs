//! Conformance surface-coverage gate for `ferrotorch-jit-script`.
//!
//! Tracking issue: #826 (Conformance Buildout C1 — Tier-1 crates).
//!
//! Loads `tests/conformance/_surface_inventory.toml` (the Layer 1 inventory)
//! and checks that every item is either:
//!   (a) referenced by a keyword in the `tests/conformance_jit_script.rs`
//!       source, OR
//!   (b) explicitly excluded in `tests/conformance/_surface_exclusions.toml`
//!       with a non-empty `reason` and a valid `tracking_issue` (must be
//!       `#NNN` or a full URL).
//!
//! Fails the build if any item is in neither state.  This is the lock-in:
//! every new `pub` item in this crate must be referenced in a conformance
//! test or added to the exclusions file with a tracking issue before CI
//! passes.
//!
//! `ferrotorch-jit-script` has a single public item — `script` — so the
//! gate is intentionally small, but the pattern is identical to
//! `ferrotorch-core/tests/conformance_surface_coverage.rs`.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Data model
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct InventoryFile {
    #[allow(dead_code, reason = "forward-compat fields")]
    #[serde(default)]
    crate_name: Option<String>,
    #[allow(dead_code, reason = "forward-compat fields")]
    #[serde(default)]
    description: Option<String>,
    #[serde(rename = "item")]
    items: Vec<InventoryItem>,
}

#[derive(Debug, Deserialize)]
struct InventoryItem {
    path: String,
    kind: String,
    #[allow(dead_code, reason = "kept for forward-compat / reporting")]
    signature: String,
    #[allow(dead_code, reason = "human-readable description")]
    #[serde(default)]
    description: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ExclusionsFile {
    #[serde(default, rename = "exclusion")]
    exclusions: Vec<Exclusion>,
}

#[derive(Debug, Deserialize)]
struct Exclusion {
    path: String,
    reason: String,
    /// Tracking issue ref — required. `#NNN` or a full GitHub URL.
    tracking_issue: String,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn conformance_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
}

fn tests_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn read_inventory() -> InventoryFile {
    let p = conformance_dir().join("_surface_inventory.toml");
    let body = fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. The surface inventory must be committed \
             alongside the conformance suite.",
            p.display()
        )
    });
    toml::from_str(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()))
}

fn read_exclusions() -> Vec<Exclusion> {
    let p = conformance_dir().join("_surface_exclusions.toml");
    if !p.exists() {
        return Vec::new();
    }
    let body = fs::read_to_string(&p).unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let parsed: ExclusionsFile =
        toml::from_str(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()));
    parsed.exclusions
}

/// Read every `tests/conformance_*.rs` (except the inventory and this gate)
/// and return their concatenated source for substring-grep coverage.
fn read_conformance_test_sources() -> String {
    let mut combined = String::new();
    let root = tests_dir();
    let entries =
        fs::read_dir(&root).unwrap_or_else(|e| panic!("read tests dir {}: {e}", root.display()));
    for entry in entries {
        let entry = entry.expect("readdir entry");
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = path.file_name().and_then(|s| s.to_str()).unwrap_or("");
        if !name.starts_with("conformance_") {
            continue;
        }
        if name == "conformance_surface_coverage.rs" {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let body =
            fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        combined.push_str(&body);
        combined.push('\n');
    }
    combined
}

/// Extract the short identifier from a fully-qualified path.
/// `ferrotorch_jit_script::script` → `script`
fn short_ident(path: &str) -> &str {
    path.rsplit("::").next().unwrap_or(path)
}

/// Build the coverage keys for a path (same logic as ferrotorch-core's gate).
/// For method paths (`...::Type::method`) require `Type::method`.
/// For free functions / macros / types the short ident is enough.
fn coverage_keys(path: &str) -> Vec<String> {
    let segs: Vec<&str> = path.split("::").collect();
    if segs.len() >= 3
        && segs[segs.len() - 2]
            .chars()
            .next()
            .is_some_and(char::is_uppercase)
    {
        let ty = segs[segs.len() - 2];
        let m = segs[segs.len() - 1];
        vec![format!("{ty}::{m}")]
    } else {
        vec![short_ident(path).to_string()]
    }
}

/// Placeholder values rejected as `tracking_issue`.
const PLACEHOLDER_TRACKING_VALUES: &[&str] = &["TBD", "T0D0", "?", "n/a", "none", "pending"];

/// Validate the shape of a `tracking_issue` field.  Accepts `#NNN` or a
/// full URL; rejects empty / placeholder values.
fn tracking_issue_valid(s: &str) -> bool {
    let s = s.trim();
    if s.is_empty() {
        return false;
    }
    let lc = s.to_ascii_lowercase();
    if PLACEHOLDER_TRACKING_VALUES
        .iter()
        .any(|p| p.eq_ignore_ascii_case(&lc))
    {
        return false;
    }
    let hash_form =
        s.starts_with('#') && s[1..].chars().all(|c| c.is_ascii_digit()) && s.len() > 1;
    let url_form = s.starts_with("http://") || s.starts_with("https://");
    hash_form || url_form
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn every_public_item_has_a_conformance_reference_or_tracking_issue() {
    let inventory = read_inventory();
    let exclusions = read_exclusions();

    // Validate exclusion entries first.
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
         tests/conformance_jit_script.rs."
    );

    let mut covered: Vec<&str> = Vec::new();
    let mut excluded: Vec<(&str, &str, &str)> = Vec::new();
    let mut uncovered: Vec<&InventoryItem> = Vec::new();

    for item in &inventory.items {
        // Glob re-exports (`path` ending in `::*`) require an explicit exclusion.
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

    eprintln!("--- conformance surface coverage (ferrotorch-jit-script, #826) ---");
    eprintln!(
        "covered {}/{} items (excluded: {}; uncovered: {})",
        covered.len(),
        inventory.items.len(),
        excluded.len(),
        uncovered.len(),
    );

    if !uncovered.is_empty() {
        eprintln!("\n  UNCOVERED items (need a conformance test OR an exclusion entry):");
        for item in &uncovered {
            eprintln!("    {}  (kind={})", item.path, item.kind);
        }
    }

    assert!(
        uncovered.is_empty(),
        "{} ferrotorch-jit-script public item(s) lack a conformance reference. \
         Either reference the item by name in tests/conformance_jit_script.rs, \
         OR add an entry to tests/conformance/_surface_exclusions.toml with \
         `reason` and `tracking_issue` fields.",
        uncovered.len()
    );

    // Stale-exclusion guard: an exclusion for an item that no longer exists
    // in the inventory is suspect.
    let inventory_paths: std::collections::BTreeSet<&str> =
        inventory.items.iter().map(|i| i.path.as_str()).collect();
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
