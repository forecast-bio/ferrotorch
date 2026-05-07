//! Conformance Layer 4 — strict per-item coverage gate for ferrotorch-ml.
//!
//! Tracking issue: <https://github.com/dollspace-gay/ferrotorch/issues/840>.
//!
//! Loads `tests/conformance/_surface_inventory.toml` (the Layer 1 denominator)
//! and scans the `tests/conformance_ml_*.rs` files for references to each item.
//! Fails the build if any inventory item is neither:
//!   (a) referenced by a conformance test (substring match on its short ident),
//!   nor
//!   (b) explicitly excluded in `tests/conformance/_surface_exclusions.toml`
//!       with a `reason` and a `tracking_issue` field pointing to a filed issue.
//!
//! This is the project's lock-in: no public API can be added to ferrotorch-ml
//! without either a conformance test reference or an exclusion entry.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Data types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct InventoryFile {
    item: Vec<InventoryItem>,
}

#[derive(Debug, Deserialize)]
struct InventoryItem {
    path: String,
    #[allow(dead_code, reason = "kept for reporting")]
    kind: String,
    #[allow(dead_code, reason = "kept for reporting")]
    description: String,
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

fn read_inventory() -> Vec<InventoryItem> {
    let p = conformance_dir().join("_surface_inventory.toml");
    let body = fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let parsed: InventoryFile =
        toml::from_str(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()));
    parsed.item
}

fn read_exclusions() -> Vec<Exclusion> {
    let p = conformance_dir().join("_surface_exclusions.toml");
    if !p.exists() {
        return Vec::new();
    }
    let body = fs::read_to_string(&p)
        .unwrap_or_else(|e| panic!("read {}: {e}", p.display()));
    let parsed: ExclusionsFile =
        toml::from_str(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()));
    parsed.exclusions
}

/// Read every `tests/conformance_ml_*.rs` file and return their concatenated
/// source. The coverage check is a substring grep — an item is "covered" iff
/// its short identifier (or `Type::method` segment for methods) appears anywhere
/// in any conformance test source. Substring grep is intentional: tests may
/// reference an item via `use`, a direct call, or a comment.
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
        // Include conformance_ml_*.rs; exclude the surface gate itself and
        // the surface inventory test (neither of which references API items
        // in the way that triggers coverage).
        if !name.starts_with("conformance_ml_") {
            continue;
        }
        if path.extension().and_then(|s| s.to_str()) != Some("rs") {
            continue;
        }
        let body = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        combined.push_str(&body);
        combined.push('\n');
    }
    combined
}

// ---------------------------------------------------------------------------
// Coverage key derivation
//
// For a path like `ferrotorch_ml::metrics::r2_score` → short ident is
// `r2_score`. For a method path like `ferrotorch_ml::adapter::Foo::bar`,
// the coverage key is `Foo::bar`. For glob re-exports (`...::*`), we check
// explicitly for an exclusion and skip the substring search.
// ---------------------------------------------------------------------------

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
        let ident = segs.last().copied().unwrap_or(path);
        vec![ident.to_string()]
    }
}

/// Placeholder values rejected as `tracking_issue`. Mirrors the gate in
/// `ferrotorch-core/tests/conformance_surface_coverage.rs`.
const PLACEHOLDER_TRACKING_VALUES: &[&str] = &["TBD", "T0D0", "?", "n/a", "none", "pending"];

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
// The gate test
// ---------------------------------------------------------------------------

#[test]
fn every_public_item_has_a_conformance_reference_or_tracking_issue() {
    let inventory = read_inventory();
    let exclusions = read_exclusions();

    // Validate exclusion entries first — a malformed entry is a failure
    // regardless of whether it covers anything.
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
        "no conformance_ml_*.rs source files found in tests/. \
         Expected at least conformance_ml_adapter.rs, conformance_ml_metrics.rs, \
         conformance_ml_datasets.rs."
    );

    let mut covered: Vec<&str> = Vec::new();
    let mut excluded: Vec<(&str, &str, &str)> = Vec::new();
    let mut uncovered: Vec<&InventoryItem> = Vec::new();

    for item in &inventory {
        // Glob re-exports end with `::*` — require an explicit exclusion.
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

    eprintln!(
        "--- ferrotorch-ml conformance surface coverage ---"
    );
    eprintln!(
        "covered {}/{} items  (excluded: {}, uncovered: {})",
        covered.len(),
        inventory.len(),
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
        "{} ferrotorch-ml public item(s) lack a conformance reference. \
         Add a test in tests/conformance_ml_*.rs that references the item \
         by name, OR add it to tests/conformance/_surface_exclusions.toml \
         with `reason` and `tracking_issue` fields.",
        uncovered.len()
    );

    // Stale-exclusion guard: an exclusion for an item that no longer exists
    // in the inventory is suspect (renamed or removed).
    let inventory_paths: std::collections::BTreeSet<&str> =
        inventory.iter().map(|i| i.path.as_str()).collect();
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
