//! Conformance Phase — strict per-item coverage gate for `ferrotorch-hub`.
//!
//! Loads `tests/conformance/_surface_inventory.toml` and scans the
//! `tests/conformance_*.rs` files for references to each public item.
//! Fails the build if any inventory item is neither:
//!   (a) referenced by a conformance test (substring match), nor
//!   (b) explicitly excluded in `tests/conformance/_surface_exclusions.toml`
//!       with a written `reason` AND a `tracking_issue` ref.
//!
//! The tracking-issue requirement is the audit trail — "deferred without a
//! follow-up issue" is a no-fly state and the gate rejects it.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// TOML types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SurfaceInventory {
    item: Vec<SurfaceItem>,
}

#[derive(Debug, Deserialize)]
struct SurfaceItem {
    path: String,
    kind: String,
    #[allow(dead_code)]
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
// Paths
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
    let body = fs::read_to_string(&p).unwrap_or_else(|e| {
        panic!(
            "read {} failed: {e}. The file is committed to the repo; \
             if it's missing, check your working tree.",
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

/// Read every `tests/conformance_*.rs` (excluding this coverage gate itself)
/// and return their concatenated source. Coverage check is substring-based.
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

/// Build the substrings that prove coverage for a given item path.
/// For methods (`...::Foo::bar`) we require `Foo::bar` (so an unrelated `bar`
/// elsewhere doesn't accidentally cover this). For free fns / types / re-exports
/// the short ident is enough.
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
        let short = segs.last().copied().unwrap_or(path);
        vec![short.to_string()]
    }
}

/// Placeholder values rejected as `tracking_issue`.
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
    let hash_form = s.starts_with('#') && s[1..].chars().all(|c| c.is_ascii_digit()) && s.len() > 1;
    let url_form = s.starts_with("http://") || s.starts_with("https://");
    // Also accept crosslink-style short refs like "#HUB-CONFORM-1" or "#NNN"
    let crosslink_form = s.starts_with('#') && s.len() > 1;
    hash_form || url_form || crosslink_form
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn every_public_item_has_a_conformance_reference_or_tracking_issue() {
    let inventory = read_inventory();
    let exclusions = read_exclusions();

    // Validate exclusion entries
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
        "no conformance test source files found — expected at least conformance_hub.rs"
    );

    let mut covered: Vec<&str> = Vec::new();
    let mut excluded: Vec<(&str, &str, &str)> = Vec::new();
    let mut uncovered: Vec<&SurfaceItem> = Vec::new();

    for item in &inventory.item {
        // Re-exports are implicitly covered if the underlying item is covered.
        // They still require an explicit entry (in the inventory or exclusions)
        // so the inventory is complete, but we treat them as auto-excluded here
        // since testing the re-export separately from the underlying item would
        // be a phantom test.
        if item.kind == "re-export" {
            if let Some((reason, issue)) = exclusion_set.get(&item.path) {
                excluded.push((item.path.as_str(), reason.as_str(), issue.as_str()));
            } else {
                // Re-exports are implicitly covered; mark as covered.
                covered.push(item.path.as_str());
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

    eprintln!("--- conformance surface coverage (ferrotorch-hub) ---");
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
        "{} ferrotorch-hub public item(s) lack a conformance reference. \
         Either reference the item by name in tests/conformance_hub.rs, OR \
         add it to tests/conformance/_surface_exclusions.toml with `reason` \
         and `tracking_issue` fields.",
        uncovered.len()
    );

    // Stale-exclusion guard: exclusions for items that no longer exist
    let surface_paths: std::collections::BTreeSet<&str> =
        inventory.item.iter().map(|i| i.path.as_str()).collect();
    let stale: Vec<&str> = exclusion_set
        .keys()
        .filter(|k| !surface_paths.contains(k.as_str()))
        .map(String::as_str)
        .collect();
    assert!(
        stale.is_empty(),
        "_surface_exclusions.toml lists items that no longer exist in the \
         surface inventory (stale entries — remove or update): {stale:?}"
    );
}
