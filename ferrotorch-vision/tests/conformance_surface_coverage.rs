//! Conformance Layer 4 — strict per-item coverage gate for `ferrotorch-vision`.
//!
//! Tracking issue: #870 (ferrotorch-vision conformance suite).
//!
//! Loads `tests/conformance/_surface_inventory.toml` and scans every
//! `tests/conformance_*.rs` file (other than this one) for references to
//! each public item. Fails the build if any item is neither (a) referenced
//! in a conformance test, nor (b) explicitly excluded in
//! `tests/conformance/_surface_exclusions.toml` with a `reason` **and** a
//! `tracking_issue` field.
//!
//! Mirrors the pattern in `ferrotorch-serialize/tests/conformance_surface_coverage.rs`
//! and `ferrotorch-train/tests/conformance_surface_coverage.rs`.

use std::collections::BTreeMap;
use std::fs;
use std::path::PathBuf;

use serde::Deserialize;

// ---------------------------------------------------------------------------
// Surface inventory types
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct SurfaceInventory {
    #[allow(dead_code)]
    crate_name: Option<String>,
    #[serde(rename = "item")]
    items: Vec<SurfaceItem>,
}

#[derive(Debug, Deserialize)]
struct SurfaceItem {
    path: String,
    kind: String,
    #[allow(dead_code, reason = "deserialized for forward-compat")]
    signature: String,
}

// ---------------------------------------------------------------------------
// Exclusions types
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
    let raw: toml::Value =
        toml::from_str(&body).unwrap_or_else(|e| panic!("parse {}: {e}", p.display()));
    // The TOML has top-level keys like `crate`, `description` plus [[item]] array.
    let items_val = raw.get("item").expect("_surface_inventory.toml: missing [[item]] array");
    let items: Vec<SurfaceItem> =
        items_val.clone().try_into().unwrap_or_else(|e| panic!("deserialize items: {e}"));
    SurfaceInventory { crate_name: None, items }
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

/// Read every `tests/conformance_*.rs` except this gate; return concatenated source.
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
        let body = fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
        combined.push_str(&body);
        combined.push('\n');
    }
    combined
}

// ---------------------------------------------------------------------------
// Coverage-key logic (mirrors ferrotorch-serialize / ferrotorch-train)
// ---------------------------------------------------------------------------

fn coverage_keys(path: &str) -> Vec<String> {
    let segs: Vec<&str> = path.split("::").collect();
    // For Type::method paths, use "Type::method" to avoid false positives.
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
        let ident = path.rsplit("::").next().unwrap_or(path);
        vec![ident.to_string()]
    }
}

/// Strip turbofish generics (`Type::<T>::method` → `Type::method`) from
/// concatenated test source so coverage keys match regardless of whether
/// tests use the turbofish form.
fn normalize_source_for_coverage(src: &str) -> String {
    // Remove `::<...>` sequences (non-nested). This handles the common
    // pattern `Resize::<f64>::new` → `Resize::new`.
    let mut out = String::with_capacity(src.len());
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut i = 0;
    while i < n {
        // Look for `::<`
        if i + 2 < n && bytes[i] == b':' && bytes[i + 1] == b':' && bytes[i + 2] == b'<' {
            // Skip until matching `>`, handling one level of nesting.
            let mut depth = 0usize;
            let mut j = i + 2; // points at '<'
            while j < n {
                if bytes[j] == b'<' {
                    depth += 1;
                } else if bytes[j] == b'>' {
                    depth -= 1;
                    if depth == 0 {
                        j += 1; // skip the closing '>'
                        break;
                    }
                }
                j += 1;
            }
            // i..j was `::<...>` — skip it (emit nothing), continue from j
            i = j;
        } else {
            out.push(bytes[i] as char);
            i += 1;
        }
    }
    out
}

// ---------------------------------------------------------------------------
// tracking_issue validation
// ---------------------------------------------------------------------------

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
// Gate test
// ---------------------------------------------------------------------------

#[test]
fn every_public_item_has_a_conformance_reference_or_tracking_issue() {
    let inventory = read_inventory();
    let exclusions = read_exclusions();

    // Validate exclusion entries.
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

    let raw_sources = read_conformance_test_sources();
    assert!(
        !raw_sources.is_empty(),
        "no conformance test source files found in tests/. Expected at least \
         `tests/conformance_vision_transforms.rs` to exist."
    );
    // Normalize turbofish (`Type::<T>::method` → `Type::method`) so coverage
    // keys like `Resize::new` match both `Resize::new(...)` and
    // `Resize::<f64>::new(...)` in test source.
    let test_sources = normalize_source_for_coverage(&raw_sources);

    let mut covered: Vec<&str> = Vec::new();
    let mut excluded: Vec<(&str, &str, &str)> = Vec::new();
    let mut uncovered: Vec<&SurfaceItem> = Vec::new();

    for item in &inventory.items {
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

    eprintln!("--- conformance surface coverage (ferrotorch-vision) ---");
    eprintln!(
        "covered {}/{} (excluded: {}; uncovered: {})",
        covered.len(),
        inventory.items.len(),
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
        "{} ferrotorch-vision public item(s) lack a conformance reference. \
         Either author a test in tests/conformance_*.rs that references the \
         item by name, OR add it to tests/conformance/_surface_exclusions.toml \
         with `reason` and `tracking_issue` fields.",
        uncovered.len()
    );

    // Stale-exclusion guard.
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
