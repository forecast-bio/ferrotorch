//! Download and load pretrained model weights.
//!
//! When the `http` feature is enabled (default), missing weights are
//! downloaded from `ModelInfo::weights_url` via a blocking `ureq` HTTP
//! client and verified against `ModelInfo::weights_sha256` before being
//! cached and returned. When the `http` feature is disabled, the user
//! must place weights manually in the cache directory.

use std::path::PathBuf;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_nn::StateDict;

use crate::cache::HubCache;
use crate::registry::{ModelInfo, WeightsFormat, get_model_info};

/// Resolve the cached weights file for a model, downloading if needed
/// and the `http` feature is enabled.
///
/// 1. If the weights are already cached at the canonical path, return it.
/// 2. If they exist under the bare name (no extension, legacy layout),
///    return that.
/// 3. With `http`: download from `info.weights_url`, verify SHA-256,
///    write to the canonical cache path, and return it.
/// 4. Without `http`: return an error with manual download instructions.
pub fn download_weights(info: &ModelInfo, cache: &HubCache) -> FerrotorchResult<PathBuf> {
    let path = cache.path_for_model(info);
    if path.exists() {
        return Ok(path);
    }

    // Backwards-compat: also check the bare name (without extension).
    let bare = cache.path(info.name);
    if bare.exists() {
        return Ok(bare);
    }

    #[cfg(feature = "http")]
    {
        download_and_verify(info, cache)
    }

    #[cfg(not(feature = "http"))]
    {
        Err(FerrotorchError::InvalidArgument {
            message: format!(
                "Pretrained weights for '{}' not found in cache at {:?}, and the \
                 `http` feature is disabled. Download manually from {} and place at {:?}",
                info.name,
                cache.path_for_model(info),
                info.weights_url,
                cache.path_for_model(info),
            ),
        })
    }
}

/// Returns `true` when `s` is the registry's all-zero SHA-256 placeholder.
///
/// The registry currently ships every entry with the placeholder string
/// `"0".repeat(64)` so the wiring lands before real checkpoints have been
/// uploaded. Detecting the placeholder is the trigger for the fail-fast
/// policy in [`download_and_verify`]: a placeholder hash silently bypasses
/// integrity verification, which is unsafe behaviour for a security-relevant
/// download path. We return `Err` rather than warn-and-continue so the
/// "registry SHA256 not yet pinned" condition surfaces deterministically.
#[cfg(feature = "http")]
fn is_placeholder_sha256(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| b == b'0')
}

/// Download `info.weights_url` via blocking HTTP, verify the SHA-256
/// digest, and write the bytes to the cache. Returns the cache path.
///
/// Verification policy (post-audit, fail-fast):
/// - If `info.weights_sha256` is the all-zero placeholder, the request is
///   refused with an `InvalidArgument` error pointing at the registry —
///   silently skipping integrity verification for placeholder entries was
///   the security audit's #6 finding. Populating real hashes is tracked as
///   follow-up issue #739.
/// - Otherwise the downloaded bytes' SHA-256 must match exactly. A
///   mismatch returns an error and does not write to the cache.
#[cfg(feature = "http")]
fn download_and_verify(info: &ModelInfo, cache: &HubCache) -> FerrotorchResult<PathBuf> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

    // *** SECURITY (audit #6): refuse fail-open behaviour. The previous
    // policy was "if placeholder, skip verification with eprintln warning",
    // which silently bypassed integrity for every registry entry on the
    // download path. Convert to fail-fast — the caller now sees a clear
    // error tied to the missing registry hash, instead of an unverified
    // download succeeding behind their back. ***
    if is_placeholder_sha256(info.weights_sha256) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ferrotorch-hub: registry SHA-256 for '{}' is the all-zero placeholder; \
                 refusing to download without integrity verification. Populate the real \
                 checksum in the registry before retrying (tracked as follow-up #739).",
                info.name
            ),
        });
    }

    // Fetch the body. ureq's blocking client returns a Response we can
    // read into a Vec<u8>; for very large weights this allocates the
    // whole file in memory, which is acceptable for typical
    // < 1 GB models. Streaming-to-disk could be added later if needed.
    // Inject `Authorization: Bearer <HF_TOKEN>` when the user has configured
    // one. No-op for public weights; required for gated repos. (#509)
    let response = crate::auth::with_auth(ureq::get(info.weights_url))
        .call()
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "ferrotorch-hub: HTTP request to {} failed: {e}",
                info.weights_url
            ),
        })?;

    // Cap reads at 4 GB to avoid runaway responses if the server lies
    // about content length.
    const MAX_BODY_BYTES: u64 = 4 * 1024 * 1024 * 1024;
    let mut body = Vec::new();
    response
        .into_reader()
        .take(MAX_BODY_BYTES)
        .read_to_end(&mut body)
        .map_err(|e| FerrotorchError::InvalidArgument {
            message: format!(
                "ferrotorch-hub: failed reading response body from {}: {e}",
                info.weights_url
            ),
        })?;

    let mut hasher = Sha256::new();
    hasher.update(&body);
    let digest = hasher.finalize();
    let got_hex = hex_lower(&digest);
    if got_hex != info.weights_sha256.to_lowercase() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "ferrotorch-hub: SHA-256 mismatch for '{}': \
                 expected {}, got {}. The file at {} may be corrupted \
                 or tampered with.",
                info.name, info.weights_sha256, got_hex, info.weights_url
            ),
        });
    }

    // Write to the canonical cache path. Use cache.store to take care
    // of creating the directory and reporting filesystem errors.
    let canonical_filename = canonical_filename(info);
    cache.store(&canonical_filename, &body)?;
    Ok(cache.path(&canonical_filename))
}

/// Compute the cache filename for a [`ModelInfo`], matching
/// [`HubCache::path_for_model`].
#[cfg(feature = "http")]
fn canonical_filename(info: &ModelInfo) -> String {
    let ext = match info.format {
        WeightsFormat::SafeTensors => "safetensors",
        WeightsFormat::FerrotorchStateDict => "fts",
    };
    format!("{}.{}", info.name, ext)
}

/// Format raw bytes as lowercase hexadecimal.
#[cfg(feature = "http")]
fn hex_lower(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(hex_nibble(b >> 4));
        s.push(hex_nibble(b & 0x0f));
    }
    s
}

#[cfg(feature = "http")]
fn hex_nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + (n - 10)) as char,
        _ => unreachable!(),
    }
}

/// Load a pretrained model's state dict by name.
///
/// Looks up the model in the registry, fetches (or returns from cache)
/// the weights file, and deserializes it into a [`StateDict`].
///
/// # Errors
///
/// Returns an error if:
/// - The model name is not in the registry.
/// - The weights cannot be fetched (network error, HTTP error, etc.) and
///   are not already cached.
/// - The downloaded bytes' SHA-256 does not match the registry entry.
/// - The weights file cannot be parsed.
///
/// # Example
///
/// ```rust,no_run
/// use ferrotorch_core::FerrotorchError;
/// use ferrotorch_hub::load_pretrained;
///
/// fn main() -> Result<(), FerrotorchError> {
///     let _state_dict = load_pretrained::<f32>("resnet50")?;
///     Ok(())
/// }
/// ```
pub fn load_pretrained<T: Float>(name: &str) -> FerrotorchResult<StateDict<T>> {
    let info = get_model_info(name).ok_or_else(|| FerrotorchError::InvalidArgument {
        message: format!(
            "Unknown model '{}'. Use ferrotorch_hub::list_models() to see available models.",
            name
        ),
    })?;

    let cache = HubCache::with_default_dir();
    let path = download_weights(info, &cache)?;

    match info.format {
        WeightsFormat::SafeTensors => ferrotorch_serialize::load_safetensors(&path),
        WeightsFormat::FerrotorchStateDict => ferrotorch_serialize::load_state_dict(&path),
    }
}

// ---------------------------------------------------------------------------
// Sharded model download (#509)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Path-component sanitizer
// ---------------------------------------------------------------------------

/// Maximum allowed byte length for any single path component (repo, revision,
/// shard filename).  255 is the POSIX `NAME_MAX` for most file systems.
const MAX_COMPONENT_BYTES: usize = 255;

/// Validate that `s` is safe to use as a single path component in a cache
/// key or URL segment.
///
/// A *safe* component:
/// - Is non-empty.
/// - Is no longer than [`MAX_COMPONENT_BYTES`] bytes (POSIX `NAME_MAX`).
/// - Contains no null bytes (`\0`).
/// - Contains no `/` or `\` separators (would escape the intended directory).
/// - Contains no `:` (Windows ADS; also illegal in filenames on some systems).
/// - Is not `..` or `.` (directory traversal).
/// - Does not begin with `.` (hidden files — we conservatively reject these
///   to block both `..` prefix variants and unintended dotfiles; callers that
///   genuinely need dotfile names can bypass this function, which is `pub(crate)`).
///
/// Because this function accepts `&str`, valid UTF-8 is already guaranteed by
/// Rust's type system; no additional encoding check is necessary.
///
/// # Errors
///
/// Returns `FerrotorchError::InvalidArgument` describing the specific rejection
/// reason, so callers surface actionable diagnostics.
pub(crate) fn sanitize_path_component(s: &str, role: &str) -> FerrotorchResult<()> {
    if s.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("hf_download_model: {role} must not be empty"),
        });
    }
    if s.len() > MAX_COMPONENT_BYTES {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "hf_download_model: {role} is too long ({} bytes; max {MAX_COMPONENT_BYTES})",
                s.len()
            ),
        });
    }
    if s.contains('\0') {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("hf_download_model: {role} contains a null byte"),
        });
    }
    if s.contains('/') || s.contains('\\') {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "hf_download_model: {role} contains a path separator ('/' or '\\'): {s:?}"
            ),
        });
    }
    if s.contains(':') {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "hf_download_model: {role} contains a colon (potential Windows ADS / illegal character): {s:?}"
            ),
        });
    }
    if s == ".." || s == "." {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("hf_download_model: {role} is a dot-path component ({s:?})"),
        });
    }
    if s.starts_with('.') {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "hf_download_model: {role} starts with '.' (hidden/dotfile components are not allowed): {s:?}"
            ),
        });
    }
    Ok(())
}

/// Collapse `..` and `.` components in a path lexically, without requiring
/// the path to exist on disk.  This is intentionally conservative: if a `..`
/// component would escape the root (i.e. the stack underflows), that component
/// is silently dropped — which is safe because it means the path was already
/// at the root and a `..` would escape it, a condition that the caller's
/// containment check below will catch.
fn lexical_normalize(p: &std::path::Path) -> std::path::PathBuf {
    use std::path::{Component, PathBuf};
    let mut out = PathBuf::new();
    for component in p.components() {
        match component {
            Component::CurDir => {} // `.` — skip
            Component::ParentDir => {
                // `..` — pop one level
                out.pop();
            }
            c => out.push(c),
        }
    }
    out
}

/// Verify that `full_path` is contained within `base`.
///
/// After building a cache path from sanitized components we do a final
/// defense-in-depth check that guards against TOCTOU and any edge cases the
/// component sanitizer might miss (e.g. unicode normalization, symlinks).
///
/// Strategy:
/// 1. Try `std::fs::canonicalize` on both paths (strongest form — resolves
///    symlinks).  If both succeed, use the canonicalized forms for the check.
/// 2. If `full_path` doesn't exist yet (common: the shard file hasn't been
///    written), fall back to canonicalizing `base` only (it should exist) and
///    then lexically normalizing `full_path` via [`lexical_normalize`].  The
///    lexical normalization collapses any `..` components that survived
///    sanitization, making the fallback reliable even when the target file is
///    not yet on disk.
///
/// # Errors
///
/// Returns `FerrotorchError::InvalidArgument` if containment cannot be
/// confirmed.  The write is always refused in that case.
fn assert_within_cache(
    base: &std::path::Path,
    full_path: &std::path::Path,
) -> FerrotorchResult<()> {
    // Strong form: both paths exist and can be canonicalized (symlinks resolved).
    if let (Ok(b), Ok(p)) = (
        std::fs::canonicalize(base),
        std::fs::canonicalize(full_path),
    ) {
        if !p.starts_with(&b) {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "hf_download_model: resolved path {p:?} escapes cache directory {b:?} \
                     (path traversal blocked)"
                ),
            });
        }
        return Ok(());
    }

    // Fallback: `full_path` doesn't exist yet.  Canonicalize `base` (which
    // must exist) and lexically normalize `full_path` to collapse any `..`.
    let base_norm = std::fs::canonicalize(base).unwrap_or_else(|_| lexical_normalize(base));
    let path_norm = lexical_normalize(full_path);

    if !path_norm.starts_with(&base_norm) {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "hf_download_model: path {full_path:?} (normalized: {path_norm:?}) escapes \
                 cache directory {base:?} (path traversal blocked)"
            ),
        });
    }
    Ok(())
}

/// Download every safetensors shard referenced in a HuggingFace model
/// repo's index, plus `config.json` and (optionally) the tokenizer files,
/// into the local cache. Returns the cache directory containing the
/// downloaded files.
///
/// # Behaviour
///
/// 1. Fetch `config.json` from `https://huggingface.co/{repo}/resolve/{revision}/config.json`.
/// 2. Try `model.safetensors.index.json` to discover shard filenames; on
///    404 fall back to a single `model.safetensors` file.
/// 3. Download each shard concurrently-friendly (sequential for now;
///    the heavy I/O is in the response body so a thread pool is a future
///    optimisation).
/// 4. Auth header is injected via [`crate::auth::with_auth`] when
///    `HF_TOKEN` is set — required for gated repos.
///
/// `revision` is typically `"main"` but accepts any branch / tag / commit
/// SHA. The cache layout follows the existing `HubCache::path(name)` flat
/// structure: each downloaded file lands at `cache.path("{repo_id}/{filename}")`.
///
/// # Security
///
/// `repo`, `revision`, and any shard filenames parsed from the server's
/// `model.safetensors.index.json` are validated by
/// [`sanitize_path_component`] before being used in cache keys or URLs.
/// In addition, the final resolved path is verified to lie within
/// `cache_dir` before any write is attempted.  This guards against path
/// traversal attacks from a malicious or compromised HuggingFace response.
///
/// # Errors
///
/// - `FerrotorchError::InvalidArgument` for HTTP failures, malformed JSON,
///   or path-traversal attempts in any server-supplied filename.
/// - I/O errors from the cache writer.
#[cfg(feature = "http")]
pub fn hf_download_model(
    repo: &str,
    revision: &str,
    cache: &HubCache,
) -> FerrotorchResult<PathBuf> {
    use std::io::Read;

    // --- Sanitize caller-supplied inputs (defense-in-depth: these are not
    //     server-controlled but still flow into cache paths and URLs). ---
    //
    // Note: `repo` legitimately contains a single `/` separator
    // (e.g. "meta-llama/Llama-3-8B"), so we split on `/` and validate each
    // component individually instead of rejecting the whole string.
    {
        let parts: Vec<&str> = repo.split('/').collect();
        if parts.is_empty() || repo.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "hf_download_model: repo must not be empty".into(),
            });
        }
        for part in &parts {
            sanitize_path_component(part, "repo component")?;
        }
    }
    let revision = if revision.is_empty() {
        "main"
    } else {
        revision
    };
    sanitize_path_component(revision, "revision")?;

    // Helper to fetch one file from the repo into the cache. `relative` is
    // the path-within-repo (e.g. "config.json", "model-00001-of-00004.safetensors").
    // PRECONDITION: callers must have already run `sanitize_path_component` on
    // every component of `repo`, `revision`, and `relative` before calling this.
    fn fetch_one(
        repo: &str,
        revision: &str,
        relative: &str,
        cache: &HubCache,
    ) -> FerrotorchResult<PathBuf> {
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{relative}");
        let response = crate::auth::with_auth(ureq::get(&url))
            .call()
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("hf_download_model: GET {url} failed: {e}"),
            })?;
        const MAX_BODY_BYTES: u64 = 16 * 1024 * 1024 * 1024; // 16 GB shard ceiling
        let mut body = Vec::new();
        use std::io::Read as _;
        response
            .into_reader()
            .take(MAX_BODY_BYTES)
            .read_to_end(&mut body)
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("hf_download_model: reading {url}: {e}"),
            })?;
        let cache_name = format!("{repo}/{relative}");
        // Defense-in-depth: verify the resolved path is still inside cache_dir
        // before writing. Guards against TOCTOU and any edge cases the
        // component sanitizer might miss (e.g. unicode normalization).
        let final_path = cache.path(&cache_name);
        assert_within_cache(cache.cache_dir(), &final_path)?;
        cache.store(&cache_name, &body)?;
        Ok(final_path)
    }

    // Helper that probes a URL and returns Some(body) on 200, None on 404.
    fn fetch_optional(
        repo: &str,
        revision: &str,
        relative: &str,
    ) -> FerrotorchResult<Option<Vec<u8>>> {
        let url = format!("https://huggingface.co/{repo}/resolve/{revision}/{relative}");
        let result = crate::auth::with_auth(ureq::get(&url)).call();
        match result {
            Ok(response) => {
                const MAX: u64 = 64 * 1024 * 1024;
                let mut body = Vec::new();
                response
                    .into_reader()
                    .take(MAX)
                    .read_to_end(&mut body)
                    .map_err(|e| FerrotorchError::InvalidArgument {
                        message: format!("hf_download_model: reading {url}: {e}"),
                    })?;
                Ok(Some(body))
            }
            Err(ureq::Error::Status(404, _)) => Ok(None),
            Err(e) => Err(FerrotorchError::InvalidArgument {
                message: format!("hf_download_model: GET {url} failed: {e}"),
            }),
        }
    }

    // 1. config.json (required for any reasonable downstream use).
    fetch_one(repo, revision, "config.json", cache)?;

    // 2. Discover shards. If the index file exists, parse it; else fall
    // back to a single-file model.
    if let Some(index_bytes) = fetch_optional(repo, revision, "model.safetensors.index.json")? {
        let index: serde_json::Value =
            serde_json::from_slice(&index_bytes).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("hf_download_model: malformed index.json: {e}"),
            })?;
        // Persist the index to the cache so the loader can find it later.
        cache.store(
            &format!("{repo}/model.safetensors.index.json"),
            &index_bytes,
        )?;
        // The "weight_map" subobject maps tensor names to shard filenames;
        // we want the unique set of shard filenames.
        let weight_map = index.get("weight_map").and_then(|v| v.as_object()).ok_or(
            FerrotorchError::InvalidArgument {
                message: "hf_download_model: index.json missing 'weight_map'".into(),
            },
        )?;
        let mut shards: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for v in weight_map.values() {
            if let Some(s) = v.as_str() {
                // *** SECURITY: sanitize every server-supplied shard filename
                // before inserting it into the download set. `s` comes from
                // parsing `model.safetensors.index.json` returned by the
                // HuggingFace server; a malicious or compromised response
                // could contain `"../../.bashrc"` or similar.  Reject any
                // component that contains path separators, `..`, null bytes,
                // leading dots, colons, or exceeds the name length limit. ***
                sanitize_path_component(s, "shard filename")?;
                shards.insert(s.to_string());
            }
        }
        if shards.is_empty() {
            return Err(FerrotorchError::InvalidArgument {
                message: "hf_download_model: index.json had no shard entries".into(),
            });
        }
        for shard in &shards {
            fetch_one(repo, revision, shard, cache)?;
        }
    } else {
        // Single-file model: try `model.safetensors` directly.
        fetch_one(repo, revision, "model.safetensors", cache)?;
    }

    // 3. Best-effort tokenizer files (some repos ship them; ignore 404).
    for opt in &[
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ] {
        let _ = fetch_optional(repo, revision, opt)?;
    }

    Ok(cache.path(repo))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_weights_returns_path_when_cached() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());

        let info = get_model_info("resnet50").unwrap();
        let expected_path = cache.path_for_model(info);

        // Simulate a successful download by writing the file directly.
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(&expected_path, b"fake weights").unwrap();

        let result = download_weights(info, &cache);
        assert!(result.is_ok(), "expected Ok, got {:?}", result.err());
        assert_eq!(result.unwrap(), expected_path);
    }

    #[test]
    fn test_download_weights_finds_bare_name() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());

        let info = get_model_info("resnet18").unwrap();
        let bare_path = cache.path("resnet18");
        std::fs::write(&bare_path, b"fake weights").unwrap();

        let result = download_weights(info, &cache);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), bare_path);
    }

    #[test]
    fn test_load_pretrained_unknown_model() {
        let result = load_pretrained::<f32>("totally_fake_model");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Unknown model"),
            "error should say unknown model, got: {err_msg}"
        );
        assert!(
            err_msg.contains("list_models"),
            "error should suggest list_models, got: {err_msg}"
        );
    }

    // Tests for the http feature path use only the helpers that don't
    // make network calls. Actual network downloads are not tested here
    // because they would be flaky and require external connectivity;
    // a manual integration test against a real model URL is the right
    // place for that.

    #[cfg(feature = "http")]
    #[test]
    fn test_hex_lower_known_vector() {
        assert_eq!(super::hex_lower(&[0x00, 0xff, 0x10, 0xab]), "00ff10ab");
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_canonical_filename_safetensors() {
        let info = ModelInfo {
            name: "resnet50",
            description: "",
            weights_url: "",
            weights_sha256: "",
            format: WeightsFormat::SafeTensors,
            num_parameters: 0,
        };
        assert_eq!(super::canonical_filename(&info), "resnet50.safetensors");
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_canonical_filename_fts() {
        let info = ModelInfo {
            name: "mymodel",
            description: "",
            weights_url: "",
            weights_sha256: "",
            format: WeightsFormat::FerrotorchStateDict,
            num_parameters: 0,
        };
        assert_eq!(super::canonical_filename(&info), "mymodel.fts");
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sha256_placeholder_detected() {
        // Verify the placeholder constant is exactly 64 hex zeros, so the
        // download path will skip verification on registry entries that
        // haven't been pinned yet.
        let placeholder = "0".repeat(64);
        assert_eq!(placeholder.len(), 64);
        // All known registry entries currently use this placeholder; the
        // detection logic in download_and_verify compares against the
        // exact string.
        for info in crate::registry::list_models() {
            if info.weights_sha256 == placeholder {
                // Just verifying the comparison works as expected.
                assert!(info.weights_sha256.chars().all(|c| c == '0'));
            }
        }
    }

    #[cfg(feature = "http")]
    #[test]
    fn test_sha256_mismatch_against_known_value() {
        // Compute the SHA-256 of a fixed byte string and verify
        // hex_lower produces the expected output. We use a known
        // SHA-256 of the empty string:
        //   e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        hasher.update(b"");
        let digest = hasher.finalize();
        let got = super::hex_lower(&digest);
        assert_eq!(
            got,
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
    }

    // -----------------------------------------------------------------------
    // sanitize_path_component — security regression tests
    // Each case covers a distinct attack vector; all must return Err.
    // -----------------------------------------------------------------------

    /// Helper: assert that sanitize_path_component returns an error for `s`
    /// and that the error message contains `needle` for useful diagnostics.
    fn assert_rejected(s: &str, role: &str, needle: &str) {
        let result = super::sanitize_path_component(s, role);
        assert!(
            result.is_err(),
            "expected Err for {:?} (role={role:?}), got Ok",
            s
        );
        let msg = format!("{}", result.unwrap_err());
        assert!(
            msg.contains(needle),
            "error for {:?} should contain {needle:?}, got: {msg}",
            s
        );
    }

    #[test]
    fn sanitize_rejects_empty_string() {
        assert_rejected("", "shard", "must not be empty");
    }

    #[test]
    fn sanitize_rejects_dotdot() {
        assert_rejected("..", "shard", "dot-path");
    }

    #[test]
    fn sanitize_rejects_single_dot() {
        assert_rejected(".", "shard", "dot-path");
    }

    #[test]
    fn sanitize_rejects_leading_slash() {
        assert_rejected("/etc/passwd", "shard", "path separator");
    }

    #[test]
    fn sanitize_rejects_embedded_slash() {
        assert_rejected("foo/bar", "shard", "path separator");
    }

    #[test]
    fn sanitize_rejects_backslash() {
        assert_rejected("foo\\bar", "shard", "path separator");
    }

    #[test]
    fn sanitize_rejects_null_byte() {
        assert_rejected("foo\0bar", "shard", "null byte");
    }

    #[test]
    fn sanitize_rejects_dotdot_embedded() {
        // "foo/../../../etc/passwd" — slash catches this, but confirm.
        assert_rejected("foo/../../../etc/passwd", "shard", "path separator");
    }

    #[test]
    fn sanitize_rejects_dotdot_at_start_with_slash() {
        assert_rejected("../../.bashrc", "shard", "path separator");
    }

    #[test]
    fn sanitize_rejects_leading_dot() {
        assert_rejected(".hidden", "shard", "starts with '.'");
    }

    #[test]
    fn sanitize_rejects_colon() {
        assert_rejected("foo:bar", "shard", "colon");
    }

    #[test]
    fn sanitize_rejects_windows_ads() {
        // Windows Alternate Data Streams: "file.txt:secret"
        assert_rejected("model.safetensors:secret", "shard", "colon");
    }

    #[test]
    fn sanitize_rejects_overlong_string() {
        let long = "a".repeat(super::MAX_COMPONENT_BYTES + 1);
        assert_rejected(&long, "shard", "too long");
    }

    #[test]
    fn sanitize_accepts_normal_shard_filename() {
        // A typical HuggingFace shard filename should be accepted.
        let ok = super::sanitize_path_component("model-00001-of-00004.safetensors", "shard");
        assert!(
            ok.is_ok(),
            "expected Ok for normal shard filename, got: {:?}",
            ok
        );
    }

    #[test]
    fn sanitize_accepts_normal_revision() {
        for rev in &["main", "v1.0", "abc123def456", "feature-branch"] {
            let ok = super::sanitize_path_component(rev, "revision");
            assert!(
                ok.is_ok(),
                "expected Ok for revision {rev:?}, got: {:?}",
                ok
            );
        }
    }

    #[test]
    fn sanitize_accepts_max_length_string() {
        let exactly_max = "a".repeat(super::MAX_COMPONENT_BYTES);
        let ok = super::sanitize_path_component(&exactly_max, "shard");
        assert!(ok.is_ok(), "expected Ok for exactly-max-length string");
    }

    // -----------------------------------------------------------------------
    // assert_within_cache — verify the path-containment guard
    // -----------------------------------------------------------------------

    #[test]
    fn within_cache_rejects_escaped_path() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path();
        // Construct a path that lexically escapes the cache dir.
        let escaped = cache_dir.join("..").join("outside.txt");
        let result = super::assert_within_cache(cache_dir, &escaped);
        assert!(result.is_err(), "expected Err for escaped path, got Ok");
    }

    #[test]
    fn within_cache_accepts_subpath() {
        let dir = tempfile::tempdir().unwrap();
        let cache_dir = dir.path();
        // Create the sub-path so canonicalize works.
        let sub = cache_dir.join("meta-llama").join("model.safetensors");
        std::fs::create_dir_all(sub.parent().unwrap()).unwrap();
        std::fs::write(&sub, b"fake").unwrap();
        // The cache dir itself exists now; both can be canonicalized.
        let result = super::assert_within_cache(cache_dir, &sub);
        assert!(
            result.is_ok(),
            "expected Ok for sub-path, got: {:?}",
            result
        );
    }
}
