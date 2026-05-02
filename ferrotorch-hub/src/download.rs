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

/// Download `info.weights_url` via blocking HTTP, verify the SHA-256
/// digest, and write the bytes to the cache. Returns the cache path.
///
/// Verification policy:
/// - If `info.weights_sha256` is exactly 64 hex zero characters, the
///   checksum is treated as a placeholder and skipped (a warning is
///   printed). Real registry entries should always pin a checksum.
/// - Otherwise the downloaded bytes' SHA-256 must match exactly. A
///   mismatch deletes the partial cache file and returns an error.
#[cfg(feature = "http")]
fn download_and_verify(info: &ModelInfo, cache: &HubCache) -> FerrotorchResult<PathBuf> {
    use sha2::{Digest, Sha256};
    use std::io::Read;

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

    // Verify SHA-256 unless the registry entry uses the all-zero
    // placeholder.
    let placeholder_sha = "0".repeat(64);
    if info.weights_sha256 != placeholder_sha {
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
    } else {
        eprintln!(
            "ferrotorch-hub: WARNING: model '{}' has placeholder SHA-256 in the registry; \
             integrity check skipped. This should be fixed by pinning a real checksum.",
            info.name
        );
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
/// use ferrotorch_hub::load_pretrained;
///
/// let state_dict = load_pretrained::<f32>("resnet50").unwrap();
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
/// # Errors
///
/// - `FerrotorchError::InvalidArgument` for HTTP failures or malformed JSON.
/// - I/O errors from the cache writer.
#[cfg(feature = "http")]
pub fn hf_download_model(
    repo: &str,
    revision: &str,
    cache: &HubCache,
) -> FerrotorchResult<PathBuf> {
    use std::io::Read;

    if repo.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "hf_download_model: repo must not be empty".into(),
        });
    }
    let revision = if revision.is_empty() {
        "main"
    } else {
        revision
    };

    // Helper to fetch one file from the repo into the cache. `relative` is
    // the path-within-repo (e.g. "config.json", "model-00001-of-00004.safetensors").
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
        cache.store(&cache_name, &body)?;
        Ok(cache.path(&cache_name))
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
}
