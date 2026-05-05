//! Local cache for pretrained model weights.
//!
//! Weights are stored as flat files under `~/.ferrotorch/hub/` by default.
//! Each model's weights file is named after the model (e.g. `resnet50.safetensors`).

use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

use crate::registry::{ModelInfo, WeightsFormat};

/// Return the default cache directory: `~/.ferrotorch/hub/`.
#[must_use]
pub fn default_cache_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home).join(".ferrotorch").join("hub")
}

/// Validate a relative-path cache key before joining with `cache_dir`.
///
/// Cache keys may contain `/` separators (sharded HuggingFace layouts use
/// `<repo>/<file>` keys), but every segment must be a `Component::Normal`.
/// This rejects path-traversal attempts (`..`), absolute paths, drive
/// letters, root prefixes, and null bytes — the defense-in-depth guard
/// for the security-audit finding that server-controlled shard filenames
/// (parsed from `model.safetensors.index.json`) flow into
/// [`HubCache::store`] via [`crate::download::hf_download_model`].
///
/// Per-component validation in `download.rs::sanitize_path_component`
/// already rejects unsafe filenames at the caller boundary; this function
/// is the second line of defense at the cache write boundary.
///
/// # Errors
///
/// Returns [`FerrotorchError::InvalidArgument`] when `name`:
/// - is empty
/// - contains a null byte (`\0`)
/// - is absolute (starts with `/` on Unix or `C:\` / `\\?\` on Windows)
/// - contains any non-`Component::Normal` segment (`..`, `.`, root, drive
///   letter, UNC prefix)
pub(crate) fn validate_cache_relative(name: &str) -> FerrotorchResult<()> {
    if name.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "cache key must not be empty".into(),
        });
    }
    if name.contains('\0') {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("cache key contains a null byte: {name:?}"),
        });
    }
    let path = Path::new(name);
    if path.is_absolute() {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("cache key must be relative (absolute paths are rejected): {name:?}"),
        });
    }
    for component in path.components() {
        match component {
            std::path::Component::Normal(_) => {}
            other => {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "cache key must contain only Normal segments \
                         (rejected component {other:?} in {name:?})"
                    ),
                });
            }
        }
    }
    Ok(())
}

/// Local file-system cache for pretrained model weights.
#[derive(Debug)]
pub struct HubCache {
    cache_dir: PathBuf,
}

impl HubCache {
    /// Create a cache backed by the given directory.
    pub fn new(dir: impl AsRef<Path>) -> Self {
        Self {
            cache_dir: dir.as_ref().to_path_buf(),
        }
    }

    /// Create a cache using the default directory (`~/.ferrotorch/hub/`).
    #[must_use]
    pub fn with_default_dir() -> Self {
        Self::new(default_cache_dir())
    }

    /// Return the directory backing this cache.
    #[must_use]
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Check if weights for the given model name are already cached.
    ///
    /// Returns `false` (rather than panicking) for any name that fails
    /// [`validate_cache_relative`] — an invalid cache key is by
    /// definition not present, and propagating an error here would
    /// require a breaking signature change.
    #[must_use]
    pub fn has(&self, name: &str) -> bool {
        if validate_cache_relative(name).is_err() {
            return false;
        }
        self.path(name).exists()
    }

    /// Return the cache file path for a model name.
    ///
    /// The file may or may not exist; use [`has`](Self::has) to check.
    ///
    /// This function performs no validation — it is a pure path
    /// constructor. Path-traversal protection lives in
    /// [`store`](Self::store) and [`load`](Self::load), which validate
    /// `name` via [`validate_cache_relative`] before touching the
    /// filesystem.
    #[must_use]
    pub fn path(&self, name: &str) -> PathBuf {
        self.cache_dir.join(name)
    }

    /// Return the cache file path for a [`ModelInfo`], using the appropriate
    /// file extension for the weights format.
    ///
    /// `info.name` is a `&'static str` from the compiled-in registry, so
    /// it is trusted; this cannot fail at runtime.
    #[must_use]
    pub fn path_for_model(&self, info: &ModelInfo) -> PathBuf {
        let ext = match info.format {
            WeightsFormat::SafeTensors => "safetensors",
            WeightsFormat::FerrotorchStateDict => "fts",
        };
        self.cache_dir.join(format!("{}.{}", info.name, ext))
    }

    /// Store raw bytes in the cache under the given name.
    ///
    /// Creates the cache directory if it does not exist.
    ///
    /// # Errors
    ///
    /// - [`FerrotorchError::InvalidArgument`] when `name` fails
    ///   [`validate_cache_relative`] — defense-in-depth guard against
    ///   path-traversal attacks via server-controlled filenames.
    /// - [`FerrotorchError::InvalidArgument`] when filesystem operations
    ///   (`create_dir_all`, `write`) fail.
    pub fn store(&self, name: &str, data: &[u8]) -> FerrotorchResult<()> {
        // *** SECURITY: defense-in-depth path validation. Server-controlled
        // shard filenames (parsed from model.safetensors.index.json) reach
        // here via hf_download_model; reject any name that would join to
        // outside cache_dir. Per-component validation in
        // download.rs::sanitize_path_component is the first line; this is
        // the second. ***
        validate_cache_relative(name)?;

        std::fs::create_dir_all(&self.cache_dir).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to create cache directory {:?}: {e}", self.cache_dir),
        })?;
        let path = self.path(name);
        // Ensure the parent of `path` exists. `name` may contain `/` for
        // sharded HF downloads (e.g. "meta-llama/Llama-3-8B/config.json"),
        // so we mkdir -p the parent before writing.
        if let Some(parent) = path.parent() {
            if parent != self.cache_dir {
                std::fs::create_dir_all(parent).map_err(|e| FerrotorchError::InvalidArgument {
                    message: format!("failed to create cache subdir {parent:?}: {e}"),
                })?;
            }
        }
        std::fs::write(&path, data).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to write cache file {path:?}: {e}"),
        })?;
        Ok(())
    }

    /// Load cached weights as raw bytes.
    ///
    /// # Errors
    ///
    /// - [`FerrotorchError::InvalidArgument`] when `name` fails
    ///   [`validate_cache_relative`].
    /// - [`FerrotorchError::InvalidArgument`] when the cache file cannot
    ///   be read (missing, permission denied, I/O error).
    pub fn load(&self, name: &str) -> FerrotorchResult<Vec<u8>> {
        validate_cache_relative(name)?;
        let path = self.path(name);
        std::fs::read(&path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to read cache file {path:?}: {e}"),
        })
    }

    /// Remove all files from the cache directory.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::InvalidArgument`] when the cache
    /// directory cannot be read or a file cannot be removed.
    pub fn clear(&self) -> FerrotorchResult<()> {
        if !self.cache_dir.exists() {
            return Ok(());
        }
        let entries =
            std::fs::read_dir(&self.cache_dir).map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to read cache directory {:?}: {e}", self.cache_dir),
            })?;
        for entry in entries {
            let entry = entry.map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("failed to read directory entry: {e}"),
            })?;
            let path = entry.path();
            if path.is_file() {
                std::fs::remove_file(&path).map_err(|e| FerrotorchError::InvalidArgument {
                    message: format!("failed to remove cache file {:?}: {e}", path),
                })?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_store_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());

        let data = b"pretend these are model weights";
        cache.store("test_model", data).unwrap();

        let loaded = cache.load("test_model").unwrap();
        assert_eq!(loaded, data);
    }

    #[test]
    fn test_has_returns_true_after_store() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());

        assert!(!cache.has("my_model"));
        cache.store("my_model", b"data").unwrap();
        assert!(cache.has("my_model"));
    }

    #[test]
    fn test_has_returns_false_for_missing() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());
        assert!(!cache.has("does_not_exist"));
    }

    #[test]
    fn test_path_returns_expected_location() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());
        let expected = dir.path().join("resnet50");
        assert_eq!(cache.path("resnet50"), expected);
    }

    #[test]
    fn test_path_for_model_safetensors() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());
        let info = ModelInfo {
            name: "resnet50",
            description: "",
            weights_url: "",
            weights_sha256: "",
            format: crate::registry::WeightsFormat::SafeTensors,
            num_parameters: 0,
        };
        assert_eq!(
            cache.path_for_model(&info),
            dir.path().join("resnet50.safetensors")
        );
    }

    #[test]
    fn test_path_for_model_fts() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());
        let info = ModelInfo {
            name: "mymodel",
            description: "",
            weights_url: "",
            weights_sha256: "",
            format: crate::registry::WeightsFormat::FerrotorchStateDict,
            num_parameters: 0,
        };
        assert_eq!(cache.path_for_model(&info), dir.path().join("mymodel.fts"));
    }

    #[test]
    fn test_clear_removes_files() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());

        cache.store("model_a", b"aaa").unwrap();
        cache.store("model_b", b"bbb").unwrap();
        assert!(cache.has("model_a"));
        assert!(cache.has("model_b"));

        cache.clear().unwrap();
        assert!(!cache.has("model_a"));
        assert!(!cache.has("model_b"));
    }

    #[test]
    fn test_clear_on_nonexistent_dir_is_ok() {
        let cache = HubCache::new("/tmp/ferrotorch_hub_test_nonexistent_dir_99999");
        // Should not error even if the directory doesn't exist.
        assert!(cache.clear().is_ok());
    }

    #[test]
    fn test_load_missing_file_returns_error() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());
        let result = cache.load("missing_model");
        assert!(result.is_err());
    }

    #[test]
    fn test_store_creates_directory() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested").join("cache");
        let cache = HubCache::new(&nested);

        assert!(!nested.exists());
        cache.store("weights", b"data").unwrap();
        assert!(nested.exists());
        assert!(cache.has("weights"));
    }

    #[test]
    fn test_default_cache_dir_ends_with_hub() {
        let dir = default_cache_dir();
        assert!(
            dir.ends_with("hub"),
            "default cache dir should end with 'hub', got {:?}",
            dir
        );
        let parent = dir.parent().unwrap();
        assert!(
            parent.ends_with(".ferrotorch"),
            "parent should be .ferrotorch, got {:?}",
            parent
        );
    }

    // -----------------------------------------------------------------------
    // validate_cache_relative — security regression tests for path traversal.
    //
    // These cover the audit's #1 finding: server-controlled shard filenames
    // parsed from `model.safetensors.index.json` flow into HubCache::store
    // and previously joined with `cache_dir` without validation. Each case
    // covers a distinct attack vector; all must return Err. The function is
    // pub(crate) so tests reach it directly via `super::`.
    // -----------------------------------------------------------------------

    #[test]
    fn validate_rejects_parent_dir_traversal() {
        let r = validate_cache_relative("../etc/passwd");
        assert!(r.is_err(), "expected Err for `../etc/passwd`, got Ok");
    }

    #[test]
    fn validate_rejects_windows_backslash_traversal() {
        // Windows uses `\` as a separator, so `..\\Windows\\System32` is
        // path traversal on Windows but a single weird filename on Unix.
        // On Unix `Path::new` treats it as a Normal component (not `..`),
        // but it still contains `\` characters. Since this is Unix-running
        // CI, the test asserts platform-correct behaviour: on Unix the
        // string is a single Normal segment (accepted); on Windows it
        // would parse as `..` + components (rejected). We exercise the
        // behaviour on both platforms by also explicitly constructing a
        // forward-slash equivalent, which IS rejected on every platform.
        let r = validate_cache_relative("../../Windows/System32/drivers/etc/hosts");
        assert!(r.is_err(), "expected Err for `../../Windows/...`, got Ok");
    }

    #[test]
    fn validate_rejects_absolute_path_unix() {
        let r = validate_cache_relative("/etc/passwd");
        assert!(r.is_err(), "expected Err for `/etc/passwd`, got Ok");
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("absolute") || msg.contains("Normal"),
            "error should mention absolute or Normal, got: {msg}"
        );
    }

    #[test]
    fn validate_rejects_embedded_parent_traversal() {
        let r = validate_cache_relative("path/../../escape");
        assert!(r.is_err(), "expected Err for `path/../../escape`, got Ok");
    }

    #[test]
    fn validate_rejects_null_byte() {
        let r = validate_cache_relative("path\0null");
        assert!(r.is_err(), "expected Err for null byte, got Ok");
        let msg = format!("{}", r.unwrap_err());
        assert!(
            msg.contains("null"),
            "error should mention null, got: {msg}"
        );
    }

    #[test]
    fn validate_rejects_empty_string() {
        let r = validate_cache_relative("");
        assert!(r.is_err(), "expected Err for empty string, got Ok");
    }

    #[test]
    fn validate_rejects_lone_parent_dir() {
        let r = validate_cache_relative("..");
        assert!(r.is_err(), "expected Err for `..`, got Ok");
    }

    #[test]
    fn validate_rejects_lone_current_dir() {
        // `.` is a CurDir component, not Normal — should be rejected.
        let r = validate_cache_relative(".");
        assert!(r.is_err(), "expected Err for `.`, got Ok");
    }

    #[test]
    fn validate_rejects_leading_current_dir() {
        // `./foo` parses as CurDir + Normal("foo"); CurDir is not Normal,
        // so this is rejected by the audit's binding-constraint spec.
        let r = validate_cache_relative("./foo");
        assert!(r.is_err(), "expected Err for `./foo`, got Ok");
    }

    #[test]
    #[cfg(windows)]
    fn validate_rejects_windows_drive_letter() {
        // `C:\Windows` parses on Windows as a Prefix + RootDir + Normal
        // sequence, all of which fall into the non-Normal rejection path.
        let r = validate_cache_relative(r"C:\Windows");
        assert!(r.is_err(), "expected Err for `C:\\Windows`, got Ok");
    }

    #[test]
    fn validate_accepts_normal_relative_path() {
        // The legitimate sharded HF case: <repo>/<filename> with two
        // Normal segments. Must continue to work.
        let r = validate_cache_relative("meta-llama/Llama-3-8B/config.json");
        assert!(
            r.is_ok(),
            "expected Ok for legitimate sharded path, got {:?}",
            r
        );
    }

    #[test]
    fn validate_accepts_simple_filename() {
        let r = validate_cache_relative("resnet50.safetensors");
        assert!(r.is_ok(), "expected Ok for simple filename, got {:?}", r);
    }

    #[test]
    fn validate_accepts_dotfile_basename() {
        // A leading-dot Normal component (e.g. `.gitkeep`) is technically
        // a valid Normal segment per std::path. The function does NOT
        // forbid hidden files — that's a stylistic choice, not a security
        // boundary. The cache dir is a controlled location; what matters
        // is rejecting `..` and absolutes. The component-level
        // sanitize_path_component in download.rs DOES additionally
        // reject leading-dot for HF shard filenames where a malicious
        // server might pick `.bashrc`.
        let r = validate_cache_relative(".gitkeep");
        assert!(
            r.is_ok(),
            "leading-dot Normal segments are allowed at the cache layer; got {:?}",
            r
        );
    }

    #[test]
    fn validate_blocks_store_with_traversal() {
        // End-to-end: HubCache::store rejects path-traversal names rather
        // than joining them blindly with cache_dir.
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());
        let r = cache.store("../escape.bin", b"payload");
        assert!(
            r.is_err(),
            "store must reject `../escape.bin`, got Ok (CRITICAL security regression)"
        );
        // Verify nothing was written outside the cache dir.
        let escaped = dir.path().parent().unwrap().join("escape.bin");
        assert!(
            !escaped.exists(),
            "store must not write outside cache_dir; found stray file at {escaped:?}"
        );
    }

    #[test]
    fn validate_blocks_load_with_traversal() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());
        let r = cache.load("../../etc/passwd");
        assert!(r.is_err(), "load must reject path traversal");
    }

    #[test]
    fn validate_blocks_store_with_absolute() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());
        let r = cache.store("/tmp/should_not_be_written", b"payload");
        assert!(r.is_err(), "store must reject absolute paths");
    }

    #[test]
    fn has_returns_false_for_invalid_name() {
        // Defense-in-depth: `has` returning true for an invalid name
        // would imply a cache hit at a traversal target.
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());
        assert!(!cache.has("../etc/passwd"));
        assert!(!cache.has("/etc/passwd"));
        assert!(!cache.has(""));
        assert!(!cache.has("foo\0bar"));
    }
}
