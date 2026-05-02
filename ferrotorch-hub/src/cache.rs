//! Local cache for pretrained model weights.
//!
//! Weights are stored as flat files under `~/.ferrotorch/hub/` by default.
//! Each model's weights file is named after the model (e.g. `resnet50.safetensors`).

use std::path::{Path, PathBuf};

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

use crate::registry::{ModelInfo, WeightsFormat};

/// Return the default cache directory: `~/.ferrotorch/hub/`.
pub fn default_cache_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_default();
    PathBuf::from(home).join(".ferrotorch").join("hub")
}

/// Local file-system cache for pretrained model weights.
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
    pub fn with_default_dir() -> Self {
        Self::new(default_cache_dir())
    }

    /// Return the directory backing this cache.
    pub fn cache_dir(&self) -> &Path {
        &self.cache_dir
    }

    /// Check if weights for the given model name are already cached.
    pub fn has(&self, name: &str) -> bool {
        self.path(name).exists()
    }

    /// Return the cache file path for a model name.
    ///
    /// The file may or may not exist; use [`has`](Self::has) to check.
    pub fn path(&self, name: &str) -> PathBuf {
        self.cache_dir.join(name)
    }

    /// Return the cache file path for a [`ModelInfo`], using the appropriate
    /// file extension for the weights format.
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
    pub fn store(&self, name: &str, data: &[u8]) -> FerrotorchResult<()> {
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
                    message: format!("failed to create cache subdir {:?}: {e}", parent),
                })?;
            }
        }
        std::fs::write(&path, data).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to write cache file {:?}: {e}", path),
        })?;
        Ok(())
    }

    /// Load cached weights as raw bytes.
    pub fn load(&self, name: &str) -> FerrotorchResult<Vec<u8>> {
        let path = self.path(name);
        std::fs::read(&path).map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("failed to read cache file {:?}: {e}", path),
        })
    }

    /// Remove all files from the cache directory.
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
}
