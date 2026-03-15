//! Download and load pretrained model weights.
//!
//! The current MVP does not include an HTTP client. Users must download weights
//! manually and place them in the cache directory. Future versions will use
//! `reqwest` or `ureq` for automatic downloads.

use std::path::PathBuf;

use ferrotorch_core::{Float, FerrotorchError, FerrotorchResult};
use ferrotorch_nn::StateDict;

use crate::cache::HubCache;
use crate::registry::{ModelInfo, WeightsFormat, get_model_info};

/// Resolve the cached weights file for a model.
///
/// If the weights are already cached, returns the file path. Otherwise returns
/// an error with instructions for manual download.
///
/// # Future
///
/// When an HTTP client dependency is added, this function will download
/// weights automatically and verify the SHA-256 checksum.
pub fn download_weights(info: &ModelInfo, cache: &HubCache) -> FerrotorchResult<PathBuf> {
    let path = cache.path_for_model(info);
    if path.exists() {
        return Ok(path);
    }

    // Also check the bare name (without extension) for backwards compatibility.
    let bare = cache.path(info.name);
    if bare.exists() {
        return Ok(bare);
    }

    Err(FerrotorchError::InvalidArgument {
        message: format!(
            "Pretrained weights for '{}' not found in cache at {:?}. \
             Download manually from {} and place at {:?}",
            info.name,
            cache.path_for_model(info),
            info.weights_url,
            cache.path_for_model(info),
        ),
    })
}

/// Load a pretrained model's state dict by name.
///
/// Looks up the model in the registry, checks the cache, and deserializes the
/// weights file into a [`StateDict`].
///
/// # Errors
///
/// Returns an error if:
/// - The model name is not in the registry.
/// - The weights are not cached (with instructions for manual download).
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_download_weights_returns_error_when_not_cached() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());

        let info = get_model_info("resnet50").unwrap();
        let result = download_weights(info, &cache);
        assert!(result.is_err());

        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("resnet50"),
            "error should mention model name, got: {err_msg}"
        );
        assert!(
            err_msg.contains("Download manually"),
            "error should include download instructions, got: {err_msg}"
        );
        assert!(
            err_msg.contains("huggingface.co"),
            "error should include the URL, got: {err_msg}"
        );
    }

    #[test]
    fn test_download_weights_returns_path_when_cached() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());

        let info = get_model_info("resnet50").unwrap();
        let expected_path = cache.path_for_model(info);

        // Simulate manual download by writing a file.
        std::fs::create_dir_all(dir.path()).unwrap();
        std::fs::write(&expected_path, b"fake weights").unwrap();

        let result = download_weights(info, &cache);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), expected_path);
    }

    #[test]
    fn test_download_weights_finds_bare_name() {
        let dir = tempfile::tempdir().unwrap();
        let cache = HubCache::new(dir.path());

        let info = get_model_info("resnet18").unwrap();

        // Write under the bare name (no extension).
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

    #[test]
    fn test_load_pretrained_not_cached() {
        // This should fail because no weights are cached in the default dir.
        let result = load_pretrained::<f32>("resnet50");
        assert!(result.is_err());
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("not found in cache") || err_msg.contains("Download manually"),
            "error should indicate missing cache, got: {err_msg}"
        );
    }
}
