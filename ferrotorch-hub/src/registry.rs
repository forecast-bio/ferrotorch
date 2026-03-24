//! Pretrained model registry.
//!
//! Contains metadata for known pretrained models. Model entries document where
//! weights come from (URLs), expected checksums, and parameter counts. The
//! registry is static and compiled into the binary.

/// Format of the serialized weights file.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightsFormat {
    /// HuggingFace SafeTensors format (`.safetensors`).
    SafeTensors,
    /// Native ferrotorch state dict format (`.fts`).
    FerrotorchStateDict,
}

/// Metadata for a pretrained model.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Short identifier (e.g. `"resnet50"`).
    pub name: &'static str,
    /// Human-readable description.
    pub description: &'static str,
    /// URL where the pretrained weights can be downloaded.
    pub weights_url: &'static str,
    /// SHA-256 hex digest of the weights file for integrity verification.
    pub weights_sha256: &'static str,
    /// Serialization format of the weights file.
    pub format: WeightsFormat,
    /// Total number of learnable parameters.
    pub num_parameters: usize,
}

/// Static registry of known pretrained models.
static MODELS: &[ModelInfo] = &[
    ModelInfo {
        name: "resnet18",
        description: "ResNet-18 trained on ImageNet-1K (top-1 acc ~69.8%)",
        weights_url: "https://huggingface.co/ferrotorch/resnet18/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 11_689_512,
    },
    ModelInfo {
        name: "resnet50",
        description: "ResNet-50 trained on ImageNet-1K (top-1 acc ~76.1%)",
        weights_url: "https://huggingface.co/ferrotorch/resnet50/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 25_557_032,
    },
    ModelInfo {
        name: "vgg16",
        description: "VGG-16 trained on ImageNet-1K (top-1 acc ~71.6%)",
        weights_url: "https://huggingface.co/ferrotorch/vgg16/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 138_357_544,
    },
    ModelInfo {
        name: "vit_b_16",
        description: "Vision Transformer (ViT-B/16) trained on ImageNet-1K (top-1 acc ~81.1%)",
        weights_url: "https://huggingface.co/ferrotorch/vit_b_16/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 86_567_656,
    },
];

/// List all available pretrained models.
pub fn list_models() -> Vec<&'static ModelInfo> {
    MODELS.iter().collect()
}

/// Get info for a specific model by name.
pub fn get_model_info(name: &str) -> Option<&'static ModelInfo> {
    MODELS.iter().find(|m| m.name == name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_list_models_non_empty() {
        let models = list_models();
        assert!(
            !models.is_empty(),
            "registry should contain at least one model"
        );
    }

    #[test]
    fn test_get_model_info_resnet50() {
        let info = get_model_info("resnet50");
        assert!(info.is_some(), "resnet50 should be in the registry");
        let info = info.unwrap();
        assert_eq!(info.name, "resnet50");
        assert_eq!(info.num_parameters, 25_557_032);
        assert_eq!(info.format, WeightsFormat::SafeTensors);
    }

    #[test]
    fn test_get_model_info_resnet18() {
        let info = get_model_info("resnet18");
        assert!(info.is_some());
        assert_eq!(info.unwrap().num_parameters, 11_689_512);
    }

    #[test]
    fn test_get_model_info_vgg16() {
        let info = get_model_info("vgg16");
        assert!(info.is_some());
        assert_eq!(info.unwrap().num_parameters, 138_357_544);
    }

    #[test]
    fn test_get_model_info_vit_b_16() {
        let info = get_model_info("vit_b_16");
        assert!(info.is_some());
        assert_eq!(info.unwrap().num_parameters, 86_567_656);
    }

    #[test]
    fn test_get_model_info_nonexistent() {
        assert!(get_model_info("nonexistent_model").is_none());
    }

    #[test]
    fn test_get_model_info_empty_string() {
        assert!(get_model_info("").is_none());
    }

    #[test]
    fn test_all_models_have_valid_fields() {
        for model in list_models() {
            assert!(!model.name.is_empty(), "model name must not be empty");
            assert!(
                !model.description.is_empty(),
                "model description must not be empty"
            );
            assert!(
                !model.weights_url.is_empty(),
                "model weights_url must not be empty"
            );
            assert_eq!(
                model.weights_sha256.len(),
                64,
                "SHA-256 hex digest must be 64 characters for model '{}'",
                model.name
            );
            assert!(
                model.num_parameters > 0,
                "model '{}' must have >0 parameters",
                model.name
            );
        }
    }
}
