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
///
/// Each entry uses the all-zero SHA-256 placeholder; the download path
/// detects the placeholder and skips integrity verification with a
/// warning. Real checksums should be pinned as the upstream weights are
/// published. The placeholder lets us wire the registry plumbing now
/// without blocking on every checkpoint being uploaded.
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
        name: "resnet34",
        description: "ResNet-34 trained on ImageNet-1K (top-1 acc ~73.3%)",
        weights_url: "https://huggingface.co/ferrotorch/resnet34/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 21_797_672,
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
        name: "vgg11",
        description: "VGG-11 trained on ImageNet-1K (top-1 acc ~69.0%)",
        weights_url: "https://huggingface.co/ferrotorch/vgg11/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 132_863_336,
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
    ModelInfo {
        name: "efficientnet_b0",
        description: "EfficientNet-B0 trained on ImageNet-1K (top-1 acc ~77.7%)",
        weights_url: "https://huggingface.co/ferrotorch/efficientnet_b0/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 5_288_548,
    },
    ModelInfo {
        name: "swin_tiny",
        description: "Swin Transformer Tiny trained on ImageNet-1K (top-1 acc ~81.2%)",
        weights_url: "https://huggingface.co/ferrotorch/swin_tiny/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 28_288_354,
    },
    ModelInfo {
        name: "convnext_tiny",
        description: "ConvNeXt Tiny trained on ImageNet-1K (top-1 acc ~82.1%)",
        weights_url: "https://huggingface.co/ferrotorch/convnext_tiny/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 28_589_128,
    },
    ModelInfo {
        name: "unet",
        description: "U-Net for semantic segmentation (Carvana / generic)",
        weights_url: "https://huggingface.co/ferrotorch/unet/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 31_037_633,
    },
    ModelInfo {
        name: "yolo",
        description: "YOLOv3 backbone for object detection (Darknet-53)",
        weights_url: "https://huggingface.co/ferrotorch/yolo/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 61_949_149,
    },
    // CL-436: MobileNetV2, MobileNetV3-Small, DenseNet-121, Inception v3.
    // Parameter counts reflect the *real* architectures from the original
    // papers — our simplified implementations will differ slightly.
    ModelInfo {
        name: "mobilenet_v2",
        description: "MobileNetV2 trained on ImageNet-1K (top-1 acc ~72.0%)",
        weights_url: "https://huggingface.co/ferrotorch/mobilenet_v2/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 3_504_872,
    },
    ModelInfo {
        name: "mobilenet_v3_small",
        description: "MobileNetV3-Small trained on ImageNet-1K (top-1 acc ~67.7%)",
        weights_url: "https://huggingface.co/ferrotorch/mobilenet_v3_small/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 2_542_856,
    },
    ModelInfo {
        name: "densenet121",
        description: "DenseNet-121 trained on ImageNet-1K (top-1 acc ~74.4%)",
        weights_url: "https://huggingface.co/ferrotorch/densenet121/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 7_978_856,
    },
    ModelInfo {
        name: "inception_v3",
        description: "Inception v3 trained on ImageNet-1K (top-1 acc ~77.5%)",
        weights_url: "https://huggingface.co/ferrotorch/inception_v3/resolve/main/model.safetensors",
        weights_sha256: "0000000000000000000000000000000000000000000000000000000000000000",
        format: WeightsFormat::SafeTensors,
        num_parameters: 27_161_264,
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
    fn test_registry_includes_all_vision_architectures() {
        // CL-385: every architecture exposed by ferrotorch_vision::models
        // should have an entry here so the vision registry's
        // pretrained=true path can resolve a download URL.
        let expected = [
            "resnet18",
            "resnet34",
            "resnet50",
            "vgg11",
            "vgg16",
            "vit_b_16",
            "efficientnet_b0",
            "swin_tiny",
            "convnext_tiny",
            "unet",
            "yolo",
            "mobilenet_v2",
            "mobilenet_v3_small",
            "densenet121",
            "inception_v3",
        ];
        for name in expected {
            assert!(
                get_model_info(name).is_some(),
                "ferrotorch_hub registry is missing entry for vision arch '{name}'"
            );
        }
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
