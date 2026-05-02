use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_nn::Module;

/// Helper used by registry constructors: build an architecture, then if
/// `pretrained == true` look up `hub_name` in `ferrotorch_hub`, fetch
/// the cached or downloaded weights, and load them into the model via
/// `load_state_dict(strict=false)`.
///
/// Returns the same model as `build()` would when `pretrained == false`.
/// When `pretrained == true` and the hub lookup fails (unknown name,
/// network error, missing cache, SHA mismatch), the error is propagated
/// to the caller — failing loudly is the right behavior here, since the
/// caller explicitly asked for pretrained weights.
fn maybe_load_pretrained<T, F, M>(
    pretrained: bool,
    hub_name: &str,
    build: F,
) -> FerrotorchResult<Box<dyn Module<T>>>
where
    T: Float,
    F: FnOnce() -> FerrotorchResult<M>,
    M: Module<T> + 'static,
{
    let mut model = build()?;
    if pretrained {
        // Look up by hub name. If the hub doesn't know this model yet
        // (e.g. vit_b_16 isn't published), bubble up a helpful error.
        let info = ferrotorch_hub::registry::get_model_info(hub_name).ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: format!(
                    "ferrotorch-vision: pretrained=true was requested for '{hub_name}' \
                     but no entry exists in ferrotorch_hub::registry. Either pass \
                     pretrained=false, register the model in ferrotorch_hub, or load \
                     weights manually from a SafeTensors file."
                ),
            }
        })?;
        let cache = ferrotorch_hub::cache::HubCache::with_default_dir();
        let path = ferrotorch_hub::download::download_weights(info, &cache)?;
        let state_dict = match info.format {
            ferrotorch_hub::registry::WeightsFormat::SafeTensors => {
                ferrotorch_serialize::load_safetensors::<T>(&path)?
            }
            ferrotorch_hub::registry::WeightsFormat::FerrotorchStateDict => {
                ferrotorch_serialize::load_state_dict::<T>(&path)?
            }
        };
        // strict=false so missing keys (e.g. classifier head reshaped to
        // a custom num_classes) don't break loading the backbone.
        model.load_state_dict(&state_dict, false)?;
    }
    Ok(Box::new(model))
}

/// A factory function that constructs a model given `pretrained` and `num_classes`.
pub type ModelConstructor<T> =
    Box<dyn Fn(bool, usize) -> FerrotorchResult<Box<dyn Module<T>>> + Send + Sync>;

/// Registry of model constructors keyed by architecture name.
pub struct ModelRegistry<T: Float> {
    models: HashMap<String, ModelConstructor<T>>,
}

impl<T: Float> ModelRegistry<T> {
    /// Create an empty registry.
    fn new() -> Self {
        Self {
            models: HashMap::new(),
        }
    }

    /// Return a sorted list of all registered model names.
    pub fn list_models(&self) -> Vec<String> {
        let mut names: Vec<String> = self.models.keys().cloned().collect();
        names.sort();
        names
    }

    /// Construct a model by name.
    pub fn get_model(
        &self,
        name: &str,
        pretrained: bool,
        num_classes: usize,
    ) -> FerrotorchResult<Box<dyn Module<T>>> {
        let constructor =
            self.models
                .get(name)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "unknown model: \"{name}\". Available: {:?}",
                        self.list_models()
                    ),
                })?;
        constructor(pretrained, num_classes)
    }

    /// Register a custom model constructor under `name`.
    pub fn register_model(&mut self, name: impl Into<String>, constructor: ModelConstructor<T>) {
        self.models.insert(name.into(), constructor);
    }
}

// ---------------------------------------------------------------------------
// Default registry with real ResNet constructors.
// ---------------------------------------------------------------------------

fn default_registry() -> ModelRegistry<f32> {
    let mut registry = ModelRegistry::new();

    // Each constructor calls maybe_load_pretrained, which builds the
    // architecture and (when pretrained == true) downloads + verifies
    // weights via ferrotorch_hub and loads them with strict=false. The
    // string passed to the helper is the registry key in
    // ferrotorch_hub::registry; if the hub does not yet have an entry
    // for this architecture, the user gets a clear error pointing them
    // at how to fix it.
    registry.register_model(
        "resnet18",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "resnet18", || {
                super::resnet::resnet18::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "resnet34",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "resnet34", || {
                super::resnet::resnet34::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "resnet50",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "resnet50", || {
                super::resnet::resnet50::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "vgg11",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "vgg11", || {
                super::vgg::vgg11::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "vgg16",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "vgg16", || {
                super::vgg::vgg16::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "vit_b_16",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "vit_b_16", || {
                super::vit::vit_b_16::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "efficientnet_b0",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "efficientnet_b0", || {
                super::efficientnet::efficientnet_b0::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "swin_tiny",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "swin_tiny", || {
                super::swin::swin_tiny::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "unet",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "unet", || super::unet::unet::<f32>(num_classes))
        }),
    );

    registry.register_model(
        "convnext_tiny",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "convnext_tiny", || {
                super::convnext::convnext_tiny::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "yolo",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "yolo", || super::yolo::yolo::<f32>(num_classes))
        }),
    );

    // CL-436: MobileNetV2, MobileNetV3-Small, DenseNet-121, Inception v3.
    registry.register_model(
        "mobilenet_v2",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "mobilenet_v2", || {
                super::mobilenet::mobilenet_v2::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "mobilenet_v3_small",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "mobilenet_v3_small", || {
                super::mobilenet::mobilenet_v3_small::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "densenet121",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "densenet121", || {
                super::densenet::densenet121::<f32>(num_classes)
            })
        }),
    );

    registry.register_model(
        "inception_v3",
        Box::new(|pretrained, num_classes| {
            maybe_load_pretrained(pretrained, "inception_v3", || {
                super::inception::inception_v3::<f32>(num_classes)
            })
        }),
    );

    registry
}

/// Global model registry for `f32` models.
///
/// Use [`list_models`], [`get_model`], and [`register_model`] for
/// convenient access without having to lock this directly.
pub static REGISTRY: LazyLock<RwLock<ModelRegistry<f32>>> =
    LazyLock::new(|| RwLock::new(default_registry()));

// ---------------------------------------------------------------------------
// Convenience free functions that operate on the global f32 registry.
// ---------------------------------------------------------------------------

/// Return all registered model names from the global registry.
pub fn list_models() -> Vec<String> {
    REGISTRY
        .read()
        .expect("model registry lock poisoned")
        .list_models()
}

/// Construct an `f32` model by name from the global registry.
pub fn get_model(
    name: &str,
    pretrained: bool,
    num_classes: usize,
) -> FerrotorchResult<Box<dyn Module<f32>>> {
    REGISTRY
        .read()
        .expect("model registry lock poisoned")
        .get_model(name, pretrained, num_classes)
}

/// Register a custom `f32` model constructor in the global registry.
pub fn register_model(name: impl Into<String>, constructor: ModelConstructor<f32>) {
    REGISTRY
        .write()
        .expect("model registry lock poisoned")
        .register_model(name, constructor);
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::Tensor;
    use ferrotorch_nn::parameter::Parameter;

    /// Minimal module for testing the registry roundtrip.
    struct DummyModel {
        num_classes: usize,
        training: bool,
    }

    impl Module<f32> for DummyModel {
        fn forward(&self, input: &Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
            let _ = self.num_classes;
            Ok(input.clone())
        }

        fn parameters(&self) -> Vec<&Parameter<f32>> {
            vec![]
        }

        fn parameters_mut(&mut self) -> Vec<&mut Parameter<f32>> {
            vec![]
        }

        fn named_parameters(&self) -> Vec<(String, &Parameter<f32>)> {
            vec![]
        }

        fn train(&mut self) {
            self.training = true;
        }

        fn eval(&mut self) {
            self.training = false;
        }

        fn is_training(&self) -> bool {
            self.training
        }
    }

    #[test]
    fn test_list_models_contains_resnets() {
        let names = list_models();
        assert!(names.contains(&"resnet18".to_string()));
        assert!(names.contains(&"resnet34".to_string()));
        assert!(names.contains(&"resnet50".to_string()));
        assert!(names.contains(&"vgg11".to_string()));
        assert!(names.contains(&"vgg16".to_string()));
        assert!(names.contains(&"vit_b_16".to_string()));
        assert!(names.contains(&"efficientnet_b0".to_string()));
        assert!(names.contains(&"swin_tiny".to_string()));
        assert!(names.contains(&"convnext_tiny".to_string()));
        assert!(names.contains(&"unet".to_string()));
        assert!(names.contains(&"yolo".to_string()));
        assert!(names.contains(&"mobilenet_v2".to_string()));
        assert!(names.contains(&"mobilenet_v3_small".to_string()));
        assert!(names.contains(&"densenet121".to_string()));
        assert!(names.contains(&"inception_v3".to_string()));
    }

    #[test]
    fn test_list_models_is_sorted() {
        let names = list_models();
        let mut sorted = names.clone();
        sorted.sort();
        assert_eq!(names, sorted);
    }

    #[test]
    fn test_get_model_unknown_name_errors() {
        let result = get_model("nonexistent_model", false, 1000);
        let err = result.err().expect("should be an error");
        let msg = format!("{err}");
        assert!(msg.contains("unknown model"));
        assert!(msg.contains("nonexistent_model"));
    }

    #[test]
    fn test_get_model_resnet18_constructs_successfully() {
        let result = get_model("resnet18", false, 1000);
        assert!(result.is_ok(), "resnet18 should construct successfully");
        let model = result.unwrap();
        assert!(!model.parameters().is_empty());
    }

    #[test]
    fn test_pretrained_lookup_known_model_in_hub() {
        // Every canonical vision architecture must have a matching entry
        // in the hub registry, so requesting pretrained=true at least
        // gets past the get_model_info check. We hard-code the expected
        // architecture names here instead of iterating list_models()
        // because other tests register dummy models into the global
        // REGISTRY (it's a process-wide LazyLock<RwLock>) and we don't
        // want test ordering to affect this assertion.
        let canonical = [
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
        for name in canonical {
            let info = ferrotorch_hub::registry::get_model_info(name);
            assert!(
                info.is_some(),
                "vision model '{name}' has no entry in ferrotorch_hub::registry; \
                 add one in ferrotorch-hub/src/registry.rs so pretrained=true works"
            );
        }
    }

    #[test]
    fn test_pretrained_false_constructs_without_network() {
        // pretrained=false must never touch the network or hub cache.
        // We can't easily prove "no network" here, but we can verify
        // the path completes successfully for every registered model.
        for name in list_models() {
            let result = get_model(&name, false, 10);
            assert!(
                result.is_ok(),
                "model '{name}' failed to construct with pretrained=false: {:?}",
                result.err()
            );
        }
    }

    #[test]
    fn test_register_and_get_model_roundtrip() {
        // Register a custom model.
        register_model(
            "dummy_test_model",
            Box::new(|_pretrained, num_classes| {
                Ok(Box::new(DummyModel {
                    num_classes,
                    training: true,
                }))
            }),
        );

        // Verify it shows up in the listing.
        let names = list_models();
        assert!(names.contains(&"dummy_test_model".to_string()));

        // Verify we can construct it.
        let model = get_model("dummy_test_model", false, 10).unwrap();
        assert!(model.is_training());
    }
}
