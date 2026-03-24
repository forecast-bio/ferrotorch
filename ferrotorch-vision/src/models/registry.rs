use std::collections::HashMap;
use std::sync::{LazyLock, RwLock};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};
use ferrotorch_nn::Module;

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

    registry.register_model(
        "resnet18",
        Box::new(|_pretrained, num_classes| {
            let model = super::resnet::resnet18::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "resnet34",
        Box::new(|_pretrained, num_classes| {
            let model = super::resnet::resnet34::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "resnet50",
        Box::new(|_pretrained, num_classes| {
            let model = super::resnet::resnet50::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "vgg11",
        Box::new(|_pretrained, num_classes| {
            let model = super::vgg::vgg11::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "vgg16",
        Box::new(|_pretrained, num_classes| {
            let model = super::vgg::vgg16::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "vit_b_16",
        Box::new(|_pretrained, num_classes| {
            let model = super::vit::vit_b_16::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "efficientnet_b0",
        Box::new(|_pretrained, num_classes| {
            let model = super::efficientnet::efficientnet_b0::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "swin_tiny",
        Box::new(|_pretrained, num_classes| {
            let model = super::swin::swin_tiny::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "unet",
        Box::new(|_pretrained, num_classes| {
            let model = super::unet::unet::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "convnext_tiny",
        Box::new(|_pretrained, num_classes| {
            let model = super::convnext::convnext_tiny::<f32>(num_classes)?;
            Ok(Box::new(model))
        }),
    );

    registry.register_model(
        "yolo",
        Box::new(|_pretrained, num_classes| {
            let model = super::yolo::yolo::<f32>(num_classes)?;
            Ok(Box::new(model))
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
