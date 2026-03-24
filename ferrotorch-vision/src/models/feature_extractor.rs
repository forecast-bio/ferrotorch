use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Module;

/// Wraps a model to extract intermediate feature maps.
///
/// Currently implements a simplified version that returns the full model
/// output. Per-layer feature extraction requires model-internal hooks,
/// which will be added per-architecture (e.g., ResNet returns feature maps
/// at each stage).
pub struct FeatureExtractor<T: Float> {
    model: Box<dyn Module<T>>,
    return_nodes: Vec<String>,
}

impl<T: Float> FeatureExtractor<T> {
    /// Create a new feature extractor.
    ///
    /// * `model` — The underlying model to wrap.
    /// * `return_nodes` — Names of intermediate nodes whose features should
    ///   be captured. Stored for future per-model hook support; currently
    ///   only the final output is returned.
    pub fn new(model: Box<dyn Module<T>>, return_nodes: Vec<String>) -> Self {
        Self {
            model,
            return_nodes,
        }
    }

    /// Run the model and return the output tensor.
    ///
    /// In the future this will return a `HashMap<String, Tensor<T>>` keyed
    /// by `return_nodes`, once per-model hooks are wired up.
    pub fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.model.forward(input)
    }

    /// The requested return-node names.
    pub fn return_nodes(&self) -> &[String] {
        &self.return_nodes
    }

    /// Access the inner model.
    pub fn model(&self) -> &dyn Module<T> {
        &*self.model
    }

    /// Access the inner model mutably.
    pub fn model_mut(&mut self) -> &mut dyn Module<T> {
        &mut *self.model
    }
}

/// Convenience constructor matching `torchvision.models.feature_extraction.create_feature_extractor`.
pub fn create_feature_extractor<T: Float>(
    model: Box<dyn Module<T>>,
    return_nodes: Vec<String>,
) -> FeatureExtractor<T> {
    FeatureExtractor::new(model, return_nodes)
}
