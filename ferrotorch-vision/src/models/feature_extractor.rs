//! Intermediate feature extraction for vision models.
//!
//! `IntermediateFeatures` is an extension trait for vision models that
//! exposes their per-stage outputs as a `HashMap<String, Tensor<T>>`.
//! This is the building block for transfer learning, multi-scale
//! detection / segmentation heads, and feature visualization.
//!
//! `FeatureExtractor` wraps a model that implements `IntermediateFeatures`
//! and yields a filtered subset of those tensors keyed by user-supplied
//! return-node names. Mirrors
//! `torchvision.models.feature_extraction.create_feature_extractor`.
//!
//! Currently `IntermediateFeatures` is implemented for [`ResNet`]. The
//! pattern (override the standard forward to also stash intermediates
//! into a HashMap) is straightforward to extend to other architectures
//! as needed; see `resnet::ResNet::forward_features` for the template.

use std::collections::HashMap;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Module;

/// Trait for vision models that can return per-stage intermediate
/// activations in addition to the final output.
///
/// The string keys are stable, model-defined names for each stage. For
/// ResNet they are: `"stem"`, `"layer1"`, `"layer2"`, `"layer3"`,
/// `"layer4"`, `"avgpool"`, `"fc"` — matching torchvision's node names.
pub trait IntermediateFeatures<T: Float>: Module<T> {
    /// Run the forward pass and return all intermediate feature tensors
    /// keyed by stage name. The map always includes the final output
    /// under its conventional name (`"fc"` for ResNet).
    fn forward_features(&self, input: &Tensor<T>) -> FerrotorchResult<HashMap<String, Tensor<T>>>;

    /// List the available intermediate node names, in execution order.
    ///
    /// Returns `Vec<String>` (rather than `Vec<&'static str>`) so
    /// architectures with a variable number of blocks can produce
    /// per-block names like `"block0"`, `"block1"`, ... keyed off
    /// their runtime configuration. CL-499.
    fn feature_node_names(&self) -> Vec<String>;
}

/// Wraps a model that implements [`IntermediateFeatures`] and returns a
/// filtered subset of its intermediate outputs on each forward call.
///
/// # Example
///
/// ```rust,no_run
/// use ferrotorch_vision::models::resnet::resnet18;
/// use ferrotorch_vision::models::feature_extractor::FeatureExtractor;
///
/// let resnet = resnet18::<f32>(1000).unwrap();
/// let extractor = FeatureExtractor::new(
///     resnet,
///     vec!["layer3".to_string(), "layer4".to_string()],
/// ).unwrap();
///
/// // let features = extractor.forward(&input).unwrap();
/// // let layer3_feat = &features["layer3"];   // [B, 256, H/16, W/16]
/// // let layer4_feat = &features["layer4"];   // [B, 512, H/32, W/32]
/// ```
pub struct FeatureExtractor<T: Float, M: IntermediateFeatures<T>> {
    model: M,
    return_nodes: Vec<String>,
    _phantom: std::marker::PhantomData<T>,
}

impl<T: Float, M: IntermediateFeatures<T>> FeatureExtractor<T, M> {
    /// Create a new feature extractor.
    ///
    /// # Errors
    ///
    /// Returns an error if any name in `return_nodes` is not a valid
    /// node for the wrapped model. Use [`IntermediateFeatures::feature_node_names`]
    /// on the underlying model to discover the available names.
    pub fn new(model: M, return_nodes: Vec<String>) -> FerrotorchResult<Self> {
        let valid: Vec<String> = model.feature_node_names();
        for name in &return_nodes {
            if !valid.iter().any(|v| v == name) {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!(
                        "FeatureExtractor: '{name}' is not a valid feature node. \
                         Available nodes: {valid:?}"
                    ),
                });
            }
        }
        Ok(Self {
            model,
            return_nodes,
            _phantom: std::marker::PhantomData,
        })
    }

    /// Run the forward pass and return only the requested feature tensors.
    pub fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<HashMap<String, Tensor<T>>> {
        let all = self.model.forward_features(input)?;
        let mut filtered = HashMap::new();
        for name in &self.return_nodes {
            // Every key in return_nodes was validated at construction
            // time, so this lookup should always succeed; treat a miss
            // as an internal invariant violation.
            let t = all
                .get(name)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!(
                        "FeatureExtractor: forward_features did not produce expected node '{name}'"
                    ),
                })?;
            filtered.insert(name.clone(), t.clone());
        }
        Ok(filtered)
    }

    /// The requested return-node names.
    pub fn return_nodes(&self) -> &[String] {
        &self.return_nodes
    }

    /// Access the wrapped model.
    pub fn model(&self) -> &M {
        &self.model
    }

    /// Access the wrapped model mutably.
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }
}

/// Convenience constructor matching
/// `torchvision.models.feature_extraction.create_feature_extractor`.
///
/// Equivalent to [`FeatureExtractor::new`].
pub fn create_feature_extractor<T: Float, M: IntermediateFeatures<T>>(
    model: M,
    return_nodes: Vec<String>,
) -> FerrotorchResult<FeatureExtractor<T, M>> {
    FeatureExtractor::new(model, return_nodes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::resnet::resnet18;
    use ferrotorch_core::randn;

    #[test]
    fn test_resnet_intermediate_features_keys() {
        let resnet = resnet18::<f32>(1000).unwrap();
        let names = resnet.feature_node_names();
        assert!(names.iter().any(|n| n == "stem"));
        assert!(names.iter().any(|n| n == "layer1"));
        assert!(names.iter().any(|n| n == "layer2"));
        assert!(names.iter().any(|n| n == "layer3"));
        assert!(names.iter().any(|n| n == "layer4"));
        assert!(names.iter().any(|n| n == "avgpool"));
        assert!(names.iter().any(|n| n == "fc"));
    }

    #[test]
    fn test_feature_extractor_filters_to_requested_nodes() {
        let resnet = resnet18::<f32>(10).unwrap();
        let extractor =
            FeatureExtractor::new(resnet, vec!["layer3".to_string(), "layer4".to_string()])
                .unwrap();
        let input: Tensor<f32> = randn(&[1, 3, 32, 32]).unwrap();
        let features = extractor.forward(&input).unwrap();
        assert_eq!(features.len(), 2);
        assert!(features.contains_key("layer3"));
        assert!(features.contains_key("layer4"));
        // layer3 has 256 channels, layer4 has 512 channels (BasicBlock).
        assert_eq!(features["layer3"].shape()[1], 256);
        assert_eq!(features["layer4"].shape()[1], 512);
    }

    #[test]
    fn test_feature_extractor_rejects_unknown_node() {
        let resnet = resnet18::<f32>(10).unwrap();
        let result = FeatureExtractor::new(resnet, vec!["bogus_node".to_string()]);
        // Don't unwrap_err (which requires Debug on Ok type) — pattern match.
        let err_msg = match result {
            Ok(_) => panic!("expected error for unknown node"),
            Err(e) => format!("{e}"),
        };
        assert!(err_msg.contains("bogus_node"));
        assert!(err_msg.contains("Available nodes"));
    }

    #[test]
    fn test_feature_extractor_full_forward_features_includes_all_nodes() {
        let resnet = resnet18::<f32>(10).unwrap();
        let input: Tensor<f32> = randn(&[1, 3, 32, 32]).unwrap();
        let all = resnet.forward_features(&input).unwrap();
        // forward_features always returns every named stage.
        for name in resnet.feature_node_names() {
            assert!(all.contains_key(&name), "missing intermediate '{name}'");
        }
    }

    #[test]
    fn test_feature_extractor_final_output_matches_module_forward() {
        // The "fc" intermediate from forward_features must equal the
        // tensor returned by Module::forward.
        let resnet = resnet18::<f32>(10).unwrap();
        let input: Tensor<f32> = randn(&[1, 3, 32, 32]).unwrap();
        let module_out = Module::<f32>::forward(&resnet, &input).unwrap();
        let features = resnet.forward_features(&input).unwrap();
        let fc_out = features.get("fc").unwrap();
        assert_eq!(module_out.shape(), fc_out.shape());
        let m = module_out.data().unwrap();
        let f = fc_out.data().unwrap();
        for (a, b) in m.iter().zip(f.iter()) {
            assert!((a - b).abs() < 1e-5);
        }
    }

    #[test]
    fn test_feature_extractor_empty_return_nodes_returns_empty_map() {
        let resnet = resnet18::<f32>(10).unwrap();
        let extractor = FeatureExtractor::new(resnet, vec![]).unwrap();
        let input: Tensor<f32> = randn(&[1, 3, 32, 32]).unwrap();
        let features = extractor.forward(&input).unwrap();
        assert!(features.is_empty());
    }

    // CL-499: smoke tests for newly-added IntermediateFeatures impls.

    #[test]
    fn test_mobilenet_v2_feature_extractor_roundtrip() {
        use crate::models::mobilenet::mobilenet_v2;
        let model: crate::models::mobilenet::MobileNetV2<f32> = mobilenet_v2(10).unwrap();
        let names = model.feature_node_names();
        assert!(names.iter().any(|n| n == "stem"));
        assert!(names.iter().any(|n| n == "classifier"));
        // MobileNetV2 has 17 inverted-residual blocks.
        assert!(names.iter().any(|n| n == "block0"));
        assert!(names.iter().any(|n| n == "block16"));

        let extractor =
            FeatureExtractor::new(model, vec!["block0".into(), "block16".into()]).unwrap();
        let input: Tensor<f32> = randn(&[1, 3, 32, 32]).unwrap();
        let features = extractor.forward(&input).unwrap();
        assert_eq!(features.len(), 2);
        assert!(features.contains_key("block0"));
        assert!(features.contains_key("block16"));
    }

    #[test]
    fn test_densenet_feature_extractor_roundtrip() {
        use crate::models::densenet::densenet121;
        let model: crate::models::densenet::DenseNet<f32> = densenet121(10).unwrap();
        let extractor = FeatureExtractor::new(
            model,
            vec!["block1".into(), "trans2".into(), "block4".into()],
        )
        .unwrap();
        let input: Tensor<f32> = randn(&[1, 3, 32, 32]).unwrap();
        let features = extractor.forward(&input).unwrap();
        assert_eq!(features.len(), 3);
    }

    #[test]
    fn test_vgg11_feature_extractor_roundtrip() {
        use crate::models::vgg::vgg11;
        let model: crate::models::vgg::VGG<f32> = vgg11(10).unwrap();
        // Just verify feature_node_names is populated and one node
        // can be requested — we don't run forward to keep the test
        // fast.
        let names = model.feature_node_names();
        assert!(names.iter().any(|n| n == "avgpool"));
        let _extractor = FeatureExtractor::new(model, vec!["avgpool".into()]).unwrap();
    }

    #[test]
    fn test_yolo_feature_extractor_roundtrip() {
        use crate::models::yolo::yolo;
        let model: crate::models::yolo::Yolo<f32> = yolo(10).unwrap();
        let names = model.feature_node_names();
        assert_eq!(
            names,
            vec!["stage1", "stage2", "stage3", "stage4", "stage5", "head"]
                .into_iter()
                .map(String::from)
                .collect::<Vec<String>>()
        );
    }
}
