//! DeepLabV3 with ResNet-50 backbone.
//!
//! Mirrors `torchvision.models.segmentation.deeplabv3_resnet50` (torchvision 0.21.x).
//!
//! ## Architecture
//!
//! ```text
//! image [B, 3, H, W]
//!   └─ ResNet-50 backbone (replace_stride_with_dilation=[False, True, True])
//!        layer3: dilation=2  (output stride 16 total from input)
//!        layer4: dilation=4  (output stride 16 — stride=1, keeps spatial dims)
//!         └─ features [B, 2048, H/16, W/16]
//!              └─ ASPP head → [B, 256, H/16, W/16]
//!                   └─ 3×3 conv → BN → ReLU
//!                        └─ 1×1 classifier → [B, num_classes, H/16, W/16]
//!                             └─ bilinear upsample → [B, num_classes, H, W]
//! ```
//!
//! Phase 9 follow-up (#1011): the backbone is now a thin wrapper around
//! [`crate::models::resnet::resnet50_dilated`] with
//! `replace_stride_with_dilation=[false, true, true]`. The previous
//! hand-rolled `ResNet50Dilated` used uniform dilation across blocks within a
//! stage; torchvision's `ResNet._make_layer` threads `previous_dilation` so
//! the *first* block in a dilated stage carries the prior stage's dilation
//! while subsequent blocks carry the new one. Concretely:
//!
//! ```text
//! layer3[0]: dilation=1   (previous_dilation; before stage's ×2 update)
//! layer3[1..]: dilation=2
//! layer4[0]: dilation=2   (previous_dilation; before stage's ×4 update)
//! layer4[1..]: dilation=4
//! ```
//!
//! Reusing `resnet50_dilated` (which already implements the threading
//! correctly — `fcn_resnet50_value_parity` is the binding regression test)
//! also unifies the ResNet path: layer3/layer4 now use the same standard
//! `Bottleneck` blocks as the rest of the network, exposing native
//! `bn2.<...>` keys (no more `conv2.bn.<...>` divergence to translate
//! test-side).
//!
//! ## Reference
//! Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation",
//! arXiv:1706.05587. torchvision 0.21.x `deeplabv3_resnet50(weights=None,
//! num_classes=21)`.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::Conv2d;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::upsample::{InterpolateMode, interpolate};

use super::aspp::Aspp;
use crate::models::feature_extractor::IntermediateFeatures;
use crate::models::resnet::{ResNet, resnet50_dilated};

// ---------------------------------------------------------------------------
// ResNet50Dilated — ResNet-50 backbone with dilated layer3 + layer4
// ---------------------------------------------------------------------------

/// ResNet-50 backbone with dilated convolutions in layer3 and layer4.
///
/// Phase 9 follow-up (#1011): this is now a thin wrapper around
/// [`resnet50_dilated`] with `replace_stride_with_dilation=[false, true, true]`.
/// The torchvision `_make_layer` threading of `previous_dilation` is therefore
/// inherited verbatim:
///
/// - layer1: 3 standard bottlenecks (stride=1), output [B, 256, H/4, W/4]
/// - layer2: 4 standard bottlenecks (stride=2), output [B, 512, H/8, W/8]
/// - layer3: 6 bottlenecks; block 0 dilation=1 (previous_dilation), blocks 1..5
///   dilation=2; output [B, 1024, H/16, W/16]
/// - layer4: 3 bottlenecks; block 0 dilation=2 (previous_dilation), blocks 1..2
///   dilation=4; output [B, 2048, H/16, W/16]
///
/// layer3 and layer4 keep the same spatial resolution as layer2 because
/// stride is replaced by dilation. The output stride is effectively 16.
///
/// The classifier head (`fc`) on the inner `ResNet` is unused: only layer4
/// activations are extracted via [`IntermediateFeatures::forward_features`].
/// Both `parameters()` and `named_parameters()` filter the `fc.*` keys so the
/// strict loader's view matches torchvision's
/// `IntermediateLayerGetter`-wrapped backbone schema (no `backbone.fc.*`).
pub struct ResNet50Dilated<T: Float> {
    inner: ResNet<T>,
    training: bool,
}

impl<T: Float> ResNet50Dilated<T> {
    /// Build a dilated ResNet-50 backbone (no classifier head exposed).
    pub fn new() -> FerrotorchResult<Self> {
        let inner = resnet50_dilated::<T>(1000, [false, true, true])?;
        Ok(Self {
            inner,
            training: false, // torchvision starts in eval mode
        })
    }

    /// Extract layer4 features: `[B, 2048, H/16, W/16]`.
    ///
    /// Routes through `IntermediateFeatures::forward_features` and pulls the
    /// `"layer4"` activation, mirroring torchvision's
    /// `IntermediateLayerGetter(return_layers={"layer4": "out"})` plumbing
    /// inside `deeplabv3_resnet50`.
    pub fn forward_layer4(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let all_features = self.inner.forward_features(input)?;
        all_features.get("layer4").cloned().ok_or_else(|| {
            ferrotorch_core::FerrotorchError::Internal {
                message: "ResNet50Dilated: backbone did not produce 'layer4' features".into(),
            }
        })
    }
}

impl<T: Float> Module<T> for ResNet50Dilated<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // For use as a pure feature extractor: return layer4 activations.
        // (DeepLabV3 does not use the avgpool/fc head.)
        self.forward_layer4(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        // Filter out the unused `fc.*` head — torchvision's
        // `deeplabv3_resnet50` wraps the backbone in
        // `IntermediateLayerGetter`, which strips avgpool + fc. The
        // state dict therefore has NO `backbone.fc.*` keys. Mirror that
        // here so the loader's view matches.
        self.inner
            .named_parameters()
            .into_iter()
            .filter_map(|(n, p)| if n.starts_with("fc.") { None } else { Some(p) })
            .collect()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        // Order MUST match `named_parameters()` — the value-parity loader
        // does `.zip(parameters_mut())` against
        // `named_parameters().map(name)`. Collect names first via an
        // immutable borrow, then index into `parameters_mut()` by position.
        let names: Vec<String> = self
            .inner
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        self.inner
            .parameters_mut()
            .into_iter()
            .zip(names.iter())
            .filter_map(|(p, n)| if n.starts_with("fc.") { None } else { Some(p) })
            .collect()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.inner
            .named_parameters()
            .into_iter()
            .filter(|(n, _)| !n.starts_with("fc."))
            .map(|(n, p)| (format!("backbone.{n}"), p))
            .collect()
    }

    // Phase 4 (#995): expose stem + every residual block under
    // `backbone.<...>` so the strict loader's `named_descendants_dyn()`
    // walk reaches every BN. We forward the inner `ResNet`'s
    // `named_children` and re-prefix with `backbone.`. The `fc` and
    // `avgpool` children carry no relevant keys for the head; they are
    // filtered to keep the backbone view clean (avgpool has no params,
    // fc is excluded above).
    fn children(&self) -> Vec<&dyn Module<T>> {
        // Direct children of the wrapper: just the inner ResNet. The
        // strict loader walks `named_children` recursively, so prefixing
        // happens via `named_children` below.
        vec![&self.inner]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        // Empty path: `backbone.<...>` prefix is materialised in
        // `named_parameters` directly. We expose the inner ResNet under
        // the path `backbone` so a recursive walk produces
        // `backbone.<inner-child>`.
        vec![("backbone".to_string(), &self.inner)]
    }

    fn train(&mut self) {
        self.training = true;
        self.inner.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.inner.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// DeepLabV3Head
// ---------------------------------------------------------------------------

/// DeepLabV3 segmentation head.
///
/// Input: `[B, 2048, H', W']`
/// Output: `[B, num_classes, H', W']`
///
/// Structurally a 5-element `Sequential` mirroring torchvision's
/// `DeepLabHead`:
/// ```text
///   index 0: ASPP                                  (`aspp`)
///   index 1: Conv2d(256, 256, 3×3, pad=1, bias=F)  (`conv_intermediate`)
///   index 2: BatchNorm2d(256, eps=1e-5, mom=0.1)   (`bn_intermediate`)
///   index 3: ReLU                                  (inline in `forward`)
///   index 4: Conv2d(256, N, 1×1, bias=T)           (`classifier`)
/// ```
///
/// Phase 9 (#1009 / #1006): the head was previously a 2-element
/// `[ASPP, Conv2d(bias=False)]` Sequential. The new layout matches
/// torchvision's `DeepLabHead(in_channels, num_classes, atrous_rates=(12,
/// 24, 36))` exactly so the strict loader can adopt torchvision state
/// dicts via the test-side `remap_torchvision_to_ferrotorch_deeplabv3_keys`
/// translator.
pub struct DeepLabV3Head<T: Float> {
    aspp: Aspp<T>,
    /// 3×3 intermediate refinement conv (256→256, bias=False).
    conv_intermediate: Conv2d<T>,
    /// BN immediately after the intermediate conv.
    bn_intermediate: BatchNorm2d<T>,
    /// Final 1×1 classifier (256→num_classes, bias=True).
    classifier: Conv2d<T>,
    training: bool,
}

impl<T: Float> DeepLabV3Head<T> {
    /// Construct a DeepLabV3 head.
    ///
    /// * `in_channels` — backbone feature channels (2048).
    /// * `num_classes` — number of segmentation classes.
    /// * `atrous_rates` — three dilation rates for the dilated 3×3
    ///   branches inside ASPP. torchvision's `deeplabv3_resnet50` default
    ///   is `(12, 24, 36)`.
    pub fn new(
        in_channels: usize,
        num_classes: usize,
        atrous_rates: (usize, usize, usize),
    ) -> FerrotorchResult<Self> {
        let aspp = Aspp::new(in_channels, 256, atrous_rates)?;
        // 3×3 conv with same-size padding, bias=False (torchvision uses
        // bias=False because the BN that follows absorbs any constant).
        let conv_intermediate = Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), false)?;
        let bn_intermediate = BatchNorm2d::new(256, 1e-5, 0.1, true)?;
        // Final classifier 1×1 conv carries bias=True (last layer before
        // the upsample, matching torchvision).
        let classifier = Conv2d::new(256, num_classes, (1, 1), (1, 1), (0, 0), true)?;
        Ok(Self {
            aspp,
            conv_intermediate,
            bn_intermediate,
            classifier,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for DeepLabV3Head<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Phase 9 (#1009): aspp → conv_intermediate → bn_intermediate
        // → relu → classifier (5-element torchvision DeepLabHead).
        let x = self.aspp.forward(input)?;
        let x = self.conv_intermediate.forward(&x)?;
        let x = Module::<T>::forward(&self.bn_intermediate, &x)?;
        let x = relu(&x)?;
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.aspp.parameters();
        p.extend(self.conv_intermediate.parameters());
        p.extend(self.bn_intermediate.parameters());
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.aspp.parameters_mut();
        p.extend(self.conv_intermediate.parameters_mut());
        p.extend(self.bn_intermediate.parameters_mut());
        p.extend(self.classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.aspp.named_parameters() {
            out.push((format!("aspp.{k}"), v));
        }
        for (k, v) in self.conv_intermediate.named_parameters() {
            out.push((format!("conv_intermediate.{k}"), v));
        }
        for (k, v) in self.bn_intermediate.named_parameters() {
            out.push((format!("bn_intermediate.{k}"), v));
        }
        for (k, v) in self.classifier.named_parameters() {
            out.push((format!("classifier.{k}"), v));
        }
        out
    }

    // Phase 4 (#995) / Phase 9 (#1009): expose all four parameter-bearing
    // children mirroring `named_parameters`. The torchvision Sequential
    // index slot 3 is occupied by ReLU (no params, no submodule here).
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.aspp,
            &self.conv_intermediate,
            &self.bn_intermediate,
            &self.classifier,
        ]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("aspp".to_string(), &self.aspp),
            ("conv_intermediate".to_string(), &self.conv_intermediate),
            ("bn_intermediate".to_string(), &self.bn_intermediate),
            ("classifier".to_string(), &self.classifier),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.aspp.train();
        self.bn_intermediate.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.aspp.eval();
        self.bn_intermediate.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// DeepLabV3
// ---------------------------------------------------------------------------

/// DeepLabV3 semantic segmentation model with ResNet-50 backbone.
///
/// Output shape: `[B, num_classes, H, W]` — same spatial size as input.
///
/// ## Usage
///
/// ```ignore
/// use ferrotorch_vision::models::segmentation::DeepLabV3;
/// let model = deeplabv3_resnet50::<f32>(21).unwrap();
/// let x = ferrotorch_core::randn(&[1, 3, 512, 512]).unwrap();
/// let logits = model.forward(&x).unwrap(); // [1, 21, 512, 512]
/// ```
pub struct DeepLabV3<T: Float> {
    backbone: ResNet50Dilated<T>,
    head: DeepLabV3Head<T>,
    training: bool,
}

impl<T: Float> DeepLabV3<T> {
    /// Construct a DeepLabV3 model with torchvision's default
    /// `deeplabv3_resnet50` atrous rates `(12, 24, 36)`.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        Self::with_atrous_rates(num_classes, (12, 24, 36))
    }

    /// Construct a DeepLabV3 model with explicit atrous rates.
    ///
    /// Phase 9 (#1009) plumbs the atrous rates through `Aspp::new` so
    /// callers can build either the torchvision `deeplabv3_resnet50`
    /// default `(12, 24, 36)` or DeepLabV3+'s smaller `(6, 12, 18)`.
    pub fn with_atrous_rates(
        num_classes: usize,
        atrous_rates: (usize, usize, usize),
    ) -> FerrotorchResult<Self> {
        let backbone = ResNet50Dilated::new()?;
        let head = DeepLabV3Head::new(2048, num_classes, atrous_rates)?;
        Ok(Self {
            backbone,
            head,
            training: false, // torchvision starts in eval mode
        })
    }

    /// Total learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

impl<T: Float> Module<T> for DeepLabV3<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];

        // Backbone: [B, 2048, H/16, W/16]
        let features = self.backbone.forward_layer4(input)?;

        // Head: [B, num_classes, H/16, W/16]
        let logits = self.head.forward(&features)?;

        // Upsample to input resolution: [B, num_classes, H, W]
        interpolate(
            &logits,
            Some([h_in, w_in]),
            None,
            InterpolateMode::Bilinear,
            false,
        )
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.backbone.parameters();
        p.extend(self.head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.backbone.parameters_mut();
        p.extend(self.head.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        // Backbone already prefixes each key with `backbone.<...>`.
        for (k, v) in self.backbone.named_parameters() {
            out.push((k, v));
        }
        for (k, v) in self.head.named_parameters() {
            out.push((format!("head.{k}"), v));
        }
        out
    }

    // Phase 4 (#995): expose backbone + head children. The backbone's own
    // `named_children` already wraps the inner ResNet under
    // `backbone`, so we forward both children at the empty top-level path
    // here and let the recursive walk concatenate prefixes.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.backbone, &self.head]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            (String::new(), &self.backbone),
            ("head".to_string(), &self.head),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.backbone.train();
        self.head.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.backbone.eval();
        self.head.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Build a DeepLabV3 segmentation model with ResNet-50 backbone.
///
/// Mirrors `torchvision.models.segmentation.deeplabv3_resnet50(weights=None,
/// num_classes=21)`.
///
/// No pretrained weights are loaded. Use `num_classes=21` for Pascal VOC,
/// `num_classes=21` for Cityscapes (commonly), or another value as needed.
pub fn deeplabv3_resnet50<T: Float>(num_classes: usize) -> FerrotorchResult<DeepLabV3<T>> {
    DeepLabV3::new(num_classes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::no_grad;

    fn tiny_rgb(b: usize, h: usize, w: usize) -> Tensor<f32> {
        use ferrotorch_core::randn;
        no_grad(|| randn(&[b, 3, h, w]).unwrap())
    }

    #[test]
    fn test_deeplabv3_output_shape_small() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let x = tiny_rgb(1, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 21, 32, 32]);
    }

    #[test]
    fn test_deeplabv3_output_shape_64x64() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let x = tiny_rgb(1, 64, 64);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 21, 64, 64]);
    }

    #[test]
    fn test_deeplabv3_batch_size_2() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let x = tiny_rgb(2, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[2, 21, 32, 32]);
    }

    #[test]
    fn test_deeplabv3_custom_num_classes() {
        let model = deeplabv3_resnet50::<f32>(5).unwrap();
        let x = tiny_rgb(1, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 5, 32, 32]);
    }

    #[test]
    fn test_deeplabv3_named_parameter_prefixes() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("backbone.")));
        assert!(names.iter().any(|n| n.starts_with("head.")));
    }

    #[test]
    fn test_deeplabv3_param_count_sanity() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let np = model.num_parameters();
        // torchvision deeplabv3_resnet50 is ~39.6M; we expect a comparable range.
        assert!(np > 30_000_000, "DeepLabV3 params too low: {np}");
    }

    #[test]
    fn test_deeplabv3_train_eval_toggle() {
        let mut model = deeplabv3_resnet50::<f32>(21).unwrap();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
    }

    /// #1142 regression lock: `named_descendants_dyn` paths for DeepLabV3
    /// must NOT start with a leading `.`.
    ///
    /// `DeepLabV3::named_children` exposes its `backbone: ResNet50Dilated`
    /// at path `""` (transparent wrapper) and `ResNet50Dilated::named_children`
    /// exposes its inner `ResNet` at path `"backbone"`. Pre-#1142 the
    /// descendant walker composed those as `"" + "." + "backbone"` =
    /// `".backbone"`, mismatching the safetensors keys (`backbone.bn1.X`)
    /// in `apply_bn_buffers_from_state_dict`. The loader's
    /// `path_to_module.get("backbone.bn1")` silently missed every
    /// backbone BN, leaving running stats at their default values and
    /// poisoning every downstream activation by 60-99×.
    #[test]
    fn deeplabv3_named_descendants_no_leading_dot() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let descendants: Vec<String> = model
            .named_descendants_dyn()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        // The very first child IS exposed at path `""` (the transparent
        // backbone wrapper). Inspect everything *under* the backbone for
        // leading dots — those are what the BN-buffer loader resolves
        // against the state-dict keys.
        let backbone_descendants: Vec<&String> = descendants
            .iter()
            .filter(|n| !n.is_empty())
            .collect();
        assert!(
            !backbone_descendants.is_empty(),
            "DeepLabV3 should expose backbone descendants",
        );
        for path in &backbone_descendants {
            assert!(
                !path.starts_with('.'),
                "named_descendants_dyn path '{}' starts with '.' — \
                 the transparent-wrapper branch in \
                 ferrotorch_nn::Module::named_descendants_dyn has \
                 regressed. This breaks BN-buffer loading for any \
                 model whose `backbone` is exposed at path `\"\"`.",
                path
            );
        }
        // Sanity: the canonical `backbone.layer1.0.bn1` path must be
        // reachable, since that's what `apply_bn_buffers_from_state_dict`
        // looks up.
        assert!(
            descendants.iter().any(|p| p == "backbone.layer1.0.bn1"),
            "missing canonical path 'backbone.layer1.0.bn1' in descendant \
             walk; got: {descendants:?}"
        );
    }
}
