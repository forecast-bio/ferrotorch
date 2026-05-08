//! Fully Convolutional Network (FCN) with ResNet-50 backbone.
//!
//! Mirrors `torchvision.models.segmentation.fcn_resnet50` (torchvision 0.21.x).
//!
//! ## Architecture
//!
//! ```text
//! image [B, 3, H, W]
//!   └─ ResNet-50 backbone (replace_stride_with_dilation=[False, True, True])
//!        layer4 output: [B, 2048, H/8, W/8]
//!         └─ FCN head
//!              ├─ conv 3×3, 2048→512, BN, ReLU
//!              └─ conv 1×1, 512→num_classes
//!                   └─ bilinear upsample → [B, num_classes, H, W]
//! ```
//!
//! Phase 6 (#994): the backbone is now built via `resnet50_dilated([false,
//! true, true])`, dilating layer3 (×2) and layer4 (×4) instead of striding —
//! exactly matching torchvision's `fcn_resnet50` reference. The previous
//! non-dilated backbone produced a structural mismatch with the
//! torchvision state-dict layer3/4 conv2 dilation/padding entries, blocking
//! `fcn_resnet50_value_parity` (architect's pre-flight Movement 5).
//!
//! The FCN head is a simplified version of the torchvision `FCNHead` which
//! applies a 3×3 conv + BN + ReLU + dropout (p=0.1) + 1×1 conv.
//!
//! ## Reference
//! Shelhamer et al., "Fully Convolutional Networks for Semantic Segmentation",
//! CVPR 2015. torchvision 0.21.x `fcn_resnet50(weights=None, num_classes=21)`.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::upsample::{InterpolateMode, interpolate};
use ferrotorch_nn::{Conv2d, Dropout};

use crate::models::feature_extractor::IntermediateFeatures;
use crate::models::resnet::{ResNet, resnet50_dilated};

// ---------------------------------------------------------------------------
// FCNHead
// ---------------------------------------------------------------------------

/// FCN segmentation head.
///
/// Input:  `[B, in_channels, H', W']`
/// Output: `[B, num_classes, H', W']`
///
/// Matches torchvision `FCNHead`:
/// - 3×3 conv → BN → ReLU → Dropout(p=0.1) → 1×1 conv
pub struct FcnHead<T: Float> {
    conv: Conv2d<T>,
    bn: BatchNorm2d<T>,
    dropout: Dropout<T>,
    classifier: Conv2d<T>,
    training: bool,
}

impl<T: Float> FcnHead<T> {
    /// Construct an FCN head.
    ///
    /// * `in_channels`  — feature channels from backbone (2048 for ResNet-50).
    /// * `num_classes`  — number of segmentation output classes.
    pub fn new(in_channels: usize, num_classes: usize) -> FerrotorchResult<Self> {
        // torchvision FCNHead uses in_channels // 4 as intermediate width.
        let inter = in_channels / 4; // 2048/4 = 512
        let conv = Conv2d::new(in_channels, inter, (3, 3), (1, 1), (1, 1), false)?;
        let bn = BatchNorm2d::new(inter, 1e-5, 0.1, true)?;
        let dropout = Dropout::new(0.1)?;
        // torchvision's FCNHead final 1×1 classifier conv carries bias=True
        // (it's the last layer before the upsample, so the bias is read by
        // the loader's `classifier.4.bias` key). Phase 6 (#994) closed the
        // bias=False structural divergence here.
        let classifier = Conv2d::new(inter, num_classes, (1, 1), (1, 1), (0, 0), true)?;
        Ok(Self {
            conv,
            bn,
            dropout,
            classifier,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for FcnHead<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.conv.forward(input)?;
        let x = Module::<T>::forward(&self.bn, &x)?;
        let x = relu(&x)?;
        let x = if self.training {
            Module::<T>::forward(&self.dropout, &x)?
        } else {
            x
        };
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.conv.parameters();
        p.extend(self.bn.parameters());
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.conv.parameters_mut();
        p.extend(self.bn.parameters_mut());
        p.extend(self.classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.conv.named_parameters() {
            out.push((format!("0.{k}"), v));
        }
        for (k, v) in self.bn.named_parameters() {
            out.push((format!("1.{k}"), v));
        }
        for (k, v) in self.classifier.named_parameters() {
            out.push((format!("4.{k}"), v));
        }
        out
    }

    // Phase 4 (#995): expose direct children mirroring `named_parameters`.
    // The torchvision-shaped indices `0` / `1` / `4` are kept so the
    // path → module index agrees with the fixture descriptors.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.conv, &self.bn, &self.dropout, &self.classifier]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("0".to_string(), &self.conv),
            ("1".to_string(), &self.bn),
            ("3".to_string(), &self.dropout),
            ("4".to_string(), &self.classifier),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.conv.train();
        self.bn.train();
        self.classifier.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv.eval();
        self.bn.eval();
        self.classifier.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// FCN
// ---------------------------------------------------------------------------

/// FCN semantic segmentation model with ResNet-50 backbone.
///
/// Output shape: `[B, num_classes, H, W]` — same spatial size as input.
///
/// ## Usage
///
/// ```ignore
/// let model = fcn_resnet50::<f32>(21).unwrap();
/// let x = ferrotorch_core::randn(&[1, 3, 512, 512]).unwrap();
/// let logits = model.forward(&x).unwrap(); // [1, 21, 512, 512]
/// ```
pub struct Fcn<T: Float> {
    backbone: ResNet<T>,
    head: FcnHead<T>,
    training: bool,
}

impl<T: Float> Fcn<T> {
    /// Construct an FCN model.
    ///
    /// The backbone is a ResNet-50 with `replace_stride_with_dilation=[false,
    /// true, true]` (Phase 6 #994), exactly matching torchvision's
    /// `fcn_resnet50`. The fc head is unused; only the backbone feature
    /// stages are read.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let backbone = resnet50_dilated::<T>(1000, [false, true, true])?;
        let head = FcnHead::new(2048, num_classes)?;
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

impl<T: Float> Module<T> for Fcn<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];

        // Extract layer4 features via IntermediateFeatures: [B, 2048, H/32, W/32]
        let all_features = self.backbone.forward_features(input)?;
        let layer4 = all_features.get("layer4").ok_or_else(|| {
            ferrotorch_core::FerrotorchError::Internal {
                message: "FCN: backbone did not produce 'layer4' features".into(),
            }
        })?;

        // FCN head: [B, num_classes, H/32, W/32]
        let logits = self.head.forward(layer4)?;

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
        // torchvision's fcn_resnet50 wraps the ResNet-50 backbone in
        // IntermediateLayerGetter, which strips the final `fc` head
        // (avgpool + Linear). The backbone state dict therefore has NO
        // `backbone.fc.{weight,bias}` keys. We mirror that here by
        // filtering the fc params out of `parameters()` and the loader's
        // `parameters_mut()` view (Phase 6 #994).
        let mut out: Vec<&Parameter<T>> = self
            .backbone
            .named_parameters()
            .into_iter()
            .filter_map(|(n, p)| if n.starts_with("fc.") { None } else { Some(p) })
            .collect();
        out.extend(self.head.parameters());
        out
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        // Filter the backbone's `fc.*` params out (see `parameters()` above).
        // The order MUST match `named_parameters()` because the value-parity
        // loader does `.zip(parameters_mut())` against
        // `named_parameters().map(name)`.
        //
        // We collect the names first (immutable borrow), then index into
        // `parameters_mut()` by position so we can apply the same filter.
        let names: Vec<String> = self
            .backbone
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let backbone_params: Vec<&mut Parameter<T>> = self
            .backbone
            .parameters_mut()
            .into_iter()
            .zip(names.iter())
            .filter_map(|(p, n)| if n.starts_with("fc.") { None } else { Some(p) })
            .collect();
        let mut out = backbone_params;
        out.extend(self.head.parameters_mut());
        out
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.backbone.named_parameters() {
            // Strip the unused fc head — torchvision's fcn_resnet50 backbone
            // (wrapped in IntermediateLayerGetter) does the same, so the
            // state dict has no `backbone.fc.*` keys.
            if k.starts_with("fc.") {
                continue;
            }
            out.push((format!("backbone.{k}"), v));
        }
        // torchvision exposes the FCN head under the top-level path
        // `classifier.<{0,1,4}.<...>>` (FCNHead is `Sequential[Conv, BN,
        // ReLU, Dropout, Conv]`, with the ReLU+Dropout claiming index 2/3
        // and only Conv/BN/Conv carrying parameters). We mirror that
        // exactly so the strict loader can adopt torchvision state dicts
        // without remap.
        for (k, v) in self.head.named_parameters() {
            out.push((format!("classifier.{k}"), v));
        }
        out
    }

    // Phase 4 (#995): expose backbone + head children. The backbone's
    // own `named_children` does NOT prefix with `backbone.<...>`, so we
    // wrap it under `backbone` here to match `named_parameters` above.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.backbone, &self.head]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("backbone".to_string(), &self.backbone),
            ("classifier".to_string(), &self.head),
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

/// Build an FCN segmentation model with ResNet-50 backbone.
///
/// Mirrors `torchvision.models.segmentation.fcn_resnet50(weights=None,
/// num_classes=21)`.
///
/// No pretrained weights are loaded.
pub fn fcn_resnet50<T: Float>(num_classes: usize) -> FerrotorchResult<Fcn<T>> {
    Fcn::new(num_classes)
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
    fn test_fcn_output_shape_small() {
        let model = fcn_resnet50::<f32>(21).unwrap();
        let x = tiny_rgb(1, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 21, 32, 32]);
    }

    #[test]
    fn test_fcn_output_shape_64x64() {
        let model = fcn_resnet50::<f32>(21).unwrap();
        let x = tiny_rgb(1, 64, 64);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 21, 64, 64]);
    }

    #[test]
    fn test_fcn_batch_size_2() {
        let model = fcn_resnet50::<f32>(21).unwrap();
        let x = tiny_rgb(2, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[2, 21, 32, 32]);
    }

    #[test]
    fn test_fcn_custom_num_classes() {
        let model = fcn_resnet50::<f32>(5).unwrap();
        let x = tiny_rgb(1, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 5, 32, 32]);
    }

    #[test]
    fn test_fcn_named_parameter_prefixes() {
        let model = fcn_resnet50::<f32>(21).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("backbone.")));
        // Phase 6 (#994): top-level head prefix renamed `head.` →
        // `classifier.` to match torchvision fcn_resnet50.
        assert!(names.iter().any(|n| n.starts_with("classifier.")));
    }

    #[test]
    fn test_fcn_param_count_sanity() {
        let model = fcn_resnet50::<f32>(21).unwrap();
        let np = model.num_parameters();
        // torchvision fcn_resnet50 is ~32.9M; expect >30M
        assert!(np > 25_000_000, "FCN params too low: {np}");
    }

    #[test]
    fn test_fcn_train_eval_toggle() {
        let mut model = fcn_resnet50::<f32>(21).unwrap();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
    }
}
