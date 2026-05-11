//! Lite R-ASPP (LRASPP) with MobileNetV3-Large dilated backbone.
//!
//! Mirrors `torchvision.models.segmentation.lraspp_mobilenet_v3_large`
//! (torchvision 0.21.x). #1146 — Phase A.4 of real-artifact-driven
//! development.
//!
//! ## Architecture
//!
//! ```text
//! image [B, 3, H, W]
//!   └─ MobileNetV3-Large dilated backbone
//!        (replace_stride_with_dilation=[True, True, True])
//!        features.4 → low  [B, 40,  H/8,  W/8]
//!        features.16 → high [B, 960, H/16, W/16]
//!        └─ LRASPP head
//!             ├─ cbr:   Conv(960→128, 1×1) → BN → ReLU            → out_high
//!             ├─ scale: AdaptiveAvgPool2d(1) → Conv(960→128, 1×1)
//!             │         → Sigmoid                                  → gate (1×1)
//!             ├─ out = out_high * bilinear_upsample(gate, low's H,W)
//!             │         (NB: torchvision interpolates the GATE TO HIGH's
//!             │          H,W first, multiplies, then interpolates the
//!             │          product to low's H,W with the high_classifier).
//!             │         See discussion at code site.
//!             ├─ low_classifier: Conv(40 → num_classes, 1×1)       → out_low_cls
//!             └─ high_classifier(bilinear_upsample(out, low's H,W)) → out_high_cls
//!                  └─ result = out_low_cls + out_high_cls          → [B, K, H/8, W/8]
//!                       └─ bilinear_upsample to (H, W)              → [B, K, H, W]
//! ```
//!
//! ## Reference
//!
//! Howard et al., "Searching for MobileNetV3", ICCV 2019 — the Lite
//! R-ASPP head is described in section 6 of the paper as a lighter-weight
//! alternative to R-ASPP / ASPP for mobile-friendly segmentation. The
//! torchvision implementation is in `torchvision/models/segmentation/lraspp.py`.
//!
//! The COCO_WITH_VOC_LABELS_V1 pretrained checkpoint targets the 21-class
//! Pascal VOC label set (background + 20 foreground classes).

use ferrotorch_core::grad_fns::activation::sigmoid;
use ferrotorch_core::grad_fns::arithmetic::{add, mul};
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::activation::ReLU;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::AdaptiveAvgPool2d;
use ferrotorch_nn::upsample::{InterpolateMode, interpolate};
use ferrotorch_nn::Conv2d;

use crate::models::mobilenet::{MobileNetV3Large, MobileNetV3LargeStaged};

// ---------------------------------------------------------------------------
// LrasppHead
// ---------------------------------------------------------------------------

/// LRASPP segmentation head.
///
/// Takes the two-scale backbone features `(low, high)` and produces dense
/// classification logits at `low`'s spatial resolution.
///
/// torchvision's `LRASPPHead`:
/// ```text
///   self.cbr   = Sequential(Conv2d(high_C, 128, 1, bias=F), BatchNorm2d(128), ReLU)
///   self.scale = Sequential(AdaptiveAvgPool2d(1), Conv2d(high_C, 128, 1, bias=F), Sigmoid)
///   self.low_classifier  = Conv2d(low_C,  num_classes, 1)        # bias=True
///   self.high_classifier = Conv2d(128,    num_classes, 1)        # bias=True
/// ```
///
/// torchvision's forward:
/// ```python
/// x_low, x_high = features["low"], features["high"]
/// x = self.cbr(x_high)                          # [B, 128, H/16, W/16]
/// s = self.scale(x_high)                        # [B, 128, 1, 1]
/// x = x * s                                     # broadcasting multiply
/// x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
/// return self.low_classifier(x_low) + self.high_classifier(x)
/// ```
///
/// Channel dimensions are fixed by the torchvision config:
/// - high input channels: 960 (MobileNetV3-Large features.16 output)
/// - low input channels: 40  (MobileNetV3-Large features.4 output)
/// - inter channels: 128
pub struct LrasppHead<T: Float> {
    /// `cbr.0` — 960→128 1×1 conv, bias=False.
    cbr_conv: Conv2d<T>,
    /// `cbr.1` — BN over the 128-ch cbr output.
    cbr_bn: BatchNorm2d<T>,
    // `cbr.2` — ReLU is parameter-free; applied inline.
    /// `scale.0` — AdaptiveAvgPool2d(1).
    scale_pool: AdaptiveAvgPool2d,
    /// `scale.1` — 960→128 1×1 conv, bias=False.
    scale_conv: Conv2d<T>,
    // `scale.2` — Sigmoid is parameter-free; applied inline.
    /// 1×1 conv from low-feature channels (40) to num_classes (bias=True).
    low_classifier: Conv2d<T>,
    /// 1×1 conv from inter channels (128) to num_classes (bias=True).
    high_classifier: Conv2d<T>,
    training: bool,
}

impl<T: Float> LrasppHead<T> {
    /// Construct an LRASPP head.
    ///
    /// * `low_channels`  — channels in the `low` backbone feature (40 for
    ///   MobileNetV3-Large).
    /// * `high_channels` — channels in the `high` backbone feature (960
    ///   for MobileNetV3-Large).
    /// * `num_classes`   — number of segmentation output classes.
    /// * `inter_channels` — width of the cbr/scale branch (128 in
    ///   torchvision).
    pub fn new(
        low_channels: usize,
        high_channels: usize,
        num_classes: usize,
        inter_channels: usize,
    ) -> FerrotorchResult<Self> {
        // Both branch convolutions are bias=False (the BN absorbs any
        // constant). torchvision's Conv2dNormActivation pattern.
        let cbr_conv = Conv2d::new(high_channels, inter_channels, (1, 1), (1, 1), (0, 0), false)?;
        // torchvision uses default BN eps=1e-5 momentum=0.1 in lraspp.py
        // (no override — the head module uses bare `nn.BatchNorm2d`).
        let cbr_bn = BatchNorm2d::new(inter_channels, 1e-5, 0.1, true)?;
        let scale_pool = AdaptiveAvgPool2d::new((1, 1));
        let scale_conv =
            Conv2d::new(high_channels, inter_channels, (1, 1), (1, 1), (0, 0), false)?;
        // Final classifiers carry bias=True (torchvision Conv2d default).
        let low_classifier =
            Conv2d::new(low_channels, num_classes, (1, 1), (1, 1), (0, 0), true)?;
        let high_classifier =
            Conv2d::new(inter_channels, num_classes, (1, 1), (1, 1), (0, 0), true)?;
        Ok(Self {
            cbr_conv,
            cbr_bn,
            scale_pool,
            scale_conv,
            low_classifier,
            high_classifier,
            training: false, // torchvision starts in eval mode
        })
    }

    /// LRASPP head forward.
    ///
    /// Returns `[B, num_classes, low.H, low.W]` — caller is responsible
    /// for the final bilinear upsample to the input spatial size.
    pub fn forward_features(
        &self,
        low: &Tensor<T>,
        high: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>> {
        // cbr branch.
        let x_cbr = self.cbr_conv.forward(high)?;
        let x_cbr = Module::<T>::forward(&self.cbr_bn, &x_cbr)?;
        let x_cbr = ReLU::new().forward(&x_cbr)?;

        // scale branch.
        let s = Module::<T>::forward(&self.scale_pool, high)?;
        let s = self.scale_conv.forward(&s)?;
        let s = sigmoid(&s)?;

        // Broadcast-multiply: x_cbr [B, 128, H/16, W/16] * s [B, 128, 1, 1].
        // ferrotorch's `mul` already supports NumPy-style broadcasting
        // (validated by the SqueezeExcitation block's `scale * input`
        // pattern in `ferrotorch-nn::se`).
        let x = mul(&x_cbr, &s)?;

        // Upsample (x * scale) → low's spatial dims.
        let low_shape = low.shape().to_vec();
        let low_h = low_shape[2];
        let low_w = low_shape[3];
        let x_up = interpolate(
            &x,
            Some([low_h, low_w]),
            None,
            InterpolateMode::Bilinear,
            false,
        )?;

        // Sum the low-resolution + high-resolution branches at low's
        // resolution.
        let low_logits = self.low_classifier.forward(low)?;
        let high_logits = self.high_classifier.forward(&x_up)?;
        add(&low_logits, &high_logits)
    }
}

impl<T: Float> Module<T> for LrasppHead<T> {
    fn forward(&self, _input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // LRASPP head consumes a TWO-tensor `(low, high)` pair, not a
        // single feature map. The single-input `Module::forward` contract
        // makes no sense here; callers must use `forward_features(low,
        // high)`. Surfacing this as an error matches the convention used
        // by `RoIHeads::forward` and other multi-input heads.
        Err(ferrotorch_core::FerrotorchError::InvalidArgument {
            message:
                "LrasppHead requires (low, high) features — call `forward_features(low, high)` \
                 instead of the single-input `Module::forward`".into(),
        })
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.cbr_conv.parameters());
        p.extend(self.cbr_bn.parameters());
        p.extend(self.scale_conv.parameters());
        p.extend(self.low_classifier.parameters());
        p.extend(self.high_classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.cbr_conv.parameters_mut());
        p.extend(self.cbr_bn.parameters_mut());
        p.extend(self.scale_conv.parameters_mut());
        p.extend(self.low_classifier.parameters_mut());
        p.extend(self.high_classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        // torchvision layout (from the state-dict dump):
        //   classifier.cbr.0.weight                     [128, 960, 1, 1]
        //   classifier.cbr.1.{weight,bias}              [128]
        //   classifier.scale.1.weight                   [128, 960, 1, 1]
        //   classifier.low_classifier.{weight,bias}     [21, 40, 1, 1]
        //   classifier.high_classifier.{weight,bias}    [21, 128, 1, 1]
        //
        // We expose those keys verbatim (sans the leading `classifier.`
        // prefix, which is added by the `Lraspp` wrapper). The
        // `cbr.{2}` Sequential slot (ReLU) and `scale.{0, 2}` slots
        // (AdaptiveAvgPool2d, Sigmoid) carry no parameters so they do
        // not appear.
        let mut p = Vec::new();
        for (n, param) in self.cbr_conv.named_parameters() {
            p.push((format!("cbr.0.{n}"), param));
        }
        for (n, param) in self.cbr_bn.named_parameters() {
            p.push((format!("cbr.1.{n}"), param));
        }
        for (n, param) in self.scale_conv.named_parameters() {
            p.push((format!("scale.1.{n}"), param));
        }
        for (n, param) in self.low_classifier.named_parameters() {
            p.push((format!("low_classifier.{n}"), param));
        }
        for (n, param) in self.high_classifier.named_parameters() {
            p.push((format!("high_classifier.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        // Expose every parameter-bearing child plus the parameter-free
        // primitives at their torchvision Sequential indices so the
        // strict loader's `named_descendants_dyn` walk can resolve
        // `cbr.0`, `cbr.1`, `scale.1`, `low_classifier`,
        // `high_classifier` — this is the same pattern used by
        // `DeepLabV3Head` (which also has parameter-free slot 3 = ReLU).
        vec![
            &self.cbr_conv,
            &self.cbr_bn,
            &self.scale_pool,
            &self.scale_conv,
            &self.low_classifier,
            &self.high_classifier,
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("cbr.0".to_string(), &self.cbr_conv as &dyn Module<T>),
            ("cbr.1".to_string(), &self.cbr_bn),
            ("scale.0".to_string(), &self.scale_pool),
            ("scale.1".to_string(), &self.scale_conv),
            ("low_classifier".to_string(), &self.low_classifier),
            ("high_classifier".to_string(), &self.high_classifier),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.cbr_bn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.cbr_bn.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Lraspp
// ---------------------------------------------------------------------------

/// LRASPP semantic segmentation model with MobileNetV3-Large dilated
/// backbone (`replace_stride_with_dilation=[True, True, True]`).
///
/// Output shape: `[B, num_classes, H, W]` — same spatial size as input.
///
/// ## Usage
///
/// ```ignore
/// use ferrotorch_vision::models::segmentation::Lraspp;
/// let model = lraspp_mobilenet_v3_large::<f32>(21).unwrap();
/// let x = ferrotorch_core::randn(&[1, 3, 520, 520]).unwrap();
/// let logits = model.forward(&x).unwrap(); // [1, 21, 520, 520]
/// ```
pub struct Lraspp<T: Float> {
    backbone: MobileNetV3Large<T>,
    head: LrasppHead<T>,
    training: bool,
}

impl<T: Float> Lraspp<T> {
    /// Construct an LRASPP model with MobileNetV3-Large dilated backbone.
    ///
    /// Channel widths are fixed to match the torchvision reference (low =
    /// 40, high = 960, inter = 128).
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let backbone = MobileNetV3Large::new_dilated(num_classes)?;
        let head = LrasppHead::new(40, 960, num_classes, 128)?;
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

    /// Per-stage diagnostic accessor (#1146): exposes the backbone's
    /// `(low, high)` feature pair without invoking the LRASPP head.
    /// Used by `examples/probe_lraspp_stages.rs` to localize parity
    /// failures by stage.
    pub fn backbone_forward_low_high(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        self.backbone.forward_low_high(input)
    }

    /// Per-block diagnostic accessor — see
    /// [`MobileNetV3Large::forward_with_block_dumps`].
    pub fn backbone_forward_with_block_dumps(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<MobileNetV3LargeStaged<T>> {
        self.backbone.forward_with_block_dumps(input)
    }
}

impl<T: Float> Module<T> for Lraspp<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];

        // Backbone → (low [B,40,H/8,W/8], high [B,960,H/16,W/16]).
        let (low, high) = self.backbone.forward_low_high(input)?;

        // Head → [B, num_classes, H/8, W/8].
        let logits_low_res = self.head.forward_features(&low, &high)?;

        // Final bilinear upsample to input resolution.
        interpolate(
            &logits_low_res,
            Some([h_in, w_in]),
            None,
            InterpolateMode::Bilinear,
            false,
        )
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        // ORDER MATTERS — torchvision's safetensors keys are emitted in
        // backbone-then-head order, so the strict value-parity loader
        // assumes `named_parameters()` yields backbone params first.
        //
        // We filter out the unused `classifier.*` Linear heads on the
        // inner `MobileNetV3Large` — torchvision's `lraspp_mobilenet_v3_large`
        // wraps the backbone in `IntermediateLayerGetter` which strips
        // the avgpool + classifier, so the LRASPP state-dict has NO
        // `backbone.classifier.{0,3}.*` keys. Mirror that here.
        //
        // Same pattern as `ResNet50Dilated` filters `fc.*` for DeepLabV3.
        let names: Vec<String> = self
            .backbone
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let mut p: Vec<&Parameter<T>> = self
            .backbone
            .parameters()
            .into_iter()
            .zip(names.iter())
            .filter_map(|(p, n)| if n.starts_with("classifier.") { None } else { Some(p) })
            .collect();
        p.extend(self.head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        // Collect names via immutable borrow first, then filter
        // `parameters_mut()` by position (mirroring `ResNet50Dilated`).
        let names: Vec<String> = self
            .backbone
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        let mut p: Vec<&mut Parameter<T>> = self
            .backbone
            .parameters_mut()
            .into_iter()
            .zip(names.iter())
            .filter_map(|(p, n)| if n.starts_with("classifier.") { None } else { Some(p) })
            .collect();
        p.extend(self.head.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        // torchvision `lraspp_mobilenet_v3_large` state-dict:
        //   backbone.<features.0...>  (MobileNetV3-Large features only)
        //   classifier.<...>          (LRASPPHead)
        //
        // The inner `MobileNetV3Large.classifier.{0,3}.*` Linear heads
        // are filtered out — see `parameters()` for rationale.
        let mut out = Vec::new();
        for (k, v) in self.backbone.named_parameters() {
            if k.starts_with("classifier.") {
                continue;
            }
            out.push((format!("backbone.{k}"), v));
        }
        for (k, v) in self.head.named_parameters() {
            out.push((format!("classifier.{k}"), v));
        }
        out
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.backbone, &self.head]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        // The backbone exposes its own children at `features.{0..16}`
        // and `classifier.{0,3}` (the classification head, unused at
        // forward time but kept structurally for state-dict parity).
        // We wrap that subtree under the path `backbone` so the strict
        // descendant walk yields `backbone.features.0`, ...,
        // `backbone.features.16`, plus `classifier.{cbr, scale,
        // low_classifier, high_classifier}` from the LRASPP head.
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

/// Build an LRASPP segmentation model with MobileNetV3-Large dilated
/// backbone.
///
/// Mirrors `torchvision.models.segmentation.lraspp_mobilenet_v3_large(
/// weights=None, num_classes=21)`. No pretrained weights are loaded —
/// use the registry path (`get_model("lraspp_mobilenet_v3_large", true,
/// 21)`) for that.
pub fn lraspp_mobilenet_v3_large<T: Float>(num_classes: usize) -> FerrotorchResult<Lraspp<T>> {
    Lraspp::new(num_classes)
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
    fn test_lraspp_output_shape_small() {
        let model = lraspp_mobilenet_v3_large::<f32>(21).unwrap();
        // Spatial dims must be multiples of 16 so the dilated stride-16
        // backbone produces non-zero feature maps. 32 is the minimum
        // workable here (block 6 stride 2 + 3 earlier stride-2's =
        // stride 8 even in the dilated variant, so >= 32 keeps
        // features.4 ≥ 4×4).
        let x = tiny_rgb(1, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 21, 32, 32]);
    }

    #[test]
    fn test_lraspp_output_shape_64x64() {
        let model = lraspp_mobilenet_v3_large::<f32>(21).unwrap();
        let x = tiny_rgb(1, 64, 64);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 21, 64, 64]);
    }

    #[test]
    fn test_lraspp_batch_size_2() {
        let model = lraspp_mobilenet_v3_large::<f32>(21).unwrap();
        let x = tiny_rgb(2, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[2, 21, 32, 32]);
    }

    #[test]
    fn test_lraspp_custom_num_classes() {
        let model = lraspp_mobilenet_v3_large::<f32>(5).unwrap();
        let x = tiny_rgb(1, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 5, 32, 32]);
    }

    #[test]
    fn test_lraspp_named_parameter_prefixes() {
        let model = lraspp_mobilenet_v3_large::<f32>(21).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        // Critical torchvision keys (verified against the actual
        // state-dict dump from
        // `lraspp_mobilenet_v3_large(weights='DEFAULT').state_dict()`).
        assert!(names.iter().any(|n| n == "backbone.features.0.0.weight"));
        assert!(names.iter().any(|n| n == "backbone.features.1.block.0.0.weight"));
        assert!(names.iter().any(|n| n == "backbone.features.16.0.weight"));
        assert!(names.iter().any(|n| n == "classifier.cbr.0.weight"));
        assert!(names.iter().any(|n| n == "classifier.cbr.1.weight"));
        assert!(names.iter().any(|n| n == "classifier.cbr.1.bias"));
        assert!(names.iter().any(|n| n == "classifier.scale.1.weight"));
        assert!(names.iter().any(|n| n == "classifier.low_classifier.weight"));
        assert!(names.iter().any(|n| n == "classifier.low_classifier.bias"));
        assert!(names.iter().any(|n| n == "classifier.high_classifier.weight"));
        assert!(names.iter().any(|n| n == "classifier.high_classifier.bias"));
    }

    #[test]
    fn test_lraspp_train_eval_toggle() {
        let mut model = lraspp_mobilenet_v3_large::<f32>(21).unwrap();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
    }
}
