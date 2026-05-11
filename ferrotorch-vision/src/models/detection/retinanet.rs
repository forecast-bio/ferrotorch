//! RetinaNet single-stage detector with ResNet-50 + FPN backbone.
//!
//! Mirrors `torchvision.models.detection.retinanet_resnet50_fpn`
//! (legacy `RetinaNet_ResNet50_FPN_Weights.COCO_V1` checkpoint).
//!
//! ## Architecture
//!
//! ```text
//! image [B, 3, H, W]
//!   └─ ResNet-50 backbone → {layer2, layer3, layer4} (C3-C5)
//!         └─ FPN (P3-P7)  → 5 levels at 256 ch each
//!               P3, P4, P5: standard FPN top-down/lateral
//!               P6 = 3x3 stride-2 Conv2d on P5      (NOT max-pool!)
//!               P7 = 3x3 stride-2 Conv2d on ReLU(P6)
//!         └─ Shared 4-conv classification head (per-level, no params per level)
//!         └─ Shared 4-conv regression head     (per-level, no params per level)
//!   ↳ AnchorGenerator: 9 anchors/cell (3 sizes × 3 aspect ratios) per level
//!   ↳ postprocess: sigmoid → score_thresh (0.05) → per-level top-K (1000)
//!                  → box decode → clip → cross-class batched_nms (IoU 0.5)
//!                  → detections_per_image (300)
//! ```
//!
//! Distinct from FasterRCNN's two-stage pipeline: there is NO RPN, NO ROI Align,
//! and class scoring is per-class **sigmoid** (focal-loss-trained), not softmax.
//!
//! ## Reference
//! Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017.
//! torchvision 0.21.x `retinanet_resnet50_fpn(weights="COCO_V1")`.

use std::collections::HashMap;

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv2d, InterpolateMode, interpolate};

use crate::models::detection::anchor_utils::decode_boxes;
use crate::models::detection::fpn::FPN_OUT_CHANNELS;
use crate::models::feature_extractor::IntermediateFeatures;
use crate::models::resnet::{ResNet, resnet50};
use crate::ops::{batched_nms, clip_boxes_to_image};

// ---------------------------------------------------------------------------
// Constants — mirror torchvision defaults
// ---------------------------------------------------------------------------

/// Number of anchors per spatial location (3 sizes × 3 aspect ratios).
pub const RETINANET_NUM_ANCHORS_PER_LOC: usize = 9;

/// Aspect ratios applied at every level.
pub const RETINANET_ASPECT_RATIOS: [f64; 3] = [0.5, 1.0, 2.0];

/// Base anchor sizes per level (P3..P7), mirroring `_default_anchorgen`.
pub const RETINANET_BASE_SIZES: [f64; 5] = [32.0, 64.0, 128.0, 256.0, 512.0];

/// Per-class score gate (matches `RetinaNet(score_thresh=0.05)`).
pub const RETINANET_SCORE_THRESH: f64 = 0.05;

/// NMS IoU threshold (matches `RetinaNet(nms_thresh=0.5)`).
pub const RETINANET_NMS_THRESH: f64 = 0.5;

/// Per-level top-K candidates pre-NMS (matches `RetinaNet(topk_candidates=1000)`).
pub const RETINANET_TOPK_CANDIDATES: usize = 1000;

/// Cross-class detection cap per image (matches `RetinaNet(detections_per_img=300)`).
pub const RETINANET_DETECTIONS_PER_IMG: usize = 300;

/// FPN strides for levels P3-P7.
const RETINANET_STRIDES: [usize; 5] = [8, 16, 32, 64, 128];

// ---------------------------------------------------------------------------
// FPN with P3-P7 levels and LastLevelP6P7 extras
// ---------------------------------------------------------------------------

/// Feature Pyramid Network specialized for RetinaNet.
///
/// Differences from `FeaturePyramidNetwork` (used by FasterRCNN):
/// - 3 lateral inputs (C3..C5 from `layer2..layer4`) instead of 4.
/// - Output levels P3..P7 instead of P2..P6.
/// - `LastLevelP6P7` extra block: P6 = 3×3 stride-2 conv on P5,
///   P7 = 3×3 stride-2 conv on ReLU(P6). NOT a max-pool.
///
/// Built as a separate type rather than parameterising the existing
/// `FeaturePyramidNetwork` so that the FasterRCNN/MaskRCNN FPN config and
/// pinned weights remain untouched.
pub struct RetinaFpn<T: Float> {
    // Lateral 1×1 convs (C3..C5).
    lateral3: Conv2d<T>,
    lateral4: Conv2d<T>,
    lateral5: Conv2d<T>,

    // 3×3 output convs (P3..P5).
    output3: Conv2d<T>,
    output4: Conv2d<T>,
    output5: Conv2d<T>,

    // LastLevelP6P7 extras.
    p6: Conv2d<T>,
    p7: Conv2d<T>,
}

impl<T: Float> RetinaFpn<T> {
    /// Build for ResNet-50 (channel counts `[512, 1024, 2048]` for C3..C5).
    pub fn new() -> FerrotorchResult<Self> {
        let in_channels = [512, 1024, 2048];
        let out_ch = FPN_OUT_CHANNELS;

        // bias=true: torchvision uses `nn.Conv2d(..., bias=True)` for both
        // lateral and output convs (Conv2dNormActivation with norm_layer=None
        // keeps default bias=True). See #1141 for the FasterRCNN-side
        // diagnosis.
        let lateral3 = Conv2d::new(in_channels[0], out_ch, (1, 1), (1, 1), (0, 0), true)?;
        let lateral4 = Conv2d::new(in_channels[1], out_ch, (1, 1), (1, 1), (0, 0), true)?;
        let lateral5 = Conv2d::new(in_channels[2], out_ch, (1, 1), (1, 1), (0, 0), true)?;

        let output3 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), true)?;
        let output4 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), true)?;
        let output5 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), true)?;

        // LastLevelP6P7: each is a 3×3 stride-2 conv with bias.
        // Both consume `out_ch` (256) inputs because `use_P5 = (in == out) = True`
        // in torchvision when both backbone.out_channels and the FPN out_channels
        // are 256.
        let p6 = Conv2d::new(out_ch, out_ch, (3, 3), (2, 2), (1, 1), true)?;
        let p7 = Conv2d::new(out_ch, out_ch, (3, 3), (2, 2), (1, 1), true)?;

        Ok(Self {
            lateral3,
            lateral4,
            lateral5,
            output3,
            output4,
            output5,
            p6,
            p7,
        })
    }

    /// Forward pass.
    ///
    /// Expects `backbone_features` to have keys `"layer2"` (C3), `"layer3"` (C4),
    /// `"layer4"` (C5) produced by `ResNet::forward_features`.
    ///
    /// Returns `HashMap` with keys `"p3"`..`"p7"`.
    pub fn forward(
        &self,
        backbone_features: &HashMap<String, Tensor<T>>,
    ) -> FerrotorchResult<HashMap<String, Tensor<T>>> {
        let c3 = backbone_features.get("layer2").ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: "RetinaFpn: backbone_features missing 'layer2' (C3)".into(),
            }
        })?;
        let c4 = backbone_features.get("layer3").ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: "RetinaFpn: backbone_features missing 'layer3' (C4)".into(),
            }
        })?;
        let c5 = backbone_features.get("layer4").ok_or_else(|| {
            FerrotorchError::InvalidArgument {
                message: "RetinaFpn: backbone_features missing 'layer4' (C5)".into(),
            }
        })?;

        // Laterals.
        let lat5 = self.lateral5.forward(c5)?;
        let lat4 = self.lateral4.forward(c4)?;
        let lat3 = self.lateral3.forward(c3)?;

        // Top-down: P5 = lat5; upsample-add into P4, P3.
        let p5_inner = lat5;

        let c4_shape = c4.shape();
        let p5_up = interpolate(
            &p5_inner,
            Some([c4_shape[2], c4_shape[3]]),
            None,
            InterpolateMode::Nearest,
            false,
        )?;
        let p4_inner = add(&p5_up, &lat4)?;

        let c3_shape = c3.shape();
        let p4_up = interpolate(
            &p4_inner,
            Some([c3_shape[2], c3_shape[3]]),
            None,
            InterpolateMode::Nearest,
            false,
        )?;
        let p3_inner = add(&p4_up, &lat3)?;

        // 3×3 output convs.
        let p3 = self.output3.forward(&p3_inner)?;
        let p4 = self.output4.forward(&p4_inner)?;
        let p5 = self.output5.forward(&p5_inner)?;

        // LastLevelP6P7 — input is P5 (use_P5 = (in_ch == out_ch) = True).
        let p6 = self.p6.forward(&p5)?;
        let p6_relu = relu(&p6)?;
        let p7 = self.p7.forward(&p6_relu)?;

        let mut out = HashMap::new();
        out.insert("p3".to_string(), p3);
        out.insert("p4".to_string(), p4);
        out.insert("p5".to_string(), p5);
        out.insert("p6".to_string(), p6);
        out.insert("p7".to_string(), p7);
        Ok(out)
    }

    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.lateral3.parameters());
        p.extend(self.lateral4.parameters());
        p.extend(self.lateral5.parameters());
        p.extend(self.output3.parameters());
        p.extend(self.output4.parameters());
        p.extend(self.output5.parameters());
        p.extend(self.p6.parameters());
        p.extend(self.p7.parameters());
        p
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.lateral3.parameters_mut());
        p.extend(self.lateral4.parameters_mut());
        p.extend(self.lateral5.parameters_mut());
        p.extend(self.output3.parameters_mut());
        p.extend(self.output4.parameters_mut());
        p.extend(self.output5.parameters_mut());
        p.extend(self.p6.parameters_mut());
        p.extend(self.p7.parameters_mut());
        p
    }

    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.lateral3.named_parameters() {
            out.push((format!("lateral3.{n}"), p));
        }
        for (n, p) in self.lateral4.named_parameters() {
            out.push((format!("lateral4.{n}"), p));
        }
        for (n, p) in self.lateral5.named_parameters() {
            out.push((format!("lateral5.{n}"), p));
        }
        for (n, p) in self.output3.named_parameters() {
            out.push((format!("output3.{n}"), p));
        }
        for (n, p) in self.output4.named_parameters() {
            out.push((format!("output4.{n}"), p));
        }
        for (n, p) in self.output5.named_parameters() {
            out.push((format!("output5.{n}"), p));
        }
        for (n, p) in self.p6.named_parameters() {
            out.push((format!("p6.{n}"), p));
        }
        for (n, p) in self.p7.named_parameters() {
            out.push((format!("p7.{n}"), p));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Shared classification / regression heads
// ---------------------------------------------------------------------------

/// Shared classification head: 4 × `Conv3×3+ReLU` then a final
/// `Conv3×3` outputting `num_anchors * num_classes` channels.
///
/// Mirrors `torchvision.models.detection.retinanet.RetinaNetClassificationHead`.
/// The same Conv weights are applied to every FPN level (parameter sharing).
pub struct RetinaNetClassificationHead<T: Float> {
    conv0: Conv2d<T>,
    conv1: Conv2d<T>,
    conv2: Conv2d<T>,
    conv3: Conv2d<T>,
    cls_logits: Conv2d<T>,
    num_anchors: usize,
    num_classes: usize,
}

impl<T: Float> RetinaNetClassificationHead<T> {
    pub fn new(
        in_channels: usize,
        num_anchors: usize,
        num_classes: usize,
    ) -> FerrotorchResult<Self> {
        let conv0 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv1 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv2 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv3 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let cls_logits = Conv2d::new(
            in_channels,
            num_anchors * num_classes,
            (3, 3),
            (1, 1),
            (1, 1),
            true,
        )?;
        Ok(Self {
            conv0,
            conv1,
            conv2,
            conv3,
            cls_logits,
            num_anchors,
            num_classes,
        })
    }

    /// Forward on a single feature map `[B, C, H, W]`. Returns
    /// `[B, H*W*num_anchors, num_classes]` logits (matching torchvision's
    /// permute/reshape so that the anchor axis is innermost per position).
    pub fn forward_level(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h = self.conv0.forward(x)?;
        let h = relu(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = relu(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = relu(&h)?;
        let h = self.conv3.forward(&h)?;
        let h = relu(&h)?;
        let logits = self.cls_logits.forward(&h)?; // [B, A*K, H, W]

        let shape = logits.shape();
        let b = shape[0];
        let ak = shape[1];
        let hh = shape[2];
        let ww = shape[3];
        let a = self.num_anchors;
        let k = self.num_classes;
        debug_assert_eq!(ak, a * k);

        // Layout: torchvision view→permute→reshape:
        //   (B, A*K, H, W) → (B, A, K, H, W) → (B, H, W, A, K) → (B, H*W*A, K)
        // Build the output directly to avoid an extra permute kernel.
        let data = logits.data_vec()?;
        let mut out = vec![cast::<f64, T>(0.0)?; b * hh * ww * a * k];
        for bi in 0..b {
            for hi in 0..hh {
                for wi in 0..ww {
                    for ai in 0..a {
                        for ki in 0..k {
                            let src = ((bi * a + ai) * k + ki) * hh * ww + hi * ww + wi;
                            let dst = ((bi * hh + hi) * ww + wi) * a * k + ai * k + ki;
                            out[dst] = data[src];
                        }
                    }
                }
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), vec![b, hh * ww * a, k], false)
    }

    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv0.parameters());
        p.extend(self.conv1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.conv3.parameters());
        p.extend(self.cls_logits.parameters());
        p
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv0.parameters_mut());
        p.extend(self.conv1.parameters_mut());
        p.extend(self.conv2.parameters_mut());
        p.extend(self.conv3.parameters_mut());
        p.extend(self.cls_logits.parameters_mut());
        p
    }

    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.conv0.named_parameters() {
            out.push((format!("conv.0.{n}"), p));
        }
        for (n, p) in self.conv1.named_parameters() {
            out.push((format!("conv.1.{n}"), p));
        }
        for (n, p) in self.conv2.named_parameters() {
            out.push((format!("conv.2.{n}"), p));
        }
        for (n, p) in self.conv3.named_parameters() {
            out.push((format!("conv.3.{n}"), p));
        }
        for (n, p) in self.cls_logits.named_parameters() {
            out.push((format!("cls_logits.{n}"), p));
        }
        out
    }
}

/// Shared regression head: same 4-conv stack, final conv outputs
/// `num_anchors * 4` channels (per-anchor (dx, dy, dw, dh) deltas).
///
/// Mirrors `torchvision.models.detection.retinanet.RetinaNetRegressionHead`.
pub struct RetinaNetRegressionHead<T: Float> {
    conv0: Conv2d<T>,
    conv1: Conv2d<T>,
    conv2: Conv2d<T>,
    conv3: Conv2d<T>,
    bbox_reg: Conv2d<T>,
    num_anchors: usize,
}

impl<T: Float> RetinaNetRegressionHead<T> {
    pub fn new(in_channels: usize, num_anchors: usize) -> FerrotorchResult<Self> {
        let conv0 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv1 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv2 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let conv3 = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let bbox_reg =
            Conv2d::new(in_channels, num_anchors * 4, (3, 3), (1, 1), (1, 1), true)?;
        Ok(Self {
            conv0,
            conv1,
            conv2,
            conv3,
            bbox_reg,
            num_anchors,
        })
    }

    /// Forward on a single feature map. Returns `[B, H*W*num_anchors, 4]`.
    pub fn forward_level(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h = self.conv0.forward(x)?;
        let h = relu(&h)?;
        let h = self.conv1.forward(&h)?;
        let h = relu(&h)?;
        let h = self.conv2.forward(&h)?;
        let h = relu(&h)?;
        let h = self.conv3.forward(&h)?;
        let h = relu(&h)?;
        let deltas = self.bbox_reg.forward(&h)?; // [B, A*4, H, W]

        let shape = deltas.shape();
        let b = shape[0];
        let a4 = shape[1];
        let hh = shape[2];
        let ww = shape[3];
        let a = self.num_anchors;
        debug_assert_eq!(a4, a * 4);

        let data = deltas.data_vec()?;
        let mut out = vec![cast::<f64, T>(0.0)?; b * hh * ww * a * 4];
        for bi in 0..b {
            for hi in 0..hh {
                for wi in 0..ww {
                    for ai in 0..a {
                        for ki in 0..4 {
                            let src = ((bi * a + ai) * 4 + ki) * hh * ww + hi * ww + wi;
                            let dst = ((bi * hh + hi) * ww + wi) * a * 4 + ai * 4 + ki;
                            out[dst] = data[src];
                        }
                    }
                }
            }
        }
        Tensor::from_storage(TensorStorage::cpu(out), vec![b, hh * ww * a, 4], false)
    }

    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv0.parameters());
        p.extend(self.conv1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.conv3.parameters());
        p.extend(self.bbox_reg.parameters());
        p
    }

    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv0.parameters_mut());
        p.extend(self.conv1.parameters_mut());
        p.extend(self.conv2.parameters_mut());
        p.extend(self.conv3.parameters_mut());
        p.extend(self.bbox_reg.parameters_mut());
        p
    }

    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.conv0.named_parameters() {
            out.push((format!("conv.0.{n}"), p));
        }
        for (n, p) in self.conv1.named_parameters() {
            out.push((format!("conv.1.{n}"), p));
        }
        for (n, p) in self.conv2.named_parameters() {
            out.push((format!("conv.2.{n}"), p));
        }
        for (n, p) in self.conv3.named_parameters() {
            out.push((format!("conv.3.{n}"), p));
        }
        for (n, p) in self.bbox_reg.named_parameters() {
            out.push((format!("bbox_reg.{n}"), p));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// RetinaNet anchor generation
// ---------------------------------------------------------------------------

/// Compute per-level cell anchor templates (centred at origin), one tensor
/// per level. Each level contributes `num_anchors_per_loc = 9` anchors
/// (3 sizes × 3 aspect ratios). Matches
/// `torchvision.models.detection.anchor_utils.AnchorGenerator.generate_anchors`
/// when invoked with the RetinaNet `(sizes, aspect_ratios)` config and
/// `int(x * 2**(k/3))` size triplets.
fn retinanet_cell_anchors<T: Float>(level: usize) -> FerrotorchResult<Vec<T>> {
    let base = RETINANET_BASE_SIZES[level];
    // torchvision: `tuple((x, int(x * 2**(1/3)), int(x * 2**(2/3))) for x in [...])`.
    // The `int(...)` truncates (Python int() on a float).
    let sizes = [
        base,
        (base * 2.0_f64.powf(1.0 / 3.0)).trunc(),
        (base * 2.0_f64.powf(2.0 / 3.0)).trunc(),
    ];
    let mut out = Vec::with_capacity(9 * 4);
    for &ratio in RETINANET_ASPECT_RATIOS.iter() {
        let sqrt_r = ratio.sqrt();
        for &s in sizes.iter() {
            let w = s / sqrt_r;
            let h = s * sqrt_r;
            // torchvision rounds the half-extents before negation so anchors
            // are integer-aligned. Same convention used by AnchorGenerator in
            // anchor_utils.rs (FasterRCNN side).
            let half_w: T = cast((w * 0.5).round())?;
            let half_h: T = cast((h * 0.5).round())?;
            out.push(cast::<f64, T>(0.0)? - half_w);
            out.push(cast::<f64, T>(0.0)? - half_h);
            out.push(half_w);
            out.push(half_h);
        }
    }
    Ok(out)
}

/// Generate anchors for every spatial cell of every level, in the EXACT
/// order required by the postprocess (positions outer, anchors-per-loc inner):
///
/// for each level:
///   for each (fy, fx) in (H_l × W_l):
///     for each a in 0..9:
///       emit anchor.
///
/// Strides per dim are derived from `image_size`, matching torchvision's
/// `AnchorGenerator.forward` (see #1141 round-4 anchor-stride diagnosis).
///
/// Returns a `Vec<Tensor<T>>` of length 5 (P3..P7), each `[H_l * W_l * 9, 4]`.
fn retinanet_anchors_per_level<T: Float>(
    feature_map_sizes: &[(usize, usize); 5],
    image_size: (usize, usize),
) -> FerrotorchResult<Vec<Tensor<T>>> {
    let mut levels = Vec::with_capacity(5);
    for (level_idx, &(fh, fw)) in feature_map_sizes.iter().enumerate() {
        let base_anchors: Vec<T> = retinanet_cell_anchors::<T>(level_idx)?;
        let num_base = base_anchors.len() / 4;
        debug_assert_eq!(num_base, 9);

        // Per-dim strides — torchvision recomputes these from the padded
        // image size, not the canonical level stride.
        let sh = image_size.0.checked_div(fh).unwrap_or(1);
        let sw = image_size.1.checked_div(fw).unwrap_or(1);
        let stride_h_t: T = cast(sh as f64)?;
        let stride_w_t: T = cast(sw as f64)?;

        let mut all: Vec<T> = Vec::with_capacity(fh * fw * 9 * 4);
        for fy in 0..fh {
            for fx in 0..fw {
                let cx: T = cast::<usize, T>(fx)? * stride_w_t;
                let cy: T = cast::<usize, T>(fy)? * stride_h_t;
                for a in 0..num_base {
                    all.push(cx + base_anchors[a * 4]);
                    all.push(cy + base_anchors[a * 4 + 1]);
                    all.push(cx + base_anchors[a * 4 + 2]);
                    all.push(cy + base_anchors[a * 4 + 3]);
                }
            }
        }

        let n = all.len() / 4;
        levels.push(Tensor::from_storage(
            TensorStorage::cpu(all),
            vec![n, 4],
            false,
        )?);
    }
    Ok(levels)
}

// ---------------------------------------------------------------------------
// Detection output
// ---------------------------------------------------------------------------

/// Per-image RetinaNet detection output.
#[derive(Debug, Clone)]
pub struct Detections<T: Float> {
    /// Predicted boxes `[N_det, 4]` in xyxy pixel coords.
    pub boxes: Tensor<T>,
    /// Per-detection sigmoid score, `[N_det]`.
    pub scores: Tensor<T>,
    /// Predicted class label `[N_det]` — 0-indexed over `num_classes`.
    pub labels: Vec<usize>,
}

// ---------------------------------------------------------------------------
// RetinaNet
// ---------------------------------------------------------------------------

/// RetinaNet single-stage detector.
pub struct RetinaNet<T: Float> {
    backbone: ResNet<T>,
    fpn: RetinaFpn<T>,
    classification_head: RetinaNetClassificationHead<T>,
    regression_head: RetinaNetRegressionHead<T>,
    num_classes: usize,
    training: bool,
}

impl<T: Float> RetinaNet<T> {
    /// Build with `num_classes` (matching torchvision's COCO_V1 value of 91).
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let backbone = resnet50(1)?;
        let fpn = RetinaFpn::new()?;
        let classification_head = RetinaNetClassificationHead::new(
            FPN_OUT_CHANNELS,
            RETINANET_NUM_ANCHORS_PER_LOC,
            num_classes,
        )?;
        let regression_head =
            RetinaNetRegressionHead::new(FPN_OUT_CHANNELS, RETINANET_NUM_ANCHORS_PER_LOC)?;
        Ok(Self {
            backbone,
            fpn,
            classification_head,
            regression_head,
            num_classes,
            training: false,
        })
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// FPN-level ordering used both for forward and anchor generation.
    const LEVEL_KEYS: [&'static str; 5] = ["p3", "p4", "p5", "p6", "p7"];

    /// Strides (P3..P7) — used only as a fallback for unit tests; production
    /// anchor generation uses per-dim strides from the padded image size.
    pub const STRIDES: [usize; 5] = RETINANET_STRIDES;

    /// End-to-end forward pass. `images` must be `[B, 3, H, W]` (already
    /// preprocessed — ImageNet mean/std normalised + padded to multiple of 32).
    pub fn forward(&self, images: &Tensor<T>) -> FerrotorchResult<Vec<Detections<T>>> {
        if images.ndim() != 4 || images.shape()[1] != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "RetinaNet::forward: expected [B, 3, H, W], got {:?}",
                    images.shape()
                ),
            });
        }
        let batch = images.shape()[0];
        let img_h = images.shape()[2];
        let img_w = images.shape()[3];

        // Backbone.
        let backbone_features = self.backbone.forward_features(images)?;
        // FPN.
        let fpn_features = self.fpn.forward(&backbone_features)?;

        // Per-level feature-map sizes for anchor generation.
        let fm_sizes: [(usize, usize); 5] = [
            {
                let s = fpn_features["p3"].shape();
                (s[2], s[3])
            },
            {
                let s = fpn_features["p4"].shape();
                (s[2], s[3])
            },
            {
                let s = fpn_features["p5"].shape();
                (s[2], s[3])
            },
            {
                let s = fpn_features["p6"].shape();
                (s[2], s[3])
            },
            {
                let s = fpn_features["p7"].shape();
                (s[2], s[3])
            },
        ];
        let anchors_per_level: Vec<Tensor<T>> =
            retinanet_anchors_per_level::<T>(&fm_sizes, (img_h, img_w))?;

        // Per-level head outputs. Store as data Vec for slicing per image.
        let mut cls_per_level: Vec<Tensor<T>> = Vec::with_capacity(5);
        let mut reg_per_level: Vec<Tensor<T>> = Vec::with_capacity(5);
        for key in Self::LEVEL_KEYS.iter() {
            let feat = &fpn_features[*key];
            cls_per_level.push(self.classification_head.forward_level(feat)?);
            reg_per_level.push(self.regression_head.forward_level(feat)?);
        }

        let num_classes = self.num_classes;
        let mut per_image_detections: Vec<Detections<T>> = Vec::with_capacity(batch);

        for b_idx in 0..batch {
            // Per-level postprocess: sigmoid → score_thresh → top-K (1000)
            // → decode → clip → stash into per-image vectors keyed by class.
            let mut all_boxes: Vec<f64> = Vec::new();
            let mut all_scores: Vec<f64> = Vec::new();
            let mut all_labels: Vec<usize> = Vec::new();

            for lv in 0..5 {
                let cls_t = &cls_per_level[lv];
                let reg_t = &reg_per_level[lv];
                let anc_t = &anchors_per_level[lv];

                // cls_t: [B, HWA, K]. Slice batch b_idx.
                let cls_shape = cls_t.shape();
                let hwa = cls_shape[1];
                let k = cls_shape[2];
                debug_assert_eq!(k, num_classes);
                let cls_data = cls_t.data_vec()?;
                let cls_offset = b_idx * hwa * k;

                let reg_data = reg_t.data_vec()?;
                let reg_offset = b_idx * hwa * 4;
                let anc_data = anc_t.data_vec()?;

                // sigmoid + score_thresh filter, then top-K=1000.
                // Build (flat_idx, score) for entries above threshold.
                let mut cand: Vec<(usize, f64)> = Vec::new();
                for flat in 0..(hwa * k) {
                    let logit = cls_data[cls_offset + flat].to_f64().unwrap_or(0.0);
                    let score = 1.0 / (1.0 + (-logit).exp());
                    if score > RETINANET_SCORE_THRESH {
                        cand.push((flat, score));
                    }
                }

                // Per-level top-K (1000) by descending score (stable sort
                // matches torchvision's `topk` deterministic tie-break).
                if cand.len() > RETINANET_TOPK_CANDIDATES {
                    cand.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    cand.truncate(RETINANET_TOPK_CANDIDATES);
                }

                // Decode the boxes for surviving anchors.
                // Group by anchor index to avoid decoding the same anchor twice.
                // (Multiple classes per anchor each carry the same decoded box.)
                let mut decoded_box_cache: HashMap<usize, [f64; 4]> = HashMap::new();
                for &(flat, score) in &cand {
                    let anchor_idx = flat / k;
                    let class_idx = flat % k;
                    let box_decoded = if let Some(b) = decoded_box_cache.get(&anchor_idx) {
                        *b
                    } else {
                        // Decode `reg_data[reg_offset + anchor_idx*4..+4]`
                        // applied to anchors `anc_data[anchor_idx*4..+4]`.
                        let a0 = anc_data[anchor_idx * 4].to_f64().unwrap_or(0.0);
                        let a1 = anc_data[anchor_idx * 4 + 1].to_f64().unwrap_or(0.0);
                        let a2 = anc_data[anchor_idx * 4 + 2].to_f64().unwrap_or(0.0);
                        let a3 = anc_data[anchor_idx * 4 + 3].to_f64().unwrap_or(0.0);
                        let d0 = reg_data[reg_offset + anchor_idx * 4]
                            .to_f64()
                            .unwrap_or(0.0);
                        let d1 = reg_data[reg_offset + anchor_idx * 4 + 1]
                            .to_f64()
                            .unwrap_or(0.0);
                        let d2 = reg_data[reg_offset + anchor_idx * 4 + 2]
                            .to_f64()
                            .unwrap_or(0.0);
                        let d3 = reg_data[reg_offset + anchor_idx * 4 + 3]
                            .to_f64()
                            .unwrap_or(0.0);

                        // BoxCoder weights (1, 1, 1, 1); same one-sided clip
                        // as anchor_utils::decode_boxes.
                        let anc_f64 = Tensor::from_storage(
                            TensorStorage::cpu(vec![a0, a1, a2, a3]),
                            vec![1, 4],
                            false,
                        )?;
                        let del_f64 = Tensor::from_storage(
                            TensorStorage::cpu(vec![d0, d1, d2, d3]),
                            vec![1, 4],
                            false,
                        )?;
                        let dec = decode_boxes::<f64>(&anc_f64, &del_f64, (1.0, 1.0, 1.0, 1.0))?;
                        let dd = dec.data_vec()?;
                        let b = [dd[0], dd[1], dd[2], dd[3]];
                        decoded_box_cache.insert(anchor_idx, b);
                        b
                    };
                    // Clip per-level (torchvision applies clip per level,
                    // before concat / NMS).
                    let x1 = box_decoded[0].clamp(0.0, img_w as f64);
                    let y1 = box_decoded[1].clamp(0.0, img_h as f64);
                    let x2 = box_decoded[2].clamp(0.0, img_w as f64);
                    let y2 = box_decoded[3].clamp(0.0, img_h as f64);
                    all_boxes.extend_from_slice(&[x1, y1, x2, y2]);
                    all_scores.push(score);
                    all_labels.push(class_idx);
                }
            }

            if all_scores.is_empty() {
                per_image_detections.push(Detections {
                    boxes: Tensor::from_storage(
                        TensorStorage::cpu(vec![]),
                        vec![0, 4],
                        false,
                    )?,
                    scores: Tensor::from_storage(
                        TensorStorage::cpu(vec![]),
                        vec![0usize],
                        false,
                    )?,
                    labels: vec![],
                });
                continue;
            }

            // Cross-class batched NMS (keyed by class label).
            let n_all = all_scores.len();
            let boxes_f64 = Tensor::from_storage(
                TensorStorage::cpu(all_boxes.clone()),
                vec![n_all, 4],
                false,
            )?;
            // Re-clip (already per-level clipped above; this is a no-op for
            // valid boxes but a guard for any decoded-out-of-range result).
            let boxes_clipped = clip_boxes_to_image(&boxes_f64, [img_h, img_w])?;
            let scores_f64 = Tensor::from_storage(
                TensorStorage::cpu(all_scores.clone()),
                vec![n_all],
                false,
            )?;
            let idxs: Vec<u32> = all_labels.iter().map(|&l| l as u32).collect();
            let keep = batched_nms::<f64>(&boxes_clipped, &scores_f64, &idxs, RETINANET_NMS_THRESH)?;

            // detections_per_img cap.
            let post = keep
                .into_iter()
                .take(RETINANET_DETECTIONS_PER_IMG)
                .collect::<Vec<_>>();

            let clipped_data = boxes_clipped.data_vec()?;
            let mut out_boxes: Vec<T> = Vec::with_capacity(post.len() * 4);
            let mut out_scores: Vec<T> = Vec::with_capacity(post.len());
            let mut out_labels: Vec<usize> = Vec::with_capacity(post.len());
            for &i in &post {
                out_boxes.push(cast::<f64, T>(clipped_data[i * 4])?);
                out_boxes.push(cast::<f64, T>(clipped_data[i * 4 + 1])?);
                out_boxes.push(cast::<f64, T>(clipped_data[i * 4 + 2])?);
                out_boxes.push(cast::<f64, T>(clipped_data[i * 4 + 3])?);
                out_scores.push(cast::<f64, T>(all_scores[i])?);
                out_labels.push(all_labels[i]);
            }
            let n_out = out_scores.len();
            per_image_detections.push(Detections {
                boxes: Tensor::from_storage(
                    TensorStorage::cpu(out_boxes),
                    vec![n_out, 4],
                    false,
                )?,
                scores: Tensor::from_storage(
                    TensorStorage::cpu(out_scores),
                    vec![n_out],
                    false,
                )?,
                labels: out_labels,
            });
        }

        Ok(per_image_detections)
    }
}

// ---------------------------------------------------------------------------
// Module trait
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for RetinaNet<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Module::forward exposes the first-image post-NMS sigmoid scores
        // as a 1-D `[N_det]` tensor — matching the contract used by
        // FasterRCNN/MaskRCNN in the #1139 verify harness, so
        // `torchvision.models.detection.retinanet_resnet50_fpn(...)(img)[0]
        // ["scores"]` is directly comparable.
        let dets = RetinaNet::forward(self, input)?;
        if dets.is_empty() || dets[0].scores.shape()[0] == 0 {
            return Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0usize], false);
        }
        Ok(dets[0].scores.clone())
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.backbone.parameters());
        p.extend(self.fpn.parameters());
        p.extend(self.classification_head.parameters());
        p.extend(self.regression_head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.backbone.parameters_mut());
        p.extend(self.fpn.parameters_mut());
        p.extend(self.classification_head.parameters_mut());
        p.extend(self.regression_head.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        // Key layout chosen to match torchvision's state_dict tree so the
        // pinning script can pass keys through with minimal rewriting:
        //   backbone.<resnet50 names>
        //   fpn.lateral{3,4,5}.{weight,bias} / fpn.output{3,4,5}.{weight,bias}
        //   fpn.p6.{weight,bias} / fpn.p7.{weight,bias}
        //   classification_head.conv.<0..3>.{weight,bias}
        //   classification_head.cls_logits.{weight,bias}
        //   regression_head.conv.<0..3>.{weight,bias}
        //   regression_head.bbox_reg.{weight,bias}
        let mut out = Vec::new();
        for (n, p) in self.backbone.named_parameters() {
            out.push((format!("backbone.{n}"), p));
        }
        for (n, p) in self.fpn.named_parameters() {
            out.push((format!("fpn.{n}"), p));
        }
        for (n, p) in self.classification_head.named_parameters() {
            out.push((format!("classification_head.{n}"), p));
        }
        for (n, p) in self.regression_head.named_parameters() {
            out.push((format!("regression_head.{n}"), p));
        }
        out
    }

    // BN buffer loader walks the ResNet backbone subtree.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.backbone]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![("backbone".to_string(), &self.backbone)]
    }

    fn train(&mut self) {
        self.training = true;
        self.backbone.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.backbone.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor
// ---------------------------------------------------------------------------

/// Construct a RetinaNet with ResNet-50 + FPN backbone for `num_classes`
/// detection classes (COCO default: 91, mirroring torchvision's pretrained
/// model).
pub fn retinanet_resnet50_fpn<T: Float>(num_classes: usize) -> FerrotorchResult<RetinaNet<T>> {
    RetinaNet::new(num_classes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{no_grad, randn};

    #[test]
    fn test_retinanet_constructs() {
        let m = retinanet_resnet50_fpn::<f32>(91).unwrap();
        assert!(m.num_parameters() > 0);
        assert_eq!(m.num_classes(), 91);
    }

    #[test]
    fn test_retinanet_named_params_prefixes() {
        let m = retinanet_resnet50_fpn::<f32>(91).unwrap();
        let names: Vec<String> = m.named_parameters().into_iter().map(|(n, _)| n).collect();
        assert!(names.iter().any(|n| n.starts_with("backbone.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.lateral3.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.lateral5.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.output3.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.p6.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.p7.")));
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("classification_head.conv.0."))
        );
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("classification_head.cls_logits."))
        );
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("regression_head.conv.0."))
        );
        assert!(
            names
                .iter()
                .any(|n| n.starts_with("regression_head.bbox_reg."))
        );
    }

    #[test]
    fn test_cls_logits_output_dim() {
        // 9 anchors * 91 classes = 819 channels.
        let head = RetinaNetClassificationHead::<f32>::new(256, 9, 91).unwrap();
        let names: Vec<(String, &Parameter<f32>)> = head.named_parameters();
        let cls_w = names
            .iter()
            .find(|(n, _)| n == "cls_logits.weight")
            .expect("cls_logits.weight missing");
        assert_eq!(cls_w.1.shape(), &[819, 256, 3, 3]);
    }

    #[test]
    fn test_bbox_reg_output_dim() {
        // 9 anchors * 4 = 36 channels.
        let head = RetinaNetRegressionHead::<f32>::new(256, 9).unwrap();
        let names: Vec<(String, &Parameter<f32>)> = head.named_parameters();
        let reg_w = names
            .iter()
            .find(|(n, _)| n == "bbox_reg.weight")
            .expect("bbox_reg.weight missing");
        assert_eq!(reg_w.1.shape(), &[36, 256, 3, 3]);
    }

    #[test]
    fn test_cls_head_forward_layout() {
        // Cls head on a 4×4 feature map → output [B, 4*4*9, K].
        let head = RetinaNetClassificationHead::<f32>::new(256, 9, 91).unwrap();
        let feat = no_grad(|| randn(&[1, 256, 4, 4]).unwrap());
        let out = no_grad(|| head.forward_level(&feat).unwrap());
        assert_eq!(out.shape(), &[1, 4 * 4 * 9, 91]);
    }

    #[test]
    fn test_reg_head_forward_layout() {
        let head = RetinaNetRegressionHead::<f32>::new(256, 9).unwrap();
        let feat = no_grad(|| randn(&[1, 256, 4, 4]).unwrap());
        let out = no_grad(|| head.forward_level(&feat).unwrap());
        assert_eq!(out.shape(), &[1, 4 * 4 * 9, 4]);
    }

    #[test]
    fn test_retina_fpn_p6_p7_are_stride_2_convs() {
        // P5 spatial 5×5 with kernel=3, stride=2, padding=1
        // → P6: floor((5 + 2 - 3) / 2) + 1 = 3.
        // P6 → kernel=3, stride=2, padding=1 → P7: floor((3 + 2 - 3) / 2) + 1 = 2.
        let fpn = RetinaFpn::<f32>::new().unwrap();
        let mut feats = HashMap::new();
        feats.insert("layer2".into(), randn(&[1, 512, 20, 20]).unwrap());
        feats.insert("layer3".into(), randn(&[1, 1024, 10, 10]).unwrap());
        feats.insert("layer4".into(), randn(&[1, 2048, 5, 5]).unwrap());
        let out = no_grad(|| fpn.forward(&feats).unwrap());
        assert_eq!(out["p3"].shape(), &[1, 256, 20, 20]);
        assert_eq!(out["p4"].shape(), &[1, 256, 10, 10]);
        assert_eq!(out["p5"].shape(), &[1, 256, 5, 5]);
        assert_eq!(
            out["p6"].shape(),
            &[1, 256, 3, 3],
            "P6 must be P5 with kernel=3 stride=2 padding=1"
        );
        assert_eq!(
            out["p7"].shape(),
            &[1, 256, 2, 2],
            "P7 must be P6 with kernel=3 stride=2 padding=1"
        );
    }

    #[test]
    fn test_retina_anchor_count_per_loc_is_9() {
        // 3 sizes × 3 aspect ratios per level.
        let anc = retinanet_cell_anchors::<f32>(0).unwrap();
        assert_eq!(anc.len() / 4, 9);
    }

    #[test]
    fn test_retina_anchor_sizes_first_level_truncated() {
        // Level 0: base 32.
        // sizes = (32, int(32 * 2**(1/3)) = 40, int(32 * 2**(2/3)) = 50).
        // For aspect 1.0, h = w = size, half_w = half_h = round(size/2).
        // half-extents at aspect 1.0: 16, 20, 25.
        // After iteration: outer loop is aspect ratio, so:
        //   ratio 0.5 (sqrt=0.707): sizes 32, 40, 50 → w=size/sqrt_r, h=size*sqrt_r
        //   ratio 1.0:               sizes 32, 40, 50 → square anchors
        //   ratio 2.0:               likewise
        // Find a ratio-1.0 size-32 anchor: it should be at index 3 (ratio 1.0
        // is the 2nd aspect ratio, size 32 is the first within it).
        // half_w = round(32/2) = 16, so anchor = (-16, -16, 16, 16).
        let anc = retinanet_cell_anchors::<f32>(0).unwrap();
        // ratios are [0.5, 1.0, 2.0]; sizes are [32, 40, 50]. Index 3 = ratio 1.0, size 32.
        let a3 = &anc[3 * 4..3 * 4 + 4];
        assert!((a3[0] - (-16.0)).abs() < 1e-4, "got {:?}", a3);
        assert!((a3[1] - (-16.0)).abs() < 1e-4, "got {:?}", a3);
        assert!((a3[2] - 16.0).abs() < 1e-4, "got {:?}", a3);
        assert!((a3[3] - 16.0).abs() < 1e-4, "got {:?}", a3);
    }

    #[test]
    fn test_retina_forward_small_image_returns_per_image_detections() {
        let m = retinanet_resnet50_fpn::<f32>(91).unwrap();
        // Use a 128×128 image so the 5-level FPN doesn't collapse to zero
        // spatial at p7 (128/128 = 1 → P7 shape (1,1)).
        let img = no_grad(|| randn(&[1, 3, 128, 128]).unwrap());
        let dets = no_grad(|| RetinaNet::forward(&m, &img).unwrap());
        assert_eq!(dets.len(), 1);
        // boxes [N, 4], scores [N], labels len N.
        let d = &dets[0];
        assert_eq!(d.boxes.shape().len(), 2);
        assert_eq!(d.boxes.shape()[1], 4);
        assert_eq!(d.scores.shape().len(), 1);
        assert_eq!(d.scores.shape()[0], d.boxes.shape()[0]);
        assert_eq!(d.labels.len(), d.boxes.shape()[0]);
    }

    #[test]
    fn test_retinanet_train_eval_toggle() {
        let mut m = retinanet_resnet50_fpn::<f32>(91).unwrap();
        assert!(!m.is_training());
        m.train();
        assert!(m.is_training());
        m.eval();
        assert!(!m.is_training());
    }
}
