//! Faster R-CNN with ResNet-50 FPN backbone.
//!
//! Mirrors `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
//!
//! ## Architecture
//!
//! ```text
//! image [B, 3, H, W]
//!   └─ ResNet-50 backbone → {layer1..layer4}   (C2–C5 feature maps)
//!         └─ FPN            → {p2..p6}          (256-ch multi-scale)
//!               └─ RPN      → proposals [N, 4]  (xyxy image coords)
//!                     └─ ROI Align (7×7)         → [N, 256, 7, 7]
//!                           └─ Detection heads
//!                                 ├─ box regressor  → [N, num_classes*4]
//!                                 └─ classifier     → [N, num_classes]
//! ```
//!
//! ## Reference
//! Ren et al., "Faster R-CNN: Towards Real-Time Object Detection with
//! Region Proposal Networks", NeurIPS 2015.
//! torchvision 0.21.x `fasterrcnn_resnet50_fpn(weights=None)`.

use std::collections::HashMap;

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;

use crate::models::detection::fpn::FeaturePyramidNetwork;
use crate::models::detection::roi_heads_postprocess::{
    PostprocessedDetections, postprocess_detections,
};
use crate::models::detection::rpn::{Rpn, RpnConfig};
use crate::models::feature_extractor::IntermediateFeatures;
use crate::models::resnet::{ResNet, resnet50};
use crate::ops::roi_align;

// ---------------------------------------------------------------------------
// ROI-level assignment
// ---------------------------------------------------------------------------

/// Assign each proposal to the FPN level that best matches its size.
///
/// Mirrors `torchvision.models.detection.roi_heads.assign_roi_level`:
/// `level = floor(k0 + log2(sqrt(area) / canonical_size))`
/// clamped to `[min_level, max_level]`.
///
/// Returns a `Vec<usize>` of length `N`, one per proposal.
fn assign_fpn_levels<T: Float>(
    proposals: &Tensor<T>,
    k0: f64,
    canonical_size: f64,
    min_level: usize,
    max_level: usize,
) -> FerrotorchResult<Vec<usize>> {
    let data = proposals.data_vec()?;
    let n = proposals.shape()[0];
    let mut levels = Vec::with_capacity(n);
    for i in 0..n {
        let x1 = data[i * 4].to_f64().unwrap_or(0.0);
        let y1 = data[i * 4 + 1].to_f64().unwrap_or(0.0);
        let x2 = data[i * 4 + 2].to_f64().unwrap_or(0.0);
        let y2 = data[i * 4 + 3].to_f64().unwrap_or(0.0);
        let area = ((x2 - x1) * (y2 - y1)).max(1.0);
        let level = (k0 + (area.sqrt() / canonical_size).log2())
            .floor()
            .clamp(min_level as f64, max_level as f64) as usize;
        levels.push(level);
    }
    Ok(levels)
}

// ---------------------------------------------------------------------------
// Detection result
// ---------------------------------------------------------------------------

/// Per-image detection output.
///
/// Mirrors `torchvision.models.detection.GeneralizedRCNN` output format after
/// `RoIHeads.postprocess_detections`: one row per surviving detection (post-NMS,
/// post-top-K). Background (class 0) is never emitted.
#[derive(Debug, Clone)]
pub struct Detections<T: Float> {
    /// Predicted boxes `[N_det, 4]` in xyxy pixel coords.
    pub boxes: Tensor<T>,
    /// Per-detection score (softmax probability of the predicted class), `[N_det]`.
    pub scores: Tensor<T>,
    /// Predicted class label `[N_det]` (always `>= 1` — background is dropped).
    pub labels: Vec<usize>,
}

// ---------------------------------------------------------------------------
// Two-MLP detection head
// ---------------------------------------------------------------------------

/// Two-layer MLP classification + regression head applied to ROI-pooled features.
///
/// Mirrors `torchvision.models.detection.faster_rcnn.TwoMLPHead` +
/// `FastRCNNPredictor`.
pub struct TwoMlpHead<T: Float> {
    fc6: Linear<T>,
    fc7: Linear<T>,
    /// Classifier: `[representation_size, num_classes]`.
    cls_score: Linear<T>,
    /// Box regressor: `[representation_size, num_classes * 4]`.
    bbox_pred: Linear<T>,
}

impl<T: Float> TwoMlpHead<T> {
    /// `roi_pool_size` is the ROI-Align output side length (typically 7).
    /// `in_channels` is the FPN channel count (256).
    /// `representation_size` is the hidden dimension (1024, matching torchvision).
    pub fn new(
        roi_pool_size: usize,
        in_channels: usize,
        representation_size: usize,
        num_classes: usize,
    ) -> FerrotorchResult<Self> {
        let flat = in_channels * roi_pool_size * roi_pool_size;
        let fc6 = Linear::new(flat, representation_size, true)?;
        let fc7 = Linear::new(representation_size, representation_size, true)?;
        let cls_score = Linear::new(representation_size, num_classes, true)?;
        let bbox_pred = Linear::new(representation_size, num_classes * 4, true)?;
        Ok(Self {
            fc6,
            fc7,
            cls_score,
            bbox_pred,
        })
    }

    /// Forward on ROI-pooled features `[N, C, P, P]`.
    ///
    /// Returns `(class_logits, box_deltas)`:
    /// - `class_logits` `[N, num_classes]`
    /// - `box_deltas`   `[N, num_classes * 4]`
    pub fn forward(&self, roi_features: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        let n = roi_features.shape()[0];
        let flat_size = roi_features.numel() / n;
        let x = reshape(roi_features, &[n as isize, flat_size as isize])?;
        let x = self.fc6.forward(&x)?;
        let x = relu(&x)?;
        let x = self.fc7.forward(&x)?;
        let x = relu(&x)?;
        let cls = self.cls_score.forward(&x)?;
        let bbox = self.bbox_pred.forward(&x)?;
        Ok((cls, bbox))
    }

    /// Trainable parameters.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.fc6.parameters());
        p.extend(self.fc7.parameters());
        p.extend(self.cls_score.parameters());
        p.extend(self.bbox_pred.parameters());
        p
    }

    /// Mutable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.fc6.parameters_mut());
        p.extend(self.fc7.parameters_mut());
        p.extend(self.cls_score.parameters_mut());
        p.extend(self.bbox_pred.parameters_mut());
        p
    }

    /// Named parameters.
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.fc6.named_parameters() {
            out.push((format!("fc6.{n}"), p));
        }
        for (n, p) in self.fc7.named_parameters() {
            out.push((format!("fc7.{n}"), p));
        }
        for (n, p) in self.cls_score.named_parameters() {
            out.push((format!("cls_score.{n}"), p));
        }
        for (n, p) in self.bbox_pred.named_parameters() {
            out.push((format!("bbox_pred.{n}"), p));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// FasterRcnn
// ---------------------------------------------------------------------------

/// Faster R-CNN with ResNet-50 FPN backbone.
///
/// This struct owns all sub-networks and exposes a single `forward` entry
/// point that produces `Vec<Detections<T>>` — one entry per image in the batch.
pub struct FasterRcnn<T: Float> {
    backbone: ResNet<T>,
    fpn: FeaturePyramidNetwork<T>,
    rpn: Rpn<T>,
    head: TwoMlpHead<T>,
    num_classes: usize,
    /// ROI Align output spatial size (7 matching torchvision default).
    roi_output_size: usize,
    /// Spatial scales per FPN level used by ROI Align.
    /// level i maps to scale = 1 / stride_i.
    roi_spatial_scales: Vec<f64>,
    training: bool,
}

impl<T: Float> FasterRcnn<T> {
    /// FPN level names in order (p2..p6).
    const FPN_LEVEL_KEYS: [&'static str; 5] = ["p2", "p3", "p4", "p5", "p6"];

    /// Spatial scales for FPN levels p2..p6 (1/stride).
    const FPN_SPATIAL_SCALES: [f64; 5] = [1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0];

    /// Create a Faster R-CNN from scratch with `num_classes` output classes
    /// (including background at index 0, matching torchvision).
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        // ResNet-50 as backbone (1000-class head is ignored; we use forward_features).
        let backbone = resnet50(1)?;
        let fpn = FeaturePyramidNetwork::new()?;
        let rpn = Rpn::new(256)?;
        let head = TwoMlpHead::new(7, 256, 1024, num_classes)?;

        Ok(Self {
            backbone,
            fpn,
            rpn,
            head,
            num_classes,
            roi_output_size: 7,
            roi_spatial_scales: Self::FPN_SPATIAL_SCALES.to_vec(),
            training: false,
        })
    }

    /// Borrow the inner `Rpn` (used by the #1141 per-stage probe to drive
    /// the head + anchor generator directly). Read-only — production
    /// callers should go through `FasterRcnn::forward`.
    pub fn rpn(&self) -> &Rpn<T> {
        &self.rpn
    }

    /// Run backbone only and return the four intermediate feature maps.
    ///
    /// Returns `HashMap` with keys `"layer1"`, `"layer2"`, `"layer3"`,
    /// `"layer4"` (C2–C5 outputs from ResNet-50).
    ///
    /// Exposed so that `MaskRcnn` can reuse backbone features for mask ROI
    /// Align without running the full detection pipeline twice.
    pub fn forward_backbone(
        &self,
        images: &Tensor<T>,
    ) -> FerrotorchResult<HashMap<String, Tensor<T>>> {
        self.backbone.forward_features(images)
    }

    /// Run FPN over backbone features and return the five pyramid levels.
    ///
    /// Returns `HashMap` with keys `"p2"` .. `"p6"`.
    ///
    /// Exposed so that `MaskRcnn` can reuse FPN features for mask ROI Align
    /// without running the full detection pipeline twice.
    pub fn forward_fpn(
        &self,
        backbone_features: &HashMap<String, Tensor<T>>,
    ) -> FerrotorchResult<HashMap<String, Tensor<T>>> {
        self.fpn.forward(backbone_features)
    }

    /// End-to-end forward pass.
    ///
    /// `images` — `[B, 3, H, W]` float tensor (RGB, any scale).
    ///
    /// Returns a `Vec<Detections<T>>` of length `B`.
    pub fn forward(&self, images: &Tensor<T>) -> FerrotorchResult<Vec<Detections<T>>> {
        if images.ndim() != 4 || images.shape()[1] != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "FasterRcnn::forward: expected [B, 3, H, W], got {:?}",
                    images.shape()
                ),
            });
        }
        let batch = images.shape()[0];
        let img_h = images.shape()[2];
        let img_w = images.shape()[3];

        // ---- Backbone ----
        let backbone_features: HashMap<String, Tensor<T>> =
            self.backbone.forward_features(images)?;

        // ---- FPN ----
        let fpn_features = self.fpn.forward(&backbone_features)?;

        // Ordered list of FPN levels.
        let level_tensors: Vec<&Tensor<T>> = Self::FPN_LEVEL_KEYS
            .iter()
            .map(|&k| &fpn_features[k])
            .collect();

        // ---- RPN ---- (per-image proposals; we loop over batch)
        let mut per_image_detections: Vec<Detections<T>> = Vec::with_capacity(batch);

        for b_idx in 0..batch {
            // Extract single-image slices from the batch.
            let single_levels: Vec<Tensor<T>> = level_tensors
                .iter()
                .map(|t| slice_batch_item(t, b_idx))
                .collect::<FerrotorchResult<Vec<_>>>()?;
            let single_refs: Vec<&Tensor<T>> = single_levels.iter().collect();

            let rpn_cfg = RpnConfig::default_eval([img_h, img_w]);
            let proposals = self.rpn.forward(&single_refs, &rpn_cfg)?;

            if proposals.shape()[0] == 0 {
                // No proposals → empty detections for this image.
                per_image_detections.push(Detections {
                    boxes: Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0, 4], false)?,
                    scores: Tensor::from_storage(
                        TensorStorage::cpu(vec![]),
                        vec![0usize],
                        false,
                    )?,
                    labels: vec![],
                });
                continue;
            }

            let n_proposals = proposals.shape()[0];

            // ---- ROI level assignment ----
            // k0=4 (FPN default), canonical=224, levels 2..6.
            let roi_levels = assign_fpn_levels(&proposals, 4.0, 224.0, 2, 6)?;

            // ---- ROI Align ---- per FPN level, then reassemble.
            let mut roi_features_all: Vec<Option<Vec<T>>> = vec![None; n_proposals];

            for (level_idx, &level_key) in Self::FPN_LEVEL_KEYS.iter().enumerate() {
                let fpn_level = level_idx + 2; // p2 = level 2 … p6 = level 6

                // Collect proposal indices that map to this FPN level.
                let indices: Vec<usize> = roi_levels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &lv)| if lv == fpn_level { Some(i) } else { None })
                    .collect();

                if indices.is_empty() {
                    continue;
                }

                let feat = &fpn_features[level_key];
                let scale = self.roi_spatial_scales[level_idx];

                // Build [K, 5] boxes tensor for roi_align (batch_idx=0 since this is a single image).
                let zero: T = cast(0.0f64)?;
                let prop_data = proposals.data_vec()?;
                let mut roi_boxes: Vec<T> = Vec::with_capacity(indices.len() * 5);
                for &i in &indices {
                    roi_boxes.push(zero);
                    roi_boxes.push(prop_data[i * 4]);
                    roi_boxes.push(prop_data[i * 4 + 1]);
                    roi_boxes.push(prop_data[i * 4 + 2]);
                    roi_boxes.push(prop_data[i * 4 + 3]);
                }

                let k = indices.len();
                let boxes_t =
                    Tensor::from_storage(TensorStorage::cpu(roi_boxes), vec![k, 5], false)?;

                let roi_out = roi_align(
                    feat,
                    &boxes_t,
                    (self.roi_output_size, self.roi_output_size),
                    scale,
                    2, // sampling_ratio=2 as in torchvision default
                )?;

                // roi_out: [K, 256, 7, 7] — store each row.
                let channels = feat.shape()[1];
                let per_roi_size = channels * self.roi_output_size * self.roi_output_size;
                let roi_data = roi_out.data_vec()?;

                for (local_idx, &global_idx) in indices.iter().enumerate() {
                    let start = local_idx * per_roi_size;
                    let row: Vec<T> = roi_data[start..start + per_roi_size].to_vec();
                    roi_features_all[global_idx] = Some(row);
                }
            }

            // Assemble into [N, 256, 7, 7].
            let channels = 256;
            let p = self.roi_output_size;
            let per_roi = channels * p * p;
            let mut stacked: Vec<T> = Vec::with_capacity(n_proposals * per_roi);
            for slot in &roi_features_all {
                if let Some(row) = slot {
                    stacked.extend_from_slice(row);
                } else {
                    // Fallback: zero-fill for proposals that didn't land on a level.
                    let zero: T = cast(0.0f64)?;
                    stacked.extend(vec![zero; per_roi]);
                }
            }

            let roi_tensor = Tensor::from_storage(
                TensorStorage::cpu(stacked),
                vec![n_proposals, channels, p, p],
                false,
            )?;

            // ---- Detection head ----
            let (class_logits, box_deltas) = self.head.forward(&roi_tensor)?;

            // ---- RoIHeads postprocess (softmax → BoxCoder decode → clip →
            //      drop background → score-thresh → small-box filter → per-class
            //      NMS → cross-class top-K). Mirrors torchvision exactly. ----
            let PostprocessedDetections {
                boxes,
                scores,
                labels,
            } = postprocess_detections::<T>(
                &class_logits,
                &box_deltas,
                &proposals,
                [img_h, img_w],
            )?;
            // Silence "unused" if n_proposals only used in shape check above.
            let _ = n_proposals;

            per_image_detections.push(Detections {
                boxes,
                scores,
                labels,
            });
        }

        Ok(per_image_detections)
    }

    /// Total trainable parameter count.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Number of detection classes including background at index 0.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

// ---------------------------------------------------------------------------
// Module trait implementation
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for FasterRcnn<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Module::forward is required for the registry but the primary API is
        // `FasterRcnn::forward` which returns `Vec<Detections<T>>`.
        // Convention (matches #1139 verification harness): expose the
        // post-NMS, post-top-K per-detection scores of the first image as a
        // 1-D `[N_det]` tensor, matching `torchvision`'s
        // `model(img)[0]["scores"]`.
        let dets = FasterRcnn::forward(self, input)?;
        if dets.is_empty() || dets[0].scores.shape()[0] == 0 {
            return Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0usize], false);
        }
        Ok(dets[0].scores.clone())
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.backbone.parameters());
        p.extend(self.fpn.parameters());
        p.extend(self.rpn.parameters());
        p.extend(self.head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.backbone.parameters_mut());
        p.extend(self.fpn.parameters_mut());
        p.extend(self.rpn.parameters_mut());
        p.extend(self.head.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.backbone.named_parameters() {
            out.push((format!("backbone.{n}"), p));
        }
        for (n, p) in self.fpn.named_parameters() {
            out.push((format!("fpn.{n}"), p));
        }
        for (n, p) in self.rpn.named_parameters() {
            out.push((format!("rpn.{n}"), p));
        }
        for (n, p) in self.head.named_parameters() {
            out.push((format!("head.{n}"), p));
        }
        out
    }

    // Phase 4 (#995): expose the ResNet backbone as a child so the
    // Phase 2 BN-buffer loader walks into it. FPN / RPN / TwoMlpHead
    // are inherent-method helpers (NOT `Module<T>` impls) — they
    // own no BN buffers, so we do not project them via this hook.
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

/// Construct a Faster R-CNN with ResNet-50 FPN backbone.
///
/// `num_classes` includes the background class (index 0), matching
/// `torchvision.models.detection.fasterrcnn_resnet50_fpn(num_classes=91)`.
pub fn fasterrcnn_resnet50_fpn<T: Float>(num_classes: usize) -> FerrotorchResult<FasterRcnn<T>> {
    FasterRcnn::new(num_classes)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Extract item `b` from a `[B, C, H, W]` tensor → `[1, C, H, W]`.
fn slice_batch_item<T: Float>(t: &Tensor<T>, b: usize) -> FerrotorchResult<Tensor<T>> {
    let shape = t.shape();
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    let stride = c * h * w;
    let data = t.data_vec()?;
    let slice = data[b * stride..(b + 1) * stride].to_vec();
    Tensor::from_storage(TensorStorage::cpu(slice), vec![1, c, h, w], false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::no_grad;

    fn make_model() -> FasterRcnn<f32> {
        fasterrcnn_resnet50_fpn::<f32>(91).unwrap()
    }

    #[test]
    fn test_faster_rcnn_constructs() {
        let model = make_model();
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_faster_rcnn_param_count_ballpark() {
        // ResNet-50 (~25.5M) + FPN (~3.3M) + RPN head (small) + TwoMLP (~26M).
        // Total should be well above 40M and below 80M.
        let model = make_model();
        let np = model.num_parameters();
        assert!(np > 40_000_000, "param count too low: {np}");
        assert!(np < 80_000_000, "param count too high: {np}");
    }

    #[test]
    fn test_faster_rcnn_named_params_prefixes() {
        let model = make_model();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("backbone.")));
        assert!(names.iter().any(|n| n.starts_with("fpn.")));
        assert!(names.iter().any(|n| n.starts_with("rpn.")));
        assert!(names.iter().any(|n| n.starts_with("head.")));
    }

    #[test]
    fn test_faster_rcnn_forward_output_structure() {
        let model = make_model();
        // Small 64×64 image to keep the test fast.
        let img = no_grad(|| ferrotorch_core::randn(&[1, 3, 64, 64]).unwrap());
        let dets = no_grad(|| model.forward(&img).unwrap());
        assert_eq!(dets.len(), 1, "one detection list per image");
        let d = &dets[0];
        // Post-NMS shape: boxes [N_det, 4], scores [N_det], labels len N_det.
        assert_eq!(d.boxes.shape().len(), 2);
        assert_eq!(d.boxes.shape()[1], 4);
        assert_eq!(d.scores.shape().len(), 1);
        assert_eq!(d.scores.shape()[0], d.boxes.shape()[0]);
        assert_eq!(d.labels.len(), d.boxes.shape()[0]);
        // Background label (0) is dropped by the postprocess.
        assert!(d.labels.iter().all(|&l| l >= 1), "no background labels");
    }

    #[test]
    fn test_faster_rcnn_two_images_batch() {
        let model = make_model();
        let imgs = no_grad(|| ferrotorch_core::randn(&[2, 3, 64, 64]).unwrap());
        let dets = no_grad(|| model.forward(&imgs).unwrap());
        assert_eq!(dets.len(), 2);
    }

    #[test]
    fn test_faster_rcnn_train_eval() {
        let mut model = make_model();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
    }

    #[test]
    fn test_two_mlp_head_shapes() {
        let head = TwoMlpHead::<f32>::new(7, 256, 1024, 91).unwrap();
        let features = ferrotorch_core::randn(&[4, 256, 7, 7]).unwrap();
        let (cls, bbox) = head.forward(&features).unwrap();
        assert_eq!(cls.shape(), &[4, 91]);
        assert_eq!(bbox.shape(), &[4, 91 * 4]);
    }
}
