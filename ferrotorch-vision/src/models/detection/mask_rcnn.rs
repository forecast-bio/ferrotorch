//! Mask R-CNN with ResNet-50 FPN backbone.
//!
//! Mirrors `torchvision.models.detection.maskrcnn_resnet50_fpn`.
//!
//! ## Architecture
//!
//! ```text
//! image [B, 3, H, W]
//!   └─ ResNet-50 backbone → {layer1..layer4}   (C2–C5 feature maps)
//!         └─ FPN            → {p2..p6}          (256-ch multi-scale)
//!               └─ RPN      → proposals [N, 4]  (xyxy image coords)
//!                     ├─ ROI Align (7×7)  → [N, 256, 7, 7]  (detection)
//!                     │       └─ TwoMlpHead → boxes + classes
//!                     └─ ROI Align (14×14) → [N, 256, 14, 14] (masks)
//!                             └─ MaskHead (4×conv) → [N, 256, 14, 14]
//!                                   └─ MaskPredictor (deconv+conv) → [N, num_classes, 28, 28]
//! ```
//!
//! ## Reference
//! He et al., "Mask R-CNN", ICCV 2017.
//! torchvision 0.21.x `maskrcnn_resnet50_fpn(weights=None)`.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::{Conv2d, ConvTranspose2d};

use crate::models::detection::faster_rcnn::{FasterRcnn, fasterrcnn_resnet50_fpn};
use crate::models::detection::roi_heads_postprocess::postprocess_masks;
use crate::ops::roi_align;

// ---------------------------------------------------------------------------
// MaskHead — 4 conv layers producing [N, 256, roi_H, roi_W]
// ---------------------------------------------------------------------------

/// Four-layer FCN head applied to mask ROI features.
///
/// Mirrors `torchvision.models.detection.mask_rcnn.MaskRCNNHeads` with
/// `layers=(256, 256, 256, 256)` and `dilation=1`.
pub struct MaskHead<T: Float> {
    conv1: Conv2d<T>,
    conv2: Conv2d<T>,
    conv3: Conv2d<T>,
    conv4: Conv2d<T>,
}

impl<T: Float> MaskHead<T> {
    /// Create a new `MaskHead`.
    ///
    /// `in_channels` — number of input feature channels (256 from FPN).
    pub fn new(in_channels: usize) -> FerrotorchResult<Self> {
        // All four convolutions: kernel 3×3, pad 1, no stride — spatial size preserved.
        let conv1 = Conv2d::new(in_channels, 256, (3, 3), (1, 1), (1, 1), true)?;
        let conv2 = Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), true)?;
        let conv3 = Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), true)?;
        let conv4 = Conv2d::new(256, 256, (3, 3), (1, 1), (1, 1), true)?;
        Ok(Self {
            conv1,
            conv2,
            conv3,
            conv4,
        })
    }

    /// Forward on `[N, in_channels, H, W]` → `[N, 256, H, W]`.
    pub fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = relu(&self.conv1.forward(x)?)?;
        let x = relu(&self.conv2.forward(&x)?)?;
        let x = relu(&self.conv3.forward(&x)?)?;
        relu(&self.conv4.forward(&x)?)
    }

    /// Trainable parameters.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.conv2.parameters());
        p.extend(self.conv3.parameters());
        p.extend(self.conv4.parameters());
        p
    }

    /// Mutable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters_mut());
        p.extend(self.conv2.parameters_mut());
        p.extend(self.conv3.parameters_mut());
        p.extend(self.conv4.parameters_mut());
        p
    }

    /// Named parameters.
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.conv1.named_parameters() {
            out.push((format!("conv1.{n}"), p));
        }
        for (n, p) in self.conv2.named_parameters() {
            out.push((format!("conv2.{n}"), p));
        }
        for (n, p) in self.conv3.named_parameters() {
            out.push((format!("conv3.{n}"), p));
        }
        for (n, p) in self.conv4.named_parameters() {
            out.push((format!("conv4.{n}"), p));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// MaskPredictor — deconv + 1×1 conv producing [N, num_classes, 28, 28]
// ---------------------------------------------------------------------------

/// Mask predictor: bilinear 2× upsample via transposed convolution followed
/// by a 1×1 projection to `num_classes` output channels.
///
/// Mirrors `torchvision.models.detection.mask_rcnn.MaskRCNNPredictor`.
///
/// Given 14×14 input (from ROI Align at output_size=14), the transposed
/// convolution with kernel=(2,2) and stride=(2,2) doubles spatial resolution
/// to 28×28, matching the torchvision default.
pub struct MaskPredictor<T: Float> {
    /// Transposed conv: 256 → 256, kernel 2×2, stride 2 (spatial ×2).
    deconv: ConvTranspose2d<T>,
    /// 1×1 conv: 256 → num_classes.
    conv_logits: Conv2d<T>,
}

impl<T: Float> MaskPredictor<T> {
    /// Create a new `MaskPredictor`.
    ///
    /// `in_channels` is typically 256 (the MaskHead output channels).
    /// `num_classes` includes background at index 0.
    pub fn new(in_channels: usize, num_classes: usize) -> FerrotorchResult<Self> {
        let deconv = ConvTranspose2d::new(in_channels, 256, (2, 2), (2, 2), (0, 0), (0, 0), true)?;
        let conv_logits = Conv2d::new(256, num_classes, (1, 1), (1, 1), (0, 0), true)?;
        Ok(Self {
            deconv,
            conv_logits,
        })
    }

    /// Forward on `[N, in_channels, 14, 14]` → `[N, num_classes, 28, 28]`.
    pub fn forward(&self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = relu(&self.deconv.forward(x)?)?;
        self.conv_logits.forward(&x)
    }

    /// Trainable parameters.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.deconv.parameters());
        p.extend(self.conv_logits.parameters());
        p
    }

    /// Mutable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.deconv.parameters_mut());
        p.extend(self.conv_logits.parameters_mut());
        p
    }

    /// Named parameters.
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.deconv.named_parameters() {
            out.push((format!("deconv.{n}"), p));
        }
        for (n, p) in self.conv_logits.named_parameters() {
            out.push((format!("conv_logits.{n}"), p));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Per-image detection result including masks
// ---------------------------------------------------------------------------

/// Per-image detection output from Mask R-CNN.
///
/// Extends [`crate::models::detection::Detections`] with a single pasted mask
/// per detection. Matches `torchvision.models.detection.MaskRCNN`'s output
/// dictionary after `paste_masks_in_image`.
#[derive(Debug, Clone)]
pub struct MaskDetections<T: Float> {
    /// Predicted boxes `[N_det, 4]` in xyxy pixel coords.
    pub boxes: Tensor<T>,
    /// Per-detection score `[N_det]` (softmax probability of the predicted class).
    pub scores: Tensor<T>,
    /// Predicted class label `[N_det]` (always `>= 1` — background is dropped).
    pub labels: Vec<usize>,
    /// Mask probabilities `[N_det, 1, H_img, W_img]` after sigmoid +
    /// class-channel-selection + `paste_masks_in_image`. Matches
    /// `torchvision.models.detection.MaskRCNN`'s `model(img)[0]["masks"]`
    /// from the full forward (post-`GeneralizedRCNNTransform.postprocess`).
    pub masks: Tensor<T>,
}

// ---------------------------------------------------------------------------
// MaskRcnn
// ---------------------------------------------------------------------------

/// Mask R-CNN with ResNet-50 FPN backbone.
///
/// Extends Faster R-CNN by adding a parallel mask branch that operates on
/// 14×14 ROI-aligned features and outputs per-class binary mask logits.
///
/// **Reuses Sprint C.1 components**: backbone (ResNet-50), FPN, RPN, ROI
/// Align, and the `TwoMlpHead` detection head from `FasterRcnn`. Only the
/// mask-specific layers (`MaskHead` + `MaskPredictor`) are new.
pub struct MaskRcnn<T: Float> {
    /// Faster R-CNN sub-model (backbone + FPN + RPN + detection head).
    ///
    /// All Sprint C.1 components are owned here; no duplication.
    faster_rcnn: FasterRcnn<T>,
    /// 4-layer FCN mask head.
    mask_head: MaskHead<T>,
    /// Deconv + 1×1 conv mask predictor.
    mask_predictor: MaskPredictor<T>,
    num_classes: usize,
    /// ROI Align spatial size for the mask branch (14×14).
    mask_roi_size: usize,
    /// Spatial scales per FPN level p2..p6 (1/stride).
    roi_spatial_scales: Vec<f64>,
    training: bool,
}

impl<T: Float> MaskRcnn<T> {
    /// FPN level names in order (p2..p6).
    const FPN_LEVEL_KEYS: [&'static str; 5] = ["p2", "p3", "p4", "p5", "p6"];

    /// Spatial scales for FPN levels p2..p6 (1/stride).
    const FPN_SPATIAL_SCALES: [f64; 5] = [1.0 / 4.0, 1.0 / 8.0, 1.0 / 16.0, 1.0 / 32.0, 1.0 / 64.0];

    /// Create a new Mask R-CNN from scratch.
    ///
    /// `num_classes` includes background at index 0, matching torchvision
    /// (default COCO: 91 classes).
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let faster_rcnn = fasterrcnn_resnet50_fpn::<T>(num_classes)?;
        let mask_head = MaskHead::new(256)?;
        let mask_predictor = MaskPredictor::new(256, num_classes)?;

        Ok(Self {
            faster_rcnn,
            mask_head,
            mask_predictor,
            num_classes,
            mask_roi_size: 14,
            roi_spatial_scales: Self::FPN_SPATIAL_SCALES.to_vec(),
            training: false,
        })
    }

    /// End-to-end forward pass.
    ///
    /// `images` — `[B, 3, H, W]` float tensor (RGB, any scale).
    ///
    /// Returns a `Vec<MaskDetections<T>>` of length `B`.
    pub fn forward(&self, images: &Tensor<T>) -> FerrotorchResult<Vec<MaskDetections<T>>> {
        if images.ndim() != 4 || images.shape()[1] != 3 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "MaskRcnn::forward: expected [B, 3, H, W], got {:?}",
                    images.shape()
                ),
            });
        }
        let batch = images.shape()[0];
        let img_h = images.shape()[2];
        let img_w = images.shape()[3];

        // ---- Reuse FasterRcnn backbone + FPN internals ----
        // We call into the FasterRcnn's sub-modules via its forward to get
        // detection results (already post-NMS, post-top-K), then separately
        // run the mask branch.
        let detections = self.faster_rcnn.forward(images)?;

        // We also need the FPN features for the mask ROI align.
        // Re-run backbone + FPN (same weights, no double training).
        let backbone_features = self.faster_rcnn.forward_backbone(images)?;
        let fpn_features = self.faster_rcnn.forward_fpn(&backbone_features)?;

        let mut results: Vec<MaskDetections<T>> = Vec::with_capacity(batch);

        for (b_idx, det) in detections.into_iter().enumerate() {
            let n_proposals = det.boxes.shape()[0];

            if n_proposals == 0 {
                // No detections → empty post-paste mask tensor with shape
                // `[0, 1, H_img, W_img]` matching torchvision's
                // `model(img)[0]["masks"]` (post-paste) when no detections
                // survived NMS.
                let empty_masks = Tensor::from_storage(
                    TensorStorage::cpu(vec![]),
                    vec![0, 1, img_h, img_w],
                    false,
                )?;
                results.push(MaskDetections {
                    boxes: det.boxes,
                    scores: det.scores,
                    labels: det.labels,
                    masks: empty_masks,
                });
                continue;
            }

            // ---- Mask ROI Align (14×14) ----
            // Assign proposals to FPN levels same as detection head.
            let roi_levels = assign_fpn_levels_mask(&det.boxes, 4.0, 224.0, 2, 6)?;

            let mut mask_roi_features_all: Vec<Option<Vec<T>>> = vec![None; n_proposals];

            for (level_idx, &level_key) in Self::FPN_LEVEL_KEYS.iter().enumerate() {
                let fpn_level = level_idx + 2;

                // Get single-image slice of this FPN level.
                let feat_b = &fpn_features[level_key];
                let feat_single = slice_batch_item_mask(feat_b, b_idx)?;

                let indices: Vec<usize> = roi_levels
                    .iter()
                    .enumerate()
                    .filter_map(|(i, &lv)| if lv == fpn_level { Some(i) } else { None })
                    .collect();

                if indices.is_empty() {
                    continue;
                }

                let scale = self.roi_spatial_scales[level_idx];
                let zero: T = cast(0.0f64)?;
                let prop_data = det.boxes.data_vec()?;

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
                    &feat_single,
                    &boxes_t,
                    (self.mask_roi_size, self.mask_roi_size),
                    scale,
                    2,
                )?;

                let channels = feat_single.shape()[1];
                let per_roi_size = channels * self.mask_roi_size * self.mask_roi_size;
                let roi_data = roi_out.data_vec()?;

                for (local_idx, &global_idx) in indices.iter().enumerate() {
                    let start = local_idx * per_roi_size;
                    let row: Vec<T> = roi_data[start..start + per_roi_size].to_vec();
                    mask_roi_features_all[global_idx] = Some(row);
                }
            }

            // Assemble [N, 256, 14, 14].
            let channels = 256usize;
            let p = self.mask_roi_size;
            let per_roi = channels * p * p;
            let mut stacked: Vec<T> = Vec::with_capacity(n_proposals * per_roi);
            for slot in &mask_roi_features_all {
                if let Some(row) = slot {
                    stacked.extend_from_slice(row);
                } else {
                    let zero: T = cast(0.0f64)?;
                    stacked.extend(vec![zero; per_roi]);
                }
            }

            let mask_roi_tensor = Tensor::from_storage(
                TensorStorage::cpu(stacked),
                vec![n_proposals, channels, p, p],
                false,
            )?;

            // ---- Mask head ----
            let mask_features = self.mask_head.forward(&mask_roi_tensor)?;

            // ---- Mask predictor ----
            let mask_logits = self.mask_predictor.forward(&mask_features)?;
            // Shape: [N_det, num_classes, 28, 28].

            // ---- Mask postprocess: sigmoid → class-select → paste-back ----
            //
            // Mirrors torchvision `maskrcnn_inference` followed by the
            // `GeneralizedRCNNTransform.postprocess` `paste_masks_in_image`
            // step, producing `[N_det, 1, H_img, W_img]` to match
            // `model(img)[0]["masks"]` from torchvision's full forward.
            let pasted_masks = postprocess_masks::<T>(
                &mask_logits,
                &det.labels,
                &det.boxes,
                [img_h, img_w],
                /* paste = */ true,
            )?;

            results.push(MaskDetections {
                boxes: det.boxes,
                scores: det.scores,
                labels: det.labels,
                masks: pasted_masks,
            });
        }

        Ok(results)
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

impl<T: Float> Module<T> for MaskRcnn<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Module::forward is required for the registry; primary API is
        // `MaskRcnn::forward` which returns `Vec<MaskDetections<T>>`.
        //
        // Convention (matches #1139 verification harness): expose the
        // first-image POST-PASTE mask tensor `[N_det, 1, H_img, W_img]`
        // (sigmoid + class-select + `paste_masks_in_image`). Matches
        // `torchvision`'s `model(img)[0]["masks"]` from the full forward.
        let img_h = input.shape()[2];
        let img_w = input.shape()[3];
        let dets = MaskRcnn::forward(self, input)?;
        if dets.is_empty() || dets[0].masks.shape()[0] == 0 {
            return Tensor::from_storage(
                TensorStorage::cpu(vec![]),
                vec![0, 1, img_h, img_w],
                false,
            );
        }
        Ok(dets[0].masks.clone())
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.faster_rcnn.parameters());
        p.extend(self.mask_head.parameters());
        p.extend(self.mask_predictor.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.faster_rcnn.parameters_mut());
        p.extend(self.mask_head.parameters_mut());
        p.extend(self.mask_predictor.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.faster_rcnn.named_parameters() {
            out.push((format!("faster_rcnn.{n}"), p));
        }
        for (n, p) in self.mask_head.named_parameters() {
            out.push((format!("mask_head.{n}"), p));
        }
        for (n, p) in self.mask_predictor.named_parameters() {
            out.push((format!("mask_predictor.{n}"), p));
        }
        out
    }

    // Phase 4 (#995): expose `faster_rcnn` so the BN-buffer loader walks
    // into the wrapped ResNet backbone, plus the mask-branch `Conv2d`s.
    // `MaskHead` / `MaskPredictor` are inherent-method helpers (NOT
    // `Module<T>` impls), so we project their inner Conv2d's directly.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.faster_rcnn,
            &self.mask_head.conv1,
            &self.mask_head.conv2,
            &self.mask_head.conv3,
            &self.mask_head.conv4,
            &self.mask_predictor.deconv,
            &self.mask_predictor.conv_logits,
        ]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("faster_rcnn".to_string(), &self.faster_rcnn),
            ("mask_head.conv1".to_string(), &self.mask_head.conv1),
            ("mask_head.conv2".to_string(), &self.mask_head.conv2),
            ("mask_head.conv3".to_string(), &self.mask_head.conv3),
            ("mask_head.conv4".to_string(), &self.mask_head.conv4),
            (
                "mask_predictor.deconv".to_string(),
                &self.mask_predictor.deconv,
            ),
            (
                "mask_predictor.conv_logits".to_string(),
                &self.mask_predictor.conv_logits,
            ),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.faster_rcnn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.faster_rcnn.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Convenience constructor
// ---------------------------------------------------------------------------

/// Construct a Mask R-CNN with ResNet-50 FPN backbone.
///
/// `num_classes` includes the background class (index 0), matching
/// `torchvision.models.detection.maskrcnn_resnet50_fpn(num_classes=91)`.
pub fn maskrcnn_resnet50_fpn<T: Float>(num_classes: usize) -> FerrotorchResult<MaskRcnn<T>> {
    MaskRcnn::new(num_classes)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// FPN level assignment for mask ROIs.
///
/// Same formula as Faster R-CNN's `assign_fpn_levels`.
fn assign_fpn_levels_mask<T: Float>(
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

/// Extract item `b` from a `[B, C, H, W]` tensor → `[1, C, H, W]`.
fn slice_batch_item_mask<T: Float>(t: &Tensor<T>, b: usize) -> FerrotorchResult<Tensor<T>> {
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

    fn make_model() -> MaskRcnn<f32> {
        maskrcnn_resnet50_fpn::<f32>(91).unwrap()
    }

    #[test]
    fn test_mask_rcnn_constructs() {
        let model = make_model();
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_mask_rcnn_param_count_ballpark() {
        // ResNet-50 (~25.5M) + FPN (~3.3M) + RPN (~1.2M)
        // + TwoMlpHead for 91 classes (~13M) + mask head (~2.4M)
        // + mask predictor (~200K). Total ~45M.
        // Accepted range: 40M–85M to allow for 91 vs other num_classes.
        let model = make_model();
        let np = model.num_parameters();
        assert!(np > 40_000_000, "param count too low: {np}");
        assert!(np < 85_000_000, "param count too high: {np}");
    }

    #[test]
    fn test_mask_rcnn_named_params_prefixes() {
        let model = make_model();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("faster_rcnn.")));
        assert!(names.iter().any(|n| n.starts_with("mask_head.")));
        assert!(names.iter().any(|n| n.starts_with("mask_predictor.")));
    }

    #[test]
    fn test_mask_head_shapes() {
        let head = MaskHead::<f32>::new(256).unwrap();
        let x = ferrotorch_core::randn(&[4, 256, 14, 14]).unwrap();
        let out = head.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[4, 256, 14, 14],
            "mask head preserves spatial size"
        );
    }

    #[test]
    fn test_mask_predictor_shapes() {
        let predictor = MaskPredictor::<f32>::new(256, 91).unwrap();
        let x = ferrotorch_core::randn(&[4, 256, 14, 14]).unwrap();
        let out = predictor.forward(&x).unwrap();
        assert_eq!(
            out.shape(),
            &[4, 91, 28, 28],
            "mask predictor 2x upsample + num_classes"
        );
    }

    #[test]
    fn test_mask_rcnn_forward_output_structure() {
        let model = make_model();
        let img = no_grad(|| ferrotorch_core::randn(&[1, 3, 64, 64]).unwrap());
        let dets = no_grad(|| model.forward(&img).unwrap());
        assert_eq!(dets.len(), 1, "one detection list per image");
        let d = &dets[0];
        let n = d.boxes.shape()[0];
        assert_eq!(d.boxes.shape().len(), 2);
        assert_eq!(d.boxes.shape()[1], 4);
        // Post-NMS scores: 1-D [N_det].
        assert_eq!(d.scores.shape().len(), 1);
        assert_eq!(d.scores.shape()[0], n);
        assert_eq!(d.labels.len(), n);
        // Background never appears post-postprocess.
        assert!(d.labels.iter().all(|&l| l >= 1));
        // Mask shape: [N_det, 1, H_img, W_img] (sigmoid + class-select +
        // `paste_masks_in_image`). Matches torchvision's
        // `model(img)[0]["masks"]` post-`GeneralizedRCNNTransform.postprocess`.
        assert_eq!(d.masks.shape()[0], n);
        assert_eq!(d.masks.shape()[1], 1);
        assert_eq!(d.masks.shape()[2], 64);
        assert_eq!(d.masks.shape()[3], 64);
    }

    #[test]
    fn test_mask_rcnn_module_forward_post_paste_shape() {
        // Locks the contract that `Module::forward` returns the POST-PASTE
        // mask tensor `[N_det, 1, H_img, W_img]` (matches torchvision
        // `model(img)[0]["masks"]`). Regression guard for #1141.
        let model = make_model();
        let img = no_grad(|| ferrotorch_core::randn(&[1, 3, 96, 128]).unwrap());
        let out = no_grad(|| <MaskRcnn<f32> as Module<f32>>::forward(&model, &img).unwrap());
        let s = out.shape();
        assert_eq!(s.len(), 4);
        assert_eq!(s[1], 1, "single mask channel post class-select");
        assert_eq!(s[2], 96, "mask height matches image height (post-paste)");
        assert_eq!(s[3], 128, "mask width matches image width (post-paste)");
    }

    #[test]
    fn test_mask_rcnn_two_images_batch() {
        let model = make_model();
        let imgs = no_grad(|| ferrotorch_core::randn(&[2, 3, 64, 64]).unwrap());
        let dets = no_grad(|| model.forward(&imgs).unwrap());
        assert_eq!(dets.len(), 2);
    }

    #[test]
    fn test_mask_rcnn_train_eval() {
        let mut model = make_model();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
    }
}
