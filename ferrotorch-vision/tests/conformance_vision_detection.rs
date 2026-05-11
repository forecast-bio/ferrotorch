//! Conformance suite for Faster R-CNN with ResNet-50 FPN backbone.
//!
//! Tracking issue: #456 (partial — Faster R-CNN only).
//! Mask R-CNN deferred to #456-mask; SSD deferred to #456-ssd.
//!
//! Reference: `torchvision 0.21.x fasterrcnn_resnet50_fpn(weights=None)`.
//!
//! ## Scope
//!
//! - Architecture inventory: each sub-module constructs successfully and has
//!   expected parameter counts.
//! - Forward parity: synthetic 64x64 and 512x512 RGB tensors with seeded
//!   weights -> forward completes without error.
//! - Output structure: boxes [N,4], scores [N,num_classes], labels [N].
//! - FPN level shapes and channel counts.
//! - RPN produces proposals on a non-trivial input.
//! - Anchor generation counts match torchvision defaults.
//! - Model registry: `fasterrcnn_resnet50_fpn` appears in `list_models()`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::uninlined_format_args
)]

use ferrotorch_core::{no_grad, randn};
use ferrotorch_nn::Module;
use ferrotorch_vision::models::detection::{
    AnchorGenerator, FeaturePyramidNetwork, MaskHead, MaskPredictor, TwoMlpHead,
    fasterrcnn_resnet50_fpn, maskrcnn_resnet50_fpn,
};
use ferrotorch_vision::{get_model, list_models};

// ---------------------------------------------------------------------------
// Helper: small fake backbone features for FPN tests
// ---------------------------------------------------------------------------

fn fake_backbone(batch: usize) -> std::collections::HashMap<String, ferrotorch_core::Tensor<f32>> {
    let mut m = std::collections::HashMap::new();
    m.insert("layer1".into(), randn(&[batch, 256, 16, 16]).unwrap());
    m.insert("layer2".into(), randn(&[batch, 512, 8, 8]).unwrap());
    m.insert("layer3".into(), randn(&[batch, 1024, 4, 4]).unwrap());
    m.insert("layer4".into(), randn(&[batch, 2048, 2, 2]).unwrap());
    m
}

// ===========================================================================
// Architecture inventory
// ===========================================================================

/// C1 - Anchor generator produces the torchvision-default count.
///
/// torchvision default: 5 levels x 3 aspect ratios = 3 anchors/cell.
/// At feature-map size 1x1 per level: 5 x 1 x 1 x 3 = 15 anchors total.
#[test]
fn test_anchor_generator_torchvision_default_count() {
    let anchor_gen = AnchorGenerator::default_fasterrcnn();
    let fm_sizes = vec![(1, 1); 5];
    let anchors = anchor_gen.generate_anchors::<f32>(&fm_sizes).unwrap();
    assert_eq!(
        anchors.shape(),
        &[15, 4],
        "15 anchors for 5 levels x 3 ratios x 1x1"
    );
}

/// C2 - FPN constructs and produces 5 output levels with 256 channels each.
#[test]
fn test_fpn_construction_and_output_channels() {
    let fpn = FeaturePyramidNetwork::<f32>::new().unwrap();
    let features = fake_backbone(1);
    let out = no_grad(|| fpn.forward(&features).unwrap());

    for key in ["p2", "p3", "p4", "p5", "p6"] {
        let t = out
            .get(key)
            .unwrap_or_else(|| panic!("FPN missing level {key}"));
        assert_eq!(
            t.shape()[1],
            256,
            "{key} should have 256 channels, got {}",
            t.shape()[1]
        );
    }
}

/// C3 - FPN spatial sizes follow backbone strides (p2=16x16 ... p6=1x1
/// for a 64-pixel backbone input given 16x16 layer1 feature maps).
#[test]
fn test_fpn_spatial_sizes() {
    let fpn = FeaturePyramidNetwork::<f32>::new().unwrap();
    let features = fake_backbone(1);
    let out = no_grad(|| fpn.forward(&features).unwrap());

    assert_eq!(out["p2"].shape(), &[1, 256, 16, 16], "p2 spatial mismatch");
    assert_eq!(out["p3"].shape(), &[1, 256, 8, 8], "p3 spatial mismatch");
    assert_eq!(out["p4"].shape(), &[1, 256, 4, 4], "p4 spatial mismatch");
    assert_eq!(out["p5"].shape(), &[1, 256, 2, 2], "p5 spatial mismatch");
    assert_eq!(out["p6"].shape(), &[1, 256, 1, 1], "p6 spatial mismatch");
}

/// C4 - RPN head produces correct objectness and delta shapes.
#[test]
fn test_rpn_head_shapes() {
    use ferrotorch_vision::models::detection::RpnHead;
    let head = RpnHead::<f32>::new(256, 3).unwrap();
    let x = randn(&[1, 256, 8, 8]).unwrap();
    let (logits, deltas) = head.forward_level(&x).unwrap();
    assert_eq!(logits.shape(), &[1, 3, 8, 8], "objectness shape");
    assert_eq!(deltas.shape(), &[1, 12, 8, 8], "delta shape (3x4)");
}

/// C5 - TwoMlpHead produces class logits and box deltas of correct shape.
#[test]
fn test_two_mlp_head_shapes() {
    let head = TwoMlpHead::<f32>::new(7, 256, 1024, 91).unwrap();
    let features = randn(&[4, 256, 7, 7]).unwrap();
    let (cls, bbox) = head.forward(&features).unwrap();
    assert_eq!(cls.shape(), &[4, 91], "class logits shape");
    assert_eq!(bbox.shape(), &[4, 91 * 4], "bbox delta shape");
}

/// C6 - Full model constructs with correct named-parameter prefixes.
#[test]
fn test_fasterrcnn_named_parameter_prefixes() {
    let model = fasterrcnn_resnet50_fpn::<f32>(91).unwrap();
    let names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(
        names.iter().any(|n| n.starts_with("backbone.")),
        "missing backbone params"
    );
    assert!(
        names.iter().any(|n| n.starts_with("fpn.")),
        "missing fpn params"
    );
    assert!(
        names.iter().any(|n| n.starts_with("rpn.")),
        "missing rpn params"
    );
    assert!(
        names.iter().any(|n| n.starts_with("head.")),
        "missing head params"
    );
}

// ===========================================================================
// Forward parity - structure invariants
// ===========================================================================

/// P1 - Single-image 64x64 forward: output structure is correct.
///
/// Post-#1141 contract: `Detections` mirrors `torchvision`'s
/// `RoIHeads.postprocess_detections` output — one (box, score, label) per
/// surviving post-NMS detection. Per-class softmax is an internal intermediate
/// and is no longer exposed through `FasterRcnn::forward`. (Use
/// `TwoMlpHead::forward` for the pre-softmax class logits.)
#[test]
fn test_forward_64x64_output_structure() {
    let model = fasterrcnn_resnet50_fpn::<f32>(91).unwrap();
    let img = no_grad(|| randn(&[1, 3, 64, 64]).unwrap());
    let dets = no_grad(|| model.forward(&img).unwrap());

    assert_eq!(dets.len(), 1, "one detection list per image");
    let d = &dets[0];
    let n = d.boxes.shape()[0];
    assert_eq!(d.boxes.shape().len(), 2, "boxes must be 2-D");
    assert_eq!(d.boxes.shape()[1], 4, "boxes must have 4 coords");
    assert_eq!(d.scores.shape().len(), 1, "scores must be 1-D (one per detection)");
    assert_eq!(
        d.scores.shape()[0],
        n,
        "scores count must match box count"
    );
    assert_eq!(d.labels.len(), n, "label count must match box count");
}

/// P2 - Batch of 2 images: two detection lists returned.
#[test]
fn test_forward_batch2_returns_two_detection_lists() {
    let model = fasterrcnn_resnet50_fpn::<f32>(91).unwrap();
    let imgs = no_grad(|| randn(&[2, 3, 64, 64]).unwrap());
    let dets = no_grad(|| model.forward(&imgs).unwrap());
    assert_eq!(
        dets.len(),
        2,
        "expect one detection list per image in batch"
    );
}

/// P3 - Score values are valid probabilities in `[0, 1]`.
///
/// Post-#1141: `Detections.scores` is `[N_det]` — the softmax probability of
/// the predicted class for each surviving detection. Background (class 0) is
/// dropped, so scores are bounded by `1.0` from above and `score_thresh` from
/// below.
#[test]
fn test_forward_scores_are_probabilities() {
    let model = fasterrcnn_resnet50_fpn::<f32>(91).unwrap();
    let img = no_grad(|| randn(&[1, 3, 64, 64]).unwrap());
    let dets = no_grad(|| model.forward(&img).unwrap());
    let d = &dets[0];
    if d.scores.shape()[0] > 0 {
        let data = d.scores.data_vec().unwrap();
        for (i, &s) in data.iter().enumerate() {
            assert!(
                (0.0..=1.0).contains(&s),
                "score[{i}] = {s} is not a valid probability in [0, 1]"
            );
        }
    }
}

/// P4 - All returned boxes are within the image boundaries.
#[test]
fn test_forward_boxes_within_image() {
    let model = fasterrcnn_resnet50_fpn::<f32>(91).unwrap();
    let h = 64usize;
    let w = 64usize;
    let img = no_grad(|| randn(&[1, 3, h, w]).unwrap());
    let dets = no_grad(|| model.forward(&img).unwrap());
    let d = &dets[0];
    if d.boxes.shape()[0] > 0 {
        let data = d.boxes.data_vec().unwrap();
        let n = d.boxes.shape()[0];
        for i in 0..n {
            let x1 = data[i * 4];
            let y1 = data[i * 4 + 1];
            let x2 = data[i * 4 + 2];
            let y2 = data[i * 4 + 3];
            assert!(
                x1 >= 0.0 && x2 <= w as f32,
                "box {i}: x out of [0,{w}]: {x1}..{x2}"
            );
            assert!(
                y1 >= 0.0 && y2 <= h as f32,
                "box {i}: y out of [0,{h}]: {y1}..{y2}"
            );
        }
    }
}

/// P5 - Labels are in [0, num_classes).
#[test]
fn test_forward_labels_in_range() {
    let num_classes = 91usize;
    let model = fasterrcnn_resnet50_fpn::<f32>(num_classes).unwrap();
    let img = no_grad(|| randn(&[1, 3, 64, 64]).unwrap());
    let dets = no_grad(|| model.forward(&img).unwrap());
    let d = &dets[0];
    for &label in &d.labels {
        assert!(
            label < num_classes,
            "label {label} out of range [0, {num_classes})"
        );
    }
}

// ===========================================================================
// End-to-end: 512x512 synthetic forward
// ===========================================================================

/// E1 - Full 512x512 forward completes without error and returns detections.
///
/// Uses a synthetic random tensor (no pretrained weights). We assert
/// structure invariants only, not numerical values.
#[test]
fn test_forward_512x512_end_to_end() {
    let model = fasterrcnn_resnet50_fpn::<f32>(91).unwrap();
    let img = no_grad(|| randn(&[1, 3, 512, 512]).unwrap());
    let dets = no_grad(|| model.forward(&img).unwrap());

    assert_eq!(dets.len(), 1);
    let d = &dets[0];
    let n = d.boxes.shape()[0];
    assert_eq!(d.boxes.shape().len(), 2);
    assert_eq!(d.boxes.shape()[1], 4);
    assert_eq!(d.scores.shape().len(), 1);
    assert_eq!(d.scores.shape()[0], n);
    assert_eq!(d.labels.len(), n);
}

// ===========================================================================
// Model registry
// ===========================================================================

/// R1 - `fasterrcnn_resnet50_fpn` is listed in the global model registry.
#[test]
fn test_registry_contains_fasterrcnn() {
    let names = list_models().unwrap();
    assert!(
        names.contains(&"fasterrcnn_resnet50_fpn".to_string()),
        "fasterrcnn_resnet50_fpn missing from registry; got: {names:?}"
    );
}

/// R2 - Registry construct with `pretrained=false` succeeds.
#[test]
fn test_registry_get_model_pretrained_false() {
    let result = get_model("fasterrcnn_resnet50_fpn", false, 91);
    assert!(
        result.is_ok(),
        "registry get_model fasterrcnn_resnet50_fpn pretrained=false failed: {:?}",
        result.err()
    );
}

/// R3 - Hub registry has an entry for `fasterrcnn_resnet50_fpn`.
#[test]
fn test_hub_registry_entry_exists() {
    let info = ferrotorch_hub::registry::get_model_info("fasterrcnn_resnet50_fpn");
    assert!(
        info.is_some(),
        "fasterrcnn_resnet50_fpn missing from ferrotorch_hub::registry"
    );
}

// ===========================================================================
// Parameter count sanity
// ===========================================================================

/// Q1 - FPN parameter count is in the expected range (~3.3M).
#[test]
fn test_fpn_parameter_count() {
    let fpn = FeaturePyramidNetwork::<f32>::new().unwrap();
    let np: usize = fpn.parameters().iter().map(|p| p.numel()).sum();
    assert!(np > 3_000_000, "FPN params too low: {np}");
    assert!(np < 4_000_000, "FPN params too high: {np}");
}

/// Q2 - Full model parameter count is in the expected range (40M-80M).
#[test]
fn test_full_model_parameter_count() {
    let model = fasterrcnn_resnet50_fpn::<f32>(91).unwrap();
    let np = model.num_parameters();
    assert!(np > 40_000_000, "FasterRCNN params too low: {np}");
    assert!(np < 80_000_000, "FasterRCNN params too high: {np}");
}

// ===========================================================================
// Train / eval state
// ===========================================================================

/// T1 - Model starts in eval mode; train/eval toggle works.
#[test]
fn test_train_eval_toggle() {
    let mut model = fasterrcnn_resnet50_fpn::<f32>(91).unwrap();
    assert!(!model.is_training(), "model should start in eval mode");
    model.train();
    assert!(
        model.is_training(),
        "model.train() should set training=true"
    );
    model.eval();
    assert!(
        !model.is_training(),
        "model.eval() should set training=false"
    );
}

// ===========================================================================
// Mask R-CNN conformance tests (M-series)
// Sprint VD.1 — #964
// ===========================================================================

/// M1 - MaskHead constructs and preserves spatial dimensions.
///
/// 4 conv layers with 3×3 kernel + pad=1: input 14×14 → output 14×14.
#[test]
fn test_mask_head_spatial_preservation() {
    let head = MaskHead::<f32>::new(256).unwrap();
    let x = randn(&[4, 256, 14, 14]).unwrap();
    let out = no_grad(|| head.forward(&x).unwrap());
    assert_eq!(
        out.shape(),
        &[4, 256, 14, 14],
        "MaskHead must preserve 14×14 spatial size"
    );
}

/// M2 - MaskPredictor doubles spatial resolution via deconv.
///
/// Input [N, 256, 14, 14] → output [N, num_classes, 28, 28].
#[test]
fn test_mask_predictor_output_shape() {
    let predictor = MaskPredictor::<f32>::new(256, 91).unwrap();
    let x = randn(&[4, 256, 14, 14]).unwrap();
    let out = no_grad(|| predictor.forward(&x).unwrap());
    assert_eq!(
        out.shape(),
        &[4, 91, 28, 28],
        "MaskPredictor must output [N, num_classes, 28, 28]"
    );
}

/// M3 - Full Mask R-CNN forward: output structure is correct.
///
/// Post-#1141 contract (matches torchvision's `model(img)[0]` after
/// `GeneralizedRCNNTransform.postprocess` + `paste_masks_in_image`):
/// - boxes  [N_det, 4]
/// - scores [N_det]               (1-D top-1 softmax probability per detection)
/// - labels [N_det]               (top-1 class per detection)
/// - masks  [N_det, 1, H_img, W_img]  (sigmoid + class-select + paste-back)
#[test]
fn test_maskrcnn_forward_output_structure() {
    let h = 64usize;
    let w = 64usize;
    let model = maskrcnn_resnet50_fpn::<f32>(91).unwrap();
    let img = no_grad(|| randn(&[1, 3, h, w]).unwrap());
    let dets = no_grad(|| model.forward(&img).unwrap());

    assert_eq!(dets.len(), 1, "one MaskDetections per image");
    let d = &dets[0];
    let n = d.boxes.shape()[0];
    assert_eq!(d.boxes.shape().len(), 2, "boxes must be 2-D");
    assert_eq!(d.boxes.shape()[1], 4, "boxes must have 4 coords");
    assert_eq!(d.scores.shape().len(), 1, "scores must be 1-D");
    assert_eq!(d.scores.shape()[0], n, "scores count must match box count");
    assert_eq!(d.labels.len(), n, "label count must match box count");
    assert_eq!(
        d.masks.shape()[0],
        n,
        "mask batch dim must match detection count"
    );
    assert_eq!(
        d.masks.shape()[1],
        1,
        "post-paste masks have a single class channel (class-selected)"
    );
    assert_eq!(d.masks.shape()[2], h, "pasted mask height must equal image height");
    assert_eq!(d.masks.shape()[3], w, "pasted mask width must equal image width");
}

/// M4 - Batch of 2 images returns two MaskDetections entries.
#[test]
fn test_maskrcnn_forward_batch2() {
    let model = maskrcnn_resnet50_fpn::<f32>(91).unwrap();
    let imgs = no_grad(|| randn(&[2, 3, 64, 64]).unwrap());
    let dets = no_grad(|| model.forward(&imgs).unwrap());
    assert_eq!(
        dets.len(),
        2,
        "expect one MaskDetections per image in batch"
    );
}

/// M5 - Named parameters carry expected prefix hierarchy.
///
/// faster_rcnn.* covers backbone+FPN+RPN+head; mask_head.* and
/// mask_predictor.* are the new Sprint VD.1 layers.
#[test]
fn test_maskrcnn_named_parameter_prefixes() {
    let model = maskrcnn_resnet50_fpn::<f32>(91).unwrap();
    let names: Vec<String> = model
        .named_parameters()
        .into_iter()
        .map(|(n, _)| n)
        .collect();
    assert!(
        names.iter().any(|n| n.starts_with("faster_rcnn.")),
        "missing faster_rcnn.* params"
    );
    assert!(
        names.iter().any(|n| n.starts_with("mask_head.")),
        "missing mask_head.* params"
    );
    assert!(
        names.iter().any(|n| n.starts_with("mask_predictor.")),
        "missing mask_predictor.* params"
    );
}

/// M6 - Mask R-CNN total parameter count is in the expected range.
///
/// torchvision maskrcnn_resnet50_fpn reports ~44M trainable parameters
/// (ResNet-50 backbone + FPN + RPN + ROIAlign heads + Mask head/predictor).
/// Accepted range: 40M-50M to allow head-size variation across class counts.
#[test]
fn test_maskrcnn_parameter_count() {
    let model = maskrcnn_resnet50_fpn::<f32>(91).unwrap();
    let np = model.num_parameters();
    assert!(np > 40_000_000, "MaskRCNN param count too low: {np}");
    assert!(np < 50_000_000, "MaskRCNN param count too high: {np}");
}

/// M7 - Model registry contains `maskrcnn_resnet50_fpn`.
#[test]
fn test_registry_contains_maskrcnn() {
    let names = list_models().unwrap();
    assert!(
        names.contains(&"maskrcnn_resnet50_fpn".to_string()),
        "maskrcnn_resnet50_fpn missing from registry; got: {names:?}"
    );
}

/// M8 - Registry construct with `pretrained=false` succeeds.
#[test]
fn test_registry_maskrcnn_pretrained_false() {
    let result = get_model("maskrcnn_resnet50_fpn", false, 91);
    assert!(
        result.is_ok(),
        "registry get_model maskrcnn_resnet50_fpn pretrained=false failed: {:?}",
        result.err()
    );
}

/// M9 - Hub registry has an entry for `maskrcnn_resnet50_fpn`.
#[test]
fn test_hub_registry_maskrcnn_entry_exists() {
    let info = ferrotorch_hub::registry::get_model_info("maskrcnn_resnet50_fpn");
    assert!(
        info.is_some(),
        "maskrcnn_resnet50_fpn missing from ferrotorch_hub::registry"
    );
}

/// M10 - Train/eval toggle propagates through the full Mask R-CNN.
#[test]
fn test_maskrcnn_train_eval_toggle() {
    let mut model = maskrcnn_resnet50_fpn::<f32>(91).unwrap();
    assert!(!model.is_training(), "model should start in eval mode");
    model.train();
    assert!(
        model.is_training(),
        "model.train() should set training=true"
    );
    model.eval();
    assert!(
        !model.is_training(),
        "model.eval() should set training=false"
    );
}

/// M11 - 512×512 end-to-end forward completes without error.
///
/// Uses synthetic random weights (no pretrained download). Asserts
/// structure only, not numerical values. Matches the post-paste output
/// contract (`[N_det, 1, H_img, W_img]`) introduced in #1141.
#[test]
fn test_maskrcnn_forward_512x512_end_to_end() {
    let h = 512usize;
    let w = 512usize;
    let model = maskrcnn_resnet50_fpn::<f32>(91).unwrap();
    let img = no_grad(|| randn(&[1, 3, h, w]).unwrap());
    let dets = no_grad(|| model.forward(&img).unwrap());

    assert_eq!(dets.len(), 1);
    let d = &dets[0];
    let n = d.boxes.shape()[0];
    assert_eq!(d.boxes.shape()[1], 4);
    assert_eq!(d.scores.shape().len(), 1);
    assert_eq!(d.scores.shape()[0], n);
    assert_eq!(d.labels.len(), n);
    assert_eq!(d.masks.shape()[0], n);
    assert_eq!(d.masks.shape()[1], 1);
    assert_eq!(d.masks.shape()[2], h);
    assert_eq!(d.masks.shape()[3], w);
}
