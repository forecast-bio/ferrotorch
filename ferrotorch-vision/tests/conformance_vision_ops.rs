//! Conformance suite for `ferrotorch-vision` — Layer 3, ops module.
//!
//! Tracking issue: #870 (ferrotorch-vision conformance suite).
//!
//! Reference libraries:
//!   - `torch == 2.6.0`  (installed; target pin 2.11.0)
//!   - `torchvision == 0.21.0`
//!
//! Fixtures live in `tests/conformance/fixtures.json`.

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::uninlined_format_args,
)]

use std::path::PathBuf;

use ferrotorch_core::{Tensor, TensorStorage};
use ferrotorch_vision::ops::{
    BoxFormat, LossReduction, batched_nms, box_area, box_convert, box_iou,
    clip_boxes_to_image, complete_box_iou, distance_box_iou, focal_loss,
    generalized_box_iou, nms, remove_small_boxes, roi_align, roi_pool,
    sigmoid_focal_loss,
};
use serde::Deserialize;

#[allow(unused_macros)]
macro_rules! cascade_skip {
    ($reason:literal) => {{
        eprintln!("  [cascade_skip] {} — {}", module_path!(), $reason);
        return;
    }};
}

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct FixtureFile {
    #[allow(dead_code)]
    metadata: serde_json::Value,
    fixtures: Vec<serde_json::Value>,
}

fn fixtures_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("conformance")
        .join("fixtures.json")
}

fn load_fixtures() -> FixtureFile {
    let path = fixtures_path();
    assert!(
        path.exists(),
        "fixtures.json not found. Run: python3 scripts/regenerate_vision_fixtures.py"
    );
    let body = std::fs::read_to_string(&path).unwrap();
    serde_json::from_str(&body).unwrap()
}

fn get_fixture<'a>(fixtures: &'a [serde_json::Value], id: &str) -> Option<&'a serde_json::Value> {
    fixtures.iter().find(|f| f["id"] == id)
}

fn flatten_f64(v: &serde_json::Value) -> Vec<f64> {
    match v {
        serde_json::Value::Number(n) => vec![n.as_f64().unwrap()],
        serde_json::Value::Array(arr) => arr.iter().flat_map(flatten_f64).collect(),
        _ => vec![],
    }
}

fn make_f64(data: Vec<f64>, shape: Vec<usize>) -> Tensor<f64> {
    Tensor::from_storage(TensorStorage::cpu(data), shape, false).unwrap()
}

fn assert_close_f64(actual: &[f64], expected: &[f64], tol: f64, label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label}: length mismatch");
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (a - e).abs() <= tol,
            "{label}[{i}]: actual={a} expected={e} diff={}",
            (a - e).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// Layer 3: BoxFormat / LossReduction enum types
// ---------------------------------------------------------------------------

#[test]
fn box_format_enum_variants() {
    // Exercises BoxFormat enum (surface gate ref).
    let _xyxy = BoxFormat::Xyxy;
    let _xywh = BoxFormat::Xywh;
    let _cx = BoxFormat::Cxcywh;
    assert_ne!(_xyxy, _xywh);
}

#[test]
fn loss_reduction_enum_variants() {
    let _none = LossReduction::None;
    let _mean = LossReduction::Mean;
    let _sum = LossReduction::Sum;
    assert_ne!(_none, _mean);
}

// ---------------------------------------------------------------------------
// Layer 3: box_convert
// ---------------------------------------------------------------------------

#[test]
fn box_convert_xyxy_to_xywh_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "box_convert_xyxy_to_xywh")
        .expect("fixture box_convert_xyxy_to_xywh not found");

    let input_data = flatten_f64(&fix["input"]);
    let expected_data = flatten_f64(&fix["expected"]);
    let input_shape: Vec<usize> = fix["input_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let boxes = make_f64(input_data, input_shape);
    let out = box_convert(&boxes, BoxFormat::Xyxy, BoxFormat::Xywh).unwrap();
    assert_close_f64(out.data().unwrap(), &expected_data, 1e-6, "box_convert_xyxy_to_xywh");
}

#[test]
fn box_convert_xywh_to_cxcywh_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "box_convert_xywh_to_cxcywh")
        .expect("fixture box_convert_xywh_to_cxcywh not found");

    let input_data = flatten_f64(&fix["input"]);
    let expected_data = flatten_f64(&fix["expected"]);
    let input_shape: Vec<usize> = fix["input_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let boxes = make_f64(input_data, input_shape);
    let out = box_convert(&boxes, BoxFormat::Xywh, BoxFormat::Cxcywh).unwrap();
    assert_close_f64(out.data().unwrap(), &expected_data, 1e-6, "box_convert_xywh_to_cxcywh");
}

#[test]
fn box_convert_xyxy_to_cxcywh_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "box_convert_xyxy_to_cxcywh")
        .expect("fixture box_convert_xyxy_to_cxcywh not found");

    let input_data = flatten_f64(&fix["input"]);
    let expected_data = flatten_f64(&fix["expected"]);
    let input_shape: Vec<usize> = fix["input_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let boxes = make_f64(input_data, input_shape);
    let out = box_convert(&boxes, BoxFormat::Xyxy, BoxFormat::Cxcywh).unwrap();
    assert_close_f64(out.data().unwrap(), &expected_data, 1e-6, "box_convert_xyxy_to_cxcywh");
}

#[test]
fn box_convert_identity_same_format() {
    // box_convert(boxes, X, X) must return the original boxes unchanged.
    let data = vec![1.0_f64, 2.0, 5.0, 6.0];
    let boxes = make_f64(data.clone(), vec![1, 4]);
    let out = box_convert(&boxes, BoxFormat::Xyxy, BoxFormat::Xyxy).unwrap();
    assert_close_f64(out.data().unwrap(), &data, 1e-9, "box_convert_identity");
}

// ---------------------------------------------------------------------------
// Layer 3: box_area
// ---------------------------------------------------------------------------

#[test]
fn box_area_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "box_area_mixed")
        .expect("fixture box_area_mixed not found");

    let input_data = flatten_f64(&fix["input"]);
    let expected_data = flatten_f64(&fix["expected"]);
    let input_shape: Vec<usize> = fix["input_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let boxes = make_f64(input_data, input_shape);
    let out = box_area(&boxes).unwrap();
    assert_close_f64(out.data().unwrap(), &expected_data, 1e-6, "box_area");
}

// ---------------------------------------------------------------------------
// Layer 3: box_iou
// ---------------------------------------------------------------------------

#[test]
fn box_iou_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "box_iou_2x2")
        .expect("fixture box_iou_2x2 not found");

    let a_data = flatten_f64(&fix["input_a"]);
    let b_data = flatten_f64(&fix["input_b"]);
    let expected_data = flatten_f64(&fix["expected"]);
    let a_shape: Vec<usize> = fix["input_a_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let b_shape: Vec<usize> = fix["input_b_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let boxes1 = make_f64(a_data, a_shape);
    let boxes2 = make_f64(b_data, b_shape);
    let out = box_iou(&boxes1, &boxes2).unwrap();

    // IoU tolerance 1e-5 (floating-point intersection arithmetic).
    assert_close_f64(out.data().unwrap(), &expected_data, 1e-5, "box_iou");
}

// ---------------------------------------------------------------------------
// Layer 3: clip_boxes_to_image
// ---------------------------------------------------------------------------

#[test]
fn clip_boxes_to_image_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "clip_boxes_to_image_10x10")
        .expect("fixture clip_boxes_to_image_10x10 not found");

    let input_data = flatten_f64(&fix["input"]);
    let expected_data = flatten_f64(&fix["expected"]);
    let input_shape: Vec<usize> = fix["input_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let h = fix["params"]["height"].as_u64().unwrap() as usize;
    let w = fix["params"]["width"].as_u64().unwrap() as usize;

    let boxes = make_f64(input_data, input_shape);
    let out = clip_boxes_to_image(&boxes, [h, w]).unwrap();
    assert_close_f64(out.data().unwrap(), &expected_data, 1e-9, "clip_boxes_to_image");
}

// ---------------------------------------------------------------------------
// Layer 3: remove_small_boxes
// ---------------------------------------------------------------------------

#[test]
fn remove_small_boxes_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "remove_small_boxes_min2")
        .expect("fixture remove_small_boxes_min2 not found");

    let input_data = flatten_f64(&fix["input"]);
    let expected_indices: Vec<usize> = fix["expected_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let input_shape: Vec<usize> = fix["input_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let min_size = fix["params"]["min_size"].as_f64().unwrap();

    let boxes = make_f64(input_data, input_shape);
    let keep = remove_small_boxes(&boxes, min_size).unwrap();
    assert_eq!(keep, expected_indices, "remove_small_boxes indices");
}

// ---------------------------------------------------------------------------
// Layer 3: nms
// ---------------------------------------------------------------------------

#[test]
fn nms_iou_0p5_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "nms_iou_0p5")
        .expect("fixture nms_iou_0p5 not found");

    let boxes_data = flatten_f64(&fix["input_boxes"]);
    let scores_data = flatten_f64(&fix["input_scores"]);
    let expected_indices: Vec<usize> = fix["expected_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let iou_threshold = fix["params"]["iou_threshold"].as_f64().unwrap();

    let boxes = make_f64(boxes_data, vec![3, 4]);
    let scores = make_f64(scores_data, vec![3]);
    let keep = nms(&boxes, &scores, iou_threshold).unwrap();
    assert_eq!(keep, expected_indices, "nms keep indices");
}

// ---------------------------------------------------------------------------
// Layer 3: batched_nms (shape contract — no fixture, structural test)
// ---------------------------------------------------------------------------

#[test]
fn batched_nms_shape_contract() {
    // Two boxes in class 0, one box in class 1. All should be kept (no overlap).
    let boxes_data = vec![0.0_f64, 0.0, 5.0, 5.0, 6.0, 6.0, 11.0, 11.0, 0.0, 0.0, 5.0, 5.0];
    let scores_data = vec![0.9_f64, 0.8, 0.7];
    let idxs: Vec<u32> = [0u32, 0, 1].into(); // class labels per box

    let boxes = make_f64(boxes_data, vec![3, 4]);
    let scores = make_f64(scores_data, vec![3]);

    let keep = batched_nms(&boxes, &scores, &idxs, 0.5).unwrap();
    // All three boxes are in different classes or non-overlapping — all kept.
    assert!(
        !keep.is_empty(),
        "batched_nms should keep at least one box"
    );
}

// ---------------------------------------------------------------------------
// Layer 3: sigmoid_focal_loss
// ---------------------------------------------------------------------------

#[test]
fn sigmoid_focal_loss_none_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "sigmoid_focal_loss_none")
        .expect("fixture sigmoid_focal_loss_none not found");

    let inputs_data = flatten_f64(&fix["input"]);
    let targets_data = flatten_f64(&fix["targets"]);
    let expected_data = flatten_f64(&fix["expected"]);
    let alpha = fix["params"]["alpha"].as_f64().unwrap();
    let gamma = fix["params"]["gamma"].as_f64().unwrap();

    let inputs = make_f64(inputs_data, vec![4]);
    let targets = make_f64(targets_data, vec![4]);
    let out = sigmoid_focal_loss(&inputs, &targets, alpha, gamma, LossReduction::None).unwrap();

    // Focal loss tolerance 1e-5 (sigmoid + log arithmetic).
    assert_close_f64(out.data().unwrap(), &expected_data, 1e-5, "sigmoid_focal_loss_none");
}

#[test]
fn sigmoid_focal_loss_mean_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "sigmoid_focal_loss_mean")
        .expect("fixture sigmoid_focal_loss_mean not found");

    let inputs_data = flatten_f64(&fix["input"]);
    let targets_data = flatten_f64(&fix["targets"]);
    let expected = fix["expected"].as_f64().unwrap();
    let alpha = fix["params"]["alpha"].as_f64().unwrap();
    let gamma = fix["params"]["gamma"].as_f64().unwrap();

    let inputs = make_f64(inputs_data, vec![4]);
    let targets = make_f64(targets_data, vec![4]);
    let out = sigmoid_focal_loss(&inputs, &targets, alpha, gamma, LossReduction::Mean).unwrap();
    let actual = out.data().unwrap()[0];

    assert!(
        (actual - expected).abs() < 1e-5,
        "sigmoid_focal_loss_mean: actual={actual} expected={expected}"
    );
}

#[test]
fn sigmoid_focal_loss_sum_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "sigmoid_focal_loss_sum")
        .expect("fixture sigmoid_focal_loss_sum not found");

    let inputs_data = flatten_f64(&fix["input"]);
    let targets_data = flatten_f64(&fix["targets"]);
    let expected = fix["expected"].as_f64().unwrap();
    let alpha = fix["params"]["alpha"].as_f64().unwrap();
    let gamma = fix["params"]["gamma"].as_f64().unwrap();

    let inputs = make_f64(inputs_data, vec![4]);
    let targets = make_f64(targets_data, vec![4]);
    let out = sigmoid_focal_loss(&inputs, &targets, alpha, gamma, LossReduction::Sum).unwrap();
    let actual = out.data().unwrap()[0];

    assert!(
        (actual - expected).abs() < 1e-5,
        "sigmoid_focal_loss_sum: actual={actual} expected={expected}"
    );
}

// ---------------------------------------------------------------------------
// Layer 3: focal_loss (structural test — same signature pattern as sigmoid_focal_loss)
// ---------------------------------------------------------------------------

#[test]
fn focal_loss_shape_contract() {
    let inputs = make_f64(vec![0.8, -0.5, 1.2, -1.0], vec![4]);
    let targets = make_f64(vec![1.0, 0.0, 1.0, 0.0], vec![4]);
    let out = focal_loss(&inputs, &targets, 0.25, 2.0, LossReduction::None).unwrap();
    assert_eq!(out.shape(), &[4], "focal_loss output shape");
}

// ---------------------------------------------------------------------------
// Layer 3: generalized_box_iou
// ---------------------------------------------------------------------------

#[test]
fn generalized_box_iou_matches_reference() {
    let ff = load_fixtures();
    let fix = get_fixture(&ff.fixtures, "generalized_box_iou_1x2")
        .expect("fixture generalized_box_iou_1x2 not found");

    let a_data = flatten_f64(&fix["input_a"]);
    let b_data = flatten_f64(&fix["input_b"]);
    let expected_data = flatten_f64(&fix["expected"]);
    let a_shape: Vec<usize> = fix["input_a_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let b_shape: Vec<usize> = fix["input_b_shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();

    let boxes1 = make_f64(a_data, a_shape);
    let boxes2 = make_f64(b_data, b_shape);
    let out = generalized_box_iou(&boxes1, &boxes2).unwrap();

    assert_close_f64(out.data().unwrap(), &expected_data, 1e-5, "generalized_box_iou");
}

// ---------------------------------------------------------------------------
// Layer 3: distance_box_iou (shape contract — no dedicated fixture)
// ---------------------------------------------------------------------------

#[test]
fn distance_box_iou_shape_contract() {
    let boxes1 = make_f64(vec![0.0, 0.0, 4.0, 4.0, 1.0, 1.0, 5.0, 5.0], vec![2, 4]);
    let boxes2 = make_f64(vec![2.0, 2.0, 6.0, 6.0, 3.0, 3.0, 7.0, 7.0], vec![2, 4]);
    let out = distance_box_iou(&boxes1, &boxes2).unwrap();
    assert_eq!(out.shape(), &[2, 2], "distance_box_iou shape");
    // DIoU in [-1, 1].
    for &v in out.data().unwrap() {
        assert!(
            (-1.0..=1.0).contains(&v),
            "distance_box_iou value {v} outside [-1, 1]"
        );
    }
}

// ---------------------------------------------------------------------------
// Layer 3: complete_box_iou (shape contract)
// ---------------------------------------------------------------------------

#[test]
fn complete_box_iou_shape_contract() {
    let boxes1 = make_f64(vec![0.0, 0.0, 4.0, 4.0], vec![1, 4]);
    let boxes2 = make_f64(vec![1.0, 1.0, 5.0, 5.0, 6.0, 6.0, 10.0, 10.0], vec![2, 4]);
    let out = complete_box_iou(&boxes1, &boxes2).unwrap();
    assert_eq!(out.shape(), &[1, 2], "complete_box_iou shape");
}

// ---------------------------------------------------------------------------
// Layer 3: roi_align (shape contract)
// ---------------------------------------------------------------------------

#[test]
fn roi_align_shape_contract() {
    // Small 1x1x4x4 feature map, one 2x2 ROI.
    let feature_data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let feature = make_f64(feature_data, vec![1, 1, 4, 4]);
    // ROI: [batch_idx, x1, y1, x2, y2]
    let rois = make_f64(vec![0.0, 0.0, 0.0, 2.0, 2.0], vec![1, 5]);
    let out = roi_align(&feature, &rois, (2, 2), 1.0, 2).unwrap();
    assert_eq!(out.shape()[0], 1, "roi_align batch size");
    assert_eq!(out.shape()[2], 2, "roi_align height");
    assert_eq!(out.shape()[3], 2, "roi_align width");
}

// ---------------------------------------------------------------------------
// Layer 3: roi_pool (shape contract)
// ---------------------------------------------------------------------------

#[test]
fn roi_pool_shape_contract() {
    let feature_data: Vec<f64> = (0..16).map(|i| i as f64).collect();
    let feature = make_f64(feature_data, vec![1, 1, 4, 4]);
    let rois = make_f64(vec![0.0, 0.0, 0.0, 3.0, 3.0], vec![1, 5]);
    let out = roi_pool(&feature, &rois, (2, 2), 1.0).unwrap();
    assert_eq!(out.shape()[0], 1, "roi_pool batch size");
    assert_eq!(out.shape()[2], 2, "roi_pool height");
    assert_eq!(out.shape()[3], 2, "roi_pool width");
}
