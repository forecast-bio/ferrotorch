//! Inference-dump binary for crosslink #1139 verification.
//!
//! Loads one of the 5 pinned pretrained models, runs forward on a single image
//! (with model-appropriate preprocessing), and dumps the output to disk in a
//! deterministic format. The companion Python script in
//! `scripts/verify_pretrained_inference.py` reads these dumps and compares
//! them against torchvision reference outputs.
//!
//! Usage:
//! ```text
//! cargo run -p ferrotorch-vision --release --example inference_dump -- \
//!     --model <name> --image <path.jpg> --output <path.bin>
//! ```
//!
//! Output format (raw little-endian):
//!   [u32: ndim][u32 × ndim: dims][f32 × prod(dims): data]
//!
//! The dumper deliberately uses `vision::get_model("<name>", true, num_classes)`
//! (the architect-mandated path) so we exercise the registry weight-loading
//! pipeline. `Module::forward` returns:
//!   SSD300         → [N_det, num_classes]  (first-image per-anchor class scores)
//!   FasterRCNN     → [N_det, num_classes]  (first-image per-proposal class scores)
//!   MaskRCNN       → [N_det, num_classes, 28, 28]  (first-image mask logits)
//!   DeepLabV3      → [B, num_classes, H, W]  (per-pixel class logits)
//!   FCN            → [B, num_classes, H, W]  (per-pixel class logits)

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Tensor};
use ferrotorch_nn::{InterpolateMode, Module, interpolate};
use ferrotorch_vision::io::read_image_as_tensor;
use ferrotorch_vision::models::bn_buffer_loader::apply_bn_buffers_from_state_dict;
use ferrotorch_vision::models::detection::{
    FasterRcnn, KeypointRcnn, MaskRcnn, fasterrcnn_resnet50_fpn,
    keypointrcnn_resnet50_fpn, maskrcnn_resnet50_fpn,
};
use ferrotorch_vision::models::get_model;

fn parse_args() -> Result<(String, PathBuf, PathBuf), String> {
    let args: Vec<String> = env::args().collect();
    let mut model: Option<String> = None;
    let mut image: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                model = Some(
                    args.get(i + 1)
                        .ok_or("--model needs a value")?
                        .clone(),
                );
                i += 2;
            }
            "--image" => {
                image = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--image needs a value")?,
                ));
                i += 2;
            }
            "--output" => {
                output = Some(PathBuf::from(
                    args.get(i + 1).ok_or("--output needs a value")?,
                ));
                i += 2;
            }
            other => return Err(format!("unknown arg: {other}")),
        }
    }

    Ok((
        model.ok_or("--model required")?,
        image.ok_or("--image required")?,
        output.ok_or("--output required")?,
    ))
}

/// Number of classes per model, matching the registered pretrained weights.
fn num_classes_for(model: &str) -> Result<usize, String> {
    match model {
        "ssd300_vgg16" => Ok(91),
        "fasterrcnn_resnet50_fpn" => Ok(91),
        "maskrcnn_resnet50_fpn" => Ok(91),
        "deeplabv3_resnet50" => Ok(21),
        "fcn_resnet50" => Ok(21),
        // #1146: LRASPP MobileNetV3-Large COCO_WITH_VOC_LABELS_V1 — 21
        // Pascal VOC classes (background + 20 foreground).
        "lraspp_mobilenet_v3_large" => Ok(21),
        // #1143: RetinaNet COCO_V1 uses 91 classes with no explicit background
        // (sigmoid scoring is per-class).
        "retinanet_resnet50_fpn" => Ok(91),
        // #1144: FCOS COCO_V1 — same 91-class convention; sigmoid scoring
        // gated by centerness.
        "fcos_resnet50_fpn" => Ok(91),
        // #1145: KeypointRCNN COCO_V1 — 2 classes (background + person).
        // The keypoint count (17) is fixed in the architecture constructor.
        "keypointrcnn_resnet50_fpn" => Ok(2),
        other => Err(format!("unknown model: {other}")),
    }
}

/// Manually bilinear-resize a `[C, H, W]` tensor's spatial dims to `(out_h, out_w)`.
///
/// Mirrors `torch.nn.functional.interpolate(mode='bilinear', align_corners=False,
/// antialias=False)` exactly — used to avoid the nearest-neighbour ferrotorch
/// `Resize` transform that would diverge from torchvision's resizing policy.
fn bilinear_resize_chw_to_bchw(
    chw: &Tensor<f32>,
    out_h: usize,
    out_w: usize,
) -> FerrotorchResult<Tensor<f32>> {
    let shape = chw.shape().to_vec();
    if shape.len() != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("expected [C, H, W], got {shape:?}"),
        });
    }
    // Promote to [1, C, H, W] for interpolate.
    let data = chw.data_vec()?;
    let bchw = Tensor::from_storage(
        ferrotorch_core::TensorStorage::cpu(data),
        vec![1, shape[0], shape[1], shape[2]],
        false,
    )?;
    interpolate(
        &bchw,
        Some([out_h, out_w]),
        None,
        InterpolateMode::Bilinear,
        false,
    )
}

/// Per-channel normalize a `[1, C, H, W]` tensor in-place semantics with
/// `(x - mean) / std`.
fn normalize_bchw(
    bchw: &Tensor<f32>,
    mean: [f32; 3],
    std: [f32; 3],
) -> FerrotorchResult<Tensor<f32>> {
    let shape = bchw.shape().to_vec();
    let b = shape[0];
    let c = shape[1];
    let h = shape[2];
    let w = shape[3];
    if c != 3 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("normalize_bchw expects C=3, got shape {shape:?}"),
        });
    }
    let mut data = bchw.data_vec()?;
    let plane = h * w;
    for bi in 0..b {
        for ci in 0..3 {
            let base = (bi * c + ci) * plane;
            let m = mean[ci];
            let s = std[ci];
            for i in 0..plane {
                data[base + i] = (data[base + i] - m) / s;
            }
        }
    }
    Tensor::from_storage(
        ferrotorch_core::TensorStorage::cpu(data),
        shape,
        false,
    )
}

/// Build a `[1, 3, H_out, W_out]` input tensor following the model's
/// torchvision preprocessing recipe.
///
/// Detection (SSD/FasterRCNN/MaskRCNN): torchvision `ObjectDetection` transform
///   just rescales u8→f32 in `[0,1]`. The model's internal
///   `GeneralizedRCNNTransform` (FasterRCNN/MaskRCNN) or its anchor layout
///   (SSD) handles further resize/normalize.
///
/// Because ferrotorch's SSD/FasterRCNN/MaskRCNN do NOT include a
/// `GeneralizedRCNNTransform` (they expect already-preprocessed input — see
/// e.g. `Ssd300::forward` doc: `[B, 3, 300, 300] tensor (RGB, normalised to
/// ImageNet stats)`), we reproduce the recipe here:
///
/// - SSD300: bilinear-resize to 300×300, normalize with ImageNet stats.
///   (Note: torchvision SSD300 uses non-ImageNet stats; we follow
///   ferrotorch's documented expectation — this is itself a candidate for a
///   divergence diagnosis.)
/// - FasterRCNN/MaskRCNN: resize so min(H,W)=800 keeping aspect, max(H,W)≤1333,
///   normalize with ImageNet stats. Pad to multiple of 32 (FPN stride).
/// - DeepLabV3/FCN: resize shorter side to 520 keeping aspect, normalize
///   ImageNet stats.
fn preprocess_for_model(model: &str, raw_chw: Tensor<f32>) -> FerrotorchResult<Tensor<f32>> {
    let shape = raw_chw.shape().to_vec();
    let h_in = shape[1];
    let w_in = shape[2];

    match model {
        "ssd300_vgg16" => {
            let resized = bilinear_resize_chw_to_bchw(&raw_chw, 300, 300)?;
            normalize_bchw(&resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        }
        "fasterrcnn_resnet50_fpn"
        | "maskrcnn_resnet50_fpn"
        | "retinanet_resnet50_fpn"
        | "fcos_resnet50_fpn"
        | "keypointrcnn_resnet50_fpn" => {
            // torchvision GeneralizedRCNNTransform: scale so min side = 800,
            // max side ≤ 1333; preserve aspect ratio.
            let min_size = 800.0_f64;
            let max_size = 1333.0_f64;
            let h = h_in as f64;
            let w = w_in as f64;
            let s_min = min_size / h.min(w);
            let s_max = max_size / h.max(w);
            let scale = s_min.min(s_max);
            let out_h = (h * scale).round() as usize;
            let out_w = (w * scale).round() as usize;
            let resized = bilinear_resize_chw_to_bchw(&raw_chw, out_h, out_w)?;
            let normed =
                normalize_bchw(&resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])?;
            // Pad to multiple of 32 (FPN stride).
            let stride: usize = 32;
            let pad_h = out_h.div_ceil(stride) * stride;
            let pad_w = out_w.div_ceil(stride) * stride;
            if pad_h == out_h && pad_w == out_w {
                Ok(normed)
            } else {
                // Zero-pad on the bottom/right.
                let normed_data = normed.data_vec()?;
                let c = 3;
                let mut padded = vec![0.0_f32; c * pad_h * pad_w];
                for ci in 0..c {
                    for r in 0..out_h {
                        let src_base = (ci * out_h + r) * out_w;
                        let dst_base = (ci * pad_h + r) * pad_w;
                        padded[dst_base..dst_base + out_w]
                            .copy_from_slice(&normed_data[src_base..src_base + out_w]);
                    }
                }
                Tensor::from_storage(
                    ferrotorch_core::TensorStorage::cpu(padded),
                    vec![1, c, pad_h, pad_w],
                    false,
                )
            }
        }
        "deeplabv3_resnet50" | "fcn_resnet50" | "lraspp_mobilenet_v3_large" => {
            // torchvision SemanticSegmentation: resize shorter side to 520,
            // preserve aspect ratio. Same recipe for LRASPP (#1146 — the
            // mobilenetv3 backbone needs identical normalization since it
            // was trained on the same ImageNet statistics).
            let resize_size = 520.0_f64;
            let h = h_in as f64;
            let w = w_in as f64;
            let scale = resize_size / h.min(w);
            let out_h = (h * scale).round() as usize;
            let out_w = (w * scale).round() as usize;
            let resized = bilinear_resize_chw_to_bchw(&raw_chw, out_h, out_w)?;
            normalize_bchw(&resized, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        }
        other => Err(FerrotorchError::InvalidArgument {
            message: format!("unknown model: {other}"),
        }),
    }
}

/// Write a `Tensor<f32>` to disk as raw little-endian:
///   `[u32 ndim][u32 × ndim shape][f32 × numel data]`.
fn dump_tensor(t: &Tensor<f32>, path: &PathBuf) -> FerrotorchResult<()> {
    let shape = t.shape().to_vec();
    let data = t.data_vec()?;
    let mut f = File::create(path).map_err(|e| FerrotorchError::Internal {
        message: format!("failed to open output: {e}"),
    })?;
    let ndim = shape.len() as u32;
    f.write_all(&ndim.to_le_bytes())
        .map_err(|e| FerrotorchError::Internal {
            message: format!("write ndim: {e}"),
        })?;
    for d in &shape {
        let d32 = *d as u32;
        f.write_all(&d32.to_le_bytes())
            .map_err(|e| FerrotorchError::Internal {
                message: format!("write dim: {e}"),
            })?;
    }
    for v in &data {
        f.write_all(&v.to_le_bytes())
            .map_err(|e| FerrotorchError::Internal {
                message: format!("write val: {e}"),
            })?;
    }
    Ok(())
}

/// Build, weight-load, and run a MaskRcnn directly (not via the registry's
/// `Box<dyn Module>`) so we can extract the full `MaskDetections` (boxes +
/// scores + masks). The registry's `Module::forward` only returns masks
/// `[N_det, 1, H, W]`, which is insufficient for mAP-style object matching
/// (round-9 #1141: per-rank pairing is structurally wrong; the harness needs
/// per-detection boxes to pair rust↔tv by box-IoU > 0.5 before comparing
/// masks).
///
/// Replicates `models::registry::maybe_load_pretrained`'s weight-loading
/// exactly: hub lookup → safetensors load → `load_state_dict(strict=false)`
/// → BN buffer apply.
fn run_maskrcnn_dump(input: &Tensor<f32>, output_path: &PathBuf) -> Result<(), String> {
    let mut model = maskrcnn_resnet50_fpn::<f32>(91)
        .map_err(|e| format!("maskrcnn_resnet50_fpn: {e}"))?;
    let info = ferrotorch_hub::registry::get_model_info("maskrcnn_resnet50_fpn")
        .ok_or_else(|| {
            "ferrotorch_hub::registry: no entry for 'maskrcnn_resnet50_fpn'".to_string()
        })?;
    let cache = ferrotorch_hub::cache::HubCache::with_default_dir();
    let path = ferrotorch_hub::download::download_weights(info, &cache)
        .map_err(|e| format!("download_weights: {e}"))?;
    let state_dict = match info.format {
        ferrotorch_hub::registry::WeightsFormat::SafeTensors => {
            ferrotorch_serialize::load_safetensors::<f32>(&path)
                .map_err(|e| format!("load_safetensors: {e}"))?
        }
        ferrotorch_hub::registry::WeightsFormat::FerrotorchStateDict => {
            ferrotorch_serialize::load_state_dict::<f32>(&path)
                .map_err(|e| format!("load_state_dict: {e}"))?
        }
    };
    model
        .load_state_dict(&state_dict, false)
        .map_err(|e| format!("load_state_dict (apply): {e}"))?;
    apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state_dict)
        .map_err(|e| format!("apply_bn_buffers: {e}"))?;
    model.eval();
    eprintln!("[inference_dump] maskrcnn loaded; running MaskRcnn::forward...");

    let dets = MaskRcnn::forward(&model, input).map_err(|e| format!("forward: {e}"))?;
    let img_h = input.shape()[2];
    let img_w = input.shape()[3];

    // Pull first-image detections (we only ever pass batch=1).
    let (masks, boxes, scores) = if let Some(d) = dets.into_iter().next() {
        (d.masks, d.boxes, d.scores)
    } else {
        // Zero detections — emit empty tensors with correct shapes.
        let masks = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![]),
            vec![0, 1, img_h, img_w],
            false,
        )
        .map_err(|e| format!("empty masks: {e}"))?;
        let boxes = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![]),
            vec![0, 4],
            false,
        )
        .map_err(|e| format!("empty boxes: {e}"))?;
        let scores = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![]),
            vec![0],
            false,
        )
        .map_err(|e| format!("empty scores: {e}"))?;
        (masks, boxes, scores)
    };

    eprintln!(
        "[inference_dump] maskrcnn output: masks={:?}, boxes={:?}, scores={:?}",
        masks.shape(),
        boxes.shape(),
        scores.shape()
    );

    // Primary output: masks, matching the existing convention.
    dump_tensor(&masks, output_path).map_err(|e| format!("dump_tensor masks: {e}"))?;

    // Companion files: `<output>.boxes.bin` and `<output>.scores.bin`. These
    // carry the per-detection metadata needed by the harness to pair rust↔tv
    // detections by box-IoU (mAP-style matching) instead of by score rank
    // (which is structurally wrong — see #1141 round-9 diagnosis).
    let boxes_path = {
        let mut p = output_path.clone();
        let fname = format!(
            "{}.boxes.bin",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("out")
        );
        p.set_file_name(fname);
        p
    };
    let scores_path = {
        let mut p = output_path.clone();
        let fname = format!(
            "{}.scores.bin",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("out")
        );
        p.set_file_name(fname);
        p
    };
    dump_tensor(&boxes, &boxes_path).map_err(|e| format!("dump_tensor boxes: {e}"))?;
    dump_tensor(&scores, &scores_path).map_err(|e| format!("dump_tensor scores: {e}"))?;
    eprintln!(
        "[inference_dump] dumped masks={output_path:?}, boxes={boxes_path:?}, scores={scores_path:?}"
    );
    Ok(())
}

/// Build, weight-load, and run a FasterRcnn directly (not via the registry's
/// `Box<dyn Module>`) so we can extract the full `Detections` (boxes +
/// scores). The registry's `Module::forward` only returns 1-D post-NMS scores,
/// which is sufficient for the per-image score harness check but NOT for
/// COCO-mAP-style box-IoU pairing (#1145 harness criterion: per-rank score
/// pairing is structurally wrong for FasterRCNN-family detectors — f32 drift
/// flips NMS keep/drop decisions on score-borderline candidates, so rank-N
/// rust and rank-N tv can be different physical detections).
///
/// Replicates `models::registry::maybe_load_pretrained`'s weight-loading
/// exactly: hub lookup → safetensors load → `load_state_dict(strict=false)`
/// → BN buffer apply.
fn run_fasterrcnn_dump(input: &Tensor<f32>, output_path: &PathBuf) -> Result<(), String> {
    let mut model = fasterrcnn_resnet50_fpn::<f32>(91)
        .map_err(|e| format!("fasterrcnn_resnet50_fpn: {e}"))?;
    let info = ferrotorch_hub::registry::get_model_info("fasterrcnn_resnet50_fpn")
        .ok_or_else(|| {
            "ferrotorch_hub::registry: no entry for 'fasterrcnn_resnet50_fpn'".to_string()
        })?;
    let cache = ferrotorch_hub::cache::HubCache::with_default_dir();
    let path = ferrotorch_hub::download::download_weights(info, &cache)
        .map_err(|e| format!("download_weights: {e}"))?;
    let state_dict = match info.format {
        ferrotorch_hub::registry::WeightsFormat::SafeTensors => {
            ferrotorch_serialize::load_safetensors::<f32>(&path)
                .map_err(|e| format!("load_safetensors: {e}"))?
        }
        ferrotorch_hub::registry::WeightsFormat::FerrotorchStateDict => {
            ferrotorch_serialize::load_state_dict::<f32>(&path)
                .map_err(|e| format!("load_state_dict: {e}"))?
        }
    };
    model
        .load_state_dict(&state_dict, false)
        .map_err(|e| format!("load_state_dict (apply): {e}"))?;
    apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state_dict)
        .map_err(|e| format!("apply_bn_buffers: {e}"))?;
    model.eval();
    eprintln!("[inference_dump] fasterrcnn loaded; running FasterRcnn::forward...");

    let dets = FasterRcnn::forward(&model, input).map_err(|e| format!("forward: {e}"))?;

    // Pull first-image detections (we only ever pass batch=1).
    let (scores, boxes) = if let Some(d) = dets.into_iter().next() {
        (d.scores, d.boxes)
    } else {
        // Zero detections — emit empty tensors with correct shapes.
        let scores = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![]),
            vec![0usize],
            false,
        )
        .map_err(|e| format!("empty scores: {e}"))?;
        let boxes = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![]),
            vec![0, 4],
            false,
        )
        .map_err(|e| format!("empty boxes: {e}"))?;
        (scores, boxes)
    };

    eprintln!(
        "[inference_dump] fasterrcnn output: scores={:?}, boxes={:?}",
        scores.shape(),
        boxes.shape(),
    );

    // Primary output: scores (matches retinanet/fcos/keypointrcnn convention).
    dump_tensor(&scores, output_path).map_err(|e| format!("dump_tensor scores: {e}"))?;

    // Companion file: `<output>.boxes.bin`. Harness pairs rust↔tv by box-IoU
    // (COCO mAP-style) and compares scores on matched pairs.
    let boxes_path = {
        let mut p = output_path.clone();
        let fname = format!(
            "{}.boxes.bin",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("out")
        );
        p.set_file_name(fname);
        p
    };
    dump_tensor(&boxes, &boxes_path).map_err(|e| format!("dump_tensor boxes: {e}"))?;
    eprintln!("[inference_dump] dumped scores={output_path:?}, boxes={boxes_path:?}");
    Ok(())
}

/// Build, weight-load, and run a KeypointRcnn directly (not via the registry's
/// `Box<dyn Module>`) so we can extract the full `KeypointDetections`
/// (boxes + scores + keypoints + keypoint_scores). The registry's
/// `Module::forward` only returns 1-D post-NMS scores, which is sufficient
/// for the per-image score harness check but NOT for keypoint pixel-distance
/// validation (#1145 harness criterion `mean_kp_pixel_diff`).
///
/// Replicates `models::registry::maybe_load_pretrained`'s weight-loading
/// exactly: hub lookup → safetensors load → `load_state_dict(strict=false)`
/// → BN buffer apply.
fn run_keypointrcnn_dump(input: &Tensor<f32>, output_path: &PathBuf) -> Result<(), String> {
    let mut model = keypointrcnn_resnet50_fpn::<f32>()
        .map_err(|e| format!("keypointrcnn_resnet50_fpn: {e}"))?;
    let info = ferrotorch_hub::registry::get_model_info("keypointrcnn_resnet50_fpn")
        .ok_or_else(|| {
            "ferrotorch_hub::registry: no entry for 'keypointrcnn_resnet50_fpn'".to_string()
        })?;
    let cache = ferrotorch_hub::cache::HubCache::with_default_dir();
    let path = ferrotorch_hub::download::download_weights(info, &cache)
        .map_err(|e| format!("download_weights: {e}"))?;
    let state_dict = match info.format {
        ferrotorch_hub::registry::WeightsFormat::SafeTensors => {
            ferrotorch_serialize::load_safetensors::<f32>(&path)
                .map_err(|e| format!("load_safetensors: {e}"))?
        }
        ferrotorch_hub::registry::WeightsFormat::FerrotorchStateDict => {
            ferrotorch_serialize::load_state_dict::<f32>(&path)
                .map_err(|e| format!("load_state_dict: {e}"))?
        }
    };
    model
        .load_state_dict(&state_dict, false)
        .map_err(|e| format!("load_state_dict (apply): {e}"))?;
    apply_bn_buffers_from_state_dict(&model as &dyn Module<f32>, &state_dict)
        .map_err(|e| format!("apply_bn_buffers: {e}"))?;
    model.eval();
    eprintln!("[inference_dump] keypointrcnn loaded; running KeypointRcnn::forward...");

    let dets = KeypointRcnn::forward(&model, input).map_err(|e| format!("forward: {e}"))?;

    // Pull first-image detections (we only ever pass batch=1).
    let (scores, boxes, keypoints, keypoint_scores) = if let Some(d) = dets.into_iter().next() {
        (d.scores, d.boxes, d.keypoints, d.keypoint_scores)
    } else {
        // Zero detections — emit empty tensors with correct shapes.
        let scores = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![]),
            vec![0usize],
            false,
        )
        .map_err(|e| format!("empty scores: {e}"))?;
        let boxes = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![]),
            vec![0, 4],
            false,
        )
        .map_err(|e| format!("empty boxes: {e}"))?;
        let keypoints = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![]),
            vec![0, 17, 3],
            false,
        )
        .map_err(|e| format!("empty keypoints: {e}"))?;
        let kp_scores = Tensor::from_storage(
            ferrotorch_core::TensorStorage::cpu(vec![]),
            vec![0, 17],
            false,
        )
        .map_err(|e| format!("empty kp_scores: {e}"))?;
        (scores, boxes, keypoints, kp_scores)
    };

    eprintln!(
        "[inference_dump] keypointrcnn output: scores={:?}, boxes={:?}, keypoints={:?}, keypoint_scores={:?}",
        scores.shape(),
        boxes.shape(),
        keypoints.shape(),
        keypoint_scores.shape(),
    );

    // Primary output: scores (matches retinanet/fcos/fasterrcnn convention).
    dump_tensor(&scores, output_path).map_err(|e| format!("dump_tensor scores: {e}"))?;

    // Companion files: `<output>.boxes.bin`, `.keypoints.bin`, `.keypoint_scores.bin`.
    // The harness uses boxes for IoU-based pairing (rust↔tv) and then
    // computes per-keypoint pixel L2 distance on matched pairs.
    let mk_path = |suffix: &str| -> PathBuf {
        let mut p = output_path.clone();
        let fname = format!(
            "{}.{suffix}.bin",
            output_path.file_name().and_then(|n| n.to_str()).unwrap_or("out")
        );
        p.set_file_name(fname);
        p
    };
    let boxes_path = mk_path("boxes");
    let keypoints_path = mk_path("keypoints");
    let kp_scores_path = mk_path("keypoint_scores");
    dump_tensor(&boxes, &boxes_path).map_err(|e| format!("dump_tensor boxes: {e}"))?;
    dump_tensor(&keypoints, &keypoints_path).map_err(|e| format!("dump_tensor keypoints: {e}"))?;
    dump_tensor(&keypoint_scores, &kp_scores_path)
        .map_err(|e| format!("dump_tensor keypoint_scores: {e}"))?;
    eprintln!(
        "[inference_dump] dumped scores={output_path:?}, boxes={boxes_path:?}, keypoints={keypoints_path:?}, keypoint_scores={kp_scores_path:?}"
    );
    Ok(())
}

fn main() -> Result<(), String> {
    let (model_name, image_path, output_path) = parse_args()?;
    let num_classes = num_classes_for(&model_name)?;

    eprintln!("[inference_dump] model={model_name} image={image_path:?} num_classes={num_classes}");

    // Load raw image as [C, H, W] tensor in [0, 1].
    let raw = read_image_as_tensor::<f32>(&image_path)
        .map_err(|e| format!("read_image_as_tensor: {e}"))?;
    eprintln!("[inference_dump] raw image shape: {:?}", raw.shape());

    // Preprocess according to torchvision recipe for this model.
    let input =
        preprocess_for_model(&model_name, raw).map_err(|e| format!("preprocess: {e}"))?;
    eprintln!("[inference_dump] preprocessed shape: {:?}", input.shape());

    // MaskRCNN takes a custom path so we can dump boxes + scores alongside
    // the masks (the Module::forward registry path only exposes the masks).
    // This is harness instrumentation only — no model code is changed.
    if model_name == "maskrcnn_resnet50_fpn" {
        return run_maskrcnn_dump(&input, &output_path);
    }

    // FasterRCNN needs a companion `.boxes.bin` so the #1145 harness can pair
    // rust↔tv detections by box-IoU (COCO mAP-style) — per-rank score pairing
    // is structurally wrong for FasterRCNN-family detectors (f32 drift flips
    // NMS keep/drop on score-borderline candidates). Module::forward alone
    // exposes only the 1-D scores.
    if model_name == "fasterrcnn_resnet50_fpn" {
        return run_fasterrcnn_dump(&input, &output_path);
    }

    // KeypointRCNN likewise needs companion boxes + keypoints + keypoint_scores
    // dumps so the #1139 harness can pair rust↔tv detections by box-IoU
    // (mAP-style) and compute per-keypoint pixel L2 distance — Module::forward
    // alone only exposes the 1-D scores.
    if model_name == "keypointrcnn_resnet50_fpn" {
        return run_keypointrcnn_dump(&input, &output_path);
    }

    // Build model via the architect-mandated registry path; this loads
    // pretrained weights from the local hub cache (pinned in #1130).
    let mut model =
        get_model(&model_name, true, num_classes).map_err(|e| format!("get_model: {e}"))?;
    model.eval();
    eprintln!("[inference_dump] model loaded; running forward...");

    let output = model
        .forward(&input)
        .map_err(|e| format!("forward: {e}"))?;
    eprintln!("[inference_dump] output shape: {:?}", output.shape());

    dump_tensor(&output, &output_path).map_err(|e| format!("dump_tensor: {e}"))?;
    eprintln!("[inference_dump] dumped to {output_path:?}");

    Ok(())
}
