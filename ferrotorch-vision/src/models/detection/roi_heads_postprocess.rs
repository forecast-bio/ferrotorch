//! RoIHeads postprocess pipeline shared by Faster R-CNN and Mask R-CNN.
//!
//! Mirrors `torchvision.models.detection.roi_heads.RoIHeads.postprocess_detections`
//! and `torchvision.models.detection.roi_heads.maskrcnn_inference` +
//! `paste_masks_in_image`.
//!
//! ## Pipeline (Faster R-CNN, per image)
//!
//! 1. Softmax over class logits → `[N_prop, num_classes]`.
//! 2. Decode `[N_prop, num_classes * 4]` box deltas against the proposals into
//!    `[N_prop, num_classes, 4]` per-class predicted boxes.
//! 3. Clip boxes to image bounds.
//! 4. Drop the background class (index 0).
//! 5. Flatten to `[N_prop * (num_classes - 1), {boxes, scores, labels}]`.
//! 6. Filter by score threshold (`0.05`).
//! 7. Drop boxes smaller than `1e-2` on either side.
//! 8. Per-class NMS (`iou=0.5`) via `batched_nms`.
//! 9. Cross-class top-K = `detections_per_img` (`100`).
//!
//! ## Pipeline (Mask R-CNN extension, per image)
//!
//! 1. Run mask head + predictor on detected-box ROI features → `[N_det, num_classes, 28, 28]` logits.
//! 2. Sigmoid.
//! 3. Select the mask channel matching each detection's predicted label.
//! 4. Bilinear paste each 28×28 mask back into a `[H_img, W_img]` canvas, cropped by the box.

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::{InterpolateMode, interpolate};

use crate::ops::{batched_nms, clip_boxes_to_image};

// ---------------------------------------------------------------------------
// Torchvision RoIHeads constants (verified against installed source).
// ---------------------------------------------------------------------------

/// Per-class score threshold below which detections are dropped.
/// Matches `torchvision.models.detection.faster_rcnn.FasterRCNN(score_thresh=0.05)`.
pub const ROI_SCORE_THRESH: f64 = 0.05;

/// IoU threshold for per-class non-max suppression.
/// Matches `torchvision.models.detection.faster_rcnn.FasterRCNN(nms_thresh=0.5)`.
pub const ROI_NMS_THRESH: f64 = 0.5;

/// Cross-class top-K limit on detections per image.
/// Matches `torchvision.models.detection.faster_rcnn.FasterRCNN(detections_per_img=100)`.
pub const ROI_DETECTIONS_PER_IMG: usize = 100;

/// Minimum side length for a box to be kept (post-decode, pre-NMS).
/// Matches `box_ops.remove_small_boxes(boxes, min_size=1e-2)` in
/// `RoIHeads.postprocess_detections`.
pub const ROI_MIN_BOX_SIDE: f64 = 1e-2;

/// Box-coder regression-weights for the FasterRCNN detection head.
///
/// Matches `torchvision.models.detection.faster_rcnn.FasterRCNN.box_coder` —
/// `BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))`.
pub const ROI_BOX_CODER_WEIGHTS: (f64, f64, f64, f64) = (10.0, 10.0, 5.0, 5.0);

/// One-sided clip applied to `dw`/`dh` before `exp` to prevent overflow.
/// Matches `torchvision.models.detection._utils.BoxCoder.bbox_xform_clip =
/// math.log(1000 / 16)`. Note this is a `max=` clamp, **not** symmetric.
pub const ROI_BBOX_XFORM_CLIP: f64 = 4.135_166_556_742_356; // log(1000.0 / 16.0)

// ---------------------------------------------------------------------------
// Per-class BoxCoder decode (RoIHeads variant)
// ---------------------------------------------------------------------------

/// Decode per-class regression deltas against a single set of proposals,
/// matching `torchvision.models.detection._utils.BoxCoder.decode_single`.
///
/// Inputs:
/// - `proposals`: `[N, 4]` in xyxy format.
/// - `deltas`: `[N, num_classes * 4]` flattened per-class regression targets,
///   layout `[dx, dy, dw, dh] × num_classes`.
/// - `weights`: `(wx, wy, ww, wh)` regression-weights — `(10, 10, 5, 5)` for
///   FasterRCNN's detection head.
/// - `bbox_xform_clip`: one-sided max clamp on `dw`/`dh` before `exp`
///   (`log(1000/16)` for torchvision FasterRCNN).
///
/// Returns `[N, num_classes, 4]` predicted boxes in xyxy format.
pub fn decode_per_class<T: Float>(
    proposals: &Tensor<T>,
    deltas: &Tensor<T>,
    weights: (f64, f64, f64, f64),
    bbox_xform_clip: f64,
) -> FerrotorchResult<Tensor<T>> {
    let n = proposals.shape()[0];
    if proposals.shape() != [n, 4] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "decode_per_class: proposals must be [N, 4], got {:?}",
                proposals.shape()
            ),
        });
    }
    let deltas_shape = deltas.shape();
    if deltas_shape.len() != 2 || deltas_shape[0] != n || deltas_shape[1] % 4 != 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "decode_per_class: deltas must be [N, num_classes * 4] with N={n}, got {:?}",
                deltas_shape
            ),
        });
    }
    let num_classes = deltas_shape[1] / 4;

    let prop = proposals.data_vec()?;
    let del = deltas.data_vec()?;

    let wx = weights.0;
    let wy = weights.1;
    let ww = weights.2;
    let wh = weights.3;
    let clip = bbox_xform_clip;

    let mut out = vec![cast::<f64, T>(0.0)?; n * num_classes * 4];

    for i in 0..n {
        let x1 = prop[i * 4].to_f64().unwrap_or(0.0);
        let y1 = prop[i * 4 + 1].to_f64().unwrap_or(0.0);
        let x2 = prop[i * 4 + 2].to_f64().unwrap_or(0.0);
        let y2 = prop[i * 4 + 3].to_f64().unwrap_or(0.0);

        let widths = x2 - x1;
        let heights = y2 - y1;
        let ctr_x = x1 + 0.5 * widths;
        let ctr_y = y1 + 0.5 * heights;

        for c in 0..num_classes {
            let row_base = i * num_classes * 4 + c * 4;
            let d_base = i * (num_classes * 4) + c * 4;
            let dx = del[d_base].to_f64().unwrap_or(0.0) / wx;
            let dy = del[d_base + 1].to_f64().unwrap_or(0.0) / wy;
            let mut dw = del[d_base + 2].to_f64().unwrap_or(0.0) / ww;
            let mut dh = del[d_base + 3].to_f64().unwrap_or(0.0) / wh;

            // torchvision: `torch.clamp(dw, max=bbox_xform_clip)` — one-sided.
            if dw > clip {
                dw = clip;
            }
            if dh > clip {
                dh = clip;
            }

            let pred_ctr_x = dx * widths + ctr_x;
            let pred_ctr_y = dy * heights + ctr_y;
            let pred_w = dw.exp() * widths;
            let pred_h = dh.exp() * heights;

            // Distance from centre to corner — torchvision uses 0.5*pred_h
            // (matching the formula exactly).
            out[row_base] = cast::<f64, T>(pred_ctr_x - 0.5 * pred_w)?;
            out[row_base + 1] = cast::<f64, T>(pred_ctr_y - 0.5 * pred_h)?;
            out[row_base + 2] = cast::<f64, T>(pred_ctr_x + 0.5 * pred_w)?;
            out[row_base + 3] = cast::<f64, T>(pred_ctr_y + 0.5 * pred_h)?;
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![n, num_classes, 4], false)
}

// ---------------------------------------------------------------------------
// FasterRCNN postprocess_detections
// ---------------------------------------------------------------------------

/// Result of [`postprocess_detections`] for a single image.
pub struct PostprocessedDetections<T: Float> {
    /// `[N_det, 4]` final boxes in xyxy pixel coords.
    pub boxes: Tensor<T>,
    /// `[N_det]` final scores (post-NMS, top-K limited).
    pub scores: Tensor<T>,
    /// `[N_det]` predicted class labels (1..num_classes; never background).
    pub labels: Vec<usize>,
}

/// Apply the torchvision `RoIHeads.postprocess_detections` pipeline to a single image.
///
/// Inputs:
/// - `class_logits`: `[N_prop, num_classes]` raw classifier output.
/// - `box_deltas`: `[N_prop, num_classes * 4]` raw regressor output.
/// - `proposals`: `[N_prop, 4]` RPN proposals (xyxy, image coords).
/// - `image_size`: `[H, W]` of the image fed into the model (used for box clip).
pub fn postprocess_detections<T: Float>(
    class_logits: &Tensor<T>,
    box_deltas: &Tensor<T>,
    proposals: &Tensor<T>,
    image_size: [usize; 2],
) -> FerrotorchResult<PostprocessedDetections<T>> {
    let shape = class_logits.shape();
    if shape.len() != 2 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "postprocess_detections: class_logits must be [N, C], got {:?}",
                shape
            ),
        });
    }
    let n = shape[0];
    let num_classes = shape[1];

    // 1) Softmax over class axis.
    let scores_full = softmax_2d::<T>(class_logits)?;
    let scores_data = scores_full.data_vec()?;

    // 2) Decode per-class boxes → [N, num_classes, 4].
    let decoded = decode_per_class::<T>(
        proposals,
        box_deltas,
        ROI_BOX_CODER_WEIGHTS,
        ROI_BBOX_XFORM_CLIP,
    )?;
    let decoded_data = decoded.data_vec()?;

    // 3) Clip boxes to image. Flatten to [N * num_classes, 4] for batch-clip,
    //    then proceed.
    let flat_decoded_t = Tensor::from_storage(
        TensorStorage::cpu(decoded_data.clone()),
        vec![n * num_classes, 4],
        false,
    )?;
    let clipped_t = clip_boxes_to_image::<T>(&flat_decoded_t, image_size)?;
    let clipped_data = clipped_t.data_vec()?;

    // 4–5) Drop background (class 0) and flatten remaining classes.
    //
    //   Effective N = n * (num_classes - 1)
    //   Boxes      : row (i, c) → clipped_data[(i * num_classes + c) * 4..]
    //   Scores     : scores_data[i * num_classes + c]
    //   Labels     : c  (already 1..num_classes)
    //
    // 6) Filter by score threshold.
    // 7) Remove small boxes.
    let mut cand_boxes: Vec<T> = Vec::new();
    let mut cand_scores: Vec<T> = Vec::new();
    let mut cand_labels: Vec<usize> = Vec::new();

    let score_thresh: T = cast::<f64, T>(ROI_SCORE_THRESH)?;
    let min_side: T = cast::<f64, T>(ROI_MIN_BOX_SIDE)?;

    for i in 0..n {
        for c in 1..num_classes {
            let s = scores_data[i * num_classes + c];
            // torchvision: `torch.where(scores > score_thresh)`. A NaN score
            // fails this comparison and is dropped (matching PyTorch's NaN
            // semantics for `>`).
            if s.partial_cmp(&score_thresh) != Some(std::cmp::Ordering::Greater) {
                continue;
            }
            let base = (i * num_classes + c) * 4;
            let x1 = clipped_data[base];
            let y1 = clipped_data[base + 1];
            let x2 = clipped_data[base + 2];
            let y2 = clipped_data[base + 3];
            let w = x2 - x1;
            let h = y2 - y1;
            let w_ok = w.partial_cmp(&min_side).is_some_and(|o| o != std::cmp::Ordering::Less);
            let h_ok = h.partial_cmp(&min_side).is_some_and(|o| o != std::cmp::Ordering::Less);
            if !(w_ok && h_ok) {
                continue;
            }
            cand_boxes.push(x1);
            cand_boxes.push(y1);
            cand_boxes.push(x2);
            cand_boxes.push(y2);
            cand_scores.push(s);
            cand_labels.push(c);
        }
    }

    let n_cand = cand_scores.len();
    if n_cand == 0 {
        return Ok(PostprocessedDetections {
            boxes: Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0, 4], false)?,
            scores: Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0usize], false)?,
            labels: vec![],
        });
    }

    // 8) Per-class NMS via batched_nms.
    let cand_boxes_t =
        Tensor::from_storage(TensorStorage::cpu(cand_boxes.clone()), vec![n_cand, 4], false)?;
    let cand_scores_t = Tensor::from_storage(
        TensorStorage::cpu(cand_scores.clone()),
        vec![n_cand],
        false,
    )?;
    let idxs: Vec<u32> = cand_labels.iter().map(|&l| l as u32).collect();
    let keep = batched_nms::<T>(&cand_boxes_t, &cand_scores_t, &idxs, ROI_NMS_THRESH)?;

    // 9) Cross-class top-K: batched_nms returns indices sorted by descending
    //    score, so the slice `keep[..K]` matches torchvision's
    //    `keep = keep[: self.detections_per_img]` exactly.
    let mut kept = keep;
    if kept.len() > ROI_DETECTIONS_PER_IMG {
        kept.truncate(ROI_DETECTIONS_PER_IMG);
    }

    let n_det = kept.len();
    let mut out_boxes: Vec<T> = Vec::with_capacity(n_det * 4);
    let mut out_scores: Vec<T> = Vec::with_capacity(n_det);
    let mut out_labels: Vec<usize> = Vec::with_capacity(n_det);
    for &k in &kept {
        out_boxes.push(cand_boxes[k * 4]);
        out_boxes.push(cand_boxes[k * 4 + 1]);
        out_boxes.push(cand_boxes[k * 4 + 2]);
        out_boxes.push(cand_boxes[k * 4 + 3]);
        out_scores.push(cand_scores[k]);
        out_labels.push(cand_labels[k]);
    }

    Ok(PostprocessedDetections {
        boxes: Tensor::from_storage(TensorStorage::cpu(out_boxes), vec![n_det, 4], false)?,
        scores: Tensor::from_storage(TensorStorage::cpu(out_scores), vec![n_det], false)?,
        labels: out_labels,
    })
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Row-wise numerically-stable softmax for a `[N, C]` tensor.
///
/// Returns `[N, C]` with each row summing to 1.0. Lifted from
/// `faster_rcnn.rs`'s private helper so the postprocess module is self-contained.
fn softmax_2d<T: Float>(logits: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let n = logits.shape()[0];
    let c = logits.shape()[1];
    let data = logits.data_vec()?;
    let mut out = vec![cast::<f64, T>(0.0)?; n * c];
    for i in 0..n {
        let row = &data[i * c..(i + 1) * c];
        let max_val = row
            .iter()
            .copied()
            .fold(row[0], |acc, x| if x > acc { x } else { acc });
        let max_f = max_val.to_f64().unwrap_or(0.0);
        let exps: Vec<f64> = row
            .iter()
            .map(|&v| (v.to_f64().unwrap_or(0.0) - max_f).exp())
            .collect();
        let sum: f64 = exps.iter().sum();
        for j in 0..c {
            out[i * c + j] = cast::<f64, T>(exps[j] / sum)?;
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n, c], false)
}

// ---------------------------------------------------------------------------
// Mask postprocess: sigmoid → class-select → paste back
// ---------------------------------------------------------------------------

/// Apply `torchvision.models.detection.roi_heads.maskrcnn_inference` followed by
/// `paste_masks_in_image` to a single image's mask logits.
///
/// Inputs:
/// - `mask_logits`: `[N_det, num_classes, mask_h, mask_w]` raw mask logits.
/// - `labels`: predicted class id per detection (length `N_det`).
/// - `boxes`: `[N_det, 4]` detection boxes in xyxy pixel coords (image space).
/// - `image_size`: `[H, W]` of the image fed into the model. Used **only** when
///   `paste=true`.
/// - `paste`: when `true`, runs the full torchvision sequence
///   (sigmoid → class-select → `paste_masks_in_image`) and returns
///   `[N_det, 1, H, W]`. When `false`, stops after sigmoid + class-select and
///   returns `[N_det, 1, mask_h, mask_w]` — matching torchvision's
///   `RoIHeads.forward` output **before** the outer `GeneralizedRCNNTransform.
///   postprocess` paste step (which the #1139 verification harness patches to
///   identity so the dump comparison is against the pre-paste tensor).
pub fn postprocess_masks<T: Float>(
    mask_logits: &Tensor<T>,
    labels: &[usize],
    boxes: &Tensor<T>,
    image_size: [usize; 2],
    paste: bool,
) -> FerrotorchResult<Tensor<T>> {
    let shape = mask_logits.shape().to_vec();
    if shape.len() != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "postprocess_masks: mask_logits must be [N, C, h, w], got {:?}",
                shape
            ),
        });
    }
    let n_det = shape[0];
    let num_classes = shape[1];
    let mh = shape[2];
    let mw = shape[3];
    let im_h = image_size[0];
    let im_w = image_size[1];

    if n_det == 0 {
        return Tensor::from_storage(
            TensorStorage::cpu(vec![]),
            vec![0, 1, im_h, im_w],
            false,
        );
    }
    if labels.len() != n_det {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "postprocess_masks: labels length {} != N_det {}",
                labels.len(),
                n_det
            ),
        });
    }
    if boxes.shape() != [n_det, 4] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "postprocess_masks: boxes must be [N_det={}, 4], got {:?}",
                n_det,
                boxes.shape()
            ),
        });
    }

    // 1) sigmoid (device-aware via ferrotorch-core).
    let mask_prob = mask_logits.sigmoid()?;
    let mp_data = mask_prob.data_vec()?;

    // 2) Class-select: pick the channel matching the predicted label per detection.
    //    → `selected[N_det, mh, mw]`.
    let per_det = num_classes * mh * mw;
    let plane = mh * mw;
    let mut selected: Vec<T> = Vec::with_capacity(n_det * plane);
    for (i, &lbl) in labels.iter().enumerate() {
        let cls = lbl.min(num_classes - 1);
        let start = i * per_det + cls * plane;
        selected.extend_from_slice(&mp_data[start..start + plane]);
    }

    if !paste {
        // Return the sigmoid + class-select tensor `[N_det, 1, mh, mw]` — the
        // `maskrcnn_inference` output torchvision exposes from `RoIHeads.forward`
        // before `GeneralizedRCNNTransform.postprocess` runs `paste_masks_in_image`.
        return Tensor::from_storage(
            TensorStorage::cpu(selected),
            vec![n_det, 1, mh, mw],
            false,
        );
    }

    // 3) torchvision `expand_masks`: pad masks by 1 px (so paste samples from a
    //    slightly larger mask). scale = (M + 2*pad) / M.
    let padding: usize = 1;
    let mh_pad = mh + 2 * padding;
    let mw_pad = mw + 2 * padding;
    let mut padded: Vec<T> = vec![cast::<f64, T>(0.0)?; n_det * mh_pad * mw_pad];
    let plane_pad = mh_pad * mw_pad;
    let zero_t: T = cast::<f64, T>(0.0)?;
    let _ = zero_t;
    for i in 0..n_det {
        let src_base = i * plane;
        let dst_base = i * plane_pad;
        for r in 0..mh {
            let src_row = src_base + r * mw;
            let dst_row = dst_base + (r + padding) * mw_pad + padding;
            padded[dst_row..dst_row + mw].copy_from_slice(&selected[src_row..src_row + mw]);
        }
    }
    // Scale used by `expand_boxes` (`scale = (M + 2*padding) / M`).
    let scale = (mh + 2 * padding) as f64 / mh as f64;

    let boxes_data = boxes.data_vec()?;

    // 4) Paste each per-detection mask back into the [H_img, W_img] canvas.
    //
    // Reproduces torchvision's `paste_mask_in_image` exactly, including:
    //   - `expand_boxes` (scale around centre by `scale`).
    //   - `.to(dtype=torch.int64)` truncation-toward-zero of the expanded box.
    //   - `w = int(box[2] - box[0] + 1)`, `h = int(box[3] - box[1] + 1)`.
    //   - Bilinear resize of the padded mask to `[h, w]`.
    //   - Crop the resized mask by `(y_0 - box[1])..(y_1 - box[1])` × same in x
    //     where `(x_0, y_0, x_1, y_1) = (max(box[0], 0), max(box[1], 0),
    //                                    min(box[2] + 1, im_w), min(box[3] + 1, im_h))`.
    let mut out: Vec<T> = vec![cast::<f64, T>(0.0)?; n_det * im_h * im_w];

    for i in 0..n_det {
        // expand_boxes: rescale around centre.
        let bx1 = boxes_data[i * 4].to_f64().unwrap_or(0.0);
        let by1 = boxes_data[i * 4 + 1].to_f64().unwrap_or(0.0);
        let bx2 = boxes_data[i * 4 + 2].to_f64().unwrap_or(0.0);
        let by2 = boxes_data[i * 4 + 3].to_f64().unwrap_or(0.0);

        let w_half = (bx2 - bx1) * 0.5 * scale;
        let h_half = (by2 - by1) * 0.5 * scale;
        let xc = (bx2 + bx1) * 0.5;
        let yc = (by2 + by1) * 0.5;

        // torchvision: `boxes.to(dtype=torch.int64)` truncates toward zero.
        let exp_x1 = (xc - w_half).trunc() as i64;
        let exp_y1 = (yc - h_half).trunc() as i64;
        let exp_x2 = (xc + w_half).trunc() as i64;
        let exp_y2 = (yc + h_half).trunc() as i64;

        // torchvision: `w = int(box[2] - box[0] + TO_REMOVE)` with TO_REMOVE = 1.
        let mut paste_w = exp_x2 - exp_x1 + 1;
        let mut paste_h = exp_y2 - exp_y1 + 1;
        if paste_w < 1 {
            paste_w = 1;
        }
        if paste_h < 1 {
            paste_h = 1;
        }
        let paste_w = paste_w as usize;
        let paste_h = paste_h as usize;

        // Build the padded mask as [1, 1, mh_pad, mw_pad] tensor for interpolate.
        let pad_slice =
            &padded[i * plane_pad..(i + 1) * plane_pad];
        let m_tensor = Tensor::from_storage(
            TensorStorage::cpu(pad_slice.to_vec()),
            vec![1, 1, mh_pad, mw_pad],
            false,
        )?;
        let resized = interpolate::<T>(
            &m_tensor,
            Some([paste_h, paste_w]),
            None,
            InterpolateMode::Bilinear,
            false,
        )?;
        let resized_data = resized.data_vec()?;

        // Paste into out[i] at [x_0:x_1, y_0:y_1] with crop offset (-exp_x1, -exp_y1).
        let im_h_i64 = im_h as i64;
        let im_w_i64 = im_w as i64;
        let x_0 = exp_x1.max(0);
        let y_0 = exp_y1.max(0);
        let x_1 = (exp_x2 + 1).min(im_w_i64);
        let y_1 = (exp_y2 + 1).min(im_h_i64);

        if x_1 <= x_0 || y_1 <= y_0 {
            continue;
        }

        let out_base = i * im_h * im_w;
        for yy in y_0..y_1 {
            // Source row in the resized mask is `yy - exp_y1`.
            let src_row = (yy - exp_y1) as usize;
            if src_row >= paste_h {
                continue;
            }
            let src_base = src_row * paste_w;
            let dst_base = out_base + (yy as usize) * im_w;
            for xx in x_0..x_1 {
                let src_col = (xx - exp_x1) as usize;
                if src_col >= paste_w {
                    continue;
                }
                out[dst_base + xx as usize] = resized_data[src_base + src_col];
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![n_det, 1, im_h, im_w], false)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::from_slice;

    /// BoxCoder roundtrip: zero deltas yield the original proposals (per class).
    #[test]
    fn decode_per_class_identity_zero_deltas() {
        // Two proposals, three classes, zero deltas → every class-row = proposal.
        let proposals = from_slice::<f32>(
            &[10.0, 20.0, 50.0, 60.0, 0.0, 0.0, 100.0, 200.0],
            &[2, 4],
        )
        .unwrap();
        let deltas = from_slice::<f32>(&[0.0_f32; 24], &[2, 12]).unwrap();
        let decoded =
            decode_per_class(&proposals, &deltas, ROI_BOX_CODER_WEIGHTS, ROI_BBOX_XFORM_CLIP)
                .unwrap();
        assert_eq!(decoded.shape(), &[2, 3, 4]);
        let d = decoded.data_vec().unwrap();
        // Row 0, class 0: should equal proposal 0 = [10, 20, 50, 60].
        assert!((d[0] - 10.0).abs() < 1e-4, "x1={}", d[0]);
        assert!((d[1] - 20.0).abs() < 1e-4, "y1={}", d[1]);
        assert!((d[2] - 50.0).abs() < 1e-4, "x2={}", d[2]);
        assert!((d[3] - 60.0).abs() < 1e-4, "y2={}", d[3]);
        // Row 1, class 2: should equal proposal 1 = [0, 0, 100, 200].
        let base = (3 + 2) * 4;
        assert!((d[base] - 0.0).abs() < 1e-4);
        assert!((d[base + 1] - 0.0).abs() < 1e-4);
        assert!((d[base + 2] - 100.0).abs() < 1e-4);
        assert!((d[base + 3] - 200.0).abs() < 1e-4);
    }

    /// BoxCoder uses (10, 10, 5, 5) weights: a delta of `wx` in `dx` shifts the
    /// box centre by exactly one width.
    #[test]
    fn decode_per_class_weights_match_torchvision() {
        // One proposal: width = 10, height = 20, centre = (5, 10).
        let proposals = from_slice::<f32>(&[0.0, 0.0, 10.0, 20.0], &[1, 4]).unwrap();
        // One class. Set dx = wx = 10 → divided by wx = 1 → shift centre by 1 width.
        let deltas = from_slice::<f32>(&[10.0, 0.0, 0.0, 0.0], &[1, 4]).unwrap();
        let decoded =
            decode_per_class(&proposals, &deltas, ROI_BOX_CODER_WEIGHTS, ROI_BBOX_XFORM_CLIP)
                .unwrap();
        let d = decoded.data_vec().unwrap();
        // Centre shifts from x=5 to x=15. Width/height unchanged (10, 20).
        // New box centred at (15, 10) → [10, 0, 20, 20].
        assert!((d[0] - 10.0).abs() < 1e-3, "x1={}", d[0]);
        assert!((d[1] - 0.0).abs() < 1e-3, "y1={}", d[1]);
        assert!((d[2] - 20.0).abs() < 1e-3, "x2={}", d[2]);
        assert!((d[3] - 20.0).abs() < 1e-3, "y2={}", d[3]);
    }

    /// `dw` is clamped one-sidedly to `bbox_xform_clip` (matches torchvision).
    /// A very-large positive `dw` saturates; a very-large negative `dw` does NOT.
    #[test]
    fn decode_per_class_dw_one_sided_clamp() {
        let proposals = from_slice::<f32>(&[0.0, 0.0, 10.0, 10.0], &[1, 4]).unwrap();
        // dw / ww = 100 → before clip, would explode via exp(100).
        // After clip to log(1000/16) ≈ 4.135 → pred_w = exp(4.135) * 10 ≈ 625.
        let big_pos = from_slice::<f32>(&[0.0, 0.0, 500.0, 0.0], &[1, 4]).unwrap();
        let dec_pos =
            decode_per_class(&proposals, &big_pos, ROI_BOX_CODER_WEIGHTS, ROI_BBOX_XFORM_CLIP)
                .unwrap();
        let p = dec_pos.data_vec().unwrap();
        let w = p[2] - p[0];
        assert!(w > 500.0 && w < 700.0, "clamped width = {w}, expected ~625");

        // Large negative dw must NOT be clamped → exp(-100) * 10 ≈ 0.
        let big_neg = from_slice::<f32>(&[0.0, 0.0, -500.0, 0.0], &[1, 4]).unwrap();
        let dec_neg =
            decode_per_class(&proposals, &big_neg, ROI_BOX_CODER_WEIGHTS, ROI_BBOX_XFORM_CLIP)
                .unwrap();
        let n = dec_neg.data_vec().unwrap();
        let wn = n[2] - n[0];
        assert!(wn < 1e-3, "neg dw should produce tiny width, got {wn}");
    }

    /// Postprocess: two overlapping proposals, both predict class 1 with high
    /// score → NMS keeps the higher-scoring one and drops the other.
    #[test]
    fn postprocess_nms_drops_overlap() {
        // num_classes = 2 (background + foreground). N_prop = 2 overlapping.
        // Class logits: row 0 strongly favours class 1, row 1 also class 1.
        let logits = from_slice::<f32>(
            // [bg, fg] logits per proposal.
            &[-2.0, 4.0, -2.0, 3.0],
            &[2, 2],
        )
        .unwrap();
        // Zero deltas → predicted boxes = proposals.
        let deltas = from_slice::<f32>(&[0.0_f32; 16], &[2, 8]).unwrap();
        // Two heavily overlapping proposals.
        let proposals = from_slice::<f32>(
            &[10.0, 10.0, 50.0, 50.0, 12.0, 11.0, 51.0, 49.0],
            &[2, 4],
        )
        .unwrap();
        let det =
            postprocess_detections::<f32>(&logits, &deltas, &proposals, [100, 100]).unwrap();
        assert_eq!(det.boxes.shape()[0], 1, "NMS should keep only one box");
        assert_eq!(det.scores.shape(), &[1]);
        assert_eq!(det.labels.len(), 1);
        assert_eq!(det.labels[0], 1, "kept the foreground class");
    }

    /// End-to-end discriminating test: a strong-signal toy input produces the
    /// expected post-NMS detections, including correct top-K truncation.
    ///
    /// This guards against the trap where the postprocess silently drops all
    /// detections even with high-confidence inputs (e.g. due to a unit-mismatch
    /// in `score_thresh` or a flipped class-axis in the softmax).
    #[test]
    fn postprocess_strong_signal_survives_full_pipeline() {
        // 3 proposals (well-separated), num_classes = 4 (bg + 3 fg).
        // Each proposal strongly favours a distinct foreground class.
        let logits = from_slice::<f32>(
            &[
                -10.0, 10.0, -10.0, -10.0, // prop 0 → class 1
                -10.0, -10.0, 10.0, -10.0, // prop 1 → class 2
                -10.0, -10.0, -10.0, 10.0, // prop 2 → class 3
            ],
            &[3, 4],
        )
        .unwrap();
        // Zero deltas → predicted box equals proposal.
        let deltas = from_slice::<f32>(&[0.0_f32; 48], &[3, 16]).unwrap();
        let proposals = from_slice::<f32>(
            &[
                10.0, 10.0, 50.0, 50.0, // well-separated boxes
                100.0, 100.0, 140.0, 140.0, 200.0, 200.0, 240.0, 240.0,
            ],
            &[3, 4],
        )
        .unwrap();

        let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [500, 500])
            .unwrap();
        assert_eq!(det.boxes.shape(), &[3, 4]);
        assert_eq!(det.scores.shape(), &[3]);
        assert_eq!(det.labels.len(), 3);
        // Each proposal yields exactly one detection at its assigned class.
        let mut labels_sorted = det.labels.clone();
        labels_sorted.sort_unstable();
        assert_eq!(labels_sorted, vec![1, 2, 3]);
        // All scores ≈ 1.0 (softmax of [+10, -10, -10, -10]).
        let s = det.scores.data_vec().unwrap();
        for &v in &s {
            assert!(v > 0.999, "expected high score, got {v}");
        }
    }

    /// Postprocess drops detections below score_thresh.
    #[test]
    fn postprocess_score_thresh_drops_low_confidence() {
        // num_classes = 2. Logits give softmax(fg) ≈ 0.01 < 0.05.
        let logits = from_slice::<f32>(&[5.0, 0.4], &[1, 2]).unwrap();
        let deltas = from_slice::<f32>(&[0.0_f32; 8], &[1, 8]).unwrap();
        let proposals = from_slice::<f32>(&[0.0, 0.0, 10.0, 10.0], &[1, 4]).unwrap();
        let det = postprocess_detections::<f32>(&logits, &deltas, &proposals, [50, 50]).unwrap();
        assert_eq!(det.boxes.shape()[0], 0);
    }

    /// Mask paste-back has correct output shape and a non-trivial pasted region.
    #[test]
    fn mask_paste_shape_and_extent() {
        // 1 detection, num_classes = 2, mask 28×28 all ones for class 1.
        let mut logits = vec![-5.0_f32; 2 * 28 * 28]; // sigmoid → ~0 for class 0
        for i in 0..28 * 28 {
            logits[28 * 28 + i] = 5.0; // class 1 → sigmoid ~1
        }
        let mask_logits = from_slice::<f32>(&logits, &[1, 2, 28, 28]).unwrap();
        let labels = vec![1usize];
        let boxes = from_slice::<f32>(&[10.0, 20.0, 50.0, 80.0], &[1, 4]).unwrap();
        let im = [100usize, 120];
        let pasted = postprocess_masks::<f32>(&mask_logits, &labels, &boxes, im, true).unwrap();
        assert_eq!(pasted.shape(), &[1, 1, 100, 120]);
        let d = pasted.data_vec().unwrap();
        // The pasted region should contain mostly-1 values inside the box.
        // Sample inside the box (centre).
        let centre_y = 50usize;
        let centre_x = 30usize;
        let v = d[centre_y * 120 + centre_x];
        assert!(v > 0.9, "inside-box pasted value should be ~1.0, got {v}");
        // Outside the box (top-left corner) should be 0.
        let outside = d[0];
        assert_eq!(outside, 0.0);
    }

    /// `paste=false` short-circuits after sigmoid + class-select.
    ///
    /// Output is `[N_det, 1, mask_h, mask_w]` and contains only the picked
    /// class channel for each detection.
    #[test]
    fn mask_no_paste_returns_pre_paste_tensor() {
        // 2 detections, num_classes = 3 (bg + 2 fg). Class 1 has logit 5
        // everywhere, class 2 has logit -5 everywhere — sigmoid → ~1.0 vs ~0.0.
        let mut logits = vec![0.0_f32; 2 * 3 * 4 * 4];
        let plane = 4 * 4;
        for det in 0..2 {
            for i in 0..plane {
                // class 0 (bg): zero (already initialised).
                logits[det * 3 * plane + plane + i] = 5.0;
                logits[det * 3 * plane + 2 * plane + i] = -5.0;
            }
        }
        let mask_logits = from_slice::<f32>(&logits, &[2, 3, 4, 4]).unwrap();
        // Detection 0 → predicted label 1 (channel of ~1.0).
        // Detection 1 → predicted label 2 (channel of ~0.0).
        let labels = vec![1usize, 2];
        let boxes = from_slice::<f32>(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0], &[2, 4]).unwrap();
        let out =
            postprocess_masks::<f32>(&mask_logits, &labels, &boxes, [16, 16], false).unwrap();
        assert_eq!(out.shape(), &[2, 1, 4, 4]);
        let d = out.data_vec().unwrap();
        // Detection 0: picked class 1 → sigmoid(5) ≈ 0.993.
        assert!(d[0] > 0.99, "det 0 should be ~1.0, got {}", d[0]);
        // Detection 1: picked class 2 → sigmoid(-5) ≈ 0.0067.
        assert!(d[16] < 0.01, "det 1 should be ~0.0, got {}", d[16]);
    }
}
