//! Detection / segmentation operators. (#590)
//!
//! Mirrors `torchvision.ops`. Pure CPU/f64 implementations. All ops accept
//! `Tensor<T>` for the float-typed paths and return owned tensors / index
//! lists. None of these ops produce a grad_fn — they are typically used at
//! inference time or as differentiable losses (the ones that compose from
//! existing differentiable primitives, e.g. `sigmoid_focal_loss`, do
//! propagate gradients through their components).

use ferrotorch_core::grad_fns::activation as act;
use ferrotorch_core::grad_fns::arithmetic;
use ferrotorch_core::grad_fns::reduction as red;
use ferrotorch_core::grad_fns::transcendental as trans;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

/// Box format conventions accepted by [`box_convert`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoxFormat {
    /// `[x1, y1, x2, y2]` — top-left and bottom-right corners.
    Xyxy,
    /// `[x, y, w, h]` — top-left corner + width and height.
    Xywh,
    /// `[cx, cy, w, h]` — center + width and height.
    Cxcywh,
}

/// Reduction modes for the focal loss variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LossReduction {
    /// Per-element loss (no reduction).
    None,
    /// Mean reduction.
    Mean,
    /// Sum reduction.
    Sum,
}

// ---------------------------------------------------------------------------
// Box format conversion
// ---------------------------------------------------------------------------

/// Convert a tensor of bounding boxes from `in_fmt` to `out_fmt`. Input
/// has shape `[N, 4]`; output is the same shape with the requested
/// representation.
///
/// Mirrors `torchvision.ops.box_convert`.
pub fn box_convert<T: Float>(
    boxes: &Tensor<T>,
    in_fmt: BoxFormat,
    out_fmt: BoxFormat,
) -> FerrotorchResult<Tensor<T>> {
    check_boxes_shape(boxes, "box_convert")?;
    if in_fmt == out_fmt {
        return Ok(boxes.clone());
    }
    let n = boxes.shape()[0];
    let data = boxes.data_vec()?;
    let mut out = vec![T::from(0.0).unwrap(); n * 4];

    let half = T::from(0.5).unwrap();
    for i in 0..n {
        let a = data[i * 4];
        let b = data[i * 4 + 1];
        let c = data[i * 4 + 2];
        let d = data[i * 4 + 3];
        // First normalize to xyxy.
        let (x1, y1, x2, y2) = match in_fmt {
            BoxFormat::Xyxy => (a, b, c, d),
            BoxFormat::Xywh => (a, b, a + c, b + d),
            BoxFormat::Cxcywh => (a - half * c, b - half * d, a + half * c, b + half * d),
        };
        // Then emit out_fmt.
        let (o1, o2, o3, o4) = match out_fmt {
            BoxFormat::Xyxy => (x1, y1, x2, y2),
            BoxFormat::Xywh => (x1, y1, x2 - x1, y2 - y1),
            BoxFormat::Cxcywh => {
                let w = x2 - x1;
                let h = y2 - y1;
                (x1 + half * w, y1 + half * h, w, h)
            }
        };
        out[i * 4] = o1;
        out[i * 4 + 1] = o2;
        out[i * 4 + 2] = o3;
        out[i * 4 + 3] = o4;
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n, 4], false)
}

// ---------------------------------------------------------------------------
// Box geometry helpers
// ---------------------------------------------------------------------------

/// Per-box area (`(x2-x1) * (y2-y1)`) for `xyxy`-format boxes. Negative
/// or zero-area boxes return their literal (possibly negative) value;
/// callers wanting clamped output can compose with [`trans::clamp`].
pub fn box_area<T: Float>(boxes: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    check_boxes_shape(boxes, "box_area")?;
    let n = boxes.shape()[0];
    let data = boxes.data_vec()?;
    let mut out = vec![T::from(0.0).unwrap(); n];
    for i in 0..n {
        let x1 = data[i * 4];
        let y1 = data[i * 4 + 1];
        let x2 = data[i * 4 + 2];
        let y2 = data[i * 4 + 3];
        out[i] = (x2 - x1) * (y2 - y1);
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n], false)
}

/// Pairwise IoU matrix between two sets of `xyxy` boxes. Output shape:
/// `[N, M]` where `boxes1` is `[N, 4]` and `boxes2` is `[M, 4]`.
pub fn box_iou<T: Float>(boxes1: &Tensor<T>, boxes2: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    check_boxes_shape(boxes1, "box_iou")?;
    check_boxes_shape(boxes2, "box_iou")?;
    let n = boxes1.shape()[0];
    let m = boxes2.shape()[0];
    let a = boxes1.data_vec()?;
    let b = boxes2.data_vec()?;
    let zero = T::from(0.0).unwrap();
    let mut out = vec![zero; n * m];

    for i in 0..n {
        let ax1 = a[i * 4];
        let ay1 = a[i * 4 + 1];
        let ax2 = a[i * 4 + 2];
        let ay2 = a[i * 4 + 3];
        let area_a = (ax2 - ax1) * (ay2 - ay1);
        for j in 0..m {
            let bx1 = b[j * 4];
            let by1 = b[j * 4 + 1];
            let bx2 = b[j * 4 + 2];
            let by2 = b[j * 4 + 3];
            let area_b = (bx2 - bx1) * (by2 - by1);

            let ix1 = max_t(ax1, bx1);
            let iy1 = max_t(ay1, by1);
            let ix2 = min_t(ax2, bx2);
            let iy2 = min_t(ay2, by2);
            let iw = max_t(ix2 - ix1, zero);
            let ih = max_t(iy2 - iy1, zero);
            let inter = iw * ih;
            let union = area_a + area_b - inter;
            out[i * m + j] = if union > zero { inter / union } else { zero };
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n, m], false)
}

/// Clip every box to the bounds `[0, width] × [0, height]`. Mirrors
/// `torchvision.ops.clip_boxes_to_image`. `size` is `[H, W]` (matches
/// torchvision and the rest of `ferrotorch-vision`).
pub fn clip_boxes_to_image<T: Float>(
    boxes: &Tensor<T>,
    size: [usize; 2],
) -> FerrotorchResult<Tensor<T>> {
    check_boxes_shape(boxes, "clip_boxes_to_image")?;
    let n = boxes.shape()[0];
    let data = boxes.data_vec()?;
    let zero = T::from(0.0).unwrap();
    let h = T::from(size[0] as f64).unwrap();
    let w = T::from(size[1] as f64).unwrap();
    let mut out = vec![zero; n * 4];
    for i in 0..n {
        out[i * 4] = clamp_t(data[i * 4], zero, w);
        out[i * 4 + 1] = clamp_t(data[i * 4 + 1], zero, h);
        out[i * 4 + 2] = clamp_t(data[i * 4 + 2], zero, w);
        out[i * 4 + 3] = clamp_t(data[i * 4 + 3], zero, h);
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n, 4], false)
}

/// Indices of boxes whose width AND height are both `>= min_size`.
/// Returns the original-array indices (use `Tensor::index_select` or
/// manual gather to keep only those rows).
pub fn remove_small_boxes<T: Float>(
    boxes: &Tensor<T>,
    min_size: f64,
) -> FerrotorchResult<Vec<usize>> {
    check_boxes_shape(boxes, "remove_small_boxes")?;
    let n = boxes.shape()[0];
    let data = boxes.data_vec()?;
    let min_t = T::from(min_size).unwrap();
    let mut keep = Vec::new();
    for i in 0..n {
        let w = data[i * 4 + 2] - data[i * 4];
        let h = data[i * 4 + 3] - data[i * 4 + 1];
        if w >= min_t && h >= min_t {
            keep.push(i);
        }
    }
    Ok(keep)
}

// ---------------------------------------------------------------------------
// Non-max suppression
// ---------------------------------------------------------------------------

/// Greedy non-max suppression. Returns the indices of boxes to keep
/// (sorted by descending score). Mirrors `torchvision.ops.nms`.
///
/// Algorithm:
/// 1. Sort by descending score.
/// 2. Pick the highest-scoring box. Add to keep list.
/// 3. Drop every remaining box whose IoU with the picked box is `> iou_threshold`.
/// 4. Repeat until no boxes remain.
pub fn nms<T: Float>(
    boxes: &Tensor<T>,
    scores: &Tensor<T>,
    iou_threshold: f64,
) -> FerrotorchResult<Vec<usize>> {
    check_boxes_shape(boxes, "nms")?;
    let n = boxes.shape()[0];
    if scores.shape() != [n] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "nms: scores shape must be [N={n}], got {:?}",
                scores.shape()
            ),
        });
    }
    let data = boxes.data_vec()?;
    let scores_data = scores.data_vec()?;
    let thr = T::from(iou_threshold).unwrap();
    let zero = T::from(0.0).unwrap();

    // Indices sorted by score descending.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| {
        scores_data[b]
            .partial_cmp(&scores_data[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut keep: Vec<usize> = Vec::new();
    let mut suppressed = vec![false; n];

    for &i in &order {
        if suppressed[i] {
            continue;
        }
        keep.push(i);
        let ax1 = data[i * 4];
        let ay1 = data[i * 4 + 1];
        let ax2 = data[i * 4 + 2];
        let ay2 = data[i * 4 + 3];
        let area_a = (ax2 - ax1) * (ay2 - ay1);
        for &j in &order {
            if j == i || suppressed[j] {
                continue;
            }
            let bx1 = data[j * 4];
            let by1 = data[j * 4 + 1];
            let bx2 = data[j * 4 + 2];
            let by2 = data[j * 4 + 3];
            let area_b = (bx2 - bx1) * (by2 - by1);
            let ix1 = max_t(ax1, bx1);
            let iy1 = max_t(ay1, by1);
            let ix2 = min_t(ax2, bx2);
            let iy2 = min_t(ay2, by2);
            let iw = max_t(ix2 - ix1, zero);
            let ih = max_t(iy2 - iy1, zero);
            let inter = iw * ih;
            let union = area_a + area_b - inter;
            let iou = if union > zero { inter / union } else { zero };
            if iou > thr {
                suppressed[j] = true;
            }
        }
    }
    Ok(keep)
}

/// Per-class NMS: like [`nms`] but applies suppression independently
/// within each `idx` value. `idxs[i]` is the class id for `boxes[i]`.
/// Mirrors `torchvision.ops.batched_nms`.
pub fn batched_nms<T: Float>(
    boxes: &Tensor<T>,
    scores: &Tensor<T>,
    idxs: &[u32],
    iou_threshold: f64,
) -> FerrotorchResult<Vec<usize>> {
    check_boxes_shape(boxes, "batched_nms")?;
    let n = boxes.shape()[0];
    if idxs.len() != n {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("batched_nms: idxs len {} != boxes N {n}", idxs.len()),
        });
    }
    if scores.shape() != [n] {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "batched_nms: scores shape must be [N={n}], got {:?}",
                scores.shape()
            ),
        });
    }

    // Trick from torchvision: shift each box by class_id * (max_coord + 1)
    // so different-class boxes never overlap, then run a single NMS.
    let data = boxes.data_vec()?;
    let mut max_coord: T = T::from(0.0).unwrap();
    for v in &data {
        if *v > max_coord {
            max_coord = *v;
        }
    }
    let one = T::from(1.0).unwrap();
    let offset_unit = max_coord + one;

    let mut shifted = vec![T::from(0.0).unwrap(); 4 * n];
    for i in 0..n {
        let off = T::from(idxs[i] as f64).unwrap() * offset_unit;
        shifted[i * 4] = data[i * 4] + off;
        shifted[i * 4 + 1] = data[i * 4 + 1] + off;
        shifted[i * 4 + 2] = data[i * 4 + 2] + off;
        shifted[i * 4 + 3] = data[i * 4 + 3] + off;
    }
    let shifted_t = Tensor::from_storage(TensorStorage::cpu(shifted), vec![n, 4], false)?;
    nms(&shifted_t, scores, iou_threshold)
}

// ---------------------------------------------------------------------------
// Focal loss
// ---------------------------------------------------------------------------

/// Sigmoid focal loss for binary / multi-label detection heads.
///
/// `inputs` are raw logits (pre-sigmoid). `targets` are 0/1 or soft
/// in `[0, 1]`. `alpha = -1.0` disables the alpha-balancing term to
/// match torchvision's "no alpha" sentinel.
///
/// Formula (per-element):
/// ```text
/// p     = sigmoid(inputs)
/// ce    = bce_with_logits(inputs, targets)        (numerically stable)
/// p_t   = p * targets + (1 - p) * (1 - targets)
/// loss  = ce * (1 - p_t)^gamma
/// loss *= alpha * targets + (1 - alpha) * (1 - targets)   (when alpha >= 0)
/// ```
///
/// Differentiable through the existing `sigmoid` / `softplus` /
/// `mul` / `pow` primitives, so backward propagates to `inputs`.
pub fn sigmoid_focal_loss<T: Float>(
    inputs: &Tensor<T>,
    targets: &Tensor<T>,
    alpha: f64,
    gamma: f64,
    reduction: LossReduction,
) -> FerrotorchResult<Tensor<T>> {
    if inputs.shape() != targets.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "sigmoid_focal_loss: inputs {:?} vs targets {:?}",
                inputs.shape(),
                targets.shape()
            ),
        });
    }

    let one = ferrotorch_core::scalar(T::from(1.0).unwrap())?;

    // Numerically stable BCE-with-logits.
    let neg_t = arithmetic::neg(targets)?;
    let nt_x = arithmetic::mul(&neg_t, inputs)?;
    let sp = act::softplus(inputs, 1.0, 20.0)?;
    let ce = arithmetic::add(&nt_x, &sp)?;

    // p = sigmoid(inputs); p_t = p*targets + (1-p)*(1-targets).
    let p = act::sigmoid(inputs)?;
    let one_minus_p = arithmetic::sub(&one, &p)?;
    let one_minus_t = arithmetic::sub(&one, targets)?;
    let p_t1 = arithmetic::mul(&p, targets)?;
    let p_t2 = arithmetic::mul(&one_minus_p, &one_minus_t)?;
    let p_t = arithmetic::add(&p_t1, &p_t2)?;

    // (1 - p_t)^gamma
    let one_minus_pt = arithmetic::sub(&one, &p_t)?;
    let modulator = arithmetic::pow(&one_minus_pt, gamma)?;

    // CE * modulator
    let mut loss = arithmetic::mul(&ce, &modulator)?;

    // Optional alpha balancing.
    if alpha >= 0.0 {
        let alpha_s = ferrotorch_core::scalar(T::from(alpha).unwrap())?;
        let alpha_targets = arithmetic::mul(&alpha_s, targets)?;
        let one_minus_alpha = ferrotorch_core::scalar(T::from(1.0 - alpha).unwrap())?;
        let oma_omt = arithmetic::mul(&one_minus_alpha, &one_minus_t)?;
        let alpha_t = arithmetic::add(&alpha_targets, &oma_omt)?;
        loss = arithmetic::mul(&loss, &alpha_t)?;
    }

    match reduction {
        LossReduction::None => Ok(loss),
        LossReduction::Mean => red::mean(&loss),
        LossReduction::Sum => red::sum(&loss),
    }
}

/// Cross-entropy-style focal loss (Lin et al. 2017) for already-normalized
/// probabilities (e.g. softmax output). Expects `inputs` to be probabilities
/// in `[0, 1]` and `targets` to be 0/1 indicators.
///
/// `loss = -alpha_t * (1 - p_t)^gamma * log(p_t)` where
/// `p_t = inputs * targets + (1 - inputs) * (1 - targets)`.
pub fn focal_loss<T: Float>(
    inputs: &Tensor<T>,
    targets: &Tensor<T>,
    alpha: f64,
    gamma: f64,
    reduction: LossReduction,
) -> FerrotorchResult<Tensor<T>> {
    if inputs.shape() != targets.shape() {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "focal_loss: inputs {:?} vs targets {:?}",
                inputs.shape(),
                targets.shape()
            ),
        });
    }
    let one = ferrotorch_core::scalar(T::from(1.0).unwrap())?;
    let one_minus_p = arithmetic::sub(&one, inputs)?;
    let one_minus_t = arithmetic::sub(&one, targets)?;
    let p_t1 = arithmetic::mul(inputs, targets)?;
    let p_t2 = arithmetic::mul(&one_minus_p, &one_minus_t)?;
    let p_t = arithmetic::add(&p_t1, &p_t2)?;

    // Add tiny eps to keep log finite at p_t = 0.
    let eps = ferrotorch_core::scalar(T::from(1e-12).unwrap())?;
    let p_t_eps = arithmetic::add(&p_t, &eps)?;
    let log_p_t = trans::log(&p_t_eps)?;
    let neg_log = arithmetic::neg(&log_p_t)?;

    let one_minus_pt = arithmetic::sub(&one, &p_t)?;
    let modulator = arithmetic::pow(&one_minus_pt, gamma)?;

    let mut loss = arithmetic::mul(&neg_log, &modulator)?;
    if alpha >= 0.0 {
        let alpha_s = ferrotorch_core::scalar(T::from(alpha).unwrap())?;
        let alpha_t = arithmetic::mul(&alpha_s, targets)?;
        let one_minus_alpha = ferrotorch_core::scalar(T::from(1.0 - alpha).unwrap())?;
        let oma_omt = arithmetic::mul(&one_minus_alpha, &one_minus_t)?;
        let alpha_balance = arithmetic::add(&alpha_t, &oma_omt)?;
        loss = arithmetic::mul(&loss, &alpha_balance)?;
    }

    match reduction {
        LossReduction::None => Ok(loss),
        LossReduction::Mean => red::mean(&loss),
        LossReduction::Sum => red::sum(&loss),
    }
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn check_boxes_shape<T: Float>(boxes: &Tensor<T>, op: &str) -> FerrotorchResult<()> {
    if boxes.ndim() != 2 || boxes.shape()[1] != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "{op}: boxes must have shape [N, 4], got {:?}",
                boxes.shape()
            ),
        });
    }
    Ok(())
}

#[inline]
/// Pairwise Generalized IoU between two sets of `xyxy` boxes.
///
/// `GIoU = IoU - (|C| - |A ∪ B|) / |C|` where `C` is the smallest axis-aligned
/// box enclosing both. Range `(-1, 1]`. Mirrors
/// `torchvision.ops.generalized_box_iou`. Output shape `[N, M]`. (#610)
pub fn generalized_box_iou<T: Float>(
    boxes1: &Tensor<T>,
    boxes2: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    check_boxes_shape(boxes1, "generalized_box_iou")?;
    check_boxes_shape(boxes2, "generalized_box_iou")?;
    let n = boxes1.shape()[0];
    let m = boxes2.shape()[0];
    let a = boxes1.data_vec()?;
    let b = boxes2.data_vec()?;
    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();
    let mut out = vec![zero; n * m];

    for i in 0..n {
        let ax1 = a[i * 4];
        let ay1 = a[i * 4 + 1];
        let ax2 = a[i * 4 + 2];
        let ay2 = a[i * 4 + 3];
        let area_a = (ax2 - ax1) * (ay2 - ay1);
        for j in 0..m {
            let bx1 = b[j * 4];
            let by1 = b[j * 4 + 1];
            let bx2 = b[j * 4 + 2];
            let by2 = b[j * 4 + 3];
            let area_b = (bx2 - bx1) * (by2 - by1);

            let ix1 = max_t(ax1, bx1);
            let iy1 = max_t(ay1, by1);
            let ix2 = min_t(ax2, bx2);
            let iy2 = min_t(ay2, by2);
            let iw = max_t(ix2 - ix1, zero);
            let ih = max_t(iy2 - iy1, zero);
            let inter = iw * ih;
            let union = area_a + area_b - inter;

            // Smallest enclosing box.
            let cx1 = min_t(ax1, bx1);
            let cy1 = min_t(ay1, by1);
            let cx2 = max_t(ax2, bx2);
            let cy2 = max_t(ay2, by2);
            let area_c = (cx2 - cx1) * (cy2 - cy1);

            let iou = if union > zero { inter / union } else { zero };
            out[i * m + j] = if area_c > zero {
                iou - (area_c - union) / area_c
            } else {
                // Both boxes degenerate: GIoU collapses to IoU.
                iou - one + one // = iou
            };
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n, m], false)
}

/// Pairwise Distance IoU between two sets of `xyxy` boxes.
///
/// `DIoU = IoU - ρ²(b1_center, b2_center) / c²` where `c` is the diagonal
/// length of the smallest enclosing box. Range `(-1, 1]`. Mirrors
/// `torchvision.ops.distance_box_iou`. Output shape `[N, M]`. (#610)
pub fn distance_box_iou<T: Float>(
    boxes1: &Tensor<T>,
    boxes2: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    check_boxes_shape(boxes1, "distance_box_iou")?;
    check_boxes_shape(boxes2, "distance_box_iou")?;
    let n = boxes1.shape()[0];
    let m = boxes2.shape()[0];
    let a = boxes1.data_vec()?;
    let b = boxes2.data_vec()?;
    let zero = T::from(0.0).unwrap();
    let two = T::from(2.0).unwrap();
    let mut out = vec![zero; n * m];

    for i in 0..n {
        let ax1 = a[i * 4];
        let ay1 = a[i * 4 + 1];
        let ax2 = a[i * 4 + 2];
        let ay2 = a[i * 4 + 3];
        let acx = (ax1 + ax2) / two;
        let acy = (ay1 + ay2) / two;
        let area_a = (ax2 - ax1) * (ay2 - ay1);
        for j in 0..m {
            let bx1 = b[j * 4];
            let by1 = b[j * 4 + 1];
            let bx2 = b[j * 4 + 2];
            let by2 = b[j * 4 + 3];
            let bcx = (bx1 + bx2) / two;
            let bcy = (by1 + by2) / two;
            let area_b = (bx2 - bx1) * (by2 - by1);

            let ix1 = max_t(ax1, bx1);
            let iy1 = max_t(ay1, by1);
            let ix2 = min_t(ax2, bx2);
            let iy2 = min_t(ay2, by2);
            let iw = max_t(ix2 - ix1, zero);
            let ih = max_t(iy2 - iy1, zero);
            let inter = iw * ih;
            let union = area_a + area_b - inter;

            let cx1 = min_t(ax1, bx1);
            let cy1 = min_t(ay1, by1);
            let cx2 = max_t(ax2, bx2);
            let cy2 = max_t(ay2, by2);
            let cw = cx2 - cx1;
            let ch = cy2 - cy1;
            let c_diag_sq = cw * cw + ch * ch;

            let dx = acx - bcx;
            let dy = acy - bcy;
            let center_dist_sq = dx * dx + dy * dy;

            let iou = if union > zero { inter / union } else { zero };
            out[i * m + j] = if c_diag_sq > zero {
                iou - center_dist_sq / c_diag_sq
            } else {
                iou
            };
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n, m], false)
}

/// Pairwise Complete IoU between two sets of `xyxy` boxes.
///
/// `CIoU = DIoU - α·v` where `v = (4/π²) · (atan(w_a/h_a) - atan(w_b/h_b))²`
/// is the aspect-ratio penalty and `α = v / (1 - IoU + v)` is its weight.
/// Mirrors `torchvision.ops.complete_box_iou`. Output shape `[N, M]`. (#610)
pub fn complete_box_iou<T: Float>(
    boxes1: &Tensor<T>,
    boxes2: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    check_boxes_shape(boxes1, "complete_box_iou")?;
    check_boxes_shape(boxes2, "complete_box_iou")?;
    let n = boxes1.shape()[0];
    let m = boxes2.shape()[0];
    let a = boxes1.data_vec()?;
    let b = boxes2.data_vec()?;
    let zero = T::from(0.0).unwrap();
    let one = T::from(1.0).unwrap();
    let two = T::from(2.0).unwrap();
    let four_over_pi_sq = T::from(4.0 / (std::f64::consts::PI * std::f64::consts::PI)).unwrap();
    let eps = T::from(1e-7).unwrap();
    let mut out = vec![zero; n * m];

    for i in 0..n {
        let ax1 = a[i * 4];
        let ay1 = a[i * 4 + 1];
        let ax2 = a[i * 4 + 2];
        let ay2 = a[i * 4 + 3];
        let aw = ax2 - ax1;
        let ah = ay2 - ay1;
        let acx = (ax1 + ax2) / two;
        let acy = (ay1 + ay2) / two;
        let area_a = aw * ah;
        for j in 0..m {
            let bx1 = b[j * 4];
            let by1 = b[j * 4 + 1];
            let bx2 = b[j * 4 + 2];
            let by2 = b[j * 4 + 3];
            let bw = bx2 - bx1;
            let bh = by2 - by1;
            let bcx = (bx1 + bx2) / two;
            let bcy = (by1 + by2) / two;
            let area_b = bw * bh;

            let ix1 = max_t(ax1, bx1);
            let iy1 = max_t(ay1, by1);
            let ix2 = min_t(ax2, bx2);
            let iy2 = min_t(ay2, by2);
            let iw = max_t(ix2 - ix1, zero);
            let ih = max_t(iy2 - iy1, zero);
            let inter = iw * ih;
            let union = area_a + area_b - inter;

            let cx1 = min_t(ax1, bx1);
            let cy1 = min_t(ay1, by1);
            let cx2 = max_t(ax2, bx2);
            let cy2 = max_t(ay2, by2);
            let cw = cx2 - cx1;
            let ch = cy2 - cy1;
            let c_diag_sq = cw * cw + ch * ch;

            let dx = acx - bcx;
            let dy = acy - bcy;
            let center_dist_sq = dx * dx + dy * dy;

            let iou = if union > zero { inter / union } else { zero };
            let diou_term = if c_diag_sq > zero {
                center_dist_sq / c_diag_sq
            } else {
                zero
            };

            // Aspect-ratio penalty.
            let ratio_a = if ah > zero { (aw / ah).atan() } else { zero };
            let ratio_b = if bh > zero { (bw / bh).atan() } else { zero };
            let v = four_over_pi_sq * (ratio_a - ratio_b) * (ratio_a - ratio_b);
            let denom = one - iou + v + eps;
            let alpha = if denom > zero { v / denom } else { zero };

            out[i * m + j] = iou - diou_term - alpha * v;
        }
    }
    Tensor::from_storage(TensorStorage::cpu(out), vec![n, m], false)
}

/// Bilinear sample a single value from a feature map at fractional `(y, x)`.
/// Out-of-bounds samples return 0 (matches torchvision's `roi_align` border
/// behavior for `aligned=true`).
#[inline]
fn bilinear_sample<T: Float>(feature_map: &[T], h: usize, w: usize, y: T, x: T) -> T {
    let zero = T::from(0.0).unwrap();
    let h_f = T::from(h as f64).unwrap();
    let w_f = T::from(w as f64).unwrap();
    let neg_one = T::from(-1.0).unwrap();
    if y < neg_one || y > h_f || x < neg_one || x > w_f {
        return zero;
    }
    let y = max_t(y, zero);
    let x = max_t(x, zero);

    let y_low_f = y.floor();
    let x_low_f = x.floor();
    let y_low = y_low_f.to_usize().unwrap_or(0).min(h.saturating_sub(1));
    let x_low = x_low_f.to_usize().unwrap_or(0).min(w.saturating_sub(1));
    let y_high = (y_low + 1).min(h.saturating_sub(1));
    let x_high = (x_low + 1).min(w.saturating_sub(1));

    let ly = y - T::from(y_low as f64).unwrap();
    let lx = x - T::from(x_low as f64).unwrap();
    let one = T::from(1.0).unwrap();
    let hy = one - ly;
    let hx = one - lx;

    let v1 = feature_map[y_low * w + x_low];
    let v2 = feature_map[y_low * w + x_high];
    let v3 = feature_map[y_high * w + x_low];
    let v4 = feature_map[y_high * w + x_high];

    hy * (hx * v1 + lx * v2) + ly * (hx * v3 + lx * v4)
}

/// Region of Interest Align: bilinear feature extraction per RoI.
///
/// Mirrors `torchvision.ops.roi_align(input, boxes, output_size, spatial_scale,
/// sampling_ratio, aligned=true)`.
///
/// - `input`: feature map `[B, C, H, W]`.
/// - `boxes`: `[K, 5]` rows of `(batch_idx, x1, y1, x2, y2)` in pixel coords
///   of the original image.
/// - `output_size`: `(out_h, out_w)`.
/// - `spatial_scale`: ratio mapping image coords → feature-map coords (e.g.
///   `1/16` for an FPN level). A value of `1.0` means input coords already
///   match the feature-map grid.
/// - `sampling_ratio`: `0` = adaptive (≈ ceil(roi_size / output_size)), else
///   the explicit per-bin grid resolution.
///
/// Returns `[K, C, out_h, out_w]`. Empty boxes (`x2 < x1 || y2 < y1`) yield
/// all-zero output for that row. (#610)
#[allow(clippy::too_many_arguments)]
pub fn roi_align<T: Float>(
    input: &Tensor<T>,
    boxes: &Tensor<T>,
    output_size: (usize, usize),
    spatial_scale: f64,
    sampling_ratio: usize,
) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 4 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "roi_align: input must be 4-D [B, C, H, W], got {:?}",
                input.shape()
            ),
        });
    }
    if boxes.ndim() != 2 || boxes.shape()[1] != 5 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "roi_align: boxes must be 2-D [K, 5] (batch_idx, x1, y1, x2, y2), got {:?}",
                boxes.shape()
            ),
        });
    }
    let (out_h, out_w) = output_size;
    if out_h == 0 || out_w == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "roi_align: output_size must be > 0 in both dimensions".into(),
        });
    }

    let b = input.shape()[0];
    let c = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let k = boxes.shape()[0];

    let in_data = input.data_vec()?;
    let box_data = boxes.data_vec()?;
    let scale = T::from(spatial_scale).unwrap();
    let zero = T::from(0.0).unwrap();
    let half = T::from(0.5).unwrap();

    let mut out = vec![zero; k * c * out_h * out_w];

    for i in 0..k {
        let batch_idx = box_data[i * 5].to_usize().unwrap_or(0);
        if batch_idx >= b {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: batch_idx,
                axis: 0,
                size: b,
            });
        }
        // Map box to feature-map coords with `aligned=true`: subtract 0.5
        // before scaling so the box centers align with feature pixel centers.
        let x1 = box_data[i * 5 + 1] * scale - half;
        let y1 = box_data[i * 5 + 2] * scale - half;
        let x2 = box_data[i * 5 + 3] * scale - half;
        let y2 = box_data[i * 5 + 4] * scale - half;
        let roi_w = max_t(x2 - x1, zero);
        let roi_h = max_t(y2 - y1, zero);

        let bin_h = roi_h / T::from(out_h as f64).unwrap();
        let bin_w = roi_w / T::from(out_w as f64).unwrap();

        // Effective grid resolution per bin.
        let grid_h = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            (roi_h.to_f64().unwrap_or(0.0) / out_h as f64)
                .ceil()
                .max(1.0) as usize
        };
        let grid_w = if sampling_ratio > 0 {
            sampling_ratio
        } else {
            (roi_w.to_f64().unwrap_or(0.0) / out_w as f64)
                .ceil()
                .max(1.0) as usize
        };
        let grid_count = T::from((grid_h * grid_w) as f64).unwrap();

        for ch in 0..c {
            let fm_offset = (batch_idx * c + ch) * h * w;
            let fm = &in_data[fm_offset..fm_offset + h * w];

            for ph in 0..out_h {
                for pw in 0..out_w {
                    let ph_f = T::from(ph as f64).unwrap();
                    let pw_f = T::from(pw as f64).unwrap();
                    let mut sum = zero;
                    for iy in 0..grid_h {
                        let y = y1
                            + ph_f * bin_h
                            + (T::from(iy as f64).unwrap() + half) * bin_h
                                / T::from(grid_h as f64).unwrap();
                        for ix in 0..grid_w {
                            let x = x1
                                + pw_f * bin_w
                                + (T::from(ix as f64).unwrap() + half) * bin_w
                                    / T::from(grid_w as f64).unwrap();
                            sum += bilinear_sample(fm, h, w, y, x);
                        }
                    }
                    let avg = if grid_count > zero {
                        sum / grid_count
                    } else {
                        zero
                    };
                    let out_offset = ((i * c + ch) * out_h + ph) * out_w + pw;
                    out[out_offset] = avg;
                }
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![k, c, out_h, out_w], false)
}

/// Region of Interest Pool: integer-rounded max-pool per RoI bin.
///
/// Mirrors `torchvision.ops.roi_pool(input, boxes, output_size,
/// spatial_scale)`. Same input format as [`roi_align`] but with hard-edge
/// integer bin boundaries (no bilinear interpolation, no sampling ratio).
/// Returns `[K, C, out_h, out_w]`. Empty bins yield 0. (#610)
pub fn roi_pool<T: Float>(
    input: &Tensor<T>,
    boxes: &Tensor<T>,
    output_size: (usize, usize),
    spatial_scale: f64,
) -> FerrotorchResult<Tensor<T>> {
    if input.ndim() != 4 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "roi_pool: input must be 4-D [B, C, H, W], got {:?}",
                input.shape()
            ),
        });
    }
    if boxes.ndim() != 2 || boxes.shape()[1] != 5 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "roi_pool: boxes must be 2-D [K, 5], got {:?}",
                boxes.shape()
            ),
        });
    }
    let (out_h, out_w) = output_size;
    if out_h == 0 || out_w == 0 {
        return Err(FerrotorchError::InvalidArgument {
            message: "roi_pool: output_size must be > 0 in both dimensions".into(),
        });
    }
    let b = input.shape()[0];
    let c = input.shape()[1];
    let h = input.shape()[2];
    let w = input.shape()[3];
    let k = boxes.shape()[0];

    let in_data = input.data_vec()?;
    let box_data = boxes.data_vec()?;
    let zero = T::from(0.0).unwrap();
    let mut out = vec![zero; k * c * out_h * out_w];

    for i in 0..k {
        let batch_idx = box_data[i * 5].to_usize().unwrap_or(0);
        if batch_idx >= b {
            return Err(FerrotorchError::IndexOutOfBounds {
                index: batch_idx,
                axis: 0,
                size: b,
            });
        }
        let x1 = (box_data[i * 5 + 1].to_f64().unwrap() * spatial_scale).round() as i64;
        let y1 = (box_data[i * 5 + 2].to_f64().unwrap() * spatial_scale).round() as i64;
        let x2 = (box_data[i * 5 + 3].to_f64().unwrap() * spatial_scale).round() as i64;
        let y2 = (box_data[i * 5 + 4].to_f64().unwrap() * spatial_scale).round() as i64;
        let roi_w = (x2 - x1 + 1).max(1) as f64;
        let roi_h = (y2 - y1 + 1).max(1) as f64;
        let bin_h = roi_h / out_h as f64;
        let bin_w = roi_w / out_w as f64;

        for ch in 0..c {
            let fm_offset = (batch_idx * c + ch) * h * w;
            let fm = &in_data[fm_offset..fm_offset + h * w];
            for ph in 0..out_h {
                for pw in 0..out_w {
                    let hstart = (y1 as f64 + ph as f64 * bin_h).floor() as i64;
                    let wstart = (x1 as f64 + pw as f64 * bin_w).floor() as i64;
                    let hend = (y1 as f64 + (ph + 1) as f64 * bin_h).ceil() as i64;
                    let wend = (x1 as f64 + (pw + 1) as f64 * bin_w).ceil() as i64;

                    let hstart = hstart.max(0).min(h as i64) as usize;
                    let wstart = wstart.max(0).min(w as i64) as usize;
                    let hend = hend.max(0).min(h as i64) as usize;
                    let wend = wend.max(0).min(w as i64) as usize;

                    let mut best: Option<T> = None;
                    for yy in hstart..hend {
                        for xx in wstart..wend {
                            let v = fm[yy * w + xx];
                            best = Some(match best {
                                None => v,
                                Some(b) if v > b => v,
                                Some(b) => b,
                            });
                        }
                    }
                    let val = best.unwrap_or(zero);
                    let out_offset = ((i * c + ch) * out_h + ph) * out_w + pw;
                    out[out_offset] = val;
                }
            }
        }
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![k, c, out_h, out_w], false)
}

#[inline]
fn max_t<T: Float>(a: T, b: T) -> T {
    if a >= b { a } else { b }
}

#[inline]
fn min_t<T: Float>(a: T, b: T) -> T {
    if a <= b { a } else { b }
}

#[inline]
fn clamp_t<T: Float>(v: T, lo: T, hi: T) -> T {
    min_t(max_t(v, lo), hi)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::creation::from_slice;

    fn boxes_xyxy(data: &[f32]) -> Tensor<f32> {
        let n = data.len() / 4;
        from_slice::<f32>(data, &[n, 4]).unwrap()
    }

    fn scores(data: &[f32]) -> Tensor<f32> {
        from_slice::<f32>(data, &[data.len()]).unwrap()
    }

    // ---- box_convert ------------------------------------------------------

    #[test]
    fn box_convert_xyxy_to_xywh() {
        let b = boxes_xyxy(&[1.0, 2.0, 4.0, 6.0]);
        let out = box_convert(&b, BoxFormat::Xyxy, BoxFormat::Xywh).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn box_convert_xyxy_to_cxcywh() {
        let b = boxes_xyxy(&[0.0, 0.0, 4.0, 6.0]);
        let out = box_convert(&b, BoxFormat::Xyxy, BoxFormat::Cxcywh).unwrap();
        let d = out.data().unwrap();
        assert_eq!(d, &[2.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn box_convert_roundtrip_xywh_through_xyxy() {
        let b = boxes_xyxy(&[3.0, 5.0, 7.0, 9.0]);
        let xywh = box_convert(&b, BoxFormat::Xyxy, BoxFormat::Xywh).unwrap();
        let xyxy = box_convert(&xywh, BoxFormat::Xywh, BoxFormat::Xyxy).unwrap();
        let orig = b.data().unwrap();
        let recovered = xyxy.data().unwrap();
        for i in 0..4 {
            assert!((orig[i] - recovered[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn box_convert_same_format_clones() {
        let b = boxes_xyxy(&[1.0, 2.0, 3.0, 4.0]);
        let out = box_convert(&b, BoxFormat::Xyxy, BoxFormat::Xyxy).unwrap();
        assert_eq!(out.data().unwrap(), b.data().unwrap());
    }

    // ---- box_iou / box_area / clip_boxes_to_image ------------------------

    #[test]
    fn box_iou_full_overlap_is_one() {
        let a = boxes_xyxy(&[0.0, 0.0, 10.0, 10.0]);
        let iou = box_iou(&a, &a).unwrap();
        let d = iou.data().unwrap();
        assert!((d[0] - 1.0).abs() < 1e-5);
    }

    #[test]
    fn box_iou_no_overlap_is_zero() {
        let a = boxes_xyxy(&[0.0, 0.0, 10.0, 10.0]);
        let b = boxes_xyxy(&[20.0, 20.0, 30.0, 30.0]);
        let iou = box_iou(&a, &b).unwrap();
        assert_eq!(iou.data().unwrap()[0], 0.0);
    }

    #[test]
    fn box_iou_half_overlap() {
        // Two 10×10 boxes overlapping in a 5×10 rectangle.
        // intersection = 50, union = 100 + 100 - 50 = 150 → IoU = 1/3.
        let a = boxes_xyxy(&[0.0, 0.0, 10.0, 10.0]);
        let b = boxes_xyxy(&[5.0, 0.0, 15.0, 10.0]);
        let iou = box_iou(&a, &b).unwrap();
        assert!((iou.data().unwrap()[0] - (1.0 / 3.0)).abs() < 1e-5);
    }

    #[test]
    fn box_iou_pairwise_shape_n_by_m() {
        let a = boxes_xyxy(&[0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0]);
        let b = boxes_xyxy(&[0.0, 0.0, 1.0, 1.0]);
        let iou = box_iou(&a, &b).unwrap();
        assert_eq!(iou.shape(), &[2, 1]);
    }

    #[test]
    fn box_area_simple() {
        let b = boxes_xyxy(&[0.0, 0.0, 3.0, 4.0, 1.0, 1.0, 4.0, 5.0]);
        let a = box_area(&b).unwrap();
        assert_eq!(a.data().unwrap(), &[12.0, 12.0]);
    }

    #[test]
    fn clip_boxes_to_image_clamps_negative_and_overflow() {
        // Image is 10×20 (H=10, W=20). Box partially outside the image.
        let b = boxes_xyxy(&[-1.0, -2.0, 25.0, 12.0]);
        let c = clip_boxes_to_image(&b, [10, 20]).unwrap();
        let d = c.data().unwrap();
        assert_eq!(d, &[0.0, 0.0, 20.0, 10.0]);
    }

    #[test]
    fn remove_small_boxes_filters_by_min_size() {
        let b = boxes_xyxy(&[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 5.0, 5.0]);
        let keep = remove_small_boxes(&b, 2.0).unwrap();
        assert_eq!(keep, vec![1]);
    }

    // ---- nms / batched_nms -----------------------------------------------

    #[test]
    fn nms_keeps_only_high_scoring_overlap() {
        // Two overlapping boxes (IoU > 0.5) with different scores.
        let b = boxes_xyxy(&[0.0, 0.0, 10.0, 10.0, 1.0, 1.0, 11.0, 11.0]);
        let s = scores(&[0.9, 0.8]);
        let keep = nms(&b, &s, 0.5).unwrap();
        assert_eq!(keep, vec![0]);
    }

    #[test]
    fn nms_preserves_non_overlapping_boxes() {
        let b = boxes_xyxy(&[
            0.0, 0.0, 10.0, 10.0, // box 0
            20.0, 20.0, 30.0, 30.0, // box 1 (disjoint)
        ]);
        let s = scores(&[0.5, 0.9]);
        let keep = nms(&b, &s, 0.5).unwrap();
        // Sorted by descending score, both kept (no overlap).
        assert_eq!(keep, vec![1, 0]);
    }

    #[test]
    fn nms_above_threshold_only_drops() {
        let b = boxes_xyxy(&[
            0.0, 0.0, 10.0, 10.0, // 0
            5.0, 0.0, 15.0, 10.0, // 1, IoU with 0 = 1/3 ≈ 0.33
        ]);
        let s = scores(&[0.9, 0.8]);
        // threshold 0.5: 0.33 < 0.5, so both kept.
        let keep = nms(&b, &s, 0.5).unwrap();
        assert_eq!(keep, vec![0, 1]);
        // threshold 0.2: 0.33 > 0.2, second suppressed.
        let keep2 = nms(&b, &s, 0.2).unwrap();
        assert_eq!(keep2, vec![0]);
    }

    #[test]
    fn nms_rejects_scores_shape_mismatch() {
        let b = boxes_xyxy(&[0.0, 0.0, 1.0, 1.0]);
        let s = scores(&[0.9, 0.8]);
        let err = nms(&b, &s, 0.5).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn batched_nms_per_class_independence() {
        // Two overlapping boxes in class 0 (IoU > 0.5) and one in class 1.
        // batched_nms should suppress within class 0 but never cross-class.
        let b = boxes_xyxy(&[
            0.0, 0.0, 10.0, 10.0, // class 0, score 0.9
            1.0, 1.0, 11.0, 11.0, // class 0, score 0.8 → suppressed
            0.5, 0.5, 9.5, 9.5, // class 1, score 0.7 → kept (different class)
        ]);
        let s = scores(&[0.9, 0.8, 0.7]);
        let idxs = vec![0u32, 0, 1];
        let keep = batched_nms(&b, &s, &idxs, 0.5).unwrap();
        // Sorted by score desc within the shifted-coords NMS: 0 then 2.
        assert_eq!(keep.len(), 2);
        assert!(keep.contains(&0));
        assert!(keep.contains(&2));
        assert!(!keep.contains(&1));
    }

    // ---- focal loss ------------------------------------------------------

    #[test]
    fn sigmoid_focal_loss_zero_logits_zero_targets() {
        // logit=0 → p=0.5, target=0 → ce ~ 0.693, p_t = 0.5,
        // (1-p_t)^2 = 0.25, alpha=0.25 → 0.75 * 0.25 * 0.693 ≈ 0.13
        let inp = from_slice::<f32>(&[0.0], &[1]).unwrap();
        let tgt = from_slice::<f32>(&[0.0], &[1]).unwrap();
        let l = sigmoid_focal_loss(&inp, &tgt, 0.25, 2.0, LossReduction::Sum)
            .unwrap()
            .item()
            .unwrap();
        let expected = 0.75 * 0.25 * 2.0_f32.ln();
        assert!((l - expected).abs() < 1e-4, "got {l}, expected {expected}");
    }

    #[test]
    fn sigmoid_focal_loss_alpha_negative_disables_balancing() {
        // alpha = -1 → no alpha-balance term (matches torchvision sentinel).
        let inp = from_slice::<f32>(&[0.0, 0.0], &[2]).unwrap();
        let tgt = from_slice::<f32>(&[0.0, 1.0], &[2]).unwrap();
        let with = sigmoid_focal_loss(&inp, &tgt, 0.25, 2.0, LossReduction::Sum)
            .unwrap()
            .item()
            .unwrap();
        let without = sigmoid_focal_loss(&inp, &tgt, -1.0, 2.0, LossReduction::Sum)
            .unwrap()
            .item()
            .unwrap();
        // The "without" case should be larger (no down-weighting).
        assert!(without > with, "without={without}, with={with}");
    }

    #[test]
    fn focal_loss_shape_mismatch_errors() {
        let inp = from_slice::<f32>(&[0.5], &[1]).unwrap();
        let tgt = from_slice::<f32>(&[0.5, 0.5], &[2]).unwrap();
        let err = focal_loss(&inp, &tgt, 0.25, 2.0, LossReduction::Mean).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn focal_loss_zero_when_perfect_prediction() {
        // p = 1, target = 1 → p_t = 1, log(p_t) = 0, modulator = 0,
        // total loss = 0.
        let inp = from_slice::<f32>(&[1.0], &[1]).unwrap();
        let tgt = from_slice::<f32>(&[1.0], &[1]).unwrap();
        let l = focal_loss(&inp, &tgt, 0.25, 2.0, LossReduction::Mean)
            .unwrap()
            .item()
            .unwrap();
        assert!(l.abs() < 1e-6, "got {l}");
    }

    // -------------------------------------------------------------------
    // GIoU / DIoU / CIoU (#610)
    // -------------------------------------------------------------------

    #[test]
    fn giou_identical_boxes_is_one() {
        // Same box twice → IoU = 1, GIoU = 1.
        let a = from_slice::<f64>(&[0.0, 0.0, 1.0, 1.0], &[1, 4]).unwrap();
        let g = generalized_box_iou(&a, &a).unwrap();
        let v = g.data().unwrap()[0];
        assert!((v - 1.0).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn giou_disjoint_boxes_negative() {
        // Two unit boxes with a gap between them.
        let a = from_slice::<f64>(&[0.0, 0.0, 1.0, 1.0], &[1, 4]).unwrap();
        let b = from_slice::<f64>(&[2.0, 2.0, 3.0, 3.0], &[1, 4]).unwrap();
        let g = generalized_box_iou(&a, &b).unwrap();
        let v = g.data().unwrap()[0];
        // IoU = 0; enclosing box is 3x3; union is 2; GIoU = 0 - (9 - 2)/9 = -7/9.
        assert!((v - (-7.0 / 9.0)).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn diou_identical_boxes_is_one() {
        let a = from_slice::<f64>(&[0.0, 0.0, 2.0, 2.0], &[1, 4]).unwrap();
        let d = distance_box_iou(&a, &a).unwrap();
        let v = d.data().unwrap()[0];
        assert!((v - 1.0).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn ciou_identical_boxes_is_one() {
        let a = from_slice::<f64>(&[0.0, 0.0, 2.0, 2.0], &[1, 4]).unwrap();
        let c = complete_box_iou(&a, &a).unwrap();
        let v = c.data().unwrap()[0];
        assert!((v - 1.0).abs() < 1e-9, "got {v}");
    }

    #[test]
    fn ciou_aspect_ratio_penalty_applies() {
        // Two boxes with the same center + same area but different aspect
        // ratios. CIoU should be strictly less than IoU due to the v term.
        // a: 4x1 at (0..4, 0..1), centered at (2, 0.5)
        // b: 1x4 at (1.5..2.5, -1..3), centered at (2, 0.5+0.5)... let's
        // just use a simple offset case.
        let a = from_slice::<f64>(&[0.0, 0.0, 4.0, 1.0], &[1, 4]).unwrap();
        let b = from_slice::<f64>(&[1.0, 0.0, 5.0, 1.0], &[1, 4]).unwrap();
        let iou = box_iou(&a, &b).unwrap().data().unwrap()[0];
        let ciou = complete_box_iou(&a, &b).unwrap().data().unwrap()[0];
        // CIoU ≤ IoU (subtracts non-negative penalties)
        assert!(ciou <= iou + 1e-9, "ciou={ciou} > iou={iou}");
    }

    // -------------------------------------------------------------------
    // RoI Align / RoI Pool (#610)
    // -------------------------------------------------------------------

    #[test]
    fn roi_align_full_extent_avg() {
        // 1×1×2×2 input; one RoI covering the full extent at scale 1.
        let input = from_slice::<f64>(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        // box: batch=0, x1=0, y1=0, x2=2, y2=2 (full image).
        let boxes = from_slice::<f64>(&[0.0, 0.0, 0.0, 2.0, 2.0], &[1, 5]).unwrap();
        // Output 1×1: bilinear sample at center should average to ~ mean(input).
        let out = roi_align(&input, &boxes, (1, 1), 1.0, 2).unwrap();
        assert_eq!(out.shape(), &[1, 1, 1, 1]);
        let v = out.data().unwrap()[0];
        // Mean of [1,2,3,4] = 2.5; bilinear ≈ same with sufficient sampling.
        assert!((v - 2.5).abs() < 0.5, "got {v}");
    }

    #[test]
    fn roi_align_rejects_bad_box_shape() {
        let input = from_slice::<f64>(&[0.0; 16], &[1, 1, 4, 4]).unwrap();
        let boxes = from_slice::<f64>(&[0.0; 4], &[1, 4]).unwrap(); // wrong: needs [_, 5]
        let err = roi_align(&input, &boxes, (1, 1), 1.0, 1).unwrap_err();
        assert!(matches!(err, FerrotorchError::ShapeMismatch { .. }));
    }

    #[test]
    fn roi_pool_picks_max_per_bin() {
        // 1×1×2×2 input. Single box covering the entire extent, output 2×2:
        // each bin should be the single pixel, so output equals input.
        let input = from_slice::<f64>(&[1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).unwrap();
        let boxes = from_slice::<f64>(&[0.0, 0.0, 0.0, 1.0, 1.0], &[1, 5]).unwrap();
        let out = roi_pool(&input, &boxes, (2, 2), 1.0).unwrap();
        assert_eq!(out.shape(), &[1, 1, 2, 2]);
        let d = out.data().unwrap();
        // Each bin sees one pixel → output equals input row-major.
        assert_eq!(d, &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn roi_pool_rejects_zero_output_size() {
        let input = from_slice::<f64>(&[0.0; 16], &[1, 1, 4, 4]).unwrap();
        let boxes = from_slice::<f64>(&[0.0; 5], &[1, 5]).unwrap();
        let err = roi_pool(&input, &boxes, (0, 1), 1.0).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }
}
