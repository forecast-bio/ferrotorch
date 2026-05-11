//! Region Proposal Network (RPN).
//!
//! Mirrors `torchvision.models.detection.rpn.RegionProposalNetwork` and
//! `RPNHead`. Given FPN feature maps the RPN produces a set of candidate
//! proposals (bounding boxes + objectness scores) for the detection heads.
//!
//! ## Pipeline (per level)
//!
//! ```text
//! FPN level → 3×3 conv (256ch, same-pad) → relu
//!                    ↓                  ↓
//!           objectness conv (A×1 / per anchor)
//!           bbox-delta  conv (A×4 / per anchor)
//! ```
//!
//! After the per-level predictions are merged the top-K proposals are
//! selected (by objectness score), box-decoding + NMS are applied, and
//! the surviving proposals are returned as a flat `[N_proposals, 4]` tensor
//! in xyxy pixel coords.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::Conv2d;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;

use crate::models::detection::anchor_utils::{AnchorGenerator, decode_boxes};
use crate::ops::{batched_nms, clip_boxes_to_image, remove_small_boxes};

// ---------------------------------------------------------------------------
// RPN head
// ---------------------------------------------------------------------------

/// Shared 3×3 conv + per-anchor classification and regression heads.
///
/// Matches `torchvision.models.detection.rpn.RPNHead`.
pub struct RpnHead<T: Float> {
    /// 3×3 conv, same padding, 256ch → 256ch.
    conv: Conv2d<T>,
    /// 1×1 conv for objectness logits: 256ch → `num_anchors`.
    cls_logits: Conv2d<T>,
    /// 1×1 conv for bbox deltas: 256ch → `num_anchors * 4`.
    bbox_pred: Conv2d<T>,
}

impl<T: Float> RpnHead<T> {
    /// Create with `in_channels` and `num_anchors_per_cell`.
    pub fn new(in_channels: usize, num_anchors: usize) -> FerrotorchResult<Self> {
        let conv = Conv2d::new(in_channels, in_channels, (3, 3), (1, 1), (1, 1), true)?;
        let cls_logits = Conv2d::new(in_channels, num_anchors, (1, 1), (1, 1), (0, 0), true)?;
        let bbox_pred = Conv2d::new(in_channels, num_anchors * 4, (1, 1), (1, 1), (0, 0), true)?;
        Ok(Self {
            conv,
            cls_logits,
            bbox_pred,
        })
    }

    /// Forward through a single feature-map level.
    ///
    /// Returns `(objectness_logits, bbox_deltas)`:
    /// - `objectness_logits`: `[B, A, H, W]`
    /// - `bbox_deltas`: `[B, A*4, H, W]`
    pub fn forward_level(&self, x: &Tensor<T>) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        let h = self.conv.forward(x)?;
        let h = relu(&h)?;
        let logits = self.cls_logits.forward(&h)?;
        let deltas = self.bbox_pred.forward(&h)?;
        Ok((logits, deltas))
    }

    /// Collect trainable parameters.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv.parameters());
        p.extend(self.cls_logits.parameters());
        p.extend(self.bbox_pred.parameters());
        p
    }

    /// Mutable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv.parameters_mut());
        p.extend(self.cls_logits.parameters_mut());
        p.extend(self.bbox_pred.parameters_mut());
        p
    }

    /// Named parameters.
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.conv.named_parameters() {
            out.push((format!("conv.{n}"), p));
        }
        for (n, p) in self.cls_logits.named_parameters() {
            out.push((format!("cls_logits.{n}"), p));
        }
        for (n, p) in self.bbox_pred.named_parameters() {
            out.push((format!("bbox_pred.{n}"), p));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// RPN configuration
// ---------------------------------------------------------------------------

/// Run-time parameters for the RPN proposal selection.
///
/// Mirrors the defaults in `torchvision.models.detection.rpn.RegionProposalNetwork`.
#[derive(Debug, Clone)]
pub struct RpnConfig {
    /// Maximum number of proposals to keep *per level* before NMS (pre-NMS topK).
    pub pre_nms_top_n: usize,
    /// Maximum number of proposals to keep after NMS.
    pub post_nms_top_n: usize,
    /// IoU threshold for NMS suppression.
    pub nms_thresh: f64,
    /// Minimum box side length (in image pixels) to keep.
    pub min_size: f64,
    /// Objectness score threshold; boxes below this are dropped before NMS.
    pub score_thresh: f64,
    /// Image spatial size `[H, W]` — used for clipping boxes.
    pub image_size: [usize; 2],
}

impl RpnConfig {
    /// Defaults matching `torchvision.models.detection.rpn` for training on
    /// COCO (pre_nms_top_n=2000 train / 1000 test; post_nms_top_n=2000/1000).
    pub fn default_eval(image_size: [usize; 2]) -> Self {
        Self {
            pre_nms_top_n: 1000,
            post_nms_top_n: 1000,
            nms_thresh: 0.7,
            min_size: 1e-3,
            score_thresh: 0.0,
            image_size,
        }
    }
}

// ---------------------------------------------------------------------------
// RPN — proposal generation
// ---------------------------------------------------------------------------

/// Region Proposal Network.
///
/// Wraps `RpnHead` and `AnchorGenerator` and applies the full proposal
/// pipeline: objectness → box decode → clip → size filter → topK → NMS.
pub struct Rpn<T: Float> {
    pub head: RpnHead<T>,
    pub anchor_gen: AnchorGenerator,
}

impl<T: Float> Rpn<T> {
    /// Create with default Faster R-CNN anchor generator.
    pub fn new(in_channels: usize) -> FerrotorchResult<Self> {
        // 3 aspect ratios per location → 3 anchors/cell.
        let anchor_gen = AnchorGenerator::default_fasterrcnn();
        let head = RpnHead::new(in_channels, 3)?;
        Ok(Self { head, anchor_gen })
    }

    /// Forward pass over all FPN levels. Returns `[N_proposals, 4]` anchors
    /// (xyxy pixel coords, clipped to `cfg.image_size`).
    ///
    /// `fpn_features` — slice of tensors in level order (p2…p6), each `[B,C,H,W]`.
    /// Only `B==1` is required for the proposal-selection logic (per-image NMS).
    pub fn forward(
        &self,
        fpn_features: &[&Tensor<T>],
        cfg: &RpnConfig,
    ) -> FerrotorchResult<Tensor<T>> {
        let _num_levels = fpn_features.len();

        // Collect per-level spatial sizes for anchor generation.
        let fm_sizes: Vec<(usize, usize)> = fpn_features
            .iter()
            .map(|t| (t.shape()[2], t.shape()[3]))
            .collect();

        // Generate all anchors using torchvision-compatible per-dim strides
        // derived from the padded image size, not the canonical per-level
        // stride.  See `anchor_utils::generate_anchors_for_image` for the
        // #1141 round-4 rationale (non-64-aligned padded image sizes give
        // p6 stride `(image_h / 13, image_w / 17)` ≠ `(64, 64)`).
        let img_h = cfg.image_size[0];
        let img_w = cfg.image_size[1];
        let all_anchors: Tensor<T> = self
            .anchor_gen
            .generate_anchors_for_image(&fm_sizes, (img_h, img_w))?;
        let anc_data = all_anchors.data_vec()?;

        // Collect objectness scores and deltas across levels.
        let mut all_scores: Vec<f64> = Vec::new();
        let mut all_deltas: Vec<f64> = Vec::new();
        let mut level_offsets: Vec<usize> = vec![0]; // cumulative anchor counts

        for feat in fpn_features.iter() {
            let (logits, deltas) = self.head.forward_level(feat)?;
            // logits: [B, A, H, W] — we support B=1 for proposal generation.
            let logits_data = logits.data_vec()?;
            let deltas_data = deltas.data_vec()?;

            let b = logits.shape()[0];
            let a = logits.shape()[1];
            let h = logits.shape()[2];
            let w = logits.shape()[3];

            // Transpose [B,A,H,W] → [B,H,W,A] and flatten for image 0.
            let _ = b;
            for fh in 0..h {
                for fw in 0..w {
                    for ai in 0..a {
                        // logits index [0, ai, fh, fw].
                        let idx = ai * h * w + fh * w + fw;
                        let logit_val = logits_data[idx].to_f64().unwrap_or(0.0);
                        // sigmoid score.
                        let score = 1.0 / (1.0 + (-logit_val).exp());
                        all_scores.push(score);

                        // deltas [0, ai*4..(ai+1)*4, fh, fw].
                        for d in 0..4 {
                            let didx = (ai * 4 + d) * h * w + fh * w + fw;
                            all_deltas.push(deltas_data[didx].to_f64().unwrap_or(0.0));
                        }
                    }
                }
            }
            let cum = level_offsets.last().unwrap() + a * h * w;
            level_offsets.push(cum);
        }

        let n_total = all_scores.len();

        // ---- Pre-NMS top-K selection (per FPN level) ----
        //
        // Mirrors torchvision `RegionProposalNetwork._get_top_n_idx`: pick the
        // top `pre_nms_top_n` anchors **independently per level**, then
        // concatenate. This is critical — global top-K across levels would
        // disproportionately pick large-anchor levels and miss small objects.
        let mut order: Vec<usize> = Vec::new();
        let mut level_of: Vec<usize> = Vec::new();
        for lv in 0..level_offsets.len() - 1 {
            let start = level_offsets[lv];
            let end = level_offsets[lv + 1];
            let level_n = end - start;
            let mut idx: Vec<usize> = (start..end).collect();
            // Partial sort within this level by descending score.
            idx.sort_unstable_by(|&a, &b| {
                all_scores[b]
                    .partial_cmp(&all_scores[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            let take = cfg.pre_nms_top_n.min(level_n);
            for &i in idx.iter().take(take) {
                order.push(i);
                level_of.push(lv);
            }
        }
        let pre_n = order.len();
        let _ = n_total;

        // ---- Decode selected boxes ----
        let mut sel_anchors: Vec<f64> = Vec::with_capacity(pre_n * 4);
        let mut sel_deltas: Vec<f64> = Vec::with_capacity(pre_n * 4);
        let mut sel_scores: Vec<f64> = Vec::with_capacity(pre_n);
        for &i in &order {
            for k in 0..4 {
                sel_anchors.push(anc_data[i * 4 + k].to_f64().unwrap_or(0.0));
                sel_deltas.push(all_deltas[i * 4 + k]);
            }
            sel_scores.push(all_scores[i]);
        }

        // Build temporary f64 tensors for decode_boxes.
        let anc_t: Tensor<f64> = Tensor::from_storage(
            TensorStorage::cpu(sel_anchors.clone()),
            vec![pre_n, 4],
            false,
        )?;
        let del_t: Tensor<f64> =
            Tensor::from_storage(TensorStorage::cpu(sel_deltas), vec![pre_n, 4], false)?;
        let decoded_f64 = decode_boxes::<f64>(&anc_t, &del_t, (1.0, 1.0, 1.0, 1.0))?;

        // ---- Clip to image ----
        let clipped = clip_boxes_to_image(&decoded_f64, cfg.image_size)?;

        // ---- Remove small boxes ----
        let keep_small = remove_small_boxes(&clipped, cfg.min_size)?;

        // ---- Score threshold ----
        // torchvision uses `>=` for the score threshold (backwards-compat per
        // `filter_proposals` comment).
        let keep_thresh: Vec<usize> = keep_small
            .into_iter()
            .filter(|&i| sel_scores[i] >= cfg.score_thresh)
            .collect();

        if keep_thresh.is_empty() {
            // Return empty [0, 4] tensor.
            return Tensor::from_storage(TensorStorage::cpu(vec![]), vec![0, 4], false);
        }

        // ---- Build tensors for per-level NMS ----
        //
        // torchvision applies `batched_nms` keyed by FPN level so that
        // proposals on different levels never suppress each other — important
        // because each level has different scale characteristics.
        let nms_n = keep_thresh.len();
        let box_data = clipped.data_vec()?;
        let mut nms_boxes_data: Vec<f64> = Vec::with_capacity(nms_n * 4);
        let mut nms_scores_data: Vec<f64> = Vec::with_capacity(nms_n);
        let mut nms_levels: Vec<u32> = Vec::with_capacity(nms_n);
        for &i in &keep_thresh {
            for k in 0..4 {
                nms_boxes_data.push(box_data[i * 4 + k]);
            }
            nms_scores_data.push(sel_scores[i]);
            nms_levels.push(level_of[i] as u32);
        }

        let nms_boxes_t = Tensor::from_storage(
            TensorStorage::cpu(nms_boxes_data.clone()),
            vec![nms_n, 4],
            false,
        )?;
        let nms_scores_t =
            Tensor::from_storage(TensorStorage::cpu(nms_scores_data), vec![nms_n], false)?;

        let keep_nms = batched_nms::<f64>(&nms_boxes_t, &nms_scores_t, &nms_levels, cfg.nms_thresh)?;

        // ---- Post-NMS top-K ----
        let post_n = cfg.post_nms_top_n.min(keep_nms.len());
        let final_keep = &keep_nms[..post_n];

        // ---- Gather final boxes in T ----
        let mut final_boxes: Vec<T> = Vec::with_capacity(final_keep.len() * 4);
        for &i in final_keep {
            for k in 0..4 {
                final_boxes.push(cast::<f64, T>(nms_boxes_data[i * 4 + k])?);
            }
        }

        let nf = final_boxes.len() / 4;
        Tensor::from_storage(TensorStorage::cpu(final_boxes), vec![nf, 4], false)
    }

    /// Trainable parameters.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        self.head.parameters()
    }

    /// Mutable parameters.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.head.parameters_mut()
    }

    /// Named parameters (prefixed `"head."`).
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.head
            .named_parameters()
            .into_iter()
            .map(|(n, p)| (format!("head.{n}"), p))
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::randn;

    #[test]
    fn test_rpn_head_output_shapes() {
        let head = RpnHead::<f32>::new(256, 3).unwrap();
        let x = randn(&[1, 256, 8, 8]).unwrap();
        let (logits, deltas) = head.forward_level(&x).unwrap();
        assert_eq!(logits.shape(), &[1, 3, 8, 8]);
        assert_eq!(deltas.shape(), &[1, 12, 8, 8]);
    }

    #[test]
    fn test_rpn_forward_returns_proposals() {
        let rpn = Rpn::<f32>::new(256).unwrap();
        // Five tiny FPN levels: p2..p6.
        let p2 = randn(&[1, 256, 4, 4]).unwrap();
        let p3 = randn(&[1, 256, 2, 2]).unwrap();
        let p4 = randn(&[1, 256, 1, 1]).unwrap();
        let p5 = randn(&[1, 256, 1, 1]).unwrap();
        let p6 = randn(&[1, 256, 1, 1]).unwrap();

        let cfg = RpnConfig::default_eval([64, 64]);
        let proposals = rpn.forward(&[&p2, &p3, &p4, &p5, &p6], &cfg).unwrap();

        // Should produce some proposals; shape is [N, 4].
        assert_eq!(proposals.shape().len(), 2);
        assert_eq!(proposals.shape()[1], 4);
    }

    #[test]
    fn test_rpn_proposals_within_image_bounds() {
        let rpn = Rpn::<f32>::new(256).unwrap();
        let p2 = randn(&[1, 256, 4, 4]).unwrap();
        let p3 = randn(&[1, 256, 2, 2]).unwrap();
        let p4 = randn(&[1, 256, 1, 1]).unwrap();
        let p5 = randn(&[1, 256, 1, 1]).unwrap();
        let p6 = randn(&[1, 256, 1, 1]).unwrap();

        let image_size = [128, 128];
        let cfg = RpnConfig::default_eval(image_size);
        let proposals = rpn.forward(&[&p2, &p3, &p4, &p5, &p6], &cfg).unwrap();

        if proposals.shape()[0] > 0 {
            let data = proposals.data_vec().unwrap();
            let n = proposals.shape()[0];
            for i in 0..n {
                let x1 = data[i * 4];
                let y1 = data[i * 4 + 1];
                let x2 = data[i * 4 + 2];
                let y2 = data[i * 4 + 3];
                assert!(x1 >= 0.0 && x2 <= 128.0, "x out of bounds: {x1}..{x2}");
                assert!(y1 >= 0.0 && y2 <= 128.0, "y out of bounds: {y1}..{y2}");
            }
        }
    }
}
