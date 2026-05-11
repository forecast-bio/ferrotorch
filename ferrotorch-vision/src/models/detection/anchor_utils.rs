//! Anchor generation for Region Proposal Networks.
//!
//! Mirrors `torchvision.models.detection.anchor_utils.AnchorGenerator`.
//!
//! For each feature-map level the generator tiles a set of anchor boxes
//! (one per (size × aspect_ratio) combination) across every spatial cell.
//! The returned boxes are in `[x1, y1, x2, y2]` pixel coords relative to
//! the **original image** — scaled by the level's stride.

use ferrotorch_core::numeric_cast::cast;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage};

// ---------------------------------------------------------------------------
// AnchorGenerator
// ---------------------------------------------------------------------------

/// Configuration for a single FPN level.
#[derive(Debug, Clone)]
pub struct LevelConfig {
    /// Anchor sizes (in pixels at the original image scale) for this level.
    pub sizes: Vec<f64>,
    /// Aspect ratios (h/w) to tile for each size.
    pub aspect_ratios: Vec<f64>,
    /// Stride of this feature-map level relative to the input image.
    pub stride: usize,
}

/// Generates multi-scale anchors matching `torchvision.models.detection.anchor_utils.AnchorGenerator`.
///
/// Anchors are centred on every spatial cell of the feature map and emitted in
/// `[x1, y1, x2, y2]` format (pixel coords on the original image).
pub struct AnchorGenerator {
    configs: Vec<LevelConfig>,
}

impl AnchorGenerator {
    /// Create a new generator from per-level configurations.
    pub fn new(configs: Vec<LevelConfig>) -> Self {
        Self { configs }
    }

    /// Default configuration for `fasterrcnn_resnet50_fpn`:
    /// five FPN levels (P2–P6), matching `torchvision` defaults.
    ///
    /// Sizes: `(32, 64, 128, 256, 512)`, one per level.
    /// Aspect ratios: `(0.5, 1.0, 2.0)` for every level.
    /// Strides: `(4, 8, 16, 32, 64)`.
    pub fn default_fasterrcnn() -> Self {
        let sizes = [32.0_f64, 64.0, 128.0, 256.0, 512.0];
        let strides = [4usize, 8, 16, 32, 64];
        let aspect_ratios = vec![0.5, 1.0, 2.0];
        let configs = sizes
            .iter()
            .zip(strides.iter())
            .map(|(&s, &stride)| LevelConfig {
                sizes: vec![s],
                aspect_ratios: aspect_ratios.clone(),
                stride,
            })
            .collect();
        Self::new(configs)
    }

    /// Compute all base anchor templates (relative to a single cell centre)
    /// for a given `LevelConfig`. Returns a `[A, 4]` slice (x1 y1 x2 y2,
    /// centred at the origin).
    ///
    /// Mirrors `torchvision.models.detection.anchor_utils.AnchorGenerator.generate_anchors`:
    /// for each `(scale, ratio)` pair, `w = scale / sqrt(ratio)`, `h = scale * sqrt(ratio)`,
    /// and the half-extents are **rounded** (`.round()`) before negation so the
    /// resulting anchors are integer-aligned to match torchvision exactly. The
    /// outer loop order also follows torchvision's `aspect_ratios × scales` (in
    /// that order — each aspect ratio is paired with every scale).
    fn cell_anchors<T: Float>(cfg: &LevelConfig) -> FerrotorchResult<Vec<T>> {
        let mut out = Vec::new();
        // torchvision's `(w_ratios[:, None] * scales[None, :]).view(-1)`
        // iterates aspect_ratios in the outer dimension and scales in the
        // inner. With one size per level (the FasterRCNN default) this is
        // simply 3 anchors per aspect ratio, all sharing the same scale.
        for &ratio in &cfg.aspect_ratios {
            let sqrt_r = ratio.sqrt();
            for &size in &cfg.sizes {
                let w = size / sqrt_r;
                let h = size * sqrt_r;
                let half_w: T = cast((w * 0.5).round())?;
                let half_h: T = cast((h * 0.5).round())?;
                out.push(cast::<f64, T>(0.0)? - half_w); // x1 = -half_w
                out.push(cast::<f64, T>(0.0)? - half_h); // y1 = -half_h
                out.push(half_w); // x2
                out.push(half_h); // y2
            }
        }
        Ok(out)
    }

    /// Generate anchors for **all levels** given the spatial sizes of each
    /// feature map, using `cfg.stride` as a square stride (for both H and
    /// W).  Useful for unit tests that don't care about the exact strides
    /// torchvision would compute for the image size.
    ///
    /// **Production callers should prefer `generate_anchors_for_image`** —
    /// torchvision derives strides per dimension as
    /// `(image_size[i] // grid_size[i])`, which is *not* a perfect match
    /// for the canonical `[4,8,16,32,64]` strides when the padded image
    /// size isn't a multiple of 64 (commonly the case at p6).
    ///
    /// `feature_map_sizes[i]` is `(H_i, W_i)` for level `i`.
    ///
    /// Returns a flat `[N_total, 4]` tensor — the concatenation of anchors
    /// across all levels and all spatial positions.
    pub fn generate_anchors<T: Float>(
        &self,
        feature_map_sizes: &[(usize, usize)],
    ) -> FerrotorchResult<Tensor<T>> {
        let strides: Vec<(usize, usize)> = self
            .configs
            .iter()
            .map(|c| (c.stride, c.stride))
            .collect();
        self.generate_anchors_with_strides(feature_map_sizes, &strides)
    }

    /// Generate anchors using torchvision-compatible per-dimension strides
    /// derived from the input image size:
    ///
    /// ```text
    /// stride_h = image_size.0 / grid_size.0
    /// stride_w = image_size.1 / grid_size.1
    /// ```
    ///
    /// This matches `AnchorGenerator.forward` in `torchvision.models.detection`,
    /// where the strides are recomputed from the actual padded input shape
    /// each call rather than read from the per-level config.  The
    /// difference matters at p6 for non-64-aligned padded image sizes
    /// (e.g. an 800×1088 input → p6 grid 13×17 → tv stride `(61, 64)` vs
    /// the canonical 64×64). This was the #1141 round-4 diagnosis for the
    /// remaining post-FPN-bias-fix divergence in proposal count.
    pub fn generate_anchors_for_image<T: Float>(
        &self,
        feature_map_sizes: &[(usize, usize)],
        image_size: (usize, usize),
    ) -> FerrotorchResult<Tensor<T>> {
        let strides: Vec<(usize, usize)> = feature_map_sizes
            .iter()
            .map(|&(fh, fw)| {
                let sh = image_size.0.checked_div(fh).unwrap_or(1);
                let sw = image_size.1.checked_div(fw).unwrap_or(1);
                (sh, sw)
            })
            .collect();
        self.generate_anchors_with_strides(feature_map_sizes, &strides)
    }

    /// Shared core for `generate_anchors` and `generate_anchors_for_image`.
    /// `strides[i] == (stride_h, stride_w)` for level `i`.
    fn generate_anchors_with_strides<T: Float>(
        &self,
        feature_map_sizes: &[(usize, usize)],
        strides: &[(usize, usize)],
    ) -> FerrotorchResult<Tensor<T>> {
        assert_eq!(
            self.configs.len(),
            feature_map_sizes.len(),
            "AnchorGenerator: number of configs ({}) must match number of \
             feature-map levels ({})",
            self.configs.len(),
            feature_map_sizes.len()
        );
        assert_eq!(
            self.configs.len(),
            strides.len(),
            "AnchorGenerator: number of strides ({}) must match number of \
             levels ({})",
            strides.len(),
            self.configs.len()
        );

        let zero: T = cast(0.0f64)?;
        let mut all_anchors: Vec<T> = Vec::new();

        for (cfg, (&(fh, fw), &(sh, sw))) in self
            .configs
            .iter()
            .zip(feature_map_sizes.iter().zip(strides.iter()))
        {
            let stride_h_t: T = cast(sh as f64)?;
            let stride_w_t: T = cast(sw as f64)?;
            let base = Self::cell_anchors::<T>(cfg)?;
            let num_base = base.len() / 4; // A anchors per cell

            // Tile over (fy, fx) grid.
            //
            // Matches torchvision `grid_anchors`:
            //   shifts_x = arange(0, grid_width) * stride_width
            //   shifts_y = arange(0, grid_height) * stride_height
            // (i.e. the cell **corner** is the centre, not `(fx + 0.5) * stride`).
            for fy in 0..fh {
                for fx in 0..fw {
                    let cx: T = cast::<usize, T>(fx)? * stride_w_t;
                    let cy: T = cast::<usize, T>(fy)? * stride_h_t;
                    let _ = zero;
                    for a in 0..num_base {
                        all_anchors.push(cx + base[a * 4]); // x1
                        all_anchors.push(cy + base[a * 4 + 1]); // y1
                        all_anchors.push(cx + base[a * 4 + 2]); // x2
                        all_anchors.push(cy + base[a * 4 + 3]); // y2
                    }
                }
            }
        }

        let n = all_anchors.len() / 4;
        Tensor::from_storage(TensorStorage::cpu(all_anchors), vec![n, 4], false)
    }

    /// Number of anchors per spatial position for level `i`.
    pub fn num_anchors_per_location(&self, level: usize) -> usize {
        let cfg = &self.configs[level];
        cfg.sizes.len() * cfg.aspect_ratios.len()
    }
}

// ---------------------------------------------------------------------------
// Box encoding / decoding
// ---------------------------------------------------------------------------

/// Decode RPN regression deltas `(dx, dy, dw, dh)` applied to anchors
/// into predicted boxes `[x1, y1, x2, y2]`.
///
/// Mirrors `torchvision.models.detection.box_coder.BoxCoder.decode_single`.
///
/// `anchors`: `[N, 4]` in xyxy format.
/// `deltas`: `[N, 4]` — (dx, dy, dw, dh).
/// `weights`: `(wx, wy, ww, wh)` — scaling factors (default: 1.0).
/// Returns `[N, 4]` predicted boxes in xyxy format.
pub fn decode_boxes<T: Float>(
    anchors: &Tensor<T>,
    deltas: &Tensor<T>,
    weights: (f64, f64, f64, f64),
) -> FerrotorchResult<Tensor<T>> {
    let n = anchors.shape()[0];
    let a = anchors.data_vec()?;
    let d = deltas.data_vec()?;

    let wx: T = cast(weights.0)?;
    let wy: T = cast(weights.1)?;
    let ww: T = cast(weights.2)?;
    let wh: T = cast(weights.3)?;
    let two: T = cast(2.0f64)?;
    // torchvision `_utils.BoxCoder.bbox_xform_clip = math.log(1000.0 / 16.0)`.
    // Applied as a one-sided `max=` clamp on `dw`/`dh` — *not* symmetric.
    let log_box_max: T = cast(4.135_166_556_742_356f64)?;

    let mut out = vec![cast::<f64, T>(0.0)?; n * 4];
    for i in 0..n {
        let x1 = a[i * 4];
        let y1 = a[i * 4 + 1];
        let x2 = a[i * 4 + 2];
        let y2 = a[i * 4 + 3];

        let w = x2 - x1;
        let h = y2 - y1;
        let cx = x1 + w / two;
        let cy = y1 + h / two;

        let dx = d[i * 4] / wx;
        let dy = d[i * 4 + 1] / wy;
        let mut dw = d[i * 4 + 2] / ww;
        let mut dh = d[i * 4 + 3] / wh;
        if dw > log_box_max {
            dw = log_box_max;
        }
        if dh > log_box_max {
            dh = log_box_max;
        }

        let px = dx * w + cx;
        let py = dy * h + cy;
        let pw = exp_t(dw) * w;
        let ph = exp_t(dh) * h;

        out[i * 4] = px - pw / two;
        out[i * 4 + 1] = py - ph / two;
        out[i * 4 + 2] = px + pw / two;
        out[i * 4 + 3] = py + ph / two;
    }

    Tensor::from_storage(TensorStorage::cpu(out), vec![n, 4], false)
}

#[inline]
fn exp_t<T: Float>(v: T) -> T {
    let vf = v.to_f64().unwrap_or(0.0);
    let r = vf.exp();
    T::from(r).unwrap_or_else(|| T::from(1.0f64).unwrap())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_anchor_generator_count() {
        // Default fasterrcnn: 5 levels, each 3 anchors/cell.
        let anchor_gen = AnchorGenerator::default_fasterrcnn();
        // Small feature maps: 2x2 per level.
        let fm_sizes = vec![(2, 2); 5];
        let anchors = anchor_gen.generate_anchors::<f32>(&fm_sizes).unwrap();
        // 5 levels x 4 cells x 3 anchors = 60 anchors.
        assert_eq!(anchors.shape(), &[60, 4]);
    }

    #[test]
    fn test_anchor_generator_box_format() {
        let anchor_gen = AnchorGenerator::default_fasterrcnn();
        let anchors = anchor_gen.generate_anchors::<f32>(&[(1, 1); 5]).unwrap();
        let data = anchors.data_vec().unwrap();
        // Every anchor: x1 < x2 and y1 < y2.
        let n = anchors.shape()[0];
        for i in 0..n {
            let x1 = data[i * 4];
            let y1 = data[i * 4 + 1];
            let x2 = data[i * 4 + 2];
            let y2 = data[i * 4 + 3];
            assert!(x1 < x2, "anchor {i}: x1={x1} >= x2={x2}");
            assert!(y1 < y2, "anchor {i}: y1={y1} >= y2={y2}");
        }
    }

    #[test]
    fn test_anchor_generator_per_dim_stride_image_800x1088() {
        // #1141: torchvision derives per-dim strides as
        // `image_size[i] // grid_size[i]`. For a padded 800×1088 input
        // the p6 grid is 13×17 → strides (61, 64) (NOT 64, 64). The
        // canonical `generate_anchors([... fm sizes ...])` uses 64 for
        // both dims and was the residual #1141 anchor divergence after
        // the FPN fix.
        let ag = AnchorGenerator::default_fasterrcnn();
        let fm_sizes = vec![(200, 272), (100, 136), (50, 68), (25, 34), (13, 17)];
        let image_size = (800usize, 1088usize);

        // Sanity check: the p6 stride should be (61, 64), not (64, 64).
        // p6 grid (13, 17) → stride_h = 800 / 13 = 61, stride_w = 1088 / 17 = 64.
        let anchors_for_image = ag
            .generate_anchors_for_image::<f32>(&fm_sizes, image_size)
            .unwrap();
        let anchors_canonical = ag.generate_anchors::<f32>(&fm_sizes).unwrap();

        // Count of p6 anchors per level (3 anchors per cell × 13 × 17 = 663).
        let p6_count = 13 * 17 * 3;
        // Find the offset of p6 in the flat tensor.
        let pre_p6: usize = fm_sizes
            .iter()
            .take(4)
            .map(|&(h, w)| h * w * 3)
            .sum();

        let a = anchors_for_image.data_vec().unwrap();
        let b = anchors_canonical.data_vec().unwrap();
        // Verify p6 anchors differ between the canonical and image-derived
        // call (this is the bug-reproducing assertion).
        let mut max_diff = 0.0f32;
        for i in 0..p6_count {
            for k in 0..4 {
                let idx = (pre_p6 + i) * 4 + k;
                let d = (a[idx] - b[idx]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
        assert!(
            max_diff > 1.0,
            "p6 anchors must differ between canonical and per-dim-stride \
             generation on an 800×1088 image (regression on stride bug); \
             max_diff={max_diff}"
        );

        // Verify the bottom-row p6 anchors land at the expected y center.
        // Last row of p6 (fy=12) with stride_h=61: cy = 12 * 61 = 732.
        // Aspect ratio 1.0, size 512 → base anchor (-256, -256, 256, 256).
        // So last-row first-col first-anchor row 1 box: [-23-? wait wrong]
        // Simpler: in the image-derived case, the LARGEST y coord seen
        // among p6 anchors is at fy=12 (cy=732) plus the largest half_h.
        // Half-h for (size=512, ratio=2.0): h=512*sqrt(2)≈724, half=362
        //   so max y2 = 732 + 362 = 1094.
        // In the canonical case (stride 64): cy = 12*64 = 768 → 768 + 362 = 1130.
        let mut max_y2_image = f32::MIN;
        let mut max_y2_canon = f32::MIN;
        for i in 0..p6_count {
            let y2_img = a[(pre_p6 + i) * 4 + 3];
            let y2_can = b[(pre_p6 + i) * 4 + 3];
            if y2_img > max_y2_image {
                max_y2_image = y2_img;
            }
            if y2_can > max_y2_canon {
                max_y2_canon = y2_can;
            }
        }
        assert!(
            max_y2_image < max_y2_canon - 10.0,
            "image-derived p6 anchors must have smaller max y2 (stride 61 \
             vs canonical 64): image={max_y2_image}, canonical={max_y2_canon}",
        );
    }

    #[test]
    fn test_decode_boxes_identity_delta_zero() {
        // With zero deltas and unit weights, decoded boxes should match anchors.
        use ferrotorch_core::from_slice;
        let anchors = from_slice::<f32>(&[10.0, 20.0, 50.0, 60.0], &[1, 4]).unwrap();
        let deltas = from_slice::<f32>(&[0.0, 0.0, 0.0, 0.0], &[1, 4]).unwrap();
        let pred = decode_boxes(&anchors, &deltas, (1.0, 1.0, 1.0, 1.0)).unwrap();
        let d = pred.data_vec().unwrap();
        // Encoded w=40, h=40, cx=30, cy=40.
        // dx=0 → px=30, dw=0 → pw=40. x1=30-20=10 ✓.
        assert!((d[0] - 10.0).abs() < 1e-4, "x1={}", d[0]);
        assert!((d[1] - 20.0).abs() < 1e-4, "y1={}", d[1]);
        assert!((d[2] - 50.0).abs() < 1e-4, "x2={}", d[2]);
        assert!((d[3] - 60.0).abs() < 1e-4, "y2={}", d[3]);
    }
}
