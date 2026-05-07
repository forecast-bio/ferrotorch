//! Anchor generation for Region Proposal Networks.
//!
//! Mirrors `torchvision.models.detection.anchor_utils.AnchorGenerator`.
//!
//! For each feature-map level the generator tiles a set of anchor boxes
//! (one per (size × aspect_ratio) combination) across every spatial cell.
//! The returned boxes are in `[x1, y1, x2, y2]` pixel coords relative to
//! the **original image** — scaled by the level's stride.

use ferrotorch_core::{FerrotorchResult, Float, TensorStorage, Tensor};
use ferrotorch_core::numeric_cast::cast;

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
    /// half-integer centred at the origin).
    fn cell_anchors<T: Float>(cfg: &LevelConfig) -> FerrotorchResult<Vec<T>> {
        let zero: T = cast(0.0f64)?;
        let half: T = cast(0.5f64)?;
        let mut out = Vec::new();
        for &size in &cfg.sizes {
            for &ratio in &cfg.aspect_ratios {
                // Width and height of the anchor at unit stride.
                let area = size * size;
                let w = (area / ratio).sqrt();
                let h = w * ratio;
                let half_w: T = cast(w * 0.5)?;
                let half_h: T = cast(h * 0.5)?;
                let _ = (zero, half);
                out.push(-half_w); // x1
                out.push(-half_h); // y1
                out.push(half_w);  // x2
                out.push(half_h);  // y2
            }
        }
        Ok(out)
    }

    /// Generate anchors for **all levels** given the spatial sizes of each
    /// feature map.
    ///
    /// `feature_map_sizes[i]` is `(H_i, W_i)` for level `i`.
    ///
    /// Returns a flat `[N_total, 4]` tensor — the concatenation of anchors
    /// across all levels and all spatial positions. Mirrors
    /// `AnchorGenerator.forward` in torchvision.
    pub fn generate_anchors<T: Float>(
        &self,
        feature_map_sizes: &[(usize, usize)],
    ) -> FerrotorchResult<Tensor<T>> {
        assert_eq!(
            self.configs.len(),
            feature_map_sizes.len(),
            "AnchorGenerator: number of configs ({}) must match number of \
             feature-map levels ({})",
            self.configs.len(),
            feature_map_sizes.len()
        );

        let zero: T = cast(0.0f64)?;
        let mut all_anchors: Vec<T> = Vec::new();

        for (cfg, &(fh, fw)) in self.configs.iter().zip(feature_map_sizes.iter()) {
            let stride_t: T = cast(cfg.stride as f64)?;
            let base = Self::cell_anchors::<T>(cfg)?;
            let num_base = base.len() / 4; // A anchors per cell

            // Tile over (fy, fx) grid.
            for fy in 0..fh {
                for fx in 0..fw {
                    // Centre of cell in image coords.
                    let cx: T = (cast::<usize, T>(fx)? + cast::<f64, T>(0.5)?) * stride_t;
                    let cy: T = (cast::<usize, T>(fy)? + cast::<f64, T>(0.5)?) * stride_t;
                    let _ = zero;
                    for a in 0..num_base {
                        all_anchors.push(cx + base[a * 4]);     // x1
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
    let log_box_max: T = cast(4.6051702f64)?; // log(100) — clamp to prevent overflow

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
        let dw = clamp_t(d[i * 4 + 2] / ww, -log_box_max, log_box_max);
        let dh = clamp_t(d[i * 4 + 3] / wh, -log_box_max, log_box_max);

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
fn clamp_t<T: Float>(v: T, lo: T, hi: T) -> T {
    if v < lo { lo } else if v > hi { hi } else { v }
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
