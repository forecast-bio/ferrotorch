//! Feature Pyramid Network (FPN) for multi-scale feature aggregation.
//!
//! Mirrors `torchvision.ops.FeaturePyramidNetwork` / the FPN used inside
//! `fasterrcnn_resnet50_fpn`. Given a dict of backbone feature maps
//! (C2–C5 from ResNet-50), the FPN produces five output levels (P2–P6)
//! via top-down lateral connections.
//!
//! Architecture (matching torchvision `BackboneWithFPN` defaults):
//!
//! ```text
//! C5 (2048ch) → lateral_c5 (256ch) → P5 ─────────────────→ output["p5"]
//!                                        ↓ upsample 2×
//! C4 (1024ch) → lateral_c4 (256ch) + P4 ──────────────────→ output["p4"]
//!                                        ↓ upsample 2×
//! C3 ( 512ch) → lateral_c3 (256ch) + P3 ──────────────────→ output["p3"]
//!                                        ↓ upsample 2×
//! C2 ( 256ch) → lateral_c2 (256ch) + P2 ──────────────────→ output["p2"]
//! P5 → 1×1 maxpool, stride 2 ──────────────────────────────→ output["p6"]
//! ```
//!
//! All lateral/output convolutions use `out_channels=256` and apply a
//! 1×1 lateral conv followed by a 3×3 output conv — matching the reference.
//! Both lateral and output convs include a bias (matching torchvision's
//! `Conv2dNormActivation(..., norm_layer=None)` which leaves Conv2d at
//! `bias=True`).  The P6 path is torchvision's `LastLevelMaxPool`, which
//! is a *pure stride-2 sub-sample* (`kernel=1, stride=2, padding=0`),
//! NOT a true 3×3 maxpool — see #1141 for the parity probe.

use std::collections::HashMap;

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::MaxPool2d;
use ferrotorch_nn::{Conv2d, InterpolateMode, interpolate};

/// Number of output channels for every FPN level.
pub const FPN_OUT_CHANNELS: usize = 256;

// ---------------------------------------------------------------------------
// FeaturePyramidNetwork
// ---------------------------------------------------------------------------

/// FPN over four backbone levels (C2–C5).
///
/// Input: a `HashMap<&str, Tensor<T>>` with keys `"layer1"`, `"layer2"`,
///        `"layer3"`, `"layer4"` (the ResNet intermediate outputs).
///
/// Output: a `HashMap<String, Tensor<T>>` with keys
///         `"p2"`, `"p3"`, `"p4"`, `"p5"`, `"p6"`.
pub struct FeaturePyramidNetwork<T: Float> {
    // Lateral 1×1 convolutions (one per backbone level).
    lateral2: Conv2d<T>,
    lateral3: Conv2d<T>,
    lateral4: Conv2d<T>,
    lateral5: Conv2d<T>,

    // 3×3 output convolutions.
    output2: Conv2d<T>,
    output3: Conv2d<T>,
    output4: Conv2d<T>,
    output5: Conv2d<T>,

    // P6 is a 3×3 max-pool on P5 with stride 2.
    pool_p6: MaxPool2d,
}

impl<T: Float> FeaturePyramidNetwork<T> {
    /// Build a default FPN for ResNet-50.
    ///
    /// Channel counts `[256, 512, 1024, 2048]` correspond to
    /// `layer1..layer4` in `ResNet::forward_features`.
    pub fn new() -> FerrotorchResult<Self> {
        let in_channels = [256, 512, 1024, 2048];
        let out_ch = FPN_OUT_CHANNELS;

        // bias=true matches torchvision: `Conv2dNormActivation(in,out, kernel_size=K,
        // padding=P, norm_layer=None, activation_layer=None)` produces a plain
        // `nn.Conv2d` whose default `bias=True`. ferrotorch's previous bias=false
        // here was the root cause of #1141 (lateral + output biases were dropped
        // by `strict=false` `load_state_dict`, breaking FPN parity end-to-end).
        let lateral2 = Conv2d::new(in_channels[0], out_ch, (1, 1), (1, 1), (0, 0), true)?;
        let lateral3 = Conv2d::new(in_channels[1], out_ch, (1, 1), (1, 1), (0, 0), true)?;
        let lateral4 = Conv2d::new(in_channels[2], out_ch, (1, 1), (1, 1), (0, 0), true)?;
        let lateral5 = Conv2d::new(in_channels[3], out_ch, (1, 1), (1, 1), (0, 0), true)?;

        let output2 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), true)?;
        let output3 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), true)?;
        let output4 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), true)?;
        let output5 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), true)?;

        // P6: `LastLevelMaxPool` in torchvision uses `kernel=1, stride=2,
        // padding=0` — pure stride-2 sub-sampling, NOT a 3×3 maxpool.
        // The previous 3×3-pool config gave a noticeably larger p6 output
        // magnitude (~3.4 max-abs-diff vs torchvision) — see #1141 probe.
        let pool_p6 = MaxPool2d::new([1, 1], [2, 2], [0, 0]);

        Ok(Self {
            lateral2,
            lateral3,
            lateral4,
            lateral5,
            output2,
            output3,
            output4,
            output5,
            pool_p6,
        })
    }

    /// Forward pass.
    ///
    /// `backbone_features` must contain keys `"layer1"`, `"layer2"`,
    /// `"layer3"`, `"layer4"` produced by `ResNet::forward_features`.
    ///
    /// Returns `{"p2", "p3", "p4", "p5", "p6"}`.
    pub fn forward(
        &self,
        backbone_features: &HashMap<String, Tensor<T>>,
    ) -> FerrotorchResult<HashMap<String, Tensor<T>>> {
        let c2 = backbone_features.get("layer1").ok_or_else(|| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: "FPN: backbone_features missing 'layer1' (C2)".into(),
            }
        })?;
        let c3 = backbone_features.get("layer2").ok_or_else(|| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: "FPN: backbone_features missing 'layer2' (C3)".into(),
            }
        })?;
        let c4 = backbone_features.get("layer3").ok_or_else(|| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: "FPN: backbone_features missing 'layer3' (C4)".into(),
            }
        })?;
        let c5 = backbone_features.get("layer4").ok_or_else(|| {
            ferrotorch_core::FerrotorchError::InvalidArgument {
                message: "FPN: backbone_features missing 'layer4' (C5)".into(),
            }
        })?;

        // --- Lateral convolutions ---
        let lat5 = self.lateral5.forward(c5)?;
        let lat4 = self.lateral4.forward(c4)?;
        let lat3 = self.lateral3.forward(c3)?;
        let lat2 = self.lateral2.forward(c2)?;

        // --- Top-down pathway ---
        // P5 is just lat5 (no higher level).
        let p5_inner = lat5;

        // P4: upsample P5 to C4's spatial size, add lateral4.
        let c4_shape = c4.shape();
        let p5_up = interpolate(
            &p5_inner,
            Some([c4_shape[2], c4_shape[3]]),
            None,
            InterpolateMode::Nearest,
            false,
        )?;
        let p4_inner = add(&p5_up, &lat4)?;

        // P3: upsample P4 to C3's spatial size, add lateral3.
        let c3_shape = c3.shape();
        let p4_up = interpolate(
            &p4_inner,
            Some([c3_shape[2], c3_shape[3]]),
            None,
            InterpolateMode::Nearest,
            false,
        )?;
        let p3_inner = add(&p4_up, &lat3)?;

        // P2: upsample P3 to C2's spatial size, add lateral2.
        let c2_shape = c2.shape();
        let p3_up = interpolate(
            &p3_inner,
            Some([c2_shape[2], c2_shape[3]]),
            None,
            InterpolateMode::Nearest,
            false,
        )?;
        let p2_inner = add(&p3_up, &lat2)?;

        // --- 3×3 output convolutions ---
        let p2 = self.output2.forward(&p2_inner)?;
        let p3 = self.output3.forward(&p3_inner)?;
        let p4 = self.output4.forward(&p4_inner)?;
        let p5 = self.output5.forward(&p5_inner)?;

        // P6: 3×3 maxpool on P5 with stride 2.
        let p6 = Module::<T>::forward(&self.pool_p6, &p5)?;

        let mut out = HashMap::new();
        out.insert("p2".to_string(), p2);
        out.insert("p3".to_string(), p3);
        out.insert("p4".to_string(), p4);
        out.insert("p5".to_string(), p5);
        out.insert("p6".to_string(), p6);
        Ok(out)
    }

    /// Collect all trainable parameters.
    pub fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.lateral2.parameters());
        params.extend(self.lateral3.parameters());
        params.extend(self.lateral4.parameters());
        params.extend(self.lateral5.parameters());
        params.extend(self.output2.parameters());
        params.extend(self.output3.parameters());
        params.extend(self.output4.parameters());
        params.extend(self.output5.parameters());
        params
    }

    /// Collect mutable parameter references.
    pub fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.lateral2.parameters_mut());
        params.extend(self.lateral3.parameters_mut());
        params.extend(self.lateral4.parameters_mut());
        params.extend(self.lateral5.parameters_mut());
        params.extend(self.output2.parameters_mut());
        params.extend(self.output3.parameters_mut());
        params.extend(self.output4.parameters_mut());
        params.extend(self.output5.parameters_mut());
        params
    }

    /// Named parameters (prefixed).
    pub fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (n, p) in self.lateral2.named_parameters() {
            out.push((format!("lateral2.{n}"), p));
        }
        for (n, p) in self.lateral3.named_parameters() {
            out.push((format!("lateral3.{n}"), p));
        }
        for (n, p) in self.lateral4.named_parameters() {
            out.push((format!("lateral4.{n}"), p));
        }
        for (n, p) in self.lateral5.named_parameters() {
            out.push((format!("lateral5.{n}"), p));
        }
        for (n, p) in self.output2.named_parameters() {
            out.push((format!("output2.{n}"), p));
        }
        for (n, p) in self.output3.named_parameters() {
            out.push((format!("output3.{n}"), p));
        }
        for (n, p) in self.output4.named_parameters() {
            out.push((format!("output4.{n}"), p));
        }
        for (n, p) in self.output5.named_parameters() {
            out.push((format!("output5.{n}"), p));
        }
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::randn;

    /// Build a minimal backbone-features map mimicking ResNet-50 output at a
    /// 64×64 input (stride: /4, /8, /16, /32 → 16, 8, 4, 2).
    fn fake_backbone_features(batch: usize) -> HashMap<String, Tensor<f32>> {
        let mut m = HashMap::new();
        m.insert("layer1".into(), randn(&[batch, 256, 16, 16]).unwrap());
        m.insert("layer2".into(), randn(&[batch, 512, 8, 8]).unwrap());
        m.insert("layer3".into(), randn(&[batch, 1024, 4, 4]).unwrap());
        m.insert("layer4".into(), randn(&[batch, 2048, 2, 2]).unwrap());
        m
    }

    #[test]
    fn test_fpn_output_keys() {
        let fpn = FeaturePyramidNetwork::<f32>::new().unwrap();
        let features = fake_backbone_features(1);
        let out = fpn.forward(&features).unwrap();
        assert!(out.contains_key("p2"), "missing p2");
        assert!(out.contains_key("p3"), "missing p3");
        assert!(out.contains_key("p4"), "missing p4");
        assert!(out.contains_key("p5"), "missing p5");
        assert!(out.contains_key("p6"), "missing p6");
    }

    #[test]
    fn test_fpn_output_channels() {
        let fpn = FeaturePyramidNetwork::<f32>::new().unwrap();
        let features = fake_backbone_features(1);
        let out = fpn.forward(&features).unwrap();
        for key in ["p2", "p3", "p4", "p5", "p6"] {
            let t = &out[key];
            assert_eq!(
                t.shape()[1],
                FPN_OUT_CHANNELS,
                "{key} should have {} channels, got {}",
                FPN_OUT_CHANNELS,
                t.shape()[1]
            );
        }
    }

    #[test]
    fn test_fpn_spatial_sizes_batch1() {
        // For a 64-pixel input, backbone strides: /4, /8, /16, /32.
        // layer1: 16×16, layer2: 8×8, layer3: 4×4, layer4: 2×2.
        // FPN p2=16×16, p3=8×8, p4=4×4, p5=2×2, p6=maxpool(p5)=1×1.
        let fpn = FeaturePyramidNetwork::<f32>::new().unwrap();
        let features = fake_backbone_features(1);
        let out = fpn.forward(&features).unwrap();

        assert_eq!(out["p2"].shape(), &[1, 256, 16, 16]);
        assert_eq!(out["p3"].shape(), &[1, 256, 8, 8]);
        assert_eq!(out["p4"].shape(), &[1, 256, 4, 4]);
        assert_eq!(out["p5"].shape(), &[1, 256, 2, 2]);
        assert_eq!(out["p6"].shape(), &[1, 256, 1, 1]);
    }

    #[test]
    fn test_fpn_named_params_include_biases() {
        // #1141: lateral + output convs MUST have biases (matching
        // torchvision's `Conv2dNormActivation(..., norm_layer=None)` which
        // keeps `nn.Conv2d(..., bias=True)`). The bias-less variant was the
        // root cause of #1141 (924/1000 post-NMS proposal mismatch on
        // image 87038); this test prevents regression.
        let fpn = FeaturePyramidNetwork::<f32>::new().unwrap();
        let names: Vec<String> = fpn
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        for k in [
            "lateral2.bias",
            "lateral3.bias",
            "lateral4.bias",
            "lateral5.bias",
            "output2.bias",
            "output3.bias",
            "output4.bias",
            "output5.bias",
        ] {
            assert!(
                names.iter().any(|n| n == k),
                "FPN missing bias param '{k}'; full list: {names:?}"
            );
        }
    }

    #[test]
    fn test_fpn_p6_uses_stride_2_subsample_not_3x3_pool() {
        // #1141: torchvision's `LastLevelMaxPool` is a kernel=1,
        // stride=2, padding=0 sub-sample, NOT a 3×3 maxpool. With
        // `kernel=1`, P6 == P5 sub-sampled at strides (0, 2, 4...), so
        // `out["p6"][:,:,i,j] == out["p5"][:,:,2i,2j]` exactly (no max).
        // The previous 3×3-pool config gave a noticeably larger p6 output
        // magnitude (~3.4 max-abs-diff vs torchvision on a real image).
        let fpn = FeaturePyramidNetwork::<f32>::new().unwrap();
        let mut features = HashMap::new();
        features.insert("layer1".into(), randn(&[1, 256, 40, 40]).unwrap());
        features.insert("layer2".into(), randn(&[1, 512, 20, 20]).unwrap());
        features.insert("layer3".into(), randn(&[1, 1024, 10, 10]).unwrap());
        features.insert("layer4".into(), randn(&[1, 2048, 5, 5]).unwrap());
        let out = fpn.forward(&features).unwrap();
        // P5: [5, 5]. kernel=1, stride=2, padding=0 → P6: [3, 3].
        // (Numerically: floor((5 - 1) / 2) + 1 = 3.)
        assert_eq!(
            out["p6"].shape(),
            &[1, 256, 3, 3],
            "P6 must be P5 sub-sampled at stride 2 (kernel=1, padding=0)"
        );
        let p5 = out["p5"].data_vec().unwrap();
        let p6 = out["p6"].data_vec().unwrap();
        let p5_shape = out["p5"].shape().to_vec();
        let p6_shape = out["p6"].shape().to_vec();
        let c = p5_shape[1];
        let p5_h = p5_shape[2];
        let p5_w = p5_shape[3];
        let p6_h = p6_shape[2];
        let p6_w = p6_shape[3];
        for ci in 0..c {
            for ph in 0..p6_h {
                for pw in 0..p6_w {
                    let p6_idx = (ci * p6_h + ph) * p6_w + pw;
                    let src_h = 2 * ph;
                    let src_w = 2 * pw;
                    let p5_idx = (ci * p5_h + src_h) * p5_w + src_w;
                    let diff = (p6[p6_idx] - p5[p5_idx]).abs();
                    assert!(
                        diff < 1e-6,
                        "P6 must equal P5 sub-sampled at stride 2 (kernel=1); \
                         at (c={ci}, h={ph}, w={pw}) got p6={:.6} p5_at_src={:.6} diff={diff:.4e}",
                        p6[p6_idx],
                        p5[p5_idx],
                    );
                }
            }
        }
    }

    #[test]
    fn test_fpn_parameter_count() {
        let fpn = FeaturePyramidNetwork::<f32>::new().unwrap();
        let np: usize = fpn.parameters().iter().map(|p| p.numel()).sum();
        // 4 lateral (1×1) + 4 output (3×3), all 256→256 except laterals which
        // are in_ch → 256. Plus 8 biases of [256] (added per #1141 to match
        // torchvision's `Conv2dNormActivation(..., norm_layer=None)` which
        // keeps Conv2d's default `bias=True`).
        // lateral2: 256*256       + 256 = 65792
        // lateral3: 512*256       + 256 = 131328
        // lateral4: 1024*256      + 256 = 262400
        // lateral5: 2048*256      + 256 = 524544
        // output2..5: 4*(256*256*3*3 + 256) = 4*(589824 + 256) = 4*590080 = 2360320
        // total = 65792+131328+262400+524544+2360320 = 3344384
        assert!(np > 3_000_000, "FPN params too low: {np}");
        assert!(np < 4_000_000, "FPN params too high: {np}");
    }
}
