//! Atrous Spatial Pyramid Pooling (ASPP) module for DeepLabV3.
//!
//! Mirrors `torchvision.models.segmentation.deeplabv3.ASPP` (torchvision 0.21.x).
//!
//! ## Architecture
//!
//! ```text
//! input [B, in_channels, H, W]
//!   ├─ 1×1 conv (rate=1)                    → [B, 256, H, W]
//!   ├─ 3×3 dilated conv (rate=6)            → [B, 256, H, W]
//!   ├─ 3×3 dilated conv (rate=12)           → [B, 256, H, W]
//!   ├─ 3×3 dilated conv (rate=18)           → [B, 256, H, W]
//!   └─ global avg pool → 1×1 conv → bilinear upsample → [B, 256, H, W]
//!        ↓ concat [B, 1280, H, W]
//!   └─ project conv 1×1                     → [B, 256, H, W]
//! ```
//!
//! The five branches are each a (conv → BN → ReLU) block. The projection
//! uses dropout (p=0.5) before the final 1×1 conv, matching torchvision.
//!
//! Phase 6 (#988) migrated the dilated 3×3 branches off a host-side
//! `data_vec()` + 7-deep CPU loop onto `Conv2d::new_full(.., dilation, groups=1)`
//! (Phase 5 of #1002). The `requires_grad=false` workaround on the prior
//! leaf-output tensor is gone; gradients now flow through the dilated convs
//! the same way they do through any other `Conv2d`. Channel concatenation
//! is now the autograd-aware `ferrotorch_core::grad_fns::shape::cat` (axis=1)
//! primitive instead of the second host-side allocate-and-copy that lived
//! in `concat_channels`. See `tests/probe_aspp_dilated_migration.rs` for the
//! per-rate (2 / 6 / 12 / 18) byte-equivalence probe that proved the
//! migration before this code changed.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::shape::cat;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::AdaptiveAvgPool2d;
use ferrotorch_nn::upsample::{InterpolateMode, interpolate};
use ferrotorch_nn::{Conv2d, Dropout};

// ---------------------------------------------------------------------------
// DilatedConv2d — thin wrapper around Conv2d::new_full with BN + ReLU
// ---------------------------------------------------------------------------

/// A 3×3 convolution with dilation, no bias, followed by BN and ReLU,
/// matching the torchvision `ASPPConv` building block.
///
/// The output spatial size equals the input spatial size (same-size
/// padding: `pad = dilation * (k-1) / 2 = dilation` for `k=3`).
///
/// Phase 6 (#988): the underlying spatial conv is now
/// `Conv2d::new_full(in, out, (3,3), (1,1), (dilation,dilation), (dilation,dilation), 1, false)`.
/// `Conv2d::forward`'s CUDA fast path skips when `dilation != (1,1)`, so the
/// dilated branches transparently fall back to the existing CPU im2col path
/// (the same path Phase 5's grouped probe exercises). Gradients flow through
/// `Conv2d`'s autograd hooks; the prior `requires_grad=false` leaf workaround
/// is removed.
pub struct DilatedConv2d<T: Float> {
    conv: Conv2d<T>,
    bn: BatchNorm2d<T>,
    training: bool,
}

impl<T: Float> DilatedConv2d<T> {
    /// Create a new `DilatedConv2d`.
    ///
    /// Uses the same Kaiming-uniform (ReLU gain) initialisation as
    /// `Conv2d::new_full` (which matches torchvision).
    pub fn new(in_channels: usize, out_channels: usize, dilation: usize) -> FerrotorchResult<Self> {
        // Same-size padding: pad = dilation * (k-1) / 2 = dilation for k=3.
        let conv = Conv2d::new_full(
            in_channels,
            out_channels,
            (3, 3),
            (1, 1),
            (dilation, dilation),
            (dilation, dilation),
            1,
            false,
        )?;
        let bn = BatchNorm2d::new(out_channels, 1e-5, 0.1, true)?;
        Ok(Self {
            conv,
            bn,
            training: true,
        })
    }

    /// Run the dilated 3×3 convolution, BN, and ReLU.
    pub fn forward_inner(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let out = self.conv.forward(input)?;
        let out = Module::<T>::forward(&self.bn, &out)?;
        relu(&out)
    }
}

impl<T: Float> Module<T> for DilatedConv2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_inner(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.conv.parameters();
        p.extend(self.bn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.conv.parameters_mut();
        p.extend(self.bn.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        // The torchvision ASPPConv exposes a single `weight` for the
        // dilated conv (no bias) and `bn.<{weight,bias}>`. We match that
        // by flattening the inner Conv2d's keys directly (no `conv.`
        // prefix), so the parent `Aspp::named_parameters` produces
        // `1.weight`, `1.bn.weight`, ... at the rate-6/12/18 branches —
        // exactly the names torchvision uses (rate-1 conv1 already
        // matches via `0.conv.weight`/`0.bn.weight` on the ASPPConv1x1
        // sibling).
        let mut out = Vec::new();
        for (k, v) in self.conv.named_parameters() {
            out.push((k, v));
        }
        for (k, v) in self.bn.named_parameters() {
            out.push((format!("bn.{k}"), v));
        }
        out
    }

    // Phase 4 (#995): expose conv + BN children so the loader can reach
    // both under whatever path the parent gave the DilatedConv2d
    // (e.g. `1` / `2` / `3` in Aspp).
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.conv, &self.bn]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        // The named_parameters above flattens conv.<...> down to <...> so
        // the path → module index keeps `bn.<...>` reachable but does
        // not double-prefix the conv weight. Pair the conv with the
        // empty path here for path consistency with named_parameters.
        vec![(String::new(), &self.conv), ("bn".to_string(), &self.bn)]
    }

    fn train(&mut self) {
        self.training = true;
        self.conv.train();
        self.bn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv.eval();
        self.bn.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ASPPConv1x1 — 1×1 conv branch (uses standard Conv2d)
// ---------------------------------------------------------------------------

struct ASPPConv1x1<T: Float> {
    conv: Conv2d<T>,
    bn: BatchNorm2d<T>,
    training: bool,
}

impl<T: Float> ASPPConv1x1<T> {
    fn new(in_channels: usize, out_channels: usize) -> FerrotorchResult<Self> {
        let conv = Conv2d::new(in_channels, out_channels, (1, 1), (1, 1), (0, 0), false)?;
        let bn = BatchNorm2d::new(out_channels, 1e-5, 0.1, true)?;
        Ok(Self {
            conv,
            bn,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for ASPPConv1x1<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.conv.forward(input)?;
        let x = Module::<T>::forward(&self.bn, &x)?;
        relu(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.conv.parameters();
        p.extend(self.bn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.conv.parameters_mut();
        p.extend(self.bn.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.conv.named_parameters() {
            out.push((format!("conv.{k}"), v));
        }
        for (k, v) in self.bn.named_parameters() {
            out.push((format!("bn.{k}"), v));
        }
        out
    }

    // Phase 4 (#995): expose conv + BN children mirroring `named_parameters`.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.conv, &self.bn]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("conv".to_string(), &self.conv),
            ("bn".to_string(), &self.bn),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.conv.train();
        self.bn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv.eval();
        self.bn.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ASPPPooling — global average pool + 1×1 conv branch
// ---------------------------------------------------------------------------

struct ASPPPooling<T: Float> {
    avgpool: AdaptiveAvgPool2d,
    conv: Conv2d<T>,
    bn: BatchNorm2d<T>,
    training: bool,
}

impl<T: Float> ASPPPooling<T> {
    fn new(in_channels: usize, out_channels: usize) -> FerrotorchResult<Self> {
        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let conv = Conv2d::new(in_channels, out_channels, (1, 1), (1, 1), (0, 0), false)?;
        let bn = BatchNorm2d::new(out_channels, 1e-5, 0.1, true)?;
        Ok(Self {
            avgpool,
            conv,
            bn,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for ASPPPooling<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];

        // Global average pool: [B, C, 1, 1]
        let x = Module::<T>::forward(&self.avgpool, input)?;
        let x = self.conv.forward(&x)?;
        let x = Module::<T>::forward(&self.bn, &x)?;
        let x = relu(&x)?;

        // Upsample back to spatial size of input.
        interpolate(
            &x,
            Some([h_in, w_in]),
            None,
            InterpolateMode::Bilinear,
            false,
        )
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.conv.parameters();
        p.extend(self.bn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.conv.parameters_mut();
        p.extend(self.bn.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.conv.named_parameters() {
            out.push((format!("conv.{k}"), v));
        }
        for (k, v) in self.bn.named_parameters() {
            out.push((format!("bn.{k}"), v));
        }
        out
    }

    // Phase 4 (#995): expose avgpool + conv + BN children. Avgpool is
    // a leaf with no params but inclusion keeps the tree faithful for
    // future-proof traversal; the named_parameters above intentionally
    // skips it.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.avgpool, &self.conv, &self.bn]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("avgpool".to_string(), &self.avgpool),
            ("conv".to_string(), &self.conv),
            ("bn".to_string(), &self.bn),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.bn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.bn.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ASPP
// ---------------------------------------------------------------------------

/// Atrous Spatial Pyramid Pooling module.
///
/// Input: `[B, in_channels, H, W]`
/// Output: `[B, 256, H, W]`
///
/// Five branches are concatenated (totalling `5 * 256 = 1280` channels) and
/// projected back to 256 channels via a 1×1 conv + BN + ReLU, with dropout
/// p=0.5 preceding the projection (matching torchvision).
///
/// Phase 9 (#1009): the dilated 3×3 atrous rates are now plumbed through
/// [`Aspp::new`] as a `(usize, usize, usize)` triplet. torchvision's
/// `deeplabv3_resnet50` default is `(12, 24, 36)`; the prior hard-coded
/// `(6, 12, 18)` matched DeepLabV3+'s smaller head rather than DeepLabV3.
pub struct Aspp<T: Float> {
    /// 1×1 convolution branch.
    conv1: ASPPConv1x1<T>,
    /// Dilated 3×3 conv branches at the configured atrous rates.
    conv_r1: DilatedConv2d<T>,
    conv_r2: DilatedConv2d<T>,
    conv_r3: DilatedConv2d<T>,
    /// Global average pool branch.
    pool: ASPPPooling<T>,
    /// 1×1 projection conv (maps 1280 → 256).
    project: Conv2d<T>,
    project_bn: BatchNorm2d<T>,
    dropout: Dropout<T>,
    training: bool,
}

impl<T: Float> Aspp<T> {
    /// Construct an ASPP module.
    ///
    /// * `in_channels` — number of input feature channels (2048 for ResNet-50 layer4).
    /// * `out_channels` — output channels (256, torchvision default).
    /// * `atrous_rates` — three dilation rates for the 3×3 dilated branches.
    ///   torchvision's `deeplabv3_resnet50` default is `(12, 24, 36)`.
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        atrous_rates: (usize, usize, usize),
    ) -> FerrotorchResult<Self> {
        let (r1, r2, r3) = atrous_rates;
        let conv1 = ASPPConv1x1::new(in_channels, out_channels)?;
        let conv_r1 = DilatedConv2d::new(in_channels, out_channels, r1)?;
        let conv_r2 = DilatedConv2d::new(in_channels, out_channels, r2)?;
        let conv_r3 = DilatedConv2d::new(in_channels, out_channels, r3)?;
        let pool = ASPPPooling::new(in_channels, out_channels)?;

        // 5 branches × 256 channels = 1280 → 256 via 1×1 conv.
        let project = Conv2d::new(
            5 * out_channels,
            out_channels,
            (1, 1),
            (1, 1),
            (0, 0),
            false,
        )?;
        let project_bn = BatchNorm2d::new(out_channels, 1e-5, 0.1, true)?;
        let dropout = Dropout::new(0.5)?;

        Ok(Self {
            conv1,
            conv_r1,
            conv_r2,
            conv_r3,
            pool,
            project,
            project_bn,
            dropout,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Aspp<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let b1 = self.conv1.forward(input)?;
        let b2 = self.conv_r1.forward(input)?;
        let b3 = self.conv_r2.forward(input)?;
        let b4 = self.conv_r3.forward(input)?;
        let b5 = self.pool.forward(input)?;

        // Concatenate along channel dim (axis=1): [B, 1280, H, W]. Phase 6
        // (#988) replaced the prior `concat_channels` host-side allocate-
        // and-copy CPU-pull with the autograd-aware `cat` primitive.
        let concatenated = cat(&[b1, b2, b3, b4, b5], 1)?;

        // Dropout → project → BN → ReLU.
        let x = if self.training {
            Module::<T>::forward(&self.dropout, &concatenated)?
        } else {
            concatenated
        };
        let x = self.project.forward(&x)?;
        let x = Module::<T>::forward(&self.project_bn, &x)?;
        relu(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.conv_r1.parameters());
        p.extend(self.conv_r2.parameters());
        p.extend(self.conv_r3.parameters());
        p.extend(self.pool.parameters());
        p.extend(self.project.parameters());
        p.extend(self.project_bn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters_mut());
        p.extend(self.conv_r1.parameters_mut());
        p.extend(self.conv_r2.parameters_mut());
        p.extend(self.conv_r3.parameters_mut());
        p.extend(self.pool.parameters_mut());
        p.extend(self.project.parameters_mut());
        p.extend(self.project_bn.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.conv1.named_parameters() {
            out.push((format!("0.{k}"), v));
        }
        for (k, v) in self.conv_r1.named_parameters() {
            out.push((format!("1.{k}"), v));
        }
        for (k, v) in self.conv_r2.named_parameters() {
            out.push((format!("2.{k}"), v));
        }
        for (k, v) in self.conv_r3.named_parameters() {
            out.push((format!("3.{k}"), v));
        }
        for (k, v) in self.pool.named_parameters() {
            out.push((format!("4.{k}"), v));
        }
        for (k, v) in self.project.named_parameters() {
            out.push((format!("project.{k}"), v));
        }
        for (k, v) in self.project_bn.named_parameters() {
            out.push((format!("project_bn.{k}"), v));
        }
        out
    }

    // Phase 4 (#995): expose direct children mirroring `named_parameters`.
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.conv1,
            &self.conv_r1,
            &self.conv_r2,
            &self.conv_r3,
            &self.pool,
            &self.project,
            &self.project_bn,
            &self.dropout,
        ]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("0".to_string(), &self.conv1),
            ("1".to_string(), &self.conv_r1),
            ("2".to_string(), &self.conv_r2),
            ("3".to_string(), &self.conv_r3),
            ("4".to_string(), &self.pool),
            ("project".to_string(), &self.project),
            ("project_bn".to_string(), &self.project_bn),
            ("dropout".to_string(), &self.dropout),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.conv1.train();
        self.conv_r1.train();
        self.conv_r2.train();
        self.conv_r3.train();
        self.pool.train();
        self.project_bn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv1.eval();
        self.conv_r1.eval();
        self.conv_r2.eval();
        self.conv_r3.eval();
        self.pool.eval();
        self.project_bn.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// Phase 6 (#988): the prior `concat_channels` host-side allocate-and-copy
// helper has been removed. Channel-axis concatenation now flows through
// `ferrotorch_core::grad_fns::shape::cat(&[...], 1)`, which is autograd-
// aware and does not pull through CPU mid-graph.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::no_grad;

    fn randn_4d(shape: [usize; 4]) -> Tensor<f32> {
        use ferrotorch_core::randn;
        randn(&shape).unwrap()
    }

    #[test]
    fn test_aspp_output_shape() {
        // torchvision deeplabv3_resnet50 default rates (Phase 9 #1009).
        let aspp = Aspp::<f32>::new(2048, 256, (12, 24, 36)).unwrap();
        // Tiny spatial: 4×4 to keep test fast.
        let x = no_grad(|| randn_4d([1, 2048, 4, 4]));
        let y = no_grad(|| aspp.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 256, 4, 4], "ASPP output shape mismatch");
    }

    #[test]
    fn test_aspp_preserves_spatial_dims() {
        // Use the smaller (6, 12, 18) DeepLabV3+ rates for the 256-channel
        // smaller-input test path; both rate sets must produce same-size
        // output for the spatial-preservation contract to hold.
        let aspp = Aspp::<f32>::new(256, 256, (6, 12, 18)).unwrap();
        let x = no_grad(|| randn_4d([2, 256, 8, 8]));
        let y = no_grad(|| aspp.forward(&x).unwrap());
        assert_eq!(y.shape(), &[2, 256, 8, 8]);
    }

    #[test]
    fn test_dilated_conv2d_same_size_output() {
        let d = DilatedConv2d::<f32>::new(64, 64, 6).unwrap();
        let x = no_grad(|| randn_4d([1, 64, 8, 8]));
        let y = no_grad(|| d.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 64, 8, 8]);
    }

    #[test]
    fn test_aspp_parameter_count() {
        let aspp = Aspp::<f32>::new(2048, 256, (12, 24, 36)).unwrap();
        let np: usize = aspp.parameters().iter().map(|p| p.numel()).sum();
        // Rough lower bound: 5 branches × 2048*256 params + project ~1M.
        assert!(np > 2_000_000, "ASPP params too low: {np}");
    }
}
