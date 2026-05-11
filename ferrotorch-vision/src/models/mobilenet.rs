//! MobileNetV2 and MobileNetV3-Small architectures.
//!
//! MobileNetV2 follows Sandler et al. 2018 *Inverted Residuals and Linear
//! Bottlenecks*; MobileNetV3-Small follows Howard et al. 2019 *Searching for
//! MobileNetV3*. Both implementations now mirror torchvision's
//! `mobilenet_v2` / `mobilenet_v3_small` parameter-naming layout exactly,
//! so the strict value-parity loader can adopt torchvision pretrained
//! state_dicts without renaming.
//!
//! Phase 7 (#1007): rebuilt from the pre-Phase 7 ReLU-only standard-Conv2d
//! placeholders into faithful reproductions of the torchvision reference.
//! Depthwise convs use [`Conv2d::new_full`]'s Phase 5 `groups`/`dilation`
//! support; SE attention uses [`SqueezeExcitation`] with the V3-specific
//! `HardSigmoid` scale activation; h-swish uses the existing
//! [`HardSwish`] primitive in `ferrotorch-nn::activation` (Phase 7
//! pre-flight: every primitive is in place — only the model wiring was
//! missing).
//!
//! ## Parameter naming (torchvision parity)
//!
//! Both models flatten their structure under a single `features.<i>`
//! [`Sequential`]-style index, with a leading stem (`features.0`), a
//! trailing head conv (`features.{18,12}`), and optionally a
//! [`SqueezeExcitation`] inside each `InvertedResidual`'s inner block
//! `Sequential`. Concretely:
//!
//! ```text
//! mobilenet_v2:
//!   features.0.{0=Conv2d, 1=BatchNorm2d}                    ← stem
//!   features.<i>.conv.{0,1,2,3}                              ← inverted-residual
//!   features.18.{0=Conv2d, 1=BatchNorm2d}                    ← head
//!   classifier.1.{weight,bias}                               ← Linear
//!
//! mobilenet_v3_small:
//!   features.0.{0=Conv2d, 1=BatchNorm2d}                     ← stem (h-swish)
//!   features.<i>.block.{0,1,2,3}                              ← inverted-residual + SE
//!   features.12.{0=Conv2d, 1=BatchNorm2d}                    ← head
//!   classifier.{0=Linear, 3=Linear}
//! ```
//!
//! The dropout activation entries (`classifier.{2 in V2, 2 in V3}`) are
//! parameter-free so they do not appear in `named_parameters`.
//!
//! See `tests/conformance_vision_models.rs::value_parity_pipeline` for
//! the strict torchvision adoption test.

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::activation::{HardSigmoid, HardSwish, ReLU, ReLU6};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::AdaptiveAvgPool2d;
use ferrotorch_nn::se::SqueezeExcitation;
use ferrotorch_nn::{Conv2d, Linear};

// ===========================================================================
// Activation kind
// ===========================================================================

/// Per-block activation choice.
///
/// V2 always uses ReLU6. V3 toggles between ReLU and HardSwish per block
/// per the inverted-residual config table.
#[derive(Debug, Clone, Copy)]
enum ActivationKind {
    Relu,
    Relu6,
    HardSwish,
}

impl ActivationKind {
    fn apply<T: Float>(self, x: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        match self {
            ActivationKind::Relu => ReLU::new().forward(x),
            ActivationKind::Relu6 => ReLU6::new().forward(x),
            ActivationKind::HardSwish => HardSwish::new().forward(x),
        }
    }
}

// ===========================================================================
// ConvBnAct — torchvision Conv2dNormActivation parity
// ===========================================================================
//
// Three child indices (`0`, `1`, `2`) exactly match torchvision's
// `Conv2dNormActivation` Sequential: 0=Conv2d, 1=BatchNorm2d, 2=activation.
// `as_activation` toggles whether the activation is rendered (for the V2
// head's no-act post-projection case torchvision also keeps the BN-only
// shape, surfaced as `Conv2d` + `BatchNorm2d` only).

struct ConvBnAct<T: Float> {
    conv: Conv2d<T>,
    bn: BatchNorm2d<T>,
    /// Activation function. None ↔ no-op (linear bottleneck projection).
    act: Option<ActivationKind>,
    training: bool,
}

impl<T: Float> ConvBnAct<T> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        bn_eps: f64,
        bn_momentum: f64,
        act: Option<ActivationKind>,
    ) -> FerrotorchResult<Self> {
        Self::new_with_dilation(
            in_ch, out_ch, kernel, stride, padding, 1, groups, bn_eps, bn_momentum, act,
        )
    }

    /// Same as [`Self::new`] but with explicit dilation (default for
    /// [`Self::new`] is 1). #1146: required for MobileNetV3-Large blocks
    /// 12-14 when used as the dilated backbone for LRASPP segmentation.
    #[allow(clippy::too_many_arguments)]
    fn new_with_dilation(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
        groups: usize,
        bn_eps: f64,
        bn_momentum: f64,
        act: Option<ActivationKind>,
    ) -> FerrotorchResult<Self> {
        let conv = Conv2d::new_full(
            in_ch,
            out_ch,
            (kernel, kernel),
            (stride, stride),
            (padding, padding),
            (dilation, dilation),
            groups,
            false, // bias=False — torchvision Conv2dNormActivation pattern
        )?;
        let bn = BatchNorm2d::new(out_ch, bn_eps, bn_momentum, true)?;
        Ok(Self {
            conv,
            bn,
            act,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for ConvBnAct<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.conv.forward(input)?;
        let x = Module::<T>::forward(&self.bn, &x)?;
        match self.act {
            Some(kind) => kind.apply(&x),
            None => Ok(x),
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv.parameters());
        p.extend(self.bn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv.parameters_mut());
        p.extend(self.bn.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.conv.named_parameters() {
            p.push((format!("0.{n}"), param));
        }
        for (n, param) in self.bn.named_parameters() {
            p.push((format!("1.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        // 0=conv, 1=bn (activation is parameter-free; we still surface it
        // through named_children so the path tree matches torchvision's
        // `Conv2dNormActivation` Sequential).
        vec![&self.conv, &self.bn]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("0".to_string(), &self.conv as &dyn Module<T>),
            ("1".to_string(), &self.bn),
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

// ===========================================================================
// MobileNetV2 InvertedResidual (`features.<i>.conv.<j>`)
// ===========================================================================
//
// Layout per torchvision:
//   expand_ratio == 1: conv = [depthwise+BN+ReLU6, project_1x1, project_BN]
//                       indices 0/1/2.
//   expand_ratio > 1:  conv = [expand_1x1+BN+ReLU6, depthwise+BN+ReLU6,
//                              project_1x1, project_BN]
//                       indices 0/1/2/3.
//
// `expand` and `depthwise` are themselves `Conv2dNormActivation` (3 children
// each). The trailing project is a bare Conv2d (`conv.<2-or-3>.weight`)
// followed by a bare BatchNorm2d (`conv.<3-or-4>.{weight,bias}`); we
// surface them at the project_idx and project_idx+1 children indices.

const V2_BN_EPS: f64 = 1e-5;
const V2_BN_MOM: f64 = 0.1;

struct V2InvertedResidual<T: Float> {
    expand: Option<ConvBnAct<T>>,
    depthwise: ConvBnAct<T>,
    project_conv: Conv2d<T>,
    project_bn: BatchNorm2d<T>,
    use_residual: bool,
    training: bool,
}

impl<T: Float> V2InvertedResidual<T> {
    fn new(
        in_ch: usize,
        out_ch: usize,
        stride: usize,
        expand_ratio: usize,
    ) -> FerrotorchResult<Self> {
        let hidden = in_ch * expand_ratio;
        let expand = if expand_ratio == 1 {
            None
        } else {
            Some(ConvBnAct::new(
                in_ch,
                hidden,
                1,
                1,
                0,
                1,
                V2_BN_EPS,
                V2_BN_MOM,
                Some(ActivationKind::Relu6),
            )?)
        };
        let depthwise = ConvBnAct::new(
            hidden,
            hidden,
            3,
            stride,
            1,
            hidden, // groups = hidden = depthwise
            V2_BN_EPS,
            V2_BN_MOM,
            Some(ActivationKind::Relu6),
        )?;
        let project_conv =
            Conv2d::new_full(hidden, out_ch, (1, 1), (1, 1), (0, 0), (1, 1), 1, false)?;
        let project_bn = BatchNorm2d::new(out_ch, V2_BN_EPS, V2_BN_MOM, true)?;
        Ok(Self {
            expand,
            depthwise,
            project_conv,
            project_bn,
            use_residual: stride == 1 && in_ch == out_ch,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for V2InvertedResidual<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = input.clone();
        if let Some(ref expand) = self.expand {
            x = expand.forward(&x)?;
        }
        let x = self.depthwise.forward(&x)?;
        let x = self.project_conv.forward(&x)?;
        let x = Module::<T>::forward(&self.project_bn, &x)?;
        // Linear (no-activation) bottleneck projection per the paper.
        if self.use_residual {
            add(&x, input)
        } else {
            Ok(x)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        if let Some(ref e) = self.expand {
            p.extend(e.parameters());
        }
        p.extend(self.depthwise.parameters());
        p.extend(self.project_conv.parameters());
        p.extend(self.project_bn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        if let Some(ref mut e) = self.expand {
            p.extend(e.parameters_mut());
        }
        p.extend(self.depthwise.parameters_mut());
        p.extend(self.project_conv.parameters_mut());
        p.extend(self.project_bn.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        // Inner sequential `conv` flattens to indices 0/1/(2|2,3).
        let mut p = Vec::new();
        if let Some(ref e) = self.expand {
            // expand at conv.0, depthwise at conv.1, project_conv.weight at
            // conv.2, project_bn.{weight,bias} at conv.3.
            for (n, param) in e.named_parameters() {
                p.push((format!("conv.0.{n}"), param));
            }
            for (n, param) in self.depthwise.named_parameters() {
                p.push((format!("conv.1.{n}"), param));
            }
            for (n, param) in self.project_conv.named_parameters() {
                p.push((format!("conv.2.{n}"), param));
            }
            for (n, param) in self.project_bn.named_parameters() {
                p.push((format!("conv.3.{n}"), param));
            }
        } else {
            // No expand: depthwise at conv.0, project_conv.weight at
            // conv.1, project_bn.{weight,bias} at conv.2.
            for (n, param) in self.depthwise.named_parameters() {
                p.push((format!("conv.0.{n}"), param));
            }
            for (n, param) in self.project_conv.named_parameters() {
                p.push((format!("conv.1.{n}"), param));
            }
            for (n, param) in self.project_bn.named_parameters() {
                p.push((format!("conv.2.{n}"), param));
            }
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = Vec::new();
        if let Some(ref e) = self.expand {
            out.push(e);
        }
        out.push(&self.depthwise);
        out.push(&self.project_conv);
        out.push(&self.project_bn);
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> = Vec::new();
        if let Some(ref e) = self.expand {
            out.push(("conv.0".to_string(), e as &dyn Module<T>));
            out.push(("conv.1".to_string(), &self.depthwise));
            out.push(("conv.2".to_string(), &self.project_conv));
            out.push(("conv.3".to_string(), &self.project_bn));
        } else {
            out.push(("conv.0".to_string(), &self.depthwise as &dyn Module<T>));
            out.push(("conv.1".to_string(), &self.project_conv));
            out.push(("conv.2".to_string(), &self.project_bn));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        if let Some(ref mut e) = self.expand {
            e.train();
        }
        self.depthwise.train();
        self.project_conv.train();
        self.project_bn.train();
    }
    fn eval(&mut self) {
        self.training = false;
        if let Some(ref mut e) = self.expand {
            e.eval();
        }
        self.depthwise.eval();
        self.project_conv.eval();
        self.project_bn.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// MobileNetV2
// ===========================================================================
//
// Block config (from Sandler et al. 2018, table 2; identical to torchvision's
// `_INVERTED_RESIDUAL_SETTING`):
//   t = expand ratio, c = out channels, n = num blocks, s = first-block stride
//
// The features Sequential is laid out as:
//   features.0     → stem ConvBnAct (3→32, 3×3, s=2, ReLU6)
//   features.1..17 → 17 inverted-residual blocks (t/c/n/s expanded)
//   features.18    → head ConvBnAct (last_in→1280, 1×1, s=1, ReLU6)
//
// classifier.0 = Dropout (parameter-free); classifier.1 = Linear.

struct V2Stage {
    t: usize,
    c: usize,
    n: usize,
    s: usize,
}

const MOBILENET_V2_STAGES: [V2Stage; 7] = [
    V2Stage {
        t: 1,
        c: 16,
        n: 1,
        s: 1,
    },
    V2Stage {
        t: 6,
        c: 24,
        n: 2,
        s: 2,
    },
    V2Stage {
        t: 6,
        c: 32,
        n: 3,
        s: 2,
    },
    V2Stage {
        t: 6,
        c: 64,
        n: 4,
        s: 2,
    },
    V2Stage {
        t: 6,
        c: 96,
        n: 3,
        s: 1,
    },
    V2Stage {
        t: 6,
        c: 160,
        n: 3,
        s: 2,
    },
    V2Stage {
        t: 6,
        c: 320,
        n: 1,
        s: 1,
    },
];

const V2_LAST_CHANNEL: usize = 1280;

/// MobileNetV2 model (torchvision `mobilenet_v2`).
pub struct MobileNetV2<T: Float> {
    stem: ConvBnAct<T>,
    blocks: Vec<V2InvertedResidual<T>>,
    head: ConvBnAct<T>,
    avgpool: AdaptiveAvgPool2d,
    classifier: Linear<T>,
    training: bool,
}

impl<T: Float> MobileNetV2<T> {
    /// Construct a MobileNetV2 with the given output class count.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let stem = ConvBnAct::new(
            3,
            32,
            3,
            2,
            1,
            1,
            V2_BN_EPS,
            V2_BN_MOM,
            Some(ActivationKind::Relu6),
        )?;
        let mut blocks: Vec<V2InvertedResidual<T>> = Vec::new();
        let mut in_ch = 32_usize;
        for stage in &MOBILENET_V2_STAGES {
            for i in 0..stage.n {
                let stride = if i == 0 { stage.s } else { 1 };
                blocks.push(V2InvertedResidual::new(in_ch, stage.c, stride, stage.t)?);
                in_ch = stage.c;
            }
        }
        let head = ConvBnAct::new(
            in_ch,
            V2_LAST_CHANNEL,
            1,
            1,
            0,
            1,
            V2_BN_EPS,
            V2_BN_MOM,
            Some(ActivationKind::Relu6),
        )?;
        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let classifier = Linear::new(V2_LAST_CHANNEL, num_classes, true)?;
        Ok(Self {
            stem,
            blocks,
            head,
            avgpool,
            classifier,
            training: true,
        })
    }

    /// Number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Index of the head ConvBnAct in the `features` sequential
    /// (= 1 + #blocks). Exposed so tests can assert the layout
    /// without hard-coding an integer.
    fn head_index(&self) -> usize {
        1 + self.blocks.len()
    }
}

impl<T: Float> Module<T> for MobileNetV2<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = self.stem.forward(input)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.head.forward(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        // classifier.0 = Dropout (parameter-free, eval-mode pass-through);
        // classifier.1 = Linear.
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.head.parameters());
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters_mut());
        for block in &mut self.blocks {
            p.extend(block.parameters_mut());
        }
        p.extend(self.head.parameters_mut());
        p.extend(self.classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.stem.named_parameters() {
            p.push((format!("features.0.{n}"), param));
        }
        for (i, block) in self.blocks.iter().enumerate() {
            for (n, param) in block.named_parameters() {
                p.push((format!("features.{}.{n}", i + 1), param));
            }
        }
        let head_idx = self.head_index();
        for (n, param) in self.head.named_parameters() {
            p.push((format!("features.{head_idx}.{n}"), param));
        }
        // classifier.1.{weight,bias}.
        for (n, param) in self.classifier.named_parameters() {
            p.push((format!("classifier.1.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = vec![&self.stem];
        for block in &self.blocks {
            out.push(block);
        }
        out.push(&self.head);
        out.push(&self.avgpool);
        out.push(&self.classifier);
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> =
            vec![("features.0".to_string(), &self.stem as &dyn Module<T>)];
        for (i, block) in self.blocks.iter().enumerate() {
            out.push((format!("features.{}", i + 1), block));
        }
        let head_idx = self.head_index();
        out.push((format!("features.{head_idx}"), &self.head));
        out.push(("classifier.1".to_string(), &self.classifier));
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.stem.train();
        for b in &mut self.blocks {
            b.train();
        }
        self.head.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.stem.eval();
        for b in &mut self.blocks {
            b.eval();
        }
        self.head.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

/// Convenience constructor for MobileNetV2.
pub fn mobilenet_v2<T: Float>(num_classes: usize) -> FerrotorchResult<MobileNetV2<T>> {
    MobileNetV2::new(num_classes)
}

// ---------------------------------------------------------------------------
// IntermediateFeatures — CL-499 (preserves the public test API)
// ---------------------------------------------------------------------------

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for MobileNetV2<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();
        let mut x = self.stem.forward(input)?;
        out.insert("stem".to_string(), x.clone());
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            out.insert(format!("block{i}"), x.clone());
        }
        let x = self.head.forward(&x)?;
        out.insert("last_conv".to_string(), x.clone());
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        out.insert("avgpool".to_string(), x.clone());
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        let logits = self.classifier.forward(&x)?;
        out.insert("classifier".to_string(), logits);
        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        let mut names = vec!["stem".to_string()];
        for i in 0..self.blocks.len() {
            names.push(format!("block{i}"));
        }
        names.push("last_conv".to_string());
        names.push("avgpool".to_string());
        names.push("classifier".to_string());
        names
    }
}

// ===========================================================================
// MobileNetV3-Small
// ===========================================================================
//
// V3 InvertedResidual: 1×1 expand → depthwise k×k → optional SE → 1×1
// project. Per-block activation is ReLU or HardSwish; SE uses HardSigmoid
// as scale activation. Block config from torchvision's
// `_mobilenet_v3_conf("mobilenet_v3_small")`.
//
// V3 BN parameters: eps=1e-3, momentum=1e-2 (NB: distinct from V2's
// 1e-5/0.1 — torchvision's `Conv2dNormActivation(norm_layer=
// partial(BatchNorm2d, eps=0.001, momentum=0.01))`).

const V3_BN_EPS: f64 = 1e-3;
const V3_BN_MOM: f64 = 1e-2;

#[derive(Debug, Clone, Copy)]
struct V3BlockCfg {
    in_ch: usize,
    kernel: usize,
    expanded: usize,
    out_ch: usize,
    use_se: bool,
    use_hs: bool,
    stride: usize,
    /// Depthwise-conv dilation. 1 in the classification configs (Small +
    /// non-dilated Large). #1146: blocks 12-14 of MobileNetV3-Large get
    /// dilation=2 when the dilated-backbone variant is constructed for
    /// LRASPP (`replace_stride_with_dilation=[True, True, True]` collapses
    /// to the last 3 blocks since torchvision only inserts dilation
    /// updates at the C4/C5 boundary).
    dilation: usize,
}

/// Round `v` up to the nearest multiple of `divisor`, clamped to a lower
/// bound — torchvision's `_make_divisible` helper. Used for V3 SE
/// squeeze-channel sizing (squeeze = `_make_divisible(expanded_channels
/// // 4, 8)`).
fn make_divisible(v: usize, divisor: usize) -> usize {
    let new_v = std::cmp::max(divisor, (v + divisor / 2) / divisor * divisor);
    if (new_v as f64) < 0.9 * (v as f64) {
        new_v + divisor
    } else {
        new_v
    }
}

/// MobileNetV3-Small block configuration (matches torchvision's
/// `_mobilenet_v3_conf("mobilenet_v3_small")` exactly).
const MOBILENET_V3_SMALL_CFG: [V3BlockCfg; 11] = [
    V3BlockCfg {
        in_ch: 16,
        kernel: 3,
        expanded: 16,
        out_ch: 16,
        use_se: true,
        use_hs: false,
        stride: 2,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 16,
        kernel: 3,
        expanded: 72,
        out_ch: 24,
        use_se: false,
        use_hs: false,
        stride: 2,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 24,
        kernel: 3,
        expanded: 88,
        out_ch: 24,
        use_se: false,
        use_hs: false,
        stride: 1,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 24,
        kernel: 5,
        expanded: 96,
        out_ch: 40,
        use_se: true,
        use_hs: true,
        stride: 2,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 40,
        kernel: 5,
        expanded: 240,
        out_ch: 40,
        use_se: true,
        use_hs: true,
        stride: 1,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 40,
        kernel: 5,
        expanded: 240,
        out_ch: 40,
        use_se: true,
        use_hs: true,
        stride: 1,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 40,
        kernel: 5,
        expanded: 120,
        out_ch: 48,
        use_se: true,
        use_hs: true,
        stride: 1,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 48,
        kernel: 5,
        expanded: 144,
        out_ch: 48,
        use_se: true,
        use_hs: true,
        stride: 1,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 48,
        kernel: 5,
        expanded: 288,
        out_ch: 96,
        use_se: true,
        use_hs: true,
        stride: 2,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 96,
        kernel: 5,
        expanded: 576,
        out_ch: 96,
        use_se: true,
        use_hs: true,
        stride: 1,
        dilation: 1,
    },
    V3BlockCfg {
        in_ch: 96,
        kernel: 5,
        expanded: 576,
        out_ch: 96,
        use_se: true,
        use_hs: true,
        stride: 1,
        dilation: 1,
    },
];

// #1146: MobileNetV3-Large block configuration. Mirrors torchvision's
// `_mobilenet_v3_conf("mobilenet_v3_large", dilated=False)` exactly
// (15 inverted-residual blocks; classification config has all dilations
// = 1). The dilated variant for segmentation uses [`mobilenet_v3_large_cfg_dilated`]
// which switches blocks 11..14 (zero-indexed within the block list, i.e.
// V3-Large `features.12..14` after the stem) to dilation=2.
const MOBILENET_V3_LARGE_CFG: [V3BlockCfg; 15] = [
    // 0: 16→16,  k3, exp=16,  SE=F, HS=F, s=1
    V3BlockCfg {
        in_ch: 16, kernel: 3, expanded: 16, out_ch: 16,
        use_se: false, use_hs: false, stride: 1, dilation: 1,
    },
    // 1: 16→24,  k3, exp=64,  SE=F, HS=F, s=2
    V3BlockCfg {
        in_ch: 16, kernel: 3, expanded: 64, out_ch: 24,
        use_se: false, use_hs: false, stride: 2, dilation: 1,
    },
    // 2: 24→24,  k3, exp=72,  SE=F, HS=F, s=1
    V3BlockCfg {
        in_ch: 24, kernel: 3, expanded: 72, out_ch: 24,
        use_se: false, use_hs: false, stride: 1, dilation: 1,
    },
    // 3: 24→40,  k5, exp=72,  SE=T, HS=F, s=2
    V3BlockCfg {
        in_ch: 24, kernel: 5, expanded: 72, out_ch: 40,
        use_se: true,  use_hs: false, stride: 2, dilation: 1,
    },
    // 4: 40→40,  k5, exp=120, SE=T, HS=F, s=1  ← LRASPP "low" tap (after this)
    V3BlockCfg {
        in_ch: 40, kernel: 5, expanded: 120, out_ch: 40,
        use_se: true,  use_hs: false, stride: 1, dilation: 1,
    },
    // 5: 40→40,  k5, exp=120, SE=T, HS=F, s=1
    V3BlockCfg {
        in_ch: 40, kernel: 5, expanded: 120, out_ch: 40,
        use_se: true,  use_hs: false, stride: 1, dilation: 1,
    },
    // 6: 40→80,  k3, exp=240, SE=F, HS=T, s=2
    V3BlockCfg {
        in_ch: 40, kernel: 3, expanded: 240, out_ch: 80,
        use_se: false, use_hs: true,  stride: 2, dilation: 1,
    },
    // 7: 80→80,  k3, exp=200, SE=F, HS=T, s=1
    V3BlockCfg {
        in_ch: 80, kernel: 3, expanded: 200, out_ch: 80,
        use_se: false, use_hs: true,  stride: 1, dilation: 1,
    },
    // 8: 80→80,  k3, exp=184, SE=F, HS=T, s=1
    V3BlockCfg {
        in_ch: 80, kernel: 3, expanded: 184, out_ch: 80,
        use_se: false, use_hs: true,  stride: 1, dilation: 1,
    },
    // 9: 80→80,  k3, exp=184, SE=F, HS=T, s=1
    V3BlockCfg {
        in_ch: 80, kernel: 3, expanded: 184, out_ch: 80,
        use_se: false, use_hs: true,  stride: 1, dilation: 1,
    },
    // 10: 80→112, k3, exp=480, SE=T, HS=T, s=1
    V3BlockCfg {
        in_ch: 80, kernel: 3, expanded: 480, out_ch: 112,
        use_se: true,  use_hs: true,  stride: 1, dilation: 1,
    },
    // 11: 112→112, k3, exp=672, SE=T, HS=T, s=1
    V3BlockCfg {
        in_ch: 112, kernel: 3, expanded: 672, out_ch: 112,
        use_se: true,  use_hs: true,  stride: 1, dilation: 1,
    },
    // 12: 112→160, k5, exp=672, SE=T, HS=T, s=2 (dilated variant: s=1, dil=2)
    V3BlockCfg {
        in_ch: 112, kernel: 5, expanded: 672, out_ch: 160,
        use_se: true,  use_hs: true,  stride: 2, dilation: 1,
    },
    // 13: 160→160, k5, exp=960, SE=T, HS=T, s=1 (dilated variant: dil=2)
    V3BlockCfg {
        in_ch: 160, kernel: 5, expanded: 960, out_ch: 160,
        use_se: true,  use_hs: true,  stride: 1, dilation: 1,
    },
    // 14: 160→160, k5, exp=960, SE=T, HS=T, s=1 (dilated variant: dil=2)
    V3BlockCfg {
        in_ch: 160, kernel: 5, expanded: 960, out_ch: 160,
        use_se: true,  use_hs: true,  stride: 1, dilation: 1,
    },
];

/// Build the dilated MobileNetV3-Large config table for segmentation.
///
/// Mirrors torchvision's `_mobilenet_v3_conf("mobilenet_v3_large",
/// dilated=True)` EXACTLY:
/// - Block 12: stride=2 (kept!), dilation=2 — input from block 11 stays at
///   33×33 because the depthwise's effective stride is overridden to 1
///   inside `V3InvertedResidual::new` whenever `dilation > 1`.
///   `use_res_connect` uses the CONFIG stride, so it remains False
///   (no residual across the stage boundary, matching torchvision).
/// - Blocks 13..14: stride=1, dilation=2.
///
/// #1146 root cause: the previous mapping (which dropped block 12's
/// stride to 1 and kept its dilation at 1) caused block 12's
/// `use_res_connect` to become True (since stride==1 && in==out is FALSE
/// because in=112, out=160, but… actually that's also False). The true
/// divergence was the depthwise dilation: block 12 needs dilation=2 to
/// keep its 5×5 kernel sampling at the right effective receptive field
/// for the layer-after-C4 receptive-field budget. Without dilation=2 the
/// per-output value differed enough that downstream block 13/14 (also
/// affected) accumulated to ~1.0 max-abs drift, which the LRASPP head
/// then compounded.
fn mobilenet_v3_large_cfg_dilated() -> [V3BlockCfg; 15] {
    let mut cfg = MOBILENET_V3_LARGE_CFG;
    // Block 12: keep stride=2 (config-level). Dilation=2; the
    // V3InvertedResidual constructor below forces the depthwise's
    // effective stride to 1 when dilation>1 (mirroring torchvision's
    // `stride = 1 if cnf.dilation > 1 else cnf.stride`).
    cfg[12].dilation = 2;
    // Blocks 13..14: dilation=2 (their stride was already 1, so no
    // effective-stride override is needed; but the dilation flag still
    // enlarges the receptive field).
    cfg[13].dilation = 2;
    cfg[14].dilation = 2;
    cfg
}

/// V3 last-stage head channel count is computed as `6 * last_block_out_ch`
/// (96 in V3-Small) = 576. The last_channel before classifier is 1024.
const V3_SMALL_HEAD_CHANNEL: usize = 576;
const V3_SMALL_LAST_CHANNEL: usize = 1024;

struct V3InvertedResidual<T: Float> {
    /// 1×1 expand ConvBnAct (None when expand_ratio == 1, i.e.
    /// `expanded == in_ch` — torchvision's `if cnf.expanded_channels !=
    /// cnf.input_channels` gate).
    expand: Option<ConvBnAct<T>>,
    depthwise: ConvBnAct<T>,
    se: Option<SqueezeExcitation<T>>,
    project: ConvBnAct<T>,
    use_residual: bool,
    training: bool,
}

impl<T: Float> V3InvertedResidual<T> {
    fn new(cfg: V3BlockCfg) -> FerrotorchResult<Self> {
        let act_kind = if cfg.use_hs {
            ActivationKind::HardSwish
        } else {
            ActivationKind::Relu
        };

        // Torchvision V3 InvertedResidual: skip the 1×1 expand when the
        // expansion ratio is 1 (`expanded == in_ch`).
        let expand = if cfg.expanded == cfg.in_ch {
            None
        } else {
            Some(ConvBnAct::new(
                cfg.in_ch,
                cfg.expanded,
                1,
                1,
                0,
                1,
                V3_BN_EPS,
                V3_BN_MOM,
                Some(act_kind),
            )?)
        };
        // Depthwise convolution. #1146:
        //   * Padding is `(kernel / 2) * dilation` so the spatial dims
        //     are preserved when dilation > 1 (matching torchvision's
        //     `Conv2dNormActivation` auto-padding).
        //   * Effective stride: torchvision's `InvertedResidual.__init__`
        //     forces `stride = 1 if cnf.dilation > 1 else cnf.stride`.
        //     This is how the dilated stages avoid downsampling while
        //     still carrying their CONFIG stride flag (which feeds into
        //     `use_res_connect` below).
        let pad = (cfg.kernel / 2) * cfg.dilation;
        let effective_stride = if cfg.dilation > 1 { 1 } else { cfg.stride };
        let depthwise = ConvBnAct::new_with_dilation(
            cfg.expanded,
            cfg.expanded,
            cfg.kernel,
            effective_stride,
            pad,
            cfg.dilation,
            cfg.expanded, // depthwise
            V3_BN_EPS,
            V3_BN_MOM,
            Some(act_kind),
        )?;
        let se = if cfg.use_se {
            // squeeze_channels = make_divisible(expanded // 4, 8).
            let sq = make_divisible(cfg.expanded / 4, 8);
            // V3 SE uses ReLU activation + HardSigmoid scale activation
            // (torchvision's `SqueezeExcitation(..., scale_activation=
            // partial(nn.Hardsigmoid, inplace=True))`).
            Some(SqueezeExcitation::new_with_activations(
                cfg.expanded,
                sq,
                Box::new(ReLU::new()),
                Box::new(HardSigmoid::new()),
            )?)
        } else {
            None
        };
        // Project: 1×1 conv → BN, no activation.
        let project = ConvBnAct::new(
            cfg.expanded,
            cfg.out_ch,
            1,
            1,
            0,
            1,
            V3_BN_EPS,
            V3_BN_MOM,
            None, // no activation on project
        )?;
        Ok(Self {
            expand,
            depthwise,
            se,
            project,
            use_residual: cfg.stride == 1 && cfg.in_ch == cfg.out_ch,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for V3InvertedResidual<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = input.clone();
        if let Some(ref e) = self.expand {
            x = e.forward(&x)?;
        }
        let x = self.depthwise.forward(&x)?;
        let x = if let Some(ref se) = self.se {
            se.forward(&x)?
        } else {
            x
        };
        let x = self.project.forward(&x)?;
        if self.use_residual {
            add(&x, input)
        } else {
            Ok(x)
        }
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        if let Some(ref e) = self.expand {
            p.extend(e.parameters());
        }
        p.extend(self.depthwise.parameters());
        if let Some(ref se) = self.se {
            p.extend(se.parameters());
        }
        p.extend(self.project.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        if let Some(ref mut e) = self.expand {
            p.extend(e.parameters_mut());
        }
        p.extend(self.depthwise.parameters_mut());
        if let Some(ref mut se) = self.se {
            p.extend(se.parameters_mut());
        }
        p.extend(self.project.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        let mut idx = 0_usize;
        if let Some(ref e) = self.expand {
            for (n, param) in e.named_parameters() {
                p.push((format!("block.{idx}.{n}"), param));
            }
            idx += 1;
        }
        for (n, param) in self.depthwise.named_parameters() {
            p.push((format!("block.{idx}.{n}"), param));
        }
        idx += 1;
        if let Some(ref se) = self.se {
            for (n, param) in se.named_parameters() {
                p.push((format!("block.{idx}.{n}"), param));
            }
            idx += 1;
        }
        for (n, param) in self.project.named_parameters() {
            p.push((format!("block.{idx}.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = Vec::new();
        if let Some(ref e) = self.expand {
            out.push(e);
        }
        out.push(&self.depthwise);
        if let Some(ref se) = self.se {
            out.push(se);
        }
        out.push(&self.project);
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> = Vec::new();
        let mut idx = 0_usize;
        if let Some(ref e) = self.expand {
            out.push((format!("block.{idx}"), e as &dyn Module<T>));
            idx += 1;
        }
        out.push((format!("block.{idx}"), &self.depthwise as &dyn Module<T>));
        idx += 1;
        if let Some(ref se) = self.se {
            out.push((format!("block.{idx}"), se as &dyn Module<T>));
            idx += 1;
        }
        out.push((format!("block.{idx}"), &self.project as &dyn Module<T>));
        out
    }

    fn train(&mut self) {
        self.training = true;
        if let Some(ref mut e) = self.expand {
            e.train();
        }
        self.depthwise.train();
        if let Some(ref mut se) = self.se {
            se.train();
        }
        self.project.train();
    }
    fn eval(&mut self) {
        self.training = false;
        if let Some(ref mut e) = self.expand {
            e.eval();
        }
        self.depthwise.eval();
        if let Some(ref mut se) = self.se {
            se.eval();
        }
        self.project.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

/// MobileNetV3-Small (torchvision `mobilenet_v3_small`).
pub struct MobileNetV3Small<T: Float> {
    stem: ConvBnAct<T>,
    blocks: Vec<V3InvertedResidual<T>>,
    head: ConvBnAct<T>,
    avgpool: AdaptiveAvgPool2d,
    classifier_0: Linear<T>,
    classifier_3: Linear<T>,
    training: bool,
}

impl<T: Float> MobileNetV3Small<T> {
    /// Construct a MobileNetV3-Small with the given output class count.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let stem = ConvBnAct::new(
            3,
            16,
            3,
            2,
            1,
            1,
            V3_BN_EPS,
            V3_BN_MOM,
            Some(ActivationKind::HardSwish),
        )?;
        let mut blocks: Vec<V3InvertedResidual<T>> = Vec::new();
        for cfg in &MOBILENET_V3_SMALL_CFG {
            blocks.push(V3InvertedResidual::new(*cfg)?);
        }
        let head = ConvBnAct::new(
            96,
            V3_SMALL_HEAD_CHANNEL,
            1,
            1,
            0,
            1,
            V3_BN_EPS,
            V3_BN_MOM,
            Some(ActivationKind::HardSwish),
        )?;
        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let classifier_0 = Linear::new(V3_SMALL_HEAD_CHANNEL, V3_SMALL_LAST_CHANNEL, true)?;
        let classifier_3 = Linear::new(V3_SMALL_LAST_CHANNEL, num_classes, true)?;
        Ok(Self {
            stem,
            blocks,
            head,
            avgpool,
            classifier_0,
            classifier_3,
            training: true,
        })
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    fn head_index(&self) -> usize {
        1 + self.blocks.len()
    }
}

impl<T: Float> Module<T> for MobileNetV3Small<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = self.stem.forward(input)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.head.forward(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        // classifier: Linear(0) → HardSwish(1) → Dropout(2, eval-mode pass-through)
        //           → Linear(3).
        let x = self.classifier_0.forward(&x)?;
        let x = HardSwish::new().forward(&x)?;
        self.classifier_3.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        for b in &self.blocks {
            p.extend(b.parameters());
        }
        p.extend(self.head.parameters());
        p.extend(self.classifier_0.parameters());
        p.extend(self.classifier_3.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters_mut());
        for b in &mut self.blocks {
            p.extend(b.parameters_mut());
        }
        p.extend(self.head.parameters_mut());
        p.extend(self.classifier_0.parameters_mut());
        p.extend(self.classifier_3.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.stem.named_parameters() {
            p.push((format!("features.0.{n}"), param));
        }
        for (i, block) in self.blocks.iter().enumerate() {
            for (n, param) in block.named_parameters() {
                p.push((format!("features.{}.{n}", i + 1), param));
            }
        }
        let head_idx = self.head_index();
        for (n, param) in self.head.named_parameters() {
            p.push((format!("features.{head_idx}.{n}"), param));
        }
        for (n, param) in self.classifier_0.named_parameters() {
            p.push((format!("classifier.0.{n}"), param));
        }
        for (n, param) in self.classifier_3.named_parameters() {
            p.push((format!("classifier.3.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = vec![&self.stem];
        for b in &self.blocks {
            out.push(b);
        }
        out.push(&self.head);
        out.push(&self.avgpool);
        out.push(&self.classifier_0);
        out.push(&self.classifier_3);
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> =
            vec![("features.0".to_string(), &self.stem as &dyn Module<T>)];
        for (i, block) in self.blocks.iter().enumerate() {
            out.push((format!("features.{}", i + 1), block));
        }
        let head_idx = self.head_index();
        out.push((format!("features.{head_idx}"), &self.head));
        out.push(("classifier.0".to_string(), &self.classifier_0));
        out.push(("classifier.3".to_string(), &self.classifier_3));
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.stem.train();
        for b in &mut self.blocks {
            b.train();
        }
        self.head.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.stem.eval();
        for b in &mut self.blocks {
            b.eval();
        }
        self.head.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

/// Convenience constructor for MobileNetV3-Small.
pub fn mobilenet_v3_small<T: Float>(num_classes: usize) -> FerrotorchResult<MobileNetV3Small<T>> {
    MobileNetV3Small::new(num_classes)
}

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for MobileNetV3Small<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();
        let mut x = self.stem.forward(input)?;
        out.insert("stem".to_string(), x.clone());
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            out.insert(format!("block{i}"), x.clone());
        }
        let x = self.head.forward(&x)?;
        out.insert("last_conv".to_string(), x.clone());
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        out.insert("avgpool".to_string(), x.clone());
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        let x = self.classifier_0.forward(&x)?;
        let x = HardSwish::new().forward(&x)?;
        let logits = self.classifier_3.forward(&x)?;
        out.insert("classifier".to_string(), logits);
        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        let mut names = vec!["stem".to_string()];
        for i in 0..self.blocks.len() {
            names.push(format!("block{i}"));
        }
        names.push("last_conv".to_string());
        names.push("avgpool".to_string());
        names.push("classifier".to_string());
        names
    }
}

// ===========================================================================
// MobileNetV3-Large (#1146)
// ===========================================================================
//
// Same building blocks as V3-Small but with the 15-entry config table.
// torchvision's `mobilenet_v3_large`:
//   features.0:    stem (Conv2dNormActivation 3→16, k=3, s=2, HardSwish)
//   features.1..15: 15 inverted-residual blocks (config above)
//   features.16:   head (Conv2dNormActivation 160→960, k=1, HardSwish)
//   avgpool: AdaptiveAvgPool2d(1)
//   classifier: Sequential[Linear(960, 1280), HardSwish, Dropout, Linear(1280, num_classes)]
//
// The `forward_features(input) → (low, high)` accessor returns the
// activations after block-index 4 (40-ch low, stride 8) and after the
// final head-conv at index 16 (960-ch high, stride 16 in the dilated
// variant, stride 32 in the classification variant). LRASPP consumes
// these two.

/// V3-Large head channel count is computed as `6 * last_block_out_ch`
/// (160 in V3-Large) = 960.
const V3_LARGE_HEAD_CHANNEL: usize = 960;
/// Width of the post-pool linear bottleneck before the classifier.
const V3_LARGE_LAST_CHANNEL: usize = 1280;

/// Per-block diagnostic snapshot from [`MobileNetV3Large::forward_with_block_dumps`]
/// (#1146). Holds the stem output, every InvertedResidual block output,
/// and the head-conv output. Consumed by `examples/probe_lraspp_stages.rs`.
pub struct MobileNetV3LargeStaged<T: Float> {
    pub stem: Tensor<T>,
    pub blocks: Vec<Tensor<T>>,
    pub head_conv: Tensor<T>,
}

/// MobileNetV3-Large (torchvision `mobilenet_v3_large`).
///
/// #1146: also acts as the dilated segmentation backbone for LRASPP when
/// constructed via [`Self::new_dilated`].
pub struct MobileNetV3Large<T: Float> {
    stem: ConvBnAct<T>,
    blocks: Vec<V3InvertedResidual<T>>,
    head: ConvBnAct<T>,
    avgpool: AdaptiveAvgPool2d,
    classifier_0: Linear<T>,
    classifier_3: Linear<T>,
    training: bool,
}

impl<T: Float> MobileNetV3Large<T> {
    /// Construct a MobileNetV3-Large with the given output class count
    /// (standard `mobilenet_v3_large` config — no dilated blocks).
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        Self::build(num_classes, &MOBILENET_V3_LARGE_CFG)
    }

    /// Construct a MobileNetV3-Large with the **dilated** segmentation
    /// config (`replace_stride_with_dilation=[True, True, True]`). This
    /// matches torchvision's `mobilenet_v3_large(weights=...,
    /// dilated=True)` exactly — only blocks 12..14 differ from the
    /// classification variant.
    ///
    /// The classifier (`classifier_0`, `classifier_3`) is still
    /// constructed so the parameter layout matches the torchvision
    /// `mobilenet_v3_large(dilated=True)` checkpoint structurally; the
    /// LRASPP wrapper simply does not invoke it on the forward path.
    pub fn new_dilated(num_classes: usize) -> FerrotorchResult<Self> {
        Self::build(num_classes, &mobilenet_v3_large_cfg_dilated())
    }

    fn build(num_classes: usize, cfg_table: &[V3BlockCfg; 15]) -> FerrotorchResult<Self> {
        let stem = ConvBnAct::new(
            3,
            16,
            3,
            2,
            1,
            1,
            V3_BN_EPS,
            V3_BN_MOM,
            Some(ActivationKind::HardSwish),
        )?;
        let mut blocks: Vec<V3InvertedResidual<T>> = Vec::new();
        for cfg in cfg_table.iter() {
            blocks.push(V3InvertedResidual::new(*cfg)?);
        }
        let head = ConvBnAct::new(
            160,
            V3_LARGE_HEAD_CHANNEL,
            1,
            1,
            0,
            1,
            V3_BN_EPS,
            V3_BN_MOM,
            Some(ActivationKind::HardSwish),
        )?;
        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let classifier_0 = Linear::new(V3_LARGE_HEAD_CHANNEL, V3_LARGE_LAST_CHANNEL, true)?;
        let classifier_3 = Linear::new(V3_LARGE_LAST_CHANNEL, num_classes, true)?;
        Ok(Self {
            stem,
            blocks,
            head,
            avgpool,
            classifier_0,
            classifier_3,
            training: true,
        })
    }

    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    fn head_index(&self) -> usize {
        1 + self.blocks.len()
    }

    /// Per-block diagnostic forward (#1146). Returns the stem output,
    /// every InvertedResidual block output, and the head-conv output.
    /// Used by `examples/probe_lraspp_stages.rs` to localize parity
    /// failures by stage.
    pub fn forward_with_block_dumps(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<MobileNetV3LargeStaged<T>> {
        let mut blocks_out: Vec<Tensor<T>> = Vec::with_capacity(self.blocks.len());
        let stem = self.stem.forward(input)?;
        let mut x = stem.clone();
        for block in &self.blocks {
            x = block.forward(&x)?;
            blocks_out.push(x.clone());
        }
        let head_conv = self.head.forward(&x)?;
        Ok(MobileNetV3LargeStaged {
            stem,
            blocks: blocks_out,
            head_conv,
        })
    }

    /// Extract the (low, high) feature pair used by LRASPP (#1146).
    ///
    /// torchvision's `lraspp_mobilenet_v3_large` wraps the backbone in
    /// `IntermediateLayerGetter({"4": "low", "16": "high"})`. The keys
    /// `"4"` and `"16"` refer to the **child indices of the features
    /// Sequential** (i.e. `features[4]` and `features[16]`). Concretely:
    ///
    /// - `features[0]`  = stem (`Conv2dNormActivation`)
    /// - `features[1..15]` = the 15 inverted-residual blocks
    /// - `features[16]` = final 1×1 head conv (`Conv2dNormActivation`)
    ///
    /// So `features[4]` is the **4th** InvertedResidual block (`blocks[3]`
    /// in our 0-indexed `Vec<V3InvertedResidual>`), and `features[16]`
    /// is the head conv output.
    ///
    /// - `low` ← activation after `blocks[3]` (40 channels, stride 8 in
    ///   both classification and dilated variants — block 3's stride=2
    ///   takes us from stride 4 to stride 8).
    /// - `high` ← activation after the head conv (`features[16]`). 960
    ///   channels. Stride is 16 in the dilated variant.
    ///
    /// #1146 root-cause #2: an earlier draft tapped `blocks[4]` (the
    /// SECOND 40-channel block) instead of `blocks[3]` — both have
    /// matching shape `[B, 40, H/8, W/8]` so the shape check did not
    /// detect the off-by-one. The per-block parity probe surfaced the
    /// mismatch (block 3 matched torchvision's "low" tap; block 4 did not).
    pub fn forward_low_high(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<(Tensor<T>, Tensor<T>)> {
        let mut x = self.stem.forward(input)?;
        let mut low: Option<Tensor<T>> = None;
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            // `features[4]` = blocks[3] (see doc above).
            if i == 3 {
                low = Some(x.clone());
            }
        }
        let high = self.head.forward(&x)?;
        let low = low.ok_or_else(|| ferrotorch_core::FerrotorchError::Internal {
            message: "MobileNetV3Large::forward_low_high: failed to capture features[4] tap"
                .into(),
        })?;
        Ok((low, high))
    }
}

impl<T: Float> Module<T> for MobileNetV3Large<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = self.stem.forward(input)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.head.forward(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        let x = self.classifier_0.forward(&x)?;
        let x = HardSwish::new().forward(&x)?;
        self.classifier_3.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        for b in &self.blocks {
            p.extend(b.parameters());
        }
        p.extend(self.head.parameters());
        p.extend(self.classifier_0.parameters());
        p.extend(self.classifier_3.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters_mut());
        for b in &mut self.blocks {
            p.extend(b.parameters_mut());
        }
        p.extend(self.head.parameters_mut());
        p.extend(self.classifier_0.parameters_mut());
        p.extend(self.classifier_3.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.stem.named_parameters() {
            p.push((format!("features.0.{n}"), param));
        }
        for (i, block) in self.blocks.iter().enumerate() {
            for (n, param) in block.named_parameters() {
                p.push((format!("features.{}.{n}", i + 1), param));
            }
        }
        let head_idx = self.head_index();
        for (n, param) in self.head.named_parameters() {
            p.push((format!("features.{head_idx}.{n}"), param));
        }
        for (n, param) in self.classifier_0.named_parameters() {
            p.push((format!("classifier.0.{n}"), param));
        }
        for (n, param) in self.classifier_3.named_parameters() {
            p.push((format!("classifier.3.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = vec![&self.stem];
        for b in &self.blocks {
            out.push(b);
        }
        out.push(&self.head);
        out.push(&self.avgpool);
        out.push(&self.classifier_0);
        out.push(&self.classifier_3);
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> =
            vec![("features.0".to_string(), &self.stem as &dyn Module<T>)];
        for (i, block) in self.blocks.iter().enumerate() {
            out.push((format!("features.{}", i + 1), block));
        }
        let head_idx = self.head_index();
        out.push((format!("features.{head_idx}"), &self.head));
        out.push(("classifier.0".to_string(), &self.classifier_0));
        out.push(("classifier.3".to_string(), &self.classifier_3));
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.stem.train();
        for b in &mut self.blocks {
            b.train();
        }
        self.head.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.stem.eval();
        for b in &mut self.blocks {
            b.eval();
        }
        self.head.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

/// Convenience constructor for MobileNetV3-Large.
pub fn mobilenet_v3_large<T: Float>(num_classes: usize) -> FerrotorchResult<MobileNetV3Large<T>> {
    MobileNetV3Large::new(num_classes)
}

/// Convenience constructor for the dilated MobileNetV3-Large backbone
/// used by LRASPP segmentation (#1146).
pub fn mobilenet_v3_large_dilated<T: Float>(
    num_classes: usize,
) -> FerrotorchResult<MobileNetV3Large<T>> {
    MobileNetV3Large::new_dilated(num_classes)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::storage::TensorStorage;

    fn dummy_image(batch: usize, ch: usize, h: usize, w: usize) -> Tensor<f32> {
        let numel = batch * ch * h * w;
        let data: Vec<f32> = (0..numel).map(|i| (i as f32) * 1e-3).collect();
        Tensor::from_storage(TensorStorage::cpu(data), vec![batch, ch, h, w], false).unwrap()
    }

    #[test]
    fn test_mobilenet_v2_named_parameters_match_torchvision_layout() {
        let model: MobileNetV2<f32> = mobilenet_v2(1000).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        // First block (features.1) has expand_ratio=1 so its conv has 3
        // sub-indices (depthwise=conv.0, project_conv.weight=conv.1,
        // project_bn.{w,b}=conv.2). features.2 has expand_ratio=6 so 4
        // sub-indices.
        assert!(names.iter().any(|n| n == "features.0.0.weight"));
        assert!(names.iter().any(|n| n == "features.0.1.weight"));
        assert!(names.iter().any(|n| n == "features.0.1.bias"));
        // Block 1 (no expand)
        assert!(names.iter().any(|n| n == "features.1.conv.0.0.weight"));
        assert!(names.iter().any(|n| n == "features.1.conv.0.1.weight"));
        assert!(names.iter().any(|n| n == "features.1.conv.0.1.bias"));
        assert!(names.iter().any(|n| n == "features.1.conv.1.weight"));
        assert!(names.iter().any(|n| n == "features.1.conv.2.weight"));
        assert!(names.iter().any(|n| n == "features.1.conv.2.bias"));
        // Block 2 (expand)
        assert!(names.iter().any(|n| n == "features.2.conv.0.0.weight"));
        assert!(names.iter().any(|n| n == "features.2.conv.1.0.weight"));
        assert!(names.iter().any(|n| n == "features.2.conv.2.weight"));
        assert!(names.iter().any(|n| n == "features.2.conv.3.weight"));
        // Head
        assert!(names.iter().any(|n| n == "features.18.0.weight"));
        assert!(names.iter().any(|n| n == "features.18.1.weight"));
        // Classifier
        assert!(names.iter().any(|n| n == "classifier.1.weight"));
        assert!(names.iter().any(|n| n == "classifier.1.bias"));
    }

    #[test]
    fn test_mobilenet_v2_output_shape() {
        let model: MobileNetV2<f32> = mobilenet_v2(10).unwrap();
        let x = dummy_image(2, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10]);
    }

    #[test]
    fn test_mobilenet_v2_param_count() {
        let model: MobileNetV2<f32> = mobilenet_v2(1000).unwrap();
        let n = model.num_parameters();
        assert!(n > 0);
        // Torchvision mobilenet_v2 has ~3.5M parameters (3,504,872).
        assert!(
            (3_300_000..=3_700_000).contains(&n),
            "MobileNetV2 param count {n} outside expected ~3.5M range"
        );
    }

    #[test]
    fn test_mobilenet_v3_small_named_parameters_match_torchvision_layout() {
        let model: MobileNetV3Small<f32> = mobilenet_v3_small(1000).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n == "features.0.0.weight"));
        // First block: in=16 expanded=16 (no expand), kernel=3, SE=true.
        // block.0=depthwise, block.1=SE, block.2=project (3 indices).
        assert!(names.iter().any(|n| n == "features.1.block.0.0.weight"));
        assert!(names.iter().any(|n| n == "features.1.block.1.fc1.weight"));
        assert!(names.iter().any(|n| n == "features.1.block.1.fc2.weight"));
        assert!(names.iter().any(|n| n == "features.1.block.2.0.weight"));
        // Second block: in=16 expanded=72, no SE → block.0=expand,
        // block.1=depthwise, block.2=project (3 indices).
        assert!(names.iter().any(|n| n == "features.2.block.0.0.weight"));
        assert!(names.iter().any(|n| n == "features.2.block.1.0.weight"));
        assert!(names.iter().any(|n| n == "features.2.block.2.0.weight"));
        // Block 4: in=24 expanded=96 SE=true → block.0=expand, block.1=
        // depthwise, block.2=SE, block.3=project (4 indices).
        assert!(names.iter().any(|n| n == "features.4.block.0.0.weight"));
        assert!(names.iter().any(|n| n == "features.4.block.1.0.weight"));
        assert!(names.iter().any(|n| n == "features.4.block.2.fc1.weight"));
        assert!(names.iter().any(|n| n == "features.4.block.2.fc2.weight"));
        assert!(names.iter().any(|n| n == "features.4.block.3.0.weight"));
        // Head + classifier
        assert!(names.iter().any(|n| n == "features.12.0.weight"));
        assert!(names.iter().any(|n| n == "classifier.0.weight"));
        assert!(names.iter().any(|n| n == "classifier.3.weight"));
    }

    #[test]
    fn test_mobilenet_v3_small_output_shape() {
        let model: MobileNetV3Small<f32> = mobilenet_v3_small(10).unwrap();
        let x = dummy_image(2, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10]);
    }

    #[test]
    fn test_mobilenet_v3_small_param_count() {
        let model: MobileNetV3Small<f32> = mobilenet_v3_small(1000).unwrap();
        let n = model.num_parameters();
        assert!(n > 0);
        // torchvision mobilenet_v3_small has 2,542,856 parameters.
        assert!(
            (2_300_000..=2_800_000).contains(&n),
            "MobileNetV3-Small param count {n} outside expected ~2.5M range"
        );
    }

    #[test]
    fn test_make_divisible_matches_torchvision() {
        // _make_divisible(16//4, 8) → 8
        assert_eq!(make_divisible(16 / 4, 8), 8);
        // _make_divisible(96//4, 8) → 24
        assert_eq!(make_divisible(96 / 4, 8), 24);
        // _make_divisible(240//4, 8) → 64
        assert_eq!(make_divisible(240 / 4, 8), 64);
        // _make_divisible(144//4, 8) → 40
        assert_eq!(make_divisible(144 / 4, 8), 40);
        // _make_divisible(120//4, 8) → 32
        assert_eq!(make_divisible(120 / 4, 8), 32);
        // _make_divisible(288//4, 8) → 72
        assert_eq!(make_divisible(288 / 4, 8), 72);
        // _make_divisible(576//4, 8) → 144
        assert_eq!(make_divisible(576 / 4, 8), 144);
    }
}
