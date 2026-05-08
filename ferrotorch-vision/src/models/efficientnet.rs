//! EfficientNet-B0 architecture (full torchvision parity).
//!
//! Tan & Le 2019 *EfficientNet: Rethinking Model Scaling for Convolutional
//! Neural Networks*. Phase 7 (#1007) replaces the pre-Phase-7 standard-Conv2d
//! placeholder with the real MBConv (mobile inverted bottleneck +
//! depthwise + squeeze-and-excite) layout. Parameter naming mirrors
//! torchvision's `efficientnet_b0` exactly:
//!
//! ```text
//!   features.0.{0=Conv2d, 1=BatchNorm2d}                      ← stem (SiLU)
//!   features.<i>.<j>.block.{0..3}                              ← MBConv
//!   features.8.{0=Conv2d, 1=BatchNorm2d}                       ← head (SiLU)
//!   classifier.1.{weight,bias}                                  ← Linear
//! ```
//!
//! Each MBConv's inner `block` Sequential contains:
//!   - When `expand_ratio == 1`: depthwise(0) → SE(1) → project(2)  (3 entries)
//!   - When `expand_ratio > 1` : expand(0) → depthwise(1) → SE(2) → project(3)
//!
//! ## Stochastic depth — eval-mode pass-through
//!
//! torchvision's MBConv wraps a `StochasticDepth` module after the residual
//! addition. In `eval()` mode it is the identity (drop probability ignored),
//! so the eval-mode forward path is value-equivalent to omitting the wrap
//! entirely. The training-mode backward (Bernoulli-gated residual scaling)
//! is **out of scope** for Phase 7's value-parity-on-eval push and is
//! filed separately (see `Phase 7` finding §15).

use ferrotorch_core::grad_fns::activation::silu as silu_fn;
use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::activation::{SiLU, Sigmoid};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::AdaptiveAvgPool2d;
use ferrotorch_nn::se::SqueezeExcitation;
use ferrotorch_nn::{Conv2d, Linear};

// EfficientNet BN parameters: torchvision's `_efficientnet` builder uses
// `BatchNorm2d(eps=1e-3, momentum=1e-2)` for the SE/MBConv path.
const EN_BN_EPS: f64 = 1e-3;
const EN_BN_MOM: f64 = 1e-2;

// ===========================================================================
// ConvBnSiLU — torchvision Conv2dNormActivation parity (SiLU activation,
// optional drop). Equivalent to MobileNet's `ConvBnAct` but with a
// hard-wired SiLU activation choice and configurable on/off.
// ===========================================================================

struct ConvBnSiLU<T: Float> {
    conv: Conv2d<T>,
    bn: BatchNorm2d<T>,
    /// `true` when the activation step is rendered (SiLU). The MBConv
    /// project path drops the activation (linear bottleneck projection).
    apply_silu: bool,
    training: bool,
}

impl<T: Float> ConvBnSiLU<T> {
    #[allow(clippy::too_many_arguments)]
    fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
        groups: usize,
        apply_silu: bool,
    ) -> FerrotorchResult<Self> {
        let conv = Conv2d::new_full(
            in_ch,
            out_ch,
            (kernel, kernel),
            (stride, stride),
            (padding, padding),
            (1, 1),
            groups,
            false,
        )?;
        let bn = BatchNorm2d::new(out_ch, EN_BN_EPS, EN_BN_MOM, true)?;
        Ok(Self {
            conv,
            bn,
            apply_silu,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for ConvBnSiLU<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.conv.forward(input)?;
        let x = Module::<T>::forward(&self.bn, &x)?;
        if self.apply_silu { silu_fn(&x) } else { Ok(x) }
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
// MBConv — torchvision EfficientNet MBConv block parity
// ===========================================================================

#[derive(Debug, Clone, Copy)]
struct MBConvCfg {
    in_ch: usize,
    expanded: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
}

impl MBConvCfg {
    fn from_block_input(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        expand_ratio: usize,
    ) -> Self {
        Self {
            in_ch,
            expanded: in_ch * expand_ratio,
            out_ch,
            kernel,
            stride,
        }
    }
}

struct MBConv<T: Float> {
    /// 1×1 expand (None when `expanded == in_ch`, i.e. expand_ratio == 1).
    expand: Option<ConvBnSiLU<T>>,
    /// Depthwise k×k.
    depthwise: ConvBnSiLU<T>,
    /// SE block (always present in EfficientNet's MBConv).
    se: SqueezeExcitation<T>,
    /// 1×1 project (no activation — linear bottleneck).
    project: ConvBnSiLU<T>,
    /// Whether to apply the residual skip (stride==1 && in==out).
    use_residual: bool,
    training: bool,
}

impl<T: Float> MBConv<T> {
    fn new(cfg: MBConvCfg) -> FerrotorchResult<Self> {
        let expand = if cfg.expanded == cfg.in_ch {
            None
        } else {
            Some(ConvBnSiLU::new(cfg.in_ch, cfg.expanded, 1, 1, 0, 1, true)?)
        };
        let depthwise = ConvBnSiLU::new(
            cfg.expanded,
            cfg.expanded,
            cfg.kernel,
            cfg.stride,
            cfg.kernel / 2,
            cfg.expanded, // depthwise
            true,
        )?;
        // EfficientNet SE: squeeze_channels = max(1, cnf.input_channels // 4)
        // (input_channels = the BLOCK input, not expanded).
        let sq = std::cmp::max(1, cfg.in_ch / 4);
        // EfficientNet MBConv SE uses SiLU activation + Sigmoid scale
        // activation (torchvision default). The default
        // `SqueezeExcitation::new` would give ReLU + Sigmoid; we explicitly
        // request SiLU to mirror torchvision's
        // `SqueezeExcitation(activation=partial(nn.SiLU, inplace=True))`.
        let se = SqueezeExcitation::new_with_activations(
            cfg.expanded,
            sq,
            Box::new(SiLU::new()),
            Box::new(Sigmoid::new()),
        )?;
        let project = ConvBnSiLU::new(cfg.expanded, cfg.out_ch, 1, 1, 0, 1, false)?;
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

impl<T: Float> Module<T> for MBConv<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = input.clone();
        if let Some(ref e) = self.expand {
            x = e.forward(&x)?;
        }
        let x = self.depthwise.forward(&x)?;
        let x = self.se.forward(&x)?;
        let x = self.project.forward(&x)?;
        // Stochastic depth: torchvision wraps the output of `block` in
        // `StochasticDepth` BEFORE the residual add. In eval() mode it's
        // identity (training=true would scale by Bernoulli mask). Phase 7
        // is value-parity-on-eval; training-mode parity tracked separately.
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
        p.extend(self.se.parameters());
        p.extend(self.project.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        if let Some(ref mut e) = self.expand {
            p.extend(e.parameters_mut());
        }
        p.extend(self.depthwise.parameters_mut());
        p.extend(self.se.parameters_mut());
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
        for (n, param) in self.se.named_parameters() {
            p.push((format!("block.{idx}.{n}"), param));
        }
        idx += 1;
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
        out.push(&self.se);
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
        out.push((format!("block.{idx}"), &self.se as &dyn Module<T>));
        idx += 1;
        out.push((format!("block.{idx}"), &self.project as &dyn Module<T>));
        out
    }

    fn train(&mut self) {
        self.training = true;
        if let Some(ref mut e) = self.expand {
            e.train();
        }
        self.depthwise.train();
        self.se.train();
        self.project.train();
    }
    fn eval(&mut self) {
        self.training = false;
        if let Some(ref mut e) = self.expand {
            e.eval();
        }
        self.depthwise.eval();
        self.se.eval();
        self.project.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Stage descriptor + EfficientNet-B0 config
// ===========================================================================

struct Stage {
    /// Number of MBConvs in this stage.
    num_blocks: usize,
    /// Block input/output channels (after the first block, in==out so
    /// expand_ratio applies symmetrically).
    out_ch: usize,
    /// Kernel size for the depthwise.
    kernel: usize,
    /// Stride of the FIRST block (subsequent blocks always stride=1).
    stride: usize,
    /// Expansion ratio (1 for the first stage in B0, 6 elsewhere).
    expand_ratio: usize,
}

/// EfficientNet-B0 stage table — matches torchvision's
/// `_MBConvConfig`-derived stages exactly.
const EFFICIENTNET_B0_STAGES: [Stage; 7] = [
    Stage {
        num_blocks: 1,
        out_ch: 16,
        kernel: 3,
        stride: 1,
        expand_ratio: 1,
    },
    Stage {
        num_blocks: 2,
        out_ch: 24,
        kernel: 3,
        stride: 2,
        expand_ratio: 6,
    },
    Stage {
        num_blocks: 2,
        out_ch: 40,
        kernel: 5,
        stride: 2,
        expand_ratio: 6,
    },
    Stage {
        num_blocks: 3,
        out_ch: 80,
        kernel: 3,
        stride: 2,
        expand_ratio: 6,
    },
    Stage {
        num_blocks: 3,
        out_ch: 112,
        kernel: 5,
        stride: 1,
        expand_ratio: 6,
    },
    Stage {
        num_blocks: 4,
        out_ch: 192,
        kernel: 5,
        stride: 2,
        expand_ratio: 6,
    },
    Stage {
        num_blocks: 1,
        out_ch: 320,
        kernel: 3,
        stride: 1,
        expand_ratio: 6,
    },
];

const EN_B0_LAST_CHANNEL: usize = 1280;

/// EfficientNet-B0 model (torchvision `efficientnet_b0`).
pub struct EfficientNet<T: Float> {
    /// `features.0` — stem ConvBnSiLU (3→32, 3×3, s=2).
    stem: ConvBnSiLU<T>,
    /// `features.<i>` for i ∈ 1..=7 — each is a Sequential of MBConvs.
    /// Stored as a flat Vec<Vec<MBConv>> so we can reproduce torchvision's
    /// `features.<stage>.<j>` indexing.
    stages: Vec<Vec<MBConv<T>>>,
    /// `features.8` — head ConvBnSiLU (320→1280, 1×1).
    head: ConvBnSiLU<T>,
    avgpool: AdaptiveAvgPool2d,
    /// `classifier.1` — final Linear (1280 → num_classes). `classifier.0`
    /// is Dropout (parameter-free, eval-mode pass-through).
    classifier: Linear<T>,
    training: bool,
}

impl<T: Float> EfficientNet<T> {
    /// Construct an EfficientNet-B0 with the given output class count.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let stem = ConvBnSiLU::new(3, 32, 3, 2, 1, 1, true)?;

        let mut stages: Vec<Vec<MBConv<T>>> = Vec::with_capacity(EFFICIENTNET_B0_STAGES.len());
        let mut in_ch = 32_usize;
        for stage_cfg in &EFFICIENTNET_B0_STAGES {
            let mut blocks = Vec::with_capacity(stage_cfg.num_blocks);
            // First block — may change channels and stride.
            blocks.push(MBConv::new(MBConvCfg::from_block_input(
                in_ch,
                stage_cfg.out_ch,
                stage_cfg.kernel,
                stage_cfg.stride,
                stage_cfg.expand_ratio,
            ))?);
            in_ch = stage_cfg.out_ch;
            // Remaining blocks — same channels, stride 1.
            for _ in 1..stage_cfg.num_blocks {
                blocks.push(MBConv::new(MBConvCfg::from_block_input(
                    in_ch,
                    stage_cfg.out_ch,
                    stage_cfg.kernel,
                    1,
                    stage_cfg.expand_ratio,
                ))?);
            }
            stages.push(blocks);
        }
        // Head: 320 → 1280, 1×1 (the 320 comes from stages[6][0].out_ch).
        let head = ConvBnSiLU::new(320, EN_B0_LAST_CHANNEL, 1, 1, 0, 1, true)?;
        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let classifier = Linear::new(EN_B0_LAST_CHANNEL, num_classes, true)?;
        Ok(Self {
            stem,
            stages,
            head,
            avgpool,
            classifier,
            training: true,
        })
    }

    /// Total number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }

    /// Index of the head ConvBnSiLU in the `features` sequential
    /// (= 1 + len(stages)).
    fn head_index(&self) -> usize {
        1 + self.stages.len()
    }
}

impl<T: Float> Module<T> for EfficientNet<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = self.stem.forward(input)?;
        for stage in &self.stages {
            for block in stage {
                x = block.forward(&x)?;
            }
        }
        let x = self.head.forward(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        // classifier.0 = Dropout (parameter-free; eval-mode pass-through).
        // classifier.1 = Linear.
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        for stage in &self.stages {
            for block in stage {
                p.extend(block.parameters());
            }
        }
        p.extend(self.head.parameters());
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters_mut());
        for stage in &mut self.stages {
            for block in stage {
                p.extend(block.parameters_mut());
            }
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
        for (i, stage) in self.stages.iter().enumerate() {
            // features.<i+1>.<j>.{block.<k>...}.<n>.
            for (j, block) in stage.iter().enumerate() {
                for (n, param) in block.named_parameters() {
                    p.push((format!("features.{}.{}.{n}", i + 1, j), param));
                }
            }
        }
        let head_idx = self.head_index();
        for (n, param) in self.head.named_parameters() {
            p.push((format!("features.{head_idx}.{n}"), param));
        }
        for (n, param) in self.classifier.named_parameters() {
            p.push((format!("classifier.1.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = vec![&self.stem];
        for stage in &self.stages {
            for block in stage {
                out.push(block);
            }
        }
        out.push(&self.head);
        out.push(&self.avgpool);
        out.push(&self.classifier);
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> =
            vec![("features.0".to_string(), &self.stem as &dyn Module<T>)];
        for (i, stage) in self.stages.iter().enumerate() {
            for (j, block) in stage.iter().enumerate() {
                out.push((format!("features.{}.{}", i + 1, j), block as &dyn Module<T>));
            }
        }
        let head_idx = self.head_index();
        out.push((format!("features.{head_idx}"), &self.head));
        out.push(("classifier.1".to_string(), &self.classifier));
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.stem.train();
        for stage in &mut self.stages {
            for b in stage {
                b.train();
            }
        }
        self.head.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.stem.eval();
        for stage in &mut self.stages {
            for b in stage {
                b.eval();
            }
        }
        self.head.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

/// Convenience constructor for EfficientNet-B0.
pub fn efficientnet_b0<T: Float>(num_classes: usize) -> FerrotorchResult<EfficientNet<T>> {
    EfficientNet::new(num_classes)
}

// ===========================================================================
// IntermediateFeatures — CL-499 (preserves the public test API)
// ===========================================================================

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for EfficientNet<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();
        let mut x = self.stem.forward(input)?;
        out.insert("stem_conv".to_string(), x.clone());
        let mut flat_idx = 0_usize;
        for stage in &self.stages {
            for block in stage {
                x = block.forward(&x)?;
                out.insert(format!("stage{flat_idx}"), x.clone());
                flat_idx += 1;
            }
        }
        let x = self.head.forward(&x)?;
        out.insert("head_conv".to_string(), x.clone());
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        out.insert("avgpool".to_string(), x.clone());

        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        let logits = self.classifier.forward(&x)?;
        out.insert("fc".to_string(), logits);
        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        let mut names = vec!["stem_conv".to_string()];
        let total: usize = self.stages.iter().map(Vec::len).sum();
        for i in 0..total {
            names.push(format!("stage{i}"));
        }
        names.push("head_conv".to_string());
        names.push("avgpool".to_string());
        names.push("fc".to_string());
        names
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{TensorStorage, no_grad};

    fn leaf_4d(data: &[f32], shape: [usize; 4], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    #[test]
    fn test_efficientnet_b0_named_parameters_match_torchvision_layout() {
        let model = efficientnet_b0::<f32>(1000).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n == "features.0.0.weight"));
        assert!(names.iter().any(|n| n == "features.0.1.weight"));
        // Stage 1 (features.1) — 1 block, expand_ratio=1 → no expand.
        // block.0 = depthwise, block.1 = SE, block.2 = project.
        assert!(names.iter().any(|n| n == "features.1.0.block.0.0.weight"));
        assert!(names.iter().any(|n| n == "features.1.0.block.1.fc1.weight"));
        assert!(names.iter().any(|n| n == "features.1.0.block.1.fc2.weight"));
        assert!(names.iter().any(|n| n == "features.1.0.block.2.0.weight"));
        // Stage 2 (features.2) — expand_ratio=6 → 4 sub-indices.
        assert!(names.iter().any(|n| n == "features.2.0.block.0.0.weight"));
        assert!(names.iter().any(|n| n == "features.2.0.block.1.0.weight"));
        assert!(names.iter().any(|n| n == "features.2.0.block.2.fc1.weight"));
        assert!(names.iter().any(|n| n == "features.2.0.block.3.0.weight"));
        assert!(names.iter().any(|n| n == "features.2.1.block.0.0.weight"));
        // Head.
        assert!(names.iter().any(|n| n == "features.8.0.weight"));
        assert!(names.iter().any(|n| n == "features.8.1.weight"));
        // Classifier.
        assert!(names.iter().any(|n| n == "classifier.1.weight"));
        assert!(names.iter().any(|n| n == "classifier.1.bias"));
    }

    #[test]
    fn test_efficientnet_b0_output_shape() {
        let model = efficientnet_b0::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01_f32; 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    #[test]
    fn test_efficientnet_b0_param_count_in_range() {
        let model = efficientnet_b0::<f32>(1000).unwrap();
        let n = model.num_parameters();
        // torchvision efficientnet_b0 has ~5.29M params (5,288,548).
        assert!(
            (4_900_000..=5_700_000).contains(&n),
            "EfficientNet-B0 param count {n} outside expected ~5.3M range"
        );
    }

    #[test]
    fn test_efficientnet_b0_custom_classes() {
        let model = efficientnet_b0::<f32>(10).unwrap();
        let input = leaf_4d(&vec![0.01_f32; 2 * 3 * 224 * 224], [2, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 10]);
    }

    #[test]
    fn test_efficientnet_train_eval() {
        let mut model = efficientnet_b0::<f32>(10).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }

    #[test]
    fn test_efficientnet_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EfficientNet<f32>>();
    }
}
