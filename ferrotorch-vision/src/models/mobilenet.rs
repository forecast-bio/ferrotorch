//! MobileNetV2 and MobileNetV3-Small architectures (simplified).
//!
//! MobileNetV2 follows Sandler et al. 2018 "Inverted Residuals and Linear
//! Bottlenecks"; MobileNetV3-Small follows Howard et al. 2019 "Searching for
//! MobileNetV3".
//!
//! **Simplifications.** Both real architectures rely on depthwise separable
//! convolutions and, for V3, squeeze-and-excite attention + h-swish. Since
//! `ferrotorch_nn` does not yet provide depthwise convolutions or SE, these
//! implementations approximate the architectures using standard [`Conv2d`]
//! in place of depthwise + pointwise pairs. Channel counts, strides, and
//! block counts are kept faithful to the original designs, so the overall
//! shape of each model matches the paper even if the per-block FLOPs differ.
//! h-swish is approximated by ReLU since the `hard_sigmoid(x + 3)` gate is
//! not yet wired through the autograd ops we rely on here.
//!
//! Both networks use [`AdaptiveAvgPool2d`] + [`Linear`] for the classifier
//! head, matching the existing ResNet/VGG/EfficientNet style in this crate.
//!
//! CL-436.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::AdaptiveAvgPool2d;
use ferrotorch_nn::{Conv2d, Linear};

// ===========================================================================
// Shared helpers
// ===========================================================================

/// Create a Conv2d with square kernel, stride, and padding (no bias).
fn conv<T: Float>(
    in_ch: usize,
    out_ch: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
) -> FerrotorchResult<Conv2d<T>> {
    Conv2d::new(
        in_ch,
        out_ch,
        (kernel, kernel),
        (stride, stride),
        (padding, padding),
        false,
    )
}

// ===========================================================================
// InvertedResidual block
// ===========================================================================

/// A single MobileNetV2 inverted-residual (bottleneck) block.
///
/// Expand → depthwise (approximated by regular 3×3 Conv) → project.
/// Residual skip is applied when `stride == 1` and input channels equal
/// output channels.
pub struct InvertedResidual<T: Float> {
    expand: Option<Conv2d<T>>,
    depthwise: Conv2d<T>,
    project: Conv2d<T>,
    use_residual: bool,
    training: bool,
}

impl<T: Float> InvertedResidual<T> {
    /// Build an inverted-residual block.
    ///
    /// * `in_ch` — input channels.
    /// * `out_ch` — output channels.
    /// * `stride` — depthwise conv stride (1 or 2).
    /// * `expand_ratio` — expansion factor for the 1×1 expand conv
    ///   (set to 1 to skip the expand step, matching the original
    ///   paper's first block).
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        stride: usize,
        expand_ratio: usize,
    ) -> FerrotorchResult<Self> {
        let hidden = in_ch * expand_ratio;
        let expand = if expand_ratio == 1 {
            None
        } else {
            Some(conv(in_ch, hidden, 1, 1, 0)?)
        };
        let depthwise = conv(hidden, hidden, 3, stride, 1)?;
        let project = conv(hidden, out_ch, 1, 1, 0)?;
        let use_residual = stride == 1 && in_ch == out_ch;
        Ok(Self {
            expand,
            depthwise,
            project,
            use_residual,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for InvertedResidual<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = input.clone();
        if let Some(ref expand) = self.expand {
            x = expand.forward(&x)?;
            x = relu(&x)?;
        }
        let x = self.depthwise.forward(&x)?;
        let x = relu(&x)?;
        let x = self.project.forward(&x)?;
        // Note: the real MobileNetV2 uses a *linear* (no-activation)
        // bottleneck projection — we follow that convention here.
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
        p.extend(self.project.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        if let Some(ref mut e) = self.expand {
            p.extend(e.parameters_mut());
        }
        p.extend(self.depthwise.parameters_mut());
        p.extend(self.project.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        if let Some(ref e) = self.expand {
            for (n, param) in e.named_parameters() {
                p.push((format!("expand.{n}"), param));
            }
        }
        for (n, param) in self.depthwise.named_parameters() {
            p.push((format!("depthwise.{n}"), param));
        }
        for (n, param) in self.project.named_parameters() {
            p.push((format!("project.{n}"), param));
        }
        p
    }

    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// MobileNetV2
// ===========================================================================

/// MobileNetV2 stage descriptor: `(expand_ratio, out_channels, num_blocks, stride)`.
struct V2Stage {
    t: usize,
    c: usize,
    n: usize,
    s: usize,
}

/// MobileNetV2 ImageNet configuration (t, c, n, s) from the paper.
const MOBILENETV2_STAGES: [V2Stage; 7] = [
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

/// A simplified MobileNetV2 for ImageNet-style classification.
///
/// Input: `[B, 3, H, W]` (`H == W == 224` is the canonical size).
/// Output: `[B, num_classes]`.
pub struct MobileNetV2<T: Float> {
    stem: Conv2d<T>,
    blocks: Vec<InvertedResidual<T>>,
    last_conv: Conv2d<T>,
    avgpool: AdaptiveAvgPool2d,
    classifier: Linear<T>,
    training: bool,
}

impl<T: Float> MobileNetV2<T> {
    /// Construct a MobileNetV2 with the given number of output classes.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let stem = conv(3, 32, 3, 2, 1)?;
        let mut blocks: Vec<InvertedResidual<T>> = Vec::new();
        let mut in_ch = 32usize;
        for stage in &MOBILENETV2_STAGES {
            for i in 0..stage.n {
                let stride = if i == 0 { stage.s } else { 1 };
                blocks.push(InvertedResidual::new(in_ch, stage.c, stride, stage.t)?);
                in_ch = stage.c;
            }
        }
        let last_conv = conv(in_ch, 1280, 1, 1, 0)?;
        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let classifier = Linear::new(1280, num_classes, true)?;
        Ok(Self {
            stem,
            blocks,
            last_conv,
            avgpool,
            classifier,
            training: true,
        })
    }

    /// Number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

impl<T: Float> Module<T> for MobileNetV2<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.stem.forward(input)?;
        let mut x = relu(&x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.last_conv.forward(&x)?;
        let x = relu(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.last_conv.parameters());
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters_mut());
        for block in &mut self.blocks {
            p.extend(block.parameters_mut());
        }
        p.extend(self.last_conv.parameters_mut());
        p.extend(self.classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.stem.named_parameters() {
            p.push((format!("stem.{n}"), param));
        }
        for (i, block) in self.blocks.iter().enumerate() {
            for (n, param) in block.named_parameters() {
                p.push((format!("blocks.{i}.{n}"), param));
            }
        }
        for (n, param) in self.last_conv.named_parameters() {
            p.push((format!("last_conv.{n}"), param));
        }
        for (n, param) in self.classifier.named_parameters() {
            p.push((format!("classifier.{n}"), param));
        }
        p
    }

    fn train(&mut self) {
        self.training = true;
        for b in self.blocks.iter_mut() {
            b.train();
        }
    }
    fn eval(&mut self) {
        self.training = false;
        for b in self.blocks.iter_mut() {
            b.eval();
        }
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
// IntermediateFeatures — CL-499
// ---------------------------------------------------------------------------

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for MobileNetV2<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();

        // Stem.
        let x = self.stem.forward(input)?;
        let mut x = relu(&x)?;
        out.insert("stem".to_string(), x.clone());

        // Each inverted-residual block gets its own named output.
        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            out.insert(format!("block{i}"), x.clone());
        }

        // Last conv + pool + classifier.
        let x = self.last_conv.forward(&x)?;
        let x = relu(&x)?;
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

/// MobileNetV3-Small stage descriptor.
///
/// `(out_channels, kernel, stride)`. The real architecture varies
/// expansion ratio, h-swish vs. ReLU, and squeeze-excite per block;
/// this simplified implementation uses a fixed expansion ratio of 3
/// and ReLU throughout so it builds on the existing primitives.
struct V3Stage {
    out_ch: usize,
    kernel: usize,
    stride: usize,
}

const MOBILENETV3_SMALL_STAGES: [V3Stage; 11] = [
    V3Stage {
        out_ch: 16,
        kernel: 3,
        stride: 2,
    },
    V3Stage {
        out_ch: 24,
        kernel: 3,
        stride: 2,
    },
    V3Stage {
        out_ch: 24,
        kernel: 3,
        stride: 1,
    },
    V3Stage {
        out_ch: 40,
        kernel: 5,
        stride: 2,
    },
    V3Stage {
        out_ch: 40,
        kernel: 5,
        stride: 1,
    },
    V3Stage {
        out_ch: 40,
        kernel: 5,
        stride: 1,
    },
    V3Stage {
        out_ch: 48,
        kernel: 5,
        stride: 1,
    },
    V3Stage {
        out_ch: 48,
        kernel: 5,
        stride: 1,
    },
    V3Stage {
        out_ch: 96,
        kernel: 5,
        stride: 2,
    },
    V3Stage {
        out_ch: 96,
        kernel: 5,
        stride: 1,
    },
    V3Stage {
        out_ch: 96,
        kernel: 5,
        stride: 1,
    },
];

/// A single MobileNetV3 block — Conv + ReLU with optional residual.
///
/// Stand-in for the real inverted-residual-with-SE block. Matches the
/// pattern used for EfficientNet's `ConvBlock` but exposed under a
/// distinct name so V3 can evolve independently.
pub struct V3Block<T: Float> {
    conv: Conv2d<T>,
    use_residual: bool,
    training: bool,
}

impl<T: Float> V3Block<T> {
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
    ) -> FerrotorchResult<Self> {
        let padding = kernel / 2;
        Ok(Self {
            conv: conv(in_ch, out_ch, kernel, stride, padding)?,
            use_residual: stride == 1 && in_ch == out_ch,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for V3Block<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let out = self.conv.forward(input)?;
        let out = relu(&out)?;
        if self.use_residual {
            add(&out, input)
        } else {
            Ok(out)
        }
    }
    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.conv.parameters()
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.conv.parameters_mut()
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.conv.named_parameters()
    }
    fn train(&mut self) {
        self.training = true;
    }
    fn eval(&mut self) {
        self.training = false;
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

/// A simplified MobileNetV3-Small for ImageNet-style classification.
pub struct MobileNetV3Small<T: Float> {
    stem: Conv2d<T>,
    blocks: Vec<V3Block<T>>,
    last_conv: Conv2d<T>,
    avgpool: AdaptiveAvgPool2d,
    classifier: Linear<T>,
    training: bool,
}

impl<T: Float> MobileNetV3Small<T> {
    /// Construct a MobileNetV3-Small with the given number of output classes.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let stem = conv(3, 16, 3, 2, 1)?;
        let mut blocks: Vec<V3Block<T>> = Vec::new();
        let mut in_ch = 16usize;
        for stage in &MOBILENETV3_SMALL_STAGES {
            blocks.push(V3Block::new(
                in_ch,
                stage.out_ch,
                stage.kernel,
                stage.stride,
            )?);
            in_ch = stage.out_ch;
        }
        let last_conv = conv(in_ch, 576, 1, 1, 0)?;
        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let classifier = Linear::new(576, num_classes, true)?;
        Ok(Self {
            stem,
            blocks,
            last_conv,
            avgpool,
            classifier,
            training: true,
        })
    }

    /// Number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

impl<T: Float> Module<T> for MobileNetV3Small<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.stem.forward(input)?;
        let mut x = relu(&x)?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        let x = self.last_conv.forward(&x)?;
        let x = relu(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        for block in &self.blocks {
            p.extend(block.parameters());
        }
        p.extend(self.last_conv.parameters());
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters_mut());
        for block in &mut self.blocks {
            p.extend(block.parameters_mut());
        }
        p.extend(self.last_conv.parameters_mut());
        p.extend(self.classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.stem.named_parameters() {
            p.push((format!("stem.{n}"), param));
        }
        for (i, block) in self.blocks.iter().enumerate() {
            for (n, param) in block.named_parameters() {
                p.push((format!("blocks.{i}.{n}"), param));
            }
        }
        for (n, param) in self.last_conv.named_parameters() {
            p.push((format!("last_conv.{n}"), param));
        }
        for (n, param) in self.classifier.named_parameters() {
            p.push((format!("classifier.{n}"), param));
        }
        p
    }

    fn train(&mut self) {
        self.training = true;
        for b in self.blocks.iter_mut() {
            b.train();
        }
    }
    fn eval(&mut self) {
        self.training = false;
        for b in self.blocks.iter_mut() {
            b.eval();
        }
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

        let x = self.stem.forward(input)?;
        let mut x = relu(&x)?;
        out.insert("stem".to_string(), x.clone());

        for (i, block) in self.blocks.iter().enumerate() {
            x = block.forward(&x)?;
            out.insert(format!("block{i}"), x.clone());
        }

        let x = self.last_conv.forward(&x)?;
        let x = relu(&x)?;
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
    fn test_mobilenet_v2_output_shape() {
        let model: MobileNetV2<f32> = mobilenet_v2(10).unwrap();
        let x = dummy_image(2, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10]);
    }

    #[test]
    fn test_mobilenet_v2_custom_classes() {
        let model: MobileNetV2<f32> = mobilenet_v2(3).unwrap();
        let x = dummy_image(1, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
    }

    #[test]
    fn test_mobilenet_v2_param_count() {
        let model: MobileNetV2<f32> = mobilenet_v2(1000).unwrap();
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_mobilenet_v2_named_parameters_prefixes() {
        let model: MobileNetV2<f32> = mobilenet_v2(10).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("stem.")));
        assert!(names.iter().any(|n| n.starts_with("blocks.0.")));
        assert!(names.iter().any(|n| n.starts_with("last_conv.")));
        assert!(names.iter().any(|n| n.starts_with("classifier.")));
    }

    #[test]
    fn test_mobilenet_v2_train_eval() {
        let mut model: MobileNetV2<f32> = mobilenet_v2(10).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }

    #[test]
    fn test_inverted_residual_expand_ratio_1_has_no_expand_conv() {
        let block: InvertedResidual<f32> = InvertedResidual::new(16, 16, 1, 1).unwrap();
        assert!(block.expand.is_none());
    }

    #[test]
    fn test_inverted_residual_expand_ratio_gt_1_has_expand_conv() {
        let block: InvertedResidual<f32> = InvertedResidual::new(16, 24, 2, 6).unwrap();
        assert!(block.expand.is_some());
    }

    #[test]
    fn test_mobilenet_v3_small_output_shape() {
        let model: MobileNetV3Small<f32> = mobilenet_v3_small(10).unwrap();
        let x = dummy_image(2, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[2, 10]);
    }

    #[test]
    fn test_mobilenet_v3_small_custom_classes() {
        let model: MobileNetV3Small<f32> = mobilenet_v3_small(5).unwrap();
        let x = dummy_image(1, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 5]);
    }

    #[test]
    fn test_mobilenet_v3_small_param_count() {
        let model: MobileNetV3Small<f32> = mobilenet_v3_small(1000).unwrap();
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_mobilenet_v3_small_named_parameters_prefixes() {
        let model: MobileNetV3Small<f32> = mobilenet_v3_small(10).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("stem.")));
        assert!(names.iter().any(|n| n.starts_with("blocks.0.")));
        assert!(names.iter().any(|n| n.starts_with("last_conv.")));
        assert!(names.iter().any(|n| n.starts_with("classifier.")));
    }

    #[test]
    fn test_mobilenet_v3_small_train_eval() {
        let mut model: MobileNetV3Small<f32> = mobilenet_v3_small(10).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }
}
