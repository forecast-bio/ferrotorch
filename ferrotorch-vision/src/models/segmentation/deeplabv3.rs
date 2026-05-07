//! DeepLabV3 with ResNet-50 backbone.
//!
//! Mirrors `torchvision.models.segmentation.deeplabv3_resnet50` (torchvision 0.21.x).
//!
//! ## Architecture
//!
//! ```text
//! image [B, 3, H, W]
//!   └─ ResNet-50 backbone (dilated)
//!        layer3: dilation=2  (output stride 16 total from input)
//!        layer4: dilation=4  (output stride 16 — stride=1, keeps spatial dims)
//!         └─ features [B, 2048, H/16, W/16]
//!              └─ ASPP head → [B, 256, H/16, W/16]
//!                   └─ 1×1 classifier → [B, num_classes, H/16, W/16]
//!                        └─ bilinear upsample → [B, num_classes, H, W]
//! ```
//!
//! The `replace_stride_with_dilation` flag mirrors torchvision's
//! `[False, True, True]` (layer1 unchanged, layer2 stride=2 unchanged, layer3
//! and layer4 get dilation instead of stride).
//!
//! ## Reference
//! Chen et al., "Rethinking Atrous Convolution for Semantic Image Segmentation",
//! arXiv:1706.05587. torchvision 0.21.x `deeplabv3_resnet50(weights=None,
//! num_classes=21)`.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::upsample::{InterpolateMode, interpolate};
use ferrotorch_nn::Conv2d;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::MaxPool2d;

use super::aspp::{Aspp, DilatedConv2d};

// ---------------------------------------------------------------------------
// Helper: standard conv helpers (mirrors resnet.rs helpers)
// ---------------------------------------------------------------------------

fn conv1x1<T: Float>(
    in_planes: usize,
    out_planes: usize,
    stride: usize,
) -> FerrotorchResult<Conv2d<T>> {
    Conv2d::new(in_planes, out_planes, (1, 1), (stride, stride), (0, 0), false)
}

// ---------------------------------------------------------------------------
// DilatedBottleneck — Bottleneck with dilated 3×3 conv for DeepLabV3
// ---------------------------------------------------------------------------

/// Bottleneck block where the 3×3 conv uses dilation > 1.
///
/// Used in ResNet-50 layer3 (dilation=2) and layer4 (dilation=4) for
/// DeepLabV3. Stride is always 1 (spatial resolution is preserved by
/// replacing stride with dilation, matching torchvision's
/// `replace_stride_with_dilation=[False, True, True]`).
struct DilatedBottleneck<T: Float> {
    conv1: Conv2d<T>,
    bn1: BatchNorm2d<T>,
    conv2_dilated: DilatedConv2d<T>,
    conv3: Conv2d<T>,
    bn3: BatchNorm2d<T>,
    downsample: Option<(Conv2d<T>, BatchNorm2d<T>)>,
    training: bool,
}

const EXPANSION: usize = 4;

impl<T: Float> DilatedBottleneck<T> {
    /// Create a dilated bottleneck.
    ///
    /// `dilation` is applied to the 3×3 middle conv. Stride is always 1.
    fn new(
        in_planes: usize,
        planes: usize,
        dilation: usize,
    ) -> FerrotorchResult<Self> {
        let conv1 = conv1x1(in_planes, planes, 1)?;
        let bn1 = BatchNorm2d::new(planes, 1e-5, 0.1, true)?;
        let conv2_dilated = DilatedConv2d::new(planes, planes, dilation)?;
        let conv3 = conv1x1(planes, planes * EXPANSION, 1)?;
        let bn3 = BatchNorm2d::new(planes * EXPANSION, 1e-5, 0.1, true)?;

        let downsample = if in_planes == planes * EXPANSION {
            None
        } else {
            let ds_conv = conv1x1(in_planes, planes * EXPANSION, 1)?;
            let ds_bn = BatchNorm2d::new(planes * EXPANSION, 1e-5, 0.1, true)?;
            Some((ds_conv, ds_bn))
        };

        Ok(Self {
            conv1,
            bn1,
            conv2_dilated,
            conv3,
            bn3,
            downsample,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for DilatedBottleneck<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let out = self.conv1.forward(input)?;
        let out = Module::<T>::forward(&self.bn1, &out)?;
        let out = relu(&out)?;

        // Dilated 3×3.
        let out = self.conv2_dilated.forward(&out)?;
        // Note: DilatedConv2d.forward already applies BN + ReLU.

        let out = self.conv3.forward(&out)?;
        let out = Module::<T>::forward(&self.bn3, &out)?;

        let identity = match &self.downsample {
            Some((ds_conv, ds_bn)) => {
                let x = ds_conv.forward(input)?;
                Module::<T>::forward(ds_bn, &x)?
            }
            None => input.clone(),
        };

        let out = add(&out, &identity)?;
        relu(&out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(self.conv2_dilated.parameters());
        p.extend(self.conv3.parameters());
        p.extend(self.bn3.parameters());
        if let Some((ref c, ref b)) = self.downsample {
            p.extend(c.parameters());
            p.extend(b.parameters());
        }
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters_mut());
        p.extend(self.bn1.parameters_mut());
        p.extend(self.conv2_dilated.parameters_mut());
        p.extend(self.conv3.parameters_mut());
        p.extend(self.bn3.parameters_mut());
        if let Some((ref mut c, ref mut b)) = self.downsample {
            p.extend(c.parameters_mut());
            p.extend(b.parameters_mut());
        }
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.conv1.named_parameters() {
            out.push((format!("conv1.{k}"), v));
        }
        for (k, v) in self.bn1.named_parameters() {
            out.push((format!("bn1.{k}"), v));
        }
        for (k, v) in self.conv2_dilated.named_parameters() {
            out.push((format!("conv2.{k}"), v));
        }
        for (k, v) in self.conv3.named_parameters() {
            out.push((format!("conv3.{k}"), v));
        }
        for (k, v) in self.bn3.named_parameters() {
            out.push((format!("bn3.{k}"), v));
        }
        if let Some((ref c, ref b)) = self.downsample {
            for (k, v) in c.named_parameters() {
                out.push((format!("downsample.0.{k}"), v));
            }
            for (k, v) in b.named_parameters() {
                out.push((format!("downsample.1.{k}"), v));
            }
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.bn1.train();
        self.conv2_dilated.train();
        self.bn3.train();
        if let Some((_, ref mut b)) = self.downsample {
            b.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        self.bn1.eval();
        self.conv2_dilated.eval();
        self.bn3.eval();
        if let Some((_, ref mut b)) = self.downsample {
            b.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// ResNet50Dilated — ResNet-50 backbone with dilated layer3 + layer4
// ---------------------------------------------------------------------------

/// ResNet-50 backbone with dilated convolutions in layer3 and layer4.
///
/// Matches torchvision's `resnet50(replace_stride_with_dilation=[False, True, True])`:
/// - layer1: standard stride=1 bottlenecks, output [B, 256, H/4, W/4]
/// - layer2: standard stride=2 bottlenecks, output [B, 512, H/8, W/8]
/// - layer3: dilation=2 (stride=1), output [B, 1024, H/16, W/16]
/// - layer4: dilation=4 (stride=1), output [B, 2048, H/16, W/16]
///
/// layer3 and layer4 keep the same spatial resolution as layer2 because
/// stride is replaced by dilation. The output stride is effectively 16.
pub struct ResNet50Dilated<T: Float> {
    // Stem
    conv1: Conv2d<T>,
    bn1: BatchNorm2d<T>,
    maxpool: MaxPool2d,

    // Residual stages (standard Bottleneck for layer1/2)
    layer1: Vec<Box<dyn Module<T>>>,
    layer2: Vec<Box<dyn Module<T>>>,

    // Dilated stages
    layer3: Vec<Box<dyn Module<T>>>,
    layer4: Vec<Box<dyn Module<T>>>,

    training: bool,
}

impl<T: Float> ResNet50Dilated<T> {
    /// Build a dilated ResNet-50 backbone (no classifier head).
    pub fn new() -> FerrotorchResult<Self> {
        let conv1 = Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), false)?;
        let bn1 = BatchNorm2d::new(64, 1e-5, 0.1, true)?;
        let maxpool = MaxPool2d::new([3, 3], [2, 2], [1, 1]);

        // Layer 1: 3 standard bottleneck blocks (in=64 → out=256)
        let layer1 = make_bottleneck_layer::<T>(64, 64, 3, 1)?;
        // Layer 2: 4 standard bottleneck blocks (in=256 → out=512, stride=2)
        let layer2 = make_bottleneck_layer::<T>(256, 128, 4, 2)?;
        // Layer 3: 6 dilated bottleneck blocks (dilation=2, stride=1)
        let layer3 = make_dilated_layer::<T>(512, 256, 6, 2)?;
        // Layer 4: 3 dilated bottleneck blocks (dilation=4, stride=1)
        let layer4 = make_dilated_layer::<T>(1024, 512, 3, 4)?;

        Ok(Self {
            conv1,
            bn1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            training: true,
        })
    }

    /// Extract layer4 features: `[B, 2048, H/16, W/16]`.
    pub fn forward_layer4(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.conv1.forward(input)?;
        let x = Module::<T>::forward(&self.bn1, &x)?;
        let x = relu(&x)?;
        let x = Module::<T>::forward(&self.maxpool, &x)?;

        let x = run_layer(&self.layer1, &x)?;
        let x = run_layer(&self.layer2, &x)?;
        let x = run_layer(&self.layer3, &x)?;
        run_layer(&self.layer4, &x)
    }
}

impl<T: Float> Module<T> for ResNet50Dilated<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // For use as a pure feature extractor: return layer4 activations.
        // (DeepLabV3 does not use the avgpool/fc head.)
        self.forward_layer4(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.bn1.parameters());
        p.extend(collect_params(&self.layer1));
        p.extend(collect_params(&self.layer2));
        p.extend(collect_params(&self.layer3));
        p.extend(collect_params(&self.layer4));
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters_mut());
        p.extend(self.bn1.parameters_mut());
        p.extend(collect_params_mut(&mut self.layer1));
        p.extend(collect_params_mut(&mut self.layer2));
        p.extend(collect_params_mut(&mut self.layer3));
        p.extend(collect_params_mut(&mut self.layer4));
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.conv1.named_parameters() {
            out.push((format!("backbone.conv1.{k}"), v));
        }
        for (k, v) in self.bn1.named_parameters() {
            out.push((format!("backbone.bn1.{k}"), v));
        }
        named_layer_params(&self.layer1, "backbone.layer1", &mut out);
        named_layer_params(&self.layer2, "backbone.layer2", &mut out);
        named_layer_params(&self.layer3, "backbone.layer3", &mut out);
        named_layer_params(&self.layer4, "backbone.layer4", &mut out);
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.bn1.train();
        for b in &mut self.layer1 {
            b.train();
        }
        for b in &mut self.layer2 {
            b.train();
        }
        for b in &mut self.layer3 {
            b.train();
        }
        for b in &mut self.layer4 {
            b.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        self.bn1.eval();
        for b in &mut self.layer1 {
            b.eval();
        }
        for b in &mut self.layer2 {
            b.eval();
        }
        for b in &mut self.layer3 {
            b.eval();
        }
        for b in &mut self.layer4 {
            b.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Layer builders
// ---------------------------------------------------------------------------

/// Build a stage of standard Bottleneck blocks.
fn make_bottleneck_layer<T: Float>(
    in_planes: usize,
    planes: usize,
    num_blocks: usize,
    stride: usize,
) -> FerrotorchResult<Vec<Box<dyn Module<T>>>> {
    use crate::models::resnet::Bottleneck;

    let mut blocks: Vec<Box<dyn Module<T>>> = Vec::with_capacity(num_blocks);
    blocks.push(Box::new(Bottleneck::new(in_planes, planes, stride)?));
    let out_planes = planes * Bottleneck::<T>::EXPANSION;
    for _ in 1..num_blocks {
        blocks.push(Box::new(Bottleneck::new(out_planes, planes, 1)?));
    }
    Ok(blocks)
}

/// Build a stage of DilatedBottleneck blocks (stride=1, all at same dilation).
fn make_dilated_layer<T: Float>(
    in_planes: usize,
    planes: usize,
    num_blocks: usize,
    dilation: usize,
) -> FerrotorchResult<Vec<Box<dyn Module<T>>>> {
    let mut blocks: Vec<Box<dyn Module<T>>> = Vec::with_capacity(num_blocks);
    // First block may need a downsample projection if channels change.
    blocks.push(Box::new(DilatedBottleneck::new(in_planes, planes, dilation)?));
    let out_planes = planes * EXPANSION;
    for _ in 1..num_blocks {
        blocks.push(Box::new(DilatedBottleneck::new(out_planes, planes, dilation)?));
    }
    Ok(blocks)
}

// ---------------------------------------------------------------------------
// Layer iteration helpers
// ---------------------------------------------------------------------------

fn run_layer<T: Float>(
    blocks: &[Box<dyn Module<T>>],
    input: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let mut x = blocks[0].forward(input)?;
    for b in &blocks[1..] {
        x = b.forward(&x)?;
    }
    Ok(x)
}

fn collect_params<T: Float>(blocks: &[Box<dyn Module<T>>]) -> Vec<&Parameter<T>> {
    blocks.iter().flat_map(|b| b.parameters()).collect()
}

fn collect_params_mut<T: Float>(blocks: &mut [Box<dyn Module<T>>]) -> Vec<&mut Parameter<T>> {
    blocks.iter_mut().flat_map(|b| b.parameters_mut()).collect()
}

fn named_layer_params<'a, T: Float>(
    blocks: &'a [Box<dyn Module<T>>],
    prefix: &str,
    out: &mut Vec<(String, &'a Parameter<T>)>,
) {
    for (i, block) in blocks.iter().enumerate() {
        for (k, v) in block.named_parameters() {
            out.push((format!("{prefix}.{i}.{k}"), v));
        }
    }
}

// ---------------------------------------------------------------------------
// DeepLabV3Head
// ---------------------------------------------------------------------------

/// DeepLabV3 segmentation head: ASPP + classifier.
///
/// Input: `[B, 2048, H', W']`
/// Output: `[B, num_classes, H', W']`
pub struct DeepLabV3Head<T: Float> {
    aspp: Aspp<T>,
    classifier: Conv2d<T>,
    training: bool,
}

impl<T: Float> DeepLabV3Head<T> {
    /// Construct a DeepLabV3 head.
    ///
    /// * `in_channels` — backbone feature channels (2048).
    /// * `num_classes` — number of segmentation classes.
    pub fn new(in_channels: usize, num_classes: usize) -> FerrotorchResult<Self> {
        let aspp = Aspp::new(in_channels, 256)?;
        let classifier = Conv2d::new(256, num_classes, (1, 1), (1, 1), (0, 0), false)?;
        Ok(Self {
            aspp,
            classifier,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for DeepLabV3Head<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.aspp.forward(input)?;
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.aspp.parameters();
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.aspp.parameters_mut();
        p.extend(self.classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.aspp.named_parameters() {
            out.push((format!("aspp.{k}"), v));
        }
        for (k, v) in self.classifier.named_parameters() {
            out.push((format!("classifier.{k}"), v));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.aspp.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.aspp.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// DeepLabV3
// ---------------------------------------------------------------------------

/// DeepLabV3 semantic segmentation model with ResNet-50 backbone.
///
/// Output shape: `[B, num_classes, H, W]` — same spatial size as input.
///
/// ## Usage
///
/// ```ignore
/// use ferrotorch_vision::models::segmentation::DeepLabV3;
/// let model = deeplabv3_resnet50::<f32>(21).unwrap();
/// let x = ferrotorch_core::randn(&[1, 3, 512, 512]).unwrap();
/// let logits = model.forward(&x).unwrap(); // [1, 21, 512, 512]
/// ```
pub struct DeepLabV3<T: Float> {
    backbone: ResNet50Dilated<T>,
    head: DeepLabV3Head<T>,
    training: bool,
}

impl<T: Float> DeepLabV3<T> {
    /// Construct a DeepLabV3 model.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let backbone = ResNet50Dilated::new()?;
        let head = DeepLabV3Head::new(2048, num_classes)?;
        Ok(Self {
            backbone,
            head,
            training: false, // torchvision starts in eval mode
        })
    }

    /// Total learnable scalar parameters.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

impl<T: Float> Module<T> for DeepLabV3<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let h_in = input.shape()[2];
        let w_in = input.shape()[3];

        // Backbone: [B, 2048, H/16, W/16]
        let features = self.backbone.forward_layer4(input)?;

        // Head: [B, num_classes, H/16, W/16]
        let logits = self.head.forward(&features)?;

        // Upsample to input resolution: [B, num_classes, H, W]
        interpolate(
            &logits,
            Some([h_in, w_in]),
            None,
            InterpolateMode::Bilinear,
            false,
        )
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.backbone.parameters();
        p.extend(self.head.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.backbone.parameters_mut();
        p.extend(self.head.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = Vec::new();
        for (k, v) in self.backbone.named_parameters() {
            out.push((k, v));
        }
        for (k, v) in self.head.named_parameters() {
            out.push((format!("head.{k}"), v));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.backbone.train();
        self.head.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.backbone.eval();
        self.head.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

/// Build a DeepLabV3 segmentation model with ResNet-50 backbone.
///
/// Mirrors `torchvision.models.segmentation.deeplabv3_resnet50(weights=None,
/// num_classes=21)`.
///
/// No pretrained weights are loaded. Use `num_classes=21` for Pascal VOC,
/// `num_classes=21` for Cityscapes (commonly), or another value as needed.
pub fn deeplabv3_resnet50<T: Float>(num_classes: usize) -> FerrotorchResult<DeepLabV3<T>> {
    DeepLabV3::new(num_classes)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::no_grad;

    fn tiny_rgb(b: usize, h: usize, w: usize) -> Tensor<f32> {
        use ferrotorch_core::randn;
        no_grad(|| randn(&[b, 3, h, w]).unwrap())
    }

    #[test]
    fn test_deeplabv3_output_shape_small() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let x = tiny_rgb(1, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 21, 32, 32]);
    }

    #[test]
    fn test_deeplabv3_output_shape_64x64() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let x = tiny_rgb(1, 64, 64);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 21, 64, 64]);
    }

    #[test]
    fn test_deeplabv3_batch_size_2() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let x = tiny_rgb(2, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[2, 21, 32, 32]);
    }

    #[test]
    fn test_deeplabv3_custom_num_classes() {
        let model = deeplabv3_resnet50::<f32>(5).unwrap();
        let x = tiny_rgb(1, 32, 32);
        let y = no_grad(|| model.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 5, 32, 32]);
    }

    #[test]
    fn test_deeplabv3_named_parameter_prefixes() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let names: Vec<String> = model.named_parameters().into_iter().map(|(n, _)| n).collect();
        assert!(names.iter().any(|n| n.starts_with("backbone.")));
        assert!(names.iter().any(|n| n.starts_with("head.")));
    }

    #[test]
    fn test_deeplabv3_param_count_sanity() {
        let model = deeplabv3_resnet50::<f32>(21).unwrap();
        let np = model.num_parameters();
        // torchvision deeplabv3_resnet50 is ~39.6M; we expect a comparable range.
        assert!(np > 30_000_000, "DeepLabV3 params too low: {np}");
    }

    #[test]
    fn test_deeplabv3_train_eval_toggle() {
        let mut model = deeplabv3_resnet50::<f32>(21).unwrap();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
    }
}
