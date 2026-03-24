//! ResNet architectures: ResNet-18, ResNet-34, and ResNet-50.
//!
//! Follows the original paper: "Deep Residual Learning for Image Recognition"
//! (He et al., 2015). This implementation omits BatchNorm2d (not yet available
//! in `ferrotorch_nn`) -- batch normalization can be slotted in later without
//! changing the overall architecture.
//!
//! All convolutions use Kaiming-initialized weights. The residual `add` uses
//! `ferrotorch_core::grad_fns::arithmetic::add` so gradients flow through
//! skip connections automatically.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::Conv2d;
use ferrotorch_nn::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::{AdaptiveAvgPool2d, MaxPool2d};

// ===========================================================================
// Helpers
// ===========================================================================

/// Create a 3x3 conv with given stride and padding=1 (same convolution).
fn conv3x3<T: Float>(
    in_planes: usize,
    out_planes: usize,
    stride: usize,
) -> FerrotorchResult<Conv2d<T>> {
    Conv2d::new(
        in_planes,
        out_planes,
        (3, 3),
        (stride, stride),
        (1, 1),
        false,
    )
}

/// Create a 1x1 conv with given stride and no padding.
fn conv1x1<T: Float>(
    in_planes: usize,
    out_planes: usize,
    stride: usize,
) -> FerrotorchResult<Conv2d<T>> {
    Conv2d::new(
        in_planes,
        out_planes,
        (1, 1),
        (stride, stride),
        (0, 0),
        false,
    )
}

// ===========================================================================
// BasicBlock (for ResNet-18 / ResNet-34)
// ===========================================================================

/// A basic residual block with two 3x3 convolutions.
///
/// ```text
/// x -----> conv1 -> relu -> conv2 -----> (+) -> relu -> out
/// |                                       ^
/// +---------- [downsample] ---------------+
/// ```
///
/// The optional `downsample` 1x1 convolution is used when the spatial size
/// or channel count changes between the input and the output.
pub struct BasicBlock<T: Float> {
    conv1: Conv2d<T>,
    conv2: Conv2d<T>,
    downsample: Option<Conv2d<T>>,
    training: bool,
}

impl<T: Float> BasicBlock<T> {
    /// Expansion factor for BasicBlock (channels are not expanded).
    pub const EXPANSION: usize = 1;

    /// Create a new `BasicBlock`.
    ///
    /// * `in_planes`  -- number of input channels.
    /// * `planes`     -- number of output channels (before expansion, but
    ///   expansion=1 for BasicBlock so this is also the final channel count).
    /// * `stride`     -- spatial stride for the first 3x3 conv. Use 2 for
    ///   downsampling layers.
    pub fn new(in_planes: usize, planes: usize, stride: usize) -> FerrotorchResult<Self> {
        let conv1 = conv3x3(in_planes, planes, stride)?;
        let conv2 = conv3x3(planes, planes, 1)?;

        let downsample = if stride != 1 || in_planes != planes * Self::EXPANSION {
            Some(conv1x1(in_planes, planes * Self::EXPANSION, stride)?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            conv2,
            downsample,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for BasicBlock<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Main path.
        let out = self.conv1.forward(input)?;
        let out = relu(&out)?;
        let out = self.conv2.forward(&out)?;

        // Skip connection.
        let identity = match &self.downsample {
            Some(ds) => ds.forward(input)?,
            None => input.clone(),
        };

        // Residual addition (differentiable).
        let out = add(&out, &identity)?;
        relu(&out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        if let Some(ref ds) = self.downsample {
            params.extend(ds.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        if let Some(ref mut ds) = self.downsample {
            params.extend(ds.parameters_mut());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.conv1.named_parameters() {
            params.push((format!("conv1.{name}"), p));
        }
        for (name, p) in self.conv2.named_parameters() {
            params.push((format!("conv2.{name}"), p));
        }
        if let Some(ref ds) = self.downsample {
            for (name, p) in ds.named_parameters() {
                params.push((format!("downsample.{name}"), p));
            }
        }
        params
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
// Bottleneck (for ResNet-50)
// ===========================================================================

/// A bottleneck residual block with 1x1 -> 3x3 -> 1x1 convolutions.
///
/// ```text
/// x --> conv1 (1x1) -> relu -> conv2 (3x3) -> relu -> conv3 (1x1) --> (+) -> relu -> out
/// |                                                                     ^
/// +----------------------- [downsample] --------------------------------+
/// ```
///
/// The expansion factor is 4: `conv3` outputs `planes * 4` channels.
pub struct Bottleneck<T: Float> {
    conv1: Conv2d<T>,
    conv2: Conv2d<T>,
    conv3: Conv2d<T>,
    downsample: Option<Conv2d<T>>,
    training: bool,
}

impl<T: Float> Bottleneck<T> {
    /// Expansion factor for Bottleneck.
    pub const EXPANSION: usize = 4;

    /// Create a new `Bottleneck`.
    ///
    /// * `in_planes`  -- number of input channels.
    /// * `planes`     -- bottleneck width (intermediate channel count).
    /// * `stride`     -- spatial stride for the 3x3 conv.
    pub fn new(in_planes: usize, planes: usize, stride: usize) -> FerrotorchResult<Self> {
        let width = planes; // No group convolution, so width = planes.

        let conv1 = conv1x1(in_planes, width, 1)?;
        let conv2 = conv3x3(width, width, stride)?;
        let conv3 = conv1x1(width, planes * Self::EXPANSION, 1)?;

        let downsample = if stride != 1 || in_planes != planes * Self::EXPANSION {
            Some(conv1x1(in_planes, planes * Self::EXPANSION, stride)?)
        } else {
            None
        };

        Ok(Self {
            conv1,
            conv2,
            conv3,
            downsample,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Bottleneck<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // 1x1 reduce.
        let out = self.conv1.forward(input)?;
        let out = relu(&out)?;

        // 3x3 process.
        let out = self.conv2.forward(&out)?;
        let out = relu(&out)?;

        // 1x1 expand.
        let out = self.conv3.forward(&out)?;

        // Skip connection.
        let identity = match &self.downsample {
            Some(ds) => ds.forward(input)?,
            None => input.clone(),
        };

        let out = add(&out, &identity)?;
        relu(&out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.conv3.parameters());
        if let Some(ref ds) = self.downsample {
            params.extend(ds.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.conv3.parameters_mut());
        if let Some(ref mut ds) = self.downsample {
            params.extend(ds.parameters_mut());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.conv1.named_parameters() {
            params.push((format!("conv1.{name}"), p));
        }
        for (name, p) in self.conv2.named_parameters() {
            params.push((format!("conv2.{name}"), p));
        }
        for (name, p) in self.conv3.named_parameters() {
            params.push((format!("conv3.{name}"), p));
        }
        if let Some(ref ds) = self.downsample {
            for (name, p) in ds.named_parameters() {
                params.push((format!("downsample.{name}"), p));
            }
        }
        params
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
// ResNet
// ===========================================================================

/// A ResNet model.
///
/// The architecture follows the standard ResNet design:
///
/// 1. 7x7 conv, stride 2 (initial feature extraction)
/// 2. 3x3 max pool, stride 2
/// 3. Four residual layers (layer1..layer4)
/// 4. Adaptive average pool to (1, 1)
/// 5. Fully connected classifier
///
/// Batch normalization is omitted (not yet in `ferrotorch_nn`).
pub struct ResNet<T: Float> {
    // Stem.
    conv1: Conv2d<T>,
    maxpool: MaxPool2d,

    // Residual layers (stored as trait objects for uniformity).
    layer1: Vec<Box<dyn Module<T>>>,
    layer2: Vec<Box<dyn Module<T>>>,
    layer3: Vec<Box<dyn Module<T>>>,
    layer4: Vec<Box<dyn Module<T>>>,

    // Head.
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<T>,

    training: bool,
}

impl<T: Float> ResNet<T> {
    /// Build a ResNet from BasicBlocks.
    ///
    /// `layers` is `[n1, n2, n3, n4]` giving the number of blocks in each
    /// of the four stages.
    fn from_basic(layers: [usize; 4], num_classes: usize) -> FerrotorchResult<Self> {
        let conv1 = Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), false)?;
        let maxpool = MaxPool2d::new([3, 3], [2, 2], [1, 1]);

        let layer1 = Self::make_basic_layer(64, 64, layers[0], 1)?;
        let layer2 = Self::make_basic_layer(64, 128, layers[1], 2)?;
        let layer3 = Self::make_basic_layer(128, 256, layers[2], 2)?;
        let layer4 = Self::make_basic_layer(256, 512, layers[3], 2)?;

        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let fc = Linear::new(512 * BasicBlock::<T>::EXPANSION, num_classes, true)?;

        Ok(Self {
            conv1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
            training: true,
        })
    }

    /// Build a ResNet from Bottleneck blocks.
    fn from_bottleneck(layers: [usize; 4], num_classes: usize) -> FerrotorchResult<Self> {
        let conv1 = Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), false)?;
        let maxpool = MaxPool2d::new([3, 3], [2, 2], [1, 1]);

        let layer1 = Self::make_bottleneck_layer(64, 64, layers[0], 1)?;
        let layer2 =
            Self::make_bottleneck_layer(64 * Bottleneck::<T>::EXPANSION, 128, layers[1], 2)?;
        let layer3 =
            Self::make_bottleneck_layer(128 * Bottleneck::<T>::EXPANSION, 256, layers[2], 2)?;
        let layer4 =
            Self::make_bottleneck_layer(256 * Bottleneck::<T>::EXPANSION, 512, layers[3], 2)?;

        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let fc = Linear::new(512 * Bottleneck::<T>::EXPANSION, num_classes, true)?;

        Ok(Self {
            conv1,
            maxpool,
            layer1,
            layer2,
            layer3,
            layer4,
            avgpool,
            fc,
            training: true,
        })
    }

    /// Create a stage of BasicBlocks.
    ///
    /// The first block may downsample (stride > 1). Subsequent blocks
    /// have stride 1 and preserve spatial dimensions.
    fn make_basic_layer(
        in_planes: usize,
        planes: usize,
        num_blocks: usize,
        stride: usize,
    ) -> FerrotorchResult<Vec<Box<dyn Module<T>>>> {
        let mut blocks: Vec<Box<dyn Module<T>>> = Vec::with_capacity(num_blocks);

        // First block may change channels/stride.
        blocks.push(Box::new(BasicBlock::new(in_planes, planes, stride)?));

        // Remaining blocks.
        let current_planes = planes * BasicBlock::<T>::EXPANSION;
        for _ in 1..num_blocks {
            blocks.push(Box::new(BasicBlock::new(current_planes, planes, 1)?));
        }

        Ok(blocks)
    }

    /// Create a stage of Bottleneck blocks.
    fn make_bottleneck_layer(
        in_planes: usize,
        planes: usize,
        num_blocks: usize,
        stride: usize,
    ) -> FerrotorchResult<Vec<Box<dyn Module<T>>>> {
        let mut blocks: Vec<Box<dyn Module<T>>> = Vec::with_capacity(num_blocks);

        blocks.push(Box::new(Bottleneck::new(in_planes, planes, stride)?));

        let current_planes = planes * Bottleneck::<T>::EXPANSION;
        for _ in 1..num_blocks {
            blocks.push(Box::new(Bottleneck::new(current_planes, planes, 1)?));
        }

        Ok(blocks)
    }

    /// Total number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

// ---------------------------------------------------------------------------
// Module impl
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for ResNet<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Stem: conv -> relu -> maxpool.
        let x = self.conv1.forward(input)?;
        let x = relu(&x)?;
        let x = Module::<T>::forward(&self.maxpool, &x)?;

        // Residual stages.
        let x = forward_layer(&self.layer1, &x)?;
        let x = forward_layer(&self.layer2, &x)?;
        let x = forward_layer(&self.layer3, &x)?;
        let x = forward_layer(&self.layer4, &x)?;

        // Global average pool: [B, C, 1, 1].
        let x = Module::<T>::forward(&self.avgpool, &x)?;

        // Flatten to [B, C] for the linear head.
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;

        // Classifier.
        self.fc.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(collect_layer_params(&self.layer1));
        params.extend(collect_layer_params(&self.layer2));
        params.extend(collect_layer_params(&self.layer3));
        params.extend(collect_layer_params(&self.layer4));
        params.extend(self.fc.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(collect_layer_params_mut(&mut self.layer1));
        params.extend(collect_layer_params_mut(&mut self.layer2));
        params.extend(collect_layer_params_mut(&mut self.layer3));
        params.extend(collect_layer_params_mut(&mut self.layer4));
        params.extend(self.fc.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.conv1.named_parameters() {
            params.push((format!("conv1.{name}"), p));
        }
        named_layer_params(&self.layer1, "layer1", &mut params);
        named_layer_params(&self.layer2, "layer2", &mut params);
        named_layer_params(&self.layer3, "layer3", &mut params);
        named_layer_params(&self.layer4, "layer4", &mut params);
        for (name, p) in self.fc.named_parameters() {
            params.push((format!("fc.{name}"), p));
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        for block in &mut self.layer1 {
            block.train();
        }
        for block in &mut self.layer2 {
            block.train();
        }
        for block in &mut self.layer3 {
            block.train();
        }
        for block in &mut self.layer4 {
            block.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for block in &mut self.layer1 {
            block.eval();
        }
        for block in &mut self.layer2 {
            block.eval();
        }
        for block in &mut self.layer3 {
            block.eval();
        }
        for block in &mut self.layer4 {
            block.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// Layer iteration helpers
// ---------------------------------------------------------------------------

/// Run a forward pass through a sequence of blocks.
fn forward_layer<T: Float>(
    blocks: &[Box<dyn Module<T>>],
    input: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let mut x = blocks[0].forward(input)?;
    for block in &blocks[1..] {
        x = block.forward(&x)?;
    }
    Ok(x)
}

/// Collect immutable parameter references from a layer.
fn collect_layer_params<T: Float>(blocks: &[Box<dyn Module<T>>]) -> Vec<&Parameter<T>> {
    blocks.iter().flat_map(|b| b.parameters()).collect()
}

/// Collect mutable parameter references from a layer.
fn collect_layer_params_mut<T: Float>(blocks: &mut [Box<dyn Module<T>>]) -> Vec<&mut Parameter<T>> {
    blocks.iter_mut().flat_map(|b| b.parameters_mut()).collect()
}

/// Collect named parameters from a layer with a prefix.
fn named_layer_params<'a, T: Float>(
    blocks: &'a [Box<dyn Module<T>>],
    layer_name: &str,
    out: &mut Vec<(String, &'a Parameter<T>)>,
) {
    for (i, block) in blocks.iter().enumerate() {
        for (name, p) in block.named_parameters() {
            out.push((format!("{layer_name}.{i}.{name}"), p));
        }
    }
}

// ===========================================================================
// Convenience constructors
// ===========================================================================

/// Construct a ResNet-18 model.
///
/// Architecture: `[2, 2, 2, 2]` BasicBlocks, ~11.2M parameters (without BN).
pub fn resnet18<T: Float>(num_classes: usize) -> FerrotorchResult<ResNet<T>> {
    ResNet::from_basic([2, 2, 2, 2], num_classes)
}

/// Construct a ResNet-34 model.
///
/// Architecture: `[3, 4, 6, 3]` BasicBlocks, ~21.3M parameters (without BN).
pub fn resnet34<T: Float>(num_classes: usize) -> FerrotorchResult<ResNet<T>> {
    ResNet::from_basic([3, 4, 6, 3], num_classes)
}

/// Construct a ResNet-50 model.
///
/// Architecture: `[3, 4, 6, 3]` Bottlenecks, ~23.5M parameters (without BN).
pub fn resnet50<T: Float>(num_classes: usize) -> FerrotorchResult<ResNet<T>> {
    ResNet::from_bottleneck([3, 4, 6, 3], num_classes)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use ferrotorch_core::{TensorStorage, no_grad};

    /// Create a 4-D tensor from flat data.
    fn leaf_4d(data: &[f32], shape: [usize; 4], requires_grad: bool) -> Tensor<f32> {
        Tensor::from_storage(
            TensorStorage::cpu(data.to_vec()),
            shape.to_vec(),
            requires_grad,
        )
        .unwrap()
    }

    // -----------------------------------------------------------------------
    // BasicBlock tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_basic_block_same_channels() {
        let block = BasicBlock::<f32>::new(64, 64, 1).unwrap();
        let input = leaf_4d(&vec![0.1; 1 * 64 * 8 * 8], [1, 64, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 64, 8, 8]);
    }

    #[test]
    fn test_basic_block_downsample() {
        let block = BasicBlock::<f32>::new(64, 128, 2).unwrap();
        let input = leaf_4d(&vec![0.1; 1 * 64 * 8 * 8], [1, 64, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        // Stride 2: spatial dims halve. Channels: 64 -> 128.
        assert_eq!(output.shape(), &[1, 128, 4, 4]);
    }

    #[test]
    fn test_basic_block_has_downsample_when_needed() {
        let block = BasicBlock::<f32>::new(64, 128, 2).unwrap();
        assert!(block.downsample.is_some());

        let block = BasicBlock::<f32>::new(64, 64, 1).unwrap();
        assert!(block.downsample.is_none());
    }

    #[test]
    fn test_basic_block_parameter_count() {
        // No downsample: 2 * (64 * 64 * 3 * 3) = 73728
        let block = BasicBlock::<f32>::new(64, 64, 1).unwrap();
        let count: usize = block.parameters().iter().map(|p| p.numel()).sum();
        assert_eq!(count, 2 * 64 * 64 * 3 * 3);

        // With downsample: 2 convs + 1x1 ds
        let block = BasicBlock::<f32>::new(64, 128, 2).unwrap();
        let count: usize = block.parameters().iter().map(|p| p.numel()).sum();
        let expected = 64 * 128 * 3 * 3     // conv1: 64->128
                     + 128 * 128 * 3 * 3    // conv2: 128->128
                     + 64 * 128 * 1 * 1; // downsample: 64->128
        assert_eq!(count, expected);
    }

    // -----------------------------------------------------------------------
    // Bottleneck tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bottleneck_same_channels() {
        // in_planes must match planes * EXPANSION = 64 * 4 = 256
        let block = Bottleneck::<f32>::new(256, 64, 1).unwrap();
        let input = leaf_4d(&vec![0.1; 1 * 256 * 8 * 8], [1, 256, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 256, 8, 8]);
    }

    #[test]
    fn test_bottleneck_first_block() {
        // First block in layer1: in_planes=64, planes=64
        // Output channels = 64 * 4 = 256.
        let block = Bottleneck::<f32>::new(64, 64, 1).unwrap();
        let input = leaf_4d(&vec![0.1; 1 * 64 * 8 * 8], [1, 64, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 256, 8, 8]);
        assert!(block.downsample.is_some()); // 64 != 256
    }

    #[test]
    fn test_bottleneck_downsample() {
        let block = Bottleneck::<f32>::new(256, 128, 2).unwrap();
        let input = leaf_4d(&vec![0.1; 1 * 256 * 8 * 8], [1, 256, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 512, 4, 4]);
    }

    #[test]
    fn test_bottleneck_parameter_count() {
        // No downsample: in=256, planes=64
        // conv1: 256*64*1*1 = 16384
        // conv2: 64*64*3*3  = 36864
        // conv3: 64*256*1*1 = 16384
        // total = 69632
        let block = Bottleneck::<f32>::new(256, 64, 1).unwrap();
        let count: usize = block.parameters().iter().map(|p| p.numel()).sum();
        assert_eq!(count, 16384 + 36864 + 16384);
    }

    // -----------------------------------------------------------------------
    // ResNet-18 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resnet18_output_shape() {
        let model = resnet18::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 1 * 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    #[test]
    fn test_resnet18_param_count() {
        let model = resnet18::<f32>(1000).unwrap();
        let total = model.num_parameters();
        // ResNet-18 without BN has ~11.17M params.
        // Exact: conv1(3*64*7*7=9408) + blocks + fc(512*1000+1000=513000)
        assert!(
            total > 11_000_000,
            "ResNet-18 should have >11M params, got {total}"
        );
        assert!(
            total < 12_000_000,
            "ResNet-18 should have <12M params, got {total}"
        );
    }

    #[test]
    fn test_resnet18_named_parameters_prefixes() {
        let model = resnet18::<f32>(1000).unwrap();
        let named = model.named_parameters();
        assert!(!named.is_empty());

        // Check that key prefixes exist.
        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("conv1.")));
        assert!(names.iter().any(|n| n.starts_with("layer1.")));
        assert!(names.iter().any(|n| n.starts_with("layer2.")));
        assert!(names.iter().any(|n| n.starts_with("layer3.")));
        assert!(names.iter().any(|n| n.starts_with("layer4.")));
        assert!(names.iter().any(|n| n.starts_with("fc.")));
    }

    // -----------------------------------------------------------------------
    // ResNet-34 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resnet34_output_shape() {
        let model = resnet34::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 1 * 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    #[test]
    fn test_resnet34_param_count() {
        let model = resnet34::<f32>(1000).unwrap();
        let total = model.num_parameters();
        // ResNet-34 without BN: ~21.3M params.
        assert!(
            total > 21_000_000,
            "ResNet-34 should have >21M params, got {total}"
        );
        assert!(
            total < 22_000_000,
            "ResNet-34 should have <22M params, got {total}"
        );
    }

    // -----------------------------------------------------------------------
    // ResNet-50 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resnet50_output_shape() {
        let model = resnet50::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 1 * 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    #[test]
    fn test_resnet50_param_count() {
        let model = resnet50::<f32>(1000).unwrap();
        let total = model.num_parameters();
        // ResNet-50 without BN: ~25.5M params.
        // (Standard ResNet-50 is ~25.6M including BN; without BN we lose ~53K
        // BN params but keep all conv/fc weights.)
        assert!(
            total > 25_000_000,
            "ResNet-50 should have >25M params, got {total}"
        );
        assert!(
            total < 26_000_000,
            "ResNet-50 should have <26M params, got {total}"
        );
    }

    // -----------------------------------------------------------------------
    // Custom num_classes
    // -----------------------------------------------------------------------

    #[test]
    fn test_resnet18_custom_classes() {
        let model = resnet18::<f32>(10).unwrap();
        let input = leaf_4d(&vec![0.01; 2 * 3 * 224 * 224], [2, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 10]);
    }

    // -----------------------------------------------------------------------
    // Gradient flow through residual connections
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_flow_through_basic_block() {
        let block = BasicBlock::<f32>::new(4, 4, 1).unwrap();
        let input = leaf_4d(&vec![0.5; 1 * 4 * 4 * 4], [1, 4, 4, 4], true);

        let output = block.forward(&input).unwrap();

        // Reduce to scalar for backward.
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        // Input should have received gradients through the skip connection.
        let grad = input.grad().unwrap();
        assert!(grad.is_some(), "input should have gradients");
        let grad_data = grad.unwrap().data().unwrap().to_vec();
        // At least some gradients should be non-zero (the skip connection
        // guarantees a path even if conv weights happen to zero out).
        let any_nonzero = grad_data.iter().any(|&g| g.abs() > 1e-10);
        assert!(any_nonzero, "gradients should flow through residual path");
    }

    #[test]
    fn test_gradient_flow_through_bottleneck() {
        let block = Bottleneck::<f32>::new(4, 2, 1).unwrap();
        let input = leaf_4d(&vec![0.5; 1 * 4 * 4 * 4], [1, 4, 4, 4], true);

        let output = block.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 8, 4, 4]); // expansion = 4, planes = 2

        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap();
        assert!(grad.is_some(), "input should have gradients");
        let grad_data = grad.unwrap().data().unwrap().to_vec();
        let any_nonzero = grad_data.iter().any(|&g| g.abs() > 1e-10);
        assert!(any_nonzero, "gradients should flow through residual path");
    }

    // -----------------------------------------------------------------------
    // Train / eval
    // -----------------------------------------------------------------------

    #[test]
    fn test_resnet_train_eval() {
        let mut model = resnet18::<f32>(10).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }

    // -----------------------------------------------------------------------
    // Send + Sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_resnet_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ResNet<f32>>();
        assert_send_sync::<BasicBlock<f32>>();
        assert_send_sync::<Bottleneck<f32>>();
    }
}
