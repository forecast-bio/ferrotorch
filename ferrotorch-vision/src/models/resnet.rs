//! ResNet architectures: ResNet-18, ResNet-34, and ResNet-50.
//!
//! Follows the original paper: "Deep Residual Learning for Image Recognition"
//! (He et al., 2015). Batch normalization is included after every convolution,
//! matching the `torchvision.models.resnet` reference implementation.
//!
//! All convolutions use Kaiming-initialized weights (bias=false). BN layers
//! use the default affine=true so weight+bias parameters are learnable, giving
//! the same parameter count as torchvision. The residual `add` uses
//! `ferrotorch_core::grad_fns::arithmetic::add` so gradients flow through
//! skip connections automatically.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::Conv2d;
use ferrotorch_nn::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::BatchNorm2d;
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

/// Create a 3×3 conv with explicit `dilation` (and matching `padding=dilation`
/// for same-size output when `stride=1`). Used by Bottleneck/BasicBlock when
/// the stage was constructed with `replace_stride_with_dilation`.
fn conv3x3_dilated<T: Float>(
    in_planes: usize,
    out_planes: usize,
    stride: usize,
    dilation: (usize, usize),
) -> FerrotorchResult<Conv2d<T>> {
    Conv2d::new_full(
        in_planes,
        out_planes,
        (3, 3),
        (stride, stride),
        // padding = dilation preserves same-size output for kernel=3 / stride=1.
        // For stride=2 the spatial halves; the FCN/DeepLab dilated paths use
        // stride=1 + dilation, so this padding choice mirrors torchvision.
        dilation,
        dilation,
        1,
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

/// A basic residual block with two 3x3 convolutions, each followed by
/// BatchNorm2d — matching the torchvision `BasicBlock` reference.
///
/// ```text
/// x --> conv1 -> bn1 -> relu -> conv2 -> bn2 --> (+) -> relu -> out
/// |                                               ^
/// +------------- [downsample conv -> bn] ----------+
/// ```
///
/// The optional `downsample` path uses a 1x1 conv + BN and is added whenever
/// the spatial size or channel count changes between input and output.
pub struct BasicBlock<T: Float> {
    conv1: Conv2d<T>,
    bn1: BatchNorm2d<T>,
    conv2: Conv2d<T>,
    bn2: BatchNorm2d<T>,
    /// When present: (conv1x1, bn) that projects the skip connection.
    downsample: Option<(Conv2d<T>, BatchNorm2d<T>)>,
    training: bool,
}

impl<T: Float> BasicBlock<T> {
    /// Expansion factor for BasicBlock (channels are not expanded).
    pub const EXPANSION: usize = 1;

    /// Create a new `BasicBlock` (no dilation; first conv stride is `stride`).
    ///
    /// * `in_planes`  -- number of input channels.
    /// * `planes`     -- number of output channels (before expansion, but
    ///   expansion=1 for BasicBlock so this is also the final channel count).
    /// * `stride`     -- spatial stride for the first 3x3 conv. Use 2 for
    ///   downsampling layers.
    ///
    /// Phase 6 (#994): one-line shim over [`BasicBlock::new_full`] with
    /// `dilation=(1, 1)` so the existing `Bottleneck::new(...)` call sites
    /// (and ResNet18/34 factories) stay byte-equivalent — the value-parity
    /// PASS observed in Phase 6 Movement 2 is the binding regression test.
    pub fn new(in_planes: usize, planes: usize, stride: usize) -> FerrotorchResult<Self> {
        Self::new_full(in_planes, planes, stride, (1, 1))
    }

    /// Create a new `BasicBlock` with explicit `dilation` (Phase 6 #994).
    ///
    /// torchvision's `BasicBlock` applies dilation to the 3×3 conv1 (the
    /// first conv) when `replace_stride_with_dilation` is True for that
    /// stage. The 3×3 conv2 stays at dilation=1.
    pub fn new_full(
        in_planes: usize,
        planes: usize,
        stride: usize,
        dilation: (usize, usize),
    ) -> FerrotorchResult<Self> {
        let conv1 = if dilation == (1, 1) {
            conv3x3(in_planes, planes, stride)?
        } else {
            conv3x3_dilated(in_planes, planes, stride, dilation)?
        };
        let bn1 = BatchNorm2d::new(planes, 1e-5, 0.1, true)?;
        let conv2 = conv3x3(planes, planes, 1)?;
        let bn2 = BatchNorm2d::new(planes, 1e-5, 0.1, true)?;

        let downsample = if stride != 1 || in_planes != planes * Self::EXPANSION {
            let ds_conv = conv1x1(in_planes, planes * Self::EXPANSION, stride)?;
            let ds_bn = BatchNorm2d::new(planes * Self::EXPANSION, 1e-5, 0.1, true)?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            downsample,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for BasicBlock<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Main path: conv -> BN -> relu -> conv -> BN.
        let out = self.conv1.forward(input)?;
        let out = Module::<T>::forward(&self.bn1, &out)?;
        let out = relu(&out)?;
        let out = self.conv2.forward(&out)?;
        let out = Module::<T>::forward(&self.bn2, &out)?;

        // Skip connection (optionally projected).
        let identity = match &self.downsample {
            Some((ds_conv, ds_bn)) => {
                let x = ds_conv.forward(input)?;
                Module::<T>::forward(ds_bn, &x)?
            }
            None => input.clone(),
        };

        // Residual addition (differentiable).
        let out = add(&out, &identity)?;
        relu(&out)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            params.extend(ds_conv.parameters());
            params.extend(ds_bn.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.bn1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.bn2.parameters_mut());
        if let Some((ref mut ds_conv, ref mut ds_bn)) = self.downsample {
            params.extend(ds_conv.parameters_mut());
            params.extend(ds_bn.parameters_mut());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.conv1.named_parameters() {
            params.push((format!("conv1.{name}"), p));
        }
        for (name, p) in self.bn1.named_parameters() {
            params.push((format!("bn1.{name}"), p));
        }
        for (name, p) in self.conv2.named_parameters() {
            params.push((format!("conv2.{name}"), p));
        }
        for (name, p) in self.bn2.named_parameters() {
            params.push((format!("bn2.{name}"), p));
        }
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            for (name, p) in ds_conv.named_parameters() {
                params.push((format!("downsample.0.{name}"), p));
            }
            for (name, p) in ds_bn.named_parameters() {
                params.push((format!("downsample.1.{name}"), p));
            }
        }
        params
    }

    // Phase 4 (#995): expose direct children so the Phase 2 BN-buffer
    // loader can reach `bn1` / `bn2` / `downsample.1` via
    // `named_descendants_dyn()`. The torchvision-shaped path layout
    // (`downsample.{0,1}` for the conv+BN pair) matches `named_parameters`
    // above so the loader's path-keyed index agrees with the fixture
    // descriptors. Default `Module::named_children` returns an empty
    // Vec — without this override every BN running statistic is
    // silently skipped (Phase 1A fallback).
    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = vec![&self.conv1, &self.bn1, &self.conv2, &self.bn2];
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            out.push(ds_conv);
            out.push(ds_bn);
        }
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> = vec![
            ("conv1".to_string(), &self.conv1),
            ("bn1".to_string(), &self.bn1),
            ("conv2".to_string(), &self.conv2),
            ("bn2".to_string(), &self.bn2),
        ];
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            out.push(("downsample.0".to_string(), ds_conv));
            out.push(("downsample.1".to_string(), ds_bn));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.bn1.train();
        self.bn2.train();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        self.bn1.eval();
        self.bn2.eval();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Bottleneck (for ResNet-50)
// ===========================================================================

/// A bottleneck residual block with 1x1 -> 3x3 -> 1x1 convolutions, each
/// followed by BatchNorm2d — matching the torchvision `Bottleneck` reference.
///
/// ```text
/// x -> conv1(1x1)->bn1->relu -> conv2(3x3)->bn2->relu -> conv3(1x1)->bn3 -> (+) -> relu -> out
/// |                                                                           ^
/// +---------------------------- [downsample conv -> bn] ----------------------+
/// ```
///
/// The expansion factor is 4: `conv3` outputs `planes * 4` channels.
pub struct Bottleneck<T: Float> {
    conv1: Conv2d<T>,
    bn1: BatchNorm2d<T>,
    conv2: Conv2d<T>,
    bn2: BatchNorm2d<T>,
    conv3: Conv2d<T>,
    bn3: BatchNorm2d<T>,
    /// When present: (conv1x1, bn) projecting the skip connection.
    downsample: Option<(Conv2d<T>, BatchNorm2d<T>)>,
    training: bool,
}

impl<T: Float> Bottleneck<T> {
    /// Expansion factor for Bottleneck.
    pub const EXPANSION: usize = 4;

    /// Create a new `Bottleneck` (no dilation, 3×3 conv stride is `stride`).
    ///
    /// * `in_planes`  -- number of input channels.
    /// * `planes`     -- bottleneck width (intermediate channel count).
    /// * `stride`     -- spatial stride for the 3x3 conv.
    ///
    /// Phase 6 (#994): one-line shim over [`Bottleneck::new_full`] with
    /// `dilation=(1, 1)`. ResNet50 (which goes through this path everywhere)
    /// stays byte-equivalent — the existing `resnet50_value_parity` PASS is
    /// the binding regression test for the shim.
    pub fn new(in_planes: usize, planes: usize, stride: usize) -> FerrotorchResult<Self> {
        Self::new_full(in_planes, planes, stride, (1, 1))
    }

    /// Create a new `Bottleneck` with explicit `dilation` (Phase 6 #994).
    ///
    /// torchvision's `Bottleneck` applies dilation to the 3×3 conv2 (the
    /// middle conv); conv1/conv3 are 1×1 and unaffected. Padding for the
    /// 3×3 is set to `dilation` to keep the output spatial size constant
    /// when `stride=1` (the case torchvision uses for
    /// `replace_stride_with_dilation`).
    pub fn new_full(
        in_planes: usize,
        planes: usize,
        stride: usize,
        dilation: (usize, usize),
    ) -> FerrotorchResult<Self> {
        let width = planes; // No group convolution, so width = planes.

        let conv1 = conv1x1(in_planes, width, 1)?;
        let bn1 = BatchNorm2d::new(width, 1e-5, 0.1, true)?;
        let conv2 = if dilation == (1, 1) {
            conv3x3(width, width, stride)?
        } else {
            conv3x3_dilated(width, width, stride, dilation)?
        };
        let bn2 = BatchNorm2d::new(width, 1e-5, 0.1, true)?;
        let conv3 = conv1x1(width, planes * Self::EXPANSION, 1)?;
        let bn3 = BatchNorm2d::new(planes * Self::EXPANSION, 1e-5, 0.1, true)?;

        let downsample = if stride != 1 || in_planes != planes * Self::EXPANSION {
            let ds_conv = conv1x1(in_planes, planes * Self::EXPANSION, stride)?;
            let ds_bn = BatchNorm2d::new(planes * Self::EXPANSION, 1e-5, 0.1, true)?;
            Some((ds_conv, ds_bn))
        } else {
            None
        };

        Ok(Self {
            conv1,
            bn1,
            conv2,
            bn2,
            conv3,
            bn3,
            downsample,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Bottleneck<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // 1x1 reduce.
        let out = self.conv1.forward(input)?;
        let out = Module::<T>::forward(&self.bn1, &out)?;
        let out = relu(&out)?;

        // 3x3 process.
        let out = self.conv2.forward(&out)?;
        let out = Module::<T>::forward(&self.bn2, &out)?;
        let out = relu(&out)?;

        // 1x1 expand.
        let out = self.conv3.forward(&out)?;
        let out = Module::<T>::forward(&self.bn3, &out)?;

        // Skip connection (optionally projected).
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
        let mut params = Vec::new();
        params.extend(self.conv1.parameters());
        params.extend(self.bn1.parameters());
        params.extend(self.conv2.parameters());
        params.extend(self.bn2.parameters());
        params.extend(self.conv3.parameters());
        params.extend(self.bn3.parameters());
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            params.extend(ds_conv.parameters());
            params.extend(ds_bn.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.conv1.parameters_mut());
        params.extend(self.bn1.parameters_mut());
        params.extend(self.conv2.parameters_mut());
        params.extend(self.bn2.parameters_mut());
        params.extend(self.conv3.parameters_mut());
        params.extend(self.bn3.parameters_mut());
        if let Some((ref mut ds_conv, ref mut ds_bn)) = self.downsample {
            params.extend(ds_conv.parameters_mut());
            params.extend(ds_bn.parameters_mut());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.conv1.named_parameters() {
            params.push((format!("conv1.{name}"), p));
        }
        for (name, p) in self.bn1.named_parameters() {
            params.push((format!("bn1.{name}"), p));
        }
        for (name, p) in self.conv2.named_parameters() {
            params.push((format!("conv2.{name}"), p));
        }
        for (name, p) in self.bn2.named_parameters() {
            params.push((format!("bn2.{name}"), p));
        }
        for (name, p) in self.conv3.named_parameters() {
            params.push((format!("conv3.{name}"), p));
        }
        for (name, p) in self.bn3.named_parameters() {
            params.push((format!("bn3.{name}"), p));
        }
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            for (name, p) in ds_conv.named_parameters() {
                params.push((format!("downsample.0.{name}"), p));
            }
            for (name, p) in ds_bn.named_parameters() {
                params.push((format!("downsample.1.{name}"), p));
            }
        }
        params
    }

    // Phase 4 (#995): direct-children override mirroring `named_parameters`
    // above so `named_descendants_dyn()` reaches `bn1` / `bn2` / `bn3` /
    // `downsample.1` for the BN-buffer loader.
    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = vec![
            &self.conv1,
            &self.bn1,
            &self.conv2,
            &self.bn2,
            &self.conv3,
            &self.bn3,
        ];
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            out.push(ds_conv);
            out.push(ds_bn);
        }
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> = vec![
            ("conv1".to_string(), &self.conv1),
            ("bn1".to_string(), &self.bn1),
            ("conv2".to_string(), &self.conv2),
            ("bn2".to_string(), &self.bn2),
            ("conv3".to_string(), &self.conv3),
            ("bn3".to_string(), &self.bn3),
        ];
        if let Some((ref ds_conv, ref ds_bn)) = self.downsample {
            out.push(("downsample.0".to_string(), ds_conv));
            out.push(("downsample.1".to_string(), ds_bn));
        }
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.bn1.train();
        self.bn2.train();
        self.bn3.train();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        self.bn1.eval();
        self.bn2.eval();
        self.bn3.eval();
        if let Some((_, ref mut ds_bn)) = self.downsample {
            ds_bn.eval();
        }
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
/// 1. 7x7 conv, stride 2, BN, ReLU (stem)
/// 2. 3x3 max pool, stride 2
/// 3. Four residual layers (layer1..layer4)
/// 4. Adaptive average pool to (1, 1)
/// 5. Fully connected classifier
pub struct ResNet<T: Float> {
    // Stem.
    conv1: Conv2d<T>,
    bn1: BatchNorm2d<T>,
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
    /// of the four stages. `replace_stride_with_dilation` (Phase 6 #994),
    /// when supplied, swaps the stride-2 downsample at layers 2/3/4 for a
    /// dilated 3×3 conv per torchvision; `None` preserves the existing
    /// dense behaviour bit-for-bit.
    fn from_basic(
        layers: [usize; 4],
        num_classes: usize,
        replace_stride_with_dilation: Option<[bool; 3]>,
    ) -> FerrotorchResult<Self> {
        let conv1 = Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), false)?;
        let bn1 = BatchNorm2d::new(64, 1e-5, 0.1, true)?;
        let maxpool = MaxPool2d::new([3, 3], [2, 2], [1, 1]);

        // torchvision: self.dilation starts at 1; layer1 is always stride=1
        // and dilate=False; layer{2,3,4} consult the flag.
        let dilate_flags = replace_stride_with_dilation.unwrap_or([false, false, false]);
        let mut current_dilation: usize = 1;

        let layer1 = Self::make_basic_layer(64, 64, layers[0], 1, false, &mut current_dilation)?;
        let layer2 = Self::make_basic_layer(
            64,
            128,
            layers[1],
            2,
            dilate_flags[0],
            &mut current_dilation,
        )?;
        let layer3 = Self::make_basic_layer(
            128,
            256,
            layers[2],
            2,
            dilate_flags[1],
            &mut current_dilation,
        )?;
        let layer4 = Self::make_basic_layer(
            256,
            512,
            layers[3],
            2,
            dilate_flags[2],
            &mut current_dilation,
        )?;

        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let fc = Linear::new(512 * BasicBlock::<T>::EXPANSION, num_classes, true)?;

        Ok(Self {
            conv1,
            bn1,
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

    /// Build a ResNet from Bottleneck blocks. See [`from_basic`] for the
    /// `replace_stride_with_dilation` semantics; the threading is identical.
    fn from_bottleneck(
        layers: [usize; 4],
        num_classes: usize,
        replace_stride_with_dilation: Option<[bool; 3]>,
    ) -> FerrotorchResult<Self> {
        let conv1 = Conv2d::new(3, 64, (7, 7), (2, 2), (3, 3), false)?;
        let bn1 = BatchNorm2d::new(64, 1e-5, 0.1, true)?;
        let maxpool = MaxPool2d::new([3, 3], [2, 2], [1, 1]);

        let dilate_flags = replace_stride_with_dilation.unwrap_or([false, false, false]);
        let mut current_dilation: usize = 1;

        let layer1 =
            Self::make_bottleneck_layer(64, 64, layers[0], 1, false, &mut current_dilation)?;
        let layer2 = Self::make_bottleneck_layer(
            64 * Bottleneck::<T>::EXPANSION,
            128,
            layers[1],
            2,
            dilate_flags[0],
            &mut current_dilation,
        )?;
        let layer3 = Self::make_bottleneck_layer(
            128 * Bottleneck::<T>::EXPANSION,
            256,
            layers[2],
            2,
            dilate_flags[1],
            &mut current_dilation,
        )?;
        let layer4 = Self::make_bottleneck_layer(
            256 * Bottleneck::<T>::EXPANSION,
            512,
            layers[3],
            2,
            dilate_flags[2],
            &mut current_dilation,
        )?;

        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let fc = Linear::new(512 * Bottleneck::<T>::EXPANSION, num_classes, true)?;

        Ok(Self {
            conv1,
            bn1,
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

    /// Create a stage of BasicBlocks. Phase 6 (#994) added the
    /// `dilate` flag and the `current_dilation` thread-through state to
    /// match torchvision's `_make_layer`:
    ///
    /// * `previous_dilation = *current_dilation` (snapshot BEFORE update)
    /// * if `dilate`: `*current_dilation *= stride`, then `stride := 1`
    /// * the FIRST block uses `previous_dilation` and the
    ///   (possibly-overridden) `stride`; SUBSEQUENT blocks use the new
    ///   `*current_dilation` with stride=1.
    fn make_basic_layer(
        in_planes: usize,
        planes: usize,
        num_blocks: usize,
        mut stride: usize,
        dilate: bool,
        current_dilation: &mut usize,
    ) -> FerrotorchResult<Vec<Box<dyn Module<T>>>> {
        let previous_dilation = *current_dilation;
        if dilate {
            *current_dilation *= stride;
            stride = 1;
        }

        let mut blocks: Vec<Box<dyn Module<T>>> = Vec::with_capacity(num_blocks);

        blocks.push(Box::new(BasicBlock::new_full(
            in_planes,
            planes,
            stride,
            (previous_dilation, previous_dilation),
        )?));

        let current_planes = planes * BasicBlock::<T>::EXPANSION;
        for _ in 1..num_blocks {
            blocks.push(Box::new(BasicBlock::new_full(
                current_planes,
                planes,
                1,
                (*current_dilation, *current_dilation),
            )?));
        }

        Ok(blocks)
    }

    /// Create a stage of Bottleneck blocks. See [`make_basic_layer`] for
    /// the `dilate`/`current_dilation` semantics.
    fn make_bottleneck_layer(
        in_planes: usize,
        planes: usize,
        num_blocks: usize,
        mut stride: usize,
        dilate: bool,
        current_dilation: &mut usize,
    ) -> FerrotorchResult<Vec<Box<dyn Module<T>>>> {
        let previous_dilation = *current_dilation;
        if dilate {
            *current_dilation *= stride;
            stride = 1;
        }

        let mut blocks: Vec<Box<dyn Module<T>>> = Vec::with_capacity(num_blocks);

        blocks.push(Box::new(Bottleneck::new_full(
            in_planes,
            planes,
            stride,
            (previous_dilation, previous_dilation),
        )?));

        let current_planes = planes * Bottleneck::<T>::EXPANSION;
        for _ in 1..num_blocks {
            blocks.push(Box::new(Bottleneck::new_full(
                current_planes,
                planes,
                1,
                (*current_dilation, *current_dilation),
            )?));
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
        // Stem: conv -> BN -> relu -> maxpool.
        let x = self.conv1.forward(input)?;
        let x = Module::<T>::forward(&self.bn1, &x)?;
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
        params.extend(self.bn1.parameters());
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
        params.extend(self.bn1.parameters_mut());
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
        for (name, p) in self.bn1.named_parameters() {
            params.push((format!("bn1.{name}"), p));
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

    // Phase 4 (#995): expose stem + every residual block + head so the
    // Phase 2 loader's `named_descendants_dyn()` walk reaches every BN
    // in the network — `bn1` (stem), `layer{i}.{j}.{bn1,bn2,bn3,
    // downsample.1}` (residual stages). The torchvision-shaped layout
    // is identical to `named_parameters` above so the loader's
    // path → module index agrees with the fixture descriptors.
    fn children(&self) -> Vec<&dyn Module<T>> {
        let mut out: Vec<&dyn Module<T>> = vec![&self.conv1, &self.bn1, &self.maxpool];
        for block in &self.layer1 {
            out.push(block.as_ref());
        }
        for block in &self.layer2 {
            out.push(block.as_ref());
        }
        for block in &self.layer3 {
            out.push(block.as_ref());
        }
        for block in &self.layer4 {
            out.push(block.as_ref());
        }
        out.push(&self.avgpool);
        out.push(&self.fc);
        out
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        let mut out: Vec<(String, &dyn Module<T>)> = vec![
            ("conv1".to_string(), &self.conv1),
            ("bn1".to_string(), &self.bn1),
            ("maxpool".to_string(), &self.maxpool),
        ];
        for (i, block) in self.layer1.iter().enumerate() {
            out.push((format!("layer1.{i}"), block.as_ref()));
        }
        for (i, block) in self.layer2.iter().enumerate() {
            out.push((format!("layer2.{i}"), block.as_ref()));
        }
        for (i, block) in self.layer3.iter().enumerate() {
            out.push((format!("layer3.{i}"), block.as_ref()));
        }
        for (i, block) in self.layer4.iter().enumerate() {
            out.push((format!("layer4.{i}"), block.as_ref()));
        }
        out.push(("avgpool".to_string(), &self.avgpool));
        out.push(("fc".to_string(), &self.fc));
        out
    }

    fn train(&mut self) {
        self.training = true;
        self.bn1.train();
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
        self.bn1.eval();
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
// IntermediateFeatures impl — exposes per-stage activations for feature
// extraction (e.g. as backbones for FPN, U-Net, etc.). CL-384.
// ===========================================================================

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for ResNet<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();

        // Stem.
        let x = self.conv1.forward(input)?;
        let x = Module::<T>::forward(&self.bn1, &x)?;
        let x = relu(&x)?;
        let x = Module::<T>::forward(&self.maxpool, &x)?;
        out.insert("stem".to_string(), x.clone());

        // Residual stages — record after each.
        let x = forward_layer(&self.layer1, &x)?;
        out.insert("layer1".to_string(), x.clone());
        let x = forward_layer(&self.layer2, &x)?;
        out.insert("layer2".to_string(), x.clone());
        let x = forward_layer(&self.layer3, &x)?;
        out.insert("layer3".to_string(), x.clone());
        let x = forward_layer(&self.layer4, &x)?;
        out.insert("layer4".to_string(), x.clone());

        // Global average pool.
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        out.insert("avgpool".to_string(), x.clone());

        // Classifier.
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        let logits = self.fc.forward(&x)?;
        out.insert("fc".to_string(), logits);

        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        vec![
            "stem".to_string(),
            "layer1".to_string(),
            "layer2".to_string(),
            "layer3".to_string(),
            "layer4".to_string(),
            "avgpool".to_string(),
            "fc".to_string(),
        ]
    }
}

// ===========================================================================
// Convenience constructors
// ===========================================================================

/// Construct a ResNet-18 model.
///
/// Architecture: `[2, 2, 2, 2]` BasicBlocks. With BN (#860) the parameter
/// count matches torchvision's ~11.7M.
pub fn resnet18<T: Float>(num_classes: usize) -> FerrotorchResult<ResNet<T>> {
    ResNet::from_basic([2, 2, 2, 2], num_classes, None)
}

/// Construct a ResNet-34 model.
pub fn resnet34<T: Float>(num_classes: usize) -> FerrotorchResult<ResNet<T>> {
    ResNet::from_basic([3, 4, 6, 3], num_classes, None)
}

/// Construct a ResNet-50 model.
///
/// Architecture: `[3, 4, 6, 3]` Bottlenecks. Matches torchvision's ~25.6M
/// parameters with BN.
pub fn resnet50<T: Float>(num_classes: usize) -> FerrotorchResult<ResNet<T>> {
    ResNet::from_bottleneck([3, 4, 6, 3], num_classes, None)
}

/// Construct a ResNet-50 model with `replace_stride_with_dilation` (Phase 6
/// #994). Mirrors torchvision's `resnet50(replace_stride_with_dilation=...)`:
/// for each entry that is `true`, the corresponding stage's stride-2 is
/// swapped for dilation×2 in the 3×3 conv2 of every block. Used by
/// `fcn_resnet50` (which uses `[false, true, true]`).
pub fn resnet50_dilated<T: Float>(
    num_classes: usize,
    replace_stride_with_dilation: [bool; 3],
) -> FerrotorchResult<ResNet<T>> {
    ResNet::from_bottleneck(
        [3, 4, 6, 3],
        num_classes,
        Some(replace_stride_with_dilation),
    )
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
        let input = leaf_4d(&vec![0.1; 64 * 8 * 8], [1, 64, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 64, 8, 8]);
    }

    #[test]
    fn test_basic_block_downsample() {
        let block = BasicBlock::<f32>::new(64, 128, 2).unwrap();
        let input = leaf_4d(&vec![0.1; 64 * 8 * 8], [1, 64, 8, 8], false);
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
        // No downsample: 2 convs + 2 BNs (weight+bias each).
        // conv1: 64*64*3*3=36864, bn1: 64*2=128, conv2: 64*64*3*3=36864, bn2: 64*2=128
        let block = BasicBlock::<f32>::new(64, 64, 1).unwrap();
        let count: usize = block.parameters().iter().map(|p| p.numel()).sum();
        let expected_no_ds = 2 * 64 * 64 * 3 * 3   // conv1 + conv2
                           + 2 * (2 * 64); // bn1 + bn2 (weight+bias)
        assert_eq!(count, expected_no_ds);

        // With downsample: 2 convs + 2 BNs + ds_conv + ds_bn.
        let block = BasicBlock::<f32>::new(64, 128, 2).unwrap();
        let count: usize = block.parameters().iter().map(|p| p.numel()).sum();
        let expected = 64 * 128 * 3 * 3     // conv1: 64->128
                     + 2 * 128              // bn1
                     + 128 * 128 * 3 * 3   // conv2: 128->128
                     + 2 * 128             // bn2
                     + 64 * 128            // downsample conv: 64->128
                     + 2 * 128; // downsample bn
        assert_eq!(count, expected);
    }

    // -----------------------------------------------------------------------
    // Bottleneck tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_bottleneck_same_channels() {
        // in_planes must match planes * EXPANSION = 64 * 4 = 256
        let block = Bottleneck::<f32>::new(256, 64, 1).unwrap();
        let input = leaf_4d(&vec![0.1; 256 * 8 * 8], [1, 256, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 256, 8, 8]);
    }

    #[test]
    fn test_bottleneck_first_block() {
        // First block in layer1: in_planes=64, planes=64
        // Output channels = 64 * 4 = 256.
        let block = Bottleneck::<f32>::new(64, 64, 1).unwrap();
        let input = leaf_4d(&vec![0.1; 64 * 8 * 8], [1, 64, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 256, 8, 8]);
        assert!(block.downsample.is_some()); // 64 != 256
    }

    #[test]
    fn test_bottleneck_downsample() {
        let block = Bottleneck::<f32>::new(256, 128, 2).unwrap();
        let input = leaf_4d(&vec![0.1; 256 * 8 * 8], [1, 256, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 512, 4, 4]);
    }

    #[test]
    fn test_bottleneck_parameter_count() {
        // No downsample: in=256, planes=64 (256 == 64*4 so no projection needed).
        // conv1: 256*64*1*1 = 16384
        // bn1:   2*64       =   128   (weight + bias)
        // conv2: 64*64*3*3  = 36864
        // bn2:   2*64       =   128
        // conv3: 64*256*1*1 = 16384
        // bn3:   2*256      =   512
        // total = 70400
        let block = Bottleneck::<f32>::new(256, 64, 1).unwrap();
        let count: usize = block.parameters().iter().map(|p| p.numel()).sum();
        assert_eq!(count, 16384 + 128 + 36864 + 128 + 16384 + 512);
    }

    // -----------------------------------------------------------------------
    // ResNet-18 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_resnet18_output_shape() {
        let model = resnet18::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 3 * 224 * 224], [1, 3, 224, 224], false);
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
        let input = leaf_4d(&vec![0.01; 3 * 224 * 224], [1, 3, 224, 224], false);
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
        let input = leaf_4d(&vec![0.01; 3 * 224 * 224], [1, 3, 224, 224], false);
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
        let input = leaf_4d(&vec![0.5; 4 * 4 * 4], [1, 4, 4, 4], true);

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
        let input = leaf_4d(&vec![0.5; 4 * 4 * 4], [1, 4, 4, 4], true);

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
