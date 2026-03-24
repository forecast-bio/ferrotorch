//! EfficientNet-B0 architecture (simplified).
//!
//! Follows the original paper: "EfficientNet: Rethinking Model Scaling for
//! Convolutional Neural Networks" (Tan & Le, 2019).
//!
//! **Simplification**: The real EfficientNet-B0 uses MBConv (mobile inverted
//! bottleneck) blocks with depthwise separable convolutions and squeeze-and-
//! excite attention. Since `ferrotorch_nn` does not yet provide depthwise
//! convolution or squeeze-excite, this implementation approximates the
//! architecture using standard `Conv2d` blocks. The channel counts, strides,
//! and number of blocks per stage match the original design so the overall
//! architecture shape is correct.
//!
//! Once depthwise convolution and SE blocks are added upstream, this module
//! can be upgraded without changing the public API.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::AdaptiveAvgPool2d;
use ferrotorch_nn::{Conv2d, Linear};

// ===========================================================================
// Helpers
// ===========================================================================

/// Create a convolution with the given kernel, stride, and padding (no bias).
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
// ConvBlock — Conv2d + ReLU with optional residual skip
// ===========================================================================

/// A single convolution block: `Conv2d -> ReLU`, with a residual skip
/// connection when the input and output shapes match exactly (same channel
/// count **and** same spatial size, i.e. stride == 1).
///
/// ```text
/// x -----> Conv2d -> ReLU -----> (+) -> out     (when skip is possible)
/// |                                ^
/// +--------------------------------+
///
/// x -----> Conv2d -> ReLU -----> out             (otherwise)
/// ```
pub struct ConvBlock<T: Float> {
    conv: Conv2d<T>,
    use_residual: bool,
    training: bool,
}

impl<T: Float> ConvBlock<T> {
    /// Create a new `ConvBlock`.
    ///
    /// * `in_ch`   -- input channels
    /// * `out_ch`  -- output channels
    /// * `kernel`  -- spatial kernel size (square)
    /// * `stride`  -- spatial stride
    /// * `padding` -- spatial padding
    pub fn new(
        in_ch: usize,
        out_ch: usize,
        kernel: usize,
        stride: usize,
        padding: usize,
    ) -> FerrotorchResult<Self> {
        let use_residual = stride == 1 && in_ch == out_ch;
        Ok(Self {
            conv: conv(in_ch, out_ch, kernel, stride, padding)?,
            use_residual,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for ConvBlock<T> {
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

// ===========================================================================
// Stage descriptor
// ===========================================================================

/// Describes one stage of the EfficientNet backbone.
struct StageConfig {
    /// Number of repeated blocks in this stage.
    num_blocks: usize,
    /// Output channel count.
    out_ch: usize,
    /// Stride for the **first** block (subsequent blocks use stride 1).
    stride: usize,
    /// Convolution kernel size.
    kernel: usize,
}

/// EfficientNet-B0 stage configuration.
///
/// The seven stages mirror the original paper's MBConv block groups:
///
/// | Stage | Operator  | Channels | Blocks | Stride | Kernel |
/// |-------|-----------|----------|--------|--------|--------|
/// |   1   | MBConv1   |    16    |   1    |   1    |   3    |
/// |   2   | MBConv6   |    24    |   2    |   2    |   3    |
/// |   3   | MBConv6   |    40    |   2    |   2    |   5    |
/// |   4   | MBConv6   |    80    |   3    |   2    |   3    |
/// |   5   | MBConv6   |   112    |   3    |   1    |   5    |
/// |   6   | MBConv6   |   192    |   4    |   2    |   5    |
/// |   7   | MBConv6   |   320    |   1    |   1    |   3    |
const EFFICIENTNET_B0_STAGES: [StageConfig; 7] = [
    StageConfig {
        num_blocks: 1,
        out_ch: 16,
        stride: 1,
        kernel: 3,
    },
    StageConfig {
        num_blocks: 2,
        out_ch: 24,
        stride: 2,
        kernel: 3,
    },
    StageConfig {
        num_blocks: 2,
        out_ch: 40,
        stride: 2,
        kernel: 5,
    },
    StageConfig {
        num_blocks: 3,
        out_ch: 80,
        stride: 2,
        kernel: 3,
    },
    StageConfig {
        num_blocks: 3,
        out_ch: 112,
        stride: 1,
        kernel: 5,
    },
    StageConfig {
        num_blocks: 4,
        out_ch: 192,
        stride: 2,
        kernel: 5,
    },
    StageConfig {
        num_blocks: 1,
        out_ch: 320,
        stride: 1,
        kernel: 3,
    },
];

// ===========================================================================
// EfficientNet
// ===========================================================================

/// A simplified EfficientNet model.
///
/// The architecture follows the EfficientNet-B0 design:
///
/// 1. Stem: 3x3 conv, stride 2 (3 -> 32 channels)
/// 2. Seven backbone stages with increasing channels (32 -> 16 -> 24 -> 40
///    -> 80 -> 112 -> 192 -> 320)
/// 3. Head: 1x1 conv to 1280 channels, global average pool, linear classifier
///
/// Batch normalization and squeeze-excite are omitted (not yet available in
/// `ferrotorch_nn`). Depthwise convolutions are replaced with standard
/// `Conv2d`.
pub struct EfficientNet<T: Float> {
    // Stem.
    stem_conv: Conv2d<T>,

    // Backbone stages (flattened into a single Vec for simplicity).
    stages: Vec<Box<dyn Module<T>>>,

    // Head.
    head_conv: Conv2d<T>,
    avgpool: AdaptiveAvgPool2d,
    fc: Linear<T>,

    training: bool,
}

impl<T: Float> EfficientNet<T> {
    /// Build an EfficientNet-B0 with the given number of output classes.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        // Stem: 3 -> 32, stride 2.
        let stem_conv = conv(3, 32, 3, 2, 1)?;

        // Build backbone stages.
        let mut stages: Vec<Box<dyn Module<T>>> = Vec::new();
        let mut in_ch = 32_usize;

        for stage_cfg in &EFFICIENTNET_B0_STAGES {
            let padding = stage_cfg.kernel / 2;

            // First block of the stage may change channels and/or stride.
            stages.push(Box::new(ConvBlock::new(
                in_ch,
                stage_cfg.out_ch,
                stage_cfg.kernel,
                stage_cfg.stride,
                padding,
            )?));

            // Remaining blocks: same channels, stride 1.
            for _ in 1..stage_cfg.num_blocks {
                stages.push(Box::new(ConvBlock::new(
                    stage_cfg.out_ch,
                    stage_cfg.out_ch,
                    stage_cfg.kernel,
                    1,
                    padding,
                )?));
            }

            in_ch = stage_cfg.out_ch;
        }

        // Head: 1x1 conv -> adaptive average pool -> linear.
        let head_conv = conv(320, 1280, 1, 1, 0)?;
        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let fc = Linear::new(1280, num_classes, true)?;

        Ok(Self {
            stem_conv,
            stages,
            head_conv,
            avgpool,
            fc,
            training: true,
        })
    }

    /// Total number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

// ---------------------------------------------------------------------------
// Module impl
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for EfficientNet<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Stem: conv -> relu.
        let x = self.stem_conv.forward(input)?;
        let x = relu(&x)?;

        // Backbone stages.
        let mut x = x;
        for block in &self.stages {
            x = block.forward(&x)?;
        }

        // Head: 1x1 conv -> relu -> global avg pool -> flatten -> fc.
        let x = self.head_conv.forward(&x)?;
        let x = relu(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;

        // Flatten to [B, 1280].
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;

        self.fc.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.stem_conv.parameters());
        for block in &self.stages {
            params.extend(block.parameters());
        }
        params.extend(self.head_conv.parameters());
        params.extend(self.fc.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.stem_conv.parameters_mut());
        for block in &mut self.stages {
            params.extend(block.parameters_mut());
        }
        params.extend(self.head_conv.parameters_mut());
        params.extend(self.fc.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();

        for (name, p) in self.stem_conv.named_parameters() {
            params.push((format!("stem_conv.{name}"), p));
        }

        for (i, block) in self.stages.iter().enumerate() {
            for (name, p) in block.named_parameters() {
                params.push((format!("stages.{i}.{name}"), p));
            }
        }

        for (name, p) in self.head_conv.named_parameters() {
            params.push((format!("head_conv.{name}"), p));
        }

        for (name, p) in self.fc.named_parameters() {
            params.push((format!("fc.{name}"), p));
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
// Convenience constructor
// ===========================================================================

/// Construct an EfficientNet-B0 model.
///
/// This is a simplified implementation using standard convolutions in place
/// of depthwise separable convolutions and squeeze-excite blocks.
///
/// # Parameters
///
/// * `num_classes` -- number of output classes (e.g. 1000 for ImageNet).
pub fn efficientnet_b0<T: Float>(num_classes: usize) -> FerrotorchResult<EfficientNet<T>> {
    EfficientNet::new(num_classes)
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
    // ConvBlock tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_conv_block_same_channels_has_residual() {
        let block = ConvBlock::<f32>::new(32, 32, 3, 1, 1).unwrap();
        assert!(block.use_residual);
    }

    #[test]
    fn test_conv_block_different_channels_no_residual() {
        let block = ConvBlock::<f32>::new(32, 64, 3, 1, 1).unwrap();
        assert!(!block.use_residual);
    }

    #[test]
    fn test_conv_block_stride2_no_residual() {
        let block = ConvBlock::<f32>::new(32, 32, 3, 2, 1).unwrap();
        assert!(!block.use_residual);
    }

    #[test]
    fn test_conv_block_forward_shape() {
        let block = ConvBlock::<f32>::new(16, 24, 3, 2, 1).unwrap();
        let input = leaf_4d(&vec![0.1; 1 * 16 * 8 * 8], [1, 16, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 24, 4, 4]);
    }

    #[test]
    fn test_conv_block_residual_forward() {
        let block = ConvBlock::<f32>::new(16, 16, 3, 1, 1).unwrap();
        let input = leaf_4d(&vec![0.1; 1 * 16 * 8 * 8], [1, 16, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 16, 8, 8]);
    }

    #[test]
    fn test_conv_block_5x5_forward_shape() {
        let block = ConvBlock::<f32>::new(24, 40, 5, 2, 2).unwrap();
        let input = leaf_4d(&vec![0.1; 1 * 24 * 16 * 16], [1, 24, 16, 16], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 40, 8, 8]);
    }

    // -----------------------------------------------------------------------
    // EfficientNet-B0 output shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_efficientnet_b0_output_shape() {
        let model = efficientnet_b0::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 1 * 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    // -----------------------------------------------------------------------
    // Parameter count
    // -----------------------------------------------------------------------

    #[test]
    fn test_efficientnet_b0_param_count() {
        let model = efficientnet_b0::<f32>(1000).unwrap();
        let total = model.num_parameters();

        // Expected parameter count (all weights, no bias):
        //
        // Stem:      3 * 32 * 3 * 3                    =      864
        //
        // Stage 1 (1 block):  32 * 16  * 3 * 3         =    4_608
        // Stage 2 (2 blocks): 16 * 24  * 3 * 3         =    3_456
        //                     24 * 24  * 3 * 3          =    5_184
        // Stage 3 (2 blocks): 24 * 40  * 5 * 5         =   24_000
        //                     40 * 40  * 5 * 5          =   40_000
        // Stage 4 (3 blocks): 40 * 80  * 3 * 3         =   28_800
        //                     80 * 80  * 3 * 3  (x2)    =  115_200
        // Stage 5 (3 blocks): 80 * 112 * 5 * 5         =  224_000
        //                    112 * 112 * 5 * 5  (x2)    =  627_200
        // Stage 6 (4 blocks):112 * 192 * 5 * 5         =  537_600
        //                    192 * 192 * 5 * 5  (x3)    = 2_764_800
        // Stage 7 (1 block): 192 * 320 * 3 * 3         =  552_960
        //
        // Head conv: 320 * 1280 * 1 * 1                = 409_600
        // FC:        1280 * 1000 + 1000 (bias)          = 1_281_000
        //
        // Total = 864 + 4608 + 3456 + 5184 + 24000 + 40000 + 28800
        //       + 115200 + 224000 + 627200 + 537600 + 2764800
        //       + 552960 + 409600 + 1281000
        //       = 6_619_272
        //
        // We check a reasonable range around this.
        assert!(
            total > 6_500_000,
            "EfficientNet-B0 should have >6.5M params, got {total}"
        );
        assert!(
            total < 6_800_000,
            "EfficientNet-B0 should have <6.8M params, got {total}"
        );
    }

    // -----------------------------------------------------------------------
    // Custom num_classes
    // -----------------------------------------------------------------------

    #[test]
    fn test_efficientnet_b0_custom_classes() {
        let model = efficientnet_b0::<f32>(10).unwrap();
        let input = leaf_4d(&vec![0.01; 2 * 3 * 224 * 224], [2, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 10]);
    }

    // -----------------------------------------------------------------------
    // Named parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_efficientnet_b0_named_parameters_prefixes() {
        let model = efficientnet_b0::<f32>(1000).unwrap();
        let named = model.named_parameters();
        assert!(!named.is_empty());

        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("stem_conv.")));
        assert!(names.iter().any(|n| n.starts_with("stages.")));
        assert!(names.iter().any(|n| n.starts_with("head_conv.")));
        assert!(names.iter().any(|n| n.starts_with("fc.")));
    }

    // -----------------------------------------------------------------------
    // Train / eval
    // -----------------------------------------------------------------------

    #[test]
    fn test_efficientnet_train_eval() {
        let mut model = efficientnet_b0::<f32>(10).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }

    // -----------------------------------------------------------------------
    // Gradient flow through residual connections
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_flow_through_conv_block() {
        let block = ConvBlock::<f32>::new(4, 4, 3, 1, 1).unwrap();
        assert!(block.use_residual);

        let input = leaf_4d(&vec![0.5; 1 * 4 * 4 * 4], [1, 4, 4, 4], true);
        let output = block.forward(&input).unwrap();

        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap();
        assert!(grad.is_some(), "input should have gradients");
        let grad_data = grad.unwrap().data().unwrap().to_vec();
        let any_nonzero = grad_data.iter().any(|&g| g.abs() > 1e-10);
        assert!(any_nonzero, "gradients should flow through residual path");
    }

    // -----------------------------------------------------------------------
    // Send + Sync
    // -----------------------------------------------------------------------

    #[test]
    fn test_efficientnet_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<EfficientNet<f32>>();
        assert_send_sync::<ConvBlock<f32>>();
    }
}
