//! ConvNeXt-Tiny architecture.
//!
//! Implements the ConvNeXt-Tiny model from "A ConvNet for the 2020s"
//! (Liu et al., 2022). This implementation replaces the depthwise 7x7
//! convolution with a regular 7x7 convolution because grouped/depthwise
//! convolutions are not yet available in `ferrotorch_nn`. This changes the
//! parameter count but preserves the overall architecture.
//!
//! Architecture summary:
//!
//! 1. Patchify stem: `Conv2d(3, 96, kernel=4, stride=4)` + `LayerNorm`
//! 2. Four stages: `[3, 3, 9, 3]` blocks with channels `[96, 192, 384, 768]`
//! 3. Each block: `Conv2d(C, C, 7, pad=3)` -> `LayerNorm` -> `Conv2d(C, 4*C, 1)` -> `GELU` -> `Conv2d(4*C, C, 1)` + residual
//! 4. Downsampling between stages: `LayerNorm` -> `Conv2d(C, 2*C, kernel=2, stride=2)`
//! 5. Head: `AdaptiveAvgPool2d(1,1)` -> `LayerNorm` -> `Linear(768, num_classes)`
//!
//! The `LayerNorm` layers operate on the channel dimension. Since `LayerNorm`
//! normalizes over the last dimension, the forward pass permutes the data to
//! `[B, H, W, C]`, applies LayerNorm, and permutes back to `[B, C, H, W]`.

use ferrotorch_core::grad_fns::arithmetic::add;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use ferrotorch_nn::activation::GELU;
use ferrotorch_nn::conv::Conv2d;
use ferrotorch_nn::linear::Linear;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::LayerNorm;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::AdaptiveAvgPool2d;

// ===========================================================================
// Helpers
// ===========================================================================

/// Permute a 4-D tensor from `[B, C, H, W]` to `[B, H, W, C]`.
fn nhwc_from_nchw<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    let (b, c, h, w) = (shape[0], shape[1], shape[2], shape[3]);
    let device = input.device();
    let data = input.data_vec()?;
    let total = b * c * h * w;
    let mut out = vec![<T as num_traits::Zero>::zero(); total];

    for bi in 0..b {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = bi * c * h * w + ci * h * w + hi * w + wi;
                    let dst = bi * h * w * c + hi * w * c + wi * c + ci;
                    out[dst] = data[src];
                }
            }
        }
    }

    Tensor::from_storage(
        TensorStorage::cpu(out),
        vec![b, h, w, c],
        input.requires_grad(),
    )?
    .to(device)
}

/// Permute a 4-D tensor from `[B, H, W, C]` to `[B, C, H, W]`.
fn nchw_from_nhwc<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    let (b, h, w, c) = (shape[0], shape[1], shape[2], shape[3]);
    let device = input.device();
    let data = input.data_vec()?;
    let total = b * c * h * w;
    let mut out = vec![<T as num_traits::Zero>::zero(); total];

    for bi in 0..b {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let src = bi * h * w * c + hi * w * c + wi * c + ci;
                    let dst = bi * c * h * w + ci * h * w + hi * w + wi;
                    out[dst] = data[src];
                }
            }
        }
    }

    Tensor::from_storage(
        TensorStorage::cpu(out),
        vec![b, c, h, w],
        input.requires_grad(),
    )?
    .to(device)
}

/// Apply `LayerNorm` on the channel dimension of a `[B, C, H, W]` tensor.
///
/// Permutes to `[B, H, W, C]`, applies LayerNorm over the last dim, then
/// permutes back to `[B, C, H, W]`.
fn channel_layer_norm<T: Float>(
    ln: &LayerNorm<T>,
    input: &Tensor<T>,
) -> FerrotorchResult<Tensor<T>> {
    let nhwc = nhwc_from_nchw(input)?;
    let normed = ln.forward(&nhwc)?;
    nchw_from_nhwc(&normed)
}

// ===========================================================================
// ConvNeXtBlock
// ===========================================================================

/// A single ConvNeXt block.
///
/// ```text
/// x -> Conv2d(C, C, 7, pad=3) -> LayerNorm -> Conv2d(C, 4*C, 1) -> GELU -> Conv2d(4*C, C, 1) -> (+x) -> out
/// ```
///
/// The first convolution is a regular 7x7 (replacing the depthwise 7x7 from
/// the original paper, since grouped convolutions are not yet available).
pub struct ConvNeXtBlock<T: Float> {
    dwconv: Conv2d<T>,
    norm: LayerNorm<T>,
    pwconv1: Conv2d<T>,
    pwconv2: Conv2d<T>,
    gelu: GELU,
    training: bool,
}

impl<T: Float> ConvNeXtBlock<T> {
    /// Create a new `ConvNeXtBlock`.
    ///
    /// * `channels` -- number of input and output channels.
    pub fn new(channels: usize) -> FerrotorchResult<Self> {
        // Regular 7x7 conv (replaces depthwise 7x7).
        let dwconv = Conv2d::new(channels, channels, (7, 7), (1, 1), (3, 3), false)?;
        // LayerNorm over the channel dimension.
        let norm = LayerNorm::new(vec![channels], 1e-6, true)?;
        // Pointwise expansion: C -> 4*C.
        let pwconv1 = Conv2d::new(channels, 4 * channels, (1, 1), (1, 1), (0, 0), false)?;
        // Pointwise contraction: 4*C -> C.
        let pwconv2 = Conv2d::new(4 * channels, channels, (1, 1), (1, 1), (0, 0), false)?;
        let gelu = GELU::new();

        Ok(Self {
            dwconv,
            norm,
            pwconv1,
            pwconv2,
            gelu,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for ConvNeXtBlock<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // 7x7 convolution.
        let out = self.dwconv.forward(input)?;

        // LayerNorm on the channel dimension.
        let out = channel_layer_norm(&self.norm, &out)?;

        // Pointwise 1x1 expand -> GELU -> 1x1 contract.
        let out = self.pwconv1.forward(&out)?;
        let out = self.gelu.forward(&out)?;
        let out = self.pwconv2.forward(&out)?;

        // Residual connection.
        add(&out, input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.dwconv.parameters());
        params.extend(self.norm.parameters());
        params.extend(self.pwconv1.parameters());
        params.extend(self.pwconv2.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.dwconv.parameters_mut());
        params.extend(self.norm.parameters_mut());
        params.extend(self.pwconv1.parameters_mut());
        params.extend(self.pwconv2.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.dwconv.named_parameters() {
            params.push((format!("dwconv.{name}"), p));
        }
        for (name, p) in self.norm.named_parameters() {
            params.push((format!("norm.{name}"), p));
        }
        for (name, p) in self.pwconv1.named_parameters() {
            params.push((format!("pwconv1.{name}"), p));
        }
        for (name, p) in self.pwconv2.named_parameters() {
            params.push((format!("pwconv2.{name}"), p));
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
// Downsample layer between stages
// ===========================================================================

/// Downsampling layer between ConvNeXt stages.
///
/// ```text
/// x -> LayerNorm(C) -> Conv2d(C, 2*C, kernel=2, stride=2) -> out
/// ```
struct Downsample<T: Float> {
    norm: LayerNorm<T>,
    conv: Conv2d<T>,
    training: bool,
}

impl<T: Float> Downsample<T> {
    fn new(in_channels: usize, out_channels: usize) -> FerrotorchResult<Self> {
        let norm = LayerNorm::new(vec![in_channels], 1e-6, true)?;
        let conv = Conv2d::new(in_channels, out_channels, (2, 2), (2, 2), (0, 0), false)?;
        Ok(Self {
            norm,
            conv,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for Downsample<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = channel_layer_norm(&self.norm, input)?;
        self.conv.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.norm.parameters());
        params.extend(self.conv.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.norm.parameters_mut());
        params.extend(self.conv.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.norm.named_parameters() {
            params.push((format!("norm.{name}"), p));
        }
        for (name, p) in self.conv.named_parameters() {
            params.push((format!("conv.{name}"), p));
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
// ConvNeXt
// ===========================================================================

/// A ConvNeXt model.
///
/// The architecture follows the ConvNeXt design:
///
/// 1. Patchify stem: `Conv2d(3, 96, 4, 4)` + `LayerNorm`
/// 2. Four residual stages with downsampling between them
/// 3. Global average pool + `LayerNorm` + linear classifier
///
/// The depthwise 7x7 convolutions from the original paper are replaced with
/// regular 7x7 convolutions. This significantly increases the parameter count
/// but proves the architecture with the available primitives.
pub struct ConvNeXt<T: Float> {
    // Stem.
    stem_conv: Conv2d<T>,
    stem_norm: LayerNorm<T>,

    // Stages: each stage is a list of ConvNeXtBlocks.
    stages: Vec<Vec<ConvNeXtBlock<T>>>,

    // Downsampling layers between stages (3 total, between stages 0-1, 1-2, 2-3).
    downsamples: Vec<Downsample<T>>,

    // Head.
    avgpool: AdaptiveAvgPool2d,
    head_norm: LayerNorm<T>,
    head_fc: Linear<T>,

    training: bool,
}

impl<T: Float> ConvNeXt<T> {
    /// Build a ConvNeXt model.
    ///
    /// * `depths` -- number of blocks per stage, e.g., `[3, 3, 9, 3]`.
    /// * `dims` -- channel count per stage, e.g., `[96, 192, 384, 768]`.
    /// * `num_classes` -- number of output classes.
    pub fn new(depths: &[usize], dims: &[usize], num_classes: usize) -> FerrotorchResult<Self> {
        if depths.len() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConvNeXt expects 4 stages (depths.len() == 4), got {}",
                    depths.len()
                ),
            });
        }
        if dims.len() != 4 {
            return Err(FerrotorchError::InvalidArgument {
                message: format!(
                    "ConvNeXt expects 4 channel dimensions (dims.len() == 4), got {}",
                    dims.len()
                ),
            });
        }

        // Patchify stem: Conv2d(3, dims[0], 4, 4) + LayerNorm.
        let stem_conv = Conv2d::new(3, dims[0], (4, 4), (4, 4), (0, 0), false)?;
        let stem_norm = LayerNorm::new(vec![dims[0]], 1e-6, true)?;

        // Build stages.
        let mut stages = Vec::with_capacity(4);
        for s in 0..4 {
            let mut blocks = Vec::with_capacity(depths[s]);
            for _ in 0..depths[s] {
                blocks.push(ConvNeXtBlock::new(dims[s])?);
            }
            stages.push(blocks);
        }

        // Downsampling layers between stages.
        let mut downsamples = Vec::with_capacity(3);
        for s in 0..3 {
            downsamples.push(Downsample::new(dims[s], dims[s + 1])?);
        }

        // Classification head.
        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let head_norm = LayerNorm::new(vec![dims[3]], 1e-6, true)?;
        let head_fc = Linear::new(dims[3], num_classes, true)?;

        Ok(Self {
            stem_conv,
            stem_norm,
            stages,
            downsamples,
            avgpool,
            head_norm,
            head_fc,
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

impl<T: Float> Module<T> for ConvNeXt<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Stem: patchify conv + LayerNorm.
        let x = self.stem_conv.forward(input)?;
        let mut x = channel_layer_norm(&self.stem_norm, &x)?;

        // Stage 0.
        for block in &self.stages[0] {
            x = block.forward(&x)?;
        }

        // Stages 1..3 with downsampling.
        for s in 1..4 {
            x = self.downsamples[s - 1].forward(&x)?;
            for block in &self.stages[s] {
                x = block.forward(&x)?;
            }
        }

        // Head: global average pool -> flatten -> LayerNorm -> linear.
        let x = Module::<T>::forward(&self.avgpool, &x)?;

        // Flatten: [B, C, 1, 1] -> [B, C].
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;

        // LayerNorm on the feature dimension.
        let x = self.head_norm.forward(&x)?;

        // Classifier.
        self.head_fc.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.stem_conv.parameters());
        params.extend(self.stem_norm.parameters());
        for stage in &self.stages {
            for block in stage {
                params.extend(block.parameters());
            }
        }
        for ds in &self.downsamples {
            params.extend(ds.parameters());
        }
        params.extend(self.head_norm.parameters());
        params.extend(self.head_fc.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.stem_conv.parameters_mut());
        params.extend(self.stem_norm.parameters_mut());
        for stage in &mut self.stages {
            for block in stage {
                params.extend(block.parameters_mut());
            }
        }
        for ds in &mut self.downsamples {
            params.extend(ds.parameters_mut());
        }
        params.extend(self.head_norm.parameters_mut());
        params.extend(self.head_fc.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();

        for (name, p) in self.stem_conv.named_parameters() {
            params.push((format!("stem.conv.{name}"), p));
        }
        for (name, p) in self.stem_norm.named_parameters() {
            params.push((format!("stem.norm.{name}"), p));
        }

        for (s, stage) in self.stages.iter().enumerate() {
            for (i, block) in stage.iter().enumerate() {
                for (name, p) in block.named_parameters() {
                    params.push((format!("stages.{s}.{i}.{name}"), p));
                }
            }
        }

        for (i, ds) in self.downsamples.iter().enumerate() {
            for (name, p) in ds.named_parameters() {
                params.push((format!("downsample.{i}.{name}"), p));
            }
        }

        for (name, p) in self.head_norm.named_parameters() {
            params.push((format!("head.norm.{name}"), p));
        }
        for (name, p) in self.head_fc.named_parameters() {
            params.push((format!("head.fc.{name}"), p));
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
// IntermediateFeatures — CL-499
// ===========================================================================

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for ConvNeXt<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();

        let x = self.stem_conv.forward(input)?;
        let mut x = channel_layer_norm(&self.stem_norm, &x)?;
        out.insert("stem".to_string(), x.clone());

        // Stage 0 (no downsample before it).
        for block in &self.stages[0] {
            x = block.forward(&x)?;
        }
        out.insert("stage0".to_string(), x.clone());

        // Stages 1..3 with downsampling.
        for s in 1..4 {
            x = self.downsamples[s - 1].forward(&x)?;
            for block in &self.stages[s] {
                x = block.forward(&x)?;
            }
            out.insert(format!("stage{s}"), x.clone());
        }

        let x = Module::<T>::forward(&self.avgpool, &x)?;
        out.insert("avgpool".to_string(), x.clone());
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        let x = self.head_norm.forward(&x)?;
        let logits = self.head_fc.forward(&x)?;
        out.insert("head_fc".to_string(), logits);
        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        vec![
            "stem".to_string(),
            "stage0".to_string(),
            "stage1".to_string(),
            "stage2".to_string(),
            "stage3".to_string(),
            "avgpool".to_string(),
            "head_fc".to_string(),
        ]
    }
}

/// Construct a ConvNeXt-Tiny model.
///
/// Architecture: `[3, 3, 9, 3]` blocks, channels `[96, 192, 384, 768]`.
///
/// Note: Because depthwise convolutions are replaced with regular convolutions,
/// the parameter count is significantly larger than the original ConvNeXt-Tiny
/// (~28M). The regular-conv variant has ~187M parameters.
pub fn convnext_tiny<T: Float>(num_classes: usize) -> FerrotorchResult<ConvNeXt<T>> {
    ConvNeXt::new(&[3, 3, 9, 3], &[96, 192, 384, 768], num_classes)
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
    // ConvNeXtBlock tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_convnext_block_output_shape() {
        let block = ConvNeXtBlock::<f32>::new(96).unwrap();
        let input = leaf_4d(&vec![0.01; 96 * 8 * 8], [1, 96, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        // Same spatial dims and channels (residual block).
        assert_eq!(output.shape(), &[1, 96, 8, 8]);
    }

    #[test]
    fn test_convnext_block_parameter_count() {
        let c: usize = 96;
        let block = ConvNeXtBlock::<f32>::new(c).unwrap();
        let count: usize = block.parameters().iter().map(|p| p.numel()).sum();

        // dwconv: C * C * 7 * 7 (regular 7x7)
        // norm: weight(C) + bias(C) = 2*C
        // pwconv1: C * 4*C * 1 * 1 = 4*C^2
        // pwconv2: 4*C * C * 1 * 1 = 4*C^2
        let expected = c * c * 7 * 7 + 2 * c + 4 * c * c + 4 * c * c;
        assert_eq!(count, expected);
    }

    #[test]
    fn test_convnext_block_batch_2() {
        let block = ConvNeXtBlock::<f32>::new(48).unwrap();
        let input = leaf_4d(&vec![0.01; 2 * 48 * 8 * 8], [2, 48, 8, 8], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 48, 8, 8]);
    }

    // -----------------------------------------------------------------------
    // Downsample tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_downsample_output_shape() {
        let ds = Downsample::<f32>::new(96, 192).unwrap();
        let input = leaf_4d(&vec![0.01; 96 * 8 * 8], [1, 96, 8, 8], false);
        let output = no_grad(|| ds.forward(&input).unwrap());
        // Spatial dims halve, channels double.
        assert_eq!(output.shape(), &[1, 192, 4, 4]);
    }

    #[test]
    fn test_downsample_parameter_count() {
        let ds = Downsample::<f32>::new(96, 192).unwrap();
        let count: usize = ds.parameters().iter().map(|p| p.numel()).sum();
        // LayerNorm(96): 2*96 = 192
        // Conv2d(96, 192, 2, 2): 96 * 192 * 2 * 2 = 73728
        let expected = 2 * 96 + 96 * 192 * 2 * 2;
        assert_eq!(count, expected);
    }

    // -----------------------------------------------------------------------
    // Permutation helpers
    // -----------------------------------------------------------------------

    #[test]
    fn test_nhwc_nchw_roundtrip() {
        let input = leaf_4d(&vec![0.5; 2 * 3 * 4 * 4], [2, 3, 4, 4], false);
        let nhwc = nhwc_from_nchw(&input).unwrap();
        assert_eq!(nhwc.shape(), &[2, 4, 4, 3]);
        let nchw = nchw_from_nhwc(&nhwc).unwrap();
        assert_eq!(nchw.shape(), &[2, 3, 4, 4]);

        // Data should be identical after roundtrip.
        let orig = input.data().unwrap();
        let back = nchw.data().unwrap();
        assert_eq!(orig.len(), back.len());
        for (a, b) in orig.iter().zip(back.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    // -----------------------------------------------------------------------
    // ConvNeXt-Tiny: forward shape
    // -----------------------------------------------------------------------

    #[test]
    fn test_convnext_tiny_output_shape() {
        let model = convnext_tiny::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    // -----------------------------------------------------------------------
    // ConvNeXt-Tiny: parameter count
    // -----------------------------------------------------------------------

    #[test]
    fn test_convnext_tiny_param_count() {
        let model = convnext_tiny::<f32>(1000).unwrap();
        let total = model.num_parameters();

        // With regular (non-depthwise) 7x7 convolutions, the parameter count
        // is much larger than the original ConvNeXt-Tiny (~28M). The exact
        // expected count is ~186.7M.
        //
        // Breakdown per stage (C = channel width):
        //   block params = C*C*49 + 2*C + 4*C^2 + 4*C^2 = 57*C^2 + 2*C
        //   stem: 3*96*16 + 2*96 = 4800
        //   stage0: 3*(57*96^2 + 192) = 1,576,512
        //   ds0: 2*96 + 96*192*4 = 73,920
        //   stage1: 3*(57*192^2 + 384) = 6,304,896
        //   ds1: 2*192 + 192*384*4 = 295,296
        //   stage2: 9*(57*384^2 + 768) = 75,651,840
        //   ds2: 2*384 + 384*768*4 = 1,180,416
        //   stage3: 3*(57*768^2 + 1536) = 100,864,512
        //   head: 2*768 + 768*1000 + 1000 = 770,536
        //   total: ~186,722,728
        assert!(
            total > 180_000_000,
            "ConvNeXt-Tiny should have >180M params (regular conv), got {total}"
        );
        assert!(
            total < 195_000_000,
            "ConvNeXt-Tiny should have <195M params (regular conv), got {total}"
        );
    }

    // -----------------------------------------------------------------------
    // ConvNeXt-Tiny: custom classes
    // -----------------------------------------------------------------------

    #[test]
    fn test_convnext_tiny_custom_classes() {
        let model = convnext_tiny::<f32>(10).unwrap();
        let input = leaf_4d(&vec![0.01; 2 * 3 * 224 * 224], [2, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 10]);
    }

    // -----------------------------------------------------------------------
    // Small ConvNeXt forward (fast test)
    // -----------------------------------------------------------------------

    #[test]
    fn test_small_convnext_forward() {
        // A tiny ConvNeXt for fast testing: small channels, few blocks.
        let model = ConvNeXt::<f32>::new(&[1, 1, 1, 1], &[8, 16, 32, 64], 10).unwrap();

        let input = leaf_4d(&vec![0.1; 2 * 3 * 32 * 32], [2, 3, 32, 32], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 10]);

        // Verify output is finite.
        let data = output.data().unwrap();
        for &v in data.iter() {
            assert!(v.is_finite(), "output contains non-finite value: {v}");
        }
    }

    // -----------------------------------------------------------------------
    // Named parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_convnext_named_parameters_prefixes() {
        let model = ConvNeXt::<f32>::new(&[1, 1, 1, 1], &[8, 16, 32, 64], 10).unwrap();
        let named = model.named_parameters();
        assert!(!named.is_empty());

        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("stem.conv.")));
        assert!(names.iter().any(|n| n.starts_with("stem.norm.")));
        assert!(names.iter().any(|n| n.starts_with("stages.0.")));
        assert!(names.iter().any(|n| n.starts_with("stages.3.")));
        assert!(names.iter().any(|n| n.starts_with("downsample.0.")));
        assert!(names.iter().any(|n| n.starts_with("downsample.2.")));
        assert!(names.iter().any(|n| n.starts_with("head.norm.")));
        assert!(names.iter().any(|n| n.starts_with("head.fc.")));
    }

    // -----------------------------------------------------------------------
    // Train / eval
    // -----------------------------------------------------------------------

    #[test]
    fn test_convnext_train_eval() {
        let mut model = ConvNeXt::<f32>::new(&[1, 1, 1, 1], &[8, 16, 32, 64], 10).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }

    // -----------------------------------------------------------------------
    // Gradient flow through residual
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_flow_through_convnext_block() {
        let block = ConvNeXtBlock::<f32>::new(4).unwrap();
        let input = leaf_4d(&vec![0.5; 4 * 4 * 4], [1, 4, 4, 4], true);

        let output = block.forward(&input).unwrap();

        // Reduce to scalar for backward.
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        // Input should have received gradients through the skip connection.
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
    fn test_convnext_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<ConvNeXt<f32>>();
        assert_send_sync::<ConvNeXtBlock<f32>>();
    }
}
