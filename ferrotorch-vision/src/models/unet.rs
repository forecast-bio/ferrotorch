//! U-Net architecture for semantic segmentation.
//!
//! Follows the original paper: "U-Net: Convolutional Networks for Biomedical
//! Image Segmentation" (Ronneberger et al., 2015).
//!
//! Architecture overview:
//!
//! ```text
//! Encoder (contracting):  3 -> 64 -> 128 -> 256 -> 512
//! Bottleneck:            512 -> 1024
//! Decoder (expanding):   1024 -> 512 -> 256 -> 128 -> 64
//! Head:                  64 -> num_classes (1x1 conv)
//! ```
//!
//! Each encoder stage applies two 3x3 convolutions with ReLU, then 2x2 max
//! pooling. Each decoder stage upsamples 2x (nearest neighbor), halves channels
//! with a 1x1 conv, concatenates the matching encoder skip connection, then
//! applies two 3x3 convolutions with ReLU.
//!
//! Batch normalization is omitted (not yet in `ferrotorch_nn`).

use std::sync::Arc;

use ferrotorch_core::autograd::no_grad::is_grad_enabled;
use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::shape::cat;
use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::tensor::{GradFn, Tensor};
use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float};

use ferrotorch_nn::Conv2d;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::MaxPool2d;

// ===========================================================================
// Nearest-neighbor 2x upsample (differentiable)
// ===========================================================================

/// Backward for `upsample_nearest_2x`.
///
/// VJP: sum each 2x2 block back into the corresponding input element.
#[derive(Debug)]
struct UpsampleNearest2xBackward<T: Float> {
    input: Tensor<T>,
    input_shape: Vec<usize>,
}

impl<T: Float> UpsampleNearest2xBackward<T> {
    fn new(input: Tensor<T>, input_shape: Vec<usize>) -> Self {
        Self { input, input_shape }
    }
}

impl<T: Float> GradFn<T> for UpsampleNearest2xBackward<T> {
    fn backward(&self, grad_output: &Tensor<T>) -> FerrotorchResult<Vec<Option<Tensor<T>>>> {
        if !self.input.requires_grad() {
            return Ok(vec![None]);
        }

        let [batch, channels, in_h, in_w] = [
            self.input_shape[0],
            self.input_shape[1],
            self.input_shape[2],
            self.input_shape[3],
        ];
        let out_h = in_h * 2;
        let out_w = in_w * 2;
        let device = grad_output.device();

        let grad_data = grad_output.data_vec()?;
        let mut grad_input = vec![<T as num_traits::Zero>::zero(); batch * channels * in_h * in_w];

        for b in 0..batch {
            for c in 0..channels {
                for ih in 0..in_h {
                    for iw in 0..in_w {
                        let dst = b * channels * in_h * in_w + c * in_h * in_w + ih * in_w + iw;
                        // Sum the 2x2 block that was replicated from this element.
                        let oh = ih * 2;
                        let ow = iw * 2;
                        let src_base = b * channels * out_h * out_w + c * out_h * out_w;
                        let g00 = grad_data[src_base + oh * out_w + ow];
                        let g01 = grad_data[src_base + oh * out_w + ow + 1];
                        let g10 = grad_data[src_base + (oh + 1) * out_w + ow];
                        let g11 = grad_data[src_base + (oh + 1) * out_w + ow + 1];
                        grad_input[dst] = g00 + g01 + g10 + g11;
                    }
                }
            }
        }

        let gi = Tensor::from_storage(
            TensorStorage::cpu(grad_input),
            self.input_shape.clone(),
            false,
        )?
        .to(device)?;
        Ok(vec![Some(gi)])
    }

    fn inputs(&self) -> Vec<&Tensor<T>> {
        vec![&self.input]
    }

    fn name(&self) -> &'static str {
        "UpsampleNearest2xBackward"
    }
}

/// Nearest-neighbor 2x upsample: `[B, C, H, W]` -> `[B, C, 2H, 2W]`.
///
/// Each spatial element is replicated into a 2x2 block. This operation
/// participates in the autograd graph so gradients flow through the decoder.
fn upsample_nearest_2x<T: Float>(input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
    let shape = input.shape();
    if shape.len() != 4 {
        return Err(FerrotorchError::InvalidArgument {
            message: format!(
                "upsample_nearest_2x: expected 4-D tensor [B, C, H, W], got shape {shape:?}"
            ),
        });
    }

    let batch = shape[0];
    let channels = shape[1];
    let in_h = shape[2];
    let in_w = shape[3];
    let out_h = in_h * 2;
    let out_w = in_w * 2;
    let device = input.device();

    let data = input.data_vec()?;
    let out_numel = batch * channels * out_h * out_w;
    let mut out_data = Vec::with_capacity(out_numel);

    for b in 0..batch {
        for c in 0..channels {
            let base = b * channels * in_h * in_w + c * in_h * in_w;
            for ih in 0..in_h {
                // Two output rows per input row.
                for _rep_h in 0..2 {
                    for iw in 0..in_w {
                        let val = data[base + ih * in_w + iw];
                        out_data.push(val);
                        out_data.push(val);
                    }
                }
            }
        }
    }

    let out_shape = vec![batch, channels, out_h, out_w];

    if is_grad_enabled() && input.requires_grad() {
        let grad_fn = Arc::new(UpsampleNearest2xBackward::new(
            input.clone(),
            shape.to_vec(),
        ));
        Tensor::from_operation(TensorStorage::cpu(out_data), out_shape, grad_fn)?.to(device)
    } else {
        Tensor::from_storage(TensorStorage::cpu(out_data), out_shape, false)?.to(device)
    }
}

// ===========================================================================
// Encoder block
// ===========================================================================

/// An encoder (contracting) block: two 3x3 convolutions + ReLU, no pooling.
///
/// Pooling is applied externally so the pre-pool feature map can be stored
/// as a skip connection.
struct EncoderBlock<T: Float> {
    conv1: Conv2d<T>,
    conv2: Conv2d<T>,
}

impl<T: Float> EncoderBlock<T> {
    /// Create a new encoder block.
    ///
    /// * `in_ch`  -- input channels.
    /// * `out_ch` -- output channels (both convs output `out_ch`).
    fn new(in_ch: usize, out_ch: usize) -> FerrotorchResult<Self> {
        let conv1 = Conv2d::new(in_ch, out_ch, (3, 3), (1, 1), (1, 1), false)?;
        let conv2 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), false)?;
        Ok(Self { conv1, conv2 })
    }

    /// Forward: Conv -> ReLU -> Conv -> ReLU.
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.conv1.forward(input)?;
        let x = relu(&x)?;
        let x = self.conv2.forward(&x)?;
        relu(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.conv1.parameters();
        p.extend(self.conv2.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.conv1.parameters_mut();
        p.extend(self.conv2.parameters_mut());
        p
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.conv1.named_parameters() {
            params.push((format!("{prefix}.conv1.{name}"), p));
        }
        for (name, p) in self.conv2.named_parameters() {
            params.push((format!("{prefix}.conv2.{name}"), p));
        }
        params
    }
}

// ===========================================================================
// Decoder block
// ===========================================================================

/// A decoder (expanding) block: upsample 2x, 1x1 conv to halve channels,
/// concatenate with skip connection, then two 3x3 convs + ReLU.
struct DecoderBlock<T: Float> {
    /// 1x1 conv to halve the channel count after upsampling.
    reduce: Conv2d<T>,
    /// First 3x3 conv after concatenation (input channels = 2 * out_ch from
    /// skip + reduced upsampled, so in_channels = out_ch + out_ch = 2 * out_ch).
    conv1: Conv2d<T>,
    /// Second 3x3 conv.
    conv2: Conv2d<T>,
}

impl<T: Float> DecoderBlock<T> {
    /// Create a new decoder block.
    ///
    /// * `in_ch`  -- channels from the previous layer (before upsampling).
    /// * `out_ch` -- output channels. The skip connection also has `out_ch`
    ///   channels, so after concatenation we have `2 * out_ch`.
    fn new(in_ch: usize, out_ch: usize) -> FerrotorchResult<Self> {
        // 1x1 conv to halve channels: in_ch -> out_ch.
        let reduce = Conv2d::new(in_ch, out_ch, (1, 1), (1, 1), (0, 0), false)?;
        // After cat with skip: 2 * out_ch -> out_ch.
        let conv1 = Conv2d::new(2 * out_ch, out_ch, (3, 3), (1, 1), (1, 1), false)?;
        let conv2 = Conv2d::new(out_ch, out_ch, (3, 3), (1, 1), (1, 1), false)?;
        Ok(Self {
            reduce,
            conv1,
            conv2,
        })
    }

    /// Forward: upsample -> reduce channels -> cat(skip) -> conv -> relu -> conv -> relu.
    fn forward(&self, input: &Tensor<T>, skip: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Upsample: [B, in_ch, H, W] -> [B, in_ch, 2H, 2W].
        let x = upsample_nearest_2x(input)?;
        // Reduce: [B, in_ch, 2H, 2W] -> [B, out_ch, 2H, 2W].
        let x = self.reduce.forward(&x)?;
        // Concatenate with skip along the channel dimension.
        // skip: [B, out_ch, 2H, 2W], x: [B, out_ch, 2H, 2W]
        // result: [B, 2*out_ch, 2H, 2W].
        let x = cat(&[x, skip.clone()], 1)?;
        // Two 3x3 convolutions with ReLU.
        let x = self.conv1.forward(&x)?;
        let x = relu(&x)?;
        let x = self.conv2.forward(&x)?;
        relu(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = self.reduce.parameters();
        p.extend(self.conv1.parameters());
        p.extend(self.conv2.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = self.reduce.parameters_mut();
        p.extend(self.conv1.parameters_mut());
        p.extend(self.conv2.parameters_mut());
        p
    }

    fn named_parameters(&self, prefix: &str) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.reduce.named_parameters() {
            params.push((format!("{prefix}.reduce.{name}"), p));
        }
        for (name, p) in self.conv1.named_parameters() {
            params.push((format!("{prefix}.conv1.{name}"), p));
        }
        for (name, p) in self.conv2.named_parameters() {
            params.push((format!("{prefix}.conv2.{name}"), p));
        }
        params
    }
}

// ===========================================================================
// UNet
// ===========================================================================

/// U-Net segmentation model.
///
/// The model takes `[B, 3, H, W]` input and produces `[B, num_classes, H, W]`
/// dense per-pixel logits. `H` and `W` must be divisible by 16 (four 2x
/// downsampling stages).
///
/// ```text
///  Input [B,3,H,W]
///    |
///  enc1 [B,64,H,W]  -------- skip1 --------+
///    | maxpool                                |
///  enc2 [B,128,H/2,W/2] ---- skip2 ------+  |
///    | maxpool                              |  |
///  enc3 [B,256,H/4,W/4] ---- skip3 ----+  |  |
///    | maxpool                            |  |  |
///  enc4 [B,512,H/8,W/8] ---- skip4 --+  |  |  |
///    | maxpool                          |  |  |  |
///  bottleneck [B,1024,H/16,W/16]       |  |  |  |
///    |                                  |  |  |  |
///  dec4 [B,512,H/8,W/8] <--- skip4 ---+  |  |  |
///  dec3 [B,256,H/4,W/4] <--- skip3 ------+  |  |
///  dec2 [B,128,H/2,W/2] <--- skip2 ---------+  |
///  dec1 [B,64,H,W]      <--- skip1 ------------+
///    |
///  head [B,num_classes,H,W]
/// ```
pub struct UNet<T: Float> {
    // Encoder.
    enc1: EncoderBlock<T>,
    enc2: EncoderBlock<T>,
    enc3: EncoderBlock<T>,
    enc4: EncoderBlock<T>,
    pool: MaxPool2d,

    // Bottleneck.
    bottleneck_conv1: Conv2d<T>,
    bottleneck_conv2: Conv2d<T>,

    // Decoder.
    dec4: DecoderBlock<T>,
    dec3: DecoderBlock<T>,
    dec2: DecoderBlock<T>,
    dec1: DecoderBlock<T>,

    // Classification head.
    head: Conv2d<T>,

    training: bool,
}

impl<T: Float> UNet<T> {
    /// Construct a U-Net for `num_classes` output classes.
    ///
    /// Input is expected to be `[B, 3, H, W]` with `H` and `W` divisible
    /// by 16.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        // Encoder: 3->64->128->256->512.
        let enc1 = EncoderBlock::new(3, 64)?;
        let enc2 = EncoderBlock::new(64, 128)?;
        let enc3 = EncoderBlock::new(128, 256)?;
        let enc4 = EncoderBlock::new(256, 512)?;
        let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);

        // Bottleneck: 512->1024.
        let bottleneck_conv1 = Conv2d::new(512, 1024, (3, 3), (1, 1), (1, 1), false)?;
        let bottleneck_conv2 = Conv2d::new(1024, 1024, (3, 3), (1, 1), (1, 1), false)?;

        // Decoder: 1024->512->256->128->64.
        let dec4 = DecoderBlock::new(1024, 512)?;
        let dec3 = DecoderBlock::new(512, 256)?;
        let dec2 = DecoderBlock::new(256, 128)?;
        let dec1 = DecoderBlock::new(128, 64)?;

        // Head: 64 -> num_classes with 1x1 conv.
        let head = Conv2d::new(64, num_classes, (1, 1), (1, 1), (0, 0), false)?;

        Ok(Self {
            enc1,
            enc2,
            enc3,
            enc4,
            pool,
            bottleneck_conv1,
            bottleneck_conv2,
            dec4,
            dec3,
            dec2,
            dec1,
            head,
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

impl<T: Float> Module<T> for UNet<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Encoder.
        let s1 = self.enc1.forward(input)?; // [B, 64, H, W]
        let x = Module::<T>::forward(&self.pool, &s1)?; // [B, 64, H/2, W/2]

        let s2 = self.enc2.forward(&x)?; // [B, 128, H/2, W/2]
        let x = Module::<T>::forward(&self.pool, &s2)?; // [B, 128, H/4, W/4]

        let s3 = self.enc3.forward(&x)?; // [B, 256, H/4, W/4]
        let x = Module::<T>::forward(&self.pool, &s3)?; // [B, 256, H/8, W/8]

        let s4 = self.enc4.forward(&x)?; // [B, 512, H/8, W/8]
        let x = Module::<T>::forward(&self.pool, &s4)?; // [B, 512, H/16, W/16]

        // Bottleneck.
        let x = self.bottleneck_conv1.forward(&x)?; // [B, 1024, H/16, W/16]
        let x = relu(&x)?;
        let x = self.bottleneck_conv2.forward(&x)?; // [B, 1024, H/16, W/16]
        let x = relu(&x)?;

        // Decoder.
        let x = self.dec4.forward(&x, &s4)?; // [B, 512, H/8, W/8]
        let x = self.dec3.forward(&x, &s3)?; // [B, 256, H/4, W/4]
        let x = self.dec2.forward(&x, &s2)?; // [B, 128, H/2, W/2]
        let x = self.dec1.forward(&x, &s1)?; // [B, 64, H, W]

        // Head: 1x1 conv to produce per-pixel class logits.
        self.head.forward(&x) // [B, num_classes, H, W]
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.enc1.parameters());
        params.extend(self.enc2.parameters());
        params.extend(self.enc3.parameters());
        params.extend(self.enc4.parameters());
        params.extend(self.bottleneck_conv1.parameters());
        params.extend(self.bottleneck_conv2.parameters());
        params.extend(self.dec4.parameters());
        params.extend(self.dec3.parameters());
        params.extend(self.dec2.parameters());
        params.extend(self.dec1.parameters());
        params.extend(self.head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.enc1.parameters_mut());
        params.extend(self.enc2.parameters_mut());
        params.extend(self.enc3.parameters_mut());
        params.extend(self.enc4.parameters_mut());
        params.extend(self.bottleneck_conv1.parameters_mut());
        params.extend(self.bottleneck_conv2.parameters_mut());
        params.extend(self.dec4.parameters_mut());
        params.extend(self.dec3.parameters_mut());
        params.extend(self.dec2.parameters_mut());
        params.extend(self.dec1.parameters_mut());
        params.extend(self.head.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        params.extend(self.enc1.named_parameters("enc1"));
        params.extend(self.enc2.named_parameters("enc2"));
        params.extend(self.enc3.named_parameters("enc3"));
        params.extend(self.enc4.named_parameters("enc4"));
        for (name, p) in self.bottleneck_conv1.named_parameters() {
            params.push((format!("bottleneck.conv1.{name}"), p));
        }
        for (name, p) in self.bottleneck_conv2.named_parameters() {
            params.push((format!("bottleneck.conv2.{name}"), p));
        }
        params.extend(self.dec4.named_parameters("dec4"));
        params.extend(self.dec3.named_parameters("dec3"));
        params.extend(self.dec2.named_parameters("dec2"));
        params.extend(self.dec1.named_parameters("dec1"));
        for (name, p) in self.head.named_parameters() {
            params.push((format!("head.{name}"), p));
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

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for UNet<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();

        let s1 = self.enc1.forward(input)?;
        out.insert("enc1".to_string(), s1.clone());
        let x = Module::<T>::forward(&self.pool, &s1)?;

        let s2 = self.enc2.forward(&x)?;
        out.insert("enc2".to_string(), s2.clone());
        let x = Module::<T>::forward(&self.pool, &s2)?;

        let s3 = self.enc3.forward(&x)?;
        out.insert("enc3".to_string(), s3.clone());
        let x = Module::<T>::forward(&self.pool, &s3)?;

        let s4 = self.enc4.forward(&x)?;
        out.insert("enc4".to_string(), s4.clone());
        let x = Module::<T>::forward(&self.pool, &s4)?;

        let x = self.bottleneck_conv1.forward(&x)?;
        let x = relu(&x)?;
        let x = self.bottleneck_conv2.forward(&x)?;
        let x = relu(&x)?;
        out.insert("bottleneck".to_string(), x.clone());

        let x = self.dec4.forward(&x, &s4)?;
        out.insert("dec4".to_string(), x.clone());
        let x = self.dec3.forward(&x, &s3)?;
        out.insert("dec3".to_string(), x.clone());
        let x = self.dec2.forward(&x, &s2)?;
        out.insert("dec2".to_string(), x.clone());
        let x = self.dec1.forward(&x, &s1)?;
        out.insert("dec1".to_string(), x.clone());

        let logits = self.head.forward(&x)?;
        out.insert("head".to_string(), logits);
        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        vec![
            "enc1".to_string(),
            "enc2".to_string(),
            "enc3".to_string(),
            "enc4".to_string(),
            "bottleneck".to_string(),
            "dec4".to_string(),
            "dec3".to_string(),
            "dec2".to_string(),
            "dec1".to_string(),
            "head".to_string(),
        ]
    }
}

/// Construct a U-Net model for semantic segmentation.
///
/// * `num_classes` -- number of output classes (channel dimension of output).
///
/// Input: `[B, 3, H, W]` where `H` and `W` are divisible by 16.
/// Output: `[B, num_classes, H, W]`.
pub fn unet<T: Float>(num_classes: usize) -> FerrotorchResult<UNet<T>> {
    UNet::new(num_classes)
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
    // Upsample tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_upsample_nearest_2x_shape() {
        let input = leaf_4d(&vec![0.1; 2 * 4 * 3 * 3], [2, 4, 3, 3], false);
        let output = no_grad(|| upsample_nearest_2x(&input).unwrap());
        assert_eq!(output.shape(), &[2, 4, 6, 6]);
    }

    #[test]
    fn test_upsample_nearest_2x_values() {
        // 1x1x2x2 -> 1x1x4x4
        let input = leaf_4d(&[1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2], false);
        let output = no_grad(|| upsample_nearest_2x(&input).unwrap());
        let data = output.data().unwrap();
        #[rustfmt::skip]
        let expected = vec![
            1.0, 1.0, 2.0, 2.0,
            1.0, 1.0, 2.0, 2.0,
            3.0, 3.0, 4.0, 4.0,
            3.0, 3.0, 4.0, 4.0,
        ];
        assert_eq!(data, &expected);
    }

    // -----------------------------------------------------------------------
    // Encoder block tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_encoder_block_shape() {
        let block = EncoderBlock::<f32>::new(3, 64).unwrap();
        let input = leaf_4d(&vec![0.01; 3 * 16 * 16], [1, 3, 16, 16], false);
        let output = no_grad(|| block.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 64, 16, 16]);
    }

    // -----------------------------------------------------------------------
    // Decoder block tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_decoder_block_shape() {
        let block = DecoderBlock::<f32>::new(128, 64).unwrap();
        let input = leaf_4d(&vec![0.01; 128 * 4 * 4], [1, 128, 4, 4], false);
        let skip = leaf_4d(&vec![0.01; 64 * 8 * 8], [1, 64, 8, 8], false);
        let output = no_grad(|| block.forward(&input, &skip).unwrap());
        assert_eq!(output.shape(), &[1, 64, 8, 8]);
    }

    // -----------------------------------------------------------------------
    // UNet forward shape tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_unet_forward_shape() {
        let model = unet::<f32>(21).unwrap();
        let input = leaf_4d(&vec![0.01; 3 * 256 * 256], [1, 3, 256, 256], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 21, 256, 256]);
    }

    #[test]
    fn test_unet_forward_small() {
        // Smallest valid spatial size: 16x16 (4 halvings -> 1x1).
        let model = unet::<f32>(2).unwrap();
        let input = leaf_4d(&vec![0.01; 3 * 16 * 16], [1, 3, 16, 16], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 2, 16, 16]);
    }

    #[test]
    fn test_unet_batch_size() {
        let model = unet::<f32>(5).unwrap();
        let input = leaf_4d(&vec![0.01; 2 * 3 * 32 * 32], [2, 3, 32, 32], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 5, 32, 32]);
    }

    // -----------------------------------------------------------------------
    // Parameter count
    // -----------------------------------------------------------------------

    #[test]
    fn test_unet_parameter_count() {
        let model = unet::<f32>(21).unwrap();
        let total = model.num_parameters();

        // Manual count (no bias, all convs have bias=false):
        //
        // Encoder:
        //   enc1: conv(3,64,3,3) + conv(64,64,3,3)     = 1728 + 36864    = 38592
        //   enc2: conv(64,128,3,3) + conv(128,128,3,3)  = 73728 + 147456  = 221184
        //   enc3: conv(128,256,3,3) + conv(256,256,3,3) = 294912 + 589824 = 884736
        //   enc4: conv(256,512,3,3) + conv(512,512,3,3) = 1179648+2359296 = 3538944
        //
        // Bottleneck:
        //   conv(512,1024,3,3) + conv(1024,1024,3,3)    = 4718592+9437184 = 14155776
        //
        // Decoder:
        //   dec4: reduce(1024,512,1,1) + conv(1024,512,3,3) + conv(512,512,3,3)
        //         = 524288 + 4718592 + 2359296 = 7602176
        //   dec3: reduce(512,256,1,1) + conv(512,256,3,3) + conv(256,256,3,3)
        //         = 131072 + 1179648 + 589824 = 1900544
        //   dec2: reduce(256,128,1,1) + conv(256,128,3,3) + conv(128,128,3,3)
        //         = 32768 + 294912 + 147456 = 475136
        //   dec1: reduce(128,64,1,1) + conv(128,64,3,3) + conv(64,64,3,3)
        //         = 8192 + 73728 + 36864 = 118784
        //
        // Head:
        //   conv(64,21,1,1) = 1344
        //
        // Total = 38592 + 221184 + 884736 + 3538944 + 14155776
        //       + 7602176 + 1900544 + 475136 + 118784 + 1344
        //       = 28937216
        let expected = 28_937_216;
        assert_eq!(
            total, expected,
            "U-Net parameter count mismatch: expected {expected}, got {total}"
        );
    }

    // -----------------------------------------------------------------------
    // Named parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_unet_named_parameters_prefixes() {
        let model = unet::<f32>(21).unwrap();
        let named = model.named_parameters();
        assert!(!named.is_empty());

        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("enc1.")));
        assert!(names.iter().any(|n| n.starts_with("enc2.")));
        assert!(names.iter().any(|n| n.starts_with("enc3.")));
        assert!(names.iter().any(|n| n.starts_with("enc4.")));
        assert!(names.iter().any(|n| n.starts_with("bottleneck.")));
        assert!(names.iter().any(|n| n.starts_with("dec4.")));
        assert!(names.iter().any(|n| n.starts_with("dec3.")));
        assert!(names.iter().any(|n| n.starts_with("dec2.")));
        assert!(names.iter().any(|n| n.starts_with("dec1.")));
        assert!(names.iter().any(|n| n.starts_with("head.")));
    }

    // -----------------------------------------------------------------------
    // Train / eval
    // -----------------------------------------------------------------------

    #[test]
    fn test_unet_train_eval() {
        let mut model = unet::<f32>(21).unwrap();
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
    fn test_unet_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<UNet<f32>>();
    }

    // -----------------------------------------------------------------------
    // Gradient flow through skip connections
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_flow_through_upsample() {
        let input = leaf_4d(&vec![0.5; 4 * 4 * 4], [1, 4, 4, 4], true);
        let output = upsample_nearest_2x(&input).unwrap();
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap();
        assert!(grad.is_some(), "input should have gradients");
        let grad_data = grad.unwrap().data().unwrap().to_vec();
        // Each input element is replicated to a 2x2 block, so the gradient
        // of sum is 4.0 for each element.
        for &g in &grad_data {
            assert!((g - 4.0).abs() < 1e-6, "expected gradient 4.0, got {g}");
        }
    }
}
