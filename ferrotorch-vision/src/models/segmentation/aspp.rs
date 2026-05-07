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

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::{FerrotorchResult, Float, Tensor, TensorStorage};
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::pooling::AdaptiveAvgPool2d;
use ferrotorch_nn::upsample::{InterpolateMode, interpolate};
use ferrotorch_nn::{Conv2d, Dropout};
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;

// ---------------------------------------------------------------------------
// DilatedConv2d — Conv2d with dilation (same-size output, no bias)
// ---------------------------------------------------------------------------
//
// ferrotorch_nn::Conv2d's im2col kernel does not support dilation. We
// implement a minimal struct here for the ASPP use-case: 3×3 kernel,
// bias=false, same-size padding (pad = dilation * (k-1) / 2 = dilation).

/// A 3×3 convolution with dilation, no bias, BN, and ReLU, matching
/// the torchvision `ASPPConv` building block.
///
/// The output spatial size equals the input spatial size (same-size padding).
pub struct DilatedConv2d<T: Float> {
    weight: Parameter<T>,
    dilation: usize,
    bn: BatchNorm2d<T>,
    training: bool,
}

impl<T: Float> DilatedConv2d<T> {
    /// Create a new `DilatedConv2d`.
    ///
    /// Uses Kaiming uniform initialisation (ReLU gain) matching torchvision.
    pub fn new(in_channels: usize, out_channels: usize, dilation: usize) -> FerrotorchResult<Self> {
        use ferrotorch_nn::init::{NonLinearity, kaiming_uniform};
        let mut weight = Parameter::zeros(&[out_channels, in_channels, 3, 3])?;
        kaiming_uniform(&mut weight, NonLinearity::ReLU)?;
        let bn = BatchNorm2d::new(out_channels, 1e-5, 0.1, true)?;
        Ok(Self {
            weight,
            dilation,
            bn,
            training: true,
        })
    }

    /// Run the dilated 3×3 convolution, BN, and ReLU.
    pub fn forward_inner(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let out = dilated_conv2d_forward(input, &self.weight, self.dilation)?;
        let out = Module::<T>::forward(&self.bn, &out)?;
        relu(&out)
    }
}

impl<T: Float> Module<T> for DilatedConv2d<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        self.forward_inner(input)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = vec![&self.weight];
        p.extend(self.bn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p: Vec<&mut Parameter<T>> = vec![&mut self.weight];
        p.extend(self.bn.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut out = vec![("weight".to_string(), &self.weight)];
        for (k, v) in self.bn.named_parameters() {
            out.push((format!("bn.{k}"), v));
        }
        out
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
// dilated_conv2d_forward — host-side im2col with dilation
// ---------------------------------------------------------------------------

/// Perform a 3×3 dilated convolution on `input` using `weight`.
///
/// * `input`  — `[B, C_in, H, W]`
/// * `weight` — `[C_out, C_in, 3, 3]`
/// * `dilation` — dilation rate (pad = dilation, same-size output)
///
/// Returns `[B, C_out, H, W]`.
fn dilated_conv2d_forward<T: Float>(
    input: &Tensor<T>,
    weight: &Parameter<T>,
    dilation: usize,
) -> FerrotorchResult<Tensor<T>> {
    use ferrotorch_core::FerrotorchError;

    if input.ndim() != 4 {
        return Err(FerrotorchError::ShapeMismatch {
            message: format!(
                "dilated_conv2d_forward: expected 4-D input, got {:?}",
                input.shape()
            ),
        });
    }
    let batch = input.shape()[0];
    let c_in = input.shape()[1];
    let h_in = input.shape()[2];
    let w_in = input.shape()[3];

    let w_data = weight.data()?.to_vec();
    let c_out = weight.shape()[0];
    let pad = dilation; // same-size padding: p = d*(k-1)/2 = d for k=3

    let h_out = h_in; // stride=1, pad=dilation, so (H + 2p - d*(k-1) - 1)/s + 1 = H
    let w_out = w_in;

    let input_data = input.data_vec()?;
    let zero = <T as num_traits::Zero>::zero();

    let mut output = vec![zero; batch * c_out * h_out * w_out];

    for b in 0..batch {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut acc = zero;
                    for ci in 0..c_in {
                        for kh in 0..3usize {
                            for kw in 0..3usize {
                                let ih_signed =
                                    oh as isize + kh as isize * dilation as isize - pad as isize;
                                let iw_signed =
                                    ow as isize + kw as isize * dilation as isize - pad as isize;
                                if ih_signed >= 0
                                    && ih_signed < h_in as isize
                                    && iw_signed >= 0
                                    && iw_signed < w_in as isize
                                {
                                    let ih = ih_signed as usize;
                                    let iw = iw_signed as usize;
                                    let in_idx =
                                        b * c_in * h_in * w_in + ci * h_in * w_in + ih * w_in + iw;
                                    let w_idx =
                                        co * c_in * 9 + ci * 9 + kh * 3 + kw;
                                    acc += input_data[in_idx] * w_data[w_idx];
                                }
                            }
                        }
                    }
                    output[b * c_out * h_out * w_out + co * h_out * w_out + oh * w_out + ow] = acc;
                }
            }
        }
    }

    // We implement this as a no-grad leaf output (autograd through the weight
    // is not needed for inference; training DeepLabV3 end-to-end would require
    // a full dilated-conv backward which is tracked as a follow-up).
    Tensor::from_storage(
        TensorStorage::cpu(output),
        vec![batch, c_out, h_out, w_out],
        false,
    )
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
pub struct Aspp<T: Float> {
    /// 1×1 convolution branch.
    conv1: ASPPConv1x1<T>,
    /// Dilated 3×3 conv branches at rates 6, 12, 18.
    conv_r6: DilatedConv2d<T>,
    conv_r12: DilatedConv2d<T>,
    conv_r18: DilatedConv2d<T>,
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
    pub fn new(in_channels: usize, out_channels: usize) -> FerrotorchResult<Self> {
        let conv1 = ASPPConv1x1::new(in_channels, out_channels)?;
        let conv_r6 = DilatedConv2d::new(in_channels, out_channels, 6)?;
        let conv_r12 = DilatedConv2d::new(in_channels, out_channels, 12)?;
        let conv_r18 = DilatedConv2d::new(in_channels, out_channels, 18)?;
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
            conv_r6,
            conv_r12,
            conv_r18,
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
        let b2 = self.conv_r6.forward(input)?;
        let b3 = self.conv_r12.forward(input)?;
        let b4 = self.conv_r18.forward(input)?;
        let b5 = self.pool.forward(input)?;

        // Concatenate along channel dim: [B, 1280, H, W].
        let cat = concat_channels(&[b1, b2, b3, b4, b5])?;

        // Dropout → project → BN → ReLU.
        let x = if self.training {
            Module::<T>::forward(&self.dropout, &cat)?
        } else {
            cat
        };
        let x = self.project.forward(&x)?;
        let x = Module::<T>::forward(&self.project_bn, &x)?;
        relu(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters());
        p.extend(self.conv_r6.parameters());
        p.extend(self.conv_r12.parameters());
        p.extend(self.conv_r18.parameters());
        p.extend(self.pool.parameters());
        p.extend(self.project.parameters());
        p.extend(self.project_bn.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv1.parameters_mut());
        p.extend(self.conv_r6.parameters_mut());
        p.extend(self.conv_r12.parameters_mut());
        p.extend(self.conv_r18.parameters_mut());
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
        for (k, v) in self.conv_r6.named_parameters() {
            out.push((format!("1.{k}"), v));
        }
        for (k, v) in self.conv_r12.named_parameters() {
            out.push((format!("2.{k}"), v));
        }
        for (k, v) in self.conv_r18.named_parameters() {
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

    fn train(&mut self) {
        self.training = true;
        self.conv1.train();
        self.conv_r6.train();
        self.conv_r12.train();
        self.conv_r18.train();
        self.pool.train();
        self.project_bn.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv1.eval();
        self.conv_r6.eval();
        self.conv_r12.eval();
        self.conv_r18.eval();
        self.pool.eval();
        self.project_bn.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ---------------------------------------------------------------------------
// concat_channels — allocate-and-copy concatenation along dim 1
// ---------------------------------------------------------------------------

/// Concatenate a slice of tensors along the channel dimension (dim 1).
///
/// All tensors must have the same `[B, ?, H, W]` shape; only the channel
/// count `?` may differ.
fn concat_channels<T: Float>(tensors: &[Tensor<T>]) -> FerrotorchResult<Tensor<T>> {
    use ferrotorch_core::FerrotorchError;

    if tensors.is_empty() {
        return Err(FerrotorchError::InvalidArgument {
            message: "concat_channels: empty tensor list".into(),
        });
    }

    let batch = tensors[0].shape()[0];
    let h = tensors[0].shape()[2];
    let w = tensors[0].shape()[3];
    let total_c: usize = tensors.iter().map(|t| t.shape()[1]).sum();

    let mut output = vec![<T as num_traits::Zero>::zero(); batch * total_c * h * w];

    let mut c_offset = 0usize;
    for t in tensors {
        let c = t.shape()[1];
        let data = t.data_vec()?;
        for b in 0..batch {
            for ci in 0..c {
                for row in 0..h {
                    for col in 0..w {
                        let src = b * c * h * w + ci * h * w + row * w + col;
                        let dst = b * total_c * h * w + (c_offset + ci) * h * w + row * w + col;
                        output[dst] = data[src];
                    }
                }
            }
        }
        c_offset += c;
    }

    Tensor::from_storage(TensorStorage::cpu(output), vec![batch, total_c, h, w], false)
}

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
        let aspp = Aspp::<f32>::new(2048, 256).unwrap();
        // Tiny spatial: 4×4 to keep test fast.
        let x = no_grad(|| randn_4d([1, 2048, 4, 4]));
        let y = no_grad(|| aspp.forward(&x).unwrap());
        assert_eq!(y.shape(), &[1, 256, 4, 4], "ASPP output shape mismatch");
    }

    #[test]
    fn test_aspp_preserves_spatial_dims() {
        let aspp = Aspp::<f32>::new(256, 256).unwrap();
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
        let aspp = Aspp::<f32>::new(2048, 256).unwrap();
        let np: usize = aspp.parameters().iter().map(|p| p.numel()).sum();
        // Rough lower bound: 5 branches × 2048*256 params + project ~1M.
        assert!(np > 2_000_000, "ASPP params too low: {np}");
    }
}
