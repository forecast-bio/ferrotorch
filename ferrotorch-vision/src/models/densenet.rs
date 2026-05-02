//! DenseNet-121 architecture (simplified).
//!
//! Follows Huang et al. 2017 "Densely Connected Convolutional Networks".
//! Every layer inside a dense block concatenates its input with its own
//! newly-produced feature maps so each layer sees the outputs of all
//! preceding layers in the block, producing feature reuse and better
//! gradient flow.
//!
//! **Simplifications.** The real DenseNet uses a `BN → ReLU → Conv`
//! bottleneck (BN + 1×1) followed by a 3×3 Conv. We omit batch norm
//! (not used by the existing vision models in this crate) and keep the
//! BN-free Conv→ReLU→Conv bottleneck. Transition layers are a 1×1 Conv
//! plus a 2×2 `AvgPool2d`. Channel counts, growth rate (32), and block
//! counts (6/12/24/16 → DenseNet-121) match the paper.
//!
//! CL-436.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::shape::{cat, reshape};
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::{AdaptiveAvgPool2d, AvgPool2d};
use ferrotorch_nn::{Conv2d, Linear};

// ===========================================================================
// Helpers
// ===========================================================================

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
// DenseLayer — a single Conv→ReLU→Conv bottleneck emitting `growth_rate` new feature maps
// ===========================================================================

/// One layer inside a dense block. Computes `growth_rate` new feature
/// maps from the concatenated input and concatenates them back onto
/// the input so the next layer sees both.
pub struct DenseLayer<T: Float> {
    bn_conv: Conv2d<T>,
    body_conv: Conv2d<T>,
    training: bool,
}

impl<T: Float> DenseLayer<T> {
    /// Build a dense layer.
    ///
    /// * `in_ch` — total input channels (grows with each layer).
    /// * `growth_rate` — number of new feature maps produced.
    /// * `bn_size` — bottleneck multiplier (typically 4).
    pub fn new(in_ch: usize, growth_rate: usize, bn_size: usize) -> FerrotorchResult<Self> {
        let inter = bn_size * growth_rate;
        Ok(Self {
            bn_conv: conv(in_ch, inter, 1, 1, 0)?,
            body_conv: conv(inter, growth_rate, 3, 1, 1)?,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for DenseLayer<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.bn_conv.forward(input)?;
        let x = relu(&x)?;
        let x = self.body_conv.forward(&x)?;
        let x = relu(&x)?;
        // Concatenate the new feature maps with the input along the
        // channel axis (axis 1 for NCHW).
        cat(&[input.clone(), x], 1)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.bn_conv.parameters());
        p.extend(self.body_conv.parameters());
        p
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.bn_conv.parameters_mut());
        p.extend(self.body_conv.parameters_mut());
        p
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.bn_conv.named_parameters() {
            p.push((format!("bn_conv.{n}"), param));
        }
        for (n, param) in self.body_conv.named_parameters() {
            p.push((format!("body_conv.{n}"), param));
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
// DenseBlock
// ===========================================================================

/// A stack of [`DenseLayer`]s forming one dense block.
pub struct DenseBlock<T: Float> {
    layers: Vec<DenseLayer<T>>,
    training: bool,
}

impl<T: Float> DenseBlock<T> {
    pub fn new(
        num_layers: usize,
        in_ch: usize,
        growth_rate: usize,
        bn_size: usize,
    ) -> FerrotorchResult<Self> {
        let mut layers: Vec<DenseLayer<T>> = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            let layer_in = in_ch + i * growth_rate;
            layers.push(DenseLayer::new(layer_in, growth_rate, bn_size)?);
        }
        Ok(Self {
            layers,
            training: true,
        })
    }

    /// Total output channels after this dense block runs.
    pub fn output_channels(&self, in_ch: usize, growth_rate: usize) -> usize {
        in_ch + self.layers.len() * growth_rate
    }
}

impl<T: Float> Module<T> for DenseBlock<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let mut x = input.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        for l in &self.layers {
            p.extend(l.parameters());
        }
        p
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        for l in &mut self.layers {
            p.extend(l.parameters_mut());
        }
        p
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (i, l) in self.layers.iter().enumerate() {
            for (n, param) in l.named_parameters() {
                p.push((format!("layer{i}.{n}"), param));
            }
        }
        p
    }
    fn train(&mut self) {
        self.training = true;
        for l in self.layers.iter_mut() {
            l.train();
        }
    }
    fn eval(&mut self) {
        self.training = false;
        for l in self.layers.iter_mut() {
            l.eval();
        }
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// TransitionLayer — reduces channels and spatial between dense blocks
// ===========================================================================

/// A transition layer: 1×1 Conv + 2×2 average pool. Halves channels
/// and spatial dimensions.
pub struct TransitionLayer<T: Float> {
    conv: Conv2d<T>,
    pool: AvgPool2d,
    training: bool,
}

impl<T: Float> TransitionLayer<T> {
    pub fn new(in_ch: usize, out_ch: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            conv: conv(in_ch, out_ch, 1, 1, 0)?,
            pool: AvgPool2d::new([2, 2], [2, 2], [0, 0]),
            training: true,
        })
    }
}

impl<T: Float> Module<T> for TransitionLayer<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.conv.forward(input)?;
        let x = relu(&x)?;
        Module::<T>::forward(&self.pool, &x)
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
// DenseNet
// ===========================================================================

/// DenseNet-121 for ImageNet-style classification.
///
/// Block config (num_layers): 6 / 12 / 24 / 16 with growth rate 32.
/// Input: `[B, 3, H, W]`. Output: `[B, num_classes]`.
pub struct DenseNet<T: Float> {
    stem: Conv2d<T>,
    block1: DenseBlock<T>,
    trans1: TransitionLayer<T>,
    block2: DenseBlock<T>,
    trans2: TransitionLayer<T>,
    block3: DenseBlock<T>,
    trans3: TransitionLayer<T>,
    block4: DenseBlock<T>,
    avgpool: AdaptiveAvgPool2d,
    classifier: Linear<T>,
    training: bool,
}

impl<T: Float> DenseNet<T> {
    /// Construct DenseNet-121 with the given number of classes.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        let growth_rate = 32usize;
        let bn_size = 4usize;
        let num_init_features = 64usize;
        let block_config = [6, 12, 24, 16];

        let stem = conv(3, num_init_features, 7, 2, 3)?;

        let mut in_ch = num_init_features;

        let block1 = DenseBlock::new(block_config[0], in_ch, growth_rate, bn_size)?;
        in_ch = block1.output_channels(in_ch, growth_rate);
        let trans1 = TransitionLayer::new(in_ch, in_ch / 2)?;
        in_ch /= 2;

        let block2 = DenseBlock::new(block_config[1], in_ch, growth_rate, bn_size)?;
        in_ch = block2.output_channels(in_ch, growth_rate);
        let trans2 = TransitionLayer::new(in_ch, in_ch / 2)?;
        in_ch /= 2;

        let block3 = DenseBlock::new(block_config[2], in_ch, growth_rate, bn_size)?;
        in_ch = block3.output_channels(in_ch, growth_rate);
        let trans3 = TransitionLayer::new(in_ch, in_ch / 2)?;
        in_ch /= 2;

        let block4 = DenseBlock::new(block_config[3], in_ch, growth_rate, bn_size)?;
        in_ch = block4.output_channels(in_ch, growth_rate);

        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let classifier = Linear::new(in_ch, num_classes, true)?;

        Ok(Self {
            stem,
            block1,
            trans1,
            block2,
            trans2,
            block3,
            trans3,
            block4,
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

impl<T: Float> Module<T> for DenseNet<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.stem.forward(input)?;
        let x = relu(&x)?;
        let x = self.block1.forward(&x)?;
        let x = self.trans1.forward(&x)?;
        let x = self.block2.forward(&x)?;
        let x = self.trans2.forward(&x)?;
        let x = self.block3.forward(&x)?;
        let x = self.trans3.forward(&x)?;
        let x = self.block4.forward(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters());
        p.extend(self.block1.parameters());
        p.extend(self.trans1.parameters());
        p.extend(self.block2.parameters());
        p.extend(self.trans2.parameters());
        p.extend(self.block3.parameters());
        p.extend(self.trans3.parameters());
        p.extend(self.block4.parameters());
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem.parameters_mut());
        p.extend(self.block1.parameters_mut());
        p.extend(self.trans1.parameters_mut());
        p.extend(self.block2.parameters_mut());
        p.extend(self.trans2.parameters_mut());
        p.extend(self.block3.parameters_mut());
        p.extend(self.trans3.parameters_mut());
        p.extend(self.block4.parameters_mut());
        p.extend(self.classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.stem.named_parameters() {
            p.push((format!("stem.{n}"), param));
        }
        for (n, param) in self.block1.named_parameters() {
            p.push((format!("block1.{n}"), param));
        }
        for (n, param) in self.trans1.named_parameters() {
            p.push((format!("trans1.{n}"), param));
        }
        for (n, param) in self.block2.named_parameters() {
            p.push((format!("block2.{n}"), param));
        }
        for (n, param) in self.trans2.named_parameters() {
            p.push((format!("trans2.{n}"), param));
        }
        for (n, param) in self.block3.named_parameters() {
            p.push((format!("block3.{n}"), param));
        }
        for (n, param) in self.trans3.named_parameters() {
            p.push((format!("trans3.{n}"), param));
        }
        for (n, param) in self.block4.named_parameters() {
            p.push((format!("block4.{n}"), param));
        }
        for (n, param) in self.classifier.named_parameters() {
            p.push((format!("classifier.{n}"), param));
        }
        p
    }

    fn train(&mut self) {
        self.training = true;
        self.block1.train();
        self.block2.train();
        self.block3.train();
        self.block4.train();
        self.trans1.train();
        self.trans2.train();
        self.trans3.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.block1.eval();
        self.block2.eval();
        self.block3.eval();
        self.block4.eval();
        self.trans1.eval();
        self.trans2.eval();
        self.trans3.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

/// Convenience constructor for DenseNet-121.
pub fn densenet121<T: Float>(num_classes: usize) -> FerrotorchResult<DenseNet<T>> {
    DenseNet::new(num_classes)
}

// ---------------------------------------------------------------------------
// IntermediateFeatures — CL-499
// ---------------------------------------------------------------------------

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for DenseNet<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();

        let x = self.stem.forward(input)?;
        let x = relu(&x)?;
        out.insert("stem".to_string(), x.clone());

        let x = self.block1.forward(&x)?;
        out.insert("block1".to_string(), x.clone());
        let x = self.trans1.forward(&x)?;
        out.insert("trans1".to_string(), x.clone());

        let x = self.block2.forward(&x)?;
        out.insert("block2".to_string(), x.clone());
        let x = self.trans2.forward(&x)?;
        out.insert("trans2".to_string(), x.clone());

        let x = self.block3.forward(&x)?;
        out.insert("block3".to_string(), x.clone());
        let x = self.trans3.forward(&x)?;
        out.insert("trans3".to_string(), x.clone());

        let x = self.block4.forward(&x)?;
        out.insert("block4".to_string(), x.clone());

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
        vec![
            "stem".to_string(),
            "block1".to_string(),
            "trans1".to_string(),
            "block2".to_string(),
            "trans2".to_string(),
            "block3".to_string(),
            "trans3".to_string(),
            "block4".to_string(),
            "avgpool".to_string(),
            "classifier".to_string(),
        ]
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
    fn test_dense_layer_concatenates_output() {
        // input [1, 4, 8, 8] + growth_rate 2 → output [1, 6, 8, 8]
        let layer: DenseLayer<f32> = DenseLayer::new(4, 2, 4).unwrap();
        let x = dummy_image(1, 4, 8, 8);
        let y = layer.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 6, 8, 8]);
    }

    #[test]
    fn test_dense_block_output_channels_calculation() {
        // 3 layers, in_ch=8, growth_rate=4 → 8 + 3*4 = 20
        let block: DenseBlock<f32> = DenseBlock::new(3, 8, 4, 4).unwrap();
        assert_eq!(block.output_channels(8, 4), 20);

        let x = dummy_image(1, 8, 8, 8);
        let y = block.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 20, 8, 8]);
    }

    #[test]
    fn test_transition_layer_halves_spatial() {
        let trans: TransitionLayer<f32> = TransitionLayer::new(8, 4).unwrap();
        let x = dummy_image(1, 8, 16, 16);
        let y = trans.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_densenet121_output_shape() {
        // Use a smaller input than ImageNet 224 to keep the test fast.
        // The stem is stride-2 7x7 and each transition is stride-2, so we
        // need H >= 32 to avoid spatial dims collapsing to 0.
        let model: DenseNet<f32> = densenet121(10).unwrap();
        let x = dummy_image(1, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 10]);
    }

    #[test]
    fn test_densenet121_custom_classes() {
        let model: DenseNet<f32> = densenet121(7).unwrap();
        let x = dummy_image(1, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 7]);
    }

    #[test]
    fn test_densenet121_param_count() {
        let model: DenseNet<f32> = densenet121(1000).unwrap();
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_densenet121_named_parameters_prefixes() {
        let model: DenseNet<f32> = densenet121(10).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("stem.")));
        assert!(names.iter().any(|n| n.starts_with("block1.")));
        assert!(names.iter().any(|n| n.starts_with("trans1.")));
        assert!(names.iter().any(|n| n.starts_with("block4.")));
        assert!(names.iter().any(|n| n.starts_with("classifier.")));
    }

    #[test]
    fn test_densenet121_train_eval() {
        let mut model: DenseNet<f32> = densenet121(10).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }
}
