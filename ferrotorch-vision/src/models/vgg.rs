//! VGG architectures: VGG-11 and VGG-16.
//!
//! Follows the original paper: "Very Deep Convolutional Networks for
//! Large-Scale Image Recognition" (Simonyan & Zisserman, 2014).
//!
//! Batch normalization is omitted (not yet available in `ferrotorch_nn`).
//! All convolutions use 3x3 kernels with padding=1 (same convolution).

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::shape::reshape;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::{AdaptiveAvgPool2d, MaxPool2d};
use ferrotorch_nn::{Conv2d, Dropout, Linear};

// ===========================================================================
// Thin wrappers so relu / dropout compose into Vec<Box<dyn Module<T>>>
// ===========================================================================

/// Conv2d followed by ReLU (no learnable params beyond the conv).
struct ConvReLU<T: Float> {
    conv: Conv2d<T>,
    training: bool,
}

impl<T: Float> ConvReLU<T> {
    fn new(in_channels: usize, out_channels: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            conv: Conv2d::new(in_channels, out_channels, (3, 3), (1, 1), (1, 1), false)?,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for ConvReLU<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.conv.forward(input)?;
        relu(&x)
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
        self.conv.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.conv.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

/// Linear -> ReLU -> Dropout (classifier block).
struct LinearReLUDropout<T: Float> {
    linear: Linear<T>,
    dropout: Dropout<T>,
    training: bool,
}

impl<T: Float> LinearReLUDropout<T> {
    fn new(in_features: usize, out_features: usize, drop_p: f64) -> FerrotorchResult<Self> {
        Ok(Self {
            linear: Linear::new(in_features, out_features, true)?,
            dropout: Dropout::new(drop_p)?,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for LinearReLUDropout<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.linear.forward(input)?;
        let x = relu(&x)?;
        self.dropout.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        self.linear.parameters()
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        self.linear.parameters_mut()
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        self.linear.named_parameters()
    }

    fn train(&mut self) {
        self.training = true;
        self.dropout.train();
    }

    fn eval(&mut self) {
        self.training = false;
        self.dropout.eval();
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// VGG configuration
// ===========================================================================

/// A single element in a VGG feature configuration.
enum VggCfg {
    /// Conv2d with the given number of output channels (3x3, pad=1).
    Conv(usize),
    /// MaxPool2d(2, 2).
    Pool,
}

/// VGG-11 configuration: `[64, M, 128, M, 256, 256, M, 512, 512, M, 512, 512, M]`.
fn vgg11_cfg() -> Vec<VggCfg> {
    use VggCfg::*;
    vec![
        Conv(64),
        Pool,
        Conv(128),
        Pool,
        Conv(256),
        Conv(256),
        Pool,
        Conv(512),
        Conv(512),
        Pool,
        Conv(512),
        Conv(512),
        Pool,
    ]
}

/// VGG-16 configuration: `[64, 64, M, 128, 128, M, 256, 256, 256, M, 512, 512, 512, M, 512, 512, 512, M]`.
fn vgg16_cfg() -> Vec<VggCfg> {
    use VggCfg::*;
    vec![
        Conv(64),
        Conv(64),
        Pool,
        Conv(128),
        Conv(128),
        Pool,
        Conv(256),
        Conv(256),
        Conv(256),
        Pool,
        Conv(512),
        Conv(512),
        Conv(512),
        Pool,
        Conv(512),
        Conv(512),
        Conv(512),
        Pool,
    ]
}

// ===========================================================================
// Feature / classifier builders
// ===========================================================================

/// Build the feature extraction layers from a VGG configuration.
///
/// Each `Conv` entry becomes a `ConvReLU` (3x3, pad=1), each `Pool`
/// becomes a `MaxPool2d(2, 2)`.
fn make_features<T: Float>(cfg: Vec<VggCfg>) -> FerrotorchResult<Vec<Box<dyn Module<T>>>> {
    let mut layers: Vec<Box<dyn Module<T>>> = Vec::new();
    let mut in_channels: usize = 3;

    for entry in cfg {
        match entry {
            VggCfg::Conv(out_channels) => {
                layers.push(Box::new(ConvReLU::new(in_channels, out_channels)?));
                in_channels = out_channels;
            }
            VggCfg::Pool => {
                layers.push(Box::new(MaxPool2d::new([2, 2], [2, 2], [0, 0])));
            }
        }
    }

    Ok(layers)
}

/// Build the classifier head.
///
/// ```text
/// Linear(512*7*7, 4096) -> ReLU -> Dropout(0.5)
/// Linear(4096, 4096)    -> ReLU -> Dropout(0.5)
/// Linear(4096, num_classes)
/// ```
fn make_classifier<T: Float>(num_classes: usize) -> FerrotorchResult<Vec<Box<dyn Module<T>>>> {
    let layers: Vec<Box<dyn Module<T>>> = vec![
        Box::new(LinearReLUDropout::new(512 * 7 * 7, 4096, 0.5)?),
        Box::new(LinearReLUDropout::new(4096, 4096, 0.5)?),
        Box::new(Linear::new(4096, num_classes, true)?),
    ];

    Ok(layers)
}

// ===========================================================================
// VGG
// ===========================================================================

/// A VGG model.
///
/// The architecture follows the standard VGG design:
///
/// 1. Feature extraction: stacks of 3x3 Conv2d + ReLU, interleaved with
///    2x2 MaxPool2d layers.
/// 2. Adaptive average pool to (7, 7).
/// 3. Classifier: two hidden Linear(4096) layers with ReLU and Dropout,
///    followed by a final Linear to `num_classes`.
///
/// Batch normalization is omitted (not yet in `ferrotorch_nn`).
pub struct VGG<T: Float> {
    features: Vec<Box<dyn Module<T>>>,
    avgpool: AdaptiveAvgPool2d,
    classifier: Vec<Box<dyn Module<T>>>,
    training: bool,
}

impl<T: Float> VGG<T> {
    /// Build a VGG model from a feature configuration and number of classes.
    fn from_cfg(cfg: Vec<VggCfg>, num_classes: usize) -> FerrotorchResult<Self> {
        let features = make_features(cfg)?;
        let avgpool = AdaptiveAvgPool2d::new((7, 7));
        let classifier = make_classifier(num_classes)?;

        Ok(Self {
            features,
            avgpool,
            classifier,
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

impl<T: Float> Module<T> for VGG<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Feature extraction.
        let mut x = self.features[0].forward(input)?;
        for layer in &self.features[1..] {
            x = layer.forward(&x)?;
        }

        // Adaptive average pool: [B, 512, H, W] -> [B, 512, 7, 7].
        let x = Module::<T>::forward(&self.avgpool, &x)?;

        // Flatten to [B, 512*7*7].
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;

        // Classifier.
        let mut x = self.classifier[0].forward(&x)?;
        for layer in &self.classifier[1..] {
            x = layer.forward(&x)?;
        }

        Ok(x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        for layer in &self.features {
            params.extend(layer.parameters());
        }
        for layer in &self.classifier {
            params.extend(layer.parameters());
        }
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        for layer in &mut self.features {
            params.extend(layer.parameters_mut());
        }
        for layer in &mut self.classifier {
            params.extend(layer.parameters_mut());
        }
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (i, layer) in self.features.iter().enumerate() {
            for (name, p) in layer.named_parameters() {
                params.push((format!("features.{i}.{name}"), p));
            }
        }
        for (i, layer) in self.classifier.iter().enumerate() {
            for (name, p) in layer.named_parameters() {
                params.push((format!("classifier.{i}.{name}"), p));
            }
        }
        params
    }

    fn train(&mut self) {
        self.training = true;
        for layer in &mut self.features {
            layer.train();
        }
        for layer in &mut self.classifier {
            layer.train();
        }
    }

    fn eval(&mut self) {
        self.training = false;
        for layer in &mut self.features {
            layer.eval();
        }
        for layer in &mut self.classifier {
            layer.eval();
        }
    }

    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// Convenience constructors
// ===========================================================================

/// Construct a VGG-11 model.
///
/// Architecture: 8 conv layers + 3 FC layers, ~132.9M parameters (without BN).
pub fn vgg11<T: Float>(num_classes: usize) -> FerrotorchResult<VGG<T>> {
    VGG::from_cfg(vgg11_cfg(), num_classes)
}

/// Construct a VGG-16 model.
///
/// Architecture: 13 conv layers + 3 FC layers, ~138.4M parameters (without BN).
pub fn vgg16<T: Float>(num_classes: usize) -> FerrotorchResult<VGG<T>> {
    VGG::from_cfg(vgg16_cfg(), num_classes)
}

// ===========================================================================
// IntermediateFeatures — CL-499
// ===========================================================================

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for VGG<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();
        // Each entry in `features` is either a ConvReLU or a MaxPool.
        // Emit one output per layer for maximum flexibility.
        let mut x = self.features[0].forward(input)?;
        out.insert("features.0".to_string(), x.clone());
        for (i, layer) in self.features.iter().enumerate().skip(1) {
            x = layer.forward(&x)?;
            out.insert(format!("features.{i}"), x.clone());
        }

        let x = Module::<T>::forward(&self.avgpool, &x)?;
        out.insert("avgpool".to_string(), x.clone());

        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let mut x = reshape(&x, &[batch as isize, features as isize])?;
        for (i, layer) in self.classifier.iter().enumerate() {
            x = layer.forward(&x)?;
            out.insert(format!("classifier.{i}"), x.clone());
        }
        Ok(out)
    }

    fn feature_node_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for i in 0..self.features.len() {
            names.push(format!("features.{i}"));
        }
        names.push("avgpool".to_string());
        for i in 0..self.classifier.len() {
            names.push(format!("classifier.{i}"));
        }
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
    // VGG-11 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vgg11_output_shape() {
        let model = vgg11::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 1 * 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    #[test]
    fn test_vgg11_param_count() {
        let model = vgg11::<f32>(1000).unwrap();
        let total = model.num_parameters();
        // VGG-11 without BN: ~132.9M params.
        //
        // Features (conv weights only, no bias):
        //   3*64*3*3 + 64*128*3*3 + 128*256*3*3 + 256*256*3*3
        //   + 256*512*3*3 + 512*512*3*3 + 512*512*3*3 + 512*512*3*3
        //   = 1728 + 73728 + 294912 + 589824
        //   + 1179648 + 2359296 + 2359296 + 2359296
        //   = 9_217_728
        //
        // Classifier (weights + biases):
        //   512*7*7*4096 + 4096 + 4096*4096 + 4096 + 4096*1000 + 1000
        //   = 102_760_448 + 4096 + 16_777_216 + 4096 + 4_096_000 + 1000
        //   = 123_642_856
        //
        // Total: 9_217_728 + 123_642_856 = 132_860_584
        assert!(
            total > 132_000_000,
            "VGG-11 should have >132M params, got {total}"
        );
        assert!(
            total < 134_000_000,
            "VGG-11 should have <134M params, got {total}"
        );
    }

    #[test]
    fn test_vgg11_custom_classes() {
        let model = vgg11::<f32>(10).unwrap();
        let input = leaf_4d(&vec![0.01; 2 * 3 * 224 * 224], [2, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 10]);
    }

    // -----------------------------------------------------------------------
    // VGG-16 tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_vgg16_output_shape() {
        let model = vgg16::<f32>(1000).unwrap();
        let input = leaf_4d(&vec![0.01; 1 * 3 * 224 * 224], [1, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 1000]);
    }

    #[test]
    fn test_vgg16_param_count() {
        let model = vgg16::<f32>(1000).unwrap();
        let total = model.num_parameters();
        // VGG-16 without BN: ~138.4M params.
        //
        // Features (conv weights only, no bias):
        //   3*64*3*3 + 64*64*3*3
        //   + 64*128*3*3 + 128*128*3*3
        //   + 128*256*3*3 + 256*256*3*3 + 256*256*3*3
        //   + 256*512*3*3 + 512*512*3*3 + 512*512*3*3
        //   + 512*512*3*3 + 512*512*3*3 + 512*512*3*3
        //   = 14_714_688
        //
        // Classifier: same as VGG-11 = 123_642_856
        //
        // Total: 14_714_688 + 123_642_856 = 138_357_544
        assert!(
            total > 138_000_000,
            "VGG-16 should have >138M params, got {total}"
        );
        assert!(
            total < 139_000_000,
            "VGG-16 should have <139M params, got {total}"
        );
    }

    #[test]
    fn test_vgg16_custom_classes() {
        let model = vgg16::<f32>(10).unwrap();
        let input = leaf_4d(&vec![0.01; 2 * 3 * 224 * 224], [2, 3, 224, 224], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[2, 10]);
    }

    // -----------------------------------------------------------------------
    // Named parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_vgg11_named_parameters_prefixes() {
        let model = vgg11::<f32>(1000).unwrap();
        let named = model.named_parameters();
        assert!(!named.is_empty());

        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("features.")));
        assert!(names.iter().any(|n| n.starts_with("classifier.")));
    }

    #[test]
    fn test_vgg16_named_parameters_prefixes() {
        let model = vgg16::<f32>(1000).unwrap();
        let named = model.named_parameters();
        assert!(!named.is_empty());

        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("features.")));
        assert!(names.iter().any(|n| n.starts_with("classifier.")));
    }

    // -----------------------------------------------------------------------
    // Train / eval
    // -----------------------------------------------------------------------

    #[test]
    fn test_vgg_train_eval() {
        let mut model = vgg11::<f32>(10).unwrap();
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
    fn test_vgg_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<VGG<f32>>();
    }
}
