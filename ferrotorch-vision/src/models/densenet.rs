//! DenseNet-121 architecture (Phase 6 #989: BN + torchvision-flat naming).
//!
//! Follows Huang et al. 2017 "Densely Connected Convolutional Networks" and
//! mirrors `torchvision.models.densenet121` (torchvision 0.21.x).
//!
//! ## Architecture
//!
//! ```text
//! input [B, 3, H, W]
//!   ├─ features.conv0     : Conv2d(3, 64, 7×7, s=2, p=3, bias=false)
//!   ├─ features.norm0     : BatchNorm2d(64)
//!   ├─ features.pool0     : MaxPool2d(3×3, s=2, p=1)
//!   ├─ features.denseblock1 : 6 × _DenseLayer (growth_rate=32)
//!   ├─ features.transition1 : (norm, conv1×1, AvgPool2d 2×2)
//!   ├─ features.denseblock2 : 12 × _DenseLayer
//!   ├─ features.transition2
//!   ├─ features.denseblock3 : 24 × _DenseLayer
//!   ├─ features.transition3
//!   ├─ features.denseblock4 : 16 × _DenseLayer
//!   └─ features.norm5     : BatchNorm2d(1024)
//!        ↓ relu (functional) → adaptive_avg_pool2d → flatten →
//!   └─ classifier         : Linear(1024, num_classes)
//! ```
//!
//! Each `_DenseLayer` runs `BN → ReLU → Conv1×1 (bn_size·growth) →
//! BN → ReLU → Conv3×3 (growth)` and concatenates the new feature maps
//! onto its input along axis=1.
//!
//! Phase 6 (#989) lands the BN-bearing version. Prior implementation was
//! BN-free with ferrotorch-internal names (`stem`, `block<i>.layer<j>`,
//! `trans<i>`, `bn_conv`/`body_conv`); the new layout produces
//! `named_parameters()` keys that match torchvision exactly so the
//! strict value-parity loader can adopt `densenet121(weights=...)` state
//! dicts without remap.

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::grad_fns::shape::{cat, reshape};
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::module::Module;
use ferrotorch_nn::norm::BatchNorm2d;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::{AdaptiveAvgPool2d, AvgPool2d, MaxPool2d};
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
// DenseLayer — one BN-ReLU-Conv-BN-ReLU-Conv bottleneck per torchvision
// ===========================================================================

/// One layer inside a dense block. Computes `growth_rate` new feature
/// maps from the concatenated input via the
/// `BN → ReLU → Conv1×1 → BN → ReLU → Conv3×3` bottleneck, then
/// concatenates the result with the original input along axis=1.
///
/// Field names match torchvision's `_DenseLayer`: `norm1`, `conv1`,
/// `norm2`, `conv2`. The intermediate ReLU activations are functional
/// (no learnable parameters), so they do not appear in
/// `named_parameters()` — but they are correctly applied between the
/// BN and Conv layers.
pub struct DenseLayer<T: Float> {
    norm1: BatchNorm2d<T>,
    conv1: Conv2d<T>,
    norm2: BatchNorm2d<T>,
    conv2: Conv2d<T>,
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
            norm1: BatchNorm2d::new(in_ch, 1e-5, 0.1, true)?,
            conv1: conv(in_ch, inter, 1, 1, 0)?,
            norm2: BatchNorm2d::new(inter, 1e-5, 0.1, true)?,
            conv2: conv(inter, growth_rate, 3, 1, 1)?,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for DenseLayer<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = Module::<T>::forward(&self.norm1, input)?;
        let x = relu(&x)?;
        let x = self.conv1.forward(&x)?;
        let x = Module::<T>::forward(&self.norm2, &x)?;
        let x = relu(&x)?;
        let x = self.conv2.forward(&x)?;
        // Concatenate the new feature maps with the input along the
        // channel axis (axis 1 for NCHW), matching torchvision.
        cat(&[input.clone(), x], 1)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.norm1.parameters());
        p.extend(self.conv1.parameters());
        p.extend(self.norm2.parameters());
        p.extend(self.conv2.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.norm1.parameters_mut());
        p.extend(self.conv1.parameters_mut());
        p.extend(self.norm2.parameters_mut());
        p.extend(self.conv2.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.norm1.named_parameters() {
            p.push((format!("norm1.{n}"), param));
        }
        for (n, param) in self.conv1.named_parameters() {
            p.push((format!("conv1.{n}"), param));
        }
        for (n, param) in self.norm2.named_parameters() {
            p.push((format!("norm2.{n}"), param));
        }
        for (n, param) in self.conv2.named_parameters() {
            p.push((format!("conv2.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.norm1, &self.conv1, &self.norm2, &self.conv2]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("norm1".to_string(), &self.norm1),
            ("conv1".to_string(), &self.conv1),
            ("norm2".to_string(), &self.norm2),
            ("conv2".to_string(), &self.conv2),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.norm1.train();
        self.norm2.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.norm1.eval();
        self.norm2.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

// ===========================================================================
// DenseBlock
// ===========================================================================

/// A stack of [`DenseLayer`]s forming one dense block. torchvision indexes
/// these 1-based (`denselayer1`, `denselayer2`, ...), so
/// `named_parameters()` mirrors that.
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
        // torchvision indexes denselayer1, denselayer2, ... (1-based).
        for (i, l) in self.layers.iter().enumerate() {
            let layer_idx = i + 1;
            for (n, param) in l.named_parameters() {
                p.push((format!("denselayer{layer_idx}.{n}"), param));
            }
        }
        p
    }
    fn children(&self) -> Vec<&dyn Module<T>> {
        self.layers
            .iter()
            .map(|l| l as &dyn Module<T>)
            .collect()
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        self.layers
            .iter()
            .enumerate()
            .map(|(i, l)| (format!("denselayer{}", i + 1), l as &dyn Module<T>))
            .collect()
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
// TransitionLayer — BN, ReLU, 1×1 Conv, 2×2 AvgPool
// ===========================================================================

/// A transition layer between dense blocks: `BN → ReLU → Conv1×1 → AvgPool2×2`.
/// Halves channels and spatial dimensions, matching torchvision's
/// `_Transition` (`norm`, `relu`, `conv`, `pool`). Note that `relu` is a
/// no-parameter activation; it does not appear in `named_parameters()`.
pub struct TransitionLayer<T: Float> {
    norm: BatchNorm2d<T>,
    conv: Conv2d<T>,
    pool: AvgPool2d,
    training: bool,
}

impl<T: Float> TransitionLayer<T> {
    pub fn new(in_ch: usize, out_ch: usize) -> FerrotorchResult<Self> {
        Ok(Self {
            norm: BatchNorm2d::new(in_ch, 1e-5, 0.1, true)?,
            conv: conv(in_ch, out_ch, 1, 1, 0)?,
            pool: AvgPool2d::new([2, 2], [2, 2], [0, 0]),
            training: true,
        })
    }
}

impl<T: Float> Module<T> for TransitionLayer<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = Module::<T>::forward(&self.norm, input)?;
        let x = relu(&x)?;
        let x = self.conv.forward(&x)?;
        Module::<T>::forward(&self.pool, &x)
    }
    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.norm.parameters());
        p.extend(self.conv.parameters());
        p
    }
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.norm.parameters_mut());
        p.extend(self.conv.parameters_mut());
        p
    }
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.norm.named_parameters() {
            p.push((format!("norm.{n}"), param));
        }
        for (n, param) in self.conv.named_parameters() {
            p.push((format!("conv.{n}"), param));
        }
        p
    }
    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![&self.norm, &self.conv, &self.pool]
    }
    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("norm".to_string(), &self.norm),
            ("conv".to_string(), &self.conv),
            ("pool".to_string(), &self.pool),
        ]
    }
    fn train(&mut self) {
        self.training = true;
        self.norm.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.norm.eval();
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
///
/// `named_parameters()` produces torchvision-flat keys:
/// `features.conv0.weight`, `features.norm0.{weight,bias}`,
/// `features.denseblock<i>.denselayer<j>.norm1.{weight,bias}`, ...,
/// `features.norm5.{weight,bias}`, `classifier.{weight,bias}`. BN
/// running statistics live under the same paths and are reachable via
/// `named_descendants_dyn()` thanks to the `named_children` overrides
/// above.
pub struct DenseNet<T: Float> {
    // features.* (torchvision-flat layout)
    conv0: Conv2d<T>,
    norm0: BatchNorm2d<T>,
    pool0: MaxPool2d,
    denseblock1: DenseBlock<T>,
    transition1: TransitionLayer<T>,
    denseblock2: DenseBlock<T>,
    transition2: TransitionLayer<T>,
    denseblock3: DenseBlock<T>,
    transition3: TransitionLayer<T>,
    denseblock4: DenseBlock<T>,
    norm5: BatchNorm2d<T>,

    // top-level
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

        // Stem (features.conv0 + norm0 + pool0)
        let conv0 = conv(3, num_init_features, 7, 2, 3)?;
        let norm0 = BatchNorm2d::new(num_init_features, 1e-5, 0.1, true)?;
        let pool0 = MaxPool2d::new([3, 3], [2, 2], [1, 1]);

        let mut in_ch = num_init_features;

        let denseblock1 = DenseBlock::new(block_config[0], in_ch, growth_rate, bn_size)?;
        in_ch = denseblock1.output_channels(in_ch, growth_rate);
        let transition1 = TransitionLayer::new(in_ch, in_ch / 2)?;
        in_ch /= 2;

        let denseblock2 = DenseBlock::new(block_config[1], in_ch, growth_rate, bn_size)?;
        in_ch = denseblock2.output_channels(in_ch, growth_rate);
        let transition2 = TransitionLayer::new(in_ch, in_ch / 2)?;
        in_ch /= 2;

        let denseblock3 = DenseBlock::new(block_config[2], in_ch, growth_rate, bn_size)?;
        in_ch = denseblock3.output_channels(in_ch, growth_rate);
        let transition3 = TransitionLayer::new(in_ch, in_ch / 2)?;
        in_ch /= 2;

        let denseblock4 = DenseBlock::new(block_config[3], in_ch, growth_rate, bn_size)?;
        in_ch = denseblock4.output_channels(in_ch, growth_rate);

        let norm5 = BatchNorm2d::new(in_ch, 1e-5, 0.1, true)?;

        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let classifier = Linear::new(in_ch, num_classes, true)?;

        Ok(Self {
            conv0,
            norm0,
            pool0,
            denseblock1,
            transition1,
            denseblock2,
            transition2,
            denseblock3,
            transition3,
            denseblock4,
            norm5,
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
        // features = conv0 → norm0 → relu → pool0 →
        //            denseblock1 → transition1 → ... → denseblock4 → norm5
        let x = self.conv0.forward(input)?;
        let x = Module::<T>::forward(&self.norm0, &x)?;
        let x = relu(&x)?;
        let x = Module::<T>::forward(&self.pool0, &x)?;
        let x = self.denseblock1.forward(&x)?;
        let x = self.transition1.forward(&x)?;
        let x = self.denseblock2.forward(&x)?;
        let x = self.transition2.forward(&x)?;
        let x = self.denseblock3.forward(&x)?;
        let x = self.transition3.forward(&x)?;
        let x = self.denseblock4.forward(&x)?;
        let x = Module::<T>::forward(&self.norm5, &x)?;

        // out = relu(features) → adaptive_avg_pool → flatten → classifier
        let x = relu(&x)?;
        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv0.parameters());
        p.extend(self.norm0.parameters());
        p.extend(self.denseblock1.parameters());
        p.extend(self.transition1.parameters());
        p.extend(self.denseblock2.parameters());
        p.extend(self.transition2.parameters());
        p.extend(self.denseblock3.parameters());
        p.extend(self.transition3.parameters());
        p.extend(self.denseblock4.parameters());
        p.extend(self.norm5.parameters());
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.conv0.parameters_mut());
        p.extend(self.norm0.parameters_mut());
        p.extend(self.denseblock1.parameters_mut());
        p.extend(self.transition1.parameters_mut());
        p.extend(self.denseblock2.parameters_mut());
        p.extend(self.transition2.parameters_mut());
        p.extend(self.denseblock3.parameters_mut());
        p.extend(self.transition3.parameters_mut());
        p.extend(self.denseblock4.parameters_mut());
        p.extend(self.norm5.parameters_mut());
        p.extend(self.classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        // Everything inside the torchvision `features.*` Sequential
        // gets a `features.` prefix here; torchvision exposes
        // `features.conv0.weight`, `features.denseblock1.denselayer1...`,
        // `features.norm5.weight`, etc.
        for (n, param) in self.conv0.named_parameters() {
            p.push((format!("features.conv0.{n}"), param));
        }
        for (n, param) in self.norm0.named_parameters() {
            p.push((format!("features.norm0.{n}"), param));
        }
        for (n, param) in self.denseblock1.named_parameters() {
            p.push((format!("features.denseblock1.{n}"), param));
        }
        for (n, param) in self.transition1.named_parameters() {
            p.push((format!("features.transition1.{n}"), param));
        }
        for (n, param) in self.denseblock2.named_parameters() {
            p.push((format!("features.denseblock2.{n}"), param));
        }
        for (n, param) in self.transition2.named_parameters() {
            p.push((format!("features.transition2.{n}"), param));
        }
        for (n, param) in self.denseblock3.named_parameters() {
            p.push((format!("features.denseblock3.{n}"), param));
        }
        for (n, param) in self.transition3.named_parameters() {
            p.push((format!("features.transition3.{n}"), param));
        }
        for (n, param) in self.denseblock4.named_parameters() {
            p.push((format!("features.denseblock4.{n}"), param));
        }
        for (n, param) in self.norm5.named_parameters() {
            p.push((format!("features.norm5.{n}"), param));
        }
        for (n, param) in self.classifier.named_parameters() {
            p.push((format!("classifier.{n}"), param));
        }
        p
    }

    fn children(&self) -> Vec<&dyn Module<T>> {
        vec![
            &self.conv0,
            &self.norm0,
            &self.pool0,
            &self.denseblock1,
            &self.transition1,
            &self.denseblock2,
            &self.transition2,
            &self.denseblock3,
            &self.transition3,
            &self.denseblock4,
            &self.norm5,
            &self.avgpool,
            &self.classifier,
        ]
    }

    fn named_children(&self) -> Vec<(String, &dyn Module<T>)> {
        vec![
            ("features.conv0".to_string(), &self.conv0),
            ("features.norm0".to_string(), &self.norm0),
            ("features.pool0".to_string(), &self.pool0),
            ("features.denseblock1".to_string(), &self.denseblock1),
            ("features.transition1".to_string(), &self.transition1),
            ("features.denseblock2".to_string(), &self.denseblock2),
            ("features.transition2".to_string(), &self.transition2),
            ("features.denseblock3".to_string(), &self.denseblock3),
            ("features.transition3".to_string(), &self.transition3),
            ("features.denseblock4".to_string(), &self.denseblock4),
            ("features.norm5".to_string(), &self.norm5),
            // avgpool has no torchvision-side path (it's functional in
            // torchvision's forward), so we keep an internal path so
            // the children walk doesn't drop a real Module.
            ("avgpool".to_string(), &self.avgpool),
            ("classifier".to_string(), &self.classifier),
        ]
    }

    fn train(&mut self) {
        self.training = true;
        self.norm0.train();
        self.denseblock1.train();
        self.transition1.train();
        self.denseblock2.train();
        self.transition2.train();
        self.denseblock3.train();
        self.transition3.train();
        self.denseblock4.train();
        self.norm5.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.norm0.eval();
        self.denseblock1.eval();
        self.transition1.eval();
        self.denseblock2.eval();
        self.transition2.eval();
        self.denseblock3.eval();
        self.transition3.eval();
        self.denseblock4.eval();
        self.norm5.eval();
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
// IntermediateFeatures — CL-499 (paths follow the new torchvision-flat layout)
// ---------------------------------------------------------------------------

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for DenseNet<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();

        let x = self.conv0.forward(input)?;
        let x = Module::<T>::forward(&self.norm0, &x)?;
        let x = relu(&x)?;
        let x = Module::<T>::forward(&self.pool0, &x)?;
        out.insert("features.pool0".to_string(), x.clone());

        let x = self.denseblock1.forward(&x)?;
        out.insert("features.denseblock1".to_string(), x.clone());
        let x = self.transition1.forward(&x)?;
        out.insert("features.transition1".to_string(), x.clone());

        let x = self.denseblock2.forward(&x)?;
        out.insert("features.denseblock2".to_string(), x.clone());
        let x = self.transition2.forward(&x)?;
        out.insert("features.transition2".to_string(), x.clone());

        let x = self.denseblock3.forward(&x)?;
        out.insert("features.denseblock3".to_string(), x.clone());
        let x = self.transition3.forward(&x)?;
        out.insert("features.transition3".to_string(), x.clone());

        let x = self.denseblock4.forward(&x)?;
        out.insert("features.denseblock4".to_string(), x.clone());

        let x = Module::<T>::forward(&self.norm5, &x)?;
        out.insert("features.norm5".to_string(), x.clone());

        let x = relu(&x)?;
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
            "features.pool0".to_string(),
            "features.denseblock1".to_string(),
            "features.transition1".to_string(),
            "features.denseblock2".to_string(),
            "features.transition2".to_string(),
            "features.denseblock3".to_string(),
            "features.transition3".to_string(),
            "features.denseblock4".to_string(),
            "features.norm5".to_string(),
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
        let mut layer_eval = layer;
        layer_eval.eval();
        let y = layer_eval.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 6, 8, 8]);
    }

    #[test]
    fn test_dense_block_output_channels_calculation() {
        // 3 layers, in_ch=8, growth_rate=4 → 8 + 3*4 = 20
        let block: DenseBlock<f32> = DenseBlock::new(3, 8, 4, 4).unwrap();
        assert_eq!(block.output_channels(8, 4), 20);

        let x = dummy_image(1, 8, 8, 8);
        let mut block_eval = block;
        block_eval.eval();
        let y = block_eval.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 20, 8, 8]);
    }

    #[test]
    fn test_transition_layer_halves_spatial() {
        let trans: TransitionLayer<f32> = TransitionLayer::new(8, 4).unwrap();
        let x = dummy_image(1, 8, 16, 16);
        let mut trans_eval = trans;
        trans_eval.eval();
        let y = trans_eval.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 4, 8, 8]);
    }

    #[test]
    fn test_densenet121_output_shape() {
        let mut model: DenseNet<f32> = densenet121(10).unwrap();
        model.eval();
        let x = dummy_image(1, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 10]);
    }

    #[test]
    fn test_densenet121_custom_classes() {
        let mut model: DenseNet<f32> = densenet121(7).unwrap();
        model.eval();
        let x = dummy_image(1, 3, 32, 32);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 7]);
    }

    #[test]
    fn test_densenet121_param_count() {
        let model: DenseNet<f32> = densenet121(1000).unwrap();
        let total = model.num_parameters();
        // torchvision densenet121 has ~7.98M params (8.0M including BN).
        // ferrotorch BN-bearing variant produces the same count.
        assert!(
            total > 7_000_000,
            "DenseNet-121 should have >7M params, got {total}"
        );
        assert!(
            total < 9_000_000,
            "DenseNet-121 should have <9M params, got {total}"
        );
    }

    #[test]
    fn test_densenet121_named_parameters_prefixes() {
        let model: DenseNet<f32> = densenet121(10).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("features.conv0.")));
        assert!(names.iter().any(|n| n.starts_with("features.norm0.")));
        assert!(names
            .iter()
            .any(|n| n.starts_with("features.denseblock1.denselayer1.")));
        assert!(names
            .iter()
            .any(|n| n.starts_with("features.transition1.")));
        assert!(names.iter().any(|n| n.starts_with("features.norm5.")));
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
