//! Inception v3 architecture (simplified).
//!
//! Follows Szegedy et al. 2016 "Rethinking the Inception Architecture for
//! Computer Vision". The core building block is an **Inception module** —
//! four parallel branches whose outputs are concatenated along the
//! channel axis:
//!
//! 1. 1×1 conv
//! 2. 1×1 conv → 3×3 conv
//! 3. 1×1 conv → 3×3 conv → 3×3 conv
//! 4. AvgPool 3×3 → 1×1 conv
//!
//! **Simplifications.** Real Inception v3 uses factorized convolutions
//! (1×n followed by n×1), multiple module variants (A/B/C/D), grid
//! reduction modules, label-smoothing loss, and auxiliary classifiers.
//! This implementation ships a single Inception-A style module used
//! across three stages and no auxiliary head. Channel counts roughly
//! match the paper's Inception-A pattern. The stem is compressed from
//! the paper's 5-layer form into a 2-layer form to keep the model
//! runnable on modest input sizes.
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
// InceptionModule
// ===========================================================================

/// A simplified Inception-A module: four parallel branches whose
/// outputs are concatenated along the channel axis.
///
/// Branch 1: `1×1 conv`                         → `pool_proj` output channels
/// Branch 2: `1×1 conv → 3×3 conv`              → `branch3x3` output channels
/// Branch 3: `1×1 conv → 3×3 conv → 3×3 conv`   → `branch_double` output channels
/// Branch 4: `AvgPool3 → 1×1 conv`              → `pool_proj` output channels
pub struct InceptionModule<T: Float> {
    branch1x1: Conv2d<T>,
    branch3x3_reduce: Conv2d<T>,
    branch3x3: Conv2d<T>,
    branch_double_reduce: Conv2d<T>,
    branch_double_3x3a: Conv2d<T>,
    branch_double_3x3b: Conv2d<T>,
    branch_pool: AvgPool2d,
    branch_pool_proj: Conv2d<T>,
    training: bool,
}

impl<T: Float> InceptionModule<T> {
    /// Build an InceptionModule.
    ///
    /// * `in_ch` — input channel count.
    /// * `branch1x1` — output channels of the 1×1 branch.
    /// * `branch3x3_reduce` / `branch3x3` — reduction and output
    ///   channels for the 1×1→3×3 branch.
    /// * `branch_double_reduce` / `branch_double` — reduction and
    ///   output channels for the 1×1→3×3→3×3 branch.
    /// * `pool_proj` — output channels of the avgpool+1×1 branch.
    ///
    /// The total output channels are
    /// `branch1x1 + branch3x3 + branch_double + pool_proj`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_ch: usize,
        branch1x1: usize,
        branch3x3_reduce: usize,
        branch3x3: usize,
        branch_double_reduce: usize,
        branch_double: usize,
        pool_proj: usize,
    ) -> FerrotorchResult<Self> {
        Ok(Self {
            branch1x1: conv(in_ch, branch1x1, 1, 1, 0)?,
            branch3x3_reduce: conv(in_ch, branch3x3_reduce, 1, 1, 0)?,
            branch3x3: conv(branch3x3_reduce, branch3x3, 3, 1, 1)?,
            branch_double_reduce: conv(in_ch, branch_double_reduce, 1, 1, 0)?,
            branch_double_3x3a: conv(branch_double_reduce, branch_double, 3, 1, 1)?,
            branch_double_3x3b: conv(branch_double, branch_double, 3, 1, 1)?,
            branch_pool: AvgPool2d::new([3, 3], [1, 1], [1, 1]),
            branch_pool_proj: conv(in_ch, pool_proj, 1, 1, 0)?,
            training: true,
        })
    }

    /// Total output channel count.
    pub fn out_channels(
        branch1x1: usize,
        branch3x3: usize,
        branch_double: usize,
        pool_proj: usize,
    ) -> usize {
        branch1x1 + branch3x3 + branch_double + pool_proj
    }
}

impl<T: Float> Module<T> for InceptionModule<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let b1 = relu(&self.branch1x1.forward(input)?)?;

        let b2a = relu(&self.branch3x3_reduce.forward(input)?)?;
        let b2 = relu(&self.branch3x3.forward(&b2a)?)?;

        let b3a = relu(&self.branch_double_reduce.forward(input)?)?;
        let b3b = relu(&self.branch_double_3x3a.forward(&b3a)?)?;
        let b3 = relu(&self.branch_double_3x3b.forward(&b3b)?)?;

        let b4a = Module::<T>::forward(&self.branch_pool, input)?;
        let b4 = relu(&self.branch_pool_proj.forward(&b4a)?)?;

        cat(&[b1, b2, b3, b4], 1)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch1x1.parameters());
        p.extend(self.branch3x3_reduce.parameters());
        p.extend(self.branch3x3.parameters());
        p.extend(self.branch_double_reduce.parameters());
        p.extend(self.branch_double_3x3a.parameters());
        p.extend(self.branch_double_3x3b.parameters());
        p.extend(self.branch_pool_proj.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.branch1x1.parameters_mut());
        p.extend(self.branch3x3_reduce.parameters_mut());
        p.extend(self.branch3x3.parameters_mut());
        p.extend(self.branch_double_reduce.parameters_mut());
        p.extend(self.branch_double_3x3a.parameters_mut());
        p.extend(self.branch_double_3x3b.parameters_mut());
        p.extend(self.branch_pool_proj.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.branch1x1.named_parameters() {
            p.push((format!("branch1x1.{n}"), param));
        }
        for (n, param) in self.branch3x3_reduce.named_parameters() {
            p.push((format!("branch3x3_reduce.{n}"), param));
        }
        for (n, param) in self.branch3x3.named_parameters() {
            p.push((format!("branch3x3.{n}"), param));
        }
        for (n, param) in self.branch_double_reduce.named_parameters() {
            p.push((format!("branch_double_reduce.{n}"), param));
        }
        for (n, param) in self.branch_double_3x3a.named_parameters() {
            p.push((format!("branch_double_3x3a.{n}"), param));
        }
        for (n, param) in self.branch_double_3x3b.named_parameters() {
            p.push((format!("branch_double_3x3b.{n}"), param));
        }
        for (n, param) in self.branch_pool_proj.named_parameters() {
            p.push((format!("branch_pool_proj.{n}"), param));
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
// Inception v3 model
// ===========================================================================

/// A simplified Inception v3 for ImageNet-style classification.
///
/// Compressed stem: two Conv+ReLU layers, then three
/// [`InceptionModule`]s at increasing channel counts, then
/// adaptive-avgpool + linear classifier. Input and output dims match
/// the existing vision models in this crate.
pub struct InceptionV3<T: Float> {
    stem_conv1: Conv2d<T>,
    stem_conv2: Conv2d<T>,
    module_a: InceptionModule<T>,
    module_b: InceptionModule<T>,
    module_c: InceptionModule<T>,
    avgpool: AdaptiveAvgPool2d,
    classifier: Linear<T>,
    training: bool,
}

impl<T: Float> InceptionV3<T> {
    /// Construct Inception v3 with the given number of output classes.
    pub fn new(num_classes: usize) -> FerrotorchResult<Self> {
        // Stem: 3 → 32 → 64
        let stem_conv1 = conv(3, 32, 3, 2, 1)?;
        let stem_conv2 = conv(32, 64, 3, 1, 1)?;

        // Three Inception modules at 64 → 128 → 192 channels.
        // Branch sizes chosen so out_channels() gives 128 and 192.
        let module_a = InceptionModule::new(64, 32, 24, 32, 24, 32, 32)?; // 128 out
        let module_b = InceptionModule::new(128, 48, 32, 48, 32, 48, 48)?; // 192 out
        let module_c = InceptionModule::new(192, 64, 48, 64, 48, 64, 64)?; // 256 out

        let avgpool = AdaptiveAvgPool2d::new((1, 1));
        let classifier = Linear::new(256, num_classes, true)?;

        Ok(Self {
            stem_conv1,
            stem_conv2,
            module_a,
            module_b,
            module_c,
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

impl<T: Float> Module<T> for InceptionV3<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        let x = self.stem_conv1.forward(input)?;
        let x = relu(&x)?;
        let x = self.stem_conv2.forward(&x)?;
        let x = relu(&x)?;

        let x = self.module_a.forward(&x)?;
        let x = self.module_b.forward(&x)?;
        let x = self.module_c.forward(&x)?;

        let x = Module::<T>::forward(&self.avgpool, &x)?;
        let batch = x.shape()[0];
        let features = x.numel() / batch;
        let x = reshape(&x, &[batch as isize, features as isize])?;
        self.classifier.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem_conv1.parameters());
        p.extend(self.stem_conv2.parameters());
        p.extend(self.module_a.parameters());
        p.extend(self.module_b.parameters());
        p.extend(self.module_c.parameters());
        p.extend(self.classifier.parameters());
        p
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut p = Vec::new();
        p.extend(self.stem_conv1.parameters_mut());
        p.extend(self.stem_conv2.parameters_mut());
        p.extend(self.module_a.parameters_mut());
        p.extend(self.module_b.parameters_mut());
        p.extend(self.module_c.parameters_mut());
        p.extend(self.classifier.parameters_mut());
        p
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut p = Vec::new();
        for (n, param) in self.stem_conv1.named_parameters() {
            p.push((format!("stem_conv1.{n}"), param));
        }
        for (n, param) in self.stem_conv2.named_parameters() {
            p.push((format!("stem_conv2.{n}"), param));
        }
        for (n, param) in self.module_a.named_parameters() {
            p.push((format!("module_a.{n}"), param));
        }
        for (n, param) in self.module_b.named_parameters() {
            p.push((format!("module_b.{n}"), param));
        }
        for (n, param) in self.module_c.named_parameters() {
            p.push((format!("module_c.{n}"), param));
        }
        for (n, param) in self.classifier.named_parameters() {
            p.push((format!("classifier.{n}"), param));
        }
        p
    }

    fn train(&mut self) {
        self.training = true;
        self.module_a.train();
        self.module_b.train();
        self.module_c.train();
    }
    fn eval(&mut self) {
        self.training = false;
        self.module_a.eval();
        self.module_b.eval();
        self.module_c.eval();
    }
    fn is_training(&self) -> bool {
        self.training
    }
}

/// Convenience constructor for Inception v3.
pub fn inception_v3<T: Float>(num_classes: usize) -> FerrotorchResult<InceptionV3<T>> {
    InceptionV3::new(num_classes)
}

// ---------------------------------------------------------------------------
// IntermediateFeatures — CL-499
// ---------------------------------------------------------------------------

impl<T: Float> crate::models::feature_extractor::IntermediateFeatures<T> for InceptionV3<T> {
    fn forward_features(
        &self,
        input: &Tensor<T>,
    ) -> FerrotorchResult<std::collections::HashMap<String, Tensor<T>>> {
        let mut out = std::collections::HashMap::new();

        let x = self.stem_conv1.forward(input)?;
        let x = relu(&x)?;
        out.insert("stem_conv1".to_string(), x.clone());
        let x = self.stem_conv2.forward(&x)?;
        let x = relu(&x)?;
        out.insert("stem_conv2".to_string(), x.clone());

        let x = self.module_a.forward(&x)?;
        out.insert("module_a".to_string(), x.clone());
        let x = self.module_b.forward(&x)?;
        out.insert("module_b".to_string(), x.clone());
        let x = self.module_c.forward(&x)?;
        out.insert("module_c".to_string(), x.clone());

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
            "stem_conv1".to_string(),
            "stem_conv2".to_string(),
            "module_a".to_string(),
            "module_b".to_string(),
            "module_c".to_string(),
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
    fn test_inception_module_concat_channel_count() {
        // in 16, branches 8/6/8/6/8/4 → out 8+8+8+4 = 28
        let module: InceptionModule<f32> = InceptionModule::new(16, 8, 6, 8, 6, 8, 4).unwrap();
        let x = dummy_image(1, 16, 8, 8);
        let y = module.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 28, 8, 8]);
        assert_eq!(InceptionModule::<f32>::out_channels(8, 8, 8, 4), 28);
    }

    #[test]
    fn test_inception_v3_output_shape() {
        let model: InceptionV3<f32> = inception_v3(10).unwrap();
        let x = dummy_image(1, 3, 16, 16);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 10]);
    }

    #[test]
    fn test_inception_v3_custom_classes() {
        let model: InceptionV3<f32> = inception_v3(3).unwrap();
        let x = dummy_image(1, 3, 16, 16);
        let y = model.forward(&x).unwrap();
        assert_eq!(y.shape(), &[1, 3]);
    }

    #[test]
    fn test_inception_v3_param_count() {
        let model: InceptionV3<f32> = inception_v3(1000).unwrap();
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_inception_v3_named_parameters_prefixes() {
        let model: InceptionV3<f32> = inception_v3(10).unwrap();
        let names: Vec<String> = model
            .named_parameters()
            .into_iter()
            .map(|(n, _)| n)
            .collect();
        assert!(names.iter().any(|n| n.starts_with("stem_conv1.")));
        assert!(names.iter().any(|n| n.starts_with("stem_conv2.")));
        assert!(names.iter().any(|n| n.starts_with("module_a.")));
        assert!(names.iter().any(|n| n.starts_with("classifier.")));
    }

    #[test]
    fn test_inception_v3_train_eval() {
        let mut model: InceptionV3<f32> = inception_v3(10).unwrap();
        assert!(model.is_training());
        model.eval();
        assert!(!model.is_training());
        model.train();
        assert!(model.is_training());
    }
}
