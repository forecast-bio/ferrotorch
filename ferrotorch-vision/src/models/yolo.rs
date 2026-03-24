//! Simplified YOLO (You Only Look Once) single-shot object detection model.
//!
//! This implements a simplified DarkNet-style backbone with 5 convolutional
//! stages followed by a 1x1 detection head. The model outputs a dense
//! prediction grid where each cell predicts bounding boxes.
//!
//! Architecture:
//!
//! ```text
//! Input: [B, 3, 416, 416]
//!
//! Backbone (5 stages, each: Conv2d -> ReLU -> MaxPool(2)):
//!   Stage 1: 3   -> 32   (416 -> 208)
//!   Stage 2: 32  -> 64   (208 -> 104)
//!   Stage 3: 64  -> 128  (104 -> 52)
//!   Stage 4: 128 -> 256  (52  -> 26)
//!   Stage 5: 256 -> 512  (26  -> 13)
//!
//! Detection head:
//!   Conv2d(512, num_anchors * (5 + num_classes), 1x1)
//!
//! Output: [B, num_anchors * (5 + num_classes), 13, 13]
//! ```
//!
//! Each anchor predicts `(x, y, w, h, objectness)` plus `num_classes` class
//! scores. Default: 3 anchors.
//!
//! Batch normalization is omitted (not yet in `ferrotorch_nn`).

use ferrotorch_core::grad_fns::activation::relu;
use ferrotorch_core::{FerrotorchResult, Float, Tensor};

use ferrotorch_nn::Conv2d;
use ferrotorch_nn::module::Module;
use ferrotorch_nn::parameter::Parameter;
use ferrotorch_nn::pooling::MaxPool2d;

// ===========================================================================
// Backbone stage: Conv2d(3x3, pad=1) -> ReLU -> MaxPool(2x2)
// ===========================================================================

/// A single backbone stage: 3x3 convolution + ReLU + 2x2 max pooling.
///
/// Spatial dimensions are halved by the pool. Channel count changes from
/// `in_channels` to `out_channels`.
struct BackboneStage<T: Float> {
    conv: Conv2d<T>,
    pool: MaxPool2d,
    training: bool,
}

impl<T: Float> BackboneStage<T> {
    /// Create a backbone stage.
    ///
    /// * `in_ch`  -- input channels.
    /// * `out_ch` -- output channels.
    /// * `stride` -- stride for the 3x3 convolution (typically 1).
    fn new(in_ch: usize, out_ch: usize, stride: usize) -> FerrotorchResult<Self> {
        let conv = Conv2d::new(in_ch, out_ch, (3, 3), (stride, stride), (1, 1), false)?;
        let pool = MaxPool2d::new([2, 2], [2, 2], [0, 0]);
        Ok(Self {
            conv,
            pool,
            training: true,
        })
    }
}

impl<T: Float> Module<T> for BackboneStage<T> {
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
// Yolo
// ===========================================================================

/// Simplified YOLO single-shot detection model.
///
/// The model takes `[B, 3, 416, 416]` images and produces a dense grid of
/// predictions with shape `[B, num_anchors * (5 + num_classes), 13, 13]`.
///
/// Each of the `num_anchors` predictions per grid cell contains:
/// - `x, y` -- center offsets within the cell
/// - `w, h` -- bounding box dimensions
/// - `objectness` -- confidence that an object exists
/// - `num_classes` class scores
pub struct Yolo<T: Float> {
    // Backbone: 5 stages.
    stage1: BackboneStage<T>,
    stage2: BackboneStage<T>,
    stage3: BackboneStage<T>,
    stage4: BackboneStage<T>,
    stage5: BackboneStage<T>,

    // Detection head: 1x1 conv.
    head: Conv2d<T>,

    /// Number of object classes.
    num_classes: usize,

    /// Number of anchor boxes per grid cell.
    num_anchors: usize,

    training: bool,
}

impl<T: Float> Yolo<T> {
    /// Construct a simplified YOLO model.
    ///
    /// * `num_classes` -- number of object classes to detect.
    /// * `num_anchors` -- number of anchor boxes per grid cell (default: 3).
    ///
    /// Input shape: `[B, 3, 416, 416]`.
    /// Output shape: `[B, num_anchors * (5 + num_classes), 13, 13]`.
    pub fn new(num_classes: usize, num_anchors: usize) -> FerrotorchResult<Self> {
        let stage1 = BackboneStage::new(3, 32, 1)?;
        let stage2 = BackboneStage::new(32, 64, 1)?;
        let stage3 = BackboneStage::new(64, 128, 1)?;
        let stage4 = BackboneStage::new(128, 256, 1)?;
        let stage5 = BackboneStage::new(256, 512, 1)?;

        let out_channels = num_anchors * (5 + num_classes);
        let head = Conv2d::new(512, out_channels, (1, 1), (1, 1), (0, 0), true)?;

        Ok(Self {
            stage1,
            stage2,
            stage3,
            stage4,
            stage5,
            head,
            num_classes,
            num_anchors,
            training: true,
        })
    }

    /// Number of object classes.
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// Number of anchor boxes per grid cell.
    pub fn num_anchors(&self) -> usize {
        self.num_anchors
    }

    /// Total number of learnable scalar parameters in the model.
    pub fn num_parameters(&self) -> usize {
        self.parameters().iter().map(|p| p.numel()).sum()
    }
}

// ---------------------------------------------------------------------------
// Module impl
// ---------------------------------------------------------------------------

impl<T: Float> Module<T> for Yolo<T> {
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>> {
        // Backbone: 5 stages of conv -> relu -> maxpool.
        let x = self.stage1.forward(input)?;
        let x = self.stage2.forward(&x)?;
        let x = self.stage3.forward(&x)?;
        let x = self.stage4.forward(&x)?;
        let x = self.stage5.forward(&x)?;

        // Detection head: 1x1 conv producing per-anchor predictions.
        self.head.forward(&x)
    }

    fn parameters(&self) -> Vec<&Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.stage1.parameters());
        params.extend(self.stage2.parameters());
        params.extend(self.stage3.parameters());
        params.extend(self.stage4.parameters());
        params.extend(self.stage5.parameters());
        params.extend(self.head.parameters());
        params
    }

    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>> {
        let mut params = Vec::new();
        params.extend(self.stage1.parameters_mut());
        params.extend(self.stage2.parameters_mut());
        params.extend(self.stage3.parameters_mut());
        params.extend(self.stage4.parameters_mut());
        params.extend(self.stage5.parameters_mut());
        params.extend(self.head.parameters_mut());
        params
    }

    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)> {
        let mut params = Vec::new();
        for (name, p) in self.stage1.named_parameters() {
            params.push((format!("backbone.stage1.{name}"), p));
        }
        for (name, p) in self.stage2.named_parameters() {
            params.push((format!("backbone.stage2.{name}"), p));
        }
        for (name, p) in self.stage3.named_parameters() {
            params.push((format!("backbone.stage3.{name}"), p));
        }
        for (name, p) in self.stage4.named_parameters() {
            params.push((format!("backbone.stage4.{name}"), p));
        }
        for (name, p) in self.stage5.named_parameters() {
            params.push((format!("backbone.stage5.{name}"), p));
        }
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
// Convenience constructor
// ===========================================================================

/// Construct a simplified YOLO model with 3 anchors.
///
/// * `num_classes` -- number of object classes.
///
/// Equivalent to `Yolo::new(num_classes, 3)`.
pub fn yolo<T: Float>(num_classes: usize) -> FerrotorchResult<Yolo<T>> {
    Yolo::new(num_classes, 3)
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
    // Forward shape tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_yolo_forward_shape_default_anchors() {
        // 20 classes (like VOC), 3 anchors: output channels = 3 * (5 + 20) = 75
        let model = Yolo::<f32>::new(20, 3).unwrap();
        let input = leaf_4d(&vec![0.01; 1 * 3 * 416 * 416], [1, 3, 416, 416], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 75, 13, 13]);
    }

    #[test]
    fn test_yolo_forward_shape_batch() {
        let model = Yolo::<f32>::new(80, 3).unwrap();
        let input = leaf_4d(&vec![0.01; 2 * 3 * 416 * 416], [2, 3, 416, 416], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        // 80 classes, 3 anchors: output channels = 3 * (5 + 80) = 255
        assert_eq!(output.shape(), &[2, 255, 13, 13]);
    }

    #[test]
    fn test_yolo_forward_shape_custom_anchors() {
        // 10 classes, 5 anchors: output channels = 5 * (5 + 10) = 75
        let model = Yolo::<f32>::new(10, 5).unwrap();
        let input = leaf_4d(&vec![0.01; 1 * 3 * 416 * 416], [1, 3, 416, 416], false);
        let output = no_grad(|| model.forward(&input).unwrap());
        assert_eq!(output.shape(), &[1, 75, 13, 13]);
    }

    #[test]
    fn test_yolo_convenience_constructor() {
        let model = yolo::<f32>(20).unwrap();
        assert_eq!(model.num_classes(), 20);
        assert_eq!(model.num_anchors(), 3);
    }

    // -----------------------------------------------------------------------
    // Parameter count
    // -----------------------------------------------------------------------

    #[test]
    fn test_yolo_parameter_count() {
        let model = Yolo::<f32>::new(20, 3).unwrap();
        let total = model.num_parameters();

        // Backbone convolutions (weight only, no bias):
        //   Stage 1: 3   * 32  * 3 * 3 =     864
        //   Stage 2: 32  * 64  * 3 * 3 =  18_432
        //   Stage 3: 64  * 128 * 3 * 3 =  73_728
        //   Stage 4: 128 * 256 * 3 * 3 = 294_912
        //   Stage 5: 256 * 512 * 3 * 3 = 1_179_648
        // Backbone total:               1_567_584
        //
        // Detection head (weight + bias):
        //   Weight: 512 * 75 * 1 * 1 = 38_400
        //   Bias:   75
        //   Head total:                 38_475
        //
        // Grand total: 1_606_059

        let backbone_params = 3 * 32 * 3 * 3
            + 32 * 64 * 3 * 3
            + 64 * 128 * 3 * 3
            + 128 * 256 * 3 * 3
            + 256 * 512 * 3 * 3;
        let out_ch = 3 * (5 + 20); // 75
        let head_params = 512 * out_ch * 1 * 1 + out_ch; // weight + bias
        let expected = backbone_params + head_params;

        assert_eq!(
            total, expected,
            "expected {expected} parameters, got {total}"
        );
    }

    #[test]
    fn test_yolo_parameter_count_coco() {
        // COCO: 80 classes, 3 anchors -> out_ch = 255
        let model = Yolo::<f32>::new(80, 3).unwrap();
        let total = model.num_parameters();

        let backbone_params = 3 * 32 * 3 * 3
            + 32 * 64 * 3 * 3
            + 64 * 128 * 3 * 3
            + 128 * 256 * 3 * 3
            + 256 * 512 * 3 * 3;
        let out_ch = 3 * (5 + 80); // 255
        let head_params = 512 * out_ch + out_ch;
        let expected = backbone_params + head_params;

        assert_eq!(
            total, expected,
            "expected {expected} parameters, got {total}"
        );
    }

    // -----------------------------------------------------------------------
    // Named parameters
    // -----------------------------------------------------------------------

    #[test]
    fn test_yolo_named_parameters_prefixes() {
        let model = Yolo::<f32>::new(20, 3).unwrap();
        let named = model.named_parameters();
        assert!(!named.is_empty());

        let names: Vec<&str> = named.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.iter().any(|n| n.starts_with("backbone.stage1.")));
        assert!(names.iter().any(|n| n.starts_with("backbone.stage2.")));
        assert!(names.iter().any(|n| n.starts_with("backbone.stage3.")));
        assert!(names.iter().any(|n| n.starts_with("backbone.stage4.")));
        assert!(names.iter().any(|n| n.starts_with("backbone.stage5.")));
        assert!(names.iter().any(|n| n.starts_with("head.")));
    }

    // -----------------------------------------------------------------------
    // Train / eval
    // -----------------------------------------------------------------------

    #[test]
    fn test_yolo_train_eval() {
        let mut model = Yolo::<f32>::new(20, 3).unwrap();
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
    fn test_yolo_is_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Yolo<f32>>();
    }

    // -----------------------------------------------------------------------
    // Gradient flow
    // -----------------------------------------------------------------------

    #[test]
    fn test_gradient_flow_through_yolo() {
        // Use tiny spatial dims to keep memory reasonable.
        // 32x32 input -> after 5 maxpool(2) -> 1x1 grid.
        let model = Yolo::<f32>::new(2, 1).unwrap();
        let input = leaf_4d(&vec![0.5; 1 * 3 * 32 * 32], [1, 3, 32, 32], true);

        let output = model.forward(&input).unwrap();
        let loss = ferrotorch_core::grad_fns::reduction::sum(&output).unwrap();
        loss.backward().unwrap();

        let grad = input.grad().unwrap();
        assert!(grad.is_some(), "input should have gradients");
        let grad_data = grad.unwrap().data().unwrap().to_vec();
        let any_nonzero = grad_data.iter().any(|&g| g.abs() > 1e-10);
        assert!(any_nonzero, "gradients should flow through YOLO");
    }
}
