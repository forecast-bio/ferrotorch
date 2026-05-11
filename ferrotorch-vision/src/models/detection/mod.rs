//! Object detection models.
//!
//! Currently implemented:
//! - **Faster R-CNN** with ResNet-50 FPN backbone (`faster_rcnn.rs`)
//!   — mirrors `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
//! - **Mask R-CNN** with ResNet-50 FPN backbone (`mask_rcnn.rs`)
//!   — mirrors `torchvision.models.detection.maskrcnn_resnet50_fpn`.
//!   Extends Faster R-CNN with a 4-conv FCN mask head and deconv predictor.
//! - **SSD300** with VGG-16 backbone (`ssd.rs`)
//!   — mirrors `torchvision.models.detection.ssd300_vgg16`.

pub mod anchor_utils;
pub mod faster_rcnn;
pub mod fpn;
pub mod mask_rcnn;
pub mod roi_heads_postprocess;
pub mod rpn;
pub mod ssd;

pub use anchor_utils::AnchorGenerator;
pub use faster_rcnn::{Detections, FasterRcnn, TwoMlpHead, fasterrcnn_resnet50_fpn};
pub use fpn::{FPN_OUT_CHANNELS, FeaturePyramidNetwork};
pub use mask_rcnn::{MaskDetections, MaskHead, MaskPredictor, MaskRcnn, maskrcnn_resnet50_fpn};
pub use rpn::{Rpn, RpnConfig, RpnHead};
pub use ssd::{
    SSD_ANCHORS_PER_SCALE, SSD_FM_SIZES, SSD_TOTAL_ANCHORS, Ssd300, SsdDetections, ssd300_vgg16,
};
