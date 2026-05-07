//! Object detection models.
//!
//! Currently implemented:
//! - **Faster R-CNN** with ResNet-50 FPN backbone (`faster_rcnn.rs`)
//!   — mirrors `torchvision.models.detection.fasterrcnn_resnet50_fpn`.
//!
//! Follow-up issues:
//! - #456-mask: Mask R-CNN (extends Faster R-CNN with a mask head)
//! - #456-ssd:  SSD (fundamentally different single-stage architecture)

pub mod anchor_utils;
pub mod faster_rcnn;
pub mod fpn;
pub mod rpn;

pub use faster_rcnn::{Detections, FasterRcnn, TwoMlpHead, fasterrcnn_resnet50_fpn};
pub use fpn::{FPN_OUT_CHANNELS, FeaturePyramidNetwork};
pub use rpn::{Rpn, RpnConfig, RpnHead};
pub use anchor_utils::AnchorGenerator;
