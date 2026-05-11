//! Semantic segmentation models.
//!
//! Currently implemented:
//! - **DeepLabV3** with ResNet-50 dilated backbone (`deeplabv3.rs`)
//!   — mirrors `torchvision.models.segmentation.deeplabv3_resnet50`.
//! - **FCN** with ResNet-50 backbone (`fcn.rs`)
//!   — mirrors `torchvision.models.segmentation.fcn_resnet50`.
//! - **LRASPP** with MobileNetV3-Large dilated backbone (`lraspp.rs`)
//!   — mirrors `torchvision.models.segmentation.lraspp_mobilenet_v3_large`
//!   (#1146, Phase A.4 of real-artifact-driven development).
//!
//! Shared utility:
//! - **ASPP** module (`aspp.rs`) — Atrous Spatial Pyramid Pooling used by
//!   DeepLabV3.

pub mod aspp;
pub mod deeplabv3;
pub mod fcn;
pub mod lraspp;

pub use aspp::Aspp;
pub use deeplabv3::{DeepLabV3, DeepLabV3Head, ResNet50Dilated, deeplabv3_resnet50};
pub use fcn::{Fcn, FcnHead, fcn_resnet50};
pub use lraspp::{Lraspp, LrasppHead, lraspp_mobilenet_v3_large};
