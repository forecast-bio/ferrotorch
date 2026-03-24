pub mod center_crop;
pub mod color_jitter;
pub mod random_apply;
pub mod random_gaussian_blur;
pub mod random_resized_crop;
pub mod random_rotation;
pub mod random_vertical_flip;
pub mod resize;
pub mod rng;
pub mod to_tensor;
pub mod vision_normalize;

pub use center_crop::CenterCrop;
pub use color_jitter::ColorJitter;
pub use random_apply::{RandomApply, RandomChoice};
pub use random_gaussian_blur::RandomGaussianBlur;
pub use random_resized_crop::RandomResizedCrop;
pub use random_rotation::RandomRotation;
pub use random_vertical_flip::RandomVerticalFlip;
pub use resize::Resize;
pub use rng::vision_manual_seed;
pub use to_tensor::VisionToTensor;
pub use vision_normalize::VisionNormalize;

/// ImageNet channel-wise means (RGB order), used for input normalization.
pub const IMAGENET_MEAN: [f64; 3] = [0.485, 0.456, 0.406];

/// ImageNet channel-wise standard deviations (RGB order).
pub const IMAGENET_STD: [f64; 3] = [0.229, 0.224, 0.225];
