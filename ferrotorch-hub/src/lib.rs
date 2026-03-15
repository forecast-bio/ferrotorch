//! Pretrained model registry, download, and caching for ferrotorch.
//!
//! This crate provides a central hub for discovering, downloading, and caching
//! pretrained model weights. It mirrors the workflow of `torch.hub` and
//! `torchvision.models` with pretrained weight support.
//!
//! # Quick start
//!
//! ```rust,no_run
//! use ferrotorch_hub::{list_models, get_model_info, load_pretrained};
//!
//! // Browse available models.
//! for model in list_models() {
//!     println!("{}: {} ({} params)", model.name, model.description, model.num_parameters);
//! }
//!
//! // Load pretrained weights (requires cached weights on disk).
//! let state_dict = load_pretrained::<f32>("resnet50").unwrap();
//! ```

pub mod cache;
pub mod download;
pub mod registry;

pub use cache::{HubCache, default_cache_dir};
pub use download::{download_weights, load_pretrained};
pub use registry::{ModelInfo, WeightsFormat, get_model_info, list_models};
