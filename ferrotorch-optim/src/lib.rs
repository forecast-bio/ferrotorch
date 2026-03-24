pub mod adagrad;
pub mod adam;
pub mod adamw;
pub mod ema;
pub mod grad_accumulator;
pub mod grad_scaler;
pub mod lbfgs;
pub mod muon;
pub mod natural_gradient;
pub mod optimizer;
pub mod rmsprop;
pub mod scheduler;
pub mod sgd;
pub mod swa;

pub use adagrad::{Adagrad, AdagradConfig};
pub use adam::{Adam, AdamConfig};
pub use adamw::{AdamW, AdamWConfig};
pub use ema::ExponentialMovingAverage;
pub use grad_accumulator::GradientAccumulator;
pub use grad_scaler::{GradScaler, GradScalerConfig, GradScalerState};
pub use lbfgs::{Lbfgs, LbfgsConfig, LineSearchFn};
pub use muon::{Muon, MuonConfig};
pub use natural_gradient::{Kfac, KfacConfig};
pub use optimizer::{Optimizer, OptimizerState, ParamGroup};
pub use rmsprop::{Rmsprop, RmspropConfig};
pub use scheduler::{
    CosineAnnealingLR, LrScheduler, LinearWarmup, MetricScheduler, PlateauMode,
    ReduceLROnPlateau, SequentialLr, StepLR, cosine_warmup_scheduler,
};
pub use sgd::{Sgd, SgdConfig};
pub use swa::{AnnealStrategy, AveragedModel, AveragingStrategy, Swalr};
