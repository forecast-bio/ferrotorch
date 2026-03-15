pub use ferrotorch_core::*;

/// Prelude module — import everything commonly needed.
pub mod prelude {
    pub use ferrotorch_core::*;
    pub use ferrotorch_nn::{Module, Parameter, Linear, Conv2d, Sequential};
    pub use ferrotorch_nn::{ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax};
    pub use ferrotorch_nn::{BatchNorm2d, LayerNorm, Dropout};
    pub use ferrotorch_nn::{CrossEntropyLoss, MSELoss};
    pub use ferrotorch_optim::{Optimizer, Adam, AdamW, Sgd};
}

/// Neural network modules and layers.
pub mod nn {
    pub use ferrotorch_nn::*;
}

/// Optimizers and learning rate schedulers.
pub mod optim {
    pub use ferrotorch_optim::*;
}
