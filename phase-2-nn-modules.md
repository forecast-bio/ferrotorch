---
title: "Phase 2 — Neural Network Modules (ferrotorch-nn)"
tags: [design-doc]
sources: []
contributors: [unknown]
created: 2026-03-15
updated: 2026-03-15
---


## Design Specification

### Summary

The neural network module crate for ferrotorch: a `Module<T>` trait with `Parameter<T>` registration, pre-built layers (Linear, Conv2d, normalization, attention, RNN), loss functions, weight initialization, and the layer-level `GradFn<T>` implementations deferred from Phase 1. Every layer composes over `Tensor<T>` and `GradFn<T>` from ferrotorch-core. Downstream crates (optim, data, vision) depend on this for model construction and training.

### Requirements

- REQ-1: A `Module<T: Float>` trait must define the contract for all neural network layers, with methods for `forward()`, `parameters()`, `parameters_mut()`, `train()`, `eval()`, `named_parameters()`, `state_dict()`, and `load_state_dict()`. The trait must be object-safe enough for `Sequential` to hold `Vec<Box<dyn Module<T>>>`. The trait must require `Send + Sync` to match `Tensor<T>`'s thread-safety guarantees.
- REQ-2: A `Parameter<T: Float>` type must wrap `Tensor<T>` with `requires_grad` always set to `true`. Parameter creation must enforce this invariant. Parameters must be the unit of registration for optimizer consumption — `Module::parameters()` returns references to `Parameter<T>`, not raw `Tensor<T>`.
- REQ-3: Layer-level `GradFn<T>` implementations must exist for all operations deferred from Phase 1: conv (conv1d, conv2d, conv_transpose2d), pool (max_pool2d, avg_pool2d, adaptive_avg_pool2d), norm (batch_norm, layer_norm, group_norm, rms_norm), dropout, embedding lookup, loss backward, and scaled_dot_product_attention. Each must implement the `GradFn<T>` trait from ferrotorch-core (`backward()`, `inputs()`, `name()`) and produce correct vector-Jacobian products.
- REQ-4: The following layers must be implemented as `Module<T>`: `Linear`, `Conv1d`, `Conv2d`, `ConvTranspose2d`, `BatchNorm1d`, `BatchNorm2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `Dropout`, `Dropout2d`, `Embedding`, `MultiheadAttention`, `LSTM`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`. Each must register its learnable tensors as `Parameter<T>` and delegate forward computation to a functional operation that records the appropriate `GradFn<T>` on the output `Tensor<T>` via `Tensor::from_operation()`.
- REQ-5: Activation wrapper modules (`ReLU`, `GELU`, `SiLU`, `Sigmoid`, `Tanh`, `Softmax`, `LeakyReLU`, `ELU`, `Mish`) must implement `Module<T>` with zero parameters, delegating to the activation `GradFn<T>` implementations already in ferrotorch-core's `grad_fns::activation`.
- REQ-6: Container modules `Sequential` and `ModuleList` must implement `Module<T>` and propagate `parameters()`, `train()`, `eval()`, `named_parameters()`, `state_dict()`, and `load_state_dict()` to all contained sub-modules. `Sequential::forward()` must chain sub-module forward calls in order. `ModuleList` must support indexed access but not implement `forward()` itself.
- REQ-7: Loss functions must be implemented as structs (not modules): `CrossEntropyLoss`, `MSELoss`, `BCEWithLogitsLoss`, `HuberLoss`, `CTCLoss`, `TripletMarginLoss`. Each must accept `Tensor<T>` inputs (predictions and targets), return a scalar `Tensor<T>` with a `GradFn<T>` attached, and support configurable reduction (`Mean`, `Sum`, `None`). `CrossEntropyLoss` must support label smoothing. All loss computations must be numerically stable (log-sum-exp for cross-entropy, clamping for BCE).
- REQ-8: Weight initialization functions must operate on `Parameter<T>` in-place: `xavier_uniform`, `xavier_normal`, `kaiming_uniform`, `kaiming_normal`, `uniform`, `normal`, `zeros`, `ones`, `constant`. Kaiming variants must accept a `NonLinearity` enum to select the correct gain. Each layer's `new()` constructor must apply the appropriate default initialization (Kaiming uniform for Linear weight, zeros for bias, etc.) matching PyTorch defaults.
- REQ-9: Layers with train/eval behavioral differences (`BatchNorm1d`, `BatchNorm2d`, `Dropout`, `Dropout2d`) must respect the `training` flag set by `Module::train()` / `Module::eval()`. BatchNorm must use batch statistics during training and running statistics during eval. Dropout must apply the random mask during training and be a no-op during eval.
- REQ-10: All public functions must return `FerrotorchResult<T>`. Invalid configurations (zero `in_features`, negative dropout probability, kernel size larger than input, embedding index out of vocabulary, incompatible state dict keys) must produce descriptive `FerrotorchError` variants, never panics.
- REQ-11: `Conv2d` forward must use the im2col + matmul approach, and `ConvTranspose2d` must use col2im. Convolution backward (`ConvBackward`) must compute correct gradients for weight, bias, and input using the transposed convolution relationship. Padding, stride, dilation, and groups must all be supported.
- REQ-12: `MultiheadAttention` must implement scaled dot-product attention with Q/K/V linear projections and an output projection. It must support an optional attention mask (causal or arbitrary boolean) and dropout on attention weights. The `GradFn<T>` must propagate gradients through the softmax-weighted value computation and all four projections.

### Acceptance Criteria

- [ ] AC-1: `Module::parameters()` on a `Linear` with `in_features=128, out_features=64, bias=true` returns exactly 2 `Parameter<T>` references — one with shape `[64, 128]` (weight) and one with shape `[64]` (bias). Both have `requires_grad() == true`.
- [ ] AC-2: `Linear::forward()` on a `Tensor<f32>` with shape `[32, 128]` produces a `Tensor<f32>` with shape `[32, 64]` that has a `GradFn` attached. Calling `backward()` on a scalar derived from this output populates gradients on the weight and bias parameters, matching PyTorch `nn.Linear` within `rtol=1e-4, atol=1e-6`.
- [ ] AC-3: `Conv2d` with `in_channels=3, out_channels=16, kernel_size=3, padding=1, stride=1` applied to a `[1, 3, 32, 32]` input produces a `[1, 16, 32, 32]` output. The backward pass computes correct gradients for weight, bias, and input, verified against PyTorch within `rtol=1e-4, atol=1e-6` for f32.
- [ ] AC-4: `BatchNorm2d` in training mode (`Module::train()`) normalizes using batch statistics and updates running mean/variance. In eval mode (`Module::eval()`) it normalizes using the stored running statistics. A test trains for 10 batches, switches to eval, and verifies the running statistics match PyTorch.
- [ ] AC-5: `Sequential` containing `[Linear(784, 256), ReLU, Dropout(0.5), Linear(256, 10)]` chains forward calls correctly. `parameters()` returns all 4 parameters (2 weights + 2 biases). `train()` / `eval()` propagates to the Dropout sub-module. `state_dict()` returns a map with keys `"0.weight"`, `"0.bias"`, `"3.weight"`, `"3.bias"`.
- [ ] AC-6: `CrossEntropyLoss` with label smoothing=0.1 applied to logits of shape `[32, 10]` and integer targets of shape `[32]` returns a scalar `Tensor<T>`. Backward produces gradients on the logits matching PyTorch `nn.CrossEntropyLoss(label_smoothing=0.1)` within tolerance. Numerical stability is verified: no NaN or Inf for logits in the range `[-100, 100]`.
- [ ] AC-7: Every layer-level `GradFn<T>` (ConvBackward, MaxPoolBackward, AvgPoolBackward, AdaptiveAvgPoolBackward, BatchNormBackward, LayerNormBackward, GroupNormBackward, RMSNormBackward, DropoutBackward, EmbeddingBackward, CrossEntropyBackward, MSEBackward, BCEWithLogitsBackward, HuberBackward, CTCBackward, TripletMarginBackward, AttentionBackward) passes a numerical gradient check with finite differences (`rtol=1e-4, atol=1e-6` for f32, `rtol=1e-7, atol=1e-10` for f64).
- [ ] AC-8: Weight initialization functions produce distributions matching their specifications: `xavier_uniform` fills values in `[-limit, limit]` where `limit = sqrt(6 / (fan_in + fan_out))`; `kaiming_normal` fills with `N(0, sqrt(2 / fan_in))` for ReLU. Verified statistically over 10,000 elements (mean and variance within 5% of theoretical values).
- [ ] AC-9: `MultiheadAttention` with `embed_dim=512, num_heads=8` applied to query/key/value tensors of shape `[32, 10, 512]` produces output of shape `[32, 10, 512]`. With a causal mask, attention weights are zero above the diagonal. Backward populates gradients on all 4 projection parameters (Q, K, V, output), verified against PyTorch.
- [ ] AC-10: `LSTM` with `input_size=128, hidden_size=256, num_layers=2, bidirectional=true` applied to input `[32, 20, 128]` (batch, seq, feature) produces output `[32, 20, 512]` and hidden state tuple `(h_n, c_n)` with shapes `[4, 32, 256]`. Backward through the output computes correct gradients on all weight matrices.
- [ ] AC-11: `Dropout(p=0.3)` in training mode zeros approximately 30% of elements (verified over 100,000 elements, within 2% of target rate) and scales surviving elements by `1/(1-p)`. In eval mode, the output equals the input exactly. The dropout mask `GradFn<T>` correctly routes gradients only through surviving elements.
- [ ] AC-12: `cargo test -p ferrotorch-nn` passes with 0 failures. Minimum 300 tests covering all modules, all loss functions, all grad_fns, all init functions, train/eval mode switching, state_dict round-trip, error paths (invalid shapes, out-of-bounds embedding indices, mismatched state dict keys), and edge cases (batch size 1, sequence length 1, single-channel input, zero-padding convolution).
- [ ] AC-13: `load_state_dict()` on a `Linear` with a state dict containing the wrong key names returns `Err(FerrotorchError::InvalidArgument { .. })`. A state dict with correct keys but wrong tensor shapes returns `Err(FerrotorchError::ShapeMismatch { .. })`.
- [ ] AC-14: A `Sequential` model can be constructed, trained for one step (forward + backward), and its parameters collected — all from one thread, then sent to another thread for inference. This verifies `Module<T>: Send + Sync`.

### Architecture

### Crate Layout

```
ferrotorch-nn/
├── Cargo.toml
├── src/
│   ├── lib.rs                    # Public re-exports
│   ├── module.rs                 # Module<T> trait, training flag
│   ├── parameter.rs              # Parameter<T> type
│   ├── linear.rs                 # Linear (fully connected)
│   ├── conv.rs                   # Conv1d, Conv2d, ConvTranspose2d
│   ├── norm.rs                   # BatchNorm1d, BatchNorm2d, LayerNorm, GroupNorm, RMSNorm
│   ├── activation.rs             # ReLU, GELU, SiLU, Sigmoid, Tanh, Softmax, LeakyReLU, ELU, Mish
│   ├── dropout.rs                # Dropout, Dropout2d
│   ├── pooling.rs                # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d
│   ├── rnn.rs                    # LSTM
│   ├── attention.rs              # MultiheadAttention
│   ├── embedding.rs              # Embedding
│   ├── container.rs              # Sequential, ModuleList
│   ├── init.rs                   # Weight initialization functions
│   ├── loss.rs                   # CrossEntropyLoss, MSELoss, BCEWithLogitsLoss, HuberLoss, CTCLoss, TripletMarginLoss
│   ├── functional.rs             # Stateless functional API (conv2d, linear, batch_norm, etc.)
│   └── grad_fns/                 # Layer-level GradFn<T> implementations (deferred from core)
│       ├── mod.rs
│       ├── conv.rs               # ConvBackward, ConvTransposeBackward
│       ├── pool.rs               # MaxPoolBackward, AvgPoolBackward, AdaptiveAvgPoolBackward
│       ├── norm.rs               # BatchNormBackward, LayerNormBackward, GroupNormBackward, RMSNormBackward
│       ├── dropout.rs            # DropoutBackward
│       ├── embedding.rs          # EmbeddingBackward
│       ├── loss.rs               # CrossEntropyBackward, MSEBackward, BCEWithLogitsBackward, HuberBackward, CTCBackward, TripletMarginBackward
│       └── attention.rs          # AttentionBackward
└── tests/
    ├── test_linear.rs            # Linear forward, backward, parameter registration
    ├── test_conv.rs              # Conv1d, Conv2d, ConvTranspose2d with various padding/stride/dilation
    ├── test_norm.rs              # BatchNorm train/eval, LayerNorm, GroupNorm, RMSNorm
    ├── test_activation.rs        # All activation modules (zero-param wrappers)
    ├── test_dropout.rs           # Dropout rate, train/eval behavior, gradient routing
    ├── test_pooling.rs           # MaxPool2d, AvgPool2d, AdaptiveAvgPool2d forward + backward
    ├── test_rnn.rs               # LSTM forward, backward, bidirectional, multi-layer
    ├── test_attention.rs         # MultiheadAttention with and without causal mask
    ├── test_embedding.rs         # Embedding lookup, out-of-bounds error, sparse gradient
    ├── test_container.rs         # Sequential chaining, ModuleList indexing, parameter propagation
    ├── test_init.rs              # Statistical verification of all init functions
    ├── test_loss.rs              # All loss functions: numerical correctness, stability, reduction modes
    ├── test_state_dict.rs        # Round-trip save/load, error on key/shape mismatch
    ├── test_grad_fns.rs          # Numerical gradient checks for all layer-level GradFn<T>
    └── test_thread_safety.rs     # Module<T>: Send + Sync across threads
```

### Core Types

**Module<T>** (`module.rs`):

```rust
/// State dict: a map from parameter names to tensors.
pub type StateDict<T> = std::collections::HashMap<String, Tensor<T>>;

/// The trait that all neural network layers implement.
///
/// Requires `Send + Sync` to match `Tensor<T>`'s thread-safety guarantees.
/// Object-safe for the subset needed by `Sequential`: `forward()` and
/// `parameters()` use `&self`, making `dyn Module<T>` viable.
pub trait Module<T: Float>: Send + Sync {
    /// Forward pass. Takes input tensor(s), returns output tensor(s).
    fn forward(&self, input: &Tensor<T>) -> FerrotorchResult<Tensor<T>>;

    /// Iterate over all learnable parameters.
    fn parameters(&self) -> Vec<&Parameter<T>>;

    /// Iterate over all learnable parameters mutably (for in-place init or loading).
    fn parameters_mut(&mut self) -> Vec<&mut Parameter<T>>;

    /// Set training mode. Affects layers like Dropout and BatchNorm.
    fn train(&mut self);

    /// Set evaluation mode.
    fn eval(&mut self);

    /// Whether the module is in training mode.
    fn is_training(&self) -> bool;

    /// Named parameters for state dict serialization.
    /// Returns (name, parameter) pairs with dot-separated hierarchical names.
    fn named_parameters(&self) -> Vec<(String, &Parameter<T>)>;

    /// Export all parameters as a state dict.
    fn state_dict(&self) -> StateDict<T>;

    /// Load parameters from a state dict. Returns an error if keys are
    /// missing/unexpected or tensor shapes do not match.
    fn load_state_dict(&mut self, state: &StateDict<T>) -> FerrotorchResult<()>;
}
```

**Parameter<T>** (`parameter.rs`):

```rust
/// A tensor registered for gradient descent.
///
/// Wraps `Tensor<T>` with the invariant that `requires_grad` is always `true`.
/// Modules store their learnable weights as `Parameter<T>` values, and
/// `Module::parameters()` returns references to them for optimizer consumption.
pub struct Parameter<T: Float = f32> {
    tensor: Tensor<T>,
}

impl<T: Float> Parameter<T> {
    /// Create a new parameter. The tensor's `requires_grad` is forced to `true`.
    pub fn new(tensor: Tensor<T>) -> Self {
        let tensor = if tensor.requires_grad() {
            tensor
        } else {
            tensor.requires_grad_(true)
        };
        Self { tensor }
    }

    /// Borrow the underlying tensor.
    pub fn tensor(&self) -> &Tensor<T> {
        &self.tensor
    }

    /// Mutable access to the underlying tensor (for in-place initialization).
    pub fn tensor_mut(&mut self) -> &mut Tensor<T> {
        &mut self.tensor
    }

    /// Delegate to `Tensor::shape()`.
    pub fn shape(&self) -> &[usize] {
        self.tensor.shape()
    }

    /// Delegate to `Tensor::grad()`.
    pub fn grad(&self) -> FerrotorchResult<Option<Tensor<T>>> {
        self.tensor.grad()
    }

    /// Zero out the accumulated gradient.
    pub fn zero_grad(&self) -> FerrotorchResult<()> {
        self.tensor.set_grad(None)
    }
}
```

### Layer-Level GradFn Implementations

These structs implement `GradFn<T>` (the trait defined in `ferrotorch_core::tensor`) and live in `src/grad_fns/`. The backward engine in ferrotorch-core calls `grad_fn.backward()` via dynamic dispatch on `Arc<dyn GradFn<T>>` — it does not need to know the concrete type. This is the same pattern used by core's arithmetic, reduction, and activation grad_fns.

| File | Structs | VJP Strategy |
|------|---------|-------------|
| `grad_fns/conv.rs` | `ConvBackward`, `ConvTransposeBackward` | Input grad via transposed convolution; weight grad via correlation of input and grad_output; bias grad via reduction over spatial dims |
| `grad_fns/pool.rs` | `MaxPoolBackward`, `AvgPoolBackward`, `AdaptiveAvgPoolBackward` | Max: route grad to argmax indices; Avg: distribute grad equally over window; Adaptive: scale by input/output ratio |
| `grad_fns/norm.rs` | `BatchNormBackward`, `LayerNormBackward`, `GroupNormBackward`, `RMSNormBackward` | Standard normalization backward: grad w.r.t. input, weight (gamma), bias (beta) using saved mean/variance |
| `grad_fns/dropout.rs` | `DropoutBackward` | Multiply grad_output by the saved binary mask and scale by `1/(1-p)` |
| `grad_fns/embedding.rs` | `EmbeddingBackward` | Scatter-add grad_output rows to the corresponding embedding weight indices |
| `grad_fns/loss.rs` | `CrossEntropyBackward`, `MSEBackward`, `BCEWithLogitsBackward`, `HuberBackward`, `CTCBackward`, `TripletMarginBackward` | Each computes the analytic gradient of its loss formula; cross-entropy uses `softmax(logits) - one_hot(targets)` for numerical stability |
| `grad_fns/attention.rs` | `AttentionBackward` | Backprop through softmax-weighted matmul: grad flows to Q, K, V via transposed attention weight computations and through all four linear projections |

### Functional API (`functional.rs`)

Stateless functions that perform the computation and attach the appropriate `GradFn<T>`. Module structs are thin wrappers that hold parameters and call these functions:

```rust
/// Functional linear transformation. Called by Linear::forward().
pub fn linear<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
) -> FerrotorchResult<Tensor<T>>;

/// Functional 2D convolution. Called by Conv2d::forward().
pub fn conv2d<T: Float>(
    input: &Tensor<T>,
    weight: &Tensor<T>,
    bias: Option<&Tensor<T>>,
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    groups: usize,
) -> FerrotorchResult<Tensor<T>>;

/// Functional batch normalization. Called by BatchNorm2d::forward().
pub fn batch_norm<T: Float>(
    input: &Tensor<T>,
    running_mean: Option<&Tensor<T>>,
    running_var: Option<&Tensor<T>>,
    weight: Option<&Tensor<T>>,
    bias: Option<&Tensor<T>>,
    training: bool,
    momentum: f64,
    eps: f64,
) -> FerrotorchResult<Tensor<T>>;

/// Functional dropout. Called by Dropout::forward().
pub fn dropout<T: Float>(
    input: &Tensor<T>,
    p: f64,
    training: bool,
) -> FerrotorchResult<Tensor<T>>;

/// Functional scaled dot-product attention.
pub fn scaled_dot_product_attention<T: Float>(
    query: &Tensor<T>,
    key: &Tensor<T>,
    value: &Tensor<T>,
    attn_mask: Option<&Tensor<T>>,
    dropout_p: f64,
    training: bool,
) -> FerrotorchResult<Tensor<T>>;
```

### Weight Initialization (`init.rs`)

```rust
/// Nonlinearity hint for Kaiming initialization gain calculation.
pub enum NonLinearity {
    Linear,
    ReLU,
    LeakyReLU(f64),
    Tanh,
    Sigmoid,
    GELU,
    SiLU,
}

pub fn xavier_uniform<T: Float>(param: &mut Parameter<T>, gain: f64) -> FerrotorchResult<()>;
pub fn xavier_normal<T: Float>(param: &mut Parameter<T>, gain: f64) -> FerrotorchResult<()>;
pub fn kaiming_uniform<T: Float>(param: &mut Parameter<T>, nonlinearity: NonLinearity) -> FerrotorchResult<()>;
pub fn kaiming_normal<T: Float>(param: &mut Parameter<T>, nonlinearity: NonLinearity) -> FerrotorchResult<()>;
pub fn uniform<T: Float>(param: &mut Parameter<T>, low: f64, high: f64) -> FerrotorchResult<()>;
pub fn normal<T: Float>(param: &mut Parameter<T>, mean: f64, std: f64) -> FerrotorchResult<()>;
pub fn zeros<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()>;
pub fn ones<T: Float>(param: &mut Parameter<T>) -> FerrotorchResult<()>;
pub fn constant<T: Float>(param: &mut Parameter<T>, value: f64) -> FerrotorchResult<()>;
```

### Loss Functions (`loss.rs`)

```rust
/// Reduction mode for loss functions.
pub enum Reduction {
    Mean,
    Sum,
    None,
}

pub struct CrossEntropyLoss {
    reduction: Reduction,
    label_smoothing: f64,
}

pub struct MSELoss {
    reduction: Reduction,
}

pub struct BCEWithLogitsLoss {
    reduction: Reduction,
}

pub struct HuberLoss {
    reduction: Reduction,
    delta: f64,
}

pub struct CTCLoss {
    reduction: Reduction,
    blank: usize,
    zero_infinity: bool,
}

pub struct TripletMarginLoss {
    reduction: Reduction,
    margin: f64,
    p: f64,
}
```

Each loss struct implements a `forward()` method (not the `Module` trait — losses are not modules in PyTorch either) that returns a `Tensor<T>` with the appropriate `GradFn<T>` attached:

```rust
impl CrossEntropyLoss {
    pub fn forward<T: Float>(
        &self,
        input: &Tensor<T>,
        target: &Tensor<T>,
    ) -> FerrotorchResult<Tensor<T>>;
}
```

### Dependencies

| Crate | Version | Purpose |
|-------|---------|---------|
| `ferrotorch-core` | workspace | `Tensor<T>`, `GradFn<T>`, `Float`, `FerrotorchError`, `Tensor::from_operation()`, backward engine |
| `ferray-core` | workspace | `Element` trait (re-exported via ferrotorch-core's `Float`) |
| `ferray-random` | workspace | Random mask generation for Dropout, random init for weight initialization |
| `thiserror` | 2.0 | Error derive macros (re-uses `FerrotorchError` from core, may add nn-specific variants) |
| `rayon` | 1.11 | Parallel im2col/col2im for convolution, parallel LSTM gate computation |

### Convolution Implementation Strategy

`Conv2d` uses the im2col (image-to-column) approach:

1. **Forward**: Unfold input patches into a 2D column matrix, then compute `output = weight_matrix @ col_matrix + bias`. This reduces convolution to a single matmul, delegating to ferray-linalg.
2. **Backward (ConvBackward)**:
   - `grad_input`: Transpose the weight matrix, matmul with grad_output columns, then col2im to fold back to input shape.
   - `grad_weight`: Matmul grad_output columns with im2col(input) transposed.
   - `grad_bias`: Sum grad_output over batch and spatial dimensions.
3. `groups > 1`: Partition input/output channels into groups and apply the above per group.
4. `ConvTranspose2d`: Swaps the forward/backward relationship — forward is a col2im, backward is an im2col.

### LSTM Implementation Strategy

`LSTM` follows PyTorch's implementation:

1. **Forward**: For each time step, compute all four gates (input, forget, cell, output) as a single fused matmul: `gates = W_ih @ x_t + W_hh @ h_{t-1} + bias`. Split and apply sigmoid/tanh activations. Multi-layer: feed the output of layer `l` as input to layer `l+1`. Bidirectional: run a second pass in reverse and concatenate outputs.
2. **Backward**: Backpropagate through time (BPTT). The `GradFn<T>` stores all intermediate gate activations and hidden states from the forward pass. Gradients flow backward through the time steps, accumulating on the weight matrices.

### Test Strategy

1. **Numerical gradient checks**: For every `GradFn<T>` in `grad_fns/`, compare analytic gradient against finite-difference approximation: `(f(x+h) - f(x-h)) / 2h`. Use `h=1e-4` for f32, `h=1e-7` for f64.
2. **PyTorch reference tests**: For each module, compute forward + backward in PyTorch, serialize inputs/outputs/gradients as `.npy` files (via ferray-io), and assert ferrotorch-nn matches within tolerance.
3. **Train/eval behavioral tests**: Verify Dropout and BatchNorm produce different outputs in train vs eval mode.
4. **State dict round-trip**: `state_dict()` followed by `load_state_dict()` on a fresh module produces identical parameters.
5. **Error paths**: Invalid shapes, out-of-bounds indices, mismatched state dict keys, zero-size dimensions.
6. **Statistical init tests**: Generate large tensors, compute sample mean/variance, assert within tolerance of theoretical values.
7. **Thread safety**: Build a model on one thread, send it to another, run forward + backward.

### Out of Scope

- GPU execution of layer operations — `Device::Cuda` is defined but only `Device::Cpu` is functional. GPU kernels for conv, pool, norm, etc. are Phase 6 (ferrotorch-gpu)
- Optimizers (SGD, Adam, etc.) — Phase 3 (ferrotorch-optim)
- Model serialization to disk formats (SafeTensors, ONNX, msgpack) — Phase 3 (ferrotorch-serialize). `state_dict()` / `load_state_dict()` operate on in-memory `HashMap`, not files
- Data loading and batching — Phase 4 (ferrotorch-data)
- Pre-built model architectures (ResNet, ViT, etc.) — Phase 5 (ferrotorch-vision)
- Python bindings for nn modules — late phase (ferrotorch-python)
- GRU and vanilla RNN — LSTM covers the primary RNN use case; GRU/RNN can be added as a follow-up without API changes
- Mixed-precision training (autocast, bf16 modules) — future feature after bf16 tensor support is stable
- Lazy modules (infer shapes on first forward) — defer to avoid complexity; shapes must be specified at construction
- Custom user-defined autograd functions — users can implement `GradFn<T>` directly since the trait is public in core
- Packed sequences for variable-length RNN inputs — defer to a follow-up; fixed-length batches with padding are sufficient for Phase 2

### resolved questions

### Q1: Should loss functions implement the Module trait?
**Decision**: No. Loss functions are structs with a `forward()` method, not `Module` implementors.

PyTorch's `nn.CrossEntropyLoss` technically inherits from `nn.Module`, but this is a historical accident — losses have no learnable parameters, no train/eval distinction, and no state dict. Making them plain structs simplifies the API and avoids confusion about whether loss "parameters" would appear in optimizer parameter groups. The `forward()` method takes predictions and targets as arguments and returns a scalar `Tensor<T>` with a `GradFn<T>`, which integrates with the backward engine identically to module outputs.

### Q2: Where do layer-level GradFn implementations live?
**Decision**: In `ferrotorch-nn/src/grad_fns/`, implementing the `GradFn<T>` trait from ferrotorch-core.

This was decided in Phase 1 (Q3): core keeps math ops (arithmetic, reduction, linalg, activation, shape, indexing, comparison), nn gets layer ops (conv, pool, norm, dropout, embedding, loss, attention). The trait is defined in core; the struct implementations live in nn. Core's backward engine dispatches via `Arc<dyn GradFn<T>>` and never needs to know the concrete type.

### Q3: Object safety of Module trait
**Decision**: `Module<T>` is object-safe. `Sequential` stores `Vec<Box<dyn Module<T>>>`.

The `forward()`, `parameters()`, `train()`, `eval()`, `is_training()`, `named_parameters()`, `state_dict()`, and `load_state_dict()` methods all use `&self` or `&mut self` and return owned types or trait-object-safe collections (`Vec`, `HashMap`). `parameters_mut()` returns `Vec<&mut Parameter<T>>` which is object-safe. This enables dynamic composition: users can build arbitrary layer sequences without compile-time type gymnastics.

### Q4: Parameter storage — newtype vs alias
**Decision**: Newtype struct wrapping `Tensor<T>`, not a type alias.

A newtype enforces the `requires_grad = true` invariant at construction time and provides a distinct type for optimizer APIs to accept. A type alias would allow accidentally passing a non-grad tensor to an optimizer. The newtype cost is a single `.tensor()` call when raw tensor access is needed, which is acceptable.

