pub mod autograd;
pub mod cpu_pool;
pub mod creation;
pub mod device;
mod display;
pub mod dtype;
pub mod einops;
pub mod einsum;
pub mod error;
pub mod fft;
pub mod flex_attention;
pub mod gpu_dispatch;
pub mod grad_fns;
mod inplace;
pub mod linalg;
mod methods;
pub mod nested;
pub mod meta_propagate;
pub mod ops;
mod ops_trait;
pub mod profiler_hook;
pub mod pruning;
pub mod quantize;
pub mod shape;
pub mod sparse;
pub mod special;
pub mod storage;
pub mod tensor;
pub mod vmap;

// Public re-exports for ergonomic use.
pub use autograd::anomaly::{
    AnomalyMode, ForwardBacktrace, check_gradient_anomaly, detect_anomaly,
};
pub use autograd::hooks::HookHandle;
pub use autograd::{
    AutocastCategory, AutocastDtype, autocast, autocast_dtype, autocast_guard, backward,
    backward_with_grad, cond, enable_grad, fixed_point, grad, grad_norm, gradient_penalty, hessian,
    is_autocast_debug, is_autocast_enabled, is_grad_enabled, jacobian, jvp, no_grad, scan,
    set_autocast_debug, set_grad_enabled, validate_cond_branches, vjp,
};
pub use autograd::{
    DualTensor, dual_add, dual_cos, dual_div, dual_exp, dual_log, dual_matmul, dual_mul, dual_neg,
    dual_relu, dual_sigmoid, dual_sin, dual_sub, dual_tanh, jacfwd, jvp_exact,
};
pub use creation::{
    arange, eye, from_slice, from_vec, full, full_like, linspace, ones, ones_like, rand, rand_like,
    randn, randn_like, scalar, tensor, zeros, zeros_like,
};
pub use device::Device;
pub use dtype::{DType, Element, Float};
pub use einops::{EinopsReduction, rearrange, rearrange_with, reduce, repeat};
pub use einsum::{einsum, einsum_differentiable};
pub use error::{FerrotorchError, FerrotorchResult};
pub use fft::{fft, fft2, ifft, ifft2, irfft, rfft};
pub use flex_attention::flex_attention;
pub use grad_fns::activation::{GeluApproximate, gelu, gelu_with, sigmoid, tanh};
pub use grad_fns::cumulative::{cummax, cummin, cumprod, cumsum, logcumsumexp};
pub use grad_fns::fft::{
    fft_differentiable, ifft_differentiable, irfft_differentiable, rfft_differentiable,
};
pub use grad_fns::reduction::{mean_dim, sum_dim};
pub use grad_fns::shape::cat;
pub use grad_fns::transcendental::{clamp, cos, exp, log, sin};
pub use methods::{chunk_t, contiguous_t, permute_t, split_t, view_t};
pub use nested::{NestedTensor, PackedNestedTensor, nested_scaled_dot_product_attention};
pub use ops::cumulative::CumExtremeResult;
pub use ops::indexing::{gather, scatter, scatter_add, where_cond};
pub use ops::search::{bucketize, histc, meshgrid, searchsorted, topk, unique, unique_consecutive};
pub use ops::tensor_ops::{cdist, diag, diagflat, roll, tril, triu};
pub use pruning::{apply_2_4_mask, magnitude_prune, sparsity_ratio};
pub use quantize::{
    FakeQuantize, HistogramObserver, MinMaxObserver, Observer, PerChannelMinMaxObserver, QParams,
    QatLayer, QatModel, QuantDtype, QuantScheme, QuantizedTensor, cuda_rng, dequantize,
    prepare_qat, quantize, quantize_named_tensors, quantized_matmul,
};
pub use shape::{broadcast_shapes, normalize_axis};
pub use sparse::{CooTensor, CsrTensor, SparseTensor};
pub use special::{digamma, erf, erfc, erfinv, expm1, lgamma, log1p, sinc, xlogy};
pub use storage::{StorageBuffer, TensorStorage};
pub use tensor::{GradFn, MemoryFormat, Tensor, TensorId};
pub use vmap::{select, stack, vmap, vmap2};
