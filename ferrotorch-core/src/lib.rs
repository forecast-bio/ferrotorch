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
pub mod linalg;
pub mod ops;
mod inplace;
mod methods;
mod ops_trait;
pub mod quantize;
pub mod shape;
pub mod nested;
pub mod pruning;
pub mod sparse;
pub mod special;
pub mod storage;
pub mod tensor;
pub mod vmap;

// Public re-exports for ergonomic use.
pub use autograd::{autocast, autocast_dtype, autocast_guard, is_autocast_debug, is_autocast_enabled, set_autocast_debug, AutocastCategory, AutocastDtype, backward, backward_with_grad, cond, enable_grad, fixed_point, grad, grad_norm, gradient_penalty, hessian, jacobian, jvp, scan, validate_cond_branches, vjp, is_grad_enabled, no_grad, set_grad_enabled};
pub use flex_attention::flex_attention;
pub use creation::{
    arange, eye, from_slice, from_vec, full, full_like, linspace, ones, ones_like, rand,
    rand_like, randn, randn_like, scalar, tensor, zeros, zeros_like,
};
pub use device::Device;
pub use dtype::{DType, Element, Float};
pub use error::{FerrotorchError, FerrotorchResult};
pub use shape::{broadcast_shapes, normalize_axis};
pub use quantize::{
    dequantize, quantize, quantize_named_tensors, quantized_matmul, QuantDtype, QuantScheme,
    QuantizedTensor, QParams, FakeQuantize, QatModel, QatLayer, prepare_qat,
    Observer, MinMaxObserver, PerChannelMinMaxObserver, HistogramObserver,
    cuda_rng,
};
pub use storage::{StorageBuffer, TensorStorage};
pub use nested::{NestedTensor, nested_scaled_dot_product_attention};
pub use pruning::{apply_2_4_mask, magnitude_prune, sparsity_ratio};
pub use sparse::{CooTensor, CsrTensor, SparseTensor};
pub use tensor::{GradFn, MemoryFormat, Tensor, TensorId};
pub use einops::{rearrange, rearrange_with, repeat, reduce, EinopsReduction};
pub use einsum::{einsum, einsum_differentiable};
pub use fft::{fft, fft2, ifft, ifft2, irfft, rfft};
pub use grad_fns::fft::{fft_differentiable, ifft_differentiable, irfft_differentiable, rfft_differentiable};
pub use grad_fns::cumulative::{cumsum, cumprod, cummax, cummin, logcumsumexp};
pub use ops::cumulative::CumExtremeResult;
pub use grad_fns::reduction::{sum_dim, mean_dim};
pub use grad_fns::shape::cat;
pub use methods::{permute_t, view_t, contiguous_t, chunk_t, split_t};
pub use vmap::{select, stack, vmap, vmap2};
pub use special::{digamma, erf, erfc, erfinv, expm1, lgamma, log1p, sinc, xlogy};
pub use grad_fns::transcendental::{exp, log, sin, cos, clamp};
pub use grad_fns::activation::{sigmoid, tanh, gelu, gelu_with, GeluApproximate};
