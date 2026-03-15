pub mod autograd;
pub mod creation;
pub mod device;
mod display;
pub mod dtype;
pub mod einops;
pub mod einsum;
pub mod error;
pub mod fft;
pub mod gpu_dispatch;
pub mod grad_fns;
pub mod linalg;
pub mod ops;
mod inplace;
mod methods;
mod ops_trait;
pub mod quantize;
pub mod shape;
pub mod sparse;
pub mod special;
pub mod storage;
pub mod tensor;
pub mod vmap;

// Public re-exports for ergonomic use.
pub use autograd::{autocast, autocast_dtype, is_autocast_enabled, AutocastDtype, backward, fixed_point, grad, grad_norm, gradient_penalty, hessian, jacobian, jvp, vjp, is_grad_enabled, no_grad};
pub use creation::{
    arange, eye, from_slice, from_vec, full, linspace, ones, rand, randn, scalar, tensor, zeros,
};
pub use device::Device;
pub use dtype::{DType, Element, Float};
pub use error::{FerrotorchError, FerrotorchResult};
pub use shape::{broadcast_shapes, normalize_axis};
pub use quantize::{
    dequantize, quantize, quantize_named_tensors, quantized_matmul, QuantDtype, QuantScheme,
    QuantizedTensor,
};
pub use storage::{StorageBuffer, TensorStorage};
pub use sparse::SparseTensor;
pub use tensor::{GradFn, Tensor, TensorId};
pub use einops::{rearrange, rearrange_with, repeat, reduce, EinopsReduction};
pub use einsum::{einsum, einsum_differentiable};
pub use fft::{fft, fft2, ifft, ifft2, irfft, rfft};
pub use grad_fns::fft::{fft_differentiable, ifft_differentiable, irfft_differentiable, rfft_differentiable};
pub use vmap::{select, stack, vmap, vmap2};
pub use special::{digamma, erf, erfc, erfinv, expm1, lgamma, log1p, sinc, xlogy};
