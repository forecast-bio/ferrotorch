# ferrotorch-core

Core tensor and autograd engine for ferrotorch -- PyTorch in Rust.

## What it provides

- **`Tensor<T>`** -- N-dimensional tensor with automatic differentiation
- **Autograd** -- reverse-mode automatic differentiation via `backward()`, `no_grad()`, and `autocast`
- **Creation functions** -- `tensor`, `zeros`, `ones`, `rand`, `randn`, `arange`, `linspace`, `eye`, `full`, `scalar`, `zeros_like`, `ones_like`, `rand_like`, `randn_like`, `full_like`
- **Arithmetic & math ops** -- element-wise operations with broadcast support, differentiable `exp`, `log`, `sin`, `cos`, `clamp`
- **Tensor manipulation** -- `permute`, `view`, `contiguous`, `chunk`, `split`
- **Reductions** -- `sum_dim`, `mean_dim` with axis and keepdim support
- **Einsum** -- differentiable Einstein summation (`einsum`, `einsum_differentiable`)
- **FFT** -- `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2` with differentiable variants
- **Quantization** -- `quantize`, `dequantize`, `quantized_matmul`, `QuantizedTensor`
- **Sparse tensors** -- `SparseTensor` for COO-format sparse data
- **Storage** -- `TensorStorage`, `StorageBuffer` for raw memory management
- **DType system** -- `DType`, `Element`, `Float` traits covering `f32`, `f64`, `f16`, `bf16`
- **Device abstraction** -- `Device` enum for CPU/GPU placement

## Quick start

```rust
use ferrotorch_core::{tensor, backward, no_grad, Tensor, Float};

fn main() {
    let x = tensor(&[1.0_f32, 2.0, 3.0]).requires_grad(true);
    let y = (&x * &x).sum(None);
    backward(&y).unwrap();
    println!("grad: {:?}", x.grad().unwrap());
}
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
