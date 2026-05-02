# ferrotorch-core

Core tensor and autograd engine for ferrotorch — PyTorch in Rust.

## What it provides

### Tensor

- **`Tensor<T>`** — N-dimensional tensor parameterized by element type
  (`f32`, `f64`, with `f16`/`bf16` storage support). Reference-counted
  via `Arc<TensorInner>`, with shape, strides, offset, and an optional
  `grad_fn` for autograd.
- **Device abstraction** — `Device::Cpu`, `Device::Cuda(ordinal)`,
  `Device::Mps(ordinal)`, `Device::Xpu(ordinal)`. Move tensors with
  `.to(device)`, `.cuda()`, `.cpu()`, and the pinned-memory variant
  `.to_pinned(device)` for fast CPU→GPU transfer.
- **Storage** — `TensorStorage` over `StorageBuffer::Cpu(Vec<T>)` or
  `StorageBuffer::Gpu(GpuBufferHandle)` with `on_device` and
  `on_device_pinned` constructors.

### Autograd

- **Reverse-mode autodiff** with `backward()`, topological-sort backward
  pass, gradient accumulation, broadcast gradient reduction.
- **`no_grad`**, **`enable_grad`**, **`set_grad_enabled`** for
  fine-grained autograd control.
- **Autocast** — `autocast(dtype, || ...)` mixed-precision regions with
  `current_autocast_snapshot` / `with_autocast_state` helpers (used by
  gradient checkpointing to preserve mixed-precision state across
  recomputation).
- **Gradient checkpointing** — `checkpoint`, `checkpoint_multi` save
  GPU RNG and autocast state, recompute the forward pass during backward.
- **Higher-order autograd**, anomaly mode, hooks (`register_hook`,
  `register_post_accumulate_grad_hook`), gradcheck.
- **Saved tensors**, fixed-point derivatives for DEQ networks,
  forward-mode AD via `DualNumber`.

### Operations

- **Creation** — `zeros`, `ones`, `full`, `tensor`, `from_slice`,
  `from_vec`, `scalar`, `eye`, `arange`, `linspace`, `rand`, `randn`,
  `*_like` variants
- **Arithmetic** (differentiable) — `add`, `sub`, `mul`, `div`, `neg`,
  `pow`, `sqrt`, `abs`, with broadcasting and operator overloading
- **Transcendental** — `exp`, `log`, `log2`, `log10`, `log1p`, `sin`,
  `cos`, `tan`, `asin`, `acos`, `atan`, `sinh`, `cosh`, `tanh`,
  `expm1`, `erf`, `erfc`
- **Reductions** — `sum`, `mean`, `prod`, `sum_dim`, `mean_dim`,
  `nansum`, `nanmean`, `logsumexp`, `logsumexp_dim`
- **Cumulative scans** (GPU-native PTX) — `cumsum`, `cumprod`,
  `cummax`, `cummin`, `logcumsumexp`
- **Linear algebra** — `mm`, `bmm`, `matmul`, `dot`, `cholesky`, `inv`,
  `lstsq`, `qr`, `svd`, `eig`, `solve`, with cuBLAS GPU dispatch
- **Shape ops** (zero-copy views) — `reshape`, `view`, `view_reshape`,
  `permute`, `transpose`, `narrow`, `squeeze`, `unsqueeze`, `flatten`,
  `expand`, `chunk`, `split`, `cat`, `stack`
- **Indexing** — `index_select`, `gather`, `scatter`, `scatter_add`,
  `masked_fill`, `masked_select`, `where_`, `nonzero`
- **Search** — `searchsorted`, `bucketize`, `unique`, `topk`, `meshgrid`
- **Einops** — `rearrange`, `repeat`, `reduce` with readable string
  patterns and zero-copy fast paths for identity permutations
- **Einsum** — differentiable Einstein summation
- **Activations** (differentiable) — `relu`, `gelu`, `silu`, `elu`,
  `mish`, `sigmoid`, `tanh`, `softmax`, `log_softmax`
- **FFT** — `fft`, `ifft`, `rfft`, `irfft`, `fft2`, `ifft2`, `fftn`,
  `ifftn`, `rfftn`, `irfftn`, with cuFFT GPU dispatch
- **Signal processing** — `signal::{stft, istft, hilbert,
  spectrogram, hann/hamming/blackman}` window functions
- **Complex tensors** — interleaved-real complex storage, complex-aware
  arithmetic, FFT, and `eig` non-symmetric eigendecomposition
- **Sparse** — `SparseTensor` (COO format) with sparse arithmetic and
  sparse-grad support for embedding tables
- **Named tensors** — `NamedTensor<T>` with `refine_names`, `align_to`,
  `rename` for advisory dim labels (PyTorch parity)
- **Stride tricks** — `as_strided`, `sliding_window_view`,
  `broadcast_to` zero-copy view manipulation
- **Masked tensors** — `masked_min`, `masked_max`, `masked_mean` with
  fused PTX kernels on GPU
- **Quantization** — INT8/INT4 per-tensor and per-channel
- **Flexible attention** — `flex_attention` with score-mod callbacks,
  composed from `bmm + softmax + cat` for full GPU dispatch
- **Pruning** — magnitude, structured, random pruning utilities
- **Vmap** — vectorized map (in-development)

## Quick start

```rust
use ferrotorch_core::{tensor, Tensor, FerrotorchResult};

fn main() -> FerrotorchResult<()> {
    let x: Tensor<f32> = tensor(&[1.0, 2.0, 3.0])?.requires_grad_(true);
    let y = (&x * &x)?.sum_t()?;
    y.backward()?;
    println!("grad of sum(x*x) wrt x: {:?}", x.grad()?);
    Ok(())
}
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
