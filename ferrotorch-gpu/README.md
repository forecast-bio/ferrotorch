# ferrotorch-gpu

CUDA GPU backend for ferrotorch.

## What it provides

- **`GpuDevice`** -- CUDA device management and context handling
- **Data transfer** -- `cpu_to_gpu`, `gpu_to_cpu`, `tensor_to_gpu`, `tensor_to_cpu`, `alloc_zeros`
- **`GpuTensor`** -- GPU-resident tensor wrapper with `cuda()` and `cuda_default()` helpers
- **CUDA buffer** -- `CudaBuffer` for raw GPU memory, `CudaAllocator` for memory management
- **BLAS** -- `gpu_matmul_f32`, `gpu_matmul_f64` via cuBLAS, `gpu_bmm_*` batched
- **cuSOLVER** -- `gpu_lu_solve`, `gpu_cholesky`, `gpu_eigh`, `gpu_eig`, `gpu_lstsq` via cuSOLVER
- **cuFFT** -- `gpu_fft_*`, `gpu_rfft_*`, `gpu_ifft_*` interleaved-complex FFTs
- **Convolution** -- `gpu_conv2d_f32`
- **Element-wise kernels** -- `gpu_add`, `gpu_sub`, `gpu_mul`, `gpu_neg`, `gpu_relu`, plus 50+ PTX kernels for reductions, scans, masked ops, scatter/gather, strided copies
- **Graph capture** -- `GpuGraphPool` for CUDA-graph stream capture and replay
- **Memory guard** -- `MemoryGuard`, `MemoryWatchdog`, `MemoryStats`, `OomPolicy` for GPU memory management and pressure monitoring

## Feature flags

| Feature | Default | Description |
|---------|---------|-------------|
| `cuda`  | **yes** | Links against CUDA driver API via cudarc |

## Quick start

```rust
use ferrotorch_gpu::{GpuDevice, cpu_to_gpu, gpu_to_cpu};

let device = GpuDevice::new(0)?;
let host_data = vec![1.0f32, 2.0, 3.0];
let gpu_buf = cpu_to_gpu(&host_data, &device)?;
let back = gpu_to_cpu(&gpu_buf, &device)?;
assert_eq!(back, host_data);
```

## Part of ferrotorch

This crate is one component of the [ferrotorch](https://github.com/dollspace-gay/ferrotorch) workspace.
See the workspace README for full documentation.

## License

MIT OR Apache-2.0
