# ferrotorch-gpu

CUDA GPU backend for ferrotorch — BLAS, cuSPARSE, cuFFT, FlashAttention-2, bf16 mixed-precision, and memory safety.

## What it provides

- **`GpuDevice`** -- CUDA device management and context handling
- **Data transfer** -- `cpu_to_gpu`, `gpu_to_cpu`, `tensor_to_gpu`, `tensor_to_cpu`, `alloc_zeros`
- **`GpuTensor`** -- GPU-resident tensor wrapper with `cuda()` and `cuda_default()` helpers
- **CUDA buffer** -- `CudaBuffer` for raw GPU memory, `CudaAllocator` for memory management
- **BLAS** -- `gpu_matmul_f32`, `gpu_matmul_f64` via cuBLAS, `gpu_bmm_*` batched
- **bf16 mixed-precision** -- `gpu_matmul_bf16`, `gpu_bmm_bf16` via cuBLAS, plus 8 elementwise/reduction/activation kernels (`gpu_add_bf16`, `gpu_mul_bf16`, `gpu_silu_bf16`, `gpu_relu_bf16`, `gpu_softmax_bf16`, `gpu_rmsnorm_bf16`, `gpu_rope_half_bf16`, `gpu_scale_bf16`) and supporting utilities for LLM bf16 pipelines
- **cuSOLVER** -- `gpu_lu_solve`, `gpu_cholesky`, `gpu_eigh`, `gpu_eig`, `gpu_lstsq` via cuSOLVER
- **cuFFT** -- axes-aware FFT via `cufftPlanMany`: `gpu_rfft_r2c_*`, `gpu_irfft_c2r_*`, `gpu_hfft_*`, `gpu_ihfft_*`, `gpu_fftn_axes_c2c_*` (n-d, arbitrary axis selection), `gpu_fftn2d_c2c_*`, `gpu_fftn3d_c2c_*`
- **cuSPARSE** -- `gpu_spmm_csr_*` SpMM, `gpu_sparse_to_dense_csr_*` / `gpu_dense_to_sparse_csr_*` CSR conversions, `gpu_csc_to_dense_*`, `gpu_csr_to_csc_*`, `gpu_coo_to_csr_*`, `gpu_csr_to_coo_*` format conversions
- **cuSPARSELt** -- `gpu_sparse_matmul_24` for 2:4 structured sparse matrix multiply (NVIDIA Ampere and later)
- **FlashAttention-2** -- `gpu_flash_attention_f32` / `gpu_flash_attention_f64` forward-pass PTX kernel
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
