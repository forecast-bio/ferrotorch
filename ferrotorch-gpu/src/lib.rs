//! CUDA GPU backend for ferrotorch.
//!
//! This crate provides device management, memory allocation, and host/device
//! data transfers built on [`cudarc`]. It is the bridge between ferrotorch's
//! CPU tensor world and NVIDIA GPUs.
//!
//! # Feature flags
//!
//! | Feature | Default | Description |
//! |---------|---------|-------------|
//! | `cuda`  | **yes** | Links against the CUDA driver API via cudarc. Disable to compile on machines without a GPU. |
//!
//! # Quick start
//!
//! ```rust,no_run
//! use ferrotorch_gpu::{GpuDevice, cpu_to_gpu, gpu_to_cpu};
//!
//! let device = GpuDevice::new(0).unwrap();
//! let host_data = vec![1.0f32, 2.0, 3.0];
//! let gpu_buf = cpu_to_gpu(&host_data, &device).unwrap();
//! let back = gpu_to_cpu(&gpu_buf, &device).unwrap();
//! assert_eq!(back, host_data);
//! ```

pub mod allocator;
pub mod backend_impl;
pub mod blas;
pub mod buffer;
pub mod conv;
pub mod device;
pub mod flash_attention;
pub mod error;
pub mod graph;
pub mod kernels;
pub mod module_cache;
pub mod memory_guard;
pub mod pool;
pub mod tensor_bridge;
pub mod transfer;

// Re-exports for ergonomic use.
pub use backend_impl::{init_cuda_backend, get_cuda_device, CudaBackendImpl};
pub use allocator::CudaAllocator;
pub use blas::{gpu_matmul_f32, gpu_matmul_f64};
pub use conv::gpu_conv2d_f32;
pub use flash_attention::gpu_flash_attention_f32;
pub use buffer::CudaBuffer;
pub use device::GpuDevice;
pub use error::{GpuError, GpuResult};
pub use kernels::{gpu_add, gpu_mul, gpu_neg, gpu_relu, gpu_sub};
pub use kernels::{gpu_broadcast_add, gpu_broadcast_sub, gpu_broadcast_mul};
pub use kernels::{gpu_softmax, gpu_dropout, gpu_transpose_2d, gpu_permute_0213, gpu_gelu, gpu_layernorm, gpu_slice_write, gpu_slice_read, gpu_small_matmul, gpu_small_bmm, gpu_embed_lookup};
pub use kernels::{gpu_add_into, gpu_mul_into, gpu_scale_into, gpu_gelu_into, gpu_embed_lookup_into, gpu_transpose_2d_into, gpu_permute_0213_into, gpu_softmax_into, gpu_layernorm_into, gpu_slice_read_into, gpu_small_matmul_into};
pub use kernels::{gpu_slice_write_indirect, gpu_causal_mask_indirect};
pub use blas::gpu_bmm_f32;
pub use blas::{gpu_matmul_f32_into, gpu_bmm_f32_into};
pub use memory_guard::{
    MemoryGuard, MemoryGuardBuilder, MemoryGuardedDevice, MemoryHook, MemoryPressureListener,
    MemoryReservation, MemoryStats, MemoryWatchdog, OomPolicy, PressureLevel,
};
pub use tensor_bridge::{cuda, cuda_default, tensor_to_cpu, tensor_to_gpu, GpuFloat, GpuTensor};
pub use pool::{empty_cache, empty_cache_all, empty_cache_for_oom, cached_bytes, cached_bytes_all, round_len};
pub use transfer::{alloc_zeros, alloc_zeros_f32, alloc_zeros_f64, cpu_to_gpu, gpu_to_cpu};
