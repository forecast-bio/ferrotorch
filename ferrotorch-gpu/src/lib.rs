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
#[cfg(feature = "cuda")]
pub mod bf16;
pub mod blas;
pub mod buffer;
pub mod conv;
#[cfg(feature = "cuda")]
pub mod cufft;
pub mod cusolver;
pub mod device;
pub mod error;
pub mod flash_attention;
pub mod graph;
pub mod kernels;
pub mod memory_guard;
pub mod module_cache;
pub mod pool;
pub mod rng;
pub mod stream;
pub mod tensor_bridge;
pub mod transfer;

// Re-exports for ergonomic use.
pub use allocator::CudaAllocator;
pub use backend_impl::{CudaBackendImpl, get_cuda_device, init_cuda_backend};
#[cfg(feature = "cuda")]
pub use bf16::{
    gpu_add_bf16, gpu_block_reduce_max_abs_bf16, gpu_causal_mask_bf16, gpu_embedding_gather_bf16,
    gpu_embedding_gather_bf16_to_f32, gpu_fatrelu_bf16, gpu_mul_bf16, gpu_relu_bf16,
    gpu_repeat_kv_bf16, gpu_rmsnorm_bf16, gpu_rope_half_bf16, gpu_scale_bf16, gpu_silu_bf16,
    gpu_softmax_bf16, gpu_transpose_from_heads_bf16, gpu_transpose_to_heads_bf16,
};
pub use blas::gpu_bmm_f32;
pub use blas::{gpu_bmm_f32_into, gpu_matmul_f32_into};
#[cfg(feature = "cuda")]
pub use blas::{
    gpu_matmul_bf16_bf16, gpu_matmul_bf16_bf16_nt, gpu_matmul_bf16_bf16_strided_batched,
    gpu_matmul_bf16_bf16_strided_batched_nt,
};
pub use blas::{gpu_matmul_f32, gpu_matmul_f64};
pub use buffer::CudaBuffer;
pub use conv::gpu_conv2d_f32;
pub use device::GpuDevice;
pub use error::{GpuError, GpuResult};
pub use flash_attention::gpu_flash_attention_f32;
pub use graph::{
    CaptureMode, CapturePool, CaptureStatus, CapturedGraph, GraphPoolHandle, begin_capture,
    capture_pool_for_handle, end_capture, end_capture_with_pool, graph_pool_handle,
    make_graphed_callable, release_graph_pool_handle,
};
#[cfg(feature = "cuda")]
pub use graph::{
    GraphCaptureGuard, begin_capture_with_mode, begin_capture_with_pool, capture_status,
    is_stream_capturing,
};
pub use kernels::{gpu_add, gpu_mul, gpu_neg, gpu_relu, gpu_sub};
pub use kernels::{
    gpu_add_into, gpu_embed_lookup_into, gpu_gelu_into, gpu_layernorm_into, gpu_mul_into,
    gpu_permute_0213_into, gpu_scale_into, gpu_slice_read_into, gpu_small_matmul_into,
    gpu_softmax_into, gpu_transpose_2d_into,
};
pub use kernels::{gpu_broadcast_add, gpu_broadcast_mul, gpu_broadcast_sub};
pub use kernels::{gpu_causal_mask_indirect, gpu_slice_write_indirect};
pub use kernels::{
    gpu_dropout, gpu_embed_lookup, gpu_gelu, gpu_layernorm, gpu_permute_0213, gpu_slice_read,
    gpu_slice_write, gpu_small_bmm, gpu_small_matmul, gpu_softmax, gpu_transpose_2d,
};
pub use memory_guard::{
    MemoryGuard, MemoryGuardBuilder, MemoryGuardedDevice, MemoryHook, MemoryPressureListener,
    MemoryReservation, MemoryStats, MemoryWatchdog, OomPolicy, PressureLevel,
};
pub use pool::{cached_bytes, empty_cache, empty_cache_all, round_len};
pub use rng::{CudaRngManager, PhiloxGenerator, PhiloxState, cuda_rng_manager, fork_rng, join_rng};
pub use tensor_bridge::{GpuFloat, GpuTensor, cuda, cuda_default, tensor_to_cpu, tensor_to_gpu};
pub use transfer::{alloc_zeros, alloc_zeros_f32, alloc_zeros_f64, cpu_to_gpu, gpu_to_cpu};
