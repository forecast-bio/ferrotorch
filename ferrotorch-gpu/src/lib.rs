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
pub mod blas;
pub mod buffer;
pub mod conv;
pub mod device;
pub mod error;
pub mod kernels;
pub mod memory_guard;
pub mod tensor_bridge;
pub mod transfer;

// Re-exports for ergonomic use.
pub use allocator::CudaAllocator;
pub use blas::{gpu_matmul_f32, gpu_matmul_f64};
pub use conv::gpu_conv2d_f32;
pub use buffer::CudaBuffer;
pub use device::GpuDevice;
pub use error::{GpuError, GpuResult};
pub use kernels::{gpu_add, gpu_mul, gpu_neg, gpu_relu, gpu_sub};
pub use memory_guard::{
    MemoryGuard, MemoryGuardBuilder, MemoryGuardedDevice, MemoryReservation, MemoryStats,
    MemoryWatchdog, OomPolicy,
};
pub use tensor_bridge::{cuda, cuda_default, tensor_to_cpu, tensor_to_gpu, GpuFloat, GpuTensor};
pub use transfer::{alloc_zeros, cpu_to_gpu, gpu_to_cpu};
