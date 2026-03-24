//! GPU-aware collective communication operations.
//!
//! These functions extend the CPU-only collectives in [`crate::collective`]
//! to work with [`GpuTensor`]s. The strategy is simple and portable:
//!
//! 1. Copy the GPU tensor to CPU.
//! 2. Perform the collective via the existing TCP/simulated backend.
//! 3. Copy the result back to GPU.
//!
//! This avoids any dependency on NCCL or vendor-specific GPU communication
//! libraries. It is slower than direct GPU-to-GPU transfers but works on
//! every system that has CUDA and a network.
//!
//! # Feature gate
//!
//! This module is only compiled when the `gpu` feature is enabled:
//!
//! ```toml
//! ferrotorch-distributed = { version = "0.1", features = ["gpu"] }
//! ```

use ferrotorch_core::FerrotorchResult;
use ferrotorch_gpu::{GpuFloat, GpuTensor, tensor_to_cpu, tensor_to_gpu};

use crate::backend::Backend;
use crate::collective::{ReduceOp, allreduce, broadcast};

// ---------------------------------------------------------------------------
// GPU allreduce
// ---------------------------------------------------------------------------

/// Allreduce a [`GpuTensor`] across all ranks via a CPU round-trip.
///
/// Each rank provides its local GPU tensor. The result is a new
/// [`GpuTensor`] on the same device, whose values are the element-wise
/// reduction of all inputs.
///
/// # Algorithm
///
/// 1. **GPU -> CPU**: [`tensor_to_cpu`] copies device memory to a host
///    [`Tensor<T>`].
/// 2. **CPU allreduce**: The existing [`allreduce`] function reduces
///    across all ranks via the backend (TCP or simulated channels).
/// 3. **CPU -> GPU**: [`tensor_to_gpu`] copies the reduced result back
///    to the original device.
///
/// # Errors
///
/// - GPU/CPU transfer errors (CUDA driver failures, non-contiguous tensor).
/// - Backend communication errors (network, channel closed, etc.).
pub fn gpu_allreduce<T: GpuFloat>(
    tensor: &GpuTensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
) -> FerrotorchResult<GpuTensor<T>> {
    // 1. GPU -> CPU
    let cpu_tensor = tensor_to_cpu(tensor)?;

    // 2. CPU allreduce (existing star-topology implementation)
    let reduced = allreduce(&cpu_tensor, backend, op)?;

    // 3. CPU -> GPU (back to the same device the input lived on)
    let gpu_result = tensor_to_gpu(&reduced, tensor.device()).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("gpu_allreduce: CPU->GPU transfer failed: {e}"),
        }
    })?;

    Ok(gpu_result)
}

// ---------------------------------------------------------------------------
// GPU broadcast
// ---------------------------------------------------------------------------

/// Broadcast a [`GpuTensor`] from `root` to all other ranks via a CPU
/// round-trip.
///
/// The `root` rank's tensor data is sent to every other rank. All ranks
/// return a [`GpuTensor`] on their respective device containing the
/// root's data.
///
/// # Algorithm
///
/// 1. **GPU -> CPU**: Each rank copies its local GPU tensor to the host.
/// 2. **CPU broadcast**: The existing [`broadcast`] function distributes
///    the root's data to all ranks via the backend.
/// 3. **CPU -> GPU**: Each rank copies the broadcast result back to GPU.
///
/// # Errors
///
/// - GPU/CPU transfer errors.
/// - Backend communication errors.
/// - [`crate::error::DistributedError::InvalidRank`] if `root >= world_size`.
pub fn gpu_broadcast<T: GpuFloat>(
    tensor: &GpuTensor<T>,
    backend: &dyn Backend,
    root: usize,
) -> FerrotorchResult<GpuTensor<T>> {
    // 1. GPU -> CPU
    let cpu_tensor = tensor_to_cpu(tensor)?;

    // 2. CPU broadcast (existing star-topology implementation)
    let bcast = broadcast(&cpu_tensor, backend, root)?;

    // 3. CPU -> GPU
    let gpu_result = tensor_to_gpu(&bcast, tensor.device()).map_err(|e| {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: format!("gpu_broadcast: CPU->GPU transfer failed: {e}"),
        }
    })?;

    Ok(gpu_result)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;
    use crate::backend::SimulatedBackend;
    use ferrotorch_gpu::{GpuDevice, tensor_to_gpu as t2g};
    use std::sync::Arc;
    use std::thread;

    /// Helper: create a GpuTensor<f32> from a flat slice on device 0.
    fn gpu_from_slice(data: &[f32], shape: &[usize]) -> GpuTensor<f32> {
        let cpu = ferrotorch_core::from_slice(data, shape).unwrap();
        let device = GpuDevice::new(0).unwrap();
        t2g(&cpu, &device).unwrap()
    }

    #[test]
    fn test_gpu_allreduce_sum_2_ranks() {
        // Rank 0: [1.0, 2.0, 3.0], Rank 1: [4.0, 5.0, 6.0]
        // Sum: [5.0, 7.0, 9.0]
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let data: Vec<f32> = if rank == 0 {
                        vec![1.0, 2.0, 3.0]
                    } else {
                        vec![4.0, 5.0, 6.0]
                    };
                    let gt = gpu_from_slice(&data, &[3]);
                    let result = gpu_allreduce(&gt, b.as_ref(), ReduceOp::Sum).unwrap();

                    // Verify by copying back to CPU.
                    let cpu = result.cpu().unwrap();
                    let out = cpu.data().unwrap();
                    assert_eq!(out.len(), 3);
                    assert!(
                        (out[0] - 5.0).abs() < 1e-6,
                        "rank {rank}: expected 5.0, got {}",
                        out[0]
                    );
                    assert!(
                        (out[1] - 7.0).abs() < 1e-6,
                        "rank {rank}: expected 7.0, got {}",
                        out[1]
                    );
                    assert!(
                        (out[2] - 9.0).abs() < 1e-6,
                        "rank {rank}: expected 9.0, got {}",
                        out[2]
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_gpu_allreduce_mean_2_ranks() {
        // Rank 0: [2.0, 4.0], Rank 1: [6.0, 8.0]
        // Mean: [4.0, 6.0]
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let data: Vec<f32> = if rank == 0 {
                        vec![2.0, 4.0]
                    } else {
                        vec![6.0, 8.0]
                    };
                    let gt = gpu_from_slice(&data, &[2]);
                    let result = gpu_allreduce(&gt, b.as_ref(), ReduceOp::Mean).unwrap();

                    let cpu = result.cpu().unwrap();
                    let out = cpu.data().unwrap();
                    assert!(
                        (out[0] - 4.0).abs() < 1e-6,
                        "rank {rank}: expected 4.0, got {}",
                        out[0]
                    );
                    assert!(
                        (out[1] - 6.0).abs() < 1e-6,
                        "rank {rank}: expected 6.0, got {}",
                        out[1]
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_gpu_broadcast_from_rank_0() {
        // Rank 0: [42.0, 99.0], Rank 1: [0.0, 0.0]
        // After broadcast(root=0): both have [42.0, 99.0]
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let data: Vec<f32> = if rank == 0 {
                        vec![42.0, 99.0]
                    } else {
                        vec![0.0, 0.0]
                    };
                    let gt = gpu_from_slice(&data, &[2]);
                    let result = gpu_broadcast(&gt, b.as_ref(), 0).unwrap();

                    let cpu = result.cpu().unwrap();
                    let out = cpu.data().unwrap();
                    assert!(
                        (out[0] - 42.0).abs() < 1e-6,
                        "rank {rank}: expected 42.0, got {}",
                        out[0]
                    );
                    assert!(
                        (out[1] - 99.0).abs() < 1e-6,
                        "rank {rank}: expected 99.0, got {}",
                        out[1]
                    );
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_gpu_allreduce_single_rank() {
        // Single rank: allreduce should return the input unchanged.
        let group = SimulatedBackend::create_group(1).unwrap();
        let gt = gpu_from_slice(&[1.0, 2.0, 3.0], &[3]);
        let result = gpu_allreduce(&gt, &group[0], ReduceOp::Sum).unwrap();

        let cpu = result.cpu().unwrap();
        let out = cpu.data().unwrap();
        assert_eq!(out, &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_gpu_allreduce_preserves_shape() {
        // Verify shape [2, 3] is preserved through the round-trip.
        let group = SimulatedBackend::create_group(1).unwrap();
        let gt = gpu_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
        let result = gpu_allreduce(&gt, &group[0], ReduceOp::Sum).unwrap();

        assert_eq!(result.shape(), &[2, 3]);
    }

    #[test]
    fn test_gpu_broadcast_invalid_root() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let gt = gpu_from_slice(&[1.0, 2.0], &[2]);
        let result = gpu_broadcast(&gt, &group[0], 5);
        assert!(result.is_err());
    }
}
