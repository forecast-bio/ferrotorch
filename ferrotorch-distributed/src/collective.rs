//! Collective communication operations.
//!
//! These functions coordinate tensors across all ranks in a process group
//! via a [`Backend`]. The current implementation uses a star topology
//! (gather at rank 0, reduce, scatter) which is correct but not optimal.
//! Ring-allreduce and tree-reduce can be layered in later without changing
//! the public API.

use std::time::Duration;

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Float, FerrotorchResult, Tensor};

use crate::backend::Backend;
use crate::error::DistributedError;

/// Default timeout for collective recv operations (60 seconds).
pub const DEFAULT_COLLECTIVE_TIMEOUT: Duration = Duration::from_secs(60);

// ---------------------------------------------------------------------------
// Reduce operations
// ---------------------------------------------------------------------------

/// Reduction operation for collective communication.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum across all ranks.
    Sum,
    /// Element-wise mean across all ranks (sum / world_size).
    Mean,
}

// ---------------------------------------------------------------------------
// Allreduce
// ---------------------------------------------------------------------------

/// Reduce a tensor across all ranks and distribute the result to every rank.
///
/// Each rank provides its local tensor. After allreduce, every rank holds
/// the same tensor whose values are the element-wise reduction of all
/// inputs.
///
/// Uses [`DEFAULT_COLLECTIVE_TIMEOUT`] (60s) for recv operations. Use
/// [`allreduce_with_timeout`] to override.
///
/// # Algorithm (star topology)
///
/// 1. Non-zero ranks send their data to rank 0.
/// 2. Rank 0 reduces all received data with its own.
/// 3. Rank 0 broadcasts the result back to all other ranks.
pub fn allreduce<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
) -> FerrotorchResult<Tensor<T>> {
    allreduce_with_timeout(tensor, backend, op, DEFAULT_COLLECTIVE_TIMEOUT)
}

/// Like [`allreduce`] but with a configurable timeout for recv operations.
///
/// Returns [`DistributedError::Timeout`] if any recv does not complete
/// within `timeout`.
pub fn allreduce_with_timeout<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
    timeout: Duration,
) -> FerrotorchResult<Tensor<T>> {
    let rank = backend.rank();
    let world_size = backend.world_size();
    let numel = tensor.numel();
    let byte_len = numel * std::mem::size_of::<T>();
    let shape = tensor.shape().to_vec();

    if world_size == 1 {
        // Single rank: nothing to reduce.
        return match op {
            ReduceOp::Sum => Ok(tensor.clone()),
            ReduceOp::Mean => Ok(tensor.clone()),
        };
    }

    // Zero-size tensor: nothing to communicate.
    if byte_len == 0 {
        return Ok(tensor.clone());
    }

    if rank == 0 {
        // Start with our own data.
        let local = tensor.data_vec()?;
        let mut accum: Vec<T> = local;

        // Receive from every other rank and accumulate.
        let mut recv_buf = vec![0u8; byte_len];
        for src in 1..world_size {
            backend.recv_timeout(&mut recv_buf, src, timeout)?;
            let peer_data = bytes_to_floats::<T>(&recv_buf);
            for (a, &b) in accum.iter_mut().zip(peer_data.iter()) {
                *a = *a + b;
            }
        }

        // Apply mean if requested.
        if op == ReduceOp::Mean {
            let divisor = T::from(world_size).unwrap();
            for a in &mut accum {
                *a = *a / divisor;
            }
        }

        // Broadcast result to all other ranks.
        let result_bytes = floats_to_bytes(&accum);
        for dst in 1..world_size {
            backend.send(&result_bytes, dst)?;
        }

        Tensor::from_storage(TensorStorage::cpu(accum), shape, false)
    } else {
        // Send our data to rank 0.
        let local = tensor.data_vec()?;
        let send_bytes = floats_to_bytes(&local);
        backend.send(&send_bytes, 0)?;

        // Receive reduced result from rank 0.
        let mut recv_buf = vec![0u8; byte_len];
        backend.recv_timeout(&mut recv_buf, 0, timeout)?;
        let result = bytes_to_floats::<T>(&recv_buf);

        Tensor::from_storage(TensorStorage::cpu(result), shape, false)
    }
}

// ---------------------------------------------------------------------------
// Broadcast
// ---------------------------------------------------------------------------

/// Broadcast a tensor from `root` to all other ranks.
///
/// The `root` rank's tensor data is sent to every other rank. All ranks
/// return a tensor with the root's data.
pub fn broadcast<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
    root: usize,
) -> FerrotorchResult<Tensor<T>> {
    let rank = backend.rank();
    let world_size = backend.world_size();
    let numel = tensor.numel();
    let byte_len = numel * std::mem::size_of::<T>();
    let shape = tensor.shape().to_vec();

    if root >= world_size {
        return Err(DistributedError::InvalidRank {
            rank: root,
            world_size,
        }
        .into());
    }

    if world_size == 1 {
        return Ok(tensor.clone());
    }

    if rank == root {
        let local = tensor.data_vec()?;
        let send_bytes = floats_to_bytes(&local);
        for dst in 0..world_size {
            if dst != root {
                backend.send(&send_bytes, dst)?;
            }
        }
        Ok(tensor.clone())
    } else {
        let mut recv_buf = vec![0u8; byte_len];
        backend.recv_timeout(&mut recv_buf, root, DEFAULT_COLLECTIVE_TIMEOUT)?;
        let result = bytes_to_floats::<T>(&recv_buf);
        Tensor::from_storage(TensorStorage::cpu(result), shape, false)
    }
}

// ---------------------------------------------------------------------------
// Barrier
// ---------------------------------------------------------------------------

/// Block until all ranks have reached this point.
pub fn barrier(backend: &dyn Backend) -> FerrotorchResult<()> {
    backend.barrier()
}

// ---------------------------------------------------------------------------
// Byte serialization helpers
// ---------------------------------------------------------------------------

/// Reinterpret a float slice as raw bytes, copying into a new `Vec<u8>`.
fn floats_to_bytes<T: Float>(data: &[T]) -> Vec<u8> {
    let byte_len = data.len() * std::mem::size_of::<T>();
    let ptr = data.as_ptr() as *const u8;
    // SAFETY: T is f32 or f64, both are POD types with no padding.
    // The slice is valid for byte_len bytes.
    unsafe { std::slice::from_raw_parts(ptr, byte_len) }.to_vec()
}

/// Reinterpret raw bytes back into a Vec of floats.
fn bytes_to_floats<T: Float>(bytes: &[u8]) -> Vec<T> {
    let t_size = std::mem::size_of::<T>();
    assert!(
        bytes.len() % t_size == 0,
        "byte buffer length {} is not a multiple of type size {}",
        bytes.len(),
        t_size,
    );
    let numel = bytes.len() / t_size;
    let mut result = Vec::with_capacity(numel);

    for i in 0..numel {
        let offset = i * t_size;
        // SAFETY: T is a POD float type (f32 or f64). We use
        // `copy_nonoverlapping` to avoid alignment requirements — the
        // source byte buffer may not be aligned for T.
        let mut val = std::mem::MaybeUninit::<T>::uninit();
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr().add(offset),
                val.as_mut_ptr() as *mut u8,
                t_size,
            );
            result.push(val.assume_init());
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SimulatedBackend;
    use std::sync::Arc;
    use std::thread;

    #[test]
    fn test_allreduce_sum_4_ranks() {
        // Each rank has [rank, rank, rank].
        // Sum should be [0+1+2+3, 0+1+2+3, 0+1+2+3] = [6, 6, 6].
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let val = rank as f32;
                    let t = ferrotorch_core::from_slice(&[val, val, val], &[3]).unwrap();
                    allreduce(&t, b.as_ref(), ReduceOp::Sum).unwrap()
                })
            })
            .collect();

        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            assert_eq!(data.len(), 3);
            for &v in data {
                assert!(
                    (v - 6.0).abs() < 1e-6,
                    "expected 6.0, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_allreduce_mean_4_ranks() {
        // Each rank has [rank, rank, rank].
        // Mean should be [6/4, 6/4, 6/4] = [1.5, 1.5, 1.5].
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let val = rank as f32;
                    let t = ferrotorch_core::from_slice(&[val, val, val], &[3]).unwrap();
                    allreduce(&t, b.as_ref(), ReduceOp::Mean).unwrap()
                })
            })
            .collect();

        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            for &v in data {
                assert!(
                    (v - 1.5).abs() < 1e-6,
                    "expected 1.5, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_broadcast_from_rank_0() {
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    // All ranks create a tensor, but only rank 0's data matters.
                    let val = if rank == 0 { 42.0f32 } else { 0.0f32 };
                    let t = ferrotorch_core::from_slice(&[val, val], &[2]).unwrap();
                    broadcast(&t, b.as_ref(), 0).unwrap()
                })
            })
            .collect();

        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            assert_eq!(data.len(), 2);
            for &v in data {
                assert!(
                    (v - 42.0).abs() < 1e-6,
                    "expected 42.0, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_barrier_completes() {
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .into_iter()
            .map(|b| {
                thread::spawn(move || {
                    barrier(b.as_ref()).unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_broadcast_invalid_root() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let t = ferrotorch_core::zeros::<f32>(&[3]).unwrap();
        let result = broadcast(&t, &group[0], 5);
        assert!(result.is_err());
    }

    #[test]
    fn test_allreduce_single_rank() {
        let group = SimulatedBackend::create_group(1).unwrap();
        let t = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let result = allreduce(&t, &group[0], ReduceOp::Sum).unwrap();
        assert_eq!(result.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_bytes_roundtrip_f32() {
        let original = vec![1.0f32, 2.5, -3.14, 0.0];
        let bytes = floats_to_bytes(&original);
        let recovered: Vec<f32> = bytes_to_floats(&bytes);
        assert_eq!(original, recovered);
    }

    #[test]
    fn test_bytes_roundtrip_f64() {
        let original = vec![1.0f64, 2.5, -3.14, 0.0];
        let bytes = floats_to_bytes(&original);
        let recovered: Vec<f64> = bytes_to_floats(&bytes);
        assert_eq!(original, recovered);
    }
}
