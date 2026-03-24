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
// All-gather
// ---------------------------------------------------------------------------

/// Gather tensors from all ranks and concatenate them along the first axis.
///
/// Each rank provides a local tensor (chunk). After all-gather, every rank
/// holds the concatenation of all chunks in rank order. All chunks must
/// have the same number of elements.
///
/// # Algorithm (star topology)
///
/// 1. Non-zero ranks send their data to rank 0.
/// 2. Rank 0 assembles the full buffer `[chunk_0 | chunk_1 | ... | chunk_{N-1}]`.
/// 3. Rank 0 broadcasts the full buffer to all other ranks.
pub fn all_gather<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
) -> FerrotorchResult<Tensor<T>> {
    all_gather_with_timeout(tensor, backend, DEFAULT_COLLECTIVE_TIMEOUT)
}

/// Like [`all_gather`] but with a configurable timeout for recv operations.
pub fn all_gather_with_timeout<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
    timeout: Duration,
) -> FerrotorchResult<Tensor<T>> {
    let rank = backend.rank();
    let world_size = backend.world_size();
    let chunk_numel = tensor.numel();
    let chunk_byte_len = chunk_numel * std::mem::size_of::<T>();
    let total_numel = chunk_numel * world_size;

    if world_size == 1 {
        return Ok(tensor.clone());
    }

    // Zero-size tensor: nothing to communicate.
    if chunk_byte_len == 0 {
        return Tensor::from_storage(
            TensorStorage::cpu(Vec::<T>::new()),
            vec![0],
            false,
        );
    }

    if rank == 0 {
        // Allocate the full gathered buffer. Slot 0 = our own data.
        let mut gathered: Vec<T> = Vec::with_capacity(total_numel);
        gathered.extend(tensor.data_vec()?);

        // Receive chunks from ranks 1..world_size.
        let mut recv_buf = vec![0u8; chunk_byte_len];
        for src in 1..world_size {
            backend.recv_timeout(&mut recv_buf, src, timeout)?;
            gathered.extend(bytes_to_floats::<T>(&recv_buf));
        }

        // Broadcast the full buffer to all other ranks.
        let full_bytes = floats_to_bytes(&gathered);
        for dst in 1..world_size {
            backend.send(&full_bytes, dst)?;
        }

        Tensor::from_storage(TensorStorage::cpu(gathered), vec![total_numel], false)
    } else {
        // Send our chunk to rank 0.
        let local = tensor.data_vec()?;
        backend.send(&floats_to_bytes(&local), 0)?;

        // Receive the full gathered buffer from rank 0.
        let full_byte_len = total_numel * std::mem::size_of::<T>();
        let mut recv_buf = vec![0u8; full_byte_len];
        backend.recv_timeout(&mut recv_buf, 0, timeout)?;
        let gathered = bytes_to_floats::<T>(&recv_buf);

        Tensor::from_storage(TensorStorage::cpu(gathered), vec![total_numel], false)
    }
}

// ---------------------------------------------------------------------------
// Reduce-scatter
// ---------------------------------------------------------------------------

/// Reduce tensors across all ranks and scatter the result so each rank
/// gets only its chunk.
///
/// Each rank provides a tensor of length `chunk_size * world_size`. The
/// tensors are element-wise reduced (sum or mean), then the result is split
/// into `world_size` equal chunks and chunk `i` is sent to rank `i`.
///
/// # Algorithm (star topology)
///
/// 1. Non-zero ranks send their data to rank 0.
/// 2. Rank 0 reduces (element-wise sum/mean) across all inputs.
/// 3. Rank 0 splits the reduced buffer into chunks and sends chunk `i` to
///    rank `i`.
pub fn reduce_scatter<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
) -> FerrotorchResult<Tensor<T>> {
    reduce_scatter_with_timeout(tensor, backend, op, DEFAULT_COLLECTIVE_TIMEOUT)
}

/// Like [`reduce_scatter`] but with a configurable timeout.
pub fn reduce_scatter_with_timeout<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
    timeout: Duration,
) -> FerrotorchResult<Tensor<T>> {
    let rank = backend.rank();
    let world_size = backend.world_size();
    let total_numel = tensor.numel();

    if total_numel % world_size != 0 {
        return Err(DistributedError::SizeMismatch {
            expected: total_numel,
            got: world_size,
        }
        .into());
    }

    let chunk_numel = total_numel / world_size;

    if world_size == 1 {
        // Single rank: chunk = entire tensor, optionally apply mean (no-op
        // since dividing by 1).
        return Ok(tensor.clone());
    }

    // Zero-size tensor: nothing to communicate.
    if total_numel == 0 {
        return Tensor::from_storage(
            TensorStorage::cpu(Vec::<T>::new()),
            vec![0],
            false,
        );
    }

    let byte_len = total_numel * std::mem::size_of::<T>();

    if rank == 0 {
        // Start with our own data.
        let mut accum = tensor.data_vec()?;

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

        // Send chunk i to rank i (skip ourselves — we keep chunk 0).
        for dst in 1..world_size {
            let start = dst * chunk_numel;
            let end = start + chunk_numel;
            let chunk_bytes = floats_to_bytes(&accum[start..end]);
            backend.send(&chunk_bytes, dst)?;
        }

        // Our chunk is the first slice.
        let our_chunk = accum[..chunk_numel].to_vec();
        Tensor::from_storage(TensorStorage::cpu(our_chunk), vec![chunk_numel], false)
    } else {
        // Send our data to rank 0.
        let local = tensor.data_vec()?;
        backend.send(&floats_to_bytes(&local), 0)?;

        // Receive our chunk from rank 0.
        let chunk_byte_len = chunk_numel * std::mem::size_of::<T>();
        let mut recv_buf = vec![0u8; chunk_byte_len];
        backend.recv_timeout(&mut recv_buf, 0, timeout)?;
        let chunk = bytes_to_floats::<T>(&recv_buf);

        Tensor::from_storage(TensorStorage::cpu(chunk), vec![chunk_numel], false)
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
    fn test_all_gather_4_ranks() {
        // Rank i has [10*i, 10*i+1]. After all-gather, all ranks should
        // hold [0, 1, 10, 11, 20, 21, 30, 31].
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let base = (rank * 10) as f32;
                    let t = ferrotorch_core::from_slice(&[base, base + 1.0], &[2]).unwrap();
                    all_gather(&t, b.as_ref()).unwrap()
                })
            })
            .collect();

        let expected = [0.0f32, 1.0, 10.0, 11.0, 20.0, 21.0, 30.0, 31.0];
        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            assert_eq!(data.len(), 8);
            for (got, want) in data.iter().zip(expected.iter()) {
                assert!(
                    (got - want).abs() < 1e-6,
                    "expected {want}, got {got}"
                );
            }
        }
    }

    #[test]
    fn test_all_gather_single_rank() {
        let group = SimulatedBackend::create_group(1).unwrap();
        let t = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let result = all_gather(&t, &group[0]).unwrap();
        assert_eq!(result.data().unwrap(), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn test_reduce_scatter_sum_4_ranks() {
        // Each rank has [1, 2, 3, 4] (total_numel = 4, chunk_numel = 1).
        // Sum across 4 ranks = [4, 8, 12, 16].
        // Rank 0 gets [4], rank 1 gets [8], rank 2 gets [12], rank 3 gets [16].
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let t = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0, 4.0], &[4]).unwrap();
                    let result = reduce_scatter(&t, b.as_ref(), ReduceOp::Sum).unwrap();
                    (rank, result.data_vec().unwrap())
                })
            })
            .collect();

        let expected_per_rank: [f32; 4] = [4.0, 8.0, 12.0, 16.0];
        for h in handles {
            let (rank, data) = h.join().unwrap();
            assert_eq!(data.len(), 1, "rank {rank} should get 1 element");
            assert!(
                (data[0] - expected_per_rank[rank]).abs() < 1e-6,
                "rank {rank}: expected {}, got {}",
                expected_per_rank[rank],
                data[0]
            );
        }
    }

    #[test]
    fn test_reduce_scatter_mean_2_ranks() {
        // 2 ranks, each has [10, 20]. Sum = [20, 40]. Mean = [10, 20].
        // Rank 0 gets [10], rank 1 gets [20].
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let t = ferrotorch_core::from_slice(&[10.0f32, 20.0], &[2]).unwrap();
                    let result = reduce_scatter(&t, b.as_ref(), ReduceOp::Mean).unwrap();
                    (rank, result.data_vec().unwrap())
                })
            })
            .collect();

        for h in handles {
            let (rank, data) = h.join().unwrap();
            assert_eq!(data.len(), 1);
            let expected = if rank == 0 { 10.0f32 } else { 20.0 };
            assert!(
                (data[0] - expected).abs() < 1e-6,
                "rank {rank}: expected {expected}, got {}",
                data[0]
            );
        }
    }

    #[test]
    fn test_reduce_scatter_single_rank() {
        let group = SimulatedBackend::create_group(1).unwrap();
        let t = ferrotorch_core::from_slice(&[1.0f32, 2.0, 3.0], &[3]).unwrap();
        let result = reduce_scatter(&t, &group[0], ReduceOp::Sum).unwrap();
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
