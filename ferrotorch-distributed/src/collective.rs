//! Collective communication operations.
//!
//! These functions coordinate tensors across all ranks in a process group
//! via a [`Backend`].
//!
//! Two allreduce algorithms are provided:
//!
//! - **Ring allreduce** — used when [`Backend::supports_full_mesh`] returns
//!   `true`. Splits data into `world_size` chunks and performs a reduce-scatter
//!   followed by an all-gather in `2*(N-1)` steps. Each step transfers only
//!   `numel/N` elements, giving optimal bandwidth utilization.
//!
//! - **Star-topology allreduce** — fallback when the backend only supports
//!   communication through rank 0 (e.g., [`TcpBackend`](crate::backend::TcpBackend)).
//!   Gathers at rank 0, reduces, and broadcasts back.

use ferrotorch_core::storage::TensorStorage;
use ferrotorch_core::{Float, FerrotorchResult, Tensor};

use crate::backend::Backend;
use crate::error::DistributedError;

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
// Allreduce (dispatch)
// ---------------------------------------------------------------------------

/// Reduce a tensor across all ranks and distribute the result to every rank.
///
/// Each rank provides its local tensor. After allreduce, every rank holds
/// the same tensor whose values are the element-wise reduction of all
/// inputs.
///
/// The algorithm is chosen automatically based on backend capabilities:
/// ring allreduce for full-mesh backends, star-topology for others.
pub fn allreduce<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
) -> FerrotorchResult<Tensor<T>> {
    let world_size = backend.world_size();

    if world_size == 1 {
        // Single rank: nothing to reduce.
        return Ok(tensor.clone());
    }

    if backend.supports_full_mesh() {
        allreduce_ring(tensor, backend, op)
    } else {
        allreduce_star(tensor, backend, op)
    }
}

// ---------------------------------------------------------------------------
// Ring allreduce
// ---------------------------------------------------------------------------

/// Ring allreduce: optimal bandwidth utilization for full-mesh backends.
///
/// Phase 1 (reduce-scatter): Each rank sends one chunk to its right
/// neighbor and receives from its left. After `N-1` steps, each rank
/// holds the fully reduced value for exactly one chunk.
///
/// Phase 2 (all-gather): Each rank sends its fully-reduced chunk to
/// its right neighbor. After `N-1` steps, all ranks have all chunks.
fn allreduce_ring<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
) -> FerrotorchResult<Tensor<T>> {
    let rank = backend.rank();
    let world_size = backend.world_size();
    let numel = tensor.numel();
    let shape = tensor.shape().to_vec();
    let t_size = std::mem::size_of::<T>();

    // Flatten tensor data into a mutable working buffer.
    let local = tensor.data()?;
    let mut data: Vec<T> = local.to_vec();

    // Compute chunk boundaries. With numel elements split across
    // world_size ranks, some chunks may be 1 element larger than others.
    let chunk_sizes: Vec<usize> = compute_chunk_sizes(numel, world_size);
    let chunk_offsets: Vec<usize> = {
        let mut offsets = Vec::with_capacity(world_size);
        let mut off = 0;
        for &sz in &chunk_sizes {
            offsets.push(off);
            off += sz;
        }
        offsets
    };

    let right = (rank + 1) % world_size;
    let left = (rank + world_size - 1) % world_size;

    // Phase 1: Reduce-scatter.
    // In step i, rank r sends chunk (r - i) mod N to the right and
    // receives chunk (r - i - 1) mod N from the left, accumulating
    // the received data into its local buffer.
    for step in 0..(world_size - 1) {
        let send_chunk_idx = (rank + world_size - step) % world_size;
        let recv_chunk_idx = (rank + world_size - step - 1) % world_size;

        let send_offset = chunk_offsets[send_chunk_idx];
        let send_len = chunk_sizes[send_chunk_idx];
        let recv_offset = chunk_offsets[recv_chunk_idx];
        let recv_len = chunk_sizes[recv_chunk_idx];

        // Send before recv to avoid deadlock in lockstep ring.
        let send_bytes = floats_to_bytes(&data[send_offset..send_offset + send_len]);
        backend.send(&send_bytes, right)?;

        let mut recv_bytes = vec![0u8; recv_len * t_size];
        backend.recv(&mut recv_bytes, left)?;
        let received = bytes_to_floats::<T>(&recv_bytes);

        // Accumulate into the recv chunk position.
        for (i, &val) in received.iter().enumerate() {
            data[recv_offset + i] = data[recv_offset + i] + val;
        }
    }

    // After reduce-scatter, the chunk that rank r fully owns (has the
    // complete sum for) is chunk (r + 1) % N. This is because the last
    // recv_chunk in the reduce-scatter loop is (r - (N-2) - 1 + N) % N
    // = (r + 1) % N.
    let owned_chunk = (rank + 1) % world_size;

    // Apply mean after reduction if requested.
    if op == ReduceOp::Mean {
        let divisor = T::from(world_size).unwrap();
        let off = chunk_offsets[owned_chunk];
        let len = chunk_sizes[owned_chunk];
        for i in 0..len {
            data[off + i] = data[off + i] / divisor;
        }
    }

    // Phase 2: All-gather.
    // In step i, rank r sends its owned chunk (or a chunk it received
    // in a previous all-gather step) to the right, and receives a
    // fully-reduced chunk from the left.
    //
    // Step 0: send owned_chunk = (r+1)%N, recv (r+1-1+N)%N = r%N
    // Step i: send (r+1-i+N)%N, recv (r-i+N)%N
    for step in 0..(world_size - 1) {
        let send_chunk_idx = (owned_chunk + world_size - step) % world_size;
        let recv_chunk_idx = (owned_chunk + world_size - step - 1) % world_size;

        let send_offset = chunk_offsets[send_chunk_idx];
        let send_len = chunk_sizes[send_chunk_idx];
        let recv_offset = chunk_offsets[recv_chunk_idx];
        let recv_len = chunk_sizes[recv_chunk_idx];

        let send_bytes = floats_to_bytes(&data[send_offset..send_offset + send_len]);
        backend.send(&send_bytes, right)?;

        let mut recv_bytes = vec![0u8; recv_len * t_size];
        backend.recv(&mut recv_bytes, left)?;
        let received = bytes_to_floats::<T>(&recv_bytes);

        // Replace (this chunk is already fully reduced by the owner).
        data[recv_offset..recv_offset + recv_len].copy_from_slice(&received);
    }

    Tensor::from_storage(TensorStorage::cpu(data), shape, false)
}

/// Compute chunk sizes for splitting `numel` elements across `n` ranks.
///
/// Returns a Vec of length `n` where each entry is the chunk size.
/// The first `numel % n` chunks get `ceil(numel/n)` elements, the rest
/// get `floor(numel/n)`. If `numel` is 0, all chunks are size 0.
fn compute_chunk_sizes(numel: usize, n: usize) -> Vec<usize> {
    if n == 0 {
        return vec![];
    }
    let base = numel / n;
    let remainder = numel % n;
    (0..n)
        .map(|i| if i < remainder { base + 1 } else { base })
        .collect()
}

// ---------------------------------------------------------------------------
// Star-topology allreduce (fallback)
// ---------------------------------------------------------------------------

/// Star-topology allreduce: all traffic routed through rank 0.
///
/// 1. Non-zero ranks send their data to rank 0.
/// 2. Rank 0 reduces all received data with its own.
/// 3. Rank 0 broadcasts the result back to all other ranks.
fn allreduce_star<T: Float>(
    tensor: &Tensor<T>,
    backend: &dyn Backend,
    op: ReduceOp,
) -> FerrotorchResult<Tensor<T>> {
    let rank = backend.rank();
    let world_size = backend.world_size();
    let numel = tensor.numel();
    let byte_len = numel * std::mem::size_of::<T>();
    let shape = tensor.shape().to_vec();

    if rank == 0 {
        // Start with our own data.
        let local = tensor.data()?;
        let mut accum: Vec<T> = local.to_vec();

        // Receive from every other rank and accumulate.
        let mut recv_buf = vec![0u8; byte_len];
        for src in 1..world_size {
            backend.recv(&mut recv_buf, src)?;
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
        let local = tensor.data()?;
        let send_bytes = floats_to_bytes(local);
        backend.send(&send_bytes, 0)?;

        // Receive reduced result from rank 0.
        let mut recv_buf = vec![0u8; byte_len];
        backend.recv(&mut recv_buf, 0)?;
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
        let local = tensor.data()?;
        let send_bytes = floats_to_bytes(local);
        for dst in 0..world_size {
            if dst != root {
                backend.send(&send_bytes, dst)?;
            }
        }
        Ok(tensor.clone())
    } else {
        let mut recv_buf = vec![0u8; byte_len];
        backend.recv(&mut recv_buf, root)?;
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

/// Reinterpret a float slice as raw bytes (zero-copy view).
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
        // SAFETY: We checked alignment above and T is a POD float type.
        // We copy byte-by-byte to avoid alignment issues.
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

    // -------------------------------------------------------------------
    // Ring allreduce tests
    // -------------------------------------------------------------------

    #[test]
    fn test_ring_allreduce_sum_3_ranks() {
        // 3 ranks, each with [rank*10+1, rank*10+2, ..., rank*10+7].
        // Sum = element-wise sum across ranks.
        let group = SimulatedBackend::create_group(3).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let base = (rank * 10) as f32;
                    let data: Vec<f32> = (1..=7).map(|i| base + i as f32).collect();
                    let t = ferrotorch_core::from_slice(&data, &[7]).unwrap();
                    allreduce(&t, b.as_ref(), ReduceOp::Sum).unwrap()
                })
            })
            .collect();

        // Expected: for element j, sum = (0*10+j) + (1*10+j) + (2*10+j) = 30 + 3*j
        let expected: Vec<f32> = (1..=7).map(|j| 30.0 + 3.0 * j as f32).collect();

        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            assert_eq!(data.len(), 7);
            for (i, (&got, &exp)) in data.iter().zip(expected.iter()).enumerate() {
                assert!(
                    (got - exp).abs() < 1e-5,
                    "element {i}: expected {exp}, got {got}"
                );
            }
        }
    }

    #[test]
    fn test_ring_allreduce_mean_3_ranks() {
        // 3 ranks, each with [rank, rank, rank, rank, rank].
        // Mean = (0+1+2)/3 = 1.0 for all elements.
        let group = SimulatedBackend::create_group(3).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let val = rank as f32;
                    let t = ferrotorch_core::from_slice(&[val; 5], &[5]).unwrap();
                    allreduce(&t, b.as_ref(), ReduceOp::Mean).unwrap()
                })
            })
            .collect();

        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            assert_eq!(data.len(), 5);
            for &v in data {
                assert!(
                    (v - 1.0).abs() < 1e-5,
                    "expected 1.0, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_ring_allreduce_sum_5_ranks() {
        // 5 ranks to test with a prime number that doesn't evenly divide
        // typical tensor sizes. Each rank has 13 elements (13 % 5 != 0).
        let world = 5;
        let numel = 13;
        let group = SimulatedBackend::create_group(world).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    // Each rank has [rank+1, rank+1, ...] (13 elements)
                    let val = (rank + 1) as f32;
                    let data = vec![val; numel];
                    let t = ferrotorch_core::from_slice(&data, &[numel]).unwrap();
                    allreduce(&t, b.as_ref(), ReduceOp::Sum).unwrap()
                })
            })
            .collect();

        // Sum of 1..=5 = 15
        let expected = 15.0f32;

        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            assert_eq!(data.len(), numel);
            for (i, &v) in data.iter().enumerate() {
                assert!(
                    (v - expected).abs() < 1e-5,
                    "element {i}: expected {expected}, got {v}"
                );
            }
        }
    }

    #[test]
    fn test_ring_allreduce_2_ranks() {
        // Minimal multi-rank case: 2 ranks.
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let val = (rank + 1) as f32; // rank 0 -> 1.0, rank 1 -> 2.0
                    let t = ferrotorch_core::from_slice(&[val, val * 10.0], &[2]).unwrap();
                    allreduce(&t, b.as_ref(), ReduceOp::Sum).unwrap()
                })
            })
            .collect();

        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            assert_eq!(data.len(), 2);
            assert!(
                (data[0] - 3.0).abs() < 1e-5,
                "expected 3.0, got {}",
                data[0]
            );
            assert!(
                (data[1] - 30.0).abs() < 1e-5,
                "expected 30.0, got {}",
                data[1]
            );
        }
    }

    #[test]
    fn test_ring_allreduce_single_element() {
        // Edge case: single element tensor with 3 ranks.
        let group = SimulatedBackend::create_group(3).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let val = (rank + 1) as f32;
                    let t = ferrotorch_core::from_slice(&[val], &[1]).unwrap();
                    allreduce(&t, b.as_ref(), ReduceOp::Sum).unwrap()
                })
            })
            .collect();

        // Sum = 1 + 2 + 3 = 6
        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            assert_eq!(data.len(), 1);
            assert!(
                (data[0] - 6.0).abs() < 1e-5,
                "expected 6.0, got {}",
                data[0]
            );
        }
    }

    #[test]
    fn test_ring_allreduce_f64() {
        // Verify ring allreduce works with f64 type.
        let group = SimulatedBackend::create_group(3).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let val = (rank + 1) as f64;
                    let t = ferrotorch_core::from_slice(&[val, val * 2.0], &[2]).unwrap();
                    allreduce(&t, b.as_ref(), ReduceOp::Mean).unwrap()
                })
            })
            .collect();

        // Mean: (1+2+3)/3 = 2.0, (2+4+6)/3 = 4.0
        for h in handles {
            let result = h.join().unwrap();
            let data = result.data().unwrap();
            assert_eq!(data.len(), 2);
            assert!(
                (data[0] - 2.0).abs() < 1e-10,
                "expected 2.0, got {}",
                data[0]
            );
            assert!(
                (data[1] - 4.0).abs() < 1e-10,
                "expected 4.0, got {}",
                data[1]
            );
        }
    }

    #[test]
    fn test_compute_chunk_sizes() {
        // Evenly divisible.
        assert_eq!(compute_chunk_sizes(12, 3), vec![4, 4, 4]);
        assert_eq!(compute_chunk_sizes(12, 4), vec![3, 3, 3, 3]);

        // Remainder distributed to first chunks.
        assert_eq!(compute_chunk_sizes(13, 3), vec![5, 4, 4]);
        assert_eq!(compute_chunk_sizes(7, 3), vec![3, 2, 2]);
        assert_eq!(compute_chunk_sizes(1, 3), vec![1, 0, 0]);

        // Edge cases.
        assert_eq!(compute_chunk_sizes(0, 3), vec![0, 0, 0]);
        assert_eq!(compute_chunk_sizes(5, 1), vec![5]);
        assert!(compute_chunk_sizes(5, 0).is_empty());
    }
}
