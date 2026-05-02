//! Asynchronous collective handles for overlapping collectives with
//! compute.
//!
//! The synchronous collectives in [`collective`](crate::collective) block
//! the calling thread until the collective completes. Many distributed
//! training patterns can overlap a collective with local compute to hide
//! communication latency — classic examples include FSDP's backward
//! prefetch (all-gather the next layer's parameters while the current
//! layer's backward runs) and gradient allreduce during backward.
//!
//! [`AsyncAllGather`] and [`AsyncReduceScatter`] wrap the synchronous
//! primitives in a background thread and expose a `wait()` method that
//! blocks until the collective completes and returns the gathered or
//! scattered tensor.
//!
//! # Thread safety
//!
//! The async primitives require the backend to be `Send + Sync + 'static`
//! (i.e. held in an `Arc`). Each rank should have at most **one**
//! outstanding async collective at a time — the Backend's send/recv
//! channels are untagged, so interleaving two concurrent collectives on
//! the same rank would corrupt the message stream. The tests in this
//! module cover the single-outstanding-op case and the FSDP prefetch hook
//! relies on it.
//!
//! CL-373.

use std::sync::Arc;
use std::sync::mpsc::{self, Receiver};
use std::thread::{self, JoinHandle};

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor};

use crate::backend::Backend;
use crate::collective::{ReduceOp, all_gather, reduce_scatter};

/// A pending asynchronous all-gather operation.
///
/// The collective runs in a background thread; call [`wait`](Self::wait)
/// to block until it completes and obtain the gathered tensor.
///
/// Dropping a `PendingCollective` without calling `wait()` detaches the
/// background thread; the collective still runs to completion on the
/// backend, but the result is discarded. Use this only when you want to
/// fire-and-forget (e.g., a broadcast where the local rank doesn't need
/// the output).
pub struct PendingCollective<T: Float> {
    recv: Option<Receiver<FerrotorchResult<Tensor<T>>>>,
    handle: Option<JoinHandle<()>>,
    /// Name of the operation, for error reporting.
    op_name: &'static str,
}

impl<T: Float> PendingCollective<T> {
    /// Block until the collective completes and return the result.
    ///
    /// Consumes the handle. Calling `wait()` twice returns an error.
    pub fn wait(mut self) -> FerrotorchResult<Tensor<T>> {
        let recv = self
            .recv
            .take()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: format!("{}: wait() called on already-consumed handle", self.op_name),
            })?;
        let result = recv.recv().map_err(|e| FerrotorchError::InvalidArgument {
            message: format!("{}: background thread disconnected: {e}", self.op_name),
        })?;
        if let Some(handle) = self.handle.take() {
            // Join the background thread; propagate any panic as an error.
            if let Err(e) = handle.join() {
                return Err(FerrotorchError::InvalidArgument {
                    message: format!("{}: background thread panicked: {e:?}", self.op_name),
                });
            }
        }
        result
    }

    /// Name of the operation (for diagnostics).
    pub fn op_name(&self) -> &'static str {
        self.op_name
    }
}

/// Spawn a background thread that runs [`all_gather`] on `tensor` and
/// returns a handle that resolves to the gathered result.
///
/// The `backend` must be held in an `Arc` so ownership can cross the
/// thread boundary. The current rank must not initiate another
/// collective on the same backend until this handle has been `wait()`ed
/// on (see module docs).
pub fn async_all_gather<T: Float + 'static>(
    tensor: Tensor<T>,
    backend: Arc<dyn Backend>,
) -> PendingCollective<T> {
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let result = all_gather(&tensor, backend.as_ref());
        // If the receiver is dropped, the send fails silently — that's
        // the "fire and forget" path. We still run the collective to
        // completion on the backend.
        let _ = tx.send(result);
    });
    PendingCollective {
        recv: Some(rx),
        handle: Some(handle),
        op_name: "async_all_gather",
    }
}

/// Spawn a background thread that runs [`reduce_scatter`] on `tensor`
/// and returns a handle that resolves to the scattered shard.
///
/// The `backend` must be held in an `Arc` so ownership can cross the
/// thread boundary. The current rank must not initiate another
/// collective on the same backend until this handle has been `wait()`ed
/// on (see module docs).
pub fn async_reduce_scatter<T: Float + 'static>(
    tensor: Tensor<T>,
    backend: Arc<dyn Backend>,
    op: ReduceOp,
) -> PendingCollective<T> {
    let (tx, rx) = mpsc::channel();
    let handle = thread::spawn(move || {
        let result = reduce_scatter(&tensor, backend.as_ref(), op);
        let _ = tx.send(result);
    });
    PendingCollective {
        recv: Some(rx),
        handle: Some(handle),
        op_name: "async_reduce_scatter",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SimulatedBackend;
    use ferrotorch_core::storage::TensorStorage;
    use std::thread;

    #[test]
    fn test_async_all_gather_matches_sync() {
        // Two ranks, each holds [rank, rank+10]. After all_gather both
        // ranks should see [0, 10, 1, 11].
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<dyn Backend>> = group
            .into_iter()
            .map(|b| Arc::new(b) as Arc<dyn Backend>)
            .collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank() as f32;
                    let t = Tensor::from_storage(
                        TensorStorage::cpu(vec![rank, rank + 10.0]),
                        vec![2],
                        false,
                    )
                    .unwrap();
                    // Async path.
                    let pending = async_all_gather(t.clone(), b.clone());
                    // (Main thread could do other work here — simulated
                    // by the thread::yield_now below.)
                    thread::yield_now();
                    let out = pending.wait().unwrap();
                    out.data_vec().unwrap()
                })
            })
            .collect();

        for h in handles {
            let data = h.join().unwrap();
            assert_eq!(data, &[0.0, 10.0, 1.0, 11.0]);
        }
    }

    #[test]
    fn test_async_reduce_scatter_matches_sync() {
        // Two ranks, each holds [1,2,3,4]. reduce_scatter(mean) gives
        // rank 0 -> [1,2], rank 1 -> [3,4].
        let group = SimulatedBackend::create_group(2).unwrap();
        let arcs: Vec<Arc<dyn Backend>> = group
            .into_iter()
            .map(|b| Arc::new(b) as Arc<dyn Backend>)
            .collect();

        let handles: Vec<_> = arcs
            .iter()
            .cloned()
            .map(|b| {
                thread::spawn(move || {
                    let rank = b.rank();
                    let t = Tensor::from_storage(
                        TensorStorage::cpu(vec![1.0f32, 2.0, 3.0, 4.0]),
                        vec![4],
                        false,
                    )
                    .unwrap();
                    let pending = async_reduce_scatter(t, b.clone(), ReduceOp::Mean);
                    let out = pending.wait().unwrap();
                    (rank, out.data_vec().unwrap())
                })
            })
            .collect();

        for h in handles {
            let (rank, data) = h.join().unwrap();
            if rank == 0 {
                assert_eq!(data.len(), 2);
                assert!((data[0] - 1.0).abs() < 1e-6);
                assert!((data[1] - 2.0).abs() < 1e-6);
            } else {
                assert_eq!(data.len(), 2);
                assert!((data[0] - 3.0).abs() < 1e-6);
                assert!((data[1] - 4.0).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn test_async_all_gather_world_size_1() {
        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let t = Tensor::from_storage(TensorStorage::cpu(vec![5.0f32, 6.0, 7.0]), vec![3], false)
            .unwrap();
        let pending = async_all_gather(t, b);
        let out = pending.wait().unwrap();
        assert_eq!(out.data_vec().unwrap(), &[5.0, 6.0, 7.0]);
    }
}
