//! Tensor-level point-to-point primitives. (#591)
//!
//! [`Backend::send`] / [`Backend::recv`] operate on raw byte buffers; this
//! module wraps them so the shape, dtype, and float semantics stay typed
//! at the call site. Mirrors `torch.distributed.send` / `torch.distributed.recv`.

use std::time::Duration;

use ferrotorch_core::{FerrotorchError, FerrotorchResult, Float, Tensor, TensorStorage};

use crate::backend::Backend;
use crate::collective::{DEFAULT_COLLECTIVE_TIMEOUT, bytes_to_floats, floats_to_bytes};

/// Send `tensor` to `dst_rank`. Blocks until the backend has written all
/// bytes (TCP) or queued the send (NCCL — see follow-up for true async).
///
/// The receiver must call [`recv`] (or [`recv_into`]) with a tensor of
/// matching shape — there is no shape negotiation on the wire.
pub fn send<T: Float>(
    tensor: &Tensor<T>,
    dst_rank: usize,
    backend: &dyn Backend,
) -> FerrotorchResult<()> {
    let world = backend.world_size();
    if dst_rank >= world {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("send: dst_rank {dst_rank} >= world_size {world}"),
        });
    }
    if dst_rank == backend.rank() {
        return Err(FerrotorchError::InvalidArgument {
            message: "send: dst_rank equals self rank — use a tensor copy instead".into(),
        });
    }
    let data = tensor.data_vec()?;
    let bytes = floats_to_bytes(&data);
    backend.send(&bytes, dst_rank)
}

/// Receive a tensor of `shape` from `src_rank`. Allocates a fresh buffer
/// of the appropriate size, fills it from the backend, and wraps it in
/// a CPU `Tensor<T>`. Uses the default collective timeout; for a custom
/// timeout call [`recv_with_timeout`].
pub fn recv<T: Float>(
    shape: &[usize],
    src_rank: usize,
    backend: &dyn Backend,
) -> FerrotorchResult<Tensor<T>> {
    recv_with_timeout(shape, src_rank, backend, DEFAULT_COLLECTIVE_TIMEOUT)
}

/// [`recv`] with a configurable timeout.
pub fn recv_with_timeout<T: Float>(
    shape: &[usize],
    src_rank: usize,
    backend: &dyn Backend,
    timeout: Duration,
) -> FerrotorchResult<Tensor<T>> {
    let world = backend.world_size();
    if src_rank >= world {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("recv: src_rank {src_rank} >= world_size {world}"),
        });
    }
    if src_rank == backend.rank() {
        return Err(FerrotorchError::InvalidArgument {
            message: "recv: src_rank equals self rank".into(),
        });
    }
    let numel: usize = shape.iter().product::<usize>().max(1);
    let byte_len = numel * std::mem::size_of::<T>();
    let mut buf = vec![0u8; byte_len];
    backend.recv_timeout(&mut buf, src_rank, timeout)?;
    let data: Vec<T> = bytes_to_floats(&buf);
    Tensor::from_storage(TensorStorage::cpu(data), shape.to_vec(), false)
}

/// Receive into a caller-owned tensor (overwrites in place). The tensor's
/// `numel()` must match the size the sender posted; otherwise the backend
/// will return a length-mismatch error.
///
/// Useful when the receive buffer is already allocated (e.g. ring-buffer
/// reuse across iterations).
pub fn recv_into<T: Float>(
    out: &mut Tensor<T>,
    src_rank: usize,
    backend: &dyn Backend,
) -> FerrotorchResult<()> {
    recv_into_with_timeout(out, src_rank, backend, DEFAULT_COLLECTIVE_TIMEOUT)
}

/// [`recv_into`] with a configurable timeout.
pub fn recv_into_with_timeout<T: Float>(
    out: &mut Tensor<T>,
    src_rank: usize,
    backend: &dyn Backend,
    timeout: Duration,
) -> FerrotorchResult<()> {
    let world = backend.world_size();
    if src_rank >= world {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("recv_into: src_rank {src_rank} >= world_size {world}"),
        });
    }
    let byte_len = out.numel() * std::mem::size_of::<T>();
    let mut buf = vec![0u8; byte_len];
    backend.recv_timeout(&mut buf, src_rank, timeout)?;
    let data: Vec<T> = bytes_to_floats(&buf);
    let shape = out.shape().to_vec();
    *out = Tensor::from_storage(TensorStorage::cpu(data), shape, false)?;
    Ok(())
}

/// Atomic send-and-receive between exactly two peers. Avoids the manual
/// rank-ordering deadlock-avoidance dance that callers would otherwise
/// have to write themselves: the lower rank sends first then receives,
/// the higher rank receives first then sends. Mirrors
/// `torch.distributed.batch_isend_irecv` for the simple two-party case.
///
/// Returns the received tensor. The send tensor and receive shape may
/// have different shapes (asymmetric exchanges are allowed).
pub fn sendrecv<T: Float>(
    send_tensor: &Tensor<T>,
    recv_shape: &[usize],
    peer: usize,
    backend: &dyn Backend,
) -> FerrotorchResult<Tensor<T>> {
    let rank = backend.rank();
    let world = backend.world_size();
    if peer >= world {
        return Err(FerrotorchError::InvalidArgument {
            message: format!("sendrecv: peer {peer} >= world_size {world}"),
        });
    }
    if peer == rank {
        return Err(FerrotorchError::InvalidArgument {
            message: "sendrecv: peer equals self rank".into(),
        });
    }

    if rank < peer {
        send::<T>(send_tensor, peer, backend)?;
        recv::<T>(recv_shape, peer, backend)
    } else {
        let r = recv::<T>(recv_shape, peer, backend)?;
        send::<T>(send_tensor, peer, backend)?;
        Ok(r)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::SimulatedBackend;
    use ferrotorch_core::creation::from_slice;

    fn pair_backends() -> (SimulatedBackend, SimulatedBackend) {
        let group = SimulatedBackend::create_group(2).unwrap();
        let mut iter = group.into_iter();
        let a = iter.next().unwrap();
        let b = iter.next().unwrap();
        (a, b)
    }

    #[test]
    fn send_recv_roundtrip_floats() {
        let (a, b) = pair_backends();
        let payload = from_slice::<f32>(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).unwrap();
        let payload_clone = payload.clone();

        let handle_send = std::thread::spawn(move || {
            send::<f32>(&payload_clone, 1, &a).unwrap();
        });
        let handle_recv =
            std::thread::spawn(move || -> Tensor<f32> { recv::<f32>(&[2, 2], 0, &b).unwrap() });

        handle_send.join().unwrap();
        let received = handle_recv.join().unwrap();
        assert_eq!(received.shape(), &[2, 2]);
        assert_eq!(received.data().unwrap(), payload.data().unwrap());
    }

    #[test]
    fn recv_into_overwrites_in_place() {
        let (a, b) = pair_backends();
        let payload = from_slice::<f32>(&[10.0, 20.0, 30.0], &[3]).unwrap();
        let payload_clone = payload.clone();

        let handle_send = std::thread::spawn(move || {
            send::<f32>(&payload_clone, 1, &a).unwrap();
        });
        let handle_recv = std::thread::spawn(move || -> Tensor<f32> {
            let mut buf = from_slice::<f32>(&[0.0, 0.0, 0.0], &[3]).unwrap();
            recv_into::<f32>(&mut buf, 0, &b).unwrap();
            buf
        });

        handle_send.join().unwrap();
        let received = handle_recv.join().unwrap();
        assert_eq!(received.data().unwrap(), &[10.0, 20.0, 30.0]);
    }

    #[test]
    fn send_rejects_self_rank() {
        let group = SimulatedBackend::create_group(1).unwrap();
        let solo = group.into_iter().next().unwrap();
        let t = from_slice::<f32>(&[1.0], &[1]).unwrap();
        let err = send::<f32>(&t, 0, &solo).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn recv_rejects_oob_rank() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let mut iter = group.into_iter();
        let a = iter.next().unwrap();
        let _b = iter.next().unwrap();
        let err = recv::<f32>(&[1], 5, &a).unwrap_err();
        assert!(matches!(err, FerrotorchError::InvalidArgument { .. }));
    }

    #[test]
    fn sendrecv_swaps_two_peers() {
        let (a, b) = pair_backends();
        let from_a = from_slice::<f32>(&[1.0, 2.0], &[2]).unwrap();
        let from_b = from_slice::<f32>(&[100.0, 200.0], &[2]).unwrap();
        let send_buf_a = from_a.clone();
        let send_buf_b = from_b.clone();

        let h_a = std::thread::spawn(move || -> Tensor<f32> {
            sendrecv::<f32>(&send_buf_a, &[2], 1, &a).unwrap()
        });
        let h_b = std::thread::spawn(move || -> Tensor<f32> {
            sendrecv::<f32>(&send_buf_b, &[2], 0, &b).unwrap()
        });

        let recv_a = h_a.join().unwrap();
        let recv_b = h_b.join().unwrap();
        // Rank 0 received what rank 1 sent.
        assert_eq!(recv_a.data().unwrap(), from_b.data().unwrap());
        assert_eq!(recv_b.data().unwrap(), from_a.data().unwrap());
    }
}
