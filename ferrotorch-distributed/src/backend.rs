//! Communication backends for distributed training.
//!
//! A [`Backend`] abstracts point-to-point messaging so that collective
//! operations and DDP are transport-agnostic. Two implementations are
//! provided:
//!
//! - [`TcpBackend`] — real multi-process backend over TCP sockets.
//! - [`SimulatedBackend`] — in-process backend using channels, suitable
//!   for unit tests without spawning multiple processes.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use ferrotorch_core::FerrotorchResult;

use crate::error::DistributedError;

// ---------------------------------------------------------------------------
// Backend trait
// ---------------------------------------------------------------------------

/// Transport-agnostic communication backend.
///
/// Every rank in a distributed job holds a `Backend` that can send/receive
/// raw byte buffers to/from any other rank, plus a collective barrier.
pub trait Backend: Send + Sync {
    /// This rank's index in the world (0-based).
    fn rank(&self) -> usize;

    /// Total number of ranks in the process group.
    fn world_size(&self) -> usize;

    /// Send `data` to `dst_rank`.
    fn send(&self, data: &[u8], dst_rank: usize) -> FerrotorchResult<()>;

    /// Receive into `dst` from `src_rank`. The caller must allocate `dst`
    /// with the correct length before calling.
    fn recv(&self, dst: &mut [u8], src_rank: usize) -> FerrotorchResult<()>;

    /// Receive into `dst` from `src_rank` with a timeout.
    ///
    /// Returns [`DistributedError::Timeout`] if the receive does not
    /// complete within `timeout`. The default implementation delegates
    /// to [`recv`](Self::recv) (no timeout).
    fn recv_timeout(
        &self,
        dst: &mut [u8],
        src_rank: usize,
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        let _ = timeout;
        self.recv(dst, src_rank)
    }

    /// Block until every rank has reached this barrier.
    fn barrier(&self) -> FerrotorchResult<()>;
}

// ---------------------------------------------------------------------------
// TCP backend
// ---------------------------------------------------------------------------

/// Real multi-process backend over TCP sockets.
///
/// Uses a simple rendezvous protocol:
/// 1. Rank 0 listens on `addr` and accepts `world_size - 1` connections.
/// 2. Non-zero ranks connect to rank 0.
/// 3. Rank 0 relays addressing information so all ranks establish
///    pairwise connections.
///
/// Each connection is wrapped in a `Mutex` to allow concurrent
/// send/recv on different pairs from different threads.
pub struct TcpBackend {
    rank: usize,
    world_size: usize,
    /// One TCP stream per peer, indexed by peer rank. `None` for the
    /// self-slot (no self-loop) and for peers that are not directly
    /// connected in the star topology (non-zero ranks only connect to
    /// rank 0).
    connections: Vec<Option<Mutex<TcpStream>>>,
}

impl TcpBackend {
    /// Launch the TCP rendezvous and return a ready-to-use backend.
    ///
    /// * `rank` — this process's rank (0-based).
    /// * `world_size` — total number of processes.
    /// * `master_addr` — `host:port` where rank 0 listens.
    pub fn new(rank: usize, world_size: usize, master_addr: &str) -> FerrotorchResult<Self> {
        if world_size < 2 {
            return Err(DistributedError::InvalidWorldSize { world_size }.into());
        }
        if rank >= world_size {
            return Err(DistributedError::InvalidRank { rank, world_size }.into());
        }

        // Phase 1: rank 0 collects one connection per non-zero rank.
        let mut peer_streams: Vec<Option<TcpStream>> = (0..world_size).map(|_| None).collect();

        if rank == 0 {
            let listener = TcpListener::bind(master_addr).map_err(|e| DistributedError::Io {
                message: format!("rank 0 bind {master_addr}: {e}"),
            })?;

            // Accept connections from ranks 1..world_size-1.
            for _ in 1..world_size {
                let (mut stream, _addr) = listener.accept().map_err(|e| DistributedError::Io {
                    message: format!("rank 0 accept: {e}"),
                })?;
                // First 8 bytes: the connecting rank as little-endian u64.
                let mut rank_buf = [0u8; 8];
                stream
                    .read_exact(&mut rank_buf)
                    .map_err(|e| DistributedError::Io {
                        message: format!("rank 0 read peer rank: {e}"),
                    })?;
                let peer_rank = u64::from_le_bytes(rank_buf) as usize;
                if peer_rank >= world_size || peer_rank == 0 {
                    return Err(DistributedError::InvalidRank {
                        rank: peer_rank,
                        world_size,
                    }
                    .into());
                }
                peer_streams[peer_rank] = Some(stream);
            }
        } else {
            // Non-zero rank: connect to rank 0 and announce our rank.
            let mut stream = TcpStream::connect(master_addr).map_err(|e| DistributedError::Io {
                message: format!("rank {rank} connect to {master_addr}: {e}"),
            })?;
            stream
                .write_all(&(rank as u64).to_le_bytes())
                .map_err(|e| DistributedError::Io {
                    message: format!("rank {rank} announce: {e}"),
                })?;
            peer_streams[0] = Some(stream);
        }

        // Phase 2: rank 0 broadcasts peer addresses so every rank can
        // form a full mesh. For simplicity in this MVP, we use a star
        // topology where all traffic goes through rank 0's connections.
        // A full mesh can be added later.

        // Collect into the final connections vec. For the star topology,
        // non-zero ranks only have a connection to rank 0, and rank 0 has
        // connections to all others. The self-slot and unconnected peers
        // are `None`.
        let connections: Vec<Option<Mutex<TcpStream>>> = peer_streams
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                if i == rank {
                    // Self-slot: no self-loop needed.
                    None
                } else {
                    // Some(stream) for connected peers, None for unconnected
                    // peers (non-zero ranks only connect to rank 0 in star
                    // topology).
                    opt.map(Mutex::new)
                }
            })
            .collect();

        Ok(Self {
            rank,
            world_size,
            connections,
        })
    }
}

impl Backend for TcpBackend {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, data: &[u8], dst_rank: usize) -> FerrotorchResult<()> {
        if dst_rank == self.rank {
            return Err(DistributedError::SelfSend { rank: self.rank }.into());
        }
        if dst_rank >= self.world_size {
            return Err(DistributedError::InvalidRank {
                rank: dst_rank,
                world_size: self.world_size,
            }
            .into());
        }

        let conn = self.connections[dst_rank]
            .as_ref()
            .ok_or(DistributedError::NoConnection { rank: dst_rank })?;

        let mut stream = conn.lock().map_err(|e| DistributedError::LockPoisoned {
            message: format!("send to rank {dst_rank}: {e}"),
        })?;

        // Length-prefixed protocol: send length (8 bytes LE) then payload.
        let len_bytes = (data.len() as u64).to_le_bytes();
        stream
            .write_all(&len_bytes)
            .map_err(|e| DistributedError::Io {
                message: format!("send len to rank {dst_rank}: {e}"),
            })?;
        stream.write_all(data).map_err(|e| DistributedError::Io {
            message: format!("send data to rank {dst_rank}: {e}"),
        })?;
        stream.flush().map_err(|e| DistributedError::Io {
            message: format!("flush to rank {dst_rank}: {e}"),
        })?;

        Ok(())
    }

    fn recv(&self, dst: &mut [u8], src_rank: usize) -> FerrotorchResult<()> {
        if src_rank == self.rank {
            return Err(DistributedError::SelfSend { rank: self.rank }.into());
        }
        if src_rank >= self.world_size {
            return Err(DistributedError::InvalidRank {
                rank: src_rank,
                world_size: self.world_size,
            }
            .into());
        }

        let conn = self.connections[src_rank]
            .as_ref()
            .ok_or(DistributedError::NoConnection { rank: src_rank })?;

        let mut stream = conn.lock().map_err(|e| DistributedError::LockPoisoned {
            message: format!("recv from rank {src_rank}: {e}"),
        })?;

        // Read length prefix.
        let mut len_bytes = [0u8; 8];
        stream
            .read_exact(&mut len_bytes)
            .map_err(|e| DistributedError::Io {
                message: format!("recv len from rank {src_rank}: {e}"),
            })?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        if len != dst.len() {
            return Err(DistributedError::SizeMismatch {
                expected: dst.len(),
                got: len,
            }
            .into());
        }

        stream.read_exact(dst).map_err(|e| DistributedError::Io {
            message: format!("recv data from rank {src_rank}: {e}"),
        })?;

        Ok(())
    }

    fn recv_timeout(
        &self,
        dst: &mut [u8],
        src_rank: usize,
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        if src_rank == self.rank {
            return Err(DistributedError::SelfSend { rank: self.rank }.into());
        }
        if src_rank >= self.world_size {
            return Err(DistributedError::InvalidRank {
                rank: src_rank,
                world_size: self.world_size,
            }
            .into());
        }

        let conn = self.connections[src_rank]
            .as_ref()
            .ok_or(DistributedError::NoConnection { rank: src_rank })?;

        let mut stream = conn.lock().map_err(|e| DistributedError::LockPoisoned {
            message: format!("recv_timeout from rank {src_rank}: {e}"),
        })?;

        // Set the read timeout for this operation.
        stream
            .set_read_timeout(Some(timeout))
            .map_err(|e| DistributedError::Io {
                message: format!("set_read_timeout for rank {src_rank}: {e}"),
            })?;

        // Read length prefix.
        let mut len_bytes = [0u8; 8];
        let result = (|| {
            stream.read_exact(&mut len_bytes).map_err(|e| {
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut
                {
                    DistributedError::Timeout {
                        seconds: timeout.as_secs(),
                    }
                } else {
                    DistributedError::Io {
                        message: format!("recv_timeout len from rank {src_rank}: {e}"),
                    }
                }
            })?;
            let len = u64::from_le_bytes(len_bytes) as usize;
            if len != dst.len() {
                return Err(DistributedError::SizeMismatch {
                    expected: dst.len(),
                    got: len,
                });
            }
            stream.read_exact(dst).map_err(|e| {
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut
                {
                    DistributedError::Timeout {
                        seconds: timeout.as_secs(),
                    }
                } else {
                    DistributedError::Io {
                        message: format!("recv_timeout data from rank {src_rank}: {e}"),
                    }
                }
            })?;
            Ok(())
        })();

        // Restore blocking mode (no timeout) regardless of outcome.
        let _ = stream.set_read_timeout(None);

        result.map_err(Into::into)
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        // Simple barrier: all ranks send a byte to rank 0, rank 0 waits
        // for all, then rank 0 sends a byte back to each.
        let tag = [0u8; 1];
        if self.rank == 0 {
            let mut buf = [0u8; 1];
            for r in 1..self.world_size {
                self.recv(&mut buf, r)?;
            }
            for r in 1..self.world_size {
                self.send(&tag, r)?;
            }
        } else {
            self.send(&tag, 0)?;
            let mut buf = [0u8; 1];
            self.recv(&mut buf, 0)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Simulated backend (in-process, channel-based)
// ---------------------------------------------------------------------------

/// Shared channel state for all simulated ranks.
///
/// `channels[src][dst]` is the `(Sender, Receiver)` pair for messages
/// from `src` to `dst`.
type ChannelMatrix = Arc<Vec<Vec<(Mutex<Sender<Vec<u8>>>, Mutex<Receiver<Vec<u8>>>)>>>;

/// In-process backend using `std::sync::mpsc` channels.
///
/// Designed for testing collectives and DDP without spawning processes.
/// Create all ranks via [`SimulatedBackend::create_group`], which returns
/// one `SimulatedBackend` per rank.
pub struct SimulatedBackend {
    rank: usize,
    world_size: usize,
    /// `channels[src][dst]` — sender side is used by `src`, receiver by `dst`.
    channels: ChannelMatrix,
}

impl SimulatedBackend {
    /// Create a group of `world_size` simulated backends, one per rank.
    ///
    /// Returns a `Vec<SimulatedBackend>` where index `i` is rank `i`.
    pub fn create_group(world_size: usize) -> FerrotorchResult<Vec<Self>> {
        if world_size == 0 {
            return Err(DistributedError::InvalidWorldSize { world_size }.into());
        }

        // Build the channel matrix: channels[src][dst].
        let mut matrix: Vec<Vec<(Mutex<Sender<Vec<u8>>>, Mutex<Receiver<Vec<u8>>>)>> = Vec::new();

        for _src in 0..world_size {
            let mut row = Vec::new();
            for _dst in 0..world_size {
                let (tx, rx) = mpsc::channel();
                row.push((Mutex::new(tx), Mutex::new(rx)));
            }
            matrix.push(row);
        }

        let shared = Arc::new(matrix);

        let backends: Vec<Self> = (0..world_size)
            .map(|rank| Self {
                rank,
                world_size,
                channels: Arc::clone(&shared),
            })
            .collect();

        Ok(backends)
    }
}

impl Backend for SimulatedBackend {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, data: &[u8], dst_rank: usize) -> FerrotorchResult<()> {
        if dst_rank >= self.world_size {
            return Err(DistributedError::InvalidRank {
                rank: dst_rank,
                world_size: self.world_size,
            }
            .into());
        }

        // channels[self.rank][dst_rank].sender
        let tx = self.channels[self.rank][dst_rank].0.lock().map_err(|e| {
            DistributedError::LockPoisoned {
                message: format!("send channel lock rank {} -> {dst_rank}: {e}", self.rank),
            }
        })?;

        tx.send(data.to_vec())
            .map_err(|e| DistributedError::ChannelClosed {
                message: format!("send rank {} -> {dst_rank}: {e}", self.rank),
            })?;

        Ok(())
    }

    fn recv(&self, dst: &mut [u8], src_rank: usize) -> FerrotorchResult<()> {
        if src_rank >= self.world_size {
            return Err(DistributedError::InvalidRank {
                rank: src_rank,
                world_size: self.world_size,
            }
            .into());
        }

        // channels[src_rank][self.rank].receiver
        let rx = self.channels[src_rank][self.rank].1.lock().map_err(|e| {
            DistributedError::LockPoisoned {
                message: format!("recv channel lock rank {src_rank} -> {}: {e}", self.rank),
            }
        })?;

        let data = rx.recv().map_err(|e| DistributedError::ChannelClosed {
            message: format!("recv rank {src_rank} -> {}: {e}", self.rank),
        })?;

        if data.len() != dst.len() {
            return Err(DistributedError::SizeMismatch {
                expected: dst.len(),
                got: data.len(),
            }
            .into());
        }

        dst.copy_from_slice(&data);
        Ok(())
    }

    fn recv_timeout(
        &self,
        dst: &mut [u8],
        src_rank: usize,
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        if src_rank >= self.world_size {
            return Err(DistributedError::InvalidRank {
                rank: src_rank,
                world_size: self.world_size,
            }
            .into());
        }

        let rx = self.channels[src_rank][self.rank].1.lock().map_err(|e| {
            DistributedError::LockPoisoned {
                message: format!(
                    "recv_timeout channel lock rank {src_rank} -> {}: {e}",
                    self.rank
                ),
            }
        })?;

        let data = rx.recv_timeout(timeout).map_err(|e| match e {
            mpsc::RecvTimeoutError::Timeout => DistributedError::Timeout {
                seconds: timeout.as_secs(),
            },
            mpsc::RecvTimeoutError::Disconnected => DistributedError::ChannelClosed {
                message: format!(
                    "recv_timeout rank {src_rank} -> {}: disconnected",
                    self.rank
                ),
            },
        })?;

        if data.len() != dst.len() {
            return Err(DistributedError::SizeMismatch {
                expected: dst.len(),
                got: data.len(),
            }
            .into());
        }

        dst.copy_from_slice(&data);
        Ok(())
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        // Same star-topology barrier as TcpBackend: gather at rank 0,
        // then scatter acknowledgement.
        let tag = [0u8; 1];
        if self.rank == 0 {
            let mut buf = [0u8; 1];
            for r in 1..self.world_size {
                self.recv(&mut buf, r)?;
            }
            for r in 1..self.world_size {
                self.send(&tag, r)?;
            }
        } else {
            self.send(&tag, 0)?;
            let mut buf = [0u8; 1];
            self.recv(&mut buf, 0)?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::thread;

    #[test]
    fn test_simulated_send_recv() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let mut iter = group.into_iter();
        let b0 = Arc::new(iter.next().unwrap());
        let b1 = Arc::new(iter.next().unwrap());

        let b0c = Arc::clone(&b0);
        let sender = thread::spawn(move || {
            b0c.send(&[1, 2, 3, 4], 1).unwrap();
        });

        let mut buf = [0u8; 4];
        b1.recv(&mut buf, 0).unwrap();
        sender.join().unwrap();

        assert_eq!(buf, [1, 2, 3, 4]);
    }

    #[test]
    fn test_simulated_barrier() {
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<SimulatedBackend>> = group.into_iter().map(Arc::new).collect();

        let handles: Vec<_> = arcs
            .into_iter()
            .map(|b| {
                thread::spawn(move || {
                    b.barrier().unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_simulated_rank_world_size() {
        let group = SimulatedBackend::create_group(3).unwrap();
        assert_eq!(group[0].rank(), 0);
        assert_eq!(group[1].rank(), 1);
        assert_eq!(group[2].rank(), 2);
        assert_eq!(group[0].world_size(), 3);
    }

    #[test]
    fn test_invalid_world_size() {
        let result = SimulatedBackend::create_group(0);
        assert!(result.is_err());
    }

    #[test]
    fn test_send_to_invalid_rank() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let result = group[0].send(&[1], 5);
        assert!(result.is_err());
    }
}
