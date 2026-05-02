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
        type ChannelPair = (Mutex<Sender<Vec<u8>>>, Mutex<Receiver<Vec<u8>>>);
        let mut matrix: Vec<Vec<ChannelPair>> = Vec::new();

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

// ---------------------------------------------------------------------------
// SubBackend — subgroup adapter
// ---------------------------------------------------------------------------

/// A backend view that restricts communication to a subset of global ranks.
///
/// `SubBackend` wraps a parent [`Backend`] and a list of member global ranks
/// to form a logical sub-process-group. It translates every `rank()`,
/// `world_size()`, `send()`, and `recv()` call into the equivalent parent-
/// backend operation using the subgroup→global rank mapping.
///
/// Because `SubBackend` implements the [`Backend`] trait, the existing
/// [`allreduce`](crate::collective::allreduce),
/// [`all_gather`](crate::collective::all_gather), and
/// [`reduce_scatter`](crate::collective::reduce_scatter) collective functions
/// work on a subgroup without any changes: they only see the `SubBackend`
/// through the trait, so "rank 0" in the collective means "the first member
/// of the subgroup" and `world_size` means the subgroup size.
///
/// Used by FSDP's [`HybridShard`](crate::fsdp::ShardingStrategy::HybridShard)
/// strategy to form an intra-node (sharding) subgroup and an inter-node
/// (replication) subgroup.
///
/// CL-327.
pub struct SubBackend {
    parent: Arc<dyn Backend>,
    /// Global ranks that are members of this subgroup, sorted ascending.
    members: Vec<usize>,
    /// This process's index within `members` (the local rank).
    local_rank: usize,
}

impl SubBackend {
    /// Create a subgroup view from a parent backend and a list of member
    /// global ranks.
    ///
    /// The caller's rank (read from `parent.rank()`) must be in `members`.
    /// `members` is sorted and deduplicated before being stored.
    ///
    /// # Errors
    ///
    /// - [`DistributedError::InvalidRank`] if the parent's rank is not in
    ///   `members`, or if any member is ≥ parent `world_size`.
    /// - [`DistributedError::InvalidWorldSize`] if `members` is empty.
    pub fn new(parent: Arc<dyn Backend>, members: Vec<usize>) -> FerrotorchResult<Self> {
        if members.is_empty() {
            return Err(DistributedError::InvalidWorldSize { world_size: 0 }.into());
        }

        let parent_world = parent.world_size();
        for &m in &members {
            if m >= parent_world {
                return Err(DistributedError::InvalidRank {
                    rank: m,
                    world_size: parent_world,
                }
                .into());
            }
        }

        // Sort and dedup so local rank ordering is deterministic.
        let mut sorted_members = members;
        sorted_members.sort_unstable();
        sorted_members.dedup();

        let parent_rank = parent.rank();
        let local_rank = sorted_members
            .iter()
            .position(|&r| r == parent_rank)
            .ok_or(DistributedError::InvalidRank {
                rank: parent_rank,
                world_size: sorted_members.len(),
            })?;

        Ok(Self {
            parent,
            members: sorted_members,
            local_rank,
        })
    }

    /// Return the global ranks that make up this subgroup, in ascending
    /// order. The index of this rank's entry is its local rank.
    pub fn members(&self) -> &[usize] {
        &self.members
    }

    /// Map a local (subgroup-relative) rank to its global rank.
    ///
    /// # Panics
    ///
    /// Panics if `local` is out of bounds.
    pub fn to_global(&self, local: usize) -> usize {
        self.members[local]
    }

    /// Map a global rank to its local subgroup rank, or `None` if the
    /// global rank is not a member of this subgroup.
    pub fn to_local(&self, global: usize) -> Option<usize> {
        self.members.iter().position(|&r| r == global)
    }

    /// The parent backend this subgroup was derived from.
    pub fn parent(&self) -> &Arc<dyn Backend> {
        &self.parent
    }
}

impl Backend for SubBackend {
    fn rank(&self) -> usize {
        self.local_rank
    }

    fn world_size(&self) -> usize {
        self.members.len()
    }

    fn send(&self, data: &[u8], dst_rank: usize) -> FerrotorchResult<()> {
        if dst_rank >= self.members.len() {
            return Err(DistributedError::InvalidRank {
                rank: dst_rank,
                world_size: self.members.len(),
            }
            .into());
        }
        self.parent.send(data, self.members[dst_rank])
    }

    fn recv(&self, dst: &mut [u8], src_rank: usize) -> FerrotorchResult<()> {
        if src_rank >= self.members.len() {
            return Err(DistributedError::InvalidRank {
                rank: src_rank,
                world_size: self.members.len(),
            }
            .into());
        }
        self.parent.recv(dst, self.members[src_rank])
    }

    fn recv_timeout(
        &self,
        dst: &mut [u8],
        src_rank: usize,
        timeout: Duration,
    ) -> FerrotorchResult<()> {
        if src_rank >= self.members.len() {
            return Err(DistributedError::InvalidRank {
                rank: src_rank,
                world_size: self.members.len(),
            }
            .into());
        }
        self.parent
            .recv_timeout(dst, self.members[src_rank], timeout)
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        // Gather-scatter barrier within the subgroup: local rank 0 waits
        // for a byte from every other member, then sends one back. We
        // use the parent backend's send/recv via the local→global rank
        // map, so this doesn't conflict with a simultaneous parent-level
        // barrier as long as the subgroups are non-overlapping at rank 0.
        let tag = [0u8; 1];
        let size = self.members.len();
        if size <= 1 {
            return Ok(());
        }
        if self.local_rank == 0 {
            let mut buf = [0u8; 1];
            for r in 1..size {
                self.recv(&mut buf, r)?;
            }
            for r in 1..size {
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

    // -----------------------------------------------------------------------
    // SubBackend tests. CL-327
    // -----------------------------------------------------------------------

    #[test]
    fn test_subbackend_local_rank_and_world_size() {
        // 4 parent ranks, subgroup is {1, 3}. Rank 3 should be local 1.
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<dyn Backend>> = group
            .into_iter()
            .map(|b| Arc::new(b) as Arc<dyn Backend>)
            .collect();

        let sub_for_rank1 = SubBackend::new(Arc::clone(&arcs[1]), vec![1, 3]).unwrap();
        let sub_for_rank3 = SubBackend::new(Arc::clone(&arcs[3]), vec![1, 3]).unwrap();

        assert_eq!(sub_for_rank1.rank(), 0);
        assert_eq!(sub_for_rank1.world_size(), 2);
        assert_eq!(sub_for_rank3.rank(), 1);
        assert_eq!(sub_for_rank3.world_size(), 2);
    }

    #[test]
    fn test_subbackend_global_rank_not_in_members_is_error() {
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<dyn Backend>> = group
            .into_iter()
            .map(|b| Arc::new(b) as Arc<dyn Backend>)
            .collect();

        // Rank 0 tries to create a subgroup that doesn't include itself.
        let result = SubBackend::new(Arc::clone(&arcs[0]), vec![1, 3]);
        assert!(result.is_err());
    }

    #[test]
    fn test_subbackend_empty_members_is_error() {
        let group = SimulatedBackend::create_group(2).unwrap();
        let arc: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let result = SubBackend::new(arc, vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_subbackend_send_recv_routes_through_parent() {
        // 4 parent ranks, subgroup {0, 2}. Local rank 0 sends to local rank 1
        // (which is global 2).
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<dyn Backend>> = group
            .into_iter()
            .map(|b| Arc::new(b) as Arc<dyn Backend>)
            .collect();

        let sub0 = SubBackend::new(Arc::clone(&arcs[0]), vec![0, 2]).unwrap();
        let sub2 = SubBackend::new(Arc::clone(&arcs[2]), vec![0, 2]).unwrap();

        let sender = thread::spawn(move || {
            sub0.send(&[9, 8, 7], 1).unwrap();
        });

        let mut buf = [0u8; 3];
        sub2.recv(&mut buf, 0).unwrap();
        sender.join().unwrap();

        assert_eq!(buf, [9, 8, 7]);
    }

    #[test]
    fn test_subbackend_barrier() {
        // 6 parent ranks, subgroup {0, 2, 4} (even ranks). Barrier runs
        // only across the subgroup.
        let group = SimulatedBackend::create_group(6).unwrap();
        let arcs: Vec<Arc<dyn Backend>> = group
            .into_iter()
            .map(|b| Arc::new(b) as Arc<dyn Backend>)
            .collect();

        let members = vec![0usize, 2, 4];

        let handles: Vec<_> = [0usize, 2, 4]
            .into_iter()
            .map(|global_rank| {
                let parent = Arc::clone(&arcs[global_rank]);
                let ms = members.clone();
                thread::spawn(move || {
                    let sub = SubBackend::new(parent, ms).unwrap();
                    sub.barrier().unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }
    }

    #[test]
    fn test_subbackend_to_global_to_local() {
        let group = SimulatedBackend::create_group(4).unwrap();
        let arc: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let sub = SubBackend::new(arc, vec![0, 2]).unwrap();

        // Members are sorted, so to_global(0)=0, to_global(1)=2.
        assert_eq!(sub.to_global(0), 0);
        assert_eq!(sub.to_global(1), 2);
        assert_eq!(sub.to_local(0), Some(0));
        assert_eq!(sub.to_local(2), Some(1));
        assert_eq!(sub.to_local(1), None);
        assert_eq!(sub.to_local(3), None);
    }

    #[test]
    fn test_subbackend_sorts_and_dedups_members() {
        let group = SimulatedBackend::create_group(4).unwrap();
        let arcs: Vec<Arc<dyn Backend>> = group
            .into_iter()
            .map(|b| Arc::new(b) as Arc<dyn Backend>)
            .collect();

        // Pass unsorted + duplicated members; expect sorted + deduped.
        let sub = SubBackend::new(Arc::clone(&arcs[2]), vec![3, 2, 0, 2, 3]).unwrap();
        assert_eq!(sub.members(), &[0, 2, 3]);
        assert_eq!(sub.rank(), 1); // rank 2 maps to local index 1 in sorted order.
        assert_eq!(sub.world_size(), 3);
    }
}
