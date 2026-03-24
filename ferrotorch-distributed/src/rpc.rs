//! Remote Procedure Call (RPC) framework for distributed training.
//!
//! Provides [`RpcContext`] for invoking functions on remote ranks, and
//! [`RRef`] for holding references to remote data. The RPC layer sits
//! above the raw [`Backend`](crate::backend::Backend) and adds:
//!
//! - Named function registry
//! - Typed serialization via raw byte encoding
//! - Synchronous (`rpc_sync`), asynchronous (`rpc_async`), and
//!   remote-reference (`remote`) call styles
//!
//! # Architecture
//!
//! Each rank runs an [`RpcContext`] which owns an [`RpcBackend`] for
//! point-to-point messaging. Functions are registered by name and
//! dispatched by a message-processing loop. [`RRef`] objects carry a
//! unique ID and the owner rank; calling [`RRef::to_here`] fetches the
//! value from the owner.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::thread;

// ---------------------------------------------------------------------------
// RPC errors
// ---------------------------------------------------------------------------

/// Errors specific to the RPC subsystem.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RpcError {
    #[error("RPC I/O error: {message}")]
    Io { message: String },

    #[error("RPC: function not found: {name}")]
    FunctionNotFound { name: String },

    #[error("RPC: invalid destination rank {dst} for world size {world_size}")]
    InvalidRank { dst: usize, world_size: usize },

    #[error("RPC: channel closed: {message}")]
    ChannelClosed { message: String },

    #[error("RPC: lock poisoned: {message}")]
    LockPoisoned { message: String },

    #[error("RPC: serialization error: {message}")]
    Serialization { message: String },

    #[error("RPC: remote reference not found: id={id} on rank {owner}")]
    RRefNotFound { id: u64, owner: usize },

    #[error("RPC: remote call failed: {message}")]
    RemoteError { message: String },

    #[error("RPC: self-call on rank {rank} — use local execution instead")]
    SelfCall { rank: usize },
}

impl From<RpcError> for ferrotorch_core::FerrotorchError {
    fn from(e: RpcError) -> Self {
        ferrotorch_core::FerrotorchError::InvalidArgument {
            message: e.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// Message protocol
// ---------------------------------------------------------------------------

/// Wire message types used by the RPC protocol.
///
/// Each message starts with a 1-byte tag followed by tag-specific payload.
#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)]
enum MessageTag {
    /// RPC call: fn_name_len(4) + fn_name + args_len(8) + args + request_id(8)
    Call = 1,
    /// RPC response: request_id(8) + payload_len(8) + payload
    Response = 2,
    /// RRef fetch request: rref_id(8) + request_id(8)
    FetchRRef = 3,
    /// RRef store notification: rref_id(8) + value_len(8) + value
    StoreRRef = 4,
    /// Shutdown signal
    Shutdown = 5,
}

impl MessageTag {
    #[allow(dead_code)]
    fn from_u8(b: u8) -> Option<Self> {
        match b {
            1 => Some(Self::Call),
            2 => Some(Self::Response),
            3 => Some(Self::FetchRRef),
            4 => Some(Self::StoreRRef),
            5 => Some(Self::Shutdown),
            _ => None,
        }
    }
}

// ---------------------------------------------------------------------------
// RPC Backend trait
// ---------------------------------------------------------------------------

/// Transport layer for RPC message passing.
///
/// Unlike the collective [`Backend`](crate::backend::Backend), the RPC
/// backend supports arbitrary-length messages (no pre-allocated receive
/// buffer) and is designed for request/response patterns.
pub trait RpcBackend: Send + Sync {
    /// Send a complete message to `dst_rank`.
    fn send(&self, dst_rank: usize, message: &[u8]) -> Result<(), RpcError>;

    /// Receive the next complete message from `src_rank`.
    ///
    /// Blocks until a message is available.
    fn recv(&self, src_rank: usize) -> Result<Vec<u8>, RpcError>;

    /// This rank's index.
    fn rank(&self) -> usize;

    /// Total number of ranks.
    fn world_size(&self) -> usize;
}

// ---------------------------------------------------------------------------
// TCP RPC Backend
// ---------------------------------------------------------------------------

/// TCP-based RPC backend using length-prefixed framing.
///
/// Rank 0 acts as a rendezvous point. All ranks form pairwise connections
/// through rank 0 (star topology). Each connection is wrapped in a Mutex
/// for thread safety.
pub struct TcpRpcBackend {
    rank: usize,
    world_size: usize,
    /// `connections[peer]` — TCP stream to each peer. `None` for self-slot
    /// and unconnected peers.
    connections: Vec<Option<Mutex<TcpStream>>>,
}

impl TcpRpcBackend {
    /// Create a TCP RPC backend. Uses the same rendezvous protocol as
    /// [`TcpBackend`](crate::backend::TcpBackend).
    pub fn new(rank: usize, world_size: usize, master_addr: &str) -> Result<Self, RpcError> {
        if world_size < 2 {
            return Err(RpcError::InvalidRank {
                dst: rank,
                world_size,
            });
        }
        if rank >= world_size {
            return Err(RpcError::InvalidRank {
                dst: rank,
                world_size,
            });
        }

        let mut peer_streams: Vec<Option<TcpStream>> = (0..world_size).map(|_| None).collect();

        if rank == 0 {
            let listener = TcpListener::bind(master_addr).map_err(|e| RpcError::Io {
                message: format!("rank 0 bind {master_addr}: {e}"),
            })?;

            for _ in 1..world_size {
                let (mut stream, _addr) = listener.accept().map_err(|e| RpcError::Io {
                    message: format!("rank 0 accept: {e}"),
                })?;
                let mut rank_buf = [0u8; 8];
                stream.read_exact(&mut rank_buf).map_err(|e| RpcError::Io {
                    message: format!("rank 0 read peer rank: {e}"),
                })?;
                let peer_rank = u64::from_le_bytes(rank_buf) as usize;
                if peer_rank >= world_size || peer_rank == 0 {
                    return Err(RpcError::InvalidRank {
                        dst: peer_rank,
                        world_size,
                    });
                }
                peer_streams[peer_rank] = Some(stream);
            }
        } else {
            let mut stream =
                TcpStream::connect(master_addr).map_err(|e| RpcError::Io {
                    message: format!("rank {rank} connect to {master_addr}: {e}"),
                })?;
            stream
                .write_all(&(rank as u64).to_le_bytes())
                .map_err(|e| RpcError::Io {
                    message: format!("rank {rank} announce: {e}"),
                })?;
            peer_streams[0] = Some(stream);
        }

        let connections: Vec<Option<Mutex<TcpStream>>> = peer_streams
            .into_iter()
            .enumerate()
            .map(|(i, opt)| {
                if i == rank {
                    None
                } else {
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

impl RpcBackend for TcpRpcBackend {
    fn send(&self, dst_rank: usize, message: &[u8]) -> Result<(), RpcError> {
        if dst_rank == self.rank {
            return Err(RpcError::SelfCall { rank: self.rank });
        }
        if dst_rank >= self.world_size {
            return Err(RpcError::InvalidRank {
                dst: dst_rank,
                world_size: self.world_size,
            });
        }

        let conn = self.connections[dst_rank]
            .as_ref()
            .ok_or(RpcError::Io {
                message: format!("no connection to rank {dst_rank}"),
            })?;

        let mut stream = conn.lock().map_err(|e| RpcError::LockPoisoned {
            message: format!("send to rank {dst_rank}: {e}"),
        })?;

        // Length-prefixed: 8-byte LE length + payload.
        let len_bytes = (message.len() as u64).to_le_bytes();
        stream.write_all(&len_bytes).map_err(|e| RpcError::Io {
            message: format!("send len to rank {dst_rank}: {e}"),
        })?;
        stream.write_all(message).map_err(|e| RpcError::Io {
            message: format!("send data to rank {dst_rank}: {e}"),
        })?;
        stream.flush().map_err(|e| RpcError::Io {
            message: format!("flush to rank {dst_rank}: {e}"),
        })?;

        Ok(())
    }

    fn recv(&self, src_rank: usize) -> Result<Vec<u8>, RpcError> {
        if src_rank == self.rank {
            return Err(RpcError::SelfCall { rank: self.rank });
        }
        if src_rank >= self.world_size {
            return Err(RpcError::InvalidRank {
                dst: src_rank,
                world_size: self.world_size,
            });
        }

        let conn = self.connections[src_rank]
            .as_ref()
            .ok_or(RpcError::Io {
                message: format!("no connection to rank {src_rank}"),
            })?;

        let mut stream = conn.lock().map_err(|e| RpcError::LockPoisoned {
            message: format!("recv from rank {src_rank}: {e}"),
        })?;

        // Read length prefix.
        let mut len_bytes = [0u8; 8];
        stream.read_exact(&mut len_bytes).map_err(|e| RpcError::Io {
            message: format!("recv len from rank {src_rank}: {e}"),
        })?;
        let len = u64::from_le_bytes(len_bytes) as usize;

        // Read payload.
        let mut buf = vec![0u8; len];
        stream.read_exact(&mut buf).map_err(|e| RpcError::Io {
            message: format!("recv data from rank {src_rank}: {e}"),
        })?;

        Ok(buf)
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

// ---------------------------------------------------------------------------
// Simulated RPC Backend (for testing)
// ---------------------------------------------------------------------------

/// Channel-based RPC backend for in-process testing.
///
/// Built on `std::sync::mpsc`, analogous to
/// [`SimulatedBackend`](crate::backend::SimulatedBackend) but using
/// the [`RpcBackend`] interface with variable-length messages.
pub struct SimulatedRpcBackend {
    rank: usize,
    world_size: usize,
    /// `channels[src][dst]` — sender is used by `src`, receiver by `dst`.
    channels: Arc<Vec<Vec<(
        Mutex<std::sync::mpsc::Sender<Vec<u8>>>,
        Mutex<std::sync::mpsc::Receiver<Vec<u8>>>,
    )>>>,
}

impl SimulatedRpcBackend {
    /// Create a group of `world_size` simulated RPC backends, one per rank.
    pub fn create_group(world_size: usize) -> Result<Vec<Self>, RpcError> {
        if world_size == 0 {
            return Err(RpcError::InvalidRank {
                dst: 0,
                world_size: 0,
            });
        }

        let mut matrix = Vec::with_capacity(world_size);
        for _src in 0..world_size {
            let mut row = Vec::with_capacity(world_size);
            for _dst in 0..world_size {
                let (tx, rx) = std::sync::mpsc::channel();
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

impl RpcBackend for SimulatedRpcBackend {
    fn send(&self, dst_rank: usize, message: &[u8]) -> Result<(), RpcError> {
        if dst_rank >= self.world_size {
            return Err(RpcError::InvalidRank {
                dst: dst_rank,
                world_size: self.world_size,
            });
        }

        let tx = self.channels[self.rank][dst_rank]
            .0
            .lock()
            .map_err(|e| RpcError::LockPoisoned {
                message: format!("send channel lock rank {} -> {dst_rank}: {e}", self.rank),
            })?;

        tx.send(message.to_vec())
            .map_err(|e| RpcError::ChannelClosed {
                message: format!("send rank {} -> {dst_rank}: {e}", self.rank),
            })?;

        Ok(())
    }

    fn recv(&self, src_rank: usize) -> Result<Vec<u8>, RpcError> {
        if src_rank >= self.world_size {
            return Err(RpcError::InvalidRank {
                dst: src_rank,
                world_size: self.world_size,
            });
        }

        let rx = self.channels[src_rank][self.rank]
            .1
            .lock()
            .map_err(|e| RpcError::LockPoisoned {
                message: format!("recv channel lock rank {src_rank} -> {}: {e}", self.rank),
            })?;

        rx.recv().map_err(|e| RpcError::ChannelClosed {
            message: format!("recv rank {src_rank} -> {}: {e}", self.rank),
        })
    }

    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }
}

// ---------------------------------------------------------------------------
// Remote Reference (RRef)
// ---------------------------------------------------------------------------

/// Global counter for unique RRef IDs.
static RREF_ID_COUNTER: AtomicU64 = AtomicU64::new(1);

/// A reference to data that may live on a remote rank.
///
/// If this rank is the owner, the value is stored locally in `local_value`.
/// Otherwise, [`to_here`](RRef::to_here) fetches it from the owner via the
/// RPC backend.
///
/// `RRef` is cheap to clone (interior `Arc`).
#[derive(Debug)]
pub struct RRef<T> {
    /// The rank that owns the actual data.
    owner_rank: usize,
    /// Unique identifier for this remote reference.
    id: u64,
    /// Cached local value (if this rank is the owner or if fetched).
    local_value: Arc<Mutex<Option<Arc<T>>>>,
}

impl<T> Clone for RRef<T> {
    fn clone(&self) -> Self {
        Self {
            owner_rank: self.owner_rank,
            id: self.id,
            local_value: Arc::clone(&self.local_value),
        }
    }
}

impl<T> RRef<T> {
    /// Create a new RRef that owns a local value.
    pub fn new_local(owner_rank: usize, value: T) -> Self {
        Self {
            owner_rank,
            id: RREF_ID_COUNTER.fetch_add(1, Ordering::Relaxed),
            local_value: Arc::new(Mutex::new(Some(Arc::new(value)))),
        }
    }

    /// Create a new RRef that points to a remote value (not yet fetched).
    pub fn new_remote(owner_rank: usize, id: u64) -> Self {
        Self {
            owner_rank,
            id,
            local_value: Arc::new(Mutex::new(None)),
        }
    }

    /// The rank that owns this data.
    pub fn owner(&self) -> usize {
        self.owner_rank
    }

    /// The unique ID of this remote reference.
    pub fn id(&self) -> u64 {
        self.id
    }

    /// Whether the value is available locally (either we own it or it was
    /// previously fetched).
    pub fn is_local(&self) -> bool {
        self.local_value
            .lock()
            .map(|guard| guard.is_some())
            .unwrap_or(false)
    }

    /// Get the local value if available.
    ///
    /// Returns `None` if the value has not been fetched yet.
    pub fn local_value(&self) -> Option<Arc<T>> {
        self.local_value
            .lock()
            .ok()
            .and_then(|guard| guard.clone())
    }

    /// Set the local value (used when fetching from remote).
    pub fn set_local_value(&self, value: T) {
        if let Ok(mut guard) = self.local_value.lock() {
            *guard = Some(Arc::new(value));
        }
    }
}

// ---------------------------------------------------------------------------
// RPC Future
// ---------------------------------------------------------------------------

/// A future representing the result of an asynchronous RPC call.
///
/// Created by [`RpcContext::rpc_async`]. The result becomes available when
/// the remote rank completes the call and sends the response back.
pub struct RpcFuture<T> {
    /// Shared state protected by a mutex + condvar for blocking wait.
    state: Arc<(Mutex<RpcFutureState<T>>, Condvar)>,
}

struct RpcFutureState<T> {
    result: Option<Result<T, RpcError>>,
}

impl<T> RpcFuture<T> {
    fn new() -> (Self, RpcFutureResolver<T>) {
        let state = Arc::new((
            Mutex::new(RpcFutureState { result: None }),
            Condvar::new(),
        ));
        let resolver = RpcFutureResolver {
            state: Arc::clone(&state),
        };
        (Self { state }, resolver)
    }

    /// Block until the result is available, then return it.
    pub fn wait(self) -> Result<T, RpcError> {
        let (lock, cvar) = &*self.state;
        let mut guard = lock.lock().map_err(|e| RpcError::LockPoisoned {
            message: format!("RpcFuture::wait: {e}"),
        })?;

        while guard.result.is_none() {
            guard = cvar.wait(guard).map_err(|e| RpcError::LockPoisoned {
                message: format!("RpcFuture::wait condvar: {e}"),
            })?;
        }

        guard.result.take().unwrap()
    }

    /// Check whether the result is ready without blocking.
    pub fn is_done(&self) -> bool {
        self.state
            .0
            .lock()
            .map(|guard| guard.result.is_some())
            .unwrap_or(false)
    }
}

/// Handle for resolving an `RpcFuture` from another thread.
struct RpcFutureResolver<T> {
    state: Arc<(Mutex<RpcFutureState<T>>, Condvar)>,
}

impl<T> RpcFutureResolver<T> {
    fn resolve(self, result: Result<T, RpcError>) {
        let (lock, cvar) = &*self.state;
        if let Ok(mut guard) = lock.lock() {
            guard.result = Some(result);
            cvar.notify_all();
        }
    }
}

// ---------------------------------------------------------------------------
// RRef Store (per-rank registry of remote reference values)
// ---------------------------------------------------------------------------

/// Stores byte-serialized values for RRefs owned by this rank.
struct RRefStore {
    store: Mutex<HashMap<u64, Vec<u8>>>,
}

impl RRefStore {
    fn new() -> Self {
        Self {
            store: Mutex::new(HashMap::new()),
        }
    }

    fn insert(&self, id: u64, data: Vec<u8>) {
        if let Ok(mut guard) = self.store.lock() {
            guard.insert(id, data);
        }
    }

    fn get(&self, id: u64) -> Option<Vec<u8>> {
        self.store
            .lock()
            .ok()
            .and_then(|guard| guard.get(&id).cloned())
    }
}

// ---------------------------------------------------------------------------
// RPC Context
// ---------------------------------------------------------------------------

/// Type alias for registered RPC functions.
///
/// Each function takes serialized arguments (`Vec<u8>`) and returns
/// serialized results (`Vec<u8>`).
type RpcFn = Box<dyn Fn(Vec<u8>) -> Vec<u8> + Send + Sync>;

/// Central RPC context for a single rank.
///
/// Manages function registration, message dispatch, and remote reference
/// storage. Each rank creates one `RpcContext` at the start of distributed
/// training and uses it throughout.
///
/// # Usage
///
/// ```ignore
/// use ferrotorch_distributed::rpc::{RpcContext, SimulatedRpcBackend};
///
/// let backends = SimulatedRpcBackend::create_group(2).unwrap();
/// let ctx = RpcContext::new(0, 2, Arc::new(backends.into_iter().next().unwrap()));
/// ctx.register("double", |data: Vec<u8>| {
///     let val = f32::from_le_bytes(data[0..4].try_into().unwrap());
///     (val * 2.0).to_le_bytes().to_vec()
/// });
/// ```
pub struct RpcContext {
    rank: usize,
    world_size: usize,
    /// Backend for sending/receiving messages.
    backend: Arc<dyn RpcBackend>,
    /// Registry of callable functions.
    registry: Arc<Mutex<HashMap<String, RpcFn>>>,
    /// Local storage for RRef values owned by this rank.
    rref_store: Arc<RRefStore>,
    /// Counter for request IDs (for matching responses to requests).
    next_request_id: AtomicU64,
    /// Pending responses: request_id -> channel for the response bytes.
    /// Used by the async message dispatch loop for matching responses to
    /// outstanding requests when running a multi-threaded RPC server.
    #[allow(dead_code)]
    pending_responses: Arc<Mutex<HashMap<u64, std::sync::mpsc::Sender<Vec<u8>>>>>,
}

impl RpcContext {
    /// Create a new RPC context for the given rank.
    pub fn new(rank: usize, world_size: usize, backend: Arc<dyn RpcBackend>) -> Self {
        Self {
            rank,
            world_size,
            backend,
            registry: Arc::new(Mutex::new(HashMap::new())),
            rref_store: Arc::new(RRefStore::new()),
            next_request_id: AtomicU64::new(1),
            pending_responses: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// This rank's index.
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Total number of ranks.
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Register a callable function by name.
    ///
    /// The function takes serialized arguments and returns serialized results.
    /// It must be `Send + Sync + 'static` since it may be called from the
    /// message processing thread.
    pub fn register<F>(&self, name: &str, f: F)
    where
        F: Fn(Vec<u8>) -> Vec<u8> + Send + Sync + 'static,
    {
        if let Ok(mut reg) = self.registry.lock() {
            reg.insert(name.to_string(), Box::new(f));
        }
    }

    /// Invoke a registered function locally (useful for self-calls and testing).
    pub fn call_local(&self, fn_name: &str, args: Vec<u8>) -> Result<Vec<u8>, RpcError> {
        let reg = self.registry.lock().map_err(|e| RpcError::LockPoisoned {
            message: format!("call_local registry lock: {e}"),
        })?;

        let func = reg
            .get(fn_name)
            .ok_or_else(|| RpcError::FunctionNotFound {
                name: fn_name.to_string(),
            })?;

        Ok(func(args))
    }

    /// Blocking synchronous RPC call to a remote rank.
    ///
    /// Sends `args` (serialized as raw bytes) to `dst_rank`, which executes
    /// the function named `fn_name` and returns the result. If `dst_rank`
    /// is this rank, the function is invoked locally without network I/O.
    ///
    /// # Protocol
    ///
    /// 1. Serialize args + fn_name into a Call message.
    /// 2. Send to dst_rank.
    /// 3. Block waiting for the Response message with matching request ID.
    pub fn rpc_sync(
        &self,
        dst_rank: usize,
        fn_name: &str,
        args: &[u8],
    ) -> Result<Vec<u8>, RpcError> {
        if dst_rank >= self.world_size {
            return Err(RpcError::InvalidRank {
                dst: dst_rank,
                world_size: self.world_size,
            });
        }

        // Self-call: execute locally.
        if dst_rank == self.rank {
            return self.call_local(fn_name, args.to_vec());
        }

        let request_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);

        // Build the Call message.
        let msg = encode_call_message(fn_name, args, request_id);
        self.backend.send(dst_rank, &msg)?;

        // Wait for response by receiving from dst_rank.
        // In a simple synchronous protocol, we expect the next message
        // from dst_rank to be our response.
        let response_bytes = self.backend.recv(dst_rank)?;
        let (resp_id, payload) = decode_response_message(&response_bytes)?;

        if resp_id != request_id {
            return Err(RpcError::RemoteError {
                message: format!(
                    "response request_id mismatch: expected {request_id}, got {resp_id}"
                ),
            });
        }

        Ok(payload)
    }

    /// Non-blocking asynchronous RPC call to a remote rank.
    ///
    /// Returns an [`RpcFuture`] that can be waited on later. The actual
    /// RPC call is dispatched on a background thread.
    pub fn rpc_async(
        self: &Arc<Self>,
        dst_rank: usize,
        fn_name: &str,
        args: &[u8],
    ) -> RpcFuture<Vec<u8>> {
        let (future, resolver) = RpcFuture::new();

        let ctx = Arc::clone(self);
        let fn_name = fn_name.to_string();
        let args = args.to_vec();

        thread::spawn(move || {
            let result = ctx.rpc_sync(dst_rank, &fn_name, &args);
            resolver.resolve(result);
        });

        future
    }

    /// Execute a function remotely and return an [`RRef`] to the result.
    ///
    /// The function is executed on `dst_rank` and the result is stored in
    /// that rank's RRef store. The returned `RRef` can later be used to
    /// fetch the result via [`RRef::to_here`].
    ///
    /// If `dst_rank` is this rank, the function is executed locally and
    /// the RRef is populated immediately.
    pub fn remote(
        &self,
        dst_rank: usize,
        fn_name: &str,
        args: &[u8],
    ) -> Result<RRef<Vec<u8>>, RpcError> {
        if dst_rank >= self.world_size {
            return Err(RpcError::InvalidRank {
                dst: dst_rank,
                world_size: self.world_size,
            });
        }

        if dst_rank == self.rank {
            // Local execution: run the function and store the result.
            let result = self.call_local(fn_name, args.to_vec())?;
            let rref = RRef::new_local(self.rank, result);
            // Also store in the RRef store for potential remote fetch.
            if let Some(val) = rref.local_value() {
                self.rref_store.insert(rref.id(), (*val).clone());
            }
            return Ok(rref);
        }

        // Remote execution: send the call and get back the result.
        // Store it locally and return an RRef that already has the value.
        let result = self.rpc_sync(dst_rank, fn_name, args)?;
        let rref_id = RREF_ID_COUNTER.fetch_add(1, Ordering::Relaxed);
        let rref = RRef::new_remote(dst_rank, rref_id);
        rref.set_local_value(result);
        Ok(rref)
    }

    /// Process a single incoming RPC call message and send the response.
    ///
    /// This is the server-side handler. When a rank receives a Call message,
    /// it dispatches to the registered function and sends the Response back.
    pub fn process_call(
        &self,
        src_rank: usize,
        message: &[u8],
    ) -> Result<(), RpcError> {
        let (fn_name, args, request_id) = decode_call_message(message)?;

        let result = self.call_local(&fn_name, args)?;

        let response = encode_response_message(request_id, &result);
        self.backend.send(src_rank, &response)?;

        Ok(())
    }

    /// Fetch an RRef value from its owner rank.
    ///
    /// If the RRef is local, returns the value directly. Otherwise sends
    /// a fetch request to the owner and waits for the response.
    pub fn fetch_rref(&self, rref: &RRef<Vec<u8>>) -> Result<Vec<u8>, RpcError> {
        // Check if value is already cached locally.
        if let Some(val) = rref.local_value() {
            return Ok((*val).clone());
        }

        let owner = rref.owner();

        if owner == self.rank {
            // We own it — look up in the RRef store.
            let data = self
                .rref_store
                .get(rref.id())
                .ok_or(RpcError::RRefNotFound {
                    id: rref.id(),
                    owner,
                })?;
            rref.set_local_value(data.clone());
            return Ok(data);
        }

        // Remote fetch: use rpc_sync with a special built-in function name.
        let id_bytes = rref.id().to_le_bytes();
        let result = self.rpc_sync(owner, "__rref_fetch__", &id_bytes)?;
        rref.set_local_value(result.clone());
        Ok(result)
    }

    /// Register the built-in RRef fetch handler.
    ///
    /// Call this after creating the context to enable remote RRef fetching.
    pub fn register_builtins(self: &Arc<Self>) {
        let store = Arc::clone(&self.rref_store);
        self.register("__rref_fetch__", move |args: Vec<u8>| {
            if args.len() < 8 {
                return Vec::new();
            }
            let id = u64::from_le_bytes(args[0..8].try_into().unwrap());
            store.get(id).unwrap_or_default()
        });
    }
}

// ---------------------------------------------------------------------------
// Message encoding / decoding
// ---------------------------------------------------------------------------

/// Encode an RPC Call message.
///
/// Format: tag(1) + fn_name_len(4 LE) + fn_name + args_len(8 LE) + args + request_id(8 LE)
fn encode_call_message(fn_name: &str, args: &[u8], request_id: u64) -> Vec<u8> {
    let name_bytes = fn_name.as_bytes();
    let name_len = name_bytes.len() as u32;
    let args_len = args.len() as u64;

    let total = 1 + 4 + name_bytes.len() + 8 + args.len() + 8;
    let mut buf = Vec::with_capacity(total);

    buf.push(MessageTag::Call as u8);
    buf.extend_from_slice(&name_len.to_le_bytes());
    buf.extend_from_slice(name_bytes);
    buf.extend_from_slice(&args_len.to_le_bytes());
    buf.extend_from_slice(args);
    buf.extend_from_slice(&request_id.to_le_bytes());

    buf
}

/// Decode an RPC Call message. Returns (fn_name, args, request_id).
fn decode_call_message(data: &[u8]) -> Result<(String, Vec<u8>, u64), RpcError> {
    if data.is_empty() || data[0] != MessageTag::Call as u8 {
        return Err(RpcError::Serialization {
            message: "expected Call message tag".into(),
        });
    }

    let mut offset = 1;

    // fn_name_len
    if offset + 4 > data.len() {
        return Err(RpcError::Serialization {
            message: "truncated Call message (fn_name_len)".into(),
        });
    }
    let name_len = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
    offset += 4;

    // fn_name
    if offset + name_len > data.len() {
        return Err(RpcError::Serialization {
            message: "truncated Call message (fn_name)".into(),
        });
    }
    let fn_name = String::from_utf8(data[offset..offset + name_len].to_vec())
        .map_err(|e| RpcError::Serialization {
            message: format!("invalid UTF-8 in fn_name: {e}"),
        })?;
    offset += name_len;

    // args_len
    if offset + 8 > data.len() {
        return Err(RpcError::Serialization {
            message: "truncated Call message (args_len)".into(),
        });
    }
    let args_len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    // args
    if offset + args_len > data.len() {
        return Err(RpcError::Serialization {
            message: "truncated Call message (args)".into(),
        });
    }
    let args = data[offset..offset + args_len].to_vec();
    offset += args_len;

    // request_id
    if offset + 8 > data.len() {
        return Err(RpcError::Serialization {
            message: "truncated Call message (request_id)".into(),
        });
    }
    let request_id = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());

    Ok((fn_name, args, request_id))
}

/// Encode an RPC Response message.
///
/// Format: tag(1) + request_id(8 LE) + payload_len(8 LE) + payload
fn encode_response_message(request_id: u64, payload: &[u8]) -> Vec<u8> {
    let payload_len = payload.len() as u64;
    let total = 1 + 8 + 8 + payload.len();
    let mut buf = Vec::with_capacity(total);

    buf.push(MessageTag::Response as u8);
    buf.extend_from_slice(&request_id.to_le_bytes());
    buf.extend_from_slice(&payload_len.to_le_bytes());
    buf.extend_from_slice(payload);

    buf
}

/// Decode an RPC Response message. Returns (request_id, payload).
fn decode_response_message(data: &[u8]) -> Result<(u64, Vec<u8>), RpcError> {
    if data.is_empty() || data[0] != MessageTag::Response as u8 {
        return Err(RpcError::Serialization {
            message: "expected Response message tag".into(),
        });
    }

    let mut offset = 1;

    // request_id
    if offset + 8 > data.len() {
        return Err(RpcError::Serialization {
            message: "truncated Response message (request_id)".into(),
        });
    }
    let request_id = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap());
    offset += 8;

    // payload_len
    if offset + 8 > data.len() {
        return Err(RpcError::Serialization {
            message: "truncated Response message (payload_len)".into(),
        });
    }
    let payload_len = u64::from_le_bytes(data[offset..offset + 8].try_into().unwrap()) as usize;
    offset += 8;

    // payload
    if offset + payload_len > data.len() {
        return Err(RpcError::Serialization {
            message: "truncated Response message (payload)".into(),
        });
    }
    let payload = data[offset..offset + payload_len].to_vec();

    Ok((request_id, payload))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    // ---- Message encoding / decoding ----

    #[test]
    fn test_call_message_roundtrip() {
        let fn_name = "add_one";
        let args = vec![1u8, 2, 3, 4];
        let request_id = 42u64;

        let encoded = encode_call_message(fn_name, &args, request_id);
        let (dec_name, dec_args, dec_id) = decode_call_message(&encoded).unwrap();

        assert_eq!(dec_name, fn_name);
        assert_eq!(dec_args, args);
        assert_eq!(dec_id, request_id);
    }

    #[test]
    fn test_response_message_roundtrip() {
        let request_id = 99u64;
        let payload = vec![10u8, 20, 30];

        let encoded = encode_response_message(request_id, &payload);
        let (dec_id, dec_payload) = decode_response_message(&encoded).unwrap();

        assert_eq!(dec_id, request_id);
        assert_eq!(dec_payload, payload);
    }

    #[test]
    fn test_call_message_empty_args() {
        let encoded = encode_call_message("noop", &[], 1);
        let (name, args, id) = decode_call_message(&encoded).unwrap();
        assert_eq!(name, "noop");
        assert!(args.is_empty());
        assert_eq!(id, 1);
    }

    #[test]
    fn test_response_message_empty_payload() {
        let encoded = encode_response_message(7, &[]);
        let (id, payload) = decode_response_message(&encoded).unwrap();
        assert_eq!(id, 7);
        assert!(payload.is_empty());
    }

    #[test]
    fn test_decode_call_invalid_tag() {
        let data = vec![0xFF, 0, 0, 0, 0];
        assert!(decode_call_message(&data).is_err());
    }

    #[test]
    fn test_decode_response_invalid_tag() {
        let data = vec![0xFF, 0, 0, 0, 0];
        assert!(decode_response_message(&data).is_err());
    }

    #[test]
    fn test_decode_call_truncated() {
        // Just the tag, no name_len.
        let data = vec![MessageTag::Call as u8];
        assert!(decode_call_message(&data).is_err());
    }

    #[test]
    fn test_decode_response_truncated() {
        // Just the tag, no request_id.
        let data = vec![MessageTag::Response as u8, 0, 0];
        assert!(decode_response_message(&data).is_err());
    }

    // ---- SimulatedRpcBackend ----

    #[test]
    fn test_simulated_rpc_backend_send_recv() {
        let group = SimulatedRpcBackend::create_group(2).unwrap();
        let mut iter = group.into_iter();
        let b0 = Arc::new(iter.next().unwrap());
        let b1 = Arc::new(iter.next().unwrap());

        let b0c = Arc::clone(&b0);
        let sender = thread::spawn(move || {
            b0c.send(1, &[1, 2, 3, 4]).unwrap();
        });

        let received = b1.recv(0).unwrap();
        sender.join().unwrap();

        assert_eq!(received, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_simulated_rpc_backend_rank_and_world() {
        let group = SimulatedRpcBackend::create_group(3).unwrap();
        assert_eq!(group[0].rank(), 0);
        assert_eq!(group[1].rank(), 1);
        assert_eq!(group[2].rank(), 2);
        assert_eq!(group[0].world_size(), 3);
    }

    #[test]
    fn test_simulated_rpc_backend_invalid_rank() {
        let group = SimulatedRpcBackend::create_group(2).unwrap();
        assert!(group[0].send(5, &[1]).is_err());
    }

    #[test]
    fn test_simulated_rpc_backend_zero_world() {
        assert!(SimulatedRpcBackend::create_group(0).is_err());
    }

    // ---- RRef ----

    #[test]
    fn test_rref_local() {
        let rref = RRef::new_local(0, vec![1u8, 2, 3]);
        assert!(rref.is_local());
        assert_eq!(rref.owner(), 0);
        let val = rref.local_value().unwrap();
        assert_eq!(*val, vec![1, 2, 3]);
    }

    #[test]
    fn test_rref_remote_initially_empty() {
        let rref: RRef<Vec<u8>> = RRef::new_remote(1, 42);
        assert!(!rref.is_local());
        assert_eq!(rref.owner(), 1);
        assert_eq!(rref.id(), 42);
        assert!(rref.local_value().is_none());
    }

    #[test]
    fn test_rref_set_local_value() {
        let rref: RRef<Vec<u8>> = RRef::new_remote(1, 42);
        assert!(!rref.is_local());

        rref.set_local_value(vec![10, 20]);
        assert!(rref.is_local());
        assert_eq!(*rref.local_value().unwrap(), vec![10, 20]);
    }

    #[test]
    fn test_rref_clone() {
        let rref = RRef::new_local(0, 42u32);
        let cloned = rref.clone();
        assert_eq!(cloned.id(), rref.id());
        assert_eq!(cloned.owner(), rref.owner());
    }

    #[test]
    fn test_rref_unique_ids() {
        let r1 = RRef::new_local(0, 1u32);
        let r2 = RRef::new_local(0, 2u32);
        assert_ne!(r1.id(), r2.id());
    }

    // ---- RpcFuture ----

    #[test]
    fn test_rpc_future_resolve_and_wait() {
        let (future, resolver) = RpcFuture::<i32>::new();
        assert!(!future.is_done());

        thread::spawn(move || {
            resolver.resolve(Ok(42));
        });

        let result = future.wait().unwrap();
        assert_eq!(result, 42);
    }

    #[test]
    fn test_rpc_future_resolve_error() {
        let (future, resolver) = RpcFuture::<i32>::new();

        thread::spawn(move || {
            resolver.resolve(Err(RpcError::RemoteError {
                message: "test error".into(),
            }));
        });

        let result = future.wait();
        assert!(result.is_err());
    }

    // ---- RpcContext ----

    #[test]
    fn test_rpc_context_register_and_call_local() {
        let group = SimulatedRpcBackend::create_group(1).unwrap();
        let ctx = RpcContext::new(0, 1, Arc::new(group.into_iter().next().unwrap()));

        ctx.register("double", |args: Vec<u8>| {
            if args.len() >= 4 {
                let val = f32::from_le_bytes(args[0..4].try_into().unwrap());
                (val * 2.0).to_le_bytes().to_vec()
            } else {
                Vec::new()
            }
        });

        let args = 3.0f32.to_le_bytes().to_vec();
        let result = ctx.call_local("double", args).unwrap();
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert!((val - 6.0).abs() < 1e-6);
    }

    #[test]
    fn test_rpc_context_call_local_not_found() {
        let group = SimulatedRpcBackend::create_group(1).unwrap();
        let ctx = RpcContext::new(0, 1, Arc::new(group.into_iter().next().unwrap()));

        let result = ctx.call_local("nonexistent", vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rpc_sync_self_call() {
        // Self-call should invoke locally without network.
        let group = SimulatedRpcBackend::create_group(1).unwrap();
        let ctx = RpcContext::new(0, 1, Arc::new(group.into_iter().next().unwrap()));

        ctx.register("echo", |args: Vec<u8>| args);

        let result = ctx.rpc_sync(0, "echo", &[1, 2, 3]).unwrap();
        assert_eq!(result, vec![1, 2, 3]);
    }

    #[test]
    fn test_rpc_sync_invalid_rank() {
        let group = SimulatedRpcBackend::create_group(1).unwrap();
        let ctx = RpcContext::new(0, 1, Arc::new(group.into_iter().next().unwrap()));

        let result = ctx.rpc_sync(5, "echo", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rpc_sync_two_ranks() {
        // Two ranks: rank 0 calls a function on rank 1.
        // We need a simple protocol where rank 1 acts as a server.
        let group = SimulatedRpcBackend::create_group(2).unwrap();
        let mut iter = group.into_iter();
        let b0 = Arc::new(iter.next().unwrap());
        let b1 = Arc::new(iter.next().unwrap());

        let ctx0 = Arc::new(RpcContext::new(0, 2, b0));
        let ctx1 = Arc::new(RpcContext::new(1, 2, b1));

        // Register "add_ten" on rank 1.
        ctx1.register("add_ten", |args: Vec<u8>| {
            if args.len() >= 4 {
                let val = f32::from_le_bytes(args[0..4].try_into().unwrap());
                (val + 10.0).to_le_bytes().to_vec()
            } else {
                Vec::new()
            }
        });

        // Rank 1 runs a server thread that processes one incoming call.
        let ctx1_clone = Arc::clone(&ctx1);
        let server = thread::spawn(move || {
            // Receive the call from rank 0.
            let msg = ctx1_clone.backend.recv(0).unwrap();
            ctx1_clone.process_call(0, &msg).unwrap();
        });

        // Rank 0 makes a synchronous call to rank 1.
        let args = 5.0f32.to_le_bytes().to_vec();
        let result = ctx0.rpc_sync(1, "add_ten", &args).unwrap();
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert!((val - 15.0).abs() < 1e-6);

        server.join().unwrap();
    }

    #[test]
    fn test_rpc_async_two_ranks() {
        let group = SimulatedRpcBackend::create_group(2).unwrap();
        let mut iter = group.into_iter();
        let b0 = Arc::new(iter.next().unwrap());
        let b1 = Arc::new(iter.next().unwrap());

        let ctx0 = Arc::new(RpcContext::new(0, 2, b0));
        let ctx1 = Arc::new(RpcContext::new(1, 2, b1));

        // Register "square" on rank 1.
        ctx1.register("square", |args: Vec<u8>| {
            if args.len() >= 4 {
                let val = f32::from_le_bytes(args[0..4].try_into().unwrap());
                (val * val).to_le_bytes().to_vec()
            } else {
                Vec::new()
            }
        });

        // Rank 1 server thread.
        let ctx1_clone = Arc::clone(&ctx1);
        let server = thread::spawn(move || {
            let msg = ctx1_clone.backend.recv(0).unwrap();
            ctx1_clone.process_call(0, &msg).unwrap();
        });

        // Rank 0 makes an async call.
        let args = 7.0f32.to_le_bytes().to_vec();
        let future = ctx0.rpc_async(1, "square", &args);

        // Wait for the result.
        let result = future.wait().unwrap();
        let val = f32::from_le_bytes(result[0..4].try_into().unwrap());
        assert!((val - 49.0).abs() < 1e-6);

        server.join().unwrap();
    }

    #[test]
    fn test_remote_self_call() {
        let group = SimulatedRpcBackend::create_group(1).unwrap();
        let ctx = RpcContext::new(0, 1, Arc::new(group.into_iter().next().unwrap()));

        ctx.register("make_vec", |args: Vec<u8>| {
            // Return args repeated 3 times.
            let mut result = Vec::new();
            for _ in 0..3 {
                result.extend_from_slice(&args);
            }
            result
        });

        let rref = ctx.remote(0, "make_vec", &[42]).unwrap();
        assert!(rref.is_local());
        let val = rref.local_value().unwrap();
        assert_eq!(*val, vec![42, 42, 42]);
    }

    #[test]
    fn test_remote_two_ranks() {
        let group = SimulatedRpcBackend::create_group(2).unwrap();
        let mut iter = group.into_iter();
        let b0 = Arc::new(iter.next().unwrap());
        let b1 = Arc::new(iter.next().unwrap());

        let ctx0 = Arc::new(RpcContext::new(0, 2, b0));
        let ctx1 = Arc::new(RpcContext::new(1, 2, b1));

        ctx1.register("concat", |args: Vec<u8>| {
            let mut result = args.clone();
            result.extend_from_slice(&args);
            result
        });

        // Rank 1 server thread.
        let ctx1_clone = Arc::clone(&ctx1);
        let server = thread::spawn(move || {
            let msg = ctx1_clone.backend.recv(0).unwrap();
            ctx1_clone.process_call(0, &msg).unwrap();
        });

        let rref = ctx0.remote(1, "concat", &[1, 2]).unwrap();
        assert!(rref.is_local()); // Value is fetched eagerly.
        let val = rref.local_value().unwrap();
        assert_eq!(*val, vec![1, 2, 1, 2]);
        assert_eq!(rref.owner(), 1);

        server.join().unwrap();
    }

    #[test]
    fn test_remote_invalid_rank() {
        let group = SimulatedRpcBackend::create_group(1).unwrap();
        let ctx = RpcContext::new(0, 1, Arc::new(group.into_iter().next().unwrap()));

        let result = ctx.remote(5, "noop", &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_rref_store_insert_and_get() {
        let store = RRefStore::new();
        store.insert(1, vec![10, 20, 30]);
        store.insert(2, vec![40, 50]);

        assert_eq!(store.get(1), Some(vec![10, 20, 30]));
        assert_eq!(store.get(2), Some(vec![40, 50]));
        assert_eq!(store.get(3), None);
    }

    #[test]
    fn test_message_tag_roundtrip() {
        for tag_val in [1u8, 2, 3, 4, 5] {
            let tag = MessageTag::from_u8(tag_val).unwrap();
            assert_eq!(tag as u8, tag_val);
        }
        assert!(MessageTag::from_u8(0).is_none());
        assert!(MessageTag::from_u8(6).is_none());
        assert!(MessageTag::from_u8(255).is_none());
    }

    #[test]
    fn test_rpc_context_multiple_functions() {
        let group = SimulatedRpcBackend::create_group(1).unwrap();
        let ctx = RpcContext::new(0, 1, Arc::new(group.into_iter().next().unwrap()));

        ctx.register("inc", |args: Vec<u8>| {
            if args.len() >= 4 {
                let val = u32::from_le_bytes(args[0..4].try_into().unwrap());
                (val + 1).to_le_bytes().to_vec()
            } else {
                Vec::new()
            }
        });

        ctx.register("dec", |args: Vec<u8>| {
            if args.len() >= 4 {
                let val = u32::from_le_bytes(args[0..4].try_into().unwrap());
                (val - 1).to_le_bytes().to_vec()
            } else {
                Vec::new()
            }
        });

        let result_inc = ctx.call_local("inc", 10u32.to_le_bytes().to_vec()).unwrap();
        assert_eq!(
            u32::from_le_bytes(result_inc[0..4].try_into().unwrap()),
            11
        );

        let result_dec = ctx.call_local("dec", 10u32.to_le_bytes().to_vec()).unwrap();
        assert_eq!(
            u32::from_le_bytes(result_dec[0..4].try_into().unwrap()),
            9
        );
    }

    #[test]
    fn test_rpc_sync_two_ranks_bidirectional() {
        // Both ranks call functions on each other.
        let group = SimulatedRpcBackend::create_group(2).unwrap();
        let mut iter = group.into_iter();
        let b0 = Arc::new(iter.next().unwrap());
        let b1 = Arc::new(iter.next().unwrap());

        let ctx0 = Arc::new(RpcContext::new(0, 2, b0));
        let ctx1 = Arc::new(RpcContext::new(1, 2, b1));

        // Register on rank 0: negate
        ctx0.register("negate", |args: Vec<u8>| {
            if args.len() >= 4 {
                let val = f32::from_le_bytes(args[0..4].try_into().unwrap());
                (-val).to_le_bytes().to_vec()
            } else {
                Vec::new()
            }
        });

        // Register on rank 1: double
        ctx1.register("double", |args: Vec<u8>| {
            if args.len() >= 4 {
                let val = f32::from_le_bytes(args[0..4].try_into().unwrap());
                (val * 2.0).to_le_bytes().to_vec()
            } else {
                Vec::new()
            }
        });

        // Rank 0 calls rank 1's "double" while rank 1 calls rank 0's "negate".
        let ctx0_clone = Arc::clone(&ctx0);
        let ctx1_clone = Arc::clone(&ctx1);

        let t0 = thread::spawn(move || {
            // First, act as server for rank 1's incoming call.
            let msg = ctx0_clone.backend.recv(1).unwrap();
            ctx0_clone.process_call(1, &msg).unwrap();

            // Then make our own call to rank 1.
            let args = 3.0f32.to_le_bytes().to_vec();
            let result = ctx0_clone.rpc_sync(1, "double", &args).unwrap();
            f32::from_le_bytes(result[0..4].try_into().unwrap())
        });

        let t1 = thread::spawn(move || {
            // First, make our call to rank 0.
            let args = 5.0f32.to_le_bytes().to_vec();
            let result = ctx1_clone.rpc_sync(0, "negate", &args).unwrap();
            let negate_result = f32::from_le_bytes(result[0..4].try_into().unwrap());

            // Then act as server for rank 0's incoming call.
            let msg = ctx1_clone.backend.recv(0).unwrap();
            ctx1_clone.process_call(0, &msg).unwrap();

            negate_result
        });

        let double_result = t0.join().unwrap();
        let negate_result = t1.join().unwrap();

        assert!((double_result - 6.0).abs() < 1e-6);
        assert!((negate_result - (-5.0)).abs() < 1e-6);
    }

    #[test]
    fn test_call_message_long_fn_name() {
        let long_name = "a".repeat(1000);
        let args = vec![0u8; 500];
        let encoded = encode_call_message(&long_name, &args, 123);
        let (name, dec_args, id) = decode_call_message(&encoded).unwrap();
        assert_eq!(name, long_name);
        assert_eq!(dec_args, args);
        assert_eq!(id, 123);
    }

    #[test]
    fn test_response_message_large_payload() {
        let payload = vec![0xABu8; 10_000];
        let encoded = encode_response_message(999, &payload);
        let (id, dec_payload) = decode_response_message(&encoded).unwrap();
        assert_eq!(id, 999);
        assert_eq!(dec_payload, payload);
    }
}
