//! Remote Procedure Call (RPC) framework for distributed training.
//!
//! Provides a simple RPC mechanism built on top of the [`Backend`] transport
//! layer. Workers can register callable functions and invoke them on remote
//! ranks by name.
//!
//! # Architecture
//!
//! - [`RpcAgent`] wraps a [`Backend`] and adds a function registry, request
//!   routing, and response correlation.
//! - [`TcpRpcBackend`] is a thin wrapper around [`TcpBackend`](crate::backend::TcpBackend)
//!   that adds length-prefixed framing for variable-size RPC messages.
//!
//! # Limitations
//!
//! - **`rpc_async` spawns an unbounded number of threads.** Each async RPC
//!   call spawns a new OS thread. This is acceptable for the typical RPC use
//!   case (infrequent coordination calls), but is not suitable for
//!   high-frequency fire-and-forget patterns. A future version may use a
//!   thread pool or async runtime.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use ferrotorch_core::{FerrotorchError, FerrotorchResult};

use crate::backend::Backend;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum allowed RPC message size (1 GiB). Messages exceeding this limit
/// are rejected to prevent out-of-memory conditions from malicious or
/// corrupted length prefixes.
const MAX_RPC_MSG_SIZE: usize = 1 << 30;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to the RPC subsystem.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum RpcError {
    #[error("RPC function not found: {name}")]
    FunctionNotFound { name: String },

    #[error("invalid RPC message: {reason}")]
    InvalidMessage { reason: String },

    #[error("no connection to rank {rank} (star topology: non-zero ranks only connect to rank 0)")]
    NoConnection { rank: usize },

    #[error("RPC internal error: {0}")]
    Internal(String),

    #[error("RPC timeout")]
    Timeout,
}

impl From<RpcError> for FerrotorchError {
    fn from(e: RpcError) -> Self {
        FerrotorchError::InvalidArgument {
            message: e.to_string(),
        }
    }
}

// ---------------------------------------------------------------------------
// RPC message types
// ---------------------------------------------------------------------------

/// A serialized RPC request.
#[derive(Debug, Clone)]
struct RpcRequest {
    /// Unique identifier for correlating responses.
    request_id: u64,
    /// Name of the remote function to call.
    function_name: String,
    /// Serialized arguments (opaque bytes).
    payload: Vec<u8>,
}

/// A serialized RPC response.
#[derive(Debug, Clone)]
struct RpcResponse {
    /// The request_id this response is for.
    request_id: u64,
    /// Serialized return value (opaque bytes).
    payload: Vec<u8>,
    /// Error message, if any.
    error: Option<String>,
}

impl RpcRequest {
    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Tag byte: 0x01 = request
        buf.push(0x01);
        buf.extend_from_slice(&self.request_id.to_le_bytes());
        let name_bytes = self.function_name.as_bytes();
        buf.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(name_bytes);
        buf.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
        buf.extend_from_slice(&self.payload);
        buf
    }

    fn deserialize(data: &[u8]) -> Result<Self, RpcError> {
        if data.is_empty() || data[0] != 0x01 {
            return Err(RpcError::InvalidMessage {
                reason: "expected request tag 0x01".into(),
            });
        }
        let mut pos = 1;
        if data.len() < pos + 8 {
            return Err(RpcError::InvalidMessage {
                reason: "request too short for request_id".into(),
            });
        }
        let request_id = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        if data.len() < pos + 4 {
            return Err(RpcError::InvalidMessage {
                reason: "request too short for name length".into(),
            });
        }
        let name_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if data.len() < pos + name_len {
            return Err(RpcError::InvalidMessage {
                reason: "request too short for function name".into(),
            });
        }
        let function_name = String::from_utf8(data[pos..pos + name_len].to_vec()).map_err(|e| {
            RpcError::InvalidMessage {
                reason: format!("invalid UTF-8 in function name: {e}"),
            }
        })?;
        pos += name_len;

        if data.len() < pos + 4 {
            return Err(RpcError::InvalidMessage {
                reason: "request too short for payload length".into(),
            });
        }
        let payload_len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if data.len() < pos + payload_len {
            return Err(RpcError::InvalidMessage {
                reason: "request too short for payload".into(),
            });
        }
        let payload = data[pos..pos + payload_len].to_vec();

        Ok(Self {
            request_id,
            function_name,
            payload,
        })
    }
}

impl RpcResponse {
    fn serialize(&self) -> Vec<u8> {
        let mut buf = Vec::new();
        // Tag byte: 0x02 = response
        buf.push(0x02);
        buf.extend_from_slice(&self.request_id.to_le_bytes());
        if let Some(err) = &self.error {
            buf.push(0x01); // has error
            let err_bytes = err.as_bytes();
            buf.extend_from_slice(&(err_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(err_bytes);
        } else {
            buf.push(0x00); // no error
            buf.extend_from_slice(&(self.payload.len() as u32).to_le_bytes());
            buf.extend_from_slice(&self.payload);
        }
        buf
    }

    fn deserialize(data: &[u8]) -> Result<Self, RpcError> {
        if data.is_empty() || data[0] != 0x02 {
            return Err(RpcError::InvalidMessage {
                reason: "expected response tag 0x02".into(),
            });
        }
        let mut pos = 1;
        if data.len() < pos + 8 {
            return Err(RpcError::InvalidMessage {
                reason: "response too short for request_id".into(),
            });
        }
        let request_id = u64::from_le_bytes(data[pos..pos + 8].try_into().unwrap());
        pos += 8;

        if data.len() < pos + 1 {
            return Err(RpcError::InvalidMessage {
                reason: "response too short for error flag".into(),
            });
        }
        let has_error = data[pos] == 0x01;
        pos += 1;

        if data.len() < pos + 4 {
            return Err(RpcError::InvalidMessage {
                reason: "response too short for payload/error length".into(),
            });
        }
        let len = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap()) as usize;
        pos += 4;

        if data.len() < pos + len {
            return Err(RpcError::InvalidMessage {
                reason: "response too short for payload/error data".into(),
            });
        }

        if has_error {
            let error_msg = String::from_utf8(data[pos..pos + len].to_vec()).map_err(|e| {
                RpcError::InvalidMessage {
                    reason: format!("invalid UTF-8 in error message: {e}"),
                }
            })?;
            Ok(Self {
                request_id,
                payload: Vec::new(),
                error: Some(error_msg),
            })
        } else {
            Ok(Self {
                request_id,
                payload: data[pos..pos + len].to_vec(),
                error: None,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// TCP RPC Backend
// ---------------------------------------------------------------------------

/// TCP-based RPC transport built on [`TcpBackend`](crate::backend::TcpBackend).
///
/// # Topology limitation
///
/// `TcpRpcBackend` inherits the **star topology** from `TcpBackend`: non-zero
/// ranks only have a direct TCP connection to rank 0. This means:
///
/// - **Rank 0 can send/recv RPC messages to/from any rank.**
/// - **Non-zero ranks can only send/recv RPC messages to/from rank 0.**
/// - **Direct rank-to-rank RPC between two non-zero ranks (e.g., rank 1 to
///   rank 2) will fail** with [`RpcError::NoConnection`].
///
/// If rank-to-rank RPC is needed, implement a relay through rank 0 or use a
/// full-mesh backend. This is a known limitation of the current TCP transport.
pub struct TcpRpcBackend {
    backend: Arc<dyn Backend>,
}

impl TcpRpcBackend {
    /// Create a new TCP RPC backend wrapping an existing [`Backend`].
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self { backend }
    }

    /// Send a raw RPC message to `dst_rank`.
    ///
    /// # Errors
    ///
    /// Returns [`RpcError::NoConnection`] if there is no direct connection
    /// to `dst_rank` (star topology: non-zero ranks can only reach rank 0).
    pub fn send(&self, data: &[u8], dst_rank: usize) -> FerrotorchResult<()> {
        self.backend.send(data, dst_rank).map_err(|e| {
            let msg = e.to_string();
            if msg.contains("no connection") || msg.contains("NoConnection") {
                RpcError::NoConnection { rank: dst_rank }.into()
            } else {
                e
            }
        })
    }

    /// Receive a raw RPC message from `src_rank`.
    ///
    /// Enforces [`MAX_RPC_MSG_SIZE`] to prevent OOM from malicious or
    /// corrupted length prefixes.
    ///
    /// # Errors
    ///
    /// Returns [`RpcError::NoConnection`] if there is no direct connection
    /// to `src_rank`.
    /// Returns [`RpcError::InvalidMessage`] if the message exceeds
    /// [`MAX_RPC_MSG_SIZE`].
    pub fn recv(&self, dst: &mut [u8], src_rank: usize) -> FerrotorchResult<()> {
        if dst.len() > MAX_RPC_MSG_SIZE {
            return Err(RpcError::InvalidMessage {
                reason: format!(
                    "RPC message size {} exceeds maximum allowed size {} (1 GiB)",
                    dst.len(),
                    MAX_RPC_MSG_SIZE
                ),
            }
            .into());
        }
        self.backend.recv(dst, src_rank).map_err(|e| {
            let msg = e.to_string();
            if msg.contains("no connection") || msg.contains("NoConnection") {
                RpcError::NoConnection { rank: src_rank }.into()
            } else {
                e
            }
        })
    }

    /// The rank of this backend.
    pub fn rank(&self) -> usize {
        self.backend.rank()
    }

    /// The world size.
    pub fn world_size(&self) -> usize {
        self.backend.world_size()
    }
}

// ---------------------------------------------------------------------------
// RPC Agent
// ---------------------------------------------------------------------------

/// Type-erased RPC handler function.
///
/// Takes serialized arguments and returns serialized result (or error).
type RpcHandler = Box<dyn Fn(&[u8]) -> Result<Vec<u8>, String> + Send + Sync>;

/// RPC agent that manages function registration and remote invocation.
///
/// Each rank creates an `RpcAgent` wrapping a [`Backend`]. Functions are
/// registered with [`register`] and invoked remotely with [`rpc_sync`].
///
/// # Response correlation
///
/// Concurrent `rpc_sync` calls are correlated by `request_id`. If a received
/// response has a different `request_id` than expected, it is buffered and
/// the agent retries the recv. Buffered responses are checked before
/// issuing new recv calls.
pub struct RpcAgent {
    backend: Arc<dyn Backend>,
    registry: Mutex<HashMap<String, Arc<RpcHandler>>>,
    next_request_id: Mutex<u64>,
    /// Buffered responses from out-of-order receives, keyed by request_id.
    buffered_responses: Mutex<HashMap<u64, RpcResponse>>,
}

impl RpcAgent {
    /// Create a new RPC agent.
    pub fn new(backend: Arc<dyn Backend>) -> Self {
        Self {
            backend,
            registry: Mutex::new(HashMap::new()),
            next_request_id: Mutex::new(1),
            buffered_responses: Mutex::new(HashMap::new()),
        }
    }

    /// Register a callable function.
    ///
    /// The handler receives serialized arguments and must return serialized
    /// results. If the registry lock is poisoned, this recovers the inner
    /// data and continues.
    pub fn register<F>(&self, name: &str, handler: F)
    where
        F: Fn(&[u8]) -> Result<Vec<u8>, String> + Send + Sync + 'static,
    {
        let mut reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        reg.insert(name.to_string(), Arc::new(Box::new(handler)));
    }

    /// Look up a registered function by name.
    fn lookup(&self, name: &str) -> Option<Arc<RpcHandler>> {
        let reg = self.registry.lock().unwrap_or_else(|e| e.into_inner());
        reg.get(name).cloned()
    }

    /// Allocate a new unique request ID.
    fn next_id(&self) -> u64 {
        let mut id = self
            .next_request_id
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        let current = *id;
        *id += 1;
        current
    }

    /// Invoke a function on a remote rank synchronously.
    ///
    /// Sends the request, then waits for a response with the matching
    /// `request_id`. If a response for a different request is received,
    /// it is buffered for later retrieval.
    ///
    /// # Errors
    ///
    /// Returns an error if the remote function is not found, if the remote
    /// handler returns an error, or if communication fails.
    pub fn rpc_sync(
        &self,
        dst_rank: usize,
        function_name: &str,
        args: &[u8],
    ) -> FerrotorchResult<Vec<u8>> {
        let request_id = self.next_id();
        let request = RpcRequest {
            request_id,
            function_name: function_name.to_string(),
            payload: args.to_vec(),
        };

        let serialized = request.serialize();
        self.backend.send(&serialized, dst_rank)?;

        // Wait for the response with the matching request_id.
        self.recv_response(dst_rank, request_id)
    }

    /// Receive a response matching the given `request_id` from `src_rank`.
    ///
    /// Checks the buffer first. If the response is not buffered, receives
    /// messages until the matching one arrives, buffering any non-matching
    /// responses along the way.
    fn recv_response(&self, src_rank: usize, expected_id: u64) -> FerrotorchResult<Vec<u8>> {
        // Check buffer first.
        {
            let mut buf = self
                .buffered_responses
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            if let Some(resp) = buf.remove(&expected_id) {
                return self.process_response(resp);
            }
        }

        // Receive until we get the right response.
        loop {
            // Receive raw message. We allocate a maximum-size buffer for
            // the receive (we don't know the size ahead of time with the
            // Backend trait).
            let mut len_buf = [0u8; 8];
            // For simplicity with the Backend trait (which requires
            // pre-allocated buffers), we serialize the response with a
            // length prefix on the wire. But the Backend itself uses
            // length-prefixed messages too. So we receive the full
            // serialized response as a single message.
            //
            // For now, receive a reasonably-sized buffer. In practice the
            // send side sends the full serialized response as one Backend
            // message.
            let _ = len_buf; // unused — Backend handles framing

            // We need a different approach: the sender sends the full
            // serialized response via backend.send(), so we need to know
            // the size to allocate. Use a two-phase protocol:
            // Phase 1: receive 8-byte length prefix
            self.backend.recv(&mut len_buf, src_rank)?;
            let msg_len = u64::from_le_bytes(len_buf) as usize;

            if msg_len > MAX_RPC_MSG_SIZE {
                return Err(RpcError::InvalidMessage {
                    reason: format!(
                        "RPC response size {} exceeds maximum {} (1 GiB)",
                        msg_len, MAX_RPC_MSG_SIZE
                    ),
                }
                .into());
            }

            // Phase 2: receive the actual message
            let mut msg_buf = vec![0u8; msg_len];
            self.backend.recv(&mut msg_buf, src_rank)?;

            let response = RpcResponse::deserialize(&msg_buf).map_err(|e| {
                FerrotorchError::InvalidArgument {
                    message: format!("failed to deserialize RPC response: {e}"),
                }
            })?;

            if response.request_id == expected_id {
                return self.process_response(response);
            }

            // Buffer the non-matching response.
            let mut buf = self
                .buffered_responses
                .lock()
                .unwrap_or_else(|e| e.into_inner());
            buf.insert(response.request_id, response);
        }
    }

    /// Process a received response, converting errors.
    fn process_response(&self, response: RpcResponse) -> FerrotorchResult<Vec<u8>> {
        if let Some(err) = response.error {
            Err(FerrotorchError::InvalidArgument {
                message: format!("remote RPC error: {err}"),
            })
        } else {
            Ok(response.payload)
        }
    }

    /// Invoke a function on a remote rank asynchronously.
    ///
    /// Spawns a thread to perform the RPC call. Returns a join handle
    /// that can be used to retrieve the result.
    ///
    /// # Limitations
    ///
    /// **Spawns an unbounded number of OS threads.** Each call to `rpc_async`
    /// creates a new thread. This is acceptable for infrequent coordination
    /// RPCs but is not suitable for high-frequency patterns. A thread pool
    /// or async runtime would be needed for that use case.
    pub fn rpc_async(
        self: &Arc<Self>,
        dst_rank: usize,
        function_name: &str,
        args: &[u8],
    ) -> std::thread::JoinHandle<FerrotorchResult<Vec<u8>>> {
        let agent = Arc::clone(self);
        let name = function_name.to_string();
        let args = args.to_vec();
        std::thread::spawn(move || agent.rpc_sync(dst_rank, &name, &args))
    }

    /// Handle an incoming RPC request: look up the function, call it, and
    /// send the response back.
    pub fn handle_request(&self, src_rank: usize, request_data: &[u8]) -> FerrotorchResult<()> {
        let request = RpcRequest::deserialize(request_data).map_err(|e| {
            FerrotorchError::InvalidArgument {
                message: format!("failed to deserialize RPC request: {e}"),
            }
        })?;

        let response = match self.lookup(&request.function_name) {
            Some(handler) => match handler(&request.payload) {
                Ok(result) => RpcResponse {
                    request_id: request.request_id,
                    payload: result,
                    error: None,
                },
                Err(err) => RpcResponse {
                    request_id: request.request_id,
                    payload: Vec::new(),
                    error: Some(err),
                },
            },
            None => RpcResponse {
                request_id: request.request_id,
                payload: Vec::new(),
                error: Some(format!(
                    "function '{}' not registered on rank {}",
                    request.function_name,
                    self.backend.rank()
                )),
            },
        };

        // Send response with length prefix.
        let serialized = response.serialize();
        let len_bytes = (serialized.len() as u64).to_le_bytes();
        self.backend.send(&len_bytes, src_rank)?;
        self.backend.send(&serialized, src_rank)?;

        Ok(())
    }

    /// The rank of this agent.
    pub fn rank(&self) -> usize {
        self.backend.rank()
    }

    /// The world size.
    pub fn world_size(&self) -> usize {
        self.backend.world_size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rpc_request_roundtrip() {
        let req = RpcRequest {
            request_id: 42,
            function_name: "add".to_string(),
            payload: vec![1, 2, 3],
        };
        let bytes = req.serialize();
        let req2 = RpcRequest::deserialize(&bytes).unwrap();
        assert_eq!(req2.request_id, 42);
        assert_eq!(req2.function_name, "add");
        assert_eq!(req2.payload, vec![1, 2, 3]);
    }

    #[test]
    fn test_rpc_response_roundtrip_ok() {
        let resp = RpcResponse {
            request_id: 7,
            payload: vec![4, 5, 6],
            error: None,
        };
        let bytes = resp.serialize();
        let resp2 = RpcResponse::deserialize(&bytes).unwrap();
        assert_eq!(resp2.request_id, 7);
        assert_eq!(resp2.payload, vec![4, 5, 6]);
        assert!(resp2.error.is_none());
    }

    #[test]
    fn test_rpc_response_roundtrip_error() {
        let resp = RpcResponse {
            request_id: 99,
            payload: Vec::new(),
            error: Some("something went wrong".into()),
        };
        let bytes = resp.serialize();
        let resp2 = RpcResponse::deserialize(&bytes).unwrap();
        assert_eq!(resp2.request_id, 99);
        assert_eq!(resp2.error.unwrap(), "something went wrong");
    }

    #[test]
    fn test_max_message_size_constant() {
        assert_eq!(MAX_RPC_MSG_SIZE, 1 << 30);
    }

    #[test]
    fn test_rpc_agent_register_lookup() {
        use crate::backend::SimulatedBackend;

        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let agent = RpcAgent::new(b);

        agent.register("echo", |args| Ok(args.to_vec()));

        let handler = agent.lookup("echo");
        assert!(handler.is_some());

        let result = handler.unwrap()(b"hello");
        assert_eq!(result.unwrap(), b"hello");
    }

    #[test]
    fn test_rpc_agent_lookup_missing() {
        use crate::backend::SimulatedBackend;

        let group = SimulatedBackend::create_group(1).unwrap();
        let b: Arc<dyn Backend> = Arc::new(group.into_iter().next().unwrap());
        let agent = RpcAgent::new(b);

        assert!(agent.lookup("nonexistent").is_none());
    }
}
