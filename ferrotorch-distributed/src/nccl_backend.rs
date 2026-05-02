//! NCCL-backed distributed backend for GPU-native collective communication.
//!
//! [`NcclBackend`] implements the [`Backend`] trait using NCCL point-to-point
//! operations (`ncclSend`/`ncclRecv`), and provides additional GPU-native
//! collective methods that operate directly on device memory without CPU
//! round-trips.
//!
//! # Initialization
//!
//! Rank 0 generates a unique ID via [`crate::nccl_sys::get_unique_id`] and
//! distributes it to all other ranks (e.g. via TCP or environment variable).
//! Each rank then calls [`NcclBackend::new`] with its rank, world_size, and
//! the shared unique ID.
//!
//! # Feature gate
//!
//! This module requires the `nccl` feature:
//! ```toml
//! ferrotorch-distributed = { version = "0.2", features = ["nccl"] }
//! ```

use std::ffi::c_void;
use std::sync::Mutex;
use std::time::Duration;

use ferrotorch_core::FerrotorchResult;

use crate::backend::Backend;
use crate::collective::ReduceOp;
use crate::error::DistributedError;
use crate::nccl_sys::{self, NcclComm, NcclDataType, NcclRedOp, NcclUniqueId};

// ---------------------------------------------------------------------------
// NcclBackend
// ---------------------------------------------------------------------------

/// NCCL-backed distributed backend.
///
/// Wraps an NCCL communicator and provides both the generic [`Backend`]
/// trait (via point-to-point send/recv) and GPU-native collective operations
/// that bypass CPU entirely.
///
/// NCCL operations run on a dedicated CUDA stream to allow overlap with
/// computation on the default stream. Call [`synchronize`](Self::synchronize)
/// to ensure all NCCL operations have completed before reading results.
pub struct NcclBackend {
    comm: Mutex<NcclComm>,
    rank: usize,
    world_size: usize,
    /// Dedicated CUDA stream for NCCL operations. Using a separate stream
    /// from the compute stream allows communication to overlap with
    /// computation (e.g., allreduce of layer N's gradients while computing
    /// layer N-1's backward pass).
    ///
    /// If null, uses the default stream (serialized with compute).
    stream: *mut c_void,
    /// Whether we own the stream (and should destroy it on Drop).
    owns_stream: bool,
}

// SAFETY: NcclComm is thread-safe when protected by a mutex.
// The stream pointer is only used within NCCL calls under the mutex.
unsafe impl Send for NcclBackend {}
unsafe impl Sync for NcclBackend {}

impl NcclBackend {
    /// Create a new NCCL backend.
    ///
    /// `rank` is this process's rank (0-based). `world_size` is the total
    /// number of ranks. `unique_id` must be the same on all ranks — rank 0
    /// should generate it with [`nccl_sys::get_unique_id`] and distribute
    /// it to others.
    ///
    /// The correct CUDA device must be set (`cudaSetDevice`) before calling.
    ///
    /// # Errors
    ///
    /// Returns an error if NCCL is not available or communicator init fails.
    pub fn new(rank: usize, world_size: usize, unique_id: NcclUniqueId) -> FerrotorchResult<Self> {
        let comm =
            nccl_sys::comm_init_rank(world_size as i32, rank as i32, unique_id).map_err(|e| {
                DistributedError::Io {
                    message: format!("NCCL comm_init_rank failed: {e}"),
                }
            })?;

        // Create a dedicated CUDA stream for NCCL operations.
        // This allows communication to overlap with computation on the
        // default stream.
        let stream = create_nccl_stream().unwrap_or(std::ptr::null_mut());
        let owns_stream = !stream.is_null();

        Ok(Self {
            comm: Mutex::new(comm),
            rank,
            world_size,
            stream,
            owns_stream,
        })
    }

    /// Create an NCCL backend using a caller-provided CUDA stream.
    ///
    /// The caller retains ownership of the stream — it will NOT be
    /// destroyed when the backend is dropped.
    pub fn with_stream(
        rank: usize,
        world_size: usize,
        unique_id: NcclUniqueId,
        stream: *mut c_void,
    ) -> FerrotorchResult<Self> {
        let comm =
            nccl_sys::comm_init_rank(world_size as i32, rank as i32, unique_id).map_err(|e| {
                DistributedError::Io {
                    message: format!("NCCL comm_init_rank failed: {e}"),
                }
            })?;

        Ok(Self {
            comm: Mutex::new(comm),
            rank,
            world_size,
            stream,
            owns_stream: false,
        })
    }

    /// Synchronize the NCCL stream — blocks until all enqueued NCCL
    /// operations have completed.
    ///
    /// Call this before reading GPU buffers that were modified by NCCL
    /// collective operations.
    pub fn synchronize(&self) -> FerrotorchResult<()> {
        if self.stream.is_null() {
            return Ok(()); // default stream is implicitly synchronized
        }
        synchronize_stream(self.stream).map_err(|msg| DistributedError::Io { message: msg }.into())
    }

    /// Get the raw NCCL communicator handle (for advanced use).
    pub fn comm(&self) -> &Mutex<NcclComm> {
        &self.comm
    }

    fn lock_comm(&self) -> FerrotorchResult<std::sync::MutexGuard<'_, NcclComm>> {
        self.comm.lock().map_err(|_| {
            DistributedError::LockPoisoned {
                message: "NCCL communicator mutex poisoned".into(),
            }
            .into()
        })
    }

    // -----------------------------------------------------------------------
    // GPU-native collective operations (no CPU round-trip)
    // -----------------------------------------------------------------------

    /// GPU all-reduce: reduces `count` elements in-place on device memory.
    ///
    /// `sendbuf` and `recvbuf` are raw CUDA device pointers. They may be
    /// the same pointer for in-place operation.
    ///
    /// # Safety
    ///
    /// Pointers must be valid device memory of the correct size.
    pub unsafe fn allreduce_raw(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
    ) -> FerrotorchResult<()> {
        let comm = *self.lock_comm()?;
        nccl_sys::all_reduce(sendbuf, recvbuf, count, datatype, op, comm, self.stream).map_err(
            |e| {
                DistributedError::Io {
                    message: format!("NCCL allreduce failed: {e}"),
                }
                .into()
            },
        )
    }

    /// GPU broadcast: broadcast `count` elements from `root` to all ranks.
    ///
    /// # Safety
    ///
    /// Pointers must be valid device memory.
    pub unsafe fn broadcast_raw(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        root: i32,
    ) -> FerrotorchResult<()> {
        let comm = *self.lock_comm()?;
        nccl_sys::broadcast(sendbuf, recvbuf, count, datatype, root, comm, self.stream).map_err(
            |e| {
                DistributedError::Io {
                    message: format!("NCCL broadcast failed: {e}"),
                }
                .into()
            },
        )
    }

    /// GPU all-gather: each rank sends `sendcount` elements, receives
    /// `sendcount * world_size` elements.
    ///
    /// # Safety
    ///
    /// Pointers must be valid device memory.
    pub unsafe fn all_gather_raw(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        sendcount: usize,
        datatype: NcclDataType,
    ) -> FerrotorchResult<()> {
        let comm = *self.lock_comm()?;
        nccl_sys::all_gather(sendbuf, recvbuf, sendcount, datatype, comm, self.stream).map_err(
            |e| {
                DistributedError::Io {
                    message: format!("NCCL all_gather failed: {e}"),
                }
                .into()
            },
        )
    }

    /// GPU reduce-scatter: reduces then distributes `recvcount` elements
    /// to each rank.
    ///
    /// # Safety
    ///
    /// Pointers must be valid device memory.
    pub unsafe fn reduce_scatter_raw(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        recvcount: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
    ) -> FerrotorchResult<()> {
        let comm = *self.lock_comm()?;
        nccl_sys::reduce_scatter(sendbuf, recvbuf, recvcount, datatype, op, comm, self.stream)
            .map_err(|e| {
                DistributedError::Io {
                    message: format!("NCCL reduce_scatter failed: {e}"),
                }
                .into()
            })
    }
}

// ---------------------------------------------------------------------------
// Backend trait implementation
// ---------------------------------------------------------------------------
//
// The Backend trait is byte-oriented P2P (send/recv). NCCL's primary value
// is collective operations on GPU buffers (allreduce, broadcast, etc.) which
// are exposed via the raw methods above and the nccl_collective module.
//
// For the P2P Backend trait, we return UnsupportedOp — users should use
// the NcclBackend's native collective methods or pair it with a TcpBackend
// for P2P communication. This matches PyTorch's ProcessGroupNCCL which also
// delegates non-collective ops to Gloo/TCP.

impl Backend for NcclBackend {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.world_size
    }

    fn send(&self, _data: &[u8], _dst_rank: usize) -> FerrotorchResult<()> {
        Err(DistributedError::UnsupportedOp {
            message: "NcclBackend does not support byte-level P2P send — use GPU-native collectives (nccl_allreduce, nccl_broadcast) or TcpBackend for P2P".into(),
        }.into())
    }

    fn recv(&self, _dst: &mut [u8], _src_rank: usize) -> FerrotorchResult<()> {
        Err(DistributedError::UnsupportedOp {
            message: "NcclBackend does not support byte-level P2P recv — use GPU-native collectives or TcpBackend for P2P".into(),
        }.into())
    }

    fn barrier(&self) -> FerrotorchResult<()> {
        // NCCL barrier = all-reduce of a single dummy element.
        // We use the raw allreduce with a 1-byte GPU buffer.
        // This requires a GPU buffer — allocate a tiny one.
        let comm = *self.lock_comm()?;

        // Use a null pointer with count=0 as a lightweight barrier.
        // NCCL allReduce with count=0 is a valid synchronization point.
        unsafe {
            nccl_sys::all_reduce(
                std::ptr::null(),
                std::ptr::null_mut(),
                0,
                NcclDataType::Float32,
                NcclRedOp::Sum,
                comm,
                self.stream,
            )
            .map_err(|e| DistributedError::Io {
                message: format!("NCCL barrier (allreduce): {e}"),
            })?;
        }

        Ok(())
    }
}

impl Drop for NcclBackend {
    fn drop(&mut self) {
        if let Ok(comm) = self.comm.lock() {
            if !(*comm).is_null() {
                unsafe {
                    let _ = nccl_sys::comm_destroy(*comm);
                }
            }
        }
        if self.owns_stream && !self.stream.is_null() {
            destroy_stream(self.stream);
        }
    }
}

// ---------------------------------------------------------------------------
// CUDA stream helpers (via dlopen to avoid compile-time CUDA dependency)
// ---------------------------------------------------------------------------

/// Create a new CUDA stream for NCCL operations.
/// Returns null on failure (caller falls back to default stream).
fn create_nccl_stream() -> Option<*mut c_void> {
    // cuStreamCreate via dlopen
    let lib = unsafe { libc::dlopen(b"libcudart.so.12\0".as_ptr() as *const _, libc::RTLD_LAZY) };
    if lib.is_null() {
        let lib = unsafe { libc::dlopen(b"libcudart.so\0".as_ptr() as *const _, libc::RTLD_LAZY) };
        if lib.is_null() {
            return None;
        }
        return create_stream_from_lib(lib);
    }
    create_stream_from_lib(lib)
}

fn create_stream_from_lib(lib: *mut c_void) -> Option<*mut c_void> {
    let sym = unsafe { libc::dlsym(lib, b"cudaStreamCreateWithFlags\0".as_ptr() as *const _) };
    if sym.is_null() {
        return None;
    }
    // cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)
    // flags = 1 = cudaStreamNonBlocking
    type CudaStreamCreateFn = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
    let create_fn: CudaStreamCreateFn = unsafe { std::mem::transmute(sym) };
    let mut stream: *mut c_void = std::ptr::null_mut();
    let result = unsafe { create_fn(&mut stream, 1) }; // 1 = cudaStreamNonBlocking
    if result == 0 { Some(stream) } else { None }
}

/// Synchronize a CUDA stream (blocks until all operations complete).
fn synchronize_stream(stream: *mut c_void) -> Result<(), String> {
    let lib = unsafe { libc::dlopen(b"libcudart.so.12\0".as_ptr() as *const _, libc::RTLD_LAZY) };
    let lib = if lib.is_null() {
        unsafe { libc::dlopen(b"libcudart.so\0".as_ptr() as *const _, libc::RTLD_LAZY) }
    } else {
        lib
    };
    if lib.is_null() {
        return Err("cudart not found".into());
    }
    let sym = unsafe { libc::dlsym(lib, b"cudaStreamSynchronize\0".as_ptr() as *const _) };
    if sym.is_null() {
        return Err("cudaStreamSynchronize not found".into());
    }
    type SyncFn = unsafe extern "C" fn(*mut c_void) -> i32;
    let sync_fn: SyncFn = unsafe { std::mem::transmute(sym) };
    let result = unsafe { sync_fn(stream) };
    if result == 0 {
        Ok(())
    } else {
        Err(format!("cudaStreamSynchronize failed: error {result}"))
    }
}

/// Destroy a CUDA stream.
fn destroy_stream(stream: *mut c_void) {
    let lib = unsafe { libc::dlopen(b"libcudart.so.12\0".as_ptr() as *const _, libc::RTLD_LAZY) };
    let lib = if lib.is_null() {
        unsafe { libc::dlopen(b"libcudart.so\0".as_ptr() as *const _, libc::RTLD_LAZY) }
    } else {
        lib
    };
    if lib.is_null() {
        return;
    }
    let sym = unsafe { libc::dlsym(lib, b"cudaStreamDestroy\0".as_ptr() as *const _) };
    if sym.is_null() {
        return;
    }
    type DestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
    let destroy_fn: DestroyFn = unsafe { std::mem::transmute(sym) };
    unsafe { destroy_fn(stream) };
}

// ---------------------------------------------------------------------------
// Helper: convert ReduceOp to NCCL
// ---------------------------------------------------------------------------

/// Convert our [`ReduceOp`] to NCCL's reduction operation enum.
pub fn reduce_op_to_nccl(op: &ReduceOp) -> NcclRedOp {
    match op {
        ReduceOp::Sum => NcclRedOp::Sum,
        ReduceOp::Mean => NcclRedOp::Avg,
    }
}

/// Check if NCCL is available on this system.
pub fn is_nccl_available() -> bool {
    nccl_sys::is_available()
}
