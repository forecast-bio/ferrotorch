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
    /// The caller MUST uphold all of the following:
    ///
    /// - `sendbuf` is a valid CUDA device pointer with at least
    ///   `count * size_of(datatype)` bytes addressable for read.
    /// - `recvbuf` is a valid CUDA device pointer with at least
    ///   `count * size_of(datatype)` bytes addressable for write.
    /// - In-place mode (`sendbuf == recvbuf`) is allowed and matches
    ///   NCCL's documented behaviour.
    /// - `datatype` matches the actual element layout of both buffers.
    /// - Every other rank in this NCCL communicator invokes
    ///   `allreduce_raw` with the same `count`, `datatype`, and `op`
    ///   (NCCL collective contract; cannot be checked in-process).
    /// - The current CUDA device matches the device on which the
    ///   communicator was initialised (`comm_init_rank` runs on the
    ///   active device).
    /// - The buffers remain alive and unmodified until the NCCL
    ///   stream is synchronised — the call is asynchronous on
    ///   `self.stream`. Use [`Self::synchronize`] to await
    ///   completion before reading `recvbuf` from another stream.
    pub unsafe fn allreduce_raw(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
    ) -> FerrotorchResult<()> {
        let comm = *self.lock_comm()?;
        // SAFETY: this `pub unsafe fn`'s `# Safety` rustdoc enumerates the
        // caller's obligations; each maps to an `nccl_sys::all_reduce`
        // precondition as follows:
        //   - "sendbuf valid for read of `count * size_of(datatype)`",
        //     "recvbuf valid for write of same", "in-place allowed",
        //     "buffers alive until stream synchronisation"
        //                          → ncclAllReduce buffer-validity contract;
        //   - "datatype matches both buffers' element layout"
        //                          → ncclAllReduce datatype/buffer pairing;
        //   - "every other rank invokes allreduce_raw with same count,
        //     datatype, op"        → NCCL collective contract (cross-rank);
        //   - "current CUDA device matches the comm's device"
        //                          → ncclAllReduce device-binding precondition.
        //
        // `comm` was obtained one line above by dereferencing the
        // `MutexGuard<NcclComm>` from `lock_comm()?`. The guard is dropped
        // at the end of this function (the `*` copies the `*mut c_void`
        // handle out of the lock); since we are holding *the only* guard
        // until the call returns, `comm` is the live, post-init handle and
        // no concurrent destroy can run on this side. `self.stream` is
        // either null (default stream — always valid) or a stream created
        // via `create_nccl_stream` in `Self::new` and owned by `self`,
        // alive for the lifetime of `&self`.
        unsafe { nccl_sys::all_reduce(sendbuf, recvbuf, count, datatype, op, comm, self.stream) }
            .map_err(|e| {
                DistributedError::Io {
                    message: format!("NCCL allreduce failed: {e}"),
                }
                .into()
            })
    }

    /// GPU broadcast: broadcast `count` elements from `root` to all ranks.
    ///
    /// # Safety
    ///
    /// The caller MUST uphold all of the following:
    ///
    /// - `sendbuf` is a valid CUDA device pointer with at least
    ///   `count * size_of(datatype)` bytes addressable for read on the
    ///   `root` rank. On non-root ranks `sendbuf` is ignored by NCCL.
    /// - `recvbuf` is a valid CUDA device pointer with at least
    ///   `count * size_of(datatype)` bytes addressable for write on
    ///   every non-root rank. On the root rank `recvbuf` may equal
    ///   `sendbuf` for in-place operation.
    /// - `datatype` matches the actual element layout of both buffers.
    /// - `root` is in the range `0..world_size` and is the same value
    ///   on every rank.
    /// - Every rank invokes `broadcast_raw` with the same `count`,
    ///   `datatype`, and `root` (NCCL collective contract).
    /// - The current CUDA device matches the communicator's device.
    /// - The buffers remain alive and unmodified until the NCCL
    ///   stream is synchronised.
    pub unsafe fn broadcast_raw(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        count: usize,
        datatype: NcclDataType,
        root: i32,
    ) -> FerrotorchResult<()> {
        let comm = *self.lock_comm()?;
        // SAFETY: this `pub unsafe fn`'s `# Safety` rustdoc enumerates the
        // caller's obligations; each maps to an `nccl_sys::broadcast`
        // precondition as follows:
        //   - "sendbuf valid on root for read of `count * size_of(datatype)`",
        //     "recvbuf valid on non-root for write of same",
        //     "buffers alive until stream synchronisation"
        //                       → ncclBroadcast buffer-validity contract;
        //   - "datatype matches both buffers' layout"
        //                       → ncclBroadcast datatype/buffer pairing;
        //   - "root in 0..world_size, same on every rank",
        //     "every rank invokes broadcast_raw with same count, datatype, root"
        //                       → NCCL collective + root-consistency contract;
        //   - "current CUDA device matches the comm's device"
        //                       → ncclBroadcast device-binding precondition.
        //
        // `comm` was obtained from `*self.lock_comm()?` immediately above,
        // copying the live `*mut c_void` out of the held `MutexGuard`;
        // since the lock is held to the end of this function no concurrent
        // destroy can race. `self.stream` is null (default stream) or a
        // dedicated stream owned by `self`, alive for `&self`'s borrow.
        unsafe { nccl_sys::broadcast(sendbuf, recvbuf, count, datatype, root, comm, self.stream) }
            .map_err(|e| {
                DistributedError::Io {
                    message: format!("NCCL broadcast failed: {e}"),
                }
                .into()
            })
    }

    /// GPU all-gather: each rank sends `sendcount` elements, receives
    /// `sendcount * world_size` elements.
    ///
    /// # Safety
    ///
    /// The caller MUST uphold all of the following:
    ///
    /// - `sendbuf` is a valid CUDA device pointer with at least
    ///   `sendcount * size_of(datatype)` bytes addressable for read.
    /// - `recvbuf` is a valid CUDA device pointer with at least
    ///   `sendcount * world_size * size_of(datatype)` bytes
    ///   addressable for write. Undersized `recvbuf` causes NCCL to
    ///   write past the buffer and is undefined behaviour.
    /// - `sendbuf` and `recvbuf` do not alias.
    /// - `datatype` matches the element layout of both buffers.
    /// - Every rank invokes `all_gather_raw` with the same `sendcount`
    ///   and `datatype` (NCCL collective contract).
    /// - The current CUDA device matches the communicator's device.
    /// - The buffers remain alive and unmodified until the NCCL
    ///   stream is synchronised.
    pub unsafe fn all_gather_raw(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        sendcount: usize,
        datatype: NcclDataType,
    ) -> FerrotorchResult<()> {
        let comm = *self.lock_comm()?;
        // SAFETY: this `pub unsafe fn`'s `# Safety` rustdoc enumerates the
        // caller's obligations; each maps to an `nccl_sys::all_gather`
        // precondition as follows:
        //   - "sendbuf valid for read of `sendcount * size_of(datatype)`",
        //     "recvbuf valid for write of `sendcount * world_size *
        //      size_of(datatype)`", "sendbuf and recvbuf do not alias",
        //     "buffers alive until stream synchronisation"
        //                       → ncclAllGather buffer-validity / no-alias
        //                          contract (NCCL UB on undersized recv);
        //   - "datatype matches both buffers' layout"
        //                       → ncclAllGather datatype/buffer pairing;
        //   - "every rank invokes all_gather_raw with same sendcount and
        //      datatype"        → NCCL collective contract;
        //   - "current CUDA device matches the comm's device"
        //                       → ncclAllGather device-binding precondition.
        //
        // `comm` came from `*self.lock_comm()?` above; the held mutex
        // guards against concurrent destroy. `self.stream` is null
        // (default stream) or a dedicated stream owned by `self`.
        unsafe { nccl_sys::all_gather(sendbuf, recvbuf, sendcount, datatype, comm, self.stream) }
            .map_err(|e| {
                DistributedError::Io {
                    message: format!("NCCL all_gather failed: {e}"),
                }
                .into()
            })
    }

    /// GPU reduce-scatter: reduces then distributes `recvcount` elements
    /// to each rank.
    ///
    /// # Safety
    ///
    /// The caller MUST uphold all of the following:
    ///
    /// - `sendbuf` is a valid CUDA device pointer with at least
    ///   `recvcount * world_size * size_of(datatype)` bytes addressable
    ///   for read.
    /// - `recvbuf` is a valid CUDA device pointer with at least
    ///   `recvcount * size_of(datatype)` bytes addressable for write.
    ///   Undersized `recvbuf` causes NCCL to write past the buffer and
    ///   is undefined behaviour.
    /// - `sendbuf` and `recvbuf` do not alias.
    /// - `datatype` matches the element layout of both buffers.
    /// - Every rank invokes `reduce_scatter_raw` with the same
    ///   `recvcount`, `datatype`, and `op` (NCCL collective contract).
    /// - The current CUDA device matches the communicator's device.
    /// - The buffers remain alive and unmodified until the NCCL
    ///   stream is synchronised.
    pub unsafe fn reduce_scatter_raw(
        &self,
        sendbuf: *const c_void,
        recvbuf: *mut c_void,
        recvcount: usize,
        datatype: NcclDataType,
        op: NcclRedOp,
    ) -> FerrotorchResult<()> {
        let comm = *self.lock_comm()?;
        // SAFETY: this `pub unsafe fn`'s `# Safety` rustdoc enumerates the
        // caller's obligations; each maps to an `nccl_sys::reduce_scatter`
        // precondition as follows:
        //   - "sendbuf valid for read of `recvcount * world_size *
        //      size_of(datatype)`", "recvbuf valid for write of
        //      `recvcount * size_of(datatype)`", "sendbuf and recvbuf
        //      do not alias", "buffers alive until stream sync"
        //                       → ncclReduceScatter buffer-validity / no-alias
        //                          contract (NCCL UB on undersized recv);
        //   - "datatype matches both buffers' layout"
        //                       → ncclReduceScatter datatype/buffer pairing;
        //   - "every rank invokes reduce_scatter_raw with same recvcount,
        //      datatype, op"    → NCCL collective contract;
        //   - "current CUDA device matches the comm's device"
        //                       → ncclReduceScatter device-binding precond.
        //
        // `comm` came from `*self.lock_comm()?` above; the held mutex
        // guards against concurrent destroy. `self.stream` is null
        // (default stream) or a dedicated stream owned by `self`.
        unsafe {
            nccl_sys::reduce_scatter(sendbuf, recvbuf, recvcount, datatype, op, comm, self.stream)
        }
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
        // SAFETY: calling NCCL `ncclAllReduce` for a count=0 barrier.
        //
        // - COMM VALIDITY: `comm` was just dereferenced from
        //   `self.lock_comm()?`, which holds the `Mutex<NcclComm>` lock for
        //   the duration of this block. The communicator was initialised
        //   in `Self::new` / `Self::with_stream` via `comm_init_rank` and
        //   has not yet been destroyed (`Drop::drop` runs only when the
        //   last reference goes away).
        // - POINTER VALIDITY: with `count = 0`, NCCL does not dereference
        //   `sendbuf`/`recvbuf` per the NCCL API contract — null pointers
        //   are explicitly accepted for zero-count operations (this is
        //   documented as "valid synchronization point" in the NCCL
        //   programming guide and used identically by PyTorch's
        //   `ProcessGroupNCCL::barrier`).
        // - DATATYPE/OP MATCH: `NcclDataType::Float32` and
        //   `NcclRedOp::Sum` are valid `repr(C)` enum values (see
        //   `nccl_sys` definitions, derived from nccl.h). With `count=0`
        //   no reduction work occurs, so the choice of dtype/op is
        //   inert.
        // - STREAM VALIDITY: `self.stream` is either null (default
        //   stream, always valid per CUDA semantics) or a stream
        //   created by `create_nccl_stream` and owned by `self`; either
        //   way it is alive for the lifetime of `self`, which encloses
        //   this call.
        // - THREAD SAFETY: the comm-lock guard is held for the entire
        //   call, serialising NCCL calls on this communicator with any
        //   other thread on this rank.
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
                // SAFETY: `nccl_sys::comm_destroy` is `unsafe` because it
                // requires a once-only call on a previously-initialised
                // `NcclComm`.
                //
                // - COMM VALIDITY: `*comm` was produced by
                //   `nccl_sys::comm_init_rank` in `Self::new` /
                //   `Self::with_stream`. Both constructors store the result
                //   in `Mutex<NcclComm>`. Because we just observed
                //   `!(*comm).is_null()`, the communicator is the
                //   originally-initialised handle (NCCL never resets to
                //   null after init except via destroy).
                // - SINGLE CALL: `Drop::drop` runs at most once per
                //   `NcclBackend` (Rust language guarantee). No other code
                //   path calls `comm_destroy` — `nccl_sys::comm_destroy`
                //   is `pub unsafe fn` and is not invoked anywhere else
                //   in the workspace. Therefore the once-only contract
                //   for this `comm` is upheld.
                // - LOCK ORDERING / EXCLUSIVITY: we hold the `Mutex` lock
                //   for the duration of the destroy call, so no other
                //   thread can issue a concurrent NCCL operation on this
                //   communicator. Destruction during in-flight operations
                //   would be a logic bug in the caller (they would have
                //   to be holding a borrow that crosses `Drop`, which is
                //   prevented by Rust's lifetime rules).
                // - ERROR HANDLING: `comm_destroy` returns `Result<(),
                //   NcclError>`. `Drop::drop` cannot return errors, so we
                //   discard via `let _ = ...`. NCCL destroy failures are
                //   non-recoverable here (the process is shutting down or
                //   the backend is being torn down) and would only
                //   surface as a leaked NCCL communicator on the
                //   GPU-side state machine — preferable to a panic in
                //   `Drop`.
                // - STREAM ORDERING: NCCL requires that all in-flight
                //   collective operations on `comm` complete before
                //   destruction. The `Backend` API does not retain any
                //   `&mut self` borrows across drop (Rust prevents this
                //   structurally), and `synchronize()` is exposed on the
                //   public API for users who want explicit barriers
                //   prior to drop.
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
//
// These helpers load CUDA runtime symbols (`cudaStreamCreateWithFlags`,
// `cudaStreamSynchronize`, `cudaStreamDestroy`) lazily via `dlopen` +
// `dlsym` so the crate compiles on machines without `libcudart.so` present
// at build time. Each `unsafe { mem::transmute(sym) }` cast turns a raw
// symbol pointer into a typed function-pointer; the SAFETY comments at
// each site spell out which CUDA Runtime API signature is being matched.
//
// References (CUDA Runtime API, all stable since CUDA 5.0):
//   - cudaStreamCreateWithFlags:
//     https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__STREAM.html
//     `__host__ cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream, unsigned int flags)`
//   - cudaStreamSynchronize: same group,
//     `__host__ cudaError_t cudaStreamSynchronize(cudaStream_t stream)`
//   - cudaStreamDestroy: same group,
//     `__host__ cudaError_t cudaStreamDestroy(cudaStream_t stream)`
//
// Mappings:
//   - `cudaStream_t` is `void *` (an opaque handle), matched here as
//     `*mut c_void` / `*mut *mut c_void` for the out-pointer.
//   - `cudaError_t` is `int` (a signed 32-bit enum), matched here as
//     `i32`. The success value is `cudaSuccess == 0`.
//   - `unsigned int` (flags) is matched as `u32`.

/// Create a new CUDA stream for NCCL operations.
/// Returns `None` on failure (caller falls back to default stream).
fn create_nccl_stream() -> Option<*mut c_void> {
    // SAFETY: `libc::dlopen` is `unsafe` because it loads arbitrary code.
    //
    // - LIBRARY NAME: the C-string literal `c"libcudart.so.12"` is
    //   NUL-terminated by the compiler and has `'static` lifetime; its
    //   `.as_ptr()` is a valid `*const c_char` for the duration of the
    //   call (and beyond).
    // - FLAGS: `RTLD_LAZY` is the documented flag for resolving
    //   symbols on demand; it does not change return-type semantics.
    // - RETURN VALUE: dlopen returns either a non-null opaque handle or
    //   null on failure; we check the result before any dlsym call.
    // - LIFETIME: the returned handle stays valid until `dlclose` —
    //   which we never call (matching CUDA driver convention: the
    //   process holds libcudart for its lifetime). This means the
    //   resolved function pointers below remain valid for the entire
    //   program duration.
    let lib = unsafe { libc::dlopen(c"libcudart.so.12".as_ptr(), libc::RTLD_LAZY) };
    if lib.is_null() {
        // SAFETY: same invariants as the previous dlopen — fallback to
        // the unversioned soname when the explicit ABI version is not
        // installed (e.g., older toolkits or Ubuntu's default symlink).
        let lib = unsafe { libc::dlopen(c"libcudart.so".as_ptr(), libc::RTLD_LAZY) };
        if lib.is_null() {
            return None;
        }
        return create_stream_from_lib(lib);
    }
    create_stream_from_lib(lib)
}

fn create_stream_from_lib(lib: *mut c_void) -> Option<*mut c_void> {
    // SAFETY: `dlsym` resolves a symbol against an opened library handle.
    //
    // - HANDLE VALIDITY: `lib` was just verified non-null by the caller
    //   (`create_nccl_stream` checks the dlopen result before calling
    //   here). The handle is owned by the process for its lifetime.
    // - SYMBOL NAME: `c"cudaStreamCreateWithFlags"` is a `'static`
    //   NUL-terminated C-string literal matching the CUDA Runtime API
    //   symbol name verbatim.
    // - RETURN VALUE: dlsym returns either a function-pointer-shaped
    //   non-null pointer or null; we check the result below before any
    //   transmute or call.
    let sym = unsafe { libc::dlsym(lib, c"cudaStreamCreateWithFlags".as_ptr()) };
    if sym.is_null() {
        return None;
    }
    // ABI: `cudaError_t cudaStreamCreateWithFlags(cudaStream_t *pStream,
    //                                              unsigned int flags)`.
    // Mapping: `cudaStream_t` -> `*mut c_void`, `cudaStream_t *` -> `*mut
    // *mut c_void`, `unsigned int` -> `u32`, `cudaError_t` -> `i32`.
    type CudaStreamCreateFn = unsafe extern "C" fn(*mut *mut c_void, u32) -> i32;
    // SAFETY: transmute of a raw symbol pointer to a typed
    // function-pointer with `extern "C"` ABI.
    //
    // - SOURCE TYPE: `*mut c_void` returned by dlsym, verified non-null
    //   above. Per POSIX, dlsym returns a pointer that may legitimately
    //   be cast to a function-pointer (the object/function ambiguity is
    //   acknowledged by POSIX 2017 Issue 7 and accepted as
    //   implementation-defined; on every platform ferrotorch supports
    //   — Linux glibc/musl, macOS — this cast is well-defined).
    // - TARGET TYPE: `unsafe extern "C" fn(*mut *mut c_void, u32) -> i32`.
    //   This signature must match the CUDA Runtime ABI for
    //   `cudaStreamCreateWithFlags` exactly:
    //     * Calling convention: C (CUDA Runtime is exported as
    //       `__host__` C functions; on x86_64-linux-gnu the host-side C
    //       ABI is the System V AMD64 ABI; on aarch64-linux-gnu it is
    //       the AAPCS64. `extern "C"` selects whichever the target
    //       platform mandates.).
    //     * Argument count: 2.
    //     * Argument 1: `cudaStream_t *` = pointer to opaque handle =
    //       `*mut *mut c_void` (8 bytes on 64-bit, matching CUDA's host
    //       ABI; no other targets are supported by NCCL).
    //     * Argument 2: `unsigned int flags` = 4-byte unsigned = `u32`.
    //     * Return: `cudaError_t` (a C enum sized as `int`) = `i32`.
    // - SIZE/LAYOUT: `*mut c_void`, `*mut *mut c_void`, `u32`, and `i32`
    //   match their C counterparts exactly under all CUDA-supported
    //   host ABIs.
    // - LIFETIME: the function pointer remains valid as long as `lib`
    //   is loaded; we never `dlclose`, so it lives for the program
    //   duration.
    // - PROVENANCE: `sym` was returned by libc; transmuting a raw
    //   pointer to a fn-pointer of correct ABI is a documented use of
    //   `mem::transmute` (Rust reference, `transmute` semantics for
    //   pointer-to-function casts).
    let create_fn: CudaStreamCreateFn = unsafe { std::mem::transmute(sym) };
    let mut stream: *mut c_void = std::ptr::null_mut();
    // SAFETY: calling the resolved CUDA Runtime function.
    //
    // - FN-PTR VALIDITY: `create_fn` is the transmuted symbol just
    //   resolved; `lib` is still loaded (no intervening `dlclose`).
    // - ARGUMENT 1: `&mut stream` is a unique mutable borrow to a
    //   stack-local `*mut c_void`, valid for the duration of the
    //   call. CUDA writes the new stream handle into `*pStream` on
    //   success; the slot is `null_mut()` on entry.
    // - ARGUMENT 2: `1` selects `cudaStreamNonBlocking`, the documented
    //   flag value (cf. cudaStreamFlags enum in CUDA Runtime headers,
    //   stable since CUDA 5.0).
    // - PRECONDITIONS: CUDA does NOT require a current device to be set
    //   for `cudaStreamCreateWithFlags` — it allocates against the
    //   current device, defaulting to device 0 if none has been set
    //   explicitly. NCCL itself requires a `cudaSetDevice` before
    //   `comm_init_rank`, which is the caller's responsibility (see
    //   `Self::new` doc).
    // - RETURN: `cudaError_t` integer; `0 == cudaSuccess`. We discard
    //   on failure and return `None` so the backend falls back to the
    //   default stream.
    let result = unsafe { create_fn(&raw mut stream, 1) }; // 1 = cudaStreamNonBlocking
    if result == 0 { Some(stream) } else { None }
}

/// Synchronize a CUDA stream (blocks until all operations complete).
fn synchronize_stream(stream: *mut c_void) -> Result<(), String> {
    // SAFETY: same dlopen invariants as `create_nccl_stream`. The
    // returned handle is checked for null below before any dlsym.
    let lib = unsafe { libc::dlopen(c"libcudart.so.12".as_ptr(), libc::RTLD_LAZY) };
    let lib = if lib.is_null() {
        // SAFETY: same dlopen invariants — fallback soname.
        unsafe { libc::dlopen(c"libcudart.so".as_ptr(), libc::RTLD_LAZY) }
    } else {
        lib
    };
    if lib.is_null() {
        return Err("cudart not found".into());
    }
    // SAFETY: same dlsym invariants as `create_stream_from_lib`. `lib`
    // was just checked non-null. Symbol name is a `'static` NUL-terminated
    // C-string literal matching the CUDA Runtime symbol verbatim.
    let sym = unsafe { libc::dlsym(lib, c"cudaStreamSynchronize".as_ptr()) };
    if sym.is_null() {
        return Err("cudaStreamSynchronize not found".into());
    }
    // ABI: `cudaError_t cudaStreamSynchronize(cudaStream_t stream)`.
    // Mapping: `cudaStream_t` -> `*mut c_void`, `cudaError_t` -> `i32`.
    type SyncFn = unsafe extern "C" fn(*mut c_void) -> i32;
    // SAFETY: transmute of dlsym result to typed C fn-pointer.
    //
    // - SOURCE: non-null `*mut c_void` from dlsym (checked above).
    // - TARGET: `unsafe extern "C" fn(*mut c_void) -> i32` — matches the
    //   CUDA Runtime ABI for `cudaStreamSynchronize` (1 pointer arg,
    //   `cudaError_t` (=`int`=`i32`) return).
    // - SIZE/LAYOUT: `*mut c_void` and `i32` match their C
    //   counterparts on every supported host platform.
    // - LIFETIME: function pointer valid for program lifetime (no
    //   `dlclose`).
    let sync_fn: SyncFn = unsafe { std::mem::transmute(sym) };
    // SAFETY: calling the resolved CUDA Runtime function.
    //
    // - FN-PTR VALIDITY: just resolved; library still loaded.
    // - ARGUMENT: `stream` is the caller-provided `*mut c_void`. Per the
    //   only callsite (`NcclBackend::synchronize`), this is either
    //   - a stream produced by `create_nccl_stream` and stored in
    //     `self.stream` — alive as long as `self` lives — OR
    //   - null if no dedicated stream was created. The caller's
    //     `synchronize` early-returns on null before reaching here, so
    //     null is never passed in practice.
    // - PRECONDITIONS: CUDA `cudaStreamSynchronize` accepts any valid
    //   stream handle, including the default stream. It blocks the
    //   calling thread.
    // - RETURN: `cudaError_t`; non-zero is propagated as `Err(String)`.
    let result = unsafe { sync_fn(stream) };
    if result == 0 {
        Ok(())
    } else {
        Err(format!("cudaStreamSynchronize failed: error {result}"))
    }
}

/// Destroy a CUDA stream.
fn destroy_stream(stream: *mut c_void) {
    // SAFETY: same dlopen invariants as `create_nccl_stream`.
    let lib = unsafe { libc::dlopen(c"libcudart.so.12".as_ptr(), libc::RTLD_LAZY) };
    let lib = if lib.is_null() {
        // SAFETY: same dlopen invariants — fallback soname.
        unsafe { libc::dlopen(c"libcudart.so".as_ptr(), libc::RTLD_LAZY) }
    } else {
        lib
    };
    if lib.is_null() {
        return;
    }
    // SAFETY: same dlsym invariants as the other helpers. `lib`
    // verified non-null above; symbol name matches the CUDA Runtime
    // export verbatim.
    let sym = unsafe { libc::dlsym(lib, c"cudaStreamDestroy".as_ptr()) };
    if sym.is_null() {
        return;
    }
    // ABI: `cudaError_t cudaStreamDestroy(cudaStream_t stream)`.
    // Mapping: `cudaStream_t` -> `*mut c_void`, `cudaError_t` -> `i32`.
    type DestroyFn = unsafe extern "C" fn(*mut c_void) -> i32;
    // SAFETY: transmute of dlsym result to typed C fn-pointer.
    //
    // - SOURCE: non-null `*mut c_void` from dlsym (checked above).
    // - TARGET: `unsafe extern "C" fn(*mut c_void) -> i32` — matches
    //   `cudaStreamDestroy`'s ABI (1 pointer arg, `cudaError_t`
    //   return).
    // - SIZE/LAYOUT: identical to `synchronize_stream` above; same
    //   guarantees apply.
    // - LIFETIME: program-lifetime fn-pointer (no `dlclose`).
    let destroy_fn: DestroyFn = unsafe { std::mem::transmute(sym) };
    // SAFETY: calling `cudaStreamDestroy` on a stream we own.
    //
    // - FN-PTR VALIDITY: just resolved; library still loaded.
    // - STREAM VALIDITY: `destroy_stream` is only called from
    //   `NcclBackend::drop` under the guard `self.owns_stream &&
    //   !self.stream.is_null()`. `self.owns_stream` is set to `true`
    //   IFF `create_nccl_stream` succeeded (returned a non-null handle)
    //   — see `Self::new`. Therefore the stream passed here was
    //   produced by `cudaStreamCreateWithFlags` on this process and
    //   has not yet been destroyed (Drop runs at most once).
    // - SINGLE CALL: `Drop::drop` is called at most once per
    //   `NcclBackend` (Rust language guarantee), so each stream is
    //   destroyed exactly once.
    // - ORDERING: the CUDA Runtime API does not require any operation
    //   on the stream to have completed before destroy; pending work
    //   is allowed to finish asynchronously. NCCL operations enqueued
    //   on this stream complete on the device-side queue independently
    //   of the host destroy call.
    // - RETURN VALUE DISCARDED: this is `Drop`-context where we cannot
    //   propagate errors. A failed destroy would only leak the stream,
    //   which is recovered when the process exits.
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
