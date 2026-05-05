//! Raw FFI bindings to the NCCL (NVIDIA Collective Communication Library).
//!
//! NCCL is loaded at runtime via `dlopen` so that the crate compiles and
//! works on systems without NCCL installed (it just returns an error when
//! you try to create an `NcclComm`).
//!
//! # Safety
//!
//! All functions in this module are inherently unsafe (raw C FFI). The safe
//! wrappers live in [`crate::nccl_backend`].

use std::ffi::c_void;
use std::sync::OnceLock;

// ---------------------------------------------------------------------------
// NCCL types (matching nccl.h)
// ---------------------------------------------------------------------------

/// Opaque NCCL communicator handle.
pub type NcclComm = *mut c_void;

/// NCCL unique ID for communicator bootstrap (128 bytes).
///
/// The `internal` byte buffer is written by NCCL via
/// [`get_unique_id`] and must be transmitted to all ranks (commonly via
/// TCP, env var, or shared filesystem) before [`comm_init_rank`].
///
/// `#[repr(C)]` is load-bearing: this struct is passed by value across
/// the C FFI boundary to NCCL and its layout must match `ncclUniqueId`
/// from `nccl.h` exactly. `#[non_exhaustive]` is a Rust surface-API
/// annotation only — it does not affect memory layout — and is added
/// to allow forward-compatible additions (e.g., a version field) without
/// breaking external struct-literal construction. External callers
/// obtain instances via [`get_unique_id`] rather than constructing
/// directly.
#[repr(C)]
#[derive(Clone, Copy)]
#[non_exhaustive]
pub struct NcclUniqueId {
    pub internal: [u8; 128],
}

/// NCCL data types.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclDataType {
    Int8 = 0,
    Uint8 = 1,
    Int32 = 2,
    Uint32 = 3,
    Int64 = 4,
    Uint64 = 5,
    Float16 = 6,
    Float32 = 7,
    Float64 = 8,
    Bfloat16 = 9,
}

/// NCCL reduction operations.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclRedOp {
    Sum = 0,
    Prod = 1,
    Max = 2,
    Min = 3,
    Avg = 4,
}

/// NCCL result codes.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NcclResult {
    Success = 0,
    UnhandledCudaError = 1,
    SystemError = 2,
    InternalError = 3,
    InvalidArgument = 4,
    InvalidUsage = 5,
    RemoteError = 6,
    InProgress = 7,
    NumResults = 8,
}

impl NcclResult {
    pub fn ok(self) -> Result<(), NcclError> {
        if self == NcclResult::Success {
            Ok(())
        } else {
            Err(NcclError::NcclStatus(self))
        }
    }
}

/// Errors from NCCL operations.
#[derive(Debug, thiserror::Error)]
pub enum NcclError {
    #[error("NCCL library not found — install libnccl2 or set LD_LIBRARY_PATH")]
    LibraryNotFound,

    #[error("NCCL symbol not found: {0}")]
    SymbolNotFound(String),

    #[error("NCCL error: {0:?}")]
    NcclStatus(NcclResult),
}

// ---------------------------------------------------------------------------
// Function pointer table (loaded via dlopen)
// ---------------------------------------------------------------------------

/// All NCCL function pointers we need, loaded at runtime.
#[allow(non_snake_case)]
struct NcclFunctions {
    ncclGetUniqueId: unsafe extern "C" fn(*mut NcclUniqueId) -> NcclResult,
    ncclCommInitRank: unsafe extern "C" fn(*mut NcclComm, i32, NcclUniqueId, i32) -> NcclResult,
    ncclCommDestroy: unsafe extern "C" fn(NcclComm) -> NcclResult,
    ncclAllReduce: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        NcclDataType,
        NcclRedOp,
        NcclComm,
        *mut c_void, // cudaStream_t
    ) -> NcclResult,
    ncclBroadcast: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize,
        NcclDataType,
        i32, // root
        NcclComm,
        *mut c_void,
    ) -> NcclResult,
    ncclAllGather: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize, // sendcount
        NcclDataType,
        NcclComm,
        *mut c_void,
    ) -> NcclResult,
    ncclReduceScatter: unsafe extern "C" fn(
        *const c_void,
        *mut c_void,
        usize, // recvcount
        NcclDataType,
        NcclRedOp,
        NcclComm,
        *mut c_void,
    ) -> NcclResult,
    ncclSend: unsafe extern "C" fn(
        *const c_void,
        usize,
        NcclDataType,
        i32, // peer
        NcclComm,
        *mut c_void,
    ) -> NcclResult,
    ncclRecv: unsafe extern "C" fn(
        *mut c_void,
        usize,
        NcclDataType,
        i32, // peer
        NcclComm,
        *mut c_void,
    ) -> NcclResult,
    ncclGroupStart: unsafe extern "C" fn() -> NcclResult,
    ncclGroupEnd: unsafe extern "C" fn() -> NcclResult,
}

// SAFETY: The function pointers are loaded once and never mutated.
unsafe impl Send for NcclFunctions {}
unsafe impl Sync for NcclFunctions {}

/// Global singleton for the loaded NCCL library.
static NCCL_LIB: OnceLock<Result<NcclFunctions, NcclError>> = OnceLock::new();

/// Attempt to load NCCL at runtime via dlopen.
fn load_nccl() -> Result<NcclFunctions, NcclError> {
    // Try common library names.
    let lib_names = [
        "libnccl.so.2",
        "libnccl.so",
        "/usr/lib/x86_64-linux-gnu/libnccl.so.2",
        "/usr/local/cuda/lib64/libnccl.so.2",
    ];

    let mut lib_handle: *mut c_void = std::ptr::null_mut();
    for name in &lib_names {
        let c_name = std::ffi::CString::new(*name).unwrap();
        // SAFETY: dlopen with RTLD_LAZY is safe — it just opens a shared library.
        lib_handle = unsafe { libc::dlopen(c_name.as_ptr(), libc::RTLD_LAZY) };
        if !lib_handle.is_null() {
            break;
        }
    }

    if lib_handle.is_null() {
        return Err(NcclError::LibraryNotFound);
    }

    // Helper macro to load a symbol.
    //
    // The `transmute(ptr)` here is parametrically inferred from each call
    // site's surrounding context (i.e. the corresponding `NcclFunctions`
    // field's `unsafe extern "C" fn(...) -> NcclResult` type). Adding the
    // explicit annotation clippy suggests would require either threading
    // the fn-pointer type as a second macro parameter — duplicating the
    // 11 signatures already declared in `NcclFunctions` — or a parallel
    // turbofish per call site mirroring the same types. The struct
    // definition above is the single source of truth; the transmute is
    // type-checked by Rust through the field-init coercion. The narrow
    // `#[allow]` is placed on the transmute statement inside the macro
    // body so it propagates to every expansion site.
    macro_rules! load_sym {
        ($name:ident) => {{
            let c_name = std::ffi::CString::new(stringify!($name)).unwrap();
            let ptr = unsafe { libc::dlsym(lib_handle, c_name.as_ptr()) };
            if ptr.is_null() {
                return Err(NcclError::SymbolNotFound(stringify!($name).into()));
            }
            #[allow(clippy::missing_transmute_annotations)]
            let f = unsafe { std::mem::transmute(ptr) };
            f
        }};
    }

    Ok(NcclFunctions {
        ncclGetUniqueId: load_sym!(ncclGetUniqueId),
        ncclCommInitRank: load_sym!(ncclCommInitRank),
        ncclCommDestroy: load_sym!(ncclCommDestroy),
        ncclAllReduce: load_sym!(ncclAllReduce),
        ncclBroadcast: load_sym!(ncclBroadcast),
        ncclAllGather: load_sym!(ncclAllGather),
        ncclReduceScatter: load_sym!(ncclReduceScatter),
        ncclSend: load_sym!(ncclSend),
        ncclRecv: load_sym!(ncclRecv),
        ncclGroupStart: load_sym!(ncclGroupStart),
        ncclGroupEnd: load_sym!(ncclGroupEnd),
    })
}

/// Get the loaded NCCL function table, loading on first call.
fn nccl() -> Result<&'static NcclFunctions, NcclError> {
    NCCL_LIB
        .get_or_init(load_nccl)
        .as_ref()
        .map_err(|e| match e {
            NcclError::LibraryNotFound => NcclError::LibraryNotFound,
            NcclError::SymbolNotFound(s) => NcclError::SymbolNotFound(s.clone()),
            NcclError::NcclStatus(s) => NcclError::NcclStatus(*s),
        })
}

// ---------------------------------------------------------------------------
// Safe wrappers
// ---------------------------------------------------------------------------

/// Generate a unique ID for NCCL communicator initialization.
///
/// Must be called by rank 0 and then broadcast to all other ranks
/// (e.g. via TCP or shared filesystem).
pub fn get_unique_id() -> Result<NcclUniqueId, NcclError> {
    let lib = nccl()?;
    let mut id = NcclUniqueId {
        internal: [0u8; 128],
    };
    // SAFETY: id is a valid NcclUniqueId struct on the stack. The
    // `&raw mut id` raw-pointer borrow (Rust 2024 idiom) hands a typed
    // pointer to ncclGetUniqueId without first creating a `&mut id`
    // reference; the FFI fn writes the 128-byte buffer and the result is
    // owned by `id` for the rest of this function. Fn-ptr resolved by
    // `load_nccl`, valid for program lifetime.
    unsafe { (lib.ncclGetUniqueId)(&raw mut id) }.ok()?;
    Ok(id)
}

/// Initialize an NCCL communicator for this rank.
///
/// `world_size` is the total number of ranks. `rank` is this process's
/// rank (0-based). `unique_id` must be the same value on all ranks.
///
/// # Safety
///
/// CUDA must be initialized and the correct device must be set before
/// calling this function (`cudaSetDevice`).
pub fn comm_init_rank(
    world_size: i32,
    rank: i32,
    unique_id: NcclUniqueId,
) -> Result<NcclComm, NcclError> {
    let lib = nccl()?;
    let mut comm: NcclComm = std::ptr::null_mut();
    // SAFETY: unique_id is valid, world_size and rank are within bounds.
    // `&raw mut comm` (Rust 2024 idiom) yields a `*mut NcclComm` without
    // creating an intermediate reference; ncclCommInitRank writes the new
    // communicator handle into `comm` on success. Fn-ptr resolved by
    // `load_nccl`, valid for program lifetime.
    unsafe { (lib.ncclCommInitRank)(&raw mut comm, world_size, unique_id, rank) }.ok()?;
    Ok(comm)
}

/// Destroy an NCCL communicator.
///
/// # Safety
///
/// `comm` must be a valid communicator that hasn't been destroyed.
pub unsafe fn comm_destroy(comm: NcclComm) -> Result<(), NcclError> {
    let lib = nccl()?;
    // SAFETY: invoking the `unsafe extern "C"` ncclCommDestroy fn-pointer.
    // The caller's `# Safety` contract on `comm_destroy` ("`comm` is a
    // valid communicator that hasn't been destroyed") discharges the
    // ncclCommDestroy precondition that the handle was produced by a
    // matching ncclCommInitRank and has not previously been destroyed.
    // `lib.ncclCommDestroy` was resolved once via dlsym in `load_nccl`
    // and remains valid for program lifetime (libnccl is never
    // dlclosed).
    unsafe { (lib.ncclCommDestroy)(comm) }.ok()
}

/// In-place all-reduce on device memory.
///
/// # Safety
///
/// `sendbuf` and `recvbuf` must be valid device pointers with at least
/// `count` elements of the specified `datatype`. `stream` must be a valid
/// CUDA stream (or null for default stream).
pub unsafe fn all_reduce(
    sendbuf: *const c_void,
    recvbuf: *mut c_void,
    count: usize,
    datatype: NcclDataType,
    op: NcclRedOp,
    comm: NcclComm,
    stream: *mut c_void,
) -> Result<(), NcclError> {
    let lib = nccl()?;
    // SAFETY: invoking the `unsafe extern "C"` ncclAllReduce fn-pointer.
    // The caller's `# Safety` contract on `all_reduce` ("`sendbuf` and
    // `recvbuf` are valid device pointers with at least `count` elements
    // of `datatype`; `stream` is a valid CUDA stream or null") discharges
    // ncclAllReduce's preconditions: buffer validity for `count *
    // size_of(datatype)` bytes on the device side, datatype matches the
    // buffers' element layout, comm is the live communicator handle the
    // caller is responsible for, stream is a valid CUDA stream (or null
    // for the default stream). `lib.ncclAllReduce` was resolved by
    // `load_nccl` and is valid for program lifetime.
    unsafe { (lib.ncclAllReduce)(sendbuf, recvbuf, count, datatype, op, comm, stream) }.ok()
}

/// Broadcast from root to all ranks on device memory.
///
/// # Safety
///
/// Same requirements as [`all_reduce`].
pub unsafe fn broadcast(
    sendbuf: *const c_void,
    recvbuf: *mut c_void,
    count: usize,
    datatype: NcclDataType,
    root: i32,
    comm: NcclComm,
    stream: *mut c_void,
) -> Result<(), NcclError> {
    let lib = nccl()?;
    // SAFETY: invoking the `unsafe extern "C"` ncclBroadcast fn-pointer.
    // The caller's `# Safety` contract on `broadcast` (same requirements
    // as `all_reduce`) discharges ncclBroadcast's preconditions: buffer
    // validity, datatype/buffer match, valid `root` rank in
    // `0..world_size`, live comm, valid CUDA stream (or null).
    // `lib.ncclBroadcast` was resolved by `load_nccl` and is valid for
    // program lifetime.
    unsafe { (lib.ncclBroadcast)(sendbuf, recvbuf, count, datatype, root, comm, stream) }.ok()
}

/// All-gather: each rank sends `sendcount` elements, receives
/// `sendcount * world_size` elements.
///
/// # Safety
///
/// Same requirements as [`all_reduce`].
pub unsafe fn all_gather(
    sendbuf: *const c_void,
    recvbuf: *mut c_void,
    sendcount: usize,
    datatype: NcclDataType,
    comm: NcclComm,
    stream: *mut c_void,
) -> Result<(), NcclError> {
    let lib = nccl()?;
    // SAFETY: invoking the `unsafe extern "C"` ncclAllGather fn-pointer.
    // The caller's `# Safety` contract on `all_gather` (same requirements
    // as `all_reduce`) discharges ncclAllGather's preconditions: sendbuf
    // valid for `sendcount` elements, recvbuf valid for `sendcount *
    // world_size` elements, datatype/buffer match, live comm, valid CUDA
    // stream (or null). `lib.ncclAllGather` was resolved by `load_nccl`
    // and is valid for program lifetime.
    unsafe { (lib.ncclAllGather)(sendbuf, recvbuf, sendcount, datatype, comm, stream) }.ok()
}

/// Reduce-scatter: reduces then distributes `recvcount` elements to each rank.
///
/// # Safety
///
/// Same requirements as [`all_reduce`].
pub unsafe fn reduce_scatter(
    sendbuf: *const c_void,
    recvbuf: *mut c_void,
    recvcount: usize,
    datatype: NcclDataType,
    op: NcclRedOp,
    comm: NcclComm,
    stream: *mut c_void,
) -> Result<(), NcclError> {
    let lib = nccl()?;
    // SAFETY: invoking the `unsafe extern "C"` ncclReduceScatter fn-pointer.
    // The caller's `# Safety` contract on `reduce_scatter` (same as
    // `all_reduce`) discharges ncclReduceScatter's preconditions: sendbuf
    // valid for `recvcount * world_size` elements, recvbuf valid for
    // `recvcount` elements, datatype/buffer match, live comm, valid CUDA
    // stream (or null). `lib.ncclReduceScatter` was resolved by
    // `load_nccl` and is valid for program lifetime.
    unsafe { (lib.ncclReduceScatter)(sendbuf, recvbuf, recvcount, datatype, op, comm, stream) }.ok()
}

/// Point-to-point send (NCCL 2.7+).
///
/// Must be paired with a matching `recv` on the peer rank, and both
/// must be within a `group_start` / `group_end` bracket.
///
/// # Safety
///
/// `sendbuf` must be a valid device pointer.
pub unsafe fn send(
    sendbuf: *const c_void,
    count: usize,
    datatype: NcclDataType,
    peer: i32,
    comm: NcclComm,
    stream: *mut c_void,
) -> Result<(), NcclError> {
    let lib = nccl()?;
    // SAFETY: invoking the `unsafe extern "C"` ncclSend fn-pointer. The
    // caller's `# Safety` contract on `send` ("`sendbuf` is a valid device
    // pointer") plus the documented pairing requirement (matching `recv`
    // on the peer rank, both bracketed by `group_start`/`group_end`)
    // discharges ncclSend's preconditions: sendbuf valid for `count`
    // elements of `datatype`, `peer` is a rank within the comm, live
    // comm, valid CUDA stream (or null). `lib.ncclSend` was resolved by
    // `load_nccl` and is valid for program lifetime.
    unsafe { (lib.ncclSend)(sendbuf, count, datatype, peer, comm, stream) }.ok()
}

/// Point-to-point receive (NCCL 2.7+).
///
/// # Safety
///
/// `recvbuf` must be a valid device pointer.
pub unsafe fn recv(
    recvbuf: *mut c_void,
    count: usize,
    datatype: NcclDataType,
    peer: i32,
    comm: NcclComm,
    stream: *mut c_void,
) -> Result<(), NcclError> {
    let lib = nccl()?;
    // SAFETY: invoking the `unsafe extern "C"` ncclRecv fn-pointer. The
    // caller's `# Safety` contract on `recv` ("`recvbuf` is a valid device
    // pointer") plus the documented pairing requirement (matching `send`
    // on the peer rank, both bracketed by `group_start`/`group_end`)
    // discharges ncclRecv's preconditions: recvbuf valid for write of
    // `count` elements of `datatype`, `peer` is a rank within the comm,
    // live comm, valid CUDA stream (or null). `lib.ncclRecv` was
    // resolved by `load_nccl` and is valid for program lifetime.
    unsafe { (lib.ncclRecv)(recvbuf, count, datatype, peer, comm, stream) }.ok()
}

/// Begin a group of NCCL operations (batches kernel launches).
pub fn group_start() -> Result<(), NcclError> {
    let lib = nccl()?;
    // SAFETY: No preconditions.
    unsafe { (lib.ncclGroupStart)() }.ok()
}

/// End a group of NCCL operations (launches all batched kernels).
pub fn group_end() -> Result<(), NcclError> {
    let lib = nccl()?;
    // SAFETY: Must follow a matching group_start.
    unsafe { (lib.ncclGroupEnd)() }.ok()
}

/// Returns `true` if NCCL is available on this system.
pub fn is_available() -> bool {
    nccl().is_ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nccl_availability_doesnt_panic() {
        // Just check it doesn't crash — NCCL may or may not be installed.
        let available = is_available();
        eprintln!("NCCL available: {available}");
    }
}
