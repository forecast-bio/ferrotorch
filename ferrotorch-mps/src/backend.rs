//! Apple Metal backend for ferrotorch-mps (#626).
//!
//! [`MtlBackend`] implements [`GpuBackend`] from `ferrotorch-core` using MSL
//! kernels compiled at runtime via the `objc2-metal` crate.
//!
//! # Platform gating
//!
//! This entire module is `#[cfg(target_os = "macos")]`. On Linux/WSL the
//! module is absent from the compilation unit, so none of the `objc2-metal`
//! bindings are referenced and the workspace build stays clean.
//!
//! # PyTorch parity (§3)
//!
//! ferrotorch is a PyTorch reimplementation. Every method that cannot execute
//! a real Metal kernel on the current platform returns
//! `Err(FerrotorchError::DeviceUnavailable)` — never a silent CPU detour.
//! On macOS, methods for the 10 implemented kernels compile and launch the
//! MSL source; the remaining ~70 GpuBackend methods return
//! `Err(FerrotorchError::InvalidArgument { message: "MSL kernel needed: ..." })`,
//! matching PyTorch's `NotImplementedError` shape for unimplemented ops.
//!
//! # Runtime kernel compilation
//!
//! `MtlBackend::new()` eagerly compiles all 10 MSL shader libraries. Compilation
//! failures return `Err` immediately — there is no lazy degrade-to-CPU path.
//! The compiled `MTLComputePipelineState` handles are cached in `MtlBackend`
//! for the lifetime of the backend.
//!
//! # Buffer representation
//!
//! GPU buffers are `Arc<MtlBuffer>` (a newtype around `Retained<MTLBuffer>`)
//! stored in `GpuBufferHandle::inner` via type-erasure. Downcast via
//! `handle.downcast_ref::<Arc<MtlBuffer>>()`.

#![cfg(target_os = "macos")]

use std::sync::Arc;

use ferrotorch_core::error::{FerrotorchError, FerrotorchResult};
use ferrotorch_core::gpu_dispatch::{GpuBackend, GpuBufferHandle, GpuRngState};
use objc2::rc::Retained;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandQueue, MTLComputeCommandEncoder,
    MTLComputePipelineState, MTLDevice, MTLFunction, MTLLibrary, MTLSize, MTLStorageMode,
};

use crate::kernels;

// ---------------------------------------------------------------------------
// MtlBuffer — owned Metal buffer handle
// ---------------------------------------------------------------------------

/// Newtype around a retained `MTLBuffer` so it can live in a `GpuBufferHandle`
/// via `Box<dyn Any + Send + Sync>`.
///
/// `MTLBuffer` is reference-counted by `objc2::rc::Retained`; wrapping in
/// `Arc` makes the `Send + Sync` bound trivially satisfied because we only
/// access buffer contents through the Metal command queue (which serialises
/// access internally).
///
/// # Safety
///
/// `objc2-metal` marks `MTLBuffer` as `!Send + !Sync` on non-macOS (the type
/// doesn't exist there). On macOS the Metal runtime guarantees thread-safe
/// reference-counting for retained objects; the `Arc` wrapper here expresses
/// that invariant at the Rust type level.
pub struct MtlBuffer {
    pub(crate) inner: Retained<MTLBuffer>,
    /// Number of elements (not bytes) in the buffer.
    pub(crate) elem_count: usize,
}

// SAFETY: Metal buffers use ObjC ARC for memory management, which is
// thread-safe. Access to buffer contents is serialised through the command
// queue; no two Rust threads write to the same buffer concurrently.
unsafe impl Send for MtlBuffer {}
unsafe impl Sync for MtlBuffer {}

// ---------------------------------------------------------------------------
// Compiled pipeline cache
// ---------------------------------------------------------------------------

/// Lazily-compiled `MTLComputePipelineState` for a single MSL kernel function.
struct Pipeline {
    state: Retained<MTLComputePipelineState>,
}

// SAFETY: Same rationale as MtlBuffer — ObjC ARC, Metal serialises access.
unsafe impl Send for Pipeline {}
unsafe impl Sync for Pipeline {}

/// All 10 compiled pipelines, cached after `MtlBackend::new()`.
struct Pipelines {
    matmul_f32: Pipeline,
    bmm_f32: Pipeline,
    add_f32: Pipeline,
    sub_f32: Pipeline,
    mul_f32: Pipeline,
    div_f32: Pipeline,
    relu_f32: Pipeline,
    sigmoid_f32: Pipeline,
    softmax_f32: Pipeline,
    sum_axis_f32: Pipeline,
}

// ---------------------------------------------------------------------------
// Helper: compile one MTLLibrary + extract one function + build pipeline
// ---------------------------------------------------------------------------

fn compile_pipeline(device: &MTLDevice, source: &str, fn_name: &str) -> FerrotorchResult<Pipeline> {
    // SAFETY: All objc2-metal calls go through the safe Rust bindings from
    // objc2-metal 0.3.2. The pointer casts inside `NSString::from_str` and
    // `newLibraryWithSource_options_error` are managed by the crate.
    unsafe {
        let src = NSString::from_str(source);
        let options = None; // use default MTLCompileOptions
        let lib: Retained<MTLLibrary> = device
            .newLibraryWithSource_options_error(&src, options)
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("MSL compile failed for `{fn_name}`: {e:?}"),
            })?;

        let name = NSString::from_str(fn_name);
        let func: Retained<MTLFunction> =
            lib.newFunctionWithName(&name)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("MSL function `{fn_name}` not found in library"),
                })?;

        let pipeline: Retained<MTLComputePipelineState> = device
            .newComputePipelineStateWithFunction_error(&func)
            .map_err(|e| FerrotorchError::InvalidArgument {
                message: format!("MTLComputePipelineState creation failed for `{fn_name}`: {e:?}"),
            })?;

        Ok(Pipeline { state: pipeline })
    }
}

// ---------------------------------------------------------------------------
// Threadgroup-width helper (#1101)
// ---------------------------------------------------------------------------

// Power-of-two threadgroup width is required by the in-kernel
// `stride = tcount / 2; stride >>= 1` reduction (softmax_f32, sum_axis_f32).
//
// The reduction loop in those MSL kernels assumes `tcount` is a power of two.
// When `tcount` is not pow-2, the first stride `tcount / 2` rounds *down*, so
// elements in the upper half (indices `2 * stride .. tcount`) are silently
// dropped — producing wrong-but-not-NaN row maxes and partial sums on Apple
// Silicon. PyTorch parity (§3) forbids that silent corruption, so the
// dispatcher rounds the threadgroup width *up* to the next power of two and
// caps it at the Metal threadgroup limit of 1024. The kernel side then
// handles inactive threads (`tid >= cols` / `tid >= axis_len`) by leaving
// the per-thread sentinels untouched (`-INFINITY` for max, `0.0` for sum)
// — the strided init loop short-circuits for those threads and the reduction
// reads the sentinels but they are identity elements for the operation.
//
// Behavioural contract:
//   pow2_tg_width(0)    = 1     // sentinel: zero-width dispatch is a bug
//                                // upstream; we still return a valid Metal
//                                // threadgroup width.
//   pow2_tg_width(1)    = 1
//   pow2_tg_width(13)   = 16
//   pow2_tg_width(257)  = 512
//   pow2_tg_width(1023) = 1024
//   pow2_tg_width(1024) = 1024
//   pow2_tg_width(2000) = 1024  // capped
fn pow2_tg_width(n: usize) -> u64 {
    n.min(1024).next_power_of_two() as u64
}

// ---------------------------------------------------------------------------
// MtlBackend
// ---------------------------------------------------------------------------

/// Apple Metal backend implementing [`GpuBackend`] for ferrotorch-mps.
///
/// Holds a reference to the system default Metal device, a command queue,
/// and the compiled pipeline states for all 10 Sprint C.7 kernels.
pub struct MtlBackend {
    device: Retained<MTLDevice>,
    queue: Retained<MTLCommandQueue>,
    pipelines: Pipelines,
}

// SAFETY: ObjC ARC + command queue serialises all Metal API access.
unsafe impl Send for MtlBackend {}
unsafe impl Sync for MtlBackend {}

impl std::fmt::Debug for MtlBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MtlBackend")
            .field("device", &"<MTLDevice>")
            .finish()
    }
}

impl MtlBackend {
    /// Create a new `MtlBackend`, compiling all 10 MSL kernels eagerly.
    ///
    /// # Errors
    ///
    /// Returns [`FerrotorchError::DeviceUnavailable`] if no Metal device is
    /// found (VM, CI without GPU passthrough, or non-Apple hardware).
    /// Returns [`FerrotorchError::InvalidArgument`] if any MSL kernel fails
    /// to compile — which indicates a ferrotorch bug, not a user error.
    pub fn new() -> FerrotorchResult<Self> {
        // SAFETY: MTLCreateSystemDefaultDevice returns nil when no Metal device
        // is present; the `?` propagates that as DeviceUnavailable.
        let device: Retained<MTLDevice> =
            unsafe { MTLDevice::new() }.ok_or(FerrotorchError::DeviceUnavailable)?;

        let queue: Retained<MTLCommandQueue> =
            unsafe { device.newCommandQueue() }.ok_or_else(|| {
                FerrotorchError::InvalidArgument {
                    message: "MTLDevice::newCommandQueue returned nil".into(),
                }
            })?;

        // Compile all MSL sources — fail fast on any compilation error.
        let mat = compile_pipeline(&device, kernels::MATMUL_F32, "matmul_f32")?;
        let bmm = compile_pipeline(&device, kernels::BMM_F32, "bmm_f32")?;
        let add = compile_pipeline(&device, kernels::ELEMENTWISE_F32, "add_f32")?;
        let sub = compile_pipeline(&device, kernels::ELEMENTWISE_F32, "sub_f32")?;
        let mul = compile_pipeline(&device, kernels::ELEMENTWISE_F32, "mul_f32")?;
        let div_p = compile_pipeline(&device, kernels::ELEMENTWISE_F32, "div_f32")?;
        let relu = compile_pipeline(&device, kernels::ACTIVATIONS_F32, "relu_f32")?;
        let sigmoid = compile_pipeline(&device, kernels::ACTIVATIONS_F32, "sigmoid_f32")?;
        let softmax = compile_pipeline(&device, kernels::SOFTMAX_F32, "softmax_f32")?;
        let sum_ax = compile_pipeline(&device, kernels::SUM_AXIS_F32, "sum_axis_f32")?;

        Ok(Self {
            device,
            queue,
            pipelines: Pipelines {
                matmul_f32: mat,
                bmm_f32: bmm,
                add_f32: add,
                sub_f32: sub,
                mul_f32: mul,
                div_f32: div_p,
                relu_f32: relu,
                sigmoid_f32: sigmoid,
                softmax_f32: softmax,
                sum_axis_f32: sum_ax,
            },
        })
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Allocate a new `MTLBuffer` of `byte_len` bytes (shared storage mode).
    fn alloc_buffer(&self, byte_len: usize, elem_count: usize) -> FerrotorchResult<Arc<MtlBuffer>> {
        // SAFETY: Metal manages the buffer memory; Rust holds a retained ref.
        let buf: Retained<MTLBuffer> = unsafe {
            self.device
                .newBufferWithLength_options(byte_len, MTLStorageMode::Shared as u64)
                .ok_or_else(|| FerrotorchError::InvalidArgument {
                    message: format!("MTLDevice::newBufferWithLength({byte_len}) returned nil"),
                })?
        };
        Ok(Arc::new(MtlBuffer {
            inner: buf,
            elem_count,
        }))
    }

    /// Wrap an `Arc<MtlBuffer>` in a `GpuBufferHandle`.
    fn wrap_buffer(buf: Arc<MtlBuffer>, device_ordinal: usize) -> GpuBufferHandle {
        let len = buf.elem_count;
        GpuBufferHandle::new(Box::new(buf), device_ordinal, len)
    }

    /// Downcast a `GpuBufferHandle` to `&Arc<MtlBuffer>`.
    fn downcast_buf<'a>(handle: &'a GpuBufferHandle) -> FerrotorchResult<&'a Arc<MtlBuffer>> {
        handle
            .downcast_ref::<Arc<MtlBuffer>>()
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "GpuBufferHandle does not contain an Arc<MtlBuffer> (wrong backend?)"
                    .into(),
            })
    }

    /// Commit a command buffer and wait for completion (synchronous).
    ///
    /// All current kernels use synchronous dispatch so callers can read
    /// results immediately. A future async path can replace this with
    /// addScheduledHandler + addCompletedHandler without changing the API.
    fn commit_and_wait(cmd_buf: &MTLCommandBuffer) -> FerrotorchResult<()> {
        unsafe {
            cmd_buf.commit();
            cmd_buf.waitUntilCompleted();
        }
        Ok(())
    }

    /// Launch a 1-D elementwise binary kernel (add/sub/mul/div).
    fn launch_binary_f32(
        &self,
        pipeline: &Pipeline,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let b_buf = Self::downcast_buf(b)?;
        let n = a.len();

        let out_buf = self.alloc_buffer(n * 4, n)?;

        let cmd_buf: Retained<MTLCommandBuffer> = unsafe { self.queue.commandBuffer() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<MTLComputeCommandEncoder> = unsafe { cmd_buf.computeCommandEncoder() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let n_u32 = n as u32;
        unsafe {
            enc.setComputePipelineState(&pipeline.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf.inner), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&n_u32 as *const u32 as *mut _),
                4,
                3,
            );

            let tg_size = pipeline.state.maxTotalThreadsPerThreadgroup().min(256);
            let grid = MTLSize {
                width: n as u64,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf)?;
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    /// Launch a 1-D elementwise unary kernel (relu/sigmoid).
    fn launch_unary_f32(
        &self,
        pipeline: &Pipeline,
        a: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let n = a.len();
        let out_buf = self.alloc_buffer(n * 4, n)?;

        let cmd_buf: Retained<MTLCommandBuffer> = unsafe { self.queue.commandBuffer() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<MTLComputeCommandEncoder> = unsafe { cmd_buf.computeCommandEncoder() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let n_u32 = n as u32;
        unsafe {
            enc.setComputePipelineState(&pipeline.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&n_u32 as *const u32 as *mut _),
                4,
                2,
            );

            let tg_size = pipeline.state.maxTotalThreadsPerThreadgroup().min(256);
            let grid = MTLSize {
                width: n as u64,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_size,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf)?;
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }
}

// ---------------------------------------------------------------------------
// GpuBackend implementation
// ---------------------------------------------------------------------------

impl GpuBackend for MtlBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    // -- Memory management ---------------------------------------------------

    fn cpu_to_gpu(
        &self,
        data: &[u8],
        elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let elem_count = if elem_size == 0 {
            0
        } else {
            data.len() / elem_size
        };
        let buf = self.alloc_buffer(data.len(), elem_count)?;

        // Shared-mode buffers expose a CPU-accessible pointer directly.
        // SAFETY: buffer is exclusively owned here; no GPU command is in
        // flight at this point.
        unsafe {
            let ptr = buf.inner.contents();
            if !ptr.is_null() {
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr.cast::<u8>(), data.len());
            }
        }

        Ok(Self::wrap_buffer(buf, device))
    }

    fn gpu_to_cpu(&self, handle: &GpuBufferHandle) -> FerrotorchResult<Vec<u8>> {
        let buf = Self::downcast_buf(handle)?;
        let byte_len = unsafe { buf.inner.length() } as usize;

        // SAFETY: Shared-mode buffer contents are CPU-accessible after the
        // most recent command buffer completes (guaranteed by commit_and_wait
        // in every kernel dispatch path).
        let slice = unsafe {
            let ptr = buf.inner.contents();
            if ptr.is_null() {
                return Err(FerrotorchError::InvalidArgument {
                    message: "MTLBuffer::contents() returned null on shared-mode buffer".into(),
                });
            }
            std::slice::from_raw_parts(ptr.cast::<u8>(), byte_len)
        };
        Ok(slice.to_vec())
    }

    fn clone_buffer(&self, handle: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        let src_bytes = self.gpu_to_cpu(handle)?;
        self.cpu_to_gpu(&src_bytes, 4, handle.device_ordinal())
    }

    fn alloc_zeros(
        &self,
        len: usize,
        elem_size: usize,
        device: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let byte_len = len * elem_size;
        let buf = self.alloc_buffer(byte_len, len)?;

        // Shared-mode buffers are zero-initialised by the Metal runtime.
        // Explicitly zero for clarity / defence-in-depth.
        unsafe {
            let ptr = buf.inner.contents();
            if !ptr.is_null() {
                std::ptr::write_bytes(ptr.cast::<u8>(), 0u8, byte_len);
            }
        }

        Ok(Self::wrap_buffer(buf, device))
    }

    fn buffer_elem_size(&self, _handle: &GpuBufferHandle) -> usize {
        // MPS buffers in this backend are always f32 (4 bytes).
        4
    }

    // -- Elementwise f32 binary ops ------------------------------------------

    fn add_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_binary_f32(&self.pipelines.add_f32, a, b)
    }

    fn sub_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_binary_f32(&self.pipelines.sub_f32, a, b)
    }

    fn mul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_binary_f32(&self.pipelines.mul_f32, a, b)
    }

    fn div_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
    ) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_binary_f32(&self.pipelines.div_f32, a, b)
    }

    fn neg_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: neg_f32 — follow-up #626".into(),
        })
    }

    // -- Unary activations f32 -----------------------------------------------

    fn relu_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_unary_f32(&self.pipelines.relu_f32, a)
    }

    fn sigmoid_f32(&self, a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        self.launch_unary_f32(&self.pipelines.sigmoid_f32, a)
    }

    // -- Linalg f32 ----------------------------------------------------------

    fn matmul_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let b_buf = Self::downcast_buf(b)?;
        let out_len = m * n;
        let out_buf = self.alloc_buffer(out_len * 4, out_len)?;

        let cmd_buf: Retained<MTLCommandBuffer> = unsafe { self.queue.commandBuffer() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<MTLComputeCommandEncoder> = unsafe { cmd_buf.computeCommandEncoder() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        unsafe {
            enc.setComputePipelineState(&self.pipelines.matmul_f32.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf.inner), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&m_u32 as *const u32 as *mut _),
                4,
                3,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&k_u32 as *const u32 as *mut _),
                4,
                4,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&n_u32 as *const u32 as *mut _),
                4,
                5,
            );

            let tg = MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };
            let grid = MTLSize {
                width: n as u64,
                height: m as u64,
                depth: 1,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf)?;
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    fn bmm_f32(
        &self,
        a: &GpuBufferHandle,
        b: &GpuBufferHandle,
        batch: usize,
        m: usize,
        k: usize,
        n: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let b_buf = Self::downcast_buf(b)?;
        let out_len = batch * m * n;
        let out_buf = self.alloc_buffer(out_len * 4, out_len)?;

        let cmd_buf: Retained<MTLCommandBuffer> = unsafe { self.queue.commandBuffer() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<MTLComputeCommandEncoder> = unsafe { cmd_buf.computeCommandEncoder() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let batch_u32 = batch as u32;
        let m_u32 = m as u32;
        let k_u32 = k as u32;
        let n_u32 = n as u32;

        unsafe {
            enc.setComputePipelineState(&self.pipelines.bmm_f32.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&b_buf.inner), 0, 1);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 2);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&batch_u32 as *const u32 as *mut _),
                4,
                3,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&m_u32 as *const u32 as *mut _),
                4,
                4,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&k_u32 as *const u32 as *mut _),
                4,
                5,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&n_u32 as *const u32 as *mut _),
                4,
                6,
            );

            let tg = MTLSize {
                width: 16,
                height: 16,
                depth: 1,
            };
            let grid = MTLSize {
                width: n as u64,
                height: m as u64,
                depth: batch as u64,
            };
            enc.dispatchThreads_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf)?;
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    // -- Softmax f32 ---------------------------------------------------------

    fn softmax_f32(
        &self,
        a: &GpuBufferHandle,
        rows: usize,
        cols: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;
        let out_len = rows * cols;
        let out_buf = self.alloc_buffer(out_len * 4, out_len)?;

        let cmd_buf: Retained<MTLCommandBuffer> = unsafe { self.queue.commandBuffer() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<MTLComputeCommandEncoder> = unsafe { cmd_buf.computeCommandEncoder() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let rows_u32 = rows as u32;
        let cols_u32 = cols as u32;
        // Each threadgroup handles one row. The kernel's tree reduction
        // (`stride = tcount / 2; stride >>= 1`) requires a pow-2 threadgroup
        // width; pow2_tg_width rounds up and caps at the Metal limit. See
        // pow2_tg_width docs and #1101 for the bug this fixes.
        let tg_w = pow2_tg_width(cols);

        unsafe {
            enc.setComputePipelineState(&self.pipelines.softmax_f32.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&rows_u32 as *const u32 as *mut _),
                4,
                2,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&cols_u32 as *const u32 as *mut _),
                4,
                3,
            );

            // One threadgroup per row.
            let grid = MTLSize {
                width: rows as u64,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf)?;
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    // -- Reductions f32 ------------------------------------------------------

    fn sum_f32(&self, a: &GpuBufferHandle, len: usize) -> FerrotorchResult<GpuBufferHandle> {
        // Reduce full tensor to scalar: treat as (1, len, 1) sum_axis.
        self.sum_axis_f32(a, &[len], 0)
    }

    fn sum_axis_f32(
        &self,
        a: &GpuBufferHandle,
        shape: &[usize],
        axis: usize,
    ) -> FerrotorchResult<GpuBufferHandle> {
        let a_buf = Self::downcast_buf(a)?;

        let outer: usize = shape[..axis].iter().product::<usize>().max(1);
        let axis_len: usize = shape.get(axis).copied().unwrap_or(1);
        let inner: usize = shape[axis + 1..].iter().product::<usize>().max(1);
        let out_len = outer * inner;
        let out_buf = self.alloc_buffer(out_len * 4, out_len)?;

        let cmd_buf: Retained<MTLCommandBuffer> = unsafe { self.queue.commandBuffer() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandQueue::commandBuffer returned nil".into(),
            })?;

        let enc: Retained<MTLComputeCommandEncoder> = unsafe { cmd_buf.computeCommandEncoder() }
            .ok_or_else(|| FerrotorchError::InvalidArgument {
                message: "MTLCommandBuffer::computeCommandEncoder returned nil".into(),
            })?;

        let outer_u32 = outer as u32;
        let axis_u32 = axis_len as u32;
        let inner_u32 = inner as u32;
        // The kernel's tree reduction (`stride = tcount / 2; stride >>= 1`)
        // requires a pow-2 threadgroup width; pow2_tg_width rounds up and
        // caps at the Metal limit. See pow2_tg_width docs and #1101.
        let tg_w = pow2_tg_width(axis_len);

        unsafe {
            enc.setComputePipelineState(&self.pipelines.sum_axis_f32.state);
            enc.setBuffer_offset_atIndex(Some(&a_buf.inner), 0, 0);
            enc.setBuffer_offset_atIndex(Some(&out_buf.inner), 0, 1);
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&outer_u32 as *const u32 as *mut _),
                4,
                2,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&axis_u32 as *const u32 as *mut _),
                4,
                3,
            );
            enc.setBytes_length_atIndex(
                std::ptr::NonNull::new_unchecked(&inner_u32 as *const u32 as *mut _),
                4,
                4,
            );

            // One threadgroup per output element.
            let grid = MTLSize {
                width: out_len as u64,
                height: 1,
                depth: 1,
            };
            let tg = MTLSize {
                width: tg_w,
                height: 1,
                depth: 1,
            };
            enc.dispatchThreadgroups_threadsPerThreadgroup(grid, tg);
            enc.endEncoding();
        }

        Self::commit_and_wait(&cmd_buf)?;
        Ok(Self::wrap_buffer(out_buf, a.device_ordinal()))
    }

    // -- Required abstract methods without Sprint C.7 implementations --------
    // These return structured errors matching PyTorch's NotImplementedError
    // for unregistered backends. Follow-up issues are tracked per method.

    fn gelu_f32(&self, _a: &GpuBufferHandle) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: gelu_f32 — follow-up #626".into(),
        })
    }

    fn dropout_f32(
        &self,
        _a: &GpuBufferHandle,
        _threshold: u32,
        _scale: f32,
        _seed: u32,
    ) -> FerrotorchResult<GpuBufferHandle> {
        Err(FerrotorchError::InvalidArgument {
            message: "MSL kernel needed: dropout_f32 — follow-up #626".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// init_mps_backend entry point
// ---------------------------------------------------------------------------

/// Initialize the MPS Metal backend and register it with `ferrotorch-core`.
///
/// Call once at startup. Returns [`FerrotorchError::DeviceUnavailable`] if no
/// Metal device is present (non-macOS platform or VM without GPU passthrough).
///
/// # Errors
///
/// - [`FerrotorchError::DeviceUnavailable`]: no Metal device found.
/// - [`FerrotorchError::InvalidArgument`]: MSL compilation failed (ferrotorch bug).
pub fn init_mps_backend_metal() -> FerrotorchResult<()> {
    let backend = MtlBackend::new()?;
    ferrotorch_core::gpu_dispatch::register_gpu_backend(Box::new(backend)).map_err(|e| {
        FerrotorchError::InvalidArgument {
            message: format!("MPS backend registration failed: {e}"),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    // These tests are macOS-only (entire module is cfg(target_os = "macos")).
    // On CI without Apple hardware they are excluded by the cfg gate.
    //
    // Note: the `pow2_tg_width_*` tests below are pure-Rust and have no
    // Metal dependency, but they live here because the helper is module-
    // private. They compile and run on macOS only by virtue of the parent
    // `cfg(target_os = "macos")` gate on `pub mod backend` in `lib.rs`.

    /// Pow-2 round-up contract for the threadgroup-width helper (#1101):
    /// non-pow-2 inputs must round up so the in-kernel `stride = tcount/2`
    /// reduction does not silently drop upper-half elements. Cap at 1024
    /// (Metal threadgroup limit).
    #[test]
    fn pow2_tg_width_rounds_up_for_non_powers_of_two() {
        assert_eq!(pow2_tg_width(0), 1);
        assert_eq!(pow2_tg_width(1), 1);
        assert_eq!(pow2_tg_width(2), 2);
        assert_eq!(pow2_tg_width(13), 16);
        assert_eq!(pow2_tg_width(257), 512);
        assert_eq!(pow2_tg_width(1023), 1024);
        assert_eq!(pow2_tg_width(1024), 1024);
        assert_eq!(pow2_tg_width(2000), 1024);
    }

    /// Pow-2 inputs must round-trip unchanged — the helper is idempotent
    /// for already-pow-2 widths within the [1, 1024] Metal cap.
    #[test]
    fn pow2_tg_width_passes_through_powers_of_two() {
        for &n in &[1usize, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024] {
            assert_eq!(
                pow2_tg_width(n),
                n as u64,
                "pow-2 input {n} must round-trip unchanged"
            );
        }
    }

    /// Verify MtlBackend::new() either succeeds or returns DeviceUnavailable.
    /// Never panics, never returns an unexpected error variant.
    #[test]
    fn mtl_backend_new_succeeds_or_unavailable() {
        match MtlBackend::new() {
            Ok(b) => {
                // If a Metal device exists, the debug repr should be non-empty.
                let dbg = format!("{b:?}");
                assert!(dbg.contains("MtlBackend"));
            }
            Err(FerrotorchError::DeviceUnavailable) => {
                // Acceptable: CI macOS runner without GPU passthrough.
            }
            Err(e) => {
                panic!("unexpected error from MtlBackend::new(): {e:?}");
            }
        }
    }

    /// Round-trip f32 data through cpu_to_gpu → gpu_to_cpu on a real Metal device.
    /// cascade_skip if no Metal device is present (CI without GPU).
    #[test]
    fn mtl_buffer_round_trip() {
        let backend = match MtlBackend::new() {
            Ok(b) => b,
            Err(FerrotorchError::DeviceUnavailable) => {
                eprintln!(
                    "  [cascade_skip] mtl_buffer_round_trip — no Metal device, \
                     tracking issue #626"
                );
                return;
            }
            Err(e) => panic!("MtlBackend::new() error: {e:?}"),
        };

        let src: Vec<f32> = vec![1.0_f32, 2.0, 3.0, 4.0];
        let bytes: Vec<u8> = src.iter().flat_map(|f| f.to_le_bytes()).collect();
        let handle = backend.cpu_to_gpu(&bytes, 4, 0).expect("cpu_to_gpu");
        assert_eq!(handle.len(), 4);

        let back = backend.gpu_to_cpu(&handle).expect("gpu_to_cpu");
        assert_eq!(back.len(), bytes.len());
        let floats: Vec<f32> = back
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        assert_eq!(floats, src);
    }
}
