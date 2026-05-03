//! Native bf16 GPU kernels.
//!
//! Hand-written PTX owned by Rust: no CUDA C++ source, no nvrtc
//! runtime compiler, no external toolchain at load time. Each kernel
//! is a `&'static str` containing PTX 7.0 targeting sm_52+, loaded
//! via `cudarc::driver::CudaContext::load_module` through the
//! existing `module_cache::get_or_compile` path.
//!
//! # The bf16 pattern (sm < 90)
//!
//! bf16 is the top 16 bits of an f32 with zero-padded low bits, so
//! in-register conversions are pure bit operations:
//!
//! - **bf16 → f32**: `mov.b32 %f, {0, %h}` where `%h` is a `.b16` and
//!   `%f` can be consumed as `.f32`.  This pattern is taken directly
//!   from NVIDIA's `cuda_bf16.hpp` (`__internal_bfloat162float`) and
//!   from PyTorch's `c10::detail::f32_from_bits` (`tmp <<= 16;
//!   memcpy(&res, &tmp)`).  It is lossless.
//!
//! - **f32 → bf16, round-to-nearest-even**: add the rounding bias
//!   `0x7FFF + bit[16]` to the f32 bits, then shift right 16.  Same
//!   pattern as the existing `F32_TO_BF16_PTX` and as PyTorch's
//!   `round_to_nearest_even` in `BFloat16.h`.
//!
//! All arithmetic happens in `.f32` registers per thread; storage is
//! always `u16` (`.b16`) in global memory.  No whole-tensor f32
//! intermediate materialisation.

#![cfg(feature = "cuda")]

use cudarc::driver::{LaunchConfig, PushKernelArg};

use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
use crate::module_cache::get_or_compile;

const BLOCK_SIZE: u32 = 256;

// ===========================================================================
// Elementwise kernels (mul, add, silu, relu, fatrelu)
// ===========================================================================

const MUL_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mul_bf16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .b16 %a_b16, %b_b16, %zero16;
    .reg .b32 %a_u32, %b_u32, %bits, %round, %lsb;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    ld.global.b16 %b_b16, [%b];
    mov.b16 %zero16, 0;
    mov.b32 %a_u32, {%zero16, %a_b16};
    mov.b32 %b_u32, {%zero16, %b_b16};
    mov.b32 %va, %a_u32;
    mov.b32 %vb, %b_u32;

    mul.f32 %vr, %va, %vb;

    mov.b32 %bits, %vr;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

const ADD_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry add_bf16_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .b16 %a_b16, %b_b16, %zero16;
    .reg .b32 %a_u32, %b_u32, %bits, %round, %lsb;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    ld.global.b16 %b_b16, [%b];
    mov.b16 %zero16, 0;
    mov.b32 %a_u32, {%zero16, %a_b16};
    mov.b32 %b_u32, {%zero16, %b_b16};
    mov.b32 %va, %a_u32;
    mov.b32 %vb, %b_u32;

    add.f32 %vr, %va, %vb;

    mov.b32 %bits, %vr;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

// SiLU: x * sigmoid(x) = x / (1 + exp(-x))
// sigmoid computed via ex2.approx.f32 on -x * log2(e).
const SILU_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry silu_bf16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %zero16;
    .reg .b32 %a_u32, %bits, %round, %lsb;
    .reg .f32 %va, %neg_a, %log2e, %x, %e, %one, %denom, %sig, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    mov.b16 %zero16, 0;
    mov.b32 %a_u32, {%zero16, %a_b16};
    mov.b32 %va, %a_u32;

    neg.f32 %neg_a, %va;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %x, %neg_a, %log2e;
    ex2.approx.f32 %e, %x;
    mov.f32 %one, 0f3F800000;
    add.f32 %denom, %one, %e;
    div.approx.f32 %sig, %one, %denom;
    mul.f32 %vr, %va, %sig;

    mov.b32 %bits, %vr;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

// ReLU: max(0, x)
const RELU_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry relu_bf16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %zero16;
    .reg .b32 %a_u32, %bits, %round, %lsb;
    .reg .f32 %va, %zero, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    mov.b16 %zero16, 0;
    mov.b32 %a_u32, {%zero16, %a_b16};
    mov.b32 %va, %a_u32;

    mov.f32 %zero, 0f00000000;
    max.f32 %vr, %va, %zero;

    mov.b32 %bits, %vr;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

// FATReLU: x if x >= threshold else 0 (ProSparse activation)
const FATRELU_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry fatrelu_bf16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n,
    .param .f32 threshold
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %zero16;
    .reg .b32 %a_u32, %bits, %round, %lsb;
    .reg .f32 %va, %thresh, %zero, %vr;
    .reg .pred %p, %p_ge;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.f32 %thresh, [threshold];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    mov.b16 %zero16, 0;
    mov.b32 %a_u32, {%zero16, %a_b16};
    mov.b32 %va, %a_u32;

    mov.f32 %zero, 0f00000000;
    setp.ge.f32 %p_ge, %va, %thresh;
    selp.f32 %vr, %va, %zero, %p_ge;

    mov.b32 %bits, %vr;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

fn launch_1d(n: usize) -> LaunchConfig {
    let grid = ((n as u32).saturating_add(BLOCK_SIZE - 1)) / BLOCK_SIZE;
    LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

fn launch_binary(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
    ptx: &'static str,
    kernel_name: &'static str,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(ctx, ptx, kernel_name, device.ordinal() as u32).map_err(|e| {
        GpuError::PtxCompileFailed {
            kernel: kernel_name,
            source: e,
        }
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(b)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Elementwise `out = a * b` on bf16 (u16-stored) GPU buffers.
pub fn gpu_mul_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(a, b, device, MUL_BF16_PTX, "mul_bf16_kernel")
}

/// Elementwise `out = a + b` on bf16 (u16-stored) GPU buffers.
pub fn gpu_add_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    b: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    launch_binary(a, b, device, ADD_BF16_PTX, "add_bf16_kernel")
}

/// Elementwise `out = silu(a) = a * sigmoid(a)` on bf16 GPU buffers.
pub fn gpu_silu_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        SILU_BF16_PTX,
        "silu_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "silu_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Elementwise `out = max(0, a)` on bf16 GPU buffers.
pub fn gpu_relu_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        RELU_BF16_PTX,
        "relu_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "relu_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

/// Elementwise `out = a >= threshold ? a : 0` on bf16 GPU buffers.
///
/// ProSparse FATReLU activation: a hard-thresholded ReLU that produces
/// ~89% sparsity on trained models.
pub fn gpu_fatrelu_bf16(
    a: &cudarc::driver::CudaSlice<u16>,
    threshold: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = a.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        FATRELU_BF16_PTX,
        "fatrelu_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "fatrelu_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&threshold)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Embedding gather
// ===========================================================================

// One block per output token; threads stride over `dim` copying u16
// elements. No arithmetic, no precision concerns.
const EMBEDDING_GATHER_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry embedding_gather_bf16_kernel(
    .param .u64 weight_ptr,
    .param .u64 indices_ptr,
    .param .u64 out_ptr,
    .param .u32 n_tokens,
    .param .u32 dim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %dim_reg, %src_row, %col, %src_elem, %dst_elem;
    .reg .u64 %weight, %indices, %out, %off;
    .reg .b16 %v16;
    .reg .pred %p_tok, %p_col;

    ld.param.u64 %weight, [weight_ptr];
    ld.param.u64 %indices, [indices_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n_tokens];
    ld.param.u32 %dim_reg, [dim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p_tok, %bid, %n_reg;
    @%p_tok bra DONE;

    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 2;
    add.u64 %indices, %indices, %off;
    ld.global.u32 %src_row, [%indices];

    mov.u32 %col, %r_tid;
LOOP:
    setp.ge.u32 %p_col, %col, %dim_reg;
    @%p_col bra DONE;

    mul.lo.u32 %src_elem, %src_row, %dim_reg;
    add.u32 %src_elem, %src_elem, %col;
    mul.lo.u32 %dst_elem, %bid, %dim_reg;
    add.u32 %dst_elem, %dst_elem, %col;

    cvt.u64.u32 %off, %src_elem;
    shl.b64 %off, %off, 1;
    add.u64 %off, %weight, %off;
    ld.global.b16 %v16, [%off];

    cvt.u64.u32 %off, %dst_elem;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    st.global.b16 [%off], %v16;

    add.u32 %col, %col, %bdim;
    bra LOOP;

DONE:
    ret;
}
";

pub fn gpu_embedding_gather_bf16(
    weight: &cudarc::driver::CudaSlice<u16>,
    indices: &cudarc::driver::CudaSlice<u32>,
    dim: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n_tokens = indices.len();
    if n_tokens == 0 || dim == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        EMBEDDING_GATHER_BF16_PTX,
        "embedding_gather_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "embedding_gather_bf16_kernel",
        source: e,
    })?;

    let total = n_tokens * dim;
    let mut out = stream.alloc_zeros::<u16>(total)?;
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n_tokens as u32;
    let dim_u32 = dim as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(weight)
            .arg(indices)
            .arg(&mut out)
            .arg(&n_u32)
            .arg(&dim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Embedding gather bf16 → f32 (fused cast)
// ===========================================================================

// Same grid layout as `embedding_gather_bf16_kernel`: one block per
// output row, threads stride over columns. Reads `.b16` from the
// weight table and writes `.f32` to the output via the standard
// `mov.b32 {0, %h}` bf16→f32 expansion.

const EMBEDDING_GATHER_BF16_TO_F32_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry embedding_gather_bf16_to_f32_kernel(
    .param .u64 weight_ptr,
    .param .u64 indices_ptr,
    .param .u64 out_ptr,
    .param .u32 n_tokens,
    .param .u32 dim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %dim_reg, %src_row, %col, %src_elem, %dst_elem;
    .reg .u64 %weight, %indices, %out, %off;
    .reg .b16 %v16, %zero16;
    .reg .b32 %v32;
    .reg .pred %p_tok, %p_col;

    ld.param.u64 %weight, [weight_ptr];
    ld.param.u64 %indices, [indices_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n_tokens];
    ld.param.u32 %dim_reg, [dim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p_tok, %bid, %n_reg;
    @%p_tok bra DONE;

    // Load the row index for this block.
    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 2;
    add.u64 %indices, %indices, %off;
    ld.global.u32 %src_row, [%indices];

    mov.b16 %zero16, 0;
    mov.u32 %col, %r_tid;
LOOP:
    setp.ge.u32 %p_col, %col, %dim_reg;
    @%p_col bra DONE;

    // Source offset: weight[src_row * dim + col], 2 bytes per element.
    mul.lo.u32 %src_elem, %src_row, %dim_reg;
    add.u32 %src_elem, %src_elem, %col;
    cvt.u64.u32 %off, %src_elem;
    shl.b64 %off, %off, 1;
    add.u64 %off, %weight, %off;
    ld.global.b16 %v16, [%off];

    // bf16 -> f32: place bf16 bits in upper half of f32.
    mov.b32 %v32, {%zero16, %v16};

    // Dest offset: out[bid * dim + col], 4 bytes per element.
    mul.lo.u32 %dst_elem, %bid, %dim_reg;
    add.u32 %dst_elem, %dst_elem, %col;
    cvt.u64.u32 %off, %dst_elem;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.b32 [%off], %v32;

    add.u32 %col, %col, %bdim;
    bra LOOP;

DONE:
    ret;
}
";

/// Gather rows from a bf16 weight table and cast to f32 in a single
/// kernel launch. Equivalent to `embedding_gather_bf16` followed by a
/// bf16→f32 cast, but without the intermediate bf16 buffer.
///
/// - `weight`: `[vocab_size, dim]` stored as bf16 (`u16`)
/// - `indices`: `[n_tokens]` as `u32` row indices
/// - Returns: `CudaBuffer<f32>` with `n_tokens * dim` elements
pub fn gpu_embedding_gather_bf16_to_f32(
    weight: &cudarc::driver::CudaSlice<u16>,
    indices: &cudarc::driver::CudaSlice<u32>,
    dim: usize,
    device: &GpuDevice,
) -> GpuResult<crate::buffer::CudaBuffer<f32>> {
    let n_tokens = indices.len();
    if n_tokens == 0 || dim == 0 {
        return crate::transfer::alloc_zeros_f32(0, device);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        EMBEDDING_GATHER_BF16_TO_F32_PTX,
        "embedding_gather_bf16_to_f32_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "embedding_gather_bf16_to_f32_kernel",
        source: e,
    })?;

    let total = n_tokens * dim;
    let mut out = crate::transfer::alloc_zeros_f32(total, device)?;
    let cfg = LaunchConfig {
        grid_dim: (n_tokens as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let n_u32 = n_tokens as u32;
    let dim_u32 = dim as u32;
    let out_slice = out.inner_mut();
    unsafe {
        stream
            .launch_builder(&f)
            .arg(weight)
            .arg(indices)
            .arg(out_slice)
            .arg(&n_u32)
            .arg(&dim_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// RMSNorm — per-row, f32 accumulator, tree reduction in shared memory
// ===========================================================================

// One block per row. Each thread strides over `cols`, accumulating
// sum(x^2) in f32, storing partials to shared memory, then reducing
// via tree-sum. Second pass multiplies by inv_rms and bf16 weight.
const RMSNORM_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 rmsnorm_bf16_sdata[256];

.visible .entry rmsnorm_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 w_ptr,
    .param .u64 out_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .f32 eps
) {
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %otid;
    .reg .u64 %in, %w, %out, %row_off, %off, %sbase, %saddr;
    .reg .b16 %x_b16, %w_b16, %zero16;
    .reg .b32 %x_u32, %w_u32, %bits, %round, %lsb;
    .reg .f32 %x_f, %w_f, %sq_sum, %eps_r, %inv_rms, %mean_sq, %r_f, %r_w, %other, %n_f;
    .reg .pred %p, %lp, %rp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %w, [w_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];
    ld.param.f32 %eps_r, [eps];

    mov.u64 %sbase, rmsnorm_bf16_sdata;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %bid, %rows_reg;
    @%p bra DONE;

    cvt.u64.u32 %row_off, %bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 1;
    cvt.rn.f32.u32 %n_f, %cols_reg;

    mov.b16 %zero16, 0;

    // Phase 1: sum(x^2) in f32
    mov.f32 %sq_sum, 0f00000000;
    mov.u32 %j, %r_tid;
SS:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SSD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;
    fma.rn.f32 %sq_sum, %x_f, %x_f, %sq_sum;
    add.u32 %j, %j, %bdim;
    bra SS;
SSD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sq_sum;
    bar.sync 0;

    mov.u32 %half, %bdim;
SR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra SRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra SRS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sq_sum, [%saddr];
    add.f32 %sq_sum, %sq_sum, %other;
    st.shared.f32 [%saddr], %sq_sum;
SRS:
    bar.sync 0;
    bra SR;
SRD:
    ld.shared.f32 %sq_sum, [%sbase];
    div.approx.f32 %mean_sq, %sq_sum, %n_f;
    add.f32 %mean_sq, %mean_sq, %eps_r;
    sqrt.approx.f32 %inv_rms, %mean_sq;
    rcp.approx.f32 %inv_rms, %inv_rms;
    bar.sync 0;

    // Phase 2: out = x * inv_rms * weight, rounded to bf16
    mov.u32 %j, %r_tid;
NM:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra NMD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;

    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %w, %off;
    ld.global.b16 %w_b16, [%off];
    mov.b32 %w_u32, {%zero16, %w_b16};
    mov.b32 %w_f, %w_u32;

    mul.f32 %r_f, %x_f, %inv_rms;
    mul.f32 %r_f, %r_f, %w_f;

    mov.b32 %bits, %r_f;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;

    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    add.u64 %off, %off, %row_off;
    st.global.u16 [%off], %bits;
    add.u32 %j, %j, %bdim;
    bra NM;
NMD:

DONE:
    ret;
}
";

pub fn gpu_rmsnorm_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    weight: &cudarc::driver::CudaSlice<u16>,
    rows: usize,
    cols: usize,
    eps: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if rows == 0 || cols == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(rows * cols)?);
    }
    if input.len() < rows * cols {
        return Err(GpuError::ShapeMismatch {
            op: "rmsnorm_bf16",
            expected: vec![rows, cols],
            got: vec![input.len()],
        });
    }
    if weight.len() < cols {
        return Err(GpuError::ShapeMismatch {
            op: "rmsnorm_bf16",
            expected: vec![cols],
            got: vec![weight.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        RMSNORM_BF16_PTX,
        "rmsnorm_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "rmsnorm_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(rows * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(weight)
            .arg(&mut out)
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Softmax — row-wise, f32 accumulator, two-pass tree reduction
// ===========================================================================

// One block per row. Pass 1: thread-local max, then shared-memory
// tree-max. Pass 2: thread-local sum of exp(v - row_max), then
// shared-memory tree-sum. Pass 3: write exp((v-row_max) * inv_sum)
// rounded to bf16.
const SOFTMAX_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 softmax_bf16_sdata[256];

.visible .entry softmax_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 rows,
    .param .u32 cols
) {
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %otid;
    .reg .u64 %in, %out, %row_off, %off, %sbase, %saddr;
    .reg .b16 %x_b16, %zero16;
    .reg .b32 %x_u32, %bits, %round, %lsb;
    .reg .f32 %x_f, %tmax, %other, %row_max, %sum, %inv_sum, %e, %scale, %log2e, %y_f;
    .reg .pred %p, %lp, %rp, %gp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];

    mov.u64 %sbase, softmax_bf16_sdata;
    mov.f32 %log2e, 0f3FB8AA3B;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %bid, %rows_reg;
    @%p bra DONE;

    cvt.u64.u32 %row_off, %bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 1;

    mov.b16 %zero16, 0;

    // Pass 1: thread-local max
    mov.f32 %tmax, 0fFF800000;   // -Inf
    mov.u32 %j, %r_tid;
MX:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra MXD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;
    setp.gt.f32 %gp, %x_f, %tmax;
    @%gp mov.f32 %tmax, %x_f;
    add.u32 %j, %j, %bdim;
    bra MX;
MXD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %tmax;
    bar.sync 0;

    mov.u32 %half, %bdim;
MR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra MRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra MRS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %tmax, [%saddr];
    setp.gt.f32 %gp, %other, %tmax;
    @%gp mov.f32 %tmax, %other;
    st.shared.f32 [%saddr], %tmax;
MRS:
    bar.sync 0;
    bra MR;
MRD:
    ld.shared.f32 %row_max, [%sbase];
    bar.sync 0;

    // Pass 2: thread-local sum of exp(v - row_max)
    mov.f32 %sum, 0f00000000;
    mov.u32 %j, %r_tid;
SE:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SED;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;
    sub.f32 %x_f, %x_f, %row_max;
    mul.f32 %scale, %x_f, %log2e;
    ex2.approx.f32 %e, %scale;
    add.f32 %sum, %sum, %e;
    add.u32 %j, %j, %bdim;
    bra SE;
SED:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sum;
    bar.sync 0;

    mov.u32 %half, %bdim;
SER:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra SERD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra SERS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sum, [%saddr];
    add.f32 %sum, %sum, %other;
    st.shared.f32 [%saddr], %sum;
SERS:
    bar.sync 0;
    bra SER;
SERD:
    ld.shared.f32 %sum, [%sbase];
    rcp.approx.f32 %inv_sum, %sum;
    bar.sync 0;

    // Pass 3: write
    mov.u32 %j, %r_tid;
WR:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra WRD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;
    sub.f32 %x_f, %x_f, %row_max;
    mul.f32 %scale, %x_f, %log2e;
    ex2.approx.f32 %e, %scale;
    mul.f32 %y_f, %e, %inv_sum;

    mov.b32 %bits, %y_f;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;

    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    add.u64 %off, %off, %row_off;
    st.global.u16 [%off], %bits;
    add.u32 %j, %j, %bdim;
    bra WR;
WRD:

DONE:
    ret;
}
";

pub fn gpu_softmax_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    rows: usize,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if rows == 0 || cols == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(rows * cols)?);
    }
    if input.len() < rows * cols {
        return Err(GpuError::ShapeMismatch {
            op: "softmax_bf16",
            expected: vec![rows, cols],
            got: vec![input.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        SOFTMAX_BF16_PTX,
        "softmax_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "softmax_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(rows * cols)?;
    let cfg = LaunchConfig {
        grid_dim: (rows as u32, 1, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&rows_u32)
            .arg(&cols_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// RoPE (half-rotation / Llama convention)
// ===========================================================================

// One thread per (head, pos, d<half_dim). Rotates the pair
// (d, d + half_dim) using cos/sin from the precomputed caches.
const ROPE_HALF_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry rope_half_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 cos_ptr,
    .param .u64 sin_ptr,
    .param .u64 out_ptr,
    .param .u32 num_heads,
    .param .u32 seq_len,
    .param .u32 head_dim,
    .param .u32 seq_offset
) {
    .reg .u32 %r_tid, %bid, %bdim, %gid, %nh_reg, %sl_reg, %hd_reg, %so_reg, %half_dim;
    .reg .u32 %d, %tmp, %pos, %head, %cs_idx, %base, %cs_base, %total;
    .reg .u64 %in, %cos_p, %sin_p, %out, %off, %off_base;
    .reg .b16 %x0_b16, %x1_b16, %c_b16, %s_b16, %zero16;
    .reg .b32 %x0_u, %x1_u, %c_u, %s_u, %bits0, %bits1, %round, %lsb;
    .reg .f32 %x0, %x1, %c, %s, %y0, %y1;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %cos_p, [cos_ptr];
    ld.param.u64 %sin_p, [sin_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %nh_reg, [num_heads];
    ld.param.u32 %sl_reg, [seq_len];
    ld.param.u32 %hd_reg, [head_dim];
    ld.param.u32 %so_reg, [seq_offset];

    shr.u32 %half_dim, %hd_reg, 1;

    // total = num_heads * seq_len * half_dim
    mul.lo.u32 %total, %nh_reg, %sl_reg;
    mul.lo.u32 %total, %total, %half_dim;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    // d = gid % half_dim, tmp = gid / half_dim, pos = tmp % seq_len, head = tmp / seq_len
    rem.u32 %d, %gid, %half_dim;
    div.u32 %tmp, %gid, %half_dim;
    rem.u32 %pos, %tmp, %sl_reg;
    div.u32 %head, %tmp, %sl_reg;

    // base = head * seq_len * head_dim + pos * head_dim
    mul.lo.u32 %base, %head, %sl_reg;
    mul.lo.u32 %base, %base, %hd_reg;
    mul.lo.u32 %tmp, %pos, %hd_reg;
    add.u32 %base, %base, %tmp;

    // cs_idx = (seq_offset + pos) * half_dim + d
    add.u32 %cs_base, %so_reg, %pos;
    mul.lo.u32 %cs_idx, %cs_base, %half_dim;
    add.u32 %cs_idx, %cs_idx, %d;

    mov.b16 %zero16, 0;

    // Load input[base + d] and input[base + d + half_dim]
    add.u32 %tmp, %base, %d;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %in, %off;
    ld.global.b16 %x0_b16, [%off_base];

    add.u32 %tmp, %tmp, %half_dim;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %in, %off;
    ld.global.b16 %x1_b16, [%off_base];

    // Load cos/sin
    cvt.u64.u32 %off, %cs_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %cos_p, %off;
    ld.global.b16 %c_b16, [%off_base];
    add.u64 %off_base, %sin_p, %off;
    ld.global.b16 %s_b16, [%off_base];

    // Upcast to f32
    mov.b32 %x0_u, {%zero16, %x0_b16};
    mov.b32 %x1_u, {%zero16, %x1_b16};
    mov.b32 %c_u, {%zero16, %c_b16};
    mov.b32 %s_u, {%zero16, %s_b16};
    mov.b32 %x0, %x0_u;
    mov.b32 %x1, %x1_u;
    mov.b32 %c, %c_u;
    mov.b32 %s, %s_u;

    // y0 = x0*c - x1*s;  y1 = x1*c + x0*s
    mul.f32 %y0, %x0, %c;
    fma.rn.f32 %y0, %x1, 0fBF800000, %y0;   // y0 -= x1 -- wrong, need s factor
    // Redo properly using fma: y0 = x0*c - x1*s = fma(-x1, s, x0*c)
    mul.f32 %y0, %x0, %c;
    neg.f32 %y1, %s;
    fma.rn.f32 %y0, %x1, %y1, %y0;
    mul.f32 %y1, %x1, %c;
    fma.rn.f32 %y1, %x0, %s, %y1;

    // Round-and-store y0 at (base + d)
    mov.b32 %bits0, %y0;
    shr.u32 %lsb, %bits0, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits0, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits0, %round, 16;

    add.u32 %tmp, %base, %d;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %out, %off;
    st.global.u16 [%off_base], %bits0;

    // Round-and-store y1 at (base + d + half_dim)
    mov.b32 %bits1, %y1;
    shr.u32 %lsb, %bits1, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits1, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits1, %round, 16;

    add.u32 %tmp, %tmp, %half_dim;
    cvt.u64.u32 %off, %tmp;
    shl.b64 %off, %off, 1;
    add.u64 %off_base, %out, %off;
    st.global.u16 [%off_base], %bits1;

DONE:
    ret;
}
";

#[allow(clippy::too_many_arguments)]
pub fn gpu_rope_half_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    cos_cache: &cudarc::driver::CudaSlice<u16>,
    sin_cache: &cudarc::driver::CudaSlice<u16>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    seq_offset: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    if head_dim == 0 || head_dim % 2 != 0 {
        return Err(GpuError::ShapeMismatch {
            op: "rope_half_bf16",
            expected: vec![head_dim],
            got: vec![head_dim],
        });
    }
    let total_io = num_heads * seq_len * head_dim;
    if input.len() < total_io {
        return Err(GpuError::ShapeMismatch {
            op: "rope_half_bf16",
            expected: vec![num_heads, seq_len, head_dim],
            got: vec![input.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        ROPE_HALF_BF16_PTX,
        "rope_half_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "rope_half_bf16_kernel",
        source: e,
    })?;

    let half_dim = head_dim / 2;
    let total = num_heads * seq_len * half_dim;
    let mut out = stream.alloc_zeros::<u16>(total_io)?;
    let cfg = launch_1d(total);
    let (nh, sl, hd, so) = (
        num_heads as u32,
        seq_len as u32,
        head_dim as u32,
        seq_offset as u32,
    );
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(cos_cache)
            .arg(sin_cache)
            .arg(&mut out)
            .arg(&nh)
            .arg(&sl)
            .arg(&hd)
            .arg(&so)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// transpose_to_heads: [seq, H*d] -> [H, seq, d]
// ===========================================================================

// Layout transform used before multi-head attention. Read position
// (s, h, d) from the input packed row `[seq, H*d]` where the h-th
// head starts at column h*d, write it to the heads-major output
// `[H, seq, d]` where head h's block starts at h*seq*d.
const TRANSPOSE_TO_HEADS_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry transpose_to_heads_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 num_heads,
    .param .u32 seq_len,
    .param .u32 head_dim
) {
    .reg .u32 %r_tid, %bid, %bdim, %gid, %H, %S, %D, %total, %rem, %d, %s, %h, %src_idx, %dst_idx, %hd;
    .reg .u64 %in, %out, %off;
    .reg .b16 %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %H, [num_heads];
    ld.param.u32 %S, [seq_len];
    ld.param.u32 %D, [head_dim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %r_tid;

    // total = H * S * D
    mul.lo.u32 %total, %H, %S;
    mul.lo.u32 %total, %total, %D;
    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    mul.lo.u32 %hd, %H, %D;     // per-seq stride in input

    // gid = h*S*D + s*D + d
    rem.u32 %d, %gid, %D;
    div.u32 %rem, %gid, %D;
    rem.u32 %s, %rem, %S;
    div.u32 %h, %rem, %S;

    // src: input[s*H*D + h*D + d]
    mul.lo.u32 %src_idx, %s, %hd;
    mad.lo.u32 %src_idx, %h, %D, %src_idx;
    add.u32 %src_idx, %src_idx, %d;

    // dst: output[gid]  (already heads-major)
    mov.u32 %dst_idx, %gid;

    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    ld.global.b16 %v, [%off];

    cvt.u64.u32 %off, %dst_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    st.global.b16 [%off], %v;

DONE:
    ret;
}
";

const TRANSPOSE_FROM_HEADS_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry transpose_from_heads_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 num_heads,
    .param .u32 seq_len,
    .param .u32 head_dim
) {
    .reg .u32 %r_tid, %bid, %bdim, %gid, %H, %S, %D, %total, %rem, %d, %s, %h, %src_idx, %dst_idx, %hd;
    .reg .u64 %in, %out, %off;
    .reg .b16 %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %H, [num_heads];
    ld.param.u32 %S, [seq_len];
    ld.param.u32 %D, [head_dim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %r_tid;

    mul.lo.u32 %total, %H, %S;
    mul.lo.u32 %total, %total, %D;
    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    mul.lo.u32 %hd, %H, %D;

    // Read in heads-major [H, S, D]: input[h*S*D + s*D + d] = gid directly.
    // Decompose gid -> (h, s, d).
    rem.u32 %d, %gid, %D;
    div.u32 %rem, %gid, %D;
    rem.u32 %s, %rem, %S;
    div.u32 %h, %rem, %S;

    // dst_idx = s*H*D + h*D + d
    mul.lo.u32 %dst_idx, %s, %hd;
    mad.lo.u32 %dst_idx, %h, %D, %dst_idx;
    add.u32 %dst_idx, %dst_idx, %d;

    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    ld.global.b16 %v, [%off];

    cvt.u64.u32 %off, %dst_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    st.global.b16 [%off], %v;

DONE:
    ret;
}
";

pub fn gpu_transpose_to_heads_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let total = num_heads * seq_len * head_dim;
    if total == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    if input.len() < total {
        return Err(GpuError::ShapeMismatch {
            op: "transpose_to_heads_bf16",
            expected: vec![seq_len, num_heads, head_dim],
            got: vec![input.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        TRANSPOSE_TO_HEADS_BF16_PTX,
        "transpose_to_heads_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "transpose_to_heads_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(total)?;
    let cfg = launch_1d(total);
    let (nh, sl, hd) = (num_heads as u32, seq_len as u32, head_dim as u32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&nh)
            .arg(&sl)
            .arg(&hd)
            .launch(cfg)?;
    }
    Ok(out)
}

pub fn gpu_transpose_from_heads_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    num_heads: usize,
    seq_len: usize,
    head_dim: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let total = num_heads * seq_len * head_dim;
    if total == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    if input.len() < total {
        return Err(GpuError::ShapeMismatch {
            op: "transpose_from_heads_bf16",
            expected: vec![num_heads, seq_len, head_dim],
            got: vec![input.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        TRANSPOSE_FROM_HEADS_BF16_PTX,
        "transpose_from_heads_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "transpose_from_heads_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(total)?;
    let cfg = launch_1d(total);
    let (nh, sl, hd) = (num_heads as u32, seq_len as u32, head_dim as u32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&nh)
            .arg(&sl)
            .arg(&hd)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// repeat_kv: [Hkv, S, D] -> [Hkv * group_size, S, D]
// ===========================================================================

// For GQA: each KV head is replicated `group_size` times in the head
// axis. Output head h comes from input head h / group_size.
const REPEAT_KV_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry repeat_kv_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 num_kv_heads,
    .param .u32 group_size,
    .param .u32 seq_len,
    .param .u32 head_dim
) {
    .reg .u32 %r_tid, %bid, %bdim, %gid, %Hkv, %G, %Hq, %S, %D, %SD, %total;
    .reg .u32 %rem, %d, %s, %h_out, %h_in, %src_idx;
    .reg .u64 %in, %out, %off;
    .reg .b16 %v;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %Hkv, [num_kv_heads];
    ld.param.u32 %G, [group_size];
    ld.param.u32 %S, [seq_len];
    ld.param.u32 %D, [head_dim];

    mul.lo.u32 %Hq, %Hkv, %G;
    mul.lo.u32 %SD, %S, %D;
    mul.lo.u32 %total, %Hq, %SD;

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    // gid = h_out * S*D + s*D + d
    rem.u32 %d, %gid, %D;
    div.u32 %rem, %gid, %D;
    rem.u32 %s, %rem, %S;
    div.u32 %h_out, %rem, %S;

    // h_in = h_out / group_size
    div.u32 %h_in, %h_out, %G;

    // src_idx = h_in * S*D + s*D + d
    mul.lo.u32 %src_idx, %h_in, %SD;
    mad.lo.u32 %src_idx, %s, %D, %src_idx;
    add.u32 %src_idx, %src_idx, %d;

    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in, %off;
    ld.global.b16 %v, [%off];

    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 1;
    add.u64 %off, %out, %off;
    st.global.b16 [%off], %v;

DONE:
    ret;
}
";

pub fn gpu_repeat_kv_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    num_kv_heads: usize,
    group_size: usize,
    seq_len: usize,
    head_dim: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let total_in = num_kv_heads * seq_len * head_dim;
    let total_out = num_kv_heads * group_size * seq_len * head_dim;
    if total_out == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    if input.len() < total_in {
        return Err(GpuError::ShapeMismatch {
            op: "repeat_kv_bf16",
            expected: vec![num_kv_heads, seq_len, head_dim],
            got: vec![input.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        REPEAT_KV_BF16_PTX,
        "repeat_kv_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "repeat_kv_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(total_out)?;
    let cfg = launch_1d(total_out);
    let (hkv, g, sl, hd) = (
        num_kv_heads as u32,
        group_size as u32,
        seq_len as u32,
        head_dim as u32,
    );
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&hkv)
            .arg(&g)
            .arg(&sl)
            .arg(&hd)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Causal mask: set scores[b, i, j] = -inf (bf16) for j > i
// ===========================================================================

// In-place: rewrites the input buffer (which is a [batch, seq_q, seq_k]
// row-major tensor) so attention scores at (i, j) with j > i are driven
// to a large-negative bf16 value (0xFF80 = -Inf for bf16). One thread
// per (batch, i, j) pair; predicate on j > i.
const CAUSAL_MASK_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry causal_mask_bf16_kernel(
    .param .u64 buf_ptr,
    .param .u32 batch,
    .param .u32 seq_q,
    .param .u32 seq_k
) {
    .reg .u32 %r_tid, %bid, %bdim, %gid, %B, %SQ, %SK, %total, %rem, %j, %i, %b;
    .reg .u32 %neg_inf;
    .reg .u64 %buf, %off;
    .reg .b16 %neg_inf_b;
    .reg .pred %p, %q;

    ld.param.u64 %buf, [buf_ptr];
    ld.param.u32 %B, [batch];
    ld.param.u32 %SQ, [seq_q];
    ld.param.u32 %SK, [seq_k];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %gid, %bid, %bdim, %r_tid;

    mul.lo.u32 %total, %B, %SQ;
    mul.lo.u32 %total, %total, %SK;
    setp.ge.u32 %p, %gid, %total;
    @%p bra DONE;

    // gid = b*SQ*SK + i*SK + j
    rem.u32 %j, %gid, %SK;
    div.u32 %rem, %gid, %SK;
    rem.u32 %i, %rem, %SQ;
    div.u32 %b, %rem, %SQ;

    setp.le.u32 %q, %j, %i;
    @%q bra DONE;

    // bf16 -Inf = 0xFF80
    mov.u32 %neg_inf, 0xFF80;
    cvt.u16.u32 %neg_inf_b, %neg_inf;

    cvt.u64.u32 %off, %gid;
    shl.b64 %off, %off, 1;
    add.u64 %off, %buf, %off;
    st.global.b16 [%off], %neg_inf_b;

DONE:
    ret;
}
";

pub fn gpu_causal_mask_bf16(
    buf: &mut cudarc::driver::CudaSlice<u16>,
    batch: usize,
    seq_q: usize,
    seq_k: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    let total = batch * seq_q * seq_k;
    if total == 0 {
        return Ok(());
    }
    if buf.len() < total {
        return Err(GpuError::ShapeMismatch {
            op: "causal_mask_bf16",
            expected: vec![batch, seq_q, seq_k],
            got: vec![buf.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        CAUSAL_MASK_BF16_PTX,
        "causal_mask_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "causal_mask_bf16_kernel",
        source: e,
    })?;

    let cfg = launch_1d(total);
    let (b, sq, sk) = (batch as u32, seq_q as u32, seq_k as u32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(buf)
            .arg(&b)
            .arg(&sq)
            .arg(&sk)
            .launch(cfg)?;
    }
    Ok(())
}

// ===========================================================================
// Scalar multiply (scale by f32 constant, round-to-bf16)
// ===========================================================================

// Used to fold 1/sqrt(head_dim) into attention scores. Takes a `scale`
// f32 parameter and multiplies the input element-wise.
const SCALE_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry scale_bf16_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .f32 scale,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .b16 %a_b16, %zero16;
    .reg .b32 %a_u32, %bits, %round, %lsb;
    .reg .f32 %va, %scale_r, %vr;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %scale_r, [scale];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 1;
    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.b16 %a_b16, [%a];
    mov.b16 %zero16, 0;
    mov.b32 %a_u32, {%zero16, %a_b16};
    mov.b32 %va, %a_u32;

    mul.f32 %vr, %va, %scale_r;

    mov.b32 %bits, %vr;
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %bits, %round, 16;
    st.global.u16 [%out], %bits;

DONE:
    ret;
}
";

pub fn gpu_scale_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    scale: f32,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    let n = input.len();
    if n == 0 {
        return Ok(device.stream().alloc_zeros::<u16>(0)?);
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        SCALE_BF16_PTX,
        "scale_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "scale_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_1d(n);
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&scale)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Block max-abs reduce  ---- per-block L-infinity magnitude
// ===========================================================================

// For each (row, block) in a [rows, n_blocks * block_size] bf16 input,
// compute max(|x|) over the `block_size` contiguous elements and write
// one f32 per (row, block) to the output.  This is the core "tap"
// kernel for the paged-weight activation profiler: run it on attn_out
// and on the gated MLP activation to get per-head / per-MLP-block
// magnitudes without any CPU round-trip.
//
// One CUDA block per output scalar.  Grid is (n_blocks, rows, 1).
// Each thread strides over block_size, keeping a local f32 max(|x|);
// the warp-wide max is produced by shared-memory tree reduction.
const BLOCK_REDUCE_MAX_ABS_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 block_reduce_max_abs_sdata[256];

.visible .entry block_reduce_max_abs_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 rows,
    .param .u32 n_blocks,
    .param .u32 block_size
) {
    .reg .u32 %r_tid, %bid_b, %bid_r, %bdim, %rows_reg, %nb_reg, %bs_reg, %j, %half, %otid, %flat;
    .reg .u64 %in, %out, %in_off, %out_off, %off, %sbase, %saddr;
    .reg .b16 %x_b16, %zero16;
    .reg .b32 %x_u32;
    .reg .f32 %x_f, %abs_f, %max_f, %other;
    .reg .pred %p, %lp, %rp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %nb_reg, [n_blocks];
    ld.param.u32 %bs_reg, [block_size];

    mov.u64 %sbase, block_reduce_max_abs_sdata;

    mov.u32 %bid_b, %ctaid.x;
    mov.u32 %bid_r, %ctaid.y;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %bid_r, %rows_reg;
    @%p bra DONE;
    setp.ge.u32 %p, %bid_b, %nb_reg;
    @%p bra DONE;

    mov.b16 %zero16, 0;

    // in_off (bytes) = ((bid_r * n_blocks + bid_b) * block_size) * 2
    mul.lo.u32 %flat, %bid_r, %nb_reg;
    add.u32 %flat, %flat, %bid_b;
    mul.lo.u32 %flat, %flat, %bs_reg;
    cvt.u64.u32 %in_off, %flat;
    shl.b64 %in_off, %in_off, 1;
    add.u64 %in_off, %in, %in_off;

    // Phase 1: thread-local max(|x|)
    mov.f32 %max_f, 0f00000000;
    mov.u32 %j, %r_tid;
ML:
    setp.ge.u32 %lp, %j, %bs_reg;
    @%lp bra MLD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 1;
    add.u64 %off, %in_off, %off;
    ld.global.b16 %x_b16, [%off];
    mov.b32 %x_u32, {%zero16, %x_b16};
    mov.b32 %x_f, %x_u32;
    abs.f32 %abs_f, %x_f;
    max.f32 %max_f, %max_f, %abs_f;
    add.u32 %j, %j, %bdim;
    bra ML;
MLD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %max_f;
    bar.sync 0;

    // Shared-memory tree max
    mov.u32 %half, %bdim;
MR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra MRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra MRS;
    add.u32 %otid, %r_tid, %half;
    cvt.u64.u32 %off, %otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %max_f, [%saddr];
    max.f32 %max_f, %max_f, %other;
    st.shared.f32 [%saddr], %max_f;
MRS:
    bar.sync 0;
    bra MR;
MRD:
    // Thread 0 writes the final f32 to out[bid_r * n_blocks + bid_b]
    setp.ne.u32 %p, %r_tid, 0;
    @%p bra DONE;
    ld.shared.f32 %max_f, [%sbase];
    mul.lo.u32 %flat, %bid_r, %nb_reg;
    add.u32 %flat, %flat, %bid_b;
    cvt.u64.u32 %out_off, %flat;
    shl.b64 %out_off, %out_off, 2;
    add.u64 %out_off, %out, %out_off;
    st.global.f32 [%out_off], %max_f;

DONE:
    ret;
}
";

/// Per-block L-infinity magnitude reduction for a bf16 activation tensor.
///
/// Treats `input` as `[rows, n_blocks * block_size]` bf16 and produces
/// `[rows, n_blocks]` f32 where each output is `max(|x|)` over the
/// corresponding `block_size`-wide slice.
///
/// Intended for activation-profiler use cases that tap attention
/// heads (one head == one block) and MLP neuron groups (one block ==
/// `mlp_block_size` neurons) directly on device.
pub fn gpu_block_reduce_max_abs_bf16(
    input: &cudarc::driver::CudaSlice<u16>,
    rows: usize,
    n_blocks: usize,
    block_size: usize,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<f32>> {
    if rows == 0 || n_blocks == 0 || block_size == 0 {
        return Ok(device.stream().alloc_zeros::<f32>(rows * n_blocks)?);
    }
    let expected = rows * n_blocks * block_size;
    if input.len() < expected {
        return Err(GpuError::ShapeMismatch {
            op: "block_reduce_max_abs_bf16",
            expected: vec![rows, n_blocks, block_size],
            got: vec![input.len()],
        });
    }
    let ctx = device.context();
    let stream = device.stream();
    let f = get_or_compile(
        ctx,
        BLOCK_REDUCE_MAX_ABS_BF16_PTX,
        "block_reduce_max_abs_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|e| GpuError::PtxCompileFailed {
        kernel: "block_reduce_max_abs_bf16_kernel",
        source: e,
    })?;

    let mut out = stream.alloc_zeros::<f32>(rows * n_blocks)?;
    let cfg = LaunchConfig {
        grid_dim: (n_blocks as u32, rows as u32, 1),
        block_dim: (BLOCK_SIZE, 1, 1),
        shared_mem_bytes: 0,
    };
    let rows_u32 = rows as u32;
    let nb_u32 = n_blocks as u32;
    let bs_u32 = block_size as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input)
            .arg(&mut out)
            .arg(&rows_u32)
            .arg(&nb_u32)
            .arg(&bs_u32)
            .launch(cfg)?;
    }
    Ok(out)
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn upload_bf16(dev: &GpuDevice, data: &[f32]) -> cudarc::driver::CudaSlice<u16> {
        let bits: Vec<u16> = data
            .iter()
            .map(|&x| half::bf16::from_f32(x).to_bits())
            .collect();
        dev.stream().clone_htod(&bits).expect("upload bf16")
    }

    fn download_bf16(dev: &GpuDevice, buf: &cudarc::driver::CudaSlice<u16>) -> Vec<f32> {
        let bits = dev.stream().clone_dtoh(buf).expect("download bf16");
        bits.into_iter()
            .map(|b| half::bf16::from_bits(b).to_f32())
            .collect()
    }

    #[test]
    fn mul_add_silu_bf16_hand_ptx() {
        let dev = GpuDevice::new(0).expect("cuda device");
        let a = upload_bf16(&dev, &[1.0, 2.0, -3.0, 0.5, 4.0]);
        let b = upload_bf16(&dev, &[2.0, 3.0, 4.0, -1.0, 0.25]);

        let m = gpu_mul_bf16(&a, &b, &dev).expect("mul");
        let s = gpu_add_bf16(&a, &b, &dev).expect("add");
        let si = gpu_silu_bf16(&a, &dev).expect("silu");

        let m_host = download_bf16(&dev, &m);
        let s_host = download_bf16(&dev, &s);
        let si_host = download_bf16(&dev, &si);

        let m_exp = [2.0, 6.0, -12.0, -0.5, 1.0];
        let s_exp = [3.0, 5.0, 1.0, -0.5, 4.25];
        for (g, e) in m_host.iter().zip(m_exp.iter()) {
            assert!((g - e).abs() < e.abs() * 0.02 + 0.01, "mul {g} vs {e}");
        }
        for (g, e) in s_host.iter().zip(s_exp.iter()) {
            assert!((g - e).abs() < e.abs() * 0.02 + 0.01, "add {g} vs {e}");
        }
        let silu_ref: Vec<f32> = [1.0f32, 2.0, -3.0, 0.5, 4.0]
            .iter()
            .map(|&x| x * (1.0 / (1.0 + (-x).exp())))
            .collect();
        for (g, e) in si_host.iter().zip(silu_ref.iter()) {
            assert!((g - e).abs() < e.abs() * 0.02 + 5e-3, "silu {g} vs {e}",);
        }
    }

    #[test]
    fn embedding_gather_bf16_picks_correct_rows() {
        let dev = GpuDevice::new(0).expect("cuda");
        let weight_f: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let weight = upload_bf16(&dev, &weight_f);
        let indices: Vec<u32> = vec![2, 0, 3];
        let idx = dev.stream().clone_htod(&indices).expect("indices");

        let out = gpu_embedding_gather_bf16(&weight, &idx, 3, &dev).expect("gather");
        let got = download_bf16(&dev, &out);
        assert_eq!(got, vec![6.0, 7.0, 8.0, 0.0, 1.0, 2.0, 9.0, 10.0, 11.0]);
    }

    #[test]
    fn rmsnorm_bf16_matches_f32_ground_truth() {
        let dev = GpuDevice::new(0).expect("cuda");
        let rows = 2usize;
        let cols = 8usize;
        let x: Vec<f32> = (0..rows * cols).map(|i| (i as f32 * 0.25) - 1.0).collect();
        let w: Vec<f32> = (0..cols).map(|i| 1.0 + (i as f32) * 0.125).collect();
        let input = upload_bf16(&dev, &x);
        let weight = upload_bf16(&dev, &w);
        let out = gpu_rmsnorm_bf16(&input, &weight, rows, cols, 1e-5, &dev).expect("rmsnorm");
        let got = download_bf16(&dev, &out);

        let x_bf: Vec<f32> = x
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let w_bf: Vec<f32> = w
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        let mut expected = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            let row = &x_bf[r * cols..(r + 1) * cols];
            let mean_sq: f32 = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
            let inv_rms = (mean_sq + 1e-5).sqrt().recip();
            for c in 0..cols {
                expected.push(half::bf16::from_f32(row[c] * inv_rms * w_bf[c]).to_f32());
            }
        }
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < e.abs() * 0.03 + 8e-3,
                "rmsnorm[{i}]: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn softmax_bf16_rows_sum_to_one() {
        let dev = GpuDevice::new(0).expect("cuda");
        let rows = 3;
        let cols = 10;
        let input_f: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.37).sin() + 1.0) * 2.0)
            .collect();
        let input = upload_bf16(&dev, &input_f);
        let out = gpu_softmax_bf16(&input, rows, cols, &dev).expect("softmax");
        let got = download_bf16(&dev, &out);

        for r in 0..rows {
            let row = &got[r * cols..(r + 1) * cols];
            let s: f32 = row.iter().sum();
            assert!((s - 1.0).abs() < 0.05, "row {r} sum = {s}, expected ~1.0",);
            for &v in row {
                assert!(v >= 0.0, "softmax value {v} < 0");
            }
        }
    }

    #[test]
    fn softmax_bf16_matches_f32_ground_truth() {
        let dev = GpuDevice::new(0).expect("cuda");
        let rows = 2;
        let cols = 6;
        let input_f: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5];
        let input = upload_bf16(&dev, &input_f);
        let out = gpu_softmax_bf16(&input, rows, cols, &dev).expect("softmax");
        let got = download_bf16(&dev, &out);

        for r in 0..rows {
            let row = &input_f[r * cols..(r + 1) * cols];
            let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = row.iter().map(|&v| (v - max).exp()).collect();
            let sum: f32 = exps.iter().sum();
            let expected: Vec<f32> = exps.into_iter().map(|e| e / sum).collect();
            for (i, (g, e)) in got[r * cols..(r + 1) * cols]
                .iter()
                .zip(expected.iter())
                .enumerate()
            {
                assert!(
                    (g - e).abs() < e.abs() * 0.04 + 5e-3,
                    "softmax[{r},{i}]: got {g}, expected {e}",
                );
            }
        }
    }

    #[test]
    fn transpose_to_heads_round_trips() {
        let dev = GpuDevice::new(0).expect("cuda");
        let (h, s, d) = (2usize, 3usize, 4usize);
        let data: Vec<f32> = (0..s * h * d).map(|i| i as f32).collect();
        let input = upload_bf16(&dev, &data);
        let th = gpu_transpose_to_heads_bf16(&input, h, s, d, &dev).unwrap();
        let back = gpu_transpose_from_heads_bf16(&th, h, s, d, &dev).unwrap();
        let got = download_bf16(&dev, &back);
        for (g, e) in got.iter().zip(data.iter()) {
            assert!((g - e).abs() < 1e-3, "round-trip {g} vs {e}");
        }
    }

    #[test]
    fn transpose_to_heads_picks_right_values() {
        // Input [seq=2, H=2, D=3]: rows [a_h0, a_h1, b_h0, b_h1]
        // Output [H=2, seq=2, D=3]: [h0_a, h0_b, h1_a, h1_b]
        let dev = GpuDevice::new(0).expect("cuda");
        let data: Vec<f32> = vec![
            10.0, 11.0, 12.0, // s=0 h=0
            20.0, 21.0, 22.0, // s=0 h=1
            30.0, 31.0, 32.0, // s=1 h=0
            40.0, 41.0, 42.0, // s=1 h=1
        ];
        let input = upload_bf16(&dev, &data);
        let out = gpu_transpose_to_heads_bf16(&input, 2, 2, 3, &dev).unwrap();
        let got = download_bf16(&dev, &out);
        assert_eq!(
            got,
            vec![
                10.0, 11.0, 12.0, 30.0, 31.0, 32.0, 20.0, 21.0, 22.0, 40.0, 41.0, 42.0
            ]
        );
    }

    #[test]
    fn repeat_kv_broadcasts_correctly() {
        // Hkv=2, G=2, S=1, D=3
        let dev = GpuDevice::new(0).expect("cuda");
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0];
        let input = upload_bf16(&dev, &data);
        let out = gpu_repeat_kv_bf16(&input, 2, 2, 1, 3, &dev).unwrap();
        let got = download_bf16(&dev, &out);
        assert_eq!(
            got,
            vec![
                1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 10.0, 20.0, 30.0, 10.0, 20.0, 30.0
            ]
        );
    }

    #[test]
    fn causal_mask_zeros_upper_triangle() {
        let dev = GpuDevice::new(0).expect("cuda");
        // 4x4 scores, all 1.0. After causal mask, upper triangle is -Inf.
        let scores: Vec<f32> = vec![1.0; 16];
        let mut buf = upload_bf16(&dev, &scores);
        gpu_causal_mask_bf16(&mut buf, 1, 4, 4, &dev).unwrap();
        let got = download_bf16(&dev, &buf);
        for i in 0..4usize {
            for j in 0..4usize {
                let v = got[i * 4 + j];
                if j <= i {
                    assert!((v - 1.0).abs() < 1e-3, "[{i},{j}] = {v}, expected 1.0");
                } else {
                    assert!(v.is_infinite() && v < 0.0, "[{i},{j}] = {v}, expected -Inf",);
                }
            }
        }
    }

    #[test]
    fn scale_bf16_multiplies_by_scalar() {
        let dev = GpuDevice::new(0).expect("cuda");
        let data: Vec<f32> = vec![1.0, 2.0, -3.0, 4.5];
        let input = upload_bf16(&dev, &data);
        let out = gpu_scale_bf16(&input, 0.5, &dev).unwrap();
        let got = download_bf16(&dev, &out);
        let expected: Vec<f32> = data.iter().map(|x| x * 0.5).collect();
        for (g, e) in got.iter().zip(expected.iter()) {
            assert!((g - e).abs() < e.abs() * 0.02 + 5e-3, "{g} vs {e}");
        }
    }

    #[test]
    fn rope_half_bf16_identity_at_pos_zero() {
        let dev = GpuDevice::new(0).expect("cuda");
        let num_heads = 2usize;
        let seq_len = 1usize;
        let head_dim = 8usize;
        let input: Vec<f32> = (0..num_heads * seq_len * head_dim)
            .map(|i| (i as f32) * 0.125 - 0.5)
            .collect();

        let max_seq = 4;
        let half_dim = head_dim / 2;
        let mut cos_buf = vec![0.0f32; max_seq * half_dim];
        let sin_buf = vec![0.0f32; max_seq * half_dim];
        for c in cos_buf.iter_mut().take(half_dim) {
            *c = 1.0;
        }

        let input_g = upload_bf16(&dev, &input);
        let cos_g = upload_bf16(&dev, &cos_buf);
        let sin_g = upload_bf16(&dev, &sin_buf);

        let out = gpu_rope_half_bf16(
            &input_g, &cos_g, &sin_g, num_heads, seq_len, head_dim, 0, &dev,
        )
        .expect("rope");
        let got = download_bf16(&dev, &out);

        let expected: Vec<f32> = input
            .iter()
            .map(|&v| half::bf16::from_f32(v).to_f32())
            .collect();
        for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (g - e).abs() < 1e-3,
                "rope identity[{i}]: got {g}, expected {e}",
            );
        }
    }

    #[test]
    fn block_reduce_max_abs_bf16_matches_cpu_groundtruth() {
        let dev = GpuDevice::new(0).expect("cuda");
        // 3 rows, 4 blocks per row, 8 elements per block. Fill with values
        // whose absolute max we can predict per block.
        let rows = 3usize;
        let n_blocks = 4usize;
        let block_size = 8usize;
        let total = rows * n_blocks * block_size;

        let mut data: Vec<f32> = Vec::with_capacity(total);
        for r in 0..rows {
            for b in 0..n_blocks {
                for i in 0..block_size {
                    // Sign flips with `i` so abs matters; magnitude hits
                    // a predictable peak at i == block_size - 1.
                    let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                    let mag = (r as f32) * 10.0 + (b as f32) + (i as f32) * 0.125;
                    data.push(sign * mag);
                }
            }
        }

        let input = upload_bf16(&dev, &data);
        let out = gpu_block_reduce_max_abs_bf16(&input, rows, n_blocks, block_size, &dev)
            .expect("block_reduce");
        let got: Vec<f32> = dev.stream().clone_dtoh(&out).expect("download f32");

        assert_eq!(got.len(), rows * n_blocks);

        // CPU ground truth with matching bf16 rounding.
        for r in 0..rows {
            for b in 0..n_blocks {
                let base = (r * n_blocks + b) * block_size;
                let expected = (0..block_size)
                    .map(|i| half::bf16::from_f32(data[base + i]).to_f32().abs())
                    .fold(0.0f32, f32::max);
                let g = got[r * n_blocks + b];
                assert!(
                    (g - expected).abs() < expected.abs() * 0.01 + 1e-3,
                    "block_reduce[{r},{b}]: got {g}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn block_reduce_max_abs_bf16_single_block_row() {
        // Edge case: n_blocks = 1, i.e. one L-inf norm per row. This is
        // how the attention tap reduces [n_heads * seq, head_dim] ->
        // [n_heads * seq] magnitudes.
        let dev = GpuDevice::new(0).expect("cuda");
        let rows = 5usize;
        let block_size = 16usize;
        let mut data: Vec<f32> = Vec::with_capacity(rows * block_size);
        for r in 0..rows {
            for i in 0..block_size {
                let sign = if (r + i) % 3 == 0 { -1.0 } else { 1.0 };
                data.push(sign * ((r as f32 + 1.0) * 0.5 + (i as f32) * 0.0625));
            }
        }
        let input = upload_bf16(&dev, &data);
        let out = gpu_block_reduce_max_abs_bf16(&input, rows, 1, block_size, &dev)
            .expect("block_reduce single");
        let got: Vec<f32> = dev.stream().clone_dtoh(&out).expect("download f32");
        assert_eq!(got.len(), rows);
        for r in 0..rows {
            let expected = (0..block_size)
                .map(|i| {
                    half::bf16::from_f32(data[r * block_size + i])
                        .to_f32()
                        .abs()
                })
                .fold(0.0f32, f32::max);
            let g = got[r];
            assert!(
                (g - expected).abs() < expected.abs() * 0.01 + 1e-3,
                "single-block row[{r}]: got {g}, expected {expected}"
            );
        }
    }
}
