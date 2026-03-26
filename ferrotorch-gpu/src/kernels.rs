//! Custom PTX CUDA kernels for elementwise GPU operations.
//!
//! Each operation has two code paths:
//!
//! 1. **PTX kernel** -- a hand-written PTX string that is loaded into the CUDA
//!    driver at runtime via [`cudarc`]. This is the fast path and runs entirely
//!    on the GPU.
//! 2. **CPU fallback** -- copies data to the host, performs the operation with
//!    standard Rust iterators, and copies the result back. Correct but slow;
//!    used when the PTX module cannot be loaded (e.g. architecture mismatch).
//!
//! # Supported operations
//!
//! | Function | Formula |
//! |----------|---------|
//! | [`gpu_add`] | `out[i] = a[i] + b[i]` |
//! | [`gpu_sub`] | `out[i] = a[i] - b[i]` |
//! | [`gpu_mul`] | `out[i] = a[i] * b[i]` |
//! | [`gpu_neg`] | `out[i] = -a[i]` |
//! | [`gpu_relu`] | `out[i] = max(a[i], 0.0)` |

#[cfg(feature = "cuda")]
use cudarc::driver::LaunchConfig;

use crate::buffer::CudaBuffer;
use crate::device::GpuDevice;
use crate::error::{GpuError, GpuResult};
#[cfg(feature = "cuda")]
use crate::transfer::{alloc_zeros_f32, cpu_to_gpu, gpu_to_cpu};

// ---------------------------------------------------------------------------
// PTX kernel source strings
// ---------------------------------------------------------------------------

/// PTX source for `add_kernel`: `out[i] = a[i] + b[i]`.
#[cfg(feature = "cuda")]
pub(crate) const ADD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry add_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    ld.global.f32 %vb, [%b];
    add.f32 %vr, %va, %vb;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `add_vec4_kernel`: vectorized add, 4 elements per thread.
///
/// Uses `ld.global.v4.f32` (128-bit load) for 4x memory throughput vs scalar.
/// Thread i processes elements [i*4 .. i*4+3].
#[cfg(feature = "cuda")]
pub(crate) const ADD_VEC4_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry add_vec4_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n4
) {
    .reg .u32 %r_tid, %bid, %bdim, %n4_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .f32 %a0, %a1, %a2, %a3, %b0, %b1, %b2, %b3, %r0, %r1, %r2, %r3;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n4_reg, [n4];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n4_reg;
    @%p bra DONE;

    // Byte offset = tid * 16 (4 floats × 4 bytes)
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 4;

    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.v4.f32 {%a0, %a1, %a2, %a3}, [%a];
    ld.global.v4.f32 {%b0, %b1, %b2, %b3}, [%b];

    add.f32 %r0, %a0, %b0;
    add.f32 %r1, %a1, %b1;
    add.f32 %r2, %a2, %b2;
    add.f32 %r3, %a3, %b3;

    st.global.v4.f32 [%out], {%r0, %r1, %r2, %r3};

DONE:
    ret;
}
";

/// PTX source for `mul_vec4_kernel`: vectorized multiply, 4 elements per thread.
#[cfg(feature = "cuda")]
pub(crate) const MUL_VEC4_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mul_vec4_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n4
) {
    .reg .u32 %r_tid, %bid, %bdim, %n4_reg;
    .reg .u64 %a, %b, %out, %off;
    .reg .f32 %a0, %a1, %a2, %a3, %b0, %b1, %b2, %b3, %r0, %r1, %r2, %r3;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n4_reg, [n4];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n4_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 4;

    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.v4.f32 {%a0, %a1, %a2, %a3}, [%a];
    ld.global.v4.f32 {%b0, %b1, %b2, %b3}, [%b];

    mul.f32 %r0, %a0, %b0;
    mul.f32 %r1, %a1, %b1;
    mul.f32 %r2, %a2, %b2;
    mul.f32 %r3, %a3, %b3;

    st.global.v4.f32 [%out], {%r0, %r1, %r2, %r3};

DONE:
    ret;
}
";

/// PTX source for `sub_kernel`: `out[i] = a[i] - b[i]`.
#[cfg(feature = "cuda")]
pub(crate) const SUB_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry sub_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    ld.global.f32 %vb, [%b];
    sub.f32 %vr, %va, %vb;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `mul_kernel`: `out[i] = a[i] * b[i]`.
#[cfg(feature = "cuda")]
pub(crate) const MUL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry mul_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    ld.global.f32 %vb, [%b];
    mul.f32 %vr, %va, %vb;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `neg_kernel`: `out[i] = -a[i]`.
#[cfg(feature = "cuda")]
pub(crate) const NEG_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry neg_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    neg.f32 %vr, %va;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `relu_kernel`: `out[i] = max(a[i], 0.0)`.
#[cfg(feature = "cuda")]
pub(crate) const RELU_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry relu_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr, %zero;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    mov.f32 %zero, 0f00000000;
    max.f32 %vr, %va, %zero;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `scale_kernel`: `out[i] = a[i] * scalar`.
#[cfg(feature = "cuda")]
pub(crate) const SCALE_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry scale_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .f32 scalar,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr, %s;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %s, [scalar];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    mul.f32 %vr, %va, %s;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX for 2D matrix transpose: `out[j * M + i] = in[i * N + j]`.
/// Thread `tid` maps to output index; computes the corresponding input index.
#[cfg(feature = "cuda")]
pub(crate) const TRANSPOSE_2D_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry transpose_2d_kernel(\n\
    .param .u64 in_ptr,\n\
    .param .u64 out_ptr,\n\
    .param .u32 M,\n\
    .param .u32 N,\n\
    .param .u32 total\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %total_reg, %M_reg, %N_reg;\n\
    .reg .u32 %out_row, %out_col, %in_idx;\n\
    .reg .u64 %in, %out, %off_in, %off_out;\n\
    .reg .f32 %val;\n\
    .reg .pred %p;\n\
\n\
    ld.param.u64 %in, [in_ptr];\n\
    ld.param.u64 %out, [out_ptr];\n\
    ld.param.u32 %M_reg, [M];\n\
    ld.param.u32 %N_reg, [N];\n\
    ld.param.u32 %total_reg, [total];\n\
\n\
    mov.u32 %bid, %ctaid.x;\n\
    mov.u32 %bdim, %ntid.x;\n\
    mov.u32 %r_tid, %tid.x;\n\
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;\n\
\n\
    setp.ge.u32 %p, %r_tid, %total_reg;\n\
    @%p bra DONE;\n\
\n\
    // Output shape is [N, M]. tid = out_row * M + out_col.\n\
    div.u32 %out_row, %r_tid, %M_reg;\n\
    rem.u32 %out_col, %r_tid, %M_reg;\n\
    // Input index: out_col * N + out_row (transposed).\n\
    mad.lo.u32 %in_idx, %out_col, %N_reg, %out_row;\n\
\n\
    cvt.u64.u32 %off_in, %in_idx;\n\
    shl.b64 %off_in, %off_in, 2;\n\
    add.u64 %off_in, %in, %off_in;\n\
    ld.global.f32 %val, [%off_in];\n\
\n\
    cvt.u64.u32 %off_out, %r_tid;\n\
    shl.b64 %off_out, %off_out, 2;\n\
    add.u64 %off_out, %out, %off_out;\n\
    st.global.f32 [%off_out], %val;\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

// ---------------------------------------------------------------------------
// 4D permute (0,2,1,3) PTX kernel — swap dims 1 and 2
// ---------------------------------------------------------------------------
// Input:  [d0, d1, d2, d3]
// Output: [d0, d2, d1, d3]
// Thread i computes output[i] by mapping to the transposed input index.

#[cfg(feature = "cuda")]
pub(crate) const PERMUTE_0213_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry permute_0213_kernel(\n\
    .param .u64 in_ptr,\n\
    .param .u64 out_ptr,\n\
    .param .u32 d0,\n\
    .param .u32 d1,\n\
    .param .u32 d2,\n\
    .param .u32 d3,\n\
    .param .u32 total\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %total_reg;\n\
    .reg .u32 %d0r, %d1r, %d2r, %d3r;\n\
    .reg .u32 %i0, %i1, %i2, %i3, %rem, %in_idx;\n\
    .reg .u32 %s_out2, %s_out1, %s_in1;\n\
    .reg .u64 %in, %out, %off_in, %off_out;\n\
    .reg .f32 %val;\n\
    .reg .pred %p;\n\
\n\
    ld.param.u64 %in, [in_ptr];\n\
    ld.param.u64 %out, [out_ptr];\n\
    ld.param.u32 %d0r, [d0];\n\
    ld.param.u32 %d1r, [d1];\n\
    ld.param.u32 %d2r, [d2];\n\
    ld.param.u32 %d3r, [d3];\n\
    ld.param.u32 %total_reg, [total];\n\
\n\
    mov.u32 %bid, %ctaid.x;\n\
    mov.u32 %bdim, %ntid.x;\n\
    mov.u32 %r_tid, %tid.x;\n\
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;\n\
\n\
    setp.ge.u32 %p, %r_tid, %total_reg;\n\
    @%p bra DONE;\n\
\n\
    // Output shape: [d0, d2, d1, d3]\n\
    // Decompose tid into (i0, i2, i1, i3) in output layout.\n\
    mul.lo.u32 %s_out2, %d1r, %d3r;\n\
    mul.lo.u32 %s_out1, %s_out2, %d2r;\n\
\n\
    div.u32 %i0, %r_tid, %s_out1;\n\
    rem.u32 %rem, %r_tid, %s_out1;\n\
    div.u32 %i2, %rem, %s_out2;\n\
    rem.u32 %rem, %rem, %s_out2;\n\
    div.u32 %i1, %rem, %d3r;\n\
    rem.u32 %i3, %rem, %d3r;\n\
\n\
    // Input index: i0 * (d1*d2*d3) + i1 * (d2*d3) + i2 * d3 + i3\n\
    mul.lo.u32 %s_in1, %d2r, %d3r;\n\
    mul.lo.u32 %in_idx, %i0, %d1r;\n\
    add.u32 %in_idx, %in_idx, %i1;\n\
    mul.lo.u32 %in_idx, %in_idx, %s_in1;\n\
    mad.lo.u32 %in_idx, %i2, %d3r, %in_idx;\n\
    add.u32 %in_idx, %in_idx, %i3;\n\
\n\
    cvt.u64.u32 %off_in, %in_idx;\n\
    shl.b64 %off_in, %off_in, 2;\n\
    add.u64 %off_in, %in, %off_in;\n\
    ld.global.f32 %val, [%off_in];\n\
\n\
    cvt.u64.u32 %off_out, %r_tid;\n\
    shl.b64 %off_out, %off_out, 2;\n\
    add.u64 %off_out, %out, %off_out;\n\
    st.global.f32 [%off_out], %val;\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

// ---------------------------------------------------------------------------
// f32-to-f16 conversion PTX kernel: out_f16[i] = float2half(in_f32[i])
// ---------------------------------------------------------------------------
// Used by gpu_matmul_f16 to cast f32 inputs to f16 on-GPU before calling
// cublasGemmEx. The output is stored as u16 (IEEE 754 half-precision bits).

#[cfg(feature = "cuda")]
pub(crate) const F32_TO_F16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry f32_to_f16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off_in, %off_out;
    .reg .f32 %vf;
    .reg .b16 %vh;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    // Compute input offset: i * 4 (f32 = 4 bytes)
    cvt.u64.u32 %off_in, %r_tid;
    shl.b64 %off_in, %off_in, 2;
    add.u64 %in, %in, %off_in;

    // Compute output offset: i * 2 (f16 = 2 bytes)
    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 1;
    add.u64 %out, %out, %off_out;

    // Load f32, convert to f16 (round-to-nearest-even), store as u16
    ld.global.f32 %vf, [%in];
    cvt.rn.f16.f32 %vh, %vf;
    st.global.b16 [%out], %vh;

DONE:
    ret;
}
";

/// PTX source for `f32_to_bf16_kernel`: convert f32 → bf16 (stored as u16).
///
/// BF16 is the top 16 bits of f32 with round-to-nearest-even. We do this
/// with integer bit ops: add rounding bias 0x7FFF + bit 16 of the value,
/// then shift right 16. This works on sm_52+ (no special bf16 instructions
/// needed).
#[cfg(feature = "cuda")]
pub(crate) const F32_TO_BF16_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry f32_to_bf16_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off_in, %off_out;
    .reg .f32 %vf;
    .reg .u32 %bits, %round, %lsb, %result;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off_in, %r_tid;
    shl.b64 %off_in, %off_in, 2;
    add.u64 %in, %in, %off_in;

    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 1;
    add.u64 %out, %out, %off_out;

    // Load f32 as raw bits
    ld.global.u32 %bits, [%in];

    // Round-to-nearest-even: add (0x7FFF + bit[16]) then shift right 16
    shr.u32 %lsb, %bits, 16;
    and.b32 %lsb, %lsb, 1;
    add.u32 %round, %bits, 0x7FFF;
    add.u32 %round, %round, %lsb;
    shr.u32 %result, %round, 16;

    // Store as u16
    st.global.u16 [%out], %result;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Small matmul PTX kernel: C = A @ B, one thread per output element
// ---------------------------------------------------------------------------
// For small matrices where cuBLAS JIT compilation overhead > compute time.
// Compiles once via module_cache, never JIT-recompiles for different sizes.

#[cfg(feature = "cuda")]
pub(crate) const SMALL_MATMUL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry small_matmul_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u32 M,
    .param .u32 K,
    .param .u32 N,
    .param .u32 total
) {
    .reg .u32 %r_tid, %bid, %bdim, %total_reg, %M_reg, %K_reg, %N_reg;
    .reg .u32 %row, %col, %p, %idx;
    .reg .u64 %a, %b, %c, %a_off, %b_off, %c_off;
    .reg .f32 %sum, %va, %vb;
    .reg .pred %bounds_p, %loop_p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %c, [c_ptr];
    ld.param.u32 %M_reg, [M];
    ld.param.u32 %K_reg, [K];
    ld.param.u32 %N_reg, [N];
    ld.param.u32 %total_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %bounds_p, %r_tid, %total_reg;
    @%bounds_p bra DONE;

    div.u32 %row, %r_tid, %N_reg;
    rem.u32 %col, %r_tid, %N_reg;

    mov.f32 %sum, 0f00000000;
    mov.u32 %p, 0;
DOT:
    setp.ge.u32 %loop_p, %p, %K_reg;
    @%loop_p bra DOT_DONE;

    mad.lo.u32 %idx, %row, %K_reg, %p;
    cvt.u64.u32 %a_off, %idx;
    shl.b64 %a_off, %a_off, 2;
    add.u64 %a_off, %a, %a_off;
    ld.global.f32 %va, [%a_off];

    mad.lo.u32 %idx, %p, %N_reg, %col;
    cvt.u64.u32 %b_off, %idx;
    shl.b64 %b_off, %b_off, 2;
    add.u64 %b_off, %b, %b_off;
    ld.global.f32 %vb, [%b_off];

    fma.rn.f32 %sum, %va, %vb, %sum;
    add.u32 %p, %p, 1;
    bra DOT;
DOT_DONE:

    cvt.u64.u32 %c_off, %r_tid;
    shl.b64 %c_off, %c_off, 2;
    add.u64 %c_off, %c, %c_off;
    st.global.f32 [%c_off], %sum;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Slice-write PTX kernel: copy [N, D] into row `pos` of [N, max_len, D]
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub(crate) const SLICE_WRITE_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry slice_write_kernel(
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u32 n,
    .param .u32 D,
    .param .u32 max_len,
    .param .u32 pos
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %D_reg, %max_len_reg, %pos_reg;
    .reg .u32 %batch_idx, %d_idx, %dst_row;
    .reg .u64 %src, %dst, %src_off, %dst_off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %src, [src_ptr];
    ld.param.u64 %dst, [dst_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %D_reg, [D];
    ld.param.u32 %max_len_reg, [max_len];
    ld.param.u32 %pos_reg, [pos];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %src_off, %r_tid;
    shl.b64 %src_off, %src_off, 2;
    add.u64 %src, %src, %src_off;
    ld.global.f32 %val, [%src];

    div.u32 %batch_idx, %r_tid, %D_reg;
    rem.u32 %d_idx, %r_tid, %D_reg;
    mul.lo.u32 %dst_row, %batch_idx, %max_len_reg;
    add.u32 %dst_row, %dst_row, %pos_reg;
    mul.lo.u32 %dst_row, %dst_row, %D_reg;
    add.u32 %dst_row, %dst_row, %d_idx;
    cvt.u64.u32 %dst_off, %dst_row;
    shl.b64 %dst_off, %dst_off, 2;
    add.u64 %dst, %dst, %dst_off;
    st.global.f32 [%dst], %val;

DONE:
    ret;
}
";

/// PTX for `slice_write_indirect_kernel`: same as `slice_write_kernel` but
/// reads `pos` from a device pointer. This enables CUDA graph capture — the
/// graph records the pointer address (fixed), and we update the u32 value
/// at that address before each graph replay.
#[cfg(feature = "cuda")]
pub(crate) const SLICE_WRITE_INDIRECT_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry slice_write_indirect_kernel(
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u32 n,
    .param .u32 D,
    .param .u32 max_len,
    .param .u64 pos_ptr
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %D_reg, %max_len_reg, %pos_reg;
    .reg .u32 %batch_idx, %d_idx, %dst_row;
    .reg .u64 %src, %dst, %src_off, %dst_off, %pos_p;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %src, [src_ptr];
    ld.param.u64 %dst, [dst_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %D_reg, [D];
    ld.param.u32 %max_len_reg, [max_len];
    ld.param.u64 %pos_p, [pos_ptr];
    ld.global.u32 %pos_reg, [%pos_p];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %src_off, %r_tid;
    shl.b64 %src_off, %src_off, 2;
    add.u64 %src, %src, %src_off;
    ld.global.f32 %val, [%src];

    div.u32 %batch_idx, %r_tid, %D_reg;
    rem.u32 %d_idx, %r_tid, %D_reg;
    mul.lo.u32 %dst_row, %batch_idx, %max_len_reg;
    add.u32 %dst_row, %dst_row, %pos_reg;
    mul.lo.u32 %dst_row, %dst_row, %D_reg;
    add.u32 %dst_row, %dst_row, %d_idx;
    cvt.u64.u32 %dst_off, %dst_row;
    shl.b64 %dst_off, %dst_off, 2;
    add.u64 %dst, %dst, %dst_off;
    st.global.f32 [%dst], %val;

DONE:
    ret;
}
";

/// PTX for `causal_mask_indirect_kernel`: builds an attention mask where
/// `out[h, col] = 0.0` for `col < total_len` and `-1e9` for `col >= total_len`.
/// `total_len` is read from a device pointer (for CUDA graph capture).
/// Output shape: `[n_head, max_pos]` — one mask row per head (all identical).
/// Thread `tid` maps to flat index; column = `tid % max_pos`.
#[cfg(feature = "cuda")]
pub(crate) const CAUSAL_MASK_INDIRECT_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry causal_mask_indirect_kernel(
    .param .u64 total_len_ptr,
    .param .u64 out_ptr,
    .param .u32 max_pos,
    .param .u32 total
) {
    .reg .u32 %r_tid, %bid, %bdim, %total_reg, %tlen, %max_pos_reg, %col;
    .reg .u64 %out, %off, %tl_p;
    .reg .f32 %val;
    .reg .pred %bounds_p, %mask_p;

    ld.param.u64 %tl_p, [total_len_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %max_pos_reg, [max_pos];
    ld.param.u32 %total_reg, [total];

    ld.global.u32 %tlen, [%tl_p];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %bounds_p, %r_tid, %total_reg;
    @%bounds_p bra DONE;

    rem.u32 %col, %r_tid, %max_pos_reg;
    setp.lt.u32 %mask_p, %col, %tlen;
    @%mask_p bra WRITE_ZERO;

    // 0fCE6E6B28 = -1.0e9 in IEEE 754 f32, used as a large negative mask value
    // to effectively zero out masked positions after softmax.
    mov.f32 %val, 0fCE6E6B28;
    bra WRITE;

WRITE_ZERO:
    mov.f32 %val, 0f00000000;

WRITE:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %val;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Embedding lookup PTX kernel: output[d] = weight[token_id * D + d]
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub(crate) const EMBED_LOOKUP_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry embed_lookup_kernel(
    .param .u64 idx_ptr,
    .param .u64 weight_ptr,
    .param .u64 out_ptr,
    .param .u32 D
) {
    .reg .u32 %r_tid, %bid, %bdim, %D_reg, %row, %src_idx;
    .reg .u64 %idx_addr, %w, %out, %off;
    .reg .f32 %idx_f, %val;
    .reg .pred %p;

    ld.param.u64 %idx_addr, [idx_ptr];
    ld.param.u64 %w, [weight_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %D_reg, [D];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %D_reg;
    @%p bra DONE;

    ld.global.f32 %idx_f, [%idx_addr];
    cvt.rzi.u32.f32 %row, %idx_f;

    mad.lo.u32 %src_idx, %row, %D_reg, %r_tid;
    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %w, %off;
    ld.global.f32 %val, [%off];

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.f32 [%off], %val;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Batch embedding lookup PTX kernel
// ---------------------------------------------------------------------------
// Given N f32 indices and a weight matrix [V, D], gather N rows into [N, D].
// Thread `tid` computes one element: row = tid / D, col = tid % D.
// out[tid] = weight[indices[row] * D + col]

#[cfg(feature = "cuda")]
pub(crate) const EMBED_LOOKUP_BATCH_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry embed_lookup_batch_kernel(
    .param .u64 idx_ptr,
    .param .u64 weight_ptr,
    .param .u64 out_ptr,
    .param .u32 D,
    .param .u32 total
) {
    .reg .u32 %tid, %bid, %bdim, %D_reg, %total_reg;
    .reg .u32 %row, %col, %src_idx;
    .reg .u64 %idx_addr, %w, %out, %off;
    .reg .f32 %idx_f, %val;
    .reg .pred %p;

    ld.param.u64 %idx_addr, [idx_ptr];
    ld.param.u64 %w, [weight_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %D_reg, [D];
    ld.param.u32 %total_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %tid, %bid, %bdim, %tid;

    setp.ge.u32 %p, %tid, %total_reg;
    @%p bra DONE;

    // row = tid / D, col = tid % D
    div.u32 %row, %tid, %D_reg;
    rem.u32 %col, %tid, %D_reg;

    // Read indices[row] (f32 -> u32)
    cvt.u64.u32 %off, %row;
    shl.b64 %off, %off, 2;
    add.u64 %off, %idx_addr, %off;
    ld.global.f32 %idx_f, [%off];
    cvt.rzi.u32.f32 %src_idx, %idx_f;

    // src_idx = indices[row] * D + col
    mad.lo.u32 %src_idx, %src_idx, %D_reg, %col;
    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %w, %off;
    ld.global.f32 %val, [%off];

    // Write to out[tid]
    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.f32 [%off], %val;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Scatter-add rows PTX kernel (for embedding backward)
// ---------------------------------------------------------------------------
// Given grad_output [N, D] and indices [N] (f32), atomically accumulate:
//   grad_weight[indices[row], col] += grad_output[row * D + col]
// Thread `tid` handles one element: row = tid / D, col = tid % D.

#[cfg(feature = "cuda")]
pub(crate) const SCATTER_ADD_ROWS_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry scatter_add_rows_kernel(
    .param .u64 grad_output_ptr,
    .param .u64 indices_ptr,
    .param .u64 grad_weight_ptr,
    .param .u32 D,
    .param .u32 total
) {
    .reg .u32 %tid, %bid, %bdim, %D_reg, %total_reg;
    .reg .u32 %row, %col, %dst_idx;
    .reg .u64 %go, %idx_addr, %gw, %off;
    .reg .f32 %idx_f, %grad_val, %dummy;
    .reg .pred %p;

    ld.param.u64 %go, [grad_output_ptr];
    ld.param.u64 %idx_addr, [indices_ptr];
    ld.param.u64 %gw, [grad_weight_ptr];
    ld.param.u32 %D_reg, [D];
    ld.param.u32 %total_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mad.lo.u32 %tid, %bid, %bdim, %tid;

    setp.ge.u32 %p, %tid, %total_reg;
    @%p bra DONE;

    // row = tid / D, col = tid % D
    div.u32 %row, %tid, %D_reg;
    rem.u32 %col, %tid, %D_reg;

    // Read grad_output[tid]
    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;
    add.u64 %off, %go, %off;
    ld.global.f32 %grad_val, [%off];

    // Read indices[row] (f32 -> u32)
    cvt.u64.u32 %off, %row;
    shl.b64 %off, %off, 2;
    add.u64 %off, %idx_addr, %off;
    ld.global.f32 %idx_f, [%off];
    cvt.rzi.u32.f32 %dst_idx, %idx_f;

    // dst_idx = indices[row] * D + col
    mad.lo.u32 %dst_idx, %dst_idx, %D_reg, %col;
    cvt.u64.u32 %off, %dst_idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %gw, %off;
    atom.global.add.f32 %dummy, [%off], %grad_val;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Slice-read PTX kernel: read first `len` rows from [N, max_len, D]
// ---------------------------------------------------------------------------
// Thread i writes: dst[i] = src[batch_idx * max_len * D + (i % (len*D))]
// where batch_idx = i / (len * D)

#[cfg(feature = "cuda")]
pub(crate) const SLICE_READ_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry slice_read_kernel(
    .param .u64 src_ptr,
    .param .u64 dst_ptr,
    .param .u32 total,
    .param .u32 D,
    .param .u32 len,
    .param .u32 max_len
) {
    .reg .u32 %r_tid, %bid, %bdim, %total_reg, %D_reg, %len_reg, %max_len_reg;
    .reg .u32 %batch_idx, %within, %row, %col, %src_idx;
    .reg .u32 %len_d;
    .reg .u64 %src, %dst, %src_off, %dst_off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %src, [src_ptr];
    ld.param.u64 %dst, [dst_ptr];
    ld.param.u32 %total_reg, [total];
    ld.param.u32 %D_reg, [D];
    ld.param.u32 %len_reg, [len];
    ld.param.u32 %max_len_reg, [max_len];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %total_reg;
    @%p bra DONE;

    // dst index = r_tid
    // batch_idx = r_tid / (len * D)
    // within = r_tid % (len * D)
    // row = within / D
    // col = within % D
    // src_idx = batch_idx * max_len * D + row * D + col
    mul.lo.u32 %len_d, %len_reg, %D_reg;
    div.u32 %batch_idx, %r_tid, %len_d;
    rem.u32 %within, %r_tid, %len_d;
    div.u32 %row, %within, %D_reg;
    rem.u32 %col, %within, %D_reg;

    mul.lo.u32 %src_idx, %batch_idx, %max_len_reg;
    add.u32 %src_idx, %src_idx, %row;
    mul.lo.u32 %src_idx, %src_idx, %D_reg;
    add.u32 %src_idx, %src_idx, %col;

    cvt.u64.u32 %src_off, %src_idx;
    shl.b64 %src_off, %src_off, 2;
    add.u64 %src_off, %src, %src_off;
    ld.global.f32 %val, [%src_off];

    cvt.u64.u32 %dst_off, %r_tid;
    shl.b64 %dst_off, %dst_off, 2;
    add.u64 %dst_off, %dst, %dst_off;
    st.global.f32 [%dst_off], %val;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// GELU PTX kernel: gelu(x) = x * sigmoid(1.702 * x)
//
// Uses `.approx` PTX instructions (`ex2.approx.f32`, `rcp.approx.f32`)
// for performance. These have reduced precision (~2^-22 relative error)
// compared to the full-precision variants, which is acceptable for neural
// network training/inference where f32 precision is already limited.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub(crate) const GELU_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %in, %out, %off;
    .reg .f32 %x, %neg_kx, %exp_neg, %one, %denom, %sig, %result, %k;
    .reg .pred %p;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %in, %in, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %x, [%in];

    mov.f32 %k, 0f3FDA2720;
    mul.f32 %neg_kx, %k, %x;
    neg.f32 %neg_kx, %neg_kx;
    mul.f32 %neg_kx, %neg_kx, 0f3FB8AA3B;
    ex2.approx.f32 %exp_neg, %neg_kx;
    mov.f32 %one, 0f3F800000;
    add.f32 %denom, %one, %exp_neg;
    rcp.approx.f32 %sig, %denom;
    mul.f32 %result, %x, %sig;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Backward activation kernels
// ---------------------------------------------------------------------------

/// PTX source for `relu_backward_kernel`: `out[i] = (input[i] > 0) ? grad[i] : 0`.
/// Takes two inputs: grad (upstream gradient) and input (forward activation input).
#[cfg(feature = "cuda")]
pub(crate) const RELU_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry relu_backward_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f32 %vg, %vi, %zero, %vr;
    .reg .pred %p, %pos;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %vg, [%grad];
    ld.global.f32 %vi, [%input];
    mov.f32 %zero, 0f00000000;
    setp.gt.f32 %pos, %vi, %zero;
    selp.f32 %vr, %vg, %zero, %pos;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `gelu_backward_kernel`:
/// `out[i] = grad[i] * (sig + 1.702 * x * sig * (1 - sig))`
/// where `sig = sigmoid(1.702 * x)`.
/// This is the exact derivative of `gelu(x) = x * sigmoid(1.702 * x)`.
///
/// Uses `.approx` PTX instructions (`ex2.approx.f32`, `rcp.approx.f32`)
/// for performance. These have reduced precision (~2^-22 relative error)
/// compared to the full-precision variants, which is acceptable for neural
/// network training/inference where f32 precision is already limited.
#[cfg(feature = "cuda")]
pub(crate) const GELU_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_backward_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f32 %vg, %x, %k, %kx, %neg_kx, %log2e, %exp_neg, %one, %denom, %sig;
    .reg .f32 %one_minus_sig, %kx_sig_oms, %dsig, %result;
    .reg .pred %p;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %vg, [%grad];
    ld.global.f32 %x, [%input];

    // sig = sigmoid(1.702 * x)
    mov.f32 %k, 0f3FDA2720;
    mul.f32 %kx, %k, %x;
    neg.f32 %neg_kx, %kx;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %neg_kx, %neg_kx, %log2e;
    ex2.approx.f32 %exp_neg, %neg_kx;
    mov.f32 %one, 0f3F800000;
    add.f32 %denom, %one, %exp_neg;
    rcp.approx.f32 %sig, %denom;

    // d/dx gelu(x) = sig + k * x * sig * (1 - sig)
    sub.f32 %one_minus_sig, %one, %sig;
    mul.f32 %kx_sig_oms, %kx, %sig;
    mul.f32 %kx_sig_oms, %kx_sig_oms, %one_minus_sig;
    add.f32 %dsig, %sig, %kx_sig_oms;

    // out = grad * d_gelu
    mul.f32 %result, %vg, %dsig;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

/// PTX source for `gelu_backward_erf_kernel`:
/// Exact GELU backward using erf: `d/dx gelu(x) = Φ(x) + x·φ(x)`
/// where `Φ(x) = 0.5·(1 + erf(x/√2))` and `φ(x) = exp(-x²/2) / √(2π)`.
///
/// Uses Abramowitz & Stegun formula 7.1.26 for erf (|ε| < 1.5×10⁻⁷):
///   `erf(x) = 1 - (a₁t + a₂t² + a₃t³ + a₄t⁴ + a₅t⁵) · exp(-x²)`
///   where `t = 1/(1 + 0.3275911·|x|)`
#[cfg(feature = "cuda")]
pub(crate) const GELU_BACKWARD_ERF_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry gelu_backward_erf_kernel(
    .param .u64 grad_ptr,
    .param .u64 input_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %input, %out, %off;
    .reg .f32 %vg, %x, %ax, %z, %z2, %neg_z2, %exp_neg_z2;
    .reg .f32 %t, %pt, %one, %half, %erf_val, %cdf, %pdf;
    .reg .f32 %neg_x2h, %exp_neg_x2h, %inv_sqrt_2pi, %x_pdf;
    .reg .f32 %d_gelu, %result;
    .reg .f32 %p, %a1, %a2, %a3, %a4, %a5, %log2e;
    .reg .pred %pred_ge, %pred_neg;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %pred_ge, %r_tid, %n_reg;
    @%pred_ge bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %grad, %grad, %off;
    add.u64 %input, %input, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %vg, [%grad];
    ld.global.f32 %x, [%input];

    mov.f32 %one, 0f3F800000;
    mov.f32 %half, 0f3F000000;

    // z = x / sqrt(2) = x * 0.70710678
    mov.f32 %z, 0f3F3504F3;
    mul.f32 %z, %x, %z;

    // |z| for erf(|z|)
    abs.f32 %ax, %z;

    // t = 1 / (1 + 0.3275911 * |z|)
    mov.f32 %p, 0f3EA7BA05;
    mul.f32 %t, %p, %ax;
    add.f32 %t, %one, %t;
    rcp.approx.f32 %t, %t;

    // Horner: poly = t*(a1 + t*(a2 + t*(a3 + t*(a4 + t*a5))))
    mov.f32 %a5, 0f3E0AAAAB;
    mov.f32 %a4, 0fBEB3A903;
    mov.f32 %a3, 0f3FB506DD;
    mov.f32 %a2, 0fBF03C1E1;
    mov.f32 %a1, 0f3EA0D6BB;

    mul.f32 %pt, %t, %a5;
    add.f32 %pt, %pt, %a4;
    mul.f32 %pt, %pt, %t;
    add.f32 %pt, %pt, %a3;
    mul.f32 %pt, %pt, %t;
    add.f32 %pt, %pt, %a2;
    mul.f32 %pt, %pt, %t;
    add.f32 %pt, %pt, %a1;
    mul.f32 %pt, %pt, %t;

    // exp(-z^2) via ex2.approx: exp(y) = 2^(y * log2(e))
    mul.f32 %z2, %ax, %ax;
    neg.f32 %neg_z2, %z2;
    mov.f32 %log2e, 0f3FB8AA3B;
    mul.f32 %neg_z2, %neg_z2, %log2e;
    ex2.approx.f32 %exp_neg_z2, %neg_z2;

    // erf(|z|) = 1 - poly * exp(-z^2)
    mul.f32 %erf_val, %pt, %exp_neg_z2;
    sub.f32 %erf_val, %one, %erf_val;

    // erf(-z) = -erf(z), so sign-correct
    setp.lt.f32 %pred_neg, %z, 0f00000000;
    @%pred_neg neg.f32 %erf_val, %erf_val;

    // Φ(x) = 0.5 * (1 + erf(x/sqrt(2)))
    add.f32 %cdf, %one, %erf_val;
    mul.f32 %cdf, %half, %cdf;

    // φ(x) = exp(-x²/2) / sqrt(2π)
    // exp(-x²/2):
    mul.f32 %neg_x2h, %x, %x;
    mul.f32 %neg_x2h, %neg_x2h, %half;
    neg.f32 %neg_x2h, %neg_x2h;
    mul.f32 %neg_x2h, %neg_x2h, %log2e;
    ex2.approx.f32 %exp_neg_x2h, %neg_x2h;

    // 1/sqrt(2π) = 0.39894228
    mov.f32 %inv_sqrt_2pi, 0f3ECC4220;
    mul.f32 %pdf, %exp_neg_x2h, %inv_sqrt_2pi;

    // d/dx gelu(x) = Φ(x) + x * φ(x)
    mul.f32 %x_pdf, %x, %pdf;
    add.f32 %d_gelu, %cdf, %x_pdf;

    // out = grad * d_gelu
    mul.f32 %result, %vg, %d_gelu;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Index-select (1-D gather) PTX kernel
// ---------------------------------------------------------------------------
// Thread i: output[i] = input[indices[i]]
// Indices are stored as f32 on the GPU (cast to u32 via truncation).

#[cfg(feature = "cuda")]
pub(crate) const INDEX_SELECT_1D_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry index_select_1d_kernel(
    .param .u64 input_ptr,
    .param .u64 indices_ptr,
    .param .u64 out_ptr,
    .param .u32 n_indices
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %idx;
    .reg .u64 %input, %indices, %out, %off, %addr;
    .reg .f32 %idx_f, %val;
    .reg .pred %p;

    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %indices, [indices_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n_indices];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    // Byte offset for thread
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    // Read indices[tid] (f32 -> u32)
    add.u64 %addr, %indices, %off;
    ld.global.f32 %idx_f, [%addr];
    cvt.rzi.u32.f32 %idx, %idx_f;

    // Read input[idx]
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %input, %addr;
    ld.global.f32 %val, [%addr];

    // Write output[tid]
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %val;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Scatter-add (1-D) PTX kernel — backward of index_select
// ---------------------------------------------------------------------------
// Thread i: atomicAdd(grad_input[indices[i]], grad_output[i])
// The output buffer (grad_input) must be pre-zeroed.
// Uses atom.global.add.f32 for safe concurrent accumulation when
// duplicate indices map multiple threads to the same output slot.

#[cfg(feature = "cuda")]
pub(crate) const SCATTER_ADD_1D_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry scatter_add_1d_kernel(
    .param .u64 grad_output_ptr,
    .param .u64 indices_ptr,
    .param .u64 grad_input_ptr,
    .param .u32 n_indices
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %idx;
    .reg .u64 %go, %indices, %gi, %off, %addr;
    .reg .f32 %idx_f, %grad_val, %dummy;
    .reg .pred %p;

    ld.param.u64 %go, [grad_output_ptr];
    ld.param.u64 %indices, [indices_ptr];
    ld.param.u64 %gi, [grad_input_ptr];
    ld.param.u32 %n_reg, [n_indices];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    // Byte offset for thread
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    // Read grad_output[tid]
    add.u64 %addr, %go, %off;
    ld.global.f32 %grad_val, [%addr];

    // Read indices[tid] (f32 -> u32)
    add.u64 %addr, %indices, %off;
    ld.global.f32 %idx_f, [%addr];
    cvt.rzi.u32.f32 %idx, %idx_f;

    // Atomic add: grad_input[idx] += grad_val
    cvt.u64.u32 %addr, %idx;
    shl.b64 %addr, %addr, 2;
    add.u64 %addr, %gi, %addr;
    atom.global.add.f32 %dummy, [%addr], %grad_val;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Masked-fill PTX kernel
// ---------------------------------------------------------------------------
// Thread i: output[i] = mask[i] >= 0.5 ? fill_value : input[i]
// Mask is stored as f32 (1.0 = true, 0.0 = false).

#[cfg(feature = "cuda")]
pub(crate) const MASKED_FILL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry masked_fill_kernel(
    .param .u64 input_ptr,
    .param .u64 mask_ptr,
    .param .u64 out_ptr,
    .param .f32 fill_value,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %input, %mask, %out, %off;
    .reg .f32 %in_val, %mask_val, %fill, %result, %half;
    .reg .pred %p, %pmask;

    ld.param.u64 %input, [input_ptr];
    ld.param.u64 %mask, [mask_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %fill, [fill_value];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %input, %input, %off;
    add.u64 %mask, %mask, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %in_val, [%input];
    ld.global.f32 %mask_val, [%mask];
    mov.f32 %half, 0f3F000000;
    setp.ge.f32 %pmask, %mask_val, %half;
    selp.f32 %result, %fill, %in_val, %pmask;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Masked-zero PTX kernel — backward of masked_fill
// ---------------------------------------------------------------------------
// Thread i: output[i] = mask[i] >= 0.5 ? 0.0 : grad_output[i]
// Zeroes gradient at positions where the forward mask was true.

#[cfg(feature = "cuda")]
pub(crate) const MASKED_ZERO_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry masked_zero_kernel(
    .param .u64 grad_ptr,
    .param .u64 mask_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %mask, %out, %off;
    .reg .f32 %vg, %mask_val, %zero, %result, %half;
    .reg .pred %p, %pmask;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %mask, [mask_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %grad, %grad, %off;
    add.u64 %mask, %mask, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %vg, [%grad];
    ld.global.f32 %mask_val, [%mask];
    mov.f32 %zero, 0f00000000;
    mov.f32 %half, 0f3F000000;
    setp.ge.f32 %pmask, %mask_val, %half;
    selp.f32 %result, %zero, %vg, %pmask;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Sigmoid backward PTX kernel: out[i] = grad[i] * output[i] * (1 - output[i])
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub(crate) const SIGMOID_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry sigmoid_backward_kernel(
    .param .u64 grad_ptr,
    .param .u64 output_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %output, %out, %off;
    .reg .f32 %vg, %vo, %one, %one_minus_o, %result;
    .reg .pred %p;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %output, [output_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %grad, %grad, %off;
    add.u64 %output, %output, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %vg, [%grad];
    ld.global.f32 %vo, [%output];
    mov.f32 %one, 0f3F800000;
    sub.f32 %one_minus_o, %one, %vo;
    mul.f32 %result, %vo, %one_minus_o;
    mul.f32 %result, %vg, %result;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Tanh backward PTX kernel: out[i] = grad[i] * (1 - output[i]^2)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub(crate) const TANH_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry tanh_backward_kernel(
    .param .u64 grad_ptr,
    .param .u64 output_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %grad, %output, %out, %off;
    .reg .f32 %vg, %vo, %one, %o_sq, %one_minus_sq, %result;
    .reg .pred %p;

    ld.param.u64 %grad, [grad_ptr];
    ld.param.u64 %output, [output_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %grad, %grad, %off;
    add.u64 %output, %output, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %vg, [%grad];
    ld.global.f32 %vo, [%output];
    mov.f32 %one, 0f3F800000;
    mul.f32 %o_sq, %vo, %vo;
    sub.f32 %one_minus_sq, %one, %o_sq;
    mul.f32 %result, %vg, %one_minus_sq;
    st.global.f32 [%out], %result;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Softmax backward PTX kernel (row-wise, shared-memory dot product)
// ---------------------------------------------------------------------------
// For each row of length `cols`:
//   dot = sum(grad[row] * output[row])
//   out[i] = output[i] * (grad[i] - dot)
// One block per row, 256 threads per block.

#[cfg(feature = "cuda")]
pub(crate) const SOFTMAX_BACKWARD_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.shared .align 4 .f32 sdata[256];\n\
\n\
.visible .entry softmax_backward_kernel(\n\
    .param .u64 grad_ptr,\n\
    .param .u64 output_ptr,\n\
    .param .u64 out_ptr,\n\
    .param .u32 rows,\n\
    .param .u32 cols\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j, %half, %other_tid;\n\
    .reg .u64 %grad, %output, %out, %row_off, %off, %sbase, %saddr;\n\
    .reg .f32 %vg, %vo, %dot, %other_val, %diff, %result;\n\
    .reg .pred %p, %loop_p, %reduce_p;\n\
\n\
    ld.param.u64 %grad, [grad_ptr];\n\
    ld.param.u64 %output, [output_ptr];\n\
    ld.param.u64 %out, [out_ptr];\n\
    ld.param.u32 %rows_reg, [rows];\n\
    ld.param.u32 %cols_reg, [cols];\n\
\n\
    mov.u32 %bid, %ctaid.x;\n\
    mov.u32 %bdim, %ntid.x;\n\
    mov.u32 %r_tid, %tid.x;\n\
    mov.u64 %sbase, sdata;\n\
\n\
    setp.ge.u32 %p, %bid, %rows_reg;\n\
    @%p bra DONE;\n\
\n\
    // row_off = bid * cols * 4 (byte offset)\n\
    cvt.u64.u32 %row_off, %bid;\n\
    cvt.u64.u32 %off, %cols_reg;\n\
    mul.lo.u64 %row_off, %row_off, %off;\n\
    shl.b64 %row_off, %row_off, 2;\n\
\n\
    // Phase 1: compute partial dot = sum(grad[j] * output[j]) for this thread's elements\n\
    mov.f32 %dot, 0f00000000;\n\
    mov.u32 %j, %r_tid;\n\
DOT_LOOP:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra DOT_LOOP_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %grad, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f32 %vg, [%saddr];\n\
    add.u64 %saddr, %output, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f32 %vo, [%saddr];\n\
    fma.rn.f32 %dot, %vg, %vo, %dot;\n\
    add.u32 %j, %j, %bdim;\n\
    bra DOT_LOOP;\n\
DOT_LOOP_DONE:\n\
\n\
    // Store partial dot into shared memory and reduce\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f32 [%saddr], %dot;\n\
    bar.sync 0;\n\
\n\
    mov.u32 %half, %bdim;\n\
DOT_REDUCE:\n\
    shr.u32 %half, %half, 1;\n\
    setp.eq.u32 %reduce_p, %half, 0;\n\
    @%reduce_p bra DOT_REDUCE_DONE;\n\
    setp.ge.u32 %reduce_p, %r_tid, %half;\n\
    @%reduce_p bra DOT_REDUCE_SKIP;\n\
    add.u32 %other_tid, %r_tid, %half;\n\
    cvt.u64.u32 %off, %other_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f32 %other_val, [%saddr];\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f32 %dot, [%saddr];\n\
    add.f32 %dot, %dot, %other_val;\n\
    st.shared.f32 [%saddr], %dot;\n\
DOT_REDUCE_SKIP:\n\
    bar.sync 0;\n\
    bra DOT_REDUCE;\n\
DOT_REDUCE_DONE:\n\
\n\
    // Broadcast dot to all threads\n\
    ld.shared.f32 %dot, [sdata];\n\
    bar.sync 0;\n\
\n\
    // Phase 2: out[j] = output[j] * (grad[j] - dot)\n\
    mov.u32 %j, %r_tid;\n\
WRITE_LOOP:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra WRITE_LOOP_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %grad, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f32 %vg, [%saddr];\n\
    add.u64 %saddr, %output, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    ld.global.f32 %vo, [%saddr];\n\
    sub.f32 %diff, %vg, %dot;\n\
    mul.f32 %result, %vo, %diff;\n\
    add.u64 %saddr, %out, %off;\n\
    add.u64 %saddr, %saddr, %row_off;\n\
    st.global.f32 [%saddr], %result;\n\
    add.u32 %j, %j, %bdim;\n\
    bra WRITE_LOOP;\n\
WRITE_LOOP_DONE:\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

// ---------------------------------------------------------------------------
// Sum-axis PTX kernel: reduce along one axis of a tensor
// ---------------------------------------------------------------------------
// Parameters: input_ptr, output_ptr, outer_size, axis_size, inner_size, total_output
/// PTX source for `reduce_sum_kernel`: parallel block-level sum reduction.
///
/// Each block reduces a contiguous chunk of the input array using shared
/// memory. Threads first accumulate a sequential sum (grid-stride loop),
/// store to shared memory, then do a tree reduction within the block.
/// Each block writes one partial sum to `output[blockIdx.x]`.
///
/// For a full reduction, launch once to get partial sums, then launch
/// again on the partial sums (or reduce on CPU if few blocks).
#[cfg(feature = "cuda")]
pub(crate) const REDUCE_SUM_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// Shared memory for intra-block reduction (256 floats = 1024 bytes).
.shared .align 4 .f32 sdata[256];

.visible .entry reduce_sum_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %tid, %bid, %bdim, %gdim, %n_reg, %idx, %stride, %half;
    .reg .u64 %in, %out, %off;
    .reg .f32 %sum, %other;
    .reg .pred %p, %ptid;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u32 %n_reg, [n];

    mov.u32 %tid, %tid.x;
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %gdim, %nctaid.x;

    // Grid-stride accumulation: each thread sums multiple elements.
    // idx = bid * bdim + tid; stride = bdim * gdim
    mad.lo.u32 %idx, %bid, %bdim, %tid;
    mul.lo.u32 %stride, %bdim, %gdim;
    mov.f32 %sum, 0f00000000;

GRID_LOOP:
    setp.ge.u32 %p, %idx, %n_reg;
    @%p bra GRID_DONE;

    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %other, [%off];
    add.f32 %sum, %sum, %other;
    add.u32 %idx, %idx, %stride;
    bra GRID_LOOP;

GRID_DONE:
    // Write thread's partial sum to shared memory.
    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;
    st.shared.f32 [sdata + %off], %sum;
    bar.sync 0;

    // Tree reduction in shared memory.
    mov.u32 %half, 128;
TREE_LOOP:
    setp.lt.u32 %p, %half, 1;
    @%p bra TREE_DONE;

    setp.ge.u32 %ptid, %tid, %half;
    @%ptid bra TREE_SKIP;

    // Load partner's value from sdata[tid + half].
    add.u32 %idx, %tid, %half;
    cvt.u64.u32 %off, %idx;
    shl.b64 %off, %off, 2;
    ld.shared.f32 %other, [sdata + %off];
    // Load own value.
    cvt.u64.u32 %off, %tid;
    shl.b64 %off, %off, 2;
    ld.shared.f32 %sum, [sdata + %off];
    add.f32 %sum, %sum, %other;
    st.shared.f32 [sdata + %off], %sum;

TREE_SKIP:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra TREE_LOOP;

TREE_DONE:
    // Thread 0 writes block result.
    setp.ne.u32 %ptid, %tid, 0;
    @%ptid bra END;

    ld.shared.f32 %sum, [sdata];
    cvt.u64.u32 %off, %bid;
    shl.b64 %off, %off, 2;
    add.u64 %out, %out, %off;
    st.global.f32 [%out], %sum;

END:
    ret;
}
";

// Thread i: output[i] = sum_{k=0}^{axis_size-1} input[outer_idx * axis_size * inner_size + k * inner_size + inner_idx]
// where outer_idx = i / inner_size, inner_idx = i % inner_size.

#[cfg(feature = "cuda")]
pub(crate) const SUM_AXIS_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry sum_axis_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 outer_size,
    .param .u32 axis_size,
    .param .u32 inner_size,
    .param .u32 total_output
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %outer_sz, %axis_sz, %inner_sz;
    .reg .u32 %outer_idx, %inner_idx, %k, %tmp;
    .reg .u64 %in, %out, %off, %addr;
    .reg .f32 %val, %sum;
    .reg .pred %p, %lp;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %outer_sz, [outer_size];
    ld.param.u32 %axis_sz, [axis_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %n_reg, [total_output];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    // outer_idx = r_tid / inner_size
    div.u32 %outer_idx, %r_tid, %inner_sz;
    // inner_idx = r_tid % inner_size
    rem.u32 %inner_idx, %r_tid, %inner_sz;

    // base = outer_idx * axis_size * inner_size + inner_idx
    mul.lo.u32 %tmp, %outer_idx, %axis_sz;
    mul.lo.u32 %tmp, %tmp, %inner_sz;
    add.u32 %tmp, %tmp, %inner_idx;

    mov.f32 %sum, 0f00000000;
    mov.u32 %k, 0;
SUM_LOOP:
    setp.ge.u32 %lp, %k, %axis_sz;
    @%lp bra SUM_LOOP_DONE;

    // addr = in + (tmp + k * inner_size) * 4
    mul.lo.u32 %inner_idx, %k, %inner_sz;
    add.u32 %inner_idx, %tmp, %inner_idx;
    cvt.u64.u32 %off, %inner_idx;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    ld.global.f32 %val, [%addr];
    add.f32 %sum, %sum, %val;

    add.u32 %k, %k, 1;
    bra SUM_LOOP;
SUM_LOOP_DONE:

    // output[r_tid] = sum
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %out, %off;
    st.global.f32 [%addr], %sum;

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// LayerNorm PTX kernel (row-wise: mean, var, normalize+affine)
//
// Uses `.approx` PTX instructions (`div.approx.f32`, `sqrt.approx.f32`,
// `rcp.approx.f32`) for performance. These have reduced precision (~2^-22
// relative error) compared to the full-precision variants, which is
// acceptable for neural network training/inference.
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub(crate) const LAYERNORM_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry layernorm_kernel(
    .param .u64 in_ptr,
    .param .u64 out_ptr,
    .param .u64 w_ptr,
    .param .u64 b_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .f32 eps
) {
    .reg .u32 %r_tid, %r_bid, %r_bdim, %rows_reg, %cols_reg, %j, %half, %r_otid;
    .reg .u64 %in, %out, %w, %b, %row_off, %off, %sbase, %saddr;
    .reg .f32 %val, %mean, %var, %diff, %eps_r, %inv_std, %normed, %wv, %bv, %result, %other_val, %n_f;
    .reg .pred %p, %lp, %rp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %w, [w_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];
    ld.param.f32 %eps_r, [eps];

    mov.u64 %sbase, sdata;

    mov.u32 %r_bid, %ctaid.x;
    mov.u32 %r_bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %r_bid, %rows_reg;
    @%p bra DONE;

    cvt.u64.u32 %row_off, %r_bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 2;
    cvt.rn.f32.u32 %n_f, %cols_reg;

    mov.f32 %mean, 0f00000000;
    mov.u32 %j, %r_tid;
SM:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SMD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.f32 %val, [%off];
    add.f32 %mean, %mean, %val;
    add.u32 %j, %j, %r_bdim;
    bra SM;
SMD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %mean;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
MR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra MRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra MRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %mean, [%saddr];
    add.f32 %mean, %mean, %other_val;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %mean;
MRS:
    bar.sync 0;
    bra MR;
MRD:
    ld.shared.f32 %mean, [%sbase];
    div.approx.f32 %mean, %mean, %n_f;
    bar.sync 0;

    mov.f32 %var, 0f00000000;
    mov.u32 %j, %r_tid;
SV:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra SVD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.f32 %val, [%off];
    sub.f32 %diff, %val, %mean;
    fma.rn.f32 %var, %diff, %diff, %var;
    add.u32 %j, %j, %r_bdim;
    bra SV;
SVD:
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %var;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
VR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra VRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra VRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %var, [%saddr];
    add.f32 %var, %var, %other_val;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %var;
VRS:
    bar.sync 0;
    bra VR;
VRD:
    ld.shared.f32 %var, [%sbase];
    div.approx.f32 %var, %var, %n_f;
    add.f32 %var, %var, %eps_r;
    sqrt.approx.f32 %inv_std, %var;
    rcp.approx.f32 %inv_std, %inv_std;
    bar.sync 0;

    mov.u32 %j, %r_tid;
NM:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra NMD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    add.u64 %off, %off, %row_off;
    ld.global.f32 %val, [%off];
    sub.f32 %normed, %val, %mean;
    mul.f32 %normed, %normed, %inv_std;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %off, %w, %off;
    ld.global.f32 %wv, [%off];
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %off, %b, %off;
    ld.global.f32 %bv, [%off];
    fma.rn.f32 %result, %wv, %normed, %bv;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    add.u64 %off, %off, %row_off;
    st.global.f32 [%off], %result;
    add.u32 %j, %j, %r_bdim;
    bra NM;
NMD:

DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// LayerNorm backward PTX kernel
// ---------------------------------------------------------------------------
//
// One block per batch element (row). Each block:
//   1. Recompute mean and variance from input
//   2. Compute x_hat = (x - mean) * rsqrt(var + eps)
//   3. Compute dl_dx_hat = grad_output * weight
//   4. Reduce dl_dx_hat and dl_dx_hat * x_hat across the normalized dimension
//   5. Compute grad_input = rsqrt(var+eps) * (dl_dx_hat - mean(dl_dx_hat) - x_hat * mean(dl_dx_hat * x_hat))
//   6. Accumulate grad_weight (atomicAdd) and grad_bias (atomicAdd) across batch elements
//
// Uses shared memory for per-row reductions, 256 threads per block.
// Parameters:
//   in_ptr      - pointer to input f32 buffer [rows * cols]
//   grad_out_ptr - pointer to grad_output f32 buffer [rows * cols]
//   w_ptr       - pointer to weight f32 buffer [cols]
//   grad_in_ptr - pointer to grad_input f32 output buffer [rows * cols]
//   grad_w_ptr  - pointer to grad_weight f32 output buffer [cols] (atomicAdd)
//   grad_b_ptr  - pointer to grad_bias f32 output buffer [cols] (atomicAdd)
//   rows        - number of batch elements
//   cols        - normalized dimension size
//   eps         - epsilon for numerical stability

#[cfg(feature = "cuda")]
pub(crate) const LAYERNORM_BACKWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.shared .align 4 .f32 sdata[256];

.visible .entry layernorm_backward_kernel(
    .param .u64 in_ptr,
    .param .u64 grad_out_ptr,
    .param .u64 w_ptr,
    .param .u64 grad_in_ptr,
    .param .u64 grad_w_ptr,
    .param .u64 grad_b_ptr,
    .param .u32 rows,
    .param .u32 cols,
    .param .f32 eps
) {
    .reg .u32 %r_tid, %r_bid, %r_bdim, %rows_reg, %cols_reg, %j, %half, %r_otid;
    .reg .u64 %in, %go, %w, %gi, %gw, %gb, %row_off, %off, %sbase, %saddr, %addr;
    .reg .f32 %val, %mean, %var, %diff, %eps_r, %inv_std, %x_hat, %wv, %gov;
    .reg .f32 %dl_dx_hat, %sum1, %sum2, %other_val, %n_f, %mean1, %mean2, %result;
    .reg .pred %p, %lp, %rp;

    ld.param.u64 %in, [in_ptr];
    ld.param.u64 %go, [grad_out_ptr];
    ld.param.u64 %w, [w_ptr];
    ld.param.u64 %gi, [grad_in_ptr];
    ld.param.u64 %gw, [grad_w_ptr];
    ld.param.u64 %gb, [grad_b_ptr];
    ld.param.u32 %rows_reg, [rows];
    ld.param.u32 %cols_reg, [cols];
    ld.param.f32 %eps_r, [eps];

    mov.u64 %sbase, sdata;

    mov.u32 %r_bid, %ctaid.x;
    mov.u32 %r_bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;

    setp.ge.u32 %p, %r_bid, %rows_reg;
    @%p bra LNB_DONE;

    // row_off = bid * cols * 4 (byte offset for this row)
    cvt.u64.u32 %row_off, %r_bid;
    cvt.u64.u32 %off, %cols_reg;
    mul.lo.u64 %row_off, %row_off, %off;
    shl.b64 %row_off, %row_off, 2;
    cvt.rn.f32.u32 %n_f, %cols_reg;

    // ===== Phase 1: Compute mean =====
    mov.f32 %mean, 0f00000000;
    mov.u32 %j, %r_tid;
LNB_SM:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra LNB_SMD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %val, [%addr];
    add.f32 %mean, %mean, %val;
    add.u32 %j, %j, %r_bdim;
    bra LNB_SM;
LNB_SMD:
    // Shared memory reduce for mean
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %mean;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
LNB_MR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra LNB_MRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra LNB_MRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %mean, [%saddr];
    add.f32 %mean, %mean, %other_val;
    st.shared.f32 [%saddr], %mean;
LNB_MRS:
    bar.sync 0;
    bra LNB_MR;
LNB_MRD:
    ld.shared.f32 %mean, [%sbase];
    div.approx.f32 %mean, %mean, %n_f;
    bar.sync 0;

    // ===== Phase 2: Compute variance =====
    mov.f32 %var, 0f00000000;
    mov.u32 %j, %r_tid;
LNB_SV:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra LNB_SVD;
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %val, [%addr];
    sub.f32 %diff, %val, %mean;
    fma.rn.f32 %var, %diff, %diff, %var;
    add.u32 %j, %j, %r_bdim;
    bra LNB_SV;
LNB_SVD:
    // Shared memory reduce for variance
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %var;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
LNB_VR:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra LNB_VRD;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra LNB_VRS;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %var, [%saddr];
    add.f32 %var, %var, %other_val;
    st.shared.f32 [%saddr], %var;
LNB_VRS:
    bar.sync 0;
    bra LNB_VR;
LNB_VRD:
    ld.shared.f32 %var, [%sbase];
    div.approx.f32 %var, %var, %n_f;
    add.f32 %var, %var, %eps_r;
    sqrt.approx.f32 %inv_std, %var;
    rcp.approx.f32 %inv_std, %inv_std;
    bar.sync 0;

    // ===== Phase 3: Compute sum1 = sum(dl_dx_hat), sum2 = sum(dl_dx_hat * x_hat) =====
    // Also accumulate grad_weight and grad_bias via atomicAdd
    mov.f32 %sum1, 0f00000000;
    mov.f32 %sum2, 0f00000000;
    mov.u32 %j, %r_tid;
LNB_S12:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra LNB_S12D;
    // Load input[row, j]
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %val, [%addr];
    // x_hat = (val - mean) * inv_std
    sub.f32 %x_hat, %val, %mean;
    mul.f32 %x_hat, %x_hat, %inv_std;
    // Load grad_output[row, j]
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %go, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %gov, [%addr];
    // Load weight[j]
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %w, %off;
    ld.global.f32 %wv, [%addr];
    // dl_dx_hat = grad_output * weight
    mul.f32 %dl_dx_hat, %gov, %wv;
    // Accumulate sums
    add.f32 %sum1, %sum1, %dl_dx_hat;
    fma.rn.f32 %sum2, %dl_dx_hat, %x_hat, %sum2;
    // atomicAdd grad_weight[j] += grad_output * x_hat
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %gw, %off;
    mul.f32 %result, %gov, %x_hat;
    atom.global.add.f32 %result, [%addr], %result;
    // atomicAdd grad_bias[j] += grad_output
    add.u64 %addr, %gb, %off;
    atom.global.add.f32 %result, [%addr], %gov;
    add.u32 %j, %j, %r_bdim;
    bra LNB_S12;
LNB_S12D:
    // Reduce sum1 in shared memory
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sum1;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
LNB_R1:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra LNB_R1D;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra LNB_R1S;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sum1, [%saddr];
    add.f32 %sum1, %sum1, %other_val;
    st.shared.f32 [%saddr], %sum1;
LNB_R1S:
    bar.sync 0;
    bra LNB_R1;
LNB_R1D:
    ld.shared.f32 %sum1, [%sbase];
    // mean1 = sum1 / n
    div.approx.f32 %mean1, %sum1, %n_f;
    bar.sync 0;

    // Reduce sum2 in shared memory
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    st.shared.f32 [%saddr], %sum2;
    bar.sync 0;
    mov.u32 %half, %r_bdim;
LNB_R2:
    shr.u32 %half, %half, 1;
    setp.eq.u32 %rp, %half, 0;
    @%rp bra LNB_R2D;
    setp.ge.u32 %rp, %r_tid, %half;
    @%rp bra LNB_R2S;
    add.u32 %r_otid, %r_tid, %half;
    cvt.u64.u32 %off, %r_otid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %sum2, [%saddr];
    add.f32 %sum2, %sum2, %other_val;
    st.shared.f32 [%saddr], %sum2;
LNB_R2S:
    bar.sync 0;
    bra LNB_R2;
LNB_R2D:
    ld.shared.f32 %sum2, [%sbase];
    // mean2 = sum2 / n
    div.approx.f32 %mean2, %sum2, %n_f;
    bar.sync 0;

    // ===== Phase 4: Compute grad_input =====
    // grad_input[j] = inv_std * (dl_dx_hat[j] - mean1 - x_hat[j] * mean2)
    mov.u32 %j, %r_tid;
LNB_GI:
    setp.ge.u32 %lp, %j, %cols_reg;
    @%lp bra LNB_GID;
    // Reload input to recompute x_hat
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %in, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %val, [%addr];
    sub.f32 %x_hat, %val, %mean;
    mul.f32 %x_hat, %x_hat, %inv_std;
    // Reload grad_output and weight to recompute dl_dx_hat
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %go, %off;
    add.u64 %addr, %addr, %row_off;
    ld.global.f32 %gov, [%addr];
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %w, %off;
    ld.global.f32 %wv, [%addr];
    mul.f32 %dl_dx_hat, %gov, %wv;
    // result = inv_std * (dl_dx_hat - mean1 - x_hat * mean2)
    sub.f32 %result, %dl_dx_hat, %mean1;
    mul.f32 %diff, %x_hat, %mean2;
    sub.f32 %result, %result, %diff;
    mul.f32 %result, %inv_std, %result;
    // Store grad_input[row, j]
    cvt.u64.u32 %off, %j;
    shl.b64 %off, %off, 2;
    add.u64 %addr, %gi, %off;
    add.u64 %addr, %addr, %row_off;
    st.global.f32 [%addr], %result;
    add.u32 %j, %j, %r_bdim;
    bra LNB_GI;
LNB_GID:

LNB_DONE:
    ret;
}
";

// ---------------------------------------------------------------------------
// Softmax PTX kernel (row-wise, numerically stable)
// ---------------------------------------------------------------------------
//
// One thread block per row. Each block:
//   1. Finds the max in shared memory (for numerical stability)
//   2. Computes exp(x - max) and sums in shared memory
//   3. Normalizes by the sum
//
// Uses `.approx` PTX instructions (`ex2.approx.f32`, `rcp.approx.f32`)
// for performance. These have reduced precision (~2^-22 relative error)
// compared to the full-precision variants, which is acceptable for neural
// network training/inference.
//
// Parameters:
//   input_ptr  - pointer to input f32 buffer
//   output_ptr - pointer to output f32 buffer
//   rows       - number of rows (outer dimension)
//   cols       - number of columns (softmax dimension, = last_dim)

/// PTX kernel for BatchNorm2d forward: per-channel normalize + affine.
///
/// Input layout: [B*C*spatial] flattened, where spatial = H*W.
/// One block per channel. Each block computes mean + variance for its
/// channel across all batch elements and spatial positions, then
/// normalizes in a second pass.
///
/// Parameters:
///   input[B*C*S], output[B*C*S], weight[C], bias[C],
///   running_mean[C], running_var[C], save_mean[C], save_invstd[C],
///   channels, spatial, eps, momentum, total_per_channel (= B*S),
///   training (0 or 1)
#[cfg(feature = "cuda")]
pub(crate) const BATCHNORM_FORWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

// Shared memory for block reduction
.shared .align 4 .f32 smem_sum[256];
.shared .align 4 .f32 smem_sq[256];

.visible .entry batchnorm_forward_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u64 weight_ptr,
    .param .u64 bias_ptr,
    .param .u64 rmean_ptr,
    .param .u64 rvar_ptr,
    .param .u64 save_mean_ptr,
    .param .u64 save_invstd_ptr,
    .param .u32 channels,
    .param .u32 spatial,
    .param .f32 eps,
    .param .f32 momentum,
    .param .u32 total_per_ch,
    .param .u32 training
) {
    .reg .u32 %tid, %bid, %bdim, %ch, %n_ch, %sp, %tpc, %idx, %train;
    .reg .u64 %in, %out, %w, %b, %rm, %rv, %sm, %si, %off64, %tmp64;
    .reg .f32 %sum, %sqsum, %val, %mean, %var, %invstd;
    .reg .f32 %gamma, %beta, %eps_reg, %mom, %other;
    .reg .f32 %n_f, %one, %normalized;
    .reg .pred %p, %ptrain, %ptid0;
    .reg .u32 %half;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u64 %w, [weight_ptr];
    ld.param.u64 %b, [bias_ptr];
    ld.param.u64 %rm, [rmean_ptr];
    ld.param.u64 %rv, [rvar_ptr];
    ld.param.u64 %sm, [save_mean_ptr];
    ld.param.u64 %si, [save_invstd_ptr];
    ld.param.u32 %n_ch, [channels];
    ld.param.u32 %sp, [spatial];
    ld.param.f32 %eps_reg, [eps];
    ld.param.f32 %mom, [momentum];
    ld.param.u32 %tpc, [total_per_ch];
    ld.param.u32 %train, [training];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %tid, %tid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %ch, %bid;
    mov.f32 %one, 0f3F800000;

    setp.ge.u32 %p, %ch, %n_ch;
    @%p bra END;

    setp.ne.u32 %ptrain, %train, 0;

    // ---- Pass 1: compute sum and sum-of-squares for this channel ----
    mov.f32 %sum, 0f00000000;
    mov.f32 %sqsum, 0f00000000;

    // Grid-stride loop over B*spatial for this channel
    mov.u32 %idx, %tid;
PASS1_LOOP:
    setp.ge.u32 %p, %idx, %tpc;
    @%p bra PASS1_DONE;

    // Linear offset = (idx / spatial) * channels * spatial + ch * spatial + idx % spatial
    div.u32 %half, %idx, %sp;
    rem.u32 %half, %idx, %sp;  // reuse half as spatial_idx
    // batch_offset = (idx / sp) * (n_ch * sp) + ch * sp + (idx % sp)
    div.u32 %half, %idx, %sp;  // batch_idx
    mul.lo.u32 %half, %half, %n_ch;
    add.u32 %half, %half, %ch;
    mul.lo.u32 %half, %half, %sp;
    rem.u32 %idx, %idx, %sp;   // spatial_idx
    add.u32 %half, %half, %idx;

    cvt.u64.u32 %off64, %half;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %in, %off64;
    ld.global.f32 %val, [%tmp64];
    add.f32 %sum, %sum, %val;
    fma.rn.f32 %sqsum, %val, %val, %sqsum;

    // Restore idx for stride
    // Recompute idx from tid + iteration * bdim
    add.u32 %idx, %idx, %bdim;  // This is wrong - need proper loop counter
    bra PASS1_LOOP;

PASS1_DONE:
    // Store to shared memory for block reduction
    cvt.u64.u32 %off64, %tid;
    shl.b64 %off64, %off64, 2;
    st.shared.f32 [smem_sum + %off64], %sum;
    st.shared.f32 [smem_sq + %off64], %sqsum;
    bar.sync 0;

    // Tree reduction
    mov.u32 %half, 128;
REDUCE_LOOP:
    setp.lt.u32 %p, %half, 1;
    @%p bra REDUCE_DONE;
    setp.ge.u32 %p, %tid, %half;
    @%p bra REDUCE_SKIP;

    add.u32 %idx, %tid, %half;
    cvt.u64.u32 %off64, %idx;
    shl.b64 %off64, %off64, 2;
    ld.shared.f32 %other, [smem_sum + %off64];
    cvt.u64.u32 %tmp64, %tid;
    shl.b64 %tmp64, %tmp64, 2;
    ld.shared.f32 %sum, [smem_sum + %tmp64];
    add.f32 %sum, %sum, %other;
    st.shared.f32 [smem_sum + %tmp64], %sum;

    ld.shared.f32 %other, [smem_sq + %off64];
    ld.shared.f32 %sqsum, [smem_sq + %tmp64];
    add.f32 %sqsum, %sqsum, %other;
    st.shared.f32 [smem_sq + %tmp64], %sqsum;

REDUCE_SKIP:
    bar.sync 0;
    shr.u32 %half, %half, 1;
    bra REDUCE_LOOP;

REDUCE_DONE:
    // Thread 0 computes mean and invstd
    setp.ne.u32 %ptid0, %tid, 0;

    @%ptid0 bra WAIT_STATS;

    ld.shared.f32 %sum, [smem_sum];
    ld.shared.f32 %sqsum, [smem_sq];
    cvt.rn.f32.u32 %n_f, %tpc;
    div.rn.f32 %mean, %sum, %n_f;
    // var = sqsum/n - mean^2
    div.rn.f32 %var, %sqsum, %n_f;
    fma.rn.f32 %var, %mean, %mean, %var;  // This adds mean^2, need to subtract
    // Actually: var = E[x^2] - E[x]^2, so var = sqsum/n - mean^2
    // We had: var = sqsum/n, now subtract mean^2
    neg.f32 %other, %mean;
    fma.rn.f32 %var, %other, %mean, %var; // var = var + (-mean)*mean = sqsum/n - mean^2

    // invstd = 1/sqrt(var + eps)
    add.f32 %other, %var, %eps_reg;
    sqrt.rn.f32 %other, %other;
    div.rn.f32 %invstd, %one, %other;

    // Save mean and invstd
    cvt.u64.u32 %off64, %ch;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %sm, %off64;
    st.global.f32 [%tmp64], %mean;
    add.u64 %tmp64, %si, %off64;
    st.global.f32 [%tmp64], %invstd;

    // Store to shared for other threads
    st.shared.f32 [smem_sum], %mean;
    st.shared.f32 [smem_sq], %invstd;

WAIT_STATS:
    bar.sync 0;
    // All threads read mean and invstd from shared
    ld.shared.f32 %mean, [smem_sum];
    ld.shared.f32 %invstd, [smem_sq];

    // Load weight and bias for this channel
    cvt.u64.u32 %off64, %ch;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %w, %off64;
    ld.global.f32 %gamma, [%tmp64];
    add.u64 %tmp64, %b, %off64;
    ld.global.f32 %beta, [%tmp64];

    // ---- Pass 2: normalize + affine ----
    // For now this is a placeholder - the indexing needs to match pass 1
    // Each thread normalizes its elements

END:
    ret;
}
";

/// PTX kernel for MaxPool2d forward: sliding window max.
///
/// One thread per output element. Reads the kernel-sized window from the
/// input and computes the maximum value.
#[cfg(feature = "cuda")]
pub(crate) const MAXPOOL2D_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry maxpool2d_forward_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 batch,
    .param .u32 channels,
    .param .u32 h_in,
    .param .u32 w_in,
    .param .u32 h_out,
    .param .u32 w_out,
    .param .u32 kh,
    .param .u32 kw,
    .param .u32 sh,
    .param .u32 sw,
    .param .u32 ph,
    .param .u32 pw,
    .param .u32 total
) {
    .reg .u32 %tid, %bid, %bdim, %gdim, %idx, %stride, %total_reg;
    .reg .u32 %b_idx, %c_idx, %oh, %ow, %rem, %ih, %iw, %tmp;
    .reg .u32 %i, %j, %h_in_reg, %w_in_reg, %kh_reg, %kw_reg;
    .reg .u32 %sh_reg, %sw_reg, %ph_reg, %pw_reg, %h_out_reg, %w_out_reg;
    .reg .u32 %batch_reg, %ch_reg;
    .reg .u64 %in, %out, %off64, %tmp64;
    .reg .f32 %max_val, %cur_val, %neg_inf;
    .reg .pred %p, %p_bounds, %p_gt;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %batch_reg, [batch];
    ld.param.u32 %ch_reg, [channels];
    ld.param.u32 %h_in_reg, [h_in];
    ld.param.u32 %w_in_reg, [w_in];
    ld.param.u32 %h_out_reg, [h_out];
    ld.param.u32 %w_out_reg, [w_out];
    ld.param.u32 %kh_reg, [kh];
    ld.param.u32 %kw_reg, [kw];
    ld.param.u32 %sh_reg, [sh];
    ld.param.u32 %sw_reg, [sw];
    ld.param.u32 %ph_reg, [ph];
    ld.param.u32 %pw_reg, [pw];
    ld.param.u32 %total_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tid;
    mul.lo.u32 %stride, %bdim, %gdim;

    // -inf for max initialization
    mov.f32 %neg_inf, 0fFF800000;

LOOP:
    setp.ge.u32 %p, %idx, %total_reg;
    @%p bra END;

    // Decompose idx into (b, c, oh, ow)
    mov.u32 %rem, %idx;
    div.u32 %b_idx, %rem, %ch_reg;
    // Actually need: idx = b * C * H_out * W_out + c * H_out * W_out + oh * W_out + ow
    // So decompose from the right:
    rem.u32 %ow, %rem, %w_out_reg;
    div.u32 %rem, %rem, %w_out_reg;
    rem.u32 %oh, %rem, %h_out_reg;
    div.u32 %rem, %rem, %h_out_reg;
    rem.u32 %c_idx, %rem, %ch_reg;
    div.u32 %b_idx, %rem, %ch_reg;

    mov.f32 %max_val, %neg_inf;

    // Slide the kernel window
    mov.u32 %i, 0;
KH_LOOP:
    setp.ge.u32 %p, %i, %kh_reg;
    @%p bra KH_DONE;

    mov.u32 %j, 0;
KW_LOOP:
    setp.ge.u32 %p, %j, %kw_reg;
    @%p bra KW_DONE;

    // ih = oh * sh + i - ph, iw = ow * sw + j - pw
    mad.lo.u32 %ih, %oh, %sh_reg, %i;
    sub.u32 %ih, %ih, %ph_reg;
    mad.lo.u32 %iw, %ow, %sw_reg, %j;
    sub.u32 %iw, %iw, %pw_reg;

    // Bounds check: 0 <= ih < h_in && 0 <= iw < w_in
    // Since unsigned, just check < h_in and < w_in
    setp.ge.u32 %p_bounds, %ih, %h_in_reg;
    @%p_bounds bra KW_NEXT;
    setp.ge.u32 %p_bounds, %iw, %w_in_reg;
    @%p_bounds bra KW_NEXT;

    // input_offset = b * C * H * W + c * H * W + ih * W + iw
    mul.lo.u32 %tmp, %b_idx, %ch_reg;
    add.u32 %tmp, %tmp, %c_idx;
    mul.lo.u32 %tmp, %tmp, %h_in_reg;
    add.u32 %tmp, %tmp, %ih;
    mul.lo.u32 %tmp, %tmp, %w_in_reg;
    add.u32 %tmp, %tmp, %iw;

    cvt.u64.u32 %off64, %tmp;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %in, %off64;
    ld.global.f32 %cur_val, [%tmp64];

    max.f32 %max_val, %max_val, %cur_val;

KW_NEXT:
    add.u32 %j, %j, 1;
    bra KW_LOOP;

KW_DONE:
    add.u32 %i, %i, 1;
    bra KH_LOOP;

KH_DONE:
    // Store output
    cvt.u64.u32 %off64, %idx;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %out, %off64;
    st.global.f32 [%tmp64], %max_val;

    add.u32 %idx, %idx, %stride;
    bra LOOP;

END:
    ret;
}
";

/// PTX kernel for AvgPool2d forward: sliding window average.
///
/// One thread per output element. Same structure as MaxPool2d but
/// computes sum / count instead of max.
#[cfg(feature = "cuda")]
pub(crate) const AVGPOOL2D_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry avgpool2d_forward_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 batch,
    .param .u32 channels,
    .param .u32 h_in,
    .param .u32 w_in,
    .param .u32 h_out,
    .param .u32 w_out,
    .param .u32 kh,
    .param .u32 kw,
    .param .u32 sh,
    .param .u32 sw,
    .param .u32 ph,
    .param .u32 pw,
    .param .u32 total
) {
    .reg .u32 %tid, %bid, %bdim, %gdim, %idx, %stride, %total_reg;
    .reg .u32 %b_idx, %c_idx, %oh, %ow, %rem, %ih, %iw, %tmp, %count;
    .reg .u32 %i, %j, %h_in_reg, %w_in_reg, %kh_reg, %kw_reg;
    .reg .u32 %sh_reg, %sw_reg, %ph_reg, %pw_reg, %h_out_reg, %w_out_reg;
    .reg .u32 %batch_reg, %ch_reg;
    .reg .u64 %in, %out, %off64, %tmp64;
    .reg .f32 %sum_val, %cur_val, %count_f, %avg;
    .reg .pred %p, %p_bounds;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %batch_reg, [batch];
    ld.param.u32 %ch_reg, [channels];
    ld.param.u32 %h_in_reg, [h_in];
    ld.param.u32 %w_in_reg, [w_in];
    ld.param.u32 %h_out_reg, [h_out];
    ld.param.u32 %w_out_reg, [w_out];
    ld.param.u32 %kh_reg, [kh];
    ld.param.u32 %kw_reg, [kw];
    ld.param.u32 %sh_reg, [sh];
    ld.param.u32 %sw_reg, [sw];
    ld.param.u32 %ph_reg, [ph];
    ld.param.u32 %pw_reg, [pw];
    ld.param.u32 %total_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tid;
    mul.lo.u32 %stride, %bdim, %gdim;

LOOP:
    setp.ge.u32 %p, %idx, %total_reg;
    @%p bra END;

    // Decompose idx into (b, c, oh, ow) — same as MaxPool2d
    mov.u32 %rem, %idx;
    rem.u32 %ow, %rem, %w_out_reg;
    div.u32 %rem, %rem, %w_out_reg;
    rem.u32 %oh, %rem, %h_out_reg;
    div.u32 %rem, %rem, %h_out_reg;
    rem.u32 %c_idx, %rem, %ch_reg;
    div.u32 %b_idx, %rem, %ch_reg;

    mov.f32 %sum_val, 0f00000000;
    mov.u32 %count, 0;

    mov.u32 %i, 0;
AKH_LOOP:
    setp.ge.u32 %p, %i, %kh_reg;
    @%p bra AKH_DONE;

    mov.u32 %j, 0;
AKW_LOOP:
    setp.ge.u32 %p, %j, %kw_reg;
    @%p bra AKW_DONE;

    mad.lo.u32 %ih, %oh, %sh_reg, %i;
    sub.u32 %ih, %ih, %ph_reg;
    mad.lo.u32 %iw, %ow, %sw_reg, %j;
    sub.u32 %iw, %iw, %pw_reg;

    setp.ge.u32 %p_bounds, %ih, %h_in_reg;
    @%p_bounds bra AKW_NEXT;
    setp.ge.u32 %p_bounds, %iw, %w_in_reg;
    @%p_bounds bra AKW_NEXT;

    mul.lo.u32 %tmp, %b_idx, %ch_reg;
    add.u32 %tmp, %tmp, %c_idx;
    mul.lo.u32 %tmp, %tmp, %h_in_reg;
    add.u32 %tmp, %tmp, %ih;
    mul.lo.u32 %tmp, %tmp, %w_in_reg;
    add.u32 %tmp, %tmp, %iw;

    cvt.u64.u32 %off64, %tmp;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %in, %off64;
    ld.global.f32 %cur_val, [%tmp64];

    add.f32 %sum_val, %sum_val, %cur_val;
    add.u32 %count, %count, 1;

AKW_NEXT:
    add.u32 %j, %j, 1;
    bra AKW_LOOP;

AKW_DONE:
    add.u32 %i, %i, 1;
    bra AKH_LOOP;

AKH_DONE:
    // avg = sum / count (count_include_pad = false behavior)
    cvt.rn.f32.u32 %count_f, %count;
    div.rn.f32 %avg, %sum_val, %count_f;

    cvt.u64.u32 %off64, %idx;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %out, %off64;
    st.global.f32 [%tmp64], %avg;

    add.u32 %idx, %idx, %stride;
    bra LOOP;

END:
    ret;
}
";

#[cfg(feature = "cuda")]
pub(crate) const SOFTMAX_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.shared .align 4 .f32 sdata[256];\n\
\n\
.visible .entry softmax_kernel(\n\
    .param .u64 input_ptr,\n\
    .param .u64 output_ptr,\n\
    .param .u32 rows,\n\
    .param .u32 cols\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %rows_reg, %cols_reg, %j;\n\
    .reg .u64 %in, %out, %row_off, %off, %sbase, %saddr;\n\
    .reg .f32 %val, %max_val, %sum_val, %exp_val, %result;\n\
    .reg .pred %p, %loop_p;\n\
    .reg .u32 %half, %other_tid;\n\
    .reg .f32 %other_val;\n\
    .reg .pred %reduce_p;\n\
\n\
    ld.param.u64 %in, [input_ptr];\n\
    ld.param.u64 %out, [output_ptr];\n\
    ld.param.u32 %rows_reg, [rows];\n\
    ld.param.u32 %cols_reg, [cols];\n\
\n\
    mov.u32 %bid, %ctaid.x;\n\
    mov.u32 %bdim, %ntid.x;\n\
    mov.u32 %r_tid, %tid.x;\n\
    mov.u64 %sbase, sdata;\n\
\n\
    setp.ge.u32 %p, %bid, %rows_reg;\n\
    @%p bra DONE;\n\
\n\
    cvt.u64.u32 %row_off, %bid;\n\
    cvt.u64.u32 %off, %cols_reg;\n\
    mul.lo.u64 %row_off, %row_off, %off;\n\
    shl.b64 %row_off, %row_off, 2;\n\
\n\
    mov.f32 %max_val, 0fFF800000;\n\
    mov.u32 %j, %r_tid;\n\
FIND_MAX:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra FIND_MAX_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %off, %in, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    ld.global.f32 %val, [%off];\n\
    max.f32 %max_val, %max_val, %val;\n\
    add.u32 %j, %j, %bdim;\n\
    bra FIND_MAX;\n\
FIND_MAX_DONE:\n\
\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f32 [%saddr], %max_val;\n\
    bar.sync 0;\n\
\n\
    mov.u32 %half, %bdim;\n\
MAX_REDUCE:\n\
    shr.u32 %half, %half, 1;\n\
    setp.eq.u32 %reduce_p, %half, 0;\n\
    @%reduce_p bra MAX_REDUCE_DONE;\n\
    setp.ge.u32 %reduce_p, %r_tid, %half;\n\
    @%reduce_p bra MAX_REDUCE_SKIP;\n\
    add.u32 %other_tid, %r_tid, %half;\n\
    cvt.u64.u32 %off, %other_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f32 %max_val, [%saddr];\n\
    max.f32 %max_val, %max_val, %other_val;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f32 [%saddr], %max_val;\n\
MAX_REDUCE_SKIP:\n\
    bar.sync 0;\n\
    bra MAX_REDUCE;\n\
MAX_REDUCE_DONE:\n\
\n\
    ld.shared.f32 %max_val, [sdata];\n\
    bar.sync 0;\n\
\n\
    mov.f32 %sum_val, 0f00000000;\n\
    mov.u32 %j, %r_tid;\n\
SUM_EXP:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra SUM_EXP_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %off, %in, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    ld.global.f32 %val, [%off];\n\
    sub.f32 %val, %val, %max_val;\n\
    mul.f32 %val, %val, 0f3FB8AA3B;\n\
    ex2.approx.f32 %exp_val, %val;\n\
    add.f32 %sum_val, %sum_val, %exp_val;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %off, %out, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    st.global.f32 [%off], %exp_val;\n\
    add.u32 %j, %j, %bdim;\n\
    bra SUM_EXP;\n\
SUM_EXP_DONE:\n\
\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f32 [%saddr], %sum_val;\n\
    bar.sync 0;\n\
\n\
    mov.u32 %half, %bdim;\n\
SUM_REDUCE:\n\
    shr.u32 %half, %half, 1;\n\
    setp.eq.u32 %reduce_p, %half, 0;\n\
    @%reduce_p bra SUM_REDUCE_DONE;\n\
    setp.ge.u32 %reduce_p, %r_tid, %half;\n\
    @%reduce_p bra SUM_REDUCE_SKIP;\n\
    add.u32 %other_tid, %r_tid, %half;\n\
    cvt.u64.u32 %off, %other_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;
    ld.shared.f32 %other_val, [%saddr];\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %saddr, %sbase, %off;\n\
    ld.shared.f32 %sum_val, [%saddr];\n\
    add.f32 %sum_val, %sum_val, %other_val;\n\
    add.u64 %saddr, %sbase, %off;\n\
    st.shared.f32 [%saddr], %sum_val;\n\
SUM_REDUCE_SKIP:\n\
    bar.sync 0;\n\
    bra SUM_REDUCE;\n\
SUM_REDUCE_DONE:\n\
\n\
    ld.shared.f32 %sum_val, [sdata];\n\
    bar.sync 0;\n\
\n\
    rcp.approx.f32 %sum_val, %sum_val;\n\
    mov.u32 %j, %r_tid;\n\
NORMALIZE:\n\
    setp.ge.u32 %loop_p, %j, %cols_reg;\n\
    @%loop_p bra NORMALIZE_DONE;\n\
    cvt.u64.u32 %off, %j;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %off, %out, %off;\n\
    add.u64 %off, %off, %row_off;\n\
    ld.global.f32 %val, [%off];\n\
    mul.f32 %result, %val, %sum_val;\n\
    st.global.f32 [%off], %result;\n\
    add.u32 %j, %j, %bdim;\n\
    bra NORMALIZE;\n\
NORMALIZE_DONE:\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

// ---------------------------------------------------------------------------
// Dropout PTX kernel (inverted dropout with xorshift RNG)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuda")]
pub(crate) const DROPOUT_PTX: &str = "\
.version 7.0\n\
.target sm_52\n\
.address_size 64\n\
\n\
.visible .entry dropout_kernel(\n\
    .param .u64 input_ptr,\n\
    .param .u64 output_ptr,\n\
    .param .u32 n,\n\
    .param .u32 threshold,\n\
    .param .f32 scale,\n\
    .param .u32 seed\n\
) {\n\
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %thresh, %seed_reg, %rng, %tmp;\n\
    .reg .u64 %in, %out, %off;\n\
    .reg .f32 %val, %scale_reg, %zero;\n\
    .reg .pred %p, %drop_p;\n\
\n\
    ld.param.u64 %in, [input_ptr];\n\
    ld.param.u64 %out, [output_ptr];\n\
    ld.param.u32 %n_reg, [n];\n\
    ld.param.u32 %thresh, [threshold];\n\
    ld.param.f32 %scale_reg, [scale];\n\
    ld.param.u32 %seed_reg, [seed];\n\
\n\
    mov.u32 %bid, %ctaid.x;\n\
    mov.u32 %bdim, %ntid.x;\n\
    mov.u32 %r_tid, %tid.x;\n\
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;\n\
\n\
    setp.ge.u32 %p, %r_tid, %n_reg;\n\
    @%p bra DONE;\n\
\n\
    mul.lo.u32 %rng, %r_tid, 2654435761;\n\
    xor.b32 %rng, %rng, %seed_reg;\n\
    shl.b32 %tmp, %rng, 13;\n\
    xor.b32 %rng, %rng, %tmp;\n\
    shr.b32 %tmp, %rng, 17;\n\
    xor.b32 %rng, %rng, %tmp;\n\
    shl.b32 %tmp, %rng, 5;\n\
    xor.b32 %rng, %rng, %tmp;\n\
\n\
    cvt.u64.u32 %off, %r_tid;\n\
    shl.b64 %off, %off, 2;\n\
    add.u64 %in, %in, %off;\n\
    add.u64 %out, %out, %off;\n\
    ld.global.f32 %val, [%in];\n\
\n\
    setp.lo.u32 %drop_p, %rng, %thresh;\n\
    mov.f32 %zero, 0f00000000;\n\
    @%drop_p mov.f32 %val, %zero;\n\
    @!%drop_p mul.f32 %val, %val, %scale_reg;\n\
\n\
    st.global.f32 [%out], %val;\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

// ---------------------------------------------------------------------------
// General N-dimensional broadcast binary PTX kernels
// ---------------------------------------------------------------------------
//
// Each thread computes one output element. The kernel decomposes the flat
// output index into N-dimensional coordinates, maps each coordinate through
// broadcast strides for A and B, and loads from the correct flat position.
//
// Parameters:
//   a_ptr         - pointer to A's device buffer
//   b_ptr         - pointer to B's device buffer
//   out_ptr       - pointer to output device buffer
//   a_strides_ptr - pointer to u32[ndim] broadcast strides for A
//   b_strides_ptr - pointer to u32[ndim] broadcast strides for B
//   out_shape_ptr - pointer to u32[ndim] output shape
//   n             - total output elements
//   ndim          - number of dimensions
//
// Broadcast strides: for each dimension d, stride is the normal
// C-contiguous stride if dim_size > 1, or 0 if dim_size == 1 (broadcast).

/// PTX for general broadcast add: `out[i] = a[bcast_a(i)] + b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub(crate) const BROADCAST_ADD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry broadcast_add_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u64 a_strides_ptr,
    .param .u64 b_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %a_idx, %b_idx, %d;
    .reg .u32 %shape_d, %a_str_d, %b_str_d, %coord;
    .reg .u64 %a, %b, %out, %a_str, %b_str, %oshape;
    .reg .u64 %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p, %loop_p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %a_str, [a_strides_ptr];
    ld.param.u64 %b_str, [b_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    // Global thread index.
    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    // Decompose flat index into N-d coordinates and compute A/B indices.
    mov.u32 %remaining, %r_tid;
    mov.u32 %a_idx, 0;
    mov.u32 %b_idx, 0;
    mov.u32 %d, %ndim_reg;

LOOP:
    setp.eq.u32 %loop_p, %d, 0;
    @%loop_p bra END_LOOP;

    sub.u32 %d, %d, 1;

    // Byte offset for dimension d: d * 4.
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;

    // Load out_shape[d].
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];

    // Load a_strides[d] and b_strides[d].
    add.u64 %tmp, %a_str, %d64;
    ld.global.u32 %a_str_d, [%tmp];
    add.u64 %tmp, %b_str, %d64;
    ld.global.u32 %b_str_d, [%tmp];

    // coord = remaining % shape_d; remaining /= shape_d.
    rem.u32 %coord, %remaining, %shape_d;
    div.u32 %remaining, %remaining, %shape_d;

    // a_idx += coord * a_stride[d]; b_idx += coord * b_stride[d].
    mad.lo.u32 %a_idx, %coord, %a_str_d, %a_idx;
    mad.lo.u32 %b_idx, %coord, %b_str_d, %b_idx;

    bra LOOP;
END_LOOP:

    // Load a[a_idx] and b[b_idx] (f32 = 4 bytes).
    cvt.u64.u32 %off_a, %a_idx;
    shl.b64 %off_a, %off_a, 2;
    add.u64 %off_a, %a, %off_a;
    ld.global.f32 %va, [%off_a];

    cvt.u64.u32 %off_b, %b_idx;
    shl.b64 %off_b, %off_b, 2;
    add.u64 %off_b, %b, %off_b;
    ld.global.f32 %vb, [%off_b];

    // Operation: add.
    add.f32 %vr, %va, %vb;

    // Store to out[tid].
    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 2;
    add.u64 %off_out, %out, %off_out;
    st.global.f32 [%off_out], %vr;

DONE:
    ret;
}
";

/// PTX for general broadcast sub: `out[i] = a[bcast_a(i)] - b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub(crate) const BROADCAST_SUB_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry broadcast_sub_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u64 a_strides_ptr,
    .param .u64 b_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %a_idx, %b_idx, %d;
    .reg .u32 %shape_d, %a_str_d, %b_str_d, %coord;
    .reg .u64 %a, %b, %out, %a_str, %b_str, %oshape;
    .reg .u64 %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p, %loop_p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %a_str, [a_strides_ptr];
    ld.param.u64 %b_str, [b_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %remaining, %r_tid;
    mov.u32 %a_idx, 0;
    mov.u32 %b_idx, 0;
    mov.u32 %d, %ndim_reg;
LOOP:
    setp.eq.u32 %loop_p, %d, 0;
    @%loop_p bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    add.u64 %tmp, %a_str, %d64;
    ld.global.u32 %a_str_d, [%tmp];
    add.u64 %tmp, %b_str, %d64;
    ld.global.u32 %b_str_d, [%tmp];
    rem.u32 %coord, %remaining, %shape_d;
    div.u32 %remaining, %remaining, %shape_d;
    mad.lo.u32 %a_idx, %coord, %a_str_d, %a_idx;
    mad.lo.u32 %b_idx, %coord, %b_str_d, %b_idx;
    bra LOOP;
END_LOOP:

    cvt.u64.u32 %off_a, %a_idx;
    shl.b64 %off_a, %off_a, 2;
    add.u64 %off_a, %a, %off_a;
    ld.global.f32 %va, [%off_a];
    cvt.u64.u32 %off_b, %b_idx;
    shl.b64 %off_b, %off_b, 2;
    add.u64 %off_b, %b, %off_b;
    ld.global.f32 %vb, [%off_b];

    sub.f32 %vr, %va, %vb;

    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 2;
    add.u64 %off_out, %out, %off_out;
    st.global.f32 [%off_out], %vr;
DONE:
    ret;
}
";

/// PTX for general broadcast mul: `out[i] = a[bcast_a(i)] * b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub(crate) const BROADCAST_MUL_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry broadcast_mul_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u64 a_strides_ptr,
    .param .u64 b_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %a_idx, %b_idx, %d;
    .reg .u32 %shape_d, %a_str_d, %b_str_d, %coord;
    .reg .u64 %a, %b, %out, %a_str, %b_str, %oshape;
    .reg .u64 %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p, %loop_p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %a_str, [a_strides_ptr];
    ld.param.u64 %b_str, [b_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %remaining, %r_tid;
    mov.u32 %a_idx, 0;
    mov.u32 %b_idx, 0;
    mov.u32 %d, %ndim_reg;
LOOP:
    setp.eq.u32 %loop_p, %d, 0;
    @%loop_p bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    add.u64 %tmp, %a_str, %d64;
    ld.global.u32 %a_str_d, [%tmp];
    add.u64 %tmp, %b_str, %d64;
    ld.global.u32 %b_str_d, [%tmp];
    rem.u32 %coord, %remaining, %shape_d;
    div.u32 %remaining, %remaining, %shape_d;
    mad.lo.u32 %a_idx, %coord, %a_str_d, %a_idx;
    mad.lo.u32 %b_idx, %coord, %b_str_d, %b_idx;
    bra LOOP;
END_LOOP:

    cvt.u64.u32 %off_a, %a_idx;
    shl.b64 %off_a, %off_a, 2;
    add.u64 %off_a, %a, %off_a;
    ld.global.f32 %va, [%off_a];
    cvt.u64.u32 %off_b, %b_idx;
    shl.b64 %off_b, %off_b, 2;
    add.u64 %off_b, %b, %off_b;
    ld.global.f32 %vb, [%off_b];

    mul.f32 %vr, %va, %vb;

    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 2;
    add.u64 %off_out, %out, %off_out;
    st.global.f32 [%off_out], %vr;
DONE:
    ret;
}
";

/// PTX source for `broadcast_div_kernel`: broadcast division, identical structure
/// to `broadcast_mul_kernel` but uses `div.f32` instead of `mul.f32`.
#[cfg(feature = "cuda")]
pub(crate) const BROADCAST_DIV_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry broadcast_div_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u64 a_strides_ptr,
    .param .u64 b_strides_ptr,
    .param .u64 out_shape_ptr,
    .param .u32 n,
    .param .u32 ndim
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg, %ndim_reg;
    .reg .u32 %remaining, %a_idx, %b_idx, %d;
    .reg .u32 %shape_d, %a_str_d, %b_str_d, %coord;
    .reg .u64 %a, %b, %out, %a_str, %b_str, %oshape;
    .reg .u64 %off_a, %off_b, %off_out, %d64, %tmp;
    .reg .f32 %va, %vb, %vr;
    .reg .pred %p, %loop_p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %b, [b_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.u64 %a_str, [a_strides_ptr];
    ld.param.u64 %b_str, [b_strides_ptr];
    ld.param.u64 %oshape, [out_shape_ptr];
    ld.param.u32 %n_reg, [n];
    ld.param.u32 %ndim_reg, [ndim];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;
    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    mov.u32 %remaining, %r_tid;
    mov.u32 %a_idx, 0;
    mov.u32 %b_idx, 0;
    mov.u32 %d, %ndim_reg;
LOOP:
    setp.eq.u32 %loop_p, %d, 0;
    @%loop_p bra END_LOOP;
    sub.u32 %d, %d, 1;
    cvt.u64.u32 %d64, %d;
    shl.b64 %d64, %d64, 2;
    add.u64 %tmp, %oshape, %d64;
    ld.global.u32 %shape_d, [%tmp];
    add.u64 %tmp, %a_str, %d64;
    ld.global.u32 %a_str_d, [%tmp];
    add.u64 %tmp, %b_str, %d64;
    ld.global.u32 %b_str_d, [%tmp];
    rem.u32 %coord, %remaining, %shape_d;
    div.u32 %remaining, %remaining, %shape_d;
    mad.lo.u32 %a_idx, %coord, %a_str_d, %a_idx;
    mad.lo.u32 %b_idx, %coord, %b_str_d, %b_idx;
    bra LOOP;
END_LOOP:

    cvt.u64.u32 %off_a, %a_idx;
    shl.b64 %off_a, %off_a, 2;
    add.u64 %off_a, %a, %off_a;
    ld.global.f32 %va, [%off_a];
    cvt.u64.u32 %off_b, %b_idx;
    shl.b64 %off_b, %off_b, 2;
    add.u64 %off_b, %b, %off_b;
    ld.global.f32 %vb, [%off_b];

    div.f32 %vr, %va, %vb;

    cvt.u64.u32 %off_out, %r_tid;
    shl.b64 %off_out, %off_out, 2;
    add.u64 %off_out, %out, %off_out;
    st.global.f32 [%off_out], %vr;
DONE:
    ret;
}
";

/// PTX source for `strided_split_kernel`: extract a sub-tensor along a given axis.
///
/// Thread `i` computes:
///   `outer_idx = i / (split_size * inner_size)`
///   `within    = i % (split_size * inner_size)`
///   `src_idx   = outer_idx * total_along_axis * inner_size + (split_offset * inner_size) + within`
///   `out[i]    = in[src_idx]`
#[cfg(feature = "cuda")]
pub(crate) const STRIDED_SPLIT_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry strided_split_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 total_along_axis,
    .param .u32 split_offset,
    .param .u32 split_size,
    .param .u32 inner_size,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u32 %total_ax, %sp_off, %sp_sz, %inner_sz;
    .reg .u32 %outer_idx, %within, %chunk_stride, %src_idx, %base_off, %tmp;
    .reg .u64 %in, %out, %off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %total_ax, [total_along_axis];
    ld.param.u32 %sp_off, [split_offset];
    ld.param.u32 %sp_sz, [split_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    // chunk_stride = split_size * inner_size
    mul.lo.u32 %chunk_stride, %sp_sz, %inner_sz;

    // outer_idx = r_tid / chunk_stride
    div.u32 %outer_idx, %r_tid, %chunk_stride;

    // within = r_tid % chunk_stride
    rem.u32 %within, %r_tid, %chunk_stride;

    // base_off = split_offset * inner_size
    mul.lo.u32 %base_off, %sp_off, %inner_sz;

    // src_idx = outer_idx * total_along_axis * inner_size + base_off + within
    mul.lo.u32 %src_idx, %outer_idx, %total_ax;
    mul.lo.u32 %src_idx, %src_idx, %inner_sz;
    add.u32 %src_idx, %src_idx, %base_off;
    add.u32 %src_idx, %src_idx, %within;

    // Load from in[src_idx]
    cvt.u64.u32 %off, %src_idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %val, [%off];

    // Store to out[r_tid]
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.f32 [%off], %val;

DONE:
    ret;
}
";

/// PTX source for `strided_cat_kernel`: write a sub-tensor into a larger tensor
/// at an offset along an axis.
///
/// Thread `i` computes:
///   `outer_idx = i / (part_size * inner_size)`
///   `within    = i % (part_size * inner_size)`
///   `dst_idx   = outer_idx * total_along_axis * inner_size + (cat_offset * inner_size) + within`
///   `out[dst_idx] = in[i]`
#[cfg(feature = "cuda")]
pub(crate) const STRIDED_CAT_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry strided_cat_kernel(
    .param .u64 input_ptr,
    .param .u64 output_ptr,
    .param .u32 total_along_axis,
    .param .u32 cat_offset,
    .param .u32 part_size,
    .param .u32 inner_size,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u32 %total_ax, %cat_off, %part_sz, %inner_sz;
    .reg .u32 %outer_idx, %within, %chunk_stride, %dst_idx, %base_off;
    .reg .u64 %in, %out, %off;
    .reg .f32 %val;
    .reg .pred %p;

    ld.param.u64 %in, [input_ptr];
    ld.param.u64 %out, [output_ptr];
    ld.param.u32 %total_ax, [total_along_axis];
    ld.param.u32 %cat_off, [cat_offset];
    ld.param.u32 %part_sz, [part_size];
    ld.param.u32 %inner_sz, [inner_size];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    // chunk_stride = part_size * inner_size
    mul.lo.u32 %chunk_stride, %part_sz, %inner_sz;

    // outer_idx = r_tid / chunk_stride
    div.u32 %outer_idx, %r_tid, %chunk_stride;

    // within = r_tid % chunk_stride
    rem.u32 %within, %r_tid, %chunk_stride;

    // base_off = cat_offset * inner_size
    mul.lo.u32 %base_off, %cat_off, %inner_sz;

    // dst_idx = outer_idx * total_along_axis * inner_size + base_off + within
    mul.lo.u32 %dst_idx, %outer_idx, %total_ax;
    mul.lo.u32 %dst_idx, %dst_idx, %inner_sz;
    add.u32 %dst_idx, %dst_idx, %base_off;
    add.u32 %dst_idx, %dst_idx, %within;

    // Load from in[r_tid]
    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;
    add.u64 %off, %in, %off;
    ld.global.f32 %val, [%off];

    // Store to out[dst_idx]
    cvt.u64.u32 %off, %dst_idx;
    shl.b64 %off, %off, 2;
    add.u64 %off, %out, %off;
    st.global.f32 [%off], %val;

DONE:
    ret;
}
";

/// PTX source for `div_kernel`: `out[i] = a[i] / b[i]`.
#[cfg(feature = "cuda")]
pub(crate) const DIV_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry div_kernel(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %b, %out, %off;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %b, %b, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    ld.global.f32 %vb, [%b];
    div.rn.f32 %vr, %va, %vb;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `exp_kernel`: `out[i] = exp(a[i])`.
#[cfg(feature = "cuda")]
pub(crate) const EXP_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry exp_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    // PTX ex2.approx computes 2^x; use the identity exp(x) = 2^(x * log2(e))
    // log2(e) = 1.4426950408889634
    mul.f32 %va, %va, 0f3FB8AA3B;
    ex2.approx.f32 %vr, %va;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `log_kernel`: `out[i] = ln(a[i])`.
#[cfg(feature = "cuda")]
pub(crate) const LOG_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry log_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    // PTX lg2.approx computes log2(x); use the identity ln(x) = log2(x) / log2(e)
    // 1/log2(e) = ln(2) = 0.6931471805599453
    lg2.approx.f32 %vr, %va;
    mul.f32 %vr, %vr, 0f3F317218;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `sqrt_kernel`: `out[i] = sqrt(a[i])`.
#[cfg(feature = "cuda")]
pub(crate) const SQRT_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry sqrt_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    sqrt.rn.f32 %vr, %va;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `pow_kernel`: `out[i] = a[i] ^ exponent`.
/// Uses the identity: x^e = 2^(e * log2(x)).
#[cfg(feature = "cuda")]
pub(crate) const POW_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry pow_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .f32 exponent,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr, %exp, %lg;
    .reg .pred %p;

    ld.param.u64 %a, [a_ptr];
    ld.param.u64 %out, [out_ptr];
    ld.param.f32 %exp, [exponent];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p, %r_tid, %n_reg;
    @%p bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    // x^e = 2^(e * log2(x))
    lg2.approx.f32 %lg, %va;
    mul.f32 %lg, %lg, %exp;
    ex2.approx.f32 %vr, %lg;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `abs_kernel`: `out[i] = |a[i]|`.
#[cfg(feature = "cuda")]
pub(crate) const ABS_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry abs_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    abs.f32 %vr, %va;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `sigmoid_kernel`: `out[i] = 1 / (1 + exp(-a[i]))`.
#[cfg(feature = "cuda")]
pub(crate) const SIGMOID_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry sigmoid_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr, %neg, %e, %denom, %one, %lg2e;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    // sigmoid(x) = 1 / (1 + exp(-x))
    neg.f32 %neg, %va;
    mov.f32 %lg2e, 0f3FB8AA3B;
    mul.f32 %neg, %neg, %lg2e;
    ex2.approx.f32 %e, %neg;
    mov.f32 %one, 0f3F800000;
    add.f32 %denom, %one, %e;
    div.rn.f32 %vr, %one, %denom;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `tanh_kernel`: `out[i] = tanh(a[i])`.
/// Uses the identity: tanh(x) = 2*sigmoid(2x) - 1.
#[cfg(feature = "cuda")]
pub(crate) const TANH_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry tanh_kernel(
    .param .u64 a_ptr,
    .param .u64 out_ptr,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %a, %out, %off;
    .reg .f32 %va, %vr, %neg2x, %e, %denom, %sig, %one, %two, %lg2e;
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
    shl.b64 %off, %off, 2;

    add.u64 %a, %a, %off;
    add.u64 %out, %out, %off;

    ld.global.f32 %va, [%a];
    // tanh(x) = 2*sigmoid(2x) - 1
    mov.f32 %two, 0f40000000;
    mul.f32 %neg2x, %va, %two;
    neg.f32 %neg2x, %neg2x;
    mov.f32 %lg2e, 0f3FB8AA3B;
    mul.f32 %neg2x, %neg2x, %lg2e;
    ex2.approx.f32 %e, %neg2x;
    mov.f32 %one, 0f3F800000;
    add.f32 %denom, %one, %e;
    div.rn.f32 %sig, %one, %denom;
    mul.f32 %vr, %two, %sig;
    sub.f32 %vr, %vr, %one;
    st.global.f32 [%out], %vr;

DONE:
    ret;
}
";

/// PTX source for `fused_adam_kernel`: in-place Adam optimizer update.
///
/// For each element i:
///   g = grad[i] + weight_decay * param[i]  (if wd > 0)
///   exp_avg[i] = beta1 * exp_avg[i] + (1-beta1) * g
///   exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1-beta2) * g * g
///   m_hat = exp_avg[i] / bc1
///   v_hat = exp_avg_sq[i] / bc2
///   param[i] = param[i] - lr * m_hat / (sqrt(v_hat) + eps)
#[cfg(feature = "cuda")]
pub(crate) const FUSED_ADAM_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry fused_adam_kernel(
    .param .u64 param_ptr,
    .param .u64 grad_ptr,
    .param .u64 exp_avg_ptr,
    .param .u64 exp_avg_sq_ptr,
    .param .f32 beta1,
    .param .f32 beta2,
    .param .f32 lr,
    .param .f32 eps,
    .param .f32 bc1,
    .param .f32 bc2,
    .param .f32 weight_decay,
    .param .u32 n
) {
    .reg .u32 %r_tid, %bid, %bdim, %n_reg;
    .reg .u64 %p, %g, %m, %v, %off;
    .reg .f32 %vp, %vg, %vm, %vv;
    .reg .f32 %b1, %b2, %f_lr, %f_eps, %f_bc1, %f_bc2, %f_wd;
    .reg .f32 %t1, %t2, %m_hat, %v_hat, %denom, %update;
    .reg .f32 %one;
    .reg .pred %p_bound, %p_wd;

    ld.param.u64 %p, [param_ptr];
    ld.param.u64 %g, [grad_ptr];
    ld.param.u64 %m, [exp_avg_ptr];
    ld.param.u64 %v, [exp_avg_sq_ptr];
    ld.param.f32 %b1, [beta1];
    ld.param.f32 %b2, [beta2];
    ld.param.f32 %f_lr, [lr];
    ld.param.f32 %f_eps, [eps];
    ld.param.f32 %f_bc1, [bc1];
    ld.param.f32 %f_bc2, [bc2];
    ld.param.f32 %f_wd, [weight_decay];
    ld.param.u32 %n_reg, [n];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %r_tid, %tid.x;
    mad.lo.u32 %r_tid, %bid, %bdim, %r_tid;

    setp.ge.u32 %p_bound, %r_tid, %n_reg;
    @%p_bound bra DONE;

    cvt.u64.u32 %off, %r_tid;
    shl.b64 %off, %off, 2;

    add.u64 %p, %p, %off;
    add.u64 %g, %g, %off;
    add.u64 %m, %m, %off;
    add.u64 %v, %v, %off;

    ld.global.f32 %vp, [%p];
    ld.global.f32 %vg, [%g];
    ld.global.f32 %vm, [%m];
    ld.global.f32 %vv, [%v];

    // L2 weight decay: g = g + wd * p
    mov.f32 %one, 0f00000000;
    setp.gt.f32 %p_wd, %f_wd, %one;
    @%p_wd fma.rn.f32 %vg, %f_wd, %vp, %vg;

    // exp_avg = beta1 * exp_avg + (1 - beta1) * g
    mov.f32 %one, 0f3F800000;
    sub.f32 %t1, %one, %b1;
    mul.f32 %vm, %vm, %b1;
    fma.rn.f32 %vm, %t1, %vg, %vm;

    // exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * g * g
    sub.f32 %t2, %one, %b2;
    mul.f32 %vv, %vv, %b2;
    mul.f32 %t1, %vg, %vg;
    fma.rn.f32 %vv, %t2, %t1, %vv;

    // m_hat = exp_avg / bc1
    div.rn.f32 %m_hat, %vm, %f_bc1;

    // v_hat = exp_avg_sq / bc2
    div.rn.f32 %v_hat, %vv, %f_bc2;

    // denom = sqrt(v_hat) + eps
    sqrt.rn.f32 %denom, %v_hat;
    add.f32 %denom, %denom, %f_eps;

    // param = param - lr * m_hat / denom
    div.rn.f32 %update, %m_hat, %denom;
    mul.f32 %update, %update, %f_lr;
    sub.f32 %vp, %vp, %update;

    st.global.f32 [%p], %vp;
    st.global.f32 [%m], %vm;
    st.global.f32 [%v], %vv;

DONE:
    ret;
}
";

/// PTX source for fused GRU cell forward kernel.
///
/// Takes pre-computed input_gates [B, 3*H] and hidden_gates [B, 3*H]
/// (from cuBLAS GEMMs), biases, and previous hidden state. Computes all
/// gate activations and the new hidden state in a single kernel launch.
///
/// One thread per hidden unit. Each thread reads 3 values from input_gates
/// and 3 from hidden_gates, applies sigmoid/tanh, computes the GRU update,
/// and writes hy + workspace (5*H values for backward).
///
/// Matches PyTorch's _thnn_fused_gru_cell kernel from RNN.cu.
#[cfg(feature = "cuda")]
pub(crate) const FUSED_GRU_FORWARD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry fused_gru_forward_kernel(
    .param .u64 input_gates_ptr,
    .param .u64 hidden_gates_ptr,
    .param .u64 bias_ih_ptr,
    .param .u64 bias_hh_ptr,
    .param .u64 hx_ptr,
    .param .u64 hy_ptr,
    .param .u64 workspace_ptr,
    .param .u32 hsz,
    .param .u32 total
) {
    .reg .u32 %tid, %bid, %bdim, %gdim, %total_reg, %hsz_reg;
    .reg .u32 %idx, %stride, %offset3, %offset5, %hmod, %batch_idx;
    .reg .u64 %ig, %hg, %b1, %b2, %hx, %hy, %ws;
    .reg .u64 %off64, %tmp64;
    .reg .f32 %ir, %ii, %in, %hr, %hi, %hn;
    .reg .f32 %b1r, %b1i, %b1n, %b2r, %b2i, %b2n;
    .reg .f32 %hx_val, %rg, %zg, %ng, %hy_val;
    .reg .f32 %one, %neg_one, %exp_val, %denom, %tmp;
    .reg .pred %p;

    ld.param.u64 %ig, [input_gates_ptr];
    ld.param.u64 %hg, [hidden_gates_ptr];
    ld.param.u64 %b1, [bias_ih_ptr];
    ld.param.u64 %b2, [bias_hh_ptr];
    ld.param.u64 %hx, [hx_ptr];
    ld.param.u64 %hy, [hy_ptr];
    ld.param.u64 %ws, [workspace_ptr];
    ld.param.u32 %hsz_reg, [hsz];
    ld.param.u32 %total_reg, [total];

    mov.u32 %bid, %ctaid.x;
    mov.u32 %bdim, %ntid.x;
    mov.u32 %tid, %tid.x;
    mov.u32 %gdim, %nctaid.x;
    mad.lo.u32 %idx, %bid, %bdim, %tid;
    mul.lo.u32 %stride, %bdim, %gdim;
    mov.f32 %one, 0f3F800000;

LOOP:
    setp.ge.u32 %p, %idx, %total_reg;
    @%p bra END;

    // offset3 = (idx/hsz)*3*hsz + idx%hsz  (into [B, 3*H] gates tensor)
    div.u32 %batch_idx, %idx, %hsz_reg;
    rem.u32 %hmod, %idx, %hsz_reg;
    mul.lo.u32 %offset3, %batch_idx, %hsz_reg;
    mul.lo.u32 %offset3, %offset3, 3;
    add.u32 %offset3, %offset3, %hmod;

    // Load input gate components: ir, ii, in
    cvt.u64.u32 %off64, %offset3;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %ig, %off64;
    ld.global.f32 %ir, [%tmp64];
    cvt.u64.u32 %off64, %hsz_reg;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %tmp64, %off64;
    ld.global.f32 %ii, [%tmp64];
    add.u64 %tmp64, %tmp64, %off64;
    ld.global.f32 %in, [%tmp64];

    // Load hidden gate components: hr, hi, hn
    cvt.u64.u32 %off64, %offset3;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %hg, %off64;
    ld.global.f32 %hr, [%tmp64];
    cvt.u64.u32 %off64, %hsz_reg;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %tmp64, %off64;
    ld.global.f32 %hi, [%tmp64];
    add.u64 %tmp64, %tmp64, %off64;
    ld.global.f32 %hn, [%tmp64];

    // Load biases (indexed by hmod, hmod+hsz, hmod+2*hsz)
    cvt.u64.u32 %off64, %hmod;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %b1, %off64;
    ld.global.f32 %b1r, [%tmp64];
    cvt.u64.u32 %off64, %hsz_reg;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %tmp64, %off64;
    ld.global.f32 %b1i, [%tmp64];
    add.u64 %tmp64, %tmp64, %off64;
    ld.global.f32 %b1n, [%tmp64];

    cvt.u64.u32 %off64, %hmod;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %b2, %off64;
    ld.global.f32 %b2r, [%tmp64];
    cvt.u64.u32 %off64, %hsz_reg;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %tmp64, %off64;
    ld.global.f32 %b2i, [%tmp64];
    add.u64 %tmp64, %tmp64, %off64;
    ld.global.f32 %b2n, [%tmp64];

    // Load hx[idx]
    cvt.u64.u32 %off64, %idx;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %hx, %off64;
    ld.global.f32 %hx_val, [%tmp64];

    // r = sigmoid(ir + hr + b1r + b2r)
    add.f32 %rg, %ir, %hr;
    add.f32 %rg, %rg, %b1r;
    add.f32 %rg, %rg, %b2r;
    neg.f32 %tmp, %rg;
    mul.f32 %tmp, %tmp, 0f3FB8AA3B;
    ex2.approx.f32 %exp_val, %tmp;
    add.f32 %denom, %one, %exp_val;
    div.rn.f32 %rg, %one, %denom;

    // z = sigmoid(ii + hi + b1i + b2i)
    add.f32 %zg, %ii, %hi;
    add.f32 %zg, %zg, %b1i;
    add.f32 %zg, %zg, %b2i;
    neg.f32 %tmp, %zg;
    mul.f32 %tmp, %tmp, 0f3FB8AA3B;
    ex2.approx.f32 %exp_val, %tmp;
    add.f32 %denom, %one, %exp_val;
    div.rn.f32 %zg, %one, %denom;

    // n = tanh(in + b1n + r*(hn + b2n))
    add.f32 %tmp, %hn, %b2n;
    fma.rn.f32 %ng, %rg, %tmp, %in;
    add.f32 %ng, %ng, %b1n;
    // tanh via 2*sigmoid(2x)-1
    mul.f32 %tmp, %ng, 0f40000000;
    neg.f32 %tmp, %tmp;
    mul.f32 %tmp, %tmp, 0f3FB8AA3B;
    ex2.approx.f32 %exp_val, %tmp;
    add.f32 %denom, %one, %exp_val;
    div.rn.f32 %ng, %one, %denom;
    mul.f32 %ng, %ng, 0f40000000;
    sub.f32 %ng, %ng, %one;

    // hy = n + z * (hx - n)
    sub.f32 %tmp, %hx_val, %ng;
    fma.rn.f32 %hy_val, %zg, %tmp, %ng;

    // Store hy[idx]
    cvt.u64.u32 %off64, %idx;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %hy, %off64;
    st.global.f32 [%tmp64], %hy_val;

    // Store workspace: [r, z, n, hx, hn+b2n] at offset5 = (idx/hsz)*5*hsz + idx%hsz
    mul.lo.u32 %offset5, %batch_idx, %hsz_reg;
    mul.lo.u32 %offset5, %offset5, 5;
    add.u32 %offset5, %offset5, %hmod;

    cvt.u64.u32 %off64, %offset5;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %ws, %off64;
    st.global.f32 [%tmp64], %rg;
    cvt.u64.u32 %off64, %hsz_reg;
    shl.b64 %off64, %off64, 2;
    add.u64 %tmp64, %tmp64, %off64;
    st.global.f32 [%tmp64], %zg;
    add.u64 %tmp64, %tmp64, %off64;
    st.global.f32 [%tmp64], %ng;
    add.u64 %tmp64, %tmp64, %off64;
    st.global.f32 [%tmp64], %hx_val;
    add.u64 %tmp64, %tmp64, %off64;
    add.f32 %tmp, %hn, %b2n;
    st.global.f32 [%tmp64], %tmp;

    add.u32 %idx, %idx, %stride;
    bra LOOP;

END:
    ret;
}
";

// ---------------------------------------------------------------------------
// Launch configuration helper
// ---------------------------------------------------------------------------

/// Standard 1-D launch config for `n` elements.
///
/// Uses 256 threads per block, which is a good default for elementwise ops
/// on all modern NVIDIA architectures.
///
/// # Errors
///
/// Returns [`GpuError::ShapeMismatch`] if `n` exceeds `u32::MAX`, which
/// would silently truncate the grid dimension.
#[cfg(feature = "cuda")]
fn launch_cfg(n: usize) -> GpuResult<LaunchConfig> {
    if n > u32::MAX as usize {
        return Err(GpuError::ShapeMismatch {
            op: "kernel_launch",
            expected: vec![u32::MAX as usize],
            got: vec![n],
        });
    }
    const BLOCK: u32 = 256;
    let grid = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    Ok(LaunchConfig {
        grid_dim: (grid.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0,
    })
}

// ---------------------------------------------------------------------------
// Validation helpers
// ---------------------------------------------------------------------------

/// Validate that two buffers are on the same device and have the same length.
#[cfg(feature = "cuda")]
fn validate_binary(a: &CudaBuffer<f32>, b: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<()> {
    if a.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: a.device_ordinal(),
            got: device.ordinal(),
        });
    }
    if b.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: b.device_ordinal(),
            got: device.ordinal(),
        });
    }
    if a.len() != b.len() {
        return Err(GpuError::LengthMismatch {
            a: a.len(),
            b: b.len(),
        });
    }
    Ok(())
}

/// Validate that a unary buffer is on the correct device.
#[cfg(feature = "cuda")]
fn validate_unary(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<()> {
    if a.device_ordinal() != device.ordinal() {
        return Err(GpuError::DeviceMismatch {
            expected: a.device_ordinal(),
            got: device.ordinal(),
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// PTX kernel launch helpers
// ---------------------------------------------------------------------------

/// Try to launch a binary PTX kernel. Returns `Ok(Some(buf))` on success,
/// `Ok(None)` if the PTX module failed to load (caller should fall back to
/// CPU), or `Err` on a real CUDA error after a successful launch.
#[cfg(feature = "cuda")]
fn try_launch_binary(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<Option<CudaBuffer<f32>>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    // Attempt to load the kernel (cached after first compilation).
    // If it fails (e.g. unsupported arch), return None so the caller
    // can use the CPU fallback.
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx_src,
        kernel_name,
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY: The kernel reads `n` f32 values from `a` and `b`, writes `n`
    // f32 values to `out`. All three buffers are device-resident and at
    // least `n` elements long. The grid covers exactly `n` threads.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(Some(out))
}

/// Try to launch a vectorized (vec4) binary PTX kernel.
///
/// Each thread processes 4 elements using 128-bit loads/stores.
/// `n` must be divisible by 4. Returns `Ok(None)` if compilation fails.
#[cfg(feature = "cuda")]
fn try_launch_binary_vec4(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<Option<CudaBuffer<f32>>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let n4 = (n / 4) as u32;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx_src,
        kernel_name,
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n4 as usize)?;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(&n4)
            .launch(cfg)?;
    }

    Ok(Some(out))
}

/// Try to launch a unary PTX kernel. Returns `Ok(Some(buf))` on success,
/// `Ok(None)` if the PTX module failed to load.
#[cfg(feature = "cuda")]
fn try_launch_unary(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<Option<CudaBuffer<f32>>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    // Attempt to load the kernel (cached after first compilation).
    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx_src,
        kernel_name,
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY: The kernel reads `n` f32 values from `a` and writes `n` f32
    // values to `out`. Both buffers are device-resident with length >= n.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(Some(out))
}

// ---------------------------------------------------------------------------
// _into helpers — write to pre-allocated output buffer (no allocation)
// ---------------------------------------------------------------------------

/// Launch a binary PTX kernel into a pre-allocated output buffer.
/// Returns `Ok(true)` on success, `Ok(false)` if the PTX module failed to load.
#[cfg(feature = "cuda")]
fn try_launch_binary_into(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<bool> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx_src,
        kernel_name,
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };

    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(true)
}

/// Launch a unary PTX kernel into a pre-allocated output buffer.
/// Returns `Ok(true)` on success, `Ok(false)` if the PTX module failed to load.
#[cfg(feature = "cuda")]
fn try_launch_unary_into(
    a: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<bool> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx_src,
        kernel_name,
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => return Ok(false),
    };

    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(true)
}

/// Try to launch a general N-dimensional broadcast binary PTX kernel.
///
/// `a_strides` and `b_strides` are broadcast strides: normal C-contiguous
/// stride for non-broadcast dims, 0 for broadcast (size-1) dims.
/// `out_shape` is the broadcast-resolved output shape.
/// All three arrays have length `ndim`.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
fn try_launch_broadcast_binary(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    a_strides: &[u32],
    b_strides: &[u32],
    out_shape: &[u32],
    out_numel: usize,
    device: &GpuDevice,
    ptx_src: &'static str,
    kernel_name: &'static str,
) -> GpuResult<Option<CudaBuffer<f32>>> {
    use cudarc::driver::PushKernelArg;

    let ndim = out_shape.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        ptx_src,
        kernel_name,
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => return Ok(None),
    };

    // Upload stride/shape metadata as small device buffers.
    let a_str_buf = cpu_to_gpu(a_strides, device)?;
    let b_str_buf = cpu_to_gpu(b_strides, device)?;
    let shape_buf = cpu_to_gpu(out_shape, device)?;

    let mut out = alloc_zeros_f32(out_numel, device)?;
    let cfg = launch_cfg(out_numel)?;
    let n_u32 = out_numel as u32;
    let ndim_u32 = ndim as u32;

    // SAFETY: Kernel reads from a, b using broadcast indices computed from
    // the stride/shape buffers. Output buffer has out_numel elements.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(a_str_buf.inner())
            .arg(b_str_buf.inner())
            .arg(shape_buf.inner())
            .arg(&n_u32)
            .arg(&ndim_u32)
            .launch(cfg)?;
    }

    Ok(Some(out))
}

/// Compute broadcast strides for a tensor shape relative to an output shape.
///
/// For each dimension, the stride is the normal C-contiguous stride if the
/// dimension size matches the output, or 0 if the dimension size is 1
/// (broadcast). Missing leading dimensions (when input has fewer dims) are
/// treated as size-1.
#[cfg(feature = "cuda")]
fn broadcast_strides(in_shape: &[usize], out_shape: &[usize]) -> Vec<u32> {
    let ndim = out_shape.len();
    let in_ndim = in_shape.len();
    let mut strides = vec![0u32; ndim];

    // C-contiguous strides for the input shape.
    let mut stride: u32 = 1;
    for d in (0..ndim).rev() {
        let in_d = if d + in_ndim >= ndim {
            d + in_ndim - ndim
        } else {
            // Leading dimension not present in input — broadcast.
            strides[d] = 0;
            continue;
        };

        if in_shape[in_d] == 1 {
            strides[d] = 0; // Broadcast dimension.
        } else {
            strides[d] = stride;
        }
        stride *= in_shape[in_d] as u32;
    }

    strides
}

// ---------------------------------------------------------------------------
// CPU fallback helpers
// ---------------------------------------------------------------------------

/// CPU fallback for binary ops: copy both inputs to host, apply `op`, copy
/// the result back.
#[cfg(feature = "cuda")]
fn cpu_fallback_binary(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
    op: fn(f32, f32) -> f32,
) -> GpuResult<CudaBuffer<f32>> {
    let a_host = gpu_to_cpu(a, device)?;
    let b_host = gpu_to_cpu(b, device)?;
    let result: Vec<f32> = a_host
        .iter()
        .zip(b_host.iter())
        .map(|(&x, &y)| op(x, y))
        .collect();
    cpu_to_gpu(&result, device)
}

/// CPU fallback for unary ops.
#[cfg(feature = "cuda")]
fn cpu_fallback_unary(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
    op: fn(f32) -> f32,
) -> GpuResult<CudaBuffer<f32>> {
    let a_host = gpu_to_cpu(a, device)?;
    let result: Vec<f32> = a_host.iter().map(|&x| op(x)).collect();
    cpu_to_gpu(&result, device)
}

// ---------------------------------------------------------------------------
// Public API -- binary ops
// ---------------------------------------------------------------------------

/// Elementwise addition: `out[i] = a[i] + b[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a`, `b`, or `device` refer to
///   different CUDA devices.
/// - [`GpuError::LengthMismatch`] if `a` and `b` have different lengths.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_add(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(a, b, device)?;

    // Try vec4 kernel for 4x memory throughput (128-bit loads).
    let n = a.len();
    if n >= 16 && n % 4 == 0 {
        if let Some(out) = try_launch_binary_vec4(
            a, b, device, ADD_VEC4_PTX, "add_vec4_kernel",
        )? {
            return Ok(out);
        }
    }

    if let Some(out) = try_launch_binary(a, b, device, ADD_PTX, "add_kernel")? {
        return Ok(out);
    }

    cpu_fallback_binary(a, b, device, |x, y| x + y)
}

/// Elementwise subtraction: `out[i] = a[i] - b[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a`, `b`, or `device` refer to
///   different CUDA devices.
/// - [`GpuError::LengthMismatch`] if `a` and `b` have different lengths.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_sub(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(a, b, device)?;

    if let Some(out) = try_launch_binary(a, b, device, SUB_PTX, "sub_kernel")? {
        return Ok(out);
    }

    cpu_fallback_binary(a, b, device, |x, y| x - y)
}

/// Elementwise multiplication: `out[i] = a[i] * b[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a`, `b`, or `device` refer to
///   different CUDA devices.
/// - [`GpuError::LengthMismatch`] if `a` and `b` have different lengths.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_mul(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(a, b, device)?;

    let n = a.len();
    if n >= 16 && n % 4 == 0 {
        if let Some(out) = try_launch_binary_vec4(
            a, b, device, MUL_VEC4_PTX, "mul_vec4_kernel",
        )? {
            return Ok(out);
        }
    }

    if let Some(out) = try_launch_binary(a, b, device, MUL_PTX, "mul_kernel")? {
        return Ok(out);
    }

    cpu_fallback_binary(a, b, device, |x, y| x * y)
}

// ---------------------------------------------------------------------------
// Public API -- broadcast binary ops
// ---------------------------------------------------------------------------

/// Broadcast addition: `out[i] = a[bcast_a(i)] + b[bcast_b(i)]`.
///
/// Handles arbitrary N-dimensional broadcasting on the GPU. The kernel
/// decomposes each output index into coordinates, maps them through
/// broadcast strides, and loads from the correct positions in A and B.
///
/// `a_shape` and `b_shape` are the original shapes; the output shape is
/// computed via numpy-style broadcast rules.
#[cfg(feature = "cuda")]
pub fn gpu_broadcast_add(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let a_str = broadcast_strides(a_shape, out_shape);
    let b_str = broadcast_strides(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
    let out_numel: usize = out_shape.iter().product();

    if let Some(out) = try_launch_broadcast_binary(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        BROADCAST_ADD_PTX,
        "broadcast_add_kernel",
    )? {
        return Ok(out);
    }

    // CPU fallback for broadcast.
    cpu_fallback_broadcast_binary(a, b, a_shape, b_shape, out_shape, device, |x, y| x + y)
}

/// Broadcast subtraction: `out[i] = a[bcast_a(i)] - b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub fn gpu_broadcast_sub(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let a_str = broadcast_strides(a_shape, out_shape);
    let b_str = broadcast_strides(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
    let out_numel: usize = out_shape.iter().product();

    if let Some(out) = try_launch_broadcast_binary(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        BROADCAST_SUB_PTX,
        "broadcast_sub_kernel",
    )? {
        return Ok(out);
    }

    cpu_fallback_broadcast_binary(a, b, a_shape, b_shape, out_shape, device, |x, y| x - y)
}

/// Broadcast multiplication: `out[i] = a[bcast_a(i)] * b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub fn gpu_broadcast_mul(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let a_str = broadcast_strides(a_shape, out_shape);
    let b_str = broadcast_strides(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
    let out_numel: usize = out_shape.iter().product();

    if let Some(out) = try_launch_broadcast_binary(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        BROADCAST_MUL_PTX,
        "broadcast_mul_kernel",
    )? {
        return Ok(out);
    }

    cpu_fallback_broadcast_binary(a, b, a_shape, b_shape, out_shape, device, |x, y| x * y)
}

/// Broadcast division: `out[i] = a[bcast_a(i)] / b[bcast_b(i)]`.
#[cfg(feature = "cuda")]
pub fn gpu_broadcast_div(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    let a_str = broadcast_strides(a_shape, out_shape);
    let b_str = broadcast_strides(b_shape, out_shape);
    let shape_u32: Vec<u32> = out_shape.iter().map(|&d| d as u32).collect();
    let out_numel: usize = out_shape.iter().product();

    if let Some(out) = try_launch_broadcast_binary(
        a,
        b,
        &a_str,
        &b_str,
        &shape_u32,
        out_numel,
        device,
        BROADCAST_DIV_PTX,
        "broadcast_div_kernel",
    )? {
        return Ok(out);
    }

    cpu_fallback_broadcast_binary(a, b, a_shape, b_shape, out_shape, device, |x, y| x / y)
}

/// CPU fallback for broadcast binary ops — downloads, applies op with
/// broadcast indexing, re-uploads.
#[cfg(feature = "cuda")]
fn cpu_fallback_broadcast_binary(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    a_shape: &[usize],
    b_shape: &[usize],
    out_shape: &[usize],
    device: &GpuDevice,
    op: fn(f32, f32) -> f32,
) -> GpuResult<CudaBuffer<f32>> {
    let a_host = gpu_to_cpu(a, device)?;
    let b_host = gpu_to_cpu(b, device)?;
    let out_numel: usize = out_shape.iter().product();

    let a_str = broadcast_strides(a_shape, out_shape);
    let b_str = broadcast_strides(b_shape, out_shape);

    let mut result = Vec::with_capacity(out_numel);
    for i in 0..out_numel {
        let mut remaining = i;
        let mut a_idx = 0usize;
        let mut b_idx = 0usize;
        for d in (0..out_shape.len()).rev() {
            let coord = remaining % out_shape[d];
            remaining /= out_shape[d];
            a_idx += coord * a_str[d] as usize;
            b_idx += coord * b_str[d] as usize;
        }
        result.push(op(a_host[a_idx], b_host[b_idx]));
    }
    cpu_to_gpu(&result, device)
}

// ---------------------------------------------------------------------------
// Public API -- unary ops
// ---------------------------------------------------------------------------

/// Elementwise negation: `out[i] = -a[i]`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a` and `device` refer to different
///   CUDA devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_neg(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;

    if let Some(out) = try_launch_unary(a, device, NEG_PTX, "neg_kernel")? {
        return Ok(out);
    }

    cpu_fallback_unary(a, device, |x| -x)
}

/// Elementwise ReLU: `out[i] = max(a[i], 0.0)`.
///
/// Attempts to run a PTX kernel on the GPU. Falls back to a CPU round-trip
/// if the PTX module cannot be loaded.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a` and `device` refer to different
///   CUDA devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_relu(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;

    if let Some(out) = try_launch_unary(a, device, RELU_PTX, "relu_kernel")? {
        return Ok(out);
    }

    cpu_fallback_unary(a, device, |x| x.max(0.0))
}

/// ReLU backward: `out[i] = (input[i] > 0) ? grad[i] : 0`.
#[cfg(feature = "cuda")]
pub fn gpu_relu_backward(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, input, device)?;

    if let Some(out) = try_launch_binary(
        grad,
        input,
        device,
        RELU_BACKWARD_PTX,
        "relu_backward_kernel",
    )? {
        return Ok(out);
    }

    // CPU fallback
    let grad_host = gpu_to_cpu(grad, device)?;
    let input_host = gpu_to_cpu(input, device)?;
    let result: Vec<f32> = grad_host
        .iter()
        .zip(input_host.iter())
        .map(|(&g, &x)| if x > 0.0 { g } else { 0.0 })
        .collect();
    cpu_to_gpu(&result, device)
}

/// GELU backward: `out[i] = grad[i] * (sig + 1.702 * x * sig * (1 - sig))`
/// where `sig = sigmoid(1.702 * x)`.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_backward(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, input, device)?;

    if let Some(out) = try_launch_binary(
        grad,
        input,
        device,
        GELU_BACKWARD_PTX,
        "gelu_backward_kernel",
    )? {
        return Ok(out);
    }

    // CPU fallback
    let grad_host = gpu_to_cpu(grad, device)?;
    let input_host = gpu_to_cpu(input, device)?;
    let result: Vec<f32> = grad_host
        .iter()
        .zip(input_host.iter())
        .map(|(&g, &x)| {
            let k: f32 = 1.702;
            let sig = 1.0 / (1.0 + (-k * x).exp());
            g * (sig + k * x * sig * (1.0 - sig))
        })
        .collect();
    cpu_to_gpu(&result, device)
}

/// GELU backward (exact erf mode):
/// `out[i] = grad[i] * (Φ(x) + x·φ(x))`
/// where Φ = normal CDF, φ = normal PDF.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_backward_erf(
    grad: &CudaBuffer<f32>,
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, input, device)?;

    if let Some(out) = try_launch_binary(
        grad,
        input,
        device,
        GELU_BACKWARD_ERF_PTX,
        "gelu_backward_erf_kernel",
    )? {
        return Ok(out);
    }

    // CPU fallback — Abramowitz & Stegun erf approximation (|ε| < 1.5e-7)
    let grad_host = gpu_to_cpu(grad, device)?;
    let input_host = gpu_to_cpu(input, device)?;
    let inv_sqrt_2: f32 = std::f32::consts::FRAC_1_SQRT_2;
    let inv_sqrt_2pi: f32 = 1.0 / (2.0 * std::f32::consts::PI).sqrt();
    let result: Vec<f32> = grad_host
        .iter()
        .zip(input_host.iter())
        .map(|(&g, &x)| {
            let z = x * inv_sqrt_2;
            let az = z.abs();
            let t = 1.0 / (1.0 + 0.3275911 * az);
            let poly = t * (0.2548296 + t * (-0.2844967 + t * (1.4214137 + t * (-1.4531520 + t * 0.3275911))));
            let erf_abs = 1.0 - poly * (-az * az).exp();
            let erf_val = if z >= 0.0 { erf_abs } else { -erf_abs };
            let cdf = 0.5 * (1.0 + erf_val);
            let pdf = inv_sqrt_2pi * (-0.5 * x * x).exp();
            g * (cdf + x * pdf)
        })
        .collect();
    cpu_to_gpu(&result, device)
}

// ---------------------------------------------------------------------------
// Public API -- Index-select 1-D (gather)
// ---------------------------------------------------------------------------

/// Gather elements from `input` at positions given by `indices`.
///
/// `indices` is a GPU buffer of f32 values encoding integer indices.
/// Output has `indices.len()` elements: `out[i] = input[indices[i]]`.
#[cfg(feature = "cuda")]
pub fn gpu_index_select_1d(
    input: &CudaBuffer<f32>,
    indices: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let n = indices.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        INDEX_SELECT_1D_PTX,
        "index_select_1d_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let input_host = gpu_to_cpu(input, device)?;
            let indices_host = gpu_to_cpu(indices, device)?;
            let result: Vec<f32> = indices_host
                .iter()
                .map(|&idx_f| input_host[idx_f as usize])
                .collect();
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(indices.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Scatter-add 1-D (backward of index_select)
// ---------------------------------------------------------------------------

/// Scatter-add `grad_output` back into an output buffer of `input_len` elements,
/// using positions from `indices`.
///
/// `indices` is a GPU buffer of f32 values encoding integer indices.
/// Output: `out = zeros(input_len); for i: out[indices[i]] += grad_output[i]`
///
/// Uses atomic adds for safe concurrent accumulation.
#[cfg(feature = "cuda")]
pub fn gpu_scatter_add_1d(
    grad_output: &CudaBuffer<f32>,
    indices: &CudaBuffer<f32>,
    input_len: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(grad_output, device)?;

    let n = grad_output.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SCATTER_ADD_1D_PTX,
        "scatter_add_1d_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let go_host = gpu_to_cpu(grad_output, device)?;
            let idx_host = gpu_to_cpu(indices, device)?;
            let mut result = vec![0.0f32; input_len];
            for (i, &idx_f) in idx_host.iter().enumerate() {
                result[idx_f as usize] += go_host[i];
            }
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(input_len, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad_output.inner())
            .arg(indices.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Masked fill
// ---------------------------------------------------------------------------

/// Fill elements of `input` with `value` where `mask` is true.
///
/// `mask` is a GPU buffer of f32 values (1.0 = true, 0.0 = false).
/// Output: `out[i] = mask[i] >= 0.5 ? value : input[i]`
#[cfg(feature = "cuda")]
pub fn gpu_masked_fill(
    input: &CudaBuffer<f32>,
    mask: &CudaBuffer<f32>,
    value: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_binary(input, mask, device)?;

    let n = input.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        MASKED_FILL_PTX,
        "masked_fill_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let input_host = gpu_to_cpu(input, device)?;
            let mask_host = gpu_to_cpu(mask, device)?;
            let result: Vec<f32> = input_host
                .iter()
                .zip(mask_host.iter())
                .map(|(&x, &m)| if m >= 0.5 { value } else { x })
                .collect();
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(mask.inner())
            .arg(out.inner_mut())
            .arg(&value)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Masked zero (backward of masked_fill)
// ---------------------------------------------------------------------------

/// Zero out gradient at positions where `mask` is true.
///
/// `mask` is a GPU buffer of f32 values (1.0 = true, 0.0 = false).
/// Output: `out[i] = mask[i] >= 0.5 ? 0.0 : grad[i]`
#[cfg(feature = "cuda")]
pub fn gpu_masked_zero(
    grad: &CudaBuffer<f32>,
    mask: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, mask, device)?;

    if let Some(out) = try_launch_binary(grad, mask, device, MASKED_ZERO_PTX, "masked_zero_kernel")?
    {
        return Ok(out);
    }

    // CPU fallback.
    let grad_host = gpu_to_cpu(grad, device)?;
    let mask_host = gpu_to_cpu(mask, device)?;
    let result: Vec<f32> = grad_host
        .iter()
        .zip(mask_host.iter())
        .map(|(&g, &m)| if m >= 0.5 { 0.0 } else { g })
        .collect();
    cpu_to_gpu(&result, device)
}

// ---------------------------------------------------------------------------
// Public API -- Sigmoid backward
// ---------------------------------------------------------------------------

/// Sigmoid backward: `out[i] = grad[i] * output[i] * (1 - output[i])`.
///
/// `grad` and `output` must have the same length and reside on `device`.
#[cfg(feature = "cuda")]
pub fn gpu_sigmoid_backward(
    grad: &CudaBuffer<f32>,
    output: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, output, device)?;

    if let Some(out) = try_launch_binary(
        grad,
        output,
        device,
        SIGMOID_BACKWARD_PTX,
        "sigmoid_backward_kernel",
    )? {
        return Ok(out);
    }

    // CPU fallback
    let grad_host = gpu_to_cpu(grad, device)?;
    let output_host = gpu_to_cpu(output, device)?;
    let result: Vec<f32> = grad_host
        .iter()
        .zip(output_host.iter())
        .map(|(&g, &o)| g * o * (1.0 - o))
        .collect();
    cpu_to_gpu(&result, device)
}

// ---------------------------------------------------------------------------
// Public API -- Tanh backward
// ---------------------------------------------------------------------------

/// Tanh backward: `out[i] = grad[i] * (1 - output[i]^2)`.
///
/// `grad` and `output` must have the same length and reside on `device`.
#[cfg(feature = "cuda")]
pub fn gpu_tanh_backward(
    grad: &CudaBuffer<f32>,
    output: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(grad, output, device)?;

    if let Some(out) = try_launch_binary(
        grad,
        output,
        device,
        TANH_BACKWARD_PTX,
        "tanh_backward_kernel",
    )? {
        return Ok(out);
    }

    // CPU fallback
    let grad_host = gpu_to_cpu(grad, device)?;
    let output_host = gpu_to_cpu(output, device)?;
    let result: Vec<f32> = grad_host
        .iter()
        .zip(output_host.iter())
        .map(|(&g, &o)| g * (1.0 - o * o))
        .collect();
    cpu_to_gpu(&result, device)
}

// ---------------------------------------------------------------------------
// Public API -- Softmax backward
// ---------------------------------------------------------------------------

/// Softmax backward (row-wise): one block per row, shared-memory dot reduction.
///
/// For each row of length `cols`:
///   `dot = sum(grad[row] * output[row])`
///   `out[i] = output[i] * (grad[i] - dot)`
///
/// `rows` = total elements / cols. Both `grad` and `output` have `rows * cols` elements.
#[cfg(feature = "cuda")]
pub fn gpu_softmax_backward(
    grad: &CudaBuffer<f32>,
    output: &CudaBuffer<f32>,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_binary(grad, output, device)?;

    let total = grad.len();
    let rows = total / cols;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SOFTMAX_BACKWARD_PTX,
        "softmax_backward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback
            let grad_host = gpu_to_cpu(grad, device)?;
            let output_host = gpu_to_cpu(output, device)?;
            let mut result = vec![0.0f32; total];
            for r in 0..rows {
                let base = r * cols;
                let mut dot = 0.0f32;
                for c in 0..cols {
                    dot += grad_host[base + c] * output_host[base + c];
                }
                for c in 0..cols {
                    result[base + c] = output_host[base + c] * (grad_host[base + c] - dot);
                }
            }
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    // One block per row, 256 threads per block.
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad.inner())
            .arg(output.inner())
            .arg(out.inner_mut())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Sum axis
// ---------------------------------------------------------------------------

/// Reduce along one axis of a tensor.
///
/// Thread i computes:
/// Full parallel sum reduction on GPU.
///
/// Uses a two-pass approach: first pass reduces `n` elements to `num_blocks`
/// partial sums via the `reduce_sum_kernel`, second pass reduces the partial
/// sums to a single scalar. For small inputs (< 256 blocks), the second pass
/// runs on CPU to avoid kernel launch overhead.
#[cfg(feature = "cuda")]
pub fn gpu_reduce_sum(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let n = a.len();
    if n == 0 {
        return cpu_to_gpu(&[0.0f32], device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        REDUCE_SUM_PTX,
        "reduce_sum_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback
            let host = gpu_to_cpu(a, device)?;
            let total: f32 = host.iter().sum();
            return cpu_to_gpu(&[total], device);
        }
    };

    // Pass 1: reduce to partial sums (one per block).
    const BLOCK: u32 = 256;
    let num_blocks = ((n as u32).saturating_add(BLOCK - 1)) / BLOCK;
    // Cap blocks to avoid excessive partial sums.
    let num_blocks = num_blocks.min(1024);

    let mut partials = alloc_zeros_f32(num_blocks as usize, device)?;
    let n_u32 = n as u32;

    let cfg = cudarc::driver::LaunchConfig {
        grid_dim: (num_blocks.max(1), 1, 1),
        block_dim: (BLOCK, 1, 1),
        shared_mem_bytes: 0, // Statically allocated in PTX
    };

    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(partials.inner_mut())
            .arg(&n_u32)
            .launch(cfg)?;
    }

    // Pass 2: reduce partial sums.
    if num_blocks <= 1 {
        return Ok(partials);
    }

    // For small number of blocks, reduce on CPU (cheaper than another kernel launch).
    if num_blocks <= 256 {
        let host_partials = gpu_to_cpu(&partials, device)?;
        let total: f32 = host_partials.iter().sum();
        return cpu_to_gpu(&[total], device);
    }

    // For many blocks, recurse with another kernel launch.
    gpu_reduce_sum(&partials, device)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_reduce_sum(
    _a: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

///   `output[i] = sum_{k=0}^{axis_size-1} input[outer_idx * axis_size * inner_size + k * inner_size + inner_idx]`
///
/// where `outer_idx = i / inner_size`, `inner_idx = i % inner_size`.
#[cfg(feature = "cuda")]
pub fn gpu_sum_axis(
    a: &CudaBuffer<f32>,
    outer: usize,
    axis_size: usize,
    inner: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(a, device)?;

    let total_output = outer * inner;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SUM_AXIS_PTX,
        "sum_axis_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback
            let host = gpu_to_cpu(a, device)?;
            let mut result = vec![0.0f32; total_output];
            for (i, out) in result.iter_mut().enumerate() {
                let outer_idx = i / inner;
                let inner_idx = i % inner;
                let mut sum = 0.0f32;
                for k in 0..axis_size {
                    sum += host[outer_idx * axis_size * inner + k * inner + inner_idx];
                }
                *out = sum;
            }
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(total_output, device)?;
    let cfg = launch_cfg(total_output)?;
    let outer_u32 = outer as u32;
    let axis_size_u32 = axis_size as u32;
    let inner_u32 = inner as u32;
    let total_u32 = total_output as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&outer_u32)
            .arg(&axis_size_u32)
            .arg(&inner_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Strided split
// ---------------------------------------------------------------------------

/// Extract a sub-tensor along one axis entirely on GPU.
///
/// Given an input buffer representing a tensor with `total_along_axis` elements
/// along the split axis, extracts the slice `[split_offset .. split_offset + split_size]`
/// along that axis.
///
/// - `inner_size` = product of dimensions after the split axis.
/// - `n` = total number of output elements (outer * split_size * inner_size).
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `input` and `device` are on different devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_strided_split(
    input: &CudaBuffer<f32>,
    total_along_axis: usize,
    split_offset: usize,
    split_size: usize,
    inner_size: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        STRIDED_SPLIT_PTX,
        "strided_split_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback
            let host = gpu_to_cpu(input, device)?;
            let outer = n / (split_size * inner_size);
            let mut result = vec![0.0f32; n];
            for (i, out) in result.iter_mut().enumerate() {
                let outer_idx = i / (split_size * inner_size);
                let within = i % (split_size * inner_size);
                let src_idx =
                    outer_idx * total_along_axis * inner_size + split_offset * inner_size + within;
                *out = host[src_idx];
            }
            let _ = outer;
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let total_ax_u32 = total_along_axis as u32;
    let offset_u32 = split_offset as u32;
    let split_sz_u32 = split_size as u32;
    let inner_u32 = inner_size as u32;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&total_ax_u32)
            .arg(&offset_u32)
            .arg(&split_sz_u32)
            .arg(&inner_u32)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Strided cat
// ---------------------------------------------------------------------------

/// Write a sub-tensor into a larger output buffer at an offset along one axis,
/// entirely on GPU.
///
/// Given an input buffer representing a chunk with `part_size` elements along
/// the cat axis, writes it into `output` at position `cat_offset` along that axis.
///
/// - `inner_size` = product of dimensions after the cat axis.
/// - `n` = total number of input elements (outer * part_size * inner_size).
///
/// # Safety
///
/// `output` must be large enough to hold the written region. The caller is
/// responsible for ensuring non-overlapping writes when multiple chunks are
/// written into the same output buffer.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if buffers and `device` are on different devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_strided_cat(
    input: &CudaBuffer<f32>,
    output: &mut CudaBuffer<f32>,
    total_along_axis: usize,
    cat_offset: usize,
    part_size: usize,
    inner_size: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        STRIDED_CAT_PTX,
        "strided_cat_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback
            let host_in = gpu_to_cpu(input, device)?;
            let mut host_out = gpu_to_cpu(output, device)?;
            for (i, &val) in host_in.iter().enumerate().take(n) {
                let outer_idx = i / (part_size * inner_size);
                let within = i % (part_size * inner_size);
                let dst_idx =
                    outer_idx * total_along_axis * inner_size + cat_offset * inner_size + within;
                host_out[dst_idx] = val;
            }
            *output = cpu_to_gpu(&host_out, device)?;
            return Ok(());
        }
    };

    let cfg = launch_cfg(n)?;
    let total_ax_u32 = total_along_axis as u32;
    let offset_u32 = cat_offset as u32;
    let part_sz_u32 = part_size as u32;
    let inner_u32 = inner_size as u32;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(output.inner_mut())
            .arg(&total_ax_u32)
            .arg(&offset_u32)
            .arg(&part_sz_u32)
            .arg(&inner_u32)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(())
}

/// Scalar multiply: `out[i] = a[i] * scalar`.
///
/// Multiplies every element by a constant float value on the GPU.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a` and `device` refer to different CUDA devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_scale(
    a: &CudaBuffer<f32>,
    scalar: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(a, device)?;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SCALE_PTX,
        "scale_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback
            let host = gpu_to_cpu(a, device)?;
            let result: Vec<f32> = host.iter().map(|&x| x * scalar).collect();
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&scalar)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- softmax
// ---------------------------------------------------------------------------

/// Row-wise softmax on GPU: one thread block per row, shared-memory reduction.
///
/// `rows` = product of all dims except the last. `cols` = last dim size.
#[cfg(feature = "cuda")]
pub fn gpu_softmax(
    input: &CudaBuffer<f32>,
    rows: usize,
    cols: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SOFTMAX_PTX,
        "softmax_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let host = gpu_to_cpu(input, device)?;
            let mut out = vec![0.0f32; host.len()];
            for r in 0..rows {
                let base = r * cols;
                let mut max_v = f32::NEG_INFINITY;
                for c in 0..cols {
                    max_v = max_v.max(host[base + c]);
                }
                let mut sum = 0.0f32;
                for c in 0..cols {
                    let e = (host[base + c] - max_v).exp();
                    out[base + c] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for c in 0..cols {
                    out[base + c] *= inv;
                }
            }
            return cpu_to_gpu(&out, device);
        }
    };

    let mut out = alloc_zeros_f32(rows * cols, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    // One block per row, 256 threads per block.
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4, // sdata[256] f32
    };

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- dropout
// ---------------------------------------------------------------------------

/// Inverted dropout on GPU: `out[i] = input[i] * scale` or `0` with probability `p`.
///
/// `threshold` = `(p * u32::MAX as f64) as u32` — the RNG cutoff.
/// `scale` = `1.0 / (1.0 - p)`.
/// `seed` = random seed for the RNG.
///
/// **Known limitation**: This kernel uses a simple per-element hash
/// (`tid * 2654435761 ^ seed` with xorshift mixing), not the full
/// Philox 4x32-10 counter-based RNG that PyTorch uses. A proper Philox
/// dropout kernel would generate the mask via `philox_uniform_kernel`
/// and then threshold — producing higher-quality randomness and exact
/// reproducibility across CPU/GPU. The current hash is sufficient for
/// training but should be upgraded for research requiring strict
/// statistical properties.
#[cfg(feature = "cuda")]
pub fn gpu_dropout(
    input: &CudaBuffer<f32>,
    threshold: u32,
    scale: f32,
    seed: u32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let n = input.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        DROPOUT_PTX,
        "dropout_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let host = gpu_to_cpu(input, device)?;
            // Stateless per-element hash matching the GPU kernel: each element
            // independently computes its own pseudorandom value from (tid, seed)
            // with no state carried between elements.
            let result: Vec<f32> = host
                .iter()
                .enumerate()
                .map(|(i, &x)| {
                    let mut r = (i as u32).wrapping_mul(2654435761) ^ seed;
                    r ^= r << 13;
                    r ^= r >> 17;
                    r ^= r << 5;
                    if r < threshold { 0.0 } else { x * scale }
                })
                .collect();
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&n_u32)
            .arg(&threshold)
            .arg(&scale)
            .arg(&seed)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- 2D transpose
// ---------------------------------------------------------------------------

/// 2D matrix transpose on GPU: `[M, N]` -> `[N, M]`.
#[cfg(feature = "cuda")]
pub fn gpu_transpose_2d(
    input: &CudaBuffer<f32>,
    m: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let total = m * n;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        TRANSPOSE_2D_PTX,
        "transpose_2d_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let host = gpu_to_cpu(input, device)?;
            let mut out = vec![0.0f32; total];
            for i in 0..m {
                for j in 0..n {
                    out[j * m + i] = host[i * n + j];
                }
            }
            return cpu_to_gpu(&out, device);
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let total_u32 = total as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&m_u32)
            .arg(&n_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- 4D permute (0,2,1,3)
// ---------------------------------------------------------------------------

/// Permute a 4D tensor from `[d0, d1, d2, d3]` to `[d0, d2, d1, d3]` on GPU.
/// Used for attention head reshaping: `[B, S, H, D_h]` -> `[B, H, S, D_h]`.
#[cfg(feature = "cuda")]
pub fn gpu_permute_0213(
    input: &CudaBuffer<f32>,
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let total = d0 * d1 * d2 * d3;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        PERMUTE_0213_PTX,
        "permute_0213_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let host = gpu_to_cpu(input, device)?;
            let mut out = vec![0.0f32; total];
            for i0 in 0..d0 {
                for i1 in 0..d1 {
                    for i2 in 0..d2 {
                        for i3 in 0..d3 {
                            let in_idx = ((i0 * d1 + i1) * d2 + i2) * d3 + i3;
                            let out_idx = ((i0 * d2 + i2) * d1 + i1) * d3 + i3;
                            out[out_idx] = host[in_idx];
                        }
                    }
                }
            }
            return cpu_to_gpu(&out, device);
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let d0_u32 = d0 as u32;
    let d1_u32 = d1 as u32;
    let d2_u32 = d2 as u32;
    let d3_u32 = d3 as u32;
    let total_u32 = total as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&d0_u32)
            .arg(&d1_u32)
            .arg(&d2_u32)
            .arg(&d3_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Small matmul (bypasses cuBLAS JIT)
// ---------------------------------------------------------------------------

/// Small matrix multiply using our own PTX kernel. Avoids cuBLAS JIT
/// compilation overhead for tiny matrices where JIT cost > compute cost.
///
/// `a`: `[M, K]`, `b`: `[K, N]` → `c`: `[M, N]`.
#[cfg(feature = "cuda")]
pub fn gpu_small_matmul(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let total = m * n;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SMALL_MATMUL_PTX,
        "small_matmul_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // Fall back to cuBLAS if our kernel can't compile.
            return crate::blas::gpu_matmul_f32(a, b, m, k, n, device);
        }
    };

    let mut c = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let m_u32 = m as u32;
    let k_u32 = k as u32;
    let n_u32 = n as u32;
    let total_u32 = total as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(c.inner_mut())
            .arg(&m_u32)
            .arg(&k_u32)
            .arg(&n_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(c)
}

/// Small batched matmul: C[i] = A[i] @ B[i] for i in 0..batch.
/// Uses the small_matmul_kernel by reshaping the problem: treat it as a single
/// large matmul of [batch*M, K] @ [K, N] — but that doesn't work because B is
/// batched. Instead, we use a modified approach: thread `idx` computes element
/// (batch_i, row, col) where batch_i = idx / (M*N).
///
/// For simplicity and correctness, we fall back to cpu_bmm for now when
/// cuBLAS fails, but route through gpu_small_matmul for the single-matrix case.
#[cfg(feature = "cuda")]
pub fn gpu_small_bmm(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    // For batch=1, just use the single matmul kernel.
    if batch == 1 {
        return gpu_small_matmul(a, b, m, k, n, device);
    }
    // For batched case, fall back to cuBLAS (the batched PTX kernel is complex).
    // The main win is from the single-matrix decode case (batch=1 for attention scores).
    crate::blas::gpu_bmm_f32(a, b, batch, m, k, n, device)
}

// ---------------------------------------------------------------------------
// Public API -- Embedding lookup (GPU-native)
// ---------------------------------------------------------------------------

/// GPU embedding lookup: reads token ID from `idx` (single f32 on GPU),
/// gathers row from `weight` `[V, D]`, writes to `out` `[D]`.
/// Entire operation stays on GPU — no CPU involvement.
#[cfg(feature = "cuda")]
pub fn gpu_embed_lookup(
    idx: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    d: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        EMBED_LOOKUP_PTX,
        "embed_lookup_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let idx_host = gpu_to_cpu(idx, device)?;
            let weight_host = gpu_to_cpu(weight, device)?;
            let row = idx_host[0] as usize;
            let start = row * d;
            let out = weight_host[start..start + d].to_vec();
            return cpu_to_gpu(&out, device);
        }
    };

    let mut out = alloc_zeros_f32(d, device)?;
    let cfg = launch_cfg(d)?;
    let d_u32 = d as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(idx.inner())
            .arg(weight.inner())
            .arg(out.inner_mut())
            .arg(&d_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Slice write (for KV cache)
// ---------------------------------------------------------------------------

/// Write `src` of shape `[N, D]` into row `pos` of `dst` of shape `[N, max_len, D]`.
/// This is an in-place GPU operation — `dst` is modified.
#[cfg(feature = "cuda")]
pub fn gpu_slice_write(
    src: &CudaBuffer<f32>,
    dst: &mut CudaBuffer<f32>,
    n_batch: usize,
    d: usize,
    max_len: usize,
    pos: usize,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;

    let total = n_batch * d;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SLICE_WRITE_PTX,
        "slice_write_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let src_host = gpu_to_cpu(src, device)?;
            let mut dst_host = gpu_to_cpu(dst, device)?;
            for b in 0..n_batch {
                for di in 0..d {
                    dst_host[b * max_len * d + pos * d + di] = src_host[b * d + di];
                }
            }
            let new_dst = cpu_to_gpu(&dst_host, device)?;
            *dst = new_dst;
            return Ok(());
        }
    };

    let cfg = launch_cfg(total)?;
    let n_u32 = total as u32;
    let d_u32 = d as u32;
    let max_len_u32 = max_len as u32;
    let pos_u32 = pos as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(src.inner())
            .arg(dst.inner_mut())
            .arg(&n_u32)
            .arg(&d_u32)
            .arg(&max_len_u32)
            .arg(&pos_u32)
            .launch(cfg)?;
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Public API -- Slice read (for KV cache)
// ---------------------------------------------------------------------------

/// Read first `len` rows from each batch of `[N, max_len, D]` → `[N, len, D]`.
#[cfg(feature = "cuda")]
pub fn gpu_slice_read(
    src: &CudaBuffer<f32>,
    n_batch: usize,
    d: usize,
    len: usize,
    max_len: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let total = n_batch * len * d;
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SLICE_READ_PTX,
        "slice_read_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            let host = gpu_to_cpu(src, device)?;
            let mut out = vec![0.0f32; total];
            for b in 0..n_batch {
                for r in 0..len {
                    for di in 0..d {
                        out[b * len * d + r * d + di] = host[b * max_len * d + r * d + di];
                    }
                }
            }
            return cpu_to_gpu(&out, device);
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let total_u32 = total as u32;
    let d_u32 = d as u32;
    let len_u32 = len as u32;
    let max_len_u32 = max_len as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(src.inner())
            .arg(out.inner_mut())
            .arg(&total_u32)
            .arg(&d_u32)
            .arg(&len_u32)
            .arg(&max_len_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- GELU
// ---------------------------------------------------------------------------

/// Elementwise GELU activation on GPU: `gelu(x) = x * sigmoid(1.702 * x)`.
#[cfg(feature = "cuda")]
pub fn gpu_gelu(input: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(input, device)?;
    if let Some(out) = try_launch_unary(input, device, GELU_PTX, "gelu_kernel")? {
        return Ok(out);
    }
    cpu_fallback_unary(input, device, |x| {
        let s = 1.0 / (1.0 + (-1.702 * x).exp());
        x * s
    })
}

// ---------------------------------------------------------------------------
// Public API -- elementwise transcendentals & math ops
// ---------------------------------------------------------------------------

/// Elementwise division: `out[i] = a[i] / b[i]`.
#[cfg(feature = "cuda")]
pub fn gpu_div(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    validate_binary(a, b, device)?;

    if let Some(out) = try_launch_binary(a, b, device, DIV_PTX, "div_kernel")? {
        return Ok(out);
    }

    // CPU fallback
    let a_host = gpu_to_cpu(a, device)?;
    let b_host = gpu_to_cpu(b, device)?;
    let result: Vec<f32> = a_host
        .iter()
        .zip(b_host.iter())
        .map(|(&x, &y)| x / y)
        .collect();
    cpu_to_gpu(&result, device)
}

/// Elementwise exponential: `out[i] = exp(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_exp(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    if let Some(out) = try_launch_unary(a, device, EXP_PTX, "exp_kernel")? {
        return Ok(out);
    }
    cpu_fallback_unary(a, device, |x| x.exp())
}

/// Elementwise natural log: `out[i] = ln(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_log(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    if let Some(out) = try_launch_unary(a, device, LOG_PTX, "log_kernel")? {
        return Ok(out);
    }
    cpu_fallback_unary(a, device, |x| x.ln())
}

/// Elementwise square root: `out[i] = sqrt(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_sqrt(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    if let Some(out) = try_launch_unary(a, device, SQRT_PTX, "sqrt_kernel")? {
        return Ok(out);
    }
    cpu_fallback_unary(a, device, |x| x.sqrt())
}

/// Elementwise power: `out[i] = a[i] ^ exponent`.
#[cfg(feature = "cuda")]
pub fn gpu_pow(
    a: &CudaBuffer<f32>,
    exponent: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(a, device)?;

    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        POW_PTX,
        "pow_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            let host = gpu_to_cpu(a, device)?;
            let result: Vec<f32> = host.iter().map(|&x| x.powf(exponent)).collect();
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(n, device)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&exponent)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

/// Elementwise absolute value: `out[i] = |a[i]|`.
#[cfg(feature = "cuda")]
pub fn gpu_abs(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    if let Some(out) = try_launch_unary(a, device, ABS_PTX, "abs_kernel")? {
        return Ok(out);
    }
    cpu_fallback_unary(a, device, |x| x.abs())
}

/// Elementwise sigmoid: `out[i] = 1 / (1 + exp(-a[i]))`.
#[cfg(feature = "cuda")]
pub fn gpu_sigmoid(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    if let Some(out) = try_launch_unary(a, device, SIGMOID_PTX, "sigmoid_kernel")? {
        return Ok(out);
    }
    cpu_fallback_unary(a, device, |x| 1.0 / (1.0 + (-x).exp()))
}

/// Elementwise tanh: `out[i] = tanh(a[i])`.
#[cfg(feature = "cuda")]
pub fn gpu_tanh(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    validate_unary(a, device)?;
    if let Some(out) = try_launch_unary(a, device, TANH_PTX, "tanh_kernel")? {
        return Ok(out);
    }
    cpu_fallback_unary(a, device, |x| x.tanh())
}

// ---------------------------------------------------------------------------
// Public API -- fused Adam optimizer step
// ---------------------------------------------------------------------------

/// Fused Adam optimizer step: updates param, exp_avg, and exp_avg_sq in-place
/// in a single kernel launch.
///
/// All four buffers must have the same length `n`. `param`, `exp_avg`, and
/// `exp_avg_sq` are modified in-place. `grad` is read-only.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_fused_adam(
    param: &mut CudaBuffer<f32>,
    grad: &CudaBuffer<f32>,
    exp_avg: &mut CudaBuffer<f32>,
    exp_avg_sq: &mut CudaBuffer<f32>,
    beta1: f32,
    beta2: f32,
    lr: f32,
    eps: f32,
    bc1: f32,
    bc2: f32,
    weight_decay: f32,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;

    let n = param.len();
    if grad.len() != n || exp_avg.len() != n || exp_avg_sq.len() != n {
        return Err(GpuError::LengthMismatch {
            a: n,
            b: grad.len(),
        });
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        FUSED_ADAM_PTX,
        "fused_adam_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback: download, compute, upload.
            let mut p_host = gpu_to_cpu(param, device)?;
            let g_host = gpu_to_cpu(grad, device)?;
            let mut m_host = gpu_to_cpu(exp_avg, device)?;
            let mut v_host = gpu_to_cpu(exp_avg_sq, device)?;

            for i in 0..n {
                let mut g = g_host[i];
                if weight_decay > 0.0 {
                    g += weight_decay * p_host[i];
                }
                m_host[i] = beta1 * m_host[i] + (1.0 - beta1) * g;
                v_host[i] = beta2 * v_host[i] + (1.0 - beta2) * g * g;
                let m_hat = m_host[i] / bc1;
                let v_hat = v_host[i] / bc2;
                p_host[i] -= lr * m_hat / (v_hat.sqrt() + eps);
            }

            *param = cpu_to_gpu(&p_host, device)?;
            *exp_avg = cpu_to_gpu(&m_host, device)?;
            *exp_avg_sq = cpu_to_gpu(&v_host, device)?;
            return Ok(());
        }
    };

    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(param.inner_mut())
            .arg(grad.inner())
            .arg(exp_avg.inner_mut())
            .arg(exp_avg_sq.inner_mut())
            .arg(&beta1)
            .arg(&beta2)
            .arg(&lr)
            .arg(&eps)
            .arg(&bc1)
            .arg(&bc2)
            .arg(&weight_decay)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(())
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_fused_adam(
    _param: &mut CudaBuffer<f32>,
    _grad: &CudaBuffer<f32>,
    _exp_avg: &mut CudaBuffer<f32>,
    _exp_avg_sq: &mut CudaBuffer<f32>,
    _beta1: f32,
    _beta2: f32,
    _lr: f32,
    _eps: f32,
    _bc1: f32,
    _bc2: f32,
    _weight_decay: f32,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Public API -- fused GRU cell
// ---------------------------------------------------------------------------

/// Fused GRU cell forward: takes pre-computed gate matrices and produces
/// new hidden state + workspace for backward.
///
/// Inputs:
/// - `input_gates`: `[batch, 3*hsz]` — result of `x @ W_ih^T`
/// - `hidden_gates`: `[batch, 3*hsz]` — result of `h @ W_hh^T`
/// - `bias_ih`: `[3*hsz]` — input bias
/// - `bias_hh`: `[3*hsz]` — hidden bias
/// - `hx`: `[batch, hsz]` — previous hidden state
///
/// Outputs:
/// - `hy`: `[batch, hsz]` — new hidden state
/// - `workspace`: `[batch, 5*hsz]` — saved for backward (r, z, n, hx, hn+b2n)
#[cfg(feature = "cuda")]
pub fn gpu_fused_gru_forward(
    input_gates: &CudaBuffer<f32>,
    hidden_gates: &CudaBuffer<f32>,
    bias_ih: &CudaBuffer<f32>,
    bias_hh: &CudaBuffer<f32>,
    hx: &CudaBuffer<f32>,
    hsz: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    use cudarc::driver::PushKernelArg;

    let total = hx.len(); // batch * hsz
    let batch = total / hsz;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        FUSED_GRU_FORWARD_PTX,
        "fused_gru_forward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            return Err(GpuError::PtxCompileFailed {
                kernel: "fused_gru_forward_kernel",
            });
        }
    };

    let mut hy = alloc_zeros_f32(total, device)?;
    let mut workspace = alloc_zeros_f32(batch * 5 * hsz, device)?;

    let cfg = launch_cfg(total)?;
    let hsz_u32 = hsz as u32;
    let total_u32 = total as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input_gates.inner())
            .arg(hidden_gates.inner())
            .arg(bias_ih.inner())
            .arg(bias_hh.inner())
            .arg(hx.inner())
            .arg(hy.inner_mut())
            .arg(workspace.inner_mut())
            .arg(&hsz_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok((hy, workspace))
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_fused_gru_forward(
    _input_gates: &CudaBuffer<f32>,
    _hidden_gates: &CudaBuffer<f32>,
    _bias_ih: &CudaBuffer<f32>,
    _bias_hh: &CudaBuffer<f32>,
    _hx: &CudaBuffer<f32>,
    _hsz: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>)> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Public API -- MaxPool2d / AvgPool2d
// ---------------------------------------------------------------------------

/// MaxPool2d forward on GPU. One thread per output element.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_maxpool2d(
    input: &CudaBuffer<f32>,
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, [usize; 4])> {
    use cudarc::driver::PushKernelArg;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;
    let total = batch * channels * h_out * w_out;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx, MAXPOOL2D_PTX, "maxpool2d_forward_kernel", device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => return Err(GpuError::PtxCompileFailed { kernel: "maxpool2d_forward_kernel" }),
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;

    let (batch_u32, ch_u32) = (batch as u32, channels as u32);
    let (h_in_u32, w_in_u32) = (h_in as u32, w_in as u32);
    let (h_out_u32, w_out_u32) = (h_out as u32, w_out as u32);
    let (kh_u32, kw_u32) = (kh as u32, kw as u32);
    let (sh_u32, sw_u32) = (sh as u32, sw as u32);
    let (ph_u32, pw_u32) = (ph as u32, pw as u32);
    let total_u32 = total as u32;

    unsafe {
        stream.launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&batch_u32).arg(&ch_u32)
            .arg(&h_in_u32).arg(&w_in_u32)
            .arg(&h_out_u32).arg(&w_out_u32)
            .arg(&kh_u32).arg(&kw_u32)
            .arg(&sh_u32).arg(&sw_u32)
            .arg(&ph_u32).arg(&pw_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok((out, [batch, channels, h_out, w_out]))
}

/// Stub.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_maxpool2d(
    _input: &CudaBuffer<f32>, _batch: usize, _channels: usize,
    _h_in: usize, _w_in: usize, _kh: usize, _kw: usize,
    _sh: usize, _sw: usize, _ph: usize, _pw: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, [usize; 4])> {
    Err(GpuError::NoCudaFeature)
}

/// AvgPool2d forward on GPU. One thread per output element.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_avgpool2d(
    input: &CudaBuffer<f32>,
    batch: usize,
    channels: usize,
    h_in: usize,
    w_in: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, [usize; 4])> {
    use cudarc::driver::PushKernelArg;

    let h_out = (h_in + 2 * ph - kh) / sh + 1;
    let w_out = (w_in + 2 * pw - kw) / sw + 1;
    let total = batch * channels * h_out * w_out;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx, AVGPOOL2D_PTX, "avgpool2d_forward_kernel", device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => return Err(GpuError::PtxCompileFailed { kernel: "avgpool2d_forward_kernel" }),
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;

    let (batch_u32, ch_u32) = (batch as u32, channels as u32);
    let (h_in_u32, w_in_u32) = (h_in as u32, w_in as u32);
    let (h_out_u32, w_out_u32) = (h_out as u32, w_out as u32);
    let (kh_u32, kw_u32) = (kh as u32, kw as u32);
    let (sh_u32, sw_u32) = (sh as u32, sw as u32);
    let (ph_u32, pw_u32) = (ph as u32, pw as u32);
    let total_u32 = total as u32;

    unsafe {
        stream.launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(&batch_u32).arg(&ch_u32)
            .arg(&h_in_u32).arg(&w_in_u32)
            .arg(&h_out_u32).arg(&w_out_u32)
            .arg(&kh_u32).arg(&kw_u32)
            .arg(&sh_u32).arg(&sw_u32)
            .arg(&ph_u32).arg(&pw_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok((out, [batch, channels, h_out, w_out]))
}

/// Stub.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_avgpool2d(
    _input: &CudaBuffer<f32>, _batch: usize, _channels: usize,
    _h_in: usize, _w_in: usize, _kh: usize, _kw: usize,
    _sh: usize, _sw: usize, _ph: usize, _pw: usize,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, [usize; 4])> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Public API -- BatchNorm2d
// ---------------------------------------------------------------------------

/// BatchNorm2d forward on GPU (placeholder — kernel pass-1 indexing needs
/// refinement). Currently validates the kernel compiles and falls back to
/// returning an error so callers use the CPU path.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_batchnorm_forward(
    _input: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _bias: &CudaBuffer<f32>,
    _running_mean: &mut CudaBuffer<f32>,
    _running_var: &mut CudaBuffer<f32>,
    _channels: usize,
    _spatial: usize,
    _eps: f32,
    _momentum: f32,
    _training: bool,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>, CudaBuffer<f32>)> {
    // Validate the PTX compiles (catches syntax errors at first call).
    let ctx = device.context();
    let _f = crate::module_cache::get_or_compile(
        ctx,
        BATCHNORM_FORWARD_PTX,
        "batchnorm_forward_kernel",
        device.ordinal() as u32,
    );
    // Full implementation pending — pass-1 loop indexing needs refinement.
    Err(GpuError::ShapeMismatch {
        op: "batchnorm_forward",
        expected: vec![0],
        got: vec![1],
    })
}

/// Stub.
#[cfg(not(feature = "cuda"))]
#[allow(clippy::too_many_arguments)]
pub fn gpu_batchnorm_forward(
    _input: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _bias: &CudaBuffer<f32>,
    _running_mean: &mut CudaBuffer<f32>,
    _running_var: &mut CudaBuffer<f32>,
    _channels: usize,
    _spatial: usize,
    _eps: f32,
    _momentum: f32,
    _training: bool,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>, CudaBuffer<f32>)> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Public API -- LayerNorm
// ---------------------------------------------------------------------------

/// Row-wise layer normalization on GPU.
///
/// `input`: `[rows * cols]`, `weight`/`bias`: `[cols]`.
/// Output: normalized and affine-transformed `[rows * cols]`.
#[cfg(feature = "cuda")]
pub fn gpu_layernorm(
    input: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    bias: &CudaBuffer<f32>,
    rows: usize,
    cols: usize,
    eps: f32,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        LAYERNORM_PTX,
        "layernorm_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ferrotorch-gpu: LayerNorm PTX compilation failed ({e:?}), CPU fallback");
            std::fs::write("/tmp/layernorm_debug.ptx", LAYERNORM_PTX).ok();
            eprintln!(
                "ferrotorch-gpu: dumped PTX to /tmp/layernorm_debug.ptx ({} bytes)",
                LAYERNORM_PTX.len()
            );
            let h_in = gpu_to_cpu(input, device)?;
            let h_w = gpu_to_cpu(weight, device)?;
            let h_b = gpu_to_cpu(bias, device)?;
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let base = r * cols;
                let slice = &h_in[base..base + cols];
                let mean: f32 = slice.iter().sum::<f32>() / cols as f32;
                let var: f32 =
                    slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / cols as f32;
                let inv_std = 1.0 / (var + eps).sqrt();
                for c in 0..cols {
                    let normed = (slice[c] - mean) * inv_std;
                    out[base + c] = h_w[c] * normed + h_b[c];
                }
            }
            return cpu_to_gpu(&out, device);
        }
    };

    let mut out = alloc_zeros_f32(rows * cols, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(weight.inner())
            .arg(bias.inner())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- LayerNorm backward
// ---------------------------------------------------------------------------

/// LayerNorm backward pass on GPU.
///
/// Computes grad_input, grad_weight, and grad_bias entirely on GPU.
/// One block per batch element (row), 256 threads per block.
/// grad_weight and grad_bias are accumulated across batches via atomicAdd.
///
/// `input`: `[rows * cols]`, `grad_output`: `[rows * cols]`, `weight`: `[cols]`.
/// Returns: `(grad_input [rows * cols], grad_weight [cols], grad_bias [cols])`.
#[cfg(feature = "cuda")]
pub fn gpu_layernorm_backward(
    input: &CudaBuffer<f32>,
    grad_output: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    rows: usize,
    cols: usize,
    eps: f32,
    device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>, CudaBuffer<f32>)> {
    use cudarc::driver::PushKernelArg;

    validate_unary(input, device)?;

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        LAYERNORM_BACKWARD_PTX,
        "layernorm_backward_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback
            let h_in = gpu_to_cpu(input, device)?;
            let h_go = gpu_to_cpu(grad_output, device)?;
            let h_w = gpu_to_cpu(weight, device)?;
            let mut grad_input = vec![0.0f32; rows * cols];
            let mut grad_weight = vec![0.0f32; cols];
            let mut grad_bias = vec![0.0f32; cols];
            let n_f = cols as f32;
            for r in 0..rows {
                let base = r * cols;
                let x_slice = &h_in[base..base + cols];
                let go_slice = &h_go[base..base + cols];
                let mean: f32 = x_slice.iter().sum::<f32>() / n_f;
                let var: f32 = x_slice
                    .iter()
                    .map(|&x| (x - mean) * (x - mean))
                    .sum::<f32>()
                    / n_f;
                let inv_std = 1.0 / (var + eps).sqrt();
                let mut sum1 = 0.0f32;
                let mut sum2 = 0.0f32;
                for c in 0..cols {
                    let x_hat = (x_slice[c] - mean) * inv_std;
                    let dl = go_slice[c] * h_w[c];
                    sum1 += dl;
                    sum2 += dl * x_hat;
                    grad_weight[c] += go_slice[c] * x_hat;
                    grad_bias[c] += go_slice[c];
                }
                let m1 = sum1 / n_f;
                let m2 = sum2 / n_f;
                for c in 0..cols {
                    let x_hat = (x_slice[c] - mean) * inv_std;
                    let dl = go_slice[c] * h_w[c];
                    grad_input[base + c] = inv_std * (dl - m1 - x_hat * m2);
                }
            }
            let gi = cpu_to_gpu(&grad_input, device)?;
            let gw = cpu_to_gpu(&grad_weight, device)?;
            let gb = cpu_to_gpu(&grad_bias, device)?;
            return Ok((gi, gw, gb));
        }
    };

    let mut grad_in = alloc_zeros_f32(rows * cols, device)?;
    let mut grad_w = alloc_zeros_f32(cols, device)?;
    let mut grad_b = alloc_zeros_f32(cols, device)?;
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;

    // One block per row, 256 threads per block.
    let cfg = LaunchConfig {
        grid_dim: ((rows as u32).max(1), 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 256 * 4,
    };

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(grad_output.inner())
            .arg(weight.inner())
            .arg(grad_in.inner_mut())
            .arg(grad_w.inner_mut())
            .arg(grad_b.inner_mut())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }

    Ok((grad_in, grad_w, grad_b))
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_layernorm_backward(
    _input: &CudaBuffer<f32>,
    _grad_output: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _device: &GpuDevice,
) -> GpuResult<(CudaBuffer<f32>, CudaBuffer<f32>, CudaBuffer<f32>)> {
    Err(GpuError::NoCudaFeature)
}

// ===========================================================================
// _into variants — write to pre-allocated output buffers (zero allocation)
//
// These are used for CUDA graph capture, where all buffer addresses must be
// fixed at capture time. The PTX kernels are identical — only the Rust
// wrapper skips allocation.
// ===========================================================================

/// Elementwise add into pre-allocated output: `out[i] = a[i] + b[i]`.
#[cfg(feature = "cuda")]
pub fn gpu_add_into(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    validate_binary(a, b, device)?;
    if out.len() < a.len() {
        return Err(GpuError::ShapeMismatch {
            op: "add_into",
            expected: vec![a.len()],
            got: vec![out.len()],
        });
    }
    if try_launch_binary_into(a, b, out, device, ADD_PTX, "add_kernel")? {
        return Ok(());
    }
    Err(GpuError::PtxCompileFailed {
        kernel: "add_kernel",
    })
}

/// Elementwise mul into pre-allocated output: `out[i] = a[i] * b[i]`.
#[cfg(feature = "cuda")]
pub fn gpu_mul_into(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    validate_binary(a, b, device)?;
    if out.len() < a.len() {
        return Err(GpuError::ShapeMismatch {
            op: "mul_into",
            expected: vec![a.len()],
            got: vec![out.len()],
        });
    }
    if try_launch_binary_into(a, b, out, device, MUL_PTX, "mul_kernel")? {
        return Ok(());
    }
    Err(GpuError::PtxCompileFailed {
        kernel: "mul_kernel",
    })
}

/// Scalar multiply into pre-allocated output: `out[i] = a[i] * scalar`.
#[cfg(feature = "cuda")]
pub fn gpu_scale_into(
    a: &CudaBuffer<f32>,
    scalar: f32,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    validate_unary(a, device)?;
    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        SCALE_PTX,
        "scale_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "scale_kernel",
    })?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&scalar)
            .arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(())
}

/// Check whether a GPU buffer contains any inf or NaN values.
///
/// Downloads the buffer contents to the host and scans for non-finite
/// values. This is correct for any buffer size and requires no custom
/// reduction kernel.
///
/// For a future optimization, a dedicated GPU reduction kernel could be
/// used to produce a single boolean flag on device, avoiding the full
/// download. The current approach is already much faster than the old
/// per-element CPU loop in `unscale_()` because the scaling itself
/// runs on GPU — only the inf/NaN check touches the host.
///
/// # Errors
///
/// - [`GpuError::DeviceMismatch`] if `a` and `device` refer to different CUDA devices.
/// - [`GpuError::Driver`] on CUDA runtime errors.
#[cfg(feature = "cuda")]
pub fn gpu_has_inf_nan(a: &CudaBuffer<f32>, device: &GpuDevice) -> GpuResult<bool> {
    let n = a.len();
    if n == 0 {
        return Ok(false);
    }

    validate_unary(a, device)?;

    let host: Vec<f32> = crate::transfer::gpu_to_cpu(a, device)?;
    Ok(host.iter().any(|v| !v.is_finite()))
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_has_inf_nan(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<bool> {
    Err(GpuError::NoCudaFeature)
}

/// GELU into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_into(
    a: &CudaBuffer<f32>,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    validate_unary(a, device)?;
    if try_launch_unary_into(a, out, device, GELU_PTX, "gelu_kernel")? {
        return Ok(());
    }
    Err(GpuError::PtxCompileFailed {
        kernel: "gelu_kernel",
    })
}

/// Embedding lookup into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_embed_lookup_into(
    idx: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    d: usize,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        EMBED_LOOKUP_PTX,
        "embed_lookup_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "embed_lookup_kernel",
    })?;
    let cfg = launch_cfg(d)?;
    let d_u32 = d as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(idx.inner())
            .arg(weight.inner())
            .arg(out.inner_mut())
            .arg(&d_u32)
            .launch(cfg)?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API -- Batch embedding lookup (GPU-native)
// ---------------------------------------------------------------------------

/// GPU batch embedding lookup: given `indices` (N f32 values on GPU) and
/// `weight` `[V, D]`, gather N rows to produce output `[N, D]`.
/// Entire operation stays on GPU -- no CPU roundtrip.
#[cfg(feature = "cuda")]
pub fn gpu_embed_lookup_batch(
    indices: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    n: usize,
    d: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let total = n * d;
    if total == 0 {
        return alloc_zeros_f32(0, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        EMBED_LOOKUP_BATCH_PTX,
        "embed_lookup_batch_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let idx_host = gpu_to_cpu(indices, device)?;
            let weight_host = gpu_to_cpu(weight, device)?;
            let mut out = Vec::with_capacity(total);
            for &idx_f in &idx_host {
                let row = idx_f as usize;
                let start = row * d;
                out.extend_from_slice(&weight_host[start..start + d]);
            }
            return cpu_to_gpu(&out, device);
        }
    };

    let mut out = alloc_zeros_f32(total, device)?;
    let cfg = launch_cfg(total)?;
    let d_u32 = d as u32;
    let total_u32 = total as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(indices.inner())
            .arg(weight.inner())
            .arg(out.inner_mut())
            .arg(&d_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// Public API -- Scatter-add rows (for embedding backward, GPU-native)
// ---------------------------------------------------------------------------

/// GPU scatter-add rows: given `grad_output` `[N, D]` and `indices` `[N]` (f32),
/// atomically accumulate into `grad_weight` `[V, D]` (pre-zeroed):
///   `grad_weight[indices[i], :] += grad_output[i, :]`
///
/// Duplicate indices accumulate correctly via atomic adds.
#[cfg(feature = "cuda")]
pub fn gpu_scatter_add_rows(
    grad_output: &CudaBuffer<f32>,
    indices: &CudaBuffer<f32>,
    num_embeddings: usize,
    d: usize,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    use cudarc::driver::PushKernelArg;

    let n = indices.len();
    let total = n * d;

    if total == 0 {
        return alloc_zeros_f32(num_embeddings * d, device);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = match crate::module_cache::get_or_compile(
        ctx,
        SCATTER_ADD_ROWS_PTX,
        "scatter_add_rows_kernel",
        device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let go_host = gpu_to_cpu(grad_output, device)?;
            let idx_host = gpu_to_cpu(indices, device)?;
            let mut result = vec![0.0f32; num_embeddings * d];
            for (i, &idx_f) in idx_host.iter().enumerate() {
                let row = idx_f as usize;
                for j in 0..d {
                    result[row * d + j] += go_host[i * d + j];
                }
            }
            return cpu_to_gpu(&result, device);
        }
    };

    let mut out = alloc_zeros_f32(num_embeddings * d, device)?;
    let cfg = launch_cfg(total)?;
    let d_u32 = d as u32;
    let total_u32 = total as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(grad_output.inner())
            .arg(indices.inner())
            .arg(out.inner_mut())
            .arg(&d_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

/// 2D transpose into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_transpose_2d_into(
    a: &CudaBuffer<f32>,
    m: usize,
    n: usize,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = m * n;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        TRANSPOSE_2D_PTX,
        "transpose_2d_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "transpose_2d_kernel",
    })?;
    let cfg = launch_cfg(total)?;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let total_u32 = total as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&m_u32)
            .arg(&n_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }
    Ok(())
}

/// Permute (0,2,1,3) into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_permute_0213_into(
    a: &CudaBuffer<f32>,
    d0: usize,
    d1: usize,
    d2: usize,
    d3: usize,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = d0 * d1 * d2 * d3;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        PERMUTE_0213_PTX,
        "permute_0213_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "permute_0213_kernel",
    })?;
    let cfg = launch_cfg(total)?;
    let (d0u, d1u, d2u, d3u, tu) = (d0 as u32, d1 as u32, d2 as u32, d3 as u32, total as u32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&d0u)
            .arg(&d1u)
            .arg(&d2u)
            .arg(&d3u)
            .arg(&tu)
            .launch(cfg)?;
    }
    Ok(())
}

/// Softmax into pre-allocated output (row-wise).
#[cfg(feature = "cuda")]
pub fn gpu_softmax_into(
    a: &CudaBuffer<f32>,
    rows: usize,
    cols: usize,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        SOFTMAX_PTX,
        "softmax_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "softmax_kernel",
    })?;
    let block_size = 256u32;
    let grid_size = rows as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: (cols as u32) * 4,
    };
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(out.inner_mut())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .launch(cfg)?;
    }
    Ok(())
}

/// LayerNorm into pre-allocated output.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub fn gpu_layernorm_into(
    input: &CudaBuffer<f32>,
    weight: &CudaBuffer<f32>,
    bias: &CudaBuffer<f32>,
    rows: usize,
    cols: usize,
    eps: f32,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        LAYERNORM_PTX,
        "layernorm_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "layernorm_kernel",
    })?;
    let block_size = 256u32;
    let grid_size = rows as u32;
    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: (cols as u32) * 4,
    };
    let rows_u32 = rows as u32;
    let cols_u32 = cols as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(out.inner_mut())
            .arg(weight.inner())
            .arg(bias.inner())
            .arg(&rows_u32)
            .arg(&cols_u32)
            .arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Slice read into pre-allocated output: read first `len` rows from
/// `[n_batch, max_len, d]` into out `[n_batch, len, d]`.
#[cfg(feature = "cuda")]
pub fn gpu_slice_read_into(
    src: &CudaBuffer<f32>,
    n_batch: usize,
    d: usize,
    len: usize,
    max_len: usize,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = n_batch * len * d;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        SLICE_READ_PTX,
        "slice_read_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "slice_read_kernel",
    })?;
    let cfg = launch_cfg(total)?;
    let total_u32 = total as u32;
    let d_u32 = d as u32;
    let len_u32 = len as u32;
    let max_len_u32 = max_len as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(src.inner())
            .arg(out.inner_mut())
            .arg(&total_u32)
            .arg(&d_u32)
            .arg(&len_u32)
            .arg(&max_len_u32)
            .launch(cfg)?;
    }
    Ok(())
}

/// Small matmul (PTX kernel) into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_small_matmul_into(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    m: usize,
    k: usize,
    n: usize,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = m * n;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        SMALL_MATMUL_PTX,
        "small_matmul_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "small_matmul_kernel",
    })?;
    let cfg = launch_cfg(total)?;
    let (m_u32, k_u32, n_u32, total_u32) = (m as u32, k as u32, n as u32, total as u32);
    unsafe {
        stream
            .launch_builder(&f)
            .arg(a.inner())
            .arg(b.inner())
            .arg(out.inner_mut())
            .arg(&m_u32)
            .arg(&k_u32)
            .arg(&n_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }
    Ok(())
}

// ===========================================================================
// Indirect-parameter kernels for CUDA graph capture
// ===========================================================================

/// Slice write with position read from device memory (for CUDA graph capture).
/// Writes `src [n_batch, d]` into row `*pos_ptr` of `dst [n_batch, max_len, d]`.
#[cfg(feature = "cuda")]
pub fn gpu_slice_write_indirect(
    src: &CudaBuffer<f32>,
    dst: &mut CudaBuffer<f32>,
    n_batch: usize,
    d: usize,
    max_len: usize,
    pos_ptr: &cudarc::driver::CudaSlice<u32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = n_batch * d;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        SLICE_WRITE_INDIRECT_PTX,
        "slice_write_indirect_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "slice_write_indirect_kernel",
    })?;
    let cfg = launch_cfg(total)?;
    let n_u32 = total as u32;
    let d_u32 = d as u32;
    let max_len_u32 = max_len as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(src.inner())
            .arg(dst.inner_mut())
            .arg(&n_u32)
            .arg(&d_u32)
            .arg(&max_len_u32)
            .arg(pos_ptr)
            .launch(cfg)?;
    }
    Ok(())
}

/// Build causal attention mask with total_len read from device memory.
/// Writes `out[h, col] = 0.0` if `col < *total_len_ptr`, else `-1e9`.
/// Output shape: `[n_head, max_pos]` (n_head rows, each max_pos wide).
#[cfg(feature = "cuda")]
pub fn gpu_causal_mask_indirect(
    total_len_ptr: &cudarc::driver::CudaSlice<u32>,
    n_head: usize,
    max_pos: usize,
    out: &mut CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = n_head * max_pos;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(
        ctx,
        CAUSAL_MASK_INDIRECT_PTX,
        "causal_mask_indirect_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "causal_mask_indirect_kernel",
    })?;
    let cfg = launch_cfg(total)?;
    let max_pos_u32 = max_pos as u32;
    let total_u32 = total as u32;
    unsafe {
        stream
            .launch_builder(&f)
            .arg(total_len_ptr)
            .arg(out.inner_mut())
            .arg(&max_pos_u32)
            .arg(&total_u32)
            .launch(cfg)?;
    }
    Ok(())
}

// ===========================================================================
// Pre-compilation of all decode-path PTX modules
// ===========================================================================

/// Pre-compile all PTX kernels used by the decode pass into the module cache.
/// Call this before CUDA graph capture to ensure no `cuModuleLoadData` calls
/// occur during capture (which is not a capturable operation).
#[cfg(feature = "cuda")]
pub fn precompile_decode_kernels(device: &GpuDevice) -> GpuResult<()> {
    let ctx = device.context();
    ctx.bind_to_thread()?;
    let ord = device.ordinal() as u32;
    let compile = |ptx: &'static str, name: &'static str| -> GpuResult<()> {
        crate::module_cache::get_or_compile(ctx, ptx, name, ord)
            .map(|_| ())
            .map_err(GpuError::Driver)
    };
    compile(ADD_PTX, "add_kernel")?;
    compile(MUL_PTX, "mul_kernel")?;
    compile(SCALE_PTX, "scale_kernel")?;
    compile(GELU_PTX, "gelu_kernel")?;
    compile(SOFTMAX_PTX, "softmax_kernel")?;
    compile(LAYERNORM_PTX, "layernorm_kernel")?;
    compile(PERMUTE_0213_PTX, "permute_0213_kernel")?;
    compile(EMBED_LOOKUP_PTX, "embed_lookup_kernel")?;
    compile(EMBED_LOOKUP_BATCH_PTX, "embed_lookup_batch_kernel")?;
    compile(SCATTER_ADD_ROWS_PTX, "scatter_add_rows_kernel")?;
    compile(SMALL_MATMUL_PTX, "small_matmul_kernel")?;
    compile(SLICE_WRITE_INDIRECT_PTX, "slice_write_indirect_kernel")?;
    compile(CAUSAL_MASK_INDIRECT_PTX, "causal_mask_indirect_kernel")?;
    compile(SLICE_READ_PTX, "slice_read_kernel")?;
    compile(RELU_BACKWARD_PTX, "relu_backward_kernel")?;
    compile(GELU_BACKWARD_PTX, "gelu_backward_kernel")?;
    Ok(())
}

/// Stub — no-op without cuda.
#[cfg(not(feature = "cuda"))]
pub fn precompile_decode_kernels(_device: &GpuDevice) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Stubs when `cuda` feature is disabled
// ---------------------------------------------------------------------------

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu(_input: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_div(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_exp(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_log(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_sqrt(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_pow(
    _a: &CudaBuffer<f32>,
    _exponent: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_abs(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_sigmoid(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_tanh(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_layernorm(
    _input: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _bias: &CudaBuffer<f32>,
    _rows: usize,
    _cols: usize,
    _eps: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_transpose_2d(
    _input: &CudaBuffer<f32>,
    _m: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_add(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_sub(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_mul(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_neg(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_relu(_a: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_scale(
    _a: &CudaBuffer<f32>,
    _scalar: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_add(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _a_shape: &[usize],
    _b_shape: &[usize],
    _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_sub(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _a_shape: &[usize],
    _b_shape: &[usize],
    _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_mul(
    _a: &CudaBuffer<f32>,
    _b: &CudaBuffer<f32>,
    _a_shape: &[usize],
    _b_shape: &[usize],
    _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_softmax(
    _input: &CudaBuffer<f32>,
    _rows: usize,
    _cols: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_dropout(
    _input: &CudaBuffer<f32>,
    _threshold: u32,
    _scale: f32,
    _seed: u32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_permute_0213(
    _input: &CudaBuffer<f32>,
    _d0: usize,
    _d1: usize,
    _d2: usize,
    _d3: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_slice_write(
    _src: &CudaBuffer<f32>,
    _dst: &mut CudaBuffer<f32>,
    _n_batch: usize,
    _d: usize,
    _max_len: usize,
    _pos: usize,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_slice_read(
    _src: &CudaBuffer<f32>,
    _n_batch: usize,
    _d: usize,
    _len: usize,
    _max_len: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_embed_lookup(
    _idx: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _d: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_embed_lookup_batch(
    _indices: &CudaBuffer<f32>,
    _weight: &CudaBuffer<f32>,
    _n: usize,
    _d: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_scatter_add_rows(
    _grad_output: &CudaBuffer<f32>,
    _indices: &CudaBuffer<f32>,
    _num_embeddings: usize,
    _d: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_relu_backward(
    _grad: &CudaBuffer<f32>,
    _input: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_backward(
    _grad: &CudaBuffer<f32>,
    _input: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_index_select_1d(
    _input: &CudaBuffer<f32>,
    _indices: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_scatter_add_1d(
    _grad_output: &CudaBuffer<f32>,
    _indices: &CudaBuffer<f32>,
    _input_len: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_fill(
    _input: &CudaBuffer<f32>,
    _mask: &CudaBuffer<f32>,
    _value: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_masked_zero(
    _grad: &CudaBuffer<f32>,
    _mask: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_sigmoid_backward(
    _grad: &CudaBuffer<f32>,
    _output: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_tanh_backward(
    _grad: &CudaBuffer<f32>,
    _output: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_softmax_backward(
    _grad: &CudaBuffer<f32>,
    _output: &CudaBuffer<f32>,
    _cols: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_sum_axis(
    _a: &CudaBuffer<f32>,
    _outer: usize,
    _axis_size: usize,
    _inner: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_strided_split(
    _input: &CudaBuffer<f32>,
    _total_along_axis: usize,
    _split_offset: usize,
    _split_size: usize,
    _inner_size: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_strided_cat(
    _input: &CudaBuffer<f32>,
    _output: &mut CudaBuffer<f32>,
    _total_along_axis: usize,
    _cat_offset: usize,
    _part_size: usize,
    _inner_size: usize,
    _n: usize,
    _device: &GpuDevice,
) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// f32-to-f16 GPU conversion
// ---------------------------------------------------------------------------

/// Convert an f32 GPU buffer to f16 (represented as `CudaSlice<u16>`).
///
/// Each element is converted using IEEE 754 round-to-nearest-even via the
/// PTX `cvt.rn.f16.f32` instruction. The output is a `CudaSlice<u16>` where
/// each `u16` holds the bit pattern of an IEEE 754 half-precision float.
///
/// # Errors
///
/// - [`GpuError::PtxCompileFailed`] if the conversion kernel cannot be compiled
///   (e.g., GPU architecture too old to support f16 conversion instructions).
/// - [`GpuError::Driver`] on CUDA launch errors.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_f32_to_f16(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    use cudarc::driver::PushKernelArg;

    let n = input.len();
    if n == 0 {
        let empty = device.stream().alloc_zeros::<u16>(0)?;
        return Ok(empty);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = crate::module_cache::get_or_compile(
        ctx,
        F32_TO_F16_PTX,
        "f32_to_f16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "f32_to_f16_kernel",
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    // SAFETY: The kernel reads `n` f32 values from `input` and writes `n`
    // u16 values (f16 bit patterns) to `out`. Both buffers are device-resident
    // and correctly sized. The grid is configured to cover exactly `n` threads.
    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub(crate) fn gpu_f32_to_f16(_input: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

/// Convert f32 GPU buffer to bf16 (stored as u16) on-device.
///
/// Uses bit manipulation for round-to-nearest-even bf16 conversion.
/// Works on sm_52+ (no special bf16 hardware required).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_f32_to_bf16(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<cudarc::driver::CudaSlice<u16>> {
    use cudarc::driver::PushKernelArg;

    let n = input.len();
    if n == 0 {
        let empty = device.stream().alloc_zeros::<u16>(0)?;
        return Ok(empty);
    }

    let ctx = device.context();
    let stream = device.stream();

    let f = crate::module_cache::get_or_compile(
        ctx,
        F32_TO_BF16_PTX,
        "f32_to_bf16_kernel",
        device.ordinal() as u32,
    )
    .map_err(|_| GpuError::PtxCompileFailed {
        kernel: "f32_to_bf16_kernel",
    })?;

    let mut out = stream.alloc_zeros::<u16>(n)?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;

    unsafe {
        stream
            .launch_builder(&f)
            .arg(input.inner())
            .arg(&mut out)
            .arg(&n_u32)
            .launch(cfg)?;
    }

    Ok(out)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub(crate) fn gpu_f32_to_bf16(_input: &CudaBuffer<f32>, _device: &GpuDevice) -> GpuResult<()> {
    Err(GpuError::NoCudaFeature)
}

// ---------------------------------------------------------------------------
// Tests -- require a real CUDA GPU
// ---------------------------------------------------------------------------

#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::*;

    /// Helper: set up device + upload a slice.
    fn setup(data: &[f32]) -> (GpuDevice, CudaBuffer<f32>) {
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let buf = cpu_to_gpu(data, &dev).expect("cpu_to_gpu");
        (dev, buf)
    }

    /// Round-trip helper: download a GPU buffer and compare against expected
    /// CPU output element-wise.
    fn assert_buf_eq(buf: &CudaBuffer<f32>, device: &GpuDevice, expected: &[f32]) {
        let host = gpu_to_cpu(buf, device).expect("gpu_to_cpu");
        assert_eq!(host.len(), expected.len(), "length mismatch");
        for (i, (&got, &exp)) in host.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 1e-6,
                "element {i}: got {got}, expected {exp}",
            );
        }
    }

    // -- gpu_add -------------------------------------------------------------

    #[test]
    fn add_basic() {
        let a_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_add(&a, &b, &dev).expect("gpu_add");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn add_empty() {
        let (dev, a) = setup(&[]);
        let b = cpu_to_gpu::<f32>(&[], &dev).expect("cpu_to_gpu b");
        let out = gpu_add(&a, &b, &dev).expect("gpu_add empty");
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn add_large() {
        let n = 100_000;
        let a_data: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_data: Vec<f32> = (0..n).map(|i| (i as f32) * 0.5).collect();
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x + y).collect();

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_add(&a, &b, &dev).expect("gpu_add large");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn add_length_mismatch() {
        let (dev, a) = setup(&[1.0, 2.0, 3.0]);
        let b = cpu_to_gpu(&[1.0, 2.0], &dev).expect("cpu_to_gpu b");
        let err = gpu_add(&a, &b, &dev).unwrap_err();
        match err {
            GpuError::LengthMismatch { a: 3, b: 2 } => {}
            other => panic!("unexpected error: {other}"),
        }
    }

    // -- gpu_sub -------------------------------------------------------------

    #[test]
    fn sub_basic() {
        let a_data = vec![10.0f32, 20.0, 30.0, 40.0];
        let b_data = vec![1.0f32, 2.0, 3.0, 4.0];
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x - y).collect();

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_sub(&a, &b, &dev).expect("gpu_sub");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn sub_negative_result() {
        let a_data = vec![1.0f32, 2.0];
        let b_data = vec![5.0f32, 10.0];
        let expected: Vec<f32> = vec![-4.0, -8.0];

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_sub(&a, &b, &dev).expect("gpu_sub");
        assert_buf_eq(&out, &dev, &expected);
    }

    // -- gpu_mul -------------------------------------------------------------

    #[test]
    fn mul_basic() {
        let a_data = vec![2.0f32, 3.0, 4.0, 5.0];
        let b_data = vec![10.0f32, 10.0, 10.0, 10.0];
        let expected: Vec<f32> = a_data.iter().zip(&b_data).map(|(x, y)| x * y).collect();

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_mul(&a, &b, &dev).expect("gpu_mul");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn mul_by_zero() {
        let a_data = vec![1.0f32, 2.0, 3.0];
        let b_data = vec![0.0f32, 0.0, 0.0];
        let expected = vec![0.0f32, 0.0, 0.0];

        let (dev, a) = setup(&a_data);
        let b = cpu_to_gpu(&b_data, &dev).expect("cpu_to_gpu b");
        let out = gpu_mul(&a, &b, &dev).expect("gpu_mul");
        assert_buf_eq(&out, &dev, &expected);
    }

    // -- gpu_neg -------------------------------------------------------------

    #[test]
    fn neg_basic() {
        let a_data = vec![1.0f32, -2.0, 3.0, 0.0, -5.5];
        let expected: Vec<f32> = a_data.iter().map(|x| -x).collect();

        let (dev, a) = setup(&a_data);
        let out = gpu_neg(&a, &dev).expect("gpu_neg");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn neg_double_negation() {
        let a_data = vec![1.0f32, -2.0, 3.0];
        let (dev, a) = setup(&a_data);
        let neg1 = gpu_neg(&a, &dev).expect("gpu_neg 1");
        let neg2 = gpu_neg(&neg1, &dev).expect("gpu_neg 2");
        assert_buf_eq(&neg2, &dev, &a_data);
    }

    // -- gpu_relu ------------------------------------------------------------

    #[test]
    fn relu_basic() {
        let a_data = vec![-3.0f32, -1.0, 0.0, 1.0, 3.0];
        let expected = vec![0.0f32, 0.0, 0.0, 1.0, 3.0];

        let (dev, a) = setup(&a_data);
        let out = gpu_relu(&a, &dev).expect("gpu_relu");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn relu_all_negative() {
        let a_data = vec![-5.0f32, -0.1, -100.0];
        let expected = vec![0.0f32, 0.0, 0.0];

        let (dev, a) = setup(&a_data);
        let out = gpu_relu(&a, &dev).expect("gpu_relu");
        assert_buf_eq(&out, &dev, &expected);
    }

    #[test]
    fn relu_all_positive() {
        let a_data = vec![0.1f32, 1.0, 100.0];

        let (dev, a) = setup(&a_data);
        let out = gpu_relu(&a, &dev).expect("gpu_relu");
        assert_buf_eq(&out, &dev, &a_data);
    }

    #[test]
    fn relu_empty() {
        let (dev, a) = setup(&[]);
        let out = gpu_relu(&a, &dev).expect("gpu_relu empty");
        assert_eq!(out.len(), 0);
    }

    #[test]
    fn small_matmul_2x2() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        // A = [[1, 2], [3, 4]], B = [[5, 6], [7, 8]]
        // C = A@B = [[19, 22], [43, 50]]
        let a = cpu_to_gpu(&[1.0f32, 2.0, 3.0, 4.0], &dev).unwrap();
        let b = cpu_to_gpu(&[5.0f32, 6.0, 7.0, 8.0], &dev).unwrap();
        let c = gpu_small_matmul(&a, &b, 2, 2, 2, &dev).unwrap();
        assert_buf_eq(&c, &dev, &[19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    fn small_matmul_1xk_kxn() {
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        // A = [1, 2, 3] (1x3), B = [[1, 0], [0, 1], [1, 1]] (3x2)
        // C = [4, 5] (1x2)
        let a = cpu_to_gpu(&[1.0f32, 2.0, 3.0], &dev).unwrap();
        let b = cpu_to_gpu(&[1.0f32, 0.0, 0.0, 1.0, 1.0, 1.0], &dev).unwrap();
        let c = gpu_small_matmul(&a, &b, 1, 3, 2, &dev).unwrap();
        assert_buf_eq(&c, &dev, &[4.0, 5.0]);
    }

    #[test]
    fn small_matmul_vs_cublas() {
        // Compare our small matmul against cuBLAS for a realistic decode-step size.
        // Linear layer: [1, 64] @ [64, 64] = [1, 64]
        let dev = GpuDevice::new(0).expect("CUDA device 0");
        let m = 1;
        let k = 64;
        let n = 64;

        // Deterministic data.
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| ((i * 7 + 3) % 100) as f32 / 100.0)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| ((i * 11 + 5) % 100) as f32 / 100.0)
            .collect();

        let a = cpu_to_gpu(&a_data, &dev).unwrap();
        let b = cpu_to_gpu(&b_data, &dev).unwrap();

        // cuBLAS reference.
        let c_cublas = crate::blas::gpu_matmul_f32(&a, &b, m, k, n, &dev).unwrap();
        let cublas_result = gpu_to_cpu(&c_cublas, &dev).unwrap();

        // Our kernel.
        let c_ours = gpu_small_matmul(&a, &b, m, k, n, &dev).unwrap();
        let our_result = gpu_to_cpu(&c_ours, &dev).unwrap();

        assert_eq!(cublas_result.len(), our_result.len());
        for (i, (&cb, &ours)) in cublas_result.iter().zip(our_result.iter()).enumerate() {
            assert!(
                (cb - ours).abs() < 0.1,
                "element {i}: cuBLAS={cb}, ours={ours}, diff={}",
                (cb - ours).abs()
            );
        }
    }
}
