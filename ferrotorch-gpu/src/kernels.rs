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

// ---------------------------------------------------------------------------
// LayerNorm PTX kernel (row-wise: mean, var, normalize+affine)
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
// Softmax PTX kernel (row-wise, numerically stable)
// ---------------------------------------------------------------------------
//
// One thread block per row. Each block:
//   1. Finds the max in shared memory (for numerical stability)
//   2. Computes exp(x - max) and sums in shared memory
//   3. Normalizes by the sum
//
// Parameters:
//   input_ptr  - pointer to input f32 buffer
//   output_ptr - pointer to output f32 buffer
//   rows       - number of rows (outer dimension)
//   cols       - number of columns (softmax dimension, = last_dim)

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
fn validate_binary(
    a: &CudaBuffer<f32>,
    b: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<()> {
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
    let f = match crate::module_cache::get_or_compile(ctx, ptx_src, kernel_name, device.ordinal() as u32) {
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
    let f = match crate::module_cache::get_or_compile(ctx, ptx_src, kernel_name, device.ordinal() as u32) {
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

    let f = match crate::module_cache::get_or_compile(ctx, ptx_src, kernel_name, device.ordinal() as u32) {
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

    let f = match crate::module_cache::get_or_compile(ctx, ptx_src, kernel_name, device.ordinal() as u32) {
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

    let f = match crate::module_cache::get_or_compile(ctx, ptx_src, kernel_name, device.ordinal() as u32) {
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

    if let Some(out) = try_launch_binary(a, b, device, ADD_PTX, "add_kernel")? {
        return Ok(out);
    }

    // CPU fallback (correct but slow).
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
        a, b, &a_str, &b_str, &shape_u32, out_numel,
        device, BROADCAST_ADD_PTX, "broadcast_add_kernel",
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
        a, b, &a_str, &b_str, &shape_u32, out_numel,
        device, BROADCAST_SUB_PTX, "broadcast_sub_kernel",
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
        a, b, &a_str, &b_str, &shape_u32, out_numel,
        device, BROADCAST_MUL_PTX, "broadcast_mul_kernel",
    )? {
        return Ok(out);
    }

    cpu_fallback_broadcast_binary(a, b, a_shape, b_shape, out_shape, device, |x, y| x * y)
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
pub fn gpu_neg(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
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
pub fn gpu_relu(
    a: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
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

    if let Some(out) = try_launch_binary(grad, input, device, RELU_BACKWARD_PTX, "relu_backward_kernel")? {
        return Ok(out);
    }

    // CPU fallback
    let grad_host = gpu_to_cpu(grad, device)?;
    let input_host = gpu_to_cpu(input, device)?;
    let result: Vec<f32> = grad_host.iter().zip(input_host.iter())
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

    if let Some(out) = try_launch_binary(grad, input, device, GELU_BACKWARD_PTX, "gelu_backward_kernel")? {
        return Ok(out);
    }

    // CPU fallback
    let grad_host = gpu_to_cpu(grad, device)?;
    let input_host = gpu_to_cpu(input, device)?;
    let result: Vec<f32> = grad_host.iter().zip(input_host.iter())
        .map(|(&g, &x)| {
            let k: f32 = 1.702;
            let sig = 1.0 / (1.0 + (-k * x).exp());
            g * (sig + k * x * sig * (1.0 - sig))
        })
        .collect();
    cpu_to_gpu(&result, device)
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

    let f = match crate::module_cache::get_or_compile(ctx, SCALE_PTX, "scale_kernel", device.ordinal() as u32) {
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
        ctx, SOFTMAX_PTX, "softmax_kernel", device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let host = gpu_to_cpu(input, device)?;
            let mut out = vec![0.0f32; host.len()];
            for r in 0..rows {
                let base = r * cols;
                let mut max_v = f32::NEG_INFINITY;
                for c in 0..cols { max_v = max_v.max(host[base + c]); }
                let mut sum = 0.0f32;
                for c in 0..cols {
                    let e = (host[base + c] - max_v).exp();
                    out[base + c] = e;
                    sum += e;
                }
                let inv = 1.0 / sum;
                for c in 0..cols { out[base + c] *= inv; }
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
/// `seed` = random seed for the xorshift RNG.
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
        ctx, DROPOUT_PTX, "dropout_kernel", device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(_) => {
            // CPU fallback.
            let host = gpu_to_cpu(input, device)?;
            let mut rng_state = seed;
            let result: Vec<f32> = host.iter().enumerate().map(|(i, &x)| {
                let mut r = (i as u32).wrapping_mul(2654435761) ^ rng_state;
                r ^= r << 13; r ^= r >> 17; r ^= r << 5;
                rng_state = r;
                if r < threshold { 0.0 } else { x * scale }
            }).collect();
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
        ctx, TRANSPOSE_2D_PTX, "transpose_2d_kernel", device.ordinal() as u32,
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
        ctx, PERMUTE_0213_PTX, "permute_0213_kernel", device.ordinal() as u32,
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
        ctx, SMALL_MATMUL_PTX, "small_matmul_kernel", device.ordinal() as u32,
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
        ctx, EMBED_LOOKUP_PTX, "embed_lookup_kernel", device.ordinal() as u32,
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
        ctx, SLICE_WRITE_PTX, "slice_write_kernel", device.ordinal() as u32,
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
        ctx, SLICE_READ_PTX, "slice_read_kernel", device.ordinal() as u32,
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
pub fn gpu_gelu(
    input: &CudaBuffer<f32>,
    device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
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
        ctx, LAYERNORM_PTX, "layernorm_kernel", device.ordinal() as u32,
    ) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("ferrotorch-gpu: LayerNorm PTX compilation failed ({e:?}), CPU fallback");
            std::fs::write("/tmp/layernorm_debug.ptx", LAYERNORM_PTX).ok();
            eprintln!("ferrotorch-gpu: dumped PTX to /tmp/layernorm_debug.ptx ({} bytes)", LAYERNORM_PTX.len());
            let h_in = gpu_to_cpu(input, device)?;
            let h_w = gpu_to_cpu(weight, device)?;
            let h_b = gpu_to_cpu(bias, device)?;
            let mut out = vec![0.0f32; rows * cols];
            for r in 0..rows {
                let base = r * cols;
                let slice = &h_in[base..base + cols];
                let mean: f32 = slice.iter().sum::<f32>() / cols as f32;
                let var: f32 = slice.iter().map(|&x| (x - mean) * (x - mean)).sum::<f32>() / cols as f32;
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
    a: &CudaBuffer<f32>, b: &CudaBuffer<f32>, out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    validate_binary(a, b, device)?;
    if try_launch_binary_into(a, b, out, device, ADD_PTX, "add_kernel")? {
        return Ok(());
    }
    Err(GpuError::PtxCompileFailed { kernel: "add_kernel" })
}

/// Elementwise mul into pre-allocated output: `out[i] = a[i] * b[i]`.
#[cfg(feature = "cuda")]
pub fn gpu_mul_into(
    a: &CudaBuffer<f32>, b: &CudaBuffer<f32>, out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    validate_binary(a, b, device)?;
    if try_launch_binary_into(a, b, out, device, MUL_PTX, "mul_kernel")? {
        return Ok(());
    }
    Err(GpuError::PtxCompileFailed { kernel: "mul_kernel" })
}

/// Scalar multiply into pre-allocated output: `out[i] = a[i] * scalar`.
#[cfg(feature = "cuda")]
pub fn gpu_scale_into(
    a: &CudaBuffer<f32>, scalar: f32, out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    validate_unary(a, device)?;
    let n = a.len();
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(ctx, SCALE_PTX, "scale_kernel", device.ordinal() as u32)
        .map_err(|_| GpuError::PtxCompileFailed { kernel: "scale_kernel" })?;
    let cfg = launch_cfg(n)?;
    let n_u32 = n as u32;
    unsafe {
        stream.launch_builder(&f)
            .arg(a.inner()).arg(out.inner_mut()).arg(&scalar).arg(&n_u32)
            .launch(cfg)?;
    }
    Ok(())
}

/// GELU into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_gelu_into(
    a: &CudaBuffer<f32>, out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    validate_unary(a, device)?;
    if try_launch_unary_into(a, out, device, GELU_PTX, "gelu_kernel")? {
        return Ok(());
    }
    Err(GpuError::PtxCompileFailed { kernel: "gelu_kernel" })
}

/// Embedding lookup into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_embed_lookup_into(
    idx: &CudaBuffer<f32>, weight: &CudaBuffer<f32>, d: usize,
    out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(ctx, EMBED_LOOKUP_PTX, "embed_lookup_kernel", device.ordinal() as u32)
        .map_err(|_| GpuError::PtxCompileFailed { kernel: "embed_lookup_kernel" })?;
    let cfg = launch_cfg(d)?;
    let d_u32 = d as u32;
    unsafe {
        stream.launch_builder(&f)
            .arg(idx.inner()).arg(weight.inner()).arg(out.inner_mut()).arg(&d_u32)
            .launch(cfg)?;
    }
    Ok(())
}

/// 2D transpose into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_transpose_2d_into(
    a: &CudaBuffer<f32>, m: usize, n: usize,
    out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = m * n;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(ctx, TRANSPOSE_2D_PTX, "transpose_2d_kernel", device.ordinal() as u32)
        .map_err(|_| GpuError::PtxCompileFailed { kernel: "transpose_2d_kernel" })?;
    let cfg = launch_cfg(total)?;
    let m_u32 = m as u32;
    let n_u32 = n as u32;
    let total_u32 = total as u32;
    unsafe {
        stream.launch_builder(&f)
            .arg(a.inner()).arg(out.inner_mut())
            .arg(&m_u32).arg(&n_u32).arg(&total_u32)
            .launch(cfg)?;
    }
    Ok(())
}

/// Permute (0,2,1,3) into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_permute_0213_into(
    a: &CudaBuffer<f32>, d0: usize, d1: usize, d2: usize, d3: usize,
    out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = d0 * d1 * d2 * d3;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(ctx, PERMUTE_0213_PTX, "permute_0213_kernel", device.ordinal() as u32)
        .map_err(|_| GpuError::PtxCompileFailed { kernel: "permute_0213_kernel" })?;
    let cfg = launch_cfg(total)?;
    let (d0u, d1u, d2u, d3u, tu) = (d0 as u32, d1 as u32, d2 as u32, d3 as u32, total as u32);
    unsafe {
        stream.launch_builder(&f)
            .arg(a.inner()).arg(out.inner_mut())
            .arg(&d0u).arg(&d1u).arg(&d2u).arg(&d3u).arg(&tu)
            .launch(cfg)?;
    }
    Ok(())
}

/// Softmax into pre-allocated output (row-wise).
#[cfg(feature = "cuda")]
pub fn gpu_softmax_into(
    a: &CudaBuffer<f32>, rows: usize, cols: usize,
    out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(ctx, SOFTMAX_PTX, "softmax_kernel", device.ordinal() as u32)
        .map_err(|_| GpuError::PtxCompileFailed { kernel: "softmax_kernel" })?;
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
        stream.launch_builder(&f)
            .arg(a.inner()).arg(out.inner_mut())
            .arg(&rows_u32).arg(&cols_u32)
            .launch(cfg)?;
    }
    Ok(())
}

/// LayerNorm into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_layernorm_into(
    input: &CudaBuffer<f32>, weight: &CudaBuffer<f32>, bias: &CudaBuffer<f32>,
    rows: usize, cols: usize, eps: f32,
    out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(ctx, LAYERNORM_PTX, "layernorm_kernel", device.ordinal() as u32)
        .map_err(|_| GpuError::PtxCompileFailed { kernel: "layernorm_kernel" })?;
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
        stream.launch_builder(&f)
            .arg(input.inner()).arg(out.inner_mut())
            .arg(weight.inner()).arg(bias.inner())
            .arg(&rows_u32).arg(&cols_u32).arg(&eps)
            .launch(cfg)?;
    }
    Ok(())
}

/// Slice read into pre-allocated output: read first `len` rows from
/// `[n_batch, max_len, d]` into out `[n_batch, len, d]`.
#[cfg(feature = "cuda")]
pub fn gpu_slice_read_into(
    src: &CudaBuffer<f32>, n_batch: usize, d: usize, len: usize, max_len: usize,
    out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = n_batch * len * d;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(ctx, SLICE_READ_PTX, "slice_read_kernel", device.ordinal() as u32)
        .map_err(|_| GpuError::PtxCompileFailed { kernel: "slice_read_kernel" })?;
    let cfg = launch_cfg(total)?;
    let total_u32 = total as u32;
    let d_u32 = d as u32;
    let len_u32 = len as u32;
    let max_len_u32 = max_len as u32;
    unsafe {
        stream.launch_builder(&f)
            .arg(src.inner()).arg(out.inner_mut())
            .arg(&total_u32).arg(&d_u32).arg(&len_u32).arg(&max_len_u32)
            .launch(cfg)?;
    }
    Ok(())
}

/// Small matmul (PTX kernel) into pre-allocated output.
#[cfg(feature = "cuda")]
pub fn gpu_small_matmul_into(
    a: &CudaBuffer<f32>, b: &CudaBuffer<f32>,
    m: usize, k: usize, n: usize,
    out: &mut CudaBuffer<f32>, device: &GpuDevice,
) -> GpuResult<()> {
    use cudarc::driver::PushKernelArg;
    let total = m * n;
    let ctx = device.context();
    let stream = device.stream();
    let f = crate::module_cache::get_or_compile(ctx, SMALL_MATMUL_PTX, "small_matmul_kernel", device.ordinal() as u32)
        .map_err(|_| GpuError::PtxCompileFailed { kernel: "small_matmul_kernel" })?;
    let cfg = launch_cfg(total)?;
    let (m_u32, k_u32, n_u32, total_u32) = (m as u32, k as u32, n as u32, total as u32);
    unsafe {
        stream.launch_builder(&f)
            .arg(a.inner()).arg(b.inner()).arg(out.inner_mut())
            .arg(&m_u32).arg(&k_u32).arg(&n_u32).arg(&total_u32)
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
        ctx, SLICE_WRITE_INDIRECT_PTX, "slice_write_indirect_kernel", device.ordinal() as u32,
    ).map_err(|_| GpuError::PtxCompileFailed { kernel: "slice_write_indirect_kernel" })?;
    let cfg = launch_cfg(total)?;
    let n_u32 = total as u32;
    let d_u32 = d as u32;
    let max_len_u32 = max_len as u32;
    unsafe {
        stream.launch_builder(&f)
            .arg(src.inner()).arg(dst.inner_mut())
            .arg(&n_u32).arg(&d_u32).arg(&max_len_u32)
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
        ctx, CAUSAL_MASK_INDIRECT_PTX, "causal_mask_indirect_kernel", device.ordinal() as u32,
    ).map_err(|_| GpuError::PtxCompileFailed { kernel: "causal_mask_indirect_kernel" })?;
    let cfg = launch_cfg(total)?;
    let max_pos_u32 = max_pos as u32;
    let total_u32 = total as u32;
    unsafe {
        stream.launch_builder(&f)
            .arg(total_len_ptr).arg(out.inner_mut())
            .arg(&max_pos_u32).arg(&total_u32)
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
pub fn gpu_gelu(
    _input: &CudaBuffer<f32>, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_layernorm(
    _input: &CudaBuffer<f32>, _weight: &CudaBuffer<f32>, _bias: &CudaBuffer<f32>,
    _rows: usize, _cols: usize, _eps: f32, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_transpose_2d(
    _input: &CudaBuffer<f32>, _m: usize, _n: usize, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

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
pub fn gpu_neg(
    _a: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_relu(
    _a: &CudaBuffer<f32>,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> {
    Err(GpuError::NoCudaFeature)
}

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_scale(
    _a: &CudaBuffer<f32>,
    _scalar: f32,
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_add(
    _a: &CudaBuffer<f32>, _b: &CudaBuffer<f32>,
    _a_shape: &[usize], _b_shape: &[usize], _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_sub(
    _a: &CudaBuffer<f32>, _b: &CudaBuffer<f32>,
    _a_shape: &[usize], _b_shape: &[usize], _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_broadcast_mul(
    _a: &CudaBuffer<f32>, _b: &CudaBuffer<f32>,
    _a_shape: &[usize], _b_shape: &[usize], _out_shape: &[usize],
    _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_softmax(
    _input: &CudaBuffer<f32>, _rows: usize, _cols: usize, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_dropout(
    _input: &CudaBuffer<f32>, _threshold: u32, _scale: f32, _seed: u32, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_permute_0213(
    _input: &CudaBuffer<f32>, _d0: usize, _d1: usize, _d2: usize, _d3: usize, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_slice_write(
    _src: &CudaBuffer<f32>, _dst: &mut CudaBuffer<f32>,
    _n_batch: usize, _d: usize, _max_len: usize, _pos: usize, _device: &GpuDevice,
) -> GpuResult<()> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_slice_read(
    _src: &CudaBuffer<f32>, _n_batch: usize, _d: usize, _len: usize, _max_len: usize, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_embed_lookup(
    _idx: &CudaBuffer<f32>, _weight: &CudaBuffer<f32>, _d: usize, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_relu_backward(
    _grad: &CudaBuffer<f32>, _input: &CudaBuffer<f32>, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

/// Stub -- always returns [`GpuError::NoCudaFeature`].
#[cfg(not(feature = "cuda"))]
pub fn gpu_gelu_backward(
    _grad: &CudaBuffer<f32>, _input: &CudaBuffer<f32>, _device: &GpuDevice,
) -> GpuResult<CudaBuffer<f32>> { Err(GpuError::NoCudaFeature) }

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
        let a_data: Vec<f32> = (0..m*k).map(|i| ((i * 7 + 3) % 100) as f32 / 100.0).collect();
        let b_data: Vec<f32> = (0..k*n).map(|i| ((i * 11 + 5) % 100) as f32 / 100.0).collect();

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
